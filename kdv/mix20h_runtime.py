"""Opt-in MIX20H admission, provenance, pair checks and safe wall-clock stop.

This module does not change the QRECON objectives, scheduler, or selector.  A
runner-created admission and a persistent 20-hour plan are required even when
main.py is invoked directly.  Exit 5 means budget stop, never train completion.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "QRC24_MIX20H_20260918_v1"
REVISION = "QRC24_MIX20H_Q1_20260918"
SELECTOR = "HQNR9585_ERGAS2040_v2"
TEACHER_SHA = "16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32"
EXIT_STOPPED_BUDGET = 5
META_PREFIX_BATCHES = 16
RESUME_FILES = ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl",
                "custom_checkpoint_0.pkl", "custom_checkpoint_1.pkl", "custom_checkpoint_2.pkl")
EVALUATION_COMMIT = "mix20h_evaluation_complete.json"


def _json(path):
    with open(path) as handle:
        return json.load(handle)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _path(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def utc(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat()


def timestamp(value):
    value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("MIX20H timestamps must have explicit UTC offset")
    return value.timestamp()


def _equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"MIX20H {label}: {actual!r} != {expected!r}")


def checkpoint_updates(path):
    """M20's validated feeder has exactly 202 optimizer updates per epoch."""
    name = Path(path).name
    match = re.fullmatch(r"(epoch|step|checkpoint|checkpoint-budget)-(\d+)", name)
    if match is None:
        raise ValueError(f"MIX20H invalid resume checkpoint name: {name}")
    step = int(match[2]) * (202 if match[1] == "epoch" else 1)
    if not 0 <= step <= 50000:
        raise ValueError(f"MIX20H resume update outside 0..50000: {step}")
    return step


def require_resume_files(path):
    path = Path(path)
    if any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in RESUME_FILES):
        raise ValueError("MIX20H resume lacks complete model/optimizer/scheduler/RNG/epoch state")


def recorded_grid_steps(wd):
    path = Path(wd) / "checkpoint_metrics.csv"
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        steps = [int(row["step"]) for row in rows]
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("MIX20H cannot verify truncated/malformed checkpoint metrics; no automatic rollback") from error
    if len(steps) != len(set(steps)):
        raise ValueError("MIX20H duplicate grid steps require explicit repair; outputs preserved")
    return steps


def validate_candidate_commit(path, run_id, config_sha256=None):
    """A candidate is resumable only AFTER the full main._evaluate transaction."""
    path = Path(path); step = checkpoint_updates(path)
    require_resume_files(path)
    marker = _json(path / EVALUATION_COMMIT)
    binding = _json(path / "mix20h_identity.json")
    expected = dict(run_id=run_id, step=step, checkpoint_sha256=sha256(path / "model.safetensors"))
    for field, value in expected.items():
        _equal(marker.get(field), value, "evaluation commit." + field)
        _equal(binding.get(field), value, "candidate binding." + field)
    _equal(marker.get("post_evaluation"), True, "post-evaluation resume state")
    _equal(binding.get("eval_mode"), "A_ON", "candidate resume view")
    if config_sha256 is not None:
        _equal(binding.get("config_sha256"), config_sha256, "candidate resume config")
    for name in RESUME_FILES:
        _equal((marker.get("state_file_sha256") or {}).get(name), sha256(path / name), "committed state." + name)
    return marker


def validate_definition(args):
    """Validate effective CLI+YAML values, not only the admitted YAML bytes."""
    a = vars(args) if not isinstance(args, dict) else args
    k = a.get("kdv") or {}; m = k.get("mix20h") or {}
    if not m:
        if k.get("campaign_id") == CAMPAIGN or re.fullmatch(
                r"PAKD50_QRC24_S[1-5]_(G23|B20A03)_W104_D121_WV3_T0_S52\d{3}_FRESH50_v4",
                Path(a.get("work_dir", "")).name):
            raise ValueError("MIX20H namespace requires explicit kdv.mix20h admission metadata")
        return None
    from kdv.mix20h_plan import validate_metadata
    validate_metadata(m, Path(a["work_dir"]).name)
    rid = Path(a["work_dir"]).name
    match = re.fullmatch(r"PAKD50_QRC24_(S[1-5])_(G23|B20A03)_W104_D121_WV3_T0_S(52\d{3})_FRESH50_v4", rid)
    if match is None:
        raise ValueError(f"MIX20H invalid exact run identity: {rid}")
    server, profile, seed = match[1].lower(), match[2], int(match[3])
    identity = dict(campaign_id=CAMPAIGN, queue_revision=REVISION, run_id=rid,
                    server_id=server, pair_id=f"M20_{server.upper()}_S{seed}", profile=profile, seed=seed, version="v4")
    for field, value in identity.items():
        if field == "run_id":
            _equal(m.get("original_run_id", m.get("run_id", rid)), value, field)
        else:
            _equal(m.get(field), value, field)
    _equal(m.get("selector"), SELECTOR, "selector")
    _equal(m.get("method"), "qrecon_continuous_v1", "method")
    _equal(m.get("eval_mode"), "A_ON", "eval_mode")
    for field, value in dict(seed=seed, num_iter=50000, batch_size=48, test_batch_size=1,
                             learning_rate=1e-4, weight_decay=.01, lr_scheduler="cosine", num_warmup=100,
                             mars="ms", res=True, num_bands=8, max_pixel=2047., trainer="kdv",
                             num_worker=4, eval_epoch=5, select_on="hqnr").items():
        _equal(a.get(field), value, field)
    for field, value in dict(hidden_size=104, depth=[1, 2, 1], in_mode="paper", attn_locations=[],
                             norm="ln", mode_modulation=False, dropout=0.).items():
        _equal(a.get("model_args", {}).get(field), value, "model_args." + field)
    for field, value in dict(crop=False, hflip=True, vflip=True, rot=True, return_meta=True).items():
        _equal(a.get("train_feeder_args", {}).get(field), value, "train_feeder_args." + field)
    for field, value in dict(exact_resume=True, campaign_id=CAMPAIGN, version="v4", input_protocol="I-NATIVE-TRANSFER",
                             aligner_policy="A-FT", aligner_lr=3e-6, candidate_grid_id="GRID1010_50K_v1").items():
        _equal(k.get(field), value, "kdv." + field)
    for field in ("budget", "phase", "aligner_schedule", "routing", "edge_route", "edge_schedule", "edge_weight", "edge_gate"):
        if k.get(field):
            raise ValueError(f"MIX20H forbids inherited kdv.{field}")
    for field, value in dict(case="R3", alpha=1., kd_weight=(.1 if profile == "G23" else .2), eps=1e-6,
                             tau=.012463942170143127).items():
        _equal((k.get("rec") or {}).get(field), value, "kdv.rec." + field)
    for field, value in dict(mode="continuous_v1", q_ref=.3276133416220546, a_weight="q", e_weight="q").items():
        _equal((k.get("qrecon") or {}).get(field), value, "kdv.qrecon." + field)
    for field, value in dict(enabled=True, kind="EDGE", mode="H", outer_weight=.002, ramp_updates=0).items():
        _equal((k.get("stat") or {}).get(field), value, "kdv.stat." + field)
    _equal((k.get("corruption") or {}).get("radius_hr"), 0., "Student jitter")
    _equal((k.get("aux") or {}).get("offset_weight"), 0., "Student offset")
    _equal((k.get("select") or {}).get("retain_all_candidates"), True, "candidate retention")
    _equal((k.get("donor") or {}).get("view_margin_hr"), 4, "aligner margin")
    for field in ("teacher", "donor"):
        _equal((k.get(field) or {}).get("expected_sha256"), TEACHER_SHA, field + " checkpoint")
    return identity


def validate_launch(args, root=ROOT, now=None):
    """Cheap fail-closed admission guard, called before loading data or models."""
    a = vars(args) if not isinstance(args, dict) else args
    identity = validate_definition(a)
    if identity is None:
        return None
    m = a["kdv"]["mix20h"]; now = time.time() if now is None else now
    wd = _path(a["work_dir"], root)
    plan_path = _path(m.get("plan_manifest", "work_dir/_qrc24_mix20h/plan_manifest.json"), root)
    admission_path = _path(m.get("admission_manifest", str(wd / "meta/mix20h_admission.json")), root)
    plan, admission = _json(plan_path), _json(admission_path)
    _equal(plan.get("activation_state"), "ready", "atomic migration activation")
    for field in ("campaign_id", "queue_revision", "server_id"):
        _equal(plan.get(field), identity[field], "plan." + field)
    for field in ("campaign_id", "queue_revision", "run_id", "server_id", "pair_id"):
        _equal(admission.get(field), identity[field], "admission." + field)
    _equal(admission.get("status"), "admitted", "admission.status")
    start = timestamp(plan.get("campaign_started_at_utc", plan.get("started_at_utc")))
    deadline = timestamp(plan["deadline_at_utc"])
    _equal(deadline - start, 20 * 3600., "persistent 20h window")
    _equal(timestamp(admission["deadline_at_utc"]), deadline, "admission.deadline")
    if not start <= now < deadline:
        raise ValueError("MIX20H plan window is not open; no start/resume after deadline")
    config_path = _path(a["config"], root); config_sha = sha256(config_path)
    _equal(admission.get("config_sha256"), config_sha, "admitted config bytes")
    _equal((plan.get("config_hashes") or {}).get(identity["run_id"]), config_sha, "prepared immutable config bytes")
    import yaml
    with open(config_path) as handle:
        admitted_config = yaml.safe_load(handle)
    for key, value in admitted_config.items():
        if key not in ("gpu", "resume", "config"):
            _equal(a.get(key), value, "effective/admitted config." + key)
    existing_manifest = wd / "meta/mix20h_run_manifest.json"
    if existing_manifest.exists() and not a.get("resume"):
        raise ValueError("MIX20H run already initialized; never silently restart it fresh (explicit exact resume required)")
    if a.get("resume"):
        resume_path = _path(a["resume"], root).resolve()
        candidate = resume_path.parent == (wd / "candidates").resolve() and resume_path.name.startswith("step-")
        if (resume_path.parent != wd.resolve() and not candidate) or not resume_path.is_dir():
            raise ValueError("MIX20H resume must be an existing checkpoint of this exact run")
        require_resume_files(resume_path)
        step = checkpoint_updates(resume_path)
        if candidate:
            validate_candidate_commit(resume_path, identity["run_id"], config_sha)
        grid = recorded_grid_steps(wd)
        if any(recorded > step for recorded in grid):
            raise ValueError("MIX20H resume would roll back behind recorded grid/selector state; outputs preserved")
        if step in grid:
            validate_candidate_commit(wd / "candidates" / f"step-{step}", identity["run_id"], config_sha)
            if not candidate and step < 50000:
                raise ValueError("MIX20H evaluated-step resume requires the committed candidate's post-evaluation RNG state")
        if resume_path.name.startswith("checkpoint-budget-"):
            marker = _json(resume_path / "mix20h_resume.json")
            _equal(marker.get("step"), int(resume_path.name.rsplit("-", 1)[1]), "resume completed update")
            _equal(marker.get("exact_resume"), True, "budget resume commit marker")
            for field, value in identity.items():
                _equal(marker.get(field), value, "budget resume." + field)
        previous = _json(wd / "meta/mix20h_run_manifest.json")
        for field in ("campaign_id", "queue_revision", "run_id", "server_id", "pair_id"):
            _equal(previous.get(field), identity[field], "resume." + field)
        _equal(timestamp(previous["deadline_at_utc"]), deadline, "resume deadline cannot reset")
    return dict(identity=identity, plan=plan, admission=admission, plan_path=str(plan_path),
                config_sha256=config_sha, work_dir=str(wd), deadline=deadline, started_at=now)


def tensor_state_sha(module):
    """Full tensor identity (names, dtype, shapes, bytes), not serialized zip metadata."""
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def runtime_contract(runtime):
    """Observed fields whose effective defaults must match the reference run."""
    return {key: runtime.get(key) for key in ("optimizer_class", "gradient_accumulation_steps", "grad_clip", "mixed_precision", "scaler")} | {
        "param_groups": [{key: group.get(key) for key in ("name", "n_params", "n_elements", "initial_lr", "weight_decay", "betas", "eps", "amsgrad")}
                         for group in runtime.get("param_groups", [])],
        "scheduler_detail": {key: runtime.get("scheduler_detail", {}).get(key) for key in
                             ("kind", "num_warmup_steps", "num_training_steps", "min_lr")}}


@contextmanager
def _lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


class Mix20Runtime:
    def __init__(self, args, context):
        self.args = args; self.context = context; self.identity = context["identity"]
        self.wd = Path(context["work_dir"]); self.meta = self.wd / "meta"
        self.path = self.meta / "mix20h_run_manifest.json"
        self.manifest = None; self.prefix = {}

    def bind(self, trainer):
        """Record actual instantiated state; reject definition drift before update 1."""
        a, k, m = trainer.args, trainer.k, trainer.k["mix20h"]
        actual = trainer._optimizer_manifest(); expected = m.get("expected_runtime")
        if not expected:
            raise ValueError("MIX20H expected_runtime must come from a recorded reference run")
        _equal(runtime_contract(actual), runtime_contract(expected), "actual optimizer/scheduler/precision")
        _equal(trainer.accelerator.num_processes, 1, "one GPU/process per server")
        scheduler = getattr(trainer.lr_scheduler, "scheduler", trainer.lr_scheduler)
        _equal(type(scheduler).__name__, "LambdaLR", "actual scheduler class")
        if not all(float(fn(50000)) == 0.0 for fn in scheduler.lr_lambdas):
            raise ValueError("MIX20H cosine terminal learning-rate factor must be zero")
        optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
        actual["optimizer_defaults"] = dict(optimizer.defaults)
        actual["actual_scheduler_class"] = type(scheduler).__name__
        actual["terminal_lr_factors"] = [float(fn(50000)) for fn in scheduler.lr_lambdas]
        _equal(len(trainer.train_data_loader), 202, "202 updates/epoch for GRID1010")
        _equal(bool(trainer.M.sampler), True, "A_ON sampler")
        _equal(trainer.aligner_view_margin, 4, "actual aligner view margin")
        _equal(trainer.teacher_manifest["file_sha256"], TEACHER_SHA, "loaded Teacher")
        _equal(trainer.donor_manifest["file_sha256"], TEACHER_SHA, "loaded Student A donor")
        if tensor_state_sha(trainer.M.aligner) != tensor_state_sha(trainer.teacher.aligner):
            raise ValueError("MIX20H fresh Student A must be the exact frozen T0 A clone")
        if not trainer.aligner_trainable or not trainer.M.aligner.training or trainer.teacher.training:
            raise ValueError("MIX20H requires trainable Student A and frozen/eval Teacher")
        if any(parameter.requires_grad for parameter in trainer.teacher.parameters()):
            raise ValueError("MIX20H Teacher must be fully frozen")
        datasets = _json(self.wd / "dataset_hashes.json")
        expected_ds = m.get("expected_dataset_hashes") or {}
        for role in ("train_feeder_args", "val_feeder_args", "test_reduced_feeder_args", "test_full_feeder_args"):
            want = expected_ds.get(role)
            if isinstance(want, dict):
                want = want.get("sha256")
            if not want:
                raise ValueError(f"MIX20H missing reference dataset hash: {role}")
            _equal(datasets[role]["sha256"], want, "dataset." + role)
        cue_path = _path(k["qrecon"]["asset"], ROOT); cue = _json(cue_path)
        npz_path = cue_path.with_suffix(".npz")
        _equal(sha256(npz_path), cue["npz_sha256"], "cue NPZ content")
        _equal(sha256(npz_path), m.get("expected_cue_sha256"), "reference cue bytes")
        _equal(sha256(cue_path), m.get("expected_cue_manifest_sha256"), "reference cue manifest")
        _equal(cue["teacher"]["file_sha256"], TEACHER_SHA, "cue Teacher")
        source = m.get("runtime_reference_source") or {}
        if not source.get("path") or not source.get("sha256"):
            raise ValueError("MIX20H runtime reference needs path and verified SHA256")
        _equal(sha256(_path(source["path"], ROOT)), source["sha256"], "runtime reference bytes")
        reference = _json(_path(source["path"], ROOT))
        _equal(runtime_contract(expected), runtime_contract(reference["training"]), "reference runtime contract")
        versions = {}
        for package in ("torch", "accelerate", "diffusers", "numpy", "safetensors"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not_installed"
        training_sources = {path: sha256(ROOT / path) for path in
                            ("main.py", "train_kdv.py", "kdv/qrecon.py", "kdv/losses_rec.py", "kdv/forward.py", "kdv/protocol.py",
                             "kdv/registry.py", "kdv/calibration.py", "kdv/teacher_assets.py", "kdv/edge_gate.py",
                             "kdv/resume.py", "kdv/mix20h_runtime.py", "kdv/mix20h_plan.py", "pa/model.py", "pa/aligner.py",
                             "pa/offset.py", "pa/warp.py", "pa/losses.py", "model/pancrafter.py", "model/pancrafter_paper.py", "feeders/feeder.py")}
        implementation_sources = dict(training_sources)
        for path in ("tools/mix20h_runner.py", "tools/mix20h_postrun.py", "tools/gen_mix20h_configs.py",
                     "tools/mix20h_launch_config.py", "tools/mix20h_start.sh", "tools/mix20h_switch.sh",
                     "tools/qrecon24_select.py", "tools/qrecon24_postrun.py", "tools/gen_pakd50_configs.py",
                     "tools/run.sh", "gspread/mix20h_upload.py"):
            implementation_sources[path] = sha256(ROOT / path)
        definition = dict(model=a.model, model_args=a.model_args, feeder=a.feeder, training_source_sha256=training_sources,
                          training={key: getattr(a, key) for key in ("seed", "num_worker", "batch_size", "test_batch_size", "num_iter", "num_warmup", "learning_rate", "weight_decay", "lr_scheduler", "mars", "res", "max_pixel")},
                          augmentation={key: val for key, val in a.train_feeder_args.items() if key != "dataroot"},
                          rec=k["rec"], stat=k["stat"], qrecon=k["qrecon"], input_protocol=k["input_protocol"],
                          actual_runtime=runtime_contract(actual), numerical_runtime=dict(tf32=actual["tf32"], framework_versions=versions,
                                                                                       optimizer_defaults=actual["optimizer_defaults"]), teacher_sha256=TEACHER_SHA,
                          dataset_hashes={key: val["sha256"] for key, val in datasets.items()}, cue_sha256=sha256(npz_path))
        normalized = copy.deepcopy(definition); normalized["rec"].pop("kd_weight", None)
        # stat.kd_weight is dormant for EDGE-H, but templates mirror reconstruction beta.
        normalized["stat"].pop("kd_weight", None)
        resolved = dict(identity=self.identity, definition=definition, recipe_hash=fingerprint(definition),
                        pair_recipe_hash=fingerprint(normalized), runtime=actual, framework_versions=versions,
                        implementation_file_hashes=implementation_sources,
                        reference_source=source, resolved_config_sha256=fingerprint(vars(a)))
        previous = _json(self.path) if self.path.exists() else None
        if previous:
            _equal(previous["recipe_hash"], resolved["recipe_hash"], "resume recipe")
            _equal(previous["config_sha256"], self.context["config_sha256"], "resume config")
            self.prefix = dict(previous.get("batch_meta_prefix", {}))
        def git(*arguments):
            return subprocess.check_output(["git", *arguments], cwd=ROOT, stderr=subprocess.DEVNULL).strip()
        release = git("rev-parse", "HEAD").decode()
        dirty_sha = hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest()
        self.manifest = dict(self.identity, original_run_id=self.identity["run_id"], method="qrecon_continuous_v1",
                             selector=SELECTOR, eval_mode="A_ON", source_train_code_ref=m.get("source_train_code_ref", "a565cafbb1d1f124dbb34450207b0374ade5aa42"),
                             release_sha=release, dirty_diff_sha=dirty_sha, config_sha256=self.context["config_sha256"],
                             implementation_file_hashes=implementation_sources,
                             resolved_config_sha256=resolved["resolved_config_sha256"], recipe_hash=resolved["recipe_hash"],
                             pair_recipe_hash=resolved["pair_recipe_hash"], init_unet_sha256=tensor_state_sha(trainer.M.backbone),
                             init_aligner_sha256=tensor_state_sha(trainer.M.aligner), teacher_sha256=TEACHER_SHA,
                             cue_sha256=sha256(npz_path), cue_manifest_sha256=sha256(cue_path), dataset_hashes=datasets,
                             started_at_utc=(previous or {}).get("started_at_utc", utc(self.context["started_at"])),
                             campaign_started_at_utc=self.context["plan"].get("campaign_started_at_utc", self.context["plan"].get("started_at_utc")),
                             clock_policy=self.context["plan"].get("clock_policy", "shared_explicit_20h"),
                             deadline_at_utc=self.context["plan"]["deadline_at_utc"], status="training",
                             actual_updates=(previous or {}).get("actual_updates", 0), training_complete=False,
                             batch_meta_prefix=self.prefix, precision=actual["mixed_precision"], pair_status="pair_unverified")
        evaluation_config = self.meta / "config.yaml"
        if evaluation_config.exists():
            _equal(sha256(evaluation_config), self.context["config_sha256"], "evaluation config snapshot")
        else:
            self.meta.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_path(a.config, ROOT), evaluation_config)
        atomic_json(self.meta / "resolved_recipe.json", resolved)
        self._write(); self._check_pair()

    def _write(self):
        atomic_json(self.path, self.manifest)

    def observe_batch(self, step, meta):
        """Hash full batch metadata for the first 16 *completed* updates, no RNG use."""
        if int(step) >= META_PREFIX_BATCHES:
            return
        if meta is None:
            raise ValueError("MIX20H missing batch metadata")
        values = meta.detach().cpu().tolist() if hasattr(meta, "detach") else meta
        key, value = str(int(step)), fingerprint(values)
        if key in self.prefix:
            _equal(self.prefix[key], value, "resumed batch metadata")
        self.prefix[key] = value
        self.manifest["batch_meta_prefix"] = self.prefix
        self.manifest["first_batch_meta_sequence_sha256"] = fingerprint([self.prefix[str(i)] for i in sorted(map(int, self.prefix))])
        self.manifest["batch_meta_prefix_complete"] = len(self.prefix) == META_PREFIX_BATCHES
        self._write(); self._check_pair()

    def _check_pair(self):
        mate = "B20A03" if self.identity["profile"] == "G23" else "G23"
        mate_id = self.identity["run_id"].replace("_" + self.identity["profile"] + "_", "_" + mate + "_")
        mate_path = self.wd.parent / mate_id / "meta/mix20h_run_manifest.json"
        lock = self.wd.parent / "_qrc24_mix20h" / "pairs" / (self.identity["pair_id"] + ".lock")
        with _lock(lock):
            reasons = []; other = _json(mate_path) if mate_path.exists() else None
            if other is None:
                reasons.append("pairmate_not_started")
            else:
                for field in ("pair_recipe_hash", "init_unet_sha256", "init_aligner_sha256"):
                    if self.manifest[field] != other.get(field):
                        reasons.append(field + "_mismatch")
                if not all(row.get("batch_meta_prefix_complete") for row in (self.manifest, other)):
                    reasons.append("batch_meta_prefix_incomplete")
                elif self.manifest["first_batch_meta_sequence_sha256"] != other.get("first_batch_meta_sequence_sha256"):
                    reasons.append("batch_meta_sequence_mismatch")
            status = "pair_unverified" if reasons else "pair_verified"
            for path, row in ((self.path, self.manifest), (mate_path, other)):
                if row is not None:
                    row.update(pair_status=status, pair_verification_reasons=reasons)
                    atomic_json(path, row)

    def expired(self, now=None):
        return (time.time() if now is None else now) >= self.context["deadline"]

    def stop_if_expired(self, trainer, step):
        """Call only between optimizer updates, before fetching the next batch."""
        if not self.expired():
            return
        if not trainer.accelerator.sync_gradients:
            raise RuntimeError("MIX20H budget stop attempted inside accumulation")
        step = int(step); checkpoint = self.wd / f"checkpoint-budget-{step}"
        trainer.accelerator.save_state(str(checkpoint))
        stopped = time.time()
        self.manifest.update(status="stopped_budget", actual_updates=step, training_complete=step == 50000,
                             resume_checkpoint=str(checkpoint), stopped_at_utc=utc(stopped),
                             budget_overrun_seconds=max(0., stopped - self.context["deadline"]),
                             auto_resume=False, official_target_complete=False, exact50k_complete=False)
        atomic_json(checkpoint / "mix20h_resume.json", dict(self.identity, step=step, exact_resume=True,
                    deadline_at_utc=self.manifest["deadline_at_utc"], status="stopped_budget"))
        self._write()
        trainer._runs_csv("stopped_budget"); trainer._write_cost("stopped_budget")
        trainer.accelerator.end_training()
        print(f"[mix20h] stopped_budget at completed update {step}; exact resume checkpoint: {checkpoint}")
        raise SystemExit(EXIT_STOPPED_BUDGET)

    def training_complete(self, step):
        self.manifest.update(actual_updates=int(step), training_complete=int(step) == 50000,
                             status="training_complete_eval_pending", training_completed_at_utc=utc())
        self._write()

    def stop_evaluation_if_expired(self, trainer, step):
        """Exports load older selected weights: NEVER checkpoint them as update 50K.

        The existing last checkpoint remains authoritative.  Only evaluation is
        interrupted here; no model/optimizer serialization and no train resume.
        """
        if not self.expired():
            return
        _equal(int(step), 50000, "export boundary actual completed updates")
        last_meta = _json(self.wd / "last_meta.json")
        _equal(last_meta.get("step"), 50000, "authoritative last checkpoint")
        if not (self.wd / "last/model.safetensors").is_file():
            raise FileNotFoundError("MIX20H completed training has no authoritative last checkpoint")
        stopped = time.time()
        self.manifest.update(status="training_complete_eval_pending_budget", actual_updates=50000,
                             training_complete=True, budget_stop_phase="evaluation", stopped_at_utc=utc(stopped),
                             authoritative_training_checkpoint=str(self.wd / "last"), auto_resume=False,
                             budget_overrun_seconds=max(0., stopped - self.context["deadline"]),
                             official_target_complete=False, exact50k_complete=False)
        self._write()
        trainer._runs_csv("training_complete_eval_pending_budget")
        trainer._write_cost("training_complete_eval_pending_budget")
        trainer.accelerator.end_training()
        print("[mix20h] budget closed during export; authoritative last preserved, evaluation pending")
        raise SystemExit(EXIT_STOPPED_BUDGET)

    def candidate(self, trainer, path, step, raw_hqnr):
        checkpoint = Path(path) / "model.safetensors"
        if not checkpoint.exists():
            raise FileNotFoundError(f"MIX20H expected candidate model missing: {checkpoint}")
        evaluation_config = self.meta / "config.yaml"
        atomic_json(Path(path) / "mix20h_identity.json", dict(self.identity, step=int(step),
                    checkpoint_sha256=sha256(checkpoint), config_sha256=self.context["config_sha256"],
                    evaluation_config_sha256=sha256(evaluation_config), fr_h5_sha256=trainer.fr_h5_sha,
                    evaluator_hash=trainer._mix20_evaluator_hash(), eval_mode="A_ON", precision=self.manifest["precision"],
                    raw_hqnr=float(raw_hqnr)))

    def commit_evaluation(self, trainer, step):
        """Called after main's selector, best-state JSON and metric writes finish.

        Re-save into the same candidate directory to capture post-evaluation RNG
        and optimizer state, not another copy.  The marker is published last.
        """
        step = int(step); path = self.wd / "candidates" / f"step-{step}"
        if (path / EVALUATION_COMMIT).exists():
            raise ValueError("MIX20H refusing to overwrite an already committed evaluation")
        binding = _json(path / "mix20h_identity.json")
        _equal(binding.get("run_id"), self.identity["run_id"], "evaluation commit run")
        _equal(binding.get("step"), step, "evaluation commit step")
        _equal(trainer._global_step, step, "live optimizer update at evaluation commit")
        before = sha256(path / "model.safetensors")
        _equal(binding.get("checkpoint_sha256"), before, "pre-commit model identity")
        trainer.accelerator.save_state(str(path))
        require_resume_files(path)
        _equal(sha256(path / "model.safetensors"), before, "model unchanged throughout evaluation")
        atomic_json(path / EVALUATION_COMMIT, dict(run_id=self.identity["run_id"], step=step,
                    checkpoint_sha256=before, post_evaluation=True, committed_at_utc=utc(),
                    state_file_sha256={name: sha256(path / name) for name in RESUME_FILES}))
        self.manifest.update(last_committed_evaluation_step=step, actual_updates=step)
        self._write()

    def has_committed_evaluation(self, step):
        path = self.wd / "candidates" / f"step-{int(step)}"
        if not (path / EVALUATION_COMMIT).exists():
            return False
        validate_candidate_commit(path, self.identity["run_id"], self.context["config_sha256"])
        return True
