"""FH20R1 trainer: immutable FH12 numerics, new A10 policy and durable diagnostics.

There is no wall-clock deadline here. A requested stop saves an exact full state;
the campaign runner owns the minimum-20-hour, completed-block stopping policy.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from safetensors.torch import save_file
import yaml

from fh12.common import ROOT, atomic_json, object_sha, read_json, resolved_path, sha256, utcnow
from fh12.training import BatchStream, rng_state, restore_rng, atomic_torch, tensor_cpu_tree
from fh12.data import build_dataset
from fh12.model import build_model, state_hash
from fh12.losses import teacher_loss, student_losses, routed_student_backward
from fh12.evaluation import FRMetrics, evaluate_model
from fh20r1.common import source_identity
from fh20r1.plan import CAMPAIGN_ID, GRID_STEPS, build_config, case_for

FULLSTATE_STEPS = (10000, 24240, 50000)
DIAGNOSTIC_EVERY = 5000
SCHEDULE_ID = "FH20R1_COSINE_W100_A10_AT_COMPLETED10000_v1"


def cosine_factor(completed_updates, warmup=100, total=50000):
    """Exactly the FH12 diffusers cosine/warmup multiplier, before next update."""
    if completed_updates < warmup:
        return float(completed_updates) / max(1, warmup)
    progress = float(completed_updates - warmup) / max(1, total - warmup)
    return max(0., .5 * (1. + math.cos(math.pi * progress)))


def aligner_factor(completed_updates, profile):
    if profile not in ("BASE", "A10", "N2INIT"):
        raise ValueError(f"Unknown FH20R1 optimizer profile {profile!r}")
    return 1. / 3. if profile == "A10" and completed_updates >= 10000 else 1.


def make_scheduler(optimizer, profile, warmup=100, total=50000):
    if [group.get("name") for group in optimizer.param_groups] != ["U", "A"]:
        raise ValueError("FH20R1 scheduler requires explicit U/A param groups")
    aligner_factor(0, profile)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [
        lambda t: cosine_factor(t, warmup, total),
        lambda t: cosine_factor(t, warmup, total) * aligner_factor(t, profile)])


class WorkJournal:
    """Nonoverlapping actual work segments, committed only with a full state."""
    def __init__(self, segments=()):
        self.segments = list(segments)
        self.active = None
        if len({row["id"] for row in self.segments}) != len(self.segments):
            raise ValueError("Duplicate timing segment IDs in resume")

    def start(self, kind, update):
        if self.active is not None or kind not in ("train", "eval", "diagnostic"):
            raise ValueError("Overlapping or invalid FH20R1 timing segment")
        self.active = (kind, int(update), time.time(), time.monotonic())

    def finish(self, update):
        if self.active is None:
            return None
        kind, start_update, epoch, monotonic = self.active
        seconds = max(0., time.monotonic() - monotonic)
        row = dict(id=uuid.uuid4().hex, kind=kind, update_from=start_update,
                   update_to=int(update), seconds=seconds,
                   start_utc=dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat(),
                   end_utc=dt.datetime.fromtimestamp(epoch + seconds, dt.timezone.utc).isoformat())
        self.segments.append(row)
        self.active = None
        return row

    def totals(self):
        return {kind: sum(float(row["seconds"]) for row in self.segments if row["kind"] == kind)
                for kind in ("train", "eval", "diagnostic")}


def write_csv_row(path, row, key="update"):
    """Atomic per-update upsert; replay after an interrupted commit adds no duplicates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if path.exists():
        with path.open(newline="") as stream:
            records = list(csv.DictReader(stream))
        records = [old for old in records if str(old[key]) != str(row[key])]
    records.append(row)
    records.sort(key=lambda record: int(record[key]))
    fields = list(dict.fromkeys(field for record in records for field in record))
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def save_committed_resume(wd, snapshot, run_id, config_sha):
    """State, identity and timing publish through one atomic directory link."""
    wd = Path(wd); wd.mkdir(parents=True, exist_ok=True)
    (wd / "meta").mkdir(exist_ok=True)
    target = wd / "last"
    if target.exists() and not target.is_symlink():
        raise ValueError("FH20R1 resume target must be an atomic bundle link")
    old = target.resolve() if target.is_symlink() else None
    pending = Path(tempfile.mkdtemp(prefix=".fh20r1_resume_", dir=wd))
    temporary_link = wd / (pending.name + ".link")
    timing_link = wd / "meta/timing_committed.json"
    published = False
    try:
        atomic_torch(pending / "training_state.pt", snapshot)
        identity = dict(update=snapshot["update"], config_sha256=config_sha, full_state=True,
                        training_state_sha256=sha256(pending / "training_state.pt"))
        atomic_json(pending / "identity.json", identity)
        atomic_json(pending / "timing_committed.json", dict(run_id=run_id, campaign_id=CAMPAIGN_ID,
                    config_sha256=config_sha, segments=snapshot["timing_segments"],
                    committed_update=snapshot["update"], full_state_identity_path="last/identity.json",
                    valid_fullstate=True, training_state_sha256=identity["training_state_sha256"],
                    note="This commit's own I/O is eligible only after a subsequent full-state commit."))
        if timing_link.is_symlink():
            if os.readlink(timing_link) != "../last/timing_committed.json":
                raise ValueError("Unexpected FH20R1 committed timing link")
        elif timing_link.exists():
            raise ValueError("FH20R1 committed timing must be an atomic bundle link")
        else:
            timing_link.symlink_to("../last/timing_committed.json")
        temporary_link.symlink_to(pending.name, target_is_directory=True)
        os.replace(temporary_link, target); published = True
    finally:
        if temporary_link.is_symlink(): temporary_link.unlink()
        if not published: shutil.rmtree(pending)
    if old is not None and old.parent == wd.resolve() and old.name.startswith(".fh20r1_resume_"):
        shutil.rmtree(old)


def loss_activity(losses, out, teacher_out, gt, raw_q=None):
    """Read-only scalar diagnostics; no new teacher query or optimizer work."""
    if teacher_out is None:
        return dict(hard=float(losses["rec"].detach()), soft=0., edge=0.,
                    offset=float(losses["off"].detach()), offset_active=int(losses["offset_active"]))
    with torch.no_grad():
        es = (out["y"].float() - gt).abs().mean(1)
        et = (teacher_out["y"].float() - gt).abs().mean(1)
        s = losses["q_weights"]
        row = dict(hard=float(losses["hard"].detach()), soft=float(losses["soft"].detach()),
                   edge=float(losses["edge"].detach()), edge_weighted=float(losses["edge_weighted"].detach()),
                   L_U=float(losses["L_U"].detach()), L_A=float(losses["L_A"].detach()),
                   soft_positive_fraction=float((losses["advantage"] > 0).float().mean()),
                   student_better_fraction=float((es < et).float().mean()),
                   difficulty_mean=float(losses["difficulty"].mean()),
                   s_mean=float(s.mean()), s_std=float(s.std(unbiased=False)),
                   s_min=float(s.min()), s_max=float(s.max()))
        if raw_q is not None:
            q = raw_q.detach().float()
            row.update(q_mean=float(q.mean()), q_std=float(q.std(unbiased=False)),
                       q_min=float(q.min()), q_max=float(q.max()))
        return row


def _gradient_norm(value, parameters):
    if not value.requires_grad:
        return 0.
    gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    squared = sum(float(grad.detach().double().square().sum()) for grad in gradients if grad is not None)
    return math.sqrt(squared)


def gradient_diagnostic(model, teacher, batch, tau_R=None, q_weights=None, corruption_seed=1234):
    """Fixed-mini-batch autograd.grad only; actual .grad/RNG/sample order untouched."""
    state = rng_state(); was_training = model.training
    existing_grads = {name: parameter.grad for name, parameter in model.named_parameters()}
    model.eval()
    try:
        gt, _lms, ms, lp, pan, _meta = batch
        out = model(pan, ms, lp)
        if teacher is None:
            generator = torch.Generator().manual_seed(int(corruption_seed))
            losses = teacher_loss(model, out, gt, pan, ms, 1, generator)
            terms = {"hard": losses["rec"], "soft": out["y"].new_zeros(()),
                     "edge": out["y"].new_zeros(()), "offset": 1e-4 * losses["off"]}
        else:
            with torch.no_grad(): teacher_out = teacher(pan, ms, lp)
            losses = student_losses(out, teacher_out, gt, tau_R, q_weights)
            terms = {"hard": losses["hard"], "soft": losses["soft"],
                     "edge": losses["edge_weighted"], "q_hard": losses["L_A"]}
        groups = {"U": list(model.backbone.parameters()), "A": list(model.aligner.parameters())}
        result = {f"raw_{term}_{group}": _gradient_norm(value, params)
                  for term, value in terms.items() for group, params in groups.items()}
        result.update(applied_soft_A=0., applied_edge_A=0.,
                      gradient_meaning="raw objective Jacobians; actual Student A receives q_hard only",
                      actual_gradients_modified=False)
        if any(parameter.grad is not existing_grads[name] for name, parameter in model.named_parameters()):
            raise AssertionError("Gradient diagnostic changed actual optimizer gradients")
        return result
    finally:
        model.train(was_training)
        restore_rng(state)


def build_n2_teacher(case, donor_state, donor_report, fresh_f2_init):
    """Same F2 named-tensor U; donor A only, preserving its trained final head."""
    if donor_state is None or donor_report.get("status") == "BLOCKED_DONOR":
        raise ValueError("BLOCKED_DONOR: cannot replace N2INIT with fresh A")
    model, manifest = build_model(case.input_layout, case.width, list(case.depth), case.seed, role="T")
    for key in ("P", "MS", "input_bias", "body"):
        if manifest["hashes"][key] != fresh_f2_init["hashes"][key]:
            raise ValueError(f"N2INIT fresh U differs from F2 initial {key}")
    fresh_a_hash = manifest["hashes"]["A"]
    model.aligner.load_state_dict({k: v.detach().clone() for k, v in donor_state.items()}, strict=True)
    manifest.update(from_scratch=False, fresh_U=True, teacher_from_scratch=False,
                    pretrained_aligner_loads=1, aligner_initialization="N2_DONOR_ONLY",
                    donor=donor_report, fresh_F2_U_verified=True, discarded_fresh_A_hash=fresh_a_hash)
    manifest["hashes"]["A"] = state_hash(model.aligner.state_dict())
    manifest["hashes"]["full"] = state_hash(model.state_dict())
    return model, manifest


def validate_config(cfg):
    case = case_for(Path(cfg["work_dir"]).name)
    if cfg != build_config(case):
        raise ValueError("FH20R1 config differs from its registered immutable recipe")
    return case


def runtime_context(cfg, root, device):
    f = cfg["fh20r1"]
    budget = read_json(resolved_path(f["budget_path"], root))
    if (budget.get("campaign_id") != CAMPAIGN_ID or budget.get("server_id") != f["server_id"]
            or budget.get("actual_start_authorized") is not True or budget.get("device") != device):
        raise ValueError("FH20R1 requires its explicitly activated local campaign/device")
    if budget.get("hard_deadline") is not None or float(budget.get("minimum_effective_hours", 0)) != 20:
        raise ValueError("FH20R1 is minimum20h, never a reused/extended FH12 deadline")
    return budget


def train_run(config_path, device="cuda", resume=False, root=ROOT):
    from fh20r1.references import load_training_reference, load_donor_aligner, calibration_indices
    root = Path(root); cfg = yaml.safe_load(Path(config_path).read_text()); case = validate_config(cfg)
    budget = runtime_context(cfg, root, device)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; FH20R1 will not silently train on CPU")
    if cfg.get("mixed_precision") != "no":
        raise ValueError("FH20R1 preserves FH12 FP32/no-AMP numerics")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if dev.type == "cuda": torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    f = cfg["fh20r1"]; wd = resolved_path(cfg["work_dir"], root)
    release = source_identity(root); config_sha = object_sha(cfg)
    data = read_json(resolved_path(f["dataset_manifest"], root)); data_sha = object_sha(data)
    datasets = {split: build_dataset(data, split, root=root) for split in ("train", "val", "rr", "fr")}
    teacher = reference = q = bridge = None
    if case.role == "S":
        teacher, reference, q, bridge = load_training_reference(case.teacher_alias, case.server_id, root, dev)
        teacher.eval().requires_grad_(False)
        model, init = build_model(case.input_layout, case.width, list(case.depth), case.seed,
                                  role="S", teacher_aligner_state=teacher.aligner.state_dict())
        q = torch.as_tensor(q, dtype=torch.float32, device=dev)
        if q.shape != (len(datasets["train"]), 4) or not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
            raise ValueError("FH20R1 reference q must cover every train index/rot view")
        if not math.isfinite(float(reference["q_ref"])) or float(reference["q_ref"]) <= 0:
            raise ValueError("Invalid imported q_ref")
        diag_alias = case.teacher_alias
    else:
        donor, donor_report = load_donor_aligner(root, server=case.server_id)
        _f2, _reference, _q, bridge = load_training_reference("F2", case.server_id, root, "cpu")
        del _f2, _q
        fresh_init = read_json(bridge["resolved_artifacts"]["init_manifest"])
        model, init = build_n2_teacher(case, donor, donor_report, fresh_init)
        diag_alias = "F2"
    teacher_hash = state_hash(teacher.state_dict()) if teacher is not None else None
    reference_sha = object_sha(bridge)
    model.to(dev).train()
    optimizer = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=cfg["learning_rate"], name="U"),
        dict(params=model.aligner.parameters(), lr=f["aligner_lr"], name="A")],
        betas=tuple(cfg["betas"]), eps=cfg["eps"], weight_decay=cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, case.profile, cfg["num_warmup"], cfg["num_iter"])
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    stream = BatchStream(len(datasets["train"]), cfg["batch_size"], cfg["seed"])
    corruption = torch.Generator().manual_seed(f["corruption_seed"])
    fixed_ids = np.asarray(calibration_indices(diag_alias, case.server_id, root), dtype=np.int64)[:8]
    if len(fixed_ids) != 8 or len(set(fixed_ids.tolist())) != 8:
        raise ValueError("Fixed gradient diagnostic requires eight original calibration indices")
    fixed_batch = default_collate([datasets["train"][(int(index), 0)] for index in fixed_ids])
    fixed_batch = tuple(value.to(dev) for value in fixed_batch)
    diag_weights = None if teacher is None else float(reference["q_ref"]) / (float(reference["q_ref"]) + q[torch.as_tensor(fixed_ids, device=dev), 0])
    update = 0; records = []; diagnostics_done = []; journal = WorkJournal()
    io_seconds = 0.; state_path = wd / "last/training_state.pt"
    if resume:
        identity = read_json(wd / "last/identity.json")
        if sha256(state_path) != identity["training_state_sha256"]:
            raise ValueError("FH20R1 resume full-state checksum mismatch")
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        for key, wanted in (("config_sha256", config_sha), ("data_sha256", data_sha),
                            ("source_identity", release), ("reference_sha256", reference_sha)):
            if saved.get(key) != wanted: raise ValueError(f"FH20R1 exact-resume mismatch: {key}")
        if not saved.get("full_state"): raise ValueError("Weights-only checkpoint cannot exact resume")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"]); stream.load_state_dict(saved["sampler"])
        corruption.set_state(saved["corruption_rng"])
        update = int(saved["update"]); diagnostics_done = list(saved["diagnostics_done"])
        journal = WorkJournal(saved["timing_segments"]); io_seconds = saved.get("io_seconds", 0.)
        records = read_json(wd / "official/raw_grid.json")["records"] if (wd / "official/raw_grid.json").exists() else []
        if any(int(record["update"]) > update for record in records):
            raise ValueError("Metric grid is ahead of the valid full-state checkpoint")
        known = {row["id"] for row in journal.segments}
        for record in records:
            segment = record.get("timing_segment")
            if segment and segment["id"] not in known:
                journal.segments.append(segment); known.add(segment["id"])
        restore_rng(saved["rng"])
        if scheduler.last_epoch != update:
            raise ValueError("Resume scheduler does not match completed update count")
    elif state_path.exists() or (wd / "meta/training_start_manifest.json").exists():
        raise ValueError("FH20R1 run exists; exact resume required, never reinitialize")
    else:
        (wd / "meta").mkdir(parents=True, exist_ok=True)
        (wd / "meta/config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        atomic_json(wd / "init_manifest.json", dict(init, reference_bridge_sha256=reference_sha))
        atomic_json(wd / "meta/training_start_manifest.json", dict(campaign_id=CAMPAIGN_ID,
                    run_id=case.run_id, config_sha256=config_sha, data_sha256=data_sha,
                    source_identity=release, reference_sha256=reference_sha, budget=budget,
                    started_at_utc=utcnow(), device=str(dev), precision="fp32", schedule_id=SCHEDULE_ID))
        atomic_json(wd / "diagnostics/loss_routing.json", dict(role=case.role,
                    U_objective="rec" if case.role == "T" else "hard+soft+.002*s*edge",
                    A_objective="rec+odd1e-4offset" if case.role == "T" else "s*hard",
                    student_soft_to_A=False, student_edge_to_A=False, profile=case.profile,
                    fixed_calibration_indices=fixed_ids.tolist(), fixed_rot=0, fixed_HV=True))

    pause = [False]
    signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in signals: signal.signal(sig, lambda *_: pause.__setitem__(0, True))
    engine = None

    def full_state():
        totals = journal.totals()
        return dict(full_state=True, update=update, model_state=tensor_cpu_tree(model.state_dict()),
                    optimizer=tensor_cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(), rng=rng_state(), sampler=stream.state_dict(),
                    corruption_rng=corruption.get_state(), config_sha256=config_sha,
                    data_sha256=data_sha, source_identity=release, reference_sha256=reference_sha,
                    timing_segments=list(journal.segments), diagnostics_done=list(diagnostics_done),
                    training_seconds=totals["train"], evaluation_seconds=totals["eval"],
                    diagnostic_seconds=totals["diagnostic"], io_seconds=io_seconds,
                    schedule_id=SCHEDULE_ID, diagnostic_indices=fixed_ids.tolist())

    def status(name):
        totals = journal.totals()
        atomic_json(wd / "meta/training_status.json", dict(status=name, actual_updates=update,
                    training_complete=update == cfg["num_iter"], training_seconds=totals["train"],
                    evaluation_seconds=totals["eval"], diagnostic_seconds=totals["diagnostic"],
                    io_seconds=io_seconds, updated_at_utc=utcnow(), hard_deadline=None))

    def save_resume():
        nonlocal io_seconds
        if journal.active is not None: raise AssertionError("Close work segment before full-state commit")
        snapshot = full_state()
        journal.start("train", update)
        started = time.monotonic()
        save_committed_resume(wd, snapshot, case.run_id, config_sha)
        io_seconds += time.monotonic() - started
        journal.finish(update)

    def save_bundle(dest, include_state):
        dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            ident = read_json(dest / "identity.json")
            if ident["state_hash"] != state_hash(model.state_dict()) or ident["config_sha256"] != config_sha or sha256(dest / "model.safetensors") != ident["model_sha256"]:
                raise ValueError("Existing immutable FH20R1 checkpoint differs")
            return ident
        pending = Path(tempfile.mkdtemp(prefix=f".{update}-", dir=dest.parent))
        try:
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(pending / "model.safetensors"))
            ident = dict(update=update, config_sha256=config_sha, data_sha256=data_sha,
                         source_identity=release, reference_sha256=reference_sha,
                         input_layout=case.input_layout, role=case.role, state_hash=state_hash(model.state_dict()),
                         model_sha256=sha256(pending / "model.safetensors"), full_state=bool(include_state))
            snap = full_state() if include_state else dict(full_state=False, update=update, config_sha256=config_sha)
            snap["model_sha256"] = ident["model_sha256"]
            atomic_torch(pending / "training_state.pt", snap)
            ident["training_state_sha256"] = sha256(pending / "training_state.pt")
            atomic_json(pending / "identity.json", ident)
            os.replace(pending, dest)
        finally:
            if pending.exists(): shutil.rmtree(pending)
        return ident

    def evaluate_update():
        nonlocal engine, records
        if update not in GRID_STEPS or any(record["update"] == update for record in records): return
        identity = save_bundle(wd / "candidates" / str(update), update == 50000)
        journal.start("eval", update); training_rng = rng_state()
        try:
            if engine is None: engine = FRMetrics(datasets["fr"])
            result = evaluate_model(model, datasets, dev, engine, None, include_q=True, with_val=True)
        finally:
            restore_rng(training_rng)
        segment = journal.finish(update)
        if teacher is not None and state_hash(teacher.state_dict()) != teacher_hash:
            raise AssertionError("Frozen reference Teacher changed")
        record = dict(update=update, checkpoint_identity=identity, timing_segment=segment, **result)
        records.append(record)
        atomic_json(wd / "official/raw_grid.json", dict(campaign_id=CAMPAIGN_ID, run_id=case.run_id,
                    expected_steps=list(GRID_STEPS), records=records,
                    complete=[row["update"] for row in records] == list(GRID_STEPS),
                    config_sha256=config_sha, data_sha256=data_sha, source_identity=release))
        atomic_json(wd / "official" / f"candidate_{update}.json", record)
        write_csv_row(wd / "checkpoint_metrics.csv", dict(update=update, hqnr_official=result["fr"]["hqnr"],
                      rr_scc_official=result["rr"]["scc"], rr_ergas_official=result["rr"]["ergas"],
                      val_ergas=result["val_ergas"], d_lambda=result["fr"]["d_lambda"], d_s=result["fr"]["d_s"]))
        shift = result["shift"]
        write_csv_row(wd / "diagnostics/frequency_shift.csv", dict(update=update,
                      fr_dy=shift["fr_mean"][0], fr_dx=shift["fr_mean"][1], fr_max_abs=shift["fr_max_abs"],
                      rr_dy=shift["rr_mean"][0], rr_dx=shift["rr_mean"][1], rr_max_abs=shift["rr_max_abs"]))
        print(f'[FH20R1 official step={update}] HQNR={result["fr"]["hqnr"]:.8f} SCC={result["rr"]["scc"]:.8f} ERGAS={result["rr"]["ergas"]:.8f} valERGAS={result["val_ergas"]:.8f}', flush=True)
        save_resume()

    def run_diagnostic():
        if update <= 0 or update % DIAGNOSTIC_EVERY or update in diagnostics_done: return
        journal.start("diagnostic", update)
        result = gradient_diagnostic(model, teacher, fixed_batch,
                    tau_R=(reference or {}).get("tau_R"), q_weights=diag_weights,
                    corruption_seed=f["corruption_seed"] + 600000)
        write_csv_row(wd / "diagnostics/gradient_norms.csv", dict(update=update, **result))
        diagnostics_done.append(update); journal.finish(update)

    try:
        status("RUNNING")
        run_diagnostic()
        if update in FULLSTATE_STEPS:
            save_bundle(wd / "restart_fullstates" / str(update), True)
        evaluate_update()
        while update < cfg["num_iter"]:
            if pause[0]: save_resume(); status("PAUSED_SIGNAL"); return 75
            if stream.cursor == stream.count: stream.new_epoch()
            journal.start("train", update)
            epoch_rng = torch.Generator().manual_seed(cfg["seed"] + 500000 + stream.epoch)
            loader = DataLoader(datasets["train"], batch_sampler=stream.remaining_batches(),
                                num_workers=cfg["num_worker"], pin_memory=dev.type == "cuda", generator=epoch_rng)
            for gt, _lms, ms, lp, pan, meta in loader:
                if pause[0]:
                    journal.finish(update); save_resume(); status("PAUSED_SIGNAL"); return 75
                model.train(); gt, ms, lp, pan = [value.to(dev, non_blocking=True) for value in (gt, ms, lp, pan)]
                optimizer.zero_grad(set_to_none=True)
                actual_lrs = [group["lr"] for group in optimizer.param_groups]
                out = model(pan, ms, lp)
                teacher_out = raw_q = None
                if teacher is None:
                    losses = teacher_loss(model, out, gt, pan, ms, update, corruption)
                    if not bool(torch.isfinite(losses["total"])): raise FloatingPointError("Nonfinite Teacher loss")
                    losses["total"].backward(); shown = float(losses["total"].detach())
                else:
                    with torch.no_grad(): teacher_out = teacher(pan, ms, lp)
                    raw_q = q[meta[:, 0].to(dev), meta[:, 1].to(dev)]
                    weights = (float(reference["q_ref"]) / (float(reference["q_ref"]) + raw_q)).detach()
                    losses = student_losses(out, teacher_out, gt, float(reference["tau_R"]), weights)
                    if not bool(torch.isfinite(losses["L_U"]) & torch.isfinite(losses["L_A"])):
                        raise FloatingPointError("Nonfinite Student objective")
                    routed_student_backward(model, losses); shown = float(losses["L_U"].detach())
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
                    raise FloatingPointError("Missing/nonfinite FH20R1 gradients")
                optimizer.step(); scheduler.step(); update += 1; stream.cursor += 1
                if update % cfg["log_iter"] == 0 or update in (9999, 10000, 10001):
                    write_csv_row(wd / "diagnostics/lr_groups.csv", dict(update=update,
                        update_index=update - 1, profile=case.profile, lr_U_used=actual_lrs[0], lr_A_used=actual_lrs[1],
                        next_lr_U=optimizer.param_groups[0]["lr"], next_lr_A=optimizer.param_groups[1]["lr"],
                        A_multiplier_used=aligner_factor(update - 1, case.profile)))
                    write_csv_row(wd / "diagnostics/loss_terms_and_activity.csv", dict(update=update,
                                  **loss_activity(losses, out, teacher_out, gt, raw_q)))
                    print(f"[FH20R1 {case.role}/{case.profile}] update={update}/50000 loss={shown:.8g} lrU={actual_lrs[0]:.8g} lrA={actual_lrs[1]:.8g}", flush=True)
                boundary = update in GRID_STEPS or update in FULLSTATE_STEPS or update % DIAGNOSTIC_EVERY == 0
                if boundary:
                    journal.finish(update); run_diagnostic(); save_resume()
                    if update in FULLSTATE_STEPS:
                        save_bundle(wd / "restart_fullstates" / str(update), True)
                    evaluate_update(); status("RUNNING")
                    if update < cfg["num_iter"]: journal.start("train", update)
                if update >= cfg["num_iter"]: break
            journal.finish(update)
        save_resume(); status("TRAINING_COMPLETE"); return 0
    except Exception:
        # Unfinished work/diagnostics are not valid time credit. Previous valid
        # fullstate remains intact and is the only permissible exact resume.
        journal.active = None; status("FAILED"); raise
    finally:
        for sig, handler in signals.items(): signal.signal(sig, handler)
