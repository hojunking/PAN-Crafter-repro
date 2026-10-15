"""Frozen, single-process PANPV v3 scene-adaptation campaign.

Only a supplied, hash-bound registry can launch this controller. It never
discovers another donor, trains a replacement donor, changes a protocol, or
writes a remote sheet. Those decisions belong to the caller before admission.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import tarfile
import tempfile

from .common import atomic_json, file_sha, load_donor, load_json


SCHEMA = "PANPV_CAMPAIGN_v1"
ENDPOINT = 20000
BLOCK = 2000
METHODS = ("SC_SCENE", "FT_WV3_SCENE")
MIN_FREE_BYTES = 10 * 1024 ** 3
REQUIRED_SOURCE_FILES = (
    "panpv/campaign.py", "panpv/campaign_eval.py", "panpv/training.py",
    "panpv/v3_data.py", "panpv/datasetcache.py", "panpv/common.py", "panpv/brovey.py", "panpv/evaluation.py",
    "deploy/pandep_shared/project/src/pan_shared/model.py",
    "deploy/pandep_shared/project/src/pan_shared/blocks.py",
    "deploy/pandep_shared/project/src/pan_shared/frontend.py",
    "deploy/pandep_shared/project/src/pan_shared/sampling.py",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _name(value):
    if (not isinstance(value, str) or not value or len(value) > 180
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value)):
        raise ValueError("Unsafe registry identifier")
    return value


def _bound_file(record, path_key="path", sha_key="sha256"):
    path = Path(record[path_key])
    sha = record[sha_key]
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or not isinstance(sha, str) or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)):
        raise ValueError("An absolute regular file and SHA256 are required")
    if file_sha(path) != sha:
        raise ValueError("Bound file SHA256 mismatch: " + str(path))
    return path.resolve()


def validate_registry(registry):
    """Read-only verification; creation of a registry is not an admission."""
    if registry.get("schema") != SCHEMA:
        raise ValueError("Unsupported campaign registry")
    _name(registry["campaign_id"])
    if (registry.get("execution_server") != "s1" or registry.get("target_sensor") != "WV2"
            or registry.get("seed") != 261002 or registry.get("device") != "cuda"
            or registry.get("micro_batch") not in (12, 24, 48)):
        raise ValueError("Only frozen s1/WV2/R01/FP32/effective48 settings are supported")
    source_root = Path(registry["source_root"]).resolve(strict=True)
    if source_root != Path(__file__).resolve().parents[1]:
        raise ValueError("Registry source_root is not the executing source tree")
    source_hashes = registry["source_sha256"]
    if not set(REQUIRED_SOURCE_FILES).issubset(source_hashes):
        raise ValueError("Incomplete executable source hash coverage")
    for relative, sha in source_hashes.items():
        path = source_root / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(source_root):
            raise ValueError("Source manifest escapes the repository")
        _bound_file(dict(path=str(path), sha256=sha))
    input_root = Path(registry["input_root"]).resolve(strict=True)
    output = Path(registry["work_root"])
    if (not output.is_absolute() or output.is_symlink()
            or not output.resolve().is_relative_to(source_root / "work_dir")
            or output.resolve() == source_root / "work_dir"
            or output.resolve().is_relative_to(input_root)
            or input_root.is_relative_to(output.resolve())):
        raise ValueError("A distinct campaign directory under work_dir is required")
    entries = registry["scenes"]
    if len(entries) != 5 or len({_name(s["scene_id"]) for s in entries}) != 5:
        raise ValueError("Exactly five distinct, explicitly registered scenes are required")
    seen = set()
    for entry in entries:
        relative = Path(entry["file"])
        path = input_root / relative
        if relative.is_absolute() or not path.resolve().is_relative_to(input_root) or path.is_symlink():
            raise ValueError("Scene path leaves the input root")
        if path.resolve() in seen:
            raise ValueError("Duplicate scene file")
        seen.add(path.resolve())
        if not path.is_file() or len(entry.get("patch_index_sha256", "")) != 64:
            raise ValueError("Missing scene or frozen split identity")
    donor = registry["donor"]
    donor_path = _bound_file(donor)
    if donor_path.suffix != ".safetensors":
        raise ValueError("Only a verified safetensors donor is accepted")
    _bound_file(donor, "provenance_path", "provenance_sha256")
    protocol = load_json(_bound_file(registry["protocol"]))
    if set(protocol.get("scenes", {})) != {s["scene_id"] for s in entries}:
        raise ValueError("Frozen evaluation protocol must cover all five scenes")
    preflight = load_json(_bound_file(registry["preflight"]))
    if (preflight.get("status") != "PASS" or preflight.get("donor_sha256") != donor["sha256"]
            or preflight.get("micro_batch") != registry["micro_batch"]
            or preflight.get("ft0_zs_parity") is not True
            or preflight.get("microbatch_verified") is not True):
        raise ValueError("Donor/FT0/production-microbatch admission is not verified")
    return protocol


def stage_plan(scene_ids):
    """Deterministic queue, including both selectors even when weights coincide."""
    stages = []
    for scene in scene_ids:
        for method in ("BROVEY8", "ZS_WV3"):
            stages.append(dict(id=f"{scene}_{method}", kind="baseline", scene_id=scene, method=method))
    for target in range(BLOCK, ENDPOINT + 1, BLOCK):
        for scene in scene_ids:
            for method in METHODS:
                stages.append(dict(id=f"{scene}_{method}_TRAIN_{target:08d}", kind="train",
                                   scene_id=scene, method=method, target_step=target))
    for scene in scene_ids:
        for method in METHODS:
            for selector in ("EXACT20K", "BEST_RRVAL_TO_20K"):
                stages.append(dict(id=f"{scene}_{method}_{selector}", kind="final",
                                   scene_id=scene, method=method, selector=selector))
    return stages


class Campaign:
    def __init__(self, registry_path):
        self.registry_path = Path(registry_path).resolve(strict=True)
        self.registry = load_json(self.registry_path)
        self.protocol = validate_registry(self.registry)
        self.registry_sha = file_sha(self.registry_path)
        self.root = Path(self.registry["work_root"]).resolve()
        self.scenes = {}
        self.datasets = {}
        self.entries = {s["scene_id"]: s for s in self.registry["scenes"]}
        self.queue = stage_plan(self.entries)
        self.stop_requested = False
        self.stop_signal = None
        self.active_engine = None

    def _owned(self, path):
        path = Path(path)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Output path escapes campaign root")
        return path

    def _space(self):
        if shutil.disk_usage(self.root).free < MIN_FREE_BYTES:
            raise OSError("SAFE_STOP_DISK: less than 10 GiB free; nothing was deleted")

    def _status(self, state, **extra):
        atomic_json(self.root / "campaign_status.json", dict(schema="PANPV_CAMPAIGN_STATUS_v1",
                    state=state, campaign_id=self.registry["campaign_id"], registry_sha256=self.registry_sha,
                    updated_utc=_now(), pid=os.getpid(), stop_signal=self.stop_signal, **extra))

    def _scene(self, scene_id):
        from .v3_data import SceneV3
        if scene_id not in self.scenes:
            entry = self.entries[scene_id]
            path = Path(self.registry["input_root"]) / entry["file"]
            scene = SceneV3(path, expected_sha256=entry["sha256"])
            summary = scene.summary()
            if scene.scene_id != scene_id or summary["patch_index_sha256"] != entry["patch_index_sha256"]:
                scene.close()
                raise ValueError("Scene/split identity does not match registry")
            self.scenes[scene_id] = scene
        return self.scenes[scene_id]

    def _model(self, method):
        if method == "BROVEY8":
            return None
        if method == "SC_SCENE":
            from deploy.pandep_shared.project.src.pan_shared.model import SingleSensorPLHUNet
            return SingleSensorPLHUNet("WV3", width=104, depth=(1, 2, 2), seed=self.registry["seed"])
        return load_donor(self.registry["donor"]["path"], self.registry["donor"]["sha256"],
                          seed=self.registry["seed"])

    def _dataset(self, scene, split):
        from .datasetcache import CachedRRDataset
        from .v3_data import RRPatchV3
        key = (scene.scene_id, split)
        if key not in self.datasets:
            print(f"PANPV materializing verified RR RAM cache {scene.scene_id} {split}", flush=True)
            cached = CachedRRDataset(RRPatchV3(scene, split))
            manifest = cached.manifest
            path = self.root / "cache_manifests" / f"{scene.scene_id}_{split}.json"
            if path.exists():
                if load_json(path)["identity_sha256"] != manifest["identity_sha256"]:
                    raise ValueError("Cached RR byte identity changed across restart")
            else:
                atomic_json(path, manifest)
            # Cache creation is preprocessing, never hidden in adaptation cost.
            event = dict(recorded_utc=_now(), pid=os.getpid(), **manifest)
            with (self.root / "cache_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.datasets[key] = cached
        return self.datasets[key]

    def _evaluate(self, scene, method, model, directory, *, weight_sha=None, final=False):
        from .campaign_eval import evaluate_scene
        self._space()
        if model is not None:
            model.to(device=self.registry["device"]).eval()
        return evaluate_scene(scene, self.protocol["scenes"][scene.scene_id], method=method,
                              model=model, device=self.registry["device"], output_dir=directory,
                              checkpoint_sha256=weight_sha, final=final)

    def _run_root(self, scene_id, method):
        return self.root / "runs" / f"PV_S1_{scene_id}_{method}_R01_SE261002"

    def _train(self, stage):
        from .training import TrainingEngine
        from deploy.pandep_shared.project.src.pan_shared.checkpoints import tensor_state_hash
        scene, method = self._scene(stage["scene_id"]), stage["method"]
        run_root = self._run_root(scene.scene_id, method)

        def diagnostics(model, step, run_dir):
            self._space()
            diagnostic = self._evaluate(scene, method, model,
                                       run_dir / "diagnostics" / f"step_{step:08d}", final=False)
            diagnostic.update(completed_step=step, model_tensor_sha256=tensor_state_hash(model.state_dict()))
            return diagnostic

        model = self._model(method)
        train_dataset, val_dataset = self._dataset(scene, "train"), self._dataset(scene, "val")
        identity = dict(campaign_id=self.registry["campaign_id"], registry_sha256=self.registry_sha,
                        scene_id=scene.scene_id, input_sha256=scene.sha256,
                        split_sha256=self.entries[scene.scene_id]["patch_index_sha256"],
                        protocol_sha256=self.registry["protocol"]["sha256"],
                        source_sha256=self.registry["source_sha256"], target_sensor="WV2",
                        train_cache_sha256=train_dataset.manifest["identity_sha256"],
                        val_cache_sha256=val_dataset.manifest["identity_sha256"],
                        parent_sha256=self.registry["donor"]["sha256"] if method == "FT_WV3_SCENE" else None)
        engine = TrainingEngine(model, train_dataset, val_dataset, run_root,
                                method=method, identity=identity, seed=self.registry["seed"],
                                device=self.registry["device"], micro_batch=self.registry["micro_batch"],
                                audit_callback=diagnostics, sensor_id="WV3")
        records = {
            "run_manifest.json": dict(identity, method=method, seed=self.registry["seed"],
                                      execution_server="s1", target_sensor="WV2", state_key_sensor="WV3",
                                      run_id=run_root.name, input_adaptation=True,
                                      comparison_scope="WITHIN_SCENE_SPATIAL_HOLDOUT_AND_NATIVE_FR_TILES"),
            "config.resolved.json": engine.config,
            "architecture_manifest.json": model.architecture_manifest(),
        }
        for name, value in records.items():
            path = run_root / name
            if path.exists() and load_json(path) != value:
                raise ValueError("Immutable run metadata changed: " + name)
            atomic_json(path, value)
        self.active_engine = engine
        try:
            if (run_root / "latest.json").exists():
                engine.resume(trusted_local=True)
            if engine.train["step"] > stage["target_step"]:
                raise ValueError("Queue receipt missing for a checkpoint beyond this stage")
            if self.stop_requested:
                engine.request_stop(self.stop_signal)
                return dict(status="PAUSED_BEFORE_UPDATE", completed_step=engine.train["step"])
            state = engine.run_until(stage["target_step"])
            if state["status"] == "PAUSED_SIGNAL":
                self.stop_requested = True
                self.stop_signal = engine._pause_signal
                return dict(status="PAUSED_SIGNAL", completed_step=state["step"])
            checkpoint = load_json(run_root / "latest.json")
            return dict(status="TRAIN_BLOCK_COMPLETE", completed_step=state["step"],
                        checkpoint=checkpoint, run_root=str(run_root),
                        validation_l1=state["best_val_l1"], selected_step=state["best_step"])
        finally:
            self.active_engine = None
            del engine, model

    def _stage(self, stage):
        scene, method = self._scene(stage["scene_id"]), stage["method"]
        if stage["kind"] == "train":
            return self._train(stage)
        model = self._model(method)
        weight_sha = self.registry["donor"]["sha256"] if method == "ZS_WV3" else None
        selected_step = 0
        if stage["kind"] == "final":
            from safetensors.torch import load_file
            pointer_name = "exact_endpoint.json" if stage["selector"] == "EXACT20K" else "best_val_l1.json"
            run_root = self._run_root(scene.scene_id, method)
            pointer = load_json(run_root / pointer_name)
            weights = self._owned(run_root / pointer["directory"] / "model.safetensors")
            weight_sha = pointer["model_sha256"]
            if file_sha(weights) != weight_sha:
                raise ValueError("Selected checkpoint bytes changed")
            model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
            selected_step = pointer["completed_step"]
            if stage["selector"] == "EXACT20K" and selected_step != ENDPOINT:
                raise ValueError("EXACT20K is not a completed 20K checkpoint")
            if stage["selector"] == "BEST_RRVAL_TO_20K":
                exact_id = f"{scene.scene_id}_{method}_EXACT20K"
                exact_receipt = self.root / "stage_receipts" / (exact_id + ".json")
                if exact_receipt.exists():
                    previous = load_json(exact_receipt)
                    if previous["result"].get("checkpoint_sha256") == weight_sha:
                        del model
                        return dict(status="EVALUATION_RETURNED", result=previous["result"]["result"],
                                    checkpoint_sha256=weight_sha, completed_step=selected_step,
                                    selector=stage["selector"], reused_stage_id=exact_id)
        try:
            result = self._evaluate(scene, method, model, self.root / "evaluations" / stage["id"],
                                    weight_sha=weight_sha, final=True)
            if result.get("status") != "COMPLETE":
                raise RuntimeError("Evaluator did not complete every frozen support: " + str(result.get("status")))
            return dict(status="EVALUATION_RETURNED", result=result, checkpoint_sha256=weight_sha,
                        completed_step=selected_step, selector=stage.get("selector", "FIXED"),
                        search_endpoint_updates=ENDPOINT if stage["kind"] == "final" else 0,
                        adaptation_cost_path=str(self._run_root(scene.scene_id, method) / "cost_breakdown.json")
                        if stage["kind"] == "final" else None)
        finally:
            del model

    def _inventory(self, directory):
        records = []
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                self._owned(path)
                if path.is_symlink():
                    raise ValueError("Artifact symlinks are forbidden")
                if path.is_file():
                    records.append(dict(path=str(path.relative_to(self.root)), sha256=file_sha(path)))
        return records

    def _receipt(self, stage):
        return self.root / "stage_receipts" / (stage["id"] + ".json")

    def _completed(self, stage):
        path = self._receipt(stage)
        if not path.exists():
            return False
        receipt = load_json(path)
        if (receipt.get("registry_sha256") != self.registry_sha or receipt.get("stage") != stage
                or receipt.get("state") != "COMPLETE"):
            raise ValueError("Existing stage receipt has different identity")
        for artifact in receipt["artifacts"]:
            path = self._owned(self.root / artifact["path"])
            if file_sha(path) != artifact["sha256"]:
                raise ValueError("Completed stage artifact changed")
        return True

    def _export(self, stage):
        """Small local logs only; no images, input H5, weights, or fullstates."""
        destination = self.root / "log_exports" / (stage["id"] + ".tar.gz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".export-", dir=destination.parent)
        os.close(fd)
        try:
            with tarfile.open(name, "w:gz") as archive:
                candidates = list((self.root / "stage_receipts").glob("*.json"))
                if stage["kind"] == "train":
                    run = self._run_root(stage["scene_id"], stage["method"])
                    allowed = ("training_manifest.json", "run_manifest.json", "config.resolved.json",
                               "architecture_manifest.json", "latest.json", "best_val_l1.json",
                               "exact_endpoint.json", "cost_breakdown.json", "train_steps.jsonl",
                               "probes.jsonl", "validation.jsonl", "selection_history.jsonl",
                               "audit_diagnostics.jsonl", "checkpoint_index.jsonl",
                               "resume_events.jsonl", "errors.jsonl")
                    candidates += [run / name for name in allowed if (run / name).is_file()]
                for path in sorted(candidates):
                    if path.is_symlink() or not path.is_file():
                        raise ValueError("Unexpected log export target")
                    self._owned(path)
                    archive.add(path, arcname=str(path.relative_to(self.root)), recursive=False)
            os.replace(name, destination)
        finally:
            if Path(name).exists():
                Path(name).unlink()
        return dict(path=str(destination), sha256=file_sha(destination), log_access="LOCAL_EXPORT_ONLY")

    def _publish(self):
        """A reporting outage never changes the experiment or drops its outbox."""
        try:
            from .reporting import publish_pending
            publish_pending(self.root, self.registry)
        except Exception as error:
            atomic_json(self.root / "reporting_status.json", dict(status="LOCAL_OUTBOX_RETAINED",
                        updated_utc=_now(), error_type=type(error).__name__, error=str(error)))

    @contextlib.contextmanager
    def _signals(self):
        previous = {}
        def stop(signum, frame):
            self.stop_requested, self.stop_signal = True, signum
            if self.active_engine is not None:
                self.active_engine.request_stop(signum)
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        try:
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def run(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".campaign.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another controller owns this campaign") from error
            manifest = self.root / "campaign_manifest.json"
            frozen = dict(registry_sha256=self.registry_sha, registry=self.registry)
            if manifest.exists() and load_json(manifest) != frozen:
                raise ValueError("Cannot change an existing campaign identity")
            atomic_json(manifest, frozen)
            atomic_json(self.root / "queue.json", dict(registry_sha256=self.registry_sha, stages=self.queue))
            with self._signals():
                current = None
                try:
                    for stage in self.queue:
                        current = stage
                        if self._completed(stage):
                            continue
                        if self.stop_requested:
                            break
                        self._space()
                        if file_sha(self.registry_path) != self.registry_sha:
                            raise ValueError("Registry changed during execution")
                        for relative, sha in self.registry["source_sha256"].items():
                            if file_sha(Path(self.registry["source_root"]) / relative) != sha:
                                raise ValueError("Executing source changed during the frozen campaign")
                        self._status("RUNNING", current_stage=stage)
                        print("PANPV campaign stage " + stage["id"], flush=True)
                        result = self._stage(stage)
                        if self.stop_requested:
                            break
                        if stage["kind"] == "train":
                            run_root = self._run_root(stage["scene_id"], stage["method"])
                            checkpoint = result["checkpoint"]
                            directory = self._owned(run_root / checkpoint["directory"])
                            artifacts = self._inventory(directory)
                        else:
                            directory_id = result.get("reused_stage_id", stage["id"])
                            artifacts = self._inventory(self.root / "evaluations" / directory_id)
                            if not artifacts:
                                raise RuntimeError("Evaluator returned no persistent report or outputs")
                        receipt = dict(schema="PANPV_STAGE_RECEIPT_v1", state="COMPLETE",
                                       registry_sha256=self.registry_sha, stage=stage, result=result,
                                       artifacts=artifacts, recorded_utc=_now(), sheet_uploaded=False)
                        atomic_json(self._receipt(stage), receipt)
                        exported = self._export(stage)
                        atomic_json(self.root / "outbox" / (stage["id"] + ".json"),
                                    dict(receipt=str(self._receipt(stage)), export=exported,
                                         status="LOCAL_ONLY_PENDING_DEDICATED_SHEET_WRITER",
                                         sheet="practical-validation", log_url=None))
                        self._publish()
                        gc.collect()
                        torch.cuda.empty_cache()
                    state = "PAUSED_SIGNAL" if self.stop_requested else "LOCAL_COMPUTE_COMPLETE"
                    if not self.stop_requested:
                        # Also retry a previous reporting outage when every
                        # compute stage was already complete on this resume.
                        self._publish()
                    self._status(state, current_stage=current, sheet_status="CHECK_REPORTING_OUTBOX")
                    return 0 if not self.stop_requested else 130
                except Exception as error:
                    self._status("ERROR_STOPPED", current_stage=current,
                                 error_type=type(error).__name__, error=str(error))
                    raise
                finally:
                    self.datasets.clear()
                    for scene in self.scenes.values():
                        scene.close()
                    gc.collect()
                    torch.cuda.empty_cache()


def run_campaign(registry_path):
    return Campaign(registry_path).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        registry = load_json(args.registry)
        validate_registry(registry)
        print(json.dumps(dict(status="REGISTRY_VALIDATED_NOT_STARTED", registry_sha256=file_sha(args.registry))))
        return 0
    return run_campaign(args.registry)


if __name__ == "__main__":
    raise SystemExit(main())
