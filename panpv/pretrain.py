"""WV3-only replacement donor, isolated from all PANPV/WV2 observations.

Reuses the deployment model, data recipe, sampler, optimizer, and resumable
trainer. The sole donor selector is the fixed 50,000-update endpoint; validation
is diagnostic and cannot substitute another checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time

import h5py
import numpy as np
import torch

from panpv.common import atomic_json, canonical_sha, file_sha, load_json
from deploy.pandep_shared.project.src.pan_shared.data import ensure_lp_cache, scan_source
from deploy.pandep_shared.project.src.pan_shared.frontend import (
    AUGMENTATION, LP_RECIPE, SENSOR_PROFILES, fixed_view, normalize_dn,
)
from deploy.pandep_shared.project.src.pan_shared.model import SingleSensorPLHUNet
from deploy.pandep_shared.project.src.pan_shared.train import Trainer


RUN_ID = "PANPV_DONOR_WV3_ONLY_W104_D122_PLH_NOA_NOKD_SE271001_EXACT50000_v1"
SOURCE_SHA = {
    "train": "f9e7ec381390ff36ddffd0714df7af07a3c5f456078b219a25f3e18372228f82",
    "val": "ba883e57408d1f3c2263cc2818a9fbbf2dbe8e62554c531288f9e8bf9f59dc87",
}
MIN_FREE_BYTES = 20 * 1024 ** 3
TOTAL_UPDATES = 50000
CANDIDATE_REVIEW = [
    {"family": "FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101..73106_BASE_FRESH50_v1",
     "rejected": True, "reason": "BASE denotes optimizer profile, not supervised-only: teacher-cloned trainable aligner and hard+soft+edge KD",
     "evidence": "fh20r1/training.py; FH20R1 SS73101 diagnostics/loss_routing.json"},
    {"family": "PANDA_REBUTTAL_B01 and B02_M12 WV3 W104 D122",
     "rejected": True, "reason": "Teacher/alignment/KD family, not deployment noA supervised donor",
     "evidence": "work_dir/_panda_rb/20261001/B02_M12/train/s1/R1/QFULL/meta/config.resolved.yaml"},
    {"family": "QG40 QB and G20 GF2 W104 D122 PLH",
     "rejected": True, "reason": "Four-band non-WV3 source and teacher/alignment/KD training"},
]


def _owned(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("Pretrain artifact path escapes its owned root")
    return path


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if load_json(path) != value:
            raise ValueError("Pretrain immutable identity changed: " + str(path))
    else:
        atomic_json(path, value)


@contextmanager
def exclusive(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = _owned(root, "pretrain.lock")
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another donor pretrain owns this directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_sources(repo_root, work_root):
    """SHA-bound WV3 train/val only; all source H5 opens are read-only."""
    repo_root, work_root = Path(repo_root).resolve(), Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work_root).free < MIN_FREE_BYTES:
        raise OSError("PAUSED_DISK: donor preparation needs a 20 GiB reserve")
    catalog_path = repo_root / "deploy/pandep_shared/project/vendor_reference/model_source/ablr2/sensor_sources.json"
    catalog = load_json(catalog_path)["sensors"]["WV3"]
    if catalog["band_order"] != SENSOR_PROFILES["WV3"]["band_order"]:
        raise ValueError("Frozen WV3 source band order mismatch")
    profile = dict(SENSOR_PROFILES["WV3"], splits={})
    for split in ("train", "val"):
        binding = catalog["splits"][split]
        if binding["sha256"] != SOURCE_SHA[split]:
            raise ValueError("Frozen WV3 train/validation source SHA changed")
        relative = Path(binding["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:3] != ("data", "PanCollection", "WV3"):
            raise ValueError("WV3 catalog must name a repository WV3 data binding")
        # Repository H5 bindings intentionally symlink to installed CANConv
        # sources. Their exact bytes, not the target directory spelling, bind them.
        path = (repo_root / relative).resolve(strict=True)
        if path.is_relative_to(work_root) or work_root.is_relative_to(path):
            raise ValueError("Source and pretrain output must be separate")
        if file_sha(path) != SOURCE_SHA[split]:
            raise ValueError("WV3 source byte SHA mismatch: " + split)
        scan = scan_source(path, "WV3", split)
        cache = _owned(work_root, "cache/" + split + "_lpan.h5")
        lp = ensure_lp_cache(path, cache, SOURCE_SHA[split], allowed_root=work_root)
        profile["splits"][split] = dict(
            dataroot=str(path), sha256=SOURCE_SHA[split],
            source_identity=binding["source_identity"], **scan, **lp,
        )
    manifest = dict(
        schema="PANPV_WV3_DONOR_DATA_v1", sensors={"WV3": profile},
        catalog_sha256=file_sha(catalog_path), recipe=LP_RECIPE,
        augmentation=AUGMENTATION, source_files_read_only=True,
        wv2_inputs_opened=False, test_inputs_opened=False,
    )
    manifest["dataset_sha256"] = canonical_sha(manifest)
    _immutable_json(_owned(work_root, "data_manifest.json"), manifest)
    return manifest


class RAMDataset:
    """Verified normalized WV3 arrays, once per launch, without H5 per update."""
    def __init__(self, manifest, split):
        if split not in ("train", "val"):
            raise ValueError("Donor data is restricted to WV3 train/val")
        self.split = split
        self.binding = manifest["sensors"]["WV3"]["splits"][split]
        self.source = Path(self.binding["dataroot"]).resolve(strict=True)
        self.lp = Path(self.binding["lpan_path"]).resolve(strict=True)
        for path, digest in ((self.source, self.binding["sha256"]),
                             (self.lp, self.binding["lpan_sha256"])):
            if file_sha(path) != digest:
                raise ValueError("Donor data bytes changed")
        before = self._signatures()
        with h5py.File(self.source, "r") as source, h5py.File(self.lp, "r") as cache:
            self.arrays = {key: normalize_dn(source[key][...], "WV3")
                           for key in ("pan", "ms", "gt")}
            self.arrays["lpan"] = normalize_dn(cache["lpan"][...], "WV3")
        self.count = int(self.binding["count"])
        shapes = {"pan": (self.count, 1, 64, 64), "ms": (self.count, 8, 16, 16),
                  "gt": (self.count, 8, 64, 64), "lpan": (self.count, 1, 16, 16)}
        if any(value.shape != shapes[key] or not np.isfinite(value).all()
               for key, value in self.arrays.items()):
            raise ValueError("Donor RAM arrays have invalid shape/values")
        self._stats = self._signatures()
        if before != self._stats:
            raise ValueError("Donor source changed while loading RAM")
        for value in self.arrays.values():
            value.setflags(write=False)

    def _signatures(self):
        return [(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
                for p in (self.source, self.lp)]

    def __len__(self):
        return self.count

    def fetch(self, sample_ids, rotations=None):
        if self._stats != self._signatures():
            raise ValueError("Donor source identity changed during training")
        if not sample_ids or any(isinstance(i, (bool, np.bool_)) or
                                not isinstance(i, (int, np.integer)) or
                                not 0 <= int(i) < self.count for i in sample_ids):
            raise IndexError("Invalid WV3 sample IDs")
        if rotations is not None and (self.split != "train" or len(rotations) != len(sample_ids)):
            raise ValueError("Explicit training rotations only")
        ids = np.asarray(sample_ids, dtype=np.int64)
        result = {}
        for key, source in self.arrays.items():
            batch = source[ids].copy()
            if rotations is not None:
                batch = np.stack([fixed_view(x, rotation) for x, rotation in zip(batch, rotations)])
            result[key] = torch.from_numpy(batch)
        return result


def validation_callback(dataset):
    """Full validation for events; fixed first 128 for progress-only probes."""
    def evaluate(trainer, kind):
        if kind not in ("probe", "validation"):
            raise ValueError("Unknown donor validation event")
        count = len(dataset) if kind == "validation" else min(128, len(dataset))
        total, elements = 0.0, 0
        for start in range(0, count, trainer.microbatch):
            batch = {key: value.to(trainer.device) for key, value in
                     dataset.fetch(list(range(start, min(count, start + trainer.microbatch)))).items()}
            pred = trainer.model("WV3", batch["pan"], batch["ms"], batch["lpan"])
            error = (pred - batch["gt"]).abs()
            if not bool(torch.isfinite(error).all()):
                raise FloatingPointError("Nonfinite donor validation error")
            total += float(error.double().sum())
            elements += error.numel()
        return {"complete": kind == "validation", "sensors": {"WV3": {
            "mean_l1": total / elements, "n_samples": count,
            "HQNR": None, "SCC": None, "ERGAS": None,
            "metric_scope": "WV3 normalized unclipped validation L1; not benchmark metrics",
        }}, "wv2_inputs_opened": False}
    return evaluate


def no_test_benchmark(trainer, checkpoint, selector):
    """Trainer requires a benchmark hook; do not fake uncomputed RR/FR metrics."""
    return {"status": "NOT_EVALUATED_BY_DESIGN", "scope": "DONOR_PRETRAIN_ONLY",
            "reason": "Fixed 50000 endpoint donor; no WV2 or test-set selection",
            "selector": selector, "HQNR": None, "SCC": None, "ERGAS": None}


def source_identity():
    package = Path(__file__).resolve().parents[1]
    files = [Path(__file__), package / "panpv/common.py"]
    shared = package / "deploy/pandep_shared/project/src/pan_shared"
    files += [shared / (name + ".py") for name in
              ("train", "checkpoints", "sampling", "data", "model", "blocks", "frontend", "losses", "common")]
    return {str(path.relative_to(package)): file_sha(path) for path in files}


def publish_donor(trainer, work_root, manifest, identity):
    if trainer.step != TOTAL_UPDATES or trainer.state["pending_events"]:
        raise ValueError("Only completed EXACT50000 donor may be published")
    checkpoint_dir = trainer.run_dir / "checkpoints/step_000050000"
    checkpoint = load_json(checkpoint_dir / "identity.json")
    trainer.store.load_weights(trainer.model, checkpoint)
    if checkpoint["completed_step"] != TOTAL_UPDATES:
        raise ValueError("Donor endpoint mismatch")
    result = dict(
        schema="PANPV_DONOR_VERIFIED_v1", status="READY", donor_run_id=RUN_ID,
        selector="FIXED_EXACT50000", completed_step=TOTAL_UPDATES,
        model_path=checkpoint["model_path"], checkpoint_sha256=checkpoint["checkpoint_sha256"],
        path=checkpoint["model_path"], sha256=checkpoint["checkpoint_sha256"],
        tensor_state_sha256=checkpoint["tensor_state_sha256"], expected_sha256=checkpoint["checkpoint_sha256"],
        architecture=trainer.model.architecture_manifest(), config=trainer.config,
        donor_sensor="WV3", target_sensor="WV2", donor_pretraining_only_sensor="WV3",
        target_data_used_for_selection=False, test_data_used_for_selection=False,
        original_s2_donor_reproduced=False,
        local_candidate_review=CANDIDATE_REVIEW,
        replacement_reason="Original S2 donor unavailable; user authorized local equivalent or fresh WV3 pretraining",
        original_s2_sha256="6ad7cff8fc60542ab687a405f7b55430926942f008a65345de99fa69fa8b3846",
        dataset_sha256=manifest["dataset_sha256"], identity=identity,
    )
    _immutable_json(_owned(work_root, "donor_manifest.json"), result)
    return dict(result, provenance_path=str(_owned(work_root, "donor_manifest.json")),
                provenance_sha256=file_sha(_owned(work_root, "donor_manifest.json")))


def launch_smoke(dataset, work_root, identity, *, device, micro_batch):
    """One disposable real-WV3 update; never used as a pretraining checkpoint."""
    proof_path = _owned(work_root, "preflight/training_smoke.json")
    proof_identity = dict(identity=identity, device=device, micro_batch=micro_batch)
    if proof_path.exists():
        proof = load_json(proof_path)
        if proof.get("identity") == proof_identity and proof.get("passed") is True:
            return proof
        raise ValueError("Donor smoke provenance changed; use a new work directory")
    model = SingleSensorPLHUNet("WV3", width=104, depth=(1, 2, 2), seed=271001)
    config = dict(mode="SINGLE", sensors=["WV3"], seed=271001,
                  microbatch=micro_batch, effective_batch_size=48)
    trainer = Trainer(model, {"WV3": dataset}, config, {"smoke_only": proof_identity},
                      _owned(work_root, "preflight/disposable_smoke"),
                      device=device, microbatch=micro_batch)
    if trainer.step:
        raise ValueError("Disposable smoke must not resume into formal donor")
    result = trainer.update()
    if trainer.step != 1 or not np.isfinite(result["loss"]) or result["grad_norms"]["head"] <= 0:
        raise ValueError("Donor one-update smoke failed")
    proof = dict(identity=proof_identity, passed=True, completed_updates=1,
                 effective_batch_size=48, micro_batch=micro_batch,
                 loss=result["loss"], grad_norms=result["grad_norms"],
                 disposable_weights_not_reused=True, wv2_data_used=False)
    atomic_json(proof_path, proof)
    return proof


def run(work_root, repo_root, *, device="cuda", micro_batch=12, prepare_only=False):
    work_root, repo_root = Path(work_root).resolve(), Path(repo_root).resolve()
    if micro_batch not in (12, 24, 48):
        raise ValueError("Microbatch must be 12, 24, or 48 with effective batch 48")
    if work_root == repo_root or not work_root.is_relative_to(repo_root / "work_dir"):
        raise ValueError("Fresh donor output must be an independent repository work_dir child")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    with exclusive(work_root):
        manifest = prepare_sources(repo_root, work_root)
        _immutable_json(_owned(work_root, "candidate_review.json"), dict(
            schema="PANPV_DONOR_LOCAL_REVIEW_v1", rows=CANDIDATE_REVIEW,
            decision="FRESH_WV3_SUPERVISED_NOA_NOKD_EXACT50000", target_metrics_used=False))
        if prepare_only:
            return {"status": "DATA_READY", "dataset_sha256": manifest["dataset_sha256"]}
        config = dict(run_id=RUN_ID, mode="SINGLE", sensors=["WV3"], width=104,
                      depth=[1, 2, 2], seed=271001, effective_batch_size=48,
                      microbatch=micro_batch, train_fraction=1.0, updates=TOTAL_UPDATES,
                      donor_selector="FIXED_EXACT50000")
        identity = dict(schema="PANPV_DONOR_PRETRAIN_IDENTITY_v1", run_id=RUN_ID,
                        config_sha256=canonical_sha(config), dataset_sha256=manifest["dataset_sha256"],
                        source_files=source_identity(), torch_version=torch.__version__,
                        cuda_version=torch.version.cuda, device_type=torch.device(device).type)
        _immutable_json(_owned(work_root, "identity.json"), identity)
        _immutable_json(_owned(work_root, "config.json"), config)
        datasets = {split: RAMDataset(manifest, split) for split in ("train", "val")}
        launch_smoke(datasets["train"], work_root, identity, device=device, micro_batch=micro_batch)
        model = SingleSensorPLHUNet("WV3", width=104, depth=(1, 2, 2), seed=271001)
        trainer = Trainer(model, {"WV3": datasets["train"]}, config, identity,
                          _owned(work_root, "run"), device=device, microbatch=micro_batch,
                          evaluate=validation_callback(datasets["val"]), benchmark=no_test_benchmark)
        status_path = _owned(work_root, "status.json")
        def control():
            if stop["requested"]:
                return "PAUSE"
            if trainer.step % 100 == 0:
                if shutil.disk_usage(work_root).free < MIN_FREE_BYTES:
                    stop["requested"] = True
                    return "PAUSE"
                atomic_json(status_path, dict(status="RUNNING", run_id=RUN_ID,
                                              completed_step=trainer.step, total_updates=TOTAL_UPDATES,
                                              updated_unix=time.time(), pid=os.getpid()))
            return "RUN"
        handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            outcome = trainer.run_until(TOTAL_UPDATES, control_check=control)
            # A stop can arrive during the final update/pending validation,
            # after the trainer's last control check. Never auto-chain WV2
            # despite that request; the completed checkpoint remains resumable.
            if stop["requested"]:
                outcome = "PAUSED_SAFE"
            atomic_json(status_path, dict(status=outcome, run_id=RUN_ID,
                                          completed_step=trainer.step, total_updates=TOTAL_UPDATES,
                                          updated_unix=time.time(), pid=os.getpid()))
            if trainer.step == TOTAL_UPDATES and outcome == "BLOCK_COMPLETE":
                return publish_donor(trainer, work_root, manifest, identity)
            return {"status": outcome, "completed_step": trainer.step}
        except BaseException as error:
            atomic_json(status_path, dict(status="FAILED", run_id=RUN_ID,
                                          completed_step=trainer.step, error_type=type(error).__name__,
                                          error=str(error), updated_unix=time.time()))
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def run_pretrain(repo_root, work_root, device="cuda", micro_batch=12):
    """Main-chain API: return fixed-endpoint donor path/SHA/provenance or pause."""
    return run(work_root, repo_root, device=device, micro_batch=micro_batch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch", type=int, default=12)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.work_root, args.repo_root, device=args.device,
                 micro_batch=args.micro_batch, prepare_only=args.prepare_only)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["status"] in ("READY", "DATA_READY", "PAUSED_SAFE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
