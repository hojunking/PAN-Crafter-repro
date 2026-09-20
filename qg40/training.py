"""Bounded QG40 FP32 trainer with exact-state resume and explicit profiles.

Admission belongs to the campaign controller. This trainer also checks the
immutable common window and stops only at completed optimizer boundaries.
"""
from __future__ import annotations

import copy
import datetime as dt
import math
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import time

import numpy as np
import torch
import yaml
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from fh12.training import (BatchStream, atomic_torch, restore_rng,
                           save_resume_bundle, tensor_cpu_tree)
from qg40.common import (ROOT, atomic_json, before_deadline, check_deadline, object_sha,
                         read_json, resolved_path, sha256, source_identity, utcnow)
from qg40.losses import routed_student_backward, student_losses, teacher_loss
from qg40.model import build_model, state_hash
from qg40.plan import CAMPAIGN_ID, GRID_STEPS, FULLSTATE_STEPS, build_config, case_for


def rng_state():
    """Capture active RNGs without initializing a GPU for CPU diagnostics."""
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def cosine_factor(completed_updates, warmup=100, total=50000):
    """FH12 diffusers cosine factor, indexed before the next optimizer update."""
    if completed_updates < warmup:
        return float(completed_updates) / max(1, warmup)
    progress = float(completed_updates - warmup) / max(1, total - warmup)
    return max(0., .5 * (1. + math.cos(math.pi * progress)))


def aligner_factor(completed_updates, profile):
    if profile not in ("BASE", "A24R", "B20", "E10", "C3"):
        raise ValueError("Unregistered QG40 schedule profile")
    return 1. / 3. if profile == "A24R" and completed_updates >= 24240 else 1.


def make_scheduler(optimizer, profile, warmup=100, total=50000):
    if [group.get("name") for group in optimizer.param_groups] != ["U", "A"]:
        raise ValueError("Scheduler requires explicit, independent U and A groups")
    aligner_factor(0, profile)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [
        lambda t: cosine_factor(t, warmup, total),
        lambda t: cosine_factor(t, warmup, total) * aligner_factor(t, profile)])


def validate_config(cfg):
    """Compare all recipe values; permit only actual asset/path binding fields."""
    case = case_for(Path(cfg["work_dir"]).name)
    expected = build_config(case)
    bound = copy.deepcopy(cfg)
    expected["work_dir"] = cfg["work_dir"]
    bindings = ("dataset_manifest", "reference_manifest", "sensor_spec", "sensor_spec_path",
                "teacher_checkpoint", "teacher_sha256", "tau_R", "q_ref", "q_cache", "q_cache_sha256")
    for key in bindings:
        expected["qg40"][key] = bound["qg40"][key]
    for key in ("train_feeder_args", "val_feeder_args", "test_reduced_feeder_args", "test_full_feeder_args"):
        expected[key]["dataroot"] = bound[key]["dataroot"]
    if bound != expected:
        raise ValueError("QG40 config differs from its registered immutable numerical recipe")
    if not bound["qg40"]["dataset_manifest"]:
        raise ValueError("Bind the verified local sensor dataset before training")
    return case


def runtime_context(cfg, root, deadline_arg=None, *, resume=False):
    window = read_json(resolved_path(cfg["qg40"]["window_path"], root))
    if window.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("QG40 requires its own common campaign window")
    start = dt.datetime.fromisoformat(window["t0_utc"].replace("Z", "+00:00"))
    end = dt.datetime.fromisoformat(window["deadline_utc"].replace("Z", "+00:00"))
    if start.tzinfo is None or end.tzinfo is None or end - start != dt.timedelta(hours=40):
        raise ValueError("QG40 must use the immutable common t0+40h deadline")
    if deadline_arg and deadline_arg != window["deadline_utc"]:
        raise ValueError("QG40 deadline cannot be overridden or restarted")
    now = dt.datetime.now(dt.timezone.utc)
    if now < start:
        raise ValueError("QG40 common window has not started")
    if not resume and now >= start + dt.timedelta(hours=36):
        raise TimeoutError("No new training after the common 36h admission cutoff")
    return window, window["deadline_utc"]


def train_run(config_path, device="cuda", resume=False, deadline_arg=None, root=ROOT):
    from qg40.data import build_dataset
    from qg40.evaluation import FRMetrics, evaluate_model
    from qg40.references import load_reference

    root = Path(root)
    cfg = yaml.safe_load(Path(config_path).read_text())
    case = validate_config(cfg)
    window, deadline = runtime_context(cfg, root, deadline_arg, resume=resume)
    check_deadline(deadline)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no implicit CPU training fallback")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    f = cfg["qg40"]
    wd = resolved_path(cfg["work_dir"], root)
    config_sha, release = object_sha(cfg), source_identity(root)
    data = read_json(resolved_path(f["dataset_manifest"], root))
    data_sha = object_sha(data)
    if data["sensor"] != case.sensor or data["num_bands"] != 4 or data["max_pixel"] != cfg["max_pixel"]:
        raise ValueError("Bound dataset differs from registered sensor/C/DN")
    datasets = {split: build_dataset(data, split, root=root) for split in ("train", "val", "rr", "fr")}
    if len(datasets["train"]) != datasets["train"].base_count:
        raise ValueError("Training sampler needs unique base IDs and explicit ROT4 draws")
    teacher = reference = q = None
    if case.role == "S":
        teacher, reference, q = load_reference(case.reference_id, case.server_id, root, str(dev),
                                               manifest_path=(resolved_path(f["reference_manifest"], root)
                                                              if f["reference_manifest"] else None),
                                               dataset_manifest=data, deadline=deadline)
        teacher.eval().requires_grad_(False)
        q = torch.as_tensor(q, dtype=torch.float32, device=dev)
        if q.shape != (len(datasets["train"]), 4) or not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
            raise ValueError("q cache must cover every original train ID and all four views")
        for key in ("tau_R", "q_ref"):
            if not math.isfinite(float(reference[key])) or float(reference[key]) <= 0:
                raise ValueError(f"Reference has unmeasured or invalid {key}")
        if reference["teacher_run_id"] != case.teacher_run_id or reference["sensor"] != case.sensor:
            raise ValueError("Student reference lineage/sensor mismatch")
    model, init = build_model(case.input_layout, case.width, list(case.depth), case.seed, role=case.role,
                              teacher_aligner_state=teacher.aligner.state_dict() if teacher is not None else None)
    if teacher is not None and set(map(id, model.parameters())) & set(map(id, teacher.parameters())):
        raise ValueError("Student and frozen Teacher cannot share parameters")
    model.to(dev).train()
    teacher_hash = state_hash(teacher.state_dict()) if teacher is not None else None
    reference_sha = object_sha(reference) if reference is not None else None
    optimizer = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=cfg["learning_rate"], name="U"),
        dict(params=model.aligner.parameters(), lr=f["aligner_lr"], name="A")],
        betas=tuple(cfg["betas"]), eps=cfg["eps"], weight_decay=cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, case.profile, cfg["num_warmup"], cfg["num_iter"])
    stream = BatchStream(len(datasets["train"]), cfg["batch_size"], cfg["seed"])
    corruption = torch.Generator(device="cpu").manual_seed(f["corruption_seed"])
    update, training_seconds, evaluation_seconds, io_seconds, diagnostic_seconds = 0, 0., 0., 0., 0.
    prior_peak_memory_bytes = 0
    exposure_counts = torch.zeros((len(datasets['train']), 4), dtype=torch.int64)
    records = []
    expected_identity = dict(config_sha256=config_sha, data_sha256=data_sha,
                             source_identity=release, reference_sha256=reference_sha)
    state_path = wd / "last/training_state.pt"
    if resume:
        identity = read_json(wd / "last/identity.json")
        if identity.get("training_state_sha256") != sha256(state_path):
            raise ValueError("Full-state resume checksum mismatch")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if not state.get("full_state"):
            raise ValueError("Weights-only candidate is not an exact resume")
        for key, value in expected_identity.items():
            if state.get(key) != value:
                raise ValueError(f"Exact resume identity mismatch: {key}")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        stream.load_state_dict(state["sampler"]); corruption.set_state(state["corruption_rng"])
        update = int(state["update"])
        from qg40.exposure import validate_counts
        exposure_counts = validate_counts(state.get('exposure_counts'), len(datasets['train']), update, cfg['batch_size'])
        training_seconds, evaluation_seconds = state["training_seconds"], state["evaluation_seconds"]
        io_seconds = state.get("io_seconds", 0.)
        diagnostic_seconds = state.get("diagnostic_seconds", 0.)
        prior_peak_memory_bytes = state.get("peak_memory_bytes") or 0
        if scheduler.last_epoch != update or not 0 <= update <= 50000:
            raise ValueError("Resume update/scheduler state mismatch")
        grid = wd / "official/raw_grid.json"
        records = read_json(grid)["records"] if grid.exists() else []
        if any(r["update"] > update for r in records):
            raise ValueError("Evaluation grid is ahead of the saved optimizer state")
        restore_rng(state["rng"])
    elif state_path.exists() or (wd / "meta/training_start_manifest.json").exists():
        raise ValueError("Run already exists; exact resume required")
    else:
        (wd / "meta").mkdir(parents=True, exist_ok=True)
        (wd / "meta/config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        atomic_json(wd / "init_manifest.json", dict(init, reference_sha256=reference_sha))
        atomic_json(wd / "meta/training_start_manifest.json", dict(campaign_id=CAMPAIGN_ID,
                    run_id=case.run_id, **expected_identity, window=window, device=str(dev),
                    precision="fp32", started_at_utc=utcnow(), sampler_hash=state_hash({
                        "order": stream.order, "rotations": stream.rotations}),
                    rng_roles=dict(data_order=cfg["seed"] + 300000, augmentation=cfg["seed"] + 400000,
                                   corruption=f["corruption_seed"], workers=cfg["seed"] + 500000)))
    pause = [False]
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in handlers:
        signal.signal(sig, lambda *_: pause.__setitem__(0, True))
    engine = None

    def peak_memory():
        return max(prior_peak_memory_bytes, torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None

    def full_state():
        return dict(full_state=True, update=update, model_state=tensor_cpu_tree(model.state_dict()),
                    optimizer=tensor_cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                    scaler={}, precision="fp32", rng=rng_state(), sampler=stream.state_dict(),
                    corruption_rng=corruption.get_state(), **expected_identity,
                    training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                    diagnostic_seconds=diagnostic_seconds, io_seconds=io_seconds,
                    peak_memory_bytes=peak_memory(), exposure_counts=exposure_counts.clone())

    def status(name):
        from qg40.exposure import exposure_report
        atomic_json(wd / "meta/training_status.json", dict(status=name, actual_updates=update,
                    training_complete=update == 50000, n_evaluated=len(records),
                    pending_steps=[s for s in GRID_STEPS if s not in {r["update"] for r in records}],
                    training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                    diagnostic_seconds=diagnostic_seconds, io_seconds=io_seconds,
                    peak_memory_bytes=peak_memory(), peak_memory_scope="training+scheduled_eval+diagnostics",
                    sample_exposure=exposure_report(exposure_counts, cfg['batch_size'], stream.epoch, stream.cursor),
                    deadline_utc=deadline, updated_at_utc=utcnow()))

    def save_resume():
        nonlocal io_seconds
        started = time.monotonic()
        save_resume_bundle(wd, full_state(), dict(update=update, full_state=True, **expected_identity))
        io_seconds += time.monotonic() - started

    def save_candidate(destination, include_state):
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = state_hash(model.state_dict())
        if destination.exists():
            identity = read_json(destination / "identity.json")
            if (identity.get("state_hash") != digest or identity.get("config_sha256") != config_sha
                    or identity.get("model_sha256") != sha256(destination / "model.safetensors")):
                raise ValueError("Existing candidate differs; refusing overwrite")
            return identity
        pending = Path(tempfile.mkdtemp(prefix=f".{update}-", dir=destination.parent))
        try:
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                      str(pending / "model.safetensors"))
            identity = dict(update=update, role=case.role, sensor=case.sensor, num_bands=4,
                input_layout=case.input_layout, **expected_identity, full_state=bool(include_state),
                state_hash=digest, model_sha256=sha256(pending / "model.safetensors"))
            snapshot = full_state() if include_state else dict(full_state=False, update=update)
            snapshot["model_sha256"] = identity["model_sha256"]
            atomic_torch(pending / "training_state.pt", snapshot)
            identity["training_state_sha256"] = sha256(pending / "training_state.pt")
            atomic_json(pending / "identity.json", identity)
            os.replace(pending, destination)
            return identity
        finally:
            if pending.exists():
                shutil.rmtree(pending)

    def evaluate_update():
        nonlocal engine, evaluation_seconds
        if update not in GRID_STEPS or any(r["update"] == update for r in records):
            return
        identity = save_candidate(wd / "candidates" / str(update), update in FULLSTATE_STEPS)
        check_deadline(deadline)
        training_rng = rng_state()
        try:
            if engine is None:
                engine = FRMetrics(datasets["fr"])
            result = evaluate_model(model, datasets, dev, engine, deadline, include_q=True, with_val=True)
        finally:
            restore_rng(training_rng)
        if teacher is not None and state_hash(teacher.state_dict()) != teacher_hash:
            raise ValueError("Frozen Teacher changed during training")
        evaluation_seconds += result["seconds"]
        record = dict(update=update, checkpoint_identity=identity, **result)
        records.append(record)
        atomic_json(wd / "official" / f"candidate_{update}.json", record)
        atomic_json(wd / "official/raw_grid.json", dict(campaign_id=CAMPAIGN_ID, run_id=case.run_id,
            sensor=case.sensor, expected_steps=list(GRID_STEPS), records=records,
            n_evaluated=len(records), complete=[r["update"] for r in records] == list(GRID_STEPS),
            **expected_identity))
        print(f'[QG40 {case.sensor} official update={update}] '
              f'rawHQNR={result["fr"]["hqnr"]:.8f} SCC={result["rr"]["scc"]:.8f} '
              f'ERGAS={result["rr"]["ergas"]:.8f} valERGAS={result["val_ergas"]:.8f}', flush=True)
        save_resume()

    def save_boundary():
        nonlocal diagnostic_seconds
        save_resume()
        if update in GRID_STEPS:
            save_candidate(wd / "candidates" / str(update), update in FULLSTATE_STEPS)
        if update in FULLSTATE_STEPS:
            save_candidate(wd / "restart_fullstates" / str(update), True)
            from qg40.p1 import record_training_diagnostics
            diagnostic_seconds += record_training_diagnostics(wd, model, teacher, datasets,
                update=update, device=dev, profile=case.profile, reference=reference or {},
                q_cache=q, consistency_weight=f['consistency_weight'], identity=expected_identity,
                deadline=deadline)
            save_resume()
        evaluate_update()

    try:
        status("RUNNING")
        save_boundary()
        while update < 50000:
            if pause[0] or not before_deadline(deadline):
                save_resume()
                status("INCOMPLETE_AT_DEADLINE" if not before_deadline(deadline) else "PAUSED_SIGNAL")
                return 75
            if stream.cursor == stream.count:
                stream.new_epoch()
            epoch_rng = torch.Generator().manual_seed(cfg["seed"] + 500000 + stream.epoch)
            loader = DataLoader(datasets["train"], batch_sampler=stream.remaining_batches(),
                num_workers=cfg["num_worker"], pin_memory=dev.type == "cuda", generator=epoch_rng)
            for gt, _lms, ms, lp, pan, meta in loader:
                if pause[0] or not before_deadline(deadline):
                    save_resume()
                    status("INCOMPLETE_AT_DEADLINE" if not before_deadline(deadline) else "PAUSED_SIGNAL")
                    return 75
                started = time.monotonic()
                model.train()
                gt, ms, lp, pan = [x.to(dev, non_blocking=True) for x in (gt, ms, lp, pan)]
                optimizer.zero_grad(set_to_none=True)
                out = model(pan, ms, lp)
                if teacher is None:
                    losses = teacher_loss(model, out, gt, pan, ms, update, corruption,
                                          consistency_weight=f["consistency_weight"])
                    if not bool(torch.isfinite(losses["total"])):
                        raise FloatingPointError("Nonfinite Teacher objective")
                    losses["total"].backward()
                else:
                    with torch.no_grad():
                        teacher_out = teacher(pan, ms, lp)
                    raw_q = q[meta[:, 0].to(dev), meta[:, 1].to(dev)]
                    weights = (reference["q_ref"] / (reference["q_ref"] + raw_q)).detach()
                    losses = student_losses(out, teacher_out, gt, reference["tau_R"], weights, profile=case.profile)
                    if not bool(torch.isfinite(losses["L_U"]) & torch.isfinite(losses["L_A"])):
                        raise FloatingPointError("Nonfinite Student objective")
                    routed_student_backward(model, losses)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise FloatingPointError("Missing/nonfinite QG40 gradients")
                optimizer.step(); scheduler.step(); update += 1; stream.cursor += 1
                from qg40.exposure import record_batch
                record_batch(exposure_counts, meta)
                training_seconds += time.monotonic() - started
                if update in GRID_STEPS or update in FULLSTATE_STEPS:
                    save_boundary(); status("RUNNING")
                if update % cfg["log_iter"] == 0:
                    print(f"[QG40 {case.role}/{case.profile}] update={update}/50000", flush=True)
                if update == 50000:
                    break
        save_resume()
        status("OFFICIAL_EVAL_COMPLETE" if len(records) == 50 else "TRAIN50K_COMPLETE")
        return 0
    except TimeoutError:
        save_resume()
        status("INCOMPLETE_AT_DEADLINE" if update < 50000 else "TRAIN50K_COMPLETE")
        return 75
    except Exception:
        status("BLOCKED_INTEGRITY")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
