"""Bounded G20 FP32 trainer with exact-state resume and explicit profiles.

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
from g20.common import (ROOT, atomic_json, before_deadline, check_deadline, object_sha,
                         read_json, resolved_path, sha256, source_identity, utcnow)
from g20.losses import routed_student_backward, student_losses, teacher_loss
from g20.model import build_model, state_hash
from g20.plan import CAMPAIGN_ID, GRID_STEPS, FULLSTATE_STEPS, build_config, case_from_config


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
    if profile not in ("BASE", "A1", "A9", "E1", "E4", "K05", "K20", "C030", "C100", "C300"):
        raise ValueError("Unregistered G20 schedule profile")
    return 1.


def make_scheduler(optimizer, profile, warmup=100, total=50000):
    if [group.get("name") for group in optimizer.param_groups] != ["U", "A"]:
        raise ValueError("Scheduler requires explicit, independent U and A groups")
    aligner_factor(0, profile)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [
        lambda t: cosine_factor(t, warmup, total),
        lambda t: cosine_factor(t, warmup, total) * aligner_factor(t, profile)])


def validate_config(cfg):
    """Compare all recipe values; permit only actual asset/path binding fields."""
    case = case_from_config(cfg)
    expected = build_config(case)
    bound = copy.deepcopy(cfg)
    expected["work_dir"] = cfg["work_dir"]
    bindings = ("dataset_manifest", "reference_manifest", "sensor_spec", "sensor_spec_path",
                "teacher_checkpoint", "teacher_sha256", "tau_R", "q_ref", "q_cache", "q_cache_sha256")
    for key in bindings:
        expected["g20"][key] = bound["g20"][key]
    for key in ("train_feeder_args", "val_feeder_args", "test_reduced_feeder_args", "test_full_feeder_args"):
        expected[key]["dataroot"] = bound[key]["dataroot"]
    if bound != expected:
        raise ValueError("G20 config differs from its registered immutable numerical recipe")
    if not bound["g20"]["dataset_manifest"]:
        raise ValueError("Bind the verified local sensor dataset before training")
    return case


def runtime_context(cfg, root, deadline_arg=None, *, resume=False):
    window = read_json(resolved_path(cfg["g20"]["window_path"], root))
    if window.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("G20 requires its own common campaign window")
    start = dt.datetime.fromisoformat(window["t0_utc"].replace("Z", "+00:00"))
    end = dt.datetime.fromisoformat(window["deadline_utc"].replace("Z", "+00:00"))
    if start.tzinfo is None or end.tzinfo is None or end - start != dt.timedelta(hours=20):
        raise ValueError("G20 must use the immutable common t0+20h deadline")
    if deadline_arg and deadline_arg != window["deadline_utc"]:
        raise ValueError("G20 deadline cannot be overridden or restarted")
    now = dt.datetime.now(dt.timezone.utc)
    if now < start:
        raise ValueError("G20 common window has not started")
    if not resume and now >= start + dt.timedelta(hours=16):
        # §10.2 permits completion of already reserved pairs after 16h. A
        # fresh block is forbidden, but its next case is not a new admission.
        from g20.common import camp, read
        field = cfg['g20']
        block = read(camp(root, field['server_id']) / 'status.json').get('blocks', {}).get(field['block_id'], {})
        receipt = block.get('reservation', {})
        admitted = dt.datetime.fromisoformat(receipt.get('at_utc', window['deadline_utc']).replace('Z', '+00:00'))
        if (block.get('status') != 'ADMITTED' or receipt.get('allowed') is not True
                or cfg['g20']['run_id'] not in receipt.get('run_ids', [])
                or not start <= admitted < start + dt.timedelta(hours=16)):
            raise TimeoutError('No new G20 block after the common 16h cutoff')
    return window, (start + dt.timedelta(hours=18)).isoformat()


def recover_exact50k_status(wd, cfg, root=ROOT):
    """Reconcile a saved exact50K after a crash, without resuming optimization.

    Missing candidates are not invented. An existing but inconsistent final
    candidate is an integrity failure, not a reason to restart training.
    """
    from safetensors.torch import load_file
    from g20.common import read
    from g20.plan import valid_sha
    from qg40.exposure import validate_counts, exposure_report
    wd, root = Path(wd), Path(root)
    status_path = wd / 'meta/training_status.json'
    previous = read(status_path)
    if previous.get('training_complete') is True:
        if previous.get('actual_updates') != 50000:
            raise ValueError('Completed training status is not exact50K')
        return False
    folder = wd / 'candidates/50000'
    if not folder.exists():
        return False
    case = validate_config(cfg)
    if wd.resolve() != resolved_path(cfg['work_dir'], root).resolve() or wd.name != case.run_id:
        raise ValueError('Recovery run/config directory identity differs')
    if yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text()) != cfg:
        raise ValueError('Recovery resolved config changed')
    start = read_json(wd / 'meta/training_start_manifest.json')
    data = read_json(resolved_path(cfg['g20']['dataset_manifest'], root))
    if data.get('sensor') != 'GF2' or data.get('num_bands') != 4 or data.get('max_pixel') != 1023:
        raise ValueError('Recovery data is not GF2/C4/DN1023')
    expected = dict(config_sha256=object_sha(cfg), data_sha256=object_sha(data),
                    source_identity=source_identity(root), reference_sha256=start.get('reference_sha256'))
    if (start.get('campaign_id') != CAMPAIGN_ID or start.get('run_id') != case.run_id
            or any(start.get(key) != value for key, value in expected.items())
            or (case.role == 'T' and expected['reference_sha256'] is not None)
            or (case.role == 'S' and not valid_sha(expected['reference_sha256']))
            or (case.reference_sha256 is not None and case.reference_sha256 != expected['reference_sha256'])):
        raise ValueError('Recovery training-start/config/data/reference/source identity differs')
    identity = read_json(folder / 'identity.json')
    state_path, model_path = folder / 'training_state.pt', folder / 'model.safetensors'
    if (identity.get('update') != 50000 or identity.get('full_state') is not True
            or identity.get('model_sha256') != sha256(model_path)
            or identity.get('training_state_sha256') != sha256(state_path)
            or identity.get('role') != case.role or identity.get('sensor') != 'GF2'
            or identity.get('num_bands') != 4 or identity.get('input_layout') != case.input_layout
            or any(identity.get(key) != value for key, value in expected.items())):
        raise ValueError('Existing exact50K candidate identity or bytes changed')
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    if (state.get('full_state') is not True or state.get('update') != 50000
            or state.get('precision') != 'fp32' or state.get('scheduler', {}).get('last_epoch') != 50000
            or state.get('model_sha256') != identity['model_sha256']
            or state_hash(state['model_state']) != identity.get('state_hash')
            or state_hash(load_file(str(model_path), device='cpu')) != identity.get('state_hash')
            or any(state.get(key) != value for key, value in expected.items())):
        raise ValueError('Exact50K full-state/model/scheduler/provenance differs')
    count = data['splits']['train']['count']
    counts = validate_counts(state.get('exposure_counts'), count, 50000, cfg['batch_size'])
    stream = BatchStream(count, cfg['batch_size'], case.seed)
    if state_hash({'order': stream.order, 'rotations': stream.rotations}) != start.get('sampler_hash'):
        raise ValueError('Recovery initial sample/view seed identity differs')
    stream.load_state_dict(state['sampler'])
    if stream.epoch * stream.count + stream.cursor != 50000:
        raise ValueError('Recovery sampler is not at 50000 completed batches')
    grid = read(wd / 'official/raw_grid.json')
    if grid:
        if (grid.get('campaign_id') != CAMPAIGN_ID or grid.get('run_id') != case.run_id
                or any(grid.get(key) != value for key, value in expected.items())):
            raise ValueError('Recovery evaluation grid identity differs')
        steps = [row['update'] for row in grid.get('records', [])]
        if len(set(steps)) != len(steps) or any(step not in GRID_STEPS for step in steps):
            raise ValueError('Recovery grid is outside the fixed candidate set')
    else:
        steps = []
    result = dict(status='TRAIN50K_COMPLETE', actual_updates=50000, training_complete=True,
        n_evaluated=len(steps), pending_steps=[step for step in GRID_STEPS if step not in steps],
        training_seconds=state['training_seconds'], evaluation_seconds=state['evaluation_seconds'],
        diagnostic_seconds=state.get('diagnostic_seconds', 0.), io_seconds=state.get('io_seconds', 0.),
        peak_memory_bytes=state.get('peak_memory_bytes'), peak_memory_scope='training+scheduled_eval+diagnostics',
        sample_exposure=exposure_report(counts, cfg['batch_size'], stream.epoch, stream.cursor),
        deadline_utc=previous.get('deadline_utc'), recovered_from_exact50k=True,
        recovery_candidate_sha256=identity['model_sha256'],
        recovery_fullstate_sha256=identity['training_state_sha256'], updated_at_utc=utcnow())
    atomic_json(status_path, result)
    return result


def train_run(config_path, device="cuda", resume=False, deadline_arg=None, root=ROOT):
    from g20.data import build_dataset
    from g20.evaluation import FRMetrics, evaluate_model
    from g20.references import load_reference

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
    f = cfg["g20"]
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
        for config_key, manifest_key in (('teacher_sha256', 'teacher_checkpoint_sha256'),
                                         ('tau_R', 'tau_R'), ('q_ref', 'q_ref'),
                                         ('q_cache_sha256', 'q_cache_sha256')):
            if f[config_key] != reference.get(manifest_key):
                raise ValueError('Resolved Student reference binding differs: ' + config_key)
        if case.reference_sha256 is not None and object_sha(reference) != case.reference_sha256:
            raise ValueError('Conditional case reference changed after the decision was locked')
        if (case.teacher_run_id is not None and reference["teacher_run_id"] != case.teacher_run_id) or reference["sensor"] != case.sensor:
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
        print(f'[G20 {case.sensor} official update={update}] '
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
            from g20.p1 import record_training_diagnostics
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
                status("PARTIAL_TIME_LIMIT" if not before_deadline(deadline) else "PAUSED_SIGNAL")
                return 75
            if stream.cursor == stream.count:
                stream.new_epoch()
            epoch_rng = torch.Generator().manual_seed(cfg["seed"] + 500000 + stream.epoch)
            loader = DataLoader(datasets["train"], batch_sampler=stream.remaining_batches(),
                num_workers=cfg["num_worker"], pin_memory=dev.type == "cuda", generator=epoch_rng)
            for gt, _lms, ms, lp, pan, meta in loader:
                if pause[0] or not before_deadline(deadline):
                    save_resume()
                    status("PARTIAL_TIME_LIMIT" if not before_deadline(deadline) else "PAUSED_SIGNAL")
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
                    raise FloatingPointError("Missing/nonfinite G20 gradients")
                optimizer.step(); scheduler.step(); update += 1; stream.cursor += 1
                from qg40.exposure import record_batch
                record_batch(exposure_counts, meta)
                training_seconds += time.monotonic() - started
                if update in GRID_STEPS or update in FULLSTATE_STEPS:
                    save_boundary(); status("RUNNING")
                if update % cfg["log_iter"] == 0:
                    print(f"[G20 {case.role}/{case.profile}] update={update}/50000", flush=True)
                if update == 50000:
                    break
        save_resume()
        status("OFFICIAL_EVAL_COMPLETE" if len(records) == 50 else "TRAIN50K_COMPLETE")
        return 0
    except TimeoutError:
        save_resume()
        status("PARTIAL_TIME_LIMIT" if update < 50000 else "TRAIN50K_COMPLETE")
        return 75
    except Exception:
        status("BLOCKED_INTEGRITY")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
