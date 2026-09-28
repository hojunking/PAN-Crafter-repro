"""Finite RB01 Student training; FH12 numerical recipe, train-only q controls.

No native RR/FR test set is opened by this module. Selection uses validation
ERGAS only; native test evaluation belongs to the separate post-training phase.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import time
import traceback
import uuid

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, default_collate
import yaml

from fh12.training import BatchStream, atomic_torch, rng_state, restore_rng, tensor_cpu_tree
from fh12.model import build_model, state_hash
from fh12.losses import student_losses, routed_student_backward
from fh12.evaluation import validation_ergas
from fh12.data import build_dataset, write_immutable_json
from fh20r1.training import make_scheduler, write_csv_row
from panda_rb.common import ROOT, atomic_json, read_json, sha256, object_sha, utcnow, source_identity, run_dir, docker_required

VAL_STEPS = tuple(1010 * n for n in range(1, 50)) + (50000,)
DIAGNOSTIC_STEPS = (0, 1000, 10000, 25000, 50000)


def geometry_student_losses(out, teacher_out, gt, tau_rec, geometry_weights):
    """The only intervention is the already detached per-sample geometry weight."""
    return student_losses(out, teacher_out, gt, tau_rec, geometry_weights)


def make_optimizer(model, cfg):
    return torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=float(cfg["learning_rate"]), name="U"),
        dict(params=model.aligner.parameters(), lr=3e-6, name="A")],
        betas=tuple(cfg["betas"]), eps=float(cfg["eps"]), weight_decay=float(cfg["weight_decay"]))


def stream_digest(n, batch_size, seed, updates=50000):
    """Pre-register the entire consumed sample/view stream without global RNG use."""
    stream = BatchStream(n, batch_size, seed)
    digest = bytes(32)
    for _ in range(updates):
        if stream.cursor == stream.count:
            stream.new_epoch()
        offset = stream.cursor * batch_size
        meta = torch.stack((stream.order[offset:offset + batch_size],
                            stream.rotations[offset:offset + batch_size]), dim=1)
        digest = hashlib.sha256(digest + meta.numpy().astype("<i8").tobytes()).digest()
        stream.cursor += 1
    return digest.hex()


def update_stream_digest(previous, meta):
    # FH12Dataset emits [source, rot, hflip, vflip], whereas BatchStream plans
    # source/rot only. Validate fixed augmentation, hash the same two columns.
    if meta.ndim != 2 or meta.shape[1] != 4 or not bool((meta[:, 2:] == 1).all()):
        raise ValueError("Training metadata must be [source,rot,1,1] (fixed HV)")
    return hashlib.sha256(bytes.fromhex(previous) + meta[:, :2].detach().cpu().numpy().astype("<i8").tobytes()).hexdigest()


def _norm(tensors):
    return math.sqrt(sum(float(t.detach().double().square().sum()) for t in tensors if t is not None))


def _quantiles(value):
    value = value.detach().float().reshape(-1)
    return dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"),
                    torch.quantile(value, value.new_tensor([0., .1, .25, .5, .75, .9, 1.])).cpu().tolist()))


def diagnostic_probe(model, teacher, batch, tau_rec, weights=None, raw_q=None, cached_e=None):
    """Read-only fixed-probe diagnostics; preserve modes, buffers, .grad and RNG."""
    saved_rng = rng_state()
    modes = [(module, module.training) for parent in (model, teacher) for module in parent.modules()]
    buffers = [(buffer, buffer.detach().clone()) for parent in (model, teacher) for buffer in parent.buffers()]
    gradients = [(p, p.grad) for p in model.parameters()]
    model.eval(); teacher.eval()
    try:
        gt, _lms, ms, lp, pan, meta = batch
        out = model(pan, ms, lp)
        with torch.no_grad():
            teacher_out = teacher(pan, ms, lp)
        applied = weights is not None
        w = weights if applied else torch.ones(gt.shape[0], device=gt.device)
        losses = geometry_student_losses(out, teacher_out, gt, tau_rec, w)
        es = (out["y"].detach() - gt).abs().mean(1, keepdim=True)
        et = (teacher_out["y"].detach() - gt).abs().mean(1, keepdim=True)
        soft_weight = .1 * (1. - losses["difficulty"]) * losses["advantage"]
        delta = out["delta"].detach()
        result = dict(sample_view=meta.detach().cpu().tolist(), geometry_applied=applied,
                      hard_raw=float(es.mean()), H=float(losses["hard"].detach()),
                      soft_residual_raw=float((out["y"].detach() - teacher_out["y"]).abs().mean()),
                      K_beta_included=float(losses["soft"].detach()),
                      edge_raw=float(losses["edge"].detach()),
                      edge_weighted=float(losses["edge_weighted"].detach()) if applied else None,
                      L_U=float(losses["L_U"].detach()) if applied else None,
                      L_A=float(losses["L_A"].detach()) if applied else None,
                      advantage_active_fraction=float((losses["advantage"] > 0).float().mean()),
                      mean_soft_weight=float(soft_weight.mean()),
                      teacher_e_mean=float(et.mean()), student_e_mean=float(es.mean()),
                      student_better_fraction=float((es < et).float().mean()),
                      e_teacher_quantiles=_quantiles(et), e_student_quantiles=_quantiles(es),
                      correction_dy_dx=delta.cpu().tolist(),
                      correction_mean_dy_dx=delta.mean(0).cpu().tolist(),
                      correction_max_abs=float(delta.abs().max()),
                      gradient_routing="U: H+K+.002*w*E; A: w*H; no soft/edge to A")
        if applied:
            result["weight_quantiles"] = _quantiles(w)
            result["weight_values"] = w.detach().cpu().tolist()
            groups = {"U": list(model.backbone.parameters()), "A": list(model.aligner.parameters())}
            result["gradient_norms"] = {
                f"{term}_{group}": _norm(torch.autograd.grad(value, params, retain_graph=True, allow_unused=True))
                for term, value in (("H", losses["hard"]), ("K", losses["soft"]),
                                    ("weighted_E", losses["edge_weighted"]),
                                    ("L_U", losses["L_U"]), ("L_A", losses["L_A"]))
                for group, params in groups.items()}
        if raw_q is not None:
            result["q_quantiles"] = _quantiles(raw_q)
            result["q_values"] = raw_q.detach().cpu().tolist()
        if cached_e is not None:
            result["e_cache_quantiles"] = _quantiles(cached_e)
        if any(p.grad is not grad for p, grad in gradients):
            raise AssertionError("Probe changed optimizer gradients")
        return result
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
        for module, training in modes:
            module.training = training
        for parameter, grad in gradients:
            parameter.grad = grad
        restore_rng(saved_rng)


def one_batch_parity(model, teacher, batch, tau_rec, q_raw, q_ref, qfull_weights, cfg):
    """Compare actual QFULL path with FH20R1's unchanged student loss/routing.

    Update index 100 uses a nonzero peak learning rate; comparing only the first
    warmup update (LR=0) would not test an actual optimizer parameter change.
    """
    preserved = rng_state()
    original_mode = teacher.training
    teacher.eval()
    try:
        baseline, controlled = copy.deepcopy(model), copy.deepcopy(model)
        gt, _lms, ms, lp, pan, _meta = batch
        with torch.no_grad():
            tout = teacher(pan, ms, lp)
        results = []
        for candidate, weights, loss_fn in ((baseline, float(q_ref) / (float(q_ref) + q_raw), student_losses),
                                             (controlled, qfull_weights, geometry_student_losses)):
            optimizer = make_optimizer(candidate, cfg)
            candidate.train(); optimizer.zero_grad(set_to_none=True)
            out = candidate(pan, ms, lp)
            losses = loss_fn(out, tout, gt, tau_rec, weights)
            routed_student_backward(candidate, losses)
            grads = {name: param.grad.detach().clone() for name, param in candidate.named_parameters()}
            optimizer.step()
            results.append((out, losses, grads, candidate.state_dict()))
        left, right = results
        differences = {"geometry_weight": float((float(q_ref) / (float(q_ref) + q_raw) - qfull_weights).abs().max()),
                       "prediction": float((left[0]["y"] - right[0]["y"]).abs().max()),
                       "correction": float((left[0]["delta"] - right[0]["delta"]).abs().max())}
        for term in ("hard", "soft", "edge", "edge_weighted", "L_U", "L_A"):
            differences[term] = float((left[1][term] - right[1][term]).abs().max())
        for label, position in (("gradient", 2), ("optimizer_update", 3)):
            differences[label] = max(float((left[position][name] - right[position][name]).abs().max())
                                     for name in left[position])
        if not all(math.isfinite(value) and value <= 3e-6 for value in differences.values()):
            raise AssertionError(f"QFULL numerical recipe parity failed: {differences}")
        return dict(status="PASS", differences=differences, atol=3e-6,
                    optimizer_update_index=100, optimizer_lr_U=1e-4, optimizer_lr_A=3e-6,
                    original_function="fh12.losses.student_losses / routed_student_backward",
                    teacher_hash=state_hash(teacher.state_dict()), device=str(gt.device))
    finally:
        teacher.train(original_mode)
        restore_rng(preserved)


def choose_validation(records):
    if not records or any(not math.isfinite(float(row["val_ergas"])) for row in records):
        raise ValueError("Validation records must be nonempty and finite")
    if len({row["update"] for row in records}) != len(records):
        raise ValueError("Duplicate validation updates")
    return min(records, key=lambda row: (float(row["val_ergas"]), int(row["update"])))


def _atomic_link(path, target):
    path = Path(path)
    if path.exists() and not path.is_symlink():
        raise ValueError(f"Expected an owned checkpoint pointer: {path}")
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex)
    temporary.symlink_to(target)
    os.replace(temporary, path)


def save_resume(directory, snapshot):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    last = directory / "last"
    pending = Path(tempfile.mkdtemp(prefix=".resume_", dir=directory))
    try:
        atomic_torch(pending / "training_state.pt", snapshot)
        atomic_json(pending / "identity.json", dict(update=snapshot["update"], full_state=True,
                    training_state_sha256=sha256(pending / "training_state.pt"),
                    config_sha256=snapshot["config_sha256"]))
        _atomic_link(last, pending.name)
    except BaseException:
        if not last.is_symlink() or last.resolve() != pending.resolve():
            shutil.rmtree(pending)
        raise
    # Published full states are retained until the user's experiment selection.
    # Updating `last` never removes a preceding committed optimizer/RNG state.


def read_resume(directory, expected):
    directory = Path(directory) / "last"
    identity = read_json(directory / "identity.json")
    if sha256(directory / "training_state.pt") != identity["training_state_sha256"]:
        raise ValueError("RB01 full-state checksum mismatch")
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    if not state.get("full_state") or int(state["update"]) != int(identity["update"]):
        raise ValueError("RB01 exact resume needs a consistent full training state")
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"RB01 exact-resume identity mismatch: {key}")
    return state


def save_weights(directory, model, identity):
    directory = Path(directory); directory.parent.mkdir(parents=True, exist_ok=True)
    wanted = dict(identity, state_hash=state_hash(model.state_dict()))
    if directory.exists():
        existing = read_json(directory / "identity.json")
        if any(existing.get(key) != value for key, value in wanted.items()) or sha256(directory / "model.safetensors") != existing["model_sha256"]:
            raise ValueError("Existing immutable RB01 weights differ")
        return existing
    temporary = Path(tempfile.mkdtemp(prefix=".weights_", dir=directory.parent))
    try:
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
                  str(temporary / "model.safetensors"))
        wanted["model_sha256"] = sha256(temporary / "model.safetensors")
        atomic_json(temporary / "identity.json", wanted)
        os.replace(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return wanted


def _validate_recipe(cfg):
    expected = dict(num_iter=50000, num_warmup=100, batch_size=48, mixed_precision="no",
                    learning_rate=1e-4, optimizer="AdamW", weight_decay=.01, eps=1e-8)
    if any(cfg.get(key) != value for key, value in expected.items()) or list(cfg.get("betas", ())) != [.9, .999]:
        raise ValueError("RB01 may not change the original registered training recipe")


def require_training_device(device):
    docker_required()
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production RB01 training requires CUDA; use isolated unit tests for CPU")
    return dev


def required_storage_bytes(runs=8):
    """Conservative normal-run budget, including retained optimizer states/raws.

    Repeated technical retries can add extra states; they are never deleted to
    satisfy this estimate and require an additional free-space check at launch.
    """
    model, _ = build_model("PLH", 104, [1, 2, 2], 9281101, role="T")
    parameter_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    states = len(set(VAL_STEPS) | set(DIAGNOSTIC_STEPS))
    # model + AdamW first/second moments + sampler/RNG/metadata and safety.
    per_run = parameter_bytes * (51 + 4 * states)
    raw_bytes = 2 * 20 * 8 * (256**2 + 512**2) * 4  # exact + distinct val selection
    per_run += raw_bytes + 512 * 1024**2
    return dict(runs=int(runs), model_state_bytes=parameter_bytes,
                normal_fullstates=states, retained_validation_candidates=50,
                estimated_per_run_bytes=per_run, required_bytes=per_run * int(runs) + 5 * 1024**3,
                automatic_checkpoint_cleanup=False, retry_storage_not_bounded=True)


def training_smoke(binding_path=None, weights_path=None, seed=9281101, device="cpu", root=ROOT):
    """Disposable parity/gradient/optimizer check, never a registered train run.

    CPU mode is deliberately synthetic and labeled as such. CUDA mode requires
    the bound actual F1, complete weights, native64 train batch48 and W104D122.
    """
    preserved = rng_state()
    try:
        from panda_rb.plan import build_config, training_runs
        row = next(item for item in training_runs(root=root) if item["seed"] == seed)
        cfg = build_config(row["run_id"], root=root)
        dev = torch.device(device)
        if dev.type == "cuda":
            if binding_path is None or weights_path is None or not torch.cuda.is_available():
                raise ValueError("CUDA smoke requires actual bound F1, geometry assets and CUDA")
            from panda_rb.binding import load_binding
            from panda_rb.weights import load_weights
            teacher, _cfg, binding, data, raw_q = load_binding(binding_path, device=device)
            arrays, manifest = load_weights(weights_path, seed=seed, binding=binding)
            dataset = build_dataset(data, "train", root=root)
            stream = BatchStream(len(dataset), 48, seed)
            batch = tuple(value.to(dev) for value in default_collate(
                [dataset[index] for index in stream.remaining_batches()[0]]))
            model, _init = build_model("PLH", 104, [1, 2, 2], seed, role="S",
                                      teacher_aligner_state=teacher.aligner.state_dict())
            q = torch.as_tensor(raw_q, dtype=torch.float32, device=dev)
            full = torch.as_tensor(arrays["QFULL"], dtype=torch.float32, device=dev)
            meta = batch[-1]; q = q[meta[:, 0], meta[:, 1]]
            weights = full[meta[:, 0], meta[:, 1]]
            tau, q_ref = binding["tau_R"], binding["q_ref"]
            torch.cuda.reset_peak_memory_stats(dev)
            provenance = dict(binding_sha256=object_sha(binding), weights_sha256=object_sha(manifest),
                              actual_batch48=True, synthetic=False, gpu=torch.cuda.get_device_name(dev))
        else:
            generator = torch.Generator().manual_seed(seed)
            teacher, _ = build_model("P0", 8, [1, 1, 1], seed, role="T")
            model, _ = build_model("PLH", 8, [1, 1, 1], seed, role="S",
                                   teacher_aligner_state=teacher.aligner.state_dict())
            batch = (torch.rand(2, 8, 32, 32, generator=generator) * 2 - 1,
                     torch.rand(2, 8, 32, 32, generator=generator) * 2 - 1,
                     torch.rand(2, 8, 8, 8, generator=generator) * 2 - 1,
                     torch.rand(2, 1, 8, 8, generator=generator) * 2 - 1,
                     torch.rand(2, 1, 32, 32, generator=generator) * 2 - 1,
                     torch.tensor([[0, 0], [1, 0]]))
            q = torch.tensor([.12, .32]); q_ref = .2; tau = .1
            weights = q_ref / (q_ref + q)
            provenance = dict(actual_batch48=False, synthetic=True, production_CUDA_status="NOT_RUN")
        model.to(dev); teacher.to(dev).eval().requires_grad_(False)
        result = one_batch_parity(model, teacher, batch, tau, q, q_ref, weights, cfg)
        result.update(provenance)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(dev)
            # Independent, isolated original Git reader: not just two calls to
            # the same live implementation bearing different model labels.
            from fh20r1.historical import run_parity
            torch.cuda.empty_cache()
            result["historical_reader_parity"] = run_parity(
                root, binding["origin_reference"], binding["paths"], data, device=device)
        return result
    finally:
        restore_rng(preserved)


def train_run(run_id, binding_path, weights_path, device="cuda", resume=False, root=ROOT):
    """Train one registered fresh Student. Invocation is explicit run authority."""
    from panda_rb.plan import build_config, case_for
    from panda_rb.binding import load_binding
    from panda_rb.weights import load_weights

    root = Path(root).resolve()
    cfg, case = build_config(run_id, root=root), case_for(run_id, root=root)
    _validate_recipe(cfg)
    dev = require_training_device(device)
    teacher, _origin_cfg, binding, data, raw_q = load_binding(binding_path, device=device)
    weight_arrays, weight_manifest = load_weights(weights_path, seed=case["seed"], binding=binding)
    weights = weight_arrays[case["case_id"]]
    teacher.eval().requires_grad_(False)
    datasets = {split: build_dataset(data, split, root=root) for split in ("train", "val")}
    q = torch.as_tensor(raw_q, dtype=torch.float32, device=dev)
    w = torch.as_tensor(weights, dtype=torch.float32, device=dev)
    if q.shape != (len(datasets["train"]), 4) or w.shape != q.shape or not bool(torch.isfinite(w).all()) or not bool(((w > 0) & (w <= 1)).all()):
        raise ValueError("RB01 q and geometry weights must cover exact full train × 4 views")
    q_ref, tau_rec = float(binding["q_ref"]), float(binding["tau_R"])
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    model, init = build_model("PLH", 104, [1, 2, 2], cfg["seed"], role="S",
                              teacher_aligner_state=teacher.aligner.state_dict())
    model.to(dev).train()
    teacher_hash = state_hash(teacher.state_dict())
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, "BASE", cfg["num_warmup"], cfg["num_iter"])
    stream = BatchStream(len(datasets["train"]), cfg["batch_size"], cfg["seed"])
    wd = run_dir(run_id, root=root); wd.mkdir(parents=True, exist_ok=True)
    release = source_identity(root)
    expected = dict(config_sha256=object_sha(cfg), source_identity=release,
                    binding_sha256=object_sha(binding), weights_sha256=object_sha(weight_manifest),
                    data_sha256=object_sha(data), teacher_state_hash=teacher_hash)
    init_manifest = dict(init, binding_sha256=expected["binding_sha256"], teacher_state_hash=teacher_hash)
    planned_digest = stream_digest(len(datasets["train"]), cfg["batch_size"], cfg["seed"], cfg["num_iter"])
    stream_manifest = dict(seed=cfg["seed"], updates=cfg["num_iter"], batch_size=cfg["batch_size"],
                           training_count=len(datasets["train"]), full_stream_sha256=planned_digest,
                           algorithm="FH12 BatchStream; chained SHA256 of batch-major little-endian int64 [source,view]",
                           augmentation="fixed HV then rot90; q mapping RNG separate")
    pair = dict(seed=cfg["seed"], init_U=init["hashes"]["body"], init_A=init["hashes"]["A"],
                init_full=init["hashes"]["full"], teacher=teacher_hash,
                binding_sha256=expected["binding_sha256"], data_sha256=expected["data_sha256"],
                stream_sha256=planned_digest)
    write_immutable_json(wd.parent / "paired_identity.json", pair)
    for path, value in (("init_manifest.json", init_manifest), ("stream_manifest.json", stream_manifest),
                        ("bindings.json", binding), ("case.json", case)):
        write_immutable_json(wd / path, value)
    (wd / "meta").mkdir(exist_ok=True)
    config_file = wd / "meta/config.resolved.yaml"
    if config_file.exists() and yaml.safe_load(config_file.read_text()) != cfg:
        raise ValueError("Stored RB01 config has changed")
    if not config_file.exists():
        config_file.write_text(yaml.safe_dump(cfg, sort_keys=False))

    update = 0; records = []; diagnostics_done = []
    digest = bytes(32).hex(); consumed = dict(count=0, sum=0., squared_sum=0., min=1., max=0.)
    consumed_counts = np.zeros(tuple(q.shape), dtype=np.int64)
    timings = dict(train=0., validation=0., diagnostics=0., io=0.)
    checkpoint_dir = wd / "checkpoints"
    if resume:
        saved = read_resume(checkpoint_dir, expected)
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        stream.load_state_dict(saved["sampler"])
        update = int(saved["update"]); records = saved["validation_records"]
        diagnostics_done = saved["diagnostics_done"]; digest = saved["consumed_stream_sha256"]
        consumed = saved["consumed_weights"]; timings = saved["timings"]
        consumed_counts = saved["consumed_sample_view_counts"]
        restore_rng(saved["rng"])
        if scheduler.last_epoch != update or any(int(row["update"]) > update for row in records):
            raise ValueError("RB01 resume scheduler or validation state is ahead of completed updates")
    elif (checkpoint_dir / "last").exists() or (wd / "meta/training_status.json").exists():
        raise ValueError("Existing RB01 run requires --resume; never replace a failed seed")

    fixed_ids = np.asarray(weight_arrays["calibration_indices"], dtype=np.int64)[:8]
    if len(fixed_ids) != 8 or len(set(fixed_ids.tolist())) != 8:
        raise ValueError("Binding must preserve eight original train calibration indices for probes")
    train_probe = tuple(value.to(dev) for value in default_collate([datasets["train"][(int(i), 0)] for i in fixed_ids]))
    val_ids = np.random.default_rng(9282026).choice(len(datasets["val"]), size=min(8, len(datasets["val"])), replace=False)
    val_probe = tuple(value.to(dev) for value in default_collate([datasets["val"][int(i)] for i in val_ids]))
    probe_idx = torch.as_tensor(fixed_ids, device=dev)
    parity_path = wd / "meta/qfull_parity.json"
    if parity_path.exists():
        parity = read_json(parity_path)
        if parity.get("status") != "PASS" or any(parity.get(key) != value for key, value in expected.items()):
            raise ValueError("Cached QFULL parity identity no longer matches this run")
    else:
        full_weights = torch.as_tensor(weight_arrays["QFULL"][fixed_ids, 0], device=dev)
        atomic_json(parity_path, dict(one_batch_parity(
            model, teacher, train_probe, tau_rec, q[probe_idx, 0], q_ref, full_weights, cfg), **expected))
    attempt_id = uuid.uuid4().hex
    attempt_path = wd / "meta/attempts" / f"{attempt_id}.json"
    attempt = dict(attempt_id=attempt_id, run_id=run_id, same_seed=cfg["seed"], resume=resume,
                   started_at_utc=utcnow(), starting_update=update, device=str(dev),
                   runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                cudnn=torch.backends.cudnn.version(),
                                gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU_TEST_ONLY"),
                   **expected)
    atomic_json(attempt_path, dict(attempt, status="RUNNING"))
    pause = [False]
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda *_: pause.__setitem__(0, True))

    def status(name, **extra):
        atomic_json(wd / "meta/training_status.json", dict(status=name, run_id=run_id,
                    actual_updates=update, training_complete=update == cfg["num_iter"],
                    timings=timings, attempt_id=attempt_id, updated_at_utc=utcnow(), **expected, **extra))

    def snapshot():
        return dict(full_state=True, update=update, model_state=tensor_cpu_tree(model.state_dict()),
                    optimizer=tensor_cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                    rng=rng_state(), sampler=stream.state_dict(), validation_records=records,
                    diagnostics_done=diagnostics_done, consumed_stream_sha256=digest,
                    consumed_weights=consumed, consumed_sample_view_counts=consumed_counts,
                    timings=timings, **expected)

    def commit():
        began = time.monotonic(); save_resume(checkpoint_dir, snapshot())
        timings["io"] += time.monotonic() - began

    def identity():
        return dict(update=update, run_id=run_id, input_layout="PLH", width=104, depth=[1, 2, 2],
                    role="S", **expected)

    def diagnose():
        if update not in DIAGNOSTIC_STEPS or update in diagnostics_done:
            return
        began = time.monotonic()
        result = dict(update=update, train=diagnostic_probe(model, teacher, train_probe, tau_rec,
                      w[probe_idx, 0], q[probe_idx, 0],
                      torch.as_tensor(weight_arrays["e_bar"][fixed_ids, 0], device=dev)),
                      validation=diagnostic_probe(model, teacher, val_probe, tau_rec),
                      validation_indices=val_ids.tolist(), consumed_weights=dict(consumed),
                      actual_update_norm_source="diagnostics/actual_updates.csv", **expected)
        mapping_key = {"QSHUF": "shuffle_source", "QESUR": "surrogate_source"}.get(case["case_id"])
        result["geometry_mapping"] = dict(case=case["case_id"], source_ids=fixed_ids.tolist(), view=0,
                    mapped_weight_source_ids=(weight_arrays[mapping_key][fixed_ids, 0].tolist()
                                              if mapping_key else fixed_ids.tolist()),
                    mapping_is_population_mean=case["case_id"] == "QMEAN",
                    e_stratum=weight_arrays["e_stratum"][fixed_ids, 0].tolist())
        previous_path = wd / "diagnostics/step_0.json"
        if update and previous_path.exists():
            previous = read_json(previous_path)
            old = np.asarray(previous["train"]["correction_dy_dx"])
            result["train"]["correction_drift_from_init_dy_dx"] = (
                np.asarray(result["train"]["correction_dy_dx"]) - old).tolist()
        atomic_json(wd / "diagnostics" / f"step_{update}.json", result)
        diagnostics_done.append(update); timings["diagnostics"] += time.monotonic() - began

    def validate():
        if update not in VAL_STEPS or any(row["update"] == update for row in records):
            return
        began = time.monotonic(); previous_rng = rng_state()
        try:
            score = validation_ergas(model, datasets["val"], dev)
        finally:
            restore_rng(previous_rng)
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation ERGAS")
        records.append(dict(update=update, val_ergas=score))
        selected = choose_validation(records)
        # Retain every registered validation candidate, not just winning points.
        save_weights(checkpoint_dir / "val_candidates" / str(update), model, identity())
        if update == cfg["num_iter"]:
            save_weights(checkpoint_dir / "exact50000", model, identity())
        timings["validation"] += time.monotonic() - began
        atomic_json(wd / "validation_grid.json", dict(records=records, expected_steps=list(VAL_STEPS),
                    selector="RR_VAL_ERGAS_MIN_THEN_EARLIER", test_FR_sweep=False,
                    complete=[row["update"] for row in records] == list(VAL_STEPS), **expected))
        print(f"[RB01 {run_id}] update={update} valERGAS={score:.8f}; HQNR/SCC/nativeERGAS deferred to selected-checkpoint postrun", flush=True)

    try:
        status("RUNNING")
        # First diagnostic/parity-independent failure must still have an exact
        # step-zero resume point; do not strand a run behind an empty status.
        commit(); diagnose(); validate(); commit()
        while update < cfg["num_iter"]:
            if pause[0]:
                commit(); status("PAUSED_SIGNAL")
                atomic_json(attempt_path, dict(attempt, status="PAUSED_SIGNAL", ending_update=update, ended_at_utc=utcnow()))
                return 75
            if stream.cursor == stream.count:
                stream.new_epoch()
            generator = torch.Generator().manual_seed(cfg["seed"] + 500000 + stream.epoch)
            loader = DataLoader(datasets["train"], batch_sampler=stream.remaining_batches(),
                                num_workers=cfg["num_worker"], pin_memory=True, generator=generator)
            for batch in loader:
                if pause[0]:
                    break
                began = time.monotonic()
                gt, _lms, ms, lp, pan, meta = batch
                gt, ms, lp, pan = [value.to(dev, non_blocking=True) for value in (gt, ms, lp, pan)]
                optimizer.zero_grad(set_to_none=True); model.train()
                out = model(pan, ms, lp)
                with torch.no_grad():
                    teacher_out = teacher(pan, ms, lp)
                weights_b = w[meta[:, 0].to(dev), meta[:, 1].to(dev)]
                losses = geometry_student_losses(out, teacher_out, gt, tau_rec, weights_b)
                if not bool(torch.isfinite(losses["L_U"]) & torch.isfinite(losses["L_A"])):
                    raise FloatingPointError("Nonfinite RB01 Student objective")
                routed_student_backward(model, losses)
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError("Missing/nonfinite RB01 Student gradients")
                measure_update = update + 1 in DIAGNOSTIC_STEPS or update == 0
                before = {name: p.detach().clone() for name, p in model.named_parameters()} if measure_update else None
                if measure_update:
                    grad_norms = {"grad_U": _norm(p.grad for p in model.backbone.parameters()),
                                  "grad_A": _norm(p.grad for p in model.aligner.parameters())}
                    used_lrs = [group["lr"] for group in optimizer.param_groups]
                optimizer.step(); scheduler.step(); update += 1; stream.cursor += 1
                digest = update_stream_digest(digest, meta)
                np.add.at(consumed_counts, (meta[:, 0].numpy(), meta[:, 1].numpy()), 1)
                values = weights_b.detach().double()
                consumed["count"] += len(values); consumed["sum"] += float(values.sum())
                consumed["squared_sum"] += float(values.square().sum())
                consumed["min"] = min(consumed["min"], float(values.min()))
                consumed["max"] = max(consumed["max"], float(values.max()))
                if measure_update:
                    delta_norms = {f"update_{group}": _norm(p.detach() - before[name]
                                   for name, p in model.named_parameters() if name.startswith(prefix))
                                   for group, prefix in (("U", "backbone."), ("A", "aligner."))}
                    write_csv_row(wd / "diagnostics/actual_updates.csv", dict(update=update,
                        lr_U=used_lrs[0], lr_A=used_lrs[1], **grad_norms, **delta_norms))
                timings["train"] += time.monotonic() - began
                if update % cfg["log_iter"] == 0:
                    print(f"[RB01 {run_id}] update={update}/50000 L_U={float(losses['L_U'].detach()):.8g} L_A={float(losses['L_A'].detach()):.8g} consumed_w={consumed['sum']/consumed['count']:.8f}", flush=True)
                if update in DIAGNOSTIC_STEPS or update in VAL_STEPS:
                    diagnose(); validate(); commit()
                    status("RUNNING")
                if update >= cfg["num_iter"]:
                    break
        if state_hash(teacher.state_dict()) != teacher_hash:
            raise AssertionError("Frozen F1 Teacher changed")
        if digest != planned_digest:
            raise AssertionError("Actual sample/view stream differs from preregistered 50K stream")
        if [row["update"] for row in records] != list(VAL_STEPS):
            raise ValueError("Incomplete prescribed validation grid")
        best = choose_validation(records)
        target = "exact50000" if best["update"] == cfg["num_iter"] else f"val_candidates/{best['update']}"
        _atomic_link(checkpoint_dir / "val_selected", target)
        primary = read_json(checkpoint_dir / "exact50000/identity.json")
        secondary = read_json(checkpoint_dir / "val_selected/identity.json")
        alias = "EXACT_50000" if primary["model_sha256"] == secondary["model_sha256"] else None
        atomic_json(checkpoint_dir / "selection_manifest.json", dict(
            primary=dict(selection_id="EXACT_50000", update=cfg["num_iter"], directory="exact50000", checkpoint_sha256=primary["model_sha256"]),
            secondary=dict(selection_id="RR_VAL_ERGAS_MIN", update=best["update"], directory="val_selected", checkpoint_sha256=secondary["model_sha256"], alias_of=alias),
            validation_records=records, complete=True, actual_updates=update, test_FR_sweep=False, **expected))
        mean = consumed["sum"] / consumed["count"]
        atomic_json(wd / "diagnostics/consumed_weights.json", dict(consumed, mean=mean,
                    std=math.sqrt(max(0., consumed["squared_sum"] / consumed["count"] - mean * mean)),
                    consumed_stream_sha256=digest, normalization_applied=False))
        atomic_torch(wd / "diagnostics/consumed_sample_view_counts.pt", torch.from_numpy(consumed_counts))
        status("TRAINING_COMPLETE")
        atomic_json(attempt_path, dict(attempt, status="TRAINING_COMPLETE", ending_update=update, ended_at_utc=utcnow()))
        return 0
    except BaseException as error:
        failure_type = "NUMERICAL_FAILURE" if isinstance(error, FloatingPointError) else "TECHNICAL_FAILURE"
        status(failure_type, failure=str(error))
        atomic_json(attempt_path, dict(attempt, status=failure_type, ending_update=update,
                    error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), ended_at_utc=utcnow(),
                    recovery="same registered run and seed only; resume last verified full state"))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
