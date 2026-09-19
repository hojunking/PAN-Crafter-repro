"""Disposable N2 Teacher even/odd backward check, never a training artifact."""
import gc
from pathlib import Path
import time

import torch

from fh12.common import ROOT, read_json, sha256
from fh12.losses import teacher_loss
from fh12.model import state_hash
from fh12.training import rng_state, restore_rng
from fh20r1.plan import N2_TEACHER_RUN, build_config, case_for
from fh20r1.references import load_donor_aligner
from fh20r1.training import build_n2_teacher
from tools.fh12_preflight import _smoke_batch


def n2_teacher_smoke(server, bridge, device="cuda", root=ROOT):
    """s2: exact production N2 initialization and two disposable AdamW updates.

    CUDA uses actual batch48 train64 and the registered W112D123. CPU retains
    initialization checks but uses two synthetic 32px samples: never a GPU or
    production-batch certificate. An unavailable donor skips only this arm.
    """
    report = dict(server_id=server, device=str(device), passed=False, disposable=True,
                  parameters_reused_in_actual_training=False, credited_new_seconds=0)
    if server != "s2":
        return dict(report, status="NOT_APPLICABLE", actual_batch48_smoke_status="NOT_APPLICABLE")
    donor, donor_report = load_donor_aligner(root, server=server)
    if donor is None:
        return dict(report, status="SKIPPED_BLOCKED_DONOR", donor=donor_report,
                    actual_batch48_smoke_status="SKIPPED_BLOCKED_DONOR")
    if (bridge.get("alias") != "F2" or bridge.get("server") != "s2"
            or bridge.get("status") != "PASS" or bridge.get("complete") is not True):
        raise ValueError("N2 smoke requires the verified local F2 bridge")
    init_path = Path(bridge["resolved_artifacts"]["init_manifest"])
    if not init_path.is_absolute(): init_path = Path(root) / init_path
    if sha256(init_path) != bridge["resolved_artifact_sha256"]["init_manifest"]:
        raise ValueError("F2 initialization manifest changed before N2 smoke")
    dev = torch.device(device); gpu = dev.type == "cuda"
    if gpu and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable for N2 Teacher smoke")
    saved_rng = rng_state(); started = time.monotonic()
    case = case_for(N2_TEACHER_RUN); cfg = build_config(case)
    model = optimizer = out = losses = tensors = gt = ms = pan = lp = module = grads = None
    try:
        model, init = build_n2_teacher(case, donor, donor_report, read_json(init_path))
        initial_a = state_hash(model.aligner.state_dict())
        if initial_a != state_hash(donor): raise AssertionError("N2 smoke did not preserve exact donor A")
        if gpu:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
            tensors = _smoke_batch(bridge["dataset_manifest"], dev, 48)
        else:
            gen = torch.Generator().manual_seed(9019)
            tensors = [torch.randn(shape, generator=gen) * .2 for shape in
                       ((2, 8, 32, 32), (2, 8, 8, 8), (2, 1, 32, 32), (2, 1, 8, 8))]
        gt, ms, pan, lp = tensors
        model.to(dev).train()
        optimizer = torch.optim.AdamW([
            dict(params=model.backbone.parameters(), lr=cfg["learning_rate"]),
            dict(params=model.aligner.parameters(), lr=cfg["fh20r1"]["aligner_lr"])],
            betas=tuple(cfg["betas"]), eps=cfg["eps"], weight_decay=cfg["weight_decay"])
        corruption = torch.Generator().manual_seed(cfg["fh20r1"]["corruption_seed"])
        updates = []
        for update in (0, 1):
            optimizer.zero_grad(set_to_none=True)
            out = model(pan, ms, lp)
            losses = teacher_loss(model, out, gt, pan, ms, update, corruption)
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError("Nonfinite N2 Teacher smoke objective")
            losses["total"].backward()
            group_checks = {}
            for name, module in (("U", model.backbone), ("A", model.aligner)):
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError(f"Missing/nonfinite N2 Teacher {name} gradients")
                group_checks[name] = len(grads)
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError("Nonfinite N2 Teacher parameters after AdamW")
            updates.append(dict(update_index=update, loss=float(losses["total"].detach()),
                                offset_active=losses["offset_active"], gradient_tensor_counts=group_checks))
        if gpu: torch.cuda.synchronize(dev)
        return dict(report, passed=True, status="PASS" if gpu else "CPU_DIAGNOSTIC_ONLY",
                    actual_batch48_smoke_status="PASS" if gpu else "NOT_RUN_CPU_DIAGNOSTIC",
                    batch_size=len(pan), patch_size=pan.shape[-1], width=case.width, depth=list(case.depth),
                    layout=case.input_layout, teacher_run=case.run_id, donor=donor_report,
                    donor_aligner_hash=initial_a, fresh_F2_U_verified=init["fresh_F2_U_verified"],
                    donor_head_preserved=True, precision="fp32", optimizer="AdamW", updates=updates,
                    peak_allocated_mb=torch.cuda.max_memory_allocated(dev)/2**20 if gpu else None,
                    elapsed_seconds=time.monotonic()-started)
    finally:
        model = optimizer = out = losses = tensors = gt = ms = pan = lp = module = grads = None
        gc.collect()
        if gpu: torch.cuda.empty_cache()
        restore_rng(saved_rng)
