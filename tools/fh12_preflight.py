#!/usr/bin/env python3
"""FH12 local preflight; never starts/resets a campaign window or legacy run.

An executing preflight requires the explicit starter's existing window.json.
--check-only is read-only: inspect frozen configs/data identities and list missing
prepared assets, without constructing caches, running CUDA, or certifying PASS.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fh12.common import (atomic_json, check_deadline, object_sha, read_json,
                         sha256, source_identity, utcnow)
from fh12.data import DEFAULT_PATHS, RECIPE, _source_info, canonical_sha, prepare_data
from fh12.model import LAYOUTS, build_model, frontend_self_test, state_hash
from fh12.losses import teacher_loss, student_losses, routed_student_backward
from fh12.plan import (CAMPAIGN_ID, METHOD_REVISION, REGISTRY_REVISION, SERVERS,
                       SOURCE_PLAN, CASES, build_config, cases_for, registry_sha256,
                       teacher_for)
from tools.gen_fh12_configs import artifacts


HASH_KEYS = {"train": "train_feeder_args", "val": "val_feeder_args",
             "rr": "test_reduced_feeder_args", "fr": "test_full_feeder_args"}


def validate_registry(root):
    """Check all deployed servers, not only the next selected teacher."""
    root = Path(root)
    mismatches = []
    for relative, expected in artifacts().items():
        path = root / relative
        if not path.is_file() or path.read_text() != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError("FH12 missing/changed frozen registry/configs: " + ", ".join(mismatches))
    configs = {}
    for case in CASES:
        relative = f"config/{case.run_id}.yaml"
        cfg = yaml.safe_load((root / relative).read_text())
        if cfg != build_config(case):
            raise ValueError(f"FH12 resolved config differs from registered case: {relative}")
        configs[case.run_id] = object_sha(cfg)
    if not (root / SOURCE_PLAN).is_file():
        raise FileNotFoundError(root / SOURCE_PLAN)
    return dict(registry_revision=REGISTRY_REVISION, registry_sha256=registry_sha256(),
                config_count=len(configs), config_sha256=configs,
                spec_path=SOURCE_PLAN, spec_sha256=sha256(root / SOURCE_PLAN))


def validate_sources(root, deadline=None):
    """Only source data are inherited; old LP/cue/Teacher assets are not used."""
    root = Path(root)
    recipe_path = root / "assets/mix20h/reference_recipe.json"
    canonical = read_json(recipe_path)["dataset_hashes"]
    splits = {}
    for split, relative in DEFAULT_PATHS.items():
        check_deadline(deadline)
        key = HASH_KEYS[split]
        expected = canonical.get(key)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Missing canonical source SHA for {split}")
        source = root / relative
        info = _source_info(source, split, 20 if split in {"rr", "fr"} else None)
        if info["sha256"] != expected:
            raise ValueError(f"FH12 {split} native data SHA differs from canonical WV3 protocol: {source}")
        splits[split] = dict(path=relative, **info)
    return dict(reference_recipe_sha256=sha256(recipe_path), splits=splits,
                old_lpan_reused=False, old_cue_reused=False, old_teacher_reused=False)


def validate_window(root, server, device, deadline_arg=None):
    path = Path(root) / "work_dir/_fh12" / server / "window.json"
    if not path.is_file():
        raise ValueError("FH12 has no activated window; use tools/fh12_start.sh (or --check-only)")
    window = read_json(path)
    if window.get("campaign_id") != CAMPAIGN_ID or window.get("server_id", window.get("server")) != server:
        raise ValueError("FH12 window belongs to a different campaign/server")
    deadline = window.get("deadline_utc")
    if not deadline:
        raise ValueError("FH12 immutable window has no deadline")
    if deadline_arg is not None and deadline_arg != deadline:
        raise ValueError("FH12 preflight cannot change the window deadline")
    if window.get("device") != device:
        raise ValueError("FH12 preflight device differs from the activated window")
    check_deadline(deadline)
    return window, deadline


def validate_prepared(root, server, sources):
    """Read-only verification; missing assets are reported, changed assets fail."""
    root = Path(root)
    camp = root / "work_dir/_fh12" / server
    expected_files = [camp / "dataset_manifest.json", camp / "lpan_manifest.json"]
    missing = [str(path.relative_to(root)) for path in expected_files if not path.is_file()]
    if missing:
        return dict(complete=False, missing=missing)
    data, lp = (read_json(path) for path in expected_files)
    if data.get("server") != server or data.get("recipe") != RECIPE or lp.get("recipe") != RECIPE:
        raise ValueError("FH12 prepared data have a different server or LP recipe")
    for split, source in sources["splits"].items():
        entry = data["splits"][split]
        if entry.get("sha256") != source["sha256"] or entry.get("count") != source["count"]:
            raise ValueError(f"FH12 prepared {split} differs from canonical source")
        if Path(entry["dataroot"]).resolve() != (root / source["path"]).resolve():
            raise ValueError(f"FH12 prepared {split} points to another source path")
        cache = Path(entry["lpan_path"])
        cache = cache if cache.is_absolute() else root / cache
        if not cache.is_file():
            missing.append(str(cache))
            continue
        if sha256(cache) != entry["lpan_sha256"]:
            raise ValueError(f"FH12 immutable {split} LP cache hash changed")
        if lp["splits"][split].get("lpan_sha256") != entry["lpan_sha256"]:
            raise ValueError(f"FH12 LP and dataset manifests disagree for {split}")
    return dict(complete=not missing, missing=missing,
                dataset_manifest_sha256=sha256(expected_files[0]),
                lpan_manifest_sha256=sha256(expected_files[1]),
                recipe_sha256=canonical_sha(RECIPE), phase_diagnostics=lp["phase_diagnostics"])


def coupled_model_checks(server):
    """Real W112D123 common-weight hashes and step0 functions, entirely on CPU."""
    seed = teacher_for(server).teacher_seed
    previous_torch = torch.get_rng_state().clone()
    previous_python = random.getstate()
    previous_numpy = np.random.get_state()
    gen = torch.Generator().manual_seed(918)
    pan = torch.randn(1, 1, 32, 32, generator=gen)
    ms = torch.randn(1, 8, 8, 8, generator=gen)
    lp = torch.randn(1, 1, 8, 8, generator=gen)
    records, reference, hashes = {}, None, None
    with torch.no_grad():
        for layout in LAYOUTS:
            model, manifest = build_model(layout, 112, [1, 2, 3], seed, role="T")
            model.eval()
            if not manifest["from_scratch"] or manifest["pretrained_aligner_loads"] or manifest["pretrained_backbone_loads"]:
                raise AssertionError("FH12 Teacher is not completely fresh")
            if not manifest["extra_kernel_zero"]:
                raise AssertionError("FH12 extra L/H kernels must initially be zero")
            common = {key: manifest["hashes"][key] for key in ("P", "MS", "input_bias", "body", "A")}
            out = model(pan, ms, lp)
            if reference is None:
                reference = {key: out[key].clone() for key in ("y", "delta", "ms_base")}
                hashes = common
            elif common != hashes or any(not torch.allclose(out[key], reference[key], atol=2e-6, rtol=2e-5) for key in reference):
                raise AssertionError(f"FH12 coupled step0 function or common tensors differ: {layout}")
            records[layout] = dict(params=sum(p.numel() for p in model.parameters()), init_manifest=manifest)
    current_numpy = np.random.get_state()
    unchanged = (torch.equal(previous_torch, torch.get_rng_state()) and previous_python == random.getstate()
                 and previous_numpy[0] == current_numpy[0]
                 and np.array_equal(previous_numpy[1], current_numpy[1])
                 and previous_numpy[2:] == current_numpy[2:])
    if not unchanged:
        raise AssertionError("FH12 model construction consumed training/data RNG")
    return dict(passed=True, width=112, depth=[1, 2, 3], seed=seed,
                rng_preserved=True, all_layouts=records)


def numerical_routing_checks():
    """Execute existing bounded regression cases, not a source-code-only claim."""
    from tools.fh12_model_tests import FH12ModelTests
    names = ["test_teacher_odd_offset_is_a_only_and_no_unet_call",
             "test_student_formula_detaches_and_fixed_denominators",
             "test_edge_is_signed_scharr_two_direction_interior",
             "test_separate_teacher_student_layout_and_gradient_routing",
             "test_extra_frequency_paths_c_gradient_remains_live",
             "test_same_name_shape_coupling_across_depth_and_width"]
    suite = unittest.TestSuite(FH12ModelTests(name) for name in names)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("FH12 numerical routing preflight failed:\n" + stream.getvalue())
    return dict(passed=True, device="cpu", n_tests=result.testsRun, test_names=names,
                log=stream.getvalue())


def _smoke_batch(data, device, batch_size):
    """Read only the required native train patches, never load the whole H5."""
    import h5py
    item = data["splits"]["train"]
    with h5py.File(item["dataroot"], "r") as source, h5py.File(item["lpan_path"], "r") as lp:
        if len(source["pan"]) < batch_size:
            raise ValueError("FH12 CUDA smoke requires the actual complete batch48")
        arrays = [source[key][:batch_size].astype(np.float32) for key in ("gt", "ms", "pan")]
        arrays.append(lp["lpan"][:batch_size].astype(np.float32))
    expected = [(batch_size, 8, 64, 64), (batch_size, 8, 16, 16),
                (batch_size, 1, 64, 64), (batch_size, 1, 16, 16)]
    for arr, shape in zip(arrays, expected):
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"Invalid native train64 smoke batch: {arr.shape} vs {shape}")
    return [torch.from_numpy(a).to(device).mul_(2. / 2047.).sub_(1.) for a in arrays]


def _largest_student(server):
    # Activation-size proxy weights each mirrored stage by its spatial area.
    # W112D121 exceeds W104D122 on servers that include both reserves.
    return max((case for case in cases_for(server) if case.role == "S"),
               key=lambda c: (c.width * (2*c.depth[0] + .5*c.depth[1] + .0625*c.depth[2]),
                              len(LAYOUTS[c.input_layout]), c.width, c.depth))


def training_smoke(server, device, data=None, deadline=None):
    """Disposable real-GPU batch48 T/S backward + AdamW allocation smoke.

    CPU mode is deliberately a small synthetic diagnostic, not a GPU-memory
    certificate. No smoke model/checkpoint/calibration value is reused in runs.
    """
    check_deadline(deadline)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FH12 training smoke requested unavailable CUDA")
    teacher_case, student_case = teacher_for(server), _largest_student(server)
    gpu = dev.type == "cuda"
    if gpu and data is None:
        raise ValueError("FH12 CUDA smoke requires immutable prepared train data")
    batch_size = 48 if gpu else 2
    tw, td = (teacher_case.width, list(teacher_case.depth)) if gpu else (8, [1, 1, 1])
    sw, sd = (student_case.width, list(student_case.depth)) if gpu else (8, [1, 1, 1])
    if gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.monotonic()
    report = dict(execution_device=str(dev), smoke_GPU="executed" if gpu else "not_run",
                  production_batch_and_architecture=gpu, batch_size=batch_size,
                  patch_size=64 if gpu else 32, precision="fp32", disposable_models=True,
                  teacher_run=teacher_case.run_id, student_run=student_case.run_id,
                  student_choice="largest_local_activation_estimate_including_reserves",
                  teacher_width=tw, teacher_depth=td, student_width=sw, student_depth=sd,
                  teacher_updates=2, student_updates=1, optimizer="AdamW", peak_allocated_mb=None)
    teacher = student = optimizer = loss = out = t_out = tensors = None
    try:
        if gpu:
            tensors = _smoke_batch(data, dev, batch_size)
        else:
            gen = torch.Generator().manual_seed(8918)
            tensors = [torch.randn(shape, generator=gen) * .2 for shape in
                       ((2, 8, 32, 32), (2, 8, 8, 8), (2, 1, 32, 32), (2, 1, 8, 8))]
        gt, ms, pan, lp = tensors
        def make_optimizer(model, lr_a):
            return torch.optim.AdamW([dict(params=model.backbone.parameters(), lr=1e-4),
                                      dict(params=model.aligner.parameters(), lr=lr_a)],
                                     betas=(.9, .999), eps=1e-8, weight_decay=.01)
        def check_gradients(model):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
                raise FloatingPointError("FH12 smoke missing/nonfinite gradients")
        teacher, _ = build_model(teacher_case.input_layout, tw, td, teacher_case.seed, role="T")
        teacher.to(dev).train()
        optimizer = make_optimizer(teacher, 1e-5)
        corruption = torch.Generator().manual_seed(teacher_case.seed + 100000)
        t_losses = []
        for update in (0, 1):
            check_deadline(deadline)
            optimizer.zero_grad(set_to_none=True)
            out = teacher(pan, ms, lp)
            loss = teacher_loss(teacher, out, gt, pan, ms, update, corruption)
            if not bool(torch.isfinite(loss["total"])):
                raise FloatingPointError("Nonfinite disposable Teacher smoke loss")
            loss["total"].backward()
            check_gradients(teacher)
            optimizer.step()
            t_losses.append(float(loss["total"].detach()))
        report["teacher_loss"] = t_losses
        # Release Teacher graph and optimizer states, matching frozen KD runtime.
        optimizer.zero_grad(set_to_none=True)
        optimizer = out = loss = None
        teacher.requires_grad_(False).eval()
        teacher_before = state_hash(teacher.state_dict())
        student, _ = build_model(student_case.input_layout, sw, sd, student_case.seed, role="S",
                                  teacher_aligner_state=teacher.aligner.state_dict())
        student.to(dev).train()
        optimizer = make_optimizer(student, 3e-6)
        check_deadline(deadline)
        optimizer.zero_grad(set_to_none=True)
        out = student(pan, ms, lp)
        with torch.no_grad():
            t_out = teacher(pan, ms, lp)
        # Synthetic calibration scalars only exercise the exact loss/routing;
        # they are explicitly not published/reused as actual tau_R/q assets.
        loss = student_losses(out, t_out, gt, .1, torch.full((batch_size,), .5, device=dev))
        if not bool(torch.isfinite(loss["L_U"]) & torch.isfinite(loss["L_A"])):
            raise FloatingPointError("Nonfinite disposable Student smoke loss")
        routed_student_backward(student, loss)
        check_gradients(student)
        optimizer.step()
        report["student_loss"] = float(loss["L_U"].detach())
        if state_hash(teacher.state_dict()) != teacher_before or any(p.grad is not None for p in teacher.parameters()):
            raise AssertionError("Frozen Teacher changed during Student GPU smoke")
        if gpu:
            torch.cuda.synchronize(dev)
            report["peak_allocated_mb"] = torch.cuda.max_memory_allocated(dev) / 2**20
            report["peak_reserved_mb"] = torch.cuda.max_memory_reserved(dev) / 2**20
        check_deadline(deadline)
        report.update(passed=True, teacher_frozen_verified=True, seconds=time.monotonic() - started)
        return report
    finally:
        teacher = student = optimizer = loss = out = t_out = tensors = None
        # Local tensor aliases must also be released before empty_cache.
        if "gt" in locals():
            del gt, ms, pan, lp
        gc.collect()
        if gpu:
            torch.cuda.empty_cache()


def evaluator_smoke(data, deadline=None):
    """One native RR and FR scene, exact production helper functions and DLPan.

    LMS predictions are dependency/formula probes, not model performance and
    never enter the official 20-scene grid or comparison sheet.
    """
    import h5py
    from tools.eval_dlpan import scc_dlpan, psnr_global, ssim_skimage
    from tools.metrics.eval_rr import sam, ergas
    from tools.metrics.q2n import q2n
    from tools.metrics.eval_fr import load_dlpan, imresize_matlab, mtf_filter, _blockproc_uqi
    check_deadline(deadline)
    dlpan_root = os.environ.get("PANCRAFTER_DLPAN", str(ROOT.parent / "DLPan-Toolbox"))
    wald = load_dlpan(dlpan_root)
    with h5py.File(data["splits"]["rr"]["dataroot"], "r") as source:
        a = np.asarray(source["lms"][0], dtype=np.float64).clip(0, 2047).transpose(1, 2, 0)[20:-21, 20:-21]
        b = np.asarray(source["gt"][0], dtype=np.float64).transpose(1, 2, 0)[20:-21, 20:-21]
    rr = dict(ergas=ergas(a, b), scc=scc_dlpan(a, b), psnr=psnr_global(a, b, 2047.),
              sam=sam(a, b), q8=q2n(b, a, 32, 32)[0], ssim=ssim_skimage(a, b, 2047.))
    check_deadline(deadline)
    with h5py.File(data["splits"]["fr"]["dataroot"], "r") as source:
        lms = np.asarray(source["lms"][0], dtype=np.float64).transpose(1, 2, 0)
        pan = np.asarray(source["pan"][0, 0], dtype=np.float64)
    reference_lp = wald.interp23tap(imresize_matlab(pan, 1/4)[..., None], 4)[..., 0]
    prediction = lms.clip(0, 2047)
    dl = 1 - q2n(lms, mtf_filter(prediction, "wv3", 4, wald), 32, 32)[0]
    ds = float(np.mean([abs(_blockproc_uqi(prediction[..., i], pan, 32)
                            - _blockproc_uqi(lms[..., i], reference_lp, 32)) for i in range(8)]))
    fr = dict(d_lambda=float(dl), d_s=ds, hqnr=float((1-dl)*(1-ds)))
    if not all(np.isfinite(float(value)) for value in list(rr.values()) + list(fr.values())):
        raise FloatingPointError("FH12 one-scene official evaluator smoke returned nonfinite metrics")
    check_deadline(deadline)
    wald_path = getattr(wald, "__file__", None)
    return dict(passed=True, execution_device="cpu", n_rr=1, n_fr=1, prediction="LMS diagnostic only",
                official_result=False, rr=rr, fr=fr, dlpan_root=dlpan_root,
                wald_source_sha256=sha256(wald_path) if wald_path else None)


def run_preflight(root, server, device="cuda", deadline_utc=None, check_only=False):
    root = Path(root).resolve()
    if server not in SERVERS or device not in ("cpu", "cuda"):
        raise ValueError("FH12 requires server s1..s5 and explicit cpu/cuda execution")
    # Validate activation before creating any artifact. Read-only checks do not
    # require activation and do not certify training readiness.
    window, deadline = (None, deadline_utc) if check_only else validate_window(root, server, device, deadline_utc)
    report = dict(schema="FH12_PREFLIGHT_v1", campaign_id=CAMPAIGN_ID,
                  method_revision=METHOD_REVISION, server=server, requested_device=device,
                  execution_device=None, checked_at=utcnow(), check_only=bool(check_only),
                  preflight_pass=False, no_global_lock=True)
    try:
        report["registry"] = validate_registry(root)
        report["sources"] = validate_sources(root, deadline)
        if check_only:
            report["prepared_assets"] = validate_prepared(root, server, report["sources"])
            report["status"] = "CHECK_ONLY_READY_FOR_PREFLIGHT" if report["prepared_assets"]["complete"] else "CHECK_ONLY_NEEDS_PREPARATION"
            report["runtime_numerics_executed"] = False
            return report
        check_deadline(deadline)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU preflight fallback")
        report["execution_device"] = device
        report["source_identity"] = source_identity(root)
        report["preflight_code_sha256"] = {
            name: sha256(ROOT / name)
            for name in ("tools/fh12_preflight.py", "tools/fh12_model_tests.py")
        }
        report["window_sha256"] = sha256(root / "work_dir/_fh12" / server / "window.json")
        report["deadline_utc"] = deadline
        report["device_name"] = torch.cuda.get_device_name() if device == "cuda" else "CPU diagnostic"
        # All four source hashes are checked BEFORE cache generation.
        data = prepare_data(root, server, deadline_utc=deadline)
        report["prepared_assets"] = validate_prepared(root, server, report["sources"])
        if not report["prepared_assets"]["complete"]:
            raise ValueError("FH12 immutable LP preparation is incomplete")
        report["prepared_data_identity"] = object_sha(data)
        check_deadline(deadline)
        report["frontend"] = frontend_self_test(device=device)
        check_deadline(deadline)
        report["coupled_init"] = coupled_model_checks(server)
        report["loss_routing"] = numerical_routing_checks()
        check_deadline(deadline)
        report["training_smoke"] = training_smoke(server, device, data=data, deadline=deadline)
        report["evaluator_smoke"] = evaluator_smoke(data, deadline=deadline)
        check_deadline(deadline)
        report["runtime_numerics_executed"] = True
        report["next_teacher"] = teacher_for(server).run_id
        report["next_students"] = [case.run_id for case in cases_for(server) if case.role == "S"][:2]
        report["status"] = "PASS"
        report["preflight_pass"] = True
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, error=str(exc), preflight_pass=False)
        if not check_only:
            atomic_json(root / "work_dir/_fh12" / server / "preflight_report.json", report)
        raise
    atomic_json(root / "work_dir/_fh12" / server / "preflight_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, choices=SERVERS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--deadline-utc")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_preflight(args.root, args.server, args.device, args.deadline_utc, args.check_only)
    except Exception as exc:
        print(json.dumps(dict(status="FAILED", error_type=type(exc).__name__, error=str(exc)), indent=2))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
