"""Disposable train-side context smoke; not training, inference or donor parity.

Compare the central 64 PAN pixels from reference/candidate halos of 128/80
PAN pixels. Use provided scene-wide LP and RR; never regenerate either here.
No predictions, checkpoints, sheets or persistent model weights are written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time

import numpy as np

from .common import atomic_json, file_sha, load_json
from .v3_data import SceneV3


def _difference(reference, candidate, tolerance):
    error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    if reference.shape != candidate.shape or not np.isfinite(error).all():
        raise ValueError("Invalid parity arrays")
    return dict(max_abs=float(error.max()), mean_abs=float(error.mean()),
                tolerance=tolerance, passed=bool(error.max() <= tolerance))


def run(args):
    if args.output.exists():
        raise ValueError("Preserve existing smoke receipts; choose another output")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch.nn import functional as F
    from .brovey import brovey8, config_metadata
    from deploy.pandep_shared.project.src.pan_shared.blocks import ResBlock
    from deploy.pandep_shared.project.src.pan_shared.model import SingleSensorPLHUNet

    package = load_json(args.package_receipt)
    if package.get("state") != "TRANSFER_VERIFIED_NOT_TRAINING_ADMISSION":
        raise ValueError("Verified transfer receipt required")
    root = Path(package["input_root"]).resolve(strict=True)
    mapping_path = root / "canonical_mapping.json"
    if file_sha(mapping_path) != package["files"]["canonical_mapping.json"]["sha256"]:
        raise ValueError("Package mapping changed after extraction")
    scenes = load_json(mapping_path)["scenes"]
    if len(scenes) != 5 or len({s["scene_id"] for s in scenes}) != 5:
        raise ValueError("Expected the five explicitly mapped distinct scenes")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed = 261002
    model = SingleSensorPLHUNet("WV3", width=104, depth=(1, 2, 2), seed=seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Exercise every normally zero-initialized path, not just MS-up or skips.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ResBlock):
                module.out_layers[3].weight.normal_(0, 0.001, generator=generator)
        model.stems["WV3"].weight[:, 1:3].normal_(0, 0.001, generator=generator)
        model.heads["WV3"].weight.normal_(0, 0.001, generator=generator)
    model.to(device=device, dtype=torch.float32).eval()
    result = dict(schema="PANPV_V3_CONTEXT_SMOKE_v1", state="RUNNING",
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  package_receipt_sha256=file_sha(args.package_receipt),
                  implementation_sha256=file_sha(__file__),
                  device=str(device), torch=torch.__version__,
                  gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  model=model.architecture_manifest(), seed=seed,
                  diagnostic_initialization="CANONICAL_PLUS_NONZERO_HEAD_RESBLOCK_OUTPUT_AND_LH_STEM_STD_0.001",
                  trained_or_pretrained=False, donor_ft0_parity_checked=False,
                  model_weights_saved=False, formal_training_started=False,
                  sheet_writes=0, amp=False, tf32=False, precision="float32",
                  input_normalization="2*DN/2047-1 exactly once; no clip",
                  lp_source="provided scene-wide LP; not recomputed",
                  brovey=config_metadata(), central_pan_shape=[64, 64],
                  reference_halo_pan=128, candidate_halo_pan=80,
                  supports=["RR", "NATIVE"], scenes=[])
    atomic_json(args.output, result)
    started = time.monotonic()
    try:
        for entry in scenes:
            relative = entry["file"]
            path = root / relative
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("Scene path leaves verified package")
            expected = package["files"][relative]["sha256"]
            if expected != entry["file_sha256"]:
                raise ValueError("Package/scene SHA disagreement")
            with SceneV3(path, expected_sha256=expected) as scene:
                if scene.scene_id != entry["scene_id"]:
                    raise ValueError("Scene mapping identity mismatch")
                index = scene._index
                valid = ((index[:, 2] == 0) & (index[:, 0] >= 128) & (index[:, 1] >= 128)
                         & (index[:, 0]+192 <= scene.height) & (index[:, 1]+192 <= scene.width))
                choices = np.flatnonzero(valid)
                if not len(choices):
                    raise ValueError("No fixed training patch has reference context")
                raw_id = None
                # Boundary-derived masks may be invalid even when the central
                # frozen patch is valid. Select by input support only, before
                # either method is run; never select using model outputs.
                for choice in choices:
                    row, col = map(int, index[choice, :2])
                    reference = [scene.read_region("rr", row-128, col-128, 320, 320),
                                 scene.read_region("native", row-32, col-32, 80, 80)]
                    if all(mask.all() for sample in reference for mask in sample["masks"].values()):
                        raw_id = int(choice)
                        break
                if raw_id is None:
                    raise ValueError("No train patch has fully valid reference inputs")
                record = dict(scene_id=scene.scene_id, source_sha256=scene.sha256,
                              raw_index=raw_id, canonical_ms_row_col=[row, col], tests=[],
                              selection_rule="FIRST_PROVIDER_TRAIN_PATCH_WITH_FULLY_VALID_REFERENCE_INPUTS")
                for scope in ("rr", "native"):
                    center, halos = (64, (128, 80)) if scope == "rr" else (16, (32, 20))
                    predictions, broveys, residual_max = [], [], []
                    for halo in halos:
                        sample = scene.read_region(scope, row-halo, col-halo,
                                                   center+2*halo, center+2*halo, normalize=False)
                        if any(not mask.all() for mask in sample["masks"].values()):
                            raise ValueError("Smoke context includes invalid input support")
                        tensors = [torch.from_numpy(sample[key]*np.float32(2/2047)-np.float32(1))
                                   .unsqueeze(0).to(device) for key in ("pan", "ms", "lpan")]
                        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                            output = model("WV3", *tensors)
                            base = F.interpolate(tensors[1], scale_factor=4, mode="bicubic", align_corners=False)
                            hpan = halo if scope == "rr" else 4*halo
                            core = output[0, :, hpan:hpan+64, hpan:hpan+64]
                            residual_max.append(float((core-base[0, :, hpan:hpan+64, hpan:hpan+64]).abs().max()))
                            predictions.append(core.cpu().numpy())
                        raw = brovey8(sample["pan"][0], sample["ms"])
                        broveys.append(raw[:, hpan:hpan+64, hpan:hpan+64])
                    if min(residual_max) <= 1e-6:
                        raise ValueError("Degenerate zero-head/MS-up-only diagnostic")
                    record["tests"].append(dict(scope=scope, model_normalized=_difference(*predictions, 1e-5),
                                                 brovey_dn=_difference(*broveys, 1e-3),
                                                 nontrivial_residual_max=residual_max))
                result["scenes"].append(record)
                atomic_json(args.output, result)
                print(scene.scene_id, record["tests"], flush=True)
        passed = all(test[method]["passed"] for scene in result["scenes"] for test in scene["tests"]
                     for method in ("model_normalized", "brovey_dn"))
        result.update(state="PASS" if passed else "FAIL_CONTEXT_PARITY", elapsed_seconds=time.monotonic()-started,
                      whole_scene_or_edge_parity_proven=False, full_campaign_gate_passed=False)
        atomic_json(args.output, result)
        return 0 if passed else 1
    except Exception as error:
        result.update(state="FAILED_EXCEPTION", error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
