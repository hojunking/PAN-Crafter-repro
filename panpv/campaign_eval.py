"""Frozen, context-aware PANPV v3 RR holdout / native FR *tile* evaluation.

This is not a benchmark-RR20/FR20 adapter and does not produce full-scene
claims. RR retains the existing source-anchored valid32 window-macro protocol.
Native FR uses the same input-selected held-out locations, a WV2 MTF, and
context-generated interp23 LMS. No audit value participates in training or
checkpoint selection. Supplied dataset Python is never executed.
"""
from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path
import time
import warnings

import h5py
import numpy as np

from .common import atomic_json, canonical_sha, file_sha, load_json
from .evaluation import PROTOCOL as RR_PROTOCOL, RRAuditAccumulator, valid_windows
from .v3_data import SceneV3


METHODS = ("BROVEY8", "ZS_WV3", "SC_SCENE", "FT_WV3_SCENE")
RR_HALO = 80  # PAN pixels == canonical MS pixels at reduced resolution.
FR_HALO = 128  # PAN pixels: NN context plus MTF20 / resize+interp23 context.
FR_PROTOCOL = {
    "id": "PANPV_WV2_NATIVE_AUDIT256_GLOBAL32_TILE_MACRO_v1",
    "scope": "FR_NATIVE_TILES_AT_FROZEN_SPATIAL_AUDIT_LOCATIONS_NOT_FULL_SCENE",
    "target": "none; native HRMS ground truth does not exist",
    "ratio": 4, "sensor": "WV2", "gnyq": [0.35] * 7 + [0.27],
    "tile_pan_hw": [256, 256], "inference_context_pan": FR_HALO,
    "window_size": 32, "window_anchor": "original_source_PAN_origin_0",
    "incomplete_windows": "exclude_identically_before_all_method_metrics",
    "lms": "interp23tap(native_MS,4)_from_extended_canonical_grid_NOT_author_LMS",
    "mtf": "pinned_genmtf_matlab_41_taps_filter_extended_prediction_before_crop",
    "d_lambda": "1-Q8(LMS,MTF(pred)); block32 shift32 on complete source-grid windows",
    "low_pan": "interp23tap(imresize_matlab(extended_native_PAN,0.25),4)",
    "d_s": "mean_bands(abs(mean_window_UQI(pred,PAN)-mean_window_UQI(LMS,lowPAN)))",
    "hqnr": "(1-d_lambda)*(1-d_s); no extra score clipping",
    "aggregation": "arithmetic_mean_of_per_tile_values_NOT_global_scene_metric",
    "reprojection_sam": "MTF(pred)[2::4,2::4] versus nativeMS; full central64MS; degrees",
    "pan_scc": "pinned_Sobel_zero_boundary_cosine(pred,8band_repeated_PAN); diagnostic",
    "analysis": "clip_only_prediction_DN_to_0_2047; no rounding_except_Q8",
    "mask": "all_input_context_valid; no output-derived mask; no shift/warp at evaluation",
    "jqm": None, "jqm_reason": "WV2 practical-data JQM variant not independently validated",
    "undefined": "null_with_count; any undefined tile makes its aggregate null",
}


def _mask_digest(sample):
    digest = hashlib.sha256()
    for key, value in sorted(sample["masks"].items()):
        array = np.asarray(value, dtype=np.uint8)
        digest.update(key.encode() + repr(array.shape).encode() + array.tobytes())
    return digest.hexdigest()


def _context(scene, entry, scope):
    halo = RR_HALO if scope == "rr" else FR_HALO // 4
    row, col = entry["row"], entry["col"]
    sample = scene.read_region(scope, row-halo, col-halo, 64+2*halo, 64+2*halo)
    # A mask on only the centre cannot make invalid model/filter context valid.
    # Fail rather than replace invalid DN, infer NoData, or silently drop tiles.
    if any(not mask.all() for mask in sample["masks"].values()):
        raise ValueError(f"Invalid {scope} context at frozen audit index {entry['raw_index']}")
    return sample


def _context_mask_only(scene, entry, scope):
    """Freeze bounded mask regions without rereading multi-GB image bands.

    The independently verified source file SHA binds all image bytes; mask
    reads here still validate every dependency pixel used by any method.
    """
    halo = RR_HALO if scope == "rr" else FR_HALO // 4
    r, c, size = entry["row"]-halo, entry["col"]-halo, 64+2*halo
    if min(r, c) < 0 or r+size > scene.height or c+size > scene.width:
        raise ValueError("Frozen audit context leaves the source scene")
    rectangles = ({"pan": (r, c, size), "target": (r, c, size),
                   "ms": (r//4, c//4, size//4), "lpan": (r//4, c//4, size//4)}
                  if scope == "rr" else {"pan": (4*r, 4*c, 4*size), "ms": (r, c, size),
                                         "lpan": (r, c, size), "common_ms": (r, c, size)})
    masks = {}
    for key, (row, col, side) in rectangles.items():
        name = f"{scope}/{key}_valid" if key != "common_ms" else "native/common_valid_ms"
        value = scene._handle()[name][0, :, row:row+side, col:col+side]
        if value.shape != (1, side, side) or not np.all(value == 1):
            raise ValueError(f"Invalid {scope} context at frozen audit index {entry['raw_index']}")
        masks[key] = value.astype(bool)
    scene._check_identity()
    return dict(masks=masks)


def implementation_identity():
    import scipy
    import skimage
    from deploy.pandep_shared.project.src.pan_shared import metrics
    paths = [Path(__file__), Path(__file__).with_name("evaluation.py"),
             Path(__file__).with_name("v3_data.py"), Path(__file__).with_name("brovey.py")]
    metric_dir = Path(metrics.__file__).parent
    paths.extend(sorted(metric_dir.glob("*.py")))
    return dict(files={str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha(path)
                       for path in paths}, numpy=np.__version__, scipy=scipy.__version__,
                skimage=skimage.__version__)


def freeze_protocol(scene, output_path=None):
    """Freeze ALL provider audit coordinates and input masks before inference.

    This is deliberately not a random re-split or a quality-based ROI picker.
    Diagnostics use the first provider audit patch in each audit block; final
    evaluation covers every provider audit patch. All contexts must be valid.
    """
    if not isinstance(scene, SceneV3) or not scene.sha256 or scene.expected_sha256 != scene.sha256:
        raise ValueError("Externally SHA-bound SceneV3 required")
    summary = scene.summary()
    entries, seen_blocks = [], set()
    for raw in np.flatnonzero(scene._index[:, 2] == 2):
        row, col, _, block = map(int, scene._index[raw])
        entry = dict(raw_index=int(raw), row=row, col=col, block_id=block)
        rr, fr = _context_mask_only(scene, entry, "rr"), _context_mask_only(scene, entry, "native")
        entry["rr_origin"] = [scene.source_row+row, scene.source_col+col]
        entry["fr_origin"] = [4*(scene.source_row+row)+scene.pan_offset_xy[1],
                              4*(scene.source_col+col)+scene.pan_offset_xy[0]]
        entry["rr_mask_sha256"], entry["fr_mask_sha256"] = _mask_digest(rr), _mask_digest(fr)
        entry["diagnostic_fr"] = block not in seen_blocks
        entry["rr_valid32_windows"] = len(list(valid_windows(np.ones((64, 64), bool), entry["rr_origin"])))
        entry["fr_valid32_windows"] = len(list(valid_windows(np.ones((256, 256), bool), entry["fr_origin"])))
        seen_blocks.add(block)
        entries.append(entry)
    if not entries or any(not e["rr_valid32_windows"] or not e["fr_valid32_windows"] for e in entries):
        raise ValueError("No complete frozen audit windows")
    result = dict(schema="PANPV_V3_FROZEN_TILE_PROTOCOL_v1", scene_id=scene.scene_id,
                  source_sha256=scene.sha256, patch_index_sha256=summary["patch_index_sha256"],
                  blocks_sha256=summary["blocks_sha256"], profile_sha256=summary["profile_sha256"],
                  rr_protocol=RR_PROTOCOL, fr_protocol=FR_PROTOCOL, rr_halo_pan=RR_HALO,
                  fr_halo_pan=FR_HALO, audit=entries, implementation=implementation_identity(),
                  native_output_scope="FR_NATIVE_SPATIAL_HOLDOUT_TILES", full_scene=False,
                  diagnostics_fr_selection="FIRST_PROVIDER_AUDIT_PATCH_PER_BLOCK_BEFORE_INFERENCE",
                  rr_benchmark_parity=False, input_adaptation=True,
                  final_uncompressed_prediction_bytes=len(entries)*8*(64*64+256*256)*4)
    result["protocol_sha256"] = canonical_sha(result)
    if output_path is not None:
        path = Path(output_path)
        if path.exists() and load_json(path) != result:
            raise ValueError("Refusing to change an existing frozen protocol")
        if not path.exists():
            atomic_json(path, result)
    return result


def _validate_protocol(scene, protocol):
    if protocol.get("protocol_sha256") != canonical_sha({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("Corrupt frozen protocol")
    summary = scene.summary()
    if (protocol.get("scene_id") != scene.scene_id or protocol.get("source_sha256") != scene.sha256
            or scene.expected_sha256 != scene.sha256 or not scene.sha256
            or protocol.get("patch_index_sha256") != summary["patch_index_sha256"]
            or protocol.get("blocks_sha256") != summary["blocks_sha256"]
            or protocol.get("profile_sha256") != summary["profile_sha256"]
            or protocol.get("implementation") != implementation_identity()
            or protocol.get("rr_protocol") != RR_PROTOCOL or protocol.get("fr_protocol") != FR_PROTOCOL
            or protocol.get("rr_halo_pan") != RR_HALO or protocol.get("fr_halo_pan") != FR_HALO):
        raise ValueError("Frozen protocol differs from current source/implementation")
    expected = [(int(raw), *map(int, scene._index[raw, [0, 1, 3]]))
                for raw in np.flatnonzero(scene._index[:, 2] == 2)]
    actual = [(e["raw_index"], e["row"], e["col"], e["block_id"]) for e in protocol["audit"]]
    if actual != expected:
        raise ValueError("Frozen audit coordinates differ from provider list")
    seen_blocks = set()
    for e in protocol["audit"]:
        if (e["rr_origin"] != [scene.source_row+e["row"], scene.source_col+e["col"]]
                or e["fr_origin"] != [4*(scene.source_row+e["row"])+scene.pan_offset_xy[1],
                                       4*(scene.source_col+e["col"])+scene.pan_offset_xy[0]]
                or e["diagnostic_fr"] != (e["block_id"] not in seen_blocks)):
            raise ValueError("Frozen source origin/diagnostic selection differs")
        seen_blocks.add(e["block_id"])


def tensor_sha256(model):
    """Content identity, explicitly different from a safetensors file SHA."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        if not np.isfinite(array).all():
            raise FloatingPointError("Nonfinite evaluated model state")
        digest.update(name.encode()+repr(array.shape).encode()+array.dtype.str.encode()+array.tobytes())
    return digest.hexdigest()


def _predict(sample, model, device):
    if model is None:
        from .brovey import brovey8
        return brovey8(sample["pan"][0], sample["ms"])
    import torch
    device = torch.device(device)
    tensors = [torch.from_numpy(sample[key]*np.float32(2/2047)-np.float32(1)).unsqueeze(0).to(device)
               for key in ("pan", "ms", "lpan")]
    with torch.autocast(device_type=device.type, enabled=False):
        value = model("WV3", *tensors)
    expected = (1, 8, *sample["pan"].shape[-2:])
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected or not torch.isfinite(value).all():
        raise ValueError("Invalid/nonfinite neural inference output")
    return ((value[0].float()+1)*np.float32(2047/2)).cpu().numpy()


def _timed_predict(sample, model, device):
    """Wall time for this complete prediction call, not a kernel-only claim.

    CUDA work before the call is excluded; CPU normalization, H2D transfer,
    model frontend/forward and the returned host copy are included. GDAL is
    measured as its actual complete CPU call. No allocator/peak counters reset.
    """
    import torch
    cuda = model is not None and torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(torch.device(device))
    start = time.perf_counter()
    result = _predict(sample, model, device)
    if cuda:
        torch.cuda.synchronize(torch.device(device))
    seconds = time.perf_counter()-start
    if not np.isfinite(seconds) or seconds < 0:
        raise RuntimeError("Invalid measured prediction duration")
    return result, float(seconds)


def _core(array, halo, size):
    return array[..., halo:halo+size, halo:halo+size]


def _metric_mask(size, origin):
    mask = np.zeros((1, size, size), dtype=np.uint8)
    for region in valid_windows(np.ones((size, size), bool), origin):
        mask[(0, *region)] = 1
    return mask


def fr_tile_metrics(prediction, sample, origin, *, halo=FR_HALO):
    """Measure a native256 centre with real surrounding filtering context.

    Never run MTF, interp23 or PAN down/up sampling on a boundaryless 32 tile.
    Complete 32 windows are anchored to original source PAN, not local zero.
    """
    from deploy.pandep_shared.project.src.pan_shared.metrics.eval_fr import (
        GNYQ_TABLE, _blockproc_uqi, imresize_matlab, mtf_filter,
    )
    from deploy.pandep_shared.project.src.pan_shared.metrics.eval_rr import sam
    from deploy.pandep_shared.project.src.pan_shared.metrics.extras import scc_dlpan
    from deploy.pandep_shared.project.src.pan_shared.metrics.interp23 import interp23tap
    from deploy.pandep_shared.project.src.pan_shared.metrics.q2n import q2n
    if halo < 128 or halo % 4 or GNYQ_TABLE["WV2"] != FR_PROTOCOL["gnyq"]:
        raise ValueError("Insufficient native metric context or changed WV2 MTF")
    size = 256+2*halo
    prediction = np.asarray(prediction)
    if (prediction.shape != (8, size, size) or sample["pan"].shape != (1, size, size)
            or sample["ms"].shape != (8, size//4, size//4)
            or not np.isfinite(prediction).all()
            or not np.isfinite(sample["pan"]).all() or not np.isfinite(sample["ms"]).all()
            or not all(mask.all() for mask in sample["masks"].values())):
        raise ValueError("FR requires fully valid native input context and float eight-band prediction")
    pred = np.clip(prediction.astype(np.float64), 0, 2047).transpose(1, 2, 0)
    pan = sample["pan"][0].astype(np.float64)
    native_ms = sample["ms"].astype(np.float64).transpose(1, 2, 0)
    lms = interp23tap(native_ms, 4)
    filtered = mtf_filter(pred, "wv2", 4)
    low_pan = interp23tap(imresize_matlab(pan, .25)[..., None], 4)[..., 0]
    windows = list(valid_windows(np.ones((256, 256), bool), origin))
    if not windows:
        raise ValueError("No native FR windows")
    # Complete square windows tile a single rectangle; q2n performs only the
    # same block32 windows (dimensions are exact multiples, no padded blocks).
    r0, c0 = windows[0][0].start+halo, windows[0][1].start+halo
    r1, c1 = windows[-1][0].stop+halo, windows[-1][1].stop+halo
    region = np.s_[r0:r1, c0:c1]
    fused, reference, p, lp = pred[region], lms[region], pan[region], low_pan[region]
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        dl = float(1-q2n(reference, filtered[region], 32, 32)[0])
        delta = np.array([_blockproc_uqi(fused[..., b], p, 32)-_blockproc_uqi(reference[..., b], lp, 32)
                          for b in range(8)])
        ds = float(np.abs(delta).mean())
        reprojection = filtered[halo+2:halo+256:4, halo+2:halo+256:4]
        truth = native_ms[halo//4:halo//4+64, halo//4:halo//4+64]
        values = dict(hqnr=(1-dl)*(1-ds), d_s=ds, d_lambda=dl,
                      reprojection_sam=sam(reprojection, truth),
                      pan_scc=scc_dlpan(fused, np.repeat(p[..., None], 8, axis=2)))
    return dict(metrics={k: float(v) if np.isfinite(v) else None for k, v in values.items()},
                valid_windows=len(windows), evaluated_pixels=len(windows)*1024,
                signed_ds=[float(v) if np.isfinite(v) else None for v in delta])


class FRTileAccumulator:
    keys = ("hqnr", "d_s", "d_lambda", "reprojection_sam", "pan_scc")

    def __init__(self):
        self.rows = []

    def add(self, prediction, sample, origin, *, sample_id):
        if any(row["raw_index"] == sample_id for row in self.rows):
            raise ValueError("Duplicate native FR tile")
        self.rows.append(dict(raw_index=sample_id, **fr_tile_metrics(prediction, sample, origin)))

    def result(self):
        undefined = {key: sum(row["metrics"][key] is None for row in self.rows) for key in self.keys}
        values = {key: float(np.mean([row["metrics"][key] for row in self.rows]))
                  if self.rows and not undefined[key] else None for key in self.keys}
        return dict(protocol=FR_PROTOCOL, metrics=dict(values, jqm=None), tiles=len(self.rows),
                    per_tile=self.rows, undefined_tile_counts=undefined,
                    valid_windows=sum(row["valid_windows"] for row in self.rows),
                    evaluated_pixels=sum(row["evaluated_pixels"] for row in self.rows),
                    full_scene=False, native_hrms_gt=False)


def evaluate_scene(scene, protocol, *, method, output_dir, model=None, device="cuda",
                   checkpoint_sha256=None, final=False):
    """Atomic, content-bound results; final-only raw float32 tiled H5 output.

    A diagnostic emits only JSON (in particular no SC-step0/bicubic output
    images). Resume reuses complete, hash-verified reports, never partial H5.
    The caller owns checkpoint choice; this function never selects a model.
    """
    import contextlib
    import torch
    from deploy.pandep_shared.project.src.pan_shared.evaluate import preserved_inference
    if method not in METHODS or type(final) is not bool or (method == "BROVEY8") != (model is None):
        raise ValueError("Explicit supported method and matching neural/Brovey input required")
    if checkpoint_sha256 is not None and (not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256)):
        raise ValueError("checkpoint_sha256 must be a lowercase full SHA256")
    _validate_protocol(scene, protocol)
    if model is not None and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError("PANPV neural evaluation requires TF32 disabled")
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(Path(scene.path).parent):
        raise ValueError("Evaluation output must not be inside the source-data directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_sha = tensor_sha256(model) if model is not None else None
    parameters = sum(p.numel() for p in model.parameters()) if model is not None else 0
    hardware = (torch.cuda.get_device_name(torch.device(device))
                if model is not None and torch.device(device).type == "cuda"
                else "CPU: " + (platform.processor() or platform.machine()))
    from .brovey import config_metadata
    identity = dict(method=method, scene_id=scene.scene_id, source_sha256=scene.sha256,
                    protocol_sha256=protocol["protocol_sha256"], final=final,
                    model_tensor_sha256=state_sha, declared_checkpoint_sha256=checkpoint_sha256,
                    brovey=config_metadata() if model is None else None,
                    device=str(device) if model is not None else "cpu_GDAL", torch=torch.__version__,
                    parameters=parameters, hardware=hardware)
    report_path = output_dir/"report.json"
    if report_path.exists():
        cached = load_json(report_path)
        sealed = {k: v for k, v in cached.items() if k != "payload_sha256"}
        if (cached.get("payload_sha256") != canonical_sha(sealed) or cached.get("identity") != identity
                or cached.get("status") != "COMPLETE"):
            raise ValueError("Refusing mismatched/corrupt completed evaluation report")
        for item in cached.get("output_files", []):
            path = output_dir/item["name"]
            if path.parent != output_dir or not path.is_file() or file_sha(path) != item["sha256"]:
                raise ValueError("Corrupt/missing cached prediction output")
        return cached
    raw_path, partial = output_dir/"predictions.h5", output_dir/"predictions.h5.partial"
    if raw_path.exists() or partial.exists():
        # Keep interrupted outputs rather than overwrite large valuable files.
        suffix = 0
        while (output_dir/f"interrupted_{suffix:03d}").exists():
            suffix += 1
        recovery = output_dir/f"interrupted_{suffix:03d}"
        recovery.mkdir()
        for path in (raw_path, partial):
            if path.exists():
                path.rename(recovery/path.name)
    rr, fr = RRAuditAccumulator(), FRTileAccumulator()
    timer = time.monotonic()
    count, fr_count = len(protocol["audit"]), 0
    inference = dict(rr_seconds=0.0, fr_seconds=0.0, rr_calls=0, fr_calls=0,
                     scope=("NORMALIZATION_H2D_PLH_FRONTEND_FORWARD_D2H_WALL_SECONDS_INCLUDING_TILE_HALOS"
                            if model is not None else "GDAL_COMPLETE_CPU_CALL_WALL_SECONDS_INCLUDING_TILE_HALOS"),
                     excludes="H5 reads/writes, metric computation, source/checkpoint hashing",
                     pure_kernel_latency=False)
    holder = h5py.File(partial, "x") if final else contextlib.nullcontext(None)
    context = preserved_inference(model) if model is not None else contextlib.nullcontext()
    with holder as stored, context:
        if final:
            stored.attrs.update(schema="PANPV_RAW_FLOAT32_NATIVE_TILES_v1", complete=False,
                                identity_sha256=canonical_sha(identity), clipped=False,
                                axis_order="NCHW", full_scene=False)
            stored.create_dataset("raw_index", data=np.array([e["raw_index"] for e in protocol["audit"]], dtype="int64"))
            stored.create_dataset("canonical_ms_row_col", data=np.array([[e["row"], e["col"]] for e in protocol["audit"]], dtype="int64"))
            for scope, size in (("rr", 64), ("fr", 256)):
                stored.create_dataset(scope+"/prediction", (count, 8, size, size), dtype="float32", chunks=(1, 8, size, size))
                stored.create_dataset(scope+"/valid", (count, 1, size, size), dtype="uint8", chunks=(1, 1, size, size), compression="gzip")
                stored.create_dataset(scope+"/metric_valid32", (count, 1, size, size), dtype="uint8", chunks=(1, 1, size, size), compression="gzip")
        for position, entry in enumerate(protocol["audit"]):
            sample = _context(scene, entry, "rr")
            if _mask_digest(sample) != entry["rr_mask_sha256"]:
                raise ValueError("Frozen RR context mask changed")
            pred, seconds = _timed_predict(sample, model, device)
            inference["rr_seconds"] += seconds
            inference["rr_calls"] += 1
            center = _core(pred, RR_HALO, 64)
            mask = _core(sample["masks"]["target"] & sample["masks"]["pan"], RR_HALO, 64)
            rr.add(center, _core(sample["target"], RR_HALO, 64), mask,
                   sample_id=entry["raw_index"], origin=entry["rr_origin"])
            if final:
                stored["rr/prediction"][position], stored["rr/valid"][position] = center, mask
                stored["rr/metric_valid32"][position] = _metric_mask(64, entry["rr_origin"])
            if final or entry["diagnostic_fr"]:
                sample = _context(scene, entry, "native")
                if _mask_digest(sample) != entry["fr_mask_sha256"]:
                    raise ValueError("Frozen native FR context mask changed")
                pred, seconds = _timed_predict(sample, model, device)
                inference["fr_seconds"] += seconds
                inference["fr_calls"] += 1
                fr.add(pred, sample, entry["fr_origin"], sample_id=entry["raw_index"])
                fr_count += 1
                if final:
                    stored["fr/prediction"][position] = _core(pred, FR_HALO, 256)
                    stored["fr/valid"][position] = _core(sample["masks"]["pan"], FR_HALO, 256)
                    stored["fr/metric_valid32"][position] = _metric_mask(256, entry["fr_origin"])
            if (position+1) % 100 == 0:
                print(f"PANPV evaluation {scene.scene_id} {method}: RR {position+1}/{count}, FR {fr_count}", flush=True)
        expected_fr = count if final else sum(e["diagnostic_fr"] for e in protocol["audit"])
        if rr.samples != count or fr_count != expected_fr:
            raise ValueError("Incomplete evaluation; not publishing a completion report")
        if model is not None and tensor_sha256(model) != state_sha:
            raise ValueError("Evaluated model changed during inference")
        if final:
            stored.attrs["complete"] = True
            stored.flush()
    output_files = []
    if final:
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        partial.rename(raw_path)
        output_files.append(dict(name=raw_path.name, path=str(raw_path), sha256=file_sha(raw_path), bytes=raw_path.stat().st_size))
    rr_result, fr_result = rr.result(), fr.result()
    result = dict(schema="PANPV_PRIVATE_TILE_EVALUATION_v1", status="COMPLETE", identity=identity,
                  rr=rr_result, fr=fr_result, eval_seconds=time.monotonic()-timer,
                  infer_seconds=inference["rr_seconds"]+inference["fr_seconds"], inference=inference,
                  parameters=parameters, params_m=parameters/1_000_000, hardware=hardware,
                  report_path=str(report_path), output_files=output_files, full_scene=False,
                  official_benchmark=False, input_adaptation=True,
                  HQNR=fr_result["metrics"]["hqnr"], SCC=rr_result["metrics"]["scc"],
                  ERGAS=rr_result["metrics"]["ergas"])
    result["payload_sha256"] = canonical_sha(result)
    atomic_json(report_path, result)
    print(f"PANPV {scene.scene_id} {method} HQNR={result['HQNR']} SCC={result['SCC']} ERGAS={result['ERGAS']}", flush=True)
    return result
