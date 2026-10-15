"""Independent, bounded-window verification of the delivered v3 derived arrays.

This implementation does not import or execute the delivered generation code.
It independently derives the declared taps and reads at most 128x128 source
pixels per band per probe. Padding is mapped at the physical scene boundary,
never at a probe/tile boundary. No source file is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


PROFILE_SHA256 = {
    "rr_profile": "65d6782362c57a0abac18ebccadde73baf816fd85cf4d69db195c98dc9926820",
    "lp_profile": "fd1b492d5d585921c186fc2c4babe36b490c618a44aa9a29bb86393d846148d0",
}
PATHS = {
    "rr/pan": ("native/pan", "rr_pan", 1),
    "rr/ms": ("native/ms", "rr_ms", 8),
    "native/lpan": ("native/pan", "lp", 1),
    "rr/lpan": ("rr/pan", "lp", 1),
}
MAX_OUTPUT_SIDE = 16
MAX_READ_SIDE = 128


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def independent_taps(kind, band=0):
    """Return offsets from 4*n and independently calculated float64 weights."""
    if kind == "lp":
        offsets = np.arange(-18, 23, dtype=np.int64)
        weights = np.exp(-0.5 * ((offsets - 2) / 1.98) ** 2)
        padding = "replicate"
    elif kind in ("rr_pan", "rr_ms"):
        if kind == "rr_ms" and (type(band) is not int or not 0 <= band < 8):
            raise ValueError("RR MS requires band 0..7")
        gain = 0.11 if kind == "rr_pan" else (0.27 if band == 7 else 0.35)
        sigma = 4 * math.sqrt(-2 * math.log(gain)) / math.pi
        radius = math.ceil(3.5 * sigma)
        offsets = np.arange(math.floor(1.5-radius), math.ceil(1.5+radius)+1, dtype=np.int64)
        weights = np.exp(-0.5 * ((offsets - 1.5) / sigma) ** 2)
        padding = "reflect"
    else:
        raise ValueError("Unknown declared filter kind")
    weights /= weights.sum(dtype=np.float64)
    return offsets, weights, padding


def boundary_indices(indices, length, padding):
    """Map global source indices: reflect excludes the edge, replicate repeats it."""
    if type(length) is not int or length <= 0:
        raise ValueError("Positive physical source extent required")
    indices = np.asarray(indices, dtype=np.int64)
    if padding == "replicate":
        return np.clip(indices, 0, length-1)
    if padding != "reflect":
        raise ValueError("Unknown physical-boundary padding")
    if length == 1:
        return np.zeros_like(indices)
    phase = indices % (2 * (length-1))
    return np.where(phase < length, phase, 2 * (length-1)-phase)


def _window(window, shape):
    if (not isinstance(window, (list, tuple)) or len(window) != 4
            or any(type(v) is not int for v in window)):
        raise ValueError("Window must be integer [row, col, height, width]")
    row, col, height, width = window
    if (min(row, col) < 0 or not 0 < height <= MAX_OUTPUT_SIDE
            or not 0 < width <= MAX_OUTPUT_SIDE
            or row+height > shape[0] or col+width > shape[1]):
        raise ValueError("Output window exceeds scene or bounded probe size")
    return row, col, height, width


def recompute_window(dataset, window, kind):
    """Read one NCHW source dataset and recompute a global output rectangle.

    Supports h5py datasets and test array objects with shape and __getitem__.
    Accumulation is columns first, then rows, with float64 intermediates.
    The final cast alone rounds to the delivery float32 representation.
    """
    shape = tuple(dataset.shape)
    channels = 8 if kind == "rr_ms" else 1
    if (len(shape) != 4 or shape[:2] != (1, channels)
            or min(shape[-2:]) <= 0 or any(v % 4 for v in shape[-2:])):
        raise ValueError("Source must be ratio-four NCHW with registered band count")
    height, width = shape[-2:]
    row, col, out_h, out_w = _window(window, (height//4, width//4))
    result = np.empty((channels, out_h, out_w), dtype=np.float32)
    for band in range(channels):
        offsets, weights, padding = independent_taps(kind, band)
        iy = boundary_indices(4*np.arange(row, row+out_h)[:, None]+offsets, height, padding)
        ix = boundary_indices(4*np.arange(col, col+out_w)[:, None]+offsets, width, padding)
        y0, y1, x0, x1 = int(iy.min()), int(iy.max())+1, int(ix.min()), int(ix.max())+1
        if max(y1-y0, x1-x0) > MAX_READ_SIDE:
            raise ValueError("Source read exceeds the independent verifier memory bound")
        source = np.asarray(dataset[0, band, y0:y1, x0:x1], dtype=np.float64)
        if source.shape != (y1-y0, x1-x0) or not np.isfinite(source).all():
            raise ValueError("Incomplete or nonfinite bounded source read")
        columns = np.zeros((y1-y0, out_w), dtype=np.float64)
        for tap, weight in enumerate(weights):
            columns += weight * source[:, ix[:, tap]-x0]
        output = np.zeros((out_h, out_w), dtype=np.float64)
        for tap, weight in enumerate(weights):
            output += weight * columns[iy[:, tap]-y0, :]
        result[band] = output.astype(np.float32)
    return result


def probe_windows(shape):
    """Corners, physical edges, interior and generation/chunk seam crossings."""
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Positive output HW required")
    height, width = map(int, shape)
    size = min(MAX_OUTPUT_SIDE, height, width)
    mid_y, mid_x = (height-size)//2, (width-size)//2
    origins = {(0, 0), (0, width-size), (height-size, 0), (height-size, width-size),
               (0, mid_x), (height-size, mid_x), (mid_y, 0), (mid_y, width-size),
               (mid_y, mid_x)}
    # These output-grid boundaries include original generation 256/1024-pixel
    # boundaries, HDF chunk boundaries, and later larger tile boundaries.
    for seam in (64, 128, 256, 512, 1024):
        if size//2 <= seam <= height-size//2:
            origins.add((min(seam-size//2, height-size), mid_x))
        if size//2 <= seam <= width-size//2:
            origins.add((mid_y, min(seam-size//2, width-size)))
        if seam <= min(height, width)-size//2:
            origins.add((max(0, seam-size//2), max(0, seam-size//2)))
    return [[y, x, size, size] for y, x in sorted(origins)]


def compare_float32(actual, reference):
    """Two float32 ULPs per element; zero and nonfinite cases are explicit."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.dtype != np.float32 or reference.dtype != np.float32:
        raise ValueError("Matching float32 delivery/reference arrays required")
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite:
        return dict(finite=False, passed=False, elements=int(actual.size),
                    max_abs_dn=None, squared_error_sum=None, rmse_dn=None,
                    max_ulp_error=None, failed_elements=int(actual.size),
                    actual_min=None, actual_max=None, reference_min=None, reference_max=None)
    magnitude = np.maximum(np.abs(actual), np.abs(reference)).astype(np.float64)
    # np.spacing(float32.max) overflows to infinity; derive the finite binary
    # step in float64 instead, with the subnormal/zero step fixed at 2**-149.
    _, exponent = np.frexp(magnitude)
    spacing = np.maximum(np.exp2(exponent-24), np.exp2(-149.))
    spacing[magnitude == 0] = np.exp2(-149.)
    error = np.abs(actual.astype(np.float64)-reference.astype(np.float64))
    failed = int(np.count_nonzero(error > 2*spacing))
    squared = float(np.square(error).sum(dtype=np.float64))
    return dict(finite=True, passed=failed == 0, elements=int(actual.size),
                max_abs_dn=float(error.max()), squared_error_sum=squared,
                rmse_dn=math.sqrt(squared/actual.size), max_ulp_error=float((error/spacing).max()),
                failed_elements=failed, actual_min=float(actual.min()), actual_max=float(actual.max()),
                reference_min=float(reference.min()), reference_max=float(reference.max()))


def _internal_dataset(file, path):
    group = file
    pieces = path.split("/")
    for index, piece in enumerate(pieces):
        if not isinstance(group.get(piece, getlink=True), h5py.HardLink):
            raise ValueError("Only internal hard-linked source objects are allowed: "+path)
        group = group[piece]
        if index < len(pieces)-1 and not isinstance(group, h5py.Group):
            raise ValueError("Invalid source group: "+path)
    if not isinstance(group, h5py.Dataset) or group.is_virtual or group.external:
        raise ValueError("Physical internal dataset required: "+path)
    return group


def verify_scene(path, *, windows=None):
    """Verify sampled values only; never imply exhaustive-array/model admission.

    ``windows`` optionally maps every destination path to explicit <=16x16
    rectangles in that destination's global pixel grid.
    """
    path = Path(path).resolve(strict=True)
    signature = (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino)
    results, profile_hashes = {}, {}
    with h5py.File(path, "r") as file:
        schema = file.attrs.get("schema", "")
        if isinstance(schema, bytes):
            schema = schema.decode()
        if schema != "PANPV_NATIVE_RR_SCENE_v3" or file.attrs.get("complete") != 1:
            raise ValueError("Finalized v3 scene required")
        for key, expected in PROFILE_SHA256.items():
            meta = _internal_dataset(file, "metadata/"+key+"_json")
            if meta.shape != () or meta.id.get_storage_size() > 65536:
                raise ValueError("Profile metadata exceeds scalar bounded contract")
            raw = meta[()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if len(raw) > 65536:
                raise ValueError("Oversized profile metadata")
            profile_hashes[key] = _sha(json.loads(raw))
            if profile_hashes[key] != expected:
                raise ValueError("Declared profile differs from frozen v3 contract: "+key)
        if windows is not None and set(windows) != set(PATHS):
            raise ValueError("Explicit probes must cover all four derived datasets")
        for target_path, (source_path, kind, channels) in PATHS.items():
            source, target = _internal_dataset(file, source_path), _internal_dataset(file, target_path)
            expected_shape = (1, channels, source.shape[-2]//4, source.shape[-1]//4)
            if target.shape != expected_shape or target.dtype != np.float32:
                raise ValueError("Derived array shape/storage mismatch: "+target_path)
            selections = probe_windows(target.shape[-2:]) if windows is None else windows[target_path]
            if not selections or len(selections) > 128:
                raise ValueError("Require 1..128 bounded windows per derived dataset")
            records = []
            for window in selections:
                row, col, height, width = _window(window, target.shape[-2:])
                reference = recompute_window(source, window, kind)
                actual = target[0, :, row:row+height, col:col+width]
                records.append(dict(window=list(window), **compare_float32(actual, reference)))
            elements = sum(v["elements"] for v in records)
            finite = all(v["finite"] for v in records)
            results[target_path] = dict(
                source=source_path, bands=channels, windows=len(records), sampled_elements=elements,
                finite=finite, passed=all(v["passed"] for v in records),
                max_abs_dn=max(v["max_abs_dn"] for v in records) if finite else None,
                rmse_dn=math.sqrt(sum(v["squared_error_sum"] for v in records)/elements) if finite else None,
                max_ulp_error=max(v["max_ulp_error"] for v in records) if finite else None,
                failed_elements=sum(v["failed_elements"] for v in records), probes=records)
        scene_id = str(file.attrs.get("scene_id", ""))
    if signature != (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino):
        raise ValueError("Scene bytes/stat identity changed during verification")
    return dict(schema="PANPV_V3_INDEPENDENT_PARITY_v1", scene_id=scene_id, path=str(path),
                passed=all(v["passed"] for v in results.values()), profile_sha256=profile_hashes,
                scope="DETERMINISTIC_SPARSE_WINDOW_PARITY_NOT_EXHAUSTIVE_ARRAY_OR_MODEL_ADMISSION",
                tolerance="abs(stored-float32(independent_float64)) <= 2*float32_ULP(max_abs_pair)",
                source_read_side_limit=MAX_READ_SIDE, output_side_limit=MAX_OUTPUT_SIDE,
                source_file_modified=False, results=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    args = parser.parse_args(argv)
    result = verify_scene(args.scene)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
