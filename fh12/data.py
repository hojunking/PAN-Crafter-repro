"""FH12 immutable native-PAN LP caches and explicit, RNG-free four-view data.

Integer indices always mean an unaugmented base patch. ``(index, rot)`` indices
mean the legacy training contract: fixed horizontal flip, fixed vertical flip,
then ``np.rot90(rot)``. The caller owns sample/augmentation RNG and resume state.
Unlike PanFeeder, constructing this dataset never resets a global RNG.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tools.repair_lpan import make_lpan


RECIPE = {
    "id": "FH12_GAUSS198_K41_REPLICATE_OFFSET2_v1",
    "sigma": 1.98, "kernel": 41, "padding": "replicate",
    "decimation": "[2::4,2::4]", "ratio": 4,
    "calculation_dtype": "float64", "cache_dtype": "float32",
    "phase_id": "gauss41-offset2-up4-bicubic-align_corners_false",
    "source_code_sha256": hashlib.sha256(inspect.getsource(make_lpan).encode()).hexdigest(),
}
AUGMENTATION = {
    "id": "FH12_LEGACY_FIXED_HV_ROT4_v1", "crop": False,
    "hflip": True, "vflip": True, "rotations": [0, 1, 2, 3],
    "order": ["horizontal_flip", "vertical_flip", "np.rot90"],
    "integer_index": "base_no_augmentation", "tuple_index": "fixed_HV_then_rot",
}
DEFAULT_PATHS = {
    "train": "data/PanCollection/WV3/train_wv3.h5",
    "val": "data/PanCollection/WV3/valid_wv3.h5",
    "rr": "data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5",
    "fr": "data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_deadline(deadline_utc):
    if deadline_utc is None:
        return
    when = (datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
            if isinstance(deadline_utc, str) else deadline_utc)
    if when.tzinfo is None:
        raise ValueError("FH12 deadline must include a timezone")
    if datetime.now(timezone.utc) >= when:
        raise TimeoutError("FH12 local pipeline deadline reached")


def write_immutable_json(path, value):
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != value:
            raise ValueError(f"Refusing to overwrite immutable FH12 asset: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp, 0o444)
        # Hard-link publication refuses even a concurrent pre-existing target.
        os.link(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


def phase_diagnostics():
    """Measure (do not silently correct) recipe phase on native HR inputs."""
    size = 64
    ramp = np.broadcast_to(np.arange(size, dtype=np.float64), (1, 1, size, size)).copy()
    impulse = np.zeros((1, 1, size, size), dtype=np.float64)
    impulse[0, 0, 32, 32] = 1.0
    def up(pan):
        return F.interpolate(torch.from_numpy(make_lpan(pan)), scale_factor=4,
                             mode="bicubic", align_corners=False).numpy()
    ramp_up = up(ramp)
    imp = up(impulse)[0, 0]
    yy, xx = np.indices(imp.shape)
    mass = float(imp.sum())
    return {
        "phase_id": RECIPE["phase_id"], "shape": [64, 64],
        "ramp_interior_mean_L_minus_P": float((ramp_up - ramp)[..., 20:-20, 20:-20].mean()),
        "impulse_native_yx": [32, 32],
        "impulse_reconstructed_centroid_yx": [float((yy * imp).sum() / mass), float((xx * imp).sum() / mass)],
        "note": "Recorded common phase only; no case-dependent phase correction.",
    }


def _source_info(path, split, expected_count=None):
    with h5py.File(path, "r") as source:
        keys = ["ms", "lms", "pan"] + ([] if split == "fr" else ["gt"])
        missing = set(keys) - set(source.keys())
        if missing:
            raise ValueError(f"{path}: missing dataset keys {sorted(missing)}")
        shapes = {key: list(source[key].shape) for key in keys}
        n, c, h, w = source["pan"].shape
        if c != 1 or h != w or h % 4 or source["ms"].shape != (n, 8, h // 4, w // 4):
            raise ValueError(f"FH12 requires native PAN1/LRMS8 ratio4, got {shapes}")
        if source["lms"].shape != (n, 8, h, w) or ("gt" in source and source["gt"].shape != (n, 8, h, w)):
            raise ValueError(f"FH12 MS/GT geometry differs: {shapes}")
        if expected_count is not None and n != expected_count:
            raise ValueError(f"FH12 {split} requires {expected_count} scenes, got {n}")
    return {"sha256": sha256_file(path), "shapes": shapes, "count": n}


def _create_lp_cache(source_path, cache_path, source_sha, deadline_utc=None, chunk=64):
    """Generate before augmentation; publish only the completed validated H5."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_sha = canonical_sha(RECIPE)
    if cache_path.exists():
        with h5py.File(cache_path, "r") as existing:
            if existing.attrs.get("source_sha256") != source_sha or existing.attrs.get("recipe_sha256") != recipe_sha:
                raise ValueError(f"LP cache identity mismatch: {cache_path}")
        return
    fd, temp = tempfile.mkstemp(prefix=".lpan-", suffix=".h5", dir=cache_path.parent)
    os.close(fd)
    try:
        with h5py.File(source_path, "r") as source, h5py.File(temp, "w") as target:
            p = source["pan"]
            n, _, h, w = p.shape
            lp = target.create_dataset("lpan", shape=(n, 1, h // 4, w // 4), dtype="float32")
            order_hashes = []
            for start in range(0, n, chunk):
                check_deadline(deadline_utc)
                native = p[start:start + chunk].astype(np.float64)
                if not np.isfinite(native).all():
                    raise ValueError("Non-finite native PAN input")
                reduced = make_lpan(native).astype(np.float32)
                if reduced.shape != (len(native), 1, h // 4, w // 4) or not np.isfinite(reduced).all():
                    raise ValueError("Invalid LP generated from actual native PAN")
                lp[start:start + chunk] = reduced
                order_hashes.extend(hashlib.sha256(x.astype("<f4").tobytes()).hexdigest() for x in native)
            target.attrs["source_sha256"] = source_sha
            target.attrs["recipe_sha256"] = recipe_sha
            target.attrs["sample_order_sha256"] = canonical_sha(order_hashes)
            target.attrs["recipe_json"] = json.dumps(RECIPE, sort_keys=True)
            target.flush()
        if sha256_file(source_path) != source_sha:
            raise ValueError("Native PAN source changed during LP cache construction")
        os.chmod(temp, 0o444)
        os.link(temp, cache_path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _legacy_lp_difference(source_path, cache_path, deadline_utc=None):
    legacy = Path(str(source_path).replace(".h5", "_pan.h5"))
    if not legacy.exists():
        return {"available": False, "path": str(legacy)}
    with h5py.File(legacy, "r") as old, h5py.File(cache_path, "r") as new:
        if "lpan" not in old or old["lpan"].shape != new["lpan"].shape:
            return {"available": True, "shape_mismatch": True, "path": str(legacy)}
        squared, count, maximum = 0.0, 0, 0.0
        for start in range(0, len(new["lpan"]), 64):
            check_deadline(deadline_utc)
            diff = new["lpan"][start:start + 64].astype(np.float64) - old["lpan"][start:start + 64]
            if not np.isfinite(diff).all():
                return {"available": True, "nonfinite_legacy": True, "path": str(legacy)}
            squared += float(np.square(diff).sum())
            count += diff.size
            maximum = max(maximum, float(np.abs(diff).max()))
    return {"available": True, "path": str(legacy), "sha256": sha256_file(legacy),
            "rmse_dn": float(np.sqrt(squared / count)), "max_abs_dn": maximum}


def prepare_data(root: Path, server: str, deadline_utc=None, data_paths=None):
    """Prepare all four immutable LP identities without touching original H5s.

    ``data_paths`` is an explicit testing/deployment override, never an LP fallback.
    Production RR/FR always require all 20 scenes.
    """
    root = Path(root).resolve()
    if server not in {f"s{i}" for i in range(1, 6)}:
        raise ValueError(f"Invalid FH12 server {server!r}")
    paths = dict(DEFAULT_PATHS if data_paths is None else data_paths)
    if set(paths) != set(DEFAULT_PATHS):
        raise ValueError("FH12 requires explicit train/val/rr/fr sources")
    asset_dir = root / "work_dir" / "_fh12" / server
    manifest_path = asset_dir / "dataset_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    splits, lp_details = {}, {}
    for split, name in paths.items():
        check_deadline(deadline_utc)
        source = (root / name).resolve()
        info = _source_info(source, split, 20 if split in {"rr", "fr"} else None)
        cache = asset_dir / "lpan" / f"{split}_{info['sha256'][:20]}_{canonical_sha(RECIPE)[:16]}.h5"
        _create_lp_cache(source, cache, info["sha256"], deadline_utc)
        cache_sha = sha256_file(cache)
        if old is not None and old["splits"][split].get("lpan_sha256") != cache_sha:
            raise ValueError(f"Immutable {split} LP content changed since preparation")
        with h5py.File(cache, "r") as lp:
            order_sha = str(lp.attrs["sample_order_sha256"])
        splits[split] = {"dataroot": str(source), "lpan_path": str(cache), **info,
                         "lpan_sha256": cache_sha, "sample_order_sha256": order_sha}
        # Comparison statistics are diagnostic; they never select the LP supply.
        lp_details[split] = {**splits[split], "legacy_comparison": _legacy_lp_difference(source, cache, deadline_utc)}
    manifest = {"schema": "FH12_DATA_v1", "server": server, "sensor": "WV3", "max_pixel": 2047.0,
                "splits": splits, "recipe": RECIPE, "augmentation": AUGMENTATION,
                "augmentation_sha256": canonical_sha(AUGMENTATION)}
    lp_manifest = {"schema": "FH12_LPAN_v1", "server": server, "recipe": RECIPE,
                   "opencv_version": cv2.__version__, "phase_diagnostics": phase_diagnostics(), "splits": lp_details}
    write_immutable_json(asset_dir / "lpan_manifest.json", lp_manifest)
    write_immutable_json(manifest_path, manifest)
    return manifest


class FH12Dataset(Dataset):
    def __init__(self, dataroot, lpan_path, augment=False, max_pixel=2047.0, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported FH12 dataset options: {sorted(kwargs)}")
        if augment:
            raise ValueError("Use explicit (index,rot) indices, not implicit augmentation")
        self.raw_h5_path = str(Path(dataroot).resolve())
        self.dataroot = self.raw_h5_path
        self.lp_path = str(Path(lpan_path).resolve())
        self.max_pixel = float(max_pixel)
        if self.max_pixel != 2047.0:
            raise ValueError("FH12 is WV3-only with 2047 DN normalization")
        with h5py.File(self.raw_h5_path, "r") as source:
            self.has_gt = "gt" in source
            self.arrays = {key: source[key][:] for key in ["lms", "ms", "pan"] + (["gt"] if self.has_gt else [])}
        with h5py.File(self.lp_path, "r") as cache:
            self.arrays["lpan"] = cache["lpan"][:]
        n, _, h, w = self.arrays["pan"].shape
        if self.arrays["lpan"].shape != (n, 1, h // 4, w // 4):
            raise ValueError("Explicit LP cache is not aligned with native PAN sample count/shape")

    def __len__(self):
        return len(self.arrays["pan"])

    def get_view(self, index: int, rot: int = 0, augment: bool = False):
        index, rot = int(index), int(rot)
        if not 0 <= index < len(self) or rot not in (0, 1, 2, 3):
            raise IndexError((index, rot))
        if not augment and rot != 0:
            raise ValueError("Nonzero rotation requires explicit augment=True")
        keys = (["gt"] if self.has_gt else []) + ["lms", "ms", "lpan", "pan"]
        values = []
        for key in keys:
            x = self.arrays[key][index]
            if augment:
                x = np.rot90(x[:, ::-1, ::-1], rot, axes=(1, 2))
            x = torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
            values.append(x.mul_(2.0 / self.max_pixel).sub_(1.0))
        values.append(torch.tensor([index, rot if augment else 0, int(augment), int(augment)], dtype=torch.int64))
        return tuple(values)

    def base(self, index):
        return self.get_view(index)

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            if len(index) != 2:
                raise ValueError("FH12 augmented index must be (index,rot)")
            return self.get_view(index[0], index[1], augment=True)
        return self.base(index)


def build_dataset(data_cfg, split, root=None, server=None, augment=False):
    if isinstance(data_cfg, (str, Path)):
        data_cfg = json.loads(Path(data_cfg).read_text())
    item = data_cfg["splits"][split]
    base = Path(root or ".")
    source, lp = base / item["dataroot"], base / item["lpan_path"]
    if sha256_file(source) != item["sha256"] or sha256_file(lp) != item["lpan_sha256"]:
        raise ValueError(f"FH12 {split} source or immutable LP SHA mismatch")
    return FH12Dataset(source, lp, augment=augment, max_pixel=data_cfg.get("max_pixel", 2047.0))
