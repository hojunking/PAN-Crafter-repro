"""Read-only consumer of the supplied whole-scene PANPV v3 dataset.

No supplied Python is imported, no RR degradation or random re-split is run,
and no input is written. Local checks are NOT a package/donor/training gate.
All region coordinates use canonical native-MS pixels (also the RR target
grid); native PAN coordinates are multiplied by four. Finite DN zero is data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from panpv.common import canonical_sha, file_sha
from panpv.data import _reject_links


SCHEMA = "PANPV_NATIVE_RR_SCENE_v3"
BANDS = ["Coastal", "Blue", "Green", "Yellow", "Red", "RedEdge", "NIR1", "NIR2"]
SPLITS = {"train": 0, "val": 1, "audit": 2}
RR_PROFILE = "PANPV_WV2_APPROX_GAUSSIAN_CENTER15_R4_FLOAT64_v1"
LP_PROFILE = "FH12_GAUSS198_K41_REPLICATE_OFFSET2_v1"


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _identity(path):
    info = os.stat(path)
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _array_sha(array):
    return hashlib.sha256(np.asarray(array, dtype="<i4").tobytes(order="C")).hexdigest()


def _json(source, key):
    name = f"metadata/{key}_json"
    try:
        obj = source[name]
        if obj.shape != () or obj.size != 1:
            raise ValueError("JSON dataset must be scalar")
        value = json.loads(_text(obj[()]))
        if not isinstance(value, dict):
            raise ValueError("JSON metadata must be an object")
        return value
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing/invalid metadata: {name}") from error


class SceneV3:
    """Validate structure, frozen indices and optionally SHA; lazily read images.

    ``verify_bytes=False`` is diagnostic only and marks RR datasets provisional.
    ``expected_sha256`` binds to an external manifest; without it the computed
    SHA is only a local fingerprint and RR datasets remain provisional. Raw-DN
    RR readers are also provisional for training (they remain usable for
    baseline diagnostics). Pixel-wide and source-TIFF audits, physical
    geometry and campaign gates remain separate and are never implied here.
    """

    def __init__(self, path, expected_sha256=None, verify_bytes=True):
        original_path = Path(path)
        if original_path.is_symlink():
            raise ValueError("Direct symlink scene inputs are not accepted")
        self.path = str(original_path.resolve(strict=True))
        if Path(self.path).suffix.lower() != ".h5" or ".partial" in Path(self.path).name:
            raise ValueError("A finalized .h5 scene is required")
        if not isinstance(verify_bytes, bool):
            raise TypeError("verify_bytes must be bool")
        if expected_sha256 is not None and (not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise ValueError("expected_sha256 must be a lowercase SHA256")
        if expected_sha256 is not None and not verify_bytes:
            raise ValueError("An expected SHA requires byte verification")
        self._file = self._pid = None
        self._source_identity = _identity(self.path)
        self.sha256 = file_sha(self.path) if verify_bytes else None
        if expected_sha256 is not None and expected_sha256 != self.sha256:
            raise ValueError("Scene SHA256 mismatch")
        self.expected_sha256 = expected_sha256
        with h5py.File(self.path, "r") as source:
            self._validate(source)
        self._check_identity()

    def _validate(self, f):
        _reject_links(f)
        attrs = f.attrs
        if (_text(attrs.get("schema", "")) != SCHEMA or attrs.get("complete") != 1
                or _text(attrs.get("axis_order", "")) != "NCHW" or attrs.get("ratio") != 4):
            raise ValueError("Unsupported/incomplete v3 schema, axes or ratio")
        if (_text(attrs.get("value_encoding", "")) != "delivery_DN_unscaled"
                or attrs.get("nominal_max") != 2047
                or json.loads(_text(attrs.get("band_order", "[]"))) != BANDS):
            raise ValueError("Unscaled DN2047 and canonical eight bands are required")
        self.scene_id = _text(attrs.get("scene_id", ""))
        if not self.scene_id:
            raise ValueError("Missing scene ID")
        self._metadata = {key: _json(f, key) for key in
                          ("normalization", "geometry", "rr_profile", "lp_profile", "split_provenance")}
        norm = self._metadata["normalization"]
        if (norm.get("nominal_max") != 2047 or norm.get("already_normalized") is not False
                or norm.get("model_expression") != "2*x/2047-1" or norm.get("clipping") != "none"):
            raise ValueError("Normalization metadata is inconsistent")
        for key, expected in (("rr_profile", RR_PROFILE), ("lp_profile", LP_PROFILE)):
            profile = self._metadata[key]
            if profile.get("id") != expected or profile.get("ratio") != 4:
                raise ValueError(f"Unsupported {key}")
        shape = f["native/ms"].shape
        if len(shape) != 4 or shape[:2] != (1, 8) or min(shape[-2:]) <= 0 or any(v % 4 for v in shape[-2:]):
            raise ValueError("native/ms must be [1,8,H,W], H/W divisible by four")
        self.height, self.width = h, w = shape[-2:]
        expected = {
            "native/pan": (1, 4*h, 4*w, "uint16"), "native/ms": (8, h, w, "uint16"),
            "native/lpan": (1, h, w, "float32"), "rr/target": (8, h, w, "uint16"),
            "rr/pan": (1, h, w, "float32"), "rr/ms": (8, h//4, w//4, "float32"),
            "rr/lpan": (1, h//4, w//4, "float32"),
        }
        expected.update({name + "_valid": (1, hh, ww, "uint8")
                         for name, (_, hh, ww, _) in list(expected.items())})
        expected["native/common_valid_ms"] = (1, h, w, "uint8")
        for name, (c, hh, ww, dtype) in expected.items():
            if (name not in f or not isinstance(f[name], h5py.Dataset)
                    or f[name].shape != (1, c, hh, ww) or f[name].dtype != np.dtype(dtype)):
                raise ValueError(f"Invalid dataset shape/dtype: {name}")
        for left, right in (("rr/target", "native/ms"), ("rr/target_valid", "native/ms_valid")):
            if h5py.h5o.get_info(f[left].id).addr != h5py.h5o.get_info(f[right].id).addr:
                raise ValueError(f"{left} must hard-link {right}, not a second degradation")
        self.source_col = _integer(attrs.get("source_ms_col"), "source_ms_col")
        self.source_row = _integer(attrs.get("source_ms_row"), "source_ms_row")
        geo = self._metadata["geometry"]
        if geo.get("accepted") is not True or geo.get("crop_ms_xywh") != [self.source_col, self.source_row, w, h]:
            raise ValueError("Geometry crop disagrees with native array")
        offset = geo.get("integer_pan_offset_xy")
        if not isinstance(offset, list) or len(offset) != 2 or any(type(v) is not int for v in offset):
            raise ValueError("Missing integer PAN offset")
        self.pan_offset_xy = tuple(offset)
        self._validate_splits(f)

    def _validate_splits(self, f):
        for name, width in (("patch_index", 4), ("blocks", 5)):
            d = f[f"splits/{name}"]
            if d.ndim != 2 or d.shape[1] != width or d.dtype != np.dtype("int32") or not len(d):
                raise ValueError(f"Invalid integer split table: {name}")
        attrs = f["splits/patch_index"].attrs
        if (attrs.get("target_size") != 64 or attrs.get("dependency_buffer") != 128
                or _text(attrs.get("columns", "")) != "row,col,split_code,block_id"):
            raise ValueError("Expected frozen 64-pixel patches with 128-pixel buffer")
        blocks, index = f["splits/blocks"][:].astype(np.int64), f["splits/patch_index"][:].astype(np.int64)
        br, bc, bh, bw, codes = blocks.T
        if (np.any(br < 0) or np.any(bc < 0) or np.any(bh <= 0) or np.any(bw <= 0)
                or np.any(br+bh > self.height) or np.any(bc+bw > self.width)
                or not np.isin(codes, [-1, 0, 1, 2]).all()):
            raise ValueError("Invalid spatial block bounds/split codes")
        for n, (r, c, hh, ww, _) in enumerate(blocks):
            other = blocks[:n]
            if np.any((other[:, 0] < r+hh) & (other[:, 0]+other[:, 2] > r)
                      & (other[:, 1] < c+ww) & (other[:, 1]+other[:, 3] > c)):
                raise ValueError("Overlapping spatial blocks")
        row, col, code, bid = index.T
        if (np.any(row < 0) or np.any(col < 0) or np.any(row % 64) or np.any(col % 64)
                or np.any(row+64 > self.height) or np.any(col+64 > self.width)
                or not np.isin(code, [0, 1, 2]).all() or np.any(bid < 0) or np.any(bid >= len(blocks))):
            raise ValueError("Invalid patch phase, bounds or split/block ID")
        if len(np.unique(index[:, :2], axis=0)) != len(index):
            raise ValueError("Duplicate patch coordinates")
        assigned = blocks[bid]
        if (np.any(code != assigned[:, 4]) or np.any(row-128 < assigned[:, 0])
                or np.any(col-128 < assigned[:, 1]) or np.any(row+64+128 > assigned[:, 0]+assigned[:, 2])
                or np.any(col+64+128 > assigned[:, 1]+assigned[:, 3])):
            raise ValueError("Patch dependency buffer crosses its assigned block")
        if set(code.tolist()) != {0, 1, 2}:
            raise ValueError("Each frozen split must be nonempty")
        provenance = self._metadata["split_provenance"]
        if (provenance.get("scene_id") != self.scene_id or provenance.get("scene_shape_hw") != [self.height, self.width]
                or provenance.get("buffer") != 128 or provenance.get("patch_size") != 64):
            raise ValueError("Split provenance is inconsistent")
        candidates = f["splits/candidate_patch_index"]
        keep = f["splits/candidate_keep"]
        if (candidates.ndim != 2 or candidates.shape[1] != 4 or candidates.dtype != np.dtype("int32")
                or keep.shape != (len(candidates),) or keep.dtype != np.dtype("uint8")):
            raise ValueError("Invalid candidate split/filter table")
        candidates, keep = candidates[:], keep[:]
        if (not np.isin(keep, [0, 1]).all()
                or not np.array_equal(index, candidates[keep.astype(bool)])):
            raise ValueError("Selected index does not match frozen candidate/filter table")
        if _array_sha(candidates) != provenance.get("patch_index_sha256"):
            raise ValueError("Candidate patch index provenance SHA mismatch")
        summary_keys = {"numpy_version", "usable_block_count", "block_counts", "patch_counts_before_data_filter",
                        "block_assignment_sha256", "patch_index_sha256", "patch_index_hash_encoding"}
        contract = {k: v for k, v in provenance.items() if k not in summary_keys}
        if canonical_sha(dict(contract=contract, blocks=blocks.tolist())) != provenance.get("block_assignment_sha256"):
            raise ValueError("Spatial block assignment provenance SHA mismatch")
        receipt = _json(f, "build_receipt").get("split", {})
        counts = {k: int(np.count_nonzero(code == v)) for k, v in (("train", 0), ("validation", 1), ("audit", 2))}
        if (receipt.get("patch_index_sha256") != _array_sha(index)
                or receipt.get("counts") != counts or receipt.get("provenance") != provenance):
            raise ValueError("Selected split receipt/hash mismatch")
        self._index, self._blocks = index, blocks
        self._index.flags.writeable = self._blocks.flags.writeable = False

    def _check_identity(self):
        if _identity(self.path) != self._source_identity:
            self.close()
            raise ValueError("Scene changed since verification")

    def _handle(self):
        self._check_identity()
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
            self._check_identity()
        return self._file

    def summary(self):
        return dict(scene_id=self.scene_id, path=self.path, sha256=self.sha256,
                    expected_sha256=self.expected_sha256, shape_ms=[self.height, self.width],
                    counts={k: int(np.count_nonzero(self._index[:, 2] == v)) for k, v in SPLITS.items()},
                    patch_index_sha256=_array_sha(self._index), blocks_sha256=_array_sha(self._blocks),
                    profiles=copy.deepcopy(self._metadata),
                    profile_sha256={k: canonical_sha(v) for k, v in self._metadata.items()},
                    local_verification="SCHEMA_AND_SPATIAL_SPLITS" + ("_AND_BYTE_SHA" if self.sha256 else ""),
                    campaign_gates_verified=False, rr_redegraded=False,
                    coordinate_frame="canonical_native_MS", hash_encoding="little-endian int32 C-order")

    def read_region(self, scope, row, col, height, width, normalize=False):
        """Return CHW NumPy values and original-grid boolean masks, no resampling."""
        if scope not in ("native", "rr") or not isinstance(normalize, bool):
            raise ValueError("scope must be native/rr and normalize must be bool")
        row, col = _integer(row, "row"), _integer(col, "col")
        height, width = _integer(height, "height", 1), _integer(width, "width", 1)
        if row+height > self.height or col+width > self.width:
            raise ValueError("Region is outside scene")
        if scope == "rr" and any(v % 4 for v in (row, col, height, width)):
            raise ValueError("RR rectangles must preserve the global ratio-four phase")
        rects = ({"pan": (4*row, 4*col, 4*height, 4*width), "ms": (row, col, height, width),
                  "lpan": (row, col, height, width)} if scope == "native" else
                 {"pan": (row, col, height, width), "target": (row, col, height, width),
                  "ms": (row//4, col//4, height//4, width//4), "lpan": (row//4, col//4, height//4, width//4)})
        f, values, masks = self._handle(), {}, {}
        for key, (r, c, h, w) in rects.items():
            a = f[f"{scope}/{key}"][0, :, r:r+h, c:c+w].astype(np.float32)
            m = f[f"{scope}/{key}_valid"][0, :, r:r+h, c:c+w]
            if not np.isin(m, [0, 1]).all() or not np.isfinite(a).all():
                raise ValueError("Nonfinite input or nonbinary mask")
            values[key] = a * np.float32(2/2047) - np.float32(1) if normalize else a
            masks[key] = m.astype(bool)
        if scope == "native":
            common = f["native/common_valid_ms"][0, :, row:row+height, col:col+width]
            if not np.isin(common, [0, 1]).all():
                raise ValueError("Nonbinary common native mask")
            masks["common_ms"] = common.astype(bool)
        self._check_identity()
        return dict(**values, masks=masks, normalized=normalize, nominal_max=2047,
                    coordinates=dict(canonical_ms_row_col=[row, col], scene_id=self.scene_id,
                                     source_ms_row_col=[self.source_row+row, self.source_col+col],
                                     source_pan_row_col=[4*(self.source_row+row)+self.pan_offset_xy[1],
                                                         4*(self.source_col+col)+self.pan_offset_xy[0]]))

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file = self._pid = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_file=None, _pid=None)
        return state


class RRPatchV3:
    """Fixed provider patch indices; shared valid support for all methods.

    Invalid low-resolution inputs are rejected, not nearest-upsampled into a
    misleading output mask. The provided 128-pixel common-support buffer is
    checked on access. All frozen splits use exactly the same 64/16 geometry.
    """

    def __init__(self, scene, split="train", normalize=True):
        if not isinstance(scene, SceneV3) or split not in SPLITS or not isinstance(normalize, bool):
            raise ValueError("Expected SceneV3, train/val/audit and bool normalize")
        self.scene, self.split, self.normalize = scene, split, normalize
        self._ids = np.flatnonzero(scene._index[:, 2] == SPLITS[split])
        # TrainingEngine rejects provisional data. A local fingerprint alone
        # does not bind the selected delivery, and raw DN is for baselines only.
        self.provisional = scene.sha256 is None or scene.expected_sha256 is None or not normalize

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, item):
        import torch
        item = _integer(item, "index")
        if item >= len(self):
            raise IndexError(item)
        raw = int(self._ids[item])
        row, col, code, block = map(int, self.scene._index[raw])
        values = self.scene.read_region("rr", row, col, 64, 64, normalize=self.normalize)
        masks = values["masks"]
        if not masks["ms"].all() or not masks["lpan"].all():
            raise ValueError("Invalid low-resolution support in frozen patch")
        support = self.scene._handle()["native/common_valid_ms"][0, 0, row-128:row+192, col-128:col+192]
        if support.shape != (320, 320) or not np.all(support == 1):
            raise ValueError("Invalid 128-pixel dependency support in frozen patch")
        mask = masks["pan"] & masks["target"]
        if not mask.any():
            raise ValueError("Empty RR target/PAN support")
        self.scene._check_identity()
        result = {k: torch.from_numpy(values[v]) for k, v in
                  (("pan", "pan"), ("ms", "ms"), ("lpan", "lpan"), ("gt", "target"))}
        result.update(mask=torch.from_numpy(mask), raw_index=raw,
                      coords=torch.tensor([row, col, code, block], dtype=torch.int64),
                      provisional=self.provisional)
        return result
