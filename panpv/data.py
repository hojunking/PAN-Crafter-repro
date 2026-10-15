"""Provided WV2 RR data: immutable spatial split candidates and contextual LP.

No Wald degradation, alignment or source mutation happens here. Coordinates are
provider-declared original-MS coordinates, i.e. the supplied RR target grid.
The filter recipe is not verified by the presence of a GNyq attribute alone.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


SPLITS = ("train", "val", "audit")
BANDS = "C,B,G,Y,R,RE,N,N2"
META_COLUMNS = "scene_idx,ms_x,ms_y,category_idx"
LP_RECIPE = dict(id="PANPV_GAUSS198_K41_GLOBAL_PHASE2_v1", sigma=1.98,
                 kernel=41, padding="replicate", ratio=4, phase=[2, 2],
                 calculation_dtype="float64", cache_dtype="float32")


def _plain(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _publish_json(path, value):
    """Create once, or verify exact content. Never overwrite an existing binding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"BLOCKED_IDENTITY: immutable JSON differs: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".panpv-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _reject_links(source):
    seen = set()
    def walk(group):
        address = h5py.h5o.get_info(group.id).addr
        if address in seen:
            return
        seen.add(address)
        for name in group:
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise ValueError("BLOCKED_INPUT_SCHEMA: external/soft HDF5 link")
            obj = group[name]
            if isinstance(obj, h5py.Group):
                walk(obj)
            elif obj.external or obj.is_virtual:
                raise ValueError("BLOCKED_INPUT_SCHEMA: external/virtual HDF5 dataset")
    walk(source)


def _rect_conflicts(meta, size, gap=0):
    """Unique pairs whose half-open footprints overlap after a total gap margin.

    Uses spatial bins; includes diagonal overlap, not just shared x/y coordinates.
    Different scene IDs are distinct geometric coordinate systems.
    """
    cell = size + gap
    bins = defaultdict(list)
    for i, (scene, x, y, _) in enumerate(meta.tolist()):
        bx, by = x // cell, y // cell
        for yy in range(by - 1, by + 2):
            for xx in range(bx - 1, bx + 2):
                for j in bins[(scene, xx, yy)]:
                    _, ox, oy, _ = meta[j]
                    if abs(x - int(ox)) < cell and abs(y - int(oy)) < cell:
                        yield j, i
        bins[(scene, bx, by)].append(i)


def inspect_rr_source(path):
    """Read actual schema/coordinate metadata; no heuristic semantic key mapping."""
    path = Path(path).resolve(strict=True)
    with h5py.File(path, "r") as source:
        _reject_links(source)
        if not {"pan", "ms", "gt", "meta"} <= set(source):
            raise ValueError("BLOCKED_INPUT_SCHEMA: supplied RR arrays required (not native FR)")
        attrs = {k: _plain(v) for k, v in source.attrs.items()}
        if (attrs.get("meta_cols") != META_COLUMNS or attrs.get("bands") != BANDS
                or attrs.get("ratio") != 4 or attrs.get("img_scale") != 2047
                or str(attrs.get("sensor", "")).lower() != "wv2"):
            raise ValueError("BLOCKED_INPUT_SCHEMA: provider band/scale/coordinate contract missing")
        scenes = json.loads(attrs["scene_index"])
        meta = source["meta"][:]
        n, bands, height, width = source["gt"].shape
        expected = {"gt": (n, 8, height, height), "pan": (n, 1, height, height),
                    "ms": (n, 8, height // 4, height // 4)}
        if (bands != 8 or height != width or height not in (64, 256) or n == 0
                or meta.shape != (n, 4) or not np.issubdtype(meta.dtype, np.integer)
                or any(source[key].shape != shape for key, shape in expected.items())
                or np.any(meta < 0) or np.any(meta[:, 0] >= len(scenes))):
            raise ValueError("BLOCKED_INPUT_SCHEMA: RR geometry or coordinates")
        if len(np.unique(meta[:, 0])) != 1:
            raise ValueError("BLOCKED_INPUT_SCHEMA: this campaign requires one original scene per source")
        if np.any(meta[:, 1:3] % 4):
            raise ValueError("BLOCKED_GEOMETRY: RR coordinates do not preserve donor ratio-four phase")
        if next(_rect_conflicts(meta, height), None) is not None:
            raise ValueError("BLOCKED_GEOMETRY: duplicate/overlapping RR source footprints")
        x, y = np.unique(meta[:, 1]), np.unique(meta[:, 2])
        dense = (height == 64 and len(x) * len(y) == n
                 and (len(x) == 1 or np.all(np.diff(x) == height))
                 and (len(y) == 1 or np.all(np.diff(y) == height)))
        schema = {key: dict(shape=list(source[key].shape), dtype=str(source[key].dtype))
                  for key in ("gt", "ms", "pan", "meta")}
    return dict(schema="PANPV_RR_SOURCE_v1", path=str(path), sha256=_sha_file(path),
                count=n, rr_size=height, scene_id=scenes[int(meta[0, 0])],
                attrs=attrs, arrays=schema, meta=meta.tolist(),
                meta_sha256=_sha_json(meta.tolist()), dense_rr_mosaic=bool(dense),
                origin_xy=[int(x.min()), int(y.min())],
                rr_extent_xyxy=[int(x.min()), int(y.min()), int(x.max()+height), int(y.max()+height)],
                coordinate_evidence="H5 attrs meta_cols and scene_index; provider SAMPLES.md",
                semantic_scope="PROVIDED_RR_TARGET_GRID_NOT_NATIVE_HRPAN")


def make_split_candidate(path, *, seed=261002, guard_rr=64, block_rr=512,
                         recipe_verified=False, source_id_evidence=None,
                         recipe_evidence=None):
    """Deterministic per-scene 80/10/10 coarse-block split, with symmetric support gap.

    guard_rr is a radius in supplied RR target pixels. Across split footprints a
    gap >= 2*guard_rr is enforced, giving each side a disjoint guard radius.
    Audit is retained first, then val, then train. Counts are AFTER exclusions.
    A provisional candidate may be inspected but cannot be used for training.
    """
    if (isinstance(guard_rr, bool) or int(guard_rr) != guard_rr or guard_rr < 0
            or int(block_rr) != block_rr or block_rr <= 0):
        raise ValueError("Invalid spatial guard/block")
    source = inspect_rr_source(path)
    size, meta = source["rr_size"], np.asarray(source["meta"], dtype=np.int64)
    if block_rr < size or block_rr % size:
        raise ValueError("Coarse block must be an integer multiple of RR source size")
    groups = defaultdict(list)
    straddling = []
    for index, (scene, x, y, _) in enumerate(meta):
        bx, by = int(x // block_rr), int(y // block_rr)
        if (x + size - 1) // block_rr != bx or (y + size - 1) // block_rr != by:
            straddling.append(index)
        else:
            groups[(int(scene), bx, by)].append(index)
    blocks = sorted(groups)
    if len(blocks) < 3:
        raise ValueError("BLOCKED_SPLIT: at least three independent coarse blocks required")
    rng_seed = int(_sha_json([seed, source["scene_id"]])[:16], 16)
    order = np.random.default_rng(rng_seed).permutation(len(blocks))
    n_audit = max(1, int(round(len(blocks) * .1)))
    n_val = max(1, int(round(len(blocks) * .1)))
    block_labels = {}
    for rank, k in enumerate(order):
        block_labels[blocks[k]] = "audit" if rank < n_audit else "val" if rank < n_audit + n_val else "train"
    assigned = [None] * len(meta)
    for block, ids in groups.items():
        for index in ids:
            assigned[index] = block_labels[block]
    removed = set(straddling)
    priority = {"audit": 2, "val": 1, "train": 0}
    # Simultaneous exclusion never resurrects a previously exposed sample.
    for i, j in _rect_conflicts(meta, size, gap=2 * guard_rr):
        if assigned[i] is not None and assigned[j] is not None and assigned[i] != assigned[j]:
            removed.add(i if priority[assigned[i]] < priority[assigned[j]] else j)
    selected = {label: [i for i, value in enumerate(assigned) if value == label and i not in removed]
                for label in SPLITS}
    if any(not ids for ids in selected.values()):
        raise ValueError("BLOCKED_SPLIT: guard leaves an empty train/val/audit partition")
    if recipe_verified:
        _verify_recipe_evidence(recipe_evidence, int(guard_rr))
    candidate = dict(schema="PANPV_SPATIAL_SPLIT_v1", source=source,
                     seed=int(seed), block_rr=int(block_rr), guard_rr=int(guard_rr),
                     required_cross_split_gap_rr=2*int(guard_rr),
                     requested_block_proportions=dict(train=.8, val=.1, audit=.1),
                     scope="PROVIDED_RR_SPATIAL_HOLDOUT", partitions=selected,
                     excluded_ids=sorted(removed), straddling_ids=straddling,
                     actual_counts={k: len(v) for k, v in selected.items()},
                     actual_fraction_of_source={k: len(v)/len(meta) for k, v in selected.items()},
                     block_labels=[dict(scene=b[0], x_block=b[1], y_block=b[2], split=block_labels[b]) for b in blocks],
                     source_id_evidence=source_id_evidence or source["coordinate_evidence"],
                     recipe_verified=bool(recipe_verified), recipe_evidence=recipe_evidence,
                     provisional=not recipe_verified, frozen=False,
                     selection_partition="val", audit_used_for_selection=False,
                     guard_reason="Declared conservative candidate radius; not a claim of unknown Wald support")
    candidate["candidate_sha256"] = _sha_json(candidate)
    validate_split(candidate)
    return candidate


def validate_split(manifest):
    """Recompute footprint separation and all selected IDs, not just a hash check."""
    value = _read(manifest)
    digest_key = "split_sha256" if value.get("frozen") else "candidate_sha256"
    if digest_key not in value or _sha_json({k: v for k, v in value.items() if k != digest_key}) != value[digest_key]:
        raise ValueError("BLOCKED_IDENTITY: split manifest digest differs")
    if value.get("frozen"):
        if not value.get("recipe_verified") or value.get("provisional"):
            raise ValueError("BLOCKED_DEGRADATION: frozen marker without verified recipe")
        _verify_recipe_evidence(value.get("recipe_evidence"), value["guard_rr"])
    source, groups = value["source"], value["partitions"]
    meta, size = np.asarray(source["meta"], dtype=np.int64), source["rr_size"]
    if _sha_json(meta.tolist()) != source["meta_sha256"]:
        raise ValueError("BLOCKED_IDENTITY: modified coordinates")
    labels = {}
    for label in SPLITS:
        ids = groups[label]
        if not ids or any(type(i) is not int or not 0 <= i < len(meta) or i in labels for i in ids):
            raise ValueError("BLOCKED_SPLIT: missing, repeated or invalid ID")
        for i in ids:
            if i in labels:
                raise ValueError("BLOCKED_SPLIT: duplicate source ID")
            labels[i] = label
    if set(labels) & set(value["excluded_ids"]):
        raise ValueError("BLOCKED_SPLIT: excluded source ID selected")
    if set(labels) | set(value["excluded_ids"]) != set(range(len(meta))):
        raise ValueError("BLOCKED_SPLIT: incomplete source accounting")
    for i, j in _rect_conflicts(meta, size, gap=2*value["guard_rr"]):
        if i in labels and j in labels and labels[i] != labels[j]:
            raise ValueError("BLOCKED_SPLIT: source/support leakage across partitions")
    if value.get("audit_used_for_selection") or value.get("selection_partition") != "val":
        raise ValueError("BLOCKED_SPLIT: audit cannot select checkpoints")
    return True


def freeze_split(candidate, path, *, recipe_verified=None, recipe_evidence=None):
    """Fail closed on unknown RR creation support. No implicit provisional override."""
    value = json.loads(json.dumps(_read(candidate)))
    validate_split(value)
    declared = value["recipe_verified"] if recipe_verified is None else recipe_verified
    evidence = recipe_evidence or value.get("recipe_evidence")
    if not declared or not evidence:
        raise ValueError("BLOCKED_DEGRADATION: split freeze needs verified RR recipe and support evidence")
    required_guard = _verify_recipe_evidence(evidence, value["guard_rr"])
    value.update(recipe_verified=True, recipe_evidence=evidence, provisional=False, frozen=True)
    value["verified_required_guard_rr"] = required_guard
    value["split_sha256"] = _sha_json(value)
    _publish_json(path, value)
    return value


def _verify_recipe_evidence(evidence, guard_rr):
    required = {"declared_kernel_radius_in_rr", "lp_radius_rr", "interpolation_radius_rr",
                "global_coordinate_origin_xy", "source_verification_reference"}
    if not isinstance(evidence, dict) or not required <= evidence.keys():
        raise ValueError("BLOCKED_DEGRADATION: structured kernel/phase/source verification evidence required")
    radii = [evidence[k] for k in ("declared_kernel_radius_in_rr", "lp_radius_rr", "interpolation_radius_rr")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in radii):
        raise ValueError("BLOCKED_DEGRADATION: invalid verified support radii")
    origin, reference = evidence["global_coordinate_origin_xy"], evidence["source_verification_reference"]
    if (radii[1] < 20 or radii[2] < 8 or not isinstance(origin, (list, tuple)) or len(origin) != 2
            or any(type(v) is not int for v in origin)
            or not isinstance(reference, str) or not reference.strip()):
        raise ValueError("BLOCKED_DEGRADATION: missing actual LP/interpolation/origin/source evidence")
    # Conservative composition, not max(): RR-generation and frontend supports
    # can accumulate along the same dependency path before a patch is consumed.
    support = float(sum(radii))
    if guard_rr < support:
        raise ValueError("BLOCKED_SPLIT: declared guard is smaller than composed verified support")
    return support


def _check_source_identity(source):
    path = Path(source["path"]).resolve(strict=True)
    if _sha_file(path) != source["sha256"]:
        raise ValueError("BLOCKED_IDENTITY: RR source changed")
    return path


def prepare_lpan_cache(path, cache_dir, *, max_chunk_rows=256):
    """Bounded-memory global LP for contiguous RR64, full-tile LP for sparse RR256.

    A disk-backed PAN mosaic is a cache, not a claim of native full-scene imagery.
    Sparse-tile replicate support is flagged; its invalid border is masked below.
    The original H5 is only opened read-only. No Wald re-degradation is performed.
    """
    import cv2
    source = inspect_rr_source(path)
    cache_dir = Path(cache_dir).resolve()
    input_path = Path(source["path"])
    if cache_dir.is_relative_to(input_path.parent) or input_path.is_relative_to(cache_dir):
        raise ValueError("BLOCKED_PATH: LP cache must be separate from original input directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(source_sha256=source["sha256"], meta_sha256=source["meta_sha256"],
                    recipe=LP_RECIPE, contextual=source["dense_rr_mosaic"],
                    cv2_version=cv2.__version__)
    cache_key = _sha_json(identity)
    manifest_path, output_path = cache_dir / f"{cache_key}.json", cache_dir / f"{cache_key}.npy"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest["identity"] != identity or _sha_file(output_path) != manifest["lpan_sha256"]:
            raise ValueError("BLOCKED_IDENTITY: LP cache differs")
        return manifest
    n, size = source["count"], source["rr_size"]
    if max_chunk_rows < size:
        raise ValueError("LP chunk must hold at least one RR source row")
    kernel = cv2.getGaussianKernel(41, 1.98)
    kernel = kernel @ kernel.T
    descriptor, temporary_name = tempfile.mkstemp(prefix=".lpan-", suffix=".npy", dir=cache_dir)
    os.close(descriptor)
    mosaic_name = None
    try:
        out = np.lib.format.open_memmap(temporary_name, mode="w+", dtype=np.float32,
                                        shape=(n, 1, size//4, size//4))
        with h5py.File(input_path, "r") as h5:
            if source["dense_rr_mosaic"]:
                meta = np.asarray(source["meta"], dtype=np.int64)
                x0, y0, x1, y1 = source["rr_extent_xyxy"]
                height, width = y1-y0, x1-x0
                descriptor, mosaic_name = tempfile.mkstemp(prefix=".rr-pan-mosaic-", suffix=".npy", dir=cache_dir)
                os.close(descriptor)
                mosaic = np.lib.format.open_memmap(mosaic_name, mode="w+", dtype=h5["pan"].dtype,
                                                   shape=(height, width))
                for start in range(0, n, 64):
                    batch = h5["pan"][start:start+64, 0]
                    if not np.isfinite(batch).all():
                        raise ValueError("BLOCKED_INPUT_SCHEMA: nonfinite RR PAN")
                    for offset, patch in enumerate(batch):
                        _, x, y, _ = meta[start+offset]
                        mosaic[y-y0:y-y0+size, x-x0:x-x0+size] = patch
                mosaic.flush()
                row_chunk = (max_chunk_rows//size)*size
                for start in range(0, height, row_chunk):
                    stop = min(height, start+row_chunk)
                    lo, hi = max(0, start-20), min(height, stop+20)
                    slab = np.asarray(mosaic[lo:hi], dtype=np.float64)
                    filtered = cv2.filter2D(slab, -1, kernel, borderType=cv2.BORDER_REPLICATE)
                    for index in np.flatnonzero((meta[:, 2]-y0 >= start) & (meta[:, 2]-y0 < stop)):
                        _, x, y, _ = meta[index]
                        yp, xp = int(y-y0-lo), int(x-x0)
                        out[index, 0] = filtered[yp+2:yp+size:4, xp+2:xp+size:4]
                del mosaic
                mode = "GLOBAL_PROVIDED_RR_MOSAIC"
            else:
                for index in range(n):
                    patch = np.asarray(h5["pan"][index, 0], dtype=np.float64)
                    if not np.isfinite(patch).all():
                        raise ValueError("BLOCKED_INPUT_SCHEMA: nonfinite RR PAN")
                    out[index, 0] = cv2.filter2D(patch, -1, kernel, borderType=cv2.BORDER_REPLICATE)[2::4, 2::4]
                mode = "ISOLATED_RR_TILE_REPLICATE_WITH_INTERIOR_MASK"
        out.flush()
        del out
        _check_source_identity(source)
        if output_path.exists():
            if _sha_file(output_path) != _sha_file(temporary_name):
                raise ValueError("BLOCKED_IDENTITY: interrupted LP cache bytes differ")
        else:
            os.link(temporary_name, output_path)
        manifest = dict(schema="PANPV_LPAN_CACHE_v1", identity=identity,
                        path=str(output_path), lpan_sha256=_sha_file(output_path),
                        shape=[n, 1, size//4, size//4], dtype="float32", source=source,
                        context_mode=mode, known_lp_radius_rr=20, interpolation_support_radius_rr=8,
                        default_context_mask_margin_rr=28,
                        native_fr_created=False, rr_degraded_again=False,
                        global_decimation_origin_xy=[2, 2], masked_context_is_not_model_receptive_field=True)
        _publish_json(manifest_path, manifest)
        return manifest
    finally:
        for temporary in (temporary_name, mosaic_name):
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


class RRDataset:
    """Read-only unaugmented supplied RR pairs, optionally 64px train subcrops.

    Parent-tile membership controls the split, never virtual crop membership.
    Default valid margin 28 accounts for LP radius20 + bicubic-upsample support8.
    It does NOT certify the U-Net receptive field or full-scene/tile parity.
    Explicit ``allow_provisional`` is diagnostic-only and marks every sample.
    Production callers must leave it false (the default).
    """
    def __init__(self, source, split_manifest, split, lp_cache, *, max_dn=2047,
                 training_crop=None, context_margin=28, allow_provisional=False):
        self.source = inspect_rr_source(source) if isinstance(source, (str, Path)) else source
        self.manifest, self.lp = _read(split_manifest), _read(lp_cache)
        validate_split(self.manifest)
        if split not in SPLITS:
            raise ValueError("Unknown split")
        if (not self.manifest.get("frozen") or self.manifest.get("provisional")) and not allow_provisional:
            raise ValueError("BLOCKED_DEGRADATION: provisional split is diagnostic-only")
        if (self.manifest["source"]["sha256"] != self.source["sha256"]
                or self.lp["identity"]["source_sha256"] != self.source["sha256"]
                or self.lp["identity"]["meta_sha256"] != self.source["meta_sha256"]):
            raise ValueError("BLOCKED_IDENTITY: source/split/LP mismatch")
        if max_dn != self.source["attrs"]["img_scale"]:
            raise ValueError("BLOCKED_NORMALIZATION: requested scale differs from supplied product")
        self.source_path = _check_source_identity(self.source)
        self.lp_path = Path(self.lp["path"]).resolve(strict=True)
        if _sha_file(self.lp_path) != self.lp["lpan_sha256"]:
            raise ValueError("BLOCKED_IDENTITY: LP bytes changed")
        self._lpan = np.load(self.lp_path, mmap_mode="r", allow_pickle=False)
        if list(self._lpan.shape) != self.lp["shape"] or self._lpan.dtype != np.float32:
            raise ValueError("BLOCKED_IDENTITY: LP shape/dtype")
        size = self.source["rr_size"]
        crop = size if training_crop is None else training_crop
        if (type(crop) is not int or crop % 4 or crop < 4 or size % crop
                or (split != "train" and training_crop is not None)):
            raise ValueError("64-style subcrops are only for train, preserving parent groups")
        if type(context_margin) is not int or context_margin < 28 or context_margin*2 >= size:
            raise ValueError("Context mask must preserve LP radius20 + bicubic support8 and leave valid source pixels")
        self.split, self.crop, self.context_margin, self.max_dn = split, crop, context_margin, max_dn
        self.augment = False
        self.provisional = not self.manifest.get("frozen") or self.manifest.get("provisional", True)
        self.entries = [(raw, y, x) for raw in self.manifest["partitions"][split]
                        for y in range(0, size, crop) for x in range(0, size, crop)]
        self._signatures = [(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
                            for p in (self.source_path, self.lp_path)]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        import torch
        if self._signatures != [(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
                                for p in (self.source_path, self.lp_path)]:
            raise ValueError("BLOCKED_IDENTITY: data changed during run")
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        index = int(index)
        raw, cy, cx = self.entries[index]
        width, margin, size = self.crop, self.context_margin, self.source["rr_size"]
        coords = np.asarray(self.source["meta"][raw], dtype=np.int64)
        with h5py.File(self.source_path, "r") as h5:
            arrays = {k: np.asarray(h5[k][raw, :, cy:cy+width, cx:cx+width], dtype=np.float32)
                      for k in ("pan", "gt")}
            arrays["ms"] = np.asarray(h5["ms"][raw, :, cy//4:(cy+width)//4, cx//4:(cx+width)//4], dtype=np.float32)
        arrays["lpan"] = np.array(self._lpan[raw, :, cy//4:(cy+width)//4, cx//4:(cx+width)//4])
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("BLOCKED_INPUT_SCHEMA: nonfinite supplied RR data")
        yy, xx = np.ogrid[cy:cy+width, cx:cx+width]
        if self.source["dense_rr_mosaic"]:
            x0, y0, x1, y1 = self.source["rr_extent_xyxy"]
            valid = ((coords[1]+xx >= x0+margin) & (coords[1]+xx < x1-margin)
                     & (coords[2]+yy >= y0+margin) & (coords[2]+yy < y1-margin))
        else:
            valid = (xx >= margin) & (xx < size-margin) & (yy >= margin) & (yy < size-margin)
        if not np.any(valid):
            raise ValueError("BLOCKED_MASK: no valid training/evaluation pixels")
        result = {k: torch.from_numpy(a * np.float32(2/self.max_dn)-np.float32(1)) for k, a in arrays.items()}
        result.update(mask=torch.from_numpy(valid[None].copy()), raw_index=raw,
                      coords=torch.from_numpy(coords), virtual_index=index,
                      crop_origin=torch.tensor([cy, cx], dtype=torch.int64),
                      provisional=bool(self.provisional))
        return result
