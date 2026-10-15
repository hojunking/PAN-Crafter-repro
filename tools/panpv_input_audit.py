#!/usr/bin/env python3
"""Read-only inventory of supplied PANPV HDF5 samples, never a training gate.

Schema/geometry labels are candidates, not inferred sensor/encoding semantics.
Full dataset numeric counts are streamed; quantiles and per-band diagnostics
use an explicitly recorded deterministic subset of at most 128 axis-0 records.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np


SCHEMA = "PANPV_READONLY_INPUT_AUDIT_v1"
SAMPLE_ELEMENT_LIMIT = 8 * 1024 * 1024
QUANTILES = [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]


class UnsafeHDF5(ValueError):
    pass


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    """Do not follow HDF5 references or emit nonstandard JSON NaN/Infinity."""
    if isinstance(value, h5py.Reference):
        return {"hdf5_reference": "not_dereferenced"}
    if isinstance(value, np.ndarray):
        if value.size > 4096:
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "omitted": "attribute exceeds 4096 values",
                    "preview": json_value(value.reshape(-1)[:32])}
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    if isinstance(value, (tuple, list)):
        return [json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_attribute": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unsupported_attribute_type": type(value).__name__}


def attributes(obj):
    return {str(k): json_value(v) for k, v in obj.attrs.items()}


def _slices(shape, elements):
    if not shape:
        yield ()
        return
    if any(dim == 0 for dim in shape):
        return
    tile, remaining = [1] * len(shape), max(1, elements)
    for axis in reversed(range(len(shape))):
        tile[axis] = min(shape[axis], remaining)
        remaining = max(1, remaining // tile[axis])
    for starts in itertools.product(*(range(0, d, t) for d, t in zip(shape, tile))):
        yield tuple(slice(start, min(start + width, dim))
                    for start, width, dim in zip(starts, tile, shape))


def _counts():
    return {"elements": 0, "finite": 0, "nonfinite": 0, "nan": 0,
            "positive_inf": 0, "negative_inf": 0, "zero": 0,
            "negative": 0, "greater_than_2047": 0, "min": None, "max": None}


def _accumulate(result, array):
    array = np.asarray(array)
    finite = np.isfinite(array)
    values = array[finite]
    result["elements"] += int(array.size)
    result["finite"] += int(finite.sum())
    result["nonfinite"] += int(array.size - finite.sum())
    result["nan"] += int(np.isnan(array).sum())
    result["positive_inf"] += int(np.isposinf(array).sum())
    result["negative_inf"] += int(np.isneginf(array).sum())
    result["zero"] += int(np.count_nonzero(values == 0))
    result["negative"] += int(np.count_nonzero(values < 0))
    result["greater_than_2047"] += int(np.count_nonzero(values > 2047))
    if values.size:
        low, high = values.min().item(), values.max().item()
        result["min"] = low if result["min"] is None else min(low, result["min"])
        result["max"] = high if result["max"] is None else max(high, result["max"])


def _describe_values(array):
    result = _counts()
    _accumulate(result, array)
    finite = np.asarray(array)[np.isfinite(array)]
    result["quantiles"] = ({str(q): float(v) for q, v in
                            zip(QUANTILES, np.quantile(finite, QUANTILES))}
                           if finite.size else {str(q): None for q in QUANTILES})
    return result


def sampled_stats(dataset, max_samples=128):
    if not dataset.shape:
        indices, sampled = [], np.asarray(dataset[()])
        count = 1
    else:
        per_record = math.prod(dataset.shape[1:])
        if per_record > SAMPLE_ELEMENT_LIMIT:
            return {"status": "SKIPPED_OVERSIZED_RECORD", "axis": 0,
                    "record_elements": per_record, "element_limit": SAMPLE_ELEMENT_LIMIT,
                    "reason": "full streamed counts remain complete; no partial-record quantile invented"}
        count = min(max_samples, dataset.shape[0],
                    SAMPLE_ELEMENT_LIMIT // max(1, per_record))
        indices = (np.linspace(0, dataset.shape[0] - 1, count, dtype=np.int64).tolist()
                   if count else [])
        sampled = (np.asarray(dataset[indices]) if indices else
                   np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype))
    result = {"status": "SAMPLED", "selection": "fixed evenly spaced axis-0 indices",
              "maximum_records": max_samples, "selected_records": count,
              "axis0_indices": indices, "sampled_only": True,
              "quantile_method": "numpy linear; finite values only",
              "all_values": _describe_values(sampled), "band_axis_candidate": None}
    # Axis candidates are deliberately not treated as confirmed physical bands.
    band_axis = (1 if sampled.ndim == 4 and sampled.shape[1] in (1, 4, 8) else
                 3 if sampled.ndim == 4 and sampled.shape[-1] in (1, 4, 8) else None)
    if band_axis is not None:
        result["band_axis_candidate"] = band_axis
        result["band_semantics"] = "UNCONFIRMED; indices are not wavelength assignments"
        result["per_band_sampled"] = [dict(band_index=i, **_describe_values(np.take(sampled, i, axis=band_axis)))
                                      for i in range(sampled.shape[band_axis])]
    return result


def numeric_stats(dataset, chunk_bytes, max_samples):
    if dataset.dtype.kind not in "biuf":
        return {"status": "NOT_REAL_NUMERIC", "dtype": str(dataset.dtype)}
    result = _counts()
    for selection in _slices(dataset.shape, chunk_bytes // max(1, dataset.dtype.itemsize)):
        _accumulate(result, dataset[selection])
    result["scope"] = "ALL_DATASET_ELEMENTS_STREAMED"
    result["greater_than_2047_interpretation"] = "diagnostic count only; NOT an encoding or saturation assertion"
    result["zero_interpretation"] = "finite zero is NOT automatically NoData"
    return {"status": "COMPLETE", "full": result,
            "sampled": sampled_stats(dataset, max_samples)}


def _limited_counts(counter, limit=1024):
    entries = sorted(counter.items())
    return {"unique_count": len(entries), "counts": [dict(value=k, count=v) for k, v in entries[:limit]],
            "display_limit": limit, "display_truncated": len(entries) > limit}


def _grid(values):
    values = sorted(values)
    gaps = [b - a for a, b in zip(values, values[1:])]
    unique = sorted(set(gaps))
    return {"distinct_coordinates": len(values), "minimum": values[0] if values else None,
            "maximum": values[-1] if values else None, "minimum_positive_step": min(gaps) if gaps else None,
            "step_gcd": math.gcd(*gaps) if gaps else None,
            "unique_steps": unique[:128], "unique_steps_count": len(unique),
            "unique_steps_truncated": len(unique) > 128,
            "interpretation": "observed coordinate spacing; NOT proof of native layout/completeness"}


def meta_audit(dataset, chunk_bytes):
    result = {"scope": "ALL_META_ROWS", "roles_are_candidates": True,
              "column_roles_candidate": ["scene_idx", "ms_x", "ms_y", "category_idx"],
              "roles_source": "user-supplied SAMPLES.md description; not independently verified",
              "shape": list(dataset.shape)}
    if dataset.ndim != 2 or dataset.shape[1] != 4 or dataset.dtype.kind not in "iuf":
        return dict(result, status="UNSUPPORTED_META_SCHEMA")
    rows, invalid = 0, 0
    scene_counts, category_counts, pair_counts, coordinates = Counter(), Counter(), Counter(), Counter()
    grids = {}
    minimum, maximum = [None] * 4, [None] * 4
    stride = max(1, chunk_bytes // max(1, dataset.dtype.itemsize * 4))
    for start in range(0, dataset.shape[0], stride):
        block = np.asarray(dataset[start:start + stride])
        for row in block:
            rows += 1
            if not np.isfinite(row).all() or not np.equal(row, np.floor(row)).all():
                invalid += 1
                continue
            scene, x, y, category = values = tuple(int(v) for v in row)
            for i, value in enumerate(values):
                minimum[i] = value if minimum[i] is None else min(minimum[i], value)
                maximum[i] = value if maximum[i] is None else max(maximum[i], value)
            scene_counts[scene] += 1
            category_counts[category] += 1
            pair_counts[(scene, category)] += 1
            coordinates[(scene, x, y)] += 1
            grid = grids.setdefault(scene, (set(), set()))
            grid[0].add(x)
            grid[1].add(y)
    return dict(result, status="COMPLETE" if not invalid else "INVALID_META_ROWS",
                rows=rows, invalid_nonfinite_or_noninteger_rows=invalid,
                column_minimum=minimum, column_maximum=maximum,
                scenes=_limited_counts(scene_counts), categories=_limited_counts(category_counts),
                scene_category_counts=[{"scene_idx": s, "category_idx": c, "count": n}
                                       for (s, c), n in sorted(pair_counts.items())],
                distinct_scene_xy=len(coordinates),
                duplicated_scene_xy_keys=sum(n > 1 for n in coordinates.values()),
                duplicate_scene_xy_extra_rows=sum(n - 1 for n in coordinates.values()),
                grid_by_scene=[{"scene_idx": scene, "x": _grid(xs), "y": _grid(ys)}
                               for scene, (xs, ys) in sorted(grids.items())],
                spatial_overlap_verified=False,
                note="Duplicate keys checked with scene ID. Different coordinates can still have overlapping support.")


def _schema_inventory(handle):
    groups, datasets, aliases, seen = [], [], [], {}

    def visit(group, path):
        address = h5py.h5o.get_info(group.id).addr
        if address in seen:
            aliases.append({"path": path, "hard_link_to": seen[address]})
            return
        seen[address] = path
        groups.append({"path": path, "attrs": attributes(group)})
        for name in sorted(group.keys()):
            child_path = path.rstrip("/") + "/" + name
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise UnsafeHDF5(f"BLOCKED_HDF5_LINK: {child_path}: {type(link).__name__}")
            child = group[name]
            if isinstance(child, h5py.Group):
                visit(child, child_path)
            elif isinstance(child, h5py.Dataset):
                if child.is_virtual or child.external:
                    raise UnsafeHDF5(f"BLOCKED_EXTERNAL_DATASET: {child_path}")
                address = h5py.h5o.get_info(child.id).addr
                if address in seen:
                    aliases.append({"path": child_path, "hard_link_to": seen[address]})
                    continue
                seen[address] = child_path
                datasets.append(child_path)
            else:
                raise UnsafeHDF5(f"BLOCKED_UNKNOWN_HDF5_OBJECT: {child_path}")
    visit(handle, "/")
    return groups, datasets, aliases


def geometry_candidates(datasets):
    by_parent = {}
    for row in datasets:
        parent, _, name = row["path"].rpartition("/")
        by_parent.setdefault(parent or "/", {})[name.lower()] = row
    result = []
    for parent, arrays in sorted(by_parent.items()):
        if not {"pan", "ms"}.issubset(arrays):
            continue
        p, m = arrays["pan"]["shape"], arrays["ms"]["shape"]
        ratio4 = (len(p) == len(m) == 4 and p[0] == m[0] and p[1] == 1 and
                  m[1] == 8 and p[2] == 4 * m[2] and p[3] == 4 * m[3])
        gt = arrays.get("gt", {}).get("shape")
        rr = bool(ratio4 and gt == [p[0], 8, p[2], p[3]])
        result.append({"group": parent, "pan_shape": p, "ms_shape": m, "gt_shape": gt,
                       "ratio4_nchw_candidate": ratio4,
                       "classification_candidate": "RR_TUPLE" if rr else
                           "NATIVE_PAIR_WITHOUT_GT" if ratio4 and gt is None else "UNRESOLVED",
                       "semantic_verified": False,
                       "warning": "Array names and shapes alone do not establish native/RR semantics."})
    return result


def audit_h5(path, *, max_samples=128, chunk_bytes=16 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise UnsafeHDF5("BLOCKED_FILE: regular non-symlink file required: " + str(path))
    if not 1 <= max_samples <= 128 or chunk_bytes < 1024:
        raise ValueError("max_samples must be 1..128 and chunk_bytes >=1024")
    before = path.stat()
    row = {"path": str(path.resolve()), "bytes": before.st_size, "sha256": file_sha(path),
           "mtime_ns": before.st_mtime_ns, "open_mode": "r", "groups": [], "datasets": []}
    with h5py.File(path, "r") as handle:
        groups, paths, aliases = _schema_inventory(handle)  # Reject links before any array read.
        row.update(groups=groups, hard_link_aliases=aliases)
        for key in paths:
            d = handle[key]
            item = {"path": key, "shape": list(d.shape), "dtype": str(d.dtype),
                    "dtype_kind": d.dtype.kind, "chunks": json_value(d.chunks),
                    "compression": d.compression, "compression_opts": json_value(d.compression_opts),
                    "shuffle": bool(d.shuffle), "fletcher32": bool(d.fletcher32),
                    "scaleoffset": json_value(d.scaleoffset), "fillvalue": json_value(d.fillvalue),
                    "attrs": attributes(d), "numeric": numeric_stats(d, chunk_bytes, max_samples)}
            if key.rsplit("/", 1)[-1].lower() == "meta":
                item["meta_audit"] = meta_audit(d, chunk_bytes)
            row["datasets"].append(item)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise UnsafeHDF5("INPUT_CHANGED_DURING_AUDIT: " + str(path))
    row["geometry_candidates"] = geometry_candidates(row["datasets"])
    native = any(x["classification_candidate"] == "NATIVE_PAIR_WITHOUT_GT"
                 for x in row["geometry_candidates"])
    row.update(status="AUDITED_NOT_ADMITTED_FOR_TRAINING", native_fr_verified=False,
               native_fr_presence="CANDIDATE_REQUIRES_METADATA_VERIFICATION" if native else
                   "NOT_IDENTIFIED_FROM_SCHEMA",
               required_unresolved=["semantic mapping", "band ordering", "encoding/normalization",
                                    "original coordinate layout and phase", "RR generation provenance",
                                    "valid mask/NoData", "native FR source"],
               training_started=False, split_frozen=False)
    return row


def audit_root(root, *, max_samples=128, chunk_bytes=16 * 1024 * 1024):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("input root must be a non-symlink directory")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".h5", ".hdf5"))
    if not paths:
        raise ValueError("no .h5/.hdf5 files under supplied root")
    rows = []
    for path in paths:
        # Reject a symlink in any relative ancestor as well as the file itself.
        if any(p.is_symlink() for p in (path, *path.parents) if p == root or root in p.parents):
            rows.append({"path": str(path), "status": "BLOCKED_FILE_SYMLINK"})
            continue
        try:
            rows.append(audit_h5(path, max_samples=max_samples, chunk_bytes=chunk_bytes))
        except (OSError, ValueError, TypeError) as error:
            rows.append({"path": str(path.resolve()), "status": "BLOCKED_INPUT_AUDIT",
                         "error_type": type(error).__name__, "reason": str(error)})
    return {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root.resolve()), "auditor_sha256": file_sha(__file__),
            "versions": {"numpy": np.__version__, "h5py": h5py.__version__,
                         "hdf5": h5py.version.hdf5_version},
            "read_only": True, "full_numeric_scan": True, "maximum_sampled_records": max_samples,
            "file_count": len(rows), "blocked_count": sum(r["status"].startswith("BLOCKED") for r in rows),
            "files": rows, "admission_status": "NOT_A_TRAINING_ADMISSION",
            "split_frozen": False, "training_started": False,
            "notes": ["User-selected category samples supersede original plan's fixed five file names.",
                      "This audit inventories all supplied H5 files; it does not enroll them as training scenes.",
                      "Scalar statistics do not verify encoding, bands, independence, or native geometry."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing supplied H5 files; read-only")
    parser.add_argument("--output", type=Path, help="new JSON artifact only; existing paths are never overwritten")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--chunk-mib", type=int, default=16)
    args = parser.parse_args(argv)
    if not 1 <= args.max_samples <= 128 or not 1 <= args.chunk_mib <= 256:
        parser.error("max-samples must be 1..128; chunk-mib must be 1..256")
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            parser.error("output already exists; refuse overwrite")
        if args.output.suffix.lower() != ".json" or not args.output.parent.is_dir():
            parser.error("output must be a .json path with an existing parent directory")
        if args.output.resolve().is_relative_to(args.root.resolve()):
            parser.error("output must be outside the read-only input root")
    report = audit_root(args.root, max_samples=args.max_samples, chunk_bytes=args.chunk_mib * 1024 * 1024)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        print(json.dumps({"output": str(args.output.resolve()), "sha256": file_sha(args.output),
                          "files": report["file_count"], "blocked": report["blocked_count"]}))
    else:
        sys.stdout.write(payload)
    return 2 if report["blocked_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
