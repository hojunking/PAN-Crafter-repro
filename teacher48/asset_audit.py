"""Read-only discovery of existing corpus and recipe evidence, not admission.

The installed PanCollection patches contain arrays, not original-acquisition
lineage. This audit never promotes benchmark FR20 or RR-upsampled patches to
native FR training, and never uses a row index as a source scene identifier.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _plain(value.item())
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def h5_metadata(path):
    """Small headers only: no tensor loading and no misleading full-file hash."""
    path = Path(path)
    result = {"path": str(path), "resolved_path": str(path.resolve()), "exists": path.is_file()}
    if not result["exists"]:
        return result
    result["size_bytes"] = path.stat().st_size
    with h5py.File(path, "r") as source:
        result["attributes"] = {k: _plain(v) for k, v in source.attrs.items()}
        result["datasets"] = {}
        result["groups"] = []
        for name, entry in source.items():
            if isinstance(entry, h5py.Dataset):
                result["datasets"][name] = {"shape": list(entry.shape), "dtype": str(entry.dtype),
                    "attributes": {k: _plain(v) for k, v in entry.attrs.items()}}
            else:
                result["groups"].append(name)
    result["full_file_sha256_verified"] = False
    return result


def discover_existing_assets(repo):
    root = Path(repo).resolve()
    catalog_path = root / "ablr2/sensor_sources.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema") != "ABLR2_SENSOR_SOURCES_v1":
        raise ValueError("Unsupported existing sensor source catalog")
    old_plan = root / "research_log/past/PAN_Aligner_TeacherOnly_NativeLMS_GTStructure_S135_3Seeds_2026-10-06_v2.md"
    if not old_plan.is_file():
        old_plan = root / "research_log/PAN_Aligner_TeacherOnly_NativeLMS_GTStructure_S135_3Seeds_2026-10-06_v2.md"
    result = {"schema": "TEACHER48_EXISTING_ASSET_AUDIT_V1", "admission_granted": False,
              "scope": "Existing registered WV3/QB/GF2 sources and local TA2 run metadata; not whole-machine absence proof",
              "catalog_path": str(catalog_path), "catalog_sha256": _sha(catalog_path),
              "sensors": {}, "prior_plan_evidence": None,
              "prohibited_shortcuts": ["FR20_BENCHMARK_AS_TRAINING", "UPSAMPLED_RR_AS_NATIVE_FR",
                                       "H5_ROW_AS_SOURCE_SCENE", "UNVERIFIED_GEOGRAPHIC_DISJOINTNESS"]}
    if old_plan.is_file():
        result["prior_plan_evidence"] = {"path": str(old_plan), "sha256": _sha(old_plan),
            "recipe_section": "lines435-438: batch48 AdamW U1e-4 A1e-5 WD.01 warmup100 cosine FP32",
            "source_limitations": "lines217,389-392: patch-level registry; true large-field needs independent DATA_REVISION"}
    for sensor in ("WV3", "QB", "GF2"):
        item = catalog["sensors"][sensor]
        splits = {}
        for split, entry in item["splits"].items():
            splits[split] = {**h5_metadata(root / entry["path"]),
                             "catalog_expected_sha256": entry["sha256"],
                             "source_identity": entry["source_identity"]}
        blocks = []
        if not splits["train"]["exists"]:
            blocks.append("REGISTERED_RR_TRAIN_FILE_MISSING")
        if "fr_train" not in splits:
            blocks.append("NO_REGISTERED_TRAINING_ONLY_NATIVE_FR_SOURCE")
        train = splits["train"]
        known_keys = set(train.get("datasets", {}))
        scene_fields = {"source_scene_id", "scene_id", "acquisition_id", "parent_scene_id"}
        if not scene_fields.intersection(known_keys | set(train.get("attributes", {}))):
            blocks.append("RR_SOURCE_SCENE_CROP_LINEAGE_NOT_EMBEDDED")
        # Existing catalog explicitly disclaims geographic separation.
        blocks.append("TRAIN_EVAL_GEOGRAPHIC_OR_NEAR_CROP_SEPARATION_NOT_CERTIFIED")
        blocks.append("ACCEPTED_SOURCE_SCENE_PRIOR_CACHE_NOT_REGISTERED")
        result["sensors"][sensor] = {"splits": splits, "band_order": item["band_order"],
            "source_provenance": item.get("source_provenance", {}), "blockers": blocks,
            "status": "BLOCKED_DATA_PROTOCOL", "baseline_evidence": []}

    old_runs = root / "work_dir/aligner_teacher_lms_v2"
    if old_runs.is_dir():
        for path in sorted(old_runs.glob("*/runs/*/config.json")):
            cfg = json.loads(path.read_text(encoding="utf-8"))
            sensor = cfg.get("dataset")
            if sensor not in result["sensors"] or cfg.get("aligner_policy") != "LEARNED":
                continue
            if result["sensors"][sensor]["baseline_evidence"]:
                continue
            # Recover only fields proven by the previous run. The new scene-
            # conditioned A architecture and FR batching were not run in TA2.
            recovered = {"teacher_architecture": {"width": cfg["width"], "depth": cfg["depth"]},
                "teacher_input_layout": cfg["input_layout"], "teacher_base": cfg["u_reference"],
                "optimizer": cfg["optimizer"], "optimizer_betas": cfg["betas"],
                "optimizer_eps": cfg["eps"], "lr_U0": cfg["lr_u"], "lr_A0": cfg["lr_a"],
                "weight_decay0": cfg["weight_decay"], "warmup_updates0": cfg["warmup_updates"],
                "effective_batch_RR": cfg["batch_size"], "microbatch_RR": cfg["micro_batch"],
                "accumulation_RR": cfg["batch_size"] // cfg["micro_batch"],
                "RR_train_HW": [64, 64], "precision": cfg["precision"],
                "augmentation": cfg["native_augmentation"], "historical_aligner_view_margin": cfg["aligner_margin"],
                "historical_NCC_ROI_margin": cfg["struct_margin"],
                "historical_scheduler": cfg["schedule"]}
            result["sensors"][sensor]["baseline_evidence"].append({"path": str(path),
                "sha256": _sha(path), "case_id": cfg["case_id"], "recovered_values": recovered,
                "not_auto_approved_for_new_method": ["FR_train_HW", "effective_batch_FR", "microbatch_FR",
                    "new_absolute_gradient_ROI", "prior_acceptance_policy", "new_scheduler_indexing"]})
    result["status"] = "BLOCKED_DATA_PROTOCOL"
    return result
