"""FH20R1 verified, read-only FH12 import and explicit N2-only calibration.

Historical source identities stay historical. A separate content-bound
bridge records current-reader parity; it is never an exact-resume override.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from fh12.common import ROOT
from fh12.data import (AUGMENTATION, RECIPE, FH12Dataset, build_dataset, canonical_sha,
                       sha256_file, write_immutable_json)
from fh12.model import build_model, state_hash
from fh20r1.historical import run_parity, validate_source_identity
from fh20r1.plan import REFERENCE_ASSETS

N2_RUN = "FH20R1_S2_T_PL_W112_D123_WV3_TS71001_N2INIT50_v1"
DONOR_RUN = "PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT"
DONOR_SHA = "9d4cbf218cd8f6743942115e882d87d664d663935a1a1ba295a9903e27f3f816"


def _read(path):
    return json.loads(Path(path).read_text())


def _consumer_identity(root):
    from fh20r1.common import source_identity
    return source_identity(root)


def _relocate(path, root):
    """Resolve stored paths without modifying the original manifest/config."""
    path, root = Path(path), Path(root).resolve()
    # FH12 resolved symlink targets can legitimately live outside the repo,
    # e.g. /home/.../data/datasets/wv3. Existing originals are read and SHA
    # verified, never rewritten or incorrectly re-rooted under repo/data.
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute():
        if ".." in path.parts:
            raise ValueError("Reference path escapes repository")
        return root / path
    for anchor in ("work_dir", "data", "assets"):
        if anchor in path.parts:
            offset = path.parts.index(anchor)
            return root.joinpath(*path.parts[offset:])
    raise ValueError(f"Historical path has no supported repository anchor: {path}")


def _spec(alias, server):
    if alias not in REFERENCE_ASSETS or REFERENCE_ASSETS[alias]["server_id"] != server:
        raise ValueError("FH20R1 reference alias belongs to a different local server")
    item = REFERENCE_ASSETS[alias]
    expected = item.get("expected_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("FH20R1 reference requires full checkpoint SHA from supplied Cases CSV")
    return dict(server=server, layout=item["input_layout"], seed=item["teacher_seed"],
                run_id=item["teacher_run_id"], expected_sha256=expected)


def bridge_path(server, alias, root=ROOT):
    return Path(root) / "work_dir" / "_fh20r1" / server / "imported_refs" / alias / "bridge_manifest.json"


def _verified_file(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"Historical asset bytes do not match recorded SHA: {path}")


def _validate_data(data, lp_manifest, root):
    resolved = copy.deepcopy(data)
    counts, sides = {"train": 9714, "val": 1080, "rr": 20, "fr": 20}, {"train": 64, "val": 64, "rr": 256, "fr": 512}
    if data.get("recipe") != RECIPE or data.get("augmentation") != AUGMENTATION:
        raise ValueError("FH12 LP/augmentation definition differs; no cache regeneration fallback")
    if lp_manifest.get("recipe") != RECIPE:
        raise ValueError("Historical LP recipe changed")
    for split in counts:
        item = data["splits"][split]
        source, lp = _relocate(item["dataroot"], root), _relocate(item["lpan_path"], root)
        _verified_file(source, item["sha256"])
        _verified_file(lp, item["lpan_sha256"])
        n, h = counts[split], sides[split]
        with h5py.File(source, "r") as f:
            for key, shape in (("pan", (n, 1, h, h)), ("ms", (n, 8, h//4, h//4)), ("lms", (n, 8, h, h))):
                if f[key].shape != shape:
                    raise ValueError(f"Wrong FH12 {split}/{key} geometry")
            if split != "fr" and f["gt"].shape != (n, 8, h, h):
                raise ValueError(f"Wrong FH12 {split} GT geometry")
        with h5py.File(lp, "r") as f:
            if (f["lpan"].shape != (n, 1, h//4, h//4) or f.attrs.get("source_sha256") != item["sha256"]
                    or f.attrs.get("sample_order_sha256") != item["sample_order_sha256"]
                    or f.attrs.get("recipe_sha256") != canonical_sha(RECIPE)):
                raise ValueError("Historical LP cache source/phase/order mismatch")
        for key in ("sha256", "lpan_sha256", "sample_order_sha256"):
            if lp_manifest["splits"][split][key] != item[key]:
                raise ValueError("Historical dataset/LP manifests disagree")
        resolved["splits"][split]["dataroot"] = str(source)
        resolved["splits"][split]["lpan_path"] = str(lp)
    return resolved


def _q_values(origin, resolved):
    with np.load(resolved["q_cache_path"], allow_pickle=False) as cache:
        q = cache["q"].copy()
        indices = cache["calibration_indices"].copy()
    if q.shape != (9714, 4) or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Historical q must cover all 9714 training patches x four views")
    if (indices.shape != (3072,) or len(np.unique(indices)) != 3072 or indices.min() < 0
            or indices.max() >= len(q) or canonical_sha(indices.tolist()) != origin["calibration_indices_sha256"]):
        raise ValueError("Historical calibration subset identity mismatch")
    if (not np.isfinite(origin["tau_R"]) or origin["tau_R"] < 1e-6 or not np.isfinite(origin["q_ref"])
            or origin["q_ref"] <= 0 or float(np.median(q[indices])) != origin["q_ref"]):
        raise ValueError("Historical tau/q_ref invalid; never substitute snapshot numbers")
    return q, indices


def validate_origin(alias, server, root=ROOT):
    """Read-only original bytes/config/data/LP/source/q validation, no GPU."""
    root, spec = Path(root).resolve(), _spec(alias, server)
    path = root / "work_dir/_fh12" / server / "references" / spec["run_id"] / "reference_manifest.json"
    origin = _read(path)
    if (origin.get("schema") != "FH12_REFERENCE_v1" or origin.get("teacher_run_id") != spec["run_id"]
            or origin.get("server") != server or origin.get("teacher_layout") != spec["layout"]
            or origin.get("teacher_update") != 50000):
        raise ValueError("Historical reference alias/layout/run/exact50K mismatch")
    if origin["teacher_checkpoint_sha256"] != spec["expected_sha256"]:
        raise ValueError("Historical whole-checkpoint SHA differs from supplied Cases CSV")
    validate_source_identity(origin["source_identity"])
    pairs = {"teacher_checkpoint": "teacher_checkpoint_sha256", "teacher_training_state": "teacher_training_state_sha256",
             "teacher_config": "teacher_config_file_sha256", "teacher_checkpoint_identity": "teacher_checkpoint_identity_sha256",
             "calibration_path": "calibration_sha256", "q_cache_path": "q_cache_sha256",
             "dataset_manifest_path": "dataset_manifest_sha256", "lpan_manifest_path": "lpan_manifest_sha256"}
    resolved = {}
    for key, hash_key in pairs.items():
        resolved[key] = str(_relocate(origin[key], root))
        _verified_file(resolved[key], origin[hash_key])
    resolved["origin_manifest"] = str(path)
    init = root / "work_dir" / spec["run_id"] / "init_manifest.json"
    if not init.is_file():
        raise FileNotFoundError(f"Historical named initialization manifest missing: {init}")
    resolved["init_manifest"] = str(init)
    cfg = yaml.safe_load(Path(resolved["teacher_config"]).read_text())
    f, m = cfg["fh12"], cfg["model_args"]
    if (canonical_sha(cfg) != origin["teacher_config_sha256"] or f["role"] != "T"
            or f["server_id"] != server or f["input_layout"] != spec["layout"] or cfg["seed"] != spec["seed"]
            or m["hidden_size"] != 112 or m["depth"] != [1, 2, 3]):
        raise ValueError("Historical Teacher configuration differs from explicit FH12 reference")
    identity = _read(resolved["teacher_checkpoint_identity"])
    if (identity["update"] != 50000 or identity["model_sha256"] != origin["teacher_checkpoint_sha256"]
            or identity["config_sha256"] != origin["teacher_config_sha256"]
            or identity["source_identity"] != origin["source_identity"]
            or identity["training_state_sha256"] != origin["teacher_training_state_sha256"]):
        raise ValueError("Historical checkpoint identity contradicts reference")
    # Hashes were verified before deserializing this trusted local full-state.
    state = torch.load(resolved["teacher_training_state"], map_location="cpu", weights_only=False)
    for key, wanted in (("update", 50000), ("config_sha256", origin["teacher_config_sha256"]),
                        ("model_sha256", origin["teacher_checkpoint_sha256"]), ("source_identity", origin["source_identity"])):
        if state.get(key) != wanted:
            raise ValueError(f"Historical training state differs: {key}")
    del state
    calibration = _read(resolved["calibration_path"])
    for key in ("source_identity", "tau_R", "q_ref", "teacher_layout", "teacher_update", "teacher_checkpoint_sha256",
                "calibration_indices_sha256", "q_shape"):
        if calibration[key] != origin[key]:
            raise ValueError(f"Historical calibration/reference mismatch: {key}")
    if origin["LP_recipe"] != RECIPE or origin["augmentation"] != AUGMENTATION:
        raise ValueError("Historical frequency/augmentation recipe is not FH12")
    data = _validate_data(_read(resolved["dataset_manifest_path"]), _read(resolved["lpan_manifest_path"]), root)
    train = data["splits"]["train"]
    for data_key, ref_key in (("sha256", "train_sha256"), ("lpan_sha256", "train_lpan_sha256"),
                               ("sample_order_sha256", "train_sample_order_sha256")):
        if train[data_key] != origin[ref_key]:
            raise ValueError("Historical calibration train source does not match data manifest")
    q, _ = _q_values(origin, resolved)
    return cfg, origin, data, q, resolved


def _load_frozen(cfg, path, device, field="fh12"):
    f, m = cfg[field], cfg["model_args"]
    model, _ = build_model(f["input_layout"], m["hidden_size"], m["depth"], cfg["seed"], role="T")
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model.to(device).eval().requires_grad_(False)


def import_reference(server, alias=None, root=ROOT, device="cuda"):
    alias = alias or f"F{str(server)[1:]}"
    root = Path(root).resolve()
    destination = bridge_path(server, alias, root)
    if destination.exists():
        # Cached bridge reuse still rechecks original bytes, but never repeats
        # parity work solely to accrue effective time.
        load_reference(alias, server, root, device="cpu")
        return destination
    cfg, origin, data, q, resolved = validate_origin(alias, server, root)
    parity = run_parity(root, origin, resolved, data, device=device)
    bridge = {"schema": "FH20R1_REFERENCE_BRIDGE_v1", "status": "PASS", "complete": True,
              "alias": alias, "server": server, "origin_reference": origin,
              "origin_manifest_sha256": sha256_file(resolved["origin_manifest"]),
              "origin_config": cfg, "resolved_artifacts": resolved, "dataset_manifest": data,
              "resolved_artifact_sha256": {k: sha256_file(p) for k, p in resolved.items()},
              "origin_source_identity": origin["source_identity"], "consumer_source_identity": _consumer_identity(root),
              "teacher_checkpoint_sha256": origin["teacher_checkpoint_sha256"],
              "tau_R": origin["tau_R"], "q_ref": origin["q_ref"], "parity": parity,
              "expected_sha_scope": "Supplied Cases CSV full SHA256 plus original whole-SHA chained artifacts",
              "original_artifacts_modified": False, "exact_resume_authorized": False}
    write_immutable_json(destination, bridge)
    return destination


def load_reference(alias, server, root=ROOT, device="cuda"):
    root = Path(root).resolve()
    bridge = _read(bridge_path(server, alias, root))
    if (bridge.get("schema") != "FH20R1_REFERENCE_BRIDGE_v1" or not bridge.get("complete")
            or bridge.get("status") != "PASS" or bridge.get("alias") != alias or bridge.get("server") != server):
        raise ValueError("Unverified or wrong FH20R1 reference bridge")
    if bridge["consumer_source_identity"] != _consumer_identity(root):
        raise ValueError("Consumer release changed after reference parity; do not edit historical source identity")
    for key, expected in bridge["resolved_artifact_sha256"].items():
        _verified_file(bridge["resolved_artifacts"][key], expected)
    if alias == "N2PL":
        return _load_n2_reference(bridge, server, root, device)
    cfg, origin, data, q, resolved = validate_origin(alias, server, root)
    if (origin != bridge["origin_reference"] or cfg != bridge["origin_config"]
            or data != bridge["dataset_manifest"] or resolved != bridge["resolved_artifacts"]):
        raise ValueError("Historical identity changed since bridge publication")
    return _load_frozen(cfg, resolved["teacher_checkpoint"], device), cfg, bridge, q


def load_training_reference(alias, server, root=ROOT, device="cuda"):
    teacher, cfg, bridge, q = load_reference(alias, server, root, device)
    return teacher, bridge["origin_reference"], q, bridge


def calibration_indices(alias, server, root=ROOT):
    # Verification is mandatory, even for diagnostic-only subset access.
    _, _, bridge, _ = load_reference(alias, server, root, device="cpu")
    with np.load(bridge["resolved_artifacts"]["q_cache_path"], allow_pickle=False) as cache:
        return cache["calibration_indices"].copy()


def load_donor_aligner(root=ROOT, server="s2"):
    """Return A-only state/report. Any unavailable donor blocks this arm only."""
    root = Path(root).resolve()
    if server != "s2":
        raise ValueError("N2 donor belongs only to the s2 initialization arm")
    run = root / "work_dir" / DONOR_RUN
    paths = {"container": run / "last/model.safetensors", "config": run / "meta/config.yaml",
             "meta": run / "last_meta.json", "lineage": run / "dataset_hashes.json"}
    report = {"donor_run": DONOR_RUN, "expected_container_sha256": DONOR_SHA,
              "donor_available": False, "status": "BLOCKED_DONOR"}
    try:
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(str(path))
        _verified_file(paths["container"], DONOR_SHA)
        cfg = yaml.safe_load(paths["config"].read_text())
        if (cfg.get("trainer") != "po" or cfg.get("num_bands") != 8 or cfg["po"].get("case") != "N2_SG"
                or cfg["po"].get("radius_hr") != 2 or _read(paths["meta"]).get("step") != 50000):
            raise ValueError("N2 donor is not 8-band N2_SG/radius2/exact50K lineage")
        from pa.offset import aligner_margin
        if aligner_margin(cfg["po"]["radius_hr"]) != 4:
            raise ValueError("N2 donor aligner view must be margin4")
        train = _relocate(cfg["train_feeder_args"]["dataroot"], root)
        lineage = _read(paths["lineage"])["train_feeder_args"]
        _verified_file(train, lineage["sha256"])
        f2_data = _read(root / "work_dir/_fh12/s2/dataset_manifest.json")
        if lineage["sha256"] != f2_data["splits"]["train"]["sha256"]:
            raise ValueError("N2/F2 train lineage mismatch")
        state = load_file(str(paths["container"]), device="cpu")
        aligner_state = {k[len("aligner."):]: v.clone() for k, v in state.items() if k.startswith("aligner.")}
        from pa.aligner import PANGlobalAligner
        with torch.random.fork_rng(devices=[]):
            aligner = PANGlobalAligner(8)
        aligner.load_state_dict(aligner_state, strict=True)
        if not all(torch.isfinite(v).all() for v in aligner_state.values()):
            raise ValueError("Nonfinite donor aligner")
        # Preserve even a legitimately small learned head; no resetting occurs.
        report.update(status="VERIFIED", donor_available=True, container_sha256=DONOR_SHA,
                      extracted_aligner_sha256=state_hash(aligner_state), source_step=50000, margin=4, ms_bands=8,
                      train_sha256=lineage["sha256"], source_paths={k: str(v) for k, v in paths.items()},
                      source_file_sha256={k: sha256_file(v) for k, v in paths.items()},
                      fc2_nonzero_count=int(torch.count_nonzero(aligner_state["fc2.weight"])) + int(torch.count_nonzero(aligner_state["fc2.bias"])),
                      extracted_keys=sorted(aligner_state), load_policy="A_ONLY_NO_HEAD_RESET", U_loaded=False,
                      optimizer_loaded=False, scheduler_loaded=False)
        return aligner_state, report
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError, yaml.YAMLError) as exc:
        report.update(reason=f"{type(exc).__name__}: {exc}", fallback="F2_PL_vs_PLH", donor_available=False)
        return None, report


def calibrate_n2pl(teacher_run=N2_RUN, root=ROOT, server="s2", device="cuda", batch_size=64):
    """New N2PL tau/q using F2's actual stored calibration IDs, never a new draw."""
    if server != "s2" or Path(teacher_run).name != N2_RUN:
        raise ValueError("FH20R1 only the explicit N2PL exact50K may be recalibrated")
    root = Path(root).resolve()
    target = bridge_path(server, "N2PL", root)
    if target.exists():
        load_reference("N2PL", server, root, device="cpu")
        return target
    _, _, f2, _ = load_reference("F2", server, root, device="cpu")
    with np.load(f2["resolved_artifacts"]["q_cache_path"], allow_pickle=False) as cache:
        indices = cache["calibration_indices"].copy()
    run = root / "work_dir" / N2_RUN
    cfg_path, candidate = run / "meta/config.resolved.yaml", run / "candidates/50000"
    cfg, identity = yaml.safe_load(cfg_path.read_text()), _read(candidate / "identity.json")
    from fh20r1.plan import build_config as registered_config, case_for
    if cfg != registered_config(case_for(N2_RUN)):
        raise ValueError("N2PL calibration config differs from registered A-only N2INIT experiment")
    release = _consumer_identity(root)
    if identity["update"] != 50000 or identity["config_sha256"] != canonical_sha(cfg) or identity["source_identity"] != release:
        raise ValueError("N2PL calibration requires this release's registered exact50K")
    _verified_file(candidate / "model.safetensors", identity["model_sha256"])
    _verified_file(candidate / "training_state.pt", identity["training_state_sha256"])
    from fh20r1.common import load_checkpoint_model
    model, _ = load_checkpoint_model(cfg, candidate, device, expected_source=release)
    data = f2["dataset_manifest"]
    dataset = build_dataset(data, "train", root=root)
    from fh12.calibration import compute_calibration, _write_npz_immutable
    calibration, arrays = compute_calibration(model, dataset, indices, device=device, batch_size=batch_size)
    asset_dir = target.parent
    q_path, cal_path = asset_dir / "q_cache.npz", asset_dir / "calibration.json"
    _write_npz_immutable(q_path, arrays)
    calibration.update(schema="FH20R1_N2PL_CALIBRATION_v1", source_identity=release,
                       teacher_checkpoint_sha256=identity["model_sha256"], teacher_update=50000,
                       teacher_layout="PL", calibration_subset_origin="F2 actual q_cache calibration_indices")
    write_immutable_json(cal_path, calibration)
    resolved = {"teacher_checkpoint": str(candidate / "model.safetensors"), "teacher_config": str(cfg_path),
                "teacher_checkpoint_identity": str(candidate / "identity.json"), "teacher_training_state": str(candidate / "training_state.pt"),
                "q_cache_path": str(q_path), "calibration_path": str(cal_path), "init_manifest": str(run / "init_manifest.json")}
    origin = {"tau_R": calibration["tau_R"], "q_ref": calibration["q_ref"], "q_shape": calibration["q_shape"],
              "teacher_checkpoint_sha256": identity["model_sha256"], "teacher_checkpoint": resolved["teacher_checkpoint"],
              "teacher_config": str(cfg_path), "teacher_layout": "PL", "teacher_update": 50000, "teacher_run_id": N2_RUN,
              "calibration_indices_sha256": calibration["calibration_indices_sha256"], "q_cache_path": str(q_path),
              "source_identity": release, "LP_recipe": RECIPE, "augmentation": AUGMENTATION}
    origin.update(server="s2", teacher_config_sha256=canonical_sha(cfg),
                  teacher_config_file_sha256=sha256_file(cfg_path),
                  q_cache_sha256=sha256_file(q_path), calibration_path=str(cal_path),
                  calibration_sha256=sha256_file(cal_path),
                  teacher_checkpoint_identity=resolved["teacher_checkpoint_identity"],
                  teacher_checkpoint_identity_sha256=sha256_file(resolved["teacher_checkpoint_identity"]),
                  teacher_training_state=resolved["teacher_training_state"],
                  teacher_training_state_sha256=sha256_file(resolved["teacher_training_state"]),
                  train_sha256=data["splits"]["train"]["sha256"],
                  train_lpan_sha256=data["splits"]["train"]["lpan_sha256"],
                  train_sample_order_sha256=data["splits"]["train"]["sample_order_sha256"],
                  dataset_manifest_path=f2["resolved_artifacts"]["dataset_manifest_path"],
                  dataset_manifest_sha256=f2["origin_reference"]["dataset_manifest_sha256"],
                  lpan_manifest_path=f2["resolved_artifacts"]["lpan_manifest_path"],
                  lpan_manifest_sha256=f2["origin_reference"]["lpan_manifest_sha256"])
    bridge = {"schema": "FH20R1_REFERENCE_BRIDGE_v1", "status": "PASS", "complete": True, "alias": "N2PL", "server": "s2",
              "origin_reference": origin, "origin_config": cfg, "resolved_artifacts": resolved,
              "resolved_artifact_sha256": {k: sha256_file(p) for k, p in resolved.items()}, "dataset_manifest": data,
              "consumer_source_identity": release, "origin_source_identity": release,
              "teacher_checkpoint_sha256": identity["model_sha256"], "tau_R": origin["tau_R"], "q_ref": origin["q_ref"],
              "F2_bridge_sha256": sha256_file(bridge_path("s2", "F2", root)),
              "calibration_indices_origin_sha256": f2["origin_reference"]["calibration_indices_sha256"],
              "historical_forward_bridge": "F2 verified reader; N2PL is a newly trained native checkpoint", "original_artifacts_modified": False}
    write_immutable_json(target, bridge)
    return target


def _load_n2_reference(bridge, server, root, device):
    if server != "s2" or sha256_file(bridge_path("s2", "F2", root)) != bridge["F2_bridge_sha256"]:
        raise ValueError("N2PL F2 calibration-subset lineage changed")
    f2 = _read(bridge_path("s2", "F2", root))
    # Validate original dataset/LP bytes on every consumption, without rewriting.
    _, _, data, _, _ = validate_origin("F2", "s2", root)
    if data != bridge["dataset_manifest"]:
        raise ValueError("N2PL input dataset differs from original F2")
    origin, cfg, resolved = bridge["origin_reference"], bridge["origin_config"], bridge["resolved_artifacts"]
    q, indices = _q_values(origin, resolved)
    if canonical_sha(indices.tolist()) != f2["origin_reference"]["calibration_indices_sha256"]:
        raise ValueError("N2PL did not use F2's exact calibration base IDs")
    cal = _read(resolved["calibration_path"])
    if cal["tau_R"] != origin["tau_R"] or cal["q_ref"] != origin["q_ref"]:
        raise ValueError("N2PL manifest/calibration scalar mismatch")
    from fh20r1.common import load_checkpoint_model
    model, identity = load_checkpoint_model(cfg, Path(resolved["teacher_checkpoint"]).parent, device,
                                            expected_source=bridge["consumer_source_identity"])
    if identity["update"] != 50000 or identity["model_sha256"] != origin["teacher_checkpoint_sha256"]:
        raise ValueError("N2PL Teacher is not the pinned exact50K")
    return model.eval().requires_grad_(False), cfg, bridge, q
