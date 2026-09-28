"""An immutable, portable F1 binding: paths may differ, artifact bytes may not.

Only explicit preparation publishes a binding. An original server path is never
rewritten to pretend a local F3/F5 teacher is F1. Native H5s are mapped explicitly
on import; the much smaller original LP caches travel with the reference bundle.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

from fh12.data import canonical_sha, sha256_file, write_immutable_json
from panda_rb.plan import ROOT, TEACHER_RUN

SCHEMA = "PANDA_RB_F1_BINDING_v1"
EXPECTED_F1_SHA = "04ef8e756ae8b229c67ef14daf1912f0658f2e5c8bfe30afa9e0b476486fc519"
PAIRS = {"teacher_checkpoint": "teacher_checkpoint_sha256",
         "teacher_training_state": "teacher_training_state_sha256",
         "teacher_config": "teacher_config_file_sha256",
         "teacher_checkpoint_identity": "teacher_checkpoint_identity_sha256",
         "calibration_path": "calibration_sha256", "q_cache_path": "q_cache_sha256",
         "dataset_manifest_path": "dataset_manifest_sha256", "lpan_manifest_path": "lpan_manifest_sha256"}


def _read(path):
    return json.loads(Path(path).read_text())


def _checked(path, digest):
    if not isinstance(digest, str) or len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError(f"F1 asset whole-file SHA mismatch: {path}")


def _origin_valid(origin):
    from fh20r1.historical import validate_source_identity
    if (origin.get("schema") != "FH12_REFERENCE_v1" or origin.get("server") != "s1"
            or origin.get("teacher_run_id") != TEACHER_RUN or origin.get("teacher_update") != 50000
            or origin.get("teacher_layout") != "P0" or origin.get("teacher_checkpoint_sha256") != EXPECTED_F1_SHA):
        raise ValueError("B01 requires original s1 F1 exact50K; no local Teacher substitution")
    validate_source_identity(origin["source_identity"])


def _validate(paths, data_paths):
    import torch
    import yaml
    from fh12.data import AUGMENTATION, RECIPE
    from fh20r1.references import _validate_data, _q_values
    origin = _read(paths["origin_manifest"])
    _origin_valid(origin)
    for key, field in PAIRS.items():
        _checked(paths[key], origin[field])
    cfg = yaml.safe_load(Path(paths["teacher_config"]).read_text())
    f, m = cfg["fh12"], cfg["model_args"]
    if (canonical_sha(cfg) != origin["teacher_config_sha256"] or f["role"] != "T"
            or f["server_id"] != "s1" or f["input_layout"] != "P0" or cfg["seed"] != 71001
            or m["hidden_size"] != 112 or m["depth"] != [1, 2, 3]):
        raise ValueError("F1 resolved config does not match historical anchor")
    identity = _read(paths["teacher_checkpoint_identity"])
    wanted = dict(update=50000, model_sha256=origin["teacher_checkpoint_sha256"],
                  config_sha256=origin["teacher_config_sha256"], source_identity=origin["source_identity"],
                  training_state_sha256=origin["teacher_training_state_sha256"])
    if any(identity.get(k) != v for k, v in wanted.items()):
        raise ValueError("F1 checkpoint identity contradicts historical reference")
    state = torch.load(paths["teacher_training_state"], map_location="cpu", weights_only=False)
    if any(state.get(k) != v for k, v in wanted.items() if k != "training_state_sha256"):
        raise ValueError("F1 full-state provenance contradicts historical reference")
    del state
    calibration = _read(paths["calibration_path"])
    for key in ("source_identity", "tau_R", "q_ref", "teacher_layout", "teacher_update",
                "teacher_checkpoint_sha256", "calibration_indices_sha256", "q_shape"):
        if calibration.get(key) != origin.get(key):
            raise ValueError("F1 calibration/reference mismatch: " + key)
    if origin["LP_recipe"] != RECIPE or origin["augmentation"] != AUGMENTATION:
        raise ValueError("F1 LP/augmentation recipe differs; no regeneration fallback")
    original_data = _read(paths["dataset_manifest_path"])
    original_lp = _read(paths["lpan_manifest_path"])
    if (original_data.get('schema') != 'FH12_DATA_v1' or original_data.get('sensor') != 'WV3'
            or original_data.get('max_pixel') != 2047.0 or original_data.get('server') != 's1'
            or original_lp.get('schema') != 'FH12_LPAN_v1' or original_lp.get('server') != 's1'):
        raise ValueError('Common F1 requires its original WV3/2047 data and LP manifests')
    if set(data_paths) != {"train", "val", "rr", "fr"}:
        raise ValueError("Explicit paths for all four native data/LP splits are required")
    resolved = copy.deepcopy(original_data)
    for split, pair in data_paths.items():
        if set(pair) != {"dataroot", "lpan_path"}:
            raise ValueError("Explicit dataroot/lpan_path required per split")
        for name, path in pair.items():
            path = Path(path)
            if not path.is_absolute() or not path.is_file():
                raise ValueError("Data mapping must name existing absolute paths, not server substitutions")
            resolved["splits"][split][name] = str(path)
    data = _validate_data(resolved, original_lp, ROOT)
    train = data["splits"]["train"]
    for field, origin_field in (("sha256", "train_sha256"), ("lpan_sha256", "train_lpan_sha256"),
                                ("sample_order_sha256", "train_sample_order_sha256")):
        if train[field] != origin[origin_field]:
            raise ValueError("F1 q calibration is not tied to these training bytes/order")
    q, _ = _q_values(origin, paths)
    bridge_path = paths.get("bridge_manifest")
    if bridge_path:
        bridge = _read(bridge_path)
        if (bridge.get("alias") != "F1" or bridge.get("origin_reference") != origin
                or bridge.get("status") != "PASS" or not bridge.get("complete")):
            raise ValueError("Supplied historical F1 bridge contradicts origin")
    return cfg, origin, data, q


def _common(paths, cfg, origin):
    return dict(schema=SCHEMA, teacher_run_id=TEACHER_RUN, teacher_update=50000,
                teacher_checkpoint_sha256=origin["teacher_checkpoint_sha256"],
                artifacts_sha256={k: sha256_file(v) for k, v in paths.items()},
                teacher_config_sha256=canonical_sha(cfg), source_identity=origin["source_identity"],
                tau_R=origin["tau_R"], q_ref=origin["q_ref"],
                calibration_indices_sha256=origin["calibration_indices_sha256"],
                dataset_manifest_sha256=origin["dataset_manifest_sha256"],
                lpan_manifest_sha256=origin["lpan_manifest_sha256"],
                augmentation=origin["augmentation"], LP_recipe=origin["LP_recipe"])


def prepare_binding(destination, root=ROOT, origin_paths=None, data_paths=None):
    """Bind original F1, or explicitly supplied portable F1 artifact paths."""
    root = Path(root)
    if origin_paths is None:
        # Read original paths exactly as stored. Docker must mount archive and
        # source paths explicitly; there is no opaque textual re-rooting here.
        manifest = root / "work_dir/_fh12/s1/references" / TEACHER_RUN / "reference_manifest.json"
        origin = _read(manifest); _origin_valid(origin)
        paths = {key: str(Path(origin[key])) for key in PAIRS}
        paths["origin_manifest"] = str(manifest.resolve())
        init = root / "work_dir" / TEACHER_RUN / "init_manifest.json"
        paths["init_manifest"] = str(init.resolve())
        bridge = root / "work_dir/_fh20r1/s1/imported_refs/F1/bridge_manifest.json"
        if bridge.is_file(): paths["bridge_manifest"] = str(bridge.resolve())
    else:
        paths = {k: str(Path(v).resolve()) for k, v in origin_paths.items()}
        if not set(PAIRS).union({"origin_manifest", "init_manifest"}) <= set(paths):
            raise ValueError("Portable F1 package is incomplete")
    if data_paths is None:
        raw = _read(paths["dataset_manifest_path"])
        data_paths = {s: {k: raw["splits"][s][k] for k in ("dataroot", "lpan_path")} for s in raw["splits"]}
    cfg, origin, data, q = _validate(paths, data_paths)
    common = _common(paths, cfg, origin)
    value = dict(schema=SCHEMA, state="BOUND_REQUIRES_RUNTIME_PARITY", ready_to_train=False,
                 common=common, common_sha256=canonical_sha(common), paths=paths,
                 data_paths=data_paths, dataset_manifest=data, origin_reference=origin,
                 tau_R=origin["tau_R"], q_ref=origin["q_ref"],
                 teacher_checkpoint_sha256=origin["teacher_checkpoint_sha256"],
                 teacher_config_sha256=canonical_sha(cfg), independent_teacher_count=1,
                 original_artifacts_modified=False, server_local_teacher_substitution=False)
    write_immutable_json(destination, value)
    return Path(destination)


def validate_binding(path):
    binding = _read(path)
    if (binding.get("schema") != SCHEMA or canonical_sha(binding["common"]) != binding.get("common_sha256")
            or binding.get("ready_to_train") is not False):
        raise ValueError("Invalid common F1 binding; readiness belongs to separate runtime preflight")
    cfg, origin, data, q = _validate(binding["paths"], binding["data_paths"])
    if (_common(binding["paths"], cfg, origin) != binding["common"] or binding["dataset_manifest"] != data
            or binding["origin_reference"] != origin or binding["tau_R"] != origin["tau_R"]
            or binding["q_ref"] != origin["q_ref"]
            or binding["teacher_checkpoint_sha256"] != origin["teacher_checkpoint_sha256"]):
        raise ValueError("Common F1 binding content changed")
    return cfg, binding, data, q


def load_binding(path, device="cpu"):
    from fh20r1.references import _load_frozen
    cfg, binding, data, q = validate_binding(path)
    teacher = _load_frozen(cfg, binding["paths"]["teacher_checkpoint"], device)
    return teacher, cfg, binding, data, q


def _copy_immutable(source, target):
    source, target = Path(source), Path(target)
    expected = sha256_file(source)
    if target.exists():
        _checked(target, expected); return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".rb-bind-", dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temp); _checked(temp, expected)
        os.chmod(temp, 0o444); os.link(temp, target)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def export_binding(binding_path, destination, weights_dir=None):
    """Copy verified reference/LP assets only; never copy or rewrite native H5s."""
    _, binding, data, _ = validate_binding(binding_path)
    destination = Path(destination)
    artifacts = {}
    for key, source in binding["paths"].items():
        relative = "artifacts/" + key + Path(source).suffix
        _copy_immutable(source, destination / relative)
        artifacts[key] = dict(path=relative, sha256=sha256_file(source))
    lps = {}
    for split, row in data["splits"].items():
        relative = f"lp/{split}.h5"
        _copy_immutable(row["lpan_path"], destination / relative)
        lps[split] = dict(path=relative, sha256=row["lpan_sha256"], native_sha256=row["sha256"])
    weight_files = {}
    if weights_dir is not None:
        from panda_rb.weights import load_weights
        from panda_rb.plan import registry
        weights_dir = Path(weights_dir)
        seeds = [s for pair in registry()['independent_student_seeds'].values() for s in pair]
        required = ['train_e_bar.npz', 'train_e_bar.json']
        for seed in seeds:
            load_weights(weights_dir, seed, binding)
            required += [f'seed_{seed}.npz', f'seed_{seed}.json']
        for name in required:
            relative = 'weights/' + name
            _copy_immutable(weights_dir / name, destination / relative)
            weight_files[name] = dict(path=relative, sha256=sha256_file(weights_dir / name))
    write_immutable_json(destination / "package.json", dict(schema="PANDA_RB_F1_PACKAGE_v1",
                         common_sha256=binding["common_sha256"], artifacts=artifacts, lp=lps,
                         weights=weight_files, all_six_weight_maps_included=bool(weight_files)))
    return destination


def import_binding(package_dir, destination, data_paths, root=ROOT, weights_dir=None):
    """data_paths maps each split to an explicit local native H5 filename."""
    package_dir = Path(package_dir).resolve()
    package = _read(package_dir / "package.json")
    if package.get("schema") != "PANDA_RB_F1_PACKAGE_v1":
        raise ValueError("Not a B01 F1 package")
    if weights_dir is not None and not package.get('all_six_weight_maps_included'):
        raise ValueError('A runtime package must include frozen e and all six weight mappings')
    def asset(row):
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe relative package path")
        path = package_dir / relative
        _checked(path, row["sha256"])
        return str(path)
    paths = {key: asset(row) for key, row in package["artifacts"].items()}
    if set(data_paths) != {"train", "val", "rr", "fr"}:
        raise ValueError("All four native data paths must be explicitly provided")
    resolved = {split: dict(dataroot=str(Path(data_paths[split]).resolve()), lpan_path=asset(row))
                for split, row in package["lp"].items()}
    for split, row in package["lp"].items(): _checked(resolved[split]["dataroot"], row["native_sha256"])
    # Validate the portable identity BEFORE publishing any binding.
    cfg, origin, _, _ = _validate(paths, resolved)
    if canonical_sha(_common(paths, cfg, origin)) != package["common_sha256"]:
        raise ValueError("Imported package is not the exact common F1 binding")
    if weights_dir is not None:
        from panda_rb.weights import load_weights
        from panda_rb.plan import registry
        seeds = [s for pair in registry(root)['independent_student_seeds'].values() for s in pair]
        required = {'train_e_bar.npz', 'train_e_bar.json'} | {
            f'seed_{s}.{ext}' for s in seeds for ext in ('npz','json')}
        if set(package.get('weights', {})) != required or not package.get('all_six_weight_maps_included'):
            raise ValueError('S3/S5 need the frozen common full-train e-cache and all six seed mappings')
        for name, row in package['weights'].items():
            if Path(name).name != name or row['path'] != 'weights/' + name:
                raise ValueError('Unsafe or inconsistent weight cache filename')
            asset(row)
        preview = dict(common_sha256=package['common_sha256'], paths=paths, q_ref=origin['q_ref'])
        for seed in seeds: load_weights(package_dir/'weights', seed, preview)
    result = prepare_binding(destination, root, paths, resolved)
    if weights_dir is not None:
        for name, row in package['weights'].items():
            _copy_immutable(asset(row), Path(weights_dir) / name)
        bound = _read(result)
        for seed in seeds: load_weights(weights_dir, seed, bound)
    return result
