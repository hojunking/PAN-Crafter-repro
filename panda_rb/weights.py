"""Frozen train-only geometry controls, with a dedicated permutation RNG."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from fh12.data import sha256_file, write_immutable_json

REVISION = "PANDA_RB_GEOMETRY_WEIGHTS_v1"
E_DEFINITION = "mean_CHW_abs_unclipped_F1_y_minus_normalized_GT_after_fixed_HV_rot4"


def array_sha(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _population(q, e_bar, q_ref):
    q, e = np.asarray(q, dtype=np.float32), np.asarray(e_bar, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 4 or q.shape != e.shape or not len(q):
        raise ValueError("Expected complete training sample x four-view q/e populations")
    if not np.isfinite(q).all() or not np.isfinite(e).all() or (q < 0).any() or (e < 0).any():
        raise ValueError("q/e must be finite nonnegative training values")
    if not np.isfinite(q_ref) or q_ref <= 0:
        raise ValueError("Use positive actual F1 q_ref, never a guessed snapshot")
    # Match the original torch scalar / Tensor expression, which uses a
    # reciprocal/multiply path and can differ from NumPy divide by one ULP.
    import torch
    s = (float(q_ref) / (float(q_ref) + torch.from_numpy(q))).numpy()
    if not np.isfinite(s).all() or not ((s > 0) & (s <= 1)).all():
        raise ValueError("Invalid raw-q conversion")
    return q, e, s


def build_weights(raw_q, e_bar, q_ref, repeat_seed):
    """Return all controls and provenance; never touch global NumPy/torch RNG."""
    q, e, full = _population(raw_q, e_bar, q_ref)
    n, views = q.shape
    source = np.arange(n, dtype=np.int64)
    permutation_seed = int.from_bytes(hashlib.sha256(
        f"{REVISION}:QSHUF:{int(repeat_seed)}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(permutation_seed)
    shuffle = np.empty_like(full); surrogate = np.empty_like(full)
    permutation = np.empty(q.shape, dtype=np.int64)
    surrogate_source = np.empty(q.shape, dtype=np.int64)
    strata = np.empty(q.shape, dtype=np.int8)
    singleton = []
    for view in range(views):
        order = np.lexsort((source, e[:, view]))
        for stratum, members in enumerate(np.array_split(order, 5)):
            strata[members, view] = stratum
            if len(members) < 2:
                permutation[members, view] = members
                singleton.extend([[int(i), view, stratum] for i in members])
            else:
                cycle = rng.permutation(members)
                permutation[cycle, view] = np.roll(cycle, 1)
        shuffle[:, view] = full[permutation[:, view], view]
        descending_s = np.lexsort((source, -full[:, view]))
        surrogate_source[order, view] = descending_s
        surrogate[:, view] = full[surrogate_source[:, view], view]
        if not np.array_equal(np.sort(full[:, view]), np.sort(surrogate[:, view])):
            raise AssertionError("QESUR changed the view multiset")
        for group in range(5):
            ids = np.flatnonzero(strata[:, view] == group)
            if (not np.array_equal(np.sort(full[ids, view]), np.sort(shuffle[ids, view]))
                    or (len(ids) > 1 and np.any(permutation[ids, view] == ids))):
                raise AssertionError("QSHUF failed conditional multiset/derangement")
    mean = float(full.mean(dtype=np.float64))
    arrays = dict(q=q.copy(), e_bar=e.copy(), QFULL=full, QMEAN=np.full_like(full, mean),
                  QSHUF=shuffle, QESUR=surrogate, shuffle_source=permutation,
                  surrogate_source=surrogate_source, e_stratum=strata)
    def stats(a):
        return dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean(dtype=np.float64)),
                    sd=float(a.std(dtype=np.float64)), quantiles=np.quantile(a, [0, .2, .4, .5, .6, .8, 1]).tolist())
    report = dict(schema=REVISION, repeat_seed=int(repeat_seed), permutation_seed=permutation_seed,
                  shape=list(q.shape), q_ref=float(q_ref), training_population_mean_s=mean,
                  mean_reduction="float64 all train sample x four views then FP32 at consumption",
                  e_definition=E_DEFINITION, tie_break="source_index_ascending", q_is_raw=True,
                  permutation_method="dedicated PCG64 random single cycle in each view/e quintile",
                  singleton_groups=singleton, fixed_for_all_updates=True, batch_renormalization=False,
                  arrays_sha256={k: array_sha(v) for k, v in arrays.items()},
                  population_stats={k: stats(arrays[k]) for k in ("q", "e_bar", "QFULL", "QMEAN", "QSHUF", "QESUR")},
                  conditional_stats=[dict(view=v, stratum=g, count=int((strata[:, v] == g).sum()),
                                          full=stats(full[strata[:, v] == g, v]),
                                          shuffled=stats(shuffle[strata[:, v] == g, v]))
                                     for v in range(views) for g in range(5) if np.any(strata[:, v] == g)],
                  marginal_multiset_preserved=["QFULL", "QSHUF", "QESUR"])
    return arrays, report


def _save_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError("Refusing to overwrite immutable q/e mapping")
        return
    fd, temp = tempfile.mkstemp(prefix=".rbweights-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
        os.chmod(temp, 0o444); os.link(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def compute_e_bar(teacher, dataset, device="cuda", batch_size=48):
    """The exact original teacher reconstruction error, no output clipping."""
    import torch
    from fh12.model import state_hash
    before = state_hash(teacher.state_dict())
    training = teacher.training
    teacher.eval()
    output = np.empty((len(dataset), 4), np.float32)
    try:
        with torch.inference_mode():
            for view in range(4):
                for start in range(0, len(dataset), batch_size):
                    rows = [dataset[(i, view)] for i in range(start, min(start+batch_size, len(dataset)))]
                    fields = [torch.stack([r[k] for r in rows]).to(device) for k in range(5)]
                    gt, _, ms, lp, pan = fields
                    y = teacher(pan, ms, lp)["y"].float()
                    values = (y - gt.float()).abs().mean(dim=1, keepdim=True).flatten(1).mean(1)
                    if not bool(torch.isfinite(values).all()):
                        raise ValueError("Nonfinite fixed F1 train error cache")
                    output[start:start+len(rows), view] = values.cpu().numpy()
    finally:
        teacher.train(training)
    if state_hash(teacher.state_dict()) != before:
        raise ValueError("F1 state changed while computing train e")
    return output


def prepare_weights(binding_path, output_dir, device="cuda", seeds=None, root=None):
    """Explicit preparation only; never invoked on import/config generation."""
    from panda_rb.bindings import load_binding
    from panda_rb.plan import ROOT, registry
    from panda_rb.common import source_identity
    from fh12.data import build_dataset
    teacher, cfg, binding, data, q = load_binding(binding_path, device=device)
    root = Path(root or ROOT)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(schema=REVISION, binding_common_sha256=binding["common_sha256"],
                    q_sha256=array_sha(q), e_definition=E_DEFINITION,
                    source_identity=source_identity(root), train_views=[0, 1, 2, 3])
    cache, metadata = output_dir / "train_e_bar.npz", output_dir / "train_e_bar.json"
    if metadata.exists():
        info = json.loads(metadata.read_text())
        if info["identity"] != identity or sha256_file(cache) != info["cache_sha256"]:
            raise ValueError("Frozen train e-cache identity mismatch")
        with np.load(cache, allow_pickle=False) as f: e = f["e_bar"].copy()
        _population(q, e, binding["q_ref"])
    else:
        dataset = build_dataset(data, "train")
        e = compute_e_bar(teacher, dataset, device)
        _save_npz(cache, {"e_bar": e})
        write_immutable_json(metadata, dict(identity=identity, cache_sha256=sha256_file(cache)))
    with np.load(binding["paths"]["q_cache_path"], allow_pickle=False) as f:
        calibration_ids = f["calibration_indices"].copy()
    if seeds is None:
        seeds = [s for values in registry(root)["independent_student_seeds"].values() for s in values]
    result = {}
    for seed in seeds:
        arrays, report = build_weights(q, e, binding["q_ref"], seed)
        arrays["calibration_indices"] = calibration_ids
        report["arrays_sha256"]["calibration_indices"] = array_sha(calibration_ids)
        file = output_dir / f"seed_{int(seed)}.npz"
        _save_npz(file, arrays)
        report.update(binding_common_sha256=binding["common_sha256"], cache_identity=identity,
                      cache_sha256=sha256_file(file), train_e_cache_sha256=sha256_file(cache))
        manifest = output_dir / f"seed_{int(seed)}.json"
        write_immutable_json(manifest, report)
        result[str(seed)] = str(manifest)
    return result


def load_weights(output_dir, seed, binding=None):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / f"seed_{int(seed)}.json").read_text())
    path = output_dir / f"seed_{int(seed)}.npz"
    if (manifest["schema"] != REVISION or manifest["repeat_seed"] != int(seed)
            or sha256_file(path) != manifest["cache_sha256"]):
        raise ValueError("Geometry mapping file/seed identity mismatch")
    if binding is not None and binding["common_sha256"] != manifest["binding_common_sha256"]:
        raise ValueError("Geometry weights came from a different F1/data/calibration")
    with np.load(path, allow_pickle=False) as f: arrays = {k: f[k].copy() for k in f.files}
    if {k: array_sha(v) for k, v in arrays.items()} != manifest["arrays_sha256"]:
        raise ValueError("Geometry mapping arrays changed")
    if binding is not None and 'paths' in binding:
        with np.load(binding['paths']['q_cache_path'], allow_pickle=False) as original:
            if (not np.array_equal(arrays['q'], original['q'])
                    or not np.array_equal(arrays.get('calibration_indices'), original['calibration_indices'])
                    or manifest['q_ref'] != binding['q_ref']):
                raise ValueError('Geometry cache q/subset is not the original bound F1 population')
        if 'cache_identity' not in manifest:
            raise ValueError('Production weights require the frozen train-e cache identity')
    if 'cache_identity' in manifest:
        ecache = output_dir / 'train_e_bar.npz'
        emeta = json.loads((output_dir / 'train_e_bar.json').read_text())
        identity = manifest['cache_identity']
        if (emeta['identity'] != identity or sha256_file(ecache) != manifest['train_e_cache_sha256']
                or emeta['cache_sha256'] != manifest['train_e_cache_sha256']
                or identity['binding_common_sha256'] != manifest['binding_common_sha256']
                or identity['q_sha256'] != array_sha(arrays['q'])
                or identity['e_definition'] != E_DEFINITION):
            raise ValueError('Weight mapping does not use the frozen common train-e cache')
        with np.load(ecache, allow_pickle=False) as f:
            if not np.array_equal(arrays['e_bar'], f['e_bar']):
                raise ValueError('Per-seed e array differs from the common population cache')
    expected, _ = build_weights(arrays["q"], arrays["e_bar"], manifest["q_ref"], seed)
    if any(not np.array_equal(arrays[k], v) for k, v in expected.items()):
        raise ValueError("Saved weights do not implement the registered controls")
    return arrays, manifest
