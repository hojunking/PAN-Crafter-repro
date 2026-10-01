"""Twelve new mappings over exactly the frozen B01 raw-q and e-bar bytes."""
from pathlib import Path
import shutil
import numpy as np
from fh12.data import write_immutable_json
from panda_rb.weights import build_weights, array_sha, load_weights, _save_npz
from panda_rb_m12.binding import validate_binding
from panda_rb_m12.common import ROOT, read_json, sha256, source_identity
from panda_rb_m12.plan import SEEDS, weights_dir


def prepare_weights(binding_path, root=ROOT, source=None):
    _, binding, _, q = validate_binding(binding_path)
    origin = Path(source or Path(root) / 'work_dir/_panda_rb/20260928/B01/common/weights')
    meta = read_json(origin / 'train_e_bar.json')
    if (sha256(origin / 'train_e_bar.npz') != meta['cache_sha256']
            or meta['identity']['binding_common_sha256'] != binding['common_sha256']
            or meta['identity']['q_sha256'] != array_sha(q)):
        raise ValueError('B01 original train-e cache/F1/q identity mismatch')
    with np.load(origin / 'train_e_bar.npz', allow_pickle=False) as f:
        e = f['e_bar'].copy()
    with np.load(binding['paths']['q_cache_path'], allow_pickle=False) as f:
        calibration = f['calibration_indices'].copy()
    target = weights_dir(root); target.mkdir(parents=True, exist_ok=True)
    # Copy exact e bytes; np.savez regeneration changes file identity.
    for name in ('train_e_bar.npz', 'train_e_bar.json'):
        dest = target / name
        if dest.exists():
            if sha256(dest) != sha256(origin / name):
                raise ValueError('M12 cache differs from original B01 bytes')
        else:
            from panda_rb.bindings import _copy_immutable
            _copy_immutable(origin / name, dest)
    result = {}
    for seed in [seed for group in SEEDS.values() for seed in group]:
        arrays, report = build_weights(q, e, binding['q_ref'], seed)
        arrays['calibration_indices'] = calibration
        report['arrays_sha256']['calibration_indices'] = array_sha(calibration)
        path = target / f'seed_{seed}.npz'
        _save_npz(path, arrays)
        report.update(binding_common_sha256=binding['common_sha256'], cache_identity=meta['identity'],
                      cache_sha256=sha256(path), train_e_cache_sha256=meta['cache_sha256'])
        write_immutable_json(target / f'seed_{seed}.json', report)
        load_weights(target, seed, binding)
        result[str(seed)] = report['cache_sha256']
    return dict(weight_files=result, e_cache_sha256=meta['cache_sha256'],
                e_cache_source_identity=meta['identity']['source_identity'],
                mapping_implementation_sha256=sha256(Path(root) / 'panda_rb/weights.py'),
                original_e_cache_recomputed=False)
