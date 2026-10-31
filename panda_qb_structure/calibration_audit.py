"""Compare actual full-training numerical tables without rewriting source bindings."""
from pathlib import Path
import json
import math
import zipfile
import numpy as np
from directshift48.reporting import atomic_json
from ta2.evaluation import file_sha
from donor30.plan import canonical_sha256
from . import registry as G
from .donor import local

NUMERIC_METADATA = ('tau_e', 'n_e', 'n_q', 'sample_count', 'bin_counts', 'g_grouping',
                    'q_ecdf', 'valid_e_counts_by_sample', 'mask_policy', 'spatial_population')
PHYSICAL_BINDING = ('sensor', 'source_split', 'sample_ids_sha256', 'source_manifest_sha256',
                    'source_h5_sha256', 'catalog_sha256', 'aligner_state_sha256',
                    'teacher_state_sha256', 'donor_checkpoint_sha256', 'scharr', 'warp')
REFERENCE_ERRORS = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile)


def _read_archive(path):
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError('Calibration must be a named-array NPZ archive')
    with loaded as archive:
        metadata = json.loads(archive['metadata_json'].tobytes())
        if (not isinstance(metadata, dict) or not isinstance(metadata.get('binding'), dict)
                or type(metadata.get('tau_e')) not in (float, int)
                or not math.isfinite(metadata['tau_e'])):
            raise ValueError('Calibration metadata is malformed')
        arrays = {key: archive[key] for key in archive.files if key != 'metadata_json'}
        if ('g_cutpoints_f64' not in arrays or not any(k.startswith('q_') for k in arrays)
                or any(a.dtype.kind != 'f' or not np.isfinite(a).all() for a in arrays.values())):
            raise ValueError('Calibration numerical arrays are malformed')
        return metadata, arrays


def _compare_contents(current, reference):
    lm, left = current; rm, right = reference
    metadata_differences = [k for k in NUMERIC_METADATA if lm.get(k) != rm.get(k)]
    binding_differences = [k for k in PHYSICAL_BINDING if lm['binding'].get(k) != rm['binding'].get(k)]
    rows = []
    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            rows.append({'name': name, 'exact': False, 'reason': 'ARRAY_MISSING'})
            continue
        a, b = left[name], right[name]
        equal = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        row = {'name': name, 'exact': bool(equal), 'current_shape': list(a.shape),
               'reference_shape': list(b.shape), 'current_dtype': str(a.dtype), 'reference_dtype': str(b.dtype)}
        if a.shape == b.shape and a.size:
            row['max_absolute_difference'] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        rows.append(row)
    exact = not metadata_differences and not binding_differences and all(row['exact'] for row in rows)
    return {'numeric_tables_exact': exact, 'metadata_differences': metadata_differences,
            'physical_binding_differences': binding_differences, 'arrays': rows,
            'current_tau_e': lm['tau_e'], 'reference_tau_e': rm['tau_e'],
            'metadata_schema_relabelled': False}


def compare_archives(current, reference):
    """Exact comparison; callers decide whether a reference failure is optional."""
    return _compare_contents(_read_archive(current), _read_archive(reference))


def historical_configs(root, server):
    """Locate only the two same-server baseline runs, independent of queue status."""
    root = Path(root); result = []
    for replica in ('R1', 'R2'):
        source = G.controls(server, replica)
        folder = root / 'work_dir' / source['package'] / source['campaign_id'] / server / 'QB/runs'
        try:
            matches = sorted(folder.glob(f"*_{source['case']}_NOSOFT_{replica}_S{source['seed']}/attempt_01/config.json"))
            if len(matches) != 1:
                result.append({'replica': replica, 'status': 'REFERENCE_CONFIG_UNAVAILABLE'})
                continue
            path = matches[0]; cfg = json.loads(path.read_text())
            if not isinstance(cfg, dict): raise ValueError('Reference config must be an object')
            if any(cfg.get(k) != source[k] for k in ('campaign_id', 'server', 'case', 'replica', 'seed')):
                result.append({'replica': replica, 'status': 'REFERENCE_CONFIG_IDENTITY_MISMATCH'})
                continue
            artifacts = cfg.get('calibration_artifacts', {})
            if not isinstance(artifacts, dict): raise ValueError('Reference artifact map is malformed')
            entry = artifacts.get('s100')
            if entry is None:
                result.append({'replica': replica, 'status': 'REFERENCE_CALIBRATION_UNAVAILABLE'})
                continue
            if (not isinstance(entry, dict) or not isinstance(entry.get('path'), str)
                    or not isinstance(entry.get('file_sha256'), str)):
                raise ValueError('Reference artifact identity is malformed')
            result.append({'replica': replica, 'status': 'FOUND', 'config_path': str(path),
                           'config_sha256': file_sha(path), 'path': str(local(entry['path'], root)),
                           'expected_sha256': entry['file_sha256']})
        except REFERENCE_ERRORS as exc:
            result.append({'replica': replica, 'status': 'REFERENCE_CONFIG_READ_FAILED', 'reason': repr(exc)})
    return result


def audit(root, campaign_root, server, artifacts):
    current = artifacts['s100']; current_path = local(current['path'], root)
    references = historical_configs(root, server)
    for row in references:
        if row['status'] != 'FOUND': continue
        try:
            path = Path(row['path'])
            if not path.is_file(): row['status'] = 'REFERENCE_FILE_UNAVAILABLE'
            elif file_sha(path) != row['expected_sha256']: row['status'] = 'REFERENCE_SHA_MISMATCH'
        except REFERENCE_ERRORS as exc:
            row.update(status='REFERENCE_FILE_READ_FAILED', reason=repr(exc))
    if file_sha(current_path) != current['file_sha256']:
        raise ValueError('Current calibration changed before numerical audit')
    identity = {'current': current, 'references': references}
    path = Path(campaign_root) / 'calibration' / ('numeric_parity_' + canonical_sha256(identity)[:24] + '.json')
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('identity') == identity:
            return {'path': str(path), 'status': saved['status'],
                    'historical_control_comparison_valid': saved['historical_control_comparison_valid']}
    # Current data failures remain fatal. Only historical reference reads below
    # are optional; their exceptions cannot hide a corrupt training archive.
    current_contents = _read_archive(current_path)
    comparisons = []; by_file = {}
    for row in references:
        item = dict(row)
        if row['status'] == 'FOUND':
            key = (row['path'], row['expected_sha256'])
            if key not in by_file:
                try:
                    by_file[key] = _compare_contents(current_contents, _read_archive(row['path']))
                except REFERENCE_ERRORS as exc:
                    by_file[key] = {'status': 'REFERENCE_ARCHIVE_READ_FAILED',
                                    'numeric_tables_exact': False, 'reason': repr(exc)}
            item.update(by_file[key])
        comparisons.append(item)
    exact = all(r.get('numeric_tables_exact') is True for r in comparisons)
    status = 'EXACT_HISTORICAL_NUMERIC_PARITY' if exact else 'HISTORICAL_COMPARISON_NOT_VERIFIED'
    saved = {'schema': 'QB_STRUCTURE_CALIBRATION_NUMERIC_AUDIT_V1', 'server': server,
             'identity': identity, 'comparisons': comparisons, 'status': status,
             'historical_control_comparison_valid': exact, 'source_stats_relabelled': False,
             'training_tables_replaced': False, 'predecessor_completion_gate': False}
    atomic_json(path, saved)
    return {'path': str(path), 'status': status, 'historical_control_comparison_valid': exact}
