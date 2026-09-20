"""Actual common-evaluator replay for the fixed TA and S92001 exact50K assets.

Evidence identifies original immutable reports, never rounded Sheet cells.
Absent assets are WAIT_PARITY; high/low model quality is not a parity gate.
"""
from pathlib import Path
import math

import yaml

from g20.common import ROOT, atomic_json, immutable_json, camp, check_deadline, object_sha, read_json, sha256, source_identity
from g20.references import R0_PINS, CORE_PATHS, verify_data_equivalence

HQNR_ATOL, ERGAS_ATOL = 1e-5, 1e-4
EVIDENCE_SCHEMA = 'G20_R0_EVALUATOR_EVIDENCE_v1'


def _path_file(entry, key):
    path = Path(entry[key + '_path'])
    if sha256(path) != entry[key + '_sha256']:
        raise ValueError('Parity evidence artifact checksum differs: ' + key)
    return path


def _prior(entry, label, local_data, root, deadline=None):
    check_deadline(deadline)
    paths = {key: _path_file(entry, key) for key in
             ('checkpoint', 'config', 'identity', 'prior_report', 'dataset_manifest')}
    cfg, identity = yaml.safe_load(paths['config'].read_text()), read_json(paths['identity'])
    source_data, report = read_json(paths['dataset_manifest']), read_json(paths['prior_report'])
    field, args = cfg.get('qg40', {}), cfg.get('model_args', {})
    role, seed, layout, width, depth = ('T', 91001, 'P0', 112, [1, 2, 3]) if label == 'TA' else ('S', 92001, 'PLH', 104, [1, 2, 2])
    if (cfg.get('seed') != seed or field.get('role') != role or field.get('sensor') != 'GF2'
            or field.get('num_bands') != 4 or field.get('input_layout') != layout
            or field.get('reference_id') != 'GF2_TA' or args.get('hidden_size') != width
            or list(args.get('depth', [])) != depth or field.get('server_id') != 's3'
            or field.get('profile') != 'BASE'):
        raise ValueError('Parity needs original GF2_TA / S92001 BASE architecture and identity')
    if label == 'S92001' and field.get('teacher_sha256') != R0_PINS['teacher_checkpoint_sha256']:
        raise ValueError('S92001 parity must use the pinned original GF2_TA reference')
    if (identity.get('update') != 50000 or identity.get('model_sha256') != entry['checkpoint_sha256']
            or identity.get('config_sha256') != object_sha(cfg)
            or identity.get('data_sha256') != object_sha(source_data)
            or paths['checkpoint'].name != 'model.safetensors'
            or paths['identity'] != paths['checkpoint'].parent / 'identity.json'):
        raise ValueError('Parity checkpoint is not the bound exact50K A/U state')
    if label == 'TA' and entry['checkpoint_sha256'] != R0_PINS['teacher_checkpoint_sha256']:
        raise ValueError('Parity TA checkpoint differs from R0 plan pin')
    if (report.get('step', report.get('update')) != 50000
            or report.get('checkpoint_identity') != identity
            or report.get('checkpoint_sha256', identity['model_sha256']) != entry['checkpoint_sha256']):
        raise ValueError('Prior report is not bound to this exact50K checkpoint')
    prior = _metric_pair(report)
    src = identity['source_identity']
    if src.get('numeric_method_revision') != 'QG40_SYNC_FREQ_C4_v1':
        raise ValueError('Prior parity asset numerical revision differs')
    for name in CORE_PATHS:
        if src.get('files', {}).get(name) != sha256(Path(root) / name):
            raise ValueError('Prior model numerical source changed: ' + name)
    equality = verify_data_equivalence(source_data, local_data,
                r0=True, origin_tensor_receipt=entry.get('origin_tensor_identity'),
                origin_authenticated=True, deadline=deadline)
    return cfg, paths, prior, equality, identity


def _metric_pair(report):
    rr, fr = report.get('rr', {}), report.get('fr', {})
    if (rr.get('n_scenes') != 20 or rr.get('crop') != '20:-21' or rr.get('q_block') != 32
            or not rr.get('official_complete') or 'q4' not in rr
            or fr.get('n_scenes') != 20 or fr.get('reference') != 'native_PAN'
            or fr.get('support') != 'full512' or fr.get('masking') is not False
            or fr.get('aggregation') != 'mean_per_scene_HQNR'
            or fr.get('hqnr_variant') != 'raw-original' or not fr.get('official_complete')
            or any(x.get('sensor') != 'GF2' or x.get('num_bands') != 4 or x.get('max_dn') != 1023 for x in (rr, fr))):
        raise ValueError('Parity requires unchanged official GF2 RR20/Q4 and raw-original FR20')
    result = dict(hqnr=float(fr['hqnr']), ergas=float(rr['ergas']))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite parity metrics')
    return result


def compare_metrics(prior, current):
    differences = {key: abs(current[key] - prior[key]) for key in ('hqnr', 'ergas')}
    return dict(prior=prior, current=current, abs_difference=differences,
                tolerance=dict(hqnr=HQNR_ATOL, ergas=ERGAS_ATOL),
                passed=differences['hqnr'] <= HQNR_ATOL and differences['ergas'] <= ERGAS_ATOL)


def discover_parity_evidence(root=ROOT, server='s1', deadline=None):
    """Read-only exact-asset discovery, returning evidence but never a PASS.

    Portable inputs may retain the original run layout below
    _g20/<server>/parity_inputs/{TA,S92001}. Put the original unmodified dataset
    manifest at that directory's dataset_manifest.json if its old path differs.
    """
    root = Path(root)
    local_path = camp(root, server) / 'dataset_manifest.json'
    if not local_path.is_file():
        return dict(status='WAIT_PARITY', evidence=None, reason='Verified local GF2 manifest absent')
    local = read_json(local_path)
    runs = set((root / 'work_dir').glob('QG40_GF2*'))
    runs.update((root / 'work_dir').glob('QGBASE_GF2*'))
    runs.update(camp(root, server).glob('parity_inputs/*'))
    found, errors = {'TA': {}, 'S92001': {}}, []
    for run in sorted(runs):
        check_deadline(deadline)
        config = run / 'meta/config.resolved.yaml'
        if not config.is_file():
            continue
        try:
            cfg = yaml.safe_load(config.read_text())
            field = cfg.get('qg40', {})
            if field.get('sensor') != 'GF2' or field.get('server_id') != 's3':
                continue
            label = 'TA' if field.get('role') == 'T' and cfg.get('seed') == 91001 else (
                    'S92001' if field.get('role') == 'S' and cfg.get('seed') == 92001 else None)
            if label is None:
                continue
            data_candidates = [run / 'dataset_manifest.json', run / 'meta/dataset_manifest.json']
            if field.get('dataset_manifest'):
                path = Path(field['dataset_manifest'])
                data_candidates.append(path if path.is_absolute() else root / path)
            data_path = next((path for path in data_candidates if path.is_file()), None)
            if data_path is None:
                raise FileNotFoundError('Original checkpoint-bound dataset manifest absent')
            paths = dict(checkpoint=run / 'candidates/50000/model.safetensors', config=config,
                identity=run / 'candidates/50000/identity.json',
                prior_report=run / 'official/exact50k.json', dataset_manifest=data_path)
            entry = {}
            for key, path in paths.items():
                entry[key + '_path'], entry[key + '_sha256'] = str(path), sha256(path)
            _prior(entry, label, local, root, deadline)
            # Duplicate byte-identical copies are not different references.
            found[label].setdefault(entry['checkpoint_sha256'], entry)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            errors.append(dict(path=str(run), reason=f'{type(error).__name__}: {error}'))
    if any(len(found[label]) > 1 for label in found):
        return dict(status='AMBIGUOUS_PARITY', evidence=None, errors=errors,
                    candidates={label: list(values) for label, values in found.items()})
    if not all(found.values()):
        return dict(status='WAIT_PARITY', evidence=None, errors=errors,
                    missing=[label for label, values in found.items() if not values])
    evidence = dict(schema=EVIDENCE_SCHEMA, assets={label: next(iter(values.values())) for label, values in found.items()})
    return dict(status='EVIDENCE_READY_NOT_REPLAYED', evidence=evidence, errors=errors)


def verify_r0_parity(root=ROOT, server='s1', device='cpu', deadline=None, evidence_path=None):
    """Replay both exact50K checkpoints. This intentionally performs inference.

    Default input is work_dir/_g20/<server>/r0_parity_evidence.json. Each TA and
    S92001 entry needs checkpoint/config/identity/prior_report/dataset_manifest
    ``*_path`` and ``*_sha256``. Reports must contain the original full checkpoint
    identity and full-precision official metrics. Missing files never yield PASS.
    """
    check_deadline(deadline)
    allow_discovery = evidence_path is None
    evidence_path = Path(evidence_path) if evidence_path else camp(root, server) / 'r0_parity_evidence.json'
    target = camp(root, server) / 'diagnostics/r0_evaluator_parity.json'
    seal_path = target.with_suffix('.seal.json')
    if seal_path.exists() and (not target.is_file() or read_json(seal_path).get('receipt_sha256') != sha256(target)):
        raise ValueError('Immutable evaluator parity PASS receipt changed')
    discovery = None
    if not evidence_path.is_file() and allow_discovery:
        discovery = discover_parity_evidence(root, server, deadline)
        if discovery.get('status') == 'EVIDENCE_READY_NOT_REPLAYED':
            immutable_json(evidence_path, discovery['evidence'])
    if not evidence_path.is_file():
        result = dict(schema='G20_R0_EVALUATOR_PARITY_v1', status='WAIT_PARITY', passed=False,
                      reason='Immutable TA and S92001 exact50K report/checkpoint evidence is absent',
                      evidence_path=str(evidence_path), discovery=discovery)
        if not seal_path.exists():
            atomic_json(target, result)
        return result
    evidence = read_json(evidence_path)
    if evidence.get('schema') != EVIDENCE_SCHEMA or set(evidence.get('assets', {})) != {'TA', 'S92001'}:
        raise ValueError('Parity evidence requires the two explicit original TA/S92001 assets')
    local = read_json(camp(root, server) / 'dataset_manifest.json')
    from g20.data import build_dataset
    from g20.evaluation import evaluate_model
    from qg40.common import load_checkpoint_model
    rows = {}
    try:
        # Validate actual origin artifacts and all local split/LP bytes before
        # either replay or reuse; a cached Boolean alone is not evidence.
        checked = {label: _prior(evidence['assets'][label], label, local, root, deadline=deadline)
                   for label in ('TA', 'S92001')}
        release = source_identity(root)
        if target.is_file() and read_json(target).get('status') == 'PASS':
            cached = read_json(target)
            if (not seal_path.is_file() or cached.get('source_identity') != release
                    or cached.get('evidence_sha256') != sha256(evidence_path)
                    or cached.get('local_data_sha256') != object_sha(local)
                    or set(cached.get('assets', {})) != {'TA', 'S92001'}):
                raise ValueError('Cached evaluator parity identity changed; original receipt retained')
            for label, (_, _, prior, equality, _) in checked.items():
                old = cached['assets'][label]
                comparison = compare_metrics(prior, _metric_pair(old['measured']))
                if (not comparison['passed'] or any(old.get(key) != value for key, value in comparison.items())
                        or old.get('checkpoint_sha256') != evidence['assets'][label]['checkpoint_sha256']
                        or old.get('prior_report_sha256') != evidence['assets'][label]['prior_report_sha256']
                        or old.get('data_equality') != equality):
                    raise ValueError('Cached evaluator parity measurement binding differs')
            check_deadline(deadline)
            return cached
        datasets = {name: build_dataset(local, name) for name in ('rr', 'fr')}
        for label in ('TA', 'S92001'):
            check_deadline(deadline)
            cfg, paths, prior, equality, identity = checked[label]
            model, loaded_identity = load_checkpoint_model(cfg, paths['checkpoint'].parent, device)
            if loaded_identity != identity:
                raise ValueError('Parity identity changed while loading checkpoint')
            current = evaluate_model(model, datasets, device, deadline=deadline,
                                     include_q=True, with_val=False, include_jqm=False)
            check_deadline(deadline)
            for key in ('checkpoint', 'config', 'identity', 'prior_report', 'dataset_manifest'):
                _path_file(evidence['assets'][label], key)
            rows[label] = dict(compare_metrics(prior, _metric_pair(current)),
                checkpoint_sha256=evidence['assets'][label]['checkpoint_sha256'],
                prior_report_sha256=evidence['assets'][label]['prior_report_sha256'], data_equality=equality,
                measured=current)
            del model
        for label in ('TA', 'S92001'):
            if _prior(evidence['assets'][label], label, local, root, deadline=deadline)[3:] != checked[label][3:]:
                raise ValueError('Parity input evidence changed during inference')
        if source_identity(root) != release:
            raise ValueError('Evaluator numerical release changed during parity inference')
    except FileNotFoundError as error:
        result = dict(schema='G20_R0_EVALUATOR_PARITY_v1', status='WAIT_PARITY', passed=False,
                      reason=str(error), completed=rows, evidence_sha256=sha256(evidence_path))
        if not seal_path.exists():
            atomic_json(target, result)
        return result
    passed = all(row['passed'] for row in rows.values())
    result = dict(schema='G20_R0_EVALUATOR_PARITY_v1', status='PASS' if passed else 'BLOCKED_PARITY',
                  passed=passed, assets=rows, evidence_sha256=sha256(evidence_path),
                  source_identity=source_identity(root), local_data_sha256=object_sha(local),
                  note='Numerical same-checkpoint replay tolerance, not retraining significance or quality threshold')
    atomic_json(target, result)
    if passed:
        immutable_json(seal_path, dict(schema='G20_EVALUATOR_PARITY_SEAL_v1',
            receipt_sha256=sha256(target), evidence_sha256=sha256(evidence_path),
            source_sha256=object_sha(result['source_identity'])))
    return result
