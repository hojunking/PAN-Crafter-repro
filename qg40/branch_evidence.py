"""Measured, isolated branch evidence; diagnostic cross-A/U is never a normal result."""
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from qg40.common import (ROOT, atomic_json, camp, check_deadline, load_checkpoint_model,
                        object_sha, read, read_json, sha256, source_identity, utcnow)
from qg40.plan import case_for, cases_for, sensor_spec


def diagnostic(root, run, step):
    wd = Path(root) / 'work_dir' / run
    path = wd / 'diagnostics' / f'update_{step}.json'
    value = read_json(path)
    checkpoint = wd / 'restart_fullstates' / str(step)
    identity = read_json(checkpoint / 'identity.json')
    if (value.get('status') != 'MEASURED' or value.get('synthetic_test') is not False
            or value.get('update') != step or identity.get('update') != step
            or value.get('distribution_samples') != 128 or value.get('gradient_samples') != 24
            or value.get('model_state_hash') != identity.get('state_hash')
            or value.get('source_identity') != source_identity(root)
            or value.get('arrays_sha256') != sha256(value['arrays_path'])
            or any(value.get(k) != identity.get(k) for k in
                   ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256'))):
        raise ValueError('Diagnostic identity/count/source mismatch')
    if (identity.get('model_sha256') != sha256(checkpoint / 'model.safetensors')
            or identity.get('training_state_sha256') != sha256(checkpoint / 'training_state.pt')):
        raise ValueError('Diagnostic source checkpoint bytes changed')
    with np.load(value['arrays_path'], allow_pickle=False) as arrays:
        ids = arrays['diagnostic_indices']
        if (ids.shape != (128,) or not np.issubdtype(ids.dtype, np.integer)
                or len(np.unique(ids)) != 128 or int(ids.min()) < 0
                or arrays['gradient_indices'].shape != (24,)
                or not np.array_equal(arrays['gradient_indices'], ids[:24])
                or object_sha(ids.tolist()) != value['diagnostic_indices_sha256']
                or arrays['native_delta'].shape != (128, 2)
                or len(value.get('gradient_rows', [])) != 24
                or [row.get('index') for row in value['gradient_rows']] != ids[:24].tolist()
                or any(not np.isfinite(arrays[key]).all() for key in arrays.files)):
            raise ValueError('Diagnostic actual train128/gradient24 arrays invalid')
    return value, dict(path=str(path), sha256=sha256(path))


def _official(root, run):
    from qg40.postrun import validate_grid, select_records
    wd = Path(root) / 'work_dir' / run
    if not read(wd / 'official/postrun_status.json').get('official_complete'):
        raise ValueError('Branch evidence requires completed official local baseline')
    cfg = yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text())
    grid = read_json(wd / 'official/raw_grid.json')
    validate_grid(run, cfg, grid, root)
    selected = select_records(grid['records'], case_for(run).sensor)
    return cfg, grid, selected


def cross_aligner(root, run, device, deadline):
    """Reproduce both diagonals, then A24240+U50000, using unchanged RR/FR."""
    from qg40.data import build_dataset
    from qg40.evaluation import FRMetrics, evaluate_model, RR_KEYS, FR_KEYS
    cfg, grid, _ = _official(root, run)
    wd = Path(root) / 'work_dir' / run
    data = read_json(cfg['qg40']['dataset_manifest'])
    datasets = {name: build_dataset(data, name, root=root) for name in ('rr', 'fr')}
    engine = FRMetrics(datasets['fr'])
    reference = {r['update']: r for r in grid['records']}
    models, diagonal, identities = {}, {}, {}
    for step in (24240, 50000):
        check_deadline(deadline)
        models[step], identities[step] = load_checkpoint_model(cfg, wd / 'candidates' / str(step), device,
                                                              expected_source=source_identity(root))
        diagonal[step] = evaluate_model(models[step], datasets, device, engine, deadline,
                                        with_val=False, include_jqm=False)
        for group, key in [('rr', key) for key in RR_KEYS] + [('fr', key) for key in FR_KEYS]:
            if abs(diagonal[step][group][key] - reference[step][group][key]) > 1e-6:
                raise ValueError(f'Diagnostic diagonal does not reproduce official {step}/{key}')
    old_a = {key: tensor.detach().clone() for key, tensor in models[24240].aligner.state_dict().items()}
    models[50000].aligner.load_state_dict(old_a, strict=True)
    cross = evaluate_model(models[50000], datasets, device, engine, deadline, with_val=False, include_jqm=False)
    late = diagonal[50000]
    early_diagnostic, _ = diagnostic(root, run, 24240)
    late_diagnostic, _ = diagnostic(root, run, 50000)
    with np.load(early_diagnostic['arrays_path'], allow_pickle=False) as early_arrays, np.load(
            late_diagnostic['arrays_path'], allow_pickle=False) as late_arrays:
        native_changed = not np.allclose(early_arrays['native_delta'], late_arrays['native_delta'], atol=1e-8, rtol=0)
    result = dict(schema='QG40_AU_CROSS_v1', run_id=run, diagnostic_only=True,
                  source_identity=source_identity(root), normal_target_unchanged=True,
                  checkpoint_identities={str(k): v for k, v in identities.items()},
                  diagonal_reproduced=True, asset_sha_verified=True, lp_synchronized=True,
                  native_c_changed=native_changed,
                  delta_h=cross['fr']['hqnr']-late['fr']['hqnr'],
                  delta_ds=cross['fr']['d_s']-late['fr']['d_s'],
                  relative_delta_e=cross['rr']['ergas']/late['rr']['ergas']-1,
                  diagonal={str(k): v for k, v in diagonal.items()}, cross=cross)
    atomic_json(wd / 'diagnostics/au_cross_24240_50000.json', result)
    return result


def student_evidence(root, server, device='cuda', deadline=None):
    baseline = next(c for c in cases_for(server) if c.baseline_id_preserved and c.role == 'S')
    folder = camp(root, server) / 'diagnostics'
    result = dict(p0_passed=False, baseline_complete=False, diagnostics_complete=False,
                  budget_available=False, evidence_files=[], at_utc=utcnow())
    started = time.monotonic()
    try:
        from qg40.controller import shared_window, evaluation_debt
        from qg40.policy import admission
        window = shared_window(root)
        deadline = deadline or window.deadline_utc.isoformat()
        upcoming = [c for c in cases_for(server) if c.queue_rank == 50 and c.profile in ('BASE', 'B20')]
        result['budget_available'] = admission(window, utcnow(), upcoming, p0_ready=True,
            evaluation_debt_hours=evaluation_debt(root, server))['allowed']
        cfg, grid, selected = _official(root, baseline.run_id)
        result['baseline_complete'] = True
        preflight = read(camp(root, server) / 'preflight.json')
        result['p0_passed'] = preflight.get('complete') is True and preflight.get('source_identity') == source_identity(root)
        first, file1 = diagnostic(root, baseline.run_id, 24240)
        last, file2 = diagnostic(root, baseline.run_id, 50000)
        if first['diagnostic_indices_sha256'] != last['diagnostic_indices_sha256']:
            raise ValueError('Diagnostic train IDs changed between steps')
        result['evidence_files'] = [file1, file2]
        result['diagnostics_complete'] = True
        normal = {r['update']: r for r in grid['records']}
        spec = sensor_spec(baseline.sensor)
        target = selected['target']
        result['b20'] = dict(has_h_eligible=target is not None,
            e_goal_missed=bool(target and target['rr']['ergas'] >= spec.ergas_goal),
            gate_implementation_verified=result['p0_passed'],
            advantage_positive_fraction=[r['advantage_fraction'] for r in (first, last)],
            soft_hard_gradient_ratio_median=[r['soft_hard_gradient_ratio_median'] for r in (first, last)])
        result['e10'] = dict(lp_phase_valid=result['p0_passed'], band_order_valid=result['p0_passed'],
            metric_valid=result['p0_passed'], late_delta_ds=normal[50000]['fr']['d_s']-normal[24240]['fr']['d_s'],
            late_delta_dlambda=normal[50000]['fr']['d_lambda']-normal[24240]['fr']['d_lambda'],
            edge_hard_gradient_cosine=[r['edge_hard_gradient_cosine'] for r in (first, last)],
            weighted_edge_hard_gradient_ratio=[r['weighted_edge_hard_gradient_ratio'] for r in (first, last)])
        if result['budget_available']:
            # Its cost is diagnostic, not a fourth model or a normal selected row.
            try:
                cross = cross_aligner(root, baseline.run_id, device, deadline)
                result['a24r'] = {key: cross[key] for key in ('native_c_changed', 'diagonal_reproduced',
                    'asset_sha_verified', 'lp_synchronized', 'delta_h', 'delta_ds', 'relative_delta_e')}
                path = Path(root) / 'work_dir' / baseline.run_id / 'diagnostics/au_cross_24240_50000.json'
                result['evidence_files'].append(dict(path=str(path), sha256=sha256(path)))
            except (OSError, ValueError, TimeoutError) as error:
                result['a24r'] = dict(status='TO_VERIFY', reason=str(error))
                # With the highest-priority test incomplete, do not silently pick a lower axis.
                result['diagnostics_complete'] = False
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        result.update(diagnostics_complete=False, reason=str(error))
    if result.get('budget_available'):
        result['budget_available'] = admission(window, utcnow(), upcoming, p0_ready=True,
            evaluation_debt_hours=evaluation_debt(root, server))['allowed']
    result['diagnostic_seconds'] = time.monotonic() - started
    atomic_json(folder / 'student_trial_evidence.json', result)
    return result


def c3_evidence(root, server):
    teacher = next(c for c in cases_for(server) if c.baseline_id_preserved and c.role == 'T')
    result = dict(evidence_files=[], at_utc=utcnow())
    try:
        from qg40.references import validate_reference
        manifest, _, _, _ = validate_reference(teacher.reference_id, server, root)
        children = [c for c in cases_for(server) if c.baseline_id_preserved and c.role == 'S']
        for child in children:
            _official(root, child.run_id)
        first, file1 = diagnostic(root, teacher.run_id, 24240)
        last, file2 = diagnostic(root, teacher.run_id, 50000)
        preflight = read(camp(root, server) / 'preflight.json')
        valid = preflight.get('complete') is True and preflight.get('source_identity') == source_identity(root)
        result.update(parent_teacher_50k_complete=True, parent_student_seeds_completed=[c.seed for c in children],
            parent_students_official_complete=True, p0_passed=valid, sign_valid=valid, units_valid=valid,
            cache_valid=valid, augmentation_valid=valid, lp_ms_phase_valid=valid,
            q_ref=manifest['q_ref'], response_gain_median=last['response_gain_median'],
            weighted_consistency_rec_a_gradient_ratio=[r['weighted_consistency_rec_a_gradient_ratio'] for r in (first, last)],
            evidence_files=[file1, file2])
    except (OSError, ValueError, KeyError) as error:
        result.update(reason=str(error), p0_passed=False)
    atomic_json(camp(root, server) / 'diagnostics/c3_evidence.json', result)
    return result
