"""Pure B01 reporting contracts; deliberately no training/evaluation imports.

Callers must verify immutable evidence before mapping it. All Sheet cells are
RAW values. JSON object digests here are bridge digests, never file-byte SHA.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

SCHEMA = 'PANDA_RB_B01_SHEETS_BRIDGE_v1'
CAMPAIGN_ID = 'PANDA_REBUTTAL_B01_WV3_S135_20260928_v1'
SPREADSHEET_ID = '1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0'
SERVERS = ('s1', 's3', 's5')
CASES = ('QFULL', 'QMEAN', 'QSHUF', 'QESUR')
SELECTIONS = ('EXACT_50000', 'RR_VAL_ERGAS_MIN')
MODES = ('A_ON', 'A_ZERO_INFERENCE_ONLY')
RADII = (0., .25, .5, 1., 2., 3., 4.)
SEEDS = {'s1': (9281101, 9281102), 's3': (9281301, 9281302), 's5': (9281501, 9281502)}
NATIVE_SCOPE = 'NATIVE_RR20_FR20_v1'
CASE_LABELS = {'QFULL': 'original q', 'QMEAN': 'mean q', 'QSHUF': 'shuffled q', 'QESUR': 'e-surrogate q'}
NATIVE_HEADERS = (
    'Experiment', 'Server', 'Selection@Step', 'HQNR↑', 'D_s↓', 'D_lambda↓',
    'JQM↑', 'ERGAS↓', 'SCC↑', 'SAM↓', 'PSNR↑', 'SSIM↑', 'Q4/Q8↑', 'RMSE↓',
    'CC↑', 'Infer(ms)', 'Mem(MB)', 'Params(M)', 'FLOPs(G)', 'Train(h)',
    'Eval(h)', 'Wall(h)', 'HQNR(V64)↑', 'Signed D_s', 'Positive D_s fraction',
    'Dataset', 'Bucket', 'Approach', 'Campaign_ID', 'Run_ID', 'Case_ID',
    'Repeat', 'Seed', 'Attempt', 'Role', 'Model / inputs', 'Teacher / reference',
    'Parent_run', 'Updates', 'Lifetime_updates', 'Selected_step', 'Checkpoint_SHA',
    'Selector', 'q_ref', 'tau_R', 'alpha', 'beta', 'lambda_E', 'U_peak_lr',
    'A_peak_lr', 'Status', 'Readback', 'Eval_scope', 'Test_aware', 'JQM_variant',
    'Cost_scope', 'Source_sheet', 'Source_row', 'Source_URL', 'Result_ID',
    'Original description / notes', 'Review', 'Sort_case', 'Date',
)
STRESS_HEADERS = (
    'Record_ID', 'Schema', 'Campaign_ID', 'Run_ID', 'Case_ID', 'Server', 'Repeat', 'Seed',
    'Checkpoint_SHA', 'Selector', 'Mode', 'Shift_ID', 'Radius_HR', 'Angle_deg', 'Dy', 'Dx',
    'N_scenes', 'N_failures', 'ERGAS', 'PSNR', 'SAM', 'Edge_error_DN',
    'Relative_response_L1_sum', 'Coverage_all_pan_paths',
    'Delta0_ERGAS', 'Delta0_PSNR', 'Delta0_SAM', 'Delta0_Edge_error_DN',
    'ERGAS_scene_SD', 'PSNR_scene_SD', 'SAM_scene_SD', 'Edge_error_scene_SD',
    'Curve_numerical_failures', 'Curve_invalid_geometry', 'ROI', 'Grid_SHA',
    'Evidence_payload_SHA', 'Source_completed_at_utc', 'Status', 'Readback',
    'Source_artifact', 'Provenance', 'Review',
)
STATUS_HEADERS = (
    'Campaign_ID', 'Run_ID', 'Case_ID', 'Server', 'Repeat', 'Seed', 'Expected_updates',
    'Actual_updates', 'Training_status', 'Native_status', 'Expected_selectors', 'Selector_count',
    'A_ON_status', 'A_ON_expected_shifts', 'A_ON_shift_count', 'A_ON_numerical_failures',
    'A_ON_invalid_geometry', 'A_ZERO_status', 'A_ZERO_expected_shifts', 'A_ZERO_shift_count',
    'A_ZERO_numerical_failures', 'A_ZERO_invalid_geometry', 'Verification_status',
    'Upload_status', 'Error', 'Source_hash', 'Last_checked_at_utc',
)
NATIVE_METRICS = ('hqnr', 'd_s', 'd_lambda', 'jqm', 'ergas', 'scc', 'sam', 'psnr', 'ssim', 'q8', 'rmse', 'cc')
STRESS_METRICS = ('ergas', 'psnr', 'sam', 'edge_error_dn', 'relative_response_l1_sum', 'coverage_all_pan_paths')
RUNTIME_KEYS = ('torch', 'cuda', 'cudnn', 'numpy', 'scipy', 'skimage', 'tf32_matmul', 'tf32_cudnn')


class ContractError(ValueError):
    """Evidence/schema conflict; never silently coerce or overwrite."""


def compact_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def object_sha(value):
    # Match original FH12/B01 payload JSON convention; this is not byte SHA.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def finite_number(value, name='value', *, optional=False, integer=False):
    if value is None:
        if optional:
            return None
        raise ContractError(f'{name}: missing required number')
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError(f'{name}: expected finite JSON number, not bool/string/nonfinite')
    if integer and (type(value) is not int):
        raise ContractError(f'{name}: expected JSON integer')
    return value


def resolve_alias(mapping, canonical, *aliases, optional=False):
    present = [(key, mapping[key]) for key in (canonical, *aliases) if key in mapping]
    if not present:
        return finite_number(None, canonical, optional=optional)
    values = [finite_number(value, key, optional=optional) for key, value in present]
    if any(value != values[0] for value in values[1:]):
        raise ContractError(f'Conflicting aliases: {canonical}/{aliases}')
    return values[0]


def validate_sha(value, name='SHA256'):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ContractError(f'{name}: expected lowercase full SHA256')
    return value


def validate_headers(actual, expected=NATIVE_HEADERS):
    if list(actual) != list(expected):
        raise ContractError('SCHEMA_CONFLICT: header names/order differ; refusing replacement')


def registry():
    """Deterministic ledger order (not the counterbalanced training schedule)."""
    return [dict(campaign_id=CAMPAIGN_ID, run_id=f'RB01_WV3_{server.upper()}_R{repeat}_SS{seed}_{case}_F50K_v1',
                 server=server, repeat=repeat, seed=seed, case_id=case, dataset='WV3')
            for server in SERVERS for repeat, seed in enumerate(SEEDS[server], 1) for case in CASES]


def validate_case(record):
    expected = next((row for row in registry() if row['run_id'] == record.get('run_id')), None)
    if expected is None:
        raise ContractError('Unregistered B01 Student')
    for key in ('server', 'repeat', 'seed', 'case_id'):
        if record.get(key) != expected[key] or (key in ('repeat', 'seed') and type(record.get(key)) is not int):
            raise ContractError(f'Registry mismatch: {key}')
    if record.get('campaign_id', CAMPAIGN_ID) != CAMPAIGN_ID or record.get('dataset', 'WV3') != 'WV3':
        raise ContractError('Campaign/dataset mismatch')
    return expected


def source_row(repeat, case_id, selection_id):
    if type(repeat) is not int or repeat not in (1, 2) or case_id not in CASES or selection_id not in SELECTIONS:
        raise ContractError('Invalid native slot')
    return 2 + (repeat - 1) * 8 + CASES.index(case_id) * 2 + SELECTIONS.index(selection_id)


def stress_source_row(record):
    row = validate_case(record)
    if record['mode'] not in MODES or not re.fullmatch(r'D0[0-4][0-9]', record['shift_id']) or int(record['shift_id'][1:]) > 48:
        raise ContractError('Invalid stress slot')
    run_index = next(i for i, value in enumerate(registry()) if value['run_id'] == row['run_id'])
    return 2 + run_index * 98 + MODES.index(record['mode']) * 49 + int(record['shift_id'][1:])


def native_result_id(record):
    validate_case(record)
    if record['selection_id'] not in SELECTIONS:
        raise ContractError('Unregistered native selector')
    return '|'.join(('RB01', CAMPAIGN_ID, record['run_id'], record['selection_id'],
                     validate_sha(record['checkpoint_sha256']), NATIVE_SCOPE))


def stress_record_id(record):
    stress_source_row(record)
    return '|'.join(('RB02', CAMPAIGN_ID, record['run_id'], validate_sha(record['checkpoint_sha256']),
                     record['mode'], validate_sha(record['grid_sha256']), record['shift_id']))


def logical_key(record):
    validate_case(record)
    if 'shift_id' in record:
        return (CAMPAIGN_ID, record['run_id'], record['mode'], record['shift_id'])
    return (CAMPAIGN_ID, record['run_id'], record['selection_id'], NATIVE_SCOPE)


def evidence_payload(record):
    """Transport retries/readbacks may not change sealed-data payload identity."""
    return {key: value for key, value in record.items()
            if key not in ('readback', 'upload_status', 'source_url', 'source_row', 'uploaded_at_utc')}


def assert_no_conflicts(records):
    unique = {}
    for record in records:
        key = logical_key(record)
        if key in unique and evidence_payload(unique[key]) != evidence_payload(record):
            raise ContractError(f'EVIDENCE_CONFLICT: {key}')
        unique.setdefault(key, record)
    return list(unique.values())


def cohort_id(provenance):
    """Never pool unknown identity. Per-case transformed maps are not cohorts."""
    names = ('numerical_source_sha256', 'evaluator_sha256', 'binding_common_sha256',
             'data_content_identity_sha256', 'teacher_sha256', 'runtime_policy_sha256',
             'raw_map_identity_sha256')
    identity = {key: provenance.get(key) for key in names}
    missing = [key for key, value in identity.items() if not value]
    if missing:
        raise ContractError(f'Incomplete cohort identity: {missing}')
    for key, value in identity.items():
        validate_sha(value, key)
    # Shared raw q/e arrays are explicit cohort evidence; seed-specific
    # shuffled/transformed maps remain provenance, not cross-case cohort keys.
    return object_sha(identity)


def _entry_base(entry, context):
    case = validate_case(entry['case'])
    if context.get('case', {}).get('run_id') != case['run_id']:
        raise ContractError('Report/context run mismatch')
    provenance = dict(entry.get('provenance', {}))
    source = context['source_identity']
    binding = entry.get('binding', {})
    inferred = dict(numerical_source_sha256=source['content_sha256'],
                    evaluator_sha256=object_sha(context['evaluator_identity']),
                    binding_common_sha256=context['binding_common_sha256'],
                    data_content_identity_sha256=object_sha(context['data_content_identity']),
                    runtime_policy_sha256=object_sha({key: source.get(key) for key in RUNTIME_KEYS}))
    raw_maps = provenance.get('raw_map_identity')
    if not isinstance(raw_maps, dict) or not all(key in raw_maps for key in ('q', 'e_bar')):
        raise ContractError('Shared raw q/e map identity missing')
    inferred['raw_map_identity_sha256'] = object_sha({key: validate_sha(raw_maps[key], key)
                                                    for key in ('q', 'e_bar')})
    if binding.get('teacher_checkpoint_sha256'):
        inferred['teacher_sha256'] = binding['teacher_checkpoint_sha256']
    for key, value in inferred.items():
        if key in provenance and provenance[key] != value:
            raise ContractError(f'Provenance identity conflict: {key}')
        provenance[key] = value
    provenance['runtime'] = {key: source.get(key) for key in RUNTIME_KEYS}
    provenance['bridge_schema'] = SCHEMA
    return case, provenance


def canonical_native(entry):
    report = entry['report']
    if report.get('schema') != 'PANDA_RB01_NATIVE_v1' or report.get('complete') is not True:
        raise ContractError('Native evidence is not complete B01 schema')
    if set(report.get('selections', {})) != set(SELECTIONS):
        raise ContractError('Native must preserve both registered selectors')
    case, provenance = _entry_base(entry, report['context'])
    cfg, binding = entry['config'], entry['binding']
    args, policy = cfg['model_args'], cfg['panda_rb']
    config_fields = {key: finite_number(policy.get(key), key) for key in ('alpha', 'beta', 'lambda_E')}
    for key in ('q_ref', 'tau_R'):
        value = finite_number(binding.get(key), key)
        if key in binding.get('common', {}) and binding['common'][key] != value:
            raise ContractError(f'Binding alias conflict: {key}')
        config_fields[key] = value
    config_fields.update(U_peak_lr=finite_number(cfg.get('learning_rate'), 'learning_rate'),
                         A_peak_lr=finite_number(policy.get('aligner_lr'), 'aligner_lr'),
                         model_inputs=f"{policy['input_layout']} / W{args['hidden_size']} / D{json.dumps(args['depth'], separators=(',', ':'))} / A_ON",
                         teacher_reference=f"F1 exact50K / {validate_sha(binding['teacher_checkpoint_sha256'])[:12]}")
    train_seconds = entry.get('training_status', {}).get('timings', {}).get('train')
    costs = dict(train_hours=None if train_seconds is None else finite_number(train_seconds, 'train seconds') / 3600,
                 eval_hours=finite_number(report.get('elapsed_seconds'), 'native elapsed seconds') / 3600)
    cost_scope = 'Train: optimizer timing seconds/3600; Eval: run-level total for BOTH selectors seconds/3600; do not sum selector rows; latency/memory/params/FLOPs/wall NOT_PROFILED'
    records = []
    for selector in SELECTIONS:
        selected = report['selections'][selector]
        if selected.get('selection_id') != selector:
            raise ContractError('Selection label differs from map key')
        update = finite_number(selected.get('update'), 'update', integer=True)
        if update <= 0 or update > 50000 or (selector == 'EXACT_50000' and update != 50000):
            raise ContractError('Selection update is outside fixed endpoint')
        rr, fr = selected['rr'], selected['fr']
        metrics = {key: finite_number(rr.get(key), key) for key in ('ergas', 'scc', 'sam', 'psnr', 'ssim')}
        metrics['hqnr'] = finite_number(fr.get('hqnr'), 'hqnr')
        metrics['d_s'] = resolve_alias(fr, 'd_s', 'ds')
        metrics['d_lambda'] = resolve_alias(fr, 'd_lambda', 'dlambda')
        if 'q2n' in rr and not ('q8' in rr or entry.get('verification', {}).get('q2n_is_q8') is True):
            raise ContractError('q2n-only source lacks verified WV3 Q8 definition')
        metrics['q8'] = resolve_alias(rr, 'q8', 'q2n')
        metrics.update({key: finite_number(selected.get('supplemental_rr', {}).get(key), key, optional=True)
                        for key in ('rmse', 'cc')})
        metrics['jqm'] = finite_number(selected.get('supplemental_fr', {}).get('jqm'), 'jqm', optional=True)
        alias = selected.get('alias_of')
        if alias is not None and (selector != SELECTIONS[1] or alias != SELECTIONS[0]):
            raise ContractError('Invalid selector alias')
        record = dict(case, selection_id=selector, checkpoint_sha256=validate_sha(selected['checkpoint_sha256']),
                      update=update, metrics=metrics, alias_of=alias, config_fields=config_fields,
                      costs=costs, cost_scope=cost_scope, jqm_variant=selected.get('jqm_variant'),
                      completed_at_utc=report['completed_at_utc'],
                      provenance=dict(provenance, alias_of=alias, native_payload_sha256=report['payload_sha256'],
                                      source_completed_at_utc=report['completed_at_utc'], rr_scene_count=20,
                                      fr_scene_count=20, native_coordinate_frame='native unshifted; A_ON',
                                      cost_scope=cost_scope), review='SELECTION_ALIAS' if alias else '')
        record['cohort_id'] = cohort_id(record['provenance'])
        records.append(record)
    first, second = records
    if first['checkpoint_sha256'] == second['checkpoint_sha256']:
        if second['alias_of'] != SELECTIONS[0] or first['metrics'] != second['metrics'] or first['update'] != second['update']:
            raise ContractError('Same checkpoint must preserve exact metric/step alias')
    elif second['alias_of'] is not None:
        raise ContractError('Alias names a different checkpoint')
    return records


def canonical_stress(entry):
    report = entry['report']
    if report.get('schema') != 'PANDA_RB02_CURVE_v1' or report.get('complete') is not True:
        raise ContractError('Stress evidence is not complete B01 schema')
    identity = report['identity']
    case, provenance = _entry_base(entry, identity['context'])
    if identity.get('source_selection') != SELECTIONS[0] or identity.get('update') != 50000 or identity.get('mode') not in MODES:
        raise ContractError('Stress must use exact50K and registered paired inference modes')
    if report.get('n_scenes') != 20 or report.get('n_shifts') != 49 or report.get('n_observations') != 980 or len(report['curve']) != 49:
        raise ContractError('Stress does not preserve 49x20 protocol')
    numerical = finite_number(report['n_numerical_failures'], 'numerical failures', integer=True)
    geometry = finite_number(report['n_invalid_geometry'], 'invalid geometry', integer=True)
    records = []
    for i, point in enumerate(report['curve']):
        if point['id'] != f'D{i:03d}':
            raise ContractError('Unexpected stress shift grid order')
        radius = finite_number(point['radius_hr'], 'radius')
        angle = resolve_alias(point, 'angle_deg', 'angle_degrees', optional=True)
        expected_radius = 0. if i == 0 else RADII[(i-1)//8+1]
        expected_angle = None if i == 0 else ((i-1) % 8) * 45
        if radius != expected_radius or angle != expected_angle:
            raise ContractError('Stress grid radius/angle differs from registration')
        dy, dx = (finite_number(point[key], key) for key in ('dy', 'dx'))
        radians = math.radians(angle or 0)
        if not math.isclose(dy, radius*math.sin(radians), abs_tol=1e-12) or not math.isclose(dx, radius*math.cos(radians), abs_tol=1e-12):
            raise ContractError('Stress grid coordinates differ from registration')
        n_scenes = finite_number(point['n_scenes'], 'n_scenes', integer=True)
        n_failures = finite_number(point['n_failures'], 'n_failures', integer=True)
        if n_scenes != 20 or not 0 <= n_failures <= 20 or numerical < 0 or geometry < 0:
            raise ContractError('Invalid scene/failure counts')
        keys = (*STRESS_METRICS, *(f'{key}_scene_std' for key in STRESS_METRICS[:4]),
                *(f'delta_from_zero_{key}' for key in STRESS_METRICS[:4]))
        values = {key: finite_number(point.get(key), key, optional=True) for key in keys}
        reasons = []
        if n_failures or any(values[key] is None for key in STRESS_METRICS): reasons.append('POINT_NUMERICAL_OR_MISSING')
        if values['coverage_all_pan_paths'] is not None and values['coverage_all_pan_paths'] < 1.: reasons.append('POINT_COVERAGE_BELOW_ONE')
        if geometry: reasons.append('CURVE_INVALID_GEOMETRY_NOT_SHIFT_COUNT')
        if numerical: reasons.append('CURVE_NUMERICAL_FAILURES')
        record = dict(case, checkpoint_sha256=validate_sha(identity['checkpoint_sha256']),
                      mode=identity['mode'], grid_sha256=validate_sha(identity['grid_sha256']), shift_id=point['id'],
                      radius_hr=radius, angle_deg=angle, dy=dy, dx=dx, n_scenes=n_scenes, n_failures=n_failures,
                      curve_numerical_failures=numerical, curve_invalid_geometry=geometry,
                      payload_sha256=report['payload_sha256'], completed_at_utc=report['completed_at_utc'],
                      source_artifact=provenance.get('source_file_relative_path', ''),
                      provenance=dict(provenance, roi='32:-32 / fixed192', scene_SD='source population SD; not Student sample SD',
                                      a_zero_interpretation='inference correction zero, not independently trained no-align'),
                      review=';'.join(reasons), **values)
        record['cohort_id'] = cohort_id(record['provenance'])
        records.append(record)
    return records


def _cell(value):
    if value is None:
        return ''
    if type(value) in (int, float):
        finite_number(value)
    if not isinstance(value, (str, bool, int, float)):
        raise ContractError('Unsupported Sheet cell value')
    return value


def build_native_row(record, spreadsheet_id=SPREADSHEET_ID, gid=None):
    validate_case(record)
    metrics, config, costs = record['metrics'], record.get('config_fields', {}), record.get('costs', {})
    slot = source_row(record['repeat'], record['case_id'], record['selection_id'])
    provenance = dict(record['provenance'], cohort_id=record.get('cohort_id') or cohort_id(record['provenance']))
    provenance['selection_is_test_aware'] = False
    provenance['test_awareness_scope'] = 'endpoint or RR validation-only selector; not a claim about historical development/test usage'
    row = [f"{record['case_id']} | {CASE_LABELS[record['case_id']]} | R{record['repeat']}", record['server'],
           f"{record['selection_id']}@{record['update']}", *[metrics.get(key) for key in NATIVE_METRICS],
           *[costs.get(key) for key in ('infer_ms', 'memory_mb', 'params_m', 'flops_g', 'train_hours', 'eval_hours', 'wall_hours')],
           None, None, None, 'WV3', 'Main', '06 | Rebuttal B01 | RB01 q reliability', CAMPAIGN_ID,
           record['run_id'], record['case_id'], record['repeat'], record['seed'], None, 'S',
           config.get('model_inputs'), config.get('teacher_reference'), None, 50000, 50000, record['update'],
           record['checkpoint_sha256'], record['selection_id'],
           *[config.get(key) for key in ('q_ref', 'tau_R', 'alpha', 'beta', 'lambda_E', 'U_peak_lr', 'A_peak_lr')],
           'NATIVE_EVAL_COMPLETE', record.get('readback', 'UPLOAD_PENDING'),
           'RB01_NATIVE_RR20_FR20;RR20:-21;FR_FULL512;A_ON', False, record.get('jqm_variant'),
           record.get('cost_scope'), f"_rb01_{record['server']}", slot,
           None if gid is None else f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit#gid={gid}&range=A{slot}:BL{slot}',
           native_result_id(record), compact_json(provenance), record.get('review', ''),
           CASES.index(record['case_id'])+1, record['completed_at_utc']]
    if len(row) != 64:
        raise ContractError(f'Internal native width {len(row)} !=64')
    return [_cell(value) for value in row]


def build_stress_row(record):
    row = [stress_record_id(record), 'RB02_SHEET_POINT_v1', CAMPAIGN_ID, record['run_id'], record['case_id'],
           record['server'], record['repeat'], record['seed'], record['checkpoint_sha256'], SELECTIONS[0],
           record['mode'], record['shift_id'], record['radius_hr'], record['angle_deg'], record['dy'], record['dx'],
           record['n_scenes'], record['n_failures'], *[record.get(key) for key in STRESS_METRICS],
           *[record.get(f'delta_from_zero_{key}') for key in STRESS_METRICS[:4]],
           *[record.get(f'{key}_scene_std') for key in STRESS_METRICS[:4]],
           record['curve_numerical_failures'], record['curve_invalid_geometry'], '32:-32 / fixed192',
           record['grid_sha256'], record['payload_sha256'], record['completed_at_utc'],
           'COLLECTED_WITH_FAILURES' if record.get('review') else 'CURVE_POINT_COMPLETE',
           record.get('readback', 'UPLOAD_PENDING'), record.get('source_artifact'),
           compact_json(dict(record['provenance'], cohort_id=record.get('cohort_id') or cohort_id(record['provenance']))),
           record.get('review', '')]
    if len(row) != 43:
        raise ContractError(f'Internal stress width {len(row)} !=43')
    return [_cell(value) for value in row]


def build_status_row(record):
    validate_case(record)
    fields = ('actual_updates', 'training_status', 'native_status')
    row = [CAMPAIGN_ID, record['run_id'], record['case_id'], record['server'], record['repeat'], record['seed'],
           50000, *[record.get(key) for key in fields], 2, record.get('native_selectors', record.get('selector_count'))]
    for mode, source_mode in zip(('a_on', 'a_zero'), MODES):
        curve = record.get('curve_status', {}).get(source_mode, {})
        row.extend([curve.get('status', record.get(f'{mode}_status')), 49,
                    curve.get('shifts', record.get(f'{mode}_shift_count')),
                    curve.get('numerical_failures', record.get(f'{mode}_failures')),
                    curve.get('invalid_geometry', record.get(f'{mode}_invalid_geometry'))])
    error = compact_json(record['errors']) if record.get('errors') else record.get('error')
    row.extend([record.get('verification_status'), record.get('upload_status'), error,
                record.get('source_hash'), record.get('last_checked_at_utc')])
    return [_cell(value) for value in row]
