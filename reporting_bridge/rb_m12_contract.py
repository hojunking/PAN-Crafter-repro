"""M12-only transport contract. No authentication, training, or old bridge imports."""
from __future__ import annotations

import hashlib
import json
import math
import re

CAMPAIGN = 'PANDA_REBUTTAL_B02_M12_WV3_S135_20261001_v1'
SCHEMA = 'PANDA_RB_M12_BRIDGE_v1'
SPREADSHEET_ID = '1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0'
SERVERS = ('s1', 's3', 's5')
CASES = ('QFULL', 'QMEAN', 'QSHUF', 'QESUR', 'QEDGE', 'QALIGN', 'H0', 'HSPMEAN', 'NOADV', 'ADVMEAN')
SELECTIONS = ('EXACT_50000', 'RR_VAL_ERGAS_MIN')
MODES = ('A_ON', 'A_ZERO_INFERENCE_ONLY', 'A_NATIVE_FIXED', 'KNOWN_SHIFT_INVERSE')
SEEDS = {s: tuple(261001000 + int(s[1:])*100 + r for r in range(1, 5)) for s in SERVERS}
NATIVE_PROTOCOL = 'M12_NATIVE_RR20_FR20_v1;RR20:-21;FR_FULL512;A_ON'
STRESS_PROTOCOL = 'M12_STRESS_FIXED160_v1;48:-48;49x20'
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
POINT_HEADERS = (
    'Record_ID', 'Schema', 'Campaign_ID', 'Run_ID', 'Case_ID', 'Server', 'Repeat', 'Seed',
    'Checkpoint_SHA', 'Selector', 'Mode', 'Shift_ID', 'Radius_HR', 'Angle_deg', 'Dy', 'Dx',
    'N_scenes', 'N_valid', 'N_failures', 'ERGAS', 'PSNR', 'SAM', 'Edge_error_DN',
    'Response_L1_sum', 'Coverage_all_pan_paths', 'Protocol', 'ROI', 'Grid_SHA',
    'Evidence_payload_SHA', 'Status', 'Source_artifact', 'Provenance', 'Review',
)
STATUS_HEADERS = ('Campaign_ID', 'Run_ID', 'Case_ID', 'Server', 'Repeat', 'Seed',
                  'Expected_updates', 'Actual_updates', 'Training_status', 'Native_selectors',
                  'Stress_curves', 'Clean_curves', 'Flagged_curves', 'Error', 'Checked_at_utc')
METRICS = ('hqnr', 'd_s', 'd_lambda', 'jqm', 'ergas', 'scc', 'sam', 'psnr', 'ssim', 'q8', 'rmse', 'cc')
COHORT_KEYS = ('numerical_source_sha256', 'evaluator_sha256', 'binding_common_sha256',
               'data_content_identity_sha256', 'teacher_sha256', 'runtime_policy_sha256',
               'raw_map_identity_sha256')


class ContractError(ValueError):
    pass


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ContractError('Missing or invalid actual SHA256')
    return value


def number(value, *, optional=False):
    if optional and value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError('Expected a finite JSON number, not bool/string/nonfinite')
    return value


def registry():
    return [dict(campaign_id=CAMPAIGN, run_id=f'RBM12_WV3_{s.upper()}_R{r}_SS{seed}_{c}_F50K_v1',
                 server=s, repeat=r, seed=seed, case_id=c, dataset='WV3')
            for s in SERVERS for r, seed in enumerate(SEEDS[s], 1) for c in CASES]


def validate_case(record):
    row = next((r for r in registry() if r['run_id'] == record.get('run_id')), None)
    if row is None:
        raise ContractError('Not a preregistered new M12 Student')
    for key in ('server', 'repeat', 'seed', 'case_id'):
        if record.get(key) != row[key] or (key in ('repeat', 'seed') and type(record.get(key)) is not int):
            raise ContractError('Registry mismatch: ' + key)
    if record.get('campaign_id', CAMPAIGN) != CAMPAIGN or record.get('dataset', 'WV3') != 'WV3':
        raise ContractError('Historical B01/ABLR2 must not enter the M12 cohort')
    return row


def source_row(record):
    validate_case(record)
    if record.get('selection_id') not in SELECTIONS:
        raise ContractError('Unregistered selector')
    return 2 + (record['repeat'] - 1)*20 + CASES.index(record['case_id'])*2 + SELECTIONS.index(record['selection_id'])


def status_row(record):
    validate_case(record)
    return 2 + next(i for i, r in enumerate(registry()) if r['run_id'] == record['run_id'])


def point_row(record):
    validate_case(record)
    if record['case_id'] not in CASES[:6] or record.get('mode') not in MODES:
        raise ContractError('Stress is registered for STEP1 only, in four modes')
    shift = record.get('shift_id', '')
    if not re.fullmatch('D[0-9]{3}', shift) or not 0 <= int(shift[1:]) < 49:
        raise ContractError('Unregistered shift')
    index = SERVERS.index(record['server'])*24 + (record['repeat'] - 1)*6 + CASES.index(record['case_id'])
    return 2 + index*196 + MODES.index(record['mode'])*49 + int(shift[1:])


def result_id(record):
    source_row(record)
    if record.get('protocol') != NATIVE_PROTOCOL:
        raise ContractError('Native protocol must never be a stress ROI')
    return 'RBM12:' + digest([CAMPAIGN, record['run_id'], record['selection_id'],
                             sha(record['checkpoint_sha256']), record['protocol'],
                             sha(record['provenance']['evaluator_sha256'])])


def point_id(record):
    point_row(record)
    if record.get('protocol') != STRESS_PROTOCOL:
        raise ContractError('Only preregistered fixed160 primary points go online')
    return 'RBM12P:' + digest([CAMPAIGN, record['run_id'], sha(record['checkpoint_sha256']),
                              record['mode'], record['shift_id'], record['protocol'],
                              sha(record['grid_sha256']), sha(record['provenance']['evaluator_sha256'])])


def cohort_id(provenance):
    return digest({key: sha(provenance.get(key)) for key in COHORT_KEYS})


def _cells(values):
    for value in values:
        if value is None:
            yield ''
        elif isinstance(value, (str, bool)):
            yield value
        elif type(value) in (int, float):
            yield number(value)
        else:
            raise ContractError('Sheet payload cannot contain arrays, dicts, credentials, or weights')


def build_native_row(record, gid):
    rownum = source_row(record)
    sha(record['checkpoint_sha256'])
    metric, config, costs = record['metrics'], record.get('config_fields', {}), record.get('costs', {})
    if record['selection_id'] == SELECTIONS[0] and record['update'] != 50000:
        raise ContractError('Primary is exact50000, not a score-selected checkpoint')
    if type(record['update']) is not int or record['update'] not in (*range(1010, 50000, 1010), 50000):
        raise ContractError('Unregistered validation grid')
    for key in ('hqnr', 'd_s', 'd_lambda', 'ergas', 'scc', 'sam', 'psnr', 'ssim', 'q8'):
        number(metric.get(key))
    provenance = dict(record['provenance'], cohort_id=cohort_id(record['provenance']),
                      alias_of=record.get('alias_of'), independent_student_unit=record['run_id'],
                      protocol=record['protocol'])
    approach = '06 | Rebuttal M12 | ' + ('01 q cue and routing' if record['case_id'] in CASES[:6] else '02 hard-soft fitting')
    row = [f"M12 {record['case_id']} | R{record['repeat']}", record['server'],
           f"{record['selection_id']}@{record['update']}", *[metric.get(k) for k in METRICS],
           *[costs.get(k) for k in ('infer_ms', 'memory_mb', 'params_m', 'flops_g', 'train_hours', 'eval_hours', 'wall_hours')],
           None, None, None, 'WV3', 'Ablations', approach, CAMPAIGN, record['run_id'], record['case_id'],
           record['repeat'], record['seed'], record.get('attempt', 1), 'S',
           'PLH / W104 / D[1,2,2] / LN / A_ON', config.get('teacher_reference'), None,
           50000, 50000, record['update'], record['checkpoint_sha256'], record['selection_id'],
           *[config.get(k) for k in ('q_ref', 'tau_R', 'alpha', 'beta', 'lambda_E', 'U_peak_lr', 'A_peak_lr')],
           'NATIVE_EVAL_COMPLETE', 'UPLOAD_PENDING', record['protocol'], False, record.get('jqm_variant'),
           record.get('cost_scope', 'Per-run costs repeated across selectors; never sum selection aliases'),
           '_rb_m12_' + record['server'], rownum,
           f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit#gid={int(gid)}&range=A{rownum}:BL{rownum}',
           result_id(record), compact(provenance), '', CASES.index(record['case_id']) + 1, record['completed_at_utc']]
    if len(row) != 64:
        raise AssertionError('Canonical native schema width changed')
    return list(_cells(row))


def build_point_row(record):
    point_row(record)
    if record.get('n_scenes') != 20 or not 0 <= record.get('n_valid', -1) <= 20:
        raise ContractError('Stress scene completeness must be explicit')
    clean = record['n_valid'] == 20 and record.get('n_failures', 0) == 0
    row = [point_id(record), SCHEMA, CAMPAIGN, record['run_id'], record['case_id'], record['server'],
           record['repeat'], record['seed'], record['checkpoint_sha256'], SELECTIONS[0], record['mode'],
           record['shift_id'], record['radius_hr'], record.get('angle_deg'), record['dy'], record['dx'],
           20, record['n_valid'], record.get('n_failures', 0),
           *[record.get(k) for k in ('ergas', 'psnr', 'sam', 'edge_error_dn', 'relative_response_l1_sum', 'coverage_all_pan_paths')],
           STRESS_PROTOCOL, '48:-48 / fixed160', sha(record['grid_sha256']), sha(record['payload_sha256']),
           'CLEAN' if clean else 'PROCESSED_WITH_FLAGS', record.get('source_artifact', ''),
           compact(record.get('provenance', {})), record.get('review', '')]
    return list(_cells(row))


def build_status_row(record):
    validate_case(record)
    return list(_cells([CAMPAIGN, record['run_id'], record['case_id'], record['server'], record['repeat'], record['seed'],
                       50000, record.get('actual_updates'), record.get('training_status', 'PENDING'),
                       record.get('native_selectors', 0), record.get('stress_curves', 0),
                       record.get('clean_curves', 0), record.get('flagged_curves', 0),
                       compact(record.get('errors', [])), record.get('checked_at_utc')]))
