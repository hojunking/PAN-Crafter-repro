"""WV3 Table 3: only fixed50 results; isolated own-row publication."""
from datetime import datetime, timedelta
from functools import lru_cache
import json
import math
from pathlib import Path
from types import SimpleNamespace

from directshift48.reporting import (HEADERS, INDEX, METRICS, serialize_row,
                                    atomic_json, canonical_json, utc_now)
from sheet_publish_v2.transport import _typed, _matrix_equal
from ta2.evaluation import file_sha
from . import plan
from .evaluation import verify_record

POLICY = 'TAB3_FIXED50_ONLY_3REPEATS_4DP_V1'
R = SimpleNamespace(atomic_json=atomic_json, utc_now=utc_now)


def destination(server):
    sensor=plan.sensor_for(server)
    return {'spreadsheet_id':'1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE',
            'sheet_title':sensor+'-ablations','sheet_id':261001010 if sensor=='WV3' else 261001011}


@lru_cache(maxsize=4096)
def _verified_file(path, inode, size, mtime):
    return file_sha(path)


def artifact(meta):
    path = plan.local_path(meta['path']); stat = path.stat()
    if _verified_file(str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns) != meta['file_sha256']:
        raise ValueError('Published fresh checkpoint changed: ' + str(path))


def choose(records, stage, cfg):
    """Only the predeclared endpoint; no best-checkpoint or seed selection."""
    if stage != 'stage1':
        raise ValueError('This is a Stage1-only Table 3 campaign')
    by_step = {}
    for record in records:
        step = record['step']
        if step in by_step and by_step[step] != record:
            raise ValueError('Conflicting evaluations for one checkpoint')
        by_step[step] = record
    return [(by_step[50000], 'FIXED50K', 'ROW0')] if 50000 in by_step else []


def joint_pass(record,cfg):
    return bool(record and record['fr']['metrics']['HQNR'] >= plan.targets(cfg['sensor'])['HQNR_min']
                and record['rr']['metrics']['ERGAS'] <= plan.targets(cfg['sensor'])['ERGAS_max'])


def approach(cfg):
    common=f"{cfg['reference_mode']} all edge/RRE; W{cfg['width']} D123 UBIL Teacher; all-L4; common eroded support"
    return f"Tab3 {cfg['table_row']} {cfg['alignment']}; S1 {common}; spa={cfg['lambda_abs']:g},eps={cfg['lambda_epsilon']:g}; R_eps=3; A/Tlr={cfg['lr_aligner']:g}/{cfg['lr']:g}; C50"


def make_row(cfg, status, record=None, selector='STATUS', slot='ROW0'):
    if record is None or record['step'] != 50000 or selector != 'FIXED50K':
        raise ValueError('Only fixed 50K results may be published')
    step = record['step']
    fixed, representative, passed = 'FIXED50K', False, joint_pass(record,cfg)
    label = 'FIXED50K'
    note = {'schema_version': 'PAN_FIXED64_V1', 'reset_epoch': '2026-10-13',
        'configuration': cfg,
        'publication': {'policy': POLICY, 'selector': selector, 'slot': slot,
            'joint_pass': passed, 'is_case_representative': representative,
            'is_fixed_step_comparison': bool(fixed),
            'targets': plan.targets(cfg['sensor'])},
        'fixed_step_comparison': cfg['primary_step'], 'reference_repeats': 3,
        'evaluation_file': None if record is None else record['evidence_path'],
        'evaluation_sha256': None if record is None else record['evidence_sha256'],
        'model_state_sha256': None if record is None else record['model_sha256'],
        'artifact': None if record is None else record['checkpoint'],
        'reference_definition': {
            'L_spa': f"GT {cfg['reference_mode']} arithmetic mean {cfg['reference_bands']}, Scharr; RGBN also used for RRE; Stage2 q not executed",
            'alignment_input': f"Stored LMS all {cfg['bands']} bands",
            'validation_selector': 'NONE; fixed 50K for every run; validation is diagnostic only',
            'teacher_input': 'corrected PAN, interp23 I_lrms; no Student training in this campaign',
            'warp': 'Stage1 correction+epsilon/Teacher/validation/test all Lanczos4-reflection',
            'fixed_step_selection': 'Predeclared; no test-best checkpoint substitution'}}
    if record:
        if 'test_rre' in record:
            note['test_rre'] = record['test_rre']
        if 'validation' in record:
            note['validation'] = {k: v for k, v in record['validation'].items()
                                  if not isinstance(v, (dict, list))}
        if record.get('reuse'):
            note['reuse'] = record['reuse']
        if record.get('stage1_re_evaluation'):
            note['stage1_re_evaluation'] = record['stage1_re_evaluation']
    rid = 'TAB3RGBN-' + plan.digest([plan.CAMPAIGN_ID, cfg['server'], cfg['run_id'], slot])
    fields = {'Experiment': cfg['experiment'], 'Server': cfg['server'],
        'Selection@Step': f'{label} · {step:,}', 'Dataset': cfg['sensor'], 'Bucket': 'ablations',
        'Approach': approach(cfg), 'Campaign_ID': plan.CAMPAIGN_ID, 'Run_ID': cfg['run_id'],
        'Case_ID': cfg['case_id'], 'Repeat': cfg['repeat'], 'Seed': cfg['seed'], 'Attempt': 1,
        'Role': 'Stage1 reconstruction',
        'Model / inputs': cfg['architecture'] + ' / ' + cfg.get('student_input', 'PAN+MS'),
        'Teacher / reference': cfg.get('donor_run_id') or '', 'Updates': step,
        'Lifetime_updates': step, 'Selected_step': step, 'Selector': selector, 'Status': status,
        'Readback': 'VERIFIED', 'Eval_scope': 'OFFICIAL_FULL20_NATIVE' if record else 'STATUS_ONLY',
        'Test_aware': False,
        'Result_ID': rid, 'U_peak_lr': cfg['lr'], 'A_peak_lr': cfg['lr_aligner'],
        'alpha': None, 'beta': None,
        'lambda_E': cfg.get('lambda_epsilon')}
    if record:
        for split in ('rr', 'fr'):
            fields.update({METRICS[k]: v for k, v in record[split]['metrics'].items()})
        fields.update(Checkpoint_SHA=record['checkpoint']['file_sha256'],
            Date=record['recorded_at'], JQM_variant=record['jqm_variant'])
        if 'inference_parameters' in record:
            fields['Params(M)'] = record['inference_parameters'] / 1e6
        fields['Eval(h)'] = record['elapsed_seconds'] / 3600
    note['reporting'] = {'policy': POLICY, 'decimal_places': 4,
        'selection_uses_unrounded_evidence': True,
        'raw_metrics': {k: fields.get(k, '') for k in set(METRICS.values())}}
    for key in set(METRICS.values()):
        if type(fields.get(key)) in (int, float):
            fields[key] = round(fields[key], 4)
    fields['Original description / notes'] = canonical_json(note)
    return serialize_row(fields), canonical_json(note)


def _stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


@lru_cache(maxsize=256)
def _training_timing(path, inode, size, mtime, step):
    seconds, first = 0., None
    with Path(path).open() as stream:
        for expected in range(1, step + 1):
            line = stream.readline()
            if not line.endswith('\n'):
                raise ValueError('Incomplete committed training-time evidence')
            row = json.loads(line)
            if row['step'] != expected or not math.isfinite(row['seconds']) or row['seconds'] < 0:
                raise ValueError('Non-contiguous or invalid committed training time')
            seconds += row['seconds']
            if first is None:
                first = _stamp(row['at']) - timedelta(seconds=row['seconds'])
    return seconds, first


def timing(directory, step, records, stage):
    path = Path(directory) / 'train.jsonl'
    if not path.exists():
        return {'status':'MISSING_TRAIN_TRACE','scope':'No training time inferred from step counts'}
    stat = path.stat()
    seconds, first = _training_timing(str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, step)
    selected = {e['step']: e for e in records if e['step'] <= step}
    expected = [s for s in plan.PLAN['evaluation_steps'] if s <= step]
    if sorted(selected) != expected:
        return {'status': 'INCOMPLETE_EVALUATION_TIMING', 'Train(h)': seconds / 3600,
                'scope': f'{stage} committed optimizer steps 1..selected_step; evaluation timing incomplete'}
    evaluation_seconds = sum(e['elapsed_seconds'] for e in selected.values())
    wall_seconds = (_stamp(selected[step]['recorded_at']) - first).total_seconds()
    if not math.isfinite(evaluation_seconds) or evaluation_seconds < 0 or wall_seconds < 0:
        raise ValueError('Invalid cumulative evaluation/wall timing')
    return {'status': 'VERIFIED', 'Train(h)': seconds / 3600,
        'Eval(h)': evaluation_seconds / 3600, 'Wall(h)': wall_seconds / 3600,
        'optimizer_steps': step, 'evaluation_count': len(selected),
        'scope': f'{stage} optimizer steps 1..selected_step, excludes eval/checkpoint/calibration/Sheet; '
                 'Eval sums recorded evaluation durations; Wall includes IO/downtime; other stages excluded'}


def shared_cost(cfg):
    # Different band counts/widths and fresh architecture: never copy an unmatched profile.
    return None


def project(root, server):
    root = Path(root); states = {}
    path = root / 'campaign_state.json'
    if path.exists():
        states = {r['run_id']: r for r in json.loads(path.read_text())['runs']}
    rows, notes, proof = [], [], {}
    configs = {c['run_id']: c for c in plan.queue(server)}
    profiles = {}
    for cfg in configs.values():
        directory = root / 'runs' / cfg['run_id']
        files = sorted(directory.glob('evaluation_*.json'))
        records = []
        for p in files:
            record = json.loads(p.read_text()); verify_record(record, cfg)
            artifact(record['checkpoint'])
            if (record['checkpoint']['model_sha256'] != record['model_sha256']
                    or record['checkpoint']['step'] != record['step']):
                raise ValueError('Evaluation does not match its retained checkpoint')
            records.append(dict(record, evidence_path=str(p), evidence_sha256=file_sha(p)))
        for record in records:
            proof[(cfg['run_id'], record['step'])] = record
        status = states.get(cfg['run_id'], {}).get('status', 'QUEUED')
        selected = choose(records,cfg['stage'],cfg)
        for record, selector, slot in selected:
            row, note_text = make_row(cfg, status, record, selector, slot)
            note = json.loads(note_text)
            if record:
                origin = directory
                times = timing(origin, record['step'], records, cfg['stage'])
                note['reporting']['timing'] = times
                for key in ('Train(h)', 'Eval(h)', 'Wall(h)'):
                    if key in times:
                        row[INDEX[key]] = times[key]
                if cfg['stage'] not in profiles:
                    profiles[cfg['stage']] = shared_cost(cfg)
                cost = profiles[cfg['stage']]
                if cost:
                    fields = cost['measurement']['sheet_cost']
                    if fields['Params(M)'] != row[INDEX['Params(M)']]:
                        raise ValueError('Measured W112 deployment parameter count differs')
                    for key, number in fields.items():
                        row[INDEX[key]] = number
                    row[INDEX['Cost_scope']] += '; shared historical identical graph; ' + times['scope']
                    note['reporting'].update(cost_basis='SHARED_HISTORICAL_IDENTICAL_INFERENCE_GRAPH',
                        cost_profile=cost['profile_path'], cost_profile_sha256=cost['profile_sha256'],
                        cost_profile_file_sha256=cost['profile_file_sha256'],
                        cost_checkpoint=cost['checkpoint'], cost_environment=cost['measurement']['environment'],
                        cost_rr=cost['measurement']['observations']['rr']['sheet_fields'],
                        cost_fr=cost['measurement']['observations']['fr']['sheet_fields'])
                else:
                    note['reporting']['cost_status'] = 'NO_VERIFIED_MATCHING_PROFILE'
                    row[INDEX['Cost_scope']] = times['scope'] + '; inference cost not measured for this graph'
            for i in range(3, 25):
                if type(row[i]) in (int, float):
                    row[i] = round(row[i], 4)
            row[INDEX['Original description / notes']] = canonical_json(note)
            if len(row[INDEX['Original description / notes']]) >= 49000:
                raise ValueError('Publication provenance exceeds safe Google cell size')
            rows.append(row); notes.append(row[INDEX['Original description / notes']])
    return {'policy': POLICY, 'destination': destination(server), 'rows': rows, 'notes': notes,
            'conflicts': [], 'configs': configs, 'proof': proof, 'view_hash': plan.digest(rows)}


def verify_owned(row, value, server):
    cfg = value['configs'].get(row[INDEX['Run_ID']])
    if cfg is None or cfg['server'] != server or row[INDEX['Dataset']] != cfg['sensor']:
        raise ValueError('Unrecognized row in this campaign; never erase it')
    note = json.loads(row[INDEX['Original description / notes']]); pub = note['publication']
    if note['configuration'] != cfg or pub['policy'] != POLICY:
        raise ValueError('Owned row has modified configuration/provenance')
    step = row[INDEX['Selected_step']]
    record = value['proof'].get((cfg['run_id'], step)) if step else None
    if step and record is None:
        raise ValueError('Owned row lacks immutable evaluation evidence')
    expected, expected_note = make_row(cfg, row[INDEX['Status']], record, pub['selector'], pub['slot'])
    expected_note = json.loads(expected_note)
    for key in ('donor_selection', 'reuse', 'stage1_re_evaluation', 'evaluation_sha256', 'model_state_sha256', 'artifact'):
        if note.get(key) != expected_note.get(key):
            raise ValueError('Owned row evidence mismatch: ' + key)
    if (note['reporting'].get('decimal_places') != 4
            or note['reporting'].get('policy') != POLICY
            or note['reporting'].get('raw_metrics') != expected_note['reporting']['raw_metrics']):
        raise ValueError('Owned rounded metrics differ from original precision evidence')
    for key in ('Result_ID', 'Experiment', 'Case_ID', 'Repeat', 'Seed', 'Selected_step',
                'Checkpoint_SHA', *set(METRICS.values())):
        if row[INDEX[key]] != expected[INDEX[key]]:
            raise ValueError('Owned row evidence mismatch: ' + key)


def row_requests(snapshot, value, server):
    owned = {}
    for number, row in enumerate(snapshot['rows'], 5):
        if row[INDEX['Campaign_ID']] != plan.CAMPAIGN_ID or row[INDEX['Server']] != server:
            continue
        verify_owned(row, value, server)
        rid = row[INDEX['Result_ID']]
        if rid in owned:
            raise ValueError('Duplicate owned Result_ID')
        owned[rid] = (number, row, snapshot['notes'][number - 5])
    wanted = {r[INDEX['Result_ID']]: (r, n) for r, n in zip(value['rows'], value['notes'])}
    if len(wanted) != len(value['rows']):
        raise ValueError('Duplicate desired Result_ID')
    if set(owned) - set(wanted):
        raise ValueError('Evidence view unexpectedly contracted; never blank historical rows')
    def cells(row, note):
        return {'values': [{'userEnteredValue': _typed(v), 'note': note if i == 0 else ''}
                           for i, v in enumerate(row)]}
    requests, additions = [], []
    for rid, (row, note) in wanted.items():
        if rid not in owned:
            additions.append(cells(row, note)); continue
        number, old, old_note = owned[rid]
        if not _matrix_equal([old], [row]) or old_note != note:
            requests.append({'updateCells': {'range': {'sheetId': destination(server)['sheet_id'],
                'startRowIndex': number, 'endRowIndex': number + 1,
                'startColumnIndex': 0, 'endColumnIndex': 64}, 'rows': [cells(row, note)],
                'fields': 'userEnteredValue,note'}})
    if additions:
        requests.append({'appendCells': {'sheetId': destination(server)['sheet_id'], 'rows': additions,
                                        'fields': 'userEnteredValue,note'}})
    return requests


def publish(root, server, *, apply=False, credential=None):
    from sheet_publish_v2.transport import GoogleSheets
    from .transport import reconcile_rows
    root = Path(root); adapter = GoogleSheets(credential or plan.ROOT / 'gspread/account.json')
    value = project(root, server)
    receipt = reconcile_rows(adapter, value, root / 'publication', server, apply=apply)
    if apply and receipt['sheet_rows']:
        requests = [{'repeatCell': {'range': {'sheetId': destination(server)['sheet_id'],
            'startRowIndex': n - 1, 'endRowIndex': n, 'startColumnIndex': 3, 'endColumnIndex': 25},
            'cell': {'userEnteredFormat': {'numberFormat': {'type': 'NUMBER', 'pattern': '0.0000'}}},
            'fields': 'userEnteredFormat.numberFormat'}} for n in receipt['sheet_rows']]
        adapter._request('POST', adapter.BASE + destination(server)['spreadsheet_id'] + ':batchUpdate',
                         json={'requests': requests})
    atomic_json(root / 'publication/last_poll.json', receipt)
    return receipt
