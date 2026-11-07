"""WV3 coefficient ablations: immutable evidence, fixed endpoints and own rows."""
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

POLICY = 'WV3_HYPER_FIXED_AND_REPRESENTATIVE_4DP_V1'
R = SimpleNamespace(atomic_json=atomic_json, utc_now=utc_now)


def destination():
    return {'spreadsheet_id': '1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE',
            'sheet_title': 'WV3-ablations', 'sheet_id': 261001010}


@lru_cache(maxsize=4096)
def _verified_file(path, inode, size, mtime):
    return file_sha(path)


def artifact(meta):
    path = plan.local_path(meta['path']); stat = path.stat()
    if _verified_file(str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns) != meta['file_sha256']:
        raise ValueError('Published fresh checkpoint changed: ' + str(path))


def choose(records, stage):
    """One representative, all S2 passes and the predeclared endpoint, deduplicated.

    Slots count upwards per run, not by selected step. This allows a queued row
    to become a representative and passing rows to change order without blank
    rows or physical row deletion that might affect another server's results.
    """
    if stage not in ('stage1', 'stage2'):
        raise ValueError('Unknown ablation stage')
    by_step = {}
    for record in records:
        step = record['step']
        if step in by_step and by_step[step] != record:
            raise ValueError('Conflicting evaluations of the same checkpoint')
        by_step[step] = record
    records = [by_step[k] for k in sorted(by_step)]
    if not records:
        return []
    selectors = {}
    if stage == 'stage1':
        candidates = [e for e in records if 'validation' in e]
        if not candidates:
            raise ValueError('Stage1 evidence lacks validation selection metrics')
        rep = min(candidates, key=lambda e: (e['validation']['RRE50'],
            e['validation']['RRE90'], e['validation']['Val_ERGAS'], e['step']))
        selectors[rep['step']] = ['BEST_VAL_RRE']
        primary, fixed = 50000, 'FIXED50K'
    else:
        qualified = [e for e in records if e['fr']['metrics']['HQNR'] >= plan.HQNR_MIN]
        rep = (min(qualified, key=lambda e: (e['rr']['metrics']['ERGAS'],
                -e['fr']['metrics']['HQNR'], e['step'])) if qualified else
               min(records, key=lambda e: (-e['fr']['metrics']['HQNR'],
                e['rr']['metrics']['ERGAS'], e['step'])))
        selectors[rep['step']] = ['HQNR_QUALIFIED_MIN_ERGAS' if qualified else 'BEST_HQNR_FALLBACK']
        for e in records:
            if joint_pass(e):
                selectors.setdefault(e['step'], []).append('PASS_ALL_STEPS')
        primary, fixed = 29000, 'FIXED29K'
    if primary in by_step:
        selectors.setdefault(primary, []).append(fixed)
    order = [rep['step']] + sorted(s for s in selectors if s != rep['step'])
    return [(by_step[s], '|'.join(selectors[s]), f'ROW{i}') for i, s in enumerate(order)]


def joint_pass(record):
    return bool(record and record['fr']['metrics']['HQNR'] >= plan.HQNR_MIN
                and record['rr']['metrics']['ERGAS'] <= plan.ERGAS_MAX)


def approach(cfg):
    if cfg['stage'] == 'stage1':
        return (f"S1 RGB; lambda_spa={cfg['lambda_abs']:g},lambda_eps={cfg['lambda_epsilon']:g}; "
                f"R_eps=3; W112 D123 BASE; bicubic; A/Tlr={cfg['lr_aligner']:g}/{cfg['lr']:g}; C50")
    return (f"S2 PANc+delta+MS; alpha={cfg['alpha']:g},gamma={cfg['gamma']:g},beta=0; "
            f"scale=1; W112 D123 UBIL; Student L4/Teacher bicubic; lr={cfg['lr']:g}; C30; "
            f"donor=local RGB R{cfg['repeat']}")


def donor_note(cfg, record):
    receipt = record.get('donor_selection')
    if not isinstance(receipt, dict) or receipt.get('donor_run_id') != cfg['donor_run_id']:
        raise ValueError('Stage2 result lacks the registered local donor evidence')
    if receipt.get('schema') == 'WV3_MAIN_RGB_DONOR_SELECTION_V1':
        if receipt.get('receipt_sha256') != plan.digest({k: v for k, v in receipt.items()
                                                       if k != 'receipt_sha256'}):
            raise ValueError('Stage1 donor selection receipt digest mismatch')
    return {**{k: v for k, v in receipt.items() if k != 'candidates'},
            'candidate_count': len(receipt.get('candidates', [])),
            'full_receipt_in_evaluation': True}


def make_row(cfg, status, record=None, selector='STATUS', slot='ROW0'):
    step = 0 if record is None else record['step']
    roles = selector.split('|')
    representative = any(s in roles for s in ('BEST_VAL_RRE', 'HQNR_QUALIFIED_MIN_ERGAS',
                                               'BEST_HQNR_FALLBACK'))
    fixed = next((s for s in roles if s.startswith('FIXED')), None)
    passed = joint_pass(record)
    if 'BEST_VAL_RRE' in roles:
        label = '대표 · Val-RRE'
    elif 'HQNR_QUALIFIED_MIN_ERGAS' in roles:
        label = '대표 · ERGAS 최소 · ' + ('조건 충족' if passed else 'ERGAS 미달')
    elif 'BEST_HQNR_FALLBACK' in roles:
        label = '대표 · HQNR 최고 · HQNR 미달'
    elif 'PASS_ALL_STEPS' in roles:
        label = '조건 충족'
    else:
        label = fixed or status
    if fixed and fixed != label:
        label += ' · ' + fixed
    if record and record.get('reuse'):
        label += ' · 기존 결과 재사용'
    note = {'schema_version': 'PAN_FIXED64_V1', 'reset_epoch': '2026-10-13',
        'configuration': cfg,
        'publication': {'policy': POLICY, 'selector': selector, 'slot': slot,
            'joint_pass': passed, 'is_case_representative': representative,
            'is_fixed_step_comparison': bool(fixed),
            'targets': {'HQNR_min': plan.HQNR_MIN, 'ERGAS_max': plan.ERGAS_MAX}},
        'fixed_step_comparison': cfg['primary_step'], 'reference_repeats': 3,
        'evaluation_file': None if record is None else record['evidence_path'],
        'evaluation_sha256': None if record is None else record['evidence_sha256'],
        'model_state_sha256': None if record is None else record['model_sha256'],
        'artifact': None if record is None else record['checkpoint'],
        'reference_definition': {
            'L_spa': 'GT RGB arithmetic mean [4,2,1], Scharr',
            'alignment_input': 'Stored LMS all8',
            'validation_selector': 'RGB GT RRE50/RRE90, ValERGAS, earlier step',
            'student_input': 'corrected PAN, corrected PAN minus original PAN, interp23 I_lrms',
            'warp': 'Stage1/Teacher bicubic-border; Student Lanczos4-reflection',
            'fixed_step_selection': 'Predeclared; no test-best checkpoint substitution'}}
    if record:
        if 'test_rre_rgb' in record:
            note['test_rre_rgb'] = record['test_rre_rgb']
        if 'validation' in record:
            note['validation'] = {k: v for k, v in record['validation'].items()
                                  if not isinstance(v, (dict, list))}
        if cfg['stage'] == 'stage2':
            note['donor_selection'] = donor_note(cfg, record)
        if record.get('reuse'):
            note['reuse'] = record['reuse']
    rid = 'WV3HYP-' + plan.digest([plan.CAMPAIGN_ID, cfg['server'], cfg['run_id'], slot])
    fields = {'Experiment': cfg['experiment'], 'Server': cfg['server'],
        'Selection@Step': f'{label} · {step:,}', 'Dataset': 'WV3', 'Bucket': 'ablations',
        'Approach': approach(cfg), 'Campaign_ID': plan.CAMPAIGN_ID, 'Run_ID': cfg['run_id'],
        'Case_ID': cfg['case_id'], 'Repeat': cfg['repeat'], 'Seed': cfg['seed'], 'Attempt': 1,
        'Role': 'Stage1 reconstruction' if cfg['stage'] == 'stage1' else 'Student+FrozenAligner',
        'Model / inputs': cfg['architecture'] + ' / ' + cfg.get('student_input', 'PAN+MS'),
        'Teacher / reference': cfg.get('donor_run_id') or '', 'Updates': step,
        'Lifetime_updates': step, 'Selected_step': step, 'Selector': selector, 'Status': status,
        'Readback': 'VERIFIED', 'Eval_scope': 'OFFICIAL_FULL20_NATIVE' if record else 'STATUS_ONLY',
        'Test_aware': bool(record and (representative or 'PASS_ALL_STEPS' in roles)),
        'Result_ID': rid, 'U_peak_lr': cfg['lr'], 'A_peak_lr': cfg['lr_aligner'],
        'alpha': cfg.get('alpha'), 'beta': 0 if cfg['stage'] == 'stage2' else None,
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
        return {'status': 'MISSING_TRAIN_TRACE', 'scope': 'No training time inferred from step counts'}
    stat = path.stat()
    seconds, first = _training_timing(str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, step)
    selected = {e['step']: e for e in records if e['step'] <= step}
    expected = list(range(1000, step + 1, 1000))
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
    """Read a measured local W112 deployment profile; never import reporting patches."""
    if cfg['stage'] != 'stage2':
        return None
    main = plan.ROOT / 'work_dir/panda_wv3_main_rgb/PANDA_WV3_MAIN_RGB_S1_20261103_v1/s1/WV3'
    path = main / 'reporting/cost.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (value.get('schema') != 'WV3_MAIN_RGB_DEPLOYMENT_COST_V1'
            or value.get('profile_sha256') != plan.digest({k: v for k, v in value.items()
                                                           if k != 'profile_sha256'})
            or value['measurement'].get('status') != 'COMPLETE'):
        raise ValueError('Historical W112 cost profile is inconsistent')
    keys = ('dataset', 'architecture', 'student_input', 'warp', 'scale', 'upsampler',
            'precision', 'source_registry_sha256', 'inference_tta')
    if {k: value['graph'].get(k) for k in keys} != {k: cfg[k] for k in keys}:
        raise ValueError('Historical cost graph differs from this W112 deployment')
    release = json.loads((plan.ROOT / 'config/panda_wv3_main_rgb/release.json').read_text())
    if value['graph'].get('source_release') != release['source_sha256']:
        raise ValueError('Historical measured source release differs')
    evidence = main / 'runs' / value['run_id'] / f"evaluation_{value['step']:07d}.json"
    if file_sha(evidence) != value['evaluation_sha256']:
        raise ValueError('Measured W112 profile has lost its original evaluation evidence')
    return dict(value, profile_path=str(path), profile_file_sha256=file_sha(path))


def project(root, server):
    from .reuse import load_reused_records
    root = Path(root); states = {}
    path = root / 'campaign_state.json'
    if path.exists():
        states = {r['run_id']: r for r in json.loads(path.read_text())['runs']}
    rows, notes, proof = [], [], {}
    configs = {c['run_id']: c for c in plan.queue(server)}
    profiles = {}
    for cfg in configs.values():
        directory = root / 'runs' / cfg['run_id']
        reused = (load_reused_records(cfg, directory)
                  if (directory / 'reuse_receipt.json').exists() else [])
        files = sorted(directory.glob('evaluation_*.json'))
        if reused and files:
            raise ValueError('A logical run cannot mix fresh and reused evaluation curves')
        records = list(reused or [])
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
        if reused:
            status = 'REUSED_COMPLETE'
        selected = choose(records, cfg['stage'])
        for record, selector, slot in selected or [(None, 'STATUS', 'ROW0')]:
            row, note_text = make_row(cfg, status, record, selector, slot)
            note = json.loads(note_text)
            if record:
                origin = plan.local_path(record['evidence_path']).parent if record.get('reuse') else directory
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
    return {'policy': POLICY, 'destination': destination(), 'rows': rows, 'notes': notes,
            'conflicts': [], 'configs': configs, 'proof': proof, 'view_hash': plan.digest(rows)}


def verify_owned(row, value, server):
    cfg = value['configs'].get(row[INDEX['Run_ID']])
    if cfg is None or cfg['server'] != server or row[INDEX['Dataset']] != 'WV3':
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
    for key in ('donor_selection', 'reuse', 'evaluation_sha256', 'model_state_sha256', 'artifact'):
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
            requests.append({'updateCells': {'range': {'sheetId': destination()['sheet_id'],
                'startRowIndex': number, 'endRowIndex': number + 1,
                'startColumnIndex': 0, 'endColumnIndex': 64}, 'rows': [cells(row, note)],
                'fields': 'userEnteredValue,note'}})
    if additions:
        requests.append({'appendCells': {'sheetId': destination()['sheet_id'], 'rows': additions,
                                        'fields': 'userEnteredValue,note'}})
    return requests


def publish(root, server, *, apply=False, credential=None):
    from sheet_publish_v2.transport import GoogleSheets
    from .transport import reconcile_rows
    root = Path(root); adapter = GoogleSheets(credential or plan.ROOT / 'gspread/account.json')
    value = project(root, server)
    receipt = reconcile_rows(adapter, value, root / 'publication', server, apply=apply)
    if apply and receipt['sheet_rows']:
        requests = [{'repeatCell': {'range': {'sheetId': destination()['sheet_id'],
            'startRowIndex': n - 1, 'endRowIndex': n, 'startColumnIndex': 3, 'endColumnIndex': 25},
            'cell': {'userEnteredFormat': {'numberFormat': {'type': 'NUMBER', 'pattern': '0.0000'}}},
            'fields': 'userEnteredFormat.numberFormat'}} for n in receipt['sheet_rows']]
        adapter._request('POST', adapter.BASE + destination()['spreadsheet_id'] + ':batchUpdate',
                         json={'requests': requests})
    atomic_json(root / 'publication/last_poll.json', receipt)
    return receipt
