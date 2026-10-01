"""Target-only RAW observation writer with durable outbox. No legacy uploader imports.

Only existing 배포용 모델/1198707876 rows >=6, A:BI and BK:BL. BJ is user
owned. Headers, formulas, other tabs and historical rows are never repaired.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from .common import CAMPAIGN, atomic_json, append_jsonl, canonical_sha, timestamp

SPREADSHEET_ID = '1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0'
SHEET_ID = 1198707876
SHEET_TITLE = '배포용 모델'
HEADERS = ('Experiment', 'Server', 'Selection@Step', 'HQNR↑', 'D_s↓', 'D_lambda↓',
           'JQM↑', 'ERGAS↓', 'SCC↑', 'SAM↓', 'PSNR↑', 'SSIM↑', 'Q4/Q8↑', 'RMSE↓',
           'CC↑', 'Infer(ms)', 'Mem(MB)', 'Params(M)', 'FLOPs(G)', 'Train(h)',
           'Eval(h)', 'Wall(h)', 'HQNR(V64)↑', 'Signed D_s', 'Positive D_s fraction',
           'Dataset', 'Bucket', 'Approach', 'Campaign_ID', 'Run_ID', 'Case_ID',
           'Repeat', 'Seed', 'Attempt', 'Role', 'Model / inputs', 'Teacher / reference',
           'Parent_run', 'Updates', 'Lifetime_updates', 'Selected_step', 'Checkpoint_SHA',
           'Selector', 'q_ref', 'tau_R', 'alpha', 'beta', 'lambda_E', 'U_peak_lr',
           'A_peak_lr', 'Status', 'Readback', 'Eval_scope', 'Test_aware', 'JQM_variant',
           'Cost_scope', 'Source_sheet', 'Source_row', 'Source_URL', 'Result_ID',
           'Original description / notes', 'Review', 'Sort_case', 'Date')
CASES = {'C00': ('Shared-Base', 0), 'C11': ('Shared-Wide', 11), 'C01': ('Shared-HalfData', 1),
         'S01': ('Single-WV3', 101), 'S02': ('Single-GF2', 102), 'S03': ('Single-QB', 103)}


class SheetConflict(ValueError):
    pass


def inspect_local_writers(process_output=None, *, own_pid=None):
    """Known local target writers only; not a distributed-lock/absence guarantee.

    Ignore our own ancestry (shell/launcher/controller). Never stop or signal a
    foreign process. Generic research uploaders without this target stay untouched.
    """
    own_pid = os.getpid() if own_pid is None else own_pid
    if process_output is None:
        process = subprocess.run(['ps', '-eo', 'pid=,ppid=,args='], text=True,
                                 capture_output=True, timeout=10, check=False)
        if process.returncode:
            raise SheetConflict('WRITER_CONFLICT: local process inventory unavailable')
        process_output = process.stdout
    records = {}
    for line in process_output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            records[int(parts[0])] = (int(parts[1]), parts[2])
    excluded = set()
    current = own_pid
    while current and current not in excluded:
        excluded.add(current)
        current = records.get(current, (0, ''))[0]
    foreign = []
    for pid, (parent, command) in records.items():
        if pid in excluded:
            continue
        lower = command.lower()
        executable = Path(command.split(None, 1)[0]).name.lower()
        # Shell argv may contain a whole heredoc with dormant code, not a writer.
        if executable in ('bash', 'sh', 'zsh', 'dash', 'timeout', 'ps'):
            continue
        own_project_writer = 'pan_shared.cli' in lower and re.search(r'\b(?:run|sync-sheet)\b', lower)
        explicit_target = str(SHEET_ID) in command or SHEET_TITLE in command
        uploader = re.search(r'upload|sync[-_]sheet|gspread|sheets', lower)
        if own_project_writer or (explicit_target and uploader):
            foreign.append({'pid': pid, 'parent_pid': parent, 'command': command})
    if foreign:
        # Do not include argv (which could contain credentials) in exceptions/logs.
        raise SheetConflict('WRITER_CONFLICT: existing local deployment writer PIDs '+','.join(str(v['pid']) for v in foreign))
    return {'status': 'PASS', 'known_foreign_writer_count': 0,
            'scope': 'known local process command lines; remote/unknown scheduled writers require operator serialization',
            'excluded_own_ancestor_pids': sorted(excluded), 'processes_stopped': 0}


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value):
        raise ValueError('Full lowercase SHA256 required')
    return value


def result_id(observation):
    return canonical_sha([CAMPAIGN, observation['run_id'], _sha(observation['checkpoint_sha256']),
                          observation['sensor'], observation['selector_scope'], _sha(observation['evaluator_sha256'])])


def _number(value):
    if value is None:
        return ''
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Measured cells require finite numbers, not strings/bools')
    return value


def build_row(observation):
    """Map one *real completed* observation; no fake/Q00/RUNNING metric rows."""
    o = observation
    if o.get('campaign_id', CAMPAIGN) != CAMPAIGN or o.get('server', 's2') != 's2':
        raise ValueError('Wrong campaign/server')
    case, mode, sensor = o['case_id'], o['mode'], o['sensor']
    if case not in CASES or mode not in ('SHARED', 'SINGLE') or sensor not in ('WV3', 'GF2', 'QB'):
        raise ValueError('Unregistered observation')
    if mode != ('SHARED' if case.startswith('C') else 'SINGLE'):
        raise ValueError('Case/mode mismatch')
    if mode == 'SINGLE' and sensor != {'S01': 'WV3', 'S02': 'GF2', 'S03': 'QB'}[case]:
        raise ValueError('Single sensor mismatch')
    if o.get('status', 'EVAL_COMPLETE') != 'EVAL_COMPLETE' or o['rr'].get('n_scenes') != 20 or o['fr'].get('n_scenes') != 20:
        raise ValueError('Only complete RR20/FR20 observations may be uploaded')
    if not re.fullmatch(r'EXACT(?:_\w+)?|BEST_(?:JOINT|SENSOR)_VAL_TO_B[0-9]{4,}', o['selector_scope']):
        raise ValueError('Selector must include exact or bounded validation search scope')
    if type(o['repeat']) is not int or o['repeat'] not in (1, 2, 3) or o['seed'] != 271000 + o['repeat']:
        raise ValueError('Unregistered repeat/seed')
    expected_width, expected_fraction = (128 if case == 'C11' else 104), (.5 if case == 'C01' else 1.)
    expected_sensors = ['WV3', 'GF2', 'QB'] if mode == 'SHARED' else [sensor]
    if o['width'] != expected_width or o['train_fraction'] != expected_fraction or o['train_sensors'] != expected_sensors:
        raise ValueError('Case architecture/data semantics differ from frozen registry')
    if type(o['attempt']) is not int or o['attempt'] < 1:
        raise ValueError('Attempt must be a positive integer')
    expected_run = f"PDSP_S2_R{o['repeat']:02d}_{case}_W{o['width']}_D122_F{int(o['train_fraction']*100):03d}_PLH_SE{o['seed']}_A{o['attempt']:02d}"
    if o['run_id'] != expected_run:
        raise ValueError('Run_ID differs from registered case/repeat/attempt')
    if (type(o['completed_step']) is not int or type(o['selected_step']) is not int
            or not 0 < o['selected_step'] <= o['completed_step']):
        raise ValueError('Selected step must be a completed positive update')
    if not re.search(r'[+-]\d\d:\d\d$|Z$', o['recorded_at']):
        raise ValueError('ISO8601 timezone required')
    rr, fr, cost = o['rr'], o['fr'], o.get('cost', {})
    row = [''] * 64
    row[0:3] = [f"{case} · {CASES[case][0]} · W{o['width']} D122 · F{int(o['train_fraction']*100):03d} · R{o['repeat']:02d} · {sensor}",
                's2', f"{o['selector_scope']}@{o['selected_step']}"]
    row[3:15] = [_number(fr.get(k)) for k in ('hqnr', 'd_s', 'd_lambda', 'jqm')] + [
        _number(rr.get(k)) for k in ('ergas', 'scc', 'sam', 'psnr', 'ssim', 'q8' if sensor == 'WV3' else 'q4', 'rmse', 'cc')]
    if any(row[i] == '' for i in (3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14)):
        raise ValueError('Official complete observations may not omit required measured metrics')
    row[15:22] = [_number(cost.get(k)) for k in ('infer_ms', 'mem_mb', 'params_m', 'flops_g', 'train_hours', 'eval_hours', 'wall_hours')]
    row[22:25] = [_number(fr.get(k)) for k in ('hqnr_v64', 'signed_ds', 'positive_fraction')]
    row[25:35] = [sensor, 'Deployment', 'Shared PLH U-Net' if mode == 'SHARED' else 'Single PLH U-Net',
                  CAMPAIGN, o['run_id'], case, o['repeat'], o['seed'], o['attempt'], mode]
    row[35:43] = [f"PLH; W{o['width']} D122; train={'+'.join(o['train_sensors'])}; fraction={o['train_fraction']}; noA",
                  'NONE', '', o['completed_step'], o['completed_step'], o['selected_step'],
                  _sha(o['checkpoint_sha256']), o['selector_scope']]
    row[45:55] = [0, 0, 0, 1e-4, 0, 'EVAL_COMPLETE', 'PENDING',
                  f"RR20+FR20; protocol={o['protocol_id']}; sensor={sensor}", False,
                  fr.get('jqm_variant', '') if fr.get('jqm') is not None else '']
    row[55] = cost.get('scope', '')
    url = o.get('source_url', '')
    if url and not url.startswith(('https://', 'http://')):
        raise ValueError('Server paths are not accessible source URLs')
    row[58:64] = [url, result_id(o), json.dumps(o['notes'], ensure_ascii=False, sort_keys=True,
                                              separators=(',', ':'), allow_nan=False), '', CASES[case][1], o['recorded_at']]
    if len(row[60]) > 45000:
        raise ValueError('Notes exceed safe cell limit; preserve detailed provenance locally')
    return row


def cell_value(cell):
    value = cell.get('effectiveValue', cell.get('userEnteredValue', {}))
    for key in ('stringValue', 'numberValue', 'boolValue'):
        if key in value:
            return value[key]
    return ''


def occupied(cell):
    # Notes/validation/merges reserve cells even when display value is blank.
    return any(cell.get(k) for k in ('userEnteredValue', 'effectiveValue', 'note', 'dataValidation'))


def _equal(a, b):
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def _row_equal(actual, expected, ignore_readback=False):
    skip = {61, 51} if ignore_readback else {61}
    return len(actual) == 64 and all(_equal(a, b) for i, (a, b) in enumerate(zip(actual, expected)) if i not in skip)


class GoogleSheetAdapter:
    """Lazily authenticated API; authority limited to a single pinned existing tab."""
    def __init__(self, credentials_path, spreadsheet_id=SPREADSHEET_ID, sheet_id=SHEET_ID):
        if spreadsheet_id != SPREADSHEET_ID or sheet_id != SHEET_ID:
            raise SheetConflict('BLOCKED_SHEET_SCHEMA: target mismatch')
        import gspread
        project = Path(__file__).resolve().parents[2]
        self.writer_lock_path = project/'work_dir'/'.locks'/f'{SPREADSHEET_ID}_{SHEET_ID}.writer.lock'
        client = gspread.service_account(filename=str(credentials_path))
        client.set_timeout((10, 30))
        self.book = client.open_by_key(spreadsheet_id)

    def validate_schema(self):
        return inspect_target(self)

    def inspect(self):
        writer_inspection = inspect_local_writers()
        metadata = self.book.fetch_sheet_metadata(params={'fields': 'spreadsheetId,sheets(properties,merges,protectedRanges)'})
        targets = [s for s in metadata['sheets'] if s['properties']['sheetId'] == SHEET_ID]
        if len(targets) != 1 or targets[0]['properties']['title'] != SHEET_TITLE:
            raise SheetConflict('BLOCKED_SHEET_SCHEMA: target missing/renamed')
        target = targets[0]
        if target['properties']['gridProperties']['columnCount'] != 64:
            raise SheetConflict('BLOCKED_SHEET_SCHEMA: target is not 64 columns')
        nrows = target['properties']['gridProperties']['rowCount']
        data = self.book.fetch_sheet_metadata(params={
            'ranges': f"'{SHEET_TITLE}'!A1:BL{nrows}", 'includeGridData': True,
            'fields': 'sheets(data(startRow,startColumn,rowData(values(userEnteredValue,effectiveValue,note,dataValidation))))'})
        rows = {}
        for sheet in data.get('sheets', []):
            for grid in sheet.get('data', []):
                for offset, row in enumerate(grid.get('rowData', [])):
                    cells = row.get('values', [])
                    rows[grid.get('startRow', 0)+offset+1] = cells + [{}] * (64-len(cells))
        return {'sheet_id': SHEET_ID, 'title': SHEET_TITLE, 'row_count': nrows, 'rows': rows,
                'merges': target.get('merges', []), 'protected_ranges': target.get('protectedRanges', []),
                'writer_inspection': writer_inspection}

    def extend(self, required_rows):
        state = self.inspect()
        if required_rows > state['row_count']:
            self.book.batch_update({'requests': [{'appendDimension': {'sheetId': SHEET_ID, 'dimension': 'ROWS',
                                                                      'length': required_rows-state['row_count']}}]})

    def write_row(self, row_number, row):
        if row_number < 6 or len(row) != 64:
            raise SheetConflict('BLOCKED_SHEET_SCHEMA: invalid destination')
        self.book.values_batch_update({'valueInputOption': 'RAW', 'data': [
            {'range': f"'{SHEET_TITLE}'!A{row_number}:BI{row_number}", 'values': [row[:61]]},
            {'range': f"'{SHEET_TITLE}'!BK{row_number}:BL{row_number}", 'values': [row[62:]]}]})

    def mark_verified(self, row_number):
        if row_number < 6:
            raise SheetConflict('BLOCKED_SHEET_SCHEMA')
        self.book.values_batch_update({'valueInputOption': 'RAW', 'data': [
            {'range': f"'{SHEET_TITLE}'!AZ{row_number}", 'values': [['VERIFIED']]}]})


def inspect_target(adapter):
    """Read-only preflight gate. No outbox, locks, writes, creation, or resize."""
    snapshot = adapter.inspect()
    validator = object.__new__(SheetsUploader)
    ids, last = validator._validate(snapshot)
    return {'status': 'PASS', 'target_sheet_id': SHEET_ID, 'target_title': SHEET_TITLE,
            'header_sha256': canonical_sha(list(HEADERS)), 'header_control_sha256': validator._header_signature(snapshot),
            'row_count': snapshot['row_count'], 'last_used_row': last, 'existing_result_ids': len(ids),
            'snapshot_sha256': canonical_sha(snapshot), 'writer_inspection': snapshot.get('writer_inspection'),
            'new_sheets': 0, 'writes': 0}


class SheetsUploader:
    def __init__(self, adapter, outbox, *, server='s2'):
        if server != 's2':
            raise SheetConflict('WRITER_CONFLICT: s2 writer only')
        self.adapter, self.outbox = adapter, Path(outbox)
        self.outbox.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _lock(self, include_target=False):
        paths = []
        if include_target and getattr(self.adapter, 'writer_lock_path', None):
            paths.append(Path(self.adapter.writer_lock_path))
        paths.append(self.outbox/'deployment-sheet.writer.lock')
        with contextlib.ExitStack() as stack:
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context(path.open('a+'))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise SheetConflict('WRITER_CONFLICT: another deployment/outbox writer holds lock') from error
            yield

    def enqueue(self, observation):
        row = build_row(observation)
        with self._lock():
            self._recover_outbox()
            path = self.outbox / (row[59] + '.json')
            if path.exists():
                existing = json.loads(path.read_text())
                if existing['row_sha256'] != canonical_sha(row):
                    raise SheetConflict('BLOCKED_SHEET_SCHEMA: immutable Result_ID payload changed')
                return path
            # WAL source of truth is fsync'ed before publishing the mutable delivery envelope.
            append_jsonl(self.outbox/'observations.jsonl', {'schema': 'PANDEP_SHEET_OBSERVATION_v1',
                'result_id': row[59], 'observation': observation, 'row': row,
                'row_sha256': canonical_sha(row), 'queued_at': timestamp()})
            atomic_json(path, self._envelope(row))
            return path

    @staticmethod
    def _envelope(row):
        envelope = {'schema': 'PANDEP_SHEET_OUTBOX_v1', 'row': row, 'row_sha256': canonical_sha(row),
                    'attempts': 0, 'next_retry_at': 0, 'status': 'PENDING', 'result_id': row[59]}
        return envelope

    def _recover_outbox(self):
        journal = self.outbox/'observations.jsonl'
        if not journal.exists():
            return
        records = {}
        with journal.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                if (record.get('schema') != 'PANDEP_SHEET_OBSERVATION_v1'
                        or canonical_sha(record['row']) != record['row_sha256']
                        or record['row'] != build_row(record['observation'])):
                    raise SheetConflict('BLOCKED_IDENTITY: immutable observation journal corrupt')
                key = record['result_id']
                if key != record['row'][59] or (key in records and records[key]['row_sha256'] != record['row_sha256']):
                    raise SheetConflict('BLOCKED_IDENTITY: conflicting observation journal Result_ID')
                records[key] = record
        for key, record in records.items():
            path = self.outbox/(key+'.json')
            if not path.exists():
                atomic_json(path, self._envelope(record['row']))
            elif json.loads(path.read_text())['row_sha256'] != record['row_sha256']:
                raise SheetConflict('BLOCKED_IDENTITY: delivery envelope conflicts with source journal')

    def _validate(self, state):
        if state['sheet_id'] != SHEET_ID or state['title'] != SHEET_TITLE:
            raise SheetConflict('BLOCKED_SHEET_SCHEMA: wrong target')
        rows = state['rows']
        if [cell_value(c) for c in rows.get(5, [])] != list(HEADERS):
            raise SheetConflict('BLOCKED_SHEET_SCHEMA: header names/order mismatch')
        ids = {}
        last = 5
        if rows.get(5, [{}])[0].get('userEnteredValue', {}).get('formulaValue') != "=ARRAYFORMULA('_records'!$A$1:$BL$1)":
            raise SheetConflict('BLOCKED_SHEET_SCHEMA: existing header formula changed')
        for number, cells in rows.items():
            if number < 6:
                continue
            if any('formulaValue' in c.get('userEnteredValue', {}) or 'errorValue' in c.get('effectiveValue', {}) for c in cells):
                raise SheetConflict('BLOCKED_SHEET_SCHEMA: formula/spill/error in data region')
            if any(occupied(c) for c in cells):
                last = max(last, number)
            key = cell_value(cells[59])
            if key:
                if key in ids:
                    raise SheetConflict('BLOCKED_SHEET_SCHEMA: duplicate existing Result_ID')
                ids[key] = number
        for merge in state.get('merges', []):
            if merge.get('endRowIndex', 0) > 5:
                last = max(last, merge['endRowIndex'])
        return ids, last

    @staticmethod
    def _header_signature(state):
        return canonical_sha({str(i): [{k: c[k] for k in ('userEnteredValue', 'note', 'dataValidation') if k in c}
                                      for c in state['rows'].get(i, [])] for i in range(1, 6)})

    def upload_row(self, row):
        with self._lock(include_target=True):
            return self._upload_row(row)

    def _upload_row(self, row):
        before = self.adapter.inspect()
        ids, last = self._validate(before)
        number = ids.get(row[59], last+1)
        old = before['rows'].get(number, [{}]*64)
        existing = number in ids.values()
        if existing and not _row_equal([cell_value(c) for c in old], row, ignore_readback=True):
            raise SheetConflict('BLOCKED_SHEET_SCHEMA: immutable existing Result_ID payload differs')
        if not existing and any(occupied(c) for c in old):
            raise SheetConflict('BLOCKED_SHEET_SCHEMA: destination not empty')
        for protection in before.get('protected_ranges', []):
            r = protection.get('range', {})
            if r.get('startRowIndex', 0) <= number-1 < r.get('endRowIndex', before['row_count']):
                raise SheetConflict('BLOCKED_SHEET_SCHEMA: destination is protected')
        if number > before['row_count']:
            self.adapter.extend(number)
        # Re-read immediately before mutation, including far-right keys and spill outputs.
        current = self.adapter.inspect()
        current_ids, current_last = self._validate(current)
        if self._header_signature(current) != self._header_signature(before):
            raise SheetConflict('WRITER_CONFLICT: header changed during transaction')
        if current_ids != ids or (not existing and current_last != last):
            raise SheetConflict('WRITER_CONFLICT: another writer changed destination/key set')
        if current['rows'].get(number, [{}]*64) != old:
            raise SheetConflict('WRITER_CONFLICT: destination changed during transaction')
        already_verified = existing and cell_value(old[51]) == 'VERIFIED'
        if not existing:
            self.adapter.write_row(number, row)
        actual = self.adapter.inspect()
        self._validate(actual)
        if not _row_equal([cell_value(c) for c in actual['rows'].get(number, [])], row, ignore_readback=True):
            raise SheetConflict('READBACK_FAILED: observation values differ')
        if self._header_signature(actual) != self._header_signature(before):
            raise SheetConflict('WRITER_CONFLICT: headers changed')
        if cell_value(actual['rows'][number][61]) != cell_value(old[61]):
            raise SheetConflict('WRITER_CONFLICT: manual review changed concurrently')
        if not already_verified:
            self.adapter.mark_verified(number)
        final = self.adapter.inspect()
        self._validate(final)
        expected = list(row); expected[51] = 'VERIFIED'
        if not _row_equal([cell_value(c) for c in final['rows'].get(number, [])], expected):
            raise SheetConflict('READBACK_FAILED: verification marker did not persist')
        return {'status': 'VERIFIED', 'row_number': number, 'result_id': row[59],
                'duplicate_noop': already_verified, 'new_sheets': 0, 'target_sheet_id': SHEET_ID}

    def sync(self, now=None):
        """Failures remain durable and never propagate into the training process."""
        now = time.time() if now is None else now
        results = []
        with self._lock(include_target=True):
            self._recover_outbox()
            for path in sorted(self.outbox.glob('*.json')):
                envelope = json.loads(path.read_text())
                if envelope.get('schema') != 'PANDEP_SHEET_OUTBOX_v1':
                    continue
                if envelope['status'] == 'VERIFIED' or envelope['next_retry_at'] > now:
                    continue
                try:
                    if canonical_sha(envelope['row']) != envelope['row_sha256']:
                        raise SheetConflict('BLOCKED_IDENTITY: outbox payload changed')
                    receipt = self._upload_row(envelope['row'])
                    envelope.update(status='VERIFIED', receipt=receipt, verified_at=timestamp())
                except Exception as error:
                    envelope['attempts'] += 1
                    envelope.update(status='BLOCKED' if isinstance(error, SheetConflict) else 'RETRY',
                                    error=f'{type(error).__name__}: {error}',
                                    next_retry_at=now+min(3600, 2**min(envelope['attempts'], 12)))
                append_jsonl(self.outbox/'delivery_events.jsonl', {
                    'result_id': envelope['result_id'], 'status': envelope['status'], 'attempts': envelope['attempts'],
                    'recorded_at': timestamp(), 'receipt': envelope.get('receipt'), 'error': envelope.get('error')})
                atomic_json(path, envelope)
                results.append({'result_id': envelope['result_id'], 'status': envelope['status'],
                                'error': envelope.get('error'), 'receipt': envelope.get('receipt')})
        return results
