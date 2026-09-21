"""CPU-only metadata supplement for verified, existing GF2 campaign rows.

No campaign package, evaluator, checkpoint loader or training entrypoint is
imported. Numerical results and their source identities remain historical facts;
this reporting release has a separate checksum and receipt.
"""
import ast
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import yaml

SCHEMA = 'GF2_METADATA_BACKFILL_v1'
CAMPAIGNS = {'g20': 'PANDA_GF2_G20_ALL5_20260920_v1', 'qg40': 'PANDA_QG40_20260920_v1'}
TABS = {s: 'GF2-' + ('s3(5090)' if s == 's3' else s) for s in ('s1', 's2', 's3', 's4', 's5')}
GRID = list(range(1010, 50000, 1010)) + [50000]
RR = {'ERGAS↓': 'ergas', 'SAM↓': 'sam', 'PSNR↑': 'psnr', 'SSIM↑': 'ssim',
      'SCC↑': 'scc', 'Q4↑': 'q4', 'RMSE↓': 'rmse', 'CC↑': 'cc'}
FR = {'HQNR↑': 'hqnr', 'D_lambda↓': 'd_lambda', 'D_s↓': 'd_s', 'JQM↑': 'jqm'}
METRICS = set(RR) | set(FR) | {'HQNR(raw)↑', 'Q2n↑', 'Q8↑', 'RR_VAL_SELECTED val ERGAS'}
PREFIXES = ('RAW_MAX', 'Target', 'Exact50K', 'RR_VAL_SELECTED', 'E_MIN_DIAG50')
COSTS = {'Params(M)', 'FLOPs(G)', 'Infer(ms)', 'Mem(MB)', 'Train(h)'}


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_evidence(wd, server):
    """Read persisted evidence only; never evaluate, import a model, or repair it."""
    wd = Path(wd).resolve()
    if server not in TABS or wd.parent.name != 'work_dir':
        raise ValueError('Expected a run directory under work_dir and server s1..s5')
    sources = {}

    def document(relative, yaml_file=False):
        path = wd / relative
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        value = yaml.safe_load(raw) if yaml_file else json.loads(raw)
        # Canonical serialization rejects nonfinite values, including YAML .nan.
        object_sha(value)
        return value

    cfg = document('meta/config.resolved.yaml', True)
    trainer = cfg.get('trainer')
    if trainer not in CAMPAIGNS or sum(k in cfg for k in CAMPAIGNS) != 1:
        raise ValueError('Only unambiguous G20/QG40 configurations are supported')
    field = cfg[trainer]
    run, campaign = wd.name, CAMPAIGNS[trainer]
    if (field.get('run_id') != run or field.get('server_id') != server
            or field.get('sensor') != 'GF2' or field.get('campaign_id') != campaign
            or field.get('sheet_tab') != TABS[server] or field.get('num_bands') != 4
            or cfg.get('num_bands') != 4 or cfg.get('max_pixel') != 1023
            or Path(cfg.get('work_dir', '')).name != run):
        raise ValueError('Run/server/campaign/GF2 configuration mismatch')
    model = cfg['model_args']
    if (field.get('width') != model.get('hidden_size') or field.get('depth') != model.get('depth')
            or cfg.get('seed') != field.get('teacher_seed' if field.get('role') == 'T' else 'student_seed')
            or field.get('role') not in ('T', 'S')):
        raise ValueError('Architecture/seed metadata differs from resolved training configuration')
    training = document('meta/training_status.json')
    postrun = document('official/postrun_status.json')
    start = document('meta/training_start_manifest.json')
    receipt = document('official/upload_receipt.json')
    grid = document('official/raw_grid.json')
    name, selection_id = ('exact50k', 'EXACT50K') if trainer == 'g20' else ('raw_max', 'RAW_MAX')
    selected = document('official/' + name + '.json')
    if (training.get('training_complete') is not True or training.get('actual_updates') != 50000
            or postrun.get('official_complete') is not True or postrun.get('actual_updates') != 50000
            or postrun.get('sheet_uploaded') is not True or receipt.get('readback_verified') is not True
            or any(receipt.get(k) != v for k, v in dict(run_id=run, server=server,
                       campaign_id=campaign, sensor='GF2', worksheet=TABS[server]).items())):
        raise ValueError('Exact50K and verified original upload are required')
    if not isinstance(receipt.get('gid'), int) or not isinstance(receipt.get('row'), int):
        raise ValueError('Original upload lacks a concrete worksheet/row receipt')
    for doc in (start, grid, selected):
        if doc.get('run_id') != run or doc.get('campaign_id') != campaign:
            raise ValueError('Stored report/start run identity differs')
    config_sha = object_sha(cfg)
    if any(doc.get('config_sha256') != config_sha for doc in (start, postrun, grid, selected)):
        raise ValueError('Stored config checksum differs')
    source = start.get('source_identity')
    if (not isinstance(source, dict) or not re.fullmatch('[0-9a-f]{64}', source.get('content_sha256', ''))
            or not isinstance(source.get('files'), dict) or object_sha(source['files']) != source['content_sha256']
            or any(doc.get('source_identity') != source for doc in (postrun, grid, selected))):
        raise ValueError('Historical source identity differs between artifacts')
    data_sha = start.get('data_sha256')
    if (not isinstance(data_sha, str) or not re.fullmatch('[0-9a-f]{64}', data_sha)
            or any(doc.get('data_sha256') != data_sha for doc in (grid, selected))
            or ('data_sha256' in postrun and postrun['data_sha256'] != data_sha)):
        raise ValueError('Historical data identity differs between artifacts')
    if (grid.get('complete') is not True or [r.get('update') for r in grid.get('records', [])] != GRID
            or selected.get('official_complete') is not True or selected.get('normal_same_step_A_U') is not True
            or selected.get('selection_id') != selection_id or selected.get('sensor') != 'GF2'
            or selected.get('n_evaluated') != 50):
        raise ValueError('Complete fixed-grid official main selection is required')
    # Independently certify the already-selected main checkpoint; never change it.
    expected = grid['records'][-1] if trainer == 'g20' else min(grid['records'], key=lambda r:
        (-r['fr']['hqnr'], r['update']))
    if (selected.get('step') != expected['update'] or any(selected.get(k) != expected.get(k)
            for k in ('rr', 'fr', 'val_ergas', 'checkpoint_identity'))):
        raise ValueError('Main selection is not the historical fixed-grid selection')
    identity = document(f'candidates/{selected["step"]}/identity.json')
    if (identity != selected.get('checkpoint_identity') or identity.get('update') != selected['step']
            or identity.get('config_sha256') != config_sha or identity.get('source_identity') != source
            or identity.get('data_sha256') != data_sha or selected.get('checkpoint_sha256') != identity.get('model_sha256')):
        raise ValueError('Main checkpoint identity differs from selected metrics')
    model_path = wd / 'candidates' / str(selected['step']) / 'model.safetensors'
    if sha256(model_path) != identity.get('model_sha256'):
        raise ValueError('Main checkpoint bytes differ from saved identity')
    sources[str(model_path.relative_to(wd))] = identity['model_sha256']
    metrics = {}
    for domain, labels in (('rr', RR), ('fr', FR)):
        for label, key in labels.items():
            value = selected[domain].get(key)
            if value is None and key == 'jqm':
                metrics[label] = ''
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Invalid saved main metric: ' + key)
            else:
                metrics[label] = value
    cost = document('official/profile.json')
    if (cost.get('config_sha256') != config_sha or cost.get('source_identity') != source
            or cost.get('num_bands') != 4 or cost.get('sensor') != 'GF2'):
        raise ValueError('Measured resource profile identity differs')
    for label, key in (('Params(M)', 'params_m'), ('FLOPs(G)', 'flops_g'),
                       ('Infer(ms)', 'infer_ms'), ('Mem(MB)', 'mem_mb')):
        value = cost.get(key)
        if key == 'mem_mb' and value is None:
            metrics[label] = ''
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid measured resource value: ' + key)
        else:
            metrics[label] = value
    values = dict(field, width=model['hidden_size'], depth=tuple(model['depth']), seed=cfg['seed'])
    values.setdefault('teacher_alias', field.get('reference_id', ''))
    case = SimpleNamespace(**values)
    from reporting_extra.sensor_sheet import metadata_values
    metadata = metadata_values(wd, cfg, case, training, postrun,
                               'Exact50K' if trainer == 'g20' else 'RAW_MAX')
    # Existing resource numbers are protected too: validate, do not re-upload.
    metrics['Train(h)'] = metadata.pop('Train(h)')
    # The supplement is metadata-only; helper regressions cannot overwrite data.
    allowed = {'Date', 'Seed', 'Model', 'Input', 'Selection', 'Wall(h)', 'Train time scope'}
    allowed.update(trainer.upper() + suffix for suffix in
                   (' started UTC', ' completed UTC', ' official completed UTC', ' Date timezone'))
    if not set(metadata).issubset(allowed):
        raise ValueError('Metadata helper attempted to modify an original result column')
    for name in ('meta/started_at.txt', 'meta/finished_at.txt'):
        if (wd / name).is_file():
            sources[name] = sha256(wd / name)
    return dict(schema=SCHEMA, wd=str(wd), run_id=run, server=server, trainer=trainer,
        campaign_id=campaign, worksheet=TABS[server], prefix=trainer.upper(),
        source_sha256=source['content_sha256'], original_receipt=receipt, source_files=sources,
        metadata=metadata, metrics=metrics, selected_step=selected['step'],
        selected_checkpoint_sha256=selected['checkpoint_sha256'])


def ready_runs(root, server):
    """Only completed local GF2 rows with a successful ordinary upload qualify."""
    result = []
    paths = [p for pattern in ('G20*/meta/config.resolved.yaml', 'QG40*/meta/config.resolved.yaml')
             for p in (Path(root) / 'work_dir').glob(pattern)]
    for path in sorted(paths):
        cfg = yaml.safe_load(path.read_text()) or {}
        trainer = cfg.get('trainer')
        field = cfg.get(trainer, {}) if trainer in CAMPAIGNS else {}
        if field.get('sensor') != 'GF2' or field.get('server_id') != server:
            continue
        wd = path.parent.parent
        paths = [wd / 'meta/training_status.json', wd / 'official/postrun_status.json', wd / 'official/upload_receipt.json']
        if not all(p.is_file() for p in paths):
            continue
        training, postrun, receipt = map(read_json, paths)
        if (training.get('training_complete') is True and training.get('actual_updates') == 50000
                and postrun.get('official_complete') is True and postrun.get('sheet_uploaded') is True
                and receipt.get('readback_verified') is True):
            result.append(wd)
    return result


def column(number):
    result = ''
    while number:
        number, digit = divmod(number - 1, 26)
        result = chr(65 + digit) + result
    return result


def a1(row, col):
    return f'{column(col)}{row}'


def labels_for(headers):
    result = {}
    for col, raw in enumerate(headers, 1):
        label = str(raw).strip()
        if label:
            if label in result:
                raise ValueError('Duplicate Sheet header: ' + label)
            result[label] = col
    return result


def same_cell(actual, expected):
    if expected == '':
        return actual in ('', None)
    if isinstance(expected, bool):
        return str(actual).lower() == str(expected).lower()
    if isinstance(expected, (int, float)):
        try:
            return math.isfinite(float(actual)) and abs(float(actual) - expected) <= 1e-12 * max(1., abs(expected))
        except (ValueError, TypeError):
            return False
    return str(actual) == str(expected)


def same_row(actual, expected):
    # get_all_values may pad rows; row_values may omit the same trailing blanks.
    count = max(len(actual), len(expected))
    return all(same_cell(actual[i] if i < len(actual) else '', expected[i] if i < len(expected) else '')
               for i in range(count))


def plan_update(table, evidence, header_row=3):
    """No row creation, selector change, rounded metric write, or Notes deletion."""
    if len(table) < header_row:
        raise ValueError('Existing header row missing')
    labels = labels_for(table[header_row - 1])
    prefix = evidence['prefix']
    keys = {prefix + ' sensor': 'GF2', prefix + ' campaign': evidence['campaign_id'],
            prefix + ' run id': evidence['run_id'], prefix + ' server': evidence['server'],
            prefix + ' source SHA256': evidence['source_sha256'], 'Run': evidence['run_id']}
    required = set(keys) | set(evidence['metrics']) | {'Notes'}
    if not required.issubset(labels):
        raise ValueError('Existing row provenance/metric headers missing: ' + ', '.join(sorted(required - set(labels))))

    def cell(row, label):
        col = labels[label]
        return row[col - 1] if col <= len(row) else ''

    matches = []
    for number, row in enumerate(table[header_row:], header_row + 1):
        if cell(row, 'Run') == evidence['run_id'] or cell(row, prefix + ' run id') == evidence['run_id']:
            if not all(same_cell(cell(row, k), v) for k, v in keys.items()):
                raise ValueError('Run alias or compound key/source/server conflict')
            matches.append(number)
    if len(matches) != 1:
        raise ValueError('Exactly one existing verified campaign row is required')
    rownum = matches[0]
    old = table[rownum - 1]
    for label, value in evidence['metrics'].items():
        if not same_cell(cell(old, label), value):
            raise ValueError('Sheet main metric differs from full-precision saved report: ' + label)
    from reporting_extra.sensor_sheet import augment_notes, display_formats
    from reporting_extra.sensor_layout import row_formats
    values = dict(evidence['metadata'])
    values['Notes'] = augment_notes(values, str(cell(old, 'Notes')))
    if str(cell(old, 'Notes')) not in values['Notes']:
        raise ValueError('Notes supplement must preserve every existing character')
    missing = [label for label in values if label not in labels]
    # Append beyond every occupied table/header cell, not into a blank group slot.
    first = max(map(len, table), default=0) + 1
    for col, label in enumerate(missing, first):
        labels[label] = col
    edits = [dict(range=a1(header_row, labels[k]), values=[[k]]) for k in missing]
    edits += [dict(range=a1(rownum, labels[k]), values=[[v]]) for k, v in values.items()
              if not same_cell(old[labels[k] - 1] if labels[k] <= len(old) else '', v)]
    owned_labels = {k: v for k, v in labels.items() if k in evidence['metrics'] or k in values}
    formats = display_formats(owned_labels, rownum, METRICS, PREFIXES)
    # Do not apply the helper's whole-row alignment to user-owned extra columns.
    formats += [item for item in row_formats({k: labels[k] for k in ('Run', 'Notes', 'Date') if k in labels}, rownum)
                if ':' not in item['range']]
    return dict(row=rownum, labels=labels, values=values, edits=edits, formats=formats,
                missing_headers=missing, previous_row=old, previous_headers=table[header_row - 1])


def _covers(region, sheet_id, row, col):
    return (region.get('sheetId', sheet_id) == sheet_id
            and region.get('startRowIndex', 0) <= row - 1 < region.get('endRowIndex', float('inf'))
            and region.get('startColumnIndex', 0) <= col - 1 < region.get('endColumnIndex', float('inf')))


def fetch_controls(ws, rows):
    """Pure structural read; intentionally independent of campaign imports."""
    title = ws.title.replace("'", "''")
    fields = ('namedRanges(namedRangeId,range),sheets(properties(sheetId),merges,'
        'protectedRanges(range,namedRangeId,unprotectedRanges),'
        'tables(range,columnProperties(columnIndex,columnType,dataValidationRule)),'
        'data(startRow,startColumn,rowData(values(userEnteredValue(formulaValue),'
        'dataValidation,chipRuns,dataSourceFormula,dataSourceTable))))')
    meta = ws.spreadsheet.fetch_sheet_metadata(params={'includeGridData': True,
        'ranges': [f"'{title}'!A{r}:{column(ws.col_count)}{r}" for r in sorted(set(rows))], 'fields': fields})
    sheets = [s for s in meta.get('sheets', []) if s.get('properties', {}).get('sheetId') == int(ws.id)]
    if len(sheets) != 1:
        raise ValueError('Cannot verify target Sheet controls')
    return dict(sheet=sheets[0], named_ranges={n['namedRangeId']: n['range'] for n in meta.get('namedRanges', [])})


def controlled_reason(controls, row, col):
    sheet = controls['sheet']; sid = sheet['properties']['sheetId']
    for protected in sheet.get('protectedRanges', []):
        region = protected.get('range') or controls['named_ranges'].get(protected.get('namedRangeId'))
        if region is None:
            raise ValueError('Unresolved protected range')
        if _covers(region, sid, row, col) and not any(_covers(r, sid, row, col) for r in protected.get('unprotectedRanges', [])):
            return 'protected range'
    if any(_covers(r, sid, row, col) for r in sheet.get('merges', [])):
        return 'merged cell'
    for table in sheet.get('tables', []):
        if _covers(table['range'], sid, row, col):
            relative = col - 1 - table['range'].get('startColumnIndex', 0)
            for prop in table.get('columnProperties', []):
                if prop.get('columnIndex') == relative and (prop.get('dataValidationRule') or
                        prop.get('columnType', 'COLUMN_TYPE_UNSPECIFIED') != 'COLUMN_TYPE_UNSPECIFIED'):
                    return 'typed table column'
    for grid in sheet.get('data', []):
        ri, ci = row - 1 - grid.get('startRow', 0), col - 1 - grid.get('startColumn', 0)
        rows = grid.get('rowData', [])
        if 0 <= ri < len(rows):
            cells = rows[ri].get('values', [])
            if 0 <= ci < len(cells):
                cell = cells[ci]
                if 'formulaValue' in cell.get('userEnteredValue', {}):
                    return 'formula'
                for key in ('dataValidation', 'chipRuns', 'dataSourceFormula', 'dataSourceTable'):
                    if cell.get(key):
                        return key
    return None


def _range_cells(value):
    def point(text):
        match = re.fullmatch(r'([A-Z]+)([1-9][0-9]*)', text)
        if not match:
            raise ValueError('Expected bounded cell format range: ' + value)
        col = 0
        for letter in match[1]:
            col = col * 26 + ord(letter) - 64
        return int(match[2]), col
    parts = value.split(':')
    if len(parts) not in (1, 2):
        raise ValueError('Invalid cell range')
    start, end = point(parts[0]), point(parts[-1])
    for row in range(start[0], end[0] + 1):
        for col in range(start[1], end[1] + 1):
            yield row, col


@contextlib.contextmanager
def write_locks(root):
    with contextlib.ExitStack() as stack:
        for name in ('.sensor_sheet_write.lock', '.gspread_write.lock', '.g20_sheet_write.lock', '.qg40_sheet_write.lock'):
            path = Path(root) / 'work_dir' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def reporting_identity():
    base = Path(__file__).parent
    names = ('sensor_backfill.py', 'sensor_sheet.py', 'sensor_layout.py')
    result = {name: sha256(base / name) for name in names}
    result['tools/sensor_sheet_sync.py'] = sha256(base.parent / 'tools/sensor_sheet_sync.py')
    return result


def sync_run(wd, server, worksheet, *, apply=False, header_row=3):
    """Supplement one existing row; ordinary receipt and numerical JSON are read-only."""
    root = Path(wd).resolve().parent.parent
    lock = write_locks(root) if apply else contextlib.nullcontext()
    with lock:
        evidence = load_evidence(wd, server)
        if worksheet.title != evidence['worksheet'] or int(worksheet.id) != evidence['original_receipt']['gid']:
            raise ValueError('Worksheet does not match verified ordinary upload')
        table = worksheet.get_all_values(value_render_option='UNFORMATTED_VALUE')
        plan = plan_update(table, evidence, header_row)
        controls = fetch_controls(worksheet, [header_row, plan['row']])
        cells = set()
        for item in plan['edits'] + plan['formats']:
            cells.update(_range_cells(item['range']))
        for row, col in cells:
            if row not in (header_row, plan['row']):
                raise ValueError('Supplement attempted to alter another row')
            reason = controlled_reason(controls, row, col)
            if reason:
                raise ValueError(f'Refusing controlled cell {a1(row, col)}: {reason}')
        summary = dict(schema=SCHEMA, run_id=evidence['run_id'], server=server,
                       worksheet=worksheet.title, row=plan['row'], apply=bool(apply),
                       changed_cells=len(plan['edits']), formats=len(plan['formats']))
        if not apply:
            return dict(summary, edits=plan['edits'], metadata=plan['values'])
        # Prevent mutable evidence changing between local validation and mutation.
        for name, expected in evidence['source_files'].items():
            if sha256(Path(wd) / name) != expected:
                raise ValueError('Evidence changed during reporting: ' + name)
        store = root / 'work_dir/_sensor_sheet' / server / evidence['run_id']
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        latest = read_json(store / 'latest.json') if (store / 'latest.json').is_file() else {}
        report_sources = reporting_identity()
        if (not plan['edits'] and latest.get('readback_verified') is True
                and latest.get('reporting_sources') == report_sources
                and latest.get('metadata_sha256') == object_sha(plan['values'])
                and latest.get('source_files') == evidence['source_files']):
            return dict(summary, readback_verified=True, unchanged=True)
        # Local writers share the locks above. Manual/remote edits cannot be
        # atomically locked by Sheets, so refuse any observed optimistic conflict
        # immediately before the first mutation instead of overwriting a note.
        current_headers = worksheet.row_values(header_row, value_render_option='UNFORMATTED_VALUE')
        current_row = worksheet.row_values(plan['row'], value_render_option='UNFORMATTED_VALUE')
        if (not same_row(current_headers, plan['previous_headers'])
                or not same_row(current_row, plan['previous_row'])):
            raise ValueError('Sheet row/headers changed during reporting; retry from fresh evidence')
        snapshot = store / (stamp + '.before.json')
        _atomic_json(snapshot, dict(schema=SCHEMA, at_utc=utcnow(), evidence=evidence,
            row=plan['row'], worksheet=worksheet.title, gid=int(worksheet.id),
            headers=plan['previous_headers'], values=plan['previous_row'], changes=plan['edits']))
        needed = max(plan['labels'].values())
        if needed > worksheet.col_count:
            worksheet.add_cols(needed - worksheet.col_count)
        if plan['edits']:
            worksheet.batch_update(plan['edits'], value_input_option='RAW')
        if plan['formats']:
            worksheet.batch_format(plan['formats'])
        after = worksheet.row_values(plan['row'], value_render_option='UNFORMATTED_VALUE')
        for label, value in plan['values'].items():
            col = plan['labels'][label]
            if not same_cell(after[col - 1] if col <= len(after) else '', value):
                raise ValueError('Metadata readback mismatch: ' + label)
        # No metric, Run, identity or other original nonmetadata value may move.
        changed_cols = {plan['labels'][label] for label in plan['values']}
        for col, value in enumerate(plan['previous_row'], 1):
            if col not in changed_cols and not same_cell(after[col - 1] if col <= len(after) else '', value):
                raise ValueError('Original row changed during metadata update: ' + column(col))
        headers = worksheet.row_values(header_row, value_render_option='UNFORMATTED_VALUE')
        if any(headers[plan['labels'][label] - 1] != label for label in plan['missing_headers']):
            raise ValueError('Metadata header readback mismatch')
        receipt = dict(summary, readback_verified=True, at_utc=utcnow(),
            reporting_sources=report_sources, source_files=evidence['source_files'],
            metadata_sha256=object_sha(plan['values']), snapshot=str(snapshot), snapshot_sha256=sha256(snapshot),
            original_upload_receipt_sha256=evidence['source_files']['official/upload_receipt.json'])
        _atomic_json(store / (stamp + '.receipt.json'), receipt)
        _atomic_json(store / 'latest.json', receipt)
        return receipt


def connect(root, *, spreadsheet=None, credentials=None):
    """Read only the literal spreadsheet title; never execute legacy collector code."""
    root = Path(root)
    if spreadsheet is None:
        tree = ast.parse((root / 'gspread/gspread_upload.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SHEET' for t in node.targets):
                spreadsheet = ast.literal_eval(node.value)
                break
    if not isinstance(spreadsheet, str) or not spreadsheet:
        raise ValueError('Explicit spreadsheet title or literal legacy SHEET is required')
    import gspread
    return gspread.service_account(filename=str(credentials or root / 'gspread/account.json')).open(spreadsheet)
