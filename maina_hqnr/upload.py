"""Recoverable MAIN-A outbox. RAW numbers retain full precision; display is4dp.

No network, credential access, inference or training occurs at import or queue
time.  Only explicitly activated flushing may contact Google Sheets.
"""
from pathlib import Path
import math

from maina_hqnr.common import (ROOT, atomic_json, camp, locked, object_sha,
    read_json, read_config, run_dir, utcnow)
from maina_hqnr.postrun import CAMPAIGN_ID, GRID, SELECTION_POLICY, SELECTIONS
from maina_hqnr.reporting import METRICS, paired_deltas

TABS = {server: f'SENS-MAINA-WV3-{server}' for server in ('s4', 's5')}
COLLECTION_TAB = 'Hyper parameter experiements all'
SECTION_TITLE = 'WV3 — MAIN-A | PLH/W104D122 | F1 | HQNR_MAX50'
SECTION_END = 'END '+CAMPAIGN_ID
HEADER = tuple(('campaign_id server cycle seed case_id axis value run_id attempt '
    'student_layout width depth teacher_alias teacher_sha calibration_sha q_cache_sha '
    'alpha beta lambda_E fixed_q_ref fixed_tau_R U_peak_lr A_peak_lr '
    'initial_U_sha initial_A_sha stream_sha data_sha source_sha evaluator_sha '
    'actual_updates candidate_grid_sha expected_candidates evaluated_candidates '
    'selection selection_split test_aware independent_test selected_step '
    'checkpoint_AU_sha evaluation_manifest_sha selection_alias_of '
    'hqnr d_lambda d_s ergas sam psnr scc ssim q8 rmse cc jqm '
    'baseline_run_id baseline_selection delta_hqnr delta_ergas delta_d_s delta_d_lambda '
    'training_seconds evaluation_seconds wall_seconds status readback_status '
    'row_key summary_sha256 started_at_utc completed_at_utc Date Methods '
    'paper_identity_status jqm_variant reason').split()) + ('Inference(s)', 'FLOPs(G)', 'Train(h)')


def _finite(value, key):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Nonfinite/missing '+key)
    return value


def validate_summary(summary, case=None):
    from maina_hqnr.plan import validate_case
    from maina_hqnr.evaluation import validate_report
    case = case or summary['case']; validate_case(case)
    if (summary.get('campaign_id') != CAMPAIGN_ID or summary.get('case') != case
            or summary.get('run_id') != case['run_id'] or case['server'] not in TABS):
        raise ValueError('Summary must belong to the canonical new MAIN-A run')
    if type(summary.get('attempt')) is not int or summary['attempt'] < 0:
        raise ValueError('Actual attempt required')
    actual = summary.get('actual_updates')
    if actual is not None and (type(actual) is not int or not 0 <= actual <= 50000):
        raise ValueError('Invalid actual update count')
    if not summary.get('complete'):
        if summary.get('status') == 'COMPLETE' or summary.get('selections'):
            raise ValueError('Partial/failure cannot carry comparable selected results')
        return summary
    if (summary.get('status') != 'COMPLETE' or actual != 50000
            or summary.get('selection_policy') != SELECTION_POLICY
            or summary.get('expected_candidates') != 50 or summary.get('evaluated_candidates') != 50
            or summary.get('candidate_grid_sha256') != object_sha(list(GRID))
            or set(summary.get('selections', {})) != set(SELECTIONS)):
        raise ValueError('Complete MAIN-A requires all50 candidates and exact50K training')
    for key in ('training_seconds', 'evaluation_seconds', 'wall_seconds'):
        if _finite(summary.get(key), key) < 0:
            raise ValueError('Negative runtime')
    for label, row in summary['selections'].items():
        digest = row.get('checkpoint_sha256')
        if row.get('update') not in GRID or label == 'EXACT_50000' and row['update'] != 50000:
            raise ValueError('Invalid selected step')
        if row.get('checkpoint_identity', {}).get('model_sha256') != digest:
            raise ValueError('Selection A/U identity differs')
        validate_report(row['rr'], 'rr', digest)
        validate_report(row['fr'], 'fr', digest, require_jqm=True)
    primary, exact = (summary['selections'][label] for label in SELECTIONS)
    same = primary['checkpoint_sha256'] == exact['checkpoint_sha256']
    if (primary.get('alias_of') or (exact.get('alias_of') or None) != ('HQNR_MAX50' if same else None)
            or same and any(primary[key] != exact[key] for key in ('rr', 'fr', 'evaluation_manifest_sha256'))):
        raise ValueError('EXACT_50000 alias must share the identical selected evaluation')
    return summary


def _cfg_path(value):
    path = Path(value)
    return (path/'meta/config.resolved.yaml' if (path/'meta/config.resolved.yaml').exists()
            else path/'config.json') if path.is_dir() else path


def status_summary(run_dir_or_config, root=ROOT):
    cfg = read_config(_cfg_path(run_dir_or_config)); field = cfg['maina_hqnr']; case = field['case']
    wd = Path(cfg['work_dir']); wd = wd if wd.is_absolute() else Path(root)/wd
    complete = read_json(wd/'official/summary.json')
    if complete:
        return validate_summary(complete, case)
    status = read_json(wd/'meta/training_status.json', {})
    failure = read_json(wd/'meta/attempt_failure.json', {})
    postrun = read_json(wd/'official/postrun_status.json', {})
    start = read_json(wd/'meta/training_start_manifest.json', {})
    if not status and not failure:
        raise ValueError('No persisted execution status; cannot fabricate a failed run')
    label = failure.get('status') or postrun.get('status') or status.get('status', 'TECHNICAL_FAILED')
    if status.get('training_complete') and label not in ('PAUSED_SAFE', 'TECHNICAL_FAILED'):
        label = 'PENDING_EVAL_NOT_COMPARABLE'
    summary = dict(campaign_id=CAMPAIGN_ID, run_id=case['run_id'], case=case, attempt=field['attempt'],
        status=label, complete=False, actual_updates=status.get('actual_updates'),
        training_seconds=status.get('training_seconds'), evaluation_seconds=postrun.get('evaluation_seconds'),
        wall_seconds=status.get('wall_seconds'), started_at_utc=start.get('started_at_utc', start.get('started_at', '')),
        completed_at_utc=failure.get('at_utc', status.get('updated_at_utc', '')),
        reason=failure.get('reason', postrun.get('reason', status.get('reason', ''))),
        evaluated_candidates=postrun.get('evaluated_candidates', 0),
        provenance=dict(runtime_content_sha256=field['source_identity']['content_sha256'],
                        binding_sha256=field['binding_sha256']))
    return validate_summary(summary, case)


def _baseline(summary, root):
    if summary['case']['case_id'] == 'BASE':
        return summary
    folder = run_dir(summary['case']['local_baseline_run_id'], root).parent
    files = list(folder.glob('attempt*/official/COMPLETE.json'))
    if len(files) != 1:
        raise ValueError('Exactly one verified completed same-cycle BASE attempt is required')
    from maina_hqnr.postrun import verify_summary_for_upload
    return verify_summary_for_upload(files[0].parent.parent, root)


def _methods(case, step=None):
    label = 'BASE' if case['axis'] == 'baseline' else {'alpha': 'α', 'beta': 'β', 'lambda_E': 'λE'}[case['axis']]+'='+str(case['value'])
    text = f'{label} | PLH/F1 | {case["server"]} cycle{case["cycle"]}'
    return text+(f' | HQNR@{step}' if step is not None else '')


def flatten_summary(summary, baseline=None):
    from maina_hqnr.plan import CALIBRATION_SHA, Q_CACHE_SHA, DATA_SHA
    validate_summary(summary); case = summary['case']; provenance = summary['provenance']
    rows = []
    for label in SELECTIONS if summary.get('complete') else ('HQNR_MAX50',):
        payload = {key: '' for key in HEADER}
        payload.update(campaign_id=CAMPAIGN_ID, server=case['server'], cycle=case['cycle'], seed=case['seed'],
            case_id=case['case_id'], axis=case['axis'], value=case['value'] if case['value'] is not None else '',
            run_id=case['run_id'], attempt=summary['attempt'], student_layout='PLH', width=104, depth='[1,2,2]',
            teacher_alias='F1', teacher_sha=case['teacher_sha'], calibration_sha=CALIBRATION_SHA, q_cache_sha=Q_CACHE_SHA,
            alpha=case['alpha'], beta=case['beta'], lambda_E=case['lambda_E'], fixed_q_ref=case['fixed_q_ref'],
            fixed_tau_R=case['fixed_tau_R'], U_peak_lr=case['U_peak_lr'], A_peak_lr=case['A_peak_lr'],
            initial_U_sha=provenance.get('initial_U_sha256', ''), initial_A_sha=provenance.get('initial_A_sha256', ''),
            stream_sha=provenance.get('stream_sha256', ''), data_sha=provenance.get('data_sha256', DATA_SHA),
            source_sha=provenance.get('runtime_content_sha256', ''), evaluator_sha=provenance.get('evaluator_sha256', ''),
            actual_updates=summary.get('actual_updates'), candidate_grid_sha=object_sha(list(GRID)), expected_candidates=50,
            evaluated_candidates=summary.get('evaluated_candidates', 0), selection=label, selection_split='FR20',
            test_aware=True, independent_test=False, baseline_run_id=case['local_baseline_run_id'], baseline_selection=label,
            training_seconds=summary.get('training_seconds'), evaluation_seconds=summary.get('evaluation_seconds'),
            wall_seconds=summary.get('wall_seconds'), status=summary['status'], readback_status='READBACK_PENDING',
            summary_sha256=object_sha(summary), started_at_utc=summary.get('started_at_utc', ''),
            completed_at_utc=summary.get('completed_at_utc', ''), Date=summary.get('started_at_utc', '')[:10],
            Methods=_methods(case), reason=summary.get('reason', ''),
            paper_identity_status=summary.get('paper_identity_status', 'PAPERSET_IDENTITY_UNVERIFIED'))
        if summary.get('training_seconds') is not None:
            payload['Train(h)'] = summary['training_seconds']/3600
        payload['row_key'] = object_sha(dict(run_id=case['run_id'], attempt=summary['attempt'], selection=label,
            case_spec_sha256=case['case_spec_sha256'], source_sha=payload['source_sha']))
        if summary.get('complete'):
            row = summary['selections'][label]; pair = paired_deltas(summary, baseline or summary, label)
            payload.update(selected_step=row['update'], checkpoint_AU_sha=row['checkpoint_sha256'],
                evaluation_manifest_sha=row['evaluation_manifest_sha256'], selection_alias_of=row.get('alias_of') or '',
                jqm_variant=row['fr']['jqm_variant'], Methods=_methods(case, row['update']))
            for key in METRICS:
                payload[key] = row['fr' if key in ('hqnr', 'd_lambda', 'd_s', 'jqm') else 'rr'][key]
            for key in ('hqnr', 'ergas', 'd_s', 'd_lambda'):
                payload['delta_'+key] = pair['values'][key]
        # No profiling receipt exists: do not substitute training seconds/FLOPs
        # from another model.  Both profiling cells deliberately stay blank.
        rows.append({key: '' if value is None else value for key, value in payload.items()})
    return rows


def enqueue_run(run_dir_or_config, root=ROOT):
    cfg_path = _cfg_path(run_dir_or_config).resolve()
    summary = status_summary(cfg_path, root)
    if summary.get('complete'):
        from maina_hqnr.postrun import verify_summary_for_upload
        summary = verify_summary_for_upload(cfg_path, root)
    baseline = _baseline(summary, root) if summary.get('complete') else None
    envelopes = []
    for payload in flatten_summary(summary, baseline):
        path = camp(root, payload['server'])/'outbox'/(payload['row_key']+'.json')
        digest = object_sha(payload); previous = read_json(path)
        if previous and previous.get('payload_sha256') == digest:
            envelopes.append(previous); continue
        envelope = dict(schema='MAINA_SHEETS_OUTBOX_v1', config_path=str(cfg_path), server=payload['server'],
                        row_key=payload['row_key'], payload=payload, payload_sha256=digest,
                        status='PENDING', api_attempts=0)
        atomic_json(path, envelope); envelopes.append(envelope)
    return envelopes


def queue_run(root, run_id, attempt=0):
    return enqueue_run(run_dir(run_id, root, attempt), root)


def _a1(row, col):
    letters = ''
    while col:
        col, rem = divmod(col-1, 26); letters = chr(65+rem)+letters
    return letters+str(row)


def _same(actual, expected):
    if expected == '':
        return actual in ('', None)
    if isinstance(expected, bool):
        return actual is expected or str(actual).upper() == str(expected).upper()
    if isinstance(expected, (int, float)):
        try:
            return math.isfinite(float(actual)) and abs(float(actual)-expected) <= 1e-12*max(1, abs(expected))
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _ensure_size(worksheet, row):
    if worksheet.col_count < len(HEADER):
        worksheet.add_cols(len(HEADER)-worksheet.col_count)
    if worksheet.row_count < row:
        worksheet.add_rows(row-worksheet.row_count)


def _format_and_verify(worksheet, row, payload):
    decimal = set(METRICS) | {'delta_'+key for key in METRICS} | {
        'alpha', 'beta', 'lambda_E', 'fixed_q_ref', 'fixed_tau_R', 'training_seconds',
        'evaluation_seconds', 'wall_seconds', 'Train(h)', 'Inference(s)', 'FLOPs(G)'}
    worksheet.batch_format([dict(range=_a1(row, column), format={'numberFormat': {'type': 'NUMBER', 'pattern': '0.0000'}})
                            for column, key in enumerate(HEADER, 1) if key in decimal])
    worksheet.batch_format([dict(range=_a1(row, HEADER.index(key)+1),
        format={'numberFormat': {'type': 'SCIENTIFIC', 'pattern': '0.0E+00'}}) for key in ('U_peak_lr', 'A_peak_lr')])
    actual = worksheet.row_values(row, value_render_option='UNFORMATTED_VALUE')
    if any(not _same(actual[i] if i < len(actual) else '', payload[key]) for i, key in enumerate(HEADER)):
        raise ValueError('Unformatted Sheet readback does not preserve full precision/raw values')
    column = HEADER.index('readback_status')+1
    worksheet.update(range_name=_a1(row, column), values=[['READBACK_VERIFIED']], value_input_option='RAW')
    actual = worksheet.row_values(row, value_render_option='UNFORMATTED_VALUE')
    if len(actual) < column or actual[column-1] != 'READBACK_VERIFIED':
        raise ValueError('Readback receipt write failed')
    return dict(worksheet=worksheet.title, gid=int(worksheet.id), row=row, row_key=payload['row_key'],
                payload_sha256=object_sha(payload), readback_verified=True, at_utc=utcnow())


def _validate_payload(payload):
    if payload.get('campaign_id') != CAMPAIGN_ID or payload.get('server') not in TABS or set(payload) != set(HEADER):
        raise ValueError('Only canonical MAIN-A payloads may be uploaded')


def apply_upsert(worksheet, payload):
    _validate_payload(payload)
    if worksheet.title != TABS[payload['server']]:
        raise ValueError('Refusing to overwrite any historical/raw representative tab')
    header = worksheet.row_values(1)
    if header and header != list(HEADER):
        raise ValueError('Dedicated MAIN-A tab has another header; do not overwrite it')
    _ensure_size(worksheet, 2)
    if not header:
        worksheet.update(range_name='A1', values=[list(HEADER)], value_input_option='RAW')
    rows = worksheet.get_all_values(value_render_option='UNFORMATTED_VALUE')
    column = HEADER.index('row_key')
    matches = [index for index, row in enumerate(rows[1:], 2) if len(row) > column and row[column] == payload['row_key']]
    if len(matches) > 1:
        raise ValueError('Duplicate MAIN-A row key')
    number = matches[0] if matches else max(2, len(rows)+1)
    _ensure_size(worksheet, number)
    worksheet.update(range_name=_a1(number, 1), values=[[payload[key] for key in HEADER]], value_input_option='RAW')
    return _format_and_verify(worksheet, number, payload)


def collection_formula():
    """A read-only live view; only s4 installs this identical deterministic formula."""
    last = _a1(2, len(HEADER)).rstrip('0123456789')
    axis = _a1(2, HEADER.index('axis')+1).rstrip('0123456789')
    arrays = []
    for server in ('s4', 's5'):
        tab = "'"+TABS[server]+"'"
        column = f'{tab}!{axis}2:{axis}'
        key = f'ARRAYFORMULA(IF({column}="baseline",0,IF({column}="alpha",1,IF({column}="beta",2,3))))'
        arrays.append(f'{tab}!A2:{last},{key}')
    query = ('select '+','.join('Col'+str(i) for i in range(1, len(HEADER)+1))+
             f" where Col1='{CAMPAIGN_ID}' and Col{HEADER.index('selection')+1}='HQNR_MAX50'"+
             ' order by '+','.join('Col'+str(column) for column in
                (len(HEADER)+1, HEADER.index('value')+1, HEADER.index('server')+1,
                 HEADER.index('cycle')+1, HEADER.index('attempt')+1)))
    return '=IFERROR(QUERY({'+(';'.join(arrays))+'},"'+query+'",0),"")'


def apply_collection_upsert(worksheet, payload):
    """Verify the live view; s4 alone initializes it, s5 never mutates the tab.

    No writer addresses a result row by a stale index.  The shared collection
    merely projects the two independently owned raw tabs.  Missing/not-yet
    recalculated projection remains in the outbox, never pauses training.
    """
    _validate_payload(payload)
    if worksheet.title != COLLECTION_TAB:
        raise ValueError('Wrong sensitivity collection tab')
    if payload['selection'] != 'HQNR_MAX50':
        return dict(readback_verified=True, status='NOT_APPLICABLE_AUXILIARY', row_key=payload['row_key'])
    rows = worksheet.get_all_values(value_render_option='UNFORMATTED_VALUE')
    starts = [i for i, row in enumerate(rows, 1) if row and row[0] == SECTION_TITLE]
    if not starts:
        if payload['server'] != 's4':
            raise ValueError('COLLECTION_PROJECTION_PENDING: s4-owned view not yet initialized; raw upload retained')
        start = len(rows)+2
        _ensure_size(worksheet, start+100)
        worksheet.update(range_name=_a1(start, 1), values=[[SECTION_TITLE], list(HEADER)], value_input_option='RAW')
        worksheet.update(range_name=_a1(start+2, 1), values=[[collection_formula()]], value_input_option='USER_ENTERED')
        starts = [start]
    if len(starts) != 1:
        raise ValueError('MAIN-A collection section is ambiguous; preserve existing content')
    start = starts[0]
    if worksheet.row_values(start+1) != list(HEADER):
        raise ValueError('MAIN-A section header changed; do not overwrite user formatting/content')
    formula_row = worksheet.row_values(start+2, value_render_option='FORMULA')
    if not any(formula_row) and payload['server'] == 's4':
        # Recover an owner crash between publishing the header and the formula.
        worksheet.update(range_name=_a1(start+2, 1), values=[[collection_formula()]], value_input_option='USER_ENTERED')
        formula_row = worksheet.row_values(start+2, value_render_option='FORMULA')
    if not formula_row or formula_row[0] != collection_formula():
        raise ValueError('Live collection formula changed; do not overwrite user content')
    if payload['server'] == 's4':
        decimal = set(METRICS) | {'delta_'+key for key in METRICS} | {'alpha', 'beta', 'lambda_E', 'fixed_q_ref',
            'fixed_tau_R', 'training_seconds', 'evaluation_seconds', 'wall_seconds', 'Train(h)', 'Inference(s)', 'FLOPs(G)'}
        worksheet.batch_format([dict(range=_a1(start+2, col)+':'+_a1(worksheet.row_count, col),
            format={'numberFormat': {'type': 'NUMBER', 'pattern': '0.0000'}})
            for col, key in enumerate(HEADER, 1) if key in decimal])
    rows = worksheet.get_all_values(value_render_option='UNFORMATTED_VALUE')
    column = HEADER.index('row_key')
    matches = [(number, row) for number, row in enumerate(rows, 1)
               if number >= start+2 and len(row) > column and row[column] == payload['row_key']]
    if len(matches) != 1:
        raise ValueError('COLLECTION_READBACK_PENDING: projection has not resolved this raw row exactly once')
    number, actual = matches[0]
    expected = dict(payload, readback_status='READBACK_VERIFIED')
    if any(not _same(actual[i] if i < len(actual) else '', expected[key]) for i, key in enumerate(HEADER)):
        raise ValueError('Collection projection raw-value readback differs')
    return dict(worksheet=worksheet.title, gid=int(worksheet.id), row=number, row_key=payload['row_key'],
                payload_sha256=object_sha(payload), readback_verified=True,
                mode='READ_ONLY_LIVE_PROJECTION', owner='s4', at_utc=utcnow())


def _worksheets(root, server):
    import gspread
    from fh12.upload import legacy_constants
    constants = legacy_constants()
    client = gspread.service_account(filename=str(Path(root)/'gspread'/Path(constants.CRED).name))
    client.http_client.set_timeout(20)
    book = client.open(constants.SHEET)
    try:
        raw = book.worksheet(TABS[server])
    except gspread.WorksheetNotFound:
        raw = book.add_worksheet(title=TABS[server], rows=100, cols=len(HEADER))
    if server == 's4':
        for other in ('s4', 's5'):
            try:
                sheet = book.worksheet(TABS[other])
            except gspread.WorksheetNotFound:
                try:
                    sheet = book.add_worksheet(title=TABS[other], rows=100, cols=len(HEADER))
                except gspread.exceptions.APIError:
                    sheet = book.worksheet(TABS[other])
            header = sheet.row_values(1)
            if header and header != list(HEADER):
                raise ValueError('New raw tab header differs; cannot initialize live projection')
            if not header:
                _ensure_size(sheet, 2)
                sheet.update(range_name='A1', values=[list(HEADER)], value_input_option='RAW')
    # This user-owned collection must already exist; never guess another title.
    try:
        collection = book.worksheet(COLLECTION_TAB)
    except gspread.WorksheetNotFound:
        collection = None  # Raw uploads still proceed; collection remains pending.
    if server == 's4' and collection is not None:
        # Expanding blank sheet capacity cannot move or replace any existing row.
        needed = len(collection.get_all_values(value_render_option='UNFORMATTED_VALUE'))+100
        needed += sum(len(book.worksheet(TABS[s]).get_all_values(value_render_option='UNFORMATTED_VALUE')) for s in ('s4', 's5'))
        _ensure_size(collection, needed)
    return raw, collection


def flush_outbox(root, server, writer=None, *, activated=False, limit=8):
    if not activated:
        raise PermissionError('Explicit MAIN-A activation required before Sheets access')
    if server not in TABS or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('Bounded server-local upload retry required')
    results = {}; attempts = 0
    with locked(camp(root, server)/'outbox.lock'):
        paths = list((camp(root, server)/'outbox').glob('*.json'))
        # A missing s4-owned projection must not starve newly completed s5 raw
        # results behind indefinitely retrying collection-only envelopes.
        paths.sort(key=lambda path: (bool(read_json(path).get('raw_receipt', {}).get('readback_verified')), path.name))
        for path in paths:
            envelope = read_json(path); payload = envelope.get('payload', {})
            if (envelope.get('schema') != 'MAINA_SHEETS_OUTBOX_v1' or envelope.get('server') != server
                    or envelope.get('row_key') != path.stem or payload.get('row_key') != path.stem
                    or envelope.get('payload_sha256') != object_sha(payload)):
                results[path.stem] = dict(status='BLOCKED_INTEGRITY'); continue
            if envelope.get('status') == 'READBACK_VERIFIED':
                receipt = envelope.get('receipt', {})
                if (receipt.get('readback_verified') is not True or receipt.get('payload_sha256') != envelope['payload_sha256']):
                    results[path.stem] = dict(status='BLOCKED_INTEGRITY', reason='Unverified upload receipt')
                continue
            if attempts >= limit:
                break
            try:
                # Re-certify saved evidence; this path cannot invoke evaluation or training.
                queued = enqueue_run(envelope['config_path'], root)
                current = next(item for item in queued if item['row_key'] == path.stem)
                envelope = current; payload = current['payload']
                envelope['api_attempts'] += 1; attempts += 1
                atomic_json(path, envelope)
                sheets = writer(server) if callable(writer) else writer if writer is not None else _worksheets(root, server)
                raw, collection = sheets
                raw_receipt = apply_upsert(raw, payload)
                envelope['raw_receipt'] = raw_receipt
                atomic_json(path, envelope)
                if collection is None:
                    raise ValueError('COLLECTION_PROJECTION_PENDING: user-owned collection tab is missing')
                collection_receipt = apply_collection_upsert(collection, payload)
                receipt = dict(readback_verified=True, payload_sha256=envelope['payload_sha256'],
                               raw=raw_receipt, collection=collection_receipt, at_utc=utcnow())
                envelope.update(status='READBACK_VERIFIED', receipt=receipt); results[path.stem] = receipt
            except Exception as exc:
                envelope.update(status='PENDING', last_error=f'{type(exc).__name__}: {exc}')
                results[path.stem] = dict(status='UPLOAD_PENDING', reason=envelope['last_error'], readback_verified=False)
            atomic_json(path, envelope)
    return results
