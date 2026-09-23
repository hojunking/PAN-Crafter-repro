"""Official-artifact-only, sensor-local Sheet rows; no training or metric edits."""
import math
from pathlib import Path

from fh12.common import ROOT, atomic_json, read_json, object_sha, utcnow
from fh12.upload import a1, label_map, legacy_constants, same_cell
from qg40.sheet_helpers import fetch_controls, controlled_reason
from reporting_extra.sensor_sheet import metadata_values, augment_notes, merge_notes, display_formats
from reporting_extra.sensor_backfill import write_locks
from reporting_extra.sensor_layout import row_formats
from ablr2.postrun import find_config, _case, _workdir, validate_grid, select_records, selection_report, verify_fullstate, SELECTIONS
from ablr2.plan import SHEET_TABS

RR_LABELS = {'ERGAS↓': 'ergas', 'SAM↓': 'sam', 'PSNR↑': 'psnr', 'SSIM↑': 'ssim',
             'SCC↑': 'scc', 'RMSE↓': 'rmse', 'CC↑': 'cc'}
FR_LABELS = {'HQNR↑': 'hqnr', 'D_lambda↓': 'd_lambda', 'D_s↓': 'd_s', 'JQM↑': 'jqm'}


def method_label(case):
    from ablr2.plan import COMPONENTS
    text=COMPONENTS[case.case_id]['name'] if case.role=='S' else ('Teacher with shift consistency' if case.case_id=='TPLUS' else 'Teacher without shift consistency')
    return ' '.join(f'{case.sweep} | {case.case_id} | {text}'.split())


def _hours(value):
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
        raise ValueError('Resource cost must be measured finite nonnegative seconds or null')
    return value / 3600


def row_values(run, root=ROOT):
    from ablr2.common import source_identity, apply_runtime_policy, assert_compatible_source
    apply_runtime_policy(root)
    cfg = find_config(run, root)
    case, wd = _case(cfg), _workdir(cfg, root)
    grid = read_json(wd / 'official/raw_grid.json')
    status = read_json(wd / 'official/postrun_status.json')
    training = read_json(wd / 'meta/training_status.json')
    if (status.get('official_complete') is not True or status.get('n_evaluated') != 50
            or status.get('actual_updates') != 50000 or training.get('training_complete') is not True
            or training.get('actual_updates') != 50000 or status.get('config_sha256') != object_sha(cfg)
            or grid.get('complete') is not True
            or status.get('source_identity') != grid.get('source_identity')):
        raise ValueError('Upload requires complete same-release fresh50K and all fifty official candidates')
    assert_compatible_source(grid.get('source_identity'),source_identity(root),root,case.server_id)
    data = validate_grid(run, cfg, grid, root)
    verify_fullstate(cfg, root)
    selected, reports = select_records(grid['records'], case.sensor), {}
    for key, _label in SELECTIONS:
        doc = read_json(wd / 'official' / f'{key}.json')
        if doc != selection_report(cfg, grid, key, selected[key]):
            raise ValueError('Selector report differs from fixed-grid evidence: ' + key)
        reports[key] = doc
    field, ref = cfg['ablr2'], {}
    if grid['reference_sha256'] is not None:
        # Clone-only/uniform-KD consume endpoint identity, never a q/tau cache.
        # This proof is exactly the object hashed into the training checkpoint.
        ref = read_json(wd / 'meta/consumed_reference.json')
        if object_sha(ref) != grid['reference_sha256']:
            raise ValueError('Teacher/reference identity differs from trained Student')
    elif case.role == 'S' and case.reference_id is not None:
        raise ValueError('Teacher-dependent Student has null checkpoint reference identity')
    cost = read_json(wd / 'official/profile.json')
    if (cost.get('config_sha256') != object_sha(cfg) or cost.get('source_identity') != grid['source_identity']
            or cost.get('sensor') != case.sensor or cost.get('num_bands') != case.num_bands
            or any(not math.isfinite(float(cost[k])) for k in ('params_m', 'flops_g', 'infer_ms'))):
        raise ValueError('Missing/mismatched measured deployment profile')
    main = reports['rr_val_selected']
    values = {'Run': method_label(case), 'Method':method_label(case), '캠페인': 'ABLR2', 'ABLR2 campaign': grid['campaign_id'], 'ABLR2 sensor': case.sensor,
              'ABLR2 server': case.server_id, 'ABLR2 run id': run, 'ABLR2 role': case.role,
              'ABLR2 case': case.case_id, 'ABLR2 phase': case.phase, 'ABLR2 wave': case.wave,
              'ABLR2 sweep': case.sweep, 'ABLR2 recipe': case.recipe_id, 'ABLR2 recipe revision': case.recipe_revision,
              'ABLR2 Teacher seed': case.teacher_seed, 'ABLR2 Student seed': case.student_seed,
              'ABLR2 reference id': case.reference_id, 'ABLR2 Teacher kind': case.teacher_kind,
              'ABLR2 Teacher SHA256': ref.get('teacher_checkpoint_sha256'), 'ABLR2 reference SHA256': grid['reference_sha256'],
              'ABLR2 calibration ID': ref.get('calibration_id'), 'ABLR2 tau_R': ref.get('tau_R'),
              'ABLR2 q_ref': ref.get('q_ref'), 'ABLR2 q cache SHA256': ref.get('q_cache_sha256'),
              'ABLR2 train-view mean s': ref.get('s_bar'),
              'ABLR2 C': case.num_bands, 'ABLR2 maxDN': case.max_dn,
              'ABLR2 band order': ','.join(data['band_order']), 'ABLR2 data SHA256': grid['data_sha256'],
              'ABLR2 source SHA256': grid['source_identity']['content_sha256'],
              'ABLR2 planned updates': 50000, 'ABLR2 actual updates': 50000, 'ABLR2 candidates': 50,
              'ABLR2 status': 'OFFICIAL_EVAL_COMPLETE', 'ABLR2 upload status': 'READBACK_PENDING',
              'ABLR2 development mode': 'TEST_AWARE_DEV', 'ABLR2 independent test': False,
              'ABLR2 checkpoint rule': 'RR_VALIDATION_ARGMIN_ERGAS_THEN_LOWER_STEP',
              'ABLR2 VAL ERGAS': main['val_ergas'], 'ABLR2 selected step': main['step'],
              'ABLR2 selected SHA256': main['checkpoint_sha256'],
              'Params(M)': cost['params_m'], 'FLOPs(G)': cost['flops_g'], 'Infer(ms)': cost['infer_ms'], 'Mem(MB)': cost.get('mem_mb'),
              'ABLR2 trainable Params(M)': cost.get('trainable_params_m'), 'ABLR2 aligner Params(M)': cost.get('aligner_params_m'),
              'ABLR2 alignment mode': cost['alignment_mode'], 'ABLR2 cost scope': cost['scope'],
              'ABLR2 FLOPs convention': cost['flops_convention'],
              'ABLR2 eval(h)': _hours(training.get('evaluation_seconds')),
              'ABLR2 save(h)': _hours(training.get('io_seconds')),
              'ABLR2 retry(h)': _hours(training.get('retry_seconds')),
              'ABLR2 Teacher calibration(h)': _hours(ref.get('calibration_seconds')),
              'Notes': 'ABLR2 main=RR_VAL_SELECTED; Exact50K secondary. RAW_MAX/E_MIN are exploratory only. '
                       'TEST_AWARE_DEV: native FR20 reused for adaptive development; no independent-test claim. '
                       'Native PAN/original LMS, no border mask. JQM is an SRF substitute. All attempts remain separate. '
                       'Every Student is a fresh run; displayed steps are components, not sequential finetuning. Run ID: '+run}
    present = dict(training)
    if training.get('training_completed_at_utc'):
        present['updated_at_utc'] = training['training_completed_at_utc']
    metadata = metadata_values(wd, cfg, case, present, status, 'RR_VAL_SELECTED')
    values.update(metadata)
    values['Notes'] = augment_notes(metadata, values['Notes'])
    rr_labels = dict(RR_LABELS, **{f'Q{case.num_bands}↑': f'q{case.num_bands}'})
    for labels, domain in ((rr_labels, 'rr'), (FR_LABELS, 'fr')):
        values.update({label: main[domain].get(key) for label, key in labels.items()})
    for key, label in SELECTIONS:
        doc = reports[key]
        prefix = 'ABLR2 ' + label
        values.update({prefix + ' step': doc['step'], prefix + ' SHA256': doc['checkpoint_sha256'],
                       prefix + ' test-aware': doc['test_aware'], prefix + ' val ERGAS': doc['val_ergas']})
        for labels, domain in ((rr_labels, 'rr'), (FR_LABELS, 'fr')):
            values.update({prefix + ' ' + display: doc[domain].get(metric) for display, metric in labels.items()})
        values[prefix + ' JQM variant'] = doc['fr'].get('jqm_variant')
        values[prefix + ' JQM status'] = doc['fr'].get('jqm_status')
    return {key: '' if value is None else value for key, value in values.items()}


def metric_formats(labels, row):
    metrics = set(RR_LABELS) | set(FR_LABELS) | {'Q4↑', 'Q8↑', 'val ERGAS', 'ABLR2 VAL ERGAS'}
    return display_formats(labels, row, metrics, tuple('ABLR2 ' + label for _, label in SELECTIONS))


def plan_upsert(table, headers, values, *, header_row=3):
    values, labels = dict(values), label_map(headers)
    if 'Run' not in labels:
        raise ValueError('Existing Sheet must have an unambiguous Run header')
    from ablr2.plan import campaign_id
    key_names = ('ABLR2 sensor', 'ABLR2 campaign', 'ABLR2 run id')
    key = tuple(values[k] for k in key_names)
    if key[0] not in ('WV3','QB','GF2') or key[1] != campaign_id(key[0]) or not key[2].startswith('ABLR2_'+key[0]+'_'):
        raise ValueError('Invalid ABLR2 compound identity')
    def cell(row, label):
        col = labels.get(label, 0)
        return row[col - 1] if 0 < col <= len(row) else ''
    matches = []
    for number, row in enumerate(table[header_row:], header_row + 1):
        identity = tuple(cell(row, label) for label in key_names)
        if identity == key:
            matches.append(number)
        elif cell(row, 'Run') == key[2]:
            raise ValueError('Cannot repurpose a historical/non-ABLR2 row')
    if len(matches) > 1:
        raise ValueError('Duplicate ABLR2 run rows')
    rownum = matches[0] if matches else max(header_row + 1, len(table) + 1)
    old = table[rownum - 1] if rownum <= len(table) else []
    if old:
        for label, value in values.items():
            previous = cell(old, label)
            immutable_metric = label in set(RR_LABELS) | set(FR_LABELS) | {'Q4↑', 'Q8↑'} or label.startswith(tuple('ABLR2 ' + name for _, name in SELECTIONS))
            if immutable_metric and previous not in ('', None) and not same_cell(previous, value):
                raise ValueError('Refusing to replace an existing observation: ' + label)
    if 'Notes' in values:
        values['Notes'] = merge_notes(values['Notes'], cell(old, 'Notes'))
    missing = [label for label in values if label not in labels]
    first = max(len(headers), max((len(row) for row in table[:header_row]), default=0)) + 1
    labels.update({label: col for col, label in enumerate(missing, first)})
    edits = [dict(range=a1(header_row, labels[label]), values=[[label]]) for label in missing]
    edits += [dict(range=a1(rownum, labels[label]), values=[[value]]) for label, value in values.items()]
    owned = {label: labels[label] for label in values}
    formats = metric_formats(owned, rownum) + [r for r in row_formats(owned, rownum) if ':' not in r['range']]
    return dict(row=rownum, labels=labels, missing_headers=missing, values=values, edits=edits, formats=formats)


def apply_upsert(ws, values, *, header_row=3):
    # Display is four decimals, but immutable-observation comparisons and
    # concurrency checks must inspect the original full-precision cell values.
    table = ws.get_all_values(value_render_option='UNFORMATTED_VALUE')
    headers = ws.row_values(header_row)
    plan = plan_upsert(table, headers, values, header_row=header_row)
    if 'ABLR2 upload status' not in values:
        raise ValueError('Owned upload status is required')
    controls = fetch_controls(ws, [header_row, plan['row']])
    touched = {a1(header_row, plan['labels'][label]): (header_row, plan['labels'][label]) for label in plan['missing_headers']}
    touched.update({a1(plan['row'], plan['labels'][label]): (plan['row'], plan['labels'][label]) for label in values})
    if any(edit['range'] not in touched for edit in (*plan['edits'], *plan['formats'])):
        raise ValueError('Upload extends outside the owned exact cells')
    for address, (row, col) in touched.items():
        reason = controlled_reason(controls, row, col)
        if reason:
            raise ValueError(f'Refusing controlled cell {address}: {reason}')
    if ws.get_all_values(value_render_option='UNFORMATTED_VALUE') != table or ws.row_values(header_row) != headers:
        raise ValueError('Sheet changed during planning; retry fresh evidence')
    if max(plan['labels'].values()) > ws.col_count:
        ws.add_cols(max(plan['labels'].values()) - ws.col_count)
    if plan['row'] > ws.row_count:
        ws.add_rows(plan['row'] - ws.row_count)
    ws.batch_update(plan['edits'], value_input_option='RAW')
    ws.batch_format(plan['formats'])
    observed = ws.row_values(plan['row'], value_render_option='UNFORMATTED_VALUE')
    for label, want in plan['values'].items():
        col = plan['labels'][label]
        if not same_cell(observed[col - 1] if col <= len(observed) else '', want):
            raise ValueError('Upload readback mismatch: ' + label)
    col = plan['labels']['ABLR2 upload status']
    ws.batch_update([dict(range=a1(plan['row'], col), values=[['READBACK_VERIFIED']])], value_input_option='RAW')
    if ws.row_values(plan['row'], value_render_option='UNFORMATTED_VALUE')[col - 1] != 'READBACK_VERIFIED':
        raise ValueError('Upload status readback failed')
    return dict(row=plan['row'], worksheet=ws.title, gid=int(ws.id), readback_verified=True, payload_sha256=object_sha(plan['values']))


def spool_run(run,root=ROOT):
    """Immutable full-precision payload is published before any network access."""
    from ablr2.common import camp,immutable_json,append_event,locked,read
    cfg=find_config(run,root);case=_case(cfg);values=row_values(run,root)
    folder=camp(root,case.server_id)/'upload_outbox';digest=object_sha(values)
    with locked(folder/'spool.lock'):
        payload=dict(schema='ABLR2X_UPLOAD_PAYLOAD_v1',run_id=run,server=case.server_id,
            sensor=case.sensor,payload_sha256=digest,values=values)
        path=folder/'payloads'/(digest+'.json');immutable_json(path,payload)
        pointer=folder/'pending'/(run+'.json');old=read(pointer)
        if old.get('payload_sha256')==digest:return old
        value=dict(run_id=run,server=case.server_id,payload_sha256=digest,payload_path=str(path),
            status='PENDING',attempts=0,updated_at_utc=utcnow())
        append_event(folder/'events.jsonl','SPOOLED',**value);atomic_json(pointer,value)
        return value


def _pending_path(case,root):
    from ablr2.common import camp
    return camp(root,case.server_id)/'upload_outbox/pending'/(case.run_id+'.json')


def upload_run(run, root=ROOT, *, activated=False, worksheet=None, summary_worksheet=None, header_row=3):
    if not activated:
        raise PermissionError('ABLR2 Sheet writes require explicit upload activation')
    cfg = find_config(run, root)
    case = _case(cfg)
    pending=spool_run(run,root)
    payload=read_json(pending['payload_path']);values=payload['values']
    if payload['run_id']!=run or object_sha(values)!=pending['payload_sha256']:
        raise ValueError('Outbox payload integrity differs')
    if worksheet is None:
        import gspread
        constants = legacy_constants()
        client = gspread.service_account(filename=str(Path(root) / 'gspread' / Path(constants.CRED).name))
        client.set_timeout(20)
        book = client.open(constants.SHEET)
        worksheet = book.worksheet(SHEET_TABS[case.server_id])
        if case.sensor=='GF2' and case.role=='S':summary_worksheet=book.worksheet('ablations')
    if worksheet.title != SHEET_TABS[case.server_id]:
        raise ValueError('Refusing a different sensor/server Sheet tab')
    with write_locks(root):
        receipt = apply_upsert(worksheet, values, header_row=header_row)
        if case.sensor=='GF2' and case.role=='S':
            if summary_worksheet is None:raise ValueError('GF2 component summary worksheet is required for complete readback')
            receipt['component_summary']=apply_component_summary(summary_worksheet,values)
        receipt.update(run_id=run, sensor=case.sensor, server=case.server_id, uploaded_at_utc=utcnow())
        atomic_json(_workdir(cfg, root) / 'official/upload_receipt.json', receipt)
        from ablr2.common import append_event,camp
        pending.update(status='READBACK_VERIFIED',receipt=receipt,updated_at_utc=utcnow())
        append_event(camp(root,case.server_id)/'upload_outbox/events.jsonl','READBACK_VERIFIED',run_id=run,payload_sha256=pending['payload_sha256'])
        atomic_json(_pending_path(case,root),pending)
    return receipt


def retry_pending(root,server,limit=8,*,activated=False,worksheets=None):
    """Bounded API batch; durable pending items never restart a training run."""
    from ablr2.common import camp,append_event
    if not activated:raise PermissionError('Explicit upload activation required')
    if type(limit) is not int or not 1<=limit<=100:raise ValueError('Bounded retry count required')
    folder=camp(root,server)/'upload_outbox';items=[];out={}
    for path in (folder/'pending').glob('*.json'):
        try:
            value=read_json(path)
            if (not isinstance(value,dict) or not isinstance(value.get('run_id'),str)
                    or value.get('server')!=server or path.name!=value['run_id']+'.json'
                    or type(value.get('attempts',0)) is not int or value.get('attempts',0)<0
                    or not isinstance(value.get('updated_at_utc',''),str)
                    or value.get('status') not in ('PENDING','READBACK_VERIFIED')):
                raise ValueError('Malformed/cross-lane outbox pointer')
            if value['status']=='READBACK_VERIFIED':
                if value.get('receipt',{}).get('readback_verified') is not True:
                    raise ValueError('Readback status has no verified receipt')
                continue
            items.append((value.get('updated_at_utc',''),path,value))
        except (ValueError,TypeError,AttributeError,KeyError,OSError) as exc:
            # Preserve the original broken bytes for diagnosis. A corrupt item
            # cannot consume the API batch or starve unrelated healthy results.
            out[path.stem]=dict(status='UPLOAD_BLOCKED_INTEGRITY',reason=f'{type(exc).__name__}: {exc}',pointer=str(path))
    for _,path,value in sorted(items,key=lambda x:(x[0],str(x[1])))[:limit]:
        run=value['run_id']
        try:
            if value.get('server')!=server or path.name!=run+'.json':raise ValueError('Cross-lane outbox identity')
            outcome=upload_run(run,root,activated=True,**(worksheets or {}))
        except Exception as exc:
            value.update(status='PENDING',attempts=value.get('attempts',0)+1,updated_at_utc=utcnow(),last_error=f'{type(exc).__name__}: {exc}')
            atomic_json(path,value);append_event(folder/'events.jsonl','UPLOAD_FAILED',run_id=run,reason=value['last_error'])
            outcome=dict(status='UPLOAD_PENDING',reason=value['last_error'])
        out[run]=outcome
    atomic_json(folder/'retry_status.json',dict(updated_at_utc=utcnow(),server=server,results=out))
    return out


SUMMARY_BEGIN='ABLR2X_GF2_COMPONENTS_v1 BEGIN'
SUMMARY_END='ABLR2X_GF2_COMPONENTS_v1 END'
SUMMARY_HEADERS=('Methods','Run ID','Recipe','Phase','Wave','Sweep','Case','Selection','Checkpoint SHA256',
    'HQNR↑','D_lambda↓','D_s↓','ERGAS↓','SAM↓','SCC↑','Q4↑','PSNR↑','SSIM↑','RMSE↓','CC↑','JQM↑','Notes')
COMPONENT_ORDER=('C00','C01','C02','C03','C17',*tuple(f'C{i:02}' for i in range(4,17)))


def component_summary_values(values):
    from ablr2.plan import campaign_id
    run=values.get('ABLR2 run id','');case=values.get('ABLR2 case')
    if (values.get('ABLR2 sensor')!='GF2' or values.get('ABLR2 server')!='s3'
            or values.get('ABLR2 campaign')!=campaign_id('GF2') or values.get('ABLR2 role')!='S'
            or case not in COMPONENT_ORDER or not run.startswith('ABLR2_GF2_s3_')
            or '_'+case+'_SS' not in run or values.get('ABLR2 status')!='OFFICIAL_EVAL_COMPLETE'
            or values.get('ABLR2 actual updates')!=50000):
        raise ValueError('Only actual completed GF2 C00..C17 component runs may enter this summary')
    method=values.get('Method',values['Run'])
    if method!=' '.join(method.split()) or any(token in method.lower() for token in ('seed','_ss','_ts')):
        raise ValueError('Component Methods must be one line without displayed seeds')
    result={'Methods':method,'Run ID':run,'Recipe':values['ABLR2 recipe revision']+' / '+values['ABLR2 recipe'],
        'Phase':values['ABLR2 phase'],'Wave':values['ABLR2 wave'],'Sweep':values['ABLR2 sweep'],
        'Case':case,'Selection':'RR_VAL_SELECTED','Checkpoint SHA256':values['ABLR2 selected SHA256'],
        'Notes':values.get('Notes','')}
    result.update({key:values.get(key,'') for key in SUMMARY_HEADERS if key.endswith(('↑','↓'))})
    for key in ('HQNR↑','D_lambda↓','D_s↓','ERGAS↓','SAM↓','SCC↑','Q4↑','PSNR↑','SSIM↑','RMSE↓','CC↑'):
        if isinstance(result[key],bool) or not isinstance(result[key],(int,float)) or not math.isfinite(result[key]):
            raise ValueError('Summary cannot fabricate missing metrics: '+key)
    return result


def plan_component_summary(table,values):
    """Discover exact owned markers; never guess absolute rows or import production results."""
    item=component_summary_values(values);runs={};starts=[];ends=[]
    for i,row in enumerate(table):
        if SUMMARY_BEGIN in row:
            if row[0]!=SUMMARY_BEGIN or any(v not in ('',None) for v in row[1:]):raise ValueError('Ambiguous summary ownership marker')
            starts.append(i)
        if SUMMARY_END in row:
            if row[0]!=SUMMARY_END or any(v not in ('',None) for v in row[1:]):raise ValueError('Ambiguous summary ownership marker')
            ends.append(i)
    if starts or ends:
        if len(starts)!=1 or len(ends)!=1 or ends[0]<=starts[0]+1:raise ValueError('Malformed/duplicate component summary region')
        first,last=starts[0],ends[0]
        header=table[first+1]
        if tuple(header[:len(SUMMARY_HEADERS)])!=SUMMARY_HEADERS or any(v not in ('',None) for v in header[len(SUMMARY_HEADERS):]):
            raise ValueError('Component summary headers changed; refusing overwrite')
        for row in table[first+2:last]:
            if not any(row):continue
            if any(v not in ('',None) for v in row[len(SUMMARY_HEADERS):]):raise ValueError('Unowned cells in component summary region')
            row=row[:len(SUMMARY_HEADERS)]
            record=dict(zip(SUMMARY_HEADERS,row+['']*(len(SUMMARY_HEADERS)-len(row))))
            run=record['Run ID']
            if not run.startswith('ABLR2_GF2_s3_') or record['Case'] not in COMPONENT_ORDER or run in runs:
                raise ValueError('Foreign/duplicate run in component summary region')
            runs[run]=record
        if item['Run ID'] in runs:
            old=runs[item['Run ID']]
            if any(not same_cell(old[k],item[k]) for k in SUMMARY_HEADERS if k not in ('Methods','Notes')):
                raise ValueError('Immutable component summary observation changed')
        elif last!=len(table)-1:
            raise ValueError('Cannot expand owned summary into following unowned rows')
    else:first=len(table);last=first-1
    runs[item['Run ID']]=item
    ordered=sorted(runs.values(),key=lambda r:(r['Recipe'],r['Phase'],r['Wave'],r['Sweep'],COMPONENT_ORDER.index(r['Case']),r['Run ID']))
    body=[[SUMMARY_BEGIN],list(SUMMARY_HEADERS)]+[[r[k] for k in SUMMARY_HEADERS] for r in ordered]+[[SUMMARY_END]]
    ncols=len(SUMMARY_HEADERS)
    # Clear only the existing owned rectangle if an interrupted expansion left
    # fewer rows in the newly rebuilt representation; no external row is touched.
    nrows=max(len(body),last-first+1)
    matrix=[row+['']*(ncols-len(row)) for row in body]+[['']*ncols for _ in range(nrows-len(body))]
    return dict(first_row=first+1,last_row=first+nrows,ncols=ncols,matrix=matrix,
        run_row=first+3+next(i for i,r in enumerate(ordered) if r['Run ID']==item['Run ID']),
        records=ordered)


def apply_component_summary(ws,values):
    if ws.title!='ablations':raise ValueError('Component summary only owns an existing ablations tab region')
    table=ws.get_all_values(value_render_option='UNFORMATTED_VALUE');plan=plan_component_summary(table,values)
    rows=list(range(plan['first_row'],plan['last_row']+1));controls=fetch_controls(ws,rows)
    for row in rows:
        for col in range(1,plan['ncols']+1):
            reason=controlled_reason(controls,row,col)
            if reason:raise ValueError(f'Refusing controlled summary cell {a1(row,col)}: {reason}')
    if ws.get_all_values(value_render_option='UNFORMATTED_VALUE')!=table:raise ValueError('Summary changed during planning')
    if plan['ncols']>ws.col_count:ws.add_cols(plan['ncols']-ws.col_count)
    if plan['last_row']>ws.row_count:ws.add_rows(plan['last_row']-ws.row_count)
    ws.batch_update([dict(range=a1(plan['first_row'],1)+':'+a1(plan['last_row'],plan['ncols']),values=plan['matrix'])],value_input_option='RAW')
    observed=ws.get_all_values(value_render_option='UNFORMATTED_VALUE')
    for offset,row in enumerate(plan['matrix']):
        actual=observed[plan['first_row']-1+offset] if plan['first_row']-1+offset<len(observed) else []
        if any(not same_cell(actual[i] if i<len(actual) else '',want) for i,want in enumerate(row)):
            raise ValueError('Component summary raw-value readback mismatch')
    formats=[]
    for row in range(plan['first_row']+2,plan['last_row']):
        for col,label in enumerate(SUMMARY_HEADERS,1):
            if label.endswith(('↑','↓')):formats.append(dict(range=a1(row,col),format={'numberFormat':{'type':'NUMBER','pattern':'0.0000'}}))
    if formats:ws.batch_format(formats)
    return dict(worksheet=ws.title,gid=int(ws.id),row=plan['run_row'],readback_verified=True,
        region=SUMMARY_BEGIN,run_id=values['ABLR2 run id'],payload_sha256=object_sha(values))
