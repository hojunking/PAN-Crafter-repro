"""Dedicated-tab-only outbox; full precision RAW values, display formatting only.

There are no credential reads or Sheet mutations until activated=True. Failed
uploads retry this outbox, never training or evaluation.
"""
import csv
import math
from pathlib import Path

from g23sens.common import (ROOT,atomic_json,camp,object_sha,read,read_json,
    read_config,run_dir,utcnow,locked)
from g23sens.reporting import METRICS,SELECTIONS,paired_deltas

CAMPAIGN_ID='PANDA_G23_SENS_WV3_S45_20260923_v1'
TABS={s:f'SENS-G23-WV3-{s}' for s in ('s4','s5')}
with (ROOT/'research_log/PANDA_G23_SENS_S45_UNLIMITED_2026-09-23/result_columns.csv').open(encoding='utf-8-sig',newline='') as _stream:
    HEADER=tuple(row['column'] for row in csv.DictReader(_stream))+('row_key','campaign_id','summary_sha256',
        'baseline_attempt','baseline_summary_sha256','started_at_utc','completed_at_utc','Date','reason')


def _finite(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Invalid metric/value '+label)
    return value


def validate_summary(summary,case):
    from g23sens.plan import validate_case
    validate_case(case)
    if (summary.get('campaign_id')!=CAMPAIGN_ID or summary.get('case')!=case
            or summary.get('run_id')!=case['run_id'] or case['server'] not in TABS):
        raise ValueError('Summary campaign/case identity mismatch')
    if type(summary.get('attempt')) is not int or summary['attempt']<0:raise ValueError('Attempt required')
    actual=summary.get('actual_updates')
    if not (actual is None and not summary.get('complete')) and (type(actual) is not int or not 0<=actual<=50000):
        raise ValueError('Invalid update count')
    cost=summary.get('training_seconds')
    if cost is not None or summary.get('complete'):_finite(cost,'training_seconds')
    if cost is not None and cost<0:raise ValueError('Negative training cost')
    if not summary.get('complete'):
        if summary.get('status')=='COMPLETE' or summary.get('selections'):raise ValueError('Failure cannot masquerade as completed metrics')
        return summary
    if actual!=50000 or summary.get('status')!='COMPLETE' or set(summary.get('selections',{}))!=set(SELECTIONS):
        raise ValueError('Official complete requires fresh exact50K plus both selections')
    for label,record in summary['selections'].items():
        if record['update'] not in tuple(range(1010,50000,1010))+(50000,):raise ValueError('Selection outside fixed validation grid')
        if label=='EXACT_50000' and record['update']!=50000:raise ValueError('Primary is not exact50K')
        for domain,keys in (('rr',('ergas','sam','psnr','ssim','scc','q8','rmse','cc')),('fr',('hqnr','d_lambda','d_s','jqm'))):
            if type(record[domain].get('n_scenes')) is not int or record[domain]['n_scenes']<1:
                raise ValueError('Actual scene population required')
            for key in keys:_finite(record[domain].get(key),key)
        if 'SRF-substitute' not in record['fr'].get('jqm_variant',''):raise ValueError('JQM surrogate variant must be explicit')
    first,second=[summary['selections'][k] for k in SELECTIONS]
    same=first['checkpoint_sha256']==second['checkpoint_sha256']
    if (first.get('alias_of') or (second.get('alias_of') or None)!=('EXACT_50000' if same else None)
            or same and any(first[k]!=second[k] for k in ('rr','fr','evaluation_manifest_sha256'))):
        raise ValueError('Checkpoint alias must share actual evaluation, not independent evidence')
    return summary


def status_summary(run,root=ROOT,attempt=0):
    wd=run_dir(run,root,attempt=attempt)
    config_path=wd/'meta/config.resolved.yaml' if (wd/'meta/config.resolved.yaml').exists() else wd/'config.json'
    cfg=read_config(config_path);field=cfg['g23sens'];case=field['case']
    if field['attempt']!=attempt or case['run_id']!=run:raise ValueError('Attempt config differs')
    failure=read(wd/'meta/attempt_failure.json')
    if failure and (failure.get('run_id')!=run or failure.get('attempt')!=attempt or failure.get('final') is not True
            or failure.get('kind') not in ('TRAIN_TECHNICAL','EVALUATION') or failure.get('status')!='TECHNICAL_FAILED'):
        raise ValueError('Technical attempt ledger identity differs')
    complete=read(wd/'official/summary.json')
    if complete and not failure:return validate_summary(complete,case)
    status=read(wd/'meta/training_status.json');start=read(wd/'meta/training_start_manifest.json')
    if not status and not failure:raise ValueError('No persisted status/failure ledger; do not fabricate failure')
    if status.get('training_complete') and not failure:raise ValueError('Training complete but evaluation missing: not a normal result')
    source=field['source_identity']
    expected=dict(run_id=run,attempt=attempt,config_sha256=object_sha(cfg),source_identity=source,
        bindings_sha256=field['binding_sha256'],case_spec_sha256=case['case_spec_sha256'])
    if status and any(status.get(k)!=v for k,v in expected.items()):raise ValueError('Failure status actual execution identity differs')
    summary=dict(campaign_id=CAMPAIGN_ID,run_id=run,case=case,attempt=attempt,complete=False,
        status=failure.get('status',status.get('status','TECHNICAL_FAILURE')),reason=failure.get('reason',status.get('reason','')),
        actual_updates=failure.get('actual_updates',status.get('actual_updates')),training_seconds=status.get('training_seconds'),
        started_at_utc=start.get('started_at_utc') or '',
        completed_at_utc=failure.get('at_utc') or status.get('completed_at_utc') or status.get('completed_at') or status.get('updated_at_utc') or '',
        attempt_failure=failure or None,
        provenance=dict(case_spec_sha256=case['case_spec_sha256'],runtime_config_sha256=object_sha(cfg),
            runtime_commit=source.get('git_release',source.get('commit','')),runtime_content_sha256=source.get('content_sha256',''),
            binding_sha256=field['binding_sha256'],teacher_checkpoint_sha256=case['fixed']['teacher_checkpoint_sha256']))
    return validate_summary(summary,case)


def _baseline(summary,root):
    case=summary['case']
    if case['case_id']=='BASE':return summary
    folder=run_dir(case['local_baseline_run_id'],root,attempt=0).parent
    paths=list(folder.glob('attempt*/official/summary.json'))
    complete=[read_json(path) for path in paths if read_json(path).get('complete')]
    if len(complete)!=1:raise ValueError('Exactly one completed local BASE attempt is required')
    from g23sens.postrun import verify_summary_for_upload
    return verify_summary_for_upload(case['local_baseline_run_id'],root,attempt=complete[0]['attempt'])


def flatten_summary(summary,baseline=None):
    case=summary['case'];validate_summary(summary,case);parameters=case['parameters'];resolved=case['resolved_values']
    if '\n' in case['run_display'] or case['run_id'] in case['run_display']:raise ValueError('Run must show only one-line model/settings')
    rows=[]
    for label in SELECTIONS if summary.get('complete') else ('EXACT_50000',):
        value={k:'' for k in HEADER};provenance=summary['provenance']
        value.update({k:provenance.get(k,'') for k in HEADER if k in provenance})
        value.update(run_id=case['run_id'],Run=case['run_display'],server=case['server'],cycle=case['cycle'],seed=case['seed'],
            case_id=case['case_id'],axis=case['sweep_axis'],value=case['sweep_value'] if case['sweep_value'] is not None else '',
            baseline_run_id=case['local_baseline_run_id'],selection=label,actual_updates=summary['actual_updates'] if summary['actual_updates'] is not None else '',
            alpha=parameters['alpha'],beta=parameters['beta'],lambda_E=parameters['lambda_E'],
            q_ref_base=case['fixed']['q_ref_base'],q_ref_scale=parameters['q_ref_scale'],q_ref_used=resolved['q_ref'],
            tau_R_base=case['fixed']['tau_R_base'],tau_R_scale=parameters['tau_R_scale'],tau_R_used=resolved['tau_R_used'],
            r_A=parameters['r_A'],U_peak_lr=1e-4,A_peak_lr=resolved['A_peak_lr'],status=summary['status'],
            attempt=summary['attempt'],training_seconds=summary['training_seconds'] if summary['training_seconds'] is not None else '',campaign_id=CAMPAIGN_ID,
            summary_sha256=object_sha(summary),started_at_utc=summary.get('started_at_utc',''),
            completed_at_utc=summary.get('completed_at_utc',''),Date=summary.get('started_at_utc','')[:10],
            reason=summary.get('reason',''),readback_status='READBACK_PENDING')
        value['Train(h)']=summary['training_seconds']/3600 if summary['training_seconds'] is not None else ''
        value['row_key']=object_sha(dict(run_id=case['run_id'],selection=label,attempt=summary['attempt'],
            case_spec_sha256=case['case_spec_sha256'],runtime_content_sha256=provenance['runtime_content_sha256']))
        if summary.get('complete'):
            record=summary['selections'][label];pair=paired_deltas(summary,baseline or summary,label)
            value.update(selected_step=record['update'],checkpoint_sha256=record['checkpoint_sha256'],
                selection_alias_of=record.get('alias_of') or '',evaluation_manifest_sha256=record['evaluation_manifest_sha256'],
                rr_n_scenes=record['rr']['n_scenes'],fr_n_scenes=record['fr']['n_scenes'],
                paper_identity_status=summary['paper_identity_status'],jqm_variant=record['fr']['jqm_variant'],
                baseline_attempt=pair['baseline_attempt'],baseline_summary_sha256=pair['baseline_summary_sha256'])
            for key in METRICS:value[key]=record['fr' if key in ('hqnr','d_lambda','d_s','jqm') else 'rr'][key]
            for key in ('hqnr','d_lambda','d_s','ergas'):value['delta_'+key]=pair['values'][key]
        rows.append({k:('' if v is None else v) for k,v in value.items()})
    return rows


def queue_run(root,run_id,attempt=0):
    summary=status_summary(run_id,root,attempt)
    if summary.get('complete'):
        from g23sens.postrun import verify_summary_for_upload
        summary=verify_summary_for_upload(run_id,root,attempt=attempt)
    baseline=_baseline(summary,root) if summary.get('complete') else None
    out=[]
    for payload in flatten_summary(summary,baseline):
        path=camp(root,payload['server'])/'outbox'/(payload['row_key']+'.json');old=read(path);digest=object_sha(payload)
        if old and old.get('payload_sha256')==digest:out.append(old);continue
        envelope=dict(schema='G23SENS_OUTBOX_v1',run_id=run_id,attempt=attempt,server=payload['server'],
            row_key=payload['row_key'],payload=payload,payload_sha256=digest,status='PENDING',api_attempts=0)
        atomic_json(path,envelope);out.append(envelope)
    return out


def _a1(row,col):
    name=''
    while col:col,rem=divmod(col-1,26);name=chr(65+rem)+name
    return name+str(row)


def _same(actual,expected):
    if expected=='':return actual in ('',None)
    if isinstance(expected,(int,float)) and not isinstance(expected,bool):
        try:return math.isfinite(float(actual)) and abs(float(actual)-expected)<=1e-12*max(1,abs(expected))
        except (TypeError,ValueError):return False
    return str(actual)==str(expected)


def apply_upsert(worksheet,payload):
    if (worksheet.title!=TABS.get(payload.get('server')) or payload.get('campaign_id')!=CAMPAIGN_ID
            or set(payload)!=set(HEADER)):raise ValueError('Only exact dedicated sensitivity tabs may be written')
    header=worksheet.row_values(1)
    if header and header!=list(HEADER):raise ValueError('Existing dedicated header differs; do not overwrite')
    if worksheet.col_count<len(HEADER):worksheet.add_cols(len(HEADER)-worksheet.col_count)
    if not header:worksheet.update(range_name='A1',values=[list(HEADER)],value_input_option='RAW')
    rows=worksheet.get_all_values(value_render_option='UNFORMATTED_VALUE');column=HEADER.index('row_key')
    matches=[i for i,row in enumerate(rows[1:],2) if len(row)>column and row[column]==payload['row_key']]
    if len(matches)>1:raise ValueError('Duplicate run/selection/attempt/provenance key')
    number=matches[0] if matches else max(2,len(rows)+1)
    if worksheet.row_count<number:worksheet.add_rows(number-worksheet.row_count)
    worksheet.update(range_name=_a1(number,1),values=[[payload[k] for k in HEADER]],value_input_option='RAW')
    decimal=set(METRICS)|{'delta_'+k for k in METRICS}|{'Train(h)','training_seconds',
        'q_ref_base','q_ref_used','tau_R_base','tau_R_used','alpha','beta','lambda_E','q_ref_scale','tau_R_scale','r_A'}
    worksheet.batch_format([dict(range=_a1(number,i),format={'numberFormat':{'type':'NUMBER','pattern':'0.0000'}})
        for i,key in enumerate(HEADER,1) if key in decimal])
    worksheet.batch_format([dict(range=_a1(number,HEADER.index(key)+1),
        format={'numberFormat':{'type':'SCIENTIFIC','pattern':'0.0E+00'}}) for key in ('U_peak_lr','A_peak_lr')])
    actual=worksheet.row_values(number,value_render_option='UNFORMATTED_VALUE')
    if any(not _same(actual[i] if i<len(actual) else '',payload[k]) for i,k in enumerate(HEADER)):
        raise ValueError('Unformatted Sheet raw-value readback differs')
    col=HEADER.index('readback_status')+1
    worksheet.update(range_name=_a1(number,col),values=[['READBACK_VERIFIED']],value_input_option='RAW')
    if worksheet.row_values(number,value_render_option='UNFORMATTED_VALUE')[col-1]!='READBACK_VERIFIED':
        raise ValueError('Readback status write failed')
    return dict(worksheet=worksheet.title,gid=int(worksheet.id),row=number,row_key=payload['row_key'],
        payload_sha256=object_sha(payload),readback_verified=True,at_utc=utcnow())


def _worksheet(root,server):
    import gspread
    from fh12.upload import legacy_constants
    constants=legacy_constants();client=gspread.service_account(filename=str(Path(root)/'gspread'/Path(constants.CRED).name))
    client.http_client.set_timeout(20);book=client.open(constants.SHEET)
    try:return book.worksheet(TABS[server])
    except gspread.WorksheetNotFound:return book.add_worksheet(title=TABS[server],rows=100,cols=len(HEADER))


def flush_outbox(root,server,writer=None,*,activated=False,limit=8):
    if not activated:raise PermissionError('Explicit sensitivity activation required before any Sheet access')
    if server not in TABS or type(limit) is not int or not 1<=limit<=100:raise ValueError('Bounded lane-local retry required')
    result={};attempts=0
    with locked(camp(root,server)/'outbox.lock'):
        for path in sorted((camp(root,server)/'outbox').glob('*.json')):
            envelope=read_json(path);payload=envelope.get('payload',{});key=envelope.get('row_key')
            if (envelope.get('schema')!='G23SENS_OUTBOX_v1' or envelope.get('server')!=server
                    or key!=path.stem or key!=payload.get('row_key') or payload.get('server')!=server
                    or envelope.get('payload_sha256')!=object_sha(payload)):
                result[path.stem]=dict(status='BLOCKED_INTEGRITY');continue
            if envelope.get('status')=='READBACK_VERIFIED':
                receipt=envelope.get('receipt',{})
                if (receipt.get('readback_verified') is not True or receipt.get('payload_sha256')!=envelope['payload_sha256']
                        or receipt.get('row_key')!=key or receipt.get('worksheet')!=TABS[server]):
                    result[path.stem]=dict(status='BLOCKED_INTEGRITY',reason='Readback receipt mismatch')
                continue
            if attempts>=limit:break
            try:
                fresh=queue_run(root,envelope['run_id'],attempt=envelope['attempt'])
                current=next(x for x in fresh if x['row_key']==key)
                if current['payload_sha256']!=envelope['payload_sha256']:envelope=current;payload=current['payload']
                envelope['api_attempts']+=1;atomic_json(path,envelope);attempts+=1
                worksheet=writer(server) if callable(writer) else writer if writer is not None else _worksheet(root,server)
                receipt=apply_upsert(worksheet,payload)
                envelope.update(status='READBACK_VERIFIED',receipt=receipt);result[key]=receipt
            except Exception as exc:
                envelope.update(status='PENDING',last_error=f'{type(exc).__name__}: {exc}')
                result[key]=dict(status='UPLOAD_PENDING',reason=envelope['last_error'],readback_verified=False)
            atomic_json(path,envelope)
    return result
