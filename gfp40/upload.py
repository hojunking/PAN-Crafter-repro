"""Dedicated GF2-P40 tabs only. Raw precision plus display formats and readback."""
import datetime as dt
import json
from pathlib import Path
from fh12.upload import a1,same_cell,legacy_constants
from reporting_extra.sensor_backfill import write_locks
from gfp40.common import (ROOT,read,read_json,read_config,run_dir,camp,object_sha,
                         source_identity,atomic_json,sha256,utcnow,resolved_path)
from gfp40.plan import (CAMPAIGN_ID,SHEET_TABS,case_for,validate_config,registry_sha256,
                       EXECUTION_POLICY,EXECUTION_POLICY_SHA256,CONFIRMATION_FAMILY)

METRICS={'rr':('ergas','scc','sam','psnr','ssim','q4','rmse','cc'),
         'fr':('hqnr','d_lambda','d_s','jqm')}


def flatten_summary(summary,case,root=ROOT):
    """Presentation never rounds the original metric values or mixes checkpoints."""
    wd=run_dir(case,root)
    cfg=read_config(wd/'meta/config.resolved.yaml') if (wd/'meta/config.resolved.yaml').is_file() else {}
    field=cfg.get('gfp40',{})
    started=read(wd/'meta/training_start_manifest.json')
    ref=read(wd/'meta/training_reference.json')
    local=read(camp(root,case.server)/'status.json').get('runs',{}).get(case.run_id,{})
    start=started.get('started_at_utc','')
    date=(dt.datetime.fromisoformat(start.replace('Z','+00:00')).astimezone(dt.timezone(dt.timedelta(hours=9))).isoformat()
          if start else '')
    values=dict(campaign_id=CAMPAIGN_ID,run_id=case.run_id,case_id=case.case_id,server=case.server,
        stage=case.stage,block_id=case.block_id,training_started_utc=start,training_started_kst=date,
        training_completed_utc=summary.get('completed_at_utc',''),
        parent_run_id=summary.get('parent_run_id',''),parent_step=case.parent_step,
        parent_model_sha=summary.get('parent_model_sha256',''),parent_teacher_sha=field.get('teacher_sha256',''),
        reference_id=case.reference_key,native_or_mixcal_id=field.get('mixed_calibration_manifest') or 'NATIVE',
        source_commit=summary.get('source_identity',{}).get('git_release',''),
        source_sha=summary.get('source_identity',{}).get('content_sha256',''),data_sha=summary.get('data_sha256',''),
        stream_seed=case.stream_seed,fresh_seed=case.model_seed,init_mode=case.init_mode,
        optimizer_reset=True,local_updates=summary.get('local_updates'),lifetime_updates=summary.get('lifetime_updates'),
        profile=summary.get('profile',case.profile),alpha=case.alpha,beta=case.beta,
        edge=case.lambda_edge,coefficient_ramp_updates=0,
        u_peak_lr=case.u_peak_lr,a_peak_lr=case.a_peak_lr,scheduler=case.scheduler,
        gamma_view=case.input_views,tau_R=ref.get('tau_R',field.get('tau_R','')),
        q_ref=ref.get('q_ref',field.get('q_ref','')),q_cache_sha=ref.get('q_cache_sha256',field.get('q_cache_sha256','')),
        training_hours=summary.get('training_hours',''),evaluation_hours=summary.get('evaluation_hours',''),
        io_hours=summary.get('io_hours',''),diagnostic_hours=summary.get('costs',{}).get('diagnostic_seconds',0.)/3600,
        calibration_hours=0.,shared_calibration_hours=read(camp(root,case.server)/'status.json').get('family_hours',0.),
        wall_hours=local.get('wall_hours',''),parent_compute=summary.get('parent_compute',{}),
        parent_training_hours=summary.get('parent_training_hours',''),lineage_training_hours=summary.get('lineage_training_hours',''),
        status='OFFICIAL_EVAL_COMPLETE',readback_status='READBACK_PENDING',test_aware=True)
    family=field.get('family',case.calibration_family)
    from gfp40.plan import FAMILIES,REGISTRY_REVISION
    distribution=FAMILIES.get(family,{})
    init=read(wd/'init_manifest.json')
    mixed=read(field['mixed_calibration_manifest']) if field.get('mixed_calibration_manifest') else {}
    data=read(field['dataset_manifest']) if field.get('dataset_manifest') else {}
    local_window_path=resolved_path(field.get('window_path') or camp(root,case.server)/'campaign_window.json',root)
    local_window=read(local_window_path)
    values.update(revision=REGISTRY_REVISION,parent_id=case.parent_id,arm=case.arm,
        source_design_run_id=getattr(case,'source_run_id',case.run_id),
        source_design_phase=getattr(case,'source_phase',case.phase),
        source_design_calibration_family=getattr(case,'source_calibration_family',case.calibration_family),
        distribution_id=family,distribution_tokens=distribution.get('tokens',[]),
        distribution_gammas=distribution.get('gammas',[]),
        execution_policy_sha256=EXECUTION_POLICY_SHA256,execution_policy=EXECUTION_POLICY,
        registry_sha256=registry_sha256(),confirmation_family=CONFIRMATION_FAMILY,
        local_window=local_window,local_window_path=str(local_window_path),
        local_window_sha256=object_sha(local_window) if local_window else '',
        local_t0_utc=local_window.get('t0_utc',''),independent_server_clock=True,
        source_archive_sha256=field.get('source_archive_sha256'),
        init_U_A_sha=init.get('hashes',{}),added_updates=summary.get('local_updates'),
        parent_updates=case.parent_updates,lifetime_student_updates=case.lifetime_student_updates,
        teacher_training_hours=summary.get('teacher_training_hours'),teacher_run_id=ref.get('teacher_run_id',''),
        scheduler_end=summary.get('scheduler_end'),
        gamma_counts=read(wd/'meta/training_status.json').get('gamma_exposure',{}),
        gamma_view=family if case.arm=='MIX' else 'NATIVE',
        calibration_id=mixed.get('calibration_id','NATIVE'),gamma_bank_sha256=mixed.get('gamma_bank_sha256',''),
        data_lp_sha={k:dict(data=v.get('sha256'),lp=v.get('lpan_sha256')) for k,v in data.get('splits',{}).items()},
        official_completed_at_utc=summary.get('official_completed_at_utc'),
        actual_updates=read(wd/'meta/training_status.json').get('actual_updates'),
        parent_student_training_hours=summary.get('parent_training_hours'),
        family_manifest_sha256=sha256(field['mixed_calibration_manifest']) if field.get('mixed_calibration_manifest') else '',
        endpoint_joint_goal=summary.get('endpoint_joint_goal',False),endpoint_strong_goal=summary.get('endpoint_strong_goal',False))
    for label in ('EXACT_FINAL','RR_VAL_SELECTED','RAW_AUX'):
        record=summary['selections'][label]
        values[label+' step']=record['update']
        values[label+' checkpoint SHA']=record.get('checkpoint_identity',{}).get('model_sha256','')
        values[label+' val ERGAS']=record['val_ergas']
        values[label+' candidate_count']=record.get('candidate_count',summary.get('n_candidates'))
        for domain,keys in METRICS.items():
            for key in keys:values[label+' '+key]=record[domain].get(key,'')
        for key in ('jqm_variant','jqm_status','jqm_reason'):
            values[label+' '+key]=record['fr'].get(key,'')
        for key in ('signed_mean','positive_fraction','reconstruction_max_abs_error'):
            values[label+' signed_Ds_'+key]=record['fr'].get('signed_ds',{}).get(key,'')
    values['notes']='Main EXACT_FINAL; secondary RR_VAL_SELECTED; RAW_AUX test-aware. Native RR20/FR20 DN1023, no masking. FT is weights-initialized, not fresh/exact parent-resume. JQM SRF-substitute.'
    return {k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else '' if v is None else v
            for k,v in values.items()}


def row_values(run,root=ROOT):
    from gfp40.postrun import summarize_grid,_verify_grid,verify_endpoint_progress
    case=case_for(run);wd=run_dir(case,root)
    cfg=read_config(wd/'meta/config.resolved.yaml');validate_config(cfg,require_bound=True)
    summary=read_json(wd/'official/summary.json');grid=read_json(wd/'official/raw_grid.json')
    data=read_json(cfg['gfp40']['dataset_manifest'])
    _verify_grid(cfg,grid,data,wd,source_identity(root))
    expected=summarize_grid(grid['records'],case.updates)
    if (summary.get('campaign_id')!=CAMPAIGN_ID or summary.get('run_id')!=case.run_id
        or summary.get('config_sha256')!=object_sha(cfg) or summary.get('complete') is not True
        or summary.get('selections')!=expected['selections']):
        raise ValueError('GFP40 upload summary differs from complete official checkpoint grid')
    if (summary.get('execution_policy')!=EXECUTION_POLICY
        or summary.get('execution_policy_sha256')!=EXECUTION_POLICY_SHA256
        or summary.get('registry_sha256')!=registry_sha256()
        or summary.get('confirmation_family')!=CONFIRMATION_FAMILY
        or any(k.startswith('selection_lock') for k in summary)):
        raise ValueError('GFP40 upload execution policy/registry/fixed family differs')
    if (summary.get('case_id')!=case.case_id or summary.get('server')!=case.server
        or summary.get('stage')!=case.phase or summary.get('profile')!=case.arm
        or summary.get('family')!=cfg['gfp40']['family']
        or summary.get('stream_seed')!=case.stream_seed
        or summary.get('parent_run_id')!=case.parent_run_id
        or summary.get('reference_key')!=case.reference_key):
        raise ValueError('GFP40 upload case/family/local-reference identity differs')
    from gfp40.policy import CampaignWindow
    window=CampaignWindow.from_dict(read_json(resolved_path(cfg['gfp40']['window_path'],root)))
    if window.to_dict().get('server')!=case.server:
        raise ValueError('GFP40 upload window belongs to a different server')
    training=read_json(wd/'meta/training_status.json')
    if not training.get('training_complete') or training.get('actual_updates')!=case.updates:
        raise ValueError('Partial training cannot be uploaded as complete endpoint')
    from gfp40.assets import _finite_state,_fullstate
    endpoint=wd/'candidates'/str(case.updates);identity=read_json(endpoint/'identity.json')
    state=_fullstate(endpoint/'training_state.pt',identity,_finite_state(endpoint/'model.safetensors',identity))
    verify_endpoint_progress(cfg,state,data)
    return flatten_summary(summary,case,root)


def apply_upsert(ws,values):
    server=values['server']
    if ws.title!=SHEET_TABS.get(server) or values['campaign_id']!=CAMPAIGN_ID:
        raise ValueError('Only this P40 server dedicated tab may be written')
    header=list(values)
    current=ws.row_values(1)
    if current and current!=header:raise ValueError('Existing P40 tab has another schema; refusing header overwrite')
    if ws.col_count<len(header):ws.add_cols(len(header)-ws.col_count)
    if not current:ws.update(range_name='A1',values=[header],value_input_option='RAW')
    rows=ws.get_all_values(value_render_option='UNFORMATTED_VALUE')
    run_col=header.index('run_id');campaign_col=header.index('campaign_id')
    matches=[i+1 for i,row in enumerate(rows[1:],1) if len(row)>run_col and row[run_col]==values['run_id']]
    if len(matches)>1:raise ValueError('Duplicate P40 run rows; do not overwrite ambiguous results')
    row=matches[0] if matches else max(2,len(rows)+1)
    if matches and rows[row-1][campaign_col]!=CAMPAIGN_ID:raise ValueError('Row belongs to another campaign')
    if ws.row_count<row:ws.add_rows(row-ws.row_count)
    ws.update(range_name=a1(row,1),values=[[values[k] for k in header]],value_input_option='RAW')
    formats=[]
    decimal_labels={label+' '+key for label in ('EXACT_FINAL','RR_VAL_SELECTED','RAW_AUX') for keys in METRICS.values() for key in keys}
    for i,key in enumerate(header,1):
        if key in decimal_labels or ' signed_Ds_' in key or key.endswith('hours') or key.endswith('val ERGAS'):
            formats.append(dict(range=a1(row,i),format={'numberFormat':{'type':'NUMBER','pattern':'0.0000'}}))
    if formats:ws.batch_format(formats)
    actual=ws.row_values(row,value_render_option='UNFORMATTED_VALUE')
    if any(not same_cell(actual[i] if i<len(actual) else '',values[key]) for i,key in enumerate(header)):
        raise ValueError('P40 raw-value readback differs')
    status_col=header.index('readback_status')+1
    ws.update(range_name=a1(row,status_col),values=[['READBACK_VERIFIED']],value_input_option='RAW')
    if ws.row_values(row,value_render_option='UNFORMATTED_VALUE')[status_col-1]!='READBACK_VERIFIED':
        raise ValueError('P40 upload status readback failed')
    return dict(worksheet=ws.title,gid=int(ws.id),row=row,readback_verified=True,payload_sha256=object_sha(values))


def upload_run(run,root=ROOT,*,activated=False,worksheet=None):
    if not activated:raise PermissionError('Explicit GFP40 activation required for live Sheet writes')
    case=case_for(run);values=row_values(run,root)
    if worksheet is None:
        import gspread
        constants=legacy_constants()
        client=gspread.service_account(filename=str(Path(root)/'gspread'/Path(constants.CRED).name))
        client.http_client.set_timeout(20)
        book=client.open(constants.SHEET)
        try:worksheet=book.worksheet(SHEET_TABS[case.server])
        except gspread.WorksheetNotFound:
            worksheet=book.add_worksheet(title=SHEET_TABS[case.server],rows=100,cols=max(120,len(values)))
    with write_locks(root):receipt=apply_upsert(worksheet,values)
    receipt.update(campaign_id=CAMPAIGN_ID,run_id=case.run_id,uploaded_at_utc=utcnow())
    atomic_json(run_dir(case,root)/'official/upload_receipt.json',receipt)
    return receipt
