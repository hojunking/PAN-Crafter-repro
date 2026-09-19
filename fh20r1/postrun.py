"""Official FH20R1 same-checkpoint selections, using unchanged FH12 metric rules."""
from pathlib import Path
import yaml
from fh12.postrun import select_records,report_selection,profile_model
from fh12.data import build_dataset
from fh20r1.common import ROOT,atomic_json,read_json,object_sha,sha256,source_identity,utcnow,load_checkpoint_model
from fh20r1.plan import CAMPAIGN_ID,build_config,case_for


def process(run,device='cuda',upload=False,upload_only=False,root=ROOT):
    root=Path(root); case=case_for(run); wd=root/'work_dir'/run
    status_path=wd/'official/postrun_status.json'
    if (wd/'meta/reuse_reference.json').exists():
        if not upload_only: raise ValueError('Semantic reuse is upload-only; no repeated evaluation')
        from fh20r1.upload import validate_reuse
        reused,_bridge,_cfg,_grid,_release=validate_reuse(run,root)
        previous=read_json(status_path) if status_path.exists() else {}
        status=dict(status='REUSED_UPLOAD_PENDING',reused=True,source_run_id=reused['source_run_id'],
                    source_official_complete=True,source_actual_updates=50000,official_complete=False,
                    actual_updates=0,credited_new_seconds=0,sheet_uploaded=previous.get('sheet_uploaded',False),
                    reuse_identity_sha256=reused['identity_sha256'])
        atomic_json(status_path,status)
        return _upload(run,status,status_path,root) if upload else 0
    if upload_only:
        status=read_json(status_path)
        if not status.get('official_complete'): raise ValueError('Unfinished official FH20R1 results')
        return _upload(run,status,status_path,root) if upload else 0
    cfg=yaml.safe_load((wd/'meta/config.resolved.yaml').read_text())
    if cfg!=build_config(case): raise ValueError('FH20R1 resolved recipe drift')
    grid=read_json(wd/'official/raw_grid.json'); training=read_json(wd/'meta/training_status.json')
    if training.get('actual_updates')!=50000 or not training.get('training_complete') or not grid.get('complete'):
        raise ValueError('FH20R1 official selection requires 50K and all50 candidates')
    if (grid['config_sha256']!=object_sha(cfg) or grid['source_identity']!=source_identity(root)
        or grid.get('campaign_id')!=CAMPAIGN_ID or grid.get('run_id')!=run):
        raise ValueError('FH20R1 official evaluator source/config mismatch')
    data=read_json(root/cfg['fh20r1']['dataset_manifest'])
    if object_sha(data)!=grid['data_sha256']:
        raise ValueError('FH20R1 official dataset manifest mismatch')
    for row in grid['records']:
        folder=wd/'candidates'/str(row['update']); identity=read_json(folder/'identity.json')
        if (identity!=row['checkpoint_identity'] or identity['model_sha256']!=sha256(folder/'model.safetensors')
            or any(identity.get(key)!=grid.get(key) for key in ('config_sha256','data_sha256','source_identity'))):
            raise ValueError('FH20R1 candidate/metric identity mismatch')
    selected=select_records(grid['records'])
    context=dict(campaign_id=CAMPAIGN_ID,run_id=run,role=case.role,block_id=case.block_id,
        profile=case.profile,teacher_alias=case.teacher_alias,config_sha256=object_sha(cfg),
        data_sha256=grid['data_sha256'],source_identity=grid['source_identity'],normal_same_step_A_U=True)
    for key,label in [('raw_max','RAW_MAX'),('target','TARGET'),('exact50k','EXACT50K'),
                      ('rr_val_selected','RR_VAL_SELECTED'),('e_min_diag','E_MIN_DIAG50')]:
        extra={}
        if key=='target':
            t=selected[key]
            extra=dict(n_eligible=selected['n_eligible'],n_evaluated=50,target_status=selected['target_status'],
                hqnr_threshold=.9585,selector_order=['ergas','-scc','-psnr','step'],
                joint_pass=bool(t and t['rr']['ergas']<2.040),test_aware=True)
        if key=='e_min_diag': extra=dict(n_evaluated=50,n_candidates=50,development_oracle=True,independent_test=False)
        filename='target_selection' if key=='target' else key
        atomic_json(wd/'official'/f'{filename}.json',report_selection(label,selected[key],context,**extra))
    cost_path=wd/'official/profile.json'
    if cost_path.exists():
        cost=read_json(cost_path)
        if cost.get('config_sha256')!=object_sha(cfg) or cost.get('source_identity')!=grid['source_identity']:
            raise ValueError('FH20R1 profile provenance changed')
    else:
        data=read_json(root/cfg['fh20r1']['dataset_manifest'])
        dataset=build_dataset(data,'rr',root=root)
        model,_=load_checkpoint_model(cfg,wd/'candidates'/str(selected['raw_max']['update']),device,
                                      expected_source=grid['source_identity'])
        cost=profile_model(model,dataset,device)
        cost.update(config_sha256=object_sha(cfg),source_identity=grid['source_identity'])
        atomic_json(cost_path,cost)
        del model
    prior=read_json(status_path) if status_path.exists() else {}
    status=dict(status='EVAL_COMPLETE_UPLOAD_PENDING',official_complete=True,sheet_uploaded=prior.get('sheet_uploaded',False),
        config_sha256=object_sha(cfg),source_identity=grid['source_identity'],actual_updates=50000,
        completed_at_utc=prior.get('completed_at_utc',utcnow()),target_status=selected['target_status'])
    if prior.get('sheet_uploaded'):
        status.update(status='COMPLETE',upload_receipt=prior.get('upload_receipt'))
    atomic_json(status_path,status)
    return _upload(run,status,status_path,root) if upload else 0


def _upload(run,status,path,root):
    try:
        from fh20r1.upload import upload_run
        receipt=upload_run(run,root=root)
        status.update(status='REUSED_LINK_COMPLETE' if status.get('reused') else 'COMPLETE',sheet_uploaded=True,upload_receipt=receipt)
        status.pop('upload_error',None)
    except Exception as exc:
        status.update(status='REUSED_UPLOAD_PENDING' if status.get('reused') else 'EVAL_COMPLETE_UPLOAD_PENDING',
                      sheet_uploaded=False,upload_error=f'{type(exc).__name__}: {exc}')
    atomic_json(path,status)
    return 0
