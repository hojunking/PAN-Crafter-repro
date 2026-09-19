"""Automatic checks after verified historical import, before any FH20R1 training."""
from pathlib import Path
import gc
import shutil
import numpy as np
import torch
import yaml
from fh12.model import build_model,frontend_self_test,state_hash
from fh12.losses import student_losses,routed_student_backward,teacher_loss
from tools.fh12_preflight import coupled_model_checks,numerical_routing_checks,evaluator_smoke,_smoke_batch
from fh20r1.common import ROOT,atomic_json,object_sha,read_json,sha256,source_identity,utcnow
from fh20r1.plan import (CAMPAIGN_ID,SOURCE_PLAN,SOURCE_CASES,CASES,GRID_STEPS,build_config,
                        cases_for,active_cases,blocks_for,REFERENCE_ASSETS)
from tools.gen_fh20r1_configs import generate


def audit_definitions(root):
    import csv,json
    root=Path(root)
    with (root/SOURCE_CASES).open(encoding='utf-8-sig',newline='') as stream: rows=list(csv.DictReader(stream))
    lookup={r['case_id']:r for r in rows}
    if len(rows)!=145 or len(lookup)!=145 or set(lookup)!={c.run_id for c in CASES}: raise ValueError('Supplied CSV case set differs')
    for c in CASES:
        r=lookup[c.run_id]; cfg=build_config(c)
        checks=dict(kind='TRAIN',role=c.role,server=c.server_id,tier=c.tier,layout=c.input_layout,width=str(c.width),
                    input_channels=str({'P0':9,'PL':10,'PH':10,'PLH':11}[c.input_layout]),
                    teacher_ref_alias=c.teacher_alias if c.role=='S' else '',
                    teacher_run_id=c.teacher_run_id if c.role=='S' else '',
                    eligible_when=c.eligible_when,profile=c.profile,teacher_seed=str(c.teacher_seed),student_seed=str(c.student_seed or ''),
                    total_updates='50000',planned_config_path=f'config/{c.run_id}.yaml')
        if any(r[k]!=v for k,v in checks.items()) or json.loads(r['depth'])!=list(c.depth) or json.loads(r['block_ids'])!=[c.block_id]:
            raise ValueError(f'Supplied CSV definition drift: {c.run_id}')
        if c.role=='S' and c.teacher_alias!='N2PL' and r['teacher_checkpoint_sha256']!=REFERENCE_ASSETS[c.teacher_alias]['expected_sha256']:
            raise ValueError(f'Supplied CSV Teacher whole SHA drift: {c.run_id}')
        for k,v in [('lr_U_peak',cfg['learning_rate']),('lr_A_peak',cfg['fh20r1']['aligner_lr']),
                    ('alpha',1.),('beta',.1),('lambda_E',0. if c.role=='T' else .002),
                    ('A_multiplier_after_10000',1/3 if c.profile=='A10' else 1.)]:
            if r[k] and float(r[k])!=v: raise ValueError(f'CSV coefficient drift: {c.run_id}/{k}')
    changed=generate(root,check=True)
    if changed: raise ValueError(f'Generated FH20R1 release mismatch: {changed[:5]}')
    return dict(n_cases=145,n_blocks=51,source_plan_sha256=sha256(root/SOURCE_PLAN),
                source_cases_sha256=sha256(root/SOURCE_CASES),numeric_values_verified=True)


def disk_estimate(root,server,reused=None):
    """Conservative CORE-only reservation. Reserves are rechecked before admission."""
    counts={}; estimates={}
    for c in cases_for(server):
        key=(c.input_layout,c.width,c.depth)
        if key not in counts:
            model,_=build_model(c.input_layout,c.width,list(c.depth),c.seed,role='T')
            counts[key]=sum(p.numel()*p.element_size() for p in model.parameters()); del model
        # 50 model candidates, three restart bundles (weights + model/Adam state),
        # exact50K candidate fullstate, last and atomic publication headroom.
        estimates[c.run_id]=counts[key]*80+1024**2*20
    needed=0
    reused=(reused or {}).get('runs',{})
    for c in active_cases(server,include_reserve=False):
        status_path=Path(root)/'work_dir'/c.run_id/'official/postrun_status.json'
        done=status_path.is_file() and read_json(status_path).get('official_complete',False)
        equivalent=reused.get(c.run_id,{})
        if not done and not (equivalent.get('validated') and equivalent.get('official_complete')):
            needed+=estimates[c.run_id]
    free=shutil.disk_usage(root).free
    return dict(free_bytes=free,core_additional_estimate_bytes=needed,minimum_safety_bytes=2*1024**3,
                passed=free>=needed+2*1024**3,per_run_estimate_bytes=estimates,
                estimate_policy='80 parameter-byte equivalents + 20MiB metadata/run; CORE sum, all-case admission estimates')


def _numeric_signature(cfg,teacher_sha):
    f=cfg.get('fh20r1',cfg.get('fh12',{}))
    keys=('seed','num_iter','num_warmup','batch_size','mixed_precision','learning_rate','optimizer','weight_decay',
          'betas','eps','lr_scheduler','train_feeder_args','model_args','num_bands','max_pixel')
    return dict(recipe={k:cfg[k] for k in keys},input_layout=f['input_layout'],profile=f.get('profile','BASE'),
                aligner_lr=f['aligner_lr'],alpha=f['alpha'],beta=f['beta'],lambda_E=f['lambda_E'],
                corruption_seed=f['corruption_seed'],init_policy=f['init_policy'],teacher_sha=teacher_sha)


def find_reusable_runs(root,server,bridge):
    """Only complete local same-method runs, proven by full signatures and bytes."""
    root=Path(root); wanted={}; reused={}; examined=0
    def numerical(name):
        return name.startswith(('fh12/','model/','pa/','tools/metrics/')) or name in ('tools/eval_dlpan.py','tools/repair_lpan.py')
    numerical_files={name:value for name,value in source_identity(root)['files'].items() if numerical(name)}
    if not numerical_files: raise ValueError('Semantic reuse requires numerical source identity coverage')
    for c in cases_for(server):
        if c.role=='S' and c.teacher_alias==bridge['alias']:
            wanted[object_sha(_numeric_signature(build_config(c),bridge['teacher_checkpoint_sha256']))]=c
    data_sha=object_sha(read_json(root/'work_dir/_fh12'/server/'dataset_manifest.json'))
    for path in (root/'work_dir').glob('*/meta/config.resolved.yaml'):
        cfg=yaml.safe_load(path.read_text())
        if cfg.get('trainer') not in ('fh12','fh20r1'): continue
        f=cfg.get('fh20r1',cfg.get('fh12',{}))
        if f.get('role')!='S' or f.get('server_id')!=server: continue
        wd=path.parent.parent
        if not (wd/'init_manifest.json').is_file(): continue
        init=read_json(wd/'init_manifest.json')
        source_teacher=init.get('teacher_ref_sha')
        source_bridge=None
        if cfg.get('trainer')=='fh20r1':
            p=root/f['reference_bridge']
            if not p.is_file(): continue
            source_bridge=read_json(p)
            source_teacher=source_bridge.get('teacher_checkpoint_sha256')
        if source_teacher!=bridge['teacher_checkpoint_sha256']: continue
        c=wanted.get(object_sha(_numeric_signature(cfg,source_teacher)))
        if c is None or c.run_id==wd.name: continue
        examined+=1
        status_path=wd/'official/postrun_status.json'
        if not status_path.exists(): continue
        status=read_json(status_path)
        if not status.get('official_complete') or status.get('actual_updates')!=50000: continue
        grid=read_json(wd/'official/raw_grid.json')
        if grid['data_sha256']!=data_sha or grid['config_sha256']!=object_sha(cfg): raise ValueError('Semantic reuse identity mismatch')
        if status.get('config_sha256')!=grid['config_sha256'] or status.get('source_identity')!=grid['source_identity']:
            raise ValueError('Semantic reuse completion/grid source mismatch')
        from fh12.postrun import select_records
        selected=select_records(grid['records'])
        # Numerical source files must match consumer; ignore execution release
        # tags, never rewrite historical provenance to current values.
        if any(grid['source_identity'].get('files',{}).get(name)!=value for name,value in numerical_files.items()):
            raise ValueError('Semantic reuse numerical source coverage/revision differs')
        for name,expected in grid['source_identity']['files'].items():
            if numerical(name) and sha256(root/name)!=expected:
                raise ValueError('Semantic reuse numerical source changed')
        expected_ref=object_sha(bridge['origin_reference'])
        if source_bridge is not None:
            if source_bridge.get('origin_reference')!=bridge['origin_reference']:
                raise ValueError('Semantic reuse calibration reference differs')
            expected_ref=object_sha(source_bridge)
        for r in grid['records']:
            folder=wd/'candidates'/str(r['update']); ident=read_json(folder/'identity.json')
            if ident!=r['checkpoint_identity'] or sha256(folder/'model.safetensors')!=ident['model_sha256']:
                raise ValueError('Semantic reuse candidate bytes mismatch')
            if (ident.get('reference_sha256')!=expected_ref or
                any(ident.get(k)!=grid.get(k) for k in ('config_sha256','data_sha256','source_identity'))):
                raise ValueError('Semantic reuse candidate context/calibration mismatch')
        for filename,key in [('raw_max','raw_max'),('target_selection','target'),('exact50k','exact50k'),
                             ('rr_val_selected','rr_val_selected'),('e_min_diag','e_min_diag')]:
            report=read_json(wd/'official'/f'{filename}.json'); expected=selected[key]
            if expected is None:
                if report.get('selection','missing') is not None or report.get('target_status')!='no_eligible':
                    raise ValueError('Semantic reuse no-eligible source report drift')
            elif (report.get('step')!=expected['update'] or
                  any(report.get(k)!=expected.get(k) for k in ('rr','fr','val_ergas','checkpoint_identity'))):
                raise ValueError('Semantic reuse source selected metrics drift')
        if c.run_id in reused: raise ValueError('Ambiguous equivalent completed runs; do not cherry-pick one')
        reused[c.run_id]=dict(validated=True,official_complete=True,source_run_id=wd.name,
            identity_sha256=object_sha(dict(config=cfg,grid=grid)),credited_new_seconds=0,
            source_official_dir=str(wd/'official'),normal_same_step_A_U=True)
    return dict(runs=reused,examined_potential_equivalents=examined,source='local full-identity artifacts',
                remote_sheet_only_rows='not reused without local model/official identity artifacts')


def batch_smoke(server,bridge,device='cuda',root=ROOT):
    from fh20r1.references import load_reference
    dev=torch.device(device); gpu=dev.type=='cuda'
    if gpu and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback')
    teacher,_,checked,_q=load_reference(bridge['alias'],server,root,device)
    if object_sha(bridge)!=object_sha(checked): raise ValueError('Bridge changed before smoke')
    c=max((c for c in cases_for(server) if c.role=='S'),key=lambda c:(c.width,sum(c.depth),len(c.input_layout)))
    if gpu:
        torch.cuda.reset_peak_memory_stats(dev)
        gt,ms,pan,lp=_smoke_batch(bridge['dataset_manifest'],device,48)
        width,depth=c.width,list(c.depth)
    else:
        g=torch.Generator().manual_seed(198)
        gt,ms,pan,lp=[torch.randn(s,generator=g)*.2 for s in ((2,8,32,32),(2,8,8,8),(2,1,32,32),(2,1,8,8))]
        width,depth=8,[1,1,1]
    original=state_hash(teacher.state_dict())
    model,_=build_model(c.input_layout,width,depth,c.seed,role='S',teacher_aligner_state=teacher.aligner.state_dict())
    model.to(device).train()
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-4},
                          {'params':model.aligner.parameters(),'lr':3e-6}],weight_decay=.01,betas=(.9,.999),eps=1e-8)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        out=model(pan,ms,lp)
        with torch.no_grad(): target=teacher(pan,ms,lp)
        losses=student_losses(out,target,gt,bridge['tau_R'],torch.full((len(pan),),.5,device=device))
        if not torch.isfinite(losses['L_U']) or not torch.isfinite(losses['L_A']): raise FloatingPointError('Nonfinite batch smoke')
        routed_student_backward(model,losses)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError('Nonfinite smoke gradients')
        opt.step()
    if state_hash(teacher.state_dict())!=original: raise ValueError('Frozen Teacher mutated in smoke')
    report=dict(passed=True,actual_batch48_smoke_status='PASS' if gpu else 'NOT_RUN_CPU_DIAGNOSTIC',
        device=str(dev),batch_size=len(pan),width=width,depth=depth,layout=c.input_layout,
        peak_allocated_mb=torch.cuda.max_memory_allocated(dev)/2**20 if gpu else None,disposable=True,
        parameters_reused_in_actual_training=False)
    del opt,model,teacher,out,target,losses,gt,ms,pan,lp
    gc.collect()
    if gpu: torch.cuda.empty_cache()
    return report


def run_preflight(server,device='cuda',root=ROOT,check_only=False):
    root=Path(root); camp=root/'work_dir/_fh20r1'/server
    report=dict(campaign_id=CAMPAIGN_ID,server_id=server,checked_at_utc=utcnow(),complete=False,preflight_pass=False)
    report['registry']=audit_definitions(root)
    if check_only:
        from fh20r1.references import validate_origin
        _cfg,origin,_data,_q,_paths=validate_origin('F'+server[1:],server,root)
        report.update(status='CHECK_ONLY_NO_ACTIVATION',origin_teacher_sha256=origin['teacher_checkpoint_sha256'],
                      runtime_numerics_executed=False)
        return report
    budget=read_json(camp/'campaign_budget.json')
    if budget.get('campaign_id')!=CAMPAIGN_ID or not budget.get('actual_start_authorized') or budget.get('device')!=device:
        raise ValueError('FH20R1 preflight requires explicit local activation/device')
    try:
        from fh20r1.references import load_reference
        teacher,cfg,bridge,_q=load_reference('F'+server[1:],server,root,device='cpu'); del teacher
        report['origin_reference_bridge_sha256']=object_sha(bridge)
        report['source_identity']=source_identity(root)
        reuse=find_reusable_runs(root,server,bridge)
        atomic_json(camp/'reuse_manifest.json',reuse)
        report['disk']=disk_estimate(root,server,reuse)
        if not report['disk']['passed']: raise RuntimeError('Insufficient disk for mandatory CORE candidate/fullstates; originals preserved')
        report['frontend']=frontend_self_test(device=device)
        report['coupled_init']=coupled_model_checks(server)
        report['loss_routing']=numerical_routing_checks()
        report['batch_smoke']=batch_smoke(server,bridge,device,root)
        from fh20r1.smoke import n2_teacher_smoke
        report['n2_teacher_smoke']=n2_teacher_smoke(server,bridge,device,root)
        report['evaluator_smoke']=evaluator_smoke(bridge['dataset_manifest'])
        report.update(complete=True,preflight_pass=True,status='PASS',historical_step0_parity_status=bridge['parity'])
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        atomic_json(camp/'preflight_report.json',report); raise
    atomic_json(camp/'preflight_report.json',report)
    return report
