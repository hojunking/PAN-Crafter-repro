"""Finite QB/GF2 study. No legacy handoff, queue mutation, or live Sheet writes."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

from pcrepro.common import (ROOT,atomic_json,read,read_json,read_config,immutable_json,
    object_sha,sha256,source_identity,apply_runtime_policy,run_dir,case_camp,locked,machine_lock,utcnow)
from pcrepro.plan import (FINITE_CAMPAIGN_ID,RECIPE_SHA256,FiniteCase,case_for,
    build_config,validate_config,verify_sources)


def folder(root=ROOT):return case_camp(root,FiniteCase('QB'))


def schedule(seeds):
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Explicit unique seeds required')
    return [FiniteCase(sensor,seed) for seed in seeds for sensor in ('QB','GF2')]


def require_idle():
    from pcrepro.handoff import gpu_processes
    pids=gpu_processes()
    if pids is None or any(pid!=os.getpid() for pid in pids):
        raise RuntimeError('GPU busy or unknown; existing processes are left running')


def require_s5(root):
    marker=Path(root)/'gspread/server.txt'
    if not marker.is_file() or marker.read_text().strip()!='s5':
        raise ValueError('s5 routing marker required: gspread/server.txt')


def manifest_for(sensor,root=ROOT):
    """Bind known original assets, not msfix; verify the exact FR20 tensors."""
    import h5py
    from pcrepro.data import CANONICAL_BANDS,validate_manifest
    root=Path(root);catalog=read_json(root/'ablr2/sensor_sources.json')['sensors'][sensor]
    provenance=catalog['source_provenance'];specs={}
    for split,item in catalog['splits'].items():
        path=item['path'];digest=item['sha256']
        if sensor=='QB' and split in ('train','val'):
            path=provenance['raw_'+split+'_path'];digest=provenance['raw_'+split+'_sha256']
        asset=root/path
        keys={'pan':'pan','ms':'ms',**({'lms':'lms'} if split=='fr' else {'gt':'gt'})}
        with h5py.File(asset,'r') as f:
            count=len(f['pan']);shapes={k:list(f[v].shape[1:]) for k,v in keys.items()}
        if split in ('rr','fr') and count!=20:raise ValueError('This study requires exactly RR20/FR20')
        ids=[f'{sensor}_{split}_H5_{i:06d}' for i in range(count)]
        specs[split]=dict(split=split,format='h5',path=path,sha256=digest,layout='NCHW',
            count=count,scene_ids=ids,keys=keys,shapes=shapes,origin='ORIGINAL_PANCOLLECTION')
        if split=='fr':
            evidence=read_json(asset.parent/'provenance.json')['files']
            expected={e['file']:e['sha256'] for e in evidence}
            refs=[]
            for i in range(20):
                name=f'Test(HxWxC)_{sensor.lower()}_data_fr{i+1}.mat'
                refs.append(dict(source_index=i,scene_id=ids[i],path=str(Path('data/PanCollection')/sensor/'full_examples_mat'/name),
                    sha256=expected[name],layout='HWC',keys=keys))
            specs[split]['paper_identity']=dict(origin='PANCOLLECTION_PAPER_MAT',mat_references=refs)
    value=dict(schema='PCREPRO_DATA_v1',dataset=sensor,bands=4,max_dn=1023 if sensor=='GF2' else 2047,
        units='DN',band_order=list(CANONICAL_BANDS[sensor]),splits=specs,
        source_catalog_sha256=sha256(root/'ablr2/sensor_sources.json'),
        provenance=provenance,scene_id_basis='Local H5 row identifiers, not author-issued geographic IDs')
    return validate_manifest(value,root,verify_files=True)


def prepare(root=ROOT,seeds=(2025,)):
    root=Path(root);verify_sources(root);apply_runtime_policy(root)
    cases=schedule(list(seeds));out=folder(root)
    existing=read(out/'registration.json')
    if existing:
        verified=verify_registration(root)
        if verified['cases']!=[c.to_dict() for c in cases]:raise ValueError('Prepared seed list is immutable')
        materialize_configs(verified,root)
        return verified
    bindings={}
    for sensor in ('QB','GF2'):
        print('Verifying original '+sensor+' train/val and RR20/FR20',flush=True)
        data=manifest_for(sensor,root)
        path=out/'datasets'/f'{sensor}.json';immutable_json(path,data);bindings[sensor]=str(path.resolve())
    bound=out/'bindings.json';immutable_json(bound,dict(datasets=bindings))
    value=dict(campaign_id=FINITE_CAMPAIGN_ID,recipe_sha256=RECIPE_SHA256,server='s5',
        cases=[c.to_dict() for c in cases],source_identity=source_identity(root),
        bindings_path=str(bound.resolve()),bindings_sha256=object_sha(read_json(bound)),
        data_sha256={s:object_sha(read_json(p)) for s,p in bindings.items()},
        automatic_legacy_handoff=False,automatic_upload=False,target_sheet='addtional-models')
    immutable_json(out/'registration.json',value)
    materialize_configs(value,root)
    return value


def materialize_configs(registration,root):
    bindings=read_json(registration['bindings_path'])['datasets']
    for item in registration['cases']:
        c=case_for(item['run_id'])
        cfg=build_config(c,bindings[c.dataset],registration['data_sha256'][c.dataset],registration['source_identity'])
        immutable_json(run_dir(c,root)/'meta/config.resolved.yaml',cfg)


def verify_registration(root=ROOT):
    verify_sources(root);apply_runtime_policy(root);v=read_json(folder(root)/'registration.json')
    cases=[case_for(row['run_id']) for row in v['cases']]
    if (not cases or len({c.run_id for c in cases})!=len(cases)
            or v.get('campaign_id')!=FINITE_CAMPAIGN_ID or v.get('server')!='s5'
            or any(not isinstance(c,FiniteCase) for c in cases)
            or [c.to_dict() for c in cases]!=v['cases']
            or v.get('automatic_legacy_handoff') is not False or v.get('automatic_upload') is not False
            or v.get('target_sheet')!='addtional-models'
            or v['source_identity']!=source_identity(root) or v['recipe_sha256']!=RECIPE_SHA256
            or object_sha(read_json(v['bindings_path']))!=v['bindings_sha256']):
        raise ValueError('Finite source/recipe/case/binding identity changed')
    binding=read_json(v['bindings_path'])['datasets']
    seeds=list(dict.fromkeys(c.seed for c in cases))
    if cases!=schedule(seeds) or set(binding)!={'QB','GF2'}:
        raise ValueError('Finite QB/GF2 schedule or bindings changed')
    if {s:object_sha(read_json(p)) for s,p in binding.items()}!=v['data_sha256']:
        raise ValueError('Finite data manifest changed')
    return v


def authorize_train(root,server,path):
    registration=verify_registration(root);cfg=read_config(path);c=validate_config(cfg,require_bound=True)
    status=read(folder(root)/'status.json');ready=read(folder(root)/'preflight'/f'{c.dataset}.json')
    if (not isinstance(c,FiniteCase) or server!='s5' or c.to_dict() not in registration['cases']
            or status.get('active_run')!=c.run_id or status.get('status')!='TRAINING'
            or not ready.get('complete') or ready.get('source_identity')!=registration['source_identity']
            or not ready.get('full_batch_smoke',{}).get('passed')
            or ready.get('full_batch_smoke',{}).get('pairs')!=48
            or ready.get('full_batch_smoke',{}).get('mars_batch')!=96
            or ready.get('data_sha')!=cfg['pcrepro']['data_sha']
            or cfg!=read_config(run_dir(c,root)/'meta/config.resolved.yaml')):
        raise ValueError('Finite controller did not authorize this exact run')
    return c


def export_rows(root=ROOT):
    registration=verify_registration(root);rows=[]
    from pcrepro.postrun import verify_summary_for_upload
    from pcrepro.checkpoint import selection_checkpoints
    for item in registration['cases']:
        c=case_for(item['run_id']);wd=run_dir(c,root)
        if not (wd/'official/summary.json').exists():continue
        summary=verify_summary_for_upload(c.run_id,root)
        paths=selection_checkpoints(wd);primary=summary['selections']['EXACT_50000']
        rows.append(dict(model='PAN-Crafter paper Eq11 reproduction',run_id=c.run_id,sensor=c.dataset,
            seed=c.seed,server='s5',pan_input='original',pan_reference='original',selection='EXACT_50000',
            checkpoint_sha256=primary['checkpoint_sha256'],checkpoint=str(paths['EXACT_50000']),
            secondary_checkpoint=str(paths['RR_VAL_ERGAS_MIN']),rr=primary['rr'],fr=primary['fr'],
            profile=summary['profile'],summary_path=str(wd/'official/summary.json'),
            summary_sha256=sha256(wd/'official/summary.json'),corrected_pan_evaluation=False))
    result=dict(schema='PCPAPER_ADDITIONAL_MODELS_TRANSFER_v1',spreadsheet_id='1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE',
        sheet_tab='addtional-models',rows=rows,complete=len(rows)==len(registration['cases']),
        uploaded=False,publication='s1: verify current tab layout, append rows and read back after transfer')
    atomic_json(folder(root)/'additional_models_rows.json',result);return result


def worker(root,stage,run):
    require_s5(root)
    registration=verify_registration(root);c=case_for(run)
    if not isinstance(c,FiniteCase) or c.to_dict() not in registration['cases']:raise ValueError('Unknown finite case')
    with locked(machine_lock('worker')):
        require_idle();stop=lambda:read(folder(root)/'control.json').get('command')=='STOP_NOW_SAFE'
        cfg=run_dir(c,root)/'meta/config.resolved.yaml'
        if stage=='preflight':
            from pcrepro.preflight import execute
            ready=execute(root,'s5',c.dataset,registration['bindings_path'],stopcheck=stop,campaign_dir=folder(root))
            if stop():raise InterruptedError('Safe preflight pause')
            ready['full_batch_smoke']=full_batch_smoke(c)
            atomic_json(folder(root)/'preflight'/f'{c.dataset}.json',ready);return 0
        if stage=='train':
            from pcrepro.training import train_run
            return train_run(cfg,root,resume=(run_dir(c,root)/'resume/index.json').exists())
        if stage=='postrun':
            from pcrepro.postrun import execute
            execute(cfg,root,stopcheck=stop);return 0
        raise ValueError('Unknown worker stage')


def full_batch_smoke(case):
    """No optimizer step; require the requested batch48/MARs96 to fit on s5."""
    import torch
    from pcrepro.model import build_model
    from pcrepro.training import mars_loss
    require_idle()
    model=build_model(4,seed=2025,max_pixel=case.max_dn).cuda().train()
    raw=torch.randint(0,case.max_dn+1,(48,1,64,64),device='cuda').float()
    pan=raw.mul(2/case.max_dn).sub(1)
    ms=torch.rand(48,4,16,16,device='cuda')*2-1;gt=torch.rand(48,4,64,64,device='cuda')*2-1
    torch.cuda.reset_peak_memory_stats()
    values=mars_loss(model,pan,ms,gt,raw);values['loss'].backward();torch.cuda.synchronize()
    if not bool(torch.isfinite(values['loss'])) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Full batch MARs smoke failed')
    result=dict(passed=True,pairs=48,mars_batch=96,optimizer_updates=0,
        scope='forward/backward only; optimizer state and checkpoint IO excluded',
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),torch=torch.__version__,cuda=torch.version.cuda)
    del values
    model.zero_grad(set_to_none=True);model.eval()
    from pcrepro.finite_cost import prepared
    from tools.additional_models_flops import count_forward
    mode=torch.ones(1,device='cuda',dtype=torch.long)
    result['native64_arithmetic_audit']=count_forward(lambda:prepared(model,pan[:1],ms[:1],mode))
    del model,raw,pan,ms,gt,mode
    torch.cuda.empty_cache();return result


def run(root=ROOT):
    root=Path(root);registration=verify_registration(root);out=folder(root)
    require_s5(root)
    with locked(machine_lock()),locked(out/'runner.lock'):
        require_idle()
        def pause(*_):atomic_json(out/'control.json',dict(command='STOP_NOW_SAFE',at_utc=utcnow()))
        previous={s:signal.signal(s,pause) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            for item in registration['cases']:
                c=case_for(item['run_id']);wd=run_dir(c,root)
                if read(out/'control.json').get('command')=='STOP_NOW_SAFE':return 75
                if (wd/'official/summary.json').exists():
                    from pcrepro.postrun import verify_summary_for_upload
                    verify_summary_for_upload(c.run_id,root);continue
                for stage in ('preflight','train','postrun'):
                    if read(out/'control.json').get('command')=='STOP_NOW_SAFE':return 75
                    if stage=='train' and read(wd/'meta/training_status.json').get('training_complete'):continue
                    require_idle()
                    state=dict(status='TRAINING' if stage=='train' else stage.upper(),active_run=c.run_id,at_utc=utcnow())
                    atomic_json(out/'status.json',state)
                    log=out/'logs'/f'{c.run_id}.{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
                    print(c.run_id+' '+stage,flush=True)
                    with log.open('a') as stream:
                        code=subprocess.call([sys.executable,'-u',str(root/'tools/pcrepro_qbgf2.py'),'worker',
                            '--stage',stage,'--run-id',c.run_id],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
                    if code:
                        atomic_json(out/'status.json',dict(state,status='PAUSED' if code==75 else 'FAILED',exit_code=code,log=str(log)))
                        return code
                export_rows(root)
            result=export_rows(root)
            if not result['complete']:raise ValueError('Finite result set incomplete')
            atomic_json(out/'status.json',dict(status='COMPLETE',active_run=None,runs=len(result['rows']),at_utc=utcnow()))
            return 0
        finally:
            for s,handler in previous.items():signal.signal(s,handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','prepare','run','status','pause','continue','worker','export'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[2025])
    parser.add_argument('--stage',choices=('preflight','train','postrun'))
    parser.add_argument('--run-id')
    args=parser.parse_args(argv)
    if args.action=='plan':return dict(campaign_id=FINITE_CAMPAIGN_ID,cases=[c.to_dict() for c in schedule(args.seeds)],training_started=False)
    if args.action=='prepare':return prepare(seeds=args.seeds)
    if args.action=='run':return dict(exit_code=run())
    if args.action=='status':return dict(registration=read(folder()/'registration.json'),status=read(folder()/'status.json'),control=read(folder()/'control.json'))
    if args.action in ('pause','continue'):
        value=dict(command='STOP_NOW_SAFE' if args.action=='pause' else 'CONTINUE',at_utc=utcnow())
        atomic_json(folder()/'control.json',value);return value
    if args.action=='worker':
        if not args.stage or not args.run_id:raise ValueError('Worker stage and run required')
        return dict(exit_code=worker(ROOT,args.stage,args.run_id))
    if args.action=='export':return export_rows()
