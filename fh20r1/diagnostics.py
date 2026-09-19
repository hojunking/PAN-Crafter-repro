"""Prespecified frozen A/U factorial diagnostics, never official model candidates."""
from pathlib import Path
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from fh12.data import build_dataset
from fh12.evaluation import FRMetrics,evaluate_model
from fh12.plan import cases_for as legacy_cases
from fh20r1.common import ROOT,atomic_json,read_json,object_sha,sha256,source_identity,utcnow,load_checkpoint_model
from fh20r1.plan import S1_SUPPORT,S1_FALLBACK

STEPS=(10100,24240,50000)


class DiagnosticUnavailable(FileNotFoundError):
    """Optional old Student artifacts absent; never malformed/core reference data."""


def _finite_metrics(row):
    for split, keys in (('fr',('hqnr','d_s','d_lambda')),('rr',('ergas','scc','psnr','sam','q8','ssim'))):
        if row[split].get('n_scenes') != 20 or len(row[split].get('per_scene',[])) != 20:
            raise ValueError('A/U diagnostics require complete paired RR20/FR20 scenes')
        for item in [row[split]] + row[split]['per_scene']:
            if not all(math.isfinite(float(item[key])) for key in keys):
                raise ValueError('Nonfinite A/U diagnostic metrics')
    if not row['rr'].get('official_complete'):
        raise ValueError('A/U diagnostic RR is not official-complete')


def _artifact_hashes(paths):
    return {str(Path(path).resolve()):sha256(path) for path in paths}


def _verify_dependencies(dependencies):
    for path,digest in dependencies.items():
        if not Path(path).is_file() or sha256(path)!=digest:
            raise ValueError(f'Diagnostic source/cache bytes changed: {path}')


def diagnostic_runs(server):
    cases=[c for c in legacy_cases(server) if c.role=='S']
    wanted={'s1':{('P0',104,(1,2,1)),('PLH',104,(1,2,1))},
            's2':set(),'s3':{('PH',104,(1,2,1)),('PH',104,(1,2,2))},
            's4':{('PLH',104,(1,2,2)),('PLH',112,(1,2,1))},
            's5':{('PLH',104,(1,2,1)),('PLH',104,(1,2,2))}}[server]
    return tuple(c.run_id for c in cases if (c.input_layout,c.width,c.depth) in wanted)


def decide_s1(crosses):
    comparisons=[]; seen=set()
    for source in crosses:
        if source['source_run'] in seen or source['layout'] not in ('P0','PLH'):
            raise ValueError('Duplicate or wrong-layout s1 diagnostic evidence')
        seen.add(source['source_run'])
        if source.get('diagonal_verified') is not True:
            raise ValueError('S1 branch cannot consume unverified diagonal evidence')
        rows={(r['step_A'],r['step_U']):r for r in source['records']}
        if len(rows)!=len(source['records']): raise ValueError('Duplicate A/U pair in diagnostic matrix')
        base=rows.get((50000,50000))
        if base is None: continue
        _finite_metrics(base)
        for a in (10100,24240):
            row=rows.get((a,50000))
            if row is None: continue
            _finite_metrics(row)
            delta_h=row['fr']['hqnr']-base['fr']['hqnr']
            delta_ds=row['fr']['d_s']-base['fr']['d_s']
            delta_e=row['rr']['ergas']-base['rr']['ergas']
            bp=base['fr']['per_scene']; rp=row['fr']['per_scene']
            if len(bp)!=20 or len(rp)!=20: raise ValueError('S1 diagnosis requires FR20 paired scenes')
            improved=sum(x['d_s']<y['d_s'] for x,y in zip(rp,bp))
            support=delta_h>=.001 and delta_ds<=-.001 and delta_e<=.003 and improved>=12
            comparisons.append(dict(source_run=source['source_run'],layout=source['layout'],step_A=a,
                step_U=50000,delta_hqnr=delta_h,delta_ds=delta_ds,delta_ergas=delta_e,
                improved_ds_scenes=improved,support=bool(support)))
    supported=[x for x in comparisons if x['support']]
    evidence_complete=len(comparisons)==4 and {x['layout'] for x in comparisons}=={'P0','PLH'}
    yes=evidence_complete and len(supported)>=2 and any(x['layout']=='PLH' for x in supported)
    return dict(branch=S1_SUPPORT if yes else S1_FALLBACK,support_count=len(supported),comparisons=comparisons,
        decision_evidence_complete=evidence_complete,
        reason='prespecified_thresholds_met' if yes else 'opposing_or_insufficient_evidence; not proof A is uninvolved',
        test_aware=True,independent_test=False)


def _numerical_origin(identity,root):
    source=identity['source_identity']; files=source.get('files',{})
    if source.get('content_sha256')!=object_sha(files): raise ValueError('Historical source manifest corrupt')
    # Consumer reuses these exact files. Git release may differ; numerical
    # source bytes may not. The separate Teacher bridge also executes parity.
    for path,expected in files.items():
        if Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('Unsafe historical numerical source path')
        if sha256(root/path)!=expected: raise ValueError(f'Historical diagnostic source drift: {path}')


def _native_c(model,datasets,device):
    """A-only native c, no extra U-Net/metric pass; original MS-frame inputs."""
    rows={}
    with torch.no_grad():
        for split,side in (('rr',256),('fr',512)):
            dataset=datasets[split]
            if len(dataset)!=20: raise ValueError('Native offset diagnosis requires RR20/FR20')
            values=[]
            for index in range(20):
                batch=dataset[index]
                ms,pan=batch[-4].unsqueeze(0).to(device),batch[-2].unsqueeze(0).to(device)
                if tuple(pan.shape)!=(1,1,side,side): raise ValueError('Wrong diagnostic native PAN geometry')
                base=F.interpolate(ms.float(),scale_factor=4,mode='bicubic',align_corners=False)
                c=model.predict_delta(pan,base)
                if c.shape!=(1,2) or not bool(torch.isfinite(c).all()): raise ValueError('Invalid diagnostic native c')
                values.append(dict(scene_id=index,dy=float(c[0,0]),dx=float(c[0,1])))
            rows[split]=values
    return rows


def _validate_native_c(rows):
    for split in ('rr','fr'):
        values=rows[split]
        if len(values)!=20 or [x['scene_id'] for x in values]!=list(range(20)):
            raise ValueError('Incomplete/out-of-order native per-scene offsets')
        if not all(math.isfinite(float(x[k])) for x in values for k in ('dy','dx')):
            raise ValueError('Nonfinite native per-scene offsets')


def _capture_cross_c(model,evaluate):
    """Collect c from the existing official forwards, without rerunning U."""
    rows={'rr':[],'fr':[]}
    def hook(_model,inputs,out):
        side=int(inputs[0].shape[-1]); split={256:'rr',512:'fr'}.get(side)
        if split is None: raise ValueError('Unexpected shape in full-image A/U evaluation')
        delta=out['delta'].detach().cpu()
        if delta.ndim!=2 or delta.shape[1]!=2 or not bool(torch.isfinite(delta).all()):
            raise ValueError('Invalid A/U output native c')
        for c in delta:
            rows[split].append(dict(scene_id=len(rows[split]),dy=float(c[0]),dx=float(c[1])))
    handle=model.register_forward_hook(hook)
    try: result=evaluate()
    finally: handle.remove()
    _validate_native_c(rows)
    return result,rows


def cross_run(run,server,data,device,root=ROOT,time_limit=None):
    root=Path(root); wd=root/'work_dir'/run; output=root/'work_dir/_fh20r1'/server/'diagnostics/au_cross'/run
    required=[wd/'meta/config.resolved.yaml',wd/'official/raw_grid.json']
    required += [wd/'candidates'/str(step)/name for step in STEPS for name in ('identity.json','model.safetensors')]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:
        raise DiagnosticUnavailable('Optional historical Student diagnostic assets absent: '+', '.join(missing))
    cfg=yaml.safe_load((wd/'meta/config.resolved.yaml').read_text())
    if (cfg['fh12']['server_id']!=server or cfg['fh12']['role']!='S' or Path(cfg['work_dir']).name!=run):
        raise ValueError('Wrong local diagnostic source')
    grid=read_json(wd/'official/raw_grid.json')
    data_path=root/'work_dir/_fh12'/server/'dataset_manifest.json'
    original_data=read_json(data_path)
    if grid['config_sha256']!=object_sha(cfg) or grid['data_sha256']!=object_sha(original_data):
        raise ValueError('Historical diagnostic config/data differs')
    for key in ('recipe','augmentation'):
        if data.get(key)!=original_data.get(key): raise ValueError('Diagnostic reference data recipe differs')
    data_files={}
    for split in ('train','val','rr','fr'):
        for key in ('sha256','lpan_sha256','sample_order_sha256'):
            if data['splits'][split][key]!=original_data['splits'][split][key]:
                raise ValueError('Diagnostic and reference dataset identities differ')
        item=data['splits'][split]
        data_files[item['dataroot']]=item['sha256']; data_files[item['lpan_path']]=item['lpan_sha256']
    _verify_dependencies(data_files)
    dependencies=dict(_artifact_hashes(required+[data_path]),**data_files)
    identities={}; records=[]; historical={r['update']:r for r in grid['records']}
    if len(historical)!=len(grid['records']): raise ValueError('Duplicate historical candidate step')
    release=source_identity(root)
    for step in STEPS:
        folder=wd/'candidates'/str(step); ident=read_json(folder/'identity.json')
        if ident['config_sha256']!=object_sha(cfg) or ident['update']!=step or ident['model_sha256']!=sha256(folder/'model.safetensors'):
            raise ValueError('Historical A/U candidate identity mismatch')
        _numerical_origin(ident,root); identities[step]=ident
        original=historical[step]
        if original['checkpoint_identity']!=ident or not original['rr'].get('official_complete'):
            raise ValueError('Historical diagonal metric identity mismatch')
        _finite_metrics(original)
        records.append(dict(step_A=step,step_U=step,rr=original['rr'],fr=original['fr'],
            shift=original['shift'],diagonal_verified='original_official_cache_full_identity',
            source_A_sha256=ident['model_sha256'],source_U_sha256=ident['model_sha256'],cache_reused=True))
    datasets=None; engine=None
    # Native A offsets were not retained per scene by FH12's metric cache.
    # Measure only A (never repeat expensive diagonal U/metric inference).
    for row in records:
        step=row['step_A']; path=output/f'A{step}_native_c.json'
        identity=dict(source_run=run,step_A=step,source_A_sha256=identities[step]['model_sha256'],
            config_sha256=object_sha(cfg),data_sha256=grid['data_sha256'],consumer_source_identity=release)
        if path.exists():
            cached=read_json(path)
            if cached.get('identity')!=identity: raise ValueError('Native c cache identity changed')
            native=cached['native_c']; _validate_native_c(native)
        else:
            if time_limit is not None and time.monotonic()>=time_limit:
                row['native_c_status']='not_measured_within_diagnostic_budget'; continue
            if datasets is None: datasets={s:build_dataset(data,s,root=root) for s in ('rr','fr')}
            started=utcnow()
            model,_=load_checkpoint_model(cfg,wd/'candidates'/str(step),device)
            model.eval().requires_grad_(False)
            native=_native_c(model,datasets,device); del model
            ended=utcnow()
            atomic_json(path,dict(identity=identity,native_c=native,diagnostic_only=True,
                A_only=True,extra_U_forward_count=0,timing=dict(start_utc=started,end_utc=ended)))
            from fh20r1.ledger import record_interval
            record_interval(root,server,interval_id=f'AUC:{run}:{step}',kind='diagnostic',start_utc=started,
                end_utc=ended,run_id=run,evidence=str(path.relative_to(root)))
        row.update(native_c=native,native_c_status='verified_A_only_native_input',native_c_source_sha256=identities[step]['model_sha256'])
        dependencies[str(path.resolve())]=sha256(path)
    # Prioritize the four prespecified s1 branch comparisons; other six matrix
    # positions still run if time remains. No performance-driven ordering.
    pairs=sorted(((a,u) for a in STEPS for u in STEPS if a!=u),key=lambda pair:(pair[1]!=50000,pair))
    for a,u in pairs:
            path=output/f'A{a}_U{u}.json'
            identity=dict(source_run=run,step_A=a,step_U=u,source_A_sha256=identities[a]['model_sha256'],
                source_U_sha256=identities[u]['model_sha256'],config_sha256=object_sha(cfg),
                data_sha256=grid['data_sha256'],consumer_source_identity=release)
            if path.exists():
                saved=read_json(path)
                if saved.get('identity')!=identity: raise ValueError('A/U cache identity changed')
                _finite_metrics(saved); _validate_native_c(saved['native_c'])
                records.append(dict(saved,cache_reused=True)); dependencies[str(path.resolve())]=sha256(path); continue
            if time_limit is not None and time.monotonic()>=time_limit: continue
            if datasets is None:
                datasets={s:build_dataset(data,s,root=root) for s in ('rr','fr')}
            if engine is None: engine=FRMetrics(datasets['fr'])
            model,_=load_checkpoint_model(cfg,wd/'candidates'/str(u),device)
            ast=load_file(str(wd/'candidates'/str(a)/'model.safetensors'))
            model.aligner.load_state_dict({k[8:]:v for k,v in ast.items() if k.startswith('aligner.')},strict=True)
            model.eval().requires_grad_(False)
            started=utcnow()
            result,native=_capture_cross_c(model,lambda:evaluate_model(model,datasets,device,engine=engine,include_q=True,with_val=False))
            _finite_metrics(result)
            ended=utcnow()
            base=next(r for r in records if r['step_A']==u and r['step_U']==u)
            diffs=[dict(scene_id=i,delta_hqnr=f['hqnr']-b['hqnr'],delta_ds=f['d_s']-b['d_s'],
                        delta_dlambda=f['d_lambda']-b['d_lambda'],delta_ergas=rr['ergas']-br['ergas'])
                   for i,(f,b,rr,br) in enumerate(zip(result['fr']['per_scene'],base['fr']['per_scene'],result['rr']['per_scene'],base['rr']['per_scene']))]
            row=dict(identity,**result,identity=identity,diagnostic_only=True,official_candidate=False,
                     cache_reused=False,paired_scene_deltas=diffs,native_c=native,
                     layout=cfg['fh12']['input_layout'],width=cfg['model_args']['hidden_size'],depth=cfg['model_args']['depth'],
                     timing=dict(start_utc=started,end_utc=ended))
            atomic_json(path,row)
            dependencies[str(path.resolve())]=sha256(path)
            from fh20r1.ledger import record_interval
            record_interval(root,server,interval_id=f'AU:{run}:{a}:{u}',kind='diagnostic',start_utc=started,
                end_utc=ended,run_id=run,evidence=str(path.relative_to(root)))
            records.append(row); del model,ast
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    report=dict(source_run=run,layout=cfg['fh12']['input_layout'],width=cfg['model_args']['hidden_size'],
        depth=cfg['model_args']['depth'],records=records,complete=len(records)==9,diagnostic_only=True,
        diagonal_verified=True,official_candidate=False,dependencies=dependencies)
    atomic_json(output/'matrix.json',report)
    return report


def q_report(bridge,server,root):
    path=Path(bridge['resolved_artifacts']['q_cache_path'])
    expected=bridge['resolved_artifact_sha256']['q_cache_path']
    _verify_dependencies({str(path):expected})
    with np.load(path,allow_pickle=False) as arrays:
        q=arrays['q']; radii=arrays['per_radius_q']; native=arrays['native_delta']
    if (q.shape!=(9714,4) or radii.shape!=(9714,4,4) or native.shape!=(9714,4,2)
            or not all(np.isfinite(x).all() for x in (q,radii,native)) or (q<0).any()
            or (radii<0).any() or not math.isfinite(float(bridge['q_ref'])) or bridge['q_ref']<=0):
        raise ValueError('Invalid q-response cache geometry or finite-value contract')
    s=bridge['q_ref']/(bridge['q_ref']+q)
    radiusmean=radii.mean((0,1),dtype=np.float64)
    out=dict(alias=bridge['alias'],q_cache_sha256=sha256(path),q_ref=bridge['q_ref'],tau_R=bridge['tau_R'],
        q_mean=float(q.mean()),q_std=float(q.std()),q_quantiles=np.quantile(q,[0,.25,.5,.75,1]).tolist(),
        s_quantiles=np.quantile(s,[0,.25,.5,.75,1]).tolist(),s_mean=float(s.mean()),
        per_radius_mean=radiusmean.tolist(),radius_slope=float(np.polyfit([.25,.5,1,2],radiusmean,1)[0]),
        native_c_mean=native.mean((0,1)).tolist(),native_c_abs_max=float(np.abs(native).max()),
        constant_A_q=.46875,cache_reused=True,performance_gate=False,
        dependencies={str(path.resolve()):expected})
    dest=Path(root)/'work_dir/_fh20r1'/server/'diagnostics/q_response'/bridge['alias']/'summary.json'
    atomic_json(dest,out); return out


def _lock_s1_branch(camp,report,output):
    path=camp/'branch_record.json'
    choice=dict(campaign_id='WV3_FH20R1_20260919_v1',server_id='s1',branch=report['branch'],
        condition=report['branch'],reason=report['reason'],report_sha256=sha256(output),locked_before_training=True)
    if path.exists():
        if read_json(path)!=choice: raise ValueError('Cannot reselect s1 branch')
    else:
        work_dir=camp.parent.parent
        if any(work_dir.glob('FH20R1_S1_*/meta/training_start_manifest.json')):
            raise ValueError('Cannot make the first s1 branch decision after training has started')
        atomic_json(path,choice)


def diagnose(server,device='cuda',root=ROOT):
    root=Path(root)
    from fh20r1.references import load_reference
    teacher,cfg,bridge,_q=load_reference('F'+server[1:],server,root,device='cpu')
    del teacher
    camp=root/'work_dir/_fh20r1'/server; output=camp/'diagnostics/report.json'
    identity=dict(bridge_sha256=object_sha(bridge),consumer_source_identity=source_identity(root))
    if output.exists():
        saved=read_json(output)
        if saved.get('identity')!=identity: raise ValueError('Existing diagnosis identity changed')
        _verify_dependencies(saved['dependencies'])
        if server=='s1': _lock_s1_branch(camp,saved,output)
        return saved
    q=q_report(bridge,server,root); crosses=[]; unavailable=[]
    # s1 diagnostic search is bounded to roughly one hour, not a campaign
    # deadline. A running single evaluation completes at its safe boundary.
    limit=time.monotonic()+3600 if server=='s1' else None
    for run in diagnostic_runs(server):
        try: crosses.append(cross_run(run,server,bridge['dataset_manifest'],device,root,time_limit=limit))
        except DiagnosticUnavailable as exc:
            unavailable.append(dict(run=run,reason=f'{type(exc).__name__}: {exc}'))
    decision=decide_s1(crosses) if server=='s1' else dict(branch='STANDARD',reason='fixed_local_plan')
    report=dict(identity=identity,server_id=server,complete=True,q_response=q,crosses=crosses,
        unavailable=unavailable,diagnostic_complete=not unavailable and all(x['complete'] for x in crosses),
        dependencies=dict(q['dependencies']),**decision)
    for cross in crosses: report['dependencies'].update(cross['dependencies'])
    atomic_json(output,report)
    if server=='s1':
        _lock_s1_branch(camp,report,output)
    return report
