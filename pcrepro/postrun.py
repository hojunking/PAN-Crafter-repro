"""Evaluate only predeclared selections; verify evidence before any Sheet upload."""
from datetime import datetime
from pathlib import Path
import time

from pcrepro.common import (ROOT,atomic_json,read,read_json,read_config,object_sha,sha256,
    source_identity,apply_runtime_policy,run_dir,camp,immutable_json,utcnow)
from pcrepro.plan import CAMPAIGN_ID,RECIPE_ID,RECIPE_SHA256,GRID,case_for,validate_config

LABELS=('EXACT_50000','RR_VAL_ERGAS_MIN')

def selected_evidence(case,root=ROOT):
    """WV2 cannot choose, substitute, or train a source of its own."""
    from pcrepro.checkpoint import selection_checkpoints,load_model_checkpoint
    source=case_for(case.source_run_id) if case.source_run_id else case
    wd=run_dir(source,root);cfg=read_config(wd/'meta/config.resolved.yaml')
    if validate_config(cfg,require_bound=True)!=source:raise ValueError('Source config differs')
    field=cfg['pcrepro']
    expected=dict(campaign_id=CAMPAIGN_ID,run_id=source.run_id,recipe_id=RECIPE_ID,
        recipe_sha256=RECIPE_SHA256,source_identity=field['source_identity'],data_sha=field['data_sha'],
        config_sha=object_sha(cfg),model_seed=source.seed,num_bands=source.num_bands,max_pixel=source.max_dn)
    if expected['source_identity']!=source_identity(root):raise ValueError('Source checkpoint runtime/code changed')
    if object_sha(read_json(field['data_manifest']))!=field['data_sha']:raise ValueError('Source data manifest changed')
    status=read_json(wd/'meta/training_status.json');grid=read_json(wd/'official/validation_grid.json')
    if (not status.get('training_complete') or status.get('actual_updates')!=50000
            or status.get('scheduler_end')!=50000 or not grid.get('complete')
            or grid.get('test_metrics_used') is not False or grid.get('expected_steps')!=list(GRID)
            or any(grid.get(k)!=v or status.get(k)!=v for k,v in expected.items())):
        raise ValueError('Source must complete fresh50K and the native validation-only grid')
    rows=grid['records']
    if [r['update'] for r in rows]!=list(GRID):raise ValueError('Validation grid changed')
    import math
    if any(not math.isfinite(r['ergas']) for r in rows):raise ValueError('Nonfinite validation selection')
    best=min(rows,key=lambda r:(r['ergas'],r['update']))
    paths=selection_checkpoints(wd);out={}
    for label in LABELS:
        model,identity=load_model_checkpoint(paths[label],expected=expected)
        del model
        row=rows[-1] if label=='EXACT_50000' else best
        if identity['update']!=row['update'] or identity['model_state_hash']!=row['model_state_hash']:
            raise ValueError('Checkpoint is not its declared fixed-grid selection')
        if label=='RR_VAL_ERGAS_MIN' and identity.get('val_ergas')!=best['ergas']:
            raise ValueError('Secondary is not earliest native val-ERGAS minimum')
        if label=='EXACT_50000':
            if not identity.get('full_state'):raise ValueError('Primary lacks fullstate')
            import torch
            state=torch.load(paths[label]/'training_state.pt',map_location='cpu',weights_only=False)
            if (not state['optimizer']['state'] or state.get('precision')!='fp32' or state['scheduler']['last_epoch']!=50000
                    or state['sampler']['completed_updates']!=50000
                    or state['validation_records']!=rows or not state.get('rng')
                    or any(float(v['step'])!=50000 for v in state['optimizer']['state'].values())):
                raise ValueError('Primary optimizer/scheduler/sampler endpoint differs')
        out[label]=dict(path=paths[label],identity=identity,val_ergas=row['ergas'])
    return out

def _config(config_path,root):
    cfg=read_config(config_path);case=validate_config(cfg,require_bound=True);f=cfg['pcrepro']
    apply_runtime_policy(root)
    if source_identity(root)!=f['source_identity']:raise ValueError('Evaluation source identity changed')
    manifest=read_json(f['data_manifest'])
    if object_sha(manifest)!=f['data_sha']:raise ValueError('Evaluation data manifest changed')
    return cfg,case,manifest

def execute(config_path,root=ROOT,device='cuda',stopcheck=None):
    import torch
    from pcrepro.data import NativeDataset,validate_manifest
    from pcrepro.evaluation import evaluate_model
    from pcrepro.checkpoint import load_model_checkpoint
    from pcrepro.profile import profile_model
    cfg,c,manifest=_config(config_path,root);wd=run_dir(c,root);f=cfg['pcrepro']
    if device!='cuda' or not torch.cuda.is_available():raise ValueError('Official postrun requires actual CUDA')
    if (wd/'official/summary.json').exists():return verify_summary_for_upload(c.run_id,root)
    started=time.monotonic()
    manifest=validate_manifest(manifest,root,verify_files=True,stopcheck=stopcheck)
    if object_sha(manifest)!=f['data_sha']:raise ValueError('Data evidence changed since preflight')
    selected=selected_evidence(c,root)
    datasets={split:NativeDataset(manifest,split,root=root) for split in ('rr','fr')}
    poststart=wd/'meta/postrun_start.json'
    if not poststart.exists():immutable_json(poststart,dict(started_at_utc=utcnow(),source_identity=f['source_identity']))
    control=camp(root,c.server)/'control.json'
    paused=lambda:read(control).get('command')=='STOP_NOW_SAFE' or bool(stopcheck and stopcheck())
    preprocessing=time.monotonic()-started;selections={};evaluated={};profile=None;profile_seconds=0.
    for label in LABELS:
        if paused():raise InterruptedError('Safe postrun pause')
        evidence=selected[label];identity=evidence['identity'];digest=identity['model_sha256']
        if digest not in evaluated:
            model,_=load_model_checkpoint(evidence['path'],device=device)
            eval_dir=wd/'official/evaluations'/digest
            metrics=evaluate_model(model,datasets,device,digest,output_dir=eval_dir,stopcheck=paused)
            evaluated[digest]=dict(metrics=metrics,path=eval_dir/'metrics.json',selection=label)
            if profile is None:
                profile_path=wd/'official/profile.json'
                saved=read(profile_path)
                if saved:
                    if saved.get('source_identity')!=f['source_identity'] or saved.get('checkpoint_sha256')!=digest:
                        raise ValueError('Profile checkpoint/source differs')
                    profile=saved['profile'];profile_seconds=saved['seconds']
                else:
                    t=time.monotonic();profile=profile_model(model,c.num_bands,device,stop_check=paused);profile_seconds=time.monotonic()-t
                    immutable_json(profile_path,dict(profile=profile,seconds=profile_seconds,checkpoint_sha256=digest,source_identity=f['source_identity']))
            del model
            torch.cuda.empty_cache()
        value=evaluated[digest];metrics=value['metrics']
        selections[label]=dict(update=identity['update'],checkpoint_sha256=digest,
            checkpoint_identity=identity,source_checkpoint_sha256=digest if c.source_run_id else None,
            val_ergas=evidence['val_ergas'],alias_of=value['selection'] if value['selection']!=label else None,
            data_sha256=f['data_sha'],evaluation_manifest_sha256=sha256(value['path']),
            evaluation_path=str(value['path'].relative_to(wd)),rr=metrics['rr'],fr=metrics['fr'])
    tr=read(wd/'meta/training_status.json');start=read(wd/'meta/training_start_manifest.json')
    costs=dict(tr.get('costs',{}));costs.update(test_seconds=sum(v['metrics']['seconds'] for v in evaluated.values()),profile_seconds=profile_seconds)
    costs['preprocessing_seconds']=costs.get('preprocessing_seconds',0)+preprocessing+read(camp(root,c.server)/'preflight'/f'{c.dataset}.json').get('preprocessing_seconds',0)
    if c.source_run_id:costs.update(training_seconds=0.,validation_seconds=0.,io_seconds=0.)
    start_at=start.get('started_at_utc',read_json(poststart)['started_at_utc']);end=utcnow()
    costs['wall_seconds']=(datetime.fromisoformat(end.replace('Z','+00:00'))-datetime.fromisoformat(start_at.replace('Z','+00:00'))).total_seconds()
    verified=all(v['metrics']['metadata']['paper_comparable'] for v in evaluated.values())
    summary=dict(campaign_id=CAMPAIGN_ID,recipe_id=RECIPE_ID,run_id=c.run_id,case=c.to_dict(),
        status='COMPLETE' if verified else 'PAPERSET_IDENTITY_UNVERIFIED',complete=True,
        actual_updates=c.updates,source_identity=f['source_identity'],data_sha256=f['data_sha'],
        config_sha256=object_sha(cfg),recipe_manifest_sha256=RECIPE_SHA256,
        evaluation_manifest_sha256=object_sha({k:v['evaluation_manifest_sha256'] for k,v in selections.items()}),
        paper_identity_status='VERIFIED' if verified else 'UNVERIFIED',
        started_at_utc=start_at,completed_at_utc=end,costs=costs,profile=profile,
        runtime=read(camp(root,c.server)/'preflight'/f'{c.dataset}.json').get('gpu',{}),
        selections=selections,source_run_id=c.source_run_id)
    if c.source_run_id:
        summary['source_wv3_training_seconds']=read(run_dir(c.source_run_id,root)/'meta/training_status.json')['costs']['training_seconds']
    from pcrepro.upload import validate_summary
    validate_summary(summary,c)
    immutable_json(wd/'official/summary.json',summary)
    return verify_summary_for_upload(c.run_id,root)

def verify_summary_for_upload(run,root=ROOT):
    """No inference/API calls. Re-read checkpoints, population, metrics and lineage."""
    c=case_for(run);wd=run_dir(c,root)
    cfg,c,manifest=_config(wd/'meta/config.resolved.yaml',root)
    summary=read_json(wd/'official/summary.json');f=cfg['pcrepro']
    from pcrepro.upload import validate_summary
    from pcrepro.evaluation import PROTOCOL,_evaluator_identity
    from tools.metrics.eval_fr import load_dlpan
    import os
    wald=load_dlpan(os.environ.get('PANCRAFTER_DLPAN',str(Path(root).parent/'DLPan-Toolbox')))
    evaluator=_evaluator_identity(wald)
    validate_summary(summary,c)
    expected=dict(source_identity=f['source_identity'],data_sha256=f['data_sha'],config_sha256=object_sha(cfg),recipe_manifest_sha256=RECIPE_SHA256)
    if not summary.get('complete') or any(summary.get(k)!=v for k,v in expected.items()):
        raise ValueError('Complete report provenance differs')
    selected=selected_evidence(c,root)
    verified=True
    for label,record in summary['selections'].items():
        evidence=selected[label];identity=evidence['identity'];digest=identity['model_sha256']
        path=wd/'official/evaluations'/digest/'metrics.json'
        if (record['checkpoint_sha256']!=digest or record['checkpoint_identity']!=identity
                or record['update']!=identity['update'] or record['val_ergas']!=evidence['val_ergas']
                or record['evaluation_path']!=str(path.relative_to(wd))
                or record['evaluation_manifest_sha256']!=sha256(path)):
            raise ValueError('Selected checkpoint/evaluation evidence changed')
        metrics=read_json(path);meta=metrics['metadata']
        if (meta.get('checkpoint_sha256')!=digest or meta.get('data_manifest_sha256')!=f['data_sha']
                or meta.get('dataset')!=c.dataset or meta.get('protocol')!=PROTOCOL
                or meta.get('protocol_sha256')!=object_sha(PROTOCOL)
                or meta.get('optimizer_updates_during_evaluation')!=0 or meta.get('evaluator')!=evaluator):
            raise ValueError('Official evaluation provenance differs')
        cursor=read_json(path.parent/'evaluation_cursor.json')
        if not cursor.get('complete') or any(meta.get(k)!=v for k,v in cursor['context'].items()):
            raise ValueError('Incomplete or mismatched scene cursor')
        for split in ('rr','fr'):
            report=metrics[split];spec=manifest['splits'][split]
            if (record[split]!=report or report.get('n_scenes')!=spec['count']
                    or report.get('scene_ids')!=spec['scene_ids'] or not report.get('official_complete')
                    or report.get('discarded_samples')!=0
                    or report.get('per_scene')!=cursor['rows'][split]):raise ValueError('Metrics/population mismatch')
            for i,row in enumerate(report['per_scene']):
                if (row['scene_index']!=i or row['scene_id']!=spec['scene_ids'][i]
                        or row['row_sha256']!=object_sha({k:v for k,v in row.items() if k!='row_sha256'})):
                    raise ValueError('Scene metric evidence changed')
                if split=='fr' and row['hqnr']!=(1-row['d_lambda'])*(1-row['d_s']):
                    raise ValueError('Per-scene HQNR formula differs')
            import numpy as np
            keys=('ergas','sam','psnr','ssim','scc','q4' if c.num_bands==4 else 'q8','rmse','cc') if split=='rr' else ('d_lambda','d_s','hqnr','jqm')
            for key in keys:
                if report.get(key) is not None and report[key]!=float(np.mean([r[key] for r in report['per_scene']])):
                    raise ValueError('Metric aggregation differs from per-scene values')
            paper=bool(spec.get('identity_validation',{}).get('verified') and meta.get('external_interp23tap_verified'))
            if report.get('paper_comparable')!=paper:raise ValueError('Paper identity evidence differs')
        if meta.get('paper_comparable')!=(metrics['rr']['paper_comparable'] and metrics['fr']['paper_comparable']):
            raise ValueError('Combined paper identity differs')
        if metrics['fr'].get('masking') is not False or metrics['fr'].get('support')!='full_native':
            raise ValueError('Non-native masked FR result')
        verified=verified and meta['paper_comparable']
    if summary.get('paper_identity_status')!=('VERIFIED' if verified else 'UNVERIFIED'):
        raise ValueError('Paper identity status is overstated')
    hashes={k:v['evaluation_manifest_sha256'] for k,v in summary['selections'].items()}
    if summary['evaluation_manifest_sha256']!=object_sha(hashes):raise ValueError('Evaluation summary hash differs')
    return summary
