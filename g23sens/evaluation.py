"""G23 native evaluation: legacy metrics, original PAN, no shift masking.

Only the model adapter is campaign-specific. The established per-scene metric
implementations are reused, and every declared scene is retained.
"""
from pathlib import Path
import os
import re
import time

import numpy as np
import torch

from g23sens.common import ROOT, atomic_json, object_sha, read_json, sha256
from pcrepro.evaluation import (rr_scene_metrics, fr_scene_metrics, JQM_VARIANT,
    PROTOCOL as BASE_PROTOCOL, _summary)
from tools.metrics.eval_rr import ergas
from tools.eval_dlpan import scc_dlpan

RR_KEYS=('ergas','sam','psnr','ssim','scc','q8','rmse','cc')
FR_KEYS=('hqnr','d_lambda','d_s','jqm')
PROTOCOL=dict(BASE_PROTOCOL,schema='G23SENS_NATIVE_EVAL_v1',
    model='G23 P0 W104D121: model(pan,ms,lpan), output y; original PAN reference',
    validation='legacy train_kdv._rr_val: float64 DN from normalized y/GT, torch per-sample plain ERGAS, batches16; no test selection')


def _check(stopcheck):
    if stopcheck is not None and stopcheck():raise InterruptedError('Safe evaluation pause')


def _prediction(model,batch,device):
    out=model(*(batch[k].to(device) for k in ('pan','ms','lpan')))
    y=out['y']
    if (not isinstance(y,torch.Tensor) or y.shape!=(len(batch['pan']),8,*batch['pan'].shape[-2:])
            or not torch.isfinite(y).all() or not torch.isfinite(out['delta']).all()):
        raise FloatingPointError('Invalid native G23 prediction/correction')
    return ((y.float().clamp(-1,1)+1)*1023.5).cpu().numpy()


@torch.no_grad()
def evaluate_validation(model,dataset,device,stopcheck=None):
    """Direct batches preserve global training RNG, unlike a DataLoader iterator."""
    if dataset.split!='val' or getattr(dataset,'augment',False):
        raise ValueError('Validation requires unaugmented native val data')
    was_training=model.training;model.eval();es=[];ss=[];total=0.
    try:
        for start in range(0,len(dataset),16):
            _check(stopcheck)
            ids=list(range(start,min(start+16,len(dataset))))
            rows=[dataset[i] for i in ids]
            if any(r['pan'].shape[-2:]!=(64,64) for r in rows):raise ValueError('Validation is native64')
            batch={k:torch.stack([r[k] for r in rows]) for k in ('pan','ms','lpan')}
            out=model(*(batch[k].to(device) for k in ('pan','ms','lpan')))
            pred=(out['y'].clip(-1,1).double()+1)/2*2047
            gt=(torch.stack([r['gt'] for r in rows]).to(device).double()+1)/2*2047
            if pred.shape!=gt.shape or not torch.isfinite(pred).all() or not torch.isfinite(out['delta']).all():
                raise FloatingPointError('Invalid native validation output/correction')
            mse=((gt-pred)**2).mean(dim=(2,3));mu=gt.mean(dim=(2,3))
            values=25.*torch.sqrt((mse/mu.clamp_min(1e-12)**2).mean(dim=1))
            total+=float(values.sum());es.extend(values.cpu().tolist())
            for p,g in zip(pred.cpu().numpy(),gt.cpu().numpy()):
                ss.append(scc_dlpan(p.transpose(1,2,0),g.transpose(1,2,0)))
            _check(stopcheck)
    finally:model.train(was_training)
    if len(es)!=len(dataset) or not es or not np.isfinite(es+ss).all():
        raise FloatingPointError('Incomplete/nonfinite validation')
    return dict(ergas=total/len(es),scc=float(np.mean(ss)),hqnr=None,
        n_scenes=len(es),support='native64',masking=False,
        hqnr_status='not_evaluated_test_postrun_only',protocol=PROTOCOL['validation'])


def evaluator_identity(wald):
    import scipy
    import skimage
    files={name:sha256(ROOT/name) for name in ('g23sens/evaluation.py','pcrepro/evaluation.py',
        'pcrepro/data.py','tools/eval_dlpan.py','tools/metrics/eval_rr.py',
        'tools/metrics/eval_fr.py','tools/metrics/q2n.py','tools/metrics/jqm.py')}
    source=getattr(wald,'__file__',None)
    files['external/DLPan/wald_utilities.py']=sha256(source) if source else 'TEST_DOUBLE_UNVERIFIED'
    return dict(files=files,numpy=np.__version__,torch=torch.__version__,
        scipy=scipy.__version__,skimage=skimage.__version__)


def _population(dataset):
    ids=list(dataset.scene_ids)
    if not ids or len(ids)!=len(dataset) or len(set(ids))!=len(ids):
        raise ValueError('Full native scene IDs/count must be explicit and unique')
    return ids


def validate_metrics(result):
    meta=result['metadata']
    if (meta.get('protocol')!=PROTOCOL or meta.get('protocol_sha256')!=object_sha(PROTOCOL)
            or meta.get('optimizer_updates_during_evaluation')!=0):
        raise ValueError('Evaluation protocol changed')
    for split,keys in (('rr',RR_KEYS),('fr',FR_KEYS)):
        report=result[split];rows=report['per_scene'];ids=report['scene_ids']
        if (len(rows)!=report['n_scenes'] or len(rows)!=len(ids) or not rows
                or report.get('discarded_samples')!=0 or report.get('official_complete') is not True):
            raise ValueError('Incomplete declared evaluation population')
        for i,row in enumerate(rows):
            if (row['scene_index']!=i or row['scene_id']!=ids[i]
                    or row['row_sha256']!=object_sha({k:v for k,v in row.items() if k!='row_sha256'})):
                raise ValueError('Scene identity/metric bytes changed')
            if not np.isfinite([row[k] for k in keys]).all():raise ValueError('Nonfinite metric')
            if split=='fr' and row['hqnr']!=(1-row['d_lambda'])*(1-row['d_s']):
                raise ValueError('HQNR must be scene-wise product')
        for key in keys:
            if report[key]!=float(np.mean([r[key] for r in rows])):raise ValueError('Scene aggregation changed')
    fr=result['fr']
    if (fr.get('masking') is not False or fr.get('alignment') is not False
            or fr.get('support')!='full_native' or fr.get('jqm_variant')!=JQM_VARIANT):
        raise ValueError('Masked/shifted/alternate FR protocol is forbidden')
    return result


@torch.no_grad()
def evaluate_checkpoint(model,datasets,device,checkpoint_sha256,output_dir=None,stopcheck=None,wald=None):
    if not re.fullmatch('[0-9a-f]{64}',str(checkpoint_sha256)):raise ValueError('Checkpoint SHA required')
    if set(datasets)!={'rr','fr'}:raise ValueError('Official evaluation accepts native RR/FR only')
    rr,fr=datasets['rr'],datasets['fr']
    if any(getattr(d,'sensor',None)!='WV3' or getattr(d,'augment',False) or getattr(d,'split_key',d.split)!=k
           for k,d in datasets.items()):raise ValueError('Only unaugmented WV3 RR/FR allowed')
    binding_sha=object_sha(rr.binding)
    if binding_sha!=object_sha(fr.binding):raise ValueError('RR/FR locked binding identity differs')
    if wald is None:
        from tools.metrics.eval_fr import load_dlpan
        wald=load_dlpan(os.environ.get('PANCRAFTER_DLPAN',str(ROOT.parent/'DLPan-Toolbox')))
    context=dict(schema='G23SENS_EVALUATION_CURSOR_v1',checkpoint_sha256=checkpoint_sha256,
        data_manifest_sha256=binding_sha,split_manifest_sha256={k:d.manifest_sha256 for k,d in datasets.items()},
        scene_ids={k:_population(d) for k,d in datasets.items()},
        protocol=PROTOCOL,protocol_sha256=object_sha(PROTOCOL),evaluator=evaluator_identity(wald))
    cursor=Path(output_dir)/'evaluation_cursor.json' if output_dir is not None else None
    state=read_json(cursor) if cursor is not None and cursor.exists() else dict(context=context,rows={'rr':[],'fr':[]},seconds=0.,complete=False)
    if state.get('context')!=context or set(state.get('rows',{}))!={'rr','fr'}:
        raise ValueError('Evaluation resume source/checkpoint/population changed')
    for split,rows in state['rows'].items():
        if len(rows)>len(datasets[split]):raise ValueError('Evaluation cursor exceeds scene count')
        for i,row in enumerate(rows):
            if (row.get('scene_index')!=i or row.get('scene_id')!=context['scene_ids'][split][i]
                    or row.get('row_sha256')!=object_sha({k:v for k,v in row.items() if k!='row_sha256'})):
                raise ValueError('Corrupt resumed per-scene metric')
    was_training=model.training;model.eval();started=time.monotonic()
    try:
        for split,dataset in datasets.items():
            for index in range(len(state['rows'][split]),len(dataset)):
                _check(stopcheck);row=dataset[index]
                sr=_prediction(model,{k:row[k][None] for k in ('pan','ms','lpan')},device)[0]
                _check(stopcheck);raw=dataset.raw(index)
                value=(rr_scene_metrics(sr,raw['gt'],'WV3') if split=='rr' else
                    fr_scene_metrics(sr,raw,'WV3',dataset.manifest['band_order'],wald,True))
                value.update(scene_index=index,scene_id=dataset.scene_ids[index]);value['row_sha256']=object_sha(value)
                state['rows'][split].append(value);state['seconds']+=time.monotonic()-started;started=time.monotonic()
                if cursor is not None:atomic_json(cursor,state)
        state['complete']=True
        if cursor is not None:atomic_json(cursor,state)
    finally:model.train(was_training)
    result={split:_summary(state['rows'][split],RR_KEYS if split=='rr' else FR_KEYS) for split in ('rr','fr')}
    for split,dataset in datasets.items():
        identity=dataset.spec.get('identity_validation',{})
        result[split].update(scene_ids=list(dataset.scene_ids),discarded_samples=0,official_complete=True,
            paper_identity_status=identity.get('status','PAPERSET_IDENTITY_UNVERIFIED'),
            identity_validation=identity,paper_comparable=bool(identity.get('verified') and getattr(wald,'__file__',None)),
            population='all declared native scenes',sensor='WV3',max_dn=2047)
    result['rr'].update(crop='20:-21',q_block=32)
    result['fr'].update(reference='native_PAN_and_original_LMS',support='full_native',masking=False,
        alignment=False,hqnr_variant='raw-original',aggregation='mean_per_scene_HQNR',jqm_variant=JQM_VARIANT)
    result.update(metadata=dict(context,optimizer_updates_during_evaluation=0,
        paper_comparable=all(result[s]['paper_comparable'] for s in ('rr','fr'))),seconds=state['seconds'])
    validate_metrics(result)
    if output_dir is not None:atomic_json(Path(output_dir)/'metrics.json',result)
    return result
