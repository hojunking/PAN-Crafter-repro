"""Sensor-bound native evaluation, using the repository's existing primitives.

Training validation is unaugmented64 RR only. Official RR/FR are post-training,
with declared scene populations and no alignment/masking or external LP input.
"""
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from fh12.common import ROOT, atomic_json, object_sha, read_json, sha256
from pcrepro.data import SENSORS, canonical_indices
from tools.eval_dlpan import psnr_global, scc_dlpan, ssim_skimage
from tools.metrics.eval_rr import ergas, sam
from tools.metrics.q2n import q2n

RR_KEYS = ('ergas','sam','psnr','ssim','scc','q2n','rmse','cc')
FR_KEYS = ('d_lambda','d_s','hqnr')
JQM_VARIANT = 'SRF-substitute (NNLS-normalized), MTF41, phase2, global CMSC, v1=0.5; not SIPSA-equivalent'
PROTOCOL = dict(schema='PCREPRO_NATIVE_EVAL_v1',ratio=4,
    prediction='float32 clamp[-1,1] then (x+1)*max_dn/2; no rounding',
    rr_support='20:-21 on each spatial axis; all declared scenes',q_block=32,
    rr_scc='DLPan Sobel gradient magnitude, zero boundary, global cosine',
    rr_psnr='global all-band MSE; sensor max_dn peak; non-DLPan convention',
    rr_ssim='Gaussian11 sigma1.5 population covariance band mean; non-DLPan convention',
    rr_rmse='global all-band DN RMSE',rr_cc='global flattened Pearson',
    q2n='existing q2n with internal uint16 conversion; Q4 for C4 and Q8 for C8',
    fr_support='full original native PAN and original LMS; no alignment or masking',
    fr_d_lambda='sensor MTF41 then Q2n against native LMS, Q32',
    fr_d_s='native PAN / interp23tap(matlab_imresize(native PAN,1/4),4), block UQI32',
    aggregation='mean of per-scene metrics; HQNR_i=(1-Dlambda_i)*(1-Ds_i)',
    jqm=JQM_VARIANT,validation='native64 unaugmented ERGAS/SCC; no FR or HQNR selection')


def _check(stopcheck):
    if stopcheck is not None and stopcheck():
        raise InterruptedError('Safe evaluation pause requested')


def _hwc(array):
    return np.asarray(array,dtype=np.float64).transpose(1,2,0)


def _finite(values,description):
    if not np.isfinite(np.asarray(values,dtype=np.float64)).all():
        raise FloatingPointError('Nonfinite '+description)


def rr_scene_metrics(sr,gt,sensor):
    """CHW DN arrays; no sensor-specific scene-count guesses or sample removal."""
    bands,max_dn=SENSORS[sensor]
    a,b=_hwc(sr),_hwc(gt)
    if a.shape!=b.shape or a.shape[-1]!=bands or min(a.shape[:2])<52:
        raise ValueError('Native RR prediction/GT shape or band count differs')
    _finite(a,'RR prediction including borders');_finite(b,'RR ground truth including borders')
    a,b=a[20:-21,20:-21],b[20:-21,20:-21]
    if a.std()==0 or b.std()==0:raise ValueError('Undefined flattened Pearson RR CC')
    row=dict(ergas=ergas(a,b),sam=sam(a,b),psnr=psnr_global(a,b,max_dn),
        ssim=ssim_skimage(a,b,max_dn),scc=scc_dlpan(a,b),q2n=float(q2n(b,a,32,32)[0]),
        rmse=float(np.sqrt(np.mean((a-b)**2))),cc=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))
    row['q4' if bands==4 else 'q8']=row['q2n']
    row['band_relative_mse']=(np.mean((a-b)**2,axis=(0,1))/np.maximum(b.mean((0,1)),1e-8)**2).tolist()
    _finite([row[k] for k in RR_KEYS],'official RR metrics')
    return row


def fr_scene_metrics(sr,raw,sensor,band_order,wald,include_jqm=True):
    from tools.metrics.eval_fr import mtf_filter,imresize_matlab,_blockproc_uqi
    bands,max_dn=SENSORS[sensor]
    a,lms,ms,pan=_hwc(sr),_hwc(raw['lms']),_hwc(raw['ms']),np.asarray(raw['pan'],dtype=np.float64)[0]
    if (a.shape!=lms.shape or a.shape[-1]!=bands or a.shape[:2]!=pan.shape
            or ms.shape!=(a.shape[0]//4,a.shape[1]//4,bands)
            or any(v%32 for v in a.shape[:2]) or min(a.shape[:2])<32):
        raise ValueError('FR requires full native ratio4 support tiled by Q32, no partial-edge crop')
    for value in (a,lms,ms,pan):_finite(value,'full native FR inputs')
    order=canonical_indices(sensor,band_order);inverse=np.argsort(order)
    filtered=mtf_filter(a[...,list(order)],sensor.lower(),4,wald)[...,inverse]
    dl=float(1-q2n(lms,filtered,32,32)[0])
    low=wald.interp23tap(imresize_matlab(pan,1/4)[...,None],4)[...,0]
    high_q=[_blockproc_uqi(a[...,b],pan,32) for b in range(bands)]
    low_q=[_blockproc_uqi(lms[...,b],low,32) for b in range(bands)]
    ds=float(np.abs(np.asarray(high_q)-low_q).mean())
    row=dict(d_lambda=dl,d_s=ds,hqnr=float((1-dl)*(1-ds)),Q_high=high_q,Q_low=low_q)
    if include_jqm:
        from tools.metrics.jqm import jqm
        result=jqm(a[...,list(order)],ms[...,list(order)],pan,sensor,ratio=4,R=max_dn,
                   lpf='mtf',window=None,v1=.5)
        row.update(jqm=float(result['JQM']),qlr=float(result['QLR']),qhr=float(result['QHR']),
                   jqm_weights=np.asarray(result['w'])[inverse].tolist(),jqm_w_source=result['w_source'])
    _finite([row[k] for k in FR_KEYS+ (('jqm','qlr','qhr') if include_jqm else ())],'official FR metrics')
    return row


def _summary(rows,keys):
    if not rows:raise ValueError('No declared evaluation scenes')
    values={k:float(np.mean([r[k] for r in rows])) for k in keys}
    _finite(list(values.values()),'scene means')
    return dict(values,per_scene=rows,n_scenes=len(rows),
        standard_deviation={k:float(np.std([r[k] for r in rows],ddof=1 if len(rows)>1 else 0)) for k in keys})


def _prediction(model,batch,device,max_dn):
    # MS inference must not construct or feed an LP/PAN residual sidecar.
    y=model(batch['pan'].to(device),batch['ms'].to(device))
    if not isinstance(y,torch.Tensor) or y.ndim!=4 or not torch.isfinite(y).all():
        raise FloatingPointError('MS inference must return a finite normalized BCHW tensor')
    if y.shape[0]!=len(batch['pan']) or y.shape[-2:]!=batch['pan'].shape[-2:] or y.shape[1]!=batch['ms'].shape[1]:
        raise ValueError('Model prediction shape differs from native input support')
    return ((y.float().clamp(-1,1)+1)*(max_dn/2)).cpu().numpy()


@torch.no_grad()
def validation_metrics(model,dataset,device,stopcheck=None):
    if dataset.split!='val' or dataset.augment or not dataset.has_gt or dataset.sensor=='WV2':
        raise ValueError('Validation requires unaugmented native64 GT; WV2 has no validation')
    was_training=model.training;model.eval();e_values=[];s_values=[]
    try:
        # Direct deterministic batching does not consume global DataLoader/Torch RNG.
        for start in range(0,len(dataset),16):
            _check(stopcheck)
            indices=range(start,min(start+16,len(dataset)))
            rows=[dataset[i] for i in indices]
            if any(r['pan'].shape[-2:]!=(64,64) for r in rows):raise ValueError('Validation support must be64')
            batch={key:torch.stack([r[key] for r in rows]) for key in ('pan','ms')}
            predictions=_prediction(model,batch,device,dataset.max_pixel)
            _check(stopcheck)
            for index,prediction in zip(indices,predictions):
                target=_hwc(dataset.raw(index)['gt']);prediction=_hwc(prediction)
                e_values.append(ergas(prediction,target));s_values.append(scc_dlpan(prediction,target))
    finally:model.train(was_training)
    if len(e_values)!=len(dataset):raise ValueError('Incomplete native validation')
    _finite(e_values+s_values,'native validation')
    return dict(ergas=float(np.mean(e_values)),scc=float(np.mean(s_values)),hqnr=None,
        n_scenes=len(e_values),support='native64',masking=False,protocol=PROTOCOL['validation'],
        hqnr_status='not_measured_no_FR_checkpoint_selection')


def validation_ergas(model,dataset,device,stopcheck=None):
    return validation_metrics(model,dataset,device,stopcheck)['ergas']


def _evaluator_identity(wald):
    import scipy
    import skimage
    paths=[Path(__file__),Path(__file__).with_name('data.py'),ROOT/'tools/eval_dlpan.py',
           ROOT/'tools/metrics/eval_rr.py',ROOT/'tools/metrics/eval_fr.py',ROOT/'tools/metrics/q2n.py',
           ROOT/'tools/metrics/jqm.py']
    files={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    source=getattr(wald,'__file__',None)
    files['DLPan/wald_utilities.py']=sha256(source) if source else 'TEST_DOUBLE_NO_PAPER_CLAIM'
    versions=dict(numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,skimage=skimage.__version__)
    return dict(files=files,versions=versions,sha256=object_sha(dict(files=files,versions=versions)))


def _cursor_rows(state,datasets,include_jqm):
    if set(state.get('rows',{}))!={'rr','fr'}:raise ValueError('Invalid evaluation cursor split structure')
    for split,rows in state['rows'].items():
        if len(rows)>len(datasets[split]):raise ValueError('Evaluation cursor exceeds declared population')
        keys=RR_KEYS if split=='rr' else FR_KEYS+(('jqm','qlr','qhr') if include_jqm else ())
        for index,row in enumerate(rows):
            payload={k:v for k,v in row.items() if k!='row_sha256'}
            if (row.get('scene_index')!=index or row.get('scene_id')!=datasets[split].scene_ids[index]
                    or row.get('row_sha256')!=object_sha(payload)):
                raise ValueError('Evaluation cursor scene identity/checksum changed')
            _finite([row[k] for k in keys],'resumed scene metrics')


@torch.no_grad()
def evaluate_model(model,datasets,device,checkpoint_sha256,output_dir=None,stopcheck=None,
                   include_jqm=True,wald=None):
    """Scene-resumable official evaluation; callers select checkpoints beforehand.

    A test-double interp23tap is allowed for unit tests but explicitly prevents a
    paper-comparable claim. Production provenance includes the external file SHA.
    """
    if not re.fullmatch('[0-9a-f]{64}',str(checkpoint_sha256)):
        raise ValueError('Exact evaluated checkpoint SHA256 is mandatory')
    if set(datasets)!={'rr','fr'}:raise ValueError('Official evaluation accepts only RR/FR, never a selection set')
    rr,fr=datasets['rr'],datasets['fr'];sensor=rr.sensor
    if (fr.sensor!=sensor or any(d.augment or d.split!=key for key,d in datasets.items())
            or rr.manifest_sha256!=fr.manifest_sha256):raise ValueError('RR/FR sensor or full manifest identity differs')
    if wald is None:
        from tools.metrics.eval_fr import load_dlpan
        wald=load_dlpan(os.environ.get('PANCRAFTER_DLPAN',str(ROOT.parent/'DLPan-Toolbox')))
    evaluator=_evaluator_identity(wald)
    context=dict(schema='PCREPRO_EVALUATION_CURSOR_v1',checkpoint_sha256=checkpoint_sha256,
        data_manifest_sha256=rr.manifest_sha256,dataset=sensor,max_dn=rr.max_pixel,bands=rr.bands,
        band_order=rr.manifest['band_order'],protocol=PROTOCOL,protocol_sha256=object_sha(PROTOCOL),
        evaluator=evaluator,include_jqm=bool(include_jqm))
    cursor=Path(output_dir)/'evaluation_cursor.json' if output_dir is not None else None
    if cursor is not None and cursor.exists():
        state=read_json(cursor)
        if state.get('context')!=context:raise ValueError('Evaluation resume checkpoint/data/evaluator/protocol identity differs')
    else:state=dict(context=context,rows=dict(rr=[],fr=[]),evaluation_seconds=0.,complete=False)
    _cursor_rows(state,datasets,include_jqm)
    was_training=model.training;model.eval();started=time.monotonic()
    try:
        for split,dataset in datasets.items():
            rows=state['rows'][split]
            for index in range(len(rows),len(dataset)):
                _check(stopcheck)
                sample=dataset[index];batch={k:sample[k][None] for k in ('pan','ms')}
                sr=_prediction(model,batch,device,dataset.max_pixel)[0]
                _check(stopcheck)
                raw=dataset.raw(index)
                result=(rr_scene_metrics(sr,raw['gt'],sensor) if split=='rr' else
                    fr_scene_metrics(sr,raw,sensor,dataset.manifest['band_order'],wald,include_jqm))
                result.update(scene_index=index,scene_id=dataset.scene_ids[index])
                result['row_sha256']=object_sha(result);rows.append(result)
                state['evaluation_seconds']+=time.monotonic()-started;started=time.monotonic()
                if cursor is not None:atomic_json(cursor,state)
        state['complete']=True
        if cursor is not None:atomic_json(cursor,state)
    finally:model.train(was_training)
    qlabel='q4' if rr.bands==4 else 'q8'
    rr_report=_summary(state['rows']['rr'],RR_KEYS+(qlabel,))
    fr_report=_summary(state['rows']['fr'],FR_KEYS+(('jqm','qlr','qhr') if include_jqm else ()))
    external_verified=getattr(wald,'__file__',None) is not None
    for split,report in (('rr',rr_report),('fr',fr_report)):
        d=datasets[split];identity=d.spec.get('identity_validation',{})
        report.update(dataset=sensor,num_bands=d.bands,max_dn=d.max_pixel,
            scene_ids=d.scene_ids,official_complete=True,identity_validation=identity,
            paper_identity_status=identity.get('status','PAPERSET_IDENTITY_UNVERIFIED'),
            paper_comparable=bool(identity.get('verified') and external_verified),
            population='all declared samples; no subset fallback',discarded_samples=0)
    rr_report.update(crop='20:-21',q_block=32,q_name=qlabel.upper())
    fr_report.update(reference='native_PAN_and_original_LMS',support='full_native',masking=False,
        alignment=False,hqnr_variant='raw-original',aggregation='mean_per_scene_HQNR',
        jqm_variant=JQM_VARIANT,jqm_status='measured' if include_jqm else 'not_measured')
    if not include_jqm:fr_report['jqm']=None
    status=('PAPERSET_IDENTITY_UNVERIFIED' if sensor=='WV2' and not all(
        d.spec.get('identity_validation',{}).get('verified') for d in datasets.values()) else 'COMPLETE')
    result=dict(rr=rr_report,fr=fr_report,metadata=dict(context,status=status,
        paper_comparable=rr_report['paper_comparable'] and fr_report['paper_comparable'],
        prediction_rounding=False,optimizer_updates_during_evaluation=0,
        external_interp23tap_verified=external_verified),seconds=state['evaluation_seconds'])
    if output_dir is not None:atomic_json(Path(output_dir)/'metrics.json',result)
    return result
