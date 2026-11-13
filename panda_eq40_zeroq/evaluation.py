"""Same-state official native RR20/FR20, unchanged metric functions and references."""
from contextlib import contextmanager
import hashlib
import time
import numpy as np
import torch
from directshift48.evaluation import metric_functions_for,metric_values,expected_ids
from panda_stage2_condq.training import rng_state,restore_rng
from ta2.evaluation import evaluator_identity,JQM_VARIANT
from .training import model_hash
from .plan import digest,targets


def test_rre(prediction,batch,split,cfg):
    """Mode-matched edge-reference diagnostic only; never transports predictions or selects checkpoints."""
    from ta2.registration import register_band
    reference=batch.gt if split=='rr' else batch.native_lms
    pan=((prediction['pan_aligned'][0,0]+1)/2).detach().cpu().numpy()
    target=((reference[0,cfg["reference_bands"]].mean(0)+1)/2).detach().cpu().numpy()
    sp,sr,margin=(.8,.8,19) if split=='rr' else (1.98,.7,24)
    from .geometry import warp_support_mask
    # Protect the full search region, including Gaussian and Scharr stencils.
    clean=margin-8-int(np.ceil(3*sp))-1
    support=warp_support_mask(*batch.pan.shape[-2:],prediction['u'].detach())
    if not bool(support[...,clean:-clean,clean:-clean].all()):
        return {'registration':{'valid':False,'shift':[0.,0.],
                    'fallback_reason':'WARP_SEARCH_SUPPORT_VIOLATION'},'rre_px':None}
    record=register_band(pan,target,sigma_p=sp,sigma_r=sr,margin=margin,bound=8)
    return dict(registration=record,
                rre_px=float(np.linalg.norm(record['shift'])) if record['valid'] else None)


def rre_summary(rows,cfg):
    values=[r['rre_reference']['rre_px'] for r in rows]
    valid=[v for v in values if v is not None]
    return {'sample_count':len(rows),'valid_count':len(valid),'failure_count':len(rows)-len(valid),
            'RRE50':float(np.median(valid)) if len(valid)==20 else None,
            'reference':cfg['reference_mode']+'_MEAN_'+('GT' if rows[0]['scene_id'].split(':')[1]=='rr' else 'STORED_LMS'),
            'unit':'current PAN pixel','all_samples_required':True}


@contextmanager
def paused(model):
    modes=[(m,m.training) for m in model.modules()]
    before=model_hash(model); rng=rng_state()
    try:
        model.eval()
        with torch.no_grad():yield before
    finally:
        for m,mode in modes:m.training=mode
        restore_rng(rng)
        if model_hash(model)!=before:raise ValueError('Evaluation changed model weights')


def evaluate(trainer,sources,pin,*,on_scene=None):
    if evaluator_identity()!=pin:raise ValueError('Official evaluator differs from the admission pin')
    trainer.audit(); cfg=trainer.cfg
    functions=metric_functions_for(sources,pin)
    device=next(trainer.model.parameters()).device
    record={'schema':'PANDA_ZEROQ_EVAL_V1','run_id':cfg['run_id'],
        'config_sha256':cfg['config_sha256'],'step':trainer.updates,'stage':cfg['stage'],
        'eval_protocol_id':digest(pin),'test_aware':True,'inference_tta':False,
        'official_reference':'original_PAN/native_LMS; FR full512 unmasked; RR all20 dim_cut21',
        'jqm_variant':JQM_VARIANT,'output_coordinates':'MS','reference_warp':False,
        'edge_reference_bands':cfg["reference_bands"],'edge_reference':cfg['reference_mode']+'_ARITHMETIC_MEAN',
        'pan_warp':'LANCZOS4_REFLECTION_ALL_PATHS','rre_margin':{'rr':19,'fr':24}}
    tick=time.monotonic()
    with paused(trainer.model) as before:
        record['model_sha256']=before
        for split in ('rr','fr'):
            rows=[]; corrected_rows=[]
            for i,sample_id in enumerate(expected_ids(sources,split)):
                batch=sources.batch(split,[i],device=device)
                side=256 if split=='rr' else 512
                if batch.pan.shape[-2:]!=(side,side) or tuple(batch.lineage.sample_ids)!=(sample_id,):
                    raise ValueError('Official native scene/grid identity mismatch')
                if split=='fr' and batch.gt is not None:raise ValueError('FR must not contain GT')
                prediction=trainer.model.native_prediction(batch)
                if prediction['y'].shape!=batch.native_lms.shape or not bool(torch.isfinite(prediction['y']).all()):
                    raise ValueError('Invalid native prediction')
                raw=((prediction['y'][0].float()+1)*(sources.peak/2)).cpu().numpy()
                arrays={k:v[0] for k,v in sources.raw(split,[i]).items()}
                metrics=metric_values(functions[split](raw,arrays),split,cfg['bands'])
                rows.append({'scene_id':sample_id,'metrics':metrics,
                    'correction_dy_dx':prediction['u'][0].cpu().tolist(),
                    'raw_prediction_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                    'rre_reference':test_rre(prediction,batch,split,cfg)})
                if split=='fr':
                    corrected_pan=((prediction['pan_aligned'][0].float()+1)*(sources.peak/2)).cpu().numpy()
                    cm=dual_fr_metrics(functions['fr'],raw,arrays,corrected_pan,metrics,cfg['bands'])
                    corrected_rows.append({**rows[-1],'metrics':cm,
                        'corrected_pan_sha256':hashlib.sha256(corrected_pan.tobytes()).hexdigest()})
                if on_scene:on_scene(split,i+1)
            record[split]={'per_scene':rows,'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}}
            if split=='fr':
                record['fr_corrected_pan']={'per_scene':corrected_rows,
                    'metrics':{k:float(np.mean([r['metrics'][k] for r in corrected_rows])) for k in corrected_rows[0]['metrics']},
                    'reference':'SAME_ALIGNER_LANCZOS4_CORRECTED_PAN',
                    'low_resolution_pan':'REGENERATED_FROM_CORRECTED_PAN_BY_UNCHANGED_D_S',
                    'prediction_and_lms_unchanged':True,'official_comparable':False}
        record['test_rre']={split:rre_summary(record[split]['per_scene'],cfg) for split in ('rr','fr')}
    record['elapsed_seconds']=time.monotonic()-tick
    record['same_state_verified']=True
    record['joint_pass']=(record['fr']['metrics']['HQNR']>=targets(cfg['sensor'])['HQNR_min'] and record['rr']['metrics']['ERGAS']<=targets(cfg['sensor'])['ERGAS_max'])
    if evaluator_identity()!=pin:raise ValueError('Evaluator changed while running')
    return record


def verify_record(record,cfg):
    if (record.get('schema')!='PANDA_ZEROQ_EVAL_V1' or record['run_id']!=cfg['run_id']
            or record['config_sha256']!=cfg['config_sha256'] or record['step'] not in cfg['evaluation_steps']
            or record.get('same_state_verified') is not True or record.get('reference_warp') is not False
            or record.get('inference_tta') is not False
            or record.get('edge_reference_bands') != cfg["reference_bands"]
            or record.get('edge_reference') != cfg['reference_mode']+'_ARITHMETIC_MEAN'
            or record.get('pan_warp') != 'LANCZOS4_REFLECTION_ALL_PATHS'
            or record.get('rre_margin') != {'rr':19,'fr':24}):
        raise ValueError('Official evaluation identity/protocol mismatch')
    expected={'rr':{'CC','ERGAS','PSNR',('Q8' if cfg['bands']==8 else 'Q4'),'RMSE','SAM','SCC','SSIM'},
              'fr':{'HQNR','D_s','D_lambda','JQM'}}
    for split,keys in expected.items():
        rows=record[split]['per_scene']
        if len(rows)!=20 or [r['scene_id'] for r in rows]!=[f"{cfg['sensor']}:{split}:{i}" for i in range(20)]:
            raise ValueError('Incomplete/reordered official evaluation')
        if any(set(r['metrics'])!=keys for r in rows) or set(record[split]['metrics'])!=keys:
            raise ValueError('Missing official metrics')
        for row in rows:
            rre=row['rre_reference'];reg=rre['registration']
            expected=float(np.linalg.norm(reg['shift'])) if reg['valid'] else None
            if rre['rre_px']!=expected:
                raise ValueError('Reference residual norm differs from registration evidence')
        if record.get('test_rre',{}).get(split)!=rre_summary(rows,cfg):
            raise ValueError('Reference test RRE summary differs from all20 evidence')
        for k in keys:
            values=[r['metrics'][k] for r in rows]
            if not np.isfinite(values).all() or float(np.mean(values))!=record[split]['metrics'][k]:
                raise ValueError('Evaluation aggregate disagrees with source scenes')

    verify_corrected_pair(record)

def dual_fr_metrics(fn,prediction,arrays,corrected_pan,original_metrics,bands):
    """Replace only PAN; official d_s recomputes its low-pass PAN internally."""
    corrected=metric_values(fn(prediction,{**arrays,'pan':corrected_pan}),'fr',bands)
    if corrected['D_lambda']!=original_metrics['D_lambda']:
        raise ValueError('D_lambda changed although output and LMS were identical')
    return corrected

def verify_corrected_pair(record):
    item=record['fr_corrected_pan'];rows=item['per_scene'];original=record['fr']['per_scene']
    if (item.get('reference')!='SAME_ALIGNER_LANCZOS4_CORRECTED_PAN'
        or item.get('prediction_and_lms_unchanged') is not True
        or item.get('official_comparable') is not False
        or item.get('low_resolution_pan')!='REGENERATED_FROM_CORRECTED_PAN_BY_UNCHANGED_D_S'
        or len(rows)!=20 or len(original)!=20):raise ValueError('Invalid dual FR protocol')
    keys={'HQNR','D_s','D_lambda','JQM'}
    for a,b in zip(original,rows):
        if any(a[k]!=b[k] for k in ('scene_id','raw_prediction_sha256','correction_dy_dx','rre_reference')):
            raise ValueError('Dual FR must evaluate the same output and scene')
        if set(b['metrics'])!=keys or len(b.get('corrected_pan_sha256',''))!=64:
            raise ValueError('Missing corrected PAN metrics/evidence')
        if a['metrics']['D_lambda']!=b['metrics']['D_lambda']:
            raise ValueError('Dual FR D_lambda mismatch')
    for k in keys:
        values=[r['metrics'][k] for r in rows]
        if not np.isfinite(values).all() or float(np.mean(values))!=item['metrics'][k]:
            raise ValueError('Corrected FR aggregate differs from all20 scenes')


def validation(*args,**kwargs):
    raise ValueError('Stage1 validation remains in the original campaign')
