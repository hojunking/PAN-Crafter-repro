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
from .plan import digest,HQNR_MIN,ERGAS_MAX


def test_rre(prediction,batch,split):
    """RGB diagnostic only; never transports predictions or selects checkpoints."""
    from ta2.registration import register_band
    reference=batch.gt if split=='rr' else batch.native_lms
    pan=((prediction['pan_aligned'][0,0]+1)/2).detach().cpu().numpy()
    target=((reference[0,[4,2,1]].mean(0)+1)/2).detach().cpu().numpy()
    sp,sr,margin=(.8,.8,16) if split=='rr' else (1.98,.7,24)
    record=register_band(pan,target,sigma_p=sp,sigma_r=sr,margin=margin,bound=8)
    return dict(registration=record,
                rre_px=float(np.linalg.norm(record['shift'])) if record['valid'] else None)


def rre_summary(rows):
    values=[r['rre_rgb']['rre_px'] for r in rows]
    valid=[v for v in values if v is not None]
    return {'sample_count':len(rows),'valid_count':len(valid),'failure_count':len(rows)-len(valid),
            'RRE50':float(np.median(valid)) if len(valid)==20 else None,
            'reference':'RGB_MEAN_GT' if rows[0]['scene_id'].split(':')[1]=='rr' else 'RGB_MEAN_STORED_LMS',
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
    record={'schema':'WV3_MAIN_RGB_EVAL_V1','run_id':cfg['run_id'],
        'config_sha256':cfg['config_sha256'],'step':trainer.updates,'stage':cfg['stage'],
        'eval_protocol_id':digest(pin),'test_aware':True,'inference_tta':False,
        'official_reference':'original_PAN/native_LMS; FR full512 unmasked; RR all20 dim_cut21',
        'jqm_variant':JQM_VARIANT,'output_coordinates':'MS','reference_warp':False}
    tick=time.monotonic()
    with paused(trainer.model) as before:
        record['model_sha256']=before
        for split in ('rr','fr'):
            rows=[]
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
                metrics=metric_values(functions[split](raw,arrays),split,8)
                rows.append({'scene_id':sample_id,'metrics':metrics,
                    'correction_dy_dx':prediction['u'][0].cpu().tolist(),
                    'raw_prediction_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                    'rre_rgb':test_rre(prediction,batch,split)})
                if on_scene:on_scene(split,i+1)
            record[split]={'per_scene':rows,'metrics':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}}
        record['test_rre_rgb']={split:rre_summary(record[split]['per_scene']) for split in ('rr','fr')}
    record['elapsed_seconds']=time.monotonic()-tick
    record['same_state_verified']=True
    record['joint_pass']=(record['fr']['metrics']['HQNR']>=HQNR_MIN and record['rr']['metrics']['ERGAS']<=ERGAS_MAX)
    if evaluator_identity()!=pin:raise ValueError('Evaluator changed while running')
    return record


def validation(trainer,sources,inventory):
    from .selection import collect_validation
    device=next(trainer.model.parameters()).device
    with paused(trainer.model):
        return collect_validation(trainer.model,sources,inventory=inventory,include_ergas=True,
            epsilon=trainer.cfg['alignment']=='LEARNED',device=device,batch_size=48)


def verify_record(record,cfg):
    if (record.get('schema')!='WV3_MAIN_RGB_EVAL_V1' or record['run_id']!=cfg['run_id']
            or record['config_sha256']!=cfg['config_sha256'] or record['step'] not in cfg['evaluation_steps']
            or record.get('same_state_verified') is not True or record.get('reference_warp') is not False
            or record.get('inference_tta') is not False):
        raise ValueError('Official evaluation identity/protocol mismatch')
    expected={'rr':{'CC','ERGAS','PSNR','Q8','RMSE','SAM','SCC','SSIM'},
              'fr':{'HQNR','D_s','D_lambda','JQM'}}
    for split,keys in expected.items():
        rows=record[split]['per_scene']
        if len(rows)!=20 or [r['scene_id'] for r in rows]!=[f'WV3:{split}:{i}' for i in range(20)]:
            raise ValueError('Incomplete/reordered official evaluation')
        if any(set(r['metrics'])!=keys for r in rows) or set(record[split]['metrics'])!=keys:
            raise ValueError('Missing official metrics')
        for row in rows:
            rre=row['rre_rgb'];reg=rre['registration']
            expected=float(np.linalg.norm(reg['shift'])) if reg['valid'] else None
            if rre['rre_px']!=expected:
                raise ValueError('RGB residual norm differs from registration evidence')
        if record.get('test_rre_rgb',{}).get(split)!=rre_summary(rows):
            raise ValueError('RGB test RRE summary differs from all20 evidence')
        for k in keys:
            values=[r['metrics'][k] for r in rows]
            if not np.isfinite(values).all() or float(np.mean(values))!=record[split]['metrics'][k]:
                raise ValueError('Evaluation aggregate disagrees with source scenes')
