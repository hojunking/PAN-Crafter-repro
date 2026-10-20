"""Input-only residual registration; never used for checkpoint selection."""
from collections import Counter
import math
import time

import numpy as np
import torch

from directshift48.reporting import atomic_json
from directshift48.evaluation import expected_ids
from donor30.plan import canonical_sha256
from pa.warp import warp_pan
from ta2.registration import ESTIMATOR, register_band


def paired_summary(rows):
    """Invalid zero fallbacks must never appear as successful zero residuals."""
    def summarize(subset):
        paired = [r for r in subset if r['before']['valid'] and r['after']['valid']]
        out = {'count':len(subset), 'paired_valid_count':len(paired),
               'failures':{which:dict(Counter(r[which]['fallback_reason'] for r in subset
                                              if not r[which]['valid'])) for which in ('before','after')}}
        for which in ('before','after'):
            shifts = np.asarray([r[which]['shift'] for r in paired], dtype=np.float64)
            out[which] = {'median_dy_dx':np.median(shifts,axis=0).tolist(),
                'norm_median':float(np.median(np.linalg.norm(shifts,axis=1))),
                'norm_p90':float(np.quantile(np.linalg.norm(shifts,axis=1),.9))} if paired else None
        return out
    return {'bands':{str(k):summarize([r for r in rows if r['band']==k]) for k in range(8)},
            'groups':{name:summarize([r for r in rows if r['band'] in bands])
                      for name,bands in {'MS1':[1,2,4,6],'MS2':[0,3,5,7]}.items()},
            'all':summarize(rows)}


def _shifts(value):
    shifts=torch.as_tensor(value,dtype=torch.float32)
    if shifts.shape==(2,): shifts=shifts[None].expand(8,-1)
    if shifts.shape!=(8,2) or not bool(torch.isfinite(shifts).all()):
        raise ValueError('Finite global2 or per-band8x2 actual shifts required')
    return shifts


def residual_registration(root, cfg, evaluation, sources, *, check=lambda:None):
    """Reuse exact same A/geometry correction evidence without rerunning a U-Net."""
    identity={'schema':'BAND30_SELECTED_INPUT_RESIDUAL_V1', 'estimator':ESTIMATOR,
        'aligner_state_sha256':evaluation['aligner_state_sha256'],
        'reference':'PROVIDED_LMS_NOT_FR_GT', 'sampling':'ACTUAL_BICUBIC_WARP_AND_REESTIMATE',
        'checkpoint_selection':'UNCHANGED_HQNR_THEN_ERGAS', 'physical_ground_truth':False,
        'source_hashes':{split:sources.manifest_sha(split) for split in ('rr','fr')},
        'geometry':cfg['geometry'],
        'corrections':{split:[{k:r[k] for k in ('scene_id','correction_dy_dx',
            'pan_applied_dy_dx','ms_applied_dy_dx')} for r in evaluation[split]['per_scene']]
            for split in ('rr','fr')}}
    key=canonical_sha256(identity)
    path=root/'diagnostics'/f'{key}.json'
    if path.exists():
        import json
        prior=json.loads(path.read_text())
        if prior.get('identity')!=identity or prior.get('status')!='COMPLETE':
            raise ValueError('Residual-registration cache identity mismatch')
        return {'path':str(path),'cache_reused':True,'identity_sha256':key}
    started=time.monotonic(); splits={}
    for split in ('rr','fr'):
        rows=[]
        expected=list(expected_ids(sources,split))
        if [r['scene_id'] for r in evaluation[split]['per_scene']] != expected:
            raise ValueError('Diagnostic scene order differs from evaluated source')
        for index,evidence in enumerate(evaluation[split]['per_scene']):
            check()
            arrays=sources.raw(split,[index])
            if split=='fr' and 'gt' in arrays: raise ValueError('No FR GT diagnostic')
            pan=torch.as_tensor(arrays['pan']/sources.peak,dtype=torch.float32)
            lms=torch.as_tensor(arrays['lms']/sources.peak,dtype=torch.float32)
            pshift,mshift=_shifts(evidence['pan_applied_dy_dx']),_shifts(evidence['ms_applied_dy_dx'])
            pviews=pan.expand(8,-1,-1,-1).contiguous()
            references=lms[0,:,None]
            with torch.no_grad():
                moved_pan=warp_pan(pviews,pshift).numpy()[:,0]
                moved_ms=warp_pan(references,mshift).numpy()[:,0]
            max_shift=max(float(pshift.abs().max()),float(mshift.abs().max()))
            # Bound8 + Gaussian6 + Scharr1 + bicubic2 + predicted displacement.
            margin=max(24,17+math.ceil(max_shift))
            if 2*margin >= min(pan.shape[-2:]):
                raise ValueError('No clean diagnostic interior for predicted shift')
            for band in range(8):
                check()
                before=register_band(pan[0,0].numpy(),lms[0,band].numpy(),margin=margin)
                after=register_band(moved_pan[band],moved_ms[band],margin=margin)
                rows.append({'scene_id':evidence['scene_id'],'band':band,'margin':margin,
                             'before':before,'after':after})
        splits[split]={'per_scene_band':rows,'summary':paired_summary(rows)}
    atomic_json(path,{'status':'COMPLETE','identity':identity,'elapsed_seconds':time.monotonic()-started,
                      'interpretation':'Registration proxy, not physical displacement ground truth',**splits})
    return {'path':str(path),'cache_reused':False,'identity_sha256':key}
