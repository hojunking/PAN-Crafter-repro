"""Production adapters for TA2 diagnostics; no fallback to GT at inference.

Frozen probe sample IDs, batch streams and offsets are independent of training
state. All callbacks use current native pixel units, including nested FOVs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ta2.data import transform_shifts
from ta2.diagnostics import (PROBE_SEED, cost_probe, gradient_vector_summary, run_diagnostics)
from ta2.evaluation import (atomic_json, assert_checkpoint_model, check_seal, digest, file_sha,
                            fr_scene, preserve_runtime, rr_scene, seal)
from ta2.losses import compute_objective
from ta2.registration import estimate_shift

OFFLINE = {'GLOBAL_FIXED', 'PER_IMAGE_GRAD', 'PER_IMAGE_INTENSITY', 'PER_IMAGE_BICUBIC', 'TRAIN_GT_PROXY', 'SHUFFLED'}


def objective_assets(model, bundle, cfg, ids, rots, device):
    """Canonical cache then the exact same h/v/rotation as the native sample."""
    fixed = pseudo = None
    ids = np.asarray(ids, dtype=np.int64)
    if model.policy in OFFLINE:
        fixed = bundle.case_shifts(cfg['case_id'], 'train')[ids].to(device)
    if model.policy == 'GLOBAL_LEARNED':
        fixed = model.global_shift[None].expand(len(ids), -1)
    if cfg['struct_descriptor'] == 'PSEUDO_HUBER':
        pseudo = bundle.case_shifts('TA2-S07', 'train')[ids].to(device)
    if rots is not None:
        fixed = transform_shifts(fixed, rots) if fixed is not None else None
        pseudo = transform_shifts(pseudo, rots) if pseudo is not None else None
    weights = None
    if cfg['band_weights'] == 'CAL_FROZEN':
        weights = torch.tensor(bundle.band_weights()['weights'], device=device, dtype=torch.float32)
    return dict(fixed_shift=fixed, pseudo_shift=pseudo, band_weights=weights)


def _named_aligner(model):
    if model.aligner is not None:
        return list(model.aligner.named_parameters())
    if model.global_shift is not None:
        return [('global_shift', model.global_shift)]
    return []


def measure_gradient_protocol(model, bundle, cfg, device, step, seed=PROBE_SEED, *, _probe_batches=32):
    """Exactly32 fixed batches48; micro12 preserves the mean-gradient vector.

    Parameter updates are intentionally not simulated here: the actual
    optimizer-delta records belong to the training loop. No probe SGD step or
    RNG advancement is permitted to alter the production model.
    """
    if _probe_batches not in (1,32):
        raise ValueError('Only fixed32 protocol or one early-trace batch is supported')
    named = _named_aligner(model)
    if not named:
        return dict(status='NOT_APPLICABLE', reason='no_aligner_parameters', completed_step=int(step),
                    fixed_probe_batches=_probe_batches, batch_size=48, rows=[], actual_update_source='training_update_records')
    rng = np.random.Generator(np.random.PCG64(seed))
    # Always the same exact 32 batches and augmentations at every progress point.
    ids = rng.integers(len(bundle.datasets['train']), size=(32, 48))[:_probe_batches]
    rots = rng.integers(4, size=(32, 48))[:_probe_batches]
    eps_rng = np.random.Generator(np.random.PCG64(seed+1))
    radius = 2*np.sqrt(eps_rng.random((32,48))); angles = eps_rng.uniform(0,2*np.pi,(32,48))
    epsilons = np.stack((radius*np.sin(angles), radius*np.cos(angles)),axis=-1).astype(np.float32)
    choices = ([48,56,64] if cfg['auxiliary']=='CROP' else [1,2,4,8])
    choice_rng = np.random.Generator(np.random.PCG64(seed+2))
    aux = choice_rng.choice(choices, size=32)[:_probe_batches]
    rows = []
    params = [p for _,p in named]
    with preserve_runtime(model):
        # Every2 scheduling is observed at this precise completed update; no
        # forced epsilon-on batch changes the protocol at odd progress points.
        objective_step = max(1,int(step))
        for batch_index in range(_probe_batches):
            gradients = {k:[torch.zeros_like(p, device='cpu') for p in params] for k in ('rec','eps','struct','scale')}
            losses = {k:0. for k in gradients}
            for start in range(0,48,12):
                batch_ids = ids[batch_index,start:start+12].tolist()
                batch_rots = rots[batch_index,start:start+12].tolist()
                batch = bundle.batch('train',batch_ids,batch_rots,device=device)
                assets = objective_assets(model,bundle,cfg,batch_ids,batch_rots,device)
                value = compute_objective(model,batch,cfg,objective_step,
                    epsilon=torch.tensor(epsilons[batch_index,start:start+12],device=device),
                    auxiliary_choice=int(aux[batch_index]) if cfg['auxiliary']!='NONE' else None,**assets)
                for key, source in (('rec','rec'),('eps','epsilon'),('struct','struct'),('scale','scale')):
                    loss=value[source]; losses[key] += float(loss.detach())/4
                    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True) if loss.requires_grad else (None,)*len(params)
                    for j,g in enumerate(gs):
                        if g is not None: gradients[key][j].add_(g.detach().cpu(),alpha=.25)
                del value,batch
            effective_struct=cfg['lambda_struct']*(min(objective_step/cfg['struct_ramp_updates'],1.) if cfg['struct_ramp_updates'] else 1.)
            weights=dict(rec=1.,eps=cfg['lambda_epsilon'],struct=effective_struct,scale=cfg['lambda_scale'])
            for row in gradient_vector_summary(gradients,named,weights):
                rows.append(dict(row,probe_batch=batch_index,completed_step=int(step),raw_losses=losses,
                                 epsilon_active=bool(cfg['lambda_epsilon'] and objective_step%cfg['epsilon_every']==0)))
    groups=sorted({r['group'] for r in rows})
    summaries=[]
    for group in groups:
        part=[r for r in rows if r['group']==group]
        summaries.append(dict(group=group,n_batches=_probe_batches,
            raw_norm={k:float(np.mean([r['raw_norm'][k] for r in part])) for k in losses},
            weighted_norm={k:float(np.mean([r['weighted_norm'][k] for r in part])) for k in losses},
            cosine={k:float(np.mean([r['cosine'][k] for r in part if r['cosine'][k] is not None]))
                    if any(r['cosine'][k] is not None for r in part) else None for k in part[0]['cosine']},
            cosine_valid_counts={k:sum(r['cosine'][k] is not None for r in part) for k in part[0]['cosine']},
            total_weighted_norm=float(np.mean([r['total_weighted_norm'] for r in part])),
            actual_parameter_update_norm=None,update_na_reason='read_actual_delta_from_training_step_record'))
    return dict(status='COMPLETE' if _probe_batches==32 else 'EARLY_HEAD_BODY_TRACE',completed_step=int(step),fixed_probe_batches=_probe_batches,batch_size=48,micro_batch=12,
                seed=seed,objective_step=max(1,int(step)),stream_sha256=digest(dict(ids=ids.tolist(),rots=rots.tolist(),eps=epsilons.tolist(),aux=aux.tolist())),
                rows=rows,summary=summaries,grad_definition='norm_of_accumulated_effective48_mean_gradient; then32batch_mean',
                actual_update_source='training_update_records',training_state_mutated=False)


collect_gradient_protocol = measure_gradient_protocol


def lightweight_gradient(model,bundle,cfg,step,device='cuda'):
    if step not in (0,1,100):
        raise ValueError('Early head/body trace is preregistered at0/1/100')
    return measure_gradient_protocol(model,bundle,cfg,device,step,_probe_batches=1)


def native_predict(model,bundle,cfg,p,ms,l,split,index):
    shift = bundle.case_shifts(cfg['case_id'],split)[index:index+1].to(p.device) if model.policy in OFFLINE else None
    return model(p,ms,l,shift=shift)


def make_correction(model,bundle,cfg):
    """Unit-intensity PAN and explicit B/L reference, in current pixel units."""
    def correction(split,p,reference,index,mode):
        if model.policy=='LEARNED':
            return model.predict_correction(p*2-1,reference*2-1)
        if model.policy=='GLOBAL_LEARNED':
            return model.global_shift[None].expand(len(p),-1)
        if model.policy in ('BYPASS','ZERO'):
            return p.new_zeros((len(p),2))
        if model.policy in ('GLOBAL_FIXED','SHUFFLED') or mode=='HOLD_NATIVE_CACHE':
            return bundle.case_shifts(cfg['case_id'],split)[index:index+1].to(p.device)
        target='GT' if model.policy=='TRAIN_GT_PROXY' and split=='train' else 'BICUBIC_MS' if model.policy=='PER_IMAGE_BICUBIC' else 'LMS'
        result=estimate_shift(p[0].cpu().numpy(),reference[0].cpu().numpy(),
            descriptor='intensity' if model.policy=='PER_IMAGE_INTENSITY' else 'gradient',
            target=target,split=split,max_dn=1.)
        return p.new_tensor(result['shift'])[None]
    return correction


def _train_corrections(model,bundle,cfg,device):
    corrections=[]
    with torch.no_grad():
        for start in range(0,len(bundle.datasets['train']),48):
            ids=list(range(start,min(start+48,len(bundle.datasets['train']))))
            batch=bundle.batch('train',ids,device=device)
            if model.policy in OFFLINE:
                c=bundle.case_shifts(cfg['case_id'],'train')[ids].to(device)
            else:
                c=model.correction(batch['pan'],batch['ms'],batch['lms'])
            corrections.extend(c.cpu().numpy())
    return np.asarray(corrections)


def run_protocol(model,bundle,cfg,run_dir,checkpoint_path,completed_step,device='cuda',gradient_records=None):
    """Complete native P01--P08 protocol for FINAL and HQNR-selected weights."""
    assert_checkpoint_model(model,checkpoint_path)
    cp_sha=file_sha(checkpoint_path)
    identity=dict(checkpoint_sha256=cp_sha,completed_step=int(completed_step),config_sha256=digest(cfg),
                  dataset_sha256=digest(bundle.manifest),probe_seed=PROBE_SEED)
    output=Path(run_dir)/'diagnostics'/f'step_{int(completed_step):06d}'/'report.json'
    if output.exists():
        report=check_seal(json.loads(output.read_text()))
        if report['context']!=identity:raise ValueError('Diagnostic checkpoint/config/data mismatch')
        return report
    with preserve_runtime(model):
        samples={}
        # Full RR20/FR20, frozen first32 canonical train and val IDs (not scores).
        for split in ('train','val','rr','fr'):
            samples[split]=[]
            count=20 if split in ('rr','fr') else min(32,len(bundle.datasets[split]))
            for i in range(count):
                b=bundle.batch(split,[i],device=device)
                p,l=(b['pan']+1)/2,(b['lms']+1)/2
                gt=(b['gt']+1)/2 if split!='fr' else None
                if cfg['a_reference']=='BICUBIC_MS':
                    ref=(F.interpolate(b['ms'],scale_factor=4,mode='bicubic',align_corners=False)+1)/2
                else:ref=l
                # B06 privileged train rule is measured as such, never copied
                # into RR/FR correction callbacks.
                if model.policy=='TRAIN_GT_PROXY' and split=='train':ref=gt
                samples[split].append((i,p,l,gt,ref))

        correction=make_correction(model,bundle,cfg)

        def forced(split,index,p,l,c):
            batch=bundle.batch(split,[index],device=device)
            return model(p*2-1,batch['ms'],l*2-1,shift=c)['prediction']

        from tools.metrics.eval_fr import load_dlpan
        import os
        wald=load_dlpan(os.environ.get('PANCRAFTER_DLPAN',str(bundle.root.parent/'DLPan-Toolbox')))
        def quality(split,index,y):
            raw=((y.detach().float()[0]+1)*(bundle.max_dn/2)).cpu().numpy()
            arr=bundle.datasets[split].arrays
            if split=='rr':return rr_scene(raw,np.asarray(arr['gt'][index]),bundle.sensor)
            return fr_scene(raw,*(np.asarray(arr[k][index]) for k in ('pan','lms','ms')),bundle.sensor,wald)

        costs=[]
        for split in ('rr','fr'):
            batch=bundle.batch(split,[0],device=device)
            precompute = None
            if model.policy == 'GLOBAL_FIXED':
                global_shift=bundle.case_shifts(cfg['case_id'],split)[0:1].to(device)
                register=None
                infer=lambda c:model(batch['pan'],batch['ms'],batch['lms'],shift=global_shift)
                precompute=bundle.offline_shifts('train')['registration_seconds']
            elif model.policy in OFFLINE:
                _,p,l,_,ref=samples[split][0]
                if model.policy=='SHUFFLED':
                    # For SHUFFLED, practical pipeline pays estimation of the
                    # donor image, not reading its cached correction only.
                    donor=bundle.batch(split,[1],device=device)
                    register=lambda: torch.tensor(estimate_shift(((donor['pan'][0]+1)/2).cpu().numpy(),
                        ((donor['lms'][0]+1)/2).cpu().numpy(),max_dn=1.)['shift'],device=device)[None]
                else:register=lambda:correction(split,p,ref,0,'REESTIMATE_CURRENT_INPUT')
                infer=lambda c:model(batch['pan'],batch['ms'],batch['lms'],shift=c)
            else:
                register=None;infer=lambda c:model(batch['pan'],batch['ms'],batch['lms'])
            with torch.no_grad():cost=cost_probe(register,infer,device=device,shape=list(batch['pan'].shape))
            costs.append(dict(cost,split=split,policy=model.policy,params_M=sum(p.numel() for p in model.parameters())/1e6,
                              training_registration_precompute_seconds=precompute,
                              registration_scope='no_per_image_registration; train_global_median_frozen' if model.policy=='GLOBAL_FIXED'
                              else 'reestimated_current_input' if model.policy in OFFLINE else 'included_in_model_forward'))
        if gradient_records is None:
            gradient_records=measure_gradient_protocol(model,bundle,cfg,device,completed_step)
        train_c=_train_corrections(model,bundle,cfg,device)
        report=run_diagnostics(samples=samples,correction=correction,estimator=estimate_shift,
            checkpoint_sha256=cp_sha,context=identity,offline=model.policy in OFFLINE,
            predict_forced=forced,quality=quality,train_corrections=train_c,
            gradient_records=gradient_records,cost_records=costs)
        # Attach measured P03 constants scope to prevent test-scene-derived means.
        report.pop('payload_sha256')
        report.update(train_constant_samples=len(train_c),train_constant_source='entire_unaugmented_train_split',
                      a_probe_reference=cfg['a_reference'],
                      source_rule='B built from original normalized LRMS once; crop/resize same B thereafter; never substituted L',
                      p03_scope='same_U_inference_only',
                      p06_actual_update_source='training_update_records')
        report=seal(report)
        assert_checkpoint_model(model,checkpoint_path)
    atomic_json(output,report,immutable=True)
    return report
