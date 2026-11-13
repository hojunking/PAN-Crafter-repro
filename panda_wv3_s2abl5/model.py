"""Input alignment and two-factor adaptive fitting, under one shared reference."""
from copy import deepcopy
import torch
from torch import nn
from donor30.masking import masked_mean, validate_mask
from panda_eq40_zeroq.model import build_student
from panda_eq40_zeroq.weighting import adaptive_weight as product_weight, conditional_rank
from panda_unified.core import (ReferenceSpatial,base_from_batch,reconstruct_with_base,
    reference_maps,warp_pan,frozen_predictions,WARP_CONTRACT)
from panda_stage2_condq.sources import validate_training_batch
from . import plan

@torch.no_grad()
def weights(e,g,q,stats,mask,cfg):
    variant=cfg['weight_variant']
    if variant=='PRODUCT':return product_weight(e,g,q,stats,mask,cfg)
    validate_mask(mask.float(),mask)
    w=torch.ones_like(mask,dtype=torch.float32)
    if variant=='UNIFORM':return w
    if variant=='DE_ONLY':
        tau=float(stats.tau_e)
        if not torch.isfinite(torch.tensor(tau)) or tau<0:raise ValueError('Invalid tau_e')
        cue=e.detach()/(e.detach()+tau).clamp_min(torch.finfo(torch.float32).tiny)
        w[mask]+=cfg['gamma']*cue[mask]
    elif variant=='RQ_ONLY':
        w[mask]+=cfg['gamma']*conditional_rank(stats,g[mask],q[mask])
    else:raise ValueError('Unknown component case; global rank is not registered')
    if not bool(torch.isfinite(w).all()):raise ValueError('Nonfinite fitting weight')
    return w.detach()

class Stage2(nn.Module):
    def __init__(self,cfg,donor,stats):
        super().__init__();plan.validate(cfg)
        donor.assert_frozen();stats.validate_binding(donor.binding);stats.assert_integrity(full_hash=True)
        if (donor.teacher.spec.sensor!='WV3' or donor.teacher.spec.width!=112
            or donor.binding['scharr']['reference_indices']!=cfg['reference_bands']
            or donor.binding.get('reference_mode')!='RGBN' or donor.binding.get('warp')!=WARP_CONTRACT):
            raise ValueError('Fixed donor architecture/reference differs')
        self.cfg,self.binding=deepcopy(cfg),deepcopy(donor.binding)
        self.aligner,self.teacher=donor.aligner,donor.teacher
        for module in (self.aligner,self.teacher):module.requires_grad_(False).eval()
        self.stats=stats;self.selection_receipt=deepcopy(donor.selection_receipt)
        self.student=build_student('WV3',112,cfg['stream_seeds']['S_init'])
        self.student.input_variant=cfg['student_input']
        self.spatial=ReferenceSpatial('WV3','RGBN')

    def train(self,mode=True):
        super().train(mode);self.aligner.eval();self.teacher.eval();return self
    def get_extra_state(self):
        return {'config_sha256':self.cfg['config_sha256'],'donor_binding':self.binding,
                'calibration_sha256':self.stats.artifact_sha256}
    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('Donor/calibration/config changed')
    def reconstruct(self,batch,pan,base):
        return reconstruct_with_base(self.student,pan,base,batch.ms)
    def native_prediction(self,batch):
        batch.validate(self.cfg['bands'])
        with torch.no_grad():
            u=self.aligner(batch.pan,batch.native_lms)
            reference_pan=warp_pan(batch.pan,u)
            base=base_from_batch(batch,'interp23')
        input_pan=reference_pan if self.cfg['input_alignment'] else batch.pan
        return {**self.reconstruct(batch,input_pan,base),'u':u,'pan_aligned':reference_pan,
                'student_pan_input':input_pan}
    def forward(self,batch):
        validate_training_batch(batch,self.binding)
        f=frozen_predictions(self.aligner,self.teacher,batch);mask=f['loss_mask']
        e,g,q=reference_maps(self.spatial,f['teacher_y'],f['gt_loss'],f['pan_loss'])
        weight=weights(e,g,q,self.stats,mask,self.cfg)
        pan=f['pan_input'] if self.cfg['input_alignment'] else batch.pan
        y=self.reconstruct(batch,pan,f['ms_base_frame'])['y']
        error=(y-batch.gt.detach()).abs().mean(1,keepdim=True)
        mean=masked_mean(weight,mask)
        variance=(masked_mean(weight.square(),mask)-mean.square()).clamp_min(0)
        return {'loss':masked_mean(weight*error,mask),'rec':masked_mean(error,mask),
            'weight_mean':mean,'weight_std':variance.sqrt(),
            'valid_fraction':mask.float().mean(),'q_zero_fraction':masked_mean((q==0).float(),mask),
            'shift_norm':f['u'].norm(dim=1).mean()}
