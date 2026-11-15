"""Only Student PAN inputs differ; frozen Teacher/q and common mask are shared."""
from copy import deepcopy
import torch
from torch import nn
from donor30.masking import masked_mean
from panda_eq40_zeroq.weighting import adaptive_weight
from panda_unified.core import (ReferenceSpatial,base_from_batch,reconstruct_with_base,
    reference_maps,warp_pan,frozen_predictions,WARP_CONTRACT)
from panda_stage2_condq.sources import validate_training_batch
from .inputs import build_student,pan_views
from . import plan


class Stage2(nn.Module):
    def __init__(self,cfg,donor,stats):
        super().__init__();plan.validate(cfg)
        donor.assert_frozen();stats.validate_binding(donor.binding);stats.assert_integrity(full_hash=True)
        if (donor.teacher.spec.sensor!=cfg['sensor'] or donor.teacher.spec.width!=cfg['width']
            or donor.binding['scharr']['reference_indices']!=cfg['reference_bands']
            or donor.binding.get('reference_mode')!='RGBN' or donor.binding.get('warp')!=WARP_CONTRACT
            or donor.binding.get('donor_run_id')!=cfg['donor_run_id']
            or donor.binding.get('donor_step')!=cfg['donor_step']):
            raise ValueError('Frozen donor differs from registered main reference')
        self.cfg,self.binding=deepcopy(cfg),deepcopy(donor.binding)
        self.aligner,self.teacher=donor.aligner,donor.teacher
        for module in (self.aligner,self.teacher):module.requires_grad_(False).eval()
        self.stats=stats;self.selection_receipt=deepcopy(donor.selection_receipt)
        self.student=build_student(cfg['sensor'],cfg['width'],cfg['stream_seeds']['S_init'],cfg['student_input'])
        self.spatial=ReferenceSpatial(cfg['sensor'],'RGBN')

    def train(self,mode=True):
        super().train(mode);self.aligner.eval();self.teacher.eval();return self

    def get_extra_state(self):
        return {'config_sha256':self.cfg['config_sha256'],'donor_binding':self.binding,
                'calibration_sha256':self.stats.artifact_sha256}

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('Donor/calibration/config changed')

    def reconstruct(self,batch,corrected,base):
        inputs=pan_views(batch.pan,corrected,self.cfg['student_input'])
        return reconstruct_with_base(self.student,inputs,base,batch.ms)

    def native_prediction(self,batch):
        batch.validate(self.cfg['bands'])
        with torch.no_grad():
            u=self.aligner(batch.pan,batch.native_lms)
            corrected=warp_pan(batch.pan,u)
            base=base_from_batch(batch,'interp23')
        result=self.reconstruct(batch,corrected,base)
        # FR corrected reference is always the same one-channel fixed PAN,
        # even when the Student consumes raw PAN or two-channel PAN views.
        return {**result,'u':u,'pan_aligned':corrected,
                'student_pan_input':pan_views(batch.pan,corrected,self.cfg['student_input'])}

    def forward(self,batch):
        validate_training_batch(batch,self.binding)
        f=frozen_predictions(self.aligner,self.teacher,batch);mask=f['loss_mask']
        e,g,q=reference_maps(self.spatial,f['teacher_y'],f['gt_loss'],f['pan_loss'])
        weight=adaptive_weight(e,g,q,self.stats,mask,self.cfg)
        y=self.reconstruct(batch,f['pan_input'],f['ms_base_frame'])['y']
        error=(y-batch.gt.detach()).abs().mean(1,keepdim=True)
        return {'loss':masked_mean(weight*error,mask),'rec':masked_mean(error,mask),
            'weight_mean':masked_mean(weight,mask),'valid_fraction':mask.float().mean(),
            'q_zero_fraction':masked_mean((q==0).float(),mask),
            'shift_norm':f['u'].norm(dim=1).mean()}
