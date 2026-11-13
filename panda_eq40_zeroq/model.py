"""Corrected PAN + interp23 MS; frozen reference and gamma interaction only."""
from copy import deepcopy
import torch
from torch import nn
from donor30.masking import masked_mean
from fh12.model import _construction_rng
from panda_method.reconstruction import Reconstructor
from panda_unified.core import (UnifiedReconstructor,UnifiedReconstructionSpec,
    build_reconstructor,ReferenceSpatial,base_from_batch,reconstruct_with_base,
    reference_maps)
from .weighting import adaptive_weight
from panda_stage2_condq.sources import validate_training_batch
from .geometry import warp_pan,frozen_predictions,WARP_CONTRACT
from . import plan

class CorrectedOnlyStudent(UnifiedReconstructor):
    def __init__(self,backbone,spec):
        if spec.pan_channels!=1:raise ValueError('Corrected PAN only required')
        Reconstructor.__init__(self,backbone,spec)
        self.pan_channels,self.role=1,'S'
        self.sampling_variant,self.input_variant='UBIL','CORRECTED'
        self.register_buffer('_unified_identity',self._identity())

def build_student(sensor,width,seed):
    # Preserve the named-fresh S initialization of every common weight. The old
    # residual-input slice was initialized to zero and is now physically absent.
    original=build_reconstructor(sensor,width,seed,role='S')
    with _construction_rng():
        old=original.backbone.input
        stem=nn.Conv2d(old.in_channels-1,width,3,padding=1)
        with torch.no_grad():
            stem.weight[:,:1].copy_(old.weight[:,:1])
            stem.weight[:,1:].copy_(old.weight[:,2:])
            stem.bias.copy_(old.bias)
        original.backbone.input=stem
    return CorrectedOnlyStudent(original.backbone,UnifiedReconstructionSpec(sensor,width,pan_channels=1))

class Stage2(nn.Module):
    def __init__(self,cfg,donor,stats):
        super().__init__();plan.validate(cfg)
        if cfg['stage']!='stage2':raise ValueError('Stage2 configuration required')
        donor.assert_frozen();stats.validate_binding(donor.binding);stats.assert_integrity(full_hash=True)
        if (donor.teacher.spec.sensor!=cfg['sensor'] or donor.teacher.spec.width!=cfg['width']
                or donor.binding['scharr']['reference_indices']!=cfg['reference_bands']
                or donor.binding.get('reference_mode')!=cfg['reference_mode']
                or donor.binding.get('warp')!=WARP_CONTRACT):
            raise ValueError('Donor sensor/width/edge reference/warp differs from Student configuration')
        self.cfg,self.binding=deepcopy(cfg),deepcopy(donor.binding)
        self.aligner,self.teacher=donor.aligner,donor.teacher
        for module in (self.aligner,self.teacher):module.requires_grad_(False).eval()
        self.stats=stats;self.selection_receipt=deepcopy(donor.selection_receipt)
        self.student=build_student(cfg['sensor'],cfg['width'],cfg['stream_seeds']['S_init'])
        self.spatial=ReferenceSpatial(cfg['sensor'],cfg['reference_mode'])

    def train(self,mode=True):
        super().train(mode);self.aligner.eval();self.teacher.eval();return self
    def get_extra_state(self):
        return {'config_sha256':self.cfg['config_sha256'],'donor_binding':self.binding,'calibration_sha256':self.stats.artifact_sha256}
    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('Stage2 donor/calibration/config changed')
    @torch.no_grad()
    def corrected(self,batch):
        u=self.aligner(batch.pan,batch.native_lms)
        return u,warp_pan(batch.pan,u),base_from_batch(batch,'interp23')
    def reconstruct(self,batch,pan,base):
        return reconstruct_with_base(self.student,pan,base,batch.ms)
    def native_prediction(self,batch):
        batch.validate(self.cfg['bands']);u,pan,base=self.corrected(batch)
        return {**self.reconstruct(batch,pan,base),'u':u,'pan_aligned':pan}
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
