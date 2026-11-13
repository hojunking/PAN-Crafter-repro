"""Sensor-general fresh Teacher/Aligner and frozen-reference Ours Student."""
from copy import deepcopy
import torch
from torch import nn
from directshift48.method import DirectPANAligner,sample_epsilon
from donor30.masking import masked_mean
from panda_unified.core import ReferenceSpatial,build_reconstructor,base_from_batch,reconstruct_with_base,adaptive_weight
from panda_stage2_condq.sources import validate_training_batch
from panda_unified.core import reference_maps
from .geometry import warp_pan,support,inject,frozen_predictions,WARP_CONTRACT
from . import plan


def build_aligner(seed,bands):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        aligner=DirectPANAligner(bands,view_margin=8)
        aligner.outputs=2
        return aligner


class Stage1(nn.Module):
    def __init__(self,cfg,inventory):
        super().__init__();plan.validate(cfg)
        if cfg['stage']!='stage1':raise ValueError('Stage1 configuration required')
        self.cfg,self.inventory=deepcopy(cfg),deepcopy(inventory)
        self.aligner=build_aligner(cfg['stream_seeds']['A_init'],cfg['bands'])
        self.teacher=build_reconstructor(cfg['sensor'],cfg['width'],cfg['stream_seeds']['T_init'],role='T')
        self.spatial=ReferenceSpatial(cfg['sensor'],cfg['reference_mode'])
        self.epsilon_rng=torch.Generator(device='cpu').manual_seed(cfg['stream_seeds']['rr_epsilon'])

    def get_extra_state(self):return {'config_sha256':self.cfg['config_sha256'],'epsilon_rng':self.epsilon_rng.get_state()}
    def set_extra_state(self,state):
        if state['config_sha256']!=self.cfg['config_sha256']:raise ValueError('Stage1 configuration changed')
        self.epsilon_rng.set_state(state['epsilon_rng'].cpu())

    def native_prediction(self,batch):
        batch.validate(self.cfg['bands'])
        u=self.aligner(batch.pan,batch.native_lms)
        pan=warp_pan(batch.pan,u)
        out=reconstruct_with_base(self.teacher,pan,base_from_batch(batch,'interp23'),batch.ms)
        return {**out,'u':u,'pan_aligned':pan}

    def forward(self,batch):
        validate_training_batch(batch,self.inventory)
        out=self.native_prediction(batch);mask=support(batch.pan,out['u'])
        rec=masked_mean((out['y']-batch.gt.detach()).abs(),mask)
        spatial=self.spatial.absolute_loss(out['pan_aligned'],batch.gt.detach(),mask=mask)
        eps=sample_epsilon(len(batch.pan),self.epsilon_rng).to(batch.pan.device)
        ue=self.aligner(inject(batch.pan,eps),batch.native_lms)
        eps_loss=(ue+eps-out['u'].detach()).abs().mean()
        return {'loss':rec+self.cfg['lambda_abs']*spatial+self.cfg['lambda_epsilon']*eps_loss,
                'rec':rec,'spatial':spatial,'epsilon':eps_loss,'valid_fraction':mask.float().mean(),
                'shift_norm':out['u'].detach().norm(dim=1).mean()}


from panda_wv3_s2_eq40.model import build_student

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
            'shift_norm':f['u'].norm(dim=1).mean()}
