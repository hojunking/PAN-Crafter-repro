"""Existing WV3 operators and exact anchored losses with explicit component switches."""
from copy import deepcopy
import torch
from torch import nn
from directshift48.method import sample_epsilon, ScharrAbsolute
from donor30.masking import masked_mean
from panda_band30.aligner import BandPANAligner
from panda_band30.reconstruction import (BandReconstructionSpec, base_from_batch,
    reconstruct_with_base, build_reconstructor as build_teacher)
from pa.warp import warp_pan as bicubic, warp_support_mask
from panda_stage2_condq.sources import validate_training_batch
from .matching import IdentityAlignment, ScharrAlignment


def build_aligner(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return BandPANAligner(2)


def support(pan,u):
    mask = warp_support_mask(*pan.shape[-2:],u.detach())
    if not bool(mask[...,7:-7,7:-7].all()):
        raise ValueError('SUPPORT_VIOLATION: bicubic correction must preserve fixed ROI8/stencil7')
    return mask


class Stage1(nn.Module):
    def __init__(self,cfg,inventory):
        super().__init__()
        self.cfg, self.inventory = deepcopy(cfg), deepcopy(inventory)
        kind = cfg['alignment']
        self.aligner = (build_aligner(cfg['stream_seeds']['A_init']) if kind=='LEARNED' else
                        ScharrAlignment() if kind=='IMAGE_SPECIFIC_NONLEARNED_GLOBAL_SCHARR' else
                        IdentityAlignment())
        self.teacher = build_teacher(BandReconstructionSpec(8,112,(1,2,3),'P0','BICUBIC_MS',1),
                                    seed=cfg['stream_seeds']['T_init'],role='T',pan_channels=1)
        self.spatial = ScharrAbsolute('WV3')
        self.epsilon_rng = torch.Generator(device='cpu').manual_seed(cfg['stream_seeds']['rr_epsilon'])

    def get_extra_state(self):
        return {'config_sha256':self.cfg['config_sha256'],'epsilon_rng':self.epsilon_rng.get_state()}

    def set_extra_state(self,state):
        if state['config_sha256'] != self.cfg['config_sha256']:
            raise ValueError('Stage1 case/seed changed during resume')
        self.epsilon_rng.set_state(state['epsilon_rng'].cpu())

    def native_prediction(self,batch):
        batch.validate(8)
        u = self.aligner(batch.pan,batch.native_lms)
        # RAW is truly unwarped, including at floating point precision.
        pan = batch.pan if self.cfg['alignment']=='NONE' else bicubic(batch.pan,u)
        out = reconstruct_with_base(self.teacher,pan,base_from_batch(batch,'interp23'),batch.ms)
        return {**out,'u':u,'pan_aligned':pan}

    def forward(self,batch):
        validate_training_batch(batch,self.inventory)
        out = self.native_prediction(batch)
        mask = (torch.ones_like(batch.pan,dtype=torch.bool) if self.cfg['alignment']=='NONE'
                else support(batch.pan,out['u']))
        rec = masked_mean((out['y']-batch.gt.detach()).abs(),mask)
        spatial,eps_loss = rec.new_zeros(()),rec.new_zeros(())
        if self.cfg['lambda_abs']:
            spatial = self.spatial.absolute_loss(out['pan_aligned'],batch.gt.detach())
        if self.cfg['lambda_epsilon']:
            eps = sample_epsilon(len(batch.pan),self.epsilon_rng).to(batch.pan.device)
            ue = self.aligner(bicubic(batch.pan,eps),batch.native_lms)
            eps_loss = (ue+eps-out['u'].detach()).abs().mean()
        loss = rec+self.cfg['lambda_abs']*spatial+self.cfg['lambda_epsilon']*eps_loss
        result={'loss':loss,'rec':rec,'spatial':spatial,'epsilon':eps_loss,
                'valid_fraction':mask.float().mean(),'shift_norm':out['u'].detach().norm(dim=1).mean()}
        if isinstance(self.aligner,ScharrAlignment):
            u=out['u'].detach()
            result.update(scharr_bound_hit_fraction=(u.abs()>=self.aligner.bound).any(1).float().mean(),
                          scharr_fractional_fraction=(u!=u.round()).any(1).float().mean())
        return result
