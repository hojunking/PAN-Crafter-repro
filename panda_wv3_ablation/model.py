"""Existing WV3 operators and exact anchored losses with explicit component switches."""
from copy import deepcopy
import torch
from torch import nn
from directshift48.method import sample_epsilon, ScharrAbsolute
from donor30.masking import masked_mean
from panda_band30.aligner import BandPANAligner, AllBandScharr
from panda_band30.reconstruction import (BandReconstructionSpec, base_from_batch,
    reconstruct_with_base, build_reconstructor as build_teacher)
from panda_tune30.reconstruction import StudentReconstructionSpec, build_reconstructor
from panda_tune30.method import warp_support
from panda_tune30.warp import warp_pan as lanczos
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
        self.spatial = AllBandScharr(2)
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


def global_rank(stats,q):
    """Exact unconditional ECDF of the same q population, without a giant concatenation."""
    stats.assert_integrity()
    tables = stats._q  # Version-pinned immutable CalibrationStats tables.
    if not tables or any(t.device != q.device for t in tables):
        raise ValueError('Exact q tables must be resident on the query device')
    count = sum(t.numel() for t in tables)
    if count == 0:
        raise ValueError('Empty train q distribution')
    values = q.contiguous()
    # Integer counts keep exact ranks even when the training population is large.
    ranks = torch.zeros_like(q,dtype=torch.int64)
    for table in tables:
        ranks += torch.searchsorted(table,values,right=True)
    return ranks.to(torch.float64).div(count).to(torch.float32)


def adaptive_weight(e_t,g,q,stats,mask,cfg):
    """All cue maps and weights detached; normalize loss by pixels, never by sum(w)."""
    with torch.no_grad():
        rank = torch.zeros_like(mask,dtype=torch.float32)
        cue = rank.clone()
        key = cfg['case_id']
        if cfg['alpha'] or (cfg['gamma'] and key!='S2_D2_NO_E_CUE'):
            if e_t is None:
                raise ValueError('Reference error required by the registered weight')
            cue = e_t/(e_t+stats.tau_e).clamp_min(torch.finfo(torch.float32).tiny)
        if cfg['gamma']:
            sl=(...,slice(8,-8),slice(8,-8))
            rank[sl] = (global_rank(stats,q[sl]) if key=='S2_D1_GLOBAL_RANK'
                        else stats.lookup_exact(g[sl],q[sl]))
        weight = (1+cfg['gamma']*rank if key=='S2_D2_NO_E_CUE'
                  else 1+cfg['alpha']*cue+cfg['gamma']*cue*rank)
        if key=='S2_D3_MEAN_WEIGHT':
            mean = (weight*mask).sum((1,2,3),keepdim=True)/mask.sum((1,2,3),keepdim=True)
            weight = mean.expand_as(weight)
        return weight


class Stage2(nn.Module):
    def __init__(self,cfg,donor,stats=None):
        super().__init__()
        self.cfg, self.binding = deepcopy(cfg),deepcopy(donor.binding)
        self.aligner,self.teacher = donor.aligner,donor.teacher
        for model in (self.aligner,self.teacher):
            model.requires_grad_(False).eval()
        self.stats = stats
        two = cfg['student_input']!='CORRECTED'
        spec = (StudentReconstructionSpec(8,112,(1,2,3),'P0','BICUBIC_MS',2) if two else
                BandReconstructionSpec(8,112,(1,2,3),'P0','BICUBIC_MS',1))
        self.student = build_reconstructor(spec,seed=cfg['stream_seeds']['S_init'],role='S',
            sampling_variant='UBIL',input_variant='CORRECTED_DIFF' if two else 'CORRECTED')
        self.spatial = ScharrAbsolute('WV3')

    def train(self,mode=True):
        super().train(mode)
        self.aligner.eval(); self.teacher.eval()
        return self

    def get_extra_state(self):
        return {'config_sha256':self.cfg['config_sha256'],'donor_binding':self.binding,
                'calibration_sha256':None if self.stats is None else self.stats.artifact_sha256}

    def set_extra_state(self,state):
        if state != self.get_extra_state():
            raise ValueError('Stage2 case/donor/calibration changed during resume')

    @torch.no_grad()
    def corrected(self,batch):
        u = self.aligner(batch.pan,batch.native_lms)
        return u,lanczos(batch.pan,u),base_from_batch(batch,'interp23')

    def reconstruct(self,batch,pan,base):
        kind = self.cfg['student_input']
        inputs = (pan if kind=='CORRECTED' else torch.cat((pan,
                  batch.pan if kind=='CORRECTED_ORIGINAL' else pan-batch.pan),dim=1))
        return reconstruct_with_base(self.student,inputs,base,batch.ms)

    def native_prediction(self,batch):
        batch.validate(8)
        u,pan,base = self.corrected(batch)
        return {**self.reconstruct(batch,pan,base),'u':u,'pan_aligned':pan}

    def forward(self,batch):
        validate_training_batch(batch,self.binding)
        u,pan,base = self.corrected(batch)
        mask = warp_support(*batch.pan.shape[-2:],u,'s100',include_teacher=True)['loss_mask']
        e,g,q = None,None,None
        if self.cfg['alpha'] or self.cfg['gamma']:
            if self.stats is None:
                raise ValueError('Exact anchored train calibration required')
            with torch.no_grad():
                if self.cfg['case_id']!='S2_D2_NO_E_CUE':
                    t = reconstruct_with_base(self.teacher,bicubic(batch.pan,u),base,batch.ms)['y']
                    e = (t-batch.gt).abs().mean(1,keepdim=True)
                if self.cfg['gamma']:
                    edges = self.spatial.descriptor(self.spatial.rgb_luminance(batch.gt))
                    q = (self.spatial.descriptor(pan)-edges).abs().mean(1,keepdim=True)
                    g = edges.mean(1,keepdim=True)
        weight = adaptive_weight(e,g,q,self.stats,mask,self.cfg)
        y = self.reconstruct(batch,pan,base)['y']
        error = (y-batch.gt.detach()).abs().mean(1,keepdim=True)
        return {'loss':masked_mean(weight*error,mask),'rec':masked_mean(error,mask),
                'weight_mean':masked_mean(weight,mask),'valid_fraction':mask.float().mean(),
                'shift_norm':u.norm(dim=1).mean()}
