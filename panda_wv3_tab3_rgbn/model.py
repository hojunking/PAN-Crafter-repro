"""Stage 1 Table 3 components on one RGBN/Lanczos/UBIL/common-mask implementation."""
from copy import deepcopy
import torch
from torch import nn
from directshift48.method import DirectPANAligner, sample_epsilon
from donor30.masking import masked_mean
from panda_unified.core import ReferenceSpatial, build_reconstructor, base_from_batch, reconstruct_with_base
from panda_stage2_condq.sources import validate_training_batch
from .geometry import warp_pan, support, inject
from .matching import ScharrAlignment
from . import plan


def build_aligner(seed, bands):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        aligner = DirectPANAligner(bands, view_margin=8)
        aligner.outputs = 2
        return aligner


class Stage1(nn.Module):
    def __init__(self, cfg, inventory):
        super().__init__(); plan.validate(cfg)
        self.cfg, self.inventory = deepcopy(cfg), deepcopy(inventory)
        self.aligner = (ScharrAlignment() if cfg['alignment']=='SCHARR'
                        else build_aligner(cfg['stream_seeds']['A_init'], cfg['bands']))
        self.teacher = build_reconstructor('WV3',112,cfg['stream_seeds']['T_init'],role='T')
        self.spatial = ReferenceSpatial('WV3','RGBN')
        self.epsilon_rng = torch.Generator(device='cpu').manual_seed(cfg['stream_seeds']['rr_epsilon'])

    def get_extra_state(self):
        return {'config_sha256':self.cfg['config_sha256'], 'epsilon_rng':self.epsilon_rng.get_state()}

    def set_extra_state(self, state):
        if state['config_sha256'] != self.cfg['config_sha256']:
            raise ValueError('Stage1 configuration changed')
        self.epsilon_rng.set_state(state['epsilon_rng'].cpu())

    def native_prediction(self, batch):
        batch.validate(8)
        u = self.aligner(batch.pan,batch.native_lms)
        pan = warp_pan(batch.pan,u)
        out = reconstruct_with_base(self.teacher,pan,base_from_batch(batch,'interp23'),batch.ms)
        return {**out,'u':u,'pan_aligned':pan}

    def forward(self, batch):
        validate_training_batch(batch,self.inventory)
        out = self.native_prediction(batch); mask = support(batch.pan,out['u'])
        rec = masked_mean((out['y']-batch.gt.detach()).abs(),mask)
        spatial = (self.spatial.absolute_loss(out['pan_aligned'],batch.gt.detach(),mask=mask)
                   if self.cfg['lambda_abs'] else rec.new_zeros(()))
        eps_loss = rec.new_zeros(())
        if self.cfg['lambda_epsilon']:
            eps = sample_epsilon(len(batch.pan),self.epsilon_rng).to(batch.pan.device)
            ue = self.aligner(inject(batch.pan,eps),batch.native_lms)
            eps_loss = (ue+eps-out['u'].detach()).abs().mean()
        return {'loss':rec+self.cfg['lambda_abs']*spatial+self.cfg['lambda_epsilon']*eps_loss,
                'rec':rec,'spatial':spatial,'epsilon':eps_loss,'valid_fraction':mask.float().mean(),
                'shift_norm':out['u'].detach().norm(dim=1).mean()}
