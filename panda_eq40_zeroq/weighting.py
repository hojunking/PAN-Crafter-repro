"""Zero discrepancy must contribute zero conditional spatial weight."""
import torch
from panda_unified.core import adaptive_weight as calibrated_weight

POLICY='EXACT_ZERO_RANK_ZERO_POSITIVE_RIGHT_ECDF_V1'


@torch.no_grad()
def conditional_rank(stats,g,q):
    # Preserve all original validation and positive-value tie conventions.
    rank=stats.lookup_exact(g,q)
    return torch.where(q==0,torch.zeros_like(rank),rank)


class ZeroSafeStats:
    def __init__(self,stats):self.stats=stats
    @property
    def tau_e(self):return self.stats.tau_e
    def lookup_exact(self,g,q):return conditional_rank(self.stats,g,q)


@torch.no_grad()
def adaptive_weight(e,g,q,stats,mask,cfg):
    if cfg.get('q_zero_policy')!=POLICY:
        raise ValueError('Explicit zero-discrepancy policy required')
    return calibrated_weight(e,g,q,ZeroSafeStats(stats),mask,cfg)
