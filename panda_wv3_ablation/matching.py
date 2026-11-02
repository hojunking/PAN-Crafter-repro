"""Image-specific Scharr translation from observed MS only; no GT argument."""
import itertools
import torch
from torch import nn
from directshift48.method import ScharrAbsolute
from pa.warp import warp_pan


class IdentityAlignment(nn.Module):
    outputs = 2

    def forward(self, pan, lms):
        return pan.new_zeros((len(pan), 2))


class ScharrAlignment(nn.Module):
    outputs = 2
    # Largest symmetric integer search radius preserving bicubic ROI8/stencil7.
    # This is unrelated to the learned Aligner's R=3 synthetic epsilon disk.
    bound = 5.
    fine_step = .25

    def __init__(self, candidate_chunk=8):
        super().__init__()
        self.edges = ScharrAbsolute('WV3')
        self.candidate_chunk = candidate_chunk

    @torch.no_grad()
    def forward(self, pan, lms):
        if (pan.ndim != 4 or pan.shape[1] != 1 or lms.shape != (len(pan),8,*pan.shape[-2:])
                or min(pan.shape[-2:]) <= 16 or pan.dtype != torch.float32
                or not bool(torch.isfinite(pan).all()) or not bool(torch.isfinite(lms).all())):
            raise ValueError('Finite native PAN and observed mean8 LMS on the same grid required')
        reference = self.edges.descriptor(lms.mean(1,keepdim=True))[...,8:-8,8:-8]
        b,h,w = len(pan), *pan.shape[-2:]
        best_u = pan.new_zeros((b,2))
        best_score = pan.new_full((b,),float('inf'))

        def search(candidates):
            nonlocal best_score, best_u
            for start in range(0, candidates.shape[1], self.candidate_chunk):
                u = candidates[:,start:start+self.candidate_chunk]
                k = u.shape[1]
                shifted = warp_pan(pan[:,None].expand(b,k,1,h,w).reshape(b*k,1,h,w),u.reshape(b*k,2))
                edges = self.edges.descriptor(shifted)[...,8:-8,8:-8].reshape(b,k,2,h-16,w-16)
                scores = (edges-reference[:,None]).abs().mean((2,3,4))
                scores[(u.abs()>self.bound).any(2)] = float('inf')
                for j in range(k):
                    v,s = u[:,j],scores[:,j]
                    old_norm,new_norm = best_u.square().sum(1),v.square().sum(1)
                    old_l1,new_l1 = best_u.abs().sum(1),v.abs().sum(1)
                    tie = ((new_norm<old_norm) | ((new_norm==old_norm) &
                        ((new_l1<old_l1) | ((new_l1==old_l1) &
                        ((v[:,0]<best_u[:,0]) | ((v[:,0]==best_u[:,0]) & (v[:,1]<best_u[:,1])))))))
                    take = (s<best_score) | ((s==best_score)&tie)
                    best_score = torch.where(take,s,best_score)
                    best_u = torch.where(take[:,None],v,best_u)

        coarse = pan.new_tensor(list(itertools.product(range(-int(self.bound),int(self.bound)+1),repeat=2)))
        search(coarse[None].expand(b,-1,2))
        offset = pan.new_tensor(list(itertools.product([i/4 for i in range(-4,5)],repeat=2)))
        search(best_u.clone()[:,None]+offset[None])
        if not bool(torch.isfinite(best_score).all()):
            raise ValueError('No finite Scharr candidate')
        return best_u
