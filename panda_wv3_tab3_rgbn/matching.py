"""Nonlearned per-image RGBN Scharr translation; inputs never include GT."""
from collections import OrderedDict
import hashlib
import itertools
import torch
from torch import nn
from panda_unified.core import ReferenceSpatial, warp_pan, support


class ScharrAlignment(nn.Module):
    outputs = 2
    bound = 5.
    fine_step = .25

    def __init__(self, candidate_chunk=8, cache_size=60000):
        super().__init__()
        if candidate_chunk < 1 or cache_size < 0:
            raise ValueError('Positive chunk and nonnegative cache size required')
        self.edges = ReferenceSpatial('WV3', 'RGBN')
        self.candidate_chunk, self.cache_size = candidate_chunk, cache_size
        # Derived acceleration only; no GT, learned state, approximate key, or persistent weights.
        self._cache = OrderedDict()

    @torch.no_grad()
    def forward(self, pan, lms):
        if (pan.ndim != 4 or pan.shape[1] != 1 or lms.shape != (len(pan),8,*pan.shape[-2:])
                or min(pan.shape[-2:]) <= 24 or pan.dtype != torch.float32 or lms.dtype != pan.dtype
                or pan.device != lms.device or not bool(torch.isfinite(pan).all())
                or not bool(torch.isfinite(lms).all())):
            raise ValueError('Finite FP32 native PAN and observed eight-band LMS required')
        if not self.cache_size:
            return self._search(pan, lms)
        keys = []
        for p,l in zip(pan.detach().cpu(), lms.detach().cpu()):
            h = hashlib.sha256(repr((tuple(p.shape), str(p.dtype), str(pan.device))).encode())
            h.update(p.contiguous().numpy().tobytes()); h.update(l.contiguous().numpy().tobytes())
            keys.append(h.digest())
        missing = [i for i,k in enumerate(keys) if k not in self._cache]
        computed = self._search(pan[missing], lms[missing]).cpu().tolist() if missing else []
        found = {i:u for i,u in zip(missing, computed)}
        values = [found[i] if i in found else self._cache[k] for i,k in enumerate(keys)]
        for k,u in zip(keys, values):
            self._cache[k] = u; self._cache.move_to_end(k)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return pan.new_tensor(values)

    @torch.no_grad()
    def _search(self, pan, lms):
        reference = self.edges.descriptor(self.edges.reference(lms))
        b,h,w = len(pan), *pan.shape[-2:]
        best_u, best_score = pan.new_zeros((b,2)), pan.new_full((b,),float('inf'))

        def search(candidates):
            nonlocal best_score, best_u
            for start in range(0, candidates.shape[1], self.candidate_chunk):
                u = candidates[:,start:start+self.candidate_chunk]; k = u.shape[1]
                images = pan[:,None].expand(b,k,1,h,w).reshape(b*k,1,h,w)
                shifts = u.reshape(b*k,2)
                shifted = warp_pan(images, shifts)
                mask = support(images, shifts).reshape(b,k,1,h,w)
                edges = self.edges.descriptor(shifted).reshape(b,k,2,h,w)
                scores = ((edges-reference[:,None]).abs()*mask).sum((2,3,4)) / (2*mask.sum((2,3,4)))
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

        coarse = pan.new_tensor(list(itertools.product(range(-5,6),repeat=2)))
        search(coarse[None].expand(b,-1,2))
        offset = pan.new_tensor(list(itertools.product([i/4 for i in range(-4,5)],repeat=2)))
        search(best_u.clone()[:,None]+offset[None])
        if not bool(torch.isfinite(best_score).all()):
            raise ValueError('No finite Scharr candidate')
        return best_u
