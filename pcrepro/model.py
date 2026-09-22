"""Isolated PAN-Crafter Eq.(11) correction; no PANDA/LP feature branch.

The three-scale D224 skeleton and direct mode modulation are reused without
editing legacy source. Local gather is F.unfold (zero padding), not a learned
convolution. The same PAN's Gaussian LP is used only as PAN-mode residual base.
"""
from __future__ import annotations

import hashlib
import torch
from torch import nn
from torch.nn import functional as F

from model.pancrafter_paper import PANCrafterPaper


def state_hash(state):
    """Stable tensor identity without importing an alignment campaign model."""
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous()
        tensor_digest=hashlib.sha256(str((str(value.dtype),tuple(value.shape))).encode()
            +value.numpy().tobytes()).hexdigest()
        digest.update(name.encode());digest.update(tensor_digest.encode())
    return digest.hexdigest()


def modes(value, batch, device):
    if value is None or isinstance(value,str) and value == 'MS':
        value = torch.ones(batch,device=device,dtype=torch.long)
    elif isinstance(value,str) and value == 'PAN':
        value = torch.zeros(batch,device=device,dtype=torch.long)
    else:
        value = torch.as_tensor(value,device=device)
        if value.ndim == 0: value = value.expand(batch)
    if value.shape != (batch,) or not bool(((value==0)|(value==1)).all()):
        raise ValueError('Mode must be per-sample MS=1 or PAN=0')
    return value.long()


class Eq11Attention(nn.Module):
    def __init__(self,dim,num_heads=8,ms_channels=8):
        super().__init__()
        if dim % num_heads: raise ValueError('Width must divide heads')
        self.dim,self.num_heads,self.ms_channels=dim,num_heads,ms_channels
        self.head_dim=dim//num_heads
        self.q=nn.Conv2d(dim+ms_channels,dim,3,padding=1,bias=False)
        self.kv_ms=nn.Conv2d(dim+ms_channels,2*dim,3,padding=1,bias=False)
        self.kv_pan=nn.Conv2d(dim+ms_channels,2*dim,3,padding=1,bias=False)
        self.proj_pan=nn.Conv2d(dim,dim,1)
        self.proj_ms=nn.Conv2d(dim,dim,1)

    @staticmethod
    def gather(value):
        b,c,h,w=value.shape
        return F.unfold(value,kernel_size=3,padding=1).reshape(b,c,9,h,w)

    def local_attention(self,q,key,value):
        b,c,h,w=q.shape
        query=q.reshape(b,self.num_heads,self.head_dim,1,h,w)*self.head_dim**-.5
        key=self.gather(key).reshape(b,self.num_heads,self.head_dim,9,h,w)
        value=self.gather(value).reshape(b,self.num_heads,self.head_dim,9,h,w)
        weight=(query*key).sum(2,keepdim=True).softmax(3)
        return (weight*value).sum(3).reshape(b,c,h,w)

    def forward(self,x,ms,_unused_lpan,pan,mode):
        if ms.shape != (x.shape[0],self.ms_channels,*x.shape[-2:]) or pan.shape != (x.shape[0],1,*x.shape[-2:]):
            raise ValueError('Raw modality priors must match feature scale')
        mode=modes(mode,len(x),x.device)
        repeated=pan.expand(-1,self.ms_channels,-1,-1)
        condition=torch.where(mode[:,None,None,None].bool(),ms,repeated)
        # Explicit modality-first order implements the declared Eq.(11) recipe.
        query=self.q(torch.cat((condition,x),1))
        k_ms,v_ms=self.kv_ms(torch.cat((ms,x),1)).chunk(2,1)
        k_pan,v_pan=self.kv_pan(torch.cat((repeated,x),1)).chunk(2,1)
        return self.proj_pan(self.local_attention(query,k_pan,v_pan)), self.proj_ms(self.local_attention(query,k_ms,v_ms))


class CorrectedPANCrafter(PANCrafterPaper):
    def __init__(self,num_bands=8,max_pixel=2047,*,hidden_size=128,depth=(2,2,4),num_heads=8):
        if num_bands not in (4,8) or max_pixel not in (1023,2047):
            raise ValueError('Explicit C4/C8 and DN1023/2047 required')
        super().__init__(out_channels=num_bands,hidden_size=hidden_size,depth=depth,
            num_heads=num_heads,mlp_ratio=4.,norm='ln',in_mode='paper',ka=3,ks=3,
            attn_locations=['enc','btl','dec'],mode_modulation=True)
        self.num_bands,self.max_pixel=num_bands,max_pixel
        self.hidden_size,self.num_heads=hidden_size,num_heads
        for block in (self.cond2_e,self.cond_bot,self.cond2_d):
            block.attn=Eq11Attention(hidden_size,num_heads,num_bands)
        coordinate=torch.arange(-20,21,dtype=torch.float64)
        kernel=torch.exp(-coordinate.square()/(2*1.98**2));kernel/=kernel.sum()
        self.register_buffer('_lp_kernel',kernel,persistent=False)

    def architecture(self):
        return dict(symbol='pcrepro.model.CorrectedPANCrafter',num_bands=self.num_bands,
            max_pixel=self.max_pixel,hidden_size=self.hidden_size,depth=list(self.depth),
            num_heads=self.num_heads,mlp_ratio=4.,local_kernel=3,attentions=3,
            norm='ln',input_channels=1+self.num_bands,pan_kv='rawPANrepeat_joint',
            query='mode_raw_modality',gather='unfold_fixed_3x3_zero_padding',
            LP_usage='PAN_residual_only_samePAN_Gaussian1.98_k41_replicate_phase2_float64_to_float32',
            resize='bicubic_align_corners_False')

    def pan_residual_base(self,pan,pan_dn=None):
        if pan_dn is None:
            # Explicit convenience path for standalone tests/inference only.
            raw=(pan.detach().double()+1)*(self.max_pixel/2.)
        else:
            if pan_dn.shape!=pan.shape or not bool(torch.isfinite(pan_dn).all()):
                raise ValueError('Same-view original PAN DN shape/finite mismatch')
            normalized=pan_dn.float().mul(2/self.max_pixel).sub(1)
            if not torch.equal(normalized,pan):
                raise ValueError('PAN DN is not the same augmented normalized input')
            raw=pan_dn.detach().double()
        kernel=self._lp_kernel.to(device=pan.device,dtype=torch.float64)
        blurred=F.conv2d(F.pad(raw,(20,20,0,0),mode='replicate'),kernel[None,None,None,:])
        blurred=F.conv2d(F.pad(blurred,(0,0,20,20),mode='replicate'),kernel[None,None,:,None])
        low=blurred[...,2::4,2::4].float().mul(2/self.max_pixel).sub(1)
        return F.interpolate(low,size=pan.shape[-2:],mode='bicubic',align_corners=False).expand(-1,self.num_bands,-1,-1)

    def residual(self,pan,ms_up,mode):
        x=self.input(torch.cat((pan,ms_up),1))
        for block in self.encoder1:x=block(x,mode)
        skip1=x;x=self.down1(x)
        for block in self.encoder2:x=block(x,mode)
        resize=lambda value: F.interpolate(value,size=x.shape[-2:],mode='bicubic',align_corners=False)
        x=self.cond2_e(x,resize(ms_up),None,resize(pan),mode)
        skip2=x;x=self.down2(x)
        for block in self.middle:x=block(x,mode)
        x=self.cond_bot(x,resize(ms_up),None,resize(pan),mode)
        x=self.up2(x);x=torch.cat((x,skip2),1)
        for block in self.decoder2:x=block(x,mode)
        x=self.cond2_d(x,resize(ms_up),None,resize(pan),mode)
        x=self.up1(x);x=torch.cat((x,skip1),1)
        for block in self.decoder1:x=block(x,mode)
        return self.output(x)

    def forward(self,pan,ms,mode=None,*,pan_dn=None):
        if (pan.ndim!=4 or pan.shape[1]!=1 or ms.ndim!=4 or ms.shape[:2]!=(len(pan),self.num_bands)
                or tuple(pan.shape[-2:])!=tuple(v*4 for v in ms.shape[-2:]) or min(pan.shape[-2:])<8):
            raise ValueError('PAN/MS shape, C4/C8, exact ratio4 differ')
        mode=modes(mode,len(pan),pan.device)
        ms_up=F.interpolate(ms.float(),size=pan.shape[-2:],mode='bicubic',align_corners=False)
        residual=self.residual(pan.float(),ms_up,mode)
        base=ms_up.clone()
        selected=mode==0
        if bool(selected.any()):
            base[selected]=self.pan_residual_base(pan[selected],None if pan_dn is None else pan_dn[selected])
        return base+residual


def build_model(num_bands=8,seed=None,max_pixel=2047,**kwargs):
    if seed is None:return CorrectedPANCrafter(num_bands,max_pixel,**kwargs)
    if isinstance(seed,bool) or not 0<=int(seed)<2**32:raise ValueError('Unwrapped uint32 seed required')
    with torch.random.fork_rng(devices=[]):
        # Constructors are CPU-only; do not reset a caller's CUDA RNG stream.
        torch.random.default_generator.manual_seed(int(seed))
        return CorrectedPANCrafter(num_bands,max_pixel,**kwargs)


def parameter_counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                fixed_gather_parameters=0)
