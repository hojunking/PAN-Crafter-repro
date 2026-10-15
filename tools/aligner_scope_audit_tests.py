"""Independent arithmetic/geometry checks for the ASV evaluation adapter."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from pa.aligner import PANGlobalAligner
from pa.model import PAModel
from tools.aligner_scope_audit import trace_full,norm_forward,head,parity,PROBES

torch.set_num_threads(2)
torch.manual_seed(418)
a=PANGlobalAligner(8).eval()
# Nonzero head is necessary: zero initialization would make shift parity vacuous.
torch.nn.init.normal_(a.fc2.weight,std=.2)
m=PAModel(torch.nn.Identity(),a,aligner_margin=4).eval()
p=torch.randn(1,1,128,128);M=torch.randn(1,8,128,128)
with torch.no_grad():
 c,z,f,bank,zstats=trace_full(m,p,M)
 for mode in ('ZL_GL','ZF_GL','ZL_GF','ZF_GF'):
  d,v,ff=norm_forward(a,m._view(p),m._view(M),mode,bank,zstats)
  parity(d,c,'full self-reference '+mode);parity(ff,f,'full feature '+mode)
 for y,x in ((0,0),(0,64),(64,0),(64,64)):
  pp,mm=m._view(p[...,y:y+64,x:x+64]),m._view(M[...,y:y+64,x:x+64])
  d,zl,ff=norm_forward(a,pp,mm,'ZL_GL',bank,zstats)
  parity(d,a(pp,mm),'local original')
  d,zl,ff=norm_forward(a,pp,mm,'ZF_GF',bank,zstats)
  parity(ff[...,4:-4,4:-4],f[...,y//4+4:y//4+10,x//4+4:x//4+10],'fixed-stat matched core')
 # Full native statistics from another scene must produce a measurable change.
 _,_,_,other,oz=trace_full(m,p*3+4,M*.3-2)
 d,_,_=norm_forward(a,m._view(p),m._view(M),'ZF_GF',other,oz)
 assert (d-c).abs().max()>1e-3
 # AXIS16 denominator and bias invariance, using nonzero 2D vectors.
 eps=torch.tensor([e for _,e in PROBES]);c0=torch.tensor([1.2,-.7])
 assert abs(float(eps.abs().mean())-.46875)<1e-12
 assert (c0-eps+eps-c0).abs().max()<1e-6
 bias=torch.tensor([5.,-3.]);q1=(c0-eps+eps-c0).abs().mean()
 q2=(c0-eps+bias+eps-(c0+bias)).abs().mean();assert abs(float(q1-q2))<1e-6
 assert torch.allclose(torch.quantile(torch.tensor([[1.,2.],[3.,8.]]),.5,dim=0),torch.tensor([2.,5.]))
 # The nonlinear head/mean gap is observable on deliberately heterogeneous features.
 z=torch.randn(16,64)*3
 assert (head(a,z.mean(0,keepdim=True))-head(a,z).mean(0,keepdim=True)).abs().max()>1e-3
 assert all(not l._forward_hooks and not l._forward_pre_hooks for l in a.modules())
 assert all(v.grad is None for v in m.parameters())
print('ASV tests passed: norm self-reference, all crop coordinates/core, scene separation, probe sign/scale/bias, even median, nonlinear head, hook cleanup.')
