"""Independent formula/gradient checks on real T0, W104D121, H5 inputs and cue."""
import ast,copy,io,json,os,pathlib,sys,tempfile,time,types,gc
from audit_common import *
os.environ.update(ENV);os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch,numpy as np,yaml,h5py
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from tools import gen_pakd50_configs as G,qedge9_cue as Q
from tools.smoke_cases import build,randomize_zero_params
from kdv import edge_gate as EG
from kdv.qrecon import QWeight
from kdv.registry import resolve
from kdv.resume import EpochState
from kdv.teacher_assets import state_hash
from kdv.losses_rec import GTAnchoredReconstructionKD
from pa.model import PAModel
from train_kdv import KDVTrainer
R=[]
def record(name,ok,**kw):
 R.append(dict(name=name,passed=bool(ok),**kw));(OUT/'math_results.json').write_text(json.dumps(R,indent=2));print(name,ok,kw,flush=True)
torch.set_num_threads(2);torch.manual_seed(7719)
cfg=yaml.safe_load((ROOT/'config/PAKD50_QRC24_S1_G22_W104_D121_WV3_T0_S1234_FRESH50_v1.yaml').read_text());cal=G.calibration();T0,tm=Q.load_T0();th=state_hash(T0)
record('T0_fixed_identity',tm['file_sha256']=='16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32' and state_hash(T0.aligner)=='493aca181f93ae68' and T0.aligner_margin==4)
contract,h5,h5p=Q.contract_from(cfg);base=np.array([13,79,111,987])
with h5py.File(h5) as f,h5py.File(h5p) as fp:rows=Q.read_rows(f,fp,base)
v={k:torch.cat([EG.apply_view(EG.to_feeder_tensor(a[i:i+1],2047.),True,True,i) for i in range(4)]) for k,a in rows.items()};gt,ms,lp,pan=v['gt'],v['ms'],v['lpan'],v['pan'];meta=torch.tensor([[int(i),r,1,1] for r,i in enumerate(base)]);batch=(gt,v.get('lms',ms),ms,lp,pan,meta)
man,z=EG.read_asset(G.QEDGE9_CUE_ASSET);qw=QWeight.load(cfg['kdv']['qrecon'],teacher=T0)
raw=2*.3276133416220546/(.3276133416220546+z['q']);observed=qw.tables['q'][z['index'],z['rot']].numpy();strata=z['e_decile']*4+z['rot'];obs_sh=qw.tables['shuffle'][z['index'],z['rot']].numpy()
record('full_raw_q_and_strata',np.allclose(raw,observed,rtol=1e-7) and all(np.array_equal(np.sort(observed[strata==s]),np.sort(obs_sh[strata==s])) for s in np.unique(strata)),views=len(raw),strata=len(np.unique(strata)),w_min=float(raw.min()),w_max=float(raw.max()),w_mean=float(raw.mean()),asset_id=man['asset_id'],npz_sha256=man['npz_sha256'])
q,c,e=Q.compute_views(T0,4,rows,contract,2047.);ix=(base[:,None]*4+np.arange(4)).reshape(-1)
record('cue_recomputed_16_views',float(abs(q-z['q'][ix]).max())<1e-4,max_q_error=float(abs(q-z['q'][ix]).max()))
ns=dict(G=G,calQ=cal,KDVTrainer=KDVTrainer,resolve=resolve,types=types,T0=T0,GTAnchoredReconstructionKD=GTAnchoredReconstructionKD,torch=torch)
n=next(n for n in ast.parse((ROOT/'tools/pakd50_unit_tests.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_stubQR');exec(compile(ast.Module(body=[n],type_ignores=[]),'fixture_constructor','exec'),ns)
# Independent Scharr implementation, no call to pa.losses or rec_crit.
def edge(y,g):
 b,c,h,w=y.shape;k=torch.tensor([[-3.,0,3],[-10,0,10],[-3,0,3]],dtype=torch.float32)/32
 out=[]
 for f in (k,k.T):
  d=torch.nn.functional.conv2d(torch.nn.functional.pad((y-g).reshape(-1,1,h,w),(1,1,1,1),mode='reflect'),f[None,None]).reshape(b,c,h,w)
  out.append(d[:,:,1:-1,1:-1].abs().mean((1,2,3)))
 return .5*(out[0]+out[1])
def terms(info,P,weight):
 y,yt=info['y'].float(),info['y_t'].float();es=(y-gt).abs().mean(1);et=(yt.detach()-gt).abs().mean(1);d=(et/(et+.012463942170143127)).detach();a=((es-et).clamp_min(0)/(es+1e-6)).clamp(0,1).detach()
 H=((1+P['alpha']*d)*es).mean((1,2));K=(P['beta']*(1-d)*a*(y-yt.detach()).abs().mean(1)).mean((1,2));E=edge(y,gt);wa,we=weight.weights(meta,'cpu')
 return (H+K+P['lam']*we*E).mean(),(wa*H).mean()
def make(prof,S):
 P=G.qrc24_profile(prof);weight=copy.copy(qw);weight.a_mode=P['a'];weight.e_mode=P['e'];tr=ns['_stubQR']('QRC24_S1_'+prof,copy.deepcopy(S),weight,server='s1',seed=1234);return tr,P,weight
U=build(cfg);randomize_zero_params(U);S=PAModel(U,copy.deepcopy(T0.aligner).requires_grad_(True),aligner_margin=4,sampler=True)
record('W104D121_size',sum(p.numel() for p in U.parameters())==1903624)
for prof in ('G11','G12','G13','G21','G22','G23','G31','G32','G33','A_UNIF','A_SHUF','A_FREEZE','E_UNIF','E_SHUF','ALL_UNIF','H11','H12','H13','H21','H22','H23','H31','H32','H33','H_ALPHA0','H_BETA0','L100','L070','L050'):
 tr,P,w=make(prof,S);u0=tr.gen.get_state().clone();total,info=tr._step(gt,ms,lp,pan,1,meta=meta);lu,la=terms(info,P,w);up=list(tr.M.backbone.parameters());ap=[p for p in tr.M.aligner.parameters() if p.requires_grad];gu=torch.autograd.grad(lu,up,retain_graph=True);ga=torch.autograd.grad(la,ap,retain_graph=True) if ap else []
 tr._qrecon_backward(info,tr.M,tr.aligner_active(1));
 def err(params,grads):
  a=torch.cat([p.grad.flatten() for p in params]);b=torch.cat([g.flatten() for g in grads]);return float((a-b).norm()/(b.norm()+1e-30)),float((a-b).abs().max())
 ue,ua=err(up,gu);ae,aa=err(ap,ga) if ap else (0.,0.)
 record(prof+'_independent_formula_gradient',abs(float(total-lu))<1e-7 and abs(float(info['_L_A_t']-la))<1e-7 and ue<1e-5 and ae<1e-5 and torch.equal(u0,tr.gen.get_state()) and float(info['loss_off'])==0 and not info['y_t'].requires_grad and all(p.grad is None for p in T0.parameters()),U_relative_error=ue,A_relative_error=ae,U_abs_error=ua,A_abs_error=aa)
 del tr,total,info,lu,la,gu,ga;gc.collect()
# Real training loop, zero-output initial U, independent AdamW reference, 8 updates.
S0=PAModel(build(cfg),copy.deepcopy(T0.aligner).requires_grad_(True),aligner_margin=4,sampler=True)
for prof in ('G22','A_FREEZE','E_SHUF'):
 tr,P,w=make(prof,S0);ref,_,rw=make(prof,S0);ah=state_hash(tr.M.aligner)
 def optimizer(m):
  groups=[dict(params=m.backbone.parameters(),name='backbone')]
  if not P['frozen']:groups.append(dict(params=m.aligner.parameters(),name='aligner',lr=P['alr']))
  return torch.optim.AdamW(groups,lr=P['ulr'],weight_decay=.01)
 ro=optimizer(ref.M);rs=get_scheduler('cosine',optimizer=ro,num_warmup_steps=100,num_training_steps=50000)
 with tempfile.TemporaryDirectory(prefix='qrc24-real-loop-') as td:
  tr.accelerator=Accelerator(cpu=True,mixed_precision='no');opt=optimizer(tr.M);sch=get_scheduler('cosine',optimizer=opt,num_warmup_steps=100,num_training_steps=50000);tr.model,tr.optimizer,tr.lr_scheduler=tr.accelerator.prepare(tr.model,opt,sch)
  tr.case='QRC24_S1_'+prof;tr.args=types.SimpleNamespace(num_iter=8,batch_size=4,log_iter=1000,save_iter=10000,work_dir=td);tr.weight_dtype=torch.float32;tr.train_data_loader=[batch]*8;tr.exact_resume=True;tr._epoch_state=EpochState();tr.accelerator.register_for_checkpointing(tr._epoch_state);tr._resume_pending=False;tr._meta_log=False;tr._fixed_diag_on=False;tr._t_run0=time.time();tr.is_diag_step=lambda _:False;tr._jsonl=lambda *a:None;tr._runs_csv=lambda _:None;tr._finish_ledger=lambda _:None;tr._write_cost=lambda _:None
  log=io.StringIO();step=tr.train(log,0);(OUT/(prof+'_train_loop.log')).write_text(log.getvalue())
  for i in range(8):
   _,info=ref._step(gt,ms,lp,pan,i,meta=meta);lu,la=terms(info,P,rw);up=list(ref.M.backbone.parameters());ap=[p for p in ref.M.aligner.parameters() if p.requires_grad];gu=torch.autograd.grad(lu,up,retain_graph=bool(ap));ga=torch.autograd.grad(la,ap) if ap else []
   for p,g in zip(up,gu):p.grad=g
   for p,g in zip(ap,ga):p.grad=g
   ro.step();rs.step();ro.zero_grad()
  du=max(float((a-b).abs().max()) for a,b in zip(tr.M.backbone.parameters(),ref.M.backbone.parameters()));da=max(float((a-b).abs().max()) for a,b in zip(tr.M.aligner.parameters(),ref.M.aligner.parameters()))
  record(prof+'_actual_train_loop',step==8 and du<1e-6 and da<1e-6 and (state_hash(tr.M.aligner)==ah if P['frozen'] else state_hash(tr.M.aligner)!=ah),U_parameter_error=du,A_parameter_error=da,last_saved=(pathlib.Path(td)/'last/model.safetensors').exists(),A_frozen=P['frozen'])
record('Teacher_unchanged',state_hash(T0)==th and all(p.grad is None for p in T0.parameters()))
print('SUMMARY',len(R),sum(r['passed'] for r in R))
