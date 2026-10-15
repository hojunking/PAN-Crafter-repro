"""Six actual KDVTrainer.train updates vs independently routed AdamW updates.
CPU, true zero-output initialization, real T0 and four real training patches.
Only diagnostics/ledger callbacks are replaced; all training and last-state saves run.
"""
import ast,copy,io,json,os,pathlib,sys,tempfile,time,types
from audit_common import *
os.environ.update(ENV);os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch,numpy as np,yaml,h5py
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from tools import gen_pakd50_configs as G,qedge9_cue as Q
from tools.smoke_cases import build
from kdv.edge_gate import EdgeGate,apply_view,to_feeder_tensor
from kdv.registry import resolve
from kdv.resume import EpochState
from kdv.teacher_assets import state_hash
from kdv.losses_rec import GTAnchoredReconstructionKD
from pa.model import PAModel
from pa.losses import output_edge_loss_per_sample
from train_kdv import KDVTrainer
torch.set_num_threads(2);torch.manual_seed(7777)
cfg=yaml.safe_load((ROOT/'config/PAKD50_QER50_W104_D121_WV3_T0_S1234_FRESH50_v1.yaml').read_text());cal=G.calibration();T0,_=Q.load_T0();th=state_hash(T0)
contract,h5,h5p=Q.contract_from(cfg);base=np.array([13,79,111,987])
with h5py.File(h5) as f,h5py.File(h5p) as fp:rows=Q.read_rows(f,fp,base)
v={k:apply_view(to_feeder_tensor(a,2047.),True,True,0) for k,a in rows.items()};gt,ms,lp,pan=v['gt'],v['ms'],v['lpan'],v['pan'];meta=torch.tensor([[int(i),0,1,1] for i in base]);batch=(gt,v.get('lms',ms),ms,lp,pan,meta)
S=PAModel(build(cfg),copy.deepcopy(T0.aligner).requires_grad_(True),aligner_margin=4,sampler=True)
ns=dict(G=G,calQ=cal,KDVTrainer=KDVTrainer,resolve=resolve,types=types,T0=T0,GTAnchoredReconstructionKD=GTAnchoredReconstructionKD,torch=torch)
n=next(n for n in ast.parse((ROOT/'tools/pakd50_unit_tests.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_stubX');exec(compile(ast.Module(body=[n],type_ignores=[]),'fixture_constructor','exec'),ns)
gate=EdgeGate.load(dict(mode='low_q',asset=G.QEDGE9_CUE_ASSET),teacher=T0);tr=ns['_stubX']('QER50',copy.deepcopy(S),gate);ref=ns['_stubX']('QER50',copy.deepcopy(S),gate)
def optimizer(m):return torch.optim.AdamW([{'params':m.backbone.parameters(),'name':'backbone'},{'params':m.aligner.parameters(),'name':'aligner','lr':1e-5}],lr=1e-4,weight_decay=.01)
refopt=optimizer(ref.M);refsched=get_scheduler('cosine',optimizer=refopt,num_warmup_steps=100,num_training_steps=50000)
with tempfile.TemporaryDirectory(prefix='qegx-train-loop-') as td:
    tr.accelerator=Accelerator(cpu=True,mixed_precision='no');opt=optimizer(tr.M);sched=get_scheduler('cosine',optimizer=opt,num_warmup_steps=100,num_training_steps=50000)
    tr.model,tr.optimizer,tr.lr_scheduler=tr.accelerator.prepare(tr.model,opt,sched)
    tr.case='QER50';tr.args=types.SimpleNamespace(num_iter=6,batch_size=4,log_iter=1000,save_iter=10000,work_dir=td)
    tr.weight_dtype=torch.float32;tr.train_data_loader=[batch]*6;tr.exact_resume=True;tr._epoch_state=EpochState();tr.accelerator.register_for_checkpointing(tr._epoch_state);tr._resume_pending=False;tr._meta_log=False;tr._fixed_diag_on=False;tr._t_run0=time.time();tr.is_diag_step=lambda _:False
    events=[];tr._jsonl=lambda name,data:events.append((name,data));tr._runs_csv=lambda _:None;tr._finish_ledger=lambda _:None;tr._write_cost=lambda _:None
    log=io.StringIO();step=tr.train(log,0);(OUT/'train_loop.log').write_text(log.getvalue())
    refstats=[]
    for i in range(6):
        total,info=ref._step(gt,ms,lp,pan,i,meta=meta);y,yt=info['y'],info['y_t'];es=(y-gt).abs().mean(1);et=(yt-gt).abs().mean(1).detach();d=(et/(et+cal['tau_R'])).detach();a=((es-et).clamp_min(0)/(es+1e-6)).detach();rec=((1+d)*es).mean()+(.1*(1-d)*a*(y-yt.detach()).abs().mean(1)).mean();ei=output_edge_loss_per_sample(y.float(),gt.float());g=gate.gate_for(meta,'cpu')
        lu=rec+cal['lambda_E']*ei.mean();la=rec+cal['lambda_E']*(g*ei).mean()+1e-4*info['loss_off']
        up=list(ref.M.backbone.parameters());ap=list(ref.M.aligner.parameters());gu=torch.autograd.grad(lu,up,retain_graph=True);ga=torch.autograd.grad(la,ap)
        for p,x in zip(up,gu):p.grad=x
        for p,x in zip(ap,ga):p.grad=x
        refstats.append(dict(step=i,A_grad_max=max(float(x.abs().max()) for x in ga)));refopt.step();refsched.step();refopt.zero_grad()
    diffA=max(float((a-b).abs().max()) for a,b in zip(tr.M.aligner.parameters(),ref.M.aligner.parameters()));diffU=max(float((a-b).abs().max()) for a,b in zip(tr.M.backbone.parameters(),ref.M.backbone.parameters()))
    r=dict(passed=step==6 and diffA<1e-6 and diffU<1e-6 and state_hash(T0)==th,step=step,A_parameter_max_diff=diffA,U_parameter_max_diff=diffU,adjusted_params=tr._ema.get('edge_route_params_adjusted'),reference_gradients=refstats,last_state_saved=(pathlib.Path(td)/'last/model.safetensors').exists(),teacher_immutable=state_hash(T0)==th,note='Real train loop/Accelerator/AdamW/cosine/checkpoint on CPU; six repeated real batches. Diagnostic and campaign ledger callbacks stubbed. Not a full main.py evaluation or 50K run.')
    (OUT/'train_loop_results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
