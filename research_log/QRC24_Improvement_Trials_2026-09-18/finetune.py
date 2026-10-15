"""Matched 1000-update small pilot; reset optimizer, separate files, original hard-only A routing."""
from common import *
import kdv.edge_gate as eg
from kdv.qrecon import QWeight
from kdv.teacher_assets import load_run_model
from kdv.losses_rec import GTAnchoredReconstructionKD
from pa.losses import output_edge_loss_per_sample
from model.pancrafter_paper import PANCrafterPaper

def batch(d,ids,rot):
 b=tensors(d,ids)
 for k,x in b.items():
  x=x.flip(-1).flip(-2)
  b[k]=torch.stack([torch.rot90(z,int(r),(-2,-1))for z,r in zip(x,rot)])
 meta=torch.tensor(np.stack((ids,rot,np.ones(len(ids)),np.ones(len(ids))),axis=1),device='cuda:0',dtype=torch.long)
 return b,meta

def main(seed=1234):
 setup();cfg=cfg_for(seed);d=dataset('train');k=cfg['kdv'];eg.ROOT=str(ROOT)
 td=k['teacher'];teacher,tm=load_run_model(str(ROOT/td['run']),td['tag'],PANCrafterPaper,td['expected_sha256'])
 teacher=teacher.to('cuda:0').eval().requires_grad_(False);teacher_hash=state_hash(teacher)
 qw=QWeight.load(k['qrecon'],teacher=teacher,train_h5=d['identity']['path'],feeder_args=cfg['train_feeder_args'],sha_fn=sha)
 crit=GTAnchoredReconstructionKD(k['rec']['tau'],alpha=1.,kd_weight=.1,eps=1e-6,mode='adaptive').to('cuda:0')
 rng=np.random.default_rng(20260918);ids=rng.integers(0,len(d['pan']),size=(1000,8));rots=rng.integers(0,4,size=(1000,8))
 orderp=OUT/f'finetune_order_{seed}.npz';np.savez(orderp,index=ids,rot=rots)
 write(OUT/f'finetune_inputs_{seed}.json',dict(seed=seed,teacher=tm,q_weight=qw.summary(),train=d['identity'],order_sha=sha(orderp),
  protocol=read(OUT/'manifest.json')['protocol']['finetune'],reference_check=read(OUT/'reference_check.json')['summary']))
 for arm in ('bicubic_joint','lms_Aref_joint','bicubic_Afreeze'):
  od=OUT/'finetune'/arm if seed==1234 else OUT/'finetune'/str(seed)/arm
  od.mkdir(parents=True,exist_ok=True);mp=od/'model.safetensors'
  if (od/'complete.json').exists():
   print('FINETUNE CACHED',seed,arm,flush=True)
   if not (OUT/'results'/str(seed)/('FT_'+arm)/'result.json').exists():
    m=model_for(seed);m.load_state_dict(load_file(str(mp)),strict=True);ref='lms' if arm=='lms_Aref_joint' else 'bicubic'
    evaluate(m,seed,'FT_'+arm,ref=ref,validation_result=validation(m,ref=ref),description='1000 updates matched pilot, reset optimizer, batch8');del m
   continue
  torch.manual_seed(20260918);torch.cuda.manual_seed_all(20260918)
  m=model_for(seed).train();initial=state_hash(m);ref='lms' if arm=='lms_Aref_joint' else 'bicubic';frz=arm=='bicubic_Afreeze'
  m.aligner.requires_grad_(not frz);ah0=state_hash(m.aligner);uh0=state_hash(m.backbone)
  up=list(m.backbone.parameters());ap=[] if frz else list(m.aligner.parameters())
  groups=[dict(params=up,lr=1e-5)]+([dict(params=ap,lr=3e-7)]if ap else [])
  opt=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=.01)
  log=[];t0=time.time();torch.cuda.reset_peak_memory_stats()
  for step in range(1000):
   b,meta=batch(d,ids[step],rots[step]);opt.zero_grad(set_to_none=True)
   with torch.no_grad():yt=teacher(b['pan'],b['ms'],b['lpan'])['y']
   out=forward(m,b,ref=ref);y=out['y'];gt=b['gt'];r=crit(y,yt,gt,return_maps=True)
   H=(r.maps['hard_weight']*(y-gt).abs().mean(1,keepdim=True)).mean((1,2,3))
   K=(r.maps['soft_weight']*(y-yt).abs().mean(1,keepdim=True)).mean((1,2,3))
   E=output_edge_loss_per_sample(y,gt);wa,we=qw.weights(meta,'cuda:0');LU=(H+K+.002*we*E).mean();LA=(wa*H).mean()
   assert bool(torch.isfinite(LU)) and bool(torch.isfinite(LA))
   gu=torch.autograd.grad(LU,up,retain_graph=bool(ap))
   for p,g in zip(up,gu):p.grad=g
   if ap:
    ga=torch.autograd.grad(LA,ap)
    for p,g in zip(ap,ga):p.grad=g
   if step==0:
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.isfinite(p.grad).all() for p in up+ap)
    write(od/'routing_check.json',dict(U_parameter_count=len(up),A_parameter_count=len(ap),U_objective='H+K+.002*wE*E',A_objective='wA*H only',teacher_grad_none=True,initial_hash=initial))
   opt.step()
   if (step+1)%100==0:
    row=dict(step=step+1,LU=float(LU),LA=float(LA),H=float(H.mean()),K=float(K.mean()),E=float(E.mean()),elapsed=time.time()-t0)
    log.append(row);write(od/'progress.json',dict(arm=arm,logs=log));print('TRAIN',arm,row,flush=True)
  assert state_hash(teacher)==teacher_hash and all(p.grad is None for p in teacher.parameters())
  assert state_hash(m.backbone)!=uh0
  assert (state_hash(m.aligner)==ah0)==frz
  m.eval();save_file({k:v.detach().cpu().contiguous()for k,v in m.state_dict().items()},str(mp))
  write(od/'complete.json',dict(arm=arm,updates=1000,seconds=time.time()-t0,initial_hash=initial,model_sha256=sha(mp),
   A_frozen=frz,A_unchanged=state_hash(m.aligner)==ah0,teacher_unchanged=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,at=stamp()))
  val=validation(m,ref=ref);evaluate(m,seed,'FT_'+arm,ref=ref,validation_result=val,description='1000 updates matched pilot, reset optimizer, batch8')
  del m,opt;torch.cuda.empty_cache()
 print('FINETUNE COMPLETE',flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=1234,choices=SEEDS);main(ap.parse_args().seed)
