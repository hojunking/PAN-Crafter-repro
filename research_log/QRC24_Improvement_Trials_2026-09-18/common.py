"""Isolated, network-disabled experiments using frozen source and checkpoint inputs."""
from pathlib import Path
import csv, datetime, hashlib, importlib, json, os, socket, sys, time
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
OLD=ROOT/'research_log/QRC24_RR_Crop64_Report_2026-09-18'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT/'source'))
def deny(*a,**kw):raise RuntimeError('Network disabled: no Sheet or external writes')
socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
import h5py, numpy as np, torch, yaml
from safetensors.torch import load_file,save_file
from kdv.teacher_assets import skeleton_from_cfg
from kdv.eval_modes import state_hash
from tools.metrics.eval_rr import ergas,sam
from tools.metrics.q2n import q2n
from tools.eval_dlpan import psnr_global,ssim_skimage,scc_dlpan
from tools.metrics.eval_fr import load_dlpan,d_lambda_k,d_s

SEEDS=(1234,3407,41001)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));t.replace(p)
def stamp():return datetime.datetime.now().astimezone().isoformat()
def asset(seed,tag):return next(a for a in read(OLD/'cohort.json')['assets'] if a['profile']=='G23' and a['seed']==seed and a['checkpoint']==tag)
def cfg_for(seed):return yaml.safe_load((OLD/asset(seed,'best_hqnr')['snapshot']/'config.yaml').read_text())
def model_for(seed,a_last=False,u_t=0.):
 cfg=cfg_for(seed);mod,cls=cfg['model'].rsplit('.',1);m,_=skeleton_from_cfg(cfg,getattr(importlib.import_module(mod),cls))
 sd=[]
 for tag in ('best_hqnr','last'):
  a=asset(seed,tag);p=OLD/a['snapshot']/'model.safetensors';assert sha(p)==a['checkpoint_sha256'];sd.append(load_file(str(p)))
 new={}
 for k,v in sd[0].items():
  if k.startswith('aligner.'):new[k]=sd[int(a_last)][k]
  elif k.startswith('backbone.'):
   new[k]=v if u_t==0 else sd[1][k] if u_t==1 else (1-u_t)*v+u_t*sd[1][k]
  else:assert torch.equal(v,sd[1][k]);new[k]=v
 m.load_state_dict(new,strict=True);return m.to('cuda:0').eval()
DATA={}
def dataset(split):
 if split in DATA:return DATA[split]
 cfg=cfg_for(1234);key={'rr':'test_reduced','fr':'test_full','val':'val','train':'train'}[split]
 p=Path(cfg[key+'_feeder_args']['dataroot']);d={}
 with h5py.File(p) as f:
  for k in ('pan','ms','lpan','lms','gt'):
   if k in f:d[k]=np.asarray(f[k],dtype=np.float32)
 lp=p.with_name(p.stem+'_pan.h5')
 if 'lpan' not in d:
  with h5py.File(lp) as f:d['lpan']=np.asarray(f['lpan'],dtype=np.float32)
 d['identity']={'path':str(p),'sha256':sha(p),'lpan_path':str(lp),'lpan_sha256':sha(lp)}
 DATA[split]=d;return d
def tensors(d,inds):return {k:torch.from_numpy(d[k][inds]).to('cuda:0')*2/2047.-1 for k in ('pan','ms','lpan','lms','gt') if k in d}
def forward(m,b,bias=(0.,0.),ref='bicubic'):
 if bias==(0.,0.) and ref=='bicubic':return m(b['pan'],b['ms'],b['lpan'])
 base=torch.nn.functional.interpolate(b['ms'],scale_factor=4,mode='bicubic')
 c=m.predict_delta(b['pan'],b['lms'] if ref=='lms' else base)
 c=c+torch.tensor(bias,device=c.device,dtype=c.dtype)[None]
 return m(b['pan'],b['ms'],b['lpan'],delta_override=c)
def dn(y):return ((y.detach().clamp(-1,1).float().cpu().numpy()+1)/2*2047.)
def rr_metrics(sr,gt):
 a=sr.astype(np.float64).transpose(1,2,0)[20:-21,20:-21];b=gt.astype(np.float64).transpose(1,2,0)[20:-21,20:-21]
 return dict(ergas=ergas(a,b),sam=sam(a,b),psnr=psnr_global(a,b,2047.),ssim=ssim_skimage(a,b,2047.),scc=scc_dlpan(a,b),q8=float(q2n(b,a,32,32)[0]))
def mean_metrics(rows):return {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('scene','delta')}
@torch.no_grad()
def validation(m,bias=(0.,0.),ref='bicubic'):
 d=dataset('val');vals=[]
 for start in range(0,len(d['pan']),16):
  b=tensors(d,slice(start,start+16));s=dn(forward(m,b,bias,ref)['y'])[:,:,16:48,16:48].astype(np.float64)
  g=d['gt'][start:start+16,:,16:48,16:48].astype(np.float64)
  e=25*np.sqrt(np.mean(np.mean((s-g)**2,axis=(2,3))/np.maximum(np.mean(g,axis=(2,3))**2,1e-12),axis=1))
  vals.extend(e.tolist())
 return {'mean_ergas':float(np.mean(vals)),'n':len(vals),'crop':'16:48 native64 center32','per_patch':vals}
WALD=None
@torch.no_grad()
def evaluate(m,seed,name,bias=(0.,0.),ref='bicubic',validation_result=None,description=None):
 global WALD
 od=OUT/'results'/str(seed)/name;od.mkdir(parents=True,exist_ok=True)
 before=state_hash(m)
 identity=dict(model_hash=before,bias=list(bias),ref=ref,source=read(OUT/'manifest.json')['source_hashes'],
               rr=dataset('rr')['identity'],fr=dataset('fr')['identity'],precision='FP32',device='cuda:0',script_sha=sha(__file__))
 prev=od/'result.json'
 if prev.exists() and read(prev).get('identity')==identity and read(prev).get('complete'):print('CACHED',seed,name,flush=True);return read(prev)
 if WALD is None:WALD=load_dlpan('/home/knuvi/Desktop/song/DLPan-Toolbox')
 results={};t=time.time()
 for split in ('rr','fr'):
  d=dataset(split);rows=[]
  for i in range(20):
   out=forward(m,tensors(d,slice(i,i+1)),bias,ref);s=dn(out['y'])[0]
   if split=='rr':metric=rr_metrics(s,d['gt'][i])
   else:
    a=s.transpose(1,2,0).astype(np.float64);l=d['lms'][i].transpose(1,2,0).astype(np.float64);p=d['pan'][i,0].astype(np.float64)
    dl=d_lambda_k(a,l,'wv3',4,32,WALD);ds=d_s(a,l,p,4,32,WALD);metric=dict(hqnr=(1-dl)*(1-ds),d_lambda=dl,d_s=ds)
   rows.append(dict(scene=i,delta=out['delta'][0].cpu().tolist(),**metric))
  results[split]={'mean':mean_metrics(rows),'per_scene':rows}
  print(f" {seed} {name} {split} {results[split]['mean']} ({time.time()-t:.1f}s)",flush=True)
 assert state_hash(m)==before
 assert not any('gspread' in k for k in sys.modules)
 out=dict(seed=seed,name=name,identity=identity,description=description,validation=validation_result,
          results=results,complete=True,state_unchanged=True,at=stamp(),seconds=time.time()-t,network_disabled=True,
          joint_pass=results['rr']['mean']['ergas']<2.04 and results['fr']['mean']['hqnr']>=.9585)
 write(prev,out);return out
def setup():
 torch.set_num_threads(4);assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.20,0)
 write(OUT/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),threads=4,mem_fraction=.20,at=stamp(),cudnn_benchmark=torch.backends.cudnn.benchmark,tf32=torch.backends.cudnn.allow_tf32))
