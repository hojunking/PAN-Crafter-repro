"""Read-only checkpoint/data audit. Writes only next to this script.

CPU FP32; no training, no candidate/metric selection, no production edits.
Registration is a diagnostic image proxy, not physical ground truth.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from pa.aligner import PANGlobalAligner
from pa.warp import warp_pan

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)

CASES = {
    'WV3': ('FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1', 38380,
            'work_dir/_fh12/s1/dataset_manifest.json', 8, 2047),
    'QB': ('QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1', 47470,
           'work_dir/_qg40/s1/QB/dataset_manifest.json', 4, 2047),
    'GF2': ('G20_GF2_S_S1_R0_TS91001_SS93002_BASE_PLH_W104_D122_FRESH50_v1', 45450,
            'work_dir/_g20/s1/GF2/dataset_manifest.json', 4, 1023),
}
EPS = np.array([(dy*r, dx*r) for r in [.25, .5, 1., 2.]
                for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]], np.float32)


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def paths(sensor):
    run, step, manifest, bands, dn = CASES[sensor]
    ck=ROOT/'work_dir'/run/'candidates'/str(step)/'model.safetensors'
    return ck, json.loads((ROOT/manifest).read_text()), bands, dn


def split_data(manifest, split, dn):
    s=manifest['splits'][split]
    with h5py.File(s['dataroot'],'r') as f:
        idx=np.linspace(0,len(f['pan'])-1,20,dtype=int) if split=='train' else np.arange(len(f['pan']))
        data={k:np.asarray(f[k][idx],np.float32) for k in f if isinstance(f[k],h5py.Dataset) and k in ['pan','ms','gt','lms']}
    data['indices']=idx
    return data


def tensors(data,dn):
    p=torch.from_numpy(data['pan'])/(dn/2)-1
    m=torch.from_numpy(data['ms'])/(dn/2)-1
    return p,m,F.interpolate(m,scale_factor=4,mode='bicubic',align_corners=False)


@torch.no_grad()
def predict(a,p,base):
    return a(p[...,4:-4,4:-4],base[...,4:-4,4:-4])


def load_aligner(sensor):
    ck,manifest,bands,dn=paths(sensor)
    state=load_file(str(ck),device='cpu')
    a=PANGlobalAligner(ms_bands=bands).eval()
    a.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('aligner.')},strict=True)
    return a,manifest,dn


def features(img, sigma, kind='gradient'):
    z=cv2.GaussianBlur(np.asarray(img,np.float32),(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
    if kind=='intensity': return z
    return np.stack([cv2.Scharr(z,cv2.CV_32F,1,0),cv2.Scharr(z,cv2.CV_32F,0,1)],-1)


def ncc(x,y):
    x=np.asarray(x,np.float64).ravel(); y=np.asarray(y,np.float64).ravel()
    x=x-x.mean(); y=y-y.mean()
    return float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-20))


def register_one(pan,reference,sigma_p,sigma_r,kind='gradient',bound=8):
    """Return sampling shift d: W(PAN,d) best matches reference, dy,dx.

    Fixed reference support margin16; exhaustive integer [-8,+8] then local
    2-D quadratic peak. Independent of learned offsets. No phase resampling.
    """
    a=features(pan,sigma_p,kind); b=features(reference,sigma_r,kind)
    margin=16
    template=np.ascontiguousarray(b[margin:-margin,margin:-margin])
    search=np.ascontiguousarray(a[margin-bound:-(margin-bound),margin-bound:-(margin-bound)])
    method=cv2.TM_CCOEFF_NORMED if kind=='intensity' else cv2.TM_CCORR_NORMED
    score=cv2.matchTemplate(search,template,method)
    iy,ix=np.unravel_index(score.argmax(),score.shape)
    shift=np.array([iy-bound,ix-bound],np.float64)
    hit=iy in (0,2*bound) or ix in (0,2*bound)
    if not hit:
        yy,xx=np.mgrid[-1:2,-1:2]
        design=np.stack([np.ones(9),yy.ravel(),xx.ravel(),yy.ravel()**2,
                         yy.ravel()*xx.ravel(),xx.ravel()**2],1)
        c=np.linalg.lstsq(design,score[iy-1:iy+2,ix-1:ix+2].ravel(),rcond=None)[0]
        h=np.array([[2*c[3],c[4]],[c[4],2*c[5]]])
        if np.linalg.eigvalsh(h).max() < -1e-8:
            off=-np.linalg.solve(h,c[1:3])
            if np.abs(off).max()<=1: shift+=off
    return dict(shift=shift.tolist(),peak=float(score.max()),boundary=bool(hit))


def registration(p,ref,kind='gradient',gt=False):
    # Comparable bandwidth; GT is already at RR PAN resolution.
    sp,sr=(.8,.8) if gt else (1.98,.7)
    bands=[register_one(p,b,sp,sr,kind) for b in ref]
    # Fixed rule: three bands with strongest positive structural correlation.
    ids=np.argsort([b['peak'] if not b['boundary'] else -2 for b in bands])[-min(3,len(bands)):]
    d=np.median([bands[i]['shift'] for i in ids],axis=0)
    spread=float(np.median(np.linalg.norm(np.asarray([bands[i]['shift'] for i in ids])-d,axis=1)))
    return dict(shift=d.tolist(),spread=spread,bands=bands,selected_bands=ids.tolist(),
                boundary=any(bands[i]['boundary'] for i in ids),kind=kind)


def warp_np(p,d):
    return warp_pan(torch.from_numpy(p[None,None].astype(np.float32)),torch.from_numpy(np.asarray(d,np.float32)[None]))[0,0].numpy()


def score_at(p,refs,reg,d,gt=False):
    sp,sr=(.8,.8) if gt else (1.98,.7)
    a=features(warp_np(p,d),sp)[16:-16,16:-16]
    return float(np.mean([ncc(a,features(refs[b],sr)[16:-16,16:-16]) for b in reg['selected_bands']]))


def unit_tests():
    rows=[]
    # Integer ramp probes avoid bicubic interpolation's fractional ramp bias.
    for n in [64,256,512]:
        yy,xx=torch.meshgrid(torch.arange(n,dtype=torch.float32),torch.arange(n,dtype=torch.float32),indexing='ij')
        p=(yy*2+xx)[None,None]
        d=torch.tensor([[2.,-3.]])
        error=(warp_pan(p,d)[...,8:-8,8:-8]-p[...,8:-8,8:-8]-1).abs().max().item()
        assert error<.001,(n,error)
        rows.append(dict(size=n,dy=2,dx=-3,ramp_error=error))
    rng=np.random.default_rng(20261006)
    ref=cv2.GaussianBlur(rng.normal(size=(128,128)).astype(np.float32),(0,0),1.2)
    reg=[]
    for d in [[.5,-.25],[1.,-2.],[4.,-4.],[-2.,1.]]:
        p=warp_np(ref,d)
        est=register_one(p,ref,.8,.8)
        error=float(np.linalg.norm(np.asarray(est['shift'])+d))
        assert error<.15,(d,est,error)
        reg.append(dict(injected=d,estimated=est['shift'],error=error))
    dump('unit_tests.json',dict(warp=rows,registration=reg))


@torch.no_grad()
def reduce4(x):
    # Controlled common-FOV surrogate, NOT official Wald dataset generation.
    a=x.numpy(); out=[]
    for scene in a:
        out.append(np.stack([cv2.GaussianBlur(b,(41,41),1.98,borderType=cv2.BORDER_REPLICATE)[2::4,2::4] for b in scene]))
    return torch.from_numpy(np.stack(out))


@torch.no_grad()
def diagnostics(sensor):
    a,manifest,dn=load_aligner(sensor)
    ck,_,bands,_=paths(sensor)
    result=dict(sensor=sensor,run=CASES[sensor][0],step=CASES[sensor][1],checkpoint=str(ck.relative_to(ROOT)),
                checkpoint_sha256=sha(ck),torch=torch.__version__,device='cpu',splits={})
    for split in ['train','rr','fr']:
        data=split_data(manifest,split,dn); p,m,base=tensors(data,dn)
        rows=[]
        for i in range(len(p)):
            c=predict(a,p[i:i+1],base[i:i+1])[0].numpy()
            response=[]
            for eps in EPS:
                pe=warp_pan(p[i:i+1],torch.from_numpy(eps[None]))
                ce=predict(a,pe,base[i:i+1])[0].numpy()
                response.append(ce.tolist())
            row=dict(index=int(data['indices'][i]),delta=c.tolist(),epsilon_prediction=response)
            if split!='train':
                pp=data['pan'][i,0]; bb=(base[i].numpy()+1)*(dn/2)
                for key,ref,isgt in [('input',bb,False)]+([('gt',data['gt'][i],True)] if split=='rr' else []):
                    reg=registration(pp,ref,gt=isgt)
                    reg['intensity_check']=registration(pp,ref,kind='intensity',gt=isgt)
                    reg['ncc_alpha']={str(alpha):score_at(pp,ref,reg,c*alpha,gt=isgt) for alpha in [0,1,4]}
                    # Re-estimate after actual bicubic warp, avoiding only vector subtraction.
                    reg['after_native']=registration(warp_np(pp,c),ref,gt=isgt)['shift']
                    reg['after_x4']=registration(warp_np(pp,c*4),ref,gt=isgt)['shift']
                    row[key]=reg
                if split=='rr':
                    # Input-upsample phase vs true RR target, same spectral bands.
                    row['base_gt_phase']=[register_one(bb[b],data['gt'][i,b],1.98,1.98)['shift'] for b in range(bands)]
            if split=='fr':
                pr,mr=reduce4(p[i:i+1]),reduce4(m[i:i+1])
                br=F.interpolate(mr,scale_factor=4,mode='bicubic',align_corners=False)
                cr=predict(a,pr,br)[0].numpy()
                row['paired_reduced_delta']=cr.tolist()
                paired=[]
                for eps in [[4.,0.],[-4.,0.],[0.,4.],[0.,-4.]]:
                    pe=warp_pan(p[i:i+1],torch.tensor([eps]))
                    cf=predict(a,pe,base[i:i+1])[0].numpy()
                    cre=predict(a,reduce4(pe),br)[0].numpy()
                    paired.append(dict(epsilon_fr=eps,fr_response=(cf-c).tolist(),reduced_response=(cre-cr).tolist()))
                row['paired_responses']=paired
            rows.append(row)
            if (i+1)%5==0: print(sensor,split,i+1,flush=True)
        s=manifest['splits'][split]
        result['splits'][split]=dict(path=s['dataroot'],count=len(rows),shapes={k:list(v.shape) for k,v in data.items()},
                                    rows=rows,sha256=sha(s['dataroot']) if split!='train' else s['sha256'])
        dump(sensor+'_diagnostics.json',result)
    return result


@torch.no_grad()
def quality(sensor,limit=20):
    from qg40.model import build_model
    from tools.eval_dlpan import scc_dlpan
    from tools.metrics.eval_rr import ergas
    ck,manifest,bands,dn=paths(sensor)
    state=load_file(str(ck),device='cpu')
    model,_=build_model('PLH',104,[1,2,2],0,role='S',bands=bands,
                        teacher_aligner_state={k[8:]:v for k,v in state.items() if k.startswith('aligner.')})
    model.load_state_dict(state,strict=True); model.eval()
    result=dict(sensor=sensor,checkpoint_sha256=sha(ck),device='cpu',alphas={})
    previous=OUT/(sensor+'_quality.json')
    if previous.exists():
        old=json.loads(previous.read_text())
        assert old['checkpoint_sha256']==result['checkpoint_sha256']
        result=old
    for split in ['rr','fr']:
        data=split_data(manifest,split,dn); p,m,base=tensors(data,dn)
        with h5py.File(manifest['splits'][split]['lpan_path'],'r') as f:
            print(sensor,split,'LP keys',list(f.keys()),flush=True)
            lp=torch.from_numpy(np.asarray(f['lpan'],np.float32))/(dn/2)-1
        if split=='fr':
            if sensor=='WV3':
                from fh12.evaluation import FRMetrics
                evaluator=FRMetrics(SimpleNamespace(raw_h5_path=manifest['splits'][split]['dataroot']))
            else:
                from qg40.evaluation import FRMetrics
                from qg40.plan import sensor_spec
                from dataclasses import replace
                spec=replace(sensor_spec(sensor),band_order=tuple(manifest['band_order']))
                evaluator=FRMetrics(SimpleNamespace(raw_h5_path=manifest['splits'][split]['dataroot'],spec=spec))
        for alpha in [0,1,4]:
            if f'{split}_{alpha}' in result['alphas']: continue
            rows=[]; predictions=[]; start=time.time()
            for i in range(min(limit,len(p))):
                c=model.predict_delta(p[i:i+1],base[i:i+1])
                out=model(p[i:i+1],m[i:i+1],lp[i:i+1],delta_override=alpha*c)
                sr=((out['y'].clamp(-1,1)+1)*(dn/2))[0].numpy()
                if split=='rr':
                    x=sr[:,20:-21,20:-21].transpose(1,2,0).astype(np.float64)
                    y=data['gt'][i,:,20:-21,20:-21].transpose(1,2,0).astype(np.float64)
                    rows.append(dict(index=i,ergas=ergas(x,y),scc=scc_dlpan(x,y)))
                else: predictions.append(sr)
                print(sensor,split,'alpha',alpha,'scene',i+1,'seconds',round(time.time()-start,1),flush=True)
            if split=='fr':
                assert limit==20,'Full512 HQNR needs all20'
                stats=evaluator(np.stack(predictions));rows=stats['per_scene']
                # QG40 also returns signed diagnostics; persist compact measurements.
                avg={k:stats[k] for k in ['hqnr','d_lambda','d_s']}
            else: avg={k:float(np.mean([r[k] for r in rows])) for k in ['ergas','scc']}
            result['alphas'][f'{split}_{alpha}']=dict(mean=avg,rows=rows,seconds=time.time()-start)
            dump(sensor+'_quality.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['unit','diagnostics','quality'],default='diagnostics')
    parser.add_argument('--sensor',choices=list(CASES),nargs='+',default=list(CASES))
    args=parser.parse_args()
    if args.mode=='unit': unit_tests()
    else:
        for sensor in args.sensor:
            (diagnostics if args.mode=='diagnostics' else quality)(sensor)
