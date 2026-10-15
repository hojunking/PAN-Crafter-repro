#!/usr/bin/env python
"""PAN_ASV_READONLY_v1: pinned-checkpoint size, normalization and GAP diagnostics.

Writes only under --output-root. Never imports or invokes campaign/Sheet writers.
--inventory-only pins inputs; --pilot runs scene 0 of both splits; --case all
runs all 20 scenes. C01-C04 share scene computations but have separate artifacts.
"""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from kdv.eval_modes import CallCounter, forward_mode
from pa.aligner import znorm
from pa.warp import warp_pan, warp_support_mask
from tools.eval_fr_paperset import build
from tools.eval_dlpan import scc_dlpan, psnr_global, ssim_skimage
from tools.metrics.eval_rr import ergas, sam, q2n
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s

PROTOCOL = 'PAN_ASV_READONLY_v1_20260918'
ASSETS = {
    'S_G23_1234': 'work_dir/PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S1234_FRESH50_v1',
    'S_G23_3407': 'work_dir/PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S3407_FRESH50_v1',
    'S_G22_1234': 'work_dir/PAKD50_QRC24_S1_G22_W104_D121_WV3_T0_S1234_FRESH50_v1',
    'T0': 'assets/pakd50/T0_run',
}
MODES = ('A_ON', 'A_ZERO_WARP', 'A_BYPASS_RAW', 'A_CROP64_MED',
         'A_CROP64_FEAT_MEAN', 'A_CROP64_SHIFT_MEAN', 'A_NORM_ZF_GL_MED64',
         'A_NORM_ZL_GF_MED64', 'A_NORM_ZF_GF_MED64')
NORMS = ('ZL_GL', 'ZF_GL', 'ZL_GF', 'ZF_GF')
GN_NAMES = ('pan_stem.2', 'ms_stem.2', 'joint1.2', 'joint2.2', 'res.a.2', 'res.b.2')
PROBES = [(r, (r*y, r*x)) for r in (.25, .5, 1., 2.) for y,x in ((1,0),(-1,0),(0,1),(0,-1))]
ATOL, RTOL = 1e-5, 1e-4


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def plain(v):
    if isinstance(v, torch.Tensor): return v.detach().cpu().numpy().tolist()
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=plain, allow_nan=False))
    tmp.replace(path)


def write_csv(path, rows):
    if not rows: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def parity(a, b, name):
    err = (a-b).abs()
    result = dict(name=name, max_abs=float(err.max()), mean_abs=float(err.mean()),
                  passed=bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL)))
    if not result['passed']: raise AssertionError(result)
    return result


def pin_inventory(out, device):
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        assert m['device'] == device
        for a in m['assets'].values():
            assert sha(a['checkpoint_file']) == a['checkpoint_sha256']
            assert sha(a['mapped_config']) == a['mapped_config_sha256']
        for p,h in m['data_sha256'].items(): assert sha(p) == h, p
        for p,h in m['source_code_sha256'].items(): assert sha(ROOT/p) == h, p
        assert m['diagnostic_sha256'] == sha(__file__), 'Use a new output root after code changes'
        return m
    data, assets, protected, existing = {}, {}, {}, {}
    for aid, rel in ASSETS.items():
        src = ROOT/rel; cfgp=src/'meta/config.yaml'; ck=src/'best_hqnr/model.safetensors'
        meta=json.loads((src/'best_hqnr_meta.json').read_text())
        cfg=yaml.safe_load(cfgp.read_text()); dest=out/'pinned'/aid
        (dest/'best_hqnr').mkdir(parents=True,exist_ok=True)
        shutil.copy2(ck,dest/'best_hqnr/model.safetensors'); shutil.copy2(cfgp,dest/'source_config.yaml')
        mappings={}
        for split,key in (('rr','test_reduced_feeder_args'),('fr','test_full_feeder_args')):
            old=cfg[key]['dataroot']
            p=ROOT/'data/PanCollection/WV3'/('reduced_examples_h5/test_wv3_multiExm1.h5' if split=='rr' else 'full_examples_mat20/test_wv3_OrigScale_mat20.h5')
            cfg[key]['dataroot']=str(p); mappings[old]=str(p)
            assert not any(cfg[key].get(k,False) for k in ('crop','hflip','vflip','rot'))
            for pp in (p,Path(str(p).replace('.h5','_pan.h5'))):
                data.setdefault(str(pp),sha(pp))
            with h5py.File(p) as f:
                assert len(f['pan'])==20 and f['pan'].shape[-1]==(256 if split=='rr' else 512)
                if split=='rr': assert 'gt' in f
        (dest/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        assert sha(ck)==sha(dest/'best_hqnr/model.safetensors')
        if aid!='T0':
            assert (src/'results/reduced_best_hqnr.mat').is_file()
            assert (src/'results/full_best_hqnr.mat').is_file()
        assets[aid]=dict(source_run=rel,source_selector='best_hqnr',checkpoint_step=meta['step'],
            checkpoint_sha256=sha(ck),source_checkpoint=str(ck),checkpoint_file=str(dest/'best_hqnr/model.safetensors'),
            source_config=str(cfgp),source_config_sha256=sha(cfgp),mapped_config=str(dest/'config.yaml'),
            mapped_config_sha256=sha(dest/'config.yaml'),path_mapping=mappings,source_meta=meta)
        paths=[ck,cfgp,src/'best_hqnr_meta.json']+list((src/'results').glob('*.json'))
        for p in paths: protected[str(p)]=sha(p)
        # Old caches lack complete RR/environment identity. Inventory them, then recompute.
        existing[aid]=[dict(path=str(p),reuse='not_verified_missing_complete_identity')
                       for p in (src/'results/aligner_eval_v2').glob('*/*/*metrics.json')]
    for pattern in ('work_dir/cases*.txt','work_dir/_eval_phase/*.json','work_dir/_eval_phase/records/*.json',
                    'work_dir/_qrecon24/*.txt','work_dir/_qrecon24/recipe_lock.json','work_dir/*budget/ledger.json',
                    'assets/qedge9/cue_T0_AXIS16_v1.*'):
        for p in ROOT.glob(pattern):
            if p.is_file(): protected[str(p)]=sha(p)
    sources={}
    for folder in ('pa','kdv','model','feeders','tools'):
        for p in (ROOT/folder).rglob('*.py'):
            rel=p.relative_to(ROOT); sources[str(rel)]=sha(p)
            dst=out/'code_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    for p in ROOT.glob('*.py'):
        sources[p.name]=sha(p);shutil.copy2(p,out/'code_snapshot'/p.name)
    wald=Path(os.environ.get('PANCRAFTER_DLPAN','/home/knuvi/Desktop/song/DLPan-Toolbox'))
    external={str(p):sha(p) for p in wald.rglob('*.py')}
    m=dict(protocol_id=PROTOCOL,created_at=time.strftime('%Y-%m-%dT%H:%M:%S'),device=device,
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,platform=platform.platform(),
        gpu=torch.cuda.get_device_name(0) if device.startswith('cuda') else None,
        dtype='float32',autocast=False,tf32_matmul=False,tf32_cudnn=False,
        precision_note='Explicit FP32 for numerical parity; paired A_ON recomputed in the same environment',
        cudnn_benchmark=False,cudnn_deterministic=False,torch_threads=torch.get_num_threads(),
        atol=ATOL,rtol=RTOL,scenes={'rr':list(range(20)),'fr':list(range(20))},modes=MODES,
        diagnostic_sha256=sha(__file__),code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_code_sha256=sources,external_evaluator_sha256=external,assets=assets,data_sha256=data,
        protected_sources=protected,training_control='none in diagnostic; separate explicit user pause recorded in ../training_pause.json')
    write_json(out/'source_inventory.json',dict(assets=assets,existing_cache=existing))
    write_json(manifest_path,m);return m


def gstats(x, groups):
    t=x.reshape(x.shape[0],groups,-1)
    return t.mean(-1,keepdim=True)[...,None,None],t.var(-1,unbiased=False,keepdim=True)[...,None,None]


def head(a,z): return a.fc2(F.silu(a.fc1(z)))


def trace_full(model,p,M):
    a=model.aligner;pv,mv=model._view(p),model._view(M);bank={};hooks=[];features=[]
    names=tuple(n for n,l in a.named_modules() if isinstance(l,torch.nn.GroupNorm))
    assert names==GN_NAMES,names
    for name in names:
        layer=a.get_submodule(name)
        def pre(layer,inputs,name=name):
            mu,var=gstats(inputs[0],layer.num_groups)
            bank[name]=dict(mean=mu,var=var,eps=layer.eps,groups=layer.num_groups,
                            weight=layer.weight.detach().clone(),bias=layer.bias.detach().clone(),shape=list(inputs[0].shape))
        hooks.append(layer.register_forward_pre_hook(pre))
    hooks.append(a.res.register_forward_hook(lambda _l,_i,o:features.append(o.detach())))
    try: c,z=a(pv,mv,return_features=True)
    finally:
        for h in hooks:h.remove()
    zstats={k:dict(mean=x.mean((2,3),keepdim=True),var=x.var((2,3),unbiased=False,keepdim=True)) for k,x in [('pan',pv),('ms',mv)]}
    return c,z,features[0],bank,zstats


@contextlib.contextmanager
def fixed_gn(a,bank):
    hooks=[]
    try:
        for name,v in bank.items():
            layer=a.get_submodule(name)
            def replace(layer,inputs,_output,v=v):
                x=inputs[0];B,C,H,W=x.shape
                t=x.reshape(B,v['groups'],C//v['groups'],H,W)
                t=(t-v['mean'])*torch.rsqrt(v['var']+v['eps'])
                return t.reshape_as(x)*layer.weight[None,:,None,None]+layer.bias[None,:,None,None]
            hooks.append(layer.register_forward_hook(replace))
        yield
    finally:
        for h in hooks:h.remove()


def norm_forward(a,p,m,mode,bank,zstats):
    if mode.startswith('ZF'):
        p=(p-zstats['pan']['mean'])/torch.sqrt(zstats['pan']['var']+1e-6)
        m=(m-zstats['ms']['mean'])/torch.sqrt(zstats['ms']['var']+1e-6)
    else:p,m=znorm(p),znorm(m)
    with fixed_gn(a,bank) if mode.endswith('GF') else contextlib.nullcontext():
        x=torch.cat([a.pan_stem(p),a.ms_stem(m)],1)
        feature=a.res(a.joint2(a.joint1(x)));z=feature.mean((2,3));c=head(a,z)
    return c,z,feature


def origins(N,size): return [(y,x) for y in range(0,N,size) for x in range(0,N,size)]


def crops(x,oo,size): return torch.cat([x[...,y:y+size,z:z+size] for y,z in oo])


def align_batch(model,p,M,oo,size,batch=16,ret=False):
    cs,zs=[],[]
    for n in range(0,len(oo),batch):
        o=oo[n:n+batch];pp,mm=model._view(crops(p,o,size)),model._view(crops(M,o,size))
        c,z=model.aligner(pp,mm,return_features=True);cs.append(c);zs.append(z)
    c,z=torch.cat(cs),torch.cat(zs)
    return (c,z) if ret else c


def metric_scene(sr,split,raw,wald):
    s=sr.transpose(1,2,0).astype(np.float64)
    if split=='rr':
        s=s[20:-21,20:-21];g=raw['gt'].transpose(1,2,0).astype(np.float64)[20:-21,20:-21]
        return dict(ergas=ergas(s,g),sam=sam(s,g),psnr=psnr_global(s,g,2047),
                    ssim=ssim_skimage(s,g,2047),scc=scc_dlpan(s,g),q8=float(q2n(g,s,32,32)[0]),
                    rmse=float(np.sqrt(((s-g)**2).mean())),cc=float(np.corrcoef(s.ravel(),g.ravel())[0,1]))
    L=raw['lms'].transpose(1,2,0).astype(np.float64);P=raw['pan'][0].astype(np.float64)
    dl=d_lambda_k(s,L,'wv3',4,32,wald);ds=d_s(s,L,P,4,32,wald)
    return dict(d_lambda=float(dl),d_s=float(ds),hqnr_raw=float((1-dl)*(1-ds)))


def aggregate(rows):
    keys=[k for k in rows[0] if k!='scene']
    return {k:float(np.mean([r[k] for r in rows])) for k in keys}|{
        k+'_sd':float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else 0. for k in keys}


def scene_work(model,item,split,scene,od,raw,wald,device):
    start=time.perf_counter();timing={};checks=[]
    lms,ms,lp,p=item[1:] if split=='rr' else item
    p,ms,lp=[x[None].to(device) for x in (p,ms,lp)]
    N=p.shape[-1];assert N==(256 if split=='rr' else 512);assert model.aligner_margin==4
    M=F.interpolate(ms,scale_factor=4,mode='bicubic');a=model.aligner
    cF,zF,fF,bank,zstats=trace_full(model,p,M)
    assert fF.shape[-1]==(N-8)//4
    checks.append(parity(head(a,zF),cF,'g(z_full)==original'))
    # C00 source forward and call-contract checks on scene 0 of each split.
    if scene==0:
        before=state_sha(model);ys={}
        for mode,expected in [('A_ON',(1,1)),('A_ZERO_WARP',(0,1)),('A_BYPASS_RAW',(0,0))]:
            with CallCounter(model) as cc:ys[mode]=forward_mode(model,p,ms,lp,mode)
            assert (cc.aligner,cc.warp)==expected,(mode,cc.as_dict())
            checks.append(dict(name=mode+'_calls',passed=True,**cc.as_dict()))
        sw=torch.ones(1,device=device)
        independent=M+model.backbone(p,lp,ms,sw)
        checks.append(parity(ys['A_BYPASS_RAW'],independent,'NOA independent residual'))
        zd=(ys['A_ZERO_WARP']-ys['A_BYPASS_RAW']).abs()
        checks.append(dict(name='ZERO_minus_NOA',max_abs=float(zd.max()),mean_abs=float(zd.mean())))
        assert state_sha(model)==before and model.sampler
    sync(device);t=time.perf_counter()
    rows,probe_rows,summ=[],[],[];native={};feature_bank={}
    warped=[warp_pan(p,torch.tensor([e],device=device,dtype=torch.float32)) for r,e in PROBES]
    supports=[warp_support_mask(N,N,torch.tensor([e],device=device,dtype=torch.float32)) for r,e in PROBES]
    for size in ([64,128,256] if split=='rr' else [64,128,256,512]):
        oo=origins(N,size);c0,z=align_batch(model,p,M,oo,size,ret=True);native[size]=c0;feature_bank[size]=z
        allcj=torch.stack([align_batch(model,pp,M,oo,size) for pp in warped])
        eps=torch.tensor([e for r,e in PROBES],device=device,dtype=torch.float32)
        residual=allcj+eps[:,None,:]-c0[None]
        q=residual.abs().mean((0,2));cn,qn,rn,pn=[x.cpu().numpy() for x in (c0,q,residual,allcj)]
        for k,(y,x) in enumerate(oo):
            rows.append(dict(scene=scene,size=size,crop_index=k,y0=y,x0=x,is_full=size==N,dy=float(cn[k,0]),dx=float(cn[k,1]),q_size=float(qn[k])))
            for j,(radius,e) in enumerate(PROBES):
                valid=supports[j][...,y+4:y+size-4,x+4:x+size-4]
                probe_rows.append(dict(scene=scene,size=size,crop_index=k,y0=y,x0=x,radius=radius,epsilon_y=e[0],epsilon_x=e[1],
                    c_native_y=float(cn[k,0]),c_native_x=float(cn[k,1]),c_probe_y=float(pn[j,k,0]),c_probe_x=float(pn[j,k,1]),
                    residual_y=float(rn[j,k,0]),residual_x=float(rn[j,k,1]),support_all_valid=bool(valid.all()),support_fraction=float(valid.float().mean())))
        summ.append(dict(scene=scene,size=size,n_crops=len(oo),q_mean=float(qn.mean()),q_median=float(np.median(qn)),q_p90=float(np.quantile(qn,.9)),
            dy_mean=float(cn[:,0].mean()),dx_mean=float(cn[:,1].mean()),dy_std=float(cn[:,0].std(ddof=1)) if len(oo)>1 else None,
            dx_std=float(cn[:,1].std(ddof=1)) if len(oo)>1 else None))
    checks.append(parity(native[N],cF,'full view reuse'))
    if scene==0:
        for size in (64,128):
            y,x=origins(N,size)[0];ref=model.predict_delta(p[...,y:y+size,x:x+size],M[...,y:y+size,x:x+size])
            checks.append(parity(native[size][:1],ref,'batch_vs_single_'+str(size)))
    sync(device);timing['C01_seconds']=time.perf_counter()-t;t=time.perf_counter()
    D,Z=native[64],feature_bank[64];cFM=head(a,Z.mean(0,keepdim=True));cSM=D.mean(0,keepdim=True);cMED=torch.quantile(D,.5,dim=0,keepdim=True)
    checks.append(parity(head(a,Z),D,'same feature bank heads'))
    checks.append(parity(cF-cSM,(cF-cFM)+(cFM-cSM),'vector decomposition'))
    # Cover every full pre-GAP feature position, including remainder tiles.
    part_z,counts=[],[]
    for y in range(0,fF.shape[-2],14):
        for x in range(0,fF.shape[-1],14):
            part=fF[...,y:y+14,x:x+14];part_z.append(part.mean((2,3)));counts.append(part.shape[-2]*part.shape[-1])
    weights=torch.tensor(counts,device=device,dtype=torch.float32);weights/=weights.sum()
    parts=torch.cat(part_z);weighted=(parts*weights[:,None]).sum(0,keepdim=True)
    checks.append(parity(weighted,zF,'full feature partition covers all positions'))
    checks.append(parity(head(a,weighted),cF,'full partition head parity'))
    dec=dict(scene=scene,c_full=plain(cF[0]),c_feature_mean=plain(cFM[0]),c_shift_mean=plain(cSM[0]),c_median=plain(cMED[0]),
        feature_generation_delta=plain((cF-cFM)[0]),head_order_delta=plain((cFM-cSM)[0]),
        feature_generation_l2=float((cF-cFM).norm()),head_order_l2=float((cFM-cSM).norm()),full_crop_l2=float((cF-cMED).norm()),
        full_partition_head_order_delta=plain((cF-(head(a,parts)*weights[:,None]).sum(0,keepdim=True))[0]))
    c3=od/'C03'/split;c3.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(c3/f'feature_bank_{scene:02}.npz',crop_features=Z.cpu().numpy(),crop_shifts=D.cpu().numpy(),
        full_feature=zF.cpu().numpy(),partition_features=parts.cpu().numpy(),partition_weights=weights.cpu().numpy())
    sync(device);timing['C03_seconds']=time.perf_counter()-t;t=time.perf_counter()
    normrows=[];norms={};features={};core=[];oo=origins(N,64)
    for mode in NORMS:
        cs,zs=[],[]
        for k in range(0,len(oo),16):
            batch=oo[k:k+16];pp,mm=model._view(crops(p,batch,64)),model._view(crops(M,batch,64))
            c,z,f=norm_forward(a,pp,mm,mode,bank,zstats);cs.append(c);zs.append(z)
            if mode=='ZF_GF':
                for j,(y,x) in enumerate(batch):
                    fc=f[j:j+1,:,4:-4,4:-4];ff=fF[...,y//4+4:y//4+f.shape[-2]-4,x//4+4:x//4+f.shape[-1]-4]
                    result=parity(fc,ff,'matched_core');core.append(dict(scene=scene,crop_index=k+j,y0=y,x0=x,**result))
        cs,zs=torch.cat(cs),torch.cat(zs);norms[mode]=torch.quantile(cs,.5,dim=0,keepdim=True);features[mode]=zs
        if mode=='ZL_GL':
            checks.append(parity(cs,D,'ZL_GL original crop shift'));checks.append(parity(zs,Z,'ZL_GL original crop feature'))
        for k,(y,x) in enumerate(oo):
            normrows.append(dict(scene=scene,norm_mode=mode,crop_index=k,y0=y,x0=x,dy=float(cs[k,0]),dx=float(cs[k,1]),
                delta_dy_vs_local=float(cs[k,0]-D[k,0]),delta_dx_vs_local=float(cs[k,1]-D[k,1]),feature_delta_l2=float((zs[k]-Z[k]).norm())))
    c4=od/'C04'/split;c4.mkdir(parents=True,exist_ok=True)
    arrays={'full_pre_gap':fF.cpu().numpy()}
    for name,v in bank.items():
        for key in ('mean','var','weight','bias'):arrays[name+'_'+key]=v[key].cpu().numpy()
    for name,v in zstats.items():
        for key in ('mean','var'):arrays['znorm_'+name+'_'+key]=v[key].cpu().numpy()
    for mode,z in features.items():arrays[mode+'_crop_features']=z.cpu().numpy()
    np.savez_compressed(c4/f'norm_reference_{scene:02}.npz',**arrays)
    write_json(c4/f'norm_reference_{scene:02}.json',dict(scene=scene,split=split,
        input_zscore_epsilon=1e-6,variance='population',
        layers={n:{k:v[k] for k in ('eps','groups','shape')} for n,v in bank.items()}))
    interaction=norms['ZF_GF']-norms['ZF_GL']-norms['ZL_GF']+norms['ZL_GL']
    dec['norm_interaction_vector']=plain(interaction[0])
    sync(device);timing['C04_seconds']=time.perf_counter()-t;t=time.perf_counter()
    shifts={'A_ON':cF,'A_ZERO_WARP':torch.zeros_like(cF),'A_CROP64_MED':cMED,
            'A_CROP64_FEAT_MEAN':cFM,'A_CROP64_SHIFT_MEAN':cSM,
            **{'A_NORM_'+key+'_MED64':norms[key] for key in NORMS[1:]}}
    sr={}
    for mode in MODES:
        if mode in ('A_ON','A_ZERO_WARP','A_BYPASS_RAW'):y=forward_mode(model,p,ms,lp,mode)
        else:y=model(p,ms,lp,delta_override=shifts[mode])['y']
        sr[mode]=((y.clamp(-1,1).float().cpu().numpy()[0]+1.)/2.*2047.)
    sync(device);timing['U_inference_seconds']=time.perf_counter()-t;t=time.perf_counter()
    metrics={mode:dict(scene=scene,**metric_scene(s,split,raw,wald)) for mode,s in sr.items()}
    timing['metric_seconds']=time.perf_counter()-t;timing['total_scene_seconds']=time.perf_counter()-start
    return dict(native=rows,probes=probe_rows,scene_summary=summ,decomposition=dec,norm=normrows,core=core,
                checks=checks,metrics=metrics,timing=timing,shifts={k:plain(v[0]) for k,v in shifts.items()})


def summarize(out,m,assets,scenes,pilot):
    rows,status,diagnostics=[],[],[]
    base=out/('pilot' if pilot else 'results')
    for aid in assets:
        od=base/aid/m['assets'][aid]['checkpoint_sha256'][:16]
        for mode in MODES:
            rrp,frp=od/'metrics'/mode/'rr.json',od/'metrics'/mode/'fr.json'
            if not rrp.exists() or not frp.exists():continue
            rr,fr=json.loads(rrp.read_text()),json.loads(frp.read_text())
            ron=json.loads((od/'metrics/A_ON/rr.json').read_text());fon=json.loads((od/'metrics/A_ON/fr.json').read_text())
            rows.append(dict(asset=aid,step=m['assets'][aid]['checkpoint_step'],mode=mode,rr_ergas=rr['ergas'],rr_sam=rr['sam'],rr_psnr=rr['psnr'],
                rr_ssim=rr['ssim'],rr_scc=rr['scc'],rr_q8=rr['q8'],fr_d_lambda=fr['d_lambda'],fr_d_s=fr['d_s'],fr_hqnr=fr['hqnr_raw'],
                delta_ergas=rr['ergas']-ron['ergas'],delta_hqnr=fr['hqnr_raw']-fon['hqnr_raw'],
                joint_pass=bool(fr['hqnr_raw']>=.9585 and rr['ergas']<2.040) if not pilot else None,n_scenes=len(scenes)))
        for case in ('C00','C01-RR','C01-FR','C02','C03','C04','C05'):
            done=all((od/'scene_records'/sp/f'{s:02}.json').exists() for sp in ('rr','fr') for s in scenes)
            status.append(dict(asset=aid,case=case,status='partial_pilot' if pilot and done else ('complete' if done else 'pending'),n_scenes_per_split=len(scenes)))
        for split in ('rr','fr'):
            recs=[json.loads(p.read_text()) for p in sorted((od/'scene_records'/split).glob('*.json'))]
            if not recs:continue
            dec=[r['decomposition'] for r in recs]
            for size in sorted({s['size'] for r in recs for s in r['scene_summary']}):
                ss=[s for r in recs for s in r['scene_summary'] if s['size']==size]
                probes=[p for r in recs for p in r['probes'] if p['size']==size]
                gain=[((p['c_probe_y']-p['c_native_y'])*p['epsilon_y']+(p['c_probe_x']-p['c_native_x'])*p['epsilon_x'])/(p['radius']**2) for p in probes]
                diagnostics.append(dict(asset=aid,split=split,size=size,n_scenes=len(ss),q_mean=float(np.mean([s['q_mean'] for s in ss])),
                    q_scene_median_mean=float(np.mean([s['q_median'] for s in ss])),
                    within_scene_dy_std_mean=float(np.mean([s['dy_std'] for s in ss])) if ss[0]['dy_std'] is not None else None,
                    within_scene_dx_std_mean=float(np.mean([s['dx_std'] for s in ss])) if ss[0]['dx_std'] is not None else None,
                    axial_response_gain_mean=float(np.mean(gain)),full_crop_l2_mean=float(np.mean([d['full_crop_l2'] for d in dec])),
                    full_crop_l2_p90=float(np.quantile([d['full_crop_l2'] for d in dec],.9)),
                    feature_generation_l2_mean=float(np.mean([d['feature_generation_l2'] for d in dec])),
                    head_order_l2_mean=float(np.mean([d['head_order_l2'] for d in dec]))))
    write_csv(base/'metrics_summary.csv',rows);write_csv(base/'case_status.csv',status)
    write_csv(base/'diagnostic_summary.csv',diagnostics)
    text=['# PAN ASV — '+('pilot (partial)' if pilot else 'C00–C05 evaluation'),'',
          'Same pinned checkpoint per asset; FP32, no training or parameter updates. Raw-original FR references.','',
          '| Asset | Mode | ERGAS ↓ | HQNR ↑ | ΔERGAS | ΔHQNR | Joint pass |','|---|---|---:|---:|---:|---:|---|']
    for r in rows:text.append(f"| {r['asset']} | {r['mode']} | {r['rr_ergas']:.6f} | {r['fr_hqnr']:.6f} | {r['delta_ergas']:+.6f} | {r['delta_hqnr']:+.6f} | {r['joint_pass']} |")
    text+=['','Interpretation: q_size measures probe-response consistency, not absolute alignment error. Each scene is one observation; crops are not independent scenes. A mode comparison is an intervention on this trained checkpoint, not a retrained-method comparison.']
    text+=['','Precision: all nine modes were paired again in explicit FP32 with TF32 disabled for feature parity. Legacy TF32-enabled metrics are reference-only and not used as the paired baseline. The checkpoint, evaluator, crop and intensity definitions are unchanged.']
    (base/'report.md').write_text('\n'.join(text)+'\n')
    return rows


def run(m,out,asset_ids,device,pilot):
    scenes=[0] if pilot else list(range(20));base=out/('pilot' if pilot else 'results');alltime=[]
    wald=load_dlpan(os.environ.get('PANCRAFTER_DLPAN','/home/knuvi/Desktop/song/DLPan-Toolbox'))
    for aid in asset_ids:
        asset=m['assets'][aid];cfg=yaml.safe_load(Path(asset['mapped_config']).read_text())
        model,_,how=build(cfg,str(Path(asset['checkpoint_file']).parents[1]),'best_hqnr');model=model.to(device).eval()
        h0=state_sha(model);od=base/aid/asset['checkpoint_sha256'][:16];od.mkdir(parents=True,exist_ok=True)
        identity=dict(protocol=PROTOCOL,asset=aid,checkpoint_sha256=asset['checkpoint_sha256'],step=asset['checkpoint_step'],
            mapped_config_sha256=asset['mapped_config_sha256'],data_sha256=m['data_sha256'],diagnostic_sha256=m['diagnostic_sha256'],
            device=device,precision=m['dtype'],tf32=False,scenes=scenes,how=how,source_selector=asset['source_selector'],
            crop_policy='native pixel scale, margin4 once, non-overlap',probe_bank=PROBES)
        write_json(od/'identity.json',identity)
        for split,key in [('rr','test_reduced_feeder_args'),('fr','test_full_feeder_args')]:
            fa=cfg[key];module,name=cfg['feeder'].rsplit('.',1);ds=getattr(importlib.import_module(module),name)(**fa)
            assert len(ds)==20;records=[]
            with h5py.File(fa['dataroot']) as h5,torch.no_grad():
                for scene in scenes:
                    path=od/'scene_records'/split/f'{scene:02}.json'
                    if path.exists():rec=json.loads(path.read_text())
                    else:
                        raw={k:h5[k][scene] for k in (['gt'] if split=='rr' else ['lms','pan'])}
                        rec=scene_work(model,ds[scene],split,scene,od,raw,wald,device)
                        rec['identity']=dict(asset=aid,checkpoint_sha256=asset['checkpoint_sha256'],step=asset['checkpoint_step'],split=split,scene=scene)
                        write_json(path,rec)
                    records.append(rec);alltime.append(dict(asset=aid,split=split,scene=scene,**rec['timing']))
                    print(f"{aid} {split} {scene+1:02}/20: {rec['timing']['total_scene_seconds']:.2f}s",flush=True)
                    write_json(base/'progress.json',dict(asset=aid,split=split,scene=scene,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S')))
            for name,filename in [('native','native_views.csv'),('probes','probe_responses.csv'),('scene_summary','scene_summary.csv')]:
                write_csv(od/'C01'/split/filename,[r for rec in records for r in rec[name]])
            write_json(od/'C03'/split/'head_decomposition.json',[r['decomposition'] for r in records])
            write_csv(od/'C03'/split/'head_decomposition.csv',[r['decomposition'] for r in records])
            write_csv(od/'C04'/split/'crop_norm_results.csv',[r for rec in records for r in rec['norm']])
            write_json(od/'C04'/split/'interior_parity.json',[r for rec in records for r in rec['core']])
            write_json(od/'C00'/f'sanity_{split}.json',[r for rec in records for r in rec['checks']])
            for mode in MODES:
                vals=[r['metrics'][mode] for r in records];metric=aggregate(vals)
                metric.update(n=len(vals),scenes=scenes,identity=identity,split=split,status='partial_pilot' if pilot else 'complete')
                write_json(od/'metrics'/mode/f'{split}.json',metric);write_csv(od/'metrics'/mode/f'per_scene_{split}.csv',vals)
        write_json(od/'C02/modes.json',{mode:{sp:json.loads((od/'metrics'/mode/f'{sp}.json').read_text()) for sp in ('rr','fr')} for mode in MODES[:4]})
        assert state_sha(model)==h0 and all(p.grad is None for p in model.parameters())
        write_json(od/'C00/state_invariance.json',dict(before=h0,after=state_sha(model),all_grad_none=True,sampler_restored=model.sampler))
        del model
        if device.startswith('cuda'):torch.cuda.empty_cache()
        summarize(out,m,asset_ids,scenes,pilot)
    changed=[p for p,h in m['protected_sources'].items() if not Path(p).exists() or sha(p)!=h]
    changed += [p for p,h in m['source_code_sha256'].items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    write_json(base/'source_invariance.json',dict(passed=not changed,changed=changed,n_checked=len(m['protected_sources'])))
    assert not changed,changed
    write_csv(base/'timing.csv',alltime)
    split_means={sp:{k:float(np.mean([r[k] for r in alltime if r['split']==sp])) for k in alltime[0] if k.endswith('seconds')} for sp in ('rr','fr')}
    write_json(base/'timing.json',dict(by_split_mean_seconds=split_means,total_measured_seconds=sum(r['total_scene_seconds'] for r in alltime),
        projected_four_assets_20_scenes_seconds=4*20*sum(v['total_scene_seconds'] for v in split_means.values()),pilot=pilot))
    summarize(out,m,asset_ids,scenes,pilot)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root',required=True);ap.add_argument('--device',required=True)
    ap.add_argument('--manifest');ap.add_argument('--inventory-only',action='store_true');ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--assets',default=','.join(ASSETS));ap.add_argument('--case',default='all',choices=['all','C00-C05'])
    args=ap.parse_args();out=Path(args.output_root).resolve()
    assert out!=ROOT and not str(out).startswith(str(ROOT/'work_dir')),'Output must be separate from training work_dir'
    out.mkdir(parents=True,exist_ok=True)
    writer_lock=(out/'.writer.lock').open('w')
    fcntl.flock(writer_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    m=pin_inventory(out,args.device)
    if args.manifest:assert Path(args.manifest).resolve()==out/'manifest.json'
    if args.inventory_only:print(json.dumps({a:v['checkpoint_step'] for a,v in m['assets'].items()}));return
    ids=args.assets.split(',');assert all(a in ASSETS for a in ids)
    try:run(m,out,ids,args.device,args.pilot)
    except Exception as e:
        write_json(out/('pilot' if args.pilot else 'results')/'failure.json',dict(error=repr(e),time=time.strftime('%Y-%m-%dT%H:%M:%S')))
        raise


if __name__=='__main__':main()
