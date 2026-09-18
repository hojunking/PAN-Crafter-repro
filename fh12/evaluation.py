"""FH12 inference and official metrics; no legacy log/proxy metric fallback."""
from pathlib import Path
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from fh12.common import check_deadline
from tools.eval_dlpan import scc_dlpan, psnr_global, ssim_skimage
from tools.metrics.eval_rr import sam, ergas
from tools.metrics.q2n import q2n

RR_KEYS = ('ergas', 'scc', 'psnr', 'sam', 'q8', 'ssim')
FR_KEYS = ('hqnr', 'd_lambda', 'd_s')


def rr_metrics(sr, gt, include_q=True):
    """NCHW DN, all 20 scenes, MATLAB dim_cut=21; no rounding except within Q2n."""
    sr, gt = np.asarray(sr, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if sr.shape != gt.shape or sr.ndim != 4 or sr.shape[1:] != (8, 256, 256) or len(sr) != 20:
        raise ValueError(f'FH12 RR must be corresponding 20x8x256x256: {sr.shape}, {gt.shape}')
    if not np.isfinite(sr).all() or not np.isfinite(gt).all():
        raise FloatingPointError('Nonfinite RR inputs, including discarded borders')
    p, r = sr.transpose(0,2,3,1)[:,20:-21,20:-21], gt.transpose(0,2,3,1)[:,20:-21,20:-21]
    rows=[]
    for a,b in zip(p,r):
        item=dict(ergas=ergas(a,b), sam=sam(a,b), scc=scc_dlpan(a,b),
                  psnr=psnr_global(a,b,2047.), ssim=ssim_skimage(a,b,2047.))
        if include_q:
            item['q8']=q2n(b,a,32,32)[0]
        item['band_relative_mse']=(np.mean((a-b)**2,axis=(0,1))/np.mean(b,axis=(0,1))**2).tolist()
        rows.append(item)
    keys=RR_KEYS if include_q else tuple(k for k in RR_KEYS if k != 'q8')
    out={k:float(np.mean([r[k] for r in rows])) for k in keys}
    if not all(np.isfinite(v) for v in out.values()):
        raise FloatingPointError('Nonfinite official RR metric')
    out.update(per_scene=rows, n_scenes=20, crop='20:-21', q_block=32,
               standard_deviation={k:float(np.std([r[k] for r in rows],ddof=1)) for k in keys},
               official_complete=bool(include_q), protocol='DLPan_RR_dim21_Q32_SCC2Dzero_PSNRglobal_SSIMgauss11')
    return out


class FRMetrics:
    """Cache only immutable native PAN reference terms; identical DLPan full-frame formulas."""
    def __init__(self, dataset):
        import h5py
        from tools.metrics.eval_fr import load_dlpan, imresize_matlab, _blockproc_uqi
        root=Path(__file__).resolve().parents[1]
        self.wald=load_dlpan(os.environ.get('PANCRAFTER_DLPAN',str(root.parent/'DLPan-Toolbox')))
        with h5py.File(dataset.raw_h5_path,'r') as f:
            self.lms=np.asarray(f['lms'],dtype=np.float64).transpose(0,2,3,1)
            self.pan=np.asarray(f['pan'],dtype=np.float64)[:,0]
        if self.lms.shape != (20,512,512,8) or self.pan.shape != (20,512,512):
            raise ValueError('FH12 FR must be native WV3 paper mat20 at 512x512')
        self.reference=[]
        for m,p in zip(self.lms,self.pan):
            lp=self.wald.interp23tap(imresize_matlab(p,1/4)[...,None],4)[...,0]
            self.reference.append([_blockproc_uqi(m[...,b],lp,32) for b in range(8)])

    def __call__(self,sr,deadline=None):
        from tools.metrics.eval_fr import mtf_filter, _blockproc_uqi
        sr=np.asarray(sr,dtype=np.float64)
        if sr.shape != (20,8,512,512) or not np.isfinite(sr).all():
            raise ValueError('Invalid/nonfinite FH12 FR prediction')
        rows=[]
        for i,f in enumerate(sr.transpose(0,2,3,1)):
            check_deadline(deadline)
            dl=1-q2n(self.lms[i],mtf_filter(f,'wv3',4,self.wald),32,32)[0]
            ds=float(np.mean([abs(_blockproc_uqi(f[...,b],self.pan[i],32)-self.reference[i][b]) for b in range(8)]))
            rows.append(dict(d_lambda=float(dl),d_s=ds,hqnr=float((1-dl)*(1-ds))))
        out={k:float(np.mean([x[k] for x in rows])) for k in FR_KEYS}
        if not all(np.isfinite(v) for v in out.values()):
            raise FloatingPointError('Nonfinite official FR metric')
        out.update(per_scene=rows,n_scenes=20,eval_mode='A_ON',reference='native_PAN',support='full512',
                   standard_deviation={k:float(np.std([r[k] for r in rows],ddof=1)) for k in FR_KEYS})
        return out


@torch.no_grad()
def infer(model,dataset,device,deadline=None,batch_size=1):
    """No GT enters forward; native coordinates, FP32, shared frequency frontend."""
    was_training=model.training
    model.eval(); sr=[]; deltas=[]
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,drop_last=False)
    try:
        for batch in loader:
            check_deadline(deadline)
            data=batch[:-1]  # final tensor is the stable scene/view identity
            if dataset.has_gt:
                _gt,lms,ms,lp,pan=data
            else:
                lms,ms,lp,pan=data
            out=model(pan.to(device),ms.to(device),lp.to(device))
            if not torch.isfinite(out['y']).all() or not torch.isfinite(out['delta']).all():
                raise FloatingPointError('Nonfinite FH12 inference')
            sr.append(((out['y'].float().clamp(-1,1)+1)*1023.5).cpu().numpy())
            deltas.append(out['delta'].float().cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(sr),np.concatenate(deltas)


@torch.no_grad()
def validation_ergas(model,dataset,device,deadline=None):
    """1080 independent validation patches, plain full-frame ERGAS for selection only."""
    was_training=model.training; model.eval(); vals=[]
    try:
        for gt,_lms,ms,lp,pan,_meta in DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0):
            check_deadline(deadline)
            y=model(pan.to(device),ms.to(device),lp.to(device))['y']
            y=(y.float().clamp(-1,1)+1)*1023.5
            target=(gt.to(device).float()+1)*1023.5
            mu=target.mean((2,3))
            per=25*torch.sqrt((((y-target)**2).mean((2,3))/mu.clamp_min(1e-12)**2).mean(1))
            vals.extend(per.cpu().tolist())
    finally:
        model.train(was_training)
    if len(vals)!=len(dataset) or not np.isfinite(vals).all():
        raise ValueError('Incomplete/nonfinite FH12 validation')
    return float(np.mean(vals))


def native_gt(dataset):
    import h5py
    with h5py.File(dataset.raw_h5_path,'r') as f:
        return np.asarray(f['gt'],dtype=np.float64)


def evaluate_model(model,datasets,device,engine=None,deadline=None,include_q=True,with_val=True):
    start=time.monotonic()
    engine=engine or FRMetrics(datasets['fr'])
    fr_sr,fd=infer(model,datasets['fr'],device,deadline)
    fr=engine(fr_sr,deadline)
    rr_sr,rd=infer(model,datasets['rr'],device,deadline)
    rr=rr_metrics(rr_sr,native_gt(datasets['rr']),include_q)
    val=validation_ergas(model,datasets['val'],device,deadline) if with_val else None
    return dict(fr=fr,rr=rr,val_ergas=val,seconds=time.monotonic()-start,
                shift=dict(fr_mean=fd.mean(0).tolist(),fr_max_abs=float(np.abs(fd).max()),
                           rr_mean=rd.mean(0).tolist(),rr_max_abs=float(np.abs(rd).max())))
