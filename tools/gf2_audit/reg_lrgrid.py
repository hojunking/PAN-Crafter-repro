"""Cleaner PAN-vs-MS registration on a common LR grid (CPU).
FR: PAN_LR = genMTF(sensor MS GNyq)-lowpass(PAN)[2::4,2::4] (same operator/phase used to make ms from gt in RR);
    compare with ms (native MS, 128x128) per band and NNLS intensity. Shift via upsampled cross-correlation (no phase norm) + NCC parabolic.
    Also inter-band: ms band b vs band 1 (G).
    Also protocol offset: D_s pan_filt (imresize 1/4 -> interp23tap) vs PAN blurred with matching gaussian -> shift.
RR test: gt bands / NNLS-intensity vs pan on the 256 grid (no resampling).
Shifts reported in the grid's own px; FR LR-grid px x4 = HR px."""
import os, sys, json, numpy as np, h5py
sys.path.insert(0, '/home/knuvi/Desktop/song/PAN-Crafter')
os.environ.setdefault('PANCRAFTER_DLPAN', '/home/knuvi/Desktop/song/DLPan-Toolbox')
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
from tools.metrics.eval_fr import genmtf_matlab, GNYQ_TABLE, load_dlpan, imresize_matlab
from skimage.registration import phase_cross_correlation
wald = load_dlpan(os.environ['PANCRAFTER_DLPAN'])
ROOT='/home/knuvi/Desktop/song/PAN-Crafter/data/PanCollection'; OUT=os.path.dirname(os.path.abspath(__file__))
def filt(img,k): p=k.shape[0]//2; return fftconvolve(np.pad(img,p,mode='edge'), k[::-1,::-1], mode='valid')
def hp(x,s=3.0): x=x.astype(np.float64); return x-gaussian_filter(x,s)
def shift_est(ref, mov, maxs=3):
    a=hp(ref); b=hp(mov); a=(a-a.mean())/(a.std()+1e-12); b=(b-b.mean())/(b.std()+1e-12)
    s,_,_=phase_cross_correlation(a, b, upsample_factor=100, normalization=None)
    # NCC parabolic (crop interior to avoid circular wrap)
    H,W=a.shape; m=8; A=a[m:-m,m:-m]; best=None
    for dy in range(-maxs,maxs+1):
        for dx in range(-maxs,maxs+1):
            B=b[m+dy:H-m+dy, m+dx:W-m+dx]; c=float((A*B).mean())
            if best is None or c>best[0]: best=(c,dy,dx)
    c0,dy,dx=best
    def cc(dy,dx): B=b[m+dy:H-m+dy, m+dx:W-m+dx]; return float((A*B).mean())
    def para(m1,c,p1): d=m1-2*c+p1; return 0.0 if d==0 else 0.5*(m1-p1)/d
    sy=para(cc(dy-1,dx),c0,cc(dy+1,dx)) if abs(dy)<maxs else 0.0; sx=para(cc(dy,dx-1),c0,cc(dy,dx+1)) if abs(dx)<maxs else 0.0
    return dict(pcc_dy=float(s[0]), pcc_dx=float(s[1]), ncc_dy=float(dy+sy), ncc_dx=float(dx+sx), ncc_peak=c0)
def nnls_int(cube_hwc, tgt): A=cube_hwc.reshape(-1,cube_hwc.shape[2]); w,_=nnls(A,tgt.ravel()); return (A@w).reshape(tgt.shape), w
def agg(rows,k):
    v=np.array([r[k] for r in rows]); return dict(mean=float(v.mean()), median=float(np.median(v)), sd=float(v.std(ddof=1)) if len(v)>1 else 0.0, absmean=float(np.abs(v).mean()), min=float(v.min()), max=float(v.max()))
def fr(sensor, h5main, tag):
    gn=GNYQ_TABLE.get(sensor.upper()) or None
    with h5py.File(h5main) as f: ms=f['ms'][:]; pan=f['pan'][:][:,0]; lms=f['lms'][:]
    nb=ms.shape[1]; gnyq=gn if gn else [0.3]*nb
    kpan=genmtf_matlab([float(np.mean(gnyq))],4,41)[:,:,0]
    rows=[]; band_rows=[]; ib_rows=[]; proto_rows=[]
    for i in range(pan.shape[0]):
        P=pan[i].astype(np.float64); M=ms[i].transpose(1,2,0).astype(np.float64)
        PL=filt(P,kpan)[2::4,2::4]
        I,w=nnls_int(M,PL); r=shift_est(PL,I); r['scene']=i; r['nnls_w']=[float(x) for x in w]; rows.append(r)
        for b in range(nb):
            rb=shift_est(PL,M[:,:,b]); rb.update(scene=i,band=b); band_rows.append(rb)
            if b!=1:
                ib=shift_est(M[:,:,1],M[:,:,b]); ib.update(scene=i,band=b); ib_rows.append(ib)
        # protocol offset of D_s pan_filt vs PAN itself (both HR grid): compare pan_filt with gaussian-blurred PAN (sigma matched ~ 4x downsample)
        pl=imresize_matlab(P,0.25); pf=wald.interp23tap(pl[:,:,None],4)[:,:,0]
        Pb=gaussian_filter(P,1.6)
        pr=shift_est(Pb,pf,maxs=2); pr['scene']=i; proto_rows.append(pr)
    res=dict(tag=tag, sensor=sensor, n=len(rows), file=h5main,
             nnls_vs_panLR={k:agg(rows,k) for k in ['pcc_dy','pcc_dx','ncc_dy','ncc_dx','ncc_peak']},
             per_band={b:{k:agg([r for r in band_rows if r['band']==b],k) for k in ['pcc_dy','pcc_dx','ncc_dy','ncc_dx']} for b in range(nb)},
             interband_vs_G={b:{k:agg([r for r in ib_rows if r['band']==b],k) for k in ['pcc_dy','pcc_dx','ncc_dy','ncc_dx']} for b in range(nb) if b!=1},
             ds_panfilt_offset_HRpx={k:agg(proto_rows,k) for k in ['pcc_dy','pcc_dx','ncc_dy','ncc_dx']},
             per_scene_nnls=[(round(r['pcc_dy'],3),round(r['pcc_dx'],3)) for r in rows])
    print(json.dumps(res)); sys.stdout.flush(); return res
def rr(sensor, h5main, tag, nmax=None):
    rows=[]; band_rows=[]
    with h5py.File(h5main) as f:
        n=f['gt'].shape[0]; idx=np.arange(n) if (nmax is None or n<=nmax) else np.unique(np.linspace(0,n-1,nmax).astype(int))
        for i in idx:
            G=f['gt'][i].transpose(1,2,0).astype(np.float64); P=f['pan'][i][0].astype(np.float64)
            I,w=nnls_int(G,P); r=shift_est(P,I,maxs=3 if P.shape[0]>=128 else 2); r['scene']=int(i); rows.append(r)
            if P.shape[0]>=128:
                for b in range(G.shape[2]):
                    rb=shift_est(P,G[:,:,b]); rb.update(scene=int(i),band=b); band_rows.append(rb)
    res=dict(tag=tag, sensor=sensor, n=len(rows), file=h5main, nnls_vs_pan={k:agg(rows,k) for k in ['pcc_dy','pcc_dx','ncc_dy','ncc_dx','ncc_peak']},
             per_band=({b:{k:agg([r for r in band_rows if r['band']==b],k) for k in ['pcc_dy','pcc_dx']} for b in range(max([r['band'] for r in band_rows])+1)} if band_rows else None),
             per_scene_nnls=[(round(r['pcc_dy'],3),round(r['pcc_dx'],3)) for r in rows][:20])
    print(json.dumps(res)); sys.stdout.flush(); return res
out={}
out['GF2_FR']=fr('gf2', f'{ROOT}/GF2/full_examples_mat20/test_gf2_OrigScale_mat20.h5','GF2 FR mat20')
out['WV3_FR']=fr('wv3', f'{ROOT}/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5','WV3 FR mat20')
out['QB_FR']=fr('qb', f'{ROOT}/QB/full_examples_mat20/test_qb_OrigScale_mat20.h5','QB FR mat20')
out['GF2_RR']=rr('gf2', f'{ROOT}/GF2/reduced_examples_h5/test_gf2_multiExm1.h5','GF2 RR test')
out['WV3_RR']=rr('wv3', f'{ROOT}/WV3/reduced_examples_h5/test_wv3_multiExm1.h5','WV3 RR test')
out['QB_RR']=rr('qb', f'{ROOT}/QB/reduced_examples_h5/test_qb_multiExm1.h5','QB RR test')
out['GF2_train']=rr('gf2', f'{ROOT}/GF2/train_gf2.h5','GF2 train 400',400)
out['WV3_train']=rr('wv3', f'{ROOT}/WV3/train_wv3.h5','WV3 train 400',400)
out['QB_train']=rr('qb', f'{ROOT}/QB/train_qb_msfix.h5','QB train msfix 400',400)
json.dump(out,open(os.path.join(OUT,'reg_lrgrid.json'),'w'),indent=1); print('DONE')
