"""F-1 lpan check + PAN-vs-MS registration at full res and reduced res. CPU only.
(a) lpan vs F-1 recipe regen (Gaussian sigma 1.98, N 41, replicate, [2::4,2::4]) and vs genMTF-PAN decimation: corr/RMSE per scene.
(b) registration: shift of PAN low-pass reference (imresize 1/4 -> interp23tap, = d_s pan_filt) vs upsampled-MS intensity,
    by (i) FFT normalized cross-correlation of high-passed images with parabolic sub-pixel fit, (ii) skimage phase_cross_correlation upsample 50.
"""
import os, sys, json, numpy as np, h5py
sys.path.insert(0, '/home/knuvi/Desktop/song/PAN-Crafter')
os.environ.setdefault('PANCRAFTER_DLPAN', '/home/knuvi/Desktop/song/DLPan-Toolbox')
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
from tools.metrics.eval_fr import genmtf_matlab, GNYQ_TABLE, load_dlpan, imresize_matlab
from skimage.registration import phase_cross_correlation
wald = load_dlpan(os.environ['PANCRAFTER_DLPAN'])
ROOT='/home/knuvi/Desktop/song/PAN-Crafter/data/PanCollection'
OUT=os.path.dirname(os.path.abspath(__file__))
MTF_PAN={'qb':0.15,'ikonos':0.17,'geoeye1':0.16,'wv2':0.11,'wv3':0.14,'gf2':0.15}  # wald_utilities.MTF_PAN; else 0.15

def gauss_kernel(n=41, sigma=1.98):
    m=(n-1)/2; y,x=np.ogrid[-m:m+1,-m:m+1]; h=np.exp(-(x*x+y*y)/(2*sigma*sigma)); return h/h.sum()
def filt(img, k):
    p=k.shape[0]//2; return fftconvolve(np.pad(img,p,mode='edge'), k[::-1,::-1], mode='valid')
def corr(a,b): return float(np.corrcoef(a.ravel(),b.ravel())[0,1])

def hp(img, s=6.0):
    x=img.astype(np.float64); return x-gaussian_filter(x, s)
def ncc_shift(ref, mov, maxs=4):
    """shift (dy,dx) s.t. mov(y+dy, x+dx) ~ ref; integer peak within +-maxs, parabolic subpixel."""
    a=hp(ref); b=hp(mov); a=(a-a.mean())/(a.std()+1e-12); b=(b-b.mean())/(b.std()+1e-12)
    H,W=a.shape
    # cross-correlation via FFT (circular; interior dominates after hp)
    F=np.fft.fft2(a); G=np.fft.fft2(b); cc=np.real(np.fft.ifft2(F*np.conj(G))); cc=np.fft.fftshift(cc)
    cy,cx=H//2,W//2; win=cc[cy-maxs:cy+maxs+1, cx-maxs:cx+maxs+1]
    iy,ix=np.unravel_index(np.argmax(win), win.shape)
    dy,dx=iy-maxs, ix-maxs
    def para(m1,c0,p1):
        d=(m1-2*c0+p1); return 0.0 if d==0 else 0.5*(m1-p1)/d
    sy=para(win[iy-1,ix],win[iy,ix],win[iy+1,ix]) if 0<iy<win.shape[0]-1 else 0.0
    sx=para(win[iy,ix-1],win[iy,ix],win[iy,ix+1]) if 0<ix<win.shape[1]-1 else 0.0
    return float(dy+sy), float(dx+sx), float(win[iy,ix]/(H*W))
def pcc_shift(ref, mov):
    s,_,_=phase_cross_correlation(hp(ref), hp(mov), upsample_factor=50, normalization='phase'); return float(s[0]), float(s[1])

def nnls_intensity(lms_hwc, target_hw):
    A=lms_hwc.reshape(-1,lms_hwc.shape[2]); w,_=nnls(A, target_hw.ravel()); return (A@w).reshape(target_hw.shape), w

def full_res(sensor, h5main, h5pan, tag):
    res=[]
    with h5py.File(h5main) as f:
        lms=f['lms'][:]; ms=f['ms'][:]; pan=f['pan'][:][:,0]
    lp=None
    if h5pan and os.path.exists(h5pan):
        with h5py.File(h5pan) as f: lp=f['lpan'][:][:,0]
    gk=gauss_kernel(); mk=genmtf_matlab([MTF_PAN.get(sensor,0.15)],4,41)[:,:,0]
    for i in range(pan.shape[0]):
        P=pan[i].astype(np.float64); L=lms[i].transpose(1,2,0).astype(np.float64)
        r=dict(scene=i)
        if lp is not None:
            lp_f1=filt(P,gk)[2::4,2::4]; lp_mtf=filt(P,mk)[2::4,2::4]
            r.update(lpan_corr_f1=corr(lp[i],lp_f1), lpan_rmse_f1=float(np.sqrt(np.mean((lp[i]-lp_f1)**2))),
                     lpan_corr_mtfpan=corr(lp[i],lp_mtf), lpan_rmse_mtfpan=float(np.sqrt(np.mean((lp[i]-lp_mtf)**2))))
        # PAN low-pass reference as in d_s
        pl=imresize_matlab(P,0.25); pf=wald.interp23tap(pl[:,:,None],4)[:,:,0]
        I_mean=L.mean(axis=2); I_nnls,w=nnls_intensity(L, pf)
        dy,dx,pk=ncc_shift(pf, I_mean); dy2,dx2=pcc_shift(pf, I_mean)
        dy3,dx3,pk3=ncc_shift(pf, I_nnls); dy4,dx4=pcc_shift(pf, I_nnls)
        # also lms band-wise vs pf
        r.update(reg_ncc_dy=dy, reg_ncc_dx=dx, reg_ncc_peak=pk, reg_pcc_dy=dy2, reg_pcc_dx=dx2,
                 reg_ncc_nnls_dy=dy3, reg_ncc_nnls_dx=dx3, reg_pcc_nnls_dy=dy4, reg_pcc_nnls_dx=dx4,
                 nnls_w=[float(x) for x in w], corr_pf_Imean=corr(pf,I_mean), corr_pf_Innls=corr(pf,I_nnls),
                 # ms vs lms consistency: lms == interp23tap(ms)?
                 lms_vs_interp23_mad=float(np.abs(L-wald.interp23tap(ms[i].transpose(1,2,0).astype(np.float64),4)).mean()))
        res.append(r)
    def agg(k): v=np.array([x[k] for x in res if k in x]); return dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()), absmean=float(np.abs(v).mean())) if v.size else None
    summ=dict(tag=tag, sensor=sensor, n=len(res), file=h5main, agg={k:agg(k) for k in ['lpan_corr_f1','lpan_rmse_f1','lpan_corr_mtfpan','lpan_rmse_mtfpan','reg_ncc_dy','reg_ncc_dx','reg_pcc_dy','reg_pcc_dx','reg_ncc_nnls_dy','reg_ncc_nnls_dx','reg_pcc_nnls_dy','reg_pcc_nnls_dx','corr_pf_Imean','corr_pf_Innls','lms_vs_interp23_mad']}, scenes=res)
    print(json.dumps({k:v for k,v in summ.items() if k!='scenes'})); sys.stdout.flush(); return summ

def reduced_res(sensor, h5main, tag, nmax=None):
    """RR: gt (HR MS) vs pan: registration of pan low-pass ref vs gt intensity; also gt vs lms (should be 0 by construction)."""
    res=[]
    with h5py.File(h5main) as f:
        n=f['gt'].shape[0]; idx=np.arange(n) if (nmax is None or n<=nmax) else np.unique(np.linspace(0,n-1,nmax).astype(int))
        for i in idx:
            G=f['gt'][i].transpose(1,2,0).astype(np.float64); P=f['pan'][i][0].astype(np.float64)
            I_nnls,w=nnls_intensity(G,P); I_mean=G.mean(axis=2)
            if P.shape[0]>=128:
                dy,dx,pk=ncc_shift(P,I_nnls); dy2,dx2=pcc_shift(P,I_nnls)
            else:
                dy,dx,pk=ncc_shift(P,I_nnls,maxs=3); dy2,dx2=pcc_shift(P,I_nnls)
            res.append(dict(scene=int(i), reg_ncc_dy=dy, reg_ncc_dx=dx, reg_pcc_dy=dy2, reg_pcc_dx=dx2, corr_P_Innls=corr(P,I_nnls), corr_P_Imean=corr(P,I_mean), nnls_w=[float(x) for x in w]))
    def agg(k): v=np.array([x[k] for x in res]); return dict(mean=float(v.mean()), median=float(np.median(v)), sd=float(v.std(ddof=1)), absmean=float(np.abs(v).mean()), min=float(v.min()), max=float(v.max()))
    summ=dict(tag=tag, sensor=sensor, n=len(res), file=h5main, agg={k:agg(k) for k in ['reg_ncc_dy','reg_ncc_dx','reg_pcc_dy','reg_pcc_dx','corr_P_Innls','corr_P_Imean']}, scenes=res)
    print(json.dumps({k:v for k,v in summ.items() if k!='scenes'})); sys.stdout.flush(); return summ

out={}
out['GF2_FR_mat20']=full_res('gf2', f'{ROOT}/GF2/full_examples_mat20/test_gf2_OrigScale_mat20.h5', f'{ROOT}/GF2/full_examples_mat20/test_gf2_OrigScale_mat20_pan.h5','GF2 FR mat20 (paper set)')
out['GF2_FR_h5']=full_res('gf2', f'{ROOT}/GF2/full_examples_h5/test_gf2_OrigScale_multiExm1.h5', f'{ROOT}/GF2/full_examples_h5/test_gf2_OrigScale_multiExm1_pan.h5','GF2 FR h5 (released, author lpan)')
out['WV3_FR_mat20']=full_res('wv3', f'{ROOT}/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5', f'{ROOT}/WV3/full_examples_mat20/test_wv3_OrigScale_mat20_pan.h5','WV3 FR mat20 (paper set)')
out['QB_FR_mat20']=full_res('qb', f'{ROOT}/QB/full_examples_mat20/test_qb_OrigScale_mat20.h5', f'{ROOT}/QB/full_examples_mat20/test_qb_OrigScale_mat20_pan.h5','QB FR mat20 (paper set)')
out['GF2_RR_test']=reduced_res('gf2', f'{ROOT}/GF2/reduced_examples_h5/test_gf2_multiExm1.h5','GF2 RR test 20')
out['WV3_RR_test']=reduced_res('wv3', f'{ROOT}/WV3/reduced_examples_h5/test_wv3_multiExm1.h5','WV3 RR test 20')
out['QB_RR_test']=reduced_res('qb', f'{ROOT}/QB/reduced_examples_h5/test_qb_multiExm1.h5','QB RR test 20')
out['GF2_train']=reduced_res('gf2', f'{ROOT}/GF2/train_gf2.h5','GF2 train (400 sampled, 64px)',400)
out['WV3_train']=reduced_res('wv3', f'{ROOT}/WV3/train_wv3.h5','WV3 train (400 sampled, 64px)',400)
out['QB_train_msfix']=reduced_res('qb', f'{ROOT}/QB/train_qb_msfix.h5','QB train msfix (400 sampled, 64px)',400)
json.dump(out, open(os.path.join(OUT,'f1_reg_check.json'),'w'), indent=1)
print('DONE')
