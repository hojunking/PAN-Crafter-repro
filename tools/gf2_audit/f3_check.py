"""F-3 style check: for each patch, ms ?= genMTF(sensor) replicate-filter(gt) decimated at phase (dy,dx).
Report phase distribution & MAD. CPU only. Same recipe as tools/repair_qb_ms.py."""
import os, sys, json, numpy as np, h5py
sys.path.insert(0, '/home/knuvi/Desktop/song/PAN-Crafter')
os.environ.setdefault('PANCRAFTER_DLPAN', '/home/knuvi/Desktop/song/DLPan-Toolbox')
from scipy.signal import fftconvolve
from tools.metrics.eval_fr import genmtf_matlab, GNYQ_TABLE
ROOT='/home/knuvi/Desktop/song/PAN-Crafter/data/PanCollection'
def lowpass(gt_chw, kernel):
    out=[]
    for b in range(gt_chw.shape[0]):
        k=kernel[:,:,b]; p=k.shape[0]//2
        out.append(fftconvolve(np.pad(gt_chw[b],p,mode='edge'), k[::-1,::-1], mode='valid'))
    return np.stack(out)
def check(path, sensor, gnyq, nsample, tag):
    kernel=genmtf_matlab(gnyq,4,41)
    with h5py.File(path) as f:
        n=f['gt'].shape[0]; idx=np.linspace(0,n-1,min(n,nsample)).astype(int); idx=np.unique(idx)
        phases={}; mads=[]; best_mads=[]; lms_mad=[]
        for i in idx:
            gt=np.asarray(f['gt'][i],dtype=np.float64); ms=np.asarray(f['ms'][i],dtype=np.float64)
            low=lowpass(gt,kernel)
            best=None
            for dy in range(4):
                for dx in range(4):
                    d=np.abs(low[:,dy::4,dx::4]-ms).mean()
                    if best is None or d<best[0]: best=(d,dy,dx)
            d22=np.abs(low[:,2::4,2::4]-ms).mean()
            mads.append(d22); best_mads.append(best[0]); phases[(best[1],best[2])]=phases.get((best[1],best[2]),0)+1
        tot=len(idx)
        res=dict(tag=tag, file=path, n_total=int(n), n_sampled=int(tot), gnyq=list(map(float,gnyq)),
                 mad_phase22_mean=float(np.mean(mads)), mad_phase22_max=float(np.max(mads)),
                 mad_bestphase_mean=float(np.mean(best_mads)), mad_bestphase_max=float(np.max(best_mads)),
                 phase_hist={f'{k[0]},{k[1]}':v for k,v in sorted(phases.items())},
                 frac_not_22=float(sum(v for k,v in phases.items() if k!=(2,2))/tot))
        print(json.dumps(res)); sys.stdout.flush(); return res
out=[]
N=int(sys.argv[1]) if len(sys.argv)>1 else 400
out.append(check(f'{ROOT}/GF2/train_gf2.h5','gf2',[0.3]*4,N,'GF2 train (GNyq 0.3 default)'))
out.append(check(f'{ROOT}/GF2/valid_gf2.h5','gf2',[0.3]*4,N,'GF2 valid (GNyq 0.3 default)'))
out.append(check(f'{ROOT}/GF2/reduced_examples_h5/test_gf2_multiExm1.h5','gf2',[0.3]*4,20,'GF2 test RR (GNyq 0.3 default)'))
out.append(check(f'{ROOT}/WV3/train_wv3.h5','wv3',GNYQ_TABLE['WV3'],N,'WV3 train control'))
out.append(check(f'{ROOT}/WV3/valid_wv3.h5','wv3',GNYQ_TABLE['WV3'],N,'WV3 valid control'))
out.append(check(f'{ROOT}/QB/train_qb.h5','qb',GNYQ_TABLE['QB'],200,'QB train ORIGINAL (F-3 positive control)'))
out.append(check(f'{ROOT}/QB/train_qb_msfix.h5','qb',GNYQ_TABLE['QB'],200,'QB train msfix (repaired control)'))
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'f3_check.json'),'w'),indent=1)
