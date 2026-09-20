"""(1) effective GNyq: for each FR scene/band, find g maximizing shift-invariant NCC between genMTF(g)-lowpass(PAN)[2::4] and native ms band.
(2) EXP baseline (lms) D_lambda/D_s/HQNR on mat20 for gf2/wv3/qb with the repo evaluator (tools.metrics.eval_fr). CPU."""
import os, sys, json, numpy as np, h5py
sys.path.insert(0, '/home/knuvi/Desktop/song/PAN-Crafter')
os.environ.setdefault('PANCRAFTER_DLPAN', '/home/knuvi/Desktop/song/DLPan-Toolbox')
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from tools.metrics.eval_fr import genmtf_matlab, GNYQ_TABLE, load_dlpan, d_lambda_k, d_s
wald=load_dlpan(os.environ['PANCRAFTER_DLPAN']); ROOT='/home/knuvi/Desktop/song/PAN-Crafter/data/PanCollection'; OUT=os.path.dirname(os.path.abspath(__file__))
def filt(img,k): p=k.shape[0]//2; return fftconvolve(np.pad(img,p,mode='edge'), k[::-1,::-1], mode='valid')
def hp(x,s=3.0): x=x.astype(np.float64); return x-gaussian_filter(x,s)
def ncc_max(ref, mov, maxs=2):
    a=hp(ref); b=hp(mov); a=(a-a.mean())/(a.std()+1e-12); b=(b-b.mean())/(b.std()+1e-12); H,W=a.shape; m=6; A=a[m:-m,m:-m]; best=-9
    for dy in range(-maxs,maxs+1):
        for dx in range(-maxs,maxs+1):
            B=b[m+dy:H-m+dy, m+dx:W-m+dx]; best=max(best,float((A*B).mean()))
    return best
GR=[0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80]
KER={g:genmtf_matlab([g],4,41)[:,:,0] for g in GR}
res={}
for sensor in ['gf2','wv3','qb']:
    h5=f'{ROOT}/{sensor.upper()}/full_examples_mat20/test_{sensor}_OrigScale_mat20.h5'
    with h5py.File(h5) as f: ms=f['ms'][:]; pan=f['pan'][:][:,0]; lms=f['lms'][:]
    nb=ms.shape[1]; best_g=np.zeros((pan.shape[0],nb)); score_at={g:np.zeros((pan.shape[0],nb)) for g in GR}
    for i in range(pan.shape[0]):
        P=pan[i].astype(np.float64); PL={g:filt(P,KER[g])[2::4,2::4] for g in GR}
        for b in range(nb):
            M=ms[i,b].astype(np.float64); sc={g:ncc_max(PL[g],M) for g in GR}
            for g in GR: score_at[g][i,b]=sc[g]
            best_g[i,b]=max(sc,key=sc.get)
    # EXP baseline
    dl=[];ds=[]
    for i in range(pan.shape[0]):
        L=lms[i].transpose(1,2,0).astype(np.float64); P=pan[i].astype(np.float64)
        dl.append(d_lambda_k(L,L,sensor,4,32,wald)); ds.append(d_s(L,L,P,4,32,wald))
    dl=np.array(dl); ds=np.array(ds); hq=(1-dl)*(1-ds)
    res[sensor]=dict(assumed_gnyq=(GNYQ_TABLE.get(sensor.upper()) or [0.3]*nb), eff_gnyq_median_per_band=[float(np.median(best_g[:,b])) for b in range(nb)],
                     eff_gnyq_mean_per_band=[float(best_g[:,b].mean()) for b in range(nb)], eff_gnyq_hist={str(g):int((best_g==g).sum()) for g in GR},
                     mean_score_by_g={str(g):float(score_at[g].mean()) for g in GR},
                     EXP=dict(d_lambda=float(dl.mean()), d_lambda_sd=float(dl.std(ddof=1)), d_s=float(ds.mean()), d_s_sd=float(ds.std(ddof=1)), hqnr=float(hq.mean()), hqnr_sd=float(hq.std(ddof=1)), per_scene_ds=[round(float(x),4) for x in ds]))
    print(sensor, json.dumps(res[sensor])); sys.stdout.flush()
json.dump(res,open(os.path.join(OUT,'eff_gnyq_exp.json'),'w'),indent=1); print('DONE')
