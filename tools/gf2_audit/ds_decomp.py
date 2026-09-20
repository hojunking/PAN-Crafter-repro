"""CPU-only: decompose D_s per band (q_high=UQI(fused_b,PAN) vs q_low=UQI(lms_b,PAN_lp)) from stored full_best_hqnr_mat20.mat."""
import os, sys, json, numpy as np, h5py, scipy.io as sio
os.environ.setdefault('PANCRAFTER_DLPAN','/home/knuvi/Desktop/song/DLPan-Toolbox')
sys.path.insert(0,'.')
from tools.metrics.eval_fr import load_dlpan, d_s, d_lambda_k, _blockproc_uqi, imresize_matlab, mtf_filter, q2n
wald=load_dlpan(os.environ['PANCRAFTER_DLPAN'])
H5={'gf2':'data/PanCollection/GF2/full_examples_mat20/test_gf2_OrigScale_mat20.h5','qb':'data/PanCollection/QB/full_examples_mat20/test_qb_OrigScale_mat20.h5','wv3':'data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5'}
runs=[('gf2','B01_GF2_P0_W112_D123_S2025_BOOTSCRATCH'),('gf2','ARCH_W168_D123_DUAL_GF2_S2025'),('gf2','B02_GF2_DONN2_W112_D123_R200_S2025_BOOTSCRATCH'),('gf2','B03_GF2_L1E4_W112_D123_S2025_DONB02'),
      ('qb','B01_QB_P0_W112_D123_S2025_BOOTSCRATCH'),('qb','ARCH_W168_D123_DUAL_QB_S2025'),('wv3','PALS24_CTRLP0_W112_D123_WV3_S2025_N2LAST_R200_v1'),('wv3','S1_T05_W168_D123_DUAL')]
cache={}
def load(sensor):
    if sensor in cache: return cache[sensor]
    with h5py.File(H5[sensor]) as f:
        lms=np.asarray(f['lms'],dtype=np.float64).transpose(0,2,3,1); pan=np.asarray(f['pan'],dtype=np.float64)[:,0]
    pf=[wald.interp23tap(imresize_matlab(p,0.25)[:,:,None],4)[:,:,0] for p in pan]
    cache[sensor]=(lms,pan,pf); return cache[sensor]
res={}
for sensor,run in runs:
    mp=f'work_dir/{run}/results/full_best_hqnr_mat20.mat'
    if not os.path.exists(mp): print('MISSING',mp); continue
    sr=sio.loadmat(mp)['sr'].astype(np.float64)
    lms,pan,pf=load(sensor)
    N,C=sr.shape[0],sr.shape[1]
    qh=np.zeros((N,C)); ql=np.zeros((N,C)); dsv=np.zeros(N); dl=np.zeros(N)
    for i in range(N):
        s=sr[i].transpose(1,2,0)
        for b in range(C):
            qh[i,b]=_blockproc_uqi(s[:,:,b],pan[i],32); ql[i,b]=_blockproc_uqi(lms[i][:,:,b],pf[i],32)
        dsv[i]=np.mean(np.abs(qh[i]-ql[i])); dl[i]=d_lambda_k(s,lms[i],sensor,4,32,wald)
    hq=((1-dl)*(1-dsv)).mean()
    res[run]=dict(sensor=sensor,hqnr=hq,d_lambda=dl.mean(),d_s=dsv.mean(),qh_band=qh.mean(0).round(4).tolist(),ql_band=ql.mean(0).round(4).tolist(),
                  signed_diff_band=(qh-ql).mean(0).round(4).tolist(),frac_scenes_qh_gt_ql=float(((qh-ql)>0).mean()),per_scene_ds=dsv.round(4).tolist(),per_scene_dl=dl.round(4).tolist())
    print(f'== {run} [{sensor}] HQNR {hq:.4f} Dl {dl.mean():.4f} Ds {dsv.mean():.4f}')
    print(f'   q_high(fused_b,PAN) per band : {res[run]["qh_band"]}')
    print(f'   q_low (lms_b,PAN_lp) per band: {res[run]["ql_band"]}')
    print(f'   signed (q_high-q_low) per band: {res[run]["signed_diff_band"]}   frac(q_high>q_low)={res[run]["frac_scenes_qh_gt_ql"]:.2f}')
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ds_decomp.json'),'w'),indent=1)
# paired per-scene diffs on GF2
a=res.get('B01_GF2_P0_W112_D123_S2025_BOOTSCRATCH'); b=res.get('ARCH_W168_D123_DUAL_GF2_S2025')
if a and b:
    d=np.array(a['per_scene_ds'])-np.array(b['per_scene_ds']); print('GF2 per-scene Ds(P0)-Ds(DUAL): mean %.4f min %.4f max %.4f n_pos %d/20'%(d.mean(),d.min(),d.max(),(d>0).sum()))
    d=np.array(a['per_scene_dl'])-np.array(b['per_scene_dl']); print('GF2 per-scene Dl(P0)-Dl(DUAL): mean %.4f min %.4f max %.4f n_pos %d/20'%(d.mean(),d.min(),d.max(),(d>0).sum()))
