"""GF2 FR metric sensitivity sweep (CPU only). Re-evaluates stored outputs under alternative metric constants."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = "/home/knuvi/Desktop/song/PAN-Crafter"
sys.path.insert(0, ROOT)
import numpy as np, h5py
from scipy.io import loadmat
import tools.metrics.eval_fr as EF
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s
wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
OUT = os.path.dirname(os.path.abspath(__file__))
h5 = f"{ROOT}/data/PanCollection/GF2/full_examples_mat20/test_gf2_OrigScale_mat20.h5"
with h5py.File(h5) as f:
    lms = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)
    pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]
srcs = {
 "B01_GF2_P0 best_hqnr mat20": f"{ROOT}/work_dir/B01_GF2_P0_W112_D123_S2025_BOOTSCRATCH/results/full_best_hqnr_mat20.mat",
 "ARCH_W168_D123_DUAL_GF2_S1234 best_hqnr mat20": f"{ROOT}/work_dir/ARCH_W168_D123_DUAL_GF2_S1234/results/full_best_hqnr_mat20.mat",
 "CANConv released cannet_gf2 mat20": "/home/knuvi/Desktop/song/CANConv/data/datasets/gf2/sr_cannet_mat20.h5",
}
def load(p):
    if p.endswith(".h5"):
        with h5py.File(p) as f: return np.asarray(f["sr"], dtype=np.float64).transpose(0, 2, 3, 1)
    return loadmat(p)["sr"].astype(np.float64).transpose(0, 2, 3, 1)
def run(sr, gnyq=None, clip=None, rnd=False, pan_in=None, lms_in=None):
    old = dict(EF.GNYQ_OVERRIDE)
    if gnyq is not None: EF.GNYQ_OVERRIDE["gf2"] = gnyq
    else: EF.GNYQ_OVERRIDE.pop("gf2", None)
    P = pan if pan_in is None else pan_in; L = lms if lms_in is None else lms_in
    dl, dsv = [], []
    for i in range(len(sr)):
        s = sr[i]
        if rnd: s = np.round(s)
        if clip is not None: s = np.clip(s, 0.0, float(clip))
        dl.append(d_lambda_k(s, L[i], "gf2", 4, 32, wald)); dsv.append(d_s(s, L[i], P[i], 4, 32, wald))
    EF.GNYQ_OVERRIDE.clear(); EF.GNYQ_OVERRIDE.update(old)
    dl, dsv = np.array(dl), np.array(dsv); h = (1 - dl) * (1 - dsv)
    return dict(hqnr=float(h.mean()), hqnr_sd=float(h.std(ddof=1)), d_lambda=float(dl.mean()), d_s=float(dsv.mean()), per_scene_hqnr=[round(float(x), 6) for x in h])
settings = [
 ("baseline (no clip, GNyq 0.3 = genMTF otherwise)", dict()),
 ("clip [0,2^10=1024] (th_values=1, L=10)", dict(clip=1024)),
 ("clip [0,2^11=2048] (L=11 as if WV3)", dict(clip=2048)),
 ("clip [0,1023]", dict(clip=1023)),
 ("round to integer DN", dict(rnd=True)),
 ("GNyq 0.20", dict(gnyq=0.20)),
 ("GNyq 0.25", dict(gnyq=0.25)),
 ("GNyq 0.35", dict(gnyq=0.35)),
 ("GNyq 0.40", dict(gnyq=0.40)),
 ("GNyq QB table [0.34,0.32,0.30,0.22]", dict(gnyq=[0.34,0.32,0.30,0.22])),
 ("GNyq GeoEye1 0.23", dict(gnyq=0.23)),
]
res = {}
for name, p in srcs.items():
    sr = load(p); res[name] = {}
    print(f"== {name}  sr range [{sr.min():.1f},{sr.max():.1f}]  frac>1023 {(sr>1023).mean():.2e}", flush=True)
    for sname, kw in settings:
        t = time.time(); r = run(sr, **kw); res[name][sname] = r
        print(f"  {sname:48s} HQNR {r['hqnr']:.4f}±{r['hqnr_sd']:.4f}  D_l {r['d_lambda']:.4f}  D_s {r['d_s']:.4f}  ({time.time()-t:.0f}s)", flush=True)
    json.dump(res, open(f"{OUT}/sweep_results.json", "w"), indent=1)
# EXP baseline (lms itself as fused) for reference
r = run(lms); res["EXP (lms as fused)"] = {"baseline": r}
print(f"== EXP(lms)  HQNR {r['hqnr']:.4f}±{r['hqnr_sd']:.4f}  D_l {r['d_lambda']:.4f}  D_s {r['d_s']:.4f}", flush=True)
json.dump(res, open(f"{OUT}/sweep_results.json", "w"), indent=1)
print("DONE", flush=True)
