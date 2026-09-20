"""EXP(=lms) FR anchor with the repo evaluator (CPU only). Sensors: gf2 / wv3 / qb / wv2 on full_examples_mat20.
Also GF2 GNyq sensitivity (0.3 default vs alternatives) and clip on/off."""
import os, sys, json, time
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np, h5py
ROOT = "/home/knuvi/Desktop/song/PAN-Crafter"
sys.path.insert(0, ROOT)
from tools.metrics import eval_fr as E
wald = E.load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
out = {}
def run(sensor, gnyq_override=None, tag=None, clip=True):
    h5 = f"{ROOT}/data/PanCollection/{sensor.upper()}/full_examples_mat20/test_{sensor}_OrigScale_mat20.h5"
    with h5py.File(h5, "r") as f:
        lms = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)
        pan = np.asarray(f["pan"], dtype=np.float64)[:, 0]
        ms = np.asarray(f["ms"], dtype=np.float64)
        shapes = {k: f[k].shape for k in f.keys()}
    R = 1023.0 if sensor == "gf2" else 2047.0
    L = 10 if sensor == "gf2" else 11
    # optional GNyq override: monkeypatch
    old = E.GNYQ_OVERRIDE.copy()
    if gnyq_override is not None:
        E.GNYQ_OVERRIDE[sensor] = gnyq_override
    rows = []
    for i in range(len(lms)):
        fused = lms[i]
        if clip: fused = np.clip(fused, 0.0, float(2 ** L))
        dl = E.d_lambda_k(fused, lms[i], sensor, 4, 32, wald)
        ds = E.d_s(fused, lms[i], pan[i], 4, 32, wald)
        rows.append((dl, ds, (1 - dl) * (1 - ds)))
    E.GNYQ_OVERRIDE.clear(); E.GNYQ_OVERRIDE.update(old)
    a = np.array(rows)
    key = tag or f"{sensor}"
    out[key] = dict(sensor=sensor, n=int(len(a)), shapes={k: list(v) for k, v in shapes.items()},
                    lms_max=float(lms.max()), pan_max=float(pan.max()), ms_max=float(ms.max()), lms_min=float(lms.min()),
                    d_lambda=float(a[:, 0].mean()), d_lambda_sd=float(a[:, 0].std(ddof=1)),
                    d_s=float(a[:, 1].mean()), d_s_sd=float(a[:, 1].std(ddof=1)),
                    hqnr=float(a[:, 2].mean()), hqnr_sd=float(a[:, 2].std(ddof=1)),
                    per_scene=[[round(float(x), 5) for x in r] for r in rows], gnyq_override=gnyq_override, clip=clip)
    print(f"{key:28s} N={len(a)}  D_l {a[:,0].mean():.4f}±{a[:,0].std(ddof=1):.4f}  D_s {a[:,1].mean():.4f}±{a[:,1].std(ddof=1):.4f}  HQNR {a[:,2].mean():.4f}±{a[:,2].std(ddof=1):.4f}   lms/pan/ms max {lms.max():.0f}/{pan.max():.0f}/{ms.max():.0f}", flush=True)
t0 = time.time()
for s in ("gf2", "wv3", "qb", "wv2"):
    if os.path.exists(f"{ROOT}/data/PanCollection/{s.upper()}/full_examples_mat20/test_{s}_OrigScale_mat20.h5"):
        run(s)
# GF2 GNyq sensitivity (D_lambda only depends on MTF; D_s does not)
for g in (0.20, 0.25, 0.28, 0.32, 0.35):
    run("gf2", gnyq_override=g, tag=f"gf2_gnyq{g:.2f}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_anchor.json"), "w"), indent=1)
print("done", time.time() - t0)
