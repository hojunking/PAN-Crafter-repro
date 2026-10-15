#!/usr/bin/env python
"""Controlled-shift benchmark (계획 §13.2–13.4) + FR audit 일치 + confidence calibration.

  python tools/uvs_controlled_shift.py --run work_dir/<run> [--ckpt best_hqnr]

RR test PAN 에 알려진 a (LR px; 0 / 0.25 / 0.5 / 1 / 2 / 3 × 8방향) 를 넣는다: P' = W(P, 4a) (PAN 3ch 강체).
  - 고유 shift 가 0 이라면 student 는 δ = −a 를 내야 한다 → shift MAE = |δ_S + a| (shift variant 만)
  - ERGAS/SCC 저하 곡선, corrected-PAN fSCC, zero-shift no-harm, 곡선 AUC(정규화)
FR 12-19: δ_S vs audit(−Δ) 일치.  confidence 5-bin vs MAE (val 표본에 합성 a).
결과 <run>/results/controlled_shift.csv (+ stdout 표).
"""
import argparse, csv, os, sys
import h5py, numpy as np, torch, torch.nn.functional as F, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from main import import_class                                      # noqa: E402
from train_uvs import UVSModel, build_inputs, x11, USES_SHIFT      # noqa: E402
from uvs.shift import ShiftModule, edge_rep, warp, warp_pan_channels, gated_delta   # noqa: E402
from utils import reduced_metrics, tensor2img, SCC_full_numpy      # noqa: E402

R = os.path.join(ROOT, "data/PanCollection/WV3")
DIRS = [(np.cos(t), np.sin(t)) for t in np.arange(8) * np.pi / 4]


def load(run, ckpt):
    cfg = yaml.safe_load(open(os.path.join(run, "meta", "config.yaml"))); u = cfg.get("uvs") or {}
    bb = import_class(cfg["model"])(**cfg["model_args"]); v = u["variant"]; sh = u.get("shift") or {}
    shift = ShiftModule(tuple(sh.get("student_channels", [8, 8])), int(sh.get("search_radius", 3)), float(sh.get("softmax_temperature", 0.07))) if v in USES_SHIFT else None
    m = UVSModel(bb, shift)
    from safetensors.torch import load_file
    m.load_state_dict(load_file(os.path.join(run, ckpt, "model.safetensors")), strict=True)
    return m.eval(), cfg, v, float(sh.get("confidence_threshold", 0.35)), sh.get("warp_mode", "bicubic")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="best_hqnr"); ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, cfg, v, thr, wm = load(a.run, a.ckpt); m = m.to(dev)
    with h5py.File(f"{R}/reduced_examples_h5/test_wv3_multiExm1.h5") as f:
        gt, lms, ms, pan = (torch.tensor(f[k][:a.n] / 1023.5 - 1, dtype=torch.float32) for k in ("gt", "lms", "ms", "pan"))
    with h5py.File(f"{R}/reduced_examples_h5/test_wv3_multiExm1_pan.h5") as f:
        lpan = torch.tensor(f["lpan"][:a.n] / 1023.5 - 1, dtype=torch.float32)
    rows = []
    def run_case(mag, dirn):
        errs, ergas, scc, fscc = [], [], [], []
        with torch.no_grad():
            for i in range(a.n):
                g, L, M, P, LP = (t[i:i + 1].to(dev) for t in (gt, lms, ms, pan, lpan))
                a_vec = torch.tensor([[mag * dirn[0], mag * dirn[1]]], device=dev)
                lpu, hf = build_inputs(P, LP, L)
                P2, LPU2, HF2 = warp_pan_channels(P, lpu, hf, a_vec, wm) if mag > 0 else (P, lpu, hf)
                LP2 = warp(LP, a_vec, 1.0, wm) if mag > 0 else LP
                d = None
                if m.shift is not None:
                    o = m.shift(edge_rep(LP2), edge_rep(M)); d = gated_delta(o["delta"], o["conf"], thr)
                    errs.append((o["delta"] + a_vec).norm().item())                    # 목표 δ = −a
                    P3, LPU3, HF3 = warp_pan_channels(P2, LPU2, HF2, d, wm)
                else:
                    P3, LPU3, HF3 = P2, LPU2, HF2
                y = L + m.backbone(None, None, None, torch.ones(1, device=dev), x_in=x11(P3, LPU3, HF3, L))
                r = reduced_metrics(x_true=g, x_pred=y, max_pixel=2047.0); ergas.append(r["ergas"]); scc.append(r["scc"])
                fscc.append(SCC_full_numpy(tensor2img(P, 2047.0), tensor2img(y, 2047.0)))   # 원 PAN 기준 fSCC
        return dict(mae=(np.mean(errs) if errs else float("nan")), ergas=np.mean(ergas), scc=np.mean(scc), fscc=np.mean(fscc))
    base = run_case(0.0, (0, 0)); rows.append(dict(mag=0.0, **base))
    print(f"a=0    ERGAS {base['ergas']:.4f} SCC {base['scc']:.5f} fSCC {base['fscc']:.4f}" + (f" shiftMAE {base['mae']:.3f}" if v in USES_SHIFT else ""))
    for mag in (0.25, 0.5, 1.0, 2.0, 3.0):
        rs = [run_case(mag, d) for d in DIRS]
        agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        rows.append(dict(mag=mag, **agg))
        print(f"a={mag:<4} ERGAS {agg['ergas']:.4f} ({100*(agg['ergas']-base['ergas'])/base['ergas']:+.1f}%) SCC {agg['scc']:.5f} fSCC {agg['fscc']:.4f}" + (f" shiftMAE {agg['mae']:.3f}" if v in USES_SHIFT else ""))
    mags = np.array([r["mag"] for r in rows]); e = np.array([r["ergas"] for r in rows])
    auc = float(np.trapz(e / e[0], mags) / mags[-1])                                     # 정규화 ERGAS 곡선 AUC (낮을수록 강건)
    print(f"ERGAS-degradation AUC(normalized) {auc:.4f}   zero-shift no-harm: 이 값은 run 의 표준 RR 과 같아야 한다")
    out = os.path.join(a.run, "results", "controlled_shift.csv"); os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows); w.writerow(dict(mag="auc", ergas=auc))
    # confidence calibration (val, 합성 a ~ U[-1,1]²) — shift variant 만
    if m.shift is not None:
        with h5py.File(f"{R}/valid_wv3.h5") as f: msv = torch.tensor(f["ms"][:512] / 1023.5 - 1, dtype=torch.float32)
        with h5py.File(f"{R}/valid_wv3_pan.h5") as f: lpv = torch.tensor(f["lpan"][:512] / 1023.5 - 1, dtype=torch.float32)
        g = torch.Generator().manual_seed(0); s = (torch.rand(512, 2, generator=g) * 2 - 1)
        with torch.no_grad():
            o0 = m.shift(edge_rep(lpv.to(dev)), edge_rep(msv.to(dev)))
            o = m.shift(edge_rep(warp(lpv.to(dev), -s.to(dev), 1.0, wm)), edge_rep(msv.to(dev)))
        err = (o["delta"] - (o0["delta"] + s.to(dev))).norm(dim=1).cpu().numpy(); c = o["conf"].cpu().numpy()
        print("confidence bin | mean conf | shift MAE | n")
        for lo_ in (0.0, 0.2, 0.4, 0.6, 0.8):
            sel = (c >= lo_) & (c < lo_ + 0.2 + (1e-9 if lo_ == 0.8 else 0))
            if sel.any(): print(f"  {lo_:.1f}–{lo_+0.2:.1f} | {c[sel].mean():.3f} | {err[sel].mean():.3f} | {int(sel.sum())}")
        # FR 12-19 vs audit
        import pandas as pd
        with h5py.File(f"{R}/full_examples_h5_repaired/test_wv3_OrigScale_multiExm1.h5") as f: msf = torch.tensor(f["ms"][12:20] / 1023.5 - 1, dtype=torch.float32)
        with h5py.File(f"{R}/full_examples_h5_repaired/test_wv3_OrigScale_multiExm1_pan.h5") as f: lpf = torch.tensor(f["lpan"][12:20] / 1023.5 - 1, dtype=torch.float32)
        tgt = -torch.tensor(pd.read_csv(os.path.join(ROOT, "outputs/global_shift_cache/wv3_fr.csv")).sort_values("sample_id")[["dy_lr_raw", "dx_lr_raw"]].values[12:20], dtype=torch.float32)
        with torch.no_grad(): of = m.shift(edge_rep(lpf.to(dev)), edge_rep(msf.to(dev)))
        D = of["delta"].cpu(); print(f"FR 12-19 δ_S vs audit(−Δ): medErr {(D - tgt).norm(dim=1).median():.3f} corr(dy,dx)=({np.corrcoef(D[:,0], tgt[:,0])[0,1]:+.2f},{np.corrcoef(D[:,1], tgt[:,1])[0,1]:+.2f}) conf {of['conf'].mean():.2f}")
    print("->", out)


if __name__ == "__main__":
    main()
