#!/usr/bin/env python
"""Shift-robust 추론 진단 (계획 §6 필수 기록 · §11.6 · §22).

  python tools/sr_infer_diag.py --run work_dir/SR_J1_C2RAND_BOTH_R050_W168_D123_DUAL [--ckpt best_hqnr]

J 계열: 추론 jitter 0 / (+0.5,+0.5) / (−0.5,−0.5) / (+0.5,−0.5) / U(−0.5,0.5) 무작위(seed) 에서 공식 HQNR·fSCC(12-19).
G1  : g1_scale(β) 0 / 0.5 / 1.0, wrong-sign(−Δ̂), confidence mean/P10/P90, center·boundary 확률, 예측 Δ̂ vs audit(−4Δ_LR).
결과 <run>/results/sr_diag.csv + stdout 표.
"""
import argparse, csv, os, sys
import numpy as np, torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from main import import_class                                    # noqa: E402
from train_sr import SRModel                                     # noqa: E402
from sr.forward import sr_infer                                  # noqa: E402
from sr.pan_align import GlobalCorrelator                        # noqa: E402
from feeders.feeder import PanFeeder                             # noqa: E402
from utils import tensor2img, SCC_full_numpy                     # noqa: E402
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s    # noqa: E402


def load(run, ckpt):
    cfg = yaml.safe_load(open(os.path.join(run, "meta", "config.yaml")))
    Model = import_class(cfg["model"]); bb = Model(**cfg["model_args"])
    sr = cfg.get("sr") or {}; v = sr["variant"]; g1 = sr.get("g1") or {}
    corr = GlobalCorrelator(bb.input.out_channels, int(g1.get("desc_channels", 16)), float(g1.get("radius_hr_px", 1.0)),
                            int(g1.get("n_per_axis", 5)), float(g1.get("tau", 0.07)), float(g1.get("gate_c0", 0.30))) if v == "g1" else None
    m = SRModel(bb, corr)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(run, ckpt, "model.safetensors"))
    m.load_state_dict(sd, strict=True)
    return m.eval(), cfg, v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="best_hqnr")
    ap.add_argument("--indices", default="12-19"); ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, cfg, v = load(a.run, a.ckpt); m = m.to(dev)
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox"))
    lo, hi = (int(x) for x in a.indices.split("-")); maxp = float(cfg.get("max_pixel", 2047.0))
    fr = PanFeeder(**cfg["test_full_feeder_args"])
    audit = None
    p = os.path.join(ROOT, "outputs/global_shift_cache/wv3_fr.csv")
    if os.path.exists(p):
        import pandas as pd
        df = pd.read_csv(p).sort_values("sample_id"); audit = torch.tensor(df[["dy_lr_raw", "dx_lr_raw"]].values, dtype=torch.float32)
    dn = lambda t: np.round((((t + 1) / 2).clamp(0, 1).float().cpu().numpy().astype(np.float64)) * maxp)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    if v == "g1":
        cases = [("beta0", dict(g1_scale=0.0)), ("beta0.5", dict(g1_scale=0.5)), ("beta1", dict(g1_scale=1.0)), ("wrong_sign", dict(g1_scale=1.0, wrong_sign=True))]
    else:
        cases = [("jit0", dict(eps=None)), ("jit+0.5+0.5", dict(eps=(0.5, 0.5))), ("jit-0.5-0.5", dict(eps=(-0.5, -0.5))),
                 ("jit+0.5-0.5", dict(eps=(0.5, -0.5))), ("jit_rand", dict(eps="rand"))]
    rows = []
    for name, kw in cases:
        dl, ds, fs, deltas, confs, pb, pc = [], [], [], [], [], [], []
        with torch.no_grad():
            for i in range(len(fr)):
                lms, ms, lpan, pan = (t[None].to(dev) for t in fr[i])
                kk = dict(kw)
                if "eps" in kk:
                    e = kk.pop("eps")
                    if e == "rand":
                        kk["eps"] = (torch.rand(1, 2, device=dev, generator=g) * 2 - 1) * 0.5
                    elif e is not None:
                        kk["eps"] = torch.tensor([e], device=dev)
                o = sr_infer(m, v, pan, lpan, ms, **kk)
                if not (lo <= i <= hi):
                    continue
                y = o["y"]; sr_np, lm, pn = dn(y[0]).transpose(1, 2, 0), dn(lms[0]).transpose(1, 2, 0), dn(pan[0, 0])
                dl.append(d_lambda_k(sr_np, lm, "wv3", 4, 32, wald)); ds.append(d_s(sr_np, lm, pn, 4, 32, wald))
                fs.append(SCC_full_numpy(tensor2img(pan, maxp), tensor2img(y, maxp)))
                if o["info"] is not None:
                    deltas.append(o["info"]["delta"][0].float().cpu()); confs.append(o["info"]["conf"][0].item())
                    pb.append(o["info"]["p_boundary"][0].item()); pc.append(o["info"]["p_center"][0].item())
        row = dict(case=name, hqnr=float(np.mean((1 - np.array(dl)) * (1 - np.array(ds)))), d_lambda=float(np.mean(dl)),
                   d_s=float(np.mean(ds)), fscc=float(np.mean(fs)))
        if deltas:
            D = torch.stack(deltas); row.update(delta_mag=float(D.norm(dim=1).mean()), conf_mean=float(np.mean(confs)),
                                                conf_p10=float(np.quantile(confs, .1)), conf_p90=float(np.quantile(confs, .9)),
                                                p_boundary=float(np.mean(pb)), p_center=float(np.mean(pc)))
            if audit is not None:
                tgt = -4.0 * audit[lo:hi + 1]
                row.update(corr_dy=float(np.corrcoef(D[:, 0], tgt[:, 0])[0, 1]), corr_dx=float(np.corrcoef(D[:, 1], tgt[:, 1])[0, 1]),
                           med_err_vs_audit=float((D - tgt).norm(dim=1).median()))
        rows.append(row)
        print(f"{name:14s} HQNR {row['hqnr']:.4f}  D_l {row['d_lambda']:.4f} D_s {row['d_s']:.4f}  fSCC {row['fscc']:.5f}"
              + (f"  |Δ̂| {row['delta_mag']:.3f} conf {row['conf_mean']:.3f} pB {row['p_boundary']:.3f} pC {row['p_center']:.3f}" if deltas else "")
              + (f"  corr(dy,dx)=({row['corr_dy']:+.2f},{row['corr_dx']:+.2f}) medErr {row['med_err_vs_audit']:.3f}" if "corr_dy" in row else ""))
    out = os.path.join(a.run, "results", "sr_diag.csv"); os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case", k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("->", out)


if __name__ == "__main__":
    main()
