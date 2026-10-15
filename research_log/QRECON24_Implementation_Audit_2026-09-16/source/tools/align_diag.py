#!/usr/bin/env python
"""Global alignment 진단 — 계획 §20 round-trip interpolation control + §19 시각 패널.

  python tools/align_diag.py                       # §20 (FR·RR, cache Δ)
  python tools/align_diag.py --panels --run work_dir/GA_C3_...   # §19 패널 PNG (ASCII 라벨)

§20: m0 = up(ms,0), mp = up(ms,Δ,α=1), mr = warp_hr(mp,-4Δ). PSNR(mr,m0)·SCC(mr,m0)·gradient-energy
비·MAD 를 기록한다. case 성능 변화가 이 크기와 같다면 network 이득이 아니라 보간 손실이 지배한다.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from align.resample import upsample_shift, warp_hr, interp23tap, border_mask     # noqa: E402
from align.cache import ShiftCache                                                # noqa: E402
from align.shiftnet import scharr_mag_t                                           # noqa: E402
from feeders.feeder_align import PanFeederAlign                                    # noqa: E402

R = os.path.join(ROOT, "data/PanCollection/WV3")
FR = f"{R}/full_examples_h5_repaired/test_wv3_OrigScale_multiExm1.h5"
RR = f"{R}/reduced_examples_h5/test_wv3_multiExm1.h5"


def roundtrip(split_code, feeder, cache, lo, hi):
    print(f"=== round-trip control ({'FR' if split_code == 3 else 'RR'} {lo}-{hi}) ===")
    rows = []
    for i in range(lo, hi + 1):
        item = feeder[i]
        ms = item[1][None].double() if split_code == 3 else item[2][None].double()
        d = cache.lookup(split_code, torch.tensor([i]))[0].double()
        m0 = interp23tap(ms)
        mr = warp_hr(upsample_shift(ms, d, 1.0), -4 * d)
        H, W = m0.shape[-2:]
        m = border_mask(d, H, W).bool().expand_as(m0)
        diff = (mr - m0)[m]
        mse = diff.pow(2).mean().item()
        psnr = 10 * np.log10(4.0 / max(mse, 1e-12))            # [-1,1] 범위 -> peak 2
        g0 = scharr_mag_t(m0.mean(1, keepdim=True)); g1 = scharr_mag_t(mr.mean(1, keepdim=True))
        ge = (g1[m[:, :1]].pow(2).mean() / g0[m[:, :1]].pow(2).mean()).item()
        a, b = g0[m[:, :1]], g1[m[:, :1]]
        scc = ((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-12)
        rows.append((i, d[0, 0].item(), d[0, 1].item(), psnr, scc.item(), ge, diff.abs().mean().item()))
        print(f"  #{i:2d} Δ=({d[0,0]:+.3f},{d[0,1]:+.3f})  PSNR {psnr:6.2f} dB  SCC {scc:.5f}  grad-energy {ge:.4f}  MAD {diff.abs().mean():.5f}")
    A = np.array(rows)
    print(f"  mean: PSNR {A[:,3].mean():.2f}  SCC {A[:,4].mean():.5f}  grad-energy {A[:,5].mean():.4f}  MAD {A[:,6].mean():.5f}")
    return rows


def panels(run, cache, lo, hi, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    from utils import to_rgb
    fr = PanFeederAlign(dataroot=FR)
    mats = {k: os.path.join(run, "results", f"full_best_hqnr{k}.mat") for k in ("", "_msframe", "_panframe")}
    sr = {k: loadmat(p)["sr"] for k, p in mats.items() if os.path.exists(p)}
    os.makedirs(out_dir, exist_ok=True)
    for i in range(lo, hi + 1):
        lms, ms, lpan, pan, meta = fr[i]
        d = cache.lookup(3, torch.tensor([i]))[0].double()
        al = upsample_shift(ms[None].double(), d, 1.0)[0]
        cols = [("PAN", pan[0].numpy(), "gray"), ("LRMS interp23tap RGB", to_rgb(lms.numpy()[None])[0], None),
                ("LRMS aligned (+delta) RGB", to_rgb(al.numpy()[None])[0], None)]
        for k, arr in sr.items():
            t = torch.tensor(arr[i] / 2047.0 * 2 - 1).float()
            cols.append((f"HRMS final{k or ''}", to_rgb(t.numpy()[None])[0], None))
        fig, ax = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4.2))
        for a_, (name, img, cm) in zip(ax, cols):
            a_.imshow(img if cm is None else img, cmap=cm); a_.set_title(name, fontsize=9); a_.axis("off")
        fig.suptitle(f"FR #{i}  delta_LR=(dy {d[0,0]:+.3f}, dx {d[0,1]:+.3f})  |delta|={d.norm():.3f} LR px", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"fr{i:02d}.png"), dpi=110); plt.close(fig)
    print(f"panels -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "outputs/global_shift_cache"))
    ap.add_argument("--indices", default="12-19")
    ap.add_argument("--panels", action="store_true")
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    cache = ShiftCache(a.cache_dir)
    lo, hi = (int(x) for x in a.indices.split("-"))
    roundtrip(3, PanFeederAlign(dataroot=FR), cache, lo, hi)
    roundtrip(2, PanFeederAlign(dataroot=RR), cache, lo, hi)
    if a.panels:
        assert a.run, "--run <GA work_dir>"
        panels(a.run, cache, lo, hi, os.path.join(a.run, "visualizations"))


if __name__ == "__main__":
    main()
