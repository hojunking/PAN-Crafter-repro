#!/usr/bin/env python
"""J3 blur σ 보정 (계획 §8.1). WV3 train 고정 500 표본에서 J1 jitter 의 gradient-energy 비 r_jit 를 구하고
depthwise Gaussian σ 후보 {0.10..0.35} 중 r_blur(σ) 가 가장 가까운 σ* 를 고른다. 후보 격자에서 |Δr|>1% 면 이웃
사이를 이분해 1% 안으로 (T13). 결과: outputs/shift_robust/blur_calib.json (σ*, r 표, 표본 ID, seed).

  python tools/calibrate_blur.py [--radius 0.5] [--n 500]
"""
import argparse, json, os, sys
import h5py, numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sr.jitter import calibrate_blur_sigma   # noqa: E402

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=float, default=0.5, help="J1 jitter max_abs_hr_px")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--candidates", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/shift_robust/blur_calib.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    with h5py.File(os.path.join(ROOT, "data/PanCollection/WV3/train_wv3.h5")) as f:
        ids = np.sort(rng.choice(f["ms"].shape[0], a.n, replace=False)); ms = f["ms"][ids]
    ms = torch.tensor(2.0 * ms / 2047.0 - 1.0, dtype=torch.float64)            # feeder 정규화와 동일
    ms_base = F.interpolate(ms, scale_factor=4, mode="bicubic")                # 원 bicubic
    # 두 기준을 모두 기록한다. 계획 원안(grad_energy)은 bicubic warp 가 gradient 를 줄이지 않아 σ→0 으로 퇴화하므로
    # J3 config 는 blur.match 로 어느 σ* 를 쓸지 고른다 (기본 mse — 검토서 참고).
    out = dict(sample_ids=ids.tolist(), n=int(a.n), candidates=a.candidates, source="train_wv3.h5", radius=a.radius)
    for match in ("grad_energy", "mse"):
        info = calibrate_blur_sigma(ms_base, a.radius, a.candidates, n_draw=8, seed=a.seed, tol=0.01, match=match)
        out[match] = info
        print(f"[{match:11s}] r_jit {info['r_jit']:.6f}  sigma* {info['sigma_star']:.4f}  r_blur* {info['r_blur_star']:.6f}"
              f"  rel_err {info['rel_err']:.4f}  within_tol={info['within_tol']} refined={info['refined']}")
        for k, v in info["table"].items(): print(f"     sigma {k}: {v:.6f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    sys.exit(0 if out["mse"]["within_tol"] and out["grad_energy"]["within_tol"] else 1)

if __name__ == "__main__":
    main()
