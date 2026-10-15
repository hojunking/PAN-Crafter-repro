"""RR 에서 aligner 의 warp 이 무엇을 바꾸는가 — 같은 모델, 보정량만 쓸기.

캠페인 표(정합 없음 2.0336 / λ* 2.0238 / donor 2.0834)는 **서로 다른 학습 run** 의 비교라
warp 자체의 효과를 분리하지 못한다. 여기서는 **하나의 checkpoint** 를 고정하고 추론 시 적용하는
보정만 alpha 배로 바꿔 RR 지표를 잰다.

  alpha = 0      보정 없음 (aligner_enabled=False; warp(0) 은 무보정과 비트 동일)
  alpha = 1      학습된 예측 그대로
  alpha = 0.5/2/3  과소·과대 보정
  alpha = -1     역부호

읽는 법:
  |alpha| 에 대해 단조 악화  -> sub-pixel 재샘플링 손실이 지배 (정합과 무관)
  어떤 alpha* 에서 최소      -> 정합이 실제로 복원에 기여

평가 경로는 gspread_upload._rr 과 동일(crop [20:-21], tools/metrics/eval_rr).
산출: work_dir/<run>/palsv18/<ckpt>_rr_sweep.json
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")

from tools.po10_diag import load_run                     # noqa: E402
from tools.pa_diag import dn                             # noqa: E402
from main import import_class                            # noqa: E402

ALPHAS = (0.0, 0.5, 1.0, 2.0, 3.0, -1.0)


def _ed():
    spec = importlib.util.spec_from_file_location("_ed", os.path.join(ROOT, "tools", "eval_dlpan.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_ed"] = m
    spec.loader.exec_module(m)
    return m


@torch.no_grad()
def sweep(run, ckpt, dev):
    import h5py
    from tools.metrics.eval_rr import evaluate

    ed = _ed()
    wd, cfg, m, R, mg = load_run(run, ckpt, dev)
    if m.aligner is None:
        return dict(run=run, ckpt=ckpt, skipped="aligner 없음")
    Feeder = import_class(cfg["feeder"])
    ds = Feeder(**cfg["test_reduced_feeder_args"])
    mp = float(ds.max_pixel)
    sensor = "wv3" if "wv3" in cfg["test_reduced_feeder_args"]["dataroot"].lower() else "qb"
    scale = ed.SCALE[sensor]
    with h5py.File(os.path.join(ROOT, ed.GT_H5[sensor])) as f:
        gt = np.asarray(f["gt"], dtype=np.float64).transpose(0, 2, 3, 1)
    sl = slice(20, -21)
    gt_c = gt[:, sl, sl, :]

    # 학습된 예측 먼저
    C = []
    for i in range(len(ds)):
        t = [x.unsqueeze(0).to(dev) for x in ds[i]]
        pan, ms, lpan = t[-1], t[-3], t[-2]
        C.append(m(pan, ms, lpan)["delta"][0].float().cpu().numpy())
    C = np.stack(C)

    out = {}
    for a in ALPHAS:
        sr = []
        for i in range(len(ds)):
            t = [x.unsqueeze(0).to(dev) for x in ds[i]]
            pan, ms, lpan = t[-1], t[-3], t[-2]
            if a == 0.0:
                o = m(pan, ms, lpan, aligner_enabled=False)
            else:
                d = torch.from_numpy((a * C[i]).astype(np.float32))[None].to(dev)
                o = m(pan, ms, lpan, delta_override=d)
            sr.append(dn(o["y"][0], mp).transpose(1, 2, 0))
        sr = np.stack(sr)[:, sl, sl, :]
        r = evaluate(sr, gt_c, scale, 32)
        out[str(a)] = dict(
            alpha=a, mean_shift_px=float(np.linalg.norm(a * C, axis=1).mean()),
            ergas=float(r["ERGAS"][0]), sam=float(r["SAM"][0]), q2n=float(r["Q2n"][0]),
            scc=float(np.mean([ed.scc_dlpan(sr[i], gt_c[i]) for i in range(len(gt_c))])),
            psnr=float(np.mean([ed.psnr_global(sr[i], gt_c[i], scale) for i in range(len(gt_c))])))
        print(f"  alpha {a:+.1f}  |shift| {out[str(a)]['mean_shift_px']:.3f} px   "
              f"ERGAS {out[str(a)]['ergas']:.4f}  SCC {out[str(a)]['scc']:.5f}  "
              f"SAM {out[str(a)]['sam']:.4f}  PSNR {out[str(a)]['psnr']:.3f}", flush=True)

    res = dict(run=run, ckpt=ckpt, learned_c_mean=list(map(float, C.mean(0))),
               learned_c_norm_mean=float(np.linalg.norm(C, axis=1).mean()),
               sweep=out,
               note="같은 checkpoint, 추론 시 delta = alpha * learned. RR crop [20:-21], "
                    "GT = reduced_examples_h5. alpha=0 은 aligner_enabled=False")
    od = os.path.join(wd, "palsv18")
    os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f"{ckpt}_rr_sweep.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"  -> {os.path.relpath(p, ROOT)}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best_hqnr")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    print(f"[{a.run} / {a.ckpt}]")
    sweep(a.run, a.ckpt, torch.device(a.device))


if __name__ == "__main__":
    main()
