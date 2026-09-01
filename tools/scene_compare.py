#!/usr/bin/env python
"""장면별 RR 지표 대조 — "20장 중 N장 승리" 기준을 실제로 산출한다.

두 계획이 공통으로 요구하는 조건이다 (s1 §8-4 "RR 20장 중 최소 12장에서 student 보다
낮은 reconstruction error", s2 §9.1 "ERGAS 0.5% 개선 또는 20장 중 12장 이상 승리").
그런데 기존 평가 경로(tools/metrics/eval_rr.evaluate)는 장면별 값을 즉시 평균으로
접어 반환해서 이 조건을 낼 수 없었다.

  python tools/scene_compare.py <A실행> <B실행> [--preset wv3]

A 를 기준으로 B 가 몇 장에서 이겼는지 세고, 이항검정 p 값을 함께 낸다 —
**12/20 은 우연히 통과할 확률이 25% 라 그 자체로는 판정 근거가 되지 못한다.**
(양측 p<0.05 는 15/20 부터다.)
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def per_scene(mat, ds):
    """장면별 ERGAS·SAM·RMSE. eval_dlpan 과 같은 크롭·스케일 규약을 쓴다."""
    import importlib.util
    import h5py
    from scipy.io import loadmat
    spec = importlib.util.spec_from_file_location("_ed", os.path.join(ROOT, "tools", "eval_dlpan.py"))
    ed = importlib.util.module_from_spec(spec)
    sys.modules["_ed"] = ed
    spec.loader.exec_module(ed)
    from tools.metrics.eval_rr import evaluate

    scale = ed.SCALE[ds]
    with h5py.File(os.path.join(ROOT, ed.GT_H5[ds])) as f:
        gt = np.asarray(f["gt"], dtype=np.float64).transpose(0, 2, 3, 1)
    sl = slice(20, -21)
    gt_c = gt[:, sl, sl, :]
    sr = loadmat(mat)["sr"].astype(np.float64)
    if sr.shape[1] in (4, 8):
        sr = sr.transpose(0, 2, 3, 1)
    sr_c = sr[:, sl, sl, :]
    rows = []
    for i in range(len(gt_c)):
        m = evaluate(sr_c[i:i + 1], gt_c[i:i + 1], scale, 32)   # 한 장씩 -> 장면값
        rows.append({"ergas": m["ERGAS"][0], "sam": m["SAM"][0],
                     "rmse": float(np.sqrt(np.mean((sr_c[i] - gt_c[i]) ** 2)))})
    return rows


def binom_two_sided(k, n):
    from math import comb
    p = sum(comb(n, i) for i in range(n + 1)
            if abs(i - n / 2) >= abs(k - n / 2)) / 2 ** n
    return min(1.0, p)


def mat_of(tag):
    for name in ("reduced_best_hqnr.mat", "reduced_best_val.mat", "reduced_best_reduced.mat"):
        p = os.path.join(ROOT, "work_dir", tag, "results", name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{tag}: reduced mat 없음")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a"); ap.add_argument("run_b")
    ap.add_argument("--preset", default="wv3")
    ap.add_argument("--metric", default="ergas", choices=["ergas", "sam", "rmse"])
    a = ap.parse_args()

    A = per_scene(mat_of(a.run_a), a.preset)
    B = per_scene(mat_of(a.run_b), a.preset)
    n = min(len(A), len(B))
    key = a.metric
    wins = sum(1 for i in range(n) if B[i][key] < A[i][key])
    ma = float(np.mean([A[i][key] for i in range(n)]))
    mb = float(np.mean([B[i][key] for i in range(n)]))
    p = binom_two_sided(wins, n)

    print(f"장면별 {key.upper()} — {a.run_b} vs {a.run_a} (낮을수록 좋음)")
    print(f"{'scene':>6s} {a.run_a[:22]:>24s} {a.run_b[:22]:>24s}  승")
    for i in range(n):
        w = "B" if B[i][key] < A[i][key] else "A"
        print(f"{i:>6d} {A[i][key]:>24.4f} {B[i][key]:>24.4f}  {w}")
    print(f"\n평균 {ma:.4f} -> {mb:.4f} ({(mb-ma)/ma*100:+.2f}%)")
    print(f"B 승 {wins}/{n}  이항 양측 p={p:.4f}"
          f"  {'유의' if p < 0.05 else '유의하지 않음 — 12/20 은 우연 25%'}")


if __name__ == "__main__":
    main()
