#!/usr/bin/env python
"""평가 지표 구현이 다른 서버로 정확히 이식됐는지 확인한다.

    python tools/verify_metrics.py

지표는 순수 numpy/scipy 연산이라 **입력이 같으면 서버가 달라도 값이 같아야 한다.**
그래서 실제 위성영상 대신 고정 시드 난수를 쓴다 — 저장소에 데이터를 넣지 않고도
동일한 검증이 되고, 영상 라이선스 문제도 없다.

full-resolution 은 DLPan-Toolbox 의 wald_utilities.py 를 런타임에 import 하므로
PANCRAFTER_DLPAN 이 필요하다. 없으면 reduced 만 검사한다.
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.metrics.eval_rr import evaluate                          # noqa: E402
from tools.metrics.eval_fr import load_dlpan, d_lambda_k, d_s       # noqa: E402

SCALE = 2047.0          # WV3 11bit
RTOL = 1e-9             # 같은 코드·같은 입력이면 이 정도로 맞아야 한다

# 이 서버(생성 시점)에서 나온 값. 다른 서버에서 어긋나면 이식이 잘못된 것이다.
EXPECTED = {
    "reduced": {"SAM": 2.69059269243812,
                "ERGAS": 2.6117638486447348,
                "Q2n": 0.9342110942114301},
    "full": {"D_lambda": 0.04943129510937505,
             "D_s": 0.016026738589190015,
             "HQNR": 0.935334188746278},
}


def make_reduced(n=2, c=8, h=256, seed=20260824):
    """결정론적 난수. 실제 영상 통계를 거칠게 모사한다(밴드별 평균이 다르고 전부 양수)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(200, 1400, size=(n, 1, 1, c))                 # 밴드별 밝기
    gt = base + rng.normal(0, 120, size=(n, h, h, c))
    sr = gt + rng.normal(0, 45, size=(n, h, h, c))                   # 예측 = GT + 오차
    return np.clip(gt, 1, SCALE), np.clip(sr, 0, SCALE)


def make_full(n=2, c=8, h=128, ratio=4, seed=20260825):
    """full-res 용. D_lambda 는 밴드 간 상관을 보므로, 밴드가 서로 무관한 순수 난수를
    넣으면 값이 0.9 대로 포화해 오류를 감출 수 있다. 공통 공간 구조에 밴드별 이득을
    곱해 실제 다분광 영상처럼 밴드 간 상관을 갖게 만든다."""
    rng = np.random.default_rng(seed)
    # 공간적으로 매끄러운 구조여야 한다. 픽셀 단위 백색 잡음은 D_lambda 의 MTF 평활에
    # 전부 사라져 Q2n 이 0 으로 떨어지고, 값이 0.9 대로 포화해 오류를 감춘다.
    struct = gaussian_filter(rng.normal(0, 1, size=(n, h, h)), sigma=(0, 3, 3))
    struct = (struct / struct.std())[..., None]                      # 밴드 공통 공간 구조
    gain = rng.uniform(0.6, 1.4, size=(n, 1, 1, c))                  # 밴드별 반응
    base = rng.uniform(300, 1200, size=(n, 1, 1, c))
    lms = np.clip(base + 180 * struct * gain
                  + rng.normal(0, 25, size=(n, h, h, c)), 1, SCALE)
    pan = np.clip(700 + 200 * struct[..., 0]
                  + rng.normal(0, 30, size=(n, h, h)), 1, SCALE)     # PAN 도 같은 구조
    sr = np.clip(lms + rng.normal(0, 30, size=(n, h, h, c)), 0, SCALE)
    return lms, pan, sr


def run_reduced():
    gt, sr = make_reduced()
    m = evaluate(sr, gt, SCALE, 32)
    return {k: float(m[k][0]) for k in ("SAM", "ERGAS", "Q2n")}


def run_full():
    dlpan = os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")
    wald = load_dlpan(dlpan)                     # 없으면 FileNotFoundError
    lms, pan, sr = make_full()
    out = {}
    dls, dss = [], []
    for i in range(len(sr)):
        dls.append(d_lambda_k(sr[i], lms[i], "wv3", 4, 32, wald))
        dss.append(d_s(sr[i], lms[i], pan[i], 4, 32, wald))
    dl, ds = float(np.mean(dls)), float(np.mean(dss))
    out["D_lambda"], out["D_s"] = dl, ds
    out["HQNR"] = float((1 - dl) * (1 - ds))
    return out


def main():
    emit = "--emit" in sys.argv
    ok = True

    got_r = run_reduced()
    try:
        got_f = run_full()
        have_full = True
    except FileNotFoundError as e:
        got_f, have_full = {}, False
        print(f"[skip] full-resolution: {e}".split("\n")[0])

    if emit:
        print("EXPECTED = {")
        print(f'    "reduced": {got_r!r},')
        print(f'    "full": {got_f!r},')
        print("}")
        return 0

    for grp, got in (("reduced", got_r), ("full", got_f)):
        exp = EXPECTED.get(grp) or {}
        if not exp:
            print(f"[skip] {grp}: 기대값이 비어 있다 (--emit 으로 생성할 것)")
            continue
        if grp == "full" and not have_full:
            continue
        for k, v in exp.items():
            g = got.get(k)
            if g is None:
                print(f"  FAIL {grp}/{k}: 계산되지 않음"); ok = False; continue
            rel = abs(g - v) / max(abs(v), 1e-12)
            mark = "OK  " if rel <= RTOL else "FAIL"
            if rel > RTOL:
                ok = False
            print(f"  {mark} {grp}/{k:9s} 기대 {v:.12g}  실측 {g:.12g}  상대오차 {rel:.2e}")

    print("\n" + ("전부 일치 — 지표 구현 이식 정상" if ok else "!! 불일치 — 이식 확인 필요"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
