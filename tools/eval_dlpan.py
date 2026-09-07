"""DLPan-Toolbox Demo_Reduced_Resolution.m 프로토콜로 reduced-resolution 지표를 낸다.

MATLAB 이 이 머신에 없어서, 옆 저장소 `../CANConv/tools/eval_rr.py` 의 파이썬 포팅을 재사용한다.
그 구현은 CANNet 논문 Table 1 과 SAM/ERGAS/Q2n 이 0.7% 이내로 일치함이 확인돼 있고,
같은 평가기를 쓰므로 CANConv 결과와 직접 비교도 된다.

  python tools/eval_dlpan.py work_dir/wv3_baseline/results/reduced_best_reduced.mat --preset wv3

프로토콜: flag_cut_bounds=1, dim_cut=21, thvalues=0(클리핑 없음), Qblocks_size=32
"""
import os, sys, argparse
import numpy as np, h5py
from scipy.io import loadmat
from scipy.ndimage import sobel, correlate
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.metrics.eval_rr import evaluate  # noqa: E402

# fspecial('sobel') — y 방향. x 방향은 전치.
_SOBEL = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])


def scc_dlpan(F, G):
    """DLPan-Toolbox Quality_Indices/SCC.m 을 그대로 옮긴 것.

    I(2:end-1, 2:end-1) 로 자른 뒤 imfilter(fspecial('sobel')) 로 y/x 기울기 크기를 만들고,
    전 밴드를 한 벡터로 본 **전역 코사인 유사도** 다(평균 미차감, 밴드 미분리).
    MATLAB imfilter 의 기본값은 **zero padding + correlation** 이라 잘린 영상의 가장자리 1px 링이
    0 과 맞닿는다 — 이 링이 값을 좌우한다.

    2026-09-07 교정 (results_log/2026-09-07_metric-comparability-audit.md): 이전 구현은
    scipy.ndimage.sobel(reflect 패딩)이라 링에서 값이 달라 SCC 가 약 +0.004 높게 나왔다.
    CANConv 배포 가중치로 논문 CANConv 행(0.985)과 대조하면 reflect 0.9897(+0.48%) 대 zero padding
    0.9854(+0.04%) 다. 146개 run 에서 두 정의의 Spearman 은 0.992 — 순위는 거의 보존되지만
    논문 표와 절대값을 맞추려면 이쪽이어야 한다. 옛 정의는 scc_scipy_reflect() 로 남긴다
    (utils.SCC_numpy = 학습 로그의 SCC 도 reflect 다).
    """
    def lap(I):
        out = np.empty((I.shape[0] - 2, I.shape[1] - 2, I.shape[2]))
        for b in range(I.shape[2]):
            x = I[1:-1, 1:-1, b]
            gy = correlate(x, _SOBEL, mode="constant", cval=0.0)
            gx = correlate(x, _SOBEL.T, mode="constant", cval=0.0)
            out[:, :, b] = np.sqrt(gy ** 2 + gx ** 2)
        return out
    lf, lg = lap(F), lap(G)
    return float((lf * lg).sum() / np.sqrt((lf ** 2).sum()) / np.sqrt((lg ** 2).sum()))


def scc_scipy_reflect(F, G):
    """2026-09-07 이전의 scc_dlpan (scipy sobel, reflect 패딩). 학습 로그 utils.SCC_numpy 와 같은 값.
    과거 문서·시트의 SCC 는 이 정의였다 (MATLAB 정의보다 평균 +0.004)."""
    lf = np.sqrt(sobel(F[1:-1, 1:-1, :], 0) ** 2 + sobel(F[1:-1, 1:-1, :], 1) ** 2)
    lg = np.sqrt(sobel(G[1:-1, 1:-1, :], 0) ** 2 + sobel(G[1:-1, 1:-1, :], 1) ** 2)
    return float((lf * lg).sum() / np.sqrt((lf ** 2).sum()) / np.sqrt((lg ** 2).sum()))


def psnr_global(F, G, peak):
    """전 밴드 통합 MSE 기준 PSNR (= MATLAB psnr(A,ref,peak) 를 3-D 배열에 그대로 부른 것).

    **PSNR 은 DLPan 표준 프로토콜(indexes_evaluation.m)에 없다.** 논문들이 구현을 밝히지 않아
    관례로 고른다. 근거는 CANConv 배포 가중치: 통합 MSE 37.47 대 논문 CANConv 행 37.441 (+0.08%),
    밴드별 PSNR 평균은 38.99 (+4%, Jensen 부등식으로 항상 통합보다 높다), peak=GT 최대값은 37.15.
    peak 를 2^L=2048 로 두면 +0.004 dB 라 무시된다.
    """
    return float(10 * np.log10(peak ** 2 / ((F - G) ** 2).mean()))


def ssim_skimage(F, G, peak):
    """SSIM — **DLPan 표준 프로토콜에 없는 지표**. 두 논문 모두 구현을 밝히지 않는다.

    2026-09-07 교정: Wang et al. 2004 / MATLAB ssim / DLPan Quality_Indices/ssim.m 의 관례
    (11×11 Gaussian σ=1.5, K=(0.01, 0.03), 모집단 분산, 밴드별 평균, 동적범위 = 데이터 범위) 로 잰다.
    skimage 기본값(7×7 균일창·표본분산)은 이보다 약 +0.002 높다. CANConv 배포 가중치가 논문
    CANConv 행(0.973)과 맞는 쪽은 Gaussian 이다 (0.9732 vs 기본값 0.9751). 방법 간 SSIM 차이가
    0.003 수준이라 이 ±0.002 는 판별력에 직접 걸린다 — SSIM 은 여전히 포화 지표로 취급할 것.
    (동적범위를 DN 에 1 이나 255 로 잘못 주면 0.82 / 0.92 가 나온다. 그런 값이 아니어야 한다.)
    """
    return float(structural_similarity(G / peak, F / peak, data_range=1.0, channel_axis=-1,
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False))


def ssim_skimage_default(F, G, peak):
    """2026-09-07 이전의 ssim_skimage (skimage 기본 7×7 균일창). utils.SSIM_numpy(학습 로그)와 같은 값."""
    return float(structural_similarity(G / peak, F / peak, data_range=1.0, channel_axis=-1))

SCALE = {"wv3": 2047.0, "qb": 2047.0, "wv2": 2047.0, "gf2": 1023.0}
GT_H5 = {s: f"data/PanCollection/{s.upper()}/reduced_examples_h5/test_{s}_multiExm1.h5"
         for s in ("wv3", "qb", "gf2", "wv2")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", nargs="+", help="reduced_*.mat (PAN-Crafter 출력)")
    ap.add_argument("--preset", required=True, choices=list(SCALE))
    ap.add_argument("--dim-cut", type=int, default=21)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--baseline", action="store_true", help="lms(보간 입력) 기준선도 계산")
    a = ap.parse_args()

    scale = SCALE[a.preset]
    with h5py.File(GT_H5[a.preset]) as f:                       # 정답은 원본 h5 에서 직접
        gt = np.asarray(f["gt"], dtype=np.float64).transpose(0, 2, 3, 1)
        lms = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1)

    cut = a.dim_cut
    sl = slice(cut - 1, -cut) if cut > 0 else slice(None)
    gt_c, lms_c = gt[:, sl, sl, :], lms[:, sl, sl, :]

    print(f"preset={a.preset}  N={len(gt)}  평가 shape={gt_c.shape[1:]}  scale={scale:.0f}")
    print(f"dim_cut={cut}  Q2n block={a.block_size}  (thvalues=0)\n")
    hdr = (f"{'':34s} {'PSNR↑*':>17} {'SSIM↑*':>17} {'SAM↓':>17} "
           f"{'ERGAS↓':>17} {'SCC↑':>17} {'Q2n↑':>17}")
    print(hdr); print("-" * len(hdr))

    rows = []
    if a.baseline:
        rows.append(("lms (보간 입력, 기준선)", lms_c))
    for p in a.mat:
        sr = loadmat(p)["sr"].astype(np.float64)                # (N,C,H,W)
        if sr.shape[1] in (4, 8):
            sr = sr.transpose(0, 2, 3, 1)
        rows.append((os.path.relpath(p).replace("work_dir/", "").replace("/results", ""),
                     sr[:, sl, sl, :]))

    for name, data in rows:
        if data.shape != gt_c.shape:
            print(f"{name:34s} shape 불일치 {data.shape} != {gt_c.shape}"); continue
        m = evaluate(data, gt_c, scale, a.block_size)
        sc = [scc_dlpan(data[i], gt_c[i]) for i in range(len(gt_c))]
        pg = [psnr_global(data[i], gt_c[i], scale) for i in range(len(gt_c))]
        ss = [ssim_skimage(data[i], gt_c[i], scale) for i in range(len(gt_c))]
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0   # MATLAB std (N-1)
        m["SCC"] = (float(np.mean(sc)), sd(sc))                  # DLPan 정의로 교체
        m["PSNRb"] = m["PSNR"]                                   # 밴드별 평균 (참고)
        m["PSNR"] = (float(np.mean(pg)), sd(pg))                 # 통합 MSE (논문과 일관)
        m["SSIM"] = (float(np.mean(ss)), sd(ss))
        print(f"{name:34s} " + " ".join(
            f"{m[k][0]:>10.4f}±{m[k][1]:<5.3f}"
            for k in ("PSNR", "SSIM", "SAM", "ERGAS", "SCC", "Q2n")))
        print(f"{'':34s} {'(밴드별 PSNR 평균 = ':>17}{m['PSNRb'][0]:.4f})")
    print("\n* SAM/ERGAS/Q2n: tools/metrics/eval_rr.py (MATLAB 원본 포팅, 논문 CANConv 행 0.5% 이내 검증)")
    print("* SCC: DLPan SCC.m 그대로 (zero-padding Sobel, 전역 코사인). 2026-09-07 이전 값(reflect)보다 약 -0.004")
    print("* PSNR/SSIM: **DLPan 표준 프로토콜에 없는 지표**. 논문이 구현을 밝히지 않아 관례로 고름 —")
    print("  PSNR 통합 MSE(밴드별 평균은 +1.5 dB), SSIM Gaussian 11×11 σ1.5(skimage 기본은 +0.002). 근거: 배포 CANConv 대조")


if __name__ == "__main__":
    main()
