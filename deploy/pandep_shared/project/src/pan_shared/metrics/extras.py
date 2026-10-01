"""Pinned pure numerical primitives. See vendor_reference/metric_sources.json for provenance.
No legacy trainer/controller imports; evaluation only.
"""
import numpy as np
from scipy.ndimage import correlate
from skimage.metrics import structural_similarity

_SOBEL = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

def scc_dlpan(F, G):
    """DLPan-Toolbox Quality_Indices/SCC.m 을 그대로 옮긴 것.

    I(2:end-1, 2:end-1) 로 자른 뒤 imfilter(fspecial('sobel')) 로 y/x 기울기 크기를 만들고,
    전 밴드를 한 벡터로 본 **전역 코사인 유사도** 다(평균 미차감, 밴드 미분리).
    MATLAB imfilter 의 기본값은 **zero padding + correlation** 이라 잘린 영상의 가장자리 1px 링이
    0 과 맞닿는다 — 이 링이 값을 좌우한다.

    2026-09-07 교정 (results_log/2026-09-07_alignment-shift-robust-and-metric-v2.md): 이전 구현은
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
