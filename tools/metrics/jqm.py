"""JQM — Joint Quality Measure (Palubinskas, Remote Sensing 2015, 7(7):9292, doi:10.3390/rs70709292).

full-resolution 무참조 지표. 두 항의 가중합이다 (논문 Eq. 11, v1 = v2 = 0.5):

    QLR = Σ_k w_k · CMSC(ms_k, (msf_k ★ lpf_k)↓)        Eq. 6  — 분광: 저역통과·데시메이션한 융합 결과 vs 원 MS (저해상도)
    QHR = CMSC(pan, Σ_k w_k · msf_k)                    Eq. 8, 9 — 공간: 분광 가중 intensity vs 원 PAN (고해상도)
    CMSC(x, y) = (1 − (μx−μy)²/R²) · (1 − (σx−σy)²/(R/2)²) · ρ⁺   Eq. 4,  R = 2^L − 1, ρ⁺ = max(ρ, 0)

논문이 정하지 않은 것과 여기서의 선택 (보고 시 명시할 것):
  - 통계 범위: 논문은 "image patches x, y" 라고만 쓴다. 여기서는 **영상 전체(전역)** 통계를 쓴다 — 저자의 CMSC 원 논문(ICIP 2014)
    과 같은 전역 정의. `window` 를 주면 S×S 비중첩 블록 평균(D_s 의 UQI 방식)으로 바꿀 수 있다.
  - lpf_k: 논문은 "밴드별일 수 있는 Gaussian 저역통과, 차단주파수 = 해상도비" 라고만 쓴다. 기본은 **센서 MTF Gaussian(genMTF.m)**
    + (2,2) 위상 데시메이션 — DLPan/Wald 프로토콜과 같은 커널이라 D_λ 와 같은 저역 정의를 공유한다. `lpf="gauss"` 는
    차단 1/ratio 의 일반 Gaussian.
  - w_k: 논문은 제공자의 분광응답함수(SRF)에서 계산한다. 우리 데이터셋엔 SRF 가 없으므로 DLPan `estimation_alpha.m` 처럼
    **MTF_PAN↓PAN 을 MS 밴드로 회귀**하되, SRF 가중치가 음수일 수 없다는 점을 살려 **비음수 최소제곱(NNLS, 절편 없음)** 을 쓴다
    (일반 LS 는 WV3 에서 음수 가중이 나온다; QHR 차이는 0.005 이내). QHR 의 intensity 는 그 가중치 그대로(PAN 스케일),
    QLR 의 밴드 가중은 합 1 로 정규화한다. 장면마다 따로 추정한다.
  - CMSC 의 σ 는 모집단(N) 정의, ρ 는 Pearson. 실측(WV3/QB 논문 세트): d1·d2 는 ~1e-4 이하라 CMSC ≈ ρ⁺ 로 움직인다 —
    QHR 은 사실상 PAN–intensity 상관이고, QB 는 PAN 대역이 MS 4밴드와 덜 겹쳐 융합을 잘해도 0.75 근처다.

PAN-Crafter·U-Know-DiffPAN 은 이 지표를 보고하지 않는다 — 추가 지표이며 논문 표와의 대조값은 없다.
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import nnls

from .eval_fr import genmtf_matlab, GNYQ_TABLE, _fspecial_gaussian, _fwind1_huang

# MTF_PAN.m 의 PAN GNyq
GNYQ_PAN = {"QB": 0.15, "IKONOS": 0.17, "GeoEye1": 0.16, "WV4": 0.16, "WV2": 0.11, "WV3": 0.14}


def cmsc(x, y, R):
    """Eq. 4 — 전역 통계. σ 는 모집단(N) 정의; ρ 는 Pearson, 음수는 0."""
    x = np.asarray(x, dtype=np.float64).ravel(); y = np.asarray(y, dtype=np.float64).ravel()
    mx, my = x.mean(), y.mean(); sx, sy = x.std(), y.std()
    d1 = (mx - my) ** 2 / R ** 2
    d2 = (sx - sy) ** 2 / (R / 2.0) ** 2
    if sx == 0 or sy == 0:
        rho = 1.0 if (sx == 0 and sy == 0) else 0.0
    else:
        rho = float(((x - mx) * (y - my)).mean() / (sx * sy))
    return float((1 - d1) * (1 - d2) * max(rho, 0.0))


def _blockwise(fn, a, b, S):
    h, w = a.shape[:2]; vals = []
    for i in range(0, h - S + 1, S):
        for j in range(0, w - S + 1, S):
            vals.append(fn(a[i:i + S, j:j + S], b[i:i + S, j:j + S]))
    return float(np.mean(vals))


def _filt(img2d, k):
    p = k.shape[0] // 2
    return fftconvolve(np.pad(img2d, p, mode="edge"), k[::-1, ::-1], mode="valid")


def _pan_kernel(sensor, ratio, n=41):
    g = GNYQ_PAN.get(sensor, 0.15)
    alpha = np.sqrt(((n - 1) * (1.0 / ratio / 2)) ** 2 / (-2 * np.log(g)))
    H = _fspecial_gaussian(n, alpha)
    return _fwind1_huang(H / H.max(), np.kaiser(n, 0.5))


def _gauss_kernel(ratio, n=41):
    """차단 1/ratio 의 일반 Gaussian (lpf="gauss"): 차단주파수에서 -3dB 가 되도록 σ 를 둔다."""
    fc = 1.0 / ratio / 2.0                      # Nyquist 기준 정규화 주파수 (cycles/pixel)
    sigma = np.sqrt(np.log(2.0)) / (2 * np.pi * fc)
    return _fspecial_gaussian(n, sigma)


def spectral_weights(ms, pan, sensor, ratio=4):
    """estimation_alpha.m 'global' 과 같은 회귀(비음수 제약): MTF_PAN↓PAN ≈ Σ w_k ms_k (절편 없음). ms (h,w,C), pan (4h,4w)."""
    pan_lr = _filt(pan, _pan_kernel(sensor, ratio))[2::ratio, 2::ratio]
    A = ms.reshape(-1, ms.shape[2]); b = pan_lr.reshape(-1)
    w, _ = nnls(A, b)                       # SRF 가중치처럼 비음수
    return w


def jqm(fused, ms, pan, sensor, ratio=4, R=2047.0, lpf="mtf", weights=None, window=None, v1=0.5):
    """fused (H,W,C) 융합 결과, ms (H/ratio,W/ratio,C) 원 MS, pan (H,W). 반환 dict(JQM, QLR, QHR, w, w_qlr)."""
    fused = np.asarray(fused, dtype=np.float64); ms = np.asarray(ms, dtype=np.float64); pan = np.asarray(pan, dtype=np.float64)
    C = fused.shape[2]
    w = np.asarray(weights, dtype=np.float64) if weights is not None else spectral_weights(ms, pan, sensor, ratio)
    w_qlr = np.clip(w, 0, None); w_qlr = w_qlr / w_qlr.sum() if w_qlr.sum() > 0 else np.full(C, 1.0 / C)
    if lpf == "mtf":
        ker = genmtf_matlab(GNYQ_TABLE.get(sensor) or [0.3] * C, ratio, 41)
        kernels = [ker[:, :, k] for k in range(C)]
    else:
        kernels = [_gauss_kernel(ratio)] * C
    f = (lambda a, b: cmsc(a, b, R)) if window is None else (lambda a, b: _blockwise(lambda p, q: cmsc(p, q, R), a, b, window))
    qlr = 0.0
    for k in range(C):
        low = _filt(fused[:, :, k], kernels[k])[2::ratio, 2::ratio]        # Eq. 6: (msf_k ★ lpf_k)↓
        qlr += w_qlr[k] * f(ms[:, :, k], low)
    intensity = np.tensordot(fused, w, axes=([2], [0]))                    # Eq. 9
    qhr = f(pan, intensity)                                                # Eq. 8
    return dict(JQM=float(v1 * qlr + (1 - v1) * qhr), QLR=float(qlr), QHR=float(qhr), w=w.tolist(), w_qlr=w_qlr.tolist())
