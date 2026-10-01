"""JQM — Joint Quality Measure (Palubinskas, Remote Sensing 2015, 7(7):9292, doi:10.3390/rs70709292),
SIPSA-Net(CVPR 2021) 보충자료 Eq. 3-9 의 규약을 따른다 (2026-09-08 사용자 결정).

full-resolution 무참조 지표, 두 항의 가중합:

    QLR = (1/N) Σ_k CMSC(MS_k, (PS_k ★ lpf_k)↓)          SIPSA Eq. 4-5 — 밴드 균등평균 (Palubinskas 원 논문 Eq. 6 은 SRF 가중합)
    QHR = CMSC(PAN, Σ_k w_k · PS_k)                      SIPSA Eq. 7 — w_k 는 제공자 SRF 로 계산한 분광응답 가중치
    JQM = 0.5·QLR + 0.5·QHR                              SIPSA Eq. 9
    CMSC(x, y) = (1 − (μx−μy)²/R²)·(1 − (σx−σy)²/(R/2)²)·ρ⁺,  R = 2^L − 1, ρ⁺ = max(ρ, 0)   (Palubinskas Eq. 4)

범위 정책 (검증 지적 2026-09-08: 합이 제한되지 않은 회귀 가중치로 intensity 가 R 을 넘으면 (1−d) 항이 음수가 되어 곱이 1 을
크게 넘는다 — 원 논문 §2.3.3 의 [0,1] 보장이 깨진다):
  - 입력(PS, MS, PAN)은 [0, R] 로 자른다 (DLPan thvalues 와 같은 취지; 배포 lms 의 보간 링잉 음수도 여기서 정리된다).
  - QHR 의 가중치는 **비음수·합 1(볼록 결합)** 만 허용한다 → intensity 는 PS 의 [min, max] ⊂ [0, R] 안에 있다.
  - 그 결과 두 감점 항과 ρ⁺ 가 모두 [0, 1] 이고 CMSC·QLR·QHR·JQM ∈ [0, 1]. 실행 시 assert 로 확인한다 — 최종값을 자르지 않는다.

w_k (SRF): 우리 데이터셋엔 제공자 SRF 가 없다. SRF_WEIGHTS[sensor] 에 값이 있으면 그것을 쓰고(합 1 정규화), 없으면
MTF_PAN↓PAN ≈ Σ w_k MS_k 의 **비음수 최소제곱을 합 1 로 정규화**한 대체 가중치를 쓴다 (DLPan estimation_alpha 방식).
이 경우 결과 dict 의 w_source 가 "nnls-normalized" 로 남는다 — 보고 시 **"SRF 대체(회귀) JQM 변형"** 임을 명시할 것.
SIPSA-Net 보고값과 같은 조건이라고 주장할 수 없다(SIPSA 공개 코드에는 JQM 함수가 없어 출력 대조도 불가).

lpf_k: 두 논문 모두 "Gaussian 저역통과" 라고만 쓴다. 기본은 센서 MTF Gaussian(genMTF.m) + (2,2) 위상 데시메이션 — D_λ 와 같은
저역 정의. `lpf="gauss"` 는 차단 1/ratio 의 일반 Gaussian. 통계는 영상 전역(Palubinskas 의 CMSC 원 논문 ICIP 2014 와 같음).
PAN-Crafter·U-Know-DiffPAN 은 JQM 을 보고하지 않는다 — 추가 지표이며 판정 기준이 아니다.
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import nnls

from .eval_fr import genmtf_matlab, GNYQ_TABLE, _fspecial_gaussian, _fwind1_huang

# MTF_PAN.m 의 PAN GNyq
GNYQ_PAN = {"QB": 0.15, "IKONOS": 0.17, "GeoEye1": 0.16, "WV4": 0.16, "WV2": 0.11, "WV3": 0.14}
# 제공자 SRF 로 계산한 분광응답 가중치 (밴드 순서 = 데이터셋 밴드 순서). 값이 확보되면 여기 넣는다 — 지금은 없음.
SRF_WEIGHTS = {}


def cmsc(x, y, R):
    """Palubinskas Eq. 4 — 전역 통계. σ 는 모집단(N) 정의; ρ 는 Pearson, 음수는 0. 입력은 [0, R] 이어야 한다."""
    x = np.asarray(x, dtype=np.float64).ravel(); y = np.asarray(y, dtype=np.float64).ravel()
    mx, my = x.mean(), y.mean(); sx, sy = x.std(), y.std()
    t1 = 1.0 - (mx - my) ** 2 / R ** 2
    t2 = 1.0 - (sx - sy) ** 2 / (R / 2.0) ** 2
    if sx == 0 or sy == 0:
        rho = 1.0 if (sx == 0 and sy == 0) else 0.0
    else:
        rho = float(((x - mx) * (y - my)).mean() / (sx * sy))
    assert 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0, f"CMSC 항이 [0,1] 밖 (t1={t1:.4f}, t2={t2:.4f}) — 입력이 [0,R] 을 벗어났다"
    return float(t1 * t2 * max(rho, 0.0))


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
    fc = 1.0 / ratio / 2.0
    sigma = np.sqrt(np.log(2.0)) / (2 * np.pi * fc)
    return _fspecial_gaussian(n, sigma)


def spectral_weights(ms, pan, sensor, ratio=4):
    """SRF 대체: MTF_PAN↓PAN ≈ Σ w_k ms_k 의 비음수 최소제곱(절편 없음)을 합 1 로 정규화. ms (h,w,C), pan (4h,4w)."""
    pan_lr = _filt(pan, _pan_kernel(sensor, ratio))[2::ratio, 2::ratio]
    A = ms.reshape(-1, ms.shape[2]); b = pan_lr.reshape(-1)
    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else np.full(ms.shape[2], 1.0 / ms.shape[2])


def convex_weights(w, C):
    w = np.asarray(w, dtype=np.float64)
    assert w.shape == (C,) and np.all(w >= 0) and w.sum() > 0, "QHR 가중치는 비음수·합>0 이어야 한다"
    return w / w.sum()


def jqm(fused, ms, pan, sensor, ratio=4, R=2047.0, lpf="mtf", weights=None, window=None, v1=0.5):
    """fused (H,W,C) 융합 결과, ms (H/ratio,W/ratio,C) 원 MS, pan (H,W). 반환 dict(JQM, QLR, QHR, w, w_source)."""
    fused = np.clip(np.asarray(fused, dtype=np.float64), 0.0, R)          # 범위 정책: [0, R]
    ms = np.clip(np.asarray(ms, dtype=np.float64), 0.0, R)
    pan = np.clip(np.asarray(pan, dtype=np.float64), 0.0, R)
    C = fused.shape[2]
    if weights is not None:
        w, src = convex_weights(weights, C), "given"
    elif sensor in SRF_WEIGHTS:
        w, src = convex_weights(SRF_WEIGHTS[sensor], C), "srf"
    else:
        w, src = spectral_weights(ms, pan, sensor, ratio), "nnls-normalized"
    if lpf == "mtf":
        ker = genmtf_matlab(GNYQ_TABLE.get(sensor) or [0.3] * C, ratio, 41)
        kernels = [ker[:, :, k] for k in range(C)]
    else:
        kernels = [_gauss_kernel(ratio)] * C
    f = (lambda a, b: cmsc(a, b, R)) if window is None else (lambda a, b: _blockwise(lambda p, q: cmsc(p, q, R), a, b, window))
    qlr = 0.0
    for k in range(C):
        low = np.clip(_filt(fused[:, :, k], kernels[k])[2::ratio, 2::ratio], 0.0, R)   # (PS_k ★ lpf_k)↓ ; 필터 링잉 대비 클립
        qlr += f(ms[:, :, k], low) / C                                                 # SIPSA Eq. 4: 균등평균
    intensity = np.tensordot(fused, w, axes=([2], [0]))                                # SIPSA Eq. 7 (볼록 결합 → [0,R])
    qhr = f(pan, intensity)
    out = dict(JQM=float(v1 * qlr + (1 - v1) * qhr), QLR=float(qlr), QHR=float(qhr), w=w.tolist(), w_source=src)
    assert 0.0 <= out["JQM"] <= 1.0, out
    return out


if __name__ == "__main__":
    # 자체 검사 (검증 지적 2026-09-08 재현): 합이 제한되지 않은 가중치·범위 밖 입력에서도 [0,1] 이어야 하고, 음수 가중치는 거부
    rng = np.random.default_rng(0)
    pan = rng.uniform(0, 2047, (64, 64)); ms = rng.uniform(0, 2047, (16, 16, 4))
    fused = np.repeat(np.repeat(ms, 4, 0), 4, 1)
    a = rng.uniform(0, 2047, (64, 64)); assert abs(cmsc(a, a, 2047) - 1.0) < 1e-12
    r1 = jqm(fused, ms, pan, "QB", 4, 2047.0, weights=[10.37, 0, 0, 0])          # 지적의 사례: 볼록 정규화 → [1,0,0,0]
    assert 0 <= r1["JQM"] <= 1 and abs(sum(r1["w"]) - 1) < 1e-12, r1
    r2 = jqm(fused * 1.5 - 300, ms, pan, "QB", 4, 2047.0)                          # 범위 밖 입력 → 클립 후 [0,1]
    assert 0 <= r2["JQM"] <= 1, r2
    try:
        jqm(fused, ms, pan, "QB", 4, 2047.0, weights=[1, -0.2, 0, 0]); bad = True
    except AssertionError:
        bad = False
    print("jqm self-test:", "FAIL — 음수 가중치가 통과됨" if bad else "OK",
          f"| 지적 사례 JQM {r1['JQM']:.4f} (w→{np.round(r1['w'],3).tolist()}) | 범위 밖 입력 JQM {r2['JQM']:.4f}")
