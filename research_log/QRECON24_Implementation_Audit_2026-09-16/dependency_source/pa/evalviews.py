"""§10 세 evaluation view — 같은 SR 에 대해 PAN reference 와 영역만 바꾼다. metric 식은 tools/metrics/eval_fr.py (B0 evaluator) 그대로.

view            PAN reference   저해상도 PAN                         영역
raw_original    원본 P          imresize↓4 → interp23tap (d_s 내부와 동일)   Ω0 = 전체 프레임 (B0 evaluator 는 crop 없음)
raw_valid       원본 P          같은 경로                              고정 V
aligned_valid   P̃ = W(P_raw, Δ̂)  P̃ 에서 같은 경로로 재생성              같은 V

- 모든 필터(MTF, imresize·interp23tap, Sobel)는 **전체 프레임 좌표에서 먼저** 계산하고 그 결과를 V 로 자른다 (phase·crop origin 보존, §10.5).
  fSCC 도 Sobel 맵을 전체 프레임에서 만든 뒤 자른다 (검토 지적 4).
- V 는 블록(32) 배수 origin 의 interior 사각형. margin 은 evaluator support 로 정한다 (SUPPORT), checkpoint 마다 바꾸지 않는다.
- 적격성: |dy|,|dx| ≤ MAX_ELIGIBLE_SHIFT 이고 세 view 의 metric 이 전부 유한해야 한다. 이유는 reason 문자열로 남긴다.
- D_λ 는 PAN 을 쓰지 않으므로 raw_valid == aligned_valid (E06) — 같은 값을 한 번만 계산해 연결한다.
- 참조(lms·pan)는 h5 의 원본 float64, SR 은 clip(−1,1)→(x+1)/2·max_pixel — tools/eval_fr_paperset.py(시트) 와 같은 규칙.
"""
import hashlib
import json
import os

import numpy as np
from scipy.ndimage import sobel

from tools.metrics.eval_fr import imresize_matlab, mtf_filter, q2n, _blockproc_uqi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = 32
# evaluator support (HR px, 한쪽). 저해상도 PAN 경로: imresize bicubic antialias ↓4 (4 tap × scale 4 / 2 = 8) → interp23tap (11 LR = 44 HR).
SUPPORT = dict(warp_bicubic_taps_hr=2, imresize_antialias_hr=8, interp23tap_hr=44, mtf_filter_hr=20, scc_sobel_hr=1, block=BLOCK)
MARGIN_BLOCKS = 2                                   # V = [64:H-64, 64:W-64] (512² → 384², 12×12 블록)
MARGIN = MARGIN_BLOCKS * BLOCK
# 저해상도 PAN 이 V 안에서 복제 테두리에 닿지 않을 최대 |Δ| : margin − (imresize 8 + interp23 44) − warp taps 2 = 10 → 보수적으로 8
MAX_ELIGIBLE_SHIFT = MARGIN - SUPPORT["imresize_antialias_hr"] - SUPPORT["interp23tap_hr"] - SUPPORT["warp_bicubic_taps_hr"] - 2
PROTOCOL_ID = "PAN_GLOBAL_ALIGN_EVAL_v2"
VIEWS = ("raw_original", "raw_valid", "aligned_valid")


def evaluator_hash():
    """평가 코드 hash — selector state 재개 검증용 (검토 지적 5)."""
    h = hashlib.sha256()
    for rel in ("tools/metrics/eval_fr.py", "tools/metrics/q2n.py", "pa/evalviews.py", "pa/warp.py"):
        h.update(open(os.path.join(ROOT, rel), "rb").read())
    return h.hexdigest()[:16]


def fixed_roi(H, W, margin=MARGIN):
    """(y0, y1, x0, x1). origin 이 4 와 32 의 배수 → decimation phase·블록 타일링이 전체 프레임과 같다 (E11). margin 은 블록 배수."""
    assert H > 2 * margin and W > 2 * margin and margin % 32 == 0
    return (margin, H - margin, margin, W - margin)


STRESS_MARGIN = 3 * BLOCK                           # corruption stress 전용 ROI (PO10 §6.3): 두 단계 warp(ε+ĉ): 저해상도 PAN support 52(imresize 8 + interp23 44) + 두 단계 warp taps 2×2 ≤ 96 → |ε|+|ĉ| ≤ 40
MAX_ELIGIBLE_TWO_STAGE = STRESS_MARGIN - SUPPORT["imresize_antialias_hr"] - SUPPORT["interp23tap_hr"] - 2 * SUPPORT["warp_bicubic_taps_hr"]


def stress_roi_manifest(H, W):
    m = dict(protocol_id=PROTOCOL_ID + "_stress", H=H, W=W, block=BLOCK, roi=fixed_roi(H, W, STRESS_MARGIN), margin_hr=STRESS_MARGIN,
             rule="fixed interior, block-aligned; eligibility |eps|_inf + |c_hat|_inf <= MAX_ELIGIBLE_TWO_STAGE (two sampling stages)", max_eligible_two_stage=MAX_ELIGIBLE_TWO_STAGE,
             support=SUPPORT, note="diagnostic only — not used for checkpoint selection", evaluator_hash=evaluator_hash())
    m["roi_hash"] = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k != "roi_hash"}, sort_keys=True).encode()).hexdigest()[:16]
    return m


def roi_manifest(H, W, n_scenes, input_h5=None, input_sha=None):
    m = dict(protocol_id=PROTOCOL_ID, H=H, W=W, ratio=4, block=BLOCK, roi=fixed_roi(H, W), margin_hr=MARGIN, support=SUPPORT,
             max_eligible_abs_shift_hr=MAX_ELIGIBLE_SHIFT, n_scenes=n_scenes, rule="fixed interior rectangle, block-aligned, same for all cases/seeds/checkpoints",
             raw_original_domain="full frame (B0 evaluator has no border crop)", dn_conversion="SR clip(-1,1)->(x+1)/2*max_pixel no rounding; lms/pan raw float64 from h5 (tools/eval_fr_paperset.py)",
             fscc="Sobel maps on full frame (utils.SCC_full_numpy definition), then crop", evaluator_hash=evaluator_hash(), input_h5=input_h5, input_sha256=input_sha)
    m["roi_hash"] = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k not in ("roi_hash", "input_h5")}, sort_keys=True).encode()).hexdigest()[:16]
    return m


def pan_low_reference(pan_hw, wald, ratio=4):
    """d_s 내부와 동일: imresize(1/ratio) → interp23tap. 전체 프레임에서 계산한다."""
    pan_lr = imresize_matlab(pan_hw, 1.0 / ratio)
    return wald.interp23tap(pan_lr[:, :, None], ratio)[:, :, 0]


def sobel_maps(x_hwc):
    """utils.SCC_full_numpy 와 같은 정의: x[1:-1,1:-1] 에 scipy sobel(axis 0/1) → 크기. [H-2, W-2, C]. 전체 프레임에서 만든다."""
    xi = x_hwc[1:-1, 1:-1, :]
    return np.sqrt(sobel(xi, axis=0) ** 2 + sobel(xi, axis=1) ** 2)


def fscc_from_maps(m_pan, m_sr, region=None):
    """m_pan [h,w,1], m_sr [h,w,C] (sobel_maps 출력). region 은 맵 좌표 slice. 전체 프레임이면 SCC_full_numpy 와 동일."""
    if region is not None:
        m_pan, m_sr = m_pan[region], m_sr[region]
    g = np.repeat(m_pan, m_sr.shape[2], axis=2)
    num = np.sum(m_sr * g, axis=(0, 1)); den = np.sqrt(np.sum(m_sr ** 2, axis=(0, 1)) * np.sum(g ** 2, axis=(0, 1)))
    return float(np.mean(num / (den + 1e-8)))


def eligibility(delta_dy_dx, views):
    d = np.asarray(delta_dy_dx, dtype=np.float64)
    if not np.isfinite(d).all():
        return False, "nan_delta"
    if np.abs(d).max() > MAX_ELIGIBLE_SHIFT:
        return False, f"shift>{MAX_ELIGIBLE_SHIFT}px"
    for v in VIEWS:
        if not all(np.isfinite(views[v][k]) for k in ("d_lambda", "d_s", "hqnr", "fscc")):
            return False, f"nan_metric:{v}"
    return True, ""


def scene_views(sr_hwc, lms_hwc, pan_hw, pan_aligned_hw, sensor, wald, delta_dy_dx, ratio=4, R=2047.0, margin=MARGIN):
    """한 장면의 세 view. 반환 (dict(view -> dict(d_lambda, d_s, hqnr, fscc)), eligible: bool, reason: str). margin 은 선택용 V(기본) 또는 stress ROI."""
    H, W = pan_hw.shape
    y0, y1, x0, x1 = fixed_roi(H, W, margin)
    V = (slice(y0, y1), slice(x0, x1)); VM = (slice(y0 - 1, y1 - 1), slice(x0 - 1, x1 - 1))     # Sobel 맵은 [1:-1] 만큼 어긋난다
    fused_deg = mtf_filter(sr_hwc, sensor, ratio, wald)                  # 전체 프레임에서 MTF (SR 은 warp 되지 않는다)
    dl_full = 1.0 - q2n(lms_hwc, fused_deg, BLOCK, BLOCK)[0]
    dl_V = 1.0 - q2n(lms_hwc[V], fused_deg[V], BLOCK, BLOCK)[0]         # 두 valid view 공통 (E06)
    m_sr = sobel_maps(sr_hwc / R)

    def ds_fscc(pan_ref, region, mregion):
        pf = pan_low_reference(pan_ref, wald, ratio)                     # 전체 프레임 → crop
        f, m, p, pl = sr_hwc[region], lms_hwc[region], pan_ref[region], pf[region]
        tot = 0.0
        for b in range(f.shape[2]):
            tot += abs(_blockproc_uqi(f[:, :, b], p, BLOCK) - _blockproc_uqi(m[:, :, b], pl, BLOCK))
        ds = tot / f.shape[2]
        fs = fscc_from_maps(sobel_maps(pan_ref[..., None] / R), m_sr, mregion)
        return ds, fs

    full = (slice(None), slice(None))
    ds0, fs0 = ds_fscc(pan_hw, full, None)
    dsr, fsr = ds_fscc(pan_hw, V, VM)
    dsa, fsa = ds_fscc(pan_aligned_hw, V, VM)
    out = dict(raw_original=dict(d_lambda=dl_full, d_s=ds0, hqnr=(1 - dl_full) * (1 - ds0), fscc=fs0),
               raw_valid=dict(d_lambda=dl_V, d_s=dsr, hqnr=(1 - dl_V) * (1 - dsr), fscc=fsr),
               aligned_valid=dict(d_lambda=dl_V, d_s=dsa, hqnr=(1 - dl_V) * (1 - dsa), fscc=fsa))
    ok, reason = eligibility(delta_dy_dx, out)
    return out, ok, reason


def raw_views(sr_hwc, lms_hwc, pan_hw, sensor, wald, ratio=4, R=2047.0, margin=MARGIN):
    """aligner 와 무관한 두 HQNR: raw_original(전체 프레임, 논문 프로토콜) · raw_valid(고정 V = 가장자리 margin 제외, 블록 정렬). 모든 run 의 시트 열 HQNR↑ / HQNR(V64)↑."""
    H, W = pan_hw.shape
    y0, y1, x0, x1 = fixed_roi(H, W, margin); V = (slice(y0, y1), slice(x0, x1)); VM = (slice(y0 - 1, y1 - 1), slice(x0 - 1, x1 - 1))
    fused_deg = mtf_filter(sr_hwc, sensor, ratio, wald)
    dl_full = 1.0 - q2n(lms_hwc, fused_deg, BLOCK, BLOCK)[0]; dl_V = 1.0 - q2n(lms_hwc[V], fused_deg[V], BLOCK, BLOCK)[0]
    pf = pan_low_reference(pan_hw, wald, ratio); m_sr = sobel_maps(sr_hwc / R); m_p = sobel_maps(pan_hw[..., None] / R)

    def ds(region):
        f, m, p, pl = sr_hwc[region], lms_hwc[region], pan_hw[region], pf[region]
        return sum(abs(_blockproc_uqi(f[:, :, b], p, BLOCK) - _blockproc_uqi(m[:, :, b], pl, BLOCK)) for b in range(f.shape[2])) / f.shape[2]
    ds0, dsV = ds((slice(None), slice(None))), ds(V)
    return dict(raw_original=dict(d_lambda=dl_full, d_s=ds0, hqnr=(1 - dl_full) * (1 - ds0), fscc=fscc_from_maps(m_p, m_sr)),
                raw_valid=dict(d_lambda=dl_V, d_s=dsV, hqnr=(1 - dl_V) * (1 - dsV), fscc=fscc_from_maps(m_p, m_sr, VM)))
