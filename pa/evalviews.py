"""§10 세 evaluation view — 같은 SR 에 대해 PAN reference 와 영역만 바꾼다. metric 식은 tools/metrics/eval_fr.py (B0 evaluator) 그대로.

view            PAN reference   저해상도 PAN                         영역
raw_original    원본 P          imresize↓4 → interp23tap (d_s 내부와 동일)   Ω0 = 전체 프레임 (B0 evaluator 는 crop 없음)
raw_valid       원본 P          같은 경로                              고정 V
aligned_valid   실제 forward P̃  P̃ 에서 같은 경로로 재생성               같은 V

- 필터·decimation 은 전체 프레임 좌표에서 먼저 하고 그 결과를 V 로 자른다 (phase·crop origin 보존, §10.5).
- V 는 블록(32) 배수 origin 의 interior 사각형. margin 은 evaluator support 로 정한다 (아래 SUPPORT), checkpoint 마다 바꾸지 않는다.
- 적격성: |dy|,|dx| ≤ MAX_ELIGIBLE_SHIFT 이면 V 안의 모든 저해상도 PAN 값이 실제 관측(복제 테두리 아님)에서 나온다.
- D_λ 는 PAN 을 쓰지 않으므로 raw_valid == aligned_valid 여야 한다 (E06) — 같은 값을 한 번만 계산해 연결한다.
"""
import hashlib
import json

import numpy as np

from tools.metrics.eval_fr import imresize_matlab, mtf_filter, q2n, _blockproc_uqi
from utils import SCC_full_numpy

BLOCK = 32
# evaluator support (HR px, 한쪽). 저해상도 PAN 경로: imresize bicubic antialias ↓4 (4 tap × scale 4 / 2 = 8) → interp23tap (11 LR = 44 HR).
SUPPORT = dict(warp_bicubic_taps_hr=2, imresize_antialias_hr=8, interp23tap_hr=44, mtf_filter_hr=20, scc_sobel_hr=1, block=BLOCK)
MARGIN_BLOCKS = 2                                   # V = [64:H-64, 64:W-64] (512² → 384², 12×12 블록)
MARGIN = MARGIN_BLOCKS * BLOCK
# 저해상도 PAN 이 V 안에서 복제 테두리에 닿지 않을 최대 |Δ| : margin − (imresize 8 + interp23 44) − warp taps 2 = 10 → 보수적으로 8
MAX_ELIGIBLE_SHIFT = MARGIN - SUPPORT["imresize_antialias_hr"] - SUPPORT["interp23tap_hr"] - SUPPORT["warp_bicubic_taps_hr"] - 2
PROTOCOL_ID = "PAN_GLOBAL_ALIGN_EVAL_v2"


def fixed_roi(H, W):
    """(y0, y1, x0, x1). origin 이 4 와 32 의 배수 → decimation phase·블록 타일링이 전체 프레임과 같다 (E11)."""
    assert H > 2 * MARGIN and W > 2 * MARGIN and MARGIN % 32 == 0
    return (MARGIN, H - MARGIN, MARGIN, W - MARGIN)


def roi_manifest(H, W, n_scenes):
    m = dict(protocol_id=PROTOCOL_ID, H=H, W=W, ratio=4, block=BLOCK, roi=fixed_roi(H, W), margin_hr=MARGIN, support=SUPPORT,
             max_eligible_abs_shift_hr=MAX_ELIGIBLE_SHIFT, n_scenes=n_scenes, rule="fixed interior rectangle, block-aligned, same for all cases/seeds/checkpoints",
             raw_original_domain="full frame (B0 evaluator has no border crop)", dn_conversion="clip(-1,1)->(x+1)/2*max_pixel, no rounding (tools/eval_fr_paperset.py)")
    m["roi_hash"] = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k != "roi_hash"}, sort_keys=True).encode()).hexdigest()[:16]
    return m


def pan_low_reference(pan_hw, wald, ratio=4):
    """d_s 내부와 동일: imresize(1/ratio) → interp23tap. 전체 프레임에서 계산한다."""
    pan_lr = imresize_matlab(pan_hw, 1.0 / ratio)
    return wald.interp23tap(pan_lr[:, :, None], ratio)[:, :, 0]


def eligible(delta_dy_dx):
    return bool(np.isfinite(delta_dy_dx).all() and np.abs(delta_dy_dx).max() <= MAX_ELIGIBLE_SHIFT)


def scene_views(sr_hwc, lms_hwc, pan_hw, pan_aligned_hw, sensor, wald, delta_dy_dx, ratio=4, R=2047.0):
    """한 장면의 세 view. 반환 dict(view -> dict(d_lambda, d_s, hqnr, fscc)), eligible(bool)."""
    H, W = pan_hw.shape
    y0, y1, x0, x1 = fixed_roi(H, W)
    V = (slice(y0, y1), slice(x0, x1))
    fused_deg = mtf_filter(sr_hwc, sensor, ratio, wald)                  # 전체 프레임에서 MTF (SR 은 warp 되지 않는다)
    dl_full = 1.0 - q2n(lms_hwc, fused_deg, BLOCK, BLOCK)[0]
    dl_V = 1.0 - q2n(lms_hwc[V], fused_deg[V], BLOCK, BLOCK)[0]         # 두 valid view 공통 (E06)

    def ds_fscc(pan_ref, region):
        pf = pan_low_reference(pan_ref, wald, ratio)                     # 전체 프레임 → crop
        f, m, p, pl = sr_hwc[region], lms_hwc[region], pan_ref[region], pf[region]
        tot = 0.0
        for b in range(f.shape[2]):
            tot += abs(_blockproc_uqi(f[:, :, b], p, BLOCK) - _blockproc_uqi(m[:, :, b], pl, BLOCK))
        ds = tot / f.shape[2]
        fs = float(SCC_full_numpy(p[..., None] / R, f / R))
        return ds, fs

    full = (slice(None), slice(None))
    ds0, fs0 = ds_fscc(pan_hw, full)
    dsr, fsr = ds_fscc(pan_hw, V)
    dsa, fsa = ds_fscc(pan_aligned_hw, V)
    out = dict(raw_original=dict(d_lambda=dl_full, d_s=ds0, hqnr=(1 - dl_full) * (1 - ds0), fscc=fs0),
               raw_valid=dict(d_lambda=dl_V, d_s=dsr, hqnr=(1 - dl_V) * (1 - dsr), fscc=fsr),
               aligned_valid=dict(d_lambda=dl_V, d_s=dsa, hqnr=(1 - dl_V) * (1 - dsa), fscc=fsa))
    return out, eligible(np.asarray(delta_dy_dx, dtype=np.float64))
