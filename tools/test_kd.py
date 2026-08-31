#!/usr/bin/env python
"""KD·mutual 프레임워크 unit test — 명세 §26. assert 기반, 실패 시 exit 1.

  python tools/test_kd.py
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kd.ops import (MTFDownsampler, AbsoluteGradient, LocalVarianceMap,
                    build_shift_candidates, mean_normalize, ramp_then_decay)
from kd.losses import (sis_loss, uncertainty_nll, uknow_weights, weighted_l1,
                       spectral_kd_loss, mutual_residual)
from kd.features import FeatureTap, FeatureProj, UncertaintyHead, WithUncertainty
from model.pancrafter_paper import PANCrafterPaper

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"[test] OK   {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"[test] FAIL {name}: {type(e).__name__}: {e}")


torch.manual_seed(0)


# ---------------------------------------------------------------- SiS (§26.1)
def t_sis_exact_shift():
    lr = torch.rand(2, 8, 16, 16)
    target = torch.roll(lr, shifts=1, dims=-1)          # 오른쪽 1px 이동한 관측
    loss, d = sis_loss(lr, target, radius=1, mode="shared_vector")
    base, _ = sis_loss(lr, target, radius=0)
    assert loss < base * 0.15, f"shift 회복 실패: {loss:.4f} vs r0 {base:.4f}"
    assert d["mean_offset_magnitude"] > 0.5


def t_sis_center_equiv():
    a, b = torch.rand(2, 8, 16, 16), torch.rand(2, 8, 16, 16)
    l0, _ = sis_loss(a, b, radius=0)
    assert torch.allclose(l0, F.l1_loss(a, b)), "radius=0 은 관례적 L1 과 같아야"


def t_sis_shared_vs_bandwise():
    lr = torch.rand(1, 8, 16, 16)
    tgt = lr.clone()
    tgt[:, :4] = torch.roll(tgt[:, :4], 1, dims=-1)     # 절반 밴드만 이동
    l_band, _ = sis_loss(lr, tgt, radius=1, mode="bandwise")
    l_shared, _ = sis_loss(lr, tgt, radius=1, mode="shared_vector", eta_sam=0.0)
    assert l_band < l_shared, "bandwise 는 밴드별 이동을 흡수해 더 낮아야 (구성 차이 확인)"


def t_sis_grad():
    pred = torch.rand(1, 8, 16, 16, requires_grad=True)
    tgt = torch.rand(1, 8, 16, 16, requires_grad=True)
    loss, _ = sis_loss(pred, tgt, radius=1)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert tgt.grad is None, "target 으로 grad 가 새면 안 된다 (detach)"


# ---------------------------------------------------------- uncertainty (§26.2)
def t_uncertainty():
    head = UncertaintyHead(32)
    theta = head(torch.randn(2, 32, 16, 16))
    assert (theta > 0).all() and torch.isfinite(theta).all()
    pred, gt = torch.rand(2, 8, 16, 16), torch.rand(2, 8, 16, 16)
    loss, _ = uncertainty_nll(pred, gt, theta)
    assert torch.isfinite(loss)
    wh, ws = uknow_weights(theta, "robust_normalized")
    assert (ws >= 0).all(), "soft weight 음수 금지"
    assert abs(wh.mean().item() - 1) < 1e-3 and abs(ws.mean().item() - 1) < 1e-3, "평균 1 정규화"
    wh2, ws2 = uknow_weights(theta, "paper_raw", tau=1.0)
    assert (ws2 >= 0).all()


def t_uncertainty_direction():
    # 오차가 큰 합성 영역에서 theta 가 커지는 방향으로 학습되는지 (몇 step gradient)
    head = UncertaintyHead(8)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    feat = torch.randn(1, 8, 16, 16)
    err = torch.zeros(1, 8, 16, 16); err[..., :8, :] = 1.0   # 위 절반만 오차
    gt = torch.zeros_like(err)
    for _ in range(200):
        theta = head(feat)
        loss, _ = uncertainty_nll(err, gt, theta)
        opt.zero_grad(); loss.backward(); opt.step()
    theta = head(feat)
    assert theta[..., :8, :].mean() > theta[..., 8:, :].mean(), "고오차 영역 theta 증가"


# ------------------------------------------------- mutual isolation (§26.3)
def t_mutual_isolation():
    ma = dict(hidden_size=32, depth=[1, 1, 1], norm="ln", in_mode="released",
              attn_locations=[], num_heads=4)
    torch.manual_seed(1); a = PANCrafterPaper(**ma)
    torch.manual_seed(2); b = PANCrafterPaper(**ma)
    for m in (a, b):
        with torch.no_grad():
            for p in m.parameters():
                if p.abs().sum() == 0:
                    p.normal_(0, 1e-3)
    pan, lpan, ms = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 8, 8), torch.rand(2, 8, 8, 8)
    s = torch.ones(2)
    out_a, out_b = a(pan, lpan, ms, s), b(pan, lpan, ms, s)
    loss_a = mutual_residual(out_a, out_b.detach())
    loss_a.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in b.parameters()), \
        "loss_a 가 peer_b 를 갱신하면 안 된다"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.parameters())


# ---------------------------------------------------------- feature KD (§26.4)
def t_feature_kd():
    ma = dict(hidden_size=32, depth=[1, 1, 2], norm="ln", in_mode="released", attn_locations=[])
    m_t = PANCrafterPaper(**{**ma, "hidden_size": 48})
    m_s = PANCrafterPaper(**ma)
    tap_t, tap_s = FeatureTap(m_t), FeatureTap(m_s)
    pan, lpan, ms = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 8, 8), torch.rand(1, 8, 8, 8)
    s = torch.ones(1)
    m_t(pan, lpan, ms, s); m_s(pan, lpan, ms, s)
    f_t, f_s = tap_t.out["bottleneck_h4"], tap_s.out["bottleneck_h4"]
    assert f_t.shape[1] == 48 and f_s.shape[1] == 32 and f_t.shape[-2:] == f_s.shape[-2:]
    proj = FeatureProj(48, 32, common=16)
    loss = proj(f_s, f_t.detach())
    loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m_t.parameters()), \
        "teacher feature 는 detach — teacher 로 grad 금지"


# ---------------------------------------------------------------- 연산자
def t_ops():
    mtf = MTFDownsampler(bands=8, scale=4)
    y = mtf(torch.rand(2, 8, 64, 64))
    assert y.shape == (2, 8, 16, 16)
    assert not any(p.requires_grad for p in mtf.parameters()), "MTF 커널 고정"
    g = AbsoluteGradient()(torch.rand(2, 1, 32, 32))
    assert (g >= 0).all()
    v = LocalVarianceMap()(torch.rand(2, 8, 32, 32))
    assert v.shape == (2, 1, 32, 32) and (v >= 0).all() and (v <= 1).all()
    c = build_shift_candidates(torch.rand(1, 8, 16, 16), 1)
    assert c.shape == (1, 8, 9, 16, 16)
    assert ramp_then_decay(0, 5000, 15000, 40000) == 0.0
    assert ramp_then_decay(20000, 5000, 15000, 40000) == 1.0
    assert ramp_then_decay(50000, 5000, 15000, 40000, total=50000) == 0.0


# ------------------------------------------------- WithUncertainty 계약
def t_wrapper_contract():
    ma = dict(hidden_size=32, depth=[1, 1, 1], norm="ln", in_mode="released", attn_locations=[])
    base = PANCrafterPaper(**ma)
    w = WithUncertainty(base, 32)
    pan, lpan, ms = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 8, 8), torch.rand(1, 8, 8, 8)
    out = w(pan, lpan, ms, torch.ones(1))
    assert out.shape == (1, 8, 32, 32), "forward 계약(residual) 유지 — 평가 경로 호환"
    theta = w.theta()
    assert theta.shape == (1, 1, 32, 32) and (theta > 0).all()
    sd = w.state_dict()
    assert any(k.startswith("head.") for k in sd) and any(k.startswith("base.") for k in sd)


check("SiS: 정확한 shift 회복", t_sis_exact_shift)
check("SiS: radius=0 == L1", t_sis_center_equiv)
check("SiS: shared_vector vs bandwise 구분", t_sis_shared_vs_bandwise)
check("SiS: grad 방향(target detach)", t_sis_grad)
check("uncertainty: 양수·정규화·음수 금지", t_uncertainty)
check("uncertainty: 고오차 영역 theta 증가", t_uncertainty_direction)
check("mutual: gradient isolation", t_mutual_isolation)
check("feature KD: 사영·shape·teacher detach", t_feature_kd)
check("연산자: MTF/edge/variance/shift/schedule", t_ops)
check("WithUncertainty: forward 계약·state_dict", t_wrapper_contract)

print(f"\n[test] {len(PASS)}/{len(PASS) + len(FAIL)} 통과"
      + (f" — 실패: {', '.join(FAIL)}" if FAIL else ""))
sys.exit(1 if FAIL else 0)
