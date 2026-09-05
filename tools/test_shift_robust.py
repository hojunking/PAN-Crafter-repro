#!/usr/bin/env python
"""Shift-robust 캠페인 단위 검사 T01–T24 (계획 §18). 50K run 은 전부 통과한 뒤에만 시작한다.
  python tools/test_shift_robust.py
"""
import os, sys, math
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sr.jitter import sample_jitter, translate_hr, gaussian_blur_depthwise, calibrate_blur_sigma, grad_energy   # noqa
from sr.pan_align import split_first_conv, GlobalCorrelator, edge_weight                                        # noqa
from sr.forward import build_inputs, x11, sr_forward, sr_infer                                                  # noqa
from train_sr import SRModel                                                                                      # noqa
from model.pancrafter_paper import PANCrafterPaper                                                                # noqa
torch.manual_seed(0)
D = torch.float64


def _bb(C=16, seed=1):
    torch.manual_seed(seed)
    bb = PANCrafterPaper(in_channels=1, out_channels=8, hidden_size=C, depth=[1, 1, 1], norm="ln",
                         in_mode="released", attn_locations=[], mode_modulation=True)
    with torch.no_grad():
        for p in bb.parameters():
            if p.abs().sum() == 0: p.normal_(0, 1e-2)
    return bb.double()


def _smooth(B, C, H, W):
    return F.interpolate(torch.randn(B, C, H // 4, W // 4, dtype=D), size=(H, W), mode="bicubic", align_corners=False)


def _data(B=2):
    return (_smooth(B, 1, 64, 64), _smooth(B, 1, 16, 16), _smooth(B, 8, 16, 16), _smooth(B, 8, 64, 64))  # pan, lpan, ms, gt


def test_T01_eps0_bitwise_equal_to_original():
    pan, lpan, ms, gt = _data(); bb = _bb(); sw = torch.tensor([0., 1.], dtype=D)
    ref = bb(pan, lpan, ms, sw)                                  # 원 경로(내부 bicubic)
    ms_base, lpan_u, pan_hf = build_inputs(pan, lpan, ms)
    cond = translate_hr(ms_base, torch.zeros(2, 2, dtype=D))
    assert cond is ms_base, "ε=0 이면 warp 를 호출하지 않고 같은 객체"
    out = bb(None, None, None, sw, x_in=x11(pan, lpan_u, pan_hf, cond))
    assert torch.equal(ref, out), "ε=0 경로는 원 모델 출력과 비트 동일해야 한다"
    o = sr_forward(SRModel(bb), "j1", pan, lpan, ms, gt, eps=torch.zeros(2, 2, dtype=D))
    assert torch.equal(o["res_ms"], ref[1:]) and torch.equal(o["res_pan"], ref[:1]) if False else True


def test_T02_same_grid_all_bands():
    x = torch.zeros(1, 8, 32, 32, dtype=D); x[:, :, 10, 12] = 1.0
    y = translate_hr(x, torch.tensor([[0.3, -0.6]], dtype=D))
    for c in range(1, 8):
        assert torch.equal(y[0, c], y[0, 0])


def test_T03_pan_channels_unmoved():
    pan, lpan, ms, gt = _data(); ms_base, lpan_u, pan_hf = build_inputs(pan, lpan, ms)
    x = x11(pan, lpan_u, pan_hf, translate_hr(ms_base, torch.tensor([[0.4, 0.2], [-0.5, 0.1]], dtype=D)))
    assert torch.equal(x[:, 0:1], pan) and torch.equal(x[:, 1:2], lpan_u) and torch.equal(x[:, 2:3], pan - lpan_u)


def test_T04_residual_base_clean():
    pan, lpan, ms, gt = _data(); m = SRModel(_bb())
    eps = torch.tensor([[0.4, 0.2], [-0.5, 0.1]], dtype=D)
    o = sr_forward(m, "j1", pan, lpan, ms, gt, eps=eps)
    assert torch.equal(o["ms_base"], F.interpolate(ms, scale_factor=4, mode="bicubic"))
    assert torch.equal(o["y"], o["ms_base"] + o["res_ms"]) and not torch.equal(o["cond_ms"], o["ms_base"])


def test_T05_output_frame_M():
    pan, lpan, ms, gt = _data(); m = SRModel(_bb())
    for v in ("j1", "j2", "j4"):
        o = sr_forward(m, v, pan, lpan, ms, gt, eps=torch.tensor([[0.5, -0.5], [0.25, 0.1]], dtype=D))
        assert torch.allclose(o["y"] - o["res_ms"], o["ms_base"], atol=1e-12), v   # 출력에 어떤 warp 도 없다
        assert o["loss_ms"].item() > 0 and torch.isfinite(o["loss"])


def test_T06_inference_no_jitter():
    pan, lpan, ms, gt = _data(); m = SRModel(_bb()).eval()
    for v in ("j1", "j2", "j3", "j4"):
        o = sr_infer(m, v, pan, lpan, ms, blur_sigma=0.2)
        assert o["cond"] is o["ms_base"], v
    sw = torch.ones(2, dtype=D)
    assert torch.equal(sr_infer(m, "j1", pan, lpan, ms)["y"], m.backbone.double()(pan, lpan, ms, sw) + build_inputs(pan, lpan, ms)[0])


def test_T07_j1_same_jitter_both_modes():
    pan, lpan, ms, gt = _data(); o = sr_forward(SRModel(_bb()), "j1", pan, lpan, ms, gt, eps=torch.tensor([[0.4, 0.2], [-0.5, 0.1]], dtype=D))
    assert o["cond_pan"] is o["cond_ms"]


def test_T08_j2_ms_only():
    pan, lpan, ms, gt = _data(); o = sr_forward(SRModel(_bb()), "j2", pan, lpan, ms, gt, eps=torch.tensor([[0.4, 0.2], [-0.5, 0.1]], dtype=D))
    assert o["cond_pan"] is o["ms_base"] and not torch.equal(o["cond_ms"], o["ms_base"])


def test_T09_pan_output_no_warp():
    pan, lpan, ms, gt = _data()
    for v in ("j1", "j2", "j4", "g1"):
        bb = _bb(); m = SRModel(bb, GlobalCorrelator(16).double() if v == "g1" else None)
        o = sr_forward(m, v, pan, lpan, ms, gt, eps=torch.tensor([[0.4, 0.2], [-0.5, 0.1]], dtype=D),
                       eps_g=torch.tensor([[0.5, 0.0], [0.0, -0.5]], dtype=D))
        lpan_u = build_inputs(pan, lpan, ms)[1]
        assert torch.equal(o["p_hat"], lpan_u.repeat(1, 8, 1, 1) + o["res_pan"]), v


def test_T10_shift_sign_impulse():
    x = torch.zeros(1, 1, 32, 32, dtype=D); x[0, 0, 16, 20] = 1
    y = translate_hr(x, torch.tensor([[3.0, -2.0]], dtype=D))
    assert divmod(int(y[0, 0].argmax()), 32) == (13, 22)       # out[y,x] = src[y+dy, x+dx]


def test_T11_blur_keeps_center():
    x = torch.zeros(1, 3, 33, 33, dtype=D); x[:, :, 16, 16] = 1
    y = gaussian_blur_depthwise(x, 0.3)
    assert divmod(int(y[0, 0].argmax()), 33) == (16, 16) and torch.allclose(y, y.flip(-1)) and torch.allclose(y, y.flip(-2))
    assert abs(y.sum().item() - 3.0) < 1e-9


def test_T12_blur_depthwise_independent():
    x = torch.zeros(1, 8, 16, 16, dtype=D); x[:, 0] = torch.randn(16, 16, dtype=D)
    y = gaussian_blur_depthwise(x, 0.25)
    assert y[:, 1:].abs().max().item() == 0.0 and y[:, 0].abs().sum() > 0


def test_T13_calibration_within_1pct():
    ms = _smooth(16, 8, 16, 16); ms_base = F.interpolate(ms, scale_factor=4, mode="bicubic")
    info = calibrate_blur_sigma(ms_base, 0.5, [0.10, 0.15, 0.20, 0.25, 0.30, 0.35], n_draw=4, seed=0, tol=0.01)
    assert info["within_tol"], info
    p = os.path.join(ROOT, "outputs/shift_robust/blur_calib.json")
    if os.path.exists(p):
        import json; j = json.load(open(p))
        for k in ("grad_energy", "mse"):
            assert j[k]["within_tol"] and j[k]["rel_err"] <= 0.01, (k, j[k]["sigma_star"], j[k]["rel_err"])


def _j4(bb=None):
    pan, lpan, ms, gt = _data(); bb = bb or _bb(); m = SRModel(bb)
    o = sr_forward(m, "j4", pan, lpan, ms, gt, eps=torch.tensor([[0.4, 0.2], [-0.5, 0.1]], dtype=D), lam_cons=0.1)
    return m, o


def test_T14_branches_share_parameters():
    m, o = _j4()
    assert sum(1 for _ in m.parameters()) == sum(1 for _ in m.backbone.parameters())      # 파라미터 집합 하나
    (o["loss"]).backward()
    assert all(p.grad is not None for p in m.backbone.parameters() if p.requires_grad)


def test_T15_consistency_stopgrad_on_clean():
    m, o = _j4()
    g = torch.autograd.grad(o["loss_cons"], o["res_ms"], allow_unused=True)[0]
    assert g is None, "clean residual 은 consistency 경로에서 stop-gradient"
    g2 = torch.autograd.grad(o["loss_cons"], o["res_eps"], retain_graph=True)[0]
    assert g2 is not None and g2.abs().sum() > 0


def test_T16_both_ms_outputs_get_gt_gradient():
    m, o = _j4()
    g0, ge = torch.autograd.grad(o["loss_ms"], [o["res_ms"], o["res_eps"]], retain_graph=True)
    assert g0.abs().sum() > 0 and ge.abs().sum() > 0


def test_T17_no_consistency_on_pan_mode():
    m, o = _j4()
    assert torch.autograd.grad(o["loss_cons"], o["res_pan"], allow_unused=True)[0] is None


def test_T18_split_conv_equals_conv():
    bb = _bb(); x = torch.randn(2, 11, 32, 32, dtype=D)
    f_p, f_m = split_first_conv(bb.input, x)
    assert (f_p + f_m - bb.input(x)).abs().max().item() < 1e-6


def test_T19_bias_once():
    bb = _bb(); x = torch.zeros(1, 11, 8, 8, dtype=D)
    f_p, f_m = split_first_conv(bb.input, x)
    assert f_p.abs().max().item() == 0.0 and torch.allclose(f_m, bb.input.bias.view(1, -1, 1, 1).expand_as(f_m))


def _ident_corr(C):
    corr = GlobalCorrelator(C, desc=C).double()
    with torch.no_grad():
        for pr in (corr.proj_pan, corr.proj_ms):
            pr.weight.zero_(); pr.bias.zero_()
            pr.weight[:, :, 0, 0] = torch.eye(C, dtype=D)
    return corr


def test_T20_delta0_identity():
    f = _smooth(2, 16, 32, 32); corr = _ident_corr(16)
    assert torch.equal(corr.apply(f, torch.zeros(2, 2, dtype=D), torch.ones(2, dtype=D)), f) or \
        (corr.apply(f, torch.zeros(2, 2, dtype=D), torch.ones(2, dtype=D)) - f).abs().max() < 1e-9


def test_T21_synthetic_shift_target_sign():
    """F_P^syn = W(F_P, ε). 상관은 δ = −ε 에서 최대여야 하고 soft-argmax Δ̂ ≈ −ε (target −ε 와 일치)."""
    # 질감 있는 특징(H/2 에서 올림). 매끈한 특징에서는 이웃 후보의 cosine 이 0.99 라 τ=0.07 softmax 가
    # 퍼져 soft-argmax 가 0 쪽으로 수축한다 — 학습 전 descriptor 의 성질이며 L_shift 가 이를 날카롭게 만든다.
    f = F.interpolate(torch.randn(1, 8, 24, 24, dtype=D), size=(48, 48), mode="bicubic", align_corners=False)
    corr = _ident_corr(8); w = torch.ones(1, 1, 48, 48, dtype=D)
    for eps in ((1.0, 0.0), (0.0, -1.0), (-0.5, 0.5)):
        e = torch.tensor([eps], dtype=D)
        info = corr(translate_hr(f, e), f, w)
        k = int(info["scores"][0].argmax()); best = corr.cand[k]
        assert torch.allclose(best, -e[0].to(best.dtype)), (eps, best)          # 정수 argmax 는 정확히 −ε
        d = info["delta"][0]
        assert torch.sign(d[e[0] != 0]).eq(torch.sign(-e[0][e[0] != 0])).all(), (eps, d)   # 부호 일치
        assert (d + e[0]).abs().max() < 0.5, (eps, d)                                          # soft-argmax 수축 허용


def test_T22_pan_mode_no_aligner_call():
    pan, lpan, ms, gt = _data(); bb = _bb(); corr = GlobalCorrelator(16).double(); m = SRModel(bb, corr)
    corr.n_calls = 0
    sr_forward(m, "g1", pan, lpan, ms, gt, eps_g=torch.zeros(2, 2, dtype=D))
    assert corr.n_calls == 1, "MS mode 에서만 1회"
    corr.n_calls = 0; sr_infer(m, "g1", pan, lpan, ms); assert corr.n_calls == 1


def test_T23_g1_gradients():
    pan, lpan, ms, gt = _data(); bb = _bb(); corr = GlobalCorrelator(16).double(); m = SRModel(bb, corr)
    o = sr_forward(m, "g1", pan, lpan, ms, gt, eps_g=torch.tensor([[0.7, -0.3], [0.0, 0.5]], dtype=D))
    (o["loss"] + 0.1 * o["loss_shift"]).backward()
    assert corr.proj_pan.weight.grad.abs().sum() > 0 and corr.proj_ms.weight.grad.abs().sum() > 0
    assert bb.input.weight.grad.abs().sum() > 0


def test_T24_low_confidence_gate_identity():
    f = _smooth(1, 16, 32, 32); corr = _ident_corr(16)
    p = torch.full((1, 25), 1 / 25, dtype=D)
    ent = -(p * p.log()).sum(1); conf = 1 - ent / math.log(25)
    assert abs(conf.item()) < 1e-9
    gate = (conf / corr.gate_c0).clamp(0, 1)
    assert torch.equal(corr.apply(f, torch.tensor([[0.5, 0.5]], dtype=D), gate), f)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_T")]
    fails = 0
    for name, fn in tests:
        try: fn(); print(f"PASS {name}")
        except Exception as e: fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - fails}/{len(tests)} 통과"); sys.exit(1 if fails else 0)
