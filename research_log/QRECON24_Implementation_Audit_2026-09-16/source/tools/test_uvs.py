#!/usr/bin/env python
"""UVS-KD 단위 검사 (계획 §4.1 impulse/known-shift, §5.5 cache 증강 변환, §6–§8 손실·스케줄·게이트).
  python tools/test_uvs.py
"""
import os, sys, math
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from uvs.shift import ShiftModule, edge_rep, warp, warp_pan_channels, gated_delta                  # noqa
from uvs.losses import (gt_residual_variance, percentile_normalize, variance_weight, routed_losses,   # noqa
                        shift_kd_loss, teacher_forcing_eta)
from feeders.feeder_uvs import transform_delta_np                                                    # noqa
torch.manual_seed(0); D = torch.float64


def _tex(B, C, H, W, f=2):
    return F.interpolate(torch.randn(B, C, H // f, W // f, dtype=D), size=(H, W), mode="bicubic", align_corners=False)


def _ident_shift(radius=3):
    m = ShiftModule(channels=(4,), radius=radius).double()
    with torch.no_grad():
        for enc in (m.enc_p, m.enc_m):
            enc[0].weight.zero_(); enc[0].bias.zero_(); enc[0].weight[:, 0, 1, 1] = 1.0
    return m


def test_U01_warp_sign_impulse():
    x = torch.zeros(1, 1, 32, 32, dtype=D); x[0, 0, 16, 20] = 1
    y = warp(x, torch.tensor([[3.0, -2.0]], dtype=D))
    assert divmod(int(y[0, 0].argmax()), 32) == (13, 22)            # W(I,δ)(y,x) = I(y+δy, x+δx)


def test_U02_lr_to_pan_scale_4x():
    x = torch.zeros(1, 1, 64, 64, dtype=D); x[0, 0, 32, 32] = 1
    y = warp(x, torch.tensor([[1.0, 0.0]], dtype=D), scale=4.0)
    assert divmod(int(y[0, 0].argmax()), 64) == (28, 32), "LR 1px = PAN 4px"
    assert warp(x, torch.zeros(1, 2, dtype=D), 4.0) is x                # identity 는 호출 없이 원 객체


def test_U03_identity_predicts_center():
    m = _ident_shift(); e = edge_rep(_tex(2, 1, 24, 24))
    o = m(e, e)
    assert o["delta"].abs().max() < 1e-6 and o["p_center"].min() > 0.9 and o["conf"].min() > 0.8


def test_U04_shift_module_sign_and_magnitude():
    """E_M = W(E_P, Δ) 이면 δ_{MS←PAN} = Δ (정수 argmax 정확, soft 는 0.05 이내)."""
    m = _ident_shift(3); e_p = edge_rep(_tex(1, 1, 32, 32))
    for dy, dx in ((1, 0), (0, -2), (-3, 3), (2, 1)):
        e_m = warp(e_p, torch.tensor([[dy, dx]], dtype=D))
        o = m(e_p, e_m); k = int(o["scores"][0].argmax())
        assert m.cand[k].tolist() == [float(dy), float(dx)], (dy, dx, m.cand[k])
        assert (o["delta"][0] - torch.tensor([dy, dx], dtype=D)).abs().max() < 0.05


def test_U05_cache_vector_transform_matches_map_transform():
    """impulse 쌍: HR map 과 δ 를 같은 flip/rot 로 변환하면 정합이 유지된다 (§5.5 규칙 검증)."""
    H = 24
    for hf in (0, 1):
        for vf in (0, 1):
            for rot in range(4):
                ref = torch.zeros(H, H, dtype=D); ref[9, 14] = 1
                dy, dx = 2, -3                                       # mov(y,x) = ref(y+dy, x+dx) 관계: mov = W(ref, δ)
                mov = torch.roll(ref, shifts=(-dy, -dx), dims=(0, 1))
                def aug(x):
                    if hf: x = x.flip(1)
                    if vf: x = x.flip(0)
                    return torch.rot90(x, rot, (0, 1))
                d2 = transform_delta_np(np.array([dy, dx], np.float32), hf, vf, rot)
                back = torch.roll(aug(mov), shifts=(int(round(d2[0])), int(round(d2[1]))), dims=(0, 1))   # W(mov', δ') = ref'
                assert torch.equal(back, aug(ref)), (hf, vf, rot, d2)


def test_U06_warp_pan_channels_consistent():
    pan, lpan_u, hf = _tex(2, 1, 64, 64), _tex(2, 1, 64, 64, 8), _tex(2, 1, 64, 64)
    d = torch.tensor([[0.3, -0.2], [0.0, 0.0]], dtype=D)
    a, b, c = warp_pan_channels(pan, lpan_u, hf, d)
    assert torch.allclose(a[1], pan[1]) and torch.allclose(b[1], lpan_u[1])             # δ=0 표본은 그대로
    assert torch.allclose(a[0], warp(pan[:1], d[:1], 4.0)[0])
    p0, l0, h0 = warp_pan_channels(pan, lpan_u, hf, torch.zeros(2, 2, dtype=D))
    assert p0 is pan and l0 is lpan_u and h0 is hf


def test_U07_variance_weight_mean_one():
    gt, lms = _tex(3, 8, 64, 64), _tex(3, 8, 64, 64, 8)
    v = percentile_normalize(gt_residual_variance(gt, lms, 5), 0.001, 0.05)
    w = variance_weight(v, 1.0)
    assert (w.mean(dim=(1, 2, 3)) - 1).abs().max() < 1e-9 and w.min() > 0 and v.min() >= 0 and v.max() <= 1


def test_U08_routing_weights_and_stopgrad():
    r_s = _tex(2, 8, 32, 32).requires_grad_(True); r_gt = _tex(2, 8, 32, 32); r_t = _tex(2, 8, 32, 32)
    u = torch.rand(2, 1, 32, 32, dtype=D)
    hard, soft = routed_losses(r_s, r_gt, r_t.detach(), u.detach())
    assert hard > 0 and soft > 0
    g = torch.autograd.grad(hard + soft, r_s)[0]; assert g.abs().sum() > 0
    # u=1 이면 soft 가 0, u=0 이면 hard 가중 1
    h1, s1 = routed_losses(r_s, r_gt, r_t, torch.ones_like(u)); assert s1.item() == 0.0
    h0, s0 = routed_losses(r_s, r_gt, r_t, torch.zeros_like(u)); assert abs(h0.item() - (r_s - r_gt).abs().mean().item()) < 1e-12


def test_U09_shift_kd_loss():
    d_s = torch.tensor([[0.5, -0.5]], dtype=D, requires_grad=True); c_s = torch.tensor([0.6], dtype=D, requires_grad=True)
    d_t = torch.tensor([[0.5, -0.5]], dtype=D); c_t = torch.tensor([0.6], dtype=D)
    l, vec, cf = shift_kd_loss(d_s, c_s, d_t, c_t)
    assert l.item() == 0.0
    l2, _, _ = shift_kd_loss(d_s, c_s, d_t + 1.0, c_t); assert l2 > 0
    assert torch.autograd.grad(l2, d_s)[0].abs().sum() > 0


def test_U10_teacher_forcing_schedule():
    assert teacher_forcing_eta(0) == 1.0 and teacher_forcing_eta(5000) == 1.0
    assert abs(teacher_forcing_eta(12500) - 0.5) < 1e-9 and teacher_forcing_eta(20000) == 0.0 and teacher_forcing_eta(50000) == 0.0


def test_U11_confidence_gate():
    d = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=D); c = torch.tensor([0.9, 0.2], dtype=D)
    g = gated_delta(d, c, 0.35)
    assert torch.allclose(g[0], 0.9 * d[0]) and g[1].abs().sum() == 0
    assert torch.allclose(gated_delta(d, c), c.unsqueeze(1) * d)          # 학습 시(threshold 없음) ĉ·δ


def test_U12_edge_rep_uses_provided_lpan_grid():
    """flip 표본에서 provided lpan(LR 증강) 과 ms(LR 증강) 의 격자가 같아야 shift 입력이 어긋나지 않는다 —
    LR 두 텐서를 같은 flip 으로 뒤집으면 δ 도 정확히 변환된 값이 나온다 (phase 문제 없음)."""
    m = _ident_shift(3); e_p = edge_rep(_tex(1, 1, 32, 32)); dy, dx = 1, -2
    e_m = warp(e_p, torch.tensor([[dy, dx]], dtype=D))
    o = m(e_p.flip(-1), e_m.flip(-1))                                       # hflip 둘 다
    exp = transform_delta_np(np.array([dy, dx], np.float32), 1, 0, 0)
    assert (o["delta"][0] - torch.tensor(exp, dtype=D)).abs().max() < 0.05, (o["delta"], exp)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_U")]
    fails = 0
    for name, fn in tests:
        try: fn(); print(f"PASS {name}")
        except Exception as e: fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - fails}/{len(tests)} 통과"); sys.exit(1 if fails else 0)
