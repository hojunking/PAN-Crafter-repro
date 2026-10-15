#!/usr/bin/env python
"""PO10 실행 전 gate G2–G7 (research_log/PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md §12). 하나라도 실패하면 exit 1.

    python tools/po10_unit_tests.py
"""
import os, sys, math
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from pa.aligner import PANGlobalAligner
from pa.model import PAModel
from pa.offset import sample_offsets, aligner_margin, valid_view, predict_c, offset_loss, po_step, lambda_off
from pa.warp import warp_pan, warp_support_mask
from main import import_class

FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)

torch.manual_seed(0); R = 1.0
# ---------------- G2 sampling
g = torch.Generator(device="cpu"); g.manual_seed(0); e = sample_offsets(20000, R, g)
check("G2 disk: mean ≈ 0", float(e.mean(0).abs().max()) < 0.02, f"mean {e.mean(0).tolist()}")
check("G2 disk: axis sd ≈ R/2", abs(float(e.std(0)[0]) - 0.5) < 0.02 and abs(float(e.std(0)[1]) - 0.5) < 0.02, f"sd {e.std(0).tolist()}")
check("G2 disk: max L2 ≤ R", float(e.norm(dim=1).max()) <= R + 1e-6, f"max {float(e.norm(dim=1).max()):.5f}")
check("G2 disk: area-uniform (P(|ε|<R/2) ≈ 1/4)", abs(float((e.norm(dim=1) < R / 2).float().mean()) - 0.25) < 0.02)
g2 = torch.Generator(device="cpu"); g2.manual_seed(0); check("G2 dedicated RNG reproducible", torch.equal(sample_offsets(20000, R, g2), e))
# ---------------- G3 sign / axis / zero-warp gradient
img = torch.zeros(1, 1, 64, 64); img[0, 0, 30, 20] = 1.0
o = warp_pan(img, torch.tensor([[1.0, 0.0]])); check("G3 dy=+1 moves content up (peak at y=29)", o[0, 0].argmax().item() == 29 * 64 + 20)
o = warp_pan(img, torch.tensor([[0.0, 1.0]])); check("G3 dx=+1 moves content left (peak at x=19)", o[0, 0].argmax().item() == 30 * 64 + 19)
d = torch.zeros(2, 2, requires_grad=True); warp_pan(torch.rand(2, 1, 64, 64), d).mean().backward(); check("G3 zero-warp gradient exists", d.grad is not None and torch.isfinite(d.grad).all())
# ---------------- model: A1 backbone + aligner with margin
Model = import_class("model.pancrafter_paper.PANCrafterPaper")
bb = Model(in_channels=1, out_channels=8, hidden_size=96, depth=[1, 2, 4], dropout=0.0, num_heads=8, mlp_ratio=4.0, ks=3, ka=3, norm="ln", in_mode="paper", attn_locations=[], mode_modulation=False)
for p_ in bb.parameters():
    if p_.abs().sum() == 0: p_.data.normal_(0, 0.01)
al = PANGlobalAligner(8)
with torch.no_grad(): al.fc2.weight.add_(0.02 * torch.randn_like(al.fc2.weight)); al.fc2.bias.add_(torch.tensor([0.2, -0.1]))
m = PAModel(bb, al, aligner_margin=aligner_margin(R))
check("§5 margin rule R=1 -> 4", aligner_margin(1.0) == 4 and aligner_margin(2.5) == 8)
pan, ms, lpan, gt = torch.rand(4, 1, 64, 64), torch.rand(4, 8, 16, 16), torch.rand(4, 1, 16, 16), torch.rand(4, 8, 64, 64)
# ---------------- G4 / G5 gradient routing
def grads(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(sum((x.abs().sum() for x in gs if x is not None), torch.zeros(())))
gen = torch.Generator(device="cpu"); gen.manual_seed(1)
ap = list(al.parameters()); up = list(bb.parameters())
tot, info = po_step(m, pan, ms, lpan, gt, case="N2_SG", update_index=1, radius_hr=R, generator=gen, lam_max=0.01, ramp=1)
check("G4 N2_SG: c0 has no grad (stop-gradient)", not info["c0"].requires_grad)
check("G5 N2: L_off -> aligner, not U-Net", grads(info["off"], ap) > 0 and grads(info["off"], up) == 0.0)
check("G5 N2: L_rec -> aligner and U-Net (corrupt step)", grads(info["rec"], ap) > 0 and grads(info["rec"], up) > 0)
gen.manual_seed(1); tot3, info3 = po_step(m, pan, ms, lpan, gt, case="N3_NOSG", update_index=1, radius_hr=R, generator=gen, lam_max=0.01, ramp=1)
check("G4 N3_NOSG: c0 connected (grad flows to aligner through both calls)", info3["c0"].requires_grad and grads(info3["off"], ap) > 0)
check("G4 same forward loss value N2 vs N3 (same ε)", abs(float(info["off"]) - float(info3["off"])) < 1e-6 and torch.equal(info["eps"], info3["eps"]))
# N2 vs N3 gradient differ (reference path)
g_n2 = torch.autograd.grad(info["off"], ap, retain_graph=True, allow_unused=True); g_n3 = torch.autograd.grad(info3["off"], ap, retain_graph=True, allow_unused=True)
diff = float(sum(((a - b).abs().sum() for a, b in zip(g_n2, g_n3) if a is not None and b is not None), torch.zeros(())))
check("G4 N2 and N3 aligner gradients differ", diff > 0)
gen.manual_seed(1); tot1, info1 = po_step(m, pan, ms, lpan, gt, case="N1", update_index=1, radius_hr=R, generator=gen, lam_max=0.01, ramp=1)
check("N1: offset coefficient exactly 0, loss == rec", info1["weight"] == 0.0 and float(tot1) == float(info1["rec"]))
totn, infon = po_step(m, pan, ms, lpan, gt, case="N2_SG", update_index=0, radius_hr=R, generator=gen, lam_max=0.01, ramp=1)
check("G4 native step: no corruption, rec -> aligner connected, weight 0", (not infon["corrupt"]) and float(infon["eps"].abs().max()) == 0 and grads(infon["rec"], ap) > 0 and infon["weight"] == 0.0)
check("§4.7 lambda ramp", abs(lambda_off(0, 0.01, 5000) - 0.01 / 5000) < 1e-12 and lambda_off(4999, 0.01, 5000) == 0.01 and lambda_off(50000, 0.01, 5000) == 0.01)
check("§4.3 offset loss zero when c_eps = c0 - eps (any c0)", float(offset_loss(torch.tensor([[0.2, -0.1]]) - torch.tensor([[0.6, 0.3]]), torch.tensor([[0.2, -0.1]]), torch.tensor([[0.6, 0.3]]), True)) < 1e-7)
# ---------------- G6 valid view: nominal 4-tap support for all |ε| ≤ R inside the crop, crop-then-normalize, padding independence
mg = aligner_margin(R); H = W = 64
worst = torch.tensor([[R, 0.0], [-R, 0.0], [0.0, R], [0.0, -R], [R / math.sqrt(2), R / math.sqrt(2)], [-R / math.sqrt(2), -R / math.sqrt(2)]])
msk = warp_support_mask(H, W, worst)
check("G6 crop [4:60] valid for all |ε| ≤ R (4-tap)", bool(msk[:, :, mg:H - mg, mg:W - mg].all()))
pe = warp_pan(pan, torch.tensor([[0.9, -0.7]] * 4))
c_border = predict_c(al, pe, torch.nn.functional.interpolate(ms, scale_factor=4, mode="bicubic"), mg)
import torch.nn.functional as F
def warp_reflect(p, d):        # 같은 warp, padding 만 reflection
    B, _, h, w = p.shape; ys = torch.arange(h, dtype=torch.float32); xs = torch.arange(w, dtype=torch.float32); yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gx = 2.0 * (xx[None] + d[:, 1].view(B, 1, 1) + 0.5) / w - 1.0; gy = 2.0 * (yy[None] + d[:, 0].view(B, 1, 1) + 0.5) / h - 1.0
    return F.grid_sample(p, torch.stack((gx, gy), -1), mode="bicubic", padding_mode="reflection", align_corners=False)
pe2 = warp_reflect(pan, torch.tensor([[0.9, -0.7]] * 4))
c_refl = predict_c(al, pe2, torch.nn.functional.interpolate(ms, scale_factor=4, mode="bicubic"), mg)
check("G6/§10.4-1 aligner output independent of outer padding (border vs reflection)", float((c_border - c_refl).abs().max()) < 1e-5, f"max diff {float((c_border - c_refl).abs().max()):.2e}")
check("G6 interior crop identical under both paddings", torch.allclose(valid_view(pe, mg), valid_view(pe2, mg), atol=1e-6))
# ---------------- G7 sequential two warps: integer roundtrip on interior
x = torch.rand(1, 1, 64, 64); e_int = torch.tensor([[2.0, -3.0]]); back = warp_pan(warp_pan(x, e_int), -e_int)
check("G7 integer shift roundtrip exact on interior", float((back[..., 8:-8, 8:-8] - x[..., 8:-8, 8:-8]).abs().max()) < 1e-5)
e_sub = torch.tensor([[0.6, -0.4]]); back = warp_pan(warp_pan(x, e_sub), -e_sub)
check("G7 subpixel roundtrip is NOT lossless (double interpolation floor recorded)", float((back[..., 8:-8, 8:-8] - x[..., 8:-8, 8:-8]).abs().max()) > 1e-4, f"max diff {float((back[..., 8:-8, 8:-8] - x[..., 8:-8, 8:-8]).abs().max()):.3e}")
# eval path: PAModel with margin at 512 (view 504) and 256 (view 248)
m.eval()
with torch.no_grad():
    for H_ in (256, 512):
        o = m(torch.rand(1, 1, H_, H_), torch.rand(1, 8, H_ // 4, H_ // 4), torch.rand(1, 1, H_ // 4, H_ // 4))
        check(f"PAModel margin at {H_}² (view {H_ - 2 * mg}²) forward finite", o["y"].shape[-2:] == (H_, H_) and torch.isfinite(o["delta"]).all())
# ---------------- 2차 검토: 두 단계 support · RNG 재개 · 진단 λ
from pa.warp import two_stage_support_mask
from pa.evalviews import STRESS_MARGIN, MAX_ELIGIBLE_TWO_STAGE
def warp_pad(p, d, padding):
    B, _, h, w = p.shape; ys = torch.arange(h, dtype=torch.float32); xs = torch.arange(w, dtype=torch.float32); yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gx = 2.0 * (xx[None] + d[:, 1].view(B, 1, 1) + 0.5) / w - 1.0; gy = 2.0 * (yy[None] + d[:, 0].view(B, 1, 1) + 0.5) / h - 1.0
    return F.grid_sample(p, torch.stack((gx, gy), -1), mode="bicubic", padding_mode=padding, align_corners=False)
x = torch.rand(1, 1, 64, 64); e_ = torch.tensor([[0.7, -0.6]]); c_ = torch.tensor([[7.0, 0.0]])
a_b = warp_pad(warp_pad(x, e_, "border"), c_ - e_, "border"); a_r = warp_pad(warp_pad(x, e_, "reflection"), c_ - e_, "reflection")
common = two_stage_support_mask(64, 64, e_, c_ - e_)
check("review-1 two-stage support: inside mask, padding-independent", float((a_b - a_r).abs()[common].max()) < 1e-6, f"inside {float((a_b - a_r).abs()[common].max()):.1e}, frac {float(common.float().mean()):.3f}")
check("review-1 two-stage support: old fixed [8:-8] crop DOES contain padding-dependent pixels for c0=(7,0)", float((a_b - a_r).abs()[..., 8:-8, 8:-8].max()) > 1e-4)
check("review-1 stress ROI eligibility bound", STRESS_MARGIN == 96 and MAX_ELIGIBLE_TWO_STAGE == 96 - 8 - 44 - 4)
from train_po import RNGState, diag_update_index
g1 = torch.Generator(device="cpu"); g1.manual_seed(5); _ = sample_offsets(48, R, g1)
st = RNGState(g1).state_dict(); a_seq = sample_offsets(48, R, g1)
g2 = torch.Generator(device="cpu"); g2.manual_seed(999); RNGState(g2).load_state_dict(st); b_seq = sample_offsets(48, R, g2)
check("review-① corruption RNG state save/restore continues the ε sequence", torch.equal(a_seq, b_seq))
check("review-② diagnostic λ uses the real step (parity kept)", diag_update_index(20000, 1) == 20001 and diag_update_index(20001, 1) == 20001 and diag_update_index(20000, 0) == 20000
      and abs(lambda_off(diag_update_index(20000, 1), 0.01, 5000) - 0.01) < 1e-12)
# ---------------- R200 (변경 명세 §4·§6·§9)
g = torch.Generator(device="cpu"); g.manual_seed(0); e2 = sample_offsets(20000, 2.0, g)
check("R200 disk: axis sd ≈ 1.0, mean length ≈ 1.333, max ≤ 2, P(|ε|≤1) ≈ 0.25",
      abs(float(e2.std(0)[0]) - 1.0) < 0.03 and abs(float(e2.norm(dim=1).mean()) - 4 / 3) < 0.03 and float(e2.norm(dim=1).max()) <= 2 + 1e-6 and abs(float((e2.norm(dim=1) <= 1).float().mean()) - 0.25) < 0.02)
check("R200 margin rule 4·ceil((2+2)/4) = 4 (train view 56², FR view 504²)", aligner_margin(2.0) == 4)
worst2 = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0], [2 / math.sqrt(2), 2 / math.sqrt(2)], [-2 / math.sqrt(2), -2 / math.sqrt(2)]])
check("R200 crop [4:60] valid for all |ε| ≤ 2 (4-tap)", bool(warp_support_mask(64, 64, worst2)[:, :, 4:60, 4:60].all()))
g.manual_seed(7); a1 = sample_offsets(48, 1.0, g); g.manual_seed(7); a2 = sample_offsets(48, 2.0, g)
check("same seed: R200 ε = 2 × R100 ε (기초 난수 동일)", torch.allclose(a2, 2 * a1))
print("\n" + ("전부 통과" if not FAIL else f"실패 {len(FAIL)}: {FAIL}"))
sys.exit(1 if FAIL else 0)
