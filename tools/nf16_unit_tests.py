#!/usr/bin/env python
"""NF16 실행 전 gate (명세 §9 G0–G2 중 코드로 닫는 것). 하나라도 실패하면 exit 1.   python tools/nf16_unit_tests.py"""
import os, sys, json, copy
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from pa.aligner import PANGlobalAligner
from pa.model import PAModel
from pa.offset import sample_offsets, offset_loss, predict_c, lambda_off, aligner_margin
from pa.losses import direct_geometry_loss, geometry_support_margin
from pa.warp import warp_pan, support_margin_ok
from kdv.forward import kdv_forward
from kdv.protocol import prepare_view, is_corrupt_update
from kdv.registry import resolve
from kdv.teacher_assets import load_donor_aligner, state_hash, freeze, assert_param_disjoint, sha256_file
from main import import_class

FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)

torch.manual_seed(0); torch.set_num_threads(4)
DONOR = "PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT"; dsrc = os.path.join(ROOT, "work_dir", DONOR, "last")
# ---------------- G0 donor asset
have_donor = os.path.exists(os.path.join(dsrc, "model.safetensors"))
if have_donor:
    al, man = load_donor_aligner(dsrc, 8); meta = json.load(open(os.path.join(ROOT, "work_dir", DONOR, "last_meta.json")))
    check("G0 donor = N2 R200 last, step 50000, aligner 22 keys 105,330 params", meta["step"] == 50000 and man["n_keys"] == 22 and man["n_params"] == 105330, f"sha {man['aligner_tensors_sha256_16']}")
    check("G0 donor head not zero (재초기화 없음)", float(al.fc2.weight.abs().sum()) > 0 and float(al.fc2.bias.abs().sum()) > 0, f"bias {al.fc2.bias.tolist()}")
else:
    al = PANGlobalAligner(8)
    with torch.no_grad(): al.fc2.weight.normal_(0, 0.05); al.fc2.bias.add_(torch.tensor([0.2, -0.1]))
    print("  (donor checkpoint 없음 — 합성 aligner 로 G1/G2 만)")
Model = import_class("model.pancrafter_paper.PANCrafterPaper")
MA = dict(in_channels=1, out_channels=8, hidden_size=112, depth=[1, 2, 3], dropout=0.0, num_heads=8, mlp_ratio=4.0, ks=3, ka=3, norm="ln", in_mode="paper", attn_locations=[], mode_modulation=False, n_attn=3)
bb = Model(**MA)
for p_ in bb.parameters():
    if p_.abs().sum() == 0: p_.data.normal_(0, 0.01)
check("G0 W112·D123 backbone 2.6589 M", abs(sum(p.numel() for p in bb.parameters()) / 1e6 - 2.6589) < 1e-3)
mg = 4; check("G0 aligner view margin 4 = aligner_margin(R=2) (64²→56², 512²→504²)", aligner_margin(2.0) == 4)
g = torch.Generator().manual_seed(3); pan = torch.rand(4, 1, 64, 64, generator=g); ms = torch.rand(4, 8, 16, 16, generator=g); lpan = torch.rand(4, 1, 16, 16, generator=g)
gt = torch.rand(4, 8, 64, 64, generator=g) + torch.linspace(0, 1, 64)[None, None, None, :]

# ---------------- G1: P0 sampler 없음 · P1 frozen hash 불변 (optimizer step 뒤) · P2–P4 gradient 경로
m0 = PAModel(copy.deepcopy(bb), None, sampler=False); o0 = m0(pan, ms, lpan)
check("G1 P0: aligner·sampler 없음, U-Net 입력 = 원 PAN, Δ=0", m0.aligner is None and torch.equal(o0["pan_aligned"], pan) and float(o0["delta"].abs().max()) == 0)
m1 = PAModel(copy.deepcopy(bb), copy.deepcopy(al), aligner_margin=mg); freeze(m1.aligner); h1 = state_hash(m1.aligner); buf1 = {k: v.clone() for k, v in m1.aligner.state_dict().items()}
opt1 = torch.optim.AdamW([p for p in m1.backbone.parameters()], lr=1e-4); m1.train(); m1.backbone.requires_grad_(True)
o1 = kdv_forward(m1, None, pan, ms, lpan, share_correction=True, teacher_needed=False, aligner_live=False); (o1["y"] - gt).abs().mean().backward(); opt1.step()
check("G1 P1: aligner requires_grad False·optimizer 밖, optimizer step 뒤 weight·buffer hash 불변, 영상별 Δ 예측(상수 아님)", state_hash(m1.aligner) == h1 and all(torch.equal(v, m1.aligner.state_dict()[k]) for k, v in buf1.items()) and assert_param_disjoint(opt1, m1.aligner)
      and float(o1["delta"].std(0).max()) > 0 and not o1["delta"].requires_grad, f"Δ {o1['delta'][0].tolist()}")
m2 = PAModel(copy.deepcopy(bb), copy.deepcopy(al), aligner_margin=mg); m2.train()
o2 = kdv_forward(m2, None, pan, ms, lpan, share_correction=False, teacher_needed=False, aligner_live=True); lrec = (o2["y"] - gt).abs().mean()
ga = torch.autograd.grad(lrec, list(m2.aligner.parameters()), retain_graph=True, allow_unused=True)
check("G1 P2–P4: native L_rec → aligner gradient 존재 (c0 를 detach 하지 않음)", any(x is not None and float(x.abs().sum()) > 0 for x in ga) and o2["delta"].requires_grad)
# P3: 홀수 update 의 aligner 전용 offset 연습
gen = torch.Generator().manual_seed(2025 + 2000); pv, eps0, cor = prepare_view(pan, "I-AEQ", 3, 2.0, gen)
check("G1 P3: I-AEQ 는 U-Net 입력이 항상 native (prepare_view 가 P 를 그대로, corrupted False)", pv is pan and not cor and not is_corrupt_update("I-AEQ", 3))
eps = sample_offsets(4, 2.0, gen)
with torch.no_grad(): p_eps = warp_pan(pan, eps)
c_eps = predict_c(m2.aligner, p_eps, o2["ms_base"], mg); loff = offset_loss(c_eps, o2["delta"], eps, stop_reference=True)
m2.zero_grad(); loff.backward(retain_graph=True)
check("G1 P3: L_off → aligner gradient (corrupted 경로), U-Net θ gradient 없음, target sg(ĉ0)", all(p.grad is not None for p in m2.aligner.parameters()) and all(p.grad is None for p in m2.backbone.parameters()))
m2.zero_grad(); l2 = offset_loss(c_eps.detach(), o2["delta"], eps, stop_reference=True)
check("G1 P3: target 이 detach 라 ĉ0 경로로는 gradient 0", not l2.requires_grad)
check("G1 P3: λ_off 즉시 0.01 (ramp 0), PO10 ramp 5K 는 t=0 에서 2e-6", lambda_off(0, 0.01, 0) == 0.01 and abs(lambda_off(0, 0.01, 5000) - 2e-6) < 1e-12)
# P4: geometry → aligner only
m2.zero_grad(); lg, info_g = direct_geometry_loss(o2["pan_aligned"], pan, gt, 2.0, 11); lg.backward(retain_graph=True)
check("G2 P4: L_geo → aligner gradient, U-Net θ gradient 없음, 유한", any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m2.aligner.parameters()) and all(p.grad is None for p in m2.backbone.parameters()) and torch.isfinite(lg))
check("G2 P4: support guard margin 11 → 4 (Gaussian 6 + Scharr 1), 현재 Δ 로 통과", geometry_support_margin(2.0, 11) == 4 and bool(support_margin_ok(64, 64, o2["delta"].detach(), 4).all()))
# 알려진 변형 대조: GT 구조로 만든 PAN 을 δ 만큼 옮기면 L_geo 의 gradient 가 되돌리는 방향
yy, xx = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij"); img = torch.zeros(2, 1, 64, 64)
gg = torch.Generator().manual_seed(9)
for b in range(2):
    for _ in range(14):
        cy, cx, sg = 8 + 48 * torch.rand(1, generator=gg), 8 + 48 * torch.rand(1, generator=gg), 2 + 3 * torch.rand(1, generator=gg); img[b, 0] += torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sg ** 2))
gt_s = img.expand(2, 8, 64, 64).clone() * torch.linspace(0.6, 1.4, 8)[None, :, None, None]
pan_shift = warp_pan(img, torch.tensor([[1.0, 0.0], [0.0, -1.0]]))            # PAN 을 GT 대비 (dy,dx) 만큼 옮김
d_var = torch.zeros(2, 2, requires_grad=True); lgs, _ = direct_geometry_loss(warp_pan(pan_shift, d_var), pan_shift, gt_s, 2.0, 11); gd = torch.autograd.grad(lgs, d_var)[0]
check("G2 P4: 알려진 변형 대조 — GT 구조로 만든 PAN 을 옮기면 L_geo gradient 가 되돌리는 부호 (−∇ ∥ 정답 −δ)", float(gd[0, 0]) > 0 and float(gd[1, 1]) < 0 and abs(float(gd[0, 1])) < abs(float(gd[0, 0])) and abs(float(gd[1, 0])) < abs(float(gd[1, 1])), f"grad {gd.tolist()}")
# 한 update 에 U-Net forward 1회 (P3 의 offset 연습은 aligner 만)
calls = {"n": 0}; orig = m2.backbone.forward
def counting(*a, **kw): calls["n"] += 1; return orig(*a, **kw)
m2.backbone.forward = counting; o3 = kdv_forward(m2, None, pan, ms, lpan, share_correction=False, teacher_needed=False, aligner_live=True); _ = predict_c(m2.aligner, p_eps, o3["ms_base"], mg)
check("G1 P3/P4: update 당 U-Net forward 1회, P_ε 는 aligner 에만", calls["n"] == 1)
# ε sampler 통계 (b=2): mean 0, 축 std ≈ 1, 평균 반경 4/3, |ε| ≤ 2
e = sample_offsets(40000, 2.0, torch.Generator().manual_seed(1))
check("G1 ε ~ disk(2): mean≈0, axis std≈1, mean radius≈4/3, max ≤ 2", float(e.mean(0).abs().max()) < 0.02 and abs(float(e.std(0).mean()) - 1.0) < 0.02 and abs(float(e.norm(dim=1).mean()) - 4 / 3) < 0.02 and float(e.norm(dim=1).max()) <= 2 + 1e-6)
gA = torch.Generator().manual_seed(1234 + 2000); gB = torch.Generator().manual_seed(1234 + 2000)
check("G1 P3/P4 같은 seed → 같은 ε 열 (같은 홀수 update 에서만 sampling)", torch.equal(sample_offsets(48, 2.0, gA), sample_offsets(48, 2.0, gB)))
# resolver: NF16 case 들
from tools.gen_nf16_configs import build
specs = {q: resolve(build(q, 1234, None, 50000, True, None)) for q in range(5)}
check("resolver: P0 A-ID·I-A / P1 A-FR / P2 A-FT / P3 I-AEQ off 0.01 ramp 0 geo 0 / P4 geo 0.01 ramp 5K",
      specs[0]["policy"] == "A-ID" and specs[1]["policy"] == "A-FR" and not specs[1]["aligner_trainable"] and specs[2]["aligner_trainable"] and specs[2]["offset_weight_effective"] == 0
      and specs[3]["protocol"] == "I-AEQ" and specs[3]["offset_weight_effective"] == 0.01 and specs[3]["offset_ramp_updates"] == 0 and specs[3]["geometry_weight_effective"] == 0
      and specs[4]["geometry_weight_effective"] == 0.01 and specs[4]["aux_ramp_updates"] == 5000 and all(not s["needs_teacher"] for s in specs.values()))
print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
