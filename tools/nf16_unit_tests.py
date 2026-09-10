#!/usr/bin/env python
"""NF16 실행 전 gate (명세 §9 G0–G2 중 코드로 닫는 것). 하나라도 실패하면 exit 1.   python tools/nf16_unit_tests.py"""
import os, sys, json, copy, yaml
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

# ---------------- 검토(2026-09-11) 5건의 gate
from train_kdv import KDVTrainer
from tools.po10_diag import native_stress_eligible, aggregate_native
# (2) 예산: 반복 pair 는 둘 다 50K 갈 예산이 있을 때만 — 사용자 반례 used 13.5h, run 당 1h: 단일 검사면 통과, pair 검사면 16.9 > 16 → DEFERRED
d1 = KDVTrainer.budget_decision(13.5, 1.0, [], 1.0, 16.0, required=False, proj_pair=0.0); d2 = KDVTrainer.budget_decision(13.5, 1.0, [], 1.0, 16.0, required=False, proj_pair=1.0)
check("review-2 budget: pair 검사 (P3+P4 7777) 가 단일 검사와 다르게 DEFERRED", d1["decision"] == "RUN" and d2["decision"] == "DEFERRED_BUDGET" and abs(d2["projected_total_hours"] - 16.9) < 1e-9)
d3 = KDVTrainer.budget_decision(2.0, 1.6, [1.6, 1.6, 1.6, 1.6], 1.0, 16.0, required=True); d4 = KDVTrainer.budget_decision(9.0, 1.6, [1.6, 1.6, 1.6, 1.6], 1.0, 16.0, required=True)
check("review-2 budget: 필수 run 은 남은 필수 사슬을 포함해 예상하고(12.6h OK), 초과면 경고만 (RUN + warn)", d3["decision"] == "RUN" and not d3["warn"] and d4["decision"] == "RUN" and d4["warn"])
check("review-2 budget: nan 예상은 비필수 run 을 막는다", KDVTrainer.budget_decision(0.0, float("nan"), [], 1.0, 16.0, required=False)["decision"] == "DEFERRED_BUDGET")
# (2b) 예상치 출처: 지정 run 은 projected_map(§10 예약) → 이미 끝난 run 은 0 (used 에 실비가 있으므로 이중 계상 금지) → 없으면 완료 평균 → throughput
class _B:
    run_id = "NF16_P3_W112_D123_WV3_S7777_N2LAST_v1"
    budget = dict(projected_map={"NF16_P4_W112_D123_WV3_S7777_N2LAST_v1": 2.0, run_id: 2.0}, projected_hours=None)
    _projection = KDVTrainer._projection
_led = dict(total_gpu_hours=16.0, entries={"NF16_P0_W112_D123_WV3_S1234_N2LAST_v1": dict(kind="run", status="FINISHED_EXPORT", hours_total=1.7)}, throughput=dict(projected_run_hours_50k=1.625))
check("review-2 budget: projected_map 이 case 별 예약을 준다 (완료 평균 1.7 이 아니라 2.0)", abs(_B._projection(_B, _led, "NF16_P4_W112_D123_WV3_S7777_N2LAST_v1") - 2.0) < 1e-12)
check("review-2 budget: 끝난 pair 는 0 (used 이중 계상 금지)", _B._projection(_B, _led, "NF16_P0_W112_D123_WV3_S1234_N2LAST_v1") == 0.0)
_led2 = dict(total_gpu_hours=16.0, entries={}, throughput=dict(projected_run_hours_50k=1.625))
class _C(_B): budget = dict()
check("review-2 budget: map·완료 run 이 없으면 smoke throughput", abs(_C._projection(_C, _led2, "X") - 1.625) < 1e-12)
# (2c) 재시작: RUNNING 으로 남은 죽은 시도(OOM/재부팅) 도 used 에 들어간다
import tempfile, types
_wd = tempfile.mkdtemp(); json.dump(dict(elapsed_hours=1.3), open(os.path.join(_wd, "memory_and_throughput.json"), "w"))
class _D:
    args = types.SimpleNamespace(work_dir=_wd)
    _crashed_hours = KDVTrainer._crashed_hours
check("review-2 budget: 죽은 시도의 GPU 시간을 memory_and_throughput.json 에서 회수 (1.3h)", abs(_D._crashed_hours(_D, dict(started="2026-09-11T01:00:00")) - 1.3) < 1e-9)
# 반복 pair 는 양쪽 config 가 서로를 가리킨다 (한쪽만 검사하면 P4 만 도는 상황이 생긴다)
_cfgs = {q: os.path.join(ROOT, "config", f"NF16_P{q}_W112_D123_WV3_S7777_N2LAST_v1.yaml") for q in (3, 4)}
if all(os.path.exists(v) for v in _cfgs.values()):
    _b = {q: yaml.safe_load(open(v))["kdv"]["budget"] for q, v in _cfgs.items()}
    check("review-2 budget: 반복 pair(P3·P4 s7777) 가 서로를 pair_with 로 가리킨다",
          _b[3]["pair_with"].endswith("P4_W112_D123_WV3_S7777_N2LAST_v1") and _b[4]["pair_with"].endswith("P3_W112_D123_WV3_S7777_N2LAST_v1")
          and all(_b[q]["projected_map"] for q in (3, 4)))
# (4) donor 로드가 전역 RNG 를 소비하지 않는다 → P0(donor 없음) 과 P1–P4 의 데이터 순서가 같다
if have_donor:
    st0 = torch.random.get_rng_state(); _ = load_donor_aligner(dsrc, 8); st1 = torch.random.get_rng_state()
    check("review-4 donor load leaves global torch RNG unchanged (DataLoader 순서 동일)", torch.equal(st0, st1))
# (5) 진단 step: I-AEQ 면 diag_every 배수의 다음 홀수 step 도
class _P: protocol = "I-AEQ"; diag_every = 1000
class _Q: protocol = "I-A"; diag_every = 1000
check("review-5 diag step covers the offset-exercise (odd) step for I-AEQ only", KDVTrainer.is_diag_step(_P, 1000) and KDVTrainer.is_diag_step(_P, 1001) and not KDVTrainer.is_diag_step(_P, 1002) and not KDVTrainer.is_diag_step(_Q, 1001))
# (3) native stress 적격: 두 단계 support + 고정 참조 support; 집계는 전부 적격일 때만
check("review-3 native stress eligibility: ε=(2,0), ĉε=(200,0) → 부적격; 작은 변위 → 적격; c_D 큰 값 → 부적격",
      not native_stress_eligible(512, 512, (2.0, 0.0), (200.0, 0.0), (0.0, 0.0)) and native_stress_eligible(512, 512, (2.0, 0.0), (-1.5, 0.3), (1.3, -0.8)) and not native_stress_eligible(512, 512, (0.0, 0.0), (0.0, 0.0), (120.0, 0.0)))
rows = [dict(ey=0.0, ex=0.0, scene=0, c_dy=0.1, c_dx=0.0, eligible=True, raw_native_hqnr=0.9, aligned_fixed_hqnr=0.95), dict(ey=0.0, ex=0.0, scene=1, c_dy=0.1, c_dx=0.0, eligible=False, raw_native_hqnr=0.8, aligned_fixed_hqnr=0.85)]
ag = aggregate_native(rows, ((0.0, 0.0),), 2.0)["(+0.0,+0.0)"]
check("review-3 aggregate: 2장 중 1장 부적격 → 값 None·eligible_all False·n 유지 (부분 평균은 subset 키로만)", ag["raw_native_hqnr"] is None and ag["eligible_all"] is False and ag["n"] == 2 and abs(ag["raw_native_hqnr_eligible_subset"] - 0.9) < 1e-12)
# (1) 정책 블록: A-FR/A-FT 에서 fixed_reference 옵션이 없어도 aligner 가 None 이 되지 않는다 (소스 구조 검사: elif 가 정책 if 에 붙어 있는지)
src = open(os.path.join(ROOT, "train_kdv.py")).read(); blk = src[src.index("# --- aligner 정책 (§4.2)"):src.index("self.aligner_view_margin = margin")]
check("review-1 aligner policy if/elif/else 가 fixed_reference 분기 앞에서 닫힌다", blk.index('elif pol == "A-SC"') < blk.index("fixed_reference_from_donor") and blk.count("aligner = None; margin = 0") == 1)

print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
