#!/usr/bin/env python
"""PALS24 실행 전 gate (계획 §6·§8·§9.4·§10.3·§11.6 중 코드로 닫는 것). 하나라도 실패하면 exit 1.   python tools/pals24_unit_tests.py

PL01 생성기·순서·block·예산 키 / PL02 NF16 P0·P2·P3 와의 config 동치(λ 외 변경 없음) / PL03 실제 wrapper 의 gradient 경로·합산 선형성 / PL04 ε sampler 와 상수 예측기 기대값 4b/(3π)
PL05 예산 gate(margin 1.1·block 완결) / PL06 λ* 선택 규칙 / PL07 장면 집계식 / PL08 캠페인 gate 격리 / PL09 trainer 진단 필드·parity / PL10 고정 probe·반응 fit / PL11 보고 표 열 / PL12 metric gate 함수"""
import copy, io, json, math, os, subprocess, sys, tempfile
import numpy as np, torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from pa.aligner import PANGlobalAligner
from pa.model import PAModel
from pa.offset import sample_offsets, offset_loss, predict_c, lambda_off, aligner_margin
from pa.warp import warp_pan
from kdv.forward import kdv_forward
from kdv.registry import resolve
from kdv.teacher_assets import load_donor_aligner
from main import import_class
from tools import gen_pals24_configs as G
from tools.pals24_select_lambda import select_lambda, HQNR_METHOD_MARGIN
from train_kdv import KDVTrainer

FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)

torch.manual_seed(0); torch.set_num_threads(4)
PY = sys.executable

# ---------------- PL01 generator
s1 = G.stage1(); s2 = G.stage2("L1E3"); bl = G.blocks("L1E3")
check("PL01 stage 1 = seed 1234 × (L1E3, L1E4, L3E3) in A1→A2→A3 order", s1 == [("L1E3", 1234), ("L1E4", 1234), ("L3E3", 1234)])
check("PL01 stage 2 = B1 P0/7777 → B2 L000/7777 → B3 λ*/7777 → C1 L000/2025 → C2 λ*/2025 → C3 P0/2025", s2 == [("CTRLP0", 7777), ("L000", 7777), ("L1E3", 7777), ("L000", 2025), ("L1E3", 2025), ("CTRLP0", 2025)])
check("PL01 blocks: B = 3 seed-7777 runs, C = 3 seed-2025 runs (P0·L000·λ* 완결 단위)", [c[1] for c in bl["B"]] == [7777] * 3 and [c[1] for c in bl["C"]] == [2025] * 3 and {c[0] for c in bl["B"]} == {"CTRLP0", "L000", "L1E3"})
check("PL01 λ 값: L000 0 · L1E4 1e-4 · L1E3 1e-3 · L3E3 3e-3 · L1E2 1e-2 (PALS24 grid; L3E5/L3E4 는 PALSV18 추가)", all(G.LAMBDA[k] == v for k, v in {"L000": 0.0, "L1E4": 1e-4, "L1E3": 1e-3, "L3E3": 3e-3, "L1E2": 1e-2}.items()) and G.NEW_LAMBDAS == ["L1E3", "L1E4", "L3E3"])
check("PL01 run 이름 규칙 PALS24_<case>_W112_D123_WV3_S<seed>_N2LAST_R200_v1", G.run_name("L1E3", 1234) == "PALS24_L1E3_W112_D123_WV3_S1234_N2LAST_R200_v1")
try:
    bad = G.stage2("L1E2"); check("PL01 stage 2 는 신규 λ 3개 중 하나만 λ* 로 받는다", False)
except AssertionError:
    check("PL01 stage 2 는 신규 λ 3개 중 하나만 λ* 로 받는다", True)
kb = G.kdv_block("L1E3", 1234, "ab" * 32, 50000, s1)
check("PL01 budget: ledger _pals24_budget · total 24 · reserve 4.0(=3 eval + 1 buffer) · margin 1.1 · required False · block 예약 2.0/run · remaining = 같은 block 나머지",
      kb["budget"]["ledger"].endswith("_pals24_budget/ledger.json") and kb["budget"]["total_gpu_hours"] == 24.0 and kb["budget"]["reserve_hours"] == 4.0 and kb["budget"]["margin"] == 1.1 and kb["budget"]["required"] is False
      and set(kb["budget"]["projected_map"].values()) == {2.0} and len(kb["budget"]["projected_map"]) == 3 and kb["budget"]["remaining_mandatory"] == [G.run_name("L1E4", 1234), G.run_name("L3E3", 1234)])
spB = {c: resolve(G.kdv_block(c, 7777, "ab" * 32, 50000, bl["B"])) for c in ("CTRLP0", "L000", "L1E3")}
check("PL01 resolver: CTRLP0 A-ID/I-A · L000 A-FT/I-NATIVE-TRANSFER λ 0 · L1E3 A-FT/I-AEQ λ 0.001 ramp 0 geo 0, teacher 없음",
      spB["CTRLP0"]["policy"] == "A-ID" and spB["CTRLP0"]["protocol"] == "I-A" and spB["L000"]["policy"] == "A-FT" and spB["L000"]["protocol"] == "I-NATIVE-TRANSFER" and spB["L000"]["offset_weight_effective"] == 0
      and spB["L1E3"]["protocol"] == "I-AEQ" and abs(spB["L1E3"]["offset_weight_effective"] - 1e-3) < 1e-15 and spB["L1E3"]["offset_ramp_updates"] == 0 and spB["L1E3"]["geometry_weight_effective"] == 0 and all(not s["needs_teacher"] for s in spB.values()))

# ---------------- PL02 NF16 동치: λ(및 seed) 외 변경 없음 — 실제 YAML 을 만들어 NF16 P3/P2/P0 config 와 비교
_DONOR_HERE = os.path.exists(os.path.join(ROOT, "work_dir", G.DONOR_RUN, "last", "model.safetensors"))   # NF16/PALS24 계보 donor 가 없는 서버(s4 등)는 생성기가 expected_sha256/step 을 null 로 둔다 (PAKD50 s4 보고 P-2)
def stripped(text):
    d = yaml.safe_load("\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#")))
    d.pop("work_dir", None); k = d.get("kdv", {})
    for key in ("campaign_id", "plan_protocol_id", "document_revision", "case_id", "budget"):
        k.pop(key, None)
    if not _DONOR_HERE:                       # donor 유무에 따라 달라지는 identity 키는 비교에서 뺀다 — 이 검사는 'λ 외 변경 없음' 이 목적
        for key in ("expected_sha256", "expected_step", "expected_tensors_sha256_16"):
            (k.get("donor") or {}).pop(key, None)
    return d
nf = {q: os.path.join(ROOT, "config", f"NF16_P{q}_W112_D123_WV3_S1234_N2LAST_v1.yaml") for q in (0, 2, 3)}
if all(os.path.exists(v) for v in nf.values()):
    tmp = tempfile.mkdtemp(); cells = [("L1E2", 1234), ("L000", 1234), ("CTRLP0", 1234)]
    made, _, _ = G.generate(cells, {c: cells for c in cells}, tmp)
    def diffkeys(a, b, pre=""):
        out = []
        for key in sorted(set(a) | set(b)):
            va, vb = a.get(key), b.get(key)
            if isinstance(va, dict) and isinstance(vb, dict):
                out += diffkeys(va, vb, pre + key + ".")
            elif va != vb:
                out.append(pre + key)
        return out
    d3 = diffkeys(stripped(open(os.path.join(tmp, G.run_name("L1E2", 1234) + ".yaml")).read()), stripped(open(nf[3]).read()))
    d2 = diffkeys(stripped(open(os.path.join(tmp, G.run_name("L000", 1234) + ".yaml")).read()), stripped(open(nf[2]).read()))
    d0 = diffkeys(stripped(open(os.path.join(tmp, G.run_name("CTRLP0", 1234) + ".yaml")).read()), stripped(open(nf[0]).read()))
    check("PL02 λ=0.01 config ≡ NF16 P3 config (주석·work_dir·campaign/budget 키 제외 전부 동일)", d3 == [], f"diff {d3}")
    check("PL02 λ=0 config ≡ NF16 P2 config", d2 == [], f"diff {d2}")
    check("PL02 CTRLP0 config ≡ NF16 P0 config (eval_epoch 5→10 만 다르다 — 재사용 registry 가 10 격자 재선택을 기록)", d0 == ["eval_epoch"], f"diff {d0}")
    # λ 만 바뀐다: L1E3 vs L1E2 의 차이는 aux.offset_weight 하나
    made2, _, _ = G.generate([("L1E3", 1234)], {("L1E3", 1234): cells}, tmp)
    d13 = diffkeys(stripped(open(os.path.join(tmp, G.run_name("L1E3", 1234) + ".yaml")).read()), stripped(open(os.path.join(tmp, G.run_name("L1E2", 1234) + ".yaml")).read()))
    check("PL02 L1E3 vs L1E2: 차이는 kdv.aux.offset_weight 하나 (λ 외 변경 없음, §2.2)", d13 == ["kdv.aux.offset_weight"], f"diff {d13}")
else:
    print("  (NF16 config 없음 — PL02 생략)")

# ---------------- PL03 실제 wrapper (W112·D123 U-Net + donor aligner) 의 gradient 경로
DONOR = os.path.join(ROOT, "work_dir", G.DONOR_RUN, "last")
if os.path.exists(os.path.join(DONOR, "model.safetensors")):
    al, man = load_donor_aligner(DONOR, 8)
else:
    al = PANGlobalAligner(8)
    with torch.no_grad(): al.fc2.weight.normal_(0, 0.05); al.fc2.bias.add_(torch.tensor([0.2, -0.1]))
    print("  (donor 없음 — 합성 aligner)")
Model = import_class("model.pancrafter_paper.PANCrafterPaper")
MA = dict(in_channels=1, out_channels=8, hidden_size=112, depth=[1, 2, 3], dropout=0.0, num_heads=8, mlp_ratio=4.0, ks=3, ka=3, norm="ln", in_mode="paper", attn_locations=[], mode_modulation=False, n_attn=3)
bb = Model(**MA)
for p_ in bb.parameters():
    if p_.abs().sum() == 0: p_.data.normal_(0, 0.01)                      # zero_module 트랩
mg = 4; g = torch.Generator().manual_seed(3); pan = torch.rand(4, 1, 64, 64, generator=g); ms = torch.rand(4, 8, 16, 16, generator=g); lpan = torch.rand(4, 1, 16, 16, generator=g)
gt = torch.rand(4, 8, 64, 64, generator=g) + torch.linspace(0, 1, 64)[None, None, None, :]
m = PAModel(copy.deepcopy(bb), copy.deepcopy(al), aligner_margin=mg); m.train()
calls = {"n": 0}; orig = m.backbone.forward
def counting(*a, **kw): calls["n"] += 1; return orig(*a, **kw)
m.backbone.forward = counting
o = kdv_forward(m, None, pan, ms, lpan, share_correction=False, teacher_needed=False, aligner_live=True); lrec = (o["y"] - gt).abs().mean()
gen = torch.Generator().manual_seed(1234 + 2000); eps = sample_offsets(4, 2.0, gen)
with torch.no_grad(): p_eps = warp_pan(pan, eps)
c_eps = predict_c(m.aligner, p_eps, o["ms_base"], mg); loff = offset_loss(c_eps, o["delta"], eps, stop_reference=True)
ap = list(m.aligner.parameters()); bp = list(m.backbone.parameters())
check("PL03 update 당 U-Net forward 1회 (P_ε 는 aligner 에만)", calls["n"] == 1)
check("PL03 L_off = component-mean L1 |ĉε + ε − sg(ĉ0)| (B×2 평균, HR px)", torch.allclose(loff, (c_eps + eps - o["delta"].detach()).abs().mean()))
g_rec = torch.autograd.grad(lrec, ap, retain_graph=True, allow_unused=True); g_off = torch.autograd.grad(loff, ap, retain_graph=True, allow_unused=True)
check("PL03 L_rec → aligner·U-Net 양쪽 gradient (ĉ0 미detach)", any(x is not None and float(x.abs().sum()) > 0 for x in g_rec) and o["delta"].requires_grad and all(x is not None for x in torch.autograd.grad(lrec, bp, retain_graph=True, allow_unused=True)))
gb = torch.autograd.grad(loff, bp, retain_graph=True, allow_unused=True)
check("PL03 L_off → aligner gradient 존재, U-Net θ 직접 gradient 없음", all(x is not None and float(x.abs().sum()) > 0 for x in g_off) and all(x is None for x in gb))
check("PL03 L_off 의 ĉ0 target 경로 gradient 없음 (sg)", torch.autograd.grad(loff, o["delta"], retain_graph=True, allow_unused=True)[0] is None)
errs = {}
for lam in (1e-4, 1e-3, 3e-3, 1e-2):
    gt_ = torch.autograd.grad(lrec + lam * loff, ap, retain_graph=True, allow_unused=True)
    num = max(float((a - (b + lam * c)).abs().max()) for a, b, c in zip(gt_, g_rec, g_off)); den = max(float(a.abs().max()) for a in gt_)
    errs[lam] = num / max(den, 1e-30)
check("PL03 합산 gradient = g_rec + λ g_off (λ 1e-4…1e-2, 상대오차 ≤ 1e-5)", all(v <= 1e-5 for v in errs.values()), f"{ {k: f'{v:.1e}' for k, v in errs.items()} }")
g0 = torch.autograd.grad(lrec + 0.0 * loff, ap, retain_graph=True, allow_unused=True)
check("PL03 λ=0 → 합산 loss·aligner gradient 가 L_rec(P2 경로) 와 일치", float(lrec + 0.0 * loff) == float(lrec) and all(torch.equal(a, b) for a, b in zip(g0, g_rec)))
check("PL03 λ_off 즉시 상수 (ramp 0): lambda_off(t, λ, 0) = λ for t ∈ {0, 1, 49999}", all(lambda_off(t, 1e-3, 0) == 1e-3 for t in (0, 1, 49999)))
check("PL03 aligner view margin 4 (64²→56², 512²→504²) = aligner_margin(b=2)", aligner_margin(2.0) == 4)

# ---------------- PL04 ε sampler: 원판 면적 uniform b=2, 상수 예측기 기대값 4b/(3π)
e = sample_offsets(200000, 2.0, torch.Generator().manual_seed(1)); r2 = e.pow(2).sum(1)
check("PL04 ε ~ disk(2): |ε| ≤ 2, E[r²] ≈ b²/2 = 2 (면적 uniform), mean 0, 축 std ≈ 1", float(e.norm(dim=1).max()) <= 2 + 1e-6 and abs(float(r2.mean()) - 2.0) < 0.02 and float(e.mean(0).abs().max()) < 0.01 and abs(float(e.std(0).mean()) - 1.0) < 0.01, f"E[r²] {float(r2.mean()):.4f}")
const_l1 = float(e.abs().mean())                                          # 상수 예측기: r_ε = ε → component-mean L1 = E|ε_k| = 4b/(3π)
check("PL04 무반응 상수 예측기의 component-mean L1 기대값 = 4b/(3π) = 0.8488 (b=2) — sampler/control 점검값, 정합 오차 하한 아님", abs(const_l1 - 4 * 2.0 / (3 * math.pi)) < 0.005, f"MC {const_l1:.4f} vs {4 * 2.0 / (3 * math.pi):.4f}")
gA = torch.Generator().manual_seed(1234 + 2000); gB = torch.Generator().manual_seed(1234 + 2000)
check("PL04 같은 seed·update → 같은 ε (λ 가 달라도 동일 ε 열; data RNG 와 분리된 generator)", torch.equal(sample_offsets(48, 2.0, gA), sample_offsets(48, 2.0, gB)))

# ---------------- PL05 예산 gate: margin 1.1 · block 완결 · 24h 상한
d = KDVTrainer.budget_decision(0.5, 2.0, [2.0, 2.0], 4.0, 24.0, required=False, margin=1.1)
check("PL05 A block 시작: used 0.5 + 1.1×6 + reserve 4 = 11.1 ≤ 24 → RUN", d["decision"] == "RUN" and abs(d["projected_total_hours"] - 11.1) < 1e-9)
d = KDVTrainer.budget_decision(13.5, 2.0, [2.0, 2.0], 4.0, 24.0, required=False, margin=1.1)
check("PL05 C block: used 13.5 + 6.6 + 4 = 24.1 > 24 → DEFERRED_BUDGET (block 전체를 보류, 일부만 돌리지 않음)", d["decision"] == "DEFERRED_BUDGET" and abs(d["projected_total_hours"] - 24.1) < 1e-9)
d12 = KDVTrainer.budget_decision(13.5, 2.0, [2.0, 2.0], 4.0, 24.0, required=False); check("PL05 margin 기본값 1.2 유지 (NF16 동작 불변: 24.7h)", abs(d12["projected_total_hours"] - 24.7) < 1e-9)
check("PL05 required False 라 초과는 경고가 아니라 DEFERRED (사용자 승인 없이 24h 를 넘기지 않는다)", KDVTrainer.budget_decision(20.0, 2.0, [], 4.0, 24.0, required=False, margin=1.1)["decision"] == "DEFERRED_BUDGET")
class _B:
    run_id = G.run_name("L000", 7777); budget = G.kdv_block("L000", 7777, None, 50000, bl["B"])["budget"]; _projection = KDVTrainer._projection
_led = dict(total_gpu_hours=24.0, entries={G.run_name("CTRLP0", 7777): dict(kind="run", status="FINISHED_EXPORT", hours=1.5)})
check("PL05 block 구성원 예상: 끝난 P0 → 0 (used 에 실비), 남은 λ* → 예약 2.0", _B._projection(_B, _led, G.run_name("CTRLP0", 7777)) == 0.0 and _B._projection(_B, _led, G.run_name("L1E3", 7777)) == 2.0)
src = open(os.path.join(ROOT, "train_kdv.py")).read()
check("PL05 trainer 가 kdv.budget.margin 을 읽어 gate 에 넘긴다", 'self.budget.get("margin", 1.2)' in src and "margin=1.2" in src)

# ---------------- PL06 λ* 선택 규칙 (§10.3)
rows = [dict(case="L1E4", lam=1e-4, hqnr=0.9530, fscc=0.890), dict(case="L1E3", lam=1e-3, hqnr=0.9541, fscc=0.885), dict(case="L3E3", lam=3e-3, hqnr=0.9500, fscc=0.900)]
b, w = select_lambda(rows); check("PL06 raw best HQNR 최대가 λ* (fSCC·closure 가 좋아 보여도 바꾸지 않음)", b["case"] == "L1E3" and w["rule"].startswith("raw best"))
rows = [dict(case="L1E4", lam=1e-4, hqnr=0.95410, fscc=0.890), dict(case="L1E3", lam=1e-3, hqnr=0.95405, fscc=0.895), dict(case="L3E3", lam=3e-3, hqnr=0.9500, fscc=0.900)]
b, w = select_lambda(rows); check("PL06 HQNR 동률(1e-4 band) → fSCC 로 가른다", b["case"] == "L1E3" and "fSCC" in w["rule"])
rows = [dict(case="L1E4", lam=1e-4, hqnr=0.95410, fscc=0.89005), dict(case="L1E3", lam=1e-3, hqnr=0.95405, fscc=0.89000), dict(case="L3E3", lam=3e-3, hqnr=0.95402, fscc=0.89009)]
b, w = select_lambda(rows); check("PL06 HQNR·fSCC 모두 동률 → 더 작은 양의 λ", b["case"] == "L1E4" and "smaller" in w["rule"])
b, w = select_lambda([dict(case="L1E4", lam=1e-4, hqnr=None, fscc=None), dict(case="L1E3", lam=1e-3, hqnr=0.95, fscc=0.88)]); check("PL06 미완 후보는 제외", b["case"] == "L1E3")
check("PL06 판정선 0.0031 (S1 §6) — raw HQNR 에만", HQNR_METHOD_MARGIN == 0.0031)

# ---------------- PL07 장면 집계: 곱의 평균 ≠ 평균의 곱 (covariance) — 기록과 비교는 곱의 평균으로
dl = np.array([0.02, 0.03, 0.01, 0.04]); ds_ = np.array([0.03, 0.01, 0.04, 0.02]); mop = float(np.mean((1 - dl) * (1 - ds_))); pom = float((1 - dl.mean()) * (1 - ds_.mean()))
check("PL07 mean-of-products − product-of-means = cov(Dλ, Ds) (모집단형)", abs((mop - pom) - float(np.mean((dl - dl.mean()) * (ds_ - ds_.mean())))) < 1e-12)
sm = os.path.join(ROOT, "work_dir", G.NF16[2], "scene_metrics.csv"); bm = os.path.join(ROOT, "work_dir", G.NF16[2], "best_hqnr_meta.json")
if os.path.exists(sm) and os.path.exists(bm):
    import csv
    b_ = json.load(open(bm)); sc = [r for r in csv.DictReader(open(sm)) if int(r["step"]) == int(b_["step"]) and r["view"] == "raw_original"]
    mop2 = float(np.mean([(1 - float(r["d_lambda"])) * (1 - float(r["d_s"])) for r in sc]))
    check("PL07 NF16 P2 best: 20 scene 곱의 평균 == best_hqnr_meta (1e-9)", len(sc) == 20 and abs(mop2 - float(b_["hqnr"])) < 1e-9, f"{mop2:.9f} vs {float(b_['hqnr']):.9f}")

# ---------------- PL08 캠페인 gate 격리: 기본 닫힘 · pals24 는 탐색 미완이면 아무것도 열지 않음
r0 = subprocess.run([PY, os.path.join(ROOT, "tools", "campaign_gate.py")], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PANCRAFTER_CAMPAIGN_GATES": ""})
r1 = subprocess.run([PY, os.path.join(ROOT, "tools", "campaign_gate.py")], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PANCRAFTER_CAMPAIGN_GATES": "pals24"})
check("PL08 gate 기본 닫힘 (stdout 비어 있음) · pals24 gate 는 A1–A3 미완이면 열지 않음", r0.stdout.strip() == "" and r1.returncode == 0 and r1.stdout.strip() == "" and "PALS24" in r1.stderr, r1.stderr.strip().splitlines()[-1][:80] if r1.stderr else "")

# ---------------- PL09 trainer 진단 필드·parity
check("PL09 _diagnose 가 ρ_g·cos ψ·raw/weighted L_off·off_unet_grad_absent 를 기록한다", all(k in src for k in ('out["rho_g"]', 'out["cos_psi"]', 'out["loss_off_raw"]', 'out["loss_off_weighted"]', 'out["off_unet_grad_absent"]', 'out["grad_off_A_weighted"]')))
class _P: protocol = "I-AEQ"; diag_every = 1000
check("PL09 진단 step 이 홀수(offset 연습) update 를 포함 — 10001/25001/49001", all(KDVTrainer.is_diag_step(_P, s) for s in (10001, 25001, 49001)) and not KDVTrainer.is_diag_step(_P, 25002))
check("PL09 I-AEQ 의 offset 연습은 step % 2 == 1 에서만, 합산 전 λ 배 (forward 중간 optimizer step 없음)", 'step % 2 == 1' in src and src.count(".backward(total)") == 1 and src.count("self.optimizer.step()") == 1)

# ---------------- PL10 고정 probe·반응 fit
from tools.po10_diag import fixed_probes, fit
pr, st = fixed_probes(2.0, "pals24"); rad = sorted({round(math.hypot(a, b), 6) for a, b in pr if (a, b) != (0.0, 0.0)})
check("PL10 pals24 probe: 0 + {0.5, 1, 2} × 8 방향 = 25, stress 없음", len(pr) == 25 and rad == [0.5, 1.0, 2.0] and st == [] and len({round(math.atan2(a, b), 6) for a, b in pr if (a, b) != (0.0, 0.0)}) == 8)
rows_ideal = [dict(sample=0, kind="probe", ey=a, ex=b, c0_dy=0.3, c0_dx=-0.2, ce_dy=0.3 - a, ce_dx=-0.2 - b, q_dy=-a, q_dx=-b, closure=0.0) for a, b in pr]
ft = fit(rows_ideal, 2.0)
check("PL10 이상 반응(ĉε = ĉ0 − ε): B = −I, intercept 0, offset_mae_component 0, EPE 0", abs(ft["B_diag"][0] + 1) < 1e-9 and abs(ft["B_diag"][1] + 1) < 1e-9 and max(abs(x) for x in ft["b"]) < 1e-9 and ft["offset_mae_component"] < 1e-9 and ft["offset_epe"] < 1e-9)
rows_const = [dict(sample=0, kind="probe", ey=a, ex=b, c0_dy=0.3, c0_dx=-0.2, ce_dy=0.3, ce_dx=-0.2, q_dy=0.0, q_dx=0.0, closure=math.hypot(a, b)) for a, b in pr]
ft = fit(rows_const, 2.0); exp_mae = float(np.mean([(abs(a) + abs(b)) / 2 for a, b in pr if (a, b) != (0.0, 0.0)]))
check("PL10 상수 예측기: B = 0, offset_mae_component = 고정 probe 의 평균 |ε| 성분, legacy closure 는 별도 이름", abs(ft["B_diag"][0]) < 1e-9 and abs(ft["offset_mae_component"] - exp_mae) < 1e-9 and "closure_mean" in ft and ft["n_fixed_probe"] == 24)

# ---------------- PL11 보고 표 열 (§13.2)
from tools import pals24_report as R
check("PL11 표 A 열: case·seed·lambda·selected_update·checkpoint·HQNR_raw_original·fSCC·Dλ·Ds·RR(py)·ΔH vs P2/P0", all(c in R.TABLE_A for c in ("selected_update", "checkpoint_sha16", "HQNR_raw_original", "fSCC_raw_original", "D_lambda", "D_s", "RR_ERGAS_py", "delta_H_vs_P2", "delta_H_vs_P0")))
check("PL11 표 B 열: exact_update·4 view·native Δ·B_resp 4항·component MAE·EPE·legacy closure", all(c in R.TABLE_B for c in ("exact_update", "raw_original", "raw_v64", "aligned_self_v64", "aligned_fixedN2_v64", "native_delta_norm_median", "B_resp_yy", "B_resp_xx", "offset_mae_component", "offset_epe", "legacy_closure_mean")))
check("PL11 표 C 열: grad norm·weighted·cos·raw/weighted Loff·support·hash·metric 재현 오차·보간/MS 대조", all(c in R.TABLE_C for c in ("grad_rec_A_norm", "grad_off_A_norm", "weighted_grad_off_A_norm", "cos_psi", "raw_Loff", "weighted_Loff", "support_n_eligible_at_best", "metric_reproduction_abs_err", "ms_swap_B_diag")))

# ---------------- PL12 metric gate 함수 (CSV 기반; G-M1 재평가는 prepare 가 GPU 로)
from tools import pals24_metric_gate as MG
con = MG.contract()
check("PL12 G-M0 contract: secondary key 는 fscc(원 PAN 참조, full-frame Sobel SCC), 선택에 쓰지 않는 것 명시, 판정선 0.0031 분리", con["selector"]["secondary_metric_key"] == "fscc" and "ORIGINAL PAN" in con["selector"]["secondary_definition"] and con["tolerances"]["method_margin_raw_hqnr"] == 0.0031 and con["tolerances"]["checkpoint_tie_hqnr"] == 1e-4)
if os.path.exists(os.path.join(ROOT, "work_dir", G.NF16[2], "best_hqnr_meta.json")):
    e2 = MG.check_run(G.NF16[2], "reuse", "L000", 1234); e0 = MG.check_run(G.NF16[0], "reuse", "CTRLP0", 1234)
    check("PL12 NF16 P2 재사용 검사: G-M2/G-M5/G-M6/G-M7/자산 전부 통과 → APPROVED", e2["approval"] == "APPROVED" and e2["checks"]["G-M6_checkpoint_correspondence"]["last_is_exact_50k"])
    check("PL12 NF16 P0: G-M3 (aligned_self == raw_v64) 통과, eval_epoch 5 → APPROVED_WITH_NOTE + 10 격자 재선택 기록", e0["checks"].get("G-M3_p0_aligned_self_eq_raw_v64", {}).get("pass_") and e0["approval"] == "APPROVED_WITH_NOTE" and (e0.get("best_on_grid10") or {}).get("hqnr"))

print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
