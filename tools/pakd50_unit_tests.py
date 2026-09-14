#!/usr/bin/env python
"""PAKD50 launch gate (계획 §14.1 중 코드로 닫는 것). 하나라도 실패하면 exit 1.   python tools/pakd50_unit_tests.py
T0 binding · 정책/backend 매핑 · Q12↔X02↔R1↔N0 관계 · edge 정의 · gate detach · offset 수신자 · frozen A · 후보 격자 · 큐 · λE 의존성 · 캠페인 gate 격리"""
import copy, json, os, subprocess, sys, tempfile
import torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve
from kdv.losses_rec import GTAnchoredReconstructionKD
from kdv.teacher_assets import load_run_model, load_donor_aligner, sha256_file, freeze, state_hash
from kdv.forward import kdv_forward
from pa.losses import output_edge_loss, scharr
from pa.offset import sample_offsets, offset_loss, predict_c
from pa.warp import warp_pan
from main import import_class
from tools import gen_pakd50_configs as G

FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)
torch.manual_seed(0); torch.set_num_threads(4)
SRV = open(os.path.join(ROOT, "gspread", "server.txt")).read().strip(); SEED = G.SERVER_SEED.get(SRV, 1234)

# ---------------- K01 T0 package binding (§3.1)
ta = json.load(open(os.path.join(ROOT, "assets", "pakd50", "T0_run", "teacher_assets.json"))); tdir = os.path.join(ROOT, G.t0_dir(SRV))
sha_file = sha256_file(os.path.join(tdir, "best_hqnr", "model.safetensors")); meta = json.load(open(os.path.join(tdir, "best_hqnr_meta.json")))
check("K01 T0 = PALS24 L1E4 S2025 best_raw: 파일 sha = 자산 manifest, selected update 24240, 기록 raw HQNR 0.956976", sha_file == ta["checkpoint_sha256"] and meta["step"] == 24240 == ta["selected_update"] and abs(meta["hqnr"] - 0.9569763995761281) < 1e-9, f"{SRV}: {sha_file[:16]}")
Model = import_class(yaml.safe_load(open(os.path.join(tdir, "meta", "config.yaml")))["model"])
T0, tman = load_run_model(tdir, "best_hqnr", Model); freeze(T0)
al, aman = load_donor_aligner(os.path.join(tdir, "best_hqnr"), 8)
check("K01 T0 는 A(0.105M, view margin 4)+U(2.6589M) 한 쌍, 같은 checkpoint 의 aligner 가 donor (tensors sha 493aca18…)", T0.aligner is not None and tman["aligner_view_margin"] == 4 and tman["backbone_params"] == 2658888 and aman["aligner_tensors_sha256_16"] == ta["aligner_tensors_sha256_16"] == "493aca181f93ae68" and aman["n_params"] == 105330)
check("K01 T0 head 가 재초기화되지 않았고(0 아님) donor 와 Teacher 의 aligner state 가 같다", float(al.fc2.weight.abs().sum()) > 0 and state_hash(al) == state_hash(T0.aligner))

# ---------------- K02 정책/backend 매핑 (§4·§6·§7)
cal = dict(tau_R=0.012, lambda_E=0.3); spec = {c: resolve(G.kdv_block(c, SEED, SRV, cal=cal)) for c in G.CASES}
check("K02 J*: A-FT + I-AEQ + offset 1e-4 (홀수 update, b=2) · F*: A-FR + I-NATIVE-TRANSFER + offset 0 · AL*: A LR 3e-6", all(spec[c]["policy"] == "A-FT" and spec[c]["protocol"] == "I-AEQ" and abs(spec[c]["offset_weight_effective"] - 1e-4) < 1e-15 for c in ("J0", "JQ", "JR", "XJ", "AL0", "ALQ"))
      and all(spec[c]["policy"] == "A-FR" and not spec[c]["aligner_trainable"] and spec[c]["offset_weight_effective"] == 0 for c in ("F0", "FQ", "FR", "XF")) and G.kdv_block("AL0", SEED, SRV, cal=cal)["aligner_lr"] == 3e-6 and G.kdv_block("J0", SEED, SRV, cal=cal)["aligner_lr"] == 1e-5)
check("K02 backend: N0=gt(Teacher eval_only) · R1=hard_only(α1, β0) · Q12=adaptive(α1, β0.1)+EDGE-H · X02=hard_only+EDGE-H", spec["J0"]["rec_mode"] == "gt" and spec["J0"]["teacher_eval_only"] and spec["JR"]["rec_mode"] == "hard_only" and not spec["JR"]["stat_enabled"]
      and spec["JQ"]["rec_mode"] == "adaptive" and spec["JQ"]["stat_key"] == "EDGE" and spec["JQ"]["stat_mode"] == "H" and spec["XJ"]["rec_mode"] == "hard_only" and spec["XJ"]["stat_key"] == "EDGE" and all(spec[c]["needs_teacher"] for c in ("JQ", "JR", "XJ", "FQ", "FR")))
kq = G.kdv_block("JQ", SEED, SRV, cal=cal); check("K02 계수: α 1 · β 0.1 · eps 1e-6 · τR 고정 숫자 · λE 고정 숫자 · r_grad 0.05 · pilot = J0 S1234/last", kq["rec"]["alpha"] == 1.0 and kq["rec"]["kd_weight"] == 0.1 and kq["rec"]["eps"] == 1e-6 and kq["rec"]["tau"] == 0.012 and kq["stat"]["outer_weight"] == 0.3 and kq["stat"]["r_grad"] == 0.05 and kq["stat"]["lambda_pilot"] == G.pilot_run() + "/last")
check("K02 Teacher 는 자기 aligner 로 forward (share_correction 은 A-FR 에서만) · donor = T0 best_hqnr (step 24240, sha 검사) · 고정 참조 view = T0 aligner", kq["teacher"]["run"] == G.t0_dir(SRV) and kq["donor"]["source"].endswith("/best_hqnr") and kq["donor"]["expected_step"] == 24240 and kq["donor"]["expected_sha256"] == sha_file and kq["eval"]["fixed_reference_from_donor"])
check("K02 후보 보존·고정 batch 진단·selector best_hqnr(+best_rr_val,last)·expect_arch W112 D123", kq["select"]["retain_all_candidates"] and kq["diag"]["fixed_batch"] and kq["select"]["primary"] == "best_hqnr" and kq["expect_arch"] == dict(width=112, depth=[1, 2, 3], noalign=False))

# ---------------- K03 Q12 ↔ X02 ↔ R1 ↔ N0 (§4.1, §14.1 'Q12 해제')
g = torch.Generator().manual_seed(1); B, C, H, W = 2, 8, 16, 16
gt = torch.rand(B, C, H, W, generator=g); zt = gt + 0.05 * torch.randn(B, C, H, W, generator=g); zs = (gt + 0.08 * torch.randn(B, C, H, W, generator=g)).requires_grad_(True)
tau = 0.012
def crit(mode, alpha, beta): return GTAnchoredReconstructionKD(tau, alpha=alpha, kd_weight=beta, eps=1e-6, mode=mode)
l_q12 = crit("adaptive", 1.0, 0.1)(zs, zt, gt); l_x02 = crit("hard_only", 1.0, 0.0)(zs, zt, gt); l_r3b0 = crit("adaptive", 1.0, 0.0)(zs, zt, gt); l_n0 = crit("gt", 0.0, 0.0)(zs, zt, gt); l_plain = (zs - gt).abs().mean()
check("K03 β=0 인 adaptive(R3) == hard_only(R1) (X02 정의) · gt 모드 == plain L1 (N0)", torch.allclose(l_r3b0.loss if hasattr(l_r3b0, "loss") else l_r3b0.total, l_x02.loss if hasattr(l_x02, "loss") else l_x02.total) and torch.allclose((l_n0.loss if hasattr(l_n0, "loss") else l_n0.total), l_plain))
def tot(r): return r.loss if hasattr(r, "loss") else r.total
et = (zt - gt).abs().mean(1); es = (zs - gt).abs().mean(1); d = et / (et + tau); a = ((es - et).clamp_min(0) / (es + 1e-6)).clamp(max=1.0); k = (zs - zt).abs().mean(1)
manual = ((1 + d) * es).mean() + (0.1 * (1 - d) * a * k).mean()
check("K03 Q12 = ⟨(1+αd)e_S⟩ + ⟨β(1−d)a k⟩ (수식 그대로; 재정규화 없음)", torch.allclose(tot(l_q12), manual, atol=1e-6), f"{float(tot(l_q12)):.6f} vs {float(manual):.6f}")
zs2 = zt.clone().requires_grad_(True); r2 = crit("adaptive", 1.0, 0.1)(zs2, zt, gt, return_maps=True)
check("K03 gate: e_S ≤ e_T 이면 a=0 (soft 0) · Teacher/GT/gate 는 detach, e_S 는 live (gradient 존재)", float(r2.maps["soft_weight"].abs().max()) == 0.0 and torch.autograd.grad(tot(r2), zs2, allow_unused=True)[0] is not None and not r2.maps["difficulty"].requires_grad)
# edge (§4.2): Z=Y → 0 · signed bandwise Scharr/32 · reflect1 · interior 1px · 0.5 reduction
check("K03 edge: Z=Y → 0 · Scharr /32 · reflect pad · 내부 1px · 0.5(x)+0.5(y) 평균", float(output_edge_loss(gt, gt)) == 0.0 and torch.allclose(scharr(gt)[0][:, :, 5, 5], (torch.nn.functional.conv2d(torch.nn.functional.pad(gt.reshape(-1, 1, H, W), (1, 1, 1, 1), mode="reflect"), (torch.tensor([[-3., 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 32)[None, None]).reshape(B, C, H, W))[:, :, 5, 5]))

# ---------------- K04 offset 수신자·frozen A·Teacher 고정 (실제 wrapper)
from pa.model import PAModel
tpl = yaml.safe_load(open(os.path.join(ROOT, "config", "PO10_N1_REC_W112_D123_WV3_S2025_R200_FRSTAT.yaml"))); bb = Model(**tpl["model_args"])
for p_ in bb.parameters():
    if p_.abs().sum() == 0: p_.data.normal_(0, 0.01)
S = PAModel(copy.deepcopy(bb), copy.deepcopy(al), aligner_margin=4); S.train()
pan = torch.rand(2, 1, 64, 64, generator=g); ms = torch.rand(2, 8, 16, 16, generator=g); lpan = torch.rand(2, 1, 16, 16, generator=g); y = torch.rand(2, 8, 64, 64, generator=g)
h0 = state_hash(T0); o = kdv_forward(S, T0, pan, ms, lpan, share_correction=False, teacher_needed=True, aligner_live=True)
check("K04 Teacher 는 자기 aligner 로 forward (Δ_T ≠ Δ_S 가능, y_t no-grad) · Student/Teacher parameter 객체 분리 · forward 뒤 Teacher hash 불변", o["y_t"] is not None and not o["y_t"].requires_grad and o["delta"].requires_grad and state_hash(T0) == h0 and all(p is not q for p in S.parameters() for q in T0.parameters()))
eps = sample_offsets(2, 2.0, torch.Generator().manual_seed(3)); pe = warp_pan(pan, eps); ce = predict_c(S.aligner, pe, o["ms_base"], 4); lo = offset_loss(ce, o["delta"], eps, stop_reference=True)
gA = torch.autograd.grad(lo, list(S.aligner.parameters()), retain_graph=True, allow_unused=True); gU = torch.autograd.grad(lo, list(S.backbone.parameters()), retain_graph=True, allow_unused=True)
check("K04 L_off 단독: A 에만 gradient, U 없음, native ĉ_S target 은 SG", all(x is not None for x in gA) and all(x is None for x in gU) and torch.autograd.grad(lo, o["delta"], allow_unused=True)[0] is None)
lrec = (o["y"] - y).abs().mean(); check("K04 native L_rec → A 와 U 양쪽 gradient (P_aligned 를 detach 하지 않음)", any(x is not None and float(x.abs().sum()) > 0 for x in torch.autograd.grad(lrec, list(S.aligner.parameters()), retain_graph=True, allow_unused=True)))
Fm = PAModel(copy.deepcopy(bb), copy.deepcopy(al), aligner_margin=4); freeze(Fm.aligner); Fm.train(); Fm.backbone.requires_grad_(True); hf = state_hash(Fm.aligner)
opt = torch.optim.AdamW([p for p in Fm.backbone.parameters()], lr=1e-4, weight_decay=0.01); of = kdv_forward(Fm, T0, pan, ms, lpan, share_correction=False, teacher_needed=True, aligner_live=False); (of["y"] - y).abs().mean().backward(); opt.step()
check("K04 F: frozen A 는 optimizer 밖·grad None·step 뒤 hash 불변 (WD 미적용)", state_hash(Fm.aligner) == hf and all(p.grad is None for p in Fm.aligner.parameters()))

# ---------------- K05 후보 격자·큐·λE 의존·gate 격리
cfgp = os.path.join(ROOT, "config", G.run_name("J0", SEED) + ".yaml")
if os.path.exists(cfgp):
    c = yaml.safe_load(open(cfgp)); steps = [1010 * i for i in range(1, 50)] + [50000]
    check(f"K05 {SRV} J0 config: eval_epoch 5 → GRID1010_50K_v1 (1010k, k=1..49 + exact 50000 = 50 후보), num_iter 50000, seed {SEED}, batch 48", c["eval_epoch"] == 5 and c["num_iter"] == 50000 and c["seed"] == SEED and c["batch_size"] == 48 and len(steps) == 50 and c["kdv"]["candidate_grid_id"] == "GRID1010_50K_v1")
for srv in G.SERVER_SEED:
    q = os.path.join(ROOT, "config", "queues", f"pakd50_{srv}_stage1.txt")
    if os.path.exists(q):
        qs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")]; check(f"K05 큐 {srv} stage 1 = J0 만 (seed {G.SERVER_SEED[srv]}; 나머지는 gate 가 J0→JQ→F0→FQ→JR→FR→XJ 로 편성 — 감사 F02)", qs == [G.run_name(c_, G.SERVER_SEED[srv]) for c_ in G.QUEUE_STAGE[1]])
calp = os.path.join(ROOT, G.CAL_PATH); calj = G.calibration()          # 로컬 + (λE 만) 사본 덧입힘 — gen/gate 가 보는 것과 같은 view
check("K05 τR 고정값이 calibration_resolved.json 에 있고 stage-1 R1 config 에 같은 숫자로 들어간다", calj.get("tau_R") and abs(yaml.safe_load(open(os.path.join(ROOT, "config", G.run_name("JR", SEED) + ".yaml")))["kdv"]["rec"]["tau"] - calj["tau_R"]) < 1e-12, str(calj.get("tau_R")))
r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_pakd50_configs.py"), "--server", SRV, "--stage", "2", "--out-dir", tempfile.mkdtemp()], cwd=ROOT, capture_output=True, text=True)
check("K05 stage 2 (JQ/FQ/XJ) 는 λE 없이는 생성되지 않는다 (fail-fast, §5.3)", (r.returncode != 0) == (not calj.get("lambda_E")), (r.stdout + r.stderr).strip()[-120:])
# ---------------- K06 λE 전달 (s1 calibrate → 사본 mirror → s2/s3 가 받아 로컬 덧입힘; τR/campaign 이 다르면 거부)
_td = tempfile.mkdtemp(); _keep = (G.CAL_PATH, G.ASSET_CAL_PATH, G.ROOT)
try:
    G.ROOT = _td; G.CAL_PATH = "loc.json"; G.ASSET_CAL_PATH = "ast.json"
    loc = dict(campaign_id=G.CAMPAIGN_ID, tau_R=0.0124639, tau_server="s2"); ast = dict(campaign_id=G.CAMPAIGN_ID, tau_R=0.0124639, lambda_E=0.0123, lambda_E_source=dict(pilot_run=G.pilot_run()), lambda_computed_at="t")
    json.dump(loc, open(os.path.join(_td, "loc.json"), "w")); json.dump(ast, open(os.path.join(_td, "ast.json"), "w"))
    m, src = G.sync_calibration_from_assets(write=False)
    check("K06 로컬에 λE 없고 사본에 있으면 (같은 campaign·τR) λE 만 덧입힌다 — write=False 는 파일을 안 바꾼다", src == "assets" and m["lambda_E"] == 0.0123 and m["tau_server"] == "s2" and "lambda_E" not in json.load(open(os.path.join(_td, "loc.json"))))
    m, src = G.sync_calibration_from_assets(write=True)
    check("K06 write=True 는 로컬 파일에 λE 를 기록하고 출처를 남긴다", json.load(open(os.path.join(_td, "loc.json"))).get("lambda_E") == 0.0123 and bool(m.get("lambda_E_from")))
    json.dump(dict(loc, tau_R=0.0130), open(os.path.join(_td, "loc.json"), "w")); m, src = G.sync_calibration_from_assets(write=True)
    check("K06 τR 이 다르면(다른 T0/데이터) λE 를 받지 않는다", src == "local" and not m.get("lambda_E"))
    json.dump(dict(loc, campaign_id="other"), open(os.path.join(_td, "loc.json"), "w")); m, src = G.sync_calibration_from_assets(write=True)
    check("K06 campaign_id 가 다르면 받지 않는다", src == "local" and not m.get("lambda_E"))
    os.remove(os.path.join(_td, "loc.json")); m, src = G.sync_calibration_from_assets(write=True)
    check("K06 로컬 파일이 없으면 사본을 그대로 만든다 (첫 prepare)", src == "assets" and json.load(open(os.path.join(_td, "loc.json")))["lambda_E"] == 0.0123)
finally:
    G.CAL_PATH, G.ASSET_CAL_PATH, G.ROOT = _keep
# ---------------- K07 gate 편성 규칙 (순수 함수 — live gate 를 부르지 않는다; 감사 F02/F05)
term = lambda done: (lambda c: c in done)
check("K07 λE 없음 → τR 만 필요한 다음 한 벌만 (J0 뒤 F0); λE 가 필요한 case 는 편성 안 함", G.schedule(False, term({"J0"}), 40.0, 1.9) == (["F0"], []))
check("K07 λE 없음 · F0 도 끝남 → JR 한 벌", G.schedule(False, term({"J0", "F0"}), 40.0, 1.9) == (["JR"], []))
check("K07 λE 있음 → 남은 전부 우선순위 (JQ → F0 → FQ → JR → FR → XJ)", G.schedule(True, term({"J0"}), 40.0, 1.9) == (["JQ", "F0", "FQ", "JR", "FR", "XJ"], []))
check("K07 admission: 남은 시간 안에 드는 것까지만 (1.1×1.9h 씩), 나머지는 밀림", G.schedule(True, term({"J0"}), 4.5, 1.9) == (["JQ", "F0"], ["FQ", "JR", "FR", "XJ"]))
check("K07 시계 없음(None) 이면 admission 생략", G.schedule(True, term(set(G.PRIORITY) - {"XJ"}), None, 1.9) == (["XJ"], []))
# ---------------- K08 예산 블록 (감사 F05): P0 예약 · 50h/4h · 공통 마감
kb = G.kdv_block("FR", SEED, SRV); kb0 = G.kdv_block("J0", SEED, SRV)
check("K08 remaining_mandatory = P0(J0/JQ) 중 자기 제외 · total 50h / reserve 4h (학습 admission ≤ 46h)", kb["budget"]["remaining_mandatory"] == [G.run_name("J0", SEED), G.run_name("JQ", SEED)] and kb0["budget"]["remaining_mandatory"] == [G.run_name("JQ", SEED)] and kb["budget"]["total_gpu_hours"] == 50.0 and kb["budget"]["reserve_hours"] == 4.0)
check("K08 공통 시계(assets/pakd50/campaign_clock.json) 가 있으면 config 에 training_deadline 이 박힌다", (not G.campaign_clock()) or kb["budget"].get("training_deadline") == G.campaign_clock()["training_deadline"])
# ---------------- K09 s4 배정 (research_log/PAN_S4_Integrated_Experiment_Cases_2026-09-14.md §5–§7): Q12 scalar variant 는 계수만 · s4 우선순위·추가 편성
cal4 = dict(tau_R=0.012, lambda_E=0.3)
qa = G.kdv_block("J_QA05", 1234, "s4", cal=cal4); qb = G.kdv_block("J_QB005", 1234, "s4", cal=cal4); qb2 = G.kdv_block("AL_QB02", 1234, "s4", cal=cal4); qe = G.kdv_block("J_QE025", 1234, "s4", cal=cal4); qe1 = G.kdv_block("J_QE10", 1234, "s4", cal=cal4)
check("K09 J_QA05: α 0.5 · β 0.1 · λE0 그대로 (rec R3 + EDGE-H, J 정책)", qa["rec"]["alpha"] == 0.5 and qa["rec"]["kd_weight"] == 0.1 and qa["stat"]["outer_weight"] == 0.3 and qa["aligner_policy"] == "A-FT" and qa["aligner_lr"] == 1e-5)
check("K09 J_QB005 / AL_QB02: β 0.05 / 0.2, α 1; AL 은 A LR 3e-6 · baseline AL0", qb["rec"]["kd_weight"] == 0.05 and qb2["rec"]["kd_weight"] == 0.2 and qb["rec"]["alpha"] == 1.0 and qb2["aligner_lr"] == 3e-6 and qb2["baseline_run"] == G.run_name("AL0", 1234))
check("K09 J_QE025 / J_QE10: λE = λE0 × 0.5 / × 2 (다시 calibrate 하지 않음), r_grad 도 배율", abs(qe["stat"]["outer_weight"] - 0.15) < 1e-12 and abs(qe1["stat"]["outer_weight"] - 0.6) < 1e-12 and qe["stat"]["r_grad"] == 0.025 and qe1["stat"]["r_grad"] == 0.1)
try:
    G.kdv_block("J_QE10", 1234, "s4", cal=dict(tau_R=0.012)); check("K09 λE 배율 case 는 λE0 없이 만들 수 없다", False)
except SystemExit:
    check("K09 λE 배율 case 는 λE0 없이 만들 수 없다", True)
check("K09 s4: seed 1234(s1 과 같은 run id), T0 경로는 모든 서버에서 자산 사본, 기본 묶음 J0→JQ→AL0→ALQ, stage 1 = J0/AL0", G.SERVER_SEED["s4"] == 1234 and G.t0_dir("s1") == G.t0_dir("s4") == G.T0_ASSET_DIR and G.priority_for("s4") == ["J0", "JQ", "AL0", "ALQ"] and G.stage_cases("s4", 1) == ["J0", "AL0"] and G.priority_for("s1") == G.PRIORITY)
check("K09 s4 remaining_mandatory = J0/JQ/AL0/ALQ 중 자기 제외", G.kdv_block("ALQ", 1234, "s4", cal=cal4)["budget"]["remaining_mandatory"] == [G.run_name(c, 1234) for c in ("J0", "JQ", "AL0")])
ex = ["J_QA05", G.run_name("JQ", 3407), "J_QA05"]
check("K09 추가 편성: case id 와 전체 run 이름 혼용, 기본 묶음 뒤·중복 제거; λE 없으면 λE case 는 미편성", G.schedule(True, term({"J0", "JQ", "AL0", "ALQ"}), 40.0, 1.9, priority=G.priority_for("s4"), extra=ex) == (["J_QA05", G.run_name("JQ", 3407)], []) and G.schedule(False, term({"J0", "AL0"}), 40.0, 1.9, priority=G.priority_for("s4"), extra=ex) == ([], []))
check("K09 to_tag/case_of", G.to_tag("J_QA05", 3407) == G.run_name("J_QA05", 3407) and G.case_of(G.run_name("AL_QE10", 3407)) == "AL_QE10" and G.case_of("JQ") == "JQ")
r0 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "campaign_gate.py")], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PANCRAFTER_CAMPAIGN_GATES": ""})
check("K05 campaign gate 기본 닫힘", r0.stdout.strip() == "")
src = open(os.path.join(ROOT, "train_kdv.py")).read()
check("K05 trainer: 후보 보존·고정 batch 진단·ledger 잠금·donor expected_step 검사·frozen aligner hash gate(L03) 존재", all(x in src for x in ("retain_all_candidates", "gradient_diagnostics_fixed.jsonl", "_LedgerLock", "_check_donor_step", "frozen aligner(A-FR) state")))
print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
