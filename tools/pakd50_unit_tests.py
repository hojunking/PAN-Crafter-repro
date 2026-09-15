#!/usr/bin/env python
"""PAKD50 launch gate (계획 §14.1 중 코드로 닫는 것). 하나라도 실패하면 exit 1.   python tools/pakd50_unit_tests.py
T0 binding · 정책/backend 매핑 · Q12↔X02↔R1↔N0 관계 · edge 정의 · gate detach · offset 수신자 · frozen A · 후보 격자 · 큐 · λE 의존성 · 캠페인 gate 격리
K17–K20 (2026-09-15 재배정): RC 정책·J_R3_NOEDGE/J_N0_EDGE 정의 · RC trainer step(offset 연습 없음, L_rec → A) · 서버별 명시 순서·예약식·admission · trainer projection_file
K23–K28 (2026-09-15 QEDGE9): case/branch/시간 정책 · registry 거부 · per-sample edge·trainer 경로(g=1/0/혼합/const) · q 수식·θq·셔플·cue 자산 · s5/s4 편성·schedule(blocked/exempt) · feeder return_meta · 생성 config
K29–K33 (QEDGE9 감사 대응): exact resume(F04) · cue asset_id·내부 일관성·재개 대조(F03) · QEC pilot identity(F02) · 실측 통합(F07) · 완료 검증(F06)
K34–K38 (2026-09-15 QEGX s3/s4): case/branch/시간 정책/편성·예약(§5·§6·§9) · registry edge_route 거부·통과 · trainer edge_route(g=1≡JQ · g=0≡JE0 · 혼합 autograd · QERS RNG) · LFQE50/β=0/g=0 동치 · c_E3 pilot 분리·시트 토큰·gate exempt"""
import copy, json, os, subprocess, sys, tempfile
import torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve, stat_tag
from kdv.losses_rec import GTAnchoredReconstructionKD
from kdv.teacher_assets import load_run_model, load_donor_aligner, sha256_file, freeze, state_hash
from pa.aligner import PANGlobalAligner
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
cal = dict(tau_R=0.012, lambda_E=0.3); spec = {c: resolve(G.kdv_block(c, SEED, SRV, cal=cal)) for c in G.CASES if c not in G.QEDGE9_CASES and c not in G.QEGX_CASES}      # QEDGE9/QEGX case 는 W104 전용 (K23/K34)
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
_mand = G.mandatory_for(SRV)          # s1–s3: J0/JQ · s4: J0/JQ/AL0/ALQ (s4 보고 P-3)
check(f"K08 config 는 서버 공용(remaining_mandatory 비움 + 서버 로컬 파일 참조) · total 50h / reserve 4h (학습 admission ≤ 46h)", kb["budget"]["remaining_mandatory"] == [] and kb["budget"]["remaining_mandatory_file"] == G.MANDATORY_FILE and kb0["budget"]["remaining_mandatory"] == [] and kb["budget"]["total_gpu_hours"] == 50.0 and kb["budget"]["reserve_hours"] == 4.0)
_mf = os.path.join(ROOT, G.MANDATORY_FILE); _mf_bak = open(_mf).read() if os.path.exists(_mf) else None
try:
    G.write_mandatory_file(SRV); _lines = [l.strip() for l in open(_mf) if l.strip() and not l.startswith("#")]
    from train_kdv import KDVTrainer as _KT
    _st = object.__new__(_KT); _st.budget = dict(kb["budget"]); _st.run_id = G.to_tag(_mand[0], SEED)
    check(f"K08 prepare 가 쓰는 서버 로컬 파일 = {'/'.join(_mand)} (seed {SEED}) · trainer 는 파일에서 읽고 자기 자신을 뺀다 (s5 보고 #2: s1/s4·s3/s5 가 config 를 공유해도 예약이 섞이지 않는다)",
          _lines == [G.to_tag(c, SEED) for c in _mand] and _st._remaining_mandatory() == [G.to_tag(c, SEED) for c in _mand if c != _mand[0]])
    _st.budget["remaining_mandatory_file"] = "work_dir/_pakd50/_does_not_exist.txt"; _st.budget["remaining_mandatory"] = ["X", _st.run_id]
    check("K08 파일이 없으면 config 목록(자기 제외) 으로 후퇴", _st._remaining_mandatory() == ["X"])
finally:
    if _mf_bak is not None:
        open(_mf, "w").write(_mf_bak)
    elif os.path.exists(_mf):
        os.remove(_mf)
_ = [G.kdv_block(c, SEED, s) for s in G.SERVER_SEED for c in ("J0",)]
check("K08 같은 seed 서버(s1/s4, s3/s5) 의 J0 config 가 서버와 무관하게 같다 (공유 파일 안전)", all(G.kdv_block("J0", G.SERVER_SEED[s], s) == G.kdv_block("J0", G.SERVER_SEED[s2], s2) for s, s2 in (("s1", "s4"), ("s3", "s5"))))
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
check("K09 s4: seed 1234(s1 과 같은 run id), T0 경로는 모든 서버에서 자산 사본, 2026-09-14 기본 묶음 J0→JQ→AL0→ALQ(완료; PREVIOUS_PRIORITY_BY_SERVER), stage 1 = J0/AL0", G.SERVER_SEED["s4"] == 1234 and G.t0_dir("s1") == G.t0_dir("s4") == G.T0_ASSET_DIR and G.PREVIOUS_PRIORITY_BY_SERVER["s4_20260914"] == ["J0", "JQ", "AL0", "ALQ"] and G.stage_cases("s4", 1) == ["J0", "AL0"] and G.priority_for("s9") == G.PRIORITY and G.priority_for("s1") == G.QEDGE9_ITEMS_BY_SERVER["s1"])
check("K09 s4 기본 묶음(예산 예약, 서버 로컬 파일로) = 현재 명시 순서 (mandatory_for == priority_for; 09-15 재배정 묶음은 PREVIOUS['s4']) · config 자체에는 서버별 목록 없음", G.mandatory_for("s4") == G.priority_for("s4") and G.PREVIOUS_PRIORITY_BY_SERVER["s4"] == ["F0", "RC0", "RCQ", "JR", "XJ", "J_R3_NOEDGE"] and G.kdv_block("ALQ", 1234, "s4", cal=cal4)["budget"]["remaining_mandatory"] == [])
ex = ["J_QA05", G.run_name("JQ", 3407), "J_QA05"]
check("K09 추가 편성: case id 와 전체 run 이름 혼용, 기본 묶음 뒤·중복 제거; λE 없으면 λE case 는 미편성", G.schedule(True, term({"J0", "JQ", "AL0", "ALQ"}), 40.0, 1.9, priority=G.PREVIOUS_PRIORITY_BY_SERVER["s4_20260914"], extra=ex) == (["J_QA05", G.run_name("JQ", 3407)], []) and G.schedule(False, term({"J0", "AL0"}), 40.0, 1.9, priority=G.PREVIOUS_PRIORITY_BY_SERVER["s4_20260914"], extra=ex) == ([], []))
check("K09 to_tag/case_of", G.to_tag("J_QA05", 3407) == G.run_name("J_QA05", 3407) and G.case_of(G.run_name("AL_QE10", 3407)) == "AL_QE10" and G.case_of("JQ") == "JQ")
# ---------------- K10–K12 s5 timing/routing (research_log/PAN_S5_Timing_Routing_Experiment_Plan_2026-09-14.md §4–§6, §10.2–10.4; S5-G02/G03/G05/G06/G07)
import types
from train_kdv import KDVTrainer
def _res(kk):
    try:
        resolve(kk); return True
    except ValueError:
        return False
base5 = G.kdv_block("JQ", 2026, "s5", cal=cal4); sp5 = resolve(base5)
check("K10 registry: aligner_schedule/routing 기본값은 J 와 같다 (None, (1,1,1))", sp5["aligner_freeze_until"] is None and sp5["aligner_freeze_from"] is None and tuple(sp5["route_A"]) == (1.0, 1.0, 1.0))
_bad5 = [dict(base5, aligner_schedule=dict(freeze_until=5000, freeze_from=100)), dict(base5, aligner_schedule=dict(warm=1)), dict(base5, aligner_schedule=dict(freeze_until=-1)),
         dict(base5, routing=dict(qX=0)), dict(base5, routing=dict(qD=2.0)), dict(G.kdv_block("J0", 2026, "s5", cal=cal4), routing=dict(qD=0.0)), dict(G.kdv_block("JR", 2026, "s5", cal=cal4), routing=dict(qK=0.0)),
         dict(G.kdv_block("JR", 2026, "s5", cal=cal4), routing=dict(qE=0.0)), dict(G.kdv_block("FQ", 2026, "s5", cal=cal4), routing=dict(qD=0.0)), dict(G.kdv_block("FQ", 2026, "s5", cal=cal4), aligner_schedule=dict(freeze_until=5000))]
check("K10 registry 거부(S5-G07): 순서 역전·알 수 없는 키·음수·qX·범위 밖·N0 위 qD·R1 위 qK·edge 없는 qE·A-FR 위 routing/schedule (10 건 전부 ValueError)", all(not _res(b) for b in _bad5))
d0, dq, pq, px, lfq, jk0, je0, dpq = (G.kdv_block(c, 2026, "s5", cal=cal4) for c in ("D0", "DQ", "PQ", "PX", "LFQ", "JK0", "JE0", "DPQ"))
check("K11 D0/DQ: freeze_until 5000 · routing 없음 · A LR 1e-5 · 계수 JQ 와 같음 · control D0", d0["aligner_schedule"] == dict(freeze_until=5000) and "routing" not in d0 and dq["aligner_lr"] == 1e-5 and dq["rec"]["kd_weight"] == 0.1 and dq["baseline_run"] == G.run_name("D0", 2026))
check("K11 PQ: qD=qK=qE=0 · 일정 없음 · control J0 | PX(X02): qK 없음 | JK0: qK=0 만 | JE0: qE=0 만 | DPQ: 일정+routing", pq["routing"] == dict(qD=0.0, qK=0.0, qE=0.0) and "aligner_schedule" not in pq and pq["baseline_run"] == G.run_name("J0", 2026)
      and px["routing"] == dict(qD=0.0, qE=0.0) and jk0["routing"] == dict(qK=0.0) and je0["routing"] == dict(qE=0.0) and dpq["aligner_schedule"] == dict(freeze_until=5000) and dpq["routing"]["qD"] == 0.0)
check("K11 LFQ: freeze_from 25000 · s5 seed 2026 · 2026-09-14 기본 묶음 J0→JQ→D0→DQ→PQ(PREVIOUS) · stage 1 = J0/D0 · s5 case 전부 registry 통과", lfq["aligner_schedule"] == dict(freeze_from=25000) and G.SERVER_SEED["s5"] == 2026 and G.PREVIOUS_PRIORITY_BY_SERVER["s5"] == ["J0", "JQ", "D0", "DQ", "PQ"] and G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"] == ["PQ", "F0", "LF0", "LFQ", "RC0", "RCQ"]
      and G.stage_cases("s5", 1) == ["J0", "D0"] and all(_res(G.kdv_block(c, 2026, "s5", cal=cal4)) for c in ("D0", "DQ", "DR", "DX", "PQ", "PR", "PX", "DPQ", "DPX", "LF0", "LFQ", "JK0", "JE0")))
def _stub(case, seed_gen=3234):        # 실제 KDVTrainer._step 을 CPU 에서 (감사 verify_readonly.py 와 같은 방식). Student = T0 복사 + U 에 작은 잡음 (e_S ≠ e_T 라 soft 항이 살아 있게)
    tr = object.__new__(KDVTrainer); tr.k = G.kdv_block(case, 2026, "s5", cal=cal4); sp_ = tr.spec = resolve(tr.k)
    tr.model = copy.deepcopy(T0).train().requires_grad_(True)
    with torch.no_grad():
        for p_ in tr.model.backbone.parameters():
            p_.add_(0.01 * torch.randn(p_.shape, generator=torch.Generator().manual_seed(7)))
    tr.accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, gradient_accumulation_steps=1, scaler=None, is_main_process=True)
    tr.teacher = T0; tr.aligner_trainable = sp_["aligner_trainable"]; tr.aligner_view_margin = 4; tr.share_correction = False
    tr.protocol = sp_["protocol"]; tr.radius_hr = sp_["radius_hr"]; tr.diag_every = 1000
    tr.rec_crit = (GTAnchoredReconstructionKD(cal4["tau_R"], alpha=float(tr.k["rec"].get("alpha", 1.0)), kd_weight=float(tr.k["rec"].get("kd_weight", 0.0)), eps=1e-6, mode=sp_["rec_mode"]) if sp_["rec_case"] != "N0" else None)
    tr.tri = sp_["tri"]; tr.stat_extra = []; tr.lam_V = (cal4["lambda_E"] if sp_["stat_enabled"] else 0.0); tr.stat_ramp = 0; tr.lam_edge = 0; tr.lam_geo = 0; tr.ramp = 5000; tr.lam_gkd = 0
    tr.gen = torch.Generator().manual_seed(seed_gen); tr.corr_seed = seed_gen; tr._ema = {}; tr._rr_val_last = float("nan"); tr.args = types.SimpleNamespace(num_iter=50000)
    tr.freeze_until, tr.freeze_from = sp_["aligner_freeze_until"], sp_["aligner_freeze_from"]; tr.route_A = tuple(sp_["route_A"]); tr._routed = any(q != 1.0 for q in tr.route_A); tr._sched_last = None
    return tr
g5 = torch.Generator().manual_seed(11)
inp5 = (torch.rand(2, 8, 64, 64, generator=g5), torch.rand(2, 8, 16, 16, generator=g5), torch.rand(2, 1, 16, 16, generator=g5), torch.rand(2, 1, 64, 64, generator=g5))   # gt, ms, lpan, pan
def _grads(loss, params):
    return [x if x is not None else torch.zeros_like(p) for x, p in zip(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True), params)]
def _routed(case, keep):               # keep = (LD, LK, LEw) 중 A 가 그대로 받는 항
    tr = _stub(case); total, info = tr._step(*inp5, 1)                   # update 1 (odd): offset 연습 포함
    ap = list(tr.M.aligner.parameters()); bp = list(tr.M.backbone.parameters())
    l0, ld, lk, le, lo = info["_l0_t"], info["_rec_hard_t"] - info["_l0_t"], info["_rec_soft_t"], info["_edge_w_t"], float(info["lam_off"]) * info["loss_off"]
    manual_A = l0 + lo + (ld if keep[0] else 0.0) + (lk if keep[1] else 0.0) + (le if keep[2] else 0.0)
    want_A = _grads(manual_A, ap); want_U = _grads(total, bp); lkA = float(sum(w.abs().sum() for w in _grads(lk, ap)))   # graph 가 살아 있을 때 먼저 (routing 은 graph 를 해제한다)
    total.backward(retain_graph=True); n = tr._apply_routing(info, tr.M)
    ref = max(float(w.abs().max()) for w in want_A)
    errA = max(float((p.grad - w).abs().max()) for p, w in zip(ap, want_A)); errU = max(float((p.grad - w).abs().max()) for p, w in zip(bp, want_U))
    return errA, errU, n, ref, lkA
eA, eU, n, ref, lkA = _routed("PQ", (False, False, False))
check(f"K12 PQ(S5-G05): backward 뒤 routing → A 의 grad = ∇φ(L0 + λ_off L_O) (LD/LK/λE LE 는 A 에 없음), U 의 grad = ∇θ L_Q 전체, step 은 한 번 (max|Δ| A {eA:.1e}/ref {ref:.1e}, U {eU:.1e}; soft→A 원 gradient 합 {lkA:.2e})", eA <= 1e-5 * (1.0 + ref) and eU == 0.0 and n > 0 and lkA > 0)
eA, eU, n, ref, _ = _routed("JK0", (True, False, True)); check(f"K12 JK0(S5-G06): A 는 L_K 만 제외 (max|Δ| {eA:.1e}/ref {ref:.1e})", eA <= 1e-5 * (1.0 + ref) and eU == 0.0)
eA, eU, n, ref, _ = _routed("JE0", (True, True, False)); check(f"K12 JE0(S5-G06): A 는 λE L_E 만 제외 (max|Δ| {eA:.1e}/ref {ref:.1e})", eA <= 1e-5 * (1.0 + ref) and eU == 0.0)
tJ = _stub("JQ"); check("K12 JQ(S5-G02): qA=(1,1,1) 이면 routing 비활성 · 일정 없음 → 기존 J 경로 그대로 (aligner_active 0/49999 True)", not tJ._routed and tJ.aligner_active(0) and tJ.aligner_active(49999))
tD, tL = _stub("D0"), _stub("LF0")
check("K12 D(S5-G03): aligner_active 4999 False / 5000 True / 49999 True · LF(S5-G04): 24999 True / 25000 False (0-based next update index)", (not tD.aligner_active(4999)) and tD.aligner_active(5000) and tD.aligner_active(49999) and tL.aligner_active(24999) and not tL.aligner_active(25000))
tD.M.aligner.requires_grad_(False); tJ0 = _stub("J0")
totD, infD = tD._step(*inp5, 4999); totJ, infJ = tJ0._step(*inp5, 4999); hA = state_hash(tD.M.aligner)
optD = torch.optim.AdamW([dict(params=list(tD.M.backbone.parameters())), dict(params=list(tD.M.aligner.parameters()), lr=1e-5)], lr=1e-4, weight_decay=0.01); totD.backward(); optD.step()
check("K12 동결 update 4999(odd; S5-G03/G08): offset 연습 생략(eq_exercise 0, L_O 0) · Δ 에 graph 없음 · U 는 학습 · AdamW step 뒤 A hash 불변(grad None → moment·WD 없음) · ε RNG 소비량은 J 와 같다",
      infD.get("off_skipped_frozen") == 1.0 and float(infD["loss_off"]) == 0.0 and not infD["delta"].requires_grad and totD.requires_grad and infJ.get("eq_exercise") == 1.0 and state_hash(tD.M.aligner) == hA
      and all(p.grad is None for p in tD.M.aligner.parameters()) and torch.equal(tD.gen.get_state(), tJ0.gen.get_state()))
tD.M.aligner.requires_grad_(True); tot5, inf5 = tD._step(*inp5, 5001)
check("K12 해제 뒤 update 5001(odd): offset 연습 재개, Δ 에 graph 있음, A 로 gradient 있음", inf5.get("eq_exercise") == 1.0 and inf5["delta"].requires_grad and any(x is not None and float(x.abs().sum()) > 0 for x in torch.autograd.grad(tot5, list(tD.M.aligner.parameters()), allow_unused=True)))
del tD, tL, tJ, tJ0, totD, infD, totJ, infJ, tot5, inf5
# ---------------- K13 고정 batch 진단이 graph 를 두 번 쓰지 않는다 (s5 보고 #1: sum-rule grad 뒤 loss 별 분해)
tQ = _stub("JQ"); tQ._fixed_batch = inp5; _recs = []; tQ._jsonl = lambda n, r: _recs.append((n, r))
try:
    tQ._diagnose_fixed(1, tQ.M); _pt = _recs[-1][1].get("per_term", {})
    check("K13 _diagnose_fixed(step 1, offset 연습 포함): 예외 없이 sum-rule + loss 별 A/U 분해(L0/LD/LK/LEw/LOw) 기록", _recs[-1][0] == "gradient_diagnostics_fixed.jsonl" and "sum_rule_max_abs_err" in _recs[-1][1]
          and all(k in _pt for k in ("L0_A_norm", "LD_A_norm", "LK_A_norm", "LEw_A_norm", "LOw_A_norm", "L0_U_norm", "LK_U_cos_L0")) and _pt["LOw_U_norm"] == 0.0)
except Exception as e:
    check("K13 _diagnose_fixed(step 1, offset 연습 포함): 예외 없이 sum-rule + loss 별 A/U 분해 기록", False, repr(e)[:160])
del tQ
# ---------------- K14 동결 경계 직전 중단·재개 (S5-G09; s5 보고 #8): 연속 실행 vs 4999 뒤 재개 — aligner_active·ε 열·Δ 가 같다 (optimizer/scheduler 복원은 accelerate save_state 의 몫)
def _run(tr, steps):
    out = {}
    for s in steps:
        tr.M.aligner.requires_grad_(tr.aligner_active(s)); tot, inf = tr._step(*inp5, s)
        out[s] = dict(eps=(inf["eps"].clone() if torch.is_tensor(inf.get("eps")) else None), delta=inf["delta"].detach().clone(), eq=inf.get("eq_exercise", 0.0), act=inf["aligner_active"], tot=float(tot))
    return out
a = _stub("D0"); cont = _run(a, [4998, 4999, 5000, 5001])
b = _stub("D0"); _run(b, [4998, 4999]); b2 = _stub("D0"); b2.model = copy.deepcopy(b.model); b2.gen.set_state(b.gen.get_state()); res = _run(b2, [5000, 5001])
check("K14 4999 뒤 재개(모델·ε RNG 상태 복원) == 연속 실행: 5000 은 A 활성·offset 없음(even), 5001 은 offset ε 동일·Δ 동일·loss 동일",
      all(res[s]["act"] == cont[s]["act"] and res[s]["eq"] == cont[s]["eq"] and torch.equal(res[s]["delta"], cont[s]["delta"]) and abs(res[s]["tot"] - cont[s]["tot"]) < 1e-9 for s in (5000, 5001))
      and torch.equal(res[5001]["eps"], cont[5001]["eps"]) and cont[4999]["act"] == 0.0 and cont[5000]["act"] == 1.0 and cont[5000]["eq"] == 0.0 and cont[5001]["eq"] == 1.0)
del a, b, b2
# ---------------- K16 seed 별 U 초기값 hash pin (s5 보고 #3)
# ---------------- K17–K20 2026-09-15 재배정 (research_log/PAN_PAKD50_S2_S4_S5_Derived_Run_Allocation_2026-09-15.md §3–§5, §7–§9)
rc0, rcq, jr3, jn0 = (G.kdv_block(c, 2026, "s5", cal=cal4) for c in ("RC0", "RCQ", "J_R3_NOEDGE", "J_N0_EDGE")); s_rc0, s_rcq, s_jr3, s_jn0 = (resolve(x) for x in (rc0, rcq, jr3, jn0))
check("K17 RC(§5 semantic fragment): A-FT + I-NATIVE-TRANSFER + A LR 1e-5 + corruption.radius_hr 0 + offset 0 · A trainable · 일정/routing 없음 · control RC0 · RCQ 는 Q12 식 그대로(α1 β0.1 λE0)",
      all(x["aligner_policy"] == "A-FT" and x["input_protocol"] == "I-NATIVE-TRANSFER" and x["aligner_lr"] == 1e-5 and x["corruption"] == dict(radius_hr=0.0) and x["aux"]["offset_weight"] == 0.0 and "aligner_schedule" not in x and "routing" not in x for x in (rc0, rcq))
      and s_rc0["aligner_trainable"] and s_rcq["aligner_trainable"] and s_rc0["offset_weight_effective"] == 0 == s_rc0["radius_hr"] and rc0["baseline_run"] == rcq["baseline_run"] == G.run_name("RC0", 2026)
      and rc0["rec"]["case"] == "N0" and rc0["teacher"].get("eval_only") and rcq["rec"] == G.kdv_block("JQ", 2026, "s5", cal=cal4)["rec"] and rcq["stat"] == G.kdv_block("JQ", 2026, "s5", cal=cal4)["stat"])
check("K17 registry 계약: I-AEQ 에 offset 0 만 넣는 방식은 거부 (그래서 RC 는 protocol 을 바꾼다) · I-NATIVE-TRANSFER 에 radius > 0 도 거부", not _res(dict(G.kdv_block("J0", 2026, "s5", cal=cal4), aux=dict(offset_weight=0.0, geometry_weight=0.0)))
      and not _res(dict(rc0, corruption=dict(radius_hr=2.0))))
check("K17 J_R3_NOEDGE = JQ 에서 edge 만 제거 (rec R3 α1 β0.1 · stat OFF · Teacher 필요 · J 정책 그대로) · J_N0_EDGE = J0 + λE edge (rec N0 · EDGE-H λE0 · Teacher eval_only, loss 에 없음)",
      jr3["rec"] == G.kdv_block("JQ", 2026, "s5", cal=cal4)["rec"] and not jr3["stat"]["enabled"] and s_jr3["needs_teacher"] and jr3["input_protocol"] == "I-AEQ" and jr3["aux"]["offset_weight"] == 1e-4
      and jn0["rec"] == dict(case="N0") and jn0["stat"]["enabled"] and jn0["stat"]["outer_weight"] == 0.3 and s_jn0["teacher_eval_only"] and not s_jn0["needs_teacher"] and jn0["baseline_run"] == jr3["baseline_run"] == G.run_name("J0", 2026))
check("K17 λE 의존: J_N0_EDGE·RCQ 는 λE 필요, J_R3_NOEDGE·RC0 는 τR/없음 · 시간 산정 유형: RC0/F0/LF0 = N0 형, RCQ/LFQ/JR/XJ/J_R3_NOEDGE/J_N0_EDGE = T 형, PQ = ROUTING",
      {"J_N0_EDGE", "RCQ"} <= G.NEEDS_LAMBDA_E and not ({"J_R3_NOEDGE", "RC0"} & G.NEEDS_LAMBDA_E) and all(G.reference_kind(c) == "N0" for c in ("RC0", "F0", "LF0")) and all(G.reference_kind(c) == "T" for c in ("RCQ", "LFQ", "JR", "XJ", "J_R3_NOEDGE", "J_N0_EDGE")) and G.reference_kind("PQ") == "ROUTING")
# K18 RC trainer step: native 만(corrupt 없음) · offset 연습 없음(ε RNG 소비 없음) · L_rec → A gradient 유지 · RCQ 는 soft/edge 도 A 로 (routing 없음)
tR, tJ = _stub("RC0"), _stub("J0"); g_before = tR.gen.get_state().clone()
totR, infR = tR._step(*inp5, 1); totJ, infJ = tJ._step(*inp5, 1); ap = list(tR.M.aligner.parameters()); bp = list(tR.M.backbone.parameters())
gA = torch.autograd.grad(totR, ap, retain_graph=True, allow_unused=True); gU = torch.autograd.grad(totR, bp, retain_graph=True, allow_unused=True)
check("K18 RC0 update 1(odd): corrupt False · eq_exercise 없음 · L_O 0 · ε RNG 미소비 (J0 은 같은 update 에 연습) · Δ 에 graph · L_rec → A 와 U 양쪽 gradient · aligner_active 0/49999",
      not infR["corrupt"] and "eq_exercise" not in infR and float(infR["loss_off"]) == 0.0 and torch.equal(tR.gen.get_state(), g_before) and infJ.get("eq_exercise") == 1.0 and infR["delta"].requires_grad
      and any(x is not None and float(x.abs().sum()) > 0 for x in gA) and any(x is not None and float(x.abs().sum()) > 0 for x in gU) and tR.aligner_active(0) and tR.aligner_active(49999))
tQ = _stub("RCQ"); totQ, infQ = tQ._step(*inp5, 1); apQ = list(tQ.M.aligner.parameters())
lkA = float(sum(w.abs().sum() for w in _grads(infQ["_rec_soft_t"], apQ))); leA = float(sum(w.abs().sum() for w in _grads(infQ["_edge_w_t"], apQ)))
check(f"K18 RCQ: Q12 전체(hard+soft+λE edge) 가 A 로도 간다 (routing 없음; soft→A {lkA:.2e}, edge→A {leA:.2e}) · offset 없음 · Teacher forward 있음", lkA > 0 and leA > 0 and float(infQ["loss_off"]) == 0.0 and infQ["y_t"] is not None)
tE = _stub("J_N0_EDGE"); totE, infE = tE._step(*inp5, 1)
check("K18 J_N0_EDGE: Teacher forward 없음(y_t None) · edge 항 > 0 · offset 연습은 J 와 같음(odd 에 ε)", infE["y_t"] is None and float(infE["_edge_w_t"]) > 0 and infE.get("eq_exercise") == 1.0 and float(infE.get("rec_soft", 0.0)) == 0.0)
del tR, tJ, tQ, tE, totR, totQ, totE, totJ
# K19 서버별 명시 순서·예약식 (§3–§4 검산)·admission (§8)·확인 seed (§7)
check("K19 명시 순서(§4): s2 JR→XJ→J_R3_NOEDGE→J_N0_EDGE · s4 F0→RC0→RCQ→JR→XJ→J_R3_NOEDGE · s5 PQ→F0→LF0→LFQ→RC0→RCQ; mandatory == priority; 항목 없는 서버는 기본 PRIORITY(s1 은 17:20 QEDGE9 v2 목록; s3 는 09-15 s3 추가 전 기본); 완료 묶음은 순서에 없다(FR 도)",
      G.priority_for("s2") == ["JR", "XJ", "J_R3_NOEDGE", "J_N0_EDGE"] and G.PREVIOUS_PRIORITY_BY_SERVER["s4"] == ["F0", "RC0", "RCQ", "JR", "XJ", "J_R3_NOEDGE"] and G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"] == ["PQ", "F0", "LF0", "LFQ", "RC0", "RCQ"]
      and all(G.mandatory_for(s) == G.priority_for(s) for s in ("s2", "s4", "s5", "s1")) and G.priority_for("s9") == G.PRIORITY and G.priority_for("s1") == G.QEDGE9_ITEMS_BY_SERVER["s1"] and G.PREVIOUS_PRIORITY_BY_SERVER["s3"] == G.PRIORITY and not ({"J0", "JQ", "F0", "FQ", "FR"} & set(G.priority_for("s2"))) and "PQ" in G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"])
_sum = lambda srv: sum(G.reservation_for(srv, c)["reservation_h"] for c in (G.PREVIOUS_PRIORITY_BY_SERVER["s4"] if srv == "s4" else (G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"] if srv == "s5" else G.priority_for(srv))))
check("K19 예약식 reservation_h = 1.10×ref + 10/60 (§3 검산): s2 4×2.33 → 10.9187 · s4 2×1.17+4×1.39 → 9.6900 · s5 1.80+3×1.35+2×1.36 → 10.4270 (측정값 없이)",
      abs(_sum("s2") - 10.9186667) < 1e-6 and abs(_sum("s4") - 9.69) < 1e-6 and abs(_sum("s5") - 10.427) < 1e-6 and abs(G.reservation_hours(2.33) - 2.7296667) < 1e-6)
check("K19 기준값 출처: N0 형/T 형은 서버 계획값(s2 는 전부 2.33) · PQ 는 routing 가예약 1.80 · 같은 서버 같은 case 실측이 있으면 그것 · routing 실측이 생기면 다른 routing case 도 그 평균 · 확인 seed 는 1.80 가예약",
      G.reference_hours("s2", "JR") == (2.33, "plan_reference") and G.reference_hours("s4", "RC0") == (1.17, "plan_reference") and G.reference_hours("s4", "RCQ") == (1.39, "plan_reference") and G.reference_hours("s5", "PQ") == (1.8, "routing_placeholder")
      and G.reference_hours("s5", "PQ", {"PQ": 1.9}) == (1.9, "measured_same_case") and G.reference_hours("s5", "PX", {"PQ": 1.9})[1] == "measured_routing_mean" and G.reference_hours("s5", "RCQ", {}, confirm=True) == (1.8, "confirm_placeholder")
      and G.reference_hours("s5", "RCQ", {"RCQ": 1.4}, confirm=True) == (1.4, "measured_same_case") and G.reservation_for("s5", G.run_name("RCQ", 9091))["seed"] == 9091 and G.reservation_for("s5", G.run_name("RCQ", 9091))["reference_kind"] == "confirm_placeholder")
_est = lambda srv: (lambda it: G.reservation_for(srv, it)["reservation_h"])
check("K19 편성: s2 완료(J0/F0/JQ/FQ) 뒤 남은 전부 명시 순서 · s4/s5 도 · 실행 중 PQ 는 제외(그 뒤부터) · s2 남은 5.0h 면 JR 만(2.73+2.73 > 5) · est callable 이면 margin 을 다시 곱하지 않는다",
      G.schedule(True, term({"J0", "F0", "JQ", "FQ"}), 40.0, _est("s2"), priority=G.priority_for("s2")) == (["JR", "XJ", "J_R3_NOEDGE", "J_N0_EDGE"], [])
      and G.schedule(True, term({"AL0", "J0", "JQ", "ALQ"}), 40.0, _est("s4"), priority=G.PREVIOUS_PRIORITY_BY_SERVER["s4"]) == (["F0", "RC0", "RCQ", "JR", "XJ", "J_R3_NOEDGE"], [])
      and G.schedule(True, term({"J0", "JQ", "D0", "DQ", "PQ"}), 40.0, _est("s5"), priority=G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"]) == (["F0", "LF0", "LFQ", "RC0", "RCQ"], [])
      and G.schedule(True, term({"J0", "F0", "JQ", "FQ"}), 5.0, _est("s2"), priority=G.priority_for("s2")) == (["JR"], ["XJ", "J_R3_NOEDGE", "J_N0_EDGE"])
      and G.schedule(True, term({"J0", "F0", "JQ", "FQ"}), 5.46, _est("s2"), priority=G.priority_for("s2")) == (["JR", "XJ"], ["J_R3_NOEDGE", "J_N0_EDGE"]))
check("K19 확인 seed(§7): s4 3407 / s5 9091 · 묶음 = WIN → 같은 policy no-KD control → F0 (F0 가 control 이면 2 run) · 최대 3", G.CONFIRM_SEED["s4"] == 3407 and G.CONFIRM_SEED["s5"] == 9091 and G.confirmation_cases("RCQ") == ["RCQ", "RC0", "F0"] and G.confirmation_cases("LFQ") == ["LFQ", "LF0", "F0"]
      and G.confirmation_cases("F0") == ["F0"] and G.confirmation_cases("JQ") == ["JQ", "J0", "F0"] and G.CONFIRM_MAX_RUNS == 3)
# K21 s3 추가 (research_log/PAN_PAKD50_Latest_Sheet_Analysis_and_S3_Experiments_2026-09-15.md §4–§7): LFX 등록 · s3 명시 순서·예약 검산 · 확인 seed 4321 후보별 묶음
lfx = G.kdv_block("LFX", 2026, "s3", cal=cal4); s_lfx = resolve(lfx); xj = G.kdv_block("XJ", 2026, "s3", cal=cal4); lfq = G.kdv_block("LFQ", 2026, "s3", cal=cal4)
check("K21 LFX = LF 일정(freeze_from 25000) + X02(rec R1 hard-only α1 β0 + EDGE-H λE0, soft 없음) · routing 없음 · offset 1e-4(I-AEQ) · control LF0 · λE 필요 · 0–24999 계약이 XJ 와 같다(일정 키만 추가) · Teacher 필요(R1 재가중)",
      lfx["aligner_schedule"] == dict(freeze_from=25000) and lfx["rec"] == xj["rec"] and lfx["stat"] == xj["stat"] and "routing" not in lfx and lfx["aux"]["offset_weight"] == 1e-4 and lfx["baseline_run"] == G.run_name("LF0", 2026) and "LFX" in G.NEEDS_LAMBDA_E
      and s_lfx["rec_mode"] == "hard_only" and s_lfx["stat_enabled"] and s_lfx["needs_teacher"] and s_lfx["aligner_freeze_from"] == 25000 and {k: v for k, v in lfx.items() if k not in ("aligner_schedule", "case_id", "baseline_run", "policy_id", "budget")} == {k: v for k, v in xj.items() if k not in ("case_id", "baseline_run", "policy_id", "budget")}
      and lfq["rec"] == G.kdv_block("JQ", 2026, "s3", cal=cal4)["rec"] and G.reference_kind("LFX") == "T" and G.reference_kind("LF0") == "N0")
_s3 = lambda it: G.reservation_for("s3", it)["reservation_h"]
check("K21 s3 2026-09-15 낮 순서 J_R3_NOEDGE→J_N0_EDGE→LF0→LFQ→LFX (완료 → PREVIOUS['s3_20260915']; 현재 순서는 QEGX K34) · mandatory == priority · 기존 control(J0/JQ/JR/XJ/F0/FQ/FR) 은 순서에 없다 · reference 1.34/1.34/1.16(J0)/1.34/1.33(XJ, case 대용값) · 예약 합 7.9943 h(§6 검산)",
      G.PREVIOUS_PRIORITY_BY_SERVER["s3_20260915"] == ["J_R3_NOEDGE", "J_N0_EDGE", "LF0", "LFQ", "LFX"] and G.mandatory_for("s3") == G.priority_for("s3") and not (set(G.PRIORITY) & set(G.priority_for("s3")))
      and G.reference_hours("s3", "LFX") == (1.33, "plan_reference_case") and G.reference_hours("s3", "LF0") == (1.16, "plan_reference") and G.reference_hours("s3", "J_N0_EDGE") == (1.34, "plan_reference")
      and abs(sum(_s3(c) for c in G.PREVIOUS_PRIORITY_BY_SERVER["s3_20260915"]) - 7.9943333) < 1e-6 and G.schedule(True, term({"J0", "JQ", "JR", "XJ", "F0", "FQ", "FR"}), 40.0, _s3, priority=G.PREVIOUS_PRIORITY_BY_SERVER["s3_20260915"]) == (["J_R3_NOEDGE", "J_N0_EDGE", "LF0", "LFQ", "LFX"], [])
      and "s3" in G.ALLOCATED_SERVERS)
check("K21 s3 확인(§5): seed 4321(3407/9091 과 분리) · 후보별 묶음 LFQ→JQ/LF0/LFQ · LFX→XJ/LF0/LFX · J_N0_EDGE→J0/XJ/J_N0_EDGE · J_R3_NOEDGE→J0/JR/J_R3_NOEDGE · XJ→J0/JR/XJ · JQ→J0/XJ/JQ · 표 밖 후보(RC0) 거부 · 확인 reference 1.34 (s4 는 종전 규칙·1.80)",
      G.CONFIRM_SEED["s3"] == 4321 and len({G.CONFIRM_SEED[s] for s in ("s3", "s4", "s5")}) == 3 and G.confirmation_cases("LFQ", "s3") == ["JQ", "LF0", "LFQ"] and G.confirmation_cases("LFX", "s3") == ["XJ", "LF0", "LFX"] and G.confirmation_cases("J_N0_EDGE", "s3") == ["J0", "XJ", "J_N0_EDGE"]
      and G.confirmation_cases("J_R3_NOEDGE", "s3") == ["J0", "JR", "J_R3_NOEDGE"] and G.confirmation_cases("XJ", "s3") == ["J0", "JR", "XJ"] and G.confirmation_cases("JQ", "s3") == ["J0", "XJ", "JQ"]
      and (lambda: (lambda f: (f() and False))(lambda: G.confirmation_cases("RC0", "s3")) if False else True)() and G.reference_hours("s3", "LFQ", {}, confirm=True) == (1.34, "confirm_reference") and G.reference_hours("s5", "RCQ", {}, confirm=True) == (1.8, "confirm_placeholder") and G.confirmation_cases("RCQ", "s5") == ["RCQ", "RC0", "F0"])
try:
    G.confirmation_cases("RC0", "s3"); check("K21 s3 확인 표 밖 후보는 SystemExit", False)
except SystemExit:
    check("K21 s3 확인 표 밖 후보는 SystemExit", True)
# K22 s4 아키텍처 이식 (research_log/PAN_PAKD50_S4_W104D121_Architecture_Allocation_2026-09-15.md §3–§6, §8–§9): W104·D121 Student U 만, T0/A·τR·λE0 고정, NA0 등록, 골격 인지 이름·예약·확인
import yaml as _yaml
_tpl = _yaml.safe_load(open(os.path.join(ROOT, "config", "PO10_N1_REC_W112_D123_WV3_S2025_R200_FRSTAT.yaml"))); _M = import_class(_tpl["model"])
_m104 = _M(**dict(_tpl["model_args"], hidden_size=104, depth=[1, 2, 1])); _n104 = sum(p.numel() for p in _m104.parameters()) / 1e6
check("K22 골격: template model_args 에서 hidden_size 104·depth [1,2,1] 만 바꾸면 backbone 1.9036 M (s3 GT-only 기록과 같은 lineage; 나머지 in_mode/norm/attn 은 template 그대로) · aligner 0.1053 M", abs(_n104 - 1.9036) < 5e-4 and abs(sum(p.numel() for p in PANGlobalAligner(8).parameters()) / 1e6 - 0.10533) < 1e-5, f"{_n104:.4f} M")
_A = G.ARCHS["W104_D121"]; _kJ = G.kdv_block("JQ", 1234, "s4", cal=cal4, arch="W104_D121"); _kN = G.kdv_block("NA0", 1234, "s4", cal=cal4, arch="W104_D121"); _kF = G.kdv_block("F0", 1234, "s4", cal=cal4, arch="W104_D121"); _sJ, _sN, _sF = resolve(_kJ), resolve(_kN), resolve(_kF)
check("K22 W104 JQ/F0 block: expect_arch (104,[1,2,1],noalign False) · T0 donor/teacher(W112) 그대로 · τR/λE0 그대로 · branch A104D121_T0FIX_E0_v1 · init hash 는 'unet@W104_D121' namespace(미기록이면 expect_init 없음) · 이름 PAKD50_JQ_W104_D121_…",
      _kJ["expect_arch"] == dict(width=104, depth=[1, 2, 1], noalign=False) and _kJ["donor"] == G.kdv_block("JQ", 1234, "s4", cal=cal4)["donor"] and {k: v for k, v in _kJ["teacher"].items() if k != "bridge"} == {k: v for k, v in G.kdv_block("JQ", 1234, "s4", cal=cal4)["teacher"].items() if k != "bridge"} and _kJ["teacher"]["bridge"] is True and G.kdv_block("JQ", 1234, "s4", cal=cal4)["teacher"]["bridge"] is False and _kJ["rec"] == G.kdv_block("JQ", 1234, "s4", cal=cal4)["rec"] and _kJ["stat"] == G.kdv_block("JQ", 1234, "s4", cal=cal4)["stat"]
      and _kJ["experiment_branch_id"] == "A104D121_T0FIX_E0_v1" and "expect_init" not in _kJ and G.init_hash_for(1234) == "c988a6c95b17f4fd" and G.init_hash_for(1234, "W104_D121") is None and _kJ["campaign_id"] == G.CAMPAIGN_ID
      and G.run_name("JQ", 1234, arch="W104_D121") == "PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v1" and _kJ["baseline_run"] == G.run_name("J0", 1234, arch="W104_D121") and _kF["expect_arch"]["noalign"] is False and _sF["policy"] == "A-FR" and not _sF["aligner_trainable"])
check("K22 NA0: A-ID/NOALIGN · I-A · na_protocol NA-STRICT · donor 없음 · aligner_lr 없음 · aligned_selector False · expect_arch noalign True · Teacher eval_only(loss 미사용) · control NA0 · fixed_reference_from_donor 없음",
      _sN["policy"] == "A-ID" and _kN["recipe"] == "NOALIGN" and _sN["protocol"] == "I-A" and _kN["na_protocol"] == "NA-STRICT" and "donor" not in _kN and "aligner_lr" not in _kN and _kN["select"]["aligned_selector"] is False and _kN["expect_arch"]["noalign"] is True
      and _sN["teacher_eval_only"] and not _sN["needs_teacher"] and _kN["baseline_run"] == G.run_name("NA0", 1234, arch="W104_D121") and "eval" not in _kN)
check("K22 항목 파싱: 'JQ@W104_D121' · run 이름 · 'JQ' → (case, arch, seed) · item_key · 알 수 없는 골격 거부 · case_of/arch_of/seed_of", G.parse_item("JQ@W104_D121") == ("JQ", "W104_D121", None, None) and G.parse_item(G.run_name("XJ", 3407, arch="W104_D121")) == ("XJ", "W104_D121", 3407, "v1")
      and G.parse_item("JQ") == ("JQ", "W112_D123", None, None) and G.item_key("JQ@W104_D121") == "JQ@W104_D121" and G.item_key("JQ") == "JQ" and G.case_of(G.run_name("J_R3_NOEDGE", 777, arch="W104_D121")) == "J_R3_NOEDGE" and G.arch_of("NA0@W104_D121") == "W104_D121"
      and G.to_tag("NA0@W104_D121", 1234) == "PAKD50_NA0_W104_D121_WV3_T0_S1234_FRESH50_v1")
try:
    G.parse_item("JQ@W999_D9"); check("K22 알 수 없는 골격은 ValueError", False)
except ValueError:
    check("K22 알 수 없는 골격은 ValueError", True)
_s4 = lambda it: G.reservation_for("s4", it)["reservation_h"]
check("K22 s4 명시 순서 앞 5 = NA0→J0→JQ→XJ→F0 (@W104_D121; 완료; 뒤 QEGX 14 항목은 K34) · 파생 묶음(F0/RC0/RCQ/JR/XJ/J_R3_NOEDGE@W112) 은 PREVIOUS · reference 는 QEGX §9.1 Sheet 실측 1.27/1.17/1.38/1.37/1.13(종전 대용 1.18/1.18/1.40/1.39/1.14 교체) → 예약 합 7.785333 h · 편성은 W112 완료 run 과 별개",
      G.priority_for("s4")[:5] == ["NA0@W104_D121", "J0@W104_D121", "JQ@W104_D121", "XJ@W104_D121", "F0@W104_D121"] and G.PREVIOUS_PRIORITY_BY_SERVER["s4"] == ["F0", "RC0", "RCQ", "JR", "XJ", "J_R3_NOEDGE"] and abs(sum(_s4(c) for c in G.priority_for("s4")[:5]) - 7.7853333) < 1e-6
      and G.reference_hours("s4", "JQ@W104_D121") == (1.38, "plan_reference_case") and G.reference_hours("s4", "JQ") == (1.39, "plan_reference") and G.schedule(True, term({G.run_name("F0", 1234), G.run_name("JQ", 1234)}), 40.0, _s4, priority=G.priority_for("s4")[:5]) == (G.priority_for("s4")[:5], [])
      and G.measured_hours_from_ledger("s4", dict(entries={G.run_name("JQ", 1234): dict(kind="run", status="FINISHED", hours=1.39), G.run_name("JQ", 1234, arch="W104_D121"): dict(kind="run", status="FINISHED", hours=0.9)})) == {"JQ": 1.39, "JQ@W104_D121": 0.9})
_cb = G.confirmation_cases("JQ", "s4"); _cr = [G.reservation_for("s4", G.to_tag(it, 3407)) for it in _cb]
check("K22 s4 옛 --confirm 묶음(§6) 은 그대로 정의(J0@W104 → WIN@W104 → J0@W112 → WIN@W112; 최대 4; 표 밖 후보 거부) — 단 seed 3407 은 이제 QEGX 명시 표의 허용 seed 라 예약은 confirm 가예약이 아니라 plan reference(J0@W104 1.17 · JQ@W104 1.38 · J0 1.17 · JQ 1.39 → 6.2877 h); QEGX §12: 옛 --confirm 은 명시 표를 대신하지 않는다",
      _cb == ["J0@W104_D121", "JQ@W104_D121", "J0", "JQ"] and G.confirmation_cases("F0", "s4") == ["J0@W104_D121", "F0@W104_D121", "J0", "F0"] and G.CONFIRM_SEED["s4"] == 3407 and G.CONFIRM_MAX_RUNS_BY_SERVER["s4"] == 4
      and [r["run"] for r in _cr] == [G.run_name("J0", 3407, arch="W104_D121"), G.run_name("JQ", 3407, arch="W104_D121"), G.run_name("J0", 3407), G.run_name("JQ", 3407)] and abs(sum(r["reservation_h"] for r in _cr) - 6.2876667) < 1e-6
      and [r["reference_kind"] for r in _cr] == ["plan_reference_case", "plan_reference_case", "plan_reference", "plan_reference"] and G.reservation_for("s4", G.run_name("J0", 9999, arch="W104_D121"))["reference_kind"] == "confirm_reference_case")
try:
    G.confirmation_cases("RC0", "s4"); check("K22 s4 확인 표 밖 후보는 SystemExit", False)
except SystemExit:
    check("K22 s4 확인 표 밖 후보는 SystemExit", True)
_cfg104 = _yaml.safe_load(open(os.path.join(ROOT, "config", "PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v1.yaml"))); _cfgN = _yaml.safe_load(open(os.path.join(ROOT, "config", "PAKD50_NA0_W104_D121_WV3_T0_S1234_FRESH50_v1.yaml")))
check("K22 생성 config: model_args hidden_size 104 · depth [1,2,1] · 나머지 model_args 는 W112 config 와 같음 · kdv.expect_arch 일치 · trainer init_dir 는 _kdv_init_w104_d121(자동) · W112 config 는 회귀 없음(hidden 112)",
      _cfg104["model_args"]["hidden_size"] == 104 and _cfg104["model_args"]["depth"] == [1, 2, 1] and {k: v for k, v in _cfg104["model_args"].items() if k not in ("hidden_size", "depth")} == {k: v for k, v in _yaml.safe_load(open(os.path.join(ROOT, "config", G.run_name("JQ", 1234) + ".yaml")))["model_args"].items() if k not in ("hidden_size", "depth")}
      and _cfg104["kdv"]["expect_arch"] == dict(width=104, depth=[1, 2, 1], noalign=False) and _cfgN["kdv"]["aligner_policy"] == "A-ID" and _cfgN["model_args"]["hidden_size"] == 104 and _yaml.safe_load(open(os.path.join(ROOT, "config", G.run_name("JQ", 1234) + ".yaml")))["model_args"]["hidden_size"] == 112
      and f"work_dir/_kdv_init_w104_d121" == f"work_dir/_kdv_init_w{_cfg104['model_args']['hidden_size']}_d{''.join(str(d) for d in _cfg104['model_args']['depth'])}")
# W104 Student 의 실제 forward: T0 aligner strict copy (hash == T0 A) + W104 U (fresh) · Teacher 는 자기 W112 A/U · 출력 shape 8×64×64 · NA0 는 aligner/sampler 없음 · offset→A 만
from kdv.teacher_assets import load_donor_aligner
_ma = dict(_tpl["model_args"], hidden_size=104, depth=[1, 2, 1]); _u104 = _M(**_ma); _al, _ = load_donor_aligner(f"{G.T0_ASSET_DIR}/{G.T0_TAG}", 8)
from tools.smoke_cases import randomize_zero_params as _rz
_rz(_u104)                                                  # zero_module 트랩: 새 U 의 출력 conv 가 0 이면 A 로 가는 L_rec gradient 도 0 (CLAUDE.md 함정) — 난수화 뒤 검사
_S104 = PAModel(_u104, _al, aligner_margin=4, sampler=True); _N104 = PAModel(_M(**_ma), None, aligner_margin=0, sampler=False)
g4 = torch.Generator().manual_seed(5); _gt, _ms, _lp, _pn = (torch.rand(2, 8, 64, 64, generator=g4), torch.rand(2, 8, 16, 16, generator=g4), torch.rand(2, 1, 16, 16, generator=g4), torch.rand(2, 1, 64, 64, generator=g4))
_o = kdv_forward(_S104, T0, _pn, _ms, _lp, share_correction=False, teacher_needed=True, aligner_live=True); _oN = kdv_forward(_N104, None, _pn, _ms, _lp, share_correction=False, teacher_needed=False, aligner_live=False)
check("K22 W104 Student forward: A == T0 A(hash) · y [2,8,64,64] · Teacher(W112) 출력 shape 동일·Student 골격과 무관(같은 입력에서 T0 단독 forward 와 동일) · NA0 는 delta 0·aligner None · L_rec → W104 U 와 A 양쪽 gradient",
      state_hash(_S104.aligner) == state_hash(T0.aligner) and tuple(_o["y"].shape) == (2, 8, 64, 64) and tuple(_o["y_t"].shape) == (2, 8, 64, 64) and torch.equal(_o["y_t"], T0(_pn, _ms, _lp)["y"]) and _N104.aligner is None and float(_oN["delta"].abs().sum()) == 0 and tuple(_oN["y"].shape) == (2, 8, 64, 64)
      and any(x is not None and float(x.abs().sum()) > 0 for x in torch.autograd.grad((_o["y"] - _gt).abs().mean(), list(_S104.aligner.parameters()), retain_graph=True, allow_unused=True)) and any(x is not None and float(x.abs().sum()) > 0 for x in torch.autograd.grad((_o["y"] - _gt).abs().mean(), list(_u104.parameters()), allow_unused=True)))
# K20 trainer 예산 gate 가 서버 로컬 예약 파일을 본다 (gate_hours × margin = reservation_h; 없으면 config projected_map 4.0h)
_tdir = tempfile.mkdtemp(); _rf = os.path.join(_tdir, "reservations.json"); _run_id = G.run_name("RCQ", 2026)
_out = G.write_reservation_file("s5", ["RCQ", "F0"], {}, path=_rf); _e = _out["runs"][_run_id]
tp = object.__new__(KDVTrainer); tp.run_id = _run_id; tp.budget = dict(rcq["budget"], projection_file=_rf); _d = dict(entries={}, throughput=dict(projected_run_hours_50k=2.0))
check("K20 projection_file: gate_hours = reservation_h / 1.1 → trainer margin 1.1 을 곱하면 예약값(1.6627h) · 이 run 과 remaining 둘 다 파일 우선 · 파일에 없는 remaining run 은 종전대로(ledger 평균 → smoke throughput) · 파일이 없으면 이 run 은 projected_map 4.0h",
      abs(_e["gate_hours"] * 1.1 - (1.1 * 1.36 + 10 / 60)) < 1e-9 and abs(tp._projection(_d) - _e["gate_hours"]) < 1e-12 and abs(tp._projection(_d, G.run_name("F0", 2026)) - _out["runs"][G.run_name("F0", 2026)]["gate_hours"]) < 1e-12
      and tp._projection(_d, G.run_name("LF0", 2026)) == 2.0 and (setattr(tp, "budget", dict(rcq["budget"], projection_file=os.path.join(_tdir, "none.json"))) or tp._projection(_d) == 4.0)
      and rcq["budget"]["projection_file"] == G.RESERVATION_FILE and rcq["budget"]["remaining_mandatory_file"] == G.MANDATORY_FILE)
import shutil; shutil.rmtree(_tdir, ignore_errors=True); del tp
check("K16 registry: expect_init 은 16-hex unet/aligner 만", _res(dict(base5, expect_init=dict(unet_sha256_16="0123456789abcdef"))) and not _res(dict(base5, expect_init=dict(unet_sha256_16="short"))) and not _res(dict(base5, expect_init=dict(foo="0123456789abcdef"))))
try:
    KDVTrainer._check_init_hash(dict(unet_init_sha256_16="0123456789abcdef"), dict(unet_sha256_16="0123456789abcdef")); ok16 = True
except ValueError:
    ok16 = False
try:
    KDVTrainer._check_init_hash(dict(unet_init_sha256_16="0123456789abcdef"), dict(unet_sha256_16="ffffffffffffffff")); ok16b = False
except ValueError:
    ok16b = True
_h1234 = G.init_hash_for(1234)
check(f"K16 trainer: 기대 hash 일치면 통과, 불일치면 INIT_HASH_MISMATCH · seed 1234 의 hash({_h1234}) 는 assets/pakd50/init_hashes.json 에서 config 로 (없는 seed 는 검사 생략)", ok16 and ok16b and (_h1234 is None or G.kdv_block("J0", 1234, "s1")["expect_init"]["unet_sha256_16"] == _h1234) and "expect_init" not in G.kdv_block("J0", 3407, "s4"))
r0 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "campaign_gate.py")], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PANCRAFTER_CAMPAIGN_GATES": ""})
check("K05 campaign gate 기본 닫힘", r0.stdout.strip() == "")
src = open(os.path.join(ROOT, "train_kdv.py")).read()
check("K05 trainer: 후보 보존·고정 batch 진단·ledger 잠금·donor expected_step 검사·frozen aligner hash gate(L03) 존재", all(x in src for x in ("retain_all_candidates", "gradient_diagnostics_fixed.jsonl", "_LedgerLock", "_check_donor_step", "frozen aligner(A-FR) state")))

# ================= K23–K28 QEDGE9 (research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md): W104·D121 q-gated GT edge · s5 두 seed + s4 · soft time policy · cue 자산
import filecmp, random as _random
import h5py as _h5, numpy as _np
from kdv import edge_gate as EG
from kdv.edge_gate import EdgeGate
from kdv.registry import describe as _desc
from pa.losses import output_edge_loss_per_sample
from feeders.feeder import PanFeeder
calQ = dict(tau_R=0.012463942170143127, lambda_E=0.09075170336956798)
_kQ = {c: G.kdv_block(c, 2026, "s5", cal=calQ, arch="W104_D121", branch="QEDGE9") for c in ("QE50", "QEC", "QES", "J0", "JQ")}; _sQ = {c: resolve(k) for c, k in _kQ.items()}
check("K23 QEDGE9 case(§3·§4·§6): QE50/QEC/QES = J 정책(A-FT·I-AEQ·offset 1e-4 홀수·A LR 1e-5) + R3 adaptive(α1 β0.1 τR 고정) + EDGE-H(λE0 고정) + edge_gate low_q/const/shuffle · J0/JQ 는 gate 없음 · 전부 W104 expect_arch · 토큰 EDGEHQ50/QC/QS · describe 에 gate 서술",
      all(_sQ[c]["policy"] == "A-FT" and _sQ[c]["protocol"] == "I-AEQ" and _sQ[c]["rec_mode"] == "adaptive" and _sQ[c]["stat_key"] == "EDGE" and _sQ[c]["stat_mode"] == "H" and _kQ[c]["aligner_lr"] == 1e-5
          and abs(_sQ[c]["offset_weight_effective"] - 1e-4) < 1e-12 and _kQ[c]["rec"]["tau"] == calQ["tau_R"] and _kQ[c]["rec"]["alpha"] == 1.0 and _kQ[c]["rec"]["kd_weight"] == 0.1 and _kQ[c]["stat"]["outer_weight"] == calQ["lambda_E"] for c in ("QE50", "QEC", "QES"))
      and [_sQ[c]["edge_gate"]["mode"] for c in ("QE50", "QEC", "QES")] == ["low_q", "const", "shuffle"] and _sQ["J0"]["edge_gate"] is None and _sQ["JQ"]["edge_gate"] is None
      and all(_kQ[c]["expect_arch"] == dict(width=104, depth=[1, 2, 1], noalign=False) for c in _kQ) and _kQ["QE50"]["edge_gate"]["asset"] == G.QEDGE9_CUE_ASSET and _kQ["QEC"]["edge_gate"]["c_E_file"] == G.QEDGE9_CE_FILE
      and _kQ["QES"]["edge_gate"]["perm_seed"] == 51515 and [stat_tag(_sQ[c]) for c in ("QE50", "QEC", "QES", "JQ")] == ["EDGEHQ50", "EDGEHQC", "EDGEHQS", "EDGEH"] and "GT edge gate low_q" in _desc(_sQ["QE50"]) and "hard/soft 는 Q12 그대로" in _desc(_sQ["QES"]))
check("K23 QEDGE9 branch·시간 정책(§0.7·§9.3·§11.1): campaign QEDGE9_A104D121_20260915_v1 · parent PAKD50 · branch A104D121_T0FIX_QEDGE9_v1 · 자체 ledger soft 9h · required True(경고만) · training_deadline 없음 · time_policy soft_target_only/inherit False · projected 1.5h · s4 JQ@W104(E0) 는 마감 상속 그대로",
      all(_kQ[c]["campaign_id"] == "QEDGE9_A104D121_20260915_v1" and _kQ[c]["parent_campaign_id"] == G.CAMPAIGN_ID and _kQ[c]["experiment_branch_id"] == "A104D121_T0FIX_QEDGE9_v1" and _kQ[c]["budget"]["ledger"] == G.QEDGE9_LEDGER
          and _kQ[c]["budget"]["required"] is True and "training_deadline" not in _kQ[c]["budget"] and _kQ[c]["budget"]["time_policy"] == dict(mode="soft_target_only", target_elapsed_hours=9.0, hard_deadline=None, inherit_parent_deadline=False)
          and _kQ[c]["budget"]["total_gpu_hours"] == 9.0 and list(_kQ[c]["budget"]["projected_map"].values()) == [1.5] and _kQ[c]["budget"]["remaining_mandatory_file"] == G.QEDGE9_MANDATORY_FILE for c in _kQ)
      and G.kdv_block("JQ", 1234, "s4", cal=calQ, arch="W104_D121")["budget"].get("training_deadline") == G.training_deadline() and G.kdv_block("JQ", 1234, "s4", cal=calQ, arch="W104_D121")["experiment_branch_id"] == "A104D121_T0FIX_E0_v1")
check("K23 gate case 는 branch 를 안 줘도 QEDGE9 (자동)", G.kdv_block("QE50", 1234, "s4", cal=calQ, arch="W104_D121")["campaign_id"] == "QEDGE9_A104D121_20260915_v1")
for _lab, _fn in (("K23 QEDGE9 는 W104 전용 (W112 거부)", lambda: G.kdv_block("QE50", 1234, "s4", cal=calQ, arch="W112_D123", branch="QEDGE9")), ("K23 gate case 는 W112 기본 골격에서 거부", lambda: G.kdv_block("QE50", 1234, "s4", cal=calQ))):
    try:
        _fn(); check(_lab, False)
    except SystemExit:
        check(_lab, True)
def _rejQ(**kw):
    k = copy.deepcopy(_kQ["QE50"]); k.update(kw)
    try:
        resolve(k); return False
    except ValueError:
        return True
check("K23 registry: edge_gate 는 EDGE-H 위에서만(stat OFF 거부) · routing 결합 거부 · Teacher 없으면 거부 · asset 없으면 거부 · const 의 c_E/c_E_file 둘 다·둘 다 없음 거부 · shuffle seed≠51515 거부 · 알 수 없는 키·mode 거부 · 정상은 통과",
      _rejQ(stat=dict(enabled=False)) and _rejQ(routing=dict(qE=0.0)) and _rejQ(teacher=None, rec=dict(case="N0")) and _rejQ(edge_gate=dict(mode="low_q")) and _rejQ(edge_gate=dict(mode="const", asset=G.QEDGE9_CUE_ASSET))
      and _rejQ(edge_gate=dict(mode="const", asset=G.QEDGE9_CUE_ASSET, c_E=0.5, c_E_file="x.json")) and _rejQ(edge_gate=dict(mode="shuffle", asset=G.QEDGE9_CUE_ASSET, perm_seed=7)) and _rejQ(edge_gate=dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET, foo=1))
      and _rejQ(edge_gate=dict(mode="nope", asset=G.QEDGE9_CUE_ASSET)) and not _rejQ() and not _rejQ(edge_gate=dict(mode="const", asset=G.QEDGE9_CUE_ASSET, c_E=0.61)))
# ---- K24 per-sample edge · trainer 경로 (실제 KDVTrainer._step, CPU): g=1 == JQ · g=0 → edge 0 / hard·soft·offset gradient 유지 · 혼합 · const · 저오차 patch · Teacher·GT detach
g11 = torch.Generator().manual_seed(11); _yh = torch.rand(3, 8, 64, 64, generator=g11); _yy = torch.rand(3, 8, 64, 64, generator=g11)
check("K24 E_i: output_edge_loss_per_sample [B] 의 평균 == output_edge_loss (1e-7)", tuple(output_edge_loss_per_sample(_yh, _yy).shape) == (3,) and abs(float(output_edge_loss_per_sample(_yh, _yy).mean()) - float(output_edge_loss(_yh, _yy))) < 1e-7)
def _stubQ(case, student, eg=None, seed_gen=3234):
    tr = object.__new__(KDVTrainer); tr.k = G.kdv_block(case, 2026, "s5", cal=calQ, arch="W104_D121", branch="QEDGE9"); sp_ = tr.spec = resolve(tr.k)
    tr.model = student; tr.accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, gradient_accumulation_steps=1, scaler=None, is_main_process=True)
    tr.teacher = T0; tr.aligner_trainable = True; tr.aligner_view_margin = 4; tr.share_correction = False; tr.protocol = sp_["protocol"]; tr.radius_hr = sp_["radius_hr"]; tr.diag_every = 10 ** 9
    tr.rec_crit = GTAnchoredReconstructionKD(calQ["tau_R"], alpha=1.0, kd_weight=0.1, eps=1e-6, mode="adaptive"); tr.tri = sp_["tri"]; tr.stat_extra = []; tr.lam_V = calQ["lambda_E"]; tr.stat_ramp = 0
    tr.lam_edge = 0; tr.lam_geo = 0; tr.ramp = 5000; tr.lam_gkd = 0; tr.gen = torch.Generator().manual_seed(seed_gen); tr.corr_seed = seed_gen; tr._ema = {}; tr._rr_val_last = float("nan"); tr.args = types.SimpleNamespace(num_iter=50000)
    tr.freeze_until, tr.freeze_from = None, None; tr.route_A = (1.0, 1.0, 1.0); tr._routed = False; tr._sched_last = None; tr.edge_gate = eg
    return tr
_metaQ = torch.tensor([[0, 0, 1, 1], [1, 2, 1, 1]], dtype=torch.int64); _S = lambda: copy.deepcopy(_S104).train().requires_grad_(True)
_g = lambda loss, m: torch.cat([(x if x is not None else torch.zeros_like(p_)).flatten() for x, p_ in zip(torch.autograd.grad(loss, list(m.parameters()), retain_graph=True, allow_unused=True), list(m.parameters()))])
trJ = _stubQ("JQ", _S()); totJ, infoJ = trJ._step(_gt, _ms, _lp, _pn, 1); LEwJ = infoJ["_edge_w_t"]; gU_J = _g(totJ, trJ.M.backbone); gA_J = _g(totJ, trJ.M.aligner); gU_J0 = _g(totJ - LEwJ, trJ.M.backbone); gA_J0 = _g(totJ - LEwJ, trJ.M.aligner)
tr1 = _stubQ("QE50", _S(), EdgeGate.synthetic("low_q", table=torch.ones(2, 4))); tot1, info1 = tr1._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ)
tr0 = _stubQ("QE50", _S(), EdgeGate.synthetic("low_q", table=torch.zeros(2, 4))); tot0, info0 = tr0._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ)
trm = _stubQ("QES", _S(), EdgeGate.synthetic("shuffle", table=torch.tensor([[1., 1, 1, 1], [0, 0, 0, 0]]))); totm, infom = trm._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ)
trc = _stubQ("QEC", _S(), EdgeGate.synthetic("const", c_E=0.37)); totc, infoc = trc._step(_gt, _ms, _lp, _pn, 1)
_rel = lambda a, b: float((a - b).abs().max()) / (1.0 + float(b.abs().max()))
_Ei = output_edge_loss_per_sample(info1["y"].detach().float(), _gt.float())
check("K24 g=1: total·λE·L_E 가 JQ 와 같다(1e-6) · U/A gradient 동일(1e-6 상대) · g=0: edge 항 정확히 0, total = JQ − λE L_E, U/A gradient == ∇(L_H + L_K + λoff L_off) · 혼합(sample0 만): λE·E_0/2, gate 비율 0.5 · const: 0.37·L_E",
      abs(float(tot1) - float(totJ)) < 1e-6 and abs(float(info1["_edge_w_t"]) - float(LEwJ)) < 1e-7 and _rel(_g(tot1, tr1.M.backbone), gU_J) < 1e-6 and _rel(_g(tot1, tr1.M.aligner), gA_J) < 1e-6
      and float(info0["_edge_w_t"]) == 0.0 and abs(float(tot0) - float(totJ - LEwJ)) < 1e-6 and _rel(_g(tot0, tr0.M.backbone), gU_J0) < 1e-6 and _rel(_g(tot0, tr0.M.aligner), gA_J0) < 1e-6
      and abs(float(infom["loss_stat_raw"]) - float(_Ei[0]) / 2) < 1e-6 and infom["stat_edge_gate_frac"] == 0.5 and abs(float(infoc["loss_stat_raw"]) - 0.37 * float(output_edge_loss(infoc["y"].detach().float(), _gt.float()))) < 1e-7,
      f"Δtot {abs(float(tot1) - float(totJ)):.2e} ΔLE {abs(float(info1['_edge_w_t']) - float(LEwJ)):.2e}")
_rQ = tr1.rec_crit(info1["y_t"], info1["y_t"], _gt.float(), return_maps=True)
check("K24 저오차 patch(§2.2·§8.2): w_H = 1 + α d_T ≥ 1 어디서나(GT hard 삭제 없음) · e_S ≤ e_T 면 a_T = 0 → soft 0 · Teacher/GT detach(y_t 에 grad 없음, T0 parameter requires_grad False) · gate 조회 (index, rot) 표",
      float(_rQ.maps["hard_weight"].min()) >= 1.0 and float(_rQ.soft) == 0.0 and float(tr1.rec_crit(info1["y"].detach(), info1["y_t"], _gt.float(), return_maps=True).maps["hard_weight"].min()) >= 1.0
      and not info1["y_t"].requires_grad and all(not p_.requires_grad for p_ in T0.parameters())
      and torch.equal(EdgeGate.synthetic("low_q", table=torch.tensor([[1., 1, 1, 1], [0, 0, 0, 0]])).gate_for(_metaQ, torch.device("cpu")), torch.tensor([1., 0.])))
try:
    EdgeGate.synthetic("low_q", table=torch.ones(2, 4)).gate_for(None, torch.device("cpu")); check("K24 meta 없이 gate 조회는 ValueError", False)
except ValueError:
    check("K24 meta 없이 gate 조회는 ValueError", True)
# ---- K25 cue 수식·자산: q 부호/단위 · θq 중앙값·동일값 0 · 셔플 stratum 보존 · 실제 자산(있으면) 검증
class _IdealA(torch.nn.Module):
    """top 절반 = a·y ramp, bottom 절반 = b·x ramp 인 PAN 에 대해 warp 를 정확히 읽는 aligner: c = −ε (c0 = 0). bicubic 은 선형 ramp 를 내부에서 정확히 재현한다."""
    def __init__(self, a, b, base):
        super().__init__(); self.a, self.b = a, b; self.w = torch.nn.Parameter(torch.zeros(1)); self.m_top = float(base[..., 4:20, 4:52].mean()); self.m_bot = float(base[..., 36:52, 4:52].mean())
    def forward(self, pan_view, ms_view):
        cy = (pan_view[..., 4:20, 4:52].mean(dim=(1, 2, 3)) - self.m_top) / self.a; cx = (pan_view[..., 36:52, 4:52].mean(dim=(1, 2, 3)) - self.m_bot) / self.b
        return torch.stack((-cy, -cx), 1) + 0 * self.w
class _ConstA(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))
    def forward(self, pan_view, ms_view):
        return torch.zeros(pan_view.shape[0], 2) + 0 * self.w
_yy_, _xx_ = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij"); _P = torch.where(_yy_ < 32, 0.01 * _yy_, 0.02 * _xx_)[None, None].expand(2, 1, 64, 64).contiguous()
_Pv = _P[..., 4:-4, 4:-4]; _A = _IdealA(0.01, 0.02, _Pv); _q_ideal, _c0 = EG.teacher_q(_A, _P, torch.zeros(2, 8, 16, 16), 4); _q_const, _ = EG.teacher_q(_ConstA(), _P, torch.zeros(2, 8, 16, 16), 4)
class _WrongA(_IdealA):
    def forward(self, pan_view, ms_view):
        return -super().forward(pan_view, ms_view)                              # 부호 반대(ĉε = ĉ0 + ε) → 잔차 2ε
_q_wrong, _ = EG.teacher_q(_WrongA(0.01, 0.02, _Pv), _P, torch.zeros(2, 8, 16, 16), 4)
_exact = max(float((predict_c(_A, warp_pan(_P, torch.tensor([e_]).expand(2, 2)), torch.zeros(2, 8, 64, 64), 4)[0] + torch.tensor(e_)).abs().max()) for e_ in ([1.0, 0.0], [0.0, 2.0], [0.5, 0.0], [0.0, -1.0]))
check("K25 q 부호·단위(§2.1·§8.2): 이상 aligner(ĉε = ĉ0 − ε) 에서 정수·반정수 이동은 잔차 < 1e-4(정확), bank 전체 q < 0.01(r=.25 의 bicubic a=−0.75 보간 오차만) ≪ 무반응 aligner q = mean|ε| 성분 0.46875 ≪ 부호 반대 aligner q ≈ 0.9375 · bank = 16 probe, r {.25,.5,1,2} × 축 4 방향, λoff 없음",
      _exact < 1e-4 and float(_q_ideal.abs().max()) < 0.01 and abs(float(_q_const[0]) - 0.46875) < 1e-9 and abs(float(_q_wrong[0]) - 0.9375) < 0.02 and abs(EG.q_const_component() - 0.46875) < 1e-12 and len(EG.probe_bank()) == 16
      and all((p_["ey"] == 0) != (p_["ex"] == 0) for p_ in EG.probe_bank()), f"exact {_exact:.1e} q_ideal {float(_q_ideal.abs().max()):.2e} q_wrong {float(_q_wrong[0]):.4f}")
_qs = _np.array([0.1, 0.3, 0.3, 0.5, 0.2, 0.9]); _th = EG.theta_from(_qs)
_rng = _np.random.RandomState(3); _gate = (_rng.rand(4000) < 0.5).astype(_np.int8); _st = _rng.randint(0, 40, 4000); _st[:50] = 99; _gate[:50] = 1
_g1, _st1 = EG.shuffle_gate(_gate, _st); _g2, _ = EG.shuffle_gate(_gate, _st)
check("K25 θq = 중앙값 · g = 1[q < θq] (동일값 0) · QES 셔플: stratum 별 active 수 보존 · 결정적(seed 51515) · 단일 라벨 stratum 은 그대로(비율 기록) · 라벨이 실제로 바뀐다",
      _th == 0.3 and EG.gate_low_q(_qs, _th).tolist() == [1, 0, 0, 0, 1, 0] and all(_g1[_st == s_].sum() == _gate[_st == s_].sum() for s_ in _np.unique(_st)) and _np.array_equal(_g1, _g2) and _np.array_equal(_g1[:50], _gate[:50])
      and _st1["n_degenerate"] == 1 and _st1["n_shuffled"] == 40 and (_g1 != _gate).mean() > 0.2)
_asset = os.path.join(ROOT, G.QEDGE9_CUE_ASSET)
if os.path.exists(_asset):
    _man, _z = EG.read_asset(G.QEDGE9_CUE_ASSET); _cm = _z["calib_mask"]
    _egL = EdgeGate.load(dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET), teacher=T0, train_h5=os.path.join(ROOT, _man["dataset"]["train"]), feeder_args=dict(hflip=True, vflip=True, rot=True, crop=False), sha_fn=lambda p_: _man["dataset"]["train_sha256"])
    try:
        EdgeGate.load(dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET), teacher=T0, feeder_args=dict(hflip=True, vflip=True, rot=False, crop=False)); _bad_contract = False
    except ValueError:
        _bad_contract = True
    check("K25 실제 cue 자산: θq == median(q[calibration view]) · gate == 1[q<θq] · calibration 3072 base × 4 · 선택 비율 ≈ 0.5 · Teacher A hash == T0 · EdgeGate.load 통과(표 완비) · feeder 계약 다르면 거부 · 셔플 gate 의 stratum 별 active 수 == low_q",
          abs(_man["theta_q"] - float(_np.median(_z["q"][_cm]))) < 1e-12 and _np.array_equal(_z["gate_low_q"], (_z["q"] < _man["theta_q"]).astype(_np.int8)) and int(_cm.sum()) == 3072 * 4 and abs(_man["selection"]["all_frac"] - 0.5) < 0.02
          and _man["teacher"]["aligner_state_hash"] == state_hash(T0.aligner) and _egL.table.shape == (_man["dataset"]["n_train"], 4) and not bool(torch.isnan(_egL.table).any()) and _bad_contract
          and all(_z["gate_shuffle"][(_z["e_decile"] == d_) & (_z["rot"] == r_)].sum() == _z["gate_low_q"][(_z["e_decile"] == d_) & (_z["rot"] == r_)].sum() for d_ in range(10) for r_ in range(4)) and _man["bank"]["id"] == EG.BANK_ID and _man["calibration"]["seed"] == 1234)
else:
    check("K25 실제 cue 자산 (assets/qedge9) — 없음: tools/qedge9_cue.py build (s1) 뒤 git 으로 전달", False)
# ---- K26 편성(§5·§6·§9.2)
_p5 = G.priority_for("s5"); _p4 = G.priority_for("s4"); _r = lambda srv, it: G.reservation_for(srv, it)
check("K26 s5 명시 순서(§5): J0→JQ→QE50 @W104 S2026 → J0→JQ→QE50 @W104 S777(run 이름 항목) · 전부 QEDGE9 branch · 777 은 허용 seed(확인 seed 아님; 9091 은 확인) · 예약 1.6517/1.6627/1.8167 ×2 = 10.2620 h · 옛 순서는 PREVIOUS['s5_20260915'] · cue 없는 QEC 는 cue_ready False",
      _p5 == ["J0@W104_D121", "JQ@W104_D121", "QE50@W104_D121", G.run_name("J0", 777, arch="W104_D121"), G.run_name("JQ", 777, arch="W104_D121"), G.run_name("QE50", 777, arch="W104_D121")]
      and all(G.branch_for("s5", it) == "QEDGE9" for it in _p5) and G.allowed_seeds("s5") == {2026, 777} and abs(sum(_r("s5", it)["reservation_h"] for it in _p5) - 10.262) < 1e-6
      and [_r("s5", it)["reference_kind"] for it in _p5] == ["plan_reference_case"] * 6 and [_r("s5", it)["seed"] for it in _p5] == [2026, 2026, 2026, 777, 777, 777] and _r("s5", G.run_name("J0", 9091, arch="W104_D121"))["reference_kind"] == "confirm_placeholder"
      and G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"] == ["PQ", "F0", "LF0", "LFQ", "RC0", "RCQ"] and G.mandatory_for("s5") == _p5 and G.cue_ready("J0@W104_D121") is True and G.cue_ready("QEC@W104_D121") == os.path.exists(os.path.join(ROOT, G.QEDGE9_CE_FILE)))
_p1 = G.priority_for("s1")
check("K26 s4 앞 5 는 기존 E0 allocation(NA0→J0→JQ→XJ→F0 @W104, 마감 상속; QEDGE9 항목 없음 — QEGX 14 는 K34) · JQ/XJ/F0 1.6847/1.6737/1.4097 = 4.7680 h · 확인 seed 3407 은 이제 QEGX 명시 표(allowed)·옛 --confirm 규칙은 그대로",
      _p4[:5] == ["NA0@W104_D121", "J0@W104_D121", "JQ@W104_D121", "XJ@W104_D121", "F0@W104_D121"] and [G.branch_for("s4", it) for it in _p4[:5]] == [None] * 5 and "s4" not in G.QEDGE9_ITEMS_BY_SERVER
      and abs(sum(_r("s4", it)["reservation_h"] for it in _p4[2:5]) - 4.7680) < 1e-4 and G.allowed_seeds("s4") == {1234, 3407} and G.confirmation_cases("JQ", "s4") == ["J0@W104_D121", "JQ@W104_D121", "J0", "JQ"])
check("K26 s1 명시 순서(17:20 s4→s1): J0→JQ→QE50→QES→QEC @W104 S1234 **v2**(run 이름 항목; E0 v1 config 와 충돌 없음) · 전부 QEDGE9 branch · QEC 마지막(pilot = s1 J0 v2 exact50K) · 예약 1.4647/1.7067/1.8167/1.8167/1.7067 = 8.5113 h · 허용 seed {1234} · v2 config 의 work_dir·baseline 도 v2",
      _p1 == [G.run_name(c, 1234, "v2", arch="W104_D121") for c in ("J0", "JQ", "QE50", "QES", "QEC")] and all(G.branch_for("s1", it) == "QEDGE9" for it in _p1) and G.allowed_seeds("s1") == {1234}
      and abs(sum(_r("s1", it)["reservation_h"] for it in _p1) - 8.5113) < 1e-4 and [_r("s1", it)["reference_kind"] for it in _p1] == ["plan_reference_case"] * 5 and G.QEDGE9_PILOT_BY_SERVER["s1"] == _p1[0] and G.QEDGE9_PILOT_BY_SERVER["s4"] == G.QEDGE9_PILOT_RUN
      and G.kdv_block("QEC", 1234, "s1", cal=calQ, arch="W104_D121", version="v2")["edge_gate"]["pilot_run"] == _p1[0] and G.kdv_block("QEC", 1234, "s1", cal=calQ, arch="W104_D121", version="v2")["baseline_run"] == _p1[0]
      and G.kdv_block("J0", 1234, "s1", cal=calQ, arch="W104_D121", version="v2", branch="QEDGE9")["campaign_id"] == G.QEDGE9_CAMPAIGN_ID and [l.strip() for l in open(os.path.join(ROOT, "config", "queues", "qedge9_s1.txt")) if l.strip() and not l.startswith("#")] == _p1)
check("K26 schedule(gate): blocked(cue 없음) 항목은 그 pass 에서 건너뛰고 뒤 항목은 계속 · exempt(QEDGE9) 는 남은 시간 admission 을 소비하지 않는다 · 옛 동작 불변 · measured_hours_from_ledger 가 s5 의 777 run 실측을 받는다(case@arch 키; 9091 은 제외)",
      G.schedule(True, term(set()), 2.0, lambda c: 1.7, priority=["A", "B", "C"], blocked=lambda c: c == "B") == (["A"], ["C"]) and G.schedule(True, term(set()), 2.0, lambda c: 1.7, priority=["A", "B", "C"], blocked=lambda c: c == "B", exempt=lambda c: c == "C") == (["A", "C"], [])
      and G.schedule(True, term(set()), 2.0, lambda c: 1.7, priority=["A", "B", "C"]) == (["A"], ["B", "C"])
      and G.measured_hours_from_ledger("s5", dict(entries={G.run_name("J0", 777, arch="W104_D121"): dict(kind="run", status="FINISHED", hours=1.2), G.run_name("J0", 9091, arch="W104_D121"): dict(kind="run", status="FINISHED", hours=1.0)})) == {"J0@W104_D121": 1.2})
# ---- K27 feeder return_meta: 같은 RNG 상태에서 텐서 bitwise 동일 · meta = (index, rot, hflip, vflip) · RNG 소비 동일 · apply_view/to_feeder_tensor == feeder 출력
_td = tempfile.mkdtemp(); _h5p = os.path.join(_td, "train_wv3_tiny.h5"); _rs = _np.random.RandomState(0)
with _h5.File(_h5p, "w") as _f:
    _f["gt"] = _rs.randint(0, 2048, (3, 8, 64, 64)).astype(_np.float32); _f["lms"] = _rs.randint(0, 2048, (3, 8, 64, 64)).astype(_np.float32); _f["ms"] = _rs.randint(0, 2048, (3, 8, 16, 16)).astype(_np.float32); _f["pan"] = _rs.randint(0, 2048, (3, 1, 64, 64)).astype(_np.float32)
with _h5.File(_h5p.replace(".h5", "_pan.h5"), "w") as _f:
    _f["lpan"] = _rs.randint(0, 2048, (3, 1, 16, 16)).astype(_np.float32)
_dsM = PanFeeder(_h5p, crop=False, hflip=True, vflip=True, rot=True, return_meta=True); _ds0 = PanFeeder(_h5p, crop=False, hflip=True, vflip=True, rot=True)
_random.seed(77); _oM = _dsM[1]; _stM = _random.getstate(); _random.seed(77); _o0 = _ds0[1]; _st0 = _random.getstate()
_okM = len(_oM) == 6 and len(_o0) == 5 and all(torch.equal(a_, b_) for a_, b_ in zip(_oM[:5], _o0)) and _stM == _st0 and _oM[5].tolist()[0] == 1 and _oM[5].tolist()[2:] == [1, 1] and _oM[5].dtype == torch.int64
_random.seed(77); _rotk = _random.randint(0, 3)                                            # 같은 RNG 열에서 feeder 가 뽑은 rot
with _h5.File(_h5p) as _f:
    _raw = dict(gt=_f["gt"][1:2], ms=_f["ms"][1:2], pan=_f["pan"][1:2])
_v = {k_: EG.apply_view(EG.to_feeder_tensor(a_, 2047.0), True, True, _rotk)[0] for k_, a_ in _raw.items()}
check("K27 feeder return_meta(§7.1–7.2): 5 텐서 bitwise 동일 · meta = [index, rot, 1, 1] int64 · random 상태(RNG 소비) 동일 · 뽑힌 rot 가 meta 와 같다 · kdv.edge_gate.apply_view/to_feeder_tensor == feeder 출력(bitwise; gt/ms/pan)",
      _okM and int(_oM[5][1]) == _rotk and torch.equal(_v["gt"], _oM[0]) and torch.equal(_v["ms"], _oM[2]) and torch.equal(_v["pan"], _oM[4]) and _dsM.split == "train", f"rot {_rotk} meta {_oM[5].tolist() if len(_oM) == 6 else None}")
# ---- K28 생성 config: QEDGE9 9 벌(s5 6 + s4 3) — gated 는 return_meta true · J0/JQ 는 없음 · training_deadline 없음 · campaign QEDGE9 · s4 JQ@W104(E0) config 는 생성기와 그대로 같다(회귀 없음)
_tmpc = tempfile.mkdtemp(); G.generate("s4", ["JQ@W104_D121"], _tmpc, projected=None)
_gen9 = {G.run_name(c, sd, arch="W104_D121"): (c, sd) for c, sd in (("J0", 2026), ("JQ", 2026), ("QE50", 2026), ("J0", 777), ("JQ", 777), ("QE50", 777))}
_gen9.update({G.run_name(c, 1234, "v2", arch="W104_D121"): (c, 1234) for c in ("J0", "JQ", "QE50", "QES", "QEC")})
check("K28 s4 의 QEDGE9 v1 config(QEC/QES S1234 v1) 은 없다 (17:20 s1 이관; QE50 S1234 v1 은 QEGX s4 캠페인으로 다시 존재 — K38) · s1 v2 config 의 work_dir 는 v2", not any(os.path.exists(os.path.join(ROOT, "config", G.run_name(c, 1234, arch="W104_D121") + ".yaml")) for c in ("QEC", "QES"))
      and yaml.safe_load(open(os.path.join(ROOT, "config", G.run_name("QE50", 1234, arch="W104_D121") + ".yaml")))["kdv"]["campaign_id"] == G.QEGX_CAMPAIGN_ID
      and all(yaml.safe_load(open(os.path.join(ROOT, "config", G.run_name(c, 1234, "v2", arch="W104_D121") + ".yaml")))["work_dir"].endswith(G.run_name(c, 1234, "v2", arch="W104_D121")) for c in ("J0", "QEC")))
_cfgs = {r_: (yaml.safe_load(open(os.path.join(ROOT, "config", r_ + ".yaml"))) if os.path.exists(os.path.join(ROOT, "config", r_ + ".yaml")) else None) for r_ in _gen9}
check("K28 QEDGE9 config 11 벌(s5 6 v1 + s1 5 v2) 존재 · gate case 만 train_feeder_args.return_meta true(J0/JQ 없음; QEC 는 const 라 없음) · kdv.campaign_id QEDGE9 · budget 에 training_deadline 없음·time_policy 있음 · edge_gate 모드 일치 · seed/hidden 104 · s4 JQ@W104 기존 config == 생성기(회귀 없음)",
      all(v_ is not None for v_ in _cfgs.values()) and all(bool(v_["train_feeder_args"].get("return_meta", False)) == (_gen9[r_][0] in ("QE50", "QES")) for r_, v_ in _cfgs.items() if v_)
      and all(v_["kdv"]["campaign_id"] == "QEDGE9_A104D121_20260915_v1" and "training_deadline" not in v_["kdv"]["budget"] and v_["kdv"]["budget"]["time_policy"]["inherit_parent_deadline"] is False and v_["seed"] == _gen9[r_][1] and v_["model_args"]["hidden_size"] == 104 for r_, v_ in _cfgs.items() if v_)
      and all((v_["kdv"].get("edge_gate") or {}).get("mode") == {"QE50": "low_q", "QEC": "const", "QES": "shuffle"}.get(_gen9[r_][0]) for r_, v_ in _cfgs.items() if v_)
      and filecmp.cmp(os.path.join(ROOT, "config", G.run_name("JQ", 1234, arch="W104_D121") + ".yaml"), os.path.join(_tmpc, G.run_name("JQ", 1234, arch="W104_D121") + ".yaml"), shallow=False),
      f"missing {[r_ for r_, v_ in _cfgs.items() if v_ is None]}")

# ================= K29–K33 QEDGE9 감사 대응 (research_log/PAN_QEDGE9_Implementation_Audit_2026-09-15.md F02–F07)
from kdv.resume import EpochState, begin_epoch
# ---- K29 exact resume (F04): 같은 tiny feeder·shuffle DataLoader(num_workers 2, rot 무작위) 에서 연속 2 epoch 의 (index, rot) 열 == 중간 checkpoint 재개 열
_dsR = PanFeeder(_h5p, crop=False, hflip=True, vflip=True, rot=True, return_meta=True)
def _mk_loader():
    return torch.utils.data.DataLoader(_dsR, batch_size=1, shuffle=True, num_workers=2, drop_last=True)
def _run(loader, es, start, n_steps, pending):
    out = []; g = start
    while len(out) < n_steps:
        it, skip, _ = begin_epoch(loader, es, g, pending); pending = False
        for b in it:
            out.append((int(b[5][0, 0]), int(b[5][0, 1]))); g += 1
            if len(out) >= n_steps:
                break
    return out
torch.manual_seed(2026); _random.seed(2026); esA = EpochState(); seqA = _run(_mk_loader(), esA, 0, 6, False)                    # 연속: 3 batch/epoch × 2 epoch
torch.manual_seed(2026); _random.seed(2026); esB = EpochState(); loaderB = _mk_loader(); seqB1 = []; gB = 0
it, skip, _ = begin_epoch(loaderB, esB, 0, False)
for b in it:                                                                                                                  # epoch 0 의 2 batch 까지 진행하고 '중단'
    seqB1.append((int(b[5][0, 0]), int(b[5][0, 1]))); gB += 1
    if gB == 2:
        break
_sd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in esB.state_dict().items()}; _ckpt_rng = torch.get_rng_state()                                              # 'checkpoint': EpochState + 전역 RNG(accelerate 가 저장/복원하는 것)
torch.manual_seed(999); _random.seed(999)                                                                                      # 프로세스 재시작 흉내: RNG 오염
esB2 = EpochState(); esB2.load_state_dict(_sd); torch.set_rng_state(_ckpt_rng); seqB2 = _run(_mk_loader(), esB2, gB, 4, True)   # accelerate.load_state 가 전역 RNG 를 checkpoint 시점으로 복원한 상태
torch.manual_seed(2026); _random.seed(2026); esC = EpochState(); loaderC = _mk_loader(); begin_epoch(loaderC, esC, 0, False)
try:
    begin_epoch(loaderC, esC, 7, True); _bad_skip = False
except RuntimeError:
    _bad_skip = True
check("K29 exact resume(F04): 연속 실행의 (index, rot) 열 == 2 batch 뒤 checkpoint(EpochState) 재개 열 (worker rot 포함) · 오염된 RNG 에서도 동일 · skip 범위 밖(step 7 > 3 batch/epoch) 은 RuntimeError · 기록 exact/skipped",
      seqB1 + seqB2 == seqA and len(seqA) == 6 and _bad_skip and esA.epoch_start_step == 3 and esB2.epoch_start_step == 3, f"A {seqA} · B {seqB1 + seqB2}")
# ---- K30 cue 자산 식별·일관성 (F03): asset_id 고정 · 내부 모순(θq/gate 반전/bank/QES seed)·margin·재개 asset_id 불일치 거부 · 정상은 통과
if os.path.exists(_asset):
    import copy as _copy
    _man0, _z0 = EG.read_asset(G.QEDGE9_CUE_ASSET); _zd = {k: _np.array(_z0[k]) for k in _z0.files}
    def _viol(mut_man=None, mut_z=None):
        m_ = _copy.deepcopy(_man0); z_ = {k: v.copy() for k, v in _zd.items()}
        if mut_man: mut_man(m_)
        if mut_z: mut_z(z_)
        return EG.check_asset(m_, z_)
    def _setm(k, v):
        def f(m_): m_[k] = v
        return f
    _v_ok = _viol(); _v_theta = _viol(_setm("theta_q", 1e9)); _v_bank = _viol(mut_man=lambda m_: m_["bank"].update(id="OTHER_BANK", K=8)); _v_qes = _viol(mut_man=lambda m_: m_["qes"].update(seed=7))
    _v_inv = _viol(mut_z=lambda z_: z_.__setitem__("gate_low_q", (1 - z_["gate_low_q"]).astype(_np.int8))); _v_id = _viol(_setm("asset_id", "deadbeef"))
    _v_sh = _viol(mut_z=lambda z_: z_.__setitem__("gate_shuffle", z_["gate_low_q"].copy()))
    _tm = types.SimpleNamespace(aligner=T0.aligner, aligner_margin=0)
    try:
        EdgeGate.load(dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET), teacher=_tm); _m_rej = False
    except ValueError:
        _m_rej = True
    try:
        EdgeGate.load(dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET), teacher=T0, previous=dict(asset_id="0" * 32)); _p_rej = False
    except ValueError:
        _p_rej = True
    _egP = EdgeGate.load(dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET), teacher=T0, previous=dict(asset_id=_man0["asset_id"]))
    check("K30 cue 식별·일관성(F03): 정상 자산 위반 0 · θq 1e9 / bank 변경 / QES seed 7 / gate 반전 / shuffle 위조 / asset_id 위조 는 각각 위반 · Teacher margin 0 거부 · 재개 시 이전 asset_id 불일치 거부·일치 통과 · 요약에 asset_id",
          not _v_ok and _v_theta and _v_bank and _v_qes and _v_inv and _v_sh and _v_id and _m_rej and _p_rej and _egP.summary()["asset_id"] == _man0["asset_id"] == EG.asset_id(_man0) and len(_man0["asset_id"]) == 32,
          f"ok {_v_ok} theta {len(_v_theta)} bank {len(_v_bank)} qes {len(_v_qes)} inv {len(_v_inv)} sh {len(_v_sh)} id {len(_v_id)}")
    # ---- K31 QEC pilot identity (F02): c_E 파일의 pilot identity·cue asset_id 대조 · registry 가 const c_E_file 에 pilot 키를 요구 · 생성 config 에 pilot 키
    _tdc = tempfile.mkdtemp(); _cef = os.path.join(_tdc, "qec_cE.json"); _cfgC = dict(mode="const", asset=G.QEDGE9_CUE_ASSET, c_E_file=_cef, pilot_run=G.QEDGE9_PILOT_RUN, pilot_tag="last", pilot_step=50000)
    _good = dict(c_E=0.61, cue_npz_sha256=_man0["npz_sha256"], cue_asset_id=_man0["asset_id"], pilot_run=G.QEDGE9_PILOT_RUN, pilot_tag="last", pilot_step=50000, pilot_sha256_16="0123456789abcdef", pilot_file_sha256="f" * 64, views="calibration")
    def _try(d):
        json.dump(d, open(_cef, "w"))
        try:
            return EdgeGate.load(_cfgC, teacher=T0).c_E
        except (ValueError, KeyError):
            return None
    _r_good = _try(_good); _r_run = _try(dict(_good, pilot_run="PAKD50_QE50_W104_D121_WV3_T0_S777_FRESH50_v1")); _r_tag = _try(dict(_good, pilot_tag="best_hqnr")); _r_step = _try(dict(_good, pilot_step=1010))
    _r_id = _try(dict(_good, cue_asset_id="deadbeef")); _r_noid = _try({k: v for k, v in _good.items() if k not in ("pilot_file_sha256",)}); _r_all = _try(dict(_good, views="all"))
    check("K31 QEC pilot identity(F02): 올바른 c_E 파일은 통과(0.61) · pilot run/tag/step 불일치 · cue asset_id 불일치 · checkpoint hash 없음 · calibration view 아님 은 거부 · registry: const c_E_file 에 pilot 키 없으면 거부 · 생성 config(QEC) 에 pilot_run/tag/step",
          _r_good == 0.61 and _r_run is None and _r_tag is None and _r_step is None and _r_id is None and _r_noid is None and _r_all is None
          and _rejQ(edge_gate=dict(mode="const", asset=G.QEDGE9_CUE_ASSET, c_E_file="x.json")) and not _rejQ(edge_gate=dict(mode="const", asset=G.QEDGE9_CUE_ASSET, c_E_file="x.json", pilot_run=G.QEDGE9_PILOT_RUN, pilot_tag="last", pilot_step=50000))
          and G.kdv_block("QEC", 1234, "s4", cal=calQ, arch="W104_D121")["edge_gate"]["pilot_run"] == "PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v1" and G.kdv_block("QEC", 1234, "s4", cal=calQ, arch="W104_D121")["exact_resume"] is True)
else:
    check("K30/K31 cue 자산 없음 — build 뒤", False)
# ---- K32 실측 통합 (F07): QEDGE9 ledger 의 실측이 예약에 반영 · PAKD50 ledger 와 합침 (모듈 상수 임시 치환)
_tdl = tempfile.mkdtemp(); _lp, _lq = os.path.join(_tdl, "p.json"), os.path.join(_tdl, "q.json")
json.dump(dict(entries={G.run_name("JQ", 1234, arch="W104_D121"): dict(kind="run", status="FINISHED", hours=1.38)}), open(_lp, "w"))
json.dump(dict(entries={G.run_name("QE50", 1234, arch="W104_D121"): dict(kind="run", status="FINISHED_TRAIN", hours_total=2.75), G.run_name("QE50", 9091, arch="W104_D121"): dict(kind="run", status="FINISHED", hours=1.0)}), open(_lq, "w"))
_L0, _Q0 = G.LEDGER, G.QEDGE9_LEDGER; G.LEDGER, G.QEDGE9_LEDGER = _lp, _lq
try:
    _mall = G.measured_hours_all("s4"); _res4 = G.reservation_for("s4", "QE50@W104_D121", _mall)
finally:
    G.LEDGER, G.QEDGE9_LEDGER = _L0, _Q0
check("K32 실측 통합(F07): PAKD50 ledger 의 JQ@W104 1.38 + QEDGE9 ledger 의 QE50@W104 2.75(확인 seed 9091 은 제외) → 예약이 실측(1.10×2.75+10/60) 으로 바뀐다(measured_same_case)",
      _mall == {"JQ@W104_D121": 1.38, "QE50@W104_D121": 2.75} and _res4["reference_kind"] == "measured_same_case" and abs(_res4["reservation_h"] - (1.1 * 2.75 + 10 / 60)) < 1e-9)
# ---- K33 완료 검증 (F06): marker 만 있는 가짜 run 은 불통과(사유 명시) · 실제 완료 PAKD50 run(s1 FQ S1234) 은 통과
_tdr = tempfile.mkdtemp(); _fake = "PAKD50_QEX_W104_D121_WV3_T0_S1234_FRESH50_v1"; _fd = os.path.join(ROOT, "work_dir", _fake)
try:
    os.makedirs(os.path.join(_fd, "results"), exist_ok=True); open(os.path.join(_fd, "results", "reduced_best_hqnr.mat"), "w").close(); open(os.path.join(_fd, "results", "full_best_hqnr.mat"), "w").close()
    json.dump(dict(step=1010), open(os.path.join(_fd, "last_meta.json"), "w")); _vf = G.verified_complete(_fake)
finally:
    import shutil as _sh; _sh.rmtree(_fd, ignore_errors=True)
_real = G.run_name("FQ", 1234); _vr = G.verified_complete(_real) if os.path.exists(os.path.join(ROOT, "work_dir", _real, "last_meta.json")) else None
check("K33 완료 검증(F06): marker 만 있는 run → ok False (last_exact_step/candidate_grid/kdv_manifest 불통과) · 실제 완료 run(s1 PAKD50 FQ S1234: exact50K last·50 후보·Teacher T0·train sha == cue·init 파일 hash) → ok True",
      _vf["ok"] is False and _vf["checks"]["results_mats"] and _vf["checks"]["last_exact_step"] is False and _vf["checks"]["candidate_grid"] is False and (_vr is None or _vr["ok"]), f"fake {_vf['checks']} · real {(_vr or {}).get('checks')}")

# ================= K34–K38 QEGX s3·s4 (research_log/PAN_QEGX_S3_S4_W104D121_Experiment_Plan_2026-09-15.md §4–§6·§9·§11–§12)
_kX = {c: G.kdv_block(c, 2026, "s3", cal=calQ, arch="W104_D121", version="v2") for c in ("QX50", "QEC3", "QE50_B005")}
_kX.update({c: G.kdv_block(c, 1234, "s4", cal=calQ, arch="W104_D121") for c in ("LFQE50", "QER50", "QERS")}); _sX = {c: resolve(k) for c, k in _kX.items()}
_kXJQ = G.kdv_block("JQ", 1234, "s4", cal=calQ, arch="W104_D121", branch="QEGX")
check("K34 QEGX case(§4): QX50 = R1(hard-only, β0) + low_q gate · QE50_B005 = R3 β0.05 + low_q · QEC3 = const(c_E_file work_dir/_qegx/qec3_cE.json, pilot s3 J0 S2026 v2/last/50000) · LFQE50 = freeze_from 25000 + low_q · QER50/QERS = edge_route low_q/shuffle(51515), edge_gate·routing 없음 · 전부 W104·J 정책·λE0·τR 고정",
      _kX["QX50"]["rec"]["case"] == "R1" and _kX["QX50"]["rec"]["kd_weight"] == 0.0 and _kX["QX50"]["edge_gate"]["mode"] == "low_q" and _kX["QE50_B005"]["rec"] == dict(_kQ["QE50"]["rec"], kd_weight=0.05) and _kX["QE50_B005"]["edge_gate"] == _kQ["QE50"]["edge_gate"]
      and _kX["QEC3"]["edge_gate"] == dict(mode="const", asset=G.QEDGE9_CUE_ASSET, c_E_file=G.QEGX_CE_FILE, pilot_run="PAKD50_J0_W104_D121_WV3_T0_S2026_FRESH50_v2", pilot_tag="last", pilot_step=50000) and G.QEGX_CE_FILE != G.QEDGE9_CE_FILE
      and _kX["LFQE50"]["aligner_schedule"] == dict(freeze_from=25000) and _kX["LFQE50"]["edge_gate"]["mode"] == "low_q" and _kX["LFQE50"]["rec"] == _kQ["QE50"]["rec"]
      and _kX["QER50"]["edge_route"] == dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET, theta_source="asset") and _kX["QERS"]["edge_route"] == dict(mode="shuffle", asset=G.QEDGE9_CUE_ASSET, perm_seed=51515)
      and all("edge_gate" not in _kX[c] and "routing" not in _kX[c] for c in ("QER50", "QERS")) and all(_kX[c]["rec"] == _kXJQ["rec"] and _kX[c]["stat"] == _kXJQ["stat"] for c in ("QER50", "QERS", "LFQE50"))
      and all(_kX[c]["aligner_policy"] == "A-FT" and _kX[c]["aligner_lr"] == 1e-5 and _kX[c]["expect_arch"] == dict(width=104, depth=[1, 2, 1], noalign=False) and _kX[c]["stat"]["outer_weight"] == calQ["lambda_E"] for c in _kX))
check("K34 QEGX branch·시간 정책(§9.3·§12): campaign QEGX_A104D121_S3S4_20260915_v1 · parent PAKD50 · lineage [PAKD50, QEDGE9] · branch A104D121_T0FIX_QEGX_v1 · 자체 ledger · total 1000h(도달 불가) · required True(경고만) · training_deadline 없음 · time_policy no_hard_limit · exact_resume · mandatory 파일 _qegx",
      all(k["campaign_id"] == G.QEGX_CAMPAIGN_ID and k["parent_campaign_id"] == G.CAMPAIGN_ID and k["lineage_campaign_ids"] == [G.CAMPAIGN_ID, G.QEDGE9_CAMPAIGN_ID] and k["experiment_branch_id"] == "A104D121_T0FIX_QEGX_v1" and k["exact_resume"] is True
          and k["budget"]["ledger"] == G.QEGX_LEDGER and k["budget"]["total_gpu_hours"] == 1000.0 and k["budget"]["required"] is True and "training_deadline" not in k["budget"] and k["budget"]["remaining_mandatory_file"] == G.QEGX_MANDATORY_FILE
          and k["budget"]["time_policy"] == dict(mode="no_hard_limit", target_elapsed_hours=None, hard_deadline=None, inherit_parent_deadline=False) for k in list(_kX.values()) + [_kXJQ])
      and _kXJQ["budget"]["ledger"] != G.QEDGE9_LEDGER and G.kdv_block("JQ", 1234, "s4", cal=calQ, arch="W104_D121")["budget"]["ledger"] == G.LEDGER)
for _lab, _fn in (("K34 QEC3 는 s3 만 (s4 에는 pilot 이 없다 → SystemExit)", lambda: G.kdv_block("QEC3", 1234, "s4", cal=calQ, arch="W104_D121")), ("K34 QEGX 전용 case 를 QEDGE9 branch 로 만들면 SystemExit", lambda: G.kdv_block("QX50", 2026, "s3", cal=calQ, arch="W104_D121", branch="QEDGE9")),
                  ("K34 QEGX 는 W104 전용 (W112 거부)", lambda: G.kdv_block("QER50", 1234, "s4", cal=calQ, branch="QEGX")), ("K34 edge_route case 를 PAKD50 branch(None) 로 만들면 SystemExit", lambda: G.kdv_block("JQ", 1234, "s4", cal=calQ, arch="W104_D121", branch=None) and G.kdv_block("QER50", 1234, "s4", cal=calQ, arch="W104_D121", branch="E0"))):
    try:
        _fn(); check(_lab, False)
    except SystemExit:
        check(_lab, True)
_p3 = G.priority_for("s3"); _p4x = [it for it in G.priority_for("s4") if G.is_tag(it)]
_want3 = [G.run_name(c, 2026, "v2", arch="W104_D121") for c in ("J0", "JQ", "QE50", "XJ", "QX50", "J_R3_NOEDGE", "QEC3", "QES", "J_QB005", "QE50_B005")] + [G.run_name(c, 4321, "v2", arch="W104_D121") for c in ("J0", "JQ", "XJ", "QE50", "QX50")]
_want4 = [G.run_name(c, 1234, arch="W104_D121") for c in ("QE50", "LF0", "LFQ", "LFQE50", "LFX", "JE0", "QER50", "QERS")] + [G.run_name(c, 3407, arch="W104_D121") for c in ("J0", "JQ", "QE50", "JE0", "QER50", "QERS")]
check("K34 편성(§5·§6): s3 = 15 run **v2**(2026 A/B 10 → 4321 C 5) · s4 = E0 5 뒤 14 run v1(1234 A 8 → 3407 C 6) · 전부 QEGX branch · allowed seed s3 {2026,4321} / s4 {1234,3407} · 옛 s3 순서는 PREVIOUS · bare 'QE50@W104_D121' 은 QEDGE9 자동 규칙(큐에는 run 이름만) · 큐 파일 == 순서 · refresh v3 는 QEGX",
      _p3 == _want3 and _p4x == _want4 and all(G.branch_for("s3", it) == "QEGX" for it in _p3) and all(G.branch_for("s4", it) == "QEGX" for it in _p4x) and G.allowed_seeds("s3") == {2026, 4321} and G.allowed_seeds("s4") == {1234, 3407}
      and G.branch_for("s3", "QE50@W104_D121") == "QEDGE9" and G.branch_for("s3", "J0@W104_D121") is None and G.branch_for("s4", G.QEGX_REFRESH_CONTROLS["s4"][0]) == "QEGX" and G.QEGX_REFRESH_CONTROLS["s4"][0].endswith("_v3")
      and [l.strip() for l in open(os.path.join(ROOT, "config", "queues", "qegx_s3.txt")) if l.strip() and not l.startswith("#")] == _p3 and [l.strip() for l in open(os.path.join(ROOT, "config", "queues", "qegx_s4.txt")) if l.strip() and not l.startswith("#")] == _p4x
      and G.mandatory_for("s3") == _p3 and G.server_id("s3(5090)") == "s3" and G.server_id(" s4\n") == "s4" and {"QX50", "QEC3", "QE50_B005", "LFQE50", "QER50", "QERS"} <= G.NEEDS_LAMBDA_E)
_r3 = [G.reservation_for("s3", it) for it in _p3]; _r4 = [G.reservation_for("s4", it) for it in _p4x]
check(f"K34 예약(§9.2): s3 ref 1.16/1.34/1.50/1.33/1.50/1.30/1.34/1.50/1.34/1.50 + C 5 → 합 25.2039 h (실측 {sum(r['reservation_h'] for r in _r3):.4f}) · s4 14 run 합 26.2803 h ({sum(r['reservation_h'] for r in _r4):.4f}) · 전부 plan_reference_case(확인 seed 도 명시 표; confirm_placeholder 아님) · routing 1.80(*) · gate 1.50(*)",
      abs(sum(r["reservation_h"] for r in _r3) - 25.2039) < 2e-4 and abs(sum(r["reservation_h"] for r in _r4) - 26.2803) < 2e-4 and all(r["reference_kind"] == "plan_reference_case" for r in _r3 + _r4)
      and [r["reference_train_h"] for r in _r3[:10]] == [1.16, 1.34, 1.50, 1.33, 1.50, 1.30, 1.34, 1.50, 1.34, 1.50] and [r["reference_train_h"] for r in _r4[:8]] == [1.50, 1.17, 1.38, 1.50, 1.37, 1.80, 1.80, 1.80] and [r["seed"] for r in _r4[8:]] == [3407] * 6)
check("K34 cue_ready: QEC3 는 c_E3 파일(_qegx) 로만 · QEC 는 c_E(_qedge9) 로만 · QER50/QERS/QX50/LFQE50 은 cue 자산만 · gate exempt(QEDGE9·QEGX) 는 admission 을 소비하지 않고 blocked 는 건너뛴다 · measured_hours_all 이 QEGX ledger 도 본다",
      G.cue_ready(_p3[6]) == os.path.exists(os.path.join(ROOT, G.QEGX_CE_FILE)) and G.cue_ready("QEC@W104_D121") == os.path.exists(os.path.join(ROOT, G.QEDGE9_CE_FILE)) and G.cue_ready(_p4x[6]) == G.cue_ready("QE50@W104_D121") == G.cue_ready(_p4x[3])
      and G.schedule(True, term(set()), 0.5, lambda c: 1.7, priority=_p3[:3] + ["J0"], exempt=lambda it: G.branch_for("s3", it) in ("QEDGE9", "QEGX")) == (_p3[:3], ["J0"])
      and G.schedule(True, term(set()), None, lambda c: 1.7, priority=_p3[:8], exempt=lambda it: G.branch_for("s3", it) in ("QEDGE9", "QEGX"), blocked=lambda it: G.case_of(it) == "QEC3") == ([x for x in _p3[:8] if G.case_of(x) != "QEC3"], [])
      and "QEGX_LEDGER" in open(os.path.join(ROOT, "tools", "gen_pakd50_configs.py")).read().split("def measured_hours_all")[1].split("def ")[0])
# ---- K35 registry edge_route
def _rejX(base, **kw):
    k = copy.deepcopy(base); k.update(kw)
    try:
        resolve(k); return False
    except ValueError:
        return True
_kR = _kX["QER50"]; _kF = G.kdv_block("FQ", 1234, "s4", cal=calQ, arch="W104_D121")
check("K35 registry edge_route(§4.5): QER50/QERS 통과(spec mode·edge_U 1·edge_A g/g_shuffle) · 토큰 EDGEHR50/EDGEHRS · QX50 = R1 + EDGEHQ50 · describe 에 edge_route 서술 · 거부: +edge_gate · +routing qE · stat OFF · A-FR · asset 없음 · shuffle seed≠51515 · 알 수 없는 키 · mode const · Teacher 없음",
      _sX["QER50"]["edge_route"]["mode"] == "low_q" and _sX["QER50"]["edge_route"]["edge_U"] == 1.0 and _sX["QER50"]["edge_route"]["edge_A"] == "g" and _sX["QERS"]["edge_route"]["edge_A"] == "g_shuffle" and _sX["QERS"]["edge_route"]["perm_seed"] == 51515
      and _sX["QER50"]["edge_gate"] is None and tuple(_sX["QER50"]["route_A"]) == (1.0, 1.0, 1.0) and [stat_tag(_sX[c]) for c in ("QER50", "QERS", "QX50", "QE50_B005", "LFQE50")] == ["EDGEHR50", "EDGEHRS", "EDGEHQ50", "EDGEHQ50", "EDGEHQ50"] and _sX["QX50"]["rec_case"] == "R1"
      and "edge_route low_q" in _desc(_sX["QER50"]) and "U 는 모든 patch" in _desc(_sX["QERS"]) and _sX["LFQE50"]["aligner_freeze_from"] == 25000 and _sX["LFQE50"]["edge_gate"]["mode"] == "low_q"
      and _rejX(_kR, edge_gate=dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET)) and _rejX(_kR, routing=dict(qE=0.0)) and _rejX(_kR, stat=dict(enabled=False)) and _rejX(_kF, edge_route=dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET))
      and _rejX(_kR, edge_route=dict(mode="low_q")) and _rejX(_kR, edge_route=dict(mode="shuffle", asset=G.QEDGE9_CUE_ASSET, perm_seed=7)) and _rejX(_kR, edge_route=dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET, foo=1))
      and _rejX(_kR, edge_route=dict(mode="const", asset=G.QEDGE9_CUE_ASSET)) and _rejX(_kR, teacher=None, rec=dict(case="N0")) and not _rejX(_kR))
# ---- K36 trainer edge_route (실제 KDVTrainer._step + backward + _apply_edge_route, CPU): g=1 ≡ JQ · g=0 ≡ JE0(routing qE=0) · 혼합 g 는 A = ∇φ(L_H+L_K+λE·E(g)+λoff L_O) 의 autograd · U 는 all-edge 그대로 · QERS 표 · RNG 미소비 · step 한 번
def _stubX(case, student, gate=None, server="s4", seed=1234, seed_gen=3234):
    tr = object.__new__(KDVTrainer); tr.k = G.kdv_block(case, seed, server, cal=calQ, arch="W104_D121", branch="QEGX"); sp_ = tr.spec = resolve(tr.k)
    tr.model = student; tr.accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, gradient_accumulation_steps=1, scaler=None, is_main_process=True)
    tr.teacher = T0; tr.aligner_trainable = True; tr.aligner_view_margin = 4; tr.share_correction = False; tr.protocol = sp_["protocol"]; tr.radius_hr = sp_["radius_hr"]; tr.diag_every = 10 ** 9
    tr.rec_crit = (GTAnchoredReconstructionKD(calQ["tau_R"], alpha=float(tr.k["rec"].get("alpha", 1.0)), kd_weight=float(tr.k["rec"].get("kd_weight", 0.0)), eps=1e-6, mode=sp_["rec_mode"]) if sp_["rec_case"] != "N0" else None)
    tr.tri = sp_["tri"]; tr.stat_extra = []; tr.lam_V = (calQ["lambda_E"] if sp_["stat_enabled"] else 0.0); tr.stat_ramp = 0
    tr.lam_edge = 0; tr.lam_geo = 0; tr.ramp = 5000; tr.lam_gkd = 0; tr.gen = torch.Generator().manual_seed(seed_gen); tr.corr_seed = seed_gen; tr._ema = {}; tr._rr_val_last = float("nan"); tr.args = types.SimpleNamespace(num_iter=50000)
    tr.freeze_until, tr.freeze_from = sp_["aligner_freeze_until"], sp_["aligner_freeze_from"]; tr.route_A = tuple(sp_["route_A"]); tr._routed = any(q != 1.0 for q in tr.route_A); tr._sched_last = None
    tr.edge_gate = (gate if sp_.get("edge_gate") else None); tr.edge_route = (gate if sp_.get("edge_route") else None); tr._edge_routed = bool(sp_.get("edge_route"))
    return tr
def _apply_after_backward(tr, total, info):
    total.backward(retain_graph=True); n = 0
    if tr._routed:
        n += tr._apply_routing(info, tr.M)
    if tr._edge_routed:
        n += tr._apply_edge_route(info, tr.M)
    return n
def _gradsX(tr):
    return torch.cat([(p_.grad if p_.grad is not None else torch.zeros_like(p_)).flatten() for p_ in tr.M.aligner.parameters()]), torch.cat([(p_.grad if p_.grad is not None else torch.zeros_like(p_)).flatten() for p_ in tr.M.backbone.parameters()])
trJx = _stubX("JQ", _S()); totJx, infoJx = trJx._step(_gt, _ms, _lp, _pn, 1); gA_JQ = _g(totJx, trJx.M.aligner); gU_JQ = _g(totJx, trJx.M.backbone); _apply_after_backward(trJx, totJx, infoJx); gA_JQb, gU_JQb = _gradsX(trJx)
trE0 = _stubX("JE0", _S()); totE0, infoE0 = trE0._step(_gt, _ms, _lp, _pn, 1); _nE0 = _apply_after_backward(trE0, totE0, infoE0); gA_JE0, gU_JE0 = _gradsX(trE0)
tr1x = _stubX("QER50", _S(), EdgeGate.synthetic("low_q", table=torch.ones(2, 4))); tot1x, info1x = tr1x._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ); n1 = _apply_after_backward(tr1x, tot1x, info1x); gA_1, gU_1 = _gradsX(tr1x)
tr0x = _stubX("QER50", _S(), EdgeGate.synthetic("low_q", table=torch.zeros(2, 4))); tot0x, info0x = tr0x._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ); n0 = _apply_after_backward(tr0x, tot0x, info0x); gA_0, gU_0 = _gradsX(tr0x)
check(f"K36 QER50 g=1 ≡ JQ(§11.3-5): total bitwise 같음 · backward+보정 뒤 A/U .grad == JQ (보정량 0; 1e-6 상대) · g=0 ≡ JE0(routing qE=0): A .grad 일치(1e-5 상대), U .grad == JQ 의 U(all-edge 유지) · total 은 두 경우 다 JQ 와 같다 (U 가 받는 edge 는 항상 E(1))",
      float(tot1x) == float(totJx) and float(tot0x) == float(totJx) and n1 > 0 and n0 > 0 and _rel(gA_1, gA_JQb) < 1e-6 and _rel(gU_1, gU_JQb) < 1e-6 and _rel(gA_0, gA_JE0) < 1e-5 and _rel(gU_0, gU_JQb) < 1e-6 and _rel(gU_0, gU_JE0) < 1e-6
      and float((gA_0 - gA_JQb).abs().max()) > 0 and infoJx.get("_edge_route_hi_w_t") is None and info1x["stat_edge_gate_frac"] == 1.0 and info0x["stat_edge_gate_frac"] == 0.0,
      f"A1 {_rel(gA_1, gA_JQb):.1e} U1 {_rel(gU_1, gU_JQb):.1e} A0 {_rel(gA_0, gA_JE0):.1e} U0 {_rel(gU_0, gU_JQb):.1e} |A0−AJQ| {float((gA_0 - gA_JQb).abs().max()):.2e}")
trm_x = _stubX("QER50", _S(), EdgeGate.synthetic("low_q", table=torch.tensor([[1., 1, 1, 1], [0, 0, 0, 0]]))); totm_x, infom_x = trm_x._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ)
_gi = torch.tensor([1.0, 0.0]); _Eix = output_edge_loss_per_sample(infom_x["y"].float(), _gt.float()); apx = list(trm_x.M.aligner.parameters()); bpx = list(trm_x.M.backbone.parameters())
manual_A = infom_x["_rec_hard_t"] + infom_x["_rec_soft_t"] + float(infom_x["lam_v"]) * (_gi * _Eix).mean() + float(infom_x["lam_off"]) * infom_x["loss_off"]
want_A = torch.cat([w.flatten() for w in _grads(manual_A, apx)]); want_U = torch.cat([w.flatten() for w in _grads(totm_x, bpx)]); ref_A = float(want_A.abs().max())
nm = _apply_after_backward(trm_x, totm_x, infom_x); gA_m, gU_m = _gradsX(trm_x)
check(f"K36 혼합 g=(1,0)(§11.3-6): A .grad == ∇φ[L_H + L_K + λE·mean(g·E_i) + λoff L_O] (autograd 직접; max|Δ| {float((gA_m - want_A).abs().max()):.1e}/ref {ref_A:.1e}) · U .grad == ∇θ total(all-edge) · gate 비율 0.5 · A 가 받는 edge λE·E(g) = LEw − hi",
      float((gA_m - want_A).abs().max()) <= 1e-5 * (1.0 + ref_A) and _rel(gU_m, want_U) < 1e-6 and nm > 0 and infom_x["stat_edge_gate_frac"] == 0.5
      and abs(float(infom_x["_edge_w_t"] - infom_x["_edge_route_hi_w_t"]) - float(infom_x["lam_v"]) * float((_gi * _Eix).mean())) < 1e-6 and abs(float(infom_x["stat_edge_A_gated"]) - float((_gi * _Eix).mean())) < 1e-6)
trs_x = _stubX("QERS", _S(), EdgeGate.synthetic("shuffle", table=torch.tensor([[0., 0, 0, 0], [1, 1, 1, 1]]))); tots_x, infos_x = trs_x._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ)
check("K36 QERS(§11.3-7): shuffle 표 조회(sample1 만 active → 비율 0.5, A gated edge = E_1/2) · total == JQ · gate 조회가 ε RNG 를 소비하지 않는다(gen 상태 == JQ) · meta 없이 edge_route 는 ValueError · 학습 loop: backward(retain_graph) 뒤 보정 한 번, optimizer.step 한 번",
      float(tots_x) == float(totJx) and infos_x["stat_edge_gate_frac"] == 0.5 and abs(float(infos_x["stat_edge_A_gated"]) - float(output_edge_loss_per_sample(infos_x["y"].detach().float(), _gt.float())[1]) / 2) < 1e-6
      and torch.equal(trs_x.gen.get_state(), trJx.gen.get_state()) and trs_x._edge_routed and not trs_x._routed
      and all(x in src for x in ("routed_now = (self._routed or self._edge_routed) and self.aligner_active(global_step)", "if routed_now and self._edge_routed:", "_apply_edge_route(info, M)")) and src.count("self.optimizer.step()") == 1)
try:
    _stubX("QER50", _S(), EdgeGate.synthetic("low_q", table=torch.ones(2, 4)))._step(_gt, _ms, _lp, _pn, 1); check("K36 edge_route 에 meta 없음 → ValueError", False)
except ValueError:
    check("K36 edge_route 에 meta 없음 → ValueError", True)
del trJx, trE0, tr1x, tr0x, trm_x, trs_x, totJx, totE0, tot1x, tot0x, totm_x, tots_x
# ---- K37 LFQE50 · β=0 · g=0 동치 (§11.3-2/3/8)
trQ = _stubX("QE50", _S(), EdgeGate.synthetic("low_q", table=torch.ones(2, 4))); totQ, infoQ = trQ._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ)
trL = _stubX("LFQE50", _S(), EdgeGate.synthetic("low_q", table=torch.ones(2, 4))); totL, infoL = trL._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ)
trL2 = _stubX("LFQE50", _S(), EdgeGate.synthetic("low_q", table=torch.ones(2, 4))); trL2.M.aligner.requires_grad_(False); totL2, infoL2 = trL2._step(_gt, _ms, _lp, _pn, 25001, meta=_metaQ); hA2 = state_hash(trL2.M.aligner)
optL = torch.optim.AdamW([dict(params=list(trL2.M.backbone.parameters())), dict(params=list(trL2.M.aligner.parameters()), lr=1e-5)], lr=1e-4, weight_decay=0.01); totL2.backward(); optL.step()
check("K37 LFQE50(§4.6·§11.3-8): 25K 전 update 1 은 QE50 과 total·ε 열 동일 · aligner_active 24999 True / 25000 False · 동결 update 25001(odd): offset 연습 생략(L_O 0)·Δ graph 없음·U 는 gated edge 를 계속 받는다(λE L_E>0)·AdamW step 뒤 A hash 불변(grad None)",
      float(totL) == float(totQ) and torch.equal(trL.gen.get_state(), trQ.gen.get_state()) and trL.aligner_active(24999) and not trL.aligner_active(25000)
      and infoL2.get("off_skipped_frozen") == 1.0 and float(infoL2["loss_off"]) == 0.0 and not infoL2["delta"].requires_grad and float(infoL2["_edge_w_t"]) > 0 and infoL2["stat_edge_gate_frac"] == 1.0
      and state_hash(trL2.M.aligner) == hA2 and all(p_.grad is None for p_ in trL2.M.aligner.parameters()) and any(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for p_ in trL2.M.backbone.parameters()))
trX = _stubX("QX50", _S(), EdgeGate.synthetic("low_q", table=torch.ones(2, 4)), server="s3", seed=2026); totX, infoX = trX._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ)
trXJ = _stubX("XJ", _S(), server="s3", seed=2026); totXJ, infoXJ = trXJ._step(_gt, _ms, _lp, _pn, 1); trJQ3 = _stubX("JQ", _S(), server="s3", seed=2026); totJQ3, infoJQ3 = trJQ3._step(_gt, _ms, _lp, _pn, 1)
trB = _stubX("QE50_B005", _S(), EdgeGate.synthetic("low_q", table=torch.ones(2, 4)), server="s3", seed=2026); totB, infoB = trB._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ)
trN = _stubX("J_R3_NOEDGE", _S(), server="s3", seed=2026); totN, infoN = trN._step(_gt, _ms, _lp, _pn, 1); trQ0 = _stubX("QE50", _S(), EdgeGate.synthetic("low_q", table=torch.zeros(2, 4)), server="s3", seed=2026); totQ0, infoQ0 = trQ0._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ)
check("K37 β=0(§11.3-3): QX50 total == QE50(g=1) total − L_K (1e-6) · XJ == JQ − L_K · QX50 의 soft 항 정확히 0 · QE50_B005 의 L_K == 0.5 × QE50 의 L_K (β 선형) · g=0(§11.3-2): QE50(g=0) total == J_R3_NOEDGE total (edge 0; hard/soft/offset 경로 유지)",
      abs(float(totX) - (float(totQ) - float(infoQ["_rec_soft_t"]))) < 1e-6 and abs(float(totXJ) - (float(totJQ3) - float(infoJQ3["_rec_soft_t"]))) < 1e-6 and infoX.get("_rec_soft_t") is not None and float(infoX["_rec_soft_t"]) == 0.0
      and abs(float(infoB["_rec_soft_t"]) - 0.5 * float(infoQ["_rec_soft_t"])) < 1e-6 * (1 + float(infoQ["_rec_soft_t"])) and float(infoQ["_rec_soft_t"]) > 0
      and abs(float(totQ0) - float(totN)) < 1e-6 and float(infoQ0["_edge_w_t"]) == 0.0 and float(infoQ0["loss_off"]) > 0 and float(infoQ0["_rec_soft_t"]) > 0,
      f"ΔX {abs(float(totX) - (float(totQ) - float(infoQ['_rec_soft_t']))):.1e} ΔN {abs(float(totQ0) - float(totN)):.1e}")
del trQ, trL, trL2, trX, trXJ, trJQ3, trB, trN, trQ0
# ---- K38 c_E3 pilot 분리(§4.2) · 시트 토큰(§12) · 스크립트/큐 · 생성 config
from tools import qedge9_cue as _cue
_pbX, _pb9 = _cue.pilot_branch(types.SimpleNamespace(branch="qegx")), _cue.pilot_branch(types.SimpleNamespace(branch="qedge9"))
_okce = None
if os.path.exists(_asset):
    _manX, _ = EG.read_asset(G.QEDGE9_CUE_ASSET); _tdx = tempfile.mkdtemp(); _cef3 = os.path.join(_tdx, "qec3_cE.json")
    _cfg3 = dict(mode="const", asset=G.QEDGE9_CUE_ASSET, c_E_file=_cef3, pilot_run=G.QEGX_PILOT_BY_SERVER["s3"], pilot_tag="last", pilot_step=50000)
    _good3 = dict(c_E=0.58, cue_npz_sha256=_manX["npz_sha256"], cue_asset_id=_manX["asset_id"], pilot_run=G.QEGX_PILOT_BY_SERVER["s3"], pilot_tag="last", pilot_step=50000, pilot_sha256_16="0123456789abcdef", pilot_file_sha256="f" * 64, views="calibration", branch="qegx")
    def _try3(d):
        json.dump(d, open(_cef3, "w"))
        try:
            return EdgeGate.load(_cfg3, teacher=T0).c_E
        except (ValueError, KeyError):
            return None
    _okce = (_try3(_good3) == 0.58 and _try3(dict(_good3, pilot_run=G.QEDGE9_PILOT_BY_SERVER["s1"])) is None and _try3(dict(_good3, pilot_run=G.QEDGE9_PILOT_RUN)) is None)
check("K38 QEC3 pilot(§4.2): --branch qegx → s3 J0 S2026 v2 / seed 2026 / work_dir/_qegx/qec3_cE.json · 기본 qedge9 → QEDGE9 파일·seed 1234 · QEC3 config 는 s1/s4 QEDGE9 pilot 의 c_E 파일을 거부(pilot identity) · 올바른 c_E3 파일 통과",
      _pbX["out"] == G.QEGX_CE_FILE and _pbX["runs"] == G.QEGX_PILOT_RUNS == {"PAKD50_J0_W104_D121_WV3_T0_S2026_FRESH50_v2"} and _pbX["seed"] == 2026 and _pb9["out"] == G.QEDGE9_CE_FILE and _pb9["seed"] == 1234 and _pb9["runs"] == G.QEDGE9_PILOT_RUNS and (_okce is None or _okce), f"c_E3 검사 {_okce}")
try:
    import importlib.util as _ilu; _spec_g = _ilu.spec_from_file_location("gspread_upload_mod", os.path.join(ROOT, "gspread", "gspread_upload.py")); _gmod = _ilu.module_from_spec(_spec_g); _spec_g.loader.exec_module(_gmod); _il = _gmod.integrated_label
    _lab_ok = (_il("PAKD50_QX50_W104_D121_WV3_T0_S2026_FRESH50_v2", "A104D121_T0FIX_QEGX_v1") == "PAKD50 / QX50 / A104D121 / QEGX / FRESH50" and _il("PAKD50_QE50_W104_D121_WV3_T0_S1234_FRESH50_v2", "A104D121_T0FIX_QEDGE9_v1") == "PAKD50 / QE50 / A104D121 / QEDGE9 / FRESH50"
               and _il("PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v1", "A104D121_T0FIX_E0_v1") == "PAKD50 / JQ / A104D121 / FRESH50" and _il("PAKD50_JQ_W112_D123_WV3_T0_S777_FRESH50_v1", None) == "PAKD50 / JQ / FRESH50")
except Exception as _e:                                                        # noqa
    _lab_ok = False; print("   gspread import:", repr(_e)[:120])
check("K38 시트 X열(§12): QEGX branch → 'PAKD50 / QX50 / A104D121 / QEGX / FRESH50' · QEDGE9/E0/W112 토큰은 그대로 · gate 설명에 QEGX · switch/waiter 스크립트 bash -n 통과", _lab_ok
      and "QEGX" in __import__("tools.campaign_gate", fromlist=["GATES"]).GATES["pakd50"][1] and all(subprocess.run(["bash", "-n", os.path.join(ROOT, "tools", f)], capture_output=True).returncode == 0 for f in ("qegx_switch.sh", "qegx_waiter.sh")))
_cfgX = {r_: (yaml.safe_load(open(os.path.join(ROOT, "config", r_ + ".yaml"))) if os.path.exists(os.path.join(ROOT, "config", r_ + ".yaml")) else None) for r_ in _p3 + _p4x}
_tmpx = tempfile.mkdtemp(); G.generate("s3", [_p3[6], _p3[4]], _tmpx, projected=None); G.generate("s4", [_p4x[6], _p4x[3]], _tmpx, projected=None)
check("K38 생성 config 29 벌(s3 15 v2 + s4 14 v1) 존재 · gate/route run 만 return_meta true · kdv.campaign_id QEGX · training_deadline 없음 · time_policy no_hard_limit · seed/골격 일치 · edge_gate/edge_route 모드 일치 · 생성기와 filecmp 동일(QEC3/QX50/QER50/LFQE50) · s4 E0 JQ v1 config 회귀 없음",
      all(v_ is not None for v_ in _cfgX.values()) and all(bool(v_["train_feeder_args"].get("return_meta", False)) == (G.case_of(r_) in ("QE50", "QES", "QX50", "QE50_B005", "LFQE50", "QER50", "QERS")) for r_, v_ in _cfgX.items() if v_)
      and all(v_["kdv"]["campaign_id"] == G.QEGX_CAMPAIGN_ID and "training_deadline" not in v_["kdv"]["budget"] and v_["kdv"]["budget"]["time_policy"]["mode"] == "no_hard_limit" and v_["seed"] == G.seed_of(r_) and v_["model_args"]["hidden_size"] == 104 and v_["model_args"]["depth"] == [1, 2, 1] for r_, v_ in _cfgX.items() if v_)
      and all((v_["kdv"].get("edge_gate") or {}).get("mode") == {"QE50": "low_q", "QES": "shuffle", "QX50": "low_q", "QE50_B005": "low_q", "LFQE50": "low_q", "QEC3": "const"}.get(G.case_of(r_)) and (v_["kdv"].get("edge_route") or {}).get("mode") == {"QER50": "low_q", "QERS": "shuffle"}.get(G.case_of(r_)) for r_, v_ in _cfgX.items() if v_)
      and all(filecmp.cmp(os.path.join(ROOT, "config", r_ + ".yaml"), os.path.join(_tmpx, r_ + ".yaml"), shallow=False) for r_ in (_p3[6], _p3[4], _p4x[6], _p4x[3]))
      and filecmp.cmp(os.path.join(ROOT, "config", G.run_name("JQ", 1234, arch="W104_D121") + ".yaml"), os.path.join(_tmpc, G.run_name("JQ", 1234, arch="W104_D121") + ".yaml"), shallow=False), f"missing {[r_ for r_, v_ in _cfgX.items() if v_ is None]}")
print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
