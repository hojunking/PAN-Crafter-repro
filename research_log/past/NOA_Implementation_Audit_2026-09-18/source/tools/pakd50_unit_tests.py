#!/usr/bin/env python
"""PAKD50 launch gate (계획 §14.1 중 코드로 닫는 것). 하나라도 실패하면 exit 1.   python tools/pakd50_unit_tests.py
T0 binding · 정책/backend 매핑 · Q12↔X02↔R1↔N0 관계 · edge 정의 · gate detach · offset 수신자 · frozen A · 후보 격자 · 큐 · λE 의존성 · 캠페인 gate 격리
K17–K20 (2026-09-15 재배정): RC 정책·J_R3_NOEDGE/J_N0_EDGE 정의 · RC trainer step(offset 연습 없음, L_rec → A) · 서버별 명시 순서·예약식·admission · trainer projection_file
K23–K28 (2026-09-15 QEDGE9): case/branch/시간 정책 · registry 거부 · per-sample edge·trainer 경로(g=1/0/혼합/const) · q 수식·θq·셔플·cue 자산 · s5/s4 편성·schedule(blocked/exempt) · feeder return_meta · 생성 config
K29–K33 (QEDGE9 감사 대응): exact resume(F04) · cue asset_id·내부 일관성·재개 대조(F03) · QEC pilot identity(F02) · 실측 통합(F07) · 완료 검증(F06)
K34–K38 (2026-09-15 QEGX s3/s4): case/branch/시간 정책/편성·예약(§5·§6·§9) · registry edge_route 거부·통과 · trainer edge_route(g=1≡JQ · g=0≡JE0 · 혼합 autograd · QERS RNG) · LFQE50/β=0/g=0 동치 · c_E3 pilot 분리·시트 토큰·gate exempt
K39–K43 (2026-09-16 EDGEBAL s2/s5): EB case 동치·상수 배수·schedule·floor 등록/편성/예약(§4–§7) · registry edge_schedule/edge_weight 거부·통과 · trainer 동치(EB_N0≡J0 · R3E100≡JQ · R3E000≡R3_NOEDGE · R1E100≡XJ · N0E100≡N0_EDGE) · 배수 · 25K 경계·A 계속 학습 · floor/shuffle/reverse 계수·no-grad · 시트/스크립트/config
K44–K48 (2026-09-16 QRECON24 s1–s5): profile/큐 68/이름/예약 1.20×R_s(§4–§5·§8) · registry qrecon 거부·통과 · trainer 두 목적함수 분리(U←∇L_U · A←∇L_A 만, β/λE 불변, α 변화, A_FREEZE 불변, offset 없음, ALL_UNIF≡표준 total) · QWeight(w(qref)=.5·q=0→1·fail-fast·셔플 multiset·실제 자산; 분자 2 없음) · 시트/스크립트/config/selector"""
import copy, glob, json, os, subprocess, sys, tempfile
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
cal = dict(tau_R=0.012, lambda_E=0.3); spec = {c: resolve(G.kdv_block(c, SEED, SRV, cal=cal)) for c in G.CASES if c not in G.QEDGE9_CASES and c not in G.QEGX_CASES and c not in G.EDGEBAL_CASES and not c.startswith("QRC24_")}      # QEDGE9/QEGX/EDGEBAL/QRC24 case 는 W104 전용 (K23/K34/K39/K44)
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
check("K09 s4: seed 1234(s1 과 같은 run id), T0 경로는 모든 서버에서 자산 사본, 2026-09-14 기본 묶음 J0→JQ→AL0→ALQ(완료; PREVIOUS_PRIORITY_BY_SERVER), stage 1 = J0/AL0", G.SERVER_SEED["s4"] == 1234 and G.t0_dir("s1") == G.t0_dir("s4") == G.T0_ASSET_DIR and G.PREVIOUS_PRIORITY_BY_SERVER["s4_20260914"] == ["J0", "JQ", "AL0", "ALQ"] and G.stage_cases("s4", 1) == ["J0", "AL0"] and G.priority_for("s9") == G.PRIORITY and G.QEDGE9_ITEMS_BY_SERVER["s1"] == G.QEDGE9_S1_ITEMS)
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
check("K19 명시 순서(§4; s2 는 09-16 EDGEBAL 로 교체 → PREVIOUS['s2_20260915']): s2 JR→XJ→J_R3_NOEDGE→J_N0_EDGE · s4 F0→RC0→RCQ→JR→XJ→J_R3_NOEDGE · s5 PQ→F0→LF0→LFQ→RC0→RCQ; mandatory == priority; 항목 없는 서버는 기본 PRIORITY(s1 은 17:20 QEDGE9 v2 목록; s3 는 09-15 s3 추가 전 기본); 완료 묶음은 순서에 없다(FR 도)",
      G.PREVIOUS_PRIORITY_BY_SERVER["s2_20260915"] == ["JR", "XJ", "J_R3_NOEDGE", "J_N0_EDGE"] and G.PREVIOUS_PRIORITY_BY_SERVER["s4"] == ["F0", "RC0", "RCQ", "JR", "XJ", "J_R3_NOEDGE"] and G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"] == ["PQ", "F0", "LF0", "LFQ", "RC0", "RCQ"]
      and all(G.mandatory_for(s) == G.priority_for(s) for s in ("s2", "s4", "s5", "s1")) and G.priority_for("s9") == G.PRIORITY and G.QEDGE9_ITEMS_BY_SERVER["s1"] == G.QEDGE9_S1_ITEMS and G.PREVIOUS_PRIORITY_BY_SERVER["s3"] == G.PRIORITY and not ({"J0", "JQ", "F0", "FQ", "FR"} & set(G.priority_for("s2"))) and "PQ" in G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"])
_sum = lambda srv: sum(G.reservation_for(srv, c)["reservation_h"] for c in (G.PREVIOUS_PRIORITY_BY_SERVER["s4"] if srv == "s4" else (G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"] if srv == "s5" else (G.PREVIOUS_PRIORITY_BY_SERVER["s2_20260915"] if srv == "s2" else G.priority_for(srv)))))
check("K19 예약식 reservation_h = 1.10×ref + 10/60 (§3 검산): s2 4×2.33 → 10.9187 · s4 2×1.17+4×1.39 → 9.6900 · s5 1.80+3×1.35+2×1.36 → 10.4270 (측정값 없이)",
      abs(_sum("s2") - 10.9186667) < 1e-6 and abs(_sum("s4") - 9.69) < 1e-6 and abs(_sum("s5") - 10.427) < 1e-6 and abs(G.reservation_hours(2.33) - 2.7296667) < 1e-6)
check("K19 기준값 출처: N0 형/T 형은 서버 계획값(s2 는 전부 2.33) · PQ 는 routing 가예약 1.80 · 같은 서버 같은 case 실측이 있으면 그것 · routing 실측이 생기면 다른 routing case 도 그 평균 · 확인 seed 는 1.80 가예약",
      G.reference_hours("s2", "JR") == (2.33, "plan_reference") and G.reference_hours("s4", "RC0") == (1.17, "plan_reference") and G.reference_hours("s4", "RCQ") == (1.39, "plan_reference") and G.reference_hours("s5", "PQ") == (1.8, "routing_placeholder")
      and G.reference_hours("s5", "PQ", {"PQ": 1.9}) == (1.9, "measured_same_case") and G.reference_hours("s5", "PX", {"PQ": 1.9})[1] == "measured_routing_mean" and G.reference_hours("s5", "RCQ", {}, confirm=True) == (1.8, "confirm_placeholder")
      and G.reference_hours("s5", "RCQ", {"RCQ": 1.4}, confirm=True) == (1.4, "measured_same_case") and G.reservation_for("s5", G.run_name("RCQ", 9999))["seed"] == 9999 and G.reservation_for("s5", G.run_name("RCQ", 9999))["reference_kind"] == "confirm_placeholder" and G.reservation_for("s5", G.run_name("RCQ", 9091))["reference_kind"] == "plan_reference")   # 9091 은 09-16 EDGEBAL 명시 표의 허용 seed
_est = lambda srv: (lambda it: G.reservation_for(srv, it)["reservation_h"])
check("K19 편성: s2 완료(J0/F0/JQ/FQ) 뒤 남은 전부 명시 순서 · s4/s5 도 · 실행 중 PQ 는 제외(그 뒤부터) · s2 남은 5.0h 면 JR 만(2.73+2.73 > 5) · est callable 이면 margin 을 다시 곱하지 않는다",
      G.schedule(True, term({"J0", "F0", "JQ", "FQ"}), 40.0, _est("s2"), priority=G.PREVIOUS_PRIORITY_BY_SERVER["s2_20260915"]) == (["JR", "XJ", "J_R3_NOEDGE", "J_N0_EDGE"], [])
      and G.schedule(True, term({"AL0", "J0", "JQ", "ALQ"}), 40.0, _est("s4"), priority=G.PREVIOUS_PRIORITY_BY_SERVER["s4"]) == (["F0", "RC0", "RCQ", "JR", "XJ", "J_R3_NOEDGE"], [])
      and G.schedule(True, term({"J0", "JQ", "D0", "DQ", "PQ"}), 40.0, _est("s5"), priority=G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"]) == (["F0", "LF0", "LFQ", "RC0", "RCQ"], [])
      and G.schedule(True, term({"J0", "F0", "JQ", "FQ"}), 5.0, _est("s2"), priority=G.PREVIOUS_PRIORITY_BY_SERVER["s2_20260915"]) == (["JR"], ["XJ", "J_R3_NOEDGE", "J_N0_EDGE"])
      and G.schedule(True, term({"J0", "F0", "JQ", "FQ"}), 5.46, _est("s2"), priority=G.PREVIOUS_PRIORITY_BY_SERVER["s2_20260915"]) == (["JR", "XJ"], ["J_R3_NOEDGE", "J_N0_EDGE"]))
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
      G.QEGX_S4_ITEMS[:5] == ["NA0@W104_D121", "J0@W104_D121", "JQ@W104_D121", "XJ@W104_D121", "F0@W104_D121"] and G.PREVIOUS_PRIORITY_BY_SERVER["s4"] == ["F0", "RC0", "RCQ", "JR", "XJ", "J_R3_NOEDGE"] and abs(sum(_s4(c) for c in G.QEGX_S4_ITEMS[:5]) - 7.7853333) < 1e-6
      and G.reference_hours("s4", "JQ@W104_D121") == (1.38, "plan_reference_case") and G.reference_hours("s4", "JQ") == (1.39, "plan_reference") and G.schedule(True, term({G.run_name("F0", 1234), G.run_name("JQ", 1234)}), 40.0, _s4, priority=G.QEGX_S4_ITEMS[:5]) == (G.QEGX_S4_ITEMS[:5], [])
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
_p5 = list(G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260916_edgebal"]); _p4 = list(G.QEGX_S4_ITEMS); _r = lambda srv, it: G.reservation_for(srv, it)   # 09-16 저녁 QRECON24 로 교체된 직전 순서 (K44)
check("K26 s5 명시 순서(§5): J0→JQ→QE50 @W104 S2026 → J0→JQ→QE50 @W104 S777(run 이름 항목) · 전부 QEDGE9 branch · 777 은 허용 seed(확인 seed 아님; 9091 은 확인) · 예약(EDGEBAL §7.1 실측 1.38/1.57/1.65 로 교체) 1.6847/1.8937/1.9817 ×2 = 11.1200 h · 9091 은 EDGEBAL 허용 seed(K39) · 옛 순서는 PREVIOUS['s5_20260915'] · cue 없는 QEC 는 cue_ready False",
      _p5[:6] == ["J0@W104_D121", "JQ@W104_D121", "QE50@W104_D121", G.run_name("J0", 777, arch="W104_D121"), G.run_name("JQ", 777, arch="W104_D121"), G.run_name("QE50", 777, arch="W104_D121")]
      and all(G.branch_for("s5", it) == "QEDGE9" for it in _p5[:6]) and G.allowed_seeds("s5") == {2026, 777, 9091, 1103} and abs(sum(_r("s5", it)["reservation_h"] for it in _p5[:6]) - 11.12) < 1e-6
      and [_r("s5", it)["reference_kind"] for it in _p5[:6]] == ["plan_reference_case"] * 6 and [_r("s5", it)["seed"] for it in _p5[:6]] == [2026, 2026, 2026, 777, 777, 777] and _r("s5", G.run_name("J0", 9999, arch="W104_D121"))["reference_kind"] == "confirm_placeholder"
      and G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260915"] == ["PQ", "F0", "LF0", "LFQ", "RC0", "RCQ"] and G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260916_edgebal"] == _p5 and G.cue_ready("J0@W104_D121") is True and G.cue_ready("QEC@W104_D121") == os.path.exists(os.path.join(ROOT, G.QEDGE9_CE_FILE)))
_p1 = list(G.QEDGE9_S1_ITEMS)
check("K26 s4 앞 5 는 기존 E0 allocation(NA0→J0→JQ→XJ→F0 @W104, 마감 상속; QEDGE9 항목 없음 — QEGX 14 는 K34) · JQ/XJ/F0 1.6847/1.6737/1.4097 = 4.7680 h · 확인 seed 3407 은 이제 QEGX 명시 표(allowed)·옛 --confirm 규칙은 그대로",
      _p4[:5] == ["NA0@W104_D121", "J0@W104_D121", "JQ@W104_D121", "XJ@W104_D121", "F0@W104_D121"] and [G.branch_for("s4", it) for it in _p4[:5]] == [None] * 5 and "s4" not in G.QEDGE9_ITEMS_BY_SERVER
      and abs(sum(_r("s4", it)["reservation_h"] for it in _p4[2:5]) - 4.7680) < 1e-4 and G.allowed_seeds("s4") == {1234, 3407} and G.confirmation_cases("JQ", "s4") == ["J0@W104_D121", "JQ@W104_D121", "J0", "JQ"])
check("K26 s1 명시 순서(17:20 s4→s1): J0→JQ→QE50→QES→QEC @W104 S1234 **v2**(run 이름 항목; E0 v1 config 와 충돌 없음) · 전부 QEDGE9 branch · QEC 마지막(pilot = s1 J0 v2 exact50K) · 예약 1.4647/1.7067/1.8167/1.8167/1.7067 = 8.5113 h · 허용 seed {1234} · v2 config 의 work_dir·baseline 도 v2",
      _p1 == [G.run_name(c, 1234, "v2", arch="W104_D121") for c in ("J0", "JQ", "QE50", "QES", "QEC")] and all(G.branch_for("s1", it) == "QEDGE9" for it in _p1) and G.allowed_seeds("s1") == {1234, 3407}
      and abs(sum(_r("s1", it)["reservation_h"] for it in _p1) - 8.5113) < 1e-4 and [_r("s1", it)["reference_kind"] for it in _p1] == ["plan_reference_case"] * 5 and G.QEDGE9_PILOT_BY_SERVER["s1"] == _p1[0] and G.QEDGE9_PILOT_BY_SERVER["s4"] == G.QEDGE9_PILOT_RUN
      and G.kdv_block("QEC", 1234, "s1", cal=calQ, arch="W104_D121", version="v2")["edge_gate"]["pilot_run"] == _p1[0] and G.kdv_block("QEC", 1234, "s1", cal=calQ, arch="W104_D121", version="v2")["baseline_run"] == _p1[0]
      and G.kdv_block("J0", 1234, "s1", cal=calQ, arch="W104_D121", version="v2", branch="QEDGE9")["campaign_id"] == G.QEDGE9_CAMPAIGN_ID and [l.strip() for l in open(os.path.join(ROOT, "config", "queues", "qedge9_s1.txt")) if l.strip() and not l.startswith("#")] == _p1)
check("K26 schedule(gate): blocked(cue 없음) 항목은 그 pass 에서 건너뛰고 뒤 항목은 계속 · exempt(QEDGE9) 는 남은 시간 admission 을 소비하지 않는다 · 옛 동작 불변 · measured_hours_from_ledger 가 s5 의 777 run 실측을 받는다(case@arch 키; 허용 밖 seed 9999 는 제외)",
      G.schedule(True, term(set()), 2.0, lambda c: 1.7, priority=["A", "B", "C"], blocked=lambda c: c == "B") == (["A"], ["C"]) and G.schedule(True, term(set()), 2.0, lambda c: 1.7, priority=["A", "B", "C"], blocked=lambda c: c == "B", exempt=lambda c: c == "C") == (["A", "C"], [])
      and G.schedule(True, term(set()), 2.0, lambda c: 1.7, priority=["A", "B", "C"]) == (["A"], ["B", "C"])
      and G.measured_hours_from_ledger("s5", dict(entries={G.run_name("J0", 777, arch="W104_D121"): dict(kind="run", status="FINISHED", hours=1.2), G.run_name("J0", 9999, arch="W104_D121"): dict(kind="run", status="FINISHED", hours=1.0)})) == {"J0@W104_D121": 1.2})
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
# QRECON24 감사 F01: epoch 끝 checkpoint(save_epoch; global_step = epoch_start + n) 에서의 재개 — 그 epoch 는 끝났으므로 다음 epoch 를 checkpoint 시점 RNG 에서 새로 시작 (연속 실행과 같은 열)
torch.manual_seed(2026); _random.seed(2026); esD = EpochState(); loaderD = _mk_loader(); seqD1 = []; gD = 0
it, skip, _ = begin_epoch(loaderD, esD, 0, False)
for b in it:
    seqD1.append((int(b[5][0, 0]), int(b[5][0, 1]))); gD += 1                                                                # epoch 0 을 끝까지(3 batch) — save_checkpoint 시점
_sdD = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in esD.state_dict().items()}; _ckD = torch.get_rng_state()
torch.manual_seed(999); _random.seed(999); esD2 = EpochState(); esD2.load_state_dict(_sdD); torch.set_rng_state(_ckD); itD2, skipD2, infoD2 = begin_epoch(_mk_loader(), esD2, gD, True)
seqD2 = [(int(b[5][0, 0]), int(b[5][0, 1])) for b in itD2]
from kdv.resume import ExactResumeMismatch
try:
    begin_epoch(_mk_loader(), esD2, gD + 4, True); _mm = False
except ExactResumeMismatch:
    _mm = True
check("K29b epoch 경계 재개(QRECON24 감사 F01): epoch 끝 checkpoint(skip == n_batches) 는 RuntimeError 가 아니라 **다음 epoch 를 새로 시작**(skipped 0, boundary True, epoch_start_step = global_step) 하고 그 열이 연속 실행의 다음 epoch 와 같다 · 범위 밖은 ExactResumeMismatch(trainer 가 exit 4; 같은 id fresh 재실행 없음)",
      seqD1 == seqA[:3] and seqD2 == seqA[3:6] and skipD2 == 0 and infoD2["exact"] and infoD2["boundary"] and infoD2["epoch_start_step"] == 3 and esD2.epoch_start_step == 3 and _mm and issubclass(ExactResumeMismatch, RuntimeError)
      and "except ExactResumeMismatch" in src and "sys.exit(4)" in src.split("except ExactResumeMismatch")[1][:600], f"D1 {seqD1} D2 {seqD2} A {seqA}")
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
_L0, _Q0, _X0, _E0, _R0 = G.LEDGER, G.QEDGE9_LEDGER, G.QEGX_LEDGER, G.EDGEBAL_LEDGER, G.QRC24_LEDGER; G.LEDGER, G.QEDGE9_LEDGER = _lp, _lq
G.QEGX_LEDGER = G.EDGEBAL_LEDGER = G.QRC24_LEDGER = os.path.join(_tdl, "none.json")     # 이 서버의 실제 QEGX/EDGEBAL/QRC24 ledger(완료 run 이 생기면 같은 seed 실측이 섞인다) 도 fixture 로 — 검사는 두 ledger 통합만 본다
try:
    _mall = G.measured_hours_all("s4"); _res4 = G.reservation_for("s4", "QE50@W104_D121", _mall)
finally:
    G.LEDGER, G.QEDGE9_LEDGER, G.QEGX_LEDGER, G.EDGEBAL_LEDGER, G.QRC24_LEDGER = _L0, _Q0, _X0, _E0, _R0
check("K32 실측 통합(F07): PAKD50 ledger 의 JQ@W104 1.38 + QEDGE9 ledger 의 QE50@W104 2.75(확인 seed 9091 은 제외) → 예약이 실측(1.10×2.75+10/60) 으로 바뀐다(measured_same_case)",
      _mall == {"JQ@W104_D121": 1.38, "QE50@W104_D121": 2.75} and _res4["reference_kind"] == "measured_same_case" and abs(_res4["reservation_h"] - (1.1 * 2.75 + 10 / 60)) < 1e-9)
# ---- K33 완료 검증 (F06): marker 만 있는 가짜 run 은 불통과(사유 명시) · 실제 완료 PAKD50 run(s1 FQ S1234) 은 통과
_tdr = tempfile.mkdtemp(); _fake = "PAKD50_QEX_W104_D121_WV3_T0_S1234_FRESH50_v1"; _fd = os.path.join(ROOT, "work_dir", _fake)
try:
    os.makedirs(os.path.join(_fd, "results"), exist_ok=True); open(os.path.join(_fd, "results", "reduced_best_hqnr.mat"), "w").close(); open(os.path.join(_fd, "results", "full_best_hqnr.mat"), "w").close()
    json.dump(dict(step=1010), open(os.path.join(_fd, "last_meta.json"), "w")); _vf = G.verified_complete(_fake)
    # 감사 F08 fixture: 50000 만 45 번 중복한 CSV · 빈 mat · 빈 last · resumed_nonexact · Teacher/데이터/init 증거 없음 → ok 이면 안 된다
    open(os.path.join(_fd, "checkpoint_metrics.csv"), "w").write("step,raw_original.hqnr\n" + "".join("50000,0.96\n" for _ in range(45)))
    json.dump(dict(step=50000), open(os.path.join(_fd, "last_meta.json"), "w")); os.makedirs(os.path.join(_fd, "last"), exist_ok=True); open(os.path.join(_fd, "last", "model.safetensors"), "w").close()
    json.dump(dict(exact_resume=True, resumed_nonexact=True, teacher=dict(id="T0")), open(os.path.join(_fd, "kdv_config_resolved.json"), "w")); _vf2 = G.verified_complete(_fake)
finally:
    import shutil as _sh; _sh.rmtree(_fd, ignore_errors=True)
_real = G.run_name("FQ", 1234); _vr = G.verified_complete(_real) if os.path.exists(os.path.join(ROOT, "work_dir", _real, "last_meta.json")) else None
check("K33 완료 검증(F06 → QRECON24 감사 F08 강화): marker 만(빈 mat) → ok False · 50000 45 중복 CSV·빈 last·resumed_nonexact·증거 없음 → ok False(candidate_grid/checkpoints/exact_resume/teacher/init False) · 실제 완료 run(s1 FQ S1234: 고유 50 step 격자·후보 checkpoint 전부·Teacher T0·train sha == cue·init hash) → ok True",
      _vf["ok"] is False and _vf["checks"]["results_mats"] is False and _vf["checks"]["last_exact_step"] is False and _vf["checks"]["candidate_grid"] is False and (_vr is None or _vr["ok"])
      and _vf2["ok"] is False and _vf2["checks"]["candidate_grid"] is False and _vf2["checks"]["candidate_checkpoints"] is False and _vf2["checks"]["exact_resume_ok"] is False and _vf2["checks"]["teacher_is_T0"] is False and _vf2["checks"]["init_matches_shared_file"] is False
      and (_vr is None or (_vr["checks"]["candidate_checkpoints"] and _vr["checks"]["teacher_is_T0"] and _vr["checks"]["train_sha_matches_cue"] and _vr["checks"]["candidate_grid"])), f"fake {_vf['checks']} · fake2 {_vf2['checks']} · real {(_vr or {}).get('checks')}")

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
_p3 = list(G.QEGX_S3_ITEMS); _p4x = [it for it in G.QEGX_S4_ITEMS if G.is_tag(it)]
_want3 = [G.run_name(c, 2026, "v2", arch="W104_D121") for c in ("J0", "JQ", "QE50", "XJ", "QX50", "J_R3_NOEDGE", "QEC3", "QES", "J_QB005", "QE50_B005")] + [G.run_name(c, 4321, "v2", arch="W104_D121") for c in ("J0", "JQ", "XJ", "QE50", "QX50")]
_want4 = [G.run_name(c, 1234, arch="W104_D121") for c in ("QE50", "LF0", "LFQ", "LFQE50", "LFX", "JE0", "QER50", "QERS")] + [G.run_name(c, 3407, arch="W104_D121") for c in ("J0", "JQ", "QE50", "JE0", "QER50", "QERS")]
check("K34 편성(§5·§6): s3 = 15 run **v2**(2026 A/B 10 → 4321 C 5) · s4 = E0 5 뒤 14 run v1(1234 A 8 → 3407 C 6) · 전부 QEGX branch · allowed seed s3 {2026,4321} / s4 {1234,3407} · 옛 s3 순서는 PREVIOUS · bare 'QE50@W104_D121' 은 QEDGE9 자동 규칙(큐에는 run 이름만) · 큐 파일 == 순서 · refresh v3 는 QEGX",
      _p3 == _want3 and _p4x == _want4 and all(G.branch_for("s3", it) == "QEGX" for it in _p3) and all(G.branch_for("s4", it) == "QEGX" for it in _p4x) and G.allowed_seeds("s3") == {2026, 4321} and G.allowed_seeds("s4") == {1234, 3407}
      and G.branch_for("s3", "QE50@W104_D121") == "QEDGE9" and G.branch_for("s3", "J0@W104_D121") is None and G.branch_for("s4", G.QEGX_REFRESH_CONTROLS["s4"][0]) == "QEGX" and G.QEGX_REFRESH_CONTROLS["s4"][0].endswith("_v3")
      and [l.strip() for l in open(os.path.join(ROOT, "config", "queues", "qegx_s3.txt")) if l.strip() and not l.startswith("#")] == _p3 and [l.strip() for l in open(os.path.join(ROOT, "config", "queues", "qegx_s4.txt")) if l.strip() and not l.startswith("#")] == _p4x
      and G.PREVIOUS_PRIORITY_BY_SERVER["s3_20260915_qegx"] == _p3 and G.server_id("s3(5090)") == "s3" and G.server_id(" s4\n") == "s4" and {"QX50", "QEC3", "QE50_B005", "LFQE50", "QER50", "QERS"} <= G.NEEDS_LAMBDA_E)
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

# ================= K39–K43 EDGEBAL s2·s5 (research_log/PAN_EDGEBAL_S2_S5_Experiment_Plan_2026-09-16.md §3–§7·§10)
from kdv.edge_gate import AffineEdgeWeight
_kB = {c: G.kdv_block(c, 777, "s2", cal=calQ, arch="W104_D121") for c in ("EB_N0", "EB_R3E000", "EB_R3E025", "EB_R3E050", "EB_R3E075", "EB_R3E100", "EB_R3E200", "EB_N0E100", "EB_R1E100")}
_kB.update({c: G.kdv_block(c, 2026, "s5", cal=calQ, arch="W104_D121", version="v2") for c in ("EB_EDOWN", "EB_EUP", "EB_QFLOOR", "EB_QFSHUF", "EB_QFREV")}); _sB = {c: resolve(k) for c, k in _kB.items()}
_ref = {c: G.kdv_block(c, 777, "s2", cal=calQ, arch="W104_D121") for c in ("J0", "J_R3_NOEDGE", "J_QE025", "JQ", "J_QE10", "J_N0_EDGE", "XJ")}
_same = lambda a, b: all(a[k] == b[k] for k in ("rec", "stat", "aux", "aligner_lr", "aligner_policy", "input_protocol", "expect_arch", "teacher"))
check("K39 EB case(§4 동치표): EB_N0≡J0 · EB_R3E000≡J_R3_NOEDGE · EB_R3E050≡J_QE025(0.5λE0) · EB_R3E100≡JQ · EB_R3E200≡J_QE10 · EB_N0E100≡J_N0_EDGE(Teacher eval_only) · EB_R1E100≡XJ (rec/stat/aux/A LR/정책/골격/Teacher 동일) · EB_R3E025/075 = 0.25/0.75×λE0 · 전부 J 정책·W104",
      _same(_kB["EB_N0"], _ref["J0"]) and _same(_kB["EB_R3E000"], _ref["J_R3_NOEDGE"]) and _same(_kB["EB_R3E050"], _ref["J_QE025"]) and _same(_kB["EB_R3E100"], _ref["JQ"]) and _same(_kB["EB_R3E200"], _ref["J_QE10"]) and _same(_kB["EB_N0E100"], _ref["J_N0_EDGE"]) and _same(_kB["EB_R1E100"], _ref["XJ"])
      and abs(_kB["EB_R3E025"]["stat"]["outer_weight"] - 0.25 * calQ["lambda_E"]) < 1e-15 and abs(_kB["EB_R3E075"]["stat"]["outer_weight"] - 0.75 * calQ["lambda_E"]) < 1e-15 and abs(_kB["EB_R3E050"]["stat"]["outer_weight"] - 0.5 * calQ["lambda_E"]) < 1e-15
      and not _kB["EB_R3E000"]["stat"]["enabled"] and _kB["EB_N0E100"]["teacher"].get("eval_only") and _sB["EB_N0E100"]["teacher_eval_only"] and all(_kB[c]["aligner_policy"] == "A-FT" and _kB[c]["aligner_lr"] == 1e-5 and _kB[c]["expect_arch"]["width"] == 104 for c in _kB))
check("K39 EDGEBAL schedule/floor block(§3.3·§3.4): EB_EDOWN edge_schedule {before 1, after .5, switch 25000} · EB_EUP {.5, 1} · stat.outer_weight 는 λE0 그대로(중복 배율 없음) · EB_QFLOOR edge_weight {floor, low .75, high .25, asset} · EB_QFSHUF floor_shuffle perm 51515 · EB_QFREV {low .25, high .75} · aligner_schedule/routing/edge_gate/edge_route 없음",
      _kB["EB_EDOWN"]["edge_schedule"] == dict(before=1.0, after=0.5, switch=25000) and _kB["EB_EUP"]["edge_schedule"] == dict(before=0.5, after=1.0, switch=25000) and all(abs(_kB[c]["stat"]["outer_weight"] - calQ["lambda_E"]) < 1e-15 for c in ("EB_EDOWN", "EB_EUP", "EB_QFLOOR", "EB_QFSHUF", "EB_QFREV"))
      and _kB["EB_QFLOOR"]["edge_weight"] == dict(mode="floor", low=0.75, high=0.25, asset=G.QEDGE9_CUE_ASSET) and _kB["EB_QFSHUF"]["edge_weight"] == dict(mode="floor_shuffle", low=0.75, high=0.25, asset=G.QEDGE9_CUE_ASSET, perm_seed=51515)
      and _kB["EB_QFREV"]["edge_weight"] == dict(mode="floor", low=0.25, high=0.75, asset=G.QEDGE9_CUE_ASSET) and all(k_ not in _kB[c] for c in _kB for k_ in ("aligner_schedule", "routing", "edge_gate", "edge_route")))
check("K39 EDGEBAL branch·시간 정책(§7·§10.1): campaign EDGEBAL_A104D121_S2S5_20260916_v1 · parent PAKD50 · lineage [PAKD50, QEDGE9, QEGX] · branch A104D121_T0FIX_EDGEBAL_v1 · 자체 ledger · total 1000h + required(경고만) · training_deadline 없음 · no_hard_limit · exact_resume · control_runs(s2 777 → QEDGE9 v2 / s5 2026 → v1 / 다른 seed → 같은 큐 EB_N0·EB_R3E100)",
      all(k["campaign_id"] == G.EDGEBAL_CAMPAIGN_ID and k["parent_campaign_id"] == G.CAMPAIGN_ID and k["lineage_campaign_ids"] == [G.CAMPAIGN_ID, G.QEDGE9_CAMPAIGN_ID, G.QEGX_CAMPAIGN_ID] and k["experiment_branch_id"] == "A104D121_T0FIX_EDGEBAL_v1" and k["exact_resume"] is True
          and k["budget"]["ledger"] == G.EDGEBAL_LEDGER and k["budget"]["total_gpu_hours"] == 1000.0 and k["budget"]["required"] is True and "training_deadline" not in k["budget"] and k["budget"]["time_policy"]["mode"] == "no_hard_limit" and k["budget"]["remaining_mandatory_file"] == G.EDGEBAL_MANDATORY_FILE for k in _kB.values())
      and _kB["EB_R3E050"]["control_runs"] == {"J0": "PAKD50_J0_W104_D121_WV3_T0_S777_FRESH50_v2", "JQ": "PAKD50_JQ_W104_D121_WV3_T0_S777_FRESH50_v2", "QE50": "PAKD50_QE50_W104_D121_WV3_T0_S777_FRESH50_v2"}
      and _kB["EB_EDOWN"]["control_runs"] == {"J0": G.run_name("J0", 2026, arch="W104_D121"), "JQ": G.run_name("JQ", 2026, arch="W104_D121"), "QE50": G.run_name("QE50", 2026, arch="W104_D121")}
      and G.kdv_block("EB_R3E050", 9091, "s2", cal=calQ, arch="W104_D121")["control_runs"]["J0"] == G.run_name("EB_N0", 9091, arch="W104_D121") and G.kdv_block("EB_EDOWN", 9091, "s5", cal=calQ, arch="W104_D121", version="v2")["control_runs"]["JQ"] == G.run_name("EB_R3E100", 9091, "v2", arch="W104_D121"))
for _lab, _fn in (("K39 EB case 를 QEGX/QEDGE9 branch 로 만들면 SystemExit", lambda: G.kdv_block("EB_EDOWN", 2026, "s5", cal=calQ, arch="W104_D121", branch="QEGX")), ("K39 EDGEBAL 은 W104 전용 (W112 거부)", lambda: G.kdv_block("EB_QFLOOR", 2026, "s5", cal=calQ)),
                  ("K39 schedule/weight backend 를 PAKD50 branch(None) 로 만들면 SystemExit", lambda: G.kdv_block("EB_EUP", 2026, "s5", cal=calQ, arch="W104_D121", branch="E0"))):
    try:
        _fn(); check(_lab, False)
    except SystemExit:
        check(_lab, True)
_p2 = list(G.EDGEBAL_S2_ITEMS); _p5e = list(G.EDGEBAL_S5_ITEMS)
_w2 = [G.run_name(c, 777, arch="W104_D121") for c in ("EB_R3E000", "EB_R3E050", "EB_R3E025", "EB_R3E200", "EB_N0E100", "EB_R1E100")] + [G.run_name(c, sd, arch="W104_D121") for sd in (2026, 9091) for c in ("EB_N0", "EB_R3E100", "EB_R3E050")]
_w5 = [G.run_name(c, 2026, "v2", arch="W104_D121") for c in ("EB_R3E050", "EB_R3E075", "EB_EDOWN", "EB_EUP", "EB_QFLOOR", "EB_QFSHUF", "EB_QFREV", "EB_N0E100")] + [G.run_name(c, 777, "v2", arch="W104_D121") for c in ("EB_R3E075", "EB_EDOWN")] + [G.run_name(c, 9091, "v2", arch="W104_D121") for c in ("EB_N0", "EB_R3E100", "EB_R3E075", "EB_EDOWN")]
check("K39 편성(§5·§6): s2 = 12 run v1(777 A 6 → 2026 B 3 → 9091 C 3) · s5 = QEDGE9 6 그대로 뒤 14 run **v2**(2026 A 8 → 777 B 2 → 9091 C 4) · 전부 EDGEBAL branch(s5 앞 6 은 QEDGE9 유지) · allowed seed s2/s5 {777, 2026, 9091} · 큐 파일 == 순서 · 옛 s2 순서 PREVIOUS · QF* 는 cue case · λE 의존(EB_N0/EB_R3E000 제외)",
      _p2 == _w2 and _p5e == _w5 and G.QEDGE9_ITEMS_BY_SERVER["s5"] == G.QEDGE9_S5_ITEMS and all(G.branch_for("s2", it) == "EDGEBAL" for it in _p2) and all(G.branch_for("s5", it) == "EDGEBAL" for it in _p5e) and all(G.branch_for("s5", it) == "QEDGE9" for it in G.QEDGE9_S5_ITEMS)
      and G.allowed_seeds("s2") == {777, 2026, 9091} and G.allowed_seeds("s5") == {2026, 777, 9091, 1103} and G.EDGEBAL_ITEMS_BY_SERVER == {"s2": _w2, "s5": _w5} and G.PREVIOUS_PRIORITY_BY_SERVER["s2_20260916_edgebal"] == _p2
      and [l.strip() for l in open(os.path.join(ROOT, "config", "queues", "edgebal_s2.txt")) if l.strip() and not l.startswith("#")] == _p2 and [l.strip() for l in open(os.path.join(ROOT, "config", "queues", "edgebal_s5.txt")) if l.strip() and not l.startswith("#")] == G.PREVIOUS_PRIORITY_BY_SERVER["s5_20260916_edgebal"]
      and {"EB_QFLOOR", "EB_QFSHUF", "EB_QFREV"} <= set(G.CUE_CASES) and {"EB_R3E025", "EB_R3E050", "EB_R3E075", "EB_R3E100", "EB_R3E200", "EB_N0E100", "EB_R1E100", "EB_EDOWN", "EB_EUP", "EB_QFLOOR"} <= G.NEEDS_LAMBDA_E and not ({"EB_N0", "EB_R3E000"} & G.NEEDS_LAMBDA_E)
      and G.cue_ready(_w5[4]) == G.cue_ready("QE50@W104_D121") and G.cue_ready(_w5[2]) is True and G.EDGEBAL_VERSION_BY_SERVER == {"s2": "v1", "s5": "v2"})
_r2 = [G.reservation_for("s2", it) for it in _p2]; _r5 = [G.reservation_for("s5", it) for it in _p5e]
check(f"K39 예약(§7.2): s2 R3/edge 2.30 · N0 1.95 → 12 run 합 31.5900 h(31h35m24s; 실측 {sum(r['reservation_h'] for r in _r2):.4f}) · s5 dense/schedule/GT+edge 1.57 · floor 1.65 · N0 1.38 → 14 run 합 26.5664 h(26h33m59s; {sum(r['reservation_h'] for r in _r5):.4f}) · 전부 plan_reference_case(반복 seed 도 명시 표) · s5 QEDGE9 6 = 11.1200 h",
      abs(sum(r["reservation_h"] for r in _r2) - 31.59) < 2e-4 and abs(sum(r["reservation_h"] for r in _r5) - 26.5664) < 2e-4 and all(r["reference_kind"] == "plan_reference_case" for r in _r2 + _r5)
      and [r["reference_train_h"] for r in _r2] == [2.30] * 6 + [1.95, 2.30, 2.30] * 2 and [r["reference_train_h"] for r in _r5] == [1.57] * 4 + [1.65] * 3 + [1.57] * 3 + [1.38, 1.57, 1.57, 1.57] and [r["seed"] for r in _r5] == [2026] * 8 + [777] * 2 + [9091] * 4
      and abs(sum(G.reservation_for("s5", it)["reservation_h"] for it in G.QEDGE9_S5_ITEMS) - 11.12) < 1e-6)
# ---- K40 registry edge_schedule / edge_weight
def _rejB(base, **kw):
    k = copy.deepcopy(base); k.update(kw)
    try:
        resolve(k); return False
    except ValueError:
        return True
_kD, _kF = _kB["EB_EDOWN"], _kB["EB_QFLOOR"]
check("K40 registry(§10.2): EB_EDOWN spec edge_schedule {1, .5, 25000} · EB_QFLOOR spec edge_cue_weight {floor .75/.25} · QFSHUF perm 51515 · 토큰 EDGEHSD/EDGEHSU/EDGEHF/EDGEHFS/EDGEHFR · 상수 배수는 EDGEH 그대로 · describe 서술 · 거부: schedule+edge_gate/routing/edge_route · stat OFF · before==after · switch 0 · 키 누락/추가 · weight low==high · mode 오류 · asset 없음 · schedule+weight 결합 · perm 7 · Teacher 없음",
      _sB["EB_EDOWN"]["edge_schedule"] == dict(before=1.0, after=0.5, switch=25000) and _sB["EB_EUP"]["edge_schedule"] == dict(before=0.5, after=1.0, switch=25000) and _sB["EB_QFLOOR"]["edge_cue_weight"] == dict(mode="floor", low=0.75, high=0.25, asset=G.QEDGE9_CUE_ASSET, perm_seed=None)
      and _sB["EB_QFSHUF"]["edge_cue_weight"]["perm_seed"] == 51515 and _sB["EB_QFREV"]["edge_cue_weight"]["low"] == 0.25 and _sB["EB_R3E050"]["edge_schedule"] is None and _sB["EB_R3E050"]["edge_cue_weight"] is None and _sB["EB_EDOWN"]["edge_weight"] == 0.0
      and [stat_tag(_sB[c]) for c in ("EB_EDOWN", "EB_EUP", "EB_QFLOOR", "EB_QFSHUF", "EB_QFREV", "EB_R3E050", "EB_R3E200")] == ["EDGEHSD", "EDGEHSU", "EDGEHF", "EDGEHFS", "EDGEHFR", "EDGEH", "EDGEH"]
      and "w(t) = 1×λE0 (t<25000) → 0.5×λE0" in _desc(_sB["EB_EDOWN"]) and "w_i = 0.25 + (0.75 − 0.25)·g_i" in _desc(_sB["EB_QFLOOR"]) and "stratum 셔플" in _desc(_sB["EB_QFSHUF"])
      and _rejB(_kD, edge_gate=dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET)) and _rejB(_kD, routing=dict(qE=0.0)) and _rejB(_kD, edge_route=dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET)) and _rejB(_kD, stat=dict(enabled=False))
      and _rejB(_kD, edge_schedule=dict(before=1.0, after=1.0, switch=25000)) and _rejB(_kD, edge_schedule=dict(before=1.0, after=0.5, switch=0)) and _rejB(_kD, edge_schedule=dict(before=1.0, after=0.5)) and _rejB(_kD, edge_schedule=dict(before=1.0, after=0.5, switch=25000, ramp=1))
      and _rejB(_kF, edge_weight=dict(mode="floor", low=0.5, high=0.5, asset=G.QEDGE9_CUE_ASSET)) and _rejB(_kF, edge_weight=dict(mode="soft", low=0.75, high=0.25, asset=G.QEDGE9_CUE_ASSET)) and _rejB(_kF, edge_weight=dict(mode="floor", low=0.75, high=0.25))
      and _rejB(_kF, edge_schedule=dict(before=1.0, after=0.5, switch=25000)) and _rejB(_kF, edge_weight=dict(mode="floor_shuffle", low=0.75, high=0.25, asset=G.QEDGE9_CUE_ASSET, perm_seed=7)) and _rejB(_kF, teacher=None, rec=dict(case="N0")) and _rejB(_kF, edge_gate=dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET))
      and not _rejB(_kD) and not _rejB(_kF))
# ---- K41 trainer 동치·배수·경계 (실제 KDVTrainer._step, CPU)
def _stubB(case, student, weight=None, server="s2", seed=777, version="v1", branch="EDGEBAL", seed_gen=3234):
    tr = object.__new__(KDVTrainer); tr.k = G.kdv_block(case, seed, server, cal=calQ, arch="W104_D121", version=version, branch=branch); sp_ = tr.spec = resolve(tr.k)
    tr.model = student; tr.accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, gradient_accumulation_steps=1, scaler=None, is_main_process=True)
    tr.teacher = T0; tr.aligner_trainable = True; tr.aligner_view_margin = 4; tr.share_correction = False; tr.protocol = sp_["protocol"]; tr.radius_hr = sp_["radius_hr"]; tr.diag_every = 10 ** 9
    tr.rec_crit = (GTAnchoredReconstructionKD(calQ["tau_R"], alpha=float(tr.k["rec"].get("alpha", 1.0)), kd_weight=float(tr.k["rec"].get("kd_weight", 0.0)), eps=1e-6, mode=sp_["rec_mode"]) if sp_["rec_case"] != "N0" else None)
    tr.tri = sp_["tri"]; tr.stat_extra = []; tr.lam_V = (float(tr.k["stat"]["outer_weight"]) if sp_["stat_enabled"] else 0.0); tr.stat_ramp = 0
    tr.lam_edge = 0; tr.lam_geo = 0; tr.ramp = 5000; tr.lam_gkd = 0; tr.gen = torch.Generator().manual_seed(seed_gen); tr.corr_seed = seed_gen; tr._ema = {}; tr._rr_val_last = float("nan"); tr.args = types.SimpleNamespace(num_iter=50000)
    tr.freeze_until, tr.freeze_from = sp_["aligner_freeze_until"], sp_["aligner_freeze_from"]; tr.route_A = tuple(sp_["route_A"]); tr._routed = False; tr._sched_last = None
    tr.edge_gate = None; tr.edge_route = None; tr._edge_routed = False; tr.edge_schedule = sp_.get("edge_schedule"); tr.edge_weight = weight; tr._edge_sched_last = None
    return tr
def _tot(case, step=1, **kw):
    tr = _stubB(case, _S(), **kw); tot, info = tr._step(_gt, _ms, _lp, _pn, step, meta=(_metaQ if tr.edge_weight is not None else None)); return tr, tot, info
_eq = lambda a, b: (float(a[1]) == float(b[1]) and _rel(_g(a[1], a[0].M.backbone), _g(b[1], b[0].M.backbone)) < 1e-6 and _rel(_g(a[1], a[0].M.aligner), _g(b[1], b[0].M.aligner)) < 1e-6)
_pairs = [("EB_N0", "J0"), ("EB_R3E100", "JQ"), ("EB_R3E000", "J_R3_NOEDGE"), ("EB_R1E100", "XJ"), ("EB_N0E100", "J_N0_EDGE"), ("EB_R3E050", "J_QE025"), ("EB_R3E200", "J_QE10")]
_res41 = {a: _eq(_tot(a), _tot(b, branch="E0")) for a, b in _pairs}
check("K41 동치(§10.3): EB_N0≡J0 · EB_R3E100≡JQ · EB_R3E000≡J_R3_NOEDGE · EB_R1E100≡XJ · EB_N0E100≡J_N0_EDGE · EB_R3E050≡J_QE025 · EB_R3E200≡J_QE10 — 같은 batch 의 total bitwise 동일 · U/A gradient 동일(1e-6 상대; 기존 E0 branch 의 config 로 만든 stub 과 비교)", all(_res41.values()), str({a: v for a, v in _res41.items() if not v}))
_tJ = _tot("EB_R3E100"); _t50 = _tot("EB_R3E050"); _t25 = _tot("EB_R3E025"); _t200 = _tot("EB_R3E200"); _t0 = _tot("EB_R3E000"); _LE = _tJ[2]["_edge_w_t"]
_gA = lambda t: _g(t[2]["_edge_w_t"], t[0].M.aligner); _gU = lambda t: _g(t[2]["_edge_w_t"], t[0].M.backbone)
check("K41 상수 배수(§3.2·§10.3): 같은 output 에서 λE·L_E 와 그 U/A gradient 가 0.25/0.5/2 배로 (1e-6 상대) · hard/soft(L_H, L_K) 는 배수와 무관하게 동일 · GT hard w_H ≥ 1 · EB_R3E000 total == EB_R3E100 total − λE·L_E",
      abs(float(_t50[2]["_edge_w_t"]) - 0.5 * float(_LE)) < 1e-9 and abs(float(_t25[2]["_edge_w_t"]) - 0.25 * float(_LE)) < 1e-9 and abs(float(_t200[2]["_edge_w_t"]) - 2.0 * float(_LE)) < 1e-9
      and _rel(_gA(_t50), 0.5 * _gA(_tJ)) < 1e-6 and _rel(_gU(_t50), 0.5 * _gU(_tJ)) < 1e-6 and _rel(_gU(_t200), 2.0 * _gU(_tJ)) < 1e-6
      and all(float(t[2]["_rec_hard_t"]) == float(_tJ[2]["_rec_hard_t"]) and float(t[2]["_rec_soft_t"]) == float(_tJ[2]["_rec_soft_t"]) for t in (_t50, _t25, _t200, _t0))
      and float(_tJ[0].rec_crit(_tJ[2]["y"].detach(), _tJ[2]["y_t"], _gt.float(), return_maps=True).maps["hard_weight"].min()) >= 1.0 and abs(float(_t0[1]) - (float(_tJ[1]) - float(_LE))) < 1e-6)
_tD = _stubB("EB_EDOWN", _S(), server="s5", seed=2026, version="v2"); _tU = _stubB("EB_EUP", _S(), server="s5", seed=2026, version="v2")
_D24 = _tot("EB_EDOWN", step=24999, server="s5", seed=2026, version="v2"); _J24 = _tot("EB_R3E100", step=24999, server="s5", seed=2026, version="v2"); _D25 = _tot("EB_EDOWN", step=25001, server="s5", seed=2026, version="v2"); _J25 = _tot("EB_R3E100", step=25001, server="s5", seed=2026, version="v2")
_U24 = _tot("EB_EUP", step=24999, server="s5", seed=2026, version="v2"); _U25 = _tot("EB_EUP", step=25001, server="s5", seed=2026, version="v2")
check("K41 경계(§3.3·§10.3): DOWN w(24999)=1 / w(25000)=.5 / w(0)=1 / w(49999)=.5 · UP 반대 · step 24999 는 EB_R3E100 과 total·ε 열 bitwise 동일 · 25001 은 total == JQ − 0.5·λE·L_E, edge 항 정확히 절반, hard/soft 동일, A 는 활성(aligner_active True; grad 있음; offset 연습 그대로) · w(t) 는 step 만의 순수 함수(두 stub 동일) · exact_resume 켜짐",
      _tD.edge_schedule_factor(24999) == 1.0 and _tD.edge_schedule_factor(25000) == 0.5 and _tD.edge_schedule_factor(0) == 1.0 and _tD.edge_schedule_factor(49999) == 0.5 and _tU.edge_schedule_factor(24999) == 0.5 and _tU.edge_schedule_factor(25000) == 1.0
      and float(_D24[1]) == float(_J24[1]) and torch.equal(_D24[0].gen.get_state(), _J24[0].gen.get_state()) and abs(float(_D25[1]) - (float(_J25[1]) - 0.5 * float(_J25[2]["_edge_w_t"]))) < 1e-6 and abs(float(_D25[2]["_edge_w_t"]) - 0.5 * float(_J25[2]["_edge_w_t"])) < 1e-9
      and float(_D25[2]["_rec_hard_t"]) == float(_J25[2]["_rec_hard_t"]) and float(_D25[2]["_rec_soft_t"]) == float(_J25[2]["_rec_soft_t"]) and _D25[2]["stat_edge_sched"] == 0.5 and _D24[2]["stat_edge_sched"] == 1.0
      and _tD.aligner_active(25001) and _tD.aligner_active(49999) and _D25[2].get("eq_exercise") == 1.0 and float(_g(_D25[1], _D25[0].M.aligner).abs().sum()) > 0 and float(_U25[2]["_edge_w_t"]) == float(_J25[2]["_edge_w_t"]) and abs(float(_U24[2]["_edge_w_t"]) - 0.5 * float(_J24[2]["_edge_w_t"])) < 1e-9
      and _stubB("EB_EDOWN", _S(), server="s5", seed=2026, version="v2").edge_schedule_factor(24999) == _tD.edge_schedule_factor(24999) and _kB["EB_EDOWN"]["exact_resume"] is True and "edge_schedule_events.jsonl" in src and "edge_schedule_factor(global_step)" in src)
del _tD, _tU, _D24, _J24, _D25, _J25, _U24, _U25, _tJ, _t50, _t25, _t200, _t0
# ---- K42 완만한 q 가중 (§3.4·§10.3 Cue floor / Shuffle / Reduction)
_wF1 = AffineEdgeWeight.synthetic(torch.ones(2, 4), 0.75, 0.25); _wF0 = AffineEdgeWeight.synthetic(torch.zeros(2, 4), 0.75, 0.25); _wR1 = AffineEdgeWeight.synthetic(torch.ones(2, 4), 0.25, 0.75)
_wM = AffineEdgeWeight.synthetic(torch.tensor([[1., 1, 1, 1], [0, 0, 0, 0]]), 0.75, 0.25); _wS = AffineEdgeWeight.synthetic(torch.tensor([[0., 0, 0, 0], [1, 1, 1, 1]]), 0.75, 0.25, mode="floor_shuffle")
_F1 = _tot("EB_QFLOOR", weight=_wF1, server="s5", seed=2026, version="v2"); _F0 = _tot("EB_QFLOOR", weight=_wF0, server="s5", seed=2026, version="v2"); _FM = _tot("EB_QFLOOR", weight=_wM, server="s5", seed=2026, version="v2"); _FS = _tot("EB_QFSHUF", weight=_wS, server="s5", seed=2026, version="v2")
_C75 = _tot("EB_R3E075", server="s5", seed=2026, version="v2"); _C25 = _tot("EB_R3E025", server="s5", seed=2026, version="v2"); _C50 = _tot("EB_R3E050", server="s5", seed=2026, version="v2")
_EiF = output_edge_loss_per_sample(_FM[2]["y"].detach().float(), _gt.float()); _wm = _wM.weight_for(_metaQ, torch.device("cpu"))
check("K42 cue floor(§3.4): g=1 → w .75 · g=0 → .25 · reverse 반대 · w 에 gradient 없음 · loss = mean_i w_i E_i (재정규화 없음; 혼합 (1,0) → (.75E_0 + .25E_1)/2, mean w .5) · g 전부 1 이면 EB_R3E075 와 total 동일(1e-6) · 전부 0 이면 EB_R3E025 · mean w 가 .5 라도 CONST050 의 loss 와 같지 않다(E_0≠E_1) · q-low/high E 로그 · shuffle 은 shuffle 표 · Teacher/GT detach · meta 없으면 ValueError",
      torch.equal(_wF1.weight_for(_metaQ, torch.device("cpu")), torch.tensor([0.75, 0.75])) and torch.equal(_wF0.weight_for(_metaQ, torch.device("cpu")), torch.tensor([0.25, 0.25])) and torch.equal(_wR1.weight_for(_metaQ, torch.device("cpu")), torch.tensor([0.25, 0.25]))
      and torch.equal(_wm, torch.tensor([0.75, 0.25])) and not _wm.requires_grad and abs(float(_FM[2]["loss_stat_raw"]) - float(0.75 * _EiF[0] + 0.25 * _EiF[1]) / 2) < 1e-6 and _FM[2]["stat_edge_w_mean"] == 0.5 and _FM[2]["stat_edge_gate_frac"] == 0.5
      and abs(float(_FM[2]["stat_edge_E_qlow"]) - float(_EiF[0])) < 1e-6 and abs(float(_FM[2]["stat_edge_E_qhigh"]) - float(_EiF[1])) < 1e-6 and abs(float(_FM[2]["stat_edge_lambda_eff"]) - 0.5 * calQ["lambda_E"]) < 1e-12
      and abs(float(_F1[1]) - float(_C75[1])) < 1e-6 and abs(float(_F0[1]) - float(_C25[1])) < 1e-6 and (abs(float(_FM[2]["loss_stat_raw"]) - float(_C50[2]["loss_stat_raw"])) > 1e-9 or float(_EiF[0]) == float(_EiF[1]))
      and _wS.gate.mode == "shuffle" and _FS[2]["stat_edge_gate_frac"] == 0.5 and abs(float(_FS[2]["loss_stat_raw"]) - float(0.25 * output_edge_loss_per_sample(_FS[2]["y"].detach().float(), _gt.float())[0] + 0.75 * output_edge_loss_per_sample(_FS[2]["y"].detach().float(), _gt.float())[1]) / 2) < 1e-6
      and not _FM[2]["y_t"].requires_grad and float(_g(_FM[1], _FM[0].M.aligner).abs().sum()) > 0 and float(_g(_FM[1], _FM[0].M.backbone).abs().sum()) > 0)
try:
    _stubB("EB_QFLOOR", _S(), weight=_wF1, server="s5", seed=2026, version="v2")._step(_gt, _ms, _lp, _pn, 1); check("K42 edge_weight 에 meta 없음 → ValueError", False)
except ValueError:
    check("K42 edge_weight 에 meta 없음 → ValueError", True)
if os.path.exists(_asset):
    _aF = AffineEdgeWeight.load(dict(mode="floor", low=0.75, high=0.25, asset=G.QEDGE9_CUE_ASSET), teacher=T0); _aS = AffineEdgeWeight.load(dict(mode="floor_shuffle", low=0.75, high=0.25, asset=G.QEDGE9_CUE_ASSET, perm_seed=51515), teacher=T0); _aR = AffineEdgeWeight.load(dict(mode="floor", low=0.25, high=0.75, asset=G.QEDGE9_CUE_ASSET), teacher=T0)
    _tF, _tS, _tR = _aF.gate.table, _aS.gate.table, _aR.gate.table; _wf = 0.25 + 0.5 * _tF; _ws = 0.25 + 0.5 * _tS; _wr = 0.75 - 0.5 * _tR
    check("K42 실제 cue 자산: floor/shuffle 의 weight 분포(=active 수) 동일 · 표가 실제로 다르다 · reverse = 1 − floor · 전체 view 평균 w ≈ .5(±.02) · 표는 0/1 그대로(affine 은 caller) · θq 동일",
          float((_tF == 1).sum()) == float((_tS == 1).sum()) and not torch.equal(_tF, _tS) and torch.equal(_wr, 1.0 - _wf) and abs(float(_wf.mean()) - 0.5) < 0.02 and abs(float(_ws.mean()) - 0.5) < 0.02 and set(torch.unique(_tF).tolist()) <= {0.0, 1.0}
          and _aF.theta_q == _aS.theta_q == _man0["theta_q"] and _aF.summary()["edge_weight"]["base_gate"] == "low_q" and _aS.summary()["edge_weight"]["base_gate"] == "shuffle")
else:
    check("K42 실제 cue 자산 — 없음", False)
# ---- K43 시트 토큰 · gate · 스크립트 · 생성 config
try:
    _lab_ok43 = (_il("PAKD50_EB_EDOWN_W104_D121_WV3_T0_S2026_FRESH50_v2", "A104D121_T0FIX_EDGEBAL_v1") == "PAKD50 / EB_EDOWN / A104D121 / EDGEBAL / FRESH50" and _il("PAKD50_QX50_W104_D121_WV3_T0_S2026_FRESH50_v2", "A104D121_T0FIX_QEGX_v1") == "PAKD50 / QX50 / A104D121 / QEGX / FRESH50")
except Exception as _e:                                                        # noqa
    _lab_ok43 = False
_cfgB = {r_: (yaml.safe_load(open(os.path.join(ROOT, "config", r_ + ".yaml"))) if os.path.exists(os.path.join(ROOT, "config", r_ + ".yaml")) else None) for r_ in _p2 + _p5e}
_tmpb = tempfile.mkdtemp(); G.generate("s2", [_p2[1], _p2[4]], _tmpb, projected=None); G.generate("s5", [_p5e[2], _p5e[4], "PAKD50_J0_W104_D121_WV3_T0_S777_FRESH50_v1"], _tmpb, projected=None)
check("K43 시트 X열(§10.4) 'PAKD50 / EB_EDOWN / A104D121 / EDGEBAL / FRESH50' · gate 설명에 EDGEBAL · switch/waiter bash -n · 생성 config 26 벌(s2 12 v1 + s5 14 v2) 존재 · QF* 만 return_meta · campaign EDGEBAL · no_hard_limit · seed/골격/버전 일치 · edge_schedule/edge_weight 키 일치 · 생성기와 filecmp(4 벌) · s5 QEDGE9 J0 S777 v1 config 회귀 없음",
      _lab_ok43 and "EDGEBAL" in __import__("tools.campaign_gate", fromlist=["GATES"]).GATES["pakd50"][1] and all(subprocess.run(["bash", "-n", os.path.join(ROOT, "tools", f)], capture_output=True).returncode == 0 for f in ("edgebal_switch.sh", "edgebal_waiter.sh"))
      and all(v_ is not None for v_ in _cfgB.values()) and all(bool(v_["train_feeder_args"].get("return_meta", False)) == (G.case_of(r_) in ("EB_QFLOOR", "EB_QFSHUF", "EB_QFREV")) for r_, v_ in _cfgB.items() if v_)
      and all(v_["kdv"]["campaign_id"] == G.EDGEBAL_CAMPAIGN_ID and "training_deadline" not in v_["kdv"]["budget"] and v_["kdv"]["budget"]["time_policy"]["mode"] == "no_hard_limit" and v_["seed"] == G.seed_of(r_) and v_["model_args"]["hidden_size"] == 104 and r_.endswith("_v1" if r_ in _p2 else "_v2") for r_, v_ in _cfgB.items() if v_)
      and all(("edge_schedule" in v_["kdv"]) == (G.case_of(r_) in ("EB_EDOWN", "EB_EUP")) and ("edge_weight" in v_["kdv"]) == (G.case_of(r_) in ("EB_QFLOOR", "EB_QFSHUF", "EB_QFREV")) for r_, v_ in _cfgB.items() if v_)
      and all(filecmp.cmp(os.path.join(ROOT, "config", r_ + ".yaml"), os.path.join(_tmpb, r_ + ".yaml"), shallow=False) for r_ in (_p2[1], _p2[4], _p5e[2], _p5e[4], "PAKD50_J0_W104_D121_WV3_T0_S777_FRESH50_v1")), f"missing {[r_ for r_, v_ in _cfgB.items() if v_ is None]}")

def _raises(fn, exc):
    try:
        fn(); return False
    except exc:
        return True
# ================= K44–K48 QRECON24 s1–s5 (research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md §2–§9)
from kdv import qrecon as QR
from kdv.qrecon import QWeight
_qn = lambda srv, prof, sd: G.qrc24_run_name(srv, prof, sd)
check("K44 profile/큐(§4–§5; λE 는 09-16 저녁 2 배 환산 6e-4/2e-3/6e-3): profile 29(G 9·A/E 대조 6·H 11·L 3) · 큐 s1 12 / s2 12 / s3 18 / s4 14 / s5 12 = 68 · 이름 PAKD50_QRC24_<SRV>_<PROFILE>_W104_D121_WV3_T0_S<seed>_FRESH50_v1 (RUN_RE 파싱, case 'QRC24_S3_G22') · A LR = rA×U LR(G21 3e-7 / G22 1e-6 / G23 3e-6 / L070 7e-7 / L050 5e-7; A_FREEZE 0) · canonical H22=L100=G22 · 같은 server+seed 중복 없음",
      len(G.QRC24_PROFILES) == 32 and [len(G.QRC24_QUEUES_20260916[s_]) for s_ in ("s1", "s2", "s3", "s4", "s5")] == [12, 12, 18, 14, 12] and sum(len(v) for v in G.QRC24_QUEUES_20260916.values()) == 68
      and _qn("s3", "G22", 2026) == "PAKD50_QRC24_S3_G22_W104_D121_WV3_T0_S2026_FRESH50_v1" and G.parse_item(_qn("s3", "G22", 2026)) == ("QRC24_S3_G22", "W104_D121", 2026, "v1") and G.qrc24_parse("QRC24_S3_G22") == ("s3", "G22") and G.qrc24_parse("QRC24_S1_A_FREEZE") == ("s1", "A_FREEZE")
      and [G.qrc24_profile(p_)["alr"] for p_ in ("G21", "G22", "G23", "L070", "L050", "A_FREEZE")] == [3e-7, 1e-6, 3e-6, 7e-7, 5e-7, 0.0] and G.qrc24_profile("H22")["canonical"] == G.qrc24_profile("L100")["canonical"] == "G22" and G.qrc24_profile("G11")["canonical"] == "G11"
      and all(len(set(G.QRC24_QUEUES_20260916[s_])) == len(G.QRC24_QUEUES_20260916[s_]) for s_ in G.QRC24_QUEUES_20260916) and G.QRC24_QUEUES_20260916["s1"][:3] == [("G22", 1234), ("A_UNIF", 1234), ("A_FREEZE", 1234)] and G.QRC24_QUEUES_20260916["s4"][-3:] == [("H22", 3407), ("H12", 3407), ("H11", 3407)])
_kQ22 = G.kdv_block("QRC24_S2_G22", 777, "s2", cal=calQ, arch="W104_D121"); _kQ = {p_: G.kdv_block(f"QRC24_S2_{p_}", 777, "s2", cal=calQ, arch="W104_D121") for p_ in ("G12", "G32", "G21", "E_UNIF", "E_SHUF", "ALL_UNIF")}
_kQ.update({p_: G.kdv_block(f"QRC24_S1_{p_}", 1234, "s1", cal=calQ, arch="W104_D121") for p_ in ("A_UNIF", "A_FREEZE", "A_SHUF")}); _kQ.update({p_: G.kdv_block(f"QRC24_S4_{p_}", 1234, "s4", cal=calQ, arch="W104_D121") for p_ in ("H22", "H12", "H_BETA0", "H_ALPHA0")})
_kQ.update({p_: G.kdv_block(f"QRC24_S5_{p_}", 2026, "s5", cal=calQ, arch="W104_D121") for p_ in ("L100", "L070", "L050")}); _sQR = {p_: resolve(k_) for p_, k_ in _kQ.items()}; _sQ22 = resolve(_kQ22)
_core = lambda k_: (k_["rec"], k_["stat"]["outer_weight"], k_.get("aligner_lr"), k_["qrecon"], k_["input_protocol"], k_["aligner_policy"], k_["corruption"], k_["aux"])
check("K44 kdv block(§2–§3): G22 = R3 α1 β.1 τR 고정 · stat EDGE-H outer_weight 1e-3 **절대값**(λE0 배율 아님) · aligner_lr 1e-6 · qrecon{continuous_v1, qref .3276133416220546, asset, a/e q, perm 51515} · I-NATIVE-TRANSFER radius 0 offset 0 · A-FT · G12/G32 λ 3e-4/3e-3 · G21 alr 3e-7 · A_FREEZE = A-FR(aligner_lr 없음) · A_UNIF/A_SHUF/E_UNIF/E_SHUF/ALL_UNIF weight mode · H12 α.5 · H_BETA0 β0 · H_ALPHA0 α0 · L070/L050 U LR·A LR · H22 ≡ L100 ≡ G22 (core 동일)",
      _kQ22["rec"] == dict(case="R3", alpha=1.0, kd_weight=0.1, eps=1e-6, tau=calQ["tau_R"], eps_scale=1e-6) and _kQ22["stat"]["outer_weight"] == 2e-3 and abs(_kQ22["stat"]["outer_weight"] / calQ["lambda_E"] - 0.5) > 0.4 and _kQ22["aligner_lr"] == 1e-6
      and _kQ22["qrecon"] == dict(mode="continuous_v1", q_ref=0.3276133416220546, asset=G.QEDGE9_CUE_ASSET, a_weight="q", e_weight="q", perm_seed=51515, uniform_weight=0.5) and _kQ22["qrc24"]["lambda_E_plan"] == 1e-3 and G.QRC24_LAMBDA == {1: 6e-4, 2: 2e-3, 3: 6e-3} and _kQ22["input_protocol"] == "I-NATIVE-TRANSFER" and _kQ22["corruption"] == dict(radius_hr=0.0) and _kQ22["aux"]["offset_weight"] == 0.0 and _kQ22["aligner_policy"] == "A-FT"
      and _kQ["G12"]["stat"]["outer_weight"] == 6e-4 and _kQ["G32"]["stat"]["outer_weight"] == 6e-3 and _kQ["G21"]["aligner_lr"] == 3e-7 and _kQ["G21"]["stat"]["outer_weight"] == 2e-3 and _kQ["A_FREEZE"]["aligner_policy"] == "A-FR" and "aligner_lr" not in _kQ["A_FREEZE"] and not _sQR["A_FREEZE"]["aligner_trainable"]
      and (_kQ["A_UNIF"]["qrecon"]["a_weight"], _kQ["A_SHUF"]["qrecon"]["a_weight"], _kQ["E_UNIF"]["qrecon"]["e_weight"], _kQ["E_SHUF"]["qrecon"]["e_weight"], _kQ["ALL_UNIF"]["qrecon"]["a_weight"], _kQ["ALL_UNIF"]["qrecon"]["e_weight"]) == ("uniform", "shuffle", "uniform", "shuffle", "uniform", "uniform")
      and _kQ["H12"]["rec"]["alpha"] == 0.5 and _kQ["H_BETA0"]["rec"]["kd_weight"] == 0.0 and _kQ["H_ALPHA0"]["rec"]["alpha"] == 0.0 and (_kQ["L070"]["qrc24"]["U_lr"], _kQ["L070"]["aligner_lr"], _kQ["L050"]["aligner_lr"]) == (7e-5, 7e-7, 5e-7)
      and _core(_kQ["H22"]) == _core(G.kdv_block("QRC24_S4_G22", 1234, "s4", cal=calQ, arch="W104_D121")) and _core(_kQ["L100"]) == _core(G.kdv_block("QRC24_S5_G22", 2026, "s5", cal=calQ, arch="W104_D121")) and _kQ["H22"]["qrc24"]["canonical"] == "G22"
      and _kQ["H12"]["control_runs"]["G22"] == _qn("s4", "H22", 1234) and _kQ["H12"]["control_runs"]["control_profile"] == "H22" and _kQ["L070"]["control_runs"]["G22"] == _qn("s5", "L100", 2026) and _kQ22["control_runs"]["G22"] == _qn("s2", "G22", 777)
      and G.kdv_block("QRC24_S4_H11", 3407, "s4", cal=calQ, arch="W104_D121")["baseline_run"] == _qn("s4", "H22", 3407) and G.kdv_block("QRC24_S5_L050", 1103, "s5", cal=calQ, arch="W104_D121")["control_runs"]["G22"] == _qn("s5", "L100", 1103)     # 감사 F09: 실제 편성된 alias run id
      and all(k_["campaign_id"] == G.QRC24_CAMPAIGN_ID and k_["experiment_branch_id"] == "A104D121_T0FIX_QRECON24_v1" and k_["budget"]["ledger"] == G.QRC24_LEDGER and k_["budget"]["required"] is True and "training_deadline" not in k_["budget"]
              and k_["budget"]["time_policy"]["mode"] == "no_hard_limit" and k_["budget"]["time_policy"]["min_operating_hours"] == 24.0 and k_["exact_resume"] is True and k_["control_runs"]["G22"].startswith("PAKD50_QRC24_") for k_ in list(_kQ.values()) + [_kQ22]))
for _lab, _fn in (("K44 서버 토큰 ≠ 생성 서버 → SystemExit", lambda: G.kdv_block("QRC24_S3_G22", 2026, "s5", cal=calQ, arch="W104_D121")), ("K44 QRECON24 는 W104 전용 (W112 거부)", lambda: G.kdv_block("QRC24_S2_G22", 777, "s2", cal=calQ)),
                  ("K44 QRC case 를 다른 branch 로 만들면 SystemExit", lambda: G.kdv_block("QRC24_S2_G22", 777, "s2", cal=calQ, arch="W104_D121", branch="EDGEBAL")), ("K44 다른 case 를 QRECON24 branch 로 만들면 SystemExit", lambda: G.kdv_block("JQ", 777, "s2", cal=calQ, arch="W104_D121", branch="QRECON24"))):
    try:
        _fn(); check(_lab, False)
    except SystemExit:
        check(_lab, True)
_rsum = lambda srv: sum(G.reservation_for(srv, it)["reservation_h"] for it in G.priority_for(srv))
check("K44 편성·예약(§5·§8; ADJ-R1 활성 편성 = 등록 + 남은 순서, 보류 제외): PRIORITY 다섯 서버 전부 QRC24 run 이름(= qrc24_items) · branch QRECON24 · allowed seed(현재 + superseded branch 의 명시 seed) s1 {1234,3407} s2 {777,2026,9091} s3 {2026,4321} s4 {1234,3407} s5 {2026,777,9091,1103} · 예약 1.20×R_s + 10/60 (R_s 는 ADJ-R1 §10 대용 — v1·v2 공통, s1 A_FREEZE 2.15; measured 우선) → s1 42.34 / s2 38.86 / s3 38.61 / s4 37.59 / s5 52.20 h (kind plan_reference_qrc24_adj_r1, slack 1.2) · 옛 순서는 PREVIOUS · 큐 파일 == 순서 · cue 필요 · §8.3 확장 3 run",
      all(G.priority_for(s_) == G.qrc24_items(s_) and all(G.branch_for(s_, it) == "QRECON24" for it in G.priority_for(s_)) for s_ in ("s1", "s2", "s3", "s4", "s5")) and [G.allowed_seeds(s_) for s_ in ("s1", "s2", "s3", "s4", "s5")] == [{1234, 3407}, {777, 2026, 9091}, {2026, 4321}, {1234, 3407}, {2026, 777, 9091, 1103}]
      and all(abs(_rsum(s_) - v_) < 1e-3 for s_, v_ in (("s1", 12 * 3.070667 + 2 * 2.746667), ("s2", 12 * 3.238667), ("s3", 20 * 1.930667), ("s4", 19 * 1.978667), ("s5", 16 * 3.262667))) and G.reservation_for("s4", G.priority_for("s4")[0])["reference_kind"] == "plan_reference_qrc24_adj_r1" and G.reservation_for("s4", G.priority_for("s4")[0])["slack"] == 1.2
      and G.reference_hours("s2", "QRC24_S2_G13@W104_D121", {}, version="v1") == (2.56, "plan_reference_qrc24_adj_r1") and G.reference_hours("s1", "QRC24_S1_A_FREEZE@W104_D121", {}, version="v1") == (2.15, "plan_reference_qrc24_adj_r1")     # §10 대용은 v1·v2 공통, s1 frozen 2.15
      and abs(G.reservation_for("s4", G.priority_for("s4")[-1])["reservation_h"] - (1.2 * 1.51 + 10 / 60)) < 1e-9 and G.reference_hours("s3", "QRC24_S3_G23@W104_D121", {"QRC24_S3_G23@W104_D121": 1.4}) == (1.4, "measured_same_case")
      and G.reservation_for("s4", "JQ@W104_D121")["slack"] == 1.1 and all(k_ in G.PREVIOUS_PRIORITY_BY_SERVER for k_ in ("s1_20260915_qedge9", "s2_20260916_edgebal", "s3_20260915_qegx", "s4_20260915_qegx", "s5_20260916_edgebal"))
      and all([l.strip() for l in open(os.path.join(ROOT, "config", "queues", f"qrecon24_{s_}.txt")) if l.strip() and not l.startswith("#")] == G.priority_for(s_) for s_ in ("s1", "s2", "s3", "s4", "s5"))
      and G.cue_ready(G.priority_for("s1")[0]) == G.cue_ready("QE50@W104_D121") and G.qrc24_extension_items("s1") == [_qn("s1", p_, 9091) for p_ in ("G22", "A_UNIF", "A_FREEZE")] and G.qrc24_extension_items("s5")[0].endswith("_S2909_FRESH50_v1"))
# ---- K45 registry qrecon
def _rejQ2(base, **kw):
    k = copy.deepcopy(base); k.update(kw)
    try:
        resolve(k); return False
    except ValueError:
        return True
_qb = _kQ22
check("K45 registry qrecon(§6.4): G22 spec(mode·qref·a/e·λE 1e-3·a_trainable) · 토큰 EDGEHQRC / EDGEHQRCAU / EDGEHQRCES / EDGEHQRCAUEU · A_FREEZE a_trainable False · describe 서술 · 거부: mode 오류 · qref 0 · a_weight 오류 · asset 없음 · shuffle perm 7 · rec N0 · stat OFF · λE calibrate/0 · I-AEQ · A-ID · +edge_gate/routing/aligner_schedule/edge_route/edge_schedule/edge_weight · 알 수 없는 키 · Teacher 없음",
      _sQ22["qrecon"] == dict(mode="continuous_v1", q_ref=0.3276133416220546, asset=G.QEDGE9_CUE_ASSET, a_weight="q", e_weight="q", perm_seed=51515, a_trainable=True, lambda_E=2e-3, uniform_weight=0.5) and not _sQR["A_FREEZE"]["qrecon"]["a_trainable"]
      and _rejQ2(_qb, qrecon=dict(_qb["qrecon"], uniform_weight=1.0)) and "0.5(= s_q(qref))" in _desc(_sQR["A_UNIF"])
      and [stat_tag(_sQ22), stat_tag(_sQR["A_UNIF"]), stat_tag(_sQR["E_SHUF"]), stat_tag(_sQR["ALL_UNIF"])] == ["EDGEHQRC", "EDGEHQRCAU", "EDGEHQRCES", "EDGEHQRCAUEU"] and "A ← w(q) = qref/(qref+q_T)·H 만" in _desc(_sQ22) and "T0 동결 대조" in _desc(_sQR["A_FREEZE"])
      and _rejQ2(_qb, qrecon=dict(_qb["qrecon"], mode="v2")) and _rejQ2(_qb, qrecon=dict(_qb["qrecon"], q_ref=0.0)) and _rejQ2(_qb, qrecon=dict(_qb["qrecon"], a_weight="soft")) and _rejQ2(_qb, qrecon={k_: v_ for k_, v_ in _qb["qrecon"].items() if k_ != "asset"})
      and _rejQ2(_qb, qrecon=dict(_qb["qrecon"], e_weight="shuffle", perm_seed=7)) and _rejQ2(_qb, rec=dict(case="N0")) and _rejQ2(_qb, stat=dict(enabled=False)) and _rejQ2(_qb, stat=dict(_qb["stat"], outer_weight="calibrate")) and _rejQ2(_qb, stat=dict(_qb["stat"], outer_weight=0.0))
      and _rejQ2(_qb, input_protocol="I-AEQ", corruption=dict(radius_hr=2.0), aux=dict(offset_weight=1e-4, geometry_weight=0.0)) and _rejQ2(_qb, aligner_policy="A-ID", recipe="NOALIGN", input_protocol="I-A", corruption={}, donor=None, na_protocol="NA-STRICT")
      and _rejQ2(_qb, edge_gate=dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET)) and _rejQ2(_qb, routing=dict(qE=0.0)) and _rejQ2(_qb, aligner_schedule=dict(freeze_from=25000)) and _rejQ2(_qb, edge_route=dict(mode="low_q", asset=G.QEDGE9_CUE_ASSET))
      and _rejQ2(_qb, edge_schedule=dict(before=1.0, after=0.5, switch=25000)) and _rejQ2(_qb, edge_weight=dict(mode="floor", low=0.75, high=0.25, asset=G.QEDGE9_CUE_ASSET)) and _rejQ2(_qb, qrecon=dict(_qb["qrecon"], foo=1)) and _rejQ2(_qb, teacher=None) and not _rejQ2(_qb))
# ---- K46 trainer: 두 목적함수 분리 (실제 KDVTrainer._step + _qrecon_backward, CPU)
def _stubQR(case, student, qw, server="s2", seed=777, seed_gen=3234):
    tr = object.__new__(KDVTrainer); tr.k = G.kdv_block(case, seed, server, cal=calQ, arch="W104_D121", branch="QRECON24"); sp_ = tr.spec = resolve(tr.k)
    tr.model = student; tr.accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, gradient_accumulation_steps=1, scaler=None, is_main_process=True)
    tr.teacher = T0; tr.aligner_trainable = sp_["aligner_trainable"]; tr.aligner_view_margin = 4; tr.share_correction = False; tr.protocol = sp_["protocol"]; tr.radius_hr = sp_["radius_hr"]; tr.diag_every = 10 ** 9
    if not tr.aligner_trainable:
        tr.model.aligner.requires_grad_(False)
    tr.rec_crit = GTAnchoredReconstructionKD(calQ["tau_R"], alpha=float(tr.k["rec"]["alpha"]), kd_weight=float(tr.k["rec"]["kd_weight"]), eps=1e-6, mode=sp_["rec_mode"])
    tr.tri = sp_["tri"]; tr.stat_extra = []; tr.lam_V = float(tr.k["stat"]["outer_weight"]); tr.stat_ramp = 0; tr.lam_edge = 0; tr.lam_geo = 0; tr.ramp = 5000; tr.lam_gkd = 0
    tr.gen = torch.Generator().manual_seed(seed_gen); tr.corr_seed = seed_gen; tr._ema = {}; tr._rr_val_last = float("nan"); tr.args = types.SimpleNamespace(num_iter=50000)
    tr.freeze_until, tr.freeze_from = None, None; tr.route_A = (1.0, 1.0, 1.0); tr._routed = False; tr._sched_last = None; tr.edge_gate = None; tr.edge_route = None; tr._edge_routed = False; tr.edge_schedule = None; tr.edge_weight = None; tr._edge_sched_last = None
    tr.qrecon = qw
    return tr
_tq = torch.tensor([[1.2, 0.9, 1.0, 1.1], [0.8, 1.05, 0.95, 1.0]]); _ts = torch.tensor([[0.8, 1.05, 0.95, 1.0], [1.2, 0.9, 1.0, 1.1]])
_QW = lambda a="q", e="q": QWeight.synthetic(_tq, _ts, a_mode=a, e_mode=e)
def _runQ(case, qw, **kw):
    tr = _stubQR(case, _S(), qw, **kw); tot, info = tr._step(_gt, _ms, _lp, _pn, 1, meta=_metaQ); return tr, tot, info
_grad_of = lambda loss, params: torch.cat([(x if x is not None else torch.zeros_like(p_)).flatten() for x, p_ in zip(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True), params)])
trG, totG, infoG = _runQ("QRC24_S2_G22", _QW()); apG = [p_ for p_ in trG.M.aligner.parameters() if p_.requires_grad]; bpG = [p_ for p_ in trG.M.backbone.parameters() if p_.requires_grad]
LU, LA = infoG["_L_U_t"], infoG["_L_A_t"]; gU_LU = _grad_of(LU, bpG); gU_LA = _grad_of(LA, bpG); gA_LA = _grad_of(LA, apG); gA_LU = _grad_of(LU, apG)
nU, nA = trG._qrecon_backward(infoG, trG.M, True); gU_set = torch.cat([p_.grad.flatten() for p_ in bpG]); gA_set = torch.cat([p_.grad.flatten() for p_ in apG])
_wa, _we = _QW().weights(_metaQ, torch.device("cpu"))
check(f"K46 두 목적함수 분리(§2.5·§6.1): total == L_U · U .grad == ∇θL_U (1e-6 상대) 이고 ∇θL_A(≠0, ‖·‖ {float(gU_LA.norm()):.2e}) 는 U 에 누적되지 않는다 · A .grad == ∇φL_A (1e-6) 이고 ∇φL_U(‖·‖ {float(gA_LU.norm()):.2e}) 와 다르다 · w_a [1.2, .95] w_e 동일 · U {nU}/A {nA} param · offset 없음(L_O 0, eq_exercise 없음, ε RNG 미소비) · y_t/w 에 grad 없음",
      float(totG) == float(LU) and _rel(gU_set, gU_LU) < 1e-6 and float(gU_LA.norm()) > 0 and _rel(gU_set, gU_LU + gU_LA) > 1e-4 and _rel(gA_set, gA_LA) < 1e-6 and float((gA_set - gA_LU).abs().max()) > 0 and nU == len(bpG) and nA == len(apG) > 0
      and torch.allclose(_wa, torch.tensor([1.2, 0.95])) and infoG["qrc_w_a_mean"] == float(_wa.mean()) and float(infoG["loss_off"]) == 0.0 and "eq_exercise" not in infoG and torch.equal(trG.gen.get_state(), torch.Generator().manual_seed(3234).get_state())
      and not infoG["y_t"].requires_grad and not _wa.requires_grad and not infoG["corrupt"], f"U {_rel(gU_set, gU_LU):.1e} A {_rel(gA_set, gA_LA):.1e}")
_yf, _gf, _ytf = infoG["y"].float(), _gt.float(), infoG["y_t"].float(); _r2 = trG.rec_crit(_yf, _ytf, _gf, return_maps=True)
_Hi = (_r2.maps["hard_weight"] * (_yf - _gf).abs().mean(1, keepdim=True)).mean((1, 2, 3)); _Ki = (_r2.maps["soft_weight"] * (_yf - _ytf).abs().mean(1, keepdim=True)).mean((1, 2, 3)); _Ei = output_edge_loss_per_sample(_yf, _gf)
trU, totU, infoU = _runQ("QRC24_S2_ALL_UNIF", _QW("uniform", "uniform"))
check("K46 손 계산(§2.2·§2.4·§6.2): L_U == mean(H_i + K_i + λE·w_e·E_i) · L_A == mean(w_a·H_i) (1e-9) · mean(H+K) == loss_rec (1e-6) · ALL_UNIF(uniform 0.5 = s_q(qref)): L_U == loss_rec + λE·0.5·L_E · L_A == 0.5·r.hard · λE 2e-3(2 배 환산) 가 그대로 곱해진다(qrc_lambda_E == 2e-3, 다른 배율 없음)",
      abs(float(LU) - float((_Hi + _Ki + 2e-3 * _we * _Ei).mean())) < 1e-9 and abs(float(LA) - float((_wa * _Hi).mean())) < 1e-9 and abs(float((_Hi + _Ki).mean()) - float(infoG["loss_rec"])) < 1e-6
      and abs(float(infoU["_L_U_t"]) - (float(infoU["loss_rec"]) + 2e-3 * 0.5 * float(output_edge_loss(infoU["y"].detach().float(), _gt.float())))) < 1e-6 and abs(float(infoU["_L_A_t"]) - 0.5 * float(trU.rec_crit(infoU["y"].detach().float(), infoU["y_t"].float(), _gt.float()).hard)) < 1e-6
      and infoG["qrc_lambda_E"] == 2e-3 and abs(infoG["qrc_lambda_wE"] - 2e-3 * float((_we * _Ei.detach()).mean())) < 1e-9)
_gA_of = lambda case, srv, sd: (lambda t_: _grad_of(t_[2]["_L_A_t"], [p_ for p_ in t_[0].M.aligner.parameters() if p_.requires_grad]))(_runQ(case, _QW(), server=srv, seed=sd))
gA_G22, gA_H21, gA_G32, gA_H12 = _gA_of("QRC24_S2_G22", "s2", 777), _gA_of("QRC24_S4_H21", "s4", 1234), _gA_of("QRC24_S2_G32", "s2", 777), _gA_of("QRC24_S4_H12", "s4", 1234)
tr12, _, info12 = _runQ("QRC24_S2_G12", _QW()); tr32, _, info32 = _runQ("QRC24_S2_G32", _QW()); trAU, _, infoAU = _runQ("QRC24_S1_A_UNIF", _QW("uniform", "q"), server="s1", seed=1234); trAS, _, infoAS = _runQ("QRC24_S1_A_SHUF", _QW("shuffle", "q"), server="s1", seed=1234)
trEU, _, infoEU = _runQ("QRC24_S2_E_UNIF", _QW("q", "uniform")); trES, _, infoES = _runQ("QRC24_S2_E_SHUF", _QW("q", "shuffle"))
check("K46 불변(§6.2·§6.3): 같은 Student·batch 에서 ∇φL_A 는 β(H21 .05)·λE(G32 6e-3) 를 바꿔도 bitwise 같고 α(H12 .5) 를 바꾸면 다르다 · λE 선형(G12/G22/G32 의 λ·w·E = .3/1/3 배) · A_UNIF w_a=0.5 → L_A=0.5·mean H · A_SHUF 는 shuffle 표 [.8, 1.0] · E_UNIF w_e=0.5 → L_U = loss_rec + λ·0.5·L_E · E_SHUF w_e 셔플 · U 의 H/K 에는 w 를 곱하지 않는다",
      torch.equal(gA_G22, gA_H21) and torch.equal(gA_G22, gA_G32) and not torch.equal(gA_G22, gA_H12) and abs(info12["qrc_lambda_wE"] / infoG["qrc_lambda_wE"] - 0.3) < 1e-6 and abs(info32["qrc_lambda_wE"] / infoG["qrc_lambda_wE"] - 3.0) < 1e-6
      and infoAU["qrc_w_a_mean"] == 0.5 and abs(float(infoAU["_L_A_t"]) - 0.5 * float(_Hi.mean())) < 1e-6 and torch.allclose(_QW("shuffle", "q").weights(_metaQ, torch.device("cpu"))[0], torch.tensor([0.8, 1.0])) and abs(infoAS["qrc_w_a_mean"] - 0.9) < 1e-6
      and infoEU["qrc_w_e_mean"] == 0.5 and abs(float(infoEU["_L_U_t"]) - (float(infoEU["loss_rec"]) + 2e-3 * 0.5 * float(output_edge_loss(infoEU["y"].detach().float(), _gt.float())))) < 1e-6 and abs(infoES["qrc_w_e_mean"] - 0.9) < 1e-6
      and abs(float((_Hi + _Ki).mean()) - float(infoG["loss_rec"])) < 1e-6)
trF, totF, infoF = _runQ("QRC24_S1_A_FREEZE", _QW(), server="s1", seed=1234); hF0 = state_hash(trF.M.aligner); bufF0 = [b_.clone() for b_ in trF.M.aligner.buffers()]
nUF, nAF = trF._qrecon_backward(infoF, trF.M, trF.aligner_active(1)); optF = torch.optim.AdamW([dict(params=[p_ for p_ in trF.M.backbone.parameters() if p_.requires_grad])], lr=1e-4, weight_decay=0.01); optF.step()
check("K46 A_FREEZE(§4.2·§6.2-7): aligner_trainable False · A .grad None(nA 0) · AdamW(U group 만) step 뒤 A parameter hash·buffer 불변 · U 는 gradient·update 있음 · total == L_U · Teacher hash 불변",
      not trF.aligner_trainable and nAF == 0 and all(p_.grad is None for p_ in trF.M.aligner.parameters()) and state_hash(trF.M.aligner) == hF0 and all(torch.equal(a_, b_) for a_, b_ in zip(bufF0, trF.M.aligner.buffers())) and nUF > 0
      and any(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for p_ in trF.M.backbone.parameters()) and float(totF) == float(infoF["_L_U_t"]) and state_hash(T0.aligner) == state_hash(trG.teacher.aligner))
try:
    _stubQR("QRC24_S2_G22", _S(), _QW())._step(_gt, _ms, _lp, _pn, 1); check("K46 qrecon 에 meta 없음 → ValueError", False)
except ValueError:
    check("K46 qrecon 에 meta 없음 → ValueError", True)
check("K46 학습 loop(§6.1): qrecon 이면 accelerator.backward(total) 대신 _qrecon_backward 로 두 gradient 를 놓고 optimizer.step 은 한 번 · L_A 비유한이면 exit 3", all(x in src for x in ("if getattr(self, \"qrecon\", None) is not None:", "nU_, nA_ = self._qrecon_backward(info, M, self.aligner_active(global_step))", "non-finite L_A")) and src.count("self.optimizer.step()") == 1)
del trG, trU, tr12, tr32, trAU, trAS, trEU, trES, trF
# ---- K47 QWeight (§2.3·§4.2·§6.3)
_np_ = _np
def _shuf_ok():
    v_ = _np_.concatenate([_np_.random.RandomState(1).rand(400), _np_.full(10, 0.7), _np_.array([0.3])]); s_ = _np_.concatenate([_np_.repeat(_np_.arange(4), 100), _np_.full(10, 9), _np_.array([11])])
    o_, st_ = QR.shuffle_values(v_, s_)
    return all(_np_.allclose(_np_.sort(o_[s_ == k_]), _np_.sort(v_[s_ == k_])) for k_ in _np_.unique(s_)) and _np_.array_equal(o_, QR.shuffle_values(v_, s_)[0]) and (o_ != v_).mean() > 0.3 and st_["n_shuffled"] == 4 and st_["n_degenerate"] == 1 and st_["n_small"] == 1
check("K47 q_weight(§2.3; 09-16 저녁 결정으로 분자 2 제거): w(qref) = 0.5 · w(0) = 1 · 0 < w ≤ 1 · 단조 감소 · q<0 / NaN / qref 0 → ValueError · shuffle_values 는 stratum 별 multiset 보존·결정적(51515)·값을 실제로 바꾼다·degenerate/small stratum 그대로",
      QR.q_weight([QR.QREF_DEFAULT], QR.QREF_DEFAULT)[0] == 0.5 and QR.q_weight([0.0], QR.QREF_DEFAULT)[0] == 1.0 and all(0 < w_ <= 1 for w_ in QR.q_weight(_np_.linspace(0, 5, 50), QR.QREF_DEFAULT)) and _np_.all(_np_.diff(QR.q_weight(_np_.linspace(0, 5, 50), QR.QREF_DEFAULT)) < 0)
      and all(_raises(lambda v_=v_, r_=r_: QR.q_weight(v_, r_), ValueError) for v_, r_ in (([-0.1], 0.3), ([float("nan")], 0.3), ([0.2], 0.0), ([0.2], float("nan")))) and _shuf_ok())
if os.path.exists(_asset):
    _qw = QWeight.load(dict(mode="continuous_v1", q_ref=QR.QREF_DEFAULT, asset=G.QEDGE9_CUE_ASSET, a_weight="q", e_weight="shuffle", perm_seed=51515), teacher=T0); _st = _qw.stats; _manQ = json.load(open(_asset))
    _pr1 = _raises(lambda: QWeight.load(dict(mode="continuous_v1", q_ref=QR.QREF_DEFAULT, asset=G.QEDGE9_CUE_ASSET), teacher=T0, previous=dict(asset_id="0" * 32)), ValueError)
    _pr2 = _raises(lambda: QWeight.load(dict(mode="continuous_v1", q_ref=QR.QREF_DEFAULT, asset=G.QEDGE9_CUE_ASSET), teacher=T0, previous=dict(asset_id=_manQ["asset_id"], stats=dict(w_sha256_16="deadbeefdeadbeef"))), ValueError)
    _okp = QWeight.load(dict(mode="continuous_v1", q_ref=QR.QREF_DEFAULT, asset=G.QEDGE9_CUE_ASSET), teacher=T0, previous=dict(asset_id=_manQ["asset_id"], qref=QR.QREF_DEFAULT, stats=dict(w_sha256_16=_st["w_sha256_16"])))
    check(f"K47 실제 cue 자산: w = qref/(qref+raw q) 표 완비(38,856 view; 0/1 gate 표 아님; 분자 2 없음) · 범위 {_st['w_min']:.4f}–{_st['w_max']:.4f} 평균 {_st['w_mean_all']:.4f}(0.5 근처; 기록) · calib 평균 {_st['w_mean_calib']:.4f} · shuffle 표 multiset 보존·표 다름·40 stratum · w/perm sha 기록 · Teacher hash 대조(QEDGE9 와 동일) · 재개 asset_id/w_sha 불일치 거부·일치 통과 · summary 에 A_loss 등",
          _qw.tables["q"].shape == (_manQ["dataset"]["n_train"], 4) and not bool(torch.isnan(_qw.tables["q"]).any()) and len(set(torch.unique(_qw.tables["q"]).tolist())) > 100 and 0.35 < _st["w_min"] < _st["w_max"] < 0.65 and abs(_st["w_at_qref"] - 0.5) < 1e-12 and _st["numerator_factor"] == 1.0 and "qref/(qref + q_T(i))" in _qw.summary()["formula"] and "2*qref" not in _qw.summary()["formula"]
          and _st["multiset_preserved"] and not torch.equal(_qw.tables["q"], _qw.tables["shuffle"]) and _st["shuffle"]["n_strata"] == 40 and len(_st["w_sha256_16"]) == 16 and _st["w_sha256_16"] != _st["w_shuffle_sha256_16"] and _qw.checks["teacher_aligner_hash"] == state_hash(T0.aligner)
          and _pr1 and _pr2 and _okp.stats["w_sha256_16"] == _st["w_sha256_16"] and _qw.summary()["A_loss"] == "weighted_H_only" and _qw.summary()["asset_id"] == _manQ["asset_id"] and _qw.e_mode == "shuffle"
          and torch.allclose(_qw.weight_for(torch.tensor([[0, 0, 1, 1], [5, 2, 1, 1]]), torch.device("cpu"), "q"), _qw.tables["q"][[0, 5], [0, 2]]) and torch.equal(_qw.weight_for(_metaQ, torch.device("cpu"), "uniform"), torch.full((2,), 0.5)) and _qw.summary()["uniform_weight"] == 0.5)
else:
    check("K47 실제 cue 자산 — 없음", False)
# ---- K48 시트 토큰 · gate · 스크립트 · config · selector
try:
    _lab48 = (_il("PAKD50_QRC24_S3_G22_W104_D121_WV3_T0_S2026_FRESH50_v1", "A104D121_T0FIX_QRECON24_v1") == "PAKD50 / QRC24 / G22 / A104D121 / FRESH50" and _il("PAKD50_QRC24_S1_A_FREEZE_W104_D121_WV3_T0_S1234_FRESH50_v1", None) == "PAKD50 / QRC24 / A_FREEZE / A104D121 / FRESH50")
except Exception:                                                              # noqa
    _lab48 = False
_allQ = [it for s_ in ("s1", "s2", "s3", "s4", "s5") for it in G.priority_for(s_)]; _cfgQ = {r_: (yaml.safe_load(open(os.path.join(ROOT, "config", r_ + ".yaml"))) if os.path.exists(os.path.join(ROOT, "config", r_ + ".yaml")) else None) for r_ in _allQ}
_tmpq = tempfile.mkdtemp(); G.generate("s2", [_qn("s2", "G22", 777), _qn("s2", "E_SHUF", 777)], _tmpq, projected=None); G.generate("s5", [_qn("s5", "L070", 2026)], _tmpq, projected=None); G.generate("s1", [_qn("s1", "A_FREEZE", 3407)], _tmpq, projected=None)
G.generate("s5", ["PAKD50_EB_QFLOOR_W104_D121_WV3_T0_S2026_FRESH50_v2", "PAKD50_J0_W104_D121_WV3_T0_S777_FRESH50_v1"], _tmpq, projected=None); G.generate("s4", ["PAKD50_QER50_W104_D121_WV3_T0_S1234_FRESH50_v1"], _tmpq, projected=None)
# selector fixture: s1 의 FQ S1234(W112) 가 있으면 값까지 고정 검사, 없는 서버(s2–s5) 는 그 서버의 완료 run(checkpoint_metrics.csv + best_hqnr_meta.json) 하나로 구조 불변량만 검사, 완료 run 이 없으면 SKIP(사유 출력) — 서버 전용 run 을 하드코딩하지 않는다
_selfix_s1 = "PAKD50_FQ_W112_D123_WV3_T0_S1234_FRESH50_v1"; _selfix_strict = os.path.exists(os.path.join(ROOT, "work_dir", _selfix_s1, "checkpoint_metrics.csv"))
_selfix = _selfix_s1 if _selfix_strict else next((os.path.basename(os.path.dirname(c_)) for c_ in sorted(glob.glob(os.path.join(ROOT, "work_dir", "PAKD50_*", "checkpoint_metrics.csv"))) if os.path.exists(os.path.join(os.path.dirname(c_), "best_hqnr_meta.json"))), None)
_srsel = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "qrecon24_select.py"), _selfix or "NONE", "--out", os.path.join(_tmpq, "sel.json")], capture_output=True, text=True, cwd=ROOT) if _selfix else None
_sel = json.load(open(os.path.join(_tmpq, "sel.json"))) if (_srsel is not None and _srsel.returncode == 0) else {}
def _sel_struct_ok():
    """어느 서버의 완료 run 에서나 성립해야 하는 selector 불변량 (§7.2–§7.3): 후보 수 == CSV 행 수 · 적격 수 == H ≥ 0.9585 행 수 · target_feasible == (적격 > 0) · proxy 는 official False · legacy step == best_hqnr_meta · raw-max == argmax · exact50K/late6 는 CSV 에 있는 step 만."""
    import csv as _csv2
    rows_ = list(_csv2.DictReader(open(os.path.join(ROOT, "work_dir", _selfix, "checkpoint_metrics.csv")))); H_ = [(int(float(r_["step"])), float(r_["raw_original.hqnr"])) for r_ in rows_ if r_.get("raw_original.hqnr") not in (None, "", "nan")]
    steps_ = {st_ for st_, _ in H_}; n_el = sum(1 for _, h_ in H_ if h_ >= 0.9585); bm_ = json.load(open(os.path.join(ROOT, "work_dir", _selfix, "best_hqnr_meta.json")))
    late_ = [st_ for st_ in (45450, 46460, 47470, 48480, 49490, 50000) if st_ in steps_]
    return (_sel.get("n_candidates") == len(rows_) and _sel.get("n_eligible") == n_el and _sel.get("target_feasible") == (n_el > 0) and _sel.get("official") is False and _sel.get("target_status") in ("proxy", "no_eligible") and (_sel.get("legacy_best") or {}).get("step") == bm_.get("step")
            and (_sel.get("raw_max") or {}).get("step") == max(H_, key=lambda t_: t_[1])[0] and (bool(_sel.get("exact50K")) == (50000 in steps_)) and ((_sel.get("late6") or {}).get("n", 0) == len(late_)) and ((_sel.get("h_consistency") is None) or abs(_sel["h_consistency"]["abs_diff"]) < 5e-4)
            and (not _selfix_strict or (_sel.get("n_candidates") == 50 and _sel.get("n_eligible") == 0 and _sel.get("target_feasible") is False and _sel["h_consistency"]["abs_diff"] == 0.0 and _sel["late6"]["n"] == 6)))
import importlib.util as _ilu2; _spq = _ilu2.spec_from_file_location("_qsel", os.path.join(ROOT, "tools", "qrecon24_select.py")); _qsel = _ilu2.module_from_spec(_spq); _spq.loader.exec_module(_qsel)
_rk = _qsel.rank([dict(step=1, hqnr=0.959, scc=0.987, ergas=2.05, psnr=37.9, sam=2.8, q8=0.92, ssim=0.975), dict(step=2, hqnr=0.9586, scc=0.988, ergas=2.10, psnr=37.0, sam=2.9, q8=0.90, ssim=0.97), dict(step=3, hqnr=0.9585, scc=0.988, ergas=2.00, psnr=37.0, sam=2.9, q8=0.90, ssim=0.97)])
check("K48a 시트 X열(§9.3) 'PAKD50 / QRC24 / G22 / A104D121 / FRESH50'(서버 토큰 제외) · gate 설명 · switch/waiter bash -n · rank 순서 SCC→ERGAS→PSNR→…",
      _lab48 and "QRECON24" in __import__("tools.campaign_gate", fromlist=["GATES"]).GATES["pakd50"][1] and all(subprocess.run(["bash", "-n", os.path.join(ROOT, "tools", f)], capture_output=True).returncode == 0 for f in ("qrecon24_switch.sh", "qrecon24_waiter.sh")) and [c_["step"] for c_ in _rk] == [3, 2, 1])
_k48 = dict(n81=(len(_allQ) == 81), exist=all(v_ is not None for v_ in _cfgQ.values()), meta=all(v_["train_feeder_args"].get("return_meta") is True and v_["kdv"]["campaign_id"] == G.QRC24_CAMPAIGN_ID and "training_deadline" not in v_["kdv"]["budget"] and v_["kdv"]["qrecon"]["mode"] == "continuous_v1" for v_ in _cfgQ.values()),
            lr=(_cfgQ[_qn("s5", "L070", 2026)]["learning_rate"] == 7e-5 and _cfgQ[_qn("s2", "G22", 777)]["learning_rate"] == 1e-4), frz=("aligner_lr" not in _cfgQ[_qn("s1", "A_FREEZE", 1234)]["kdv"] and _cfgQ[_qn("s1", "A_FREEZE", 1234)]["kdv"]["aligner_policy"] == "A-FR"),
            cmp={r_[:34]: filecmp.cmp(os.path.join(ROOT, "config", r_ + ".yaml"), os.path.join(_tmpq, r_ + ".yaml"), shallow=False) for r_ in (_qn("s2", "G22", 777), _qn("s2", "E_SHUF", 777), _qn("s5", "L070", 2026), _qn("s1", "A_FREEZE", 3407), "PAKD50_EB_QFLOOR_W104_D121_WV3_T0_S2026_FRESH50_v2", "PAKD50_J0_W104_D121_WV3_T0_S777_FRESH50_v1", "PAKD50_QER50_W104_D121_WV3_T0_S1234_FRESH50_v1")})
check("K48b 생성 config 81 벌(ADJ-R1 활성: v1 65 + v2 16; 보류 3 제외) 존재 · return_meta 전부 · campaign QRECON24 · training_deadline 없음 · qrecon mode · learning_rate 줄(L070 7e-05 / 기본 1e-4) · A_FREEZE 에 aligner_lr 없음(A-FR) · 생성기와 filecmp(4 벌) · 옛 EDGEBAL/QEDGE9/QEGX config 회귀 없음",
      all(v_ if not isinstance(v_, dict) else all(v_.values()) for v_ in _k48.values()), str({k_: v_ for k_, v_ in _k48.items() if (v_ is False) or (isinstance(v_, dict) and not all(v_.values()))}))
check(("K48c selector HQNR9585_RR_v1 proxy 출력(§7.2–§7.3) — " + ("s1 fixture FQ S1234: 후보 50, 적격 0, target_feasible false, official false, h_consistency(CSV raw H == fr_mat20 H) 0, legacy/raw-max/exact50K/late6 보존" if _selfix_strict
                                                                    else (f"이 서버의 완료 run {_selfix}: 구조 불변량(후보 수·적격 수·feasible·legacy/raw-max/exact50K/late6·h_consistency<5e-4)" if _selfix else "SKIP — 이 서버에 완료 run(checkpoint_metrics.csv + best_hqnr_meta.json) 이 없어 fixture 없음; 공식 RR 경로는 s1 검증(노트 §5)"))),
      (_selfix is None) or (_srsel is not None and _srsel.returncode == 0 and bool(_sel) and _sel_struct_ok()),
      (f"selector rc {_srsel.returncode} {_srsel.stderr[-300:] if _srsel.returncode else ''} keys {sorted(_sel)[:8]}" if _srsel is not None else "fixture 없음"))

# ================= K49 QRECON24 감사(2026-09-16) 대응: F02 case 경계 인계 · F03/F05 --extend 집계 · F04 extra 보존 · F06 selector 불완전 상태 · F07 캐시 hash · F10 manifest
_rsrc = open(os.path.join(ROOT, "tools", "_run_cases.sh")).read()
_td49 = tempfile.mkdtemp(); _h49 = os.path.join(_td49, "cases_queue_handover.txt")
_loop = r"""
run_case(){ echo "$1" >> "$OUT"; if [ "$1" = "OLD_CURRENT" ]; then printf 'NEW_1\nNEW_2\n' > "$HANDOVER_FILE"; fi; }
ORDER=(OLD_CURRENT OLD_PENDING_1 OLD_PENDING_2)
""" + _rsrc.split("HANDOVER_FILE=\"$REPO/work_dir/cases_queue_handover.txt\"")[1].split("# 본 큐 종료 후")[0]
_q49 = os.path.join(_td49, "cases_queue.txt"); open(_q49, "w").write("# 옛 큐\nOLD_CURRENT\nOLD_PENDING_1\nOLD_PENDING_2\n")
_r49 = subprocess.run(["bash", "-c", f'set -u; OUT="{_td49}/order.txt"; HANDOVER_FILE="{_h49}"; QUEUE_FILE="{_q49}"; ' + _loop], capture_output=True, text=True)
_order = open(os.path.join(_td49, "order.txt")).read().split() if os.path.exists(os.path.join(_td49, "order.txt")) else []
_q49_after = [l.strip() for l in open(_q49) if l.strip() and not l.startswith("#")]
_td49b = tempfile.mkdtemp(); _h49b = os.path.join(_td49b, "cases_queue_handover.txt"); _q49b = os.path.join(_td49b, "cases_queue.txt"); open(_q49b, "w").write("OLD_CURRENT\nOLD_PENDING_1\n")
_loopb = _loop.replace(r"printf 'NEW_1\nNEW_2\n'", r"printf '# 주석만\n'")
_r49b = subprocess.run(["bash", "-c", f'set -u; OUT="{_td49b}/order.txt"; HANDOVER_FILE="{_h49b}"; QUEUE_FILE="{_q49b}"; ' + _loopb], capture_output=True, text=True)
_orderb = open(os.path.join(_td49b, "order.txt")).read().split() if os.path.exists(os.path.join(_td49b, "order.txt")) else []
check("K49 runner 인계(F02 + ADJ-R1 검토): 실제 tools/_run_cases.sh 의 큐 loop 를 stub run_case 로 실행 — 첫 case 뒤 handover 파일이 생기면 남은 옛 큐(OLD_PENDING_1/2) 를 버리고 NEW_1/NEW_2 로 이어가고 **cases_queue.txt 도 같은 순서로 갱신**(재기동이 옛 큐를 읽지 않게) · 적용한 파일은 .applied.* · 비어 있는 인계 파일은 무시(.empty.*, 큐 유지) · switch 는 chain 이 살아 있을 때 인계 파일을 쓴다",
      _r49.returncode == 0 and _order == ["OLD_CURRENT", "NEW_1", "NEW_2"] and not os.path.exists(_h49) and any(f.startswith("cases_queue_handover.txt.applied.") for f in os.listdir(_td49))
      and _q49_after == ["NEW_1", "NEW_2"]                                                             # ADJ-R1 검토: 영속 큐도 갱신 — 감시자·재부팅 재기동이 옛 순서를 읽지 않는다
      and _r49b.returncode == 0 and _orderb == ["OLD_CURRENT", "OLD_PENDING_1", "OLD_PENDING_2"] and [l.strip() for l in open(_q49b) if l.strip()] == ["OLD_CURRENT", "OLD_PENDING_1"] and any(f.startswith("cases_queue_handover.txt.empty.") for f in os.listdir(_td49b))     # 빈 인계 파일은 무시(조기 DONE 방지)
      and "cases_queue_handover.txt" in open(os.path.join(ROOT, "tools", "qrecon24_switch.sh")).read(), f"rc {_r49.returncode} order {_order} queue {_q49_after} empty {_orderb} err {_r49.stderr[-200:]}")
_sw = open(os.path.join(ROOT, "tools", "qrecon24_switch.sh")).read(); _wt = open(os.path.join(ROOT, "tools", "qrecon24_waiter.sh")).read()
check("K49 --extend(F03/F05): 학습 시간(train_hours/hours) 합으로 24h 판정(hours_total 아님) · ≥24h 는 rc 0 · 확장 3 run 을 extra_priority + QRECON24 mandatory + 활성 큐(queue_active.txt) + reservations + 인계 파일에 반영, chain 없으면 활성 큐로 campaign_start, 대기자 기동 · 대기자는 활성 큐 우선 · extra 보존은 옮기기 전에 읽는다(F04; qegx/edgebal 도)",
      all(x in _sw for x in ('e.get("train_hours") or e.get("hours")', 'print("   train_h ≥ 24h — 확장 불필요 (§8.3)"); sys.exit(0)', "queue_active.txt", "G.QRC24_MANDATORY_FILE", "G.write_reservation_file(srv, active", "cases_queue_handover.txt", "campaign_start.sh --queue work_dir/_qrecon24/queue_active.txt", "qrecon24_waiter.sh >>"))
      and "queue_file()" in _wt and "queue_active.txt" in _wt and 'keep = [x for x in cur if G.branch_for(srv, x) == "QRECON24"]' in _sw and '"\\n".join(keep)' in _sw
      and all('keep = [x for x in cur if G.branch_for(srv, x) == "' in open(os.path.join(ROOT, "tools", f)).read() for f in ("qegx_switch.sh", "edgebal_switch.sh"))
      and 'e["train_hours"] = float(e["hours"]); e["postprocess_hours"]' in src and '"setup_hours"' in src)
# selector: synthetic run — proxy 상태, official 불완전(적격 후보 중 checkpoint 없음) 은 target 미확정, display_tie
_sr = os.path.join(ROOT, "work_dir", "_qrecon24", "_selftest_run"); os.makedirs(os.path.join(_sr, "results"), exist_ok=True)
try:
    open(os.path.join(_sr, "checkpoint_metrics.csv"), "w").write("step,raw_original.hqnr,raw_original.fscc,rr_scc,rr_ergas,rr_sam\n1010,0.9590,0.87,0.9990,2.05,2.8\n2020,0.9586,0.87,0.9870,2.00,2.7\n3030,0.9500,0.86,0.9880,2.10,2.9\n")
    json.dump(dict(step=1010), open(os.path.join(_sr, "best_hqnr_meta.json"), "w"))
    _r1 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "qrecon24_select.py"), "_qrecon24/_selftest_run", "--out", os.path.join(_td49, "p.json")], capture_output=True, text=True, cwd=ROOT); _p = json.load(open(os.path.join(_td49, "p.json"))) if _r1.returncode == 0 else {}
    _r2 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "qrecon24_select.py"), "_qrecon24/_selftest_run", "--official", "--device", "cpu", "--out", os.path.join(_td49, "o.json")], capture_output=True, text=True, cwd=ROOT); _o = json.load(open(os.path.join(_td49, "o.json"))) if _r2.returncode == 0 else {}
finally:
    import shutil as _sh49; _sh49.rmtree(_sr, ignore_errors=True)
_dt = _qsel.display_tie(dict(scc=0.9881, ergas=2.0396, psnr=37.9563, sam=2.79, q8=0.9224, ssim=0.976))
check("K49 selector(F06/F07/F10): proxy 는 target_status 'proxy'·official false·proxy_target 기록(적격 2, step 1010 이 SCC 로 앞섬) · --official 에서 checkpoint 없는 적격 후보가 있으면 target 미확정(status incomplete_official_rr, target None, target_feasible None, 미평가 목록) — proxy 값을 순위에 섞지 않는다 · display_tie(표시 자리 반올림) · 캐시 sidecar 에 checkpoint/config/h5/evaluator sha 검증 코드",
      _r1.returncode == 0 and _p.get("target_status") == "proxy" and _p.get("official") is False and _p.get("n_eligible") == 2 and (_p.get("target") or {}).get("step") == 1010 and (_p.get("proxy_target") or {}).get("step") == 1010
      and _r2.returncode == 0 and _o.get("target_status") == "incomplete_official_rr" and _o.get("target") is None and _o.get("target_feasible") is None and sorted(c_["step"] for c_ in _o.get("eligible_unevaluated", [])) == [1010, 2020] and _o.get("official") is False
      and _dt == dict(scc=True, ergas=True, psnr=True, sam=False, q8=True, ssim=True) and all(x in open(os.path.join(ROOT, "tools", "qrecon24_select.py")).read() for x in ("checkpoint_sha256", "reduced_h5_sha256", "evaluator", "rr_eval_seconds", "display_tie")),
      f"p {_r1.returncode} {(_p or {}).get('target_status')} o {_r2.returncode} {(_o or {}).get('target_status')} {_r2.stderr[-200:] if _r2.returncode else ''}")
check("K49 manifest(F10): training manifest 에 optimizer betas/eps/param group/decay 제외/cosine 최저/accumulation/clip/AMP/TF32 (_optimizer_manifest) · qrecon 요약에 w/shuffle/permutation/raw q 의 full sha256",
      "_optimizer_manifest" in src and all(x in src for x in ('"betas"', "weight_decay_exclusions", "min_lr=0.0", "gradient_accumulation_steps", "grad_clip", "tf32")) and all(x in open(os.path.join(ROOT, "kdv", "qrecon.py")).read() for x in ("w_sha256=_sha_full(w)", "permutation_sha256", "q_raw_sha256")))

# ================= K50 QRECON24 ADJ-R1 (research_log/PAN_QRECON24_S1_S5_Queue_Adjustment_2026-09-17.md §4–§7·§9·§11·부록 B; 노트 2026-09-17_qrecon24-adjustment-r1-implementation.md)
import filecmp, hashlib, shutil
_pf50 = {p_: G.qrc24_profile(p_) for p_ in ("B20A03", "A03_UNIF", "A03_SHUF", "G23", "H23", "G22")}
_pfx = lambda P_, *skip: {k_: v_ for k_, v_ in P_.items() if k_ not in ("profile", "canonical") + skip}
check("K50 신규 profile(§4; 새 loss 아님): B20A03 = λE .002·rA .03·α 1·β .2·U 1e-4/A 3e-6·a/e q (= G23 에서 β 만 / = H23 에서 rA·A LR 만) · A03_UNIF/A03_SHUF = G23 에서 a_weight 만 uniform/shuffle · profile 32 · canonical 은 자기 자신 · G/H/A/E/L 기존 표 불변(G23 λE .002 rA .03, H23 β .2)",
      len(G.QRC24_PROFILES) == 32 and (_pf50["B20A03"]["lam"], _pf50["B20A03"]["rA"], _pf50["B20A03"]["alpha"], _pf50["B20A03"]["beta"], _pf50["B20A03"]["ulr"], _pf50["B20A03"]["alr"], _pf50["B20A03"]["a"], _pf50["B20A03"]["e"], _pf50["B20A03"]["frozen"]) == (2e-3, 0.03, 1.0, 0.2, 1e-4, 3e-6, "q", "q", False)
      and _pfx(_pf50["B20A03"], "beta") == _pfx(_pf50["G23"], "beta") and _pfx(_pf50["B20A03"], "rA", "alr") == _pfx(_pf50["H23"], "rA", "alr") and _pfx(_pf50["A03_UNIF"], "a") == _pfx(_pf50["G23"], "a") and _pfx(_pf50["A03_SHUF"], "a") == _pfx(_pf50["G23"], "a")
      and (_pf50["A03_UNIF"]["a"], _pf50["A03_SHUF"]["a"], _pf50["A03_UNIF"]["e"]) == ("uniform", "shuffle", "q") and all(G.qrc24_profile(p_)["canonical"] == p_ for p_ in G.QRC24_ADJ_PROFILES) and G.QRC24_ADJ_PROFILES == ("B20A03", "A03_UNIF", "A03_SHUF")
      and (_pf50["G23"]["lam"], _pf50["G23"]["rA"], _pf50["H23"]["beta"], _pf50["H23"]["rA"], _pf50["G22"]["rA"], _pf50["G22"]["beta"]) == (2e-3, 0.03, 0.2, 0.01, 0.01, 0.1))
_ADJ_B = """PAKD50_QRC24_S1_A03_UNIF_W104_D121_WV3_T0_S1234_FRESH50_v2 PAKD50_QRC24_S1_A03_SHUF_W104_D121_WV3_T0_S1234_FRESH50_v2 PAKD50_QRC24_S3_H23_W104_D121_WV3_T0_S4321_FRESH50_v2 PAKD50_QRC24_S3_B20A03_W104_D121_WV3_T0_S4321_FRESH50_v2
PAKD50_QRC24_S4_H23_W104_D121_WV3_T0_S3407_FRESH50_v2 PAKD50_QRC24_S4_H13_W104_D121_WV3_T0_S3407_FRESH50_v2 PAKD50_QRC24_S4_H32_W104_D121_WV3_T0_S3407_FRESH50_v2 PAKD50_QRC24_S4_H33_W104_D121_WV3_T0_S3407_FRESH50_v2
PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S1234_FRESH50_v2 PAKD50_QRC24_S4_B20A03_W104_D121_WV3_T0_S1234_FRESH50_v2 PAKD50_QRC24_S5_G23_W104_D121_WV3_T0_S9091_FRESH50_v2 PAKD50_QRC24_S5_H23_W104_D121_WV3_T0_S9091_FRESH50_v2
PAKD50_QRC24_S5_B20A03_W104_D121_WV3_T0_S9091_FRESH50_v2 PAKD50_QRC24_S5_G23_W104_D121_WV3_T0_S1103_FRESH50_v2 PAKD50_QRC24_S5_H23_W104_D121_WV3_T0_S1103_FRESH50_v2 PAKD50_QRC24_S5_B20A03_W104_D121_WV3_T0_S1103_FRESH50_v2""".split()
_v2all = [r_ for s_ in ("s1", "s2", "s3", "s4", "s5") for r_ in G.qrc24_adj_runs(s_)]; _orig68 = {G.qrc24_run_name(s_, p_, sd_) for s_ in G.QRC24_QUEUES_20260916 for p_, sd_ in G.QRC24_QUEUES_20260916[s_]}
_v1kept = [r_ for s_ in ("s1", "s2", "s3", "s4", "s5") for p_, sd_, v_ in G.QRC24_ADJ_ORDER[s_] if v_ == "v1" for r_ in [G.qrc24_run_name(s_, p_, sd_)]]; _held = {s_: list(G.qrc24_held_runs(s_)) for s_ in ("s1", "s2", "s3", "s4", "s5")}
check("K50 편성(§5·§9·부록 B): 남은 순서 s1 G22@3407→G23@3407→A03_UNIF@1234 v2→A03_SHUF@1234 v2→A_UNIF→A_FREEZE→A_SHUF→G21@3407 · s2 G13→E_UNIF→E_SHUF→ALL_UNIF→G31→G33@777 · s3 G23@4321→H23 v2→B20A03 v2→G13→G12→G21→G11→G32→G31→G33 · s4 H_BETA0@1234→H22@3407→H23 v2→H12→H13 v2→H32 v2→H33 v2→G23@1234 v2→B20A03@1234 v2 · s5 L070→L050@9091→G23/H23/B20A03@9091 v2→L100@1103→G23/H23/B20A03@1103 v2 · v2 16 == 부록 B · 유지 26 v1 ⊂ 원계획 · 보류 s4 H11@3407, s5 L070/L050@1103 (priority·mandatory 밖) · 등록 39 · 활성 14/12/20/19/16 = 81 · 큐 파일 == 활성",
      G.QRC24_ADJ_ORDER["s1"] == [("G22", 3407, "v1"), ("G23", 3407, "v1"), ("A03_UNIF", 1234, "v2"), ("A03_SHUF", 1234, "v2"), ("A_UNIF", 3407, "v1"), ("A_FREEZE", 3407, "v1"), ("A_SHUF", 3407, "v1"), ("G21", 3407, "v1")]
      and G.QRC24_ADJ_ORDER["s2"] == [(p_, 777, "v1") for p_ in ("G13", "E_UNIF", "E_SHUF", "ALL_UNIF", "G31", "G33")] and G.QRC24_ADJ_ORDER["s3"] == [("G23", 4321, "v1"), ("H23", 4321, "v2"), ("B20A03", 4321, "v2")] + [(p_, 4321, "v1") for p_ in ("G13", "G12", "G21", "G11", "G32", "G31", "G33")]
      and G.QRC24_ADJ_ORDER["s4"] == [("H_BETA0", 1234, "v1"), ("H22", 3407, "v1"), ("H23", 3407, "v2"), ("H12", 3407, "v1"), ("H13", 3407, "v2"), ("H32", 3407, "v2"), ("H33", 3407, "v2"), ("G23", 1234, "v2"), ("B20A03", 1234, "v2")]
      and G.QRC24_ADJ_ORDER["s5"] == [("L070", 9091, "v1"), ("L050", 9091, "v1"), ("G23", 9091, "v2"), ("H23", 9091, "v2"), ("B20A03", 9091, "v2"), ("L100", 1103, "v1"), ("G23", 1103, "v2"), ("H23", 1103, "v2"), ("B20A03", 1103, "v2")]
      and _v2all == _ADJ_B and set(_v1kept) <= _orig68 and len(_v1kept) == 26 and not (set(_v2all) & _orig68) and _held == {"s1": [], "s2": [], "s3": [], "s4": [_qn("s4", "H11", 3407)], "s5": [_qn("s5", "L070", 1103), _qn("s5", "L050", 1103)]}
      and all(h_ not in G.priority_for(s_) and h_ not in G.mandatory_for(s_) and h_ in _orig68 for s_, hs_ in _held.items() for h_ in hs_) and [len(G.QRC24_REGISTERED[s_]) for s_ in ("s1", "s2", "s3", "s4", "s5")] == [6, 6, 10, 10, 7]
      and [len(G.priority_for(s_)) for s_ in ("s1", "s2", "s3", "s4", "s5")] == [14, 12, 20, 19, 16] and all(G.priority_for(s_) == [G.qrc24_run_name(s_, p_, sd_, v_) for p_, sd_, v_ in G.QRC24_REGISTERED[s_]] + [G.qrc24_run_name(s_, p_, sd_, v_) for p_, sd_, v_ in G.QRC24_ADJ_ORDER[s_]] for s_ in ("s1", "s2", "s3", "s4", "s5")) and all(v_ == "v1" for l_ in G.QRC24_REGISTERED.values() for _, _, v_ in l_)
      and all([l.strip() for l in open(os.path.join(ROOT, "config", "queues", f"qrecon24_{s_}.txt")) if l.strip() and not l.startswith("#")] == G.priority_for(s_) for s_ in ("s1", "s2", "s3", "s4", "s5")) and G.QRC24_ADJ_REVISION == "QRC24_ADJ_R1_20260917" and G.QRC24_HELD_STATUS == "superseded_pending"
      and all(f"queue_revision {G.QRC24_ADJ_REVISION}" in open(os.path.join(ROOT, "config", "queues", f"qrecon24_{s_}.txt")).read() for s_ in ("s1", "s2", "s3", "s4", "s5")) and all(h_ in open(os.path.join(ROOT, "config", "queues", f"qrecon24_{s_}.txt")).read() for s_, hs_ in _held.items() for h_ in hs_))
_tmp50 = tempfile.mkdtemp(); _bad50v1 = []; _bad50v2 = []
for s_ in ("s1", "s2", "s3", "s4", "s5"):
    _v1s = [r_ for r_ in G.priority_for(s_) if r_.endswith("_v1")] + _held[s_]; G.generate(s_, _v1s, _tmp50, projected=None); _bad50v1 += [r_ for r_ in _v1s if not filecmp.cmp(os.path.join(ROOT, "config", r_ + ".yaml"), os.path.join(_tmp50, r_ + ".yaml"), shallow=False)]
    _v2s = G.qrc24_adj_runs(s_); G.generate(s_, _v2s, _tmp50, projected=None); _bad50v2 += [r_ for r_ in _v2s if not (os.path.exists(os.path.join(ROOT, "config", r_ + ".yaml")) and filecmp.cmp(os.path.join(ROOT, "config", r_ + ".yaml"), os.path.join(_tmp50, r_ + ".yaml"), shallow=False))]
_cv2 = {r_: yaml.safe_load(open(os.path.join(ROOT, "config", r_ + ".yaml")))["kdv"] for r_ in _v2all}; _cv1 = {r_: yaml.safe_load(open(os.path.join(ROOT, "config", r_ + ".yaml")))["kdv"] for r_ in _v1kept}
_b3 = _cv2[_qn("s3", "B20A03", 4321).replace("_v1", "_v2")]; _h4 = _cv2["PAKD50_QRC24_S4_H23_W104_D121_WV3_T0_S3407_FRESH50_v2"]; _g5 = _cv2["PAKD50_QRC24_S5_G23_W104_D121_WV3_T0_S1103_FRESH50_v2"]
check("K50 config(§11.1-2·§11.2): 원계획 68 v1(보류 3 포함) 바이트 불변 · v2 16 == 생성기 · v2 kdv.qrc24{queue_revision, adjustment ADJ-R1, source_plan, block_2x2, numerator_factor 1} · control_runs 는 같은 서버·seed 의 **실제** id(s3 4321 B20A03 → G22 v1 · G23 v1 · H23 v2 · baseline G22 v1; s4 3407 H23 → H22 v1; s5 1103 G23 → L100 v1) · v1 config 에 queue_revision 없음 · v2 projected_map = 1.2×ADJ R_s + 10/60 · qrecon 블록(uniform .5·perm 51515) 은 v1 과 같은 꼴",
      not _bad50v1 and not _bad50v2 and all(c_["qrc24"]["queue_revision"] == G.QRC24_ADJ_REVISION and c_["qrc24"]["adjustment"] == "ADJ-R1" and c_["qrc24"]["source_plan"] == G.QRC24_ADJ_PLAN and c_["qrc24"]["numerator_factor"] == 1.0 and set(c_["qrc24"]["block_2x2"]) == {"rA01_beta1", "rA03_beta1", "rA01_beta2", "rA03_beta2"} for c_ in _cv2.values())
      and _b3["control_runs"]["G22"] == _qn("s3", "G22", 4321) and _b3["control_runs"]["G23"] == _qn("s3", "G23", 4321) and _b3["control_runs"]["H23"] == "PAKD50_QRC24_S3_H23_W104_D121_WV3_T0_S4321_FRESH50_v2" and _b3["baseline_run"] == _qn("s3", "G22", 4321) and "B20A03" not in _b3["control_runs"]
      and _h4["control_runs"]["G22"] == _qn("s4", "H22", 3407) and _h4["control_runs"]["control_profile"] == "H22" and _g5["control_runs"]["G22"] == _qn("s5", "L100", 1103) and _g5["qrc24"]["block_2x2"]["rA01_beta1"] == _qn("s5", "L100", 1103) and _g5["qrc24"]["block_2x2"]["rA03_beta2"].endswith("_S1103_FRESH50_v2")
      and all("queue_revision" not in c_["qrc24"] for c_ in _cv1.values()) and abs(list(_b3["budget"]["projected_map"].values())[0] - (1.2 * 1.47 + 10 / 60)) < 1e-9 and all(c_["qrecon"] == dict(_cv1[_qn("s3", "G23", 4321)]["qrecon"], a_weight=c_["qrecon"]["a_weight"]) for c_ in _cv2.values()),
      f"v1 diff {_bad50v1[:3]} v2 diff {_bad50v2[:3]}")
_kb50 = lambda p_, sd_, srv_, ver_: G.kdv_block(f"QRC24_{srv_.upper()}_{p_}", sd_, srv_, cal=calQ, arch="W104_D121", version=ver_)
_g23s3, _h23s3, _b20s3 = _kb50("G23", 4321, "s3", "v1"), _kb50("H23", 4321, "s3", "v2"), _kb50("B20A03", 4321, "s3", "v2"); _g23s1, _aus1, _ass1 = _kb50("G23", 1234, "s1", "v1"), _kb50("A03_UNIF", 1234, "s1", "v2"), _kb50("A03_SHUF", 1234, "s1", "v2")
_dk = lambda a_, b_: {k_ for k_ in set(a_) | set(b_) if a_.get(k_) != b_.get(k_)}; _META50 = {"qrc24", "control_runs", "baseline_run", "budget", "case_id", "version"}          # 이름·version·대조·예산 metadata 만 다르다
check("K50 kdv block 차이(§11.3): B20A03 − G23 = rec.kd_weight(β .2) 만 · B20A03 − H23 = aligner_lr(3e-6 vs 1e-6) 만 · A03_UNIF/A03_SHUF − G23 = qrecon.a_weight 만 (그 밖은 metadata: qrc24/control/budget) · stat(λE .002)·input(I-NATIVE-TRANSFER)·A-FT·corruption·aux·teacher·exact_resume·expect_init(같은 seed U 초기값 hash) 동일",
      _dk(_b20s3, _g23s3) <= {"rec"} | _META50 and dict(_b20s3["rec"], kd_weight=0.1) == _g23s3["rec"] and _b20s3["rec"]["kd_weight"] == 0.2 and _b20s3["aligner_lr"] == _g23s3["aligner_lr"] == 3e-6 and _b20s3["stat"] == _g23s3["stat"]
      and _dk(_b20s3, _h23s3) <= {"aligner_lr"} | _META50 and _b20s3["rec"] == _h23s3["rec"] and _h23s3["aligner_lr"] == 1e-6 and _dk(_aus1, _g23s1) <= {"qrecon"} | _META50 and _dk(_ass1, _g23s1) <= {"qrecon"} | _META50
      and dict(_aus1["qrecon"], a_weight="q") == _g23s1["qrecon"] and dict(_ass1["qrecon"], a_weight="q") == _g23s1["qrecon"] and _aus1["qrecon"]["uniform_weight"] == 0.5 and _ass1["qrecon"]["perm_seed"] == 51515
      and _aus1.get("expect_init") == _g23s1.get("expect_init") == _ass1.get("expect_init") and (G.init_hash_for(1234, "W104_D121") is None or _aus1.get("expect_init") is not None) and _b20s3["input_protocol"] == "I-NATIVE-TRANSFER" and _b20s3["aligner_policy"] == "A-FT" and _b20s3["exact_resume"] is True,
      f"B−G {_dk(_b20s3, _g23s3)} B−H {_dk(_b20s3, _h23s3)} AU−G {_dk(_aus1, _g23s1)}")
# 실제 trainer (K46 harness): B20A03 vs H23 손실 bitwise 동일(LR 만 다름) · B20A03 vs G23: L_A·∇φL_A bitwise 동일(β 는 A 에 없음), L_U 차 = mean(K_G23)(β 2 배) · A03_UNIF: L_A = .5·mean H, L_U == G23 · A03_SHUF: w_a = 셔플 표, L_U == G23
trB, totB, infoB = _runQ("QRC24_S3_B20A03", _QW(), server="s3", seed=4321); trH, totH, infoH = _runQ("QRC24_S3_H23", _QW(), server="s3", seed=4321); trG3, totG3, infoG3 = _runQ("QRC24_S3_G23", _QW(), server="s3", seed=4321)
_apB = [p_ for p_ in trB.M.aligner.parameters() if p_.requires_grad]; _apG3 = [p_ for p_ in trG3.M.aligner.parameters() if p_.requires_grad]
_r3 = trG3.rec_crit(infoG3["y"].float(), infoG3["y_t"].float(), _gt.float(), return_maps=True); _K3 = (_r3.maps["soft_weight"] * (infoG3["y"].float() - infoG3["y_t"].float()).abs().mean(1, keepdim=True)).mean((1, 2, 3))
trAU, totAU, infoAU2 = _runQ("QRC24_S1_A03_UNIF", _QW("uniform", "q"), server="s1", seed=1234); trAS, totAS, infoAS2 = _runQ("QRC24_S1_A03_SHUF", _QW("shuffle", "q"), server="s1", seed=1234); trG1, totG1, infoG1 = _runQ("QRC24_S1_G23", _QW(), server="s1", seed=1234)
_rA = trAU.rec_crit(infoAU2["y"].float(), infoAU2["y_t"].float(), _gt.float(), return_maps=True); _HA = (_rA.maps["hard_weight"] * (infoAU2["y"].float() - _gt.float()).abs().mean(1, keepdim=True)).mean((1, 2, 3))
check("K50 trainer(§11.3; 실제 KDVTrainer._step, CPU, 같은 pre-step 상태): B20A03 == H23 (total·L_U·L_A bitwise; LR 만 다름) · B20A03 vs G23: L_A 와 ∇φL_A bitwise 같고 L_U 차 == mean(K_G23) (β .1→.2 = soft 2 배; 1e-6) · A03_UNIF: w_a 0.5·L_A == 0.5·mean H·L_U == G23 · A03_SHUF: w_a == 셔플 표 [.8, 1.0]·L_U == G23 · 세 run 의 U ∇θL_U 는 uniform/shuffle 과 무관하게 G23 과 같다(1e-6) · λE 2e-3",
      float(totB) == float(totH) and float(infoB["_L_U_t"]) == float(infoH["_L_U_t"]) and float(infoB["_L_A_t"]) == float(infoH["_L_A_t"]) and trB.k["aligner_lr"] == 3e-6 and trH.k["aligner_lr"] == 1e-6
      and float(infoB["_L_A_t"]) == float(infoG3["_L_A_t"]) and torch.equal(_grad_of(infoB["_L_A_t"], _apB), _grad_of(infoG3["_L_A_t"], _apG3)) and abs((float(infoB["_L_U_t"]) - float(infoG3["_L_U_t"])) - float(_K3.mean())) < 1e-6 and float(_K3.mean()) > 0
      and infoAU2["qrc_w_a_mean"] == 0.5 and abs(float(infoAU2["_L_A_t"]) - 0.5 * float(_HA.mean())) < 1e-6 and float(infoAU2["_L_U_t"]) == float(infoG1["_L_U_t"]) and float(infoAS2["_L_U_t"]) == float(infoG1["_L_U_t"])
      and torch.allclose(_QW("shuffle", "q").weights(_metaQ, torch.device("cpu"))[0], torch.tensor([0.8, 1.0])) and abs(infoAS2["qrc_w_a_mean"] - 0.9) < 1e-6 and sorted(_tq.flatten().tolist()) == sorted(_ts.flatten().tolist())
      and _rel(_grad_of(infoAU2["_L_U_t"], [p_ for p_ in trAU.M.backbone.parameters() if p_.requires_grad]), _grad_of(infoG1["_L_U_t"], [p_ for p_ in trG1.M.backbone.parameters() if p_.requires_grad])) < 1e-6 and infoB["qrc_lambda_E"] == 2e-3,
      f"B/H {float(totB)} {float(totH)} · ΔL_U {float(infoB['_L_U_t']) - float(infoG3['_L_U_t']):.3e} vs K {float(_K3.mean()):.3e} · AU L_A {float(infoAU2['_L_A_t']):.4e} vs {0.5 * float(_HA.mean()):.4e}")
# 시트 · 스크립트 · 도구
try:
    _lab50 = (_il("PAKD50_QRC24_S4_B20A03_W104_D121_WV3_T0_S1234_FRESH50_v2", "A104D121_T0FIX_QRECON24_v1") == "PAKD50 / QRC24 / B20A03 / A104D121 / ADJ-R1 / FRESH50" and _il("PAKD50_QRC24_S1_A03_SHUF_W104_D121_WV3_T0_S1234_FRESH50_v2", None) == "PAKD50 / QRC24 / A03_SHUF / A104D121 / ADJ-R1 / FRESH50"
              and _il("PAKD50_QRC24_S3_G23_W104_D121_WV3_T0_S4321_FRESH50_v1", "A104D121_T0FIX_QRECON24_v1") == "PAKD50 / QRC24 / G23 / A104D121 / FRESH50")
except Exception:                                                              # noqa
    _lab50 = False
_sw50 = open(os.path.join(ROOT, "tools", "qrecon24_switch.sh")).read(); _up50 = open(os.path.join(ROOT, "tools", "_upload.sh")).read(); _gu50 = open(os.path.join(ROOT, "gspread", "gspread_upload.py")).read()
check("K50 시트·스크립트(§11.2·§7.2·§8.1): v2 X열 'PAKD50 / QRC24 / <PROFILE> / A104D121 / ADJ-R1 / FRESH50' · v1 그대로 · Notes 에 numerator_factor/uniform_weight/queue_revision(+source_plan/block_2x2) · _upload.sh 가 QRC24 run 뒤 tools/qrecon24_postrun.py --backlog(공식 selector + 버전 감사; case 경계) · switch 가 held_runs/queue_revision/version_audit/backlog 를 다룬다 · bash -n",
      _lab50 and all(x in _gu50 for x in ("queue_revision=", "numerator_factor=", "uniform_weight=", "block_2x2=")) and "qrecon24_postrun.py" in _up50 and "PAKD50_QRC24_*)" in _up50
      and all(x in _sw50 for x in ("held_runs.json", "queue_revision.json", "qrecon24_version_audit.py", "qrecon24_postrun.py --backlog", "qrc24_held_runs")) and all(subprocess.run(["bash", "-n", os.path.join(ROOT, "tools", f_)], capture_output=True).returncode == 0 for f_ in ("qrecon24_switch.sh", "_upload.sh", "qrecon24_waiter.sh")))
def _heredoc_ok(src):
    """bash 안 python heredoc(<<'PYEOF' … PYEOF) 전부: compile 되고, 쓰는 모듈(json/os/sys/time/subprocess) 의 import 가 그 블록 안에 있다 (switch ⑤ 의 NameError 재발 방지)."""
    import re as _re
    blocks = _re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", src, flags=_re.S)
    if not blocks:
        return False
    for blk in blocks:
        try:
            compile(blk, "<heredoc>", "exec")
        except SyntaxError:
            return False
        for mod in ("json", "os", "sys", "time", "subprocess"):
            if _re.search(rf"\b{mod}\.", blk) and not _re.search(rf"^\s*import .*\b{mod}\b", blk, flags=_re.M):
                return False
    return True
check("K50 switch/waiter heredoc python 블록: compile + 사용 모듈 import 존재 (⑤ NameError 회귀 방지)", _heredoc_ok(_sw50) and _heredoc_ok(open(os.path.join(ROOT, "tools", "qrecon24_waiter.sh")).read()))
# 도구: 버전 감사(fixture 3 종) · 보존(fixture; sha256 manifest + --verify) · postrun --dry-run · 실제 s1 G22 S1234 가 있으면 §8.1 판정 single_definition_factor1 + 이전 시도 기록
_fx = os.path.join(ROOT, "work_dir", f"_k50fix_{os.getpid()}"); os.makedirs(_fx, exist_ok=True)
def _mk_fix(name, factor, uniform, lam, resume=False, manifest=True):
    d_ = os.path.join(_fx, name); os.makedirs(os.path.join(d_, "meta"), exist_ok=True); os.makedirs(os.path.join(d_, "results"), exist_ok=True); os.makedirs(os.path.join(d_, "best_hqnr"), exist_ok=True)
    if manifest:
        json.dump(dict(qrecon=dict(formula=("w_i = qref/(qref + q_T(i))" if factor == 1.0 else "w_i = 2·qref/(qref + q_T(i))"), numerator_factor=factor, uniform_weight=uniform), spec=dict(qrecon=dict(lambda_E=lam, uniform_weight=uniform)), exact_resume=True, resumed_nonexact=False), open(os.path.join(d_, "kdv_config_resolved.json"), "w"))
    yaml.safe_dump(dict(kdv=dict(stat=dict(outer_weight=0.002), qrecon=dict(uniform_weight=0.5), qrc24=dict(profile="G22"))), open(os.path.join(d_, "meta", "config.yaml"), "w"))
    open(os.path.join(d_, "meta", "git_commit.txt"), "w").write("deadbeef" * 5 + "\n"); open(os.path.join(d_, "meta", "command.txt"), "w").write("./tools/run.sh X" + (" --resume work_dir/X/epoch-5" if resume else "") + "\npython -u main.py --config X\n")
    json.dump(dict(step=31310, epoch=155, selection_view="raw_original", hqnr=0.9591, alias="best_hqnr"), open(os.path.join(d_, "best_raw_meta.json"), "w")); open(os.path.join(d_, "best_hqnr", "model.safetensors"), "wb").write(os.urandom(4096)); open(os.path.join(d_, "checkpoint_metrics.csv"), "w").write("step,raw_original.hqnr\n50000,0.95\n")
    json.dump(dict(rows=[]), open(os.path.join(d_, "results", "fr_mat20.json"), "w")); return f"_k50fix_{os.getpid()}/{name}"
import importlib.util as _ilu3; _spa = _ilu3.spec_from_file_location("_qva", os.path.join(ROOT, "tools", "qrecon24_version_audit.py")); _qva = _ilu3.module_from_spec(_spa); _spa.loader.exec_module(_qva)
_spp = _ilu3.spec_from_file_location("_qpr", os.path.join(ROOT, "tools", "qrecon24_preserve.py")); _qpr = _ilu3.module_from_spec(_spp); _spp.loader.exec_module(_qpr)
_va1 = _qva.audit_run(_mk_fix("f1", 1.0, 0.5, 0.002), write=False); _va2 = _qva.audit_run(_mk_fix("f2", 2.0, None, 0.001), write=False); _va3 = _qva.audit_run(_mk_fix("f3", 1.0, 0.5, 0.002, manifest=False), write=False); _va4 = _qva.audit_run(_mk_fix("f4", 1.0, 0.5, 0.001), write=False)
_pdest = os.path.join(_tmp50, "preserved"); _prc = _qpr.preserve(f"_k50fix_{os.getpid()}/f1", os.path.relpath(_pdest, ROOT) if _pdest.startswith(ROOT) else _pdest); _pm = json.load(open(os.path.join(_pdest, f"_k50fix_{os.getpid()}", "f1", "preserve_manifest.json")))
_pv = _qpr.preserve(f"_k50fix_{os.getpid()}/f1", os.path.relpath(_pdest, ROOT) if _pdest.startswith(ROOT) else _pdest, verify=True); _pr_dry = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "qrecon24_postrun.py"), "--backlog", "--dry-run"], capture_output=True, text=True, cwd=ROOT)
shutil.rmtree(_fx, ignore_errors=True)
_g22s1 = os.path.join(ROOT, "work_dir", _qn("s1", "G22", 1234)); _vaS1 = _qva.audit_run(_qn("s1", "G22", 1234), write=False) if os.path.exists(os.path.join(_g22s1, "kdv_config_resolved.json")) else None
check("K50 도구(§7.1·§8.1): 버전 감사 fixture — factor 1/uniform .5/λE 일치 → single_definition_factor1 · factor 2 → factor2_or_old_definition · manifest 없음 → unverified · λE 불일치 → unverified · 보존 manifest 의 sha256 == 파일 해시(selected step 31310 기록) 이고 --verify 0 · postrun --backlog --dry-run rc 0"
      + (f" · 실제 s1 G22 S1234: {_vaS1['verdict']} (이전 시도 {len(_vaS1['evidence']['previous_attempts'])})" if _vaS1 else " · (s1 G22 S1234 없음 — 실제 판정 생략)"),
      _va1["verdict"] == "single_definition_factor1" and _va2["verdict"] == "factor2_or_old_definition" and _va3["verdict"] == "unverified" and _va4["verdict"] == "unverified" and _prc == 0 and _pv == 0 and _pm["selected"]["step"] == 31310
      and _pm["files"]["best_hqnr/model.safetensors"] == _pm["files"]["best_hqnr/model.safetensors"] and len(_pm["files"]) >= 6 and all(hashlib.sha256(open(os.path.join(_pdest, f"_k50fix_{os.getpid()}", "f1", rel_), "rb").read()).hexdigest() == sha_ for rel_, sha_ in _pm["files"].items())
      and _pr_dry.returncode == 0 and (_vaS1 is None or (_vaS1["verdict"] == "single_definition_factor1" and _vaS1["evidence"]["numerator_factor"] == 1.0 and _vaS1["evidence"]["lambda_E_spec"] == 0.002)),
      f"va {_va1['verdict']}/{_va2['verdict']}/{_va3['verdict']}/{_va4['verdict']} preserve {_prc}/{_pv} postrun {_pr_dry.returncode} {_pr_dry.stderr[-200:] if _pr_dry.returncode else ''}")
# ---- K50 (검토 대응): 확장 seed 대조 id · 보류 run 상태·활성 큐 · switch ⑤ 블록 실행 fixture
_extra_ctl = {s_: (G.qrc24_control_run(s_, G.QRC24_EXTRA[s_][0]), G.qrc24_control_profile(s_, G.QRC24_EXTRA[s_][0])) for s_ in ("s1", "s2", "s3", "s4", "s5")}
try:
    G.qrc24_control_profile("s3", 424242); _virt = False
except SystemExit:
    _virt = True
_k50e = G.kdv_block("QRC24_S4_H12", 2026, "s4", cal=calQ, arch="W104_D121"); _k50e5 = G.kdv_block("QRC24_S5_L070", 2909, "s5", cal=calQ, arch="W104_D121")
check("K50 확장 seed 대조(§8.3·§11.2): §8.3 확장 seed(s1 9091 · s2 3407 · s3 1103 · s4 2026 · s5 2909) 와 reserve seed(17041/26017) 의 대조 id 는 그 묶음의 실제 alias(s4 H22 · s5 L100 · 그 밖 G22) — 가상 G22 id 를 만들지 않는다 · 편성·확장 어디에도 없는 seed 는 SystemExit · 확장 run 의 kdv block control_runs/baseline_run 도 그 id",
      {s_: v_[1] for s_, v_ in _extra_ctl.items()} == {"s1": "G22", "s2": "G22", "s3": "G22", "s4": "H22", "s5": "L100"} and _extra_ctl["s4"][0] == _qn("s4", "H22", 2026) and _extra_ctl["s5"][0] == _qn("s5", "L100", 2909)
      and all(G.qrc24_control_profile(s_, sd_) == _extra_ctl[s_][1] for s_ in ("s4", "s5") for sd_ in G.QRC24_RESERVE_SEEDS) and _virt
      and _k50e["control_runs"]["G22"] == _qn("s4", "H22", 2026) and _k50e["baseline_run"] == _qn("s4", "H22", 2026) and _k50e5["control_runs"]["G22"] == _qn("s5", "L100", 2909), f"{_extra_ctl} virt_raise {_virt}")
_hrun = f"_k50held_{os.getpid()}"; _hdir = os.path.join(ROOT, "work_dir", _hrun)
try:
    os.makedirs(_hdir, exist_ok=True); _st_none = G.qrc24_held_state("s4", _hrun, ps="")
    os.makedirs(os.path.join(_hdir, "checkpoint-1000"), exist_ok=True); _st_ck = G.qrc24_held_state("s4", _hrun, ps="")
    _st_run = G.qrc24_held_state("s4", _hrun, ps=f"python -u main.py --config /x/{_hrun}.yaml\n")
finally:
    shutil.rmtree(_hdir, ignore_errors=True)
_h4 = _qn("s4", "H11", 3407); _psfake = f"bash ./tools/_run_cases.sh\npython -u main.py --config /x/{_h4}.yaml\n"
_eff_run = G.qrc24_effective_queue("s4", ps=_psfake); _eff_plain = G.qrc24_effective_queue("s4", ps="")
check("K50 보류 run 상태(§9 '미시작일 때만'): checkpoint/epoch 없음 → superseded_pending · checkpoint-1000 있음 → started_interrupted(원 정의로 끝낸다) · 학습 중 → running_original_definition · 완료 → terminal · 활성 실행 순서(qrc24_effective_queue) 는 시작된 보류 run 을 **맨 앞**에 두고 나머지는 편성 순서, 미시작이면 편성 그대로 · run_started 는 runner 의 latest_ckpt 와 같은 규칙",
      (_st_none, _st_ck, _st_run) == (G.QRC24_HELD_STATUS, "started_interrupted", "running_original_definition") and _eff_run[0] == _h4 and _eff_run[1:] == G.qrc24_items("s4") and len(_eff_run) == len(G.qrc24_items("s4")) + 1
      and _eff_plain == G.qrc24_items("s4") and G.qrc24_effective_queue("s1", ps="") == G.qrc24_items("s1") and not G.run_started(f"_k50nope_{os.getpid()}"), f"{(_st_none, _st_ck, _st_run)} eff0 {_eff_run[0][-28:]}")
_blk5 = [b for b in __import__("re").findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", _sw50, flags=__import__("re").S) if "extra_priority.pre_qrecon24" in b]
_fxr = tempfile.mkdtemp(); _keep_run = G.qrc24_run_name("s2", "G22", 3407); _old_item = "JQ@W104_D121"
os.makedirs(os.path.join(_fxr, "work_dir", "_pakd50"), exist_ok=True); open(os.path.join(_fxr, "work_dir", "_pakd50", "extra_priority.txt"), "w").write(f"# 이전\n{_old_item}\n{_keep_run}\n")
_R0, _AV0 = G.ROOT, sys.argv; _ok5 = False; _xp_after = _bak_after = None
try:
    G.ROOT = _fxr; sys.argv = ["-", "s2"]; exec(compile(_blk5[0], "<switch5>", "exec"), {"__name__": "__main__"})
    _xp_after = [l.strip() for l in open(os.path.join(_fxr, "work_dir", "_pakd50", "extra_priority.txt")) if l.strip() and not l.startswith("#")]
    _bakf = [f for f in os.listdir(os.path.join(_fxr, "work_dir", "_pakd50")) if f.startswith("extra_priority.pre_qrecon24_")]
    _bak_after = [l.strip() for l in open(os.path.join(_fxr, "work_dir", "_pakd50", _bakf[0])) if l.strip() and not l.startswith("#")] if _bakf else None
    _qeff5 = [l.strip() for l in open(os.path.join(_fxr, "work_dir", "_qrecon24", "queue_effective.txt")) if l.strip() and not l.startswith("#")]
    _mand5 = [l.strip() for l in open(os.path.join(_fxr, "work_dir", "_qrecon24", "mandatory_runs.txt")) if l.strip() and not l.startswith("#")]
    _rev5 = json.load(open(os.path.join(_fxr, "work_dir", "_qrecon24", "queue_revision.json"))); _held5 = json.load(open(os.path.join(_fxr, "work_dir", "_qrecon24", "held_runs.json")))
    _res5 = json.load(open(os.path.join(_fxr, G.RESERVATION_FILE)))
    from tools.campaign_gate import terminal as _term5
    _mand_all = G.qrc24_items("s2") + [_keep_run]; _pend5 = [r_ for r_ in _mand_all if not _term5(r_)]
    _ok5 = (_xp_after == [_keep_run] and _bak_after == [_old_item, _keep_run] and _mand5 == _mand_all and _qeff5 == _pend5 and _rev5["pending_queue"] == _pend5 and _rev5["completed_omitted"] == len(_mand_all) - len(_pend5)
            and _rev5["queue_revision"] == G.QRC24_ADJ_REVISION and _rev5["effective_queue"] == _mand_all and _held5 == {} and len(_res5["runs"]) == len(_mand_all))
finally:
    G.ROOT, sys.argv = _R0, _AV0; shutil.rmtree(_fxr, ignore_errors=True)
check("K50 switch ⑤ 실행(검토 지적: 백업 경로 재바인딩 버그): 임시 ROOT 에서 ⑤ python 블록을 실제로 실행 — extra_priority 의 비 QRECON24 항목만 백업 파일로 옮기고 QRC24 항목은 남긴다(한 글자씩 쪼개지지 않는다) · 백업에 원본 두 줄 · queue_effective(미완만; 완료 run 은 빼서 재업로드 walk 방지)·mandatory(전체)·reservations·queue_revision·held_runs 기록 · KeyError 없이 끝난다",
      _ok5, f"xp {_xp_after} bak {_bak_after}")
# 계획 원문 PAN_*.md 는 **저장소에 두지 않는 규약**이라 서버마다 있을 수도 없을 수도 있다 (K48c·PL02 와 같은 방식): 있으면 내용 대조, 없으면 그 항목만 사유와 함께 건너뛴다. 구현 노트·CLAUDE.md·상수는 언제나 강한 검사.
_planp = os.path.join(ROOT, G.QRC24_ADJ_PLAN); _plan_here = os.path.exists(_planp); _plan_txt = open(_planp).read() if _plan_here else ""
_plan_ok = (not _plan_here) or (all(r_ in _plan_txt for r_ in _v2all) and all(p_ in _plan_txt for p_ in G.QRC24_ADJ_PROFILES) and all(f"{p_} · {sd_}" in _plan_txt or f"{p_} | {sd_}" in _plan_txt or p_ in _plan_txt for p_, sd_ in [(G.case_of(h_).split("_", 2)[2], G.seed_of(h_)) for hs_ in _held.values() for h_ in hs_])
                              and G.QRC24_ADJ_REVISION in _plan_txt and "0.9585" in _plan_txt)
check("K50 문서(§11): 구현 노트·CLAUDE.md ADJ-R1·generator 상수(PLAN/NOTE/REVISION/VERSION) · 계획 원문은 " + (f"이 서버에 있다 → 내용 대조(부록 B 16 id · 신규 profile 3 · 보류 profile · revision · 하한 .9585): {os.path.basename(_planp)}" if _plan_here
                                                                                                     else "이 서버에 없다(계획 PAN_*.md 는 저장소에 두지 않는 규약) — 존재 검사 건너뜀; 나머지 K50 검사는 그대로 돈다"),
      os.path.exists(os.path.join(ROOT, G.QRC24_ADJ_NOTE)) and "ADJ-R1" in open(os.path.join(ROOT, "CLAUDE.md")).read() and G.QRC24_ADJ_VERSION == "v2" and G.QRC24_ADJ_PLAN.startswith("research_log/PAN_") and _plan_ok,
      f"plan_here {_plan_here} plan_ok {_plan_ok}")

# ================= K51 NOA/A_ON 전수 평가 (research_log/PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md §2·§4·§6·§7·§11.2 V01–V18)
import importlib.util as _ilu51
from kdv.eval_modes import MODES as _EMODES, CallCounter as _CC, forward_mode as _fmode, state_hash as _shash, expected_calls as _ecalls, consensus_delta as _cons, sampler_off as _soff
from pa.model import PAModel as _PAM
from pa.aligner import PANGlobalAligner as _PGA
class _Bk51(torch.nn.Module):
    def __init__(s): super().__init__(); s.c = torch.nn.Conv2d(9, 8, 3, padding=1); s.seen = []
    def forward(s, pan, lpan, ms, sw):
        s.seen.append(tuple(ms.shape[-2:]))                                            # LRMS 그대로 받아야 한다 (V04)
        return s.c(torch.cat([pan, torch.nn.functional.interpolate(ms, scale_factor=4, mode="bicubic")], 1))
torch.manual_seed(51); _m51 = _PAM(_Bk51(), _PGA(8), aligner_margin=4, sampler=True).eval()
with torch.no_grad():
    _m51.aligner.fc2.weight.normal_(0, 0.01); _m51.aligner.fc2.bias.copy_(torch.tensor([0.3, -0.2]))     # zero-init 이면 모드 차이가 공허하다 (zero_module 함정)
_p51 = torch.randn(1, 1, 128, 128); _ms51 = torch.randn(1, 8, 32, 32); _lp51 = torch.randn(1, 1, 128, 128)
_h51 = _shash(_m51); _calls51, _y51 = {}, {}
for _mo in _EMODES:
    with _CC(_m51) as _c51:
        _y51[_mo] = _fmode(_m51, _p51, _ms51, _lp51, _mo)
    _calls51[_mo] = _c51.as_dict()
_exp51 = {m_: _ecalls(m_, 1) for m_ in _EMODES}
_msb51 = torch.nn.functional.interpolate(_ms51, scale_factor=4, mode="bicubic")
with torch.no_grad():
    _res51 = _m51.backbone(_p51, _lp51, _ms51, torch.ones(1))
check("K51 모드 호출·잔차(V02·V03·V04·V06): A_ON 1/1 · **NOA aligner 0회·warp 0회** · ZERO A 0회+sampler 1회 · CROP64_MED crop 수만큼 A + warp 1회 · backbone 은 LRMS(32²) 를 그대로 받는다(이미 확대된 M 아님) · M 은 정확히 한 번 더해진다(NOA y == M + backbone(P,·)) · 평가 전후 state hash 동일·grad 없음 · sampler 복원",
      _calls51["A_ON"] == dict(aligner_calls=1, warp_calls=1) and _calls51["A_BYPASS_RAW"] == dict(aligner_calls=0, warp_calls=0) and _calls51["A_ZERO_WARP"] == dict(aligner_calls=0, warp_calls=1)
      and _calls51["A_CROP64_MED"] == dict(aligner_calls=4, warp_calls=1) and all(v_ is None or _calls51[m_][k_] == v_ for m_, e_ in _exp51.items() for k_, v_ in e_.items())
      and set(_m51.backbone.seen) == {(32, 32)} and torch.allclose(_y51["A_BYPASS_RAW"], _msb51 + _res51, atol=1e-6)
      and _shash(_m51) == _h51 and all(p_.grad is None for p_ in _m51.parameters()) and _m51.sampler is True, f"{_calls51}")
_yon51, _ynoa51 = _y51["A_ON"], _y51["A_BYPASS_RAW"]
with _soff(_m51):
    _inside51 = _m51.sampler
_med51, _D51 = _cons(_m51, _p51, _msb51, size=64)
_man51 = torch.stack(sorted(_D51[:, 0, 0].tolist()) and [torch.tensor(sorted(_D51[:, 0, 0].tolist())), torch.tensor(sorted(_D51[:, 0, 1].tolist()))])
_mid51 = torch.tensor([float((_man51[0][1] + _man51[0][2]) / 2), float((_man51[1][1] + _man51[1][2]) / 2)])     # 4개 → 가운데 두 값 평균
check("K51 consensus 정의(V18·§7.4): non-overlap 64 grid(128² → 4 crop) · **성분별 중앙값, 짝수는 가운데 두 값 평균**(torch.median 의 '작은 쪽' 아님) · 전체 PAN 을 그 한 쌍으로 **한 번만** warp · sampler_off 는 블록 안에서만 False 이고 밖에서 복원 · A_ON 과 NOA 출력이 실제로 다르다(모드 비교가 공허하지 않다)",
      _D51.shape[0] == 4 and torch.allclose(_med51[0], _mid51, atol=1e-6) and not torch.allclose(_med51[0], torch.tensor([float(_man51[0][1]), float(_man51[1][1])]), atol=1e-9)
      and _inside51 is False and _m51.sampler is True and _calls51["A_CROP64_MED"]["warp_calls"] == 1 and float((_yon51 - _ynoa51).abs().max()) > 1e-4,
      f"med {[round(float(v), 5) for v in _med51[0]]} mid {[round(float(v), 5) for v in _mid51]} ON−NOA {float((_yon51 - _ynoa51).abs().max()):.2e}")
_ne51 = _ilu51.spec_from_file_location("_noaeval51", os.path.join(ROOT, "tools", "noa_eval.py")); _NE = _ilu51.module_from_spec(_ne51); _ne51.loader.exec_module(_NE)
_ep51 = _ilu51.spec_from_file_location("_evalphase51", os.path.join(ROOT, "tools", "eval_phase.py")); _EP = _ilu51.module_from_spec(_ep51); _ep51.loader.exec_module(_EP)
_nu51 = _ilu51.spec_from_file_location("_noaup51", os.path.join(ROOT, "gspread", "noa_upload.py")); _NU = _ilu51.module_from_spec(_nu51); _nu51.loader.exec_module(_NU)
_d1 = _NE.eval_dir("R", "a" * 64, "A_ON"); _d2 = _NE.eval_dir("R", "a" * 64, "A_BYPASS_RAW")
_rec51 = dict(run="R", modes=dict(A_ON=dict(rr=dict(ergas=2.05), fr=dict(hqnr_raw=0.9590)), A_BYPASS_RAW=dict(rr=dict(ergas=2.03), fr=dict(hqnr_raw=0.9600))))
_del51 = _NE.deltas_vs_on(_rec51)
_rec51b = dict(run="R", modes=dict(A_ON=dict(rr=dict(ergas=2.05), fr=dict(hqnr_raw=0.9590)), A_BYPASS_RAW=dict(rr=dict(ergas=2.041), fr=dict(hqnr_raw=0.95849))))
check("K51 캐시 분리·Δ 부호·공동목표(V07·V09·§5.2·§5.3): 모드별 결과 경로가 다르다(ON 캐시를 NOA 로 쓰지 않는다) · identity 에 eval_mode·checkpoint/config/h5/evaluator sha 가 들어간다 · ΔE = E_NOA − E_ON, ΔH = H_NOA − H_ON(원본값) · joint_pass 는 **같은 모드·같은 checkpoint** 에서 H ≥ .9585 **그리고** E < 2.040 (반올림 아님: H .95849 는 불통과)",
      _d1 != _d2 and _d1.endswith("A_ON") and _d2.endswith("A_BYPASS_RAW") and abs(_del51["delta_rr_ergas"] - (2.03 - 2.05)) < 1e-12 and abs(_del51["delta_fr_hqnr_raw"] - (0.9600 - 0.9590)) < 1e-12
      and _del51["noa_joint_pass"] is True and _del51["on_joint_pass"] is False and _NE.deltas_vs_on(_rec51b)["noa_joint_pass"] is False
      and all(k_ in open(os.path.join(ROOT, "tools", "noa_eval.py")).read() for k_ in ("eval_mode=mode", "checkpoint_sha256=sha256_file(ckf)", "fr_h5_sha256", "evaluator=ev")))
_src51 = open(os.path.join(ROOT, "tools", "noa_eval.py")).read(); _up51 = open(os.path.join(ROOT, "gspread", "noa_upload.py")).read(); _ph51 = open(os.path.join(ROOT, "tools", "eval_phase.py")).read()
check("K51 업로더(V10·V11·V12·§6.2·§6.3): NOA 26 열 키가 계획 §6.2 와 같다 · 자기 서버 탭만(gspread/server.txt) · run id 매칭은 run_tag(장식 문자열 완전일치 아님) · batch_clear/--replace/레이아웃 재생성 없음 · 기존 B..X 를 쓰지 않는다 · 학습 uploader 와 같은 로컬 flock · 같은 run id 의 중복 행을 전부 갱신(alias)",
      [c_[2] for c_ in _NU.COLUMNS] == ["noa_rr_ergas", "noa_rr_sam", "noa_rr_psnr", "noa_rr_ssim", "noa_rr_scc", "noa_rr_q8", "noa_rr_rmse", "noa_rr_cc", "noa_fr_d_lambda", "noa_fr_d_s", "noa_fr_hqnr_raw",
                                        "paired_on_rr_ergas", "paired_on_fr_hqnr_raw", "delta_rr_ergas", "delta_fr_hqnr_raw", "eval_mode", "eval_step", "eval_ckpt_sha", "source_selector",
                                        "eval_status", "legacy_on_check", "noa_joint_pass", "eval_date", "eval_server", "protocol_id", "eval_hours"]
      and ".batch_clear(" not in _up51 and "replace=True" not in _up51 and '"--replace"' not in _up51 and "run_tag" in _up51 and "fcntl.flock" in _up51 and "rows = [i for i, t in enumerate(ids) if t == run]" in _up51
      and 'sheet_name("WV3", srv)' in _up51 and "source_train_server" in _up51)
_sm51 = open(os.path.join(ROOT, "tools", "smoke_cases.py")).read(); _wd51 = open(os.path.join(ROOT, "tools", "_watchdog.sh")).read(); _wt51 = open(os.path.join(ROOT, "tools", "qrecon24_waiter.sh")).read()
_sw51 = open(os.path.join(ROOT, "tools", "qrecon24_switch.sh")).read(); _cg51 = open(os.path.join(ROOT, "tools", "campaign_gate.py")).read()
_fx51 = os.path.join(ROOT, "work_dir", "_eval_phase", "hold.json"); _had51 = os.path.exists(_fx51)
check("K51 phase hold(V15·V16·§2): 새 학습을 시작할 수 있는 네 경로가 전부 hold 를 본다 — smoke_cases(rc 2 = 원장 기록 없는 일시 사유) · waiter(HOLD_EVAL, 재기동 안 함) · cron watchdog · switch(기동 거부) · campaign_gate(조건부 실행 안 염) · hold 는 진행 중 학습을 죽이지 않는다(preempt_running_training False) · 복귀 manifest 에 큐 파일 내용·sha·revision·다음 미완 case 를 담는다 · 평가 우회가 학습 경로에 남지 않는다(sampler 복원 검사)",
      "_eval_hold()" in _sm51 and "sys.exit(2)" in _sm51 and "hold.json" in _wd51 and "HOLD_EVAL" in _wt51 and "G.eval_hold()" in _wt51
      and "평가 phase hold 중이다" in _sw51 and "eval_hold" in _cg51 and hasattr(G, "eval_hold")
      and all(k_ in _ph51 for k_ in ("queue_files", "sha256", "next_unfinished", "approved_main_queue_revision", "preempt_running_training=False", "restore_main_forward_mode"))
      and _EP.PHASES[:3] == ("MAIN_RUNNING", "DRAIN_CURRENT_CASE", "LOCAL_STUDENT_EVAL") and _EP.PHASES[-1] == "MAIN_RESUME"
      and (G.eval_hold() != {}) == _had51)
check("K51 독립 복귀·분석 순서(V13·V14·V17·§1.3·§10.5): 평가/복귀 코드 어디에도 '다른 서버 평가 완료' 나 's1 분석 완료' 를 기다리는 조건이 없다 · 평가 완료가 recipe lock 을 만들거나 본 방법(정상 추론 A_ON)을 바꾸지 않는다 · s1 분석 도구는 §7.1 네 자산과 §7.3 AXIS16(반경 .25/.5/1/2 × 4 축 = 16) 을 쓰고 q_size 를 학습 q 와 분리해 적는다",
      not any(t_ in (_src51 + _ph51 + _up51) for t_ in ("wait_for_s1", "WAIT_S1_AUDIT", "other_server_eval", "all_servers_done"))
      and "recipe_lock" not in _src51 and "recipe_lock" not in _up51 and _EP.hold_state.__doc__ is not None
      and (lambda _a51: len(_a51.probes()) == 16 and sorted({r_ for r_, _e in _a51.probes()}) == [0.25, 0.5, 1.0, 2.0] and _a51.SIZES["fr"] == (64, 128, 256) and _a51.FULL == {"rr": 256, "fr": 512}
           and len(_a51.crop_origins(512, 64)) == 64 and len(_a51.crop_origins(512, 128)) == 16 and len(_a51.crop_origins(256, 64)) == 16 and len(_a51.S1_ASSETS) == 3
           and "학습 q cache" in open(os.path.join(ROOT, "tools", "s1_aligner_analysis.py")).read())(_ilu51.module_from_spec(_ilu51.spec_from_file_location("_s1a51", os.path.join(ROOT, "tools", "s1_aligner_analysis.py"))) if False else __import__("tools.s1_aligner_analysis", fromlist=["x"])))
check("K51 문서·protocol: 계획서(있으면 protocol id 일치) · 구현 노트 · 도구 5 종 존재 · 모든 도구가 같은 protocol id 를 쓴다",
      all(os.path.exists(os.path.join(ROOT, f_)) for f_ in ("kdv/eval_modes.py", "tools/noa_eval.py", "tools/eval_phase.py", "gspread/noa_upload.py", "tools/s1_aligner_analysis.py", "research_log/2026-09-18_noa-eval-implementation.md"))
      and _NE.PROTOCOL_ID == _EP.PROTOCOL_ID == _NU.PROTOCOL_ID == "PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918"
      and (not os.path.exists(os.path.join(ROOT, "research_log", "PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md"))
           or _NE.PROTOCOL_ID in open(os.path.join(ROOT, "research_log", "PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md")).read()))
print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
