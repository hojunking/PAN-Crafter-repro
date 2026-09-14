#!/usr/bin/env python
"""PAKD50 config 생성 — L1E4 aligner 재사용·적응 × Q12/R1 fitting, raw HQNR 0.959–0.960 (research_log/PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md §3–§7, §15).

    python tools/gen_pakd50_configs.py --server s1|s2|s3 [--stage 1|2]     # 서버 seed block (s1 1234 · s2 777 · s3 2026) 의 config + 큐
    python tools/gen_pakd50_configs.py --all --stage 1                     # 세 서버 stage 1 전부

Stage 1 (τR 만 필요): J0 → F0 → JR → FR.  Stage 2 (λE 필요 — J0 seed1234 exact50K pilot 뒤 tools/pakd50_calibrate.py 가 고정): JQ → FQ → XJ.  C1(P/JK0/JE0/D/AL/LF)·C2(TCOPY/CONT) 는 별도 release.
공통 계약(계획 §3·§6): Teacher T0 = PALS24 L1E4 S2025 best_raw 의 A+U (frozen, 자기 aligner 로 forward) · Student A = T0 aligner 복사(A-FT/A-FR), U = seed 별 저장 초기값 · W112·D123 · 50K · batch 48 ·
U LR 1e-4 / A LR 1e-5 · offset λ 1e-4(J 만, 홀수 update, b=2) · candidate grid GRID1010_50K_v1 (eval_epoch 5 = 1010 update 마다 + exact 50000; 계획 GRID1K 의 실행 대체, 노트 §2) · 모든 후보 checkpoint 보존.
약명→세팅: J0 = A trainable(L0+LO) / U N0 · JQ = J / Q12(R3+EDGE-H) · JR = J / R1 · F0 = A frozen / N0 · FQ = F / Q12 · FR = F / R1 · XJ = J / X02(R1+EDGE-H, Q12 의 soft 제거) · AL0/ALQ = J 인데 A LR 3e-6."""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve, describe

CAMPAIGN_ID = "PAKD50_W112D123_WV3_20260914_v1"; PROTOCOL = "FRESH50"; GRID_ID = "GRID1010_50K_v1"
PLAN = "research_log/PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md"; SUMMARY = "research_log/PAN_Integrated_Method_Summary_2026-09-14.md"; NOTE = "research_log/2026-09-14_pakd50-implementation.md"
T0_RUN = "PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1"; T0_TAG = "best_hqnr"; T0_ASSET_DIR = "assets/pakd50/T0_run"      # s1 은 work_dir 원본, s2/s3 는 git 으로 받은 사본 (같은 layout: meta/config.yaml + best_hqnr/model.safetensors + best_hqnr_meta.json)
SERVER_SEED = {"s1": 1234, "s2": 777, "s3": 2026, "s4": 1234}     # s4(2026-09-14 배정, research_log/PAN_S4_Integrated_Experiment_Cases_2026-09-14.md): 개발 seed 1234 = s1 과 같은 논리 run id 로 서버 교차(bridge); 독립 seed 가 아니다. 확인 seed 3407 은 --seed 로
LEDGER = "work_dir/_pakd50_budget/ledger.json"; TOTAL_HOURS = 50.0; RESERVE_HOURS = 4.0; MARGIN = 1.1; RUN_RESERVED_HOURS = 4.0     # 서버당 slot 1: 0–46h 학습, 46–50h 감사 (§9.1)
CAL_PATH = "work_dir/_pakd50/calibration_resolved.json"                                                                            # tools/pakd50_calibrate.py 산출 (τR: T0, λE: J0 S1234 exact50K)
POLICY = {"J": dict(pol="A-FT", proto="I-AEQ", off=1e-4, alr=1e-5), "F": dict(pol="A-FR", proto="I-NATIVE-TRANSFER", off=0.0, alr=1e-5), "AL": dict(pol="A-FT", proto="I-AEQ", off=1e-4, alr=3e-6)}
BACKEND = {"N0": dict(rec="N0", edge=False), "R1": dict(rec="R1", edge=False), "Q12": dict(rec="R3", edge=True), "X02": dict(rec="R1", edge=True),
           # Q12 단일축 scalar variant (s4 배정 §6; 상위 계획 C1 목록의 QA05/QB005/QB02/QE025/QE10 — 계수만 바뀌고 loss 정의는 같다): α = rec.alpha(L_D), β = rec.kd_weight(L_K), λE = λE0 × lam_mult
           "Q12_A05": dict(rec="R3", edge=True, alpha=0.5), "Q12_B005": dict(rec="R3", edge=True, kd_weight=0.05), "Q12_B02": dict(rec="R3", edge=True, kd_weight=0.2),
           "Q12_E025": dict(rec="R3", edge=True, lam_mult=0.5), "Q12_E10": dict(rec="R3", edge=True, lam_mult=2.0)}
CASES = {"J0": ("J", "N0"), "JQ": ("J", "Q12"), "JR": ("J", "R1"), "XJ": ("J", "X02"), "F0": ("F", "N0"), "FQ": ("F", "Q12"), "FR": ("F", "R1"), "XF": ("F", "X02"), "AL0": ("AL", "N0"), "ALQ": ("AL", "Q12")}
for _p in ("J", "AL"):                                    # J_QA05 … J_QE10 (anchor JQ, no-KD J0) · AL_QA05 … AL_QE10 (§7 결합: anchor ALQ, no-KD AL0)
    for _v in ("QA05", "QB005", "QB02", "QE025", "QE10"):
        CASES[f"{_p}_{_v}"] = (_p, "Q12_" + _v[1:])
PURPOSE = {"J0": "Teacher-final-A → fresh-U, native GT + offset (joint baseline; seed1234 는 λE pilot)", "JQ": "주력: joint + Q12 (실패 지도 + adaptive soft + GT edge)", "JR": "joint + R1 (실패 지도 재가중만)", "XJ": "joint + X02 (Q12 의 soft 제거 대조)",
           "F0": "frozen T0 aligner + native GT (frozen baseline)", "FQ": "frozen + Q12", "FR": "frozen + R1", "XF": "frozen + X02", "AL0": "joint, A LR 3e-6, N0", "ALQ": "joint, A LR 3e-6, Q12"}
STAGE1 = ["J0", "F0", "JR", "FR"]; STAGE2 = ["JQ", "FQ", "XJ"]
PRIORITY = ["J0", "JQ", "F0", "FQ", "JR", "FR", "XJ"]     # 계획 §9.4: P0 J0/JQ → P1 F0/FQ, JR → FR 은 뒤 → P3 X02(XJ). 큐에는 J0 만 두고 나머지는 gate 'pakd50' 가 이 순서로 편성 (감사 F02)
MANDATORY = ["J0", "JQ"]                                  # P0 — 다른 run 의 예산 gate 가 이 둘의 예상 시간을 remaining_mandatory 로 예약 (감사 F05)
QUEUE_STAGE = {1: ["J0"], 2: ["JQ"]}                      # 큐 파일 내용 (stage 2 는 DONE 뒤 재진입용) — 나머지는 gate 편성
# s4 (배정 §5·§10.1): B0 J0 → B1 JQ(λE0 뒤) → E0 AL0 → E1 ALQ 가 기본 묶음. scalar(E2/E3)·결합(E4)·seed 3407 확인(C1–C3) 은 진단을 보고 사람이 고르므로
# work_dir/_pakd50/extra_priority.txt 에 case id(J_QA05 …) 또는 전체 run 이름(seed 3407 등) 을 한 줄씩 적으면 gate 가 기본 묶음 뒤에 그 순서로 편성한다.
PRIORITY_BY_SERVER = {"s4": ["J0", "JQ", "AL0", "ALQ"]}
MANDATORY_BY_SERVER = {"s4": ["J0", "JQ", "AL0", "ALQ"]}
STAGE_BY_SERVER = {"s4": {1: ["J0", "AL0"], 2: ["JQ", "ALQ"]}}
EXTRA_PRIORITY_FILE = "work_dir/_pakd50/extra_priority.txt"


def priority_for(server):
    return list(PRIORITY_BY_SERVER.get(server, PRIORITY))


def mandatory_for(server):
    return list(MANDATORY_BY_SERVER.get(server, MANDATORY))


def stage_cases(server, stage):
    return list(STAGE_BY_SERVER.get(server, {1: STAGE1, 2: STAGE2})[stage])


def extra_priority():
    """서버 로컬 추가 편성 목록 (case id 또는 전체 run 이름; # 주석) — 없으면 []."""
    p = os.path.join(ROOT, EXTRA_PRIORITY_FILE)
    if not os.path.exists(p):
        return []
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]


def is_tag(item):
    return item.startswith("PAKD50_")


def to_tag(item, seed, version="v1"):
    return item if is_tag(item) else run_name(item, seed, version)


def case_of(item):
    """run 이름 또는 case id → case id (PAKD50_<case>_W112_…)."""
    return item[len("PAKD50_"):].split("_W112")[0] if is_tag(item) else item
NEEDS_LAMBDA_E = {c for c, (p, b) in CASES.items() if BACKEND[b]["edge"]}


def run_name(case, seed, version="v1", protocol=PROTOCOL):
    return f"PAKD50_{case}_W112_D123_WV3_T0_S{seed}_{protocol}_{version}"


def pilot_run():
    return run_name("J0", 1234)


def t0_dir(server):
    """T0 경로는 모든 서버에서 자산 사본(assets/pakd50/T0_run) — s4 가 s1 과 같은 run id(seed 1234) 를 쓰므로 config 가 서버와 무관하게 같아야 한다.
    (s1 의 J0-1234 는 work_dir 원본 경로로 시작했다 — 같은 sha 파일이며 trainer 가 expected_sha256 으로 검사한다.)"""
    return T0_ASSET_DIR


def t0_identity(server):
    """T0 의 파일 sha·selected update (best_hqnr_meta.json) — 없으면 None (gate 가 막는다)."""
    from kdv.teacher_assets import sha256_file
    d = os.path.join(ROOT, t0_dir(server)); f = os.path.join(d, T0_TAG, "model.safetensors"); m = os.path.join(d, f"{T0_TAG}_meta.json")
    sha = sha256_file(f) if os.path.exists(f) else None; step = json.load(open(m))["step"] if os.path.exists(m) else None
    return sha, step


ASSET_CAL_PATH = "assets/pakd50/calibration_resolved.json"        # 서버 간 전달용 고정값 사본 (s1 의 calibrate 가 mirror; s2/s3 는 여기서 λE 를 받는다)
LAMBDA_E_KEYS = ("lambda_E", "lambda_E_source", "lambda_computed_at", "lambda_server")


def _load_json(p):
    p = os.path.join(ROOT, p); return json.load(open(p)) if os.path.exists(p) else {}


def sync_calibration_from_assets(write=False):
    """서버 로컬 calibration(work_dir) 에 λE 가 없고 사본(assets) 에 있으면 — 같은 campaign_id · 같은 τR(1e-9) 일 때만 — λE 항목을 덧입힌다.
    반환 (merged dict, source) · source ∈ {'local', 'assets', 'none'}. write=True 면 로컬 파일에 저장(없으면 사본 그대로 생성). τR 이 다르면 덧입히지 않는다(fail-safe)."""
    loc, ast = _load_json(CAL_PATH), _load_json(ASSET_CAL_PATH)
    if not loc:
        if write and ast:
            os.makedirs(os.path.dirname(os.path.join(ROOT, CAL_PATH)), exist_ok=True); json.dump(ast, open(os.path.join(ROOT, CAL_PATH), "w"), indent=1, ensure_ascii=False)
        return dict(ast), ("assets" if ast else "none")
    if loc.get("lambda_E") or not ast.get("lambda_E"):
        return loc, "local"
    same = (loc.get("campaign_id") == ast.get("campaign_id") and loc.get("tau_R") is not None and ast.get("tau_R") is not None
            and abs(float(loc["tau_R"]) - float(ast["tau_R"])) < 1e-9)
    if not same:
        return loc, "local"
    m = dict(loc); m.update({k: ast[k] for k in LAMBDA_E_KEYS if k in ast}); m["lambda_E_from"] = ASSET_CAL_PATH
    if write:
        json.dump(m, open(os.path.join(ROOT, CAL_PATH), "w"), indent=1, ensure_ascii=False)
    return m, "assets"


def calibration():
    """서버 로컬 값 + (λE 만) 사본 덧입힘 — s2/s3 가 s1 의 λE 를 받으면(pull) 재준비 없이 stage 2 가 열린다."""
    return sync_calibration_from_assets(write=False)[0]


CLOCK_PATH = "assets/pakd50/campaign_clock.json"     # 세 서버 공통 절대 시계 (s1 prepare 시각 = start; training_deadline = +46h, final_deadline = +50h). 감사 F05


def campaign_clock():
    return _load_json(CLOCK_PATH)


def training_deadline():
    return (campaign_clock() or {}).get("training_deadline")


def hours_to_deadline(now=None):
    """공통 training_deadline 까지 남은 시간(h) — 시계 파일이 없으면 None."""
    import time as _t
    dl = training_deadline()
    if not dl:
        return None
    return (_t.mktime(_t.strptime(dl[:19], "%Y-%m-%dT%H:%M:%S")) - (now if now is not None else _t.time())) / 3600.0


def schedule(has_lambda_e, is_terminal, remaining_hours, est_hours, margin=MARGIN, priority=None, extra=()):
    """gate 'pakd50' 의 순수 편성 규칙 (계획 §9.4 P0 J0/JQ → P1 F0/FQ/JR → FR → P3 XJ; §9.6 'JQ 가 가능한 시점에 다음 slot 부터').
    λE 가 없으면 τR 만 필요한 **다음 한 벌만** (그 사이 λE 를 다시 확인), 있으면 남은 전부를 우선순위대로.
    priority: 서버별 기본 묶음(case id) · extra: 추가 편성(case id 또는 전체 run 이름; 순서 유지, 중복 제거).
    admission(§11.2): 누적 margin×est ≤ 남은 시간 (remaining_hours None 이면 생략). 반환 (편성 항목, 예산으로 밀린 항목)."""
    seq, seen = [], set()
    for it in list(priority if priority is not None else PRIORITY) + list(extra):
        if it not in seen:
            seen.add(it); seq.append(it)
    order = [c for c in seq if not is_terminal(c)]
    if not has_lambda_e:
        order = [c for c in order if case_of(c) not in NEEDS_LAMBDA_E][:1]
    todo, dropped, t = [], [], 0.0
    for c in order:
        if remaining_hours is None or t + margin * est_hours <= remaining_hours:
            todo.append(c); t += margin * est_hours
        else:
            dropped.append(c)
    return todo, dropped


def kdv_block(case, seed, server, cal=None, projected=None, version="v1", pin=True):
    """계획 §4·§6 의 FRESH50 kdv 블록. pin: calibration_resolved.json 의 τR/λE 를 숫자로 고정(서버 간 동일 package); 없으면 calibrate(그 서버에서 T0/pilot 로 산출)."""
    pol_id, be_id = CASES[case]; P, B = POLICY[pol_id], BACKEND[be_id]; cal = cal if cal is not None else calibration()
    sha, step = t0_identity(server); t0 = t0_dir(server); me = run_name(case, seed, version)
    rec = dict(case=B["rec"])
    if B["rec"] != "N0":
        rec.update(alpha=float(B.get("alpha", 1.0)), kd_weight=(float(B.get("kd_weight", 0.1)) if B["rec"] == "R3" else 0.0), eps=1.0e-6, tau=(float(cal["tau_R"]) if (pin and cal.get("tau_R")) else "calibrate"), eps_scale=1.0e-6)
    stat = dict(enabled=False)
    if B["edge"]:
        lam = cal.get("lambda_E"); mult = float(B.get("lam_mult", 1.0))     # variant 는 λE0(공통 package) 의 명시적 배율 — 다시 calibrate 하지 않는다 (배율이면 λE 가 없을 때 'calibrate' 로 두지 않고 막는다)
        if mult != 1.0 and not (pin and lam):
            raise SystemExit(f"!! {case}: λE 배율 {mult} 는 고정된 λE0 가 있어야 한다 (assets/pakd50/calibration_resolved.json)")
        stat = dict(enabled=True, kind="EDGE", mode="H", window=5, alpha=1.0, kd_weight=0.1, tau="calibrate", outer_weight=(float(lam) * mult if (pin and lam) else "calibrate"), lambda_pilot=f"{pilot_run()}/last", r_grad=0.05 * mult, ramp_updates=0)
    needs_teacher = (B["rec"] != "N0")
    k = dict(campaign_id=CAMPAIGN_ID, plan_protocol_id=PROTOCOL, candidate_grid_id=GRID_ID, case_id=case, policy_id=pol_id, backend_id=be_id, run_kind="CONTROLLED", version=version, check_run_name=False,
             input_protocol=P["proto"], aligner_policy=P["pol"], diag_every=1000, diag=dict(fixed_batch=True), calibration=dict(n_patches=3072, seed=1234), aligner_lr=P["alr"],
             select=dict(primary="best_hqnr", secondary=["best_rr_val", "last"], retain_all_candidates=True), expect_arch=dict(width=112, depth=[1, 2, 3], noalign=False),
             rec=rec, stat=stat, geom_kd=dict(mode="G0"), recipe="N2_SG", eval=dict(fixed_reference_from_donor=True),
             donor=dict(source=f"{t0}/{T0_TAG}", view_margin_hr=4, expected_sha256=sha, expected_step=step),
             teacher=dict(id="T0", run=t0, tag=T0_TAG, expected_sha256=sha, bridge=False, **({} if needs_teacher else dict(eval_only=True))),
             baseline_run=run_name(("J0" if pol_id in ("J",) else "F0" if pol_id == "F" else "AL0"), seed, version),
             budget=dict(ledger=LEDGER, total_gpu_hours=TOTAL_HOURS, reserve_hours=RESERVE_HOURS, margin=MARGIN, required=False, projected_hours=projected, projected_map={me: RUN_RESERVED_HOURS},
                         remaining_mandatory=[run_name(c, SERVER_SEED[server], version) for c in mandatory_for(server) if c != case],   # 서버 기본 묶음 예약(서버 seed): 완료된 것은 trainer 가 0 으로 센다
                         **({"training_deadline": training_deadline()} if training_deadline() else {})))            # 공통 절대 마감 — trainer 가 예상 종료 ≤ 마감 을 검사
    if P["proto"] == "I-AEQ":
        k["corruption"] = dict(radius_hr=2.0, corruption_seed_offset=2000)
        k["aux"] = dict(offset_weight=float(P["off"]), offset_ramp_updates=0, offset_stop_reference=True, geometry_weight=0.0, ramp_updates=5000, geometry_sigma_hr=2.0, geometry_margin_hr=11)
    else:
        k["aux"] = dict(offset_weight=0.0, geometry_weight=0.0)
    return k


def render(tag, case, seed, server, k, updates, eval_epoch, tpl):
    import yaml
    sp = resolve(k); pol_id, be_id = CASES[case]; sha, step = t0_identity(server)
    t = re.sub(r"^(#.*\n)+", "", tpl)
    head = (f"# {tag} — {PURPOSE[case]}. 생성: tools/gen_pakd50_configs.py --server {server}. 손으로 고치지 말 것.\n"
            f"# 캠페인 {CAMPAIGN_ID} · protocol {PROTOCOL} · grid {GRID_ID} · 계획 {PLAN} · 요약 {SUMMARY} · 노트 {NOTE}\n"
            f"# 세팅: {describe(sp)} · 정책 {pol_id}(A {'trainable' if sp['aligner_trainable'] else 'frozen'}, offset λ {k.get('aux', {}).get('offset_weight', 0)}, A LR {k['aligner_lr']}) · backend {be_id} (rec {k['rec']['case']}{', EDGE-H λE ' + str(k['stat'].get('outer_weight')) if k['stat'].get('enabled') else ''})\n"
            f"# 약명→세팅: J0/JQ/JR/XJ = A trainable(joint) + N0/Q12/R1/X02 · F0/FQ/FR/XF = A frozen + … · AL0/ALQ = joint, A LR 3e-6 · Q12 = (1+αd)L1 + β(1−d)a|S−T| + λE·signed Scharr edge · R1 = (1+αd)L1 · X02 = R1 + edge\n"
            f"# Teacher T0 = {T0_RUN}/{T0_TAG} (step {step}, file sha {(sha or '?')[:16]}…; A+U frozen, 자기 aligner 로 forward) · Student A = T0 aligner 복사(view margin 4), U = init_unet_seed{seed}.pt · τR/λE = {CAL_PATH} (고정) \n"
            f"# 골격 W112·D123 · 9ch · 50K · batch 48 · AdamW 1e-4/{k['aligner_lr']} wd 0.01 cosine warmup100 · eval_epoch {eval_epoch} (= {GRID_ID}: 1010 update 마다 + exact 50000, 50 후보 보존) · 예산 {LEDGER} {TOTAL_HOURS}h(+감사 {RESERVE_HOURS}h)\n")
    t = re.sub(r"^eval_epoch: \d+$", f"eval_epoch: {eval_epoch}", t, flags=re.M)
    t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
    t = re.sub(r"^trainer: po\npo:\n(  .*\n)+", "", t, flags=re.M)
    t = t.replace("mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)",
                  "mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)\ntrainer: kdv\nkdv:\n" + "\n".join("  " + l for l in yaml.safe_dump(k, sort_keys=False, allow_unicode=True, default_flow_style=False).splitlines()) + "\n")
    t = re.sub(r"^num_iter: \d+", f"num_iter: {updates}", t, flags=re.M); t = re.sub(r"^seed: \d+", f"seed: {seed}", t, flags=re.M)
    assert "trainer: kdv" in t and "trainer: po" not in t and f"seed: {seed}" in t and "hidden_size: 112" in t and "depth: [1, 2, 3]" in t and f"eval_epoch: {eval_epoch}" in t
    return head + t


def generate(server, cases, out_dir, updates=50000, eval_epoch=5, projected=None, version="v1", pin=True, seed=None):
    seed = seed or SERVER_SEED[server]; tpl = open(os.path.join(ROOT, "config", "PO10_N1_REC_W112_D123_WV3_S2025_R200_FRSTAT.yaml")).read(); made = []
    for case in cases:
        tag = run_name(case, seed, version); k = kdv_block(case, seed, server, projected=projected, version=version, pin=pin)
        os.makedirs(out_dir, exist_ok=True); open(os.path.join(out_dir, tag + ".yaml"), "w").write(render(tag, case, seed, server, k, updates, eval_epoch, tpl)); made.append(tag)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=None, choices=list(SERVER_SEED)); ap.add_argument("--all", action="store_true"); ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--cases", default=None, help="쉼표 목록 (기본: stage 별 목록)"); ap.add_argument("--updates", type=int, default=50000); ap.add_argument("--eval-epoch", type=int, default=5)
    ap.add_argument("--projected-hours", type=float, default=None); ap.add_argument("--version", default="v1"); ap.add_argument("--out-dir", default=os.path.join(ROOT, "config")); ap.add_argument("--no-pin", action="store_true", help="τR/λE 를 숫자로 고정하지 않고 그 서버에서 calibrate")
    ap.add_argument("--seed", type=int, default=None, help="서버 seed 대신 (s4 확인 seed 3407 등; 큐 파일은 만들지 않는다 — extra_priority.txt 에 run 이름을 적을 것)")
    a = ap.parse_args()
    servers = list(SERVER_SEED) if a.all else [a.server or open(os.path.join(ROOT, "gspread", "server.txt")).read().strip()]
    cal = calibration()
    for srv in servers:
        cases = [c.strip() for c in a.cases.split(",")] if a.cases else stage_cases(srv, a.stage)
        if not a.no_pin and (not cal.get("lambda_E")) and any(c in NEEDS_LAMBDA_E for c in cases):
            sys.exit(f"!! {[c for c in cases if c in NEEDS_LAMBDA_E]} 는 λE 가 고정된 뒤에 만든다 — tools/pakd50_calibrate.py --lambda-e (J0 S1234 exact50K 필요). 현재 {CAL_PATH}: {list(cal)}")
        made = generate(srv, cases, a.out_dir, a.updates, a.eval_epoch, a.projected_hours, a.version, not a.no_pin, seed=a.seed)
        if a.out_dir == os.path.join(ROOT, "config") and a.version == "v1" and a.seed is None and not a.cases:
            q = os.path.join(ROOT, "config", "queues", f"pakd50_{srv}_stage{a.stage}.txt")
            with open(q, "w") as f:
                f.write(f"# PAKD50 {srv} (seed {SERVER_SEED[srv]}) stage {a.stage} — {PLAN} §9.4/§9.6/§11.2. 큐에는 {'J0 (P0 · s1 은 λE pilot 겸함)' if a.stage == 1 else 'JQ (DONE 뒤 재진입용)'} 만 둔다.\n"
                        f"# 나머지는 매 pass 조건부 gate 'pakd50'(work_dir/campaign_gates_enabled.txt) 이 우선순위 {' → '.join(PRIORITY)} 로 편성한다: λE(J0 S1234 exact50K pilot → tools/pakd50_calibrate.py, {ASSET_CAL_PATH} 사본) 가 없으면 τR 만 필요한 다음 한 벌, 있으면 남은 전부.\n"
                        f"# 예산: 공통 절대 시계 {CLOCK_PATH} (학습 마감 = start+46h, 감사 +4h) — 예상 종료가 마감을 넘기는 run 은 시작하지 않는다 ({LEDGER} 50h/4h). 약명→세팅은 config 머리 주석 / {NOTE}\n"
                        + "\n".join(run_name(c, SERVER_SEED[srv], a.version) for c in QUEUE_STAGE[a.stage]) + "\n")
            print("queue:", os.path.relpath(q, ROOT))
        print(f"[{srv}] " + " ".join(made))
    sha, step = t0_identity(servers[0]); print("T0:", T0_RUN, T0_TAG, "step", step, "sha", (sha or "?")[:16], "| calibration:", {k: cal.get(k) for k in ("tau_R", "lambda_E")})


if __name__ == "__main__":
    main()
