#!/usr/bin/env python
"""PAKD50 config 생성 — L1E4 aligner 재사용·적응 × Q12/R1 fitting, raw HQNR 0.959–0.960 (research_log/PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md §3–§7, §15).

    python tools/gen_pakd50_configs.py --server s1|s2|s3 [--stage 1|2]     # 서버 seed block (s1 1234 · s2 777 · s3 2026) 의 config + 큐
    python tools/gen_pakd50_configs.py --all --stage 1                     # 세 서버 stage 1 전부

Stage 1 (τR 만 필요): J0 → F0 → JR → FR.  Stage 2 (λE 필요 — J0 seed1234 exact50K pilot 뒤 tools/pakd50_calibrate.py 가 고정): JQ → FQ → XJ.  C1(P/JK0/JE0/D/AL/LF)·C2(TCOPY/CONT) 는 별도 release.
공통 계약(계획 §3·§6): Teacher T0 = PALS24 L1E4 S2025 best_raw 의 A+U (frozen, 자기 aligner 로 forward) · Student A = T0 aligner 복사(A-FT/A-FR), U = seed 별 저장 초기값 · W112·D123 · 50K · batch 48 ·
U LR 1e-4 / A LR 1e-5 · offset λ 1e-4(J 만, 홀수 update, b=2) · candidate grid GRID1010_50K_v1 (eval_epoch 5 = 1010 update 마다 + exact 50000; 계획 GRID1K 의 실행 대체, 노트 §2) · 모든 후보 checkpoint 보존.
약명→세팅: J0 = A trainable(L0+LO) / U N0 · JQ = J / Q12(R3+EDGE-H) · JR = J / R1 · F0 = A frozen / N0 · FQ = F / Q12 · FR = F / R1 · XJ = J / X02(R1+EDGE-H, Q12 의 soft 제거) · AL0/ALQ = J 인데 A LR 3e-6.
2026-09-15 재배정(research_log/PAN_PAKD50_S2_S4_S5_Derived_Run_Allocation_2026-09-15.md): s2/s4/s5 의 명시 순서(PRIORITY_BY_SERVER) + 새 case J_R3_NOEDGE(J / R3, edge 없음) · J_N0_EDGE(J / N0 + EDGE-H, Teacher 미사용) ·
RC0/RCQ(RC = A trainable LR 1e-5 인데 Student 단계 offset 연습 없음: I-NATIVE-TRANSFER, radius 0, offset 0 / N0·Q12) + slot 예약식 reservation_h = 1.10 × reference_train_h + 10/60 (gate 편성·trainer 예산 gate 공통).
    python tools/gen_pakd50_configs.py --server s4 --cases F0,RC0,RCQ,JR,XJ,J_R3_NOEDGE     # 재배정 목록 config (큐 파일은 그대로 J0 만; 편성은 gate)
    python tools/gen_pakd50_configs.py --plan [--server s2]                                # 재배정 순서·예약·누적 (dry-run; 완료·실행 중 run 은 work_dir 로 제외)"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve, describe

CAMPAIGN_ID = "PAKD50_W112D123_WV3_20260914_v1"; PROTOCOL = "FRESH50"; GRID_ID = "GRID1010_50K_v1"
PLAN = "research_log/PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md"; SUMMARY = "research_log/PAN_Integrated_Method_Summary_2026-09-14.md"; NOTE = "research_log/2026-09-14_pakd50-implementation.md"
T0_RUN = "PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1"; T0_TAG = "best_hqnr"; T0_ASSET_DIR = "assets/pakd50/T0_run"      # s1 은 work_dir 원본, s2/s3 는 git 으로 받은 사본 (같은 layout: meta/config.yaml + best_hqnr/model.safetensors + best_hqnr_meta.json)
SERVER_SEED = {"s1": 1234, "s2": 777, "s3": 2026, "s4": 1234, "s5": 2026}     # s5(배정 research_log/PAN_S5_Timing_Routing_Experiment_Plan_2026-09-14.md): 탐색 seed 2026 = s3 교차(bridge), 확인 seed 9091 은 --seed 로     # s4(2026-09-14 배정, research_log/PAN_S4_Integrated_Experiment_Cases_2026-09-14.md): 개발 seed 1234 = s1 과 같은 논리 run id 로 서버 교차(bridge); 독립 seed 가 아니다. 확인 seed 3407 은 --seed 로
LEDGER = "work_dir/_pakd50_budget/ledger.json"; TOTAL_HOURS = 50.0; RESERVE_HOURS = 4.0; MARGIN = 1.1; RUN_RESERVED_HOURS = 4.0     # 서버당 slot 1: 0–46h 학습, 46–50h 감사 (§9.1)
CAL_PATH = "work_dir/_pakd50/calibration_resolved.json"                                                                            # tools/pakd50_calibrate.py 산출 (τR: T0, λE: J0 S1234 exact50K)
POLICY = {"J": dict(pol="A-FT", proto="I-AEQ", off=1e-4, alr=1e-5), "F": dict(pol="A-FR", proto="I-NATIVE-TRANSFER", off=0.0, alr=1e-5), "AL": dict(pol="A-FT", proto="I-AEQ", off=1e-4, alr=3e-6),
          # s5 timing/routing (배정 §4–§6; 상위 계획 §7.2–7.6): 계수·LR 은 J 와 같고 A 의 업데이트 시점·수신 경로만 다르다. kdv.aligner_schedule / kdv.routing (registry 가 미지원 조합을 거부)
          "D": dict(pol="A-FT", proto="I-AEQ", off=1e-4, alr=1e-5, schedule=dict(freeze_until=5000)),                       # 0–4999 A 동결(LO 없음), 5000 부터 J
          "LF": dict(pol="A-FT", proto="I-AEQ", off=1e-4, alr=1e-5, schedule=dict(freeze_from=25000)),                      # 0–24999 J, 25000 부터 A 동결(LO 없음)
          "P": dict(pol="A-FT", proto="I-AEQ", off=1e-4, alr=1e-5, routing=dict(qD=0.0, qK=0.0, qE=0.0)),                   # A 는 L0+LO 만, U 는 backend 전체
          "DP": dict(pol="A-FT", proto="I-AEQ", off=1e-4, alr=1e-5, schedule=dict(freeze_until=5000), routing=dict(qD=0.0, qK=0.0, qE=0.0)),
          "JK0": dict(pol="A-FT", proto="I-AEQ", off=1e-4, alr=1e-5, routing=dict(qK=0.0)),                                  # A 에서 L_K(soft) 만 차단
          "JE0": dict(pol="A-FT", proto="I-AEQ", off=1e-4, alr=1e-5, routing=dict(qE=0.0)),                                 # A 에서 λE L_E 만 차단
          # 재배정 2026-09-15 §5 (s4/s5): Stage 2 의 relative-offset 보조 연습만 제거 — A 는 T0 복사·trainable(LR 1e-5), 입력은 매 update native(I-NATIVE-TRANSFER; radius 0), L_rec → A gradient 유지.
          # I-AEQ 에 offset 0 을 넣는 방식은 registry 계약(I-AEQ 는 offset > 0) 과 맞지 않아 protocol 자체를 바꾼다.
          "RC": dict(pol="A-FT", proto="I-NATIVE-TRANSFER", off=0.0, alr=1e-5, radius=0.0)}
BASELINE_OF = {"J": "J0", "F": "F0", "AL": "AL0", "D": "D0", "DP": "D0", "LF": "LF0", "P": "J0", "JK0": "J0", "JE0": "J0", "RC": "RC0"}   # no-KD control (배정 §5–§6; RC 는 재배정 §5)
BACKEND = {"N0": dict(rec="N0", edge=False), "R1": dict(rec="R1", edge=False), "Q12": dict(rec="R3", edge=True), "X02": dict(rec="R1", edge=True),
           # Q12 단일축 scalar variant (s4 배정 §6; 상위 계획 C1 목록의 QA05/QB005/QB02/QE025/QE10 — 계수만 바뀌고 loss 정의는 같다): α = rec.alpha(L_D), β = rec.kd_weight(L_K), λE = λE0 × lam_mult
           "Q12_A05": dict(rec="R3", edge=True, alpha=0.5), "Q12_B005": dict(rec="R3", edge=True, kd_weight=0.05), "Q12_B02": dict(rec="R3", edge=True, kd_weight=0.2),
           "Q12_E025": dict(rec="R3", edge=True, lam_mult=0.5), "Q12_E10": dict(rec="R3", edge=True, lam_mult=2.0),
           # 재배정 2026-09-15 §5 (s2/s4 backend 성분 전체 제거): R3_NOEDGE = 재가중 GT hard + adaptive soft, GT edge 없음 (JQ 에서 λE L_E 제거) · N0_EDGE = plain GT L1 + λE GT edge (Teacher 를 학습 loss 에 쓰지 않음 → eval_only)
           "R3_NOEDGE": dict(rec="R3", edge=False), "N0_EDGE": dict(rec="N0", edge=True)}
CASES = {"J0": ("J", "N0"), "JQ": ("J", "Q12"), "JR": ("J", "R1"), "XJ": ("J", "X02"), "F0": ("F", "N0"), "FQ": ("F", "Q12"), "FR": ("F", "R1"), "XF": ("F", "X02"), "AL0": ("AL", "N0"), "ALQ": ("AL", "Q12")}
for _p in ("J", "AL"):                                    # J_QA05 … J_QE10 (anchor JQ, no-KD J0) · AL_QA05 … AL_QE10 (§7 결합: anchor ALQ, no-KD AL0)
    for _v in ("QA05", "QB005", "QB02", "QE025", "QE10"):
        CASES[f"{_p}_{_v}"] = (_p, "Q12_" + _v[1:])
CASES.update({"D0": ("D", "N0"), "DQ": ("D", "Q12"), "DR": ("D", "R1"), "DX": ("D", "X02"), "PQ": ("P", "Q12"), "PR": ("P", "R1"), "PX": ("P", "X02"),
              "DPQ": ("DP", "Q12"), "DPX": ("DP", "X02"), "LF0": ("LF", "N0"), "LFQ": ("LF", "Q12"), "JK0": ("JK0", "Q12"), "JE0": ("JE0", "Q12"),
              "J_R3_NOEDGE": ("J", "R3_NOEDGE"), "J_N0_EDGE": ("J", "N0_EDGE"), "RC0": ("RC", "N0"), "RCQ": ("RC", "Q12")})     # 재배정 2026-09-15 §5
_PURPOSE_S5 = {"D0": "s5: 초기 5K A 동결 뒤 joint, N0 (delayed-joint 의 no-KD control)", "DQ": "s5: 초기 5K A 동결, U 는 처음부터 Q12 (delayed joint KD)", "DR": "s5 fallback: D 일정 + R1", "DX": "s5: DQ 의 soft 제거 대조 (D 일정 + X02)",
                "PQ": "s5: protected routing — A 는 L0+LO 만, U 는 Q12 전체", "PR": "s5 fallback: protected + R1", "PX": "s5: PQ 의 soft 제거 대조 (protected + X02)",
                "DPQ": "s5: delayed + protected 결합 (Q12)", "DPX": "s5: DPQ 의 soft 제거 대조", "LF0": "s5: 25K 이후 A 동결, N0 (late-freeze control)", "LFQ": "s5: 25K 이후 A 동결, Q12",
                "JK0": "s5: JQ 에서 A 로 가는 L_K(soft) 만 차단", "JE0": "s5: JQ 에서 A 로 가는 λE L_E 만 차단",
                # 재배정 2026-09-15 §5: JR = weighted_GT · J_R3_NOEDGE = weighted_GT + adaptive soft · XJ = weighted_GT + λE edge · JQ = 셋 다 · J_N0_EDGE = plain GT + λE edge (Teacher 없이)
                "J_R3_NOEDGE": "재배정 s2/s4: joint + R3 adaptive soft, GT edge 전체 제거 (JQ − edge)", "J_N0_EDGE": "재배정 s2: joint + plain GT L1 + λE GT edge, Teacher 를 loss 에 쓰지 않음 (Teacher-free edge 대조)",
                "RC0": "재배정 s4/s5: A trainable(LR 1e-5) 인데 Student 단계 offset 연습 없음(native 만), N0 (RC 의 no-KD control)", "RCQ": "재배정 s4/s5: RC 정합 정책 + Q12 전체"}
PURPOSE = {"J0": "Teacher-final-A → fresh-U, native GT + offset (joint baseline; seed1234 는 λE pilot)", "JQ": "주력: joint + Q12 (실패 지도 + adaptive soft + GT edge)", "JR": "joint + R1 (실패 지도 재가중만)", "XJ": "joint + X02 (Q12 의 soft 제거 대조)",
           "F0": "frozen T0 aligner + native GT (frozen baseline)", "FQ": "frozen + Q12", "FR": "frozen + R1", "XF": "frozen + X02", "AL0": "joint, A LR 3e-6, N0", "ALQ": "joint, A LR 3e-6, Q12"}
STAGE1 = ["J0", "F0", "JR", "FR"]; STAGE2 = ["JQ", "FQ", "XJ"]
PRIORITY = ["J0", "JQ", "F0", "FQ", "JR", "FR", "XJ"]     # 계획 §9.4: P0 J0/JQ → P1 F0/FQ, JR → FR 은 뒤 → P3 X02(XJ). 큐에는 J0 만 두고 나머지는 gate 'pakd50' 가 이 순서로 편성 (감사 F02)
MANDATORY = ["J0", "JQ"]                                  # P0 — 다른 run 의 예산 gate 가 이 둘의 예상 시간을 remaining_mandatory 로 예약 (감사 F05)
QUEUE_STAGE = {1: ["J0"], 2: ["JQ"]}                      # 큐 파일 내용 (stage 2 는 DONE 뒤 재진입용) — 나머지는 gate 편성
# s4 (배정 §5·§10.1): B0 J0 → B1 JQ(λE0 뒤) → E0 AL0 → E1 ALQ 가 기본 묶음. scalar(E2/E3)·결합(E4)·seed 3407 확인(C1–C3) 은 진단을 보고 사람이 고르므로
# work_dir/_pakd50/extra_priority.txt 에 case id(J_QA05 …) 또는 전체 run 이름(seed 3407 등) 을 한 줄씩 적으면 gate 가 기본 묶음 뒤에 그 순서로 편성한다.
# s5 (배정 §0·§9.2): B0 J0 → B1 JQ(λE0) → T0 D0 → T1 DQ(λE0) → R0 PQ(λE0). 조건부(LF pair / JK0·JE0 / DPQ / soft-off DX·PX·DPX) 와 seed 9091 확인은 extra_priority.txt 로.
# 2026-09-15 재배정 (research_log/PAN_PAKD50_S2_S4_S5_Derived_Run_Allocation_2026-09-15.md §4·§9): s2/s4/s5 의 **명시 순서** — 완료된 기본 묶음(s2 J0/F0/JQ/FQ · s4 AL0/J0/JQ/ALQ · s5 J0/JQ/D0/DQ) 은
# 다시 넣지 않고(work_dir 완료 판정으로도 제외), 옛 꼬리(s2 FR 등) 는 편성에서 빠진다. s5 PQ 는 실행 중이면 그 run 을 유지하고 남은 시간만 센다. s1/s3 는 신규 배정 없음(기본 PRIORITY 유지).
# 그 전 묶음(2026-09-14): s4 J0→JQ→AL0→ALQ · s5 J0→JQ→D0→DQ→PQ (research_log/2026-09-14_pakd50-s{4,5}-review-and-implementation.md).
PRIORITY_BY_SERVER = {"s2": ["JR", "XJ", "J_R3_NOEDGE", "J_N0_EDGE"], "s4": ["F0", "RC0", "RCQ", "JR", "XJ", "J_R3_NOEDGE"], "s5": ["PQ", "F0", "LF0", "LFQ", "RC0", "RCQ"]}
MANDATORY_BY_SERVER = dict(PRIORITY_BY_SERVER)        # 기본 16 run(4+6+6) 전부가 예산 예약 대상 (완료된 run 은 trainer 가 0 으로 센다)
PREVIOUS_PRIORITY_BY_SERVER = {"s4": ["J0", "JQ", "AL0", "ALQ"], "s5": ["J0", "JQ", "D0", "DQ", "PQ"]}
ALLOC_PLAN = "research_log/PAN_PAKD50_S2_S4_S5_Derived_Run_Allocation_2026-09-15.md"
ALLOCATED_SERVERS = ("s2", "s4", "s5")
# 시간 산정 (재배정 §2–§3): 같은 서버 완료 case 의 Sheet Train(h)(2026-09-15 00:05 live read; 새 case 실측이 아니라 **계획 기준값**) — N0 형(rec N0·edge 없음) 은 J0, Teacher/Q12 형은 JQ 의 값.
# s2 는 이번 4 case 전부 2.33 (보수). s5 D0 1.15 는 일반화하지 않는다. routing case(PQ 등) 는 완료 기록이 없어 1.80h 가예약 — 같은 서버에서 첫 실측이 나오면 그것으로 바꾼다 (reference_hours).
REFERENCE_TRAIN_H = {"s2": dict(N0=1.97, T=2.33, all=2.33), "s4": dict(N0=1.17, T=1.39), "s5": dict(N0=1.35, T=1.36)}
ROUTING_PLACEHOLDER_H = 1.80; CONFIRM_PLACEHOLDER_H = 1.80          # 미실측 경로 가예약 (실측·성능 예측이 아니다; 결과 수치로 기록하지 않는다)
RESERVE_SLACK = 1.10; RESERVE_POST_H = 10.0 / 60.0                   # reservation_h = 1.10 × reference_train_h + 10/60 (10 % 변동 여유 + run 뒤 export/업로드/전환 10 분 가예약)
CONFIRM_SEED = {"s4": 3407, "s5": 9091}; CONFIRM_MAX_RUNS = 3        # §7: 외부 분석의 WIN 확정 뒤에만, 서버당 최대 3 run (WIN · 같은 policy 의 no-KD control · F0; F0 가 control 이면 2)
RESERVATION_FILE = "work_dir/_pakd50/reservations.json"             # 서버 로컬: run → gate_hours (= reservation_h / MARGIN; trainer 가 margin 을 곱하면 reservation_h). gate 가 매 pass 갱신
STAGE_BY_SERVER = {"s4": {1: ["J0", "AL0"], 2: ["JQ", "ALQ"]}, "s5": {1: ["J0", "D0"], 2: ["JQ", "DQ", "PQ"]}}     # --stage 용 (2026-09-14 기본 묶음; 재배정 목록은 --cases / 편성은 gate)
EXTRA_PRIORITY_FILE = "work_dir/_pakd50/extra_priority.txt"
MANDATORY_FILE = "work_dir/_pakd50/mandatory_runs.txt"      # 서버 로컬: 이 서버의 기본 묶음 run 이름 (trainer 예산 gate 의 remaining_mandatory 출처; prepare 가 mandatory_for(server) 로 쓴다)


def write_mandatory_file(server, version="v1"):
    p = os.path.join(ROOT, MANDATORY_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
    runs = [run_name(c, SERVER_SEED[server], version) for c in mandatory_for(server)]
    with open(p, "w") as f:
        f.write(f"# {server} 기본 묶음 (예산 gate 예약; 완료된 run 은 trainer 가 0 으로 센다) — gen_pakd50_configs.write_mandatory_file\n" + "\n".join(runs) + "\n")
    return runs


INIT_HASH_PATH = "assets/pakd50/init_hashes.json"          # seed → 저장 U 초기값(init_unet_seed<seed>.pt) 의 tensor sha256_16. 값이 있으면 config expect_init 으로 박혀 trainer 가 fail-fast (s5 보고 #3)


def init_hash_for(seed):
    return ((_load_json(INIT_HASH_PATH) or {}).get("unet") or {}).get(str(int(seed)))


def servers_sharing_seed(server):
    return [s for s, sd in SERVER_SEED.items() if sd == SERVER_SEED[server]]


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


def seed_of(item, default=None):
    """run 이름 → seed (…_S<seed>_<protocol>_); case id 면 default."""
    m = re.search(r"_S(\d+)_", item) if is_tag(item) else None
    return int(m.group(1)) if m else default


def reference_kind(case):
    """시간 산정 유형 (재배정 §3): ROUTING(정책에 routing) · N0(rec N0·edge 없음) · T(Teacher/Q12 형: 그 밖 전부)."""
    pol_id, be_id = CASES[case]
    if POLICY[pol_id].get("routing"):
        return "ROUTING"
    B = BACKEND[be_id]
    return "N0" if (B["rec"] == "N0" and not B["edge"]) else "T"


def reference_hours(server, case, measured=None, confirm=False):
    """run 의 시간 산정 기준 h 와 출처. measured: 같은 서버 완료 run 의 실측 {case: h} (ledger) — 같은 case 실측이 있으면 그것이 우선(§3 '첫 실측이 나오면 곧바로 교체').
    없으면: confirm(확인 seed) → 1.80 가예약(§7) · ROUTING → 같은 서버 routing 실측 평균 → 없으면 1.80 가예약 · N0/T → 서버 계획 기준값 (s2 는 전부 2.33) → 없으면 실측 평균 → None."""
    measured = measured or {}
    if measured.get(case):
        return float(measured[case]), "measured_same_case"
    if confirm:
        return CONFIRM_PLACEHOLDER_H, "confirm_placeholder"
    kind = reference_kind(case)
    if kind == "ROUTING":
        rs = [float(h) for c, h in measured.items() if c in CASES and reference_kind(c) == "ROUTING"]
        return (sum(rs) / len(rs), "measured_routing_mean") if rs else (ROUTING_PLACEHOLDER_H, "routing_placeholder")
    tbl = REFERENCE_TRAIN_H.get(server)
    if tbl:
        return float(tbl.get("all", tbl[kind])), "plan_reference"
    if measured:
        return sum(float(h) for h in measured.values()) / len(measured), "measured_mean"
    return None, "none"


def reservation_hours(ref):
    """slot 예약 (재배정 §3): 1.10 × reference_train_h + 10/60."""
    return RESERVE_SLACK * float(ref) + RESERVE_POST_H


def measured_hours_from_ledger(server, ledger=None):
    """예산 ledger(work_dir/_pakd50_budget/ledger.json) 의 FINISHED run → {case: h} (이 서버 seed 의 run 만; 같은 case 여러 번이면 평균)."""
    d = ledger if ledger is not None else _load_json(LEDGER); acc = {}
    for rid, e in (d.get("entries") or {}).items():
        if e.get("kind") != "run" or not str(e.get("status", "")).startswith("FINISHED") or not is_tag(rid) or "#" in rid:
            continue
        h = e.get("hours_total") or e.get("hours")
        if not h or seed_of(rid) != SERVER_SEED.get(server):
            continue
        acc.setdefault(case_of(rid), []).append(float(h))
    return {c: sum(v) / len(v) for c, v in acc.items()}


def reservation_for(server, item, measured=None, seed=None):
    """편성 항목(case id 또는 run 이름) → dict(run, case, seed, reference_train_h, reference_kind, reservation_h, gate_hours)."""
    seed = seed or SERVER_SEED[server]; case = case_of(item); sd = seed_of(item, seed); confirm = (sd != SERVER_SEED[server])
    ref, kind = reference_hours(server, case, measured, confirm=confirm)
    if ref is None:
        return None
    res = reservation_hours(ref)
    return dict(run=to_tag(item, seed), case=case, seed=sd, reference_train_h=ref, reference_kind=kind, reservation_h=res, gate_hours=res / MARGIN)


def write_reservation_file(server, items, measured=None, seed=None, path=None):
    """서버 로컬 RESERVATION_FILE: run → gate_hours 등 (trainer budget.projection_file; margin 1.1 을 곱하면 reservation_h 가 된다). items 는 편성 목록(case id / run 이름)."""
    p = os.path.join(ROOT, path or RESERVATION_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
    rows = [r for r in (reservation_for(server, it, measured, seed) for it in items) if r]
    out = dict(note=f"{server}: reservation_h = {RESERVE_SLACK} × reference_train_h + {RESERVE_POST_H * 60:.0f}/60 ({ALLOC_PLAN} §3); gate_hours = reservation_h / {MARGIN} (trainer 가 budget.margin 을 곱한다). "
                    "reference 는 계획 기준값/가예약이지 실측이 아니다 (kind 참조); 같은 서버 같은 case 실측이 있으면 그것.", server=server, margin=MARGIN, formula="1.10*ref+10/60",
               runs={r["run"]: {k: v for k, v in r.items() if k != "run"} for r in rows})
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    return out


def confirmation_cases(win_case):
    """§7 확인 seed 묶음: WIN · 같은 policy 의 no-KD control · F0 (control 이 F0 면 2 run). 순서 유지, 중복 제거."""
    if win_case not in CASES:
        raise SystemExit(f"!! 알 수 없는 case {win_case}")
    ctl = BASELINE_OF[CASES[win_case][0]]; out = []
    for c in (win_case, ctl, "F0"):
        if c not in out:
            out.append(c)
    assert len(out) <= CONFIRM_MAX_RUNS
    return out
PURPOSE.update(_PURPOSE_S5)
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
    admission(§11.2): 누적 margin×est ≤ 남은 시간 (remaining_hours None 이면 생략). 반환 (편성 항목, 예산으로 밀린 항목).
    est_hours 가 callable 이면 항목별 **예약 시간**(reservation_hours: 여유가 이미 들어 있다) 으로 보고 margin 을 곱하지 않는다 (재배정 §3·§8)."""
    seq, seen = [], set()
    for it in list(priority if priority is not None else PRIORITY) + list(extra):
        if it not in seen:
            seen.add(it); seq.append(it)
    order = [c for c in seq if not is_terminal(c)]
    if not has_lambda_e:
        order = [c for c in order if case_of(c) not in NEEDS_LAMBDA_E][:1]
    todo, dropped, t = [], [], 0.0
    for c in order:
        need = float(est_hours(c)) if callable(est_hours) else margin * float(est_hours)
        if remaining_hours is None or t + need <= remaining_hours:
            todo.append(c); t += need
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
             **({"expect_init": dict(unet_sha256_16=init_hash_for(seed))} if init_hash_for(seed) else {}),
             donor=dict(source=f"{t0}/{T0_TAG}", view_margin_hr=4, expected_sha256=sha, expected_step=step),
             teacher=dict(id="T0", run=t0, tag=T0_TAG, expected_sha256=sha, bridge=False, **({} if needs_teacher else dict(eval_only=True))),
             baseline_run=run_name(BASELINE_OF[pol_id], seed, version),
             budget=dict(ledger=LEDGER, total_gpu_hours=TOTAL_HOURS, reserve_hours=RESERVE_HOURS, margin=MARGIN, required=False, projected_hours=projected, projected_map={me: RUN_RESERVED_HOURS},
                         remaining_mandatory=[], remaining_mandatory_file=MANDATORY_FILE,     # 서버 기본 묶음 예약은 **서버 로컬 파일**(prepare 가 씀) — 같은 seed 서버(s1/s4, s3/s5) 가 config 파일을 공유하므로 config 에 박지 않는다 (s5 보고 #2)
                         projection_file=RESERVATION_FILE,                                     # 서버 로컬 run 별 예약 (재배정 §3 식; gate 가 씀) — 있으면 projected_map(4.0h 보수값) 보다 우선
                         **({"training_deadline": training_deadline()} if training_deadline() else {})))            # 공통 절대 마감 — trainer 가 예상 종료 ≤ 마감 을 검사
    if P.get("schedule"):
        k["aligner_schedule"] = dict(P["schedule"])
    if P.get("routing"):                                                     # backend 에 없는 항의 q 는 적지 않는다 (registry: N0 위 qD / R1 위 qK / edge 없는 qE 는 거부)
        rq = {q: v for q, v in P["routing"].items() if (q == "qD" and B["rec"] != "N0") or (q == "qK" and B["rec"] == "R3") or (q == "qE" and B["edge"])}
        if not rq:
            raise SystemExit(f"!! {case}: 정책 {pol_id} 의 routing 이 backend {be_id} 에서 아무 항도 바꾸지 않는다 — J 와 같은 실험 (만들지 않음)")
        k["routing"] = rq
    if P["proto"] == "I-AEQ":
        k["corruption"] = dict(radius_hr=2.0, corruption_seed_offset=2000)
        k["aux"] = dict(offset_weight=float(P["off"]), offset_ramp_updates=0, offset_stop_reference=True, geometry_weight=0.0, ramp_updates=5000, geometry_sigma_hr=2.0, geometry_margin_hr=11)
    else:
        k["aux"] = dict(offset_weight=0.0, geometry_weight=0.0)
        if "radius" in P:                                                    # RC: 계획의 semantic fragment 그대로 corruption.radius_hr 0.0 (registry: I-NATIVE-TRANSFER 에 radius > 0 은 거부)
            k["corruption"] = dict(radius_hr=float(P["radius"]))
    return k


def render(tag, case, seed, server, k, updates, eval_epoch, tpl):
    import yaml
    sp = resolve(k); pol_id, be_id = CASES[case]; sha, step = t0_identity(server)
    t = re.sub(r"^(#.*\n)+", "", tpl)
    head = (f"# {tag} — {PURPOSE[case]}. 생성: tools/gen_pakd50_configs.py (seed {seed} 를 쓰는 서버 {'/'.join(servers_sharing_seed(server)) if seed == SERVER_SEED[server] else server} 공용 — 서버별 값은 config 에 없다). 손으로 고치지 말 것.\n"
            f"# 캠페인 {CAMPAIGN_ID} · protocol {PROTOCOL} · grid {GRID_ID} · 계획 {PLAN} · 요약 {SUMMARY} · 노트 {NOTE}\n"
            f"# 세팅: {describe(sp)} · 정책 {pol_id}(A {'trainable' if sp['aligner_trainable'] else 'frozen'}, offset λ {k.get('aux', {}).get('offset_weight', 0)}, A LR {k['aligner_lr']}) · backend {be_id} (rec {k['rec']['case']}{', EDGE-H λE ' + str(k['stat'].get('outer_weight')) if k['stat'].get('enabled') else ''})\n"
            f"# 약명→세팅: J0/JQ/JR/XJ = A trainable(joint) + N0/Q12/R1/X02 · F0/FQ/FR/XF = A frozen + … · AL0/ALQ = joint, A LR 3e-6 · Q12 = (1+αd)L1 + β(1−d)a|S−T| + λE·signed Scharr edge · R1 = (1+αd)L1 · X02 = R1 + edge\n"
            f"#           D*/LF* = joint 에서 A 를 0–4999 동결 / 25000 부터 동결 (kdv.aligner_schedule; 동결 구간 LO 없음, ε RNG 는 같은 순서) · P*/JK0/JE0 = A 가 직접 받는 항만 제한 (kdv.routing qA=(qD,qK,qE); U 는 backend 전체) · J_Q*/AL_Q* = Q12 계수만\n"
            f"#           J_R3_NOEDGE = J + R3 adaptive(edge 없음) · J_N0_EDGE = J + N0 + λE edge(Teacher 미사용) · RC0/RCQ = A trainable(LR 1e-5) 인데 offset 연습 없음(I-NATIVE-TRANSFER, radius 0) + N0/Q12 — 재배정 {ALLOC_PLAN}\n"
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


def plan_rows(server, measured=None, is_terminal=None, extra=()):
    """재배정 dry-run: 서버 순서(priority_for + extra) 에서 완료·실행 중이 아닌 run 의 예약과 누적 (재배정 §4 표 형식). is_terminal(run 이름) 이 없으면 work_dir 완료 판정."""
    seed = SERVER_SEED[server]; measured = measured if measured is not None else measured_hours_from_ledger(server)
    if is_terminal is None:
        from tools.campaign_gate import terminal as is_terminal          # noqa
    rows, t = [], 0.0
    for it in list(priority_for(server)) + list(extra):
        r = reservation_for(server, it, measured, seed)
        if r is None:
            continue
        r["status"] = "done_or_failed" if is_terminal(r["run"]) else "planned"
        if r["status"] == "planned":
            t += r["reservation_h"]
        r["cumulative_h"] = t; rows.append(r)
    return rows


def plan_table(server):
    rem = hours_to_deadline(); rows = plan_rows(server)
    print(f"[{server}] seed {SERVER_SEED[server]} — 재배정 {ALLOC_PLAN} §4; 예약 = {RESERVE_SLACK}×ref + {RESERVE_POST_H * 60:.0f}min; 학습 마감까지 {'?' if rem is None else '%.2f' % rem} h")
    for r in rows:
        print(f"  {r['case']:<12} {r['status']:<14} ref {r['reference_train_h']:.2f} h ({r['reference_kind']}) → 예약 {r['reservation_h']:.4f} h · 누적 {r['cumulative_h']:.4f} h")
    tot = sum(r["reservation_h"] for r in rows if r["status"] == "planned")
    print(f"  planned {sum(r['status'] == 'planned' for r in rows)} run · 예약 합 {tot:.4f} h" + ("" if rem is None else f" · 마감 안 {'OK' if tot <= rem else '초과 — gate admission 이 뒤를 민다'}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=None, choices=list(SERVER_SEED)); ap.add_argument("--all", action="store_true"); ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--cases", default=None, help="쉼표 목록 (기본: stage 별 목록)"); ap.add_argument("--updates", type=int, default=50000); ap.add_argument("--eval-epoch", type=int, default=5)
    ap.add_argument("--projected-hours", type=float, default=None); ap.add_argument("--version", default="v1"); ap.add_argument("--out-dir", default=os.path.join(ROOT, "config")); ap.add_argument("--no-pin", action="store_true", help="τR/λE 를 숫자로 고정하지 않고 그 서버에서 calibrate")
    ap.add_argument("--seed", type=int, default=None, help="서버 seed 대신 (s4 확인 seed 3407 등; 큐 파일은 만들지 않는다 — extra_priority.txt 에 run 이름을 적을 것)")
    ap.add_argument("--plan", action="store_true", help="재배정 순서·예약(1.10×ref+10/60)·누적 dry-run 표만 (config 를 만들지 않는다; 완료·실행 중 run 은 work_dir 로 제외)")
    a = ap.parse_args()
    if a.plan:
        return plan_table(a.server or open(os.path.join(ROOT, "gspread", "server.txt")).read().strip())
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
                        f"# 나머지는 매 pass 조건부 gate 'pakd50'(work_dir/campaign_gates_enabled.txt) 이 우선순위 {' → '.join(priority_for(srv))} 로 편성한다: λE(J0 S1234 exact50K pilot → tools/pakd50_calibrate.py, {ASSET_CAL_PATH} 사본) 가 없으면 τR 만 필요한 다음 한 벌, 있으면 남은 전부.\n"
                        f"# 예산: 공통 절대 시계 {CLOCK_PATH} (학습 마감 = start+46h, 감사 +4h) — 예상 종료가 마감을 넘기는 run 은 시작하지 않는다 ({LEDGER} 50h/4h). 약명→세팅은 config 머리 주석 / {NOTE}\n"
                        + "\n".join(run_name(c, SERVER_SEED[srv], a.version) for c in QUEUE_STAGE[a.stage]) + "\n")
            print("queue:", os.path.relpath(q, ROOT))
        print(f"[{srv}] " + " ".join(made))
    sha, step = t0_identity(servers[0]); print("T0:", T0_RUN, T0_TAG, "step", step, "sha", (sha or "?")[:16], "| calibration:", {k: cal.get(k) for k in ("tau_R", "lambda_E")})


if __name__ == "__main__":
    main()
