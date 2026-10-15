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
    python tools/gen_pakd50_configs.py --plan [--server s2]                                # 재배정 순서·예약·누적 (dry-run; 완료·실행 중 run 은 work_dir 로 제외)
2026-09-15 s3 추가(research_log/PAN_PAKD50_Latest_Sheet_Analysis_and_S3_Experiments_2026-09-15.md §4–§7): s3(seed 2026) 명시 순서 J_R3_NOEDGE → J_N0_EDGE → LF0 → LFQ → LFX(새 case: LF 일정 + X02 backend),
확인 seed 4321 은 후보별 묶음(§5 표) 최대 3 run, 확인 reference 는 s3 JQ 1.34 h.
2026-09-15 s4 아키텍처 이식(research_log/PAN_PAKD50_S4_W104D121_Architecture_Allocation_2026-09-15.md): Student U 만 **W104·depth[1,2,1]**(T0/A·τR·λE0 고정 이식, branch A104D121_T0FIX_E0_v1),
case NA0(aligner 없음, A-ID/NOALIGN) → J0 → JQ → XJ → F0 @ seed 1234; 편성 항목은 `case@arch`(예: JQ@W104_D121), run 이름 `PAKD50_<case>_<W…_D…>_WV3_T0_S<seed>_FRESH50_v1`.
확인 seed 3407 은 J0@W104 · WIN@W104 · J0@W112 · WIN@W112 (WIN ∈ {JQ, XJ, F0}; 최대 4).
2026-09-15 QEDGE9(research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md): W104·D121 에서 **q 기반 GT edge gate** — 새 case QE50(고정 T0 aligner 의 q < θq 인 patch 만 GT edge; Q12 hard/soft 그대로) ·
QEC(모든 patch edge × 상수 c_E, s4 W104 J0 S1234 exact50K pilot 로 산출) · QES(gate 를 e decile × aug state stratum 안에서 고정 permutation 51515 로 셔플). 새 논리 캠페인 QEDGE9_A104D121_20260915_v1 / branch A104D121_T0FIX_QEDGE9_v1 —
PAKD50 의 50h·09-16 절대 마감을 상속하지 않는다(soft target 9h, kdv.budget.time_policy). s5(2026·777 두 seed) J0→JQ→QE50 ×2 · s1(1234, v2) J0→JQ→QE50→QES→QEC (전부 @W104_D121). cue 자산 tools/qedge9_cue.py.
QEGX (2026-09-15 저녁, research_log/PAN_QEGX_S3_S4_W104D121_Experiment_Plan_2026-09-15.md): s3(2026·4321, 15 run **v2**) q-edge × soft(QX50/QEC3/QES/β0.05) · s4(1234·3407, 14 run v1) edge 수신 모듈(QER50/QERS = kdv.edge_route)·LF 결합.
캠페인 QEGX_A104D121_S3S4_20260915_v1 / branch A104D121_T0FIX_QEGX_v1, 시간 상한 없음(자체 ledger, required 경고만). 전환 tools/qegx_switch.sh, QEC3 pilot tools/qedge9_cue.py pilot --branch qegx."""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve, describe

CAMPAIGN_ID = "PAKD50_W112D123_WV3_20260914_v1"; PROTOCOL = "FRESH50"; GRID_ID = "GRID1010_50K_v1"
# Student 골격 (2026-09-15 s4 이식): 기본 W112_D123 = T0 와 같은 골격. 다른 골격은 U 만 바꾸고 T0(A+U)·τR·λE0 는 그대로(coefficient transfer) — parent campaign_id 는 유지, experiment_branch_id 로 구분.
ARCH_DEFAULT = "W112_D123"
ARCHS = {"W112_D123": dict(width=112, depth=[1, 2, 3], params_m=2.6589, branch=None, note="기본 골격 (T0 와 같음)"),
         "W104_D121": dict(width=104, depth=[1, 2, 1], params_m=1.9036, branch="A104D121_T0FIX_E0_v1", note="s4 이식: Student U 만 W104·depth[1,2,1] (backbone 1.9036 M; s3 GT-only 기록과 같은 model_args), T0/A·τR·λE0 고정")}
ARCH_LABEL = {"W112_D123": "", "W104_D121": "A104D121"}        # 시트 X열 토큰 (기본 골격은 표기 없음)
# QEDGE9 (2026-09-15): 새 논리 캠페인·branch — Teacher/architecture parent 는 PAKD50 그대로, 시간 정책만 분리(§0.7·§9.3). 실행명은 업로더 호환을 위해 PAKD50 prefix·골격 토큰 유지(§11.1).
QEDGE9_PLAN = "research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md"; QEDGE9_NOTE = "research_log/2026-09-15_qedge9-implementation.md"
QEDGE9_CAMPAIGN_ID = "QEDGE9_A104D121_20260915_v1"; QEDGE9_BRANCH = "A104D121_T0FIX_QEDGE9_v1"; QEDGE9_ARCH = "W104_D121"; QEDGE9_CASES = ("QE50", "QEC", "QES")
QEDGE9_CUE_ASSET = "assets/qedge9/cue_T0_AXIS16_v1.json"; QEDGE9_CE_FILE = "work_dir/_qedge9/qec_cE.json"                 # cue 자산(git; s1 이 만들고 hash 로 전달) · c_E 는 서버 로컬(s4 pilot)
QEDGE9_LEDGER = "work_dir/_qedge9_budget/ledger.json"; QEDGE9_MANDATORY_FILE = "work_dir/_qedge9/mandatory_runs.txt"; QEDGE9_SOFT_HOURS = 9.0; QEDGE9_RESERVED_H = 1.5
QEDGE9_TIME_POLICY = dict(mode="soft_target_only", target_elapsed_hours=QEDGE9_SOFT_HOURS, hard_deadline=None, inherit_parent_deadline=False)   # §9.3: trainer 는 required=True 로 경고만, training_deadline 없음
QEDGE9_PILOT_RUN = "PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v1"; QEDGE9_PILOT_TAG = "last"; QEDGE9_PILOT_STEP = 50000   # §6.1 QEC pilot (W104 J0 S1234 exact50K) — cue 도구·trainer 가 identity 를 강제
# 2026-09-15 17:20 사용자 결정: seed 1234 묶음은 **s4 대신 s1** 이 돈다(s5 는 그대로). s1 에는 W104 control 이 없으므로 J0/JQ@W104 S1234 를 s1 이 새로 학습하고(같은 정의, 다른 호스트), 이름은 s4 의 E0 config(v1) 와 겹치지 않게 **v2**.
# QEC 의 pilot 도 그 서버의 J0 exact50K (s1: v2). s4 의 QEDGE9 항목은 편성에서 뺀다(E0 allocation 만 계속).
QEDGE9_VERSION_BY_SERVER = {"s1": "v2", "s5": "v1", "s4": "v1"}
QEDGE9_PILOT_BY_SERVER = {"s4": QEDGE9_PILOT_RUN, "s1": "PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v2"}; QEDGE9_PILOT_RUNS = set(QEDGE9_PILOT_BY_SERVER.values())
# QEGX (2026-09-15 저녁; research_log/PAN_QEGX_S3_S4_W104D121_Experiment_Plan_2026-09-15.md): s3(seed 2026 + 사전 고정 확인 4321) 는 q-edge × soft(QX50/QEC3/QES/β0.05), s4(1234 + 3407) 는 q-edge 의 수신 모듈(QER50/QERS/JE0)·LF 결합.
# 전부 W104·D121·T0 고정. 시간 상한 없음(자체 ledger, required 경고만, PAKD50 절대 마감·QEDGE9 9h 미상속). s3 묶음은 s5 QEDGE9 의 J0/JQ/QE50@W104 S2026 **v1** config 와 이름이 겹치므로 15 run 전부 **v2**;
# s4 는 기존 E0 J0/JQ/XJ/NA0/F0 v1 을 검증(verified_complete) 뒤 재사용하고 신규 14 run 은 v1. 전제 불통과면 control 을 **v3** 로 새로(s1 의 QEDGE9 v2 와도 겹치지 않게; tools/qegx_switch.sh --refresh-controls).
QEGX_PLAN = "research_log/PAN_QEGX_S3_S4_W104D121_Experiment_Plan_2026-09-15.md"; QEGX_NOTE = "research_log/2026-09-15_qegx-implementation.md"
QEGX_CAMPAIGN_ID = "QEGX_A104D121_S3S4_20260915_v1"; QEGX_BRANCH = "A104D121_T0FIX_QEGX_v1"; QEGX_ARCH = QEDGE9_ARCH
QEGX_LEDGER = "work_dir/_qegx_budget/ledger.json"; QEGX_MANDATORY_FILE = "work_dir/_qegx/mandatory_runs.txt"; QEGX_CE_FILE = "work_dir/_qegx/qec3_cE.json"      # QEC3 의 c_E3 (s3 pilot) — s1 QEDGE9 QEC 파일과 별도 (§4.2)
QEGX_SOFT_HOURS = 1000.0; QEGX_RESERVED_H = 1.8                                                                                                 # 상한 없음을 trainer 예산 gate 에 사상: 사실상 도달 불가능한 total + required(경고만) (§9.3·§12)
QEGX_TIME_POLICY = dict(mode="no_hard_limit", target_elapsed_hours=None, hard_deadline=None, inherit_parent_deadline=False)
QEGX_VERSION_BY_SERVER = {"s3": "v2", "s4": "v1"}
QEGX_PILOT_BY_SERVER = {"s3": "PAKD50_J0_W104_D121_WV3_T0_S2026_FRESH50_v2"}; QEGX_PILOT_RUNS = set(QEGX_PILOT_BY_SERVER.values())     # QEC3 pilot = s3 J0@W104 S2026 exact50K (§4.2); 다른 서버에 QEC3 없음
QEGX_CASES = ("QX50", "QEC3", "QE50_B005", "LFQE50", "QER50", "QERS")                # QEGX 전용 case (다른 branch 로 만들지 않는다); QE50/QES/J_QB005/JE0/LF*/J_R3_NOEDGE/XJ/JQ/J0 는 기존 정의 재사용
QEGX_REFRESH_CONTROLS = {"s4": [f"PAKD50_{c}_W104_D121_WV3_T0_S1234_FRESH50_v3" for c in ("J0", "JQ", "XJ")]}     # §6 전제 불통과 시 대조 새로고침(약 3h8m + XJ); QEGX branch 로 편성(extra_priority)
CUE_CASES = ("QE50", "QEC", "QES", "QX50", "QEC3", "QE50_B005", "LFQE50", "QER50", "QERS")   # cue 자산(θq/gate 표) 이 있어야 시작하는 case (gate 의 cue_ready)
ALL_PILOT_RUNS = QEDGE9_PILOT_RUNS | QEGX_PILOT_RUNS


def server_id(raw):
    """gspread/server.txt 의 표시 문자열 → 내부 server id: 's3(5090)' → 's3' (QEGX §12; 시트 탭 이름은 표시 문자열 그대로 둔다)."""
    return re.sub(r"\s*\(.*\)\s*$", "", (raw or "").strip())
RUN_RE = re.compile(r"^PAKD50_(?P<case>.+?)_(?P<arch>W\d+_D\d+)_WV3_T0_S(?P<seed>\d+)_(?P<proto>[A-Z0-9]+)_(?P<ver>v\d+)$")
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
          "RC": dict(pol="A-FT", proto="I-NATIVE-TRANSFER", off=0.0, alr=1e-5, radius=0.0),
          # s4 아키텍처 이식 §4: NA0 = aligner 없음(A-ID/NOALIGN, native 입력, PAN warp 없음 NA-STRICT), plain GT. T0 는 평가 bin 전용(eval_only) — no-align 모델에 Teacher 의 aligned PAN 을 넣지 않는다
          "NA": dict(pol="A-ID", proto="I-A", off=0.0, alr=None, noalign=True)}
BASELINE_OF = {"J": "J0", "F": "F0", "AL": "AL0", "D": "D0", "DP": "D0", "LF": "LF0", "P": "J0", "JK0": "J0", "JE0": "J0", "RC": "RC0", "NA": "NA0"}   # no-KD control (배정 §5–§6; RC 는 재배정 §5; NA 는 s4 이식 §4)
BACKEND = {"N0": dict(rec="N0", edge=False), "R1": dict(rec="R1", edge=False), "Q12": dict(rec="R3", edge=True), "X02": dict(rec="R1", edge=True),
           # Q12 단일축 scalar variant (s4 배정 §6; 상위 계획 C1 목록의 QA05/QB005/QB02/QE025/QE10 — 계수만 바뀌고 loss 정의는 같다): α = rec.alpha(L_D), β = rec.kd_weight(L_K), λE = λE0 × lam_mult
           "Q12_A05": dict(rec="R3", edge=True, alpha=0.5), "Q12_B005": dict(rec="R3", edge=True, kd_weight=0.05), "Q12_B02": dict(rec="R3", edge=True, kd_weight=0.2),
           "Q12_E025": dict(rec="R3", edge=True, lam_mult=0.5), "Q12_E10": dict(rec="R3", edge=True, lam_mult=2.0),
           # 재배정 2026-09-15 §5 (s2/s4 backend 성분 전체 제거): R3_NOEDGE = 재가중 GT hard + adaptive soft, GT edge 없음 (JQ 에서 λE L_E 제거) · N0_EDGE = plain GT L1 + λE GT edge (Teacher 를 학습 loss 에 쓰지 않음 → eval_only)
           "R3_NOEDGE": dict(rec="R3", edge=False), "N0_EDGE": dict(rec="N0", edge=True),
           # QEDGE9 (2026-09-15 §3·§6): Q12(R3 adaptive + EDGE-H, α1 β0.1 λE0) 그대로인데 edge 의 적용 patch 만 gate — low_q(q_T<θq) / const(c_E) / shuffle(stratum 내 permutation). kdv.edge_gate (registry 가 EDGE-H 위에서만 허용)
           "QE50": dict(rec="R3", edge=True, edge_gate="low_q"), "QEC": dict(rec="R3", edge=True, edge_gate="const"), "QES": dict(rec="R3", edge=True, edge_gate="shuffle"),
           # QEGX (2026-09-15 §4): QX50 = QE50 에서 output soft 만 제거(R1 hard-only + gated edge; QX50 ≠ XJ) · QE50_B005 = QE50 에서 β 0.1→0.05 · QEC3 = const 대조(c_E3 = s3 J0@W104 S2026 exact50K pilot; 별도 파일)
           # QER50/QERS = **edge_route**(§4.5): total 은 all-edge E(1) 그대로(U 는 모든 patch 의 GT edge), A 의 .grad 에서 λE·mean((1−g)E_i) 를 뺀다(A 는 q_T<θq / shuffle patch 의 edge 만) — routing.qE·edge_gate 와 별도 경로, 둘과 결합하지 않는다
           "QX50": dict(rec="R1", edge=True, edge_gate="low_q"), "QE50_B005": dict(rec="R3", edge=True, edge_gate="low_q", kd_weight=0.05), "QEC3": dict(rec="R3", edge=True, edge_gate="const3"),
           "QER50": dict(rec="R3", edge=True, edge_route="low_q"), "QERS": dict(rec="R3", edge=True, edge_route="shuffle")}
CASES = {"J0": ("J", "N0"), "JQ": ("J", "Q12"), "JR": ("J", "R1"), "XJ": ("J", "X02"), "F0": ("F", "N0"), "FQ": ("F", "Q12"), "FR": ("F", "R1"), "XF": ("F", "X02"), "AL0": ("AL", "N0"), "ALQ": ("AL", "Q12")}
for _p in ("J", "AL"):                                    # J_QA05 … J_QE10 (anchor JQ, no-KD J0) · AL_QA05 … AL_QE10 (§7 결합: anchor ALQ, no-KD AL0)
    for _v in ("QA05", "QB005", "QB02", "QE025", "QE10"):
        CASES[f"{_p}_{_v}"] = (_p, "Q12_" + _v[1:])
CASES.update({"D0": ("D", "N0"), "DQ": ("D", "Q12"), "DR": ("D", "R1"), "DX": ("D", "X02"), "PQ": ("P", "Q12"), "PR": ("P", "R1"), "PX": ("P", "X02"),
              "DPQ": ("DP", "Q12"), "DPX": ("DP", "X02"), "LF0": ("LF", "N0"), "LFQ": ("LF", "Q12"), "JK0": ("JK0", "Q12"), "JE0": ("JE0", "Q12"),
              "J_R3_NOEDGE": ("J", "R3_NOEDGE"), "J_N0_EDGE": ("J", "N0_EDGE"), "RC0": ("RC", "N0"), "RCQ": ("RC", "Q12"),     # 재배정 2026-09-15 §5
              "LFX": ("LF", "X02"), "NA0": ("NA", "N0"), "QE50": ("J", "QE50"), "QEC": ("J", "QEC"), "QES": ("J", "QES"),
              "QX50": ("J", "QX50"), "QEC3": ("J", "QEC3"), "QE50_B005": ("J", "QE50_B005"), "LFQE50": ("LF", "QE50"), "QER50": ("J", "QER50"), "QERS": ("J", "QERS")})     # QEGX 2026-09-15 §4                                                                                              # s3 추가 2026-09-15 §4.3: 0–24999 XJ 와 같은 joint, 25000 부터 A 동결(offset 중단), U 는 X02(R1 + λE edge; soft 없음) 50K
_PURPOSE_S5 = {"D0": "s5: 초기 5K A 동결 뒤 joint, N0 (delayed-joint 의 no-KD control)", "DQ": "s5: 초기 5K A 동결, U 는 처음부터 Q12 (delayed joint KD)", "DR": "s5 fallback: D 일정 + R1", "DX": "s5: DQ 의 soft 제거 대조 (D 일정 + X02)",
                "PQ": "s5: protected routing — A 는 L0+LO 만, U 는 Q12 전체", "PR": "s5 fallback: protected + R1", "PX": "s5: PQ 의 soft 제거 대조 (protected + X02)",
                "DPQ": "s5: delayed + protected 결합 (Q12)", "DPX": "s5: DPQ 의 soft 제거 대조", "LF0": "s5: 25K 이후 A 동결, N0 (late-freeze control)", "LFQ": "s5: 25K 이후 A 동결, Q12",
                "JK0": "s5: JQ 에서 A 로 가는 L_K(soft) 만 차단", "JE0": "s5: JQ 에서 A 로 가는 λE L_E 만 차단",
                # 재배정 2026-09-15 §5: JR = weighted_GT · J_R3_NOEDGE = weighted_GT + adaptive soft · XJ = weighted_GT + λE edge · JQ = 셋 다 · J_N0_EDGE = plain GT + λE edge (Teacher 없이)
                "J_R3_NOEDGE": "재배정 s2/s4: joint + R3 adaptive soft, GT edge 전체 제거 (JQ − edge)", "J_N0_EDGE": "재배정 s2: joint + plain GT L1 + λE GT edge, Teacher 를 loss 에 쓰지 않음 (Teacher-free edge 대조)",
                "RC0": "재배정 s4/s5: A trainable(LR 1e-5) 인데 Student 단계 offset 연습 없음(native 만), N0 (RC 의 no-KD control)", "RCQ": "재배정 s4/s5: RC 정합 정책 + Q12 전체",
                "LFX": "s3: 25K 이후 A 동결 + X02(재가중 GT hard + λE GT edge, output soft 없음) — XJ 의 LF 판 (control LF0; 비교 LFX−XJ, LFQ−LFX)",
                "NA0": "s4 이식: aligner 없음(A-ID/NOALIGN, PAN 원본 직접 입력), plain GT L1 — 같은 서버·현재 규약의 no-align 기준 (Teacher 는 평가 bin 전용)",
                "QE50": "QEDGE9 주력: joint + Q12 hard/soft 그대로(wH≥1), GT edge 는 고정 T0 aligner 의 q(AXIS16) < θq(train calibration 중앙값) 인 patch 만 — λE·Σ g_i E_i / B (재정규화 없음)",
                "QEC": "QEDGE9 대조: 모든 patch edge × 상수 c_E = Σ g E_pilot / Σ E_pilot (pilot = s4 W104 J0 S1234 exact50K, train calibration view) — 단순 edge 총강도 감소",
                "QES": "QEDGE9 대조: low_q gate 를 (T0 e_T_roi32 decile × aug state) stratum 안에서 고정 permutation(51515) 으로 셔플(active 수 보존) — q–sample 연결의 필요성",
                "QX50": "QEGX s3: QE50 에서 output soft 만 제거 — hard (1+αd_T)L1(R1) + q_T<θq patch 의 GT edge(λE·Σ g_i E_i / B) + offset; QX50 ≠ XJ(all-edge). 'QE50 에서 β=0' 의 동치",
                "QE50_B005": "QEGX s3: QE50 에서 β 만 0.1 → 0.05 (α·τR·λE·θq·A LR 동일; J_QB005 와 대응쌍 — 완전 제거(QX50)/절반/기본 을 좁게 비교)",
                "QEC3": "QEGX s3 상수 대조: 모든 patch edge × c_E3 = Σ g E_pilot / Σ E_pilot (pilot = s3 J0@W104 S2026 exact50K, calibration view; work_dir/_qegx/qec3_cE.json) — s1 QEDGE9 QEC 파일과 별도",
                "LFQE50": "QEGX s4: 0–24999 는 QE50(q-low gated edge + Q12) 과 같은 학습, 25000 부터 A 완전 동결(offset 중단; ε RNG 규약은 LF 와 같음); U 는 gated edge·Q12 로 50K 까지 — LFQ 와 대응",
                "QER50": "QEGX s4 핵심(edge_route low_q): U 에는 모든 patch 의 GT edge(E(1)), A 에는 q_T<θq patch 의 edge 만 — total 은 JQ 와 같고 A .grad 에서 λE·mean((1−g)E_i) 를 뺀다; JQ(1/1)·JE0(1/0)·QE50(g/g) 과 대응",
                "QERS": "QEGX s4 대조(edge_route shuffle): QER50 의 A gate 를 QES 와 같은 stratum 셔플 라벨로 — routing 의 실제 q–sample 연결 대조"}
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
# QEDGE9 (2026-09-15 §5–§6): s5 = 두 고유 seed(2026 서버 seed · 777) 의 J0→JQ→QE50 @W104 (fresh50K 셋; 체인 학습 아님) · s4 = 기존 JQ/XJ/F0@W104 잔여 완료 뒤 QE50→QEC→QES @W104 (seed 1234).
# 항목이 전체 run 이름이면 그 seed 로(777 은 s5 의 허용 seed — 확인 seed 가 아니다: allowed_seeds). 완료된 run 은 편성에서 빠지고, cue 자산(θq / c_E) 이 없는 gate run 은 gate 가 그 pass 에서 건너뛴다(cue_ready).
PRIORITY_BY_SERVER = {"s2": ["JR", "XJ", "J_R3_NOEDGE", "J_N0_EDGE"],
                      "s5": ["J0@W104_D121", "JQ@W104_D121", "QE50@W104_D121", "PAKD50_J0_W104_D121_WV3_T0_S777_FRESH50_v1", "PAKD50_JQ_W104_D121_WV3_T0_S777_FRESH50_v1", "PAKD50_QE50_W104_D121_WV3_T0_S777_FRESH50_v1"],
                      # QEGX (2026-09-15 저녁 §5–§6): s3 15 run(전부 v2; seed 2026 A/B 10 + 4321 C 5) · s4 = 기존 E0 5 항목(완료분은 terminal 로 건너뜀) 뒤 신규 14 run(v1; 1234 A 8 + 3407 C 6). 신규는 전부 run 이름 항목(seed·version 포함).
                      "s3": [f"PAKD50_{c}_W104_D121_WV3_T0_S2026_FRESH50_v2" for c in ("J0", "JQ", "QE50", "XJ", "QX50", "J_R3_NOEDGE", "QEC3", "QES", "J_QB005", "QE50_B005")]
                            + [f"PAKD50_{c}_W104_D121_WV3_T0_S4321_FRESH50_v2" for c in ("J0", "JQ", "XJ", "QE50", "QX50")],
                      "s4": ["NA0@W104_D121", "J0@W104_D121", "JQ@W104_D121", "XJ@W104_D121", "F0@W104_D121"]      # s4 아키텍처 이식 (2026-09-15 §4; 전부 완료 → 편성에서 terminal)
                            + [f"PAKD50_{c}_W104_D121_WV3_T0_S1234_FRESH50_v1" for c in ("QE50", "LF0", "LFQ", "LFQE50", "LFX", "JE0", "QER50", "QERS")]
                            + [f"PAKD50_{c}_W104_D121_WV3_T0_S3407_FRESH50_v1" for c in ("J0", "JQ", "QE50", "JE0", "QER50", "QERS")],
                      "s1": ["PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v2", "PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v2", "PAKD50_QE50_W104_D121_WV3_T0_S1234_FRESH50_v2", "PAKD50_QES_W104_D121_WV3_T0_S1234_FRESH50_v2", "PAKD50_QEC_W104_D121_WV3_T0_S1234_FRESH50_v2"]}   # QEDGE9 s1(seed 1234): control 부터 새로(v2), QEC 는 pilot(J0 v2 exact50K) 뒤 마지막
MANDATORY_BY_SERVER = dict(PRIORITY_BY_SERVER)        # 기본 run 전부가 예산 예약 대상 (완료된 run 은 trainer 가 0 으로 센다)
QEDGE9_ITEMS_BY_SERVER = {"s5": list(PRIORITY_BY_SERVER["s5"]), "s1": list(PRIORITY_BY_SERVER["s1"])}     # 이 항목들만 QEDGE9 branch(campaign/budget); s4 는 기존 E0 allocation 그대로(QEDGE9 없음)
QEGX_ITEMS_BY_SERVER = {"s3": list(PRIORITY_BY_SERVER["s3"]), "s4": [it for it in PRIORITY_BY_SERVER["s4"] if it.startswith("PAKD50_")] + QEGX_REFRESH_CONTROLS["s4"]}     # QEGX branch 항목 (s4 의 E0 5 항목 제외; 대조 새로고침 v3 포함)
PREVIOUS_PRIORITY_BY_SERVER = {"s4": ["F0", "RC0", "RCQ", "JR", "XJ", "J_R3_NOEDGE"], "s4_20260914": ["J0", "JQ", "AL0", "ALQ"], "s5": ["J0", "JQ", "D0", "DQ", "PQ"], "s3": list(PRIORITY),
                               "s5_20260915": ["PQ", "F0", "LF0", "LFQ", "RC0", "RCQ"],      # s5 재배정(2026-09-15 아침) — 시트 확인 결과 전부 완료 → QEDGE9 로 교체
                               "s3_20260915": ["J_R3_NOEDGE", "J_N0_EDGE", "LF0", "LFQ", "LFX"],   # s3 추가(2026-09-15 낮; W112 seed 2026) — 완료(§9.1 실측 있음) → QEGX 로 교체
                               "s4_20260915_e0": ["NA0@W104_D121", "J0@W104_D121", "JQ@W104_D121", "XJ@W104_D121", "F0@W104_D121"]}   # s4 E0 이식 묶음 — 완료; 현재 순서 앞에 그대로 남겨 control 검증·terminal 표시
ALLOC_PLAN = "research_log/PAN_PAKD50_S2_S4_S5_Derived_Run_Allocation_2026-09-15.md"; ALLOC_PLAN_S3 = "research_log/PAN_PAKD50_Latest_Sheet_Analysis_and_S3_Experiments_2026-09-15.md"
ALLOC_PLAN_S4 = "research_log/PAN_PAKD50_S4_W104D121_Architecture_Allocation_2026-09-15.md"
ALLOCATED_SERVERS = ("s2", "s3", "s4", "s5")
# 시간 산정 (재배정 §2–§3): 같은 서버 완료 case 의 Sheet Train(h)(2026-09-15 00:05 live read; 새 case 실측이 아니라 **계획 기준값**) — N0 형(rec N0·edge 없음) 은 J0, Teacher/Q12 형은 JQ 의 값.
# s2 는 이번 4 case 전부 2.33 (보수). s5 D0 1.15 는 일반화하지 않는다. routing case(PQ 등) 는 완료 기록이 없어 1.80h 가예약 — 같은 서버에서 첫 실측이 나오면 그것으로 바꾼다 (reference_hours).
REFERENCE_TRAIN_H = {"s2": dict(N0=1.97, T=2.33, all=2.33), "s4": dict(N0=1.17, T=1.39), "s5": dict(N0=1.35, T=1.36), "s3": dict(N0=1.16, T=1.34)}     # s3: J0 1.16 / JQ 1.34 (§6)
REFERENCE_CASE_TRAIN_H = {"s3": {"LFX": 1.33,                       # case 별 대용값 (s3 §4: LFX 는 XJ 1.33) — 유형 표보다 우선
                                 # QEGX §9.1: s3 W104 는 미측정 → 같은 유형의 s3 W112 Sheet Train(h) 대용(J0 1.16 / JQ 1.34 / XJ 1.33 / J_R3_NOEDGE 1.30; QEC3·J_QB005 는 JQ), cached-gate run 은 1.50h 임시(*) — 실측 아님
                                 "J0@W104_D121": 1.16, "JQ@W104_D121": 1.34, "XJ@W104_D121": 1.33, "QE50@W104_D121": 1.50, "QX50@W104_D121": 1.50, "J_R3_NOEDGE@W104_D121": 1.30,
                                 "QEC3@W104_D121": 1.34, "QES@W104_D121": 1.50, "J_QB005@W104_D121": 1.34, "QE50_B005@W104_D121": 1.50},
                          # s4 W104: QEGX §9.1 에서 읽은 **Sheet Train(h) 실측**(NA0 1.27 / J0 1.17 / JQ 1.38 / XJ 1.37 / F0 1.13; 종전 큰 골격 대용값 1.18/1.18/1.40/1.39/1.14 교체) · LF* 는 같은 유형(N0→J0, Q12→JQ, X02→XJ) ·
                          # cached-gate 1.50h(*) · routing(JE0/QER50/QERS) 1.80h(*) 임시 기준 — 첫 해당 run 실측(case@arch, 같은 서버 ledger) 이 나오면 자동 교체(measured_same_case)
                          "s4": {"NA0@W104_D121": 1.27, "J0@W104_D121": 1.17, "JQ@W104_D121": 1.38, "XJ@W104_D121": 1.37, "F0@W104_D121": 1.13,
                                 "QE50@W104_D121": 1.50, "QEC@W104_D121": 1.40, "QES@W104_D121": 1.50,                                             # QEDGE9 §9.1–9.2 (s4 QEDGE9 항목은 s1 로 이관됐지만 키는 보존)
                                 "LF0@W104_D121": 1.17, "LFQ@W104_D121": 1.38, "LFQE50@W104_D121": 1.50, "LFX@W104_D121": 1.37, "JE0@W104_D121": 1.80, "QER50@W104_D121": 1.80, "QERS@W104_D121": 1.80},
                          "s5": {"J0@W104_D121": 1.35, "JQ@W104_D121": 1.36, "QE50@W104_D121": 1.50},                                              # QEDGE9 §9.2: s5 W112 J0/JQ 관측 대용 + QE50 1.50(*); 예약 합 10.2620 h
                          "s1": {"J0@W104_D121": 1.18, "JQ@W104_D121": 1.40, "QE50@W104_D121": 1.50, "QEC@W104_D121": 1.40, "QES@W104_D121": 1.50}}                 # QEDGE9 s1 (17:20): s4 W104 관측(J0 1.17/JQ 1.38)·계획 §9.2 값 대용 — s1 W104 실측 없음; 예약 합 8.1633 h
ROUTING_PLACEHOLDER_H = 1.80; CONFIRM_PLACEHOLDER_H = 1.80          # 미실측 경로 가예약 (실측·성능 예측이 아니다; 결과 수치로 기록하지 않는다)
CONFIRM_REFERENCE_H = {"s3": 1.34}                                   # 확인 run 의 서버별 공통 대용값 (s3 §6: JQ 1.34; 상한 보장 아님) — 없으면 CONFIRM_PLACEHOLDER_H
CONFIRM_REFERENCE_CASE_H = {"s4": {"J0": 1.18, "*": 1.40}}           # s4 이식 §8.2: 확인 4 run 은 J0 1.18 · WIN 1.40 (골격 무관 대용값)
CONFIRM_MAX_RUNS_BY_SERVER = {"s4": 4}
RESERVE_SLACK = 1.10; RESERVE_POST_H = 10.0 / 60.0                   # reservation_h = 1.10 × reference_train_h + 10/60 (10 % 변동 여유 + run 뒤 export/업로드/전환 10 분 가예약)
CONFIRM_SEED = {"s4": 3407, "s5": 9091, "s3": 4321}; CONFIRM_MAX_RUNS = 3   # §7: 외부 분석의 WIN 확정 뒤에만, 서버당 최대 3 run (s2/s4/s5: WIN · 같은 policy 의 no-KD control · F0; s3: §5 표의 후보별 묶음)
CONFIRM_BUNDLE_BY_SERVER = {"s3": {"LFQ": ["JQ", "LF0", "LFQ"], "LFX": ["XJ", "LF0", "LFX"], "J_N0_EDGE": ["J0", "XJ", "J_N0_EDGE"], "J_R3_NOEDGE": ["J0", "JR", "J_R3_NOEDGE"], "XJ": ["J0", "JR", "XJ"], "JQ": ["J0", "XJ", "JQ"]},
                            "s4": {w: ["J0@W104_D121", f"{w}@W104_D121", "J0", w] for w in ("JQ", "XJ", "F0")}}     # s4 이식 §6: 두 골격 × (J0, WIN); 종전 s4 확인 예약(J0/JQ/WIN@W112) 을 대체
RESERVATION_FILE = "work_dir/_pakd50/reservations.json"             # 서버 로컬: run → gate_hours (= reservation_h / MARGIN; trainer 가 margin 을 곱하면 reservation_h). gate 가 매 pass 갱신
STAGE_BY_SERVER = {"s4": {1: ["J0", "AL0"], 2: ["JQ", "ALQ"]}, "s5": {1: ["J0", "D0"], 2: ["JQ", "DQ", "PQ"]}}     # --stage 용 (2026-09-14 기본 묶음; 재배정 목록은 --cases / 편성은 gate)
EXTRA_PRIORITY_FILE = "work_dir/_pakd50/extra_priority.txt"
MANDATORY_FILE = "work_dir/_pakd50/mandatory_runs.txt"      # 서버 로컬: 이 서버의 기본 묶음 run 이름 (trainer 예산 gate 의 remaining_mandatory 출처; prepare 가 mandatory_for(server) 로 쓴다)


def allowed_seeds(server):
    """이 서버가 확인 seed 가 아닌 본 실험으로 돌리는 seed 집합: 서버 seed + 명시 순서에 전체 run 이름으로 들어 있는 seed (QEDGE9 s5 의 777)."""
    out = {SERVER_SEED[server]} if server in SERVER_SEED else set()
    for it in PRIORITY_BY_SERVER.get(server, []):
        if is_tag(it):
            out.add(seed_of(it))
    return out


def branch_for(server, item, seed=None):
    """편성 항목의 branch: QEGX 전용 case(QEGX_CASES) 이거나 서버 QEGX 목록(QEGX_ITEMS_BY_SERVER) 의 run 이면 'QEGX';
    QEDGE9 case(QE50/QEC/QES) 이거나 서버 QEDGE9 목록의 run 이면 'QEDGE9'; 아니면 None(기본 PAKD50 계약).
    QEGX §12 검사: s3/s4 큐의 QE50/QES 는 run 이름 항목이라 QEGX 목록이 먼저 잡힌다 — bare 'QE50@W104_D121' 은 QEDGE9 자동 규칙(9h ledger) 으로 가므로 QEGX 큐에는 run 이름만 쓴다."""
    case, arch, sd, ver = parse_item(item)
    if case in QEGX_CASES:
        return "QEGX"
    sd = sd if sd is not None else (seed if seed is not None else SERVER_SEED.get(server))
    tag = run_name(case, sd, ver or "v1", arch=arch)
    if tag in {to_tag(x, SERVER_SEED[server]) for x in QEGX_ITEMS_BY_SERVER.get(server, [])}:
        return "QEGX"
    if case in QEDGE9_CASES:
        return "QEDGE9"
    return "QEDGE9" if tag in {to_tag(x, SERVER_SEED[server]) for x in QEDGE9_ITEMS_BY_SERVER.get(server, [])} else None


def cue_ready(item):
    """QEDGE9 gate run 의 자산 준비 여부 (§11.3: θq/c_E 미산출이면 대기, placeholder 0 으로 학습하지 않는다) — gate 는 준비 안 된 run 을 그 pass 에서 건너뛴다."""
    case = case_of(item)
    if case not in CUE_CASES:
        return True
    j = os.path.join(ROOT, QEDGE9_CUE_ASSET)
    if not os.path.exists(j):
        return False
    try:
        m = json.load(open(j)); npz = m.get("npz")
    except Exception:
        return False
    if not npz or not os.path.exists(os.path.join(ROOT, npz)) or m.get("theta_q") is None:
        return False
    if case in ("QEC", "QEC3"):                                              # const 대조는 자기 branch 의 c_E 파일 (QEC: s1/s4 QEDGE9 · QEC3: s3 QEGX — 서로 대신하지 않는다)
        c = os.path.join(ROOT, QEDGE9_CE_FILE if case == "QEC" else QEGX_CE_FILE)
        return os.path.exists(c) and (json.load(open(c)).get("c_E") is not None)
    return True


def write_mandatory_file(server, version="v1"):
    p = os.path.join(ROOT, MANDATORY_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
    runs = [to_tag(c, SERVER_SEED[server], version) for c in mandatory_for(server)]      # 항목이 case@arch 면 그 골격의 run 이름
    with open(p, "w") as f:
        f.write(f"# {server} 기본 묶음 (예산 gate 예약; 완료된 run 은 trainer 가 0 으로 센다) — gen_pakd50_configs.write_mandatory_file\n" + "\n".join(runs) + "\n")
    return runs


INIT_HASH_PATH = "assets/pakd50/init_hashes.json"          # seed → 저장 U 초기값(init_unet_seed<seed>.pt) 의 tensor sha256_16. 값이 있으면 config expect_init 으로 박혀 trainer 가 fail-fast (s5 보고 #3)


def init_hash_for(seed, arch=ARCH_DEFAULT):
    """seed 별 저장 U 초기값 hash — 골격별 namespace (기본 골격 'unet', 그 외 'unet@<arch>'): W112 hash 를 W104 에 강제하지 않는다 (s4 이식 §9.1)."""
    ns = "unet" if arch == ARCH_DEFAULT else f"unet@{arch}"
    return ((_load_json(INIT_HASH_PATH) or {}).get(ns) or {}).get(str(int(seed)))


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


def parse_item(item):
    """편성 항목 → (case, arch, seed|None, version|None). 항목 형식: 'JQ' · 'JQ@W104_D121' · 전체 run 이름 PAKD50_<case>_<W…_D…>_WV3_T0_S<seed>_<proto>_<ver>."""
    if is_tag(item):
        m = RUN_RE.match(item)
        if not m:
            raise ValueError(f"run 이름 형식이 아니다: {item}")
        return m.group("case"), m.group("arch"), int(m.group("seed")), m.group("ver")
    case, _, arch = item.partition("@"); arch = arch or ARCH_DEFAULT
    if arch not in ARCHS:
        raise ValueError(f"알 수 없는 골격 {arch} (항목 {item}); 등록: {list(ARCHS)}")
    return case, arch, None, None


def to_tag(item, seed, version="v1"):
    if is_tag(item):
        return item
    case, arch, _, _ = parse_item(item); return run_name(case, seed, version, arch=arch)


def case_of(item):
    """run 이름 또는 편성 항목 → case id."""
    return parse_item(item)[0]


def arch_of(item):
    return parse_item(item)[1]


def item_key(item):
    """예약·실측 조회 키: 기본 골격이면 case, 아니면 'case@arch'."""
    case, arch, _, _ = parse_item(item); return case if arch == ARCH_DEFAULT else f"{case}@{arch}"


def seed_of(item, default=None):
    """run 이름 → seed; case 항목이면 default."""
    return parse_item(item)[2] if is_tag(item) else default


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
    measured = measured or {}; key = item_key(case); case = case_of(case)
    if measured.get(key):
        return float(measured[key]), "measured_same_case"
    if confirm:
        if server in CONFIRM_REFERENCE_CASE_H:
            t = CONFIRM_REFERENCE_CASE_H[server]; return float(t.get(case, t["*"])), "confirm_reference_case"
        return (CONFIRM_REFERENCE_H[server], "confirm_reference") if server in CONFIRM_REFERENCE_H else (CONFIRM_PLACEHOLDER_H, "confirm_placeholder")
    if key in REFERENCE_CASE_TRAIN_H.get(server, {}):
        return float(REFERENCE_CASE_TRAIN_H[server][key]), "plan_reference_case"
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
        try:
            sd = seed_of(rid)
        except ValueError:
            continue
        if not h or sd not in allowed_seeds(server):
            continue
        acc.setdefault(item_key(rid), []).append(float(h))       # 골격이 다르면 다른 키 (W104 실측을 W112 로 덮어쓰지 않는다)
    return {c: sum(v) / len(v) for c, v in acc.items()}


def measured_hours_all(server):
    """같은 서버 실측(case@arch 키)을 PAKD50 · QEDGE9 · QEGX ledger 에서 모아 평균 (감사 F07: 새 branch 의 실측이 예약에 반영되지 않던 문제)."""
    acc = {}
    for lp in (LEDGER, QEDGE9_LEDGER, QEGX_LEDGER):
        d = _load_json(lp)
        for rid, e in (d.get("entries") or {}).items():
            if e.get("kind") != "run" or not str(e.get("status", "")).startswith("FINISHED") or not is_tag(rid) or "#" in rid:
                continue
            h = e.get("hours_total") or e.get("hours")
            try:
                sd = seed_of(rid)
            except ValueError:
                continue
            if not h or sd not in allowed_seeds(server):
                continue
            acc.setdefault(item_key(rid), []).append(float(h))
    return {c: sum(v) / len(v) for c, v in acc.items()}


def verified_complete(run, expect_step=50000, min_candidates=45):
    """완료 marker(results .mat 두 개) 를 넘어 계획 §8.2·§8.3 의 완결·동치 검사 (감사 F06): exact-50K last state · 평가 후보 격자 · Teacher 파일 sha == T0 · train h5 sha == cue 자산 · U init hash == seed 공유 init 파일.
    반환 dict(ok, checks{name: bool|None}, notes) — None 은 판단 자료 없음(예: cue 자산/Teacher 없는 run)."""
    rd = os.path.join(ROOT, "work_dir", run); c = {}; notes = []
    c["results_mats"] = os.path.exists(os.path.join(rd, "results", "reduced_best_hqnr.mat")) and os.path.exists(os.path.join(rd, "results", "full_best_hqnr.mat"))
    lm = _load_json(os.path.join("work_dir", run, "last_meta.json")); c["last_exact_step"] = (lm.get("step") == expect_step) and os.path.exists(os.path.join(rd, "last", "model.safetensors"))
    cm = os.path.join(rd, "checkpoint_metrics.csv"); rows = []
    if os.path.exists(cm):
        import csv as _csv
        rows = list(_csv.DictReader(open(cm)))
    steps = {int(float(r["step"])) for r in rows if r.get("step")}
    c["candidate_grid"] = (expect_step in steps) and (len(rows) >= min_candidates)
    kc = _load_json(os.path.join("work_dir", run, "kdv_config_resolved.json")); c["kdv_manifest"] = bool(kc)
    ith = _load_json(os.path.join("work_dir", run, "init_and_teacher_hashes.json")); t = (ith.get("teacher") or {})
    if t:
        sha, _ = t0_identity(None); c["teacher_is_T0"] = (t.get("file_sha256") == sha)
    else:
        c["teacher_is_T0"] = None; notes.append("Teacher 없음(no-KD/no-align run)")
    ds = _load_json(os.path.join("work_dir", run, "dataset_hashes.json")); cue = _load_json(QEDGE9_CUE_ASSET)
    if ds and cue:
        c["train_sha_matches_cue"] = ((ds.get("train_feeder_args") or {}).get("sha256") == (cue.get("dataset") or {}).get("train_sha256"))
    else:
        c["train_sha_matches_cue"] = None; notes.append("dataset_hashes 또는 cue 자산 없음")
    ih = _load_json(os.path.join("work_dir", run, "initialization_hashes.json")); init_f = ih.get("unet_init_file")
    if ih.get("unet_init_sha256_16") and init_f and os.path.exists(init_f):
        try:
            import torch
            from train_pa import _sha_tensors
            c["init_matches_shared_file"] = (_sha_tensors(torch.load(init_f, map_location="cpu")) == ih["unet_init_sha256_16"])
        except Exception as ex:                                                        # noqa
            c["init_matches_shared_file"] = None; notes.append(f"init 파일 검사 실패 {ex!r}")
    else:
        c["init_matches_shared_file"] = None; notes.append("초기값 hash/파일 없음")
    ok = all(v for v in c.values() if v is not None) and c["results_mats"] and c["last_exact_step"] and c["candidate_grid"] and c["kdv_manifest"]
    return dict(run=run, ok=bool(ok), checks=c, notes=notes)


def reservation_for(server, item, measured=None, seed=None):
    """편성 항목(case id 또는 run 이름) → dict(run, case, seed, reference_train_h, reference_kind, reservation_h, gate_hours)."""
    seed = seed or SERVER_SEED[server]; case, arch, sd, _ = parse_item(item); sd = sd if sd is not None else seed; confirm = (sd not in allowed_seeds(server))
    ref, kind = reference_hours(server, (case if arch == ARCH_DEFAULT else f"{case}@{arch}"), measured, confirm=confirm)
    if ref is None:
        return None
    res = reservation_hours(ref)
    return dict(run=to_tag(item, seed), case=case, arch=arch, seed=sd, reference_train_h=ref, reference_kind=kind, reservation_h=res, gate_hours=res / MARGIN)


def write_reservation_file(server, items, measured=None, seed=None, path=None):
    """서버 로컬 RESERVATION_FILE: run → gate_hours 등 (trainer budget.projection_file; margin 1.1 을 곱하면 reservation_h 가 된다). items 는 편성 목록(case id / run 이름)."""
    p = os.path.join(ROOT, path or RESERVATION_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
    rows = [r for r in (reservation_for(server, it, measured, seed) for it in items) if r]
    out = dict(note=f"{server}: reservation_h = {RESERVE_SLACK} × reference_train_h + {RESERVE_POST_H * 60:.0f}/60 ({ALLOC_PLAN} §3); gate_hours = reservation_h / {MARGIN} (trainer 가 budget.margin 을 곱한다). "
                    "reference 는 계획 기준값/가예약이지 실측이 아니다 (kind 참조); 같은 서버 같은 case 실측이 있으면 그것.", server=server, margin=MARGIN, formula="1.10*ref+10/60",
               runs={r["run"]: {k: v for k, v in r.items() if k != "run"} for r in rows})
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    return out


def confirmation_cases(win_case, server=None):
    """확인 seed 묶음 (≤3 run). s2/s4/s5 (§7): WIN · 같은 policy 의 no-KD control · F0 (control 이 F0 면 2 run). s3 (§5 표): 후보별 고정 묶음 (표에 없는 후보는 거부). 순서 유지, 중복 제거."""
    if win_case not in CASES:
        raise SystemExit(f"!! 알 수 없는 case {win_case}")
    if server in CONFIRM_BUNDLE_BY_SERVER:
        b = CONFIRM_BUNDLE_BY_SERVER[server].get(win_case)
        if b is None:
            raise SystemExit(f"!! {server}: 확인 묶음 표에 없는 후보 {win_case} — 표에 있는 후보 {list(CONFIRM_BUNDLE_BY_SERVER[server])} 만")
        assert len(b) <= CONFIRM_MAX_RUNS_BY_SERVER.get(server, CONFIRM_MAX_RUNS)
        return list(b)
    ctl = BASELINE_OF[CASES[win_case][0]]; out = []
    for c in (win_case, ctl, "F0"):
        if c not in out:
            out.append(c)
    assert len(out) <= CONFIRM_MAX_RUNS
    return out
PURPOSE.update(_PURPOSE_S5)
_QV = {"QA05": "α 0.5 (L_D 재가중 절반)", "QB005": "β 0.05 (soft 절반)", "QB02": "β 0.2 (soft 두 배)", "QE025": "λE ×0.5", "QE10": "λE ×2"}          # Q12 단일축 scalar variant (s4 배정 §6) — 그 밖 정의는 JQ/ALQ 와 같다
PURPOSE.update({f"J_{v}": f"s4 §6 (QEGX s3 §4.4 대응쌍): JQ 에서 {t} 만" for v, t in _QV.items()}); PURPOSE.update({f"AL_{v}": f"s4 §7: ALQ(A LR 3e-6) 에서 {t} 만" for v, t in _QV.items()})
NEEDS_LAMBDA_E = {c for c, (p, b) in CASES.items() if BACKEND[b]["edge"]}


def run_name(case, seed, version="v1", protocol=PROTOCOL, arch=ARCH_DEFAULT):
    return f"PAKD50_{case}_{arch}_WV3_T0_S{seed}_{protocol}_{version}"


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


def schedule(has_lambda_e, is_terminal, remaining_hours, est_hours, margin=MARGIN, priority=None, extra=(), exempt=None, blocked=None):
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
        if blocked is not None and blocked(c):                            # QEDGE9: cue 자산 대기 — 이번 pass 에는 편성하지 않는다 (다음 pass 재검사; 뒤 항목은 계속)
            continue
        if exempt is not None and exempt(c):                              # QEDGE9 §0.7·§9.3: 절대 마감 admission 대상이 아니다 (soft target 만)
            todo.append(c); continue
        need = float(est_hours(c)) if callable(est_hours) else margin * float(est_hours)
        if remaining_hours is None or t + need <= remaining_hours:
            todo.append(c); t += need
        else:
            dropped.append(c)
    return todo, dropped


def kdv_block(case, seed, server, cal=None, projected=None, version="v1", pin=True, arch=ARCH_DEFAULT, branch=None):
    """계획 §4·§6 의 FRESH50 kdv 블록. pin: calibration_resolved.json 의 τR/λE 를 숫자로 고정(서버 간 동일 package); 없으면 calibrate(그 서버에서 T0/pilot 로 산출).
    arch: Student U 골격 (기본 W112_D123; s4 이식 W104_D121 — T0·donor·τR·λE0 는 그대로, expect_arch 와 init hash namespace 만 바뀐다)."""
    pol_id, be_id = CASES[case]; P, B = POLICY[pol_id], BACKEND[be_id]; cal = cal if cal is not None else calibration(); A = ARCHS[arch]
    branch = branch or ("QEGX" if case in QEGX_CASES else ("QEDGE9" if B.get("edge_gate") else None))     # gate/route case 는 branch 가 있어야 한다: QEGX 전용 case → QEGX, 그 밖의 gate case → QEDGE9 (W104 전용; 아래 검사)
    if branch != "QEGX" and (case in QEGX_CASES or B.get("edge_route")):
        raise SystemExit(f"!! {case}: QEGX 전용 case 는 QEGX branch 로만 만든다 (현재 {branch})")
    sha, step = t0_identity(server); t0 = t0_dir(server); me = run_name(case, seed, version, arch=arch); noalign = (P["pol"] == "A-ID")
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
             **({"experiment_branch_id": A["branch"], "architecture_signature": f"{arch}:model.pancrafter_paper.PANCrafterPaper:in_mode=paper:norm=ln:attn_locations=[]:n_attn=3:mode_modulation=false"} if A["branch"] else {}),
             input_protocol=P["proto"], aligner_policy=P["pol"], diag_every=1000, diag=dict(fixed_batch=True), calibration=dict(n_patches=3072, seed=1234), **({} if noalign else dict(aligner_lr=P["alr"])),
             select=dict(primary="best_hqnr", secondary=["best_rr_val", "last"], retain_all_candidates=True, **({"aligned_selector": False} if noalign else {})), expect_arch=dict(width=A["width"], depth=list(A["depth"]), noalign=noalign),
             rec=rec, stat=stat, geom_kd=dict(mode="G0"), recipe=("NOALIGN" if noalign else "N2_SG"), **({} if noalign else dict(eval=dict(fixed_reference_from_donor=True))),
             **({"expect_init": dict(unet_sha256_16=init_hash_for(seed, arch))} if init_hash_for(seed, arch) else {}),
             **({"na_protocol": "NA-STRICT"} if noalign else dict(donor=dict(source=f"{t0}/{T0_TAG}", view_margin_hr=4, expected_sha256=sha, expected_step=step))),
             # bridge: trainer/smoke 의 "Teacher 폭 == Student 폭" 검사를 푸는 명시 flag (kdv §3.2-3) — 골격 이식(cross-capacity, T0 고정) 에서만 True; 다른 동작 변화 없음
             teacher=dict(id="T0", run=t0, tag=T0_TAG, expected_sha256=sha, bridge=(arch != ARCH_DEFAULT), **({} if needs_teacher else dict(eval_only=True))),
             baseline_run=run_name(BASELINE_OF[pol_id], seed, version, arch=arch),
             budget=dict(ledger=LEDGER, total_gpu_hours=TOTAL_HOURS, reserve_hours=RESERVE_HOURS, margin=MARGIN, required=False, projected_hours=projected, projected_map={me: RUN_RESERVED_HOURS},
                         remaining_mandatory=[], remaining_mandatory_file=MANDATORY_FILE,     # 서버 기본 묶음 예약은 **서버 로컬 파일**(prepare 가 씀) — 같은 seed 서버(s1/s4, s3/s5) 가 config 파일을 공유하므로 config 에 박지 않는다 (s5 보고 #2)
                         projection_file=RESERVATION_FILE,                                     # 서버 로컬 run 별 예약 (재배정 §3 식; gate 가 씀) — 있으면 projected_map(4.0h 보수값) 보다 우선
                         **({"training_deadline": training_deadline()} if training_deadline() else {})))            # 공통 절대 마감 — trainer 가 예상 종료 ≤ 마감 을 검사
    if branch == "QEDGE9":                                                  # QEDGE9 §0.7·§9.3·§11.1: 새 논리 캠페인/branch, PAKD50 50h·절대 마감 미상속(soft 9h, required → 경고만; NaN/오류/중복 보호는 trainer 그대로)
        if arch != QEDGE9_ARCH:
            raise SystemExit(f"!! {case}: QEDGE9 는 {QEDGE9_ARCH} 전용 (현재 {arch})")
        k.update(campaign_id=QEDGE9_CAMPAIGN_ID, parent_campaign_id=CAMPAIGN_ID, experiment_branch_id=QEDGE9_BRANCH, exact_resume=True,     # exact_resume: 감사 F04 (kdv/resume.py; 재개 시 같은 batch 열)
                 budget=dict(ledger=QEDGE9_LEDGER, total_gpu_hours=QEDGE9_SOFT_HOURS, reserve_hours=0.0, margin=MARGIN, required=True, projected_hours=projected, projected_map={me: QEDGE9_RESERVED_H},
                             remaining_mandatory=[], remaining_mandatory_file=QEDGE9_MANDATORY_FILE, projection_file=RESERVATION_FILE, time_policy=dict(QEDGE9_TIME_POLICY)))
    if branch == "QEGX":                                                    # QEGX §9.3·§12: 새 논리 캠페인(parent PAKD50, lineage QEDGE9), 시간 상한 없음 — total 1000h + required(경고만), 절대 마감 없음, exact_resume
        if arch != QEGX_ARCH:
            raise SystemExit(f"!! {case}: QEGX 는 {QEGX_ARCH} 전용 (현재 {arch})")
        k.update(campaign_id=QEGX_CAMPAIGN_ID, parent_campaign_id=CAMPAIGN_ID, lineage_campaign_ids=[CAMPAIGN_ID, QEDGE9_CAMPAIGN_ID], experiment_branch_id=QEGX_BRANCH, exact_resume=True,
                 budget=dict(ledger=QEGX_LEDGER, total_gpu_hours=QEGX_SOFT_HOURS, reserve_hours=0.0, margin=MARGIN, required=True, projected_hours=projected, projected_map={me: QEGX_RESERVED_H},
                             remaining_mandatory=[], remaining_mandatory_file=QEGX_MANDATORY_FILE, projection_file=RESERVATION_FILE, time_policy=dict(QEGX_TIME_POLICY)))
    if branch in ("QEDGE9", "QEGX") and B.get("edge_gate"):
        if B["edge_gate"] == "const3":                                      # QEC3 (§4.2): s3 pilot(J0@W104 S2026 v2 exact50K) 의 c_E3, 별도 파일 — 다른 서버에는 pilot 이 없어 만들지 않는다
            if server not in QEGX_PILOT_BY_SERVER:
                raise SystemExit(f"!! {case}: QEC3 의 pilot(c_E3) 은 {list(QEGX_PILOT_BY_SERVER)} 만 (현재 {server}) — QEC3 는 s3 에서만")
            k["edge_gate"] = dict(mode="const", asset=QEDGE9_CUE_ASSET, c_E_file=QEGX_CE_FILE, pilot_run=QEGX_PILOT_BY_SERVER[server], pilot_tag=QEDGE9_PILOT_TAG, pilot_step=QEDGE9_PILOT_STEP)
        else:
            k["edge_gate"] = {"low_q": dict(mode="low_q", asset=QEDGE9_CUE_ASSET, theta_source="asset"),
                              "const": dict(mode="const", asset=QEDGE9_CUE_ASSET, c_E_file=QEDGE9_CE_FILE, pilot_run=QEDGE9_PILOT_BY_SERVER.get(server, QEDGE9_PILOT_RUN), pilot_tag=QEDGE9_PILOT_TAG, pilot_step=QEDGE9_PILOT_STEP),   # 감사 F02: pilot identity(그 서버의 J0 exact50K) 를 config 에 박고 trainer 가 c_E 파일과 대조
                              "shuffle": dict(mode="shuffle", asset=QEDGE9_CUE_ASSET, perm_seed=51515)}[B["edge_gate"]]
    if branch == "QEGX" and B.get("edge_route"):                            # QER50/QERS (§4.5): U 는 all-edge, A 는 gated edge — kdv.edge_route (registry: EDGE-H·trainable A 위에서만, edge_gate/routing 과 결합 금지)
        k["edge_route"] = {"low_q": dict(mode="low_q", asset=QEDGE9_CUE_ASSET, theta_source="asset"), "shuffle": dict(mode="shuffle", asset=QEDGE9_CUE_ASSET, perm_seed=51515)}[B["edge_route"]]
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


def render(tag, case, seed, server, k, updates, eval_epoch, tpl, arch=ARCH_DEFAULT, branch=None):
    import yaml
    sp = resolve(k); pol_id, be_id = CASES[case]; sha, step = t0_identity(server); A = ARCHS[arch]
    t = re.sub(r"^(#.*\n)+", "", tpl)
    if (k.get("edge_gate") and k["edge_gate"]["mode"] in ("low_q", "shuffle")) or k.get("edge_route"):   # QEDGE9 §7.1 / QEGX §4.5: 학습 loader 가 (index, rot) 를 그대로 준다 — RNG 추가 소비 없음 (feeders.feeder return_meta)
        t2 = re.sub(r"^(train_feeder_args:\n(?:  .*\n)*?  rot: True\n)", r"\1  return_meta: true\n", t, count=1, flags=re.M)
        assert t2 != t, "template 의 train_feeder_args 에 rot: True 가 없다 — return_meta 삽입 실패"; t = t2
    if arch != ARCH_DEFAULT:                                                 # Student U 골격만 바꾼다 (model class·in_mode·norm·attn 은 template 그대로; s4 이식 §3·§9.1)
        t = re.sub(r"^  hidden_size: 112$", f"  hidden_size: {A['width']}", t, flags=re.M); t = re.sub(r"^  depth: \[1, 2, 3\]", f"  depth: [{', '.join(map(str, A['depth']))}]", t, flags=re.M)
        t = re.sub(r"^expect_params_m: [0-9.]+", f"expect_params_m: {A['params_m']}", t, flags=re.M)        # smoke 의 params 검사 (backbone 만)
    head = (f"# {tag} — {PURPOSE[case]}. 생성: tools/gen_pakd50_configs.py (seed {seed} 를 쓰는 서버 {'/'.join(servers_sharing_seed(server)) if seed == SERVER_SEED[server] else server} 공용 — 서버별 값은 config 에 없다). 손으로 고치지 말 것.\n"
            f"# 캠페인 {CAMPAIGN_ID} · protocol {PROTOCOL} · grid {GRID_ID} · 계획 {PLAN} · 요약 {SUMMARY} · 노트 {NOTE}\n"
            f"# 세팅: {describe(sp)} · 정책 {pol_id}(A {'없음' if sp['policy'] == 'A-ID' else ('trainable' if sp['aligner_trainable'] else 'frozen')}, offset λ {k.get('aux', {}).get('offset_weight', 0)}, A LR {k.get('aligner_lr', '—')}) · backend {be_id} (rec {k['rec']['case']}{', EDGE-H λE ' + str(k['stat'].get('outer_weight')) if k['stat'].get('enabled') else ''})\n"
            f"# 약명→세팅: J0/JQ/JR/XJ = A trainable(joint) + N0/Q12/R1/X02 · F0/FQ/FR/XF = A frozen + … · AL0/ALQ = joint, A LR 3e-6 · Q12 = (1+αd)L1 + β(1−d)a|S−T| + λE·signed Scharr edge · R1 = (1+αd)L1 · X02 = R1 + edge\n"
            f"#           D*/LF* = joint 에서 A 를 0–4999 동결 / 25000 부터 동결 (kdv.aligner_schedule; 동결 구간 LO 없음, ε RNG 는 같은 순서) · P*/JK0/JE0 = A 가 직접 받는 항만 제한 (kdv.routing qA=(qD,qK,qE); U 는 backend 전체) · J_Q*/AL_Q* = Q12 계수만\n"
            f"#           J_R3_NOEDGE = J + R3 adaptive(edge 없음) · J_N0_EDGE = J + N0 + λE edge(Teacher 미사용) · RC0/RCQ = A trainable(LR 1e-5) 인데 offset 연습 없음(I-NATIVE-TRANSFER, radius 0) + N0/Q12 — 재배정 {ALLOC_PLAN}\n"
            f"# Teacher T0 = {T0_RUN}/{T0_TAG} (step {step}, file sha {(sha or '?')[:16]}…; A+U frozen, 자기 aligner 로 forward) · Student A = T0 aligner 복사(view margin 4), U = init_unet_seed{seed}.pt · τR/λE = {CAL_PATH} (고정) \n"
            f"# 골격 W{A['width']}·D{''.join(map(str, A['depth']))} · 9ch · 50K · batch 48 · AdamW 1e-4/{k.get('aligner_lr', '—')} wd 0.01 cosine warmup100 · eval_epoch {eval_epoch} (= {GRID_ID}: 1010 update 마다 + exact 50000, 50 후보 보존) · 예산 {LEDGER} {TOTAL_HOURS}h(+감사 {RESERVE_HOURS}h)\n")
    t = re.sub(r"^eval_epoch: \d+$", f"eval_epoch: {eval_epoch}", t, flags=re.M)
    t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
    t = re.sub(r"^trainer: po\npo:\n(  .*\n)+", "", t, flags=re.M)
    t = t.replace("mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)",
                  "mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)\ntrainer: kdv\nkdv:\n" + "\n".join("  " + l for l in yaml.safe_dump(k, sort_keys=False, allow_unicode=True, default_flow_style=False).splitlines()) + "\n")
    t = re.sub(r"^num_iter: \d+", f"num_iter: {updates}", t, flags=re.M); t = re.sub(r"^seed: \d+", f"seed: {seed}", t, flags=re.M)
    assert "trainer: kdv" in t and "trainer: po" not in t and f"seed: {seed}" in t and f"hidden_size: {A['width']}" in t and f"depth: [{', '.join(map(str, A['depth']))}]" in t and f"eval_epoch: {eval_epoch}" in t and f"expect_params_m: {A['params_m']}" in t
    if arch != ARCH_DEFAULT:
        head = head.replace("\n", f"\n# 골격 {arch}: {A['note']} · branch {A['branch']} · 편성 항목 <case>@{arch} · 시트 X열 'PAKD50 / <case> / {ARCH_LABEL[arch]} / FRESH50' · 계획 {ALLOC_PLAN_S4}\n", 1)
    if branch == "QEDGE9":
        head = head.replace("\n", f"\n# QEDGE9 (계획 {QEDGE9_PLAN}, 노트 {QEDGE9_NOTE}): 캠페인 {QEDGE9_CAMPAIGN_ID} · branch {QEDGE9_BRANCH} (parent {CAMPAIGN_ID}) · 시간 정책 soft target {QEDGE9_SOFT_HOURS}h(절대 마감·50h 미상속, kdv.budget.time_policy) ·"
                            + (f" GT edge gate {k['edge_gate']['mode']} (cue {QEDGE9_CUE_ASSET}; θq/c_E 는 자산에서만, 없으면 시작 안 함)" if k.get("edge_gate") else " gate 없음(J0/JQ 대조군; 학습 정의는 PAKD50 과 같고 예산 metadata 만 다르다)")
                            + f" · 시트 X열 'PAKD50 / {case} / {ARCH_LABEL[arch]} / QEDGE9 / FRESH50'\n", 1)
    if branch == "QEGX":
        what = (f" GT edge gate {k['edge_gate']['mode']} (cue {QEDGE9_CUE_ASSET}{'; c_E3 ' + QEGX_CE_FILE + ' = s3 pilot ' + k['edge_gate']['pilot_run'] + ' exact50K' if k['edge_gate']['mode'] == 'const' else ''})" if k.get("edge_gate")
                else (f" edge_route {k['edge_route']['mode']}: U 는 all-edge E(1), A 는 gated edge 만 (A .grad 에서 λE·mean((1−g)E_i) 를 뺀다; cue {QEDGE9_CUE_ASSET})" if k.get("edge_route")
                      else " gate/route 없음(대조군; 학습 정의는 PAKD50 과 같고 캠페인·예산 metadata 만 다르다)"))
        head = head.replace("\n", f"\n# QEGX (계획 {QEGX_PLAN}, 노트 {QEGX_NOTE}): 캠페인 {QEGX_CAMPAIGN_ID} · branch {QEGX_BRANCH} (parent {CAMPAIGN_ID}, lineage {QEDGE9_CAMPAIGN_ID}) · 시간 정책 no_hard_limit(상한 없음; 자체 ledger {QEGX_LEDGER}, required 경고만, 절대 마감 미상속) ·"
                            + what + (f" · A 일정 freeze_from {k['aligner_schedule']['freeze_from']}" if (k.get("aligner_schedule") or {}).get("freeze_from") else "") + (f" · β {k['rec'].get('kd_weight')}" if k["rec"].get("kd_weight") not in (None, 0.1) else "")
                            + f" · 시트 X열 'PAKD50 / {case} / {ARCH_LABEL[arch]} / QEGX / FRESH50'\n"
                            + f"# 약명→세팅(QEGX): QE50/QES = Q12 인데 GT edge 를 q_T<θq patch 만 / stratum 셔플 (kdv.edge_gate) · QX50 = QE50 − soft(R1) · QE50_B005 = QE50 β0.05 · QEC3 = 상수 c_E3(s3 pilot) · LFQE50 = LF 일정 + QE50 · QER50/QERS = U all-edge, A 는 gated edge 만 (kdv.edge_route) · J_QB005 = JQ β0.05\n", 1)
    return head + t


def generate(server, cases, out_dir, updates=50000, eval_epoch=5, projected=None, version="v1", pin=True, seed=None):
    seed = seed or SERVER_SEED[server]; tpl = open(os.path.join(ROOT, "config", "PO10_N1_REC_W112_D123_WV3_S2025_R200_FRSTAT.yaml")).read(); made = []
    for item in cases:
        case, arch, isd, iver = parse_item(item); sd = isd if isd is not None else seed; ver = iver or version; br = branch_for(server, item, sd)      # 항목이 run 이름이면 그 seed·version (QEDGE9 s5 의 777 · s1 의 v2)
        tag = run_name(case, sd, ver, arch=arch); k = kdv_block(case, sd, server, projected=projected, version=ver, pin=pin, arch=arch, branch=br)
        os.makedirs(out_dir, exist_ok=True); open(os.path.join(out_dir, tag + ".yaml"), "w").write(render(tag, case, sd, server, k, updates, eval_epoch, tpl, arch=arch, branch=br)); made.append(tag)
    return made


def plan_rows(server, measured=None, is_terminal=None, extra=()):
    """재배정 dry-run: 서버 순서(priority_for + extra) 에서 완료·실행 중이 아닌 run 의 예약과 누적 (재배정 §4 표 형식). is_terminal(run 이름) 이 없으면 work_dir 완료 판정."""
    seed = SERVER_SEED[server]; measured = measured if measured is not None else measured_hours_all(server)
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
    plan_doc = {"s3": QEGX_PLAN + " §5 (s3 15 run v2)", "s4": QEGX_PLAN + " §6 (E0 5 완료 + 14 run v1)", "s5": QEDGE9_PLAN, "s1": QEDGE9_PLAN + " (seed 1234 묶음, 17:20 s4→s1)"}.get(server, ALLOC_PLAN)
    print(f"[{server}] seed {SERVER_SEED[server]} — 배정 {plan_doc} §4; 예약 = {RESERVE_SLACK}×ref + {RESERVE_POST_H * 60:.0f}min; 학습 마감까지 {'?' if rem is None else '%.2f' % rem} h")
    for r in rows:
        print(f"  {(r['case'] + ('' if r['arch'] == ARCH_DEFAULT else '@' + r['arch'])):<18} S{r['seed']:<5} {(branch_for(server, r['run']) or 'PAKD50'):<7} {r['status']:<14} ref {r['reference_train_h']:.2f} h ({r['reference_kind']}) → 예약 {r['reservation_h']:.4f} h · 누적 {r['cumulative_h']:.4f} h")
    tot = sum(r["reservation_h"] for r in rows if r["status"] == "planned"); tot_x = sum(r["reservation_h"] for r in rows if r["status"] == "planned" and branch_for(server, r["run"]) in ("QEDGE9", "QEGX"))
    print(f"  planned {sum(r['status'] == 'planned' for r in rows)} run · 예약 합 {tot:.4f} h" + (f" (그중 마감 admission 제외 branch {tot_x:.4f} h)" if tot_x else "")
          + ("" if rem is None else f" · PAKD50 branch 마감 안 {'OK' if tot - tot_x <= max(rem, 0.0) else '초과 — gate admission 이 뒤를 민다'}"))


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
        if not a.no_pin and (not cal.get("lambda_E")) and any(case_of(c) in NEEDS_LAMBDA_E for c in cases):
            sys.exit(f"!! {[c for c in cases if case_of(c) in NEEDS_LAMBDA_E]} 는 λE 가 고정된 뒤에 만든다 — tools/pakd50_calibrate.py --lambda-e (J0 S1234 exact50K 필요). 현재 {CAL_PATH}: {list(cal)}")
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
