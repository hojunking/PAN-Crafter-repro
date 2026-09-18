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
캠페인 QEGX_A104D121_S3S4_20260915_v1 / branch A104D121_T0FIX_QEGX_v1, 시간 상한 없음(자체 ledger, required 경고만). 전환 tools/qegx_switch.sh, QEC3 pilot tools/qedge9_cue.py pilot --branch qegx.
EDGEBAL (2026-09-16, research_log/PAN_EDGEBAL_S2_S5_Experiment_Plan_2026-09-16.md): GT edge 를 얼마나·언제 — s2(777·2026·9091, 12 run v1) 상수 배수 r·성분별 기여 · s5(2026·777·9091, 14 run v2) 시간배분 DOWN/UP/CONST075 + 완만한 q 가중 FLOOR/SHUF/REVERSE.
캠페인 EDGEBAL_A104D121_S2S5_20260916_v1 / branch A104D121_T0FIX_EDGEBAL_v1, 상한 없음. 새 키 kdv.edge_schedule / kdv.edge_weight. 전환 tools/edgebal_switch.sh.
QRECON24 (2026-09-16 저녁, research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md): 확정 method(연속 q 가중 w=qref/(qref+q_T) — 계획서의 분자 2 는 09-16 저녁 결정으로 뺐다; U ← H+K+λE·w·E, A ← w·H 만; offset 없음) 의 전 서버 튜닝 68 run —
s1 12(1234·3407: G22/A_UNIF/A_FREEZE/G21/G23/A_SHUF) · s2 12(777: G 3×3 + E_UNIF/E_SHUF/ALL_UNIF) · s3 18(2026·4321: G 3×3) · s4 14(1234: H 3×3 + H_ALPHA0/H_BETA0; 3407: H22/H12/H11) · s5 12(2026·777·9091·1103: L100/L070/L050).
캠페인 QRECON24_A104D121_S1S5_20260916_v1 / branch A104D121_T0FIX_QRECON24_v1, 상한 없음(24h 최소 운영구간). 새 키 kdv.qrecon. 이전 QEDGE9/QEGX/EDGEBAL 큐는 superseded(PREVIOUS_PRIORITY_BY_SERVER). 전환 tools/qrecon24_switch.sh."""
import argparse, glob, json, os, re, subprocess, sys
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
# EDGEBAL (2026-09-16; research_log/PAN_EDGEBAL_S2_S5_Experiment_Plan_2026-09-16.md): q gate 를 더 복잡하게 만들기 전에 **GT edge 를 얼마나·언제** 줄지 — s2(777 탐색 + 2026·9091 반복) 상수 배수 r ∈ {0, .25, .5, 2}·성분별 기여(N0+edge, R1+edge),
# s5(2026 탐색 + 777·9091 반복) 시간배분(DOWN 1→.5 / UP .5→1 / CONST075) 과 완만한 q 가중(FLOOR .75/.25 · SHUF · REVERSE). 전부 W104·D121·T0 고정·J 정책, 시간 상한 없음. 이름 충돌 방지: s2 v1 · s5 v2 (실행 identity, 수식 차이 아님).
EDGEBAL_PLAN = "research_log/PAN_EDGEBAL_S2_S5_Experiment_Plan_2026-09-16.md"; EDGEBAL_NOTE = "research_log/2026-09-16_edgebal-implementation.md"
EDGEBAL_CAMPAIGN_ID = "EDGEBAL_A104D121_S2S5_20260916_v1"; EDGEBAL_BRANCH = "A104D121_T0FIX_EDGEBAL_v1"; EDGEBAL_ARCH = QEDGE9_ARCH
EDGEBAL_LEDGER = "work_dir/_edgebal_budget/ledger.json"; EDGEBAL_MANDATORY_FILE = "work_dir/_edgebal/mandatory_runs.txt"; EDGEBAL_SOFT_HOURS = 1000.0; EDGEBAL_RESERVED_H = 2.3
EDGEBAL_TIME_POLICY = dict(mode="no_hard_limit", target_elapsed_hours=None, hard_deadline=None, inherit_parent_deadline=False)
EDGEBAL_VERSION_BY_SERVER = {"s2": "v1", "s5": "v2"}; EDGEBAL_SWITCH_UPDATE = 25000
EDGEBAL_CASES = ("EB_N0", "EB_R3E000", "EB_R3E025", "EB_R3E050", "EB_R3E075", "EB_R3E100", "EB_R3E200", "EB_N0E100", "EB_R1E100", "EB_EDOWN", "EB_EUP", "EB_QFLOOR", "EB_QFSHUF", "EB_QFREV")
EDGEBAL_CONTROLS = {"s2": {777: {"J0": "PAKD50_J0_W104_D121_WV3_T0_S777_FRESH50_v2", "JQ": "PAKD50_JQ_W104_D121_WV3_T0_S777_FRESH50_v2", "QE50": "PAKD50_QE50_W104_D121_WV3_T0_S777_FRESH50_v2"}},    # §5: s2 가 돌린 QEDGE9 v2 3 벌(재사용 검증 뒤)
                    "s5": {2026: {"J0": "PAKD50_J0_W104_D121_WV3_T0_S2026_FRESH50_v1", "JQ": "PAKD50_JQ_W104_D121_WV3_T0_S2026_FRESH50_v1", "QE50": "PAKD50_QE50_W104_D121_WV3_T0_S2026_FRESH50_v1"},
                           777: {"J0": "PAKD50_J0_W104_D121_WV3_T0_S777_FRESH50_v1", "JQ": "PAKD50_JQ_W104_D121_WV3_T0_S777_FRESH50_v1", "QE50": "PAKD50_QE50_W104_D121_WV3_T0_S777_FRESH50_v1"}}}    # §6: s5 QEDGE9 두 seed
CUE_CASES = ("QE50", "QEC", "QES", "QX50", "QEC3", "QE50_B005", "LFQE50", "QER50", "QERS", "EB_QFLOOR", "EB_QFSHUF", "EB_QFREV")   # cue 자산(θq/gate 표) 이 있어야 시작하는 case (gate 의 cue_ready); QRC24_* 도 (raw q)
# QRECON24 (2026-09-16; research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md): 사용자 확정 method 의 전 서버 튜닝 — 연속 q 가중 w = qref/(qref+q_T)(분자 2 없음; 09-16 저녁 결정) 를 A(weighted GT hard 만) 와 U(GT edge) 가 공유,
# U ← H+K+λE·w·E / A ← w·H (같은 forward 에서 parameter 집합별 gradient 분리; Student offset·jitter 없음). 68 run FRESH50(s1 12 · s2 12 · s3 18 · s4 14 · s5 12), 상한 없음(24h 는 최소 운영구간).
# 조정값: λE(절대) 6e-4/2e-3/6e-3(09-16 저녁: 분자 2 제거에 맞춘 2 배 환산; 계획서 3e-4/1e-3/3e-3) · rA .003/.01/.03 · α .5/1/1.5 · β .05/.1/.2 · U LR 1e-4/7e-5/5e-5 · seed. qref·q 함수 고정. uniform 대조 = s_q(qref) = 0.5.
# 이름 PAKD50_QRC24_<SRV>_<PROFILE>_W104_D121_WV3_T0_S<seed>_FRESH50_v1 (서버 토큰 = 파일 충돌 방지). ADJ-R1(09-17) 추가 16 run 은 _v2 (조정 편성 이력 표시; q 정의는 같다).
QRC24_PLAN = "research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md"; QRC24_NOTE = "research_log/2026-09-16_qrecon24-implementation.md"
QRC24_CAMPAIGN_ID = "QRECON24_A104D121_S1S5_20260916_v1"; QRC24_BRANCH = "A104D121_T0FIX_QRECON24_v1"; QRC24_ARCH = QEDGE9_ARCH; QRC24_QREF = 0.3276133416220546
QRC24_LEDGER = "work_dir/_qrecon24_budget/ledger.json"; QRC24_MANDATORY_FILE = "work_dir/_qrecon24/mandatory_runs.txt"; QRC24_SOFT_HOURS = 1000.0
QRC24_TIME_POLICY = dict(mode="no_hard_limit", target_elapsed_hours=None, hard_deadline=None, inherit_parent_deadline=False, min_operating_hours=24.0)
QRC24_LAMBDA = {1: 6e-4, 2: 2e-3, 3: 6e-3}; QRC24_LAMBDA_PLAN = {1: 3e-4, 2: 1e-3, 3: 3e-3}     # 09-16 저녁: 분자 2 제거 → λE 2 배 환산(λE_new·s_q = λE_old·w_q); 계획서 값은 PLAN 에 기록만
QRC24_UNIFORM_W = 0.5                                                                             # uniform 대조 = s_q(qref) (종전 계획의 1 아님)
QRC24_DEFAULT = dict(lam=QRC24_LAMBDA[2], rA=0.01, alpha=1.0, beta=0.1, ulr=1e-4, a="q", e="q", frozen=False)
QRC24_PROFILES = {**{f"G{i}{j}": dict(lam=QRC24_LAMBDA[i], rA=r) for i in (1, 2, 3) for j, r in ((1, 0.003), (2, 0.01), (3, 0.03))},
                  "A_UNIF": dict(a="uniform"), "A_FREEZE": dict(frozen=True), "A_SHUF": dict(a="shuffle"), "E_UNIF": dict(e="uniform"), "E_SHUF": dict(e="shuffle"), "ALL_UNIF": dict(a="uniform", e="uniform"),
                  **{f"H{i}{j}": dict(alpha=al, beta=be) for i, al in ((1, 0.5), (2, 1.0), (3, 1.5)) for j, be in ((1, 0.05), (2, 0.1), (3, 0.2))}, "H_ALPHA0": dict(alpha=0.0), "H_BETA0": dict(beta=0.0),
                  "L100": dict(ulr=1e-4), "L070": dict(ulr=7e-5), "L050": dict(ulr=5e-5),
                  # ADJ-R1 (2026-09-17 §4) 신규 세 profile — 새 loss 가 아니라 기존 범위 안의 결합/대조: B20A03 = G23 에서 β 만 .1→.2 (= H23 에서 rA 만 .01→.03) · A03_UNIF/A03_SHUF = G23(rA .03) 에서 A 의 q 연결만 상수 .5 / stratum 셔플
                  "B20A03": dict(rA=0.03, beta=0.2), "A03_UNIF": dict(rA=0.03, a="uniform"), "A03_SHUF": dict(rA=0.03, a="shuffle")}
QRC24_CANONICAL = {"H22": "G22", "L100": "G22"}                                                  # 같은 수학적 설정의 별칭 (§4.4; 같은 server+seed 에 중복 편성 없음)


def qrc24_control_profile(server, seed):
    """그 서버·seed 에 **실제로 편성된** canonical-G22 profile(s4 H22 · s5 L100 · 그 밖 G22) — run id 는 alias 가 아니다 (감사 F09).
    §8.3 확장 seed(QRC24_EXTRA)·reserve seed 는 확장 묶음 안의 alias(s4 H22 · s5 L100). 어디에도 없으면 SystemExit — 가상 G22 id 를 만들지 않는다 (ADJ-R1 §11.2; 검토 지적)."""
    for p_, sd_, _v in QRC24_QUEUES.get(server, []):
        if sd_ == seed and QRC24_CANONICAL.get(p_, p_) == "G22":
            return p_
    sd0, profs = QRC24_EXTRA.get(server, (None, ()))
    if seed == sd0 or seed in QRC24_RESERVE_SEEDS:
        for p_ in profs:
            if QRC24_CANONICAL.get(p_, p_) == "G22":
                return p_
    raise SystemExit(f"!! QRECON24 {server} S{seed}: 편성/확장 묶음에 canonical-G22 대조가 없다 — 가상 G22 id 를 만들지 않는다 (ADJ-R1 §11.2)")


def qrc24_item_version(server, profile, seed):
    """활성 큐(QRC24_QUEUES) 에 편성된 (profile, seed) 의 version ('v1' 원계획 / 'v2' ADJ-R1 추가) — 없으면 None."""
    for p_, sd_, v_ in QRC24_QUEUES.get(server, []):
        if p_ == profile and sd_ == seed:
            return v_
    return None


def qrc24_control_run(server, seed, profile=None):
    """같은 서버·seed 의 대조 run **실제 id**(편성된 version 으로; ADJ-R1 §11.2 '가상 G22 ID 참조 금지'). profile 없으면 canonical-G22 alias(s4 H22 · s5 L100; 확장 seed 는 확장 묶음의 alias — v1 이름)."""
    p_ = profile or qrc24_control_profile(server, seed)
    return qrc24_run_name(server, p_, seed, qrc24_item_version(server, p_, seed) or "v1")
# 원계획 §5 (2026-09-16; 68 run) — 이력·등록 완료(부록 A 39) 판정의 기준. 편성은 아래 ADJ-R1 의 QRC24_QUEUES.
QRC24_QUEUES_20260916 = {"s1": [(p, sd) for sd in (1234, 3407) for p in ("G22", "A_UNIF", "A_FREEZE", "G21", "G23", "A_SHUF")],
                         "s2": [(p, 777) for p in ("G22", "G12", "G32", "G21", "G23", "G11", "G13", "G31", "G33", "E_UNIF", "E_SHUF", "ALL_UNIF")],
                         "s3": [(p, sd) for sd in (2026, 4321) for p in ("G22", "G12", "G32", "G21", "G23", "G11", "G13", "G31", "G33")],
                         "s4": [(p, 1234) for p in ("H22", "H12", "H21", "H11", "H23", "H32", "H13", "H31", "H33", "H_ALPHA0", "H_BETA0")] + [(p, 3407) for p in ("H22", "H12", "H11")],
                         "s5": [(p, sd) for sd in (2026, 777, 9091, 1103) for p in ("L100", "L070", "L050")]}
# ADJ-R1 (2026-09-17; research_log/PAN_QRECON24_S1_S5_Queue_Adjustment_2026-09-17.md §4–§5·§9·§11·부록 B): 미시작 run 순서 재편 + 기존 범위 안의 결합/대조 16 run 추가(v2 id) + 미시작 3 run 보류.
# loss 식·gradient 경로·Teacher·q 정의·λE 환산(6e-4/2e-3/6e-3)·uniform .5·50K 는 그대로. 기존 26 run 은 v1 id 유지(config 바이트 불변 — K50 이 지킨다). 24h 규칙은 원 campaign 의 누적(0 부터 다시 세지 않음).
QRC24_ADJ_PLAN = "research_log/PAN_QRECON24_S1_S5_Queue_Adjustment_2026-09-17.md"; QRC24_ADJ_NOTE = "research_log/2026-09-17_qrecon24-adjustment-r1-implementation.md"
QRC24_ADJ_REVISION = "QRC24_ADJ_R1_20260917"; QRC24_ADJ_VERSION = "v2"; QRC24_ADJ_PROFILES = ("B20A03", "A03_UNIF", "A03_SHUF")
QRC24_HELD = {"s4": {("H11", 3407): "H13/H23 의 β=.2 반복이 우선 (§9; 미시작일 때만 보류)"},
              "s5": {("L070", 1103): "전체 LR 축소 효과가 두 seed 에서 일관되지 않음 (§1.4·§9; 미시작일 때만 보류)", ("L050", 1103): "위와 같음; seed 1103 은 결합 반복에 사용 (§9; 미시작일 때만 보류)"}}
QRC24_HELD_STATUS = "superseded_pending"                                                                                     # 계획상 상태(실패·완료 아님; runner 상태명 아님) — 편성·mandatory 에서 제외, 이미 시작/완료면 원 정의로 끝난다
QRC24_ADJ_ORDER = {"s1": [("G22", 3407, "v1"), ("G23", 3407, "v1"), ("A03_UNIF", 1234, "v2"), ("A03_SHUF", 1234, "v2"), ("A_UNIF", 3407, "v1"), ("A_FREEZE", 3407, "v1"), ("A_SHUF", 3407, "v1"), ("G21", 3407, "v1")],          # §5.1
                   "s2": [(p, 777, "v1") for p in ("G13", "E_UNIF", "E_SHUF", "ALL_UNIF", "G31", "G33")],                                                                                                                       # §5.2
                   "s3": [("G23", 4321, "v1"), ("H23", 4321, "v2"), ("B20A03", 4321, "v2")] + [(p, 4321, "v1") for p in ("G13", "G12", "G21", "G11", "G32", "G31", "G33")],                                                    # §5.3
                   "s4": [("H_BETA0", 1234, "v1"), ("H22", 3407, "v1"), ("H23", 3407, "v2"), ("H12", 3407, "v1"), ("H13", 3407, "v2"), ("H32", 3407, "v2"), ("H33", 3407, "v2"), ("G23", 1234, "v2"), ("B20A03", 1234, "v2")],   # §5.4
                   "s5": [("L070", 9091, "v1"), ("L050", 9091, "v1"), ("G23", 9091, "v2"), ("H23", 9091, "v2"), ("B20A03", 9091, "v2"), ("L100", 1103, "v1"), ("G23", 1103, "v2"), ("H23", 1103, "v2"), ("B20A03", 1103, "v2")]}   # §5.5


def _qrc24_active(server):
    """활성 편성 = 등록 완료(원계획 순서; 완료 history 유지 — runner 가 건너뛴다) + ADJ-R1 남은 순서(§5). 보류(QRC24_HELD) 는 빠진다."""
    rem = {(p, sd) for p, sd, _ in QRC24_ADJ_ORDER[server]}; held = set(QRC24_HELD.get(server, {}))
    return [(p, sd, "v1") for p, sd in QRC24_QUEUES_20260916[server] if (p, sd) not in rem and (p, sd) not in held] + list(QRC24_ADJ_ORDER[server])


QRC24_QUEUES = {srv: _qrc24_active(srv) for srv in QRC24_QUEUES_20260916}
QRC24_REGISTERED = {srv: [x for x in QRC24_QUEUES[srv] if (x[0], x[1]) not in {(p, sd) for p, sd, _ in QRC24_ADJ_ORDER[srv]}] for srv in QRC24_QUEUES}           # 부록 A 등록 39 (s1 6 · s2 6 · s3 10 · s4 10 · s5 7)
for _srv, _lst in QRC24_ADJ_ORDER.items():                                                                                    # 정합 검사 (import 시): 유지 항목은 원계획 안 · 추가(v2) 는 원계획 밖 · 보류는 원계획 안이고 순서에 없음 · 중복 없음
    _orig = set(QRC24_QUEUES_20260916[_srv])
    assert all(((p, sd) in _orig) == (v == "v1") for p, sd, v in _lst), f"ADJ-R1 {_srv}: v1 는 원계획 항목, v2 는 신규 항목이어야 한다"
    assert all(h in _orig and h not in {(p, sd) for p, sd, _ in _lst} for h in QRC24_HELD.get(_srv, {})), f"ADJ-R1 {_srv}: 보류 항목 불일치"
    assert len({(p, sd) for p, sd, _ in QRC24_QUEUES[_srv]}) == len(QRC24_QUEUES[_srv]), f"ADJ-R1 {_srv}: 중복 편성"
assert [len(QRC24_REGISTERED[s_]) for s_ in ("s1", "s2", "s3", "s4", "s5")] == [6, 6, 10, 10, 7] and sum(v == "v2" for l_ in QRC24_ADJ_ORDER.values() for _, _, v in l_) == 16 and sum(len(v) for v in QRC24_HELD.values()) == 3
QRC24_REFERENCE_H = {"s1": 2.20, "s2": 2.30, "s3": 1.35, "s4": 1.94, "s5": 2.17}; QRC24_RESERVE_SLACK = 1.20            # 원 §8.1 서버별 편성 대용 R_s(실측 아님) · §8.2 예약 = 1.20×R_s + 10/60. ADJ-R1 뒤 이 값은 **v1 config 의 projected_map 에만**(바이트 불변) — 예약·계획표는 QRC24_ADJ_REFERENCE_H
QRC24_ADJ_REFERENCE_H = {"s1": 2.42, "s2": 2.56, "s3": 1.47, "s4": 1.51, "s5": 2.58}                                    # ADJ-R1 §10 대용(이번 QRC24 Sheet Train(h); s5 는 느린 시나리오 2.58 — 빠른 1.46 은 큐 머리에 병기) — 남은 모든 QRC24 항목(v1·v2) 의 예약, 같은 case 실측이 없을 때; ledger 실측이 우선
QRC24_ADJ_REFERENCE_CASE_H = {"s1": {"A_FREEZE": 2.15}}                                                                    # §10: s1 frozen(A_FREEZE) 2.15 h (7×2.42 + 1×2.15 = 19.09 h)
QRC24_EXTRA = {"s1": (9091, ("G22", "A_UNIF", "A_FREEZE")), "s2": (3407, ("G12", "G22", "G32")), "s3": (1103, ("G21", "G22", "G23")), "s4": (2026, ("H22", "H12", "H11")), "s5": (2909, ("L100", "L070", "L050"))}   # §8.3 24h 미달 시 추가 3 run
QRC24_RESERVE_SEEDS = (17041, 26017)


def qrc24_run_name(server, profile, seed, version="v1"):
    return f"PAKD50_QRC24_{server.upper()}_{profile}_W104_D121_WV3_T0_S{seed}_FRESH50_{version}"


def qrc24_parse(case):
    """case 'QRC24_S3_G22' → ('s3', 'G22'); 아니면 None."""
    if not case.startswith("QRC24_"):
        return None
    parts = case.split("_"); srv = parts[1].lower(); prof = "_".join(parts[2:])
    if srv not in SERVER_SEED or prof not in QRC24_PROFILES:
        raise ValueError(f"QRC24 case 형식이 아니다: {case} (서버 {srv}, profile {prof})")
    return srv, prof


def qrc24_profile(prof):
    """profile → 전체 계수 dict (기본값 + 덮어쓰기): lam(절대 λE), rA, alpha, beta, ulr, a/e weight mode, frozen, alr(= rA×ulr; frozen 이면 0), canonical."""
    P = dict(QRC24_DEFAULT); P.update(QRC24_PROFILES[prof]); P["alr"] = (0.0 if P["frozen"] else float(f"{P['rA'] * P['ulr']:.10g}")); P["canonical"] = QRC24_CANONICAL.get(prof, prof); P["profile"] = prof
    return P


def qrc24_items(server):
    return [qrc24_run_name(server, p, sd, v) for p, sd, v in QRC24_QUEUES[server]]


def qrc24_held_runs(server):
    """보류 run id → 사유 (§9; v1 id — 원계획 항목이다)."""
    return {qrc24_run_name(server, p, sd): why for (p, sd), why in QRC24_HELD.get(server, {}).items()}


# Narrow R2 (research_log/PAN_QRC24_Narrow_R2_SeedLock_ERGAS_2026-09-17.md; revision QRC24_NARROW_R2_20260917): 넓은 grid 를 끝내고 마지막 β 비교(G23 β.1 ↔ B20A03 β.2) 만 닫은 뒤
# **설정 하나(C*) 를 동결**하고 사전 지정 seed 20 개를 돈다. 목표 2 순위는 SCC 가 아니라 **ERGAS**(selector HQNR9585_ERGAS2040_v2).
QRC24_R2_PLAN = "research_log/PAN_QRC24_Narrow_R2_SeedLock_ERGAS_2026-09-17.md"; QRC24_R2_REVISION = "QRC24_NARROW_R2_20260917"; QRC24_R2_SELECTOR = "HQNR9585_ERGAS2040_v2"
QRC24_LOCK_ID = "QRC24_LOCK_V1_20260917"; QRC24_LOCK_FILE = "work_dir/_qrecon24/recipe_lock.json"; QRC24_LOCK_CANDIDATES = ("G23", "B20A03")
# 2026-09-18 사용자 결정: **lock 게이팅을 없앤다.** 공동 목표 통과 seed 수를 확인하지 않고 각 서버가 지정 seed 를 바로 돈다.
# C* 는 R2 §4.3 의 잠정 기준 그대로 G23 로 고정한다(lock 파일이 있으면 그 profile 이 우선). lock 도구는 보고용으로만 남는다.
QRC24_SEED_PROFILE = "G23"
QRC24_R2_BETA_CLOSE = {"s1": [("B20A03", 1234), ("B20A03", 3407)], "s2": [("B20A03", 777)], "s3": [("B20A03", 2026)], "s4": [], "s5": []}     # §4.1 최대 4 run — 기존 G23 의 짝만 채운다(s3 4321 은 이미 있다). s4 는 G23·1234 host bridge 로 끝
QRC24_R2_BETA_VERSION = "v3"                                                                        # 편성 이력 표기일 뿐 수식 변경이 아니다 (§8.1)
QRC24_R2_SEEDS = {"s1": (41001, 41006, 41011, 41016), "s2": (41002, 41007, 41012, 41017), "s3": (41003, 41008, 41013, 41018),
                  "s4": (41004, 41009, 41014, 41019), "s5": (41005, 41010, 41015, 41020)}          # §6.2 사전 지정 seed 20 개 (서버마다 4)
QRC24_R2_SEED_MIN = 41000                                                                           # 이 이상 seed 는 lock 이 없으면 config 를 만들지 않는다 (§6.1)


def qrc24_recipe_lock():
    """동결된 공통 설정 C* (§6.1). 없으면 {} — seed 단계 config 생성이 **거부**된다."""
    p_ = os.path.join(ROOT, QRC24_LOCK_FILE)
    if not os.path.exists(p_):
        return {}
    try:
        return json.load(open(p_)) or {}
    except Exception:                                                                  # noqa
        return {}


def qrc24_locked_profile():
    lk = qrc24_recipe_lock(); return lk.get("profile")


def qrc24_beta_close_items(server):
    """§4.1 마지막 β 비교로 새로 돌릴 run (기존 G23 은 재사용하고 B20A03 짝만 채운다). 이미 완료/실행 중이면 호출자가 걸러낸다."""
    return [qrc24_run_name(server, p_, sd, QRC24_R2_BETA_VERSION) for p_, sd in QRC24_R2_BETA_CLOSE.get(server, [])]


def qrc24_seed_profile():
    """seed 단계가 쓰는 설정. lock 파일이 있으면 그 profile, 없으면 기본값 G23 — **더 이상 lock 을 요구하지 않는다**(2026-09-18 사용자 결정)."""
    return qrc24_locked_profile() or QRC24_SEED_PROFILE


def qrc24_seed_items(server, profile=None):
    """서버별 사전 지정 seed run (R2 §6.2 의 seed 배정). lock 없이 바로 편성된다."""
    prof = profile or qrc24_seed_profile()
    return [qrc24_run_name(server, prof, sd, "v1") for sd in QRC24_R2_SEEDS[server]]


def eval_hold():
    """평가 phase hold 상태 (계획 PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918 §2). {} 면 평상시.
    대기자·gate 가 매 pass 이 모듈을 새로 import 하므로, 이 함수 하나로 '새 학습 금지' 가 전 경로에 즉시 먹는다."""
    p_ = os.path.join(ROOT, "work_dir", "_eval_phase", "hold.json")
    if not os.path.exists(p_):
        return {}
    try:
        return json.load(open(p_)) or dict(phase="LOCAL_STUDENT_EVAL")
    except Exception:                                                                  # noqa
        return dict(phase="LOCAL_STUDENT_EVAL", reason="hold.json 손상 — 안전하게 hold 로 본다")


def run_started(run):
    """runner 기준 '시작됨': work_dir/<run>/{epoch-*, checkpoint-*} 가 있다 (tools/_run_cases.sh latest_ckpt 와 같은 규칙; last/ 는 세지 않는다). 재개 가능한 중단 run 을 '미시작' 으로 보지 않기 위한 것."""
    d = os.path.join(ROOT, "work_dir", run)
    return bool(glob.glob(os.path.join(d, "epoch-*")) or glob.glob(os.path.join(d, "checkpoint-*")))


def qrc24_held_state(server, run, ps=None):
    """보류 run 의 실제 상태 (ADJ-R1 §9 '미시작일 때만 보류'): terminal(완료/실패) · running_original_definition(지금 학습 중) · started_interrupted(체크포인트가 있고 미완 — 원 정의로 끝낸다) · superseded_pending(미시작 = 보류)."""
    from tools.campaign_gate import terminal as _terminal          # noqa
    if _terminal(run):
        return "terminal"
    if ps is None:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    if any("main.py" in l and "--config" in l and run in l for l in ps.splitlines()):
        return "running_original_definition"
    return "started_interrupted" if run_started(run) else QRC24_HELD_STATUS


def qrc24_held_states(server, ps=None):
    """{run: (state, reason)} — 보류 목록 전체."""
    if ps is None:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return {r: (qrc24_held_state(server, r, ps), w) for r, w in qrc24_held_runs(server).items()}


def qrc24_r2_items(server):
    """이어서 돌 run: ① **사전 지정 seed 4 개**(설정은 qrc24_seed_profile(), 전 서버 공통) ② 마지막 β 비교의 B20A03 짝(있으면; 더 이상 무엇도 막지 않는 참고 비교).
    2026-09-18 결정으로 lock·공동목표 확인 게이트가 없다 — 각 서버가 pull 하면 바로 돈다."""
    return qrc24_seed_items(server) + list(qrc24_beta_close_items(server))


def qrc24_effective_queue(server, ps=None):
    """이 서버에서 실제로 돌릴 순서: **시작됐는데 안 끝난 보류 run**(원 정의로 끝낸다 — ADJ-R1 §9) 을 맨 앞에, 그다음 활성 편성(qrc24_items), 그다음 **Narrow R2** (β-close → lock 후 seed).
    완료 run 은 runner 가 건너뛴다."""
    st = qrc24_held_states(server, ps); first = [r for r, (s_, _) in st.items() if s_ in ("running_original_definition", "started_interrupted")]
    seq = first + [r for r in qrc24_items(server) if r not in first]
    return seq + [r for r in qrc24_r2_items(server) if r not in seq]


def qrc24_adj_runs(server):
    """ADJ-R1 에서 추가된 v2 run id (부록 B)."""
    return [qrc24_run_name(server, p, sd, v) for p, sd, v in QRC24_ADJ_ORDER[server] if v == QRC24_ADJ_VERSION]


def qrc24_block_2x2(server, seed):
    """A LR × β 2×2 (§6.1; λE .002 · α 1 · U LR 1e-4): rA .01/β .1 = G22 alias · rA .03/β .1 = G23 · rA .01/β .2 = H23 · rA .03/β .2 = B20A03 — 같은 서버·seed 에 편성된 실제 id, 없으면 None."""
    cells = {"rA01_beta1": qrc24_control_profile(server, seed), "rA03_beta1": "G23", "rA01_beta2": "H23", "rA03_beta2": "B20A03"}
    return {k_: (qrc24_run_name(server, p_, seed, qrc24_item_version(server, p_, seed)) if qrc24_item_version(server, p_, seed) else None) for k_, p_ in cells.items()}


def write_qrc24_queue_file(server, path=None):
    """config/queues/qrecon24_<srv>.txt = 활성 편성(QRC24_QUEUES; 등록 완료 + ADJ-R1 순서) — 머리 주석에 revision·보류·시간 대용."""
    items = qrc24_items(server); held = qrc24_held_runs(server); adj = qrc24_adj_runs(server); reg = len(QRC24_REGISTERED[server]); rem = len(QRC24_ADJ_ORDER[server])
    t_fast = {"s5": " (빠른 시나리오 1.46 h 는 §10 병기; 예약은 느린 2.58 h)"}.get(server, "")
    hdr = (f"# QRECON24 {server} — ADJ-R1 (2026-09-17; 계획 {QRC24_ADJ_PLAN} §5·§9·§11, 노트 {QRC24_ADJ_NOTE}; queue_revision {QRC24_ADJ_REVISION}). 원계획 {QRC24_PLAN} §5 의 등록 완료 {reg} run 은 앞에 그대로(runner 가 건너뜀), "
           f"남은 편성 {rem} run(유지 {rem - len(adj)} v1 · 추가 {len(adj)} v2). 전부 W104·D121·T0 고정, FRESH50.\n"
           f"# method qrecon_continuous_v1: w = qref/(qref+q_T)(분자 1) · λE 절대 6e-4/2e-3/6e-3(환산값; 다시 2 배 하지 않음) · uniform 대조 0.5 · U ← H+K+λE·w·E / A ← w·H 만. 신규 profile B20A03(λE .002·rA .03·α 1·β .2) · A03_UNIF/A03_SHUF(G23 에서 A 가중만 .5/셔플).\n"
           f"# 보류(superseded_pending; 미시작일 때만 — 이미 시작/완료면 원 정의로 끝낸다): {', '.join(held) if held else '없음'}. 시간 대용 R_s {QRC24_ADJ_REFERENCE_H[server]} h{t_fast}, 예약 1.20×R_s + 10/60(ledger 실측 우선). 24h 는 원 campaign 누적 최소 운영구간(다시 세지 않음).\n"
           f"# 적용은 tools/qrecon24_switch.sh (chain 이 살아 있으면 case 경계 인계 파일; 진행 중 run 은 끝까지). v2 config 는 kdv.qrc24.queue_revision 을 갖고 시트 X열에 ADJ-R1 토큰이 붙는다.\n")
    path = path or os.path.join(ROOT, "config", "queues", f"qrecon24_{server}.txt"); open(path, "w").write(hdr + "\n".join(items) + "\n"); return items


def qrc24_extension_items(server):
    """§8.3: 신규 완료 Train(h) 합 < 24h 일 때 추가 3 run (지정 seed; 같은 server+profile+seed 결과가 있으면 다음 reserve seed)."""
    sd, profs = QRC24_EXTRA[server]; return [qrc24_run_name(server, p, sd) for p in profs]


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
          "NA": dict(pol="A-ID", proto="I-A", off=0.0, alr=None, noalign=True),
          # QRECON24 §2.6: Student 단계 offset/jitter 없음 — native 입력만(I-NATIVE-TRANSFER, radius 0). A 는 T0 복사 trainable(QRC; LR = rA×U LR, kdv_block 에서) 또는 frozen 대조(QRCF)
          "QRC": dict(pol="A-FT", proto="I-NATIVE-TRANSFER", off=0.0, alr=1e-6, radius=0.0), "QRCF": dict(pol="A-FR", proto="I-NATIVE-TRANSFER", off=0.0, alr=1e-6, radius=0.0)}
BASELINE_OF = {"J": "J0", "F": "F0", "AL": "AL0", "D": "D0", "DP": "D0", "LF": "LF0", "P": "J0", "JK0": "J0", "JE0": "J0", "RC": "RC0", "NA": "NA0", "QRC": "J0", "QRCF": "J0"}   # no-KD control (배정 §5–§6; RC 는 재배정 §5; NA 는 s4 이식 §4); QRC 는 kdv_block 에서 같은 seed G22 로 덮어쓴다
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
           "QER50": dict(rec="R3", edge=True, edge_route="low_q"), "QERS": dict(rec="R3", edge=True, edge_route="shuffle"),
           # EDGEBAL (2026-09-16 §3–§4): w_i(t) 만 다르다 — 상수 r 은 λE0 배율(lam_mult; EB_R3E050 ≡ J_QE025, EB_R3E200 ≡ J_QE10) · 시간배분 edge_schedule(0-based update 25000 분기; A 는 계속 학습) ·
           # 완만한 q 가중 edge_weight(w = high + (low−high)·g, g 는 기존 low_q/shuffle 0/1 표 그대로; FLOOR .75/.25 · REVERSE .25/.75 · SHUF 는 stratum 셔플 라벨). 동치: EB_N0 ≡ J0 · EB_R3E000 ≡ R3_NOEDGE · EB_R3E100 ≡ JQ · EB_N0E100 ≡ N0_EDGE · EB_R1E100 ≡ X02
           "EB_N0": dict(rec="N0", edge=False), "EB_R3E000": dict(rec="R3", edge=False), "EB_R3E025": dict(rec="R3", edge=True, lam_mult=0.25), "EB_R3E050": dict(rec="R3", edge=True, lam_mult=0.5),
           "EB_R3E075": dict(rec="R3", edge=True, lam_mult=0.75), "EB_R3E100": dict(rec="R3", edge=True), "EB_R3E200": dict(rec="R3", edge=True, lam_mult=2.0), "EB_N0E100": dict(rec="N0", edge=True), "EB_R1E100": dict(rec="R1", edge=True),
           "EB_EDOWN": dict(rec="R3", edge=True, edge_schedule=dict(before=1.0, after=0.5)), "EB_EUP": dict(rec="R3", edge=True, edge_schedule=dict(before=0.5, after=1.0)),
           "EB_QFLOOR": dict(rec="R3", edge=True, edge_weight=dict(mode="floor", low=0.75, high=0.25)), "EB_QFSHUF": dict(rec="R3", edge=True, edge_weight=dict(mode="floor_shuffle", low=0.75, high=0.25)),
           "EB_QFREV": dict(rec="R3", edge=True, edge_weight=dict(mode="floor", low=0.25, high=0.75))}
CASES = {"J0": ("J", "N0"), "JQ": ("J", "Q12"), "JR": ("J", "R1"), "XJ": ("J", "X02"), "F0": ("F", "N0"), "FQ": ("F", "Q12"), "FR": ("F", "R1"), "XF": ("F", "X02"), "AL0": ("AL", "N0"), "ALQ": ("AL", "Q12")}
for _p in ("J", "AL"):                                    # J_QA05 … J_QE10 (anchor JQ, no-KD J0) · AL_QA05 … AL_QE10 (§7 결합: anchor ALQ, no-KD AL0)
    for _v in ("QA05", "QB005", "QB02", "QE025", "QE10"):
        CASES[f"{_p}_{_v}"] = (_p, "Q12_" + _v[1:])
CASES.update({"D0": ("D", "N0"), "DQ": ("D", "Q12"), "DR": ("D", "R1"), "DX": ("D", "X02"), "PQ": ("P", "Q12"), "PR": ("P", "R1"), "PX": ("P", "X02"),
              "DPQ": ("DP", "Q12"), "DPX": ("DP", "X02"), "LF0": ("LF", "N0"), "LFQ": ("LF", "Q12"), "JK0": ("JK0", "Q12"), "JE0": ("JE0", "Q12"),
              "J_R3_NOEDGE": ("J", "R3_NOEDGE"), "J_N0_EDGE": ("J", "N0_EDGE"), "RC0": ("RC", "N0"), "RCQ": ("RC", "Q12"),     # 재배정 2026-09-15 §5
              "LFX": ("LF", "X02"), "NA0": ("NA", "N0"), "QE50": ("J", "QE50"), "QEC": ("J", "QEC"), "QES": ("J", "QES"),
              "QX50": ("J", "QX50"), "QEC3": ("J", "QEC3"), "QE50_B005": ("J", "QE50_B005"), "LFQE50": ("LF", "QE50"), "QER50": ("J", "QER50"), "QERS": ("J", "QERS")})     # QEGX 2026-09-15 §4
CASES.update({c: ("J", c) for c in EDGEBAL_CASES})                                                                                                                     # EDGEBAL 2026-09-16 §4 (전부 J 정책)
BACKEND["QRC24"] = dict(rec="R3", edge=True)                                                                                                                             # QRECON24: R3(α·β 는 profile) + GT edge(λE 절대값; kdv_block) + kdv.qrecon
CASES.update({f"QRC24_{srv.upper()}_{prof}": (("QRCF" if QRC24_PROFILES[prof].get("frozen") else "QRC"), "QRC24") for srv in SERVER_SEED for prof in QRC24_PROFILES})                                                                                              # s3 추가 2026-09-15 §4.3: 0–24999 XJ 와 같은 joint, 25000 부터 A 동결(offset 중단), U 는 X02(R1 + λE edge; soft 없음) 50K
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
                "QERS": "QEGX s4 대조(edge_route shuffle): QER50 의 A gate 를 QES 와 같은 stratum 셔플 라벨로 — routing 의 실제 q–sample 연결 대조",
                "EB_N0": "EDGEBAL: joint + plain GT (J0 와 학습 동치; 반복 seed 의 no-KD 기준)", "EB_R3E000": "EDGEBAL s2: joint + R3 adaptive, GT edge 없음 (J_R3_NOEDGE 와 동치) — JQ 와의 차 = edge 만의 효과",
                "EB_R3E025": "EDGEBAL s2: Q12 인데 GT edge 계수 0.25×λE0 (모든 patch)", "EB_R3E050": "EDGEBAL s2·s5: Q12 인데 GT edge 계수 0.5×λE0 (모든 patch; J_QE025 와 동치) — q 로 절반을 고르는 QE50 과 nominal 강도 대조",
                "EB_R3E075": "EDGEBAL s5: Q12 인데 GT edge 계수 0.75×λE0 (DOWN/UP 의 산술평균 계수 대조)", "EB_R3E100": "EDGEBAL: Q12 그대로 λE0 (JQ 와 학습 동치; 반복 seed 의 기존 Q12 기준)", "EB_R3E200": "EDGEBAL s2: Q12 인데 GT edge 계수 2×λE0 (J_QE10 과 동치)",
                "EB_N0E100": "EDGEBAL: joint + plain GT + λE0 GT edge, Teacher 를 학습 loss 에 쓰지 않음 (J_N0_EDGE 와 동치; A 는 여전히 T0 복사)", "EB_R1E100": "EDGEBAL s2: joint + (1+αd_T)L1 + λE0 edge, output soft 없음 (XJ 와 동치)",
                "EB_EDOWN": "EDGEBAL s5: Q12 인데 GT edge 계수 w(t) = 1 (t<25000) → 0.5 (t≥25000); A·U 계속 학습, optimizer/cosine 재시작 없음 (평균 계수 0.75)", "EB_EUP": "EDGEBAL s5: EDOWN 의 반대 순서 0.5 → 1",
                "EB_QFLOOR": "EDGEBAL s5: Q12 인데 GT edge 계수 w_i = 0.25 + 0.5·g_i (q_T<θq 0.75 / q-high 0.25; q-high 를 끄지 않는 완만한 cue)", "EB_QFSHUF": "EDGEBAL s5 대조: FLOOR 의 g 를 (e_roi32 decile × aug state) stratum 셔플 라벨(51515) 로",
                "EB_QFREV": "EDGEBAL s5 대조: FLOOR 의 방향 반전 (q-low 0.25 / q-high 0.75)"}
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
QEDGE9_S5_ITEMS = ["J0@W104_D121", "JQ@W104_D121", "QE50@W104_D121", "PAKD50_J0_W104_D121_WV3_T0_S777_FRESH50_v1", "PAKD50_JQ_W104_D121_WV3_T0_S777_FRESH50_v1", "PAKD50_QE50_W104_D121_WV3_T0_S777_FRESH50_v1"]
QEDGE9_S1_ITEMS = ["PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v2", "PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v2", "PAKD50_QE50_W104_D121_WV3_T0_S1234_FRESH50_v2", "PAKD50_QES_W104_D121_WV3_T0_S1234_FRESH50_v2", "PAKD50_QEC_W104_D121_WV3_T0_S1234_FRESH50_v2"]   # QEDGE9 s1(seed 1234, v2; 09-16 시트: 5 벌 완료)
# EDGEBAL (2026-09-16 §5–§6): s2 12 run v1(777 A 6 → 2026 B 3 → 9091 C 3) · s5 = QEDGE9 6 항목 뒤 14 run v2(2026 A 8 → 777 B 2 → 9091 C 4). 전부 run 이름 항목. (09-16 저녁 QRECON24 로 교체 — 미완 항목은 extra_priority 로만)
EDGEBAL_S2_ITEMS = [f"PAKD50_{c}_W104_D121_WV3_T0_S777_FRESH50_v1" for c in ("EB_R3E000", "EB_R3E050", "EB_R3E025", "EB_R3E200", "EB_N0E100", "EB_R1E100")] + [f"PAKD50_{c}_W104_D121_WV3_T0_S{sd}_FRESH50_v1" for sd in (2026, 9091) for c in ("EB_N0", "EB_R3E100", "EB_R3E050")]
EDGEBAL_S5_ITEMS = ([f"PAKD50_{c}_W104_D121_WV3_T0_S2026_FRESH50_v2" for c in ("EB_R3E050", "EB_R3E075", "EB_EDOWN", "EB_EUP", "EB_QFLOOR", "EB_QFSHUF", "EB_QFREV", "EB_N0E100")]
                    + [f"PAKD50_{c}_W104_D121_WV3_T0_S777_FRESH50_v2" for c in ("EB_R3E075", "EB_EDOWN")] + [f"PAKD50_{c}_W104_D121_WV3_T0_S9091_FRESH50_v2" for c in ("EB_N0", "EB_R3E100", "EB_R3E075", "EB_EDOWN")])
# QEGX (2026-09-15 저녁 §5–§6): s3 15 run(전부 v2) · s4 = E0 5 뒤 14 run v1. (09-16 저녁 QRECON24 로 교체)
QEGX_S3_ITEMS = [f"PAKD50_{c}_W104_D121_WV3_T0_S2026_FRESH50_v2" for c in ("J0", "JQ", "QE50", "XJ", "QX50", "J_R3_NOEDGE", "QEC3", "QES", "J_QB005", "QE50_B005")] + [f"PAKD50_{c}_W104_D121_WV3_T0_S4321_FRESH50_v2" for c in ("J0", "JQ", "XJ", "QE50", "QX50")]
QEGX_S4_ITEMS = (["NA0@W104_D121", "J0@W104_D121", "JQ@W104_D121", "XJ@W104_D121", "F0@W104_D121"] + [f"PAKD50_{c}_W104_D121_WV3_T0_S1234_FRESH50_v1" for c in ("QE50", "LF0", "LFQ", "LFQE50", "LFX", "JE0", "QER50", "QERS")]
                 + [f"PAKD50_{c}_W104_D121_WV3_T0_S3407_FRESH50_v1" for c in ("J0", "JQ", "QE50", "JE0", "QER50", "QERS")])
# QRECON24 (2026-09-16 저녁 §5): 다섯 서버 전부 새 명시 큐(68 run; 서버·profile·seed 사전 고정). 이전 QEDGE9/QEGX/EDGEBAL 의 진행 중 run 은 정상 종료(runner 가 끝낸다), 미완 항목은 superseded (필요하면 extra_priority 에 run 이름으로).
PRIORITY_BY_SERVER = {srv: qrc24_items(srv) for srv in ("s1", "s2", "s3", "s4", "s5")}
MANDATORY_BY_SERVER = dict(PRIORITY_BY_SERVER)        # 기본 run 전부가 예산 예약 대상 (완료된 run 은 trainer 가 0 으로 센다)
QEDGE9_ITEMS_BY_SERVER = {"s5": list(QEDGE9_S5_ITEMS), "s1": list(QEDGE9_S1_ITEMS)}     # 이 항목들만 QEDGE9 branch(campaign/budget); s4 는 기존 E0 allocation 그대로(QEDGE9 없음)
EDGEBAL_ITEMS_BY_SERVER = {"s2": list(EDGEBAL_S2_ITEMS), "s5": list(EDGEBAL_S5_ITEMS)}
QRC24_ITEMS_BY_SERVER = {srv: list(PRIORITY_BY_SERVER[srv]) for srv in PRIORITY_BY_SERVER}
QEGX_ITEMS_BY_SERVER = {"s3": list(QEGX_S3_ITEMS), "s4": [it for it in QEGX_S4_ITEMS if it.startswith("PAKD50_")] + QEGX_REFRESH_CONTROLS["s4"]}     # QEGX branch 항목 (s4 의 E0 5 항목 제외; 대조 새로고침 v3 포함)
PREVIOUS_PRIORITY_BY_SERVER = {"s4": ["F0", "RC0", "RCQ", "JR", "XJ", "J_R3_NOEDGE"], "s4_20260914": ["J0", "JQ", "AL0", "ALQ"], "s5": ["J0", "JQ", "D0", "DQ", "PQ"], "s3": list(PRIORITY),
                               "s5_20260915": ["PQ", "F0", "LF0", "LFQ", "RC0", "RCQ"],      # s5 재배정(2026-09-15 아침) — 시트 확인 결과 전부 완료 → QEDGE9 로 교체
                               "s3_20260915": ["J_R3_NOEDGE", "J_N0_EDGE", "LF0", "LFQ", "LFX"],   # s3 추가(2026-09-15 낮; W112 seed 2026) — 완료(§9.1 실측 있음) → QEGX 로 교체
                               "s2_20260915": ["JR", "XJ", "J_R3_NOEDGE", "J_N0_EDGE"],             # s2 재배정(2026-09-15; W112 seed 777) — 완료(12:00 시트) → EDGEBAL 로 교체 (09-16)
                               "s4_20260915_e0": ["NA0@W104_D121", "J0@W104_D121", "JQ@W104_D121", "XJ@W104_D121", "F0@W104_D121"],
                               # 09-16 저녁 QRECON24 로 교체된 직전 순서 (QRECON24 §부록 A 시트: s1 5/5 · s2 EB 4/12 · s3 14/15 · s4 13/14 · s5 5/14 완료)
                               "s1_20260915_qedge9": list(QEDGE9_S1_ITEMS), "s2_20260916_edgebal": list(EDGEBAL_S2_ITEMS), "s3_20260915_qegx": list(QEGX_S3_ITEMS), "s4_20260915_qegx": list(QEGX_S4_ITEMS),
                               "s5_20260916_edgebal": list(QEDGE9_S5_ITEMS) + list(EDGEBAL_S5_ITEMS)}   # s4 E0 이식 묶음 — 완료; 현재 순서 앞에 그대로 남겨 control 검증·terminal 표시
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
                          # s5 W104: EDGEBAL §7.1 Sheet 실측(seed 2026 J0 1.38 / JQ 1.57 / QE50 1.65; 종전 QEDGE9 §9.2 의 W112 대용 1.35/1.36/1.50 교체) · §7.2 dense/schedule/GT+edge 1.57 · cached floor 1.65 · plain N0 1.38 — 실측 아님
                          "s5": {"J0@W104_D121": 1.38, "JQ@W104_D121": 1.57, "QE50@W104_D121": 1.65,
                                 **{f"{c}@W104_D121": 1.57 for c in ("EB_R3E050", "EB_R3E075", "EB_R3E100", "EB_EDOWN", "EB_EUP", "EB_N0E100")}, **{f"{c}@W104_D121": 1.65 for c in ("EB_QFLOOR", "EB_QFSHUF", "EB_QFREV")}, "EB_N0@W104_D121": 1.38},
                          # s2 W104: EDGEBAL §7.1–7.2 (seed 777 J0 1.95 / JQ 2.30 / QE50 2.29) — R3/edge 형 2.30 · plain N0 1.95 (빨라질 가능성 사전 차감 없음)
                          "s2": {"J0@W104_D121": 1.95, "JQ@W104_D121": 2.30, "QE50@W104_D121": 2.29, "EB_N0@W104_D121": 1.95,
                                 **{f"{c}@W104_D121": 2.30 for c in ("EB_R3E000", "EB_R3E025", "EB_R3E050", "EB_R3E100", "EB_R3E200", "EB_N0E100", "EB_R1E100")}},
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
    for lst in (PRIORITY_BY_SERVER.get(server, []), QEDGE9_ITEMS_BY_SERVER.get(server, []), QEGX_ITEMS_BY_SERVER.get(server, []), EDGEBAL_ITEMS_BY_SERVER.get(server, [])):     # 이전 branch 의 명시 seed 도 본 실험 seed (superseded 라도 실측·예약 종류는 그대로)
        for it in lst:
            if is_tag(it):
                out.add(seed_of(it))
    return out


def branch_for(server, item, seed=None):
    """편성 항목의 branch: QEGX 전용 case(QEGX_CASES) 이거나 서버 QEGX 목록(QEGX_ITEMS_BY_SERVER) 의 run 이면 'QEGX';
    QEDGE9 case(QE50/QEC/QES) 이거나 서버 QEDGE9 목록의 run 이면 'QEDGE9'; 아니면 None(기본 PAKD50 계약).
    QEGX §12 검사: s3/s4 큐의 QE50/QES 는 run 이름 항목이라 QEGX 목록이 먼저 잡힌다 — bare 'QE50@W104_D121' 은 QEDGE9 자동 규칙(9h ledger) 으로 가므로 QEGX 큐에는 run 이름만 쓴다."""
    case, arch, sd, ver = parse_item(item)
    if case.startswith("QRC24_"):
        return "QRECON24"
    if case in EDGEBAL_CASES:
        return "EDGEBAL"
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
    if case not in CUE_CASES and not case.startswith("QRC24_"):
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


def reference_hours(server, case, measured=None, confirm=False, version=None):
    """run 의 시간 산정 기준 h 와 출처. measured: 같은 서버 완료 run 의 실측 {case: h} (ledger) — 같은 case 실측이 있으면 그것이 우선(§3 '첫 실측이 나오면 곧바로 교체').
    없으면: confirm(확인 seed) → 1.80 가예약(§7) · ROUTING → 같은 서버 routing 실측 평균 → 없으면 1.80 가예약 · N0/T → 서버 계획 기준값 (s2 는 전부 2.33) → 없으면 실측 평균 → None."""
    measured = measured or {}; key = item_key(case); case = case_of(case)
    if measured.get(key):
        return float(measured[key]), "measured_same_case"
    if case.startswith("QRC24_"):                                             # ADJ-R1 §10 대용 R_s (v1·v2 공통; 실측 아님 — 같은 case@arch 실측이 나오면 위에서 교체). 원 §8.1 R_s(QRC24_REFERENCE_H) 는 v1 config 의 projected_map 에만 남는다
        prof_ = qrc24_parse(case)[1]
        return float(QRC24_ADJ_REFERENCE_CASE_H.get(server, {}).get(prof_, QRC24_ADJ_REFERENCE_H[server])), "plan_reference_qrc24_adj_r1"
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


def reservation_hours(ref, slack=RESERVE_SLACK):
    """slot 예약 (재배정 §3): 1.10 × reference_train_h + 10/60. QRECON24 §8.2 는 slack 1.20 (신규 gradient 경로 미실측 여유)."""
    return float(slack) * float(ref) + RESERVE_POST_H


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
    for lp in (LEDGER, QEDGE9_LEDGER, QEGX_LEDGER, EDGEBAL_LEDGER, QRC24_LEDGER):
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


GRID_STEPS_50K = tuple(range(1010, 50000, 1010)) + (50000,)                    # GRID1010_50K_v1: 1010 간격 49 + exact 50000 = 50 후보


def verified_complete(run, expect_step=50000, min_candidates=50):
    """완료 marker(results .mat 두 개) 를 넘어 계획 §8.2·§8.3 / QRECON24 §7.3·§10.1 의 완결·동치 검사 (QEDGE9 감사 F06 → QRECON24 감사 F08 강화):
    results mat 두 개가 비어 있지 않음 · exact-50K last state · 후보 격자 = GRID1010_50K_v1 의 **고유** step 50 개 전부 + 각 step 의 실제 checkpoint(candidates/step-N/model.safetensors) · kdv manifest ·
    Teacher 를 쓰는 run 이면 Teacher 파일 sha == T0 **필수** · cue 자산이 있으면 train h5 sha == cue **필수** · 저장 U 초기값 파일이 있으면 hash 일치 **필수** · exact_resume run 은 비정확 재개(resumed_nonexact) 없음.
    반환 dict(ok, checks{name: bool|None}, notes) — None 은 그 검사가 해당 없음(예: Teacher 없는 run) 이고 ok 에서 제외; 필요한 증거가 없는데 판단 불가면 False."""
    rd = os.path.join(ROOT, "work_dir", run); c = {}; notes = []
    _nonempty = lambda p_: os.path.exists(p_) and os.path.getsize(p_) > 1024
    c["results_mats"] = _nonempty(os.path.join(rd, "results", "reduced_best_hqnr.mat")) and _nonempty(os.path.join(rd, "results", "full_best_hqnr.mat"))
    lm = _load_json(os.path.join("work_dir", run, "last_meta.json")); c["last_exact_step"] = (lm.get("step") == expect_step) and _nonempty(os.path.join(rd, "last", "model.safetensors"))
    cm = os.path.join(rd, "checkpoint_metrics.csv"); rows = []
    if os.path.exists(cm):
        import csv as _csv
        rows = list(_csv.DictReader(open(cm)))
    steps = sorted({int(float(r["step"])) for r in rows if r.get("step")}); grid = [s_ for s_ in GRID_STEPS_50K if s_ <= expect_step]
    c["candidate_grid"] = (expect_step in steps) and (len(steps) >= min_candidates) and all(s_ in steps for s_ in grid)
    missing_ck = [s_ for s_ in steps if not _nonempty(os.path.join(rd, "candidates", f"step-{s_}", "model.safetensors"))]
    c["candidate_checkpoints"] = (len(steps) > 0) and not missing_ck
    if missing_ck:
        notes.append(f"후보 checkpoint 없음 {len(missing_ck)} step (예 {missing_ck[:3]})")
    kc = _load_json(os.path.join("work_dir", run, "kdv_config_resolved.json")); c["kdv_manifest"] = bool(kc)
    if kc:
        c["exact_resume_ok"] = (None if not kc.get("exact_resume") else not bool(kc.get("resumed_nonexact")))
    else:
        c["exact_resume_ok"] = False; notes.append("kdv manifest 없음")
    ith = _load_json(os.path.join("work_dir", run, "init_and_teacher_hashes.json")); t = (ith.get("teacher") or {}); needs_t = bool((kc.get("teacher") or {}) if kc else False)
    if t:
        sha, _ = t0_identity(None); c["teacher_is_T0"] = (t.get("file_sha256") == sha)
    elif needs_t:
        c["teacher_is_T0"] = False; notes.append("Teacher 를 쓰는 run 인데 Teacher hash 기록이 없다")
    else:
        c["teacher_is_T0"] = None; notes.append("Teacher 없음(no-KD/no-align run)")
    ds = _load_json(os.path.join("work_dir", run, "dataset_hashes.json")); cue = _load_json(QEDGE9_CUE_ASSET)
    if cue:
        c["train_sha_matches_cue"] = bool(ds) and ((ds.get("train_feeder_args") or {}).get("sha256") == (cue.get("dataset") or {}).get("train_sha256"))
        if not ds:
            notes.append("dataset_hashes 없음")
    else:
        c["train_sha_matches_cue"] = None; notes.append("cue 자산 없음(이 서버)")
    ih = _load_json(os.path.join("work_dir", run, "initialization_hashes.json")); init_f = ih.get("unet_init_file")
    if ih.get("unet_init_sha256_16") and init_f and os.path.exists(init_f):
        try:
            import torch
            from train_pa import _sha_tensors
            c["init_matches_shared_file"] = (_sha_tensors(torch.load(init_f, map_location="cpu")) == ih["unet_init_sha256_16"])
        except Exception as ex:                                                        # noqa
            c["init_matches_shared_file"] = False; notes.append(f"init 파일 검사 실패 {ex!r}")
    elif ih.get("unet_init_sha256_16"):
        c["init_matches_shared_file"] = None; notes.append("저장 초기값 파일이 이 서버에 없다(hash 기록만)")
    else:
        c["init_matches_shared_file"] = False; notes.append("초기값 hash 기록 없음")
    required = ("results_mats", "last_exact_step", "candidate_grid", "candidate_checkpoints", "kdv_manifest")
    ok = all(c[k_] for k_ in required) and all(v for v in c.values() if v is not None)
    return dict(run=run, ok=bool(ok), checks=c, notes=notes, required=list(required))


def reservation_for(server, item, measured=None, seed=None):
    """편성 항목(case id 또는 run 이름) → dict(run, case, seed, reference_train_h, reference_kind, reservation_h, gate_hours)."""
    seed = seed or SERVER_SEED[server]; case, arch, sd, ver = parse_item(item); sd = sd if sd is not None else seed; confirm = (sd not in allowed_seeds(server))
    ref, kind = reference_hours(server, (case if arch == ARCH_DEFAULT else f"{case}@{arch}"), measured, confirm=confirm, version=ver)
    if ref is None:
        return None
    slack = QRC24_RESERVE_SLACK if case.startswith("QRC24_") else RESERVE_SLACK
    res = reservation_hours(ref, slack)
    return dict(run=to_tag(item, seed), case=case, arch=arch, seed=sd, reference_train_h=ref, reference_kind=kind, reservation_h=res, gate_hours=res / MARGIN, slack=slack)


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
_QRC_TXT = {"A_UNIF": "A 의 hard 가중 = 0.5(= s_q(qref); q 배분 제외; edge 는 w(q))", "A_FREEZE": "T0 A 처음부터 동결(optimizer 밖; U 만 학습)", "A_SHUF": "A 의 w(q) 를 (e decile × rot) stratum 셔플(51515)", "E_UNIF": "U edge 가중 = 0.5(= s_q(qref); A 는 w(q))",
            "E_SHUF": "U edge 의 w(q) 를 stratum 셔플", "ALL_UNIF": "A·edge 모두 0.5(= s_q(qref); q 없는 같은 loss 종류·직접 λE 기준)", "H_ALPHA0": "α = 0 (추가 hard 재가중 제거; GT hard 는 1)", "H_BETA0": "β = 0 (직접 soft 제거 대조)",
            # ADJ-R1 2026-09-17 §4 (같은 서버·seed 의 A LR × β 2×2: G22 alias / G23 / H23 / B20A03)
            "B20A03": "ADJ-R1: G23(rA .03) 에서 β 만 .1→.2 (= H23 에서 rA 만 .01→.03); A LR×β 2×2 의 결합점 — 같은 q 식·gradient 경로", "A03_UNIF": "ADJ-R1: G23(rA .03·β .1) 에서 A 의 hard 가중만 상수 0.5(= s_q(qref)); edge 는 w(q) — 같은 LR 에서 A 의 실제 q 제거",
            "A03_SHUF": "ADJ-R1: G23(rA .03·β .1) 에서 A 의 w(q) 만 (e decile × rot) stratum 셔플(51515); edge 는 w(q) — 같은 LR 에서 q–sample 연결 검증"}
for _srv in SERVER_SEED:
    for _pf, _pd in QRC24_PROFILES.items():
        _P = qrc24_profile(_pf)
        PURPOSE[f"QRC24_{_srv.upper()}_{_pf}"] = (f"QRECON24 {_srv} {_pf}" + (f"(= {_P['canonical']})" if _P['canonical'] != _pf else "") + f": U ← H+K+λE·w·E(GT edge, λE 절대 {_P['lam']:g}) / A ← w·H 만(hard-only, offset 없음) · "
                                                  f"w = qref/(qref+q_T) · α {_P['alpha']:g} β {_P['beta']:g} · U LR {_P['ulr']:g} / A LR {_P['alr']:g}(rA {_P['rA'] if not _P['frozen'] else 0})" + (f" · {_QRC_TXT[_pf]}" if _pf in _QRC_TXT else ""))
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
    branch = branch or ("QRECON24" if case.startswith("QRC24_") else ("EDGEBAL" if case in EDGEBAL_CASES else ("QEGX" if case in QEGX_CASES else ("QEDGE9" if B.get("edge_gate") else None))))     # gate/route/EB/QRC case 는 branch 가 있어야 한다 (W104 전용; 아래 검사)
    if (branch == "QRECON24") != case.startswith("QRC24_"):
        raise SystemExit(f"!! {case}: QRC24 case 는 QRECON24 branch 로만, 다른 case 는 QRECON24 branch 로 만들지 않는다 (현재 {branch})")
    if branch != "QEGX" and (case in QEGX_CASES or B.get("edge_route")):
        raise SystemExit(f"!! {case}: QEGX 전용 case 는 QEGX branch 로만 만든다 (현재 {branch})")
    if branch != "EDGEBAL" and (case in EDGEBAL_CASES or B.get("edge_schedule") or B.get("edge_weight")):
        raise SystemExit(f"!! {case}: EDGEBAL 전용 case 는 EDGEBAL branch 로만 만든다 (현재 {branch})")
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
    if branch == "QRECON24":                                                # QRECON24 §2–§4·§8–§9: 사용자 확정 method — 연속 q 가중, U/A 목적함수 분리, offset 없음, λE 절대값, A LR = rA×U LR
        if arch != QRC24_ARCH:
            raise SystemExit(f"!! {case}: QRECON24 는 {QRC24_ARCH} 전용 (현재 {arch})")
        qsrv, prof = qrc24_parse(case)
        if qsrv != server:
            raise SystemExit(f"!! {case}: 서버 토큰 {qsrv} ≠ 생성 서버 {server} (이름의 서버 토큰은 파일 충돌 방지용 — 그 서버에서만 만든다)")
        Pq = qrc24_profile(prof)
        if int(seed) >= QRC24_R2_SEED_MIN:                                    # seed 단계 — 2026-09-18 결정으로 lock 요구를 없앴다. 설정은 qrc24_seed_profile()(기본 G23) 하나로 전 서버 공통
            _lk = qrc24_recipe_lock()
            k.setdefault("qrc24_lock", dict(lock_id=(_lk.get("lock_id") if _lk else None), profile=qrc24_seed_profile(), gated=False,
                                            note="lock 게이팅 없음(2026-09-18) — 공동 목표 통과 seed 수를 확인하지 않고 지정 seed 를 돈다. 설정은 전 서버 동일"))
        k["rec"] = dict(case="R3", alpha=float(Pq["alpha"]), kd_weight=float(Pq["beta"]), eps=1.0e-6, tau=(float(cal["tau_R"]) if (pin and cal.get("tau_R")) else "calibrate"), eps_scale=1.0e-6)
        k["stat"] = dict(enabled=True, kind="EDGE", mode="H", window=5, alpha=1.0, kd_weight=0.1, tau="calibrate", outer_weight=float(Pq["lam"]), lambda_note="absolute (QRECON24 §2.4; not a multiple of lambda_E0)", ramp_updates=0)
        if not Pq["frozen"]:
            k["aligner_lr"] = float(Pq["alr"])
        else:
            k.pop("aligner_lr", None)
        k["qrecon"] = dict(mode="continuous_v1", q_ref=QRC24_QREF, asset=QEDGE9_CUE_ASSET, a_weight=Pq["a"], e_weight=Pq["e"], perm_seed=51515, uniform_weight=QRC24_UNIFORM_W)
        k["qrc24"] = dict(profile=prof, canonical=Pq["canonical"], lambda_E=float(Pq["lam"]), rA=(0.0 if Pq["frozen"] else float(Pq["rA"])), U_lr=float(Pq["ulr"]), A_lr=float(Pq["alr"]), alpha=float(Pq["alpha"]), beta=float(Pq["beta"]),
                       a_weight=Pq["a"], e_weight=Pq["e"], A_frozen=bool(Pq["frozen"]), q_ref=QRC24_QREF, method="qrecon_continuous_v1", A_loss="weighted_H_only", A_soft=0, A_edge=0, student_offset=0,
                       q_weight_formula="qref/(qref+q_T)", lambda_E_plan=float(Pq["lam"]) / 2.0, uniform_weight=QRC24_UNIFORM_W, change_note="2026-09-16 저녁: 분자 2 제거 → λE 2 배 환산, uniform 0.5; rA/α/β/qref 유지")
        _adj = (version == QRC24_ADJ_VERSION)
        if int(seed) >= QRC24_R2_SEED_MIN:                                    # R2 seed 단계: seed 별 G22 대조가 없다(같은 C* 를 seed 만 바꿔 20 번 돈다) — 가상 id 를 만들지 않고 lock 과 비교 block 을 가리킨다
            _lk2 = qrc24_recipe_lock()
            _ctl = None
            _ctrl = {"recipe_lock": _lk2.get("lock_id"), "locked_profile": _lk2.get("profile"), "canonical": None, "control_profile": None,
                     "reference_block": [qrc24_run_name(server, prof, sd_, "v1") for _p, sd_, _v in QRC24_QUEUES.get(server, []) if _p == prof][:1] or None,
                     "note": "seed 단계는 같은 C* 의 seed 반복이다 — 대조는 같은 lock 의 다른 seed 들(§6.3 분포 보고)이지 seed 별 G22 가 아니다"}
        else:
            _ctl = qrc24_control_run(server, seed)                             # 대조 id 는 편성된 실제 version(v1) — v2 run 이 가상 '..._v2' G22 를 가리키지 않는다 (ADJ-R1 §11.2)
            _ctrl = {"G22": _ctl, "canonical": "G22", "control_profile": qrc24_control_profile(server, seed)}
        if _adj:                                                              # ADJ-R1: 같은 서버·seed 의 2×2/대조 상대(실제 id) + revision·출처 (v1 config 는 바이트 불변)
            _ctrl.update({p_: qrc24_control_run(server, seed, p_) for p_ in ("G23", "H23", "B20A03") if p_ != prof and qrc24_item_version(server, p_, seed)})
            k["qrc24"].update(queue_revision=QRC24_ADJ_REVISION, adjustment="ADJ-R1", source_plan=QRC24_ADJ_PLAN, block_2x2=qrc24_block_2x2(server, seed), numerator_factor=1.0, lambda_E_plan=float(Pq["lam"]),
                              lambda_E_plan_note="ADJ-R1 §2·§4 는 환산값(.0006/.002/.006) 을 직접 정의 — 다시 2 배 하지 않는다 (원계획 09-16 의 절반값이 아니다)",
                              adj_note="2026-09-17 ADJ-R1: 미시작 순서 재편 + 기존 범위 안의 결합/대조 추가; loss·gradient 경로·Teacher·q·λE 환산·uniform .5 불변")
        k.update(campaign_id=QRC24_CAMPAIGN_ID, parent_campaign_id=CAMPAIGN_ID, lineage_campaign_ids=[CAMPAIGN_ID, QEDGE9_CAMPAIGN_ID, QEGX_CAMPAIGN_ID, EDGEBAL_CAMPAIGN_ID], experiment_branch_id=QRC24_BRANCH, exact_resume=True,
                 control_runs=_ctrl, baseline_run=_ctl,
                 budget=dict(ledger=QRC24_LEDGER, total_gpu_hours=QRC24_SOFT_HOURS, reserve_hours=0.0, margin=MARGIN, required=True, projected_hours=projected, projected_map={me: reservation_hours((QRC24_ADJ_REFERENCE_H if _adj else QRC24_REFERENCE_H)[server], QRC24_RESERVE_SLACK)},
                             remaining_mandatory=[], remaining_mandatory_file=QRC24_MANDATORY_FILE, projection_file=RESERVATION_FILE, time_policy=dict(QRC24_TIME_POLICY)))
    if branch == "EDGEBAL":                                                 # EDGEBAL §10.1–10.2·§7: 새 캠페인(parent PAKD50, lineage QEDGE9·QEGX), 시간 상한 없음(total 1000h + required 경고만, 절대 마감·9h·50h 미상속), exact_resume
        if arch != EDGEBAL_ARCH:
            raise SystemExit(f"!! {case}: EDGEBAL 은 {EDGEBAL_ARCH} 전용 (현재 {arch})")
        ctl = (EDGEBAL_CONTROLS.get(server) or {}).get(seed) or {c: run_name(("J0" if c == "J0" else ("JQ" if c == "JQ" else "QE50")), seed, version, arch=arch) for c in ("J0", "JQ", "QE50")}
        ctl = dict(ctl); ctl.update({k: (run_name({"J0": "EB_N0", "JQ": "EB_R3E100"}[k], seed, version, arch=arch) if k in ("J0", "JQ") and seed not in (EDGEBAL_CONTROLS.get(server) or {}) else v) for k, v in ctl.items()})
        k.update(campaign_id=EDGEBAL_CAMPAIGN_ID, parent_campaign_id=CAMPAIGN_ID, lineage_campaign_ids=[CAMPAIGN_ID, QEDGE9_CAMPAIGN_ID, QEGX_CAMPAIGN_ID], experiment_branch_id=EDGEBAL_BRANCH, exact_resume=True,
                 control_runs=ctl,                                              # §10.4 Notes control_run_id: 그 서버·seed 의 대조(기존 QEDGE9 J0/JQ/QE50 또는 같은 큐의 EB_N0/EB_R3E100)
                 budget=dict(ledger=EDGEBAL_LEDGER, total_gpu_hours=EDGEBAL_SOFT_HOURS, reserve_hours=0.0, margin=MARGIN, required=True, projected_hours=projected, projected_map={me: EDGEBAL_RESERVED_H},
                             remaining_mandatory=[], remaining_mandatory_file=EDGEBAL_MANDATORY_FILE, projection_file=RESERVATION_FILE, time_policy=dict(EDGEBAL_TIME_POLICY)))
        if B.get("edge_schedule"):                                          # §3.3: w(t) piecewise (0-based optimizer update; trainer 가 step 만의 순수 함수로 읽는다 → resume 동일). stat.outer_weight 는 λE0 그대로(중복 배율 없음)
            k["edge_schedule"] = dict(before=float(B["edge_schedule"]["before"]), after=float(B["edge_schedule"]["after"]), switch=EDGEBAL_SWITCH_UPDATE)
        if B.get("edge_weight"):                                            # §3.4: w_i = high + (low−high)·g_i, g 는 기존 cue 자산의 low_q / shuffle 0/1 표 그대로 (kdv/edge_gate.AffineEdgeWeight)
            ew = B["edge_weight"]; k["edge_weight"] = dict(mode=ew["mode"], low=float(ew["low"]), high=float(ew["high"]), asset=QEDGE9_CUE_ASSET, **({"perm_seed": 51515} if ew["mode"] == "floor_shuffle" else {}))
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
    if (k.get("edge_gate") and k["edge_gate"]["mode"] in ("low_q", "shuffle")) or k.get("edge_route") or k.get("edge_weight") or k.get("qrecon"):   # QEDGE9 §7.1 / QEGX §4.5 / EDGEBAL §3.4 / QRECON24 §2.3: 학습 loader 가 (index, rot) 를 그대로 준다 — RNG 추가 소비 없음 (feeders.feeder return_meta)
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
    if k.get("qrc24"):                                                       # QRECON24 §3·§4.4: U peak LR (L070/L050); A LR 은 kdv.aligner_lr (= rA × U LR). warmup100 + 50K cosine 은 template 그대로
        _lr = "%.1e" % k["qrc24"]["U_lr"]                                    # '7.0e-05' — 소수점 있는 표기 (PyYAML 1.1 은 '7e-05' 를 문자열로 읽는다)
        t = re.sub(r"^learning_rate: .*$", f"learning_rate: {_lr}", t, count=1, flags=re.M); assert re.search(r"^learning_rate: " + re.escape(_lr) + r"$", t, flags=re.M)
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
    if branch == "QRECON24":
        Q = k["qrc24"]
        head = head.replace("\n", f"\n# QRECON24 (계획 {QRC24_PLAN}, 노트 {QRC24_NOTE}): 캠페인 {QRC24_CAMPAIGN_ID} · branch {QRC24_BRANCH} (parent {CAMPAIGN_ID}) · 시간 정책 no_hard_limit(24h 는 최소 운영구간; 자체 ledger {QRC24_LEDGER}, required 경고만) · "
                            + f"profile {Q['profile']}(canonical {Q['canonical']}) · λE 절대 {Q['lambda_E']:g} · rA {Q['rA']:g} (U LR {Q['U_lr']:g} / A LR {Q['A_lr']:g}{'; A 동결' if Q['A_frozen'] else ''}) · α {Q['alpha']:g} β {Q['beta']:g} · A weight {Q['a_weight']} / edge weight {Q['e_weight']} · qref {QRC24_QREF}"
                            + f" · 시트 X열 'PAKD50 / QRC24 / {Q['profile']} / {ARCH_LABEL[arch]}{' / ADJ-R1' if Q.get('queue_revision') else ''} / FRESH50'\n"
                            + (f"# ADJ-R1 (계획 {QRC24_ADJ_PLAN} §4–§6·§11, 노트 {QRC24_ADJ_NOTE}): queue_revision {Q['queue_revision']} · 추가 run(v2 id; 기존 26 run 은 v1 유지) · 2×2(rA .01/.03 × β .1/.2; 같은 서버·seed 실제 id) {Q['block_2x2']} · 대조 {k['control_runs']}"
                               " · 신규 profile: B20A03 = λE .002·rA .03·α 1·β .2 (G23 에서 β 만 / H23 에서 rA 만) · A03_UNIF/A03_SHUF = G23 에서 A hard 가중만 상수 .5 / stratum 셔플(51515), edge 는 w(q)\n" if Q.get("queue_revision") else "")
                            + "# method qrecon_continuous_v1: w = qref/(qref+q_T) (T0 AXIS16 raw q, sg; 계획서의 분자 2 는 09-16 저녁 결정으로 제거 — w(qref)=0.5) · U ← mean(H + K + λE·w^E·E) · A ← mean(w^A·H) 만(soft·edge·offset 없음; 같은 forward 에서 parameter 집합별 autograd.grad, 단일 total backward 금지) · native 입력(I-NATIVE-TRANSFER, jitter 없음)\n"
                            + "# 약명→세팅: G<ij> = λE {6e-4,2e-3,6e-3}[i](계획서 3e-4/1e-3/3e-3 의 2 배 환산) × rA {.003,.01,.03}[j] · H<ij> = α {.5,1,1.5}[i] × β {.05,.1,.2}[j] · L100/070/050 = U LR 1e-4/7e-5/5e-5 (rA .01 고정) · A_UNIF/A_SHUF/A_FREEZE = A 가중 0.5(= s_q(qref))/셔플/동결 · E_UNIF/E_SHUF = edge 가중 0.5/셔플 · ALL_UNIF = 둘 다 0.5 · H_ALPHA0/H_BETA0 = α 0/β 0 · G22 = H22 = L100\n", 1)
    if branch == "EDGEBAL":
        B_ = BACKEND[be_id]; mult = float(B_.get("lam_mult", 1.0)) if B_["edge"] else 0.0
        what = (f" GT edge w(t) = {k['edge_schedule']['before']} (t<{k['edge_schedule']['switch']}) → {k['edge_schedule']['after']} (kdv.edge_schedule; A 계속 학습, optimizer/cosine 재시작 없음)" if k.get("edge_schedule")
                else (f" GT edge w_i = {k['edge_weight']['high']} + ({k['edge_weight']['low']} − {k['edge_weight']['high']})·g_i, g = {'stratum 셔플 라벨(51515)' if k['edge_weight']['mode'] == 'floor_shuffle' else '1[q_T<θq]'} (kdv.edge_weight; cue {QEDGE9_CUE_ASSET})" if k.get("edge_weight")
                      else (f" GT edge 상수 {mult}×λE0 = {k['stat'].get('outer_weight')} (모든 patch)" if B_["edge"] else " GT edge 없음")))
        head = head.replace("\n", f"\n# EDGEBAL (계획 {EDGEBAL_PLAN}, 노트 {EDGEBAL_NOTE}): 캠페인 {EDGEBAL_CAMPAIGN_ID} · branch {EDGEBAL_BRANCH} (parent {CAMPAIGN_ID}, lineage QEDGE9·QEGX) · 시간 정책 no_hard_limit(상한 없음; 자체 ledger {EDGEBAL_LEDGER}, required 경고만, 절대 마감 미상속) ·"
                            + what + f" · rec {k['rec']['case']} · control {k['control_runs']} · 시트 X열 'PAKD50 / {case} / {ARCH_LABEL[arch]} / EDGEBAL / FRESH50'\n"
                            + "# 약명→세팅(EDGEBAL): EB_N0 = J0 · EB_R3E000 = R3 edge 없음(J_R3_NOEDGE) · EB_R3E<rrr> = Q12 인데 GT edge 계수 r×λE0(모든 patch; 100 = JQ) · EB_N0E100 = plain GT + λE0 edge(Teacher 미사용) · EB_R1E100 = XJ · EB_EDOWN/EUP = 25K 에서 1→.5 / .5→1 · EB_QFLOOR/QFSHUF/QFREV = w_i = .25+.5g / 셔플 g / .75−.5g\n", 1)
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
    plan_doc = {srv: QRC24_ADJ_PLAN + f" §5.{i} ADJ-R1 (활성 {len(QRC24_QUEUES[srv])} = 등록 {len(QRC24_REGISTERED[srv])} + 남은 {len(QRC24_ADJ_ORDER[srv])}; 보류 {len(QRC24_HELD.get(srv, {}))}; §10 R_s {QRC24_ADJ_REFERENCE_H[srv]} h{' (A_FREEZE 2.15)' if srv == 's1' else ''}, 예약 1.20×R_s + 10/60, ledger 실측 우선)" for i, srv in enumerate(("s1", "s2", "s3", "s4", "s5"), 1)}.get(server, ALLOC_PLAN)
    slacks = sorted({r.get("slack", RESERVE_SLACK) for r in rows}) or [RESERVE_SLACK]
    print(f"[{server}] seed {SERVER_SEED[server]} — 배정 {plan_doc}; 예약 = {'/'.join(f'{x:g}' for x in slacks)}×ref + {RESERVE_POST_H * 60:.0f}min (QRECON24 1.2, 그 밖 1.1); PAKD50 절대 마감까지 {'?' if rem is None else '%.2f' % rem} h (마감 제외 branch 에는 무관)")
    for r in rows:
        print(f"  {(r['case'] + ('' if r['arch'] == ARCH_DEFAULT else '@' + r['arch'])):<18} S{r['seed']:<5} {(branch_for(server, r['run']) or 'PAKD50'):<7} {r['status']:<14} ref {r['reference_train_h']:.2f} h ({r['reference_kind']}) → 예약 {r['reservation_h']:.4f} h · 누적 {r['cumulative_h']:.4f} h")
    tot = sum(r["reservation_h"] for r in rows if r["status"] == "planned"); tot_x = sum(r["reservation_h"] for r in rows if r["status"] == "planned" and branch_for(server, r["run"]) in ("QEDGE9", "QEGX", "EDGEBAL", "QRECON24"))
    print(f"  planned {sum(r['status'] == 'planned' for r in rows)} run · 예약 합 {tot:.4f} h" + (f" (그중 마감 admission 제외 branch {tot_x:.4f} h)" if tot_x else "")
          + ("" if rem is None else f" · PAKD50 branch 마감 안 {'OK' if tot - tot_x <= max(rem, 0.0) else '초과 — gate admission 이 뒤를 민다'}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=None, choices=list(SERVER_SEED)); ap.add_argument("--all", action="store_true"); ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--cases", default=None, help="쉼표 목록 (기본: stage 별 목록)"); ap.add_argument("--updates", type=int, default=50000); ap.add_argument("--eval-epoch", type=int, default=5)
    ap.add_argument("--projected-hours", type=float, default=None); ap.add_argument("--version", default="v1"); ap.add_argument("--out-dir", default=os.path.join(ROOT, "config")); ap.add_argument("--no-pin", action="store_true", help="τR/λE 를 숫자로 고정하지 않고 그 서버에서 calibrate")
    ap.add_argument("--seed", type=int, default=None, help="서버 seed 대신 (s4 확인 seed 3407 등; 큐 파일은 만들지 않는다 — extra_priority.txt 에 run 이름을 적을 것)")
    ap.add_argument("--plan", action="store_true", help="재배정 순서·예약(1.10×ref+10/60)·누적 dry-run 표만 (config 를 만들지 않는다; 완료·실행 중 run 은 work_dir 로 제외)")
    ap.add_argument("--qrc24-queues", action="store_true", help="config/queues/qrecon24_s{1..5}.txt 를 활성 편성(ADJ-R1) 으로 다시 쓴다 (config 생성 없음)")
    a = ap.parse_args()
    if a.qrc24_queues:
        for s_ in ("s1", "s2", "s3", "s4", "s5"):
            print(s_, len(write_qrc24_queue_file(s_)), "run →", os.path.join("config", "queues", f"qrecon24_{s_}.txt"))
        return
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
