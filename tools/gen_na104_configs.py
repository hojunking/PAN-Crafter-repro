#!/usr/bin/env python
"""W104·D122 no-align KD 캠페인 config 생성 — **FINAL 계획(2026-09-12)** 판.

    python tools/gen_na104_configs.py --all                 # 전 case(v1 88 + 신규 18 + 반복 24) + s2·s3 단계별 큐 + stage plan JSON
    python tools/gen_na104_configs.py --server s2           # s2 block 만
    python tools/gen_na104_configs.py --updates 200 --version dry --only T00,Q00,X05   # dry run (전부 _dry, Teacher/pilot 도 dry 로)

계획: research_log/01_S2_FINAL_EXPERIMENT_PLAN.md · research_log/02_S3_FINAL_EXPERIMENT_PLAN.md (부록 = 공통 규약)
원 계획: research_log/PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md · 노트 research_log/2026-09-11_na104-implementation.md

골격: **W104 · depth [1,2,2] · 9ch → 8ch · aligner 없음(A-ID, sampler 없음) · PAN warp 없음(na_protocol NA-STRICT)**.
trainer 는 기존 kdv (train_kdv.py) 그대로 — 정합 항(G·offset·geometry) 은 resolver 가 A-ID 에서 막는다.

이름·버전 규칙 (FINAL §8, 부록 §6.2)
  - 기존 88 case 의 seed 1234(Teacher 는 2025) → **v1** (이미 돈 run 의 이름을 바꾸지 않는다)
  - 신규 18 정의(X01–X12·PX·CX·LX) 와 **추가 seed(777·2026·legacy 2025/777) → v2**
  - **Teacher 는 항상 `T00 … S2025_v1/best_hqnr`, λ_V pilot 은 항상 `Q00 … S1234_v1/last`** — `--seed`·`--version` 으로 자동 교체되지 않는다
    (`--version dry` 는 dry run 전용 예외: 전부 _dry 로, Teacher/pilot 도 dry 를 가리킨다)
  - 반복 seed 의 비교 baseline 은 그 seed 의 Q00(v2), λ pilot 은 그대로 S1234 v1
선택: 주 selector best_hqnr(raw_original HQNR → fSCC → 늦은 step) — 저장소 확정 지시. best_rr_val·last 는 보조. aligned view/selector 없음.
"""
import argparse, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve, describe, rec_tag, stat_tag, tri_tag                      # noqa: E402

CAMPAIGN = "S2_W104_D122_NOALIGN_KD_20260911"
TEACHER_ID = "T104_v1"
WIDTH, DEPTH = 104, [1, 2, 2]
ARCH = f"W{WIDTH}_D{''.join(map(str, DEPTH))}"
PLAN_S2 = "research_log/01_S2_FINAL_EXPERIMENT_PLAN.md"; PLAN_S3 = "research_log/02_S3_FINAL_EXPERIMENT_PLAN.md"


def rec(case, **kw):
    """REC-N0/R0/R1/R2/R3 (§4.2). alpha_R=1, beta_R=0.1, eps_R=1e-6 은 계획의 시작 제안값."""
    d = dict(case=case, alpha=1.0, kd_weight=0.1, eps=1.0e-6, tau="calibrate", eps_scale=1.0e-6); d.update(kw); return d


def stat(kind, mode, pilot, **kw):
    """STAT-IV/GV/GC/SC/M2/EDGE × H/T/FIX/WH/AD/HAD/WFIX/TMATCH. window 5·population moment·λ_V 는 공통 pilot 의 출력 gradient RMS 비 0.05."""
    d = dict(enabled=True, kind=kind, mode=mode, window=5, alpha=1.0, kd_weight=0.1, tau="calibrate",
             outer_weight="calibrate", lambda_pilot=pilot, r_grad=0.05, ramp_updates=0); d.update(kw); return d


OFF = dict(enabled=False)


def cases(pilot):
    """id → (purpose, rec, stat, tri, extra). seed·version·updates 는 build/main 에서 정한다."""
    C = {}
    C["T00"] = ("정식 Teacher: 같은 W104·D122 no-align 을 plain GT L1 로 학습 (§3.1), seed 2025 — Student 와 seed 를 분리한다", rec("N0"), OFF, None, {})
    # ---- P1 §11.1 핵심 reconstruction (같은 초기값·full-N)
    C["Q00"] = ("독립 Student 기준: KD 없음·통계 없음 (N0). Teacher 숫자 하나로 대신하지 않는다 (§3.3)", rec("N0"), OFF, None, {})
    C["Q01"] = ("일반 고정가중 output KD (R0) — Q00 대비 KD 자체의 이득", rec("R0"), OFF, None, {})
    C["Q02"] = ("Teacher 실패 지도로 GT hard 만 재가중 (R1) — Teacher output 을 target 으로 쓰지 않는다", rec("R1"), OFF, None, {})
    C["Q03"] = ("Teacher 오차로 hard/soft 배분 (R2) — Q02 대비 Teacher output 추가", rec("R2"), OFF, None, {})
    C["Q04"] = ("adaptive: Student 우위 영역의 모방 해제까지 (R3) — Q03 대비 a_T gate, Q02 대비 KD 고유 효과", rec("R3"), OFF, None, {})
    # ---- P2 §11.2 GV·edge·삼각관계
    C["Q05"] = ("Teacher 없이 GT gradient-variance 통계만 (N0+GV-H) — 통계 supervision 과 통계 KD 를 가른다", rec("N0"), stat("GV", "H", pilot), None, {})
    C["Q06"] = ("R3 + GT 구조 통계 (GV-H)", rec("R3"), stat("GV", "H", pilot), None, {})
    C["Q07"] = ("R3 + Teacher 통계 실패 지도로 GT 통계 hard 재가중 (GV-WH)", rec("R3"), stat("GV", "WH", pilot), None, {})
    C["Q08"] = ("R3 + Teacher 통계만 (GV-T, 계수 1) — 픽셀 GT hard 는 유지", rec("R3"), stat("GV", "T", pilot), None, {})
    C["Q09"] = ("R3 + 고정 hard/soft 통계 KD (GV-FIX)", rec("R3"), stat("GV", "FIX", pilot), None, {})
    C["Q10"] = ("R3 + adaptive 통계 KD (GV-AD)", rec("R3"), stat("GV", "AD", pilot), None, {})
    C["Q11"] = ("Teacher 없이 signed Scharr edge L1 (N0+EDGE-H) — variance 고유 효과와 가른다", rec("N0"), stat("EDGE", "H", pilot), None, {})
    C["Q12"] = ("R3 + signed edge (EDGE-H) — Q06 대비 일반 edge 대조", rec("R3"), stat("EDGE", "H", pilot), None, {})
    C["Q13"] = ("R3 soft 에 band 별 GT-방향 gate (TRI-A-SIGN)", rec("R3"), OFF, dict(a=dict(mode="sign")), {})
    C["Q14"] = ("R3 soft 에 pixel cosine 감쇠 (TRI-A-COS)", rec("R3"), OFF, dict(a=dict(mode="cosine")), {})
    C["Q15"] = ("R3 soft 에 sign + 크기비 cap (TRI-A-CAP)", rec("R3"), OFF, dict(a=dict(mode="sign_cap")), {})
    C["Q16"] = ("R3 soft 를 band 별 (1−d_c)a_c 로 세분 (TRI-A-BANDADV)", rec("R3"), OFF, dict(a=dict(mode="band_adv")), {})
    C["Q17"] = ("GV-AD 통계 soft 에 성분별 방향 gate (TRI-B-SIGN)", rec("R3"), stat("GV", "AD", pilot), dict(b=dict(mode="sign")), {})
    C["Q18"] = ("GV-AD 통계 soft 에 sign + 크기비 (TRI-B-CAP)", rec("R3"), stat("GV", "AD", pilot), dict(b=dict(mode="sign_cap")), {})
    C["Q19"] = ("GV-AD 통계 soft 를 성분별 τ_V,j·a_j 로 (TRI-B-COMPADV)", rec("R3"), stat("GV", "AD", pilot), dict(b=dict(mode="comp_adv")), {})
    # ---- P3 §11.3 다른 출력 통계 (IV/GC/SC 의 다섯 mode + GT-only 대조)
    for base, kind, name in ((20, "IV", "국소 밝기 분산"), (25, "GC", "방향 gradient 공동 변화"), (30, "SC", "8밴드 분광 공동 변화")):
        for j, mode in enumerate(("H", "WH", "T", "FIX", "AD")):
            C[f"Q{base + j:02d}"] = (f"R3 + {name} 통계 ({kind}-{mode})", rec("R3"), stat(kind, mode, pilot), None, {})
    C["Q35"] = ("Teacher 없이 IV-H — 학습에도 Teacher 가 필요 없는 GT-only 대조", rec("N0"), stat("IV", "H", pilot), None, {})
    C["Q36"] = ("Teacher 없이 GC-H — GT-only 대조", rec("N0"), stat("GC", "H", pilot), None, {})
    C["Q37"] = ("Teacher 없이 SC-H — GT-only 대조", rec("N0"), stat("SC", "H", pilot), None, {})
    # ---- P3 §11.4 성분·표현의 결합
    C["Q38"] = ("GC-AD 통계 soft 에 성분별 방향 gate (GC-B-SIGN)", rec("R3"), stat("GC", "AD", pilot), dict(b=dict(mode="sign")), {})
    C["Q39"] = ("SC-AD 통계 soft 에 성분별 방향 gate (SC-B-SIGN)", rec("R3"), stat("SC", "AD", pilot), dict(b=dict(mode="sign")), {})
    C["Q40"] = ("A-SIGN(픽셀) + GV-AD(통계) — 2×2 대응의 A 셀", rec("R3"), stat("GV", "AD", pilot), dict(a=dict(mode="sign")), {})
    C["Q41"] = ("A-SIGN + GV-B-SIGN — 2×2 대응의 AB 셀", rec("R3"), stat("GV", "AD", pilot), dict(a=dict(mode="sign"), b=dict(mode="sign")), {})
    C["Q42"] = ("R3 + GV-H + SC-H (두 GT 통계 항의 합, λ 는 각 표현에서 따로)", rec("R3"), stat("GV", "H", pilot, extra=[stat("SC", "H", pilot)]), None, {})
    C["Q43"] = ("R3 + GV-AD + SC-AD (두 adaptive 통계 항의 합)", rec("R3"), stat("GV", "AD", pilot, extra=[stat("SC", "AD", pilot)]), None, {})
    C["Q44"] = ("GC-AD + B-CAP", rec("R3"), stat("GC", "AD", pilot), dict(b=dict(mode="sign_cap")), {})
    C["Q45"] = ("GC-AD + B-COMPADV", rec("R3"), stat("GC", "AD", pilot), dict(b=dict(mode="comp_adv")), {})
    C["Q46"] = ("SC-AD + B-CAP", rec("R3"), stat("SC", "AD", pilot), dict(b=dict(mode="sign_cap")), {})
    C["Q47"] = ("SC-AD + B-COMPADV", rec("R3"), stat("SC", "AD", pilot), dict(b=dict(mode="comp_adv")), {})
    # ---- §11.5 선택적 C 계열: **별도 protocol NA-TSENS** (Teacher probe 에서만 PAN 을 옮긴다; 학습 입력·target·모델은 그대로)
    C["CS00"] = ("Q04 와 같은 학습 + Teacher PAN 민감도 진단만 (감쇠 미적용) — 새 학습 알고리즘이 아니다",
                 rec("R3"), OFF, dict(c=dict(mode="sens", h=0.05, apply=False)), dict(na_protocol="NA-TSENS"))
    C["CS01"] = ("C-NASENS: Teacher 출력의 입력 민감도 r = s/(s+|J|²) 로 rec soft 감쇠", rec("R3"), OFF, dict(c=dict(mode="sens", h=0.05)), dict(na_protocol="NA-TSENS"))
    C["CS02"] = ("A-SIGN × C-NASENS", rec("R3"), OFF, dict(a=dict(mode="sign"), c=dict(mode="sens", h=0.05)), dict(na_protocol="NA-TSENS"))
    C["CS03"] = ("GV-B-SIGN 통계 soft 에 통계-J 기반 감쇠 (pixel J 재사용 금지 — 통계 Jacobian 을 다시 계산)",
                 rec("R3"), stat("GV", "AD", pilot), dict(b=dict(mode="sign"), c=dict(mode="sens", phi="stat", h=0.05)), dict(na_protocol="NA-TSENS"))
    # ---- §12 필수 대조군
    C["CTLHSCALE"] = ("대조: R1 의 공간 w_H 를 batch 평균 스칼라로 (어려운 위치 지도 vs 단순 loss scale)", rec("R1", control="hscale"), OFF, None, {})
    C["CTLRSHUF"] = ("대조: R3 의 d_T·a_T 를 함께 공간 permutation (실패 지도의 위치 정보)", rec("R3", control="rshuffle"), OFF, None, {})
    C["CTLAMASS"] = ("대조: A-SIGN 과 soft 총계수량만 맞춘 스칼라 감쇠 (방향 선택 vs 단순 KD 감소)", rec("R3"), OFF, dict(a=dict(mode="mass")), {})
    C["CTLASHUF"] = ("대조: A mask 의 공간 위치만 섞음 (올바른 위치·부호 정보의 필요성)", rec("R3"), OFF, dict(a=dict(mode="shuffle")), {})
    C["CTLBMASS"] = ("대조: B mask 의 soft 총계수 대응 (GV)", rec("R3"), stat("GV", "AD", pilot), dict(b=dict(mode="mass")), {})
    C["CTLBSHUF"] = ("대조: B mask 위치 섞음 (GV)", rec("R3"), stat("GV", "AD", pilot), dict(b=dict(mode="shuffle")), {})
    C["CTLCMASS"] = ("대조: C-NASENS 와 soft 총계수 대응 (민감도 정보 vs KD 감소)", rec("R3"), OFF, dict(c=dict(mode="sens", h=0.05, control="mass")), dict(na_protocol="NA-TSENS"))
    C["CTLGVSCHALF"] = ("대조: Q43(GV-AD + SC-AD) 과 같은 두 항이되 합의 전체 배율만 ×0.5 (결합 이득이 단순 gradient 증가인지)",
                        rec("R3"), stat("GV", "AD", pilot, lambda_scale=0.5, extra=[stat("SC", "AD", pilot, lambda_scale=0.5)]), None, {})
    C["CTLTAU05"] = ("의존성: τ_R ×0.5", rec("R3", tau_scale=0.5), OFF, None, {})
    C["CTLTAU20"] = ("의존성: τ_R ×2", rec("R3", tau_scale=2.0), OFF, None, {})
    C["CTLBETA03"] = ("의존성: β_R 0.3 (KD 강도, 1 미만 유지)", rec("R3", kd_weight=0.3), OFF, None, {})
    C["CTLBETA05"] = ("의존성: β_R 0.5", rec("R3", kd_weight=0.5), OFF, None, {})
    C["CTLLAMV03"] = ("의존성: λ_V ×0.3 (GV-AD 의 통계 loss scale 민감도 — FIX 계수 탐색이 아니다)", rec("R3"), stat("GV", "AD", pilot, lambda_scale=0.3), None, {})
    C["CTLLAMV30"] = ("의존성: λ_V ×3 (GV-AD)", rec("R3"), stat("GV", "AD", pilot, lambda_scale=3.0), None, {})
    # ---- §11.6 표현 변형 (VX)
    C["VXW3AD"] = ("표현: GV-AD 창 3×3", rec("R3"), stat("GV", "AD", pilot, window=3), None, {})
    C["VXW7AD"] = ("표현: GV-AD 창 7×7", rec("R3"), stat("GV", "AD", pilot, window=7), None, {})
    C["VXW3H"] = ("표현: GV-H 창 3×3 (같은 창의 H/AD 대응)", rec("R3"), stat("GV", "H", pilot, window=3), None, {})
    C["VXW7H"] = ("표현: GV-H 창 7×7", rec("R3"), stat("GV", "H", pilot, window=7), None, {})
    C["VXM357AD"] = ("표현: GV-AD 창 3/5/7 별도 loss 의 평균", rec("R3"), stat("GV", "AD", pilot, windows=[3, 5, 7]), None, {})
    C["VXM357H"] = ("표현: GV-H 창 3/5/7 평균", rec("R3"), stat("GV", "H", pilot, windows=[3, 5, 7]), None, {})
    C["VXSTD"] = ("표현: GV-AD 를 sqrt(v+1e-12) 로 비교 (REP-STD, τ_V·λ_V 재calibration)", rec("R3"), stat("GV", "AD", pilot, transform="std", transform_eps=1.0e-12), None, {})
    C["VXLOG"] = ("표현: GV-AD 를 log(v+1e-9) 로 비교 (REP-LOGVAR)", rec("R3"), stat("GV", "AD", pilot, transform="logvar", transform_eps=1.0e-9), None, {})
    C["VXRES"] = ("표현: GV-AD 를 residual(Ŷ−M) 에서 (REP-RESIDUAL — 출력 variance 와 같지 않다)", rec("R3"), stat("GV", "AD", pilot, domain="residual"), None, {})
    C["VXM2H"] = ("표현: 비중심 gradient 이차 모멘트 GT 대조 (STAT-M2-H) — 같은 reduction 의 GC-H(Q25) 와 비교", rec("R3"), stat("M2", "H", pilot), None, {})
    C["VXM2AD"] = ("표현: STAT-M2-AD — GC-AD(Q29) 와 비교", rec("R3"), stat("M2", "AD", pilot), None, {})
    # ---- §13.3 Teacher-copy 초기화 · §13.2 장기 fitting · 공정성
    C["TCOPYN0"] = ("초기화: Student 를 T00 가중치로 시작한 KD 없는 continuation (fresh optimizer, 50K)", rec("N0"), OFF, None, {"_tcopy": True})
    C["TCOPYR1"] = ("초기화: T00 가중치에서 R1 continuation", rec("R1"), OFF, None, {"_tcopy": True})
    C["TCOPYR3"] = ("초기화: T00 가중치에서 R3 continuation (초기 e_S=e_T → soft 0 에서 시작)", rec("R3"), OFF, None, {"_tcopy": True})
    for cid, case_, st_, what in (("LONG2NN0", rec("N0"), OFF, "N0"), ("LONG2NR1", rec("R1"), OFF, "R1"), ("LONG2NR3", rec("R3"), OFF, "R3"),
                                  ("LONG2NGVAD", rec("R3"), stat("GV", "AD", pilot), "R3+GV-AD")):
        C[cid] = (f"장기 HORIZON: 처음부터 2N(100K) cosine 으로 {what} (N 시점 LR 이 다르므로 N-horizon 의 prefix 가 아니다)", case_, st_, None, {"_long": 2})
    for cid, case_, st_, what in (("CONTN0", rec("N0"), OFF, "supervised"), ("CONTR3", rec("R3"), OFF, "R3"), ("CONTGVAD", rec("R3"), stat("GV", "AD", pilot), "R3+GV-AD")):
        C[cid] = (f"장기 CONTINUATION: 공통 N checkpoint(Q00/last) 에서 fresh optimizer·같은 25K tail 로 {what} 분기", case_, st_, None, {"_cont": True})
    C["COSTMATCH"] = ("공정성: 참조 KD run 의 실측 학습시간 budget 에 맞춘 GT-only 학습 (N 은 측정 뒤 사전 결정)", rec("N0"), OFF, None, {"_cost": True})
    # ==== FINAL 계획(2026-09-12) 부록 §4 — 신규 18 정의 (v2). 기존 case 를 몰래 바꾸지 않는다
    C["X01"] = ("X01 R1 + GT gradient variance(GV-H) — R3 조합의 개선에 output soft 가 필요한가 (Q02·Q06·Q05)", rec("R1"), stat("GV", "H", pilot), None, {})
    C["X02"] = ("X02 R1 + signed GT edge(EDGE-H) (Q02·Q12·Q11)", rec("R1"), stat("EDGE", "H", pilot), None, {})
    C["X03"] = ("X03 N0 + GV-FIX — 일반 GT reconstruction 위에 고정 GT/Teacher 통계 KD (Q05·Q09)", rec("N0"), stat("GV", "FIX", pilot), None, {})
    C["X04"] = ("X04 R1 + GV-FIX — Teacher-error hard reconstruction + 고정 통계 KD (X01·Q09·X03)", rec("R1"), stat("GV", "FIX", pilot), None, {})
    C["X05"] = ("X05 R3 + GV-HAD — 통계 hard 는 plain, soft 만 adaptive: H_V + β_V(1−d_V)a_V K_V (Q06·Q09·Q10)", rec("R3"), stat("GV", "HAD", pilot), None, {})
    C["X06"] = ("X06 R3 + GV-WFIX — 통계 hard 는 weighted, soft 는 fixed: (1+α_V d_V)H_V + β_V K_V (Q07·Q09·Q10)", rec("R3"), stat("GV", "WFIX", pilot), None, {})
    C["X07"] = ("X07 R1 의 d_T 지도만 공간 shuffle (soft 는 0 유지) — CTLRSHUF(R3 의 d·a 공동 shuffle) 와 다른 질문 (Q02·CTLHSCALE)", rec("R1", control="rshuffle"), OFF, None, {})
    C["X08"] = ("X08 R3 + GV-TMATCH — Teacher-only 통계 계수를 FIX 와 같은 β_V=0.1 로 대응 (Q08·Q09)", rec("R3"), stat("GV", "TMATCH", pilot), None, {})
    C["X09"] = ("X09 GC-AD 의 B-MASS 대조 (해당 표현 map 으로 계산; GV 용 CTLBMASS 로 대신하지 않는다) (Q38)", rec("R3"), stat("GC", "AD", pilot), dict(b=dict(mode="mass")), {})
    C["X10"] = ("X10 GC-AD 의 B-SHUF 대조 (Q38)", rec("R3"), stat("GC", "AD", pilot), dict(b=dict(mode="shuffle")), {})
    C["X11"] = ("X11 SC-AD 의 B-MASS 대조 (Q39)", rec("R3"), stat("SC", "AD", pilot), dict(b=dict(mode="mass")), {})
    C["X12"] = ("X12 SC-AD 의 B-SHUF 대조 (Q39)", rec("R3"), stat("SC", "AD", pilot), dict(b=dict(mode="shuffle")), {})
    C["PX01"] = ("PX01 Teacher-copy(T00/best_hqnr 가중치만) 뒤 R3+GV-FIX, fresh optimizer 50K (TCOPYN0·R1·R3 대조)", rec("R3"), stat("GV", "FIX", pilot), None, {"_tcopy": True})
    C["PX02"] = ("PX02 Teacher-copy 뒤 R3+EDGE-H (TCOPYN0)", rec("R3"), stat("EDGE", "H", pilot), None, {"_tcopy": True})
    C["CX01"] = ("CX01 공통 parent(Q00 S1234 v1/last) 에서 R3+GV-FIX, fresh optimizer 25K tail (CONTN0·CONTR3·CONTGVAD)", rec("R3"), stat("GV", "FIX", pilot), None, {"_cont": True})
    C["CX02"] = ("CX02 공통 parent 에서 R3+EDGE-H 25K tail (CONTN0)", rec("R3"), stat("EDGE", "H", pilot), None, {"_cont": True})
    C["LX01"] = ("LX01 처음부터 100K horizon 의 R3+GV-FIX (LONG2NN0·R1·R3·GVAD)", rec("R3"), stat("GV", "FIX", pilot), None, {"_long": 2})
    C["LX02"] = ("LX02 처음부터 100K horizon 의 R3+EDGE-H (LONG2NN0)", rec("R3"), stat("EDGE", "H", pilot), None, {"_long": 2})
    return C


NEW_IDS = {"X01", "X02", "X03", "X04", "X05", "X06", "X07", "X08", "X09", "X10", "X11", "X12", "PX01", "PX02", "CX01", "CX02", "LX01", "LX02"}
CORE10 = ["Q00", "Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q09", "Q11", "Q12"]         # 부록 §6.1 사전 지정 seed 반복 (매 seed 에서 Q00 먼저)
CORE_SEEDS = [777, 2026]                                                              # R1 = 777 (P2 뒤), R2 = 2026 (P4 뒤). 2026 은 Teacher seed(2025) 와 겹치지 않게 고른 값
LEGACY_SEED_CHECK = [("Q00", 2025), ("Q04", 2025), ("Q10", 2025), ("Q10", 777)]        # 원 반복안 보존 (§R3). seed 2025 의 N0 는 Teacher 를 재현할 수 있다 → TIED_TO_TEACHER_SEED

# 직접 대조 (FINAL 계획의 각 표 '직접 대조' 열) — stage plan JSON 에 실어 분석 시 오독을 막는다
CONTRASTS = {
    "Q00": [], "Q01": ["Q00"], "Q02": ["Q00"], "Q03": ["Q02"], "Q04": ["Q03", "Q02"], "Q05": ["Q00"], "Q11": ["Q00", "Q05"], "Q06": ["Q04", "Q05"], "Q07": ["Q06"],
    "Q08": ["Q06", "Q09", "X08"], "Q09": ["Q06", "Q08", "Q10"], "Q10": ["Q07", "Q09"], "Q12": ["Q04", "Q06", "Q11", "Q09"], "Q35": ["Q00"], "Q36": ["Q00"], "Q37": ["Q00"],
    "X01": ["Q02", "Q06", "Q05"], "X02": ["Q02", "Q12", "Q11"], "X03": ["Q05", "Q09"], "X04": ["X01", "Q09", "X03"], "X05": ["Q06", "Q09", "Q10"], "X06": ["Q07", "Q09", "Q10"],
    "X07": ["Q02", "CTLHSCALE"], "X08": ["Q08", "Q09"], "X09": ["Q38"], "X10": ["Q38"], "X11": ["Q39"], "X12": ["Q39"],
    "CTLHSCALE": ["Q02"], "CTLRSHUF": ["Q04"], "CTLBETA03": ["Q04"], "CTLBETA05": ["Q04"], "CTLTAU05": ["Q04"], "CTLTAU20": ["Q04"], "CTLLAMV03": ["Q10"], "CTLLAMV30": ["Q10"],
    "Q13": ["Q04", "CTLAMASS", "CTLASHUF"], "CTLAMASS": ["Q13"], "CTLASHUF": ["Q13"], "Q14": ["Q04", "Q13"], "Q15": ["Q04", "Q13"], "Q16": ["Q04", "Q13"],
    "Q17": ["Q10"], "CTLBMASS": ["Q17"], "CTLBSHUF": ["Q17"], "Q18": ["Q10", "Q17"], "Q19": ["Q10", "Q17"],
    "CS00": ["Q04"], "CS01": ["CS00", "CTLCMASS"], "CTLCMASS": ["CS01"], "CS02": ["Q13", "CS01"], "CS03": ["Q17"],
    "TCOPYN0": [], "TCOPYR1": ["TCOPYN0"], "TCOPYR3": ["TCOPYN0"], "PX01": ["TCOPYN0", "TCOPYR1", "TCOPYR3"], "PX02": ["TCOPYN0", "TCOPYR1", "TCOPYR3"],
    "CONTN0": [], "CONTR3": ["CONTN0"], "CONTGVAD": ["CONTN0"], "CX01": ["CONTN0", "CONTR3", "CONTGVAD"], "CX02": ["CONTN0", "CONTR3", "CONTGVAD"],
    "LONG2NN0": [], "LONG2NR1": ["LONG2NN0"], "LONG2NR3": ["LONG2NN0"], "LONG2NGVAD": ["LONG2NN0"], "LX01": ["LONG2NN0", "LONG2NR1", "LONG2NR3", "LONG2NGVAD"], "LX02": ["LONG2NN0", "LONG2NR1", "LONG2NR3", "LONG2NGVAD"],
    "COSTMATCH": [],
    "Q25": ["Q04", "Q36"], "Q28": ["Q25"], "Q26": ["Q25"], "Q29": ["Q26", "Q28"], "Q27": ["Q25"], "Q20": ["Q04", "Q35"], "Q23": ["Q20"], "Q21": ["Q20"], "Q24": ["Q21", "Q23"], "Q22": ["Q20"],
    "Q30": ["Q04", "Q37"], "Q33": ["Q30"], "Q31": ["Q30"], "Q34": ["Q31", "Q33"], "Q32": ["Q30"],
    "Q38": ["Q29", "X09", "X10"], "Q44": ["Q29", "Q38"], "Q45": ["Q29", "Q38"], "Q39": ["Q34", "X11", "X12"], "Q46": ["Q34", "Q39"], "Q47": ["Q34", "Q39"],
    "Q40": ["Q10"], "Q41": ["Q10", "Q17", "Q40"], "Q42": ["Q06", "Q30"], "Q43": ["Q10", "Q34", "CTLGVSCHALF"], "CTLGVSCHALF": ["Q43"],
    "VXW3H": ["Q06"], "VXW3AD": ["Q10", "VXW3H"], "VXW7H": ["Q06"], "VXW7AD": ["Q10", "VXW7H"], "VXM357H": ["Q06"], "VXM357AD": ["Q10", "VXM357H"],
    "VXM2H": ["Q25"], "VXM2AD": ["Q29"], "VXSTD": ["Q10"], "VXLOG": ["Q10"], "VXRES": ["Q10"],
}

# ==== 서버별 단계 (FINAL 계획 S2-1 / S3-1). 항목은 (case_id, seed). P0_DONE 은 시트 보고 완료분 — 큐 맨 앞에 두면 체인이 완료를 확인하고 건너뛴다
S = 1234; T = 2025
STAGES = {
    "s2": [
        ("P0_DONE", "완료 보고분(재학습 금지·산출물 검증) — 체인은 완료를 확인하고 건너뛴다", [(c, T if c == "T00" else S) for c in ["T00", "Q00", "Q01", "Q02", "Q03", "Q04", "Q05", "Q11", "Q06", "Q07", "Q10", "Q09", "Q08", "Q12"]]),
        ("P1", "상호 서버 후보 보강 — s3 의 약한 긍정 후보(Q36) 를 s2 에서 같은 서버 대조로", [("Q36", S)]),
        ("P2", "구성요소·통계 hard/soft 분리 (X01–X06·X08)", [(c, S) for c in ["X01", "X02", "X03", "X04", "X05", "X06", "X08"]]),
        ("R1", "core10 @ Student seed 777 (Q00 먼저; Teacher·λ pilot 은 그대로)", [(c, 777) for c in CORE10]),
        ("P3", "재가중·계수 대조", [(c, S) for c in ["CTLHSCALE", "X07", "CTLRSHUF", "CTLBETA03", "CTLBETA05", "CTLTAU05", "CTLTAU20", "CTLLAMV03", "CTLLAMV30"]]),
        ("P4", "픽셀 방향·통계 성분 선택과 MASS/SHUF 대조", [(c, S) for c in ["Q13", "CTLAMASS", "CTLASHUF", "Q14", "Q15", "Q16", "Q17", "CTLBMASS", "CTLBSHUF", "Q18", "Q19"]]),
        ("R2", "core10 @ Student seed 2026", [(c, 2026) for c in CORE10]),
        ("P5", "입력 민감도 C 분기 (NA-TSENS): 진단만 → 감쇠 → 총량 대조 → A/B 결합", [(c, S) for c in ["CS00", "CS01", "CTLCMASS", "CS02", "CS03"]]),
        ("P6", "Teacher-copy·공통 parent continuation (+ 유력 GV-FIX/EDGE 의 PX/CX)", [(c, S) for c in ["TCOPYN0", "TCOPYR1", "TCOPYR3", "PX01", "PX02", "CONTN0", "CONTR3", "CONTGVAD", "CX01", "CX02"]]),
        ("P7", "100K horizon (+ LX) · 시간 대응 COSTMATCH(측정 뒤)", [(c, S) for c in ["LONG2NN0", "LONG2NR1", "LONG2NR3", "LONG2NGVAD", "LX01", "LX02", "COSTMATCH"]]),
        ("LEGACY_SEED_CHECK", "원 반복안 보존 — seed 2025 의 N0 가 Teacher 를 재현하면 TIED_TO_TEACHER_SEED 로 표시", list(LEGACY_SEED_CHECK)),
    ],
    "s3": [
        ("P0_DONE", "완료 보고분(재학습 금지·산출물 검증) — 체인은 완료를 확인하고 건너뛴다", [(c, T if c == "T00" else S) for c in ["T00", "Q00", "Q04", "Q06", "Q10", "Q01", "Q02", "Q03", "Q35", "Q36", "Q37", "Q07", "Q05", "Q11"]]),
        ("P1", "s2 후보를 s3 에서 완성 — GV 모드 사다리의 빈칸", [(c, S) for c in ["Q09", "Q12", "Q08"]]),
        ("P2", "구성요소·통계 hard/soft 분리 (X01–X06·X08) + R1 의 hard 대조(CTLHSCALE·X07)", [(c, S) for c in ["X01", "X02", "X03", "X04", "X05", "X06", "X08", "CTLHSCALE", "X07"]]),
        ("R1", "core10 @ Student seed 777", [(c, 777) for c in CORE10]),
        ("P3", "GC·IV·SC 전 모드 (GC 먼저)", [(c, S) for c in ["Q25", "Q28", "Q26", "Q29", "Q27", "Q20", "Q23", "Q21", "Q24", "Q22", "Q30", "Q33", "Q31", "Q34", "Q32"]]),
        ("P4", "방향·성분·복합 통계: A/B 기준, GC/SC 의 표현별 MASS/SHUF(X09–X12), 2×2, GV+SC", [(c, S) for c in ["Q13", "CTLAMASS", "CTLASHUF", "Q17", "CTLBMASS", "CTLBSHUF", "Q38", "X09", "X10", "Q44", "Q45", "Q39", "X11", "X12", "Q46", "Q47", "Q40", "Q41", "Q42", "Q43", "CTLGVSCHALF"]]),
        ("R2", "core10 @ Student seed 2026", [(c, 2026) for c in CORE10]),
        ("P5", "보류 표현 변형 전부 (창 3/7/357 · M2 · STD · LOG · residual)", [(c, S) for c in ["VXW3H", "VXW3AD", "VXW7H", "VXW7AD", "VXM357H", "VXM357AD", "VXM2H", "VXM2AD", "VXSTD", "VXLOG", "VXRES"]]),
        ("LEGACY_SEED_CHECK", "원 반복안 보존 — seed 2025 의 N0 가 Teacher 를 재현하면 TIED_TO_TEACHER_SEED 로 표시", list(LEGACY_SEED_CHECK)),
    ],
}


def build(cid, spec_tuple, seed, teacher_run, pilot, baseline_run, version):
    purpose, rc, st, tri, extra = spec_tuple
    k = dict(campaign_id=CAMPAIGN, run_kind="CONTROLLED", version=version, check_run_name=False,
             recipe="NOALIGN", aligner_policy="A-ID", input_protocol="I-A", na_protocol=extra.get("na_protocol", "NA-STRICT"),
             diag_every=1000, calibration=dict(n_patches=3072, seed=1234),
             # 판정·시트·Teacher·진단이 같은 checkpoint 를 쓴다 — 저장소 확정 지시(무조건 HQNR→SCC). 계획 §14.2 의 독립 RR-validation 선택은 보조(best_rr_val)
             select=dict(primary="best_hqnr", secondary=["best_rr_val", "last"], aligned_selector=False),
             expect_arch=dict(width=WIDTH, depth=list(DEPTH), noalign=True),      # 손으로 고친 config 가 다른 폭으로 도는 것을 trainer 가 막는다
             rec=dict(rc), stat=dict(st), geom_kd=dict(mode="G0"))
    if tri:
        k["tri"] = dict({kk: dict(v) for kk, v in tri.items()}, shuffle_seed=4321)
    sp0 = resolve(dict(k, teacher=dict(id=TEACHER_ID, run=f"work_dir/{teacher_run}", tag="best_hqnr")))     # Teacher 가 학습 loss 에 필요한지 판정용
    if cid != "T00":
        # GT-only arm(N0·통계 H) 도 **평가 전용**으로 같은 고정 Teacher 를 싣는다 — 모든 Student 를 같은 Teacher 오차 bin 에서 비교 (§15.2). 학습 loss 에는 안 쓴다
        k["teacher"] = dict(id=TEACHER_ID, run=f"work_dir/{teacher_run}", tag="best_hqnr", expected_sha256=None, bridge=False,
                            **({} if sp0["needs_teacher"] else dict(eval_only=True)))
        k["baseline_run"] = baseline_run                     # 비교 baseline = 같은 Student seed 의 Q00 (λ pilot 과는 별개 identity, 부록 §6.2)
    if extra.get("_tcopy"):                              # Teacher-copy: 가중치만 T00/best_hqnr 에서, optimizer·schedule 은 새로 (schedule step 0)
        k["run_kind"] = "ADAPTIVE_PATH"
        k["phase"] = dict(parent_run=teacher_run, parent_tag="best_hqnr", parent_step=0, optimizer_state_policy="fresh",
                          reason="INIT_TCOPY (§13.3): 같은 가중치에서 KD 없음/R1/R3(+PX) 를 비교한다", matched_continuation="TCOPYN0")
    if extra.get("_cont"):                               # CONTINUATION: 공통 parent Q00 S1234 v1/last 에서 fresh optimizer·같은 tail
        k["run_kind"] = "ADAPTIVE_PATH"
        k["phase"] = dict(parent_run=pilot.rsplit("/", 1)[0], parent_tag="last", parent_step=0, optimizer_state_policy="fresh",
                          reason="LONG CONTINUATION (§13.2): 공통 N checkpoint 에서 tail 을 모두에게 같게 (restart/tail — 낮은 LR 미세조정이 아니다)", matched_continuation="CONTN0")
    return k, purpose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None, choices=["s2", "s3"], help="서버 block")
    ap.add_argument("--all", action="store_true", help="전 case 생성 + s2·s3 큐 둘 다")
    ap.add_argument("--only", default=None, help="쉼표로 구분한 case id (예: T00,Q00,X05). 없는 id 는 오류로 알린다")
    ap.add_argument("--seed", type=int, default=1234, help="기본 Student seed (v1 block)"); ap.add_argument("--teacher-seed", type=int, default=2025)
    ap.add_argument("--no-core-repeat", action="store_true", help="core10 × seed 777/2026 반복을 만들지 않는다")
    ap.add_argument("--no-legacy-seed", action="store_true", help="원 반복안(Q00/Q04/Q10@2025, Q10@777) 을 만들지 않는다")
    ap.add_argument("--updates", type=int, default=50000, help="N (계획 §13.1: 50,000)")
    ap.add_argument("--tail-updates", type=int, default=25000, help="CONTINUATION 의 tail update 수 (양쪽에 같게)")
    ap.add_argument("--cost-match-updates", type=int, default=None, help="COSTMATCH 의 N (참조 KD 의 실측 시간에서 사전 결정; 없으면 BLOCKED_COST_MEASUREMENT 로 두고 만들지 않는다)")
    ap.add_argument("--eval-epoch", type=int, default=10, help="평가 주기(epoch). 2026-09-11 에 5->10 (평가가 wall-clock 의 53%). 새 run 은 10 을 쓴다 (부록 §5.2)")
    ap.add_argument("--version", default="v1", help="v1 이면 규칙(기존 seed1234 = v1, 신규/추가 seed = v2). 다른 값(예: dry) 은 전부 그 값으로 강제")
    ap.add_argument("--template", default="config/PA_A1_REC_W96_D124_9CH_S1234.yaml"); ap.add_argument("--params", type=float, default=None)
    a = ap.parse_args()
    import yaml
    if a.params is None:
        from main import import_class
        _c = yaml.safe_load(open(os.path.join(ROOT, a.template)))
        _m = import_class(_c["model"])(**dict(_c["model_args"], hidden_size=WIDTH, depth=DEPTH))
        a.params = round(sum(p.numel() for p in _m.parameters()) / 1e6, 4)
    force = None if a.version == "v1" else a.version

    def ver_of(cid, seed):
        if force:
            return force
        if cid in NEW_IDS:
            return "v2"
        if cid == "T00":
            return "v1"
        return "v1" if seed == a.seed else "v2"

    def name(cid, sp, seed):
        tt = tri_tag(sp)
        return f"NA104_{cid}_{ARCH}_WV3_{rec_tag(sp)}_{stat_tag(sp)}{('_' + tt) if tt else ''}_S{seed}_{ver_of(cid, seed)}"

    # 1) 고정 identity: Teacher = T00 S2025 v1 · λ pilot = Q00 S1234 v1/last (부록 §6.2 — --seed/--version 이 바꾸지 않는다)
    tmp = cases("TBD/last")
    teacher_run = name("T00", resolve(build("T00", tmp["T00"], a.teacher_seed, "TBD", "TBD/last", "TBD", ver_of("T00", a.teacher_seed))[0]), a.teacher_seed)
    q0_sp = resolve(build("Q00", tmp["Q00"], a.seed, teacher_run, "TBD/last", "TBD", ver_of("Q00", a.seed))[0])
    pilot = f"{name('Q00', q0_sp, a.seed)}/last"
    C = cases(pilot)

    def baseline_of(seed):
        return name("Q00", resolve(build("Q00", C["Q00"], seed, teacher_run, pilot, "TBD", ver_of("Q00", seed))[0]), seed)

    # 2) 편성: (stage, cid, seed) 셀
    cells = []
    if a.only:
        want = [x.strip() for x in a.only.split(",") if x.strip()]
        missing = [x for x in want if x not in C]
        if missing:
            sys.exit(f"!! 없는 case id: {missing} — 조용히 빠뜨리지 않는다 (부록 §8)")
        cells = [("ONLY", cid, a.teacher_seed if cid == "T00" else a.seed) for cid in want]
    else:
        srvs = ["s2", "s3"] if a.all else ([a.server] if a.server else ["s2", "s3"])
        seen = set()
        for srv in srvs:
            for stage, _desc, items in STAGES[srv]:
                for cid, seed in items:
                    if (stage.startswith("R") and a.no_core_repeat) or (stage == "LEGACY_SEED_CHECK" and a.no_legacy_seed):
                        continue
                    if (cid, seed) in seen:
                        continue
                    seen.add((cid, seed)); cells.append((stage, cid, seed))
    blocked_cost = a.cost_match_updates is None
    if blocked_cost:
        cells = [c for c in cells if c[1] != "COSTMATCH"]

    tpl = re.sub(r"^(#.*\n)+", "", open(os.path.join(ROOT, a.template)).read())
    made, rows = [], []
    for stage, cid, seed in cells:
        k, purpose = build(cid, C[cid], seed, teacher_run, pilot, ("" if cid == "T00" else baseline_of(seed)), ver_of(cid, seed))
        sp = resolve(k); tag = name(cid, sp, seed)
        upd = a.updates * C[cid][4].get("_long", 1)
        if C[cid][4].get("_cont"):
            upd = a.tail_updates
        if C[cid][4].get("_cost"):
            upd = a.cost_match_updates or a.updates
        head = (f"# {tag} — [{cid}] {purpose}. 생성: tools/gen_na104_configs.py. 손으로 고치지 말 것.\n"
                f"# 계획: {PLAN_S2} · {PLAN_S3} (FINAL 2026-09-12) · 원 계획 research_log/PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md · 구현 train_kdv.py + kdv/ · 노트 research_log/2026-09-11_na104-implementation.md\n"
                f"# 세팅: {describe(sp)}\n"
                f"# 골격 {ARCH}(hidden {WIDTH}, depth {DEPTH}, {a.params} M) · 9ch→8ch · **aligner·sampler 없음, PAN warp 없음** · MS base 1회 합산 · AdamW 1e-4/wd 0.01 cosine warmup100 · batch 48 · {upd} updates · seed {seed} · version {ver_of(cid, seed)}\n"
                f"# 초기값 work_dir/_kdv_init_w104_d122 (같은 seed 의 Student 가 공유) · Teacher = {teacher_run}/best_hqnr (id {TEACHER_ID}, 고정) · λ_V pilot = {pilot} (고정) · baseline = {k.get('baseline_run', '—')}\n"
                f"# 선택: 주 selector best_hqnr(raw_original HQNR→fSCC→늦은 step; 저장소 확정 지시) · 보조 best_rr_val·last. Teacher·시트·진단 모두 best_hqnr. aligned view/selector 없음. eval_epoch {a.eval_epoch}\n")
        t = re.sub(r"^eval_epoch: \d+$", f"eval_epoch: {a.eval_epoch}", tpl, flags=re.M)
        t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", t)
        t = re.sub(r"^trainer: pa\npa:\n(  .*\n)+", "", t, flags=re.M)
        t = t.replace("mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)",
                      "mars: ms                      # PAN mode·loss·batch 복제 제거 (단일 task)\ntrainer: kdv\nkdv:\n"
                      + "\n".join("  " + l for l in yaml.safe_dump(k, sort_keys=False, allow_unicode=True, default_flow_style=False).splitlines()) + "\n")
        t = re.sub(r"^num_iter: \d+", f"num_iter: {upd}", t, flags=re.M)
        t = re.sub(r"^  hidden_size: \d+", f"  hidden_size: {WIDTH}", t, flags=re.M)
        t = re.sub(r"^  depth: \[.*?\]", f"  depth: {DEPTH}", t, flags=re.M)
        t = re.sub(r"expect_params_m: [\d.]+", f"expect_params_m: {a.params}", t)
        t = re.sub(r"^seed: \d+", f"seed: {seed}", t, flags=re.M)
        assert f"hidden_size: {WIDTH}" in t and f"depth: {DEPTH}" in t and "trainer: kdv" in t and "trainer: pa" not in t and f"num_iter: {upd}" in t and f"seed: {seed}" in t
        open(os.path.join(ROOT, "config", tag + ".yaml"), "w").write(head + t)
        made.append(tag)
        ph = k.get("phase") or {}
        rows.append(dict(stage=stage, case_id=cid, run_id=tag, seed=seed, version=ver_of(cid, seed), updates=upd,
                         rec=rec_tag(sp), stat=stat_tag(sp), tri=(tri_tag(sp) or "—"), protocol=sp["na_protocol"],
                         teacher=("none" if not sp["has_teacher"] else ("eval_only" if sp["teacher_eval_only"] else "training")),
                         teacher_run=(teacher_run if cid != "T00" else None), lambda_pilot=(pilot if sp["stat_enabled"] else None), baseline_run=k.get("baseline_run"),
                         parent=(f"{ph['parent_run']}/{ph['parent_tag']}" if ph else None), parent_schedule_step=(ph.get("parent_step") if ph else None),
                         contrasts=CONTRASTS.get(cid, []), new_definition=(cid in NEW_IDS), setting=describe(sp)))

    # 3) 큐 + stage plan (execute:false, 계획 JSON) — 서버별
    by_key = {(r["case_id"], r["seed"]): r for r in rows}
    for srv in (["s2", "s3"] if a.all else ([a.server] if a.server else ["s2", "s3"])):
        if a.only:
            break
        qp = os.path.join(ROOT, "config", "queues", f"na104_{srv}{'' if not force else '_' + force}.txt")
        plan = dict(server=srv, campaign=CAMPAIGN, plan=(PLAN_S2 if srv == "s2" else PLAN_S3), execute=False, generated_by="tools/gen_na104_configs.py",
                    teacher_run=teacher_run, lambda_pilot=pilot, primary_selector="best_hqnr", secondary=["best_rr_val", "last"], eval_epoch=a.eval_epoch,
                    version_rule="기존 case seed1234 → v1 · 신규 18 정의/추가 seed → v2 · Teacher/λ pilot 고정", stages=[])
        qlines = []
        for stage, desc, items in STAGES[srv]:
            runs = []
            for cid, seed in items:
                r = by_key.get((cid, seed))
                if r is None:
                    if cid == "COSTMATCH" and blocked_cost:
                        runs.append(dict(case_id=cid, seed=seed, status="BLOCKED_COST_MEASUREMENT", note="참조 KD 의 실측 시간 budget 으로 N 을 정한 뒤 --cost-match-updates 로 생성한다. 50K 로 채우지 않는다"))
                    continue
                runs.append(dict(r, status=("SHEET_REPORTED_50K → local 검증 후 재사용 (체인이 완료를 확인하고 건너뜀)" if stage == "P0_DONE" else "PENDING")))
                qlines.append(r["run_id"])
            plan["stages"].append(dict(stage=stage, description=desc, n=len([x for x in runs if x.get("run_id")]), runs=runs))
        n_new = sum(len([x for x in st_["runs"] if x.get("run_id")]) for st_ in plan["stages"] if st_["stage"] != "P0_DONE")
        plan["slots"] = dict(sheet_reported=len(plan["stages"][0]["runs"]), new_or_verify=n_new, total=len(plan["stages"][0]["runs"]) + n_new,
                             blocked=[x["case_id"] for st_ in plan["stages"] for x in st_["runs"] if not x.get("run_id")])
        json.dump(plan, open(os.path.join(ROOT, "config", "queues", f"na104_{srv}_stage_plan.json"), "w"), indent=1, ensure_ascii=False)
        with open(qp, "w") as f:
            f.write(f"# NA104 {srv} **FINAL 단계별 큐** — {PLAN_S2 if srv == 's2' else PLAN_S3} (2026-09-12)\n"
                    f"# 주 지표는 원본 FR 논문 세트 20장의 HQNR (raw_original, 장면별 계산 후 평균). ERGAS·학습 L1 은 보조 진단이다.\n"
                    f"# Teacher {teacher_run}/best_hqnr · λ pilot {pilot} (둘 다 고정 — seed/version 으로 바뀌지 않는다)\n"
                    f"# 단계: {' → '.join(st_['stage'] for st_ in plan['stages'])} — 구조·직접 대조는 na104_{srv}_stage_plan.json\n"
                    f"# 맨 앞 P0_DONE 은 완료 보고분 — 체인이 reduced/full_best_hqnr.mat 을 보고 건너뛴다(재학습 금지). 정상 진행 중 run 은 원 설정대로 완료한다.\n"
                    f"# 돌고 있는 체인은 work_dir/cases_queue.txt(복사본) 를 보므로, 이 순서는 run 경계에서 campaign_start.sh 로 적용한다.\n"
                    + "\n".join(qlines) + "\n")
        print(f"queue {srv}: {os.path.relpath(qp, ROOT)} ({len(qlines)} run; 완료분 {plan['slots']['sheet_reported']} + 신규/확인 {plan['slots']['new_or_verify']} = {plan['slots']['total']}"
              + (f"; BLOCKED {plan['slots']['blocked']}" if plan['slots']['blocked'] else "") + ")")
    n_new_ids = sum(1 for r in rows if r["new_definition"])
    print(f"config {len(made)}벌 (신규 정의 {n_new_ids}, v2 {sum(1 for r in rows if r['version'] == 'v2')}) · params {a.params} M · Teacher {teacher_run} · pilot {pilot}")
    if blocked_cost:
        print("  COSTMATCH: BLOCKED_COST_MEASUREMENT — 참조 KD 의 실측 시간으로 N 을 정한 뒤 --cost-match-updates <N> 로 생성 (부록 §6.4)")
    if not a.no_legacy_seed and any(s == a.teacher_seed and c != "T00" for _, c, s in cells):
        print(f"  ! seed {a.teacher_seed} 는 Teacher seed — 그 seed 의 N0 는 초기값·데이터 순서가 T00 과 같아 Teacher 를 재현할 수 있다 → TIED_TO_TEACHER_SEED 로 표시하고 독립 T/S 증거로 세지 않는다")


if __name__ == "__main__":
    main()
