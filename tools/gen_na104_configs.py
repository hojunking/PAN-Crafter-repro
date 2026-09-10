#!/usr/bin/env python
"""W104·D122 no-align KD 캠페인 config 생성 (research_log/PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md).

    python tools/gen_na104_configs.py --server s2      # config/NA104_*.yaml + config/queues/na104_s2.txt
    python tools/gen_na104_configs.py --server s3      # s3 block (P3 표현 확장 + 같은 서버의 대조 anchor)
    python tools/gen_na104_configs.py --all            # 전 case + 두 큐
    python tools/gen_na104_configs.py --updates 300 --version dry --only T00,Q00,Q04,Q10   # dry run 용

골격: **W104 · depth [1,2,2] · 9ch → 8ch · aligner 없음(A-ID, sampler 없음) · PAN warp 없음(na_protocol NA-STRICT)**.
trainer 는 기존 kdv (train_kdv.py) 를 그대로 쓴다 — 정합 항(G·offset·geometry) 은 resolver 가 A-ID 에서 막는다.
Teacher T00 은 같은 골격의 plain GT L1(seed 2025), Student 는 저장된 같은 초기값(seed 1234, work_dir/_kdv_init_w104_d122) 을 공유한다.
선택: **주 selector = best_rr_val(valid_wv3.h5 plain ERGAS)** + 고정 final-N(last); best_hqnr(=best_raw) 는 FR test 로 고른 exploratory 로만 남긴다 (계획 §14.2).
aligned view/selector 는 만들지 않는다 — aligner 가 없으면 같은 ROI 에서 raw_valid 와 같다 (§14.1).
약명(T00/Q00/CTL…) 은 노트 research_log/2026-09-11_na104-implementation.md 의 표로만 읽는다 — 실행명에 rec·stat·tri 설정 토큰이 함께 들어간다.
"""
import argparse, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve, describe, rec_tag, stat_tag, tri_tag                      # noqa: E402

CAMPAIGN = "S2_W104_D122_NOALIGN_KD_20260911"
TEACHER_ID = "T104_v1"
WIDTH, DEPTH = 104, [1, 2, 2]
ARCH = f"W{WIDTH}_D{''.join(map(str, DEPTH))}"


def rec(case, **kw):
    """REC-N0/R0/R1/R2/R3 (§4.2). alpha_R=1, beta_R=0.1, eps_R=1e-6 은 계획의 시작 제안값."""
    d = dict(case=case, alpha=1.0, kd_weight=0.1, eps=1.0e-6, tau="calibrate", eps_scale=1.0e-6); d.update(kw); return d


def stat(kind, mode, pilot, **kw):
    """STAT-IV/GV/GC/SC/M2/EDGE × H/T/FIX/WH/AD (§5–§6). window 5·population moment·λ_V 는 공통 pilot 의 출력 gradient RMS 비 0.05."""
    d = dict(enabled=True, kind=kind, mode=mode, window=5, alpha=1.0, kd_weight=0.1, tau="calibrate",
             outer_weight="calibrate", lambda_pilot=pilot, r_grad=0.05, ramp_updates=0); d.update(kw); return d


OFF = dict(enabled=False)


def cases(pilot):
    """id → (purpose, rec, stat, tri, extra kdv keys). seed·updates 는 build 에서 결정한다."""
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
    C["Q08"] = ("R3 + Teacher 통계만 (GV-T) — 픽셀 GT hard 는 유지", rec("R3"), stat("GV", "T", pilot), None, {})
    C["Q09"] = ("R3 + 고정 hard/soft 통계 KD (GV-FIX)", rec("R3"), stat("GV", "FIX", pilot), None, {})
    C["Q10"] = ("R3 + adaptive 통계 KD (GV-AD) — 통계 축의 주력 후보", rec("R3"), stat("GV", "AD", pilot), None, {})
    C["Q11"] = ("Teacher 없이 signed Scharr edge L1 (N0+EDGE-H) — variance 고유 효과와 가른다", rec("N0"), stat("EDGE", "H", pilot), None, {})
    C["Q12"] = ("R3 + signed edge (EDGE-H) — Q06 대비 일반 edge 대조", rec("R3"), stat("EDGE", "H", pilot), None, {})
    C["Q13"] = ("R3 soft 에 band 별 GT-방향 gate (TRI-A-SIGN) — GT 와 충돌하는 band 의 모방 해제", rec("R3"), OFF, dict(a=dict(mode="sign")), {})
    C["Q14"] = ("R3 soft 에 pixel cosine 감쇠 (TRI-A-COS) — scalar 방향 대조", rec("R3"), OFF, dict(a=dict(mode="cosine")), {})
    C["Q15"] = ("R3 soft 에 sign + 크기비 cap (TRI-A-CAP)", rec("R3"), OFF, dict(a=dict(mode="sign_cap")), {})
    C["Q16"] = ("R3 soft 를 band 별 (1−d_c)a_c 로 세분 (TRI-A-BANDADV) — 평균 gate 가 놓친 좋은 band", rec("R3"), OFF, dict(a=dict(mode="band_adv")), {})
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
    C["CTLBMASS"] = ("대조: B mask 의 soft 총계수 대응", rec("R3"), stat("GV", "AD", pilot), dict(b=dict(mode="mass")), {})
    C["CTLBSHUF"] = ("대조: B mask 위치 섞음", rec("R3"), stat("GV", "AD", pilot), dict(b=dict(mode="shuffle")), {})
    C["CTLCMASS"] = ("대조: C-NASENS 와 soft 총계수 대응 (민감도 정보 vs KD 감소)", rec("R3"), OFF, dict(c=dict(mode="sens", h=0.05, control="mass")), dict(na_protocol="NA-TSENS"))
    C["CTLGVSCHALF"] = ("대조: Q43(GV-AD + SC-AD) 과 같은 두 항이되 **합의 전체 배율만** ×0.5 — 결합의 이득이 단순 gradient 증가인지 가른다",
                        rec("R3"), stat("GV", "AD", pilot, lambda_scale=0.5, extra=[stat("SC", "AD", pilot, lambda_scale=0.5)]), None, {})
    C["CTLTAU05"] = ("의존성: τ_R ×0.5", rec("R3", tau_scale=0.5), OFF, None, {})
    C["CTLTAU20"] = ("의존성: τ_R ×2", rec("R3", tau_scale=2.0), OFF, None, {})
    C["CTLBETA03"] = ("의존성: β_R 0.3 (KD 강도, 1 미만 유지)", rec("R3", kd_weight=0.3), OFF, None, {})
    C["CTLBETA05"] = ("의존성: β_R 0.5", rec("R3", kd_weight=0.5), OFF, None, {})
    C["CTLLAMV03"] = ("의존성: λ_V ×0.3 (통계 loss scale)", rec("R3"), stat("GV", "AD", pilot, lambda_scale=0.3), None, {})
    C["CTLLAMV30"] = ("의존성: λ_V ×3", rec("R3"), stat("GV", "AD", pilot, lambda_scale=3.0), None, {})
    # ---- §11.6 표현 변형 (VX)
    C["VXW3AD"] = ("표현: GV-AD 창 3×3", rec("R3"), stat("GV", "AD", pilot, window=3), None, {})
    C["VXW7AD"] = ("표현: GV-AD 창 7×7", rec("R3"), stat("GV", "AD", pilot, window=7), None, {})
    C["VXW3H"] = ("표현: GV-H 창 3×3 (같은 창의 H/AD 대응)", rec("R3"), stat("GV", "H", pilot, window=3), None, {})
    C["VXW7H"] = ("표현: GV-H 창 7×7", rec("R3"), stat("GV", "H", pilot, window=7), None, {})
    C["VXM357AD"] = ("표현: GV-AD 창 3/5/7 별도 loss 의 평균", rec("R3"), stat("GV", "AD", pilot, windows=[3, 5, 7]), None, {})
    C["VXM357H"] = ("표현: GV-H 창 3/5/7 평균", rec("R3"), stat("GV", "H", pilot, windows=[3, 5, 7]), None, {})
    C["VXSTD"] = ("표현: GV-AD 를 sqrt(v+eps) 로 비교 (REP-STD, τ_V·λ_V 재calibration)", rec("R3"), stat("GV", "AD", pilot, transform="std", transform_eps=1.0e-12), None, {})
    C["VXLOG"] = ("표현: GV-AD 를 log(v+eps) 로 비교 (REP-LOGVAR)", rec("R3"), stat("GV", "AD", pilot, transform="logvar", transform_eps=1.0e-9), None, {})
    C["VXRES"] = ("표현: GV-AD 를 residual(Ŷ−M) 에서 (REP-RESIDUAL — 출력 variance 와 같지 않다)", rec("R3"), stat("GV", "AD", pilot, domain="residual"), None, {})
    C["VXM2H"] = ("표현: 비중심 gradient 이차 모멘트 GT 대조 (STAT-M2-H) — 같은 reduction 의 GC-H 와 비교", rec("R3"), stat("M2", "H", pilot), None, {})
    C["VXM2AD"] = ("표현: STAT-M2-AD — GC-AD 와 비교", rec("R3"), stat("M2", "AD", pilot), None, {})
    # ---- §13.3 Teacher-copy 초기화 (세 갈래를 같은 가중치에서)
    C["TCOPYN0"] = ("초기화: Student 를 T00 가중치로 시작한 KD 없는 continuation", rec("N0"), OFF, None, {"_tcopy": True})
    C["TCOPYR1"] = ("초기화: T00 가중치에서 R1 continuation", rec("R1"), OFF, None, {"_tcopy": True})
    C["TCOPYR3"] = ("초기화: T00 가중치에서 R3 continuation (초기 e_S=e_T → soft 0 에서 시작)", rec("R3"), OFF, None, {"_tcopy": True})
    # ---- §13.2 장기 fitting: HORIZON(처음부터 2N) 과 CONTINUATION(공통 N checkpoint 에서 같은 tail)
    for cid, case_, st_, what in (("LONG2NN0", rec("N0"), OFF, "N0"), ("LONG2NR1", rec("R1"), OFF, "R1"), ("LONG2NR3", rec("R3"), OFF, "R3"),
                                  ("LONG2NGVAD", rec("R3"), stat("GV", "AD", pilot), "R3+GV-AD")):
        C[cid] = (f"장기 HORIZON: 처음부터 2N schedule 로 {what} (N 시점 LR 이 다르므로 N-horizon 의 prefix 가 아니다)", case_, st_, None, {"_long": 2})
    for cid, case_, st_, what in (("CONTN0", rec("N0"), OFF, "supervised"), ("CONTR3", rec("R3"), OFF, "R3"), ("CONTGVAD", rec("R3"), stat("GV", "AD", pilot), "R3+GV-AD")):
        C[cid] = (f"장기 CONTINUATION: 공통 N checkpoint(Q00/last) 에서 같은 tail schedule 로 {what} 분기", case_, st_, None, {"_cont": True})
    C["COSTMATCH"] = ("공정성: KD run 의 실제 학습시간만큼 N0 를 더 학습 (update 대응과 별도의 시간 대응)", rec("N0"), OFF, None, {"_cost": True})
    return C


# 서버 block (§13.4: 대응 비교는 같은 서버 안에서 끝난다). s3 는 P3 표현 확장을 맡되 그 비교에 필요한 anchor(Q00/Q04/Q10) 를 같이 돈다.
S2_IDS = (["T00", "Q00"] + [f"Q{i:02d}" for i in range(1, 20)]
          + ["CTLHSCALE", "CTLRSHUF", "CTLAMASS", "CTLASHUF", "CTLBMASS", "CTLBSHUF", "CTLTAU05", "CTLTAU20", "CTLBETA03", "CTLBETA05", "CTLLAMV03", "CTLLAMV30"]
          + ["CS00", "CS01", "CS02", "CS03", "CTLCMASS"]
          + ["TCOPYN0", "TCOPYR1", "TCOPYR3", "CONTN0", "CONTR3", "CONTGVAD", "LONG2NN0", "LONG2NR1", "LONG2NR3", "LONG2NGVAD", "COSTMATCH"])
# s3 는 P3 표현 확장을 맡되, 그 비교에 필요한 anchor 를 같은 서버에서 함께 돈다:
#   Q00(독립 baseline·λ pilot) · Q04(R3) · Q06(R3+GV-H) · Q10(R3+GV-AD) · Q13(A-SIGN) · Q17(GV-B-SIGN) — Q10/Q40/Q17/Q41 이 §8.1 의 A/B 2×2 네 셀이다
S3_IDS = (["T00", "Q00", "Q04", "Q06", "Q10", "Q13", "Q17"] + [f"Q{i:02d}" for i in range(20, 48)]
          + ["CTLGVSCHALF"]                                      # Q43 의 총배율 대조는 Q43 과 같은 서버에
          + ["VXW3AD", "VXW7AD", "VXW3H", "VXW7H", "VXM357AD", "VXM357H", "VXSTD", "VXLOG", "VXRES", "VXM2H", "VXM2AD"])
REPEAT_IDS = ["Q00", "Q04", "Q10"]                       # §11.6 REPEAT-S3: 핵심 대응을 Student seed 3개로


def build(cid, spec_tuple, seed, teacher_run, pilot, updates, version, teacher_seed):
    purpose, rc, st, tri, extra = spec_tuple
    k = dict(campaign_id=CAMPAIGN, run_kind="CONTROLLED", version=version, check_run_name=False,
             recipe="NOALIGN", aligner_policy="A-ID", input_protocol="I-A", na_protocol=extra.get("na_protocol", "NA-STRICT"),
             diag_every=1000, calibration=dict(n_patches=3072, seed=1234),
             select=dict(primary="best_rr_val", aligned_selector=False),
             expect_arch=dict(width=WIDTH, depth=list(DEPTH), noalign=True),      # §20: 손으로 고친 config 가 다른 폭으로 도는 것을 trainer 가 막는다
             rec=dict(rc), stat=dict(st), geom_kd=dict(mode="G0"))
    if tri:
        k["tri"] = dict({kk: dict(v) for kk, v in tri.items()}, shuffle_seed=4321)
    sp0 = resolve(dict(k, teacher=dict(id=TEACHER_ID, run=f"work_dir/{teacher_run}", tag="best_rr_val")))     # Teacher 가 학습 loss 에 필요한지 판정용
    if cid != "T00":
        # GT-only arm(N0·통계 H) 도 **평가 전용**으로 같은 고정 Teacher 를 싣는다 — 모든 Student 를 같은 Teacher 오차 bin 에서 비교하기 위해서다 (§15.2).
        # eval_only=True 면 학습 forward·loss 에는 전혀 쓰이지 않는다 (resolver 가 모순을 막는다).
        k["teacher"] = dict(id=TEACHER_ID, run=f"work_dir/{teacher_run}", tag="best_rr_val", expected_sha256=None, bridge=False,
                            **({} if sp0["needs_teacher"] else dict(eval_only=True)))
    if cid != "T00":
        k["baseline_run"] = pilot.rsplit("/", 1)[0]
    if extra.get("_tcopy"):                              # §13.3 Teacher-copy: 가중치만 T00 에서, optimizer·schedule 은 새로 (step 0)
        k["run_kind"] = "ADAPTIVE_PATH"
        k["phase"] = dict(parent_run=teacher_run, parent_tag="best_rr_val", parent_step=0, optimizer_state_policy="fresh",
                          reason="INIT_TCOPY (§13.3): 같은 가중치에서 KD 없음/R1/R3 를 비교한다", matched_continuation="TCOPYN0")
    if extra.get("_cont"):                               # §13.2 CONTINUATION: 공통 N checkpoint 에서 같은 tail schedule
        k["run_kind"] = "ADAPTIVE_PATH"
        k["phase"] = dict(parent_run=pilot.rsplit("/", 1)[0], parent_tag="last", parent_step=0, optimizer_state_policy="fresh",
                          reason="LONG CONTINUATION (§13.2): 공통 N checkpoint 에서 tail 을 양쪽에 같게 적용", matched_continuation="CONTN0")
    return k, purpose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None, choices=["s2", "s3"], help="서버 block (기본: 두 큐 다 쓰되 config 는 --all 로)")
    ap.add_argument("--all", action="store_true", help="전 case 생성 + s2·s3 큐 둘 다")
    ap.add_argument("--only", default=None, help="쉼표로 구분한 case id (예: T00,Q00,Q04)")
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--teacher-seed", type=int, default=2025)
    ap.add_argument("--repeat-seeds", default="2025,777", help="§11.6 REPEAT-S3 의 추가 Student seed")
    ap.add_argument("--repeat", action="store_true", help="핵심 대응(Q00/Q04/Q10) 을 추가 seed 로도 생성")
    ap.add_argument("--updates", type=int, default=50000, help="N (§13.1: 새 실행 제안 50,000 — 과거 다른 구조의 확인값이 아니다)")
    ap.add_argument("--tail-updates", type=int, default=25000, help="CONTINUATION 의 tail update 수 (양쪽에 같게)")
    ap.add_argument("--cost-match-updates", type=int, default=None, help="COSTMATCH 의 update 수 (KD run 의 실측 시간에서 정한다; 없으면 COSTMATCH 를 만들지 않는다)")
    ap.add_argument("--version", default="v1"); ap.add_argument("--template", default="config/PA_A1_REC_W96_D124_9CH_S1234.yaml")
    ap.add_argument("--params", type=float, default=None)
    a = ap.parse_args()
    import yaml
    if a.params is None:
        from main import import_class
        _c = yaml.safe_load(open(os.path.join(ROOT, a.template)))
        _m = import_class(_c["model"])(**dict(_c["model_args"], hidden_size=WIDTH, depth=DEPTH))
        a.params = round(sum(p.numel() for p in _m.parameters()) / 1e6, 4)

    def name(cid, sp, seed):
        tt = tri_tag(sp)
        return f"NA104_{cid}_{ARCH}_WV3_{rec_tag(sp)}_{stat_tag(sp)}{('_' + tt) if tt else ''}_S{seed}_{a.version}"

    # 1) 이름을 먼저 정한다 — Teacher 와 pilot(Q00) 을 다른 case 가 참조한다
    tmp = cases("TBD/last")
    t_sp = resolve(dict(build("T00", tmp["T00"], a.teacher_seed, "TBD", "TBD/last", a.updates, a.version, a.teacher_seed)[0]))
    teacher_run = name("T00", t_sp, a.teacher_seed)
    q0_sp = resolve(dict(build("Q00", tmp["Q00"], a.seed, teacher_run, "TBD/last", a.updates, a.version, a.teacher_seed)[0]))
    q00_run = name("Q00", q0_sp, a.seed)
    pilot = f"{q00_run}/last"
    C = cases(pilot)

    ids = ([x.strip() for x in a.only.split(",")] if a.only else
           (list(C) if a.all else (S2_IDS if a.server == "s2" else S3_IDS if a.server == "s3" else list(C))))
    cells = [(cid, a.teacher_seed if cid == "T00" else a.seed) for cid in ids if cid in C]
    if a.repeat:
        cells += [(cid, s) for s in (int(x) for x in a.repeat_seeds.split(",")) for cid in REPEAT_IDS if cid in C]
    if a.cost_match_updates is None:
        cells = [(cid, s) for cid, s in cells if cid != "COSTMATCH"]
        print("  (COSTMATCH 는 --cost-match-updates <N> 로 KD run 의 실측 시간에 맞춰 생성한다 — 지금은 만들지 않았다)")
    if a.repeat and str(a.teacher_seed) in a.repeat_seeds.split(","):
        print(f"  ! 반복 seed 에 Teacher seed({a.teacher_seed}) 가 들어 있다 — 그 seed 의 N0 arm 은 초기값·데이터 순서가 T00 과 같아 Teacher 를 재생할 수 있다 (계획 §3.3). "
              "초기값 hash 를 비교하고, 그 결과만으로 KD 효과를 판단하지 않는다.")

    tpl = re.sub(r"^(#.*\n)+", "", open(os.path.join(ROOT, a.template)).read())
    made, rows = [], []
    for cid, seed in cells:
        k, purpose = build(cid, C[cid], seed, teacher_run, pilot, a.updates, a.version, a.teacher_seed)
        sp = resolve(k); tag = name(cid, sp, seed)
        upd = a.updates * C[cid][4].get("_long", 1)
        if C[cid][4].get("_cont"):
            upd = a.tail_updates
        if C[cid][4].get("_cost"):
            upd = a.cost_match_updates or a.updates
        head = (f"# {tag} — [{cid}] {purpose}. 생성: tools/gen_na104_configs.py. 손으로 고치지 말 것.\n"
                f"# 계획: research_log/PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md · 구현 train_kdv.py(trainer kdv) + kdv/ · 노트 research_log/2026-09-11_na104-implementation.md\n"
                f"# 세팅: {describe(sp)}\n"
                f"# 골격 {ARCH}(hidden {WIDTH}, depth {DEPTH}, {a.params} M) · 9ch→8ch · **aligner·sampler 없음, PAN warp 없음** · MS base 1회 합산 · AdamW 1e-4/wd 0.01 cosine warmup100 · batch 48 · {upd} updates · seed {seed}\n"
                f"# 초기값 work_dir/_kdv_init_w104_d122 (같은 seed 의 Student 가 공유) · Teacher = {teacher_run}/best_rr_val (id {TEACHER_ID}) · λ_V pilot = {pilot}\n"
                f"# 선택: 주 selector best_rr_val(valid_wv3.h5 plain ERGAS) + 고정 final-N(last). best_hqnr(=best_raw) 는 FR test 로 고른 exploratory (독립 hold-out 아님). aligned view/selector 없음\n")
        t = re.sub(r"work_dir: .*", f"work_dir: {ROOT}/work_dir/{tag}", tpl)
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
        made.append(tag); rows.append((cid, tag, purpose, describe(sp), upd, seed))

    for srv, wanted in (("s2", S2_IDS), ("s3", S3_IDS)):
        if not (a.all or a.server == srv):
            continue
        qs = [t for cid, t, *_ in rows if cid in wanted and (cid != "T00" or True)]
        order = {cid: i for i, cid in enumerate(wanted)}
        qs = [t for _, t in sorted(((order.get(cid, 999), t) for cid, t, *_ in rows if cid in wanted))]
        qp = os.path.join(ROOT, "config", "queues", f"na104_{srv}{'' if a.version == 'v1' else '_' + a.version}.txt")
        with open(qp, "w") as f:
            f.write(f"# NA104 (W104·D122 no-align KD) {srv} block · 계획 §10 우선순위 순서 · {a.updates} updates · Teacher seed {a.teacher_seed} / Student seed {a.seed} · 시간 제한 없음\n"
                    f"# 순서 의존: T00(Teacher) → Q00(독립 baseline, λ_V pilot = {pilot}) → 나머지. CONT* 는 Q00/last, TCOPY* 는 T00/best_rr_val 에서 분기한다\n"
                    "# 약명 → 세팅은 research_log/2026-09-11_na104-implementation.md §3 표. 실행명 자체에 rec·stat·tri 토큰이 들어 있다\n"
                    + "\n".join(qs) + "\n")
        print(f"queue {srv}: {os.path.relpath(qp, ROOT)} ({len(qs)} run)")
    print(f"config {len(made)}벌 · params {a.params} M · Teacher {teacher_run} · pilot {pilot}")
    for cid, tag, purpose, desc, upd, seed in rows[:6]:
        print(f"  {cid:10s} {tag}")
    if len(rows) > 6:
        print(f"  … 외 {len(rows) - 6}벌")


if __name__ == "__main__":
    main()
