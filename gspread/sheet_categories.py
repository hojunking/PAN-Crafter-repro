"""접근법 범주 분류 — 시트 정리용. 단일 소스."""
import re

# (범주키, 표시명, 판별함수)  — 위에서부터 처음 맞는 것으로 확정한다
CATS = [
 ("MS", "⑨ MS-only (MARs PAN mode 제거) 계열 — 시드복제·dual 대조군 포함",
  lambda t: t.startswith(("MS1_","MS2_")) or t == "R4_dual_ctrl"),
 ("REF", "① 기준·참조 — 논문 / 외부모델 / 배포본 재현 / 논문충실 재구성",
  lambda t: t.startswith(("■","□")) or t.startswith(("wv3_baseline","wv3_fixed","paper_wv3","paper_ln","paper_nocrop","s1_A0","s1_A1"))),
 ("SEED", "② 시드 변동 측정 — 판정선(2σ) 산출용",
  lambda t: re.search(r"_s(1234|7777|2025)\b", t) is not None),
 ("P25", "③ 25K 예비 스크리닝 — 구조·세팅 탐색 (50K 와 가로 비교 금지)",
  lambda t: t.startswith(("p25_","x3_maxcut"))),
 ("SUBMOD", "④ 서브모듈 제거 경량화 (c 계열) — 무엇을 빼도 되는가",
  lambda t: re.match(r"^c\d", t) or t.startswith(("m1_single","CM3A_btl_nopan"))),
 ("ARCH", "⑤ 아키텍처 탐색 24h (N/R/A/L 계열) — 폭·깊이·비대칭·신규구조",
  lambda t: re.match(r"^(N\d|R\d|A\d|L1_)", t) is not None),
 ("ATTN", "⑥ Swin·CM3A attention + 압축 귀속 (SW/d122 계열)",
  lambda t: t.startswith(("SW","CM3A_d122","d122","LR_"))),
 ("KD", "⑦ KD 캠페인 — Teacher(T) / Student 사다리(K0~K4)",
  lambda t: re.match(r"^(T\d_|K\d)", t) is not None),
 ("SE", "⑧ SE(SENet) ablation — 채널 재조정",
  lambda t: t.startswith("SE") and not t.startswith("SEED")),
 ("MUT", "⑩ Mutual / DML 상호학습 (s2)",
  lambda t: t.startswith(("M0_","M1_","M2_","M3_","dml_"))),
 ("GA", "⑫ Global alignment (GA 계열) — interp23tap phase 수정 + 전역 sub-pixel shift (frozen/partial/trainable)",
  lambda t: t.startswith("GA_")),
 ("SR", "⑬ Shift-robust conditioning + M-frame PAN guidance (SR/AF 계열) — random jitter·blur control·consistency·global correlator",
  lambda t: t.startswith(("SR_", "AF_"))),
 ("UVS", "⑭ UVS-KD (s2) — uncertainty routing · GT residual variance · shift-token KD · teacher forcing (teacher c0_hqnr → d122)",
  lambda t: t.startswith("UVS_")),
 ("KDV", "⑳ s2 W112·D123 GT-anchored adaptive KD · 출력 통계 variance · aligner 재사용 (S2W112D123 계열) — REC N0/R1/R3, STAT GV-H/AD, aligner A-FR/A-FT/A-SC/A-ID, Teacher T112",
  lambda t: t.startswith("S2W112")),
 ("PO10", "⑲ PAN 추가 변위 + offset consistency (PO10, N1–N3) — 학습 PAN 에 원판 R=1 HR px 무작위 변위, ĉε+ε≈ĉ0 감독, aligner 고정 내부 view 4px",
  lambda t: t.startswith("PO10_")),
 ("PA", "⑱ PAN 앞단 전역 정합 A1–A3 (PA 계열) — 학습되는 global shift CNN 이 PAN 만 MS 프레임으로 sampling, W96·D124 B0 위 · A1 recon / A2 +edge / A3 +geo",
  lambda t: t.startswith("PA_A")),
 ("BASE96", "⑰ 새 baseline W96·D124 U-Net — MS+PAN 9ch · LPAN/HPAN 없음 · PAN task 없음(단일 HRMS) · MARs γ/β 제거 · WV3 3-seed (s3)",
  lambda t: t.startswith("BASE_W96_D124_MSPAN_")),
 ("MULTISET", "⑯ 아키텍처 고정 다중 데이터셋 3-seed — W168·d123·dual·11ch·nocrop·attn 없음, WV3/QB/GF2 학습(seed 2025·1234·7777) + WV2 zero-shot",
  lambda t: t.startswith("ARCH_W168_D123_DUAL_") or "_zs_" in t),
 ("S1GRID", "⑮ Teacher 후보 격자 (S1 계열) — 폭 W·깊이 D·MS2/DUAL 스크리닝, HQNR 선택 (S1_T05_W168_D123_DUAL 이 SR anchor)",
  lambda t: t.startswith("S1_")),
 ("MISC", "⑪ 기타 대조군",
  lambda t: True),
]
ORDER = [c[0] for c in CATS]
NAME  = {c[0]: c[1] for c in CATS}
SEP = "▍"          # 구분행 B열 접두. refile_sheet 와 gspread_upload 가 같이 쓴다

# 캠페인 설명 — 구분행의 Notes 에 들어간다. 여기 있는 범주만 업로드 시 구분행을 자동으로 넣는다
# (업로드는 시트 맨 아래에 덧붙이므로, 새 캠페인이 지난 실험과 섞여 보이지 않게 한다).
DESC = {
 "KDV": ("[캠페인] s2 W112·D123 KD·variance·aligner 재사용 · seed 1234(Teacher 2025) · 2026-09-10 · research_log/PAN_S2_W112_KD_Variance_Plan_and_References_2026-09-10/. "
         "Student/Teacher 모두 W112·D123 U-Net(9ch, 단일 HRMS, γβ 제거; 계획 원안 D124 → 사용자 결정 D123, s1 PO10 R200 과 같은 골격). 이름 S2W112D123_<recipe>_<input>_<aligner>_<rec>_<stat>_<geomKD>_s<seed>_<ver>: "
         "recipe NOALIGN(aligner 없음)/A1(donor = s1 PA_A1 seed2025 aligner)/T112DFR(donor frozen + 복원 supervised Teacher) · input IA(native) · "
         "aligner AID(없음)/AFR(donor frozen, Teacher 와 correction 공유)/AFT(donor 초기화 후 학습)/ASC(독립 초기화 학습) · "
         "rec N0(L1)/R1((1+d_T)L1)/R3(adaptive hard/soft, τ_R train calibration) · stat OFF/GVH(GT gradient-variance 5×5)/GVAD(GT+Teacher adaptive, τ_V·λ_V calibration) · G0(alignment KD 없음). "
         "시트 HQNR = best_raw 의 raw_original(전체 프레임) + HQNR(V64). best_aligned·best_rr_val·last, fitting_bins.csv, gradient 진단은 run 폴더"),
 "PO10": ("[캠페인] PO10 offset consistency 10 GPU-h · s1 seed 2025 · 2026-09-10 · research_log/PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md. "
          "A1 골격·aligner·init 그대로, optimizer update 를 native/corrupt 1:1 교대: corrupt 는 PAN 에만 원판(R=1 HR px, audit 부록 E train P90 0.25 LR px×4) 무작위 변위 ε. "
          "N1 L_rec / N2 +0.01·|ĉε+ε−sg(ĉ0)| (5K ramp) / N3 같은 loss, stop-gradient 없음. aligner 는 고정 내부 crop(4 px) 만 본다. "
          "시트 HQNR = best_raw 의 raw_original. 반응 진단(offset_response_*.csv)·gradient 진단은 run 폴더. 과거 A1 은 전체 view 라 배경 기준"),
 "PA": ("[캠페인] PAN 앞단 전역 정합 A1–A3 · 2026-09-09 · research_log/PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md. "
        "B0(BASE_W96_D124_MSPAN) 골격·학습 조건 그대로, 앞에 dual-stem global shift CNN(0.105M, zero-init head) 을 붙여 PAN 만 Δ̂ 로 bicubic sampling(P̃). "
        "MS base·GT·출력은 M-frame 고정. A1 L_rec / A2 +0.1·L_edge(Scharr) / A3 +0.01·L_geo(normalized gradient outer product), 5K ramp. "
        "서버-seed block: s1 2025 (A1→A2→A3) · s2 1234 (A2→A3→A1) · s3 7777 (A3→A1→A2). 시트 HQNR 은 best_raw 의 raw_original view(원 PAN, 전체 프레임). "
        "raw_valid·aligned_valid·best_aligned 는 run 폴더 checkpoint_metrics.csv / pa_diag.json"),
 "BASE96": ("[캠페인] 새 baseline W96·D124 WV3 3-seed · s3 · 2026-09-09 · research_log/PAN_research_baseline_W96_D124_2026-09-09.md 의 고정 기준: "
            "U-Net W96 · depth [1,2,4] · 입력 MS+PAN 만(9ch, in_mode paper) · LPAN/HPAN 채널 없음 · PAN reconstruction task·loss 없음(mars ms) · "
            "MARs mode γ/β 제거(mode_modulation false) · attention 없음 · crop=False · bicubic 잔차 base · 50K AdamW 1e-4/wd0.01 cosine · "
            "seed 2025·1234·7777. best 선택·보고 = 논문 세트(.mat 20장 전체) HQNR. config: tools/gen_w96_d124_mspan_configs.py · "
            "큐 config/queues/base_w96_d124_mspan_wv3_3seed.txt. 과거 W168·d123 dual 결과는 직접 대조군이 아니다(문서 §7)"),
 "MULTISET": ("[캠페인] 아키텍처 고정 다중 데이터셋 3-seed · 2026-09-08/09 · S1_T05_W168_D123_DUAL 구조(W168 · depth [1,2,3] · dual MARs · 11ch · "
              "crop=False · attention 없음 · 50K · AdamW 1e-4/wd0.01 cosine) 그대로, 데이터셋만 바꾼다: WV3(seed 2025·1234·7777) · "
              "QB(4밴드, 학습셋 ms 복구본 F-3, FR lpan 복구본) · GF2(4밴드, max_pixel 1023) 각 3 seed + WV3 checkpoint 의 WV2 zero-shot(_zs_wv2). "
              "best 선택·보고 모두 논문 세트(.mat 20장 전체, 지표 v2) HQNR — 12-19 부분집합 없음. 각 서버(s1/s2/s3)가 같은 큐(config/queues/arch_w168_multiset_3seed.txt)를 돈다. "
              "tools/arch_multiset_prepare.sh 로 준비·기동"),
 "UVS": ("[캠페인] UVS-KD 30h · s2 · 2026-09-06 · teacher c0_hqnr(7.17M, CM3A 3) → student d122(3.18M). 공통: 입력·잔차 base = 제공 lms, "
         "PAN 3ch 는 δ(LR px)×4 강체 warp, PAN mode 는 raw. teacher 신호(R_T·U_T·δ_T·c_T)는 cache. "
         "B0 baseline / K0 output KD / K1 uncertainty routing / K2 +GT residual variance / S0 shift-token KD / M1 K2+shift / "
         "M2 +teacher forcing(η 1→0, 5K–20K) / M3 +shift-effect loss / R1 seed / C1 w96. 판정 HQNR→fSCC + controlled-shift AUC. "
         "계획 research_log/2026-09-06_uvs-kd_30h_experiment-plan.md · 검토 research_log/2026-09-06_uvs-kd-plan-review.md"),
 "SR": ("[캠페인] Shift-robust 30h · s1 · 2026-09-06 · backbone W168·d123 dual 11ch nocrop 50K, **원 bicubic·원 feeder·M-frame 출력** 고정 "
        "(interp23tap·cache·inverse 전부 폐기). 바뀌는 것은 네트워크가 보는 MS 조건 채널뿐: J1 ±0.5 HR px 무작위 전역 jitter(두 mode) / "
        "J2 MS mode 만 / J3 위치 이동 없는 matched Gaussian blur(σ* 보정) / J4 clean+jitter 두 branch + 잔차 consistency λ0.1 / "
        "G1 first conv 를 PAN·MS 기여로 분리해 synthetic-supervised global correlator 로 PAN feature 만 M-frame 으로 sampling. "
        "판정 HQNR(장면별 평균)→fSCC(12-19). anchor S1_T05_W168_D123_DUAL HQNR 0.9571 / fSCC 0.8785. "
        "계획 research_log/s1_w168_d123_shift_robust_alignment_30h_plan.md · 검토 research_log/2026-09-06_shift-robust-plan-review.md"),
 "GA": ("[캠페인] Global alignment 40h · s1 · 2026-09-04 21:27 기동 · backbone W152·d123 dual 11ch nocrop 50K 고정. "
        "공통 변경: 입력 MS/LPAN 업샘플을 bicubic(phase 1.5) → interp23tap(=데이터셋 lms 정확 재현) 로 교체. "
        "Δ = GT 없이 MTF↓PAN vs LRMS 로 추정한 전역 sub-pixel shift(LR px; FR 12-19 ≈(−0.16,+0.18), train 은 추정 노이즈). "
        "P0 phase 보정만 / C1 frozen round-trip(M 출력) / C3 frozen dual-frame(P 출력, GT loss 만 inverse) / "
        "C2 조건입력 부분 shift α / C4 trainable ShiftNet(pretrain gate FAIL 시 미실행). "
        "판정 HQNR→fSCC(12-19). anchor S1_T05_W152_D123_DUAL HQNR 0.9546. "
        "계획 research_log/s1_w152_d123_global_alignment_40h_plan.md · 검토 research_log/2026-09-04_global-alignment-plan-review.md"),
}


def separator_cell(key):
    return f"{SEP}{NAME[key]}"

def run_tag(cell):
    """B열 문자열에서 실행명만 뽑는다.  'K0_R4_base (50K) · w96 ...' -> 'K0_R4_base'"""
    s = cell.strip()
    if s.startswith(("■","□")):
        return s
    return re.split(r"\s*[(·]", s)[0].strip()

def classify(cell):
    t = run_tag(cell)
    for key, _, f in CATS:
        if f(t):
            return key
    return "MISC"
