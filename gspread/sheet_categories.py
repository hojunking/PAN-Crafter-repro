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
 ("MISC", "⑪ 기타 대조군",
  lambda t: True),
]
ORDER = [c[0] for c in CATS]
NAME  = {c[0]: c[1] for c in CATS}
SEP = "▍"          # 구분행 B열 접두. refile_sheet 와 gspread_upload 가 같이 쓴다

# 캠페인 설명 — 구분행의 Notes 에 들어간다. 여기 있는 범주만 업로드 시 구분행을 자동으로 넣는다
# (업로드는 시트 맨 아래에 덧붙이므로, 새 캠페인이 지난 실험과 섞여 보이지 않게 한다).
DESC = {
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
