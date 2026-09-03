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
 ("MISC", "⑪ 기타 대조군",
  lambda t: True),
]
ORDER = [c[0] for c in CATS]
NAME  = {c[0]: c[1] for c in CATS}

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
