"""PAN 앞단 전역 정합(A1–A3) — research_log/PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md 의 구현.

aligner.py   §3  dual-stem global shift CNN (Δ̂=(dy,dx), HR px, zero-init head)
warp.py      §4  PAN 만 bicubic/border/align_corners=False 로 sampling (0 우회 없음) + sampling support mask
losses.py    §5  L_rec(전 영역 L1) · L_edge(A2, Scharr signed) · L_geo(A3, normalized gradient outer product)
model.py     §2  PAModel: Ŷ = M + Fθ(concat(P̃, M)),  M=bicubic↑S
evalviews.py §10 raw_original / raw_valid / aligned_valid 세 view (같은 SR, 고정 V, P̃ 로 저해상도 PAN 재생성)
selector.py  §10.8 running-max·tie band·later-step 선택기 (best_raw / best_aligned 독립)
"""
