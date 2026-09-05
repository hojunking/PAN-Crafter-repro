"""Shift-robust conditioning + M-frame PAN guidance (research_log/s1_w168_d123_shift_robust_alignment_30h_plan.md).

jitter    : conditioning MS 의 HR 전역 sub-pixel translation(J1/J2), matched blur(J3), 보정
pan_align : first conv 의 PAN/MS 기여 분리(§10), synthetic-supervised global correlator(G1, §11),
            local field 적용(§12 진단)
좌표계는 전부 M-frame 이다. 출력·GT·잔차 base 는 움직이지 않는다. 부호 규약은 align/resample.py 와 같다:
  out[y,x] = src[y+dy, x+dx]  (warp_hr)
"""
