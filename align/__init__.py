"""Global alignment (research_log/s1_w152_d123_global_alignment_40h_plan.md).

resample  : interp23tap(정확한 lms 재현) · phase-2 bicubic(계획 원안) · HR warp · border mask
estimator : Scharr+ZNCC(primary) / Census+Hamming(secondary) 전역 sub-pixel shift 추정
cache     : split 별 shift cache (CSV + cache_meta.json + SHA256)
shiftnet  : GlobalShiftNet + 구조맵 입력
model     : AlignedModel — backbone 앞/뒤 wrapper. 코어 U-Net 은 건드리지 않는다
"""
