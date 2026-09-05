"""UVS-KD: Uncertainty–Variance guided Shift distillation (research_log/2026-09-06_uvs-kd_30h_experiment-plan.md).

shift  : MS 격자(LR) cost-volume shift 모듈 (Teacher/Student 채널만 다름), PAN warp, 좌표 규약
losses : GT residual variance 가중, uncertainty hard/soft routing, shift-token KD, teacher forcing 스케줄
좌표 규약 (align/resample.py 와 동일): W(I, δ)(y,x) = I(y+δy, x+δx). 이번엔 PAN 을 MS/GT 격자에 맞추므로
모듈 출력은 δ_{MS←PAN} (LR px) 이고 full-res PAN 에는 4δ 를 쓴다. audit cache 의 Δ(P←M) 와는 δ = −Δ 관계다.
"""
