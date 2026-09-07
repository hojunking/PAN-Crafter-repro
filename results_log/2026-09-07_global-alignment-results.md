# Global alignment 40h 결과 — 정렬은 안 되고, jitter 만 남았다 (2026-09-07 정리)

계획: [s1_w152_d123_global_alignment_40h_plan.md](../research_log/s1_w152_d123_global_alignment_40h_plan.md) ·
검토·구현·9건 수정: [2026-09-04_global-alignment-plan-review.md](../research_log/2026-09-04_global-alignment-plan-review.md) ·
계획 대비 차이·이슈 상세: [2026-09-05_global-alignment-plan-vs-implementation.md](../research_log/2026-09-05_global-alignment-plan-vs-implementation.md)
7벌 완주 + 대조 run 1벌(C4 2벌은 pretrain gate FAIL 로 미학습). 판정: best checkpoint HQNR(12-19) → fSCC. anchor `S1_T05_W152_D123_DUAL` 0.9546.

| run | best HQNR | fSCC | plateau | final | 비고 |
|---|---:|---:|---:|---:|---|
| anchor (bicubic) | 0.9546 | 0.8846 | 0.9516 | 0.9502 | |
| P0 (interp23tap) | 0.9543 | 0.8864 | 0.9492 | 0.9465 | phase 수정: 무효과, 후반 plateau −0.0024 |
| C1 round-trip | 0.9383 | 0.8555 | 0.9159 | 0.9124 | 붕괴 (best ep5, inverse warp 과선명화) |
| C3 dual-frame(P 출력) | 0.9245 | 0.9008 | 0.9224 | 0.9241 | 공식 D_λ 좌표 충돌 0.018 |
| C2 α0.25 / 0.5 / 0.75 / 1.0 | 0.9534 / **0.9553** / 0.9538 / **0.9553** | 0.887 / 0.892 / 0.892 / 0.900 | 0.9518 / 0.9529 / 0.9526 / 0.9528 | 0.9514 / 0.9525 / 0.9533 / 0.9552 | 유일한 양성 |
| CTRL (bicubic + C2 jitter) | 0.9539 | 0.8973 | 0.9523 | 0.9534 | 커널 교체와 독립 |

핵심 사실 — 상세는 위 문서들:
1. 데이터셋 `lms` 는 interp23tap 이고 계획의 phase-2 bicubic 이 아니다. train patch 의 audit Δ 는 추정 노이즈(오차 0.27 px > shift 0.06)라
   C 계열의 "정렬" 은 학습 시 sd 0.076 LR px jitter 였다.
2. **C2 의 이득은 정렬이 아니라 jitter**: C2 checkpoint 에 추론 shift 를 빼도 HQNR 이 같다(0.9555/0.9560). 원 커널 대조(CTRL)에서도 final +0.003.
3. 1차 P0 는 LR 증강 × phase-2 오정렬(표본 75%, 1 HR px)로 무효화·재기동했다. 이후 9건 수정(집계식·tie anchor·50K 평가·provenance·rc=4·smoke).
4. 판정 밴드 0.011 은 공식 HQNR 에서 측정된 적이 없다(2026-09-04 문서). 실측 상한 ~0.0027.
5. 후속 캠페인(shift-robust, [2026-09-07_shift-robust-results.md](2026-09-07_shift-robust-results.md))이 무작위 jitter 만으로 같은 효과를 재현했다.
