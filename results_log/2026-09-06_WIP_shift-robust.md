# [WIP] Shift-robust conditioning + M-frame PAN guidance 30h — s1 (2026-09-06)

계획: [s1_w168_d123_shift_robust_alignment_30h_plan.md](../research_log/s1_w168_d123_shift_robust_alignment_30h_plan.md) ·
검토·구현: [2026-09-06_shift-robust-plan-review.md](../research_log/2026-09-06_shift-robust-plan-review.md)
판정: **best checkpoint HQNR(12-19, 장면별 평균) → fSCC**. anchor `S1_T05_W168_D123_DUAL` HQNR **0.9571** / fSCC 0.8785 / D_λ 0.0231 / D_s 0.0202.

## 상태
- 구현·검증 완료(T01–T24 24/24 · smoke 5/5). **기동 대기** — 대조 run `GA_CTRL_C2_BICUBIC15_A100`(≈03:15 종료) 뒤.
- 큐 `config/queues/s1_shift_robust.txt`: J1 → J3 → J4 → J2 → G1, 이후 `gate_sr` 가 winner seed 1234 반복(또는 radius refinement) 1벌.
- J3 의 σ* = **1.225 HR px (MSE 매칭)**. grad-energy 매칭은 σ* 0.10 ≈ 항등이라 쓰지 않음(검토서 §2).

## 결과 표 (완주 시 채운다)

| run | 세팅 | best HQNR | ep | fSCC | D_λ | D_s | plateau(≥100) | final |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| anchor `S1_T05_W168_D123_DUAL` | 원 bicubic · 무perturbation | 0.9571 | 100 | 0.8785 | 0.0231 | 0.0202 | | |
| `SR_J1_C2RAND_BOTH_R050` | ±0.5px 무작위 jitter, 두 mode | | | | | | | |
| `SR_J3_BLURCTRL_MATCH_R050` | Gaussian σ1.225(MSE 매칭), 두 mode | | | | | | | |
| `SR_J4_CJCONS_R050_L010` | clean+jitter branch, consistency λ0.1 | | | | | | | |
| `SR_J2_C2RAND_MSONLY_R050` | jitter MS mode 만 | | | | | | | |
| `AF_G1_PAN2M_GLOBALCORR` | PAN feature global correlator | | | | | | | |
| (gate) seed 1234 / refinement | | | | | | | | |

## 볼 곳
- 학습 로그: `eps y/x mean±std`, `geR`(jitter 조건의 gradient-energy 비), J4 `cons/lam/ratio`, G1 `shift/delta/err/conf/pB/pC`
- `metrics.csv`: `hqnr_official, fscc_official, d_lambda_official, d_s_official`, G1 `g1_*`
- 완주 후: `tools/sr_infer_diag.py --run work_dir/<run>` (추론 jitter / β sweep), `tools/local_align_diag.py --run work_dir/<winner>` (§12)
