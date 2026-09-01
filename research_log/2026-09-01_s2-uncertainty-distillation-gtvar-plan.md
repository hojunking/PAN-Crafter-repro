# s2 — MS-only Student Uncertainty Distillation + GT-Variance 계획 (2026-09-01)

> 상태: 실행 전 계획  
> 서버: **s2 전용**  
> Teacher: **s2에서 독립 학습하는** `S2_T00_W160_D122_MS2_UQREADY`  
> Student: W96 · d124 · 11ch · no-attn · crop=False · **clean MS-only**  
> 반복: 각 설정 seed 2회

| 단계 | 신규 설정 수 | Seed 포함 run 수 | 예상 s2 GPU 시간 |
|---|---:|---:|---:|
| s2 local teacher mean + UQ head | 1 | 1 | 2.8–3.2h |
| Plain KD 대조 | 1 | 2 | 4.0–4.8h |
| Uncertainty λ sweep | 3 | 6 | 12.0–14.4h |
| GT-output variance 비교 | 1 | 2 | 4.2–5.0h |
| Smoke·평가·집계 | — | — | 1.5–2.0h |
| **권장 전체** | **6** | **11** | **24.5–29.4h** |

사용자가 요청한 핵심 student 8 run(`uncertainty 6 + GT variance 2`)만 실행하면, s2 local teacher 생성까지 포함해 약 **20.5–25.7h**다. Plain KD 2회는 uncertainty routing의 순수 효과를 분리하기 위한 권장 대조군이다.

---

## 1. 연구 질문

1. Teacher uncertainty가 높은 위치에서는 GT를 더 따르고, 낮은 위치에서는 teacher mean을 더 따르는 것이 clean MS-only student에 유효한가?
2. U-Know 계열의 기준값 `λ=0.1` 주변에서 distillation 강도의 적정 범위는 어디인가?
3. 결과가 seed 2025/1234에서 같은 방향으로 재현되는가?
4. GT HRMS의 local-detail variance와 student reconstruction의 local-detail variance를 맞추면 detail fitting이 개선되는가?

### 중요한 정의

이 문서의 **GT variance**는 정답의 확률적 label uncertainty가 아니다.

```text
GT variance      = GT residual의 local detail-energy map
Student variance = Student output residual의 같은 local variance 연산 결과
Teacher UQ       = Teacher reconstruction error에 calibration된 predictive variance
```

GT variance와 Teacher UQ는 의미가 다르다. 따라서 GT variance를 Teacher/Student predictive uncertainty의 pixel-wise 정답으로 사용하지 않는다.

---

## 2. s2 local teacher 생성

s1 checkpoint는 사용하지 않는다. s2에서 아래 teacher를 처음부터 독립 학습한다.

```text
ID: S2_T00_W160_D122_MS2_UQREADY
width=160 · depth=[1,2,2]
11ch · no-attn · crop=False
PAN input 유지 · clean MS-only reconstruction
teacher seed=2025
```

절차:

1. s2에서 mean backbone을 새로 초기화하고 clean MS2 50K 학습한다.
2. best-HQNR mean checkpoint를 고정한다.
3. 마지막 decoder feature에 1-channel log-variance head를 붙인다.
4. Mean backbone을 freeze하고 uncertainty head만 5–10K calibration한다.

```text
Conv3x3(C → C/4) → SiLU → Conv1x1(C/4 → 1) → log variance
```

Teacher local error:

\[
e_T^{loc}
=
A_3\left(\frac1C\sum_c(Y_c-\mu_{T,c})^2\right)
\]

Head-only loss:

\[
L_{T,U}
=
\frac12\exp(-s_T)e_T^{loc}+\frac12s_T,
\qquad s_T=\log\sigma_T^2
\]

예상 시간:

| 작업 | 예상 |
|---|---:|
| Teacher mean 50K | 2.2–2.5h |
| Head-only 5–10K | 0.3–0.5h |
| Calibration 평가 | 0.3h |
| **합계** | **2.8–3.2h** |

구조와 설정은 s1 Case 0과 같지만 가중치, optimizer, checkpoint는 공유하지 않는다.

Teacher는 모든 s2 case에서 완전히 고정한다.

```python
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

with torch.no_grad():
    teacher_mean, teacher_logvar = teacher(...)
```

Teacher calibration에서 `Spearman(variance,error)>0`, monotone error quintile, global-variance 대비 NLL 개선을 확인한 후 student queue로 넘어간다.

---

## 3. 고정 Student

```yaml
architecture: res_unet
width: 96
depth: [1, 2, 4]
attention_locations: []
input_channels: 11
crop: false

task: ms_only_plain
pan_input: true
pan_reconstruction: false
pan_mode: false
mode_modulation: false

iterations: 50000
optimizer: AdamW
lr: 1.0e-4
weight_decay: 0.01
scheduler: cosine
warmup_steps: 100
checkpoint_selection: val_hqnr
```

기존 s2 clean-MS2 baseline 두 seed는 다음과 같다.

| Run | Seed | ERGAS↓ | SCC↑ | HQNR↑ | Train |
|---|---:|---:|---:|---:|---:|
| `MS2_R4_plain_msonly` | 2025 | 2.0640 | 0.9910 | 0.9539 | 1.33h |
| `MS2_R4_plain_msonly_s1234` | 1234 | 2.0628 | 0.9909 | 0.9544 | 1.33h |

동일 commit/data/evaluator라면 이 두 run을 no-KD baseline으로 재사용한다. 코드 경로가 달라졌다면 `S2_B0_MS2_NOKD`를 두 seed로 다시 실행하며 총 2.7–3.2h가 추가된다.

---

## 4. Uncertainty-guided distillation 정의

Teacher의 log-variance를 고정된 train-set q05/q95로 robust normalization한다.

\[
u_T=
\operatorname{clip}
\left(
\frac{s_T-q_{05}}{q_{95}-q_{05}+\epsilon},0,1
\right)
\]

가중치는 각 이미지에서 평균 1로 정규화한다.

\[
w_{hard}=\operatorname{MeanNorm}(1+u_T)
\]

\[
w_{soft}=\operatorname{MeanNorm}(\max(1-u_T,0.05))
\]

Student loss:

\[
L_{hard}^{U}
=
\operatorname{mean}
\left[w_{hard}\,\rho(\hat Y_S-Y)\right]
\]

\[
L_{soft}^{U}
=
\operatorname{mean}
\left[w_{soft}\,\rho(\hat Y_S-\operatorname{sg}(\hat Y_T))\right]
\]

\[
L_{UKD}=L_{hard}^{U}+\lambda_U(t)L_{soft}^{U}
\]

`ρ`는 기존 MS2와 같은 reconstruction penalty를 사용한다. 이번 sweep에서는 full-output KD를 사용한다. 현재 결과에서 full KD가 spectral-only KD보다 안정적이었으므로 spectral/feature KD를 동시에 섞지 않는다.

여기서 `λ_U`는 별도의 uncertainty NLL weight가 아니라, **uncertainty로 routing된 teacher soft target의 강도**다.

### λ schedule

모든 λ case에서 최대값만 바꾸고 schedule 형태는 고정한다.

```text
0–5K      : 0
5–15K     : 0 → lambda_max linear ramp
15–40K    : lambda_max
40–50K    : lambda_max → 0 linear decay
```

50K 끝에서는 GT hard reconstruction으로 다시 수렴하게 한다.

---

## 5. Case matrix

### 5.1 Plain KD 대조 — 권장 2 run

Uncertainty routing 없이 같은 teacher mean을 증류한다.

\[
L_{plainKD}=L_{MS}+0.1\,\rho(\hat Y_S-\operatorname{sg}(\hat Y_T))
\]

| 순서 | ID | Seed | 예상 |
|---:|---|---:|---:|
| 0 | `S2_PKD_L010_S2025` | 2025 | 2.0–2.4h |
| 1 | `S2_PKD_L010_S1234` | 1234 | 2.0–2.4h |

이 두 run이 있어야 uncertainty case의 개선을 "새 teacher의 일반 KD 효과"와 분리할 수 있다.

### 5.2 Simple uncertainty λ sweep — 요청한 6 run

중심값 0.10을 기준으로 약 1/3배와 3배를 둔 log-spaced sweep이다.

| 순서 | ID | `lambda_U_max` | Seed | 예상 |
|---:|---|---:|---:|---:|
| 2 | `S2_UKD_L003_S2025` | 0.03 | 2025 | 2.0–2.4h |
| 3 | `S2_UKD_L003_S1234` | 0.03 | 1234 | 2.0–2.4h |
| 4 | `S2_UKD_L010_S2025` | 0.10 | 2025 | 2.0–2.4h |
| 5 | `S2_UKD_L010_S1234` | 0.10 | 1234 | 2.0–2.4h |
| 6 | `S2_UKD_L030_S2025` | 0.30 | 2025 | 2.0–2.4h |
| 7 | `S2_UKD_L030_S1234` | 0.30 | 1234 | 2.0–2.4h |

`0.03/0.10/0.30`은 세 점만으로 under-coupled/기준/over-coupled 영역을 분리하기 위한 범위다. 더 좁은 미세탐색은 이 결과 이후에만 수행한다.

### 5.3 GT-output variance comparison — 요청한 2 run

두 uncertainty seed 결과를 평균해 `lambda_U*`를 먼저 고정한다. 그런 다음 GT variance term만 추가한다.

| 순서 | ID | 설정 | Seed | 예상 |
|---:|---|---|---:|---:|
| 8 | `S2_GTVAR_BESTU_S2025` | best `lambda_U` + GT-output variance | 2025 | 2.1–2.5h |
| 9 | `S2_GTVAR_BESTU_S1234` | best `lambda_U` + GT-output variance | 1234 | 2.1–2.5h |

---

## 6. GT variance loss의 정확한 정의

Raw HRMS의 밝기 분산이 아니라 upsampled MS 대비 residual에서 local detail variance를 계산한다.

\[
R_{GT}=Y-\operatorname{UpMS},
\qquad
R_S=\hat Y_S-\operatorname{UpMS}
\]

Window `k`의 band-mean local variance:

\[
V_k(R)
=
\frac1C\sum_c
\left[A_k(R_c^2)-A_k(R_c)^2\right]
\]

Multi-scale map:

\[
V(R)=0.5V_3(R)+0.3V_5(R)+0.2V_9(R)
\]

Training set에서 고정한 scale `κ`로 두 map을 동일하게 normalize한다.

\[
\widetilde V(R)=\frac{V(R)}{V(R)+\kappa}
\]

GT-output variance loss:

\[
L_{GTVar}
=
\operatorname{SmoothL1}
\left(
\widetilde V(R_S),
\operatorname{sg}(\widetilde V(R_{GT}))
\right)
\]

최종 loss:

\[
L_G=L_{UKD}(\lambda_U^*)+\lambda_V L_{GTVar}
\]

초기값:

```yaml
lambda_gtvar: 0.10
```

1K gradient audit에서 다음을 확인한다.

```text
||grad(lambda_V * L_GTVar)|| / ||grad(L_hard)|| = 0.05–0.10 목표
```

`lambda_V=0.10`을 적용했을 때 위 gradient ratio가 0.10을 넘으면 `lambda_V=0.05`로 낮춘다. 이 조정은 두 seed 전에 한 번만 하고 양 seed에 동일하게 고정한다.

이 case는 기존 K3의 `1 + beta * V_GT` hard weighting과 다르다. 기존 K3는 detail 영역의 reconstruction 비중을 높였고, 이번 case는 사용자가 요청한 대로 **GT와 student output의 local variance를 직접 비교**한다.

---

## 7. Seed 운용

```text
student model-init seeds: 2025, 1234
teacher checkpoint: 동일
split/evaluator: 동일
```

가능하면 data-order와 augmentation seed는 2025로 고정하고 student initialization만 바꾼다. 현재 runner가 하나의 global seed만 지원하면 global seed 2025/1234를 사용하고, 그 결과를 initialization뿐 아니라 data-order를 포함한 전체 run 변동성으로 해석한다.

각 λ는 두 seed를 모두 끝낸 뒤 비교한다. seed 하나가 좋아도 다른 seed가 반대 방향이면 안정적인 효과로 채택하지 않는다.

---

## 8. 실행 순서

```text
0–3.2h     s2 local teacher mean + uncertainty head
3.2–8.0h   PKD λ=0.10 × 2 seeds
8.0–12.8h  UKD λ=0.03 × 2 seeds
12.8–17.6h UKD λ=0.10 × 2 seeds
17.6–22.4h UKD λ=0.30 × 2 seeds
22.4h      seed-paired λ 선택
22.4–27.4h GT-output variance × 2 seeds
27.4–29.4h RR/FR/UQ/variance 분석 및 buffer
```

실제 first-1K throughput으로 시간표를 갱신한다.

| 실행 범위 | s2 총 GPU/wall time |
|---|---:|
| s2 teacher + 요청 핵심 student 8 run | 20.5–25.7h |
| s2 teacher + Plain KD 포함 student 10 run | 24.5–29.4h |
| no-KD baseline도 신규 2회 | 27.2–32.6h |

s1과 s2는 처음부터 독립 실행된다. s2는 s1 완료나 checkpoint 전달을 기다리지 않는다.

---

## 9. 판정 규칙

### 9.1 Reconstruction/FR

1. 두 seed 평균 `|ΔHQNR| > 0.011`이면 HQNR로 결정
2. HQNR 동급이면 SCC → ERGAS → SAM/Q2n
3. seed별 방향이 일치하는지 반드시 병기
4. 기존 MS2 no-KD 및 same-teacher plain KD와 모두 비교

채택 최소조건:

- 두 seed 모두 HQNR이 baseline보다 0.011 이상 악화하지 않음
- 평균 SCC 비열화 없음
- 평균 ERGAS 최소 0.5% 개선 또는 RR 20장 중 12장 이상 승리
- 한 seed의 큰 개선이 다른 seed의 악화를 가린 결과가 아님

### 9.2 Uncertainty routing 진단

```text
Spearman(teacher UQ, teacher absolute error)
teacher UQ top-10/20/30% error lift
hard/soft weight mean, std, min, max
weighted soft-KD gradient / hard-gradient ratio
```

### 9.3 GT-output variance 진단

```text
SmoothL1(V_S, V_GT)
Spearman(V_S, V_GT)
high-V_GT / low-V_GT region MAE
output gradient magnitude
edge/texture region SCC 및 spectral-angle error
V_S collapse 여부(std≈0)
```

GTVar loss가 낮아져도 ERGAS/HQNR이 악화하거나 edge overshoot가 생기면 실패다. Local variance 일치는 방향 없는 texture energy만 맞춰도 낮아질 수 있으므로 정성 영상과 gradient/SAM을 함께 본다.

---

## 10. 이번 queue에서 제외하는 것

```text
PAN reconstruction / dual MARs
student predictive-uncertainty head와 V_GT의 pixel equality
feature KD
spectral-only KD
SiS / edge loss
GT variance hard weighting
mutual learning
teacher 교체
```

한 번에 바꾸는 것은 `lambda_U` 또는 `L_GTVar` 하나뿐이다. GT variance hard weighting은 이번 direct comparison이 끝난 뒤 별도 ablation으로 남긴다.

---

## 11. 결과 표 템플릿

| Case | λU | λV | Seed | ERGAS↓ | SAM↓ | SCC↑ | Q2n↑ | HQNR↑ | Dλ↓ | Ds↓ | V corr↑ | Train |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MS2 no-KD | 0 | 0 | 2025 |  |  |  |  |  |  |  | — |  |
| MS2 no-KD | 0 | 0 | 1234 |  |  |  |  |  |  |  | — |  |
| Plain KD | 0.10 | 0 | 2025 |  |  |  |  |  |  |  | — |  |
| Plain KD | 0.10 | 0 | 1234 |  |  |  |  |  |  |  | — |  |
| UKD | 0.03 | 0 | 2025 |  |  |  |  |  |  |  | — |  |
| UKD | 0.03 | 0 | 1234 |  |  |  |  |  |  |  | — |  |
| UKD | 0.10 | 0 | 2025 |  |  |  |  |  |  |  | — |  |
| UKD | 0.10 | 0 | 1234 |  |  |  |  |  |  |  | — |  |
| UKD | 0.30 | 0 | 2025 |  |  |  |  |  |  |  | — |  |
| UKD | 0.30 | 0 | 1234 |  |  |  |  |  |  |  | — |  |
| UKD+GTVar | best | 0.10 | 2025 |  |  |  |  |  |  |  |  |  |
| UKD+GTVar | best | 0.10 | 1234 |  |  |  |  |  |  |  |  |  |
