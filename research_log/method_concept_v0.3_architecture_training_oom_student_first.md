# Variance-Regularized Mutual Overfitting for PAN-Sharpening

> **Method concept v0.3 — Recommended Teacher–Student Architecture, Student-First Validation, Joint Training, and 24 GB OOM Plan**  
> 기준 자료: *U-Know-DiffPAN*, Sec. 3.1–3.4, Fig. 2–3, Supp. A.4/Fig. 9, Table 4

---

## 0. 최종 권장안

본 연구의 **첫 번째 주 모델**은 다음 조합으로 시작하는 것이 가장 안전하다.

```text
Teacher: FSA-T-Lite-UQ
- diffusion model
- frequency/conditional U-Net denoiser
- FFA + 저해상도 HQFE
- residual mean + log-variance 출력

Student: FSA-S-UQ
- diffusion model
- plain ResBlock U-Net denoiser
- 추가 frequency conditioning 없음
- residual mean + log-variance 출력
```

핵심 판단은 다음과 같다.

1. **원 U-Know-DiffPAN의 Teacher와 Student는 모두 diffusion model이다.**
2. Student의 denoiser가 plain ResBlock 기반 CNN U-Net인 것은 맞지만, Student 전체가 일반적인 single-pass CNN인 것은 아니다.
3. 원 FSA-S는 noisy residual `X_t`와 timestep `t`를 입력받고, DDIM sampling을 반복하는 diffusion student다.
4. 본 연구의 Student는 원 FSA-S를 기반으로 하되 **variance head와 mutual-learning interface를 추가한 FSA-S-UQ**로 정의한다.
5. 24 GB에서 가장 중요한 OOM 대책은 Student를 극단적으로 줄이는 것이 아니라, **두 모델의 backward graph를 동시에 유지하지 않는 alternating one-stage mutual update**다.
6. Student를 먼저 단독으로 구현·학습·프로파일링하는 것은 권장한다. 다만 해당 checkpoint를 본 학습에 재사용하면 엄밀한 의미의 `from-scratch one-stage`가 아니므로, main setting과 warm-start ablation을 구분한다.

---

# Part I. 먼저 바로잡아야 할 개념

## 1. “Teacher는 diffusion, Student는 CNN”이라는 표현

이 표현은 **backbone 수준에서는 일부 맞지만, 학습 프레임워크 수준에서는 틀리다.**

### 1.1 Diffusion과 CNN은 서로 배타적인 분류가 아니다

- **Diffusion**: 데이터를 노이즈화하고, timestep별 denoising을 학습하며, 추론 시 반복 sampling하는 학습·생성 프레임워크
- **CNN/U-Net**: 각 timestep에서 denoising function을 구현하는 신경망 backbone

즉, diffusion model의 denoiser가 CNN U-Net일 수 있다.

원 U-Know-DiffPAN은 다음 구조다.

```text
FSA-T = Diffusion process + frequency-aware CNN/U-Net denoiser
FSA-S = Diffusion process + plain ResBlock CNN/U-Net denoiser
```

### 1.2 원 FSA-S가 일반 CNN student가 아닌 이유

원 논문의 FSA-S는 다음을 수행한다.


a. HRMS residual target을 정의한다.

\[
X_0=I_{MS}^{HR}-I_{MS}^{LR}
\]

b. 임의 timestep에서 noisy residual을 생성한다.

\[
X_t=\sqrt{\bar\alpha_t}X_0+\sqrt{1-\bar\alpha_t}\epsilon
\]

c. Student가 다음 입력으로 clean residual을 예측한다.

\[
\tilde X_0=\psi(X_t,I_{PAN},I_{MS}^{LR},t)
\]

d. 추론 시 한 번의 CNN forward로 끝나는 것이 아니라, 원 논문 설정에서는 DDIM 25-step sampling을 수행한다.

따라서 FSA-S는 **plain ResBlock을 사용한 diffusion denoiser**이지, `PAN + LRMS → HRMS`를 한 번에 계산하는 deterministic CNN은 아니다.

### 1.3 세 모델 유형 비교

| 구분 | Teacher FSA-T | 원 Student FSA-S | 일반 single-pass CNN Student |
|---|---|---|---|
| Diffusion forward/noise | 사용 | 사용 | 사용하지 않음 |
| 입력 `X_t` | 사용 | 사용 | 사용하지 않음 |
| timestep `t` | 사용 | 사용 | 사용하지 않음 |
| denoiser backbone | U-Net + FFA/HQFE | plain ResBlock U-Net | residual CNN 등 |
| 반복 sampling | 사용 | 사용 | 없음 |
| inference forward 횟수 | 다회 | 다회 | 1회 |
| U-Know 원 논문 모델인가 | 예 | 예 | 아니오 |

---

# Part II. 원 FSA-S와 본 연구 Student의 차이

## 2. 원 U-Know-DiffPAN의 FSA-S

원 논문에서 FSA-S는 다음 특징을 갖는다.

- FSA-T와 동일한 수의 encoder/decoder scale 사용
- 각 block은 ResBlock만 사용
- FFA, FTCA, SWTCA 및 prior conditioning 제거
- PAN, LRMS, noisy residual `X_t`, timestep `t` 사용
- clean residual `X_0` 예측
- Student 자체 variance는 출력하지 않음
- pretrained/frozen Teacher의 output, uncertainty, feature를 이용해 학습
- two-stage KD 구조

원 논문 Table 4의 규모는 다음과 같다.

| 모델 | Parameters | FLOPs | Time | 보고 Memory |
|---|---:|---:|---:|---:|
| FSA-T | 25.492M | 1.402T | 25.495 s | 5.910 GB |
| FSA-S | 9.115M | 0.346T | 12.287 s | 2.136 GB |

Student는 Teacher의 약 35.8% parameter와 24.7% FLOPs를 사용한다.

> 주의: 위 memory는 원 논문의 complexity benchmark 값이며, **두 모델을 동시에 backward하는 joint-training peak memory가 아니다.**

## 3. 본 연구의 FSA-S-UQ

본 연구의 초기 Student는 원 FSA-S를 최대한 보존하고 다음만 추가한다.

```text
원 FSA-S
+ Student log-variance head
+ GT-variance-weighted individual loss
+ T→S/S→T mutual interface
+ 필요한 경우 저해상도 feature tap 1–2개
= FSA-S-UQ
```

### 3.1 원 FSA-S 대비 변경점

| 항목 | 원 FSA-S | 제안 FSA-S-UQ |
|---|---|---|
| Diffusion student | 예 | 예 |
| U-Net macro topology | 유지 | 유지 |
| ResBlock-only body | 예 | 예 |
| Teacher additional condition | 없음 | 없음 |
| 출력 mean/residual | 예 | 예 |
| Student uncertainty | 없음 | **추가** |
| 학습 방식 | frozen Teacher KD | **online mutual learning** |
| S→T feedback | 없음 | **제한적으로 추가** |
| GT-derived variance | 없음 | **importance prior로 추가** |

### 3.2 Variance head 권장

원 FSA-T는 bandwise uncertainty를 출력한다. 본 연구의 routing 안정성을 위해 첫 구현은 다음을 권장한다.

```text
Mean head    : C_MS channels
Variance head: 1 spatial channel
```

\[
s_S\in\mathbb R^{1\times H\times W},\qquad
\sigma_S^2=\exp(s_S)
\]

1-channel log-variance는 모든 spectral band에 broadcast한다.

장점:

- mutual routing map이 단순함
- variance 자유도가 작아 error를 variance로 우회하는 현상을 줄임
- parameter와 VRAM 증가가 사실상 미미함

원 논문과의 직접적인 호환성을 위해 `C_MS-channel variance`도 ablation으로 유지한다.

---

# Part III. 권장 Teacher–Student 아키텍처

## 4. 공통 macro topology

공식 구현 기준으로 다음 topology를 baseline으로 사용한다.

```text
PAN patch resolution: 64 × 64
Scales              : 64 → 32 → 16 → 8
Base width           : 32
Channel schedule     : 32 → 64 → 64 → 128
Bottleneck attention : 8 × 8
Skip connections     : 동일 위치
Prediction target    : X0 residual
```

> 논문 schematic은 `C → 2C → 4C → 8C`로 표현되지만, 공개 구현은 `inner_channel=32`, `channel_mults=(1,2,2,4)`를 사용한다. 재현성과 24 GB 제약을 위해 공개 구현을 우선 기준으로 삼는다.

## 5. Teacher: FSA-T-Lite-UQ

### 5.1 기본 구조

| Stage | Resolution | Channel | Block | 추가 모듈 |
|---|---:|---:|---|---|
| Stem | 64×64 | 32 | Conv | PAN/LRMS concat |
| E1 | 64×64 | 32 | ResBlock ×3 | FFA는 생략 또는 1회 |
| E2 | 32×32 | 64 | ResBlock ×3 | FFA |
| E3 | 16×16 | 64 | ResBlock ×3 | FFA |
| Bottleneck | 8×8 | 128 | ResBlock + SA | global/frequency modeling |
| D3 | 16×16 | 64 | ResBlock ×3 | FTCA + SWTCA |
| D2 | 32×32 | 64 | ResBlock ×3 | 선택적 HQFE 1회 |
| D1 | 64×64 | 32 | ResBlock ×3 | HQFE 사용하지 않음 |
| Output | 64×64 | `C_MS + 1` | 2 heads | mean + log-variance |

### 5.2 원 FSA-T와 다른 점

원 FSA-T는 각 decoder level의 HQFE와 별도 prior network conditioning을 적극적으로 사용한다. Joint training용 Lite 버전에서는 다음을 조정한다.

- full-resolution HQFE 제거
- HQFE는 16×16과 필요 시 32×32에만 배치
- prior network는 frozen + `no_grad`, 또는 prior output을 사전 계산
- all-stage feature 저장 금지
- uncertainty head를 log-variance 형태로 정리

이 변경은 source-faithful reproduction이 아니라 **joint-training memory를 위한 FSA-T-Lite**다. 따라서 실험에서는 full FSA-T Teacher-only upper bound와 구분한다.

## 6. Student: FSA-S-UQ

### 6.1 기본 구조

| Stage | Resolution | Channel | Block | 추가 모듈 |
|---|---:|---:|---|---|
| Stem | 64×64 | 32 | Conv | PAN/LRMS concat |
| E1 | 64×64 | 32 | ResBlock ×2 | 없음 |
| E2 | 32×32 | 64 | ResBlock ×2 | 없음 |
| E3 | 16×16 | 64 | ResBlock ×2 | 없음 |
| Bottleneck | 8×8 | 128 | ResBlock + SA | attention 1회 |
| D3 | 16×16 | 64 | ResBlock ×2 | 없음 |
| D2 | 32×32 | 64 | ResBlock ×2 | 없음 |
| D1 | 64×64 | 32 | ResBlock ×2 | 없음 |
| Output | 64×64 | `C_MS + 1` | 2 heads | mean + log-variance |

### 6.2 왜 Student width를 바로 1/2로 줄이지 않는가

Mutual learning에서는 Student가 Teacher에게도 유효한 correction을 제공해야 한다. 따라서 첫 구현은 다음을 유지한다.

```text
Teacher channels: [32, 64, 64, 128]
Student channels: [32, 64, 64, 128]
```

차이는 width가 아니라 다음에서 만든다.

- block depth: 3 vs 2
- frequency conditioning 유무
- prior conditioning 유무
- HQFE 유무

OOM이 확인된 뒤에만 Student를 다음처럼 축소한다.

```text
S-small channels: [24, 48, 48, 96]
```

### 6.3 권장 feature tap

중간 feature mutual loss는 다음 두 위치만 사용한다.

```text
16 × 16 feature
 8 ×  8 bottleneck feature
```

full-resolution feature distillation은 activation retention 비용이 크므로 초기 실험에서는 사용하지 않는다.

---

# Part IV. Student를 먼저 돌리는 실험

## 7. Student-first는 두 가지 의미로 구분해야 한다

### 7.1 구현 및 검증 순서를 Student부터 시작

권장한다.

- Student architecture가 정상적으로 학습되는지 확인
- variance head가 collapse하지 않는지 확인
- 64×64 patch에서 peak VRAM 측정
- Student 단독 성능 확보
- Teacher 없이도 GT overfitting이 가능한지 확인

이 경우 main joint training은 다시 random initialization으로 시작할 수 있으므로 `one-stage from scratch` 주장을 유지할 수 있다.

### 7.2 Student를 pretrain한 후 그 checkpoint로 joint training

가능하지만 논문 포지셔닝이 달라진다.

```text
Student pretraining → Teacher–Student joint training
```

이는 single-run from-scratch one-stage가 아니라 **student-warm-started joint training**이다.

따라서 다음처럼 구분한다.

| 설정 | 논문에서의 역할 |
|---|---|
| Joint from scratch | main one-stage setting |
| Student warm-start | optimization/stability ablation |

## 8. Student-only 학습 objective

Student 단독 단계에서는 Teacher 관련 loss를 모두 제거한다.

\[
\mathcal L_{S\text{-only}}
=
\mathcal L_{rec}^{S}
+
\lambda_{nll}\mathcal L_{NLL}^{S}
+
\lambda_{cal}\mathcal L_{cal}^{S}
\]

GT-derived variance를 이용한 reconstruction weight는 다음과 같이 둔다.

\[
w_V=1+\gamma V_{GT}
\]

\[
\mathcal L_{rec}^{S}
=
\frac1N\sum_{p,c}
w_V(p)\rho(\mu_S(p,c)-X_0(p,c))
\]

초기 검증 순서:

1. mean head만 학습
2. variance head 추가
3. GT-derived variance weighting 추가
4. uncertainty–error correlation 확인
5. diffusion sampling 25-step 성능과 추론시간 확인

## 9. Student-first 권장 실행 순서

```text
S0. 100–500 iterations smoke test
    - shape, NaN, variance range, loss 확인

S1. Student-only baseline
    - FSA-S-UQ
    - GT individual loss만 사용
    - peak VRAM 기록

S2. Student-only ablation
    - mean only
    - mean + NLL
    - mean + NLL + GT variance

S3. Teacher-only profile
    - full FSA-T
    - FSA-T-Lite

S4. Joint output-only mutual
    - feature mutual 제외

S5. Low-resolution feature mutual 추가
```

---

# Part V. 전체 one-stage joint training

## 10. 공통 noisy input을 사용해야 하는 이유

두 diffusion network가 같은 위치와 난이도를 비교하려면 동일한 timestep과 동일한 noise realization을 사용한다.

```text
same X0
same timestep t
same Gaussian noise ε
same Xt
```

\[
X_t=\sqrt{\bar\alpha_t}X_0+\sqrt{1-\bar\alpha_t}\epsilon
\]

Teacher와 Student가 서로 다른 `t` 또는 `ε`를 보면 prediction difficulty가 달라져 uncertainty와 local error를 직접 비교하기 어렵다.

## 11. 모델 출력

\[
(\mu_T,s_T,F_T)=T(X_t,I_{PAN},I_{MS}^{LR},t)
\]

\[
(\mu_S,s_S,F_S)=S(X_t,I_{PAN},I_{MS}^{LR},t)
\]

- `μ`: clean residual prediction
- `s`: log-variance
- `F`: 16×16, 8×8 feature만 선택

## 12. Individual loss

\[
\mathcal L_{ind}^{m}
=
\mathcal L_{rec}^{m}
+
\lambda_{nll}\mathcal L_{NLL}^{m}
+
\lambda_{cal}\mathcal L_{cal}^{m},
\qquad m\in\{T,S\}
\]

두 모델 모두 항상 GT anchor를 갖는다. Mutual target만으로 학습하지 않는다.

## 13. Reliability-gated mutual loss

local empirical error를 다음과 같이 계산한다.

\[
E_m=A_k\left(\frac1C\sum_c(\mu_m-X_0)^2\right)
\]

Teacher가 더 정확하고 더 확실할 때만 T→S를 활성화한다.

\[
g_{T\rightarrow S}
=
\operatorname{sg}
\left[
\sigma\left(\frac{E_S-E_T}{\tau_e}\right)
\sigma\left(\frac{s_S-s_T}{\tau_s}\right)
\right]
\]

반대 방향도 동일하게 정의한다.

\[
g_{S\rightarrow T}
=
\operatorname{sg}
\left[
\sigma\left(\frac{E_T-E_S}{\tau_e}\right)
\sigma\left(\frac{s_T-s_S}{\tau_s}\right)
\right]
\]

Mutual prediction loss:

\[
\mathcal L_{T\rightarrow S}
=
\frac{\sum w_Vg_{T\rightarrow S}\rho(\mu_S-\operatorname{sg}(\mu_T))}
{\sum w_Vg_{T\rightarrow S}+\epsilon}
\]

\[
\mathcal L_{S\rightarrow T}
=
\frac{\sum w_Vg_{S\rightarrow T}\rho(\mu_T-\operatorname{sg}(\mu_S))}
{\sum w_Vg_{S\rightarrow T}+\epsilon}
\]

Teacher를 Student 수준으로 끌어내리는 것을 막기 위해 다음을 사용한다.

\[
0<\alpha<1
\]

초기값:

```text
alpha = 0.25
```

## 14. 전체 objective

\[
\mathcal L_{total}
=
\mathcal L_{ind}^{T}
+
\mathcal L_{ind}^{S}
+
\lambda_{mut}(t)
\left(
\mathcal L_{T\rightarrow S}
+
\alpha\mathcal L_{S\rightarrow T}
\right)
+
\lambda_f\mathcal L_{feat}
\]

초기에는 `λ_f=0`으로 두고 output-level mutual이 안정된 후 추가한다.

## 15. Training schedule

| 진행률 | 학습 내용 |
|---:|---|
| 0–10% | Individual GT/NLL만 학습 |
| 10–30% | Mutual weight 선형 증가 |
| 30–100% | Reliability-gated bidirectional mutual |

이 schedule은 별도 Teacher checkpoint를 사용하지 않으므로 one-stage로 볼 수 있다.

---

# Part VI. 24 GB OOM 분석

## 16. 가장 위험한 memory source

joint training peak memory의 주요 원인은 parameter 수보다 activation이다.

1. Teacher와 Student의 backward graph 동시 보존
2. Teacher HQFE의 FFT/SWT/cross-attention intermediate
3. U-Net skip feature 유지
4. 여러 stage feature distillation을 위한 activation retention
5. prior network와 prior output
6. 두 optimizer의 Adam state
7. EMA model 두 개 유지
8. 큰 physical batch

원 논문의 FSA-T 5.910 GB와 FSA-S 2.136 GB를 단순 합산해 joint training이 8 GB라고 판단하면 안 된다.

- 해당 값은 training backward peak가 아님
- mutual loss는 두 graph를 동시에 유지할 수 있음
- optimizer state와 gradient가 추가됨
- feature distillation은 intermediate tensor 생존 시간을 늘림

## 17. OOM 위험도

| 설정 | 예상 위험 | 설명 |
|---|---|---|
| Student-only FSA-S-UQ | 낮음 | 가장 먼저 profiling |
| Teacher-only full FSA-T | 중간 | HQFE/prior 영향 큼 |
| Teacher-only FSA-T-Lite | 낮음–중간 | low-resolution HQFE |
| Strict simultaneous joint | 높음 | 두 backward graph 동시 유지 |
| Alternating joint | **중간 이하** | 한 번에 하나의 train graph |
| Joint + all-stage feature KD | 매우 높음 | 초기에는 금지 |

수치는 구현에 따라 달라지므로 반드시 실제 peak memory를 측정한다.

## 18. 권장 시작 설정

```text
GPU                 : RTX 4090 24 GB
PAN patch           : 64 × 64
Precision           : BF16 또는 FP16 autocast
Student-only batch  : 8부터 시작
Teacher-only batch  : 4부터 시작
Strict joint batch  : 1부터 시작
Alternating joint   : 2부터 시작
Effective batch     : gradient accumulation으로 16–32
Feature mutual      : OFF
EMA                 : Student만 우선 사용
```

위 값은 보장값이 아니라 안전한 profiling 시작점이다.

## 19. 가장 효과적인 OOM 대책: alternating one-stage update

### 19.1 Strict simultaneous update

```text
Teacher forward with grad
Student forward with grad
combined loss
single backward
```

장점:

- 가장 직관적인 joint optimization
- 같은 iteration에서 완전 동시 업데이트

단점:

- 두 모델 activation graph가 동시에 살아 있음
- OOM 가능성이 가장 큼

### 19.2 Memory-safe alternating update

같은 minibatch, 같은 `X_t`, 같은 `t`를 사용하되 한 방향씩 업데이트한다.

#### Step A: Student update

```text
Teacher forward: no_grad, detached target
Student forward: with grad
Student individual loss + T→S mutual
backward Student only
optimizer_S.step()
```

#### Step B: Teacher update

```text
Student forward: no_grad, detached target
Teacher forward: with grad
Teacher individual loss + α·S→T mutual
backward Teacher only
optimizer_T.step()
```

장점:

- peak 시점에 하나의 train graph만 유지
- 24 GB에서 가장 현실적
- Teacher pretraining 없이 동일 training run에서 상호학습

단점:

- forward 계산량 증가
- 두 방향 update가 완전히 동시적이지 않음
- peer target이 반 step 차이를 가짐

본 연구의 main implementation으로는 이 방식을 권장한다.

## 20. Alternating training pseudocode

```python
# Same batch, same timestep, same noise
x0 = gt - lms
t = sample_timestep(batch_size)
eps = torch.randn_like(x0)
xt = q_sample(x0, t, eps)
v_gt = compute_gt_detail_variance(x0).detach()

# -------------------------------------------------
# A. Update Student: Teacher is a detached peer
# -------------------------------------------------
optimizer_s.zero_grad(set_to_none=True)

with torch.no_grad():
    t_mean, t_logvar, t_feat = teacher(xt, pan, lms, t)

s_mean, s_logvar, s_feat = student(xt, pan, lms, t)
loss_s = student_individual_loss(
    s_mean, s_logvar, x0, v_gt
)
loss_s += lambda_ts * mutual_t_to_s(
    t_mean.detach(), t_logvar.detach(),
    s_mean, s_logvar, x0, v_gt
)

loss_s.backward()
optimizer_s.step()

# -------------------------------------------------
# B. Update Teacher: Student is a detached peer
# -------------------------------------------------
optimizer_t.zero_grad(set_to_none=True)

with torch.no_grad():
    s_mean_ref, s_logvar_ref, _ = student(xt, pan, lms, t)

t_mean, t_logvar, t_feat = teacher(xt, pan, lms, t)
loss_t = teacher_individual_loss(
    t_mean, t_logvar, x0, v_gt
)
loss_t += alpha * lambda_st * mutual_s_to_t(
    s_mean_ref.detach(), s_logvar_ref.detach(),
    t_mean, t_logvar, x0, v_gt
)

loss_t.backward()
optimizer_t.step()
```

중요 사항:

- peer output에 반드시 `detach`/`no_grad`
- `retain_graph=True` 사용하지 않음
- 두 방향에서 같은 `X_t`, `t`, `ε` 사용
- mutual gate 자체도 stop-gradient

## 21. OOM 대책 우선순위

OOM이 발생하면 다음 순서로 조정한다.

### Priority 1 — Graph 구조

1. strict simultaneous update를 alternating update로 변경
2. all-stage feature loss 제거
3. peer output을 반드시 detach
4. `retain_graph=True` 제거

### Priority 2 — Batch/precision

5. physical batch 감소
6. gradient accumulation 증가
7. BF16/FP16 autocast 적용

> Gradient accumulation 자체는 한 microbatch의 peak memory를 줄이지 않는다. physical batch를 줄인 뒤 effective batch를 회복하는 용도다.

### Priority 3 — Activation

8. Teacher HQFE와 bottleneck에 activation checkpointing
9. HQFE를 16×16/8×8에만 사용
10. feature tap을 8×8 하나로 축소
11. SWT 및 GT variance를 `no_grad`로 계산
12. frozen prior network를 `eval + no_grad`로 실행

### Priority 4 — Model 축소

13. Student width `[32,64,64,128] → [24,48,48,96]`
14. Student bottleneck attention 제거
15. Teacher 32×32 HQFE 제거
16. 마지막으로 Teacher width 축소

Student를 너무 먼저 축소하면 S→T mutual signal이 약해질 수 있으므로 graph와 activation을 먼저 줄인다.

## 22. 추가 memory 관리 원칙

- `optimizer.zero_grad(set_to_none=True)` 사용
- validation은 training graph가 완전히 해제된 뒤 실행
- logging용 feature를 GPU list에 누적하지 않음
- visualization tensor는 즉시 CPU로 이동
- EMA는 처음에는 Student만 유지
- `torch.cuda.empty_cache()`는 살아 있는 tensor를 해제하지 못하므로 근본 대책으로 사용하지 않음
- FFT/SWT output을 모든 scale에서 복제하지 않음
- condition은 필요한 scale로 한 번만 resize

## 23. Peak memory profiling

```python
import torch

@torch.no_grad()
def report_memory(tag: str) -> None:
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[{tag}] allocated={allocated:.2f} GB, "
        f"reserved={reserved:.2f} GB, peak={peak:.2f} GB"
    )

# Before each isolated profiling run
torch.cuda.reset_peak_memory_stats()
```

각 설정에서 다음을 따로 기록한다.

```text
S-only forward
S-only forward + backward
T-only forward
T-only forward + backward
strict joint forward + backward
alternating Student step
alternating Teacher step
```

목표 peak는 driver/display fluctuation을 감안해 **21–22 GB 이하**로 둔다.

---

# Part VII. 정말 deterministic CNN Student를 쓰고 싶은 경우

## 24. 이것은 원 FSA-S와 다른 별도 방법이다

Teacher는 diffusion이고 Student는 single-pass CNN인 hybrid도 가능하다.

```text
Diffusion Teacher:
(Xt, PAN, LRMS, t) → X0 prediction + variance

Deterministic CNN Student:
(PAN, LRMS) → X0 prediction + variance
```

이 Student는 다음을 제거한다.

- `X_t`
- timestep embedding
- forward diffusion
- DDIM sampling
- iterative denoising

따라서 이름을 FSA-S로 유지하기보다 다음처럼 별도 명명하는 것이 정확하다.

```text
Det-S
CNN-S
OnePass-S
```

## 25. Hybrid의 장단점

| 장점 | 단점 |
|---|---|
| Student memory와 inference가 매우 작음 | 원 U-Know architecture pair와 달라짐 |
| Teacher–Student inductive bias가 명확히 다름 | feature alignment가 어려움 |
| Student-first 구현이 쉬움 | Student uncertainty와 Teacher uncertainty의 조건이 다름 |
| deployment 효율성이 큼 | S→T가 Teacher를 저하할 위험 |
| joint OOM 위험 감소 | method가 더 복잡해짐 |

## 26. Hybrid에서 mutual loss를 적용하는 방법

서로 입력 상태가 다르므로 intermediate feature matching은 피하고, clean residual output 공간에서만 정렬한다.

```text
사용 권장:
- output mean mutual
- variance/calibration
- wavelet/high-frequency output mutual

사용 비권장:
- all-stage feature matching
- timestep-dependent feature equality
```

Teacher가 random timestep에서 예측한 `X0`와 CNN Student의 clean residual을 비교한다.

\[
\mathcal L_{T\rightarrow CNN}
=
\rho(\mu_{CNN}-\operatorname{sg}(\mu_T))
\]

S→T는 실제 Student error가 Teacher보다 낮은 위치에서만 아주 약하게 사용한다.

```text
alpha ≤ 0.1–0.25
```

## 27. Hybrid Student 구조 후보

검증된 PAN-sharpening residual CNN의 원칙을 따르는 최소 구조:

```text
Input: upsampled LRMS + PAN + optional PAN gradient
Stem: 3×3 Conv, 48–64 channels
Body: 8–12 residual blocks, no BatchNorm
Fusion: residual/detail feature concatenation
Mean head: C_MS-channel residual
Variance head: 1-channel log-variance
Output: LRMS + predicted residual
```

다만 이 구조는 **원 U-Know FSA-S의 재현이 아니며**, main method로 채택하려면 dual-diffusion baseline과 반드시 비교해야 한다.

---

# Part VIII. 필수 실험 매트릭스

## 28. Architecture 및 training 비교

| ID | Teacher | Student | Training | 목적 |
|---|---|---|---|---|
| A | 없음 | 원 FSA-S-like | Student-only diffusion | Student baseline |
| B | 없음 | FSA-S-UQ | Student-only diffusion | variance 효과 |
| C | FSA-T-Lite | FSA-S-UQ | strict joint | 이상적 joint |
| D | FSA-T-Lite | FSA-S-UQ | alternating joint | 24 GB main |
| E | FSA-T-Lite | Det-S | alternating hybrid | one-pass 대안 |
| F | full FSA-T | 없음 | Teacher-only | upper bound |

## 29. OOM/efficiency 보고 항목

각 모델 및 학습 모드에 대해 다음을 보고한다.

- parameter 수
- training peak VRAM
- physical/effective batch
- iteration time
- forward 횟수
- inference DDIM step 수
- Student inference latency
- Teacher/Student individual performance
- joint 이후 각 모델 성능

## 30. Mutual learning의 필요성 검증

\[
P(E_T<E_S),\qquad P(E_S<E_T)
\]

\[
E_{oracle}(p)=\min(E_T(p),E_S(p))
\]

Student가 Teacher보다 정확한 위치가 거의 없으면 S→T mutual의 근거가 약하다.

Uncertainty routing 정확도:

\[
Acc_{route}
=
P\left(
\arg\min_m s_m
=
\arg\min_m E_m
\right)
\]

---

# Part IX. 최종 실행 권고

## 31. 권장 개발 순서

```text
1. FSA-S-UQ Student-only 구현
2. Student memory/성능/variance 검증
3. full FSA-T Teacher-only 재현 또는 profile
4. FSA-T-Lite 구축
5. output-only alternating mutual learning
6. GT variance 및 uncertainty gate 검증
7. 8×8 feature mutual 추가
8. 필요 시 16×16 feature mutual 추가
9. 마지막으로 deterministic CNN Student hybrid 비교
```

## 32. Main method로 권장하는 조합

```text
FSA-T-Lite-UQ + FSA-S-UQ
same Xt / t / noise
alternating one-stage mutual update
output-level mutual first
low-resolution feature mutual second
```

이 조합의 장점:

- U-Know-DiffPAN의 검증된 Teacher–Student pair를 가장 많이 보존
- Teacher와 Student의 output 조건이 동일
- variance와 local error를 직접 비교 가능
- 24 GB에서는 alternating update로 peak memory를 낮출 수 있음
- Student가 충분한 capacity를 유지해 S→T correction 가능

## 33. 가장 중요한 정리

> **원 U-Know-DiffPAN의 Student는 일반 single-pass CNN이 아니라 diffusion student다.**

> **본 연구의 권장 Student는 원 FSA-S backbone을 유지하면서 variance head와 online mutual-learning 기능을 추가한 FSA-S-UQ다.**

> **Student를 먼저 단독으로 돌리는 것은 좋은 개발 순서다. 다만 그 checkpoint를 main joint training에 재사용할 경우 one-stage from-scratch와 구분해야 한다.**

> **24 GB에서의 핵심 OOM 대책은 모델을 무조건 작게 만드는 것이 아니라, peer를 `no_grad`로 계산하는 alternating one-stage mutual update를 사용하는 것이다.**

---

## Reference basis

- *U-Know-DiffPAN: An Uncertainty-aware Knowledge Distillation Diffusion Framework with Details Enhancement for PAN-Sharpening*
  - Sec. 3.1: two-stage framework and residual prediction
  - Sec. 3.2: diffusion process of FSA-T
  - Sec. 3.3: FFA, FTCA, SWTCA
  - Sec. 3.4: FSA-S and U-Know loss
  - Supplementary A.4/Fig. 9: FSA-S is diffusion-based and ResBlock-only
  - Table 4: FSA-T/FSA-S computational complexity
