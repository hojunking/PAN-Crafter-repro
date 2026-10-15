# PAN-Crafter-Lite Student-First Experimental Strategy

## 0. 문서 목적

이 문서는 PAN-Crafter를 Teacher 후보로 두고, 그 핵심 구조를 유지한 경량 Student를 먼저 독립적으로 학습·평가하기 위한 실험 전략을 정리한다.

현재 단계의 목표는 mutual learning이나 uncertainty loss를 바로 적용하는 것이 아니다. 먼저 Student 자체가 다음 조건을 만족하는지 확인하는 것이 우선이다.

1. 원 PAN-Crafter의 핵심 성능 원인인 **MARs**와 **CM3A**를 유지하는가?
2. Teacher 대비 충분히 작고 빠른가?
3. 크기 감소에 비해 성능 손실이 허용 가능한가?
4. 이후 Teacher–Student 상호학습에 참여할 만큼 독립적인 복원 능력이 있는가?

---

# 1. PAN-Crafter 기준 모델

## 1.1 원 논문 기본 구조

PAN-Crafter는 diffusion sampling을 사용하지 않는 **single-pass U-Net 기반 PAN-sharpening 네트워크**다.

공개 논문과 구현의 기본 설정은 다음과 같다.

```yaml
Teacher_candidate: PAN-Crafter

hidden_size: [128, 128, 128, 128]
s_embed_size: 128
depth: [2, 2, 2, 2]
num_heads: 8
local_attention_window: 3

CM3A_locations:
  - encoder H/2
  - encoder H/4
  - bottleneck H/8
  - decoder H/4
  - decoder H/2

MARs:
  MS_mode: true
  PAN_mode: true
```

논문이 보고한 계산량은 다음과 같다.

| 항목 | PAN-Crafter |
|---|---:|
| Parameters | 7.17 M |
| FLOPs | 79.03 G |
| Inference time | 0.009 s |
| Inference memory | 1,751.9 MB |

측정 대상은 reduced-resolution의 `256 × 256 × 8` HRMS 출력이다.

---

## 1.2 핵심 효율성 원인

PAN-Crafter가 효율적인 이유는 다음과 같다.

### Single-pass reconstruction

Diffusion처럼 여러 번 denoising하지 않고 한 번의 forward로 HRMS를 생성한다.

### Local attention

CM3A는 전체 영상의 모든 위치를 비교하는 global attention이 아니라 각 pixel 주변의 `3 × 3` local window만 탐색한다.

\[
O\left(2(HW)^2C\right)
\quad\rightarrow\quad
O\left(2(HW)k^2C\right)
\]

여기서 \(k=3\)이다.

### 저·중해상도 중심의 attention 배치

고해상도 stage에는 ResBlock만 사용하고, 상대적으로 비싼 CM3A는 `H/2`, `H/4`, `H/8`에서 수행한다.

### MARs의 weight sharing

HRMS reconstruction과 PAN back-reconstruction을 서로 다른 네트워크로 처리하지 않는다. 하나의 backbone을 공유하고 mode-dependent parameter만 사용한다.

### Residual reconstruction

MS mode와 PAN mode 모두 저주파 baseline에 대한 고주파 residual을 예측한다. 따라서 전체 영상을 처음부터 생성하지 않고 보정 성분에 집중한다.

---

# 2. MARs의 두 mode는 무엇을 학습하는가?

## 2.1 두 mode의 입력은 동일하다

MS mode와 PAN mode 모두 기본 입력은 다음 두 영상이다.

\[
I_{PAN},\qquad I_{MS}^{LR}
\]

여기서 \(I_{MS}^{LR}\)는 원래 낮은 해상도의 MS를 PAN 크기로 4배 보간한 영상이다.

따라서 두 입력 tensor의 spatial size는 같지만, 정보의 성격은 다르다.

- PAN: 센서가 직접 획득한 고해상도 1-band 영상
- LRMS: 낮은 native resolution에서 획득된 multi-band 영상을 보간한 영상

즉, **같은 tensor 해상도는 같은 실제 공간 정보를 의미하지 않는다.**

---

## 2.2 mode별 차이

두 mode는 입력 영상 쌍은 같지만 다음 요소가 다르다.

- reconstruction target
- residual baseline
- CM3A query의 modality prior
- ResBlock의 mode modulation
- MS/PAN feature 결합 계수
- 최적화 목적

| 구분 | MS mode | PAN mode |
|---|---|---|
| 입력 | PAN + upsampled LRMS | PAN + upsampled LRMS |
| 최종 target | HRMS | multi-channel repeated PAN |
| Residual baseline | upsampled LRMS | low-pass PAN을 다시 확대·반복한 영상 |
| 주된 residual | HRMS의 누락된 spatial–spectral detail | PAN의 고주파 spatial detail |
| CM3A query prior | LRMS | PAN |
| Self-attention 기준 | MS feature | PAN feature |
| Cross-modality 정보 | PAN structure를 MS에 정렬 | MS texture를 PAN structure에 정렬 |
| 역할 | 최종 PAN-sharpening | 공간 고주파 보조 자기지도 |
| 추론 시 사용 | 사용 | 사용하지 않음 |

수식으로 보면 다음과 같다.

### MS mode

\[
\hat I_{MS}^{HR}
=
P_\theta(I_{PAN},I_{MS}^{LR};\text{MS})
+
I_{MS}^{LR}
\]

### PAN mode

\[
\hat I_{PAN}^{rep}
=
P_\theta(I_{PAN},I_{MS}^{LR};\text{PAN})
+
I_{PAN}^{LR,rep}
\]

PAN mode의 target은 원 PAN을 MS band 수만큼 반복한 영상이며, baseline은 PAN을 4배 축소한 뒤 다시 확대해 만든 low-frequency PAN이다. 따라서 PAN mode의 네트워크 출력은 사실상 PAN의 고주파 보정량이다.

---

## 2.3 정확한 해석

다음 표현은 대체로 맞지만 약간의 수정이 필요하다.

> MS mode가 spectral 정보를 학습하고 PAN mode가 spatial 정보를 학습한다.

더 정확한 표현은 다음과 같다.

> **MS mode는 최종 HRMS 복원을 담당하면서 LRMS의 spectral fidelity와 PAN의 spatial structure를 함께 학습한다. PAN mode는 PAN의 고주파 구조를 복원하는 auxiliary task로서 shared backbone에 spatial sharpness를 강화한다.**

즉, 두 mode가 각각 완전히 독립적으로 spectral과 spatial을 담당하는 것은 아니다.

- MS mode도 PAN 구조를 적극적으로 사용한다.
- PAN mode도 MS feature와 상호작용한다.
- 두 mode는 같은 backbone을 공유한다.
- mode-specific \(\alpha,\beta,\gamma\)가 동일 feature를 서로 다른 방식으로 조절한다.

핵심 장점은 **동일 입력을 두 개의 상보적인 reconstruction target으로 해석하도록 강제하는 것**이다.

```text
MS mode:
LRMS의 분광 특성을 유지하며 PAN detail을 주입

PAN mode:
PAN의 sharp spatial residual을 명시적으로 복원

Shared backbone:
두 task의 공통 표현을 공유
```

이 때문에 PAN mode는 별도 inference branch가 아니라 학습 중에만 사용하는 spatial-detail auxiliary supervision으로 이해하는 것이 적절하다.

---

# 3. Student 기본 전략

## 3.1 목표

Student는 Teacher의 절반보다 조금 작은 모델을 목표로 한다.

\[
\frac{P_S}{P_T}
\approx
0.45\sim0.48
\]

Teacher의 7.17M parameters를 기준으로 예상 Student 크기는 다음과 같다.

\[
P_S
\approx
3.2\sim3.5\text{M}
\]

이는 구현 전 추정치이며 실제 parameter count와 FLOPs는 반드시 코드로 측정해야 한다.

---

## 3.2 주 Student: PAN-Crafter-S96

```yaml
Student: PAN-Crafter-S96

hidden_size: [96, 96, 96, 96]
s_embed_size: 96
depth: [1, 1, 2, 1]
num_heads: 8
mlp_ratio: 4.0
local_attention_window: 3

CM3A_locations:
  - encoder H/2
  - encoder H/4
  - bottleneck H/8
  - decoder H/4
  - decoder H/2

MARs:
  MS_mode: true
  PAN_mode: true
```

### Stage별 해석

| Stage | Teacher | Student |
|---|---:|---:|
| Full-resolution encoder | 2 ResBlocks | 1 ResBlock |
| H/2 encoder | 2 ResBlocks | 1 ResBlock |
| H/4 encoder | 2 ResBlocks | 2 ResBlocks |
| H/8 bottleneck | 2 ResBlocks | 1 ResBlock |
| Decoder | 대응 stage와 동일한 축소 원칙 | 대응 stage와 동일한 축소 원칙 |
| Feature width | 128 | 96 |
| CM3A | 5개 | 5개 |
| MARs | 유지 | 유지 |

---

## 3.3 96-channel을 선택한 이유

Student width는 Teacher width의 75%다.

\[
\frac{96}{128}=0.75
\]

Convolution과 projection parameter가 대체로 channel width의 제곱에 비례하므로, width만 줄였을 때의 대략적인 비율은 다음과 같다.

\[
\left(\frac{96}{128}\right)^2
=
0.5625
\]

여기에 depth를 `[2,2,2,2]`에서 `[1,1,2,1]`로 줄이면 전체 parameter 비율이 약 45–48%까지 내려갈 것으로 예상한다.

96은 다음 조건도 만족한다.

- `96 / 8 heads = 12 channels per head`
- `96 / 32 GroupNorm groups = 3 channels per group`

따라서 원 PAN-Crafter의 head 수와 GroupNorm 설정을 크게 변경하지 않아도 된다.

---

# 4. Student에서 유지할 것과 줄일 것

## 4.1 유지할 요소

첫 Student 실험에서는 다음을 유지한다.

- MARs의 MS/PAN dual-mode training
- mode-dependent \(\alpha,\beta,\gamma\)
- 모든 5개 CM3A 위치
- local attention window \(k=3\)
- U-Net encoder–decoder topology
- skip connection
- residual reconstruction
- MS mode와 PAN mode의 동일한 loss 정의

PAN-Crafter ablation에서는 MARs가 큰 성능 향상을 제공했고, CM3A는 MARs와 결합했을 때 가장 강한 효과를 냈다. 따라서 첫 Student에서 이 두 구성 요소를 제거하면 경량화 효과와 핵심 mechanism 손실이 동시에 발생해 원인 분석이 어려워진다.

---

## 4.2 축소할 요소

첫 Student에서는 다음 두 요소만 축소한다.

1. Feature width: `128 → 96`
2. ResBlock depth: `[2,2,2,2] → [1,1,2,1]`

즉, 첫 실험의 목적은 다음과 같다.

> **PAN-Crafter의 inductive bias는 유지하고 순수 capacity만 줄였을 때 성능과 효율성이 어떻게 변하는지 측정한다.**

---

## 4.3 첫 실험에서 넣지 않을 요소

다음 요소는 Student 독립 성능을 확인한 뒤 추가한다.

- Teacher output distillation
- Mutual learning
- Teacher–Student feature matching
- uncertainty head
- NLL loss
- GT-derived variance
- CM3A pruning
- mode별 asymmetric loss
- Student-specific PAN gradient branch

이들을 한 번에 추가하면 Student architecture 자체의 기여를 분리할 수 없다.

---

# 5. Student-first 실험 순서

## Experiment 0 — Teacher baseline 확인

원 공개 설정으로 PAN-Crafter를 재현한다.

목적:

- 데이터 pipeline 검증
- metric 구현 검증
- training/inference time 기준 확보
- parameter/FLOPs counting 기준 확보
- 공개 성능과 재현 성능 간 차이 확인

Teacher 재학습 비용이 부담되면 공개 checkpoint를 이용해 inference 기준만 먼저 측정할 수 있다.

---

## Experiment 1 — Student-S96 독립 학습

주 Student를 원 PAN-Crafter와 동일한 MARs objective로 학습한다.

\[
\mathcal L_{\mathrm{MARs}}
=
\left\|
\hat I_{MS}^{HR}-I_{MS}^{HR}
\right\|_1
+
\lambda
\left\|
\hat I_{PAN}^{rep}-I_{PAN}^{rep}
\right\|_1
\]

초기 설정:

```yaml
lambda_pan: 1.0
local_window: 3
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 0.01
scheduler: cosine
warmup_steps: 100
iterations: 50000
```

---

## Experiment 2 — Student depth 위치 비교

주 모델의 성능이 예상보다 낮으면 block의 총수를 크게 바꾸지 않고 배치 위치를 비교한다.

### S96-Mid

```yaml
depth: [1, 1, 2, 1]
```

중간 해상도 표현력을 유지하는 주 모델이다.

### S96-Detail

```yaml
depth: [2, 1, 1, 1]
```

full-resolution detail refinement를 강화하는 대안이다.

| 모델 | 예상 장점 | 예상 단점 |
|---|---|---|
| S96-Mid | 계산 효율, alignment feature 유지 | 마지막 local refinement가 약할 수 있음 |
| S96-Detail | edge와 작은 물체 복원에 유리할 가능성 | FLOPs와 activation memory 증가 |

---

## Experiment 3 — 선택적 CM3A 축소

Student-S96가 충분히 강한 뒤에만 수행한다.

| 모델 | CM3A 사용 위치 |
|---|---|
| S96-All | H/2, H/4, H/8, H/4, H/2 |
| S96-No-H2 | H/4, H/8, H/4 |
| S96-Mid-Only | H/4 encoder, H/8, H/4 decoder |

첫 결과가 좋다면 CM3A를 줄일 이유가 없으므로 이 실험은 선택 사항이다.

---

# 6. 원 논문 학습 조건과 공정 비교

PAN-Crafter 논문은 다음 조건을 사용한다.

| 항목 | 설정 |
|---|---|
| PAN training patch | 64 × 64 × 1 |
| Native MS patch | 16 × 16 × \(C_{MS}\) |
| Iterations | 50,000 |
| Warmup | 100 steps |
| Optimizer | AdamW |
| Learning rate | \(1\times10^{-4}\) |
| Weight decay | 0.01 |
| Scheduler | Cosine |
| Nominal batch | 48 |
| MARs effective batch | 96 |
| PAN loss weight | \(\lambda=1.0\) |
| Local window | \(k=3\) |

MARs에서는 각 triplet을 batch dimension에서 복제해 한쪽은 MS mode, 다른 쪽은 PAN mode로 처리한다.

따라서 nominal batch가 48이면 실제 network input은 mode duplication 후 96개다.

Student와 Teacher의 공정 비교를 위해 다음을 우선 유지한다.

- 동일 dataset split
- 동일 random seed
- 동일 augmentation
- 동일 iteration 수
- 동일 effective batch
- 동일 scheduler
- 동일 loss weight
- 동일 평가 코드

4090의 메모리가 충분하더라도 Student batch를 임의로 키우면 optimization 조건이 달라지므로, 첫 비교에서는 batch를 같게 유지한다.

---

# 7. 측정해야 할 결과

## 7.1 Reduced-resolution 품질

- PSNR
- SSIM
- SAM
- ERGAS
- SCC
- Q4 또는 Q8

## 7.2 Full-resolution 품질

- HQNR
- \(D_s\)
- \(D_\lambda\)

## 7.3 계산 효율성

- Parameters
- FLOPs
- 1-image inference time
- 평균 inference time
- peak inference memory
- peak training memory
- iteration time
- samples/second
- 전체 50K training time

논문과 inference 효율을 비교할 때는 동일하게 `256 × 256 × 8` HRMS 출력 조건을 별도로 측정한다.

---

## 7.4 MARs mode별 진단

전체 loss만 기록하지 말고 다음을 분리한다.

\[
\mathcal L_{\mathrm{MS}}
=
\left\|
\hat I_{MS}^{HR}-I_{MS}^{HR}
\right\|_1
\]

\[
\mathcal L_{\mathrm{PAN}}
=
\left\|
\hat I_{PAN}^{rep}-I_{PAN}^{rep}
\right\|_1
\]

기록 항목:

- MS-mode train loss
- PAN-mode train loss
- MS/PAN loss 비율
- MS-mode validation PSNR
- PAN-mode residual MAE
- mode별 gradient norm
- mode별 feature norm

PAN loss가 지나치게 빠르게 0에 가까워지거나 MS loss보다 과도하게 크면 \(\lambda\) 조정이 필요할 수 있다. 첫 실험에서는 원 논문의 \(\lambda=1.0\)을 그대로 사용한다.

---

# 8. 결과 기록 템플릿

## 8.1 품질 및 효율성

| Model | Width | Depth | Params (M) | FLOPs (G) | Time (ms) | Infer Mem (GB) | Train Mem (GB) | PSNR | SAM | ERGAS | HQNR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PAN-Crafter-T | 128 | 2-2-2-2 | 7.17 | 79.03 | 9.0* | 1.75* |  |  |  |  |  |
| S96-Mid | 96 | 1-1-2-1 |  |  |  |  |  |  |  |  |  |
| S96-Detail | 96 | 2-1-1-1 |  |  |  |  |  |  |  |  |  |

`*`는 논문 보고값이며, 실제 장비에서 다시 측정한다.

---

## 8.2 Teacher 대비 상대 결과

| Model | Param ratio | FLOPs ratio | Speed-up | Memory ratio | PSNR gap | SAM gap | HQNR gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| S96-Mid |  |  |  |  |  |  |  |
| S96-Detail |  |  |  |  |  |  |  |

---

# 9. 이후 mutual learning으로 넘어갈 판단 기준

아래 기준은 논문의 공식 기준이 아니라 현재 연구를 위한 working criterion이다.

## 최소 조건

- Student parameter가 Teacher의 약 40–50%
- Student FLOPs와 inference time이 의미 있게 감소
- reduced-resolution 성능 손실이 과도하지 않음
- full-resolution HQNR이 안정적으로 유지
- Student가 texture, edge 또는 특정 scene에서 독립적인 강점을 보임

실용적인 초기 목표는 다음과 같이 둘 수 있다.

```text
Target parameter ratio: 0.45–0.48
Target PSNR gap: ≤ 0.3 dB
Target inference speed-up: ≥ 1.5×
Target inference memory ratio: ≤ 0.65
```

Student 성능이 지나치게 낮으면 mutual learning은 사실상 Teacher→Student one-way distillation이 된다.

반대로 Student가 Teacher와 유사한 성능을 유지하면서 특정 영역에서 다른 오류 패턴을 보이면 bidirectional mutual learning의 근거가 생긴다.

---

# 10. Student-first 단계의 핵심 질문

Student 단독 실험은 다음 질문에 답해야 한다.

1. Width 96과 depth `[1,1,2,1]`이 실제로 3.2–3.5M 범위인가?
2. Teacher 대비 FLOPs가 얼마나 감소하는가?
3. 논문 성능 대비 PSNR, SAM, ERGAS, HQNR 손실은 어느 정도인가?
4. MARs 효과가 작은 Student에서도 유지되는가?
5. PAN mode가 Student의 spatial sharpness를 실제로 개선하는가?
6. S96-Mid와 S96-Detail 중 어느 depth 배치가 유리한가?
7. Student가 Teacher와 다른 오류 영역을 보이는가?
8. 향후 두 모델을 동시에 올려도 24GB 안에서 학습 가능한가?

---

# 11. 권장 첫 실행 설정

```yaml
model:
  name: PAN-Crafter-S96-Mid
  hidden_size: [96, 96, 96, 96]
  s_embed_size: 96
  depth: [1, 1, 2, 1]
  num_heads: 8
  mlp_ratio: 4.0
  local_window: 3
  keep_all_cm3a: true
  use_mars: true

training:
  iterations: 50000
  optimizer: AdamW
  learning_rate: 1.0e-4
  weight_decay: 0.01
  scheduler: cosine
  warmup_steps: 100
  lambda_pan: 1.0
  mixed_precision: bf16_or_fp16

evaluation:
  reduced_resolution: true
  full_resolution: true
  profile_params: true
  profile_flops: true
  profile_inference_time: true
  profile_inference_memory: true
  profile_training_memory: true
```

---

# 12. 최종 요약

현재 Student 전략은 다음과 같다.

\[
\boxed{
\begin{aligned}
T &: C=128,\ d=[2,2,2,2],\ 5\times CM3A,\ MARs \\
S &: C=96,\ d=[1,1,2,1],\ 5\times CM3A,\ MARs \\
P_S/P_T &: \text{목표 }0.45\sim0.48
\end{aligned}
}
\]

Student에서 PAN-Crafter의 핵심인 MARs와 CM3A는 그대로 유지하고, width와 ResBlock depth만 줄인다.

MARs의 정확한 역할은 다음과 같다.

```text
동일 입력:
PAN + upsampled LRMS

MS mode:
최종 HRMS reconstruction
LRMS spectral fidelity + PAN spatial structure

PAN mode:
PAN high-frequency back-reconstruction
shared backbone에 spatial sharpness를 주는 auxiliary supervision
```

따라서 MARs는 동일 네트워크가 단순히 같은 입력을 두 번 보는 구조가 아니다. **같은 PAN/LRMS 쌍에 대해 서로 다른 target, residual baseline, attention query, modulation parameter를 사용해 spatial–spectral 상보성을 학습하는 multi-task reconstruction 구조**다.

---

## Source basis

- *PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening*, main paper Sections 3.1–3.3, Tables 4 and 10.
- Supplementary Tables 11–17.
