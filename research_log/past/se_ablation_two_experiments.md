# R4 기반 SENet 최소 2-Case 검증 계획

## 1. 목적

현재 R4는 attention이 없는 경량 U-Net 계열이며, c6와 동일한 depth `[1, 2, 4]`를 유지한 채 width를 `128 → 96`으로 줄인 구조다. R4는 낮은 비용과 낮은 `D_s`를 보이지만, c6 대비 ERGAS·SAM·SCC·`D_λ`가 약화됐다.

이번 SENet 실험의 질문은 하나다.

> **SE channel recalibration이 R4의 제한된 채널을 더 효율적으로 사용하게 하여 spectral/reconstruction 성능을 회복할 수 있는가?**

아키텍처 탐색을 넓히지 않고, 가장 가능성이 높은 위치 두 곳만 독립적으로 검증한다.

---

## 2. 공통 기준

### 2.1 Baseline

```yaml
model: R4_w96_d124_noattn
width: 96
depth: [1, 2, 4]
attention: none
input: 11ch
crop: false
normalization: LayerNorm
dropout: 0
MARs: dual   # 현재 c6/R4 계열 기준
```

기존 R4 결과를 baseline으로 재사용하되, SE case는 동일 코드 커밋·seed·optimizer·scheduler·evaluation path를 사용한다.

### 2.2 SE 모듈

표준 Squeeze-and-Excitation 구성을 사용한다.

```text
Input feature F: [B, C, H, W]
  → Global Average Pooling
  → Linear/1×1 Conv: C → C/r
  → ReLU
  → Linear/1×1 Conv: C/r → C
  → Sigmoid
  → channel-wise multiplication with F
```

수식은 다음과 같다.

\[
z_c=\frac{1}{HW}\sum_{h,w}F_c(h,w)
\]

\[
s=\sigma\left(W_2\,\mathrm{ReLU}(W_1z)\right)
\]

\[
\widetilde F_c=s_cF_c
\]

공통 설정:

```yaml
se_reduction: 8
hidden_dim_at_C96: 12
pooling: global_average
hidden_activation: ReLU
gate_activation: Sigmoid
se_dropout: 0
mode_specific_se: false
```

`r=16`은 C=96에서 hidden dimension이 6으로 지나치게 작으므로 첫 실험에서는 사용하지 않는다.

---

## 3. Experiment 1 — Bottleneck SE-ResBlock

### ID

```text
SE1_R4_BTL_SE
```

### 가설

R4의 width 축소로 생긴 channel capacity 손실은 저해상도 bottleneck에서 channel 중요도를 재조정하면 일부 회복될 수 있다.

### 삽입 위치

H/4 bottleneck의 4개 ResBlock에 각각 SE를 삽입한다.

```text
Encoder
  H   : ResBlock ×1
  H/2 : ResBlock ×2

Bottleneck H/4
  SE-ResBlock ×4

Decoder
  H/2 : ResBlock ×2
  H   : ResBlock ×1
```

각 ResBlock에서는 **기존 residual branch의 마지막 convolution 출력과 residual addition 사이**에 SE를 둔다.

```text
x
 → existing LN / SiLU / Conv / mode modulation path
 → residual feature r
 → SE(r)
 → existing residual addition
```

현재 코드의 ResBlock 연산 순서를 바꾸지 않고, residual feature에 channel gate만 추가한다.

### 예상 비용

C=96, r=8인 SE 한 개의 weight는 대략 다음과 같다.

\[
2C^2/r=2\times96^2/8\approx2.3K
\]

4개 block에 적용해도 약 9–10K parameters 수준으로, 전체 모델 대비 증가량은 매우 작다.

### 확인 질문

1. ERGAS·SAM·`D_λ`가 R4보다 개선되는가?
2. R4의 낮은 `D_s`와 HQNR 특성이 크게 훼손되지 않는가?
3. MS mode와 PAN mode에서 SE gate 분포가 실제로 달라지는가?

---

## 4. Experiment 2 — H/2 Decoder Skip-Fusion SE

### ID

```text
SE2_R4_DEC_H2_SE
```

### 가설

R4의 문제는 bottleneck capacity보다, decoder에서 encoder skip·PAN detail·MS reconstruction feature가 결합될 때 제한된 채널 안에서 정보 선택이 충분하지 않은 것일 수 있다.

### 삽입 위치

H/2 decoder의 upsample feature와 encoder skip feature를 결합한 뒤, fusion convolution을 통과한 feature에 SE 하나를 적용한다.

```text
Bottleneck output
  → Upsample to H/2
  → Concatenate with H/2 encoder skip
  → Fusion Conv to C=96
  → SE
  → H/2 Decoder ResBlocks
```

정확한 위치는 다음과 같다.

```text
concat(up_feature, skip_feature)
 → existing fusion_conv
 → SE
 → existing decoder body
```

SE는 concatenation 이전이 아니라 **fusion conv 이후**에 둔다. 따라서 gate dimension은 2C가 아니라 C=96이고, 기존 skip-fusion 구현을 거의 변경하지 않는다.

### 예상 비용

SE 한 개만 추가하므로 약 2.3K parameters 수준이다.

### 확인 질문

1. skip-fusion 이후 spectral/spatial channel 선택이 개선되는가?
2. ERGAS·SAM·SCC가 좋아지면서 `D_s`가 유지되는가?
3. bottleneck SE보다 더 효율적인 위치인가?

---

## 5. 실험 매트릭스

| ID | 구조 | 신규 SE 수 | 목적 |
|---|---|---:|---|
| R4 | 기존 `w96 d124 no-attn` | 0 | 기준선 |
| SE1 | R4 + bottleneck SE-ResBlock ×4 | 4 | compressed channel capacity 보완 |
| SE2 | R4 + H/2 decoder skip-fusion SE ×1 | 1 | spatial/spectral fusion channel 선택 |

이번 단계에서는 `SE1+SE2` 결합 모델을 실행하지 않는다. 두 위치 중 하나가 독립적으로 유효할 때만 후속 결합 ablation을 연다.

c6는 신규 실험이 아니라 성능 상한 reference로 사용한다.

---

## 6. 학습 규칙

```yaml
iterations: 50000
optimizer: current R4 baseline과 동일
scheduler: current R4 baseline과 동일
seed: current R4 baseline과 동일
augmentation: nocrop + 기존 flip/rotation
checkpoint_selection: final 50K 또는 기존 validation-ERGAS 정책과 동일
FR_HQNR_for_checkpoint_selection: false
```

FR 8장 HQNR로 checkpoint를 선택하지 않는다. 모든 case의 checkpoint 선택 정책은 동일해야 한다.

---

## 7. 필수 로그

### 품질

```text
RR: ERGAS, SAM, SCC, Q8, PSNR, SSIM
FR: D_lambda, D_s, HQNR
```

### 비용

```text
Params
FLOPs
Inference latency
Peak memory
Training time
```

### SE 진단

각 SE 위치와 MARs mode별로 다음을 기록한다.

```text
gate mean / std / min / max
channel-wise gate histogram
MS-mode gate vs PAN-mode gate cosine similarity
saturated gate ratio: s < 0.05 or s > 0.95
SE parameter gradient norm
```

MS/PAN mode gate가 완전히 동일하고 gate variance도 거의 0이면 SE가 실질적으로 사용되지 않는 것으로 판단한다.

---

## 8. 성공 기준

SE의 목표는 HQNR만 올리는 것이 아니라 R4의 spectral/reconstruction 약점을 줄이는 것이다.

### 채택 조건

다음 중 하나를 만족해야 한다.

1. ERGAS와 SAM이 모두 개선되고 `D_s` 악화가 제한적임
2. ERGAS 또는 SAM이 명확히 개선되고, 다른 하나가 0.3% 이내 비열화이며 `D_λ`도 개선됨
3. R4와 품질이 동급이면서 비용 증가는 1% 미만이고 gate 분석에서 mode-dependent channel selection이 확인됨

### 실패 조건

```text
HQNR만 상승하고 ERGAS·SAM·SCC가 동반 악화
SE gate가 거의 상수로 수렴
latency 증가가 품질 이득보다 큼
SE1과 SE2 모두 R4에 지배됨
```

---

## 9. 결과별 결정

| 결과 | 결정 |
|---|---|
| SE1만 유효 | bottleneck SE를 R4 Student 후보에 유지 |
| SE2만 유효 | skip-fusion SE를 R4 Student 후보에 유지 |
| 둘 다 유효 | 후속 1회로 `SE1+SE2` 결합 검증 |
| 둘 다 무효 | SE 계열 종료, loss/KD 연구에 집중 |
| HQNR만 개선 | 채택하지 않고 trade-off ablation으로 기록 |

---

## 10. 구현 체크리스트

- [ ] `use_se=false`일 때 기존 R4 checkpoint와 동작이 변하지 않음
- [ ] SE output shape가 input feature shape와 동일함
- [ ] MS mode와 PAN mode 모두 forward/backward 성공
- [ ] SE parameter gradient가 non-zero임
- [ ] 512×512 FR 입력에서 메모리 문제 없음
- [ ] params/FLOPs/latency를 실제 측정함
- [ ] SE1과 SE2 이외의 architecture/loss 변수는 변경하지 않음

