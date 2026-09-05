# s1 — Shift-Robust Conditioning + M-frame PAN Guidance Alignment 30시간 실험 계획서

**작성일:** 2026-09-06  
**총 예산:** 단일 GPU 기준 최대 30시간  
**주 서버:** s1  
**기준 구조:** `S1_T05_W168_D123_DUAL (50K) · w168 d123 11ch attn:0 nocrop`  
**최종 선택 기준:** `HQNR 최대 → HQNR 동률(1e-4) 시 fSCC 최대 → 더 늦은 checkpoint`  
**연구 범위:** Mutual learning은 종료한다. 이번 캠페인에는 KD, SiS, uncertainty, GT variance, Swin, SE, CM3A 재도입을 포함하지 않는다.

---

## 0. 이번 캠페인의 핵심 결정

### 유지

```text
Backbone              : width=168, depth=[1,2,3]
Input                 : 11ch
Input order           : [PAN, LPAN, PAN-LPAN, ↑MS(8ch)]
Attention             : 없음
Crop / scale jitter   : 없음
Task                  : dual MARs
MS mode               : HRMS reconstruction
PAN mode              : PAN back-reconstruction
Mode modulation       : βMS, γMS, βPAN, γPAN 유지
Upsampling            : 기존 PAN-Crafter bicubic 그대로
Optimizer             : AdamW
LR / weight decay     : 1e-4 / 0.01
Scheduler             : cosine + warmup 100
Iteration             : 50,000
Batch                 : 48, dual MARs 실효 96
PAN loss weight       : 1.0
Seed                  : 2025
```

기존 bicubic은 그대로 사용한다.

```python
ms_base = F.interpolate(
    ms,
    scale_factor=4,
    mode="bicubic",
    align_corners=False,
)
```

### 폐기·보류

```text
interp23tap 기반 주력 학습                  : 폐기
train 16×16 patch의 shift cache            : 폐기
C1/C4-A 입력 shift + 최종 output inverse   : 폐기
C3/C4-B P-frame final + inverse loss branch: 폐기
기존 noisy cache를 label로 쓰는 ShiftNet   : 폐기
MS/PAN output frame 변경                   : 폐기
local dense flow의 즉시 end-to-end 학습    : 이번 캠페인에서는 보류
```

### 고정 좌표계

```text
MS condition의 원본 기준   : M-frame
Residual base              : M-frame
GT                         : M-frame
최종 HRMS                  : M-frame
PAN structure feature      : 필요할 때 M-frame 쪽으로 sampling
PAN mode output / target   : 원래 PAN frame
Output inverse             : 없음
```

---

# 1. 이번 캠페인의 근거

기존 global-alignment 캠페인에서 다음이 확인됐다.

1. `C1`의 input shift → U-Net → final inverse는 HQNR과 RR 품질이 크게 붕괴했다.
2. `C3`의 PAN-frame final output은 fSCC가 높아졌지만, M-frame LRMS를 reference로 사용하는 공식 \(D_\lambda\)와 충돌해 HQNR이 크게 낮아졌다.
3. `C2`처럼 **네트워크가 보는 MS condition만 작게 이동**하고 residual base·GT·output은 그대로 두면 후반 \(D_s\) 악화가 억제됐다.
4. C2 모델은 추론 시 shift를 제거해도 HQNR이 유지됐다. 따라서 기존 cache shift는 실제 정합이라기보다 training-time perturbation으로 작동했을 가능성이 높다.
5. train LRMS \(16\times16\) patch에서는 shift 추정 오차가 실제 추정 shift보다 컸다. 따라서 sample별 cache를 실제 정합 label로 사용할 수 없다.
6. FR scene에서는 실제 global sub-pixel displacement가 관찰됐으며, global correction 이후에도 일부 edge 주변 local residual이 남았다.

이번 30시간은 다음 두 축을 분리한다.

```text
축 A: C2의 효과가 통제된 random jitter에서도 재현되는가?
축 B: output을 움직이지 않고 PAN feature만 M-frame으로 가져오면 유효한가?
```

---

# 2. 실험 질문

## H1 — Random C2 재현

> Noisy cache 없이, conditioning MS에만 통제된 작은 sub-pixel translation을 주어도 HQNR 및 후반 \(D_s\) 안정화가 재현되는가?

## H2 — Dual MARs mode 범위

> 같은 jitter를 MS/PAN 두 mode에 모두 주는 것이 좋은가, HRMS를 복원하는 MS mode에만 주는 것이 좋은가?

## H3 — Shift와 blur 분리

> C2의 이득이 위치 perturbation 때문인가, sub-pixel interpolation에 따른 약한 smoothing 때문인가?

## H4 — Clean–jitter consistency

> Clean MS와 jittered MS가 같은 M-frame residual을 생성하도록 직접 제약하면 C2보다 더 안정적인가?

## H5 — M-frame global PAN guidance

> MS를 PAN frame으로 옮기지 않고, PAN의 shallow structure feature를 M-frame으로 가져오면 HQNR과 fSCC를 함께 유지할 수 있는가?

## H6 — Local residual의 후속 가능성

> 신뢰도 높은 edge 부근에서만 PAN feature를 local sampling하면 global correction 이상의 이득이 있는가?

H6는 이번 캠페인에서 **inference diagnostic까지만** 수행한다. Learnable local module의 50K 학습은 diagnostic gate가 통과한 다음 캠페인으로 넘긴다.

---

# 3. 기준 아키텍처

## 3.1 Backbone

```text
Run family     : W168-D123-DUAL
Width          : 168
Depth          : [1,2,3]
Attention      : 0
Input          : 11ch
Task           : dual MARs
Output         : 8ch HRMS residual 또는 repeated PAN residual
```

## 3.2 11채널 입력

```text
0: PAN
1: LPAN = ↓PAN 후 기존 bicubic ↑
2: PAN-HF = PAN - LPAN
3~10: bicubic-upsampled LRMS 8 bands
```

입력 순서를 바꾸지 않는다.

## 3.3 Dual MARs

### MS mode

\[
\hat Y = M_{\mathrm{base}} + R_\theta(P, M_{\mathrm{cond}};\mathrm{MS})
\]

\[
L_{\mathrm{MS}} = \|\hat Y-Y_{\mathrm{GT}}\|_1
\]

### PAN mode

\[
\hat P^{rep} = P_{\mathrm{LP}}^{rep}
+R_\theta(P,M_{\mathrm{cond}};\mathrm{PAN})
\]

\[
L_{\mathrm{PAN}}
=
\|\hat P^{rep}-P^{rep}\|_1
\]

### 기본 loss

\[
L_{\mathrm{MARs}}
=
L_{\mathrm{MS}}
+
L_{\mathrm{PAN}}
\]

PAN mode에는 shift의 역함수, output warp, border mask를 적용하지 않는다.

---

# 4. 공통 구현 원칙

## 4.1 기존 bicubic 경로를 건드리지 않는다

```python
ms_base = existing_bicubic(ms)
lpan_hr = existing_bicubic(lpan_lr)
pan_hf  = pan - lpan_hr
```

기존 feeder의 flip·rotation·입력 순서를 그대로 사용한다.

새 jitter는 **기존 geometric augmentation과 bicubic upsampling이 끝난 뒤** `ms_base`의 복사본에만 적용한다. 이 순서를 사용하면 phase-specific augmentation 변환이나 shift-vector 회전 규칙이 필요 없다.

## 4.2 Jitter는 영상 전체의 작은 global translation

한 sample에 대해 하나의 \((\epsilon_y,\epsilon_x)\)만 생성한다.

```text
모든 spatial position : 동일한 shift
8개 MS band           : 동일한 shift
PAN                    : 이동하지 않음
MS residual base       : 이동하지 않음
GT                     : 이동하지 않음
최종 output            : 이동하지 않음
```

Pixel별 random shuffle, band별 서로 다른 shift, object별 independent shift를 사용하지 않는다.

## 4.3 Sub-pixel warp

```python
def translate_hr(
    x: Tensor,               # [B,C,H,W]
    epsilon_hr: Tensor,      # [B,2], order=(dy,dx), HR pixel
) -> Tensor:
    # out[y,x] = src[y + dy, x + dx]
    return F.grid_sample(
        x,
        build_grid(epsilon_hr),
        mode="bicubic",
        padding_mode="border",
        align_corners=False,
    )
```

- shift 단위는 **HR pixel**로 고정한다.
- 부호 규약은 unit test로 검증한다.
- `epsilon=(0,0)`이면 warp를 호출하지 않고 input tensor를 그대로 반환한다.
- 모든 MS band에 같은 grid를 사용한다.

---

# 5. Random C2 정의

## 5.1 기본 분포

기존 cache 적용값의 표준편차와 유사한 perturbation을 만들기 위해:

\[
\epsilon_x,\epsilon_y
\sim
U(-0.5,\;0.5)
\quad\text{HR pixel}
\]

로 둔다.

Uniform \([-0.5,0.5]\)의 축별 표준편차는 약 0.289 HR pixel, 즉 약 0.072 LR pixel이다. 기존 C2의 약 0.076 LR pixel 수준과 가깝다.

```yaml
jitter:
  enabled: true
  distribution: uniform
  max_abs_hr_px: 0.5
  probability: 1.0
  same_for_all_ms_bands: true
  apply_after_bicubic: true
  inference_enabled: false
```

## 5.2 입력 구성

\[
M_{\mathrm{cond}}=T_\epsilon(M_{\mathrm{base}})
\]

```python
x11 = torch.cat(
    [
        pan,
        lpan_hr,
        pan_hf,
        ms_cond,
    ],
    dim=1,
)
```

최종 MS output은 항상 clean base를 사용한다.

```python
res_ms = model(x11, mode="MS")
pred_ms = ms_base + res_ms
```

즉 `ms_cond`를 residual base로 사용하지 않는다.

---

# 6. Run J1 — Random C2, 두 mode 공통

**Run ID**

```text
SR_J1_C2RAND_BOTH_R050_W168_D123_DUAL
```

## Forward

```python
epsilon = sample_uniform(-0.5, 0.5)  # HR px
ms_cond = translate_hr(ms_base, epsilon)

# 같은 ms_cond를 dual batch에 복제
x_ms  = cat([pan, lpan_hr, pan_hf, ms_cond], dim=1)
x_pan = cat([pan, lpan_hr, pan_hf, ms_cond], dim=1)
```

## Loss

\[
L
=
\|M_{\mathrm{base}}+R_{\mathrm{MS}}-Y\|_1
+
\|P_{\mathrm{LP}}^{rep}+R_{\mathrm{PAN}}-P^{rep}\|_1
\]

## 목적

기존 cache 기반 C2가 random jitter만으로 재현되는지 확인한다.

## 필수 기록

```text
epsilon_x/y mean, std, min, max
loss_ms / loss_pan
HQNR / fSCC / D_lambda / D_s
best / plateau(eval index≥100) / final
inference jitter=0 결과
inference jitter=±0.5 결과
```

---

# 7. Run J2 — Random C2, MS mode only

**Run ID**

```text
SR_J2_C2RAND_MSONLY_R050_W168_D123_DUAL
```

## Forward

```python
epsilon = sample_uniform(-0.5, 0.5)

ms_cond_ms  = translate_hr(ms_base, epsilon)
ms_cond_pan = ms_base

x_ms  = cat([pan, lpan_hr, pan_hf, ms_cond_ms],  dim=1)
x_pan = cat([pan, lpan_hr, pan_hf, ms_cond_pan], dim=1)
```

## 목적

PAN reconstruction task에도 jitter를 주는 것이 shared representation regularization에 필요한지 분리한다.

## 판정

| 결과 | 해석 |
|---|---|
| J1 > J2 | PAN mode jitter도 shared backbone regularization에 기여 |
| J2 > J1 | PAN mode에서는 jitter가 입력 PAN과 충돌 |
| J1 ≈ J2 | PAN mode jitter 기여가 작음 |

파라미터 공유는 mode별 입력을 다르게 주는 것을 막지 않는다.

---

# 8. Run J3 — Blur control

**Run ID**

```text
SR_J3_BLURCTRL_MATCH_R050_W168_D123_DUAL
```

## 목적

C2의 이득이 spatial translation 때문인지, warp interpolation이 만든 smoothing 때문인지 분리한다.

## 8.1 Blur sigma 사전 보정

WV3 train에서 고정 500 sample을 사용한다.

J1 jitter와 clean MS 사이의 gradient-energy ratio를 구한다.

\[
r_{\mathrm{jit}}
=
\frac{
E[|\nabla T_\epsilon(M)|]
}{
E[|\nabla M|]
}
\]

후보 sigma:

```text
0.10 / 0.15 / 0.20 / 0.25 / 0.30 / 0.35 HR pixel
```

각 sigma의 depthwise Gaussian blur에서:

\[
r_{\mathrm{blur}}(\sigma)
=
\frac{
E[|\nabla G_\sigma(M)|]
}{
E[|\nabla M|]
}
\]

를 계산하고:

\[
\sigma^*
=
\arg\min_\sigma
|r_{\mathrm{blur}}(\sigma)-r_{\mathrm{jit}}|
\]

를 선택한다.

선택된 sigma와 calibration sample ID를 저장한다.

## 8.2 Forward

```python
ms_cond = depthwise_gaussian_blur(ms_base, sigma=sigma_star)
```

위치는 이동하지 않는다. 두 mode에 동일한 blurred condition을 사용한다.

## 해석

| 결과 | 결론 |
|---|---|
| J1만 개선 | shift correspondence robustness 가능성 |
| J1≈J3 | 주 효과가 conditioning smoothing일 가능성 |
| 둘 다 개선, J1 우세 | shift와 smoothing이 함께 작용 |
| J3만 개선 | C2 해석을 low-pass regularization으로 수정 |

---

# 9. Run J4 — Clean–Jitter Consistency

**Run ID**

```text
SR_J4_CJCONS_R050_L010_W168_D123_DUAL
```

## 9.1 Branch

MS mode에서 같은 weight를 두 번 사용한다.

```text
Clean MS branch
Jittered MS branch
```

PAN mode는 clean condition만 사용한다.

```python
x_ms_clean  = cat([pan, lpan_hr, pan_hf, ms_base], dim=1)
x_ms_jitter = cat([pan, lpan_hr, pan_hf, ms_jit],  dim=1)
x_pan_clean = cat([pan, lpan_hr, pan_hf, ms_base], dim=1)
```

## 9.2 Output

\[
R_0=F_\theta(P,M_{\mathrm{base}};\mathrm{MS})
\]

\[
R_\epsilon=F_\theta(P,T_\epsilon M_{\mathrm{base}};\mathrm{MS})
\]

\[
\hat Y_0=M_{\mathrm{base}}+R_0
\]

\[
\hat Y_\epsilon=M_{\mathrm{base}}+R_\epsilon
\]

## 9.3 Loss

두 MS branch 모두 GT anchor를 갖는다.

\[
L_{\mathrm{MS}}
=
\frac12\|\hat Y_0-Y\|_1
+
\frac12\|\hat Y_\epsilon-Y\|_1
\]

Consistency는 clean residual을 stop-gradient target으로 사용한다.

\[
L_{\mathrm{cons}}
=
\|
R_\epsilon-\operatorname{sg}(R_0)
\|_1
\]

PAN loss는 clean condition에서 기존대로 계산한다.

\[
L
=
L_{\mathrm{MS}}
+
L_{\mathrm{PAN}}
+
\lambda_{\mathrm{cons}}(t)L_{\mathrm{cons}}
\]

Schedule:

```text
0~5K    : λcons 0 → 0.1 linear warmup
5K~50K  : λcons = 0.1
```

## 9.4 Collapse 방지

- clean/jitter 양쪽에 GT loss를 적용한다.
- consistency만 clean branch를 stop-gradient한다.
- final HRMS가 아니라 **residual**을 맞춘다.
- PAN mode에는 consistency를 적용하지 않는다.
- `λcons * Lcons / LMS` 비율을 기록한다.

경고:

```text
평균 비율 > 0.30 : consistency 과도
평균 비율 < 0.01 : consistency 사실상 무효
```

J4는 dual batch 96에 MS branch 하나가 추가되므로 기존 학습의 약 1.5배 시간을 예상한다.

---

# 10. M-frame PAN feature alignment

## 10.1 First Conv 분리

기존 첫 Conv의 weight를 그대로 두고 channel contribution만 분리한다.

입력 순서:

```text
PAN group : channel 0~2
MS group  : channel 3~10
```

기존 Conv:

\[
F_0=\operatorname{Conv}([P_3,M_8])
\]

분리:

\[
F_P=\operatorname{Conv}(P_3;W_{0:3})
\]

\[
F_M=\operatorname{Conv}(M_8;W_{3:11})+b
\]

\[
F_0=F_P+F_M
\]

`shift=0`일 때 원래 Conv와 수치적으로 같아야 한다.

Bias는 한 번만 더한다.

## 10.2 MS mode

PAN contribution만 M-frame 방향으로 sampling한다.

\[
F_0^{MS}=F_M+\widetilde F_P^M
\]

## 10.3 PAN mode

기존 path를 그대로 사용한다.

\[
F_0^{PAN}=F_M+F_P
\]

PAN mode에는 PAN→M alignment를 적용하지 않는다.

---

# 11. Run G1 — Synthetic-supervised Global PAN Feature Aligner

**Run ID**

```text
AF_G1_PAN2M_GLOBALCORR_W168_D123_DUAL
```

기존 noisy train cache를 사용하지 않는다.

## 11.1 Candidate offsets

HR feature 기준:

\[
\Omega_g
=
\{-1.0,-0.5,0,0.5,1.0\}^2
\]

총 25개 후보를 사용한다.

WV3 FR 판정 subset에서 관찰된 전역 이동이 약 0.9 HR pixel 수준이므로 이 범위로 시작한다.

## 11.2 Descriptor

```python
d_pan = l2_normalize(conv1x1_pan(f_pan), dim=1)  # 16ch
d_ms  = l2_normalize(conv1x1_ms(f_ms),   dim=1)  # 16ch
```

두 1×1 projection만 trainable하게 추가한다.

## 11.3 Synthetic training shift

Training 때 PAN feature에 알려진 synthetic shift를 적용한다.

```python
epsilon_g ~ Uniform(-1.0, +1.0) HR pixel
probability = 0.75

25% sample:
  epsilon_g = 0
```

\[
F_P^{syn}=W(F_P,\epsilon_g)
\]

각 candidate \(\delta\)에 대해:

\[
s(\delta)
=
\operatorname{mean}_{x}
w_{\mathrm{edge}}(x)
\cdot
\langle
D_M(x),
D_P^{syn}(x+\delta)
\rangle
\]

Edge weight:

```python
w_edge = normalize(abs(PAN-HF))
w_edge = 0.25 + 0.75 * w_edge
```

확률:

\[
p(\delta)
=
\operatorname{Softmax}(s(\delta)/\tau)
\]

\[
\tau=0.07
\]

예측 correction:

\[
\hat\Delta_g
=
\sum_{\delta\in\Omega_g}
p(\delta)\delta
\]

`warp(x, delta)` 부호 규약에서 synthetic shift를 되돌리는 target은:

\[
\Delta_g^*=-\epsilon_g
\]

이다.

## 11.4 Confidence

\[
c_g
=
1-
\frac{H(p)}{\log 25}
\]

적용 gate:

\[
g_g=\operatorname{clip}(c_g/0.30,\;0,\;1)
\]

Aligned PAN contribution:

\[
\widetilde F_P^M
=
F_P^{syn}
+
g_g
\left[
W(F_P^{syn},\hat\Delta_g)-F_P^{syn}
\right]
\]

## 11.5 Loss

\[
L_{\mathrm{shift}}
=
\operatorname{SmoothL1}
(\hat\Delta_g,-\epsilon_g)
\]

\[
L
=
L_{\mathrm{MARs}}
+
0.1L_{\mathrm{shift}}
\]

PAN mode에서는 synthetic shift와 global aligner를 사용하지 않는다.

## 11.6 Inference

Synthetic shift는 없다.

```python
epsilon_g = 0
global correlator는 실제 PAN/MS pair에서 Δ를 추정
PAN feature만 M-frame 쪽으로 sampling
```

FR에서 audit scene shift는 **평가용 비교 target**으로만 사용하며 학습 label로 사용하지 않는다.

필수 진단:

```text
predicted Δ vs audit Δ: dy/dx correlation, median error
confidence mean / P10 / P90
center probability
candidate boundary probability
wrong-sign control
beta sweep: 0 / 0.5 / 1.0
```

---

# 12. Local alignment diagnostic

이번 30시간에는 learnable local module을 50K로 학습하지 않는다.

## 12.1 목적

Global correction 이후 edge 주변 local residual을 PAN feature에 적용했을 때 실제 model metric이 좋아지는지 먼저 확인한다.

## 12.2 대상 checkpoint

다음 중 HQNR→fSCC 기준 winner를 사용한다.

```text
J1
J2
J4
G1
```

## 12.3 외부 field

Audit에서 사용한 global-corrected TV-L1 또는 재계산한 local residual field를 사용한다.

```text
FR scene 단위
GT 사용 금지
PAN / LRMS 구조 map으로 field 계산
```

## 12.4 적용

First Conv의 PAN contribution에만 적용한다.

\[
F_P^{local}
=
F_P^g
+
c_l(x)
\left[
W(F_P^g,\Delta_l(x))-F_P^g
\right]
\]

Confidence gate:

```text
edge energy 상위 30%
forward-backward consistency 통과
flow magnitude <= 2 HR pixel
low-texture 영역은 gate=0
```

## 12.5 비교

```text
L0: no alignment
L1: global only
L2: global + gated local
L3: global + ungated local
L4: wrong-sign local
```

## 12.6 다음 캠페인 진입 gate

Learnable local module을 열려면 다음을 모두 만족해야 한다.

```text
HQNR drop <= 0.0005
fSCC gain >= 0.005
FR 12-19 중 6장 이상 동일 방향
gated local > ungated local
wrong-sign local은 개선하지 않음
WV2 진단에서 gate가 대부분 identity로 동작
```

이 gate를 못 넘으면 local alignment는 종료한다.

---

# 13. 30시간 실행 큐

기존 `S1_T05_W168_D123_DUAL`을 같은 s1 환경의 anchor로 사용한다. 새 trainer의 zero-jitter path는 unit/smoke에서 기존 path와 동치임을 확인하므로 별도의 full baseline 재학습은 하지 않는다.

## 13.1 기본 큐

| 순서 | 작업 | 예상 시간 | 누적 |
|---:|---|---:|---:|
| 0 | 구현·unit test·실배치 smoke | 3.5h | 3.5h |
| 1 | `J1 C2 random, both modes` | 3.5h | 7.0h |
| 2 | `J3 matched blur control` | 3.5h | 10.5h |
| 3 | `J4 clean–jitter consistency` | 5.2h | 15.7h |
| 4 | `J2 C2 random, MS mode only` | 3.5h | 19.2h |
| 5 | `G1 global PAN feature aligner` | 4.0h | 23.2h |
| 6 | global/local inference diagnostic | 1.5h | 24.7h |
| 7 | winner seed repeat 또는 conditional refinement | 3.5h | 28.2h |
| 8 | 최종 평가·표·sample panel | 1.3h | 29.5h |
| 9 | buffer | 0.5h | 30.0h |

## 13.2 마지막 3.5시간의 선택 규칙

### 우선: seed repeat

다음 중 winner를 seed 1234로 반복한다.

```text
J1 / J2 / J4 / G1
```

Winner 기준:

```text
1. best HQNR
2. HQNR 차이 <= 1e-4이면 fSCC
3. best가 동급이면 plateau HQNR
4. plateau도 동급이면 final HQNR
```

### 단, 명확한 실패 시 refinement

모든 새 case가 anchor보다 HQNR에서 0.002 이상 낮고 fSCC도 낮으면 seed repeat 대신 jitter radius refinement를 실행한다.

```text
D_s가 높게 남음:
  R075 = Uniform(-0.75,+0.75) HR px

D_lambda가 악화:
  R025 = Uniform(-0.25,+0.25) HR px
```

한 번에 radius와 PAN loss weight를 같이 바꾸지 않는다.

---

# 14. D_lambda–D_s 균형 조정

이번 캠페인에서 우선 조정하는 변수는 jitter radius다.

| Radius | 예상 경향 |
|---|---|
| 작음 | spectral anchor 보존, robustness 약함 |
| 중간 | C2 재현 가능성이 가장 높음 |
| 큼 | spatial robustness 강화 가능, MS condition 무시·blur 위험 |

초기에는:

```text
R050 = ±0.5 HR pixel
λPAN = 1.0 고정
```

을 사용한다.

PAN loss weight \(\lambda_{\mathrm{PAN}}\)은 이번 30시간에 변경하지 않는다. J1/J2/J4에서 같은 방향이 재현되고도 \(D_\lambda\)–\(D_s\) 불균형이 남을 때 다음 캠페인에서 별도 축으로 연다.

---

# 15. Checkpoint 선정

## 15.1 평가 주기

기존 trainer의 촘촘한 평가 주기를 유지한다. 정확한 50K checkpoint도 종료 시 반드시 평가한다.

## 15.2 공식 HQNR

FR index 12–19에서 장면별로:

\[
HQNR_i=(1-D_{\lambda,i})(1-D_{s,i})
\]

를 계산한 뒤 평균한다.

\[
HQNR=\frac{1}{8}\sum_i HQNR_i
\]

`(1-mean Dλ)(1-mean Ds)`를 selection에 사용하지 않는다.

## 15.3 Tie-break

```python
HQNR_EPS = 1e-4
FSCC_EPS = 1e-4

1. running maximum HQNR 기준
2. max와 차이가 <= 1e-4이면 fSCC가 큰 checkpoint
3. fSCC도 <= 1e-4 동률이면 더 늦은 checkpoint
```

---

# 16. 필수 평가

## Primary

```text
HQNR
fSCC
```

## Decomposition

```text
D_lambda
D_s
```

## RR 참고

```text
ERGAS
SAM
SCC
Q2n
PSNR
SSIM
```

## 안정성

```text
best HQNR
plateau mean HQNR
final HQNR
best iteration
late-stage D_s slope
late-stage D_lambda slope
```

## Scene별

```text
FR 12-19 각 장면 HQNR
FR 0-11 diagnostic
FR 0-19 전체
개선 장면 수
```

---

# 17. fSCC 기록

fSCC는 full-resolution에서 output HRMS의 luminance/high-frequency와 입력 PAN의 high-frequency 구조 간 상관을 나타내는 보조 spatial metric으로 사용한다.

```text
Checkpoint 선택:
  HQNR 우선
  HQNR 동률에서 fSCC

해석:
  fSCC 상승만으로 success 선언 금지
  D_lambda / D_s / HQNR과 함께 해석
```

C3처럼 PAN 구조에 더 맞아 fSCC가 올라도 공식 \(D_\lambda\)와 HQNR이 크게 악화될 수 있으므로 단독 판정하지 않는다.

---

# 18. Unit test

## Jitter path

```text
T01 epsilon=0이면 기존 input과 bitwise 동일
T02 8개 MS band에 동일 grid
T03 PAN/LPAN/PAN-HF는 이동하지 않음
T04 residual base는 clean ms_base
T05 GT와 final output frame은 M-frame
T06 inference에서는 jitter=0
T07 J1은 두 mode 동일 jitter
T08 J2는 MS mode만 jitter
T09 PAN output에 inverse/warp 없음
T10 shift sign impulse test
```

## Blur control

```text
T11 위치 중심이 바뀌지 않음
T12 channel별 독립 depthwise filter
T13 calibrated gradient-energy ratio 오차 <= 1%
```

## Consistency

```text
T14 clean/jitter branch가 같은 backbone parameter 사용
T15 clean residual은 consistency path에서 stop-gradient
T16 두 MS output 모두 GT gradient 수신
T17 PAN mode consistency 없음
```

## PAN feature alignment

```text
T18 split Conv 합이 원래 Conv와 1e-6 이내
T19 bias가 한 번만 적용
T20 candidate delta=0에서 identity
T21 synthetic +delta 후 target -delta 부호 검증
T22 PAN mode aligner 호출 0회
T23 G1 descriptor/shift loss에 gradient 존재
T24 low-confidence gate에서 identity
```

---

# 19. Smoke test

각 run 시작 전 1K 이하 smoke를 수행한다.

```text
실배치 48 × dual MARs
forward / backward / AdamW step
NaN / Inf 없음
GPU peak memory
iteration time
loss_ms / loss_pan non-zero
jitter statistics 예상 범위
output range
FR 512×512 inference
exact 11ch order
```

J4는 clean+jitter MS branch 때문에 별도 peak memory를 기록한다. OOM이면 batch를 임의로 줄이지 않고 gradient accumulation으로 **실효 batch 48**을 유지한다.

---

# 20. Logging

Step EMA:

```text
loss_total
loss_ms
loss_pan
loss_cons
loss_shift
epsilon_x/y mean/std
jitter gradient-energy ratio
global predicted dx/dy
global confidence
global boundary probability
```

Checkpoint:

```text
hqnr
fscc
d_lambda
d_s
rr_ergas
rr_sam
rr_scc
best_iteration
final_iteration
```

Run directory:

```text
work_dir/<run_id>/
├── config_resolved.yaml
├── code_commit.txt
├── train_log.jsonl
├── checkpoint_metrics.csv
├── scene_metrics.csv
├── best_hqnr.pt
├── best_hqnr_meta.json
├── last.pt
├── predictions/
└── visualizations/
```

---

# 21. 시각 샘플

모든 best checkpoint에서 동일한 FR 12–19 crop을 저장한다.

Panel:

```text
PAN
clean bicubic LRMS
jittered/blurred condition
output RGB
PAN-output edge overlay
difference from anchor
D_lambda / D_s / HQNR / fSCC
```

G1:

```text
predicted global shift
audit global shift
confidence
center probability
aligned PAN feature edge map
```

Local diagnostic:

```text
global-only
global+local
flow magnitude
gate map
wrong-sign result
```

대표 sample:

```text
HQNR 개선 상위 3
HQNR 악화 상위 3
fSCC 개선 상위 3
D_s 악화 상위 3
```

---

# 22. 판정 규칙

## J1 성공

다음 중 하나를 만족해야 한다.

```text
A. HQNR >= anchor + 0.0005, fSCC 비악화
B. best HQNR 동급이지만 plateau/final HQNR >= anchor + 0.003
   그리고 late-stage D_s 증가가 억제
```

## C2를 shift-robustness로 해석

다음을 모두 요구한다.

```text
J1 > J3 blur control
inference jitter=0에서도 성능 유지
jittered inference에서도 급격한 성능 하락 없음
두 seed에서 late-stage D_s 안정화 재현
```

## G1 성공

```text
HQNR >= J1/J4 winner - 0.0005
fSCC >= winner + 0.003
predicted shift와 audit shift 방향 상관이 양수
wrong-sign control은 개선하지 않음
boundary saturation < 10%
```

## Local 다음 단계 진입

§12.6 gate를 모두 만족해야 한다.

---

# 23. 결과에 따른 다음 방향

## Case A — J1/J4 성공, G1 실패

주 방향:

```text
Shift-Robust Conditioning
```

다음 캠페인:

```text
jitter radius
jitter probability
consistency weight
W152/c6 architecture transfer
W168 dual Teacher uncertainty/KD 재개
```

## Case B — J1/J4와 G1 모두 성공

주 방향:

```text
Shift-Robust Conditioning
+
M-frame Global PAN Guidance Alignment
```

다음 캠페인에서 learnable local gated sampler를 연다.

## Case C — Blur control과 J1이 동급

주 방향을 misalignment robustness가 아니라:

```text
MS-conditioning low-pass regularization
```

으로 수정한다.

## Case D — 모든 C2 계열 실패

기존 cache C2의 양성 결과는:

```text
interp23tap
cache distribution
환경/seed
```

중 하나에 의존한 것으로 판단하고 mainline에서 제거한다.

## Case E — Local diagnostic만 강하게 성공

Global learnable module보다:

```text
edge-confidence 기반 local PAN-feature sampling
```

을 우선한다. 단 output과 GT는 계속 M-frame에 둔다.

---

# 24. s2 병렬 활용 시 선택 사항

이 절은 30시간 s1 예산에 포함하지 않는다.

s2가 비어 있다면 mutual learning 대신 architecture-dependence control을 수행한다.

우선순위:

```text
1. W152-D123-DUAL + J1
2. c6 W128-D124-DUAL + J1
3. W168-D123-DUAL + J4
```

각 서버에서는 해당 서버의 local baseline과만 비교한다. 서버 간 절대 HQNR을 직접 비교하지 않는다.

목적:

```text
C2가 W168 특유의 현상인지
작은/중간 backbone에서도 late-stage D_s 안정화가 재현되는지
Teacher 후보 W168에서 consistency가 유효한지
```

---

# 25. 최종 30시간 요약

```text
Anchor:
  W168-D123-DUAL
  original bicubic
  dual MARs
  M-frame output

J1:
  conditioning MS에 ±0.5 HR px random global jitter
  두 mode 공통

J2:
  같은 jitter를 MS mode에만 적용

J3:
  위치 이동 없이 smoothing만 맞춘 blur control

J4:
  clean/jitter 두 MS branch
  residual consistency λ=0.1
  PAN mode clean

G1:
  first Conv의 PAN contribution만 분리
  synthetic-supervised global correlation
  PAN feature를 M-frame으로 sampling
  output inverse 없음

Local:
  50K 학습 전 FR inference diagnostic만 수행

Selection:
  HQNR → fSCC
```

가장 효과가 기대되는 순서는:

\[
\boxed{
\text{J4 clean–jitter consistency}
\;\gtrsim\;
\text{J1 random C2}
\;>\;
\text{G1 M-frame global PAN guidance}
\;>\;
\text{local learnable alignment}
}
\]

다만 J4의 효과를 주장하려면 J3 blur control과 seed repeat가 반드시 필요하다.
