# UVS-KD: Uncertainty–Variance Guided Shift Distillation
## 30시간 실험 설계서

- 작성일: 2026-09-06
- 대상: PanCollection WV3, PAN-Crafter 기반 Teacher–Student PAN-sharpening
- 총 실험 예산: 단일 GPU 약 30시간
- 목적: Teacher가 복원 불확실성, GT detail 중요도, 전역 PAN–MS 이동 단서를 Student에 전달했을 때 경량 Student가 misalignment에 더 강건해지는지 검증

---

# 0. 이번 캠페인의 결정

이번 30시간 실험에서는 **dense local flow, CM3A attention-map distillation, 일반 feature KD를 우선 제외**하고, 아래의 단순한 신호만 검증한다.


a) Teacher restoration uncertainty map

\[
U_T(x)\in[0,1]
\]

b) GT residual variance map

\[
V_{GT}(x)\in[0,1]
\]

c) Teacher global shift token

\[
z_T=[\Delta y_T,\Delta x_T,c_T]
\]

여기서 \(c_T\)는 shift confidence이다. Teacher가 Student에게 직접 전달하는 기하학 정보는 끝까지 **방향 2개와 신뢰도 1개, 총 3개 숫자**로 제한한다.

최종 주력 case는 다음이다.

> **UVS-TF**: U-Know식 uncertainty routing + GT residual variance weighting + global shift-vector KD + scheduled shift teacher forcing

핵심 데이터 흐름은 다음과 같다.

```text
Raw PAN, MS
    │
    ├─ Teacher Shift Module ──> [dy_T, dx_T, confidence_T]
    │
    ├─ Teacher Fusion Model ──> HRMS_T, uncertainty_T
    │
    └─ GT ────────────────────> GT residual variance

Raw PAN, MS
    │
    ├─ Student Shift Module ──> [dy_S, dx_S, confidence_S]
    │                              │
    │                  early: Teacher shift 사용
    │                  late : Student shift 사용
    │                              ▼
    └──────────────────────> PAN warp → 11ch input → Student → HRMS_S
```

---

# 1. 근거와 범위

## 1.1 audit 결과가 지지하는 선택

PanCollection 정합 audit에서 FR 입력의 fSCC 개선은 local warp보다 scene-wise global translation에서 더 크게 나타났다.

| 센서 | 전역 보정 이득 | 전역 이후 local 이득 |
|---|---:|---:|
| WV3 | +0.1549 | +0.0384 |
| QB | +0.3057 | +0.0446 |
| GF2 | +0.0668 | +0.0430 |
| WV2 | +0.0080 | **−0.0398** |

따라서 첫 30시간에는 dense flow보다 global shift token이 적합하다. local warp는 이득이 더 작고, WV2에서는 해로웠으므로 confidence gate 없이 우선 도입하지 않는다.

또한 train patch의 PAN–LRMS shift는 센서별 P99가 약 2.3–2.75 LR pixel까지 관찰되었다. 따라서 shift search range는 MS/LR 격자에서 \([-3,3]^2\)로 둔다.

## 1.2 bicubic phase 문제의 처리

기존 `F.interpolate(ms, scale_factor=4, mode="bicubic", align_corners=False)`와 PanCollection의 제공 `lms` 사이에는 일관된 \((-0.5,-0.5)\) PAN-pixel phase 차이가 확인되었다. 이 고정 오차는 KD 대상이 아니다.

이번 실험에서는 다음을 공통 규칙으로 사용한다.

> **Teacher와 Student 모두 PanCollection이 제공하는 `lms`를 upsampled MS와 residual base로 사용한다.**

즉, 모든 run의 11채널 입력은 다음과 같다.

\[
I=[P,\;LP(P),\;P-LP(P),\;LMS]
\]

shift를 적용하는 case에서는 PAN을 먼저 이동한 뒤 PAN-derived 채널을 다시 계산한다.

\[
P^a=W(P,4\delta),
\qquad
I^a=[P^a,LP(P^a),P^a-LP(P^a),LMS]
\]

\(\delta\)는 LR/MS-grid pixel 단위이므로 full PAN 격자에서는 \(4\delta\)를 사용한다.

## 1.3 PAN-Crafter와 U-Know에서 가져오는 요소

- PAN-Crafter의 full Teacher는 CM3A와 MARs를 통해 misaligned PAN–MS 입력을 처리하는 강한 fusion model로 사용한다.
- PAN-Crafter가 MS mode에서 LRMS geometry를 우선한다는 점을 따라, **PAN을 MS/GT frame으로 이동**한다.
- U-Know-DiffPAN의 핵심처럼 Teacher uncertainty가 높은 곳은 GT hard target을, 낮은 곳은 Teacher soft target을 더 신뢰한다.
- GT variance는 Teacher uncertainty와 다른 역할을 갖는다. uncertainty는 **누구를 믿을지**, GT variance는 **어디를 중요하게 볼지** 결정한다.

---

# 2. 검증할 가설

## H1. Uncertainty routing

일반 output KD보다 uncertainty-aware hard/soft routing이 Student를 개선한다.

\[
\text{U-KD} > \text{plain output KD}
\]

## H2. GT variance

GT 전체 intensity가 아니라 GT high-frequency residual의 local variance를 가중치로 사용할 때 edge/detail 복원이 개선된다.

\[
\text{UV-KD} > \text{U-KD}
\]

## H3. Shift cue

Teacher의 global shift token을 직접 distill하면 Student의 shift recovery와 shifted-input robustness가 개선된다.

\[
\text{UVS-Vec} > \text{UV-KD}
\]

## H4. Early teacher forcing

학습 초기에 Teacher shift로 정합된 입력을 제공하고 이후 Student shift로 전환하면, 처음부터 Student shift만 쓰는 것보다 안정적이다.

\[
\text{UVS-TF} > \text{UVS-Vec}
\]

## H5. 실질적 성공 조건

shift MAE만 낮아지는 것으로는 부족하다. 아래 두 종류의 개선이 동시에 필요하다.

1. synthetic/controlled shift에서 Student shift MAE 및 성능 저하 곡선 개선
2. 표준 RR 또는 FR의 SCC, \(D_s\), HQNR 중 적어도 하나에서 baseline 대비 no-harm 이상의 개선

---

# 3. Teacher–Student 구성

## 3.1 Fusion Teacher

기존 sheet의 `c0_hqnr` checkpoint를 사용한다.

| 항목 | 설정 |
|---|---|
| backbone | PAN-Crafter full reconstruction |
| width | 128 |
| depth | `[2, 2, 4]` |
| alignment | CM3A 3개, encoder/bottleneck/decoder |
| training | dual MARs |
| params | 7.173M |
| 기존 train time | 5.21h / 50K |
| 기존 WV3 ERGAS | 2.0734 |
| 기존 WV3 HQNR | 0.9542 |

Teacher 전체를 처음부터 다시 학습하지 않는다. 기존 checkpoint에 아래 두 head를 부착한다.

- `T_unc`: 1-channel restoration uncertainty head
- `T_shift`: global shift module

필요한 경우에만 aligned-input으로 10K 정도 짧게 fine-tuning한다.

## 3.2 Primary Student

첫 검증에는 sheet에서 안정적이고 거의 무손실인 `d122`를 사용한다.

| 항목 | 설정 |
|---|---|
| width | 128 |
| depth | `[1, 2, 2]` |
| attention | 없음 |
| MARs | 유지 |
| params | 3.1795M |
| 기존 train time | 2.29h / 50K |
| 기존 infer time | 5.89ms |
| 기존 WV3 ERGAS | 2.0884 |
| 기존 WV3 HQNR | 0.9539 |

선정 이유는 다음과 같다.

- full Teacher와 width가 같아 KD 최적화가 단순하다.
- `w96`보다 run-to-run 변동이 작다.
- 이미 Teacher에 매우 근접하므로, 성능 차이를 구조 부족보다 KD 기전 차이로 해석하기 쉽다.

## 3.3 Compression confirmation Student

주력 방법이 성립한 뒤 마지막 1회만 `d122_w96`에 적용한다.

| 항목 | 설정 |
|---|---|
| width | 96 |
| depth | `[1, 2, 2]` |
| params | 1.7948M |
| 기존 train time | 약 1.94–2.11h |
| 주의 | 기존 실행 간 ERGAS 차이가 약 1.67%로 큼 |

`w96` 결과는 primary claim이 아니라 압축 가능성 확인용이다.

---

# 4. Shift module: 단순한 3-value cue

## 4.1 좌표와 부호

warp는 아래처럼 정의한다.

\[
W(I,\delta)(y,x)=I(y+\delta_y,x+\delta_x)
\]

\(\delta_{A\leftarrow B}\)는 B를 움직여 A에 맞추는 양이다. 이번에는 PAN을 MS/GT geometry에 맞추므로 출력은

\[
\delta_{MS\leftarrow PAN}
\]

이다.

실험 시작 전 impulse test와 known-shift test로 다음을 확인한다.

- identity 입력의 예측 및 warp가 정확히 center
- \(+dy,+dx\)의 부호가 audit 코드와 일치
- LR-grid 1 pixel이 PAN-grid 4 pixel로 적용

## 4.2 입력 표현

shift estimation은 PAN 격자가 아니라 audit에서 가장 안정적이었던 MS 격자에서 수행한다.

\[
P_{LR}=\operatorname{MTFDown}(P)
\]

\[
M_I=\frac{1}{C}\sum_c M_c
\]

두 입력에 Scharr gradient magnitude와 per-patch normalization을 적용한다.

\[
E_P=\operatorname{Norm}(|\nabla P_{LR}|),
\qquad
E_M=\operatorname{Norm}(|\nabla M_I|)
\]

raw intensity 대신 edge representation을 사용해 PAN–MS radiometric 차이를 줄인다.

## 4.3 Teacher shift module

Teacher shift module은 두 개의 작은 modality-specific encoder와 cost volume으로 구성한다.

```text
E_P ── Conv 1→16→32→32 ──┐
                          ├─ correlation volume, offsets [-3,3]²
E_M ── Conv 1→16→32→32 ──┘
                                   │
                              softmax(T=0.07)
                                   │
                 soft-argmax → dy_T, dx_T
                 entropy     → confidence_T
```

각 offset \(\Delta\in\Omega\), \(\Omega=[-3,3]^2\)에 대해 overlap 영역의 cosine correlation을 구한다.

\[
p_T(\Delta)=\operatorname{Softmax}(C_T(\Delta)/T_s)
\]

\[
\delta_T=\sum_{\Delta\in\Omega}p_T(\Delta)\Delta
\]

\[
c_T=1-\frac{H(p_T)}{\log |\Omega|}
\]

Teacher가 Student에게 넘기는 값은 cost volume 전체가 아니라 다음 세 값뿐이다.

\[
\boxed{z_T=[\Delta y_T,\Delta x_T,c_T]}
\]

## 4.4 Student shift module

Student는 같은 연산을 더 작은 채널로 수행한다.

```text
E_P ── Conv 1→8→8 ──┐
                     ├─ same 7×7 offset correlation
E_M ── Conv 1→8→8 ──┘
```

출력은

\[
z_S=[\Delta y_S,\Delta x_S,c_S]
\]

이다. parameter overhead는 Student backbone에 비해 무시 가능한 수준으로 유지한다.

## 4.5 Confidence gating

실제 적용 shift는 confidence를 곱한 값으로 정의한다.

\[
\hat\delta_T=c_T\delta_T,
\qquad
\hat\delta_S=c_S\delta_S
\]

inference에서는 다음을 사용한다.

\[
\delta_{infer}=
\begin{cases}
\hat\delta_S,&c_S\ge0.35\\
0,&c_S<0.35
\end{cases}
\]

이는 WV2와 저텍스처 장면에서 불필요한 warp가 성능을 해치는 것을 방지하기 위한 no-harm gate이다.

---

# 5. Teacher cue 준비

## 5.1 `T_shift` 학습 데이터

Teacher shift module은 두 신호로 학습한다.

### A. Audit pseudo-label

train patch audit에서 얻은 고신뢰 \(\delta_A\)와 confidence \(q_A\)를 사용한다.

\[
L_{T,real}=q_A\operatorname{SmoothL1}(\delta_T,\delta_A)
\]

가능하면 다음 기준을 통과한 patch만 사용한다.

- primary–secondary estimator 차이가 설정 threshold 이하
- peak margin 통과
- search boundary hit 아님
- audit high-confidence flag

### B. Synthetic translation consistency

고신뢰 real pair에 추가 translation \(a\)를 적용하고 total correction을 예측하게 한다.

\[
P^{aug}=W(P,-4a)
\]

작은 translation에서 additive approximation을 사용한다.

\[
\delta_T(P^{aug},M)\approx\delta_A+a
\]

sampling mixture는 다음과 같다.

| 확률 | 범위, LR pixel | 목적 |
|---:|---:|---|
| 0.20 | exactly 0 | no-harm 및 identity |
| 0.60 | uniform \([-1,1]^2\) | 일반적인 부화소 shift |
| 0.20 | uniform \([-3,3]^2\) | audit tail coverage |

## 5.2 `T_unc` 학습

Teacher의 마지막 MS-mode decoder feature에서 1-channel positive uncertainty를 예측한다.

```text
final decoder feature
    → Conv 3×3, C→C/4
    → SiLU
    → Conv 1×1, C/4→1
    → SoftPlus + 1e-4
    → theta_T
```

Teacher residual error의 band mean을

\[
e_T(x)=\frac{1}{C}\sum_c|Y_T(x,c)-Y_{GT}(x,c)|
\]

로 두고 U-Know 계열 heteroscedastic objective로 uncertainty head를 학습한다.

\[
L_{T,unc}=\left\langle
\frac{e_T}{2\theta_T}+\frac12\log\theta_T
\right\rangle
\]

Student KD에 사용할 때는 training-set running percentile로 \([0,1]\)에 정규화한다.

\[
U_T=\operatorname{clip}
\left(
\frac{\theta_T-Q_{10}}{Q_{90}-Q_{10}+\epsilon},0,1
\right)
\]

Teacher output, uncertainty, shift는 모두 `stop_gradient`한다.

## 5.3 Teacher adaptation 순서

1. `c0_hqnr` backbone 고정
2. `T_shift`만 5K–10K 학습
3. `T_unc`를 붙여 5K 학습
4. aligned PAN을 MS-mode 입력으로 넣었을 때 Teacher RR/FR no-harm 확인
5. 필요할 때만 backbone + `T_unc`를 10K fine-tuning

MS mode에서는 aligned PAN을 사용하고, PAN mode에서는 기존 raw PAN과 기존 PAN reconstruction target을 유지한다. 이렇게 하면 첫 캠페인에서 MARs의 의미를 크게 바꾸지 않는다.

## 5.4 Teacher gate

Student full run 전에 아래를 통과해야 한다.

| 항목 | 통과 기준 |
|---|---:|
| synthetic shift MAE, \(|\delta|\le1\) | ≤ 0.12 LR px |
| synthetic shift MAE, 전체 \([-3,3]\) | ≤ 0.25 LR px |
| identity MAE | ≤ 0.05 LR px |
| uncertainty vs actual error Spearman | ≥ 0.30 |
| aligned Teacher RR ERGAS 변화 | raw Teacher 대비 악화 ≤ 0.5% |
| high-confidence FR predicted shift vs audit | median vector difference ≤ 0.20 LR px |

Teacher shift gate가 실패하면 shift Student full run을 강행하지 않는다. 해당 시간은 `UV-KD` 반복과 oracle/pseudo-shift 진단에 사용한다.

## 5.5 Teacher cache — 30시간 예산의 필수 조건

sheet의 `d122` 2.29h는 Student 단독 학습 시간이다. 매 iteration마다 full `c0_hqnr` Teacher를 함께 forward하면 각 KD run이 이 시간보다 크게 늘어나므로, **Teacher adaptation이 끝난 뒤 training set 전체의 교사 신호를 한 번만 cache**한다.

patch별 저장 항목은 다음과 같다.

```text
patch_id
R_T       : Teacher residual, float16, C×64×64
U_T       : normalized uncertainty, float16, 1×64×64
δ_T       : [dy, dx], float32, LR-pixel
c_T       : scalar confidence, float32
```

WV3 9,714 patch 기준 `R_T`와 `U_T`의 예상 저장량은 대략 0.7–0.8 GB 수준이다. `V_GT`도 선택적으로 cache할 수 있으나 on-the-fly 계산 비용이 작으므로 필수는 아니다.

random flip/90° rotation을 적용할 때 cache도 같은 방식으로 변환한다.

- horizontal flip: \((dy,dx)\rightarrow(dy,-dx)\)
- vertical flip: \((dy,dx)\rightarrow(-dy,dx)\)
- 90° CCW rotation: \((dy,dx)\rightarrow(-dx,dy)\)
- 180° rotation: \((dy,dx)\rightarrow(-dy,-dx)\)
- 270° CCW rotation: \((dy,dx)\rightarrow(dx,-dy)\)

`R_T`, `U_T`는 영상과 동일하게 flip/rotate한다. 이 변환을 unit test하지 않으면 augmentation 이후 shift KD의 부호가 깨질 수 있다.

이번 30시간 캠페인에서는 Student 학습에 별도의 synthetic shift augmentation을 추가하지 않는다. train data가 가진 intrinsic shift에 대해 먼저 기전을 검증하고, synthetic translation은 Teacher shift-module pretraining과 controlled-shift 평가에만 사용한다. 이렇게 해야 `K2_uvkd → M1_uvs` 비교가 추가 augmentation 효과와 혼동되지 않는다.

---

# 6. GT residual variance

## 6.1 계산 대상

GT 자체가 아니라 upsampled LRMS에 추가되어야 하는 residual을 사용한다.

\[
R_{GT}=Y_{GT}-LMS
\]

## 6.2 local variance

band별 \(5\times5\) local variance를 계산한 후 평균한다.

\[
V_{raw}(x)
=
\frac1C\sum_c
\left[
\operatorname{AvgPool}(R_{GT,c}^2)
-
\operatorname{AvgPool}(R_{GT,c})^2
\right]
\]

training running percentile로 정규화한다.

\[
V_{GT}=\operatorname{clip}
\left(
\frac{V_{raw}-Q_{10}}{Q_{90}-Q_{10}+\epsilon},0,1
\right)
\]

loss scale이 달라지지 않도록 importance weight는 평균 1로 정규화한다.

\[
w_V=\frac{1+\alpha_VV_{GT}}
{\operatorname{mean}(1+\alpha_VV_{GT})}
\]

초기값은

\[
\alpha_V=1.0
\]

이다. variance map은 `detach`하며 inference에는 사용하지 않는다.

---

# 7. Student loss

Student와 Teacher는 full image가 아니라 residual을 비교한다.

\[
R_S=Y_S-LMS,
\qquad
R_T=Y_T-LMS
\]

## 7.1 Hard GT loss

Teacher uncertainty가 높은 곳에서는 Teacher output보다 GT를 더 신뢰한다.

\[
L_{hard}
=
\left\langle
w_V(1+U_T)\rho(R_S-R_{GT})
\right\rangle
\]

## 7.2 Soft Teacher loss

Teacher uncertainty가 낮은 곳에서는 Teacher의 detail을 모방한다.

\[
L_{soft}
=
\left\langle
w_V(1-U_T)\rho(R_S-R_T)
\right\rangle
\]

\(\rho\)는 Charbonnier 또는 기존 L1을 사용한다. 첫 캠페인에서는 코드 변경을 최소화하기 위해 기존 L1을 우선한다.

## 7.3 Shift token KD

\[
L_{vec}
=
\left\langle
c_T\operatorname{SmoothL1}(\delta_S-\delta_T)
\right\rangle
\]

confidence도 맞추되 보조항으로만 둔다.

\[
L_{conf}=\left\langle(c_S-c_T)^2\right\rangle
\]

\[
L_{shift}=L_{vec}+0.1L_{conf}
\]

## 7.4 Optional shift-effect loss

vector MAE는 좋아졌지만 실제 aligned PAN의 구조가 Teacher와 다를 때만 추가한다.

\[
H(P)=P-LP(P)
\]

\[
L_{warp}
=
\left\langle
c_TV_{GT}
\left|
H(W(P,4\hat\delta_S))
-
H(W(P,4\hat\delta_T))
\right|
\right\rangle
\]

이 항은 core method가 아니라 conditional run이다.

## 7.5 MARs PAN loss

기존 PAN-Crafter auxiliary PAN reconstruction loss는 유지한다.

\[
L_{PAN}=\|\hat P^{rep}-P^{rep}\|_1
\]

shift KD 및 uncertainty/variance routing은 MS mode에만 적용한다.

## 7.6 Total loss

주력 case의 최종 loss는 다음이다.

\[
\boxed{
L
=
L_{hard}
+\lambda_sL_{soft}
+\lambda_{pan}L_{PAN}
+\lambda_\delta L_{shift}
+\lambda_wL_{warp}
}
\]

초기값은 다음과 같다.

| 계수 | 값 |
|---|---:|
| \(\lambda_s\) | 0.1 |
| \(\lambda_{pan}\) | 1.0 |
| \(\lambda_\delta\) | 0.25 |
| \(\lambda_w\) | 0.0, conditional case에서 0.05 |
| \(\alpha_V\) | 1.0 |
| confidence threshold | 0.35 |

---

# 8. Shift teacher forcing

## 8.1 사용 shift

Teacher와 Student의 confidence-gated shift를 섞는다.

\[
\delta_{use}^{(t)}
=
\eta(t)\operatorname{sg}(\hat\delta_T)
+
[1-\eta(t)]\hat\delta_S
\]

여기서 `sg`는 stop-gradient이다.

## 8.2 schedule

| iteration | \(\eta(t)\) | 의미 |
|---:|---:|---|
| 0–5K | 1.0 | Teacher-aligned input으로 fusion 우선 학습 |
| 5K–20K | 1.0 → 0.0 선형 감소 | Teacher cue에서 Student cue로 전환 |
| 20K–50K | 0.0 | Student shift만 사용 |

이 schedule은 두 문제를 분리한다.

1. 초기: 정합된 PAN detail을 어떻게 주입할지 학습
2. 후기: raw input에서 Student가 직접 shift를 복원

## 8.3 no-shift case

`c_S < 0.35`이면 warp를 identity로 둔다. 이 threshold는 validation의 confidence–error calibration으로 0.25, 0.35, 0.50 중 선택하되 full 50K sweep은 하지 않는다.

---

# 9. 실험 case

## 9.1 필수 및 조건부 run

| ID | 구성 | 검증 목적 | 우선순위 |
|---|---|---|---|
| `B0_lms` | d122, provided `lms`, 기존 loss | phase-correct 공통 baseline | 필수 |
| `K0_outkd` | B0 + unweighted output KD | 일반 KD 대조 | 필수 |
| `K1_ukd` | B0 + uncertainty hard/soft routing | uncertainty 효과 | 필수 |
| `K2_uvkd` | K1 + GT residual variance | uncertainty와 detail importance 결합 | 필수 |
| `S0_shift` | B0 + shift token KD + Student warp | shift cue 단독 기전 | 필수 |
| `M1_uvs` | K2 + shift token KD, Student shift를 처음부터 사용 | 통합 효과, teacher forcing 제외 | 필수 |
| `M2_uvs_tf` | M1 + shift teacher forcing | **주력 방법** | 필수 |
| `M3_uvs_tf_warp` | M2 + shift-effect loss | vector가 실제 warp로 연결되지 않을 때 | 조건부 |
| `R1_m2_seed` | M2 second seed | 재현성 | M2가 통과하면 |
| `C1_m2_w96` | M2를 d122_w96에 적용 | 압축 확장성 | M2가 통과하면 |

## 9.2 해석 가능한 비교

- `K0 → K1`: uncertainty routing의 순수 효과
- `K1 → K2`: GT variance의 순수 효과
- `B0 → S0`: shift cue의 순수 효과
- `K2 → M1`: UV-KD 위에 shift를 더한 효과
- `M1 → M2`: early teacher forcing 효과
- `M2 → M3`: vector KD와 shift-effect KD 차이

## 9.3 이번에 제외할 case

### Dense local flow KD

전역 보정보다 추가 이득이 작고 WV2에서 악화가 확인되었다. global cue가 성립한 뒤 후속 캠페인에서만 고려한다.

### CM3A attention-map KD

CM3A attention은 correspondence뿐 아니라 feature selection이 섞여 있어 displacement target으로 해석하기 어렵다. sheet에서도 CM3A 제거의 손실이 매우 작아 shift supervision의 교사 신호로 바로 쓰기에는 근거가 약하다.

### 일반 feature KD

Teacher와 Student의 feature가 서로 다른 좌표에 놓일 수 있으므로 raw feature L1은 misalignment를 오히려 강제할 수 있다. 먼저 global shift cue를 검증한다.

### width 80, 9ch, LR-only

기존 sheet에서 품질 저하 또는 구조적 실패가 확인되었다. 방법론 검증 단계에서는 구조 bottleneck을 추가하지 않는다.

---

# 10. 30시간 실행 순서

sheet의 `d122` 50K 학습 시간 2.29h를 기준으로 산정한다. Teacher full checkpoint가 이미 존재하고, Student KD 동안 Teacher output·uncertainty·shift cue를 cache에서 읽는다는 전제이다. online Teacher forward를 유지하면 아래 시간표는 성립하지 않는다.

| 단계 | 작업 | 예상 시간 | 누적 |
|---|---|---:|---:|
| P0 | warp/부호/phase unit test, controlled-shift evaluator | 1.0h | 1.0h |
| P1 | `T_shift` + `T_unc` 준비, Teacher gate, 교사 신호 cache | 2.5h | 3.5h |
| R0 | `B0_lms` | 2.3h | 5.8h |
| R1 | `K0_outkd` | 2.3h | 8.1h |
| R2 | `K1_ukd` | 2.3h | 10.4h |
| R3 | `K2_uvkd` | 2.3h | 12.7h |
| R4 | `S0_shift` | 2.3h | 15.0h |
| R5 | `M1_uvs` | 2.3h | 17.3h |
| R6 | `M2_uvs_tf` | 2.3h | 19.6h |
| R7 | `M3_uvs_tf_warp` 또는 실패 진단 run | 2.3h | 21.9h |
| R8 | `R1_m2_seed` 또는 top-case repeat | 2.3h | 24.2h |
| R9 | `C1_m2_w96` 또는 두 번째 repeat | 2.1h | 26.3h |
| E0 | RR/FR/controlled-shift 평가, plot, contact sheet | 2.0h | 28.3h |
| BUF | 중단·재시작·추가 진단 여유 | 1.7h | 30.0h |

## 10.1 조건부 분기

### Teacher gate 실패

- `S0`, `M1`, `M2`, `M3`를 full run하지 않는다.
- audit pseudo-shift를 직접 사용하는 `oracle/pseudo cue` 1회로 shift idea의 상한만 측정한다.
- 남는 시간은 `K1`, `K2` second seed와 uncertainty calibration에 사용한다.

### M1의 shift MAE는 개선되지만 품질이 개선되지 않음

- `M2`를 진행한다.
- M2 이후에도 동일하면 `M3`에 shift-effect loss를 추가한다.

### M1의 shift MAE 자체가 개선되지 않음

- `M3`를 진행하지 않는다.
- shift sign, LR-to-PAN scale factor, confidence collapse, low-texture patch 비율을 점검한다.

### M2가 명확히 우세

- `R1_m2_seed`를 우선한다.
- repeat에서도 유지되면 `C1_m2_w96`를 수행한다.

---

# 11. 공통 학습 설정

sheet와의 직접 비교를 위해 아래를 유지한다.

```yaml
experiment:
  dataset: WV3
  iterations: 50000
  batch_size: 48
  effective_batch_with_MARs: 96
  optimizer: AdamW
  lr: 1.0e-4
  weight_decay: 0.01
  scheduler: cosine
  warmup_steps: 100
  crop: false
  normalization: LayerNorm
  input_channels: 11
  residual_base: provided_lms
  augmentation:
    hflip: true
    vflip: true
    rot90: true
  checkpoint_selection_internal: official_HQNR_FR_index_12_19
  also_save_final_50k: true

teacher:
  checkpoint: c0_hqnr
  width: 128
  depth: [2, 2, 4]
  cm3a: [enc, bottleneck, dec]
  freeze_for_student: true
  cache:
    enabled: true
    residual_dtype: float16
    uncertainty_dtype: float16
    shift_dtype: float32
    transform_with_flip_rotation: true

student_primary:
  name: d122
  width: 128
  depth: [1, 2, 2]
  attention: none

shift:
  coordinate_unit: LR_pixel
  search_radius: 3
  cost_volume_size: 7
  softmax_temperature: 0.07
  confidence_threshold: 0.35
  warp_fullres_scale: 4
  warp_mode: bilinear
  padding_mode: border
  teacher_channels: [16, 32, 32]
  student_channels: [8, 8]

uncertainty:
  output_channels: 1
  activation: softplus
  eps: 1.0e-4
  percentile_low: 10
  percentile_high: 90

variance:
  source: GT_minus_LMS
  window: 5
  percentile_low: 10
  percentile_high: 90
  alpha: 1.0

loss:
  lambda_soft: 0.1
  lambda_pan: 1.0
  lambda_shift: 0.25
  lambda_warp: 0.0
  confidence_loss_ratio: 0.1

teacher_forcing:
  eta_0_5k: 1.0
  eta_5k_20k: linear_1_to_0
  eta_20k_50k: 0.0
```

내부 checkpoint selection은 기존 sheet와 비교하기 위한 것이다. 논문용 최종 수치는 별도 validation 또는 fixed-final checkpoint로 다시 확인해야 한다.

---

# 12. 구현 순서

## 12.1 반드시 먼저 구현할 공통 함수

```python
mtf_down_pan(pan)                  # audit과 동일한 MTF/phase
build_pan_channels(pan_aligned)   # P, LP(P), P-LP(P)
warp_pan_lr_shift(pan, delta_lr)  # 내부에서 4× 변환
compute_gt_residual_variance(gt, lms, window=5)
normalize_by_running_percentile(x, q10, q90)
shift_cost_volume(edge_pan_lr, edge_ms, radius=3)
shift_from_posterior(prob)         # dy, dx, normalized entropy confidence
```

## 12.2 Student training pseudo-code

```python
# ms, lms, pan, gt: one RR training batch

# Teacher signal was generated once after Teacher adaptation.
r_t, u_t, delta_t, c_t = teacher_cache.load(patch_id)

# Apply exactly the same random flip/rotation to image maps and shift vector.
r_t, u_t, delta_t = transform_teacher_cache(
    r_t, u_t, delta_t, augmentation_state
)
y_t = lms + r_t

v_gt = compute_gt_residual_variance(gt, lms)
w_v = normalize_mean(1.0 + alpha_v * v_gt)

dy_s, dx_s, c_s = student_shift(pan, ms)
delta_s = stack([dy_s, dx_s])

eta = teacher_forcing_ratio(step)
delta_use = eta * stopgrad(c_t * delta_t) + (1.0 - eta) * (c_s * delta_s)

pan_s = warp_pan_lr_shift(pan, delta_use)
x_s_ms = concat(build_pan_channels(pan_s), lms)
y_s = student(x_s_ms, mode="MS")

# PAN mode remains the original raw-PAN MARs branch.
y_s_pan = student(build_original_11ch(pan, lms), mode="PAN")

r_gt = gt - lms
r_t = stopgrad(y_t - lms)
r_s = y_s - lms

l_hard = mean(w_v * (1.0 + u_t) * abs(r_s - r_gt))
l_soft = mean(w_v * (1.0 - u_t) * abs(r_s - r_t))

l_vec = mean(c_t * smooth_l1(delta_s, stopgrad(delta_t)))
l_conf = mean((c_s - stopgrad(c_t)) ** 2)
l_shift = l_vec + 0.1 * l_conf

l_pan = l1(y_s_pan, pan_target_repeated)

loss = l_hard + 0.1*l_soft + l_pan + 0.25*l_shift
```

## 12.3 gradient 관련 주의

- `delta_t`, `c_t`, `y_t`, `U_T`, `V_GT`는 모두 detach한다.
- `delta_s`에는 `L_shift`와 warp를 통한 reconstruction gradient가 모두 흐른다.
- PAN-mode branch는 첫 캠페인에서 shift module과 분리한다.
- teacher forcing 중에도 Student shift head는 `L_shift`로 계속 학습한다.
- border pixel은 warp loss와 shift evaluation에서 유효 overlap mask로 제외한다.

---

# 13. 평가 설계

## 13.1 표준 지표

### RR

- ERGAS
- SAM
- SCC
- Q2n/Q8
- 가능하면 PSNR, SSIM

### FR

- HQNR
- \(D_s\)
- \(D_\lambda\)
- audit fSCC

기존 sheet와 동일하게 전체 수치와 index 12–19를 모두 기록한다. audit에서 FR scene 0–11과 12–19의 shift 분포가 달랐으므로 두 구간을 분리해서도 보고한다.

## 13.2 Controlled-shift benchmark

RR test PAN에 알려진 shift를 추가한다.

| magnitude, LR px | directions |
|---:|---|
| 0 | identity |
| 0.25 | 8방향 |
| 0.5 | 8방향 |
| 1.0 | 8방향 |
| 2.0 | 8방향 |
| 3.0 | 8방향 |

각 모델에 대해 다음을 기록한다.

- Student shift MAE
- corrected PAN fSCC
- ERGAS/SCC degradation curve
- zero-shift no-harm
- 성능 곡선 AUC

## 13.3 Real FR shift agreement

Teacher/Student 예측과 audit estimator를 고신뢰 장면에서 비교한다.

\[
E_{shift}=\|\delta_{model}-\delta_{audit}\|_2
\]

센서별 또는 scene-group별 방향 편향을 확인한다.

## 13.4 Confidence calibration

confidence bin을 5개로 나누고 실제 shift error를 본다.

| confidence bin | mean predicted confidence | shift MAE | sample count |
|---|---:|---:|---:|
| 0.0–0.2 |  |  |  |
| 0.2–0.4 |  |  |  |
| 0.4–0.6 |  |  |  |
| 0.6–0.8 |  |  |  |
| 0.8–1.0 |  |  |  |

신뢰도가 높을수록 MAE가 단조 감소해야 confidence gate가 의미가 있다.

## 13.5 Uncertainty 검증

- \(U_T\)와 Teacher absolute error의 Spearman correlation
- uncertainty 상위 20%와 하위 20%의 평균 Teacher error
- \(U_T\)와 \(V_{GT}\)의 상관

\(U_T\)와 \(V_{GT}\)의 상관이 지나치게 높아도 두 신호가 완전히 동일한 것은 아니다. 최종 해석은 다음 기준을 따른다.

- \(U_T\): Teacher failure routing
- \(V_{GT}\): detail importance

---

# 14. Go / No-Go 기준

## 14.1 Teacher cue

| 항목 | Go |
|---|---:|
| Teacher synthetic shift MAE, ≤1 LR px | ≤0.12 |
| Student `M2` synthetic shift MAE | ≤0.20 |
| confidence high/low bin의 MAE 차 | high bin이 최소 30% 낮음 |
| zero-shift warp magnitude | median ≤0.05 LR px |

## 14.2 Student quality

w128급 기존 서버 변동을 고려해, 아래 중 하나를 만족해야 의미 있는 개선으로 본다.

- ERGAS relative improvement ≥0.5%
- SCC improvement ≥0.0003
- HQNR improvement ≥0.002
- \(D_s\) reduction ≥0.003
- controlled-shift AUC improvement ≥10%

단, standard RR이 0.5% 이상 악화되면 shift robustness만 좋아져도 주력 방법으로 채택하지 않는다.

## 14.3 주력 case 채택

`M2_uvs_tf`가 다음을 만족하면 후속 local residual 연구로 확장한다.

1. `K2_uvkd`보다 controlled-shift AUC 우세
2. `M1_uvs`보다 shift MAE 또는 standard metric 우세
3. `B0_lms` 대비 zero-shift no-harm
4. second seed에서 방향 재현

---

# 15. 실패 원인 진단표

| 관찰 | 가능 원인 | 다음 조치 |
|---|---|---|
| shift MAE가 처음부터 감소하지 않음 | 부호/4× scale 오류 | impulse 및 known-shift unit test 재실행 |
| \(c_S\)가 모두 0에 가까움 | posterior temperature가 낮거나 correlation이 평탄 | \(T_s=0.1\) 또는 encoder norm 점검 |
| \(c_S\)가 모두 1에 가까움 | entropy collapse | confidence loss 축소, temperature 증가 |
| shift MAE는 좋으나 ERGAS 악화 | PAN warp가 GT geometry와 반대 | direction overlay와 edge profile 확인 |
| shift MAE는 좋으나 fSCC 변화 없음 | vector는 맞지만 warp implementation 불일치 | `M3` shift-effect loss 수행 |
| U-KD가 output KD보다 나쁨 | uncertainty calibration 실패 | percentile, Spearman, hard/soft weight 분포 확인 |
| GT variance 추가 시 SAM 악화 | edge에 과도한 spectral weighting | \(\alpha_V:1.0\to0.5\) |
| FR HQNR만 상승하고 RR 악화 | metric-specific overfit | fixed 50K와 controlled-shift 결과 우선 |
| WV2에서 악화 | 저텍스처 false shift | threshold 상향 또는 identity gate 강화 |

---

# 16. 결과 기록 템플릿

## 16.1 Main table

| Run | ERGAS↓ | SAM↓ | SCC↑ | Q2n↑ | HQNR↑ | Ds↓ | Dλ↓ | shift MAE↓ | shift AUC↑ | Params | Train h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_lms |  |  |  |  |  |  |  | — |  |  |  |
| K0_outkd |  |  |  |  |  |  |  | — |  |  |  |
| K1_ukd |  |  |  |  |  |  |  | — |  |  |  |
| K2_uvkd |  |  |  |  |  |  |  | — |  |  |  |
| S0_shift |  |  |  |  |  |  |  |  |  |  |  |
| M1_uvs |  |  |  |  |  |  |  |  |  |  |  |
| M2_uvs_tf |  |  |  |  |  |  |  |  |  |  |  |
| M3_uvs_tf_warp |  |  |  |  |  |  |  |  |  |  |  |

## 16.2 Mechanism table

| Run | U routing | GT variance | shift KD | teacher forcing | warp loss |
|---|---|---|---|---|---|
| B0 |  |  |  |  |  |
| K0 |  |  |  |  |  |
| K1 | ✓ |  |  |  |  |
| K2 | ✓ | ✓ |  |  |  |
| S0 |  |  | ✓ |  |  |
| M1 | ✓ | ✓ | ✓ |  |  |
| M2 | ✓ | ✓ | ✓ | ✓ |  |
| M3 | ✓ | ✓ | ✓ | ✓ | ✓ |

## 16.3 필수 plot

1. shift magnitude 대 ERGAS/SCC curve
2. predicted shift 대 ground-truth shift scatter
3. confidence bin 대 shift MAE
4. Teacher uncertainty 대 actual error
5. GT variance, uncertainty, error map 3열 비교
6. raw/aligned PAN–MS edge overlay
7. baseline/M2의 double-edge crop 비교

---

# 17. 이번 30시간 후의 판정 문장

## 성공 시

> A lightweight PAN-sharpening student can acquire explicit global PAN–MS alignment behavior from a stronger teacher through a three-value shift token. Restoration uncertainty determines whether the student follows the teacher or the hard target, while GT residual variance concentrates the supervision on detail-sensitive regions. Scheduled teacher forcing stabilizes the transition from teacher-resolved alignment to autonomous student alignment.

## uncertainty/variance만 성공 시

> Uncertainty and GT residual variance improve detail-aware distillation, but the current global shift teacher does not yet provide sufficiently calibrated geometric cues.

## shift만 성공 시

> Explicit shift-token distillation improves geometric robustness, whereas the current uncertainty–variance weighting is redundant or insufficiently calibrated.

## 전체 실패 시

> The dominant limitation is not the absence of a simple global shift cue, or the cue estimator/warp coordinate system has not reached the required accuracy. Dense local alignment should not be introduced before this ambiguity is resolved.

---

# 18. 참고 자료

1. `2026-09-03_alignment-audit-s2-detail.md`
   - §II.4: bicubic half-pixel phase discrepancy
   - §II.5: global/local correction fSCC
   - §II.7: global-first Go/No-Go
   - §IV.D: train patch shift distribution
2. `pancrafter.pdf`
   - Sec. 3.1–3.3: MARs, CM3A, MS-mode geometry
   - Sec. 4.2: 50K training setup
   - Sec. 4.3: HRMS를 LRMS geometry에 맞추는 해석
3. `uknowdiff.pdf`
   - Sec. 3.2: positive uncertainty map과 heteroscedastic loss
   - Sec. 3.4: hard/soft/feature KD와 uncertainty routing
4. Google Sheet `pan-cvpr27` → `결과정리`
   - `c0_hqnr`, `d122`, `d122_w96`의 성능·parameter·runtime

