# PANDA 구현 변경 인계 — Scene Prior → Sensor/Dataset Global Prior (`b_global`)

> **작성일:** 2026-10-13  
> **문서 성격:** 기존 구현을 수정하기 위한 **변경 명세서**. 실험 case, ablation, 결과 예측, 실행 일정은 포함하지 않는다.  
> **우선 적용:** RR-supervised Stage 1 (PAN 64×64, LRMS 16×16, HRMS GT 64×64), Stage 2 frozen Aligner.  
> **기준:** `PANDA_Method_Implementation_Validation_Plan_2026-10-13_v1_3.md`, `PANDA_RR_ScenePrior_PatchResidual_Implementation_Handoff_2026-10-13.md`, `PAN_Epsilon_Offset_Method_2026-10-13.md`.  
> **상태 구분:** **[확정]** 대화에서 합의된 변경 · **[기존 확인]** 제공 자료/코드에 확인된 내용 · **[서버 확인]** 실제 구현 또는 데이터에서 확인해야 하는 사실 · **[적용 금지]** 이번 변경과 충돌하는 처리.

## 0. 변경 범위와 핵심 결정

### 0.1 왜 변경하는가

현재 PanCollection 학습 자료는 **RR 공간해상도의 PAN 64×64 / LRMS 16×16 / GT 64×64** patch를 제공하지만, 각각의 patch가 어느 원본 source scene에 속하는지 식별할 수 있는 **source-scene mapping이 없다**. 따라서 이전에 정의했던 `b_s`(scene마다 다른 global shift)를 각 학습 sample에 매핑하는 구조는 현재 데이터에서 구현할 수 없다.

**[확정]** 이를 **센서별 학습 patch 전체로부터 robust하게 얻는 단 하나의 고정 `b_global`**로 대체한다. 별도의 scene ID가 필요하지 않다. 개별 patch의 Aligner는 `b_global` 주변 residual을 학습한다.

**중요한 용어 변경:** 새 `b_global`은 **scene-level**이 아니라 **sensor-level / dataset-level global shift prior**다. `scene`을 `sensor`로 단순 치환하면서 마치 각 scene의 공통 물리적 변위를 추정한 것처럼 기술하지 않는다.

### 0.2 변경 전후

| 항목 | 이전 v1.3 | 이번 구현 변경 |
|---|---|---|
| Prior ID | source scene `s` | sensor/training dataset (`WV3`, `QB`, `GF2` 등) |
| Prior 수 | scene별 `b_s` | **해당 sensor 학습 집합마다 1개 `b_global`** |
| Prior 추정 입력 | 원본 source scene의 신뢰 구간 | **train split에 실제 저장된 RR PAN/LMS patch들의 개별 matching candidate** |
| 원본 patch→scene mapping | 필요 | **불필요** |
| Prior 값 | 고정 `(dy,dx)`, 방향+크기 | **고정 `(dy,dx)`, 방향+크기 유지** |
| Aligner head | `[h; b_s]` → `(α,β)` | **`[h; b_global]` → `(α,β)`**, shape 유지 |
| 최종 correction | `b_s + α d_s + β n_s` | **`b_global + α d_global + β n_global`** |
| Stage 1 FR consistency | 이전 전체안에 있었음 | **삭제**: 독립 FR training native PAN 미보유 |
| Stage 2 Aligner | freeze/shared | **유지** |
| U-Net/Aligner ResBlock | 합의한 LN-based common ResBlock | **유지** |
| 기타 Stage 1/2 loss | RR reconstruction/spatial/ε, Stage 2 e/q | **유지** |

**[적용 금지]** source-scene ID를 만들어내는 작업, RR test 256을 학습 prior의 source scene으로 사용하는 작업, FR test에서 gradient update하는 작업, 미정의 patch grouping, source mapping을 요구하는 코드의 silent fallback.

---

## 1. 데이터 및 좌표계 계약

### 1.1 각 데이터와 실제 역할

| 구분 | PAN | LRMS | LMS | GT | 의미 |
|---|---|---|---|---|---|
| `train`, `val` | `[N,1,64,64]` | `[N,C,16,16]` | `[N,C,64,64]` | `[N,C,64,64]` | RR-resolution 학습·검증 patch |
| `rr` test | `[N,1,256,256]` | `[N,C,64,64]` | `[N,C,256,256]` | `[N,C,256,256]` | **학습에 쓰지 않는 RR 평가** |
| `fr` test | `[N,1,512,512]` | `[N,C,128,128]` | `[N,C,512,512]` | **없음** | **학습에 쓰지 않는 native-resolution 평가** |

`C=8` for WV3, `C=4` for QB/GF2. `train PAN 64`와 `RR test PAN 256`은 같은 **RR GSD 계층**이며 단지 tensor 크기와 포함 영역이 다르다. `FR PAN 512`는 **native PAN GSD 계층**으로 RR과 pixel 단위의 물리적 의미가 다르다.

### 1.2 `b_global`의 단위와 부호

- **[확정]** 학습용 `b_global^RR`는 **RR PAN pixel** 단위의 `(dy,dx)` signed vector로 보관한다.
- **[기존 확인]** `warp_pan`은 `W(P,u)[y,x] = P[y+dy,x+dx]`의 **sampling-coordinate sign**을 사용한다. 양의 `dx`는 영상 내용이 왼쪽으로 이동하는 sampling convention이다.
- **[서버 확인]** 사용 중인 구조 matching estimator가 산출하는 displacement가 **MS→PAN**인지 **PAN→MS**인지 확인하고, 필요한 경우 inverse translation으로 변환해서 `b_global`을 저장한다. 저장 vector는 **그대로 `warp_pan(P,b_global)`에 적용하는 PAN→MS correction convention**이어야 한다.
- **[적용 금지]** 근거 없이 `abs(dy), abs(dx)`를 평균하거나 displacement magnitude만 평균하는 처리. 방향/부호가 반드시 보존되어야 한다.

### 1.3 `b_global` 산정에 사용하면 안 되는 데이터

Prior 추정은 **해당 sensor의 train split만** 사용한다. `val`은 검증, `rr`/`fr`은 평가용으로 분리한다. 특히 과거 ε radius 등의 산정에 evaluation input 통계가 포함된 이력이 있었지만 **새 `b_global`의 추정에 test input을 섞지 않는다**.

`GT`는 `L_abs`와 reconstruction supervision용으로 남긴다. 사전 matching prior는 **PAN ↔ LMS(동일 PAN grid의 train observation)**를 기준으로 계산하며, GT의 spatial 정보를 미리 prior로 가져오지 않는다. 따라서 prior 생성 자체가 reconstruction GT에 종속되도록 만들지 않는다.

---

## 2. Dataset-global prior 산정: 구현해야 할 새 preprocessing

### 2.1 입력 데이터와 전체 파이프라인

각 sensor `S`의 **원본 train patch(기하 증강 전)**를 대상으로 기존 구조 matching estimator가 candidate를 계산한다.

```text
TRAIN sensor S (RR grid)
  ├─ (P_1 64², LMS_1 64²) ── matcher ── candidate v_1, quality/confidence_1
  ├─ (P_2 64², LMS_2 64²) ── matcher ── candidate v_2, quality/confidence_2
  ├─ ...
  └─ (P_N 64², LMS_N 64²) ── matcher ── candidate v_N, quality/confidence_N
                                        │
                               validity + outlier handling
                                        │
                            robust vector aggregation
                                        │
                         b_global^RR(S) = (dy, dx)
                                        │
                               freeze / version / hash
```

**[확정]** `b_global`은 후보 변위들을 단순히 더한 **산술 총합이 아니다**. 여러 patch의 displacement를 **대표하는 2차원 벡터**다. 무조건 단순 평균을 택하기보다는 기존 audit/registration code에서 이미 근거를 가진 **신뢰도 기반 robust aggregation**을 재사용한다.

**[서버 확인]** 기존 estimator의 local matching, confidence 산정, outlier rejection, weighted median / RANSAC 등의 **구체 알고리즘과 설정은 실제 검증된 코드에서 채택한 그대로 기록**한다. 현재 대화에는 완전히 결정된 `match_window / confidence_cutoff / RANSAC tolerance / weighted-median 정의`가 없으므로 **이 문서가 숫자를 임의로 지정하지 않는다**. 구현 서버는 estimator 구현 해시 및 모든 설정을 asset manifest에 저장한다.

### 2.2 구체 산정 계약

1. Sensor별 train H5의 실제 `(pan,lms)`만 읽는다. `lms`의 band order/normalization을 기준 코드에 맞춰 검증한다.
2. 사전 matching 결과 `v_i`마다 **`(dy,dx)`, estimator direction, pixel grid, confidence, valid/rejection reason**을 남긴다.
3. candidate가 **MS-grid**로 보고됐다면 RR PAN-grid로 변환한다(4× scale과 pipeline phase 처리 포함). 여기서 scale 변환은 지상 공간단위 변환이고 interpolation offset을 임의 더하지 않는다.
4. 실제 warp sign으로 변환한 signed candidate만 aggregation한다.
5. 신뢰도/품질 조건을 만족한 후보만 대상으로 **robust 2-D consensus**를 구한다. 성분별 weighted median을 사용한다면 이를 명시하고, vector geometric median과 혼용하지 않는다. 기존 estimator가 RANSAC을 포함한다면 outlier 정의와 집계 순서를 고정한다.
6. prior asset을 별도 **train-only, read-only** 파일로 저장하고 센서별 고정된 `b_global`을 학습·평가에 로딩한다.
7. 유효 후보가 불충분하거나 최종 `b_global`의 의미가 불분명하면 `0`으로 임의 대체하지 말고 **fail-fast / 검증 필요**로 처리한다.

**신뢰성 한계:** 서로 다른 source scene의 shift 방향이 다르면 robust aggregate 자체가 희석되거나 일부 군집의 값으로 수렴할 수 있다. 이 `b_global`은 검증된 물리적 sensor calibration GT가 아니라 **고정된 학습 데이터 기반 initialization/reference**다. 여기서는 이를 바꾸기 위한 다른 실험 case를 제안하지 않는다.

### 2.3 필수 asset provenance

```yaml
schema: PANDA_B_GLOBAL_RR_V1
sensor: <WV3_or_QB_or_GF2>
train_source_path: <verified source path>
train_source_sha256: <actual file SHA256>
estimator:
  implementation: <exact function/code path>
  code_sha256: <actual code SHA256>
  settings: <exact config used>
  candidate_direction: <estimator's original direction>
  output_direction: PAN_TO_MS_WARP_CONVENTION
  candidate_grid: <PAN_RR|MS_RR>
  prior_grid: PAN_RR
aggregation:
  method: <exact verified robust method>
  valid_count: <integer>
  rejected_count: <integer>
  settings: <exact run settings>
prior_dy_dx: [<dy>, <dx>]
normalization_and_band_handling: <actual method>
created_from_splits: [train]
source_scene_mapping_used: false
frozen: true
```

기존 v1.3의 prior assets에서 **scene ID를 필수 키로 받던 schema**는 제거한다. 단, 단일 sensor를 넘어서 여러 센서를 하나의 모델에 혼합하는 새로운 설계는 이번 변경 범위에 없다.

---

## 3. 학습 sample에서 augmentation 적용 시 prior 변환

원본 train patch들은 augmentation을 거친다. 영상의 flip/rotation을 수행하고 `b_global`은 그대로 두면 displacement 방향이 어긋난다. 그러므로 **canonical `b_global^RR`을 train split의 원래 좌표계에서 계산한 후 sample augmentation에 맞춰 동일한 vector transform**을 적용한다.

`b`의 성분 순서는 항상 `(dy,dx)`다.

| 영상에 적용하는 변환 | prior 벡터에 적용 |
|---|---|
| Identity | `(dy, dx)` |
| Horizontal flip (`x` 반전) | `(dy, -dx)` |
| Vertical flip (`y` 반전) | `(-dy, dx)` |
| `np.rot90(k=1)` (반시계 90°) | `(-dx, dy)` |
| `np.rot90(k=2)` | `(-dy, -dx)` |
| `np.rot90(k=3)` | `(dx, -dy)` |

여러 augmentation이 이어지면 실제 loader가 영상에 적용한 **순서대로 vector transform을 compose**한다. 기존 `feeders/feeder.py`에는 H/V flip과 `np.rot90` 처리가 있으며 단순히 `flip 여부` 또는 `rotation k`만 일부 확인하는 것으로는 부족하다. **[서버 확인]** 실제 사용 중인 loader의 augmentation metadata/순서가 prior transform과 1:1로 대응하는지 확인한다.

이후 sample의 효과적 prior는:

\[
\mathbf b_i^{RR}=T_i(\mathbf b_{global}^{RR})
\]

이다. **sensor-level prior가 고정**이라는 말은 canonical asset이 학습되지 않는다는 뜻이다. augmentation된 입력에 사용할 때의 숫자는 augmentation에 따라 달라질 수 있다.

---

## 4. Aligner: 변경되는 output/head, 변경하지 않는 backbone

### 4.1 PAN Aligner 구성 [유지]

```text
PAN[B,1,H,W] ── stem Conv3×3 1→16 + GN + SiLU ─┐
                                                    ├─ concat 32
LMS[B,C,H,W] ── stem Conv3×3 C→16 + GN + SiLU ───┘
                         │
             joint Conv3×3 s2 32→32 + GN + SiLU
                         │
             joint Conv3×3 s2 32→64 + GN + SiLU
                         │
           U-Net과 공유하는 LN-based ResBlock(C=64)
                         │
                   GAP → h[B,64]
```

**[확정 유지]** Aligner의 64ch ResBlock은 reconstruction U-Net ResBlock과 동일한 구조로 유지한다: `ChannelLayerNorm→SiLU→Conv3×3→ChannelLayerNorm→SiLU→Dropout(0)→zero-initialized Conv3×3→skip-add`, 합산 뒤 SiLU 없음. Stem/joint의 GN은 **그대로** 둔다. Teacher/Student U-Net 자체의 기존 capacity·구조는 변경하지 않는다.

### 4.2 Prior basis + regression head [변경]

기존 v1.3의 `scene b_s` 입력을 **sensor-global `b_global` (학습 sample에서는 augmentation 변환된 `b_i`)**로 대체한다. 2차원 prior는 direction과 magnitude를 함께 포함한다.

\[
\mathbf d_i=\frac{\mathbf b_i}{\|\mathbf b_i\|_2},\qquad
\mathbf n_i=(-d_{i,x},d_{i,y})
\]

여기서 `b=(dy,dx)`, `d=(d_y,d_x)`, `n=(-d_x,d_y)`다. **[서버 확인]** 매우 작은 `||b||`에 대한 **basis fallback / fail-fast 처리**는 기존 v1.3에서 미결정이었던 부분이므로 실제 prior 값·현 구현을 확인해 결론을 기록한다. 정해지지 않은 상수를 조용히 더해 direction을 왜곡하지 않는다.

\[
 h_i=\operatorname{GAP}(\operatorname{CNN}(P_i,LMS_i))\in\mathbb R^{64}
\]

\[
 z_i=[h_i;\operatorname{sg}(\mathbf b_i)]\in\mathbb R^{66}
\]

\[
 (\alpha_i,\beta_i)=\operatorname{FC}_{32\to2}^{zero}
 (\operatorname{SiLU}(\operatorname{FC}_{66\to32}(z_i)))
\]

\[
\boxed{\mathbf u_i=\mathbf b_i+\alpha_i\mathbf d_i+\beta_i\mathbf n_i}
\]

즉 실제 regression head는 `FC(66→32) → SiLU → FC(32→2)`이며 **마지막 FC만 zero-init**한다. 이것은 `b_global`을 학습 가능 embedding으로 바꾸는 것이 아니다.

**[주의]** 같은 sensor에서 `b_global`은 canonical 상수이므로 head의 별도 conditioning 효과는 제한적이며, 주효과는 final correction에 더하는 fixed offset/initialization이다. 단, training augmentation 후에는 벡터 방향이 바뀔 수 있다. 이 구현상의 성질을 감추고 `scene-specific contextual guidance`라고 서술하지 않는다.

### 4.3 기존 구현과의 인터페이스 차이

```python
# 고정 prior asset [2] (dy,dx) for sensor S
b_sensor_rr = load_frozen_prior_asset(sensor)

# batch augmentation metadata를 동일하게 반영
b_batch_rr = transform_prior_by_augmentation(b_sensor_rr, augmentation_meta)  # [B,2]

# native and shifted epsilon branches BOTH receive the same b_batch_rr
u0 = aligner(pan, native_lms, b_batch_rr)
# ...
ue = aligner(pan_eps, native_lms, b_batch_rr)

# inside aligner model:
h = joint_conv_resblock_gap(pan, native_lms)        # [B,64]
z = concat([h, b_batch_rr.detach()], dim=1)         # [B,66]
ab = fc2(silu(fc1(z)))                               # [B,2] = alpha,beta
unit_direction, perp_direction = basis(b_batch_rr)   # each [B,2]
u = b_batch_rr + ab[:, :1]*unit_direction + ab[:, 1:]*perp_direction
```

**[적용 금지]** `b_global`을 patch별로 재추정하거나 학습 중 EMA로 갱신하지 않는다. Synthetic epsilon을 더한 PAN에서 `b_global`을 다시 matching하지 않는다. `epsilon`을 prior에 더해서 shifted branch conditioning을 다르게 만들지 않는다.

**Checkpoint 주의:** 기존 v1.3과 FC1 shape가 같아도 `scene b_s` 대신 `sensor b_global`이 공급되는 방식이 달라졌다. `strict=True` load 성공만으로 의미상 호환성을 보장하지 않는다. metadata 및 실제 conditioning 값 확인이 필요하다.

---

## 5. Stage 1: **RR-only** loss 및 gradient 경로

원본 학습은 RR PAN 64/LRMS 16/GT 64에서 수행한다. 현 데이터에는 별도 **FR native-resolution 학습 PAN**이 없으므로 Stage 1 학습에 `L_epsilon^FR`을 사용하지 않는다.

\[
\boxed{\mathcal L_{S1}=\mathcal L_{rec}^{RR}+\mathcal L_{abs}^{RR}+\lambda_{\epsilon}\mathcal L_{\epsilon}^{RR}}
\]

### 5.1 Native RR reconstruction

\[
\mathbf u_0=A_\phi(P,LMS,\mathbf b_i),\quad
\widetilde P=W(P,\mathbf u_0)
\]

\[
\hat Y_T=B_M+F_T(\widetilde P,B_M),\qquad B_M=\text{existing MS reconstruction base}
\]

\[
\mathcal L_{rec}^{RR}=\|\hat Y_T-Y\|_1.
\]

`B_M`은 기존 backbone recipe의 upsampled MS/residual base를 의미하며 **native LMS `R_A`와 반드시 같은 operator라는 뜻이 아니다**. Native LMS는 Aligner reference로 유지한다.

### 5.2 Absolute RR spatial supervision (SIPSA principle)

\[
Y_{lum}=\frac{Y_R+Y_G+Y_B}{3}
\]

\[
\boxed{\mathcal L_{abs}^{RR}=\big\|\,|\nabla \widetilde P|-|\nabla Y_{lum}|\,\big\|_1}
\]

`∇`는 기존 확정된 fixed 2D gradient operator로, `L_abs`와 Stage 2 `q`/Student spatial loss에서 동일하게 사용한다. 학습 sample의 실제 RGB index를 sensor band order로부터 명시적으로 선택한다.

### 5.3 RR epsilon consistency [기존 sampling 유지]

\[
P_{\epsilon}=W(P,\epsilon),\quad
\mathbf u_{\epsilon}=A_\phi(P_{\epsilon},LMS,\mathbf b_i)
\]

\[
\boxed{\mathcal L_{\epsilon}^{RR}=\operatorname{mean}|\mathbf u_{\epsilon}+\epsilon-\operatorname{sg}(\mathbf u_0)|}
\]

기존 `PAN_Epsilon_Offset_Method_2026-10-13.md`의 sampling 규칙은 변경하지 않는다.

\[
u_1,u_2\sim U(0,1),\quad
r=b\sqrt{u_1},\quad\theta=2\pi u_2,\quad
\epsilon=(r\sin\theta,r\cos\theta),\quad b=2\;\text{RR PAN pixels}.
\]

- Sample마다 2D vector 하나, 매 update 새로 생성, 0 중심 원판 면적 균등.
- MS/GT/LMS는 고정, PAN에만 synthetic warp 1회.
- `warp_pan`: bicubic, `padding=border`, `align_corners=False`, FP32.
- Native `u0` target은 **stop-gradient**; epsilon branch는 U-Net reconstruction으로 연결하지 않음.
- 이전 합의의 `lambda_epsilon=1e-3` 및 every update를 **현행 구현의 지정 초기 설정으로 유지**한다. 변경/실측 결과와 혼동하지 않는다.

### 5.4 Loss별 정확한 gradient routing

| Loss | Teacher U-Net `F_T` | PAN Aligner `A_φ` |
|---|---|---|
| `L_rec` | **update** | **update** (differentiable warp 경유) |
| `L_abs^RR` | no | **update** (warp 경유) |
| `L_epsilon^RR` | no | **update** (shifted Aligner branch) |

**[오해 금지]** “U-Net은 recon loss만 받고, PAN Aligner는 abs/epsilon loss만 받는다”가 아니다. **Aligner는 recon + abs + epsilon 모두 받는다.** `u0`의 stop-gradient는 오직 `L_epsilon`의 target branch에만 적용한다. `L_rec`과 `L_abs`의 native `u0`에서는 gradient를 끊지 않는다.

---

## 6. Stage 2: Aligner freeze, e/q routing **유지**

Stage 2의 정의를 변경하지 않는다. Stage 1에서 훈련된 **하나의** Aligner를 freeze하여 Teacher와 Student 모두 동일한 warped PAN을 사용한다. Student용 trainable Aligner clone은 없다.

\[
\hat Y_T=F_T(W(P,u_0),B_M),\quad
\hat Y_S=F_S(W(P,u_0),B_M)
\]

이 식에서 두 reconstruction network가 MS base를 더하는 residual 형태라면 **각각 동일한 base addition이 포함되어야 한다**. 실제 wrapper 출력과 비교한다.

Teacher reconstruction error:

\[
e(p)=\frac1C\sum_c|\hat Y_T^c(p)-Y^c(p)|,\quad
\tau_e=\mathrm{median}_{train}(e),\quad
w_e(p)=\frac{e(p)}{e(p)+\tau_e}
\]

Stage 1 spatial discrepancy map:

\[
q(p)=\big||\nabla \widetilde P(p)|-|\nabla Y_{lum}(p)|\big|,\quad
\tau_q=\mathrm{median}_{train}(q),\quad
w_q(p)=\frac{q(p)}{q(p)+\tau_q}
\]

Student soft, hard, spatial supervision:

\[
\mathcal L_{S2}=\mathbb E_p\left[(1-w_e(p))\,\ell_{soft}(p)+w_e(p)\,\ell_{hard}(p)\right]
+\lambda_S\mathbb E_p\left[w_q(p)\,\ell_{spa}^{GT}(p)\right]
\]

- `ell_soft`: Student ↔ frozen Teacher HRMS prediction L1.
- `ell_hard`: Student ↔ HRMS GT L1.
- `ell_spa^GT`: Student RGB mean luminance ↔ GT RGB mean luminance의 absolute 2D gradient L1.
- `tau_e`,`tau_q`: train distribution median으로 사전 산정 후 고정.
- `lambda_S`: 기존 합의 초기 설정 1, 여기서는 변경하지 않음.
- `MIND`: 평가/진단용, train loss에 추가하지 않음.

이 문서는 `q`를 과거 epsilon-response 기반 `q_old`로 되돌리지 않는다. **현재 합의된 q는 absolute-gradient discrepancy**다.

---

## 7. RR/FR 평가·추론에서 b_global 사용: 구현상 주의

### 7.1 RR evaluation (PAN 256)

RR 평가 영상은 **train과 같은 RR pixel GSD 계층**이다. 따라서 prior의 물리적 의미가 같다는 데이터 전제가 성립하면 canonical `b_global^RR`를 RR 평가에 그대로 적용할 수 있다. 테스트 영상에 대해 prior를 다시 fitting하거나 train prior aggregation에 test를 추가하지 않는다.

- Whole-image Aligner forward: `PAN 256`에 대해 **영상 하나당 residual `(α,β)` 한 쌍**, 즉 2D translation 하나를 출력한다.
- `64×64`에서 학습했다는 이유로 RR 256을 내부적으로 64 tile 단위로 자동 분해하지 않는다.
- 이 모델은 **global translation estimator**이지 dense nonrigid local flow가 아니다.

### 7.2 FR evaluation (PAN 512)

FR 평가에서는 별도 FR input이 있지만 **GT와 독립적인 FR training pair는 없다.** 따라서 `L_epsilon^FR`을 학습에 넣지 않는다. 

FR PAN pixels는 RR PAN pixel보다 finer grid이므로, **동일 물리적 이동**을 단위 변환하는 경우 이론적으로:

\[
\mathbf b_{global}^{FR} \approx 4\mathbf b_{global}^{RR}
\]

(해당 센서의 PAN/MS resolution ratio=4 가정). **이는 pixel-unit 변환이지 서로 다른 source scene의 실제 센서 misalignment가 동일하다는 증거가 아니다.** 4배 변환을 자동 적용할지, 실제 구현이 다른 normalization/coordinate contract를 사용하는지는 서버의 확정 규칙으로 명시하고 반드시 검증한다. 이 문서는 FR prior의 절대값이 정확하다고 가정하지 않는다.

**[서버 확인 — 구현 차단 항목]** FR에서 사용할 `b_global` policy(`current-grid scaling`의 정확한 좌표계, factor 근거, 미확정인 경우 처리)를 기록하지 않은 상태로 묵시적으로 `b_RR` 또는 `4b_RR`를 공급하는 구현을 금지한다. RR에서 학습한 raw `b` 입력을 FR에서 크게 만드는 것은 FC1 conditioning distribution도 바꾸므로 별도 확인이 필요하다.

FR에서는 기존처럼 HQNR, `D_s`, `D_lambda`와 ε response 등을 **학습 없이 evaluation/diagnostics로만** 계산한다. 여기서는 새로운 실험 case를 설계하지 않는다.

---

## 8. 기존 코드와의 변경점 목록 (실험 계획 아님)

| 영역 | 해야 할 변경 | 그대로 둘 것 |
|---|---|---|
| Source data | train PAN/LMS patch 접근 및 provenance/augmentation metadata 검증 | train/val/RR/FR split 경계 |
| New preprocessing | **sensor별 `b_global` 추정 및 frozen asset 저장** | 기존 검증된 structural matching estimator (세부 구현 확인) |
| Prior lookup | `source_scene_id → b_s` lookup 제거, `sensor → b_global` lookup으로 대체 | signed `(dy,dx)` convention |
| Prior augmentation | `b_global`에 정확한 spatial augmentation transform 적용 | image augmentation 자체 |
| PAN Aligner input | `prior [B,2]` 전달, `fc1 66→32` 유지 | PAN/LMS stem, joint Conv, GAP |
| Aligner output | `u=b_global+α d_global+β n_global` | U-Net 공통 LN-based ResBlock, FC2 zero-init |
| Stage 1 dataset | **RR train patch만 학습** | native LMS, RR reconstruction/GT |
| Stage 1 loss | **`L_rec+L_abs^RR+λε L_epsilon^RR`** | ε distribution, warping convention, stopgrad anchor |
| Stage 1 FR path | **train loop의 `L_epsilon^FR` 제거** | FR eval/diagnostics 가능 |
| Stage 2 | **shared frozen Aligner** | `e/q` median reliability routing, Student loss |
| Evaluation | RR 256/FR 512 구분, FR prior scaling policy 기록 | official metric 구현/평가 split |
| Manifest | 센서 prior asset SHA, matcher code/config, transform, model config 기록 | reproducible run identities |

### 8.1 코드 레벨 점검 대상으로 이미 알려진 파일

실제 수정 파일명/branch는 서버에서 확인한다. 다음은 기존 코드베이스에서 대응이 알려진 위치다.

```text
pa/aligner.py            # PANGlobalAligner forward; ResBlock은 v1.2 결정을 반영한 구현 확인
pa/offset.py             # uniform-disk sampling, ε loss helper; 오래된 PO10 전체 train loop 재사용 주의
pa/warp.py               # warp_pan sampling convention
feeders/feeder.py        # train augmentation, RR/FR H5 shape 및 channel order
ta2/data.py              # train/rr/fr split provenance, native LMS
model/pancrafter_paper.py# shared LN-based ResBlock, reconstruction U-Net (변경 대상 아님)
<active training module> # Stage1 RR-only total, gradient routing, training dataset
<active student module>  # Stage2 A freeze, e/q computation
```

**중요:** 기존 `pa/aligner.py`의 원본 64ch ResBlock이 GN/post-SiLU인 파일은 이 프로젝트에서 **이미 합의한 LN-based common ResBlock을 반영해야 하는 상태**였으므로, 새 `b_global` 변경 때문에 우연히 구 ResBlock으로 되돌리지 않는다.

---

## 9. 필수 구현 검증 — 실험 case와 구분

이 절은 모델 성능 비교 실험이나 ablation 계획이 아니라 **바른 코드가 실행되는지 확인하는 validation contract**다.

- **Source integrity:** 각 sensor의 `b_global`은 train split만으로 계산하며 train source hash/estimator hash를 asset에 기록한다.
- **Estimator sign:** synthetic impulse/known translation에서 matcher vector → `warp_pan` correction의 dy/dx sign이 일치하는지 확인한다.
- **Vector consensus:** 단순 scalar magnitude 평균이 아닌 **signed 2D shift**를 집계하며 invalid candidates가 포함되지 않는지 확인한다.
- **Augmentation:** flip/rot90 후 저장 prior vector가 image warp와 같은 좌표 transform을 갖는지 확인한다.
- **Head input:** `[B,64] + [B,2] → [B,66]`, 실제 마지막 2채널이 `T_i(b_global)`인지 forward hook으로 확인한다.
- **Initialization:** FC2 zero-init 때 `α=β=0`이고 `u_i=b_i`가 정확히 성립해야 한다.
- **Boundary cases:** `||b||` near zero, non-finite prior, large prior, crop support 및 warp border behavior를 서버에서 명시적으로 처리한다.
- **ε target:** `c0`와 `c_epsilon`은 정확히 **동일한** prior를 공유하고, `c0` detach는 ε loss에만 적용된다.
- **Gradients:** `L_rec → U-Net+Aligner`, `L_abs → Aligner only`, `L_epsilon → shifted Aligner only`.
- **Training split:** FR data는 train optimizer에 접근하지 않으며 RR/FR test input이 `b_global` 추정에 섞이지 않는다.
- **Stage 2 freeze:** Aligner state hash가 Stage 2 전후 일치하고 Teacher/Student corrected PAN이 동일해야 한다.
- **RR/FR scales:** RR 평가의 prior 단위는 RR PAN pixels, FR은 explicit GSD-based conversion policy 하에서만 사용한다.

---

## 10. 구현 종료 조건과 미해결 서버 확인 사항

### 10.1 이번 변경을 구현했다고 말할 수 있는 조건

- **sensor별 `b_global` frozen asset**이 train patch-only provenance와 함께 생성되어 있음.
- Scene-ID 의존성 없이 학습을 실행할 수 있음.
- Aligner 입력/출력에서 `b_global`가 direction+magnitude를 포함한 `(dy,dx)`로 사용되고 final correction은 prior+residual임.
- Stage 1 RR-only loss가 정확하게 연결되고 `L_epsilon^FR` 학습 gradient가 없음.
- Stage 2 frozen shared Aligner와 기존 e/q fitting이 유지됨.
- 기본 shape/gradient/augmentation/sign tests가 통과하고 설정·코드 identity가 저장됨.

### 10.2 실제 서버에서 확인해야 하는 **미결정 구현 세부**

다음은 현재 주어진 자료만으로 수치를 확정할 수 없으므로, **실제 사용 중인 구현/데이터를 확인해 기록**한다. 임의 기본값 생성 금지.

1. Train patch matcher의 정확한 descriptor, output sign, confidence mask, robust aggregation 방식과 하이퍼파라미터.
2. `b_global` valid candidate 수, zero/near-zero magnitude일 때 basis 처리를 어떻게 정의할지.
3. 실제 branch의 U-Net input layout(Teacher `P0`, 일부 Student `PLH` 여부), native LMS/upsampled MS의 입력 구분.
4. 실제 학습 코드가 이미 LN-based Aligner ResBlock을 반영했는지, 두 zero-init layer의 초기 gradient 상태.
5. 실제 augmentation 구성과 metadata 저장 규칙 및 deterministic b vector transform.
6. FR inference의 prior unit-conversion policy 및 4× 가정의 실제 적용 가능성.
7. Stage 2 `q`와 `e` scalarization, training median 산정 및 gradient detach/caching의 실제 구현.

**주의:** 미결정 항목은 정확성 확인 없이 억지로 채우지 않는다. 이 문서는 **method 변경을 위한 implementation handoff**이지 새로운 실험 case 등록 문서가 아니다.
