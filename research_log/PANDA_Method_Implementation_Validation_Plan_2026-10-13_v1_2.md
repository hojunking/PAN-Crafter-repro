# PANDA — Stage 1/2 방법론 구현·검증·실험 기획서

**부제:** Scene-level Global Translation Prior + Patch Residual PAN Aligner / RR–FR Shift Equivariance / Reconstruction–Spatial Reliability Adaptive Student Fitting  
**문서 버전:** `MethodPlan-v1.2 / 2026-10-13`  
**목적:** 다른 세션과 실제 구현·실험 서버가 **이번 대화에서 합의한 방법을 누락·자의적 변경 없이 검증하고**, 검증 결과에 근거해 구현·실험 계획을 확정하도록 하는 기술 인계서  
**성격:** *연구 설계안 + 구현 명세 후보 + 검증 체크리스트*. 실제 코드가 이미 이 방법을 구현했다는 뜻이 아니며, 미검증 결과를 성능 사실로 기술하지 않는다.

**v1.2 수정 명세:** 사용자 의도는 **Teacher와 Student U-Net의 ResBlock을 새로 통일하라는 것이 아니라, PAN Aligner의 64채널 residual block을 기존 reconstruction U-Net의 ResBlock과 동일하게 통일하는 것**이다. Teacher/Student의 기존 U-Net architecture와 capacity는 유지한다. 통일 범위는 **Residual block 내부**이며 PAN/MS stem·joint conv까지 LN으로 바꾸는 것이 아니다.

> **가장 중요한 구분**
>
> - `[FIXED]`: 사용자와 합의한 **방법론상의 고정 결정**. 임의 변경 금지.
> - `[CURRENT]`: 제공된 기존 자료 또는 조회한 저장소 코드에서 **실제로 확인된 구현/측정**. 새 방법이 구현됐다는 뜻은 아님.
> - `[PROPOSED]`: 고정 방법을 구현하기 위한 구체적 연결·운영안. 채택 전 서버에서 원리 및 호환성 확인.
> - `[VERIFY]`: 근거가 불충분하거나 구현·데이터 의존적이므로 반드시 확인할 항목. 숫자·알고리즘을 임의로 채우지 않음.
> - `[EXCLUDED]`: 이번 baseline에서 명시적으로 제외된 요소.
>
> 본 문서의 `FIXED`와 `VERIFY`를 혼합하면 **실험할 가설을 이미 검증된 사실처럼 취급하는 오류**가 발생한다. 구현을 맡은 세션은 먼저 §20의 blocking 항목을 확인한다.

---

## 0. 한 페이지 요약 / 최종 결정사항

### 0.1 Stage 1 — Scene-guided geometry learning

**Scene-level global shift는 방향과 크기를 모두 갖는 벡터**로 제공한다. 기존 structural/edge matching은 *scene prior 산정용*으로만 사용하며 학습 대상이 아니다. Aligner는 입력 patch에 대해 추가 residual correction을 학습한다.

\[
\boxed{\mathbf b_s=(b_y,b_x)\ \text{(fixed, scene-level, direction + magnitude)}}
\]

\[
\boxed{\mathbf c_{s,i}=\mathbf b_s+\alpha_{s,i}\mathbf d_s+\beta_{s,i}\mathbf n_s}
\]

이 correction으로 **PAN만** MS 기준 좌표로 이동한다.

\[
\boxed{\mathcal L_{\rm S1}=\mathcal L_{\rm rec}^{RR}+\mathcal L_{\rm abs}^{RR}+\lambda_\epsilon\big(\mathcal L_\epsilon^{RR}+\mathcal L_\epsilon^{FR}\big)}
\]

- `[FIXED]` Aligner의 MS reference는 데이터셋 `lms` 형태의 **native LMS**를 우선 사용한다. reconstruction용 LRMS upsample/base와 혼동하지 않는다.
- `[FIXED]` `L_rec`: **RR native correction**에 대한 HRMS GT reconstruction L1.
- `[FIXED]` `L_abs^RR`: **RR native corrected PAN** ↔ **GT HRMS RGB 평균 luminance**, SIPSA-Net 계열 **절댓값 2-D gradient 간 L1**. ε-shifted PAN에 이 loss를 적용하지 않는다.
- `[FIXED]` `L_ε^RR`,`L_ε^FR`: **각 해상도에서 같은 Aligner의 known-translation response consistency**. FR에는 HRMS GT를 사용하지 않는다.
- `[FIXED]` `ε`: 원점 중심 **원판 면적 균등**, **반경 `b=2.0` current PAN pixels**, sample당 전역 2-D translation 하나, update마다 새 표본, native PAN에 1회 bicubic warp.
- `[FIXED]` ε branch **매 update**. 기본 시작 가중치 **`λ_ε=1e-3`**. 이 값은 기존 완료 실험의 최적 검증값이 아니라 **새 baseline 초기 후보**다.
- `[FIXED]` RR↔FR 직접 scale-consistency loss **없음**.
- `[FIXED]` 별도 scene-prior loss, 직교 residual 규제, learned scene embedding **없음**.
- `[FIXED]` **PAN Aligner 내부 64-channel ResBlock을 reconstruction U-Net의 ResBlock과 동일하게 변경**한다: Channel-wise LN, SiLU, Conv3×3 두 개, 두 번째 Conv zero-init, residual 합산 후 activation 없음. Aligner stem/joint의 GN은 유지한다. (§4.2·§8.2)

### 0.2 Stage 2 — Frozen geometry + reliability-adaptive Student fitting

- `[FIXED]` Stage 1 Aligner **하나를 고정(freeze)**하여 Teacher와 Student가 동일한 corrected PAN을 사용한다. Student Aligner clone/fine-tuning **없음**.
- `[FIXED]` **Teacher/Student reconstruction U-Net의 기존 아키텍처는 유지**한다(현재 코드 candidate는 공통 `PANCrafterPaper.ResBlock`, Channel-wise LN; width/depth는 역할별 기존 설정). **새 변경 대상은 Stage 1 PAN Aligner의 ResBlock**으로, 위 U-Net 공통 블록과 같은 연산·정규화·초기화를 적용한다. 블록 정의를 논문/기획서에서 한 번만 설명하기 위한 변경이며, 성능 영향이 없다고 단정하지 않는다.
- `[FIXED]` Teacher reconstruction error `e`로 **soft/hard 연속 blending**; Student error/Teacher–Student gap으로 routing하지 않는다.
- `[FIXED]` **새 정의의 `q`**는 Stage 1 `L_abs^RR`의 reduction 이전 spatial-gradient discrepancy. 기존 ε-response 기반 `q_old`와 **다른 양**이다.
- `[FIXED]` `q`가 GT RGB spatial supervision의 강도를 제어한다.
- `[FIXED]` `τ_e=median_train(e)` 및 `τ_q=median_train(q)`를 사전 산정하고 고정한다. Stage 2 spatial multiplier **`λ_S=1` 초기값**.

\[
\boxed{\mathcal L_{\rm S2}=\mathbb E_p\big[(1-w_e)L_{\rm soft}(p)+w_e L_{\rm hard}(p)\big]+\lambda_S\mathbb E_p\big[w_q(p)L_{\rm edge}^{GT}(p)\big]}
\]

\[
w_e(p)=\frac{e(p)}{e(p)+\tau_e},\qquad
w_q(p)=\frac{q(p)}{q(p)+\tau_q}.
\]

### 0.3 현재 방법에서 **하지 않는 것**

`L_scale`, RR–FR output displacement 일치 강제, hard 0/1 switch, Teacher–Student advantage gate, Student Aligner fine-tune, scene direction-only baseline, magnitude 학습, 추가 FR GT matching loss, MIND 학습 loss, learned uncertainty head, sensor별 ε radius, ε ramp, Stage 2용 새로운 ε loss **모두 제외**한다. 향후 비교 실험이 필요할 때만 각각 독립적으로 부활시킨다.

---

## 1. 근거 문서, 코드, 사실의 범위

### 1.1 1차 인계 자료

| 근거 | 용도 | 문서가 보장하지 않는 것 |
|---|---|---|
| `PANDA_Stage1_Stage2_Final_Method_2026-10-13.md` | 이번 대화에서 합의된 Stage 1/2 설계, 3개 top-level loss, scene magnitude 포함, Stage 2 freeze, `e/q` 정의 | 구현 완료/성능 개선의 증거가 아님 |
| `PAN_Epsilon_Offset_Method_2026-10-13.md` | **기존 ε 구현의 가장 자세한 기준**: sampling, warp sign, generator, 주기/가중치 이력, probe, 알려진 한계 | 새 `L_ε^FR` 결합 훈련이 이미 수행됐다는 의미 아님 |
| `2026-09-03_alignment-audit-s2-detail.md` | PAN/MS 상대 위치·위상·spectral floor·within/across scene 방향분포·robust local field 근거 | scene 원인(지형/리샘플링)을 물리적으로 규명하지 않음 |
| `Pansharpening_research.pdf`, 11–15쪽 | **기존** `q_old/e` 4분면 및 Stage 2 학습 구조 | 새 `q_new`의 4분면 유효성을 증명하지 않음 |
| `ICLR2027_pansharpening____Reviewed_before_Experiment.pdf` | 기존 PANDA 초안의 Stage 1/2 정의 및 reconstruction route | 이번에 결정된 방법과 다름. Fig. 5 번호/내용도 재확인 필요 |
| `Lee_SIPSA-Net_...pdf`, §3.3 | RGB 평균 luminance + absolute 2-D gradient 차이의 spatial loss 원리 | PAN을 MS frame으로 이동시키는 PANDA와 reference frame 방향이 다름 |

### 1.2 2026-10-13 코드 조회에서 확인된 **기존** 인터페이스

아래는 `hojunking/PAN-Crafter-repro` 저장소 `main` 파일을 이 문서 작성 중 조회한 결과다. 실제 실험 서버의 branch/commit이 같은지는 **반드시** 확인한다.

| 기존 파일 | 확인 사항 | 조회한 blob SHA |
|---|---|---|
| `pa/aligner.py` | **현재 구현의 residual64는 GN/ReflectionPad/합산 후 SiLU 기반으로 U-Net 블록과 다름**; dual stem 1→16/C→16, joint 32→32→64, GAP, FC64→32→2, input z-normalization | `e9dd406ae7f0a2dc491273e21b38f8e450dfa28d` |
| `pa/warp.py` | `P̃(y,x)=P(y+dy,x+dx)`, `grid_sample` bicubic/border/`align_corners=False`, FP32; support-mask helpers | `7dc6b3bb83fa019ed5f6331e02ef5d5032b7a966` |
| `pa/offset.py` | `sample_offsets` 원판분포, `sg(c0)`; **오래된 PO10 학습 루프도 포함** | `1029d50299bd5c813d8f0a859b3908daf3727ca7` |
| `ta2/diagnostics.py` | ε probe AXIS16/DIAG16/DISK64/OOD8, gain·MAE·EPE | `4324dd0f335a073ae670ffb8877d10cd33ff994f` |
| `ta2/losses.py` | 일부 별도 실험에서 NCC/NGF structural objective; **이번 SIPSA loss와 동일하지 않음** | `2f7cb04246bb80c61533632fa566a8378117ee69` |
| `ta2/model.py` | 기존 Teacher-only U112/D(1,2,3), 별도 reference 선택; 이번 prior-aware Aligner 미구현 | `465d7f70ff593e412ace8e3a15e24d9cd53b2b2e` |
| `ablr2/model.py` | 구 실험군 Teacher112/Student104, depth T(1,2,3)/S(1,2,2), 일부 Student LPAN/HPAN | `590fda6306c3bc1b1b8d71dc47215b112dab062c` |
| `model/pancrafter_paper.py` | **Aligner로 이식할 동일 ResBlock의 기준 구현**: ChannelLayerNorm, pre-norm ResBlock, zero-init 2번째 Conv, residual sum 뒤 activation 없음. Teacher/Student U-Net 자체는 유지 | `4d02d41f2f441fa97948d88902340d169ce09c76` |
| `model/pancrafter.py` | 공통 DownConv(3×3/s2), UpConv(2×2 transpose/s2), `zero_module` 및 저자 배포 ResBlock(GN)의 **차이 확인** | 실험 서버에서 pin |

**주의:** `pa/offset.py`의 PO10 경로에는 shifted PAN으로 U-Net을 reconstruction하는 **구버전의 교대 학습 루프**가 있다. 이를 새 baseline에 그대로 재사용해서는 안 된다. 새 baseline의 reconstruction은 native PAN path만 사용한다. 기존 `ta2/losses.py`의 NCC/NGF는 합의한 SIPSA-style absolute-gradient L1으로 **대체/별도 구현**해야 하며, 이름이 비슷하다고 혼용하지 않는다.

---

## 2. 좌표계와 shape 규약 — 구현 전 강제 확인

### 2.1 네 종류의 격자를 혼동하지 않는다

- **Native sensor PAN grid**: 실제 PAN의 픽셀 격자. 예: FR 512×512.
- **Native observed LRMS grid**: 실제 센서의 LRMS 격자. 예: FR 128×128, 일반적으로 PAN/MS 비율 4.
- **PAN-grid LMS**: `lms`로 제공된 pre-upsampling observation. PAN과 **같은 텐서 크기**지만 **HRMS GT가 아니다**.
- **Reconstruction base `U(M)`**: reconstruction network가 사용하는 LRMS 보간/잔차 base. 기존 코드의 bicubic과 같을 수 있으나 PAN Aligner의 native LMS reference와 같은 텐서라고 가정하면 안 된다.

### 2.2 Dataset/branch tensor 예시

| 용도 | PAN `P` | LRMS `M` | PAN-grid reference `R_A` | HRMS GT `Y` | 비고 |
|---|---|---|---|---|---|
| RR train patch 예시 | `[B,1,64,64]` | `[B,C,16,16]` | `[B,C,64,64]` | `[B,C,64,64]` | RR supervised |
| RR eval 예시 | `[B,1,256,256]` | `[B,C,64,64]` | `[B,C,256,256]` | `[B,C,256,256]` | RR supervised evaluation |
| FR/native 예시 | `[B,1,512,512]` | `[B,C,128,128]` | `[B,C,512,512]` | **없음** | FR ε training/evaluation |
| native 512 pair를 한 단계 축소한 RR 예시 | `[B,1,128,128]` | `[B,C,32,32]` | `[B,C,128,128]` | 대응되는 MS-level target가 있을 때만 | 같은 FOV 축소 예시, 공식 RR test와 **구분** |

위의 shape는 **예시**다. 실제 loader의 shape, crop의 물리적 범위, PanCollection의 Wald degradation 파이프라인, `lms` 생성/저장 상태를 서버에서 기록한다. **512→256은 자동으로 ‘RR↔FR 2배 scale’이 아니다.** 이미지 크기와 물리적 sampling ratio는 다르다.

### 2.3 고정 reference frame

\[
\boxed{\text{MS/GT spatial frame 고정, PAN만 warp}}
\]

Aligner는 `(PAN, native LMS)`를 같은 좌표 크기로 받아 PAN을 **MS frame에 맞추는 correction**을 예측한다. Teacher/Student는 보정된 PAN과 기존 reconstruction base를 사용한다. **보간영상의 위상과 reconstruction target의 실제 geometry는 별도 점검한다.**

### 2.4 정확한 warp sign

`pa/warp.py` 기준:

\[
\widetilde P(y,x)=P(y+dy,x+dx),\quad\mathbf c=(dy,dx).
\]

`dx>0`이면 **PAN 내용은 화면 왼쪽으로** 이동한다. 

Normalized `grid_sample` 좌표 (`align_corners=False`):

\[
g_x=\frac{2(x+dx+1/2)}{W}-1,\qquad
 g_y=\frac{2(y+dy+1/2)}{H}-1.
\]

기존 audit의 매칭 vector는 일부 비교에서 `MS→PAN`을 반환한다. 새 Aligner prior는 `PAN→MS` 보정량이므로 **mathematical inversion/sign audit을 하지 않은 vector를 그대로 prior로 넣으면 안 된다**.

### 2.5 학습 augmentation과 shift 벡터

`flip`, `rot90`, crop이 원본 PAN/MS에 적용되면 **scene prior도 같은 좌표변환을 거쳐야 한다**. 기본 원리:

\[
\mathbf b_s^{aug}=T_{aug}(\mathbf b_s),\qquad
\mathbf d_s^{aug}=T_{aug}(\mathbf d_s).
\]

수평/수직 flip 및 90도 회전의 `T_aug`는 **실제 image augmentation의 축/회전 순서**로부터 작성하고 impulse test로 검증한다. 벡터는 위치값이 아니라 translation이므로 crop 원점 이동에 의한 추가 translation은 없어야 하지만, resize/sampling-origin 차이는 별도다. **augmentation된 영상에 augmentation 이전의 원방향을 조건으로 주는 버그를 최우선 점검한다.**

---

## 3. Scene-level global prior — 정의, 추정, 전달

### 3.1 Global + residual 가정

scene `s`, patch `i`에서 필요한 correction을:

\[
\boxed{\mathbf c_{s,i}=\mathbf b_s+\Delta\mathbf u_{s,i}}
\]

로 분해한다. `b_s`는 같은 source scene의 대표 correction이며  `Δu`는 patch별 **하나의 2-D translation**이다. **patch pixel별 dense-flow를 예측하는 모델이 아니다.** `local`은 *patch-to-patch variable*이라는 뜻이다.

### 3.2 Prior가 반드시 direction + magnitude인 이유

\[
\mathbf b_s=(b_y,b_x),\quad m_s=\|\mathbf b_s\|_2.
\]

Direction-only 방식은 회귀좌표계를 회전시키는 효과는 있지만 `α,β`가 자유로우면 전역 shift 크기의 초기 anchor를 제공하지 않는다. 이번 설계에서는 **scene의 대표 absolute magnitude까지** 제공하여, 특히 HRMS GT가 없는 FR에서 **native global offset을 residual network가 처음부터 발견할 부담을 줄이는 것**을 목표로 한다. 이 효과는 **가설**이며 실험으로 검증해야 한다.

### 3.3 Off-line estimator (새 loss 아님)

현재 audit의 개념적 순서:

```text
native scene PAN + observed MS/LMS
  -> modality 차이에 상대적으로 견고한 spatial representation
  -> 신뢰할 수 있는 local displacement candidates
  -> local confidence/outlier filtering
  -> robust scene-level translation aggregation
  -> canonical b_s, confidence, provenance
```

기존 audit가 사용한 방법의 구성요소: Scharr gradient magnitude, median/MAD 정규화, structural window 선별, ZNCC matching, census cross-check, weighted median + RANSAC 기반 robust aggregation. **이 조합의 정확한 호출 순서·hyperparameter·기준 좌표계는 현 서버 코드/기존 audit로 검증한다.** 특정 외부 논문의 표준 단일 알고리즘인 것처럼 기술하지 않는다.

출력으로 최소한 다음이 필요하다.

| 필드 | 용도 |
|---|---|
| `source_scene_id` | 같은 scene의 patch들을 구분하는 불변 ID |
| `prior_shift_dy_dx` | **PAN→MS sampling correction**, canonical grid의 2-vector |
| `prior_grid_name`, `pixel_scale`, `origin_phase` | RR/FR 변환 근거 |
| `n_windows`, `n_valid`, `inlier_fraction` | estimator evidence |
| `direction_concentration`, `dispersion`, `confidence` | global prior의 대표성/불확실성 진단 |
| `estimator_version`, `reference_type`, `scene_input_hash` | 재현성 |
| `mapping_rule` | pair-C 등으로부터 부호·해상도 변환한 절차 |

**임의값 금지:** confidence cutoff, RANSAC tolerance, estimator window size, expected shift 범위는 새 방법론에서 확정하지 않았다. **기존 감사(audit) 구현을 우선 검증/재사용하고, 값이 없으면 서버에서 근거를 기록한 뒤 결정**한다.

### 3.4 Scene source identity와 데이터 접근 — Blocking

기존 20개 benchmark FR patch의 `scene index`가 곧 원본 acquisition scene ID라는 근거는 없다. 학습 patch가 **어떤 native source scene의 어떤 coordinate crop인지** 추적할 수 있어야 `b_s`를 올바르게 주입할 수 있다. 서버는 다음을 확인한다.

- RR training patch → native source scene ID/좌표/회전 정보 매핑 가능 여부.
- FR training에 접근 가능한 **원본/native scene**의 실제 보유 여부. **FR benchmark test를 학습으로 무심코 사용하면 안 됨.**
- inference 중 patch/타일이 scene-global prior를 공유할 수 있는 인터페이스/metadata가 있는지.
- original source scene이 제공되지 않고 독립 패치만 제공된다면 scene-global prior의 정의가 성립하는지. **근거 없는 센서 평균 prior 또는 추정한 가짜 scene ID로 대체하지 않는다.**

이 항목이 충족되지 않으면 **Stage 1/2 학습을 시작하기 전에 운영 프로토콜을 다시 정의**해야 한다.

### 3.5 Low-confidence scene 및 zero-vector

\[
\mathbf d_s=\mathbf b_s/\|\mathbf b_s\|_2
\]

는 `||b_s||≈0`에서 불안정하다. 또한 WV2처럼 방향 집중도가 낮은 장면에 scene prior를 기계적으로 강제해서는 안 된다.

**미확정 사항:** zero/low-confidence prior를 처리할 명시적 fallback(예: prior 무효 상태, 일반 dy/dx head, 방향 고정의 기준 등)은 이번 대화에서 확정되지 않았다. 임의로 `b=0`, `d=(1,0)` 같은 회귀좌표계를 조용히 주입하지 말고, **우선 발생 건수·조건을 측정하고 fallback 정책을 문서화한 뒤 구현**한다. 이번 baseline에서 새 confidence-learning network/loss는 추가하지 않는다.

### 3.6 Resolution/phase-aware prior 변환

`b_s^F`가 native FR PAN pixel 단위이고 RR PAN grid가 물리적으로 `k`배 coarse하다면 **순수 translation의 이상적인 scale**은:

\[
\boxed{\mathbf b_s^R\approx\mathbf b_s^F/k}.
\]

단, 이것은 **동일 FOV/물리적 공간/기하 기원(origin)**이 검증된 경우에만 직접 적용한다. PAN과 MS가 서로 다른 PSF/MTF·decimation phase를 가질 수 있으므로, 실제 변환 연산 `T_{F→R}`은 dataset generation과 좌표 규약을 확인하여 정의해야 한다. **필요한 phase correction을 독립 측정 없이 0이라고 가정하지 않는다.**

`L_scale`은 쓰지 않지만, **prior의 단위 변환 자체는 반드시 필요**하다. 이것은 loss가 아니라 deterministic input transformation이다.

### 3.7 Scene prior는 learnable인가?

`[FIXED]` 아니다. `b_s`, `d_s`, `n_s`를 모듈 파라미터에 넣지 않고 constant conditioning으로 전달한다. reconstruction gradient로 prior estimator 자체를 변경하지 않는다.

---

## 4. PAN Aligner architecture — 현재 코드와 새 변경 분리

### 4.1 기존 PANGlobalAligner에서 확인한 골격 `[CURRENT]`

기존 `pa/aligner.py` 기준 입력 `[B,1,H,W]`와 `[B,C,H,W]`:

| 순서 | 레이어 / 특징 | tensor channels |
|---|---|---|
| 1 | PAN stem: ReflectionPad(1) → Conv 3×3, stride1, no bias → GN(4) → SiLU | 1→16 |
| 2 | LMS/MS stem: 동일 구조, GN(4) | C→16 |
| 3 | 채널 concatenation | 16+16=32 |
| 4 | Joint1: 3×3 Conv, stride2, GN(8), SiLU | 32→32 |
| 5 | Joint2: 3×3 Conv, stride2, GN(8), SiLU | 32→64 |
| 6 | **[CURRENT: 교체 대상]** `_Res`: ReflectionPad+Conv3×3(no bias)+GN+SiLU → ReflectionPad+Conv3×3(no bias)+GN → **SiLU(x+residual)** | 64 |
| 7 | Global Average Pooling | `[B,64]` |
| 8 | FC1 + SiLU | 64→32 |
| 9 | FC2 zero-init | 32→2 `(dy,dx)` |

기존 WV3(8-band) 파라미터 수는 코드 docstring 기준 **105,330**. **새 Aligner에서는 residual block 구현과 conditional head가 모두 바뀌므로 기존 파라미터 수를 그대로 사용하면 안 된다.** 실제 변경 후 parameter count 및 FLOPs는 실행 서버에서 재측정한다.

기존 aligner의 입력 전처리는 **aligner 전용 복사본**에 대한 sample×channel별 공간 z-normalization이며, 평균·분산의 계산 그래프는 detach한다.

\[
Z(x)=\frac{x-\mathrm{sg}(\mu_{hw}(x))}{\sqrt{\mathrm{sg}(\mathrm{var}_{hw}(x))+10^{-6}}}.
\]

이 값은 **reconstruction이나 최종 warped PAN의 radiometry를 바꾸지 않는다**. `10^-6`은 조회한 현재 코드 값이며 새 하이퍼파라미터 실험 대상이 아니다.

### 4.2 PAN Aligner의 ResBlock을 U-Net ResBlock과 동일하게 교체 `[FIXED: 아키텍처 변경]`

**이번 v1.2 수정의 핵심**은 `PANGlobalAligner`의 `self.res = _Res(64,8)`을 **Teacher/Student reconstruction U-Net에서 사용하는 `PANCrafterPaper.ResBlock`과 동일한 내부 구조**로 교체하는 것이다. 기존 `_Res`는 GroupNorm, reflection padding, 두 Conv bias=False, 합산 후 SiLU를 사용한다. 재구성 U-Net의 블록은 **Channel-wise LayerNorm + pre-activation Conv×2 + 두 번째 Conv zero-init + 합산 후 activation 없음**이다. 이 차이를 동일한 블록으로 통일한다.

| 항목 | 현재 Aligner `_Res(64)` `[CURRENT]` | 새 Aligner 공통 `ResBlock(64)` `[FIXED]` |
|---|---|---|
| 블록 진입 | Conv→GN→SiLU | **ChannelLayerNorm→SiLU→Conv** |
| 1번째 Conv | ReflectionPad(1), 3×3, bias=False | **3×3 padding=1, U-Net 구현과 동일한 bias** |
| 2번째 경로 | ReflectionPad(1)→Conv3×3(no bias)→GN | **ChannelLayerNorm→SiLU→Dropout(p=0)→Conv3×3** |
| Residual 마지막 Conv 초기화 | 기본 초기화 | **weight와 bias zero-init (`zero_module`)** |
| Shortcut | identity add | **identity add** (64→64 동일 채널) |
| 합산 후 activation | **SiLU(x+h)** | **x+h, 추가 SiLU 없음** |
| Normalization | GroupNorm(8,64) | **채널 기준 `LayerNorm(64)`**, affine=True |
| 추가 모듈 | 없음 | `mode_modulation=False`, `se=False` |

**공통 ResBlock의 정확한 수식은 §8.2에 한 번만 정의한다.** Aligner의 64채널 residual stage와 Teacher/Student U-Net body에서 해당 공통 정의를 재사용한다.

**변경 범위가 아닌 것:** PAN stem, native LMS stem, `joint1`, `joint2`는 기존 **GroupNorm** 및 convolution/activation 구성을 그대로 유지한다. **Aligner 전부를 LN으로 전환하는 것이 아니다.** Teacher/Student reconstruction U-Net의 block count, width, input layout, normalization도 변경하지 않는다.

**실제 코드 재사용 방식 `[PROPOSED / VERIFY]`:** `PANCrafterPaper.ResBlock(channels=64,out_channels=64,norm="ln",dropout=0.,mode_mod=False,se=False)`를 직접 사용하거나 `CommonResBlock`로 추출하여 A/T/S가 동일 연산 구현을 참조한다. 기존 `ResBlock.forward(x,s)`는 `mode_mod=False`여도 `s` 인자를 받으므로 호출부에서 올바르게 처리해야 한다. 추출 시 재구성 U-Net의 기존 가중치 key/state 및 수치 결과가 의도치 않게 변하지 않도록 검증한다. **비슷한 블록을 새로 작성하는 것과 동일 구현을 재사용하는 것은 다르다.**

**초기화 주의:** Aligner FC2는 기존처럼 zero-init이며, 공통 ResBlock의 마지막 Conv도 zero-init이다. 초기 correction은 여전히 scene prior `c0=b_s`여야 한다. 초기 gradient를 점검할 때 FC2·ResBlock 마지막 Conv·앞쪽 feature extractor의 단계별 전달 차이를 고려한다. 원래 `_Res`와 weight/state_dict가 호환된다고 가정하지 않으며 **Stage 1 학습/초기화 및 이전 체크포인트 migration 여부를 검증**한다. 성능 변화가 critical하지 않을 것이라는 판단은 가설이지 측정된 사실이 아니다.

### 4.2.1 새 scene-aware regression head `[PROPOSED]`

PAN/MS stem·joint feature extractor는 기존 구조를 유지하되, **residual block만 위 공통 ResBlock으로 교체**한다. 여기에 기존 `64→32→2` head를 scene-direction conditioning 방식으로 변경한다.

\[
h_i=\mathrm{GAP}\big(\mathrm{Backbone}_\phi(Z(P_i),Z(R_{A,i}))\big)\in\mathbb R^{64}.
\]

\[
(\alpha_i,\beta_i)=H_\phi([h_i;d_{s,y};d_{s,x}]).
\]

예시 호환 방식: 기존 `64→32→2`를 **`66→32→2`**로 교체. 이 차원 선택은 기존 head에 방향 2채널을 붙이는 **구현 제안**이며 현재 구현된 상태가 아니다. magnitude는 head에 넣지 않아도 최종 output에서 `b_s`가 더해진다.

\[
\mathbf n_s=(-d_{s,x},d_{s,y})\quad\text{for ordering }(dy,dx),
\]

\[
\boxed{\Delta\mathbf u_i=\alpha_i\mathbf d_s+\beta_i\mathbf n_s}
\]

\[
\boxed{\mathbf c_{s,i}=\mathbf b_s+\Delta\mathbf u_i}.
\]

**좌표/각도 유의:** 코드의 vector 순서는 `(dy,dx)`다. `d=(cosθ,sinθ)`을 쓰려면 θ의 기준축이 **+dy**여야 한다. 일반적인 +x 기준 `atan2(dy,dx)`와 섞으면 90도 회전·축전환 오류가 난다. 구현에서는 가능하면 `d=b/||b||`를 직접 써서 각도 표현을 건너뛴다.

### 4.3 Regression initialization

기존과 같이 FC2 weight/bias를 모두 0으로 시작한다.

\[
\alpha_i=\beta_i=0\quad\Rightarrow\quad\mathbf c_{s,i}=\mathbf b_s
\]

이것은 **scene prior로 초기 correction**한다는 뜻이다. 최종 output에 `tanh/clamp(±b)`를 사용하지 않는다. `b=2`는 ε augmentation radius일 뿐 Aligner correction의 최대치가 아니다.

### 4.4 One translation per patch

출력 shape `[B,2]`. 이는 patch 내 모든 위치에 동일한 translation을 적용한다. `α,β`가 image feature에 의해 결정되므로 서로 다른 patch에서는 shift가 달라질 수 있다. dense warp/local flow라는 표현을 쓰지 않는다.

### 4.5 Full-image FR input

Conv/GAP/FC 구조라 spatial dimensions 자체는 고정되어 있지 않다. 하지만 **구조적으로 64/128/256/512 입력을 처리할 수 있다는 것**과 **각 해상도에서 correct magnitude와 shift response를 학습한다는 것**은 다르다.

- `[FIXED]` 같은 Aligner weights를 RR/FR branch가 공유한다.
- `[VERIFY]` 입력 normalization, texture/statistics, receptive-field physical footprint가 달라질 때 global shift prediction이 어떻게 변하는가.
- `[VERIFY]` full-scene vs tile/crop 입력 시 scene prior와 patch residual 통계가 일관적인가.

### 4.6 Aligner view crop 문제 `[VERIFY]`

오래된 `pa/offset.py`에는 padding 단서를 억제하기 위해 aligner 입력에 고정 margin crop을 적용하는 경로가 있고, 최신 `ta2/model.py`에는 full-field alignment path가 있다. **새 방법에서 어떤 경로를 채택할지는 합의되지 않았다.** 서버는 실제 training pipeline을 기준으로 이 차이를 확인해야 한다. 임의 margin 값 도입이나 padding artifact 방치는 둘 다 금지한다.

---

## 5. Differentiable PAN warp

### 5.1 실제 실행 정의 `[CURRENT]`

`pa/warp.py`:

```python
# conceptual API; implementation on experiment server must be audited
warped_pan = warp_pan(pan, correction_dy_dx)  # [B,1,H,W], [B,2]
```

- `grid_sample`, mode=`bicubic`, padding_mode=`border`, `align_corners=False`.
- PAN/shift warp는 FP32 경로(원본 float64이면 그 형식을 유지하는 구현).
- `ε`를 위한 첫 warp는 gradient를 끊는다. native correction warp는 Aligner gradient를 받아야 한다.
- `c=0`에서도 sampler를 우회하는지 여부는 실제 코드 일관성 검사. 기존 warp 함수 자체는 호출되면 항상 sampling한다.

### 5.2 두 번의 warp는 한 번의 warp와 완전히 같지 않다

\[
W(W(P,\epsilon),c_\epsilon)\not\equiv W(P,\epsilon+c_\epsilon)
\]

bicubic resampling의 interpolation blur, 4-tap support, boundary replicate가 누적될 수 있다. 그러나 **ε consistency는 pixel image 차이가 아니라 correction-vector 차이를 직접 감독**하므로, 두 번 warp된 영상으로 GT absolute structural loss를 계산하지 않는다.

### 5.3 Boundary / valid support

`padding_mode=border`는 out-of-FOV sampling을 가장자리 값으로 채운다. 모델이 이 패턴에서 ε를 쉽게 추론하면 **실제 지형의 정합학습이 아니라 padding artifact 학습**이 될 수 있다.

기존 코드에는 `warp_support_mask`, `two_stage_support_mask`, `support_margin_ok`가 있다. 서버는 image loss의 valid ROI, fixed interior mask, Aligner view crop, 큰 prior correction 상황에서의 support fraction을 비교해야 한다.

**미확정:** 이번 합의에서는 `L_abs`에 어떤 support mask·margin을 쓸지 고정하지 않았다. 구현할 때 무조건 zero/valid mask라는 새 방법을 슬쩍 도입하지 말고, objective definition과 ROI 정책을 사전에 기록하고 ablation을 분리한다.

---

## 6. ε 생성 방식 — 현재 구현을 바탕으로 한 정확한 명세

### 6.1 정의와 분포 `[FIXED]`

ε는 **native PAN의 관측 displacement**가 아니다. **우리가 독립적으로 주입하는 known additional displacement**이며, 매 update/sample에서 랜덤하게 뽑는다.

반경:

\[
\boxed{b=2.0\ \mathrm{pixels\ on\ the\ current\ PAN\ grid}}
\]

단위: **현재 Aligner에 공급된 PAN tensor의 픽셀**, `(dy,dx)`.

원판 면적 균등분포:

\[
u_1,u_2\stackrel{\rm iid}{\sim}{\cal U}(0,1),
\quad r=b\sqrt{u_1},\quad\theta=2\pi u_2,
\]

\[
\boxed{\epsilon=(r\sin\theta,r\cos\theta).}
\]

반경 `b`와 무관한 기하학적 기대값:

\[
\mathbb E[\epsilon]=0,\quad
\mathbb E[\|\epsilon\|_2]=\frac{2b}{3},\quad
\operatorname{median}(\|\epsilon\|_2)=\frac b{\sqrt2},\quad
\operatorname{std}(\epsilon_y)=\operatorname{std}(\epsilon_x)=\frac b2.
\]

`b=2`이면 평균 norm 약 `1.333 px`, median norm 약 `1.414 px`, 축별 sd `1 px`. 이는 **sampling 검증용 이론값**이지 모델 성능 수치가 아니다.

왜 `sqrt(u1)`인가? `r=b*u1`은 좁은 면적의 원점 주변에 샘플이 과도하게 집중된다. `sqrt`를 써야 disk의 **면적에 균등**하다. `dy,dx`를 독립 Uniform(-b,b)로 뽑으면 정사각형이 되어 대각선에서 norm이 `sqrt(2)*b`까지 증가하므로 기존 구현과 다르다.

### 6.2 Sampling 단위와 PRNG `[CURRENT→FIXED 적용]`

- sample마다 `[2]` vector 하나; 전체 patch에 동일하게 적용.
- 매 optimizer update마다 새 ε; 이전 ε 위에 누적하지 않음.
- augment 완료 후의 **원래 PAN**에만 한 번 적용.
- dedicated CPU `torch.Generator`로 data loader/shuffle/image augmentation RNG와 분리.
- 각 sample의 ε, random seed, update index를 추적 가능하게 로그 또는 재생성 가능 상태로 남긴다.
- 실험조건 비교에서는 같은 ε stream을 제공하기 위해 해당 step의 ε 사용 여부와 **무관하게** 동일하게 RNG를 소비하도록 하는 기존 TA2 비교 원칙을 고려한다. 구현된 branch마다 RNG advance가 동일한지 서버에서 검사한다.

참조 코드: `pa/offset.py::sample_offsets(batch, radius_hr, generator)`.

### 6.3 Epsilon injection

동일 batch에서:

\[
\mathbf c_0=A_\phi(P,R_A;\mathbf b_s),
\]

\[
P_\epsilon=\operatorname{sg}[W(P,\epsilon)],
\]

\[
\mathbf c_\epsilon=A_\phi(P_\epsilon,R_A;\mathbf b_s).
\]

**MS/GT/native LMS/source-scene prior는 ε branch에서 변하지 않는다.** `(P, R_A)`와 `(Pε,R_A)`는 동일 PAN-derived feature pipeline을 거친다. augmentation된 prior는 두 forward에서 **같은** prior를 쓴다.

### 6.4 Corrective response의 기대관계

\[
\boxed{\mathbf c_\epsilon+\epsilon\simeq\mathbf c_0}
\]

이 관계는 `W(P,c)`가 **sampling displacement**를 뜻한다는 현재 warp convention에서 나온다.

Residual parameterization을 사용하면:

\[
\mathbf c_0=\mathbf b_s+\Delta\mathbf u_0,\quad
\mathbf c_\epsilon=\mathbf b_s+\Delta\mathbf u_\epsilon,
\]

\[
\boxed{\Delta\mathbf u_\epsilon+\epsilon\simeq\Delta\mathbf u_0.}
\]

**핵심:** `b_s`는 cancel되지만 **ε loss로 b_s의 절대 accuracy를 검증할 수는 없다**. 절대정합과 상대반응을 혼동하면 안 된다.

### 6.5 실제 loss `[FIXED]`

\[
\boxed{
\mathcal L_\epsilon
=\frac{1}{2B}\sum_{i=1}^{B}\sum_{a\in\{y,x\}}
\left| c_{\epsilon,i,a}+\epsilon_{i,a}-\operatorname{sg}(c_{0,i,a})\right|
}
\]

- `mean`은 **batch×2 offset components** 평균이다.
- ε norm으로 나누지 않는다(작은 ε에서 loss의 의미가 달라짐).
- native branch `c0`의 ε loss target만 detach한다. **`L_rec`/`L_abs` native `c0`의 gradient까지 detach하면 안 된다**.
- synthetic warp `Pε`에 gradient를 흘리지 않는다.
- shifted PAN은 **Teacher U-Net 입력에 주지 않는다**.
- ε loss에는 **GT/Y가 없다**.

### 6.6 기존 baseline과 새 ε schedule 구분

기존 완료 실험:

| 이력 | ε 작동 주기 | 가중치 | 상태 |
|---|---|---|---|
| 기존 Teacher WV3/GF2 | 홀수 0-based update(격 update) | `1e-4` | 과거 실험 |
| 기존 Teacher QB | 동일 | `3e-4` | 과거 실험 |
| 2026-10-06 TA2 Teacher-only 계획 | every update | `1e-3` 주력, `0/1e-4/1e-2` 대조 | **계획**, 완료된 최적 성능 아님 |
| **새 합의 Stage 1** | **every update** | **`1e-3` 시작** | **이번 설계** |

별도 ramp, sensor-specific λ, alternating U-Net corruption은 이번 설계에 사용하지 않는다.

### 6.7 기존 response 측정의 중요성

기존 진단 문서는 반경 `2 px` 내 추가 shift에 대해 다음 **gain**을 보고한다.

| 기존 Student | train 64² | RR 256² | FR 512² |
|---|---:|---:|---:|
| WV3 | 0.043 | 0.038 | 0.029 |
| QB | 0.043 | 0.042 | 0.020 |
| GF2 | 0.199 | 0.204 | 0.140 |

이 값은 **이번 새 모델의 결과가 아니다**. 기존 모델의 반응이 약하다는 baseline evidence다. FR WV3/QB의 residual MAE는 기존 무반응 기준선에 가까운 수준으로 보고됐으며, 새 λ/FR ε/scene magnitude prior가 이를 개선하는지는 실험으로 확인해야 한다.

### 6.8 Sampling radius `b=2`의 근거·제한

기존 radius 결정은 WV3 native-resolution observed mismatch P95≈`1.98 PAN px` 등을 참고했으며, QB native P95≈`2.98 PAN px`는 radius2 밖이다. **b는 Aligner output clamp가 아니다.** b는 추가 shift perturbation 범위를 뜻할 뿐이다.

**논문 프로토콜상 중요:** 기존 `b=2`의 근거에 **평가용 FR test input 통계**가 포함되어 있었다는 점을 ε 요약 문서가 명시한다. 새 baseline의 개발 고정값으로 2를 사용할 수는 있으나, 최종 paper/reproducibility 버전은 **training-only native inputs로 산정 근거를 다시 정당화**해야 한다. test GT를 쓰지 않았더라도 “test를 전혀 보지 않았다”고 주장해서는 안 된다.

---

## 7. RR vs FR — 같은 Aligner, 서로 다른 관측/감독

### 7.1 핵심 구조

```text
                          native/source scene s
                             prior b_s (fixed)
                                  │
                     ┌────────────┴─────────────┐
                     │                          │
               RR supervised branch         FR native branch
              P_R / M_R / LMS_R / Y_R       P_F / M_F / LMS_F
                     │                          │
              Shared A_phi (trainable)     Shared A_phi (trainable)
                     │                          │
            c0_R , cε_R from ε_R           c0_F , cε_F from ε_F
                     │                          │
           L_rec(Y_R) + L_abs(Y_R)         L_ε^FR only
              + L_ε^RR                        │
                     └────────────┬─────────────┘
                                  │
                         A_phi shared update
```

FR branch에 **GT 이미지가 없어도** native PAN/MS pair만 있으면 `L_ε^FR`은 계산할 수 있다. 하지만 **training에 사용 가능한 native FR 입력**이 실제로 존재해야 한다. RR train patch와 FR benchmark test만 있는 경우, test input을 학습에 투입하지 않는 정상 protocol에서는 `L_ε^FR` 학습을 실행할 수 없다. 이 경우 자료 확보/프로토콜 수정 전까지 **blocking**한다.

### 7.2 RR과 FR의 ε는 같은 physical shift가 아니다 `[FIXED]`

\[
b_{RR}=2\text{ RR-PAN pixels},\qquad b_{FR}=2\text{ FR-PAN pixels}.
\]

**동일한 숫자지만 다른 물리적 길이**를 의미한다. 이 설계가 의도하는 것은 *해상도별 현재 grid에서 known perturbation에 대해 `Δc≈-ε`가 성립*하게 하는 것이다.

같은 native scene에서 FR PAN GSD가 RR PAN보다 `k`배 작은 경우, 물리적 거리 `D`를 나타내는 pixel displacement는 일반적으로:

\[
D\longleftrightarrow c^F\approx k c^R.
\]

그러나 **이 relation을 loss로 강제하지 않는다**. `b_s^F → b_s^R`의 deterministic prior 변환과, 별도로 sampling한 ε response learning을 **서로 다른 메커니즘**으로 유지한다.

### 7.3 무엇이 여전히 해결되지 않았는가

FR ε loss는 native absolute correction `c0^F`를 직접 알려주지 않는다. **global scale/position의 기본 anchor는 prior `b_s^F`가 담당**하고, patch residual의 FR absolute 크기는 ε consistency만으로 보장되지 않는다. `L_abs^RR`이 RR patch에 기초해 학습된 상대 보정이 FR content에도 일반화되는지는 반드시 관측한다.

### 7.4 Sample scheduling / batch mixing `[VERIFY]`

매 update에 RR와 FR 둘 다 forward한다는 **method objective**는 확정했지만, 다음은 실험 서버의 자원/loader 조건에 의존한다.

- RR과 FR을 같은 optimizer step에서 함께 처리하는지, gradient accumulation으로 동등하게 처리하는지.
- 다른 patch size의 VRAM 최댓값과 batch 수.
- RR/FR branch별 batch loss mean을 어떻게 맞출지.
- FR 예제가 특정 source scene에 집중되는 경우 scene imbalance를 어떻게 피할지.
- RR과 FR이 반드시 같은 FOV의 pair로 묶여야 하는 것은 아니다(`L_scale` 없음). 그러나 사용 prior의 scene identity와 geometry는 각 batch 내에서 정확해야 한다.

**임의 RR:FR 샘플비, batch size, gradient scaling 상수를 문서에 넣지 않는다.** 실제 `Lε_RR/Lε_FR` 발생 횟수와 weighted gradient contribution을 기록한 뒤 결정한다.

---

## 8. Reconstruction U-Net 구조 유지 및 Aligner와 공통 ResBlock 정의 `[FIXED]`

### 8.1 사용자 확정 결정 및 적용 범위

**사용자가 이번에 추가로 통일하려는 대상은 Teacher↔Student가 아니라 `PAN Aligner residual block ↔ reconstruction U-Net ResBlock`이다.** Teacher \(F_T\)와 Student \(F_S\)는 **기존 U-Net 구조를 그대로 유지**한다. 둘은 현재 기존 lightweight candidate에서 동일한 `PANCrafterPaper` 3-scale U-Net class와 ResBlock/Channel-wise LN을 쓰고, hidden width와 block depth로 용량을 구분한다. 이 Teacher/Student architecture 결정은 이번 수정에서 다시 바꾸지 않는다.

**실제로 바뀌는 구성은 Aligner의 `_Res(64)` 하나**다. U-Net의 공통 ResBlock을 Aligner에서도 사용한다. ResBlock 내부 Norm/activation/초기화를 바꾸면 Aligner의 학습과 성능에 영향을 줄 수 있으므로, 영향이 미미하다고 단정하지 않는다. 설계 목표는 중복된 ResBlock 정의 제거와 동일한 블록 규칙의 재사용이다. 논문에서는 **공통 ResBlock을 한 번 설명**하고, Aligner backbone과 Teacher/Student U-Net에서 어디에 배치했는지를 별도로 기술한다.

- **`[FIXED]`** U-Net ResBlock internal: `ChannelLayerNorm → SiLU → Conv3×3 → ChannelLayerNorm → SiLU → Dropout(p=0) → zero-initialized Conv3×3`, shortcut은 동일 채널이면 identity, 변경 채널이면 Conv1×1.
- **`[FIXED]`** ResBlock 내부 Norm: spatial location마다 channel dimension에 `LayerNorm(C)`, affine learnable. **Aligner ResBlock(64), Teacher U-Net ResBlock, Student U-Net ResBlock 모두 LN**을 사용한다. **PAN Aligner의 stem 및 joint convolution에 있던 GN은 그대로 유지**한다. 전체 Aligner가 LN인 것은 아니다.
- **`[FIXED]`** single-task ResBlock: `mode_modulation=False`. `se=False`, attention-free candidate에서는 `n_attn=0, attn_locations=[]`, `swin_depth=0,swin_mid=0`. 현재 확인된 Teacher/Student lightweight branches가 이 구성을 사용한다.
- **`[FIXED]`** `ResBlock` 내부의 레이어 종류, kernel, 활성화, zero-init 및 residual sum 규칙을 **Aligner/Teacher/Student 세 경로에서 동일**하게 사용한다. Teacher/Student U-Net의 그 외 architecture는 손대지 않는다.
- **`[VERIFY]`** 실제 서버에서 선택하는 Teacher/Student의 width/depth, input stem layout(`P0` 대 추가 LPAN/HPAN), dropout/checkpoint는 코드·config·state hash로 pin한다. **기존 branch마다 입력 channels가 달랐을 수 있으므로 사용자가 결정하지 않은 입력 layout을 이 문서에서 임의로 변경하지 않는다.** 입력 stem의 차이가 있더라도 body의 ResBlock/Norm은 통일한다.
- **`[EXCLUDED]`** Teacher/Student U-Net 재설계; Aligner stem/joint GN을 LN으로 변경; Aligner만 post-add SiLU 유지; Aligner만 다른 convolution padding/bias 규칙 적용; normalization sweep; 신규 attention/SE/Swin; Stage 2에서 U-Net body 변경.

### 8.2 A/T/S 공통 ResBlock의 정확한 계산 순서

현재 U-Net 코드 `model/pancrafter_paper.py: ResBlock`에서 `mode_mod=False, se=False, dropout=0`일 때이며, **새 PAN Aligner의 64채널 residual stage에도 이 연산을 동일하게 적용**한다:

\[
h_1 = \operatorname{Conv}_{3\times3}\big(\operatorname{SiLU}(\operatorname{LN}_{ch}(x))\big),
\]

\[
h_2 = \operatorname{Conv}^{zero}_{3\times3}
\big(\operatorname{Dropout}_{0}(\operatorname{SiLU}(\operatorname{LN}_{ch}(h_1)))\big),
\]

\[
\boxed{\operatorname{ResBlock}(x)=S(x)+h_2}
\]

여기서 \(S=\mathrm{Identity}\) (채널 동일), 또는 \(S=\mathrm{Conv}_{1\times1}\) (채널 변환). `zero`는 마지막 3×3 convolution의 kernel/bias를 모두 0으로 초기화한다는 뜻이다. 모델을 학습하면서 가중치는 **정상적으로 업데이트된다.** Zero-init은 freeze가 아니다.

**추가 구현 주의:** `model/pancrafter_paper.py`는 `ResBlock`을 `out_layers[0]` 등으로 나누어 호출한다. **실제 forward 순서**, `nn.LayerNorm`의 적용 차원([B,H,W,C]로 채널 마지막 permutation), 3×3 zero-padding, convolution bias 및 `zero_module` 호출 위치를 A/T/S 공통으로 확인한다. 기존 Aligner는 reflection padding과 GN/합산 후 SiLU를 사용했으므로, **신규 공통 블록으로 바꾸면 학습 결과 및 checkpoint 호환성이 달라진다.** U-Net은 변경하지 않는다.

### 8.3 공통 3-scale U-Net 전체 흐름

현재 재구성 코드에서 attention-free `PANCrafterPaper` 기본 구조는 다음과 같다.

```text
Input x_in: [B, C_in, H, W]    # PAN + MS base 등, 구체 layout은 선택된 기존 recipe 기준
       │
Conv 3×3: C_in → C (hidden width)
       │
Encoder full-res: d0 × common ResBlock(C)
       ├──────────────────────────────────────────── skip1
       │
DownConv 3×3 stride2  → [B,C,H/2,W/2]
       │
Encoder half-res: d1 × common ResBlock(C)
       ├────────────────────────── skip2
       │
DownConv 3×3 stride2 → [B,C,H/4,W/4]
       │
Bottleneck: d2 × common ResBlock(C)
       │
UpConv transpose 2×2 stride2 → H/2
       │
Concat skip2: 2C → decoder2 first common ResBlock(2C→C)
       │         total d1 blocks (existing mirrored default)
       │
UpConv transpose 2×2 stride2 → H
       │
Concat skip1: 2C → decoder1 first common ResBlock(2C→C)
       │         total d0 blocks (existing mirrored default)
       │
LayerNorm(C) → SiLU → zero-init Conv 3×3 (C→C_MS)
       │
Predict residual ΔY; output Y_hat = B_M + ΔY
```

**Teacher/Student의 출력은 언제나 MS band 수 \(C_{MS}\) 채널**이며, RGB 3채널로 축소하지 않는다. RGB 평균 luminance는 spatial loss용 projection일 뿐 reconstruction output definition이 아니다.

`PANCrafterPaper`의 현재 `UpConv`가 실제로 `ConvTranspose2d(kernel=2,stride=2)`이며 `DownConv`는 `Conv2d(kernel=3,stride=2,pad=1)`이다. H/W가 작거나 홀수인 경우 skip spatial size가 안 맞을 수 있으므로 PAN patch shape의 4-divisibility를 검증한다.

### 8.4 Teacher/Student capacity 및 입력 인터페이스

아래는 **현재 코드에서 발견한 concrete candidate**다. 사용자가 새 아키텍처의 모든 수치를 새로 고정한 것이 아니므로 서버에서 실제 선택된 run과 대조하여 기록한다.

| 구분 | 기존 candidate | 공통 유지 | 아직 pin해야 할 것 |
|---|---|---|---|
| Teacher | `hidden_size=112`, `depth=(1,2,3)` | common ResBlock + `norm="ln"` | actual checkpoint, `input_layout` |
| Student | `hidden_size=104`, `depth=(1,2,2)` | common ResBlock + `norm="ln"` | actual checkpoint/new init, `input_layout` |
| Both | `mode_modulation=False`, `dropout=0`, attention-free U-Net candidate | `ResBlock` source 그대로 | extra LPAN/HPAN front-end 여부 |

한 코드 branch에서는 Teacher 입력 `P0`가 `PAN(1)+MS(C)`이고 Student `PLH`는 추가 LPAN/HPAN를 사용하여 `PAN(1)+LPAN(1)+HPAN(1)+MS(C)`였다. **입력 branch의 차이를 숨기지 말 것.** 공통 U-Net 설명을 두 번 하지 않는 목표는 *body/block/norm의 통일*이며, front-end channel composition은 별도 한 줄로 표시하면 된다. 새로운 성능 주장 또는 FLOPs 감소 효과를 Norm/ResBlock 통일만으로 가정하지 않는다.

비교 실험을 공정하게 하려면 동일 input layout으로 Teacher/Student를 비교하는 통제 실험이 유용할 수 있다. 그러나 이 결정은 이번 사용자 합의에 없는 별도 아키텍처 실험이므로 **기본 method에 몰래 추가하지 않는다.**

---

### 8.5 PAN-native correction만 reconstruction에 사용 `[FIXED]`

\[
\mathbf c_0^R=A_\phi(P^R,R_A^R;\mathbf b_s^R),\qquad
\widetilde P^R=W(P^R,\mathbf c_0^R).
\]

Reconstruction base:

\[
B_M=U(M^R)
\]

Teacher predicts an HRMS residual:

\[
\hat Y_T=B_M+F_T(\widetilde P^R,B_M,\text{optional existing inputs}).
\]

**주의:** 새 확정은 **PAN Aligner의 ResBlock을 U-Net과 동일한 LN·SiLU·Conv·zero-init 블록으로 교체**한다는 것이다(§4.2·§8.2). Teacher/Student의 U-Net body·capacity는 그대로 두고, 실제 기존 training recipe의 input stem layout과 LRMS residual base를 pin한다. Stage 1에서 ResBlock 변경 효과와 scene prior/consistency 효과를 구분하여 해석해야 한다.

Reconstruction loss:

\[
\boxed{\mathcal L_{\rm rec}^{RR}=\operatorname{mean}_{n,p,c}|\hat Y_T(n,p,c)-Y(n,p,c)|.}
\]

- HRMS GT `Y`는 **RR observation의 target**이다.
- Teacher+Aligner 양쪽으로 gradient가 흐른다.
- shifted `Pε`를 reconstruction에 보내지 않는다.
- FR에는 reconstruction target `Y^F`를 만들어 넣지 않는다.

### 8.6 Aligner reference와 U-Net base를 분리한 채 기록

이번 baseline의 Aligner reference는 **native LMS**다. 기존 `F.interpolate(ms, scale_factor=4, bicubic, align_corners=False)`와 dataset `lms` 사이에는 audit상 약 `(-0.5,-0.5) PAN-pixel` 고정 phase offset이 관측되었다. 이것은 **센서 PAN/MS acquisition shift와 다른 preprocessing offset**이다.

서버는 최소한 아래를 기록한다.

| 텐서 | 생성 원천 | 학습에서 하는 역할 |
|---|---|---|
| `P` | observed PAN | 이동되는 spatial source |
| `M` | observed/synthesized LRMS | spectral observation |
| `R_A` | **native LMS** | Aligner geometry reference |
| `B_M=U(M)` | **현재 고정된 reconstruction base 생성 방식** | reconstruction residual base |
| `Y` | RR HRMS GT | Stage 1/2 GT target |

`R_A`를 native LMS로 바꾼다는 결정이 **Teacher의 residual base까지 LMS로 바꾼다는 뜻은 아니다**. 실제 코드에서 두 입력이 분리되지 않는다면 수정이 필요하다. 반대로 분리했을 때 reconstruction target phase 문제/비교 공정성은 기록·검증해야 한다.

---

## 9. Stage 1 absolute RR spatial supervision — SIPSA 계열

### 9.1 사용하는 GT 채널 `[FIXED]`

GT HRMS에서 RGB 채널을 sensor별 실제 band index로 추출한다.

\[
\boxed{Y_{\rm lum}=(Y_R+Y_G+Y_B)/3.}
\]

WV3의 8개 밴드를 평균한 pseudo-PAN이 아니다. **RGB 3개만** 사용한다. `0.299/0.587/0.114` 계수 luminance가 아니라, SIPSA-Net이 설명한 **RGB 단순 평균**을 따른다. 채널 순서를 불변으로 가정하지 않고 데이터셋 메타데이터/loader 코드로 검증한다.

### 9.2 SIPSA-Net과의 같음/다름

같은 원리:

- RGB mean luminance 사용.
- PAN/MS radiometric polarity/gradient sign이 달라도 대응될 수 있으므로 **absolute edge/gradient** 비교.
- signed raw RGB/PAN intensity L1 대신 **spatial derivative mismatch의 L1**.

다른 점:

- SIPSA-Net 기본 공간 기준은 PAN frame으로 MS 쪽 정보를 정렬하지만, PANDA는 **PAN을 MS frame으로** 움직인다.
- PANDA에서는 **RR GT luminance**를 `L_abs^RR`의 기준으로 사용한다.
- 이 loss를 ε-shifted PAN이 아니라 **native corrected PAN**에 건다.

### 9.3 Spatial descriptor 정의

`D_y,D_x`: **고정, 비학습 2-D gradient operator**라고 둔다. `G(I)`는 다음 중 paper의 `|∇I|` 표기를 어떻게 tensor로 구체화하는지 나타내는 한 가지 고정 함수다.

일반식:

\[
G(I)=(|D_y I|,|D_x I|),
\]

\[
\boxed{\mathcal L_{\rm abs}^{RR}
=\operatorname{mean}_{n,p,k\in\{y,x\}}
\left|G_k(\widetilde P^R)(n,p)-G_k(Y_{\rm lum})(n,p)\right|.}
\]

`G`를 L2 gradient magnitude 하나로 만들 수도 있지만, **그 경우 위 per-axis 절댓값-미분과 숫자가 달라진다**. SIPSA 논문에서 핵심은 `absolute gradient → L1`이고, 정확한 discrete kernel(`Sobel`, `Scharr`, 차분 등)과 component reduction은 해당 논문 설명/코드를 보아야 재현할 수 있다. 사용자는 **공식 구현 전체 복사까지는 요구하지 않았다.** 서버는 하나의 differentiable fixed implementation을 선택하고 다음을 명시한다.

- derivative kernels의 정확한 정수 계수, scale/normalization.
- `|Dy|,|Dx|` channelwise L1인지, `sqrt(Dy²+Dx²)` magnitude L1인지.
- gradient border padding/valid ROI와 reduction denominator.
- PAN/GT의 radiometric normalization 방식(공통 DN/rescale 여부).
- low-gradient patch에서 loss가 얼마나 유용한지.

**허용되지 않는 것:** “SIPSA와 완전 동일 구현”이라고 주장하면서 실제로는 다른 discrete operator를 쓰고 기록하지 않는 것.

### 9.4 왜 raw luminance L1이 아닌가

\[
\|\widetilde P-Y_{\rm lum}\|_1
\]

는 intensity/contrast/spectral-response difference를 location mismatch로 해석한다. 절댓값 gradient L1은 이 문제를 완화하지만 **완전히 제거하지는 못한다**. 서로 다른 object contrast magnitude와 band visibility 때문에 correct alignment에서도 loss가 0이 아닐 수 있다. 따라서 `L_abs`는 *weak structural anchor*이지 실제 sensor displacement GT가 아니다.

### 9.5 Gradient flow

\[
\mathcal L_{\rm abs}^{RR}
\to G(\widetilde P)
\to W(P,c_0)
\to c_0
\to A_\phi.
\]

GT luminance `Y_lum`은 detach/reference이다. 이 loss가 Teacher U-Net output을 사용하지 않으므로 **Teacher U-Net에 직접 gradient를 주지 않는다.**

---

## 10. Stage 1 최종 objective·경로

### 10.1 세 개 top-level 명칭 유지 `[FIXED]`

\[
\boxed{\mathcal L_{\rm S1}=\mathcal L_{\rm rec}+\mathcal L_{\rm con}^{RR}+\mathcal L_{\rm con}^{FR}}
\]

\[
\boxed{\mathcal L_{\rm con}^{RR}=\mathcal L_{\rm abs}^{RR}+\lambda_\epsilon\mathcal L_\epsilon^{RR}}
\]

\[
\boxed{\mathcal L_{\rm con}^{FR}=\lambda_\epsilon\mathcal L_\epsilon^{FR}}
\]

즉:

\[
\boxed{\mathcal L_{\rm S1}=\mathcal L_{\rm rec}^{RR}+\mathcal L_{\rm abs}^{RR}
+\lambda_\epsilon(\mathcal L_\epsilon^{RR}+\mathcal L_\epsilon^{FR})}
\]

초기 고정값:

\[
b^{RR}=b^{FR}=2,\qquad\lambda_\epsilon=10^{-3},\qquad\lambda_{\rm abs}=1.
\]

**`λ_abs=1`은 다른 loss와 실측 gradient magnitude가 동등함이 증명된 값이 아니다.** 초기 변경 최소화를 위한 결정이며 비정상적 scale이면 조건부 재검토한다.

### 10.2 Gradient route table

| Stage 1 term | `A_phi` | Teacher `F_T` | scene prior `b_s` | synthetic ε |
|---|---|---|---|---|
| `L_rec` | **update** | **update** | fixed | 사용 안 함 |
| `L_abs^RR` | **update** | 직접 update 없음 | fixed | 사용 안 함 |
| `L_ε^RR` | **update** | update 없음 | fixed | constant |
| `L_ε^FR` | **update** | update 없음 | fixed | constant |

**ε branch의 `sg(c0)`는 `L_rec`/`L_abs`에서 c0가 detach된다는 뜻이 아니다.** 같은 forward tensor를 분기별로 다르게 사용해야 한다.

### 10.3 Pseudocode — Stage 1 one optimizer update

아래는 **실험 스펙을 설명하기 위한 의사코드**다. 바로 실행 가능한 실제 코드가 아니며, batch scheduling과 AMP/dataloader 세부는 서버에서 결정한다.

```python
# Inputs:
#   rr: P_R, M_R, LMS_R, GT_R, scene_ids, aug_transforms
#   fr: P_F, M_F, LMS_F,      scene_ids, aug_transforms
#   fixed estimator/cache provides prior in canonical grid

# 1. Fixed direction+magnitude prior, transformed to current PAN grid
b_R = convert_prior_to_grid_and_augmentation(rr.scene_ids, rr.geometry)
b_F = convert_prior_to_grid_and_augmentation(fr.scene_ids, fr.geometry)

# 2. Native branch, RR
c0_R = scene_aware_aligner(P_R, LMS_R, b_R)      # differentiable
aligned_R = warp_pan(P_R, c0_R)                 # differentiable to c0_R
pred_T = reconstruct_teacher(aligned_R, M_R)    # current F_T input contract
L_rec = mean_abs(pred_T - GT_R)
Ylum = mean_rgb(GT_R)                           # fixed channel indexing
L_abs = sip_sa_absolute_gradient_L1(aligned_R, Ylum)

# 3. RR epsilon branch: ONLY A, no corrupted U-Net reconstruction
eps_R = sample_uniform_disk(batch_rr, b=2, dedicated_rng)
P_eps_R = warp_pan(P_R, eps_R).detach()
c_eps_R = scene_aware_aligner(P_eps_R, LMS_R, b_R)
L_eps_R = mean_abs(c_eps_R + eps_R - c0_R.detach())

# 4. FR epsilon branch: same A, no HRMS GT
eps_F = sample_uniform_disk(batch_fr, b=2, dedicated_rng)
c0_F = scene_aware_aligner(P_F, LMS_F, b_F)
P_eps_F = warp_pan(P_F, eps_F).detach()
c_eps_F = scene_aware_aligner(P_eps_F, LMS_F, b_F)
L_eps_F = mean_abs(c_eps_F + eps_F - c0_F.detach())

# 5. Objective / update: scalar mean reduction must be audited
L_stage1 = L_rec + L_abs + 1e-3 * (L_eps_R + L_eps_F)
L_stage1.backward()
optimizer_stage1.step()  # updates A_phi and F_T only
```

### 10.4 Resulting checkpoint

Stage 1 완료 후 `A_phi`와 `F_T`를 함께 저장한다. Checkpoint provenance에는 최소한:

- Aligner backbone/head 정확한 state dict, input norm, parameters count.
- `F_T` state dict, reconstruction-base/input specification.
- prior estimator version, prior-cache digest, scene ID mapping digest.
- augmentation/MTF/LMS/interpolation/warp versions.
- ε RNG state, training seed, sampler radius & distribution, schedule, optimizer state.
- Stage 1 calibration/validation 측정/선정 경로.

을 포함한다. Stage 2는 **반드시 이 checkpoint의 frozen A/T와 같은 preprocessing**을 재사용한다.

---

## 11. Stage 2 — Frozen geometry, Teacher/Student reference

### 11.1 Stage 2 fixed architecture decision

\[
\boxed{A_S=A_T=A_{\rm Stage1},\quad\mathrm{requires\_grad}(A)=\mathrm{False}.}
\]

Teacher `F_T`도 frozen. Student reconstruction `F_S`만 trainable.

\[
c=A_{\rm Stage1}(P,R_A;b_s),\qquad\widetilde P=W(P,c),
\]

\[
\hat Y_T=F_T(\widetilde P,M),\qquad
\hat Y_S=F_S(\widetilde P,M).
\]

**같은 `\widetilde P` tensor 또는 numerically identical corrected PAN**을 Teacher와 Student가 사용해야 한다. Student용 trainable clone을 `A_S`라고 만들어두고 frozen이라 주장하는 구현은 범용성이 떨어진다. 우선 실제로 **단일 공유 module**로 구현할 것을 권한다.

Student backbone과 input feature composition은 **이번 수정의 변경대상이 아니다**. 기존 선택(예: Student LPAN/HPAN 추가 여부)을 pin한 후 그 조건을 모든 Stage 2 ablation에서 동일하게 유지한다. Aligner freeze와 Student architecture 변경을 같은 실험에서 동시에 시행하면 인과 해석이 불가능하다.

### 11.2 Stage 2에서 사용할 수 있는 target

Stage 2는 **RR supervised** training으로 정의한다. GT HRMS `Y`가 없으면 `e`, `q`를 계산할 수 없기 때문이다. FR에서는 **학습된 Student를 inference/evaluation**하며 training-only `e/q`를 요구하지 않는다.

---

## 12. Stage 2 guidance `e` — Reconstruction error

### 12.1 정확한 대상/shape

Frozen Teacher output `Y_T`와 RR GT `Y`는 모두 `[B,C,H,W]`이며 spectral channel에서 평균하여 pixel-level error map `[B,1,H,W]`을 계산한다.

\[
\boxed{
 e_i(p)=\frac1C\sum_{c=1}^C
 \left|\hat Y_{T,i,c}(p)-Y_{i,c}(p)\right|
}
\]

- `e`는 **Teacher 자신의 GT error**다.
- 기존의 `e_S` 또는 `max(e_S-e_T,0)` Teacher-advantage term을 사용하지 않는다.
- `e`는 RR GT에서만 계산된다. FR inference에는 입력하지 않는다.
- Teacher는 frozen이므로 `e`는 **Student 학습으로 바뀌지 않는다**. augmentation이 달라지면 같은 장소의 `e` map은 달라질 수 있음.

### 12.2 Calibration reference scale `τ_e` `[FIXED]`

Stage 2가 시작되기 전에 **Stage 1 frozen Teacher의 training set**에서 `e` 분포를 구한다.

\[
\boxed{\tau_e=\operatorname{median}_{(i,p)\in\mathcal D_{\rm train}}e_i(p).}
\]

- training-only. validation/test GT를 섞지 않는다.
- pixel-level values의 global median인지 sample-median의 median인지 수치가 다르다. 합의한 수식의 원의미는 **training 위치들에 대한 median**이다. 서버는 정확한 aggregation을 기록해야 한다.
- 현재는 센서별 네트워크/학습 집합이라면 **해당 모델의 training population**을 대상으로 계산한다. 추후 multi-sensor joint training을 한다면 per-sensor vs pooled reference를 명시적으로 결정해야 한다.
- 이 값은 **모델 파라미터가 아니라 training distribution에서 측정 후 고정하는 scale**이다.

### 12.3 Continuous blending

Hard weight:

\[
\boxed{w_e(p)=\frac{e(p)}{e(p)+\tau_e}.}
\]

Soft weight: `1-w_e`.

| 관계 | Hard weight `w_e` | Soft weight `1-w_e` |
|---|---:|---:|
| `e=0` | 0 | 1 |
| `e=τ_e/2` | 1/3 | 2/3 |
| `e=τ_e` | 1/2 | 1/2 |
| `e=2τ_e` | 2/3 | 1/3 |
| `e≫τ_e` | ~1 | ~0 |

이것은 **0/1 hard switching이 아니라 연속 blending**이다. hard와 soft가 경계에서 부드럽게 역할을 나눈다.

### 12.4 Soft/hard per-pixel losses

\[
L_{\rm soft}(p)=\frac1C\sum_c|\hat Y_S^c(p)-\operatorname{sg}(\hat Y_T^c(p))|,
\]

\[
L_{\rm hard}(p)=\frac1C\sum_c|\hat Y_S^c(p)-Y^c(p)|.
\]

\[
\boxed{
\mathcal L_{\rm fit}=\frac1N\sum_p
\left[(1-w_e(p))L_{\rm soft}(p)+w_e(p)L_{\rm hard}(p)\right].
}
\]

`w_e`의 양 항 합이 언제나 1이므로 GT/Teacher의 **유효 총 reconstruction supervision 계수**가 pixel마다 일단 1로 유지된다(두 target 사이 weight의 합 기준). 그러나 두 loss의 방향/크기가 같다는 뜻은 아니다.

**의도적 변경 사항:** 옛 PANDA의 `GT L1 everywhere + difficulty-based additional GT + Teacher-advantage soft`와 달리, 이 새 식은 `e≈0`인 곳에서 GT hard 비중이 거의 0이다. 이로 인해 Teacher의 작은 잔여 bias를 Student가 그대로 모방할 수 있다. 이는 **실험으로 확인할 명시적 위험**이며, 마치 unconditional GT reconstruction이 항상 유지되는 것처럼 기술하면 안 된다.

### 12.5 Numerical considerations `[VERIFY]`

`τ_e`가 0 또는 수치해상도에 매우 가까운 경우 `e/(e+τ)`가 불안정하다. 추가 **학습 조절용 하이퍼파라미터**를 새로 만들지 않되, 고정 numerical denominator floor의 존재·dtype·기준을 구현 기록으로 남긴다. Teacher output과 GT의 normalization 단위가 같아야 한다. `τ_e`를 다른 normalization으로 계산하고 학습 중 적용하는 오류를 금지한다.

---

## 13. Stage 2 guidance `q` — 새로운 spatial discrepancy

### 13.1 **과거 `q_old`와 새 `q_new`는 다르다**

기존 PANDA `q_old`:

\[
q_{old,i}=\operatorname{mean}_{\epsilon\in E}
\|c_i(\epsilon)+\epsilon-c_i(0)\|_1.
\]

이것은 **sample-level epsilon-response consistency error**로, absolute registration accuracy를 측정하지 않는다.

이번 수정 `q_new`:

\[
\boxed{
q_i(p)=\operatorname{mean}_{k\in\{y,x\}}
\Big|G_k(\widetilde P_i)(p)-G_k(Y_{i,\rm lum})(p)\Big|.
}
\]

여기서 `G`는 Stage 1 `L_abs^RR`에 사용한 **동일 fixed gradient descriptor**이며, `Y_lum`은 GT RGB 평균이다. 즉 Stage 1의 per-pixel absolute structural residual을 **reduce하기 전** 가져온다.

**중요:** 새 `q`가 높은 것은 *PAN vs visible GT의 gradient discrepancy가 높다*는 뜻이다. 이것이 반드시 **물리적 위치 오차**임을 뜻하지 않는다. spectral response, PSF difference, contrast amplitude, 건물 그림자, saturation, cloud/occlusion, high-frequency edge 자체가 원인이 될 수 있다.

### 13.2 `q` 계산 프로토콜

- Stage 1에서 학습된 Aligner는 Stage 2에서 frozen.
- native PAN을 correction한 `P̃`로 계산. ε-shifted PAN 사용 금지.
- Student prediction을 q 생성에 사용하지 않는다.
- q를 계산하기 위한 GT luminance는 RR training에서만 허용.
- map shape `[B,1,H,W]`로 고정한 뒤 Student GT spatial loss와 pixelwise multiply.
- `q`는 `detach` 및 `no_grad`로 계산. 그 값이 Student를 통해 다시 geometry를 update할 경로는 없다.

### 13.3 `τ_q` calibration `[FIXED]`

\[
\boxed{\tau_q=\operatorname{median}_{(i,p)\in\mathcal D_{train}}q_i(p).}
\]

\[
\boxed{w_q(p)=\frac{q(p)}{q(p)+\tau_q}.}
\]

`w_q≈0`이면 GT spatial emphasis 작음. `w_q≈1`이면 GT spatial emphasis 큼.

`τ_q`도 frozen Stage 1 pipeline + training GT에서만 산정한다. q의 per-component aggregation/mask/normalization이 calibration과 Student training에서 달라지면 **가중치 의미가 깨진다**.

### 13.4 q의 대표성 위험 — 필수 진단

1. **Low texture:** PAN/GT 모두 gradient가 거의 0이라 `q≈0`이라면, 이것은 *정합이 확실하다*가 아니라 *정합을 식별할 정보가 부족하다*일 수 있다.
2. **High-contrast edge:** 정확히 정렬되어도 PAN/visible response amplitude가 다르면 `q`가 높을 수 있다.
3. **Channel dependence:** q는 RGB-derived이므로 NIR/RedEdge spectral reconstruction quality와 일치할 필요가 없다.
4. **Spatial weighting overlap:** `q`도 edge 기반이고 Student spatial loss도 edge 기반이라, high-gradient 지역에 과도하게 큰 가중치가 집중될 수 있다.
5. **Corrected PAN image artifact:** bicubic warp/border/blur가 q에 반영될 수 있다.

따라서 아래를 Stage 2 시작 전에 분석한다.

- `q` vs `|∇Y_lum|` 상관 및 상위 quantile 동시발생률.
- `q` vs Teacher error `e` 산점도 및 joint histogram.
- high/low `q`에서 MIND before/after와 실제 GT reconstruction gain의 분포.
- low-texture/vegetation/urban 등 장면별 분리.
- q weight 적용/미적용 시 `L_spa` gradient concentration과 GT spectral metrics.

이 분석의 목적은 **`q`가 의미 있는 GT spatial emphasis 신호인지 검증**하는 것이다. 결과가 나쁘다면 임의로 새 gate/weight를 끼워 넣지 않고 원 설계의 실패로 기록한 뒤 다음 버전을 논의한다.

---

## 14. Stage 2 Student spatial loss

### 14.1 Student RGB luminance

\[
\hat Y_{S,\rm lum}=(\hat Y_{S,R}+\hat Y_{S,G}+\hat Y_{S,B})/3.
\]

GT:

\[
Y_{\rm lum}=(Y_R+Y_G+Y_B)/3.
\]

### 14.2 Q-weighted GT spatial supervision

\[
L_{\rm edge}^{GT}(p)=\operatorname{mean}_{k\in\{y,x\}}
\big|G_k(\hat Y_{S,\rm lum})(p)-G_k(Y_{\rm lum})(p)\big|,
\]

\[
\boxed{
\mathcal L_{\rm spa}^S=\frac1N\sum_p w_q(p)L_{\rm edge}^{GT}(p).
}
\]

`L_abs^RR`과 동일한 gradient definition과 RGB channel indexing을 사용한다. Teacher-aligned PAN vs GT에서 계산한 `q`는 GT edge loss **가중치**일 뿐, Student가 Teacher PAN을 다시 직접 복원하도록 강제하지 않는다.

### 14.3 Stage 2 최종 식 `[FIXED]`

\[
\boxed{\mathcal L_{\rm S2}=\mathcal L_{\rm fit}+\lambda_S\mathcal L_{\rm spa}^S,
\qquad\lambda_S=1\ \text{(초기 고정값)}.}
\]

완전히 전개하면:

\[
\boxed{
\mathcal L_{\rm S2}
=\frac1N\sum_p\left[
\frac{\tau_e}{e(p)+\tau_e}\,L_{soft}(p)
+\frac{e(p)}{e(p)+\tau_e}\,L_{hard}(p)
+\lambda_S\frac{q(p)}{q(p)+\tau_q}\,L_{edge}^{GT}(p)
\right].
}
\]

기존처럼 Teacher–Student relative improvement weight나 별도 trainable Student Aligner loss를 추가하지 않는다.

### 14.4 q/e 4분면에 따른 해석

| `e` (Teacher HRMS error) | `q` (PAN–GT spatial discrepancy) | 의미 | 실제 가중치 효과 |
|---|---|---|---|
| 낮음 | 낮음 | Teacher recon 비교적 좋고 구조 차이 작음 | Teacher soft 중심; 추가 spatial 약함 |
| 높음 | 낮음 | geometry보다 Teacher reconstruction 문제 | GT hard 중심; spatial 약함 |
| 낮음 | 높음 | Teacher GT error는 낮지만 PAN↔GT structural 차이 큼 | Teacher soft + 강한 GT spatial emphasis |
| 높음 | 높음 | 두 종류 어려움 동시 존재 | GT hard + 강한 GT spatial emphasis |

이는 **smooth weighting의 의미를 설명하는 quadrant**이지 hard threshold 기반 4개 개별 branch가 아니다. 도표를 만들 때만 median 기반 high/low 구분을 쓰는 것은 가능하지만, 실제 loss에는 threshold-switch를 넣지 않는다.

### 14.5 Stage 2 gradient route

| 객체 | Frozen? | Loss gradient |
|---|---|---|
| Stage 1 Aligner `A_phi` | **yes** | 없음 |
| Teacher `F_T` | **yes** | 없음 |
| scene prior `b_s` | **fixed** | 없음 |
| Student `F_S` | **no** | `L_soft`, `L_hard`, `L_spa^S` |
| `e`, `q`, `w_e`, `w_q` | **detached** | 없음 |

이전 Stage 2의 “Student Aligner에는 GT hard만 별도 역전파” 같은 복잡한 분기/optimizer는 **사용하지 않는다**.

### 14.6 Stage 2 pseudocode

```python
# Stage 1 checkpoint already selected/frozen.
# No student aligner clone. RR supervised batch required.
with torch.no_grad():
    b = convert_prior_to_grid_and_augmentation(scene_id, geometry)
    c = A_stage1(P_R, LMS_R, b)                   # frozen
    P_aligned = warp_pan(P_R, c)                 # frozen geometry
    Y_T = teacher_reconstruct(P_aligned, M_R)     # frozen
    e = mean_over_spectral_bands(abs(Y_T - GT_R))
    q = pixelwise_sipsa_grad_difference(P_aligned, mean_rgb(GT_R))
    w_e = e / (e + tau_e)                         # tau_e fixed train median
    w_q = q / (q + tau_q)                         # tau_q fixed train median

Y_S = student_reconstruct(P_aligned, M_R)          # trainable
L_soft_px = mean_over_spectral_bands(abs(Y_S - Y_T))
L_hard_px = mean_over_spectral_bands(abs(Y_S - GT_R))
L_fit = mean_spatial((1-w_e)*L_soft_px + w_e*L_hard_px)
L_edge_px = pixelwise_sipsa_grad_difference(mean_rgb(Y_S), mean_rgb(GT_R))
L_spa = mean_spatial(w_q * L_edge_px)

L_s2 = L_fit + 1.0 * L_spa
L_s2.backward()
optimizer_student.step()                          # only F_S parameters
```

**Loss normalization 결정 확인:** `mean_spatial`의 denominator가 valid pixels인지 all pixels인지, `w_q`의 크기로 normalize하는지 여부는 서로 다른 objective다. 합의한 수식은 **weight를 곱한 뒤 fixed valid pixel set에 대해 mean**이다. weight sum으로 normalize하면 추가적인 동작 변화가 있으므로 사전 선언 없는 치환 금지.

---

## 15. 실험/평가 지표 — 학습 loss와 독립된 주장 기준

### 15.1 Stage 1 known-shift response `[PRIMARY DIAGNOSTIC]`

\[
\Delta\mathbf c_i=\mathbf c_{\epsilon,i}-\mathbf c_{0,i}.
\]

\[
\boxed{
\mathrm{gain}
=\frac{-\sum_i\Delta\mathbf c_i\cdot\epsilon_i}
{\sum_i\epsilon_i\cdot\epsilon_i}.
}
\]

이론적으로 ideal response `gain=1`; shift 입력에 무반응하는 constant predictor이면 `gain=0`. **gain은 absolute registration 정확도나 HQNR 개선률이 아니다.**

같이 측정:

\[
\mathrm{MAE}_\epsilon
=\operatorname{mean}|\Delta\mathbf c+\epsilon|,
\]

\[
\mathrm{EPE}_\epsilon
=\operatorname{mean}_i\|\Delta\mathbf c_i+\epsilon_i\|_2,
\]

그리고 `gain_y`,`gain_x`, response slope vs radius, response vs scene prior quality를 확인한다.

### 15.2 기존 고정 ε probe 묶음 그대로 활용

| Probe | 실제 정의 | 용도 |
|---|---|---|
| `AXIS16` | 4축 방향 × radius `{0.25,0.5,1,2}` | 주진단; 무반응 기준선 비교 가능 |
| `DIAG16` | 4 대각 방향 × 동일 radius | isotropic response 평가 |
| `DISK64_HELD` | 학습 표본과 분리된 fixed seed disk 64개 | 일반 샘플 분포 검증 |
| `OOD_AXIS` | 축 방향 radius `{4,8}` × 4 방향 | 반경 밖 extrapolation **진단 전용**, 학습 범위 성능과 합치지 않음 |

이 probe는 `ta2/diagnostics.py`와 ε 설명서에 근거한다. 기존 axis16의 **constant/no-response baseline MAE=0.46875 PAN px**가 참조치이며, grid 정의가 달라지면 기준선도 재계산한다.

**RR, FR, sensor, patch size 별로 반드시 분리**하고 서로 다른 current-grid pixel 수치를 같은 physical displacement처럼 단순 비교하지 않는다.

### 15.3 Absolute-ish geometry evaluation

- before/after aligned PAN vs observed MS 계열 영상의 **MIND structural discrepancy** (`MIND` 자체는 고정 descriptor, 학습에 사용하지 않음).
- 가능하다면 독립적인 manually verified landmark의 TRE, 또는 GT reconstruction에 유리한 shift sweep 등의 소규모 cross-check.
- Matching estimator가 prior에 사용됐다면 그 동일 matcher의 before/after 개선을 **유일한 성공 증거로 삼지 않음**.
- GT가 없는 FR에서는 `MIND`/외부 판독 및 최종 PS no-reference quality를 **보조 근거**로 제한하고, 실제 physical registration GT가 있다고 주장하지 않는다.

**MIND 구현도 고정 필요:** MS의 어느 representation(RGB luminance / individual band / native grid)을 비교할지, PAN/MS PSF/grid matching, patch window 설정을 먼저 정한다. 이는 평가 프로토콜이지 새 trainable module이 아니다.

### 15.4 Reconstruction evaluation

| 환경 | 볼 항목 | 해석 제한 |
|---|---|---|
| RR (GT 있음) | PSNR, SAM, ERGAS, SSIM/SCC, Q4/Q8, per-band MAE | GT geometry/preprocessing에 의존 |
| FR (GT 없음) | HQNR, `D_λ`, `D_s`, 시각적 ghost/double-edge와 spectral distortion | `D_s`는 직접 registration error가 아님, `D_λ`는 spatial matching metric 아님 |
| Stage 1 recon contribution | `L_rec` no alignment vs prior-only vs learned correction, fixed backbone/control | 효과의 원인을 geometry로만 단정하지 말고 통제 조건 필요 |
| Stage 2 | RR 성능, FR HQNR, student-only runtime/memory | 동일 eval code와 단일 GPU 측정 환경 요구 |

### 15.5 Scene-prior contribution diagnostics

- `||b_s||` / `||Δu_i||` / `||c_i||`의 median/P90/P95, component mean.
- per-scene predicted shift vs offline anchor direction/magnitude.
- scene direction concentration **within source scene** 및 across source scenes를 구분.
- posterior residual `α_i`, `β_i`, residual-to-prior norm ratio, `r_perp=|β|/(|α|+|β|+tiny)`.
- low-confidence/low-texture patch에서 prior-only/corrected result 대비 차이.
- scene별 sample 수, crop 위치와 landmark 위치의 대응 가능성.
- abnormal prior/magnitude overflow, warp support fraction, aligner output finite rate.

### 15.6 새로운 q/e joint diagnostic

과거 4분면 Fig. 5는 **옛 `q_old`(ε-response)**에 근거한 것이다. 이번 `q_new`는 GT gradient discrepancy이므로 **새 q로 4분면을 다시 만들어야 한다**.

각 pixel 또는 patch 단위로 `e`,`q`의 hist/heatmap, correlation/conditional distribution 및 quadrant 비율을 기록한다. patch-level quadrant로 요약할 때 어떤 aggregation을 썼는지(예: mean 또는 median)를 명시한다. 이때 teacher error map과 `Y_lum`의 edge magnitude map도 함께 제시해야 `q`가 단순 texture strength의 proxy인지 분리할 수 있다.

---

## 16. Hyperparameter inventory — 변경하지 않을 값과 실제 의사결정 대상

### 16.1 이번 baseline에서 고정한 method choices

| 구성 | 현재 결정 | 구분 | 설명 |
|---|---|---|---|
| Scene prior | **direction + magnitude** | `[FIXED]` | `b_s=(dy,dx)`; learnable 아님 |
| Scene residual | `α d_s + β n_s` | `[FIXED]` | final shift는 `b_s+residual` |
| Scene prior loss | **없음** | `[EXCLUDED]` | `|β|` penalty/regularizer 없음 |
| Aligner MS input | **native LMS** | `[FIXED]` | reconstruction base와 구분 |
| PAN movement | **PAN→MS frame** | `[FIXED]` | MS/GT를 warp하지 않음 |
| ε sampling | 원점 중심 uniform disk | `[FIXED]` | current PAN px; random 2-vector |
| ε radius | **2.0** RR/FR 각각 current grid | `[FIXED: 개발 baseline]` | physical distance 동일하다는 뜻 아님 |
| ε shift unit | `(dy,dx)` current PAN pixel | `[FIXED]` | data augmentation 반영 |
| ε draw cadence | sample/update당 새 ε | `[FIXED]` | independent generator |
| ε branch schedule | **every update** | `[FIXED]` | 격 update 없음 |
| ε target stop-grad | **`sg(c0)`** | `[FIXED]` | native rec/abs의 c0는 detach 금지 |
| λ_ε | **1e-3 시작** | `[FIXED: 초기값]` | 새 baseline 성능 미검증 |
| Stage 1 spatial | `RGB mean → absolute 2-D gradient → L1` | `[FIXED]` | SIPSA-style, RR native correction |
| λ_abs | **1** | `[FIXED: 초기값]` | loss scale 검증 필요 |
| Stage 1 FR GT loss | **없음** | `[EXCLUDED]` | FR에는 HRMS GT 없음 |
| L_scale | **없음** | `[EXCLUDED]` | cross-scale displacement loss 없음 |
| Stage 2 Aligner | **freeze/shared** | `[FIXED]` | clone/fine-tune 없음 |
| Aligner ResBlock | **U-Net ResBlock으로 교체** | `[FIXED: 변경]` | 64ch Channel-wise LN/SiLU/3×3 Conv×2/zero-init second Conv/identity add; no post-add SiLU |
| Aligner stems/joints | **기존 GN 유지** | `[FIXED: 유지]` | ResBlock 내부만 LN으로 통일; stem/joint GN 변경하지 않음 |
| Teacher/Student U-Net body | **기존 공통 ResBlock + Channel-wise LayerNorm 유지** | `[FIXED: 유지]` | width/depth 외 T/S architecture 재설계 없음 |
| U-Net optional modules | `mode_mod=False`, `dropout=0`, `n_attn=0` (현재 lightweight candidate) | `[FIXED: 공통 block 원칙]` | 서버에서 실제 branch config 확인, 신규 모듈 추가 금지 |
| Teacher/Student capacity | Teacher 112/(1,2,3), Student 104/(1,2,2) 현재 candidate | `[VERIFY]` | 입력 layout 및 실제 config pin 후 확정 |
| Teacher | Stage 1 모델 freeze | `[FIXED]` | soft target reference |
| Stage 2 `e` | Teacher GT spectral-channel mean abs error | `[FIXED]` | Student gap 비교 없음 |
| τ_e | training-set pixel error median | `[FIXED]` | calibration result로 저장 |
| Stage 2 `q` | Stage 1 `L_abs`의 per-pixel structural residual | `[FIXED]` | 기존 `q_old` 대체 |
| τ_q | training-set q median | `[FIXED]` | q definition 종속 |
| `w_e`,`w_q` | `x/(x+τ)` | `[FIXED]` | continuous blending |
| λ_S | **1 시작** | `[FIXED: 초기값]` | 필요할 때만 검토 |
| MIND | independent validation | `[FIXED]` | training loss로 미사용 |

### 16.2 진짜로 확인할 scalar 및 수치의 우선순위

| 우선순위 | 항목 | 현재값/산정법 | 판단 방법 | 조정 허용 조건 |
|---|---|---|---|---|
| P0 | **λ_ε** | **1e-3** | RR/FR ε gain, reconstruction/HQNR 동시 관찰 | response가 약하거나 reconstruction과 충돌이 확인될 때 |
| P0 | `b_s` magnitude·sign·unit | native scene에서 robust 추정 | impulse, paired-grid, per-scene reliability | 데이터 근거가 틀리면 추정 pipeline 수정. 임의 숫자 tuning 금지 |
| P0 | `b=2` ε radius | 기존 값 | train-only mismatch distribution, OOD probe | 논문 근거 재산정 또는 coverage 부족 명확할 때 |
| P0 | 실제 FR training source | **미확인** | provenance/data registry | 없으면 FR branch 학습 불가능. 데이터 확보/프로토콜 선언 필요 |
| P1 | `λ_abs=1` | 초기 고정 | 원/가중 loss, Aligner gradient norm, HQNR 변화 | 동일 계수로 학습 파괴가 확인될 때만 |
| P1 | `λ_S=1` | 초기 고정 | spatial vs fit gradient, bandwise RR 성능, FR HQNR | Student spectral regression 악화/gradient imbalance가 확인될 때 |
| P1 | τ_e, τ_q | train median | 분포/zero 및 calibration stability | 현재 규칙 수정이 아닌 값 산정 버그 검증 |
| P2 | `τ_e × {0.5,1,2}` | **후속 ablation** | soft/hard sensitivity | baseline 완료 후 |
| P2 | per-sensor/branch-specific ε radius | **미사용** | 센서별 gain/coverage | baseline failure가 센서/scale별로 분리될 때 |

**학습률, optimizer, 배치크기, warm-up, epoch/update count, checkpoint frequency, seed 수, Scheduler, mixed precision**은 실제 학습에 중요한 하이퍼파라미터지만 **이번 방법론 합의에서 값이 정해지지 않았다.** 현재 저장소의 선택된 Teacher/Student baseline recipe를 pin한 뒤, 변경 대상과 구분한다. 이 문서에서 임의로 `lr=1e-4`, `batch=...`, `3 seeds` 같은 값을 사실처럼 고정하지 않는다.

### 16.3 새로운 learnable parameter는 어디에 추가되는가?

Scene prior 추정기는 learnable module이 아니다. **Aligner의 feature extractor에서 64ch residual block을 공통 U-Net ResBlock으로 교체**하고, conditional head 입력을 64→66으로 바꾸므로 **블록 내부 bias/norm·초기화 차이와 head 크기 변화**가 생긴다. 따라서 이전 Aligner의 105,330 parameters 및 FLOPs를 재사용하지 말고 구현된 code/state dict에서 재계산한다. Student는 별도 Aligner를 갖지 않으므로 inference 비용은 **prior estimator + Aligner 1개 + Student backbone** 기준으로 보고해야 한다.

---

## 17. 구현/논문에서 혼동하기 쉬운 항목 — 금지해야 할 대체

| 쉽게 발생하는 치환 | 왜 잘못인가 | 올바른 처리 |
|---|---|---|
| `b_s`와 epsilon radius `b=2`를 같은 값으로 사용 | global scene prior 크기와 augmentation radius는 별도 의미 | 다른 변수/namespace 사용 |
| `R_A = bicubic↑MS`로 무심코 변경 | native LMS 합의 위반, phase 오차 복귀 가능 | `R_A=native_lms` 고정; 별도 복원 base 유지 |
| Teacher LN / Student GN 또는 서로 다른 ResBlock | 공통 U-Net 내부 구조 고정 합의 위반; 결과 원인 분리 불가 | `PANCrafterPaper.ResBlock` + `norm='ln'` 공유, 코드/config hash 검증 |
| Aligner 전체 GN을 U-Net LN으로 일괄 변경 | 변경 범위는 **Aligner의 ResBlock 내부**뿐 | Aligner stem/joint는 기존 GN 유지; **Aligner ResBlock만 U-Net과 동일한 LN** |
| Aligner `_Res(64)`를 그대로 두고 같다고 설명 | 기존 블록은 GN/ReflectionPad/합산 후 SiLU로 **U-Net ResBlock과 다름** | 코드에서 실제로 공통 ResBlock으로 교체하고 state/count/test로 증명 |
| `GT Y`를 FR branch에 만들어 넣음 | FR HRMS GT 없음 | FR ε-only |
| `cε+ε≈0` 학습 | native absolute `c0`를 무시한 잘못된 target | `cε+ε≈sg(c0)` |
| `c0` 전역 detach | native reconstruction/absolute supervision이 Aligner를 학습 못함 | **ε target 경로에만 stop-gradient** |
| corrected `Pε`를 Teacher F_T로 보내기 | 구 PO10 corrupted reconstruction 경로 재도입 | ε branch는 Aligner만 |
| `b_F=4*b_R`를 예외없이 실행 | crop/phase/degradation grid 검증 생략 | 실제 PAN GSD/MTF phase mapping 검사 |
| fixed prior와 epsilon을 함께 warp image에 순차 적용하여 그 결과를 GT 비교 | double resampling/extra term 문제 | `L_abs`는 native corrected PAN만 |
| `q_old`와 `q_new` 동일 취급 | 하나는 known-ε response; 하나는 PAN/GT gradient discrepancy | 이름·그림·표 모두 개정 |
| `q_new`를 absolute registration GT라 주장 | spectral response/texture가 개입 | structural discrepancy proxy로 제한 |
| `w_e`를 hard switching으로 변경 | 합의된 continuous mixing 이탈 | `e/(e+τe)` 유지 |
| Student Aligner를 trainable로 놓고 A freeze라고 기재 | 실제 geometry 달라짐 | module 공유/freeze 및 parameter test |
| Teacher error median을 test GT에서 산정 | 데이터 누수 | train-only frozen Teacher |
| test FR images를 FR ε training에 넣고 standard generalization이라 주장 | 평가 설정 변경 | 금지 또는 별도 명시된 test-time adaptation 실험 |
| MIND loss를 Stage 1 학습에 사용 | 합의 위반, evaluation independence 상실 | MIND evaluation-only |
| prior 계산 시간을 Student inference cost에서 제외 | 실제 제품 pipeline 비용 누락 | scene-level amortized cost와 per-patch cost 모두 보고 |

---

## 18. 구현 서버 preflight / unit test — 실험 시작 전 PASS 필요

**수치 임계값을 근거 없이 만드는 대신, 수학적 불변성·입력 shape·gradient 흐름·자료 provenance의 PASS/FAIL을 검사한다.** Floating-point 비교 tolerance는 실제 dtype/GPU/연산에 맞춰 실험 서버가 정하고 기록한다.

### Gate A — 데이터·프로토콜

- [ ] 각 RR train patch의 source scene ID와 crop/augmentation 정보 확인.
- [ ] FR **training** native PAN/LRMS/LMS 실제 확보 및 split separation 확인.
- [ ] RR test/FR test가 teacher/student 학습이나 `τ_e,τ_q` 산정에 포함되지 않음을 증명.
- [ ] 4× PAN/LRMS ratio, `lms` shape/channel 순서, `gt`의 정의 검증.
- [ ] dataset native LMS와 reconstruction base `bicubic↑M`를 각각 hash 및 비교.
- [ ] scene prior 추정에 사용한 영상 split, scene ID, reference type 기록.
- [ ] source scene → inference patch mapping이 실제 제공되는지 확인.

### Gate B — Scene-prior coordinate

- [ ] `b_s=(dy,dx)`의 의미를 warp sampling sign과 연결한 impulse test.
- [ ] audit pair C estimator가 반환하는 `MS→PAN` shift를 `PAN→MS`로 정확히 변환.
- [ ] prior scale 변환 시 native pixel units 및 MTF decimation origin 검증.
- [ ] flip/rot90 후 `(dy,dx)`가 동일하게 변환되는지 synthetic impulse 확인.
- [ ] `m_s≈0` 또는 low-confidence source scene이 있는 경우 처리 규칙을 정의하기 전 fail-fast/flag.
- [ ] 같은 source scene의 여러 patch가 같은 global prior를 받고, 다른 scene에 새지 않는지 test.
- [ ] final correction `b_s+αd+βn`와 cartesian inverse transform round-trip test.

### Gate C — Aligner + 공통 U-Net architecture

- [ ] 기존 PAN/MS stem: 1/C→16, joint stride2, residual64, GAP, final FC 구조 기록.
- [ ] `pa/aligner.py`의 **구 `_Res(64)`가 공통 `ResBlock(64,norm="ln",mode_mod=False,dropout=0,se=False)`로 실제 교체**되었는지 확인.
- [ ] A/T/S의 ResBlock internal computation order, LN axis/affine, Conv kernel/bias/padding, second Conv zero-init 및 residual sum after-activation **완전 동일**한지 forward/type test.
- [ ] 새 Aligner 64ch block에서 합산 후 SiLU가 **없고**, shortcut이 identity인지 검증.
- [ ] Aligner stem/joint의 GN(4,8) + SiLU는 기존대로 남아 있으며 **ResBlock 내부에만 LN**이 적용됐는지 확인.
- [ ] 이전 Aligner state_dict 호환이 깨지는 범위와 scratch/reinit 또는 명시적 migration 정책을 검증. U-Net checkpoint는 변경하지 않음.
- [ ] 새 Aligner parameter count/FLOPs와 training 첫 step의 FC2/ResBlock gradient를 기록. 초기 `c0=b_s` 유지 확인.
- [ ] 새로운 scene-aware head 실제 input dimension/output dimension과 params count 기록.
- [ ] final FC zero-init 시 `c0=b_s` 재현.
- [ ] `α`, `β`가 학습되지만 `b_s`, `d_s`, `n_s`는 gradient 없는 constants임을 검증.
- [ ] Aligner가 `[B,2]`를 반환하고 모든 값 finite.
- [ ] batch 각 sample마다 scene prior를 올바르게 붙이는지 mixed-scene batch test.
- [ ] Native LMS vs reconstruction base가 코드에서 구분되며 변경값이 실제 A input으로 들어가는지 test.
- [ ] Teacher와 Student에서 **동일한 ResBlock class**를 사용하고 layer 순서/skip/zero-init이 동일한지 확인.
- [ ] Teacher/Student 모든 ResBlock 및 output norm이 Channel-wise `LayerNorm`이며 `GroupNorm`이 섞이지 않는지 test.
- [ ] PAN Aligner의 **stem/joint `GroupNorm` 유지** 및 **residual `LayerNorm` 교체**를 각각 test.
- [ ] U-Net `mode_modulation`, dropout, attention, Swin, SE의 실제 flag를 T/S 둘 다 기록하고 의도하지 않은 차이를 차단.
- [ ] Teacher/Student input stem layout, channel count, width/depth 및 output band count를 manifest로 비교하고 서로 다른 부분을 명시.
- [ ] U-Net ResBlock/Norm 자체를 Stage 1→Stage 2에서 바꾸지 않았고, Student 용량 차이만 의도적으로 준 것인지 확인.

### Gate D — ε generation, geometry, reproducibility

- [ ] ε norm이 모든 샘플에서 `≤2` current PAN pixels.
- [ ] 원판 면적 균등 sampling이 이론 분포와 일치 (통계 smoke test).
- [ ] `(dy,dx)` 순서, radian angle convention 검증.
- [ ] epsilon generator가 data augmentation/shuffle RNG를 흔들지 않는지 검증.
- [ ] ε가 augmentation **후** unshifted PAN에서 생성되어 누적되지 않음.
- [ ] ε에 PAN만 이동; `M`, `LMS`, `Y`, `b_s`는 고정.
- [ ] `Pε=warp(P,ε).detach()`이며 Teacher U-Net에 공급되지 않음.
- [ ] ε target에만 `sg(c0)`가 적용됨.
- [ ] `ε=0`/analytical synthetic input으로 `cε+ε−c0` 부호 점검.
- [ ] fixed probe로 no-response baseline gain/MAE 재현(동일 probe 정의에 한함).
- [ ] RR/FR 모두 동일한 A weights, 현재 grid 기준 `b=2` 사용.
- [ ] border-filled pixel로 ε가 노출되는 지 여부와 support ratio 진단.

### Gate E — Stage 1 loss/gradient

- [ ] `L_rec` 단독 backward가 Teacher와 Aligner의 비영(또는 유효) gradient를 만듦.
- [ ] `L_abs` 단독 backward는 **Aligner**로 gradient가 가고 Teacher U-Net에는 직접 gradient가 없음.
- [ ] `Lε_RR`, `Lε_FR` 단독 backward는 Aligner로만 gradient가 감.
- [ ] `L_abs`의 GT RGB band index, PAN warp, derivative operator, reduction이 고정된 정의와 일치.
- [ ] ε branch 활성 시 **매 optimizer update**에서 두 ε term이 모두 계산됨.
- [ ] `L_rec`, `L_abs`, `Lε_RR`, `Lε_FR`의 raw scale/weighted scale/gradient norm 로그 확인.
- [ ] NaN/Inf, saturated warp sample, empty ROI 발생 시 조용히 0 loss로 대체하지 않음.

### Gate F — Stage 2 freeze/routing

- [ ] Stage 1 A와 Teacher의 parameter gradients 없음 / parameter state hash 불변.
- [ ] Student만 gradient가 있고 optimizer parameter group에 A/T가 없음.
- [ ] Student와 Teacher가 동일한 `P̃` 입력을 받음.
- [ ] `e`/`q`가 GT train에서만 생성되고 `w_e`/`w_q`는 Student gradient에 종속되지 않음.
- [ ] frozen Teacher error histogram을 기준으로 `τ_e` 산정 후 저장.
- [ ] frozen A/GT gradient residual 기준으로 `τ_q` 산정 후 저장.
- [ ] `e=0`, `e=τ`, `e=2τ` 등의 blending 단위 테스트.
- [ ] q 증가 시 `w_q`가 단조 증가하는지, q=0·τq handling 검증.
- [ ] gradient operator 및 spatial loss normalization이 Stage 1과 정확히 일치.
- [ ] FR Student inference에서 Teacher/GT/e/q를 참조하는 코드 경로가 없음.

---

## 19. 연구 실험 계획 — **방법 검증 먼저**, 추가 튜닝 나중

### 19.1 제안되는 실험 순서 (개별 run 식별자/예산은 서버에서 확정)

| 순서 | 실험/진단 단위 | 반드시 확인할 질문 | 선행조건 |
|---|---|---|---|
| `P0` | 데이터·scene linkage·prior estimator preflight | 실제 scene-level `b_s`를 train/inference에 일관되게 줄 수 있는가? | data provenance |
| `P1` | Warp/augment/ε generator pure unit tests | `(dy,dx)` sign, scale, known-ε relation 정확한가? | 코드 hash pin |
| `P2` | Fixed prior-only sanity (`Δu=0` 진단) | estimated scene shift 자체가 native PAN↔MS correspondence를 개선하는가? | P0–P1 |
| `P3` | Prior-aware head + **Aligner↔U-Net common ResBlock** smoke test | 기존 Aligner GN ResBlock 제거, 공통 LN ResBlock reuse, FC zero-init/gradient routing/αβ 수렴 및 T/S U-Net 불변을 검증 | P2 |
| `P4` | Stage 1 full (RR rec+abs+ε, FR ε) | RR/FR known shift gain과 reconstruction을 함께 개선하는가? | admissible FR train source |
| `P5` | Frozen A/T 후 `τ_e,τ_q` calibration 및 q/e quadrant 재작성 | 새 `q_new`가 독립적 신뢰 축으로 의미 있는가? | P4 |
| `P6` | Stage 2 frozen-A e/q fitting | Student가 RR/FR 품질을 유지·개선하는가? | P5 |
| `P7` | 핵심 contribution ablations | scene prior / RR abs / FR ε / e / q의 개별 이득을 분리할 수 있는가? | P4–P6 |
| `P8` | 정식 다중 seed + sensor replication/리포트 | 우연 seed나 특정 scene 편향이 아닌가? | P7 결과 확인 |

**GPU 시간/seed 수/step 수/실험 마감은 현재 문서에서 사실로 정하지 않는다.** 서버 자원·기존 recipe에 맞춰 등록하고, 각 실험 간 총 학습량과 selection protocol을 동등하게 유지한다.

### 19.2 최소 대조군 설계 `[PROPOSED]`

**Stage 1:**

| 대조군 | Scene prior | RR absolute | RR ε | FR ε | 목적 |
|---|---|---|---|---|---|
| Previous reference | 기존 방식 | 없음 | 기존 방식 | 없음 | 과거 재현 및 차이 범위 측정; recipe 다른 경우 직접 인과 비교 금지 |
| Prior-only control | global fixed `b_s`, residual 고정 0 | 상황에 따라 별도 recon training | 없음 | 없음 | matching prior 자체의 기여 |
| No-FR control | new `b_s+residual` | 사용 | 사용 | **없음** | FR ε branch의 효과 |
| No-absolute control | new `b_s+residual` | **없음** | 사용 | 사용 | GT spatial anchor 효과 |
| Full S1 | new `b_s+residual` | **사용** | **사용** | **사용** | 최종 Stage 1 |

이 표는 **ablation 방향**이다. Full 대비 하나씩 제거하되 optimizer/data/architecture/seed를 일치시켜야 한다. 적절한 `Prior-only` 재학습 조건은 정해지지 않았으므로 서버에서 공정 비교를 설계한다.

**Stage 2:**

| 대조군 | Geometry | Loss routing | 목적 |
|---|---|---|---|
| S-GT | shared frozen A | GT L1 only | Student 기본 성능 확인 |
| S-e | shared frozen A | e-based soft/hard blend | soft/hard routing의 이득 확인 |
| S-eq | shared frozen A | e blend + q-weighted GT edge | **이번 full Stage 2** |

`Student trainable Aligner clone`, `q_old`, `hard switching`, direction-only prior, sensor-specific radius, **추가적인 ResBlock/Norm architecture sweep**은 **후속 선택적 ablation**으로 미룬다. 이번 baseline에는 **Aligner ResBlock→U-Net 공통 블록 교체가 포함**되며, 기존 Aligner 블록과의 통제 비교는 필요 시 별도 ablation으로 수행한다.

### 19.3 Acceptance/Go–No-Go 기준의 작성 원칙

정량 pass threshold는 **기존 실험 변동성/validation distribution** 없이 정하면 근거 없는 수치다. 서버에서 적어도 다음 정보를 확인한 뒤 기준을 고정한다.

- ε gain 및 MAE의 기존 run 분산/seed-to-seed scatter와 evaluation artifact noise.
- RR/FR quality metrics의 통계적 변동과 sensor별 분산.
- scene prior estimator confidence/matching disagreement 분포.
- actual wall time, throughput, OOM boundary, preprocessing runtime.

Go/No-Go의 **논리 조건**은 아래처럼 미리 지정한다.

| Phase | 반드시 성립해야 하는 논리 |
|---|---|
| Feasibility | data split/provenance 명확; scene ID mapping; FR training inputs; prior runtime 가능 |
| Correctness | `c0≈b_s` init, ε sign 정합, gradient routing·freeze 성립 |
| Alignment | RR/FR의 known shift response가 무반응 baseline과 구분되며 비정상 collapse 없음 |
| Task | RR reconstruction 및 FR quality가 대조군 대비 악화되는지 숨기지 않음 |
| Guidance | 새로운 q가 GT spatial fitting의 합리적인 난이도/구조 signal인지 확인 |
| Deployment | Student-only 경로에서 inference prior + frozen A + F_S만 실행 |

방법이 이 조건을 만족하지 못하면 즉시 다른 loss/weight를 붙여 “완성”시키지 않고 **실패한 구성 요소를 고립시켜 반례와 함께 기록**한다.

---
