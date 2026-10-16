# PANDA Stage 2 — GT-Hard + Selective Teacher-Soft + Conditional-q Loss

**Implementation Specification / 2026-10-16 / Draft v1.0**

> **상태:** 구현 지향 기술 명세서. **현재 저장소에 구현/실험된 결과가 아님.** 기존 결정과 이번 세션에서 합의한 수식 방향을 기술하되, 아직 정해지지 않은 수치 하이퍼파라미터와 calibration 세부값은 **[잠정] / [미결정]**으로 표시한다.
>
> **대상:** `hojunking/PAN-Crafter-repro`, 조사한 `main` HEAD `828a1aa4349394365d1eef638579570d4613dc40` (2026-10-16 조회). 기존 `directshift48/` Stage 1 및 `donor30/` MASK2 캠페인은 **원본 보존**. 새로운 namespace 또는 분리된 실험 branch에 신규 Stage 2 loss/revision으로 추가한다.

## 0. 요구사항 및 불변조건

### 0.1 확정된 모델·학습 계약

| 항목 | 구현 요구사항 |
|---|---|
| Frozen Aligner | Stage 1 `DirectPANAligner` **하나**를 Teacher와 Student가 공유. Stage 2에서 업데이트 금지 |
| Frozen Teacher | Stage 1 `Reconstructor` W112 / depths `(1,2,3)` / `P0` / `BICUBIC_MS` |
| Trainable Student | Teacher와 **동일 구조** W112 / depths `(1,2,3)` / `P0` / `BICUBIC_MS` |
| Student weight initialization | **Fresh random**, `role="S"`로 별도 생성. Teacher state를 Student에 복사 금지 |
| 업데이트 대상 | Student reconstruction U-Net의 모든 trainable parameter만 |
| 학습 데이터 | 선택한 센서의 **RR train** 및 HRMS GT, frozen donor A/T |
| 입력 좌표 | MS 기준 프레임 고정, **원본 PAN만** frozen Aligner가 예측한 shift로 warp |
| 기본 GT Hard | **전체 픽셀 × 모든 MS 밴드** 감독, warp-valid loss mask **미사용** |
| Teacher Soft | Teacher-advantage gate가 양수인 위치의 작은 보조항 |
| Scharr `q` | 보정된 PAN 대 GT RGB의 구조 차이; **별도 output spatial loss가 아니라 GT Hard weight에만 이용** |
| Conditional `r_q` | **동일 센서, 전체 RR train pixel population**에서 GT edge strength별 조건부 q ECDF |
| ROI8 | q/g 통계와 q-Hard 추가 weight는 fixed interior ROI8에서만 적용. ROI 밖 q 추가 weight=0 |
| 추론 | Frozen Aligner + Student만 사용. **GT/Teacher/q/CDF/soft를 전혀 호출하지 않음** |

### 0.2 도입 금지 사항

- Stage 2의 `LPAN/HPAN`, trainable Student Aligner, PAN을 x4로 확대한 shift, Stage 2 ε-consistency, FR self-supervision.
- Teacher/GT 상호배타적 L1 혼합: `(1-w_e)·|S-T| + w_e·|S-GT|`.
- Student 예측의 Scharr를 GT/PAN과 직접 맞추는 별도 spatial output loss.
- 옛 Fig. 5의 `q_epsilon` 또는 `q_resp`를 현재 q-Hard에 혼용.
- 기존 masked `tau_e`, `tau_q`나 `donor30` calibration artifact의 무검증 재사용.

### 0.3 설계 / 검증 상태 표시

- **[확정 방향]** GT-always Hard + selective Teacher Soft의 **두 항**, fresh same-capacity Student, frozen A/T, no separate spatial, conditional-q의 Hard 사용.
- **[채택한 구현 설계안]** Hard weight의 additive form `1 + alpha*d_e + gamma*d_e*r_q_bar`; full-image mean; ROI 밖 `r_q_bar=0`.
- **[잠정]** GT edge-strength bin 개수 `K=5`; exact ECDF의 상세 직렬화 방식; 초기 실험 계수 조합.
- **[미결정]** `alpha`, `beta`, `gamma`, 안정화용 `delta`의 실험 최종값, donor checkpoint 선택, raw-q 대조 정규화의 정확한 형태. **명세서 작성 자체가 실험 승인/학습 시작을 뜻하지 않음.**

---

## 1. 완전한 notation / tensor shape

이 문서에서 등장하는 모든 수학 기호 및 코드 변수의 의미를 명시한다.

### 1.1 크기·인덱스·데이터

| 기호/변수 | 뜻 | 예시 또는 shape |
|---|---|---|
| `B` | 현재 forward의 batch 내 패치 수 | 4 또는 48 |
| `H` | PAN-grid 패치 세로 픽셀 수 | RR train 64 |
| `W` | PAN-grid 패치 가로 픽셀 수 | RR train 64 |
| `C` | MS 채널(밴드) 수 | WV3=8, GF2/QB=4 |
| `n` | 배치 속 샘플 인덱스 | `1,…,B` |
| `h,w` | PAN-grid 내 행/열 인덱스 | `1,…,H`, `1,…,W` |
| `p=(h,w)` | 패치 내 한 픽셀의 위치 | 좌표 쌍 |
| `c` | MS 밴드 인덱스 | `1,…,C` |
| `P` / `pan` | 원본 PAN | `[B,1,H,W]` |
| `M` / `ms` | 저해상도 MS(LRMS) | `[B,C,H/4,W/4]` |
| `L_A` / `native_lms` | PAN-grid에 **저장된** LMS. Aligner의 MS-side 입력 | `[B,C,H,W]` |
| `B_M` / `ms_base` | LRMS의 bicubic 업샘플링 결과. Reconstructor residual의 기준 | `[B,C,H,W]` |
| `Y` / `gt` | RR train HRMS GT | `[B,C,H,W]` |
| `A_T` / `aligner` | Stage 1에서 확보한 frozen DirectPANAligner | 모듈 |
| `F_T` / `teacher` | frozen Teacher reconstruction U-Net | 모듈 |
| `F_S` / `student` | fresh trainable Student reconstruction U-Net | 모듈 |
| `u=(dy,dx)` / `correction` | 패치마다 Aligner가 예측한 PAN-grid translation | `[B,2]` |
| `dy,dx` | PAN 샘플링 위치의 행/열 방향 이동량 | current PAN pixel 단위 |
| `\mathcal W(P,u)` / `warp_pan` | 원본 PAN에 단 한 번 적용하는 bicubic warp | `[B,1,H,W]` |
| `\widetilde P` / `pan_aligned` | 이동 완료된 PAN | `[B,1,H,W]` |
| `\hat Y_T` / `teacher_y` | Teacher가 예측한 HRMS | `[B,C,H,W]` |
| `\hat Y_S` / `student_y` | Student가 예측한 HRMS | `[B,C,H,W]` |
| `\Omega` | 전체 PAN-grid 픽셀 집합 | `H×W` |
| `\Omega_8` | 사방 8픽셀 제외한 fixed interior 영역 | RR 64×64이면 48×48 |
| `\langle f\rangle` | batch+전체 공간 픽셀의 **산술평균** | 아래 수식 |

정확한 평균의 뜻:

\[
\langle f\rangle
=\frac{1}{BHW}\sum_{n=1}^{B}\sum_{h=1}^{H}\sum_{w=1}^{W} f(n,h,w).
\]

- `BHW = B × H × W`: 현재 forward의 전체 샘플-픽셀 개수. 예: `B=4, H=W=64`이면 `BHW=16,384`.
- `e_T`, `e_S`, `Delta_TS`에서 **밴드 평균(`1/C`)을 먼저 취했으므로**, 최종 `\langle\cdot\rangle`에는 `C`를 다시 넣지 않는다.
- `torch.Tensor.mean()`이 `[B,1,H,W]` 맵 전체에 적용되면 정확히 위 식이다. 서로 다른 이미지 크기나 불균등 배치를 섞는 경우에는 명시적인 pixel count로 가중해야 한다.

### 1.2 감독·edge·가중치 표기

| 기호/변수 | 정의/의미 | shape/범위 |
|---|---|---|
| `e_T(p)` | Teacher–GT **전 밴드 평균 절대오차** | `[B,1,H,W]`, ≥0 |
| `e_S(p)` | Student–GT **전 밴드 평균 절대오차** | `[B,1,H,W]`, ≥0 |
| `Delta_TS(p)` | Student–Teacher **전 밴드 평균 절대오차** | `[B,1,H,W]`, ≥0 |
| `tau_e` | RR train 전체 픽셀의 `e_T` **전역 중앙값** | scalar, ≥0 |
| `d_e(p)` | Teacher error에 비례하는 bounded difficulty | `[B,1,H,W]`, [0,1] |
| `a_T(p)` | Teacher가 Student보다 정확할 때만 켜지는 advantage | `[B,1,H,W]`, [0,1] (delta>0) |
| `delta` / `δ` | advantage 분모 안정화용 작은 **양의 상수** | scalar, >0 |
| `S_x,S_y` | 가로/세로 Scharr filter(정확한 커널은 §3) | 고정 필터 |
| `*` | 2D convolution/상관 연산(코드는 `conv2d`) | 연산자 |
| `Y_RGB` | GT 중 RGB 3밴드의 **pixel-wise 평균** | `[B,1,H,W]` |
| `g(p)` | GT RGB의 두 축 Scharr 절대응답 평균: **GT edge strength** | `[B,1,H,W]`, ≥0 |
| `q(p)` | corrected PAN과 GT RGB 사이 두 축 절대 Scharr 응답의 차이 평균 | `[B,1,H,W]`, ≥0 |
| `K` | `g` 값을 나누는 조건부 그룹 개수 | 양의 정수, 예: 5 [잠정] |
| `\mathcal B_j` | 전역 train `g` 분포로 만든 j번째 edge-strength 그룹 | `j=0,…,K_eff-1` |
| `K_eff` | 동일 분위수 경계 병합 이후 실제 비어 있지 않은 그룹 수 | `1,…,K` |
| `F_{q|G\in\mathcal B_j}` | 그룹 `j` 내 train q의 경험적 누적분포(ECDF) | 함수, [0,1] |
| `r_q(p)` | 현재 pixel q의 해당 g 그룹 내 조건부 누적 백분위 | `[B,1,H,W]` 내부 ROI 값 |
| `\bar r_q(p)` / `r_q_bar` | ROI8 안에는 `r_q`, **밖에는 0**인 지도 | `[B,1,H,W]`, [0,1] |
| `w_H(p)` | GT Hard에 곱하는 pixel-wise 공간 weight | `[B,1,H,W]`, ≥1 |
| `alpha` / `α` | Teacher difficulty 기반 GT Hard 추가 강도 | ≥0 |
| `gamma` / `γ` | q 기반 GT Hard **추가** 강도 | ≥0 |
| `beta` / `β` | Teacher Soft 항의 전체 강도 | ≥0 |
| `L_H` / `hard_weighted` | GT Hard 총손실 | scalar |
| `L_K` / `soft_unscaled` | beta를 곱하기 **전** selective Soft | scalar |
| `L_S2` / `loss` | `L_H+beta*L_K`, 최종 학습 loss | scalar |
| `sg(z)` | `stop-gradient`, PyTorch `z.detach()` | 연산자 |
| `[z]_+` | `max(z,0)`, PyTorch `clamp_min(0)` | 연산자 |

`S_x,S_y`는 edge 방향의 **Scharr 응답**을 얻는 필터이며 `q`는 **signed gradient의 직접 차이도, gradient vector의 Euclidean magnitude 차이도 아니다**. 먼저 축별 응답의 절댓값을 취한 후 차이를 계산한다.

---

## 2. Forward graph와 모델 계약

\[
\begin{aligned}
u &= A_T(P,L_A),\\
\widetilde P &= \mathcal W(P,u),\\
B_M &= \mathcal U_{\mathrm{bicubic}}(M),\\
\hat Y_T &= B_M+F_T([\widetilde P,B_M]),\\
\hat Y_S &= B_M+F_S([\widetilde P,B_M]).
\end{aligned}
\]

- `A_T` 입력: **원본 PAN `P` + stored LMS `L_A`**. `M` 또는 `B_M`를 Aligner stem에 넣지 않는다.
- `u=(dy,dx)`는 PAN-grid pixel 단위. `warp_pan(P,u)(y,x)=P(y+dy,x+dx)`; code는 `grid_sample(mode="bicubic",padding_mode="border",align_corners=False)` 및 FP32.
- `B_M`은 `F.interpolate(M,scale_factor=4,mode="bicubic",align_corners=False)` 결과. `L_A`와 다른 자료/phase이다.
- Teacher와 Student는 **동일한** `\widetilde P`, 동일한 `M`, 동일한 `L_A`를 사용해 `Reconstructor.forward()`를 호출한다. `Reconstructor`는 내부에서 `B_M`를 직접 생성한다.
- Teacher와 Aligner: `eval()`, `requires_grad_(False)`, `torch.no_grad()`; Student: `train()`, `requires_grad_(True)`.
- Teacher–Student 간 파라미터 공유 금지. 새 Student 생성 시 `build_reconstructor(ReconstructionSpec(...), seed=...,role="S")` 사용; Teacher state copy 금지.
- 학습 optimizer의 parameter set은 **정확히** `F_S.parameters()`와 같아야 한다. 동결된 A/T는 optimizer에서 제외한다.
- 기존 `DirectPANAligner` 내부 **input view crop8**은 그대로 유지. 이 crop은 output loss ROI나 warp-valid mask와 다른 개념이다.

### 2.1 기존 코드와 비교했을 때의 차이

**실제 확인한 코드 (조회한 `main` HEAD 기준)**

| 기존 코드 | 확인된 행위 | 새 설계에서 해야 할 일 |
|---|---|---|
| `directshift48/method.py::DirectPANAligner` | native PAN/stored LMS 입력, crop8, sample-wise shift | 수정 없이 재사용 |
| `directshift48/method.py::ScharrAbsolute` | RGB mean, Scharr x/y `/32`, reflect pad 1 | `descriptor`, `rgb_luminance` 재사용 |
| `pa/warp.py::warp_pan` | FP32 bicubic + border + align_corners False | 그대로 재사용 |
| `panda_method/reconstruction.py::Reconstructor` | residual + bicubic MS, 딕셔너리 `{'y': ...}` | W112-D123/P0로 동일 T/S 구축 |
| `donor30/method.py::loss_terms` | E식 `(1-w_e)*soft + w_e*hard`; EQ 별도 Scharr; valid mask | **복사하지 말고 신규 loss 함수 작성** |
| `donor30/method.py::DirectStage2` | `MaskedDonorStage1` 의존, 옛 모드, `lambda_spatial` | 신규 Stage 2 클래스/버전 정의 |
| `donor30/method.py::DirectCalibrationScales` | masked train e 중앙값 + q 중앙값 `tau_q` | 신규 **unmasked `tau_e` + conditional ECDF** 형식으로 교체 |
| `donor30/masking.py` | bicubic support mask로 GT/Teacher fitting pixel 제거 | **새 Hard/Soft reduction에 전달하지 않음** |
| `donor30/training.py::StudentTrainer` | Student-only optimizer, 12×4 accumulation, 체크포인트 계약 | 원리 재사용, 새 명세/메트릭/스키마로 별도 구현 |

**주의:** `donor30`의 frozen donor가 MASK2 방식으로 학습됐더라도, Stage 2에서 **Hard/Soft만 unmasked**로 변경할 수 있다. 이는 **donor 자체를 재학습하거나 donor 손실 이력을 바꾸는 것과 다르다.** 새 full-image `tau_e`와 conditional-q calibration은 반드시 다시 수행한다.

---

## 3. 픽셀별 신호 정의

### 3.1 Teacher/Student reconstruction errors

\[
\begin{aligned}
e_T(n,p)&=\frac{1}{C}\sum_{c=1}^{C}|\hat Y_{T,n,c}(p)-Y_{n,c}(p)|,\\
e_S(n,p)&=\frac{1}{C}\sum_{c=1}^{C}|\hat Y_{S,n,c}(p)-Y_{n,c}(p)|,\\
\Delta_{TS}(n,p)&=\frac{1}{C}\sum_{c=1}^{C}|\hat Y_{S,n,c}(p)-\hat Y_{T,n,c}(p)|.
\end{aligned}
\]

각 결과 shape `[B,1,H,W]`. `e_T`는 no-grad Teacher prediction에서 생성; `e_S`는 **Student gradient 유지**; `Delta_TS` 역시 **Student gradient 유지**.

### 3.2 Difficulty / Teacher advantage

\[
\boxed{d_e=\operatorname{sg}\!\left(\frac{e_T}{e_T+\tau_e}\right)}
\qquad
\boxed{a_T=\operatorname{sg}\!\left(\frac{[e_S-e_T]_+}{e_S+\delta}\right)}.
\]

- `tau_e`: 선택한 **동일 frozen donor**가 **RR train 전 픽셀**에 대해 만든 `e_T` 값의 중앙값. `donor30` MASK2가 저장한 **valid-pixel median을 그대로 쓰지 않는다**.
- `delta>0`: denominator 안정화; **설정/manifest에 기록하는 필수 수치**. 예시 `1e-6`은 초기 후보일 뿐 확정된 최적값이 아니다.
- `d_e`는 모든 픽셀에서 계산되고, `a_T`는 Student 상태에 따라 **매 update 재계산**된다.
- `a_T` 계산에 사용하는 `e_S`는 `e_S.detach()`여야 한다. Soft error `Delta_TS`는 detach하지 않는다.
- `tau_e=0`이고 `e_T=0`인 픽셀에선 `d_e=0`으로 정의한다(0/0 회피). 코드에서는 분모를 FP32 `torch.finfo(...).tiny` 이상으로 clamp한다.
- `e_S=e_T`이면 `a_T=0`; Teacher가 더 나쁘면 `a_T=0`.

### 3.3 Scharr operator — 정확한 재현 규약

실제 `directshift48/method.py::ScharrAbsolute.descriptor()`에 맞춘다.

\[
S_x=\frac1{32}
\begin{bmatrix}-3&0&3\\-10&0&10\\-3&0&3\end{bmatrix},
\qquad S_y=S_x^{\mathsf T}.
\]

1. PAN/GT image 데이터는 training pipeline의 **실제 FP32 radiometry 단위**를 그대로 사용. edge 계산 직전 z-score, 추가 normalization, clipping 금지.
2. `Y_RGB=(Y_R+Y_G+Y_B)/3`를 **먼저** 만든 후 Scharr 적용.
3. Reflect padding 1 pixel을 사용해 3×3 filter 응답을 원본 그리드에 맞춘다.
4. `descriptor(X) = [ |S_x*X|, |S_y*X| ]`: `[B,2,H,W]`.
5. 현재 센서 RGB index (0-based): `WV3=(4,2,1)`, `GF2=(2,1,0)`, `QB=(2,1,0)`.

이때

\[
\begin{aligned}
q(p)&=\frac12\sum_{k\in\{x,y\}}\left| |S_k*\widetilde P|(p)-|S_k*Y_{RGB}|(p)\right|,\\
g(p)&=\frac12\sum_{k\in\{x,y\}}|S_k*Y_{RGB}|(p).
\end{aligned}
\]

- `q,g` 둘 다 `[B,1,H,W]`.
- q는 **수치적으로 PAN/GT의 cross-modal 구조 차이**이고 **실제 displacement GT, 정합 실패 확률, Teacher q의 우열**이 아니다.
- q/g는 weight 계산에만 쓰이므로 **항상 gradient detach**.
- **Scharr ROI8**: q/g를 통계 또는 weight로 사용할 때 `[...,8:-8,8:-8]`만 택한다. RR64라면 48×48. 그 외 좌표의 `r_q_bar`는 0.
- PAN warping이 ROI8 내 구조의 Scharr **3×3 stencil에서 원본 관측 이외의 border padding 값**을 읽는 경우 기존 Stage 1의 conservative support guard(`_native_support` 또는 동등 규칙)가 계속 실패 처리해야 한다. 이는 **loss pixel mask를 적용한다는 뜻이 아니다**. `warp_support_mask`를 기록/guard 판단에는 사용 가능하지만 GT/Soft reduction의 분자나 분모에는 넣지 않는다.

---

## 4. `g` 조건부 `q` 전역 ECDF calibration

### 4.1 통계적 정의

전역 RR train에서 ROI8 픽셀 `(g_i,q_i)` 쌍을 모은다. `g`를 전역 분위수로 나눈 `K`개 그룹 `\mathcal B_j`를 정의하고, 각 그룹에서 q의 오른쪽 연속 경험적 CDF를 만든다.

\[
\boxed{
 r_q(p)= F_{q\mid G\in\mathcal B(g(p))}(q(p))
 =\frac{\#\{i:g_i\in\mathcal B(g_p),\,q_i\le q_p\}}
 {\#\{i:g_i\in\mathcal B(g_p)\}}.
}
\]

중요: **한 train patch에서 q를 정렬하면 안 된다. 현재 mini-batch마다 g/q 분포를 재학습하면 안 된다.** Training-global 고정 calibration이어야 한다.

### 4.2 필수 calibration 순서

1. **Donor 선택 및 고정:** Stage 1 Aligner와 Teacher를 **같은 checkpoint**에서 로드, hash 기록, eval/no-grad.
2. **데이터 목록 고정:** 해당 센서의 RR train sample ID 전체 inventory와 source manifest SHA256 고정. train 전용. validation/test/FR 샘플 유입 금지.
3. **Unaugmented train iterator:** 기존 calibration의 unaugmented registered sample 규약을 기본으로 유지. Native `P`, `M`, `L_A`, GT를 FP32로 읽는다. Stage 2 학습 중 augmentation이 있는 경우 그 차이는 로그에 기록한다.
4. Frozen A→aligned PAN, Frozen Teacher→`teacher_y` 한 번씩 실행. `e_T`는 **전체 64×64** 픽셀에서 수집; Scharr의 `g,q`는 **ROI8 48×48** 픽셀에서 쌍으로 수집.
5. 각 batch의 `q/g` ROI 사용 전 `_native_support` 동등 guard로 full 3×3 stencil support 확인. 원인 없는 부분 삭제/자동 clamp 금지.
6. `tau_e = median(all e_T values)` 계산. 중앙값의 표본 개수가 짝수면 가운데 두 값 평균(`midpoint_even_global_pixels`).
7. `g` 전체 ROI train pixel 값으로 `K`개의 분위 경계를 계산. **[잠정 초기값: K=5]**.
8. 각 `g` 그룹 내 `q`만 분리 후 오름차순 정렬해서 q ECDF 표 저장.
9. calibration artifact에 모든 판별정보, 경계값, q 정렬 배열, 개수, 중앙값, 해시, 알고리즘 정의를 저장하고 **immutable**하게 고정.
10. Stage 2 시작/재개 전에 현재 donor/data/operator와 calibration의 binding이 일치하는지 검증. 하나라도 불일치하면 **fail-fast / recalibrate**.

### 4.3 ECDF의 정확한 수치 규약 (권장 reference implementation)

- `K`를 결정했으면 train `g` 전체에서 `j/K` 분위 경계(`j=1,…,K-1`)를 산출한다. 예: NumPy quantile의 `method='linear'` 사용; API/버전/float 자료형을 manifest에 기록.
- 분위 경계가 중복될 경우 **중복 경계만 제거**하고 실제 `K_eff`를 기록한다. 이때 비어 있지 않은 bin에만 q 정렬 데이터를 둔다. 내부 경계값과 같은 g는 `searchsorted(...,side='right')` 규칙으로 **상위 그룹**에 배치한다.
- `g` 경계는 **FP64** 숫자로 저장하며 학습 시에도 `g` 값을 비교할 때 동일한 FP64 경계 비교 규칙을 사용한다(Scharr 생성은 FP32). Calibration과 학습의 bin 배정 경계가 달라지지 않도록 함.
- 그룹 `j`의 q 벡터 `sorted_q_j`는 원래 FP32 Scharr 결과를 오름차순 정렬한 값. **exact** 모델에서는 정렬 표본 전체를 보존한다.
- 예측 q 값 `v`의 ECDF는 다음처럼 정의한다.

  ```python
  count_leq = torch.searchsorted(sorted_q_j, v, right=True)
  r_q = count_leq.float() / len(sorted_q_j)
  ```

- `right=True`는 수학식의 `q_i <= q_p`와 일치한다. **tie가 많은 경우 zero q도 r_q=0이라고 가정하면 안 된다.** 실제 ties 비율을 기록한다. `v`가 모든 표본보다 작으면 0, 크거나 같으면 1.
- CDF는 **확률적인 정합 confidence가 아니라 조건부 상대 순위**. 각 g 그룹에서 높은 percentile이 독립적인 spatial failure를 의미한다는 주장은 금지.
- 학습 성능을 위해 GPU `searchsorted` 벡터화를 사용 가능하다. 단 메모리 예산에 맞지 않으면 **고정 quantile knot/CDF LUT**로 근사해도 되나 이는 **다른 calibration version**으로 manifest에 해상도/보간/최대 오차/경계 규약을 명시하고 exact reference와 비교해야 한다. 묵시적인 근사/자동 fallback 금지.

### 4.4 Conditional-q lookup 참조 pseudocode

아래 `lookup_exact`는 정확한 ECDF를 제공하기 위한 데이터 경로의 **참조 코드**다. 보정 통계는 **한 번만** 학습 전 계산·저장하며, 매 train batch에서는 저장된 경계/표본을 조회한다.

```python
import torch

@torch.no_grad()
def lookup_exact(g_roi, q_roi, *, g_cutpoints_f64, q_sorted_by_bin):
    """g_roi,q_roi: matching FP32 [B,1,h_roi,w_roi] maps.
    g_cutpoints_f64: unique interior g-boundaries, sorted FP64.
    q_sorted_by_bin: list of sorted FP32 tensors for every nonempty g bin.
    All tensors must be on the same device before execution.
    """
    assert g_roi.shape == q_roi.shape
    assert len(q_sorted_by_bin) == len(g_cutpoints_f64) + 1
    g_flat = g_roi.detach().reshape(-1).double()
    q_flat = q_roi.detach().reshape(-1).float()
    groups = torch.bucketize(g_flat, g_cutpoints_f64, right=True)
    ranks = torch.empty_like(q_flat)
    for j, sorted_q in enumerate(q_sorted_by_bin):
        chosen = (groups == j)
        if not chosen.any():
            continue
        assert sorted_q.numel() > 0 and sorted_q.dtype == torch.float32
        counts = torch.searchsorted(sorted_q, q_flat[chosen], right=True)
        ranks[chosen] = counts.to(torch.float32) / sorted_q.numel()
    return ranks.reshape_as(q_roi)
```

**호출 전:** 각 그룹의 정렬된 q tensor 및 FP64 cutpoint를 장치에 올려두거나 명시적인 캐시 정책을 사용한다. 위 함수는 자동 `.to(device)` 호출이나 calibration 재집계를 수행하지 않는다. 대규모 저장 배열의 CPU/GPU 메모리 및 조회 비용을 smoke/profiling으로 측정해야 한다. `torch.bucketize(right=True)`와 `searchsorted(right=True)`의 정확한 경계/tie 규칙은 테스트로 확인한다.

### 4.5 저장 파일(새 schema 예시)

`condq_calibration_v1.json` (metadata), `condq_sorted_q_v1.npz` (배열) 등을 사용한다. 파일명은 구현시 조정 가능하지만 아래 필드는 필수.

```yaml
schema: PANDA_S2_CONDQ_TRAIN_GLOBAL_V1
sensor: WV3                     # or GF2 / QB
source_split: RR_train_only
sample_ids_sha256: REQUIRED
source_manifest_sha256: REQUIRED
aligner_state_sha256: REQUIRED
teacher_state_sha256: REQUIRED
donor_checkpoint_id: REQUIRED
reconstruction_spec: {bands: 8, width: 112, depth: [1, 2, 3], input_layout: P0, base_reference: BICUBIC_MS}
scharr: {kernel: "absolute_xy_/32", padding: reflect1, rgb_indices: [4, 2, 1], roi_margin: 8}
warp: bicubic_border_FP32_align_corners_false_current_pan_px
teacher_error_reduction: all_bands_mean_all_train_pixels_unmasked
tau_e: CALIBRATED_VALUE
n_e: CALIBRATED_PIXEL_COUNT
g_grouping: {k_requested: 5, k_effective: CALIBRATED, quantile_method: linear, boundary_side: right, edge_dtype: float64}
q_ecdf: {operator: right_continuous_leq, dtype: float32, storage: exact_sorted_arrays}
n_q: CALIBRATED_ROI_PIXEL_COUNT
bin_counts: [CALIBRATED_COUNTS]
calibration_artifact_sha256: REQUIRED
```

`n_e`는 전체 패치 수×전체 픽셀 수, `n_q`는 전체 패치 수×ROI 픽셀 수(동일 크기 입력, 배치 미누락 전제)이다. **두 숫자를 혼동하지 않는다.** 기존 `tau_q`는 새 공식에 필요 없다.

---

## 5. Stage 2 수식 — 구현 최종 후보

### 5.1 Hard: GT is always supervised

\[
\boxed{
 w_H(p)=1+\alpha\,d_e(p)+\gamma\,d_e(p)\,\bar r_q(p)
}
\]

\[
\boxed{
 \mathcal L_H=\langle w_H\,e_S\rangle.
}
\]

- `alpha >= 0`: Teacher error로 **기존 PANDA Hard**를 강화.
- `gamma >= 0`: `d_e`가 큰 위치 중 `r_q`까지 큰 곳을 **추가 강화**.
- `w_H >= 1`. Teacher가 매우 정확한 위치(`d_e≈0`)에서는 q가 높아도 Hard는 거의 기본값. 이것은 q를 독립적인 spatial loss가 아니라 **Teacher difficulty를 조정하는 보조조건**으로 사용한다는 설계 선택이다.
- ROI8 외부 `r_q_bar=0`이므로 `w_H=1+alpha*d_e`. GT 감독 자체는 ROI 외부에서도 **항상 적용**.
- `q`가 RGB 평균에서 얻어져도 **Hard error는 모든 C개 밴드를 감독**. RGB 외 밴드에 다른 weight를 주는 band-aware 모델이 아니다.

분해해 로깅하면:

\[
\mathcal L_H
=\underbrace{\langle e_S\rangle}_{L_{GT}}
+\alpha\underbrace{\langle d_e e_S\rangle}_{L_{e\text{-extra}}}
+\gamma\underbrace{\langle d_e\bar r_q e_S\rangle}_{L_{q\text{-extra}}}.
\]

`L_GT`, `L_e-extra`, `L_q-extra`는 각각 **계수 미포함 raw scalar**로 저장하면 계수 변경 효과를 분리할 수 있다.

### 5.2 Soft: selectively learn from an advantageous Teacher

\[
\boxed{
 \mathcal L_K=\left\langle (1-d_e)\,a_T\,\Delta_{TS}\right\rangle.
}
\]

- Teacher가 Student보다 GT에 가까운 `a_T>0` 위치에만 Teacher prediction fitting.
- Teacher error가 높으면 `1-d_e`가 감소하여 Teacher target에 대한 신뢰를 줄인다.
- `a_T`는 detach해야 하지만, `Delta_TS`는 detach하면 안 된다. Teacher prediction만 detach.
- Soft는 GT Hard를 **대체하지 않는다**. 두 supervision weight의 합을 1로 강제하지 않는다.

### 5.3 Total

\[
\boxed{
 \mathcal L_{S2}=\mathcal L_H+\beta\mathcal L_K.
}
\]

**최종 학습 gradient의 유일한 수신자**는 Student U-Net이다. 이 손실에는 `q-spatial`, trainable Aligner penalty 또는 ε-consistency가 없다.

### 5.4 수식 sanity check

| 조건 | 기대 결과 |
|---|---|
| `alpha=gamma=beta=0` | 일반적인 전 밴드 GT-only L1과 **정확히 동일** |
| `gamma=0`, `alpha>0`, `beta>0` | q와 무관한 **PANDA-e Hard+Soft** |
| `r_q_bar=0` | `w_H=1+alpha*d_e` |
| `d_e=0` | `w_H=1`, `L_K`는 advantage로 결정 |
| `a_T=0` | 그 픽셀의 Soft gradient=0 |
| `r_q_bar=1`, `d_e≈1` | `w_H≈1+alpha+gamma` |
| ROI8 외부 | **GT Hard+Teacher Soft 둘 다 유효**, q 추가 강화만 사라짐 |
| `student_y=gt` | GT reconstruction error=0, advantage도 0 (delta>0) |
| `teacher_y=gt`, `student_y!=gt` | Hard 기본 유효, Teacher Soft가 선택적으로 활성화 |

---

## 6. Implementation blueprint

### 6.1 신규 파일 구성 (권장 경로, 실제 생성 전)

```text
panda_stage2_condq/
  __init__.py
  spec.py                  # immutable Stage2 loss+model+source contract
  calibration.py           # tau_e, global g-quantiles, q ECDF, hash
  losses.py                # compute_loss + diagnostics
  method.py                # Frozen A/T + fresh Student forward, inference_model
  training.py              # Student-only optimizer/accumulation/save/load
  tests/
    test_losses.py
    test_calibration.py
    test_gradients.py
    test_pipeline.py
    test_inference.py
  README.md                 # config and run instructions
```

- 위 경로는 **권장안**. 이름보다 중요한 것은 기존 `donor30/` 파일을 **덮어쓰지 않는 것**이다.
- Old `donor30`의 `mode='e'/'eq'`는 폐기한 **상호배타적 E**라는 뜻. 새로운 구현에서 동일 이름을 무심코 재사용하지 않는다. 신규 variant 이름 `gt_only`, `panda_e`, `panda_condq`로 명시한다.
- `method_version`, `loss_schema`, `calibration_schema`, source/donor/model spec가 바뀌면 old checkpoint의 strict load가 실패해야 한다.

### 6.2 실행 흐름: calibration

```text
Pinned RR TRAIN inventory (unaugmented)
    -> Frozen A predicts u
    -> Warp original P once -> corrected PAN
    -> Frozen Teacher HRMS prediction
         -> e_T for ALL pixels -> exact tau_e median
    -> Scharr(corrected PAN), Scharr(GT RGB)
         -> q/g at ROI8 -> global g quantile bins
         -> each bin q sorted -> fixed conditional q ECDF
    -> write + checksum calibration artifact
```

학습에 사용 중인 live Stage 1 모듈이나 checkpoint를 직접 수정하지 않는다.

### 6.3 실행 흐름: each RR training forward

```text
RR train batch: P [B,1,H,W], M [B,C,H/4,W/4], L_A [B,C,H,W], GT [B,C,H,W]
      |
      +--> frozen A(P,L_A) -> u [B,2] -> warp P -> P_tilde
      |                     |           |
      |                     |           +--> boundary support guard (NO loss masking)
      |                     |
      |                     +--> frozen Teacher(P_tilde,M,L_A) -> Y_T
      |
      +--------------------------------> trainable Student(P_tilde,M,L_A) -> Y_S
                                                       |
                         |-----------------------------+----------------|
                         |                                              |
                     e_T, e_S                                  Delta_TS
                         |                                              |
            d_e, advantage a_T (detached)                   soft target fitting
                         |
       Scharr(P_tilde), Scharr(GT RGB) -> q,g ROI8 -> fixed ECDF -> r_q_bar
                         |
              w_H = 1 + alpha*d_e + gamma*d_e*r_q_bar
                         |
        HARD = mean_all(w_H * e_S)       SOFT = mean_all((1-d_e)*a_T*Delta_TS)
                         |                            |
                         +------------ total = HARD + beta*SOFT
                                                   |
                                        Student optimizer only
```

Inference path는 위 과정에서 **Teacher 및 GT 이후의 모든 경로가 없어지고** `A -> warp -> Student`만 남는다.

### 6.4 PyTorch loss 계산 참조 pseudocode

아래는 타입/gradient/reduction을 보여주는 **구현 참조용 pseudocode**다. `stats.lookup_exact()` 및 `stats.tau_e`는 §4 artifact를 읽는 신규 코드의 인터페이스이며, 기존 저장소에 이미 존재하는 함수라는 뜻이 아니다.

```python
import torch


def compute_s2_loss(student_y, teacher_y, gt, aligned_pan,
                    *, scharr, stats, alpha, beta, gamma, delta,
                    roi_margin=8):
    """FP32 [B,C,H,W] images; output loss scalars and detached diagnostics."""
    assert student_y.shape == teacher_y.shape == gt.shape
    B, C, H, W = gt.shape
    assert aligned_pan.shape == (B, 1, H, W)
    assert H > 2*roi_margin and W > 2*roi_margin
    assert min(alpha, beta, gamma) >= 0 and delta > 0

    y = gt.detach().float()
    t = teacher_y.detach().float()
    s = student_y.float()  # must preserve Student graph

    e_s = (s - y).abs().mean(dim=1, keepdim=True)  # [B,1,H,W], HAS grad
    with torch.no_grad():
        e_t = (t - y).abs().mean(dim=1, keepdim=True)
        tau = torch.as_tensor(stats.tau_e, dtype=e_t.dtype, device=e_t.device)
        d_e = e_t / (e_t + tau).clamp_min(torch.finfo(e_t.dtype).tiny)
        advantage = (e_s.detach() - e_t).clamp_min(0) / (e_s.detach() + delta)

        gt_rgb = scharr.rgb_luminance(y)
        pan_edges = scharr.descriptor(aligned_pan.detach().float())  # [B,2,H,W]
        gt_edges = scharr.descriptor(gt_rgb)                         # [B,2,H,W]
        q = (pan_edges - gt_edges).abs().mean(dim=1, keepdim=True)
        g = gt_edges.mean(dim=1, keepdim=True)

        sl = (slice(None), slice(None),
              slice(roi_margin, H-roi_margin), slice(roi_margin, W-roi_margin))
        r_q_bar = torch.zeros_like(e_t)
        r_q_bar[sl] = stats.lookup_exact(g[sl], q[sl])
        # lookup_exact: train-global g bin -> exact q ECDF P(Q <= q | bin)
        w_h = 1 + alpha * d_e + gamma * d_e * r_q_bar

    delta_ts = (s - t).abs().mean(dim=1, keepdim=True)   # HAS grad
    hard = (w_h * e_s).mean()                            # full B,H,W
    soft = ((1 - d_e) * advantage * delta_ts).mean()    # full B,H,W
    loss = hard + beta * soft

    return {
        'loss': loss, 'hard_weighted': hard, 'soft_unscaled': soft,
        'soft_weighted': beta * soft,
        'gt_raw': e_s.mean(),
        'e_extra_raw': (d_e * e_s).mean(),
        'q_extra_raw': (d_e * r_q_bar * e_s).mean(),
        'w_h_mean': w_h.detach().mean(),
        'soft_active_fraction': (advantage > 0).float().mean(),
        'r_q_roi_mean': r_q_bar[sl].mean(),
    }
```

**참조 코드에 붙는 필수 guard** (forward wrapper/`stats` 클래스에서 구현):

- 배치의 `source split='train'`, `resolution='RR'`, `sensor`, `native_lms` provenance, finite FP32 및 shape 검증.
- `stats.lookup_exact(g,q)`는 **동일 shape**의 detached `[B,1,h_roi,w_roi]` 값을 반환해야 하며 `0<=r<=1`; `stats`에는 frozen donor/A/T/data hash 검증 기능이 있어야 한다.
- Teacher/Aligner를 **forward의 `no_grad()` 영역**에 넣고 `student()`는 영역 **밖**에서 호출.
- Scharr valid ROI/stencil guard를 **loss 계산 전** 수행; 실패한 batch는 optimizer update 없음.
- `loss` 및 각 weight/gradient가 finite인지 검사. 실제 `alpha`, `beta`, `gamma`, `delta`는 생성자에서 float/finite/range 확인.
- `grad advantage`를 통해 Student가 우회적으로 `a_T`를 최적화하는 경로가 없어야 한다.

### 6.5 Variant별 불필요한 분기 실행 금지

위 `compute_s2_loss()`는 모든 항의 gradient 의미를 이해하기 위한 **공통 reference pseudocode**이므로 `teacher_y` 및 `stats`가 이미 존재한다고 가정한다. 실제 production training에서는 variant마다 필요 신호만 계산한다.

| variant | Teacher forward | `tau_e` | q/g / Conditional CDF | 손실 |
|---|---|---|---|---|
| `gt_only` | **불필요** | **불필요** | **불필요** | `mean(abs(student_y-GT))` |
| `panda_e` | 필요 | 필요 | **불필요** | `L_H(gamma=0) + beta*L_K` |
| `panda_condq` | 필요 | 필요 | **필요** | `L_H(gamma>0) + beta*L_K` |

- **공통** frozen Aligner는 모든 variant에서 반드시 사용한다. GT-only에서도 PAN alignment를 생략하면 공정한 비교가 아니다.
- `gt_only`는 Teacher와 calibration의 생성/실행을 건너뛰되 **동일 donor의 Aligner**와 동일 Student 구조/초기값을 사용한다. 단 Teacher는 소스 donor의 재현 가능성 metadata에 계속 기록한다.
- `panda_e`는 `tau_e`만 있으면 학습 가능하다. `panda_condq`를 위해 만든 통합 calibration artifact를 같이 로드해도 되지만 q 계산은 건너뛴다.
- 이 최적화는 수학적 손실을 바꾸지 않는다. q를 계산하지 않는다고 support guard 정책을 몰래 변경해서는 안 되며, 공통 실험에서는 q ROI support guard를 세 variant에 동일하게 적용하는 것이 비교에 유리하다.

### 6.6 학습, gradient accumulation, 저장

- 기존 donor30 Student trainer의 optimizer 검증, finite gradient, checkpoint strict reload 개념을 재사용하되 **새 스키마**로 구현.
- 비교 run 간 동일 `Student` initialization seed, 동일 data order/augment, 동일 optimizer, 동일 donor checkpoint 및 calibration artifact 사용.
- 기존 donor30 캠페인의 학습 조건은 LR=1e-4/AdamW(weight decay .01), warmup100+cosine50K, effective batch48=12×4, FP32 등. 새 실험에서 이를 이어갈 경우 **공통 고정값으로 기록**하고, 변경하면 별도 축으로 취급. 기존 W104 run의 결과와 직접적인 단일요인 인과 비교라고 서술하지 않는다.
- 동일 크기 샘플에서는 microbatch 출력 scalar에 `microbatch_B / total_B`를 곱해 gradient accumulation. 예: 12개 microbatch, 각각 B=4, total B=48이면 `1/12`씩 loss backward.
- optimizer step 후 A/T parameter와 buffer가 변하지 않았는지 hash/version 검사. Student만 업데이트.
- 상태 저장: model state, optimizer, scheduler, RNG, update, Student init seed, donor A/T hash, loss params, calibration artifact SHA, sensor/data manifest, method version, eval protocol, loss reduction, mask policy(`full_image_no_warp_valid_mask`), checkpoint selection rule.
- resume은 **정확히 같은 contract**에서만 허용. 기존 E/EQ/MASK2 checkpoint의 무조건적 호환/재시작 금지.

---

## 7. Full-image 감독과 ROI/경계 규칙 (혼동 방지)

| 종류 | 범위 | 용도 |
|---|---|---|
| **Stage 2 Hard loss** | 전 이미지 픽셀 | Full-band GT fitting, warp-valid mask **없음** |
| **Stage 2 Soft loss** | 전 이미지 픽셀 | selective Teacher prediction fitting, warp-valid mask **없음** |
| **Conditional-q** | fixed ROI8만 | q/g 계산과 `gamma*d_e*r_q_bar` 강화 |
| **Aligner CNN crop8** | Aligner의 내부 **입력 view** | z-norm/GN/ε 경계 단서 차단; 수정 금지 |
| **PAN warp support guard** | ROI8의 Scharr stencil(사방 7픽셀 support 영역) | extrapolated input 구조 값으로 q를 계산하지 못하게 fail-fast |
| **FR/RR 평가** | 기존 공식 evaluator 규칙 | Stage 2의 train mask 제거와 **별도로 유지** |

- `warp_support_mask` 계산은 **경계 적합성 검사나 진단**에 남길 수 있으나, `(w_H*e_S).mean()` / `(...Delta_TS).mean()`의 픽셀을 삭제하거나 분모를 `valid_count`로 바꾸면 명세 위반이다.
- warped PAN 경계는 `border` extension으로 채워지지만 해당 픽셀의 GT 자체는 존재한다. Full-image loss가 반드시 우월하다는 뜻이 아니므로 **masked/unmasked 변경은 독립 ablation**으로 평가할 것.
- 기존 `directshift48` Stage 1의 원래 reconstruction loss도 unmasked였다. `donor30` MASK2에서 후속 마스킹이 도입된 것과 혼동하지 않는다.

---

## 8. 필수 테스트 및 수용 기준

### 8.1 단위 테스트 — calibration/수식

1. **Shape:** WV3 `[B,8,64,64]`, GF2 `[B,4,64,64]`; `e_t`, `e_s`, `Delta_TS`, `q`, `g`, `r_q_bar`, `w_h`는 각각 `[B,1,64,64]`.
2. **GT-only equivalence:** `alpha=beta=gamma=0`에서 `compute_s2_loss().loss == (student_y-gt).abs().mean()` 허용 FP 오차 내 동치.
3. **No-mask:** warp-valid mask가 빈 영역을 가지더라도 Hard/Soft가 전체 픽셀에서 계산되는지 확인. 단, q용 ROI support guard에 실패하면 업데이트 자체가 차단되어야 한다.
4. **ROI:** RR64에서 정확히 중앙 48×48에만 `r_q_bar`가 비영(0이 아닌 rank) 가능; 외부 zero. `gamma=0`이면 ROI 여부가 Hard에 영향 없음.
5. **PANDA-e reduction:** `gamma=0` 결과가 `mean((1+alpha*d_e)*e_s) + beta*mean((1-d_e)*a_T*Delta_TS)`와 동일.
6. **Teacher gate:** `e_S<=e_T`인 곳에서 `a_T=0`; 반대로 Teacher가 우수한 곳에서만 양수.
7. **Stopgrad:** `d_e`, `a_T`, `r_q_bar`, `w_H`는 Student로 연결된 backward graph가 없어야 함. `e_S`, `Delta_TS`는 Student로 gradient가 존재해야 함.
8. **Bounds:** `w_H>=1`, `0<=r_q<=1`, `0<=a_T<=1`, `0<=d_e<=1`, 모든 값 finite. zero error/median 및 ties 테스트.
9. **Global calibration:** 동일 test pixel의 `(g,q)`라면 다른 batch/patch 구성에서도 **동일 r_q** 반환.
10. **ECDF right tie:** 한 bin의 q=`[0.1,0.1,0.3]`에서 `q=0.1 -> 2/3`, `q=0.2 -> 2/3`, `q=0.3 -> 1`, `q=0.0 -> 0`.
11. **g cutpoint tie:** 경계와 같은 `g`를 상위 그룹에 배정하는 규칙이 calibration/inference lookup에 일치.
12. **No leakage:** validation/test/FR 표본, 중복/누락 sample ID, 다른 센서/A/T SHA와 calibration pairing 시 예외 발생.
13. **Band consistency:** `q`는 RGB 3밴드에서만 계산, Hard는 모든 C개 밴드를 포함. NIR를 변경해도 `q`는 변하지 않을 수 있지만 `e_S`는 변함.
14. **Microbatch accumulation:** batch48 한 번의 backward와 batch4×12 accumulation의 Student gradient가 허용 오차 내 일치(동일 RNG, dropout0 등의 조건).

### 8.2 파이프라인/체크포인트 테스트

1. Same-spec Teacher/Student의 구조는 W112/D123/P0로 일치하되 **파라미터 객체는 공유하지 않음**; Teacher와 Student initial state hash 다름.
2. Paired variant들(`gt_only/panda_e/panda_condq`)의 **fresh Student 초기 state hash는 같은 seed 기준으로 동일**.
3. `train()` 호출 후에도 Aligner/Teacher는 `eval()` 상태; 두 모듈에 gradient 없음; optimizer에는 Student만.
4. 한 샘플에 대해 Teacher/Student에 들어가는 `pan_aligned`, `ms`, `native_lms`가 동일함. PAN warp는 한 번이고 `u*4` 없음.
5. **Inference**는 Teacher/GT/q/g/CDF/tau_e 없이 `A+S`로 FR512와 RR256를 정상 실행하고 출력 `[B,C,H,W]` 유지.
6. 새 `method_version`/calibration SHA 불일치로 구형 MASK2/EQ checkpoint strict reload가 거부됨.
7. Calibration 중간 실패 시 **부분/가짜 완료** artifact를 저장하지 않음; save/verify를 원자적으로 처리.
8. Frozen donor hash는 calibration 전후/Student train 전후 동일함.
9. FR HQNR은 기존 evaluator의 original PAN 기준 `D_s`, `D_lambda`; RR ERGAS는 기존 RR eval의 공식 영역 규약을 변경하지 않음.

### 8.3 로깅 필수 항목

| 분류 | 필드 예시 | 기록 이유 |
|---|---|---|
| Total/parts | `loss`, `hard_weighted`, `soft_unscaled`, `soft_weighted` | Hard/Soft 기여 분리 |
| Hard breakdown | `gt_raw`, `e_extra_raw`, `q_extra_raw` | alpha/gamma 변화 해석 |
| Weights | `d_e_mean`, `a_T_mean`, `soft_active_fraction`, `w_h_mean/max/p95` | 실제 강조 범위 확인 |
| q/g | `q_roi_mean`, `g_roi_mean`, `r_q_roi_mean`, `r_q_roi_quantiles`, `bin_counts` | 조건부 rank 사용 확인 |
| Gradient | Hard vs Soft Student parameter gradient norm/cosine(주기적 진단) | 옛 E 혼합의 충돌 완화 여부 |
| Geometry | `u` 분포, `warp_support_fraction`, `spatial_support_ok` | alignment/경계 진단 (**loss mask 아님**) |
| Evaluation | RR ERGAS, SCC, SAM, PSNR; FR HQNR, D_s, D_lambda | 단일 checkpoint의 RR–FR trade-off 추적 |
| Provenance | donor hash/step, Student seed, calibration SHA, dataset/sensor, loss variant/alpha/beta/gamma/delta | 재현·감사·artifact 대응 |

`q_roi_mean`과 `w_h_mean`은 적용 영역이 다르므로 동일 종류의 통계처럼 비교하지 않는다. q/g 및 `r_q`는 ROI8에서, Hard/Soft는 전체 이미지에서 계산한다.

---

## 9. 첫 번째 실험 및 중단 규칙

### 9.1 우선순위가 높은 비교

| Variant | alpha | gamma | beta | 목적 |
|---|---:|---:|---:|---|
| `gt_only` | 0 | 0 | 0 | Same-capacity fresh Student의 전 밴드 GT L1 기준 |
| `panda_e` | 양수 | 0 | 양수 | GT-always Hard + advantage Soft, Teacher prior의 효과 |
| `panda_condq` | `panda_e`와 동일 | 양수 | `panda_e`와 동일 | **조건부 q-Hard 추가 효과만** 분리 |
| `panda_rawq` (추가 선택) | 동일 | 양수 | 동일 | raw q 정규화 vs conditional q 비교(정규화 규약 별도 선언) |

**비교의 필수 공통조건:** 동일 센서/RR train/test, 동일 frozen A/T checkpoint 및 eval spec, 동일 Student architecture W112-D123/P0, paired Student seed, optimizer/learning-rate schedule, batch/data/augmentation, Full-image mask policy, checkpoint 평가 위치.

- 첫 구현 smoke test에서는 `alpha=1`, `beta=0.1`을 **과거 PANDA 원고 기준의 출발 후보**로 사용할 수 있으나 새 구조에서 최적이라고 주장하지 않는다. `gamma`는 **필수 명시 파라미터**이며 성능 기준 없이 자동으로 `alpha`에 묶지 않는다.
- 비교를 위해 `gamma=0`은 필수 대조. `gamma>0`의 실제 grid 또는 값은 실험 계획 승인 시 확정한다. `K=5` 역시 첫 후보이며 최종 하이퍼파라미터가 아니다.
- Raw-q를 비교하려면 `raw q -> [0,1]` 변환 방법(예: 같은 train-only reference scale)도 별도 고정해야 하므로, **미정인 상태에서 임의 코드로 삽입하지 않는다**.
- 먼저 무학습 synthetic unit tests → train-only calibration 완성/해시 고정 → 1~2 batch forward/backward smoke → 짧은 진단 run → 정식 학습의 순서로 진행한다. **짧은 진단 run은 공식 50K 결과가 아니다**.

### 9.2 Official 평가 주의

- RR 지표 개선이 FR HQNR 개선을 보장하지 않는다. 각 checkpoint마다 **같은 Student state**로 RR20/FR20 동시 평가.
- FR `HQNR=(1-D_s)(1-D_lambda)`를 분해 보고. 원본 PAN과 corrected PAN 사이의 좌표계 차이가 `D_s`에 영향을 줄 수 있다.
- 선택 checkpoint가 공식 test HQNR을 반복 확인해 선택된 경우 **test-aware selection bias**를 기록한다. 동일 결과를 독립 holdout 검증으로 주장하지 않는다.
- 기존 masked/W104/EQ 자료는 **맥락과 동기**로만 사용한다. 이번 새 Stage 2의 alpha/gamma 조정 효과라는 결과로 소급 설명하지 않는다.

---

## 10. 인수인계 / Implementer checklist

### 10.1 구현 전에 결정해야 할 입력

- [ ] 채택할 donor checkpoint (`sensor`, run ID, step, A/T state hashes). MASK2 donor 사용 여부와 무관하게 Stage 2 신규 calibration 수행.
- [ ] Student 초기화 seed(모든 variant paired), optimizer 공통조건.
- [ ] `alpha`, `beta`, `gamma` 값 또는 승인된 탐색 범위. GT/panda_e/condq의 축별 동등성 확인.
- [ ] `delta>0` 값 및 FP32 numerical policy.
- [ ] Conditional-q bin 개수 `K`(초기 추천 5) 및 exact ECDF 저장 메모리 예산/형식.
- [ ] 새 namespace/실험 run ID, 기존 registry/outbox와의 충돌 회피 방안.
- [ ] 공식 evaluator 및 checkpoint selection rule (변경 없이 사용).

### 10.2 구현 완료의 정의

- [ ] Student는 W112-D123/P0 fresh이고, optimizer/frozen check 정상.
- [ ] 새로운 train-only full-image `tau_e` + conditional-q ECDF calibration artifact가 donor/data/operator hash에 묶여 있음.
- [ ] GT-only와 `panda_e`, `panda_condq`가 공통 인터페이스로 실행되며 위 test 모두 통과.
- [ ] Hard/Soft의 GT 및 Teacher 신호가 올바르게 detach되어 **Student에만 gradient**가 존재.
- [ ] ROI8 이외 Hard/Soft가 삭제되지 않음; Stage 1 Aligner crop8과 PAN warp 규칙 그대로.
- [ ] 기존 `donor30` 체크포인트와 저장소의 기존 캠페인/실험 기록 손상 없음.
- [ ] Inference는 A+S만 사용하며 기존 공식 RR/FR inference 결과 인터페이스 유지.
- [ ] 단위 테스트/smoke 및 calibration hash/gradient 로그를 저장하여, **모델 학습 실행과 코드 검증을 구분**.

---

## 11. 근거 및 읽은 코드/자료

1. 사용자 제공: `PANDA_Stage2_Loss_Modeling_Session_Handoff_2026-10-16.md` — 모델 제약, E/EQ 실패 관측, q/g 정의, mask 정책, 조건부 ECDF 및 미결정 항목. **이 문서의 주요 연구 근거.**
2. 이번 세션에서 정리한 PANDA Stage 2의 최종 후보 수식: `w_H=1+alpha*d_e+gamma*d_e*r_q_bar`; `L_K=mean((1-d_e)*a_T*Delta_TS)`; `L_S2=L_H+beta*L_K`.
3. GitHub 코드 확인(`main` HEAD `828a1aa4349394365d1eef638579570d4613dc40`):
   - [`directshift48/method.py`](https://github.com/hojunking/PAN-Crafter-repro/blob/main/directshift48/method.py) — DirectPANAligner, ScharrAbsolute, ROI8/guard, native prediction.
   - [`panda_method/reconstruction.py`](https://github.com/hojunking/PAN-Crafter-repro/blob/main/panda_method/reconstruction.py) — Reconstructor spec/forward, fresh Student builder.
   - [`pa/warp.py`](https://github.com/hojunking/PAN-Crafter-repro/blob/main/pa/warp.py) — bicubic PAN warp/support.
   - [`donor30/method.py`](https://github.com/hojunking/PAN-Crafter-repro/blob/main/donor30/method.py) — **구형** E/EQ loss, masked `tau_e/tau_q` calibration, donor-bound Stage 2.
   - [`donor30/masking.py`](https://github.com/hojunking/PAN-Crafter-repro/blob/main/donor30/masking.py) — 구형 mask 규약.
   - [`donor30/training.py`](https://github.com/hojunking/PAN-Crafter-repro/blob/main/donor30/training.py) — Student-only trainer/accumulation/strict checkpoint validation.
   - [`donor30/EXPERIMENT_PLAN.txt`](https://github.com/hojunking/PAN-Crafter-repro/blob/main/donor30/EXPERIMENT_PLAN.txt) — 과거 MASK2 학습·평가 일정. **새 Stage 2 방법을 이미 구현했다는 근거가 아님**.

**최종 주의:** 여기 적힌 exact ECDF 직렬화, `K=5`, 독립 `gamma`, delta 안정화, 신규 namespace는 **기술적으로 검증 가능한 구현 권장안**이다. 기존 실험에서 성공이 입증된 모델 설계 또는 코드 반영 완료 사항으로 표현하지 않는다.
