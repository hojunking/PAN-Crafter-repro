# PAN 추가 변위와 offset consistency: 10 GPU-hour 실험 계획

**작성일:** 2026-09-10  
**대상:** WV3 · s1 · seed 2025를 우선하는 단일 대응 block  
**고정 모델:** W96–D124 U-Net · PAN+MS 9채널 · HRMS 단일 복원  
**예산:** 추가 총 10 GPU-hours. 한 GPU에서 순차 실행하는 예산이며, 세 서버 각각 10시간을 뜻하지 않는다.  
**상태:** 실행을 위한 새 설계. 아래 실험의 실제 데이터 학습·성능·GPU 소요 시간은 아직 측정하지 않았다.  
**상위 문서:** `PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md`  
**직접 근거:** `2026-09-10_pa-a1-a3-s1-results.md`와 이후 대화에서 합의한 상대 변위 감독.

> **이번 실험의 질문**  
> 원래 PAN–MS의 정합 오차가 거의 일정하다면 상수 보정은 자연스러울 수 있다. 그러나 PAN을 추가로 이동했을 때에는 보정량도 그만큼 바뀌어야 한다.  
> **PAN에 알려진 무작위 변위를 추가하고, 보정량의 변화가 그 변위를 상쇄하도록 학습한다.**  
> 먼저 corruption만 사용하는 대조군과 직접 offset supervision을 비교하고, 여유가 있으면 stop-gradient 유무까지 분리한다. 정합 CNN의 아키텍처는 바꾸지 않는다.

---

## 0. 먼저 읽을 실행 요약

| Case | 입력 변화 | Loss | 비교 목적 | 우선순위 |
|---|---|---|---|---|
| **N1 — CORRUPT-REC** | Native/corrupted PAN을 1:1 step으로 사용 | HRMS reconstruction | 입력 변동을 보여 주는 것만으로 충분한가? | 필수 |
| **N2 — OFFSET-SG** | N1과 동일 | Recon + 상대 offset loss, clean 예측 stop-gradient | 정합량에 직접 감독을 주면 반응이 학습되는가? | **필수·주 후보** |
| **N3 — OFFSET-NOSG** | N1과 동일 | Recon + 같은 상대 offset loss, 양쪽 역전파 | stop-gradient라는 학습 방향 선택의 영향 | 시간 적격 시 |

세 run 모두 같은 초기 U-Net·같은 초기 aligner·같은 data/augmentation 순서·같은 추가 변위를 사용한다. **독립적으로 처음부터 학습**하며 N1 가중치를 N2에 이어 학습하지 않는다.

- 같은 step에서 큰 U-Net은 **한 번만** 실행한다.
- Corrupted step의 정합 CNN은 두 번 실행한다. N1/N2의 clean CNN 호출은 `no_grad`, N3는 gradient를 유지한다.
- Native step에서는 clean aligner에도 recon gradient를 준다. **모델 전체나 clean 경로를 영구 freeze하지 않는다.**
- 추가 변위는 PAN에만 적용한다. MS condition·MS residual base·GT·최종 출력 좌표는 유지한다.
- 입력 edge matching으로 sample별 정합 GT를 만들지 않는다. Offset loss의 정확한 label 성분은 **우리가 추가한 ε**이다.
- 원본/유효영역/정합 반영 평가와 `best_raw`, `best_aligned`, `last`는 v2를 유지한다.
- 이번 10시간에는 B-2, PAN reconstruction, LPAN/HPAN 입력, KD, EMA teacher, A2 edge/A3 geometry loss, architecture sweep을 넣지 않는다.

### 기존 명세에서 바뀌는 항목

| 항목 | A1–A3 v2 | 이번 문서 |
|---|---|---|
| 입력 corruption | 없음 | PAN에만 training-time global translation |
| 정합 감독 | Recon / edge / geometry | Recon 및 **보정량 변화**의 L1 |
| Aligner 아키텍처 | 0.1053M dual-stem CNN | **동일** |
| Aligner가 보는 공간 범위 | 전체 입력 | **공통의 고정 내부 crop** — padding 단서 배제, §5 |
| 복원 U-Net 입력·출력 크기 | 원래 크기 | **동일**; aligner crop을 U-Net에 적용하지 않음 |
| 학습 PAN의 sampling | Native에서 한 번 | Corrupted step에서 두 번, §6 |
| L_rec 영역 | 기존 B0 full domain | **동일**; moving mask 없음 |
| 평가 프로토콜 | v2 세 view | **동일** |

내부 crop은 이번 계획에서 추가한 안전 설계이다. 이 때문에 **과거 A1을 새 N1의 완전한 one-factor 대조라고 부르지 않는다.** 핵심 인과 비교는 같은 crop을 사용하는 **N2−N1, N3−N2**이다. Corruption 자체의 기여를 과거 A1과 엄밀히 분리하려면 같은 crop의 native-only 재학습이 추가로 필요하며, 이번 10시간에는 넣지 않는다.

---

## 1. 자료에서 확인되는 근거와 이번 가설

### 1.1 확인된 결과 [S1]

s1·seed 2025 결과 보고서는 다음을 기록한다.

| 항목 | 보고된 내용 |
|---|---|
| B0 / A1 / A2 / A3 HQNR raw-original | 0.9516 / 0.9492 / 0.9529 / 0.9531 |
| Aligner의 FR 변위 | 대략 `(dy,dx)=(+0.18~+0.23, −0.04~−0.07)` HR px |
| 추가 PAN shift에 대한 반응 기울기 | −0.02~−0.09; 이상적인 동일 축 반응은 −1 |
| Aligner의 gradient | Recon에서도 유입됨. A3 geometry gradient도 상당했으나 반응은 유사 |
| 합성 module 검사 | 같은 CNN에서 corr 0.997, 오차 0.08 px로 보고 |
| 범위 | s1 한 block. 다른 서버/seed의 일반화 결론은 아님 |

보고서의 “학습 분포에 변동이 없어 상수가 되었다”는 설명은 유력한 **원인 해석**이다. 모든 영상의 물리적 정합량이 같다는 검증으로 확대하지 않는다. 또한 train signed shift 평균이 0에 가깝다는 결과는 flip/rotation의 좌표를 되돌린 값인지 확인해야 한다.

### 1.2 이번에 확인할 가설

- **H1:** 알려진 PAN corruption을 추가하면 상수 correction만으로는 복원이 어려워져 입력 반응이 생길 수 있다.
- **H2:** Recon gradient가 정합에 민감하지 않아도 offset loss가 aligner를 직접 감독할 수 있다.
- **H3:** Clean 예측을 이번 target으로 고정하는 방식과 양쪽을 함께 수정하는 방식의 결과가 다를 수 있다.

**성공의 범위를 분리한다.** 알려진 추가 변위 복구가 좋아진 것, 원래 입력의 복원 품질이 좋아진 것, 원래 절대 정합량이 정확해진 것은 서로 다른 주장이다.

---

## 2. 공통 모델과 좌표

### 2.1 기호

| 기호 | 뜻 | WV3 train 예시 |
|---|---|---|
| P | 현재 pair의 원 PAN | `[B,1,64,64]` |
| S | 원 LRMS | `[B,8,16,16]` |
| M=U₄(S) | B0와 같은 bicubic MS base | `[B,8,64,64]` |
| Y | HRMS GT | `[B,8,64,64]` |
| ε | 추가 PAN sampling shift | `[B,2]`, `(dy,dx)`, HR pixel |
| ĉ₀ | 추가 변형 없는 pair의 예측 correction | `[B,2]` |
| ĉε | 추가 변형된 pair의 예측 correction | `[B,2]` |
| c* | 진짜 필요한 전체 correction | 미지의 값; 학습 label로 제공하지 않음 |
| Fθ | MS residual을 출력하는 W96–D124 U-Net | 9ch → 8ch |
| Aφ | 기존 global aligner | PAN/MS → 2개 correction |

`0`은 “원래 오차가 0”이 아니라 “추가 corruption이 0”이라는 뜻이다. `c`는 영상 residual R이나 confidence가 아니라 **PAN을 읽는 위치의 보정 벡터**이다.

### 2.2 고정할 아키텍처 [S2 §3]

```text
PAN stem: Conv3×3 1→16, GN4, SiLU
MS stem : Conv3×3 8→16, GN4, SiLU
concat  : 32ch
joint   : Conv3×3 32→32 stride2, GN8, SiLU
          Conv3×3 32→64 stride2, GN8, SiLU
          ResBlock 64ch ×1
head    : GAP → Linear64→32 → SiLU → Linear32→2
```

Convolution은 bias=False, GN affine=True, Linear bias=True. Spatial padding은 기존 reflect를 유지한다. BatchNorm/dropout/attention/gate/dense flow를 새로 넣지 않는다. 마지막 `32→2` weight/bias만 0 초기화한다. 나머지 layer를 모두 0으로 초기화하지 않는다.

Aligner용 sample·band별 z-score 정규화는 **§5 내부 crop 후** 수행한다. Warp와 U-Net에는 B0 intensity scale의 원래 영상값을 사용한다. GT는 두 네트워크의 forward에 넣지 않는다.

### 2.3 Warp 부호

\[
\mathcal W(P,e)[y,x]=P[y+e_y,x+e_x].
\]

양의 dy는 아래쪽 원본 위치를 읽으므로 영상 내용은 위로 보인다. Aligner 출력 순서는 `(dy,dx)`지만 `grid_sample` grid의 마지막 순서는 `(x,y)`이다.

\[
g_x=2\frac{x+0.5+d_x}{W}-1,\qquad
g_y=2\frac{y+0.5+d_y}{H}-1.
\]

**단위는 현재 입력 PAN 격자**이다. LR 픽셀 단위 통계를 HR로 바꿀 때만 한 번 ×4한다. 64² 입력을 512²로 바꾸었다고 예측 displacement에 ×8을 하지 않는다.

---

## 3. Corruption 규모: 관측 통계에서 정하고 0 중심으로 뽑는다

### 3.1 이번 10시간의 명시적인 규모 근거

기존 audit 부록 E는 WV3 **train**의 해상도 맞춤 HR 격자 분석에서 global 변위 크기의 P90을 **0.250 LR px**, 중앙값을 0.072 LR px로 기록한다. [S3 부록 E]

이번 실행안은 이 **train aggregate P90을 augmentation 규모의 임시 proxy**로 사용한다.

\[
R_d=r\,s_{d,90}^{LR},
\qquad R_{WV3}=4\times0.250=1.000\;\text{HR px}.
\]

`s_{d,90}^{LR}`는 audit 표에 기록된 전역 변위 크기 P90이다. 원문 표만으로 개별 벡터 분포를 복원하지 않는다. 실제 audit 코드의 norm 정의와 적용 단위도 manifest에 확인·기록한다.

**이 결정의 지위:**

1. `R=1.0`은 실제 misalignment의 최대값이라는 주장이 아니다. 과거 train 분석에서 기록된 규모를 적용하는 **본 계획의 새 sampling 규칙**이다.
2. 작은 train patch의 추정에는 노이즈·spectral bias가 있다는 경고를 유지한다. [S3 §II.8.5, S4 §1.2] 따라서 `calibration_quality=train_aggregate_proxy`로 표시하고, 정확한 센서 noise 분포나 검증 완료된 데이터셋 적응형 모델이라고 부르지 않는다.
3. **단위·통계량을 확인할 수 없으면 시작하지 않는다.** 부록의 LR px 표기를 다시 확인하고 manifest에 근거를 적는다. P90 0.250을 현재 PAN 격자의 0.250으로 잘못 읽지 않는다.
4. FR test의 1.79 px, 장면별 audit 벡터, 현재 CNN의 +0.2 예측 분포를 학습 radius나 target으로 사용하지 않는다.
5. 원문에는 dy·dx 각각의 신뢰할 만한 P90 값이 없다. 이를 임의로 만들어 넣지 않는다.
6. 신뢰도 높은 학습 원본의 새 calibration을 이미 보유했다면 첫 run 전에 source를 바꾸고 **전 case에 같은 새 값**을 사용한다. run 중 radius를 수정하거나 성능을 보고 고르지 않는다.

P90 선택과 아래 disk 분포도 설계 선택이다. “하이퍼파라미터가 모두 없다”가 아니라 **수동 radius sweep 없이 관측된 규모를 사전 규칙으로 사용한다**는 범위로 설명한다. 실제 train 데이터의 작은 변동으로 FR의 더 큰 절대 오차까지 다룰 수 있는지는 별도 일반화 문제다.

### 3.2 왜 이번에는 원판 안에서 sampling하는가?

이전 대화의 축별 uniform은 후보였고, 사용자 합의는 **관측 규모 사용 + 0 중심 양·음 방향**이었다. 이번 문서는 원문에 제공된 통계가 벡터의 크기라는 점에 맞춰, **2차원 원판의 면적에 대해 uniform**으로 구체화한다.

\[
u_1,u_2\sim U(0,1),\quad
\rho=R_d\sqrt{u_1},\quad \theta=2\pi u_2,
\]
\[
\boxed{\epsilon=(\rho\sin\theta,\rho\cos\theta)}.
\]

- 모든 방향이 가능하며 평균은 `(0,0)`이다.
- **전체 이동 크기**가 \(\|\epsilon\|_2\le R_d\)이다. 축별 ±R 정사각형과 달리 대각선에서 √2R로 늘어나지 않는다.
- `radius = R × rand()`가 아니라 `R × sqrt(rand())`를 사용한다. 전자는 원점 부근에 면적당 표본이 더 몰리는 다른 분포다.
- 이 분포에서 축별 표준편차는 R/2이다. R=1이면 각 축 표준편차 약 0.5 HR px이다. 이는 분포에서 유도한 검사값이다.
- R=1 HR px는 현재 LRMS 기준 총 변위 0.25 LR px에 해당한다. 원래 J4의 **축별 ±0.5**와 분포·강도가 동일하지 않다.

### 3.3 적용 규칙

기존 B0의 공통 flip/rotation, intensity scaling, bicubic 처리가 끝난 뒤 **PAN만** 움직인다. 한 sample당 ε 하나를 새로 뽑으며, 같은 영상의 모든 위치에 같은 벡터를 적용한다. Band별·위치별 noise, rotation, scale, affine은 없다.

각 corrupted update에서 batch의 각 sample마다 새로운 ε를 뽑는다. 같은 sample이 다음에 등장해도 다시 뽑는다. 이전 corrupted PAN을 누적 이동하지 않고 매번 원 PAN P에서 만든다. ε는 영상 생성과 loss 계산에만 사용하며 **CNN 입력에 concatenate하지 않는다.**

---

## 4. Native/Corrupted 학습과 loss

### 4.1 Native step을 유지한다

Optimizer update index가 0부터 시작할 때 다음을 고정한다.

```text
0, 2, 4, ... : native step — 추가 변위 없음
1, 3, 5, ... : corrupted step — sample별 random PAN 변위
```

즉, native:corrupted=1:1이다. 이 비율은 본 계획의 새 고정값이며 최적이라고 검증된 값은 아니다. 같은 batch를 두 번 학습하는 것이 아니라 **다음 batch마다 경로를 교대**한다. 총 update와 sample 노출량은 A1과 같게 한다.

Native step:

\[
\hat c_0=A_\phi^V(P,M),\quad
\widetilde P_0=\mathcal W(P,\hat c_0),\quad
\hat Y_0=M+F_\theta([\widetilde P_0,M]),
\]
\[
L=L_{rec}=\operatorname{mean}|\hat Y_0-Y|.
\]

여기서 \(A_\phi^V\)는 §5의 고정 내부 view를 사용하는 **같은 A1 CNN**이다. Native recon에서는 ĉ₀를 detach하지 않는다. 이 경로가 원래 관측에서의 복원 품질을 계속 감독한다.

### 4.2 Corrupted step 공통 forward

\[
P_\epsilon=\mathcal W(P,\epsilon),\qquad
\hat c_0=A_\phi^V(P,M),\qquad
\hat c_\epsilon=A_\phi^V(P_\epsilon,M),
\]
\[
\widetilde P_\epsilon=\mathcal W(P_\epsilon,\hat c_\epsilon),\qquad
\boxed{\hat Y_\epsilon=M+F_\theta([\widetilde P_\epsilon,M])}.
\]

정답 영상은 계속 **이동하지 않은 HRMS GT Y**이다. 복원 U-Net은 한 번만 실행한다. 참조 ĉ₀를 얻는 작은 aligner 호출만 추가된다.

### 4.3 Consistency target의 의미

원래 필요한 진짜 correction을 c*라고 하면 이상적인 연속 translation에서 corrupted pair의 correction은 c*−ε이다. 따라서 알려진 것은 **차이 −ε**이며, c* 자체는 아니다.

\[
\boxed{\hat c_\epsilon+\epsilon\approx\hat c_0}.
\]

설명용 한 축 예시: 원래 예측 +0.20, 추가 sampling shift +0.60이면 corrupted target은 **−0.40**이다. −0.60으로 전체 correction을 감독하면 원래 correction이 0이라고 가정하게 된다.

**PAN edge와 LRMS edge에서 sample별 pseudo-GT를 추정하는 단계는 없다.** Audit 통계는 §3의 규모 근거로만 사용하며 sample별 offset label로 쓰지 않는다. ĉ₀는 같은 CNN의 현재 예측이지 정합 참값이나 별도 Teacher가 아니다.

### 4.4 N1: corruption + reconstruction

\[
\boxed{L_{N1}=\operatorname{mean}|\hat Y_\epsilon-Y|}.
\]

추가 변형으로 입력 변동만 제공한다. ĉ₀는 동일 연산·진단을 위해 no-grad로 계산하되 loss에는 사용하지 않는다. `L_offset`을 로그로 계산해도 학습 coefficient는 정확히 0이다.

### 4.5 N2: stop-gradient offset consistency

\[
t_\epsilon=\operatorname{sg}(\hat c_0)-\epsilon,
\]
\[
L_{off}^{SG}=\frac1{2B}\sum_{i=1}^B\sum_{k\in\{y,x\}}
\left|\hat c_{\epsilon,i,k}+\epsilon_{i,k}-\operatorname{sg}(\hat c_{0,i,k})\right|,
\]
\[
\boxed{L_{N2}=L_{rec}+\lambda_{off}(t)L_{off}^{SG}}.
\]

이번 계산에서 참조를 직접 수정하지 않고 corrupted 예측이 알려진 관계를 따르도록 한다. `torch.no_grad()`로 clean aligner를 실행하거나 loss target에서만 `detach()`한다. 두 방법을 혼용해 native recon까지 차단하지 않는다. [R1, R2]

### 4.6 N3: stop-gradient 제거 대조

\[
L_{off}^{sym}=\operatorname{mean}|\hat c_\epsilon+\epsilon-\hat c_0|,
\qquad
\boxed{L_{N3}=L_{rec}+\lambda_{off}(t)L_{off}^{sym}}.
\]

두 aligner 호출 모두 gradient를 유지한다. Forward의 loss 값과 coefficient는 N2와 같다. `.detach()`만 제거하고 clean 호출을 `no_grad()` 안에 남기면 N3가 되지 않는다.

- 추가 ½를 곱하거나 λ를 반으로 줄이지 않는다. 그것은 별도 gradient-matching 대조이다.
- N2/N3의 차이는 gradient 경로와 그에 따른 전체 크기·방향을 포함한다. 단순히 “target만 바뀐 효과”라고 부르지 않는다.
- No-SG가 반드시 collapse한다는 가정은 하지 않는다. 같은 상수 b를 두 입력에 내면 offset loss가 mean|ε|로 남는다.
- Shared head의 공통 additive bias는 symmetric loss의 양쪽에서 직접 미분이 상쇄될 수 있다. SG에서는 표본별 비대칭 gradient가 생길 수 있으므로 native bias drift도 기록한다. 이것은 수식에서 따라오는 차이이지 성능 우열의 예측은 아니다.

### 4.7 가중치와 절대 기준

이번 pilot의 새 설계값:

```text
lambda_off_max = 0.01
ramp_updates   = 5,000 optimizer updates
lambda_off(t)  = 0.01 × min(1, (t+1)/5000)
```

Native step과 N1에서는 coefficient를 0으로 둔다. Offset loss는 HR px 단위의 평균 L1이며 **sample별 |ε|로 나누지 않는다.** ε가 0에 가까울 때 분모가 불안정해지는 일을 피한다. λ=0.01은 검증된 최적값이 아니고, 기존 A3의 0.01과 loss 단위가 같다는 뜻도 아니다. 이번 10시간에는 λ sweep이나 run 도중 자동 gradient-ratio 조정을 하지 않는다.

Reconstruction이 PAN 위치에 둔감해도 L_off는 aligner 출력에 직접 gradient를 준다. 그러나 \(\hat c_0=b,\hat c_\epsilon=b-\epsilon\)이면 b가 틀려도 offset 관계는 맞는다. **절대 정합 anchor는 native recon에서 간접적으로만 제공되며, 그것도 물리적 참값을 보장하지 않는다.** 이번 실험에서는 edge/geometry loss를 같이 넣어 이 한계를 숨기지 않는다.

---

## 5. Padding으로 offset을 맞히는 우회로를 막는다

### 5.1 고정 estimator view — 모든 N-case에 동일

원본이 이미 64² patch이면 진짜 관측 halo가 있다고 가정하지 않는다. Shift 과정에서 생긴 border 복제 띠를 aligner가 보고 ε를 맞힐 수 있으므로 **aligner에 들어가는 clean/corrupt PAN과 M만 같은 내부 영역으로 자른다.**

\[
m_A=4\left\lceil\frac{R_d+2}{4}\right\rceil.
\]

이번 R=1에서는 **m_A=4**, train aligner view는 **56×56**이다. 2는 보수적인 bicubic 주변 support 여유이고, 4의 배수 정리는 기존 convolution stride와 크기를 맞추기 위한 새 구현 선택이다.

```text
A^V(P,M) = A(P[...,4:-4,4:-4], M[...,4:-4,4:-4])
```

- PAN/MS 두 view의 원점·크기를 동일하게 한다. 이 경우 global translation 값은 원래 HR px 단위 그대로다.
- Crop 후 정규화한다. Full-frame의 padding 오염 통계를 먼저 계산해 crop 안으로 전달하지 않는다.
- **이미 warp된 full Pε에서 crop**한다. 원 P를 먼저 잘라 그 작은 crop을 warp하면 다시 인공 경계가 생긴다.
- Native/corrupted step, N1/N2/N3, inference 모두 같은 margin 규칙을 사용한다.
- FR 512²라면 aligner view는 504²이며, U-Net은 512² 전체로 복원한다.
- 실제 source-coordinate 4-tap support 검사를 통해 crop 전체가 모든 허용 ε에서 유효한지 확인한다.

**학습용 PAN 영상, MS residual base, GT, U-Net 입력·출력은 자르지 않는다.** Aligner가 보정한 PAN은 원래 전체 크기에서 sampling한다.

### 5.2 비교의 한계

새 내부 view는 원 A1의 전체 입력과 다르므로 과거 A1 대비 변화에 crop 영향이 포함된다. N1/N2/N3는 동일 wrapper를 사용하므로 서로의 비교는 유지된다. 이 안전 처리를 조용히 끄거나 N2에만 켜지 않는다.

이 조치가 모든 interpolation cue를 제거하는 것은 아니다. Subpixel interpolation 자체의 위상 흔적은 내부에도 남을 수 있으므로 held-out 반응과 다른 resampling 조건의 진단을 함께 보고한다.

---

## 6. Warp·mask·loss domain 구현

### 6.1 이번 버전은 sequential sampling을 명시적으로 사용

```text
P → W(P, ε) = Pε → aligner correction ĉε → W(Pε, ĉε) = P̃ε
```

Corrupted step에는 **두 번의 raster interpolation**이 존재한다. Native/inference는 한 번이다. 이를 기존 v2의 “한 번 warp”와 동일하다고 설명하지 않는다.

좌표의 연속 모형에서는 shift가 합해지지만,

\[
\mathcal W(\mathcal W(P,\epsilon),c)
\not\equiv \mathcal W(P,\epsilon+c)
\]

가 일반적인 discrete 영상에서는 성립한다. 첫 보간의 smoothing/overshoot와 경계 정보 손실이 남기 때문이다. 이번 N1/N2/N3는 **같은 sequential 경로**를 사용한다. Composed single-sampling으로 일부 case만 바꾸지 않는다.

원 영상으로부터 알려진 ε를 합산해 한 번에 sampling하는 방식은 다음 별도 비교 후보이다. 이번 구현에서 코드가 이미 composed라면 run 이름·manifest를 구분하고 모든 대조를 같은 경로로 다시 맞춰야 한다.

### 6.2 Gradient/precision

- Aligner, grid, warp, offset loss는 FP32/autocast disabled.
- U-Net AMP·GradScaler 정책만 실제 A1을 계승한다.
- Learned ĉ가 0이어도 `grid_sample`을 실행한다. 0이면 return하는 shortcut은 aligner gradient를 끊는다.
- Corruption ε는 학습할 parameter가 아니므로 생성 warp를 no-grad로 실행해도 된다. **두 번째 learned warp는 no-grad로 감싸지 않는다.**
- `c.detach()`, `.item()`으로 grid 재조립, numpy/scipy warp는 learned correction 경로에서 금지한다.
- Model output을 tanh×radius로 제한하지 않는다. Corruption radius와 전체 correction의 가능한 범위는 다르다.
- 학습용 PAN/HRMS에 clamp를 추가하지 않는다. 기존 evaluator의 export/clipping만 유지한다.
- CUDA sampling backward의 재현성은 실제 설치 환경에서 확인한다. Seed가 같다고 서버 간 수치 동치를 주장하지 않는다. [R3]

### 6.3 Reconstruction mask는 새로 만들지 않는다

L_rec는 A1/B0와 동일한 전체 GT 영역의 평균 L1을 유지한다. PAN의 일부 관측이 border로 대체되더라도 MS와 GT 자체가 사라진 것은 아니며, 평가 mask를 loss에 자동 연결하지 않는다.

특히 **예측 ĉ에 따라 L_rec 영역을 줄이면 안 된다.** 큰 shift를 내어 어려운 GT를 제외할 수 있다. 고정 내부 recon crop 실험도 이번 세 run에는 넣지 않는다.

대신 다음을 기록한다.

- 추가 warp 및 두 번째 warp의 source support와 border 사용 비율.
- 고정 내부/경계의 reconstruction 오차 분해 — 학습 loss는 바꾸지 않음.
- 큰 correction/비유한 값/정의 불가능 crop에서 명시적인 failure.

Native 평가의 고정 ROI와 corruption stress 평가의 ROI는 별도 manifest를 사용한다. Corruption stress에는 ε와 c의 **두 단계 실제 support**를 고려한다. ε+c의 최종 좌표만 보면 첫 번째 interpolation에서 이미 잃은 정보를 놓칠 수 있다.

---

## 7. 초기화·학습 설정·RNG

**주 서버는 완료된 A1 block이 있는 s1, seed는 2025**로 둔다. 새 case를 서버별로 나누지 않는다. 다른 서버를 써야 하면 세 case 모두 같은 서버로 옮기고, s1 수치와 서버 효과를 섞지 않는다.

1. 실제 A1 `config_resolved`, data manifest, evaluator manifest, code commit을 연결한다.
2. 동일한 random-init U-Net과 aligner state를 한 번 저장하고 모든 run이 로드한다.
3. B0/A1의 완료된 weight로 fine-tune하지 않는다. 추가 학습 효과를 혼합하지 않는다.
4. 50,000 optimizer updates를 목표로 한다. 결과 보고서는 last=50K라고 기록한다. [S1 §1] Epoch label이 아니라 update 수로 비교한다.
5. Optimizer·LR·weight decay·batch·AMP·scheduler는 실제 A1의 resolved config를 그대로 사용한다. 과거 다른 backbone의 값을 추정해 넣지 않는다.
6. 학습 시작 전에 A1의 실제 값이 기존 기준과 같은지 확인한다: AdamW, LR 1e-4, weight decay 0.01, warmup 100, batch 48은 **확인 후보**이며 자동 덮어쓰는 확정값이 아니다.
7. Native/corrupt 교대는 optimizer update 기준이다. Gradient accumulation이면 모든 microbatch가 그 update의 동일 경로를 사용한다.
8. Corruption RNG는 model/data/augmentation RNG와 분리한다. N1/N2/N3가 같은 sample과 update에 같은 ε를 받는지 첫 1,000개 기록의 hash로 비교한다.
9. Resume은 update index, data sampler, corruption generator, scaler/optimizer/scheduler, selector state를 모두 복구한다. 설정 hash 불일치면 기존 run 이름으로 이어가지 않는다.

추가 λ, crop, radius를 run 중 성능에 따라 변경하지 않는다. 변경이 필요하면 원 run을 보존하고 새 protocol ID로 구분한다.

---

## 8. 구현 reference core

**인터페이스:** 기존 aligner는 `aligner(P_view, M_view) → [B,2]`, 기존 U-Net은 `residual_unet(cat[P̃,M]) → residual`을 반환한다고 가정한다. 실제 wrapper가 이미 M을 더한다면 중복 합산하지 않도록 adapter를 작성한다.

아래 코드는 loss와 forward의 reference이다. DataLoader·실제 W96 U-Net 생성·optimizer·eval·로그 저장을 포함한 완제품 trainer는 아니다. §15의 범위에서 CPU 검사를 수행했다.

```python
from __future__ import annotations

import math
from typing import Literal
import torch
from torch import Tensor, nn
import torch.nn.functional as F

Case = Literal['N1', 'N2_SG', 'N3_NOSG']


def sample_offsets(batch: int, radius_hr: float, generator: torch.Generator,
                   device: torch.device) -> Tensor:
    """Uniform in a 2-D disk, not uniform in radius. Output order: (dy, dx)."""
    if batch <= 0 or not math.isfinite(radius_hr) or radius_hr <= 0:
        raise ValueError('batch and observed radius must be positive')
    # A dedicated CPU generator keeps this RNG separate from model/data RNGs.
    u = torch.rand(batch, 2, generator=generator, dtype=torch.float32)
    radius = radius_hr * u[:, 0].sqrt()
    angle = 2.0 * math.pi * u[:, 1]
    return torch.stack((radius * angle.sin(), radius * angle.cos()), 1).to(device)


def warp_pan(pan: Tensor, delta_yx: Tensor) -> Tensor:
    """out[y,x] = pan[y+dy,x+dx]; no learned-zero shortcut."""
    if pan.ndim != 4 or pan.shape[1] != 1:
        raise ValueError('pan must have shape [B,1,H,W]')
    b, _, h, w = pan.shape
    if delta_yx.shape != (b, 2):
        raise ValueError('delta must have shape [B,2] in HR pixels')
    with torch.autocast(device_type=pan.device.type, enabled=False):
        p = pan.float()
        d = delta_yx.to(device=p.device, dtype=torch.float32)
        if not bool(torch.isfinite(d).all()):
            raise FloatingPointError("non-finite displacement before grid sampling")
        # This cast preserves the autograd path to d; do not use torch.tensor(d).
        y, x = torch.meshgrid(
            torch.arange(h, device=p.device, dtype=torch.float32),
            torch.arange(w, device=p.device, dtype=torch.float32), indexing='ij')
        gx = 2.0 * (x[None] + 0.5 + d[:, 1, None, None]) / w - 1.0
        gy = 2.0 * (y[None] + 0.5 + d[:, 0, None, None]) / h - 1.0
        grid = torch.stack((gx, gy), -1)  # grid is (x,y), unlike delta
        return F.grid_sample(p, grid, mode='bicubic', padding_mode='border',
                             align_corners=False)


def aligner_margin(radius_hr: float) -> int:
    return 4 * math.ceil((radius_hr + 2.0) / 4.0)


def valid_view(x: Tensor, margin: int) -> Tensor:
    if margin < 0 or min(x.shape[-2:]) <= 2 * margin + 4:
        raise ValueError('invalid fixed aligner crop')
    return x if margin == 0 else x[..., margin:-margin, margin:-margin]


def predict_c(aligner: nn.Module, pan: Tensor, ms_base: Tensor,
              margin: int) -> Tensor:
    # The supplied A1 aligner performs its own per-band z-score normalization.
    with torch.autocast(device_type=pan.device.type, enabled=False):
        c = aligner(valid_view(pan.float(), margin),
                    valid_view(ms_base.float(), margin))
    if c.shape != (pan.shape[0], 2):
        raise ValueError('aligner must return [B,2]')
    if not bool(torch.isfinite(c).all()):
        raise FloatingPointError('non-finite aligner output')
    return c


def offset_loss(c_eps: Tensor, c0: Tensor, eps: Tensor,
                stop_reference: bool) -> Tensor:
    reference = c0.detach() if stop_reference else c0
    # No per-sample division by |eps|, which can be arbitrarily close to zero.
    return (c_eps + eps - reference).abs().mean()


def training_loss(aligner: nn.Module, residual_unet: nn.Module,
                  pan: Tensor, ms_base: Tensor, gt: Tensor, *,
                  case: Case, update_index: int, radius_hr: float,
                  generator: torch.Generator, lambda_max: float = 0.01,
                  ramp_updates: int = 5000, amp_enabled: bool = False):
    """Reference core. The caller supplies B0 preprocessing and optimizer logic."""
    if case not in ('N1', 'N2_SG', 'N3_NOSG'):
        raise ValueError('unknown case')
    if update_index < 0 or ramp_updates < 1 or lambda_max < 0:
        raise ValueError('invalid training controls')
    if ms_base.shape != gt.shape or ms_base.shape[0] != pan.shape[0] or \
            ms_base.shape[-2:] != pan.shape[-2:]:
        raise ValueError('M/GT/P shape mismatch')
    if amp_enabled and pan.device.type != 'cuda':
        raise ValueError('this reference enables AMP only for CUDA')
    margin = aligner_margin(radius_hr)
    corrupt = update_index % 2 == 1  # native, corrupt, native, corrupt, ...
    if corrupt:
        eps = sample_offsets(pan.shape[0], radius_hr, generator, pan.device)
        with torch.no_grad():
            p_eps = warp_pan(pan, eps)
        if case == 'N3_NOSG':
            c0 = predict_c(aligner, pan, ms_base, margin)
        else:
            with torch.no_grad():
                c0 = predict_c(aligner, pan, ms_base, margin)
        c = predict_c(aligner, p_eps, ms_base, margin)
        p_aligned = warp_pan(p_eps, c)  # sequential 2-raster-warp route
        l_off = offset_loss(c, c0, eps, stop_reference=(case != 'N3_NOSG'))
    else:
        eps = torch.zeros(pan.shape[0], 2, device=pan.device)
        c = predict_c(aligner, pan, ms_base, margin)  # native rec has gradient
        c0 = c
        p_aligned = warp_pan(pan, c)
        l_off = c.new_zeros(())
    with torch.autocast(device_type=pan.device.type, enabled=amp_enabled):
        residual = residual_unet(torch.cat((p_aligned, ms_base), dim=1))
    pred = ms_base.float() + residual.float()
    l_rec = (pred - gt.float()).abs().mean()  # full B0 domain; no moving mask
    weight = 0.0 if (case == 'N1' or not corrupt) else \
        lambda_max * min(1.0, (update_index + 1) / ramp_updates)
    total = l_rec + weight * l_off
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('non-finite training objective')
    # Live tensors allow optional gradient diagnostics before total.backward().
    return total, {'rec': l_rec, 'off': l_off, 'weight': weight,
                   'c': c, 'c0': c0, 'eps': eps, 'pred': pred,
                   'pan_aligned': p_aligned, 'corrupt': corrupt}
```

### Trainer에 연결하는 순서

```python
# 공통 augmentation/intensity scaling이 완료된 pan, lrms, gt를 사용
M = existing_b0_bicubic(lrms)          # 실제 A1 경로 그대로
optimizer.zero_grad(set_to_none=True)
loss, info = training_loss(
    aligner, residual_unet, pan, M, gt,
    case=case_id,
    update_index=global_optimizer_step,
    radius_hr=calibration['radius_hr'],
    generator=corruption_generator,
    amp_enabled=resolved_a1_amp,
)
# 진단이 필요한 update에는 info['rec']/info['off']에서 autograd.grad를 먼저 계산.
# 그 뒤 기존 A1의 GradScaler/clip/optimizer/scheduler 순서를 그대로 사용.
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
apply_existing_a1_gradient_clipping_if_any()
scaler.step(optimizer)
scaler.update()
scheduler.step()
# info의 live tensor를 그대로 history에 append하지 말고 detach하여 수치만 저장.
```

Gradient accumulation을 쓰는 실제 trainer에서는 마지막 블록을 microbatch마다 optimizer update하는 형태로 복사하지 않는다. Info가 보유한 graph는 해당 update 후 해제한다.

---

## 9. Aligner에 어떤 gradient가 들어오는지 기록

사용자가 제기한 “정합이 recon에 중요하지 않으면 gradient가 약하지 않은가?”를 실제로 구분하기 위한 항목이다. **Loss 숫자만 비교하지 않는다.**

매 1,000 optimizer update 부근의 고정된 native/corrupt minibatch에서, 같은 forward의 loss를 분리해 기록한다. 진단 backward는 optimizer step을 추가하지 않는다.

\[
g_{rec}^{\phi}=\nabla_\phi L_{rec},\qquad
g_{off}^{\phi}=\nabla_\phi L_{off},\qquad
g_{off,w}^{\phi}=\lambda_{off}g_{off}^{\phi}.
\]

필수 값:

| 로그 | 해석 |
|---|---|
| `dLrec_dc`의 norm/RMS | Loss가 PAN 보정량에 얼마나 민감한가? |
| `grad_rec_phi`, `grad_off_phi`, `grad_off_weighted_phi` | Aligner parameter에 전달되는 실제 크기 |
| 같은 parameter 벡터에서 두 gradient의 cosine | 서로 같은 방향인지, 충돌하는지 |
| `grad_off_theta` | U-Net parameter에는 None/0이어야 함 |
| Native `grad_rec_phi` | clean correction이 학습에서 고정되지 않았는지 |
| N2 `c0.requires_grad` / N3 `c0.requires_grad` | stop-gradient 설정이 실제로 다른지 |
| 마지막 head/앞단 stem의 gradient 분리 | 초기 zero head와 이후 학습을 구분 |
| `parameter_update_norm_phi` / `prediction_change_norm` | Gradient가 optimizer update와 행동 변화로 이어지는지 |

`torch.autograd.grad(..., retain_graph=True, allow_unused=True)`를 진단용으로 사용한다. None gradient는 0으로 기록하되 원래 unused 여부도 남긴다. Norm이 0이면 cosine을 NaN/undefined로 표시한다. 분모에 작은 상수를 넣어 의미 없는 cosine을 만들지 않는다. [R4]

N1에서도 offset loss의 가상 gradient를 진단할 수 있지만 **optimizer update에는 coefficient 0**을 유지한다. N2와 N3는 λ가 같아도 전체 gradient 크기가 달라질 수 있다. 그 차이 자체를 함께 보고한다.

**주의:** L1 reconstruction 값이 작다고 gradient가 비례해 작아지는 것은 아니다. 중요한 것은 PAN 위치와 output에 대한 Jacobian이다. 또한 random initialization의 첫 backward에서 zero head 앞의 stem gradient가 0인 것은 예상되는 동작이다. 두 번째 이후 학습까지 확인한다.

---

## 10. 평가와 checkpoint

### 10.1 Native 품질: 기존 세 view 유지 [S2 §10]

추론에는 추가 ε를 넣지 않는다. 같은 native 예측 Yhat와 실제 사용 PAN P̃로 다음을 모두 계산한다.

| View | PAN reference | 영역 |
|---|---|---|
| Raw-original | 원 P | 기존 evaluator의 Ω₀ |
| Raw-valid | 원 P | 고정 공통 V |
| Aligned-valid | 실제 forward의 P̃ | 같은 V |

- 원본 MS·GT·예측을 이동하지 않는다.
- Aligned 평가의 저해상도 PAN도 P̃에서 기존 evaluator 경로로 다시 만든다.
- Raw-valid와 Aligned-valid의 fSCC를 모두 기록해 crop 효과와 reference 효과를 나눈다.
- 실제 보고서에 사용된 **FR mat20, metric v2 py**를 유지한다. 12–19로 되돌리지 않는다. [S1 머리말]
- Scene별 HQNR `(1−Dλ_i)(1−D_s_i)`를 계산한 뒤 평균한다.
- GT 기반 RR 지표는 같은 M-frame끼리 계산한다.
- Fixed V에서 한 장면이라도 support가 실패하면 aligned selection을 INVALID로 기록한다. 해당 장면을 빼거나 crop을 줄여서 best를 만들지 않는다.

### 10.2 저장 기준

`best_raw`, `best_aligned`, 정확한 `last(50K)`를 독립 보존한다. 두 best는 running maximum HQNR을 기준으로 하고, `1e-4` 동률에서 해당 view의 fSCC, 다시 동률이면 더 늦은 checkpoint를 고른다. Native 평가만 checkpoint 선택에 사용한다.

최종 표는 모든 저장 checkpoint에 대해 세 view를 교차 보고한다. 같은 크기로 직접 비교할 수 없는 dynamic-mask 점수는 diagnostic으로만 남긴다. 과거 A1/B0에 새 모델의 shift를 이식한 점수를 주 비교로 사용하지 않는다.

### 10.3 기능 검사: 추가 변위에 대한 반응

Training/calibration과 분리된 sample ID 및 별도 ε seed를 사용한다. 원 sample의 참 정합량은 몰라도 다음 차이는 검사할 수 있다.

\[
q_i(\epsilon)=\hat c_\epsilon-\hat c_0.
\]

2차원 선형 반응을 fit하면

\[
q\approx B\epsilon+b,\quad\text{목표 }B\approx-I_2,\ b\approx0.
\]

기록:

- 같은 축 slope: −1에 가까울수록 추가 shift를 올바른 크기로 상쇄.
- Cross-axis slope: 0에 가까워야 함.
- Offset closure error: mean `||ĉε−ĉ₀+ε||₂`, P50/P90, HR px.
- Epsilon에 비해 상대 error가 어떤지 **고정 R**로 나눈 값도 기록. Sample별 ε=0 근처로 나누지 않음.
- Native ĉ₀의 mean/median/IQR와 training 중 bias drift.
- 학습 크기 64², RR 256², FR 512²에서 각각 보고. 새로운 크기에서 ×4/×8로 출력을 보정하지 않음.

Corruption 분포 내부 random sample과, 진단 전 고정한 `0, ±R/2, ±R` 축 방향 probe를 사용한다. `±2R`은 외삽 stress로만 추가한다. 학습 범위 밖의 실패를 범위 내 실패와 섞지 않는다.

FR test의 synthetic stress는 **최종 diagnostic만** 수행하며, 이 결과로 radius·λ·학습 시점을 선택하지 않는다. 정합 GT가 없는 FR의 native 절대 정합 성능과 synthetic closure error를 같은 뜻으로 쓰지 않는다.

### 10.4 Padding·MS 활용·보간 대조

1. 같은 valid center를 남기고 outer padding 방식을 바꿔 aligner 예측이 달라지는지 확인한다. §5가 정확하면 crop 안의 입력이 같아야 한다.
2. MS를 다른 sample과 바꾸거나 상수로 대체하는 **진단만** 수행한다. 그때도 synthetic closure만 잘 맞는다면 PAN의 보간 흔적만 읽는 shortcut 가능성을 배제할 수 없다. Native quality·MS 의존성과 함께 해석한다.
3. Same-weights `zero correction / learned / wrong sign`을 비교한다. Learned가 항상 양쪽보다 좋다고 미리 가정하지 않는다.
4. `W(W(P,ε), ĉ₀−ε)`와 `W(P,ĉ₀)`의 공통 내부 차이를 계산한다. 전자는 **알려진 상대 offset만 정확히 되돌린 비교 경로**이며, 절대 정합 oracle이 아니다. 두 번 보간의 오차 바닥을 보여 준다.
5. 성능이 좋아도 synthetic 생성 kernel을 바꾼 진단에서 반응이 사라지면 geometric 일반화보다 interpolation cue 의존 가능성을 기록한다. 주 benchmark를 다른 kernel로 바꾸지 않는다.

---

## 11. 10시간 실행 큐

### 11.1 예약 예산 — GPU 실측 예측 아님

| 순서 | 작업 | GPU-hour 예약 | 누적 |
|---:|---|---:|---:|
| 0 | Source/규모 확인, unit·실배치 smoke, throughput 측정 | 1.0 | 1.0 |
| 1 | N1 CORRUPT-REC, 동일 full budget | 2.2 | 3.2 |
| 2 | N2 OFFSET-SG, 동일 full budget | 2.6 | 5.8 |
| 3 | N3 OFFSET-NOSG, **시간 적격 시** 같은 full budget | 2.7 | 8.5 |
| 4 | 세 view 최종 export·반응/gradient 진단·표 | 1.0 | 9.5 |
| 5 | Buffer | 0.5 | **10.0** |

각 run 예산은 주기적 native evaluation을 포함하도록 profiling한다. 코드를 작성하는 사람의 시간과 서버 대기 시간은 GPU-hours와 분리한다. N3보다 N1/N2의 완주와 진단을 우선한다. 추가 architecture/X-case나 seed 반복으로 남은 시간을 자동 채우지 않는다.

### 11.2 실행량 결정 규칙

50K를 2.6시간에 끝낸다는 보장은 없다. 시작 전에 native/corrupt 각각 warmup 제외 update time, eval/export 시간을 측정한다.

\[
\hat T_{case}=N\frac{t_{native}+t_{corrupt}}2+T_{eval}+T_{export}.
\]

- N1/N2 두 full run과 최소 1시간 최종 진단이 들어가는지 먼저 확인한다.
- N3는 예상 잔여 비용에 20% 여유를 더하고도 10시간 한도 안에 들어갈 때만 시작한다. 그렇지 않으면 `DEFERRED_BUDGET`로 남긴다.
- **N3만 짧게 돌린 값을 50K N2와 비교하지 않는다.**
- N1/N2 full 두 개가 들어가지 않으면, 첫 full run 시작 전에 공통 25K screening으로 프로토콜을 변경할 수 있다. 이 경우 scheduler 길이도 양쪽 모두 25K로 맞추고 ID에 `SCREEN25K`를 붙이며 과거 50K B0/A1 대비 full-budget 개선이라고 부르지 않는다.
- 25K pair+진단도 안 들어가면 full queue를 시작하지 않고 gate 결과를 보고한다.
- 실제 도중 deadline을 넘길 위험이 생기면 추가 case를 시작하지 않는다. 중단된 run은 `INTERRUPTED_BUDGET`이며 partial checkpoint를 정상 last로 이름 붙이지 않는다.

### 11.3 제안 run ID

```text
PO10_N1_REC_W96_D124_WV3_S2025
PO10_N2_OFFSG_W96_D124_WV3_S2025
PO10_N3_OFFNOSG_W96_D124_WV3_S2025
```

50K/25K, scale profile, crop, seed는 resolved manifest와 run ID suffix에 구분한다. 기존 A1·A2·A3 경로에 덮어쓰지 않는다.

---

## 12. 실행 전 gate

| Gate | 반드시 확인할 내용 |
|---|---|
| G0 provenance | 기존 A1 config/data/metric hash, 50K update 정의, 초기 tensor hash |
| G1 scale | 통계의 train 출처·P90·LR 단위 확인, R=1 HR px 환산; FR/CNN 출력을 규모로 쓰지 않음 |
| G2 sampling | Sample별 2D disk, 평균 0, 축별 표준편차 R/2, L2 상한 R, 독립 RNG |
| G3 sign | Impulse에서 `(dy,dx)` 부호·axis·4배 환산, 0 warp gradient 유지 |
| G4 reference | SG에서는 corrupted loss→c0 차단, no-SG에서는 연결. Native recon은 둘 다 연결 |
| G5 loss routing | Loff→aligner 연결, Loff→U-Net 없음, Lrec→aligner/U-Net 연결 |
| G6 valid view | 모든 허용 ε에서 A-view의 nominal 4-tap support 유효, crop 후 normalize |
| G7 interpolations | 실제 sequential 두 warp 사용, integer roundtrip 확인; subpixel 완전 복구를 가정하지 않음 |
| G8 actual smoke | 실제 batch·AMP·accumulation·optimizer·eval로 finite/OOM/시간 확인 |
| G9 eval | Native 세 view, ROI/scene hash, 정확한 last, invalid handling, 독립 dual-best selector |

단위 검사 tolerance는 환경 정밀도에 맞게 기록한다. Sign/order/gradient 연결 실패는 모델 성능 실패가 아니라 **구현 오류**이므로 full run을 시작하지 않는다. 성능이 낮다고 gate 기준을 사후 완화하지 않는다.

---

## 13. 로그와 결과 판정

### 13.1 필수 파일

```text
work_dir/<run_id>/
  config_resolved.yaml
  source_and_init_hashes.json
  corruption_scale_manifest.json
  selection_roi_manifest.json
  train_log.jsonl
  gradient_diagnostics.jsonl
  offset_response_native64.csv
  offset_response_rr256.csv
  offset_response_fr512.csv
  interpolation_controls.json
  checkpoint_metrics.csv
  scene_metrics.csv
  best_raw.pt / best_raw_meta.json
  best_aligned.pt / best_aligned_meta.json
  last.pt
  budget_ledger.json
```

`train_log`: update, native/corrupt, rec/off/weighted_off, ε mean/std/max_norm, ĉ₀/ĉε mean/median/std, correction closure, finite/support flags, iteration time, memory. N1에서도 raw off diagnostic을 남긴다.

### 13.2 결과표

| Case | Budget/update | best_raw HQNR | Raw-valid / Aligned-valid HQNR | Raw / Raw-valid / Aligned fSCC | RR ERGAS | response diag(B) | off EPE | native bias | Train GPUh |
|---|---|---|---|---|---|---|---|---|---|
| Historical A1 | 기존 50K | 보고서 값 + evaluator 일치 확인 | … | … | … | … | … | … | 기존 기록 |
| N1 | … | … | … | … | … | … | … | … | … |
| N2 | … | … | … | … | … | … | … | … | … |
| N3 | … 또는 deferred | … | … | … | … | … | … | … | … |

Historical A1은 전체 aligner view이므로 배경 기준이다. N2−N1이 direct offset loss의 주 효과이며 N3−N2가 stop-gradient의 효과이다. 한 seed 결과를 유의한 3-seed 개선으로 보고하지 않는다.

### 13.3 해석

- **N2 반응 개선 + native 품질 개선:** 직접 offset 감독을 유력 후보로 유지. 절대 정합 참값 검증까지 끝난 것은 아님.
- **반응만 개선, native 동일:** 추가 변형에 대한 강건성은 학습됐지만 기존 분포에서 정합이 성능 병목이 아닐 수 있음.
- **반응 개선, native 악화:** 절대 bias drift, warp smoothing, PAN 활용도, 크기/정규화 일반화를 먼저 확인. λ를 즉시 키우지 않음.
- **N1≈N2이며 둘 다 개선:** 입력 변동만으로 충분할 가능성. 추가 loss 필요성이 약함.
- **N1 반응 약함, N2 강함:** 약한 recon 기반 정합 신호를 직접 offset supervision이 보완했다는 근거.
- **N3만 더 좋음:** 비대칭 target이 반드시 유리하지 않다는 결과. Gradient 크기·방향까지 함께 비교.
- **Loss만 작고 MS 교체에도 반응 동일:** padding/interpolation shortcut 또는 단일 모달리티 단서 가능성. 정합 성공으로 부르지 않음.
- **모든 경우 native ĉ₀가 거의 일정:** 원래 데이터가 공통 offset이면 그 자체는 실패 기준이 아니다. 알려진 ε에 대한 차이의 반응을 우선 본다.

Checkpoint는 HQNR 기준을 유지하지만, **방법론 기능 판정은 response와 native 품질 두 축**으로 한다. 10시간 결과만으로 반응 slope 임계값을 새로운 성능 선택 기준으로 삼지 않는다.

---

## 14. 설정 template

다음은 새 구현에 필요한 명세 template이며, 실제 repository의 기존 key 이름을 확인하지 않고 복사 실행하는 완제품 config는 아니다.

```yaml
protocol_id: pan_offset_rel_10h_v1
budget:
  total_gpu_hours: 10
  primary_server: s1
  seed: 2025
  required_cases: [N1, N2_SG]
  conditional_cases: [N3_NOSG]
  full_updates: 50000
  paired_screen_updates_if_needed: 25000
  drop_optional_before_shortening_required: true

inherit:
  a1_resolved_config: REQUIRED_ACTUAL_PATH
  dataset_manifest: REQUIRED_ACTUAL_PATH
  native_eval_manifest: REQUIRED_ACTUAL_PATH
  init_unet_hash: REQUIRED_ACTUAL_VALUE
  init_aligner_hash: REQUIRED_ACTUAL_VALUE

model:
  width: 96
  depth: [1, 2, 4]
  input: [PAN, bicubic_MS]
  pan_reconstruction: false
  lpan_hpan_channels: false
  aligner_arch: A1_dualstem_105330
  aligner_output_order: [dy, dx]
  aligner_output_units: current_PAN_HR_pixel
  aligned_modality: PAN_only
  output_frame: M

corruption:
  scale_source: training_audit_appendix_E_WV3_global_norm_P90
  calibration_quality: train_aggregate_proxy
  observed_value_lr: 0.250
  ratio: 4
  radius_hr: 1.000
  distribution: uniform_disk_area
  center_yx: [0, 0]
  sample_each_corrupt_update_and_sample: true
  add_after_common_augmentation: true
  native_corrupt_step_ratio: [1, 1]
  rng: dedicated_cpu_generator
  pass_epsilon_to_network: false

aligner_view:
  policy: fixed_interior_crop
  margin_rule: '4 * ceil((radius_hr + 2) / 4)'
  resolved_margin_hr: 4
  apply_to: [aligner_P, aligner_M]
  normalize_after_crop: true
  apply_to_unet_or_gt: false
  inference_same_rule: true

warp:
  convention: 'out[y,x]=source[y+dy,x+dx]'
  mode: bicubic
  padding_mode: border
  align_corners: false
  learned_zero_shortcut: false
  corrupted_route: sequential_two_raster_warps
  native_route: single_learned_warp
  geometry_precision: fp32

loss:
  rec: mean_L1_full_A1_domain
  offset: mean_abs_c_eps_plus_eps_minus_c0
  offset_lambda_max: 0.01
  offset_ramp_optimizer_updates: 5000
  native_offset_lambda: 0
  normalize_by_sample_epsilon: false
  case_overrides:
    N1: {offset_lambda_max: 0, clean_reference_gradient: false}
    N2_SG: {offset_lambda_max: 0.01, clean_reference_gradient: false}
    N3_NOSG: {offset_lambda_max: 0.01, clean_reference_gradient: true}

selection:
  primary: best_raw
  secondary: best_aligned
  keep_exact_last: true
  views: [raw_original, raw_valid, aligned_valid]
  scene_set: fr_mat20
  metric_version: inherited_A1_v2_py
  hqnr_aggregation: mean_scene_products
  fixed_roi_required: true
  select_on_synthetic_stress: false
```

---

## 15. 이 문서를 만들면서 확인한 범위

본 문서의 reference core는 **CPU PyTorch 2.10.0**에서 다음을 확인했다. 이는 실제 학습 실험의 결과가 아니다.

- 20,000개 disk sample의 mean `(dy,dx)` ≈ `(0.00047, 0.00102)`, std ≈ `(0.49923, 0.49803)`, 최대 L2 norm ≈ 0.99993. R=1 분포 검사값과 일치.
- Learned zero-shift warp에서 값이 identity에 가깝고 correction gradient가 존재함.
- 양의 sampling dy=1에서 impulse 내용이 위쪽으로 1pixel 이동함.
- 정수 shift→inverse의 공통 내부 영역 복원 확인. **Subpixel roundtrip 무손실은 주장하지 않음.**
- SG loss는 참조 leaf의 gradient를 차단하고, no-SG는 양쪽으로 반대 부호 gradient를 보냄. 같은 입력에서 forward loss 값은 같음.
- 원래 correction이 0이 아닌 경우에도 `c_eps=c0−eps`의 상대 loss가 0이 되는 것 확인.
- 첨부 ZIP의 A1 reference aligner를 사용해 **105,330 parameters**를 확인하고, dummy residual convolution으로 N1/N2/N3 각각 4 update의 finite backward를 확인.

**확인하지 않은 것:** 실제 PanCollection tensor, 실제 W96–D124 U-Net, 서버 trainer integration, 실제 CUDA/AMP backward, 실배치 throughput, 파일 evaluator와 ROI의 support, 모델 성능. §12의 서버 gate가 이 검사를 대체하지 않으며 반드시 별도로 실행해야 한다.

---

## 16. 자료 출처와 새 제안의 구분

### 내부 자료

- **[S1]** `2026-09-10_pa-a1-a3-s1-results.md`: s1 결과, 반응 기울기, gradient, synthetic 검사 보고, 50K last, mat20 평가.
- **[S2]** `PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md`: §3 aligner 구조, §4 warp/gradient, §7 초기화·학습 동치, §10 세 view/ROI/selector. 본 문서의 crop·corruption·offset loss는 신규 변경.
- **[S3]** `2026-09-03_alignment-audit-s2-detail.md`: 부록 E의 WV3 train global norm P90=0.250 LR px; §I.0 단위/부호; §II.8.5 spectral 하한. 이 통계는 임시 규모 proxy이며 정합 label이 아니다.
- **[S4]** `2026-09-04_global-alignment-plan-review (1).md`: §1.2 작은 train patch의 추정기 노이즈 경고. 과거 해당 문서의 interp23tap/dual-MARs 학습 결정은 승계하지 않음.
- **[S5]** `PAN_A1_A3_S1_Result_Review_and_NextCases_2026-09-10.md`: §5 상대 변위 감독의 제안과 §7 sampling/mask 위험. 이번 문서는 이후 사용자가 합의한 관측 기반 0 중심 범위를 반영해 과거 ±2 합성 검사 범위의 무조건 재사용을 대체함.

### 구현 API 확인 — 외부 1차 자료

- **[R1]** PyTorch `Tensor.detach`: `https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.detach.html`
- **[R2]** PyTorch `no_grad`: `https://docs.pytorch.org/docs/stable/generated/torch.no_grad`
- **[R3]** PyTorch `grid_sample`: `https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.grid_sample.html`
- **[R4]** PyTorch `autograd.grad`: `https://docs.pytorch.org/docs/2.13/generated/torch.autograd.grad.html`

API 동작을 확인하기 위한 참고이며 서버에 해당 문서 버전이 설치돼 있다고 가정하지 않는다. Noisier2Noise라는 용어는 추가 corruption의 동기를 설명하는 수준으로 사용한다. **알려진 상대 기하 변환의 자기감독**을 원래 denoising 논문의 무편향 복원 정리와 동일시하지 않는다.

### 이번 계획에서 새로 정한 것

세 case 구성, train aggregate P90을 이용한 임시 규모 규칙, disk 분포, native/corrupt 1:1, aligner-only 4px 내부 crop, λ=0.01/5K ramp, 10시간 배치는 모두 **이번 실행을 위한 설계값**이다. 기존 실험에서 검증됐다고 주장하지 않는다. 필요한 실제 A1 파일 경로·hash·throughput은 `REQUIRED`로 남기며 임의로 채우지 않는다.

---

## 17. 실행자 인계문

> W96–D124/9ch/A1 global aligner를 유지한다. 추가 변위는 PAN에만 넣고, GT와 MS base는 움직이지 않는다. 원본 입력의 correction을 c0, 추가 ε 입력의 correction을 cε라고 할 때, 목표는 **cε+ε≈c0**이다. c0는 edge 추정 GT가 아니라 같은 CNN의 현재 예측이다. N2에서는 이번 target만 stop-gradient하고, native reconstruction에서는 c0를 계속 학습한다. N3는 양쪽 역전파 대조다. 복원 U-Net은 step마다 한 번만 돌린다. 변위 규모는 train aggregate P90에서 얻은 R=1 HR px를 임시 proxy로 사용하고 실제 물리적 최대 오차라고 부르지 않는다. Padding 단서를 줄이기 위해 정합 CNN 입력만 고정 내부 crop하며 모든 새 case에 동일하게 적용한다. Native 품질과 synthetic 반응을 함께 보고, 절대 정합이 증명됐다고 확대하지 않는다. 10 GPU-hours 내에 N1/N2와 최종 진단을 우선 완주하고 N3는 예산 적격일 때만 추가한다.
