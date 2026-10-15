# WV3 FH12 — LPAN·HPAN 중심, Fresh Teacher → Student 전 서버 12시간 실험 계획

작성일: 2026-09-18  
Campaign: `WV3_FH12_TSCRATCH_20260918_v1`  
Method revision: `FH12_SYNC_FREQ_NATIVE_TEACHER_v1`  
대상: **s1–s5 전부 WV3**  
상태: **실험 설계·명시적 case registry·동기 warp 참조 코드 작성. 원격 학습 YAML/runner/Sheet 미적용.**

## 0. 이번 결정과 범위

중심 질문은 **PAN과 MS를 유지한 상태에서 LPAN 추가, HPAN 추가, 두 성분 추가가 각각 어떤 효과를 내는가**이다. `LPAN-only`라는 표현은 PAN을 제거한다는 뜻이 아니라 **추가 입력이 LPAN 하나**라는 뜻이다.

이번에는 기존 T0를 그대로 사용하지 않는다. 각 서버에서 **Teacher U-Net과 PAN aligner를 모두 새로 초기화해 50K 학습**하고, 그 Teacher를 고정해 Student를 fresh50K 학습한다. s4는 자기 Teacher 하나 아래에서 네 Student 입력을 모두 비교해 Teacher 차이가 섞이지 않는 핵심 대조를 담당한다.

- 핵심 입력: `P0`(9ch), `PL`(10ch), `PH`(10ch), `PLH`(11ch).
- Teacher 기본 구조: **W112D123**. 이번 12시간에는 Teacher capacity를 추가 탐색하지 않는다.
- Student 기본 구조: **W104D121**. 남는 시간에 **D122**를 우선, **W112D121**을 후순위로 확인한다. D124·W160·attention·PAN reconstruction grid는 열지 않는다.
- Student 계수는 G23의 `α=1, β=.1, λE=.002, lrU=1e-4, lrA=3e-6`을 유지한다.
- 목표: **raw-original HQNR ≥ .9585인 동일 checkpoint 중 ERGAS 최소**, 공동목표 ERGAS < 2.040.
- **서버 간 결과 대기·공통 recipe lock 없음.** 자기 Teacher와 calibration이 필요한 것은 local pipeline 의존성이며, 다른 서버 완료를 기다리는 조건이 아니다.
- QB/GF2 QG26은 계속 보류한다. 생성된 탭은 보존하되 이번 캠페인에서 해당 데이터 학습을 기동하지 않는다.

**기본 목표는 fresh Teacher 5개 + Student 11개 = 16개 학습 run이다.** 예비는 D122 3개와 width-only 2개, 즉 최대 5개 Student다. 12시간은 완료 개수 보장이 아니라 서버별 wall-clock 상한이다. 준비·기존 run 잔여·calibration·공식 평가까지 포함한다. 전체 조합의 전수 grid나 모든 조합의 독립 다중-seed 검증을 12시간에 완료한다는 뜻은 아니다.

## 1. 판단의 근거와 이번에 새로 제안하는 부분

이전 s1/s2 분석에서는 명목 9ch↔11ch 대응에서 HQNR·D_lambda 개선 방향이 반복됐지만 ERGAS 방향은 일관되지 않았다. 따라서 입력 표현을 먼저 분리하고 capacity를 소수 보조 case로 두는 것이 목적이다. 해당 과거 결과는 서로 다른 선택 시점과 평가 이력을 포함한 개발 관측이며, 본 캠페인의 예상 개선량으로 더하지 않는다.

프로젝트 연구자료의 Stage 1은 **PAN–MS로 보정량을 추정하고, native reconstruction은 A/U를 함께 갱신하며, synthetic PAN shift는 A-only consistency에 사용**하는 구조다. Stage 2는 frozen Teacher의 e/q로 Student를 감독한다. 본 계획은 이 흐름을 보존하되, reconstruction 입력에 L/H를 명시적으로 추가한다. [S1, 연구자료 pp.8–15]

PAN-Crafter의 down/up PAN residual 및 PAN auxiliary task와 본 프로젝트의 L/H concat은 구분한다. 본 실험은 **PAN reconstruction loss와 MARs mode γ/β를 추가하지 않는다.** 논문의 PAN auxiliary는 별도 mode와 loss를 가진다. [S2, Eq.(3)–(4), pp.3–4]

### 1.1 기존 T0의 재현이 아니라 새 Teacher 초기화 실험이다

기존 T0의 실제 lineage는 **N2 donor → donor A + fresh U의 L1E4 → Student**다. 이번 요청은 Teacher부터 새로 학습하는 것이므로 12시간 핵심안은 **donor가 없는 native-from-scratch L1E4형 Teacher 50K**로 정한다. [S3]

이는 시간을 맞추기 위해 donor를 사용하고도 fresh라고 부르는 설계가 아니다. **A도 U도 기존 학습 가중치를 읽지 않는다.** N2 단계 생략은 의도적인 새 초기화 프로토콜이며, 과거 T0와의 성능 차이를 L/H 효과 하나로 귀속하지 않는다. 네 입력 Teacher 모두 같은 새 프로토콜을 사용한다. N2→L1E4 전체 재구축은 이번 mandatory가 아니며, 실패 시 조용히 끼워 넣지 않는다.

## 2. Aligner 입력과 주파수 입력: 무엇을 어디에 넣는가

기호:

- `P`: native HR PAN, 1채널.
- `X`: LRMS, WV3 8채널.
- `M = U4(X)`: bicubic MS base, 8채널, 최종 출력/GT의 기준 좌표계.
- `LP`: PAN을 저역통과·4배 decimation한 LR PAN.
- `L = U4(LP)`: HR 격자로 올린 LPAN, 1채널.
- `H = P − L`: signed HPAN residual, 1채널. **wavelet의 horizontal subband H가 아니다.**

### 2.1 모든 case에서 Aligner는 동일한 두 입력을 본다

\[
\mathbf c=A_\phi(P_{m4},M_{m4}),\qquad \mathbf c=(c_y,c_x).
\]

**PAN 단독에서 c를 예측하는 것이 아니다.** 기존처럼 PAN과 upsampled MS의 관계를 사용한다. PAN stem은 1채널, MS stem은 8채널이다. LPAN/HPAN을 Aligner stem에 넣거나 Aligner별 c를 따로 만들지 않는다. [S4]

Aligner 입력만 margin 4 HR pixel을 crop한다. Train PAN64에서는 56, RR256에서는 248, FR512에서는 504 크기를 본다. U-Net·warp·MS base는 전체 크기다. Align 입력을 강제로 64로 resize하지 않는다.

**같은 Aligner 입력 구조라는 것과, 학습된 c가 모든 case에서 같다는 것은 다르다.** Reconstruction 경로가 다르면 A에 도달하는 gradient가 달라져 학습된 A도 달라질 수 있다. 고정 Teacher를 사용하는 Student 입력 대조에서도 Student A는 각 run에서 독립적으로 적응한다.

### 2.2 추천하는 동기 이동 정의

\[
\begin{aligned}
\widetilde P &= \mathcal W(P,\mathbf c),\\
\widetilde L &= \mathcal W(U_4(LP),\mathbf c),\\
\widetilde H &= \widetilde P-\widetilde L.
\end{aligned}
\]

| Layout | reconstruction U-Net 입력 순서 | WV3 채널 |
|---|---|---:|
| P0 | `[P_tilde, M]` | 9 |
| PL | `[P_tilde, L_tilde, M]` | 10 |
| PH | `[P_tilde, H_tilde, M]` | 10 |
| PLH | `[P_tilde, L_tilde, H_tilde, M]` | 11 |

모든 경우 최종 출력은 `Z = M + F(input)`이고, **MS base를 정확히 한 번만 더한다.** LRMS·M·GT를 c로 움직이지 않는다.

Teacher forward에는 **Teacher 자신의 c_T**, Student forward에는 **Student 자신의 c_S**를 사용한다. Student가 Teacher의 c를 직접 받아 추론하는 구조가 아니다. `teacher_input_layout`과 `student_input_layout`은 별도 필드로 유지한다.

### 2.3 c를 LPAN·HPAN에 함께 적용해도 되는가?

**고정된 동일 grid로 같은 HR 격자의 영상을 resample한다는 조건에서는 적절하다.** Bicubic interpolation은 grid가 주어지면 영상값에 대한 선형 연산이므로

\[
\mathcal W(P-L,\mathbf c)=\mathcal W(P,\mathbf c)-\mathcal W(L,\mathbf c)
\]

가 성립한다. 수치 연산 순서에 따른 작은 오차는 가능하다. 따라서 HPAN을 별도로 한 번 더 warp하지 않고, **동기 warp한 PAN과 LPAN의 차이**로 만들면 분해 일관성을 유지할 수 있다. 이것은 “추정된 c가 기하학적으로 정확하다”는 증명과는 다르다. [S5, S6; 선형성 식은 본 계획의 수학적 검토]

권장 구현은 `cat([P, L], channel)`에 **한 grid를 한 번 적용**하고 두 결과를 split한 뒤 H를 뺀다. c는 `(dy, dx)` HR pixel이며 grid의 마지막 차원만 `(x, y)`다.

\[
g_x=2(x+c_x+0.5)/W-1,\qquad g_y=2(y+c_y+0.5)/H-1.
\]

`mode=bicubic`, `padding_mode=border`, `align_corners=False`, frontend/warp FP32를 유지한다. c>0은 source sampling 위치를 증가시키는 부호다. **양의 dx를 그림 내용의 오른쪽 이동이라고 설명하면 부호가 반대일 수 있다.** 기존 warp와 같은 정의를 쓴다. [S5]

금지 사항:

1. LR 크기의 LP에 HR 단위 c를 그대로 적용하지 않는다. LR에서의 c/4와 upsample 후 c도 interpolation/phase 때문에 단순 등가라고 가정하지 않는다.
2. `P_tilde`와 이동하지 않은 native L을 빼지 않는다.
3. `W(U4(D4(P)),c)`를 임의로 `U4(D4(W(P,c)))`로 바꾸지 않는다. Filtering/decimation/warp는 일반적으로 교환되지 않는다.
4. P→L/HP마다 다른 padding, clip, normalization 또는 detach를 두지 않는다.
5. Student의 c를 LP/HP 분기에서 detach하지 않는다. Hard reconstruction의 gradient가 모든 사용 중인 PAN-derived 경로를 통해 A까지 도달해야 한다.

현재 `pa/model.py`는 P만 정합하고 LP 입력은 그대로 전달한다. 따라서 **`in_mode=released`만 바꾸면 이번 정의가 구현되지 않는다.** 또한 기존 모드는 9/11 두 종류이므로 PL/PH 10ch stem은 별도 지원이 필요하다. [S7]

### 2.4 LPAN 자체의 phase와 출처는 별개로 검증한다

같은 c를 적용하면 P/L의 **기존 상대적 phase를 보존**할 뿐, 잘못 생성된 LP의 phase까지 자동으로 고치지 않는다. 예를 들어 decimation offset과 bicubic center convention이 다르면 상수 phase가 남을 수 있다. Ramp/impulse로 이를 측정하고 `phase_id`로 기록한다. 이 기록을 보고 case별로 최적인 phase를 임의 변경하지 않는다.

이번 LP 공급 규약은 repository `repair_lpan.make_lpan`과 동일한 **Gaussian σ=1.98, kernel41, replicate padding, `[2::4,2::4]` decimation**을 사용한다. 이것은 저장소에서 배포본을 근사 복원한 레시피이지 공식 센서 MTF의 정답이라고 부르지 않는다. [S8]

Train/valid/RR/FR 각각 **실제 입력 PAN 배열에서 위 함수로 새 immutable cache를 생성**하고 모든 layout에서 같은 cache identity를 사용한다. 생성은 augmentation 이전이며 P/LP/MS/GT에 같은 기존 증강을 적용한다. 원본 H5를 덮어쓰지 않는다. 기존 LP 파일과의 오차도 비교 기록하되 캠페인 도중 공급원을 혼용하지 않는다. LP 재생성은 이번 신규 캠페인의 명시된 공통 전처리다.

PH에서도 L을 계산해야 H=P−L을 만들 수 있다. **LPAN을 concat하지 않는 것과 LP 계산을 하지 않는 것은 다르다.**

정규화는 `P_n=2P_DN/2047−1`, `L_n=2L_DN/2047−1`, `H_n=P_n−L_n`이다. H는 음수를 포함하므로 별도로 `−1`을 또 빼거나 clamp하지 않는다. Intermediate overshoot를 채널별로 잘라내면 위 선형성도 깨진다. 최종 출력 평가의 DN 변환·clip 정책은 기존 공식 evaluator와 동일하게 고정한다.

## 3. Fresh Teacher 학습: 5개 local pipeline

### 3.1 공통 Teacher 사양

| 항목 | 사양 |
|---|---|
| U-Net | W112, depth `[1,2,3]`, residual 출력 8채널 |
| 입력 | 서버에 지정된 P0 / PL / PH / PLH |
| Aligner | 기존 PANGlobalAligner, PAN1/MS8, margin4 |
| 초기화 | **U fresh, A fresh; 마지막 Linear의 weight/bias만 0** |
| Donor / 기존 T0 로드 | **없음** |
| Task | MS-only, PAN reconstruction OFF, mode modulation OFF, attention OFF |
| 학습 | 50,000 optimizer updates, batch48, AdamW β=(.9,.999), eps1e−8, wd=.01 |
| LR | U peak1e−4 / A peak1e−5, cosine→0, warmup100 |
| Reconstruction | 최종 HRMS와 GT의 full-frame band/pixel mean L1 |
| Consistency | 아래 L_off, λ=1e−4, 기존 반경2 pixel 원판 probe |
| α/β/q-edge | Teacher에는 없음 |

Teacher 초기화 seed는 s1–s4 **71001**, s5 **71002**다. 동일 seed 숫자만으로 다른 채널 모델의 공통 weight가 같아지지 않으므로 §6의 coupled initialization을 적용한다. 기존 초기화 파일이나 학습된 N2 A를 잘못 로드하면 preflight 실패다.

### 3.2 Teacher loss와 gradient

\[
\begin{aligned}
\mathbf c_0&=A_T(P,M),\quad P_\epsilon=\mathcal W(P,\epsilon),\quad
\mathbf c_\epsilon=A_T(P_\epsilon,M),\\
L_{\rm rec,T}&=\operatorname{mean}|Z_T-Y|,\\
L_{\rm off}&=\frac{1}{2B}\sum_i\|\mathbf c_{\epsilon,i}+\epsilon_i-\operatorname{sg}(\mathbf c_{0,i})\|_1,\\
L_T(t)&=L_{\rm rec,T}+\mathbf1[t\bmod2=1]\,10^{-4}L_{\rm off}.
\end{aligned}
\]

`t`는 0-based optimizer update index로 명시한다. 저장된 update 수는 `t+1`이다. Native reconstruction은 매 update 수행한다. ε는 반경2 HR pixel 원판의 면적 균등 분포(`radius=2√u`, `angle=2πv`)이며 별도 corruption RNG를 쓴다.

**Synthetic shifted P는 consistency A-only branch에만 들어간다. Teacher U에는 들어가지 않는다.** 따라서 이 branch를 위해 L/H를 따로 shift하거나 concat할 필요가 없다. Reconstruction branch에서만 native P/L을 c0로 함께 warp한다.

`L_rec`은 Teacher A와 U를 갱신한다. `L_off`는 Teacher A만 갱신한다. `sg(c0)`는 offset target에만 쓰며 native reconstruction의 c0를 detach하지 않는다. 위 coefficient는 과거 L1E4형 목적함수를 유지한 것이고 fresh A initialization만 새 프로토콜이다. [S1, S3]

### 3.3 Teacher reference 선택은 exact50K로 사전 고정

이번 핵심 비교에서 **KD reference는 각 Teacher의 exact50,000 checkpoint 하나**로 고정한다. 이는 Teacher마다 HQNR peak가 다른 step에서 선택되는 효과를 입력 효과와 분리하기 위한 의도적인 실험 규칙이다. 기존 T0의 best-HQNR 선택을 그대로 재현하는 것은 아니다.

Teacher 자체의 성능표에는 RAW_MAX, HQNR-gated ERGAS target, RR-val-selected, Exact50K를 모두 기록한다. 다만 그 결과를 보고 Student가 참조하는 Teacher checkpoint를 자동으로 바꾸지 않는다. 다른 Teacher 선택 전략은 후속 별도 experiment다.

Teacher가 HQNR .9585를 못 넘었다는 이유만으로 Student를 막지 않는다. 파일/데이터/forward 오류, non-finite, 체크포인트 무결성 문제만 해당 local pipeline을 중지한다. Fresh A의 shift-consistency가 나쁠 경우 현상을 기록하고, 기존 donor로 몰래 대체하지 않는다.

## 4. Teacher별 calibration과 q cache

각 서버의 T exact50K에서 A/U 전체를 eval/frozen으로 읽는다. `T_P0/T_PL/T_PH/T_PLH`는 이름만 다른 동일 자산이 아니다. **τ_R, q_ref, raw-q cache를 각 Teacher별로 새로 만든다.** s5의 반복 Teacher도 별도다.

### 4.1 τ_R

\[
e_T(i,p)=\frac18\sum_{b=1}^{8}|Z_{T,i,b}(p)-Y_{i,b}(p)|,
\qquad \tau_R=\max(\operatorname{median}_{i,p\in\mathcal C} e_T(i,p),10^{-6}).
\]

Train-only, augmentation 없는 고정 3,072 base patch, calibration index seed1234를 사용한다. Residual이 아니라 MS base를 더한 최종 HRMS의 오류다. Same Teacher 아래의 Student input/W/D가 바뀌어도 이 calibration은 재사용한다. 과거 WV3 τ_R 숫자를 복사하지 않는다.

### 4.2 q / q_ref

\[
q_{i,r}=\frac1{2K}\sum_{j=1}^K
\|A_T(\mathcal W(P_{i,r},\epsilon_j),M_{i,r})+\epsilon_j-A_T(P_{i,r},M_{i,r})\|_1,
\quad K=16.
\]

AXIS16: 반경 `.25, .5, 1, 2` HR pixel × 네 축 방향. q probe 역시 Aligner의 P/M만 사용한다. Full train의 실제 `(index, rot)` 4상태를 덮고, 기존 feeder의 고정 flip/rot 계약을 그대로 재현한다. 새로운 8-view augmentation으로 확대하지 않는다.

\[
q_{ref}=\operatorname{median}_{i\in\mathcal C,r}q_{i,r},\qquad
s_{i,r}=\operatorname{sg}\frac{q_{ref}}{q_{ref}+q_{i,r}}.
\]

분자 2 없음. Batch normalization/threshold/temperature 추가 없음. q는 finite·nonnegative, q_ref는 finite·positive여야 한다. q_ref=0이면 수식을 임의 보정하지 말고 별도 실패 사유로 기록한다.

q_ref만으로 좋은 aligner라고 판정하지 않는다. Constant A의 AXIS16 q는 이 정의에서 `.46875 px`가 될 수 있고, 자기 median으로 나누면 s≈.5가 된다. Absolute q·반경별 slope·native c·output error를 같이 기록한다. q는 native physical alignment GT 오차가 아니다.

Calibration stage는 `teacher_checkpoint_sha`, `teacher_layout`, `LP recipe/hash`, train SHA, index/augmentation SHA, τ_R/q_ref, q cache SHA를 포함한 **읽기 전용 local reference manifest**를 발행한다. 이것은 글로벌 설정 선택 lock이 아니라 Stage 2가 어떤 Teacher를 읽는지 확인하는 provenance다. 다른 서버의 manifest를 기다리지 않는다.

## 5. Student: G23 방식은 유지하고 입력과 W/D만 비교

Teacher는 자기 layout으로 forward한다. Student는 자기 layout으로 forward한다. 예를 들어 `T_PLH → S_P0`에서도 Teacher 출력을 만들 때 PLH가 유지되어야 한다.

Teacher A/U frozen. Student A는 해당 Teacher A의 **독립 clone**으로 초기화하고 trainable하게 둔다. Student U는 fresh다. Teacher U 가중치를 Student에 복사하거나 Student끼리 fine-tuning으로 연결하지 않는다.

\[
\begin{aligned}
d_T&=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\\
a_T&=\operatorname{sg}\frac{[e_S-e_T]_+}{e_S+10^{-6}},\\
\ell_{H,i}&=\operatorname{mean}_p[(1+\alpha d_T)e_S],\\
\ell_{K,i}&=\operatorname{mean}_p[\beta(1-d_T)a_T\,\operatorname{mean}_b|Z_S-Z_T|],\\
L_U&=\frac1B\sum_i[\ell_{H,i}+\ell_{K,i}+\lambda_Es_i\ell_{E,i}],\\
L_A&=\frac1B\sum_i s_i\ell_{H,i}.
\end{aligned}
\]

`α=1`, `β=.1`, `λE=.002`; U LR1e−4, A LR3e−6. β는 ℓK에 이미 포함되므로 두 번 곱하지 않는다. 모든 픽셀의 고정 분모를 사용한다. Teacher/GT/difficulty/advantage/q는 detach, live eS와 |ZS−ZT|는 유지한다.

Edge는 기존 signed Scharr x/y, 두 방향 평균, 1px interior 규약을 그대로 유지한다. 새 LP/HP를 보고 edge strength나 α를 다시 튜닝하지 않는다. Student synthetic jitter/offset loss 없음. PAN auxiliary와 stat/geometry KD 없음.

**U와 A의 parameter 집합에 각각 gradient를 계산**한다. Soft·edge를 포함한 LU를 A까지 backward하지 않는다. 반면 LA의 hard-loss gradient는 P/L/H의 사용 중인 모든 c 경로를 통해 A로 전달되어야 한다. H=P~−L~의 chain rule을 끊거나 채널 수만큼 LA를 추가해서는 안 된다. [S1 pp.12–15]

## 6. 초기화·증강·재현성 통제

### 6.1 입력 9/10/11ch의 공통 초기 함수

각 role/width/depth/seed에 대해 **fresh 9ch template**을 만들고 같은 구조의 공통 backbone tensor를 모든 layout에 복사한다. 이는 학습 가중치 전이가 아니다.

- P kernel과 8개 MS kernel, bias, 공통 body는 동일하게 복사한다.
- 추가 L/H input kernel은 **0으로 초기화**한다. 첫 학습 step부터 trainable이다.
- 실제 layout의 channel mapping을 명시해 P/L/H/MS kernel 위치를 정확히 이동한다.
- Step0 출력과 c, MS base가 P0/PL/PH/PLH에서 허용오차 안에 같음을 확인한다.

이렇게 해야 입력 채널이 늘며 무관한 body RNG까지 바뀌는 문제를 줄인다. 이는 **이번 새 실험의 coupled-init 규칙**이지 과거 아카이브의 초기화와 동일하다는 뜻이 아니다. 서로 다른 W/D는 표현 구조가 달라 step0 함수의 완전한 동일성을 요구하지 않는다. 같은 이름·shape의 fresh tensor만 대응시키고 추가 block은 별도 고정 RNG로 초기화한다.

Training data order, augmentation, corruption RNG를 model construction RNG와 분리한다. Loader/feeder를 생성할 때 전역 seed가 재설정되는 경로가 있으면 training RNG를 명시적으로 복원한다. NumPy/Python/Torch/CUDA RNG와 optimizer/scheduler/AMP scaler를 exact resume state에 포함한다.

### 6.2 LP/HP의 해석상 주의

P=L+H이므로 P를 유지한 PL와 PH는 동일한 low/high 선형 span을 표현할 수 있다. PLH는 중복 parameterization이다. 따라서 PL/PH/PLH 차이를 추가 정보량의 차이로만 설명하지 않는다. 초기화·weight decay·최적화와의 상호작용을 포함한다. 특히 새로운 extra kernel을 0으로 놓는 통제 아래서 처음부터 셋의 업데이트 궤적이 같아야 하는 것은 아니다.

### 6.3 동일 seed와 서버

s1–s4는 T71001/S72001을 대응시킨다. 그러나 CUDA/실행환경과 Teacher가 다르므로 완전 독립 반복이나 정확한 cross-server 재현으로 간주하지 않는다. s5는 **T71002/S72002의 독립 pipeline 반복**이다. s5와 s4의 차이는 Student seed만의 효과가 아니다.

주요 입력 효과의 가장 해석력 높은 비교는 **s4 동일 Teacher·동일 Student seed·동일 서버에서 네 layout을 학습한 결과**다.

## 7. 서버별 12시간 실행 순서

### 7.1 기본 및 예비 배정

| 서버 | 새 Teacher | 기본 Student 순서 — 전부 W104D121 | 남는 시간의 보조 case |
|---|---|---|---|
| s1 | P0 / W112D123 / T71001 | **P0 → PLH**, S72001 | 추가 mandatory 없음. 기존 자산/후반 E 진단과 완료 보고 우선 |
| s2 | PL / W112D123 / T71001 | **PL → P0**, S72001 | 추가 mandatory 없음. Calibration/장면별 오류 정리 |
| s3 | PH / W112D123 / T71001 | **PH → P0**, S72001 | **PH W104D122 → PH W112D121**, 같은 Teacher/S72001 |
| s4 | PLH / W112D123 / T71001 | **PLH → P0 → PL → PH**, S72001 | **PLH W104D122 → PLH W112D121**, 같은 Teacher/S72001 |
| s5 | PLH / W112D123 / T71002 | **PLH**, S72002 | **PLH W104D122**, 같은 Teacher/S72002 |

각 서버는 `PREFLIGHT → fresh Teacher → freeze exact50K + calibration → Student 순서`로 독립 진행한다. s5가 다른 서버의 Teacher가 배포될 때까지 기다리는 편성이 아니다.

**우선순위는 네 입력의 기본 대조 완성 > 필요한 공식 평가 > D122 > W112이다.** s4의 PL/PH 기본 대조를 빼고 D/W를 먼저 시작하지 않는다. s3 D122는 HP 입력에서의 depth-only 단서, s4/s5 D122는 PLH 입력에서의 단서다. D122에서 layout까지 동시에 바꾸지 않는다.

### 7.2 비교가 답하는 질문

| 비교 | 질문 | 제한 |
|---|---|---|
| s1/s2/s3/s4의 matching T→S | Teacher부터 동일 입력을 쓰는 전체 pipeline 효과 | Teacher output/A/q/τ와 서버까지 함께 달라지는 end-to-end 비교 |
| s4의 P0/PL/PH/PLH | **Teacher를 고정했을 때 Student 주파수 입력의 효과** | Student A는 각 run에서 적응; U만의 순수 고정-A 효과는 아님 |
| s1 P0↔PLH | 주파수 없는 Teacher 아래에서도 PLH가 유효한가 | 같은 local Teacher/seed의 입력 대조 |
| s2 PL↔P0, s3 PH↔P0 | 해당 주파수 Teacher 아래에서 Student에 그 입력을 주는 효과 | 입력과 Teacher 효과 분리에 보조 |
| s4 PLH↔s5 PLH | 전체 pipeline의 다른 T/S seed 반복 | 서버·Teacher seed·Student seed가 모두 달라짐 |
| 각 서버 D121↔D122 또는 W104↔W112 | 한 축의 capacity 변경 | local Teacher/input/seed는 동일; 전체 grid는 아님 |

### 7.3 정확한 Student case ID

Teacher case는 `FH12_Sx_T_<layout>_W112_D123_WV3_S<teacher_seed>_FRESH50_v1`이다. Student는 Teacher layout/seed와 Student seed를 모두 ID에 넣는다. 기존 G23/520xx나 QG26 run을 rename해 편입하지 않는다.

| 서버 | 순서 | 등급 | Student input | 구조 | S seed | Teacher input / seed | run ID |
|---|---:|---|---|---|---:|---|---|
| s1 | 30 | CORE | `P0` | W104D121 | 72001 | `P0` / 71001 | `FH12_S1_S_P0_W104_D121_WV3_TP0_TS71001_SS72001_FRESH50_v1` |
| s1 | 40 | CORE | `PLH` | W104D121 | 72001 | `P0` / 71001 | `FH12_S1_S_PLH_W104_D121_WV3_TP0_TS71001_SS72001_FRESH50_v1` |
| s2 | 30 | CORE | `PL` | W104D121 | 72001 | `PL` / 71001 | `FH12_S2_S_PL_W104_D121_WV3_TPL_TS71001_SS72001_FRESH50_v1` |
| s2 | 40 | CORE | `P0` | W104D121 | 72001 | `PL` / 71001 | `FH12_S2_S_P0_W104_D121_WV3_TPL_TS71001_SS72001_FRESH50_v1` |
| s3 | 30 | CORE | `PH` | W104D121 | 72001 | `PH` / 71001 | `FH12_S3_S_PH_W104_D121_WV3_TPH_TS71001_SS72001_FRESH50_v1` |
| s3 | 40 | CORE | `P0` | W104D121 | 72001 | `PH` / 71001 | `FH12_S3_S_P0_W104_D121_WV3_TPH_TS71001_SS72001_FRESH50_v1` |
| s3 | 50 | RESERVE_D | `PH` | W104D122 | 72001 | `PH` / 71001 | `FH12_S3_S_PH_W104_D122_WV3_TPH_TS71001_SS72001_FRESH50_v1` |
| s3 | 60 | RESERVE_W | `PH` | W112D121 | 72001 | `PH` / 71001 | `FH12_S3_S_PH_W112_D121_WV3_TPH_TS71001_SS72001_FRESH50_v1` |
| s4 | 30 | CORE | `PLH` | W104D121 | 72001 | `PLH` / 71001 | `FH12_S4_S_PLH_W104_D121_WV3_TPLH_TS71001_SS72001_FRESH50_v1` |
| s4 | 40 | CORE | `P0` | W104D121 | 72001 | `PLH` / 71001 | `FH12_S4_S_P0_W104_D121_WV3_TPLH_TS71001_SS72001_FRESH50_v1` |
| s4 | 50 | CORE | `PL` | W104D121 | 72001 | `PLH` / 71001 | `FH12_S4_S_PL_W104_D121_WV3_TPLH_TS71001_SS72001_FRESH50_v1` |
| s4 | 60 | CORE | `PH` | W104D121 | 72001 | `PLH` / 71001 | `FH12_S4_S_PH_W104_D121_WV3_TPLH_TS71001_SS72001_FRESH50_v1` |
| s4 | 70 | RESERVE_D | `PLH` | W104D122 | 72001 | `PLH` / 71001 | `FH12_S4_S_PLH_W104_D122_WV3_TPLH_TS71001_SS72001_FRESH50_v1` |
| s4 | 80 | RESERVE_W | `PLH` | W112D121 | 72001 | `PLH` / 71001 | `FH12_S4_S_PLH_W112_D121_WV3_TPLH_TS71001_SS72001_FRESH50_v1` |
| s5 | 30 | CORE | `PLH` | W104D121 | 72002 | `PLH` / 71002 | `FH12_S5_S_PLH_W104_D121_WV3_TPLH_TS71002_SS72002_FRESH50_v1` |
| s5 | 40 | RESERVE_D | `PLH` | W104D122 | 72002 | `PLH` / 71002 | `FH12_S5_S_PLH_W104_D122_WV3_TPLH_TS71002_SS72002_FRESH50_v1` |

`FH12_case_registry.csv`에는 Teacher 5개, Student 16개(기본11+예비5), preflight5, calibration5의 **총31개 row**가 있다. 이 중 학습 run은 **기본16, 예비5**다. Preparation row를 학습 seed 수에 포함하지 않는다.

## 8. 12시간 예산과 실행 중단·재개 규칙

### 8.1 기준은 wall-clock이며 새 12시간 campaign이다

이 요청의 12시간은 이번 FH12에 명시적으로 배정한 새 창이다. 각 서버의 적용 확인 시각을 `window_start_utc`로 한 번 기록하고 `deadline=start+12h`로 정한다. 재시작/재시도/다음 stage에서 시계를 리셋하지 않는다. 전 서버가 같은 시각에 준비될 때까지 기다리지 않는다.

기존 run이 실행 중이면 source/config/seed를 바꾸지 않고 정상 완료를 우선한다. 그 잔여시간은 창에 포함한다. 남은 시간이 새 기본 pipeline을 수용하지 못하면 `DEFERRED_BUDGET`로 남기고, 명령을 받았다는 이유로 진행 중인 학습을 강제 종료하지 않는다. QB/GF2 또는 미시작 MIX20H case를 뒤에 자동으로 붙이지 않는다.

### 8.2 최초 예약값 — 실측이 아닌 계획 가정

T/S 시간은 해당 stage의 50K 및 필요한 기본 평가를 포함하는 예약값이다. 코드 구현이 이미 완료되어 있다는 보장은 없다.

| 서버 | Teacher 예약(h) | Student W104D121 1개(h) | 기본 Student 수 | setup+calibration(h) | 마감 정리(h) | 기본 합계(h) |
|---|---:|---:|---:|---:|---:|---:|
| s1 | 2.6 | 2.5 | 2 | 1.25 | .75 | **9.60** |
| s2 | 3.0 | 2.6 | 2 | 1.25 | .75 | **10.20** |
| s3 | 1.7 | 1.4 | 2 | 1.25 | .75 | **6.50** |
| s4 | 1.7 | 1.4 | 4 | 1.25 | .75 | **9.30** |
| s5 | 3.2 | 3.5 | 1 | 1.25 | .75 | **8.70** |

D122는 초기 예약을 같은 서버 base의1.10배, W112D121은1.20배로 잡되, 이는 측정값이 아니다. Teacher/Student 최초 1K 정도와 실제 evaluator 2개 checkpoint를 재어 projection을 갱신한다. **새 입력·fresh Teacher의 속도는 현재 원격 GPU에서 측정하지 않았다.**

s4·s5의 D122가 반드시 창 안에 들어간다고 약속하지 않는다. 예비 admission은 다음으로 한다.

\[
T_{remaining}\ge .75h+1.15\,[\widehat T_{next}+\widehat T_{unfinished\ core}+\widehat T_{required\ eval}].
\]

이미 next에 포함한 eval을 다시 더하지 않도록 ledger에서 학습/평가 항목을 구분한다. 기본 대조의 completion reserve를 소모하면서 예비 W를 시작하지 않는다. 시간 부족 때문에 그 run만30K로 줄이거나 batch를 바꾸지 않는다.

### 8.3 마감과 실패

- 정상 run은50K를 완주한다. 최초 admission을 보수적으로 해 deadline 초과를 피한다.
- Unexpected slowdown으로 마감에 미달하면 안전 경계에서 full-state를 저장하고 `PAUSED_DEADLINE(actual_update)`로 종료한다. 50K 결과로 올리지 않는다.
- Upload 실패는 local metrics를 보존하고 업로드만 재시도한다. 학습을 다시 돌리지 않는다.
- Numerical/LP identity/gradient 실패는 해당 case 실패로 남기고 구현을 고친 새 revision으로 진행한다. 성공한 다른 서버는 대기하지 않는다.
- 결과가 낮다는 이유로 임의의 donor/τ/q/seed/loss에 fallback하지 않는다.

## 9. 체크포인트와 ERGAS 평가: 최고 HQNR 행만 보지 않는다

### 9.1 고정 후보 격자

모든 Teacher/Student에서 저장 후보는 `{1010*k, k=1..49} ∪ {50000}`의 50개로 통일한다. **Epoch 간격을 복사하지 않고 optimizer update로 관리**한다. Legacy teacher template의 `eval_epoch=10`을 그대로 쓰면 이 격자와 다를 수 있다.

Native A_ON, RR full256/FR full512, MS/GT는 원 M-frame, FR reference PAN은 원 native PAN을 사용한다. Aligned-PAN 기준이나 V64를 raw HQNR 하한 대신 사용하지 않는다. 데이터는 현재 검증된 WV3 mat20 protocol을 경로·hash·장면 순서로 확정하며, 과거 H5 12–19 subset으로 돌아가지 않는다.

### 9.2 각 run의 필수 보고 네 종류

| 선택 ID | 정의 | 목적 |
|---|---|---|
| RAW_MAX | raw HQNR 최대 candidate | 기존 본 열의 의미 유지 |
| **TARGET** | H≥.9585 중 **공식 RR ERGAS 최소**, tie SCC→PSNR→낮은 step | **이번 최우선 결과** |
| EXACT50K | 정확한 50000 | 동일 학습량 비교 |
| RR_VAL_SELECTED | candidate별 validation ERGAS 최소점 | “ERGAS로 뽑으면 달라지는가” 확인 |

각 선택점에서 H·E·SCC·PSNR뿐 아니라 D_lambda/D_s/SAM/Q8/SSIM을 같은 checkpoint로 평가한다. Target 적격이 없으면 `no_eligible`, Target 수치는 빈칸, RAW_MAX/Exact/Val-selected는 보존한다. Teacher reference는 이 표의 Target으로 자동 변경하지 않고 §3.3의 exact50K다.

RAW_H grid가 완전해야 Target을 인증할 수 있다. 적격 candidate의 공식 RR 평가도 전부 완료해야 `Target official=true`를 쓴다. 몇 개만 평가한 뒤 min을 전체 최저라고 쓰지 않는다.

### 9.3 공식 RR의 전체 최저점 진단

시간이 허용되면 기존 50개 후보에 공식 RR을 모두 계산해 `E_MIN_DIAG50`과 그 checkpoint의 H를 기록한다. 이 값은 **RR test를 사용한 개발용 oracle 진단**이며 validation 선택값/독립 최종 성능과 구분한다.

모든 50개를 계산하지 못하면 `E_MIN_OBSERVED(n/50)`로 표시한다. Required Target/Exact/Val-selected와 기본 입력 대조를 먼저 완결하고, s1/s2의 남는 시간을 이 진단에 우선 쓴다. 12시간에는 최소한 E-val-selected와 Exact50K를 확보하므로, best-H만 보고 architecture를 판단하는 문제는 줄일 수 있다. E-val-selected가 공식 test-E 최소점과 같다는 보장은 없다.

### 9.4 목표 판정

같은 선택점의 raw H≥.9585 이후 E를 최소화한다. E<2.040이면 공동목표다. H .9586/E2.039가 H .9605/E2.070보다 우선이다. 두 metric의 서로 다른 checkpoint 값을 합치지 않는다. Paper-reported PAN-Crafter WV3 H=.958/E=2.040을 기준으로, 기존 더 엄격한 H gate .9585를 유지한다. [S2 Table1]

Test-aware Target/튜닝이 포함된 개발 실험이다. 다른 seed의 반복만으로 미사용 독립 test 평가가 되는 것은 아니다.

## 10. Sheet 업로드와 산출물

새 탭을 임의 생성하거나 과거행을 overwrite하지 않는다. `WV3-s1`, `WV3-s2`, `WV3-s3(5090)`, `WV3-s4`, `WV3-s5`의 기존 RR/FR/Cost/Target/Exact 열을 사용하고 **campaign=FH12, Role=T/S**를 명시한다.

- 본 RR/FR: RAW_MAX 같은 checkpoint. 비교 주 지표는 Target.
- Target: H/E/SCC/PSNR/SAM/SSIM/Q8 및 D_lambda/D_s, step, SHA, n_eligible, complete status.
- Exact50K 및 RR_VAL_SELECTED: 각자 같은 checkpoint의 H/E와 step/SHA. RR_VAL_SELECTED용 열이 없으면 오른쪽에 별도 그룹 추가.
- `FH12 teacher input`, `student input`, `W`, `D`, `teacher seed`, `student seed`, `teacher_ref_sha`, `tau_R`, `q_ref`, `LP recipe/hash`, `init policy`, `training/evaluator release`를 기록한다.
- Params/FLOPs/추론시간은 실제 **전체 Student A+frontend+U**, RR256 조건으로 다시 측정한다. HP는 L 계산 비용을 포함한다. Teacher 학습/calibration은 별도 시간.
- Header 이름과 schema로 열을 찾는다. QG26/Q4 열을 잘못 가져오지 않는다. WV3는 Q8이다.
- `(campaign,run_id)`로 upsert하며 Teacher와 Student, 같은 seed의 다른 layout을 별도 행으로 유지한다. 업로드 후 readback을 검증한다. 실패하면 업로드만 재시도한다.
- 새 Student 전부에 NOA 재평가를 붙이는 것은 이번 mandatory가 아니다. 이전 NOA 열은 빈칸으로 둔다.

Local results:

```text
work_dir/_fh12/<server>/
  takeover_manifest.json
  effective_queue.json
  window.json
  dataset_manifest.json
  lpan_manifest.json
  preflight_report.json
  references/<teacher_case>/reference_manifest.json
  references/<teacher_case>/calibration.json
  references/<teacher_case>/q_cache.npz
work_dir/<run_id>/
  meta/config.resolved.yaml
  meta/training_start_manifest.json
  init_manifest.json
  candidates/<step>/...
  candidate_identity.jsonl
  official/raw_grid.json
  official/target_selection.json
  official/exact50k.json
  official/rr_val_selected.json
  official/e_min_diag.json
  diagnostics/frequency_shift.csv
  diagnostics/loss_routing.json
```

각 artifact에는 실제 resolved config와 input layout을 넣는다. Filename만 같은 파일을 동일 Teacher로 취급하지 않는다.

## 11. 실행 전 필수 검사 — 대규모 학습보다 먼저

| 검사 | 완료 조건 |
|---|---|
| Data/LP identity | 실제 native PAN에서 생성했는지 확인한다. 4배 해상도 관계, 표본 순서, 새 cache SHA, phase recipe를 기록하며 원본은 보존한다. |
| Aligner input | PAN1/MS8만 받고 margin4를 적용한다. 고정된 A에서 LP/HP만 바꾸면 c는 변하지 않아야 한다. |
| Layout | P0/PL/PH/PLH의 9/10/10/11채널과 정확한 채널 순서를 확인한다. |
| Zero-shift | 기존 low/high concat과 허용오차 내에서 일치하며 MS base는 한 번만 더한다. |
| Shared-grid | `W(P−L,c)`와 `W(P,c)−W(L,c)`의 forward 및 c-gradient가 일치한다. |
| Synthetic shape | train64/RR256/FR512를 시험한다. dy/dx 순서, 부호, 경계 조건을 확인한다. |
| Coupled init | 같은 W/D/seed의 P/MS/body/A hash가 일치한다. 추가 kernel은 0이며 step0 예측이 일치한다. |
| Teacher scratch | Donor/T0의 pretrained state 로드가 0회다. A의 최종 head만 zero이며 A/U는 모두 trainable이다. |
| Consistency | Shifted branch의 U 호출이 0회다. Offset target의 c0는 detach하지만 reconstruction의 c0는 live다. |
| Student routing | LU→U, LA→A로 분리한다. Soft/edge/offset은 A에 직접 기여하지 않는다. |
| Separate Teacher/S layout | Bridge case에서도 Teacher는 자기 layout을 유지하고 eval state/hash가 변하지 않는다. |
| New calibration | Source Teacher의 exact50K SHA가 일치한다. 옛 τ/q/T0 경로를 재사용하지 않는다. |
| Evaluation | Raw-original mat20과 공식 RR protocol을 유지한다. 50K와 best step, Target의 공식 완료 여부를 구분한다. |
| Resume | Optimizer/scheduler/scaler/RNG/sample order/layout을 검증한다. 중간 seed/구조 변경은 금지한다. |

제공한 `sync_frontend.py --self-test`는 CPU의 synthetic 64/256/512 입력에서 실제로 수행했다. 동기 분해와 HP 직접 warp의 최대 오차는 **4.77e−7**, c-gradient의 최대 차이는 **1.49e−8**이었다. **이는 보간식의 구현 검사이지 WV3 성능 검증이나 전체 trainer 연결 완료가 아니다.** JSON에 실제 결과를 첨부했다. 최종 사용하는 Torch/CUDA에서도 검사를 다시 수행한다.

c≈0에서만 맞는 검사로 끝내지 않고 양/음 subpixel 및 1pixel 이상의 shift를 포함한다. GPU에서는 precision과 backward 재현성의 영향을 기록하며, 다른 장비에서 bitwise 동일 결과를 보장하지 않는다.

## 12. 구현·등록 완료의 정의

현재 저장소의 기존 코드를 그대로 실행하는 것으로는 본 계획이 완성되지 않는다. 필요한 변경은 다음이다.

1. 명시적인 `input_layout`과 실제 stem의 9/10/10/11채널 지원을 추가한다. `x_in` 사용 시 중복 upsample과 이전 API의 LP fallback을 금지한다.
2. `PAModel`에 동기화된 P/L frontend를 연결한다. Teacher와 Student의 layout을 독립적으로 유지하며 모든 학습·평가·프로파일 경로에 같은 구현을 사용한다.
3. Fresh Teacher protocol을 별도로 등록한다. 기존 `I-AEQ`의 donor 가정, expected SHA, fixed-reference-from-donor 설정을 새 case에 남기지 않는다. `w_off`라는 과거 PAN auxiliary 계수와 `offset_weight=1e−4`를 구분한다.
4. Calibration/cue 생성기가 명시적인 Teacher 자산을 받도록 수정한다. 공용 `T0_run`을 조용한 기본값으로 사용하지 않는다.
5. FH12 registry를 실제 학습 YAML과 effective runner queue까지 연결한다. 예비 case를 기본 mandatory 잔여 작업에 넣지 않는다.
6. Target/Exact/RR-val-selected의 공식 평가와 header 기반 업로드를 연결한다. Teacher별 calibration/cache를 분리한다.

제공한 CSV/JSON은 **구체적인 설계 case registry**이며, 기존 runner가 곧바로 읽는다고 보장하지 않는다. `runtime_state=PLANNED_NOT_REGISTERED`로 표시했다. 원격 등록이 끝났을 때만 각 서버는 아래를 보고해야 한다.

```text
server / campaign / local release / spec hash / window_start / deadline
next teacher run ID / next two student run IDs
actual YAML paths / effective queue path / registry revision
PREFLIGHT PASS / forward layout / teacher from_scratch=true
target selector / RR+FR protocol / no_global_lock=true
```

`paper/released` 문자열 하나만 바꾼 YAML, case ID만 있는 TXT, 실행하지 않은 계획 MD를 “등록 완료”로 처리하지 않는다. 반대로 성능이 낮다는 이유로 다음 서버를 막는 global lock은 없다.

## 13. 12시간 후 분석과 다음 결정

먼저 각 case의 실행 정의, 실제 update 수, Teacher/LP/reference identity를 확인한다. 이어 **s4의 고정 Teacher 네 조건**에서 PL/PH/PLH가 P0보다 HQNR 적격 checkpoint를 더 확보하는지, 그 안에서 ERGAS가 낮아지는지 분석한다.

Exact50K와 RR-val-selected의 ERGAS를 함께 보고 **복원 능력의 개선과 checkpoint 선택 시점 차이**를 구분한다. D_lambda/D_s, 밴드별 MSE/GT mean², 장면별 대응 차이를 기록한다. 여러 crop을 독립 scene처럼 세지 않는다.

s1/s2/s3의 bridge 및 matching Teacher 결과는 Teacher 경로의 영향을 해석하는 보조 근거다. 다섯 pipeline의 평균을 단일 설정의 seed 평균으로 쓰지 않는다. 입력 이득을 확인한 다음 같은 입력에서 D122/width의 효과를 읽는다. D와 W를 동시에 바꾼 결과로 각 축의 효과를 주장하지 않는다.

| 결과 | 해석 및 다음 조치 |
|---|---|
| PL/PH/PLH에서 H가 올라가고 E는 비슷하다 | 주파수 입력의 FR 기여로 기록한다. 이후 capacity/시간축을 확인한다. |
| H는 올라가지만 E가 나빠진다 | 두 목표의 상충 관계로 기록한다. PAN auxiliary를 자동으로 추가하지 않는다. |
| D122에서 E<2.040이지만 H 하한은 미달한다 | RR capacity의 단서이지 공동목표 성공은 아니다. |
| Fresh Teacher부터 결과가 좋지 않다 | 새 초기화 protocol과 L/H 조건의 영향을 분리한다. 진행 중인 case의 donor를 교체하지 않는다. |
| 한 seed에서만 근소한 차이가 있다 | 우선 가설로 남긴다. 이후 같은 Teacher의 새 Student seed 또는 새 Teacher seed block을 확장한다. |

## 14. 제공 파일과 근거

- 본 MD: 방법·case·시간·결과선택·구현인계.
- `FH12_case_registry.csv`: 총 31행(기본 학습 16 + 예비 학습 5 + 준비 10).
- `FH12_queue_s1.csv` … `FH12_queue_s5.csv`: 서버별 고정 순서와 local dependency.
- `FH12_case_manifest.json`: 기계 판독 가능한 계획 상수·layout·loss·seed·candidate grid.
- `sync_frontend.py`: 명시적인 shared-grid 참조 구현 및 독립 CPU 검사. Aligner/Trainer/Student runner는 아니다.
- `synthetic_sync_test.json`: 이번 환경에서 실행한 synthetic 결과.

### 근거와 제안의 구분

[S1] 프로젝트 `Pansharpening_research.pdf`, pp.8–15: native PAN Teacher reconstruction, relative-shift A-only consistency, frozen Teacher Student fitting.

[S2] PAN-Crafter, arXiv:2505.23367v2, Eq.(3)–(4) pp.3–4, Table 1 p.6. 논문의 PAN reconstruction과 이번 concat ablation은 별개다.

[S3] `config/PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1.yaml`: 기존 Teacher의 N2 donor / offset1e−4 / U1e−4 / A1e−5 정의. 이번에는 그 초기화를 변경한다.
https://github.com/hojunking/PAN-Crafter-repro/blob/main/config/PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1.yaml

[S4] `pa/aligner.py`, blob `e9dd406ae7f0a2dc491273e21b38f8e450dfa28d`: PAN1/MS8, GN+GAP, 마지막 Linear zero initialization.
https://github.com/hojunking/PAN-Crafter-repro/blob/main/pa/aligner.py

[S5] `pa/warp.py`, blob `7dc6b3bb83fa019ed5f6331e02ef5d5032b7a966`: (dy,dx),bicubic,border,align_cornersFalse.
https://github.com/hojunking/PAN-Crafter-repro/blob/main/pa/warp.py

[S6] PyTorch official `grid_sample` / `interpolate` documentation. 설치한 runtime version은 별도 기록하며, 최신 문서에 맞춰 런타임을 임의로 upgrade하지 않는다.
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

[S7] `pa/model.py`, blob `cffe4f6ff9caa93877afa6a2a4d61d6d7a37d4d3`: 기존 P만 warp하고 LP는 그대로 backbone에 전달한다.
https://github.com/hojunking/PAN-Crafter-repro/blob/main/pa/model.py

[S8] `tools/repair_lpan.py`, blob `d22ebef2dd3e693748a4876b814cb84214d3b04a`: Gaussian1.98/kernel41/replicate/offset2. 손상된 FR LP에 대한 과거 기록은 현재 각 서버의 실제 파일 검증을 대신하지 않는다.
https://github.com/hojunking/PAN-Crafter-repro/blob/main/tools/repair_lpan.py

이 문서의 shared-shift설계·scratch Teacher 선택·exact50K reference·case 배정·시간 예약은 **이번 새 실험 제안**이다. 논문이나 과거 코드가 그 설계를 이미 검증했다는 주장이 아니다. 원격 GPU training, 큐 기동, Sheet 업로드는 이번 파일 작성에서 수행하지 않았다.
