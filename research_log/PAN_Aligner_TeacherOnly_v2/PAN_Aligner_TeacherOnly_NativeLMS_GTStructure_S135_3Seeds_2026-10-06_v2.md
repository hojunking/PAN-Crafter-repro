# PAN Aligner Teacher-only 실효성·개선 실험 v2
## Native LMS 입력 · ε consistency + 직접 GT 구조 정합 · s1/s3/s5 · 모든 case 3회 반복

- 작성일: **2026-10-06**
- Campaign: `PANDA_ALIGNER_TEACHER_LMS_GT_S135_20261006_v2`
- 결과 확인: `pan-cvpr27` → **`analysis`**, sheet ID `261006100`
- 이전 계획: `PANDA_ALIGNER_CAUSAL_20261006_v1`의 새 실행 계획을 **대체**한다. 기존 기록·완료 실험·참고 보고서는 삭제하지 않는다.
- 이번 산출물: 실험 명세, 52조건 registry, 468개 실행 정의, 분석 업로드 계약. **원격 코드 배포·학습 기동·서버 중단을 수행했다는 기록이 아니다.**

---

## 0. 확정 범위와 바로 실행할 순서

| 항목 | 이번 결정 |
|---|---|
| 모델 | **Teacher 크기의 PAN Aligner + U-Net 한 벌**. Student·KD·q/e fitting 없음 |
| 서버 | **s1=WV3, s3=QB, s5=GF2**. 이번 실험을 위한 배정이며 실제 현재 작업을 추정한 표가 아님 |
| 새 주력 A 입력 | **원 PAN + 데이터셋의 native LMS**. LMS는 PAN-sharpened 영상이 아님 |
| 핵심 신호 | native reconstruction + **ε consistency** + **warped PAN↔GT의 직접 구조 정합 loss** |
| 대조 | 무정합 재학습, zero warp, 데이터셋 공통 shift, **영상마다 사전에 구한 고정 shift**, 학습 가능한 전역 shift, shuffled shift |
| 반복 | **52조건 × 데이터셋별 3개 seed × 3개 데이터셋 = 468개 학습** |
| 학습 길이 | 48조건은 fresh50K, 별도 학습 길이 대조 4조건은 fresh100K |
| 종료 | 전체 wall-clock 제한 없음. 등록된 각 조건의 R1/R2/R3를 완결한다. 성능이 좋아질 때까지 seed를 다시 뽑는 무한 반복은 하지 않는다 |
| 주 품질 선택 | **HQNR_MAX50**. ERGAS로 선택하지 않는다 |
| 원인 비교 | **EXACT_FINAL**과 고정 progress checkpoint를 함께 사용. 최고점 선택 효과와 분리 |
| 업로드 | 계획/실측/집계 분리, scalar·반경별 응답·gradient·대응 차이·3회 변동을 `analysis`에 숫자로 올림 |

**기동 순서:** 활성 프로세스·데이터 확인 → 안전 전환 → LMS/좌표/gradient 단위검사 → P0_CORE의 모든 3회 → 나머지 사전 등록된 block의 모든 3회 → 종료 보고. s1/s3/s5는 독립 진행하고 다른 서버 결과를 기다리지 않는다. s2/s4는 이번 계획의 중단·기동·추가 작업 대상이 아니다.

### 0.1 이번 변경이 해결하려는 것과 아직 보장하지 않는 것

첨부 보고서에서는 (a) 학습 크기부터 ε 반응이 약하고, (b) bicubic(MS)에는 GT 대비 약 반 픽셀 위상이 있지만 native LMS는 거의 0이고, (c) ε 차분은 절대 위치 bias를 정하지 못함을 확인했다.[S1 §4.3, §6]

따라서 **입력 기준 좌표를 native LMS로 맞추고, 실제 GT 구조와의 직접적인 미분 가능 정합 신호를 A에 추가**한다. 다만 GT는 HRMS 영상 정답이지 변위 정답이 아니다. 이 실험에서 얻는 것은 `GT-referenced structural alignment`이며, 물리적인 센서 displacement GT를 새로 확보했다는 뜻은 아니다.

기존 Teacher도 reconstruction loss의 gradient가 A로 전달되었다. 따라서 “기존 A에는 GT 정보가 전혀 없었다”는 해석은 정확하지 않다. 이번 추가는 **U를 통해 간접적으로 받던 GT 지도 외에, U와 무관하게 warped PAN 구조를 GT에 맞추는 직접 경로**다.

---

## 1. 네트워크·데이터·변위 단위

### 1.1 모델 고정

- Teacher U-Net: **P0 / W112 / depth[1,2,3]**, attention-free, LN, mode modulation off, 단일 HRMS 출력. 기존 Teacher 구현의 down/up/skip/residual 구조를 유지한다.
- PANGlobalAligner: 기존 dual stem, stride-2 convolution 두 번, residual, GAP, 64→32→2 head. normalization·마지막 head zero 초기화·global 2-vector 출력 유지.[S1 §3.3]
- Aligner 입력 채널: PAN 1 + MS 밴드 8(WV3) 또는 4(QB/GF2). 단, PAN stem/MS stem은 기존처럼 분리한다. 추가 네트워크나 correlation-volume/flow head를 만들지 않는다.
- 무정합/고정 shift 대조는 **같은 U**를 사용하고 A-CNN만 제거한다. 전역 학습 shift 대조 B07만 별도 2변수를 가진다.
- Teacher-only는 역할 명칭이다. 이번 실험에서는 다른 Teacher를 참조하거나 Student를 학습하지 않는다. **모든 학습 run은 fresh U/A**로 시작하며 기존 donor 연장학습은 하지 않는다.
- LPAN/HPAN, PAN reconstruction(MARs), Teacher distillation, q-weighting, PANMIX, 임의 post-shift 배율은 추가하지 않는다.

### 1.2 입력을 세 개로 구분한다

\[
M=\mathrm{LRMS},\quad B=\mathcal I_{\mathrm{bic}}(M),\quad L=\mathrm{LMS}_{\mathrm{dataset}}.
\]

`M`은 원본 저해상도 MS이고, 현재 bicubic이 만드는 것은 LRMS 그 자체가 아니라 **PAN 크기의 upsampled MS 참조 B**다. 이번 주력 변경은 다음이다.

\[
c=A_\phi(P,L),\qquad P_c=W(P,c).
\]

PAN을 사용해 LMS를 합성하거나 PAN detail을 LMS에 주입하지 않는다. **A의 두 입력은 P와 L**이다. 같은 PAN을 reference에도 섞으면 자기 신호와 정합하는 지름길이 생길 수 있다.

기본 인과 비교에서는 U의 입력과 residual base를 기존 B로 고정한다.

\[
\hat Y=B+U_\theta([P_c,B]).
\]

A만 L로 바꾸는 효과와 U까지 바꾸는 효과를 혼합하지 않기 위해서다. 다만 이 상태에서는 U base의 기존 위상이 남으므로, **U00–U03에서는 U의 입력과 base도 L로 바꾼다.**

\[
\hat Y=L+U_\theta([P_c,L]).
\]

U03이 사용자 의도에 가장 가까운 **A와 U 모두 native LMS 좌표 표현을 사용하는 통합 후보**다. M07은 A 참조·loss의 인과효과를 분리하는 중심 대조다. 출력 base를 바꾸면서 구조·채널 수·feature 폭을 함께 바꾸지 않는다.

### 1.3 Native LMS 제공·생성 계약 — 가장 먼저 검사

1. 가능하면 실제 train/val/RR/FR H5의 `lms`를 직접 읽는다. `ms`, `lms`, `gt`의 키·형태·밴드 순서·DN 범위를 manifest에 기록한다.
2. L shape은 `[B,C,H,W]`, P는 `[B,1,H,W]`, M은 `[B,C,H/4,W/4]`여야 한다. GT는 학습/validation/RR에만 존재한다.
3. 코드 `align/resample.py`에는 dataset LMS 재현용 `interp23tap` 이식과 phase/padding 정의가 존재한다.[S2] **이것이 현재 모든 센서·수정 데이터 파일에도 자동으로 맞는다고 가정하지 않는다.** 우선 저장 LMS와 실제 재생성값의 차이를 측정한다.
4. L이 없을 때만 검증된 **동일 dataset 연산자**로 M에서 재생성한다. source code hash·filter coefficients·padding·zero-stuff phase·dtype·scale·crop order를 고정한다. bicubic fallback이나 “(-0.5,-0.5)를 대충 적용한 보정본”은 금지한다.
5. `interp23tap` 방식은 coarse LR image에 geometry augmentation한 뒤 다시 upsample하면 native LMS와 위상이 달라질 수 있다.[S2] **원래 좌표에서 L을 만들거나 읽은 뒤, P/L/GT에 같은 HR flip·rotation·crop을 적용**한다. M에는 대응 LR 연산을 적용하고 U에서 쓰는 B의 생성 순서는 기존 기준으로 고정한다.
6. 소스 LR crop만으로 L을 재생성할 때는 필터 halo·global phase가 필요하다. 큰 원영상의 LMS를 만든 후 crop하는 것과 작은 patch에 circular padding을 적용하는 것은 다를 수 있다. stored patch LMS를 우선하며, full-scene/padding lineage가 다르면 독립 reference revision으로 기록한다.
7. QB는 기존 사용된 `msfix`와 L이 대응하는지 별도로 검사한다. 다른 MS 파일의 L을 붙이지 않는다. 맞는 L이 없으면 해당 branch를 `BLOCKED_LMS_LINEAGE`로 두고, native-LMS 원인 실험을 B로 몰래 대체하지 않는다.
8. **동일 밴드 L↔GT, B↔GT**의 registration phase를 train/val/RR에서 독립 측정한다. 보고서의 약0/반픽셀 결과를 새 데이터에서 확인하되 이를 FR의 기하 GT로 확대하지 않는다. FR은 L↔재생성 L의 연산 일치만 확인할 수 있고 GT alignment 인증은 불가하다.
9. tiny numeric error 허용치는 dtype/DN과 단위검사로 정하고 `max_abs`, `RMSE`, band-wise NCC, phase를 모두 올린다. 일치 실패를 높은 NCC 한 값으로 덮지 않는다.

### 1.4 좌표 부호

\[
W(P,c)[y,x]=P[y+c_y,x+c_x],\qquad c=(dy,dx)\text{ in current PAN pixels}.
\]

양의 dx는 오른쪽 source를 읽으므로 내용은 왼쪽으로 이동한다. A 입력 크기가 바뀌어도 1은 현재 PAN 1픽셀이다. 64→512 배열 크기 비율8을 곱하지 않는다. 실제 스케일 변화와 넓은 FOV 변화는 별도 검정한다.[S1 §3]

Native LMS를 생성하는 연산자와 **PAN을 warp하는 sampler는 별개**다. 기존 `pa/warp.py`의 bicubic warp는 유지한다. MS reference의 불필요한 bicubic upsampling 위상을 제거한다는 이유로 PAN sampler까지 nearest로 바꾸지 않는다.

---

## 2. 직접 구조 정합 loss의 실행 가능한 정의

### 2.1 기존 reconstruction과 ε consistency

\[
L_{rec}=\operatorname{mean}|\hat Y-Y|,
\]

\[
P_\epsilon=W(P,\epsilon),\quad c_0=A(P,R_A),\quad c_\epsilon=A(P_\epsilon,R_A),
\]

\[
L_\epsilon=\operatorname{mean}|c_\epsilon+\epsilon-\operatorname{sg}(c_0)|.
\]

`R_A`는 해당 case의 B 또는 L. 기존처럼 Pε 생성값은 detach하고, R_A도 fixed observation이다. ε 반지름2, 면적균등 원판, native P/GT/MS는 고정한다. 주력 계수는 `λε=0.001`, 매 update 적용이다. 이는 **새 실험값**이며 기존 donor가 이미 그렇게 학습됐다는 뜻이 아니다.

### 2.2 주 loss: warped PAN ↔ HRMS GT의 multi-scale gradient correlation

다음은 **이번에 새로 제안하는 설계**이며, 기존 보고서의 진단 estimator 자체를 학습 loss라고 재명명하는 것이 아니다. hard argmax로 shift label을 구해 맞추는 대신, 실제 warp된 영상 구조의 불일치가 c로 미분되도록 한다.

PAN/GT를 DN/maxDN의 `[0,1]` intensity scale로 놓고, clip 없이 다음 feature를 계산한다.

\[
g_{P,\sigma}=\nabla_{Scharr}(G_\sigma * W(P,c_0)),\qquad
 g_{Y,b,\sigma}=\nabla_{Scharr}(G_\sigma*Y_b),
\quad\sigma\in\{0.8,1.6\}.
\]

- Gaussian kernel size는 `2*ceil(3σ)+1`, normalized separable filter, reflect padding. fixed support 안에서만 비교한다.
- 각 dx/dy map에서 공간 평균을 각각 빼고 두 map을 vector로 연결한다. 이를 \(\tilde g\)라 한다.
- band/scale correlation:

\[
\rho_{b,\sigma}=\frac{\langle\tilde g_{P,\sigma},\tilde g_{Y,b,\sigma}\rangle}
{\sqrt{\|\tilde g_{P,\sigma}\|^2+10^{-12}}
 \sqrt{\|\tilde g_{Y,b,\sigma}\|^2+10^{-12}}}.
\]

\[
L_{str}^{GT}=\frac1{|\Sigma|}\sum_{\sigma\in\Sigma}
\sum_b\omega_{b,\sigma}\left(1-\rho_{b,\sigma}^2\right).
\]

각 sample의 target GT gradient RMS≥`1e-4`인 band/scale만 `valid`로 하고, 기본 ω는 그 안에서 동일 가중치다. **valid 판정은 target만 보고 detach**한다. 모든 band가 low-texture면 그 sample의 구조항을0으로 두되 batch에서 지우지 않고 `low_texture_fraction`, `valid_band_count`를 보고한다. raw loss는 sample 평균이며 valid sample 수에 따라 자동 loss 배율을 바꾸지 않는다.

제곱 correlation은 PAN/MS의 contrast polarity 차이를 허용하는 설계지만, 반복 패턴이나 잘못된 대응을 제거해 주지는 않는다. `raw ρ`와 `ρ²`를 모두 기록한다. 진단에는 학습 descriptor와 다른 intensity-NCC estimator도 사용한다.

**이 loss는 U의 출력 edge와 GT edge를 맞추는 loss가 아니다.** 계산은 `W(P,c)`와 GT 사이에서 끝나므로 직접 gradient는 A에만 들어간다.

\[
\nabla_\theta L_{str}=0,\qquad \nabla_\phi L_{str}\ne0.
\]

기존 Student의 `λ_E=0.002`와 혼동하지 않도록 이름을 **`lambda_struct`**로 고정한다. 이번 기준값은 **0.01**이고 별도 격자에서0.001/0.01/0.1을 비교한다.

### 2.3 기본 Teacher 목적함수

\[
L_T=L_{rec}+\lambda_\epsilon L_\epsilon+\lambda_{str}L_{str}^{GT}.
\]

기본 gradient 경로:

- **U ← native reconstruction만.**
- **A ← native reconstruction + ε consistency + 직접 PAN–GT 구조 정합.**

S06만 A에 reconstruction gradient를 전달하지 않는다. 이 경우 U loss는 예측된 c를 detach한 forward로 계산하고, A는 ε와 구조 loss로 학습한다. U 자체를 다른 모델로 바꾸지 않는다.

첫 fc2가 zero 초기화되어 초기 step에서 body gradient가0일 수 있다. 이를 즉시 버그로 판단하지 않고 **fc2 및 body를 따로**, step0/1/100 이후 추적한다. 실제 총 weighted-gradient와 parameter update도 보고한다.

### 2.4 구조 loss 변형의 정확한 범위

- **S00:** target만 L로 변경; 같은 σ·descriptor 유지. GT direct supervision과 LMS self-supervision을 분리한다. blur 차이가 남을 수 있음을 해석한다.
- **S01:** `0.5 Lstr(GT)+0.5 Lstr(L)`; total coefficient는 그대로0.01이다.
- **S02:** `1-(gP·gY)^2/((||gP||²+η²)(||gY||²+η²))`, η=`1e-3`, target-edge-weighted pixel 평균, 두σ 평균. 이는 제안된 normalized-gradient 방향 대조다. 기본 NCC와 숫자 크기·gradient가 같다고 가정하지 않는다.
- **S03:** σ=0.8 하나만 사용. 두 scale을 더하던 합을 남겨 loss를 반으로 만드는 것이 아니라 항상 scale 평균이다.
- **S04:** 학습 CAL set의 PAN/LMS gradient correlation에서 얻은 band별 median absolute correlation을 정규화해 **dataset-global frozen weights**로 사용. test·GT 정합 결과·seed별 winner로 가중치를 고르지 않는다. low-texture invalid band는 재정규화한다.
- **S05:** 구조항을 `0.5 Lstr(W(P,c0),Y)+0.5 Lstr(W(P,ε+cε),Y)`로 바꾼다. corrected synthetic branch는 원본 P에서 total shift로 **한 번** warp한다. U는 여전히 native reconstruction만 수행한다.
- **S07:** `c*_GT`를 train P↔GT 사전 estimator로 계산해 `Huber(c0-c*_GT, delta=0.25)`를 구조항 대신 사용. unit은pixel이며 descriptor loss와 같은 계수를 썼다고 동등한 강도는 아니다. label confidence·회귀 gradient를 별도로 기록한다. **pseudo shift는 물리 변위 GT가 아니다.**
- **O04:** ε loss의 `sg(c0)`만 제거. GT 구조항은 그대로 두어 absolute anchoring이 있는 상황에서 one-way/symmetric consistency를 비교한다.

### 2.5 support·low-texture·큰 이동 안전장치

- 학습 구조항은 주력64 patch의 **고정 16px 내부 영역**을 사용한다. loss 판정 영역과 공식 평가 영역은 다르다.
- 모든 condition에서 native/shifted 구조 비교의 동일 고정 영역을 유지한다. 모델이 큰 shift를 내어 유효 영역을 줄이고 loss를 회피하게 두지 않는다.
- 실제 sampler footprint+Gaussian+Scharr의 원본 support를 검증한다. fixed support가 유효하지 않은 예측은 `STRUCT_SUPPORT_VIOLATION`으로 기록하고 상태를 보존한다. mask를 전부0으로 만들어 학습을 계속하지 않는다.
- crop48/56 대조도 margin16을 우선 사용하되 사전 단위검사에서 충분한 sample support를 확인한다. 확대된 보조 영상은 기준 영역을 동일 geometry로 사상한다.
- 출력 shift에 임의 clamp, ×4, GT 기반 clip을 추가하지 않는다. nan/inf·좌표 폭주·support 위반 등 안전 실패는 같은 seed attempt로 기록하고 신호를 무시하지 않는다. 단순히 HQNR가 낮다는 이유로 run을 중지하지 않는다.

---

## 3. PAN Aligner 없는 비교를 제대로 정의한다

### 3.1 세 가지 ‘상수’를 구분한다

| 용어 | 실제 정의 | Case |
|---|---|---|
| 데이터셋 공통 고정 shift | train에서 얻은 c들을 중앙값으로 집계한 한 벡터를 모든 영상에 적용 | B02 |
| **영상별 고정 shift** | **각 영상마다** PAN–LMS 오차를 사전 추정한 `c_i`를 그 영상 전체에 균일 적용; U 학습 중 이 값은 바뀌지 않음 | **B03/B04/B05** |
| 학습 가능한 전역 shift | CNN 없이2개 scalar를 reconstruction으로 학습 | B07 |

사용자가 요청한 “이미지 당 오차를 구해서 그만큼 shift”는 **B03/B04**다. B02의 단일 공통 상수로 대신하지 않는다.

### 3.2 사전 추정기

주 estimator는 보고서의 방식에 맞춰 고정한다.[S1 §4.2]

- native 현재 PAN 픽셀의 `[-8,+8]` 정수2D grid, Scharr gradient NCC, top3 신뢰 band의 성분별 median, 3×3 peak 이웃의 quadratic subpixel refinement.
- LMS/B 참조에는 PANσ1.98 / 참조σ0.7의 bandwidth matching, trainGT에는 양쪽σ0.8. 이는 진단용 근사이며 센서 MTF 인증 아님.
- B04는 동일 support/range에서 intensity-NCC를 사용한다. loss 기전과 estimator를 독립적으로 확인하려는 대조다.
- 회귀2차곡면이 local maximum이 아니거나 grid boundary hit, target low-texture, valid band 부족이면 fallback을 **zero shift**로 하고 원인·confidence·개수를 기록한다. nearest integer fallback처럼 다른 정책을 몰래 섞지 않는다.
- correlation peak, peak-gap, gradient/intensity disagreement는 저장하되 test 성능을 보고 thresholds를 바꾸지 않는다. 추가 confidence gate는 새case로 등록해야 한다.
- native train의 독립 patch마다 값을 구한다. parent-scene registry가 확인되면 `parent-level constant`를 별도 진단할 수 있으나 patch와 scene을 같은 조건으로 합치지 않는다.
- validation/RR/FR의 shift는 **그 이미지의 PAN/LMS만 사용**해 계산한다. RR GT를 사전 정합에 사용하면 정식 inference 결과가 아니라 `ORACLE_RR_ONLY`로 분리한다.

### 3.3 사전 shift와 augmentation 순서

offline c는 원 frame에서 저장한다. flip/rotation 후에는 c도 같은 변환을 적용한다. hflip이면 dx 부호, vflip이면 dy 부호, rot90CCW이면 `(dy,dx)→(-dx,dy)`를 적용하되 **impulse/ramp 검증을 통과한 공통 함수**만 사용한다.[S2]

warp된 영상을 작은 patch로 미리 잘라 보관한 뒤 늘어나는 border artifact가 learned A와 다르게 학습되는 일을 피한다. source P와 canonical c를 저장하고, 대응 augmentation 후 **한 번 warp**하여 U에 입력한다. pixel-intensity normalization·crop·clip 조건은 다른 case와 동일하다.

### 3.4 Synthetic ε 평가에서 고정 shift의 의미

B03의 shift는 학습 parameter가 아니지만 **새 이미지가 들어오면 추정기는 다시 실행**할 수 있다. 따라서 추가 shift 응답 평가에서 두 모드를 분리한다.

1. `REESTIMATE_CURRENT_INPUT`: Pε와 L로 offline estimator를 다시 실행. learned A와 비교할 수 있는 **비학습 정합기 전체**의 성능.
2. `HOLD_NATIVE_CACHE`: native c_i를 그대로 적용. frozen correction이 새 변위를 복원하지 못하는 것을 확인하는 진단.

두 모드를 섞어 “사전 정합은 ε에 항상 무반응”이라고 결론내리지 않는다. B00/B02/B07의 global constant는 expected gain0이다. B03의 재추정 전처리 시간은 inference cost에 포함한다.

### 3.5 B06/B08 해석 제한

B06은 train의 P↔GT proxy로 고정 shift를 만들지만, validation/RR/FR에서는 P↔LMS estimator만 쓴다. **GT 정보가 학습에만 추가된 대조이자 train/test alignment-rule mismatch를 포함**한다. GT를 쓰는 학습이 원칙적으로 불가능하다는 뜻은 아니지만, 실용 사전 LMS baseline과 동일 정보 조건은 아니다.

B08은 원본 이미지 ID 순서의 고정 cyclic derangement로 shift를 교환하고 그 후 augmentation한다. seed마다 유리한 순열을 뽑지 않는다. train·val·RR·FR 각각 별도 순열이고 GT는 쓰지 않는다. 이 대조는 shift 크기분포를 유지하며 **영상과 shift의 대응성**만 끊는다.

---

## 4. 모든 학습 case: 52개

### 4.1 공통 기본값

표에서 별도 표시하지 않은 값은 모두 다음과 같다.

`A_reference=NATIVE_LMS; U_reference=BICUBIC_MS; λε=1e-3; ε every1/radius2; Lstr=GT multi-scale squared gradient NCC; λstr=1e-2; U LR=1e-4; A LR=1e-5; native rec→U/A; fresh50K; 3 seed`.

**M07은 조합의 중심점일 뿐, 이미 가장 좋은 것으로 선택된 recipe가 아니다.** 모든 case의 수치와 불리한 반복을 그대로 보고한다.

#### 01 NO-ALIGNER / FIXED SHIFT

| Case | 내용 | 기준 대조 | A 참조 / U 참조 | ε 계수·주기 | 구조 target / 계수 | 예산 |
|---|---|---|---|---|---|---|
| **TA2-B00** | 무정합 U-Net 재학습; sampler bypass | — | NA / BICUBIC_MS | OFF | OFF | 50K × 3 |
| **TA2-B01** | zero shift sampler; 항등 resampling 대조 | TA2-B00 | NA / BICUBIC_MS | OFF | OFF | 50K × 3 |
| **TA2-B02** | 학습 split의 사전 LMS 추정치 중앙값 하나 고정 | TA2-B00 | NATIVE_LMS / BICUBIC_MS | OFF | OFF | 50K × 3 |
| **TA2-B03** | 영상별 PAN-LMS gradient 등록 shift 고정 후 U 학습 | TA2-B00 | NATIVE_LMS / BICUBIC_MS | OFF | OFF | 50K × 3 |
| **TA2-B04** | 영상별 PAN-LMS intensity 등록 shift 고정 후 U 학습 | TA2-B03 | NATIVE_LMS / BICUBIC_MS | OFF | OFF | 50K × 3 |
| **TA2-B05** | 영상별 PAN-bicubic MS 등록 shift 고정 | TA2-B03 | BICUBIC_MS / BICUBIC_MS | OFF | OFF | 50K × 3 |
| **TA2-B06** | train GT로 사전 shift; val/RR/FR 적용 shift는 LMS만 | TA2-B03 | NATIVE_LMS / BICUBIC_MS | OFF | OFF | 50K × 3 |
| **TA2-B07** | CNN 없이 학습 가능한 전역 2변수 shift + U-Net | TA2-B00 | NA / BICUBIC_MS | OFF | OFF | 50K × 3 |
| **TA2-B08** | 영상별 사전 LMS shift를 고정 순열로 교환 | TA2-B03 | NATIVE_LMS / BICUBIC_MS | OFF | OFF | 50K × 3 |

#### 02 REFERENCE × EPSILON × STRUCTURE

| Case | 내용 | 기준 대조 | A 참조 / U 참조 | ε 계수·주기 | 구조 target / 계수 | 예산 |
|---|---|---|---|---|---|---|
| **TA2-M00** | BICUBIC_MS A 입력 | rec + no epsilon + no direct structure | TA2-B00 | BICUBIC_MS / BICUBIC_MS | OFF | OFF | 50K × 3 |
| **TA2-M01** | BICUBIC_MS A 입력 | rec + epsilon + no direct structure | TA2-M00 | BICUBIC_MS / BICUBIC_MS | 0.001/every1 | OFF | 50K × 3 |
| **TA2-M02** | BICUBIC_MS A 입력 | rec + no epsilon + GT 구조 정합 | TA2-M00 | BICUBIC_MS / BICUBIC_MS | OFF | GT / 0.01 | 50K × 3 |
| **TA2-M03** | BICUBIC_MS A 입력 | rec + epsilon + GT 구조 정합 | TA2-M00 | BICUBIC_MS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-M04** | NATIVE_LMS A 입력 | rec + no epsilon + no direct structure | TA2-M00 | NATIVE_LMS / BICUBIC_MS | OFF | OFF | 50K × 3 |
| **TA2-M05** | NATIVE_LMS A 입력 | rec + epsilon + no direct structure | TA2-M01 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | OFF | 50K × 3 |
| **TA2-M06** | NATIVE_LMS A 입력 | rec + no epsilon + GT 구조 정합 | TA2-M02 | NATIVE_LMS / BICUBIC_MS | OFF | GT / 0.01 | 50K × 3 |
| **TA2-M07** | NATIVE_LMS A 입력 | rec + epsilon + GT 구조 정합 | TA2-M03 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |

#### 03 HISTORICAL EPSILON

| Case | 내용 | 기준 대조 | A 참조 / U 참조 | ε 계수·주기 | 구조 target / 계수 | 예산 |
|---|---|---|---|---|---|---|
| **TA2-H00** | 기존 bicubic Teacher recipe; epsilon lambda0, 격 update | TA2-M01 | BICUBIC_MS / BICUBIC_MS | DATASET_LEGACY/every2 | OFF | 50K × 3 |
| **TA2-H01** | H00에서 A 참조만 native LMS | TA2-H00 | NATIVE_LMS / BICUBIC_MS | DATASET_LEGACY/every2 | OFF | 50K × 3 |

#### 04 JOINT LOSS GRID

| Case | 내용 | 기준 대조 | A 참조 / U 참조 | ε 계수·주기 | 구조 target / 계수 | 예산 |
|---|---|---|---|---|---|---|
| **TA2-G00** | epsilon=0.0001; GT structure=0.001; 3x3 격자 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.0001/every1 | GT / 0.001 | 50K × 3 |
| **TA2-G01** | epsilon=0.0001; GT structure=0.01; 3x3 격자 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.0001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-G02** | epsilon=0.0001; GT structure=0.1; 3x3 격자 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.0001/every1 | GT / 0.1 | 50K × 3 |
| **TA2-G03** | epsilon=0.001; GT structure=0.001; 3x3 격자 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.001 | 50K × 3 |
| **TA2-G05** | epsilon=0.001; GT structure=0.1; 3x3 격자 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.1 | 50K × 3 |
| **TA2-G06** | epsilon=0.01; GT structure=0.001; 3x3 격자 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.01/every1 | GT / 0.001 | 50K × 3 |
| **TA2-G07** | epsilon=0.01; GT structure=0.01; 3x3 격자 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.01/every1 | GT / 0.01 | 50K × 3 |
| **TA2-G08** | epsilon=0.01; GT structure=0.1; 3x3 격자 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.01/every1 | GT / 0.1 | 50K × 3 |

#### 05 STRUCTURAL SIGNAL

| Case | 내용 | 기준 대조 | A 참조 / U 참조 | ε 계수·주기 | 구조 target / 계수 | 예산 |
|---|---|---|---|---|---|---|
| **TA2-S00** | GT 대신 native LMS에 직접 구조 정합; 같은 descriptor | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | NATIVE_LMS / 0.01 | 50K × 3 |
| **TA2-S01** | GT/LMS 구조 정합 각각 0.5로 결합 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | HALF_GT_HALF_LMS / 0.01 | 50K × 3 |
| **TA2-S02** | GT gradient 방향 기반 descriptor로 교체 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-S03** | GT gradient NCC 단일 scale sigma0.8 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-S04** | train LMS-PAN로 고정한 밴드 reliability weight | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-S05** | native 및 epsilon 복원 PAN에 GT 구조 loss 평균 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-S06** | A는 epsilon+직접 구조만; U는 reconstruction | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-S07** | 직접 구조 대신 train GT 사전 shift proxy 회귀+epsilon | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |

#### 06 OPTIMIZATION

| Case | 내용 | 기준 대조 | A 참조 / U 참조 | ε 계수·주기 | 구조 target / 계수 | 예산 |
|---|---|---|---|---|---|---|
| **TA2-O00** | M07 epsilon만 격 update로 변경 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every2 | GT / 0.01 | 50K × 3 |
| **TA2-O01** | M07 A LR=3e-6 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-O02** | M07 A LR=1e-4 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-O03** | GT structure 계수만 첫5K linear ramp | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-O04** | epsilon native anchor stop-gradient 제거 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |

#### 07 U INPUT / BASE COORDINATES

| Case | 내용 | 기준 대조 | A 참조 / U 참조 | ε 계수·주기 | 구조 target / 계수 | 예산 |
|---|---|---|---|---|---|---|
| **TA2-U00** | B00 U 입력·residual base도 native LMS로 변경 | TA2-B00 | NA / NATIVE_LMS | OFF | OFF | 50K × 3 |
| **TA2-U01** | B03 U 입력·base를 native LMS로 변경 | TA2-B03 | NATIVE_LMS / NATIVE_LMS | OFF | OFF | 50K × 3 |
| **TA2-U02** | M05 epsilon-only; U 입력·base도 native LMS | TA2-M05 | NATIVE_LMS / NATIVE_LMS | 0.001/every1 | OFF | 50K × 3 |
| **TA2-U03** | M07 epsilon+GT 구조; U 입력·base도 native LMS | TA2-M07 | NATIVE_LMS / NATIVE_LMS | 0.001/every1 | GT / 0.01 | 50K × 3 |

#### 08 SIZE / SCALE

| Case | 내용 | 기준 대조 | A 참조 / U 참조 | ε 계수·주기 | 구조 target / 계수 | 예산 |
|---|---|---|---|---|---|---|
| **TA2-X00** | M07 + A 보조 crop48/56/64; native U64 고정 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-X01** | M05 + A 보조 확대1/2/4/8; epsilon-only | TA2-M05 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | OFF | 50K × 3 |
| **TA2-X02** | M07 + A 보조 확대1/2/4/8; epsilon+GT 구조 | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |
| **TA2-X03** | X02 + 정규화된 추가변위 scale consistency | TA2-X02 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 50K × 3 |

#### 09 LONGER BUDGET

| Case | 내용 | 기준 대조 | A 참조 / U 참조 | ε 계수·주기 | 구조 target / 계수 | 예산 |
|---|---|---|---|---|---|---|
| **TA2-D00** | B00 fresh100K; 처음부터 cosine100K | TA2-B00 | NA / BICUBIC_MS | OFF | OFF | 100K × 3 |
| **TA2-D01** | B03 fresh100K; 처음부터 cosine100K | TA2-B03 | NATIVE_LMS / BICUBIC_MS | OFF | OFF | 100K × 3 |
| **TA2-D02** | M05 fresh100K; 처음부터 cosine100K | TA2-M05 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | OFF | 100K × 3 |
| **TA2-D03** | M07 fresh100K; 처음부터 cosine100K | TA2-M07 | NATIVE_LMS / BICUBIC_MS | 0.001/every1 | GT / 0.01 | 100K × 3 |


### 4.2 M00–M07: 가장 중요한 완전 요인 설계

| A 참조 | ε | 직접 GT 구조 | Case |
|---|---|---|---|
| B | OFF | OFF | M00 |
| B | ON | OFF | M01 |
| B | OFF | ON | M02 |
| B | ON | ON | M03 |
| L | OFF | OFF | M04 |
| L | ON | OFF | M05 |
| L | OFF | ON | M06 |
| L | ON | ON | **M07** |

M01↔M05는 ε recipe에서 참조만 바꾼 효과, M05↔M07은 **동일 native LMS·ε에서 direct GT loss 추가 효과**, M06↔M07은 직접 정합에 ε를 함께 넣는 효과다. M00↔M04도 보아 normalization/입력 자체 효과를 분리한다.

같은 반복에서 `[(M07−M06)−(M05−M04)]`를 계산하면 native LMS에서 ε×구조 interaction을 기술할 수 있다. 단3seed의 interaction을 확정적 통계 우위로 부르지 않는다. 지표별 부호(↑/↓)를 유지한다.

### 4.3 3×3 계수 격자

| λε \ λstr | 0.001 | 0.01 | 0.1 |
|---|---|---|---|
| 0.0001 | G00 | G01 | G02 |
| 0.001 | G03 | **M07** | G05 |
| 0.01 | G06 | G07 | G08 |

중앙점은 M07을 재사용하고 별도 G04 실험으로 세지 않는다. 같은3seed M07을 모든 관련 비교의 중앙점으로 사용한다. 새로운 최댓값을 얻었다는 이유로 비교범위를 축소하거나 기본값의 seed만 바꾸지 않는다.

### 4.4 역사 대조와 장기 대조

H00/H01의 λ0는 WV3/GF2=`1e-4`, QB=`3e-4`, every2다.[S1 §6.1] 이번 통합 main factorial은 세 센서 모두 같은 λε=`1e-3`를 사용해 기전을 비교한다. 원래 donor 조건 재현은 H로 분리한다. H도 donor continuation이 아니라 fresh50K다.

D00–D03은 처음부터 cosine100K의 별도 학습이다. **50K checkpoint에서 cosine을 재시작한 추가50K와 같지 않다.** 해당100K run의50K 상태도 보존하지만, 50K-budget run과 LR history가 같다고 해석하지 않는다.

### 4.5 U/X 실험을 섞어 해석하지 않는다

- U00–U03: U의 MS conditioning과 output residual base를 **함께** native LMS로 바꾸는 명시적 input/base bundle 대조다. Aligner만 바꾸는 M과 구분한다.
- X00: native U/P64 forward는 그대로; ε/struct auxiliary A branch에48/56/64 crop을 적용하고 native/aux loss를 각0.5 평균한다. crop이64일 때도 loss 배율이 커지지 않는다.
- X01/X02: native U는64로 유지, P/L/GT HR fields를 **같은 geometry mapping**으로1/2/4/8 배 확대해 auxiliary A를 학습. native와 auxiliary loss는0.5씩, auxiliary correction은 현재 확대 격자픽셀로 예측하고 `ε_base`에 scale을 곱한다. 학습 loss의 ε 잔차는 scale로 나누어 원 좌표단위로 평균한다.
- **기존 작은 patch를 확대하는 것은 실제 native FR512 장면을 학습한 것이 아니다.** P/L/GT를 공통 HR geometry로 변환하는 통제 증강이며, LRMS→LMS 연산을 bicubic으로 되돌리는 것이 아니다.
- X03: X02와 동일한 입력·branch수에서 `λscale=0.001`의 `|Δc_aux/k−sg(Δc_native)|`를 추가. 각 스케일의 known ε loss도 남겨0반응해를 막는다.
- ε injection은 base P에서 먼저 하고 모든 HR field에 공통 resize map을 적용한다. finite-grid composition residual을 사전 측정하고 `SYNTHETIC_SCALE`로 표시한다. native FR에 임의×4 correction을 적용하지 않는다.
- true large-field 학습 자료가 따로 확인되면 새 DATA_REVISION case로 설계한다. RR/FR test patch를 train으로 가져오지 않고, 64patch mosaic을 지리적으로 연속된512라고 부르지 않는다.

---

## 5. 서버·seed·예산과 queue

| 서버 | Dataset | Bands / maxDN | R1 | R2 | R3 |
|---|---|---|---:|---:|---:|
| **s1** | **WV3** | 8 / 2047 | 261006101 | 261006102 | 261006103 |
| **s3** | **QB** | 4 / 2047 | 261006301 | 261006302 | 261006303 |
| **s5** | **GF2** | 4 / 1023 | 261006501 | 261006502 | 261006503 |

이 숫자는 이번 사전 설계값이다. 기존 성공 seed를 가져온 것이 아니다. 각 데이터셋의 같은 R번호에서 모든 case는 **동일 initial U와 동일 initial A, 동일 native sample/augmentation stream**을 사용한다. 아키텍처 제거로 constructor RNG 소비 순서가 달라져 U가 바뀌지 않도록 U snapshot을 먼저 하나 만들고 A도 별도 RNG로 만든다. learned-global B07은 CNN A가 없지만 U 초기값은 같다.

ε RNG, crop/scale RNG, native feeder RNG, 진단 probe RNG는 분리한다. ε가 없는 case가 다음 native mini-batch를 다르게 받게 만들지 않는다. 한 case의 학습된 weight를 다른 case의 초기값으로 재사용하지 않는다. 같은 seed retry는 다른 독립 반복이 아니다.

### 5.1 시간 제한 없음의 해석

- 52조건 각각 R1/R2/R3를 완료할 때까지 실행한다. 전체 시간 상한은 두지 않는다.
- 목표 성능·긍정 결과가 나올 때까지 무한 재추첨하지 않는다. **3회 반복은 seed 차이를 측정·완화하는 것이지 제거를 보장하지 않는다.**
- 등록된3회가 끝나면 전체 비교를 보고 `REGISTERED_BLOCK_COMPLETE`로 남긴다. 새 case는 별도 revision/registry에 추가 후 역시3회 수행한다.
- 큰 λ 발산, LMS 자산 미확인, 디스크 부족은 실패/보류 사유다. 처리하지 못한 조건을 완료한 것처럼 세지 않는다.

### 5.2 계산량

- 서버별: **52×3=156개 학습**.
- 전체: **468개**, 그중 fresh50K432개 + fresh100K36개.
- 총 optimizer update: **25,200,000**.
- run당 HQNR 후보50개를 평가하므로 최소 **23,400 checkpoint × FR20 = 468,000 scene 추론**. 이는 ε/gradient/다른 진단 비용을 포함하지 않는다.
- 실제 wall time·추가 메모리는 각 서버 smoke와 최초block 실측으로 예산화한다. 서버 종류/이전 run 시간만으로 완료시각을 보장하지 않는다.

### 5.3 우선 실행 block

P0_CORE는 `B00 B02 B03 B04 B05 B07 M00 M01 M02 M03 M04 M05 M06 M07 U00 U03`의16조건이다. **먼저 이16조건의 모든3회(서버당48run)**를 완결해 질문의 핵심을 확보한다. 이어서 registry에 정의한 remaining block을 각3회 실행한다. 우선순위가 낮다는 이유로 결과가 불리한 case를 중단하지 않는다.

queue는 첫 repetition에 기본순서, 이후 사전 정의한 rotation/reversal을 사용한다. 파일 `s1_WV3_Queue_156.csv`, `s3_QB_Queue_156.csv`, `s5_GF2_Queue_156.csv`의 `queue_seq`가 실행 순서다. **다른 서버나 s2/s4의 결과를 기다리는 단계가 없다.**

---

## 6. 공통 학습·품질 선택·평가 규약

### 6.1 공통 학습

- Batch48, AdamW lr_U=`1e-4`, lr_A=`1e-5`, weight_decay0.01, betas(0.9,0.999), eps1e-8, warmup100+cosine, FP32.
- no Aligner/fixed shift에서는 A optimizer 없음. B07 전역2변수는 A쪽LR로 관리하고 실제 추가parameter2개를 별도 기록한다.
- native patches PAN64/MS16/LMS64/GT64. 동일 split·band·DN·source·augment 함수를 고정한다. rotation/flip의 실제 확률은 source 구현과 함께 기록하고 임의로 변경하지 않는다.
- geometry 확대보조가 memory를 초과하면 sample별 auxiliary microbatch로 정확한 mean gradient를 누적한다. native U batch48과 샘플 총수는 변경하지 않는다. dropout0/GN per-sample의 동등성을 확인한다.
- loss logging: rec, raw/weightedε, raw/weightedstruct, optionalscale, grad_norm, actual U/A updates, ε effective count. λ가 들어간항에 다시λ를 곱하지 않는다.

### 6.2 Checkpoint 보존·선택

주 reporting은 **HQNR_MAX50**, 보조이자 인과비교 기준은 **EXACT_FINAL**이다.

- 50K grid: `{1010,2020,...,49490,50000}` 50개.
- 100K grid: `{2020,4040,...,98980,100000}` 50개. 50K 중간 state는 추가 보존하되 grid밖이라 HQNR_MAX50 후보에 몰래 넣지 않는다.
- 동일 공식 native FR20, **원 PAN/native LMS 평가 reference**, 원 full-frame support, 장면별 HQNR 평균.
- 동률은 full-precision HQNR가 같을 때 더 이른 completed step. **ERGAS tie-break/필터 없음**.
- 50/50 후보가 전 장면 평가되기 전에는 `HQNR_SELECTION_COMPLETE`로 표시하지 않는다. evaluator missing만 있으면 학습을 다시 하지 않고 평가복구한다.
- RR 지표는 선택된 **같은 A/U 또는 같은 U+fixed shift 규약**에서 계산한다. 다른 checkpoint의 낮은ERGAS를 합치지 않는다.
- 공식 FR20을 선택에 썼으므로 `test_aware=true`, `selection_split=FR20`, `independent_test=false`. validation-selected로 오표기하지 않는다.
- gain/structural signal의 학습 기전 비교는 primary EXACT_FINAL과 공통progress점으로 하고, HQNR-selected 지표는 별도 열/행에 둔다.

### 6.3 공식 평가와 진단 분리

RR은 현재 공식20장·기존crop을 유지하고, FR도 원래20장 전부 평가한다.[S1 §7.1] 품질 계산 reference를 warped PAN으로 바꾸거나 structural loss에서 쓴mask로공식HQNR를계산하지 않는다.

FR에는 GT가 없으므로 **GT 구조loss·GT residual은 N/A**, native LMS residual은 proxy라고 표시한다. RR GT 정합 proxy도 물리 displacement GT로 부르지 않는다. RR20/FR20의 같은 index를 지리적으로 대응한다고 가정하지 않는다.

B06·S07은 GT가 train supervision에만 들어가며 추론에는 들어가지 않는다. 검증용 RR oracle shift는 `Record_type=ORACLE`, 공식성능·HQNR selector·seed aggregate에서 제외한다.

---

## 7. 모든 case에서 공통 수집할 진단

### 7.1 데이터·좌표 검증 P00

Stored LMS와 재생성 LMS의 일치, L/B와 GT 사이의 위상, 각 augmentation의 좌표 변환, correction 부호, zero sampler, 양·음의 fractional shift, 64/256/512의 픽셀 단위를 검사한다. 직접 구조 loss의 c-gradient는 작은 유한차분과 대조한다. **Source 검증과 실제 CUDA smoke를 별도로 보고한다.**

LMS가 LRMS로부터만 만들어졌는지도 검사한다. P나 GT 값을 바꾸어도 L이 같아야 한다. PAN에 의존하는 LMS 생성 경로가 발견되면 자기참조 위험으로 중지한다. 공식 inference에 GT sentinel을 넣었을 때 GT가 입력으로 접근되면 실패다. A의 구조 loss만 backward했을 때 U gradient가 생겨도 실패다.

### 7.2 Shift response P01

- `AXIS16`: 축 4방향 × 반경 0.25/0.5/1/2px.
- `DIAG16`: 대각 4방향 × 같은 **L2 반경**.
- `DISK64_HELD`: 학습 RNG와 다른 seed로 만든 고정 offset 64개.
- `OOD_AXIS`: 반경 4/8px. In-range gain과 섞지 않는다.

\[
\Delta c=c_\epsilon-c_0,\quad
 gain=-\frac{\sum\Delta c\cdot\epsilon}{\sum\epsilon\cdot\epsilon},\quad
 EPE=\operatorname{mean}\|\Delta c+\epsilon\|_2.
\]

Gain의 분자·분모, y/x 성분, 반경별 결과를 함께 보고한다. 완전 상수의 AXIS16 성분 MAE=0.46875는 그 probe에서만 비교한다.[S1 §6.3] 같은 GSD에서 64→512 영역을 넓힌 실험과 같은 FOV를 축소한 실험은 다른 `Measurement_domain`으로 기록한다.

### 7.3 Native geometry P02

각 scene의 native c, 평균 벡터, `mean(norm(c))`, `norm(mean(c))`, 표준편차·p95를 구분한다. Warp 전후의 GT/LMS gradient proxy와 **intensity proxy**, 두 추정기의 disagreement, boundary hit, low-texture 비율과 valid count를 올린다. 잔차는 단순 벡터 차이가 아니라 **실제 warped PAN에서 다시 추정**한다.

GT 구조 loss가 낮아졌다는 사실만으로 정합 개선을 선언하지 않는다. 학습에 쓰지 않은 intensity estimator, 알려진 ε 회복, native 복원 품질을 함께 본다. 모든 proxy가 좋아져도 물리적 변위 GT를 통한 검증으로 부르지는 않는다.

### 7.4 Per-image adaptation P03

모든 learned A의 FINAL/HQNR-selected 모델에서 current c, train 평균·중앙값 c, shuffled c, zero c를 같은 U에 적용해 비교한다. 이것은 **추론 개입**이며 B00/B03의 실제 재학습과 별개다. 상수는 학습 split에서만 정한다. 여러 shuffle·scene 결과는 독립 학습 seed로 세지 않는다.

### 7.5 MS dependence P04

PAN-only 이동의 Δc≈−ε, LMS-only 이동의 Δc≈+ε, joint 이동의 Δc≈0, 다른 scene의 LMS를 준 경우를 검사한다. Texture가 빈약한 subset, normal/clean-support/border-only 결과도 분리한다. 진단용 LMS shift를 공식 reconstruction 경로에는 적용하지 않는다.

### 7.6 크기·스케일 P05

같은 FR scene에서 resize 없는 nested 64/128/256/512 crop을 사용해 추가 shift에 대한 gain을 비교한다. 서로 다른 crop의 native c 차이를 전부 오류라고 단정하지 않는다. 영역별 실제 대응이 다를 수도 있다.

Same-FOV control에서는 P/L에 공통 geometry와 검증된 degradation을 적용해 Δc의 스케일 관계를 본다. 이를 공식 Wald RR 재생성이라고 주장하지 않고, 서로 다른 공식 RR/FR 장면을 동일 FOV로 간주하지 않는다. LMS 생성 연산의 phase와 진단용 resize의 좌표 매핑을 각각 기록한다.

### 7.7 Gradient conflict P06

고정 train 32 mini-batch probe에서 같은 batch에 대해 다음 gradient를 따로 계산한다.

\[
g_R=\nabla_A L_{rec},\quad
 g_\epsilon=\lambda_\epsilon\nabla_A L_\epsilon,\quad
 g_S=\lambda_{str}\nabla_A L_{str}.
\]

Raw/weighted norm, reconstruction 대비 비율, cosine(rec,ε), cosine(rec,struct), cosine(ε,struct), 실제 parameter update 크기를 보고한다. Whole A와 stem/body/fc1/fc2/bias를 나눈다. Zero norm이면 cosine은 N/A이지0이 아니다. 초기·중기·후기(예: 1010/25250/50000; 100K는 같은 진행 비율)에 측정하고 총 weighted gradient도 확인한다.

### 7.8 Cost P07

Learned A+U의 시간뿐 아니라 고정 이동 baseline의 **registration_ms + warp/U_ms = pipeline_ms**를 기록한다. Native c를 미리 cache한 뒤 cache 읽기만 포함한 시간을 실용 inference 비용으로 보고하지 않는다.

Batch1, RR256/FR512, 실제 GPU·precision, warmup/repeat 수를 고정한다. 동일 기기에서의 반복 변동과 서로 다른 서버의 차이를 구분한다. 학습 비용과 정합 사전 계산 비용도 분리한다.

### 7.9 진단 전용 양성 대조·Oracle P08

학습 없는 P↔shifted P의 known-shift 검증과, RR GT를 사용해 직접 구한 per-image optimal shift는 **진단 전용**이다. 이를 3개의 독립 학습 case로 세지 않는다. Oracle이 좋아도 실용 PAN–LMS 정합기가 같다고 주장하지 않으며, oracle도 나쁘면 정합 목적과 pansharpening 목적의 차이를 조사한다.

---

## 8. 3회 반복의 집계·결론 규칙

1. Case별 R1/R2/R3 값을 모두 보존하고 mean±sample STD(ddof=1), median, min/max, N_success/N_attempted를 표시한다.
2. 같은 데이터셋·반복·initial U/A·stream의 **paired difference**를 계산한다. 서로 다른 seed의 최고점만 비교하지 않는다.
3. 한 seed가 실패하면 성공한2회 평균을3회 평균으로 표시하지 않는다. 운영 실패의 retry는 같은 seed/config를 유지한다. 수치 경로를 수정했다면 revision을 올리고 해당 대조군도 같은 수정본으로 검정한다.
4. Checkpoint50개·scene20장·crop·ε 반복은 독립 training seed가 아니다. Scene/parent bootstrap CI와 seed 간 STD는 별개다. 지리적 독립성이 미확인인 경우도 표시한다.
5. 기본16 core의3회가 끝나기 전에 유리한 case만 반복하는 선택을 하지 않는다. 나머지52조건도 등록된 계획대로 진행한다. FR test로 recipe를 골랐다면 그 개발 선택 이력을 유지한다.
6. **Gain, 구조 정합 proxy, RR 품질, FR 품질**의 네 축을 나눠 결론낸다. Gain이1에 가까워졌다고 최종 method 개선을 확정하거나, HQNR 상승만으로 물리적 정합을 주장하지 않는다.

| 필수 비교 | 질문 |
|---|---|
| B00 ↔ B03/B04/B07 | CNN Aligner 없이 어느 정도까지 가능한가? |
| B03 ↔ B05 | 사전 정합에서도 dataset LMS와 bicubic 참조가 다른가? |
| M01 ↔ M05; M03 ↔ M07 | A의 참조만 native LMS로 바꾼 효과는 무엇인가? |
| M05 ↔ M07 | ε에 직접 GT 구조 지도를 추가하면 개선되는가? |
| M06 ↔ M07 | 구조 지도만 사용할 때보다 ε를 병행하는 것이 필요한가? |
| B03 ↔ M07 | 영상별 고정 사전 보정보다 joint learned 정합이 유리한가? |
| M07 ↔ S07 | 미분 가능한 영상 구조 지도와 사전 변위 pseudo-label 회귀가 다른가? |
| M07 ↔ S06 | Reconstruction gradient가 A에 필요한가, 구조 학습과 충돌하는가? |
| B00/B03/M05/M07 ↔ U00/U01/U02/U03 | U의 입력·base 좌표계도 중요한가? |
| M05 ↔ X01; M07 ↔ X00/X02; X02 ↔ X03 | 입력 크기·스케일 학습이 별도로 필요한가? |
| B00/B03/M05/M07 ↔ D00/D01/D02/D03 | 100K 예산·schedule로 바꾸면 경향이 달라지는가? |

---

## 9. `analysis` 업로드 계약: 시트만으로 판단할 수 있어야 한다

### 9.1 기존 시트 보존과 v2 추가

- 기존 `analysis`의 A:DH 112열은 이름과 순서를 유지한다. 이전33개 PLAN과5개 lane 선언은 **SUPERSEDED_PLAN/ASSIGNMENT**로 구분하고 삭제하지 않는다.
- 기존 보고서6행은 `REPORT_ONLY`로 유지한다. 새 Teacher 실측이나 새 seed로 재사용하지 않는다.
- 새52개 case는 `Record_type=PLAN_V2`, Status=`PLANNED_NOT_RUN`, Role=`T`로 등록한다.
- 확장 schema는 **A:EV 152열**이다. DI:EV 40열만 추가하여 기존 열 순서를 보존한다.
- 모든 새 row의 Campaign_ID와 Schema_version=`AAX_ANALYSIS152_v2`를 검증한다. v1 수집기를 v2 결과에 그대로 재사용하지 않는다.
- 시트의 계획 등록은 실제 기동을 뜻하지 않는다. s1/s3/s5의 PID·GPU·active workdir·update 증가를 확인한 후에만 RUNNING으로 표시한다.

### 9.2 반드시 올라와야 하는 record

| Record_type | 입력할 내용 | 완료 조건 |
|---|---|---|
| PLAN_V2 | case definition, reference, loss, control, expected repeats | 메트릭 공란 |
| STATUS | run/attempt/seed, 실제update, 상태, 오류, pending eval | 새run부터종료까지상태갱신 |
| ASSET / UNIT | LMS lineage·재생성오차·phase·부호·gradient/GT leakage 검사 | 각승격조건의PASS/FAIL수치 |
| SUMMARY | HQNR-selected 및EXACT_FINAL의nativequality·gain·c·구조proxy·cost | 동일checkpoint와50/50후보검증 |
| RESPONSE | train/val/RR/FR·반경별gain/MAE/EPE·axis성분·N·분자/분모 | 신뢰구간과범위밖별도 |
| CURVE | 고정progress의loss·c·gain·structvalid·gradient개요 | 단순최종값만등록금지 |
| GRAD | wholeA/layergroup별raw/weightednorm·ratio·cos·actualupdate | zero/undefined구분 |
| PAIRED | 각repeat의대조차이,mean/STD,sceneclusterCI별도 | controlID와pairedscope검증 |
| SEED_AGG |3repeat개별값·N·mean/STD/min/max,실패율 | 특정seed유리값만요약금지 |
| ORACLE / COUNTERFACTUAL | GT진단또는동결개입결과 | 정식native학습비교와구분 |

지표열은현재schema위치를유지하며 FR=`HQNR,D_s,D_lambda,JQM`, RR=`ERGAS,SCC,SAM,PSNR,SSIM,Q4/Q8,RMSE,CC` 순서다. 구조분석의중요숫자는계산된Estimate또는dedicated열에쓴다. **원시JSON 링크만쓰고완료로표시하지않는다.**

### 9.3 추가40열의 용도

`Campaign_ID, Model_scope, Aligner_policy, A_reference, U_reference, LMS_SHA, LMS_operator_SHA, Struct_target, Struct_descriptor, Struct_branch, Lambda_struct, A_receives_rec, Struct_loss, Grad_struct_raw_norm, Grad_struct_weighted_norm, Grad_cos_rec_struct, Grad_cos_eps_struct, Offline_shift_SHA, Shift_estimator, Shift_confidence, Global_or_per_image, N_train_seeds, Seed_std, Delta_gain, Delta_proxy_GT, Delta_proxy_LMS, Learned_time_ms, Registration_ms, Pipeline_ms, Params_M, FLOPs_G, Effective_updates, Expected_repeats, Completed_repeats, Phase_dx, Phase_dy, Phase_MAE, Failure_fraction, Repeat_status, Upload_receipt_SHA`.

비용은같은run·실측조건의값만쓴다. train시간을inference대신넣지않고,GT없음/zero-gradient/지원안되는cost는N/A이유를남긴다. per-scene원시자료는externalCSV에보존하되,사전정한고정장면목록의paired수치와worst/best/median분포·coverage는시트에올려외부파일을열지않아도위험을판단할수있게한다.

### 9.4 쓰기·재시도·서버 독립성

Row key는 `campaign | dataset | server | case | replica | attempt | checkpointSHA | recordtype | split | probe/radius/scope`를 포함한다. 같은 기록의 전송 retry는 중복 없이 처리하고, 같은 ID에 다른 수치가 생기면 덮어쓰지 않고 충돌을 보고한다.

각 서버는 local outbox에 JSON/CSV와 SHA를 먼저 저장한다. 검증된 atomic append 또는 독립 lane 원본에 쓴 뒤 중앙 `analysis`가 읽는 방식을 사용한다. **서버 간 공유 lock·파일 전달·실험 완료 대기는 없다.** 각 서버의 자체 중복 실행 방지만 유지한다.

동시 등록 후에는 Record_ID 중복 여부, column mapping, 숫자의 full precision을 readback한다. Readback은 전송 검증이지 GPU 평가 자체의 검증은 아니다. Header 이름으로 매핑하고 이번의 **정확한152열**과 v1의112열을 구분한다.

대량 pixel vector·이미지·모든 배치 dump는 원시 파일에 보존한다. 그러나 핵심 summary/curve/response/paired/seed aggregate의 **숫자는 시트에도** 있어야 한다. 전체 workbook의 grid cell 여유와 수식 성능을 확인하며 필요한 행만 늘린다. 상세 자료를 분할할 때도 중앙 분석에 필요한 요약 숫자를 생략하지 않는다.

이 결과를 기존 `paper`, `ablations`, `배포용 모델`, `_records`에 자동으로 대표 승격하지 않는다. 이번 campaign은 새로운 Teacher 학습 recipe의 원인 검정이다.

### 9.5 색상·가독성

기존 색상 체계를 유지한다. v1 계획은 회색, PLAN_V2는 연한 청록, 실측 SUMMARY는 녹색, RESPONSE/GRAD는 보라, PAIRED는 청록, REFERENCE는 청회색, 미완은 노랑, 실패는 연한 빨강으로 구분한다. 미측정 성능은 공란이며0으로 채우지 않는다. **실험 설명은 한 줄**로 쓰고 seed·긴 run ID·SHA는 오른쪽 열에 둔다.

---

## 10. 바로 전환하기 위한 구현·기동 체크리스트

1. **s1/s3/s5만** 현재 PID, cwd, container, GPU UUID, controller, queue, source, outbox를 확인한다. s2/s4의 과거 analysis 배정은 폐기하지만 그 서버의 다른 작업을 중단하지 않는다.
2. 실제 활성 runner의 안전한 checkpoint 경계에서 현재 작업을 보존한다. Weights, optimizer, scheduler, RNG, sampler, evaluation ledger를 남기고 `PAUSED_FOR_TEACHER_LMS_V2`로 표시한다. 포괄적인 kill이나 파일 삭제는 하지 않는다.
3. `work_dir/aligner_teacher_lms_v2/<server>/`의 별도 namespace와 고정 source를 만든다. 기존 Teacher/Student/보고서 파일을 덮어쓰지 않는다.
4. Native LMS 입력, P0 Teacher, 공통 warp, 세 loss, gradient route, 사전 정합기, augmentation 변위 변환을 구현한다. `align/resample.py`에서는 검증된 interp23tap만 재사용할 수 있다. 그 파일의 MS→PAN warp 경로 전체를 새 PAN-only frontend로 잘못 가져오지 않는다.
5. Actual resolved config와 registry를 대조한다. Historical λ는 dataset에 따라 숫자로 해석하고 `DATASET_LEGACY` 문자열을 loss에 넘기지 않는다.
6. Shape64/256/512, identity/bypass, 양·음 fractional shift, constant/gradient 입력, stored LMS parity, HR augmentation, GT leakage 부재, 구조 loss 유한차분, A/U gradient 분리, RNG 일치, resume parity, 50후보 selection, tie 처리와 readback을 검사한다.
7. 실제 CUDA에서 B00/B03/M05/M07/U03의 짧은 smoke를 수행한다. 이후 원 initial snapshot을 다시 읽어 fresh 실험을 시작한다. Smoke의 학습된 상태를 정식 초기값으로 쓰지 않는다.
8. Preflight에서 성능을 보고 계수를 임의 수정하지 않는다. 단위·정규화·구현 버그를 수정했다면 source/config revision을 올리고 해당 비교군 전부가 같은 수정본을 쓰게 한다.
9. LMS branch가 자산 문제로 막혀도 독립적인 bicubic historical branch는 진행할 수 있다. 그러나 missing LMS를 bicubic으로 대체하거나 전체52조건이 완료됐다고 표시하지 않는다.
10. 기동과 함께 기존 보존 연구나 새 후보를 prune하지 않는다. 전체 시간 제한이 없더라도 디스크 부족 안전 정지와 outbox 보존은 유지한다.

현재 실행기 CLI의 존재를 확인하지 않았으므로 **존재하지 않는 start 명령을 실행 가능한 것처럼 제공하지 않는다.** CSV/JSON은 실행 명세이며 별도의 runner 구현·검증이 필요하다. 제공된 registry CSV/JSON과 `validate_plan.py`를 GPU 학습 runner로 간주하지 않는다.

---

## 11. 결과를 보고 결론낼 때의 결정표

| 관측 | 지지할 수 있는 해석 | 아직 주장할 수 없는 것 |
|---|---|---|
| M01→M05의 위상·응답·품질 동시 개선 | Native LMS가 더 적절한 참조일 가능성 | 모든 센서의 물리적 정합 오차 제거 |
| M05→M07의 gain·독립 proxy·품질 개선 | ε 외의 직접 GT 구조 지도가 유효 | 영상 GT와 변위 GT가 동일 |
| M06→M07의 추가 shift 회복 개선 | ε consistency가 상대 변위 반응에 기여 | ε만으로 절대 bias도 제거 |
| B03과 M07의 성능 차이가 작음 | 영상별 사전 global 정합으로 충분할 가능성 | 단3회로 통계적 동등성 확정 |
| B07≈M07이고 shuffle 영향도 작음 | Sample-adaptive 정합보다 공통 bias의 역할 가능성 | 모든 scene에서 A가 불필요 |
| S07은 상수화하지만 M07은 반응 유지 | Pseudo 변위 회귀와 영상 구조 지도가 다르게 작동 | 한 seed로 일반화 |
| Geometry는 개선되지만 FR HQNR 하락 | 정합 목적과 원 PAN 참조 HQNR 목적의 상충 | 움직인 PAN 참조로 공식 HQNR 대체 |
| U03이 M07보다 반복적으로 좋음 | U의 input/base 좌표 표현도 중요할 가능성 | 전체 이득을 A만의 기여로 설명 |
| 64에서는 양호하지만512에서는 저반응 | 큰 FOV·정규화·global vector 가정 점검 필요 | 배열 크기8배를 곱하면 해결 |
| 학습 구조 loss만 개선되고 다른 proxy는 불변 | Descriptor 맞춤 또는 비기하 효과 가능성 | 실제 절대 정합 개선 확정 |

최종 결론은 실제 관측에 맞춘다. “LMS+ε+GT 구조 조합이 가장 좋아야 한다”는 조건으로 case나 seed를 제거하지 않는다.

---

## 12. 번들 파일과 인계 프롬프트

- 본 MD: 설계/수식/데이터/실행/업로드/해석 계약.
- `TeacherOnly_Cases_52.csv`: 조건별 변경 요소·기본값·control ID.
- `TeacherOnly_Run_Registry_468.csv`: 정확한 dataset/server/seed/case/updates/run ID.
- `s1_WV3_Queue_156.csv`, `s3_QB_Queue_156.csv`, `s5_GF2_Queue_156.csv`: 로컬 실행 순서.
- `experiment_spec.json`: 기계 검증 가능한 상위 명세.
- `analysis_columns_v2.json`, `analysis_column_index_v2.csv`, `analysis_upload_template_v2.csv`, `analysis_plan_records_v2.json`: 기존112열을 보존한152열 계약.
- `static_validation.json`: Registry 단위 검증 결과. **Loss/CUDA/훈련 정확성을 인증하는 파일이 아니다**.
- `sheet_update_receipt.json`: 실제 template 변경과 readback 기록.
- `sources/`: 첨부 보고서 및 구 계획의 보존 사본. 기존 운영 코드·모델 가중치는 포함하지 않는다.

### 실행 담당자에게 전달할 시작 문구

> 이 MD v2가 이전 5서버 Teacher–Student 계획을 대체한다. 이번에는 s1=WV3, s3=QB, s5=GF2에서 Teacher 크기 PAN Aligner+U-Net만 사용한다. s2/s4는 건드리지 않는다. native LMS가 실제 해당 LRMS/GT 좌표에 대응하는지 먼저 검증하고, Aligner에 PAN과 LMS를 넣는다. 직접 PAN↔GT 구조 loss와 ε consistency를 각각/함께 적용하는 완전 요인 대조와 무정합·이미지별 고정 shift 대조를 포함해 52개 조건을 각3개의 fresh seed로 실행한다. 모든 case를 registry 순서대로 완결하고, 모든 숫자·실패·phase·response·gradient·same-seed paired 차이를 analysis 152열 schema로 올려라. HQNR_MAX50는 주 품질 선택, EXACT_FINAL은 인과비교용이며 GT는 공식 inference에 사용하지 않는다. 실제 활성 PID/자산/안전전환/단위검사/GPU smoke를 확인한 후에만 기동 완료로 보고해라. 원격 실행되지 않은 계획을 완료 결과로 표시하지 말고, old PLAN은 superseded로 보존한다.

## 13. 이 문서 작성 중 반영한 시트 상태

`analysis!A55:I106`에 신규52개 계획을 등록했고, `A108:I110`에 s1/WV3·s3/QB·s5/GF2의 `NOT_STARTED` 배정을 기록했다. 기존33개 v1 계획은 `SUPERSEDED_PLAN`,5개 lane 선언은 `SUPERSEDED_ASSIGNMENT`로 보존했다. 보고서6행은 유지했다. 기존112열을 보존하고 `DI7:EV7`에40개 구조·LMS·seed·비용 필드를 추가했다.

재조회에서52개 Case_ID,152개 열 구성, 주요 factorial 설정, Record_ID, 새 실측0행, 세 서버 `NOT_STARTED`와 기존 paper Ours 수치를 확인했다. 이는 **계획/시트 변경 확인**이며 원격 실행 receipt가 아니다.

## 14. 근거와 설계값 구분

**[S1] 사용자 첨부:** `2026-10-06_PAN_Aligner_RR_FR_effectiveness_report.md`, §3(좌표/구조), §4.2–4.3(추정기와bicubic/nativeLMSphase), §6(기존εobjective와gain), §7(추론개입과native품질), §10(다음검증). 기존 결과의 사후 진단이며 새 loss의 실효성 증거가 아니다.

**[S2] 이번 조회한 저장소:** `hojunking/PAN-Crafter-repro`, commit `a2705bd1b92ed9a7f0c54a6b4b9ee4d7b82ea1a5`, `align/resample.py`. Dataset LMS용 interp23tap의 계수·phase·padding과 augmentation 순서 주의사항이 있다. 파일 주석의 과거 검증을 현재 모든 데이터 파일에 대한 검증으로 간주하지 않는다.

**[S3] 이전 실행 명세:** `PAN_Aligner_Causal_Experiments_and_Analysis_Sheet_2026-10-06.md`. Teacher–Student transfer와5서버 배정은 이번 범위에서 폐기한다. 기존112열 schema와 참고 보고서는 보존한다.

**신규 제안값:** 52조건, 구조 NCC/NGF의 정의와 가중치, fresh3seed, 서버별 데이터셋 배정, U/X 대조와152열 확장은 이번 실험 설계다. 성능·정합 개선이 이미 검증된 값은 아니다.
