# QB·GF2 전환 준비 검토 — 분포 기반 calibration과 WV3 고정 Method의 이식

**작성일:** 2026-09-20  
**상태:** 자료·코드의 읽기 검토와 준비안 작성. 원격 코드 수정, 데이터 생성, Teacher/Student 학습, 서버 큐 변경, Sheet 변경은 하지 않았다.  
**출발점:** `PANDA_WV3_PLH_W104D122_BASE_F1_20260920`  
**제안 실행 계열:** `QGBASE` — 아래 run ID는 제안이며 아직 repository registry/YAML/runner에 등록한 것이 아니다.  
**새 시간 예산:** 아직 정하지 않았다. 종료한 WV3의 20시간 최소조건을 QB/GF2에 자동 승계하거나 새로 20시간을 시작하지 않는다.

> **고정할 것은 Method·구조·첫 학습계수이고, 재측정할 것은 해당 센서와 해당 Teacher의 reconstruction-error scale, alignment-consistency scale, 입력·texture·기하 분포다.**
> **오정합 정도, q, reconstruction error는 같은 값이 아니다.**
> 첫 기준은 **센서별 새 P0·W112D123 Teacher → PLH·W104D122 Student·BASE**다. 과거 QB/GF2 donor 기반 Teacher를 이 기준으로 이름만 바꾸어 재사용하지 않는다.

## 1. 이번 조회로 확인한 범위

### 1.1 WV3 종결조건

종결문서에 고정한 주 자산은 s1/F1/PLH/W104D122/BASE/S73101의 TARGET step38380이다. H=0.958732552970, E=2.056060193721이며 이는 50K 학습 후 선택한 checkpoint다. 다른 센서에서 같은 Student seed·step이 최적이라고 가정하지 않는다. 종결 후 추가 업로드된 WV3 결과가 있어도 여기서 재탐색하거나 주 자산을 자동 교체하지 않는다. [S1]

Teacher F1은 **P0 입력이면서 Aligner를 사용하는 fresh A/U Teacher**다. 과거 bootstrap B01의 `P0=Aligner 없음` 표기와 다르다. 이 문서의 P0는 **입력 layout [aligned PAN, MS base]**만 뜻한다.

### 1.2 현재 QB/GF2 Sheet

이번 live export에서 `QB-s4`와 `GF2-s5`는 기존 준비용 형식과 PAN-Crafter 보고 기준행만 있었다. 기존 `QB-s1`, `GF2-s1`에는 B01/B02/B03/B04 준비 자산이 있으나 새 고정 Method와 동일한 실험은 아니다. 다른 s2/s3 탭에도 과거 큰 DUAL 모델 결과가 남아 있다. 아래는 진단 참고용 선택행이며, 이번에 raw checkpoint로 재평가한 값은 아니다. [S2]

| Dataset | 과거 자산 | HQNR | ERGAS | D_s | 새 기준에서의 역할 |
|---|---|---:|---:|---:|---|
| QB | B01 P0, Aligner 없음 | .9149 | 3.5910 | .0257 | 과거 raw 입력 기준 |
| QB | B02 N2 donor | .8211 | 5.8681 | .1390 | donor 경로 조사 |
| QB | B03 L000 | .8411 | 3.6117 | .1154 | off 없는 donor 경로 조사 |
| QB | B03 L1E4 S2025 | .8292 | 3.6239 | .1198 | 과거 Teacher 진단 |
| GF2 | B01 P0, Aligner 없음 | .9314 | .5854 | .0433 | 과거 raw 입력 기준 |
| GF2 | B02 N2 donor | .9670 | .6461 | .0119 | 높은 FR, RR 비용의 과거 사례 |
| GF2 | B03 L1E4 S2025 | .9526 | .5737 | .0225 | 과거 Teacher 진단 |
| GF2 | B04 L1E4 S1234 | .9500 | .5594 | .0261 | 과거 Teacher 진단 |

QB에서 donor 이후 FR 저하가 컸다는 사실은 준비점검의 우선순위를 높인다. 그러나 **현 QB의 물리적 misalignment가 더 크다는 증명도, consistency loss가 원인이라는 증명도 아니다.** L000도 나빴으며 데이터·phase·선택시점·A/U 공동적응이 섞여 있다.

기존 GF2 Teacher를 바로 쓰자는 초기 준비안보다 **이번 WV3 종결조건을 우선**한다. 새 기준을 공정하게 옮기려면 QB/GF2 모두 P0 fresh Teacher부터 시작한다. 과거 B03는 별도 diagnostic lineage다.

### 1.3 현재 실행 코드에서 확인한 이식 장애

확인한 것은 repository의 **FH12/FH20R1 경로**다. 다른 미조회 branch나 서버 로컬 구현이 이미 준비돼 있는지는 인증하지 않았다.

| 파일 | 확인된 센서 고정 부분 | 필요한 변경 |
|---|---|---|
| `fh12/model.py` | MS8 shape, `ms_bands=8`, 입력채널 계산의 +7, stem의 `-8:` 복사, 기본 backbone band8 | 명시적 C를 사용해 Teacher C+1/Student C+3, output C, A stem C로 일반화 |
| `fh12/losses.py` | Student/Teacher/GT `shape[1] != 8` guard | sensor contract C로 검사. loss reduction/detach/gradient routing은 그대로 |
| `fh12/data.py` | WV3 경로·NCHW8·max_pixel=2047만 허용 | C·DN·split 경로·N을 sensor manifest로 받음 |
| `fh12/evaluation.py` | RR20×8×256, FR20×8×512, `q8`, gain `wv3`, DN2047, 역변환1023.5 | C4/Q4, QB/GF2 MTF, maxDN/2, 각 sensor test identity로 매개변수화 |
| calibration/reference | FH12 local Teacher/reader/runtime를 강하게 묶음 | 같은 센서 shared reference를 읽기전용 import하는 새 소비 계약 |
| 기존 uploader/selector | WV3 탭과 목표값을 전제하는 부분 | sensor×server 라우팅·Q4·새 목표 policy·같은 checkpoint 기록 |

**데이터 경로와 YAML num_bands만 바꾸면 완료되는 포팅이 아니다.** 기존 shape 검사를 지우는 것이 아니라 센서별 올바른 검사로 바꾼다. C=8을 선택했을 때 기존 WV3 forward/loss/warp/metric이 보존되는지 회귀 검사를 수행한다. 이것은 WV3 신규 학습이 아니다. [S3–S7]

## 2. 유지·재측정·추후대조를 구분한다

| 분류 | 항목 | 첫 실행 |
|---|---|---|
| Method·구조 유지 | P0 Teacher W112D123 → PLH Student W104D122; A_ON; native M-frame | 유지 |
| 목적함수 유지 | Teacher rec+con; Student H/K/q-edge 및 q-hard A adjustment | 유지 |
| 계수·optimizer 유지 | alpha1,beta.1,lambdaE.002; Teacher A LR1e-5; Student A LR3e-6; U LR1e-4 | 유지 |
| 센서 정의에 따라 지정 | C, maxDN, band order, RR/FR source, MTF/Q4 | 검증 후 지정 |
| 반드시 재측정 | tau_R, q_ref, q cache, e/q/c/texture 분포 | 해당 sensor Teacher로 측정 |
| 후속 가설만 | A LR schedule, beta/edge scale, Teacher consistency 강도·probe/support | 기준 실험 후 한 축 대조. 자동 적용 없음 |

`tau_R`와 `q_ref`는 수치 목표에 맞춰 고르는 자유 계수가 아니다. 측정 원본을 보존한다. 별도 scale multiplier를 시험한다면 새로운 method/profile revision으로 분리하며 baseline을 덮어쓰지 않는다.

## 3. 데이터·방사 분포: 단위와 좌표를 먼저 확인한다

### 3.1 센서 계약

| 항목 | QB | GF2 |
|---|---|---|
| MS band C | 4 | 4 |
| Teacher P0 입력 | PAN1+MS4=5ch | 5ch |
| Student PLH 입력 | PAN1+LP1+HP1+MS4=7ch | 7ch |
| U output | 4-band residual | 4-band residual |
| A 입력 | native PAN1, upsampled MS4 | 동일 |
| nominal DN max | 2047 | 1023 |
| 입력 정규화 | 2DN/2047−1 | 2DN/1023−1 |
| 역정규화 | (y+1)×1023.5 | (y+1)×511.5 |
| train PAN/MS | 64/16 | 64/16 |
| RR / FR PAN | 256 / 512 | 256 / 512 |
| RR quality | 실제 Q4 | 실제 Q4 |
| 공식 FR MTF | QB preset | GF2 preset |

C·bit depth·명목 train count와 patch 크기는 첨부 논문의 데이터 표를 참고했지만, **현재 로컬 H5가 DN인지 [0,1]인지 이미 정규화됐는지는 실제 파일에서 확인**한다. 단순 observed max를 새 maxDN으로 사용하는 min–max 정규화는 하지 않는다. [S8,S9]

센서별로 raw/normalized p1,p5,p50,p95,p99, mean/std, zeros/invalid/saturation, 밴드간 공분산, PAN/LP/HP energy, edge energy와 저texture 비율을 기록한다. train·validation·RR·FR을 섞어 새 정규화 상수를 fit하지 않는다. FR은 구조·품질 진단에만 쓰고 reconstruction GT를 가정하지 않는다.

4밴드는 WV38의 임의 앞4개가 아니다. 각 H5의 band order/메타데이터와 MTF 배열 순서를 확인한다. 시각화 RGB 순서도 학습채널 순서와 별도 기록한다. PAN을 단순 mean(MS)와 동일 신호라고 간주해 차이를 전부 shift로 해석하지 않는다.

### 3.2 QB MS/GT phase 문제

`tools/repair_qb_ms.py`에는 **해당 배포본 QB train/valid의 약67–70% MS에서 LR1px/HR4px phase mismatch가 관측됐다는 기록**이 있다. 이 수치는 이번에 재측정한 것이 아니라 repository 감사 기록이다. 모든 QB 영상 자체의 물리적 오정합이라는 주장으로 확대하지 않는다. [S10]

현재 후보는 `train_qb_msfix.h5`, `valid_qb_msfix.h5`다. 사용 전 실제 source hash, 생성 MTF/phase, sample order, `GT/PAN unchanged`, LMS 재생성 규약을 검증한다. 존재하는 파일명만으로 PASS하지 않는다. 원본을 보존하고 mismatch가 이미 없는 데이터에 중복 복구를 하지 않는다.

**M=Up(MS)와 GT가 어긋나 있으면, PAN만 움직이는 A로 이를 해결할 수 없다.** 그러므로 이 문제는 학습 hyperparameter보다 먼저 해결할 데이터 계약이다. 원 test PAN/MS/GT를 모델 성능에 맞춰 이동시키거나 test GT로 input을 다시 생성하지 않는다.

추가로 repository 설명에는 `LR 1px / HR 4px`라는 표현과 decimation phase `(1,2)/(2,1)` 대 `(2,2)`가 함께 나온다. 이 표현만으로 현재 파일의 실제 이동량을 확정하지 않는다. **한 phase index 차이가 어느 좌표계에서 얼마의 이동을 뜻하는지 ramp·sample 위치와 실제 residual로 다시 확인**하고, 기록의 단위 표현이 부정확하면 별도 감사 결과로 정정한다. PAN Aligner의 synthetic 반경을 이 미검증 숫자에 맞춰 확대하지 않는다.

훈련 데이터 복구를 사용한 사실과 비교군의 데이터 규약 차이는 논문에 기록한다. `msfix`를 사용한 결과를 원 배포본 그대로의 재현으로 표시하지 않는다.

### 3.3 50K의 sample exposure

논문에 보고된 N과 B48을 단순 계산하면 50K의 명목 노출은 WV3 약247회/patch, QB 약140회, GF2 약121회다. 실제 batch stream/drop-last/중복 구조는 별도로 기록한다.

즉 같은50K도 센서별 데이터 coverage가 다르다. 그러나 첫 이식에서는50K를 유지하고, 결과가 부족하다는 이유만으로 QB/GF2만100K로 늘리지 않는다. 연장은 별도 schedule·budget 대조로 정의한다.

## 4. Reconstruction error calibration: tau_R 하나와 전체 분포

센서 D의 고정 Teacher를 T_D라 하면:

\[
e_{T,D}(i,p)=\frac1{C_D}\sum_b|Z_{T,D,i,b}(p)-Y_{D,i,b}(p)|,
\]
\[
\tau_{R,D}=\max(\operatorname{median}_{i\in I_D,p} e_{T,D}(i,p),10^{-6}).
\]

`I_D`는 해당 train에서 seed1234로 고정 선택한3072개 base patch다. 실제 index 배열을 보존한다. 생성한 reference를 사용하는 Student마다 다시뽑지 않는다. 표본수가3072보다 적다면 기존 프로토콜과 달라지므로 별도 승인이 필요하다.

Teacher A/U frozen `.eval()`, no augmentation, no_grad, FP32, **최종 HRMS 출력(잔차+MS base)과 GT**, 기존 full-support를 사용한다. residual만의 오차, patch median의 평균, RR test E, FR metric을 tau로 대체하지 않는다. [S5]

\[
d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\qquad
a_T=\operatorname{sg}\frac{[e_S-e_T]_+}{e_S+10^{-6}}.
\]

### 핵심 해석

- 같은 모델출력/정답의 오차 단위만 k배 바꾸고 tau도 k배면 d는 같다는 수학적 성질이 있다. 하지만 센서가 바뀌면 오차 분포·Teacher·texture가 달라져 단순 bit-depth 환산으로 새tau를 정할 수 없다.
- 예를 들어 가상 e_T=.003에서 WV3 tau=.0121186을 쓰면 d≈.198이지만, 그 센서의 measured tau=.003이면 d=.5다. 이는 예시이지 GF2 실제 tau의 측정값이 아니다.
- median을 맞춰도 upper tail, band별 잔차, low-texture 빈도와 Student advantage는 같아지지 않는다.
- C8→C4에서도 band **mean**은 유지한다. sum으로 바꾸거나 loss계수를2배로 보정하지 않는다.
- 현재 `d_T`는 GT reconstruction error를 이용한 difficulty다. 확률적 uncertainty나 예측 variance 자체로 바꾸어 부르지 않는다.

추가 report: e_T p1/p5/p25/p50/p75/p95/p99, per-band L1/MSE/bias, high/low-texture별 잔차, train-calibration/validation 차이, tau floor 사용 여부, e_S−e_T와 advantage 활성분율. diagnostic strata는 calibration sampling이나 loss의 band 가중을 묵시 변경하지 않는다.

## 5. Alignment: native shift, consistency error, reconstruction error를 분리

### 5.1 기존 q 정의를 유지해 새 센서에서 측정

\[
c_0=A_T(P,M),\quad c_j=A_T(W(P,\epsilon_j),M),
\]
\[
q_{i,r}=\frac1{2K}\sum_{j=1}^{K}
\|c_{i,r,j}+\epsilon_j-c_{i,r,0}\|_1,\quad K=16.
\]
\[
q_{\rm ref,D}=\operatorname{median}_{i\in I_D,r\in\{0,1,2,3\}}q_{i,r},\qquad
s_{i,r}=\operatorname{sg}\frac{q_{\rm ref,D}}{q_{\rm ref,D}+q_{i,r}}.
\]

r는 현재 실행 코드의 **fixed H/V flips 후 네 rotation** ID다. 임의의 random augmentation으로 바꾸거나 base q를 증강 view에 잘못 연결하지 않는다.

AXIS16은 radius [.25,.5,1,2]의 4방향, 현재 **HR PAN pixel** 단위다. train 전체×4 view를 cache한다. 명목 N에서 QB68556 view/GF279236 view이며, 실제 H5 N으로 확정한다. q_ref는 전체 train median이 아니라 위3072 calibration base ID×4의 median이라는 현재 규약을 유지한다. `q_median_full_train`은 진단으로 별도 기록한다. [S5]

### 5.2 q가 낮다고 native 정합의 절대오차가 낮은 것은 아니다

입력에 반응하지 않는 상수A는 현재 probe에서:

\[
q_{\rm const}=\tfrac12\operatorname{mean}(.25,.5,1,2)=.46875.
\]

또한 모든 prediction에 동일한 bias b를 더하면 q의 차분에서 b가 소거된다. 따라서 **q는 공통 native shift bias를 검증하지 못한다.** c의크기도 native displacement GT가 아니라 모델이 선택한 보정량이다. 상대 consistency와 reconstruction 목표가 제공하는 신호를 구분한다.

자기 q_ref로 정규화하면 median 위치의 s는 항상.5다. 따라서 센서마다 s가 비슷하다고 같은 정합정확도를 갖는다고 읽지 않는다. absolute q, radius별 residual/radius, 실제 `Δc` 대 `−epsilon` slope와 cross-axis response를 함께 본다. 이상적 slope−1, 상수A slope0은 현재 부호 규약에서의 수학적 기준이다.

`q_ref≈.46875`만으로 q를 uniform으로 대체하지 않는다. 유한 positive q_ref이면 baseline을 유지하고 진단 warning을 남긴다. q_ref=0은 현재 정의가 허용하지 않는 수치/프로토콜 상태이므로 임의epsilon으로 수선하지 말고 보고한다.

### 5.3 어느 정도의 misalignment를 볼 것인가?

처음에는 **추가 synthetic radius2와 probe grid를 유지**한다. 그것은 실제 native shift의 상한이 아니다. 현재 A의 마지막 layer는2차원 linear이고 c를±2로 제한하지 않는다. 센서 GSD를 근거로 c/probe를 무조건2배·4배로 환산하지 않는다. [S11]

실제 진단:
1. native PAN–MS를 공통공간 대역으로 비교하되 분광차이·저texture·반복패턴에 대한 신뢰도도 기록한다. proxy shift를 GT로 취급하지 않는다.
2. MS–GT downsampling consistency를 native PAN–MS 상대위치 문제와 분리한다.
3. c의 mean/median/p95/p99/max, dy/dx방향성, warp source의 border sampling 비율, scene내 crop별차이를 기록한다.
4. crop64/128와 fullRR256/FR512의c/q를 비교한다. U-Net은full image를 유지한다.
5. 큰 native shift·국소방향 불일치가 반복되면 현재global translation 근사의 한계로 기록한다. dense-flow나 새aligner로 자동 교체하지 않는다.

Aligner input z-score·GroupNorm·GAP의 context 통계도 크기에 따라 달라질 수 있다. raw variance 대비 epsilon 비율과 low-texture view의 반응을 기록하되, 더 좋은 test metric이 나오는 norm으로 즉시 교체하지 않는다.

## 6. LPAN/HPAN 생성과 sensor MTF를 혼동하지 않는다

첫 baseline은 WV3의 **알고리즘용 LP 레시피**를 유지한다. Gaussian sigma=1.98, kernel=41, replicate padding, `[2::4,2::4]` decimation이며 생성은 float64, cache는 FP32다. LP를 HR로 올릴 때에는 bicubic과 `align_corners=False`를 사용한다. 하지만 **cache bytes는 각 센서의 실제 PAN으로 새로 생성**한다. 이 고정 Gaussian이 모든 센서의 물리적 MTF라는 뜻은 아니다. [S4]

반대로 RR 데이터 감사와 FR metric은 각 센서의 공식 MTF preset과 band order를 사용한다. **입력 feature의 LP 연산, Wald 데이터 축소, FR 품질 평가의 filter는 서로 다른 역할**이다. WV3 gain으로 QB/GF2를 평가하지 않는다.

동기 forward는 다음이다.

\[
M=U_4(MS),\quad L=U_4(LPAN),\quad c=A(P_{m4},M_{m4}),
\]
\[
\widetilde P=W(P,c),\quad \widetilde L=W(L,c),\quad
\widetilde H=\widetilde P-\widetilde L,\quad
Z_S=M+F_S([\widetilde P,\widetilde L,\widetilde H,M]).
\]

C=4에서 Student 입력은 7채널이며 A stem은 PAN1/MS4다. LP를 LR 크기인 상태로 HR 단위 c만큼 이동시키지 않는다. H는 signed 차이이므로 abs, clipping, 추가 영상 정규화를 하지 않는다. 같은 grid에서 `W(P−L,c)=W(P,c)−W(L,c)`이지만, filtering/downsampling과 warp까지 모두 교환 가능한 것은 아니다.

이미 LP에 있는 phase 차이를 같은 c가 저절로 수정하는 것도 아니다. Impulse/ramp 시험과 source correspondence를 기록한다. P/L/H는 정보 중복이 있으므로 각각 독립적인 새 관측 밴드로 해석하지 않는다.

후속 sensor-MTF LP 실험은 baseline 이후 필요할 때만 별도 input-recipe revision으로 정의한다. Metric preset과 동시에 바꾸거나 원 cache를 덮어쓰지 않는다.

## 7. Method 불변과 분포에 따른 실제 gradient

### 7.1 Teacher

\[
L_T(t)=\operatorname{mean}|Z_T-Y|
+\mathbf1[t\bmod2=1]10^{-4}\operatorname{mean}|c_\epsilon+\epsilon-\operatorname{sg}(c_0)|.
\]

t는 0-based optimizer update다. ε는 radius=2 HR-pixel 원판에서 면적 균등하게 생성하며 별도 RNG를 사용한다. Synthetic PAN은 A-only branch에만 들어가고 U에는 전달하지 않는다. Native reconstruction은 A/U 모두 갱신하며 native c를 detach하지 않는다. 마지막 A head만 0으로 초기화하고 A 전체를 0으로 만들지 않는다.

**현재 주 경로에는 N2 donor가 없다.** 이후 성능이나 q가 부족하다는 이유로 donor를 묵시적으로 로드하지 않는다.

### 7.2 Student

\[
\ell_{H,i}=\operatorname{mean}_p[(1+d_T)e_S],
\quad
\ell_{K,i}=\operatorname{mean}_p[0.1(1-d_T)a_T\operatorname{mean}_b|Z_S-Z_T|],
\]
\[
L_U=\operatorname{mean}_i[\ell_{H,i}+\ell_{K,i}+.002s_i\ell_{E,i}],
\quad L_A=\operatorname{mean}_i[s_i\ell_{H,i}].
\]

Teacher A/U는 frozen, Student A는 독립 clone이자 trainable이고 U는 fresh다. GT, error-derived weight, q weight는 detach하지만 e_S와 soft discrepancy는 미분 가능해야 한다. Edge는 최종 HRMS/GT의 signed Scharr 차이이며 PAN/HP를 정답으로 바꾸지 않는다.

U/A gradient를 별도로 계산한 뒤 optimizer step을 진행한다. Student A에 KD, edge, consistency를 직접 주지 않는다. **KD·edge·q·adjustment를 제거하는 case는 첫 준비 범위에 없다.** [S7,S12]

### 7.3 같은 계수라도 실제 학습 비중은 달라진다

τ 재calibration은 difficulty의 오차 scale을 맞추지만, Teacher의 con/rec와 Student의 edge/hard gradient 균형을 자동으로 같게 만들지는 않는다. 특히 consistency는 pixel 좌표 단위이고 reconstruction은 영상 오차 단위다. 영상 오차 분포가 달라지는 문제를 하나의 정규화 상수로 모두 해결할 수는 없다.

반대로 **L1 scalar loss가 작다는 이유만으로 gradient도 작다고 단정하지 않는다.** 실제 gradient norm을 읽어야 한다.

고정 train 진단 표본에서 10K/24,240/50K에 다음을 기록한다.

- Hard/soft/weighted-edge 크기, U gradient norm, 보조항과 hard의 gradient cosine.
- A gradient norm, Teacher의 consistency 활성 update에서 weighted-con/rec gradient norm 비율.
- e_S−e_T, advantage>0 비율, soft weight 평균, d/s 분포, epsilon이 지배하는 오차의 비율.
- Frozen Teacher의 state hash, cache/온라인 계산 일치, 진단 RNG 격리.

Soft가 후반에 작아지는 것은 Student가 Teacher보다 정확해져 gate가 꺼진 정상 동작일 수 있다. Controlled synthetic input에서는 gradient 경로의 존재를 검사하고, 실제 학습에서 자연스럽게 0인 항을 구현 실패로 오인하지 않는다.

## 8. 서버 배정안: QB 3대, GF2 2대

이 표는 현재 프로세스를 조회해 즉시 적용한 배정이 아니다. GPU/VRAM과 기존 run 상태는 QG00에서 확인한다. QB의 과거 donor 문제와 전처리 위험에 추가 자원을 배정한 안이며, 실제 QB의 native shift가 더 크다는 가정은 아니다.

| 서버 | 센서·reference | 준비·Stage 1 | 첫 Student | 역할 |
|---|---|---|---|---|
| s4 | QB TA | 새 P0 W112D123 T81001 → exact50K calibration | PLH W104D122 S82001, 82002 | QB 주 파이프라인 |
| s3 | GF2 TA | 새 P0 W112D123 T91001 → exact50K calibration | PLH W104D122 S92001, 92002 | GF2 주 파이프라인 |
| s1 | QB TB | 포팅 회귀·QB data/q 감사; 독립 새 Teacher T81002 | PLH W104D122 S82001, 82002 | 두 번째 Teacher reference의 민감도 |
| s2 | QB TA 공유 | QB legacy/metric/source·동기 frontend 검증, TA 준비 후 import | PLH W104D122 S82003 | 같은 Teacher의 Student 반복 |
| s5 | GF2 TA 공유 | GF2 DN/MTF/LP/JQM 검증, TA 준비 후 import | PLH W104D122 S92003 | 같은 Teacher의 Student 반복 |

기본 제안은 **Teacher 3개 + Student 8개 = 11개 50K 학습**이다. 각 센서의 주 reference TA에 Student 3개 seed를 확보하고, QB TB의 2개 seed는 별도 집단으로 보관한다. TA/TB가 다른 서버에서만 관측되면 차이는 Teacher와 서버 환경을 함께 포함한다. 이를 Teacher의 인과효과로 단정하지 않는다. 필요하면 같은 checkpoint 재평가로 평가환경 차이를 먼저 분리한다. 추가 same-server Teacher 대조는 차후 예산으로 정한다.

**WV3 best Student seed73101을 다른 센서의 최적 seed로 가정하지 않는다.** 신규 seed는 식별·재현 규칙이지 성능 선별 규칙이 아니다.

### 기다림과 reference 무결성

다섯 서버의 공통 성능 결정이나 recipe lock을 두지 않는다. QB는 GF2를, GF2는 QB를 기다리지 않는다. 다만 **s2/s5가 사용할 Teacher와 calibration 완료는 실제 필수 의존성**이다. 이것까지 없다고 표현하지 않는다. 준비 중에는 데이터·metric·legacy·코드 검증을 진행한다.

TA 패키지가 완료되면 checkpoint/config/source/data/LP/normalization/calibration indices/τ/q/cache를 읽기 전용으로 복사·검증하고 Student에 연결한다. 임의로 다른 Teacher를 고르거나 seed마다 Teacher를 바꾸지 않는다. Cache가 다르면 표시만 같은 reference로 사용하지 않는다.

Teacher가 PAN-Crafter를 넘지 못했다는 이유만으로 Student를 막지 않는다. 오류·non-finite·잘못된 데이터/좌표·누락 reference는 차단하고, 낮은 성능·높은 q는 warning과 분석 대상으로 구분한다.

## 9. 목표와 평가

| Dataset | HQNR 선행조건 | 후순위 목표 | 출처·해석 |
|---|---:|---:|---|
| QB | raw HQNR > .920 | ERGAS < 3.570 | PAN-Crafter Table 8 보고값 대비 초과 |
| GF2 | raw HQNR > .964 | ERGAS < .552 | Table 7 보고값 대비 초과 |
| GF2 stronger record | 동일 | ERGAS < .522 | 본문 Table 2의 별도 보고값도 넘는 기준 |

GF2 .522와 .552는 원문 안의 불일치이며 여기서 오타로 단정하지 않는다. 두 조건을 모두 기록한다. H를 넘은 후보 안에서 E를 최소화하고, 동률은 SCC/PSNR/낮은 step 순이다. 원본 정밀도를 사용하며 성능을 보고 나서 하한을 낮추지 않는다. 위 목표는 **PAN-Crafter의 보고값 비교**이지 현재 모든 연구의 SOTA 인증이 아니다. [S8]

유지할 선택점은 RAW_MAX/TARGET/EXACT50K/RR_VAL_SELECTED/E_MIN_DIAG50다. TARGET/E_MIN은 test-aware 개발 결과임을 명시한다. 후보는 1010, 2020, …, 49490, 50000의 50개다. 센서별 epoch 길이가 달라도 update 격자는 같다. 적격 후보가 없으면 `no_eligible`과 빈 Target을 보존한다.

평가자는 센서에 맞는 C/maxDN/MTF/Q4/RR GT/FR source를 받되 다음 규약을 유지한다.

- 공식 RR는 기존 support `20:-21`과 Q block32를 먼저 재현한다.
- 공식 FR는 native PAN/full512와 원 reference LMS로 평가한다. A의 c에 맞춰 PAN reference를 이동하거나 test mask를 추가하지 않는다.
- 모델의 MS base는 bicubic(MS), 평가 reference LMS는 정의된 원 LMS다. 둘이 자동으로 같다고 가정하지 않는다.
- HQNR는 장면별 `(1−Dλ)(1−Ds)`의 평균이다. 평균 Dλ와 평균 Ds의 곱으로 대체하지 않는다.
- JQM이 SRF-substitute 방식이면 variant와 추정자료를 기록하고 SIPSA와 동일한 JQM으로 표시하지 않는다.
- Params/FLOPs/infer/memory는 C4 모델 전체에서 재측정하고 MAC/FLOP 관례를 명시한다.

## 10. Sheet 구성

기존 탭의 과거 결과를 보존하고 신규 QGBASE namespace로 추가한다.

| 서버 | 업로드 tab |
|---|---|
| s1 | QB-s1 |
| s2 | QB-s2 |
| s3 | GF2-s3(5090) |
| s4 | QB-s4 |
| s5 | GF2-s5 |

현재 준비된 QB-s4/GF2-s5 형식이 최신 WV3의 모든 추가열을 이미 갖췄다고 가정하지 않는다. **현재 WV3의 의미별 header mapping을 재조회해 필요한 열만 확장**한다. 옛 88열/122열/246열 같은 고정 열번호를 복사하지 않는다.

RR/FR/Cost, 선택점별 metric/step/SHA, Sensor/Teacher ID/TS/SS/C/maxDN/LP phase/τ/q/cache/hash/method revision/actual updates/status를 기록한다. Q8→Q4는 표시뿐 아니라 실제 계산 key도 바꾼다. NOA/V64/JQM 미측정값은 빈칸이다. Upsert key는 `(sensor,campaign,run)`이며 요약행에는 selector와 source hash까지 명시한다.

## 11. 첫 실행이 낮을 때의 전략

**첫 번째 목적은 WV3에서 고정한 Method를 센서별 calibration으로 정확히 이식한 baseline을 완성하는 것**이다. 처음부터 여러 loss 축을 섞지 않는다.

| 관측 | 먼저 확인할 것 | 다음 단계: 아직 실행하지 않은 대조 |
|---|---|---|
| C·정규화·phase·metric 오류 | QG01–05, QG15 | 오류 수정 후 새 revision으로 재실행 범위를 정의한다. 튜닝으로 덮지 않는다. |
| q가 상수 반응 기준에 가깝고 c response가 미약함 | Cache/온라인 parity, sign/units, texture, Teacher con/rec gradient | Baseline 보존 후 Teacher 학습계수·초기화 중 한 축을 대조한다. Consistency를 제거하지 않는다. |
| H 통과, E 부족 | VAL/Exact50K, band-relative MSE, soft 활성, 학습 coverage | β 또는 학습 길이 중 한 축만 같은 seed에서 대조한다. Architecture 재탐색은 뒤에 둔다. |
| E가 좋으나 H 미달, 후반 Ds 상승 | 같은 run의 A/U 교차, c 추세, 정규화 context | 증거가 있을 때 A LR schedule만 대조한다. A는 trainable이며 loss는 유지한다. |
| Dλ 손실과 HP/edge 영향 정황 | LP phase, band order, edge/U gradient, PAN–MS 분광 관계 | Edge 계수 또는 input recipe 중 한 축을 대조한다. 첫 PLH baseline은 삭제하지 않는다. |
| 공동목표 달성 | 같은 조건의 남은 seed와 장면별 분포 | 큰 새 grid보다 반복성과 기록을 우선한다. |

튜닝은 **한 축·두 값·같은 Teacher/seed/50K·다른 seed 확인**을 기본으로 한다. 학습 길이를 바꾸는 대조만은 그 축의 정의에 맞게 update 수와 scheduler를 별도로 기록한다. 결과가 부족하다는 이유로 τ/q를 목표 성능에 맞춰 조정하거나 KD·edge·q·A를 끄지 않는다. 현재는 측정값이 없어 새로운 최적 계수를 정할 근거가 없다.

## 12. 실행 준비 완료와 예산

별첨 checklist 20개에서 P0는 정의·자산·수치 무결성이고 P1은 분석이다. P1 미완료가 다른 센서 전체의 대기로 전파되지 않게 한다. P0 손상은 영향을 받는 경로만 중지한다.

최초 가동 전에 다음을 확정한다.

1. 현재 서버/GPU·데이터·디스크 inventory와 WV3 drain 기록.
2. QB msfix/FR mat20, GF2 DN·MTF·LP source 정의.
3. 4밴드 전체 경로의 smoke와 8밴드 회귀 PASS.
4. 명시적인 Teacher/Student run ID·dependency·Sheet route. Reference 숫자는 측정 전 null로 둔다.
5. 이번 새 작업의 시간정책. 위 11개 run은 기준 cohort 제안이며 처리시간은 아직 측정하지 않았다.

총시간은 데이터 준비 + Teacher + calibration + Student50K·공식 평가 + import·backup으로 산정한다. QB/GF2의 N은 WV3보다 크므로 τ 표본 수가 같아도 full-train q cache 비용은 증가할 수 있다. 실제 warmup 이후 throughput과 첫 postrun으로 예약값을 정한다. Validation과 평가비용을 빼고 학습시간만으로 완료 개수를 보장하지 않는다.

## 13. 준비 산출물·출처

### 제공물

- 본MD: 준비검토·Method 계약·서버배정·수치/분포진단·평가정의.
- `PAN_QB_GF2_Preflight_Checklist_2026-09-20.csv`: 담당/시점/완료기준/실패조치가있는20개작업.
- `PAN_QB_GF2_Baseline_Proposal_2026-09-20.csv`: Teacher3+Student8의명시적제안ID. YAML/runner등록은아님.
- `PAN_QG_Preparation_LegacyAssets_2026-09-20.csv`: 이번Sheet에서읽은역사적B01–B04자산.
- `PAN_QG_Preparation_Source_2026-09-20.xlsx`: 읽기용Sheet snapshot.

Snapshot SHA256: `26c2ccf760038c2af42aac3ae76e95d964e8f2dea216059ccc67cf02beb9a330`

### 근거

[S1] 첨부 `PAN_WV3_Closeout_Review_FrozenRecipe_2026-09-20.md` 및 `PAN_WV3_FrozenRecipe_2026-09-20.json`, 특히 종결조건/Teacher/Student/Method 불변 부분.  
[S2] Live `pan-cvpr27` 이번 export; QB-s1 rows7–11, GF2-s1 rows11–15, QB-s4/GF2-s5 준비tab.  
[S3] `hojunking/PAN-Crafter-repro/fh12/model.py`, 이번main조회 blob `5d15ecd0b245c69026da422d96ae231ce854d968`.  
[S4] `fh12/data.py`, blob `8d56ee6dd8e139ff0e8f224488d3482aebe7338c`.  
[S5] `fh12/calibration.py`, 이번main조회; median3072/errorfullpixel, all train×4AXIS16, subset×4q_ref.  
[S6] `fh12/evaluation.py`, blob `1e5977117878b865a5c0155f8d9b793965df30e1`.  
[S7] `fh12/losses.py`, blob `541645adf666138d8ce859a9a236e5065213214c`.  
[S8] 제공 `pancrafter.pdf`: p6 dataset/training/metrics; p7 Table2; p13 Tables7–8. GF2 .522/.552 불일치 그대로 기록.  
[S9] 제공 `uknowdiff.pdf`: p6 Table1 C/bit/N/GSD/patch. τ=1은그논문의다른계수이며본tau_R와다름.  
[S10] `tools/repair_qb_ms.py`, blob `623acaec230b3e256eba85d24dfa94d56f39e04a`; 배포파일audit기록이지이번로컬재검증결과는아님.  
[S11] `pa/aligner.py`, blob `e9dd406ae7f0a2dc491273e21b38f8e450dfa28d`.  
[S12] 제공 `Pansharpening_research.pdf`: pp8–10 Teacher native+con, p11 q/e구분, pp12–15 KD/edge/Agradient구분.

**현재 확인하지 않은 것:** 서버 raw H5 actual DN/phase/hash, GPU작업상태, 신규 sensor Teacher/τ/q/c분포, 포팅의 실행 PASS, 신규 Sheet 업로드. 이 항목들은 측정·검증 전이며 완료로 표시하지 않는다.
