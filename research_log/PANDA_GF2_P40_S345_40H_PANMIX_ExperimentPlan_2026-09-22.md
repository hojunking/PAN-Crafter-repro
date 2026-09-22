# PANDA GF2 P40 — PANMIX 전이·강도·학습 경로 심층 검정 (s3–s5, 40시간)

- **Campaign:** `PANDA_GF2_P40_S345_20260922_v1` / short ID `GFP40`.
- **작성일:** 2026-09-22.
- **대상:** GF2, s3·s4·s5의 기존 RTX 5090 환경. s1/WV3와 s2/QB ablation은 변경하지 않는다.
- **예산:** 실제 공통 `t0`부터 40시간 wall-clock. 병렬 장비 점유 상한은 3대×40h=120 GPU-hour이며, 준비·전환·cache/calibration·평가·I/O·대기도 창에 포함한다.
- **근거:** 2026-09-22 08:36 KST B20 결과 분석 [S1] 및 B20 설계 [S2]. 작성 중 s5 `A1:F15`, `BB1:BL15`를 다시 읽었으며 완료 목록은 C01–C07, C08은 여전히 미등록이었다. 원격 process나 미업로드 결과는 조회하지 않았다.
- **문서의 성격:** 새 실험 설계·registry. 실제 학습 시작, 원격 queue 변경, 코드 배포, Sheet 쓰기는 하지 않았다. 기존 `gfb20`의 20h 상수를 40으로 고쳐 바로 실행하는 파일이 아니다.

## 0. 이번에 바뀌는 우선순위

**PANMIX 하나를 중심으로 40시간을 사용한다.** 새 Teacher 생산, H010·K1·E100의 추가 계수 탐색, PAN reconstruction task, backbone 변경은 포함하지 않는다. 세 서버가 자기 기존 Teacher를 유지하므로 Teacher 전송을 기다리지 않는다.

이번에 답할 질문은 네 가지다.

1. **전이:** s5의 한 Teacher 아래에서 관측한 개선이 s3·s4의 다른 Teacher/reference에서도 나타나는가?
2. **강도:** 고주파 gain 폭과 native 입력 비율 중 무엇이 성능을 결정하는가? 단순히 강하게 증강할수록 좋은가?
3. **학습 경로:** native 학습 후 짧은 PANMIX가 좋은가, 처음부터 적용해야 하는가, 다시 native로 학습하면 효과가 사라지는가?
4. **검증:** 결정한 단일 recipe를 사전 고정한 새 Student seed에서 사용했을 때 같은 방향의 개선과 공동목표가 반복되는가?

| 서버 | 주요 역할 | 우선 등록 학습 구간 | 선택적 3번째 확인 seed | 등록 상한 |
|---|---|---:|---:|---:|
| **s3** | R3 전이, FT20K/60K, 새 seed 확인 | **16** | 3 | **19** |
| **s4** | R4 전이, gain 폭·native 비율, 새 seed 확인 | **28** | 3 | **31** |
| **s5** | R5 재검정, 증강 방향, native/PANMIX 2×2 학습 경로, 새 seed 확인 | **30** | 3 | **33** |
| **합계** | 기존 Teacher 3개 재사용, 새 Teacher 학습 0 | **74** | 9 | **83** |

**74개는 독립 모델 74개가 아니라 실행할 학습 구간 수**다. 공통 fresh100K trunk와 그 뒤의 두 개 또는 네 개20K fork를 별도 학습 구간으로 센다. 우선 등록 74개는 fresh100K 10개, FT20K 60개, FT60K 4개이며 총 추가 optimizer update는 2,440,000이다. 선택적 9구간까지 모두 실행하면 2,860,000이다. 최대 수는 전량 완료 보장이 아니다. 특히 s5의 선택적 3번째 확인 seed는 초기 시간 추정상 들어가지 않는다.

## 1. 관측한 사실과 새 가설의 구분

### 1.1 PANMIX에서 이미 관측한 것 [S1]

| 원 Student 부모 | Native+mixed-calibration 대조 | PANMIX | 변화 |
|---|---|---|---|
| s5 S95002 | H 0.951770 / E 0.547873 / Ds 0.028004 | **H 0.958158 / E 0.548328 / Ds 0.022500** | ΔH +0.006387, Ds −19.65%, E +0.083% |
| s5 S95001 | H 0.948229 / E 0.550164 / Ds 0.031624 | **H 0.958131 / E 0.551180 / Ds 0.022870** | ΔH +0.009901, Ds −27.68%, E +0.185% |

이는 같은 R5_100 Teacher에서 서로 다른 두 부모의 **후기20K**에 대한 결과다. Calibration scale만 바꾼 대조의 H 변화는 약 10^-6 수준이었다. PANMIX의 fresh100K 효과, 다른 Teacher 전이, 물리적 정합 정확도와의 관계는 아직 미확정이다.

H010은 후기5 pair에서 E만 조금 개선했고 H/Ds는 악화했다. K1도 공간 이득이 없었으며 E100의 첫 fresh100K는 H −0.003642였다. 따라서 이번 예산을 그 축의 재탐색에 쓰지 않는다 [S1].

### 1.2 이번에 처음 검정하는 것

- gain 폭 `.125 / .25 / .5`, native 비율 `.25 / .5 / .75`의 차이.
- 낮추기만/높이기만 하는 PAN gain과 균형 증강의 차이.
- 동일 120K 학습량과 동일 optimizer reset 경계를 갖는 `N→N`, `N→M`, `M→N`, `M→M`.
- FT60K의 최종 균형. 이것은 기존20K 이후 단순 이어 학습이 아니라 **다른 cosine horizon을 가진 별도 run**이다.
- 하나로 잠근 PANMIX 분포가 3개 reference·새 Student seed에서 반복되는지.

어느 항목도 이미 성공한 값으로 서술하지 않는다. HQNR 상승만으로 실제 정합이 정확해졌다거나 FR에 GT가 있는 것처럼 주장하지 않는다.

## 2. 불변 method와 평가 계약

| 항목 | 고정 |
|---|---|
| Teacher | **P0 / W112 / D=[1,2,3]**, 기존 로컬100K A/U, frozen |
| Student | **PLH / W104 / D=[1,2,2]** |
| Aligner | `PANGlobalAligner`, PAN1+upsampled MS4만 입력, margin4, sample-wise `(dy,dx)` |
| Task/기능 | 단일 HRMS. Attention, mode modulation, MARs, PAN reconstruction **OFF** |
| 데이터 | 기존 GF2 C4/maxDN1023, train19,809, 기존 val/RR20/FR20, source SHA 고정 |
| 입력 크기 | train PAN64/MS16/GT64; 기존 RR PAN256, FR PAN512 |
| 정규화 | `2DN/1023−1`, per-image min–max/clip/abs/rounding 추가 없음 |
| Forward | 고정 MS frame, MS base 한 번; PAN/upLP 같은 bicubic grid, HP=aligned PAN−aligned LP |
| Warp | FP32/bicubic/border/align_corners=False, correction clamp 또는 임의 rescale 없음 |
| LP | 기존 σ1.98/k41/replicate/[2::4,2::4], float64 생성→float32 cache |
| Student loss 계수 | **α=1, β=.1, λE=.002**를 모든 case에서 유지 |
| Gradient | **U←hard+selective soft+q-edge, A←q-hard만**, 같은 forward에서 recipient별 미분 |
| Precision/optimizer | FP32, AMP OFF, AdamW β=(.9,.999), eps1e−8, wd=.01 |
| 공식 평가 | 기존 native PAN/original LMS, RR20·FR20, Q4, DN1023, 기존 RR20:-21/full FR512 규약 |
| 추론 | gamma1 원본 입력. Teacher·q·e·augmentation 제거; 기존 Student A/U만 사용 |

`dT`, Student-dependent `aT`, `s`는 detached weight이고 eS와 Student–Teacher discrepancy는 live다. Scharr는 기존 /32, 방향 각각0.5 평균, edge 경계1px 제외를 유지한다. qref의 모집단은 전체 train median이 아니라 calibration3072×geometry view×분포 token이다 [S2,S3].

**입력 증강과 calibration protocol만 확장한다.** 이를 WV3/QB와 다른 backbone으로 표현하지 않으며, native-only 학습과 완전히 같은 training protocol이라고 숨기지도 않는다. 원 영상의 MS/GT 위상·평가 MTF를 바꾸거나 test 입력을 재생성하지 않는다.

## 3. 부모와 reference: 기존 native100K에서 시작한다

| Parent | 서버 | 원 Student | 사용 endpoint | Reference |
|---|---|---|---:|---|
| P3HI | s3 | L100I1-S04, SS95002 BASE | 100000 | R3_100, TS94001 |
| P3LO | s3 | L100I1-S03, SS95001 BASE | 100000 | R3_100, TS94001 |
| P4A | s4 | L100I1-S07, SS95001 BASE | 100000 | R4_100, TS94002 |
| P4B | s4 | L100I1-S10, SS95002 BASE | 100000 | R4_100, TS94002 |
| P5B | s5 | L100I1-S14, SS95002 BASE | 100000 | R5_100, TS94003 |
| P5A | s5 | L100I1-S11, SS95001 BASE | 100000 | R5_100, TS94003 |

모두 **B20 이전 native BASE100K의 U+A**다. 이미 PANMIX로 fitting한 C03/C04를 새 강도 실험의 기본 부모로 쓰지 않는다. 그래야 gain/비율별 실험을 같은 출발점에서 비교할 수 있다. 높은 부모와 낮은 부모를 함께 사용하고, 성능을 보고 부모를 바꾸지 않는다.

`GFP40_ParentAssets.json`에 원 run ID·보고된 checkpoint SHA·원 reference와 출처 셀을 넣었다. 보고된 SHA는 로컬 bytes를 이번 문서에서 검증한 결과가 아니다. 실제 경로와 누락 SHA는 실행 환경에서 바인딩한다. Teacher/reference가 없으면 그 서버의 해당 block을 보류하고, 다른 Teacher로 자동 대체하지 않는다.

### 3.1 Source·실행 전 감사

직전 B20 결과가 기록한 source commit `060ba8d209925854f1f3c30496c6366c4cb09016`은 당시 GitHub fetch에서 조회되지 않았다. 접근 가능한 `5d9f4d...`와 동일하다고 가정하지 않는다 [S1].

실행 담당자는 B20 frozen checkout 또는 source archive를 확보해 **모델·loss·warp·LP·data·metric 핵심 파일의 SHA**와 P40 변경 diff를 남긴다. 원격 GitHub에서 commit을 찾지 못했다는 이유만으로 모든 자산을 버릴 필요는 없지만, 접근 가능한 다른 release를 조용히 같은 실행 source로 취급해서는 안 된다.

각 서버에서 다음을 통과한 뒤 학습한다.

1. 원 parent U/A bytes, Teacher A/U, data/LP, native calibration/q-cache identity 확인.
2. native parent 재평가와 보존된 예측/지표 대조. 수치 tolerance는 실행 전 고정하며, 실패 후 점수에 맞춰 완화하지 않는다.
3. C4 구조·정규화·LP·gradient-routing smoke와 parent-loading hash 확인.
4. 새 augmentation은 train에만 작동하고 val/RR/FR gamma1은 원 PAN·LP를 직접 반환하는지 확인.
5. Teacher frozen hash, Student A trainable 여부, fresh와 FT 초기화의 차이 검증.

B20 parity 규칙과 tolerance를 그대로 재사용할 수 있는 경우 우선 재사용한다. 그렇지 않으면 같은 모델의 반복 실행으로 수치오차를 측정해 P40 tolerance를 **결과 분석 전에** 잠그고 기록한다. 숨은 부분 mask·밴드 삭제·19-scene fallback은 허용하지 않는다.

## 4. 공통 augmentation 및 calibration

### 4.1 PAN 생성

\[
P_\gamma=G(P)+\gamma(P-G(P)).
\]

G는 native PAN 격자의 separable Gaussian **σ=1 HR pixel, k=7, reflect**다. Float64 원 DN에서 생성하고 float32 cache한다. γ=1은 수식으로 재생성하지 않고 **원본 tensor를 직접 반환**한다. 이는 물리적 MTF 보정의 주장이 아니라 학습 view 변화다.

증강 PAN에서 기존 LP 알고리즘으로 LPγ를 다시 만든다. 원 source patch에서 gain·LP를 생성한 뒤 **기존 fixed-HV→ROT4**를 PAN/MS/LP/GT에 동일하게 적용한다. PAN/LP를 서로 다른 γ로 만들거나 LP 생성 순서를 뒤집지 않는다. γ>1에서 범위가 넘는 화소는 overshoot 진단에 기록하고 임의 clamp하지 않는다.

Teacher와 Student는 같은 augmented PAN/MS pair를 보고, 각자 자기 Aligner의 correction으로 복원한다. Teacher를 증강 데이터로 다시 학습하지 않는다. Student dT/aT는 그 update의 동일 view Teacher/Student/GT로 계산한다.

### 4.2 사전 등록한 분포

| ID | γ support | 확률 | native 비율 | 변경 축 | E[(γ−1)²] |
|---|---|---|---:|---|---:|
| **G025** | .75,1,1.25 | .25,.50,.25 | .50 | B20 incumbent | .03125 |
| **G0125** | .875,1,1.125 | .25,.50,.25 | .50 | gain 폭 절반 | .0078125 |
| **G050** | .5,1,1.5 | .25,.50,.25 | .50 | gain 폭 두 배 | .125 |
| **P075** | .75,1,1.25 | .125,.75,.125 | .75 | native 비율 증가 | .015625 |
| **P025** | .75,1,1.25 | .375,.25,.375 | .25 | native 비율 감소 | .046875 |
| **LOW** | .75,1 | .50,.50 | .50 | 낮추기만 하는 대조 | .03125 |
| **HIGH** | 1,1.25 | .50,.50 | .50 | 높이기만 하는 대조 | .03125 |

G025/LOW/HIGH는 **native로부터의 평균 제곱 변화량**과 증강 발생 비율이 같다. 그러나 E[γ]는 각각 1/.875/1.125이고, γ의 분산은 같지 않다. 따라서 이를 ‘분산까지 완전히 동일한 방향 대조’라고 쓰지 않는다. LOW/HIGH는 방향/평균 이동 기전을 묻는 진단이며, 이번 자동 최종 후보에는 넣지 않는다.

σ·kernel 크기는 이번에 튜닝하지 않는다. gain 폭·확률·시점을 먼저 분리한 뒤 남는 문제를 다음 revision으로 넘긴다.

### 4.3 각 family에 맞는 native 대조

- **NATIVE0:** native 입력 + 원 native τ/qref/q.
- **CTRL(f):** native 입력 + family f의 τf/qref_f + native γ=1의 q.
- **MIX(f):** family f 입력 + **같은** τf/qref_f + 실제 γ에 대응하는 q.

`MIX(f)−CTRL(f)`를 그 family의 **입력 view 및 view-dependent supervision 효과**로 읽는다. `CTRL(f)−NATIVE0`는 scale 효과다. `MIX(f)−MIX(G025)`는 각자의 scale까지 포함한 **전체 augmented recipe 차이**이며 입력만의 효과라고 축소하지 않는다.

### 4.4 Teacher별 q와 exact mixed median

기존 calibration3072 base ID/seed1234, AXIS16 반경 .25/.5/1/2의 축4방향, batch16/FP32를 유지한다.

- τf: geometry 무증강 calibration3072의 **전체 pixel band-mean |Teacher−GT|**를 해당 γ의 정수 token multiplicity로 pooling해 median, floor1e−6.
- qref_f: 같은3072 ID×ROT4×γ token의 q median.
- q의 probe는 **Warp(Pγ,ε)**다. gain(Warp(P,ε))로 바꾸지 않는다.
- 실제 q-cache는 해당 Teacher의 전체 train×4 geometry×필요 γ를 포함한다.
- 반복 token은 G025/G0125/G050에 `[1,2,1]`, P075에 `[1,6,1]`, P025에 `[3,2,3]`, LOW/HIGH에 `[1,1]`이다. 짝수 전체 수의 median은 가운데 두 값의 평균까지 원 median 규약과 같아야 한다.
- 메모리 절약을 위해 exact weighted order-statistics를 써도 되나 explicit token expansion과 작은 fixture/실제 subset에서 수치 일치를 검증한다. median을 mean이나 근사 percentile로 바꾸지 않는다.

**bank 재사용:** s4는 γ union `[.5,.75,.875,1,1.125,1.25,1.5]`의 최대7개 slice만 필요하다. P025/P075는 G025와 q값을 공유하고 scale만 다시 계산한다. s5의 LOW/HIGH도 기존3γ bank를 재사용할 수 있다. s3는 처음3γ, MSTAR가 추가γ를 요구할 때 해당 slice만 검증 후 추가한다.

같은 원 데이터와 augmentation 알고리즘의 PAN/LP 캐시는 서버 간 checksum 검증 후 복사할 수 있다. **Teacher-dependent q·τ·qref는 다른 Teacher로 복사하지 않는다.** 증강 bank는 N=19,809일 때 `[N,4,G]`; native slice 재사용도 Teacher/runtime/parity가 확인돼야 한다. Cache 축소·nearest γ 보간·일부 patch q를 전체처럼 사용하지 않는다.

기존 bank와 manifest는 덮지 않고 `teacher_sha + data_sha + gamma_bank_sha + view_version + family_tokens`로 새 식별자를 발행한다. 학습 중 scale을 다시 갱신하지 않는다.

## 5. FT와 fresh 학습의 공통 규칙

### 5.1 Weights-initialized FT

FT는 부모 Student U와 A를 모두 가져오고 **U/A AdamW moment·counter·scheduler를 모두 reset**한다. 원 Teacher/reference는 유지하되 CTRL/MIX에서는 지정한 family scale을 사용한다. A를 Teacher clone으로 다시 덮어쓰지 않는다.

- FT20K: U peak1e−5 / A peak3e−7, warmup100, cosine20K→0.
- FT60K: 같은 peak LR, warmup100, cosine60K→0.
- 모든 case α1/β.1/λE.002, batch48.
- 정상20K/60K 종료는 고정 update 수다. 중간에 HQNR가 높아졌다고 학습을 끝내지 않는다.

완료 update t에서 다음 update의 multiplier는 기존 인덱싱으로 다음과 같이 둔다.

\[
f_N(t)=\begin{cases}t/100,&0\le t<100,\\
\frac12\left[1+\cos\left(\pi\frac{t-100}{N-100}\right)\right],&100\le t\le N.\end{cases}
\]

f(0)=0,f(100)=1,f(N)=0, 마지막 scheduler counter=N을 검증한다. 같은 FT run이 중단된 경우에만 자기 full-state로 exact-resume한다.

**20K 대60K의 endpoint 비교는 학습 길이+cosine 경로의 효과**다. 60K run 내20K/40K/60K는 같은 경로의 시간 변화로 분석할 수 있지만, 그20K를 별도 cosine20K endpoint와 동일 조건이라고 하지 않는다.

### 5.2 Fresh100K와 120K 계통

Fresh는 W104D122 U를 사전 seed로 새로 초기화하고 로컬 Teacher A를 독립 clone한다. U peak1e−4 / A peak3e−6, warmup100/cosine100K, 원 loss·batch를 유지한다. Fresh100K 후의20K tail은 Student A/U를 보존하면서 optimizer를 reset한다.

**100K+20K=120K는 처음부터 cosine120K와 같지 않다.** 이 계획의 schedule 비교는 모두 reset 경계를100K로 통일한다.

### 5.3 Seed·stream 대응

FT seed는 모델 재초기화 seed가 아니라 data-stream seed다. 같은 부모의 CTRL/MIX는 t=0 U/A hash, source ID 순서, geometry view, batch size, LR, optimizer 초기화를 맞춘다.

- source order RNG: stream seed+300000.
- geometry RNG: +400000.
- worker RNG: +500000.
- γ draw RNG: +600000; 별도 generator/counter.

Native arm도 같은 uniform γ draw를 소비하지만 effective γ는1이다. Family별 확률이 다르면 같은 uniform number를 각 CDF에 매핑한다. **서로 다른 family의 effective γ hash가 같아야 한다고 요구하지 않는다.** `(source,geometry,uniform_draw)`의 대응과 `(drawn_gamma,effective_gamma)`를 구분해 기록한다. Diagnostic·evaluation이 training RNG를 소비하지 않도록 상태를 복원한다.

## 6. s3 — 다른 Teacher 전이와 duration

### 6.1 기본 전이

P3HI와 P3LO에서 각각 NATIVE0/CTRL(G025)/MIX(G025)를20K 수행한다. 이는 B20의 s5 성능을 s3 점수에 단순히 더하는 실험이 아니라, **R3에 대한 독립적인 전이 검정**이다.

### 6.2 FT60K

같은 두 **원 native100K 부모**에서 CTRL(G025)/MIX(G025)를60K 수행한다. 이미20K 끝난 가중치에서 시작하지 않는다. 같은 stream을 사용해 원 source/geometry의 prefix를 대응시키되 LR horizon 차이는 명시한다. 20K에서 plateau인지, 더 긴 augmented optimization에서 E 비용이 회복되는지 본다.

60K가 좋더라도 확인 단계의20K tail을 몰래60K로 바꾸지 않는다. 이번 global lock은 **입력 분포 한 family**에만 적용한다. Duration 변경의 최종 재현은 별도의 다음 revision이다.

### 6.3 새 seed 확인

모델 seed98611/98612의 native100K trunk를 만들고 각 trunk에서 CTRL(MSTAR)/MIX(MSTAR)20K를 수행한다. 이 두 비교는 기존 좋은 부모를 선택한 효과와 분리한다. 시간이 허용되면98613도 양쪽 arm을 함께 수행한다.

| Case | 단계 | 초기화/부모 | 입력 arm | scale family | stream seed | 추가 updates | 우선순위 |
|---|---|---|---|---|---|---|---|
| A01 | TRANSFER | P3HI | NATIVE0 | NATIVE | 98101 | 20000 | PRIMARY |
| A02 | TRANSFER | P3HI | CTRL | G025 | 98101 | 20000 | PRIMARY |
| A03 | TRANSFER | P3HI | MIX | G025 | 98101 | 20000 | PRIMARY |
| A04 | TRANSFER | P3LO | MIX | G025 | 98102 | 20000 | PRIMARY |
| A05 | TRANSFER | P3LO | CTRL | G025 | 98102 | 20000 | PRIMARY |
| A06 | TRANSFER | P3LO | NATIVE0 | NATIVE | 98102 | 20000 | PRIMARY |
| A07 | LONG60 | P3HI | CTRL | G025 | 98101 | 60000 | PRIMARY |
| A08 | LONG60 | P3HI | MIX | G025 | 98101 | 60000 | PRIMARY |
| A09 | LONG60 | P3LO | MIX | G025 | 98102 | 60000 | PRIMARY |
| A10 | LONG60 | P3LO | CTRL | G025 | 98102 | 60000 | PRIMARY |
| A11 | LOCKED_CONFIRM | FRESH | NATIVE0 | NATIVE | 98611 | 100000 | PRIMARY |
| A12 | LOCKED_CONFIRM | A11 | CTRL | MSTAR | 98711 | 20000 | PRIMARY |
| A13 | LOCKED_CONFIRM | A11 | MIX | MSTAR | 98711 | 20000 | PRIMARY |
| A14 | LOCKED_CONFIRM | FRESH | NATIVE0 | NATIVE | 98612 | 100000 | PRIMARY |
| A15 | LOCKED_CONFIRM | A14 | MIX | MSTAR | 98712 | 20000 | PRIMARY |
| A16 | LOCKED_CONFIRM | A14 | CTRL | MSTAR | 98712 | 20000 | PRIMARY |
| A17 | LOCKED_CONFIRM | FRESH | NATIVE0 | NATIVE | 98613 | 100000 | OPTIONAL |
| A18 | LOCKED_CONFIRM | A17 | CTRL | MSTAR | 98713 | 20000 | OPTIONAL |
| A19 | LOCKED_CONFIRM | A17 | MIX | MSTAR | 98713 | 20000 | OPTIONAL |

## 7. s4 — gain 폭과 native 비율

P4A/P4B 각각에서 NATIVE0 한 개와 G025/G050/G0125/P025/P075의 CTRL/MIX를 수행한다. 각 부모당11개, 총22개 FT20K다.

실행 순서는 **native pair → G025 → G050 → G0125 → P025 → P075**다. 각 family는 두 부모×두 arm의4개를 하나의 예약 묶음으로 처리한다. 시간을 맞추기 위해 강한 gain의 나쁜 첫 결과만 보고 두 번째 부모를 제외하지 않는다.

- G025→G0125/G050: native 비율을 고정한 gain 폭 효과.
- G025→P025/P075: gain 폭을 고정한 augmentation 빈도 효과.
- 각 family의 CTRL을 통해 scale 변화가 주효과인지 확인한다.

Screen 뒤 MSTAR를 잠그고98611/98612의 native100K→CTRL/MIX20K를 확인한다. 동일 후보를 s3/s5에서도 확인하되 Teacher-dependent calibration은 각자 계산한다. s4의 절대 점수를 다른 서버의 절대 점수와 빼서 gain 효과라고 하지 않는다.

| Case | 단계 | 초기화/부모 | 입력 arm | scale family | stream seed | 추가 updates | 우선순위 |
|---|---|---|---|---|---|---|---|
| B01 | SCREEN | P4A | NATIVE0 | NATIVE | 98201 | 20000 | PRIMARY |
| B02 | SCREEN | P4B | NATIVE0 | NATIVE | 98202 | 20000 | PRIMARY |
| B03 | SCREEN | P4A | CTRL | G025 | 98201 | 20000 | PRIMARY |
| B04 | SCREEN | P4A | MIX | G025 | 98201 | 20000 | PRIMARY |
| B05 | SCREEN | P4B | MIX | G025 | 98202 | 20000 | PRIMARY |
| B06 | SCREEN | P4B | CTRL | G025 | 98202 | 20000 | PRIMARY |
| B07 | SCREEN | P4A | CTRL | G050 | 98201 | 20000 | PRIMARY |
| B08 | SCREEN | P4A | MIX | G050 | 98201 | 20000 | PRIMARY |
| B09 | SCREEN | P4B | MIX | G050 | 98202 | 20000 | PRIMARY |
| B10 | SCREEN | P4B | CTRL | G050 | 98202 | 20000 | PRIMARY |
| B11 | SCREEN | P4A | CTRL | G0125 | 98201 | 20000 | PRIMARY |
| B12 | SCREEN | P4A | MIX | G0125 | 98201 | 20000 | PRIMARY |
| B13 | SCREEN | P4B | MIX | G0125 | 98202 | 20000 | PRIMARY |
| B14 | SCREEN | P4B | CTRL | G0125 | 98202 | 20000 | PRIMARY |
| B15 | SCREEN | P4A | CTRL | P025 | 98201 | 20000 | PRIMARY |
| B16 | SCREEN | P4A | MIX | P025 | 98201 | 20000 | PRIMARY |
| B17 | SCREEN | P4B | MIX | P025 | 98202 | 20000 | PRIMARY |
| B18 | SCREEN | P4B | CTRL | P025 | 98202 | 20000 | PRIMARY |
| B19 | SCREEN | P4A | CTRL | P075 | 98201 | 20000 | PRIMARY |
| B20 | SCREEN | P4A | MIX | P075 | 98201 | 20000 | PRIMARY |
| B21 | SCREEN | P4B | MIX | P075 | 98202 | 20000 | PRIMARY |
| B22 | SCREEN | P4B | CTRL | P075 | 98202 | 20000 | PRIMARY |
| B23 | LOCKED_CONFIRM | FRESH | NATIVE0 | NATIVE | 98611 | 100000 | PRIMARY |
| B24 | LOCKED_CONFIRM | B23 | CTRL | MSTAR | 98711 | 20000 | PRIMARY |
| B25 | LOCKED_CONFIRM | B23 | MIX | MSTAR | 98711 | 20000 | PRIMARY |
| B26 | LOCKED_CONFIRM | FRESH | NATIVE0 | NATIVE | 98612 | 100000 | PRIMARY |
| B27 | LOCKED_CONFIRM | B26 | MIX | MSTAR | 98712 | 20000 | PRIMARY |
| B28 | LOCKED_CONFIRM | B26 | CTRL | MSTAR | 98712 | 20000 | PRIMARY |
| B29 | LOCKED_CONFIRM | FRESH | NATIVE0 | NATIVE | 98613 | 100000 | OPTIONAL |
| B30 | LOCKED_CONFIRM | B29 | CTRL | MSTAR | 98713 | 20000 | OPTIONAL |
| B31 | LOCKED_CONFIRM | B29 | MIX | MSTAR | 98713 | 20000 | OPTIONAL |

## 8. s5 — 방향성과 학습 시점

### 8.1 G025/LOW/HIGH의 방향 대조

원 P5B/P5A 각각에서 family별 CTRL/MIX20K, 총12구간이다. 새 stream98301/98302를 써서 기존 B20과 구분한다. 기존 cache를 재사용할 수 있지만 과거 completed run을 새 run처럼 다시 집계하지 않는다.

LOW가 좋고 HIGH가 나쁘면 gain 변화의 방향/평균 이동이 중요할 가능성이 있다. G025가 양쪽보다 안정적이면 양방향 변동의 가치가 있다. 그러나 LOW/HIGH의 평균γ가 다르므로 결과를 순수 randomness의 인과효과로 단정하지 않는다. **이 두 family는 자동 MSTAR 후보에서 제외**하고 다음 연구 질문으로 보존한다.

### 8.2 100K+20K의 2×2 schedule

모델 seed98411/98412 각각에서 다음을 구성한다. 이 실험 전체는 **G025의 같은 mixed scale**을 처음부터 끝까지 사용한다. Native 입력 trunk도 원 native scale 대신 CTRL(G025)다. 따라서 초기 두 trunk의 차이를 calibration 변경이 아니라 입력 view의 차이로 읽을 수 있다.

| 경로 | 첫100K | 후기20K | 질문 |
|---|---|---|---|
| **NN** | native, G025 scale | native, G025 scale | 같은 총 update·reset 경계의 대조 |
| **NM** | native, G025 scale | PANMIX G025 | 후기 도입 |
| **MN** | PANMIX G025 | native, G025 scale | native 복귀로 이득이 사라지는가? |
| **MM** | PANMIX G025 | PANMIX G025 | 초기부터 계속 증강 |

각 seed에서 fresh trunk는2개, tail은4개이므로 **6구간**이다. 두 seed는12구간이다. 네 endpoint 모두120K, 같은100K reset 경계, 동일 tail LR 및 tail stream을 갖는다. Trunk를 공유하는 네 경로를 독립4seed로 부르지 않는다.

이 2×2는 R5에서의 학습 시점/전환 기전 연구이며 global MSTAR confirmation과 구분한다. MSTAR 확인의 native trunk는 **원 native scale**이다. 두 trunk 설정이 다르므로 수치가 비슷하다는 이유로 서로 대신 쓰거나 합치지 않는다.

### 8.3 MSTAR 새 seed 확인

다른 서버와 같은98611/98612 seed로 별도 native100K→CTRL/MIX(MSTAR)20K를 수행한다. Schedule study의98411/98412를 후보 선택 후의 확인 seed로 바꾸어 재분류하지 않는다. s5는 시간이 가장 빠듯하므로3번째 seed는 실측상 여유가 있을 때만 가능하다.

| Case | 단계 | 초기화/부모 | 입력 arm | scale family | stream seed | 추가 updates | 우선순위 |
|---|---|---|---|---|---|---|---|
| C01 | DIRECTION | P5B | CTRL | G025 | 98301 | 20000 | PRIMARY |
| C02 | DIRECTION | P5B | MIX | G025 | 98301 | 20000 | PRIMARY |
| C03 | DIRECTION | P5A | MIX | G025 | 98302 | 20000 | PRIMARY |
| C04 | DIRECTION | P5A | CTRL | G025 | 98302 | 20000 | PRIMARY |
| C05 | DIRECTION | P5B | CTRL | LOW | 98301 | 20000 | PRIMARY |
| C06 | DIRECTION | P5B | MIX | LOW | 98301 | 20000 | PRIMARY |
| C07 | DIRECTION | P5A | MIX | LOW | 98302 | 20000 | PRIMARY |
| C08 | DIRECTION | P5A | CTRL | LOW | 98302 | 20000 | PRIMARY |
| C09 | DIRECTION | P5B | CTRL | HIGH | 98301 | 20000 | PRIMARY |
| C10 | DIRECTION | P5B | MIX | HIGH | 98301 | 20000 | PRIMARY |
| C11 | DIRECTION | P5A | MIX | HIGH | 98302 | 20000 | PRIMARY |
| C12 | DIRECTION | P5A | CTRL | HIGH | 98302 | 20000 | PRIMARY |
| C13 | SCHEDULE_2x2 | FRESH | CTRL | G025 | 98411 | 100000 | PRIMARY |
| C14 | SCHEDULE_2x2 | FRESH | MIX | G025 | 98411 | 100000 | PRIMARY |
| C15 | SCHEDULE_2x2 | C13 | CTRL | G025 | 98511 | 20000 | PRIMARY |
| C16 | SCHEDULE_2x2 | C13 | MIX | G025 | 98511 | 20000 | PRIMARY |
| C17 | SCHEDULE_2x2 | C14 | CTRL | G025 | 98511 | 20000 | PRIMARY |
| C18 | SCHEDULE_2x2 | C14 | MIX | G025 | 98511 | 20000 | PRIMARY |
| C19 | SCHEDULE_2x2 | FRESH | MIX | G025 | 98412 | 100000 | PRIMARY |
| C20 | SCHEDULE_2x2 | FRESH | CTRL | G025 | 98412 | 100000 | PRIMARY |
| C21 | SCHEDULE_2x2 | C20 | MIX | G025 | 98512 | 20000 | PRIMARY |
| C22 | SCHEDULE_2x2 | C20 | CTRL | G025 | 98512 | 20000 | PRIMARY |
| C23 | SCHEDULE_2x2 | C19 | MIX | G025 | 98512 | 20000 | PRIMARY |
| C24 | SCHEDULE_2x2 | C19 | CTRL | G025 | 98512 | 20000 | PRIMARY |
| C25 | LOCKED_CONFIRM | FRESH | NATIVE0 | NATIVE | 98611 | 100000 | PRIMARY |
| C26 | LOCKED_CONFIRM | C25 | CTRL | MSTAR | 98711 | 20000 | PRIMARY |
| C27 | LOCKED_CONFIRM | C25 | MIX | MSTAR | 98711 | 20000 | PRIMARY |
| C28 | LOCKED_CONFIRM | FRESH | NATIVE0 | NATIVE | 98612 | 100000 | PRIMARY |
| C29 | LOCKED_CONFIRM | C28 | MIX | MSTAR | 98712 | 20000 | PRIMARY |
| C30 | LOCKED_CONFIRM | C28 | CTRL | MSTAR | 98712 | 20000 | PRIMARY |
| C31 | LOCKED_CONFIRM | FRESH | NATIVE0 | NATIVE | 98613 | 100000 | OPTIONAL |
| C32 | LOCKED_CONFIRM | C31 | CTRL | MSTAR | 98713 | 20000 | OPTIONAL |
| C33 | LOCKED_CONFIRM | C31 | MIX | MSTAR | 98713 | 20000 | OPTIONAL |

## 9. MSTAR: 한 번만 선택하고 잠근다

MSTAR는 실행 가능한 profile 문자열이 아니라 **아직 바인딩하지 않은 설계 변수**다. 구체 family, tokens, source/result SHA, 결정시각, 선택 이유가 적힌 immutable `selection_lock.json`이 생겨야 CTRL/MIX(MSTAR)를 실행한다. Native confirmation trunk는 후보에 의존하지 않아 먼저 시작할 수 있다.

### 9.1 결정 시점과 후보 집합

- 가능한 한 s4의22개 screen 결과가 모이면 결정하고, **공통 t0+22h에서 반드시 잠근다.**
- 후보는 `G025, G0125, G050, P075, P025`뿐이다.
- s4의 두 부모×CTRL/MIX가 모두 완료되고 integrity가 확인된 family만 신규 승격 자격을 얻는다.
- LOW/HIGH,60K,강한 KD/edge,새 loss 조합은 이번 잠금에 들어가지 않는다.
- 결과·자료가 부족하거나 개선 후보가 없으면 **G025**를 선택하고 `NO_VALIDATED_UPGRADE` 또는 `PARTIAL_SCREEN_FALLBACK`을 남긴다. 이는 G025가 모든 reference에서 성공했다는 의미가 아니다.
- 같은 lock SHA를 세 서버가 수신·검증한다. 다른 Teacher를 공유하는 것이 아니라 분포 정의만 공유한다. 통신 실패로 아직 수신하지 못한 서버는 core/diagnostics/native trunk를 계속할 수 있으나 MSTAR tail은 보류한다.

### 9.2 신규 family의 승격 기준

아래는 운영 기준이며 통계적 유의성 보장이 아니다. 지표는 항상 동일 선택점의 같은 checkpoint에서 가져온다.

**입력 이득:** 두 부모 모두 MIX(f)−CTRL(f)의 ΔH>0, ΔDs<0, 평균ΔH≥.002. 평균 상대E 비용≤+.5%, 각 부모≤+1%, 각ΔDλ≤+.001. VAL-selected에서 한 부모라도 ΔH<−.001이면 자동 승격하지 않고 `CHECKPOINT_SENSITIVE`로 보존한다.

**Incumbent보다 나은가:** 두 부모의 MIX(f)−MIX(G025)에 대해 평균ΔH≥.001, 각ΔH≥−.0005, 각ΔDs≤+.0005, 평균 상대E 비용≤+.5%, 각≤+1%, 각ΔDλ≤+.001. 한 run 최고치만으로 고르지 않는다. 이 비교에는 각 family의 scale 차이가 포함된다.

여러 후보가 통과하면 **두 부모 중 작은 ΔH가 더 큰 후보**를 선택한다. 동률1e−8 이내이면 평균 상대E가 낮은 후보, 다시 동률이면 사전 순서 `G0125 → G050 → P075 → P025`다. 아무 신규 후보도 통과하지 않으면 G025를 유지한다.

s3/s5 결과는 전이/기전 해석에 사용하지만 **s4 후보의 나쁜 부모를 대체하거나 새 후보를 끼워 넣는 용도로 사용하지 않는다.** 이미 시작된 확인 seed의 점수를 보고 lock을 다시 열지 않는다. 사용자 변경 지시는 새 revision과 시각을 남긴 뒤 아직 시작하지 않은 완결 block에만 적용한다.

### 9.3 새 seed 확인의 해석

주 확인은 3 Teacher block×2 Student seed=6개 paired endpoint다. 같은 seed를 쓰더라도 서로 다른 Teacher·서버이므로 먼저 **로컬 ALT−CTRL**을 계산하고 Teacher별 두 seed를 보고한다. 전체6개를 완전히 독립·동일분포 표본처럼 단순 검정하지 않는다.

확인 목표: 6 pair 중≥5에서 ΔH>0와 ΔDs<0, 어느 Teacher block도 평균ΔH<−.001이 아님, 전체 평균ΔH≥.002, RR 비용 guard 유지. 이는 40시간 campaign의 운영적 재현 기준이며 p-value나 보편적 안정성의 증명은 아니다. 4 pair 이하만 완료되면 각 pair를 보고하되6-pair 성공으로 표시하지 않는다.

별도로 **같은 EXACT_FINAL checkpoint에서 HQNR>.964와 ERGAS<.552**를 체크한다. PAN-Crafter 보충 Table7/12의.552를 운영 기준으로 사용하고 본문 Table2의.522는 별도 strong goal로 보존한다 [S4]. 둘을 임의로 통일하지 않는다. 한 seed가 통과했다고 전체 recipe의 재현 성공으로 표시하지 않는다.

## 10. 기전 진단 — 학습 case와 분리된 read-only replay

### 10.1 A와 U의 변화 분리

각 서버의 G025 CTRL/MIX endpoint에서 선택한 **같은 부모 pair**에 대해 아래 네 조합을 고정 입력으로 평가한다.

| 조합 | A | U |
|---|---|---|
| CC | CTRL | CTRL |
| CM | CTRL | MIX |
| MC | MIX | CTRL |
| MM | MIX | MIX |

동일 parent에서 출발한 A/U만 교환한다. PAN·LP는 선택한 A의 correction으로 함께 warp하고 MS frame은 유지한다. CM/MC는 **기전 진단 전용**이며 공식 main-result 후보, 추론 방식, 새 학습 부모로 채택하지 않는다.

\[
\text{U effect at }A_C=F(C,M)-F(C,C),\quad
\text{A effect at }U_C=F(M,C)-F(C,C),
\]
\[
\text{interaction}=F(M,M)-F(C,M)-F(M,C)+F(C,C).
\]

F는 H,E,Ds,Dλ 등 지표 각각이며 좋은 방향의 부호는 지표에 따라 다르다. 이 factorial replay는 두 학습된 모듈의 개입 반응을 보여주는 것이지 훈련 전체의 유일한 원인을 증명하는 것은 아니다.

### 10.2 native 품질과 perturbation 민감도

고정 train128(gradient24), native validation, RR20에서 parent/CTRL/MIX의 다음을 기록한다.

- c의 mean/median/p95와 parent 대비 drift; c는 예측 보정이고 displacement GT가 아니다.
- native GT pixel/edge error, 가장자리1px와 내부 error, band별 error, 출력값 overshoot/잔차 에너지.
- 같은 고정γ sweep에서 U 출력 변화와 A 변화. 가능하면 A를 native c로 고정한 forward replay도 수행해 입력주파수의 직접 반응과 A 반응을 분리한다. 이때 freeze는 진단 forward에만 적용한다.
- hard/soft/edge weighted gradient와 active soft mass. 값이 작다는 이유로 기본 loss routing을 바꾸지 않는다.
- FR20의 band별 signed UQI operand 차이와 Ds reconstruction check, scene별H/Ds/Dλ. signed UQI 차이는 순수 Pearson 상관이나 실제 과선명화 GT가 아니다.

독립 정합 추정기는 기존 검증된 구현이 있을 때만 붙인다. 없으면 `ESTIMATOR_NOT_AVAILABLE`로 남기고 숫자를 만들어 채우지 않는다. 신뢰할 수 있는 새 추정기가 없어도 feature·loss·native metric 진단과 기본 학습은 가능하다.

### 10.3 가시적 품질 점검

20개 장면 전체 지표, 사전에 고정한 scene/crop index의 RGB·NIR·edge·RR error map을 저장한다. 좋아 보이는 건물 한 장만 고르지 않는다. FR에는 GT가 없으므로 RR 정확도, native reference 지표, 시각 결과를 나눠 기술한다. 출력 blur/warp로 좋은 점수를 만드는 사후 조정은 하지 않는다.

각 scene을 지우는 leave-one-scene-out 통계나 scene별 분포는 **해석용**이다. 이를 보고 공식20장 평균에서 장면을 제외하거나 좋은 subset을 최종 결과로 바꾸지 않는다.

## 11. 평가·보존·Sheet 규약

- Primary: `EXACT_FINAL`의 전체 정상 native RR20/FR20.
- Secondary: 고정 후보 집합에서 native validation ERGAS 최소, 동률이면 이른 step의 `RR_VAL_SELECTED`.
- Auxiliary: 같은 집합의 `RAW_AUX`, **test-aware development**로 명시.
- FT20K 공식 후보:1000,2000,…,20000(20개).
- FT60K 공식 후보:3000,6000,…,60000(20개). Curve용20K/40K는 추가 보존하되 RAW/VAL 선택 후보에 끼우지 않는다.
- Fresh100K 공식 후보: 기존 B20 fresh100K의 검증된50개 grid를 그대로 재사용하고 manifest에 정수 목록·SHA를 기록한다. 이 문서가 임의로 새로운 grid 숫자를 복원했다고 주장하지 않는다. 가져오지 못하면 구현 전에 해당 규약을 고정하고 이후 모든 fresh에 같게 적용한다.
- 각 신규 FT의 t0 parent endpoint는 별도 anchor다. 0-update를 새 학습 성공 후보로 추가하지 않는다.

`parent_updates`, `added_updates`, `lifetime_student_updates`, `Teacher train(h)`, `parent Student train(h)`, `new train(h)`, calibration/eval/I/O/wall, shared-trunk 비용을 분리한다. 20K FT를 fresh120K라 부르지 않는다. 공통 trunk는 campaign 비용에 한 번만 합산하고, 각 추론 모델의 총 학습 lineage 비용도 별도로 제시한다.

전용 Sheet 탭은 `GF2-P40-s3/s4/s5`를 사용하고, 별도 summary는 pair/case ID와 동일 checkpoint 지표로 구성한다. 기존 B20/L100 결과나 열 제목을 덮지 않는다. Machine-readable metadata 최소 항목:

`campaign,revision,server,case,block,phase,parent_id,parent_run,parent_model_sha,Teacher_run/SHA,source_archive_SHA,data/LP_SHA,distribution_id/tokens/G_SHA,calibration_id/τ/qref/qbank_SHA,model_seed,stream_seed,draw/effective_gamma,init_U/A_SHA,actual_updates,scheduler_end,selection,checkpoint_SHA,RR/FR metrics,signedDs,eval_complete,readback_status,cost,lineage`.

중간 높은H와 다른 checkpoint의 낮은E를 섞지 않는다. Import된 과거 run은 `REUSED_EXTERNAL`과 원 run ID를 유지하고 신규 학습 수에 포함하지 않는다.

## 12. 40시간 운영과 비용

### 12.1 기존 B20과의 인계

새40h는 승인된 **별도 P40 campaign**이다. B20의 t0/완료기록/시간 한도를 바꾸지 않는다. 이미 진행 중인 case 또는 예약된 pair는 기본적으로 안전 완료·보존 후 인계하며 중간 Teacher나 source를 바꾸지 않는다. 새 작업을 위해 GPU worker를 강제 종료하거나 동시에 경쟁 실행하지 않는다.

실제 전환/준비를 시작할 때 t0를 한 번 발행한다. 아직 준비·학습을 시작하지 않은 시간을 임의로 소급하거나 미래t0 이전 비용을 숨기지 않는다. B20 대기가 새t0 이후 발생하면 P40의 wall-clock 예산도 소모한다. 사전 cached 자산의 역사적 비용은 별도 기록하고 새 cache 시간처럼 다시 청구하지 않는다.

이번 문서의 live read에서 C08은 미등록이었다. 담당자는 B20 C07/C08·C09/C10의 현재 full-state와 local 완료 여부를 먼저 확인해 **이미 돌고 있는 fresh pair를 중복 기동하지 않는다.** P40은 새로운 seed98411/98412,98611/98612를 사용한다. 기존 pair는 그대로 기존 campaign 결과이며 이번 후보나 확인으로 사후 재분류하지 않는다.

### 12.2 Clock와 admission

| 시각 | 규칙 |
|---|---|
| t0 | 공통40h clock 생성·source freeze·자산 audit 시작 |
| t0+22h까지 | s4 screen에서 MSTAR 한 번 잠금. 미완이면 사전 G025 fallback |
| **t0+34h** | 새로운 training block 입장 금지 |
| **t0+37h** | 모든 optimizer update 종료; 미완이면 full-state safe pause |
| **37–40h** | 평가부채·보존·집계·Sheet readback 우선 |
| t0+40h | 종료. 미완·미입장 수와 비용을 그대로 보고 |

한 block은 모든 대조군·변경안·필요 cache·평가·보존 시간을 함께 예약한다. s3 TRANSFER는 부모별3arm, LONG60은 부모별2arm, s4 SCREEN과 s5 DIRECTION은 family별두부모4arm, s5 SCHEDULE은 seed별2trunk+4tail, CONFIRM은 seed별native100K+2tail이다.

MSTAR가 잠기기 전에도 CONFIRM trunk를 시작할 수 있다. 다만 해당 block의 tail과 아직 필요한 cache 비용까지 예약한다. 같은 서버에서 다음 native trunk를 선행시켜22h 대기를 줄일 수 있지만, 미확정 family의 tail을 임의 실행하지 않는다. 각 stage 의존성을 DAG로 관리하고 parent가 공식 완료·보존되기 전에 소비하지 않는다.

예약식:

`현재경과 + 남은예약작업 + 새block(학습+평가+I/O+cache) + 필수진단 <= 37h`.

그와 별도로 closeout3h를37–40h에 확보한다. 현재 실행시간·처리량이 초기값보다 커지면 최근 같은 서버·길이·view의 실측에 1.15 안전계수를 적용한 값과 초기 추정 중 큰 값을 쓴다. Optimizer-only 시간을 전체비용으로 오인해 예약을 낮추지 않는다.

### 12.3 초기 시간 추정 — 실측 보장 아님

B20의20K 순수학습 약.39–.41h,100K 약1.9–2.1h 관측을 출발점으로, 평가·I/O를 포함한 입장 추정은 **FT20K .65h, FT60K 1.65h, fresh100K 2.60h**로 둔다. 새 광범위 cache 구축과 광학 view에 대한 실측은 아직 없다 [S1].

| 항목 | s3 | s4 | s5 |
|---|---:|---:|---:|
| 준비·cache·추가MSTAR slice 초기 허용 | 5.0h | 6.0h | 2.0h |
| 우선74구간 중 해당 서버 학습+평가+I/O | 18.3h | 22.1h | 31.2h |
| 결정 통신·대기 여유 | 2.0h | 0h | 0h |
| 기전 진단 | 1.5h | 1.5h | 1.5h |
| closeout | 3.0h | 3.0h | 3.0h |
| **우선 계획 총추정** | **29.8h** | **32.6h** | **37.7h** |
| 선택적3번째 확인 seed 추가 | 3.9h | 3.9h | 3.9h |
| 모두 포함 시 | 33.7h | 36.5h | **41.6h → 초기값에서는 불허** |

s5는 기존 검증된3γ PAN/LP/q bank 재사용을 전제로2h를 배정했다. MSTAR가 새γ를 요구하거나 기존 source/parity를 충족하지 못하면 실제 추가시간을 먼저 예약한다. 여유시간은 현장검증·source 복구·평가부채를 위한 것이며, 빈 시간을 채우려고 새로운 계수·seed를 무제한 추가하지 않는다.

### 12.4 시간 부족 시 축소 순서

**시작하지 않은 block에만 적용한다.** 이미 시작한 pair를 첫 결과가 나쁘다는 이유로 중단하지 않는다.

1. 선택적 98613 확인 3개씩은 먼저 제외한다.
2. s3의 두 번째 부모 LONG60을 제외할 수 있다. 기존20K 전이 pair와 새 seed 확인을 우선한다.
3. s4의 아직 시작하지 않은 P075, 그다음 P025, 그다음 G0125 family를 뒤에서 제외한다. G025 기준 전이와 G050의 큰 대조를 우선한다. 미완 family에는 선택 자격이 없다.
4. s5의 HIGH direction block 또는 두 번째 schedule98412 전체6구간을 확인 예산보다 먼저 축소한다. 최소1개 schedule2×2와 기본 G025 재검정은 남긴다.
5. 그래도 불가능하면 두 번째 fresh confirmation98612의 3구간 전체를 미입장으로 남긴다. 대조군 하나만 끝내고 paired 확인 완료라고 하지 않는다.

기존 필수 cache 범위를 줄이거나 학습 batch를 조용히 바꾸는 방식으로 시간을 맞추지 않는다.

## 13. 안전·실패·재개

- NaN/Inf, source/parent/data/reference SHA mismatch, 누락 q slice, Teacher mutation, gradient routing 오류는 `BLOCKED_INTEGRITY`다. 성능이 좋은 다른 cache로 바꿔 진행하지 않는다.
- OOM은 batch size나 crop을 자동 변경하지 말고 state를 보존한다. 동일 조건의 I/O/loader 수정은 새 source 기록과 필요한 parity 검사 후 재개한다.
- FT의 native validation E가 parent보다 10% 이상 나쁜 상태가 연속 두 번 발생하면 안전 pause 후 검토한다. 중간 H 하락이나 목표 미달은 중단 사유가 아니다. Fresh 초기의 random 모델에는 trained-parent E 기준을 적용하지 않는다.
- 물리적 원인 진단이 부재해도 기본 학습은 가능하지만, 이를 확인했다고 보고하지 않는다.
- 성공 seed가 나왔다고 다음 사전 예약 seed를 취소하지 않는다. 실패를 숨기는 단독 ALT 재추첨은 없다.
- 새 FT/fresh run의 중단은 그 run의 U/A+AdamW+scheduler+RNG+sampler+counts+cache identity로 재개한다. 부모 단계의 optimizer를 임의로 혼합하지 않는다.

## 14. Controller 설계 의사코드

```python
# 설계용. 원격 실행 코드 또는 기존 runner의 실제 CLI가 아니다.
window = bind_new_immutable_40h_clock(actual_t0)
release = verify_source_and_local_parent_bundles()
protect_existing_B20_workers_and_s1_s2()

while elapsed(window) < 37 * HOUR:
    refresh_local_state_and_evaluation_debt()
    if not mstar_locked() and (s4_screen_complete() or elapsed(window) >= 22 * HOUR):
        lock_once(select_by_registered_rule_or_G025(), evidence_hashes())
    block = next_ready_registered_block(prioritize_complete_pairs=True)
    if block is None:
        process_diagnostics_or_evaluation_debt()
        continue
    if elapsed(window) >= 34 * HOUR or not fits_whole_block_before_37h(block):
        record_not_admitted(block)
        continue
    verify_required_family_bank_and_pair_initialization(block)
    run_or_exact_resume_registered_segments(block)
    save_fullstate_and_native_metrics_before_optional_heavy_diagnostics()
    update_immutable_pair_results()

safe_pause_training_and_closeout_until_40h()
report_all_completed_failed_partial_not_admitted_cases()
```

HQNR 목표 통과만으로 전체 예약 실험을 종료하지 않는다. 실제 구현에서는 기존 controller의 signal/deadline/lock/readback 동작을 재사용하면서 **40h와 새 registry를 별도 namespace로 구현**한다. `MSTAR` 미바인딩 template, 사전에 없는 case, 원 Teacher가 다른 parent는 실행할 수 없다.

## 15. 최종 산출물과 논문 해석 범위

40h가 끝나면 단일 최고행만 보고하지 않고 다음을 남긴다.

1. **Reference 전이표:** R3/R4/R5×두 native 부모의 G025 입력 효과와 scale 효과.
2. **강도 곡선:** gamma 폭/native 확률별 두 부모의 H/E/Ds/Dλ, matched CTRL 및 G025 대비 차이.
3. **경로표:** NN/NM/MN/MM, 같은120K·같은 reset 경계·두 seed.
4. **Locked 확인표:** MSTAR, 3개 Teacher×사전에 지정한 새 seed의 모든 CTRL/MIX 결과. 미완은 미완으로 표시.
5. **기전표·영상:** A/U replay와 native RR error·FR signed Ds·고정 장면의 시각 자료.
6. **비용·무결성:** 새 학습/부모/Teacher/cache/공유 trunk 비용, source·데이터·reference·checkpoint SHA, actual update, Sheet readback.

최종 recipe는 G025일 수도 있고 개선 후보일 수도 있다. **모든 지표와 seed가 계층적으로 좋아야 한다는 전제는 없다.** R5에서만 효과가 있으면 reference-specific 결과로, 후기에만 효과가 있으면 late-augmentation schedule로 기술한다.

이 campaign은 이미 반복 조회한 FR test 지표로 탐색·선택하므로 **test-aware development**다. 새 seed 확인은 훈련 재현성 검정이며, test가 untouched로 바뀌는 것은 아니다. 실제 held-out 세트가 없는데 있다고 쓰지 않는다. 논문 최종 수치에는 선택 과정·고정 seed 전체·현재 training-view 확장을 명시한다.

## 16. 출처

- **[S1]** `sources/GF2_B20_Current_Results_Analysis_20260922_0836KST.md`, 특히 §3(PANMIX 두 부모), §5–6(다른 축의 음성 결과), §8(후속 우선순위). 동봉 paired/all-selection CSV는 해당 시점의 원값이다.
- **[S2]** `sources/PANDA_GF2_B20_S345_20H_ExperimentPlan_2026-09-22.md`, §2–4(method/parent/FT), §8(PAN gain/calibration), §9(승격·test 관측 범위).
- **[S3]** `sources/PANDA_Method_v4_Architecture_Conformance_Audit_2026-09-21.md`, §3–6(동일 구조·gradient·calibration/Scharr/stop-gradient).
- **[S4]** 사용자 첨부 `pancrafter.pdf`, Table7 및 Table12: GF2 H.964/E.552. 본문 Table2의.522와 구분한다. 이 논문의 MARs/CM3A를 PANDA에 추가하지 않는다.
- **[S5]** 사용자 최종 방법 문서 `PAN_Final_Method_qe_AlignmentAware_Fitting_2026-09-20_KR_v4_notation.md`, §1.1, 4.1, 5.3–5.4, 7. 구현의 stop-gradient·median 범위 등은 [S3]와 함께 읽는다.
- **[S6]** 작성 중 native Sheets readback `GF2-B20-s5!A1:F15`, `BB1:BL15`: C01–C07, C08 미등록. 이를 전체 워크북이나 실행 상태의 재감사로 확대해 표현하지 않는다.

**이 자료는 실행 인계용 설계다.** Registry/CSV/queue JSON의 참조와 case 개수는 로컬 정적 검증했으나, GPU 회귀·cache 구축·원격 기동을 수행한 것은 아니다.
