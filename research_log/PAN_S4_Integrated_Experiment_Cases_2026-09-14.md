# PAKD50 — s4 통합 실험 배정 및 실행 기준

**작성일:** 2026-09-14  
**문서 상태:** s4 배정안 및 s1 구현 인계 명세. 이 문서의 신규 case는 아직 실행하지 않았다.  
**상위 캠페인:** `PAKD50_W112D123_WV3_20260914_v1`  
**s4 작업 묶음:** `PAKD50_S4_EXTENSION_v1` — 상위 캠페인과 별도 방법으로 집계하지 않는다.  
**목표:** WV3 FR20 원 PAN 기준 **raw-original HQNR ≥0.959**, 확장 목표 **≥0.960**. V64가 아니다.  
**저장소 검토 snapshot:** `hojunking/PAN-Crafter-repro`, `3d068859e0d4b8d3cbd74242419fbf9fc27b0c4f`. 실행 시에는 실제 승인된 release와 자산을 다시 확인한다.  
**동반 문서:** [gspread·s4 구현 인계](PAN_S4_Gspread_Implementation_Handoff_2026-09-14.md), [기계 판독용 case registry](PAN_S4_CaseRegistry_2026-09-14.yaml).

---

## 1. 배정 결론

> **s1–s3는 기존 주력 비교를 계속한다. s4는 동일 조건의 서버 교차 대조를 먼저 확보하고, aligner 적응 강도와 Q12 감독 강도의 작은 조정을 탐색한 뒤, 선택한 방법을 seed3407에서 확인한다.**

s4를 처음부터 새로운 Teacher·backbone·대규모 loss 탐색 서버로 만들지 않는다. 추가 서버의 가치는 서로 비교할 수 없는 최고 기록을 하나 더 만드는 데 있지 않다. 기존 세 서버의 중심 비교를 방해하지 않으면서, 목표까지 남은 개선 여지를 찾고 결과의 재현 범위를 넓히는 데 있다.

| 서버 | 유지할 역할 | s4 추가에 따른 변경 |
|---|---|---|
| s1 | 기준 코드 구현·회귀 검사·T0 및 calibration package·Student seed1234 | s4 지원과 업로더 변경은 s1에서 구현한다. 진행 중인 trainer checkout은 바꾸지 않는다. |
| s2 | 동일 release의 Student seed777 대응 비교 | 기본 큐 유지. s4 탐색 승자가 정해지면 필요한 확인 run만 추가한다. |
| s3 | 동일 release의 Student seed2026 대응 비교 | 기본 큐 유지. 서버 표기 `s3(5090)`와 학습 서버 ID `s3`를 혼동하지 않는다. |
| **s4** | **seed1234 서버 교차 → 작은 LR·계수 탐색 → seed3407 확인** | 하드웨어·도입 시각을 실측하고 잔여 예산에 맞춰 큐를 축소한다. |

**중요:** s4의 개발 seed는 1234로 한다. 3407은 기존 계획에서 미리 정한 추가 확인 seed이므로, 여러 hyperparameter를 시험하는 기본 seed로 소비하지 않는다. 같은 seed1234를 s1과 s4에서 실행한 것은 독립 seed가 하나 늘어난 것이 아니다. [P §8.5, §10]

## 2. 기존 계획과 저장소에서 새롭게 확인한 사항

| 항목 | 조회한 구현·기록 | s4에서의 처리 |
|---|---|---|
| 기본 골격 | W112–D123, 9ch, 단일 HRMS, FRESH50 | 유지한다. D122로 바꾸지 않는다. |
| Teacher | T0: L1E4 seed2025의 같은 A/U, 저장소 tag `best_hqnr` | 자기 frozen aligner를 쓰는 Teacher 경로를 유지한다. |
| T0 자산 기록 | selected update 24240, 기록 raw HQNR 0.956976 | 파일·tensor hash와 s4 재현을 먼저 검사한다. s4 실측값으로 선기록하지 않는다. |
| τR | 구현 노트에 통합 T0 calibration 약 0.012464 | 실제 공통 JSON의 원 정밀도 값을 사용한다. 문서의 반올림 값을 config에 재입력하지 않는다. |
| λE | J0 seed1234 exact50K pilot 이후 생성 | s1 package를 받는다. s4의 J0로 다시 선정하지 않는다. |
| 실제 평가 grid | `GRID1010_50K_v1` | 기존 계획의 설명용 GRID1K가 아니라 실제 1010 간격을 계승한다. |
| C0 기본 case | J0/F0/JR/FR, λE 확보 후 JQ/FQ/XJ; AL0/ALQ도 생성기 정의 존재 | s4 첫 탐색은 이미 표현 가능한 LR·scalar 중심으로 한다. |
| 미구현 연구 기능 | P/부분 routing, D/LF freeze schedule, TCOPY/CONT 등은 C1/C2로 보류 | config key만 추가해 실행하지 않는다. trainer와 검사까지 구현된 release가 필요하다. |
| 서버 허용 범위 | generator의 서버 map, prepare shell은 s1/s2/s3 | **현재 snapshot에서 `--server s4` 학습 준비는 바로 실행할 수 없다.** |

근거: [R1–R3]. 위 표는 repository snapshot 읽기다. 현재 서버 프로세스·완료 run·이후 push까지 실시간 확인한 결과는 아니다.

구현 노트의 s1 smoke 기반 약 1.9h/run 예상은 s4의 처리시간이 아니다. s4의 GPU 수·모델·VRAM은 미확인이다. 기본 운영은 **GPU slot 1개, run 1개씩**으로 잡는다.

## 3. 시간과 도입 시점: s4의 시계를 임의로 새로 시작하지 않는다

상위 50시간은 기존 캠페인의 경과시간 기준이다. s4가 늦게 들어왔다고 기존 마감에 다시 50시간을 더하지 않는다.

```text
T0_campaign = 기존 승인된 캠페인 ledger의 시작시각
D_train     = T0_campaign + 46시간
D_final     = T0_campaign + 50시간
T4_ready    = s4가 실제 사용 가능해진 시각
```

시각은 timezone이 포함된 형태로 저장한다. 기존 로그가 timezone 없는 문자열이면 서버 timezone을 확인한 뒤 명시적으로 변환한다. 별도 승인된 s4 연장 예산이 생기면 `extension_budget`으로 분리한다.

현재 prepare는 서버별 로컬 시계와 예산 파일을 만든다. s4에는 공유 `D_train/D_final`을 우선 적용하는 admission 검사를 추가한다. 나머지 서버의 진행 중인 ledger를 초기화하지 않는다. [R3]

새 case 묶음 G의 승인 조건:

\[
 t_{now}+1.1\sum_{r\in G}\widehat t_r+t_{reserved\_confirmation}\le D_{train}.
\]

`widehat t_r`에는 training·정기 평가·checkpoint 저장·필수 후처리를 포함한다. 이미 G에 확인 run을 포함했다면 확인 비용을 또 예약하지 않는다. 안전 배율 1.1은 기존 계획의 운영값이지 성능 보장값이 아니다. 46–50h의 최종 감사 시간은 별도로 보존한다.

**λE 대기시간, 데이터 전송, gate, 장애 시간도 실제 경과시간을 소비한다.** 빈 GPU를 채우려고 확인 예산을 먼저 사용하지 않는다.

## 4. 모든 s4 case가 공유하는 학습 계약

| 항목 | 고정 기준 |
|---|---|
| Teacher package | T0의 같은 checkpoint A/U, frozen/eval/no-grad |
| Student A 초기값 | T0의 최종 aligner 전체 복사. N2 donor로 되돌리지 않음 |
| Student U 초기값 | seed별 저장 초기 state. seed1234 교차 대조는 s1의 실제 초기 state를 hash까지 맞춤 |
| Backbone | W112, depth=[1,2,3], PAN1+MS8 → HRMS8 residual |
| Base·좌표 | 기존 bicubic M, residual base 1회 합산, M/GT/output frame 유지 |
| 복원 입력 | Native PAN을 예측 correction으로 한 번 보정. 추가 jitter를 U-Net에 넣지 않음 |
| Aligner view | 기존 margin4 → band별 z-score, normalization 중복 금지 |
| Warp | 기존 bicubic/border/align_corners=False, (dy,dx), HR pixel |
| FRESH50 | 50,000 optimizer updates, effective batch48, 동일 augmentation/data 순서 규약 |
| U optimizer | 기존 AdamW, peak LR1e-4, WD0.01, warmup100+cosine |
| A optimizer | J peak1e-5; AL peak3e-6; 나머지 optimizer 설정 동일 |
| Offset | trainable A에만 격번 1e-4, b=2 uniform disk, target=현재 Student native c0.detach |
| Gate | dT/aT detach, 실제 Student residual·edge는 live |
| Q12 calibration | 동일 T0 τR, 동일 J0-1234 exact50K λE0; variant는 λE0의 명시적 배율 |
| 추론 | Student A+U만. Teacher/GT/gate/추가 jitter/TTA/ensemble 없음 |

Teacher의 U-Net만 freeze하고 Student의 live aligned PAN을 Teacher에도 넣는 방식은 사용하지 않는다. F 정책에서 두 branch가 같은 보정값을 공유하는 최적화는 T0와 동일한 frozen A라는 검사를 통과한 경우만 기존 구현대로 유지한다. [A §2–4, §11; K §4–7; R1]

### 4.1 Loss 표기

\[
 e_T=\operatorname{mean}_c|Z_T-Y|,\quad e_S=\operatorname{mean}_c|Z_S-Y|,\quad k=\operatorname{mean}_c|Z_S-\operatorname{sg}Z_T|,
\]
\[
 d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\qquad
 a_T=\operatorname{sg}\operatorname{clip}_{[0,1]}\frac{[e_S-e_T]_+}{e_S+10^{-6}},
\]
\[
 L_0=\langle e_S\rangle,\quad L_D=\langle\alpha d_T e_S\rangle,\quad L_K=\langle\beta(1-d_T)a_Tk\rangle.
\]

`LE`는 GT와 Student 최종 HRMS의 밴드별 signed Scharr /32, reflect1, 내부1px 차이의 x/y 평균이다. GT variance/covariance 또는 Teacher edge KD가 아니다.

\[
 L_{Q12}=L_0+L_D+L_K+\lambda_E L_E,\qquad
 L_{R1}=L_0+L_D,\qquad L_{X02}=L_0+L_D+\lambda_E L_E.
\]

기본 α=1, β=0.1. LK 안에 β가 이미 들어 있다. Soft-active pixel 수나 가중치 합으로 재정규화하지 않는다. J/AL에서는 backend 전체가 A와 U를 모두 학습시키며 offset만 U에 직접 전달되지 않는다. [K §4–6]

## 5. 첫 필수 묶음: 서버 교차와 aligner LR

### S4-B00 / J0 — s1 대응 no-KD 교차 대조

| 항목 | 설정 |
|---|---|
| Run ID | `PAKD50_J0_W112_D123_WV3_T0_S1234_FRESH50_v1` |
| 실행 장소·역할 | s4 / BRIDGE |
| A / U loss | A: L0+LO, U: L0 |
| LR | A1e-5, U1e-4 |
| Teacher 사용 | 학습 loss에는 없음; 공통 fitting bin 진단에 사용 가능 |
| 선행조건 | T0 재현·초기 U hash·dataset/evaluator/AMP 설정 일치 |
| 비교 | s4 J0 vs s1 같은 seed J0; 이후 s4 JQ의 control |

서버별 sheet가 다르므로 s1과 s4에서 같은 논리적 Run ID를 사용할 수 있다. 다만 결과 집계의 execution key는 `(server, run_id, execution_id)`다. s4에서 같은 run을 다시 학습하면 기존 폴더를 덮어쓰지 않는다.

### S4-B01 / JQ — s1 대응 Q12 교차 대조

| 항목 | 설정 |
|---|---|
| Run ID | `PAKD50_JQ_W112_D123_WV3_T0_S1234_FRESH50_v1` |
| 실행 장소·역할 | s4 / BRIDGE 및 이후 탐색의 Q12 anchor |
| A / U loss | A: Q12+LO, U: Q12 |
| 계수 | α1, β0.1, λE0 |
| 선행조건 | s1의 λE0 package 확보 및 동일 hash |
| 비교 | **s4 JQ−J0**, s1의 동일 paired 차이와 방향 비교 |

\[
 \Delta_Q^{s4}=H_{JQ,s4}-H_{J0,s4},\qquad
 I_{env}=\Delta_Q^{s4}-\Delta_Q^{s1}.
\]

`I_env`는 한 seed에서의 환경·실행 민감성 진단이다. 순수 GPU 인과 효과나 유의성 검정값이 아니다. 같은 checkpoint 재평가가 다르면 seed가 아니라 evaluator/export/environment 동치부터 확인한다.

### S4-E00/E01 / AL0–ALQ — s4 첫 성능 탐색

| 항목 | AL0 | ALQ |
|---|---|---|
| Case ID | `AL0` | `ALQ` |
| Backend | N0 | Q12 |
| A peak LR | **3e-6** | **3e-6** |
| U peak LR | 1e-4 | 1e-4 |
| A 업데이트 | 첫 update부터 trainable | 첫 update부터 trainable |
| Off·jitter | 1e-4, b=2 유지 | 동일 |
| No-KD control | 본 run | **같은 s4 AL0** |
| 추가 비교 | AL0−J0 | **ALQ−JQ**, ALQ−AL0 |

**가설:** 완성 Teacher의 aligner를 Student에 복사한 상태에서는, 기존 native 재학습 때보다 작은 A 업데이트가 복원·정합 균형을 유지하는 데 유리할 수 있다. 이는 신규 가설이다. λoff를 줄여서 offset 반응 자체를 바꾸는 실험과 다르다.

검사할 것은 raw HQNR, Dλ/Ds, ERGAS@selected/last/plateau, native shift drift, loss별 A gradient다. ALQ가 JQ보다 좋더라도 ALQ−AL0가 0이면 이를 KD 순증분으로 설명하지 않는다.

첫 묶음은 **J0/JQ/AL0/ALQ 총4run**이다. λE가 아직 없으면 J0→AL0를 먼저 진행하고 JQ/ALQ는 package가 확보된 후 실행한다. 대기 중 임의의 λE를 쓰지 않는다.

## 6. 두 번째 묶음: Q12 단일축 조정 — 최대 두 개만

첫 scalar 라운드는 **J 정책·seed1234·공통 λE0**에서 최대 두 개다. 이렇게 해야 이후 AL과 결합할 경우 네 모서리의 대응 비교가 가능하다. 다음 다섯 후보를 전부 실행하는 grid가 아니다.

| ID | Case ID | α | β | λE/λE0 | 우선 적용할 관측 |
|---|---|---:|---:|---:|---|
| S4-E10 | **J_QA05** | **0.5** | 0.1 | 1 | Teacher-error 상위 bin 과강조, 반복 ERGAS·Dλ 비용 |
| S4-E11 | **J_QB005** | 1 | **0.05** | 1 | Soft gradient 충돌 또는 Teacher bias 모방 의심 |
| S4-E12 | **J_QB02** | 1 | **0.2** | 1 | Q12>X02 근거와 유효한 soft gradient가 있으나 신호가 작음 |
| S4-E13 | **J_QE025** | 1 | 0.1 | **0.5** | Edge로 인한 ringing·RR/FR 비용 또는 A gradient 충돌 |
| S4-E14 | **J_QE10** | 1 | 0.1 | **2** | GT edge 잔여오차가 크고 gradient 방향이 유효하며 ringing 증가 없음 |

모든 경우 A LR1e-5, U LR1e-4, horizon50K, 초기값·data order·offset·gate 정의는 JQ와 같다. **직접 anchor는 JQ, no-KD 기준은 J0**다. 별도의 N0를 중복 학습할 필요는 없다.

`QE025/QE10`은 calibration 목표 비율을 0.025/0.10로 해석할 수 있는 배율 이름이다. 실제 λE가 0.025/0.10이라는 뜻도, A/U parameter gradient가 그 비율이라는 뜻도 아니다.

### 6.1 구체적인 선택 규칙

**우선순위 1:** 반복 RR·분광 비용이 있으면 J_QA05. 하나의 seed에서 raw-selected ERGAS만 나쁜 것은 충분한 근거가 아니다. last·plateau·다른 서버의 대응 결과를 함께 본다.

**우선순위 2:** edge 또는 soft 중 진단 근거가 더 명확한 축 하나. edge 비용이면 J_QE025, 유효한 edge 부족이면 J_QE10. Soft가 유효하지만 약하면 J_QB02, 충돌이면 J_QB005.

β의 양쪽이나 edge의 양쪽을 처음부터 둘 다 열지 않는다. 근거가 불명확하면 scalar 수를 채우지 않고 확인 seed를 먼저 실행한다. 과거 후반 soft/hard loss ratio 0.052%만으로 β 확대를 결정하지 않는다. [K §7.4, §8.5]

### 6.2 Q12 soft 제거 대조의 위치

s1–s3의 XJ가 이미 같은 release·package로 확보되면 β 조정의 개발 참고로 사용한다. s4에서 soft 자체의 인과 기여를 주장할 때는 **같은 s4 seed의 XJ**가 필요하다.

`XJ = J + X02`, 즉 α1·edge λE0를 유지하고 β0. 기본 Q12의 XJ를 조정된 α/edge의 WIN에 그대로 대응시키지 않는다. WIN의 α·edge와 같은 `WIN_X02`를 따로 명명한다.

## 7. 세 번째 묶음: 좋은 두 단독 변화만 한 번 결합

### S4-E20 / AL + 선택된 scalar 하나

실행 조건은 **ALQ−JQ가 유망하고, J_scalar−JQ도 유망하며, 확인 묶음까지 끝낼 시간이 남는 것**이다. 단독 변화의 점수를 더해 결합 이득을 예측하지 않는다.

예를 들어 α0.5가 선택되면:

```text
case_id = AL_QA05
A peak LR = 3e-6
U peak LR = 1e-4
α = 0.5, β = 0.1, λE = λE0
A receives Q12_variant + LO; U receives Q12_variant
```

| 조건 | 기존 확보 run / 추가 run |
|---|---|
| 기본 LR + 기본 Q12 | JQ |
| 작은 A LR + 기본 Q12 | ALQ |
| 기본 LR + 선택 scalar | J_scalar |
| 작은 A LR + 선택 scalar | **AL_scalar, 추가1run** |

관측 가능한 결합 차이:

\[
 I_{comb}=H_{AL,scalar}-H_{AL,Q12}-H_{J,scalar}+H_{J,Q12}.
\]

같은 seed·환경의 difference-in-differences 진단이다. 양수 하나가 시너지의 보편적 증명은 아니다. AL_scalar−AL0도 별도로 보고한다. AL_QA05+β0.2+edge0.5처럼 세 축을 한꺼번에 결합하지 않는다.

ALQ가 불리하면 이 단계를 생략한다. J_scalar가 유리하면 그 방법 자체를 WIN으로 확인한다.

## 8. 방향이 다를 때의 대체 분기

| 분기 | 추가 case | 필요한 control | s4 실행 조건 |
|---|---|---|---|
| Q12보다 hard-only가 유망 | **JR** | 기존 J0, JQ | mainline R1 결과 또는 통합 soft/edge 비용이 근거 |
| Frozen이 더 유망 | **F0/FQ** | 같은 s4 F0 | JQ/ALQ가 불안정하고 mainline FQ가 유리 |
| 기존 추가 loss의 A 전달이 의심 | **PQ** | 기존 J0·JQ | s1 C1 routing 구현·회귀 검사 완료 후 |
| 초기 A 업데이트만 불안정 | **D0/DQ** | **D0 신규 필요** | s1 C1 freeze schedule 구현·resume 검사 완료 후 |
| 후반 plateau 추가 fitting | **EX0/EXM, CONT10** | 같은 exact50K parent에서 두 tail | s1 C2 구현 및 시간이 매우 제한될 때 |

이 표는 scalar와 결합 탐색을 **대체**하는 선택지다. 기본 예산에 전부 추가하지 않는다.

PQ는 A에 L0+LO, U에는 Q12를 준다. Native aligned PAN을 전체 detach하여 L0→A까지 끊는 구현은 틀리다.

D0/DQ는 global update0–4999에서 A를 freeze, 5000부터 열고 U의 50K schedule을 이어간다. A moment·WD가 freeze 중 바뀌지 않아야 하며 off parity를 유지한다. [P §7.2–7.4]

CONT10을 채택하면 parent A/U는 **exact50K**이고, 두 tail 모두 새 AdamW·warmup100+cosine10K, U peak1e-5, trainable A peak1e-6으로 한다. EX0는 plain GT(+off), EXM은 선택 backend(+off). λE/τR는 공통 package를 유지한다. Parent가 KD 계보이면 EX0는 전체 pipeline의 KD-free baseline이 아니다. [P §8.4]

50K용 epoch grid를 10K tail에 그대로 붙이지 않는다. 별도 tail evaluator·exact step 지원까지 확인하기 전에는 CONT10을 실행 불가 상태로 남긴다. 단순 `num_iter=60000` 수정은 이 protocol이 아니다.

## 9. 확인 묶음: seed3407을 결과를 보기 전에 고정한다

탐색 종료 시 `winner_lock.json`에 **case의 전체 resolved config·코드·T0·calibration hash·선택 근거**를 저장한 뒤 seed3407을 실행한다. 확인 seed를 보고 α/β/λE를 다시 고르면 해당 seed는 이후 개발용으로 표시한다.

| 선택한 WIN | seed3407에서 실행할 기본 묶음 | 이 묶음으로 확인하는 것 |
|---|---|---|
| JQ | J0, JQ | 기본 joint KD의 추가 seed 확인 |
| J_scalar | J0, JQ, J_scalar | KD 이득과 scalar의 추가 이득 |
| ALQ | AL0, JQ, ALQ | 작은 A LR 아래 KD 이득, Q12 아래 LR 정책 차이 |
| AL_scalar | AL0, ALQ, AL_scalar | AL 아래 KD 이득과 scalar 추가 이득 |
| FQ | F0, JQ, FQ | Frozen 아래 KD 이득, Q12 아래 정책 차이 |
| JR | J0, JR; 여유 시 JQ | R1 이득, 필요 시 Q12 대비 순위 |

AL_scalar의 확인 묶음만으로는 seed3407의 J↔AL 차이를 전부 확인한 것이 아니다. 그 질문까지 필요하면 JQ를 네 번째 run으로 추가한다.

**s4 탐색1seed + s4 확인1seed로 신규 WIN의 3-seed 검증이 끝나지는 않는다.** 기존 s1–s3에서 WIN을 실제로 실행하지 않았다면, 신규 WIN은 실행된 seed 수만 보고한다. 여유가 있으면 다음 묶음은 s4의 seed777에서 `control / unmodified anchor / WIN`이고, 또는 s2/s3에 같은 WIN을 넘겨 기존 대응 control과 비교한다.

동일 seed의 s2↔s4 반복은 같은 seed의 환경 교차다. 통계에서 두 결과를 독립 초기화로 중복 계산하지 않는다. 방법 간 비교는 공통 seed 집합에서 한다.

## 10. 실제 큐와 예산

### 10.1 기본 우선순위 큐

```text
G0  s4 환경·T0 재현·학습 경로·평가기·초기 U hash 검사
B0  J0_S1234
B1  JQ_S1234                     # λE0 확보 이후만
E0  AL0_S1234
E1  ALQ_S1234
E2  근거가 있는 J_scalar 1개
E3  근거가 있는 다른 축 J_scalar 1개, 또는 생략
E4  AL + scalar 1개 결합, 또는 생략
C0  winner_lock
C1  seed3407의 필수 control
C2  seed3407의 기본 Q12 anchor, WIN과 다를 때
C3  seed3407의 WIN
V0  시간이 남으면 exact WIN_X02 또는 두 번째 확인 묶음
A0  selected/last/plateau 재평가·source 연결·gspread 업로드 확인
```

λE가 없을 때 실행 순서는 J0→AL0→package 확인→JQ→ALQ로 바꿀 수 있다. 이는 case 학습 규약의 변경이 아니다. Stage 번호는 계산 의존성일 뿐, Q12를 임의 λ로 시작할 권한이 아니다.

### 10.2 s4에 30시간이 남고 50K run이 2시간일 때의 예시

**예시 가정:** gate·배포2h, 한 run에 정기 평가까지2h, 안전 배율1.1, 최종 감사4h. 실제 s4 처리시간을 측정한 수치가 아니다.

| s4 투입 후 경과시간 | 작업 | 학습 run 수 |
|---:|---|---:|
| 0–2.0h | 환경·자산·초기 U·평가기 동치, throughput | 0 |
| 2.0–10.8h | J0/JQ/AL0/ALQ | 4 |
| 10.8–15.2h | scalar 최대2개 | 2 |
| 15.2–17.4h | 유망한 단독 변화의 결합1개 | 1 |
| 17.4–24.0h | seed3407 control/anchor/WIN | 3 |
| 24.0–26.0h | 재시도·미측정 후처리 여유 | 0 |
| 26.0–30.0h | 재평가·집계·export·upload 확인 | 0 |

**총10run, 실험 slot 안전 예산22h + gate2h + audit4h =28h**, 나머지2h는 여유다. 결합이 불필요하면 1run을 채우기 위해 새 case를 만들지 말고 X02 또는 확인에 넘긴다. WIN이 JQ라면 확인2run만 필요할 수 있다.

### 10.3 가용시간이 짧을 때

아래 상한도 run2h·gate2h·audit4h·안전 배율1.1 가정이다. `H_remain`은 **s4 투입부터 상위 최종 마감까지 남은 시간**이다.

| H_remain | 계산상 50K run 상한 | 권장 축소 |
|---:|---:|---|
| 12h | 2 | J0/JQ 서버 교차 **또는** 이미 고정한 WIN의 확인쌍. 신규 탐색 없음 |
| 24h | 8 | core4 + scalar1 + 확인3. 결합 탐색 생략 |
| 30h | 10 | core4 + scalar2 + 결합1 + 확인3 |
| 40h | 15 | 위10run 완료 후 두 번째 확인 묶음·X02·필요한 정책 대조 우선 |

`floor((H_remain−gate−audit)/(1.1×run시간))`은 단순 상한이다. 실제 case별 시간·calibration 지연·mandatory 후처리가 다르면 다시 산정한다. 상위 캠페인의 최종감사가 이미 다른 시각에 고정되었으면 그 마감에 맞춰 계산한다.

### 10.4 복수 GPU가 생길 때

GPU slot을 늘릴 수 있어도 실효 batch48과 update 정의는 유지한다. 첫 확장은 DDP가 아니라 **서로 다른 대응 run을 각 slot에 독립 배정**하는 방식이다. Seed3407은 slot이 비어 있다는 이유로 WIN 선택 전에 실행하지 않는다. 공용 I/O 병목과 GPU별 처리시간을 계측한다.

## 11. 평가·선택·seed 해석

### 11.1 공통 checkpoint 계약

\[
\mathcal G=\{1010,2020,\ldots,49490\}\cup\{50000\}.
\]

총50후보다. Actual `eval_epoch=5`와 loader 길이가 이 grid를 생성하는지 s4에서 검사한다. 데이터 규모나 batch가 바뀌어 grid가 바뀌면 같은 protocol로 집계하지 않는다.

- 공식 선택: `best_hqnr` = raw-original HQNR, 기존 running-max tie band1e-4 및 fSCC comparator 유지.
- 고정 시점: exact50000 `last`.
- 후반 plateau: **45450, 46460, 47470, 48480, 49490, 50000**의 공통6개 평가.

Selected·last·plateau는 서로 다른 요약이다. RR와 FR를 다른 checkpoint에서 가져와 하나의 model row로 합치지 않는다. 논문 FR20과 원 PAN 참조를 유지한다. HQNR(V64), self-aligned 점수, run별 최고의 scene 조합은 목표 달성값이 아니다. [R1; A §6]

### 11.2 ERGAS는 변동을 감안하되 삭제하지 않는다

\[
\Delta H=H_{WIN}-H_{control},\quad
\Delta E=ERGAS_{WIN}-ERGAS_{control},\quad
\Delta E_{rel}=100\Delta E/ERGAS_{control}.
\]

ΔH 양수, ΔE 음수가 개선이다. `ergas_at_raw_selected`, `ergas_at_exact50k`, `ergas_plateau45_50`를 따로 기록한다.

상위 계획의 운영 기준을 계승한다. 한 block의 +2% 이내 ERGAS 후퇴로 즉시 탈락시키지 않는다. Last/plateau 평균 +2% 초과와 반복 양성 ΔE가 있으면 RR trade-off로 표시한다. 25K 이후 공통3회에서 +5% 초과이면 checkpoint·scale·drift 감사를 먼저 한다. **2%/5%는 통계 검정선이나 화질 보장선이 아니라 사전 지정한 경고 기준**이다. 신규 WIN이2seed뿐이면 2/3 기준을 통과했다고 쓰지 않는다. [P §10.4]

Core J0/JQ는 계산·계약 실패 외에는50K 완주가 원칙이다. Optional variant는 기존 계획의 25K 후 부진·drift와 잔여 budget을 근거로 다음 seed 승격을 보류할 수 있다. 조기 중단 run을 50K 성능으로 업로드하지 않는다.

### 11.3 목표 판정과 최종 순위

**Asset hit:** 원 정밀도의 재현 raw HQNR≥0.959. 0.95896을 반올림한 0.9590은 미달이다. ≥0.960은 별도 표시한다.

**Repeated gain:** 공통3개 대응 block의 평균·중앙값 ΔH 양성, 적어도2/3 양성이라는 상위 계획의 운영 기준. 적은 seed의 유의성 증명은 아니다.

**Method target:** 같은 고정 방법의 공통3block 평균≥0.959. 최고 seed 하나나 server 중복을 섞어 평균을 만들지 않는다.

보고 항목은 n_unique_seed, n_execution, 평균±표본sd, 중앙값, min/max, paired Δ, target-hit 수다. n=1의 sd는0이 아니라 NA다. Seed1234의 s1/s4 복제는 환경 교차표에 별도 기록한다.

신규 WIN이 목표 근처에 들어오면 다음 작업은 새로운 계수 탐색보다 seed 확인이다. 목표를 넘지 못했으면 실제 최고값·반복 범위·RR 비용·미실행 case를 그대로 남긴다. 0.960 달성을 보장하는 수치 예측은 하지 않는다.

### 11.4 최소 진단

| 그룹 | 필요한 기록 |
|---|---|
| 정합 | cS 분포, cS−cT, selected/last drift, 고정 synthetic offset response |
| 감독 | plain L1, weighted hard/soft, raw/weighted edge, d/a 분위수, soft active fraction |
| Gradient | A/U 각각의 hard·soft·edge·offset norm과 cosine, 고정 diagnostic batch |
| Fitting | 공통 Teacher-error bin의 N0/후보 잔여 오차, 밴드별·edge 오차 |
| 평가 | per-scene raw HQNR/Dλ/Ds, RR ERGAS/SAM/PSNR, 같은 checkpoint의 fSCC |
| 실행 | training_git_sha, config/Teacher/A/U-init/calibration/evaluator hashes, server/GPU 환경 |

단일 모델20장 장면의 표준편차는 학습 seed 표준편차가 아니다. 이번 FR20은 개발·checkpoint 선택에 반복 사용된 benchmark라는 한계를 유지한다.

## 12. s1→s4 배포와 결과 업로드

s1에서 **학습 경로를 바꾸지 않는 s4 지원·gspread 변경 release**를 우선 만든다. 기존 release에서 신규 trainer 기능까지 한 번에 합치지 않는다.

```text
s1 개발 checkout: 구현 → unit/regression → 명시적 파일만 commit/push
s4 관리 checkout: fetch/pull --ff-only → 승인 commit의 실행 worktree 생성
s4 실행 worktree: data/assets 연결 → manifest/hash 검증 → gate → 승인 queue 실행
별도 publisher checkout: gspread schema 검증 → 완료 run 결과 업로드
```

학습이 돌아가는 worktree에서 pull하지 않는다. Publisher만 먼저 갱신할 때도 `training_git_sha`와 `uploader_git_sha`를 각각 남긴다. 코드 update가 곧 기존 학습 run의 코드 version 변경은 아니다.

업로드는 기존 `tools/_upload.sh → gspread/gspread_upload.py` 경로를 유지한다. 대상은 같은 `pan-cvpr27` spreadsheet의 **`WV3-s4` worksheet**다. 새 spreadsheet를 만들지 않는다.

기존 B:W metric format을 유지하고 **X열 `통합실험`**을 붙인다. 예: `PAKD50 / JQ / FRESH50`. J0·F0·AL0 등 통합 대조군도 이 열을 채운다. 분류·schema 이관·Notes 열 처리·재시도는 동반 구현 문서에 명시했다.

**현재 외부 변경 상태:** 사용자께서 직접 Sheet 수정이 불필요하다고 알려주기 전에 `WV3-s4` 빈 탭과 X3 `통합실험` 헤더가 생성되었다. 기존 탭의 값은 변경하지 않았고 s4 실험 결과도 쓰지 않았다. 이후 직접 Sheet 조작은 하지 않았다. 업로더는 이 빈 탭이 이미 있으면 안전하게 재사용하고, 없으면 같은 형식으로 생성하도록 구현한다.

## 13. 작업 완료의 기준

s4 배정은 등록된 case 수가 아니라 **선택한 비교 묶음의 완료 여부**로 평가한다. 다음 결과를 인계한다.

| 산출물 | 내용 |
|---|---|
| `s4_manifest.json` | 실사용 가능 시각, 공유 마감, hardware/환경, release·asset identity |
| `s4_case_registry_resolved.json` | 실제 승인한 case·seed·control·순서, 실행/대기/제외 이유 |
| `winner_lock.json` | seed3407 확인 전에 고정한 방법·계수·hash·선택 근거 |
| `s4_paired_results.csv` | 각 control과의 selected/last/plateau ΔH·ΔERGAS |
| `s4_environment_bridge.csv` | seed1234 s1↔s4 교차, seed 수에 중복 합산하지 않음 |
| `s4_seed_summary.csv` | 고유 seed와 실행 수를 구분한 후보 요약 |
| `s4_upload_status.json` | 실제 worksheet/row·uploader revision·업로드 성공/대기/실패 |
| `s4_final_report.md` | 목표 달성 수준, 비용, 예외, 미실행 case와 다음 한 가지 우선순위 |

## 14. 근거 registry

[A] `PAN_Aligner_L1E4_TechnicalSpec_Evidence_KD_Handoff_2026-09-14.md`: §2–6, §8.6, §11.  
[K] `PAN_KD_Q12_R1_Spec_and_Aligner_Interface_2026-09-14.md`: §4–9.  
[P] `PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md`: §5–13.

[R1] [PAKD50 C0 구현 노트, 검토 commit 고정](https://github.com/hojunking/PAN-Crafter-repro/blob/3d068859e0d4b8d3cbd74242419fbf9fc27b0c4f/research_log/2026-09-14_pakd50-implementation.md).  
[R2] [config 생성기](https://github.com/hojunking/PAN-Crafter-repro/blob/3d068859e0d4b8d3cbd74242419fbf9fc27b0c4f/tools/gen_pakd50_configs.py).  
[R3] [prepare·자산 재현·예산 검사](https://github.com/hojunking/PAN-Crafter-repro/blob/3d068859e0d4b8d3cbd74242419fbf9fc27b0c4f/tools/pakd50_prepare.sh).  
[R4] [결과 업로드 체인](https://github.com/hojunking/PAN-Crafter-repro/blob/3d068859e0d4b8d3cbd74242419fbf9fc27b0c4f/tools/_upload.sh).

**한 문장 인계:** s4는 T0·W112D123·동일 평가를 유지한 채 J0/JQ 서버 교차와 AL0/ALQ를 먼저 끝내고, 진단으로 고른 scalar 최대2개와 결합 최대1개만 탐색한 뒤, 고정 WIN을 seed3407에서 대응 검증한다.
