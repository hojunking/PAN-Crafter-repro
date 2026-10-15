# PAKD50 파생 실험 재배정 — s2 / s4 / s5

**작성일:** 2026-09-15 KST  
**현재 Sheet·시각 기준:** 2026-09-15T00:05:50+09:00 부근 live read  
**상태:** 실행·배포 인계 계획. Repository push, Sheet 변경, 서버 프로세스 변경을 수행하지 않았다.

## 1. 이번 배정의 범위

s2·s4·s5만 학습에 사용한다. s1은 별도 분석용이며 s3에도 신규 작업을 배정하지 않는다. 재평가 전수 작업, A/U 교차 추론, gradient 진단 캠페인, 새 분석 보고서 작성은 이 큐의 업무와 시간에 포함하지 않는다. 기존 학습 중 정기 평가와 기본 export/gspread는 유지한다.

기존 결과로부터 파생된 세 묶음을 바로 실행한다.

- s2: Q12의 output soft / GT edge 전체 손실 제거 및 Teacher-free edge 대조.
- s4: Student 단계 offset 연습 제거 + 같은 seed frozen 대조; 이어서 s3에 배정했던 backend 대조를 seed1234로 이전.
- s5: 남은 protected routing, 25K 이후 aligner 동결, offset 제거를 seed2026으로 반복.

**기본 16run(4+6+6), 외부 분석에서 WIN 확정 후 확인 최대 6run(0+3+3)이다.** 완료된 FQ·ALQ·D0·DQ 및 J0 재현 실험을 다시 큐에 넣지 않는다. PQ는 Sheet에 없다는 이유만으로 미실행이라고 단정하지 않는다. 이미 학습 중이면 그 run을 유지하고 남은 시간만 센다.

## 2. 이번에 읽은 실제 Train(h)

출처: pan-cvpr27의 WV3-s2 / WV3-s4 / WV3-s5, B3:X200(또는 s4 X198). 실제 읽은 행과 값은 `live_sheet_runtime_snapshot.json`에 보관했다.

| 서버 | Case/version | Sheet 행 | Train(h) | 환산 hh:mm:ss | raw selected HQNR |
|---|---|---:|---:|---|---:|
| s2 | J0 v1 | 67 | 1.97 | 01:58:12 | 0.9549 |
| s2 | F0 v1 | 69 | 1.91 | 01:54:36 | 0.9530 |
| s2 | JQ v1 | 70 | 2.33 | 02:19:48 | 0.9583 |
| s2 | FQ v1 | 71 | 2.25 | 02:15:00 | 0.9513 |
| s4 | AL0 v1 | 6 | 1.17 | 01:10:12 | 0.9558 |
| s4 | J0 v1 | 7 | 1.17 | 01:10:12 | 0.9594 |
| s4 | JQ v1 | 8 | 1.39 | 01:23:24 | 0.9558 |
| s4 | ALQ v1 | 9 | 1.39 | 01:23:24 | 0.9550 |
| s4 | J0 v2 | 10 | 1.17 | 01:10:12 | 0.9568 |
| s5 | J0 v1 | 6 | 1.35 | 01:21:00 | 0.9570 |
| s5 | JQ v1 | 7 | 1.36 | 01:21:36 | 0.9576 |
| s5 | D0 v1 | 8 | 1.15 | 01:09:00 | 0.9530 |
| s5 | DQ v1 | 9 | 1.34 | 01:20:24 | 0.9549 |

**1.39h는 1시간 39분이 아니라 1시간 23분 24초다. 2.33h는 2시간 19분 48초다.** 표시값이 소수 둘째 자리까지라 환산 초는 표시값의 산술값이지 초 단위 원시 측정 정밀도가 아니다.

현재 조회에서 s2 FQ(2.25h), s4 ALQ(1.39h)와 J0_v2(1.17h), s5 DQ(1.34h)가 확인됐다. 기존 보고서에서 실행 중이던 상태를 현재 상태로 복사하지 않는다. J0_v2는 v1과 다른 실행 identity이므로 합치거나 고득점 v1만 선택하지 않는다. 대응 비교의 고정 control hash는 기존 분석 담당자가 확정한 것을 사용한다.

### Train(h)의 의미

조회한 `gspread/gspread_upload.py`의 `collect()`는 `meta/finished_at.txt - meta/started_at.txt`를 시간으로 바꾼다. `tools/run.sh`는 `main.py` 실행을 감싸므로 그 실행 안의 정기 평가 시간은 이미 들어 있다. 이 값에 다시 `50회 평가 × 평균 평가시간`을 통째로 더하지 않는다. run 종료 뒤 별도 `_upload.sh`의 평가/업로드/큐 전환은 구분한다.

Sheet만으로 현재 프로세스 진행률, 남은 시간, 후처리 시간을 직접 얻을 수는 없다. `finished_at.txt`는 실패 때도 생길 수 있으므로 완료 산출물과 exit status를 함께 보며, 행이 없으면 곧바로 새 run을 중복 시작하지 않는다.

## 3. 시간 산정: 관측, 대용 추정, 운영 여유를 분리

새로운 case의 완료 실측은 아직 없다. 아래 값은 같은 서버의 완료 case를 대용으로 사용한 **계획 기준값**이지 새 case를 실제 측정했다는 뜻이 아니다.

| 서버 | N0형 기준 | Q12/Teacher형 기준 | 별도 취급 |
|---|---:|---:|---|
| s2 | 1.97h | 2.33h | 이번 4개 case는 보수적으로 모두 2.33h 대용값 |
| s4 | 1.17h | 1.39h | F0/RC0는 J0 대용값, 나머지 4개는 JQ 대용값 |
| s5 | 1.35h | 1.36h | D0의 1.15h를 다른 case에 일반화하지 않음. PQ는 1.80h 임시 가예약 |

PQ에는 손실별 gradient routing 비용이 있으나 해당 서버의 완료 기록이 없다. **1.80h는 실측값이나 성능 예측이 아니라 미측정 경로에 대한 가예약**이다. 첫 실제 timing/완주 시간이 나오면 곧바로 교체한다. LF0/LFQ/RC도 동결·offset 제거에 따른 실제 시간 절감을 선반영하지 않고 같은 서버 기본 모델의 시간을 대용값으로 둔다.

모든 run의 초기 슬롯 예약은 아래 하나의 식으로 계산한다.

```text
reservation_h = 1.10 × reference_train_h + 10/60
```

10%는 실행 시간 변동 여유, 10분은 run 뒤 기본 export/업로드/전환용 미실측 가예약이다. 두 값은 관측값이 아니며 결과 수치로 기록하지 않는다. 학습 안 정기 평가는 다시 더하지 않고, 별도 분석 비용은 이 식에 넣지 않는다. 실제 후처리가 10분을 초과하면 숨기지 말고 이후 큐 비용을 갱신한다. 실제 출력시간을 재측정할 때는 warmup 및 CUDA 동기화를 포함하고 서로 다른 forward/backward 경로를 구분한다.

## 4. 서버별 확정 실행 순서

### s2 — seed 777

| 순서 | Case | 시간 산정 기준 h | 기준 구분 | 예약 h | 누적 예약 |
|---:|---|---:|---|---:|---|
| 1 | JR | 2.33 | 같은 서버 완료 case 대용값 | 2.73 | 02:43:47 |
| 2 | XJ | 2.33 | 같은 서버 완료 case 대용값 | 2.73 | 05:27:34 |
| 3 | J_R3_NOEDGE | 2.33 | 같은 서버 완료 case 대용값 | 2.73 | 08:11:20 |
| 4 | J_N0_EDGE | 2.33 | 같은 서버 완료 case 대용값 | 2.73 | 10:55:07 |

### s4 — seed 1234

| 순서 | Case | 시간 산정 기준 h | 기준 구분 | 예약 h | 누적 예약 |
|---:|---|---:|---|---:|---|
| 1 | F0 | 1.17 | 같은 서버 완료 case 대용값 | 1.45 | 01:27:13 |
| 2 | RC0 | 1.17 | 같은 서버 완료 case 대용값 | 1.45 | 02:54:26 |
| 3 | RCQ | 1.39 | 같은 서버 완료 case 대용값 | 1.70 | 04:36:11 |
| 4 | JR | 1.39 | 같은 서버 완료 case 대용값 | 1.70 | 06:17:55 |
| 5 | XJ | 1.39 | 같은 서버 완료 case 대용값 | 1.70 | 07:59:40 |
| 6 | J_R3_NOEDGE | 1.39 | 같은 서버 완료 case 대용값 | 1.70 | 09:41:24 |

### s5 — seed 2026

| 순서 | Case | 시간 산정 기준 h | 기준 구분 | 예약 h | 누적 예약 |
|---:|---|---:|---|---:|---|
| 1 | PQ | 1.80 | 미실측 routing 가예약 | 2.15 | 02:08:48 |
| 2 | F0 | 1.35 | 같은 서버 완료 case 대용값 | 1.65 | 03:47:54 |
| 3 | LF0 | 1.35 | 같은 서버 완료 case 대용값 | 1.65 | 05:27:00 |
| 4 | LFQ | 1.36 | 같은 서버 완료 case 대용값 | 1.66 | 07:06:46 |
| 5 | RC0 | 1.35 | 같은 서버 완료 case 대용값 | 1.65 | 08:45:52 |
| 6 | RCQ | 1.36 | 같은 서버 완료 case 대용값 | 1.66 | 10:25:37 |


### 시간 합계

| 서버 | 신규 예약 run | 학습 대용값 합 h | 여유 포함 슬롯 예약 h | hh:mm:ss |
|---|---:|---:|---:|---|
| s2 | 4 | 9.32 | 10.9187 | 10:55:07 |
| s4 | 6 | 7.90 | 9.6900 | 09:41:24 |
| s5 | 6 | 8.57 | 10.4270 | 10:25:37 |

검산:

```text
s2 = 1.10 × (4 × 2.33) + 4 × 10/60 = 10.9186667 h
s4 = 1.10 × (2 × 1.17 + 4 × 1.39) + 6 × 10/60 = 9.6900000 h
s5 = 1.10 × (1.80 + 3 × 1.35 + 2 × 1.36) + 6 × 10/60 = 10.4270000 h
```

기본 단계 학습 대용값 합은 **25.79 server-slot-h**, 여유 포함 합은 **31.0357 server-slot-h**다. 세 서버를 병렬로 쓸 때의 경과시간은 합이 아닌 최대값으로 **10.9187h = 10:55:07**다. 완료 case를 건너뛰거나 이미 진행 중인 PQ의 일부가 끝났으면 실제 잔여 비용만 남긴다.

## 5. 각 case의 실험 의미와 대조

| Case | 실제 변화 | 현재 생성기 상태 |
|---|---|---|
| JR | Teacher-error 재가중 GT hard만; output soft와 GT edge 없음 | 기존 등록 |
| XJ | 재가중 GT hard + GT edge. Output soft를 A/U 전체에서 제거 | 기존 등록 |
| J_R3_NOEDGE | 재가중 GT hard + adaptive output soft. GT edge 전체 제거 | 생성기 등록 필요 |
| J_N0_EDGE | plain GT L1 + GT edge. Teacher 정보를 학습 loss에 사용하지 않음 | 생성기 등록 필요 |
| F0 | T0 aligner 완전 동결 + fresh U-Net plain GT 학습 | 기존 등록 |
| RC0 | T0 aligner trainable, LR 1e-5; Student 단계 offset 연습 제거; plain GT | 생성기 등록 필요 |
| RCQ | RC0의 정합 정책 + Q12 전체 | 생성기 등록 필요 |
| PQ | U: Q12 전체; A: plain GT L1 + offset만. qD=qK=qE=0 | 기존 등록 |
| LF0 | 0–24999 joint N0; 25000부터 A와 offset 동결/중단, U는 50K까지 N0 | 기존 등록 |
| LFQ | 0–24999 joint Q12; 25000부터 A와 offset 동결/중단, U는 50K까지 Q12 | 기존 등록 |

### s2 / s4: backend 구성요소 전체 제거

```text
JR             = weighted_GT
J_R3_NOEDGE     = weighted_GT + adaptive_output_soft
XJ             = weighted_GT + lambda_E * GT_edge
JQ             = weighted_GT + adaptive_output_soft + lambda_E * GT_edge
J_N0_EDGE      = plain_GT + lambda_E * GT_edge
```

JQ−XJ는 edge를 유지한 output soft의 추가 기여, JQ−J_R3_NOEDGE는 soft를 유지한 edge의 추가 기여다. J_R3_NOEDGE−JR는 edge 없는 soft 효과다. J_N0_EDGE는 Teacher 없이 GT edge만으로 얻는 결과다. s2의 N0/JQ와 s4의 N0/JQ는 기존 완료 자산을 사용한다.

s4의 JR/XJ/J_R3_NOEDGE는 기존 s3 담당 업무의 두 번째 seed 대조를 이전한 것이다. 원래 s3의 seed2026 run을 s4의 seed1234로 이름만 바꾸는 것이 아니라, s4의 저장 초기값으로 새 학습한다. s2·s4에서 두 seed777/1234의 동일 대조를 얻는다.

### s4 / s5: offset 연습을 Stage 2에서 제거

RC0/RCQ는 **Stage 1 L1E4와 Teacher T0를 그대로 둔다.** Student aligner는 T0에서 복사하고 LR 1e-5로 계속 학습한다. Stage 2의 relative-offset 보조 연습만 제거한다.

```yaml
# Semantic fragment; requires case/policy registration in the config generator.
aligner_policy: A-FT
protocol: I-NATIVE-TRANSFER
aligner_lr: 1.0e-5
corruption:
  radius_hr: 0.0
aux:
  offset_weight: 0.0
```

`I-AEQ`에 weight=0만 넣는 방식은 registry 계약과 맞지 않는다. RCQ는 기존 Q12 식을 동일하게 사용한다. Native PAN 정합·reconstruction→A gradient는 유지한다. RC0−J0, RCQ−JQ, RCQ−RC0, RC0/F0 및 RCQ/F0를 같은 서버·seed에서 비교할 수 있다.

s4와 s5에 F0를 각각 배정한 이유는 다른 서버의 F0로 local strong control을 대체하지 않기 위해서다. RC0/RCQ도 두 서버에서 서로 다른 seed1234/2026으로 반복한다.

### s5: routing과 후반 동결

PQ는 A에 plain_GT+offset만, U에 Q12 전체를 보낸다. Soft와 edge 자체를 전체 모델에서 지우는 case가 아니므로 XJ/J_R3_NOEDGE와 구분한다.

LF0/LFQ는 처음 25K에는 기존 J0/JQ와 같은 규약, 25K부터 A만 동결하고 offset도 중단한다. U는 원래 50K 전역 cosine을 유지한다. Teacher target 경로는 계속 고정이다. 기본은 fresh50이며 정확한 25K full training state가 없으면 중간 checkpoint를 25K로 대신 사용하지 않는다.

현재 s5 D0와 DQ는 완료 행이 있으므로 재실행하지 않는다. PQ가 실행 중이면 완료 후 F0부터 이어 간다. PQ가 실패로 종료됐으면 원인을 수정한 version으로 재개/재실행 비용을 따로 기록하며 완료로 세지 않는다.

## 6. 공통 설정 — 손실·구조를 동시에 넓히지 않음

W112–D123, T0 A/U package, Student=T0 aligner+fresh U-Net, 50K/batch48, U LR1e-4, A LR1e-5(단 F/LF 동결 구간 제외), alpha1/beta0.1, tau_R=0.012463942170143127, lambda_E=0.09075170336956798을 유지한다. Offset 유지 case는 1e-4 격번이며 추가 PAN perturbation은 aligner에만 사용한다. Edge를 켠 모든 비교는 같은 lambda_E를 쓰고 case마다 재calibration하지 않는다.

기존 raw-original HQNR 선택 규약, exact50K와 고정 plateau 로그는 유지한다. V64로 목표값을 대체하지 않는다. 판단·분석은 별도 담당 작업이며 학습 서버는 기본 결과 파일과 gspread 업로드만 남긴다.

D121/새 Teacher/Teacher-copy/variance/새 scalar sweep/추가 gradient 분석 캠페인은 이번 확정 16run에 넣지 않는다. 이 문서는 아키텍처 탐색을 취소하는 결정이 아니라, 요청한 현 결과의 파생 실험 재배정이다.

## 7. 확인 seed 예약 — 기본 단계 후, 외부 분석의 WIN 전달 시에만

s4 seed3407, s5 seed9091에 각각 최대 3run을 예약한다. WIN·같은 policy의 no-KD control·F0를 동일 새 seed에서 학습한다. F0가 이미 matched control이면 중복 실행하지 않아 2run으로 줄어든다. WIN 선정 후 case/계수/release를 고정하고 확인 seed 결과를 본 뒤 계수를 다시 바꾸지 않는다.

| 서버 | 확인 seed | 예약 run | 비용 산정 가정 | 예약 h |
|---|---:|---:|---|---:|
| s4 | 3407 | 최대 3 | 각 run 1.80h 가예약 + 10% + 10분 | 6.4400 |
| s5 | 9091 | 최대 3 | 각 run 1.80h 가예약 + 10% + 10분 | 6.4400 |

확인 case가 아직 정해지지 않아 **실측 완료시간으로 제시하지 않는 포괄 예약**이다. 기본 단계에서 측정한 동일 서버/정책 시간으로 바꾸며 routing이 더 느리면 1.80h를 상한처럼 고집하지 않는다.

기본 단계의 모든 결과를 기다려 WIN을 고르는 barrier를 적용하면:

```text
core parallel       = max(10.9186667, 9.6900000, 10.4270000)
                    = 10.9186667 h
confirmation        = max(6.44, 6.44) = 6.44 h
core + confirmation = 17.3586667 h = 17:21:31
```

따라서 **최대 22run, 합산 43.9157 server-slot-h, 병렬 경과시간 예약 17.3587h**다. 이는 배포·구현 대기, 현재 이미 실행 중인 비중복 작업의 잔여시간, 외부 WIN 결정 대기를 제외한 계산이다. `max(core+confirmation의 서버별 합)`만 써서 전체 결과 대기 시간을 누락하지 않는다.

## 8. 실제 마감과 admission

공통 clock의 학습 마감은 **2026-09-16 11:22:31 KST**, 최종 마감은 **15:22:31 KST**다. 조회 기준시각 2026-09-15 00:05:50 KST부터 학습 마감까지 **35:16:41**다. 보고서의 작성 당시 잔여 38.6h를 재사용하지 않는다. 최종 4h는 training_deadline 뒤에 이미 남겨져 있으므로 이 35:16:41에서 다시 4h를 빼지 않는다.

```text
ready_s = actual server availability + required deployment/implementation delay
finish_core = max_s(ready_s + remaining_core_cost_s)
finish_all  = finish_core + external_winner_wait + max_s(confirmation_cost_s)
admit only if finish_all <= training_deadline
```

현재 Sheet로는 실행 중인 프로세스와 step을 확인할 수 없으므로 정확한 절대 종료 시각을 확정하지 않는다. 위 시간은 동일 시작·빈 슬롯 기준의 예약값이다. 현재 실행이 이번 큐와 겹치면 같은 run의 remaining만 남기고, 다른 run이면 그 잔여시간을 선행 대기로 추가한다. 네트워크/API 지연이 표의 10분 allowance를 넘는다면 actual post time으로 갱신한다.

배포/대기 없이 조회시각부터 시작한다는 가정에서는 위 기본+확인 예약이 마감 내에 들어온다. 그러나 그것을 실제 서버 진행 상황을 확인한 일정으로 보고하지 않는다. 신규 case를 다른 서버에 이동할 때에는 기존 hash의 control을 단순히 가져다 비교하지 말고 same-server 대응 비교 필요성을 함께 관리한다.

## 9. 실행 인계

1. 새 run을 시작하기 전에 각 서버의 완료/실행 중 run을 전체 identity로 제거한다. 이미 완료된 FQ/ALQ/D0/DQ를 재실행하지 않는다.
2. s2 기존 기본 gate의 FR 등 옛 꼬리 큐가 새 순서를 앞질러 가지 않도록, **다음 run 경계에서** 본 서버별 명시 순서로 편성한다. 진행 중 프로세스를 무작정 종료하지 않는다.
3. J_R3_NOEDGE/J_N0_EDGE/RC0/RCQ 네 case와 RC policy를 생성기에 등록한 release를 사용한다. 아직 등록되지 않은 이름을 단순히 extra_priority에 넣는 것으로 구현됐다고 처리하지 않는다. s2는 JR/XJ, s4는 F0, s5는 PQ/LF처럼 기존 등록 case를 먼저 수행할 수 있다.
4. 새 policy/손실 mapping 최소 실행 검사는 필요하지만, 별도 재분석 프로젝트를 학습 큐에 다시 배정하지 않는다. 기존 run의 학습중 checkout을 pull로 바꾸지 않는다.
5. Control 버전·Teacher/init/calibration hash는 별도 분석 담당자가 확정한 자산을 그대로 사용한다. v1/v2 중 결과가 좋은 쪽을 사후 선택하지 않는다.
6. 업로드는 기존 `_upload.sh → gspread_upload.py`, tab `WV3-s2/s4/s5`, X열 `통합실험`을 유지한다. 수동 Sheet 작업은 없다.

## 10. 근거

- 사용자 첨부 `2026-09-14-pakd50-campaign-results.md` §2–3: joint Q12, F0, AL0 및 후반 패턴을 근거로 삼았다. 본 문서는 그 결과를 다시 분석하는 보고서가 아니다.
- Live Sheet 위 표 및 `live_sheet_runtime_snapshot.json`: 이번 실제 Train(h), 완료 행, 버전 구분.
- Repository `gspread/gspread_upload.py` collect의 시간 계산, `tools/run.sh` 종료시각 저장, `tools/gen_pakd50_configs.py`의 기존 policy/case 매핑과 `assets/pakd50/campaign_clock.json`. 조회 시점 code search commit: 4054135aa4a00443edb893433563c1f738c0c785. 현재 서버가 그 commit으로 실행 중이라고 단정하지 않았다.

동봉 `allocation_plan.json`은 **계획용 명세**이며 Trainer YAML이 아니다. 서버별 txt는 case 순서 명세이며 실제 실행 queue를 자동 변경하지 않는다. 시간 검산은 `verify_budget.py`로 재실행할 수 있다.
