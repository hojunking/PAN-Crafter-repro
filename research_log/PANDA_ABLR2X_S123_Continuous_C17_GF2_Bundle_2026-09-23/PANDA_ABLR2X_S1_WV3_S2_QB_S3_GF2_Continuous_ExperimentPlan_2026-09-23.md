# PANDA ABLR2X — s1/WV3·s2/QB cycle 확장 + s3/GF2 component cycle

**설계일:** 2026-09-23  
**확장 ID:** `ABLR2X_S123_C17_20260923_v1`  
**s1/s2:** 진행 중인 ABLR2 상태·원 case·seed를 보존하는 append-only 확장  
**s3:** `PANDA_ABL_GF2_S3_ADAPTIVE_LOOP_20260923_v1` 신규 component campaign  
**실행 기간:** 총시간·cycle 수 제한 없이, 사용자 중지 또는 로컬 안전 조건까지 계속  
**신규 Student case:** `C17 = C03-TZERO`  
**보호 대상:** s4/s5의 작업, 세 센서의 기존 생산·PANMIX·P40·논문 재현 결과  
**문서 상태:** 실험 설계·case manifest·정적 검증 자료. 이 문서 작성으로 원격 코드 배포, queue 수정, 학습 기동을 수행하지 않았다.

> 이번 요청은 새로운 구성 여러 개를 추가하는 것이 아니다. 기존 C00–C16에 **C17 하나**를 더하고, 같은 18개 Student 구성의 반복 엔진을 GF2까지 확장한다. s1/s2는 처음부터 다시 시작하지 않는다. s3의 GF2 component 실험에는 PANMIX나 생산 모델의 100K/120K recipe를 섞지 않는다.

---

## 0. 적용 결정 요약

| 항목 | 결정 |
|---|---|
| s1 | WV3 ABLR2 계속. 이미 진행한 sweep의 C17을 보충하고 이후 모든 full sweep에 포함 |
| s2 | QB ABLR2 계속. s1과 같은 append 정책, 서버 간 대기 없음 |
| s3 | GF2 전용 ABLR2X lane 신설. 각 sweep에서 TPLUS/TZERO를 새로 학습한 뒤 Student 18구성 진행 |
| 기본 Student 수 | C00–C16 17구성 + C17 1구성 = 18구성 |
| 한 full sweep | 로컬 TPLUS/TZERO 2개 + Student 18개 = 20개 학습 |
| 분석 단위 | 기본 5 full sweep. 새 seed와 reference를 결과 전에 등록 |
| 학습 길이 | Teacher/Student 모두 fresh50K. 각 run의 cosine horizon도 50K |
| 반복 방식 | BOOT5 → paired 재시험 → 제한된 공통 recipe fitting → REFRESH5 → 반복 |
| 상시 실행 | `service_mode=UNTIL_OPERATOR_STOP`, `max_campaign_cycles=null` |
| 서버 의존성 | 공유 selection lock, 공통 시작 시각, 다른 서버 결과 대기, Teacher 전달 없음 |
| 안전 | 로컬 중복 실행 방지, 디스크·수치·출처 검증, 안전 checkpoint와 중지 명령 유지 |
| Sheet | 원본 run은 각 센서 탭에 append; `ablations`는 실제 component 결과만 단계순 표시 |

**최초 5회에 대한 전체 정의는 300개다.** 기존 s1/s2의 190개를 재실행하라는 뜻이 아니다. 새로 정의하는 것은 s1/s2 C17 10개와 s3 GF2 100개, 합계 **110개**다. 기존 미완 queue는 그대로 이어간다. 실행 중이거나 미업로드인 항목의 존재는 시작 시 로컬 state로 확인한다.

## 1. 확인한 근거와 실행 상태의 경계

### 1.1 현재 코드에서 확인한 제약

조회한 저장소 기준 commit은 `6dde5ea81d4d841b406835ac8d2903e5eeec8535`다. 이는 접근 가능한 저장소의 기준이지, 각 서버의 실행 중 checkout이 이 commit이라는 증거는 아니다. [S1–S4]

현재 `ablr2/plan.py`와 `ablr2/README.md`에는 다음 제약이 남아 있다.

- lane은 s1/WV3, s2/QB만 허용한다.
- executable main case는 C00–C16으로 제한돼 있다.
- `SensorSpec.max_dn`, `Case.max_dn`, `build_config.max_pixel` 및 일부 manifest 검증에 2047이 고정돼 있다.
- 첫 BOOT5는 기존 CSV와 SHA에 고정되어 있어 파일을 덮어쓰면 source 검증이 실패한다.
- 런처는 commit에 고정된 별도 worktree로 실행된다. 원본 checkout에서 pull만 해도 실행 중 source가 자동으로 바뀌지는 않는다.
- 기본 런처는 72시간의 갱신형 lease를 사용하며 자동 연장하지 않는다.

따라서 **기존 명령에 `--server s3`만 넣거나 cycle 제한만 null로 바꾸면 되는 작업이 아니다.** C17·GF2·연속 실행·기존 상태 승계를 지원하는 명시적인 실행 확장이 필요하다. 본 bundle의 JSON/CSV는 `DESIGN_ONLY`이며 현재 runner에 바로 넣는 production config가 아니다.

### 1.2 Sheet에서 확인한 등록 범위

이번 조회에서 s1/WV3, s2/QB는 모두 P01의 TPLUS/TZERO와 C00–C16, P02의 TPLUS/TZERO·C00–C03·C12–C16까지 등록돼 있었다. C17은 조회한 목록에 없다. 단, 이번 읽기는 run ID 목록 확인이며 각 run의 raw fullstate나 GPU process 재검증은 아니다. [S6]

계획의 backfill 대상은 이 정적인 목록에 고정하지 않는다. 실행 시 `state.json`, immutable case registry, endpoint receipt, evaluation receipt를 읽고 **실제로 존재하는 모든 완료·진행 sweep**의 C17 누락분을 계산한다. 기존에 더 진행된 상태를 P02로 되돌리면 안 된다.

### 1.3 이전 table 해석의 보완

C02를 FULL과 비교하면 초기화뿐 아니라 Teacher supervision 등도 달라진다. C04를 FULL과 비교하면 uniform KD 외에도 hard/edge/q가 달라진다. 따라서 이들은 문맥을 고정한 C02↔C03, C03↔C04 등의 비교로 읽어야 한다. 기존 case가 여러 기능을 커버한다는 사실이 모든 what-if를 단일 변수로 분리했다는 뜻은 아니다.

이번 신규 C17은 그 중 **consistency-trained alignment prior만의 전이 효과**를 추가로 분리한다. FULL의 scratch-A, FULL의 uniform-only KD, e/q 교환, gradient routing 통합 같은 신규 변형은 이번 승인 범위에 자동 추가하지 않는다.

---

## 2. 신규 case C17의 정확한 정의

### 2.1 질문

> Stage 1의 shift consistency가 최종 Student에 도움이 된다면, Teacher prediction이나 e/q weighting을 사용하지 않아도 그 효과가 **Student Aligner 초기값**을 통해 나타나는가?

C11은 FULL에서 reference를 TZERO로 바꾸므로 초기 A뿐 아니라 Teacher prediction·difficulty·q가 함께 달라진다. C17은 C03과 같은 단순 Student 학습에서 **초기 A의 출처만** 바꾼다.

| 설정 | C02 | C03 | C17 — 신규 |
|---|---|---|---|
| Student 입력 | PLH | PLH | PLH |
| Student U | 같은 seed의 fresh 초기값 | 같은 초기값 | 같은 초기값 |
| Student A 시작 | Scratch | TPLUS의 정확한50K A clone | **matched TZERO의 정확한50K A clone** |
| Student A trainable | 예 | 예 | 예 |
| 기본 GT reconstruction | 사용 | 사용 | 사용 |
| Hard 추가 강조 α | 0 | 0 | 0 |
| Output KD β | 0 | 0 | 0 |
| GT edge λE | 0 | 0 | 0 |
| A의 hard 가중 | 상수0.5 | 상수0.5 | 상수0.5 |
| Teacher U inference / eT / dT / aT | 없음 | 없음 | 없음 |
| q / qref / τR / q-cache | 없음 | 없음 | 없음 |
| 직접 변경 | A 학습 시작 방식 | consistency-trained A 전이 | **unregularized A 전이** |

C17의 목적함수는 C03과 같다.

\[
\mathcal L_{\mathcal P}=\operatorname{mean}_{n,p}e_{\mathrm S}(n,p),\qquad
\mathcal L_{\mathcal A}=0.5\operatorname{mean}_{n,p}e_{\mathrm S}(n,p).
\]

`q_edge=CONST_HALF`는 edge가 꺼져 있어 사용되지 않는 설정이다. `q_aligner=CONST_HALF`는 실제 A의 hard objective에 적용한다. α=β=λE=0이어도 Student A는 계속 갱신한다. C17은 frozen-A 실험이 아니다.

### 2.2 허용되는 차이

C03↔C17에서 허용되는 차이는 `case_id`, run identity, 명칭, Teacher endpoint identity와 **clone한 A의 tensor**다. Student U 초기 tensor, optimizer 초기 상태, 네트워크 크기, data/LP, native sample/view 순서, LR schedule, batch, update 수, 평가 규칙은 동일해야 한다.

Teacher pair TPLUS/TZERO는 같은 Teacher seed의 같은 A/U 초기값과 native data/view 순서를 사용한다. TPLUS의 synthetic shift RNG는 native stream과 분리되어야 한다. TPLUS와 TZERO가 다른 학습 경로를 거치며 학습한 A의 차이가 검정 대상이다. 그 차이를 물리적 변위 GT 또는 등록 정확도라고 부르지 않는다.

### 2.3 Teacher 사용 범위

C17은 checkpoint에서 **A만 clone**한다. Teacher U 가중치를 Student U로 이식하지 않는다. 학습 중 Teacher prediction, Teacher error, q를 계산하거나 calibration을 읽지 않는다. C17 자체에는 새 calibration이 필요 없다. 다른 FULL 계열 case를 위한 TZERO calibration 생성은 별도 job이며 C17의 입력으로 연결하지 않는다.

Teacher endpoint는 validation-selected나 best-HQNR가 아니라 **정확한50,000 update의 checkpoint**다. C03은 TPLUS/현재 recipe의 양성 reference, C17은 같은 Teacher seed의 λcon=0 reference에 연결한다.

### 2.4 새 관계

- **A17: C17 → C03.** `Δ = C03 − C17`; shift-consistency-trained prior의 추가 효과. 반복 재시험 대상에 포함한다.
- **A17_SCRATCH: C02 → C17.** Reconstruction-only pretrained A와 scratch A 비교. 이미 수집한 결과로 함께 표시하되 별도의 자동 재시험을 기본으로 열지 않는다.
- **C07↔C11**은 그대로 유지한다. A17은 초기화 경로, C07↔C11은 전체 reference 경로이므로 서로 대체하지 않는다.

---

## 3. 전체 component catalog — 기존17개 + 신규1개

| 순서/분류 | Case | 한 줄 의미 | Teacher / A / fitting 핵심 |
|---|---|---|---|
| 누적0 | C00 | PAN+MS 기본 복원 | L/H zero slots, A identity, Teacher 없음, plain GT |
| 누적1 | C01 | LPAN/HPAN 입력 추가 | PLH, A identity, Teacher 없음 |
| 누적2 | C02 | Scratch PAN Aligner 추가 | PLH, fresh trainable A, Teacher 없음 |
| 누적3 | C03 | Consistency-trained Aligner로 초기화 | TPLUS A clone만 사용; plain GT, A trainable |
| **누적3의 what-if** | **C17** | **Consistency 없이 학습한 Aligner로 초기화** | **C03과 같고 TZERO A clone만 사용** |
| 누적4 | C04 | Uniform Teacher output distillation | C03 + uniform KD; hard plain, edge 없음 |
| 누적5 | C05 | e-guided adaptive reconstruction fitting | Hard 추가 강조 + selective soft; edge 없음 |
| 누적6 | C06 | GT edge supervision 추가 | e-guided fitting + constant0.5 edge/A weighting |
| 누적7 | C07 | FULL PANDA | e-guided hard/soft + q-edge + q-hard A |
| FULL 제거 | C08 | LPAN/HPAN 입력 제거 | Teacher·loss 그대로, Student L/H slots zero |
| FULL 제거 | C09 | Student alignment 제거 | Student warp bypass, frozen Teacher reference 유지 |
| FULL 제거 | C10 | Teacher 전체 제거 | A fresh, prediction/d/q/clone 없음, GT edge는 유지 |
| FULL 제거 | C11 | Teacher shift consistency 제거 | 새 matched TZERO reference, FULL fitting 유지 |
| FULL 제거 | C12 | Hard 추가 강조 제거 | α=0, selective soft에서 dT/aT는 유지 |
| FULL 제거 | C13 | Output distillation 제거 | β=0, Teacher A/e/q 경로는 유지 |
| FULL 제거 | C14 | GT edge 제거 | λE=0, q-hard A는 유지 |
| FULL 제거 | C15 | Sample별 q weighting 제거 | q-derived s를 해당 reference의 전체 train-view 평균으로 치환 |
| FULL 제거 | C16 | Student A adaptation 제거 | Teacher clone A를 고정; correction은 계속 적용 |

C17은 **case 번호순 C16 다음에만 실행해야 하는 것이 아니라 C03에 연결되는 세부 대조**다. 논문의 표시 순서는 C02 → C17 → C03으로 제시할 수 있고, 실제 실행의 기존 C00–C16 순서는 바꾸지 않는다.

Teacher 전체를 제거하는 C10과 output KD만 제거하는 C13, Student A를 없애는 C09와 고정하는 C16을 혼동하지 않는다.

---

## 4. s1·s2: 기존 cycle을 중단·초기화하지 않는 추가 방식

### 4.1 논리적 연속성과 실행 source 교체를 구분한다

사용자의 “계속 돌린다”는 것은 기존 sweep·seed·결과·누적 비용을 이어간다는 의미다. 고정 source로 실행 중인 Python process에 파일을 덮어씌우라는 뜻이 아니다.

1. 실제 local runtime path, source commit, current job, fullstate, immutable case manifest, seed ledger를 읽고 backup한다.
2. C00–C16 원 정의와 원 CSV는 변경하지 않는다. 별도 extension registry에 C17만 추가한다.
3. source 변경이 필요하면 **현재 run의 안전한 종료 지점**에서 runner 소유권을 인계한다. 실행 중 optimizer를 임의로 kill하거나 같은 job을 두 runner에서 재개하지 않는다.
4. 완료·running·pending job identity와 seed ledger는 승계한다. `r000/P02`를 `r000/P01`로 되돌리거나 사용 시간을 0으로 만들지 않는다.
5. 이미 진행된 sweep의 C17 누락분을 추가하고 기존 pending 순서로 돌아간다.

### 4.2 C17 backfill 규칙

| 발견 상태 | 처리 |
|---|---|
| 해당 sweep의 C03와 TZERO endpoint가 모두 있음 | 같은 Student seed·init group·recipe로 C17만 추가 |
| 현재 C03가 학습/평가 중 | 그 job을 완료한 뒤 C17 추가 |
| 아직 C03 이전 | 기존 순서를 유지하고 C03 바로 뒤에 C17 삽입 |
| 동일 C17 run이 이미 정의/진행/완료 | 중복 추가하지 않음 |
| C03에 필요한 초기값·stream 재현 증거가 없음 | `PAIRING_NOT_VERIFIED`; 신규 C03 anchor를 동일 새 runtime에서 C17과 함께 실행하고 원 C03은 보존 |
| TZERO exact50K가 없지만 같은 run의 fullstate가 있음 | 원 source/config/seed로 그 Teacher를 정상 재개. 다른 seed Teacher로 대체 금지 |
| TZERO 자산이 소실·불일치 | `REFERENCE_UNAVAILABLE`; 원 sweep을 성공 처리하지 않고 다른 준비된 일을 진행. 복구 비교는 별도 repair identity |

이미 P01/P02가 끝났거나 C03를 지나갔다면, 다음 안전한 job 경계에서 **가장 오래된 C17 debt부터** 보충한다. 기본적으로 한 경계에서 backfill 최대2개를 처리한 뒤 원 pending 업무를 진행해, 과거 debt가 현재 full sweep을 계속 굶기지 않게 한다. 최초5회에 대해서는 센서당 C17 총5개다.

### 4.3 완료 flag와 통계 보존

`legacy_core17_complete`와 `extended18_complete`를 분리한다. P01의 원17개가 이미 끝났다면 그 사실은 유지하며, C17 미실행 때문에 원 완료 기록을 지우지 않는다. 대신 새18개 표에는 `C17 pending`으로 표시한다.

이미 임계값이 고정됐으면 그대로 승계한다. 아직 고정 전이라도 **기존17개 관계 graph에서 계산하는 sensor-global threshold 공식**은 변경하지 않는다. 추가 A17에는 같은 센서 기준을 적용하되, A17 자체의 최초5개 pair가 모이기 전에는 재현성 판정을 완료하지 않는다.

### 4.4 이전 C03를 대조군으로 재사용하는 조건

같은 seed 숫자만으로는 충분하지 않다. 다음을 만족한 경우에만 원 C03 결과를 새 C17의 anchor로 재사용한다.

- Model/loss/warp/optimizer/data-view RNG의 수치 경로와 resolved numerical configuration이 일치한다.
- 공통 fresh U hash가 재현되고 data/view sequence hash가 일치한다.
- Teacher pair의 공통 초기 hash와 native stream 계약이 검증된다.
- 평가 preset, validation 분할, 후보 grid, 선택 규칙이 같다.
- control-source 확장에 따른 numerical parity receipt를 보존한다. source commit 문자열이 다르다는 이유만으로 같은 수치 경로라고 가정하지 않는다.

검증이 성립하지 않으면 **C03도 새로 짝지어 실행**한다. 이 추가 run은 repair용 동일 조건 anchor이지 새로운 component가 아니다. CSV의 110개는 이런 조건부 repair 비용을 포함하지 않은 정의량이다.

---

## 5. s3/GF2 신규 cycle

### 5.1 s3 기존 작업과의 전환

s3가 아직 P40나 다른 생산 작업을 수행 중이면 새 ablation을 같은 GPU에 중복 실행하지 않는다. 사용자의 이번 재배치를 기준으로 **기존 s3 campaign은 현재 run의 종료 후 신규 admission을 중지**하고, checkpoint·평가 backlog·state를 보존한다. 이미 평가된 결과의 업로드 retry는 로컬 I/O 범위에서 가능하다.

운영상 즉시 전환이 필요하면 `STOP_NOW_SAFE`에 해당하는 정식 안전 정지를 사용해 fullstate를 남긴다. 무단 강제 종료는 하지 않는다. 기존 P40의 완료하지 못한 pair를 완료로 표시하지 않으며 그 결과를 ABLR2X에 가져오지도 않는다.

s4/s5의 clock·queue·분포·confirmation에는 손대지 않는다. s3 ablation은 두 서버의 진행과 무관하게 시작·반복한다.

### 5.2 처음부터 matched Teacher pair를 학습한다

각 sweep은 **TPLUS50K / TZERO50K → 로컬 calibration → Student18개**다.

- GF2의 기존 TA, R3_100, PANMIX 부모 또는 최고-HQNR 모델을 이 cycle의 Teacher로 선택하지 않는다.
- TPLUS/TZERO는 같은 Teacher seed의 fresh A/U와 동일 native stream이다.
- TPLUS의 기본 λcon=1e−4, TZERO의 λcon=0이다.
- 두 Teacher의 exact50K endpoint에서 각기 τR, qref, 전체 q-cache, mean-s를 만든다.
- C00/C01/C02/C10은 Teacher I/O 자체가 없는 case이다. 운영상 Teacher-first queue를 사용할 뿐 숨은 supervision을 연결하지 않는다.
- C03/C04/C17은 endpoint만 필요하다. calibration 값을 학습에 소비하지 않는다.
- Teacher 자신의 HQNR가 낮아도 정상적으로 완료된 run이면 계획된 Student를 진행한다.

### 5.3 GF2 최초5회 seed와 규모

| Sweep | Teacher seed — TPLUS/TZERO 공통 | Student seed — C00–C17 공통 | 학습 수 |
|---|---:|---:|---:|
| P01 | 981001 | 991001 | 2T + 18S = 20 |
| P02 | 981002 | 991002 | 20 |
| P03 | 981003 | 991003 | 20 |
| P04 | 981004 | 991004 | 20 |
| P05 | 981005 | 991005 | 20 |
| **합계** | **10 Teacher run** | **90 Student run** | **100개** |

이는 새로운 사전 정의 seed다. 이전 GF2 생산 seed에서 골라낸 것이 아니다. 실행 전 local/global-used registry와 충돌을 검사하며, 충돌이 있으면 결과를 보기 전에 deterministic collision resolution과 새 manifest를 기록한다.

GF2 새 run ID 예시:

```text
ABLR2_GF2_s3_r000_P01_TPLUS_TS981001_FRESH50
ABLR2_GF2_s3_r000_P01_TZERO_TS981001_FRESH50
ABLR2_GF2_s3_r000_P01_C03_SS991001_FRESH50
ABLR2_GF2_s3_r000_P01_C17_SS991001_FRESH50
```

경로는 별도의 `work_dir/ablr2/GF2/s3/` 하위에 둔다. G20/L100/B20/P40/reproduction 경로를 재사용하지 않는다.

### 5.4 PANMIX와 100K를 기본에서 제외하는 이유

이번 목표는 **WV3/QB와 같은 C00–C17의 method component study**다. GF2만 PANMIX를 켜면 구성 차이와 증강 차이가 섞인다. GF2만 Student를 기존 생산100K로 바꾸면 원50K ablation과 다른 학습 recipe가 된다.

따라서 최초 cycle은 native PAN·50K로 맞춘다. GF2의 고성능 recipe에서 component 기여를 다시 볼 필요가 생기면, 별도 승인 revision에서 **모든 관련 row에 동일한100K 또는 증강 조건**을 적용한다. 새50K ablation FULL이 기존PANMIX120K보다 낮다는 이유만으로 component 실험 실패로 판정하지 않는다.

---

## 6. 센서별 공통 구조·수치 설정

| 항목 | s1/WV3 | s2/QB | s3/GF2 |
|---|---|---|---|
| Teacher | P0 / W112 / D[1,2,3] | 동일 | 동일 |
| Student | W104 / D[1,2,2] | 동일 | 동일 |
| MS/output bands | 8 | 4 | 4 |
| Teacher input channels | 9 | 5 | 5 |
| Student stored slots | 11 | 7 | 7 |
| P0 Student case | L/H slots를 학습·추론에서 모두0 | 동일 | 동일 |
| PAN Aligner | 동일 global-translation 구조 | 동일 | 동일 |
| Task / Attention / mode modulation | HRMS-only / OFF / OFF | 동일 | 동일 |
| maxDN | 2047 | 2047 | **1023** |
| Normalize | 2DN/2047−1 | 동일 | **2DN/1023−1** |
| Inverse scale factor | 1023.5 | 1023.5 | **511.5** |
| RR 다중밴드 Q | Q8 | Q4 | Q4 |
| nominal train count | 9,714 | 17,139 | 19,809 |
| nominal val count | 1,080 | 현 manifest 확인 | 2,201, 현 manifest 재검증 |
| Teacher/Student updates | 50K/50K | 동일 | 동일 |
| Batch | 48 | 48 | 48 |
| U peak LR | 1e−4 | 동일 | 동일 |
| A peak LR — Teacher/Student | 1e−5 / 3e−6 | 동일 | 동일 |
| Optimizer | AdamW, β=(.9,.999), eps=1e−8, wd=.01 | 동일 | 동일 |
| Scheduler | warmup100 + cosine50K | 동일 | 동일 |
| Precision | 기존 pinned FP32/runtime flags | 같은 원칙 | 같은 원칙 |
| 기본 FULL α / β / λE / λcon | 1 / .1 / .002 / 1e−4 | 동일 | 동일 |
| PANMIX | OFF | OFF | OFF |

s1/s2의 현재 cycle이 이미 승인된 R01–R05 recipe에 진입했다면 이 표를 이유로 R00로 되돌리지 않는다. 기존 recipe를 계승하고 C17의 제거 override만 적용한다. s3의 BOOT5는 R00로 시작한다.

### 6.1 GF2용 실제 이식 점검

**4-band라는 이유로 QB 설정을 복사하면 안 된다.** 다음은 현재 code에서 수정/검증해야 하는 항목이다. [S2–S5]

1. `SensorSpec`의 maxDN·train/val count·Q key·MTF sensor·band order를 센서에 따라 구성한다.
2. `Case.max_dn`, `build_config.max_pixel`, dataset-manifest validators의2047 고정값을 센서값으로 바꾼다.
3. canonical GF2 train/val/RR/FR 및 LP의 실제 파일 SHA·tensor shape·DN 범위를 확인한다. QB msfix를 GF2에 적용하지 않는다.
4. RR inverse normalization, ERGAS/SAM/PSNR/SSIM/SCC/Q4 계산, FR spectral MTF·PAN down/up 규약은 기존 검증된 GF2 경로와 parity 확인한다.
5. GF2의 FR spectral MTF는 기존 GF2 evaluator의 설정을 그대로 사용한다. QB preset을 fallback으로 쓰지 않는다.
6. train PAN64, native MS16, RR PAN256, FR PAN512, RR/FR20장, 기존 RR support 및 FR 전체 frame을 유지한다.
7. 경로명 토큰에 의존해 maxDN를 추정하지 말고 bound sensor manifest의 값으로 고정한다.

GF2의 파일 경로·band order·SHA를 본 bundle에서 추측해 채우지 않는다. 실제 s3의 검증된 source manifest에서 바인딩한다. G20의 데이터 pin은 비교 근거이며 오래된 파일 이름만 보고 승인하지 않는다.

### 6.2 LP·calibration·gradient 계약

- LP는 native PAN에서 기존 Gaussian σ1.98/k41/replicate/[2::4,2::4]로 생성한다. 생성float64/cachefloat32 규약과 canonical tensor hash를 보존한다.
- Aligner는 PAN/MS만 본다. PAN과 LP는 같은 correction/grid로 warp하고 HP=aligned PAN−aligned LP로 만든다. MS/GT/output은 이동하지 않는다.
- Teacher native reconstruction은 A/U, shift consistency는 A에 직접 gradient를 준다. shifted PAN은 Teacher U로 보내지 않는다.
- calibration은 고정 train3072 base IDs(seed1234), q probe AXIS16, fixed4 geometry views, 전체 train×4 q-cache를 사용한다. qref는 calibration subset×4 median이며 문서의 전체-train 약식표현과 구분한다.
- τR는 해당 reference에서 측정한다. C15의 mean-s는 전체 train×4 view에서 계산한다. 센서나 Teacher가 바뀌면 scale/cache를 다시 만든다.
- FULL U는 hard+soft+q-edge, FULL A는 q-hard만 받는다. Teacher는 frozen이다.
- dT/aT 재가중치의 stop-gradient, signed Scharr 방향 평균0.5, 기존 valid edge support를 유지한다.
- no-A case는 warp를 bypass한다. frozen-A case는 weight decay·buffer update도 금지한다.

---

## 7. 실행 순서와 상시 반복 controller

### 7.1 Full sweep 순서

원17개 case의 순환·역순 규칙을 보존한다. `offset=4*(p−1) mod17`로 회전한 뒤 짝수 sweep에서는 뒤집는다. 그 결과에서 **C03 바로 뒤에 C17만 삽입**한다. 기존17개 사이의 상대 순서는 변하지 않는다.

s3/P01 예시:

```text
TPLUS → TZERO → 각 reference calibration
C00 → C01 → C02 → C03 → C17 → C04 → C05 → C06 → C07
→ C08 → C09 → C10 → C11 → C12 → C13 → C14 → C15 → C16
```

실험 순서와 논문 표시 순서는 별개다. 논문 표는 기능상 C02/C17/C03를 묶을 수 있다. 실행 순서를 점수에 맞춰 바꾸지 않는다.

### 7.2 State machine — 실행 명세, production CLI가 아님

```text
LOCAL PREFLIGHT / STATE MIGRATION
    ├─ s1/s2: legacy state 이어받기 + C17 debt 등록
    └─ s3: 기존 GPU job 안전 인계 + GF2 BOOT5 신규 등록
             ↓
FIRST COMPLETE 5-SWEEP COVERAGE (18 Student cases)
             ↓
SENSOR-LOCAL ANALYSIS (legacy thresholds 유지/최초 고정)
             ↓
관계별 불안정/역전 검토
    ├─ 허용된 RECHECK5 debt → 양쪽 case + FULL을5개 seed에서 실행
    ├─ 개선 가설 → 기존 bank 중 최대2개 recipe fitting
    └─ 새 후보 없음 → 현 recipe 유지
             ↓
FULL REFRESH5 (2 Teacher + 18 Student) ×5
             ↓
분석·결과 저장 → 다음 cycle
```

각 서버는 자기 progress만 사용한다. WV3의 P05가 끝나기 전에 GF2가 끝나거나 반대여도 대기하지 않는다. Google Sheet 업로드 실패도 다른 서버나 로컬 학습의 진행 barrier로 사용하지 않는다.

### 7.3 재시험은 유한 묶음, 바깥 반복은 무제한

기존 관계에 A17을 추가한다. `relation × exact numerical recipe`별 집중 재시험은 **추가5pair 한 묶음**이다. score가 좋은 첫 seed에서 중단하지 않는다. A17 재시험에는 기본적으로 C17·C03·C07을 같은 새 Student seed로 실행한다. 이때 C17은 TZERO, C03/C07은 matched TPLUS를 써야 한다.

동일 seed의 C03와 C17은 서로 다른 endpoint A를 사용하는 것이 의도된 조건 차이이다. Teacher hash가 다르다는 이유로 pair를 무효화하지 말고 **같은 Teacher seed의 정식 matched ±consistency pair인지** 확인한다.

집중 재시험을 소진하면 같은 행만 계속 재추첨하지 않고 공통 fitting 또는 REFRESH5로 이동한다. 이후의 full refresh에는 C17도 계속 들어간다. 과거 좋았던 행도 함께 재측정한다.

### 7.4 허용한 공통 recipe bank

| Recipe | R00 대비 한 축 변경 |
|---|---|
| R00 | 기존 BASE |
| R01 | β=.2 |
| R02 | λE=.001 |
| R03 | α=.1 |
| R04 | TPLUS의 λcon=3e−4; matched 새 Teacher/정식 reference 생성 |
| R05 | Student A LR:24,240 update 이후 기존의1/3 |

새 GF2 lane도 이 bank로 시작한다. PANMIX/G050/K1/E100/100K를 자동으로 수입하지 않는다. 후보는 해당 센서의 local two-block DEV 비교로 선택한다. FULL 자체의 품질이 개선되는지가 기준이며 ablated 모델만 악화시켜 간격을 넓힌 후보는 채택 이유가 아니다.

**C17에도 해당 recipe의 공통 U/A schedule은 적용하지만 α=β=λE=0은 유지한다.** R04에서 C03는 양성 reference TC3, C17은 여전히 λcon=0의 matched TZERO다. 양성 reference 바꾸기와 C17 Student objective 바꾸기를 혼동하지 않는다.

기존 numerical-equivalence/reuse 규칙은 유지한다. β만 바꾸는 recipe에서 β=0인 C17을 같은 seed/source로 다시 등록한 것이 기존 C17과 수치적으로 같으면 결과 재사용 receipt를 남길 수 있다. 이를 독립 반복으로 중복 집계하지 않는다. REFRESH/VERIFY의 새로운 seed는 실제로 새로 실행한다.

### 7.5 연속 실행 모드와 기존72h lease의 차이

`max_campaign_cycles=null`만으로는 실제 연속 실행이 아니다. 현재 launcher의72h lease는 만료되면 멈춘다. 이번 사용자 요청에 맞춘 실행 확장은 **명시적인 `UNTIL_OPERATOR_STOP` 승인 기록**을 별도로 만들고, 새 job admission을 finite wall-clock lease에 묶지 않는 모드를 지원해야 한다.

이는 오래된 lease를 몰래 연장하는 구현이 아니다. `old authorization → new until-stop authorization` 전환 시각·범위·source를 보존한다. 기존 이미 승인된 run의 재개 계약을 임의로 바꾸지 않는다. 새 모드로 전환되기 전에는 기존72h 런처가 무한 실행된다고 안내하지 않는다.

다음은 계속 유지한다.

- `STOP_AFTER_RUN`, `STOP_AFTER_SWEEP`, `PAUSE_AFTER_BLOCK`, `STOP_NOW_SAFE`, `CONTINUE`.
- 로컬 GPU 소유권·PID/duplicate guard. 다른 서버의 공유 lock은 없음.
- transient I/O 재시도 최대2회, 그 뒤 보존·로컬 failure 기록.
- NaN/발산은 결과로 보존하고 같은 구성의 새 seed로 그 실패를 가리지 않음.
- source/data/reference 불일치, disk 부족, GPU 치명 오류는 안전 정지. 사용자 수동 중지 뒤 자동 재시작 금지.

단기 작업 묶음은 유한하므로 종료·복구 지점이 계속 생긴다. 무한 반복을 하나의 끝나지 않는 training run으로 구현하지 않는다.

### 7.6 자원·plateau 관리

첫5회의 총종료 시각을 가정하지 않는다. s3 하나의 full5 wave는100개 학습,50K 기준 총5,000,000 optimizer updates이며 평가/calibration 비용이 더 든다. GPU 속도는 로컬 실측값으로 예약·보고한다.

디스크는 각 신규 block 이전에 검사한다. 기존에 더 엄격한 정책이 없으면 운영 시작값으로 `free space >= max(100GiB, 2×next-block projected write)`를 적용하고, 실제 다음 묶음의 후보 checkpoint·fullstate·결과 파일 보존 비용을 예측한다. 이 수치는 운영 안전값이지 최적 실험값이 아니다. 장기 무제한 실행은 저장공간이 무한하다는 뜻이 아니므로 부족하면 안전 pause하며 보호 자산을 자동 삭제하지 않는다.

같은 recipe가 plateau여도 사용자 중지 전까지 balanced REFRESH5는 계속 가능하다. 3cycle 연속 새로운 절대 개선/관계 안정화가 없으면 `LOW_INFORMATION_GAIN`을 기록한다. 점수에 맞게 새 값을 임의 발명하거나 결과를 바꾸지 않는다.

---

## 8. 평가·통계·결과표 계약

### 8.1 Checkpoint 선택

원 ABLR2와 동일하게 주 선택은 `RR_VALIDATION_ARGMIN_ERGAS_THEN_LOWER_STEP`다. candidate grid는1010,2020,…,49490,50000의50개다. 보조 주 분석은 Exact50K, RAW-H 최고와 test-E 최소는 탐색 보조다.

모든 RR/FR 지표는 해당 행의 같은 checkpoint에서 가져온다. Teacher가 전달하는 endpoint는 별개로 exact50K다. C17의 결과 선택을 C03와 다른 기준으로 바꾸지 않는다.

GF2의 과거 HQNR>.964 / ERGAS<.552는 별도 참고선으로 남길 수 있지만, ablation row의 유효성이나 첫5회 admission을 판정하는 조건이 아니다. 이 component campaign은 공정한 내부 비교가 목적이며 생산 최고값 경쟁과 구분한다.

### 8.2 반복과 불확실성

`ΔH`, `ΔDs`, `ΔDλ`, `relative ΔERGAS`를 같은 local Teacher/Student-seed pair에서 계산한다. sweep을 구성하지 않는 targeted RECHECK 결과를 full-matrix 평균에 섞지 않는다. recipe가 다르면 평균을 합치지 않는다.

최초5회 sensor threshold는 기존 legacy graph17개에서 계산하고 고정한다. 원 설계의 MAD 기반 운영 공식과 4/5 support 기준을 계승한다. A17은 같은 sensor threshold를 사용하지만 기전의 정합 정확도를 증명했다고 쓰지 않는다. 여전히 FR20을 반복 개발에 쓰므로 `TEST_AWARE_DEV`이며 새 training seed가 독립 test set을 만드는 것은 아니다.

FULL이 모든 행을 반드시 이기도록 결과를 교체하지 않는다. 반복은 개선 가능한 공통 recipe를 탐색하는 과정이며, 역전·무차이·trade-off도 모두 보존한다.

### 8.3 논문 table과 case 연결

| Table 범주 | 이번 활용 |
|---|---|
| Main component flow | 기존 C00–C07 등 큰 기능군의 문맥별 비교 |
| Alignment/reference what-if | **C02, C17, C03** 초기화 비교 + C07/C09/C11/C16 |
| Reliability fitting what-if | C03/C04/C05/C06/C07의 단계 비교와 C12–C15 제거군 |

C17↔C03가 생겼다고 `C02↔FULL`을 pure scratch-init ablation이라고 적어서는 안 된다. 마찬가지로 C04는 FULL에서 KD 선택성만 제거한 모델이 아니다. 본 table은 실제 실행 정의로 명명한다.

### 8.4 Sheet 반영 규칙

- 원본 결과는 `WV3-s1`, `QB-s2`, `GF2-s3(5090)`의 ABLR2 namespace로 append한다. 별도 raw 탭을 쓰려면 같은 단일 writer 계약으로 명시한다.
- `ablations`에는 **실제 완료된 GF2 C00–C17만** 새 GF2 component 영역으로 넣는다. 이전 B20/P40를 다시 ablation으로 가져오지 않는다.
- 표시 순서는 sweep별 Teacher pair → C00/C01/C02/C03/C17/C04…C16 또는 논문별 명시된 그룹이다. 실험을 이어 학습했다는 오해를 피하도록 각 Student는 fresh run임을 note로 남긴다.
- methods 문구는 줄바꿈·표시seed 없이 한 줄이다. 예: `P02 | C17 | Aligner initialized from a Teacher trained without shift consistency`.
- 정확한 run/seed/source/Teacher/checkpoint/selection은 note 또는 원본metadata로 보존한다.
- 수치·비용은 해당 run의 실측만 사용한다. 미완 case에는0이나 다른 production 결과를 채우지 않는다.
- 공유 Sheet는 보고용이지 서버 간 제어용이 아니다. 네트워크 장애 시 로컬 append-only outbox에 저장하고 업로드만 재시도한다.
- 레이아웃이 달라진 summary 탭의 절대행번호를 runner가 추측해 덮어쓰지 않는다. 현재 구조를 읽는 idempotent summary writer로 실제 case를 배치한다.

---

## 9. 구현 변경 목록과 합격 조건

### 9.1 최소 변경 모듈

| 영역 | 필요한 변경 |
|---|---|
| `ablr2/plan.py` | GF2/s3 lane·sensor parameterization·C17 override·extension registry·new graph·full18 wave |
| BOOT5 source validation | 원190행과 SHA 유지. C17/GF2를 별도 immutable overlay로 승인 |
| `student_order` | 원17개 상대순서 보존, C03 뒤C17 삽입 |
| `full_wave/recheck_cases/screen_cases` | C17에 TZERO dependency, GF2 seed block,18-case refresh,구성별 reuse |
| `SensorSpec`, `build_config`, data manifest | maxDN1023·counts·Q4·GF2 source binding,2047 상수 의존 제거 |
| evaluation/model adapters | GF2 native pipeline parity, C4/C8 FULL 기존 path parity |
| reference loader | C17은 A clone only, U prediction/q/calibration I/O 금지 |
| runtime launcher/controller | until-stop 명시 승인, 기존 state 승계, safe release handover |
| `upload/plots/analysis` | C17/18-row coverage/GF2 지원, 기존17-row 분석과 version 구분 |
| pinned runtime release | 실행 중인 worktree를 수정하지 않고 새 release로 safety boundary handover |

source 검증을 통과시키려고 원 SHA 목록을 무작정 새 해시로 덮어쓰지 않는다. 확장 전·후 계약과 provenance를 구분한다.

### 9.2 실행 전 확인해야 할 테스트

1. legacy190개 row/ID/seed/기존 case 수치 설정 불변.
2. C17=C03에서 Teacher source만 TZERO인 정의 검증; α=β=λE=0, q/τ access 없음.
3. C03/C17의 U 초기 hash·native stream 일치와 Teacher pair 초기 hash 일치.
4. C17 A clone source는 TZERO exact50K이고 Student A는 trainable; Teacher U/A는 Stage2에서 불변.
5. C00/C01/C02/C10에 숨은 Teacher binding 없음.
6. C09 warp bypass, C16 gradient/optimizer decay/buffer freeze 검증.
7. s3/GF2만 추가로 허용; s4/s5 ablation admission은 거부.
8. GF2 DN1023 round trip,4band/Q4/sensor MTF/data shapes; WV3/QB parity 회귀.
9. legacy source/SHA 보존, C17 overlay idempotent, 이미 완료 run 재실행 금지.
10. running-state import, SIGTERM-safe save, runner중복 방지, seed/cycle/비용 counter 이어쓰기.
11. A17의 재시험5개와 full18 refresh 생성. Teacher단독 등록을 C17 완료로 세지 않음.
12. finite72h와 until-stop 모드 명시 구분, STOP_NOW_SAFE/STOP_AFTER_RUN/CONTINUE 기능.
13. upload 실패에 의한 training 재실행 없음; summary label 개행·표시seed 없음.
14. C03 legacy numerical reuse가 입증되지 않으면 새paired anchor로 fallback.
15. local disk/resource safety, incomplete pair·failedrun 보존.

bundle의 `build_registry.py`/`test_registry.py`는 **설계 manifest의 정적 무결성만 검사**한다. GPU 학습·실제 source parity·s3 데이터 접근·런처 migration이 통과했다는 뜻이 아니다.

---

## 10. 제공 파일과 실행 인계

| 파일 | 용도 |
|---|---|
| 본 MD | 전체 실험·운영·migration 계약 |
| `registries/ABLR2X_S1S2_Append_C17_10.csv` | 기존 최초5회에 추가할 C17만10개 |
| `registries/ABLR2X_GF2_S3_BOOT5_100.csv` | 신규 GF2 최초5회100개 |
| `registries/ABLR2X_NewDefinitions_110.csv` | 위 두 목록의 합집합; 새 정의만 |
| `registries/ABLR2X_First5_ReferenceUniverse_300.csv` | 기존190개+추가110개의 전체 설명용목록; 통째 재기동 금지 |
| `registries/ABLR2X_ComponentCatalog_18.csv` | C00–C17 flag별 정의 |
| `registries/ABLR2X_ComparisonGraph_19.csv` | legacy17관계 + 신규primary1 + 진단1 |
| `registries/ABLR2X_DesignRegistry.json` | 신규lane·C17·continuous mode·고정policy 설계 |
| `legacy/` | 원 CSV/catalog/graph/registry 불변 사본 |
| `build_registry.py` | 위 manifest 재생성. 실제서버·GoogleSheet·GPU를 호출하지 않음 |
| `test_registry.py` | 정적 계약 unit test |
| `registries/validation.json` | 생성목록 수·ID·계약검증 결과 |

정적 생성/검증 명령은 아래와 같다. **학습 기동 명령이 아니다.**

```bash
python build_registry.py --out registries
python -m unittest test_registry.py -v
```

실제 runner 확장·검증 후 s1/s2는 local state로부터 extension admission을 수행하고, s3는 안전전환 이후 신규GF2 lane을 시작한다. 기존 `ablr2_start.sh --server s3`는 이식 전에는 거부되므로 성공하는 명령처럼 복사해 배포하지 않는다.

### 인계용 작업 지시

> 기존 s1/WV3·s2/QB ABLR2의 완료/진행/pending state, seed ledger, recipe, 시간·비용, frozen runtime provenance를 보존한다. C17(C03-TZERO)을 추가하여 완료 sweep에는 backfill, 다음 sweep에는 C03 뒤에 삽입한다. s3에는 같은 C00–C17을 native GF2/DN1023/fresh50K로 반복하는 독립 lane을 만든다. s3의 기존 작업은 안전run경계에서 인계하고 s4/s5는 변경하지 않는다. 각full sweep은 matched TPLUS/TZERO2개와 Student18개다. 분석은5sweep단위, outercycle은 until-operator-stop이며 sharedlock·공통clock·다른서버대기는 없다. 기존72h lease 만료를 그대로 두고 무한실행이라고 부르지 말고, 명시적인continuousauthorization을 구현한다. 기존case수치parity, C03/C17 pairedinit/stream, GF2evaluation, state migration을 검증한 후 실제커밋·배포·PID/runID·localstate로 실행여부를 보고한다. 좋은결과만남기거나 rawtest최고와validation선택을섞지않는다.

---

## 11. 근거 및 변경 우선순위

이번 사용자 지시(서버3개, C17추가, until-stop)는 이전문서의 s3보호·s1s2만허용·72h재승인 운영보다 우선하는 **새 설계 범위**다. 실제 구현에는 아직 이전제약이 있으므로 변경목록을 별도로 명시했다.

- **[S1] 기존 설계** — `legacy/ABLR2_DesignRegistry_2026-09-21.json`, 원 `PANDA_ABL_S1_WV3_S2_QB_AdaptiveRepeat_ExperimentPlan_2026-09-21_v2.md`. 원50K·5sweep·recipebank·seed·comparison 규약.
- **[S2] 조회한 실행 정의** — https://github.com/hojunking/PAN-Crafter-repro/blob/6dde5ea81d4d841b406835ac8d2903e5eeec8535/ablr2/plan.py . lane/C00–C16/2047고정값, sourcepin, seed와dynamicwave검증.
- **[S3] 조회한 실행 README** — https://github.com/hojunking/PAN-Crafter-repro/blob/6dde5ea81d4d841b406835ac8d2903e5eeec8535/ablr2/README.md . pinned worktree·72h lease·safe stop·독립lane·noTeacher I/O.
- **[S4] 조회한 data 검증** — https://github.com/hojunking/PAN-Crafter-repro/blob/6dde5ea81d4d841b406835ac8d2903e5eeec8535/ablr2/data.py . source bindings·QBmsfix·maxpixel검증·LPcache·native4view.
- **[S5] 검증된 GF2경로 비교 기준** — https://github.com/hojunking/PAN-Crafter-repro/blob/6dde5ea81d4d841b406835ac8d2903e5eeec8535/g20/plan.py . GF2SensorSpec재사용 및 sensor/data 경로. G20 학습run을 새ablation으로재사용한다는 의미는아님.
- **[S6] 이번 live run ID 조회** — https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit ; `WV3-s1!B106:C170`, `QB-s2!B14:C90`. 로컬실행인증이아닌등록목록점검.
- **[S7] 사용자 최신 Method 원문** — `붙여넣은 마크다운(1).md`: Align-First PAN Alignment Learning → Reliability-Decoupled Adaptive Fitting; consistency의relative-response범위, frozen reference/StudentAinit, PLH와weighting독립, e/q서로다른gradient경로.

**기록할 제한:** 본 계획은 아키텍처·component 정의·반복 방식의 설계다. s1/s2가 실제로 새case를 실행하기 시작했거나 s3가 ablation으로 전환됐다는 확인은 별도의배포/기동후보고가필요하다.
