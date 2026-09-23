# PANDA Component Ablation — s1 WV3 / s2 QB 상시 반복 실험 계획

**작성일:** 2026-09-21  
**설계 ID:** `PANDA_ABL_S1WV3_S2QB_ADAPTIVE_LOOP_20260921_v2`  
**이전 계획:** `PANDA_QB_ABL_S12_20260921_v1`의 서버·센서 배치와 반복 정책을 대체한다. C00–C16의 component 정의는 유지한다.  
**사용 서버:** s1=WV3, s2=QB. s3–s5의 GF2 실험에는 접근하거나 변경하지 않는다.  
**산출물 상태:** MD·설계 registry·첫 5회 case 목록 작성. 학습 기동, 원격 코드 수정, queue 변경, Sheet 쓰기는 수행하지 않았다.

> **원하는 운영:** 각 센서에서 전체 component 표를 먼저 5회 반복한다. 이후 예상과 다른 관계를 자동으로 찾아 양쪽 조건을 함께 재시험하고, 필요하면 공통 학습 recipe를 개선한다. 이 바깥 반복은 사용자 중지 전까지 계속할 수 있다.
>
> **결과의 의미:** “상위 구성이 한 번 이기는 값이 나올 때까지 그 행만 교체”하는 방식은 사용하지 않는다. 모든 시도와 역전은 보존하고, 같은 seed·reference 묶음의 결과로 흐름을 판단한다. 계단형 모양은 검정할 가설이지, 결과를 덮어쓰는 종료 조건이 아니다. 상시 반복을 하더라도 끝내 개선되지 않는 관계는 그대로 남을 수 있다.

---

## 0. 바로 적용할 변경과 시작 단위

| 항목 | 이번 결정 |
|---|---|
| 서버 배치 | **s1 WV3 / s2 QB**. 각 서버가 자기 센서의 Teacher부터 학습한다. |
| 센서 간 독립성 | seed·Teacher·calibration·기준값·반복 상태·중지 명령 모두 센서별이다. 다른 서버 결과를 기다리지 않는다. |
| 기본 구조 | Teacher P0/W112D123, Student PLH/W104D122, 기존 PAN Aligner. 단일 HRMS task 유지. |
| 기본 길이 | Teacher·Student 모두 fresh **50,000 updates**. GF2의 100K를 자동 이식하지 않는다. |
| 핵심 구성 | 누적 C00–C07 + FULL 제거 C08–C16 = **17개 Student 구성**. |
| “1회”의 정의 | 같은 센서에서 **matched TPLUS/TZERO 2개 + C00–C16 전체 17개**를 완결한 한 sweep. |
| 첫 5회 | 센서별 **Teacher10 + Student85 = 95개 학습**, 두 센서 합계 **190개 학습**. Calibration20개는 별도 비용. |
| 5회 동안 | 공통 BASE recipe 고정. 좋은 결과가 일찍 나와도 중단·seed 교체·임계값 조정 없음. |
| 5회 이후 | 센서별 운영 기준 산출 → 역전/불안정 관계의 paired 재시험 → 제한된 fitting → 전체 표 5회 refresh. |
| 상시 반복 | `max_campaign_cycles=null`. 전체 반복 횟수 상한은 두지 않되, 개별 재시험·tuning 묶음은 유한하다. |
| 같은 가설 재시험 | relation×recipe당 추가 **5 pair 한 묶음**. 반복적으로 안 되면 단순 seed 재추첨 대신 진단·recipe 검토로 이동한다. |
| 전체 표 refresh | 각 바깥 cycle에서 17행×5 sweep을 사전 확정해 수행한다. 과거 통과한 행도 다시 측정한다. |
| 최종 확인 | 특정 recipe를 잠근 후 새 5 sweep. 정확히 같은 recipe에 대해 확인이 통과할 때까지 seed 묶음을 교체하지 않는다. |
| 기본 자동 추가 대상 아님 | F01/F02, X00–X06, 100K, PAN 재구성, architecture·data phase·평가기 변경. |

**실행 총시간은 이번 요청에서 지정되지 않았다.** GF2의 20시간 window를 가져오지 않는다. 아래 자원 lease는 장기 실행의 안전장치이며 실험 전체의 종료 시간은 아니다.

기존 WV3 생산 recipe는 종결 상태로 보존한다. 이번 요청은 WV3의 **별도 ablation·개발 campaign을 새로 여는 승인**으로 해석하며, 기존 대표 checkpoint·논문 기록을 소급 교체하지 않는다. [S2]

---

## 1. 근거·관측·새 설계를 분리한다

### 1.1 그대로 유지하는 근거

이전 ablation 계획의 C00–C16은 Teacher의 A 초기화, prediction, difficulty, q를 분리했고, 누적표와 FULL 제거표를 함께 정의했다. 그 component 의미와 gradient-routing 계약을 유지한다. [S1]

WV3 최종 고정과 QB 기본 recipe는 backbone, Teacher–Student 역할, adaptive hard/soft, q-edge, q-hard A adjustment가 같다. 밴드 수·데이터·평가 preset과 학습된 reference는 센서별이다. [S2,S3]

과거 GF2 감사의 DUAL↔P0 비교는 task·입력·폭·modulation이 동시에 달라 효과를 분리하지 못했다. 이번 반복에서는 그와 같은 동시 변경을 component 효과로 읽지 않는다. 이 감사의 B01은 현재 PANDA가 아니다. [S4 §0, §4.3]

### 1.2 이번에 새로 제안하는 것

**초기 5회, seed 표, 5-pair 재시험, 센서별 MAD 기반 운영 기준, cycle 스케줄, recipe 후보의 실행 우선순위, 72시간 갱신형 lease**는 이번 설계값이다. 이미 측정된 최적 반복 수·통계적 임계값·성공 확률이 아니다.

이번 작성에서는 최신 Sheet나 서버 프로세스를 새로 감사하지 않았다. 첨부 이전 계획과 제공된 진단이 근거다. 시작 전 현재 checkout·진행 중 작업·데이터·자산을 확인하며, 오래된 정적 문서의 상태를 현재 실행 상태로 간주하지 않는다.

### 1.3 해석상 바로잡을 점

- Teacher를 포함한 Student가 항상 Teacher 자체보다 모든 지표에서 좋아야 하는 것은 아니다. Teacher와 Student의 구조·입력·학습 목적이 다르므로 이것은 component 인과 대조가 아니다.
- Teacher의 기여는 **C03−C02(A 초기화), C04−C03(output target), C07−C10(전체 Teacher 경로)**로 검사한다.
- `a_T>0`은 **Teacher가 Student보다 GT에 가까운 위치**다. Student가 우세한 위치라는 반대 의미로 쓰지 않는다.
- q는 상대 shift 반응의 진단값이고 native 변위 GT가 아니다. q가 좁은 범위에 모이면 native 지표의 이득이 작을 수 있다. 그때 임계값을 억지로 바꾸지 않는다.
- 한 번의 좋은 결과, DEV 반복 평균의 좋은 결과, 잠근 recipe의 별도 seed 확인은 다른 수준의 결과다.

---

## 2. 센서·모델·데이터 고정표

| 항목 | s1 / WV3 | s2 / QB |
|---|---|---|
| Teacher | P0, W112, D[1,2,3] | 동일 |
| Student | W104, D[1,2,2], C00–C16 switch | 동일 |
| MS / HRMS bands | 8 | 4 |
| Teacher 입력 | PAN1 + upsampled MS8 = 9ch | PAN1 + upsampled MS4 = 5ch |
| Student 저장 frontend | P/L/H + MS8 = **11 slots** | P/L/H + MS4 = **7 slots** |
| P0 Student ablation | L/H slot을 학습·추론 모두 0 | 동일 |
| Aligner MS stem | 8ch | 4ch |
| Task / attention / modulation | HRMS-only / OFF / OFF | 동일 |
| maxDN | 2047 | 2047 |
| 정규화 | `2*DN/2047−1` | 동일 |
| train / val | 검증된 WV3 source·복구 LP | 검증된 QB msfix train/val·복구 LP |
| nominal train 수 | 9,714 | 17,139 |
| nominal validation 수 | 1,080 | 이 문서에서 새 수치를 가정하지 않음. 실제 H5 확인 |
| PAN train / RR / FR 크기 | 64 / 256 / 512 | 동일 |
| 배율 | 4 | 4 |
| RR 다중밴드 지표 | Q8 | Q4 |
| FR MTF | WV3 preset | QB preset |
| 공식 FR reference | 원 native PAN / original LMS | 동일 원칙 |
| 학습 / batch | 50K / 48 | 동일 |
| U peak LR / Student A peak LR | 1e−4 / 3e−6 | 동일 |
| Teacher A peak LR | 1e−5 | 동일 |
| Optimizer | AdamW(.9,.999), eps1e−8, wd.01 | 동일 |
| Scheduler | warmup100 + cosine50K | 동일 |
| Precision | FP32, AMP OFF. 실제 runtime flags를 pin | 동일 |
| 기본 FULL 계수 | α=1, β=.1, λE=.002, λcon=1e−4 | 동일 |
| Teacher / calibration | 각 sweep 새 로컬 Teacher, 센서별 재측정 | 동일 |
| 기본 기록 탭 | `WV3-s1`, 별도 ABLR2 namespace | `QB-s2`, 별도 ABLR2 namespace |

입력 mask는 저장 parameter shape를 맞추는 대조다. 비활성 channel은 유효한 입력 정보·학습 자유도를 바꾸므로 이를 완전히 같은 활성 모델 용량이라고 과장하지 않는다. no-A/frozen-A의 활성 parameter·연산량·Teacher 비용도 따로 기록한다.

확인할 source는 파일명만이 아니라 data/LP canonical tensor hash, sample order, GT/PAN 불변, band order다. 테스트 입력을 GT로 재생성하거나 test 성능에 맞게 phase·MTF를 변경하지 않는다. [S2,S3]

---

## 3. FULL 수식과 variant 계약

### 3.1 Forward

\[
M=U_4(MS),\quad L=U_4(LPAN),\quad c=A(P_{m4},M_{m4}),
\]
\[
\widetilde P=W(P,c),\quad\widetilde L=W(L,c),\quad
\widetilde H=\widetilde P-\widetilde L.
\]

Teacher는 `[P̃,M]`, Student FULL은 `[P̃,L̃,H̃,M]`를 사용한다. PAN과 upsampled LP는 같은 FP32 bicubic grid, border padding, `align_corners=False`로 warp한다. HP는 signed 차이이고 별도 clipping/abs/정규화를 하지 않는다. MS base는 한 번만 더하며 MS·GT·출력 좌표는 이동하지 않는다.

LP 생성은 기존 native PAN의 Gaussian σ1.98/k41/replicate/[2::4,2::4]를 유지한다. 알고리즘용 LP를 평가 MTF와 혼동하지 않는다. Student 추론은 자기 A+U만 사용한다.

### 3.2 Teacher

\[
L_T=\operatorname{mean}|Z_T-Y|+
\mathbf1[t\bmod2=1]\lambda_{\rm con}
\operatorname{mean}|c_\epsilon+\epsilon-\operatorname{sg}(c_0)|.
\]

- **TPLUS:** λcon=1e−4, fresh A/U, 정확한 50K endpoint.
- **TZERO:** λcon=0, TPLUS와 같은 초기 A/U·data/augmentation prefix, fresh50K.
- synthetic shift는 반경2 HR pixel 원판의 면적균등, 독립 CPU RNG다. synthetic PAN은 A-only branch로만 들어간다.
- TZERO가 synthetic branch를 실행하지 않더라도 native data/RNG stream은 TPLUS와 같아야 한다. 별도 corruption generator를 쓴다.
- Teacher native rec는 A/U 모두 학습한다. consistency의 기준 c0만 detach한다.
- Teacher 성능이 낮아도 수치적으로 유효하면 그 sweep의 Student를 진행한다. 좋은 Teacher만 뽑아 파일럿 5개를 구성하지 않는다.

### 3.3 Reference bundle

Teacher별 train3072 base IDs의 full-pixel unaugmented error 중앙값으로 τR를 계산한다. q는 AXIS16의 [.25,.5,1,2]×4방향, 실제 fixed-HV/ROT4 view로 계산한다. qref는 calibration3072×4 view median, cache는 전체 train×4 view다.

\[
\tau_R=\max(\operatorname{median}_{i,p}\operatorname{mean}_b|Z_T-Y|,10^{-6}),
\quad s_{i,r}=\frac{q_{\rm ref}}{q_{\rm ref}+q_{i,r}}.
\]

C15의 \(\bar s\)와 X03의 \(\bar d\)는 해당 Teacher의 **전체 train×실제 view** 평균이다. 정의·표본 집합·hash를 저장한다. C06의 상수 .5와 C15의 측정 평균은 같다고 가정하지 않는다.

Teacher A/U, endpoint, source/data/LP/view/calibration IDs, τR/qref/cache를 bundle로 보존한다. TPLUS와 TZERO, WV3와 QB의 bundle을 섞지 않는다. calibration 값은 실행 전 null이며 가중치 튜닝용 자유변수가 아니다.

### 3.4 Student

\[
e_T=\operatorname{mean}_b|Z_T-Y|,\quad e_S=\operatorname{mean}_b|Z_S-Y|,
\quad d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},
\quad a_T=\operatorname{sg}\frac{[e_S-e_T]_+}{e_S+10^{-6}}.
\]
\[
\ell_{H,i}=\operatorname{mean}_p[(1+\alpha d_T)e_S],\qquad
\ell_{K,i}=\operatorname{mean}_p[\beta(1-d_T)a_T\operatorname{mean}_b|Z_S-Z_T|],
\]
\[
L_U=\operatorname{mean}_i[\ell_{H,i}+\ell_{K,i}+\lambda_Es_{E,i}\ell_{E,i}],
\qquad L_A=\operatorname{mean}_i[s_{A,i}\ell_{H,i}].
\]

FULL은 α=1, β=.1, λE=.002, sE=sA=si다. GT edge는 기존 signed Scharr이며 방향별 0.5, 경계 1px 제외, band mean을 유지한다.

**U gradient는 LU에서, A gradient는 LA에서 별도로 계산한 다음 optimizer step을 수행한다.** Soft/edge를 A의 직접 목적함수에 추가하지 않는다. C00–C06의 q 미도입 단계는 s0=.5, C15는 측정된 s̄를 쓴다. 제거 case의 0은 명시적인 ablation이며 FULL의 기본 식을 바꾸는 것이 아니다.

---

## 4. Component case 정의 — 17행을 고정한다

### 4.1 누적 구성 C00–C07

| Case | 입력 | Student A | Teacher 사용 | Supervision | 비교 |
|---|---|---|---|---|---|
| C00 | P0 mask | 없음 | 없음 | GT L1 | 기준 |
| C01 | PLH | 없음 | 없음 | GT L1 | C01−C00: LP/HP |
| C02 | PLH | fresh trainable | 없음 | GT L1, A는 .5×hard | C02−C01: scratch A |
| C03 | PLH | clone trainable | A 초기화만 | GT L1, A는 .5×hard | C03−C02: Teacher A transfer |
| C04 | PLH | clone trainable | A + prediction | plain hard + **uniform KD** | C04−C03: output KD |
| C05 | PLH | clone trainable | A + prediction + d | adaptive hard + selective soft | C05−C04: adaptive fitting 묶음 |
| C06 | PLH | clone trainable | 위와 같음 | C05 + GT edge, sE=sA=.5 | C06−C05: edge |
| C07 | PLH | clone trainable | A + prediction + d + q | **FULL 식** | C07−C06: q-dependent 배분 |

C00–C07은 이어 학습하는 단계가 아니라 **각각 독립적인 fresh Student**다. “Teacher 다음 Student”는 reference dependency 순서를 뜻하며, “C00 다음 C01”은 weight continuation이 아니다.

C04의 uniform KD는 a와 (1−d)가 모두 1이다. C05는 hard 강조와 soft selection을 함께 추가하므로 개별 gate의 인과효과로 나누지 않는다. C03은 output KD가 없다.

### 4.2 FULL 제거 C08–C16

| Case | 제거 | 정확한 조작 | 주 대조 |
|---|---|---|---|
| C08 | LP/HP | L/H를 처음부터 0. Teacher는 P0 그대로 | C07−C08 |
| C09 | Student 정합 | c=0, warp bypass, A optimizer 없음. Teacher 경로 유지 | C07−C09 |
| C10 | Teacher 전체 | A/U fresh, α=β=0, d/q/Teacher I/O 없음, s=.5, GT edge 유지 | C07−C10 |
| C11 | Teacher consistency | matched TZERO 전체 bundle 소비 | C07−C11 |
| C12 | hard 추가 강조 | α=0만. Soft trust의 d는 유지 | C07−C12 |
| C13 | output KD | β=0만. Clone/d/q 유지 | C07−C13 |
| C14 | GT edge | λE=0만. q-hard A 유지 | C07−C14 |
| C15 | sample별 q | sE=sA=해당 train-view 평균 s̄ | C07−C15 |
| C16 | Student A adjustment | clone A를 eval/frozen, buffer·decay·step 없음. 보정은 수행 | C07−C16 |

C00/C01/C02/C10은 Teacher를 만들지 않는 독립 실행으로도 동작해야 한다. 전체 sweep이 Teacher부터 시작하더라도 **이 run들의 dataset/loss/model 안에는 Teacher I/O가 없어야 한다.** C03은 clone용 A는 필요하지만 Teacher output과 calibration 값은 쓰지 않는다.

C09는 Student 정합만 제거한다. Teacher까지 정합이 없는 대조는 optional X00이다. C16은 no-alignment가 아니라 no-adjustment다. C11의 효과는 prediction·A clone·calibration까지 달라지는 reference 전체 효과다.

### 4.3 기대 관계 graph

```text
L01: C01 > C00     L02: C02 > C01     L03: C03 > C02
L04: C04 > C03     L05: C05 > C04     L06: C06 > C05
L07: C07 > C06
D08..D16: C07 > C08..C16
E00: C07 > C00
```

총 **17개 비교 관계**다. `>`는 기대하는 공동 품질 향상이며, 모든 metric에서 매번 엄격한 증가를 강제하지 않는다. Flow 모양을 보고 행 순서를 바꾸지 않는다.

별도의 Teacher endpoint↔C07 표는 보존하되, 위 graph에 넣어 Student가 반드시 더 큰 Teacher를 이겨야 할 제약으로 사용하지 않는다.

---

## 5. 첫 5회 BOOT5 — 내부 반복의 기준을 만드는 단계

### 5.1 정확한 seed 배정

| Sweep | s1 WV3 Teacher | s1 Student | s2 QB Teacher | s2 Student |
|---|---:|---:|---:|---:|
| P01 | 781001 | 791001 | 881001 | 891001 |
| P02 | 781002 | 791002 | 881002 | 891002 |
| P03 | 781003 | 791003 | 881003 | 891003 |
| P04 | 781004 | 791004 | 881004 | 891004 |
| P05 | 781005 | 791005 | 881005 | 891005 |

각 Teacher seed에서 TPLUS/TZERO를 matched 초기화로 둘 다 학습한다. 각 Student seed는 **17개 구성 전체의 공통 U 초기화·sample/view 순서**에 사용한다. 같은 sweep의 clone A는 해당 TPLUS, C11만 TZERO를 소비한다. C02/C10의 fresh A는 별도의 같은 scratch-A 초기화 규칙을 공유한다.

Teacher와 Student가 sweep마다 함께 바뀌므로 BOOT5의 분산은 **전체 파이프라인 반복 변동**이다. Teacher 고정 Student seed 변동만이라고 부르지 않는다. **Teacher 5 × Student 1** 구조로 보고한다.

### 5.2 한 sweep의 내부 순서

1. Source·sensor·recipe·seed·grid·component flags를 immutable manifest로 고정한다.
2. TPLUS/TZERO를 둘 다 학습하고 각자의 calibration을 완결한다. 홀수 sweep은 TPLUS부터, 짝수는 TZERO부터다.
3. C00–C16을 사전에 정한 순서로 각각 fresh 학습한다.
4. 각 run의 endpoint/VAL 및 별도 RAW 진단을 저장·readback한다.
5. 17개 구성의 기술적 상태와 전체 metrics가 완결되면 sweep을 닫는다. 좋은 모양을 완결 조건으로 쓰지 않는다.

Student 실행 순서는 `[C00,…,C16]`을 `4*(k−1) mod17`칸 회전하고, 짝수 k에서는 역순으로 한다. **표의 표시 순서는 항상 C00→C07/제거표 고정**이다. 실행 순서 분산이지 결과의 재배열이 아니다.

### 5.3 BOOT5에서 하지 않는 것

5회 중간에 FULL만 새 seed로 추가하거나, poor Teacher를 버리거나, 유리한 checkpoint rule로 갈아타지 않는다. 수치 오류 수정이 필요하면 source revision과 영향받는 전체 비교를 명시한다.

낮은 성능은 실패가 아니며 그대로 유효한 관측이다. 반복되는 NaN/발산은 안정성 결과와 technical 원인을 분리해 기록하고, 5개의 좋은 run이 모일 때까지 조용히 대체하지 않는다.

5개 완성된 paired 값이 없으면 그 관계는 `PILOT_INCOMPLETE`다. 임계값을 0이나 다른 센서 값으로 채우지 않는다. 기술적 결측을 복구하거나 명시적인 계획 revision으로 표본수를 변경해야 한다.

---

## 6. 평가와 ‘좋은 흐름’의 숫자 기준

### 6.1 checkpoint와 metric을 먼저 고정한다

| 항목 | 규칙 |
|---|---|
| 주 checkpoint | 기존 50개 grid 중 **RR validation ERGAS 최소**, 완전 동률이면 낮은 step |
| 후보 grid | `1010,2020,…,49490,50000` |
| 보조 주 분석 | **Exact50K**. FULL만 early best, 다른 행은 last로 혼합하지 않음 |
| 지표 묶음 | 같은 checkpoint의 RR test/FR native 지표를 함께 보존 |
| RAW_MAX/TARGET/E_MIN | test-aware 진단. 모양 판정의 기본 checkpoint로 사용하지 않음 |
| FR 개발 선택 | 독립 FR 개발 세트가 확인되지 않았으므로 기본 운영은 **TEST_AWARE_DEV** |

이미 사용한 native FR20의 H를 반복 우선순위나 fitting guard에 쓰면 그 사실을 명시한다. 새 Student seed로 확인해도 같은 FR20이 새로운 독립 test set이 되지는 않는다.

독립 FR_DEV가 실제로 있고 사용이 허가된 경우에만 BOOT5 이전에 고정하여 대체한다. 없는데 존재한다고 가정하거나 기존 test를 나중에 나눠 미사용 자료라고 하지 않는다.

### 6.2 paired 차이

관계 e의 parent=P, child=C, sweep k에 대해:

\[
\Delta H_{e,k}=H_{C,k}-H_{P,k},\qquad
rE_{e,k}=E_{C,k}/E_{P,k}-1.
\]

H는 양수, rE는 음수가 개선이다. E는 같은 선택 checkpoint의 RR ERGAS다. Fitting용의 실제 validation ERGAS(`E_val`)와 이 RR test ERGAS를 다른 필드로 보존한다. SAM/Q4·Q8/Dλ/Ds도 보고하되, 순위가 좋은 metric만 나중에 주지표로 바꾸지 않는다.

### 6.3 BOOT5에서 운영 허용폭을 산출한다

각 센서별로 17개 관계 모두에서:

\[
MAD_H(e)=\operatorname{median}_{k=1..5}
|\Delta H_{e,k}-\operatorname{median}_j\Delta H_{e,j}|,
\]
\[
MAD_E(e)=\operatorname{median}_{k=1..5}
|rE_{e,k}-\operatorname{median}_j rE_{e,j}|.
\]

그 후 센서 공통값을 다음과 같이 정한다.

\[
\epsilon_H=\operatorname{clip}(0.25\operatorname{median}_eMAD_H(e),10^{-5},10^{-3}),
\]
\[
\epsilon_E=\operatorname{clip}(0.25\operatorname{median}_eMAD_E(e),10^{-4},0.005),
\]
\[
\delta_H=\max(0.001,2\epsilon_H),\qquad
\delta_E=\max(0.003,2\epsilon_E).
\]

`clip(x,a,b)=min(max(x,a),b)`다. rE는 비율이므로 .005는 0.5%, .003은 0.3%다. Epsilon은 운영상 허용할 작은 차이, delta는 개선 후보를 고를 때 요구할 효과 크기다.

**이 계산은 자동 탐색의 운영 허용폭일 뿐 통계적 유의성·noise floor·정합 정확도 한계가 아니다.** 17개 관계는 공통 run을 공유하므로 독립 관측 17개도 아니다. 5회는 작은 표본이고 이질적 component의 변동까지 섞인다. 원 MAD와 cap 적용 여부를 모두 보고한다.

위 상수와 산식은 BOOT5 전에 고정한다. 측정된 epsilon/delta는 처음에는 null이며, 5회 완료 후 센서별 `thresholds_v1.json`에 저장한다. **역전된 행만 다른 기준을 쓰거나, 통과할 때까지 허용오차를 늘리지 않는다.** 이후 재보정은 사용자 승인 새 threshold revision이며 기존 판정은 보존한다.

### 6.4 관계별 운영 상태

판정은 사전 등록된 완성 batch 전체로 하며 기본 batch 크기는 5다. n이 5보다 크면 support 개수는 `ceil(.8*n)`으로 일반화한다. 주값은 paired 중앙값이다.

| 상태 | 운영 규칙 |
|---|---|
| `JOINT_GAIN` | median ΔH≥δH, median rE≤−δE, 최소 4/5에서 ΔH≥−εH 및 rE≤εE |
| `H_GAIN_SAFE` | median ΔH≥δH, ΔH>εH가 4/5 이상, median rE≤εE, 개별 rE≤1% |
| `RR_GAIN_SAFE` | median rE≤−δE, rE<−εE가 4/5 이상, median ΔH≥−εH, 개별 ΔH≥−max(.003,2εH) |
| `REVERSAL` | parent와 child를 뒤집으면 위 개선 규칙을 충족. 예상과 반대의 지속적 방향 |
| `TRADEOFF` | H 이득/E 손실 또는 반대가 각각 허용폭 밖. 한 숫자로 성공 처리하지 않음 |
| `NEAR_ZERO` | 두 중앙값이 각각 ±epsilon 범위. 동등성 입증이 아니라 현재 운영 해상도에서 작은 차이 |
| `UNSTABLE` | 위 상태에 들지 않으며 seed 간 방향이 혼재 |
| `INVALID/INCOMPLETE` | 기술적 검증 실패/결측. 좋고 나쁨의 판정 대상이 아님 |

구현 순서는 먼저 유효성, 그 다음 child의 개선 규칙, parent의 개선 규칙, TRADEOFF, NEAR_ZERO, UNSTABLE이다. Parent 규칙에서는 `E_parent/E_child−1`을 다시 계산한다. rE의 부호만 뒤집어 정확한 비율인 것처럼 쓰지 않는다.

주 VAL 선택에서 개선이더라도 Exact50K에서 큰 반대 방향이면 `CHECKPOINT_SENSITIVE`를 추가한다. 역전 우선순위에서는 이것도 확인 대상이다. 더 좋아 보이는 checkpoint 표로 주표를 교체하지 않는다.

### 6.5 flow는 점수가 아니라 별도의 대시보드다

다음 수치만 집계한다.

- 누적 7관계 중 개선/미검출/역전/상충 개수.
- FULL 제거 9관계 중 동일 개수와 각 paired 차이.
- E00(FULL−C00)의 절대 개선, FULL 자체의 H/E와 센서 목표 달성률.
- 같은 5개 sweep을 쓴 전체 case의 평균±SD와 개별 점.

`FLOW_CANDIDATE_DEV`는 E00의 개선이 있고, 누적/제거 관계에 운영상 REVERSAL이 없을 때의 **검토 알림**이다. 모든 행의 모든 지표가 엄격히 단조롭다는 뜻도, 논문 확증이라는 뜻도 아니다. 알림이 떴다고 현재 5회 묶음을 조기 종료하지 않는다.

q 등 일부 관계가 NEAR_ZERO이면 그대로 보인다. 사소한 차이를 크게 만들기 위해 계수나 seed를 선별하지 않는다. 기존 benchmark 목표는 별도 metadata이지 component 효과의 epsilon/delta가 아니다.

---

## 7. 내부 반복 I — 역전 관계의 paired 재시험

### 7.1 우선순위

BOOT5 이후 각 센서는 다음 순서로 관계를 고른다.

1. 기술적 무결성 문제는 성능 탐색과 분리해 먼저 복구한다.
2. `REVERSAL` 또는 FULL의 E00/C10 비교 악화.
3. `CHECKPOINT_SENSITIVE`, `UNSTABLE`.
4. 의미 있는 TRADEOFF.
5. NEAR_ZERO는 맨 뒤이며 즉시 무한 재추첨하지 않는다.

동급에서는 아직 재시험하지 않은 관계, 마지막 측정이 오래된 관계, relation ID 순서다. **지금 성능이 가장 낮은 baseline을 골라 차이를 벌리는 순위가 아니다.**

### 7.2 한 relation의 재시험 묶음

- Parent와 child를 **같은 새 Student seed로 동시에 등록**한다. 대조군은 예전의 최저 run으로 고정하지 않는다.
- FULL C07이 쌍에 없으면 같은 seed의 C07을 anchor로 추가한다. 이미 같은 seed·reference·recipe·source로 있으면 중복 학습하지 않는다.
- Student seed 5개를 한꺼번에 등록하고 모두 완료한다. 한 번 이겼다고 남은 4개를 취소하지 않는다.
- Teacher는 **현재 recipe의 가장 최근 완성 5-sweep 패널**의 TPLUS/TZERO를 순서대로 한 번씩 쓴다. 처음에는 BOOT5가 그 패널이다. 좋은 Teacher 순으로 선별하지 않는다.
- R04 등 Teacher recipe가 바뀌면 그 recipe에 맞는 bundle을 사용한다. 예전 R00 Teacher를 현재 reference로 몰래 재사용하지 않는다.
- C11 관계는 각각 matched TPLUS/TZERO를 사용한다. C10 등 Teacher-free는 참조를 소비하지 않는다.
- 한 relation 재시험은 FULL이 포함되면 Student 10개, 별도 anchor가 필요하면 최대 15개다. Teacher 새 학습은 기본적으로 없다.

이 묶음은 각 관측 Teacher에서 Student의 새 seed 효과를 추가한 것이다. 같은 Teacher의 여러 Student를 Teacher 여러 번 재학습한 독립 반복으로 세지 않는다.

### 7.3 재시험 뒤의 분기

| 결과 | 후속 |
|---|---|
| 재시험에서도 예상 방향 유지 | `SUPPORTED_DEV`로 기록. 다음 전체 refresh에서도 계속 측정 |
| 첫 5회와 방향이 바뀜 | `SEED/REFERENCE_SENSITIVE_DEV`; 기전 진단·recipe 검토 |
| 역전이 반복 | `REPEATED_NEGATIVE_DEV`; 같은 recipe의 새 seed 추첨 대신 fitting으로 이동 |
| 거의 차이 없음 | `SMALL_OR_UNDETECTED_DEV`; native 수치 점프를 강요하지 않고 필요한 기전 study로 이동 |
| metric 상충 유지 | `PERSISTENT_TRADEOFF_DEV`; FULL 절대 목표와 학습 균형 검토 |

**relation×recipe당 재시험 batch 상한은 1회(5 pair)**다. 상시 루프의 수명이 무한하더라도 동일 관계 한 곳에서 빠져나오지 못하는 루프는 만들지 않는다. 전체 refresh에서는 그 관계를 계속 관측하므로 좋았던 관계를 영구 확정하지도 않는다.

---

## 8. 내부 반복 II — 공통 recipe fitting

### 8.1 사전 허용한 후보

이전 계획의 6개 recipe를 두 센서의 공통 후보 집합으로 사용한다. 센서별 승자는 다를 수 있다. 각 후보는 **R00 기준의 한 축 변경**이며 임의 조합을 자동 생성하지 않는다. [S1 §8]

| Recipe | α | β | λE | T λcon | Student A schedule |
|---|---:|---:|---:|---:|---|
| R00 | 1 | .1 | .002 | 1e−4 | BASE |
| R01 | 1 | .2 | .002 | 1e−4 | BASE |
| R02 | 1 | .1 | .001 | 1e−4 | BASE |
| R03 | .1 | .1 | .002 | 1e−4 | BASE |
| R04 | 1 | .1 | .002 | 3e−4 | BASE |
| R05 | 1 | .1 | .002 | 1e−4 | 24,240 완료 후 BASE A LR×1/3 |

이 값들은 WV3/QB의 공통 최적값으로 확정된 것이 아니다. 센서별로 따로 DEV 검정한다. 이번 자동 bank 밖의 100K·다른 LR·새 loss·PAN task·data 처리는 사용자 revision 없이 추가하지 않는다.

### 8.2 어떤 후보를 먼저 검정하는가

| 미해결 관계 | 먼저 검토할 후보/진단 |
|---|---|
| C04−C03, C07−C13 등 prediction KD | R01, soft 활성률·weighted gradient |
| C05−C04, C07−C12 등 hard fitting | R03, d와 hard 항의 실제 질량 |
| C06−C05, C07−C14 등 edge | R02, hard/edge gradient·경계 오차 |
| C07−C11 등 Teacher consistency | R04, 같은 seed의 새 TPLUS와 calibration |
| C03−C02, C07−C16 등 A 초기화/조정 | R05, cT/cS·early/late 추세 |
| C01−C00, C07−C08 등 주파수 | 입력/LP/mask regression, optional F01/F02. 불리한 입력 비율을 발명하지 않음 |
| C07−C15 등 q | s 분포·평균 규모 대조, optional X04/X05. qref 수동 조정 없음 |

공통 동급 순위는 R01→R02→R03→R04→R05다. 기전 우선순위와 근거를 로그로 남기고 후보 목록은 학습 전에 pin한다.

### 8.3 한 fitting round의 처리

1. 최대 **2개 미검정 후보**를 등록한다.
2. 고정 DEV block P01/P02의 seed·data stream으로 incumbent와 후보 FULL을 대응시킨다.
3. 문제 relation의 parent/child/필요 C07도 같은 후보 recipe에서 검정한다. 다른 행은 의도적으로 낮은 설정으로 남기지 않는다.
4. 동일한 연산·초기값·참조·RNG·source·선택 규칙을 이미 실행했으면 재사용한다. Recipe 문자열만 다른 중복 job은 새 관측으로 세지 않는다.
5. R04라면 해당 Teacher seed마다 새 TC3와 전체 calibration을 만든다. TZERO 재사용은 수치 identity가 정확히 동일할 때만 가능하다.
6. 두 block을 끝낸 뒤 FULL 자체의 개선으로 후보를 고른다. 2회 screen은 확증이 아니다.

**후보 선택 목적함수는 “FULL과 제거군의 간격 최대화”가 아니다.** 비교 행의 간격이 벌어져도 FULL 자체가 좋아지지 않으면 승격 이유가 되지 않는다.

### 8.4 구체적인 DEV 승격 규칙

기본 선택축은 실제 RR validation ERGAS다. 후보 FULL이 두 block 모두 E_val를 개선하고 중앙 상대 개선≥0.3%이며 H/Dλ 손실 guard를 넘지 않으면 후보가 된다. 이 경로의 guard는 각 ΔH≥−max(.003,2εH), Dλ 중앙 악화≤.001/개별≤.002다.

독립 FR_DEV가 없는 기본 운영에서는 H를 쓰는 다음 경로를 **TEST_AWARE_DEV**로 명시한다. 두 block 모두 FULL ΔH>εH, 중앙 ΔH≥δH, E_val 중앙 악화≤0.5%/개별≤1%, Dλ 중앙 악화≤.001/개별≤.002이면 H 중심 후보가 될 수 있다.

두 조건을 모두 충족한 후보를 우선하고, 동급이면 더 낮은 median E_val, 그 다음 높은 median H, 그 다음 recipe ID 순이다. 좋은 계단 모양 자체는 tie-break가 아니다. 어느 후보도 유효하지 않으면 incumbent를 유지한다.

선정은 `PROVISIONAL_RECIPE`이고 즉시 논문 최종 조건이나 과거 FULL의 대체가 아니다. 다음 REFRESH5에서 새 Teacher·새 Student로 전체 표를 완결해야 한다. 서로 다른 후보는 R00에서 각각 한 축을 바꿨지만, 후보끼리는 둘 이상의 값이 다를 수 있다는 점도 기록한다.

### 8.5 recipe를 case에 적용하는 순서

`공통 recipe → case 제거 override → 의존성 해결 → immutable config` 순서다.

R01이어도 C13의 β는 0, C04는 uniform KD, C10은 Teacher-free다. R03이어도 C12의 α는 0. R02이어도 C14의 edge는 0이다. R05이어도 C16/identity A에 optimizer를 실행하지 않는다.

FULL만 다른 recipe를 쓰고 제거군은 원 recipe를 유지하는 혼합표는 만들지 않는다. 새 recipe 결과와 옛 recipe 결과는 서로 다른 revision이다.

---

## 9. 바깥 반복 — 상시 controller와 전체 표 refresh

### 9.1 lifecycle

```text
PREFLIGHT
  -> BOOT5 (R00 전체 표 5회)
  -> CALIBRATE_THRESHOLDS (센서별 1회)
  -> ANALYZE_RELATIONS
  -> RECHECK5 (해당 relation에서 아직 하지 않은 경우)
  -> FIT_ROUND (최대 2개 미검정 후보, 없으면 생략)
  -> REFRESH5 (선정/유지 recipe의 전체 표 새 5회)
  -> PUBLISH_DEV_REPORT
  -> ANALYZE_RELATIONS ...
```

각 서버는 자기 상태만으로 진행한다. 두 센서의 5회가 동시에 끝나야 한다는 barrier는 없다. 느린 s1을 기다리며 s2를 멈추거나 반대로 하지 않는다.

### 9.2 REFRESH5의 중요한 역할

- 새 Teacher 5 seed와 새 Student 5 seed를 결과를 보기 전에 한꺼번에 등록한다.
- 각 sweep마다 TPLUS/TZERO 및 17개 case를 완결한다. **95개 학습/센서**의 전체 묶음이다.
- FULL에 유리해진 한 관계만 갱신하지 않고, 기존에 잘 나온 관계도 다시 측정한다.
- REFRESH5도 DEV다. 그 결과를 보고 다음 recipe를 고르므로 자동으로 독립 확증이 되지는 않는다.
- 같은 정확한 recipe/source/data/선택 규약의 완결 full sweep은 누적 평균에 모두 포함한다. 첫 BOOT5도 동일 조건이면 포함한다.
- Recipe가 다르면 평균을 합치지 않는다. 관계별 추가 RECHECK 결과는 표본수가 달라지므로 별도 paired 패널로 보존하고 전체 17행 평균에 몰래 합치지 않는다.

### 9.3 좋은 모양이 나와도 한 batch를 끝낸다

REFRESH5의 두 번째 sweep에서 기대 관계가 모두 좋게 나와도 남은 3회를 실행한다. 알림만 남긴다. 5회 끝의 전체 결과에서 flow 표를 만든다.

새로운 전체 표의 부정적 결과가 이전 좋은 결과를 대체해 사라지지 않고, 이전 좋은 결과만 대표로 남지도 않는다. 버전별·wave별 결과와 누적 paired 분포를 함께 저장한다.

### 9.4 후보 소진·plateau

정한 6 recipe를 모두 검정했거나 같은 relation의 추가 재시험 한 묶음을 다 썼는데도 흐름이 개선되지 않으면 `PLATEAU_UNRESOLVED`를 보고한다.

상시 실행 모드에서는 새 값을 즉석으로 발명하는 대신, 현 incumbent의 **균형 잡힌 REFRESH5**를 계속 수행할 수 있다. 세 cycle 연속 FULL 절대 개선이나 관계 분류의 안정화가 없으면 운영자에게 `LOW_INFORMATION_GAIN` 알림을 보낸다. 장기 엔진은 종료하지 않아도 되지만 기대 모양의 달성을 보장하지 않는다.

새 component나 새 조합을 시험하려면 별도의 승인 revision이 필요하다. 동일한 성능 불리 case만 무한 재추첨하는 while-loop는 없다.

### 9.5 내부 반복에 대한 핵심 답

“Teacher를 추가한 Student가 낮게 나왔다”면 그 Student 한 행을 성공할 때까지 덮어쓰는 것이 아니라:

```text
해당 의미를 가진 comparison 선택
-> matched parent/child + 필요 FULL의 새 5 pair
-> 전체 5 pair 결과 해석
-> 반전 지속이면 공통 recipe 또는 기전 검토
-> 새/유지 recipe의 전체 17행을 다시 5회
```

이 구조로 **계속 시도할 수 있는 자동화**와 **component 효과를 읽을 수 있는 대조**를 함께 유지한다.

---

## 10. 새 5회 확인 VERIFY5

DEV에서 후보가 정해지면 recipe, 17 case, reference 정책, 학습 길이, 평가 규약, threshold revision을 잠근 후 VERIFY5를 한 번 등록할 수 있다. 새 Teacher 5 + 새 Student 5로 **95개 학습/센서**다.

확인 wave는 정확한 동일 recipe당 최대 1회다. 실패하면 `VERIFY_NEGATIVE`를 보존한다. 똑같은 recipe에서 또 다른 seed 5개를 골라 통과한 wave만 최종으로 보고하지 않는다. 기술적 재실행은 같은 config/seed/fullstate 계약으로 별도 표시한다.

확인 뒤에는 상시 DEV controller가 계속 동작할 수 있다. 다른 recipe가 나오면 그 revision의 확인은 새로 등록하지만 이전 확인들의 존재와 결과를 함께 보고한다. 반복 선택된 후보의 같은 FR20 평가를 미사용 데이터의 불편 성능 추정으로 제시하지 않는다.

주표에 들어갈 확인 wave의 17행은 모두 같은 5 sweep을 쓴다. 일부 case만 확인했다면 그 범위를 명시하고 전체 표가 확인됐다고 쓰지 않는다.

보고 문장은 “5회에서 이러한 paired 방향과 분산을 관측했다” 수준으로 한다. 이 운영 gate만으로 통계적 유의성이나 보편적 우월성을 선언하지 않는다. 완전히 새로운 외부 검증 데이터가 필요하면 별도로 확보한 자료에서 확인해야 한다.

---

## 11. 추가 ablation — 기존 정의를 유지하되 조건부로

### 11.1 LP/HP: F01/F02

C08=P0와 C07=PLH를 재사용하고, F01=PL, F02=PH를 같은 seed/reference에 추가한다. 2개 구성×5 sweep=센서별 **Student 10개 추가**다.

H=P−L이므로 LP/HP는 독립적인 새 센서 관측이 아니다. 정보량 증가보다 명시적 주파수 표현·최적화의 효과로 읽는다. Input-mask capacity 정의는 C+3 slots로 WV3/QB에 맞춘다.

### 11.2 추가 PAN shift 강건성

저장된 C00/C02/C03/C07/C11/C16의 native VAL-selected checkpoint를 고정한다. 반경 0, .25, .5, 1, 2의 axis16+native1, 추가 반경 1.5의 diagonal4로 총 21조건이다. MS/GT는 고정하고 shifted PAN에서 LP를 재생성한다. 0 shift는 native tensor를 그대로 쓴다.

RR의 GT 복원 저하와 `cε+ε−c0`를 보고한다. 이는 추가 shift 응답 검정이지 native 변위 GT 검정이 아니다. Shift별 checkpoint를 재선택하지 않으며 공식 FR reference를 이동하지 않는다.

### 11.3 X00–X06의 정의

| ID | 정의 |
|---|---|
| X00 | TIDENTITY를 새로 학습하고 Teacher/Student 정합 모두 제거 |
| X01 | soft advantage aT=1만 변경 |
| X02 | soft trust (1−dT)=1만 변경 |
| X03 | hard weight를 train 평균 1+αd̄로 고정 |
| X04 | q-edge만 train 평균 s̄ |
| X05 | q-A만 train 평균 s̄ |
| X06 | Teacher-free C10의 fresh100K. Budget 대조이며 동일 GPU 시간 대조가 아님 |

기본 자동 queue에는 이 7개를 넣지 않는다. 필요한 관계·추가 비용·비교 상대를 정한 승인 revision으로만 연다. 주표 5회가 마음에 들지 않는다는 이유만으로 선택 study를 무한 추가하지 않는다.

---

## 12. Scheduler 상태기계와 구현 인터페이스

### 12.1 상태

| 상태 | 의미 / 다음 행동 |
|---|---|
| `DEFINED_NOT_LAUNCHED` | 설계만 존재. 문서 생성으로 기동하지 않음 |
| `WAIT_LEASE` | 유효한 운영자 자원 허가 대기 |
| `PREFLIGHT` | sensor/source/model/metric/runtime 확인 |
| `BOOT5` | 첫 전체 5 sweep 수행 |
| `CALIBRATE` | 센서별 5회 기반 운영 기준 작성·고정 |
| `ANALYZE` | 전체 패널과 relation 상태 계산 |
| `RECHECK5` | 문제 relation의 양쪽+anchor 재시험 |
| `FIT_ROUND` | 정한 후보 최대 2개 대응 검정 |
| `REFRESH5` | 새 5 seed의 전체 17행 |
| `VERIFY5` | 별도 잠근 확인 wave |
| `DRAIN` | 신규 입장 중지, 현재 등록 block/fullstate 정리 |
| `PAUSED_SAFE` | 재개 가능한 상태 |
| `STOPPED_BY_USER` | 명시적 종료 |
| `BLOCKED_NUMERICS` | code/data/method 오류. 성능 실패와 분리 |

### 12.2 필요한 영구 기록

```text
work_dir/ablr2/<sensor>/<server>/
  policy.json                    # lane/lease/stop/update policy
  component_catalog.json         # 17+선택 case 정의와 hash
  seed_ledger.jsonl              # phase/revision/sweep/stream별 seed
  recipe_ledger.jsonl            # 모든 제안/실행/채택/기각
  reference_ledger.jsonl         # Teacher/bundle/calibration 연결
  task_ledger.jsonl              # 모든 시도와 parent task
  comparison_graph.json
  thresholds_v1.json             # BOOT5 후 실측 값
  state.json                    # restart 지점
  leases/                       # 누적 사용시간과 갱신 이력
  runs/<run_id>/                 # config/fullstate/checkpoint/metrics
  reports/<recipe>/<wave>/       # 전체 표·paired 값·flow 상태
  reports/all_attempts.csv
  reports/complete_panel_metrics.csv
  reports/targeted_rechecks.csv
  reports/exploratory_best.csv
  reports/verification_results.csv
```

성공값으로 같은 cell/file을 덮는 대신 append-only 원기록을 두고, summary는 원기록에서 재생성한다. 좋은 행만 선택한 `exploratory_best`는 어떤 경우에도 `complete_panel_metrics`를 대신하지 않는다.

### 12.3 seed와 identity

BOOT5는 §5 표를 사용한다. 이후 seed는 성능과 무관한 hash로 결정한다.

```python
key = f"ABLR2|{sensor}|{phase}|{recipe_revision}|{wave}|{sweep}|{role}|{stream}|{master_seed}"
seed = int.from_bytes(sha256(key.encode()).digest()[:8], "big") % (2**31 - 1)
```

충돌은 ledger를 확인해 key에 정해진 증가 counter를 붙여 해결한다. 결과가 나쁘다는 이유로 hash 입력을 바꾸지 않는다. 같은 sweep 내 component명은 공통 U/data seed의 key에 넣지 않는다. 후보 간 paired screen에서는 recipe 문자열과 무관한 명시적 `paired_init_group`으로 동일 초기값을 강제한다.

Run ID는 `ABLR2_<sensor>_<server>_<recipe>_<phase/wave/sweep>_<case>_<seed>_FRESH50` 형식이다. Source/calibration/Teacher SHA와 실제 tensor hash는 별도로 저장한다.

### 12.4 controller 의사코드 — 실제 launch 명령이 아님

```python
# 아래 함수들은 구현자가 trainer/queue/ledger에 연결해야 하는 인터페이스다.
# 이 문서는 GPU training API나 원격 실행기를 구현/기동하지 않는다.

def lane_controller(sensor, server, state_store, policy):
    state = state_store.load_or_init(sensor=sensor, server=server)
    while not state.user_stop:
        lease = wait_for_operator_lease(server, policy.lease_hours)
        state = recover_inflight_from_fullstate(state)
        verify_sensor_binding(sensor, server)  # s1 WV3 / s2 QB만 허용

        if not state.boot5_complete:
            run_registered_full_wave(state, phase="BOOT5", recipe="R00", sweeps=5,
                                     new_teachers=True, lease=lease)
            if not complete_and_valid_wave(state, "BOOT5"):
                persist_and_pause_or_repair(state)
                continue
            state.thresholds = calibrate_once_from_boot5(state)
            save_immutable_thresholds(state)

        build_cycle_manifest(state)  # metric을 보기 전에 seed·case·순서 pin
        relation = choose_unresolved_relation(state, max_recheck_batches_per_recipe=1)
        if relation is not None:
            run_paired_recheck(state, relation, pairs=5, include_full_anchor=True,
                              teacher_pool="CURRENT_RECIPE_LATEST_BALANCED5",
                              lease=lease)
            # 첫 positive pair에서 break하지 않는다.

        candidates = choose_registered_untried_recipes(state, max_count=2)
        if candidates:
            run_matched_recipe_screen(state, candidates, screen_blocks=("P01", "P02"),
                                      include_relation_controls=True, lease=lease)
            state.incumbent = choose_by_full_absolute_quality(state)

        run_registered_full_wave(state, phase="REFRESH5", recipe=state.incumbent,
                                 sweeps=5, new_teachers=True, lease=lease)
        publish_all_completed_panels_and_attempts(state)
        notify_flow_reversal_plateau_or_candidate(state)  # 알림은 종료/승격이 아님
        if verification_requested_and_not_done_for_exact_recipe(state):
            run_locked_verify_wave(state, sweeps=5, new_teachers=True, lease=lease)
            publish_verification_without_replacing_negative_seeds(state)
        state.cycle += 1
        state_store.atomic_save(state)
```

실제 구현에서는 wave가 lease를 넘으면 같은 seed/case 목록을 유지해 다음 lease에서 이어간다. `run_registered_full_wave`가 함수 호출 한 번으로 무조건 다 끝난다고 가정하지 않는다. 중지 명령은 case/update 안전 경계에서도 확인한다.

### 12.5 오류 재실행과 성능 재시험을 구분한다

| 상황 | 처리 |
|---|---|
| 저장/네트워크/일시적 환경 오류 | 같은 config·seed의 fullstate에서 최대 2회 자동 복구. 세 번째는 알림·safe pause |
| 업로드 오류 | metrics 재업로드만. 학습 재실행 없음 |
| 실제 NaN/발산 | `DIVERGED` 기록. 동일 seed를 조용히 대체하지 않음. 재현되는 문제는 recipe/source revision 검토 |
| 낮은 HQNR/높은 ERGAS | `VALID_NEGATIVE_RESULT`. 정상 관측으로 저장. Technical retry에 넣지 않음 |
| 동일 resume | 새로운 독립 run이나 새 seed로 집계하지 않음 |

---

## 13. 자원·중지·재개

### 13.1 상시 실행의 범위

`max_campaign_cycles=null`은 전체 학습 횟수의 사전 상한이 없다는 설정이다. 무제한 disk/권한을 뜻하지 않는다. 운영자의 초기 기동 시각·허가 범위·자원 한도를 기록한다.

기본 설계의 local lease는 **72시간**이며 운영자가 갱신할 수 있다. Lease 만료 전에 다음 작업의 학습·평가·저장 예산이 들어오지 않으면 그 작업을 입장시키지 않는다. 현재 진행 중 작업은 계획된 안전 경계에서 저장·정지한다. Lease 재개로 누적 시간을 0으로 초기화하지 않는다.

이는 상시 엔진의 갱신형 자원 허가다. **실험 총시간을 72시간으로 제한하거나 첫 5회가 그 안에 끝난다고 보장하는 것이 아니다.** 운영자가 다른 lease 길이를 정하면 기동 전에 policy로 고정한다.

두 서버에 공통 성능 lock이나 Teacher 전송 의존성이 없다. 한 센서가 pause돼도 다른 센서는 자기 lease 안에서 진행할 수 있다.

### 13.2 비용을 정직하게 센다

첫 BOOT5만 각 센서 95개, 전체 190개 학습이다. 구성 5개를 학습하는 작업이 아니며, 하나의 학습 길이를 50K의 5배로 늘리는 작업도 아니다.

\[
T_{\rm BOOT5,sensor}\approx5\left(T_{T+}+T_{T0}+T_{cal+}+T_{cal0}
+\sum_{c=0}^{16}(T_{train,c}+T_{eval,c}+T_{save,c})\right).
\]

새 REFRESH5마다 같은 규모가 추가된다. RECHECK/HPO/VERIFY 비용은 별도로 더한다. 서버별 실제 GPU 모델·runtime과 첫 run의 실측을 사용하고, s1 WV3의 시간을 s2 QB에서 그대로 가져오지 않는다. 한 case의 training 시간을 전체 calibration/평가 시간으로 오인하지 않는다.

작업 예약은 최근 실측 train subprocess+validation+official eval+save+retry의 보수적 예측×1.15로 갱신한다. 허가 시간·메모리·disk를 넘으면 PAUSE이지 불리한 case 삭제가 아니다.

### 13.3 storage 정책

각 run의 resolved config, source/teacher/data hash, full-precision metrics, 선택 checkpoint, 최종 fullstate, paired receipt는 보존한다. 전체 candidate 중간 weights는 유리/불리한 결과와 무관한 **동일한 기한·공간 정책**으로만 prune한다.

그 정책은 기동 전에 승인받고, 별도 backup과 checksum이 확인되지 않은 유일한 재현 자산은 삭제하지 않는다. 디스크 압박 시 우선 신규 입장을 멈춘다.

### 13.4 중지

`STOP_AFTER_RUN`, `STOP_AFTER_SWEEP`, `PAUSE_AFTER_BLOCK`, `STOP_NOW_SAFE`를 구분한다. 학습 중 임의 kill이나 다른 campaign의 process 종료는 하지 않는다. 재개 시 기존 seed/순서/optimizer/scheduler/RNG/loader state를 복원한다.

s3–s5 GF2 계획, 기존 WV3 생산 reference, 기존 QB 대표 결과는 읽기 전용으로 유지한다. 새 ABLR2가 이들을 자동으로 대체하지 않는다.

---

## 14. 기록·표·그림 규칙

### 14.1 원본 기록 필드

`campaign, sensor, server, phase, recipe_revision, threshold_revision, wave, sweep, case, run_id, attempt, teacher_seed, student_seed, teacher_type/SHA, calibrationID, τR, qref, qcacheSHA, α/β/λE/λcon, A schedule, data/source/init/streamSHA, planned/actual updates, candidate grid, checkpoint rule, selected step, endpoint/VAL/RAW metric, validationE, train/cal/eval/save/retry시간, status, invalid_reason, test_aware, budget_lease`를 저장한다.

삭제한 모듈이 Teacher를 간접적으로 사용하는지 확인할 flags도 포함한다. 정확한 measurement가 없으면 null이며 0점으로 채우지 않는다.

### 14.2 표를 분리한다

| 파일/표 | 포함 | 포함하지 않는 것 |
|---|---|---|
| `all_attempts` | 기술 실패/발산/낮은 성능까지 모든 시도 | 결과 선별 |
| `complete_panel_metrics` | 같은 recipe의 완성 17행×사전 등록 sweep | 관계별 추가 재시험을 일부 행에만 합산 |
| `targeted_rechecks` | parent/child/anchor 추가 5 pair, Teacher별 차이 | 전체 matrix 평균인 것처럼 표시 |
| `exploratory_best` | 개발 중 찾은 최고값과 tuning 비용 | component 증명/최종 대표 |
| `verification_results` | 잠근 recipe의 새 5회 전체 결과 | 실패 seed를 교체한 wave |

### 14.3 집계

센서별 paired Δ와 case 평균±표본 SD, median, 개별 sweep 값을 모두 남긴다. Teacher 5×Student 1을 85개의 독립 repeat로 세지 않는다. 장면 20개도 학습 seed 20개가 아니다.

추가 재시험에서 Teacher를 재사용하면 Teacher 묶음별로 먼저 평균하고 동일 가중으로 모으거나 관계별 패널로 분리한다. 같은 Teacher에 Student가 많다는 이유로 전체 결과를 지배하게 하지 않는다.

s1 WV3와 s2 QB의 raw ERGAS를 한 평균으로 합치지 않는다. 센서와 서버가 완전히 결부되므로 센서 차이의 순수 원인성을 이 실험만으로 선언하지 않는다.

### 14.4 그림

누적표는 C00–C07 고정 순서, 제거표는 C07 대비 delta 고정 순서다. HQNR와 ERGAS는 별도 그림이며 같은 seed 집합과 checkpoint 규칙을 사용한다. 역전·작은 차이·오차 막대가 그대로 보여야 한다.

“예쁜 계층”이란 먼저 component의 역할이 논리적으로 구분된 구조를 뜻한다. 실제 숫자가 모든 칸에서 단조가 아니면 해당 조건부 효과나 상충을 설명한다. 출력값의 반올림·축·행 제거로 없는 단조 관계를 만들지 않는다.

---

## 15. 기동 전 구현 검증

| 검사 | 완료 조건 |
|---|---|
| Lane binding | s1/WV3·s2/QB 이외 학습 입장 거부 |
| FULL parity | 각 센서 C07이 기존 FULL 수식/forward/routing과 일치 |
| Sensor C | WV3 8/Q8/11 slots, QB 4/Q4/7 slots. 명시적 head·loss band mean |
| Teacher-free | C00/C01/C02/C10에서 Teacher I/O·clone·τ/q 참조 0 |
| TPLUS/TZERO | 초기 A/U·native sample/view prefix 동일, 각자 endpoint/calibration |
| Masked frequency | 학습·추론 mask 동일, 공통 P/MS weight·추가 zero-init |
| no-A/frozen-A | identity는 warp 없음. Frozen은 보정 유지하되 step/decay/buffer 갱신 없음 |
| Loss switch | C12 hard α만 0, C13 β만 0, C14 edge만 0. 제거 override가 공통 recipe 뒤 |
| q control | CONST_HALF와 TRAIN_MEAN 구분, train 평균 정의·hash 보존 |
| Sampling | component명/실행 순서가 U/data/corruption seed를 바꾸지 않음 |
| Result selection | VAL 규칙 고정, 각 case 동일 50 candidate, RAW 별도 |
| Pilot | 센서별 5 sweep×17구성. Partial에서 threshold 쓰기 거부 |
| Threshold | 산식·cap·freeze 검증. 개별 관계 성공용 튜닝 거부 |
| Retries | 낮은 성능은 technical retry 아님. 한 positive에서 batch 종료 금지 |
| Reference pool | 재시험은 현재 recipe의 5-sweep bundle. s1↔s2 전송 없음 |
| Resume | ledger 중복·crash·upload retry·lease 만료·STOP 안전 경계 |
| Publication | full panel과 best 단일 run·adaptive 재시험 패널 혼합 금지 |

이 문서의 pseudo-code/JSON은 **설계 계약**이다. 기존 G20/L100 validator를 지워 guard를 우회하지 않고, 별도 ABLR2 runner에서 등록한 ablation 제거를 허용하도록 구현해야 한다. 실제 trainer/controller 연결·GPU 회귀 검증·운영자 기동 전에는 실험이 시작된 것으로 표시하지 않는다.

---

## 16. 제공 파일과 적용 순서

| 파일 | 용도 |
|---|---|
| 본 MD | 센서 배치, 17 case, 5회 pilot, 내부/외부 loop, 기준, 실패·보고 규칙 |
| `ABLR2_BOOT5_190_TrainingCases_2026-09-21.csv` | 첫 두 센서 5회 전체 학습 190개의 run ID·seed·reference·순서 |
| `ABLR2_ComponentCatalog_2026-09-21.csv` | 기본 17+frequency 2+선택 mechanism 7. MASKED_CPLUS3로 일반화 |
| `ABLR2_ComparisonGraph_2026-09-21.csv` | 누적 7+제거 9+종합 1의 17관계 |
| `ABLR2_DesignRegistry_2026-09-21.json` | 설계 mode·lane·seed·recipe bank·반복 상한/무상한 구분 |
| `build_validation.json` | case 수·identity·Teacher-free·Teacher 우선 순서의 정적 검사 |

적용 순서는 **기존 작업 상태 확인 → 별도 ABLR2 구현/검증 → 두 센서 BOOT5 등록 → 사용자 lease 승인 후 기동**이다. 런처 경로가 실제로 있다고 가정한 가짜 쉘 명령은 제공하지 않는다.

최초 5회가 끝나면 센서별 `thresholds_v1`, 17관계 paired 표, 우선 재시험 relation, 다음 후보 recipe, 누적 compute를 산출한다. 그 다음부터는 해당 정책 안의 반복을 진행하고 사용자가 조정할 수 있도록 모든 상태를 보존한다.

---

## 17. 출처와 명시적 변경 이력

**[S1]** `PAN_QB_ABLATION_S12_ComponentPlan_2026-09-21.md` 및 `PAN_QB_ABLATION_ComponentCases_2026-09-21.csv`. C00–C16, optional F/X, FULL 수식, 기본 recipe bank, 동일 조건 대조 정의를 계승했다. 이전 문서의 “s1/s2 모두 QB”와 고정 DEV 2 block/최종 3 seed 구성은 이번 문서가 대체한다.

**[S2]** `PAN_WV3_Closeout_Review_FrozenRecipe_2026-09-20.md`, §7–8. WV3 P0/W112D123 Teacher·PLH/W104D122 Student, 50K, batch48, α/β/edge/LR 및 routing의 근거. 기존 WV3 종결 자산은 보존하고 이번에는 별도 ablation만 재개한다.

**[S3]** `PAN_QG40_QB_GF2_40H_ExperimentPlan_2026-09-20.md`, §4–5. QB msfix/C4/DN2047/Q4, 센서별 calibration과 native PAN 평가의 근거.

**[S4]** `2026-09-20_gf2-hqnr-gap-audit.md`, §0·§4.3·§7. 여러 설정을 한꺼번에 바꾸면 개별 효과를 분리하기 어렵다는 이전 프로젝트 관측. 그 문서의 GF2 결과를 WV3/QB component 효과로 전용하지 않았다.

**[U1] 최신 사용자 지정:** s1 WV3/s2 QB로 변경, 전체 알고리즘 약 5회 후 기준 설정, 예상과 다른 결과에 대해 지속적인 추가 시도를 자동화.

**본 v2의 새 설계:** BOOT5=Teacher도 새로 바꾸는 전체 19학습 단위, 5회 190학습 목록, 센서별 운영 허용폭, 한 relation의 5 pair 재시험, 최대 2후보 fitting, 전체 REFRESH5 상시 loop, 별도 VERIFY5, 자원 lease·append-only ledger.

**한계:** 기대한 계층적 우열이 실제로 나온다는 보장은 없다. 5회로 일반적 분산이나 통계적 유의성을 확정할 수 없다. 상시 모니터링과 test-aware 개발에서 선택된 결과는 그 선택 과정을 공개해야 한다. 이번 산출물은 실험 실행 결과가 아니다.
