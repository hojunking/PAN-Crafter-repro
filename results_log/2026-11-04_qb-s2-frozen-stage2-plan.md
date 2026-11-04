# s2 QB 후속 설계 — 기존 Aligner/Teacher 고정, Stage 2만 학습

사용자 요청: 새 PAN Aligner 학습 없이, 나온 s2 QB 결과를 바탕으로 다양한 Stage 2 case를 구성한다.
이 문서는 승인된 설계안이다. 후속 실행 패키지는 [s2 실행 README](../panda_qb_s2_frozen/README.md)에 구현했다. 기존 큐·Sheet와 원격 학습은 이 서버에서 변경하지 않았다.

## 1. 확인된 결과와 범위

2026-11-04 `QB-ablations`의 PQS2 행과 원본 정밀도의 row notes를 읽었다.
현재 게시상 확인되는 것은 아래 **4개 donor+Student 조합**이며, 각 Student는 30K 완료 및 평가 30개로 기록되어 있다.
원래 등록한 나머지 A3/E.03/D101, A4/E.06/D101 결과는 해당 검색 범위에서 확인되지 않았다.
이를 완료·실패로 간주하지 않으며, s2 프로세스/원격 파일 실재 여부를 직접 점검한 것은 아니다.

목표: **동일 checkpoint에서 HQNR ≥ 0.9315, ERGAS ≤ 3.500**.

| 기존 s2 donor 설정 | 선택 Stage1 step | Student 대표 step | HQNR | ERGAS | HQNR 충족 checkpoint / 30 |
|---|---:|---:|---:|---:|---:|
| A2 / E.03 / D101 | 44K | 30K | 0.946 | 3.520 | 20 |
| A4 / E.03 / D101 | 48K | 25K | 0.930 | 3.520 | 0 |
| A2 / E.03 / D301 | 48K | 26K | 0.943 | 3.517 | 20 |
| **A4 / E.03 / D301** | **48K** | **26K** | **0.945** | **3.515** | **20** |

여기서 A2/A4는 Stage1 λ_spa(코드 λ_abs)=2/4, E.03은 λ_epsilon=.03,
D101/D301은 Stage1 seed 261016101/261016301이다. 위 Student seed는 모두 261018101이다.
표시는 3자리이며 판정은 원본 수치 기준이다. 동시 통과는 네 Student 모두 0개다.

**분석**

- A4/D301은 HQNR 기준을 유지하면서 현재 s2에서 가장 낮은 적격 ERGAS를 냈다.
- 다만 A2/D301 대비 차이는 **3.515128041 − 3.516678813 = −0.001550773**에 불과하다.
  같은 Student seed 한 개의 대표값 차이이므로 A4 donor의 우월성으로 확정하지 않는다.
- A4/D101은 30개 checkpoint 전체에서 HQNR 기준을 충족하지 못했다. 같은 λ_spa라도
  Stage1 seed/선택된 donor에 따라 결과가 달라지므로 이번 주 donor로 삼지 않는다.
- A3/E.03 및 A4/E.06의 미게시 결과를 기다리는 조건은 새 Stage2 설계에 넣지 않는다.
  나중에 결과가 오더라도 이미 고정한 donor를 자동 교체하지 않는다.

## 2. 가장 근거 있는 Stage2 후보

이번 live Sheet에서 s3 `QB_REMAINING QB5_W128H050`의 **두 seed 완료 결과**를 추가 확인했다.
일부 과거 로컬 계획에는 미실행으로 남아 있지만 최신 게시 근거가 이를 갱신한다.

| s3 W128H050 | 대표 step | HQNR | ERGAS |
|---|---:|---:|---:|
| R1 / S261018101 | 25K | 0.940 | 3.504 |
| R2 / S261018102 | 22K | 0.932 | 3.504 |

정확한 ERGAS는 3.504243238 / 3.504496072다. R2 HQNR은 .932190742로 기준에 가까우므로
ERGAS만 보고 안정적인 목표 달성을 기대하지 않는다.

이 설정은 **Student W128, α=.125, γ=.25**다.
H050은 `w = 1 + α*d_e + γ*d_e*r_q`의 두 추가 계수를 절반으로 줄이는 뜻이며,
loss 전체에 0.5를 곱하거나 α를 0.5로 설정하는 뜻이 아니다.

s3 결과의 donor는 과거 A2/E.03/D301 **39K**이며 현재 s2 A2/D301 **48K**와 다른 weights다.
하드웨어/실행 환경 및 evaluator identity도 동일하다고 가정하지 않는다.
따라서 이는 s2로 옮겨 확인할 후보 근거이며 s2에서의 개선량을 보장하는 증거가 아니다.

## 3. 기존 s2 donor 두 개만 사용

| 별칭 | 고정 donor | 역할 |
|---|---|---|
| **D4** | `QB6_S1_A400_E030_D301`, 검증 선택 48K | 주 탐색 |
| **D2** | `QB6_S1_A200_E030_D301`, 검증 선택 48K | 조합의 donor 의존성 확인 |

**Stage1 학습 0회.** 두 donor 각각의 **동일 checkpoint Aligner와 paired Teacher를 모두 고정**한다.
Stage1 source campaign/run/seed/selection receipt를 그대로 보존하며, 다른 서버 weights는 가져오지 않는다.
어떤 donor 설정을 후속에 쓸지는 기존 Student test 결과를 참고한 탐색 결정이다.
각 donor 내부의 checkpoint는 이미 validation으로 선택된 것을 그대로 사용하며 test로 다시 고르지 않는다.

원본 source campaign: `PANDA_S2_QB_STAGE1_BALANCE_20261102_v1`.

- D4 checkpoint SHA: `30883a100962a94b004db9ba8758ed6bffcd84eba7d2c4bcd6e7815c71891127`
- D2 checkpoint SHA: `b07e72474e676e9bf1bb54eb4096096cf5111cfefe976093f93248a8a647eb2c`

전체 checkpoint 경로, model/selector receipt/calibration SHA, 원본 Student baseline 근거는
[proposal.json](assets/2026-11-04_qb-s2-stage2-plan/proposal.json)의 `donors`에 고정했다.
이는 Sheet의 verified artifact metadata를 보존한 것이며, 실행 전에 s2에서 파일 SHA를 직접 확인해야 한다.
파일이 없다는 이유로 Stage1을 다시 학습하는 fallback은 만들지 않는다.

## 4. 8개 recipe, 각 Student seed 3개

모든 case는 **fresh Student**, cosine30K, LR1e-4다. 완료 Student를 이어 학습하는 방식은 쓰지 않는다.

| Case | Donor | Student 폭 | α | γ | 역할 / 우선순위 |
|---|---|---:|---:|---:|---|
| Q0_BASE | D4 | 112 | .250 | .500 | 기준; R1 기존 결과 재사용 |
| Q1_H050 | D4 | 112 | .125 | .250 | Hard 추가 가중치 절반 효과 |
| Q2_W128 | D4 | 128 | .250 | .500 | 폭만 늘린 효과 |
| **Q3_W128H050** | **D4** | **128** | **.125** | **.250** | **최우선: 유력 조합** |
| Q4_A0125 | D4 | 112 | .125 | .500 | 후순위: α만 낮춰 H050 원인 분리 |
| Q5_G025 | D4 | 112 | .250 | .250 | 후순위: γ만 낮춰 H050 원인 분리 |
| Q6_D2_BASE | D2 | 112 | .250 | .500 | 두 번째 donor 기준; R1 재사용 |
| **Q7_D2_W128H050** | **D2** | **128** | **.125** | **.250** | **최우선: 조합의 donor 간 전이 확인** |

Q0/Q1/Q2/Q3은 D4에서 **폭 112/128 × Hard 기본/절반**의 2×2 비교다.
Q0/Q1/Q4/Q5는 α/γ 조합의 2×2 비교다.
D2의 Q6/Q7은 조합의 전이 확인만 하므로 D2에서도 폭/Hard 각각의 효과를 분리했다고 주장하지 않는다.

- Student seeds: R1=261018101, R2=261018102, R3=261018103.
- 같은 반복 내 데이터/augmentation/Student 초기화 seed 파생 규칙을 맞춘다.
- 폭이 달라지는 경우 동일 seed라도 초기 함수가 완전히 같아지는 것은 아니다.
- Q0/Q6 R1은 이미 완료된 동일 recipe 결과를 보존·참조하며 재학습하지 않는다.
- **24개 비교 = 기존 baseline 2개 + 신규 Student 22개**.
- 그중 핵심 6개 recipe(Q0/1/2/3/6/7)는 **신규 16회**, α/γ 분해(Q4/5)는 **후순위 6회**다.

## 5. 실행 순서

한 case를 세 번 연속 실행하지 않고 **R1 전체 → R2 전체 → R3 전체**로 순회한다.

- R1: Q0/Q6은 기존 기록 연결 → **Q3 → Q7 → Q1 → Q2 → Q4 → Q5** (신규 6회).
- R2: Q0 → Q6 → Q3 → Q7 → Q1 → Q2 → Q4 → Q5 (8회).
- R3: 같은 순서 (8회).

첫 Q3/Q7부터 기존에 가장 가까웠던 조합을 두 frozen donor에서 확인할 수 있다.
중간 분석 후 큐를 바꿀 경우 변경 시점·근거를 새 revision으로 기록한다.
현재 s2에서 별도로 실행 중인 job의 존재/완료 여부는 이번 원격 상태 조회로 확인하지 못했으므로,
구현/실행 단계에서 확인하여 이중 실행을 피한다. 기존 6-case 전체 완료를 불필요한 선행 조건으로 넣지 않는다.

## 6. 공통 방법과 유지할 조건

- Frozen Aligner scale **1.0**, paired Teacher 고정, β=0.
- 입력 `[corrected PAN, corrected PAN - original PAN, interp23 I_lrms]`, **6채널 → QB 4밴드**.
- Student D123/UBIL, Teacher W112/BASE. W128에서도 Teacher를 바꾸지 않는다.
- Teacher/Student 둘 다 **Lanczos-4 reflection**. 현재 WV3의 Teacher-bicubic 설정을 복사하지 않는다.
- q 참조는 RGB `[2,1,0]` 산술평균. Aligner 입력 MS는 **original stored LMS 4밴드**다.
- reconstruction base는 등록된 **msfix LRMS의 interp23**. train/val original LMS override를 유지한다.
- AdamW, WD=.01, betas .9/.999, eps1e-8, batch48/micro12, warmup100 → cosine0, FP32.
- 기존 paired augmentation 유지. 회전 일관성 loss/TTA/SWA/새 band loss 없음.
- 같은 donor/geometry/train inventory에 대한 **기존 exact calibration NPZ를 SHA 검증 후 재사용**한다.
  Student 폭·α/γ·LR·seed는 해당 통계 identity의 변수가 아니다.
- 공식 RR20/FR20은 원래 좌표·참조로 평가한다. target/output warp나 평가 지표 변경을 하지 않는다.
- Student 8·9·10K HQNR이 모두 .90 미만일 때 기존 조기 중단을 적용한다.
- 실제 support 오류를 shift clamp나 adaptive ROI로 덮지 않는다.

기존 QB에서 근거가 약했던 C25, C40/C60, 큰 LR 감소, B96, WD0, DIFF 제거, UBIL 제거는 이번 큐에서 제외했다.
LR/스케줄을 다시 보더라도 이번 width/Hard 결과를 확인한 뒤 좁은 범위로 분리하는 편이 낫다.

## 7. 판단 기준과 비용

대표 최소 ERGAS 하나만 비교하지 않고 아래를 함께 기록한다.

1. 동일 Student seed의 **고정30K 결과 차이**.
2. 25–30K의 6개 평가 평균, HQNR 충족 checkpoint 수, 공식 대표 및 joint-pass 여부.
3. seed별 paired 차이 및 평균/표준편차. 세 seed만으로 유의성을 확정하지 않는다.
4. RR/FR 장면별 차이로 소수 장면이 전체 개선을 만드는지 점검하되 공식 scene을 제외하지 않는다.

다른 seed/checkpoint의 HQNR과 ERGAS를 합쳐 목표 달성이라고 보고하지 않는다.
정확한 목표 판정은 full precision이고 표시는 3자리다.

현재 s2의 실제 완료 C30 baseline은 Wall **5.493924528h**다.
다른 행의 25K/26K Wall은 선택 checkpoint까지의 비용이므로 전체 run 시간으로 사용하지 않았다.
W128 속도는 미측정이며 임시로 1.3배를 가정한다.

| 신규 작업 | 횟수 | 1회 임시 시간 | 합계 |
|---|---:|---:|---:|
| W112 C30 | 13 | 5.5h | 71.5h |
| W128 C30 | 9 | 7.15h, 미측정 | 64.35h |
| 전체 | **22** | | **135.85h** |

현재 s2 기준 잠정 **135–145시간**, 첫 W128 완료 후 갱신한다. 고정 시간 제한이나 중간 자동 마감은 없다.
모든 1K 평가 weight와 complete resume state를 보존한다.

## 8. 구현에 필요한 경계

현재 `panda_qb_s2`는 원래 Stage1→Student 12-job 등록만 허용한다.
따라서 이 proposal JSON/CSV를 기존 실행기에 넣는 것만으로 실행되지 않는다.
별도 `panda_qb_s2_frozen` consumer/registry와 원본 s2 donor import를 구현했다. 실제 실행에는 새 패키지의 launcher를 사용한다.

- source campaign/선택 receipt/A+T checkpoint/calibration identity를 그대로 보존한다.
- 새 consumer의 forward/loss/gradient/data/sampler/seed/평가 정책이 기존과 같은지 확인한 뒤 baseline R1을 연결한다.
- 기존 baseline을 새로 학습한 결과라고 재표기하거나 Sheet에 중복 게시하지 않는다.
- 새 namespace로 기록하고 매 1K weights·optimizer/RNG/stream을 보관한다.
- 새 run은 `QB-ablations`, server=s2, experiment에 donor/폭/α/γ/seed를 구분한다.
- 기존 동시 통과 전부 + case별 대표 규칙을 유지한다. Sheet 값은 3자리, 원본은 notes/로컬에 보존한다.
- s1 WV3 main, s3 ablation, 기존 s2 결과/weights는 이 설계에서 변경하지 않는다.

## 자료

- [설계 JSON](assets/2026-11-04_qb-s2-stage2-plan/proposal.json)
- [신규 22-run 순서 CSV](assets/2026-11-04_qb-s2-stage2-plan/new_runs.csv)
- [기존 baseline 포함 24개 비교 CSV](assets/2026-11-04_qb-s2-stage2-plan/comparison_grid.csv)
- [확인한 Sheet 행과 원본 notes](assets/2026-11-04_qb-s2-stage2-plan/sheet_snapshot.json)
- 이전 근거: [s2 6-case 설계](2026-11-02_qb-s2-30h-plan.md), [A4 결과 분석](2026-11-02_qb-a4-results-and-s2-direction.md), [QB40 후속 분석](2026-11-01_qb-results-and-40h-followup-plan.md).

원격 학습은 s2에서 새 실행 README의 명령을 실행할 때 시작한다. 원래 proposal JSON은 설계 당시 기록이며, 실제 실행 등록은 `config/panda_qb_s2_frozen/plan.json`과 실행 CSV다.
