# PANDA s4 WV3 — 신규 Direct donor 3회 + 고정 donor Stage 2, 40시간

작성일: 2026-10-17. 실행 레지스트리: `pandas4/plan.py`.
이번 요청의 범위는 **case 설계·구현·검증·커밋**이다. 현재 서버나 s4에서
학습을 자동 시작하지 않는다. 실행은 s4에서 명시적으로 launcher를 호출한다.

## 1. 목적과 선행 결과

s4에서 기존 **Direct PAN Aligner**와 Teacher를 처음부터 세 번 학습하고,
그중 하나의 A/T 체크포인트를 고정하여 Stage 2의 GT 대조, difficulty weighting,
selective soft KD, conditional-q 및 가중치 민감도를 비교한다. 새 S2D 구조는
s1에서 검증 중이므로 이 캠페인에 섞지 않는다.

2026-10-17 설계 시점에 확인한 s1의 **완료 실행**은 다음과 같다. 아래 수치는
각 실행의 최대 HQNR 시점에 함께 측정된 ERGAS이며, 서로 다른 시점의 최댓값과
최솟값을 조합한 것이 아니다.

| 근거 실행 | 선택 시점 | HQNR | 같은 시점 ERGAS | 해석 |
| --- | --- | --- | --- | --- |
| donor24 A1/R1 | 7,000 | 0.960203446 | 2.376434176 | HQNR gate 통과 경험이 있으나 ERGAS가 좋지는 않음 |
| donor24 A3/R1 | 13,000 | 0.956406218 | 2.196241649 | gate 미달, 더 낮은 ERGAS 쪽의 donor 후보 |
| panda40 Direct A1/R1 | 6,000 | 0.956963348 | 2.488913461 | A1의 gate 통과가 안정적으로 재현된 것은 아님 |

근거 파일은 해당 캠페인의 `campaign_state.json`과 실행별
`evaluation_0007000.json`, `evaluation_0013000.json`,
`evaluation_0006000.json`이다. 원본은
`work_dir/donor24/PANDA_WV3_DONOR_S13_24H_20261016_v1/s1/WV3/` 및
`work_dir/panda40/PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1/s1/WV3/`에 있다.

따라서 A1을 확정 최적이라고 주장하지 않는다. **A1 두 seed + A3 한 seed**로
A1의 불안정성과 HQNR–ERGAS 절충을 함께 확인한다. 이는 동일 조건의 3-seed
검증이 아니라 **서로 다른 두 설정에서 총 세 donor 실행**이다. A1과 A3는
absolute와 epsilon 가중치를 함께 바꾸므로 이 둘의 차이를 단일 loss 항의
인과 효과라고 해석하지 않는다.

## 2. 공통 고정 조건

| 항목 | 설정 |
| --- | --- |
| 서버·데이터 | s4, WV3; 등록된 RR train만 학습에 사용, FR train 없음 |
| Aligner | 기존 Direct PAN Aligner, native LMS reference, GT 입력 없음 |
| Teacher / Student | W112, D(1,2,3), P0, BICUBIC_MS residual base |
| 복원기 입력 | aligned PAN + MS, **LPAN/HPAN 미사용** |
| 초기화 | donor의 A/T와 각 Stage 2 Student 모두 fresh; Teacher 복사 초기화 없음 |
| 학습량 | 각 실행 H50 = 50,000 optimizer updates |
| optimizer | AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 0.01 |
| LR | 복원기 1e-4; Stage 1 Aligner 1e-5; Stage 2 A/T는 동결 |
| scheduler | warmup 100 + cosine, minimum LR 0 |
| 배치·정밀도 | effective 48 = microbatch 12 × accumulation 4, FP32, AMP/TF32 off |
| 반복 seed | R1 = 261016101, R2 = 261016102 |
| 독립 stream | A/T/S 초기화, RR 표본, augmentation, epsilon을 label별 분리 |
| 공식 평가 | 동일 model state의 원본 reference RR20 + FR20, 기존 metric 구현 유지 |
| 필수 로그 | HQNR, SCC, ERGAS; Hard/Soft 및 support·weight·gradient 진단 |

Stage 1의 reconstruction loss는 기존 MASK2다. Stage 2도 MASK2가 주 비교이며,
FULL은 별도의 명시적 대조군이다. **공식 평가는 loss mask를 적용하지 않는다.**
기존 RR metric의 프로토콜 crop은 유지하며, aligned PAN으로 원본 PAN reference를
바꾸거나 shift에 따라 평가 영역을 추가로 잘라내지 않는다.

## 3. donor 획득: 정확히 세 번의 신규 학습

| 순서 | case / replica | λrec | λabs | λepsilon | Aligner LR | 설명 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | D_A1 / R1 | 1 | 2 | 0.003 | 1e-5 | HQNR 통과 경험이 있는 조건 |
| 2 | D_A1 / R2 | 1 | 2 | 0.003 | 1e-5 | 같은 조건의 독립 seed |
| 3 | D_A3 / R1 | 1 | 3 | 0.0015 | 1e-5 | 낮은 ERGAS 쪽의 절충 후보 |

세 실행 모두 기존 Direct 구조, MASK2, H50이다. epsilon은 매 update,
**current-PAN pixel 단위 radius 3의 area-uniform disk**이며 FR에서 ×4하지
않는다. s1/s3의 파일을 donor 후보에 섞거나 기존 s4 모델을 찾아 학습을 생략하지
않는다. 새 캠페인의 세 실행이 모두 완료되어야 donor를 확정한다.

### donor 선택과 고정

각 완료 실행에서 실제 저장된 중간 checkpoint까지 포함하여 다음 순서로 고른다.

1. HQNR ≥ **0.959** 후보가 있으면 **ERGAS 최소 → HQNR 최대 → 이른 step**.
2. 모두 미달이면 **HQNR 최대 → ERGAS 최소 → 이른 step**을 고르고
   `exploratory`로 명시하여 Stage 2를 진행한다.
3. SCC는 기록·분석용이며 선택 또는 동률 해소에 쓰지 않는다.

ERGAS 상한은 donor의 필수 통과 조건이 아니다. 따라서 HQNR gate를 통과했어도
ERGAS가 나쁜 donor가 선택될 수 있다. HQNR ≥ 0.959와 ERGAS ≤ 2.040의
동시 목표 달성 여부는 별도로 보고하며, gate 통과를 동시 목표 달성으로 쓰지 않는다.

선택한 파일의 SHA, A/T identity, 평가 증거를 확인하고 **RR20·FR20을 재실행**해
동일한 결과인지 검증한다. 이후 `fixed_donor.json`을 작성하며 모든 Stage 2
case·seed는 이 **단일 파일의 동일 A/T**를 사용한다. donor 선택 seed와 Student
학습 seed를 각각 기록한다. Stage 2 결과를 본 뒤 donor를 바꾸지 않는다.

## 4. Stage 2 case와 우선순위

기준 Q는 `(α, β, γ) = (0.5, 0.05, 0.5)`다. 값은 사전 등록한 탐색 가설이며
최적값으로 주장하지 않는다. Student 초기화·데이터·augmentation seed를
case 간 맞추어 같은 R1끼리, 같은 R2끼리 비교한다.

| 순서 | case | variant | α | β | γ | loss 영역 | 반복 / 성격 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | G | gt_only | 0 | 0 | 0 | MASK2 | R1 → R2, 필수 기준 |
| 2 | Q | panda_condq | 0.5 | 0.05 | 0.5 | MASK2 | R1 → R2, 필수 주 방법 |
| 3 | E | panda_e | 0.5 | 0 | 0 | MASK2 | R1, Hard difficulty 증분 |
| 4 | ES | panda_e | 0.5 | 0.05 | 0 | MASK2 | R1, selective soft 증분 |
| 5 | Q_ALOW | panda_condq | 0.25 | 0.05 | 0.5 | MASK2 | R1, α만 감소 |
| 6 | Q_BHIGH | panda_condq | 0.5 | 0.1 | 0.5 | MASK2 | R1, β만 증가 |
| 7 | Q_GHIGH | panda_condq | 0.5 | 0.05 | 1 | MASK2 | R1, γ만 증가 |
| 8 | Q_FULL | panda_condq | 0.5 | 0.05 | 0.5 | 전체 픽셀 | R1, mask/calibration 대조 |

총 Stage 2 최대 10회, donor 포함 최대 13회다. 먼저 G의 두 seed를 끝내고 Q의
두 seed를 진행한다. 뒤의 E/ES 및 민감도 case는 남은 예산에 따라 순서대로
승인한다. 뒤쪽 네 case는 명시적 예비이며, 시간 부족 시 건너뛴다.
seed/가중치 승패를 보고 큐를 재정렬하거나 성능이 낮다는 이유로 중도 종료하지 않는다.

### Loss의 정확한 의미

밴드 평균 절대 오차를 `eS = mean_band(|S-GT|)`, `eT = mean_band(|T-GT|)`라 한다.

```text
d_e = stopgrad(eT / (eT + tau_e))
a_T = stopgrad(max(eS-eT, 0) / (eS + 1e-6))
w_H = 1 + alpha*d_e + gamma*d_e*r_q_bar
L_H = mean_domain(w_H * eS)
L_S = mean_domain((1-d_e) * a_T * mean_band(|S-T|))
L = L_H + beta*L_S
```

G는 Teacher forward/calibration 없이 GT-only다. E는 difficulty에 따른 Hard
증분, ES는 selective soft KD 추가, Q는 conditional-q Hard 증분이다. `w_H≥1`로
GT supervision은 항상 유지한다. Stage 2의 별도 output spatial loss나 epsilon
loss는 없다. Aligner와 Teacher에는 gradient를 흘리지 않고 Student만 최적화한다.

MASK2는 Stage 1과 동일하게 detach한 bicubic warp support를 사용한다.
Hard와 Soft를 각각 **샘플별 유효 픽셀 평균 → batch 평균**으로 줄인다.
ROI8 Scharr stencil의 support 검사는 유지하고 위반하면 실패로 처리한다.
q contribution은 fixed ROI8 밖에서 0이다.

calibration은 선택된 frozen donor의 **전체 RR train, augmentation 없는 출력**으로
구축한다. K=5 conditional empirical CDF와 정확한 median을 사용하며 validation,
RR test, FR test를 calibration 표본으로 넣지 않는다. MASK2의 `tau_e`는 유효
train 픽셀, FULL의 `tau_e`는 전체 train 픽셀에서 각각 구한다. 두 정책의 artifact를
섞지 않는다. **Q 대 Q_FULL은 mask뿐 아니라 그 mask에 일관된 tau_e도 함께 바꾸는
프로토콜 대조**이며, tau_e까지 고정한 단일 변수 비교가 아니다.

## 5. 40시간 예산과 자동 실행 경계

40시간은 s4 캠페인의 최초 시작부터 계산하는 총 경과시간이다. donor 학습,
GPU cost 측정, preflight, donor 재평가, calibration, Stage 2 학습, 공식 평가,
checkpoint 저장을 모두 포함하며 재시작해도 시계를 초기화하지 않는다.
마지막 **1,800초**는 closeout으로 남기고 추정에 **1.2 safety factor**를 적용한다.

s1 Direct H50의 관측시간은 약 **2.56시간**이고 세 donor의 단순 합은 약
**7.7시간**이다. 이는 s4 실행시간 보장이 아니다. s4 GPU가 미확인이고 새 Stage 2의
실제 학습 비용도 아직 측정되지 않았으므로, 최대 13회 완료를 약속하지 않는다.

예산 검사는 두 단계다.

- donor 시작 전: Direct disposable pilot을 실측한다. donor 세 회와 G/Q
  두 seed씩 총 네 anchor가 들어가는지 먼저 확인한다. Student 시간이 아직
  없으므로 이 단계의 Student estimate는 Direct 실측의 **2배**를 사용하는
  보수적 임시값이며 calibration은 최소 600초를 배정한다. 여기에 safety factor와
  closeout을 반영한다. 넣을 수 없다면 donor만 실행하고 끝내는 계획으로 조용히
  축소하지 않고 예산 부족 상태를 드러낸다.
- donor 선택 후: 실제 G/Q disposable pilot과 calibration으로 Stage 2 시간을
  갱신하고 네 anchor가 남은 예산에 맞는지 다시 확인한다. 이후 각 그룹/단일
  실행을 시작하기 전에 필요한 H50·평가·저장 시간을 재검사한다.

G와 Q의 2-seed 그룹은 남은 두 실행을 모두 끝낼 추정 예산이 있을 때 승인한다.
단일 seed 예비는 각각 승인한다. 시작 후에도 실제 deadline/사용자 중단은 우선하므로
불가피한 부분 실행은 `STOPPED`와 중단 checkpoint로 남긴다. `SKIPPED_BUDGET`이나
부분 실행을 완료된 H50 결과 또는 정상 2-seed 비교로 표시하지 않는다.

## 6. 평가, 모델 저장, 시트, cost

Stage 2 checkpoint도 donor와 같은 **HQNR gate → ERGAS** 규칙을 사용한다.
best-HQNR backup을 함께 저장하고 선택에 SCC를 사용하지 않는다. 모델 파일은
해당 시점의 동일-state RR20/FR20 증거, file SHA, optimizer/stream/RNG와 연결한다.
마지막/중단 모델은 resume 목적과 성능 선별 목적을 구분한다.

시트는 기존 표시 규칙을 유지한다. **HQNR≥0.960 및 ERGAS≤2.045를 동시에
충족한 모든 평가 step**, 없으면 run별 best-HQNR fallback을 표시한다. 모델 선택의
0.959 gate와 시트의 joint-quality 표시를 혼동하지 않는다. 모든 공식 평가와
미표시 결과는 로컬 JSON/outbox에 남긴다.

각 평가 직후 로컬 immutable outbox를 만들고 별도 publisher가 기존
`WV3-ablations`에 반영한다. 네트워크 실패 시 학습과 분리해 재시도하며,
Result_ID readback을 확인한다. s4 campaign/run namespace만 소유하고 기존 s1/s3
또는 과거 캠페인 행을 정렬·삭제·덮어쓰지 않는다. Run display에는 case,
`a/b/q` 또는 `a/eps/lrA`, MASK2/FULL, replica를 짧게 표기한다.

GPU cost는 Stage 1 **A+warp+T**, Stage 2 **A+warp+S**의 실제 deployment graph로
측정한다. Stage 2의 training-only Teacher, loss, calibration은 추론 cost에서
제외한다. CPU 측정을 GPU latency로 기입하지 않으며 동일 서버의 실제 구조·코드
signature가 일치하는 측정만 재사용한다. 학습/평가 시간도 실행 로그의 실측치와
예산 추정치를 구분한다.

## 7. 결과 해석 기준

- G 대 Q는 2-seed paired 비교다. seed별 HQNR·ERGAS·SCC와 선택 step을 먼저
  보이고 평균/범위를 덧붙인다. 두 seed를 통계적 확정이라고 과장하지 않는다.
- E/ES와 가중치 민감도는 R1 단일 seed 탐색이다. Q/R1을 공통 기준으로 비교한다.
  이 결과에서 찾은 승자를 같은 자료의 다중 탐색 효과 없이 확정 최적이라고 쓰지 않는다.
- checkpoint 선택 결과뿐 아니라 시간별 HQNR–ERGAS 추이, Hard/Soft 성분,
  `a_T`, `w_H`, q rank, 유효 support 비율과 학습 비용을 함께 검토한다.
- 최고 HQNR만 개선되거나 ERGAS만 개선된 경우 동시 목표 달성으로 쓰지 않는다.
  donor 자체가 탐색용이면 Stage 2 결론에도 그 한계를 명시한다.
- s3는 별도 donor를 선택하므로 s3 대 s4의 절대값 차이는 순수한 가중치 효과가
  아니다. 먼저 **동일 s4 donor 내부 비교**를 해석한다.
- RR/FR 공식 평가를 checkpoint 선택에 사용하는 test-aware 탐색이다. 동일 평가
  자료로 선택한 숫자를 미사용 held-out 최종 일반화 성능이라고 주장하지 않는다.

## 8. s4 전달 및 실행

구현은 새 `pandas4/`와 `scripts/run_pandas4*.sh`로 추가한다. 실행 중인 s1의
`panda40/` 및 그 code-pin 대상 파일을 변경하지 않는다. 기존 순수 Stage 1/Stage 2
엔진은 재사용하되 s4의 config, provenance, reporting, cost와 runtime identity는
별도 namespace로 검증한다.

사용자가 commit을 push한 뒤 s4 저장소 루트에서 실행한다.

```bash
git pull --ff-only
bash scripts/run_pandas4_docker.sh
```

server는 s4로 고정되어 별도 인자가 없다. **pull만으로 시작되지 않는다.** launcher가
읽기 전용 데이터/환경 확인, 세 donor, donor 고정, calibration, Stage 2,
평가·저장·publisher를 연결한다. 수동 donor 선택이나 별도 calibration 단계는 없다.
기존 job을 자동 종료하지 않으므로 s4 GPU에 다른 작업이 있으면 먼저 상태를 확인하고
사용자가 그 작업의 종료 여부를 결정해야 한다.

데이터와 인증은 s4의 로컬 자산이며 Git에 넣지 않는다. 산출물은 s4의
`work_dir/pandas4/<campaign>/s4/WV3/` 아래에 보존한다. 이번 커밋 단계에서는
실제 s4 donor 생성, calibration 완료, GPU training 성공 또는 업로드 완료를
미리 주장하지 않는다.

## 9. 구현 검증 기록 (2026-10-17)

- 신규 테스트 52개 통과: 전체 smoke 51개 통과 후 donor replay의 요약 metric
  변조 거부 테스트를 추가하고 관련 13개 provenance/policy 테스트를 재실행했다.
- 기존 panda40 계획·시간 제한 및 Stage 2 선택 규칙 회귀 테스트 15개 통과.
- 실제 WV3 등록 데이터의 읽기 전용 검사 통과: RR train 9,714개 및
  train/val/RR/FR 경로, evaluator/source pin 확인.
- 실제 s4 설정을 이용한 CPU 합성 통합 검사: Direct effective-batch 48 update,
  complete checkpoint 저장/재로딩, 같은 A/T의 Stage 2 import, MASK2/FULL 각각의
  calibration, 전체 8종 Stage 2 생성, Q의 effective-batch 48 update 및
  optimizer/model/sampler 복원 확인. 비용의 GPU 측정만 이 테스트에서 대체했다.
- 모든 실제 case의 학생 초기화·동결 donor binding·loss 계수·mask·배포 cost
  signature를 확인했다. 합성 metric은 TEST로 구분되어 실제 업로드가 거부된다.
- publisher의 타 서버 행 보존, concurrent append, 실패 재시도·readback,
  CPU 측정의 GPU cost 등록 거부, Torch 없는 host 실행 경로를 검증했다.
- 실행 중 s1의 code pin **151개 파일 모두 동일**하고 기존 tracked 파일 변경은 없다.
  테스트는 CPU/임시 디렉터리에서 수행했으며 실제 Sheets 쓰기나 신규 학습 실행은 하지 않았다.

s4의 GPU·전체 train calibration 실측과 실제 업로드 권한은 s4에서 실행해야
확인할 수 있다. 따라서 이 검증을 H50 학습 성공이나 모든 예비 case의 40시간 내
완료 보장으로 해석하지 않는다.
