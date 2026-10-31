# 2026-10-31 QB 전체 결과 검토 및 Student 구조 후속 큐

## 1. 결론과 근거 범위

QB의 현재 목표는 **같은 checkpoint에서 HQNR ≥ 0.9315 AND ERGAS ≤ 3.5**다.
최선은 여전히 s1 `QB_BASE R2 @26K`의 **HQNR 0.945383746 / ERGAS 3.509656381 / PSNR 38.449570371**이다.
HQNR에는 여유가 있지만 ERGAS가 약 0.009656, 목표 대비 0.276% 부족하다.
이 작은 차이는 seed·checkpoint 선택에 민감하므로, 단일 대표 수치의 소폭 차이를 확정적인 개선으로 해석하지 않는다.

이번 설계는 [10월 31일 원본 평가 분석](2026-10-31_experiment-progress-and-qb-results.md),
[10월 30일 정체 구간 분석](2026-10-30_qb-plateau-audit.md),
그 분석의 [local audit](assets/2026-10-31_qb-progress/audit.json)과
[paired comparison](assets/2026-10-31_qb-progress/comparison.json)을 바탕으로 한다.
10월 31일 집계의 로컬 4개 mature 캠페인에는 Student 30개·공식 평가 863개가 있으며 공동 목표 달성은 없다.
별도로 s3 자체 Stage 1 이후 Student 8개가 모두 완료됐으며 이들에도 공동 통과가 없다.
s3는 Sheet에 게시된 대표·설정·완료 기록을 근거로 하며, 이 설계 시점에 원격의 전체 trajectory와 scene별 평가를 직접 읽은 것은 아니다.

낮은 LR·짧은 cosine·Hard 가중치 축소를 더 촘촘히 반복하기보다,
**지금까지 QB에서 모두 함께 사용했던 correction residual 입력과 UBIL 업샘플 블록을 분리해서 확인**한다.
기존 결과에서 관심이 남은 **WD=0**은 서버별 자체 Stage 1에서 같은 seed의 대조군과 비교한다.
총 **s1 10개 + s3 8개 = 신규 Student 18개**, 모두 fresh 30K이며 전체 실행 시간 제한을 두지 않는다.

## 2. 지금까지의 결과가 지지하는 것과 지지하지 않는 것

### 2.1 Stage 1은 현재 A200_E030 / 반경 3 기준 유지

과거 QB에서 λ_abs·λ_epsilon 변경은 Student HQNR에 큰 차이를 만들었다.
예를 들어 `QB_A200_E003`은 대표 HQNR 약 .8249, `QB_A050_E003`은 약 .8429로 낮았다.
`A200_E030`에서는 s1의 대표 HQNR .9427/.9454가 나왔고,
독립적으로 학습한 s3의 같은 계수 조합에서도 CONTROL .9356/.9497을 얻었다.
이는 이 Stage 1 설정이 정상적인 Student HQNR을 재현할 수 있다는 근거다.
Teacher 자신의 낮은 FR HQNR을 Student의 실패로 취급해서는 안 된다.

현재 기준 주변의 다른 Stage 1 시도는 개선 근거가 약하다.
반경 4의 `QB_A200_E030_R4` 대표 ERGAS는 3.564075,
`QB_A100_E030_R3`는 3.567756이며,
`QB_A200_E010_R3`는 기존 HQNR 규칙으로 조기 종료됐다.
따라서 이번 큐에서 반경이나 Stage 1 loss 계수를 다시 넓게 바꾸지 않는다.
각 서버가 이미 자체 학습·검증 선택을 마친 A/T를 사용해 Student 변경의 영향을 분리한다.

### 2.2 낮은 LR과 짧은 cosine을 추가 탐색의 중심에 두지 않음

아래는 HQNR 통과 checkpoint 중 ERGAS 최소로 선택한 s1 대표값이다.

| 설정 | R1 ERGAS | R2 ERGAS | 판단 |
|---|---:|---:|---|
| 기존 C30 / LR=1e-4 | 3.514885 | 3.509656 | 기준 유지 |
| LR=8e-5 | 3.528125 | 3.528824 | 두 seed 악화 |
| LR=5e-5 | 3.551016 | 3.555931 | 두 seed 악화 |
| C25 / LR=1e-4 | 3.524364 | 3.524107 | 단축 효과 없음 |
| α=0, γ=0 | 3.519430 | 3.512524 | ERGAS 개선 없이 HQNR 여유 감소 |
| α=.25, γ=.25 | 3.514881 | 3.536086 | R1 사실상 동률, R2 악화 |

LR050은 고정 30K 비교에서도 RR 20장 중 R1 19장, R2 20장에서 ERGAS가 악화했다.
LR080도 16장/17장에서 악화했다. C25 역시 같은 25K 비교에서 두 seed 모두 불리하다.
반대로 과거 LR120·C40은 학습 GT L1을 더 낮췄지만 공식 RR ERGAS가 나빠졌다.
따라서 단순 학습 부족이나 cosine 종료 시점 하나로 현재 정체를 설명하기 어렵다.
독립적인 Student validation 복원 trajectory가 없어 이를 과적합이라고 확정하지는 않는다.

α·γ를 모두 없애면 30K HQNR이 기준보다 R1 −.006313, R2 −.012699 낮아졌다.
특히 R1은 D_s가 좋아져도 D_lambda가 나빠져 HQNR이 하락했다.
그러므로 모든 HQNR 변화가 공간 정합의 변화라는 해석도 맞지 않는다.
이번 신규 구조 비교에서는 기존 α=.25, γ=.5를 유지한다.

### 2.3 용량·Hard 계수·weight decay 결과는 반복 확인이 필요

W128, α=.125, H050 및 γ 축소에서는 작은 이득이 seed마다 반복되지 않았다.
W128 대표는 R1 3.511541 / R2 3.511738로 서로 가깝지만,
각 seed의 기준 및 같은 30K와 비교하면 일관적인 우위는 아니다.
이번에는 폭 확대와 기존 loss 계수 추가 세분화를 뒤로 둔다.

s3의 동일 donor·동일 seed 대표값 비교는 다음과 같다.

| 변경 | ΔERGAS R1 / R2 | ΔHQNR R1 / R2 | 판단 |
|---|---|---|---|
| WD=.01 → 0 | +.003385 / −.014136 | +.009359 / +.002751 | 재확인 후보 |
| WD=.01 → .1 | +.018710 / −.022318 | +.005962 / −.010609 | seed에 따라 방향 반전 |
| α=.25 → .75 | +.020486 / −.018985 | +.010024 / −.000367 | seed에 따라 방향 반전 |

WD0도 ERGAS 개선이 확정된 것은 아니다.
다만 두 seed 모두 HQNR이 높아졌고 R1의 ERGAS 손해가 비교적 작아,
다른 WD·α 후보보다 우선해서 추가 paired 비교를 진행할 이유가 있다.

## 3. 아직 확인하지 않은 QB의 2×2 Student 비교

로컬 `work_dir/panda*/**/QB/runs/**/config.json`을 읽어 확인한 Stage 2 설정 **32개는 모두 CORRECTED_DIFF + UBIL**이었다.
이 수에는 현재 mature 분석 범위 밖의 초기 SENSOR5 두 run도 들어간다.
s3 이번 8개도 같은 입력·업샘플 조합이다.
WV3에서 관련 구조를 시험한 전력은 있지만, 그것을 QB에서 동일 donor·동일 seed로 수행한 비교로 대신할 수 없다.

| Student 입력 / 업샘플 | UBIL | BASE |
|---|---|---|
| CORRECTED_DIFF: `[PANc, PANc−PAN, I_lrms]` | 기존 각 서버의 R1/R2 대조군 | **신규 QB4_DIFFBASE** |
| CORRECTED: `[PANc, I_lrms]` | **신규 QB4_P0UBIL** | **신규 QB4_P0BASE** |

`PANc = Lanczos4(PAN, u)`이며 Aligner scale은 모든 경우 **1**이다.
`I_lrms`는 기존 복원용 interp23 LRMS base다.
QB의 Aligner 입력 Stored LMS 경로와 원본/보정 MS 데이터 경로는 이번 비교에서 변경하지 않는다.
P0는 corrected PAN만을 사용하는 입력 변형을 가리키며, 원본 PAN만 넣거나 Aligner를 끄는 실험이 아니다.

동일 s1 A/T의 Teacher는 RR ERGAS 3.492595, FR HQNR .759511이고,
Student BASE R2는 3.509656 / .945384다.
같은 Aligner의 RR·FR shift는 동일하지만 Teacher와 Student의 입력·구조·학습 기간·loss 가중 방식은 다르다.
따라서 Teacher의 RR 이득을 DIFF 제거 또는 BASE 업샘플 하나의 효과로 단정할 수 없다.
이번 2×2는 그중 **입력과 업샘플 블록 두 요인을 분리**하기 위한 것이다.
CORRECTED-only가 HQNR을 크게 잃을 가능성도 결과로 받아들인다.

### 3.1 초기화와 파라미터 차이

- CORRECTED는 5채널, CORRECTED_DIFF는 6채널이다. W112 첫 conv의 추가 파라미터는 **1,008개**다.
- 기존 builder는 DIFF 채널의 첫 conv 가중치를 **0으로 초기화**하고 PAN/MS 공통 열과 bias를 복사한다. 같은 sampling·seed에서는 DIFF 유무가 초기 예측 함수를 바꾸지 않도록 비교할 수 있다. 학습 이후에는 두 함수가 달라질 수 있다.
- BASE 업샘플은 `2×2 ConvTranspose, stride 2`, UBIL은 `bilinear ×2, align_corners=False + 3×3 Conv`다. 두 업샘플 위치를 함께 바꾸면 W112에서 **125,440개** 파라미터 차이가 생긴다.
- BASE CORRECTED는 2,650,820개, UBIL CORRECTED는 2,776,260개이며, DIFF를 쓰면 각각 1,008개가 추가된다. 업샘플 변경은 약 4.7%의 파라미터 차이도 포함한다.
- 공유 이름·shape 레이어는 기존 named-seed 초기화를 유지하지만 바뀐 업샘플 kernel은 shape가 달라 초기화도 다르다. 따라서 결과는 **업샘플 블록 변경의 효과**이며 순수 interpolation 효과나 회전 대칭성의 효과로 해석하지 않는다. 다운샘플은 두 변형 모두 기존 stride-2 3×3 conv를 유지한다.

## 4. Stage 1과 대조군의 출처

두 서버가 하나의 Stage 1 checkpoint를 공유하도록 강제하지 않는다.
각 서버가 이미 학습한 **자체 A/T 쌍**을 이어 사용한다.

| 서버 | 자체 Stage 1 출처 | 관측된 선택본 | 새 Student에 대한 의미 |
|---|---|---|---|
| s1 | `PANDA_S13_S1LOSS9_RGB_L4_STAGE12_20261026_v1`, A200_E030 | 44K | 기존 s1 구조·WD 비교의 공통 입력과 Teacher guidance 유지 |
| s3 | `PANDA_S3_QB_DEEPEN_OWNSTAGE1_20261030_v1`, A200_E030, seed 261016301 | 39K | s3가 독립 학습한 Stage 1을 계속 사용 |

s3는 원격 파일의 실제 검증 선택 receipt를 읽고 checkpoint를 연결한다.
현재 게시된 39K를 s1의 경로나 별도 전달 파일로 대체하지 않으며,
공통 checkpoint를 요구하는 제한이나 이전 캠페인 종료 영수증을 새 실행의 조건으로 추가하지 않는다.
두 서버 모두 λ_abs=2, λ_epsilon=.03, epsilon 반경 3 조건이지만 Stage 1 seed·선택본·calibration은 서로 다르다.
새로운 Stage 1을 이 Student 비교와 동시에 도입하지 않아 donor 변경을 추가 요인으로 만들지 않는다.

기존 R1/R2 DIFF+UBIL 대조군은 다음과 같다.

| 서버 | R1 대조군 | R2 대조군 |
|---|---|---|
| s1 | S1LOSS9 `QB_A200_E030 R1` | SENSOR_REFINE `QB_BASE R2` |
| s3 | QB_DEEPEN `QB3_CONTROL R1` | QB_DEEPEN `QB3_CONTROL R2` |

기존 대조군을 재사용하려면 새 구현과 기존 구현에서 동일 donor와 calibration,
모델 초기화, Student seed, 데이터 순서, augmentation, optimizer, loss mask,
평가 및 checkpoint 의미가 동등함을 확인해야 한다.
단순히 run 이름이 비슷하거나 대표 성능이 가까운 것을 동등성 근거로 쓰지 않는다.
공통 경로의 CPU 검증과 실제 저장 설정·stream seed·builder state 비교가 이 확인의 대상이다.
동등성이 보존되면 기존 R1/R2 대조군을 동일 설정으로 다시 실행하지 않는다.

실제 s1 과거 R1/R2 및 REFIT의 calibration을 비교한 결과, τ_e=.010525926947593689,
모집단·물리적 A/T·데이터 binding, edge cutpoint 4개 및 q ECDF 39,488,256개 값이 정확히 일치했다.
신규 캠페인의 full-training calibration도 계산 후 해당 서버의 과거 R1/R2와 자동 비교해
`calibration/numeric_parity_*.json`에 남긴다. 경로·캠페인 provenance를 바꾸어 동일한 통계로 가장하지 않는다.
과거 비교 자료가 없거나 수치가 다르면 학습은 계속하되 historical 비교를 미검증으로 표시한다.
그 경우 차이를 해결하거나 새 대조군을 실행하기 전에는 엄밀한 paired 효과로 해석하지 않는다.

R1/R2 Student seed는 기존 261018101/261018102이며,
추가 R3는 두 서버 모두 261018103으로 CONTROL과 WD000을 짝지어 실행한다.
각 서버 내부 비교를 우선하고, 다른 Stage 1을 사용한 서버 간 수치를 단일 변수의 효과처럼 직접 계산하지 않는다.

## 5. 신규 전체 큐

공통: **30K fresh Student, warmup 100 + cosine→0, LR=1e-4, W112/D123,
α=.25, β=0, γ=.5, Aligner scale 1, RGB reference, A/T/S Lanczos4**.
WD는 기본 .01이고 `QB4_WD000`만 0이다.
추가 loss, soft loss, rotation consistency, TTA, SWA/EMA, 밴드 가중 loss를 도입하지 않는다.
기존 paired 입력 augmentation은 이전 대조군과 동일하게 유지한다.

### 5.1 s1: 10 run

| 순서 | case | replica | 입력 | 업샘플 | WD | 질문 |
|---:|---|---|---|---|---:|---|
| 1 | QB4_DIFFBASE | R1 | CORRECTED_DIFF | BASE | .01 | UBIL 블록 영향 |
| 2 | QB4_DIFFBASE | R2 | CORRECTED_DIFF | BASE | .01 | 같은 비교 반복 |
| 3 | QB4_P0UBIL | R1 | CORRECTED | UBIL | .01 | DIFF 채널 영향 |
| 4 | QB4_P0UBIL | R2 | CORRECTED | UBIL | .01 | 같은 비교 반복 |
| 5 | QB4_P0BASE | R1 | CORRECTED | BASE | .01 | 입력×업샘플 상호작용 |
| 6 | QB4_P0BASE | R2 | CORRECTED | BASE | .01 | 같은 비교 반복 |
| 7 | QB4_WD000 | R1 | CORRECTED_DIFF | UBIL | 0 | s1 donor에서 WD0 첫 확인 |
| 8 | QB4_WD000 | R2 | CORRECTED_DIFF | UBIL | 0 | 같은 비교 반복 |
| 9 | QB4_CONTROL | R3 | CORRECTED_DIFF | UBIL | .01 | 추가 seed 대조군 |
| 10 | QB4_WD000 | R3 | CORRECTED_DIFF | UBIL | 0 | R3 paired WD 비교 |

### 5.2 s3: 8 run

| 순서 | case | replica | 입력 | 업샘플 | WD | 질문 |
|---:|---|---|---|---|---:|---|
| 1 | QB4_P0UBIL | R1 | CORRECTED | UBIL | .01 | DIFF 채널 영향 |
| 2 | QB4_P0UBIL | R2 | CORRECTED | UBIL | .01 | 같은 비교 반복 |
| 3 | QB4_DIFFBASE | R1 | CORRECTED_DIFF | BASE | .01 | UBIL 블록 영향 |
| 4 | QB4_DIFFBASE | R2 | CORRECTED_DIFF | BASE | .01 | 같은 비교 반복 |
| 5 | QB4_P0BASE | R1 | CORRECTED | BASE | .01 | 입력×업샘플 상호작용 |
| 6 | QB4_P0BASE | R2 | CORRECTED | BASE | .01 | 같은 비교 반복 |
| 7 | QB4_CONTROL | R3 | CORRECTED_DIFF | UBIL | .01 | 추가 seed 대조군 |
| 8 | QB4_WD000 | R3 | CORRECTED_DIFF | UBIL | 0 | 기존 R1/R2 이후 WD0 재확인 |

s3의 WD0 R1/R2는 이미 완료됐으므로 재등록하지 않는다.
서버별 구조 시험 순서를 달리해 입력·업샘플 두 가설에 대한 초기 결과를 함께 얻는다.
시간 제한 없이 각 등록 run을 완료하되, 중간 결과를 사용자와 검토한 후 다음 큐를 갱신할 수 있다.
아직 결과가 없는 구조와 WD0를 결합한 추가 grid는 이번 큐에 선제적으로 늘리지 않는다.

## 6. 판정·게시·weight 보존

공식 대표 선정은 기존 규칙을 유지한다.

1. HQNR ≥ .9315인 checkpoint가 있으면 그중 ERGAS가 가장 낮은 결과를 대표로 게시한다.
2. HQNR 기준을 통과한 checkpoint가 없으면 HQNR이 가장 높은 결과를 대표로 게시한다.
3. 동일 checkpoint에서 H/E 공동 목표를 충족한 결과는 모두 기존 규칙에 따라 함께 게시한다.
4. `QB-ablations`에 서버·case·replica·입력·업샘플·LR·WD·α/β/γ·scale·자체 donor 출처를 알 수 있게 기록한다.
5. 기존과 같이 공식 평가 checkpoint와 weight·sidecar를 보존하고, 게시 metric은 같은 상태의 RR/FR 및 실제 저장 파일과 연결한다.

기존 QB Stage 2 조기 종료 규칙인 **8K·9K·10K의 HQNR이 모두 .90 미만일 때 안전 checkpoint를 남기고 중단**을 유지한다.
1K HQNR만으로 탈락시키지 않는다. 이 규칙을 Stage 1 Teacher에 적용하지 않는다.

분석은 대표 하나 외에도 다음을 함께 본다.

- 같은 seed·같은 donor의 고정 30K와 25–30K 평균 ERGAS/HQNR.
- RR 20장 paired ERGAS 변화, D_s와 D_lambda 분해, 목표 공동 통과 여부.
- DIFF 효과를 UBIL·BASE 각각에서, 업샘플 효과를 DIFF·CORRECTED 각각에서 비교한 뒤 상호작용 확인.
- R1/R2의 방향 및 WD 비교 R3의 반복성. 단일 seed .00x 차이는 확정 효과로 보지 않음.

서로 다른 대표 시점의 최소값 비교와 고정 step 비교를 구분한다.
반복적으로 테스트 결과를 참고하는 현 단계는 탐색이며 독립 최종 검증과 동일시하지 않는다.

## 7. 기존 실행과의 연결

s1은 **현재 진행 중인 WV3 run을 원래 horizon까지 완료 → 신규 QB 10개 → 변경하지 않은 잔여 WV3 TUNE30/HARD36 재개** 순서다.
WV3의 등록 case·하이퍼파라미터·완료 이력을 삭제하거나 이번 QB 가설 때문에 변경하지 않는다.
s3는 완료된 자체 Stage 1과 QB_DEEPEN 결과를 바탕으로 새 QB 8개를 진행한다.
서버 간 donor 전달이나 상대 서버 완료 대기는 필요하지 않다.

이 문서는 큐의 과학적 근거와 등록 계획이다. 실제 시작·진행 상태는 launcher와 campaign state로 별도 확인한다.

## 8. 구현·검증 및 실제 인계 상태

2026-10-31 **15:06 KST** 기준으로 s1 실행 전환을 완료했다.
당시 진행 중이던 `S100DIFFUBILC30LR120 R2`를 7,039 step에서 optimizer·scheduler·RNG·sampler를 포함해
안전 저장한 뒤, 동일 설정과 동일 학습 상태로 해당 run만 30K까지 이어가는 컨테이너로 재개했다.
재개 후 실제 train log의 7,100→7,101 step 증가 및 GPU 프로세스를 확인했다.
새 priority watcher 상태는 `FINISH_CURRENT`이며, QB를 이미 시작한 것으로 기록하지 않는다.

- 실행 순서: 현재 WV3 R2 완료 → 신규 s1 QB 10개 → 기존 잔여 WV3 TUNE30 → HARD36.
- 기존 학습 소스 525개 SHA256은 이전 실행 기록과 모두 동일했다. WV3 설정·등록 큐를 수정하지 않았다.
- 패키지 검증 24개가 통과했다. 실제 optimizer update/완전 resume, 두 서버의 기존 engine 대비 한 step 정확 일치,
  input 초기 함수 일치, 네 모델 graph, 자체 Stage 1 출처, Sheet 경로 및 실행기 인계를 포함한다.
- 로컬 donor 검증과 실제 QB 데이터 read-only check도 통과했다. CUDA cost profiling/실제 데이터 preflight는
  현재 WV3 완료 후 새 QB 실행기가 수행하므로 아직 완료된 것으로 주장하지 않는다.
- s3는 코드·등록 큐·명령을 준비했다. 이 세션에는 원격 실행 권한 경로가 없으므로 실제 시작은 s3 에이전트가
  Git pull 후 수행한다. 사용자 push 이후 사용할 명령과 상태 파일은 [패키지 README](../panda_qb_structure/README.md)에 있다.
