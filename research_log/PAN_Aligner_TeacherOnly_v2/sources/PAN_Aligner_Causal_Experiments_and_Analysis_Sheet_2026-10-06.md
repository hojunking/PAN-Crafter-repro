# PAN Aligner 원인 분리·개선 실험 — 즉시 실행 인계 명세

**Campaign:** `PANDA_ALIGNER_CAUSAL_20261006_v1`  
**작성:** 2026-10-06  
**결과 원장:** `pan-cvpr27`의 `analysis` 탭  
**문서의 의미:** 새 실험의 실행 명세다. 원격 학습을 실제로 시작했다는 기록이 아니다. 기존 checkpoint·paper 대표·ablation 결과·운영 코드는 이 문서로 덮어쓰지 않는다.

## 0. 실행 담당자가 먼저 읽을 요약

1. 아래 계획은 **33개 case 종류**다. `D`는 기존 모델 진단, `L`은 학습 가능성 검사, `T`는 Teacher 연장학습 대조, `X`는 조건부 크기/스케일 학습, `F/S`는 fresh Teacher–Student 확인이다. case 수와 실제 학습 run 수를 구분한다.
2. 먼저 **D00–D07 → L00–L04 → T00–T10**을 실행한다. 최종 PS 지표만이 아니라 epsilon 반응, 장면 의존성, 참조 위상, gradient를 함께 수집한다. 모든 case가 같은 데이터셋에서 같은 고정 대조와 비교되도록 한다.
3. **Aligner와 reconstruction U의 아키텍처는 변경하지 않는다.** correlation volume·flow head·attention·새 encoder는 추가하지 않는다. 참조 전처리·loss 경로를 바꾸는 case는 `METHOD_REVISION`을 명시한다.
4. s1–s5는 독립적으로 진행한다. 공유 selection lock·MSTAR 전달·다른 서버 완료 대기는 없다. 같은 서버의 비교 묶음만 기다린다. 실제 실행 중 프로세스와 source를 확인하고 안전한 handoff 후 시작한다.
5. **`analysis` 업로드가 실험 완료의 일부**다. 결과 JSON을 서버에만 두고 ‘완료’로 표시하지 않는다. 이 세션의 분석자는 서버에 직접 접근하지 않고 시트의 숫자·검정 결과만으로 판단할 수 있어야 한다.
6. 새로운 native 결과는 **동일 checkpoint에서 RR/FR를 함께 평가**한다. FR quality는 원래 native PAN/LMS full-frame 기준이다. diagnostic crop·blur·mask·reference 변경을 공식 HQNR에 섞지 않는다.
7. 이번 원인 비교의 기준점은 고정된 `EXACT_LOCAL_BUDGET`이다. fresh Student의 별도 성능 보고는 **HQNR_MAX50와 EXACT_FINAL 둘 다** 남긴다. 과거 HQNR 선택 요청을 ERGAS 선택으로 되돌리는 것이 아니다.
8. 이번 보고서의 GF2는 **G20/R0**다. 최종 paper의 **P40/B09/R4_100/G050**는 별도 cohort로 추가 진단한다. 두 결과를 같은 Teacher/seed 실험처럼 합치지 않는다.
9. 디스크 정리와 동시에 실행하지 않는다. 기존 대표·Teacher·부모·후보·outbox·full-state는 보호한다. 현재 운영 상태는 시트의 완료행만으로 추정하지 않는다.

---

## 1. 출발점: 관측 사실과 이번 검정 가설

### 1.1 첨부 보고서에서 확인된 사실 [S1]

- `c=(dy,dx)`는 **현재 입력 PAN 픽셀 단위**다. `W(P,c)[y,x]=P[y+c_y,x+c_x]`; 영상 내용 이동과 sampling displacement의 부호가 반대다. FR에서 코드가 보정을 1/4로 줄이는 것은 아니다.
- Aligner는 dual stem → stride2 conv 두 번 → residual → GAP → MLP → global 2-vector 구조다. 512를 64로 resize하는 경로가 없다. wrapper가 외곽4px를 빼므로 64/256/512 입력은 내부56/248/504를 본다.
- Student epsilon response gain: WV3 train/RR/FR=`0.043/0.038/0.029`; QB=`0.043/0.042/0.020`; GF2 G20=`0.199/0.204/0.140`. Teacher도 유사하다.
- AXIS16의 상수 예측기 MAE는 **0.46875px**다. 이는 축4방향×반경{0.25,0.5,1,2}에서 성분별 절대값 평균이다. 다른 probe 분포의 상수 기준으로 그대로 사용하지 않는다.
- RR의 `bicubic(MS)↔GT`에는 약(-0.5,-0.5) sampling phase가 있지만, native LMS↔GT는 거의0이다. upstream 원인 전체나 FR의 동일 위상 정답까지 확정된 것은 아니다.
- 기존 epsilon branch는 Aligner만 학습하며, synthetic PAN을 U에 넣지 않는다. Student는 Teacher A를 clone하지만 Teacher epsilon loss를 계속 적용하지 않는다.
- 기존 보고서의 alpha=0/1/4는 **추론 개입 배율**이다. hard loss의 alpha와 혼동하지 않도록 이번 문서는 `kappa_shift`로 부른다.

### 1.2 가설 — 관측 사실로 미리 단정하지 않는다

| ID | 가설 | 분리하는 case |
|---|---|---|
| H1 | 현재 이득의 상당 부분이 sample-adaptive 정합이 아닌 공통 bias다 | D02 |
| H2 | A가 MS의 상대 위치보다 PAN boundary/절대 texture에 의존한다 | D03, L04 |
| H3 | 큰 입력의 정규화/GAP/content 혼합이 반응을 약화시킨다 | D04, X01 |
| H4 | Aligner 참조의 보간 위상이 reconstruction 목표와 충돌한다 | D05, T07/T08 |
| H5 | epsilon gradient가 너무 약하거나 reconstruction gradient와 충돌한다 | D07, L01–L03, T00–T06 |
| H6 | epsilon를 복원 목표에 연결해야 실효성 있는 정합이 된다 | T09/T10 |
| H7 | 같은 GSD의 입력 크기 일반화와 같은 FOV의 스케일 전이는 별개다 | D04 vs D06, X02/X03 |
| H8 | Teacher가 개선되어도 Student fitting에서 반응을 잃을 수 있다 | F/S, S03 |

사전 edge registration을 변위 GT로 사용하지 않는다. 과거의 사전 edge-shift 학습 실패와 상수 수렴 경험은 사용자 제공 맥락이며, 이번 실험은 그 수치를 새로 측정한 것으로 표현하지 않는다.

---

## 2. 대상 모델·서버·운영

### 2.1 동결 anchor

| Cohort | Student / 선택점 | 정확한 donor | 범위 |
|---|---|---|---|
| WV3_F1 | FH20R1, PLH/W104D122, SS73101, step38380 | F1/P0/W112D123, exact50K | 보고서 및 paper anchor |
| QB_TBC3 | QG40, PLH/W104D122, SS82001, step47470 | TB_C3/P0/W112D123, exact50K | 보고서 및 paper anchor |
| GF2_G20_R0 | G20, PLH/W104D122, SS93002, step45450 | R0가 가리키는 QGBASE GF2 TA bytes | 보고서 anchor; 전역 최고 아님 |
| GF2_P40_R4 | P40 s4 B09, local20K/lifetime120K | R4_100/P0/W112D123, exact100K | 실제 paper 대표로의 별도 전이 검정 |

전체 run ID와 hash는 `checkpoint_anchors.csv`에 있다. QB/G20 donor SHA는 이 문서에서 추정하지 않았다. 원 `resolved config`와 `*_teacher.json` 또는 실제 reference manifest에서 확정하고 ASSET 행에 업로드한다. P40의 lambda0도 원 config에서 읽는다.

**Student와 donor를 교차 대체하지 않는다.** 특히 GF2 G20의 R0를 R4_100으로 바꾸거나 WV3 F1 대신 해당 서버의 다른 Teacher를 선택하는 fallback은 금지다. 자산 미확인 case는 `BLOCKED_ASSET`로 기록하고 해당 branch만 멈춘다.

### 2.2 서버 배정 — 현재 가동 상태를 나타내는 표가 아님

| 서버 | Dataset/cohort | 첫 역할 | 추가 역할 |
|---|---|---|---|
| s1 | WV3_F1 | D/L/T 기전 진단 | local recipe 고정 후 fresh 확인 |
| s2 | QB_TBC3 | D/L/T 기전 진단 | 같은 서버에서 새 seed 확인2회 |
| s3 | GF2_G20_R0 | 보고서 GF2를 동일 모델로 검정 | fresh native50K 확인 |
| s4 | WV3_F1 | 다른 pilot stream으로 독립 반복 | 크기/스케일 대조 및 fresh 확인 |
| s5 | GF2_P40_R4 | 현재 paper B09에도 저반응이 있는지 먼저 확인 | R4 donor 기전 진단; 생산 경로 확인은 조건부 |

서버 역할은 이번 계획의 배정이다. 2026-10-06 live workbook에는 RB-M12/practical-validation 등 후속 작업도 존재한다. 따라서 과거 G23가 아직 실행 중이라고 가정하거나 과거 stop 명령을 무조건 호출하지 않는다.

### 2.3 즉시 전환·보존 절차

1. 실제 PID/cwd/container/GPU UUID/queue/controller/outbox/현재 run을 조회한다.
2. 이 계획으로 바꿀 **해당 서버의 후속 admission**을 막는다. 실제 활성 runner가 제공하는 안전 중단만 사용한다.
3. 현재 update 경계에서 A/U·optimizer·scheduler·Python/NumPy/Torch/CUDA RNG·sampler cursor·평가 ledger를 저장하고 `PAUSED_FOR_ALIGNER_ANALYSIS`로 남긴다. 지원되지 않는 경우 current-run 안전 경계까지 기다린 사유를 기록한다.
4. 기존 watchdog이 같은 GPU에 다시 학습을 시작하지 않는지 확인한다. 광범위한 pkill/컨테이너 삭제/강제 kill을 사용하지 않는다.
5. 새 namespace `work_dir/aligner_causal/<campaign>/<server>/`와 source snapshot을 만든다. dirty worktree와 과거 평가 출력은 건드리지 않는다.
6. 자산·단위검사 후 D00부터 시작한다. **시트에는 STATUS=RUNNING, 실제 PID/GPU/active_release/actual_updates를 기록한 뒤에만 실행 중으로 표시**한다.
7. 다른 서버의 상태는 시작 조건이 아니다. 임시 자산 복제는 시작 전에 완료하며, 특정 서버의 실험 결과를 기다려 참조를 고르지 않는다.

새 executor/runner CLI는 구현·검증 대상이다. 존재하지 않는 `tools/aligner_causal_runner.py start` 명령을 이미 실행 가능한 것으로 문서에서 가정하지 않는다. 기존 분석 스크립트는 보고서 §11의 위치를 읽어 재사용하되 원본 파일을 직접 수정하지 않는다.

---

## 3. 고정 데이터·probe·통계 규약

### 3.1 표본 집합

| 집합 | 샘플·용도 | 규칙 |
|---|---|---|
| LEGACY20 | 기존 보고서의 train20/RR20/FR20 재현 | 원 인덱스·band·LP·mask·CPU FP32 규약 유지 |
| CAL_TRAIN | 학습 split에서3072 patch, 고정 seed26100601 | 상수 보정·phase 통계·reference calibration용; sample 수 부족 시 전량, 중복추출 금지 |
| PROBE_TRAIN | 학습128 patch, seed26100602 | 고정 epsilon 응답·학습 curve; 훈련 성능으로 표시 |
| PROBE_VAL | 기존 validation128 patch, seed26100603 | pilot promotion용; 공식 test에서 대체 금지 |
| GRAD32 | 학습32 mini-batch×48 patch, seed26100604 | 같은 batch에서 rec/off 비교; sampled indices 보존 |
| RR20 | 기존 RR20 전 장면 | GT 있음; 물리 변위 GT는 아님 |
| FR20 | 기존 MAT20-converted FR20 전 장면 | GT 없음; test-aware 기록 유지 |

QB train/val은 anchor가 사용한 msfix와 split 계약을 따라야 한다. WV3/QB DN2047, GF2 DN1023, 밴드 수8/4/4. 데이터·LP·GT를 이 실험의 편의를 위해 새로 만들거나 고치지 않는다. 예외는 명시적 X diagnostic 변환뿐이다.

### 3.2 변위 probe

- `AXIS16`: 축4방향×{0.25,0.5,1,2}px. 보고서와 동일한 주 probe.
- `DIAG16`: 대각4방향×같은 반경. 각 성분이 아니라 **L2 반경**이 r이 되도록 r/sqrt(2)를 사용한다.
- `DISK64_HELD`: 면적균등 반경2 원판의 고정64개 추가 검증 offset. 학습 epsilon RNG와 별도 seed26100605.
- `OOD_AXIS`: 축4방향×{4,8}px는 별도 진단. in-range 결과와 합쳐 gain을 계산하지 않는다.
- no-offset 항목은 native correction을 기록하지만 gain 분모에 넣지 않는다.
- gain은 전체 하나와 반경별·축별로 모두 보고한다. 큰 epsilon가 합계 gain을 지배할 수 있으므로 pooled gain 하나만으로 작은 shift 반응을 평가하지 않는다.

### 3.3 좌표·support·resampling

`W(P,c)(x)=P(x+c)`를 전 과정에서 고정한다. PAN-only에서 예상 response는 `-epsilon`, MS-reference-only는 `+epsilon`, joint translation은 relative response0이다. MS-only/joint는 **Aligner 진단에만** 사용하고 native PS reconstruction의 MS/GT를 이동시키지 않는다.

서로 다른 두 프로토콜을 별도 행으로 남긴다.
- `LEGACY_M4`: wrapper margin4, 기존 report와 비교.
- `CLEAN_COMMON`: 각 비교쌍의 유효 support **교집합**과 사전에 고정한 내부 crop. fractional warp, 반경, cumulative warp의4tap footprint를 고려한다. 모든 variant에 같은 영역을 사용한다. 경계 영역은 별도 `BORDER_ONLY`로 진단한다.

공식 HQNR/RR의 support는 변경하지 않는다. clean-support 진단이 full-frame HQNR를 대체하지 않는다. 내부 영역에서 벗어나는 shift나 유효 pixel 부족은 제외 개수·비율과 이유를 올린다. 반사/복제 padding이 큰 shift 추정의 shortcut이 되는지 D03에서 확인한다.

추가 합성 shift와 보정은 가능한 한 **원본 PAN/LPAN에 합성 sampling shift로 한 번** 적용한다. `W(W(P,epsilon),c)`와 `W(P,epsilon+c)`는 이산 bicubic 영상에서 같지 않으므로 UNIT에서 차이와 support를 측정하고 평가용 경로를 고정한다.

### 3.4 비교·불확실성

- 동결 개입은 같은 checkpoint/장면의 paired 차이를 쓴다. 이 결과를 ‘no-aligner를 재학습한 효과’라고 쓰지 않는다.
- RR20/FR20은 지리적 대응 집합이 아니다. 같은 번호의 RR/FR 차이를 paired sample로 만들지 않는다.
- crop·epsilon·밴드는 독립 scene 표본이 아니다. scene bootstrap은 원 parent_scene_ID 단위로2000회, seed26100606. shuffle 반복은 19개 독립 학습으로 세지 않는다.
- 여러 training seed의 mean/std와 scene bootstrap CI는 분리한다. n=2 fresh seed는 유의성 보장이 아니라 예비 재현 확인이다.
- core native 평가를 동일 runtime에서2회 실행해 반복 오차 바닥을 측정한다. CPU 보고서와 GPU 재평가의 차이는 별도 `RUNTIME_PARITY`로 남긴다. 원 보고서 숫자를 새 GPU run의 대조군으로 직접 빼지 않는다.

---

## 4. D00–D07: 학습 전에 끝낼 원인 분리

### D00 — 자산·단위검사

필수 검사는 (1)정확한 Student/donor/LP/data hash, (2)64/256/512 ramp sign 및 단위, (3)zero warp와 bypass 오차, (4)shared PAN/LPAN warp와 HP subtraction, (5)A/U gradient 분리, (6)epsilon label과 parameter ordering dy/dx, (7)모든 sampler offset/reduction, (8)정규화 stats 범위와 current input 크기, (9)native full-frame evaluator, (10)readback round trip이다.

`SOURCE_MATCH`와 `GPU_BITWISE_MATCH`는 다른 상태다. 주요 source hash가 같아도 라이브러리·precision·augmentation·sampler까지 같다고 자동 인증하지 않는다. teacher SHA가 미확정이면 그 cohort의 T/L/F 학습을 시작하지 않는다.

### D01 — 기존 Teacher/Student anchor 재현

선택 Student 및 정확한 donor를 train/RR/FR에서 검사한다. 기존 run의 명명 선택점이 실제로 남아 있으면 native-c/gain/품질의 시간 변화를 추가한다. 없는 checkpoint는 `NOT_AVAILABLE`로 남기고 결과 CSV로 가중치를 복구했다고 하지 않는다.

보고서값, 과거 공식값, 이번 재추론값은 `Evidence_kind=REPORT_REFERENCE / HISTORICAL_OFFICIAL / NEW_MEASUREMENT`로 분리한다. 특히 GF2 P40는 보고서에 없는 신규 cohort다.

### D02 — adaptive correction인지 확인

**학습 없음. 동일 Student checkpoint에서** 다음 mode를 모든 RR20/FR20에 평가한다.

| Mode | 적용값 | 목적 |
|---|---|---|
| CURRENT | 해당 영상의 c_i | 현재 baseline |
| ZERO_WARP | c=0을 같은 sampler로 적용 | 정합 이동 제거 |
| ZERO_BYPASS | sampler 자체 bypass | 항등 resampling 영향 분리 |
| TRAIN_MEAN | CAL_TRAIN c의 성분별 평균 | 학습 기반 공통 bias 대조 |
| TRAIN_MEDIAN | CAL_TRAIN c의 성분별 중앙값 | outlier에 둔감한 상수 대조 |
| SHUFFLE19 | 20장 c를 k=1..19 circular derangement | 보정값 분포를 유지하고 장면 대응만 제거 |
| KAPPA4 | 4*c_i | 보고서 배율 진단 재현; 배포 기본값 아님 |

shuffle은 한 scene의 output metric을19회 평균한 뒤 scene-level paired 차이를 계산한다. 전역 c 분포 자체는 그대로 유지한다. constants는 test scene 평균이 아닌 CAL_TRAIN에서 결정하고 train64 단위 값을 그대로 적용한 진단임을 명시한다. FR만 보고 상수를 재최적화하지 않는다.

필수: 모든 mode의 native HQNR/Ds/Dlambda/JQM 및 RR metrics, scene별 c, ΔHQNR/ΔERGAS, paired CI. 품질차이가 적다는 것은 ‘같다’의 증명이 아니다. 추가 실험에 대한 정보로 분류한다.

### D03 — relative shift, MS 의존성, shortcut

PAN-only, MS-ref-only, both-shift, wrong-MS(정해진 다른 scene의 같은 크기 참조)에서 AXIS16/DIAG16/DISK64_HELD를 비교한다. signed response matrix를 적합해 `d_c=-J*epsilon` 기준 J_yy/J_xx와 cross-axis response를 올린다. MS-only는 부호를 변환해 동일한 recovery gain1이 이상값이 되도록 하되 원 vector도 CSV에 보존한다.

wrong-MS에서도 같은 gain이 나오면 PAN border/phase 단서만 학습했을 가능성을 검토한다. 그러나 wrong-MS는 distribution shift이기도 하므로 한 대조만으로 ‘MS를 전혀 안 쓴다’고 확정하지 않는다. 양성 대조 L04와 clean-support/joint 결과를 함께 본다.

boundary 검사: 기존 M4와 충분한 원 support의 중앙 crop을 비교하고, 동일 shift를 먼저 큰 FR canvas에 가한 뒤 동일 위치를 crop하는 방식도 검사한다. integer와 fractional offset을 분리한다. reconstruction output 평가와 이 진단용 stress metric을 섞지 않는다.

### D04 — 크기와 정규화/GAP

같은 FR20에서 **resize 없이** 중심 정렬64/128/256/512 crops를 사용한다. crop origin과 size는 MS ratio4와 맞춘다. 같은 sample의 local epsilon response가 crop 크기에 따라 바뀌는지 측정한다.

64 tiles의 예측 median/mean을 큰 장면의 global correction으로 적용하는 진단도 추가한다. 각 tile의 norm/분산과 edge-energy를 기록하되, test 결과에 맞춰 aggregation weight를 학습하지 않는다.

정규화 분해는 진단 hook으로만 한다.
- native per-input z-normalization/GN.
- 원 큰 canvas에서 계산한 입력 z-stats를 crop에 적용하되 GN은 native.
- 가능한 경우 frozen recorded GN stats hook을 별도 counterfactual로 추가.
이 결과를 새 모델 배포 방식으로 합치지 않는다. GN이 전역 통계를 쓰므로 convolution 수용영역25px만으로 입력 전체 영향 범위를 단정하지 않는다.

### D05 — 참조 위상

보고서의 gradient-NCC와 intensity-NCC 두 proxy를 유지한다. 각 reference에 맞는 bandwidth 조건·탐색반경·band 선택 규칙은 원 스크립트에서 가져오고 manifest로 고정한다. GT와 동일 band의 reference 비교를 별도로 계산한다.

현재 c를 실제로 적용한 PAN을 다시 estimator에 넣어 잔차를 측정한다. `||estimated_shift-c||`만 계산해 개선이라고 대체하지 않는다. band별 벡터, peak score, peak gap, boundary hit, valid band 수를 올린다.

native LMS와 bicubic의 차이가 blur인지 phase인지 민감도를 보기 위해 같은 blur 조건의 proxy도 별도 진단한다. **반 픽셀 보정값은 test에서 fit하지 않는다.** phase-aware reference를 후속으로 추가할 경우 training-only 동일-band 분석과 forward-coordinate 검증을 거쳐 별도 revision으로 한다. 이번 T07/T08의 필수 변경은 native LMS 참조다.

### D06 — 같은 FOV 스케일

보고서의 Gaussian sigma1.98/kernel41/replicate/[2::4] diagnostic downsampling을 재현한다. FR PAN512→128, LRMS128→32; 공식 Wald RR 재생성이라고 부르지 않는다.

먼저 원 FR에서 ±2px shift를 주입하고 같은 연산으로 축소해 저해상도 ±0.5px를 만든다. 주 지표는 절대 c 비율이 아니라 `Delta_c_high - 4*Delta_c_low` 및 각 scale의 recovery gain/MAE다. (native c mean norm의 비율도 보조로 기록하되 실제 물리 보정률이라고 해석하지 않는다.)

### D07 — A gradient

동일 batch의 `g_rec=grad_A L_rec`, `g_off=grad_A L_off`, `lambda0*g_off`를 optimizer step 없이 측정한다. 전체A와 pan_stem/ms_stem/joint/res/fc1/fc2, fc2.bias를 각각 기록한다.

필수: L_rec/L_off, 세 norm, ratio, cosine, active/inactive duty, per-sample grad 평균의 norm과 norm 평균(가능한 축소 probe), 비정상/zero-gradient 개수. 0 norm의 cosine은 숫자0으로 채우지 말고 공란+`UNDEFINED_ZERO_NORM` 사유를 남긴다. 최종 head zero 초기화 직후 body gradient가0인 현상은 원 초기화 의도와 구분한다.

---

## 5. L00–L04: A가 알려진 이동을 배울 수 있는가

L00은 무학습 step0 참조다. L01–L04는 **5K, batch48, AdamW 원 wd/betas/eps, warmup100+cosine5K**, offset 매 update, 반경2 원판면적균등이다. A-only 목적에는 loss를 λ배로 매우 작게 줄이지 않고 unit epsilon loss를 사용한다. L01/L02는 LR 비교이며 최종 method 후보가 아니다.

| Case | 시작 A | 목적 | A peak LR | U |
|---|---|---|---:|---|
| L01 | 정확한 donor A clone | relative epsilon loss | 1e-5 | 고정, monitor-only |
| L02 | 같은 donor A clone | 동일 | 1e-4 | 동일 |
| L03 | 원 초기화의 fresh A | 동일 | 1e-4 | 동일 |
| L04 | L03와 matched fresh A | same-PAN reference의 known absolute shift L1 | 1e-4 | 실제 PS 학습 없음 |

L04의 참조는 이동 전 PAN을 센서 밴드 수만큼 복제한 tensor다. reference/PAN에 같은 source blur를 적용하는 별도 variant가 필요하면 구분한다. 이는 network/warp/optimizer의 양성 대조이지 PAN–MS 정합의 성공 증거가 아니다. target=-epsilon이므로 native offset의 절대값까지 알려진 **synthetic 과제**다.

step0,100,500,1000,2000,3000,4000,5000에서 fixed probes, gradient·head/body norm과 native correction drift를 기록한다. 원점 부근만 잘하는지 radius별 점검한다. A-only는 절대 alignment bias가 미정이므로 native 품질이 나빠져도 구조가 이동을 못 배웠다고 해석하지 않는다.

---

## 6. T00–T10: 원 Teacher의 10K continuation 대조

### 6.1 공통 조건

- **모든 case는 동일 donor U/A에서 독립 복사해 시작**한다. T01의 학습된 checkpoint를 T02에 넘기지 않는다.
- fresh optimizer; U LR1e-4/A LR1e-5; batch48; warmup100+cosine10K; native train64; 원 norm·preprocessing·LP 유지.
- pilot seed는 데이터/augmentation/epsilon stream seed다. pretrained donor를 공유하므로 independent fresh Teacher seed로 세지 않는다.
- lambda0=resolved donor 값: 보고서WV3/G20=1e-4, QB_C3=3e-4. R4는 확인 후 결정한다.
- hard/soft/edge Student 계수는 이 Teacher pilot에서 튜닝하지 않는다.
- native reconstruction은 매 update. T00/T01/T02/T07 offset는 odd 0-based update; 다른 T는매 update.
- every2에도 RNG draw index는 같은 사전 stream의 update별 epsilon을 대응시킨다. 건너뛴 update가 뒤의 epsilon 순서를 바꾸지 않도록 한다.

| Case | lambda/lambda0 | every | A 참조 | U 학습 | 추가 reconstruction |
|---|---:|---:|---|---|---|
| T00 | 1 | 2 | bicubic MS | native | 없음 |
| T01 | 10 | 2 | 동일 | 동일 | 없음 |
| T02 | 100 | 2 | 동일 | 동일 | 없음 |
| T03 | 1 | 1 | 동일 | 동일 | 없음 |
| T04 | 10 | 1 | 동일 | 동일 | 없음 |
| T05 | 100 | 1 | 동일 | 동일 | 없음 |
| T06 | 10 | 1 | 동일 | **frozen** | A는 native reconstruction 유지 |
| T07 | 1 | 2 | **native LMS** | native | 없음 |
| T08 | 10 | 1 | **native LMS** | native | 없음 |
| T09 | 10 | 1 | bicubic MS | native-only | **A에만 shifted reconstruction** |
| T10 | 10 | 1 | bicubic MS | **native+shifted** | U/A 모두 shifted reconstruction |

T00/T03/T01/T04/T02/T05는 3계수×2빈도의 완전 비교다. T00/T04/T07/T08은 원/강한epsilon×bicubic/LMS의2×2대조다. 따라서 ‘reference 변경이 유효한데 단지 loss가 커져서 그랬는지’를 분리할 수 있다.

T07/T08에서는 **A가 보는 참조만** 바꾼다. U input의 MS base·GT·PAN/LP warp·HP construction은 그대로다. native LMS가 split 전체에서 제공되지 않거나 spatial contract가 맞지 않으면 BLOCKED, 보간으로 가짜 native LMS를 만들어 대체하지 않는다. 여러 blur/위상 차이가 포함된 reference-package 대조임을 명시한다.

### 6.2 T09/T10의 정확한 graph

`c0=A(P,R)`, `P_eps=W(P,epsilon)`; P_eps는 detach. `c_eps=A(P_eps,R)`.

- native U 입력: 원 synchronized frontend로 `W(P,c0), W(L,c0), HP, MSbase`.
- shifted U 입력: 원 P/L에 **epsilon+c_eps**로 one-shot warp, HP 재계산, MSbase·GT고정.
- `R_native=mean|Y_native-GT|`, `R_shift=mean|Y_shift-GT|`.
- T09: `L_U=R_native`; `L_A=(R_native+R_shift)/2 + lambda*L_off`.
- T10: `L_U=(R_native+R_shift)/2`; `L_A=(R_native+R_shift)/2 + lambda*L_off`.

A와 U의 parameter-specific gradient를 각각 계산한다. T09에서 U output을 `no_grad`로 만들면 A로도 gradient가 끊기므로 금지다. T09의 shifted branch는 U parameter update만 차단하되 input/c_eps로 미분되어야 한다. T09와T10에서 호출 수·shift/지원영역은 같게 한다.

native와shifted를1:1평균해 reconstruction의 단순2배 증폭을 피한다. native/shifted 둘 다 모든 case의 동일 crop에서 평가하는 보조 대조를 남겨 support 효과를 분리한다. loss support를 변경할 필요가 있으면 원 지원영역 baseline을 추가하고 T04 값에 조용히 적용하지 않는다.

T09/T10은 아키텍처가 아니라 **Teacher training objective 변경**이다. 과거 PO 구현에 synthetic reconstruction이 있었다는 보고서의 경고를 반영해 이전 PO 코드와 차이도 기록한다. 새 loss를 최초 제안했다고 자동 주장하지 않는다.

### 6.3 기록 주기·완결

- step0,100,500,1000 및 이후1000마다: PROBE_TRAIN/VAL AXIS16, native val reconstruction, gradient audit16 fixed mini-batches(전체 GRAD32의 앞16), c drift.
- step0,5000,10000: report-anchor RR/FR epsilon probe와 native quality. 공식 test 결과는 관측/보고용이며 pilot recipe 선택에는 사용하지 않는다.
- 완료 결과는 exact added10000. best gain step을 골라 exact10K라 쓰지 않는다.
- 새로운 loss branch 때문에 step당 시간이 늘어나면 training/evaluation 시간·effective batches를 함께 남긴다. 동일10K를 동일 wall-time이라고 표현하지 않는다.

---

## 7. X01–X03: 크기와 스케일은 구분해서 조건부 실행

**T04가 validation64에서 충분히 학습되었으나 큰 입력 또는 스케일 반응이 약한 경우**에 우선 실행한다. 사전 고정 comparator는 T04/X02다. 다른 pilot winner로 parent를 몰래 바꾸지 않는다. 수행하지 않으면 `NOT_PROMOTED_GATE` 사유를 남긴다.

| Case | 변경 | 해석 범위 |
|---|---|---|
| X01 | epsilon branch에서 native train64의 crop32/48/64를 균등 사용. U native64 유지 | 같은 GSD의 작은 window 혼합. native512 학습이 아님 |
| X02 | 같은 train patch를 k=1/2/4/8로 확대. A의 offset branch만 multi-scale, residual을 k로 정규화 | synthetic magnification 학습. native FR 분포와 같지 않음 |
| X03 | X02와 같은 paired calls/stream에 scale difference consistency 추가 | 알려진 offset 회복과 스케일 관계를 함께 학습 |

실제 더 큰 **학습용** native PAN/MS pair가 있다면, dataset manifest와 disjoint split을 감사한 뒤 `X_NATIVE` revision을 추가할 수 있다. 공식 RR20/FR20을 학습에 넣어 일반화 결과를 만드는 것은 금지다. 첨부 자료만으로 큰 native train pair가 존재한다고 가정하지 않는다.

X02에서는 원 train P/R에 native epsilon를 먼저 주입하고, 두 tensor를 같은 center mapping으로 k배 확대한다.

\[
L_{\mathrm{off}}^{(k)}=\operatorname{mean}\left|\frac{c_\epsilon^{(k)}-\operatorname{sg}(c_0^{(k)})}{k}+\epsilon\right|.
\]

상수 bias는 같은 스케일 내 차분으로 분리한다. 원본에 epsilon를 주고 resize한 tensor와 큰 스케일에서 직접 k·epsilon로 warp한 tensor는 수치적으로 같지 않으므로 UNIT에 차이를 기록한다.

**Synthetic offset branch의 공통 물리적 support:** 원64 기준 margin4에 해당하는 `4*k`를 확대 tensor에서 제외한다. 이를 위해 `A`를 직접 호출하는 전용 probe wrapper를 만들되 crop을 중복 적용하지 않는다. 이 branch의 내부 크기는56k다. native production inference의 wrapper margin4는 변경하지 않는다. 고정 margin4의 큰 입력 진단도 D04에서 별도로 남겨 두므로, 두 프로토콜을 섞지 않는다.

X03의 추가 항은 다음이다.

\[
L_{\mathrm{scale}}=\frac12\operatorname{mean}\left|\frac{\Delta c^{(k)}}k-\operatorname{sg}(\Delta c^{(1)})\right|+\frac12\operatorname{mean}\left|\operatorname{sg}\left(\frac{\Delta c^{(k)}}k\right)-\Delta c^{(1)}\right|.
\]

`lambda_scale=10*lambda0`이며 known-offset loss도 유지한다. 이 관계만 쓰면 zero response도 통과하므로 scale loss 단독 실험을 성공 기준으로 쓰지 않는다. X02도 동일 paired forward를 계산하고 `lambda_scale=0`으로 두어 sample 수와 계산 경로를 맞춘다.

공통10K, A LR1e-5/U LR1e-4, offset 매 update다. U native64는 매번48개 patch를 유지한다. 큰 A view는 microbatch로 나누되 optimizer 한 step의 총 유효 sample48과 scale 비율을 맞춘다. OOM이면 microbatch만 줄이고 effective batch·LR·offset mix를 바꾸지 않는다.

---

## 8. F/S: 개선이 실제 학습과 Student까지 남는가

### 8.1 Pilot promotion

다음은 **운영용 사전 기준**이지 통계적 유의성이나 물리 registration의 인증 기준이 아니다. exact10K의 동일 PROBE_VAL에서 판정한다.

1. In-range AXIS16에서 각 축 gain이 [0.5,1.5]이며, offset MAE가 matched T00의50% 이하.
2. VAL64 reconstruction ERGAS의 상대 악화가 T00 대비+1% 이내. 기존 validation evaluator가 ERGAS를 제공하지 않으면, 실행 전에 별도 protocol로 원본 L1 상대값을 지정하고 그 사실을 기록한다. 진행 중 더 유리한 지표로 바꾸지 않는다.
3. 비정상값이 없고, 비교용 유효 support≥95%, 필수 자산·gradient routing 단위검사가 완료됨.
4. Clean-support와 joint/wrong-MS 결과를 모두 올림. Shortcut을 배제할 수 없으면 `MECHANISM_INCONCLUSIVE`로 표시하고 gain 개선만으로 정합 성공이라 하지 않는다.

후보는 T01–T05, T07–T10 및 완료된 X01–X03다. **T06(U frozen), L 계열, 추론 KAPPA4는 F01 후보에서 제외**한다. 학습된 U를 고정한 진단을 무학습 fresh U에 그대로 적용하지 않기 위함이다.

여러 후보가 통과하면 **VAL offset MAE 최소 → VAL ERGAS 최소 → 낮은 case 번호** 순서로1개를 잠근다. 모든 후보와 제외 사유를 DECISION 행으로 올린다. 공식 RR20/FR20의 수치는 이 결정에 사용하지 않는다. 아무도 통과하지 않으면 `NO_ELIGIBLE_RECIPE`를 보고하고 F01/S01을 억지로 생성하지 않는다.

서버별 선택이 달라도 다른 서버를 기다리지 않는다. 그러나 **서로 다른 recipe를 같은 variant의 독립 반복으로 합산하지 않는다.** 같은 recipe의 반복은 다음 local fresh2seed로 확보한다.

### 8.2 Fresh Teacher 확인

- F00: 원 Teacher 학습 recipe를 fresh U/A로 재학습한다.
- F01: 잠근 개선 학습 규칙을 fresh U/A에 적용한다. F00과 초기 tensor·data/epsilon stream·batch·학습량을 맞춘다.
- WV3/QB/G20: Teacher50K. P40/R4 생산 cohort: Teacher100K.
- 각 조건은 fresh seed2개로 대응한다. seed는 `lane_assignments.csv`에 있다. 초기화가 이전 학습 checkpoint를 로드했다면 fresh로 표시하지 않는다.
- Teacher는 **exact endpoint**를 Student donor로 사용한다. epsilon가 잘 나온 중간 Teacher를 사후 선택하지 않는다.
- 새 Teacher마다 원 calibration 규약으로 tau_R/q_ref/q-cache를 만든다. 다른 Teacher에 옛 q-cache를 혼용하지 않는다. q 변화로 edge/A 학습의 평균 강도도 바뀔 수 있으므로 분포와 gradient를 함께 기록한다.

### 8.3 Student 전이

| Student | Reference | A 학습 | 질문 |
|---|---|---|---|
| S00 | F00 | clone 후 기존 q-hard | 기존 Teacher→FULL 기준 |
| S01 | F01 | clone 후 기존 q-hard | Teacher 개선이 기존 Student 학습으로 전이되는가? |
| S02 | F00 | warp bypass, A 업데이트 없음 | 같은 reference에서 실제 no-Student-aligner 재학습 대조(C09형) |
| S03 | F01 | q-hard + epsilon 유지 | 상대 반응을 Student에서도 유지해야 하는가? |

S02는 Teacher 전체 제거가 아니다. Teacher prediction/difficulty/q-edge는 S00과 같고, Student A만 bypass한다. graph와 representation 채널 수는 기존 C09 정의와 확인한다. 추가적인 생략이 필요하면 별도 METHOD_REVISION으로 표시한다.

S03의 epsilon 계수·빈도는 잠근 F01 설정이다. A에만 직접 gradient를 주고 U에는 추가 epsilon gradient가 없도록 한다. 기존 Student method의 변경임을 명시한다. S00/S01/S02는 F/S 확인의 기본 묶음이며, S03는 별도 조건부 확장이다.

P40 생산 cohort의 Student는 양측 모두 **native100K → G050 mix20K**를 맞춘다. 연구용50K Student를 B09의 직접 대체 결과로 표시하지 않는다. native endpoint와 MIX endpoint, Teacher별 G050 calibration을 각각 보존한다.

### 8.4 Checkpoint 선택과 저장

- Teacher/pilot 기전: exact budget 및 사전 trajectory. 모델마다 best gain step을 골라 섞지 않는다.
- Student50K: `1010,2020,...,49490,50000`50후보에서 native FR20 HQNR 최대, 동률이면 더 이른 step. 이 `HQNR_MAX50`는 test-aware 개발 보고다.
- 모든 Student에서 EXACT_FINAL도 별도 보고한다. 생산100K/20K의 후보 grid는 원 P40 정의를 고정하고 실제 expected_count를 기록한다. 50후보가 아니면 HQNR_MAX50라고 부르지 않는다.
- Native quality 선택과 기전 probe 선택은 다른 목적이다. F01 recipe는 test 결과를 사용하지 않고 잠가야 한다.
- `paper`, `유의미한결과`, `배포용 모델`의 대표를 자동 교체하지 않는다.

---

## 9. 핵심 지표 정의

`d=c_eps-c0`라 두며 아래는 PAN-only 기준이다. MS-only는 기대 부호를 별도로 명시한다.

| 필드 | 정의 |
|---|---|
| Gain_num / Gain_den | `-sum(d dot epsilon)` / `sum(epsilon dot epsilon)` |
| Gain | 두 누적량의 비. epsilon=0은 분모에서 제외 |
| Offset_MAE_px | sample·probe·component에 대한 `mean(abs(d+epsilon))` |
| Offset_EPE_px | `mean(norm(d+epsilon,2))` |
| Gain_x / Gain_y | 해당 축 probe만의 recovery gain |
| Joint_shift_EPE_px | 공통 support에서 `mean(norm(A(W(P,eps),W(R,eps))-A(P,R)))` |
| Native_c_dy / dx | `mean(c_i)`의 각 성분 |
| Mean_norm_c | `mean(norm(c_i,2))`. 평균 벡터의 norm이 아님 |
| Std_c | `sqrt(mean(norm(c_i-mean(c),2)^2))` |
| Grad_ratio | `norm(lambda*g_off)/(norm(g_rec)+tiny)`; tiny·zero norm 상태 명시 |
| Grad_cos | 양쪽 norm이 nonzero일 때만 cosine 계산 |
| Delta_HQNR | variant-control |
| Delta_ERGAS_pct | `100*(variant/control-1)` |
| Delta_offset_MAE / proxy | variant-control. 낮을수록 개선 |

J_yy/J_xx/J_yx/J_xy는 RESPONSE의 `Metric_name`과 `Estimate`로 각각 올린다. PAIRED의 CI는 Metric_name으로 어떤 차이의 CI인지 표시한다. 하나의 CI를 여러 지표에 공통으로 쓰지 않는다.

기존 q와 새 offset MAE는 support·probe 분포·reduction이 일치할 때만 비교한다. 이름이 비슷하다는 이유로 같은 값으로 취급하지 않는다.

기전 판정은 **응답 gain / MS 의존성 / native 정합 proxy / 최종 RR·FR 품질**을 분리한다. FR에는 GT가 없으며, 이 네 축의 개선 방향이 달라도 이상한 것이 아니다. 복합점수 하나로 충돌을 숨기지 않는다.

---

## 10. analysis 시트 업로드 계약

### 10.1 위치와 초기 상태

- Spreadsheet: `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`.
- 탭: **analysis**, sheetId=`261006100`.
- 행1–5: 제목·상태 집계·범례. 행6: 열 그룹. **행7: 112열 고정 schema**. 행8부터 누적 원장.
- 초기 상태: PLAN33행, 보고서 REFERENCE6행, 서버 NOT_STARTED5행. **새 학습 결과가 아니다.**
- 모든 숫자는 full precision으로 저장하며 표시만4–6자리. 방법 설명은 개행 없이 한 줄. 긴 metadata는 오른쪽 접힌 열에 보존한다.
- PLAN 회색 / REFERENCE 청회색 / 실측 SUMMARY 녹색 / RESPONSE·GRAD 보라 / PAIRED 청록 / PENDING 노랑 / FAILED 빨강.
- 초기2000행은 저장·필터·수식의 준비 범위다. 확장 시 마지막 실제 행을 기준으로 dashboard 범위·filter·서식을 함께 확장한다. 제목의 집계는 물리 행 수이며 독립 run 수가 아니다.

### 10.2 필수 레코드

| Record_type | 단위 | 반드시 올릴 내용 |
|---|---|---|
| PLAN | case 종류 | 변경점·대조·budget. 성능칸 공란 |
| ASSET | cohort/asset | 원/실제 SHA, 위치, 보유 여부, precision/runtime/source identity |
| STATUS | run/phase event | 실제 update, PID/active release, 기대·완료 평가 수, 중단 사유 |
| UNIT | 단위검사 | 검사 이름, 오차·허용조건·pass/fail·protocol |
| SUMMARY | run/checkpoint/variant | native FR/RR + gain_train/RR/FR + shift 통계 + scope |
| CURVE | run/고정 step | train/val reconstruction, epsilon MAE/gain, c drift, 실제 학습량 |
| RESPONSE | role/split/크기/probe/반경/방향 | gain 누적량·MAE/EPE·교차축 반응·유효 표본 수 |
| SCENE | checkpoint/variant/split/scene | scene별 metric/c/proxy. RR과 FR은 별도 행 |
| GRAD | run/step/layer group | rec/off raw/off weighted norm, ratio/cos, zero norm 및 빈도 |
| PHASE | split/reference/proxy | 동일 band phase·band별 벡터·peak·유효율 |
| PAIRED | variant-control | effect/CI, n_scenes/n_seeds, paired scope |
| DECISION | 완료 local block | 전체 후보의 gate 통과·제외 사유, 잠근 recipe, 선택 데이터 |

**Raw JSON 링크 하나만 올리면 미완료**다. 핵심 summary, 반경별 response, 필수 native20scene, paired 차이, gradient 통계는 숫자열에 있어야 한다. 전체 epsilon별 vector·상세 layer gradient·이미지는 immutable CSV/JSON/PNG로 보존하고 URI와 hash로 연결한다. 원시파일을 열지 않아도 어느 가설을 지지하는지 판단 가능해야 한다.

입력 크기·epsilon 반경·reference·support가 다른 결과는 별도 행이다. 측정되지 않은 항목은 공란으로 두고 Reason에 NA 이유를 쓴다. `source_report_sha256.txt`에 기록된 기존 보고서 값은 신규 실측의 대조값이 아니라 참고 자료다.

### 10.3 업로드 주기

- 시작·자산 누락·중단·실패: 즉시 STATUS.
- 매1000 update 또는 phase 종료 중 빠른 시점: CURVE. L은 §5의 추가 시점 포함.
- D case 완료: SUMMARY와 필수 상세·PAIRED.
- T/X endpoint, F/S endpoint/선택점 평가 완료: 관련 모든 결과.
- F/S의 A 반응: 초기 clone,100/500/1000, 이후5K마다.
- expected_count와 completed_count가 일치하기 전 COMPLETE를 표시하지 않는다.

### 10.4 고유 key와 동시 쓰기

`Record_ID=hash(campaign,server,cohort,run_id,attempt,case,checkpoint_sha,selector,record_type,split,scene/crop,probe/radius/direction,layer,replica,protocol_revision)`.

같은 payload를 재전송하면 같은 ID다. 계산을 수정하면 새 revision ID와 supersedes ID를 Reason에 기록한다. 새로운 attempt를 새로운 seed로 세지 않는다.

- 각 서버는 local outbox와 local uploader 하나를 사용한다.
- 공유된 마지막 행을 읽은 다음 그 행에 쓰는 방식은 금지다. provider의 원자적 `appendCells` 또는 `values.append`를 사용한다.
- 다른 서버의 행을 수정·정렬·삭제하거나 시트 전체를 clear하지 않는다.
- 응답 유실 retry로 중복 ID가 생기면 payload 충돌을 확인한 뒤 분석 view에서 dedup한다. provider가 exactly-once를 보장한다고 가정하지 않는다.
- write 후 실제 값을 재조회해 ID·payload·숫자·count를 대조한다. 그 전에 READBACK_VERIFIED를 쓰지 않는다. 검증 결과는 확인된 자기 행만 업데이트하거나 별도 receipt로 남긴다.
- 429/네트워크 실패는 backoff와 outbox로 복구한다. 학습을 다시 하지 않는다. outbox 보존에 필요한 디스크 하한에 닿으면 그 서버만 안전 pause한다.
- 전체 workbook의 cell 사용량을 검사한다. 기본 SUMMARY/RESPONSE와20scene를 누락하지 않도록 필요하면 승인된 보조 원장을 확장한다. 용량 때문에 상세가 빠지면 DETAIL_UPLOAD_PENDING이며 COMPLETE가 아니다.

### 10.5 완료 조건

계획 update/표본 완료, matched source/init/stream 검증 또는 차이 명시, 필수 child record 수 일치, finite 숫자와 단위/split/selector/평가 SHA, native RR/FR의 같은 A/U checkpoint, 마지막 readback까지 확인한 경우에만 완료다.

`FAILED_NUMERICAL`, `FAILED_IO`, `BLOCKED_ASSET`, `BLOCKED_PROTOCOL`, `PENDING_EVAL`, `UPLOAD_PENDING`, `NOT_PROMOTED_GATE`를 구분한다. 실패를 빈행으로 없애지 않는다.

`analysis_columns.json`은 열 순서, `analysis_upload_template.csv`는 빈 양식, `analysis_column_dictionary.csv`는 열의 단위와 입력 계약이다. 제공 validator는 로컬 레코드 구조 검사이며 실제 Sheet write/readback이나 GPU 검증을 대체하지 않는다.

---

## 11. 시트에서 판단할 기준

| 관측 | 가능한 설명 | 다음 판단 |
|---|---|---|
| CURRENT≈CONST≈SHUFFLE, zero만 다름 | 공통 bias/공동 적응의 영향 가능성 | Adaptive 정합 주장 보류; D03/D05 확인 |
| L04조차 실패 | 좌표·학습·표현 문제 | Head zero/gradient/positive control 검사 |
| L04 성공, L01–L03 실패 | Cross-modal 참조/목표 난도 | Reference·blur·specificity 검사 |
| L01/L02 성공, T00 실패 | Signal strength/공동 최적화 영향 | λ×빈도와 T06 충돌 비교 |
| T06만 성공 | U/A 공동 적응 영향 가능성 | Frozen trained U를 fresh 기본법으로 확대하지 않음 |
| T07/T08가 대응 실험보다 개선 | A reference-package 영향 | Phase·blur 효과 구분, FR 물리 정합 단정 금지 |
| T09가 T04보다 개선 | A의 task-supervision 연결 가능성 | Fresh Teacher/Student 전이 검정 |
| 64 gain 좋음, 같은 GSD의512만 나쁨 | 입력 크기/content/statistics 영향 | D04/X01, boundary·low texture 층화 |
| 같은 GSD는 양호, 같은 FOV scale만 약함 | Scale 전이 영향 | X02/X03·좌표계 점검 |
| Teacher 개선, S01에서 gain 하락 | Student refinement의 반응 약화 가능성 | S03 비교 |
| Gain/proxy 개선, HQNR 악화 | 정합·재구성·무참조 목적 불일치 | 지표를 바꾸지 않고 충돌 보고 |
| GF2 G20/P40 방향이 다름 | Recipe/reference 의존성 | Cohort별 결론, 센서 전체로 합치지 않음 |

작은 차이가 반복 오차·scene·seed 변동보다 큰지도 함께 보고한다. 빈 metadata로 구현 정상이나 물리 정합 GT의 존재를 추정하지 않는다.

---

## 12. 예산·보존·구현 수락 검사

### 12.1 실행량

한 lane 첫 block은 **L4개×5K + T11개×10K = 130K optimizer updates**다. D와 L00은 학습0이다. Native LMS가 없는 T07/T08은 BLOCKED로 남긴다. 다섯 lane 전체 수행 시650K pilot updates이지만, 다섯 independent fresh Teacher를 뜻하지 않는다.

X는 조건부3개×10K. F/S는 잠근 recipe의 fresh seed2개에 대한 별도 비용이다. P40의100K/120K 경로를50K와 같은 시간으로 추정하지 않는다. **195개 work 선언은 진단·학습·조건부 확인의 합계**이며 전부 즉시 학습시키는 queue가 아니다.

이번 요청에는20/40시간 제한이 없다. 과거 상한을 재사용하지 않는다. 동시에 좋은 순서가 나올 때까지 무한 seed 탐색을 하지 않는다. 사전 block을 완결하고 analysis로 판정한다. 시간 제한이 추가되면 미완 상태와 대응 대조를 보존한다.

### 12.2 보존

Anchor/donor/H5/LP/source는 read-only다. 명명 checkpoint·curve·full-state·원시 vector·manifest·outbox는 보존한다. F/S 후보는 평가 종료 전에 제거하지 않는다. 기존 ablation·PC-Repro·main 모델은 이번 작업의 disk cleanup 대상이 아니다.

### 12.3 구현 후 반드시 실행할 검사

1. Warp64/256/512, dy/dx 부호, zero 항등, composed/double warp 차이.
2. A의 실제 view56/248/504와 GAP64dim, 출력2vector.
3. T case의 λ/frequency/U freeze/reference/gradient routing.
4. T09에서 U로 shifted 직접 gradient 없음, A에는 존재. T10은 둘 다 존재.
5. X의 단위 정규화·support·zero-response 반례.
6. 동결 개입의 동일 A/U, constant/shuffle identity.
7. Scene/patch/seed의 개수 및 비대응 RR/FR 처리.
8. Native evaluator·FR GT 부재·proxy 분리.
9. Candidate count·selector·같은 checkpoint 지표.
10. Outbox 중복·실패·retry 및 실제 Sheet readback.

이 문서/CSV의 정적 검사는 실제 Torch/CUDA/학습 회귀 검사와 다르다. 실행 결과를 시트 UNIT으로 올려야 한다.

---

## 13. 시트 결과의 형태

아래는 레코드 종류의 예시이며 숫자는 의도적으로 생략했다. 실제 업로드는112열 schema에 맞춘다.

```text
SUMMARY | PILOT | T04 | WV3 | WV3_F1 | s1 | T | COMPLETE | ... | EXACT_10K | 10000 | native RR20+FR20 | numeric metrics
RESPONSE | PILOT | T04 | WV3 | WV3_F1 | s1 | T | COMPLETE | ... | PROBE_VAL | 64 | r=1,x+ | gain numerator/denominator, MAE, EPE
GRAD | PILOT | T04 | WV3 | WV3_F1 | s1 | T | COMPLETE | ... | fc2.bias | rec/off norms, ratio, cosine
PAIRED | PILOT | T04 | WV3 | WV3_F1 | s1 | T | COMPLETE | ... | T04-T00 | metric=HQNR | estimate, CI, n
```

Raw 파일을 직접 열라는 메모만으로 숫자 업로드를 대체하지 않는다. 논문 figure가 필요해도 우선 numerical 결과와 선택 근거를 올린다.

## 14. 실행 담당자 시작 지시

> 이 폴더의 `experiment_registry.json`과 MD를 기준으로 PAN Aligner 원인 분리 캠페인을 구현·실행하라. 기존 runner의 활성 상태와 자산을 확인한 뒤 안전 handoff하고, s1 WV3/F1, s2 QB/TB_C3, s3 GF2/G20R0, s4 WV3 독립 반복, s5 GF2/P40R4로 각자 진행한다. 먼저 D00–D07, L00–L04, T00–T10을 완결한다. 아키텍처는 바꾸지 않고 loss/참조 변경은 revision으로 표시한다. 결과는 analysis에 STATUS/SUMMARY/RESPONSE/SCENE/GRAD/PAIRED/DECISION을 숫자와 SHA로 업로드하고 재조회한다. 기전 exact budget와 Student HQNR_MAX50를 구분한다. 기존 paper·ablation·배포 대표·원본 시트·checkpoint는 변경하지 않는다. 미확인은 BLOCKED/PENDING으로 남긴다. 실제 PID/run/update 증가와 analysis 기록을 기동 receipt로 보고하라.

## 출처와 우선순위

**[S1]** 동봉 `2026-10-06_PAN_Aligner_RR_FR_effectiveness_report.md`: §2 대상/실행 범위, §3 좌표·구조, §4 phase/proxy, §6 epsilon/gain, §7 동결 quality, §9 support, §11 script.

**[S2]** Report의 commit `a2705bd1b92ed9a7f0c54a6b4b9ee4d7b82ea1a5`: `pa/aligner.py`, `pa/warp.py`, `fh12/model.py`, `fh12/losses.py` 및 QB/G20 실제 resolved 구현. 코드 설명은 보고서와 이전 세션 조회에 기반한다. 새 실행은 자신의 actual source SHA를 다시 고정한다.

**[S3]** 2026-10-06 live workbook의 paper 대표·metadata. P40 full SHA는 이전 보존 기록에 기반하며 기동 전에 다시 검증한다. 서버 파일 검사를 새로 수행했다는 뜻이 아니다.

우선순위는 최신 사용자 지시 → 이번 사전등록 명세 → 보고서 관측 → 과거 일반 계획이다. 관측 결과와 이번에 제안한 성공 기준을 구분한다.
