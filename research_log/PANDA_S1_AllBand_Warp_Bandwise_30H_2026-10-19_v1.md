# s1 WV3 — 전 밴드 정렬 + warp/좌표계 + 밴드별 정렬, 30시간 case 계획

> 실행 후속: 2026-10-19 사용자 실행 승인 후 [panda_band30](../panda_band30/README.md)에 구현했고
> s1 Docker 기동 기록은 [WIP](../results_log/2026-10-19_WIP_s1_allband-bandwise30.md)에 남긴다.
> 아래의 상태·산출물 설명은 case 구성 당시의 기록이다. JSON/CSV는 등록 계획 원본으로 보존한다.

작성: 2026-10-19. **상태: case 구성 완료 / 새 학습 경로 미구현 / 학습 미시작.**
이 문서와 JSON/CSV는 구현할 실험의 등록 명세이며 현재 러너에 넘겨 실행할 수 있는 config가 아니다. 이번 작업에서 기존 s3 코드·실행 큐·결과는 변경하지 않았다.

- [Case CSV](PANDA_S1_AllBand_Warp_Bandwise_30H_2026-10-19_v1_Cases.csv)
- [기계 판독 명세](PANDA_S1_AllBand_Warp_Bandwise_30H_2026-10-19_v1.json)
- 비교 원형: [s3 warp30](../panda_warp30/README.md), commit `6410328`.
- 근거: [s1 분석 §9–10](../results_log/2026-10-19_s1_ms-shift-alignment-feasibility.md#9-밴드별-오정합--s3-분석의-s1-교차검증과-방향-시사점), [Stage 2 loss 분석](../results_log/2026-10-19_panda-stage2-loss-analysis.md).

## 1. 사용자 결정과 정확한 변경 범위

1. s1/WV3에서 s3의 fresh Stage 1 → frozen A/T Stage 2 비교를 반복한다.
2. Stage 1의 글로벌 정렬 기준을 RGB가 아닌 MS 전 8밴드로 바꾼다.
3. **MS 단독 이동 + 출력 역warp(`ms_return`)는 R1/R2 모두 제외**한다. 교차정합의 MS 반쪽 이동은 유지한다.
4. 별도 분기로 8개 밴드별 shift, 총 16개 scalar 출력을 추가한다.
5. 사용자가 승인한 경로: **PAN 복사본 8개를 각각 정렬 → MS 8밴드와 연결 → U-Net 16채널 입력**. MS·GT 이동은 학습 증강에만 적용하며 MS-only 추론을 다시 넣지 않는다.
6. LRMS 업샘플링은 s3와 같은 post-augmentation interp23tap + 위상 보정, 비교군은 s3와 같은 기존 bicubic이다. 소수 shift warp는 bicubic 유지.
7. HQNR→ERGAS 선택, 원본 unmasked RR20/FR20, 모든 공식 평가의 로컬·시트 보존을 유지한다.

**중요한 정정:** 현재 Aligner의 입력 `ms_stem`은 이미 8밴드이다. 이번 글로벌 변경은 입력 밴드 추가가 아니라, `ScharrAbsolute`의 RGB GT 참조를 8밴드 참조로 바꾸는 것이다.

## 2. 30시간 큐

모든 학습 run의 계획 horizon은 50K optimizer updates이다. 우선 8벌 + 조건부 1벌을 중심으로 구성하며, 등록된 14벌 전체를 30시간에 완료한다는 뜻은 아니다.

| 순서 | case | 실제 변경 / 질문 | Stage 1 donor | 우선순위 |
|---|---|---|---|---|
| 1 | G8_S1_U23 | 8밴드 평균 목표, 단일 shift, interp23tap fresh Stage 1 | 새로 학습 | 우선 |
| 2 | G8_PAN_U23 | PAN→MS 기준 Stage 2 | G8_S1_U23 | 우선 |
| 3 | G8_MID_U23 | PAN/MS 반반, 출력 반환 없음 | 동일 | 우선 |
| 4 | G8_MIDRET_U23 | 반반 + 학습 내부 출력 역warp | 동일 | 우선 |
| 5 | B8A_S1_U23 | 밴드별 16출력 + 합성 FR-scale Aligner 증강, 16ch Teacher | 새로 학습 | 우선 |
| 6 | B8A_PAN_U23 | 위 A/T 고정 + fresh 16ch Student, 밴드별 PAN→MS | B8A_S1_U23 | 우선 |
| 7 | G8_S1_BIC | 8밴드 평균 목표, 기존 bicubic fresh Stage 1 | 새로 학습 | 우선 |
| 8 | G8_PAN_BIC | bicubic Stage 2, 전체 보간 경로 대조 | G8_S1_BIC | 우선 |
| 9 | G8_PAN_U23_S050 | 글로벌 보정 배율 0.5 | G8_S1_U23 | 첫 잔여시간 case |
| 10 | G8_PAN_U23_S000 | Student 보정 0, Teacher full shift 유지 | 동일 | 예비 |
| 11 | G8_PAN_U23_S025 | 글로벌 보정 배율 0.25 | 동일 | 예비 |
| 12–14 | G8_PAN_U23 / G8_MID_U23 / G8_MIDRET_U23 R2 | 세 좌표 경로 동일 두 번째 seed | 동일 | 세 벌 전체 시간 확보 시 |

새 밴드별 분기가 마지막까지 밀려 한 번도 실행되지 않는 것을 피하기 위해 5–6번에 넣었다. Stage 1 donor를 필요한 순서에 따라 생성하고, 그 donor의 전체 50K 완료 및 선택 checkpoint 검증 후에만 해당 Stage 2로 넘어간다. 다른 서버 donor나 과거 모델로 자동 대체하지 않는다.

`G8` = 8밴드 평균 목표의 글로벌 2-output Aligner, `B8A` = bandwise 16-output + auxiliary augmentation, `U23` = interp23tap, `BIC` = s3 기존 bicubic, `MIDRET` = midpoint + output return.

## 3. 시간 근거와 운영 규칙

s3의 2.1h/Stage 2 prior를 s1에 그대로 쓰지 않는다.

- s1 [panda40 기록](../work_dir/panda40/PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1/s1/WV3/campaign_state.json): Direct Stage 1 H50 여섯 벌 2.550–2.570h. 계획 기준은 한 벌 **2.60h**.
- s1 [conditional-q 측정](../work_dir/panda30r/PANDA_S134_DONOR_SCHEDULE_30H_20261018_v1/s1/WV3/preflight_stage2_q.json): update p90 0.220395s, RR20+FR20 평가 20.013s. 50K + 35회 평가 약 3.256h, checkpoint당 10초라는 **기존 추정치**를 더하면 **3.353h**.
- 새 interp23tap·bandwise·추가 Aligner forward·역warp의 실제 시간은 아직 측정하지 않았다. 아래는 과거 연산 기준의 계산이지 새 모델 실측이 아니다.

| 범위 | 과거 연산 기준 합계 | 20% 여유 적용 |
|---|---:|---:|
| 우선 8벌 = Stage 1×3 + Stage 2×5 | 24.565h | 29.478h |
| + 0.5배 Stage 2 한 벌 | 27.918h | 33.502h |

따라서 0.5배까지 무조건 30h 안에 완료한다고 약속하지 않는다. 우선 8벌도 새 연산이 느리면 미완료될 수 있다.

- 총 30h는 실행 준비·profile·calibration·학습·평가·저장·진단을 포함한 누적 wall time.
- 29.5h에서 학습/새 평가를 안전 중단하고 마지막 30분은 저장·필수 진단 마감용으로 예약한다. 중단 재실행으로 시계를 초기화하지 않는다.
- optional은 실제 update p90, 전체 평가, 저장, 준비/진단 잔여 비용에 20% 여유를 더한 전체 run 시간이 남을 때만 시작한다. R2는 세 벌 묶음으로 판단한다.
- 미완료는 partial/STOPPED_SAFE, 예비 미실행은 BUDGET_SKIPPED로 남긴다. 50K를 몰래 30K/40K로 줄이지 않는다.
- 같은 서버·같은 출력 폴더 중복 실행만 즉시 거부한다. 서버 간 lock/대기는 없다. 기존 실험을 자동 종료하지 않는다.
- bandwise 전후 잔여 정합 진단은 모든 35개 checkpoint에 반복하지 않고 선택 모델/동일 A·geometry에서 재사용한다. 미완료 진단은 pending으로 표시하며 결과를 만들어내지 않는다.

## 4. 공통 학습·선택 조건

- Teacher/Student backbone W112·D123, attention 없음, LN, 단일 HRMS task.
- 글로벌 input 9ch. bandwise input 16ch. 출력은 모두 WV3 8밴드.
- Stage 1 seed 261016101. Stage 2 R1 261018101, R2 261018102. s3와 같은 stream-seed 규약.
- batch48=micro12×4, FP32/no TF32. LR(T/S)=1e-4, LR(A)=1e-5, wd=.01, 기존 warmup/cosine.
- 글로벌 Stage 1: `Lrec + 2·Labs + .003·LepsPAN`.
- Stage 2: 기존 conditional-q, α=.25 / β=.05 / γ=.5. A/T 동결, S fresh. 이번에는 β sweep·all-band q 등 loss 확장을 동시에 하지 않는다.
- checkpoint는 **공식 FR20 unmasked HQNR 최대 → 동점일 때 RR20 ERGAS 최소 → 먼저 평가한 update**. SCC는 기록한다.
- Dλ/잔여 정합은 별도 물리 정합 진단이며 checkpoint 선택을 바꾸지 않는다. 진단 best가 필요하면 별도 표기로 보관하되 운영 donor로 자동 대체하지 않는다.

## 5. 글로벌 전 밴드 목표와 보간 대조

글로벌 G8은 기존 RGB intensity mean→Scharr 구조를 유지하되 mean의 밴드만 `[4,2,1]`에서 `[0,1,2,3,4,5,6,7]`로 바꾼다.

`Labs_G8 = mean_ROI,xy |D(W(P,u)) − D(mean_k GT_k)|`

D는 기존 각 축 Scharr 절댓값이다. **이 식은 밴드별 Scharr loss 8개 평균과 다르다.** 보고서의 단순 융합에서 추정 shift 8개를 평균한 결과와도 수학적으로 같은 연산은 아니다. +.021 HQNR이나 −2.4% ERGAS가 학습 모델에서 재현된다고 전제하지 않는다.

s3와 비교를 유지하기 위해:
- U23 arm은 LRMS flip/rotation 이후 interp23tap과 기존 정수 위상 보정.
- BIC arm은 s3의 기존 torch bicubic 그대로. 이번 s1에서 BIC만 위상 보정 bicubic으로 바꾸지 않는다.
- 저장 LMS를 U-Net base로 직접 대체하는 추가 arm도 이번 30h에는 넣지 않는다. Aligner의 제공 LMS 입력은 그대로 유지한다.

`G8_PAN_U23 ↔ G8_PAN_BIC`은 각각 별도 fresh Stage 1을 거친 **전체 보간 경로 비교**다. 나머지 G8 U23 좌표/배율 case는 정확히 같은 선택 A/T를 공유한다.

## 6. 승인된 밴드별 16-output / 16ch 구조

`A(P,LMS) → u[B,8,2]`, 각 밴드의 순서는 `(dy,dx)`.

`P_k = W(P,u_k)`, `U_input = concat(P_0…P_7, B_0…B_7)`.

B는 위상 보정 interp23tap base이며 MS를 따로 정렬하지 않는다. GT도 정상 reconstruction 경로에서는 원래 GT 그대로다. 최종 밴드 k는 원래 MS_k/GT_k 좌표를 따른다. 이는 PAN–MS 각 밴드 쌍을 정합하는 방식이지, 8개 MS 밴드를 공통 PAN 좌표로 재투영하는 방식은 아니다.

- Aligner 마지막 linear 32→2를 32→16으로 변경: +462 params.
- U-Net input stem 9→16ch: W112·3×3 기준 +7,056 params.
- A+한 복원기 배포 기준 합계 +7,518 params의 산술 차이. 실제 Params/FLOPs/latency는 실측한다. W112/D123가 같아도 입력 구조까지 동일하다고 표기하지 않는다.
- 공통 초기 함수 통제: 기존 PAN stem weight를 8개 PAN channel에 1/8씩 복제, MS weight/bias 유지. 8개 PAN이 같으면 원래 9ch stem과 같아야 한다.
- 16개 head는 zero 초기화하되 테스트에는 반드시 nonzero head/residual을 사용한다.

밴드별 absolute loss:

`Labs_B8 = mean_k,ROI,xy |D(W(P,u_k)) − D(GT_k)|`.

밴드 평균으로 나누므로 8배 loss 계수를 만들지 않는다. 8개 PAN을 평균해 다시 한 장으로 U-Net에 넣는 우회는 금지한다.

Stage 2 Hard/Soft는 기존대로 모든 MS 밴드를 사용한다. q/g의 RGB 기준은 유지하고, q용 PAN만 `mean(P_[4,2,1])`로 명시한다. **새 참조 연산이므로** bandwise donor·mask·geometry에 맞는 full-train calibration을 다시 한다. 모든 u_k가 같으면 글로벌 PAN q와 일치해야 한다.

## 7. FR-scale 합성 증강 — 이번 계획의 명시적 설계 기본값

다음 수치와 보조 분기 방식은 30h 계획을 위해 선택한 설계 기본값이며, 데이터에서 최적이라고 입증된 설정이 아니다. 사용자 확인은 16ch PAN 복사본 구조에 대한 것이다.

### 7.1 이동 분포

RR train patch에서만, 공유 flip/rotation을 마친 현재 PAN 격자 기준으로:

- 공통 이동 `c ~ uniform-area disk(R=1.5 px)`.
- 밴드별 잔차 `r_k ~ independent uniform-area disk(R=0.5 px)`.
- `δ_k = c + r_k`. 최대 이동 2px, 밴드 간 추가 차이 최대 1px.
- 원판 면적 균등은 반지름 `R·sqrt(U)`, 각도 균등으로 생성한다.
- FR20에서 밴드별 평균·방향·분포를 fit하지 않는다. 특정 MS1/MS2 방향을 hardcode하지 않는다.
- `FR-scale`은 이동 크기에 대한 합성 robustness 실험이라는 뜻이다. FR 영상이나 FR GT 학습이 아니다.

독립적인 밴드별 R2 원판은 두 밴드 사이 최대 4px 차이를 만들기 때문에 첫 실험에는 쓰지 않는다.

### 7.2 안전한 auxiliary 경로

raw LR16에 fractional shift를 적용한 뒤 넓은 interp23tap을 쓰면 경계 영향이 기존 HR warp MASK2로 보장되지 않는다. 이를 피하기 위해 **제공 LMS/GT의 HR Aligner-only 보조 분기**로 구성한다.

- 정상 분기: 원 RR PAN·LRMS·제공 LMS·GT로 U-Net reconstruction을 그대로 학습.
- 보조 분기: `LMS'_k=W(LMS_k,δ_k)`, `GT'_k=W(GT_k,δ_k)`, **PAN은 그대로**.
- 보조 분기에서 raw LRMS 또는 U-Net을 호출하지 않는다. 이를 정상 reconstruction batch로 위장하지 않는다.
- MS별 독립 이동은 16출력 분기에만 적용한다. 하나의 글로벌 shift가 서로 다른 8개 δ를 동시에 맞추도록 요구하지 않는다.
- PAN-only epsilon과 MS-side δ는 독립 분기로 계산하며 같은 입력에 겹쳐 warp하지 않는다.

현재 backward sampling `W(I,u)(x)=I(x+u)`에서는:

- PAN-only: `u_eps = u − epsilon`.
- MS-side: `u_MSaug,k = u_k + δ_k`.
- MS consistency: `mean |u_MSaug,k − δ_k − stopgrad(u_k)|`.

부호를 기존 PAN epsilon 코드에서 그대로 복사하면 틀린다. 만약 나중에 raw LRMS까지 이동하는 reconstruction 증강을 추가하면 LRMS에는 δ/4, LMS/GT에는 δ를 적용해야 한다. 이 /4는 격자 단위 변환이지 FR Aligner 출력에 1/4을 적용한다는 뜻이 아니다.

### 7.3 B8A 분기 손실과 해석 제한

B8A의 계획 loss는:

`Lrec_native + 2·mean(Labs_native, Labs_MSaux) + .003·mean(LepsPAN, LeqMS)`.

전역 계수는 2/.003으로 유지하지만 native absolute 기여는 1, PAN epsilon 기여는 .0015가 된다. **기존 loss와 동일한 augmentation-only 대조가 아니며 δ=0에서 기존 전체 loss와 정확히 같지도 않다.** 이 분기는 head·stem·per-band objective·robustness 학습을 묶은 탐색이다. 성공하더라도 개별 구성 요소의 기여를 분리해 주장하지 않는다. `Band8 no-aug` 대조는 다음 캠페인 우선 과제로 남긴다.

지원 영역은 aux LMS의 Aligner crop8, GT Scharr stencil7, 각 예측 PAN warp를 모두 검사한다. 예측 u는 무제한이므로 δ≤2라는 이유만으로 유효성을 가정하지 않는다. ROI8 위반은 optimizer update 전에 실패하며, 이동 clamp·자동 ROI 변경으로 덮지 않는다. reconstruction/Hard/Soft mask는 8개 support의 교집합을 사용해 기존 sample별 평균 방식을 보존한다.

## 8. 출력 좌표·배율 대조

`d=s·u`이며 RR/FR 모두 현재 PAN 픽셀 단위다. 자동 ×4 또는 ¼ 없음.

| G8 frame | PAN 입력 | MS base 입력 | 출력 / 학습 목표 |
|---|---|---|---|
| PAN | W(P,d) | B | 원 MS 좌표, 원 GT/Teacher |
| MID | W(P,d/2) | W(B,−d/2) | midpoint, GT/Teacher도 −d/2 |
| MIDRET | W(P,d/2) | W(B,−d/2) | 출력에 +d/2 역warp, 원 GT/Teacher |

MID의 두 입력이 반씩 이동하는 것은 전체 보정량 0.5배가 아니다. full MID의 상대 보정은 u이다. 출력 inverse는 학습 autograd 내부와 평가에 동일하게 적용한다.

B8A는 full per-band PAN→MS만 훈련한다. bandwise midpoint·MS-only·출력 역warp는 이번 범위에서 제외한다. HQNR의 half-shift 선호는 G8 S050으로 먼저 비교한다. B8A half-scale 추론을 추가한다면 **추론-only 진단**으로 별도 표기하고 full-scale 학습 결과나 다른 checkpoint 선택 규칙으로 섞지 않는다.

## 9. 공식 평가와 밴드별 잔여 정합

- 공식 metric은 원본 RR20/FR20 전체·unmasked·참조 이동 없음. Dλ, Ds, HQNR, RR ERGAS/SCC 및 RMSE·CC·JQM 등을 전부 보존한다.
- 현재 workbook `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE` / `WV3-ablations`.
- 목표선 미달을 포함해 **모든 정규 공식 평가 step**을 올린다. 로컬 동일 record/outbox·scene별 metric·full precision JSON을 보존하고 시트 표시 precision은 기존 fixed64 규칙을 따른다.
- 로그 HQNR/SCC/ERGAS는 필수. Aligner shift는 band×dy/dx 전체와 실제 적용 배율을 남기며 평균 shift 하나로 축약하지 않는다.
- 별도 정합 진단은 `ta2.registration.register_band`로 원 PAN↔각 제공 LMS 밴드와 실제 warp 후 PAN↔LMS를 다시 측정한다. RR-GT 비교는 RR 전용 별도 진단이며 FR GT 사용은 금지한다.
- 동일한 전후 유효 장면 집합에서 밴드별 잔여 dy/dx·norm median/p90, MS1=[1,2,4,6]/MS2=[0,3,5,7] 그룹, valid count/failure reason/peak/gap/boundary를 함께 보고한다. 실패 fallback=0을 정합 성공으로 집계하지 않는다.
- 추정 잔여는 물리적 참값이 아니다. 합성 δ를 아는 equivariance 오차와 자연 영상 정합 추정 오차를 구분한다. 기존 proxy 표의 “0/0” 역시 실제 학습 모델의 잔여 0을 보장하지 않는다.
- frozen A가 같은 Student끼리는 정합 진단을 같은 A hash·geometry·배율·원자료 hash로 재사용할 수 있다. 출력 SR의 공식 metric은 모델별로 각각 계산한다.

## 10. 비교 해석과 구현 체크리스트

1. s1 G8과 s3 RGB 결과 차이는 **서버 × 정렬 목표가 함께 바뀐 비교**다. 같은 seed라고 all-band 효과의 독립 증거가 되지 않는다. 원인 분리를 위한 s1 RGB/U23 matched control은 이번 30h 밖의 후속이다.
2. 같은 s1/G8 donor 안에서 PAN↔MID↔MIDRET를 비교하면 좌표 처리 효과를 비교할 수 있다.
3. B8A와 G8 차이는 구조+목표+augmentation 묶음 효과다. 각 요소나 augmentation 단독 개선이라고 주장하지 않는다.
4. 새 namespace(예: `panda_band30`)와 새 checkpoint schema를 쓴다. s3 `panda_warp30` 및 공통 모델 클래스·global registry를 덮어쓰지 않는다.
5. 새 code/evaluator/source/upsampler/head/stem/geometry/augmentation/loss-reduction/calibration 계약을 checkpoint·resume·reporting·cost에 고정한다.
6. 필수 검증: nonzero head에서 밴드 routing, 모든 shift가 같을 때 global 경로와 stem/q 동등성, 단일 밴드 δ의 독립 반응·정확한 부호, full48와 micro12×4 일치, 새 augmentation RNG exact resume, A/T 동결, inverse gradient, RR256/FR512 GT-free 추론, 별도 calibration 거부, 공식 원본 참조, 시트 64열·전체 step 업로드.
7. 새 bandwise 경로의 실제 속도·support·gradient 검증을 완료하기 전에는 이 명세를 “실행 가능 구현 완료”로 취급하지 않는다.

## 11. 이번 산출물 범위

case 계획 MD·JSON·CSV를 만들었다. PAN 8복사본/16ch는 사용자 승인 사항을 반영했다. 증강의 구체적 분포와 auxiliary-only 방식은 위에 명시한 구현 설계 기본값이다. **학습 코드 이식·GPU 실험 기동·commit/push는 이번 case 구성 작업에서 수행하지 않았다.**
