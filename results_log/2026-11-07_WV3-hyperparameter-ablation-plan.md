# WV3 하이퍼파라미터 ablation 설계안 — s1 / s3

2026-11-07 초안, 2026-11-08 사용자 선택으로 기준 확정. **구현 후 s1 학습 시작**.
실행·검증·s3 인계는 [2026-11-08 인계 기록](2026-11-08_WV3-hyperparameter-s1-s3-handoff.md)을 따른다. 아래는 승인된 설계 근거다.
설계 데이터: [proposal.json](assets/2026-11-07-wv3-hyperparameter-design/proposal.json),
[s1 case 순서](assets/2026-11-07-wv3-hyperparameter-design/cases_s1.csv),
[s3 case 순서](assets/2026-11-07-wv3-hyperparameter-design/cases_s3.csv).

## 1. 확정된 비교 기준

사용자 선택: **기존 WV3 기준 유지 — W112, 현재 warp 경로로 계수 효과만 비교**.
이번 민감도 실험의 기준은 **WV3 / RGB / W112**다.

| 경로 | 확정 설정 |
|---|---|
| Stage1 Teacher 구조 | W112/D123/BASE |
| Stage1 native 보정·controlled shift 주입 | bicubic-border |
| Stage2 Teacher 입력 | bicubic-border |
| Stage2 Student 구조·입력 warp | W112/D123/UBIL, Lanczos4-reflection |
| PAN Aligner scale / soft loss β | 1 / 0 |

기존 RGB W112 결과 중 설정·seed·donor·평가 조건이 정확히 일치하는 대조군을 재사용한다.
이 선택은 이번 하이퍼파라미터 ablation에 적용하며, Lanczos 통일이나 W128 구조 변경을
같은 계수 곡선에 섞지 않는다. s1은 로컬 RGB donor를 사용하고 s3는 Stage1 계수 실험을 맡는다.
서버 간 weight 이동은 없다.

실행 registry와 CSV는 `config/panda_wv3_hyper/`에 구현됐다. 기존 s1 대조군9회 재사용 검증을 완료했다.

## 2. 기존 근거

같은 RGB donor/Student 반복, LR1e-4, C30, W112, 고정29K의 3회 평균±표본표준편차:

| 기존 case | α | γ | HQNR | ERGAS |
|---|---:|---:|---:|---:|
| M0_OURS | .25 | .25 | .958443 ± .000550 | 2.043507 ± .003521 |
| M1_G0125 | .25 | .125 | .959720 ± .000831 | 2.044375 ± .001109 |
| M2_A0125 | .125 | .25 | .958477 ± .001230 | 2.041855 ± .002151 |

γ를 낮추면 HQNR은3/3 반복에서 개선되었지만 ERGAS 방향은 일정하지 않았다.
α=.125의 ERGAS 차이도 seed별 부호가 다르다. .125–.25 사이만 촘촘히 반복하기보다
0부터1까지 넓혀 민감도와 성능이 나빠지는 구간을 확인한다.
HARD36은 등록 코드와 계획은 있으나 로컬 실행 결과가 없어 완료 실험으로 세지 않는다.

기존 mean8 G8 donor 기반 component/TUNE30 결과를 새 RGB 곡선의 반복으로 합치지 않는다.
기존 numerical evaluation 재사용과 weight 보관 상태는 구분한다. M0/M1/M2의
29K/30K 평가 JSON은 모두 남아 있지만 일부 Student weight는 이전 정리로 삭제되었다.
재사용 receipt에는 원본 config·donor/calibration·seed·평가기·고정 시점 평가의 해시를 기록한다.
Student weight가 남아 있으면 실제 파일 SHA를 검증하고, 삭제됐다면 `METRICS_ONLY_WEIGHT_PRUNED`로
표시하여 수치만 재사용한다. 삭제된 weight로 새 추론까지 재현했다고 표시하지 않는다.

## 3. s3 — Stage1 λ_spa와 λ_ε

목적은 정합 supervision과 controlled-shift consistency의 강도에 대한 민감도를 보이는 것이다.
코드의 `lambda_abs`는 원고의 `lambda_spa`에 해당한다.

`L_S1 = L_rec + lambda_spa L_spa + lambda_epsilon L_epsilon`

| Case | λ_spa | λ_ε | 구분 |
|---|---:|---:|---|
| H1_REF | 2 | .003 | RGB A5_ALL 기준점 |
| H1_SPA0 | 0 | .003 | RGB A4_REC_EPS 대조 |
| H1_SPA05 | .5 | .003 | spatial 약한 가중 |
| H1_SPA1 | 1 | .003 | spatial 기준의1/2 |
| H1_SPA4 | 4 | .003 | spatial 기준의2배 |
| H1_SPA8 | 8 | .003 | spatial 강한 가중 |
| H1_EPS0 | 2 | 0 | RGB A3_REC_SPA 대조 |
| H1_E0003 | 2 | .0003 | consistency 기준의1/10 |
| H1_E001 | 2 | .001 | consistency 기준의1/3 |
| H1_E010 | 2 | .01 | consistency 기준의약3.3배 |
| H1_E030 | 2 | .03 | consistency 기준의10배 |

공통 기준점은 한 번만 실행한다. **11설정×3회=33 runs 상당**이다.
원래 s3 RGB A3/A4/A5가 같은 seed/config로 정상 완료되고 고정50K 근거가 있으면
9회를 재사용하고 **신규24회**만 학습한다. s1에는 해당 s3 학습 근거가 없어
현재 완료됐다고 단정하지 않는다. 원격 검증 후 누락된 대조군만 새로 학습한다.

- Aligner+Teacher fresh 학습, W112/D123/BASE, 50K, A LR1e-5/T LR1e-4.
- R_ε=3px 원판 면적 균일 분포. Aligner 출력 범위나 scale을3으로 제한한다는 뜻이 아니다.
- L_spa 참조는 GT RGB `[4,2,1]` 평균의 Scharr. Aligner 입력 LMS와 Teacher 복원은8밴드 전체.
- seed는261016101/102/103. 같은 반복에서 초기화·데이터·augmentation·ε stream을 공유한다.
- 비교 주 지표: 고정50K의 RGB RRE50(RR/FR), FR D_lambda, Teacher RR ERGAS/SAM/PSNR.
- 진단: validation RRE90/실패율, 공통 ε probe의 EPE50/gain, predicted-u dy/dx 표준편차.
  shift 표준편차만으로 상수 수렴 여부를 확정하지 않는다.
- validation-selected donor는 기존 RRE50→RRE90→ValERGAS→이른 step 규칙으로 따로 기록한다.
  Stage1 계수 곡선의 주 표는 모든 case의 동일50K다.

이 비교는 Stage1 정합/Teacher 성능에 대한 결과다. 이 표만으로 최종 Student 개선을 주장하지 않는다.
최종 Stage2 영향이 필요해지면 s3에서 해당 로컬 donor별 동일 Student recipe를 별도 후속 평가한다.
현재 핵심안에는 s3 Student 학습을 넣지 않는다.

## 4. s1 — Stage2 α와 γ

`w_AF = 1 + alpha d_e + gamma d_e r_q`, β=0.

| Case | α | γ | 구분 |
|---|---:|---:|---|
| H2_REF | .25 | .25 | 기존 M0_OURS 재사용 |
| H2_A0 | 0 | .25 | 직접 d_e 항 제거 |
| H2_A0125 | .125 | .25 | 기존 M2_A0125 재사용 |
| H2_A050 | .5 | .25 | α 기준의2배 |
| H2_A100 | 1 | .25 | α 기준의4배 |
| H2_G0 | .25 | 0 | d_e r_q 항 제거 |
| H2_G0125 | .25 | .125 | 기존 M1_G0125 재사용 |
| H2_G050 | .25 | .5 | γ 기준의2배 |
| H2_G100 | .25 | 1 | γ 기준의4배 |
| H2_GT | 0 | 0 | 동일 정합·residual 입력의 uniform GT loss |

독립 sweep9점과 두 항 모두 제거한 대조1점으로 **10설정×3회=30 runs 상당**이다.
정확한 기존3설정×3회=9회를 재사용하면 **신규21회,630K Student updates**다.
이21회는 재사용 검증 통과를 전제로 한다. 검증에 실패한 기존 대조군은 해당 반복만 새로 학습하며,
따라서 실제 신규 실행 수는21–30회다. 기존 Student weight 삭제만으로 수치 재사용을 거부하지는 않는다.

- α=0이어도 γ>0이면 `gamma d_e r_q`를 통해 Teacher difficulty가 남는다.
  Teacher guidance 전체 제거 대조는 H2_GT다. PAN Aligner와 residual 입력은 유지한다.
- 각 신규 Student는 fresh 초기화. W112/D123/UBIL, LR1e-4, cosine horizon30K.
- 같은 반복의 모든 case는 같은 s1 RGB donor·calibration·Student 초기화·학습 순서를 공유한다.
  R1/R2/R3 donor는 기존 validation 선택43K/49K/38K다. 다른 서버 weight를 받지 않는다.
- Student seed261018101/102/103. 반복 간에는 donor와 Student seed가 함께 달라지므로
  3회 분산을 순수 Student seed 효과라고 부르지 않는다. 계수 효과는 같은 반복끼리 paired 비교한다.
- loss는 기존대로 유효 pixel 수로 나눈다. 가중치 합으로 재정규화하지 않고 LR도 보정하지 않는다.
  계수 변화가 전체 gradient 규모에도 영향을 주므로 weight_mean, raw/weighted loss, gradient를 함께 기록한다.
- α=γ=0에서도 기존 Teacher/Student support 교집합과 입력 경로를 유지한다.

## 5. 비교와 순서

- 모든 설정은 **R1 전체 → R2 전체 → R3 전체** 순서다. 재사용 가능한 행은 근거 검증 후 완료 표시한다.
- Stage1 고정50K, Stage2 **고정29K가 주 비교**다. Stage2 horizon은30K이며 최종30K는 보조다.
- case별 평균±표준편차와 같은 반복의 기준점 대비 차이를 기록한다. 작은 ERGAS 차이는 순위만으로 과장하지 않는다.
- test 최고 checkpoint를 case마다 골라 계수 곡선을 만들지 않는다. run 대표값/목표 충족률은 별도 보조다.
- 고정 학습률·스케줄·width·batch·warp를 유지한다. scale1, β0, 회전 consistency/TTA/SWA/EMA/밴드 가중 loss 제외.
- 공통 AdamW WD.01, batch48/micro12, warmup100→cosine0, FP32, 기존 paired augmentation.
- 성능 미달로 조기 종료하지 않는다. 수치/support 실패는 실패로 기록하며, 임의 shift clamp로 성공 처리하지 않는다.
  유효 반복 수가3보다 작으면 실패를 누락하거나 다른 seed로 대체해3회 평균처럼 표시하지 않는다.

신규 우선순위:

1. s1은 α0/γ0/둘 다0 대조와 .5 설정, 이후1.0 설정.
2. s3는 중심 주변 λ_spa1/4, λ_ε.001/.01, 이후 .5/8와 .0003/.03.
3. 계수 효과를 확인한 뒤 아래 후속 축을 판단한다. 첫 R1만 보고 전체 반복을 취소하지 않는다.

## 6. 후속 축 — 핵심 큐에 자동 포함하지 않음

| 후속 | 서버 | 설정 | 추가 runs |
|---|---|---|---:|
| controlled shift 범위 | s3 | R_ε=4px; 나머지 기준 유지 | 3 |
| α·γ 상호작용 | s1 | (.125,.125), (.5,.5) | 6 |
| 조건부 rank 그룹 수 | s1 | K=1/3/10; 기준K5 유지 대조 | 9 |

R_ε는 과거 합의대로3→4만 우선 확인한다. K=1은 global rank 대조이고,
K를 바꾸면 같은 donor에 대해 RR train 전체 calibration을 새로 만든다.
Teacher-error τ_e는 데이터에서 산출하는 통계이므로 이번에 임의 수동 상수 sweep으로 바꾸지 않는다.

## 7. 예상량과 게시

기존 경로·대조군 재사용이 모두 확인되면 신규 s1 Stage2 21회, s3 Stage1 24회다.
s1 최근 실측 기준 Stage1 50K 약2.6–2.8h, W112 Stage2 30K 약1.7–1.9h이므로:

- s1 신규21회 약36–40h.
- s1 기존 대조군9회가 모두 재사용 불가라면30회, 약51–57h.
- s3 신규24회는 **s1 속도로 환산하면 약62–67h**. s3 실측 시간이라는 뜻이 아니다.
- s3 대조군이 모두 누락됐다면33회, 같은 환산 약86–92h. calibration/사전검사 시간을 별도로 둔다.

게시 위치는 `WV3-ablations`. Experiment 예는 `WV3-HYP-H2-A050 R1 (RGB)`처럼 case를 드러낸다.
Approach에 λ_spa/λ_ε 또는 α/γ, scale, warp, width, LR, horizon, donor 식별자를 담는다.
소수점4자리 표시는 하되 판정과 평균은 원본 정밀도를 사용한다.

- Stage1 기존 validation 대표 규칙, Stage2 HQNR≥.9585·ERGAS≤2.040 동시 충족 + 대표1행 규칙 유지.
- 논문용 **FIXED50K/FIXED29K 행**은 대표와 구분해서 게시한다. 같은 checkpoint이면 행을 합치고 용도를 함께 표시한다.
- fixed 결과는 목표 미달이어도 누락하지 않는다. 이전 다른 캠페인 행/확장 열은 유지한다.
- infer/memory/FLOPs는 동일 추론 구조일 때 근거와 범위를 명시해 공유하고 학습시간은 실제 run 값으로 기록한다.

완료 후 기존 ablation 보관 정책을 적용할 수 있지만, 고정29K/30K/50K,
validation-selected donor, 참조 weight/calibration, 진행 중 resume은 보존한다.
원래30K 실험의≤20K,50K 실험의≤30K 정리는 이러한 참조 예외를 먼저 확인한 뒤 수행한다.
하이퍼파라미터 목표 미달만으로 논문용 endpoint weight를 삭제하지 않는다.

## 8. 구현 시 필요한 구분

현재 main Stage2 wrapper는 α≤0 또는 γ≤0를 거부한다. 별도 hyper-ablation wrapper에서
zero endpoints를 지원하면서 원래 support/calibration과 positive 계수 경로를 검증해야 한다.
기존 pinned 실험 파일을 수정해 우회하지 않는다.

현재 RGB Stage1의 sample_epsilon 함수는 반경3이 상수다. 향후 R4는 config 숫자만 바꾸면
적용되지 않으므로 캠페인 내부 sampler와 분포·support·RNG 복구 검증이 필요하다.

새 registry/실행기와 재사용 receipt 구현은 완료했다. 실행 방법과 실제 시작 상태는 2026-11-08 인계 기록을 참조한다.
