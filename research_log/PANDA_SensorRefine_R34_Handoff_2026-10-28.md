# QB·GF2 우선 큐 재편 및 ε 반경 4px 후속 실험

2026-10-28 사용자 지시. s1은 QB, s3는 GF2를 담당한다. ε 반경 3px를 기준으로 유지하고 **추가 반경은 4px만** 시험한다. 기존 WV3 TUNE30/HARD36의 case·설정·seed·학습 소스는 변경하지 않는다. 시간 상한은 없다.

이 문서는 [기존 S1LOSS9 인계](PANDA_S1Loss9_Priority_Handoff_2026-10-26.md)의 실행 순서와 Git 전달 절차를 대체한다. 기존 문서의 `stage1-loss9-priority` 브랜치·bundle·인증 불가 서술을 현재 배포 상태의 근거로 사용하지 않는다. 이번 전달 대상에는 기존 `panda_s1loss9/` 전체와 새 `panda_sensor_refine/`가 함께 포함된다. 커밋·push 성공 여부와 원격 commit SHA는 실제 Git 출력으로 확인한다.

## 판단 근거

- QB의 유력 결과는 기존 LOSS9 `A200_E030`의 validation-selected Stage 1 44K → Student 29K이며 HQNR=.942697525, ERGAS=3.514884846이다. 목표는 HQNR≥.9315, ERGAS≤3.5다. 아직 동시 달성이 아니다.
- QB에서 아주 낮은 λ_ε=.0003 및 약한 절대 정합·강한 ε 조합의 우선순위를 낮춘다. 완료 결과는 삭제하지 않으며 이미 진행 중인 독립 case도 보존한다.
- GF2는 QB의 결과를 센서 최적값으로 간주하지 않는다. 기본 계수·절대 정합 약화·ε 약화 대조를 남겨 총 6개 기존 조합을 먼저 확인한다. `A200_E030`을 중심으로 하는 후속 실험은 명시적인 전이 가설이다.
- GF2의 기존 긴 스케줄에서는 54K에 HQNR=.966800717, ERGAS=.524201839였다. 따라서 새 GF2 긴 학습은 60K를 유지한다. QB의 긴 학습은 후반 개선 추세를 확인하는 40K다.
- ε 반경 4는 학습 이동량 분포에 따른 확대 후보이며 아직 성능상 최적값이 아니다. 1·2px 신규 실험은 등록하지 않는다.

## 서버별 우선순위

```text
현재 실행 중인 등록된 run을 안전하게 보존·완료
  → 유지한 기존 S1LOSS9 조합의 Stage 1 + 해당 Student
  → SENSOR_REFINE: 유력 donor 고정 Student 4개
  → SENSOR_REFINE: ε=4px Stage 1 + Student
  → SENSOR_REFINE: 손실 계수 주변 2조합 × (Stage 1 + Student)
  → 기존 WV3 TUNE30 잔여 case
  → 기존 WV3 HARD36 잔여 case
```

기존 완료 run은 완료 checkpoint·평가 증거를 확인한 후 재실행하지 않는다. 현재 실행 중인 run이 새 우선순위 목록 밖이라도 이미 시작한 학습을 무기록 폐기하지 않는다. 기존 SENSOR5의 **미시작** 후속 case만 외부 제어 receipt에 퇴역으로 기록한다. 그 캠페인의 config나 완료 기록을 고쳐 전부 완료한 것처럼 만들지 않는다.

### s1 / QB: 유지·제외

기존 LOSS9 중 `A200_E003`, `A020_E003`, `A050_E003`, `A200_E0003`, `A200_E030`, `A020_E0003`의 두 Stage는 완료 결과를 보존한다. 진행 중인 `A050_E0003`와 다음 `A050_E030`의 두 Stage를 유지한다.

`A020_E030` Stage 1은 11,197 update에서 native support 위반으로 실패했다. 해당 실패 기록과 checkpoint를 보존하고, 그 donor에 의존하는 Student는 보류한다. 실패 Stage 1을 완료 모델처럼 재사용하지 않는다.

옛 SENSOR5 donor를 사용하는 미시작 QB `LR050`, `LR200`, `LONG`, `HARD4` 4개는 퇴역시키고 아래 새 donor 기반 4개로 대체한다. 이미 실행한 SENSOR5 결과는 그대로 남긴다.

### s3 / GF2: 유지·제외

기존 LOSS9 전체를 새로 9조합 실행하지 않고 다음 6조합을 순서대로 유지한다. 각 조합은 새 Stage 1 50K → 해당 validation-selected donor의 Student 30K다.

| 우선순위 | 기존 pair | λ_abs | λ_ε | 유지 이유 |
|---:|---|---:|---:|---|
| 1 | A200_E030 | 2 | .03 | QB 유력 중심을 GF2에서 검증 |
| 2 | A020_E003 | .2 | .003 | QB의 다른 HQNR 유력 축 |
| 3 | A050_E030 | .5 | .03 | 절대 정합 약화와 강한 ε의 중간점 |
| 4 | A200_E003 | 2 | .003 | 기존 계수의 동일 캠페인 대조 |
| 5 | A050_E003 | .5 | .003 | 절대 정합 계수의 중간점 |
| 6 | A200_E0003 | 2 | .0003 | GF2의 ε 약화 단일 요인 대조 |

미시작 `A020_E0003`, `A050_E0003`, `A020_E030`은 제외한다. 전자의 두 상호작용은 QB에서 유망하지 않았고, 마지막 조합은 실패 전력이 있다. 이 판단은 GF2에서 나쁨이 확정됐다는 뜻이 아니라 계산 우선순위 조정이다. s3에서 이미 실행된 case가 있으면 그 결과와 이력을 보존한다.

## 새 SENSOR_REFINE 등록 case

캠페인: `PANDA_S13_SENSOR_REFINE_R34_20261028_v1`, run prefix: `PSRF`.

첫 4개 Student는 **같은 서버·같은 센서의 기존 LOSS9 `S1_A200_E030`**에서 validation RRE 기준으로 선택한 A/T를 동결한다. QB의 44K를 GF2에 강제하지 않고, GF2 자신의 validation-selected checkpoint를 사용한다. 기존 donor의 checkpoint·선택 receipt·binding을 인증하고 새 Student를 처음부터 학습한다. Stage 2 weight를 가져오거나 중간 checkpoint에서 fine-tuning하지 않는다.

| 순서 | s1 / QB | s3 / GF2 | 변경점 |
|---:|---|---|---|
| 1 | QB_BASE R2, 30K, LR=1e−4 | GF2_BASE R2, 30K, LR=1e−4 | 동일 donor에서 새 Student seed 반복 |
| 2 | QB_LR120 R1, 30K, LR=1.2e−4 | GF2_LR150 R1, 30K, LR=1.5e−4 | 센서별 LR 증가 |
| 3 | QB_C40 R1, 40K, LR=1e−4 | GF2_C60 R1, 60K, LR=1e−4 | 처음부터 해당 길이의 cosine |
| 4 | QB_H050 R1, 30K | GF2_H050 R1, 30K | α=.125, γ=.25로 Hard 강조 절반 |
| 5–6 | S1_A200_E030_R4 → QB_A200_E030_R4 | S1_A200_E030_R4 → GF2_A200_E030_R4 | λ_abs=2, λ_ε=.03, ε 반경 **4** |
| 7–8 | S1_A200_E010_R3 → QB_A200_E010_R3 | S1_A200_E010_R3 → GF2_A200_E010_R3 | λ_abs=2, λ_ε=.01, ε 반경3 |
| 9–10 | S1_A100_E030_R3 → QB_A100_E030_R3 | S1_A100_E030_R3 → GF2_A100_E030_R3 | λ_abs=1, λ_ε=.03, ε 반경3 |

각 신규 Stage 1은 50K, 해당 Student는 30K다. 서버별 신규 10 run은 Stage 1 3개 + Student 7개이며, update 합계는 QB 370K, GF2 390K다. validation donor 선택·전수 calibration·평가·cost 측정 시간은 update 수에 포함되지 않는다.

공통 설정: Aligner scale=1, RGB 구조 참조, 모든 PAN warp Lanczos-4, MS 입력/출력 전체 4밴드, interp23 MS base, Student `[PANc, PANc−PAN, I_ms^lr]` 6채널, W112D123 UBIL. 별도 기재가 없으면 Stage 2 α=.25, γ=.5, β=0, LR=1e−4, warmup100 후 cosine이다. Stage 1 LR_A=1e−5, LR_T=1e−4, reconstruction 계수1이다. rotation loss·TTA·SWA/EMA·밴드 가중 loss는 사용하지 않는다.

## ε 반경과 경계 오류 처리

ε는 **RR PAN pixel 단위의 원판 면적 균등 분포**다. Aligner의 native 예측값 u에 걸리는 clamp나 scale이 아니다. R4 sampler가 실제로 반경4를 사용하며 config·manifest·Teacher binding·checkpoint·Sheet 설명이 같은 반경을 가리켜야 한다. 기존 R3 donor를 R4 donor로 재표기하지 않는다.

이번 비교에서는 Aligner view margin8과 native loss ROI margin8/stencil7을 유지한다. R4+Lanczos4의 합성 입력 경계는 margin8 안에 들어오지만, **native u는 별도로 support 검사를 통과해야 한다.** ε 범위 확대가 native support 위반을 자동으로 해결하지 않는다.

예측값 clamp, 동적 loss 영역, border fallback 등으로 학습 목적함수를 조용히 바꾸지 않는다. 알려진 support 위반은 실패 run과 저장 상태를 기록하고 해당 donor 의존 Student를 보류한 뒤 독립 case를 계속한다. checkpoint가 손상됐거나 optimizer가 poisoned 상태인 경우, OOM·NaN·코드/데이터 불일치 같은 다른 오류는 정상 완료로 처리하지 않는다. 실패·보류가 있는 큐는 전체 완료와 구별되는 종료 상태를 남긴다.

특히 GF2의 기존 `S1_A200_E030` 중심 donor 자체가 인증된 support 위반으로 실패하면, 그 donor에 의존하는 새 `BASE R2`·`LR150`·`C60`·`H050` 4개 Student를 보류한다. 별도 donor를 학습하는 R4 및 손실 계수 2조합의 총 6개 run은 계속 실행한다. 다른 donor로 임의 교체하거나 실패한 중심 조합을 완료했다고 기록하지 않는다. QB에도 같은 의존성 규칙이 적용된다.

본 학습 이전의 disposable GPU preflight에서 같은 support 위반이 발생한 경우도 별도로 구분한다. 실제 run 디렉터리·상태가 아직 없는 경우에만 `SENSOR_REFINE_PREFLIGHT_SUPPORT_FAILURE_V1` receipt를 만들며 `FAILED_PREFLIGHT_NO_PRODUCTION_UPDATES`, `production_updates=0`, `production_weights_started=false`를 기록한다. 요청 config·code pin·receipt SHA를 묶고 Student라면 실제 donor/selection의 출처도 기록한다. 이미 학습한 run을 이 방식으로 0-step 실패로 바꾸지 않는다. 해당 Stage 1의 Student는 보류하고 독립 case는 계속하되, 최종 상태는 `FINISHED_WITH_EXPLICIT_FAILURES`로 정상 전수 완료와 구별한다.

## 설치와 실행

기존 학습 checkout의 파일과 checkpoint를 유지하기 위해 새 구현은 `panda_sensor_refine/`로 분리했다. 이번 변경 전의 LOSS9/TUNE30/HARD36 source pin을 [통합 테스트](../tools/test_sensor_refine_integration.py)에서 대조한다. `git pull` 자체는 실험을 시작하지 않는다.

s3의 기존 RGB/L4 SENSOR5가 준비된 저장소에서 다음을 실행한다. `PAN_HOST_PYTHON`은 torch와 Google Sheets 인증 라이브러리가 있는 기존 학습·게시 Python의 실제 경로로 지정한다. 기존 s3 코드가 `31c9f75`보다 오래되어 RGB/L4 전환이 안 되어 있다면 먼저 [SENSOR5 인계](PANDA_Sensor5_Stage12_Handoff_2026-10-26.md)의 전환 조건을 충족해야 한다.

```bash
git pull --ff-only origin main
export PAN_HOST_PYTHON=/실제/s3/학습환경/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$PAN_HOST_PYTHON" -m panda_sensor_refine catalog --server s3 --csv
"$PAN_HOST_PYTHON" -m unittest discover -s panda_sensor_refine/tests -v
"$PAN_HOST_PYTHON" -m unittest discover -s tools -p test_sensor_refine_integration.py -v
bash scripts/enqueue_panda_sensor_refine.sh s3
```

이 서버는 같은 launcher에 `s1`을 사용한다. 새 controller는 기존 controller의 실제 프로세스·container identity를 확인하고 제어권을 인계한다. **SENSOR5가 이미 완료됐고 S1LOSS9가 아직 없는 s3도 지원**하며, 옛 LOSS9 launcher가 요구하던 `WAIT_SENSOR5` 상태를 수작업으로 위조할 필요가 없다. 기존 active run은 완전한 optimizer·RNG·data stream 상태로 보존한다.

원격에서 `pull → catalog → 테스트 → enqueue` 후 controller status, 실제 trainer step 증가, Sheet readback을 확인해야 s3 실행을 확인했다고 할 수 있다. s1에서 commit/push만 성공한 것을 s3 학습 시작으로 표현하지 않는다.

## Sheet 업로드 규칙

| 서버 | 데이터셋/탭 | HQNR 최소 | ERGAS 최대 |
|---|---|---:|---:|
| s1 | QB / QB-ablations | .9315 | 3.5 |
| s3 | GF2 / GF2-ablations | .9645 | .522 |

- 같은 checkpoint의 원본 FR20/RR20 결과로 판정하며 경계값은 포함한다.
- Stage 1은 validation-selected RRE donor 대표를 게시한다. test HQNR로 Stage 1 donor를 고르지 않는다.
- Student는 두 목표를 동시 충족한 step 전부와 run별 대표를 게시한다. 대표는 HQNR 조건 충족 중 ERGAS 최소이며, HQNR 조건 충족이 없으면 HQNR 최고다. 대표가 이미 통과 step이면 한 행만 게시한다.
- 접근 설명에 **ε 반경, λ_abs, λ_ε, Stage 2 LR/horizon·α·γ·β=0, scale1, RGB/L4, 입력 구성**이 보여야 한다.
- PSRF 행과 이전 PSL9/SENSOR5/WV3 행을 구분한다. 원본 평가·receipt·checkpoint SHA를 유지하고 다른 캠페인 행은 덮어쓰지 않는다.
- 옛 SENSOR5 config의 GF2 `.548`은 역사 기록이다. 새 LOSS9/SENSOR_REFINE의 판단은 `.522`다. 과거 publisher의 정책 갱신은 해당 원본 증거를 확인한 별도 작업이며 기존 training code를 고쳐 맞추지 않는다.

## 검증 범위

통합 테스트는 두 서버의 10-run 구성·센서별 스케줄, R3/R4 구분, 원래 WV3/LOSS9 source pin 불변, 실제 publication projection의 목표/대표 선정, 반경·계수 표시, 다른 캠페인 행 보존 및 쓰기 없는 dry-run을 확인한다. 가짜 수치를 실제 시트에 올리지 않는다. 실제 학습은 run 시작 시 GPU update/resume preflight를 통과해야 하며, 미시작 run의 GPU 검증이 완료됐다고 주장하지 않는다.

오류 처리 검증에는 preflight-only 실패와 실제 학습 중 실패의 증거 구분, 이미 시작한 run의 preflight 실패 재분류 거부, 실패 donor에 종속된 Student만 보류하는 규칙을 포함한다. 이 검증은 모든 case가 반드시 끝까지 학습된다는 보장이 아니다. 미시작·완료·실패·보류를 남김없이 계산하고, 학습 오류를 감춘 정상 완료나 전체 큐의 무기록 중단을 피하는 것이 목적이다.

## 2026-10-28 추가: QB Student 10K 조기 종료

사용자 승인 정책 `USER_20261028_QB_S2_HQNR_8_9_10K_ALL_LT_090`을 현재 retained LOSS9와 이후 SENSOR_REFINE의 **QB Stage 2**에 적용한다. 위 launcher를 통해 실행해야 원본 LOSS9 코드의 변경 없이 제어 계층에서 적용된다.

- 정확히 8K·9K·10K의 공식 FR20 평균 HQNR이 **모두 < .90**이면 10K 평가/checkpoint 저장 후 Student를 종료하고 다음 독립 case로 진행한다. 하나라도 .90 이상이면 계속한다. 1K 값만으로 중단하지 않는다.
- 평가 누락은 중단 판정이 아니다. 같은 run/attempt/seed, RR20/FR20 전수, 동일 모델 상태의 평가, 고정 추론 조건, checkpoint SHA를 확인한다. 손상·출처 불일치는 성능 조기 종료로 위장하지 않는다.
- 정책 적용 시 이미 10K를 넘긴 미완료 run은 같은 고정 8/9/10K 창으로 판단하되 실제 안전 종료 step을 기록한다. 이미 완료된 과거 run은 재분류하지 않는다.
- `EARLY_STOPPED_HQNR` 및 `attempt_01/early_stop.json`에 근거와 실제 종료 step, 완전한 optimizer/data/RNG checkpoint를 보존한다. 30K 완료로 표시하지 않는다. 기존 대표 결과와 목표 통과 step 게시 정책은 유지한다.
- REFINE 종료 상태 `FINISHED_WITH_EXPLICIT_EARLY_STOPS`는 모든 run의 완료·실패·보류·조기 종료 증거를 검증한 뒤에만 인정한다. 이후 WV3 TUNE30/HARD36으로 넘어간다.
- Stage 1, GF2, WV3에는 이 .90 규칙을 적용하지 않는다. QB 목표는 HQNR ≥ .9315 / ERGAS ≤ 3.5, GF2 목표는 ≥ .9645 / ≤ .522 그대로다.

s1 실제 적용: `QB_A050_E0003`의 8K=.802535771, 9K=.826606902, 10K=.823267386으로 조건 충족. **13,549 step**에 안전 종료했다. 다음 `S1_A050_E030`을 계속 진행하고, 해당 Student에도 동일 규칙을 적용한다. 종료 checkpoint와 두 번의 제어 경계 기록은 `work_dir/panda_sensor_refine_launchers/s1/qb_earlystop_20261028/`, `qb_earlystop_terminal_fix/`에 있다. 원본 LOSS9 admission/config/evaluator/code pin 및 WV3 코드는 변경하지 않았다. 아직 production admission이 없던 SENSOR_REFINE의 제어 코드만 갱신했다.

이미 이전 SENSOR_REFINE 버전의 code pin을 고정하여 실행 중인 다른 checkout에 이번 파일을 덮어쓰지 않는다. 특히 s3 GF2에는 QB 규칙이 해당하지 않으므로, 기존 고정 checkout의 실행을 유지하고 신규 설치 또는 미시작 캠페인에서 이 버전을 사용한다. 시작된 SENSOR_REFINE의 admission SHA를 수작업으로 바꿔 재개하지 않는다.

### QB HQNR 변동 분석

재현: `python tools/qb_hqnr_volatility.py`. [공식 평가 집계 JSON](../results_log/assets/2026-10-28_qb-hqnr-volatility.json), [학습 곡선](../results_log/assets/2026-10-28_qb-hqnr-volatility.png).

- 완료된 LOSS9 QB 6개와 미완료 1개의 모든 공식 평가에서 Aligner 가중치 SHA·추론 조건은 고정이고, FR 장면별 shift 변화는 정확히 0 px다. 변하는 것은 Student 출력이다.
- 현재 중단 case의 8K→9K HQNR 변화 +.024071 중 D_s 항은 +.023974, 10K→11K 변화 −.016669 중 D_s 항은 −.016584다. 두 구간 모두 20/20 장면이 같은 방향으로 변했다. 소수 장면 하나가 평균을 끌어간 현상이 아니다.
- 분해는 장면별 `H=(1−Dλ)(1−Ds)`에 대해 중간값 계수를 이용한 정확한 유한차분 후 평균했다. 평균 Dλ·평균 Ds의 곱으로 평균 HQNR을 대신하지 않았다. D_s는 위치뿐 아니라 PAN과 출력의 국소 평균·분산·공분산에 반응하므로, 이 수치만으로 Student의 기하 이동량을 확정하지 않는다.
- 5–15K 구간, 완료 QB 6개 평균 1K 간 절대 HQNR 변화량은 약 .01018, WV3 DIFF+UBIL C30 3개는 약 .00155다. 상승 추세를 포함한 기술 통계이며 donor·센서·Stage 1 구조 참조가 다른 참고 비교다. 공통 Student LR=1e−4, batch48, warmup100, cosine30K이므로 단순히 QB의 LR이나 batch가 더 커서 생긴 차이는 아니다.
- 전수 RGB 정합 진단의 QB train GT↔PAN 중앙값은 약 .502 RR PAN px, RR20은 .505 px, FR20 LMS↔PAN은 2.657 FR PAN px다. 해상도마다 pixel 단위가 다르므로 물리적 오정합이 5배라는 뜻은 아니다. 다만 Student가 보는 픽셀 간격과 차영상 구조가 달라져 FR에서 입력 사용 방식의 변화가 크게 드러날 수 있다.
- 기존 같은 checkpoint 입력 ablation에서 `A020_E003 @15K`는 정상 입력 H=.935809, 차영상 제거 H=.739838이었다. 반면 `A200_E003 @15K`는 .801444→.715800이었다. 원 PAN을 되찾을 수 있는 차영상의 사용 방식이 중요하다는 직접 근거다. 다른 donor는 shift 외에도 Teacher difficulty와 Hard 가중을 바꾸므로, case 간 차이를 shift 크기 하나로 설명하지 않는다.
- 초기의 큰 LR과 RR GT 목적/FR PAN 기준 D_s의 차이가 출력의 디테일·원 PAN/정합 PAN 활용 변화를 증폭한다는 해석이 가장 일관된다. 정확한 내부 이동량/밴드별 contrast 기여까지 분리한 인과 실험은 이번 분석에 포함되지 않았다. scale1에서 .9427이 나온 사실은 '완전 정합이면 QB HQNR이 반드시 붕괴한다'는 이전 일반화를 반박한다.

조기 종료 관련 테스트는 strict .90 경계, 데이터셋·phase 범위, 누락/변조/NaN/장면 중복, 안전 checkpoint와 재시작, 외부 종료 신호 구분, 완료 기록 보존, 최종 accounting·프로세스 exit·WV3 후속 전환을 검증한다. 기존 core/controller/integration을 합쳐 40개 테스트가 통과했다.
