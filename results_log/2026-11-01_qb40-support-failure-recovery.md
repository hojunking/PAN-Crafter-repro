# QB40 support 실패 분리 및 s1 재개 — 2026-11-01

## 확인된 중단

기존 WV3 `S100DIFFUBILC30G025 R1`은 30K까지 완료했다.
이어 시작한 QB40의 `QB5_S1_A200_E100 R1`은 14,219 update에서
`SUPPORT_VIOLATION: every predicted PAN view must support fixed ROI8/stencil7`로 실패했다.
컨테이너 종료 시각은 13:40:56 KST, exit code 1, OOMKilled=false다.
기존 runner는 이 예외를 전체 큐로 전파해 독립 실험도 시작하지 못했다.

## 수정 범위

- Stage 1의 알려진 support 실패만, nonpoisoned 상태·중단 checkpoint SHA·학습 trace·설정이
  일치할 때 보류한다. 실패 row를 완료로 바꾸거나 부분 Stage 1을 donor로 선택하지 않는다.
- 해당 Stage 1을 요구하는 Student만 `BLOCKED_FAILED_STAGE1`로 남긴다.
- 독립 실험은 기존 순서대로 진행한다. 이후 같은 오류가 발생해도 같은 정책을 적용한다.
- 종료 집계에서 완료/early stop/실패/의존 보류/시간 부족 미실행을 구별한다.
  실패 branch가 있으면 `FINISHED_WITH_FAILED_BRANCHES`로 종료하며 모든 case 성공으로 표시하지 않는다.
- 알려지지 않은 오류, optimizer 부분 실패, checkpoint 저장 실패는 여전히 전체 중단한다.
- ROI, warp, loss, Aligner scale, LR, seed, 데이터, 평가 및 과학적 선택 규칙은 그대로다.

## 보존과 복구

원래 `admission.json`, 실패 run 및 1K–14K 평가 weights, 14,219 update 중단 weights를 보존했다.
중단 checkpoint SHA256:
`096e3c22dde5729f74231dbd9cd1616938eb46c90731895bb288423911477f5e`.

`control_recovery.json`으로 제어 코드 변경을 명시하고, 기존 실패 상태와 컨테이너 로그를
`recovery_before_control_v2/`에 복사했다. 새 native admission은 `admission_control_v2.json`이다.
기존 checkpoint의 소스 pin을 바꾸거나 이전 trainer를 새 코드로 resume하지 않는다.
원래 40시간 시계도 유지하므로 중단 중 경과 시간도 포함한다.

별도 QB 분석 프로세스가 GPU를 사용해 종료를 기다렸다. 해당 프로세스 종료 후
17:09 KST에 원래 컨테이너 `panda-qb40-s1-v1`을 재가동했다.
새 admission 검증, 실패 branch 1개 및 의존 Student 2개 분리, launcher/publisher/priority watcher
가동을 확인했다. QB 종료와 Sheet readback 이후 기존 WV3 TUNE30/HARD36 재개 순서를 유지한다.

## s1 남은 실행 순서

| 순서 | Case | Replica | 학습 |
|---:|---|---|---|
| 1 | QB5_C60LR050 | R1 | 기존 s1 donor, Student 60K / LR 5e-5 |
| 2 | QB5_C60LR050 | R2 | 동일 설정, 다른 Student seed |
| 3 | QB5_S1_A400_E030 | R1 | 새 Stage 1, λ_abs=4 / λ_epsilon=.03 / 50K |
| 4–5 | QB5_A400_E030 | R1/R2 | 위 Stage 1 성공 시 Student 각 30K |
| 6–7 | QB5_B96C15LR100 | R1/R2 | 기존 s1 donor, batch96 / 15K / LR 1e-4 |
| 8–9 | QB5_B96C15LR200 | R1/R2 | 기존 s1 donor, batch96 / 15K / LR 2e-4 |

보류: 실패한 `QB5_S1_A200_E100 R1`, 의존 `QB5_A200_E100 R1/R2`.
새 A4/E.03 Stage 1도 실패하면 해당 Student 두 개만 보류하고 batch96 실험으로 넘어간다.
40시간 내 새 case를 끝낼 수 없는 경우에는 기존 정책대로 pending으로 기록한다.

## 검증 및 Sheet

관련 테스트 34개 통과: 새 failure/recovery 10개, 기존 runner 3개, enqueue 5개,
priority 11개, contract 5개. 실제 Docker 환경에서 QB train 17,139개와 데이터 경로 검사도 통과했다.
새 테스트에는 실패 branch 이후 9개 독립 작업 진행, 의존 차단, 잘못된 완료 집계 거부,
변조 checkpoint/불안전 optimizer 거부, 학습 코드·데이터 변경 거부, 원래 budget 보존이 포함된다.

실데이터 Stage 2 preflight도 PASS: 전체 calibration support 검사, 두 optimizer update 및
저장/재개 상태 검증을 통과했다. update replay의 최대 절대 차이는 9.31e-10이며 bitwise 일치는 아니다.
실패한 Stage 1의 평가 14개 및 중단 1개 checkpoint 모두 sidecar의 SHA와 일치했다.

`QB5_C60LR050 R1`의 첫 1K 공식 평가는 HQNR=0.784699110, ERGAS=4.472927732다.
초기 HQNR이 낮으므로 성능 개선을 확인한 결과로 해석하지 않는다.
기존 8K/9K/10K HQNR이 모두 .90 미만일 때의 Student 조기 중단 규칙을 유지한다.

Sheet는 기존 `QB-ablations`, HQNR≥.9315 및 ERGAS≤3.5 동시 달성 전체와
case/seed별 대표 행을 유지한다. 공식 1K 평가 및 모든 평가 checkpoint 저장 규칙도 유지한다.
17:16:58 KST 확인 시 실제 학습은 1,199 update였으며, 첫 1K checkpoint의 파일 SHA와
`QB-ablations` 대표 행 1개의 게시/readback이 검증됐다.
[확인 기록](assets/2026-11-01_qb40-support-recovery/verification.json)에 당시 상태를 남겼다.

## s3 인계

아직 QB40을 시작하지 않았다면 수정 커밋 pull 후 기존 `panda_qb40.enqueue --server s3 --wait`로 실행한다.
s1의 실패 case를 s3에서 미리 제외하지 않는다. s3 자체 Stage 1의 실제 결과로 분기한다.
이미 학습 중인 서버는 실행 중 source pin 변경을 피하도록 해당 학습이 끝난 후 pull한다.
이번 세션에서 s3 원격 학습을 직접 재시작한 것은 아니다.
