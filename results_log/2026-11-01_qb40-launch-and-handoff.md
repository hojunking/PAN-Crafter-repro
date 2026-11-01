# QB40 실행 등록 및 s3 인계 — 2026-11-01

승인된 서버별 40시간 계획을 `panda_qb40` 실행 패키지로 구현했다.
캠페인은 `PANDA_S13_QB40_LOCALSTAGE1_20261101_v1`이다.
상세 case·명령·출처·시간 정책은 [실행 README](../panda_qb40/README.md)에 있다.

## s1 실행 상태

2026-11-01 11:58 KST에 새 우선순위 컨트롤러를 가동했다.
현재 WV3 `S100DIFFUBILC30G025 R1`을 18,573 step에서 complete-state 저장한 뒤,
같은 optimizer/RNG/data stream을 복원하여 `finish-current` 모드로 이어 학습한다.
18,795 step까지 실제 train log 진행을 확인했다.

이 case가 30K를 완료하면 QB40의 `QB5_S1_A200_E100` 50K부터 자동 시작한다.
이 문서 작성 시 QB40 GPU 학습은 아직 시작 전이며 **실행 컨트롤러는 가동 중**이다.
QB40 이후 기존 WV3 TUNE30/HARD36의 남은 case를 재개한다. 이전 runtime 소스는 수정하지 않았다.

[실행 receipt](assets/2026-11-01_qb40-launch/launch_receipt.json)에 이전 컨트롤러,
checkpoint 해시, 복원 컨테이너 및 새 컨트롤러 상태를 보존했다.

## s3

s3 실행 코드는 준비됐으며, 이 세션은 s3를 직접 실행하지 않았다.
사용자가 push한 뒤 s3 에이전트가 pull하고 README의 `panda_qb40.enqueue --server s3 --wait`
명령을 실행하면 된다. 새 Stage1은 s3 seed 261016301로 독립 학습한다.
기존 기준 donor도 s3의 QB_DEEPEN 자체 학습 모델만 읽는다.

## 등록 내용

- s1: 새 Stage1 2개 + Student 10개. 명목 35.12h + 준비/변동 여유 4.88h.
- s3: 새 Stage1 3개 + Student 16개. 명목 36.36h + 준비/변동 여유 3.64h.
- Stage1: A2/E.10, A4/E.03; s3에 A4/E.10 추가. 각 full50K 후 기존 validation RRE 선택.
- Student: 기존 own donor의 B48 C60 LR5e-5, B96 C15 LR1e-4/2e-4,
  새 donor별 기본 C30 반복, s3 W128H050.
- 배치 96은 실제 micro12 × accumulation8. 15K warmup50, 30K/60K warmup100.
- scale1, RGB, A/T/S Lanczos4, DIFF/UBIL, β0, 기존 Hard loss와 공식 평가를 유지한다.
- QB의 절대 8/9/10K HQNR 모두 <.90 중단 규칙, 모든 평가 weights 보존을 유지한다.
- 두 서버 결과는 QB-ablations에 목표 H≥.9315 AND E≤3.5 통과 전부 및 대표 결과를 게시한다.
  대표는 H 충족 후보 중 최소 E, 없으면 최대 H다. Approach에 실제 하이퍼파라미터를 표시한다.
- 40시간 창에서 시작하지 못하는 후순위 case는 `pending_groups`로 남긴다.
  이미 시작한 full-horizon 학습을 시간 때문에 중간 donor로 선택하지 않는다.

## 검증

총 **28개 고유 테스트 통과**. [검증 기록](assets/2026-11-01_qb40-launch/validation.json).

실제 batch96 optimizer update와 8회 누적 stream 소비, checkpoint/optimizer/RNG 재개,
Stage1 실제 gradient 및 full-state 재개, W128 forward를 CPU에서 실행했다.
기존 엔진과 동일한 가중치/입력에서 Stage2 Hard loss 및 모든 Student gradient가 정확히 일치했다.
두 서버 출처·seed·목표/게시 namespace, Docker 마운트·중복 실행 방지,
현재 WV3의 안전 전환과 예산 부족 시 pending 회계도 검증했다.

s1 실제 데이터 catalog/train17139 및 RR/FR/validation 접근 검증 통과.
기존 s1 Stage1의 full50K 후보 이력·선택44K checkpoint SHA를 재검증했다.
Native CUDA cost 측정 및 실제 데이터 update/replay preflight는 현재 WV3가 끝난 뒤
런처가 자동 수행하며, 사전 검증 모델을 본 학습이나 Sheet 성능으로 재사용하지 않는다.
