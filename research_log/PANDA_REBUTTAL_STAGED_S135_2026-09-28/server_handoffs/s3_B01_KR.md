# s3: B01 실행 담당자 인계

작성일: 2026-09-28. 상태: 실행 제안 / 아직 기동하지 않음.

## 대상

**WV3, RB01 네 case 전부를 각각 두 번 + 각 모델의 RB02 shift 평가.** s3는 다른 서버가 맡은 case를 기다리지 않으며, case를 분담하지도 않는다.

- Repeat 1, seed `9281301`: QMEAN → QFULL → QESUR → QSHUF
- Repeat 2, seed `9281302`: QSHUF → QESUR → QFULL → QMEAN

각 학습은 fresh Student50K다. 총8개 run이며, 각 run의 EXACT50K에서 A_ON과 A_ZERO_INFERENCE_ONLY 두 stress curve를 만든다. 총16개 curve record다. Teacher F1은 세 서버에서 동일한 weights/calibration을 사용한다.

## 순서

1. `MASTER_PLAN_KR.md`와 `B01_RB01_RB02_DETAILED_KR.md`를 기준으로 구현·자산·runtime을 검증한다.
2. 사용 가능한 GPU와 기존 owner를 확인한다. 기존 queue를 강제 종료하지 않는다. s2/s4 또는 다른 서버 상태를 바꾸지 않는다.
3. 해당 repeat의 공통 초기 U/A·50K stream·원 F1/reference를 고정한다. 다른 seed의 초기값이나 이전 case의 학습 weights를 재사용하지 않는다.
4. 위 순서대로 학습→native evaluation→RB02→다음 case를 처리한다. QFULL이 먼저가 아니어도 초기값은 미리 고정되어 있어야 한다.
5. 8개 train run과16개 curve record의 완료/실패 원자료를 보존하고 ZIP·SHA·다운로드 링크를 반환한다.
6. **STOP_FOR_REVIEW.** repeat3, 무한 cycle, RB03/RB04 자동 실행은 하지 않는다.

정확한 ID는 `planning/training_runs_24.csv`에서 `s3`로 filter하고, 평가 ID는 `planning/stress_curves_48.csv`를 따른다. 이것은 실행형 trainer script가 아닌 명세이며, 원 frozen runtime을 유지하는 별도 worker 연결이 필요하다.

## 핵심 금지

제출 paper/Table3를 수정하지 않는다. 과거 G23/PCREPRO/다른 Teacher를 새 baseline으로 대체하지 않는다. 같은 checkpoint 두 번 추론을 독립 Student seed2회로 세지 않는다. q를 섞는 난수가 sampler를 바꾸면 안 된다. shift된 PAN과 native LP를 혼합하지 않는다. A_ZERO는 trained no-align baseline이 아니다.
