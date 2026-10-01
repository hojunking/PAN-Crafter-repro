# s3 실행 인계 — PANDA M12, 각 case 4회

상위 규약: `EXPERIMENT_CASES_KR.md`. 실제 실행 상태: **미기동**.

이 서버는 WV3의 **10개 case 전체를 각각 4회** 수행한다. 다른 서버와 case/센서를 분담하지 않는다.
Student seed: `261001301, 261001302, 261001303, 261001304`. 동일 repeat의 10개 case는 같은 초기 U/A, F1, train/view stream을 사용한다.

## 순서

1. 가용 GPU/원 F1 binding/data/runtime 확인. 기존 B01 결과는 회수하되 새 M12 반복에 합산하지 않는다.
2. STEP1: QFULL, QMEAN, QSHUF, QESUR, QEDGE, QALIGN의 4-repeat, 총 **24개 fresh50K**. 각 모델 native 평가 및 4-mode stress.
3. STEP2: H0, HSPMEAN, NOADV, ADVMEAN의 4-repeat, 총 **16개 fresh50K**. STEP1 QFULL baseline을 사용하며 다른 case에서 이어 학습하지 않는다.
4. 자료 검증·upload package·결과 보고 후 `STOP_FOR_REVIEW`. 성능이 낮아도 등록한 반복을 다른 seed로 교체하지 않는다.

정확한 실행 순서는 `planning/server_schedule.csv`에서 server=s3를 따른다. 이 서버의 목표는 Student40 / native selection80 / stress curve96이다.

## 명령

`tools/rb_m12_runner.py`는 **새로 구현할 인터페이스**다. 기존 `panda_rb_runner.py`의 24개/2-repeat 가드를 수정해 대체하지 않는다.
구현·테스트와 바인딩 검증 후 아래 명령을 사용한다.

```bash
python3 -B tools/rb_m12_runner.py preflight --server s3 --plan /actual/path/planning/experiment_registry.json --binding /actual/path/verified_M12_bindings.json
python3 -B tools/rb_m12_runner.py run --server s3 --plan /actual/path/planning/experiment_registry.json --binding /actual/path/verified_M12_bindings.json --activate
```

## 보고/Sheet

native 결과는 hidden M12 source → `_records` → **WV3-ablations**의 M12 그룹으로 간다. `paper`, 논문용 `ablations`, 기존 수동 `유의미한결과`에는 쓰지 않는다.
이 서버는 검증된 evidence package를 writer s1에 전달한다. 공유 수식/setup을 직접 변경하지 않는다.

실제 run/seed/initial state/stream/checkpoint/protocol SHA, 실패·미완료·coverage 경고, 평균±SD와 paired 차이를 모두 반환한다. inference-only mode 반복은 Student 반복 수를 늘리지 않는다.
