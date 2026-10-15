# B01 s1 — native 평가 scene_index 중복 오류 및 복구

## 원인과 보존 상태

2026-09-28 20:47:27 KST, 첫 QFULL의 50K 학습이 완료된 뒤 native 평가 CSV row 구성에서 중단됐다.

```text
TypeError: dict() got multiple values for keyword argument 'scene_index'
```

`reporting_extra.evaluation`의 RR/FR `per_scene`에는 이미 `scene_index`가 있다. 기존 `dict(scene_index=i, **metrics)` 호출이 같은 keyword를 두 번 전달했다. 학습·지표 수식의 문제가 아니라 평가 결과 저장 코드의 결함이다. 기존 통합 테스트 mock이 실제 supplemental schema의 `scene_index`를 빠뜨려 검출하지 못했다.

- run: `RB01_WV3_S1_R1_SS9281101_QFULL_F50K_v1`
- 학습: `TRAINING_COMPLETE`, 50,000 updates, 재학습하지 않음.
- exact50K SHA: `ef5d5b5b42255d00c741abdfd73b0a90a58ec72aca4f18386c991336bf96ef4a`
- val-selected: 49,490 step, SHA `643902b98b04e14db6dc36153a9451432c18a5e3669536ee8993c9f997938399`
- 중단 전 raw: RR testimg0 한 장. 재평가 시 저장된 파일의 byte 동일성을 검사하며 덮어쓰지 않는다.

## 수정·검증

`panda_rb/evaluation.py`의 해당 한 줄을 `dict(metrics, split=..., scene_index=i, source_h5_row=i, ...)`로 수정했다. loop의 실제 scene identity가 우선하며 metric 및 supplemental 원본 필드는 유지된다.

실제 RR/FR per-scene schema, JQM w/w_source 보존, authoritative scene metadata, 40개 native 출력 순서를 검사하는 회귀 테스트를 추가했다. 수정 전 함수를 메모리에서 재구성하면 동일 TypeError가 발생하며, 수정 후 평가 36개 및 복구 4개 테스트가 모두 통과했다.

## 실행 중 실험의 출처 보존

이미 학습한 checkpoint와 공통 e-cache는 원본 frozen release에 결속돼 있다. 이를 바꾸거나 SHA manifest를 사후 수정하면 안 되므로 다음 방식으로 복구했다.

1. 원본 release/Teacher/cache/선택점/checkpoint를 그대로 보존하고 SHA 재검증.
2. 동결 소스 외부의 별도 entrypoint에서 **검증된 원본 evaluator SHA에 한해** 위 dict constructor 한 줄만 적용한 `_native_selection`을 메모리에서 사용.
3. 원래 checkpoint/config/data/source 검증을 모두 수행한 뒤 native와 stress의 `context` 및 `evaluator_identity`에 실제 수정 SHA를 추가.
4. 원래 controller의 `retry_technical` 경로로 재개. 완료된 selection manifest가 있어 QFULL 학습은 건너뛰고 평가부터 진행. 이후 등록 case도 동일 수정이 유지된다.

- recovery ID: `PANDA_B01_SCENE_INDEX_SERIALIZATION_v1`
- original evaluator SHA: `51f249d96e6543a4ed3cede1daeb5873cefc3510c74f83c749ee6eedc97b18da`
- corrected evaluator SHA: `696034877aeef0a7b93be8ca1f174de13404ffa8a211e7493924e97ee710e53a`
- 원본 frozen release: `a0beb7e5711da8f86a29e184280edc4b453bedd1b0d7d99447c5909edaf67fa2`
- native 함수 외 inference/metric/loss/selector/augmentation/seed/optimizer 변경 없음.

복구 entrypoint·보호 파일 SHA·이전 실패 상태·기동 영수증은 `work_dir/_panda_rb/20260928/B01/control/s1/recoveries/`에 별도로 보존했다. 이전 실패 컨테이너와 로그도 삭제하지 않았다.

## 재개

2026-09-29 **11:24:10 KST**에 재개했다.

```bash
python3 -B tools/panda_rb_recover_scene_index.py --server s1
docker logs -f panda-rb-b01-s1-sceneindex-3783be18
```

이 복구된 기존 cohort를 다시 기동해야 하면 위 recovery 명령을 사용한다. 일반 start는 수정된 working tree를 새 release로 해석하므로 원본 cohort를 조용히 교체하지 않고 거부한다. s3/s5의 수정·기동은 이 작업에서 수행하지 않았다. 다른 서버가 수정 시 전체 source identity를 바꿨다면 최종 통합 전 해당 서버의 수정 manifest와 동등성을 별도 확인해야 하며, pooling 검증을 완화하지 않는다.

## 복구 후 실측 확인

- B01 전체 CPU 회귀 **108개 통과**.
- exact50K/val-selected 모두 native 평가 완료, RR/FR raw 총 **80장** 저장. 서로 다른 checkpoint 2개로 명시된다.
- exact50K native: HQNR `0.9506087266505915`, SCC `0.987944` (stdout 반올림), ERGAS `2.050902` (stdout 반올림).
- A_ON 및 A_ZERO_INFERENCE_ONLY 각각 **980개 관측 완료**, 두 모드 모두 numerical failure **0개**.
- 복구 manifest의 보호 파일 SHA 재검사 결과 변경 **0개**. QFULL의 training attempt는 여전히 **1개**로 재학습하지 않았음을 확인했다.
- 다음 run `RB01_WV3_S1_R1_SS9281101_QMEAN_F50K_v1`이 **400/50000 step** 이상 진행됨을 stdout에서 확인했다. 이 숫자는 기록 시점의 진행률이며 완료 주장이나 최종 성능 판정이 아니다.
