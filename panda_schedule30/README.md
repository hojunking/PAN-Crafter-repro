# Schedule30: 양의 PAN 보정 + 기존 Hard loss 스케줄 실험

캠페인: `PANDA_S13_SCHEDULE_HARDONLY_20261023_v1`. 기존 Scale30 결과를 보존하고, 우선순위가 낮은 미실행 case를 제외한 뒤 별도 namespace의 후속 큐를 실행한다. **벽시계 시간 제한은 없으며 등록된 유한 큐가 끝나면 종료한다.** 기존 30시간 캠페인의 admission·deadline·경과시간은 초기화하거나 연장하지 않는다.

고정 조건은 WV3, W112·D123 Student, phase-aware `interp23`, FP32, AdamW, 유효 batch 48, 원본 GT·RR20/FR20 평가다. Student 입력에 동결 PAN Aligner의 예측 `u`를 **양의 scale 0.1 또는 0.2로 실제 적용**한다. Teacher·Aligner는 같은 서버의 기존 mature donor를 재사용한다. Soft β=0, rotation loss=0, Student 단일 view, TTA·앙상블 없음. 기존 C4 입력 증강은 유지한다. 구조 변경이나 밴드별 새 loss는 이번 실행 큐에 포함하지 않는다.

## 등록 큐

기본 Hard 계수는 α=0.25·γ=0.5, 기본 peak LR은 `1e-4`다. `H050`은 α=0.125·γ=0.25, `LR150`은 peak LR `1.5e-4`다. Fresh case는 warmup 100 update 후 해당 horizon까지 cosine을 적용하고 종료 LR은 0이다. R1/R2 seed는 각각 `261018101`/`261018102`다.

| 서버 | 순서 | Case | 반복 | PAN scale | 절대 종료 step | 새 update | LR / Hard 변경 |
|---|---:|---|---|---:|---:|---:|---|
| s1 | 1 | S020F25C35 | R1 | 0.2 | 35K | 10K | 기존 S020BASE 25K에서 `2e-5 → 1e-6` |
| s1 | 2 | S010F25C35 | R1 | 0.1 | 35K | 10K | 기존 S010BASE 25K에서 `2e-5 → 1e-6` |
| s1 | 3 | S020C30 | R1 | 0.2 | 30K | 30K | 기본 |
| s1 | 4 | S010C30 | R1 | 0.1 | 30K | 30K | 기본 |
| s1 | 5 | S020C35 | R1 | 0.2 | 35K | 35K | 기본 |
| s1 | 6 | S020C30LR150 | R1 | 0.2 | 30K | 30K | LR150 |
| s1 | 7 | S020C30 | R2 | 0.2 | 30K | 30K | 기본 |
| s1 | 8 | S010C30 | R2 | 0.1 | 30K | 30K | 기본 |
| s1 | 9 | S020BASE | R2 | 0.2 | 50K | 50K | 기존 큐에서 이관한 대조군 |
| s1 | 10 | S010BASE | R2 | 0.1 | 50K | 50K | 기존 큐에서 이관한 대조군 |
| s3 | 1 | S020F25C35 | R1 | 0.2 | 35K | 10K | s3 S020BASE 25K에서 `2e-5 → 1e-6` |
| s3 | 2 | S020C30 | R1 | 0.2 | 30K | 30K | 기본 |
| s3 | 3 | S020C30H050 | R1 | 0.2 | 30K | 30K | H050 |
| s3 | 4 | S020C35H050 | R1 | 0.2 | 35K | 35K | H050 |
| s3 | 5 | S020C30LR150 | R1 | 0.2 | 30K | 30K | LR150 |
| s3 | 6 | S020C30 | R2 | 0.2 | 30K | 30K | 기본 |
| s3 | 7 | S020C30H050 | R2 | 0.2 | 30K | 30K | H050 |
| s3 | 8 | S020H050 | R2 | 0.2 | 50K | 50K | H050의 두 번째 seed 대조군 |

정의의 기준은 [contract.py](contract.py)와 [plan.py](plan.py)다. 20K~35K는 1K마다 평가하며, continuation은 새로 학습한 26K~35K만 평가한다. 모든 선택은 기존 시험 지표를 참고한 탐색이며 독립 확인 실험으로 해석하지 않는다. 서로 다른 서버는 donor가 달라 절대 수치를 직접 차감하지 않는다.

## 기존 큐 제외와 이관

| 대상 | 조치 | 이유·보존 범위 |
|---|---|---|
| s1 S040BASE R1 | 5,489 update에서 checkpoint-safe stop 확인 | 높은 scale 구간의 HQNR 열세를 더 확인하기보다 스케줄 비교 우선. 조기 종료가 수렴 성능을 증명하는 것은 아님. stop checkpoint와 모든 평가 보존 |
| s1 S050BASE R1 | 미실행이면 제외 | 기존 mature HALF의 scale 0.5 대응 결과가 있으며, 낮은 scale보다 HQNR이 불리한 구간 |
| s1 S020BASE/S010BASE R2 | 기존 큐에서 미실행이면 제외하고 새 큐 후반으로 이관 | 필요한 seed 대조군은 유지하고 스케줄 후보를 먼저 실행 |
| s3 S020LR150H050 R2 | 미실행이면 제외; 정확히 이 case가 실행 중일 때만 안전 정지 허용 | LR·Hard를 함께 바꾼 R1 조합보다 두 요인을 분리한 짧은 스케줄과 H050 재현을 우선 |
| 이미 완료된 모든 case | 그대로 보존 | 결과·체크포인트·Sheet 행을 삭제하거나 실패로 재분류하지 않음 |
| 다른 실행 중 case | 정상 종료를 기다림 | 임의의 다른 trainer를 중단하지 않음 |

[transition.py](transition.py)는 기존 `pending_control.json`의 공식 미실행 group skip 인터페이스만 사용한다. 실행 중 case의 설정을 바꾸지 않는다. 대상 이외의 미완료·실패 상태는 자동 전환하지 않고 점검하도록 종료한다.

## 실행과 재개

Git으로 이 변경을 받은 뒤 해당 서버의 기존 Python·Docker 이미지·데이터·DLPan·인증 파일을 사용한다. 필요하면 `PAN_HOST_PYTHON`, `PAN_SCHEDULE30_IMAGE`, `PANCRAFTER_DLPAN`을 기존 로컬 경로/이미지로 지정한다. 인증 파일 내용은 출력하거나 커밋하지 않는다. 이 저장소에서 s3로 SSH 접속하지 않으며 s3 에이전트가 로컬에서 실행한다.

```bash
python -m panda_schedule30 catalog --server s3 --csv
python -m panda_schedule30 smoke
bash scripts/start_panda_schedule30_transition.sh s3
```

s1에서는 마지막 인자를 `s1`로 바꾼다. transition launcher가 기존 큐 제외, 필요한 대상 case의 안전 정지, GPU idle 확인, 격리 cost 측정, 실제 데이터 CUDA update/resume preflight, 새 학습·게시기 시작을 순서대로 수행한다. 직접 시작 명령과 동시에 실행하지 않는다. 기존 캠페인의 정상 종료와 GPU idle을 별도로 확인한 경우에만 직접 실행할 수 있다:

```bash
bash scripts/start_panda_schedule30_costed.sh s3
```

새 컨테이너 이름은 `panda-schedule30-hardonly-wv3-s3-v1`이다. 안전 정지 후 **동일 코드·동일 config·동일 이미지**로 재개할 때는 기존 컨테이너를 유지한다. GPU가 비어 있고 저장된 상태가 정상인지 확인한 뒤:

```bash
docker start panda-schedule30-hardonly-wv3-s3-v1
bash scripts/start_panda_schedule30_best_publisher.sh s3
docker logs --tail 40 panda-schedule30-hardonly-wv3-s3-v1
```

게시기만 다시 읽게 할 때는 `bash scripts/restart_panda_schedule30_publisher.sh s3`를 사용한다. 컨테이너가 이미 실행 중이면 중복 실행하지 않는다. source pin이 달라졌거나 poisoned/실패 상태라면 파일을 덮어써 재개하지 말고 원인을 확인한다. **실행·전환 대기 중에는 pin 대상 Python·shell·공유 소스, case registry, 평가기 파일을 편집하거나 Git 갱신하지 않는다.** 코드 변경은 새 캠페인 identity로 분리한다.

## 25K 분기와 증거

각 서버의 `panda_mature30/.../<server>/WV3/donors/rre_selection.json`에 지정된 기존 donor를 그대로 재사용한다. 재선정·다른 서버 donor 복사·Stage1 재학습은 없다.

`F25C35`는 같은 서버·seed의 Scale30 `S010BASE` 또는 `S020BASE`의 `models/step_0025000.pt`와 sidecar가 필요하다. [fork.py](fork.py)는 기존 loader로 원 checkpoint의 code/evaluator/config/SHA를 검증하고 Student 가중치, Adam moments와 step, 데이터 stream, RNG, 동결 A/T 및 calibration을 보존한다. 바뀌는 것은 등록된 후반 LR 스케줄뿐이다. `25001`에서 LR `2e-5`, `35000`에서 `1e-6`이며 warmup을 다시 하지 않는다.

절대 update는 25K에서 이어져 35K로 끝나고, 새 학습량·시간은 10K만 계산한다. 원본 checkpoint/receipt는 수정하지 않는다. 일반 resume는 config 변경을 허용하지 않으며, 이 분기는 별도 캠페인의 명시적 source-binding 프로토콜이다.

## Sheet와 cost

목적지는 기존 `WV3-ablations` 탭이다. **반올림하지 않은 동일 checkpoint의 FR20 HQNR·RR20 ERGAS**로 판정한다.

1. HQNR ≥ **0.9585**이고 ERGAS ≤ **2.040**인 checkpoint를 모두 게시한다.
2. run/seed/attempt별로 HQNR ≥ 0.9585인 후보 중 ERGAS가 가장 낮은 결과 하나를 대표로 유지한다. 대표에는 ERGAS 상한을 요구하지 않는다.
3. HQNR 하한을 만족하는 후보가 없으면 최고 HQNR을 대표로 게시한다. 동률은 ERGAS 최소 → 이른 step 순이다. 자격을 만족하는 대표의 ERGAS 동률은 HQNR 최고 → 이른 step 순이다.
4. 대표와 조건 충족 행이 같은 checkpoint면 중복하지 않는다. 고정 final30K/35K/50K 행을 따로 만들지 않는다. 평가가 없으면 수치를 만들지 않는다. Stage1은 기존 BEST_VAL_RRE 규칙을 유지한다.

새 publisher 정책은 `PANDA_SCHEDULE30_RRE_STAGE2_H9585_PASS_ALL_OR_BEST_FALLBACK_V1`이다. `PSC30` 새 캠페인 소유 행의 A:BL만 갱신하며 다른 캠페인·서버 행과 오른쪽 추가 열을 보존한다. 실제 A+S 배포 경로의 cost를 서버별 격리 측정해 signature로 연결한다. 측정하지 않은 값은 추정하거나 다른 서버 값으로 채우지 않는다. 학습 시간은 해당 case의 실제 추가 update만 포함하며 원 donor·25K prefix의 비용을 다시 청구하지 않는다.

상태 경로는 `work_dir/panda_schedule30/PANDA_S13_SCHEDULE_HARDONLY_20261023_v1/<server>/WV3`다. `preflight/passed.json`, `campaign_state.json`, run의 `train.jsonl`, `sheet_publish_panda_schedule30/last_poll.json`을 확인한다. 게시 성공은 `readback_verified=true`로 확인한다. 전환 기록은 `work_dir/panda_schedule30_launchers/<server>/transition_receipt.json`이다. 이 README 자체는 실행 시작 확인서가 아니다.
