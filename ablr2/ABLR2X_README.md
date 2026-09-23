# ABLR2X — 기존 s1/s2 연속 실행 + C17 + s3/GF2

2026-09-23 번들의 원 190개 정의·원 source SHA·기존 case ID·seed·실험 비용을 유지한다.
s1/WV3와 s2/QB는 새 BOOT를 시작하지 않는다. 실제 등록된 BOOT/REFRESH/VERIFY를 조사해 누락 C17만 추가한다.
s3/GF2만 새 BOOT5 100개를 시작한다. s4/s5는 지원 lane이 아니며 건드리지 않는다.

## 배포와 시작

구현 commit → push → 각 서버 pull 후, **해당 서버에서 한 번** 실행한다.
pull/build 자체는 학습·lease 갱신·Sheet 업로드를 시작하지 않는다.

```bash
# 서버에 맞게 s1 / s2 / s3 중 하나만 지정
PYTHON=/정확한/호스트/python \
bash tools/ablr2_docker_start.sh --server s1 --until-operator-stop
```

`--until-operator-stop`이 새 실행의 명시적 무기한 승인이다. 기존 72h lease 파일을 연장하거나 덮어쓰지 않는다.
기존 s1/s2 runner에는 협력적 `STOP_AFTER_RUN`을 요청하고, 원 고정 release에서 현재 run을 마친 뒤 전환한다.
대기 중에는 새 학습을 실행하지 않는다. 기존 작업의 finite lease가 먼저 끝나거나 fullstate 복구가 필요하면,
그 원 release에서 복구해야 한다. 새 승인으로 과거 작업을 몰래 재개하지 않는다.

전환 후 동일 Python 환경에서 원/새 소스의 실제 CPU 수치 probe를 비교한다.
통과한 source-bound receipt만 데이터/Teacher의 읽기 호환 근거로 사용한다.
실제 C03↔C17 초기 U·sample/view 순서·matched Teacher 증거는 별도로 검증한다.

Docker를 쓰지 않는 경우:

```bash
bash tools/ablr2_start.sh --server s2 --until-operator-stop
```

기본적으로 공식 평가 후 Sheet 업로드를 시도한다. `--no-upload`로 끌 수 있다.
기본 Docker 이미지는 `hojunqueen/pancrafter-env:torch2.4.0-cu118`이며, 로컬 이미지의 SHA로 고정한다. 자동 pull하지 않는다.
실행 코드에 미커밋 변경이 있으면 시작을 거부한다. 원본 checkout의 이후 pull은 고정 runtime에 영향을 주지 않는다.

## s3의 선행 GF2 실험 전환

선행 작업을 강제 종료하거나 queue·checkpoint·cron을 삭제하지 않는다.
현재 source가 run-boundary 정지를 지원하면 완료를 기다린다.
**구형 GFP40/GFB20의 고정 runner는 run-boundary 정지를 지원하지 않을 수 있다.** 이 경우
`RUN_BOUNDARY_UNSUPPORTED`로 중단하며, run이 끝났다고 가장하거나 block-stop을 run-stop으로 바꾸지 않는다.
운영자가 계획의 긴급 SAFE 전환을 선택한 경우에만 다음 옵션을 사용한다.

```bash
bash tools/ablr2_docker_start.sh --server s3 --until-operator-stop --boundary SAFE
```

SAFE 전환은 원 controller의 안전정지를 요청하고, fullstate 보존·원 프로세스 종료·GPU idle을 확인한 뒤 새 학습을 시작한다.
지원되지 않는 다른 선행 controller나 소유자가 불명확한 watchdog이 있으면 명시적으로 대기한다.

## C17과 재사용 판정

- C17은 C03과 동일한 PLH Student W104/D122이며, matched TZERO의 **exact50K Aligner만** 복제한다.
- Aligner는 학습 가능하다. alpha/beta/edge=0, plain GT, Teacher U/output/q/cache/calibration을 loss에 사용하지 않는다.
- C17은 예정 C03 다음에 삽입한다. 이전 완료 panel의 누락분은 오래된 순서로 처리하되 기존 작업 사이 최대 2개만 실행한다.
- 원 C03을 비교에 재사용할 근거가 부족하면 별도 `PAIR_REPAIR` C03을 같은 seed/조건으로 실행한다.
  원 C03 관측은 보존하며, repair는 19번째 component나 새 독립 반복으로 세지 않는다.
- TZERO가 없거나 원 source를 확인할 수 없으면 `REFERENCE_UNAVAILABLE`로 남긴다. 다른 seed/production Teacher로 대체하지 않는다.
- `legacy_core17_complete`와 `extended18_complete`를 구분한다. A17 판정에는 검증된 5쌍이 필요하다.

## 수치·평가·반복 계약

모든 기본 run은 fresh50K, batch48, FP32이며 기존 C00–C16의 190개 config가 원 구현과 정확히 동일하다.
GF2는 4-band/DN1023, train19809/val2201, RR20/FR20, Q4/GF2 MTF를 사용한다.
LP는 기존 sigma1.98/k41/replicate/[2::4] 레시피이며, QB msfix를 GF2에 적용하지 않는다.
GF2 BOOT의 기존 실제 사용 seed를 조사하고 충돌 시 결과 관측 전에 결정론적 mapping receipt를 고정한다. 원 CSV는 수정하지 않는다.

새 계획의 주 checkpoint는 고정 50-candidate의 **val-ERGAS 최소**, 보조 주 분석은 exact50K이다.
HQNR best/test ERGAS best는 탐색용으로 별도 저장한다. 같은 checkpoint의 metric만 한 관측으로 묶는다.
FR HQNR은 원 native PAN/LMS·전체 support·장면별 평균이며 masking/shift-reference로 바꾸지 않는다.
JQM은 SRF-substitute 보조 variant임을 표시한다. 같은 FR20의 반복 개발은 `TEST_AWARE_DEV`이지 새 독립 test set이 아니다.

각 lane은 BOOT5 → 필요 관계 RECHECK5 → 최대 2 recipe/고정 2 DEV block → 전체 REFRESH5를 반복한다.
운영 threshold는 기존 legacy17 관계로 계산/고정한다. A17_SCRATCH는 설명용이며 자동 queue에 넣지 않는다.
서버 간 Teacher 공유·lock·승인 barrier는 없다. 로컬 중복 실행과 디스크 여유만 검사한다.
디스크 여유는 최소 100GiB 및 다음 block 예상 쓰기의 2배 중 큰 값이다. 자동 삭제하지 않는다.

## Sheet와 상태

raw tab은 `WV3-s1`, `QB-s2`, `GF2-s3(5090)`이다. 실제 GF2 C00–C17 결과만 `ablations`의 별도 marker 영역에 추가한다.
기존 B20/P40/다른 표를 덮어쓰지 않는다. 숫자 원값은 보존하고 표시 정밀도만 제한한다.
업로드 실패는 outbox에 남겨 제한된 수만 재시도하며 학습을 재시작하지 않는다.

```bash
python tools/ablr2_runner.py status --server s1
python tools/ablr2_runner.py stop --server s1 --command STOP_AFTER_RUN
python tools/ablr2_runner.py stop --server s1 --command STOP_NOW_SAFE
python tools/ablr2_runner.py stop --server s1 --command CONTINUE
python tools/ablr2_runner.py retry-uploads --server s1
```

각 lane 상태는 `work_dir/ablr2/<sensor>/<server>/`에 있다.
기존 `registration.json`, `preflight.json`, `runtime_release.json`, lease는 보존하고 새 실행은 `_ablr2x` 파일을 쓴다.
`migration/`에는 원 상태의 byte snapshot과 source parity proof를 남긴다.
`extension/debts.json`, 각 C17의 `meta/paired_anchor.json`에서 누락/repair 사유를 확인할 수 있다.

## 검증 범위

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
python -m unittest discover -s ablr2 -t . -p 'test_*.py'
```

CPU 회귀·실제 CPU probe는 GPU bitwise parity, 실제 full-data calibration, 원격 기동 또는 live Sheet 쓰기를 보증하지 않는다.
각 서버에서 데이터 SHA/LP·native GPU forward/원 runtime 전환 proof를 실제 시작 시 확인한다.
구현 작업 자체로 기존 실험을 중지하거나 새 학습·live Sheet 수정을 수행하지 않는다.
