# GF2 결과 시트 보완 (2026-09-21)

## 수정 범위

G20/QG40 업로더에서 누락됐던 `Date`를 실제 학습 기록으로 채운다.
업로드 시각이나 run 이름의 날짜를 사용하지 않는다. 학습 완료 시각을
Asia/Seoul 날짜로 표시하며, 완료 시각이 확인되지 않으면 실제 시작 날짜를
사용한다. 복구 작업의 시각을 학습 종료 시각으로 오인하지 않는다.

| 열 | 의미 / 표시 |
| --- | --- |
| Date | 실제 학습 완료일 우선, 한국 시간 `YYYY-MM-DD` |
| Seed / Model / Input | 현재 모델의 seed, 폭·깊이, P0/PLH 입력 구성 |
| Selection | G20=`Exact50K`, QG40=`RAW_MAX`; 기존 선택을 변경하지 않음 |
| Train(h) | 기존 optimizer 학습 구간 합계, 소수 2자리 표시 |
| Wall(h) | 학습 시작부터 trainer 완료 기록까지의 경과 시간, 소수 2자리 |
| Train time scope | Train(h)에 평가·I/O 시간이 포함되지 않음을 명시 |
| `G20/QG40 … UTC` | 학습 시작·완료·공식 후처리 완료 시각을 각각 보존 |
| Notes | 짧은 모델/seed/input/selection 요약 + 기존 메모·예외 기록 보존 |

지표는 4자리, Params는 4자리, FLOPs/Mem은 1자리, Infer/시간은 2자리로
**표시만** 제한한다. 저장된 실수 정밀도와 공식 metric JSON은 변경하지 않는다.
원본 Run ID도 변경하지 않는다. GF2의 Q4를 WV3의 Q8로 바꾸지 않는다.

## 진행 중인 실험: 보고 전용 보완 도구

진행 중인 G20/QG40은 실행 소스가 고정돼 있다. 새 `g20/upload.py`,
`qg40/upload.py`를 실행 중인 runtime에 덮어쓰거나, 그 runtime에서
`git pull`하여 기존 source 검증을 깨뜨리지 않는다.

기존 실험은 독립적인 CPU 보고 도구를 사용한다. 이 도구에는 Torch나
평가 코드가 필요하지 않으며, `PyYAML`, `gspread`, 기존 서비스 계정만 필요하다.
`--root`는 **실제 work_dir와 gspread/account.json이 있는 호스트 저장소**다.
비어 있는 Docker runtime의 host-side work_dir를 지정하지 않는다.

각 서버의 저장소에서, 서버 번호만 바꿔 실행한다:

```bash
# 읽기 전용으로 변경할 항목 확인
python tools/sensor_sheet_sync.py --root "$PWD" --server s1

# 이미 올라온 완료 결과의 누락 메타데이터 보완
python tools/sensor_sheet_sync.py --root "$PWD" --server s1 --apply

# 이후 완료·원본 업로드된 결과도 60초 간격으로 보완 (foreground)
python tools/sensor_sheet_sync.py --root "$PWD" --server s1 --apply --watch 60
```

열 순서가 다른 기존 탭(특히 GF2-s4)은 같은 서버에서 한 번 정렬할 수 있다:

```bash
python tools/sensor_sheet_sync.py --root "$PWD" --server s1 --repair-layout
python tools/sensor_sheet_sync.py --root "$PWD" --server s1 --repair-layout --apply
```

정렬은 기존 열 전체를 이동하므로 과거 날짜·지표·메모와 상세 provenance가
같이 이동한다. `Run → 캠페인 → RR → FR → 비용 → Date → Notes → 설정`을
앞쪽에 모으고, 상세 provenance는 뒤에 보존한다. 알려진 RR/FR/Cost 그룹
병합만 해제하며, 수식·보호 셀·사용자 그룹 제목 등이 발견되면 쓰지 않고
중단한다. 현재 서버의 잠금으로 보호할 수 없는 다른 서버 탭을 중앙에서
일괄 정렬하지 않는다. 정렬 백업은 `_sensor_sheet/<server>/layout/`에 남는다.

이 명령은 학습, 재평가, checkpoint 선택, 원본 업로드를 실행하지 않는다.
원본 업로드가 검증된 로컬 GF2 결과만 기존 compound-key 행에 추가한다.
다른 서버의 기록이나 미완료 결과는 추정해 채우지 않는다. watch 프로세스를
종료하면 자동 보완도 멈추며 cron/서비스를 설치하지 않는다.

원본 upload receipt·config·checkpoint SHA·source/data identity·본표 metric
정밀도를 확인한 뒤, 기존 업로더와 파일 잠금을 공유해 갱신한다. 수식·보호
범위·드롭다운 등 제어 셀은 거부한다. 기존 숫자·Run·원본 receipt는 보존한다.
보완 전 snapshot과 별도 readback receipt는 다음 경로에 남는다:

```text
work_dir/_sensor_sheet/<server>/<run_id>/
```

`reporting_extra/sensor_*.py`, `reporting_extra/__init__.py`,
`tools/sensor_sheet_sync.py`만 별도 배포하면 기존 고정 runtime을 바꾸지 않고
사용할 수 있다. 새 캠페인을 만들 때에는 수정된 정식 업로더와 배포 번들이
처음부터 메타데이터를 기록한다.

## 테스트

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
python -m unittest reporting_extra.test_sensor_sheet \
  reporting_extra.test_sensor_layout reporting_extra.test_sensor_backfill \
  reporting_extra.test_sensor_repair_layout \
  reporting_extra.test_sensor_sync_cli \
  g20.test_upload qg40.test_upload g20.test_deployment qg40.test_deployment -q
```
