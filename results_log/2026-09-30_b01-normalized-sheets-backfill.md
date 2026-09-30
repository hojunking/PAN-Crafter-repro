# B01 normalized Sheets bridge — s1 백필·실제 readback

작업 `PANDA_RB_B01_SHEETS_BRIDGE_v1`, 2026-09-30. 최종 readback 15:42 KST.

## 완료 범위

독립 bridge 구현과 **s1 기존 완료 결과 백필**을 완료했다. 전체 3서버 완료 보고는 아니다.
s3·s5의 검증 패키지는 미제공이며 실제 실행 상태를 추정하지 않았다.

| 항목 | s1 확인·등록 | 전체 기대 |
|---|---:|---:|
| Student | 8 | 24 |
| Native selection 관측 | 16 | 48 |
| Stress curve | 16 | 48 |
| Stress point | 784 | 2352 |
| Case별 primary Student | 2 | 6 |

원 stress numerical failure는 0, **invalid geometry는 640건(각 curve 40건)**이다.
원 수치·coverage·실패를 보존하고 `COLLECTED_WITH_FAILURES`로 표시했다.
영향받은 point/radius 통계에는 failed/missing Student 수를 병기한다.
새 학습, checkpoint 재선택, metric 재평가, 원 실험 수정은 하지 않았다.

## 코드와 검증

신규 `reporting_bridge/`와 `tools/rb_b01_sheets.py`는 기존 numerical identity glob 밖이다.
collector는 frozen 원 검증 함수 및 source/config/binding/선택 manifest/checkpoint/raw hash,
native CSV와 stress 49×20 grid·원 집계를 대조한다. 계약은 canonical native 64열,
stress 43열, 상태 24슬롯이며 ID/행은 결정적이다. 통계 단위는 Student다.
source/runtime/F1/data/evaluator/raw-map mismatch는 cohort로 분리한다.

`panda_rb/*`, `tools/panda_rb_*`, controller/queue/watchdog/기존 uploader/frozen runtime과
원 결과 파일은 이번 작업에서 수정하지 않았다. 기존 dirty worktree와 이전 복구 수정은
이번 bridge 커밋에 포함하지 않는다. cron/무한 worker/새 학습/B02를 시작하지 않았다.

- 원 pinned Docker의 frozen report: Student 8/native 16/curve 16, 검증 성공.
- 합성 단위·통합 테스트 **62개 PASS**; 수식 1,000개 이상 독립 계산 대조.
- 빈 source 연결 후 기존 **3,892개 관측**을 Result_ID별 보존 확인.
- `paper`/`ablations`/`유의미한결과` 값·수식·note·format·merge, 기존 view 수식,
  보호 범위 및 기존 탭 상대 순서/숨김 상태 보존 검사 PASS.
- 최초 백필: native+stress 800관측, 상태 24행. 원시값 824행 readback PASS.
- 첫 실제 summary 검증에서 빈 설명의 `COUNTIFS("<>")` 집계 차이를 탐지했다.
  수치 변경 없이 상태 수식만 `LEN(...)>0`으로 교정하고 회귀 테스트를 추가했다.
  최종 summary **692행 전체** 상태/평균/SD/paired 집계가 원 per-seed 기대값과 일치한다.
- 동일 payload 재전달: **신규 관측 0 / 동일 800 / 상태 변경 0**.
- 최종 `VIEW_READBACK_VERIFIED`: source, `_records`, `WV3-main`, curve/summary 일치.
  기존 관측/표/수식 회귀 및 새 spill/수식 오류 없음.
- 업로드 후 재수집: 원 native/curve evidence와 검증 provenance가 전과 완전히 일치,
  검증 오류 0. 원 report의 `sheets_uploaded=False`는 수정하지 않았다.
- Google CellData로 header/freeze/4자리 표시 및 원정밀도 보존을 확인했다.
  브라우저 픽셀 렌더 검증은 수행하지 않았다.

## 실제 Sheet 변경 및 식별자

Spreadsheet: `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`, `pan-cvpr27`.
기존 46탭 뒤에 7탭을 추가했다. `_records!A2`의 마지막 반환식만 확장하고
legacy parser/allrows, `WV3-main!A5/A6`는 보존했다. 기존 수식이 06 B01 그룹을 표시한다.

| 신규 탭 | 실제 gid |
|---|---:|
| `_rb01_s1` / `_rb01_s3` / `_rb01_s5` | 269300000 / 269300001 / 269300002 |
| `_rb02_points` / `_rb_b01_status` | 269300003 / 269300004 |
| `RB02-curves` / `RB-B01` | 269300005 / 269300006 |

[RB-B01 요약](https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit#gid=269300006)
· [RB02-curves](https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit#gid=269300005).

```text
Numerical source SHA:
656b67d4c23bef830611db8676652a7760e88db3f2c2b3b17fec2f4f87abed34
Input evidence package payload SHA (파일 byte SHA와 다름):
bbc8887db3cd405edb5dbecf3ff73275357d894edd60d506f4dc1f85098c3734
Prepared payload SHA (자기 hash 필드 제외):
a71f73311006aa2336486454d9b72b464c062abfd14097dbfaafc4411c18ffde
_records A2 formula before SHA:
25ec87113604cf4c38d9aac4a2272d43ce3f9103844dcee6566bed684310bede
_records A2 formula after SHA:
517d1e027d98972a83f5e60086eb28096508acf3b38d324ddc56714ce9d0c1d8
```

## 실행 기록·재시도·남은 작업

별도 `work_dir/_rb_sheet_upload/B01/` 아래에 보존했다.

- `incoming/B01_s1_report_readback.json`, `incoming/s1.evidence.json`: 원 보고서와 검증 패키지.
- `inspect/snapshot.json`, `plan/plan.json`: 원 formula/note와 rollback·diff 근거.
- `setup/setup_receipt.json`, `upload_dry_run/upload_receipt.json`: 생성 gid·회귀·최초 쓰기 계획.
- `receipts/upload_receipt.json`: 동일 payload 재시도, 신규 0관측.
- `receipts/history/`, `receipts/spool/`, `receipts/events-upload.jsonl`: 불변 payload와 전체 단계·실패 이력.
- `verification/verification.json`: 최종 source/view/summary/보호 표 확인.
- `post_upload_evidence/s1.evidence.json`: 원 증거 불변성 재검증.

**미수집**: s3·s5 각 8 Student/16 curve의 원 서버 검증 패키지. 없는 실험을 새로 돌리지 않는다.
각 원 서버는 frozen 검증 후 collect만 하고, writer s1은 기존 s1 패키지를 포함한 누적
prepare→upload→verify를 수행한다. 상세 재시도 명령은 `reporting_bridge/README.md`에 있다.
pull만으로 학습·업로드는 자동 실행되지 않는다.

구현 commit은 이 문서·해당 실행 지시서·신규 bridge 파일만 포함한다.
credential/weight/raw 영상/로컬 spool은 Git에 포함하지 않는다.
