# FH20R1 R2: 수치 method 보존 + 우선순위 인계

이 패키지는 기존 `WV3_FH20R1_20260919_v1`의 **우선순위 controller**다.
새 캠페인이나 새 20시간 시계가 아니다. 수치 method, 원본 source/HEAD,
145개 case 정의, Teacher F1–F5, 현재 branch, loss, seed, config는 유지한다.
현재 선택된 branch의 CORE84+RESERVE40을 그대로 사용한다.

## 다른 서버에 전달

실행 중인 서버에서 **일반 pull/checkout/merge로 HEAD를 바꾸지 않는다.**
원 checkpoint와 reference는 HEAD까지 검증한다. `tools/fh20r1_*`, `fh20r1/`,
`fh12/`를 덮어쓰지 않고 아래 신규 경로만 추가한다. s2의 지역 GPU parity
예외와 resource policy도 원 소스/receipt를 그대로 읽는다. 이 패키지는
그 예외를 새로 승인하거나 허용 오차를 넓히지 않는다.

배포 tar를 각 서버의 PAN-Crafter 저장소에 옮긴 뒤:

```bash
tar -xzf pan-fh20r1-r2-overlay.tar.gz
bash tools/r2_start.sh
```

기존 `gspread/server.txt`에서 s1–s5를 인식한다. 다른 환경이면 실행 시
`PYTHON=/absolute/path/to/pancrafter/python`을 지정한다. 기존 캠페인 등록이
없는 서버에서 새 실험을 임의로 시작하지 않는다.

Git으로 전달할 경우 배포 브랜치만 push한다 (학습 중 checkout은 유지):

```bash
git push origin fh20r1-r2
```

받는 서버에서는 HEAD를 바꾸는 `pull` 대신 신규 경로만 추출한다:

```bash
git fetch origin fh20r1-r2
git archive FETCH_HEAD campaign_r2 tools/r2_runner.py tools/r2_start.sh tools/r2_bundle.py results_log/PAN_FH20R1_R2_MethodPreserving_Adjustment_2026-09-19.md | tar -x
bash tools/r2_start.sh
```

tar 방식과 Git 방식 중 하나만 사용한다. R2가 이미 적용된 저장소에서 코드가
바뀐 overlay를 덮어쓰면 registration 검증이 거부하므로, 등록 후 업데이트는
별도 인계 검토가 필요하다. 같은 배포본의 반복 실행은 중복 학습을 시작하지 않는다.

검토만 하려면 (학습·hold·watchdog·ledger를 변경하지 않음):

```bash
bash tools/r2_start.sh --dry-run
python tools/r2_runner.py audit
python tools/r2_runner.py status
```

`audit`은 별도 `priority_r2/` 보고서만 기록한다. source/config/branch/공식
완료 증거/ledger 무결성 실패는 오류 종료다. 미수행 P1 과학적 진단은 모든
서버의 학습 대기 조건으로 만들지 않는다.

## 정확한 인계 동작

1. 원 source/runtime, 전체 config, 기록된 branch·Teacher bridge·기존 budget을
   검증하고 원 상태/ledger prefix/hold를 별도 registration에 저장한다.
2. 명시적 `start`에 한해, unhashed `_watchdog.sh`에 R2 routing만 추가한다.
   기존 사용자 수정과 실행 권한은 보존한다. 파일 배포만으로 학습은 바뀌지 않는다.
3. R2 고유 nonce의 admission hold를 게시한다. 기존 **train/postrun/upload는
   자연 종료**하며, 기존 controller만 다음 case 입장 시 멈춘다. kill/signal 없음.
4. 동일 `.runner.lock`을 확보한 뒤 다시 대조한다. 인계 사이에 원 controller가
   뒤늦게 admission한 block까지 포함해 기존 pair/quartet 순서를 먼저 끝낸다.
5. 자신의 nonce일 때만 원 hold를 복원하고, 완전히 미시작 block에 R2 순서를
   적용한다. 완료 결과는 검증 후 재사용하며 미업로드는 upload-only로 재시도한다.
6. 재시작/watchdog도 R2 경로만 사용한다. 등록 손상·source 변화 시 기존 queue로
   자동 fallback하지 않는다. P0 오류는 신규 admission을 막고 기존 자산은 보존한다.

`start`의 `R2_REQUESTED`는 인계 **요청**이다. 실제 적용은
`priority_r2/applied_readback.json`의 `applied=true`, revision, 현재 block,
next-two IDs, 기존 ledger hour로 확인한다. 활성 학습 step은 `status`의
`live_updates`에 표시한다. 현재 case가 길면 인계도 그 종료까지 기다린다.
이미 시작한 reserve는 중간에 20h를 넘어도 끝내며, 미시작 reserve만 생략한다.

`start`는 기존 watchdog cron을 확인하고, 없으면 해당 저장소의 15분 주기·
재부팅 복구 entry를 등록한다. 다른 job·주석·다른 저장소 entry는 보존한다.
기존 crontab을 읽지 못하면 덮어쓰지 않으며 `recovery.status=WARNING`을
반환한다. 이 경우 현재 인계는 진행하지만 무인 재부팅 복구는 보장하지 않는다.
등록 결과는 `watchdog_installation.json`에 남는다. 현재 작업을 재시작하려고
기존 trainer를 종료하거나 source identity JSON을 수정하면 안 된다.

## 순서

| 서버 | CORE 선호 block 순서 | RESERVE |
|---|---|---|
| s1 | 01→02→03→04→05→06→07 | 08–11 |
| s2 | 01→02→03→04→05 | 06–09 |
| s3 | 01→02→06→03→04→05→07→08→09→10 | 11–14 |
| s4 | 01→02→06→03→07→04→05 | 08–09 |
| s5 | 01→02→06→03→04→05 | 07–08 |

기존 admitted block이 항상 이 표보다 우선한다. s1은 BASE fallback,
s2는 F2 fallback, s3–s5는 STANDARD라는 **실제 local branch 기록**이
필요하다. 다른 branch가 기록돼 있으면 자동 전환하지 않고 검토를 요청한다.

## 결과와 확인 범위

`work_dir/_fh20r1/<server>/priority_r2/`에 다음을 별도로 기록한다.

- `runtime_reconciliation.json`, `method_invariant_receipt.json`,
  `frequency_frontend_receipt.json`: 원 자산과 기존 검증 receipt 대조.
- `S73101_selection_audit.json`, `eligible_curve.csv`: 고정50후보,
  5선택점 SHA, eligible count·연속구간, 원 장면별 결과.
- `block_summary.csv/json`: **TARGET 우선** 비교, 미적격 공란,
  동일 server/Teacher/seed 대응차·완료 quartet 상호작용. 다른 선택점 혼합 없음.
- `selection_upload_receipt.json`: 로컬 원본·기존 receipt 검증. 이 수집기 자체가
  live Sheet를 다시 읽은 것은 아니므로 실시간 Sheet PASS라고 표시하지 않는다.
- `campaign_time_and_completion.json`: 원 FH20R1 interval union 재사용.
- `verification_status.json`: V00–V14의 실제 상태와 남은 확인 항목.

기존 Sheet RAW_MAX 본열·TARGET 열·선택 규칙은 바꾸지 않는다. 별도 비교
요약은 TARGET을 대표로 읽으며, 로컬 CSV로 제공한다. 새 Google Sheet 요약
탭을 자동 생성하지 않는다. 기존 RMSE/CC/JQM 보충 worker와도 충돌하지 않는다.

V03의 s2 교차환경 동일 checkpoint 추론, V05의 반경별 q 분석, 추가 V06 A/U
교차, V07 donor 원본 검증, V09 z-score/GN/GAP 분리, V10 gradient 활동,
V11 공통 tensor/batch·역사적 width bridge는 새로 실행한 것으로 표시하지 않는다.
해당 GPU/원본 자산이 필요한 항목은 `TO_VERIFY`/`NOT_IMPLEMENTED`로 남기며
수치가 낮다는 이유로 loss/Teacher/branch를 자동 변경하지 않는다.

이 overlay의 보고서 수집은 기존 저장 결과의 읽기/재집계이므로 학습시간을
새로 부풀려 credit하지 않는다. 추후 실제 신규 GPU 진단을 수행할 때는 원
`fh20r1.ledger.record_interval`의 고유 ID·committed evidence를 사용해 한 번만
기록한다. 기존 시간 prefix 삭제/변경, 중복 credit, 대기시간 credit는 금지한다.

원 MD에 나열된 companion JSON/CSV는 제공되지 않았다. 배포의
`spec/priority_overlay.json`·`spec/queue_overlay.csv`는 **제공 MD §4/부록B와
기존 registry를 대조해 생성한 구현 명세**이며, 원 제공 파일로 가장하지 않는다.

## 개발 검증

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  python -m unittest discover -s campaign_r2 -p 'test_*.py' -q
python tools/r2_bundle.py
```

배포 tar에는 신규 controller/검증 코드/명세/원 MD만 포함되며,
credentials·데이터·checkpoint·원 trainer·원 watchdog 파일은 포함하지 않는다.
