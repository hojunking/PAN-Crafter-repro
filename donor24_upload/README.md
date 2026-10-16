# donor24 업로드 위치 수정 / s1·s3 무중단 적용

## 원인과 변경

기존 `donor24.publication`의 새 행은 `appendCells`로 추가된다. 과거
업로더가 빈 문자열(`stringValue: ""`)로 지운 행도 이 API에서는 사용한
행으로 취급될 수 있어, 화면에 보이는 결과 표와 멀리 떨어진 위치에
새 결과가 기록됐다. HQNR/ERGAS 조건에 따른 배치가 아니다.

2026-10-16 동일 문서의 임시 숨김 탭에서 재현: 데이터 6행까지,
100행에 빈 문자열 기록 → `appendCells`는 101행, `values.append`는
7행에 기록. 서식만 있는 100행은 이 증상을 재현하지 않았다.
테스트 탭은 제거했고 원본 증빙은 s1의
`work_dir/donor24/sheet_maintenance_20261016/empty_probe_*.json`에 있다.

새 업로더는 `'WV3-ablations'!A5:BL` 전체 값 테이블에 RAW append한다.
Google이 추가 행을 할당하므로 s1/s3가 빈 행 번호를 추측하여 덮어쓰지
않는다. 행 삽입·정렬·삭제, 다른 서버 결과 재배치는 하지 않는다.
기존의 값·서식·헤더·필터는 보존한다. 이미 멀리 있는 실제 데이터까지
자동 이동하지는 않으며 수동 재배치와 업로드를 동시에 수행하면 안 된다.

Result_ID와 전체 JSON provenance가 값과 함께 기록된다. A열 메모는
새 스냅샷에서 해당 ID를 찾은 뒤 추가한다. 통신 실패 또는 두 단계 사이
종료 시 다음 폴링에서 ID로 재조회하여 메모만 복구한다. 모든 값과 메모를
재조회하기 전에는 업로드 성공으로 기록하지 않는다. POST 자체는 재시도하지
않는다. 동일 서버의 복수 writer는 기존 로컬 파일 락으로 차단한다.

선택 조건 **HQNR ≥ 0.960 AND ERGAS ≤ 2.045 모든 step / 없으면 run별
best-HQNR 하나**와 cost 필수 검증은 그대로다. 모델·loss·평가에는 변경이 없다.

## 이미 학습 중인 s3에 적용

사용자가 이 커밋을 push한 후 해당 서버의 저장소에서:

```bash
git pull --ff-only
python -m donor24_upload restart --server s3 --apply
```

기존 업로더와 같은 Python 환경(gspread/google-auth 의존성이 설치된 환경)을
사용한다. s1은 `--server s1`. 이 명령은 데이터·인증·시트를 먼저 검증한 뒤
**현재 저장소/서버의 호스트 업로더만** PID/시작시각/명령어를 확인하여 교체한다.
학습 컨테이너와 GPU 프로세스는 건드리지 않는다. `--apply`를 빼면 읽기 점검만
한다. pull만으로 이미 실행된 구버전 업로더가 바뀌지는 않는다.

신규 캠페인의 `scripts/run_donor24_docker.sh`는 새 업로더를 자동 실행한다.
현재 학습 중이면 이 Docker 런처를 다시 실행하지 않는다.

## 소스 핀과 검증

`donor24/*.py`는 실행 중 학습에서 전체 SHA 검증 대상이다. 이 수정은 별도
`donor24_upload/` 모듈에서 호스트 프로세스 내부의 transport 함수만 교체하므로
그 파일들을 변경하지 않는다. 따라서 s3 pull로 학습 소스 핀이 바뀌지 않는다.

```bash
python -m unittest donor24_upload.test_transport -v
python -m donor24 smoke
```

`work_dir/donor24/<campaign>/<server>/WV3/sheet_publish_donor24/last_poll.json`에서
`transport=DONOR24_LOGICAL_TABLE_APPEND_V1`, `readback_verified=true`,
`publisher_pid`, `sheet_rows`를 확인한다. `restart_receipt.json`에는 이전/새 PID와
초기 재조회 성공 여부가 남는다. 재조회가 55초 내 끝나지 않으면 명령은 2를
반환하고 업로더는 유지하므로 로그를 확인한다. 실패를 성공으로 숨기지 않는다.
