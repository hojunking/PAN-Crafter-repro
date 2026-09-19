# FH12 / FH20R1 보충 평가·업로드

진행 중 학습과 원래 checkpoint 선택/공식 9개 지표를 바꾸지 않고, 완료된
run의 RMSE·CC·JQM을 추가 평가한다. 기존 누락 결과와 이후 완료 결과를
동일하게 처리한다. **현재 범위는 WV3 FH12/FH20R1이며, 과거 다른 캠페인은
이 worker가 임의로 재평가하지 않는다.**

## 서버별 한 번 실행

현재 학습은 Git HEAD까지 검사한다. 실행 중 `git commit/pull`로 HEAD를
변경하면 기존 postrun/resume/reference 검증이 실패할 수 있다. 이 보충
패키지는 `reporting_extra/`, `tools/extra_metrics*`만 배포하고 HEAD를 유지한다.
기존 학습 파일, `_watchdog.sh`, 모델, metric 구현, checkpoint는 덮어쓰지 않는다.

### Git으로 전달할 때 (실행 중 캠페인 권장)

이 변경은 `reporting-extra-metrics` 브랜치에만 커밋한다. 학습 중인 로컬
`main`의 HEAD와 기존 index는 유지한다. 전달하는 서버에서:

```bash
git push origin reporting-extra-metrics
```

받는 각 서버의 PAN-Crafter 디렉터리에서:

```bash
git fetch origin reporting-extra-metrics
git archive FETCH_HEAD reporting_extra tools/extra_metrics.py tools/extra_metrics_start.sh tools/extra_metrics_bundle.py | tar -x
bash tools/extra_metrics_start.sh
```

`fetch`와 경로를 한정한 `archive` 추출은 현재 브랜치/HEAD/index 및 학습
파일을 바꾸지 않는다. **학습 중에는 이 브랜치로 checkout/merge/pull하지 않는다.**
첫 설치 이후 자체 수정한 보충 파일이 있다면 추출 전에 보관한다.

### tar 파일로 전달할 때

각 서버의 PAN-Crafter 디렉터리에 배포 tar를 풀고 실행:

```bash
tar -xzf /path/to/pan-extra-metrics-overlay.tar.gz
bash tools/extra_metrics_start.sh
```

`gspread/server.txt`의 기존 s1–s5 설정으로 로컬 서버만 처리한다.
기존 pancrafter Python 환경, 데이터/checkpoint, Google Sheets 서비스 계정이
필요하다. 환경 위치가 다르면 `PYTHON=/absolute/path/to/python`을 지정한다.

실행 명령 하나가 CPU worker와 **독립된 5분 주기 재시작 cron**을 등록한다.
기존 학습 cron/runner/대기열에는 손대지 않는다. cron 권한이 없으면 worker는
시작되지만 `restart_warning`을 출력하므로 재부팅 후 같은 명령으로 시작한다.
이미 시작된 경우 중복 학습/평가를 만들지 않는다. `--no-cron`도 지원한다.

```bash
python tools/extra_metrics.py status
tail -f work_dir/_extra_metrics/s1/worker.log
# 실제 run별 계산/업로드 상세는 같은 폴더의 <run_id>.log
```

worker는 CPU 2 threads, nice +15, GPU 비활성으로 순차 처리한다. 실행 중인
학습을 종료하거나 GPU를 점유하지 않는다. CPU/RAM/디스크 사용은 추가된다.
공식 완료 감지는 120초 간격, 실패한 작업은 15분 후 재시도한다. 작업 실패가
나머지 누락 결과를 막지 않는다. 공식 원본 행 자체가 없으면 해당 캠페인의
기존 업로더로 원본 검증·업로드를 한 번 수행한 뒤 보충 지표를 채운다.
동일 Run의 충돌 행이나 불완전한 본표 스키마는 임의 복구하지 않고 실패한다.

## 평가와 기록

- 대상: 실제 50,000 update 학습 및 공식 postrun 완료 run. reuse 링크 제외.
- 선택: RAW_MAX, TARGET, EXACT50K, RR_VAL_SELECTED, E_MIN_DIAG50.
  각 선택점의 원본 step·전체 checkpoint SHA와 일치해야 하며 중복 step은 한 번만 계산.
- 본표 RMSE/CC/JQM은 본표의 기존 9개 지표와 같은 **RAW_MAX checkpoint**.
- RMSE/CC: RR 전체 20장, 기존 DN(2047)·dim_cut=21 crop(`20:-21`),
  장면별 전체 밴드 RMSE/Pearson CC를 구한 뒤 장면 평균.
- JQM: FR 전체 20장, native PAN/LRMS, masking/shift/crop 없음.
  기존 `tools.metrics.jqm`의 **NNLS-normalized SRF 대체 변형**이다.
  SIPSA의 원 SRF 기반 수치와 완전히 동일하다는 주장은 하지 않는다.
- 원 checkpoint를 CPU FP32로 재추론한다. 원 GPU 출력과 bitwise 동일하다고
  주장하지 않으며 backend/변형/protocol을 JSON과 시트 provenance에 명시한다.
- TARGET 부적격은 빈칸이며 0으로 대체하지 않는다. 추가 지표는 선택 기준이 아니다.
- 원 config/grid/선택 JSON/모델/데이터/수치 코드 SHA와 라이브러리 버전을
  검증한다. historical Git HEAD가 다를 수 있으나 수치 코드 불일치는 거부한다.
  기존 학습의 source identity 검증을 완화하거나 다시 쓰지 않는다.

별도 산출물:

```text
work_dir/<run>/supplemental_metrics/
  step_<step>.json       # 재시도 시 재사용; 원본/변형 체크섬 검증
  report.json           # 5개 선택점 + per-scene/평균/표준편차
  upload_receipt.json   # 실제 Sheet readback 검증 결과
work_dir/_extra_metrics/<server>/
  enabled.json
  status.json           # RUNNING / RETRY_PENDING / VERIFIED, 로그 위치
  worker.log
  <run>.log
```

시트의 정확한 campaign/run 및 선택점 SHA 행에만 추가한다. 원 9개 metric,
NOA/V64/Notes는 보존한다. 숫자는 RAW 정밀도로 업로드하고 표시 형식만
`0.0000`을 적용한다. 수식·검증규칙·보호/병합 셀 등은 덮어쓰지 않고 실패한다.
업로드는 기존 `.gspread_write.lock`을 공유하며 업로드 후 값을 다시 읽어
검증해야 VERIFIED가 된다. 기존 평가 JSON·checkpoint는 수정하지 않는다.
본표 행 자체가 없어서 기존 업로더로 복구한 경우에만 그 업로더가 기존
official/upload_receipt.json을 갱신할 수 있다.

## 검증

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  python -m unittest reporting_extra.test_evaluation reporting_extra.test_upload \
  reporting_extra.test_worker -q
```

원본 지표 수정, 학습 재개/중단, GPU 실행, 신규 모델 학습은 포함하지 않는다.
