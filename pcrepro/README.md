# PCREPRO: s3·s4·s5 PAN-Crafter 재현

기준: `research_log/PANCRAFTER_Reproduction_CasePlan_KR_2026-09-22.md`.
기존 PANDA/GF2 모델·trainer를 바꾸지 않고 별도 모듈로 추가했다.
원본 문서가 말하는 동봉 bundle은 제공되지 않아 이 MD로 recipe/생성기/검사를 구현했다.

## 실행

먼저 변경사항을 commit/push하고 각 서버에서 pull한다. 실제 데이터 binding은
[DATA_BINDINGS.md](DATA_BINDINGS.md)에 따라 각 서버에서 작성한다.
`bindings.example.json`은 빈 양식이지 검증된 원본 자산이 아니다.
인증정보는 기존 `gspread/`의 service account를 사용하며 키를 Git에 넣지 않는다.
기존 실행 환경의 Python과 외부 DLPan-Toolbox가 필요하다.

```bash
# 읽기 확인. 파일 이름만으로 자동 승인하지 않는다.
python tools/pcrepro_runner.py inspect-data

# s3 예시. s4/s5는 server 인자만 변경한다.
bash tools/pcrepro_start.sh --server s3 --bindings /actual/path/pcrepro-bindings.json

python tools/pcrepro_runner.py status --server s3
```

`start`는 **실제 실행 명령**이다. commit된 불변 worktree에서 기존 GF2 worker의
정확한 소유권을 검사하고, 해당 서버의 tagged watchdog 재시작을 막은 뒤,
optimizer 경계의 atomic full-state 보존을 확인하고 새 학습을 시작한다.
알 수 없는 launcher/다른 서버/s1·s2 route/GPU 점유가 있으면 추정하여 죽이지 않고 차단한다.
임의 `pkill`, SIGKILL, 기존 결과 삭제는 하지 않는다. 일반 구현/test 명령은 전환을 실행하지 않는다.

새 campaign에는 공유 Teacher·공통 clock·40h 종료·서버 간 barrier가 없다.
동일 물리 호스트의 중복 runner/worker만 nonblocking local lock으로 막는다.
환경을 가짜로 맞추거나 GPU 이름을 바꾸지 않고 실제 runtime을 기록한다.

## 순서와 선택 기준

| 서버 | 매 cycle 순서 |
|---|---|
| s3 | WV3 fresh50K → 같은 WV3의 WV2 평가 → QB fresh50K → GF2 fresh50K |
| s4 | QB → GF2 → WV3 → WV2 |
| s5 | GF2 → WV3 → WV2 → QB |

cycle0은 모두 seed2025, 이후 `1000000 + 3*(cycle-1) + (server_number-3)`.
WV2는 같은 서버·cycle의 WV3 checkpoint 두 개만 평가하며 training update/비용은 0이다.
WV3 실패 시 해당 WV2만 차단하고 다른 데이터셋은 계속한다.
데이터셋이 모두 차단되면 busy loop 없이 pause한다.

이번 재현의 primary는 `EXACT_50000`, secondary는 1000-step native validation grid의
`RR_VAL_ERGAS_MIN`(동률은 먼저 나온 점)이다. **기존 campaign의 HQNR 선택 규칙은 바꾸지 않는다.**
training에는 test FR/HQNR 평가를 넣지 않으며 `HQNR=NOT_MEASURED(TRAIN_VAL_ONLY)`로 표시한다.
최종 두 선택점은 같은 SHA이면 한 번만 평가한다. 성능 기준으로 다음 seed를 취소하지 않는다.

## 정지·재개·장애

```bash
python tools/pcrepro_runner.py control --server s3 --command STOP_NOW_SAFE
# 또는 STOP_AFTER_CURRENT_RUN / STOP_AFTER_CYCLE
python tools/pcrepro_runner.py control --server s3 --command CONTINUE
bash tools/pcrepro_start.sh --server s3

python tools/pcrepro_runner.py report --server s3
python tools/pcrepro_runner.py retry-uploads --server s3
# runner가 멈추고 GPU가 비어 있을 때만 미완료 평가를 명시적으로 재시도
python tools/pcrepro_runner.py retry-evaluations --server s3
```

`CONTINUE`는 설정만 변경한다. 멈춘 runner는 `start`로 다시 켠다.
임시 학습 I/O 실패는 같은 full-state로 최대2회 재시도한다. NaN/OOM에서 batch,
precision, geometry를 바꾸지 않는다. source/data mismatch는 차단한다.
postrun은 scene cursor를 보존하고, 임시 평가 실패는 후속 cycle에서 한 건씩 재시도한다.
반복 결정적 평가 실패는 `blocked_evaluations`에 남기며 임의 재학습하지 않는다.
디스크는 최소20GiB와 추정 다음 저장 비용 중 큰 요구량을 적용한다.

## Sheet 및 결과

`PC-Repro-s3/s4/s5` 전용 탭만 사용한다. 기존 WV3/GF2 탭은 수정하지 않는다.
run당 한 행, 두 선택점 별도 열, 실제 UTC/KST 날짜·seed·cycle·update·SHA·비용을 기록한다.
RR Q4/Q8·RMSE·CC, FR HQNR/Dλ/Ds/JQM을 해당 선택점에서 함께 계산한다.
JQM은 **SRF-substitute 보조 지표**이며 SIPSA와 같은 정의라고 표기하지 않는다.
표시는 소수4자리이고 원값은 반올림 없이 저장·업로드한다.
Sheet 장애는 학습을 중단하지 않으며 durable spool과 raw-value readback으로 완료를 확인한다.

주 파일은 `work_dir/<run_id>/{meta,checkpoints,resume,official}`,
운영 파일은 `work_dir/_pcrepro/<server>/`에 있다.
마지막2 resume state + best-validation model + exact50K model/fullstate를 보존한다.
cycle report는 JSON/JSONL/CSV로 저장하며 seed2025와 이후 반복을 분리한다.

## 검증 범위와 한계

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m unittest discover -s pcrepro -t . -p 'test_*.py'
bash -n tools/pcrepro_start.sh
```

실제 C128·D224 모델 실측: C4 **7,129,476**, C8 **7,207,816** total/trainable parameters.
기존 frozen gather convolution을 정확한 `unfold`로 교체해 고정 weight/bias 파라미터를 없앴다.
7.17M에 맞추기 위해 depth/FFN을 줄이지 않았다.

preflight는 원본 SHA/tensor·실제 GPU·소규모 MARs forward/backward·공식 평가 의존성을 검사한다.
CPU 단위 테스트와 batch1 smoke가 batch48 GPU 메모리/50K 성능을 보장하지는 않는다.
실제 서버 stop/resume·본 학습·live Sheet readback은 실행 후 검증해야 한다.
256/512 profile의 FLOPs는 Conv/Linear/attention MAC×2 **부분 집계**이며 제외 연산을 명시한다.
전체 논문 FLOPs와 동등한 비용이라고 주장하지 않는다.
WV3 RR20 중19 MAT만 대응된 경우나 WV2 paper set 미확인은 그대로 표시한다.
