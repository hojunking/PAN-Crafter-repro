# FH20R1 구현 및 배포 안내

기준: `PAN_FH20R1_AllServers_20Hplus_ExperimentPlan_2026-09-19.md`, 사용자가 제공한 `PAN_FH20R1_Cases_2026-09-19.csv`.

## 구현 범위

기존 `fh12/`, 모델, loss, RR/FR evaluator는 수정하지 않고 `fh20r1/`와 전용 진입점을 추가했다. 기존 두 shell의 변경은 **명시적으로 등록한 FH20R1만** 새 runner로 보내는 분기다. 배포/pull만으로 기존 학습이나 캠페인이 전환되지 않는다.

별도 원본 Registry JSON은 제공되지 않았다. case 정의·전체 Teacher SHA는 제공 CSV를, atomic block 순서는 계획서 부록 A를 기준으로 구현했다. 생성 결과는 `config/queues/FH20R1_registry.json`, `FH20R1_case_registry.csv`, 서버별 queue JSON, 145개 YAML이다. 제공 문서/CSV 자체는 수정하지 않았다.

- 145개는 모든 대안·예비를 포함한 **정의 수**다. 상호배타적인 양쪽 분기를 모두 실행하지 않는다.
- primary CORE는 총 85개(T 1 + S 84), primary RESERVE 포함 최대 125개이며, 51개 Student atomic block과 11개 준비·진단·calibration stage를 정의한다.
- `BASE/A10`, input/W/D, Teacher·Student seed, 계수와 부록 A의 pair/quartet 순서를 자동 대조한다.

## 주요 동작

| 항목 | 구현 |
|---|---|
| 시간 | 새 campaign의 유효 작업 **최소 20h**. hard deadline 없음. 해당 CORE를 모두 마친 후, 필요하면 유한 RESERVE를 block 단위로 수행 |
| 집계 | 저장된 full-state에 연결된 train/eval/diagnostic 구간과 calibration의 시간 합집합. 이전 FH12 시간·대기·업로드·중복 cache 계산은 제외 |
| 기존 Teacher | CSV whole SHA, 원 config/source/data/LP/τ/q 체인을 읽기 전용 검증. 원 Git reader와 현재 reader의 실제 forward/loss/q 비교 후 별도 bridge 발행 |
| Student | fresh U + 자기 Teacher A 복제. Teacher frozen. 기존 hard/soft/edge와 U/A gradient routing 재사용 |
| A10 | 완료 update가 10000인 다음 update, 즉 **10001번째 optimizer update**부터 A cosine LR에 1/3 적용. 누적 곱/optimizer reset 없음 |
| N2PL | s2만 donor의 A 전체를 가져오고 U는 F2와 동일 fresh 초기화. donor head 보존. exact50K에서 F2의 실제 calibration index 배열로 새 τ/q 산출 |
| s1 분기 | 지정된 기존 두 Student의 A/U 교차 진단으로 한 번 결정. 부족한 진단은 inconclusive 대안, 손상된 identity/reference는 오류로 중단 |
| s2 대안 | donor/reference 명시적 unavailable만 지정 대안으로 연결. 학습 중단·인프라 오류·무결성 오류를 과학적 실패로 바꾸지 않음 |
| 재개 | full-state/config/source/data/reference 검사 후 exact resume. 같은 run을 새 초기화로 덮어쓰지 않음 |
| 보관 | 공식 후보 50개, restart full-state 10000/24240/50000 및 last. 10000은 공식 후보에 추가하지 않음 |
| 중복 run | 전체 학습·평가 identity가 같은 완료 자산만 출처를 연결해 재사용. 새 actual updates·신규 계산시간 0, 원본 50K와 구분 |
| 운영 | 기존 실행은 정상 종료까지 drain. block 입장 전 디스크 검사, 준비 보고서·admission event·시간 ledger 기록. 종료 후 구 큐로 복귀하지 않음 |

### 평가·시트

기존 FH12 evaluator를 그대로 호출한다. RR20, FR20, native PAN/full-frame HQNR, 기존 clipping/range/support를 유지한다. shift에 맞춰 FR reference를 옮기거나 마스킹하는 평가가 아니다.

본 행은 **HQNR 최대 RAW_MAX**다. 계획서의 TARGET(HQNR ≥ .9585 안에서 ERGAS/SCC/PSNR/step 순), EXACT50K, RR_VAL_SELECTED, E_MIN_DIAG50을 별도로 보관하며, 모든 선택점의 9개 지표는 같은 정상 A/U checkpoint에서 가져온다. RR_VAL_SELECTED가 본 행의 선택 기준을 바꾸지 않는다. 로그에는 HQNR·SCC·ERGAS를 출력한다.

A/U 교차는 진단 파일로 분리한다. 기존 대각선 공식 평가를 재사용하고, 비대각선 평가에는 각 장면 metric 차이와 native c를 남긴다. 계산하지 않은 JQM/NOA/V64는 채우지 않는다.

기존 서버별 WV3 탭/GID를 사용하고 `(campaign, run)`으로 upsert/readback한다. 학습·평가 완료와 업로드 성공을 구분한다. 네트워크 실패로 학습을 다시 수행하지 않으며, 필요한 경우 upload만 재시도한다.

## 각 서버 실행

코드·config·위 두 원본 문서를 commit/push/pull한 뒤 각 서버에서 아래 **한 명령**을 실행한다. 서버는 기존 로컬 설정에서 감지한다.

```bash
bash tools/fh20r1_start.sh
```

내부에서 원 자산 검사 → 실제 GPU batch48 smoke → 로컬 진단/분기 → 해당 큐 실행을 자동 처리한다. 수동 승인 파일 생성이나 서버 공통 결과 대기는 없다. N2PL Teacher는 s2에서만 준비한다.

조회 및 업로드 재시도:

```bash
python tools/fh20r1_runner.py status
python tools/fh20r1_runner.py retry-upload
```

`tools/fh20r1_start.sh --dry-run`은 계획만 확인한다. `tools/fh20r1_preflight.py --server s1 --device cpu --check-only`는 읽기 전용 검사이며 실제 GPU 실행 준비 PASS를 뜻하지 않는다. CPU로 실제 캠페인을 시작하는 CLI는 금지한다.

실행 로그/보고서는 `work_dir/_fh20r1/<server>/`에 남는다. GPU 자산·메모리·수치 검증이 통과한 뒤 `readiness_report.json`의 `launch_ready=true`를 발행한다. CORE와 최소 시간이 끝났어도 시트가 미전달이면 `WORK_COMPLETE_UPLOAD_PENDING`; 유한 예비까지 소진해도 20h 미달이면 `CAPACITY_EXHAUSTED_BELOW20`이다.

## 검증 상태와 한계

- 생성된 145개 정의 및 51개 block 순서의 CSV/계획서 대조와 generator `--check` 통과.
- 기존 FH12 회귀 테스트 107개 통과. 기존 `fh12/`·모델·metric 파일의 diff 없음.
- FH20R1 전체 통합 테스트 125개 통과: registry 7, preflight 9, reference 15, diagnostics 14, runner/ledger/shell 40, N2 smoke 5, trainer 11, postrun/upload/reuse 24. 기존 107개와 합해 총 232개다.
- s1 실제 원본 Teacher/데이터/LP/q의 읽기 전용 검사 통과. dry-run은 runtime campaign/hold를 만들지 않음을 확인.
- s1 원 Git reader와 현재 reader의 실제 CPU 비교: train8/RR2/FR2, output/shift/metric/loss/gradient/q 21개 비교에서 최대 절대 차이 0. 이는 GPU 또는 모든 20장 평가의 비트 단위 동일성 주장과 다르다.
  실제 비교 기록: [F1_CPU_parity.json](FH20R1_Implementation_Validation_2026-09-19/F1_CPU_parity.json).
- s1 디스크 점검 당시 여유 약 25.9GiB, primary CORE 추가 보수 추정 약 9.5GiB + 안전 여유 2GiB. 실제 각 서버와 RESERVE는 실행 시 다시 검사한다.
- 실제 GPU smoke, 50K 학습, 네트워크 시트 업로드, s2–s5 로컬 자산 검증은 이 구현 작업에서 실행하지 않았다. 각 서버의 명시적 시작 후 자동 수행된다.
- 이번 작업에서는 실험 시작, commit, push, 기존 실험 파일 삭제를 수행하지 않았다. 작업 전부터 있던 무관한 research log 삭제/이동 등은 그대로 보존했다.

테스트 실행은 `pancrafter` 환경에서 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2`로 제한했다.

```bash
python -m unittest discover -s tools -p 'fh20r1_*tests.py' -v
python -m unittest discover -s tools -p 'fh12_*tests.py' -v
python tools/gen_fh20r1_configs.py --check
bash -n tools/fh20r1_start.sh tools/_run_cases.sh tools/_watchdog.sh
```
