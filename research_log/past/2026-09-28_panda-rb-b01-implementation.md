# PANDA B01 구현·검증 및 서버 인계

작성: 2026-09-28. 대상은 `PANDA_REBUTTAL_STAGED_S135_2026-09-28`의 MASTER/DETAILED/registry/CSV다. 원 계획 bundle, 제출본, 과거 결과, 기존 Sheet를 수정하지 않았다.

후속 상태: 사용자 기동·커밋 요청에 따라 **s1은 2026-09-28 19:21 KST에 시작**했다. 아래의 미시작/미커밋 표기는 최초 구현 검증 당시의 기록이며, 실제 기동 증거는 `results_log/2026-09-28_panda-rb-b01-s1-launch.md`를 따른다. s3/s5는 아직 기동하지 않았다.

## 구현 범위

신규 `panda_rb/` 및 `tools/panda_rb_*.py`를 추가했다. 기존 FH12 모델·입력·loss·평가 수식은 변경하지 않는다. 기존 셸 파일 변경은 `_watchdog.sh`, `_run_cases.sh`에 **명시적 B01 소유권 파일이 있을 때만** 과거 큐 재기동을 차단하는 분기를 추가한 것이다. 기존 사용자 수정/R2 분기를 보존했다. git pull만으로 실행되거나 소유권이 등록되지 않는다.

| 계획 항목 | 구현/검증 |
| --- | --- |
| s1/s3/s5, 24 Student + 48 stress curves | registry·CSV·순서 일치 확인, 서버별 8회/16곡선 후 `STOP_FOR_REVIEW`, s2/s4 거부 |
| 동일 F1 exact50K | model/fullstate/config/calibration/q/data/LP/source 전체 SHA 검증, 경로와 무관한 공통 identity, F3/F5 대체 금지 |
| QFULL/QMEAN/QSHUF/QESUR | 전체 train×4 views, 단일 고정 F1 e-cache, view/e-quintile 내 deterministic derangement, view별 multiset 보존 |
| 실제 변경 범위 | `U = mean(H+K+.002*w*E)`, `A = mean(w*H)`의 geometry weight만 변경. 기존 pixel e/d/a 및 gradient routing 재사용 |
| paired block | 동일 U/A 초기 SHA, Teacher/data identity, 전 50K sample/view stream digest, 별도 permutation RNG |
| checkpoint | 주 결과 exact50000, 보조 min validation1080 ERGAS/동점 시 앞 step. 1010 배수 49회+50000. FR test sweep/최고 HQNR 선택 없음 |
| 진단/재개 | 고정 0/1K/10K/25K/50K 진단, RNG·buffer·gradient 보존, optimizer/scheduler/sampler/RNG/fullstate 저장, 기술 실패 동일 seed 재개 |
| native | RR/FR 각각 20장, RR `[20:-21]`, FR 원 PAN full512 무마스킹. float32 CHW DN raw 및 SHA |
| stress | exact50K의 A_ON/A_ZERO, 49×20 관측. PAN 한 번 shift 후 LP 재생성, MS/GT 불변, 고정192 ROI. dy/dx L1 sum 및 보수적 PAN/LP/HP coverage 기록; adaptive mask/clamp 없음 |
| 추가 metric | 기존 공식 9종과 RMSE/CC/JQM. JQM은 NNLS/SRF 대체 변형으로 명시; SIPSA 수치와 동일 프로토콜이라고 주장하지 않음 |
| 집계 | Student seed가 통계 단위. sample SD, case−QFULL paired delta 및 서버별 delta, A_ON−A_ZERO, radius 내 방향평균 후 seed 통계. 실패/missing n 명시 |
| 실행/보관 | 고정 Docker image, 내용 SHA로 동결한 소스, Git/외부 자산 읽기 전용. 50 val 후보·성공한 resume state 보존. 기존 프로세스 종료/자동 checkpoint 삭제 없음 |

이번 계획의 exact50K/val-ERGAS 규칙은 **새 B01에만** 적용한다. 과거 HQNR selector 및 표는 변경하지 않았다. 학습 중에는 validation ERGAS만 계산하고, 선택점 native 후처리에서 HQNR/SCC/ERGAS를 함께 출력한다.

## 실제 확인 결과

- 원 F1 SHA: `04ef8e756ae8b229c67ef14daf1912f0658f2e5c8bfe30afa9e0b476486fc519`.
- `tau_R=0.012118559330701828`, `q_ref=0.4532603621482849`.
- raw q `(9714,4)`, train/val/RR/FR = `9714/1080/20/20`.
- 실제 QMEAN 전체 집단 평균 `0.4996898875487226`; 0.5 또는 calibration subset 평균을 사용하지 않는다.
- 실제 RTX 4090, F1 + PLH W104 D122, native64/16, batch48 QFULL smoke: prediction/correction/loss 차이 0, gradient 최대 차이 `7.45058e-9`, AdamW parameter update 차이 `2.18279e-11` (허용 `3e-6`). Teacher 불변. 최대 GPU 할당 `3,255,696,384 bytes`.
- 원 Git `a5e9f858...`의 분리된 reader와 현재 reader를 실제 GPU로 비교: train8×4 views/AXIS16, RR2, FR2의 prediction/correction/metric/q/loss/output gradient **21개 항목 모두 차이 0**. 초기 state SHA도 일치했다.
- 신규 전체 테스트 103개와 재사용 FH12 모델·학습·평가 회귀 36개 통과. s1/s3/s5 dry-run 및 shell 문법 검사 통과. 원 계획 bundle의 체크섬 18개 모두 일치했다.
- smoke는 임시 binding/폐기 가능한 모델로 수행했다. 등록 실험, live asset cache, 소유권 파일을 만들거나 50K 학습을 시작하지 않았다.

검토 중 실제 metadata가 4열인데 계획 digest가 2열인 오류를 발견해 수정했다. 실제 `[index,view,HV,HV]`에서 `[index,view]`를 hash하고 고정 augmentation flag를 검증한다. 최초 진단 실패 전에 step0 fullstate를 남기도록 보강했다. 재개 컨테이너 중복 기동, 원 Git reader용 `.git` mount, val-selected symlink의 ZIP 누락, Q8 집계 누락도 수정했다.

단위·통합 테스트는 `tools/test_panda_rb_*.py`. 실제 학습 loop를 작은 CPU fixture로 중단/재개해 최종 state 일치를 확인하며, native 40 raw 저장/alias/tamper 및 stress 980 관측 보존도 검사한다. 합성 fixture 결과를 실제 실험 metric으로 보고하지 않는다.

## 실행

현재 **본 실험은 미시작**이다. 아래 명령은 사용자가 실행할 때만 실제 기동한다.

배포 시 신규 모듈·tools·계획 bundle뿐 아니라 현재 미추적 상태인 기존 `reporting_extra/evaluation.py` 의존성도 대상 서버에 있어야 한다. 관련 없는 기존 dirty 변경/연구 로그 이동은 이 작업에서 커밋하지 않았다. 아직 commit/push하지 않았으므로 다른 서버에 코드가 전달되었다고 볼 수 없다.

```bash
# 읽기 전용 계획/기동 구성 확인
python3 tools/panda_rb_start.py --server s1 --dry-run

# s1 시작: binding → 전체 공통 e/6-seed maps → parity/GPU smoke → 고정 F1 probe → 본선
python3 tools/panda_rb_start.py --server s1
docker logs -f panda-rb-b01-s1
```

Docker image ID는 `sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887`이다. 자동 image pull/다른 Torch fallback은 하지 않는다. s3/s5의 실제 GPU·동일 image 호환성은 해당 서버 preflight 통과 전까지 미확인이다. 다른 image가 필요하면 한 서버만 예외 처리하지 말고 공통 runtime 정책을 별도 결정해야 한다.

학습 전 검증은 자동 실행된다. 별도 사람이 작성하는 승인 파일은 필요 없다. 이미 실행 중인 GPU 작업/과거 큐가 있으면 **종료하지 않고 거부**한다. 여유 저장공간을 체크하며, 8회 기준 초기 보수적 추정은 약 29.7 GiB다. 추가 기술 재시도는 보존량을 늘리므로 기동 때마다 잔여 공간을 다시 계산한다.

### 공통 F1/cache를 s3/s5로 전달

원본 F1과 전체 train e-cache는 s1에서 한 번만 만든다. s3/s5가 다른 GPU로 e/rank를 재생성하지 않는다. s1 자동 preflight가 common maps를 완료한 뒤 다음을 수행한다.

```bash
python3 tools/panda_rb_transfer.py export work_dir/RB_B01_F1_package
```

이 package 디렉터리를 사용자가 선택한 방식으로 s3/s5에 복사한다. 여기서는 SSH/네트워크 복사를 수행하지 않았다. native H5는 중복 포장하지 않고, 대상 서버의 정확한 파일 SHA를 검사한다. 대상에서 `local_data_paths.json`을 다음 구조로 지정한다(아래는 **경로 형식 예시**이며 실제 파일을 적어야 한다).

```json
{"train":"/absolute/train.h5","val":"/absolute/val.h5","rr":"/absolute/rr20.h5","fr":"/absolute/fr20.h5"}
```

```bash
python3 tools/panda_rb_transfer.py import work_dir/RB_B01_F1_package --data-paths local_data_paths.json
python3 tools/panda_rb_start.py --server s3
# s5에서는 마지막 --server만 s5로 지정
```

### 장애·완료 및 결과

- signal 중단은 같은 시작 명령으로 resume. 기술 오류는 원인을 확인한 뒤 `--retry-technical`; 새 seed를 뽑지 않는다.
- 수치 발산은 원래 seed의 실패로 보존한다. 다음 등록 case만 진행한다.
- 서버당 등록된 8회/16곡선을 처리하면 STOP. 다음 dataset/RB 단계/추가 seed를 자동 시작하지 않는다. 실패가 있으면 완료 수와 실패 수를 구분한다.
- 재개 시 같은 고정 소스/image/binding을 요구한다. source를 수정한 뒤 기존 B01의 결과를 섞지 않는다.
- 출력은 `work_dir/_panda_rb/20260928/B01/`. 기존 작업, 제출 표, Sheet에 쓰지 않는다.
- Docker 내부 `tools/panda_rb_runner.py report --server s1` 또는 `package --server s1 --output <새 ZIP 경로>`로 실제 결과와 SHA를 만든다. 측정 결과가 하나도 없으면 ZIP 생성을 거부한다. package는 selected weights/native raw/stress scalar·고정 raw subset/진단/common maps/frozen source를 포함하며, 모든 resume·val 후보는 로컬에 계속 보관한다.

## 아직 실행하지 않은 검증

24회 전체 50K, 48개 실제 stress 곡선, 전체 train e-cache 생성, 고정256 validation F1 분석, s3/s5의 GPU 및 전송 후 SHA 검증은 본 실행 시 수행한다. 실제 six-seed 성능·hardware parity·장시간 저장공간 사용량을 이미 검증했다고 해석하면 안 된다. 커밋·push·원격 기동·Sheet 업로드도 수행하지 않았다.
