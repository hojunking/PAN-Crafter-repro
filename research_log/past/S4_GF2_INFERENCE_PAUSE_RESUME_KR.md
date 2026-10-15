# s4: 현재 실험 안전 일시정지 → 논문용 GF2 추론·추출 → 동일 실험 재개

작성일: 2026-09-24  
문서 유형: **s4에서 실행할 작업 지시서. 이 문서 작성만으로 서버 작업이 수행된 것은 아니다.**  
작업 ID: `PANDA_S4_GF2_PAPER_EXPORT_20260924_v1`

## 0. 실행 목적과 완료 조건

s4에서 현재 진행 중인 실험을 폐기하거나 새 cycle로 넘기지 않는다. 진행 중인 run의 재개 가능한 상태를 안전하게 저장하고 GPU를 잠시 반납받아, 현재 `paper` 표의 GF2 Ours checkpoint로 RR20·FR20 추론을 수행한다. 필수 결과 저장과 검증이 끝나면 **같은 실험·run·attempt·seed·sampler 위치·optimizer 상태에서 기존 무한 반복을 재개**한다.

작업 범위는 s4 하나뿐이다. s1/s2/s3/s5의 실험, 원본 checkpoint, 기존 queue, 기존 Sheet 결과는 변경하지 않는다. 새로운 GF2 학습, finetuning, checkpoint 재선택도 하지 않는다.

완료는 다음 두 항목을 별도로 판정한다.

1. **결과 추출:** 지정 SHA의 정상 A_ON 추론으로 GF2 full-band 예측 40개와 장면·모델·데이터 출처 및 재평가 결과가 저장되어 있다.
2. **기존 실험 복귀:** 이전 실행이 실제로 다시 진행 중이며, 중단 전 run의 이어 학습 또는 이미 끝난 run의 평가/다음 case 진행이 확인된다.

렌더링 설정 미확정, PNG 생성, ZIP 압축, 네트워크 업로드 때문에 학습을 계속 멈춰 두지 않는다. 추론 실패 시에도 학습 재개가 안전한 상태라면 GPU를 정리한 뒤 기존 실험을 복귀시킨다. **추출 성공과 복귀 성공을 하나의 상태로 합치지 않는다.**

## 1. 반드시 사용할 모델

아래는 직전 `paper`-원결과 대조에서 고정한 추출 대상이다. 실행 직전에 Sheet가 변경되었다면 이를 기록하고 이 작업의 대상을 임의로 다른 모델로 바꾸지 않는다.

| 항목 | 고정값 |
|---|---|
| 데이터셋 / 원 실행 서버 | GF2 / s4 |
| Run | `GFP40_GF2_S4_B09_MIX_G050_FT20K_RS98202_v1` |
| 모델 | 전체 Student Aligner + 동기화된 PAN/LPAN/HPAN frontend + PLH/W104D122 reconstruction network |
| 선택점 | `EXACT_FINAL` |
| 이 FT run의 local update | **20000** |
| 누적 Student update | 120000 |
| 모델 SHA256 | `493b87c3500bc70491f929d1bfbf04604a5f39167e36b59e96bc8050db577009` |
| 정량표 연결 | `paper!B52:L52` ↔ `GF2-P40-s4` 10행 |
| 밴드 수 / DN 범위 | 4 / 0–1023 |
| 평가 | native RR20, native FR20, FP32, 정상 A_ON |

원 실행 루트를 `GF2_ROOT`라고 하면 확인할 상대 경로는 다음이다.

```text
work_dir/GFP40_GF2_S4_B09_MIX_G050_FT20K_RS98202_v1/
  candidates/20000/model.safetensors
  candidates/20000/identity.json
  meta/config.resolved.yaml
  official/summary.json
```

**`candidates/120000`, 다른 run의 B20 C03, RAW_AUX, Teacher 출력은 대체재가 아니다.** A와 U를 같은 checkpoint에서 함께 읽는다. 학습용 G050/PANMIX를 추론 입력에 적용하지 않으며, raw/native PAN과 원래 MS 좌표계를 유지한다.

### 재평가 대조값

다음은 P40 B09 원 선택점의 값이다. 렌더링 PNG가 아니라 저장된 full-band 예측으로 대조한다.

| 지표 | 기준 원값 |
|---|---:|
| HQNR | 0.9650681941034535 |
| D_lambda | 0.01886538721048394 |
| D_s | 0.016349904469339097 |
| ERGAS | 0.5510757826437531 |
| SCC | 0.9935579252100899 |
| SAM | 0.625287441360411 |
| Q4 | 0.9857753213474446 |
| PSNR | 44.54156976203516 |
| SSIM | 0.9860308932072043 |

알려진 불일치: 직전 감사에서 `paper!L52`에는 0.986115768538626이 있었으나, P40 B09의 SSIM은 위의 0.9860308932072043이다. 추론값을 이전 표의 잘못 섞인 숫자에 맞추지 않는다. 이 작업에서 Sheet를 수정하지 말고 보고서에 남긴다.

## 2. 작업 전: 현재 실행과 두 runtime을 실제로 확인

이 문서가 가정하는 우선 경로는 s4의 `g23sens` sensitivity runner다. 저장소에서 해당 구현을 확인했지만, **현재 s4의 실제 PID·GPU·run·cycle은 여기서 확인되지 않았다.** 실행 담당자가 다음을 먼저 확인한다.

- `nvidia-smi`로 사용 중인 물리 GPU UUID와 PID를 확인한다.
- process의 command/cwd 또는 Docker label로 현재 GPU를 소유한 runner와 하위 worker를 특정한다.
- 현재 sensitivity의 실행 루트 `SENS_ROOT`, 원 GF2의 호환 실행 루트 `GF2_ROOT`를 각각 찾는다. 서로 같다고 가정하지 않는다.
- 원래 interpreter, Docker container ID/image content ID, GPU UUID, 실행 command/working directory, `--no-upload` 여부를 기록한다.
- 원래 실행이 진행 중인지, 이미 사용자가 멈춘 상태인지 구분한다. 원래 멈춘 작업을 이번 요청을 이유로 임의 시작하지 않는다.

호스트에서의 읽기 전용 확인 예시:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory --format=csv
ps -eo pid,ppid,pgid,lstart,args

docker ps --filter label=org.pancrafter.g23sens.server=s4 \
  --format '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}'
```

Docker 정보는 필요한 항목만 기록한다. credential이 포함될 수 있는 전체 환경변수나 전체 `docker inspect` 출력은 결과 ZIP에 넣지 않는다. 위 명령은 서버에서 실행하는 명령이며 ChatGPT 작업 컨테이너에서 s4 상태를 보여주는 명령이 아니다.

s4 G23의 실제 공유 lane은 `work_dir/g23sens/s4/`다. 확인 대상은 다음과 같다.

```text
state.json
runtime_release.json
runtime_bindings.json
preflight.json
STOP_NOW_SAFE          # 존재할 수도, 없을 수도 있음
STOP_AFTER_RUN         # 존재할 수도, 없을 수도 있음
runs/<active_run_id>/attemptNNN/config.json
runs/<active_run_id>/attemptNNN/last/identity.json
runs/<active_run_id>/attemptNNN/last/training_state.pt
```

`attemptNNN`을 000으로 가정하지 말고 `state.json`의 실제 attempt와 `g23sens.common.run_dir` 규약으로 찾는다.

현재 owner가 `g23sens`가 아니면 아래 stop/start를 적용하지 않는다. 실제 owner의 안전 pause/full-state resume 경로를 코드에서 확인하여 같은 불변조건을 만족시킨 뒤 진행한다. 지원 경로가 없으면 **추론은 시작하지 않고 BLOCKED로 보고**한다. generic kill이나 새 runner로의 전환은 하지 않는다.

## 3. 일시정지 전 CPU 사전 점검

학습을 오래 멈추지 않도록 다음은 GPU를 건드리지 않고 먼저 준비한다.

1. 지정 GF2 checkpoint와 identity, resolved config, 원 데이터 manifest 및 RR/FR·LP 자산의 존재·SHA를 확인한다.
2. `identity.json`의 source/config/data 정보를 확인하고 해당 모델을 읽을 호환 frozen runtime과 Python 환경을 찾는다. 현재 sensitivity runtime으로 원 GF2를 무조건 읽지 않는다.
3. 원 평가에 필요한 DLPan 코드 등 dependency의 존재를 점검한다. 누락 시 자동으로 다른 metric·데이터로 대체하지 않는다.
4. 기존 예측 배열이 있으면 먼저 run/step/SHA/data/scene provenance를 확인한다. 완전하게 검증된 배열이면 재추론 없이 재사용할 수 있다.
5. 별도 export 작업 디렉터리와 로그, transaction journal을 준비한다. 원 run의 `candidates`, `official`, `meta` 아래에는 쓰지 않는다.
6. 원 실험의 재개 명령과 복귀 검증 절차를 pause 이전에 확정한다.

실제 경로를 확인한 뒤 사용할 SHA 점검 예시:

```bash
# GF2_ROOT는 위 단계에서 실제로 확인한 원 run 접근 루트이다.
: "${GF2_ROOT:?실제 GF2 runtime/work_dir 루트를 먼저 지정}"
GF2_RUN='GFP40_GF2_S4_B09_MIX_G050_FT20K_RS98202_v1'
GF2_SHA='493b87c3500bc70491f929d1bfbf04604a5f39167e36b59e96bc8050db577009'
CKPT_DIR="$GF2_ROOT/work_dir/$GF2_RUN/candidates/20000"
printf '%s  %s\n' "$GF2_SHA" "$CKPT_DIR/model.safetensors" | sha256sum -c -
```

단순 SHA 점검만으로 전체 데이터/환경 검증이 끝난 것은 아니다. 원 loader의 config·state·identity 검사도 그대로 수행한다.

## 4. Pause transaction 기록

별도 디렉터리 예: `work_dir/_qualitative_interlude/s4/<실제 UTC timestamp>/`.

`transaction.json`에는 최소한 다음 정보를 실제 값으로 저장한다.

```text
operation_id, created_utc, current_phase
was_running_before, original_owner, original_command, original_cwd
original_container_id, original_image_id, original_gpu_uuid
sens_root, gf2_root, source_identity, binding_sha256
active_run_id, cycle, position, seed, attempt, completed_update
original_stop_markers: 존재 여부·내용·SHA
created_stop_marker: 경로·내용·SHA·이번 작업 소유 여부
pause_requested, pause_verified, paused_update, resume_bundle_sha256
export_status, metric_check_status, render_status, transfer_status
resume_requested, resume_verified, resumed_from_update, final_status
```

`state.json`, active config, runtime bindings, 원래 stop marker는 pause 전후의 작은 snapshot으로 보존한다. snapshot은 감사 자료이지 나중에 통째로 lane에 덮어씌우는 복구 파일이 아니다. 현재 training state를 이전 snapshot으로 되돌려서는 안 된다.

이미 `STOP_NOW_SAFE` 또는 `STOP_AFTER_RUN`이 있으면 그 뜻을 확인한다. 기존 사용자 중단 요청을 이번 작업의 marker로 오인하지 않는다. 새 요청이 겹치거나 소유권이 모호하면 자동 resume 승인을 만들지 않는다.

## 5. 현재 실험을 안전하게 일시정지

### 5.1 확인된 G23 명령

동일 공유 lane에 접근하는 실제 runtime/interpreter에서 실행한다.

```bash
# SENS_ROOT와 PYTHON은 기존 owner에서 확인한 값이다.
: "${SENS_ROOT:?현재 G23 frozen runtime을 먼저 지정}"
: "${PYTHON:?기존 환경의 Python 실행 파일을 먼저 지정}"
"$PYTHON" "$SENS_ROOT/tools/g23sens_runner.py" status --server s4
"$PYTHON" "$SENS_ROOT/tools/g23sens_runner.py" stop --server s4 --safe-now
```

호스트 Python 환경이 적합하지 않고 원 서비스가 Docker라면, **실행 중인 그 컨테이너**에서 동일한 stop 명령을 호출한다. 임의의 새 이미지로 control 명령을 실행하지 않는다.

```bash
# SENS_CID는 확인한 기존 s4 container ID이다. SENS_ROOT는 container 내부에서도 유효해야 한다.
docker exec "$SENS_CID" python "$SENS_ROOT/tools/g23sens_runner.py" \
  stop --server s4 --safe-now
```

위 두 방법 중 하나만 사용한다. 명령은 `STOP_NOW_SAFE`를 생성하는 **요청**이지, 즉시 GPU가 반환되었다는 확인이 아니다.

### 5.2 안전중단 완료 판정

현재 G23 trainer는 안전 경계에서 model/U·A, optimizer, scheduler, criterion, RNG, sampler, 누적 시간과 identity를 `last` resume bundle로 보존한다. `--safe-now`는 현재 처리 중인 update/평가 경계의 정리를 기다릴 수 있다.

추론으로 넘어가려면 모두 확인한다.

- 현재 owner가 지원하는 paused 상태에 도달한다. G23에서는 통상 `PAUSED_SAFE`, admission 전이면 `PAUSED_BY_OPERATOR`도 가능하다.
- 진행 중 학습 run이 있었다면 `last/identity.json`과 `last/training_state.pt`가 완전하며, identity에 기록된 training-state SHA가 파일과 일치한다.
- full-state의 update, scheduler 위치, sampler 위치와 config/source/binding hash가 맞는다. 검사는 신뢰한 로컬 resume bundle에만 수행한다.
- paused update를 `j`로 기록한다. pause 요청 직전보다 한두 update 진행되었더라도 저장·확인한 `j`를 복귀 기준으로 삼는다.
- train/postrun/preflight worker와 원 서비스가 종료되거나 확실히 quiescent하며, 해당 GPU를 점유하지 않는다.
- 서비스/runner/GPU lock의 **실제 holder**가 남아 있지 않다. lock 파일의 존재만으로 점유 여부를 판정하지 않고 파일도 지우지 않는다.

이미 학습이 50K 끝나고 postrun 중이었다면 그 상태를 보존한다. 재개 시 학습을 다시 할 필요가 없고 미완료 평가 단계만 이어가야 한다.

안전중단 확인에 시간이 오래 걸리면 관측 상태를 보고하고 기다리거나 작업을 보류한다. 시간 초과를 이유로 강제 종료하지 않는다.

### 금지

`kill -9`, 광범위 `pkill`, `docker kill`, `docker stop`의 timeout 강제종료, `SIGSTOP`, `docker pause`를 pause 대체로 사용하지 않는다. 특히 프로세스 정지/SIGSTOP이나 Docker pause는 GPU 메모리 반환을 보장하는 안전 checkpoint 경로가 아니다. 기존 watchdog·다른 lane·전체 Docker daemon을 끄지 않는다.

## 6. GF2 inference-only worker

추론은 기존 sensitivity supervisor와 **분리된 단발 worker**에서 실행한다. 종료 시 그 worker의 CUDA context가 사라지도록 한다. training 중인 checkpoint를 수정하거나 메모리 안의 sensitivity model을 GF2 모델로 갈아끼우지 않는다.

### 사용할 기존 코드 경로

직전 감사 및 이번 코드 확인 기준의 호출 대상이다. 아래 항목은 함수이지 독립 CLI가 아니다. frozen runtime의 실제 signature를 확인한 뒤 작은 export wrapper에서 호출한다.

```text
gfp40.postrun.load_candidate
  (cfg, checkpoint_directory, device='cuda', expected_source=...)

g20.data.build_dataset
  원 resolved config가 가리키는 dataset manifest로 RR/FR dataset 구성

gfp40.evaluation.infer
  g20.evaluation.infer를 재노출한 기존 native inference 함수

gfp40.evaluation.rr_metrics / FRMetrics
  기존 RR20 / FR20 evaluator
```

**`gfp40_start.sh`나 원 P40 campaign의 start/train/postrun controller를 새로 켜지 않는다.** 종료된 40h budget/clock을 다시 만들거나 기간 검사를 우회할 필요가 없는 inference-only wrapper로 작업한다. `deadline=None` 사용이 해당 함수 계약상 허용되는지도 실제 runtime에서 확인한다. 과거 `identity.json`이나 source 검증을 끄는 방식으로 실행하지 않는다.

### 불변 조건

| 항목 | 실행 규칙 |
|---|---|
| model | 지정 checkpoint의 전체 A + frontend + U, strict load |
| mode | `eval`, no-grad / inference mode, FP32 |
| 데이터 순서 | RR20/FR20 원 manifest 순서, shuffle 없음 |
| 입력 | 정상 native PAN/MS 및 기존 LP 자산, synthetic shift/PANMIX 없음 |
| 출력 좌표 | 원 MS 좌표계, A_ON |
| GT | forward 입력 금지, RR 검증·오차맵에만 사용 |
| 값 변환 | 기존 infer 함수의 DN 변환·clamp 정책 유지, 8-bit 양자화 금지 |
| RR 저장 | 전체 256×256; metric용 `20:-21` crop을 raw/PNG에 적용하지 않음 |
| FR 저장 | 전체 512×512, masking 없음 |

## 7. 가장 먼저 저장할 산출물

렌더러 설정을 아직 확보하지 못해도 아래 raw package는 완료한다.

```text
Ours_s4_GF2_raw_<timestamp>/
  README.md
  manifest.json
  scene_manifest.json
  metrics/
    original_selection.json
    re_evaluation.json
    metric_comparison.json
  provenance/
    checkpoint_identity.json
    config.resolved.yaml
    dataset_manifest.json
    source_identity.json
    checkpoint.sha256
  Ours_raw/GF2/
    reduced/pred/testimg0.npy ... testimg19.npy
    full/pred/testimg0.npy ... testimg19.npy
  references/GF2/                   # 해당 RR/FR 20장만, 허용된 원본 입력의 slice
    reduced/gt/...                  # renderer에서 RR error를 다시 계산할 수 있게
    reduced/pan/...
    reduced/ms/...
    full/pan/...
    full/ms/...
  reports/
    export_status.json
    raw_validation.json
    rendering_requirements.json
  checksums.sha256
```

위는 **새 export package의 요구 구조**이지 서버에 이미 존재한다고 확인한 디렉터리가 아니다.

- 예측은 NumPy float32, scene별 `(C,H,W)`로 저장한다. RR 20개 `(4,256,256)`, FR 20개 `(4,512,512)`다. 기존 infer의 축 순서를 확인하고 필요시 명시적으로 변환한다.
- 입력 reference는 저장 시 DN 단위·축·밴드 순서를 명시한다. 원본 전체 train dataset이나 무관한 파일은 포함하지 않는다. 재배포할 수 없는 데이터가 있으면 해당 reference는 제외하고 정확한 데이터 ID/SHA와 제외 이유를 기록한다.
- `scene_manifest.json`에 `split`, 0-based 순번, 원 scene ID 또는 **미확정 표시**, 원 H5/MAT 및 index/key, input slice hash, pred 경로/hash, 축/shape/dtype/band order/maxDN을 기록한다.
- `testimg10`을 문자열 정렬로 `testimg2` 앞에 놓았다는 이유로 장면 index를 바꾸지 않는다.
- 비교 자료의 scene 대응은 별도 검증 항목이다. 20장/크기가 같다는 이유로 CANConv와 같은 장면으로 인증하지 않는다.
- `checksums.sha256`에는 checksum 파일 자체를 제외한 전달 파일을 넣는다.
- 기본 결과 package에는 모델 weight나 optimizer resume state를 넣지 않는다. 사용자 요청은 정성 출력 회수이며, 학습 복구용 대용량 state나 credential은 전달 대상이 아니다.

## 8. 수치·파일 검증과 GPU 반납

추론된 배열을 먼저 저장한 뒤, **그 배열을** 원 evaluator에 넣어 원 선택점과 대조한다. 지표만 다시 계산한 뒤 raw 예측을 잃어버리지 않는다.

검증 항목은 장수/shape/dtype/finite/DN 범위, 전체 밴드, index 대응, 모델 SHA, 데이터 SHA, 정상 A_ON, RR crop 규칙, FR native reference다. raw 값의 단위와 기존 infer의 clip 정책을 명시한다.

`metric_comparison.json`에는 지표별 expected·actual·absolute difference를 모두 남긴다. 허용 오차는 결과를 보기 전에 같은 runtime/장치 재추론의 재현성 기준으로 정하고 근거를 적는다. 사전 근거가 없으면 임의의 epsilon으로 PASS를 만들지 않고 `PARITY_REVIEW_REQUIRED`로 둔다. 지표를 맞추려고 출력, 정렬, crop, checkpoint를 변경하지 않는다.

GPU가 필요한 작업이 끝나면 inference worker를 정상 종료하고 UUID 기준 GPU 점유 해제를 확인한다. 원 evaluator가 CPU-only 단계로 분리될 수 있으면 raw 저장과 GPU worker 종료 후 학습 복귀를 먼저 하고 CPU metric/압축을 나중에 수행할 수 있다. 이 경우 `metric_check_status=PENDING_CPU`를 별도로 기록하며 검증 완료로 보고하지 않는다.

**원 renderer 미확정은 학습 복귀를 막지 않는다.** 표와 같은 checkpoint의 raw 결과 확보가 이번 긴급 GPU 점유의 핵심이다.

## 9. 원 실험 자동 재개: 성공·실패 공통 cleanup

pause와 export는 독립적인 shell 명령을 수동으로 끊어 실행하는 데서 끝내지 말고, 실행 담당자는 다음 상태 흐름의 **단발 orchestration 작업**으로 묶는다.

```text
PRECHECK
  → SNAPSHOT_ORIGINAL_OWNER
  → REQUEST_SAFE_PAUSE
  → VERIFY_FULL_STATE_AND_GPU_RELEASE
  → INFER_AND_SAVE_RAW
  → VERIFY_OR_RECORD_EXPORT_FAILURE
  → EXIT_EXPORT_WORKER_AND_RELEASE_GPU
  → RESUME_ORIGINAL_OWNER
  → VERIFY_REAL_PROGRESS
  → FINISH_REPORT
```

`try/finally` 또는 동등한 cleanup에서 복귀를 수행하되 다음 조건을 모두 만족할 때만 실행한다.

- 이번 작업 전 원 서비스가 실제 실행 중이었다.
- 이번 작업이 요청한 pause가 정상적으로 확인되었다.
- export worker가 종료되어 원 GPU를 안전하게 돌려줄 수 있다.
- 원 실험 full-state/config/source/bindings가 검증 가능하다.
- 이번 pause 이후 사용자의 새 stop/취소 요청이 없고, stop marker 소유권이 유지된다.

추출에서 예외가 나거나 renderer/업로드가 실패해도 위 조건이 맞으면 원 실험을 복귀한다. 원 resume state 자체가 손상되었거나 소유권이 모호하면 fresh restart하지 않고 `RESUME_BLOCKED`로 보고한다.

### 9.1 Marker 처리

G23의 start는 `STOP_NOW_SAFE`와 `STOP_AFTER_RUN`을 자동으로 지우지 않는다. 이번 작업이 만든 **정확한 marker의 내용/SHA가 변하지 않았을 때만** 감사 디렉터리로 이동하여 lane에서 해제한다. 다른 marker나 기존 사용자 marker를 `rm -f STOP*`로 지우지 않는다. 실행 중 새로운 stop 요청이 들어오면 이를 존중한다.

### 9.2 원 Docker였던 경우

원 s4 container가 정상 pause 후 exited 상태이고, 기록한 image/content ID·command·mount·runtime이 유지되며 동일 lane의 다른 owner가 없다면 **기존 container ID를 그대로 재시작**하는 것이 우선이다.

```bash
# 소유권·marker·GPU 해제·resume bundle 확인을 모두 마친 뒤에만 실행.
docker start "$SENS_CID"
docker logs --tail 100 "$SENS_CID"
```

기존 container가 지워졌거나 mount/image가 바뀌었다면 위 명령을 흉내 내어 새 환경을 만들지 않는다. 기록해 둔 pinned image/runtime/binding으로 동일 서비스 launch를 복구해야 하며, 재개 가능성이 검증될 때까지 BLOCKED로 기록한다. 자동 image pull, 최신 tag로 변경, 원 container 삭제는 금지다.

### 9.3 원 호스트 프로세스였던 경우

원 supervisor가 종료되었고 동일 환경·GPU·working directory가 확인되면 기록한 실행 방식을 사용한다. G23 서비스 호출의 확인된 핵심은 다음과 같다.

```bash
"$PYTHON" "$SENS_ROOT/tools/g23sens_runner.py" start \
  --server s4 --in-place --foreground
```

이 명령은 foreground 서비스다. 실제로는 원래 사용하던 supervisor/session 방식으로 지속시킨다. 원래 `--no-upload`를 사용했다면 그대로 유지한다. `train --resume`를 수동 호출해 controller admission을 우회하지 않는다. controller가 유효한 `last`를 확인하고 같은 attempt에 `--resume`를 전달하도록 둔다.

### 9.4 실제 복귀 확인

PID만 생겼거나 start가 `START_REQUESTED`를 반환한 것으로 끝내지 않는다. startup preflight가 다시 실행될 수 있으므로 이후 실제 진행까지 확인한다.

학습 중에 멈췄다면:

- 동일 run_id, cycle, seed, case spec, attempt를 유지한다.
- 이전 `j` update의 state에서 model/U·A, optimizer moments, scheduler, criterion, RNG, sampler를 복구한다.
- 다음 학습이 `j+1`부터 이어지고 이후 training_status/log/새 last가 `j`보다 진행됨을 확인한다.
- learning rate/warmup, initialization, 시간 ledger를 초기화하지 않는다. 일시정지 시간·GF2 추론 시간을 sensitivity의 optimizer training time에 더하지 않는다.
- 재개용 state를 못 읽었다는 이유로 같은 case의 fresh attempt를 자동 허용하지 않는다. 기존 controller의 기술 재시도 경로로 빠졌으면 이번 interlude의 정상 복귀로 보고하지 않는다.

이미 TRAINED/EVALUATED 또는 case 경계였다면:

- 미완료 평가/업로드만 이어가거나 controller가 정확히 다음 case를 진행해야 한다.
- 끝난 50K를 다시 학습하지 않는다. cursor를 수동으로 되감거나 새 seed를 만들지 않는다.

오케스트레이터 자체가 비정상 종료되면 `transaction.json`으로 recovery한다. `finally`만으로 호스트 전원장애나 SIGKILL까지 처리된다고 가정하지 않는다. recovery는 기존 owner의 생존 여부를 먼저 확인하고 **중복 launch 없이** 남은 단계만 수행한다.

## 10. CANConv 형식 렌더링과 전달

기존 renderer/scene manifest/ROI가 확보되면 CPU 측에서 다음 120 PNG를 추가한다.

```text
Ours/GF2/full/output/testimg0.png ... testimg19.png
Ours/GF2/full/compare_output/testimg0.png ... testimg19.png
Ours/GF2/reduced/output/testimg0.png ... testimg19.png
Ours/GF2/reduced/compare_output/testimg0.png ... testimg19.png
Ours/GF2/reduced/errormap/testimg0.png ... testimg19.png
Ours/GF2/reduced/compare_errormap/testimg0.png ... testimg19.png
```

full은 512×512 RGB, reduced output/compare는 256×256 RGB, reduced errormap은 256×256 RGBA, compare_errormap은 256×256 RGB다. 원본 renderer 없이 RGB band order, contrast/stretch/gamma, ROI·inset·보간, error formula/color limits를 추정해서 공정한 비교 완료로 표시하지 않는다.

renderer가 없으면 `RAW_READY_RENDER_PENDING`으로 raw package를 먼저 전달한다. 0-byte PNG나 다른 연구의 PNG를 Ours 이름으로 넣지 않는다.

반환할 정보:

```text
원 실험 pause run/cycle/case/seed/attempt/update
정확한 GF2 run/step/SHA 및 RR20/FR20 저장 경로
raw count/shape/finite/metric comparison 결과
renderer·scene correspondence 검증 상태
원 실험 복귀 PID/container, resumed_from_update, 재개 후 관측 update
결과 ZIP 절대 경로, byte 크기, SHA256, 접근 가능한 전달 링크(있을 때만)
```

결과 ZIP 권장명: `Ours_s4_GF2_raw_<timestamp>.zip`; PNG까지 확정되면 별도 `Ours_s4_GF2_visual_<timestamp>.zip`.

원 서버 경로만 보고하고 다운로드 가능하다고 하지 않는다. 실제 ZIP을 대화 첨부로 전달하거나, 사용자에게 접근 권한이 있는 기존 Drive/스토리지에 올리고 **그 파일 링크**를 반환해야 한다. credential·서비스 계정 JSON·SSH 키·전체 환경변수·원 train state는 제외한다. 공개 공유 권한을 새로 열거나 임시 public HTTP server를 띄우지 않는다.

## 11. 최종 상태 예시

| export | 원 실험 | 보고 |
|---|---|---|
| raw 및 수치 검증 완료, renderer 없음 | 동일 run 정상 재개 | `RAW_READY_RENDER_PENDING / TRAINING_RESUMED` |
| PNG까지 검증 완료 | 정상 재개 | `VISUAL_READY / TRAINING_RESUMED` |
| GF2 asset/추론/metric 불일치 | 정상 재개 | `EXPORT_BLOCKED 또는 PARITY_REVIEW_REQUIRED / TRAINING_RESUMED` |
| 추출 완료 여부와 무관 | resume 손상/새 사용자 stop | `RESUME_BLOCKED` 사유 명시; 성공으로 종료하지 않음 |

## 12. 근거와 구현 경계

고정 모델/metric/파일 규약의 기준: 첨부 `ours_export_targets.json`, `QUALITATIVE_EXPORT_AUDIT_KR.md` 및 CANConv ZIP 감사. 이번 문서는 그 대상을 변경하지 않는다.

이번에 read-only로 확인한 저장소 snapshot: `6e746d43804c3a78ce781a7ce57adedb85b6431a`. 이는 **조회한 코드 snapshot**이지 현재 s4의 실행 commit을 인증한 것이 아니다.

- `g23sens/README.md`: safe-now, marker 유지, 무기한 cycle 및 exact resume 운영.
- `tools/g23sens_runner.py`: `stop --server s4 --safe-now`, `status`, `start --in-place`, marker 검사.
- `g23sens/controller.py`: full-state hash 검증, code75 pause, 동일 attempt resume, TRAINED 평가 재개.
- `g23sens/training.py`: 안전 update 경계, fullstate, scheduler/sampler/RNG 복구, 원 상태 보존.
- `tools/g23sens_docker_start.py`: 고정 image·runtime·GPU, server label, restart=no.
- `gfp40/postrun.py`: `load_candidate` 및 SHA/config/state 검증.
- `gfp40/evaluation.py`, `g20/evaluation.py`: native GF2 RR20/FR20, 기존 metric과 inference.

저장소: https://github.com/hojunking/PAN-Crafter-repro/tree/6e746d43804c3a78ce781a7ce57adedb85b6431a

**이미 구현된 기능:** G23 안전 pause·full-state resume와 기존 GF2 loader/evaluator.  
**이 지시서에 따라 서버에서 연결할 작업:** 일시정지 transaction의 자동 cleanup·복귀 검증, 지정 checkpoint의 raw export wrapper, 원 renderer 연결, 최종 결과 전달. 새 wrapper 파일이 이미 저장소에 존재하거나 서버에서 테스트되었다고 가정하지 않는다.
