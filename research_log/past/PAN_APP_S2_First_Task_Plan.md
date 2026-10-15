# PAN-APP — s2 첫 작업 지시서

작성일: 2026-09-30
상태: 실행 계획. s2에 접속하거나 파일·프로세스·환경을 변경한 기록이 아님.
작업 식별자(신규 제안): PAN_APP_S2_M1

## 1. 이번 작업의 목표와 범위

s2에서 기존 QB C01 모델 한 개를 고정하고, 학습 시스템과 분리된 명령행 추론 경로가 원래 구현과 같은 출력을 내는지 확인한다.

이번 완료 상태는 `QB C01 / prepared PAN+MS+LPAN → HRMS 단독 추론 및 수치 동등성 확인`이다. 아직 raw PAN/MS 파일만으로 실행되는 완성 프로그램, 모든 센서를 지원하는 모델, 저사양 PC 인증, GUI 완성을 뜻하지 않는다.

- 계산·개발·기록은 s2에서만 수행한다. s1/s3/s4/s5에 접속하거나 작업을 배정하지 않는다.
- 먼저 QB를 쓰는 이유는 인계서상 s2의 기존 담당이 QB였기 때문이다. 현재 로컬 파일 존재는 다시 확인한다.
- 새 학습, fine-tuning, 통합 데이터셋 학습, 모델 확대는 이번 범위에 넣지 않는다.
- GUI, TensorRT/ONNX 변환, FP16/INT8, HPAN 입력 재매개변수화, 새로운 정합·LPAN 필터도 이번 범위에서 제외한다.
- 기존 연구의 소스·checkpoint·optimizer·로그·data·queue·uploader·논문 시트는 변경하지 않는다.
- 원 인계서의 디스크 정리 과제를 이 제품 개발 과제로 실행하지 않는다.

## 2. 대상 모델 규약

기준은 기존 catalog의 C01이며, FULL에서 모듈을 임의로 끈 모델이 아니다.

| 항목 | 이번 고정값 |
|---|---|
| 센서 | QB, 4 MS bands |
| Component | C01 |
| Backbone | W104, D[1,2,2] |
| Attention / mode modulation | OFF / OFF |
| Aligner | IDENTITY; trainable aligner와 warp 없음 |
| Teacher / KD / q / e / probe | 추론 의존성 없음 |
| Backbone 입력 | PAN + upsampled LPAN + HPAN + upsampled MS, 총 7채널 |
| 모델 출력 | 4밴드 HRMS; upsampled MS + residual을 정확히 한 번 적용 |
| 해상도 비율 | PAN:MS = 4:1 (가로·세로 각각) |
| 기준 실행 | FP32, batch 1, 원 구현의 수치 설정 유지 |

기존 검토에서 QB C01 파라미터 수는 2,093,316개로 기록되었다. 이는 대조값이며 실제 s2에서 선택한 모델을 생성·로드하여 다시 센다. 불일치는 원인을 보고하고 임의로 무시하지 않는다.

현재 연결 저장소의 `ablr2/model.py`에는 backbone 호출까지 `autocast(enabled=False)` 안에 있다. 이 첫 단계에서는 이를 바꾸지 않는다. s2의 실제 checkpoint source snapshot과 현재 원격 코드가 같은 버전인지 별도로 확인한다.

## 3. Step 1 — s2 상태와 필수 자산만 읽기 전용으로 확인

다음을 실제 값으로 기록한다.

1. host 식별자와 host/container 구분, 컨테이너 사용 시 host 경로와 container 경로 대응.
2. OS, GPU 모델·UUID·VRAM·드라이버, RAM, 작업 디스크 여유 공간.
3. 실행 중인 작업과 해당 GPU 사용 여부. s2의 기존 QB ABLR2/ABLR2X controller·queue가 있는지도 확인.
4. 기존 PAN 저장소의 실제 경로, 실행 Python/conda/container 환경, 관련 패키지 버전.
5. 로컬 QB C01 checkpoint·config·source snapshot·selected-step metadata·평가 기록의 위치.
6. QB RR/FR 또는 validation 데이터와 대응 LPAN cache의 실제 위치·가독성.

확인 범위는 관련 저장소·run 디렉터리·필수 파일로 제한한다. 전체 디스크 재귀 스캔이나 전체 데이터셋 복사는 하지 않는다.

다음 행동은 하지 않는다: 학습 프로세스 kill, 기존 환경의 pip upgrade, 사용 중 저장소 checkout/pull, 디스크 정리, 다른 서버의 자동 탐색·복사.

기존 작업이 자원을 사용하고 있으면 CPU로 가능한 준비만 수행하고 GPU 측정은 보류한다. 전용 사용 전환이 필요하면 현재 작업과 안전한 종료/중단 지점을 보고한다. 전환 권한이 불명확하면 임의 중단하지 않는다.

산출물: `reports/preflight.json`
미확인 값은 `null`과 사유로 남긴다. 이전 대화의 사양으로 채우지 않는다.

## 4. Step 2 — checkpoint 한 개와 독립 작업공간 고정

### checkpoint 선택

- 완료 기록과 source/config 연결이 확인된 로컬 QB C01 run 한 개를 재현 기준으로 선택한다.
- 해당 run의 기존 `RR_VAL_SELECTED` 선택점을 사용한다. 여러 run의 test HQNR를 새로 비교해 최고 모델을 고르지 않는다.
- 초기 run 선택 규칙과 이유를 manifest에 기록한다. 동일 용도로 여러 개가 있으면 run_id 사전순 등 성능과 무관한 규칙을 사용할 수 있다.
- 실제 파일, selected step, checkpoint 파일 SHA256, source snapshot/commit 및 관련 코드 hash를 연결한다.
- 파일 SHA와 모델 tensor/state hash는 별도 필드로 기록한다.
- C01을 찾을 수 없으면 `BLOCKED_MISSING_C01`로 보고한다. FULL/C03 대체나 자동 재학습으로 진행하지 않는다.

### 독립 작업공간

검증된 쓰기 가능 위치 아래 `pan-app-s2/`와 같은 별도 폴더를 만든다. 이름은 제안이며 현재 존재하는 서버 경로가 아니다.

```text
pan-app-s2/
  src/                 # 신규 inference wrapper
  configs/             # QB sensor profile와 모델 manifest
  tests/               # parity test
  fixtures/            # 소수의 고정 입력 및 reference 출력
  reports/             # preflight / parity / profile / 요약
  artifacts/           # 검증한 가중치 사본 또는 불변 자산 연결 정보
```

원래 실행환경은 변경하지 않는다. 필요하면 검증된 환경의 별도 복제본을 사용하되 먼저 디스크 여유를 확인한다. 원본 데이터는 읽기 전용으로 참조하고, fixture에 필요한 소수 샘플만 복사한다.

산출물: `configs/model_manifest.json`, `configs/qb_profile.json`

manifest에는 run_id, case_id, selected_step, selector, checkpoint_path, checkpoint_file_sha256, source_commit/source_snapshot, source_file_hashes, model_args, num_params, band_order, normalization, interpolation, LPAN provenance, source/asset license 식별을 기록한다.

## 5. Step 3 — 원 구현으로 기준 입력·출력 확보

GPU 사용이 가능한 상태에서만 수행한다.

- 기존 dataset loader에서 고정된 RR 샘플 하나와 FR 샘플 하나를 읽는다. 샘플 인덱스와 원파일을 기록한다.
- 논문상의 RR PAN은 256×256, FR PAN은 512×512이며 MS는 각각 64×64와 128×128이다. 실제 loader의 tensor shape로 재확인한다.
- `ms`(LR 관측)와 `lms`(upsampled 관측)를 이름만 보고 바꿔 쓰지 않는다.
- 첫 검증에는 기존 cache의 LPAN을 그대로 사용한다. 새로운 저역통과 필터를 만들지 않는다.
- PAN/MS/LPAN의 shape, dtype, 범위, 밴드 순서, normalization, 해당 fixture hash를 저장한다.
- 정확한 source snapshot의 원 구현과 선택한 checkpoint로 FP32 기준 출력을 만든다.
- clamp 전 normalized HRMS와 원 구현의 후처리 결과를 구분해 저장한다.
- 이 샘플은 소프트웨어 회귀 검증용이다. 향후 모델 선택·학습에는 사용하지 않는다.

기준 출력은 신규 wrapper의 출력으로 만들지 않는다. 독립적인 원 구현 출력이어야 한다.

산출물 예: `fixtures/qb_rr_000.npz`, `fixtures/qb_fr_000.npz`, `fixtures/manifest.json`

## 6. Step 4 — 학습 코드 없는 최소 추론 wrapper 작성

아래 계약의 함수/클래스를 구현한다.

`infer_prepared(pan, ms, lpan) -> hrms`

이 이름과 파일 이름은 앞으로 구현할 인터페이스 제안이다. 이미 존재하는 CLI 명령이라고 보고하지 않는다.

- 순수 모델 생성과 가중치 로드에 필요한 코드만 포함한다.
- checkpoint key mapping이 필요하면 대응을 명시하고 누락/예상 외 key 없이 로드한다.
- scheduler, optimizer, Teacher inference, cue calibration, training controller, uploader를 초기화하거나 호출하지 않는다.
- 모델은 한 번 로드하고 반복 호출한다.
- 기본값은 FP32와 batch 1이다. `model.eval()`과 `torch.inference_mode()`를 적용하고 원 구현과 비교한다.
- PAN/MS/LPAN shape, 밴드 수, 4배 관계, finite 값, 입력 스케일을 검사한다.
- MS residual 더하기, clamp, DN 역변환을 원 구현의 순서와 같은 위치에서 각각 한 번 적용한다.
- 제품 반환값은 필요한 HRMS만으로 제한한다. 진단 tensor를 장면 전체에 누적하지 않는다.
- GPU가 없을 때 CPU 경로가 실제 호출 가능하도록 장치 선택을 분리한다.

정규화식·band order·LPAN 생성 규칙이 manifest에서 확인되지 않으면 추정하지 말고 blocker로 남긴다.

산출물: 추론 wrapper와 실행 예제, 최소 의존성 목록.

## 7. Step 5 — 수치 동등성 검증

동일한 fixture와 가중치로 원 구현과 신규 wrapper를 비교한다. 같은 s2 환경·device·FP32 설정에서 먼저 검증하고 CPU smoke test는 별도로 기록한다.

- 출력 shape, 밴드 수, finite 여부.
- clamp 전 normalized 출력의 MAE/RMSE/max absolute error.
- 역변환·clamp 이후 값의 동등성.
- residual 중복 합산과 입력 밴드 순서 오류가 없는지.
- 잘못된 밴드 수와 비정상 PAN/MS 크기를 넣으면 명확히 실패하는지.

첫 수치 비교 기준 제안: `torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)`.
이 값은 normalized FP32 구현 동등성 검증용 제안이며 원격탐사 품질의 보편 기준이 아니다. 실패하면 중간 입력·업샘플링·LPAN/HPAN·residual·후처리를 순서대로 비교한다. 통과를 위해 임의로 허용오차를 넓히지 않는다.

통과 후 고정된 소수 추가 샘플로 회귀 범위를 확장한다. CPU/GPU 차이는 동종 device wrapper 동등성 결과와 분리하고 실제 오차를 보고한다.

산출물: `reports/parity.json`, `tests/test_inference_parity.py`.

## 8. Step 6 — s2 기준선만 측정

Step 5를 통과하고 GPU가 사용 가능한 경우에만 수행한다.

- QB, FP32, batch 1, PAN 256×256 / MS 64×64 / LPAN 64×64.
- 이어서 PAN 512×512 / MS 128×128 / LPAN 128×128.
- 워밍업 10회, 측정 50회를 초기 설정으로 제안한다. 실행 시간과 자원 상황에 따라 횟수를 기록하고 조정한다.
- GPU 동기화 또는 CUDA event를 사용하여 timing을 수행한다.
- cold model load, GPU-resident forward, transfer 포함 호출을 구분한다.
- median/p95 latency, peak allocated/reserved VRAM, 프로세스 GPU 점유의 관측값과 측정 방법을 남긴다.
- LPAN cache 생성, GeoTIFF 읽기/쓰기, GUI 비용은 포함되지 않았음을 명시한다.
- 이전 s2 7.37ms 기록과의 차이를 숨기지 않고 environment/scope 차이부터 확인한다.
- s2 메모리를 제한하는 시험을 하더라도 이를 RTX 3070/4GB 노트북의 실제 인증으로 표현하지 않는다.

산출물: `reports/profile_s2_fp32.json`.

## 9. 첫 보고와 중단 지점

`reports/M1_REPORT.md`에 다음을 남기고, 신규 학습·GUI·모델 변환은 시작하지 않은 상태로 첫 보고를 끝낸다.

| 항목 | 실제 결과 |
|---|---|
| s2 환경 / GPU / 독립 workspace | 확인값 또는 미확인 사유 |
| 기존 작업 처리 | 그대로 유지 / 종료됨을 확인 / 자원 대기 |
| C01 run / step / selector / 파일 SHA | 확인값 |
| model/source/전처리 일치 | PASS / FAIL / BLOCKED |
| 원 구현 vs wrapper | MAE / RMSE / max abs / PASS 또는 FAIL |
| CPU smoke test | PASS / FAIL / NOT_RUN |
| 256·512 기준 profile | 실측 / NOT_RUN 및 사유 |
| raw PAN/MS만으로 입력 가능한가 | 아직 아님: cached LPAN fixture 경로 검증 단계 |
| 다음 blocker | 필요한 정확한 파일/설정/자원 |

`M1_READY`는 checkpoint와 source 확인, fixture 기준 출력, 단독 wrapper, parity 통과 및 s2 GPU 기준선 측정이 완료된 경우에만 사용한다. CPU-only 완료는 별도로 `M1_CPU_ONLY`로 표시한다. GPU 점유 중이면 `WAITING_FOR_GPU`, 필수 자산 부족이면 `BLOCKED`와 구체적인 이유를 남긴다.

## 10. 다음 단계 — 이번에는 실행하지 않음

1. 기존 cache와 동일한 LPAN을 raw PAN에서 생성하는 경로 검증.
2. PAN/MS GeoTIFF 입력, 메타데이터 검증과 HRMS 저장.
3. 타일/halo/좌표 위상/NoData/경계 연속성 검증.
4. 정확도 기준을 유지하는 backend·정밀도 최적화와 실제 저사양 장치 검증.
5. GUI 및 배포 패키지. 서버에서 개발한 것과 대상 운영체제에서 인증한 것을 구분.
6. 사내 데이터 학습 및 QB+GF2 공동 학습. 그다음 필요할 때 4/8밴드 통합 학습.

## 근거와 한계

- 사용자 지시: 본 task는 s2에서만 진행. 위 작업은 사내 제품 개발용이며 기존 논문 실험을 재개·변경하는 지시가 아니다.
- `PAN_Next_Session_Handoff_2026-09-29.md`, §§4.1–4.3: s2의 과거 QB 담당, C01 정의, W104/D[1,2,2], 4/8밴드, DN 규약, component selector. 현재 PID/파일 존재는 문서로 인증되지 않는다.
- `59391_PANDA_PAN_Alignment_Awar.pdf`, §4.1, p.7: 센서별 학습과 RR/FR PAN 크기.
- `hojunking/PAN-Crafter-repro:ablr2/model.py`, 이번 조회 blob SHA `590fda6306c3bc1b1b8d71dc47215b112dab062c`: IDENTITY bypass, 4배 shape 검사, FP32 구간, PLH 입력, base+residual. 이 Git blob SHA는 checkpoint 파일 SHA256이 아니다.
- `C01_hardware_profile_review.md`: 이전 세션의 QB C01 parameter/profile 검토값. 신규 s2 실측이 아니다.
- PyTorch 공식 문서: Autograd mechanics / torch.testing.assert_close / CUDA semantics. 참고 위치: `docs.pytorch.org/docs/stable/notes/autograd`, `docs.pytorch.org/docs/stable/testing.html`, `docs.pytorch.org/docs/stable/notes/cuda.html`.
