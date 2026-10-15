# G20 s1 — 승인된 R0 교차 GPU 수치 차이 예외와 복구

후속: 사용자가 환경을 그대로 유지하고 진행하도록 승인했다. 아래는 첫 복구 시도의
기록이며, 이후 조치는 `2026-09-21_g20-s1-existing-env-launch.md`에 기록한다.

사용자 승인: **“승인한다”** (2026-09-21 KST). 앞선
`2026-09-21_g20-s1-reference-parity-diagnostic.md`의 미복구 상태에 대한 후속 작업이다.
현재 단계: **운영 Torch 2.4 환경의 공식 지표 재현 검사 실패로 복구 중단. G20 학습 미시작.**
GPU 원소별 수치 차이 예외 승인을 공식 지표 허용치 완화나 runtime 변경 승인으로 확대하지 않았다.

## 변경 범위

- 이번 캠페인의 **s1 / 고정 R0 / RTX 4090**만 허용한다. 다른 reference·서버에는 기존 strict 경로를 유지한다.
- 기존 R0 checkpoint, q-cache, 원본 probe, τR, qref, 데이터/LP, seed·loss·optimizer·평가 방식은 바꾸지 않는다.
- 기존 `registration.json`은 그대로 두고, 원본 SHA와 변경 소스 전체를 묶는 별도 immutable source amendment를 발행한다.
- 수정 파일은 `g20/{controller,preflight,references,upload}.py`, 추가 수치 검증 모듈은
  `g20/{gpu_exception,migration}.py`다. QG40/FH12/model/학습/evaluator 수치 코드는 바꾸지 않았다.
- 고정 실행 worktree의 Git base는 `8a63d10604901609facfd22b2b195724ea0122f6`이며,
  수정 실행본 content SHA256은 `4c0c2a1eb034ffb1b8574276f89625b7d934700c9adaba081524d617b7a53d28`이다.
- 개발 checkout의 기존 미커밋 문서·그림·watchdog 변경은 이 배포에 포함하지 않았다.

## 판정과 보존

원래 비교의 `atol=3e-6, rtol=2e-5`를 바꾸지 않는다. 실패한 원소 수·최대 차이와
원본/현재/반복 측정 배열을 모두 보존하며, 예외는 **ACCEPTED_WITH_PARITY_EXCEPTION**으로
기록한다. 역사적 GPU 재현을 PASS로 바꾸지 않는다.

승인된 작은 변동에도 상한을 둔다: q의 최대 절대 차이 `5e-5`, 모델 출력 `2e-4`,
delta `5e-5` HR pixel. 출력 상한은 GF2 정규화 기준 약 **0.1023 DN**이다.
이는 기존 실패 수치에 근거한 이번 운영 예외의 상한이지, GPU 오차의 보편적 보장이나
통계적 유의성 기준이 아니다. 현재 측정과 반복 측정 모두 상한을 만족해야 한다.

추가 필수 검사:

1. 원본 아티팩트·입력 동일성 검사를 그대로 통과한다.
2. 같은 운영 GPU에서 q(고정 16 ID × 4 view)와 원본 probe를 두 번 계산하여
   기존 strict 허용치로 일치해야 한다. q-cache는 전체 train **19,809 × 4**이며
   calibration의 3,072 subset과 혼동하지 않는다.
3. TA/S92001 공식 RR20·FR20 재평가에서 기존 `ΔHQNR≤1e-5, ΔERGAS≤1e-4`를 만족해야 한다.
4. import 이후 `load_reference()`를 실제 CUDA로 다시 호출하여 Student 소비 경로도 검사한다.
5. R0 사용 결과의 기존 Sheet Notes에 예외와 승인·측정 receipt SHA를 기록한다.

원본 registration/preflight/status 및 수정 전 소스는 `work_dir/_g20/s1/source_amendment/`에
보존한다. 이후 소스·runtime 변경은 새 identity 불일치로 다시 차단된다.

## 검증·실행 증거

- CPU G20 전체 테스트: **181개 통과**.
- 운영 GPU preflight: **64개 테스트 + 실제 CUDA forward 통과**.
- 승인: `work_dir/_g20/s1/parity_exception_authorization.json`.
- source amendment: `work_dir/_g20/s1/source_amendment/receipt.json`.
- 단계별 실제 결과: `work_dir/_g20/s1/source_amendment/recovery_result.json`.
- 공식 재평가: `work_dir/_g20/s1/diagnostics/r0_evaluator_parity.json` — **BLOCKED_PARITY**.
- 공식 재평가 seal, `gpu_exception_measurements.npz`, `reference_manifest.json`은
  **미발행**이다. 필수 지표 검사를 통과하지 않아 예외 측정/bridge 발행 전 차단됐다.

## 운영 환경 추가 차단 사유

앞선 진단의 지표 PASS는 Torch **2.8.0+cu128 / cuDNN 91002**에서 얻었다.
이번 원래 운영 환경인 Torch **2.4.0 / CUDA 11.8 / cuDNN 90100**에서 새로 측정한 결과는 다음과 같다.
입력 데이터와 원본 checkpoint 동일성은 통과했다.

| 원본 자산 | ΔHQNR (한도 1e-5) | ΔERGAS (한도 1e-4) | 판정 |
| --- | ---: | ---: | --- |
| TA | 1.3029850773449425e-5 | 7.288530774629542e-5 | HQNR 한도 초과 |
| S92001 | 4.24246957653196e-6 | 4.9103770145886294e-5 | 통과 |

복구 명령은 preflight 성공 후 parity 결과가 실패 상태였고, import가 유효한 PASS seal 부재를
검출하여 exit 2로 종료했다. parity CLI의 exit 0 자체는 검증 성공의 증거가 아니다.
`status.json`은 기존 **BLOCKED_INTEGRITY**, `runs={}`, `blocks={}`를 유지한다.
원본 registration·시간창·sensor spec·dataset manifest·R0 archive의 SHA256은
`source_amendment/protected_before.json`과 복구 후 모두 동일함을 재확인했다.

권고: 지표 허용치를 유지하고 **R0 생성 환경과 같은 Torch 2.8.0 / CUDA 12.8로 s1 runtime을
명시적으로 전환**한 다음, 신규 identity와 승인 증거를 기록하고 모든 검사를 다시 수행한다.
이는 이번 GPU 수치 예외와 별도 환경 변경이므로 사용자 추가 승인 전에는 진행하지 않는다.
이전 Torch 2.8 진단 PASS를 새로운 운영 source identity의 PASS seal로 재사용하지 않는다.

공통 시간창은 바꾸지 않았다. 신규 블록 마감 **9월 21일 14:05:13 KST**,
학습 마감 **16:05:13**, 전체 마감 **18:05:13**이다.
