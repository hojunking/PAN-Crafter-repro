# G20 s1 — 기존 Torch 2.4 환경 유지 및 승인 예외 적용

사용자 지시(2026-09-21 KST):
> 아니다 그냥 진행해. 환경을 맞추지 않아도 괜찮으니 실험 진행해

이는 `2026-09-21_g20-s1-approved-gpu-parity-recovery.md`의 추가 차단에 대한
후속 승인이다. Torch 2.8 환경 전환은 하지 않는다.

## 변경과 보존

- 운영 환경: 기존 Torch **2.4.0 / CUDA 11.8 / cuDNN 90100 / RTX 4090** 유지.
- 원본 R0 checkpoint·q-cache·calibration·probe·데이터 및 LP 원본 유지.
- 모델·loss·optimizer·seed·학습량·공식 RR/FR 평가 코드 유지.
- 원래 strict 판정과 실패 측정값은 그대로 보존한다. 예외를 PASS라고 기록하지 않는다.
- 기존 registration과 v1 승인/소스 변경 기록은 수정하지 않는다. 별도 v2 승인과
  변경 기록이 이전 파일의 SHA256을 연결하며, 신규 소스 전체도 고정한다.
- 다른 서버와 R1~R4의 strict 경로에는 이 예외를 적용하지 않는다.
- 사용자가 수정한 문서·그림·watchdog는 이번 운영 배포에 포함하지 않았다.

## 실제 Torch 2.4 차이와 제한된 수용 범위

| 항목 | 실제 관측 최대 차이 | 이번 s1/R0 수용 상한 |
| --- | ---: | ---: |
| q | 2.0056962966918945e-5 | 5e-5 |
| probe output (정규화값) | 1.1372566223144531e-4 | 2e-4 |
| probe delta (HR pixel) | 1.1584162712097168e-4 | 2e-4 |
| TA 공식 HQNR | 1.3029850773449425e-5 | 2e-5 |
| S92001 공식 HQNR | 4.24246957653196e-6 | 2e-5 |
| TA 공식 ERGAS | 7.288530774629542e-5 | 1e-4 |
| S92001 공식 ERGAS | 4.9103770145886294e-5 | 1e-4 |

이는 이 고정 reference와 관측된 실행 환경 차이에 대한 운영상 승인이지, 모든 GPU의
동등성이나 통계적 무차이를 주장하는 기준이 아니다. 원래 지표 허용치
`ΔHQNR≤1e-5, ΔERGAS≤1e-4`는 수정하지 않고, 별도
`ACCEPTED_WITH_METRIC_EXCEPTION` 기록을 발행한다.

동일 GPU 반복 및 Student의 reference 재로딩 시에는 기존 strict
`atol=3e-6, rtol=2e-5` 검사를 유지한다. 원소별 최대 차이·실패 개수·원본/반복 배열을
보존하고, R0 소비 결과의 Sheet Notes에는 tensor/metric 예외와 receipt SHA를 남긴다.

## 검증 및 운영 증거

- CPU G20 전체 테스트 **208개 통과** (24.901초), `git diff --check` 통과.
- 별도 승인: `work_dir/_g20/s1/metric_exception_authorization.json`,
  `parity_exception_authorization_v2.json`.
- 변경 이력: `work_dir/_g20/s1/source_amendment_v2/receipt.json`.
- 이전 시도 보존: 같은 폴더의 `prior_attempt/`, `prior_sources/`, `archive/`.
- 최종 복구 검증: 같은 폴더의 `recovery_result.json` 및 단계별 로그.
- 실험 상태: `work_dir/_g20/s1/status.json`과 개별 run의 `g20_train.log`.

운영 재검증 완료: preflight **64개 테스트 및 실제 CUDA forward** 통과,
공식 지표 재측정/승인 예외 발행, R0 import 및 실제 CUDA `load_reference()` 성공.
fresh local q/output/delta는 봉인한 로컬 값과 **최대 차이 0.0**으로 strict 검사를 통과했다.
원본 registration·clock·sensor/data manifest·R0 archive의 SHA256은 이전과 모두 동일하다.

실행 소스 SHA256: `56e80b5cd0d286def8293715747831cb4abd4b7ae9ebd02a01fddb1a8a543e69`.
기존 `pan-g20-s1-20260920` 컨테이너를 **2026-09-21 03:19:24 KST**에 재시작했다.
실제 학습 시작: **2026-09-21 03:20:12 KST**. `S1_REFERENCE_PAIR` 5-run 블록이
기존 예약 규칙으로 ADMITTED(8시간)가 되었으며, 첫 Teacher를 실행 중이다.

- run: `G20_GF2_T_S1_TS91002_C100_P0_W112_D123_FRESH50_v1`
- 구성: GF2 / P0 / W112·D123 / Teacher seed 91002 / C100 / 50,000 updates.
- 학습 로그에서 **1,300 / 50,000 updates** 이상 진행 확인.
- 첫 정기 평가 **step 1,010** 완료, 이후 학습 재진입 확인.
- raw HQNR **0.90331437**, SCC **0.98357021**, ERGAS **0.89085230** 정상 출력.
  이는 초기 학습 상태의 정상 동작 확인이며, 최종 성능 판정이 아니다.
- 공식 평가 파일: 위 run의 `official/candidate_1010.json`, `official/raw_grid.json`.
- GPU 실측: RTX 4090 이용률 **89%**, 사용 메모리 **3,981 MiB** (학습 중 한 시점).
- 기존 host recovery를 재사용한다. 아직 완료되지 않은 결과의 Sheet 업로드 성공을
  주장하지 않으며, 기존 평가 완료 후 업로드 경로를 유지한다.

공통 시간창은 유지한다. 신규 블록 마감 **9월 21일 14:05:13 KST**,
학습 마감 **16:05:13**, 전체 마감 **18:05:13**.
