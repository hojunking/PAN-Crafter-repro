# QG40 s2 개시 및 GF2 calibration 복구 — 2026-09-20

후속 확인(12:49 KST): s2는 실제 학습을 아직 시작하지 못했고 Teacher import 검증에서 BLOCKED_INTEGRITY 상태다. 자세한 원인과 조치는 [s2 reference 차단 진단](2026-09-20_qg40-s2-reference-blocker.md)을 참고한다. 아래 10:05 상태는 당시 기록이다.

## 승인과 s2 준비

사용자가 이전 턴의 s1→s2 QB 학습·검증 파일 2개(총 5,772,328,960 bytes) 전송 승인 요청에 “진행해라”로 승인했다. 기존 승인 차단은 해소되었으며 09:33 KST부터 data/qg40_exact_sources/QB/*.part 경로로 전송 중이다. 기존 s2 데이터는 보존한다.

s2 전용 Docker 환경 pancrafter-env:qg40-s2-cu128, CPU 161개 테스트, 기존 idle-service 정책 6개 테스트 및 CMP170HX 소규모 CUDA 순전파/역전파 검증은 이전 배포에서 통과했다. source identity는 s4와 허용되는 Git HEAD 차이 외 동일하다.

s4의 QB_TA.tar.gz(42,572,944 bytes)를 s2 incoming으로 전달하고 전체 SHA256을 검증했다.
SHA256: 8424026acbde8b1b73f424995a9f25df01df2cbb807810e1c3e205733aca6e94.
QB TA는 이번 캠페인의 immutable reference 1개이므로 완료된 패키지를 1회 전달했다. 추가 전송 대기 작업은 필요하지 않다. 실제 import 수치 검증은 s2 시작 후 수행된다.

## GF2 calibration 실패 원인

09:33 KST 상태 확인에서 s3의 GF2 Teacher는 exact50K를 완료했으나 q-cache/online AXIS16 parity가 매번 1.6689300537109375e-05 차이로 실패했다. reference_manifest가 발행되지 않아 s5가 WAIT_REFERENCE 상태였다. 반복 계산을 멈추기 위해 운영용 host recovery에 maintenance hold를 추가하고 현재 시도의 종료를 기다렸다.

동일 체크포인트, 데이터, 코드, 환경에서 읽기 전용 비교를 수행했다.

| 비교 | 최대 절대 오차 |
| --- | ---: |
| 기존 batch64 cache와 batch64 재계산 | 0 |
| 기존 batch64 cache와 batch16 검증 | 1.6689300537109375e-05 |
| batch16 그룹 계산과 batch16 검증 | 0 |
| batch16 검증 반복 | 0 |

GPU 배치 크기에 따른 수치 차이로 재현되었다. 허용오차는 atol=3e-6, rtol=2e-5를 유지했다.

## 복구

공개된 reference나 Student가 아직 없음을 확인했다. 미완성 q_cache.npz 및 calibration.json을 다음 위치로 보관했다:
.git/codex-qg40-deploy-20260920/s3/unpublished-GF2_TA-batch64-20260920/

기존 함수가 지원하는 batch_size=16 인자로 calibration을 재계산했다. seed1234 train3072, 전체 train×4-view q cache, Teacher exact50K checkpoint, 데이터, 코드 SHA 및 parity 허용오차를 유지했다. 체크포인트 재학습이나 parity 예외는 없었다.

- Teacher checkpoint SHA: 0fb973376f9f00947d295620d96062b621145607aee311e28c889caefb656dab
- Source content SHA: 0cf7bf2af1a81036a115b3b9a181c4cc0d3f292e7a7927383f25c5819dee9d8c
- 복구 시작: 09:38:15 KST / 완료: 09:39:28 KST
- 실제 reference parity 최대 오차: 0
- tau_R: 0.005695597268640995 → 0.005695626139640808 (차이 2.8870999813079834e-08)
- q_ref: 0.4086422026157379 → 0.4086490869522095 (차이 6.884336471557617e-06)

검증 후 reference를 발행하고 GF2_TA.tar.gz를 s5로 전달했다.
SHA256: 6dac45d59d0c43556e06637dd52358cce7bf09e6ea47b06c68c4f2e57d966782.
09:40 KST에 목적지 SHA 검증 및 s5 import 수치 검증이 통과했다. maintenance hold를 보관 해제하고 s3 큐를 재개했다.

09:42 KST 기준 s3의 Teacher 결과는 UPLOAD_VERIFIED이며 s3의 SS92001, s5의 SS92003 Student 학습이 시작되었다. s1/s4도 정상 학습 중이며 s2는 데이터 전송 중이다.

진단 원자료: gf2-batch-parity-diagnostic.json, gf2-calibration-recovery.json.
재현용 운영 스크립트: qg40_gf2_batch_diagnostic.py, qg40_gf2_calibration_recovery.py. numerical source 파일은 수정하지 않았다.

## 최종 확인 — 10:05 KST

s2의 두 기준 데이터 파일은 전송 및 전체 SHA256/source provenance 검증을 통과했고 config/qg40/QB_sensor.json에 바인딩되었다. 원본 파일은 보존했다. 서버 간 직접 VPN 전송에는 기존 SSH 키의 임시 agent를 사용했으며 종료 시 agent를 제거했다. 개인키 파일이나 서버 인증 설정은 복사·변경하지 않았다.

10:04:13 KST에 pan-qg40-s2-20260920 컨테이너를 시작했다. QG40 161개 테스트 및 유휴 서비스 정책 6개 테스트가 통과했고 등록이 완료되었다. 현재 WAIT_WV3_CASE_BOUNDARY로, 진행 중인 WV3 SS73205 case가 끝나면 QG40 preflight → QB Teacher import → Student SS82003을 자동 진행한다. s2의 실제 reference 수치 import와 GPU preflight는 아직 해당 순서를 기다리고 있다.

| 서버 | 현재 상태 | 진행 중 QG40 update | Sheet 검증 완료 건수 |
| --- | --- | ---: | ---: |
| s1 | RUNNING | 31,310 | 2 |
| s2 | WAIT_WV3_CASE_BOUNDARY | 미시작 | 0 |
| s3 | RUNNING | 18,180 | 1 |
| s4 | RUNNING | 37,370 | 3 |
| s5 | RUNNING | 18,180 | 0 |

완료된 QG40 6건 모두 각 공식 upload_receipt.json의 readback_verified=true를 확인했다. 미완료 학습의 Sheet 업로드를 완료로 표기하지 않았다. 모든 서버에서 공통 admission cutoff와 deadline을 유지했다.
