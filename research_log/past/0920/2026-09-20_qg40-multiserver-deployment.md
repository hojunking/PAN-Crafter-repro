# QG40 다중 서버 배포 기록 — 2026-09-20

후속 업데이트(10:05 KST): s2 데이터 전송 승인·검증·실행기 등록이 완료되었다. s3 GF2 calibration 수치 오류도 복구했고 s3/s5 Student가 학습 중이다. 현재 상태와 복구 증거는 2026-09-20_qg40-s2-start-gf2-recovery.md를 따른다. 아래는 06:05 KST의 최초 배포 기록이다.

- 원격 main: `a19dfeb4733480b6132890a9ad4a9fff20c3bdd4`
- 캠페인: `PANDA_QG40_20260920_v1`
- 레지스트리: `QG40_REGISTRY_20260920_v2_PREPARATION`
- 수치 구현: `QG40_SYNC_FREQ_C4_v1`
- 공통 시작: 2026-09-20 05:12:28 KST
- 신규 학습 진입 마감: 2026-09-21 17:12:28 KST
- 전체 마감: 2026-09-21 21:12:28 KST

## 배포와 검증

사용자는 main에 push한 QG40 코드를 반영하고 tools/qg40_start.sh 실행을 요청했다. README/IMPLEMENTATION의 기존 WV3 checkpoint 검증 조건에 따라 git fetch 후 QG40 147개 파일을 archive로 반영했다. 기존 HEAD와 WV3 수치 구현을 유지했다. s1/s3/s4/s5 HEAD는 cccedee, s2 HEAD는 기존 0ec1132이다.

모든 서버에서 QG40 CPU 테스트 161개가 통과했다. s2의 새 환경은 CMP 170HX에서 소규모 CUDA 순전파·역전파도 통과했다. 테스트 중 출력되는 P1 timeout은 시간 제한 동작을 검증하는 의도적인 주입 사례이다. s2의 기존 유휴 서비스 정책 테스트 6개도 통과했다.

QG40 source content SHA256:
`0cf7bf2af1a81036a115b3b9a181c4cc0d3f292e7a7927383f25c5819dee9d8c`

s2에는 기존 WV3 환경을 보존하면서 별도 이미지 `pancrafter-env:qg40-s2-cu128`를 만들었다. Torch 2.8.0+cu128, CUDA 12.8, cuDNN 91002로 s4와 맞췄다. 비교된 source_identity 차이는 허용되는 git_release뿐이다. 기존 s2 GPU variance 예외를 QG40에 적용하지 않았다.

s2의 unhashed 운영 파일 tools/fh12_runner.py에 QG40 전용 환경 변수 QG40_USE_APPROVED_S2_IDLE_SERVICES=1일 때만 기존 PID/start time/command hash 기반 유휴 서비스 정책을 적용했다. 원본 백업은 .git/codex-qg40-deploy-20260920/s2/fh12_runner.before-qg40-idle-policy.py이다.

## 데이터 바인딩

원본 파일을 보존하고 추가 복사본을 data/qg40_exact_sources에 저장한다. 파일 전체 SHA256 검증 후 .part를 최종 이름으로 변경하고 config/qg40/<sensor>_sensor.json에 명시적인 경로를 기록한다. 원본 source catalog, numerical compatibility, parity 기준을 완화하지 않는다.

- s1: QB 및 GF2 기준 파일이 catalog와 일치.
- s3/s5: GF2 FR 파일 불일치를 s1 기준 복사본(220,207,104 bytes)으로 해결. SHA e52d151f262a74f96e59f03124f86a27d80841073ce60376856ae679020d00dc.
- s4: QB train/val msfix 및 FR 기준 파일 3개 전송 완료. 전체 SHA 검증과 명시적 바인딩을 마친 뒤 06:04:39 KST에 QG40 컨테이너를 기동했다.
- s2: QB train/val msfix가 불일치. 두 파일 합계 5,772,328,960 bytes의 s1→s2 전송은 자동 승인 검토가 거부하여 별도 사용자 승인 대기. 이 전송 및 간접 대체 전송은 수행하지 않았다.

## 실행 상태 (06:05 KST 점검)

| 서버 | QG40 상태 | 실행 계획 |
| --- | --- | --- |
| s1 | WAIT_WV3_CASE_BOUNDARY | 진행 중 WV3 case 후 QB TB Teacher |
| s2 | 등록 전 데이터 전송 승인 대기 | s4 QB TA reference import 후 학생 |
| s3 | WAIT_WV3_CASE_BOUNDARY | 진행 중 WV3 case 후 GF2 TA Teacher |
| s4 | WAIT_WV3_CASE_BOUNDARY | 기준 데이터 검증/등록 완료, 진행 중 WV3 case 후 QB TA Teacher |
| s5 | WAIT_REFERENCE | 실제 CUDA preflight 26개 method checks 통과, s3 GF2 TA 대기 |

s1은 native 실행기를 유지한다. s2–s5는 pan-qg40-sN-20260920 전용 Docker 컨테이너를 준비했다. s3/s4/s5 host recovery는 기존 crontab을 보존하며 별도 tag로 추가했다. s4 recovery는 모든 파일이 정확한 크기로 도착한 다음 전체 SHA 검증/바인딩을 거쳐 컨테이너를 시작한다. BLOCKED_INTEGRITY 상태는 자동 재학습하지 않는다.

s3→s5 GF2_TA reference 전송 worker를 설치했다. 공개 완료된 export archive만 전달하며 목적지에서 SHA를 검증한 후 atomic rename한다. 아직 Teacher가 완성되지 않아 WAIT_EXPORT 상태이다. s4→s2 QB_TA 자동 전달은 s2 데이터 승인 이후 연결해야 한다.

## 기존 오류 및 한계

s5의 이전 WV3 case FH20R1_S5_S_PLH_W112_D122_WV3_TF5_TS71002_SS72002_BASE_FRESH50_v1은 36,218 updates에서 CUDA launch timeout으로 중단되어 있었다. 기존 checkpoint/history를 보존했고 fresh restart하지 않았다. 새 QG40 CUDA preflight는 통과했지만 장시간 GPU 안정성 회복을 증명한 것은 아니다.

신규 QG40 50K 완료 결과는 아직 없으므로 신규 Sheet 업로드 완료를 주장하지 않는다. 배포된 실행기는 공식 평가 완료 후 즉시 Sheet 업로드 및 readback 검증을 수행하며, 실패한 업로드를 재시도한다.

상세 delivery, runtime fingerprint, 상태 스냅샷 및 데이터 SHA는 동명의 QG40_deployment_20260920.json에 기록했다. 서버별 원본 배포 증거는 .git/codex-qg40-deploy-20260920/sN에 보존한다.
