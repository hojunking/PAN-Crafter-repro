# s2 QG40 미시작 진단 및 인계 — 2026-09-20 12:49 KST

후속 복구: 14:12:32 KST에 실제 학습을 시작했다. [실험 재개 기록](2026-09-20_qg40-s2-experiment-started.md)을 참고한다. 아래는 해결 전 진단 기록이다.

## 현재 상태

s2(honbul)는 QG40 Student를 아직 시작하지 못했다. 12:49 KST 실측 상태는 BLOCKED_INTEGRITY, runs={}이며 QG40 학습 프로세스가 없다. pan-qg40-s2-20260920은 exit 2로 종료되어 있다. GPU 사용률은 0%, 총 65,536 MiB 중 701 MiB가 사용 중이다. 기존 ComfyUI 등 허용된 유휴 서비스 2개는 유지했다.

이 문서는 10:05 KST의 WAIT_WV3_CASE_BOUNDARY 상태 이후 확인 결과다. 컨테이너 시작을 학습 시작으로 해석하면 안 된다. QG40 학습 완료 결과 및 해당 Sheet 업로드는 아직 없다.

## 유휴 서비스 오인 대기 수정

12:26 KST에는 WAIT_LOCAL_RESOURCE였다. PID 1643인 ComfyUI의 main.py를 일반 학습 프로세스 inventory가 감지했다. 기존 GPU 프로세스 검사에 적용했던 사용자 승인 유휴 서비스 정책이 inventory에는 적용되지 않은 것이 원인이다.

.git/codex-qg40-launch.py의 s2 전용 운영 래퍼에서 기존 tools.mix20h_resource_policy.filter_idle_services를 inventory 결과에도 적용했다. exact PID, start_ticks, command hash, GPU 메모리 여유 및 사용률 검증을 그대로 사용한다. 미등록 작업은 제외되지 않음을 확인했다. ComfyUI 등 프로세스를 종료하지 않았다.

이후 CPU 161개 테스트, 정책 6개 테스트 및 데이터/GPU preflight를 통과했다. 12:31:38 KST의 preflight.json은 complete=true, real_native_forward=true, method_tests=26이다. historical R2가 pin한 tools/fh20r1_runner.py와 QG40 numerical source는 수정하지 않았다.

## 남은 차단 1: GPU 간 Teacher probe 출력 차이

원본 s4 GPU export를 s2 GPU로 import할 때 parity 검증에 실패했다.

- s4 GPU 대비 s2 GPU 출력 최대 절대 차이: 0.0001634359359741211
- delta 최대 절대 차이: 0.000039523118
- 같은 frozen Teacher의 실제 s2 GPU q-cache/online 검사는 통과: 최대 차이 0.0000017583370208740234
- 적용 허용오차: atol=3e-6, rtol=2e-5. 변경하지 않았다.

공식 export_reference/import_reference API가 지원하는 device='cpu' 경로를 별도 archive로 시험했다. 원본 GPU archive, Teacher checkpoint, calibration, q-cache 및 기존 reference는 보존했다.

CPU export/import probe는 통과했다:
- output: 6.556510925292969e-7
- delta: 1.1175870895385742e-8
- CPU q-cache/online: 1.7881393432617188e-6

그러나 다음 데이터 signature 검증에서 실패하여 reference_manifest.json은 발행되지 않았다. pending_reference.json의 complete=true는 전체 import 성공을 의미하지 않는다. CPU 전송이 통과한 사실을 GPU 출력 일치로 해석하면 안 된다.

원본 GPU archive SHA256:
8424026acbde8b1b73f424995a9f25df01df2cbb807810e1c3e205733aca6e94

별도 CPU archive SHA256:
07d5559fceb6ef8324de6e3b14636ccac5b9daa37d98ea7aef60c527ec18e01a

Teacher checkpoint SHA256:
426b6f721a73eaacb4686c901e8930f0079da859801479f4259965da49eca065

## 남은 차단 2: CPU 최적화별 LP cache의 비트 차이

오류: Local sensor/split/LP/band source differs from Teacher calibration

raw source SHA, msfix 검증, PAN/GT, band/MTF/source provenance, recipe, row order는 일치한다. 차이는 train/val의 lpan_canonical_sha256 두 항목이다. RR/FR LP 캐시 내용은 일치한다.

| 항목 | s2 | s4 |
| --- | --- | --- |
| CPU | AMD Ryzen 9 5900X | AMD Ryzen 9 9950X |
| OpenCV | 4.10.0 | 4.10.0 |
| IPP 계산 경로 | AVX2 (l9) | AVX-512F/CD/BW/DQ/VL (k0) |
| val LP canonical SHA | 41109ab5de530dcb49fcccaff02f97167ce697331db72f91686e619dd1b4f78d | 4bfb0a170da75215c3bfc276361452030356df77f959cac511f138cab928a6b6 |
| train LP canonical SHA | d52906c292a0a63175c9b8795533a77e73144ab2f3387c4f0f47c10948a89e50 | 5e0f0659e8bcadc83c6db140479a19e80fc700155bdf364f2086e17cbe766ddc |

원본과 캐시를 변경하지 않는 별도 진단 프로세스에서 val 1,905개 전체를 재계산했다. 각 서버의 기본 IPP 결과는 그 서버 기존 캐시와 비트 단위로 동일했다. 같은 프로세스에서 IPP를 비활성화하면 양 서버 모두 아래 동일한 SHA를 생성했다:
4556288930ce61dddf4bf6c62e9b510605ffd21df5175ddaf85871df78196fed

각 서버의 기본 캐시와 IPP 비활성화 결과 간 최대 절대 차이는 2.3283064365386963e-10이었다. 바뀐 원소 수는 s2 2,048개, s4 2,046개였다. 이는 val 전체 진단이며 train 전체에 대한 수치 오차 측정은 아니다. 해당 IPP 변경은 진단 프로세스 안에서만 적용했고 운영 환경이나 캐시에 반영하지 않았다.

따라서 같은 원본과 레시피에도 CPU 실행 경로에 따라 cache canonical hash가 달라지는 문제가 재현되었다.

## 코드 수정 시 필요한 처리

현재 qg40/data.py:97의 verify_lp_cache는 현지 재계산과 np.array_equal을 요구한다(112행). qg40/references.py:32의 data_signature는 LP canonical SHA를 포함하고, 114행에서 Teacher calibration 당시 signature와의 완전 일치를 요구한다. 따라서 s4 캐시만 복사하면 현지 재계산 검사에서 실패하고, s2에서 IPP만 끄면 s4의 이미 확정된 Teacher signature와 여전히 다르다.

기존 Teacher/Student와 source pin을 유지하는 명시적인 호환 처리 또는 버전 이행이 필요하다. 한 가지 검토안은 원본 provenance와 owner LP bytes/hash를 고정한 채, 소비 서버의 LP 재계산을 근거 있는 수치 오차 기준으로 검증하는 경로를 별도 정의하는 것이다. 이는 제안이며 이번 진단에서 적용하지 않았다. 모든 LP 해시 검사를 단순 제거하거나 BLOCKED 상태만 지우는 조치는 하지 않았다.

현재 numerical source content SHA는 s2/s4 모두 다음을 유지한다:
0cf7bf2af1a81036a115b3b9a181c4cc0d3f292e7a7927383f25c5819dee9d8c

공통 t0, admission cutoff, deadline을 변경하지 않았다. 기존 s4 학습/Teacher/calibration도 변경하지 않았다.

## 증거 위치

s2 .git/codex-qg40-deploy-20260920/s2/:
- inventory-policy-before-restart.json
- qb-cross-device-parity-diagnostic.json
- cpu-reference-recovery.json (성공 기록이 아님)
- reference-data-signature-difference.json
- opencv-lp-portability-diagnostic.json
- status-after-diagnosis.json
- unpublished-QB_TA-gpu-import/ (최초 실패 import 보존)
- qg40-launch.before-inventory-policy.py

s4 .git/codex-qg40-deploy-20260920/s4/:
- cpu-reference-export.json
- opencv-lp-portability-diagnostic.json

현재 CPU import 파일은 s2 work_dir/_qg40/s2/references/QB_TA/에 남아 있으나 미발행이다. 자동 복구는 BLOCKED_INTEGRITY를 강제로 재시작하지 않는다.

이 문서의 로컬 동반 기록: 2026-09-20_qg40-s2-reference-blocker.json.
