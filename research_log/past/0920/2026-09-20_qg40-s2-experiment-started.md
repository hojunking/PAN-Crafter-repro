# s2 QG40 실험 재개 완료 — 2026-09-20

사용자가 “실행을 못하는건가? 실험 진행해”라고 지시하여, 보관하던 검증 완료 수정본을 s2 운영 경로에 적용하고 학습을 시작했다. 이전 “push하지 않고 수정본만 보관” 결정 중 GitHub push 보류는 유지했다. GitHub로 전송하지 않았으며 s1의 로컬 커밋 91d2f5981f66c015604dae7b715672f351ab6af6과 동일한 검증 파일을 s2의 기존 격리 복사본에서 운영 경로로 반영했다. 운영 Git HEAD는 0ec113218f60f861dab18ab84f3083b730890979 그대로다.

## 실제 시작 확인

- 학습 시작: 2026-09-20 14:12:32 KST.
- 최종 확인: 14:16:11 KST, 상태 RUNNING.
- 실험: QGBASE_QB_S_PLH_W104_D122_TA_TS81001_SS82003_BASE_FRESH50_v1.
- QB / Student SS82003 / Teacher TA TS81001 / BASE / W104 D122 / PLH / FP32 / batch48.
- 실제 로그: 100부터 1,800 update까지 계속 증가.
- 저장된 상태: actual_updates=1010, n_evaluated=1. 상태 JSON은 주기적으로 저장되므로 당시 학습 로그보다 늦다.
- 첫 update1010 체크포인트 공식 평가 완료 후 학습을 계속 진행 중이다.
- GPU: 94%, 5,512 / 65,536 MiB.
- 컨테이너: pan-qg40-s2-20260920, 독립 Docker 실행.
- 전체 50,000 update 학습 완료나 최종 Sheet 업로드를 완료로 표기한 것은 아니다.

## 복구와 검증

기존 미시작 s2 캠페인 디렉터리 전체를 .git/codex-qg40-deploy-20260920/s2/unconsumed-campaign-before-portability-20260920/에 보존했다. 기존 학습 run/checkpoint가 없는 것을 확인한 후 새 검증 release로 등록했다. 이전 registration의 original_wv3_hold도 새 등록 메타데이터에 보존했다.

s4 frozen Teacher가 실제 사용한 LP 파일을 바이트 그대로 사용한다. 파일 SHA, canonical content SHA, source SHA, recipe, dtype/shape, 순서와 모든 PAN 대응을 재검증한다. CPU 재계산에서만 1e-10 DN 절대 오차, rtol=0인 명시적 호환 경로를 쓴다. 실제 최대 오차는 train17139에서 5.820766091346741e-11 DN, val1905에서 4.547473508864641e-13 DN이었다.

운영 경로의 preflight → 공식 CPU reference import → 실제 CUDA Teacher q-cache 검증은 14:09:38 KST에 모두 통과했다. CPU import output 오차는 6.556510925292969e-7, delta는 1.1175870895385742e-8, CUDA q-cache/online 최대 오차는 1.7583370208740234e-6이었다. 기존 reference parity의 atol=3e-6, rtol=2e-5는 유지했다. 과거 GPU 간 full-output 차이가 사라졌다고 주장하지 않으며, 공식 CPU 전송 경로로 검증했다.

수치 source 중 모델·loss·training·calibration·metric 구현은 바이트 단위로 동일하다. 변경된 validation 코드와 전용 호환성 manifest를 새 source identity에 포함하고 이전 Teacher source와의 호환을 정확한 파일 해시 목록으로 제한했다. 기존 Teacher·calibration·q-cache·seed·학습 설정은 유지했다.

- 새 source content SHA: 304c42b960c1f8b6db725fe620121e162f26835dd0544f7b7c827c0d841d4fb3.
- config SHA: 45f230c13ab92b05c1c0fdd9ef3c242caa23ab250a5c602a3e7b2b19f6a8b945.
- data manifest SHA: c5cea2c00e0059afad056e483f7c492b666eee242360ce75bbecbdb9420f3431.
- Teacher checkpoint SHA: 426b6f721a73eaacb4686c901e8930f0079da859801479f4259965da49eca065.

전체 QG40 175개 테스트 및 기존 유휴 서비스 정책 6개 테스트가 운영 실행기에서도 통과했다. 유휴 서비스는 종료하지 않았다.

공통 t0(09-20 05:12:28 KST), 신규 입장 마감(09-21 17:12:28 KST), 종료(09-21 21:12:28 KST)를 유지했다. 재개로 시계를 리셋하지 않았다.

## 지속 실행과 결과

자동 복구 maintenance hold를 검증 완료 후 해제했다. 기존 5분 간격 및 부팅 시 Docker 복구 cron이 남아 있으며, 실제 helper 실행 결과 ALREADY_RUNNING을 확인했다. SSH 종료와 독립적으로 계속 실행된다.

실행기는 run 학습 완료 → postrun 공식 평가/선택 → upload → Sheet readback 검증 순서로 진행하며 업로드 실패를 재시도한다. 이 경로와 기존 승인을 유지했다. 이번 시작 확인으로 새 학습 결과의 Sheet 업로드 완료를 주장하지 않는다.

## 증거

s2 .git/codex-qg40-deploy-20260920/s2/:
- portability-deployment.json
- portability-production-validation.json
- portability-training-confirmed.json
- portability-start-final-status.json
- source-before-portability/
- unconsumed-campaign-before-portability-20260920/
- portability-maintenance-hold.resolved.json

재현용 운영 스크립트는 로컬 server_ops/qg40_s2_portability/deploy_local.py와 validate_production.py에 보관한다. 배포 스크립트는 재실행을 전제로 하지 않으며 기존 학습이 시작된 이후에는 초기화 작업을 다시 하면 안 된다.

