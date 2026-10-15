# s2 QG40 복구 수정본 보관 — 2026-09-20 13:58 KST

후속 지시로 s2 운영 반영 및 실제 학습 재개를 완료했다. GitHub push 보류는 유지한다. 최신 상태는 [실험 재개 기록](2026-09-20_qg40-s2-experiment-started.md)을 참고한다. 아래 내용은 보관 결정 당시 기록이다.

상태: LOCAL_ONLY_NO_DEPLOY

사용자가 “push하지 않고 수정본만 보관”을 선택했다. 원격 push와 s2 운영 배포는 진행하지 않는다. 추후 별도 지시 없이 이 수정본을 자동 반영하지 않는다.

## 보관 위치와 코드

- s1 작업 디렉터리: /home/knuvi/Desktop/song/PAN-Crafter-ops-qg40-s2-20260920
- 로컬 브랜치: ops/qg40-s2-portability-20260920
- 로컬 커밋: 91d2f5981f66c015604dae7b715672f351ab6af6
- 기반 커밋: a19dfeb4733480b6132890a9ad4a9fff20c3bdd4
- 변경: qg40/common.py, qg40/data.py, qg40/portability.py, qg40/portability_release.json, qg40/test_portability.py, research_log/0920/2026-09-20_qg40-s2-portability-fix.md
- s2 격리 검증 경로: /home/honbul/Desktop/song/PAN-Crafter/.git/codex-qg40-portability-source
- 원격 브랜치가 존재하지 않음을 git ls-remote로 확인했다.
- push를 포함한 최초 명령은 자동 승인 심사가 실행 전에 거절했다. 이유는 이 복구 코드·문서와 GitHub 목적지에 대한 명시적 전송 승인 부족이었다. 이후 push 없이 로컬 커밋만 만들었다.

## 검증 결과

기존 161개와 새 무결성 회귀 테스트 14개, 총 175개가 통과했다. 테스트 경로 의존성을 제거한 후 신규 14개도 다시 통과했다.

이미 시작된 격리 실데이터 검증은 13:58:40 KST에 정상 종료했다.
- 실제 GPU preflight: complete=true, real_native_forward=true, method_tests=26.
- s4 frozen Teacher의 데이터 signature와 s2 격리 데이터 signature가 완전히 일치했다.
- 공식 CPU reference import: PASS. output=6.556510925292969e-7, delta=1.1175870895385742e-8. 기존 atol=3e-6, rtol=2e-5 유지.
- 실제 s2 CUDA q-cache/online 검사: PASS, 최대 오차=1.7583370208740234e-6.
- canonical LP train: 전체 17,139개 검증, 최대 native PAN 재계산 차이=5.820766091346741e-11 DN.
- canonical LP val: 전체 1,905개 검증, 최대 차이=4.547473508864641e-13 DN.
- CPU LP 재계산만 최대 1e-10 DN, rtol=0인 별도 검증 경로를 사용한다. 학습에 사용하는 LP 파일 바이트와 canonical SHA는 s4 원본과 정확히 같다.
- 새 검증 소스 SHA: 304c42b960c1f8b6db725fe620121e162f26835dd0544f7b7c827c0d841d4fb3.
- Teacher checkpoint SHA: 426b6f721a73eaacb4686c901e8930f0079da859801479f4259965da49eca065.

이 검증은 격리 경로에서 수행되었다. 운영 Student 학습을 시작하거나 학습 완료를 검증한 것은 아니다. 외부 Sheet에 이 검증을 학습 결과로 업로드하지 않았다.

## 운영 상태

s2 운영 레포지토리의 기존 source identity 대상 파일에서 변경 0개를 확인했다. 운영 경로에는 qg40/portability.py가 없고, 상태는 BLOCKED_INTEGRITY, runs={}이다. 운영 컨테이너는 종료 상태다. s1 운영 레포지토리와 main 브랜치도 변경하지 않았다.

재개하려면 추후 승인된 전달 방식으로 수정본을 배포하고, 아직 소비되지 않은 s2 등록/사전검사 기록을 보존한 후 새 release로 등록해야 한다. 공통 시간창은 유지하고 운영 경로에서 preflight → CPU import → CUDA q-cache 검사를 다시 통과한 뒤 실제 Student 학습 update 증가를 확인해야 한다. 이 단계는 사용자의 보관 요청에 따라 수행하지 않았다.

원자료는 qg40_s2_portability/portability-real-validation.json 및 lp-full-pair-diagnostic.json에 있다.
