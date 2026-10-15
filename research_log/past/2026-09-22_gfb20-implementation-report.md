# GFB20 구현·검증 보고 — 2026-09-22

## 결과와 범위

`PANDA_GF2_B20_S345_20H_Bundle_2026-09-22`의 registry/CSV/queue를 읽는 별도 `gfb20/`와 CLI를 추가했다. core FT18개, 조건부 포함 최대36개를 s3/s4/s5에 배정한다. 기존 캠페인의 수치 구현은 필요한 경로를 재사용하며, 기존 G20/L100/ABLR2 코드·실행 및 s1/WV3·s2/QB 정책은 수정하지 않았다.

이번 작업은 구현·로컬 검증이다. **커밋, push, 새 학습 시작, 실제 Sheet 쓰기는 하지 않았다.** 기존 dirty worktree의 다른 변경·삭제 파일은 유지했다. 운영 명령과 제한은 `gfb20/README.md`에 정리했다.

## 구현 연결

| 영역 | 파일 | 핵심 확인 |
| --- | --- | --- |
| 원본 설계 | `gfb20/plan.py`, `policy.py` | 원본 checksum 고정, 36/18개 구성, paired block, 공통 16/18/20h 제한 |
| 배포·실행 | `deployment.py`, `controller.py`, `resources.py`, `tools/gfb20_runner.py`, `tools/gfb20_start.sh` | committed frozen release, 기존 worker 보호, 전체 block 예약, 동일 상태 재개, 평가 부채 |
| 부모·데이터 | `assets.py`, `native_data.py`, `preflight.py`, `anchor.py` | 원본 U/A·fullstate·SHA, 기존 Teacher/cache/runtime, QG40 wrapper, 부모 native 재평가 |
| 학습 | `training.py`, `stream.py`, `losses.py`, `diagnostics.py` | FT optimizer reset, paired RNG, 고정 LR/계수 ramp, detach/gradient routing, B 필수 진단 |
| s5 분포 | `augmentation.py`, `data.py`, `calibration.py` | PAN gain→LP→geometry, gamma1 bypass, full q cache/parity, mixed calibration |
| 평가·보고 | `evaluation.py`, `postrun.py`, `reporting.py`, `upload.py` | native RR20/FR20, 고정 후보, selector 분리, RMSE/CC/JQM, 전용 탭/readback |

원본 SHA256SUMS 자체의 SHA는 `32a752c09c44401df8f53bf09d33bb8a03dca35e7b2ad02d55766f82c416d307`이며, 그 안의 14개 파일도 검증했다. 원본 bundle은 변경하지 않았다.

## 통합 검토에서 수정한 결함

- 실제 소비한 첫 256 stream 항목과 JSON 재개 기록의 tuple/list 차이를 제거하고, update마다 전체 epoch를 다시 만드는 경로를 제거했다.
- 학습 완료 직후 controller가 중단돼도 재학습하지 않고 postrun으로 복구한다. 안전 신호 중단은 terminal failure가 아니다.
- 저장 후보/정확한 endpoint를 무거운 진단보다 먼저 보존하고, fresh fork는 trunk의 공식 평가까지 완료되어야 허용한다.
- P3OLD의 구 QG40 data schema를 원본 변경 없이 새 lane wrapper로 연결했다.
- 허용된 native q parity 잔차는 stable resume identity에서 분리했다. fresh parent는 변하는 비용·시간 요약 SHA 대신 고정 checkpoint/선택 identity에 결합했다.
- s4 성공 지표만으로 승격하지 않고, 0/5K/20K 진단의 실제 checkpoint/source/config 대응을 요구한다. 양 K1 부모의 LOW_ACTIVE_MASS이면 E100을 고정하되 K1의 성공을 이월하지 않는다.
- 미래 t0 이전 준비 실행과 완료 전 fresh 의존성 소비를 차단했다.
- 업로드 전 실제 후보 grid·checkpoint bytes·endpoint sampler·노출·끝 LR를 다시 검증한다. 기존 Sheet schema를 덮지 않는다.

## 검증 증거

- GFB20 CPU unit/regression **89개 통과**: assets13, anchor5, controller7, native data5, operations7, PAN views12, plan5, policy9, postrun8, training18.
- 기존 L100 CPU regression **118개 통과**.
- 기존 ABLR2 CPU regression **131개 통과**.
- 최종 통합 재실행에서도 **총 338개 테스트 통과**(26.081초). 테스트 내 1–4 update 출력은 임시 fixture smoke이며 실제 캠페인 실행이 아니다.
- 실제 W104 D122 Student / W112 D123 Teacher 구조의 CPU diagnostic smoke 수행: 128개 진단 표본 경로 및 첫 24개 gradient backward 정상. 작은 fixture를 사용한 smoke이며 실 GF2 전체셋/GPU 검증은 아니다.
- 새 Python 33개 AST 검사, 파일 36개 trailing-whitespace 검사, launcher bash syntax 검사, 원본 bundle checksum 검사 통과. `work_dir/_gfb20`는 생성하지 않았다.
- Worksheet 테스트는 mock이다. 실제 Google Sheet를 쓰거나 credential을 출력하지 않았다.

## 남은 현장 검증 및 명시적 제한

1. 실제 s3–s5의 등록 부모·Teacher·원본 reference/cache/data 및 RTX5090 환경 일치 여부는 시작 시 검증한다. 이 로컬 작업으로 원격 자산 존재·수치 일치를 확인했다고 주장하지 않는다.
2. full mixed q-cache와 calibration은 s5에서 실제 실행한다. 19809×4×3 전체 요구를 축소하지 않았다. 그 실행 비용도 공통 20시간에 포함한다.
3. 부모 equality의 `atol=1e-6/rtol=0`은 구현상 사전 고정값이다. GPU 반복 측정 근거가 아니며 실패할 때 자동 완화하지 않는다.
4. 기존 L100 queue를 자동 drain시키는 기능은 없다. 기존 worker가 종료될 때까지 안전 대기하며 대기 시간도 예산에 포함한다. 시작 시 기존 큐 인계를 조율해야 한다.
5. 선택적 P1 sensitivity replay와 새 독립 정합 추정기는 미구현이다. 검증된 추정기를 연결하지 않았으므로 `ESTIMATOR_NOT_AVAILABLE`로 분리한다. B gradient와 C view/q는 필수 검증 경로다.
6. GFB20 launcher 자체에는 Docker 생성 기능이 없다. 등록 부모를 생성한 동일 RTX5090 Python/runtime 사용을 전제로 한다.
7. 이 캠페인의 primary/secondary 선택은 새 bundle의 EXACT_FINAL/RR_VAL_SELECTED이다. 기존 캠페인의 HQNR 선택 정책 변경이 아니다. 모든 결과는 test-aware development로 보고한다.
8. `STOP_NOW_SAFE`의 즉시 control 감시는 trainer에 있다. 준비·mixed cache·postrun 단계에서는 다음 scheduling 경계까지 중단 반응이 지연될 수 있다. 이 명령이 모든 단계의 즉시 중단을 보장한다고 해석하지 않는다.

현 상태는 **구현·로컬 회귀 통과, 배포 및 원격 P0 검증 전**이다.
