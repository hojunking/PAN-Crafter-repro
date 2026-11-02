# s2 QB 6-case 실행 구현·인계 — 2026-11-02

승인된 [6-case 설계](2026-11-02_qb-s2-30h-plan.md)를 `panda_qb_s2`에 등록했다.
새 s2에서는 **Stage 1 50K → 해당 Student 30K**를 6회, 총 12개 job으로 실행한다.
λ_abs/λ_ε/Stage 1 seed 순서는 A2/.03/D101, A4/.03/D101, A2/.03/D301,
A4/.03/D301, A3/.03/D101, A4/.06/D101이다. 모든 Student seed는 261018101이다.
예상 시간은 s1 실측 기준 34–36시간이며 30시간 강제 종료는 없다.

실행·재개·데이터 준비 명령은 [인계 README](../panda_qb_s2/README.md),
정확한 12개 job은 [실행 CSV](../config/panda_qb_s2/cases_s2.csv),
입력 명세는 [plan.json](../config/panda_qb_s2/plan.json)에 있다.
이 기록은 코드 구현과 로컬 검증 결과다. **s2 원격 학습을 시작한 기록이 아니다.**

**동작**

- 독립 campaign `PANDA_S2_QB_STAGE1_BALANCE_20261102_v1`, output `work_dir/panda_qb_s2`,
  run/Sheet prefix `PQS2`를 사용한다. s1/s3의 기존 컨테이너·donor·큐 종료를 요구하지 않는다.
- 각 Stage 1을 처음부터 학습하고 full50K 이후 기존 validation RRE 선택 규칙으로 donor를 정한다.
  같은 계수라도 D101/D301 donor와 calibration을 별개로 검증·저장한다.
- Student는 기존 scale1/Lanczos-4/DIFF/UBIL/Hard-only를 유지한다.
  LR1e-4, cosine30K, W112/D123, batch48, α.25/γ.5/β0이다.
- 알려진 support 실패는 저장된 checkpoint와 trace 근거를 확인한 뒤 해당 Student만 보류한다.
  다른 독립 case는 계속한다. 다른 오류는 중단하며 부분 donor로 대체하지 않는다.
- Student 8·9·10K HQNR이 모두 .90 미만인 기존 조기 중단을 유지한다.
  모델·optimizer·RNG·sampler를 저장·복원하며 완료한 Stage 1을 재학습하지 않는다.
- 큐 종료 후 마지막 Sheet readback을 확인한다. WV3 자동 후속 실행은 없다.

**구버전 서버의 실행 경로**

기존 실행 컨테이너를 복제하지 않고 Dockerfile.env 기반 환경과 명시적인 mount로 시작한다.
코드·데이터는 read-only, 새 output만 writable이다. image ID와 시작 기록을 저장한다.
중복 명령은 기존 컨테이너를 확인하고, 중단 후 `--resume`은 동일 컨테이너를 재개한다.
다른 서버를 기다리는 승인/lock 조건은 없다. 로컬 파일 mutex는 중복 writer 방지용이다.

전달 코드 471개 파일의 SHA를 [release.json](../config/panda_qb_s2/release.json)으로 검증한다.
라이브러리 버전, 평가 함수·DLPan SHA, QB H5 SHA와 original LMS override를 확인한다.
실제 CUDA update/save-resume 및 RR/FR native 추론 사전 검사는 s2에서 각 단계 학습 전에 수행한다.
환경이 다르면 manifest를 덮어쓰지 않고 기준 환경을 설치한다.

**Sheet와 weights**

목표는 동일 checkpoint에서 **HQNR≥.9315 AND ERGAS≤3.5**다.
`QB-ablations`에 server=s2로 동시 통과 step 전부와 run별 대표 하나를 게시한다.
대표는 HQNR 충족 후보 중 ERGAS 최저, 없으면 HQNR 최고이며 중복을 합친다.
Stage 1 선택 RRE와 계수·두 단계 seed·LR·길이·입력 구성 등을 행에 남긴다.
모든 1K 평가 weights와 완전 재개 상태를 저장하므로 목표 통과 weights도 보존한다.

**검증**

검증 결과와 원본 테스트 출력은 [validation.json](assets/2026-11-02_qb-s2-implementation/validation.json),
[unittest 로그](assets/2026-11-02_qb-s2-implementation/unittest.txt)에 보존한다.

- 21개 테스트: 6개 donor의 실제 loss·gradient 기준 구현 일치, Student 일치,
  두 단계 optimizer/RNG/stream의 완전 재개, donor seed 혼동 거절,
  full queue/재개/지원영역 실패 격리, Sheet 목표·대표·중복 처리,
  launcher idempotence/launch receipt 복구, 게시기 실패와 최종 readback.
- 커밋할 Git tree만 추출한 clean checkout에서 release 검증·전체 테스트·CSV 일치를 확인했다.
  로컬 미추적 학습 코드나 과거 work_dir 없이 검증했다.
- 기준 Docker 이미지에서 GPU 없이 native 검사 통과:
  train17139 / val1905 / RR20 / FR20, 입력 파일 SHA, 평가 코드와 라이브러리 버전.
- 같은 Docker 경로에서 실제 `QB-ablations` 읽기 접속·헤더·게시 형식 검사 통과.
  이 검사는 Sheet 값을 수정하지 않았다. s2에서 본 실행 전에 다시 검사한다.
- 기존 QB 기준 463개 파일은 HEAD와 동일하다. 기존 s1/s3 구현·큐는 수정하지 않았다.

GPU 학습이나 native CUDA 사전 검사를 현재 s1 GPU에서 실행하지 않았다.
따라서 s2 장치에서의 CUDA 실행·전체 장시간 학습 성공은 아직 확인되지 않았다.
사용자가 push한 뒤 s2가 전체 커밋을 받고 README의 launcher를 실행하면 된다.
