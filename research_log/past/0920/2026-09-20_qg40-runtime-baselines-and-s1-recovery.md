# 현재 실험의 Git·실행 기준과 s1 복구 — 2026-09-20

기록 시각: 2026-09-20 21:47 KST. 현재 실행 캠페인은 QG40이다.
G20은 코드 배포 및 s3 원본 확인 단계이며 아직 공통 clock/실행을 시작하지 않았다.

## 복구 결과

s1의 QB Student seed82005는 17:27에 50,000-step 학습과 fixed-grid 50회 평가를 마쳤지만,
postrun에서 현재 source_identity와 저장된 source_identity의 git_release 차이로 중단됐다.
비교 결과 수치 코드 파일, content SHA, Torch/CUDA/cuDNN 등 나머지 identity는 모두 같았다.

등록·checkpoint·config의 SHA나 git_release를 수정하거나 검증을 생략하지 않았다.
원래 Git HEAD의 별도 worktree에 원본 QG40 코드를 보존하고, 컨테이너 안에서 기존 절대 경로와
동일한 native Python 환경을 연결했다. 실제 Git metadata 역시 이 고정 worktree의 HEAD를 읽는다.

- 복구된 run: QG40_QB_S_PLH_W104_D122_TB_TS81002_SS82005_BASE_FRESH50_v1
- 학습 재실행 없음. config, training status/start manifest, exact50K weights/identity,
  last fullstate, raw_grid의 복구 전후 SHA가 모두 동일하다.
- postrun/upload 명령 모두 exit 0.
- Sheet QB-s1 19행: READBACK_VERIFIED, 50,000 updates, n_evaluated=50.
- 업로드 완료: 21:40:31 KST. 실제 Sheet API 및 Google-rendered 화면에서 확인.
- 고정 실행기 시작: 21:42:32 KST. 다음 QB Student seed82008 학습을 시작했고
  21:46경 700 updates, GPU 91%를 관측했다. 첫 평가 이후 상태는 후속 확인을 따른다.
- 기존 QG40 cron 두 항목만 고정 실행기용 host recovery로 변경했다.
  다른 cron은 보존했고 readback을 검증했다.
- host recovery의 중복 호출은 ALREADY_RUNNING으로 종료했다.
  무결성 실패/완료/마감/미업로드 결과 처리에 관한 7개 gate 조건이 통과했다.

## 세 종류의 기준을 구분한다

1. **개발 main의 HEAD**: 문서·그림·새 캠페인 개발을 포함한다. 실행 중인 QG40의 기준으로 바꾸지 않는다.
2. **코드 전달 commit**: 원격에서 어떤 파일을 가져왔는지를 나타낸다.
3. **실험 실행 identity**: 실제 Git base + 전체 source content SHA + 라이브러리 + 데이터/설정/Reference의 고정 값이다.
   기존 checkpoint는 이 identity를 기준으로 검증한다.

| 용도 | 커밋 | 적용 |
|---|---|---|
| 기존 실행 Git base | cccedeeeffd7ed19686e5ea684489ceee23cf313 | s1/s3/s4/s5 QG40 checkpoint의 git_release |
| QG40 배포 코드 | a19dfeb4733480b6132890a9ad4a9fff20c3bdd4 | 기존 HEAD를 보존하고 QG40 파일을 overlay한 배포본 |
| s2 실행 Git base | 0ec113218f60f861dab18ab84f3083b730890979 | s2 서버 전용 checkout/등록 기준 |
| s2 복구 코드 | 91d2f5981f66c015604dae7b715672f351ab6af6 | local-only portability 수정본; GitHub에 push하지 않음 |
| G20 배포 코드 | e2d838a3f9ea45170fb71052710b38acc9992c61 | 다섯 서버의 G20 신규 파일 77개 배포 기준 |
| 최신 개발 그림 commit(관측값) | f94ee4826732622cab4fbc09e0e1ca88fd498c28 | 21:43:44의 그림 변경; QG40 실행 기준이 아님 |

21:44에 개발 main이 f94ee48로 바뀐 이후에도 s1의 고정 실행 HEAD는 cccedee이고 등록 identity가 유지됐다.
기존 history를 reset/rebase/amend하지 않았고, 운영 문서와 helper는 별도 local ops branch로 기록한다.

## 서버별 현재 실행 기준

| 서버 | 데이터셋 | 실행 git_release | source content SHA256 앞 12자리 | Torch / CUDA / cuDNN |
|---|---|---|---|---|
| s1 | QB | cccedee | 0cf7bf2af1a8 | 2.4.0 / 11.8 / 90100 |
| s2 | QB | 0ec1132 | 304c42b960c1 | 2.8.0+cu128 / 12.8 / 91002 |
| s3 | GF2 | cccedee | 0cf7bf2af1a8 | 2.8.0+cu128 / 12.8 / 91002 |
| s4 | QB | cccedee | 0cf7bf2af1a8 | 2.8.0+cu128 / 12.8 / 91002 |
| s5 | GF2 | cccedee | 0cf7bf2af1a8 | 2.8.0+cu128 / 12.8 / 91002 |

원본 content SHA:
0cf7bf2af1a81036a115b3b9a181c4cc0d3f292e7a7927383f25c5819dee9d8c

s2 portability content SHA:
304c42b960c1f8b6db725fe620121e162f26835dd0544f7b7c827c0d841d4fb3

21:44 상태: s2/s3/s4 학습 중; s5는 배정된 9건을 완료·업로드하고 FINITE_QUEUE_FINISHED로 정상 종료.
s1 완료 업로드 수는 복구 후 8건이다.

## s1 실행 위치와 운영

개발 checkout:
`/home/knuvi/Desktop/song/PAN-Crafter`

고정 실행 worktree:
`/home/knuvi/Desktop/song/PAN-Crafter-runtime-qg40-s1-20260920`

실행 컨테이너:
`pan-qg40-s1-frozen-20260920`

컨테이너의 저장소 경로는 원래와 동일한 `/home/knuvi/Desktop/song/PAN-Crafter`이다.
소스 worktree와 native conda 설치, Git metadata는 읽기 전용으로 연결하고,
work_dir는 원본 결과 디렉터리를 공유한다. 원본 config와 manifest의 절대 경로를 바꾸지 않는다.
Git 실행 환경은 실제 고정 worktree metadata를 가리키며 임의 문자열로 commit 값을 대체하지 않는다.

운영 스크립트:
`/home/knuvi/Desktop/song/PAN-Crafter/.git/codex-qg40-s1-recovery-20260920/host_recovery.py`

검증·복구 증거:
`/home/knuvi/Desktop/song/PAN-Crafter/.git/codex-qg40-s1-recovery-20260920/`
(runtime.json, runtime-validation.json, postrun-recovery.json,
completed-run-protected-before.json, state.before.json, registration.before.json,
crontab.before.txt, container-identity.json, host-recovery.json)

등록·clock·checkpoint 원본 경로는 기존 work_dir 그대로다.
재시작은 host recovery를 통해 수행한다. main checkout에서 QG40 start/run을 다시 실행하지 않는다.
호스트 helper는 source bytes, 실제 worktree HEAD, 기존 컨테이너 identity를 확인하고,
BLOCKED_INTEGRITY나 닫힌 큐를 임의로 새 학습으로 바꾸지 않는다.

## 이후 코드 전달 기준

- QG40가 도는 저장소에서 일반 pull/reset으로 실행 HEAD를 바꾸지 않는다.
- 논문 그림·문서·G20 개발은 개발 checkout에서 진행한다.
- 새 실행본은 별도 고정 위치와 검증된 source manifest로 연결한다.
- 완료 원본의 등록 정보·git_release·tau/qref/SHA를 현재 개발 HEAD에 맞춰 다시 쓰지 않는다.
- s2 portability는 기존 local-only 결정을 유지한다. 본 복구도 GitHub에 push하지 않는다.
- 공통 QG40 deadline 2026-09-21 21:12:28 KST와 신규 입장 마감 17:12:28 KST는 그대로다.

G20 착수는 별도 단계다. s3 GF2_TA/TA/S92001 원본 위치는
2026-09-20_g20-sync-s3-reference-availability.md에 있으며, common evaluator replay PASS나
G20 학습 시작을 이 복구 기록에서 주장하지 않는다.
