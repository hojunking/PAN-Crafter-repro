# GFP40 구현·검증 보고 — 2026-09-22

## 결과

제공된 P40 MD와83-case CSV에 따라 별도 `gfp40/` 및 `tools/gfp40_runner.py`, `tools/gfp40_start.sh`를 구현했다. 이후 사용자의 서버 간 대기 제거 요청에 따라 **고정 G025 confirmation·서버별 독립40h 실행**으로 운영 정의를 수정했다. PRIMARY74/OPTIONAL9, s3 19/s4 31/s5 33 구간이며 새 Teacher는0이다. 실행 명령과 전제는 `gfp40/README.md`에 정리했다.

이번 범위는 **구현과 로컬 검증**이다. 커밋·push·원격 기동·새 학습·실제 Sheet 쓰기를 하지 않았다. B20 clock/queue/결과와 s1/WV3·s2/QB 실행은 변경하지 않았다.

## 원본과 파생 자료

- Plan SHA256: `246938acd8df82a379c0020358152aa9198be5a0a0df9b80e46def0ca3095eea`.
- CSV SHA256: `5818dbb59c11316b985173b6022367e9233399414064b1e8affe97618c9b8c09`.
- MD가 언급한 author ParentAssets/Registry/Queue JSON은 제공되지 않았다. 구현 registry는 MD+CSV 파생임을 명시했다.
- `parent_registry.json`은 Git5d9f4d6의 원 B20 ParentAssets와 archive bytes를 SHA 검증한 뒤 native100K6부모만 추출했다. 출처를 보존했으며 원격 자산 검증 완료를 주장하지 않는다.
- `060ba8...`의 실제 B20 frozen archive를 별도 검사하고 공유 모델·loss·warp·LP·data·metric SHA 및 P40 diff를 보존한다. 다른 commit과 임의로 동일시하지 않는다.

## 후속 수정 — 서버 간 선택 대기 제거

사용자는 서버 간 소통 실패로 실험이 진행되지 않는 문제를 피하도록 요청했다. 원본 MD/CSV는 위 SHA 그대로 보존하며, 다음 실행 차이를 registry/method revision과 파생 case에 명시한다.

- 원본의 s4 MSTAR 선택·공유 lock 전달·공통 clock 의존성을 제거한다. confirmation은 모든 서버에서 **G025 사전 고정**이며 s4 screen은 탐색 결과 보고만 한다. screen 승자는 이번 confirmation의 입력 분포를 바꾸지 않는다.
- 실행 phase는 `FIXED_CONFIRM`, 관련 run ID의 `MSTAR` 부분은 `G025_FIXED`로 구분한다. `MSTAR_LOCK` 의존성은 제거하지만 자기 서버 fresh100K 부모의 공식 완료 전 fork 금지는 유지한다.
- `bash tools/gfp40_start.sh --server s3`처럼 자기 서버 명령만 실행하면 최초 시작 시 local t0를 자동 기록한다. s4/s5도 동일하며 공유 경로·lock 전달·다른 서버 완료 통보가 필요 없다. 최초 시작에 자기 서버의 실제 t0를 명시하는 옵션은 유지한다.
- clock은 `work_dir/_gfp40/<server>/campaign_window.json`이다. 서로 다른 서버 시작 시각은 허용하고 각 서버에서34h 새 block 중단·37h optimizer 중단·40h 종료 규칙을 적용한다. 재개는 원래 로컬 clock을 재사용하며40h를 다시 시작하지 않는다.
- 전체 block의 미완 tail/cache 예약은 유지하되 미정 MSTAR 승자용 cache 예약은 고정 G025 예약으로 바꾼다. 같은 서버의 기존 worker 안전 대기·자산 검증은 남는다.
- 완료 후6-pair 집계는 동일 clock/lock을 요구하지 않는다. 서버별 clock을 각각 보존하고 동일 고정 G025 정책·방법/registry revision·Teacher/parent 계보·endpoint identity를 확인한다. 결과 파일 수집은 학습의 선행 조건이 아니다.
- 제거 범위는 서버 간 선택 동기화다. 같은 서버의 중복 runner·경쟁 쓰기를 막는 로컬 process/file lock 및 checkpoint/reference 무결성 검사는 유지한다.

Teacher는 원래부터 R3_100/R4_100/R5_100으로 서버별 독립이었다. 새 Teacher를 재학습하면 기존100K Student 부모와 연결된 Teacher/reference·q/τ/qref 계보 및40h 예산이 바뀐다. 따라서 이번 독립 실행 수정에서는 기존 로컬 Teacher를 유지하며, 새 Teacher 학습을 임의로 추가하거나 기존 Teacher ID로 대체하지 않는다. 신규 Teacher를 포함하는 변경은 별도 실험 정의·자산·예산 조정 대상이다.

## 구현 대응

| 계획 영역 | 구현 |
| --- | --- |
| 83-case/74-primary,20/60/100K | `plan.py`, `training.py`, `postrun.py` |
| 7분포·uniform pairing·bank | `augmentation.py`, `data.py`, `stream.py`, `calibration.py` |
| 원 native 부모·Teacher·source 감사 | `assets.py`, `parent_registry.json`, `preflight.py`, `anchor.py` |
| 고정 G025·로컬34/37/40h·독립 DAG | `policy.py`, `controller.py`, `resources.py` |
| frozen release·보호·재개 | `deployment.py`, CLI/shell launcher |
| CC/CM/MC/MM·native 진단·영상 | `diagnostics.py`, `replay.py` |
| native 평가·전용Sheet·6-pair 집계 | `evaluation.py`, `postrun.py`, `upload.py`, `reporting.py` |

핵심 수식과 α1/β.1/edge.002 및 U/A recipient별 gradient routing은 유지한다. FT는 자기 부모 U/A와 새 optimizer, fresh는 새U+독립 Teacher A clone이다. FT60K의20K/40K는 curve 진단이며3000 간격20후보에 끼우지 않는다. P40 primary EXACT_FINAL/secondary RR_VAL_SELECTED/aux RAW_AUX는 기존 캠페인의 HQNR 정책 변경이 아니다.

## 최초 구현의 통합 감사 이력

아래는 최초 MD/CSV 대응 구현에서 확인·수정한 이력이다. 이 중22h 선택·MSTAR 전송·동일 lock 집계 관련 내용은 위 후속 수정으로 대체됐으며 현재 실행 조건이 아니다.

1. `research_log/past` 이동으로 공통 plan import가 실패했다. `fh12/source_documents.py`와 QG40/G20/L100 plan의 총8줄 추가·5줄 교체로 원 경로 부재 시 동일 SHA archive를 읽도록 했다. 원본이 변조돼도 archive로 우회하는 경로는 없다. 사용자 파일 이동/삭제를 되돌리지 않았고 numerical 수식은 바꾸지 않았다.
2. 22h 선택 cutoff를 학습 종료시각이 아닌 **최초 공식 평가 완료시각**으로 고정했다. 늦게 평가한 결과를 조기 결과로 잘못 승격하지 않는다.
3. 이미 입장한 pair/block을 재추정 예산 초과로 취소하지 않는다. 미입장 block만 축소하고 진행 중 작업은37h safety boundary에서 보존한다.
4. 미완 gamma/family cache 예약을 elapsed setup 비용으로 소진시키지 않는다. native trunk 입장에도 향후 MSTAR tail의 cache 공간·시간을 포함한다.
5. lock 공유 디렉터리 I/O 실패는 pending transport이며 학습 자식 종료 사유가 아니다. lock 자체 불일치는 무결성 오류다.
6. replay 부분 실패 후 재개를 허용하고 최종 미완 상태를 보고한다. G025 대표 pair가 준비되면 필수 replay를 우선 수행한다.
7. 학습 완료 직후 중단은 postrun으로 복구한다. 부모 공식 완료 전 fork 금지, 발산 safety pause를 terminal 성공/묵시적 재학습으로 바꾸지 않는다.
8. 날짜·계보·실제 gamma 노출·모든 metric을 P40 전용 탭에 업로드하며 full precision readback과 소수4자리 표시를 분리했다. 한 checkpoint의 H와 다른 checkpoint의 E를 섞지 않는다.
9. 세 서버 집계는 같은 family 이름뿐 아니라 개별 confirmation row의 MSTAR lock SHA까지 확인한다. 중복 서버·다른 clock·다른 lock·미완 pair로 성공을 만들지 않는다.
10. 데이터 signature와 Sheet에 실제 사용되는 `l100/references.py`, `gspread/gspread_upload.py`, `gspread/sheet_categories.py`도 frozen source 해시에 포함하고 회귀 검사한다.

## 로컬 검증

- 독립 실행 수정 후 GFP40 unit/regression **123개 통과**(CPU,26.602초): 원본/구조/83config, 정확 median과 gamma순서, immutable shard/resume, source/parent/reference/anchor, loss routing,20/60/100K endpoint, 로컬 clock/예산/DAG/재개, selector/Sheet mock/replay/서버집계 등.
- 신규 실행 검증: `--server`만으로 최초 로컬 clock 생성, 같은 서버 재개 시 clock 불변, 타 서버 clock 거부, s4 산출물·lock 없이 각 서버 PRIMARY DAG 완주(mock), 서로 다른 세 t0의6-pair 집계 통과, 옛 lock payload·변경된 family/policy 거부. 최초 clock 생성보다 현재 시각을 먼저 읽어 시작을 거부하던 순서 오류를 재현하고 수정했다.
- 변경된 공통 계획 로더의 QG40/G20/L100 기존 regression **34개 통과**.
- 실제 Student/Teacher architecture의 native128·gradient24 CPU 진단 smoke 통과. synthetic fixture이며 실 GF2 전체셋 및 GPU 결과가 아니다.
- CLI help와 shell syntax, Python40파일 AST 확인, 원본 MD/CSV SHA 검증. `work_dir/_gfp40`는 생성하지 않았다. 실제 API credential은 읽거나 출력하지 않았다.

## 실행 전에 남는 검증·명시적 운영 선택

- 실제 s3–s5 RTX5090/runtime, 원 부모/Teacher/reference/data/LP/full q bytes 및 B20 frozen archive는 현장 preflight에서 검증한다. 없는 자산을 추정 생성하거나 다른 Teacher로 대체하지 않는다.
- full19809×ROT4×gamma bank와3072 calibration의 실제 GPU 시간은 아직 측정하지 않았다. 누락 gamma당.75h+미완 family pooling당.20h는 **보수적 engineering 예약 floor**이며 실측에1.15를 곱한 값이 크면 올린다. s5의2h 재사용 가정을 미검증 상태에 적용하지 않는다.
- 부모 equality는 B20의 사전 고정1e-6absolute/0relative를 재사용한다. 이번5090 반복실측 허용폭이라는 주장이 아니며 자동 완화하지 않는다.
- confirmation은 G025 고정이며 서버 간 선택 파일·공통 시각·원격 통신 없이 진행한다. SSH 대상/인증을 추측한 원격 접속·배포는 없다. 결과 파일을 모으는 후속 집계는 학습을 차단하지 않는다.
- 같은 서버의 기존 worker는 종료까지 기다리며 자동 queue drain/강제 종료는 없다. 대기·전환이 자기 서버 t0 이후라면40h에 포함된다. 다른 서버의 완료는 기다리지 않는다.
- 독립 물리 정합 추정기는 검증된 구현이 없어 `ESTIMATOR_NOT_AVAILABLE`이다. A/U replay와 native 품질·perturbation 결과를 그 대체 GT라고 주장하지 않는다.
- 모든 결과는 test-aware development다. 선택적3번째 확인 seed나 미완4pair만으로6pair 성공을 표시하지 않는다.

## 배포 커밋 검증

사용자 커밋 요청에 따라 관련47파일만 선별했다. 사용자의 기존 문서 이동/삭제 및 다른 G20·watchdog·그림 작업은 포함하지 않았다. Git 인덱스의 배포 대상만 별도 임시 디렉터리에 추출해 GFP40 **123개**(25.988초)와 공통 계획 로더 **34개**를 다시 통과했다. 원본 MD/CSV2개와 종속 기존 문서7개 SHA, runtime source105파일, shell syntax도 확인했다. 원본 CSV의 CRLF와 SHA는 보존했다.

현재 상태: **배포 대상 구현·회귀 검증 완료, 미기동, 실제 서버 P0 검증 전**. 이 작업에서는 push·실제 학습·Sheet 쓰기를 수행하지 않았다.
