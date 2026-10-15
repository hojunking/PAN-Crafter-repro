# GF2 L100 LOCAL-T v2 구현·검증 기록

대상: `PANDA_GF2_L100_S345_LOCALT_20H_ExperimentPlan_2026-09-21_v2.md`.
이 작업은 구현·검증·커밋이다. 원격 배포, GPU 학습, 실제 Sheet 쓰기, push는 수행하지 않았다.

## 반영

| 계획 항목 | 구현 |
|---|---|
| s3/s4/s5 local Teacher graph | `l100/plan.py`, Teacher 4개·Student 14개, 교차 서버 참조 거부 |
| ARW/E02 보류 | 4개 registry만 등록, executable config/자동 release 없음 |
| fresh50K/fresh100K | horizon별 scheduler, 50개 grid, exact endpoint와 midpoint 분리 |
| H010 / E1 | 기존 Teacher/BASE/E1 수치 재사용, H010은 alpha만 변경 |
| local reference | endpoint·fullstate·scheduler·sampler·data·source·calibration·q-cache SHA 검증 |
| Teacher→Student 전환 | exact endpoint 검증→local calibration, 49개 FR 평가부채 허용; 성능 gate 없음 |
| 시간·자원 | 공통 t0, 신규16h/optimizer18h/종결20h, whole chain/pair 예약, 재시도 walltime 포함 |
| 재개 | optimizer/RNG/stream/exposure 복구, exact endpoint crash 복구, Teacher 완료 후 calibration 중단 재개 |
| 기존 실행 보호 | 기존 runner/GPU 사용 시 대기, 강제 kill·기존 queue 변경 없음 |
| 배포 | 커밋된 별도 runtime checkout, 로컬 데이터/credential 연결, 주 checkout pull과 분리 |
| 평가 | 기존 GF2 RR20/Q4 및 FR20/native PAN/full512/무마스크 함수 그대로 사용 |
| 저장·업로드 | EXACT_FINAL/VAL/MID50 구분, best HQNR→SCC→ERGAS 별도 유지; 전체50개 검증 후 업로드 |
| Sheet 형식 | 실제 학습 날짜(KST), seed/architecture/input/selection/시간, RMSE/CC/JQM, 표시4자리 |

## 검토 중 예방한 문제

- 고정 50K trainer/reference/postrun을 100K에 재사용하는 잘못된 경로를 분리했다.
- calibration batch64와 온라인 검증 batch16의 수치 차이 재발을 막기 위해 새 reference의
  생성·검증을 batch16으로 통일하고 config/manifest에 명시했다. 수식·IDs·뷰·FP32는 유지한다.
- launcher가 자기 child의 runner lock을 막는 시작 경쟁과 중복 제출을 방지했다.
- trainer의 중단 코드75를 CLI에서0으로 숨기지 않게 했다.
- 18h 이후에도 저장된 후보 평가가 가능하며, optimizer를 다시 실행하지 않는다.
- 정상 시간 소진, 저장 중인 trainer, 무결성 오류, 평가 대기, 업로드 대기, 미실행 block을 구분한다.
- exact endpoint checkpoint만 존재한다고 학습 완료로 간주하지 않는다. full-state와 현재 source/data를 검증한다.
- 재사용 GF2 데이터는 manifest 자기 일치뿐 아니라 공식 로컬 source catalog의 경로·SHA와 비교한다.
- 100K 최종값을 기존 Exact50K 열에 기록하거나, 복구 시각을 학습 종료일로 기록하지 않는다.

## 검증

Python: `/home/knuvi/miniconda3/envs/pancrafter/bin/python`, CUDA 비활성 CPU 검증.

- L100 전체 단위/통합 테스트 **118개 통과**.
- 기존 `g20.test_training`, `g20.test_evaluation`, `g20.test_data`,
  `qg40.test_reference_parity` **27개 통과**.
- 실제 preflight가 호출하는 `method_checks()` **71개 통과**, 실패·오류0.
- 작은 실제 C4 Teacher/H010 Student에서 연속4 update와 2 update→pause→resume의
  모델·AdamW·scheduler·sampler·exposure·corruption RNG가 비트 단위 일치했다.
  Student metric I/O는 해당 smoke에서 synthetic fixture이며, 실제 성능 수치가 아니다.
- 생산 크기 Teacher W112/D123와 Student W104/D122의 CPU forward:
  각각 2,755,574 / 2,198,070 parameters, output `[1,4,64,64]`, finite, A clone hash 동일.
  GPU 속도·메모리·FLOPs 검증으로 해석하지 않는다.
- `build` 재검증: config18, deferred4, registry SHA
  `09e06242ba0512cb3738d53efbd7e775fe337dd860b3bb7509cb35f7c89af88c`.
- s3/s4/s5 dry-run과 shell 문법 확인. 공통 clock 생성·실제 launch 없음.

## 명시적 한계

1. MD가 참조한 원본 CSV/DesignRegistry는 미제공이다. `config/l100/*MD_Derived*`는
   MD의 22개 case를 구현용으로 생성한 자료이며 원본 복구물이 아니다.
2. 확장 P1 gradient/border/원 추정기 분석은 `P1_UNAVAILABLE`로 표시한다. 공통 train128/
   gradient24 IDs·calibration 관계·진단 checkpoint를 보존한다. P0 오류를 우회하지 않는다.
3. 과거 L100 v1 자산의 자동 migration/reuse는 하지 않는다. 기존 clock/Student 참조가
   발견되면 명시적 검토가 필요하며 새 clock으로 초기화하지 않는다.
4. 원격 s3–s5의 실제 GPU/환경/데이터 접근은 아직 이 작업에서 검증하지 않았다.
   시작 시 로컬 P0, Teacher 완료 후 calibration/온라인 q 검증을 수행한다.
5. 계획상 시간은 완료 보장이 아니다. 불충분한 경우 pair/triple 전체를 미입장시키고
   `PARTIAL_QUEUE`, `PAIR_INCOMPLETE`, 평가·업로드 대기를 사실대로 기록한다.

실행 명령과 공통 clock 전달 방법: `l100/README.md`.
기존 dirty G20 수정·과거 로그 이동·그림 변경은 이 커밋에 포함하지 않는다.
