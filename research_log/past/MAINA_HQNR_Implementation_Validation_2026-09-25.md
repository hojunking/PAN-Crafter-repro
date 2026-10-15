# MAIN-A WV3 s4/s5 구현 및 검증 — 2026-09-25

대상: `PANDA_WV3_TableA_S45_HQNR_Continuous_ExperimentPlan_2026-09-25.md`.

## 구현 범위

신규 `maina_hqnr/` 패키지와 `tools/maina_hqnr_*` 진입점으로 추가했다.
기존 FH12/FH20R1 수치 코드, G23 캠페인, s1–s3 실행 코드는 수정하지 않았다.
계획서에 언급된 companion CSV/JSON은 제공되지 않아, MD에서 도출한 파일을
`maina_hqnr/spec/`에 별도 생성했으며 제공된 원본이라고 표기하지 않았다.

| 요구사항 | 구현 |
| --- | --- |
| PLH/W104/D122 + 원 F1, 7조건 | 원 BASE config·모델·loss 경로 재사용, α/β/λE만 변경 |
| 고정 F1·calibration·q·데이터/LP | 원 SHA 검증, 경로 전용 relocation binding, 재생성·F4/F5 대체 금지 |
| 독립 무기한 반복 | s4/s5 독립 cursor·seed·회전 순서, cycle별 초기 U/A·50K stream 공유 |
| HQNR_MAX50 | 정해진 50개 A/U 저장본, native FR20 raw PAN 최대 HQNR, 동률이면 이른 step |
| EXACT_50000 | 별도 보조 선택, 동일 SHA는 동일 평가 alias |
| 완료 판정 | 실제50K full-state·모든 후보·전체20 scene·유한 지표·SHA·동일 protocol 필수 |
| 재개·실패 | optimizer/scheduler/RNG/sampler 보존, BASE 실패 중지, variant 기술 실패 최대2회 자동 재시도 |
| 평가/업로드 복구 | 평가 실패는 후보 재사용, API 실패는 outbox 재시도; 재학습하지 않음 |
| 기존 G23 안전 전환 | 실제 PID/cwd/mount/GPU 확인, 지원 stop marker, update/run 경계 보존, 강제 kill 없음 |
| Sheets | 전용 s4/s5 탭, 12개 metric, 실제 날짜/시간·선택 step·provenance, 원 정밀도+4자리 표시 |
| 집계 | 동일 서버/cycle/seed BASE paired delta, balanced 축별 seed 집합, s4 cycle0 anchor 별도 |

## 검토 과정에서 수정한 사항

- CLI `--server`와 config 서버가 다르면 실행 전에 거절한다.
- production admission은 config·source·binding·실제 GPU·preflight가 모두 일치해야 한다.
- `assets`는 읽기 전용이며, production binding은 frozen runtime에서만 생성한다.
- 첫 optimizer update 전에는 `INITIALIZING`/`RESUMING`으로 표시한다.
- 50K 표시만 있는 잘못된 checkpoint를 차단하도록 모델 tensor, AdamW 전체 moments,
  scheduler, Python/NumPy/Torch/CUDA RNG, 원 sampler 재생, 초기화·stream ledger를 대조한다.
- s4/s5가 공유 시트의 오래된 행 번호로 서로의 결과를 덮는 경쟁을 제거했다.
  전용 raw 탭은 각 서버가 작성하고, collection은 **s4만 초기화하는 정렬된 live projection**이다.
  s5는 projection에 쓰지 않으며, collection 준비 전에도 학습·raw 업로드는 진행한다.
  formula/readback 실패는 outbox 보류이며 기존 사용자 영역을 덮어쓰지 않는다.
- 중간 source 변경, 다른 Docker image/GPU, 모호한 기존 owner, 손상된 자산은 묵시적으로 대체하지 않는다.
- 원 CUDA RNG 저장본이 uint8[16]임을 실제 F1 full-state에서 확인했다. CPU RNG 크기를
  CUDA에도 요구하면 정상 저장본을 거절하는 오류를 수정하고 정상·손상 상태 회귀 테스트를 추가했다.

## 실제 수행한 검증

원 F1 checkpoint·bridge·calibration·q-cache·10개 참조 artifact와 WV3 train/val/RR/FR 및
네 LP cache의 SHA를 실제 로컬 파일에서 확인했다. 원 실행 코드 42개 파일의 내용 SHA는
`79a7e4a154547b355e94e4198a5d85a5a4ecf220ebc039da1b3a3404a91bbc89`와 일치했다.

원 Docker image `sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887`에서
네트워크 차단, GPU 미할당, repository/data 전부 read-only mount로 **CPU 검증**을 실행했다.
`verify_assets('s4', persist=False)`와 `run_acceptance(ROOT, 's4', bindings, device='cpu')` 결과:

- 원 BASE 대비 frontend·warp·output·loss·U/A gradient **119개 비교, 최대 절대 오차 0**.
- 7조건의 fresh U/A와 전체50K sample/view stream SHA가 동일.
- 실제 W104D122, batch48, worker4의 두 update 및 update1 full-state 복원 후 재실행:
  **모델 최대 절대 오차 0**, optimizer/RNG/sampler 일치.
- Teacher 변경 없음. production 상태 기록 없음.

s4 cycle0의 실제 공통 hash:

```text
U      5137e50877893b3090edc33de5174f854d8983e1bef21554dce6e2c2febdc408
A      1e01b69360dbd7526b77e55718050fad405ade38eb59f56c215ae07edabc65cd
stream 9d6796c2d9c6cb71ff85ffa8b8130fccf4d53271a8db1f5e13c01235a74fd01c
```

CPU acceptance는 의도적으로 `status=CPU_TEST_ONLY`, `passed=false`이다.
이는 수치 검증 실패가 아니라 **production CUDA admission으로 사용하지 않도록 한 구분**이다.
각 대상 GPU에서 실행하는 실제 acceptance는 launcher에 연결되어 있다.

최종 신규 테스트 **100개**가 host 환경과 원 pinned Docker 이미지 양쪽에서 모두 통과했다.
원 FH12/FH20R1 회귀 테스트도 **45개** 통과했다:

```bash
python -m unittest tools.fh20r1_training_tests tools.fh20r1_diagnostics_tests tools.fh12_training_tests tools.fh12_evaluation_tests -q
python -m unittest discover -s maina_hqnr -p 'test_*.py' -q
```

신규 테스트는 seed/case/config, 자산 변경, loss gradient routing, sampler·resume, controller
retry, safe cutover, Docker/dry-run 보호, 후보50개 선택, 불완전 결과 차단, paired 통계,
Sheets 모의 API 및 정밀도/readback/동시 작성 방지 경로를 검증한다.
강화된 50K 완료 검사는 임시 synthetic fixture의 실제 AdamW/safetensors/50-candidate ledger와
원 BatchStream을 사용해 정상 저장 스키마 통과 및 손상 거절을 확인했다.
이 fixture의 모의 update counter는 실제50K 학습 증거로 사용하지 않았다.
공유 조회 수식도 모의 API로 검증했으며 실제 Google Sheets 계산·readback은 아직 수행하지 않았다.

## 실행 전달 및 미실행 범위

실행 방법과 자산 경로 이전은 `maina_hqnr/README.md`에 정리했다. push/pull 후 각 서버에서:

```bash
bash tools/maina_hqnr_docker_start.sh --server s4 --gpu 0
bash tools/maina_hqnr_docker_start.sh --server s5 --gpu 0
```

해당 서버의 명령 하나만 실행한다. 필요한 자산이 있으면 안전 전환→GPU acceptance→BASE 시작이
자동으로 이어진다. **Git은 F1 checkpoint/H5/LP 캐시를 전송하지 않는다**. 누락 시에는
원본 복사와 필요 시 `--asset-map`이 필요하며, 다른 Teacher로 대신 실행하지 않는다.

이 구현 작업에서 본선 학습, 실제 s4/s5 G23 중단, 원격 서버 명령, 실제 Sheets 쓰기,
CUDA acceptance, 최종50K 성능 재현은 수행하지 않았다. s1 진행 중 학습도 변경하지 않았다.
네이티브 데이터 동일성만 확인했으며 논문별 scene 대응은
`PAPERSET_IDENTITY_UNVERIFIED`, FR20 checkpoint 선택은 `test_aware=true`로 유지한다.
