# ABLR2 s1/WV3 · s2/QB 구현 검증

대상: `PANDA_ABL_S1WV3_S2QB_AdaptiveRepeat_Bundle_2026-09-21_v2`.
상태: 별도 `ablr2/` 구현. 아래 본문은 최초 구현 검증 시점의 기록이며, 후속 Docker 배포 검증은 마지막 절에 추가했다.

## 반영 범위

- 제공 번들 7개 파일의 SHA256SUMS 검증 통과. CSV의 190개 BOOT5 run ID, 순서, seed, Teacher 참조를 유지한다. s1/WV3와 s2/QB만 허용한다.
- 17개 component의 C+3 입력 mask, identity/frozen A, 독립 U/A gradient, Teacher-free/C03/C04 의존성, train-view 평균 C15를 구현했다. 기존 G20/L100/FH12 수치·실행 경로는 수정하지 않았다.
- 새 계획의 VAL-ERGAS 최소 checkpoint와 Exact50K를 주 분석으로 분리하고, HQNR best/RAW 진단은 별도 보존한다. 기존 캠페인의 checkpoint 기준은 그대로다.
- BOOT5 → 센서별 threshold freeze → RECHECK5 → 제한된 recipe FIT → 전체 REFRESH5를 연결했다. VERIFY5는 운영자 요청, 동일 recipe당 1회이며 실패 결과도 남긴다.
- 동일 fitting 작업은 검증된 source/data/reference/수치 설정 fingerprint가 같은 경우만 실제 관측을 재사용한다. 새 seed·독립 관측으로 세지 않는다.
- 72h 명시적 갱신 lease, fullstate 재개, 네 가지 안전 중지, 일시적 오류 최대 2회 재시도, 업로드만 재시도, 완료된 panel·recheck·탐색 최고값·VERIFY의 분리된 CSV/JSON/그림을 추가했다.
- 기본 시작은 커밋된 고정 worktree에서 실행된다. 진행 중 원본 checkout의 pull이 수치 코드를 바꾸지 않는다. 다른 캠페인의 process를 종료하거나 결과를 삭제하지 않는다.

## 직접 확인한 내용

CPU 테스트는 `python -m unittest discover -s ablr2 -t . -p 'test_*.py' -q`로 실행하여 **126개 모두 통과**했다. 모델/gradient/RNG/seed/정책, immutable config, reference, checkpoint/evaluation, Sheet mock, lease·crash·동일 상태 재개, report/plot, 배포 mock을 포함한다. Python 파일 36개의 AST/공백 검사와 launcher의 `bash -n`도 통과했다.

실제 W112 D123 Teacher와 W104 D122 Student의 64×64 PAN / 16×16 MS 입력을 사용하여 두 센서의 TPLUS/C00/C07/C16 총 8종 CPU forward를 확인했다.

| 센서 | Teacher params | FULL Student params | no-A Student params | Teacher/Student 입력 채널 |
|---|---:|---:|---:|---:|
| WV3 | 2,764,218 | 2,206,138 | 2,100,808 | 9 / 11 |
| QB | 2,755,574 | 2,198,070 | 2,093,316 | 5 / 7 |

WV3/QB 각 20장 합성 DN fixture에서 기존 RR 평가기와 ERGAS/SCC/PSNR/SAM/SSIM/Q8·Q4를 직접 대조했으며 차이는 모두 0이었다. 이 검사는 실제 논문 데이터에서 MATLAB과 bitwise 동일하다는 주장이 아니다.

로컬 원본 H5의 장수·geometry도 읽기 전용으로 확인했다.

| 센서 | train | val | RR | FR | bands / maxDN |
|---|---:|---:|---:|---:|---|
| WV3 | 9,714 | 1,080 | 20 | 20 | 8 / 2047 |
| QB | 17,139 | 1,905 | 20 | 20 | 4 / 2047 |

WV3 4개 H5 SHA를 실제 로컬 파일에서 측정하여 catalog에 고정했다. QB는 기존 pinned msfix/native RR/native MAT20 catalog를 사용한다. 원격 s2 파일의 동일성까지 확인한 것은 아니다.

## 검토 중 수정한 실제 결함

1. JSON으로 고정한 config의 `1e-08`을 YAML 1.1이 문자열로 읽는 문제: JSON-first loader.
2. readonly config를 trainer가 다시 쓰던 문제: 동일 내용 확인 후 재사용.
3. endpoint 복구 함수 연결과 technical retry의 `--resume` 누락.
4. 외부 dataset manifest를 자기 신고 SHA만으로 신뢰하던 문제: 자체 receipt가 없으면 full LP 및 QB raw/msfix 검사.
5. 중지 기록 timestamp 중복, 최초 seed ledger 발행 중 crash 복구, VERIFY가 DEV 비교 증거를 덮어쓰던 문제.
6. VERIFY 전환 중 crash 후 다음 stage가 먼저 시작되던 문제와 확인용 잠긴 recipe/DEV recipe 복귀 처리.
7. 표시 자릿수와 원본 수치 혼동: Sheet는 원본 값을 유지하고 표시 형식을 적용하며, 재시도 비교는 unformatted 값을 읽는다.

## 운영 해석과 남은 실행 검증

- `CHECKPOINT_SENSITIVE`의 “큰 반대 효과”는 같은 frozen threshold로 Exact50K가 역방향 개선 판정을 얻는 경우로 구현했다. 별도 임의 허용폭을 만들지 않았다.
- VERIFY의 운영 결과는 기존 FLOW 조건 충족 여부를 `VERIFY_SUPPORTED_DEV`/`VERIFY_NEGATIVE`로 보존한다. 통계적 유의성·독립 test 성능을 뜻하지 않는다.
- 첫 자원 예약은 run당 8h, Teacher calibration 3h의 미측정 보수적 값에 ×1.15를 적용한다. 이후 해당 서버의 실제 train/eval/save/retry/calibration 비용으로 갱신한다. 72h 안에 BOOT5가 끝난다는 의미가 아니다.
- Teacher consistency의 `t`는 기존 구현과 동일한 0-based update index로 유지한다. 별도 corruption RNG가 native sample/view 순서를 바꾸지 않는다.
- 실제 GPU batch48 학습, 전체 train3072 calibration·전체 train×4 q-cache, 모든 원본 LP/MSfix 전수 검증, 실제 Sheet 접근 권한과 원격 s1/s2 기동은 아직 실행하지 않았다. 해당 검사는 명시적인 시작 후 각 서버의 preflight/학습에서 수행한다.
- 실제 FR은 native PAN/original LMS, full512, RR은 원본 20장·DN2047·20:-21·Q32다. JQM은 SRF-substitute variant라고 명시하며 SIPSA와 완전 동일하다고 표시하지 않는다.

실행 명령과 운영 절차: [ablr2/README.md](../ablr2/README.md).

## 후속 Docker 배포 검증

사용자의 커밋·s1 Docker 기동 요청에 따라 `tools/ablr2_docker_start.sh`를 추가했다. 로컬 이미지를 SHA256 ID로 고정하고, 커밋 고정 worktree에서 foreground runner를 실행한다. 소스·데이터·DLPan·credential은 읽기 전용이며 `work_dir`만 쓰기 가능한 bind이다. 동일 lane 중복 기동을 잠금으로 방지하고 기존 컨테이너/실험을 종료하지 않는다. lease는 명시적으로 최대 72h이며 자동 갱신하지 않는다.

- 이미지: `hojunqueen/pancrafter-env:torch2.4.0-cu118`, 실제 ID `sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887`.
- 해당 이미지에서 Docker 실행기 테스트 5개를 포함한 **131개 CPU 회귀 테스트 통과**.
- s1 RTX 4090에서 PyTorch 2.4.0 / CUDA 11.8의 작은 CUDA 행렬 연산 정상 확인. 이는 실제 batch48 학습이나 전체 calibration 완료 증거와 구분한다.
- 커밋 이후 실제 시작 결과는 `work_dir/ablr2/WV3/s1/`의 등록·preflight·학습 로그에 기록한다. s2는 사용자 전달 후 로컬 데이터/환경 검증을 거쳐 별도로 시작한다.
