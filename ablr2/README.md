# ABLR2 — 기존 2026-09-21 실행 안내 (보존본)

**2026-09-23 ABLR2X 확장 실행은 [ABLR2X 실행 안내](ABLR2X_README.md)를 사용한다.**
아래 2개 lane·17-component·72h 설명은 원 실행을 복구할 때 참고하는 이전 계약이다.
현재 확장은 s1/WV3·s2/QB 유지 + s3/GF2·C17 추가이며, 별도 명시 승인으로 `UNTIL_OPERATOR_STOP`을 사용한다.

2026-09-21 v2 번들의 별도 component-ablation 캠페인이다. s1은 WV3, s2는 QB만 허용하며 s3–s5 GF2, 기존 생산 checkpoint·결과·queue는 변경하지 않는다. 학습은 운영자가 아래 시작 명령을 실행할 때만 기동한다.

## 배포와 최초 시작

구현을 검증한 뒤 **commit → push → 각 서버 pull**이 먼저 필요하다. 기본 런처는 커밋되지 않은 실행 소스를 거부하고, 현재 커밋의 별도 detached worktree를 만들어 실행한다. 이후 원본 checkout에서 pull해도 진행 중인 실험 소스는 바뀌지 않는다.

각 서버의 CUDA/PyTorch 환경을 활성화한 후 저장소 루트에서 실행한다. 환경의 Python이 다르면 `PYTHON=/정확한/python/경로`를 앞에 지정한다.

```bash
# s1에서: WV3 전용
bash tools/ablr2_start.sh --server s1 --lease-hours 72

# s2에서: QB 전용
bash tools/ablr2_start.sh --server s2 --lease-hours 72
```

이 명령의 `--lease-hours 72`가 최초 자원 사용 승인이다. `build`나 pull만으로 lease 생성·학습 기동이 되지 않는다. 두 서버는 독립적으로 진행한다. 기존 작업/GPU가 사용 중이면 기다리며, 다른 실험을 종료하지 않는다. 데이터와 LP·QB msfix 검증, 수치 회귀, 로컬 GPU forward 확인을 통과한 후 학습한다.

실제 `start`의 기본값은 공식 평가 후 Sheet 업로드 활성화다. 업로드를 원하지 않으면 시작 명령에 `--no-upload`를 추가한다. Google 서비스 계정·Sheet 접근 권한과 센서별 고정 데이터/DLPan 의존성이 각 서버에 있어야 한다. 현재 구현 작업에서는 Sheet를 쓰지 않는다.

## Docker 시작

Docker/NVIDIA Container Toolkit과 로컬 이미지가 준비된 서버에서는 다음 명령을 쓴다. `PYTHON`은 **호스트**의 source 검증용 환경이며, 실제 학습은 컨테이너 이미지의 Python/CUDA로 실행한다.

```bash
# s1, 원본 checkout에서 실행 (s2에서는 --server s2)
PYTHON=/home/knuvi/miniconda3/envs/pancrafter/bin/python \
bash tools/ablr2_docker_start.sh --server s1 --lease-hours 72

# 읽기 전용 명령 미리보기; worktree/lease/container를 생성하지 않음
PYTHON=/home/knuvi/miniconda3/envs/pancrafter/bin/python \
bash tools/ablr2_docker_start.sh --server s1 --dry-run
```

기본 로컬 image tag는 `hojunqueen/pancrafter-env:torch2.4.0-cu118`이며 `--image`로 명시 변경할 수 있다. 자동 pull하지 않고, 시작 시 실제 image SHA256 ID로 고정한다. `--gpu 0`이 기본값이며 단일 GPU index/UUID를 지정한다. 같은 release/image의 lane이 이미 실행 중이면 중복 실행이나 lease 갱신 없이 반환한다.

커밋 고정 worktree와 원본 소스, 데이터 symlink의 실제 대상, DLPan, credential은 읽기 전용으로 연결하고 `work_dir`만 쓰기 가능하게 연결한다. host UID/GID, shared memory 8GB, host PID namespace를 사용한다. PID 공유는 기존 학습을 감지하기 위한 것으로 privileged 모드나 Docker socket을 제공하지 않는다. 이미지 자체의 설치 코드를 평가에 대신 사용하지 않도록 호스트 DLPan 경로를 명시 전달한다.

출력의 `docker logs --tail 80 <container>`로 기동/학습 로그를 확인한다. 컨테이너 자동 재시작·lease 자동 갱신·종료 컨테이너 삭제는 하지 않는다. 만료/안전 중지 후에는 `CONTINUE`가 필요한 경우 먼저 설정하고 **같은 Docker 시작 명령**을 다시 실행해 기존 고정 runtime·state를 재개한다. 기존 종료 컨테이너는 로그 증거로 보존되며 새 실행 이름이 부여된다. 다른 서버에서는 Python 경로와 이미지 존재 여부를 해당 서버 환경에 맞춘다.

## 실행량과 자동 반복

- BOOT5: 센서마다 5 sweep × (TPLUS/TZERO 2개 + Student C00–C16 17개) = **95개 학습**. 두 센서 합계 **190개 학습 + calibration 20개**이며, 모두 fresh50K이다.
- BOOT5 전체 완료 후 센서별 운영 임계값을 한 번 고정한다. 불완전한 panel이나 좋은 일부 결과로 임계값을 만들지 않는다.
- 이후 등록된 관계의 paired RECHECK5, 최대 2개 후보의 FIT_ROUND, 전체 17행의 REFRESH5를 진행한다. 관계×recipe당 재시험은 1 batch이며, R00–R05 밖의 recipe/optional F·X/100K를 자동 추가하지 않는다.
- 성능이 낮아도 유효한 관측으로 보존한다. 좋은 첫 seed에서 batch를 중단하거나 불리한 행을 교체하지 않는다. 확인용 VERIFY5는 운영자가 요청하며, 동일한 정확한 recipe에서 실패한 확인 wave를 새 seed로 교체하지 않는다.

**72시간은 갱신형 lease이지 총 실험 시간이나 BOOT5 완료 보장이 아니다.** 자동 갱신하지 않으며, 만료/다음 작업 예약 부족 시 안전하게 멈춘다. 갱신해도 seed·순서·누적 사용시간은 초기화하지 않는다. 최초 예약값은 학습/평가/저장 8시간/run, Teacher calibration 3시간의 **미측정 보수적 추정**이다. 이후 해당 서버의 실제 비용과 ×1.15 안전계수를 사용한다. 디스크 부족 시 입장을 멈추며 자동 삭제하지 않는다.

## 상태·중지·재개

상태/로그 위치는 `work_dir/ablr2/WV3/s1/` 또는 `work_dir/ablr2/QB/s2/`이다. `runtime_release.json`의 `path`가 고정 실행 checkout이다. 아래 관리 명령은 그 checkout에서 실행한다(`--server s2`로 QB에 동일 적용).

```bash
python tools/ablr2_runner.py status --server s1
python tools/ablr2_runner.py stop --server s1 --command STOP_NOW_SAFE
python tools/ablr2_runner.py stop --server s1 --command CONTINUE
python tools/ablr2_runner.py lease --server s1 --hours 72
```

`STOP_NOW_SAFE`는 완료된 optimizer update/안전 경계에서 fullstate를 저장한다. `STOP_AFTER_RUN`, `STOP_AFTER_SWEEP`, `PAUSE_AFTER_BLOCK`도 지원한다. 임의 kill이나 다른 campaign 종료는 하지 않는다. 재개는 중지 제어를 `CONTINUE`로 바꾸고, **원본 checkout에서 최초 시작 명령을 다시 실행**하면 기존 고정 runtime·상태를 이어 쓴다. 실행 중인 runner는 중복 생성하지 않는다.

```bash
# 고정 실행 checkout에서: BOOT5/임계값이 완성된 이후 확인 wave 요청
python tools/ablr2_runner.py verify --server s1

# 이미 평가된 결과만 업로드 재시도; 학습은 다시 하지 않음
python tools/ablr2_runner.py retry-uploads --server s1
```

일시적 I/O 오류는 동일 config/seed/fullstate로 최대 2회 자동 복구한다. NaN/발산은 별도 실패로 남기며 다른 seed로 대체하지 않는다. 업로드 실패는 평가 결과를 보존한 채 업로드만 재시도한다. source/data/reference가 달라지면 중단하며, 기존 runtime 소스를 덮어써서 재개하지 않는다.

## 수치·평가 계약

- Student는 WV3 11/QB 7개의 고정 입력 slot을 쓰며, P0 ablation의 L/H는 학습·추론 모두 0이다. identity A는 warp를 우회하고, frozen A는 보정만 유지하며 optimizer/decay/buffer 갱신에서 제외한다.
- C00/C01/C02/C10은 Teacher I/O가 없다. C03/C04는 endpoint만 사용하고 calibration/q를 읽지 않는다. 나머지 reference는 해당 센서·서버·Teacher seed·exact50K·data/LP/source SHA에 묶인다. C15의 상수는 실제 전체 train×4-view 평균이다.
- 이번 **새 계획**의 주 선택은 고정 50-candidate 중 validation ERGAS 최소(동률은 이른 step), 보조 주 분석은 Exact50K이다. HQNR best는 별도 보존한다. 기존 campaign의 HQNR 선택 규칙은 수정하지 않는다.
- 공식 RR/FR 모두 20장이다. RR은 DN2047, `20:-21` support와 block32의 WV3 Q8/QB Q4를 사용한다. FR은 원 native PAN/original LMS, 전체 512 support, masking/shift된 reference 없이 센서별 MTF와 장면별 HQNR 평균을 사용한다.
- RMSE·CC 등 공식 결과와 JQM을 기록하되 JQM은 **SRF substitute 보조 variant**임을 표시한다. 동일한 FR20을 반복 개발에 쓰므로 `TEST_AWARE_DEV`이며 새 학습 seed가 독립 test set을 뜻하지 않는다. VAL/Exact/RAW 및 전체 panel/targeted recheck/탐색 최고값을 혼합하지 않는다.

## 검증과 한계

```bash
PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
python -m unittest discover -s ablr2 -t . -p 'test_*.py' -q
```

CPU 회귀는 C4/C8 기존 FULL parity, component/gradient/RNG 계약, selector·반복 정책, 데이터 receipt, 안전 정지·동일 fullstate 재개 등을 검증한다. 작은 합성 데이터나 mock을 사용한 테스트는 실제 GPU/3072-patch calibration·전체 source 검증 완료 증거가 아니다. 실제 환경 검증은 각 서버의 preflight와 학습 때 수행한다. 현재 구현만으로 원격 서버의 기동·속도·성능·단조 개선을 보장하지 않는다.

원 계획: [실험 계획](../research_log/PANDA_ABL_S1WV3_S2QB_AdaptiveRepeat_Bundle_2026-09-21_v2/PANDA_ABL_S1_WV3_S2_QB_AdaptiveRepeat_ExperimentPlan_2026-09-21_v2.md)
