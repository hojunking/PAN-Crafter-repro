# s2 QB Stage 1 balance — 실행·인계

승인된 [설계](../results_log/2026-11-02_qb-s2-30h-plan.md)를 실행하는 독립 패키지다.
이 커밋 전체를 s2에 전달한다. 기존 s1/s3 큐와 donor 파일을 필요로 하지 않는다.
이 서버에서 검증한 범위는 [구현 기록](../results_log/2026-11-02_qb-s2-implementation-and-handoff.md)에 있다.
실행 등록은 완료됐으며, **s2 실제 학습은 아래 launcher를 실행할 때 시작한다.**

**실행할 6개 case**

| 순서 | case 식별자 | λ_abs | λ_ε | Stage 1 seed | Stage 1 → Student |
|---:|---|---:|---:|---:|---|
| 1 | A200_E030_D101 | 2 | .03 | 261016101 | 50K → 30K |
| 2 | A400_E030_D101 | 4 | .03 | 261016101 | 50K → 30K |
| 3 | A200_E030_D301 | 2 | .03 | 261016301 | 50K → 30K |
| 4 | A400_E030_D301 | 4 | .03 | 261016301 | 50K → 30K |
| 5 | A300_E030_D101 | 3 | .03 | 261016101 | 50K → 30K |
| 6 | A400_E060_D101 | 4 | .06 | 261016101 | 50K → 30K |

각 Stage 1 직후 그 donor의 Student를 실행한다. 총 12개 job이며
[등록 JSON](../config/panda_qb_s2/plan.json), [전체 CSV](../config/panda_qb_s2/cases_s2.csv)가 실행 명세다.
모든 Student seed는 261018101이다. 모든 모델은 s2에서 처음부터 학습한다.
s1 실측 기준 약 **34–36시간**, 시간 제한은 없다. 설치·데이터 전달 시간은 별도다.
큐가 끝나면 종료하며 WV3로 연결하지 않는다.

공통 Student: W112/D123, UBIL, `[PANc, PANc−PAN, Ilrms]` 6채널→4밴드,
Aligner scale=1, LR=1e-4, warmup100/cosine30K/minLR0, batch48=micro12×4,
AdamW WD=.01, α=.25/γ=.5/β=0. Stage 1은 Aligner LR=1e-5, Teacher LR=1e-4,
warmup100/cosine50K다. 두 단계 모두 Lanczos-4와 RGB 구조 참조를 쓴다.
Aligner는 original Stored LMS 4밴드 전체를 입력받고, 복원 base는 msfix LRMS의 interp23이다.
합성 ε는 반경3px disk이며 native 예측 shift를 자르지 않는다.
기존 augmentation을 유지하며 회전 일관성 loss·TTA·SWA는 추가하지 않는다.

**구버전 s2 준비**

사용자가 push한 뒤 **전달받은 전체 커밋**을 받는다. 기존 로컬 작업이 있는 경우 새 worktree를 권한다.
기존 저장소에서 `git fetch origin` 후
`git worktree add --detach ../PAN-Crafter-qb-s2 <전달받은_커밋_SHA>`를 실행하고 새 경로로 이동한다.
일부 새 파일만 과거 버전 위에 복사하지 않는다.

필요한 로컬 파일은 다음과 같다. 데이터와 credential은 Git에 포함하지 않는다.

- `gspread/server.txt`: 내용 `s2` 한 줄. `gspread/account.json`: 기존 게시용 service-account credential.
- [데이터 catalog](../ablr2/sensor_sources.json)의 QB train/val/rr/fr H5 4개와
  [original LMS override](../config/bglobal_rr/qb_original_lms.json)의 train/val H5 2개.
  각 JSON의 `path`에 실제 파일 또는 **그 파일의 절대 실제 경로를 가리키는 파일 symlink**를 둔다.
  디렉터리 전체를 외부 경로로 symlink하면 컨테이너의 중간 경로가 누락될 수 있다.
- 출력은 `work_dir/panda_qb_s2` 아래에 새로 저장한다. 평가 weights만 약 17GiB이고,
  resume·calibration·로그를 포함할 여유 공간이 추가로 필요하다.

데이터 내용 SHA와 split 수(train17139/val1905/RR20/FR20)를 시작 전에 검사한다.
QB DN peak는 2047이다. msfix 파일, original LMS 또는 다른 split을 서로 대체하지 않는다.

호스트에는 Python3, Docker, NVIDIA Container Toolkit과 Docker 사용 권한이 필요하다.
호스트 Python에는 학습 라이브러리를 설치하지 않아도 된다. 실행 환경은 다음처럼 준비한다.

```bash
docker build -f Dockerfile.env -t pancrafter-env:qb-s2 .
python3 -m panda_qb_s2.enqueue --server s2 --image pancrafter-env:qb-s2 --wait
```

기준은 PyTorch2.4.0/CUDA11.8/cuDNN90100/NumPy1.26.4, FP32 및 AMP/TF32/benchmark off다.
기존 이미지가 같은 환경이면 그 로컬 image tag/ID를 사용할 수 있다.
DLPan은 Dockerfile에 고정된 버전이 포함된다. 별도 동일 checkout을 쓰려면
`--dlpan /절대경로/DLPan-Toolbox`를 추가한다. 평가 코드와 metric-library 버전도 검사한다.

launcher는 코드·환경·데이터 검사 → Sheet 읽기 검사 → GPU 유휴 대기 → 비용 측정 → 학습 순서로 실행한다.
`--wait`는 다른 GPU 작업이 끝날 때까지 기다린다. 실행 container는
`panda-qb-s2-balance-v1`이고, trainer와 CPU 전용 Sheet 게시기가 함께 실행된다.
각 단계의 실제 데이터 update·save/resume·RR/FR 추론 검사는 그 단계 학습 전에 수행한다.
저장된 이미지 ID와 source hash가 실행 기록에 남는다.

[release.json](../config/panda_qb_s2/release.json)은 전달 코드의 내용 검증용이다.
불일치하면 커밋·환경·파일 누락을 바로잡는다. s2에서 manifest를 재생성하여 차이를 숨기지 않는다.
다른 서버의 종료나 별도 사람 승인을 기다리는 조건은 없다.
`.lock` 파일은 같은 출력에 launcher/게시기를 두 번 띄우지 않기 위한 로컬 중복 실행 방지용이다.

**상태·재개**

```bash
python3 -m panda_qb_s2.enqueue --server s2 --status
docker logs --tail 80 panda-qb-s2-balance-v1
```

전체 상태: `work_dir/panda_qb_s2/PANDA_S2_QB_STAGE1_BALANCE_20261102_v1/s2/QB/worker_status.json`.
같은 경로의 `campaign_state.json`은 각 job 상태를, `publisher.log`와
`sheet_publish_panda_qb_s2/last_poll.json`은 게시 상태를 기록한다.
같은 실행 명령을 다시 주면 실행 중인 컨테이너를 중복 생성하지 않는다.

중단이 필요할 때 `docker stop --time 180 panda-qb-s2-balance-v1`로 저장 시간을 준다.
중단·실패 원인을 확인하고 해소한 뒤 다음 명령으로 같은 컨테이너를 재개한다.

```bash
python3 -m panda_qb_s2.enqueue --server s2 --resume --wait
```

모델·optimizer·RNG·sampler를 복원하며 완료한 Stage 1을 다시 학습하지 않는다.
컨테이너나 출력 디렉터리를 삭제해서 재시작하지 않는다.
알려진 support 실패는 근거를 보존하고 해당 donor의 Student만 보류한다.
그 외 오류는 진단할 수 있도록 중단한다. 부분 학습 donor로 대체하지 않는다.

**선택·게시·저장 규칙**

- Stage 1 full50K 이후 validation RRE50→RRE90→ValERGAS→이른 step으로 donor를 선택한다.
  Teacher HQNR/test 성능으로 선택하지 않는다.
- Student의 8K·9K·10K HQNR이 모두 .90 미만일 때만 기존 조기 중단을 적용한다.
- 목표는 같은 checkpoint의 **HQNR≥.9315 AND ERGAS≤3.5**다.
- `QB-ablations`에 server=`s2`로 모든 동시 통과 checkpoint와 run별 대표 하나를 게시한다.
  대표는 HQNR 충족 후보 중 ERGAS 최저, 없으면 HQNR 최고이며 중복은 합친다.
  Stage 1 행에는 선택 RRE 정보를, Approach에는 계수·두 단계 seed·학습 설정을 표시한다.
- 모든 1K 공식 평가 weights와 완전 재개 상태를 저장한다. 목표 달성 여부와 무관하다.
- 전체 완료는 마지막 Sheet readback까지 확인한 뒤 기록한다. 실패·조기 중단·보류 개수도 함께 남긴다.

개발 검증은 학습 환경에서 다음으로 실행한다.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s panda_qb_s2/tests -v
```
