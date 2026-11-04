# s2 QB — 기존 s2 Aligner/Teacher 고정, Stage 2 후속 실험

사용자가 승인한 [설계안](../results_log/2026-11-04_qb-s2-frozen-stage2-plan.md)을 실행한다.
**Stage 1 학습 0회, fresh Student 22회. 다른 서버 weight를 전송하지 않는다.**
기존 `panda_qb_s2` 캠페인과 출력은 읽기만 하며, 새 출력은
`work_dir/panda_qb_s2_frozen/PANDA_S2_QB_FROZEN_S2_20261104_v1/s2/QB`에 기록한다.

## 사용할 모델

둘 다 s2의 `PANDA_S2_QB_STAGE1_BALANCE_20261102_v1`에서 학습한 모델이다.

| 별칭 | 원본 Stage 1 case | 선택 checkpoint | 사용 case |
|---|---|---:|---|
| D4 | QB6_S1_A400_E030_D301 | 48K | Q0–Q5 |
| D2 | QB6_S1_A200_E030_D301 | 48K | Q6–Q7 |

동일 checkpoint의 Aligner와 paired Teacher를 고정한다. 원본 선택 receipt, run/seed,
checkpoint SHA, calibration SHA를 [등록 JSON](../config/panda_qb_s2_frozen/plan.json)에 명시했다.
파일 누락·SHA 불일치·다른 서버 모델이면 중단한다. 새 Stage 1 학습이나 다른 donor로 대체하지 않는다.
선택을 다시 수행하지 않으며, 기존 exact RR-train calibration을 그대로 읽는다.

## 케이스와 순서

| case | donor | 폭 | α | γ |
|---|---|---:|---:|---:|
| Q0_BASE | D4 | 112 | .250 | .500 |
| Q1_H050 | D4 | 112 | .125 | .250 |
| Q2_W128 | D4 | 128 | .250 | .500 |
| Q3_W128H050 | D4 | 128 | .125 | .250 |
| Q4_A0125 | D4 | 112 | .125 | .500 |
| Q5_G025 | D4 | 112 | .250 | .250 |
| Q6_D2_BASE | D2 | 112 | .250 | .500 |
| Q7_D2_W128H050 | D2 | 128 | .125 | .250 |

R1/R2/R3의 Student seed는 261018101/261018102/261018103이다.

- R1: Q3 → Q7 → Q1 → Q2 → Q4 → Q5. Q0/Q6 R1은 기존 결과를 연결한다.
- R2와 R3: 각각 Q0 → Q6 → Q3 → Q7 → Q1 → Q2 → Q4 → Q5.
- 총 24개 비교, 기존 baseline 2개 + 신규 22개. baseline을 새 학습으로 게시하지 않는다.
- [실행 CSV](../config/panda_qb_s2_frozen/cases_s2.csv)와 `catalog`가 실행 순서의 기준이다.

공통: scale1, DIFF, D123/UBIL, β0, LR1e-4, warmup100/cosine30K/minLR0,
batch48=micro12×4, AdamW WD.01. Teacher는 W112/BASE 유지.
입력 `[PANc, PANc−PAN, interp23 I_lrms]` 6채널→QB 4밴드.
A 입력 original Stored LMS4, 복원 base는 기존 msfix LRMS interp23.
Teacher/Student Lanczos-4, RGB 참조, 기존 paired augmentation 그대로다.
회전 loss/TTA/SWA/추가 band loss는 없다. 각 Student는 처음부터 학습한다.

시간 제한은 없다. s2 실측 W112 30K 약 5.5h, W128은 임시 1.3배로 가정하여
전체 약 135–145h다. W128 첫 결과에서 다시 추정한다.

## s2 실행

사용자가 push한 뒤 s2 에이전트가 **전체 커밋**을 받는다. 부분 파일만 복사하지 않는다.
이 코드는 s1에서 구현·검증한 것이며 s2 원격 학습을 이미 시작했다는 뜻은 아니다.

1. s2 저장소에 새 커밋을 pull한다. 실행 중인 컨테이너가 그 checkout을 읽는다면,
   실행 중 코드가 바뀌지 않도록 별도 worktree를 만든다.
2. s2의 기존 `work_dir/panda_qb_s2`와 데이터·credential을 유지한다.
   별도 worktree라면 **같은 s2 디스크의 기존** `work_dir/panda_qb_s2`를 해당 위치에 symlink한다.
   다른 서버에서 파일을 가져오지 않는다. launcher는 이 원본을 `/workspace/work_dir/panda_qb_s2`에 read-only mount한다.
3. `gspread/server.txt`는 `s2`, `gspread/account.json`은 s2의 기존 credential이다.
   QB H5 6개 경로·SHA 및 Docker 환경은 [기존 준비 안내](../panda_qb_s2/README.md)를 따른다.
4. 현재 s2의 다른 학습과 겹치지 않게 아래 명령을 실행한다. `--wait`는 GPU 사용 중이면 기다린다.
   기존 큐를 종료시키는 명령은 포함하지 않았다. 원래 6개 case 전체 완료를 요구하는 조건도 없다.

기존 s2 이미지가 기준 환경과 같다면 그 tag/ID를 `--image`에 사용한다.
새로 만들 경우:

```bash
docker build -f Dockerfile.env -t pancrafter-env:qb-s2 .
python3 -m panda_qb_s2_frozen.enqueue --server s2 --image pancrafter-env:qb-s2 --wait
```

환경은 PyTorch2.4.0/CUDA11.8/cuDNN90100/NumPy1.26.4, FP32, AMP/TF32/benchmark off다.
기존처럼 DLPan 별도 설치 경로가 필요하면 `--dlpan /절대경로/DLPan-Toolbox`를 추가한다.

실행 순서는 코드/환경/데이터·두 donor·baseline 검증 → Sheet read-only 검사 → GPU 대기 →
W112/W128 비용 측정 → trainer와 CPU Sheet 게시기 실행이다.
각 recipe의 첫 학습 전에 실제 s2 CUDA update/save-resume/RR·FR 추론 사전 검사를 수행한다.
학습 전에 source release SHA가 고정된다. manifest 불일치를 s2에서 재생성하여 숨기지 않는다.

## 상태와 재개

컨테이너 이름: `panda-qb-s2-frozen-v1`.

```bash
python3 -m panda_qb_s2_frozen.enqueue --server s2 --status
docker logs --tail 80 panda-qb-s2-frozen-v1
```

중단 시 `docker stop --time 180 panda-qb-s2-frozen-v1`로 checkpoint 저장 시간을 준다.
원인을 해결한 뒤 같은 컨테이너를 재개한다.

```bash
python3 -m panda_qb_s2_frozen.enqueue --server s2 --resume --wait
```

optimizer·RNG·sampler까지 복원하며 완료 case는 건너뛴다. 같은 실행 명령은 중복 컨테이너를 만들지 않는다.
상태는 campaign의 `worker_status.json`, `campaign_state.json`, `publisher.log`,
`sheet_publish_panda_qb_s2_frozen/last_poll.json` 및 `last_error.json`에서 확인한다.
큐 완료 후 마지막 Sheet readback까지 확인한다. WV3 등의 후속 큐는 자동 실행하지 않는다.

## Sheet / weights / 판정

- `QB-ablations`, Server=`s2`, Experiment=`QB S2 Frozen <case> <replica>`.
- Approach에 원본 s2 Stage1 계수·seed·48K, Student 폭/α/γ/LR/batch/seed를 표시한다.
- **같은 checkpoint HQNR≥.9315 AND ERGAS≤3.5**인 모든 step + run별 대표 한 개.
  대표는 HQNR 충족 후보 중 ERGAS 최소, 없으면 HQNR 최고. 중복 step은 합친다.
- 결과/비용 입력은 소수점 3자리. 원본 평가와 notes는 정밀도를 유지하고 판정에도 원본을 쓴다.
- Params, infer, mem, FLOPs는 실제 s2의 해당 배포 구조 A+S 측정값이다.
  infer 단위는 ms/sample, 기본 게시 split은 FR, RR 수치도 notes에 보존한다.
  같은 구조로 초기화한 모델의 비용이며 학습 완료 checkpoint를 직접 측정했다고 표시하지 않는다.
- Train/Eval/Wall은 해당 선택 step까지의 실측 시간을 h로 기록한다. Stage1 비용은 새 학습 시간에 합산하지 않는다.
- 모든 1K 평가 weight와 완전 재개 상태를 보존한다. 통과 weight도 같은 경로에 남는다.
- HQNR@8K/9K/10K가 모두 .90 미만이면 기존 조기 중단 규칙을 적용한다.
- 고정 30K·25–30K 추세·seed별 차이도 분석하며 최고 대표 하나만으로 효과를 확정하지 않는다.

## 개발 검증

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s panda_qb_s2_frozen/tests -v
python -m panda_qb_s2_frozen.release
```

synthetic CPU 검사와 실제 s2 장시간 GPU 실험은 구별한다. 원격 파일 존재·환경·CUDA 동작은
s2 launcher/preflight가 검사하며, 이 서버에서 대신 확인했다고 기록하지 않는다.
