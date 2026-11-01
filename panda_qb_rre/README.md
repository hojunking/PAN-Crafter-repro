# QB RRE donor 우선 큐 — 2026-11-01

캠페인: `PANDA_S13_QB_RRE_FOCUS_20261101_v1`.
사용자 승인에 따라 QB40의 미시작 큐를 교체한다. 실행 중인 case는 원래 horizon까지
완료하고 전환한다. 기존 결과·실패 기록·weights를 삭제하거나 완료로 바꾸지 않는다.

## 실행 목록

두 서버 모두 같은 순서다. 각 서버에서 Stage 1을 독립 학습하고 그 서버의 Student만 사용한다.

| 순서 | case | 설정 | seed |
|---:|---|---|---|
| 경계 | 현재 실행 중인 QB40 case | 원래 설정과 horizon 유지 | 기존 seed |
| 1 | QB5_S1_A400_E030 | Stage 1 50K, λ_abs=4, λ_ε=.03, ε 반경 3px | s1:261016101 / s3:261016301 |
| 2 | QB5_A400_E030 R1 | 기준 Student 30K | 261018101 |
| 3 | QB5_A400_E030 R2 | 기준 Student 30K | 261018102 |
| 4 | QB5_A400_E030 R3 | 기준 Student 30K | 261018103 |

Student: LR=1e-4, cosine→0, warmup100, B48=micro12×4, W112 D123, WD=.01,
DIFF/UBIL, `[PANc, PANc−PAN, Ilrms]`, α=.25, γ=.5, β=0, Aligner scale=1.
Stage 1 LR_A=1e-5 / LR_T=1e-4. RGB reference, Lanczos-4, 기존 Hard loss를 유지한다.
회전 일관성 loss, TTA, SWA, 밴드 가중 loss를 추가하지 않는다.

- 미시작 B96 반복은 실행 목록에서 제외한다. s3의 완료된 네 결과는 보존한다.
- C60 추가 반복, ε=.10 branch의 미시작 작업, W128H050은 결과 검토 전까지 보류한다.
  이번 자동 실행 목록에는 포함하지 않는다. 현재 case가 이 중 하나면 그 case만 완료한다.
- A3/A6, 두 번째 Stage 1 seed는 A4 결과에 따라 결정할 후보이며 자동 등록하지 않는다.
- s1은 위 네 작업과 Sheet 최종 readback을 마친 뒤 기존 WV3 TUNE30 → HARD36을 재개한다.
  s3는 위 큐 종료 후 정지한다. 서버 간 완료 대기나 공용 donor 제한은 없다.
- 명목 학습 시간은 s1 10.33h, s3 7.11h이며 현재 case의 잔여 시간·준비 비용은 별도다.
  이번 유한 큐에는 벽시계 시간 때문에 case를 생략하는 제한을 두지 않는다.

## s3 인계와 실행

사용자가 push한 뒤 s3 에이전트가 실행한다. 이 커밋은 기존 QB40 학습 소스를 수정하지 않는다.
실행 중인 QB40이 있는 경우, pull 범위에 다른 커밋의 QB40/공유 학습 코드 변경까지 포함되는지
먼저 확인한다. 그런 변경은 진행 중인 코드에 덮어쓰지 말고 원래 실행 소스를 보존해야 한다.
이번 커밋의 새 `panda_qb_rre`와 `config/panda_qb_rre`는 기존 QB40과 별도 namespace다.

```bash
git pull --ff-only
conda activate pancrafter
python -m panda_qb_rre catalog --server s3 --csv
python -m panda_qb_rre.handoff plan --server s3
python -m panda_qb_rre.handoff apply --server s3
```

`apply`는 기존 QB40 큐 컨트롤러를 종료하고 현재 trainer에 안전 저장을 요청한다.
optimizer/RNG/data stream이 포함된 체크포인트를 검증한 뒤 같은 이미지·환경·설정으로
`finish_current` adapter가 현재 case만 이어 학습한다. 학습 step이나 LR을 초기화하지 않는다.
경계 완료와 기존 결과의 Sheet readback을 확인한 후 새 큐를 자동 시작한다.
동일 `apply`를 다시 호출하면 기존 컨트롤러를 확인하며 중복 학습을 만들지 않는다.
진단되지 않은 실패는 자동 재시작하거나 정상 완료로 취급하지 않는다.

s1에서는 같은 명령의 `--server s1`을 사용한다. `enqueue`를 직접 실행하면 경계 전환과
WV3 후속 연결을 건너뛰므로, 기존 QB40을 교체할 때는 위 `handoff apply`를 사용한다.

현재 이미 A4/E.03을 QB40에서 완료한 s3는 위 새 Stage 1과 같은 seed의 중복 실행이 된다.
그 경우 에이전트는 기존 완료 donor의 full50K/validation receipt와 Student 실행 여부를 먼저
확인하고 재사용 전환을 별도로 구성해야 한다. 기존 weights를 새 캠페인 결과로 복사해
새 학습처럼 기록하지 않는다. 2026-11-01 20시대 Sheet 확인에서는 s3의 A4 완료 기록은 없었다.

## 선택·비교·보존

Stage 1은 full50K 후 **검증 RRE50 → RRE90 → validation ERGAS → 이른 step** 순으로 선택한다.
Teacher HQNR이나 test 수치로 선택하지 않는다. RRE50=.23 같은 새 중단 경계도 두지 않는다.
새 Stage 1이 기존의 알려진 support 검사로 실패하면 저장 증거를 확인하고 해당 Stage 1과
Student 세 개를 실패/의존 보류로 명시한다. 부분 학습 donor로 Student를 시작하지 않는다.

공식 평가와 checkpoint는 1K마다 보존한다. Student 공식 8K·9K·10K HQNR이 모두 .90 미만이면
기존 조기 중단 규칙을 적용한다. Stage 1에는 이 규칙을 적용하지 않는다.

두 서버 모두 **QB-ablations**에 게시한다.

- 동일 checkpoint에서 **HQNR≥.9315 AND ERGAS≤3.5**를 충족한 결과 전부.
- run/seed별 대표 한 개: HQNR 충족 후보 중 최소 ERGAS, 없으면 최대 HQNR.
- 대표와 목표 통과 결과가 같으면 한 번만 게시.
- Stage 1은 기존 RRE 선택 결과를 Teacher 행으로 게시.
- Approach에 Stage 1 계수, α/γ/β, scale, LR/horizon, batch, width, sampling 등을 표시.

새 행은 `PQRRE-` namespace를 사용하며 QB40 `PQ40-` 행을 수정 대상으로 삼지 않는다.
비교는 서버별 기존 A2/E.03의 동일 Student seed와 하며, 공식 대표 결과 및 고정 30K를 함께 본다.
seed 평균은 반복 신뢰도 분석에만 사용한다. 이동 제거 oracle ERGAS는 목표 판정에 쓰지 않는다.

## 상태와 검증

```text
work_dir/panda_qb_rre/handoff_s1/status.json
work_dir/panda_qb_rre/handoff_s3/status.json
work_dir/panda_qb_rre/launcher_<server>/status.json
work_dir/panda_qb_rre/PANDA_S13_QB_RRE_FOCUS_20261101_v1/<server>/QB/campaign_state.json
work_dir/panda_qb40/PANDA_S13_QB40_LOCALSTAGE1_20261101_v1/<server>/QB/rre_priority_boundary.json
```

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s panda_qb_rre/tests -v
```

기존 실행 소스 pin을 보존하기 위해 QB40 runtime/게시 경로를 별도 namespace로 분리했다.
실제 Aligner/Teacher 클래스는 기존 S1LOSS9를 그대로 사용한다. Stage 1 손실/gradient,
Student 손실/gradient 및 optimizer·RNG·stream 재개를 QB40과 비교 검증한다.
실서버에서는 새 큐 시작 전 native cost 측정, 각 case의 실제 데이터 update/replay 검사를 수행한다.
