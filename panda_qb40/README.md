# QB40 — 2026-11-01 실행 인계

캠페인: `PANDA_S13_QB40_LOCALSTAGE1_20261101_v1`.
[승인된 40시간 계획](../config/panda_qb40/plan.json)을 실제 실행하는 패키지다.
[결과 분석](../results_log/2026-11-01_qb-results-and-40h-followup-plan.md)의 후속이다.
이전 보고서의 `PLANNED_NOT_REGISTERED` 상태는 본 구현 전 시점의 기록이다.

## 실행

s1은 기존 WV3의 **현재 case만 완료 → QB40 → 나머지 WV3 TUNE30 → HARD36** 순서다.
현재 case에 optimizer/RNG/data stream을 포함한 안전 저장을 요청하고, 기존 finish-current
adapter로 동일 상태를 이어 학습한다. 설정·학습 step은 변경하지 않는다.

```bash
python -m panda_qb40.priority plan
python -m panda_qb40.priority apply
```

s3 에이전트는 사용자의 push 이후 해당 서버에서 pull하고 기존 `pancrafter` 환경으로 실행한다.
s1 완료나 공용 Stage 1을 기다리는 조건은 없다. `--wait`는 해당 서버 GPU 사용 중일 때만 기다린다.

```bash
git pull --ff-only
python -m panda_qb40 catalog --server s3 --csv
mkdir -p work_dir/panda_qb40/launcher_s3
nohup env PYTHONDONTWRITEBYTECODE=1 python -u -m panda_qb40.enqueue --server s3 --wait > work_dir/panda_qb40/launcher_s3/enqueue.log 2>&1 < /dev/null &
```

동일 명령을 중복 실행해도 이 캠페인 출력의 동시 작성은 차단한다. 이는 프로세스 중복 방지이며
다른 서버/캠페인의 완료나 공유 모델을 요구하는 제한이 아니다. 시작 전 큐는 수정 가능하다.
실행 중인 checkpoint의 데이터·모델·optimizer·소스 해시는 정확한 재개를 위해 보존한다.

원본 SENSOR5 Docker의 이미지/환경/마운트를 재사용한다. QB symlink 대상 데이터는 두 서버 모두
read-only로 추가 마운트하고, 새 `work_dir/panda_qb40`만 쓰기 가능하게 마운트한다.
기존 서버 로컬 donor 경로/해시가 불일치하면 다른 서버 모델로 대체하지 않고 오류를 기록한다.

## 실행 목록

아래 순서를 그대로 등록했다. R1/R2/R3 Student seed는 261018101/102/103이다.

| 서버 | 순서 | 실행 |
|---|---:|---|
| s1 | 1–3 | 새 Stage 1 A2/E.10 50K → 해당 Student C30 R1/R2 |
| s1 | 4–5 | 기존 로컬 donor, B48 C60 LR5e-5 R1/R2 |
| s1 | 6–8 | 새 Stage 1 A4/E.03 50K → 해당 Student C30 R1/R2 |
| s1 | 9–10 | 기존 로컬 donor, B96 C15 LR1e-4 R1/R2 |
| s1 | 11–12 | 기존 로컬 donor, B96 C15 LR2e-4 R1/R2 |
| s3 | 1–4 | 기존 로컬 donor, B96 C15 LR1e-4 R1/R2 → LR2e-4 R1/R2 |
| s3 | 5–7 | 새 Stage 1 A2/E.10 50K → 해당 Student C30 R1/R2 |
| s3 | 8–10 | 새 Stage 1 A4/E.03 50K → 해당 Student C30 R1/R2 |
| s3 | 11–12 | 기존 로컬 donor, B48 C60 LR5e-5 R1/R2 |
| s3 | 13–15 | 새 Stage 1 A4/E.10 50K → 해당 Student C30 R1/R2 |
| s3 | 16–17 | 새 A2/E.10 및 A4/E.03 donor의 Student C30 R3 |
| s3 | 18–19 | 기존 로컬 donor, W128 H050 C30 R1/R2 |

- s1: 새 Stage 1 2개 + Student 10개, 추정 학습 35.12시간 + 여유 4.88시간.
- s3: 새 Stage 1 3개 + Student 16개, 추정 학습 36.36시간 + 여유 3.64시간.
- 새 Stage 1 seed는 s1=261016101, s3=261016301. 서버 간 독립 모델이다.
- 기존 donor는 s1 S1LOSS9 A2/E.03 및 s3 QB_DEEPEN 자체 학습 A2/E.03이다.
  선택 step을 새로 고정하지 않고, 해당 서버의 원래 full50K validation 선택 receipt를 검증한다.
- 새 Stage 1은 각 50K 전체를 학습한 후 기존 RRE50 → RRE90 → validation ERGAS → 빠른 step
  기준으로 선택한다. Teacher FR HQNR로 중단하거나 중간 checkpoint를 최종 donor로 지정하지 않는다.
- 각 Student는 fresh initialization. `[PANc, PANc−PAN, Ilrms]`, scale=1, DIFF/UBIL,
  RGB reference 및 A/T/S Lanczos4, β=0, 기존 Hard loss만 사용한다.
- 기본 W112, α=.25/γ=.5. W128H050만 W128, α=.125/γ=.25. WD=.01.
- B48=12×4 누적, B96=12×8 누적. C15 warmup50, C30/C60 warmup100, cosine LR→0.
- `L_epsilon` 계수 .10은 주입 shift 반경이 아니다. 모든 case의 반경은 기존 3px.

## 시간·재개·결과 보존

40시간은 **현재 WV3 완료 후 QB40 자체 profiling이 시작되는 시점부터** 계산한다.
새 run 시작 전 기존 관측 시간과 해당 batch/graph의 실측 train/eval/save 시간, 종료 여유를 고려한다.
실측 부족 시 계획상의 보수적 추정치를 사용하며, 이후 같은 graph/batch 관측치로 보강한다.
이미 승인해 시작한 run은 끝까지 진행한다. 시간 초과를 이유로 Stage 1을 부분 선택하지 않는다.
시간이 부족한 후순위 작업은 `BUDGET_COMPLETE_WITH_PENDING`과 `pending_groups`로 남기며
완료로 표시하지 않는다. 실제 속도가 추정보다 느리면 총 시간이 40시간을 넘을 수 있다.
이 상태는 40시간 창의 종료이며 모든 case 완료를 뜻하지 않는다.

공식 RR20/FR20 평가 1K마다 해당 weights와 SHA/complete resume state를 보존한다.
목표를 달성한 weights뿐 아니라 모든 평가 checkpoint를 남긴다.
QB Student의 공식 8K/9K/10K HQNR이 **모두 엄격히 .90 미만**이면 기존 규칙대로 안전 중단한다.
B96도 이 절대 update 규칙을 바꾸지 않는다. Teacher Stage 1에는 적용하지 않는다.
수치/코드 오류는 `FAILED`로 명시하고 반복 재시작하지 않는다.
단, Stage 1의 알려진 `SUPPORT_VIOLATION`은 실패 기록·정상 저장된 중단 checkpoint·학습 trace를
검증한 뒤 해당 branch만 보류한다. 그 Stage 1을 필요로 하는 Student는
`BLOCKED_FAILED_STAGE1`로 남기고, 독립 case는 원래 순서대로 진행한다.
이 경우 최종 상태는 `FINISHED_WITH_FAILED_BRANCHES`이며 실패/보류 개수를 별도로 표시한다.
실패한 Stage 1의 부분 checkpoint는 donor로 선택하지 않는다. ROI/warp/손실은 바꾸지 않는다.
알 수 없는 오류, optimizer 부분 실패 또는 checkpoint 저장 오류는 계속 전체 중단 대상으로 둔다.

### 2026-11-01 s1 복구

첫 `QB5_S1_A200_E100` Stage 1이 14,219 update에서 support 검사로 중단됐다.
해당 Stage 1과 `QB5_A200_E100 R1/R2`를 보류하고, `QB5_C60LR050 R1`부터
독립 작업 9개를 이어 실행한다. 기존 실패 기록·모든 weights·원래 admission은 보존한다.

```bash
python -m panda_qb40.recovery --server s1 --wait
```

이 명령은 진단된 실패 컨테이너를 검증하고, 제어 코드 변경만 기록한 명시적 복구 receipt를 만든다.
학습 코드·데이터·평가·실험 설정 변경이나 이전 trainer의 재개는 허용하지 않는다.
`admission_control_v2.json`에 새 제어 버전의 admission을 기록하고 원래 `admission.json`은 유지한다.
40시간 시계의 시작점도 유지한다. QB 큐 종료 및 Sheet readback 후 기존 WV3 큐를 이어간다.

s3가 QB40을 아직 시작하지 않았다면 수정 커밋을 pull한 후 위의 기존 enqueue 명령으로 실행한다.
새 실패 분기 처리는 s3에도 적용되지만, s1의 실패 판정을 s3 모델에 전파하지 않는다.
이미 QB40을 실행 중인 서버에서는 실행 중 소스가 바뀌지 않도록 pull을 학습 종료 후에 한다.
복구 명령은 기존 실행이 모두 인증된 Stage 1 support 실패인 상황만 처리하며,
정상 실행 중인 trainer나 이미 완료된 다른 case가 있는 캠페인을 임의로 재등록하지 않는다.

## Sheet

두 서버 모두 **QB-ablations**, 목표 **HQNR ≥ .9315 AND ERGAS ≤ 3.5**.
동일 checkpoint의 두 조건 동시 달성 결과를 전부 게시하고,
case/seed별 대표 결과는 HQNR 조건을 만족하는 후보 중 최소 ERGAS,
HQNR 조건 충족 후보가 없으면 최대 HQNR을 게시한다. 중복된 대표/통과 행은 한 번 게시한다.
새 Stage 1은 validation RRE 선택 결과를 별도 Teacher 행으로 게시한다.

Approach에 Stage 1 계수, scale, width, α/γ/β, LR/horizon, batch/누적 횟수,
warmup, WD 및 실험 질문을 표시한다. 새 `PQ40-` namespace의 행만 관리한다.
전체 종료 시 Sheet readback 확인 후 s1의 WV3를 재개한다.

상태 경로:

```text
work_dir/panda_qb40/priority_s1/status.json
work_dir/panda_qb40/launcher_s1/status.json
work_dir/panda_qb40/launcher_s3/status.json
work_dir/panda_qb40/<campaign>/<server>/QB/campaign_state.json
work_dir/panda_qb40/<campaign>/<server>/QB/sheet_publish_panda_qb40/last_poll.json
```

## 구현 계보와 검증

실행 중인 이전 패키지의 소스 pin을 변경하지 않기 위해 S1LOSS9의 Stage 1·선택·runtime·공식 평가·
게시 코드를 새 namespace에 분리했다. Student의 W128 지원 및 source를 유지하는 calibration은
SENSOR_FOCUS에서 가져왔다. Aligner/Teacher 실제 클래스와 수학 연산은 원래 S1LOSS9를 재사용한다.
새로운 손실/augmentation/TTA/SWA/EMA는 없다.

테스트는 batch96 실제 optimizer update·stream 소비·checkpoint/RNG 재개, Stage1 loss/gradient와
full-state 재개, W128 forward, 양 서버 registry/target/출처, GPU 중복 실행 방지,
WV3 안전 전환 및 budget/pending 처리를 검증한다.
실서버 시작 시에도 native cost 측정 후 각 새 case의 실제 데이터 update/replay preflight를 수행한다.
이 disposable 모델은 본 실험 초기화나 Sheet 성능 기록으로 사용하지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s panda_qb40/tests -v
```
