# FOLLOW30: full Aligner scale 1, existing Hard and single checkpoints

2026-10-24 사용자 수정 반영. **앞서 편성한 낮은 scale/SWA 큐를 대체한다.**
캠페인 `PANDA_S13_FOLLOW_HARDONLY_20261024_v1`, plan revision `FULL_SCALE_NO_WEIGHT_AVERAGING_20261024`.
이 캠페인은 아직 학습 시작 전이다. 사용자 승인에 따라 NEXT30의 **아직 시작하지 않은 scale<1 case만 제외**한다.
이미 시작·완료한 case와 full-scale S100C30/S100DIFFC30 R1·R2는 유지하며, 그 종료 후 시간 제한 없이 이어서 실행한다.
기존 NEXT30 코드와 checkpoint는 변경하지 않고, 공식 pending-control 파일만 사용한다.

## SWA 제외와 주력 변경 근거

SWA는 독립적인 새 논문 기여로 제안하지 않더라도, 성능에 적용했다면 학습·모델 생성 절차에
평균 규칙을 밝혀야 하는 추가 학습 기법이다. [SWA 원 논문](https://arxiv.org/abs/1803.05407).
사용자 결정에 따라 **4개 SWA 학습 궤적, 평균 누적·재개·평가·export 코드와 별도 게시 경로를 제거**했다.
EMA·model soup로 대체하지 않는다. 일반 학습 중 저장된 **단일 checkpoint**만 평가·게시한다.
낮은 scale .05/.1 추가 탐색과 .2 R3도 이번 후속 큐에서는 제외한다. 이미 진행 중인 NEXT30의
낮은 scale 실험과 과거 결과는 그대로 유지한다.

동일 s1·R1·30K·Lanczos4 공식 평가:

| 입력 | HQNR | ERGAS |
|---|---:|---:|
| scale .1 / corrected PAN 9ch | .956300 | 2.067809 |
| scale .2 / corrected PAN 9ch | .955780 | 2.059521 |
| **scale 1 / DIFF 10ch** | **.957647** | **2.052012** |
| scale 1 / corrected PAN 9ch | .916943 | 2.039418 |

scale 1 DIFF가 낮은 scale과 견줄 뿐 아니라 두 지표가 더 좋은 관측이 있으므로 이를 주력으로 한다.
s3 R1 대표(29K)도 .958687 / 2.049594로 근접했다. 다만 s3 R2 대표(15K)는 .958686 / 2.087480이며
같은 근접도를 재현하지 못했다. 서로 다른 대표 step을 학습량 대응 대조로 사용하지 않는다.
출처·비교 범위와 s3 인계는 [FOLLOW30 인계 문서](../research_log/PANDA_Follow30_Handoff_2026-10-24.md)에 함께 기록했다.

**scale 1 적용과 Aligner를 기능적으로 충분히 활용하는 것은 별개의 주장이다.**
DIFF는 `[PANc, PANc−PAN, MSbase8]`이므로 원본 PAN도 복원 가능한 입력이다.
기존 탐색 진단에서 원 PAN만 주면 ERGAS가 2.0520→2.1268로 나빠졌지만 HQNR는 거의 같았다.
PAN 1.5px 어긋남 주입 시 DIFF ERGAS 4.5074, corrected-only 9ch 2.2225였다.
이는 보정 관련 입력이 RR 정확도에 쓰이는 신호지만, 9ch의 어긋남 내성이 보존된 것은 아니다.
입력 개입은 학습 분포도 바꾸므로 기하의 순수 인과 효과로 단정하지 않는다.
이 진단은 기존 로컬 probe이며, 수치와 한계는 위 인계 문서에 적었다. 실행 코드에는 probe 산출물이 필요하지 않다.

## 공통 설정

- **전 case scale 1.0 고정**: 예측 shift 전량을 PAN에 적용. MS·출력·평가 참조는 옮기지 않는다.
- WV3, 같은 서버의 Mature30 `BEST_VAL_RRE` frozen Aligner/Teacher·calibration 재사용.
  서버 간 donor가 다르므로 결과를 seed 반복으로 합치지 않는다.
- Student warp Lanczos4. frozen Teacher의 기존 bicubic 경로는 유지한다. Stage1 재학습 없음.
- 기본 입력 9ch `[PANc, MSbase8]`; 기존 승인된 DIFF 입력 10ch `[PANc, PANc−PAN, MSbase8]`.
  차영상 첫 conv 열은 0 초기화, 추가 1,008 parameter, 출력 8밴드.
- Student만 AdamW 학습. 기존 GT Hard 유지, soft loss 0, rotation loss 0, band-weighted loss 없음.
  기존 C4 데이터 변환은 유지하고 공식 평가에는 TTA·ensemble을 쓰지 않는다.
- Hard 기본 α=.25, γ=.5. H050은 α=.125, γ=.25로 **두 추가 가중 계수**를 반감한다.
  Hard 전체 loss의 절반이나 Aligner shift의 절반이라는 뜻이 아니다.
- fresh warmup 100 update 후 cosine→0. 공식 RR20/FR20 매 1K 평가.
- R1/R2/R3 seed: 261018101 / 261018102 / 261018103.

## 등록한 8개 설정 / 22개 run

| case | 입력·구조 | 일정 / LR / Hard | s1 | s3 | 목적 |
|---|---|---|---|---|---|
| **S100DIFFC50** | DIFF / BASE | fresh 50K, 1e-4, 기본 | R1·R2 | R1·R2 | 주력: 긴 cosine에서 ERGAS 개선과 HQNR 유지 |
| S100DIFFLR150C50 | DIFF / BASE | fresh 50K, 1.5e-4, 기본 | — | R1·R2 | 위 50K 기준과 LR만 비교 |
| S100DIFFH050C50 | DIFF / BASE | fresh 50K, 1e-4, H050 | — | R1·R2 | 위 50K 기준과 Hard 계수 묶음만 비교 |
| S100DIFFUBILC30 | DIFF / UBIL | fresh 30K, 1e-4, 기본 | R1·R2 | R1·R2 | 기존 DIFFC30과 upsample만 비교 |
| S100DIFFF15C30 | DIFF / BASE | NEXT30 동일 seed 15K→30K, 2e-5→1e-6 cosine | R1·R2 | R1·R2 | 동일 부모에서 이른 LR 감소 |
| S100DIFFC30 | DIFF / BASE | fresh 30K, 1e-4, 기본 | R3 | R3 | DIFF 세 번째 seed |
| S100C30 | corrected-only / BASE | fresh 30K, 1e-4, 기본 | R3 | R3 | 위 R3의 같은 seed·full-scale 입력 대조 |
| S100UBILC30 | corrected-only / UBIL | fresh 30K, 1e-4, 기본 | R1·R2 | — | 원 PAN 통로가 없을 때의 UBIL 대조 |

**s1 10 run / 310K 새 update, s3 12 run / 450K 새 update, 합계 22 run / 760K**.
DIFF 주력 18개, corrected-only 대조 4개다. 모든 run의 Aligner scale은 1이다.
정확한 순서는 [s1 CSV](cases_s1.csv), [s3 CSV](cases_s3.csv), 실행 원본은 [plan.py](plan.py).

s1은 DIFFC50→DIFFUBIL→15K 분기→R3 입력 쌍→corrected-only UBIL 순이다.
s3는 DIFFC50→50K LR150→50K H050→DIFFUBIL→15K 분기→R3 입력 쌍 순이다.
corrected-only 9ch의 큰 HQNR 격차를 고려해 광범위한 LR/Hard 탐색은 하지 않는다.
마지막 UBIL 대조는 기존 NEXT30 BASE 결과와 합쳐 s1의
`입력 DIFF/9ch × upsample BASE/UBIL` 대응 비교를 완성하는 소수 실험이다.

DIFFC50과 C30의 차이는 cosine horizon 정책이다. 같은 30K 비교는 LR 경로 차이,
각각 최종 30K/50K 비교는 LR 경로와 추가 학습량이 함께 달라지므로 순수 학습량 효과로 부르지 않는다.
H050은 α/γ를 동시에 바꾸므로 두 항 각각의 기여를 분리하는 실험도 아니다.

## 재개·평가·기여 판단

15K 분기는 새 폴더에서 이어 학습한다. 원본 Student·AdamW moments/step·RNG·batch stream·
frozen donor·calibration을 보존하고 다음 update부터 LR schedule만 바꾼다.
부모의 package/campaign/run/같은 seed/정확한 step/파일 SHA/code pin/evaluator pin을 확인한다.
원본 work_dir는 읽기 전용이며, 부모가 아직 없으면 다른 checkpoint로 대체하지 않고 기다린다.

성능과 Aligner 활용은 나눠 판단한다. 새 학습 method를 추가하지 않고, 유망 checkpoint에서
기존 진단 도구로 다음을 확인할 수 있다(이 큐 등록 작업에서는 GPU 진단을 실행하지 않았다).

1. 공식 native RR20/FR20 성능: 동일 checkpoint의 H/E, D_lambda/D_s.
2. 같은 Student에서 원 PAN·MS를 고정하고 u만 바꿔 PANc와 차영상을 함께 재계산한 민감도.
   raw PAN 접근은 유지되므로 보정 정보에 대한 의존성을 점검하지만, 분포 변화도 보고한다.
3. 기존 controlled PAN 어긋남 주입 진단: DIFF와 corrected-only가 어떤 내성을 유지하는지 비교.
   진단 결과는 공식 지표·Sheet 통과 결과와 분리한다. 학습 augmentation으로 추가하지 않는다.

seed 한 번의 ERGAS 2.039/2.041 차이로 방법의 우열을 확정하지 않는다.
동일 seed·동일 step 대응과 후반 곡선을 보고, 단일 seed 통과와 반복 여부를 구분한다.
고정된 `.001` HQNR noise threshold나 seed 2–3개의 절대 신뢰 구간을 설정하지 않는다.

## Sheet 규칙

- 공동 목표: **동일 checkpoint HQNR ≥ .9585 AND ERGAS ≤ 2.040**.
- 모든 공동 통과 + run별 대표 하나. 대표는 HQNR 충족 중 최소 ERGAS, 없으면 최고 HQNR.
- Approach에 full-u(scale=1), L4, DIFF/9ch, UBIL/BASE, LR, horizon, Hard, fork를 표시한다.
- RAW 개별 checkpoint만 허용한다. 평균 모델 설정·provenance·artifact는 게시 검증에서 거부한다.
- 기존 fixed64·same-state RR/FR·checkpoint SHA·실측 비용 규칙을 유지한다.
  준비 과정에서 Sheet에 쓰지 않았다.

## s3 인계 / 실행

코드와 스크립트를 Git으로 전달하고 checkpoint·data·credentials는 Git에 넣지 않는다.
s3는 자신의 NEXT30/Mature30 결과와 donor를 사용한다. `gspread/server.txt`는 `s3`여야 한다.
기존 training/uploader Python 환경에서 읽기 전용 점검:

```bash
python -m panda_follow30 catalog --server s3 --csv
python -m panda_follow30 audit-sources --server s3
python -m panda_follow30 check --server s3
python -m panda_follow30.transition --server s3 --prepare-only
python -m panda_follow30 smoke
```

NEXT30 부모가 아직 없으면 `WAITING_REGISTERED_NEXT_PARENT`로 표시한다.
미착수 low-scale case 제외를 등록하고, 유지한 NEXT30 case의 정상 완료 후 이어서 실행할 명령:

```bash
bash scripts/start_panda_follow30_transition.sh s3
```

NEXT30가 이미 정상 종료됐으면 `bash scripts/start_panda_follow30_costed.sh s3`도 가능하다.
두 경로 모두 유지한 case의 완료 증거·승인된 skip 증거·idle GPU·실측 cost·실데이터 CUDA preflight를 확인한다.
s1은 서버 인자만 `s1`로 바꾼다. watcher가 적용한 요청·대기 상태는 인계 문서에 명시한 receipt로 확인한다.

결과 위치: `work_dir/panda_follow30/PANDA_S13_FOLLOW_HARDONLY_20261024_v1/<server>/WV3`.
Docker는 기존 work_dir와 저장소를 read-only로, 새 `work_dir/panda_follow30`만 writable로 mount한다.
실행 후 source code pin이 바뀌면 자동 대체하지 않는다.

## 검증 범위

CPU 검사: 전 case scale1/평균 배제, 비교 대조군의 실제 등록 여부와 동일 seed, 정확한 source LR,
continuation 완전 상태 재개, 새 H050 및 corrected-only UBIL의 실제 Hard update/재개,
동결 Aligner/Teacher 무변경, 기존 Sheet 통과·대표 선택 규칙을 확인한다.
새 GPU 학습·공식 평가·목표 달성 여부는 아직 검증하지 않았으며 시작 시 CUDA preflight가 필수다.

선택적 큐 전환까지 포함한 CPU 28개 테스트와 launcher 7개 shell syntax 검사가 통과했다.
실제 s1 제외 요청·전환 워처 등록 상태는 인계 문서에 기록했다.
22개 실행 config와 CSV의 일치, 단일 checkpoint 게시 정책, 평균 모델 거부를 포함한다.
기존 실험 코드는 유지하며, 실행 시 새 게시 결과에도 기존 통과·대표 선택 규칙을 적용한다.
