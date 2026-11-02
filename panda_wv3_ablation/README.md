# WV3 component ablations — anchored Stage 1 / Stage 2

2026-11-02 구현. s3는 Stage 1, s1은 Stage 2만 실행한다. 서버 간 weight 이동은 없다.
이 코드를 커밋했다고 학습이 시작되지는 않는다. 아래 launcher의 `--check-only`를 빼고 실행하면 시작한다.

## 실행 순서

- **s3: 5 case 전체 R1 → 5 case 전체 R2 → 5 case 전체 R3 = 15 runs**
  - A0 RAW → A1 SCHARR → A2 REC → A3 REC_SPA → A4 REC_EPS.
- **s1: 9 case 전체 R1 → 9 case 전체 R2 → 9 case 전체 R3 = 27 runs**
  - B0 CORR_GT → B1 DIFF_GT → B2 E_ONLY → B3 EQ_ONLY → B5 NO_DIFF_AF
    → D1 GLOBAL_RANK → D2 NO_E_CUE → D3 MEAN_WEIGHT → D4 TWO_PANS.
- R1/R2/R3의 Stage 1 seed는 261016101/102/103, Student seed는 261018101/102/103.
- 시간 제한·성능 미달 조기 종료·Full 재학습·다른 캠페인으로의 자동 전환은 없다.

[s1 CSV](../config/panda_wv3_ablation/cases_s1.csv),
[s3 CSV](../config/panda_wv3_ablation/cases_s3.csv),
[전체 설계와 해석 범위](../research_log/PANDA_WV3_Anchored_Ablation_Plan_2026-11-02.md).

## 원본 reference와 실제 설정

| 구분 | Stage 1 | Stage 2 |
|---|---|---|
| 기준 | `PB30_WV3_S1_G8_S1_U23_R1_S261016101` / `G8_MATURE_U23` | `TUNE30 L4 S100DIFFUBILC30G025 R1` |
| 고정 비교 | 50K | 29K, 30K는 보조 |
| horizon | 50K | 30K |
| 모델 | W112/D123/BASE | W112/D123/UBIL |
| LR | A 1e-5 / T 1e-4 | S 1e-4 |
| 계수 | λ_abs=2, λ_ε=.003, ε disk R3px (항 존재 시) | α=.25, γ=.25, β=0 (항 존재 시) |
| 정합 | learned A는 scale1; mean8 GT spatial target | 동일 s1 A/T 동결, scale1; RGB q target |
| warp | bicubic border | Student L4 reflection / Teacher bicubic border |

양 단계 AdamW WD=.01, warmup100, cosine floor0, batch48=micro12×4, FP32다.
Stored LMS 8밴드는 Aligner 입력이며 reconstruction base는 기존 interp23다.
9↔10채널 공통 weight는 동일하고 추가 PAN 열은 0으로 시작한다.
새 loss·TTA·회전 일관성·SWA·밴드 가중을 도입하지 않는다.

Stage 1에서 꺼진 항의 forward/gradient를 생략한다. Stage 2에서 T가 필요 없는 case도
Student L4/Teacher bicubic support 교집합을 유지한다. weight sum으로 loss를 재정규화하지 않는다.
D4의 C,P 입력은 C,C−P와 같은 정보이므로 표현/최적화의 대조로 해석한다.

Scharr baseline은 **관측 PAN/mean8 Stored LMS만** 보고 영상별 한 shift를 찾는다.
정수 coarse 각 축 ±5px, ±1px/.25px fine, ROI8이다. ±5는 bicubic stencil7을 보존하는 최대 대칭 정수 범위이며
ε disk R3와 별개다. GT·test 지표로 shift를 선택하지 않는다.
처음 제안한 ±3px는 train/val 사전 점검 후 위 support 기준으로 넓혔다.
표본 32개씩에서 ±5 경계 도달률도 train 21.875%, val 43.75%여서 완전한 정합이라고 보장할 수 없다.
train log에 boundary-hit/subpixel 비율을 기록한다. 이 수치는 전체 데이터 분포 추정이 아니다.

## pull 이후 실행

이전 버전 파일 일부만 복사하지 말고 이 커밋 전체를 pull한다. s3는 s1 checkpoint가 필요 없다.
s1은 [references.json](../config/panda_wv3_ablation/references.json)에 명시된 기존 로컬 50K A/T와 RRE receipt를 사용한다.
원본 SHA가 다른 donor로 자동 대체하지 않는다. source/환경/데이터/evaluator/Sheet는 시작 전에 검사한다.

기존 환경 이미지가 있으면 같은 tag를 사용할 수 있다. 없으면 저장소의 `Dockerfile.env`로 준비한다.

```bash
# 호스트에는 학습용 Python 패키지가 없어도 된다.
python3 -m panda_wv3_ablation catalog --server s1
python3 -m panda_wv3_ablation catalog --server s3

# 각 서버에서 자기 서버 명령을 실행한다. --check-only는 GPU 학습/Sheet 쓰기 없음.
python3 -m panda_wv3_ablation.enqueue --server s1 --image hojunqueen/pancrafter-env:latest --check-only
python3 -m panda_wv3_ablation.enqueue --server s3 --image hojunqueen/pancrafter-env:latest --check-only

# 실제 시작: 기존 로컬 GPU 작업이 끝날 때까지 기다린다. 기존 작업을 정지하지 않는다.
python3 -m panda_wv3_ablation.enqueue --server s1 --image hojunqueen/pancrafter-env:latest --wait
python3 -m panda_wv3_ablation.enqueue --server s3 --image hojunqueen/pancrafter-env:latest --wait
```

이미지 내 DLPan 버전이 다르거나 설치 경로가 외부라면 `--dlpan /실제/DLPan-Toolbox`를 준다.
데이터 symlink의 실제 WV3 H5 경로는 양 서버 모두 read-only로 마운트한다.
원본 저장소는 read-only, 이 캠페인의 `work_dir/panda_wv3_ablation`만 writeable이다.
native 환경은 `python -m panda_wv3_ablation run --server s1 --device cuda:0`으로도 실행할 수 있다.

```bash
docker logs -f panda-wv3-components-s1-v1
docker logs -f panda-wv3-components-s3-v1

# 중단 시 마지막 완료 update를 저장한다. 중단한 같은 컨테이너를 재개한다.
docker stop --time 180 panda-wv3-components-s1-v1
docker start -a panda-wv3-components-s1-v1
```

## 저장·복구·Sheet

출력: `work_dir/panda_wv3_ablation/PANDA_WV3_COMPONENT_S13_20261102_v1/<server>/WV3/`.

- `campaign_state.json`: round-major 순서, 각 run status/step/error. 중단한 run부터 재개한다.
- `runs/<run>/config.json`, `train.jsonl`: 실제 설정과 매 update 손실·LR·shift/support·시간.
- `models/step_*.pt`: **1K 평가마다 model/optimizer/sampler/augmentation/ε RNG를 모두 저장**한다.
  목표 달성·대표·29K·30K·50K weights를 삭제하지 않는다.
- `resume/`: 250 step마다 원자적으로 저장하는 최근 두 세대. 영구 `models/`와 분리한다.
  종료 신호에도 마지막 완료 update를 저장한다. crash 이후 미확정 train log suffix는 백업 후 되감는다.
- `evaluation_*.json`: 같은 상태의 RR20/FR20 전체 결과·모델 SHA·checkpoint 파일 SHA.
- Stage 1의 `validation_*.json`: 동일 1080 validation 표본의 RRE/ERGAS와 learned A의 ε response.
  validation RRE50→RRE90→ValERGAS→이른 step으로 대표를 고른다. test로 donor를 고르지 않는다.
- `fixed_step_*.json`: 논문용 고정 50K/29K/30K 결과. Sheet의 대표값과 구분한다.

`WV3-ablations`의 Experiment는 `WV3-ABL-S2-B2-E-ONLY R1`처럼 case를 표시한다.
Approach에는 loss flags/계수·입력·LR·warp·horizon이 들어간다.
Stage 1은 Val-RRE 대표, Stage 2는 **HQNR≥.9585 AND ERGAS≤2.040의 모든 step + run 대표**를 게시한다.
대표는 HQNR 조건을 만족하는 것 중 최소 ERGAS, 없으면 최고 HQNR이다. 대표와 통과 step 중복을 합친다.
미측정 지표는 빈 셀이다. 원본 reference 행, 타 서버·타 캠페인, BM 이후 열은 보존한다.

실행 중 시작/평가/완료/실패마다 자동 게시한다. Sheet 장애는 로컬 학습·weight 저장을 없애지 않으며
`publication/last_error.json`에 남긴다. 이후 poll은 새 cloud snapshot에서 재시도한다.
필요하면 별도 게시기로 완료 후에도 재시도할 수 있다:

```bash
python -m panda_wv3_ablation publish --server s1 --apply --watch
python -m panda_wv3_ablation publish --server s3 --apply --watch
```

수치/support 오류의 run은 FAILED로 기록하고 독립된 다음 case를 진행한다.
데이터·code·checkpoint 오류는 캠페인을 중지한다. 종료 상태 `FINISHED_WITH_FAILURES`를 전체 성공으로 해석하지 않는다.
현재 source release를 변경하면 기존 checkpoint의 pins와 달라져 조용히 다른 코드로 재개하지 않는다.

## 검증

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s panda_wv3_ablation/tests -v
python -m panda_wv3_ablation.release --environment
```

깨끗한 Git staged checkout에서도 **21개 테스트와 394개 파일 source 검증을 통과**했다.
[검증 기록](../results_log/assets/2026-11-02_wv3-component-ablation/implementation_validation.json).

검증 범위: 실제 W112 forward/backward, 기존 Stage1/2 loss·gradient 일치, 추가 채널 초기화,
donor 동결, Scharr shift 부호·subpixel·support, pooled rank, mean weight, 완전한 CPU resume,
순회 큐와 crash trace, 목표 경계·대표 변경·외부 행 보존·append 위치, s3 WV3 mount.
원본 29K checkpoint의 RR20/FR20 CPU replay는 HQNR .9603361215 / ERGAS 2.0340588129로
기존 GPU 기록과 각각 약 1.2e-9 / 8.3e-8 차이였다. 이는 새 ablation 결과가 아니다.
실제 GPU 학습 전체 완료를 의미하지 않는다.
