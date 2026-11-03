# WV3 Stage 1 RGB ablation — s3

2026-11-03 사용자 요청에 따른 새 Stage 1 캠페인이다. 기존 WV3 main의 G8 donor 및
s1 Stage 2 ablation은 수정하지 않는다. 기존 mean8 run을 RGB로 이름만 바꾸거나
이어 학습하지 않고, 각 case/seed에서 Aligner와 Teacher를 처음부터 학습한다.

## s3에서 pull 후 바로 전환

사용자가 push한 커밋 전체를 s3에서 pull한 뒤 저장소 루트에서 실행한다.

```bash
python3 -m panda_wv3_ablation_rgb.enqueue --server s3 --image hojunqueen/pancrafter-env:latest --replace-legacy
docker logs -f panda-wv3-components-rgb-s3-v1
```

기존 이미지 tag가 다르면 `--image`만 해당 환경 이미지로 지정한다. DLPan 경로가 별도이면
`--dlpan /실제/DLPan-Toolbox`를 추가한다. 원래 ablation과 동일한 torch/numpy/CUDA build,
평가기, WV3 데이터 registry를 검증한다. GPU 종류는 고정하지 않는다.

- 소스·환경·데이터·RGB validation inventory·Sheet 읽기 검사를 먼저 수행한다.
- `--replace-legacy`는 이름, campaign/server label, checkout mount가 모두 맞는
  `panda-wv3-components-s3-v1`만 정지한다. 종료 신호로 마지막 완료 update 저장을 요청한다.
- 기존 컨테이너·weight·Sheet 행은 보존하고 자동 재시작을 해제한다.
- s1, QB, GF2 등 다른 GPU 작업은 정지하지 않는다. 다른 작업이 있으면 시작을 중단하며,
  기다리려면 `--wait`를 추가한다. native Python으로 실행된 구 실험은 이 Docker 전환기의
  정지 대상이 아니므로, s3 에이전트가 해당 프로세스를 확인하고 정상 종료해야 한다.
- `--check-only`를 추가하면 기존 실험 정지·학습 시작·Sheet 쓰기 없이 점검한다.
- 새 RGB 컨테이너가 이미 실행 중이면 중복 실행하지 않는다. 같은 release의 정지된
  RGB 컨테이너는 해당 RGB checkpoint부터 재개한다.

s3 직접 접속이 가능한 에이전트가 위 명령으로 전환해야 한다. 커밋/pull 자체는 학습을 시작하지 않는다.

## Case와 순서

| Case | 정합 | λ_spa | λ_ε | 학습 |
|---|---|---:|---:|---|
| A0 RAW | 없음 | 0 | 0 | Teacher만 |
| A1 SCHARR | 영상별 RGB Scharr shift | 0 | 0 | Teacher만 |
| A2 REC | PAN Aligner | 0 | 0 | Aligner + Teacher |
| A3 REC_SPA | PAN Aligner | 2 | 0 | Aligner + Teacher |
| A4 REC_EPS | PAN Aligner | 0 | 0.003 | Aligner + Teacher |
| A5 ALL | PAN Aligner | 2 | 0.003 | **새 RGB Full/Ours** |

**6 case 전체 R1 → 6 case 전체 R2 → 6 case 전체 R3 = 18 runs.**
seed는 기존과 같은 261016101/102/103이다. 한 case의 세 반복을 먼저 실행하지 않는다.
A0/A2/A4는 L_spa가 꺼져 있으므로 RGB loss 변경 자체의 효과를 받지 않는 대조군이다.
과거 mean8 Full/Ours는 RGB 학습 대조군으로 재사용하지 않는다.

모든 run은 W112/D123/BASE, 50K, AdamW WD=.01, A LR=1e-5/T LR=1e-4,
warmup100+cosine floor0, batch48=micro12×4, FP32, epsilon disk R3이다.
기존 bicubic border warp, interp23 reconstruction base, 공통 데이터 augmentation을 유지한다.
시간 제한이나 성능 미달 조기 종료는 없다. 수치/support 실패를 기록하고 독립된 다음 case로 진행한다.

## RGB 정의와 평가

- Aligner 입력: 기존 stored LMS **8채널 전체**. Teacher 출력/GT reconstruction도 8채널 전체.
- L_spa: 보정 PAN과 **GT의 `(R+G+B)/3`**에 Scharr 절대 수평/수직 응답을 적용한 L1.
  WV3 RGB 인덱스는 0-based `[4,2,1]`. GT를 먼저 평균한 뒤 edge를 계산한다.
- Scharr baseline: **stored LMS의 같은 RGB 평균**을 참조한다. GT를 매칭 입력으로 쓰지 않는다.
- validation selector: RGB GT 참조로 재생성한 고정 eligibility의 검증 RRE50→RRE90→ValERGAS→이른 step.
  1080 validation patches, ROI16, sigma .8/.8, ±8px, 추정 실패 12px 처리.
  기존 mean8 validation inventory/receipt는 재사용하지 않는다.
- 공식 RR20/FR20 복원 지표 및 참조는 기존 evaluator 그대로다.
- 각 평가에 **RGB test RRE50**도 기록한다: RR GT 평균, FR stored LMS 평균;
  RR sigma .8/.8와 margin16; FR sigma1.98/.7와 margin24; 각 축 ±8px.
  20개 잔차 벡터 길이의 중앙값이다. 실패 표본은 0이나 누락으로 처리하지 않고,
  failure_count와 함께 RRE50을 null로 둔다. test RRE는 checkpoint 선택에 쓰지 않는다.

## 결과와 Sheet

출력은 `work_dir/panda_wv3_ablation_rgb/PANDA_WV3_STAGE1_RGB_ABL_S3_20261103_v1/s3/WV3/`이다.
각 run의 1K 평가마다 complete checkpoint를 영구 저장하고, 250 step마다 복구용 checkpoint를 저장한다.
고정 50K 결과와 validation-selected 대표 결과를 함께 보존한다.

`WV3-ablations` Experiment 예:

```text
WV3-ABL-S1-A3-REC-SPA R1 (RGB)
WV3-ABL-S1-A5-ALL R1 (RGB)
```

Approach에는 RGB mean/bicubic, loss 계수, LR, C50을 표시한다.
Stage 1의 기존 게시 규칙대로 **run별 validation-RRE 대표 1행**을 유지한다.
HQNR .9585/ERGAS 2.040 기준은 결과 설명용으로 보존하며 validation 선택을 test 기준으로 바꾸지 않는다.
RGB test RRE50과 failure counts는 notes에 포함한다. 다른 캠페인 행과 BM 이후 열은 수정하지 않는다.

## 검증 명령

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s panda_wv3_ablation_rgb/tests -v
python -m panda_wv3_ablation_rgb.release --environment
python -m panda_wv3_ablation_rgb catalog --server s3
```

source manifest 변경 시 기존 RGB checkpoint로 조용히 재개하지 않는다.
같은 이름의 legacy checkpoint를 복사해 넣어도 campaign/config/schema/pins 검증에서 거부한다.
