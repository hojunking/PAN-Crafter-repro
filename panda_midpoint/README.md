# PANDA midpoint Stage 2 — s3 / WV3

기존 Stage 1 donor와 Stage 2 loss를 유지하면서, frozen Aligner가 예측한
전역 이동량을 PAN과 MS에 반씩 나누는 실험입니다. Aligner, Teacher,
Student의 파라미터 구조를 바꾸거나 ensemble/TTA를 추가하지 않습니다.

## 기본 큐

세 case는 동일한 s3 로컬 mature donor와 같은 R1 seed `261018101`을
사용합니다. 기본 실행은 **3 case × 1 seed = 3 run**입니다.
R2(`261018102`)는 명시적으로 추가할 때만 실행하며 기본 큐에는 없습니다.

| 순서 | case | α | β | γ | 선택 근거 |
|---:|---|---:|---:|---:|---|
| 1 | `MID_B005` | .25 | .05 | .5 | 기존 M_Q 계수와 대응하는 기준 |
| 2 | `MID_B010` | .25 | .10 | .5 | β=.05와 .20 사이의 보간 탐색 |
| 3 | `MID_B020` | .25 | .20 | .5 | 기존 M_B2 계수; s4에서 유망했던 stronger Soft |

β=.10은 α=.25와 결합한 구성이 과거에 확증됐다는 뜻이 아닙니다.
세 설정 중 어느 것도 통계적으로 입증된 최적 설정으로 취급하지 않습니다.
동일 seed는 loss 계수 사이의 대응 비교를 위한 것이며, seed 재현성까지
입증하는 설계는 아닙니다.

공통 설정은 fresh W112·D(1,2,3)·P0·BICUBIC_MS Student, `panda_condq`,
AdamW peak LR `1e-4`, weight decay `.01`, warmup100, cosine50K,
effective batch48(micro12 × accumulation4), FP32/no TF32입니다.
기존 H/V flip probability1 및 rot90 증강도 유지합니다. 새 LR, EMA,
추가 consistency loss, Aligner 미세조정은 섞지 않습니다.

## 이동과 loss의 좌표

warp 규약은 `W(I,u)(x) = I(x+u)`입니다. 매 입력마다 **원 PAN과 원 제공
LMS**로 frozen Aligner의 `u=(dy,dx)`를 계산합니다. 고정하는 것은 Aligner
가중치이지 모든 영상에 공통으로 쓸 하나의 shift가 아닙니다.

```text
u          = A(original PAN, provided LMS)
PAN_mid    = W(original PAN, +u/2)
base_mid   = W(bicubic(original LRMS), -u/2)
GT_mid     = W(original HRMS GT, -u/2)
Teacher_MS = T(W(original PAN, +u), original LRMS, provided LMS)
Teacher_mid = W(Teacher_MS, -u/2)
residual   = Student_U(concat(PAN_mid, base_mid))
Student_mid = base_mid + residual
```

위 마지막 두 줄은 입력과 residual 합산 기준을 나타내는 개념식이며,
실제 Reconstructor가 residual base를 한 번만 더해야 합니다.
shift 단위는 현재 PAN 격자의 pixel입니다. RR/FR에 임의의 ×4 또는 ÷4를
추가하지 않습니다. PAN과 MS 원본에서 각각 직접 warp하며, 이미 이동한
PAN을 다시 반대로 이동시키는 방식으로 midpoint PAN을 만들지 않습니다.

Teacher는 Stage 1에서 학습한 입력 경로를 그대로 실행한 뒤 **출력만**
midpoint로 옮깁니다. Teacher에 midpoint 입력을 직접 넣는 별도 동작은
사용하지 않습니다. Teacher와 Aligner는 eval/frozen이며 Stage 2 optimizer에
포함되지 않습니다.

기존 `compute_loss`의 GT-always Hard, Teacher difficulty, selective Soft,
conditional-q 수식과 reduction을 재사용합니다. eT/eS/gate는 `GT_mid` 기준,
q는 `PAN_mid`와 `GT_mid` 기준입니다. τ와 conditional-q CDF는 RR train
전체에서 이 좌표 정책으로 다시 보정합니다. 새 geometry/Teacher transport
binding을 포함하므로 기존 full-shift calibration을 재사용하면 안 됩니다.

MASK2의 유효 sampling support 정책은 유지하되, 영역은 양방향 half-warp와
Teacher 출력 transport에 필요한 유효 영역의 교집합입니다. 기존 단일 PAN
warp mask를 그대로 복사하지 않습니다. ROI8 Scharr stencil이 유효하지
않으면 fail-closed하며 adaptive crop으로 조용히 우회하지 않습니다.

### 의도적으로 유지하는 기존 MS base

Aligner 입력은 **제공 LMS**, 복원기의 base는 **torch bicubic LRMS**입니다.
현재 두 upsampling 경로에 알려진 위상 차이가 있지만 이번 실험에서
LMS-base로 교체하거나 추가적인 고정 위상 보정을 넣지 않습니다. 그런
변경은 midpoint 효과와 다른 변경을 섞게 됩니다. 이동은 기존 bicubic
base 전체에 동일하게 적용합니다.

## donor와 실행

기존 s3 `panda30r` 캠페인의 M donor만 사용합니다. 즉 로컬 early A1/R1
Aligner를 고정하고 Teacher를 누적50K까지 학습한 최종 mature checkpoint
입니다. 세 case에 같은 checkpoint/SHA를 pin합니다.

- 다른 서버 donor를 가져오지 않습니다.
- M donor가 없거나 provenance가 맞지 않으면 명확히 실패합니다.
- Stage 1 신규 학습이나 자동 donor fallback을 시작하지 않습니다.
- 기존 캠페인/실행 프로세스를 종료하거나 그 결과 디렉터리를 덮어쓰지 않습니다.
- 임의의 20h/30h/48h 제한을 상속하지 않으며 등록한 각 run은50K가 목표입니다.

s3에서 필요한 기존 데이터, evaluator, Docker image, 로컬 업로드 자격증명,
donor가 준비된 상태에서 실행합니다.

```bash
bash scripts/run_panda_midpoint_docker.sh s3
```

`git pull` 자체가 학습을 시작하지는 않습니다. 현재 작업 서버가 s1이고
s3 원격 접속 경로가 없다면 이 명령은 **s3에서 별도로 실행해야 하며**,
로컬 구현 완료를 s3 학습 시작으로 표현하지 않습니다.

결과 루트는 다음과 같이 기존 캠페인과 분리됩니다.

```text
work_dir/panda_midpoint/PANDA_S3_MIDPOINT_STAGE2_20261019_v1/s3/WV3/
```

## 평가, 선택, 업로드

배포 모델은 A+S만 포함하고 GT/Teacher/calibration 없이 추론합니다.
RR와 FR 모두 **midpoint native 출력 그대로** 평가합니다. 원 MS 좌표로
돌리는 inverse output warp를 추가하지 않습니다.

공식 metric의 참조는 기존 원 PAN, 원 LMS, 원 GT 그대로입니다.
RR20 `dim_cut21`, FR20 full512, unmasked 평가를 유지합니다. 학습용으로
이동한 GT나 support mask를 공식 평가에 적용하지 않습니다. 따라서 공식
수치에는 원 참조와 midpoint 출력 사이의 위치 차이도 반영됩니다.

- 기존 checkpoint 정책: HQNR≥`.959`에서 최소 ERGAS; 미달이면 최대 HQNR.
  동점 처리 역시 기존 selector를 유지합니다.
- 기존 sheet 정책: HQNR≥`.960` 및 ERGAS≤`2.045`를 함께 통과한 step들,
  미통과 run은 best-HQNR 행을 게시합니다.
- 현재 workbook의 `WV3-ablations` 경로와 로컬 결과/outbox 기록을 유지합니다.
  업로드 실패를 학습 성공이나 업로드 성공으로 오인하지 않습니다.
- full evaluated checkpoints와 동일 상태의 RR/FR metric을 보존합니다.

이 실험은 midpoint 학습의 탐색입니다. 기존 모델 출력을 `-u/2` 이동시킨
s1 사전 검사에서는 지표가 악화된 사례가 있으므로 동시 개선을 전제하지
않습니다. 이동된 GT의 subpixel 보간은 감독 신호도 바꿉니다. train loss의
절대 감소만으로 성공을 판단하지 말고 원 참조의 공식 metric 및 기존
s3 M_Q 대응 결과와 함께 해석해야 합니다.

참고: [기존 loss 분석](../results_log/2026-10-19_panda-stage2-loss-analysis.md),
[s1 이동 사전 검사](../results_log/2026-10-19_s1_ms-shift-alignment-feasibility.md).

## 구현 검증 — 2026-10-19

신규 CPU 테스트 42개와 기존 `panda_stage2_condq` 회귀 테스트 93개를 통과했습니다.
실제 W112/D123 모델, RR256/FR512 추론 크기, nonzero output head로 이동 방향과
residual base를 검사했습니다. synthetic H5 전체 calibration, old-geometry cache
거부, frozen A/T gradient isolation, 완전 checkpoint 재개, 원 평가 참조 불변,
로컬 outbox 및 업로드 행 구성도 검증했습니다. shell syntax 검사도 통과했습니다.

```bash
python -m panda_midpoint smoke
python -m unittest discover -s panda_stage2_condq/tests -t .
```

이는 실제 s3 GPU 학습이나 Google Sheets 전송 완료를 뜻하지 않습니다.
s3 자산·Docker 환경의 사전 검사는 위 실행 명령에 포함되어 있으며,
학습 및 실시간 업로드는 변경사항을 전달한 후 **s3 세션에서** 시작합니다.
