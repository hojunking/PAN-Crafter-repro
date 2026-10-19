# PANDA midpoint Stage 2 v2 — s3 / WV3 / U23

기존 Stage 1 donor와 Stage 2 loss를 유지하면서, frozen Aligner가 예측한
전역 이동량을 PAN과 MS에 반씩 나누는 실험입니다. Aligner, Teacher,
Student의 파라미터 구조를 바꾸거나 ensemble/TTA를 추가하지 않습니다.

v2는 **Student의 LRMS 업샘플링만** 기존 bicubic에서 DLPan 방식
`interp23tap`으로 교체합니다. LRMS flip/rotation 이후 ×4 업샘플링하고,
증강에 따른 정수 위상 보정을 적용합니다. PAN/MS/GT/output의 실제 소수
픽셀 이동용 warp는 기존 bicubic 그대로입니다. `interp23tap`은 임의 shift를
받는 warp가 아니며 이번 변경을 “bicubic warp 제거”라고 부르지 않습니다.

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

공통 설정은 fresh W112·D(1,2,3)·P0·`INTERP23_MS_POST_AUG_PHASE_CORRECTED`
Student, `panda_condq`,
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
base       = phase_correct(interp23tap(augmented LRMS, ratio=4), augmentation)
base_mid   = W(base, -u/2)
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

위 식의 original PAN/LMS/GT는 현재 배치의 동일한 flip/rotation이 적용된
영상이며, original은 추가 alignment warp 이전을 뜻합니다. 평가에서는
증강이 없으므로 정수 위상 보정도 identity입니다.

Teacher는 Stage 1에서 학습한 **bicubic LRMS 입력 경로**를 그대로 실행한 뒤 **출력만**
midpoint로 옮깁니다. Teacher에 midpoint 입력을 직접 넣는 별도 동작은
사용하지 않습니다. Teacher와 Aligner는 eval/frozen이며 Stage 2 optimizer에
포함되지 않습니다.

기존 `compute_loss`의 GT-always Hard, Teacher difficulty, selective Soft,
conditional-q 수식과 reduction을 재사용합니다. eT/eS/gate는 `GT_mid` 기준,
q는 `PAN_mid`와 `GT_mid` 기준입니다. τ와 conditional-q CDF는 RR train
전체에서 이 좌표 정책으로 다시 보정합니다. 새 geometry/Teacher transport
binding을 포함하므로 기존 full-shift calibration을 재사용하면 안 됩니다.
v1 midpoint 캐시도 v2 binding으로 재생성합니다. 다만 Student base만 바꾼
것이므로 v1과 비교한 eT/g/q의 수치 자체는 같고, 재생성은 새 실행의
불변 provenance를 위한 것입니다.

MASK2의 유효 sampling support 정책은 유지하되, 영역은 양방향 half-warp와
Teacher 출력 transport에 필요한 유효 영역의 교집합입니다. 기존 단일 PAN
warp mask를 그대로 복사하지 않습니다. ROI8 Scharr stencil이 유효하지
않으면 fail-closed하며 adaptive crop으로 조용히 우회하지 않습니다.

### v2의 Student base 및 증강 위상

Aligner 입력은 기존 **제공 LMS**를 그대로 유지합니다. 새 Student의 base는
LRMS로부터 계산한 `interp23tap` 결과이며, 저장 LMS를 대입한 것이 아닙니다.
기존 frozen Teacher는 학습 당시의 bicubic base를 유지합니다. 동결된
Teacher의 업샘플러까지 바꾸면 동일한 donor 함수가 아니기 때문입니다.

×4 `interp23tap`의 LR sample 위치는 HR 격자에서 `4j+2`입니다. 픽셀 중심
기준 위상 `4j+1.5`와의 차이를 `v=(.5,.5)`로 두고, 실제 Hflip → Vflip →
CCW rot90의 좌표 변환을 `R`로 두면 증강 뒤 업샘플링 결과에 `R·v-v`만큼
정수 roll을 적용해야 함께 증강한 GT/LMS 격자와 위상이 맞습니다. 축별
보정은 0 또는 −1 HR pixel이며, 추가 subpixel/bicubic 보간이 아닙니다.
이를 생략하면 증강 조합에 따라 1 HR pixel 차이가 생깁니다.

필터는 기존 연구용 Torch `interp23tap` 구현의 23-tap 분리형 CDF 계수와
circular padding을 사용합니다. 이 경계 조건은 기존 bicubic의 border
padding과 다릅니다. MASK2는 여전히 실제 translation warp의 4-tap sampling
support를 처리하며 `interp23tap`을 새 23-tap translation warp로 취급하지 않습니다.
이 mask는 interp23tap의 circular 경계 영향이나 CNN 수용영역 전체를
제거한다는 보장이 아닙니다.

v1 midpoint 대비 v2의 차이는 Student 업샘플러와 증강 위상 처리이며,
midpoint 좌표와 loss는 같습니다. 기존 PAN-only 기준군과 비교하면 좌표와
업샘플러가 함께 다르므로 midpoint 단독 효과로 해석하면 안 됩니다. 이번 큐의 3개
case 사이에서는 동일한 v2 업샘플러와 donor를 사용합니다. 새 inverse-output
warp, MS→PAN-only 경로, shift 배율 실험은 이 변경에 포함하지 않습니다.

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
work_dir/panda_midpoint/PANDA_S3_MIDPOINT_STAGE2_20261019_v2/s3/WV3/
```

run ID는 `PMID_U23_...`, Docker container는 `panda-midpoint-wv3-s3-v2`입니다.
동일 checkout에서 v1 학습이 실행 중이면 종료 전 pull/코드 교체를 하지
않습니다. v1의 source hash 검사가 변경을 감지해 중단할 수 있으므로,
v1 완료 후 전달하거나 별도 checkout을 사용해야 합니다.
v1 결과/체크포인트/보정 캐시를 v2에서 이어 학습하거나 덮어쓰지 않습니다.
v2의 geometry/upsampling policy가 checkpoint·calibration·reporting binding에
포함되며, 기존 v1 실행이 있다면 이를 종료하지 않고 새 실행은 idle GPU
조건을 만족할 때 시작합니다.

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

## 구현 검증

2026-10-19 CPU 재검증: midpoint v2 **55개**, 기존 `panda_stage2_condq`
회귀 **93개**, 합계 **148개 테스트 통과**. Shell syntax와 `git diff --check`,
s3 catalog 생성도 통과했습니다.

v1에서 검증한 좌표/loss/Teacher 경로에 더해, v2 검증에는 ratio4 필터
일치, flip/rotation 16개 조합의 정수 위상 보정, 입력 gradient, 실제
Student residual base 및 frozen Teacher bicubic 경로 보존을 포함합니다.
또한 RR256/FR512 추론, calibration/checkpoint의 이전 policy 거부,
원 평가 참조 불변, 로컬 outbox/업로드 행의 U23 provenance 및 shell syntax를
확인했습니다. Docker 실행 검사는 stub 기반으로 mount·v2 경로·실패 시
학습/업로더 미기동을 확인했으며 실제 컨테이너 기동은 하지 않았습니다.

```bash
python -m panda_midpoint smoke
python -m unittest discover -s panda_stage2_condq/tests -t .
```

이는 실제 s3 GPU 학습이나 Google Sheets 전송 완료를 뜻하지 않습니다.
s3 자산·Docker 환경의 사전 검사는 위 실행 명령에 포함되어 있으며,
학습 및 실시간 업로드는 변경사항을 전달한 후 **s3 세션에서** 시작합니다.
