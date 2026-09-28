# B01 상세 명세 — RB01 q 정보성 + RB02 shift 검사

작성일: 2026-09-28  
대상: s1 / s3 / s5, 각각 두 repeat, WV3  
상태: **SPECIFIED / NOT_IMPLEMENTED_AS_A_TRAINING_RUNNER / NOT_STARTED**

## 1. 바로 구현할 작업과 건드리지 않을 것

RB01의 네 학습 조건과 RB02의 두 inference mode만 구현한다. 제출 PDF·Table3·메인 결과·기존 sheet를 고치지 않는다. 새 checkpoint, raw, metric, 비교표는 별도 namespace에 둔다. 기존 PNG renderer 복구·calibrate·논문 수치에 맞춘 후처리를 하지 않는다.

본 문서의 모든 경로는 **제안 namespace 또는 source의 상대경로**다. s1/s3/s5 로컬 absolute root, checkpoint 존재, GPU 사용 가능 상태는 실행 담당자가 검증한다. `planning/experiment_registry.json`의 Teacher SHA가 null인 것은 고정 reference의 실제 bytes를 아직 회수하지 않았기 때문이다. null을 통과시키거나 다른 Teacher로 대체하지 말고 공통 binding으로 채운 뒤 실행한다.

## 2. 공통 source와 계산식

원 config: `config/FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1.yaml`.

Teacher source:

```text
work_dir/FH12_S1_T_P0_W112_D123_WV3_S71001_FRESH50_v1/
  candidates/50000/model.safetensors
  candidates/50000/identity.json
  meta/config.resolved.yaml
```

F1의 원 bridge/reference/calibration과 data manifest를 복사하거나 로컬 동일 bytes를 바인딩한다. 원 run의 `meta/config.resolved.yaml`과 `work_dir/_fh20r1/s1/imported_refs/F1/bridge_manifest.json`은 검색 출발점이며, 거기서 실제 reference와 q asset 경로를 해석한다. source 서버 경로를 문자열 치환해 다른 서버 자산을 같은 reference라고 가정하지 않는다.

원 `fh12.model.build_model`, `fh12.model.sync_frontend`, `fh12.losses.student_losses`, `fh12.losses.routed_student_backward`, `fh12.training.BatchStream`의 수치 의미를 재사용한다. 과거 budget/deadline가 박힌 campaign controller는 다시 기동하지 않는다. 새 wrapper에서 운영 부분만 분리하고 baseline 수치 동등성부터 확인한다.

표기:

```text
H_n = mean_p [(1 + alpha*d_T) * e_S]
K_n = mean_p [beta*(1-d_T)*a_T*abs(S-T)_channel_mean]
E_n = 원 signed horizontal/vertical GT edge loss
L_U = mean_n [H_n + K_n + lambda_E * w_E,n * E_n]
L_A = mean_n [w_A,n * H_n]
```

`d_T`, `a_T`, Teacher, GT, 모든 weight는 원 구현대로 detached다. live Student residual의 gradient는 유지한다. U는 L_U, A는 L_A로 각각 미분하며 beta를 중복 곱하지 않는다. 네 조건 모두 alpha=1, beta=.1, lambdaE=.002이며 w_E=w_A다. 새 case에서 바뀌는 것은 그 공통 geometry weight다.

`q_ref`와 `tau_rec`를 과거 G23 값으로 대입하지 않는다. 이번 reference F1의 실제 calibration을 사용한다. q/e 자산과 train×view 대응을 먼저 검증한다.

## 3. paired initialization과 stream

각 서버 repeat마다 seed를 하나 고정한다. 네 case가 공유하는 초기 U state, F1 A clone, 50K 전체 sample/augmentation stream을 등록한다. 각 case는 그 초기값으로 새로 시작하며 앞선 case의 학습 결과를 이어받지 않는다.

- 모델용 seed, sample/view stream seed는 registry의 Student seed에 연결한다.
- q permutation/tie-breaking은 별도 전용 RNG를 쓴다. data loader/global RNG 상태를 소비하지 않는다.
- 동일 repeat 네 case의 U/A init, Teacher, data, LP, stream hash는 같아야 한다.
- 다른 repeat 또는 다른 서버는 별도의 Student seed를 쓴다.
- runtime 사전 검사는 score가 아닌 identity와 한 batch의 numerical parity를 확인한다. 같은 source를 재현한 원 T0/F1 이름만으로 동등성을 선언하지 않는다.

## 4. RB01 weight 구성

### 4.1 QFULL

sample n과 원 training view v의 raw q에서 `s[n,v]=q_ref/(q_ref+q[n,v])`를 사용한다. 이 weight로 원 L_U/L_A 계산을 그대로 수행한다. 원 자산에 weight가 이미 들어 있으면 그것이 raw q에서 생성된 값인지 확인하고 변환을 다시 적용하지 않는다.

### 4.2 QMEAN

전체 training sample×실제 view의 s를 한 번 평균하여 `s_bar`를 고정한다. 모든 sample의 w_E와 w_A에 이 상수를 넣는다. `s_bar=.5`라고 가정하지 않는다. calibration subset의 평균과 full training view population 평균을 혼동하지 않는다. q asset이 calibration subset만 덮으면 같은 frozen recipe로 나머지 train view를 계산해 별도 cache로 완성한다.

이 control은 s 평균을 보존하지만 variance는 제거한다. FULL의 평균을 1로 재정규화하거나 A LR/lambdaE를 보정하지 않는다.

### 4.3 QSHUF

Teacher는 그대로 두고 **기존 s 값의 sample 연결만** 바꾼다.

1. training sample/view에 대해 `e_bar[n,v]=mean_p e_T[n,p,v]`를 고정 reference로 계산한다. augmentation, input/clip 단위는 원 훈련 error 정의와 같다.
2. 각 view별 e_bar quintile을 train에서만 정한다. quantile tie는 고정된 source-index tie-break로 처리한다.
3. `(view, e-quintile)` 안에서 sample index를 별도 RNG로 permutation한다. 가능한 group은 자기 자신과 매칭되지 않는 derangement를 사용한다. size1 group은 변경 불가로 기록한다.
4. 각 sample에 `s[pi_v(n),v]`를 배정한다. mapping은 50K 동안 고정하며 매 batch/epoch 다시 섞지 않는다.
5. 원/셔플 weight sorted-array, min/max, mean/SD가 전체와 각 view/stratum별로 같은지 확인한다. 새 permutation은 repeat seed에 결합해 결정한다.

핵심 비교는 q 값의 분포나 e 수준이 아니라 **해당 sample과 q의 대응**이다. 한 번의 permutation에만 결론이 의존하지 않도록 여섯 Student repeat에는 여섯 고정 permutation을 사용한다. permutation 난수 자체를 독립적인 7번째 학습 반복으로 세지 않는다.

### 4.4 QESUR

각 view 내 e_bar가 낮은 sample부터 정렬하고, 같은 view의 원 s를 높은 값부터 정렬해 순서대로 대응한다. 따라서 Teacher reconstruction error가 작은 sample에 큰 geometry weight를 주면서 **원 s와 완전히 같은 weight multiset**을 유지한다.

동점은 고정 source-index 또는 별도 지정 RNG로 처리하고 mapping SHA를 보존한다. 이 mapping은 train에서만 만들고 test/val 성능을 보며 방향이나 quantile을 바꾸지 않는다. geometry 신호만 e-derived surrogate로 바꾸며 pixel-level hard/soft weighting의 e, d, a는 원 코드 그대로다.

QFULL/QSHUF/QESUR의 marginal distribution이 같아도 minibatch별 조합과 50K에서 실제 소비한 sample 횟수 때문에 consumed-weight 평균은 미세하게 다를 수 있다. 이를 로그로 남기되 batch별 평균 재정규화로 제거하지 않는다. 그 정규화는 별도의 방법 변경이다.

## 5. 저장·학습·선택 규약

24개 run은 registry의 ID를 그대로 쓴다. precision FP32, native64/16, batch48, Student fresh50K, 원 AdamW/warmup100/cosine을 유지한다. 고정 F1 Teacher는 eval+requires_grad_false다. Stage-2에서 새로운 shift-consistency loss나 Student augmentation을 추가하지 않는다.

val grid는 원 1010 간격 및 마지막50000을 재사용한다. primary EXACT_50000, secondary val ERGAS minimum/동률이면 이른 step이다. 테스트 HQNR을 50개 checkpoint마다 계산해 선택하지 않는다. 모든 native metrics는 선택된 동일 A/U checkpoint에서 계산한다.

val/RR/FR 데이터는 optimizer update에 사용하지 않는다. F1 calibration은 원래 training subset에서 만들어진 고정 자산이며, 그 subset도 기존 전체 train의 일부이므로 Student training에서 임의 제외하지 않는다. val은 selection 용도이며 기존에 반복 개발한 test set을 untouched 새 데이터라고 표시하지 않는다.

최소 진단 시점: update 0, 1000, 10000, 25000, 50000. 고정 train probe와 검증용 sample에서 다음을 기록한다. 진단은 sampler RNG나 optimizer를 변경하지 않아야 한다.

- q/e/s quantile 및 sample mapping.
- `a_T>0` 비율, mean soft weight, Teacher/Student e, Student가 Teacher보다 좋은 위치 비율.
- H/K/E의 raw 및 weighted 값, U/A gradient norm, 실제 optimizer update norm.
- A correction의 분포, seed별 drift, native residual/edge loss.

T/학생 probe forward는 train update와 분리하고 진단으로 모델 buffer가 바뀌지 않게 한다. 진단 실패로 본 학습을 다른 seed로 재시작하지 않는다.

## 6. RB02: native와 controlled-shift 평가

### 6.1 주 목적과 source

각 RB01 run의 EXACT_50000 Student를 사용한다. 네 train case×여섯 independent Student가 source다. A_ON과 A_ZERO는 같은 checkpoint에서의 inference mode이며 새로운 학습 case로 세지 않는다.

현재 원 `FH12Model.forward`는 `aligner_enabled`와 `delta_override`를 받는다. 실행 wrapper의 signature와 동등성을 확인한 뒤 `delta_override=zeros` 또는 동일한 source 경로로 correction을 0으로 고정한다. **PAN-only bypass로 끝내지 않고 PAN/LP/HP의 synchronized frontend 전체에 같은 zero correction을 적용**한다. `A_ZERO` 때 estimator를 진단 목적으로 계산해도 reconstruction에는 사용하지 않는다.

추가 훈련, TTA, 보정량 fitting, error에 맞춘 warp, RR GT 입력은 금지한다. 제출 모델의 원 checkpoint와 Teacher F1을 함께 점검하고 싶으면 `HISTORICAL_DIAGNOSTIC`으로 분리한다. 같은 고정 F1을 여섯 번 복사한 값을 여섯 Teacher seed 반복이라고 집계하지 않는다.

### 6.2 shift 입력의 생성

`planning/shift_grid_v1.json`은 총49개 `(dy,dx)`를 갖는다. L2 반경 0.25/.5/1/2/3/4와 45도 간격8방향이며 0은 한 번만 포함한다.

PAN에만 알려진 shift를 적용하고 MS/GT는 원 좌표를 유지한다. 원 `warp_pan`의 샘플 좌표 부호, 단위, padding, align_corners를 impulse/ramp로 확인한다. 실제 pixel 이동의 부호와 grid parameter 부호를 혼동하지 않는다. Eq.(7)의 `u_eps+eps-u0`가 그 구현에서 맞는지 검사해 manifest에 기록한다.

입력 PAN을 한 번 이동시킨 뒤 **이 이동된 PAN으로 같은 recipe의 LP를 다시 만든다**. 원 native LP를 그대로 두면 stale sidecar가 위치 정보를 누설한다. 그 다음 모델의 동일-grid PAN/LP warp와 HP subtraction을 수행한다. 원 MS를 이동하거나 PAN을 두 번 이동해 원하는 숫자를 만들지 않는다.

zero shift는 원 PAN과 원 native LP를 재사용해 native prediction과 대조한다. fractional shift 후 reverse warp가 interpolation 때문에 완전한 identity라고 가정하지 않는다. 큰 shift에서 padding·smoothing 자체의 영향도 발생하므로 비교 조건을 동일하게 유지한다.

### 6.3 지표와 support

1. **Native RR/FR:** 원 프로토콜 그대로. RR `[20:-21,20:-21]`, FR full512/raw original PAN·LMS. 모든 20 scene 사용.
2. **Stress RR:** full256 입력으로 inference 후 고정 `[32:-32,32:-32]`의192×192 구간에서 ERGAS, PSNR, SAM, edge error를 계산한다. 원 paper 표의 평가 crop을 바꾼 것이 아니라 별도 stress diagnostic이다. stress PSNR/SAM/ERGAS는 같은 192×192 support를 사용한다.
3. **Relative response:** 정상 Aligner의 `||u(P_eps,M)+eps-u(P,M)||_1`을 HR pixel 단위로 저장한다. coordinate 축은 dy/dx로 명시하고, L1 sum/mean 중 어떤 축 reduction인지 고정한다. 여기서는 **dy/dx absolute sum**을 기록하고 원 q의 coordinate-mean 값과 구분한다.
4. 각 curve에서 r=0 대비 degradation을 계산한다. 변형 input의 점수가 낮다는 사실만이 아니라 A_ON/A_ZERO 차이, seed별 평균·분산·failures를 본다.
5. stress 구간에서 model/shift별 유리한 crop을 다시 고르지 않는다. Aligner는 무조건 정해진 범위로 clamp된다고 가정하지 않는다. 실제 correction이 매우 커서 ROI 안에 invalid sampling이 생기면 geometric coverage와 해당 값을 기록하며 scene을 숨기거나 보정량을 새로 clamp하지 않는다.
6. FR에 인위적 shift를 넣어 HQNR reference를 교체하는 평가를 이번 필수 작업으로 하지 않는다. 이 경우 metric 의미가 달라질 수 있으므로 native FR만 기본 표에 남긴다.

정상 correction을 0으로 만들면 본래 학습 때와 입력 분포가 달라진다. A_ON−A_ZERO는 **이미 학습된 모델이 inference correction에 얼마나 의존하는지**를 측정하며, “no-align architecture보다 우월함” 또는 “consistency loss가 유효함”을 단독 증명하지 않는다.

Native pair의 실제 displacement=0이라고 놓지 않는다. relative-shift 상쇄, 원래의 absolute registration, 최종 reconstruction error를 서로 구분한다. q와 같은 식만 줄었다는 이유로 q의 유용성을 순환적으로 증명하지 않는다.

### 6.4 e와 q의 역할 진단

train q/e는 cache 구축과 weight 정의에만 쓴다. 추가학습 없이 reference F1의 고정 validation probe를 별도로 읽어 e-비슷한 sample 안에서 q와 stress degradation의 관계를 본다. 예: validation 원본 patch에서 outcome을 보기 전에 고정 seed로 최대256개를 고른다. 그 subset/seed/scene ID를 모든 서버가 공유한다.

64×64 validation probe는 원 val support에서 계산하고, 256×256 RR stress와 섞지 않는다. q probe E와 스트레스 grid의 중복/비중복을 표시한다. 학습 shift 분포가 연속적인 경우 대각선이라고 “미학습 shift”라고 단정하지 않는다. q 낮음/높음 구간의 reconstruction 악화량, 저-texture 실패, large correction을 함께 저장한다.

이 reference-only 분석의 sample 수나 동일 Teacher를 서버에서 반복 계산한 횟수를 Student-seed n=6의 표에 추가하지 않는다. 여섯 source model의 QFULL/QMEAN/QSHUF/QESUR stress 결과가 주 실험이다.

## 7. native raw 및 stress 보존

각 source run에서 raw 저장 규약은 다음과 같다.

```text
RB01/<server>/<repeat>/<case>/
  case.json
  bindings.json
  init_manifest.json
  stream_manifest.json
  meta/                         # code/data/Teacher/precision/timing
  checkpoints/                  # exact50K, val-selected, resume pointer
  diagnostics/
  native/
    Ours_raw/WV3/reduced/pred/testimg0.npy ... testimg19.npy
    Ours_raw/WV3/full/pred/testimg0.npy ... testimg19.npy
    references/WV3/reduced/gt/...    # 동일 reference는 공통 보존본/hash로 연결 가능
    metrics.json
    per_scene.csv
  stress/
    A_ON/per_scene.csv
    A_ZERO_INFERENCE_ONLY/per_scene.csv
    curve_summary.json
    shift_inputs_manifest.json
  completion.json
```

native prediction은 float32 CHW DN0–2047, RR8×256×256/FR8×512×512, source H5 row order다. 24개 model의 native예측만 약4.69GiB(배열 본체 기준)다. GT/PAN/LP는 동일 data identity를 공통 cache로 보존하여 불필요한 복제를 줄인다.

49×20×2×24 전체 stress prediction을 영구 저장할 필요는 없다. 모든 조건의 per-scene scalar, prediction digest, source/shift/checkpoint identity와 재생성 가능 정보는 남긴다. 그림용 raw는 outcome을 보기 전에 고정한 scene/shift subset을 저장한다. 실패 case raw는 추가 보존 가능하며 “대표적 향상”만 남기지 않는다.

Native zero shift는 native inference 값을 재사용할 수 있지만 stress ROI로 metric을 다시 계산해야 한다. 같은 raw의 두 다른 metric crop을 두 독립 실행이라고 세지 않는다.

## 8. 완료 조건과 실패 처리

각 서버의 두 repeat, 모든 네 case가 registry에 남아 있어야 한다. full-state 기술 오류는 가능한 같은 run에서 resume한다. 완료 파일이 있다면 checkpoint/data/metric SHA를 검증하고 중복 학습·중복 등록하지 않는다.

어떤 변형이 실제로 발산하면 failure를 기록한다. case를 삭제하거나 “여섯 성공 결과”가 나올 때까지 다른 seed를 추첨하지 않는다. 기술적 원인과 수치적 실패를 분리하고 완료 n을 보고한다. 다른 서버의 성공 결과를 실패한 서버의 반복으로 복사하지 않는다.

최소 검증:

- same-block 공통항 identity, QFULL baseline one-batch parity, exact resume check.
- QMEAN 실제 training mean, QSHUF/QESUR multiset 보존과 deterministic mapping.
- Native RR20/FR20 파일·dtype·finite·source-index 검사.
- RB02 source step=50000, mode=2, grid=49, 모든20scene 처리 여부.
- primary/secondary alias, actual updates, Teacher frozen hash, no calibration from test.
- independent model n과 inference/scene/shift 측정 수의 분리.

완료 후 local control은 `STOP_FOR_REVIEW`. 과거 indefinite runner로 자동 복귀시키거나 RB03를 자동 실행하지 않는다. 기존 작업의 보존/재개가 필요한 경우 그 실행 담당자의 승인된 운영 절차를 별도로 따른다.

## 9. 반환 보고 양식

```text
B01 / server / two seeds
RB01: QFULL, QMEAN, QSHUF, QESUR 각각 completed/failed/blocked
RB02: 각 source checkpoint A_ON/A_ZERO curve complete 여부
Teacher/data/calibration/source SHA 동등성
run별 native metrics, QFULL 대비 paired delta
shift곡선, e/q 분포, gate/gradient/update 진단
기술 오류·발산 및 재시도 기록
실제 train/eval/IO 시간, GPU/runtime
결과 ZIP 경로·bytes·SHA256·사용자 접근 링크
다음 batch: STOP_FOR_REVIEW
```

metadata나 계획 MD만 담긴 ZIP을 raw/실험 완료 package로 표시하지 않는다. 결과가 좋아야 완료인 것이 아니라, 사전 지정 범위를 수행하고 관측/실패를 정확히 보존했을 때 기술적으로 완료한 것이다.
