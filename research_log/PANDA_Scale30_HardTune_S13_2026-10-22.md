# PAN Aligner 양의 scale과 기존 Hard 계수 튜닝

2026-10-22 사용자 요청: 회전 일관성을 제외하고 PAN Aligner를 scale 0.1–0.5로
활용한다. 기존 하이퍼파라미터도 조정하고, 서버별 30시간 실험을 구성한다.
s1의 기존 실험은 정지하고 새 캠페인을 시작한다. s3에는 Git으로 구현과
case를 전달하여 해당 서버 에이전트가 실행한다.

Campaign: `PANDA_S13_SCALE_HARDTUNE_30H_20261022_v1`

## 확인할 가설

PAN 보정 `s*u`를 유지하면서 기존 Hard 가중과 Student 학습률을 조절하면
HQNR ≥ .959와 ERGAS ≤ 2.045를 **동일 checkpoint**에서 만족할 수 있는가?
가능성을 시험하는 계획이며 달성을 전제하지 않는다. 기존 양의 scale 결과가
공동 목표를 안정적으로 달성했다는 증거는 없다. 작은 scale과 기존 학습 조건의
조합에 아직 확인할 여지가 있어 .1/.2를 먼저 실행한다.

Stage2 선정은 HQNR 최대 → ERGAS 최소 → 이른 step이다. 공동 목표 통과는
별도 주석이며 선정 순서를 바꾸지 않는다. Test HQNR로 선정하는 탐색이므로
독립 검증 또는 확증 실험으로 해석하지 않는다. 서버 간 donor가 다르므로
효과는 같은 서버·같은 donor·같은 seed의 대응 비교를 우선한다.

## 서버별 등록 큐

모든 run은 fresh 50K, R1 seed 261018101, R2 seed 261018102다.
기본값은 α=.25, γ=.5, LR=1e-4. H050/H200은 기존 α와 γ를 함께
0.5배/2배 하여 Hard 가중의 강도를 바꾸며, 두 계수의 개별 효과를 분리하는
설계는 아니다. 모든 β=0, Student view=1, consistency λ=0이다.

| 순서 | s1 case | scale | α | γ | LR |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | S020BASE R1 | .2 | .25 | .5 | 1e-4 |
| 2 | S010BASE R1 | .1 | .25 | .5 | 1e-4 |
| 3 | S020LR075 R1 | .2 | .25 | .5 | 7.5e-5 |
| 4 | S030BASE R1 | .3 | .25 | .5 | 1e-4 |
| 5 | S040BASE R1 | .4 | .25 | .5 | 1e-4 |
| 6 | S050BASE R1 | .5 | .25 | .5 | 1e-4 |
| 7 | S020BASE R2 | .2 | .25 | .5 | 1e-4 |
| 8·예비 | S010BASE R2 | .1 | .25 | .5 | 1e-4 |

| 순서 | s3 case | scale | α | γ | LR |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | S020BASE R1 | .2 | .25 | .5 | 1e-4 |
| 2 | S020H050 R1 | .2 | .125 | .25 | 1e-4 |
| 3 | S020LR150 R1 | .2 | .25 | .5 | 1.5e-4 |
| 4 | S020H200 R1 | .2 | .5 | 1 | 1e-4 |
| 5 | S020LR150H050 R1 | .2 | .125 | .25 | 1.5e-4 |
| 6 | S010LR150H050 R1 | .1 | .125 | .25 | 1.5e-4 |
| 7 | S020BASE R2 | .2 | .25 | .5 | 1e-4 |
| 8·예비 | S020LR150H050 R2 | .2 | .125 | .25 | 1.5e-4 |

s1의 1–6은 scale 범위와 낮은 LR를, s3의 1–5는 기준·약한 Hard·높은 LR·
강한 Hard·조합을 비교한다. s3의 5–6은 동일 튜닝에서 scale 효과를 비교한다.
R2는 baseline 재현 변동과 우선 조합의 반복을 확인한다. 8번째는 예비이며,
다른 run도 남은 예산에 50K 전체가 들어오지 않으면 skip을 명시한다.
8개 완주를 보장하기 위해 30시간을 늘리거나 horizon을 줄이지 않는다.

## 고정 방법과 예산

`L = mean_MASK2[(1 + α*d_e + γ*d_e*r_q_bar) * mean_bands(abs(S-GT))]`

- Student PAN은 학습·RR/FR 추론 모두 frozen Aligner의 `s*u`로 실제 bicubic
  warp한다. q는 실제 shifted PAN과 원래 RGB GT 기준으로 계산한다.
- 새 loss·paired view·회전 일관성·TTA·ensemble은 없다. 기존 C4 데이터 증강 유지.
- Teacher는 frozen full-u error map을 제공한다. Soft distillation β=0.
- MS·GT·출력 좌표를 이동시키지 않는다. phase-aware interp23tap base 유지.
- WV3 W112 D123, MS8+PAN1, FP32, AMP/TF32 off, effective48=12×4,
  AdamW, 기존 warmup·50K schedule·35회 평가를 유지하고 peak LR만 등록값으로 바꾼다.
- Stage1은 자체 Mature30 원본 Val-RRE50 → RRE90 → val-ERGAS → 이른 step receipt의
  동일 checkpoint A/T를 재사용한다. s1 donor 50K SHA는
  `62f1c445b8500070429f46503a68fb972a9144a4321fc9da10ccd45891186b8d`.
  s3는 자체 receipt를 사용하며 s1 weight로 대체하지 않는다.
- 전체 RR train 9,714개로 geometry별 ECDF/tau_e를 계산한다. 같은 scale에서는
  α·γ·LR·seed와 무관하게 동일 통계를 재사용한다.
- 서버별 독립 30시간: 비용 측정 시작부터 준비·calibration·preflight·학습·평가·저장
  포함. 마지막 30분 closeout. 재시작해도 시작 시각과 deadline을 초기화하지 않는다.
- 최근 1,000 update p90, 평가/저장 실측, calibration 여유와 12% 안전계수로
  순서대로 admission한다. run당 사전 추정 3.65h는 추정치이며 실측으로 갱신한다.

## 평가와 시트 규칙

기존 원본 참조 RR20/FR20, RR native256 dim_cut21, FR native512 full-frame
unmasked 평가를 그대로 사용한다. FR test를 학습·calibration에 넣지 않는다.
Python DLPan 포트 결과이며 MATLAB 실행과 bitwise 동일하다고 주장하지 않는다.

- Workbook `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE`, `WV3-ablations`,
  sheetId 261001010, A:BL 64열 스키마 유지.
- Stage1은 원본 RRE 선정 donor. Stage2는 run/seed/attempt별 **현재까지 HQNR 최고
  checkpoint 한 행을 갱신**한다. 5K 제한·모든 step 신규 행·별도 final50K 행 없음.
- 최고 HQNR 동률은 ERGAS 최소 → 이른 step. 목표를 만족한 다른 checkpoint를
  HQNR 최고 행으로 바꾸거나 다른 step 지표를 합치지 않는다.
- 전체 metric·checkpoint SHA·step을 묶고, scale·α·γ·LR를 명시한다.
- 비용은 각 실제 scale의 A+S 추론 경로를 고립 GPU에서 측정한다. 독립적인 동일
  구조 측정이며 선정 weight 자체로 latency를 측정했다고 주장하지 않는다.
  profile identity와 선정 checkpoint SHA/step 및 해당 평가까지 학습시간을 연결한다.
- 기존 A열 anchor append, 서버 식별, foreign row 보존, native readback 검증 유지.
  raw 전체 평가·장면별 수치·checkpoint·resume state는 로컬에 보존한다.

## 실행과 산출물

```bash
bash scripts/start_panda_scale30_costed.sh s1
# s3의 별도 에이전트가 자체 서버에서 실행:
bash scripts/start_panda_scale30_costed.sh s3
```

위 명령은 기존 작업을 정지시키지 않는다. 현재 서버의 작업 정지는 사용자 지시에
따라 별도로 처리하고 GPU가 비어 있음을 확인한다. 기존 ZERO30의 정상 종료를
기다리는 선택적 launcher도 있으나 이번 s1 전환은 즉시 실행 명령을 사용한다.

실행 순서: 실제 A+S 비용 → donor/data 확인 → 실데이터 CUDA update/resume 및
실제 PAN warp 검증 → fresh Student 학습 → HQNR-best publisher.
검증용 Student는 본 실험/시트에 사용하지 않는다. 검증 실패 시 증거와 예산을
보존하고 종료한다. 실행 후 소스·설정 수정은 새 캠페인으로 분리한다.

Root: `work_dir/panda_scale30/PANDA_S13_SCALE_HARDTUNE_30H_20261022_v1/<server>/WV3`.

- `campaign_state.json`: 영속 예산, queue 상태, 실측 admission.
- `preflight/passed.json`: 실제 CUDA 검증과 code/evaluator pin.
- `runs/PS30_WV3_<SERVER>_<CASE>_NOSOFT_<R>_S<seed>/attempt_01/`: train/evaluation/model.
- `sheet_publish_panda_scale30/`: projection, last_poll, last_error, terminal_upload.
- [s3 인계](PANDA_Scale30_S3_Handoff_2026-10-22.md), [s1 WIP](../results_log/2026-10-22_WIP_s1_scale30.md).
