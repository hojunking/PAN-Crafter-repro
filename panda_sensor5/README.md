# SENSOR5: QB / GF2 Stage 1 + Stage 2

Campaign: `PANDA_S13_SENSOR5_RGB_L4_STAGE12_20261026_v2`

Method: `PANDA-SENSOR5-RGB-L4-DIFFUBIL-HARDONLY-v2`

2026-10-26 추가 사용자 결정으로 QB/GF2부터 Stage 1·2의 RGB 구조 참조와 Lanczos-4 warp를 통일한다. v1 가중치·calibration을 이어 쓰지 않는 별도 v2 캠페인이다. 서버별 **새 Stage 1 한 번 + 새 Student 다섯 번**을 실행한다. 현재 WV3 run의 등록 horizon까지 완료한 다음 시작하고, 센서별 여섯 run을 모두 마치면 **남은 WV3 TUNE30 → HARD36**을 자동 재개한다. 새 센서 실험을 기다리는 동안 WV3 다음 run은 시작하지 않는다.

| 서버 | 데이터셋 | 공동 목표 | Stage 1 | Stage 2 합계 |
|---|---|---|---:|---:|
| s1 | QB | HQNR ≥ 0.9315, ERGAS ≤ 3.5 | 50K | 170K / 5 run |
| s3 | GF2 | HQNR ≥ 0.9645, ERGAS ≤ 0.548 | 50K | 180K / 5 run |

새 센서 큐에는 시간 상한을 두지 않는다. WV3의 기존 30h/36h 예산은 원래 설정과 기록에 보존하되, 사용자 요청대로 남은 등록 case 전체를 재개하도록 별도 continuation 기록에 budget override를 남긴다. 센서 실험 대기 시간은 WV3의 active elapsed에 포함하지 않는다.

## 실험 근거와 비교 범위

WV3의 유력 조건 `S100DIFFUBILC30`을 센서별로 이식한다. Aligner scale 1, corrected PAN 차영상 입력, bilinear upsample, W112/D123, LR 1e-4, cosine 30K, α=.25, β=0, γ=.5가 중심점이다. 각 센서에서 같은 Stage 1 donor와 같은 Student seed를 쓰므로 LR·horizon·추가 Hard 강도의 영향을 비교할 수 있다. 단일 seed의 탐색이며, seed 재현성 또는 타 센서 목표 달성을 보장하지 않는다.

## Stage 1: 센서마다 처음부터 학습

- Aligner와 Teacher를 함께 50K 학습한다. 기존 WV3 가중치나 다른 서버의 donor는 가져오지 않는다.
- 영상당 하나의 전역 `(dy, dx)`를 예측한다. 정합 기준은 **GT RGB [2,1,0] 단순 평균의 Scharr**로 Stage 2 q/g와 같다. Aligner의 MS 입력과 Teacher/Student 출력·복원 손실은 NIR을 포함한 전체 4밴드를 유지한다.
- Teacher 입력은 corrected PAN 1채널 + `I_lrms` 4채널, 출력은 HRMS 4밴드이다. `I_lrms`는 LRMS를 interp23으로 확대한 재구성 base다.
- Teacher W112/D123의 기존 BASE down/up 구조를 유지한다. Stage 1 native 보정·ε 주입, Stage 2 frozen Teacher·Student 모두 정규화된 8-tap Lanczos-4와 whole-sample reflection을 사용한다.
- LR A=1e-5, T=1e-4; 100 update warmup + cosine 50K; weight decay .01; FP32.
- loss: reconstruction 1, absolute alignment 2, shift equivariance .003. ε는 반경 3 RR PAN pixel의 uniform-area disk다. 이 equivariance는 기존 **이동** 일관성이고 rotation consistency loss가 아니다.
- effective batch 48, microbatch 12. 기존 paired flip/rot90 augmentation은 유지한다.
- seed `261016101`. 1K마다 평가·저장하고 50K 전체 완료 후 고정된 validation 집단에서 **RRE50 → RRE90 → validation ERGAS → 이른 step** 순서로 donor를 선택한다. RR20/FR20 test 수치는 donor 선정에 쓰지 않는다.
- 이 donor의 Aligner·Teacher를 동결해 모든 Stage 2 case가 공유한다.

## Stage 2: 데이터셋별 다섯 case

| 순서 | case 접미사 | 초기 LR | cosine horizon | α | β | γ | 확인할 축 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | BASE | 1e-4 | 30K | .25 | 0 | .5 | WV3 유력 조건 이식 |
| 2 | LR050 | 5e-5 | 30K | .25 | 0 | .5 | LR 절반 |
| 3 | LR200 | 2e-4 | 30K | .25 | 0 | .5 | LR 두 배 |
| 4 | LONG | 1e-4 | QB 50K / GF2 60K | .25 | 0 | .5 | 더 긴 감쇠·학습 |
| 5 | HARD4 | 1e-4 | 30K | 1 | 0 | 2 | 추가 Hard 두 항을 함께 4배 |

case 이름 앞에 `QB_` 또는 `GF2_`가 붙는다. 모두 R1, seed `261018101`, fresh Student다. LONG도 BASE checkpoint의 연장이 아니라 처음부터 긴 cosine으로 학습한다. HARD4는 α·γ 개별 효과를 분리하는 실험이 아니며, 기본 L1 계수 1은 바꾸지 않는다.

공통 경로:

```text
u = frozen Aligner(PAN, native LMS)         # full scale 1
PANc = Lanczos4(PAN, u)
Student input = [PANc, PANc - PAN, I_lrms]  # 6 channels
Student output = HRMS                     # 4 bands
loss = mean[(1 + α d_e + γ d_e r̄_q) · band_mean(|Student - GT|)]
```

Student는 W112/D123, UBIL(bilinear ×2 + 3×3 conv), 기존 stride-2 convolution downsample을 쓴다. 차영상 stem의 추가 열은 0으로 초기화한다. Teacher difficulty와 conditional discrepancy는 같은 L4 corrected PAN 경로에서 계산한다. difficulty는 전체 4밴드 Teacher 복원 오차, q/g는 RGB 평균의 Scharr 구조 참조를 쓴다. β=0이므로 Teacher 출력으로 Soft reconstruction distillation을 하지 않는다.

Student warmup 100 update, 최소 LR 0, weight decay .01, FP32, effective batch 48/microbatch 12. rotation consistency loss, TTA, SWA/EMA, band-weighted loss, 새 정합 보조 loss를 추가하지 않는다. 공식 평가는 원본 PAN·원본 좌표 기준의 단일 forward다.

## 센서와 데이터 처리

| 항목 | QB | GF2 |
|---|---|---|
| 밴드 / 확대율 | 4 / ×4 | 4 / ×4 |
| 정규화 peak | 2047 | 1023 |
| train / val patch 수 | 17,139 / 1,905 | 19,809 / 2,201 |
| train / val LRMS | 등록된 msfix | 등록된 GF2 |
| Aligner native LMS | 원본 저장 LMS override | 등록된 GF2 native LMS |
| 재구성 `I_lrms` | LRMS에서 interp23 | LRMS에서 interp23 |
| RGB 표시 | [2,1,0] | [2,1,0] |

QB는 기존 `config/bglobal_rr/qb_original_lms.json`의 혼합 경로를 명시적으로 유지한다. RR20/FR20, Q4, 센서별 FR MTF는 기존 공식 evaluator를 사용한다. 파일 SHA·밴드 수·population·evaluator·환경을 실행 전 고정한다. GF2 전체 calibration의 추정 필요량 3.344GiB에 맞춰 CPU collector 한도를 4GiB로 설정한다. 이 값은 프로세스 전체 RAM 사용 상한이 아니다.

## Sheet 게시

- QB: `QB-ablations` / gid `261001011`; GF2: `GF2-ablations` / gid `261001012`.
- Stage 1은 실제 새 학습의 validation RRE 선정 모델을 게시한다. Stage 2는 **공동 목표를 만족한 모든 step + run별 대표 하나**를 게시한다.
- 대표는 해당 센서 HQNR 목표를 만족한 결과 중 ERGAS 최소; HQNR 목표를 못 넘겼으면 HQNR 최대다. 대표 결과는 ERGAS 목표 미달이어도 게시한다.
- 같은 모델 상태의 RR20와 FR20, checkpoint SHA, 실제 native cost를 검증한다. 실행 이름·메모에 RGB 참조, A/T/S L4, scale, 입력, 구조, LR/horizon, α/β/γ, 선택 역할을 표시한다. v2 run ID는 PSN5R2로 구분하며 기존 v1 행은 그대로 보존한다.
- 해당 센서 탭의 목표 안내와 게시 규칙 셀만 검증·백업 후 갱신한다. 기존 WV3 행과 다른 캠페인 행은 보존한다.

## 실행 / 인계

기존 학습·게시 Python 환경, Docker GPU 권한, DLPan-Toolbox, 데이터 경로, `gspread/account.json`, 서버에 맞는 `gspread/server.txt`가 필요하다. 기존 TUNE30 Docker 이미지와 실제 체크포인트를 재사용한다. 데이터·인증정보·work_dir는 Git 전송 대상이 아니다.

```bash
# 이미 v1을 사용 중인 서버는 먼저 아래 인계문의 pre-update 안전 종료 절차를 따른다.
# 새 소스를 반영한 뒤 기존 학습 환경에서 저장소 루트로 이동한다.
export PYTHONDONTWRITEBYTECODE=1
python -m panda_sensor5 check --server s3
python -m panda_sensor5 smoke
python -m panda_sensor5.migration --server s3 --apply
python -m panda_sensor5.queue --server s3 --enqueue
```

s1은 `--server s1`로 동일하다. `--enqueue`는 전체 순서의 실행 명령이다. 내부의 `start_panda_sensor5_costed.sh`를 먼저 수동 실행하지 않는다. 큐는 기존 HARD36 대기 watcher를 회수하고, 현재 WV3 학습의 안전한 checkpoint 저장·동일 run 복원을 거쳐 그 run만 완료한 뒤 센서 캠페인을 시작한다. 기존 큐가 이미 정상 종료했으면 새 데이터셋으로 넘어간다. 실패·불완전한 증거는 자동 무시하지 않는다.

코드·데이터·환경을 pin하므로 실행 중 소스를 수정하지 않는다. 기존 WV3 코드는 수정하지 않고 별도 제어 어댑터가 이어받는다. 대기·재개 상태와 원래 예산은 별도 receipt에 남는다. s3에서 HARD36이 이미 시작된 경우에도 원래 HARD36 admission을 보존한다. 이후 TUNE30의 기존 budget skip이 완료된 변화만 별도 continuation receipt로 검증하고, 기존 완료 checkpoint·평가·코드·환경의 변경은 허용하지 않는다.

상태 경로:

```text
work_dir/panda_sensor5_launchers/<server>/status.json
work_dir/panda_sensor5_launchers/<server>/chain.json
work_dir/panda_sensor5_launchers/<server>/enqueue.json
work_dir/panda_sensor5/<campaign>/<server>/<dataset>/campaign_state.json
work_dir/panda_sensor5/<campaign>/<server>/<dataset>/sheet_publish_panda_sensor5/last_poll.json
```

학습 컨테이너는 `panda-sensor5-stage12-qb-s1-v2` 또는 `panda-sensor5-stage12-gf2-s3-v2`이다. `docker logs --tail 30 <name>`으로 진행을 확인한다. 전체 큐 순서는 `CUTOVER → START_SENSOR5 → WAIT_SENSOR5 → RESUME_TUNE30 → RESUME_HARD36 → COMPLETED`다. `BLOCKED`이면 status의 오류와 watcher 로그를 먼저 확인하며, 체크포인트나 admission 파일을 삭제해 강제로 통과시키지 않는다.

등록된 case 목록은 [s1 CSV](cases_s1.csv), [s3 CSV](cases_s3.csv)와 `python -m panda_sensor5 catalog --server s1 --csv`로 확인할 수 있다.


## v1 → v2 전환과 QB 진단

실행 중인 v1 소스에 직접 덮어쓰면 source pin이 중단을 발생시킨다. **checkout 전에** 새 commit의 `scripts/stop_panda_sensor5_before_update.py`를 `git show`로 외부 경로에 추출해 실행한다. 그 후 소스를 반영하고 migration을 적용한다. [정확한 s3 인계 절차](../research_log/PANDA_Sensor5_Stage12_Handoff_2026-10-26.md)를 따른다. migration은 기존 결과·모델·Sheet 행과 WV3 완료 증거를 보존하고 fresh Stage 1부터 새 donor/calibration을 만든다. 이미 WV3 재개 단계로 넘어간 서버는 자동으로 중단하지 않고 별도 확인한다.

RGB/L4 전환에 맞춰 validation의 RGB texture eligibility와 정합 참조를 함께 변경했다. RRE estimator는 기존 gradient NCC, Gaussian σ=.8, search bound8을 유지하며 ROI margin을20으로 둔다. 검색±8과 Gaussian3+Scharr1의 footprint를 빼면 corrected PAN의 검사 영역은 `[8:-8]`이다. 기존 bicubic용 `[4:-4]`를 L4에 그대로 적용해 정상적인 1–2픽셀 보정을 탈락시키지 않는다. 학습 ROI8/stencil7은 그대로다.

공유 masked reduction 라이브러리의 `MASK2` enum에 남아 있는 `BICUBIC4` 문자열은 호환용 식별자다. 실제 mask는 모든 경로에서 Lanczos의 floor−3…floor+4 support로 생성하며 별도의 operator manifest와 calibration metadata에 기록한다. 새 local calibration collector는 이전 bicubic donor/artifact를 거부한다. 기존 WV3의 고정 소스는 수정하지 않는다.

[QB 낮은 HQNR의 실측 분석](../results_log/2026-10-26_qb-low-hqnr-audit.md)에 기존 v1 결과를 보존했다. 큰 FR shift가 곧 오정합이라는 해석은 독립 정합 진단으로 지지되지 않았다. 원본 PAN 기준 HQNR과 MS 기준 정합 사이의 차이가 큰 영향을 보였으며, RGB/L4 변경만으로 목표가 달성된다고 보장하지 않는다. 진단용 zero shift 또는 보정된 PAN 참조 결과는 공식 결과로 게시하거나 donor 선정에 쓰지 않는다.
