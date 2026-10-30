# 2026-10-30 GF2 종료 및 s3 QB 후속 실험 인계

## 1. 결정과 실행 범위

사용자 요청: GF2는 최종 목표 달성으로 종료하고, s3도 QB 실험에 배정한다.
현재 s1 QB refit 큐는 유지한다. 이 문서와 신규 패키지는 기존 실행 코드의 pin을
바꾸지 않는다. s3에 직접 접속한 작업은 없으며, **아래 절차를 s3 에이전트가 실행해야
실제 GF2 종료와 QB 시작이 이루어진다.** Git push는 사용자가 담당한다.

- GF2 종료 기준: 동일 checkpoint에서 **HQNR≥.9645, ERGAS≤.522, PSNR≥45.08**.
- QB 목표: 동일 checkpoint에서 **HQNR≥.9315, ERGAS≤3.5**.
- s3 신규 큐: **자체 Stage 1 50K 1개 + fresh Student 8개, 각 30K**, 시간 제한 없음.
- s1은 기존에 자체 학습한 Stage 1을 유지한다. s3에 s1 모델을 복제하도록 강제하지 않는다.
- s1 기존 QB 및 이후 WV3 큐는 변경하지 않는다. s3 신규 큐 완료 후 WV3를 자동 재개하지 않는다.
- 중간 결과에 따른 추가 변경은 이후 사용자 요청 시 검토한다.

## 2. GF2 목표 모델과 보존

2026-10-30 실시간 `GF2-ablations`의 확인 결과:

| 항목 | 값 |
|---|---|
| campaign | `PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1` |
| case / replica / step | **GF2P_C60LR200 / R2 / 47,000** |
| HQNR | **0.9646694883162039** |
| ERGAS | **0.5148830158510865** |
| PSNR | **45.084138095671655** |
| checkpoint SHA256 | `9027206526d343b5a82aa735057f13de7ed0778ec2ad8fadaec73d27760182d6` |

원본 경로는 s3 checkout 아래:

```text
work_dir/panda_gf2_psnr/PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1/s3/GF2/runs/PSFC_GF2_S3_GF2P_C60LR200_NOSOFT_R2_S261018102/attempt_01/models/step_0047000.pt
```

Sheet publication에는 실제 파일 SHA 검증, weights_saved, checkpoint_verified,
campaign_resume_complete가 모두 기록되어 있다. 이는 s3 publisher의 증거이며,
s1에서 s3 디스크를 직접 읽어 검증한 것은 아니다. `panda_gf2_closeout`이 s3에서
다시 실제 파일 SHA와 동일 상태의 평가를 검증한 뒤 별도 archive에 복사한다.

**43K 대표 모델을 47K 목표 모델의 대용으로 보존하면 안 된다.** 43K는
H=.9658742654/E=.5148040063으로 H/E 대표이지만 PSNR=45.0756201로 최종 목표에
미달한다. 신규 closeout은 47K와 다른 모든 로컬 3목표 통과 checkpoint, config,
평가, SHA, frozen donor, selector, calibration을 보호한다. 원본 weight는 삭제하지 않는다.
목표 여유는 HQNR 약 .0001695, PSNR 약 .0041381로 작다. 이 달성을 seed 간
안정성 검증 완료로 확대 해석하지 않되, 사용자 결정대로 GF2 학습은 종료한다.

## 3. QB 분석에 따른 설계

분석 범위 23 Stage 2 run/647 checkpoint에서 동시 목표 달성은 없다.
최선은 `QB_BASE R2 @26K`: **H=.945383746/E=3.509656381**로 ERGAS가 .009656 남았다.

- LR080은 같은 30K에서 두 seed 모두 ERGAS가 악화했다(+.012834/+.019671).
  장면별로도 16/20, 17/20장이 악화해 단일 이상치로 설명되지 않는다.
- LR120·C40은 train GT L1을 더 낮췄으나 RR ERGAS는 악화했다. 단순히 LR나
  학습 길이를 늘리는 방향을 확대할 근거가 약하다. 독립 Student validation
  궤적이 없어 과적합으로 확정하지는 않는다.
- 같은 frozen Aligner를 쓴 Teacher는 E=3.492595/H=.759511이다.
  Student는 실제 shift를 바꾸지 않으면서 H=.945384/E=3.509656에 도달했다.
  이는 복원 함수·입력·loss·학습 과정의 차이를 살펴볼 근거이며,
  Teacher가 QB 동시 목표를 달성했다거나 residual 하나가 원인이라는 뜻은 아니다.
- 작은 α 감소와 W128은 seed 간 개선이 반복되지 않았다. 아키텍처를 더 크게
  바꾸기 전에 기존 optimizer의 weight decay와 반대 방향의 α를 분리해 확인한다.

[전체 분석과 장면별 근거](2026-10-30_qb-plateau-audit.md),
[시각화](assets/2026-10-30_qb-plateau/audit.png).

## 4. s3 신규 QB 큐

먼저 s3에서 검증된 QB Stage 1 `A200_E030` 설정으로 **새로 50K 학습**한다.
Stage 1 seed는 261016301이며 λ_abs=2, λ_epsilon=.03, epsilon 반경3, RGB/Lanczos4다.
50K 전체 후보의 validation RRE50 → RRE90 → validation ERGAS → 이른 step으로
그 서버의 Aligner·Teacher를 선정한다. 44K나 test 성능을 기준으로 강제 선정하지 않는다.

후속 Student 공통: **위 s3 자체 Stage 1 선정 모델**, fresh Student, W112/UBIL,
30K warmup/cosine, LR=1e-4, effective batch48, scale1, β0, γ.5.
입력은 corrected PAN + (corrected PAN − original PAN) + I_lrms이며 RGB/Lanczos4를 유지한다.
Stage 1이 완료·선정된 후 8개 Student가 동일한 s3 자체 Aligner/Teacher를 사용한다.
서버별 독립 학습과 s3 내부 하이퍼파라미터의 통제 비교를 함께 수행하는 구성이다.

| 순서 | case | seed | α | AdamW weight decay | 목적 |
|---|---|---|---:|---:|---|
| 1–2 | QB3_CONTROL | R1/R2 | .25 | .01 | s3 자체 Stage 1을 사용한 기준 |
| 3–4 | QB3_WD000 | R1/R2 | .25 | 0 | decay 제거 |
| 5–6 | QB3_WD100 | R1/R2 | .25 | .1 | 강한 decay |
| 7–8 | QB3_A075 | R1/R2 | .75 | .01 | Teacher-error 강조 증가 |

WD와 α는 기존 방법의 하이퍼파라미터다. 효과는 아직 검증되지 않았다.
서버 간 CONTROL 차이에는 Stage 1 학습 차이가 포함된다. 이를 GPU 차이 또는
WD/α의 효과로 해석하지 않는다. WD/α 효과는 s3 내부의 같은 Student seed와 비교한다.
α 증가는 공간별 가중치와 전체 loss 크기를 함께 바꾸므로 순수한 공간 강조만의
효과로 해석하지 않는다. 회전 loss/TTA/SWA/EMA/soft loss/밴드 가중 loss는 추가하지 않는다.
기존 QB 8·9·10K HQNR 모두 .90 미만인 경우의 조기 종료를 유지한다.

현재 Aligner의 Stored LMS 입력과 reconstruction의 interp23 MS 입력을 그대로 사용한다.
앞서 논의한 Aligner 입력 통일이나 low-resolution MS 직접 입력은 이 큐에 섞지 않는다.

실행 원장: [cases_s3.csv](../panda_qb_deepen/cases_s3.csv),
[experiment_manifest.json](../panda_qb_deepen/experiment_manifest.json).
완료 비교는 대표값 외에 같은 30K, 25–30K 평균, 20장별 변화, 두 seed 방향을 함께 본다.

서버별 Stage 1 차이를 해석할 때는 선정 step, validation RRE50/RRE90/ERGAS,
실제 예측 shift 분포를 먼저 비교한다. 이어 각 서버 CONTROL의 1K/10K/30K
HQNR·ERGAS·D_s·D_lambda 변화를 본다. s3의 모든 Student가 함께 악화된다면
개별 WD/α보다는 그 서버 donor와 학습·데이터 경로부터 확인한다. 이 경우에도
test 성능이 좋은 Stage 1 checkpoint로 선정 규칙을 사후 변경하지 않는다.

## 5. s1 유지 큐

17시대 확인 시 LR080 R1/R2 완료, LR050 R1 진행 중이며 이후 순서는 다음과 같다.

1. LR050 R1 현재 run 완료
2. LR050 R2
3. C25 R1/R2
4. AG000 R1/R2
5. G025 R1/R2
6. 기존에 등록된 나머지 WV3 TUNE30/HARD36

실제 진행 step은 계속 변한다. 이 작업에서 s1 trainer나 controller를 재시작하지 않았다.

## 6. s3 전달 자산: 코드와 기존 QB 데이터

Git에 신규 코드/실험 원장/문서가 들어간다. **s1 donor archive 전달은 필요하지 않다.**
s3는 자체 Stage 1을 학습하므로 새 Aligner·Teacher와 선정 근거가 그 서버에 생성된다.
QB의 msfix train/val, original LMS train/val, RR20/FR20 데이터는 기존 등록 파일과
내용이 같아야 한다. launcher가 실제 symlink 대상의 읽기 전용 mount와 데이터 해시를
확인하며, 자체 donor에 맞춰 calibration을 계산한다.

## 7. s3 에이전트 실행 순서

checkout 최신 커밋과 QB 데이터를 확보한 뒤, 프로젝트의 기존 Python 환경에서 실행한다.
`gspread/server.txt`는 기존 s3 값을 유지한다. s1에서 서버 표기를 바꿔 실행하지 않는다.

```bash
export PYTHONDONTWRITEBYTECODE=1
python -m panda_gf2_closeout plan
python -m panda_gf2_closeout apply --wait
python -m panda_gf2_closeout verify
python -m panda_qb_deepen catalog --csv
```

closeout은 목표 모델 보존 후 GF2 enqueue를 먼저 종료하고, 소유가 확인된 GF2
trainer에 TERM을 전달한다. 학습 코드의 안전 저장이 끝나면 최종 GF2 Sheet 게시를
검증하고 `work_dir/panda_gf2_closeout/s3/retirement.json`을 생성한다.
미시작 run은 제외, 중단 run은 안전 중단으로 기록하며 완료로 위장하지 않는다.
오류 시 강제 kill이나 파일 삭제로 우회하지 말고 보고된 불일치를 확인한다.

그다음 다음 명령을 지속 가능한 세션에서 실행한다. 필요한 Docker 환경/데이터 mount,
check와 cost, trainer 및 Sheet publisher 시작은 enqueue가 담당한다.

```bash
python -u -m panda_qb_deepen.enqueue --wait
```

QB 게시 대상은 `QB-ablations`이며 하이퍼파라미터, weight decay, 자체 Stage 1 선정 근거,
실행 server를 함께 기록한다. H/E 공동 통과 결과 전체 + H 통과 중 최소 E 대표,
H 통과가 없으면 최대 H 대표 규칙을 유지한다. GF2는 `GF2-ablations`에 마지막 상태를
게시하고, 성공한 47K 행이 남는지 readback으로 확인해야 QB로 넘어간다.

상태 확인 경로:

```text
work_dir/panda_gf2_closeout/s3/retirement.json
work_dir/panda_qb_deepen/launcher_s3/status.json
work_dir/panda_qb_deepen/PANDA_S3_QB_DEEPEN_OWNSTAGE1_20261030_v1/s3/QB/campaign_state.json
```

## 8. 가중치 저장 점검 결과

현 runtime은 목표 달성 여부와 무관하게 평가 1K마다 weight를 저장한다.
s1 관련 최근 9개 run의 실제 checkpoint **253개, 약 12.07 GB** SHA를 검증했고
오류는 없었다. 대표 9개 payload를 CPU에서 열어 Student, frozen Aligner/Teacher,
optimizer/scheduler 및 데이터 순서/RNG의 재개 정보를 확인했다.
calibration NPZ와 donor/selector도 별도 보존해야 정확히 재개할 수 있다.

원본 evaluation JSON의 `weights_saved=false`만으로 저장 실패라고 판단하면 안 된다.
평가 후 checkpoint를 저장하는 순서 때문에 생길 수 있으며, `models/step_*.pt`와
sidecar의 실제 파일 SHA 및 publication artifact를 확인해야 한다.
자세한 범위와 한계는 [checkpoint 보존 감사](2026-10-30_checkpoint-retention-audit.md)에 기록했다.

## 9. 구현 검증과 남은 실행 단계

- GF2 종료·보존 검증 10개: 목표 모델과 대표 모델 구분, 파일 손상 거부,
  성공 모델 전체 보관, controller→trainer 종료 순서, s1 실행 거부, 보관본 변조 거부.
- QB 큐 검증 6개: 자체 Stage 1의 A/T 실제 CPU 업데이트와 정확한 재개,
  Student 네 설정의 실제 AdamW 업데이트와 정확한 재개, 전체 50K 선정 조건,
  9개 run·새 seed·QB 게시 기준·데이터 mount·GF2 종료 선행 조건.
- 기존 실행 패키지 소스 변경 없음. 새 패키지 Python 구문 검사 완료.

CPU 검증은 실험 성능을 예측하는 결과가 아니다. 실제 s3의 GPU·데이터·프로파일링과
full-train calibration 검증은 해당 서버 실행 시 수행한다. 현재 세션에서 s3의
GF2를 종료하거나 QB를 시작했다고 기록하지 않는다.
