# NEXT30 — Lanczos4 후속 실험과 full-shift PAN 차영상

캠페인 `PANDA_S13_NEXT_HARDONLY_20261023_v1`, prefix `PNX30`.
2026-10-23 사용자 결정에 따라 warp를 **Lanczos-4 한 방식**으로 줄이고 다음 미시작 실험부터 학습·추론에 적용한다. 현재 시작한 bicubic run은 원래 설정으로 완료한다. 미시작 schedule과 Sampling30 구조 실험은 이 큐로 통합한다.

## 방법

- 모든 새 Student는 fresh 초기화와 양의 PAN Aligner 보정을 사용한다. 같은 서버의 mature val RRE donor를 동결한다.
- Teacher는 원래 학습 경로인 **full-u bicubic**을 유지한다. Student의 Lanczos4 PAN/q에 맞춰 train-only conditional-q calibration을 다시 계산한다.
- Lanczos4는 separable 8 tap, 축별 가중 합 1 정규화, whole-sample reflection이다. 부호는 `out[y,x]=in[y+dy,x+dx]`. 구현은 `warp.py`에 고정한다.
- 실제8tap과 Teacher bicubic support의 교집합을 쓴다. **GT Hard 수식은 그대로지만 경계의 유효 픽셀 집합은 기존 bicubic과 다를 수 있다.** 새 구조·입력 대조군끼리는 같은 규약을 공유한다. 공식 RR20/FR20 평가 영역과 GT 좌표는 유지한다.
- WV3, W112·D123, interp23 MS base, AdamW, batch48, FP32, warmup100, cosine floor0. 기본 LR=1e-4, α=.25/γ=.5/β=0. 등록표의 H050는 α=.125/γ=.25, LR150은 LR=1.5e-4다.
- 기존 C4 입력 증강만 유지한다. rotation loss, band-weighted loss, soft distillation, TTA를 추가하지 않는다. 총 wall-time 제한은 없다.

## Full-shift 차영상

`S100DIFFC30`은 `u=A(PAN,LMS)`를 축소 없이 적용한다. scale을 적용하지 않는다는 요청을 **scale=1.0**으로 기록한다.

```text
PANc = Lanczos4(PAN, u)
Student input = [PANc, PANc - PAN, MS base 8 bands]
Student output = HRMS 8 bands
Loss = existing GT Hard
```

9→10채널로 첫3×3 Conv에 112×1×3×3=1,008개 가중치가 추가된다. 공통 가중치와 corrected PAN/MS 입력 열은 동일 seed 대조군 값을 유지하고 새 차영상 열은0초기화한다. 따라서 초기 출력은 대조군과 동일하다. 추가 GT/loss는 없다. `S100C30`은 동일 full-u·Lanczos4·BASE·30K·LR/loss/seed에서 차영상만 없는 대조군이다.

PANc와 delta로 원 PAN도 선형 복원할 수 있다. 성능 향상만으로 Aligner의 정합 자체가 최적이라고 해석하지 않는다. shift clamp, scale 축소, bicubic fallback은 없다. ROI8의 Scharr stencil까지 포함한 conservative8tap support는 축별 `floor(u)∈[-4,3]`을 요구하며 calibration과 각 update에서 확인한다. 위반 시 raw/applied shift를 기록하고 중단한다.

s1 실제 donor의 CPU 검증에서 전체 RR-train9,714개와64표본×4기존C4 view 모두 통과했다. raw shift는 dy[-1.2754,1.4048], dx[-1.0367,.8842]px였다. 이는 s3나 모든 증강 조합을 대신 검증하지 않는다. [증거](../results_log/assets/2026-10-23_next30_support_cpu.json)

## 큐와 전환

[전체 현재·후속 표](../results_log/2026-10-23_NEXT30_current_and_queue.md), [s1 CSV](cases_s1.csv), [s3 CSV](cases_s3.csv)에 모든 계수와 순서가 있다. 등록 상한은 s1 18개/s3 17개다. R1/R2 seed는261018101/261018102다.

새 L4 S020C30 R1 → full-u 대조 S100C30 R1 → 차영상 S100DIFFC30 R1을 먼저 실행한다. 이후 스케줄·Hard/LR·down/up 구조 및 R2를 비교한다. 기존 schedule에서 이미 시작·완료한 fresh case는 완료 증거를 유지하고 새 대응 case의 중복 실행을 생략한다. **S020C30 R1/R2는 새 L4 구조 대조군이므로 예외적으로 항상 실행한다.** fork25K→35K는 새 warp로 이식하지 않는다. 기존 Sampling30의 bicubic 구조6개는 별도 실행하지 않는다.

```bash
python -m panda_next30 catalog --server s3 --csv
python -m panda_next30 smoke
bash scripts/start_panda_next30_transition.sh s3
```

s1은 인자만 바꾼다. 기존 torch/google-auth/requests 환경을 사용하고 필요하면 `PAN_HOST_PYTHON`, `PAN_NEXT30_IMAGE`, `PANCRAFTER_DLPAN`을 지정한다. 두 서버 모두 WV3이며 historical mount helper의 s1=WV3는 물리 서버와 별개다.

전환기는 정확히 확인된 기존 Sampling30 **대기 watcher만** retire한다. trainer에는 신호를 보내지 않는다. Schedule30 미시작 group만 pending_control로 새 큐에 넘기고 이미 시작한 group은 완료시킨다. 완료 checkpoint SHA·동일상태 평가·donor/data 증거를 검증한 뒤 GPU idle에서 비용 측정 → 새 train-only calibration → CUDA update/resume/native inference 검증 → 새 trainer/publisher를 시작한다. Sampling30이 이미 시작됐다면 자동 덮어쓰지 않는다.

구현 중 다음 경계를 먼저 확보할 때는 `python -m panda_next30.transition --server s1 --prepare-only`를 사용할 수 있다. 이는 전환 준비만 하며 소스 pin/GPU 실행을 등록하지 않는다. 자동 실행에는 start 스크립트가 필요하다. 전환 watcher 등록 후 pin 대상 소스·스크립트를 수정하지 않는다. 안전 정지한 trainer는 같은 코드·설정의 기존 컨테이너로 재개한다.

## Sheet

Approach는 예를 들어 다음과 같다.

```text
L4 warp | full-u (scale=1) | BASE | PANc+delta+MS (10ch) | Hard alpha=0.25 gamma=0.5 | lr=0.0001 cosine=30K | interp23
```

Model/inputs에도 `[PANc,PANc-PAN,MS base] → 8bands`를 기록한다. notes에는 전체 설정·warp/input manifest·checkpoint 증거를 남긴다. 기존 캠페인 행을 새 방식으로 재표시하지 않는다.

동일 상태 full precision **HQNR≥.9585 AND ERGAS≤2.040** 통과 전체 + run/seed/attempt 대표 하나를 게시한다. 대표는 HQNR≥.9585 중 최소 ERGAS, 없으면 최고 HQNR→최소 ERGAS→이른 step이다. 같은 checkpoint는 중복하지 않고 Stage1 RRE 및 기존 결과를 유지한다. 실제 서버·배포 경로별 비용만 쓴다.

## 상태

- 전환: `work_dir/panda_next30_launchers/<server>/transition_receipt.json`
- 준비: 같은 폴더의 `preparation_receipt.json`
- 학습: `work_dir/panda_next30/PANDA_S13_NEXT_HARDONLY_20261023_v1/<server>/WV3/campaign_state.json`
- 실제 큐: campaign state의 `migration_effective_queue`, 생략 기록 `migration_skips`
- CUDA: campaign root의 `preflight/passed.json`
- 게시: root의 `sheet_publish_panda_next30/last_poll.json`

CPU검증·watcher등록은 새GPU학습·목표달성의 증거가 아니다. 새GPU작업은 현재run 종료 후 시작된다. s3는 [인계서](../results_log/2026-10-23_NEXT30_s3_handoff.md)를 따른다.
