# 2026-10-23 Schedule30 s3 실행 인계

사용자 요청: PAN Aligner를 양의 scale로 계속 사용하고 rotation loss 없이, 기존 결과를 바탕으로 낮은 우선순위 case를 제외하고 스케줄 실험을 추가한다. **새 큐에는 시간 제한이 없다.** 기존 30시간 캠페인의 예산이나 원본 증거를 재설정하지 않고 새 `PANDA_S13_SCHEDULE_HARDONLY_20261023_v1` 캠페인에서 진행한다.

이 인계는 Git으로 전달한다. 작성 세션에서 s3에 SSH 접속하거나 원격 실행하지 않았다. **s1 신규 학습도 이 문서만으로 실행 중이라고 판단하지 않는다.** 실제 시작은 launcher receipt, 컨테이너 상태, 증가하는 train step으로 별도 확인한다. 현재 s1의 기존 S040BASE R1은 5,489 update에서 `STOPPED_SAFE / STOP_REQUESTED`이며 stop checkpoint와 평가를 보존했다.

작성 세션의 추가 확인(2026-10-23 15:55 KST): s1 새 컨테이너가 실행됐으며 `S020F25C35 R1`의 25001 이후 train trace가 증가했다. 41개 자동 테스트, 8개 고유 설정의 실제 데이터 CUDA preflight가 통과했다. 새 Stage1 대표 row 178의 Sheet readback도 확인했다. s3 실행 성공을 의미하지 않는다. [분석과 후속 후보](../research_log/PANDA_Schedule30_Review_2026-10-23.md)를 함께 참고한다.

전체 계약·s1/s3 큐·재개 방법은 [panda_schedule30/README.md](../panda_schedule30/README.md), 원본 case registry는 [plan.py](../panda_schedule30/plan.py), [contract.py](../panda_schedule30/contract.py)다.

## s3 큐와 제외 대상

전부 scale **0.2**, β=0, rotation loss=0, single-view, no TTA다. 기본 α=0.25·γ=0.5·peak LR `1e-4`. Fresh는 warmup 100 후 해당 horizon까지 cosine→0이다.

| 순서 | Case | 반복 | 종료 step / 새 update | 변경 |
|---:|---|---|---|---|
| 1 | S020F25C35 | R1 | 35K / 10K | s3 S020BASE 25K의 상태를 보존하고 LR `2e-5 → 1e-6` |
| 2 | S020C30 | R1 | 30K / 30K | 짧은 cosine |
| 3 | S020C30H050 | R1 | 30K / 30K | α=0.125·γ=0.25 |
| 4 | S020C35H050 | R1 | 35K / 35K | H050 + 35K cosine |
| 5 | S020C30LR150 | R1 | 30K / 30K | peak LR `1.5e-4` |
| 6 | S020C30 | R2 | 30K / 30K | 짧은 cosine의 두 번째 seed |
| 7 | S020C30H050 | R2 | 30K / 30K | H050의 두 번째 seed |
| 8 | S020H050 | R2 | 50K / 50K | H050의 기존 horizon 대조군 |

| 기존 큐 대상 | 처리·이유 |
|---|---|
| s3 S020LR150H050 R2 | pending이면 제외. 정확히 이 case만 실행 중이면 checkpoint-safe stop 허용. 두 요인을 함께 바꾼 조합보다 LR·Hard 분리와 H050 재현 우선 |
| s3 다른 실행 중 case | 정상 종료 대기. 기본 R2 등 다른 case를 임의 정지하지 않음 |
| s3 이미 완료된 case | 완료 기록·모든 평가·checkpoint·기존 Sheet 행 유지 |
| s1 S040BASE/S050BASE R1 | S040은 5,489에서 안전 정지, S050은 pending 제외. 기존 높은 scale의 HQNR 열세·scale 0.5 반복보다 스케줄 비교 우선 |
| s1 S020BASE/S010BASE R2 | 기존 pending 큐에서 새 큐 후반으로 이관하여 대조군 유지 |

s3 판단은 확인 가능한 게시 대표값에 근거한 실험 우선순위 조정이며, s3 전체 궤적을 로컬에서 확인한 것처럼 해석하지 않는다. s1/s3 donor가 다르므로 결과는 같은 서버·donor·seed끼리 비교한다.

## Git 전달 후 로컬 실행

변경을 받은 뒤 `gspread/server.txt=s3`, 기존 데이터·DLPan·검증된 Docker 이미지·host Python·인증 파일을 확인한다. `.git` 사용자 작업, 기존 실행 소스, credential을 덮어쓰지 않는다. 이후 실행/전환 대기 중에는 pin 대상 파일을 편집하거나 Git 갱신하지 않는다. `work_dir`와 credential은 Git으로 전달하지 않는다.

필수 원본은 **s3 자체**의 다음 두 증거다.

- `work_dir/panda_mature30/PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2/s3/WV3/donors/rre_selection.json` 및 그 receipt가 지정한 A/T checkpoint.
- `work_dir/panda_scale30/PANDA_S13_SCALE_HARDTUNE_30H_20261022_v1/s3/WV3/runs/PS30_WV3_S3_S020BASE_NOSOFT_R1_S261018101/attempt_01/models/step_0025000.pt`와 `.json` sidecar 및 원 calibration.

없거나 SHA가 다르면 s1 파일로 대체하거나 다른 step을 임의 사용하지 않는다. [fork.py](../panda_schedule30/fork.py)는 Student, Adam moments/step, stream, RNG, A/T, calibration을 기존 loader로 검증한 뒤 보존한다. 원본 25K checkpoint는 수정하지 않고 새 캠페인의 25001~35000 update만 학습한다.

```bash
python -m panda_schedule30 catalog --server s3 --csv
python -m panda_schedule30 smoke
bash scripts/start_panda_schedule30_transition.sh s3
```

필요하면 위 명령 전에 `PAN_HOST_PYTHON`을 기존 torch·google-auth·requests Python으로, `PAN_SCHEDULE30_IMAGE`를 검증된 기존 로컬 이미지로 지정한다. 새 패키지 설치나 image pull은 필요하지 않다.

transition은 old pending skip 등록 → 허용된 대상만 안전 정지/다른 case 종료 대기 → GPU idle → 격리 cost 측정 → 실제 데이터 CUDA update·checkpoint restore·추론 검증 → 새 trainer와 publisher 시작 순서다. 기존 state/admission을 지우거나 old 30시간을 다시 시작하지 않는다. 기존 비정상 종료나 엉뚱한 미완료 run이 있으면 자동으로 넘어가지 않는다.

## 확인·재개

```bash
docker logs --tail 40 panda-schedule30-hardonly-wv3-s3-v1
```

- 전환: `work_dir/panda_schedule30_launchers/s3/transition_receipt.json`의 상태·오류·새 컨테이너.
- 새 캠페인: `work_dir/panda_schedule30/PANDA_S13_SCHEDULE_HARDONLY_20261023_v1/s3/WV3/`의 `preflight/passed.json`, `campaign_state.json`, 각 run의 `train.jsonl`.
- 게시: 같은 root의 `sheet_publish_panda_schedule30/last_poll.json`에서 새 policy와 `readback_verified=true`.

안전 정지 후 같은 소스·config·이미지로 재개하고 GPU가 비어 있을 때:

```bash
docker start panda-schedule30-hardonly-wv3-s3-v1
bash scripts/start_panda_schedule30_best_publisher.sh s3
```

게시기만 재시작할 때는 `bash scripts/restart_panda_schedule30_publisher.sh s3`. 이미 실행 중인 컨테이너를 새로 만들거나 삭제하지 않는다. code pin 불일치·poisoned state·checkpoint 오류가 있으면 기록을 보존하고 원인을 확인한다. 평범한 resume에서 LR/queue/seed/loss를 바꾸지 않는다.

## Sheet 규칙

기존 `WV3-ablations` 탭에서 **동일 checkpoint의 전체 정밀도**로 HQNR ≥ **0.9585**, ERGAS ≤ **2.040**인 모든 결과를 게시한다. 각 run/seed/attempt의 대표는 HQNR ≥ 0.9585 중 **최소 ERGAS**다. 대표에 ERGAS 상한은 요구하지 않는다. HQNR 충족 후보가 없으면 **최대 HQNR → 최소 ERGAS → 이른 step** 대표를 유지한다. 대표가 조건 충족 행과 겹치면 하나만 게시하며 final 전용 행은 만들지 않는다. Stage1 BEST_VAL_RRE는 유지한다.

정책 ID는 `PANDA_SCHEDULE30_RRE_STAGE2_H9585_PASS_ALL_OR_BEST_FALLBACK_V1`. `PSC30` 소유 행의 A:BL만 다루고 이전 캠페인·다른 서버 행과 추가 열은 보존한다. metrics는 같은 model state에 묶고 cost는 실제 서버·배포 graph signature가 일치하는 격리 측정만 연결한다. 미측정 cost를 추정하지 않는다. 25K 분기의 `Selected_step/Lifetime_updates=35K`, `Updates=10K`를 구분하며 시간도 새 분기의 실제 측정만 기록한다.

실행 후 s3 에이전트는 commit, 전환 상태, 실제 첫 train step, preflight 결과, Sheet readback을 별도 실행 기록에 남긴다. 이 문서에는 아직 s3 실행 성공이나 목표 달성을 선언하지 않는다.
