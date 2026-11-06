# QB 정성 결과 패키지와 GF2 s3 선명도 점검 인계

## QB: 추천 모델의 실제 출력 보관

- 모델: **QB_LR120 R1 @30,000**, s1.
- Run: `PSRF_QB_S1_QB_LR120_NOSOFT_R1_S261018101`.
- 원본: `work_dir/panda_sensor_refine/PANDA_S13_SENSOR_REFINE_R34_20261028_v1/s1/QB/runs/PSRF_QB_S1_QB_LR120_NOSOFT_R1_S261018101/attempt_01/models/step_0030000.pt`.
- Checkpoint SHA-256: `c64d23c71ba121016bcd20e487adddc1e24446dbcc90808c51d1ca6d263e55f4`.
- ZIP: `results_log/paper_figs/QB_LR120_R1_30K_images_npy_20261106.zip`.
- ZIP SHA-256: `4bd2739fc1d397754eb01a5d50ae2137764eaa035c51ce77dc434ae12d9a2e53`.
- 크기: 413,691,778 bytes (약414 MB), 437 files. ZIP/이미지/NPY/weight는 Git에 넣지 않는다.

RR20 + FR20 예측 NPY, RGB PNG, PAN·LMS·LRMS·RR GT, 비교 패널, 참조 NPZ,
원본 config/evaluation/sidecar, 지표와 SHA 근거를 포함한다. NPY는 clipping하지 않은
**float32 DN, CHW, Blue/Green/Red/NIR**이며 RR `(4,256,256)`, FR `(4,512,512)`다.
40개 예측이 공식 평가 raw SHA와 일치하고 ZIP CRC/NPY를 검증했다.
FR20 PNG도 앞서 선명도를 측정한 이 모델 출력과 픽셀 단위로 일치한다.

| HQNR | D_s | D_lambda | ERGAS | SAM | PSNR |
|---:|---:|---:|---:|---:|---:|
| .9367561293 | .0306416284 | .0337723588 | 3.5445266291 | 4.3692210444 | 38.3670546443 |

이는 **선명도 후보**이며 ERGAS≤3.5를 만족한 최종 정량 모델이라는 뜻은 아니다.
전달받은 PAN PNG를 쓴 앞선 측정에서 경계 기울기(±2px)2.075348,
HF .035246, 평탄부 잡음 .013919였다. 현재 Ours21K 대비 경계+15.41%,
잡음+0.42%; PAN-Crafter-s5 대비 경계+2.34%, U-Know 대비−9.41%다.
본 ZIP은 PAN을 H5에서 직접 렌더링하므로 경계값은2.07536044이다.
두 PAN 표시 경로의 차이는 약0.0000122이며, 예측 PNG/NPY 자체는 같다.

재생성 명령은 `panda_sensor_visuals/README.md`에 있다.

## GF2: s3 로컬 Stage1 유지, 저장 모델 점검 및 필요시 초기 Stage2 재현

새 패키지 `panda_gf2_sharpness`와 `config/panda_gf2_sharpness`를 전달한다.
현재 정성 비교 모델 **C60LR150 R1@47K**와 정량 목표 모델 **C60LR200 R2@47K**를
분리해서 다룬다. W112/D123/UBIL, scale1, Lanczos4, α.25/γ.50/β0인 기존 recipe를 유지한다.

| 순서 | 기존 조합 | Seed | LR |
|---|---|---:|---:|
| 1 | GF2_C60LR150 R1 | 261018101 | 1.5e-4 |
| 2 | GF2P_C60LR200 R1 | 261018101 | 2e-4 |
| 3 | GF2_C60LR150 R2 | 261018102 | 1.5e-4 |
| 4 | GF2P_C60LR200 R2 | 261018102 | 2e-4 |

- 기존1/2/5/10/20/30/47/60K 중 저장 weight를 먼저 점검한다.
- 1/2/5K weight가 없고 원래 평가 기록이 있으면, 그 조합만 Stage2를0에서5K까지 재현한다.
  **optimizer cosine horizon60K와 warmup100을 유지**하며, 실행만5K에서 멈춘다.
  0.5/1/2/3/5K weight를 별도 폴더에 남긴다. 최대4×5K=20K updates.
- s3의 기존 `RGB_MATURE_U23`42K Aligner/Teacher와 원래 calibration을 재사용한다.
  Stage1 재학습·서버 간 weight 이동·원본 실험 수정은 없다.
- 원래 기록이 있는 step은 RR20/FR20 raw SHA와 정확히 일치해야 한다.
  불일치하면 재현 실패 근거를 남기고 멈추며, 기존 결과로 취급하지 않는다.
- 선명도에 정량 목표를 필터로 강제하지 않는다. 경계 기울기와 잡음을 함께 보고
  HQNR≥.9645, ERGAS≤.522, PSNR≥45.08 충족 여부를 별도로 확인한다.
- 기존 Sheet 행은 그대로 두고 `ranking.csv/json`, FR20 PNG/NPY를 로컬에 기록한다.
  기존 실험의 출력 점검이므로 중복 실험행을 올리지 않는다.

사용자가 push한 뒤 s3에서 pull하고 실행:

```bash
docker build -f Dockerfile.env -t pancrafter-env:gf2-sharpness .
python3 -m panda_gf2_sharpness.enqueue --image pancrafter-env:gf2-sharpness --wait
```

대기 명령은 tmux 등 지속 세션에서 실행한다. GPU가 비면 detached container로 시작한다.
기존 학습을 중단하지 않는다. 자세한 경로·재개·상태·선정 후 ZIP 명령은
[`panda_gf2_sharpness/README.md`](../panda_gf2_sharpness/README.md)를 따른다.

## 검증 범위

- QB 실제40개 추론: 원래 출력 SHA 일치, FR20 기존 진단 PNG 일치, ZIP CRC/NPY 검증.
- 사용자 원본 선명도 코드와 새 계산기의 동일 입력 결과 일치.
- GF24조합 CPU 합성 시험: 원래60K 스케줄, Hard-only, frozen A/T,
  전체 optimizer/RNG/stream 재개 후 다음 업데이트 동일성, export/native inference 일치.
- 누락 weight와 손상 weight 구분, RR/FR SHA 불일치 검출, latest 포인터 유실 복구,
  원본 읽기 전용·새 output만 쓰기 가능한 Docker mount 구성을 검사한다.
- Git에 포함되는 소스 및 수치 환경 manifest를 검증한다.
- **s3 실제 GF2 평가/재현은 아직 실행하지 않았다.** s3 로컬 파일로 수행할 작업이다.
  s1의 WV3 학습과 WV2 대기 큐는 변경하지 않았다.
