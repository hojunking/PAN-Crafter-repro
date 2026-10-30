# GF2 PSNR 45.08 후속 튜닝 — 2026-10-30

## 1. 새 목표와 판단

GF2 목표는 **HQNR ≥ 0.9645 AND ERGAS ≤ 0.522 AND PSNR ≥ 45.08 dB**이다.
세 값은 같은 단일 checkpoint의 공식 FR20/RR20 평가에서 동시에 만족해야 한다.
QB 목표(.9315 / 3.5), WV3 목표(.9585 / 2.040)는 유지한다.

GF2 SENSOR_FOCUS 5개 run은 모두 60K 완료됐고, 모두 기존 HQNR/ERGAS 조건을 달성했다.
현재 최선은 **GF2_C60LR150 R1 @47K: .9697040443 / .5181195524 / 45.0351455911**이다.
공개된 결과 중 새 세 목표 동시 달성은 없다. PSNR 차이는 **0.0448544089 dB**이다.
현재 PSNR은 scene별 전 밴드 통합 MSE의 dB 값을 평균한다. 따라서 필요한 개선은
scene MSE의 기하평균 기준 약 **1.03%**이며, 산술평균 MSE가 정확히 1.03% 감소해야 한다는 뜻은 아니다.

기존 donor로 Student만 바꿔 5개 모두 H/E를 달성했으므로 Stage1 추가 탐색보다
동일한 frozen Teacher/Aligner 아래 Stage2 튜닝을 우선한다. 두 seed의 차이와
평가 checkpoint 선택 효과를 감안하며, 작은 ERGAS 차이를 확정적 우열로 취급하지 않는다.

## 2. 실행 순서 — s3 전용, 기존 WV3 완료 후

새 campaign: `PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1`

| 순서 | case | seed | step | 폭 | 초기 LR | α | β | γ | 비교 목적 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1–2 | GF2P_C60LR200 | R1, R2 | 60K | 112 | 2e−4 | .25 | 0 | .5 | 현재 최선에서 LR만 증가 |
| 3–4 | GF2P_C50LR150 | R1, R2 | 50K | 112 | 1.5e−4 | .25 | 0 | .5 | cosine 종료 시점만 앞당김 |
| 5–6 | GF2P_C60LR150A0125 | R1, R2 | 60K | 112 | 1.5e−4 | .125 | 0 | .5 | 기존 Teacher error 가중 α만 감소 |
| 7–8 | GF2P_C60LR150W128 | R1, R2 | 60K | 128 | 1.5e−4 | .25 | 0 | .5 | 현재 최선에서 폭만 증가 |

총 8개 fresh Student, 합계 **460K updates**. 시간 제한 없음.
각 case는 기존 `GF2_C60LR150`과 비교하면 한 가지 설정만 다르다.
`comparison_control`도 원본 focus campaign의 같은 replica/seed `GF2_C60LR150`을 가리킨다.
`C50`은 기존 checkpoint를 이어 학습하는 실험이 아니라 처음부터 50K cosine으로 학습한다.
R1/R2 seed는 기존과 같은 `261018101` / `261018102`이며 원본 정수 stream seed를 사용한다.

LR150이 동일 seed의 LR100보다 PSNR을 R1 +.09737, R2 +.02153 dB 개선했다.
LR200은 기존 30K 실험에서 유망했으나 60K에서 이득과 HQNR 유지 여부는 미확인이다.
R1 47K 이후 공개된 H/E 통과 결과의 PSNR은 45.00–45.035 근방이므로,
80K 단순 연장보다 50K cosine을 먼저 시험한다. 성공을 보장하는 근거는 아니다.
α 감소는 일반 GT 오차 대비 Teacher error 기반 추가 가중을 줄이는 가설이며 GF2에서는 미검증이다.
W128은 LR100에서 이득이 일부 있었지만 HQNR 여유가 작아 보조 순위로 둔다.

## 3. 고정 조건

- Stage1: 원본 `panda_sensor5`, `PANDA_S13_SENSOR5_RGB_L4_STAGE12_20261026_v2`,
  `RGB_MATURE_U23 @42K`를 그대로 재사용한다. Stage1 재학습/새 donor 선별은 없다.
- donor checkpoint SHA256:
  `937ac56b559b76a151df661bbb853119512d6f9cab283346c61186db27ff7478`
- source receipt SHA256:
  `52626a232557267d6852bc2b49bde8fc05aa6ccd5c1fc759a79fbe1101bedf95`
- Stage1 λ_abs=2, λ_ε=.003, ε radius=3 RR px. RGB reference와 Lanczos-4를 유지한다.
- Stage2 PAN Aligner scale=1. 입력은 `[P_corrected, P_corrected−P_original, I_lrms]`, 6채널.
  출력은 4밴드 HRMS. 기존 UBIL, depth [1,2,3], interp23 MS base.
- 기존 GT Hard loss. β=0, γ=.5. rotation loss/TTA/SWA/EMA/밴드 가중 loss 없음.
- AdamW, weight decay .01, batch48/micro12, warmup100, cosine 최저 LR0, 1K마다 공식 평가.
- 기존 frozen donor SHA와 원본 source config/receipt를 검증한다. donor를 새 namespace로 위장하지 않는다.

## 4. s3 실행

사용자가 push한 뒤 s3 에이전트가 pull하고, 기존 host publisher에 사용하는 torch 포함 Python으로 실행한다.
이 코드는 현재 s1 실행 파일을 수정하지 않는다. 기존 s1 QB/WV3 및 s3의 남은 WV3 순서를 바꾸지 않는다.

```bash
python -m unittest discover -s panda_gf2_psnr/tests -v
python -m panda_gf2_psnr catalog
mkdir -p work_dir/panda_gf2_psnr/launcher_s3
nohup python -u -m panda_gf2_psnr.enqueue --wait > work_dir/panda_gf2_psnr/launcher_s3/controller.log 2>&1 < /dev/null &
```

`enqueue`는 `gspread/server.txt=s3`에서만 실행된다.
기존 `panda_sensor_focus_launchers/s3/status.json`의 `COMPLETED`를 기다린다.
그 후 기존 GF2 5개 run의 최종 checkpoint/evaluation을 검증하고, GPU 유휴 확인,
원본 SENSOR5 Docker 환경의 data/donor 확인, 독립 cost 측정, 8개 학습, Sheet publisher를 순서대로 시작한다.
이미 실행한 동일 컨테이너는 재사용하며 다른 프로세스를 종료하지 않는다.

새 출력 root:
`work_dir/panda_gf2_psnr/PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1/s3/GF2/`

컨트롤러 상태:
`work_dir/panda_gf2_psnr/launcher_s3/status.json`

이미 종료된 컨테이너의 실패를 자동으로 지워 재실행하지 않는다. 오류 시 로그/최종 resume를 점검한 뒤 처리한다.
본 s1 세션에서는 s3 파일/GPU에 접근하지 않았으므로 **s3 학습 시작은 아직 확인하지 못했다**.

## 5. Sheet 규칙

대상은 기존 `GF2-ablations` 탭이다. QB/WV3 결과를 이동하거나 덮어쓰지 않는다.

- 기존 H/E 조건을 만족한 모든 step을 계속 게시한다.
- 대표는 **HQNR≥.9645 중 ERGAS 최소**, 없으면 HQNR 최고 1개를 유지한다.
- 새 campaign은 PSNR≥45.08까지 만족했는지를 `3목표 충족/미달`로 별도 표시한다.
- 공개 row의 `H/E 충족`을 세 목표 달성으로 오해하지 않도록 구분한다.
- PSNR 최고 checkpoint도 H/E 조건을 만족하면 전체 통과 step에 포함되어 게시된다.
  대표가 무조건 PSNR 최고라는 뜻은 아니다.
- 새 목표는 banner와 notes에, LR·α·γ·폭·스케줄 등은 기존 Approach/config에 기록한다.
- 과거 run의 게시 규칙/행은 바꾸지 않는다. 새 campaign ID로 소유 행을 분리한다.

## 6. 검증과 구현 범위

`panda_gf2_psnr.adapter`는 전용 프로세스에서 기존 검증된 focus 런타임에 새 registry를 주입한다.
원본 focus Python 파일/컨트롤러 파일에는 수정이 없다. 기존 모듈이 이미 로드된 혼합 프로세스는 거부한다.
새 파일도 새 campaign의 source pin에 포함된다. 기존 s1 학습/컨트롤러 SHA pin이 그대로임을 확인했다.

CPU 테스트 10개 통과: registry, 3목표 경계와 finite 검사, 기존 대표/행 ID 보존,
선행 WV3 대기, Docker write 범위, 전용 프로세스 격리, 네 설정의 실제 1-step 학습,
정확한 optimizer/data/RNG resume, W112/W128 cost signature 및 publisher 빈 상태 검사,
대기 중 source 변경 차단과 기존 admission의 코드 일치 검사.
이는 합성 데이터의 구현 검증이며 GF2 성능 결과가 아니다. 실제 s3 data/donor/GPU 검증은 실행 시 수행한다.

상세 분석: [2026-10-30_sensor-results-and-gf2-psnr.md](../results_log/2026-10-30_sensor-results-and-gf2-psnr.md)

QB 포함 최신 운영 규칙: [시간 제한 없는 후속 큐](PANDA_QB_GF2_OpenEnded_Tuning_2026-10-30.md).
중간 결과에 따른 큐 변경은 사용자가 요청할 때 수행한다. 실행 중인 source/registry를 직접 수정하지 않는다.
