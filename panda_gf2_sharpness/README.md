# GF2 선명도 비교 — s3 인계, 2026-11-06

현재 GF2 정성 결과는 U-Know 대비 경계 기울기 약−11%, PAN-Crafter-s5 대비−16%였다.
먼저 **기존 weight의 초기·중기·후기 출력**을 비교한다. 기존 실험의 초기 weight가
삭제된 경우에만 같은 s3 Stage1 Aligner/Teacher와 원래 설정으로 Stage2 앞5K를 재현한다.
새 Aligner 학습이나 다른 서버 weight 복사는 없다.

## 대상과 순서

| 순서 | 기존 case | Seed | Width | LR | α / γ / β |
|---|---|---:|---:|---:|---|
| 1 | GF2_C60LR150 R1 — 현재 정성 비교 모델의 recipe | 261018101 | 112 | 1.5e-4 | .25 / .50 / 0 |
| 2 | GF2P_C60LR200 R1 — 정량 목표 달성 recipe | 261018101 | 112 | 2e-4 | .25 / .50 / 0 |
| 3 | GF2_C60LR150 R2 | 261018102 | 112 | 1.5e-4 | .25 / .50 / 0 |
| 4 | GF2P_C60LR200 R2 — 목표 달성47K 모델 | 261018102 | 112 | 2e-4 | .25 / .50 / 0 |

현재 정성 Ours는 C60LR150 R1@47K이며, 최종 정량 목표 모델은 C60LR200 R2@47K이다.
두 모델을 혼동하지 않는다. 비교 대상 step은 **1/2/5/10/20/30/47/60K 중 남아 있는 것**이다.
재현이 필요하면0.5/1/2/3/5K를 저장하고 **5K에서 종료**한다. 최대4×5K=20K Student updates다.
원래 **cosine horizon은60K**, warmup100, floor0 그대로다. horizon을1K나5K로 줄이지 않는다.
기존 paired augmentation·학습 데이터·stream seed·AdamW batch48/microbatch12·WD.01을 유지한다.
UBIL, corrected PAN + correction residual + interp23 MS4, scale1, 양쪽Lanczos4다.
회전 consistency/TTA/SWA/EMA/soft loss/band loss는 추가하지 않는다.

## 정확한 로컬 donor

- s3 `PANDA_S13_SENSOR5_RGB_L4_STAGE12_20261026_v2`의 `RGB_MATURE_U23`@42K.
- SHA-256: `937ac56b559b76a151df661bbb853119512d6f9cab283346c61186db27ff7478`.
- Stage1 λ_abs2, λ_epsilon.003, εmax3px. 원래 validation-selected receipt와 완료50K 근거를 검증한다.
- 재현에는 **해당 기존 run의 config, 평가 JSON, calibration NPZ/해시**가 필요하다.
  이 파일들은 기존 weight 정리에서 삭제 대상이 아니었다. 누락/변조 시 임의로 대체하지 않고
  정확한 누락 정보를 기록한다. calibration을 새로 계산하여 원래 출력과 혼동하지 않는다.

## s3 실행

사용자 push 후 s3에서 **전체 커밋을 pull**한다. 이 변경은 새 패키지를 추가하며 실행 중인
W128/ablation 학습 소스는 수정하지 않는다. 기존 GPU 작업을 중단시키지 않고 유휴 시점에 시작한다.

```bash
docker build -f Dockerfile.env -t pancrafter-env:gf2-sharpness .
python3 -m panda_gf2_sharpness.enqueue --image pancrafter-env:gf2-sharpness --wait
```

대기 launcher는 tmux 등 지속 세션에서 실행한다. Docker가 시작된 뒤에는 detached container가
원본 weight 점검 → 필요한 경우5K 재현 → FR20 추론·선명도 측정 → ranking.csv 작성을 수행한다.
원본 저장소/데이터/기존 실험 폴더는 read-only, 새 output 폴더만 writable이다.

```bash
python3 -m panda_gf2_sharpness.enqueue --status
docker logs --tail 50 panda-gf2-sharpness-s3-v1
python3 -m panda_gf2_sharpness.enqueue --resume --wait
```

ML 환경에서 읽기 전용 inventory만 확인하려면:

```bash
python -m panda_gf2_sharpness catalog
python -m panda_gf2_sharpness audit
```

기존 weight만 평가하고 재현 학습을 생략하려면 해당 서버의 참조 ML 환경에서
`python -m panda_gf2_sharpness run --no-replay`를 사용한다. GPU가 유휴일 때 실행한다.
실제 서버 마커 `gspread/server.txt`가 s3인 경우만 학습을 시작한다.
이 서버(s1)에서 s3가 실행됐다고 간주하지 않는다.

## 재현과 결과 판정

- 남아 있는 원본 checkpoint는 파일SHA/config/step/동일상태 RR·FR 기록을 검증한다.
- 추론 출력은 장면별 원래 raw_prediction_sha256과 **비트 단위로 일치**해야 저장한다.
- 재현 weight는 새 `replays/`에만 보관한다. 원래 평가가 있는1/2/3/5K에서 RR20/FR20
  raw 출력과 대조하며, 다르면 mismatch를 기록하고 중단한다. 이 경우 기존 결과로 표시하지 않는다.
- 재현은 optimizer·scheduler·RNG·stream을 함께 저장하고 중단 후 복구한다.
  Stage1 Aligner/Teacher가 그대로인지 확인한다. 원본47K까지 다시 학습하지 않는다.
- 선명도는 사용자 제공 `51_sharpness_qb_gf2.py`의 핵심 연산과 동일하다.
  RGB 평균, PAN Sobel 상위10%, 출력 gradient의5×5 max(±2px), 전체 HFσ1.5,
  PAN local-std9 하위20%에서 출력 local-std5, FR20 장면별 평균이다.
- raw NPY에는 대비 변환이 없다. PNG에만 각 출력별 RGB[2,1,0] sorted1–99%를 적용한다.
  PAN은 원본 DN에서1–99% 렌더링한다. 원격 비교 PNG와의 반올림 차이는 별도로 볼 수 있다.
- 경계 기울기와 평탄부 잡음을 함께 본다. 정량 목표는 HQNR≥.9645, ERGAS≤.522,
  PSNR≥45.08이며, 선명도 후보 탐색에 이 조건을 필터로 강제하지 않는다.
- 이것은 **기존 실험 출력 점검/재현**이다. 기존 Sheet 대표행이나 정량 수치를 바꾸거나
  중복 실험행을 게시하지 않는다. 새로운 정량 실험을 시작하는 명령이 아니다.

출력: `work_dir/panda_gf2_sharpness/`

- `inventory.json`: 존재/삭제/평가누락 구분; invalid weight는 missing으로 처리하지 않는다.
- `replays/<원래run>/attempt_01/`: 필요할 때만 생성한 별도5K 재현, 원본은 읽기만 한다.
- `exports/<원래run>/step_XXXXXXX/`: FR20 PNG + 4밴드 NPY + 원본 평가/설정·SHA.
- `ranking.csv`, `ranking.json`: 선명도/잡음/HQNR/ERGAS/PSNR와 각 결과 경로.
- `worker_status.json`: 실제 진행·완료·중단·실패 상태.

후보를 선정한 뒤 RR/FR ZIP을 만들려면:

```bash
python -m panda_sensor_visuals.export --run-dir /실제/원본또는재현/attempt_01 --step 1000 --output work_dir/panda_gf2_sharpness/selected/GF2_selected_1K --zip
```

NPY는 CHW float32 DN, GF2의 정상 범위는0–1023이나 예측 NPY는 clipping하지 않는다.
RR20 (4,256,256), FR20 (4,512,512). SHA 검증된 prediction 및 참조 NPZ/PNG, 지표, metadata가 포함된다.

## 검증

```bash
python -m unittest discover -s panda_sensor_visuals/tests -v
python -m unittest discover -s panda_gf2_sharpness/tests -v
python -m panda_gf2_sharpness.release --environment
```

원격 GF2 실제 weight/훈련 검증은 s3에서 수행한다. s1에서는 QB 실제40개 추론의 공식SHA 일치,
ZIP CRC·NPY 검증과 GF2 합성 그래프·스케줄·재개·소스이식성 검증을 수행한다.
