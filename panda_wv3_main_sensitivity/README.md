# WV3 main 조건에 맞춘 Stage 2 sensitivity

기존 LR=1e-4 sensitivity와 별도 캠페인이다. main `M5_LR110`과 같은
LR=1.1e-4를 사용하여 논문 표의 기준 설정 차이를 해소한다. 성능 최대화나
기본 계수가 최고인 곡선을 만드는 실험이 아니다.

| Case | α | γ | 반복당 실행 |
|---|---:|---:|---|
| H2_REF | .25 | .25 | M5_LR110 동일 반복의 기존 수치 재사용 |
| H2_A0125 | .125 | .25 | 신규 Stage 2 |
| H2_G0125 | .25 | .125 | 신규 Stage 2 |
| H2_A0 | 0 | .25 | 신규 Stage 2 |
| H2_G0 | .25 | 0 | 신규 Stage 2 |
| H2_A050 | .5 | .25 | 신규 Stage 2 |
| H2_G050 | .25 | .5 | 신규 Stage 2 |
| H2_A100 | 1 | .25 | 신규 Stage 2 |
| H2_G100 | .25 | 1 | 신규 Stage 2 |

R1 모든 case → R2 모든 case → R3 모든 case. 9설정×3회, 기존 대조군 3회와 신규
24회다. 재사용은 추가 독립 관측으로 세지 않는다. 반복마다 기존 s1 RGB donor
43K/49K/38K와 원래 Student seed 261018101/102/103, calibration을 그대로 고정한다.
각 반복 내부에서 계수만 변경하며, 반복 간에는 donor와 Student seed가 모두 다르다.
Stage 1 재학습 및 서버 간 weight 이동은 없다.

W112/D123/UBIL, scale1, β0, Teacher bicubic / Student Lanczos4,
corrected PAN + correction residual + interp23 MS 입력, batch48/micro12,
AdamW WD.01, warmup100, cosine30K→0를 유지한다. rotation loss/TTA/SWA는 없다.

주 결과는 **고정29K, 3회 전체 평균±표본 표준편차**이며 30K도 보관한다.
Sheet 대표 모델과 논문 sensitivity 집계는 구분한다. 실패 수를 공개하고 정상적으로
완료한 낮은 결과는 제외하지 않는다. 인프라 실패 재시도는 같은 설정·seed를 사용한다.
성능에 따른 조기 종료나 반복 수 변경은 없다. 29K와 LR은 이미 test 결과를 확인한
기존 main에서 온 조건이므로, 새로운 미사용 test 검증이라고 주장하지 않는다.

실행은 s1에서:

```bash
python3 -m panda_wv3_main_sensitivity.enqueue --server s1 --wait
docker logs --tail 30 panda-wv3-main-sensitivity-s1-v1
```

기존 작업을 중단하지 않는다. 소스·환경·데이터·donor SHA·Sheet 및 실제 CUDA에서
저장/복원과 기존 main 수치 경로를 확인한 후 시작한다. 새 campaign 폴더만 쓰기
마운트하며 기존 main과 sensitivity 파일은 읽기 전용이다.

출력: `work_dir/panda_wv3_main_sensitivity/PANDA_WV3_MAIN_SENS_LR110_S1_20261109_v1/s1/WV3/`.
Sheet: `WV3-ablations`, Experiment `WV3-MAIN-SENS-LR110-… R… (RGB)`.
기존 4자리·대표/목표 통과 결과 게시 규칙과 fixed29K 게시를 유지한다.
원래 값은 notes와 local JSON에 보관하며 모든 신규 1K 평가 weight를 보존한다.

구현은 `panda_wv3_hyper`의 검증된 공통 모듈을 이 패키지 namespace에서 읽는다.
별도 plan/reuse/release/preflight/launcher만 덮어쓰며 기존 모듈의 전역 값을
변경하지 않는다. release.json은 상속된 파일과 이 패키지 파일 모두의 SHA를 고정한다.
