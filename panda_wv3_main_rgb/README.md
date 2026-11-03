# WV3 RGB main — s1

기존 ablation Ours 전체 구성에서 RGB Stage 1을 새로 학습하고, 6개 Student 설정을 3회 반복한다.
설계일 2026-11-03, 구현일 2026-11-04. 기존 s1 ablation과 s3 RGB ablation 소스를 변경하지 않는다.

## 큐

각 반복마다 s1에서 독립 Stage 1 Full 50K를 먼저 학습한다. 그 반복의 donor로 아래 여섯 Student를
처음부터 학습한 후 다음 반복으로 진행한다. Stage 1 3회 + Stage 2 18회 = 21 runs, 총 696K updates.
시간 제한은 없다. s3 weight를 받지 않는다.

| Case | Step | LR | α | γ |
|---|---:|---:|---:|---:|
| M0_OURS | 30K | 1e-4 | .25 | .25 |
| M1_G0125 | 30K | 1e-4 | .25 | .125 |
| M2_A0125 | 30K | 1e-4 | .125 | .25 |
| M3_C27 | 27K | 1e-4 | .25 | .25 |
| M4_C35 | 35K | 1e-4 | .25 | .25 |
| M5_LR110 | 30K | 1.1e-4 | .25 | .25 |

Stage 1은 새 s3 RGB A5_ALL과 같다: W112/D123/BASE, λ_spa=2, λ_ε=.003, epsilon disk R3,
A LR1e-5/T LR1e-4. L_spa는 GT RGB [4,2,1]의 산술평균을 Scharr 처리한다. Aligner MS 입력과
Teacher 출력은 8밴드 전체다. RGB validation RRE50→RRE90→ValERGAS→이른 step으로 donor를 선택한다.
동일 50K 학습의 1K 간격 후보만 사용하며, 테스트 지표는 donor 선택에 사용하지 않는다.

Stage 2는 frozen A/T, scale1, [corrected PAN, corrected−original PAN, interp23 I_lrms],
W112/D123/UBIL, β=0을 유지한다. 가중 GT L1은 1+αd_e+γd_e r_q이며 가중치 합으로 정규화하지 않는다.
d_e는 Teacher의 8밴드 복원 오차, q는 corrected PAN과 RGB GT의 Scharr 차이에서 구한다.
매 donor마다 RR train 전체로 calibration을 새로 생성한다. 같은 반복/geometry의 여섯 case만 공유한다.
Stage 1과 Teacher는 bicubic, Student는 Lanczos4로 기존 Ours와 새 RGB ablation의 warp 조건을 유지한다.
공통 AdamW WD=.01, batch48/micro12, warmup100→cosine0, FP32, 기존 paired data augmentation.
추가 rotation loss, SWA, TTA, band-weighted loss는 없다.

## 실행 및 전환

```bash
python3 -m panda_wv3_main_rgb.enqueue --server s1 --image hojunqueen/pancrafter-env:latest --after-current-ablation --enqueue
```

전환기는 CPU 소스/환경/데이터/Sheet 사전 점검을 먼저 수행한다. 등록 시점의 기존 s1 ablation run이
정상 완료되면 기존 컨테이너를 정상 종료하고 새 main 큐를 실행한다. 기존 run이 실패하면 자동 우회하지 않는다.
기존 checkpoint와 남은 ablation 큐는 보존한다. 기존 코드에는 stop-after-run hook이 없으므로 완료 감지 사이에
다음 ablation이 초기화될 수 있으나 정상 종료 시 복구 checkpoint를 남긴다. s3나 다른 GPU 작업은 정지하지 않는다.
`--target-run`으로 전환 대상 run을 명시할 수 있다. `--check-only`는 점검만 한다.

학습 컨테이너: `panda-wv3-main-rgb-s1-v1`.
출력: `work_dir/panda_wv3_main_rgb/PANDA_WV3_MAIN_RGB_S1_20261103_v1/s1/WV3/`.
모든 1K 평가 checkpoint의 A/T/S model weights와 설정·SHA를 보존한다. 평가 snapshot은 중복 optimizer
상태를 제외한다. 250 step마다 optimizer·데이터 순서·RNG를 포함한 완전한 복구 상태를 저장하고
최근 두 세대를 유지한다. 학습 재개는 이 complete 상태만 사용하며 평가 snapshot으로 재개하지 않는다.
Stage 1 실패 시 해당 반복의 여섯 Student는 BLOCKED_DONOR로 기록하고 다음 독립 반복으로 진행한다.
인프라/증거 오류는 우회하지 않고 ERROR로 남겨 복구한다.

## Sheet 및 판정

Stage 2만 `WV3-main`에 `(RGB)` Experiment 이름으로 게시한다. Stage 1 선택 근거와 SHA는 로컬 receipt와
Stage 2 notes에 포함한다. RGB test RRE는 RR GT/FR stored LMS 참조 진단이며 donor 선택에는 쓰지 않는다.

- 같은 checkpoint의 HQNR ≥ .9585 AND ERGAS ≤ 2.040인 평가를 모두 게시.
- run별 대표 하나: HQNR 충족 중 ERGAS 최소, 없으면 HQNR 최고.
- 대표와 통과 checkpoint가 같으면 합치며, 최초 대기 행을 대표 행으로 갱신해 빈 행을 만들지 않는다.
- fixed64, 타 캠페인/확장 열 보존, source checkpoint hash 및 Sheet readback 검증.
- 25K 공통 시점, 각 horizon−1K/최종 시점, 마지막 5회 평가 평균, 3반복 평균/표준편차를 구분해 분석한다.
- 대표 선정은 test-aware 탐색이다. 단일 최고값만으로 반복 안정성을 주장하지 않는다.

## 검증

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s panda_wv3_main_rgb/tests -v
python -m panda_wv3_main_rgb.release --environment
python -m panda_wv3_main_rgb catalog --server s1
```

새 main 결과는 fresh RGB Stage 1 결과다. 과거 mean8 G8 donor의 Ours checkpoint를 재학습한 결과로 표시하지 않는다.
