# WV3 anchored component ablation — s3 Stage 1 / s1 Stage 2

2026-11-02. 사용자 지정 두 checkpoint를 기준으로 기존 component 초안을 수정했다.
**s3는 Stage 1만, s1은 Stage 2만 수행하며 서버 간 weight를 옮기지 않는다.**
새 ablation case마다 서로 다른 seed로 3회 반복한다. 지정한 두 결과는 고정 reference로 두며 재학습 case로 세지 않는다.
실행 코드와 순회 큐를 등록했다. 실제 학습 시작·기존 큐 중단·Sheet 쓰기는 수행하지 않았다.
구현·서버별 실행 명령은 [실행 안내](../panda_wv3_ablation/README.md)를 따른다.

- [기준 모델·설정·파일 SHA 확인](../results_log/assets/2026-11-02_wv3-component-ablation/reference_audit.json)
- [구현과 일치하는 case 명세](../results_log/assets/2026-11-02_wv3-component-ablation/cases.json)
- [서버·반복·seed별 전체 42개 run](../results_log/assets/2026-11-02_wv3-component-ablation/runs.csv)
- [참고 원고](<method_writing/method (1).tex>)

## 1. 고정하는 두 reference

| 구분 | Stage 1 reference | Stage 2 최종 reference |
|---|---|---|
| Run | `PB30_WV3_S1_G8_S1_U23_R1_S261016101` | `PTN30_WV3_S1_S100DIFFUBILC30G025_NOSOFT_R1_S261018101` |
| 표시 이름 | `G8_MATURE_U23` | `TUNE30 L4 S100DIFFUBILC30G025 R1` |
| 서버 | s1 | s1 |
| Checkpoint | 50,000 | **29,000** |
| 학습 horizon | 50,000 | **30,000** |
| HQNR | 0.920738760 | **0.960336123** |
| ERGAS | 2.040115077 | **2.034058730** |
| D_s | 0.068903752 | 0.028071625 |
| D_lambda | 0.011200771 | 0.011928787 |
| PSNR | 37.992542926 | 38.033557136 |

두 checkpoint 모두 로컬 파일 존재와 SHA256을 확인했다.
Stage 1은 validation RRE로 선택된 50K이며 RRE50=0.094861632px, RRE90=0.241326225px다.
controlled-shift response EPE50=0.227346152px는 별도 진단으로 RRE50과 혼용하지 않는다.
RRE는 GT 기반 registration proxy이며 물리적 displacement GT가 아니다.
선택 순서는 RRE50 → RRE90 → validation ERGAS → 이른 update다.

- Stage 1 file SHA256: `62f1c445b8500070429f46503a68fb972a9144a4321fc9da10ccd45891186b8d`
- Stage 2 file SHA256: `64b96cc49b6f0c6780ba3de8b8b5b0ff58868b79deb89f6097b82d0b0d106006`

29K는 사용자가 지정한 비교 checkpoint다. 새 Student도 cosine horizon은 30K로 유지하고
**고정 29K를 주 비교, 30K를 보조 비교**로 저장한다. horizon을 29K로 바꾸지 않는다.
과거 지표를 본 뒤 선택한 reference임을 명시하고, blind validation selection으로 표현하지 않는다.
원본 evaluation JSON의 checkpoint 필드가 null이므로 weights 보존은 해당 JSON이 아니라
실제 `models/step_0029000.pt`와 selection receipt의 donor 파일을 직접 확인했다.

## 2. 실제 하이퍼파라미터와 경로

| 항목 | Stage 1 | Stage 2 |
|---|---|---|
| Network | T: W112 / D123 / BASE | S: W112 / D123 / UBIL |
| Reconstruction LR | 1e-4 | 1e-4 |
| Aligner LR | 1e-5 | 동결 |
| λ_abs = 원고 λ_spa | **2.0** | 해당 없음 |
| λ_epsilon | **0.003** | 해당 없음 |
| ε 주입 | PAN에 면적 균일 disk, R=3 current-PAN px | 없음 |
| α / γ / β | 해당 없음 | **0.25 / 0.25 / 0** |
| Aligner 적용 scale | 1.0 | **1.0** |
| PAN warp | bicubic, border | Student Lanczos-4, reflection / frozen T bicubic, border |
| 구조 비교용 GT | **8밴드 평균 후 Scharr** | **RGB(4,2,1) 평균 후 Scharr** |
| Aligner MS 입력 | 관측 Stored LMS 전체 8밴드 | 동일 |
| Reconstruction base | interp23 MS, 8밴드 | 동일 |
| Reconstruction 입력 | corrected PAN + MS, 9ch | corrected PAN + corrected−original PAN + MS, **10ch** |

양 단계 AdamW, WD=.01, betas=(.9,.999), eps=1e-8,
effective batch48=microbatch12×4, warmup100, cosine floor0, FP32를 유지한다.
기존 flip/90° 입력 augmentation과 named RNG streams를 유지하며 rotation-consistency loss/TTA/SWA는 추가하지 않는다.
ε 반경 3px는 합성 주입 범위이고 learned native shift의 clamp 상한이 아니다.
ε branch의 native anchor는 stop-gradient이고 주입 PAN은 T reconstruction에 넣지 않는다.

UBIL은 bilinear ×2 (`align_corners=False`) + 3×3 conv이며 downsampling은 기존 stride-2 3×3 conv다.
T는 기존 2×2 transposed-conv upsampling을 유지한다.
Stage 2는 T weights를 복사하지 않고 새 Student를 초기화한다.
DIFF의 추가 first-conv 1,008개 weight는 **0으로 초기화**하며 공통 PAN/MS 열과 나머지 초기값은 보존한다.
이전 초안의 추가 채널 random initialization은 폐기한다.

**원고와의 차이:** 현재 원고는 Stage 1 GT 구조 참조도 RGB라고 적지만 지정 donor는 mean8이다.
새 RGB/L4/common-MS 통일을 이번 ablation에 섞으면 원래 reference와 구성 요소 이외의 차이까지 생긴다.
이번 비교는 위 역사적 경로를 유지하며, 원고의 해당 설명은 이후 실제 채택 method에 맞춰 정리해야 한다.
기존 config의 `base_reference=BICUBIC_MS` 및 inherited RGB 항목만 읽어 실행 경로를 판정하지 않는다.
명시적 `upsampler=interp23`, `alignment_target=SCHARR_OF_MEAN8_GT`와 실제 호출 코드로 확인했다.

## 3. s3 — Stage 1 ablation: 5 case × 3회

모든 reconstruction network는 **T와 같은 W112/D123/BASE, 50K**다.
아래 결과는 Stage 1 reconstruction 결과이며 UBIL Student 30K 결과로 표시하지 않는다.
각 case의 reconstruction objective는 GT L1이다. RAW는 원 영상 전체, 정합 case는 해당 warp의 유효 support를 사용한다.

| Case | 방식 | Stage 1 objective | 비교 목적 |
|---|---|---|---|
| S1_A0_RAW | Aligner 없이 원 PAN + MS | L_rec | 단순 U-Net |
| S1_A1_SCHARR | 영상별 Scharr global shift + U-Net | L_rec | 비학습 정합 |
| S1_A2_REC | learned A + T | L_rec | reconstruction gradient만의 정합 |
| S1_A3_REC_SPA | learned A + T | L_rec + 2 L_spa | controlled shift 제거 |
| S1_A4_REC_EPS | learned A + T | L_rec + .003 L_epsilon | 직접 spatial 감독 제거 |
| REF_S1_ALL | 지정한 기존 50K 결과, 새 run 없음 | L_rec + 2 L_spa + .003 L_epsilon | 완전한 Stage 1 reference |

`PAN Aligner − abs loss (w/o controlled shift)`는 이 설계에서 S1_A3처럼
reconstruction + abs/spatial supervision을 뜻한다. abs만 남겨 T 학습을 없애는 구성이 아니다.
S1_A2/A3/A4에서 손실 외 A/T 구조·LR·초기값·stream은 같은 반복 seed끼리 맞춘다.
꺼진 ε branch의 forward를 생략해도 별도 data/augmentation RNG가 달라지지 않아야 한다.

Scharr는 **각 영상마다 하나의 (dy,dx)**를 PAN 전체에 적용한다.
관측 PAN과 관측 Stored LMS의 mean8 구조를 비교하며 shift 추정에 GT를 사용하지 않는다.
Stage 1 reference와 맞추어 보정 PAN에는 bicubic을 사용한다.
확정 검색은 각 축 [-5,5]px 정수 coarse + 최적점 주변 ±1px/.25px fine, 같은 범위 제한·ROI8이다.
±5는 bicubic 고정 ROI8/stencil7을 보존하는 최대 대칭 정수 범위다. 합성 ε disk R3와 구분한다.
동점 규칙은 최소 norm → L1 → (dy,dx)로 고정한다.
기존 integer-only matcher를 그대로 쓴 subpixel baseline이라고 표기하지 않는다.
[train/validation 사전 점검](../results_log/assets/2026-11-02_wv3-component-ablation/scharr_preflight.json)을 마쳤다.
처음 32개씩의 표본에서 ±5 경계 도달률은 train 21.875%, val 43.75%였다.
범위 확대가 경계 문제를 해소한 것은 아니며 전체 misalignment 분포 추정도 아니다.
실행 중 boundary-hit/subpixel 비율을 기록하고 baseline의 제한으로 해석한다.
test 결과로 검색 범위를 고르지 않는다. 추론 시간에는 검색 비용도 포함한다.

주 비교는 모든 case의 **고정 50K**다. learned A에 대한 validation RRE 선택 결과는 보조로 함께 저장한다.
native shift norm/방향, support fraction, RRE50/90, ε response EPE/gain, Teacher RR/FR를 기록한다.
다른 정합량에 따른 support 차이도 포함된 strategy 비교이므로 순수한 translation 효과로 단정하지 않는다.
Stage 1 결과만으로 해당 Aligner를 사용한 Stage 2의 성능 향상을 주장하지 않는다.

## 4. s1 — Stage 2 ablation: 핵심 5 + 메커니즘 4 case, 각 3회

**모두 같은 로컬 `G8_MATURE_U23` A/T를 동결**한다. s3 산출물을 읽지 않는다.
Student와 입력 augmentation은 동일하고 새 Student seed만 바꾼다.
`C`는 corrected PAN, `D=C−P`, 모든 입력에 같은 MS base 8채널이 추가된다.
GT loss는 `R_m[w e_S]`이며 teacher imitation/soft loss는 전부 0이다.

| Case | PAN 입력 | weight w | 비교 목적 |
|---|---|---|---|
| S2_B0_CORR_GT | C | 1 | full Aligner만 사용한 기준 |
| S2_B1_DIFF_GT | C,D | 1 | residual 입력 기여 |
| S2_B2_E_ONLY | C,D | 1+.25 d_e | additive reference-error 항 |
| S2_B3_EQ_ONLY | C,D | 1+.25 d_e r_q | interaction 항 |
| S2_B5_NO_DIFF_AF | C | 1+.25 d_e+.25 d_e r_q | full weight에서 residual 제거 |
| REF_S2_FULL | 지정한 기존 29K 결과, C,D | 1+.25 d_e+.25 d_e r_q | 최종 reference, 새 run 없음 |

**α=0은 Teacher cue 제거가 아니다.** B3에도 d_e가 남는다.
B2는 spatial cue 제거, B1은 adaptive weight 전체 제거다.
B0/B1은 T forward·calibration 없이 학습하되, loss mask는 다른 B case와 같은
Student-L4/Teacher-bicubic support 교집합을 계산해 보존한다.
mask를 바꾸어 loss 제거 효과와 supervision 영역 차이를 섞지 않는다.
calibration을 쓰는 case는 같은 donor와 동일 train-only 통계를 사용한다.

다음은 각 반복 안에서 핵심 case들 다음에 수행하는 메커니즘 ablation이다. 제안 method에 새 기능을 추가하는 실험이 아니다.

| Case | Full에서 변경 | 질문 |
|---|---|---|
| S2_D1_GLOBAL_RANK | edge-group별 CDF → train 전체 단일 CDF | edge conditioning의 효과 |
| S2_D2_NO_E_CUE | w=1+.25 r_q | d_e를 완전히 제거했을 때의 효과 |
| S2_D3_MEAN_WEIGHT | full w를 각 영상 유효영역 평균 상수로 변경 | 영상 내 pixel별 가중 배치의 효과 |
| S2_D4_TWO_PANS | C,D → C,P | signed difference 표현의 최적화 효과 |

D1은 per-patch CDF가 아니라 train-global CDF이며 범위 [0,1]을 유지한다.
D2에서는 T prediction/e-calibration이 필요하지 않지만 동일 A와 q calibration은 필요하다.
D3는 sample 평균 weight를 보존하며 이미지 간 난도 가중은 남는다.
D4는 관측 정보가 추가되는 비교가 아니다. C,D와 C,P는 서로 선형 복원 가능하다.
추가 PAN 열도 0으로 초기화하여 같은 seed에서 처음의 출력 함수를 맞춘다.
무학습 sanity check에서는 `W'_C=W_C+W_D, W'_P=-W_D` 변환으로 동일 출력인지 확인한다.
이후 차이는 parameterization/최적화·정규화의 영향으로 해석한다.

## 5. 반복, 우선순위, 해석 범위

| 서버 | 담당 | R1 seed | R2 seed | R3 seed | 신규 학습 수 |
|---|---|---:|---:|---:|---:|
| s3 | Stage 1 5 case | 261016101 | 261016102 | 261016103 | 15 |
| s1 | Stage 2 핵심 5 case | 261018101 | 261018102 | 261018103 | 15 |
| s1 | Stage 2 메커니즘 4 case | 같은 R1 | 같은 R2 | 같은 R3 | 12 |

**핵심 10 case/30 runs, 메커니즘 4 case/12 runs, 전체 14 case/42 runs**다.
이는 신규 ablation 수이며 기존 두 reference를 3회 반복으로 부풀려 세지 않는다.
s3는 A0→A1→A2→A3→A4, s1은 B0→B1→B2→B3→B5 후 D1→D2→D3→D4를 수행한다.
반복별로 위 순서를 한 번씩 순회해 첫 반복의 전체 component 결과를 먼저 얻는다.
**s1은 메커니즘 case까지 포함한 9개 전체 R1 → 9개 전체 R2 → 9개 전체 R3**,
**s3는 5개 전체 R1 → 5개 전체 R2 → 5개 전체 R3**다.
`priority`는 비교 목적의 메타데이터일 뿐 실행 순서를 재정렬하지 않는다.
실행기는 `config/panda_wv3_ablation/plan.json`의 case 순서를 반복별로 순회한다.
`runs.csv`와 서버별 `cases_s1.csv`/`cases_s3.csv`는 이 실행 큐에서 생성했다.

각 ablation은 개별 3개 값과 평균±sample SD를 표시한다.
같은 반복 seed의 ablation끼리는 paired 차이를 계산한다.
**Full reference는 현재 고정한 R1 한 개**이므로 그 값에 3회 평균이나 분산을 붙이지 않는다.
Stage 1의 s3 세 seed와 s1 고정 full donor 사이에는 donor seed/server 차이가 남는다.
이번 결과만으로 Stage 1 loss의 작은 차이가 유의하다고 확정할 수 없다.
이는 지정 endpoint를 재학습하지 않는 현재 범위의 한계다.
Stage 2도 fixed donor 조건부 Student 변동이며 Stage 1까지 독립적인 3회 반복이 아니다.
향후 엄밀한 Full 대조 3회가 필요하면 별도 범위로 추가하며 이번 queue에는 포함하지 않는다.

기존 초안 C1/C2/C3(제거 donor → full Stage 2)는 제외한다.
s3가 Stage 2까지 학습하거나 s3 weight를 s1로 옮겨야 하므로 새 서버 분리 방침과 맞지 않는다.
RGB/L4 통일 실험, LR sweep, rotation loss, SWA도 이번 component queue에 넣지 않는다.

## 6. Sheet 표시·평가·실행 검증

Sheet tab은 `WV3-ablations`, **Experiment는 case 의미를 포함**한다.

- `WV3-ABL-S1-A3-REC-SPA R1` — Approach에 `S1; rec+2spa; eps=0; mean8; bicubic; A/Tlr=1e-5/1e-4; C50`.
- `WV3-ABL-S2-B2-E-ONLY R1` — `S2; C+DIFF; a=.25,g=0,b=0; S100; UBIL; L4/Tbic; lr=1e-4; C30`.
- Server/Repeat/Seed/Case_ID/Run_ID/Selected_step/donor SHA는 별도 기존 열·provenance에도 기록한다.

기존 목표는 **HQNR≥.9585 AND ERGAS≤2.040**이다.
Stage 2의 동시 통과 step은 모두 게시하고, run 대표는 HQNR≥.9585인 것 중 최저 ERGAS,
없으면 최고 HQNR로 선택한다. Stage 1 대표는 validation RRE50 → RRE90 → ValERGAS → 이른 step으로 선택한다.
미달·실패 case도 status/대표 결과에서 숨기지 않는다.
기존 reference 행은 이름·수치·날짜를 덮어쓰지 않는다.
Sheet의 대표 checkpoint와 논문용 고정 50K/29K 표는 별도 산출물로 분리한다.
조건을 통과한 최고값만으로 component 기여를 판단하지 않는다.

공식 RR/FR single-forward 평가, 원본 GT/PAN 참조, 정상 데이터 split/전처리를 유지한다.
QB의 8/9/10K HQNR<.90 조기 중단은 WV3에 적용하지 않는다.
50K/29K/30K, validation 선택, 목표 통과 및 대표 weights를 보존하고 provenance를 남긴다.
정합 support 위반·nonfinite는 실패를 기록하며 shift clamp나 평가 crop 변경으로 숨기지 않는다.

`panda_wv3_ablation`에 Stage 1 no-A/nonlearned/disabled-loss와 Scharr subpixel 검색,
Stage 2 global-CDF/no-e/mean-weight/two-PAN을 명시적으로 구현했다.
공유 초기화·RNG·gradient·동결 donor·완전 resume·순회 큐·Sheet 게시 규칙을 CPU 테스트로 검증했다.
원본 29K weights를 새 inference 경로로 RR20/FR20 재평가한 결과 HQNR .9603361215 / ERGAS 2.0340588129로
원본 기록과 각각 약 1.2e-9 / 8.3e-8 차이였다. [재평가 근거](../results_log/assets/2026-11-02_wv3-component-ablation/reference_replay_cpu.json).
이는 신규 ablation 결과가 아니다. 실제 GPU 학습과 결과 게시는 별도 실행 이후 시작한다.
실행 등록의 원본은 `config/panda_wv3_ablation/plan.json`, source 검증은 `release.json`이다.
results의 `cases.json`/`runs.csv`는 이 등록 내용을 읽기 편하게 기록한 사본이다.
