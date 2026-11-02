# WV3 component ablation — method 흐름 기준 설계

> **이 문서는 초기안이며 아래 내용으로 실행하지 않는다.** 사용자 지정 checkpoint와
> s3=Stage 1 / s1=Stage 2 / weight 이동 없음 방침을 반영한
> [최신 anchored ablation 설계](PANDA_WV3_Anchored_Ablation_Plan_2026-11-02.md)로 대체한다.
> 아래의 신규 RGB/L4 통일, γ=.5, 서버별 전체 pipeline 반복, 54/66 jobs는 최신안에 적용하지 않는다.
> 연결된 `cases.json`도 최신 V3 구현 기록으로 갱신했다.

2026-11-02. 기준 원고는 [method (1).tex](<method_writing/method (1).tex>),
상세 입력·support 규약은 [10월 30일 method 명세](method_writing/PANDA_Method_2026-10-30.md)다.
사용자 확인: Scharr uniform shift는 **각 영상에서 하나의 이동량을 추정하여 영상 전체에 적용**하는 방식이다.
이 문서는 실행 전 설계다. 기존 큐를 바꾸거나 학습을 시작하지 않았다.
[기계 판독용 case 목록](../results_log/assets/2026-11-02_wv3-component-ablation/cases.json)은 proposal이며 실행기 입력이 아니다.

**설계의 핵심**

원고의 기여는 (1) 학습 가능한 정합과 그 감독, (2) signed PAN correction residual,
(3) reference reconstruction error와 edge-conditioned spatial discrepancy를 이용한 GT 가중이다.
따라서 정합 비교, Stage 2 비교, 전체 pipeline에서의 Stage 1 제거 비교를 나눈다.
서로 다른 Teacher의 품질과 Stage 2 가중을 동시에 바꾼 결과를 순수한 정합 효과로 부르지 않는다.

Stage 1의 원고 $L_{\mathrm{spa}}$는 코드의 `L_abs`에 대응한다.
`PAN Aligner + abs, w/o controlled shift`는 **$L_{rec}+\lambda_{spa}L_{spa}$**이며,
복원 감독까지 제거한 abs-only Aligner가 아니다.
Stage 2에는 별도의 Student spatial loss가 없고 아래 GT 목적함수 하나만 있다.

$$L_{AF}=\mathcal R_m[(1+\alpha d_e+\gamma d_e r_q)e_S].$$

**현재 원고와 기존 실험을 맞추는 전제**

- 신규 Stage 1·2의 구조 참조는 GT RGB 평균으로 통일한다. WV3 RGB 인덱스는 0-based `(4,2,1)`이다.
  복원 target/output과 Aligner MS 입력은 8밴드 전체다.
- $I_{ms}^{lr}=\operatorname{interp23}(I_{ms})$를 한 번 생성하여 Aligner·Teacher·Student·residual skip에 공유한다.
  기존 Stored LMS와 거의 같았다는 관측만으로 실행 텐서 공유를 대체하지 않는다.
- native PAN, controlled ε, frozen reference, Student 모두 Lanczos-4를 쓴다. learned correction은 scale 1이다.
- 기존 WV3 G8/MATURE donor는 mean8 참조·bicubic이다. 이번 논문식 component test의 donor로 그대로 재사용하지 않는다.
  새로운 RGB/L4/common-MS donor와 calibration을 만들고 별도 method revision을 기록한다.
- 현재 QB/GF2 RGB 구현은 4밴드 제약이 있다. WV3 8밴드 대응과 위 MS 공유를 검증한 후 실행해야 한다.

**A. 정합 방식·Stage 1 감독: final reconstruction 조건을 고정한다**

아래 모든 최종 U-Net은 새로 초기화하여 30K 학습한다. 입력은 9채널 `[PAN_used, I_ms^lr]`,
loss는 무가중 GT L1, backbone은 같은 W112/D123/UBIL이다. residual과 adaptive weight는 끈다.
Learned A를 사용하는 A2–A5만 각각 Stage 1 A/T를 50K 학습하고 A를 동결한다.
최종 30K U-Net에 Stage 1 T weights를 전달하지 않는다.

| ID | 정합 | Stage 1 objective | final U-Net PAN | 검증 질문 |
|---|---|---|---|---|
| A0 RAW | 없음 | 없음 | 원 PAN | 단순 U-Net 기준 |
| A1 SCHARR | 영상별 비학습 Scharr global translation | 없음 | Scharr 보정 PAN | 비학습 정합만으로 개선되는가 |
| A2 REC | learned Aligner | $L_{rec}$ | learned corrected PAN | 복원 gradient만으로 배우는 정합 |
| A3 REC_SPA | learned Aligner | $L_{rec}+\lambda_{spa}L_{spa}$ | 동일 | controlled shift 없이 직접 구조 감독의 효과 |
| A4 REC_EPS | learned Aligner | $L_{rec}+\lambda_\epsilon L_\epsilon$ | 동일 | spatial 감독 없이 shift consistency의 효과 |
| A5 ALIGN_FULL | learned Aligner | 세 항 전부 | 동일 | 완전한 정합 학습 |

A2/A3/A4/A5는 spatial/controlled-shift 항의 2×2 비교다.
A3→A5가 사용자가 요청한 controlled-shift 기여, A4→A5가 spatial 감독의 기여다.
각 donor의 Teacher 결과도 기록하지만 위 표의 **주 비교값은 모두 final U-Net의 30K 결과**다.
Stage 1 Teacher 50K 숫자와 Student 30K 숫자를 한 열에서 섞어 비교하지 않는다.
Learned alignment의 추가 Stage 1 비용은 별도 training cost로 공개한다.

Scharr baseline은 PAN과 **관측 MS에서 만든** $I_{ms}^{lr}$의 RGB 평균을 비교한다.
학습·검증·추론에서 GT를 shift 추정에 넣지 않는다. raw abs-Scharr XY 차이를 같은 내부 영역에서 최소화한다.
Train에서는 기존 augmentation 이후의 입력에 적용한다. PAN만 이동하고 GT/MS/output은 이동하지 않는다.
현재 `panda_method/edge_match.py`는 integer-only이므로 이를 유일한 대조로 쓰면
learned continuous shift의 이득에 정수 양자화 한계가 섞인다.
**A1은 정수 coarse search + subpixel refinement를 기본안으로 둔다.** 기존 integer matcher는 보조 진단으로 남긴다.

초기 등록안은 각 축 `[-3,3]` 현재 PAN 픽셀의 정수 검색, 최적점 주변 ±1px의 .25px fine search다.
후보는 같은 범위에 제한하고 모두 동일 Lanczos-4·고정 ROI8로 평가한다.
동점은 최소 norm→L1→(dy,dx) 순으로 고정한다. 이 검색 상한은 ε disk 반경과 별개다.
정수 후보들의 warp/edge 값은 기존 matcher와 일치하는지 검사한다.
본 실험 전 train/validation에서 boundary-hit 비율과 .25px 세분화 적합성을 확인한다.
범위 부족이면 실행 전 명세를 고치며, test 성능으로 범위를 고르거나 learned A를 같은 값으로 clamp하지 않는다.
남는 검색 해상도·범위 차이는 baseline의 한계로 보고한다. 추론 시간에는 검색 비용을 포함한다.

**B. Stage 2: 같은 ALIGN_FULL donor와 calibration을 공유한다**

$D=\tilde I_{pan}-I_{pan}$, $e$ column은 additive $\alpha d_e$ 항,
$eq$ column은 interaction $\gamma d_e r_q$ 항이다.

| ID | Student PAN 입력 | weight | 직접 비교 |
|---|---|---|---|
| B0 = A5 | $\tilde I_{pan}$ | 1 | 중복 실행하지 않는 기준 |
| B1 DIFF_GT | $[\tilde I_{pan},D]$ | 1 | A5 대비 correction residual |
| B2 DIFF_E | 동일 | $1+\alpha d_e$ | B1 대비 additive reference error |
| B3 DIFF_EQ | 동일 | $1+\gamma d_e r_q$ | B1 대비 interaction 항 |
| B4 FULL | 동일 | $1+\alpha d_e+\gamma d_e r_q$ | 전체 method |
| B5 NO_DIFF_AF | $\tilde I_{pan}$ | 전체 weight | B4 대비 residual 제거 |

모든 입력에는 같은 8밴드 $I_{ms}^{lr}$가 추가된다. D 없음은 9채널, 있음은 10채널이다.
A5/B1/B5/B4로 residual×adaptive weighting의 2×2 interaction을 볼 수 있고,
B1/B2/B3/B4로 α항×γ항을 분리할 수 있다. 이를 위해 B5를 생략하지 않는다.

**α=0은 Teacher/reference-error cue 제거가 아니다.** B3에도 $d_e$가 남는다.
γ=0은 spatial cue 제거이며 B2, α=γ=0은 무가중 B1이다.
β는 전부 0이고 Teacher imitation을 새로 추가하지 않는다.
B0/B1에서는 Teacher forward·e/q calibration을 학습 loss에 사용하지 않는다.
다른 B case는 동일 A/T와 동일 train-only calibration을 사용한다.

Residual 추가분은 첫 3×3 Conv의 `112×1×3×3=1,008`개 weight다.
9↔10채널 비교는 공통 PAN/MS channel과 나머지 network의 초기값을 명시적으로 맞추고,
residual channel만 별도 RNG로 초기화한다. 단순히 같은 seed를 전달하여 첫 Conv 이후 RNG가 달라지는 것을 피한다.
일반 9채널 baseline을 유지하고 임의의 zero-channel 구조로 바꾸지 않는다.

**C. Stage 1 항 제거를 전체 method에서도 재확인한다**

A 묶음은 Stage 1이 만든 A의 정합 효과를 분리한다. 실제 method에서는 T가 $d_e$도 만들기 때문에
다음 비교가 추가로 필요하다. 모두 B4와 동일한 10채널·전체 adaptive weight를 사용한다.

| ID | Stage 1 donor | Stage 2 | 의미 |
|---|---|---|---|
| C1 FULL_NO_EPS | A3의 A/T | B4와 동일 | 전체 method에서 $L_\epsilon$ 제거 |
| C2 FULL_NO_SPA | A4의 A/T | B4와 동일 | 전체 method에서 $L_{spa}$ 제거 |
| C3 FULL_REC_ONLY | A2의 A/T | B4와 동일 | Stage 1 보조 감독 두 항 제거 |

각 C case는 자신의 Stage 1에서 나온 **A/T 한 쌍**과 그에 맞는 새 calibration을 사용한다.
다른 donor의 A와 full T를 조합하지 않는다. A 묶음에서 학습한 donor를 재사용하므로 추가 Stage 1은 없다.
C1/C2/C3/B4는 Stage 1 두 항의 전체 pipeline 2×2 비교다.

**D. 메커니즘 검증 — 새로운 제안 method가 아닌 제거·대체 대조**

| ID | B4에서 바꿀 것 | 검증 질문 |
|---|---|---|
| D1 GLOBAL_RANK | $F_{j(g)}(q)$ → 전체 train의 단일 $F(q)$ | edge-strength conditioning의 효과 |
| D2 SPATIAL_ONLY | weight → $1+\gamma r_q$ | reference error $d_e$를 완전히 없애면 어떠한가 |
| D3 MEAN_WEIGHT | sample마다 유효 영역의 $\bar w_n=\mathrm{mean}_{m_n}w_{AF}$를 모든 유효 픽셀에 동일하게 적용 | 픽셀별 위치 선택의 효과와 평균 loss 크기 효과 구분 |
| D4 TWO_PANS | $[\tilde I_{pan},D]$ → $[\tilde I_{pan},I_{pan}]$ | signed residual 표현과 단순 두-PAN 입력의 차이 |

D1의 $F(q)$ 역시 train-only·고정 통계이며 per-patch CDF로 바꾸지 않는다. 범위는 동일 [0,1]이다.
D2는 Teacher prediction이 필요하지 않는다. A와 GT로 q/rank를 만들고 e-calibration을 제거한다.
D3는 sample별 평균 weight, warp support, 이미지 간 난도 차이를 보존하면서 픽셀 내부 분포만 없앤다.
따라서 모든 adaptive weight 효과를 증명하는 대조라기보다 **sample 내부 공간 배치**의 대조다.
LR 배율을 조절하는 것으로 이 대조를 대신하지 않는다.

D4와 residual 입력은 서로 복원 가능하고 첫 연산이 linear Conv이므로 표현 가능한 정보·함수족은 같다.
따라서 residual이 새로운 관측 정보를 추가한다고 주장하지 않는다.
독립적인 표준 초기화로 훈련하면 최적화·정규화·parameterization 차이를 비교한다.
별도 무학습 sanity check에서는 conv weight를
$W'_c=W_c+W_D, W'_P=-W_D$로 변환하여 두 입력의 출력을 일치시킨다.
이 관계를 보이지 않는 새로운 정합 feature라는 주장으로 확대하지 않는다.

과거 HARD36에서 s3 `AG000 R1`은 30K에 HQNR .959608/ERGAS 2.033885였다.
같은 seed의 `G000 R1`은 .959433/2.036384였다.
[저장된 10월 29일 Sheet snapshot](../results_log/assets/2026-10-29_wv3-hard36-sheet.json)의 값이며 최신 live Sheet 재조회는 아니다.
기존 donor/방법과 신규 통일 명세는 다르므로 신규 결과로 재검증한다.
적어도 과거 목표 통과만으로 adaptive weight가 필요하거나 더 좋았다고 결론 내릴 수는 없다.
낮은 값의 ablation도 누락 없이 기록한다.

**고정 recipe와 선택·평가**

- WV3 전체 8밴드, Teacher W112/D123 BASE, final U-Net W112/D123 UBIL.
  U-Net sampling 구조 자체는 이번 component 비교에서 바꾸지 않는다.
- Stage 1 50K: A LR1e-5/T LR1e-4, λ_spa=2, λ_ε=.003(항 존재 시), ε disk R=3px.
  이는 기존 WV3 recipe를 출발점으로 고정한 것이며 새 RGB/L4에서 최적이라고 주장하지 않는다.
  QB의 λ_ε=.03을 근거 없이 가져오지 않는다.
- Final U-Net 30K: LR1e-4, α=.25/γ=.5(항 존재 시), β=0.
- 두 단계 AdamW WD=.01, batch48=micro12×4, warmup100, cosine floor0, FP32.
  flip/90° augmentation은 기존대로 동일하게 적용한다. 회전 일관성 loss·TTA·SWA·band-weighted loss는 없다.
- 항을 끄는 case는 그 gradient와 불필요한 추가 forward를 제거한다. data/augmentation/ε RNG는 독립 유지한다.
  loss on/off 때문에 나머지 초기화나 샘플 순서가 바뀌지 않게 한다.
- Stage 1은 full50K 후 같은 validation RRE50→RRE90→ValERGAS→이른 step 규칙을 적용한다.
  selector의 target·warp·MS 입력도 새 RGB/L4/common-MS와 맞춘다. test로 donor를 고르지 않는다.
  선택된 donor뿐 아니라 고정50K alignment/Teacher 수치도 같이 남긴다.
- **주 ablation 수치는 final U-Net 고정30K**의 paired 결과다. 25–30K 곡선은 안정성 진단이고 독립 반복 수가 아니다.
  필요하면 사전 선언한 validation-only Student 선택 결과를 별도 열에 둔다.
  test HQNR 조건에 맞춘 run 대표 checkpoint를 주 ablation 인과 비교값으로 사용하지 않는다.
- 공식 RR20/FR20 원본 참조·single-forward 평가를 유지한다. HQNR/ERGAS 외 D_s/D_lambda/PSNR/SAM/SCC를 함께 보고한다.
  Aligner에는 validation RRE와 controlled ε response error를 별도로 기록한다.
  RRE는 물리적 정합 GT가 아닌 proxy이며 ε response는 native alignment 정확도와 다른 성질이다.
- 학습 support는 기존 정의를 명시적으로 유지한다. Stage 2 B 비교는 동일 donor여서 같은 mask다.
  A 비교는 shift에 따라 warp support가 달라질 수 있으므로 valid fraction을 기록하고
  공통 ROI의 진단 지표를 병기한다. 이를 공식 평가의 새 crop/mask로 사용하지 않는다.
  따라서 A의 차이는 support 처리를 포함한 alignment strategy 효과이며 순수한 이동 효과로 단정하지 않는다.
  mask 차이가 해석을 좌우하면 별도 후속으로 같은 고정 학습 ROI를 사용하는 대조를 설계한다.
- WV3는 QB의 8/9/10K HQNR<.90 조기 중단 규칙을 적용하지 않는다. 낮은 성능으로 case를 생략하지 않는다.
  실제 수치/support 오류는 실패 근거를 남기며 shift clamp나 임의 ROI 축소로 숨기지 않는다.

**서버 배치·우선순위·반복 수**

s1/s3에 case 종류를 나눠 맡기면 서버·donor 효과가 component 효과와 섞인다.
대신 각 서버가 동일 case 묶음을 자기 Stage 1부터 수행한다. 한 묶음 안에서는 A/T donor를 필요한 case끼리 공유한다.
이는 서버 간 동일 donor 강제가 아니며 weights를 전달하지 않는다.

| block | 서버 | Stage 1 seed | final U-Net seed |
|---|---|---:|---:|
| R1 | s1 | 261102101 | 261102111 |
| R2 | s3 | 261102301 | 261102311 |
| R3 | s3, R2 이후 | 261102302 | 261102312 |

같은 block 내 A2/A3/A4/A5는 같은 A/T 초기값·샘플 순서를 사용한다. final U-Net도 case 간 같은 초기값·stream을 쓴다.
모든 핵심 비교는 같은 block 안에서 Δ를 계산하고 세 block의 평균·표준편차와 개별 값을 함께 제시한다.
서버 간 최고값만 뽑아 비교하지 않는다. Student seed 두 개를 공유 donor 하나에서 반복했다면
그 결과를 Stage 1까지 독립적인 반복으로 세지 않는다. 위 제안은 block마다 donor를 새로 학습한다.

1. 우선 A0/A1 및 네 donor 준비 → A2–A5 → B1/B4: 기본 정합과 residual·전체 방식의 효과 확인.
2. B2/B3/B5: Stage 2 항별 기여와 residual×weight 상호작용 완성.
3. C1/C2/C3: 논문에 제시할 전체 method의 Stage 1 제거 비교 완성.
4. D1–D4: conditioning·reference cue·weight 크기·residual parameterization 해석.

A+B는 중복 제외 **11개 final-model case**, C까지 **14개**다. D까지는 **18개**다.
각 block의 Stage 1 donor는 네 개뿐이다. 3개 block 기준:

- A+B+C: Stage 1 12회 + final U-Net 42회 = **54개 학습 job**.
- D까지: Stage 1 12회 + final U-Net 54회 = **66개 학습 job**.

기존 통일 전 checkpoint를 재활용하여 이 반복 수를 충족했다고 세지 않는다.
새 통일 구현의 첫 full block으로 실행 정확성을 확인하되, 그 결과가 나쁘다는 이유로
full만 다시 튜닝하고 ablation은 예전 recipe에 남기는 식의 비교는 하지 않는다.

**Sheet·보고 및 원고 확인 사항**

`WV3-ablations`에 server/block/donor seed/Student seed/Stage1 loss flags/residual flag/α/γ/rank mode를 기록한다.
기존 목표 **HQNR≥.9585 AND ERGAS≤2.040**, 모든 동시 통과 step과 대표 한 개의 게시 규칙을 유지한다.
논문용 고정30K 표는 별도 local artifact로 만들어 Sheet 대표값과 구분한다.
모든 평가 weights를 저장하고 실패·미달 case도 완료 목록에서 누락하지 않는다.

원고에서는 weight를 $w_{AF}$로 정의하고 마지막 loss 식에는 미정의 $w_{rec}$를 사용한다.
두 식을 $w_{AF}$로 맞추면 된다. 이번 설계 작업은 원고 파일 자체를 수정하지 않았다.
Fig.5의 controlled-shift response residual과 Stage 2의 공간 q는 서로 다른 신호이므로
ablation 설명과 도표에서도 구분한다.
