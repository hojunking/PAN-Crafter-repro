# s1 — PAN 정합 능력·Recon/Consistency 4분면·KD 참조 유용성 검증

**문서 ID:** `EQREC4-S1-v1`  
**작성일:** 2026-09-14  
**실행 서버 / 데이터:** s1 / WV3  
**기준 방법:** L1E4, W112–D123, PAN+MS 9ch, λ_off=1e-4  
**기간:** 실제 실행 시작 T0부터 약 20시간을 목표로 한 작업 묶음. 20시간을 채우거나 정확히 끊지 않는다.  
**목적:** 새 최고점을 찾는 sweep이 아니라, 현재 방법이 무엇을 학습했는지와 Teacher cue로 사용할 수 있는지를 검증한다.  
**문서의 지위:** 기존 보고서의 관측과 아래 신규 실험 설계를 구분한 실행·분석 지시서. 서버 실행이나 새로운 결과가 이미 완료됐다는 뜻은 아니다.

---

## 0. 실행자가 먼저 읽을 요약

이번에는 **네 개의 새 모델을 만들어 네 loss 조합을 강제로 만드는 실험을 하지 않는다.** 같은 frozen Teacher의 같은 checkpoint에서 각 sample의 native reconstruction error `e_T`와 추가 PAN 이동에 대한 offset consistency error `q_T`를 측정하고, 그 sample들을 네 집단으로 나눈다.

| 집단 | Native reconstruction error | Offset consistency error | 핵심 질문 |
|---|---|---|---|
| **E↓C↓** | 낮음 | 낮음 | 두 과제에 모두 강한가? Student에게도 유용한가? |
| **E↓C↑** | 낮음 | 높음 | U-Net 보상, 낮은 PAN 의존성, native에만 맞춘 fitting 중 무엇인가? |
| **E↑C↓** | 높음 | 낮음 | 절대 위치 기준, spectral 복원, detail 복원 중 무엇이 막혔는가? |
| **E↑C↑** | 높음 | 높음 | 공통 난이도인가, 정합·복원 두 문제가 함께 있는가? |

여기서 낮음/높음은 calibration 분포의 상대적 구분이다. `E↓C↓=좋은 Teacher`, `E↑C↑=쓸모없는 Teacher`라는 정답 label을 미리 부여하지 않는다. **Teacher보다 Student가 더 잘할 수 있고, 높은 consistency error를 가진 Teacher도 native target으로 유용할 수 있다.**

### 이번에 꼭 남길 답

1. **정합 능력:** 알려진 추가 이동의 상쇄, 실제 native pair의 위치 개선, 복원에서의 유용성을 각각 검증한다.
2. **네 집단의 원인:** loss 크기를 설명하는 데 그치지 않고, correction 치환·출력 stress·band/edge 분석·국소 업데이트로 설명을 구분한다.
3. **KD 연결:** 같은 Teacher error와 Teacher advantage 안에서도 consistency가 soft/hard 선택의 실제 이득을 추가로 설명하는지 확인한다.
4. **실패도 답이다:** sample 부족, provenance 부족, proxy 불일치, KD 효과 미검출을 각각 기록한다. 효과 미검출을 곧바로 보편적인 무효로 선언하지 않는다.

### 실행 순서

`G00 재현/자료 gate → D10 sample atlas → D20 기하 검사 → D30 correction 개입 → D40 stress/스펙트럼 → D50 gradient 개입 → K10 hard/soft 진단 → K20 조건부 gate pilot → 보고`

새 width/depth, local flow 학습, λ 추가 탐색, PAN reconstruction, feature KD, GT variance는 이번에 추가하지 않는다. **L1E4를 고정한 결정은 유지**한다.

---

### 문서 탐색

| 내용 | 위치 |
|---|---|
| 고정 모델·자료 분할·측정량 | [모델 계약](#model-contract) · [데이터 분할](#data-split) · [측정량](#metric-definition) |
| 네 집단과 실제 정합 검증 | [D10](#d10-atlas) · [D20](#d20-geometry) · [D30](#d30-intervention) · [D40](#d40-stress) · [D50](#d50-gradient) |
| Soft/hard와 KD 연결 | [K10 대응 실험](#k10-utility) · [K20 pilot](#k20-pilot) · [gradient 구현](#gradient-routing) |
| 판정·운영·재현 | [분석 규칙](#analysis-rules) · [산출물](#outputs) · [실행 큐](#execution) · [YAML](#config) · [근거](#sources) |

## 1. 이번 설계가 출발하는 근거와 아직 모르는 것

### 1.1 보고서에서 확인된 관측

| 관측 | 원문 수치 / 상태 | 이번 설계에 주는 의미 |
|---|---|---|
| L1E4 묶음은 현재 무정합 baseline보다 유리 | PALSV18 보고서 평균 HQNR 0.95597, 무정합 대비 +0.00457 | 고정 방법을 버리지 않고 기전을 검증한다. [S1 §1.1] |
| λ 증가에 따른 정합 지표와 HQNR은 단조롭게 일치하지 않음 | L1E4 EPE .474 / HQNR .95597, L3E4 EPE .365 / HQNR .95374 | 작은 q가 높은 native 품질을 보장한다고 가정하지 않는다. [S1 §1.1, §1.3] |
| learned correction의 개입 효과가 관측됨 | L3E4·1234: learned .9579, zero .9465, wrong-sign .9416 | 선택한 **L1E4**에서 같은 개입을 반복한다. [S1 §1.4] |
| scene-shuffle은 거의 무해했지만 constant는 다름 | 같은 case: shuffle .9578, constant .9470 | 동적 추정 무용/상수 대체 가능을 확정하지 않는다. [S1 §1.4] |
| fSCC 비용이 있음 | L1E4는 P0 대비 3 seed 모두 fSCC 하락 | 위치 개선과 smoothing/detail 손실을 분리한다. [S1 §4; S2 §2.6] |
| 정합 검증이 부분적으로만 실행됨 | V2–V4 일부 case, blur_status=unmatched_blur_control | 이 캠페인은 비어 있던 검증을 채운다. [S1 §4] |
| spectral 차이만으로 apparent shift가 측정된 예가 있음 | 기하 동일 band 분할: WV3 .358 px, 전밴드/절반 .176 px | 구조 추정값을 물리적 GT로 바로 쓰지 않는다. [S4 §8.5] |

**주의:** 위 표의 EPE/HQNR은 보고서 집계이며 sample-level 상관 자료가 아니다. Probe·split·checkpoint가 다른 과거 수치를 같은 척도로 연결하지 않는다. [S3 §10]

### 1.2 과거 해석을 이번의 전제로 고정하지 않는다

- “실측 shift를 넣었으니 모든 misalignment를 완전히 해결했다”는 주장은 검증되지 않았다.
- 09-04 전역 sweep은 **MS 조건을 PAN 방향으로 이동**한 과거 backbone 실험이다. 현재 PAN-only correction과 구분한다. [S5 §3]
- 09-12의 “PAN 이동은 항상 손해”라는 해석은 이후 L1E4 결과로 범위가 수정됐다. [S2 §3]
- 0.45–0.7 px의 이득 창은 보고서의 실험군 해석이다. **0.55 px를 정답이나 cap으로 사용하지 않는다.**
- 작은 q는 native 절대 보정의 정확성이나 Teacher 출력 uncertainty가 아니다. [S3 §4.5, §11.5]

### 1.3 검증할 가설

| ID | 검증 문장 | 단위 / 직접 endpoint |
|---|---|---|
| **H1** | 같은 checkpoint 안에서 q가 낮은 sample이 native 복원도 좋은가? | RR: q–Teacher L1, FR: q–raw-original HQNR |
| **H2** | q가 낮으면 알려진 추가 PAN 이동에 대한 복원 저하가 작은가? | 같은 native 대비 ΔL1 / ΔHQNR, 고정 ROI·고정 참조 |
| **H3a** | 같은 U-Net 가중치에서 learned correction이 zero보다 유리한가? | sample별 learned−zero 이득 |
| **H3b** | q가 H3a 이득의 크기를 예측하는가? | q–개입 이득의 조건부 관계 |
| **H4** | e_T와 a_T 외에 q가 Teacher 참조의 실제 유용성을 설명하는가? | 대응 hard/soft 업데이트의 별도 validation 이득 |
| **Hspec** | loss의 엇갈림이 spectral/texture/보간 특성과 연결되는가? | band·edge·동일 기하 spectral 대조 및 출력 민감도 |

Hspec은 물리적 원인 하나를 확정하는 가설이 아니다. **데이터–모델 상호작용을 구분**하는 탐색 질문이다.

---

<a id="model-contract"></a>
## 2. 모델·자산·학습 의미를 고정한다

### 2.1 현재 forward

\[
M=U_{bicubic}(S),\quad c_0=A_\phi(P,M),\quad
\widetilde P_0=\mathcal W(P,c_0),\quad
\hat Y_0=M+F_\theta([\widetilde P_0,M]).
\]

\[
P_\epsilon=\mathcal W(P,\epsilon),\quad
c_\epsilon=A_\phi(P_\epsilon,M),\quad
L_{off}=\operatorname{mean}_{B,2}|c_\epsilon+\epsilon-\operatorname{sg}(c_0)|.
\]

| 항목 | 고정 내용 |
|---|---|
| Backbone | W112, depth=[1,2,3], 기존 `fixed` class |
| 입력/출력 | PAN 1 + bicubic MS 8 = 9ch, residual 8ch |
| Aligner | 기존 약 0.105M dual-stem global CNN; 출력 `(dy,dx)` 두 숫자 |
| Aligner view | margin4, crop 후 sample·band별 z-score, wrapper에서 한 번만 |
| Warp | bicubic / border / align_corners=False, FP32 offset/warp |
| 부호 | `W(P,c)[y,x] = P[y+c_y,x+c_x]` |
| 좌표 | MS condition·MS base·GT·출력 배열은 그대로; PAN만 sampling |
| 학습 | 기본 Rec→A/F, Off→A only, λ_off=1e-4, 홀수 optimizer update에서 Off |
| Training jitter | b=2 HR px, 0 중심 원판 면적 uniform; sample별 global shift |
| U-Net 학습 입력 | native PAN을 c0로 보정한 입력만; jittered PAN은 학습 시 U-Net에 미전달 |
| 이번 stress | frozen inference에서만 jittered PAN의 U-Net 출력도 평가 |

원래 L1E4는 A/F 공동 학습이다. **이번 진단에서 모델을 freeze하는 것은 평가를 위한 것이며, 채택 방법을 frozen-aligner 방식으로 바꾸는 것이 아니다.** [S3 §2–5]

### 2.2 Frozen model registry

| 모델군 | Seed | 우선 checkpoint | 역할 |
|---|---|---|---|
| **L1E4** | 1234 / 7777 / 2025 | **best_raw 및 exact50K last 모두** | 핵심: 채택한 방법의 같은-checkpoint 정합·품질 |
| L000 / NF16 P2 | 같은 3 seed | best_raw 및 exact50K last | 약한 consistency가 추가한 정보 |
| L3E4 | 같은 3 seed | best_raw, 필요 시 exact50K last | q는 좋지만 native 품질/안정성이 다른 대조 |
| P0 | 같은 3 seed | best_raw 및 last | 전체 방법/무정합 참고; q를 0으로 채우지 않음 |
| N2 donor | 2025 | **exact50K last** | 높은 변위 반응의 대조; 최종 Teacher와 구분 |
| L1E2 / NF16 P3 | 1234 | 존재하는 정확한 ckpt만 | 선택: 강한 consistency 대조 |

**Primary Teacher:** `PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1 / best_raw`의 A+U 한 쌍. 기존 고정 자산을 재사용하며 이번 분석 점수로 다시 Teacher를 고르지 않는다. 정확한 선택 update와 hash는 G00에서 조회한다.

**반복 Teacher:** 같은 family S1234/S7777 best_raw. Exact50K는 별도 strata로 반복한다. **Best U-Net과 last aligner를 섞지 않는다.** N2의 초기 step1010 best를 last 대신 쓰지 않는다.

실제 파일명·extension은 manifest에서 찾는다. 문서에 없는 `.pt` 경로를 만들어 있다고 가정하지 않는다. 모델군별 raw/diag CSV가 있으면 hash와 정의가 일치할 때 재사용한다.

### 2.3 학생 상태 — KD 진단용 신규 선택

K10의 두 Teacher–Student pair를 결과 확인 전에 고정한다.

- Pair A: **T=L1E4 S2025 best_raw**, S=L1E4 S1234의 **20K 이상 최초 저장 checkpoint**.
- Pair B: **T=L1E4 S1234 best_raw**, S=L1E4 S7777의 **20K 이상 최초 저장 checkpoint**.
- 각 Student는 해당 checkpoint의 A+U를 그대로 복사한다. T/S는 같은 architecture, 다른 객체다.
- Mid checkpoint가 없으면 **해당 Student exact50K**로 대체하되 pair를 `late_student_fallback`으로 명시한다. 이는 같은 학습 단계가 아니므로 표를 분리한다.
- 파일이 없으면 가상 checkpoint를 만들지 않는다. Source와 실제 update를 재고정한 manifest를 남긴 뒤 K 단계만 진행한다.
- 서로 다른 Teacher를 섞어 한 sample의 q와 e를 만들지 않는다. Teacher가 Student보다 좋은 sample만 남기지 않는다.

이 선택은 **새 KD 진단 조건**이다. 최종 배포 Teacher·Student 정책을 확정하는 결정은 아니다.

---

<a id="data-split"></a>
## 3. 데이터 분할·sample 단위: 여기서 틀리면 H1–H4가 모두 흐려진다

### 3.1 분석 데이터

| 데이터 | 기본 범위 | 용도 |
|---|---:|---|
| WV3 train-format native64 | 아래 A/B/C/D 합계 목표 4,096개 고유 patch | 네 집단, gradient, KD 진단 |
| RR test256 | 기존 20장 전부 | GT L1·band/edge·개입·stress |
| FR paper512 | 기존 `.mat` 20장 전부 | raw-original HQNR/fSCC·입력 정합 proxy·stress |

64/256/512의 결과는 **서로 다른 strata**다. 같은 번호라는 이유로 RR과 FR scene을 1:1 대응시키지 않는다. Source metadata가 실제 대응을 보장할 때만 별도 paired 분석을 한다.

### 3.2 4,096 patch의 역할 분리 — 신규 진단 분할

| 분할 | 목표 n | 역할 |
|---|---:|---|
| **A: calibration** | 512 | e/q threshold, 입력 통계 scaling, estimator/ROI 사전 확인 |
| **B: adaptation fit** | 2,048 | K10/K20 짧은 업데이트의 학습 입력 |
| **C: policy validation** | 512 | K10 업데이트의 유용성 측정·gate 후보 calibration |
| **D: locked diagnostic holdout** | 1,024 | H1/H2/H3 재확인, K20 미사용 평가 |

가능한 경우 **원본 scene/strip 단위**로 분리하고 인접·중첩 patch를 다른 분할로 보내지 않는다. 위 n은 목표이며 원본 그룹 경계를 맞추기 위해 조금 달라져도 된다. 분할 seed는 신규 `314159`로 고정하고 manifest를 저장한다.

**중요한 한계:** 기존 Teacher/Student는 원 train 전체를 사전학습했을 수 있다. 따라서 A/B/C/D는 **이번 추가 적응에서의 역할 분리**이지, pretrained 모델이 처음 보는 완전 미관측 test가 아니다. `seen_in_pretraining`을 기록한다. 이 설계가 해당 한계를 제거했다고 쓰지 않는다.

원본 scene/strip ID가 없으면 근접성·중복 검사를 하고 `source_group_unknown`으로 남긴다. Patch 수를 독립 scene 수로 보고하지 않는다. H4의 일반화 주장은 제한하고 이번 추가 업데이트 안의 탐색 근거로만 사용한다.

### 3.3 전처리·증강

- D10–D50: **canonical native view**, flip/rotation 추가 없음. 각 sample을 같은 텐서로 모든 model에 넣는다.
- K10/K20: 이번에는 해석 가능한 짧은 적응 진단을 위해 **동일 canonical B pool, 추가 flip/rotation 없음**으로 통제한다. 모든 arm이 동일하다. 이는 원 50K recipe 전체를 재현하는 학습이 아니다.
- Teacher cache key는 `(model_hash, sample_id, scale, preprocessing_hash, augmentation_code)`다.
- 향후 증강을 켜는 full KD에는 실제 증강 입력에서 Teacher를 다시 계산해야 한다. GN/stride가 있는 CNN의 출력을 단순 회전해 같은 예측이라고 가정하지 않는다.
- Intensity scaling·band order·MTF phase는 기존 성공 run과 일치해야 한다.

### 3.4 네 집단을 만들 수 있는지 먼저 확인

A에서 각 checkpoint·scale별 `median(e_T)`, `median(q_A)`를 정하고 B/D에 그대로 적용한다. RR/FR은 scale이 달라 별도 descriptive median 표도 제공하되, **test median은 학습 gate에 쓰지 않는다.**

동점은 사전 규칙 `<= median → low`로 처리한다. 값이 모두 같아 한 쪽이 비면 rank를 억지로 나누지 않는다. 실험적 최소 해석 지원은 **집단당 고유 patch 32개와 확인 가능한 source group 5개**로 제안한다. 이는 통계적 power 보장이 아니라 운영상의 최소선이다.

부족하면 해당 집단은 `insufficient_support`, 연속값 분석은 계속한다. 동일 sample 반복·augmentation으로 독립 표본 수를 부풀리지 않는다. 진단 상세 대상은 각 집단 **최대 64개**를 source/texture 층화 추출한다. 희소 집단은 모두 쓰고 결측을 남긴다.

---

<a id="metric-definition"></a>
## 4. 측정량·부호·probe를 하나로 고정한다

### 4.1 Native e, sample consistency q

\[
e_T(i)=\frac1{CHW}\|\hat Y^0_{T,i}-Y_i\|_1,
\qquad a_T(i)=e_S(i)-e_T(i).
\]

`e_T`는 실제 reconstruction L1이다. ERGAS/PSNR/HQNR으로 대체하지 않는다. `a_T>0`은 native L1에서 Teacher 우위다. 입력 pair와 출력 frame이 같아야 한다.

\[
r_{i,k}=A_T(W(P_i,\epsilon_k),M_i)+\epsilon_k-A_T(P_i,M_i),
\]
\[
q_A(i)=\frac1{2K_A}\sum_{k\in\mathcal P_A}\|r_{i,k}\|_1,
\qquad EPE_A(i)=\frac1{K_A}\sum_{k\in\mathcal P_A}\|r_{i,k}\|_2.
\]

**q에는 λ를 곱하지 않는다.** q와 EPE의 reduction을 구분하고 dy/dx 잔차도 원시값으로 보존한다. Teacher 평가 중에는 SG 유무가 숫자를 바꾸지 않지만, D50 gradient에서는 training의 SG 정의를 따른다.

### 4.2 Probe A와 B를 분리한다

| Probe | 반경 / 방향 | 역할 |
|---|---|---|
| **A: 분류·주 q** | r={.25,.5,1,2} HR px, θ={0,90,180,270}° | 16개, 집단 정의 |
| **B: 확인·stress** | 같은 반경, θ={45,135,225,315}° | 16개, 다른 방향에서 q와 H2 확인 |
| Identity | ε=(0,0) | Unit test; q 평균에 넣어 희석하지 않음 |
| Kernel 확인 | r={.5,1}, A/B 일부, bilinear | 선택한 bicubic 특이 반응인지 검사 |

`epsilon=(r sinθ, r cosθ)`이다. 두 은행 모두 양·음 방향을 포함한다. **네 반경을 균등 가중한 진단 분포**이며 training의 면적-uniform 원판과 같지 않다. 총 평균뿐 아니라 반경별 결과를 반드시 저장한다.

반응 행렬은 `cε−c0 = B_resp ε + b_fit + residual`로 fitting하고, 대각/비대각·부호·적합잔차를 모두 기록한다. 이상적인 PAN-only 반응은 `−I`; `|B_yy|` 하나로 “정합 정확도 %”를 만들지 않는다.

추가 반응이 전혀 없는 기준선 `q_const(A)=mean_A ||ε||_1/2`를 같은 probe로 계산한다. 단순히 작은 r에서 q가 작다는 이유로 좋다고 하지 않는다.

**축/대각선의 L1 기하 차이도 통제한다.** 같은 길이 r에서도 축 방향의 무반응 q는 r/2, 대각선은 r/√2다. 따라서 Probe A와 B의 raw component-L1 평균이 같아야 한다고 요구하지 않는다. 반경별 EPE, signed response, bank별 무반응 대비 비율을 보조로 본다. 네 집단의 주 label은 q_A로 유지하되, q_B로 label 재현성을 볼 때는 **A calibration에서 별도로 측정한 q_B median**을 사용한다. D나 FR에서 threshold를 다시 맞추지 않는다. `q/q_const(bank)`는 진단용 추가 열이며 기존 Loff의 reduction을 바꾸는 것이 아니다.

### 4.3 관측 단위와 ROI

| 작업 | Native 공식값 | 대응 개입/저하량의 공통 ROI 기본안 |
|---|---|---|
| patch64 | full64 L1 | margin16 → 32², L1/edge만 |
| RR256 | full256 기존 RR metric | margin32 → 192² |
| FR512 | raw_original full512 | **V96 → 320²**, native도 V96로 다시 측정 |

ROI는 calibration의 두 warp support·filter support 검사 후 고정한다. 기존 검증된 ROI가 다르면 정의를 계승하고 변경 이유를 기록한다. **Native V64와 stress V96의 차이를 성능 저하라고 빼지 않는다.**

큰 correction으로 support를 벗어난 sample을 유리하게 제외하거나 c를 clamp하지 않는다. `invalid_support`와 비율을 보고하고, 동일한 eligible 집합에서만 paired 진단을 한다. 공식 FR20 전체 점수는 별도로 보존한다.

공통 ROI는 raster 경계 유실을 통제하는 장치이지, 전체 U-Net receptive field의 경계 영향까지 완벽히 제거하는 보증은 아니다. Parent context 확장 검사 또는 ROI 민감도 검사를 일부 sample에 추가한다.

### 4.4 FR의 한계

FR에는 HRMS GT가 없으므로 **e_T가 없다.** FR에서 네 reconstruction/consistency 집단을 직접 만들었다고 보고하지 않는다. `q vs HQNR`, `q vs Dλ/Ds/fSCC`를 별도로 분석한다. `1−HQNR`은 별도 metric 축이며 reconstruction L1이나 Teacher error의 대체 정답이 아니다.

---

## 5. 작업 목록과 약 20시간 배정

아래는 **새 설계의 예약치**다. GPU 진단과 CPU estimator/통계 시간이 섞여 있으므로 실제 GPU 사용 시간과 wall time을 ledger에 별도 기록한다. T0는 s1 runner를 시작할 때 기록한다. 이미 남아 있는 다른 캠페인 예산을 더하지 않는다.

| 단계 | 실행 내용 | 예약 시간 |
|---|---|---:|
| **G00** | 자산·provenance·metric·support·graph gate | 1.0h |
| **D10** | 같은-checkpoint sample atlas, 네 집단, H1 | 2.5h |
| **D20** | 알려진 기하/상대 반응·modality 의존성 | 2.0h |
| **D30** | learned/zero/셔플/외부 보정·native 위치·loss surface | 3.0h |
| **D40** | 출력 stress, PAN 민감도, band·spectral·blur | 3.0h |
| **D50** | loss gradient·국소 개입 | 1.5h |
| **K10** | 네 집단의 hard/soft 대응 단기 실험, H4 진단 label | 3.5h |
| **K20** | 조건부 cue gate 검증 pilot, 주 T/S pair | 2.5h |
| **Report** | 통계·그림·판정표·이력 | 1.0h |
| **합계** | | **20.0h** |

K20 추가 T/S pair 반복은 약 2–3h의 **선택 확장**이다. 필요하면 20h를 넘어 완전한 대응 묶음을 끝내고 이유를 기록한다. 반대로 유효 sample/asset이 없으면 20h를 채우지 않는다.

**우선순위:** L1E4 네 집단과 learned/zero 검증 > H2와 spectral/blur 분리 > K10 > K20 > 다른 λ 추가 상세. 시간이 밀리면 주변 λ의 전수 stress를 줄이고 선택한 L1E4의 필수 검증을 남긴다. 새 최고점이 보인다는 이유로 중간에 λ/구조를 추가하지 않는다.

---

## 6. G00 — 시작 gate

1. 성공 run의 resolved config, checkpoint update/hash, A/U state_dict, data manifest, evaluator hash를 저장한다.
2. Primary Teacher의 원래 `best_raw` 결과를 동일 export/evaluator로 재현한다. RR L1은 별도로 새 계산하고 기존 ERGAS를 L1로 부르지 않는다.
3. All model `eval`, gradient probe 외에는 `no_grad`; buffer·parameter hash가 평가 전후 같아야 한다.
4. `W` 부호·(dy,dx)/(x,y)·FP32·zero identity·bicubic support를 impulse/coordinate-ramp로 검사한다.
5. 실제 module view가 PAN/MS 모두 margin4 후 z-score인지 확인한다. Reconstruction에는 z-score 텐서를 보내지 않는다.
6. D10–D40의 probe RNG, sampler 순서가 training RNG나 원 checkpoint를 바꾸지 않도록 분리한다.
7. K 단계는 실제 wrapper에서 `Rec→A/U`, `Off→A only`, `추가 hard/soft→U only`를 각각 검사한다.
8. Old `pals24` gate가 남았는지 **운영 상태를 확인**한다. 이 문서는 무조건 삭제할 권한/명령을 주지 않는다. 현재 실행 중 작업과 충돌하지 않도록 별도 캠페인 이름을 쓴다. [S1 §7]

Gate 실패는 `implementation_invalid`, 데이터 독립성 미확인은 `provenance_limited`로 나눈다. 후자는 descriptive 분석을 막지는 않지만 일반화/인과 주장의 범위를 제한한다.

<a id="d10-atlas"></a>
## 7. D10 — 같은-checkpoint 네 집단 atlas와 H1

### 실행

**Core 전수:** L1E4와 L000의 3 seed × best/last에서 A/B/C/D native64의 e_T, q_A, q_B, EPE, c0, 입력 통계를 계산한다. 같은 모델에서 RR20/FR20도 별도로 전수 평가한다. L3E4는 우선 3 seed best, N2는 last의 native/response만 계산하고 상세 범위는 시간에 맞춘다.

입력 통계는 PAN/MS intensity 범위·contrast, Scharr energy, 방향 다양성, band별 variance, 채널 간 구조 상관이다. 이름만으로 spectral 원인을 확정하지 않는다.

### 분석 지시

- **H1 primary:** 각 checkpoint·scale 안에서 Spearman(q_A,e_T), Spearman(q_A,HQNR)를 각각 산출한다. Pooled 3-seed 상관만 제시하지 않는다.
- q_A 분위수별 e_T와 q_B, native L1의 mean/median/IQR를 함께 보고한다. q_A/q_B 순위가 불안정하면 단일 noisy q로 gate를 만들지 않는다.
- 네 집단별 n, source 수, band/edge L1, 입력 texture, c0, 이후 D30/D40의 개입 효과를 붙인다.
- q/e 둘 다 연속값으로 분석한다. Median 4분면은 해석 도구이지 유일한 증거가 아니다.
- 보조로 하위/상위 30%의 극단 집단을 보여 주되 가운데 40%를 주 표에서 삭제하지 않는다.
- 같은 입력 난이도 구간에서도 q–e 관계가 남는지 본다. Correction 크기 조건부 분석은 매개변수를 통제할 수 있으므로 **보조 분석**이라고 표시한다.
- `E↓ 집단의 e가 낮다`는 당연한 정의다. 이것을 결과로 재발견했다고 쓰지 않는다. **의미 있는 결과는 같은 집단의 개입 이득·stress·Student 개선이 다른가**이다.

### 필수 그림·표

1. 모델/ckpt별 `log(q_A+eps)`–`log(e_T+eps)` scatter, 사전 threshold 표시.
2. q_A–q_B 반복성 scatter 및 4분면 label의 일치율.
3. 네 집단 occupancy 표와 source-group 분포.
4. FR은 별도 q–HQNR scatter, raw Dλ/Ds/fSCC 분해.
5. 집단별 대표 4개씩: **입력 기반 source/texture 층화 무작위 선택**. 손실이 특히 유리한 예만 고르지 않는다. 실패/악화 예도 포함한다.

**H1 결론:** within-checkpoint 연관성이 확인되면 해당 조건의 관계까지만 말한다. 관계가 없거나 비선형이면 그 결과를 유지한다. **모델 간 평균이 반대라고 sample-level H1을 기각하지 않는다.**

---

<a id="d20-geometry"></a>
## 8. D20 — ‘정합이 진짜 잘 되는가’를 세 층으로 검사

### D20-A: 실제 native pair에 알려진 상대 이동 — 필수

A/B probe에서 다음을 비교한다.

\[
\begin{aligned}
\text{PAN-only:}\;&A(W(P,\epsilon),M)-A(P,M)\simeq-\epsilon,\\
\text{MS-only:}\;&A(P,W(M,\epsilon))-A(P,M)\simeq+\epsilon,\\
\text{Common:}\;&A(W(P,\epsilon),W(M,\epsilon))-A(P,M)\simeq0.
\end{aligned}
\]

- PAN-only는 L1E4/L000 3 seed best/last와 N2 last에서 core 전수.
- MS-only/common은 L1E4 세 seed의 네 집단 각 최대64 patch + RR/FR20, r={.5,1}, 8방향 우선.
- MS-only/common은 **aligner의 진단 입력만** 바꾼다. GT/base 이동이나 새로운 mainline 학습이 아니다.
- 공통 이동은 finite crop/stride 때문에 완전 불변이 자동 보장되지 않는다. 내부 view와 parent-context 영향도 함께 확인한다.
- 반응 행렬 전체, 반경별 q/EPE, off-diagonal coupling, signed bias, 실패율을 기록한다.

**해석:** PAN-only 반응만 좋고 MS-only/common은 체계적으로 틀리면, 상대 위치 추정 외의 shortcut이나 domain 의존성을 의심한다. 이것만으로 원인을 확정하지 않는다.

### D20-B: 알려진 ‘전체’ 기하를 가진 합성 대조 — 필수 calibration

Native 센서 pair에는 절대 정합 GT가 없다. 따라서 다음 **동일 격자 합성 대조**를 별도로 둔다.

1. RR GT 또는 그 same-grid symmetric blur 결과를 8ch anchor Z로 삼는다. 두 modality는 **같은 raster origin**을 사용하며 이 대조에서는 추가 down/up-sampling phase를 만들지 않는다.
2. `M_syn=Z`, `P_syn=sum_c w_c Z_c`로 둔다. w는 음수가 아니고 합 1.
3. w는 세 종류: 전 band 균등, 짝수 index 균등, 홀수 index 균등. Index는 source band metadata에 연결하고, 자동으로 가시광/NIR라고 부르지 않는다.
4. 같은 `P_syn`에 알려진 δ를 적용한다. δ∈{0, ±.5, ±1, ±2}의 축 방향 및 대각 subset.
5. CNN 출력의 **absolute synthetic error** `||A(W(P_syn,δ),M_syn)+δ||₂`와, baseline prediction을 뺀 relative q를 함께 측정한다.

이 쌍의 기하 target은 생성 규약상 알려져 있다. 다만 실제 LRMS/PAN의 sensor 응답·PSF와 동일한 데이터는 아니며, **native 성능의 재현이 아니라 calibration/OOD 진단**이다. 이 대조에서 틀렸다는 이유만으로 L1E4의 복원 효과를 무효화하지 않는다.

특히 `relative q는 낮은데 absolute error는 높음`이면 **공통 보정 bias가 q에서 보이지 않는 상황**을 직접 기록할 수 있다. w만 바꿔 apparent shift가 달라지면 spectral appearance 의존성의 통제된 근거가 된다. 그것을 실제 센서 오정합의 물리적 원인으로 바로 일반화하지 않는다.

### D20-C: cross-modal 단서 검사 — 필수 작은 subset

네 집단 각 최대64 patch에서:

| 입력 대조 | 측정 | 주의 |
|---|---|---|
| 정상 PAN/MS | baseline q, c0, signed response | 기준 |
| 다른 scene의 MS | q, response 변화 | 관계 파괴 대조; task 자체도 OOD |
| band별 상수 MS | q, response 변화 | texture 정보 제거; 정규화 eps 확인 |
| PAN positive affine intensity 변환 | response 변화 | crop z-score 때문에 강건할 수 있으나 수치/범위 확인 |
| bicubic→bilinear probe 생성 | q, signed response | kernel 변화와 shift 효과를 구분 |
| border→reflection probe 생성 | 내부 response | support가 충분한 같은 내부 영역만 비교 |

**통과 주장 범위:** “정상 cross-modal pair에서 알려진 변위 반응을 보인다”를 확인한다. MS-swap에서 나빠졌다는 한 결과만으로 interpolation shortcut을 전부 배제하지 않는다.

---

<a id="d30-intervention"></a>
## 9. D30 — Native 보정의 실효성·절대 기준·복원 최적점

### D30-A: 같은 A/U checkpoint, 적용 correction만 치환 — 최우선

**L1E4의 3 seed best/last**에서 learned, zero, wrong-sign은 core 전수; 나머지 개입은 RR/FR20과 네 집단 상세 subset에서 수행한다. U-Net과 MS는 고정하고 각 후보 PAN은 **원래 P에서 한 번만 warp**한다.

| ID | 적용 correction | 검증 질문 |
|---|---|---|
| I-L | `c_i` | 현재 방식 |
| I-Z | `0` | 현재 U-Net이 learned correction에 의존하는가? |
| I-W | `−c_i` | 방향이 유효한가? |
| I-S | 다른 장면의 `c_j` | scene별 예측의 추가 기여가 있는가? |
| I-C | A calibration의 2D vector median | 공통 보정으로 근사 가능한가? |
| I-G | 독립 estimator의 `c_geo` | 구조 기반 위치를 더 맞추면 복원도 좋은가? |

I-S는 10개의 고정 derangement를 쓰고 `||c_j−c_i||` 분포를 같이 기록한다. 집단별 두 종류를 구분한다: 전체 scene 셔플과 **e/texture가 비슷한 scene 안의 셔플**. Correction 분산이 작으면 셔플은 약한 개입이므로 무효 결과를 “동적 정합 불필요”로 단정하지 않는다.

I-C는 augmentation 전 canonical vector로 계산한다. Train64 vector median을 FR512에 썼다는 scale mismatch를 기록한다. FR 예측에서 사후 얻은 상수는 별도 `test_informed_constant` 진단이며 학습법/공식 결과가 아니다. **norm .55를 (dy,dx)=(.55,.55)로 넣지 않는다.**

I-G는 estimator의 sign/unit/신뢰도 gate를 먼저 통과해야 한다. 값을 못 구한 sample을 0으로 바꾸어 성공표에 섞지 않는다. GT를 사용해 추정한 RR I-G는 `GT-informed diagnostic`으로 표시한다.

### H3 endpoint

\[
g_{corr}^{RR}(i)=L_1(\hat Y_i^{zero},Y_i)-L_1(\hat Y_i^{learned},Y_i),
\]
\[
g_{corr}^{FR}(i)=HQNR_i^{learned}-HQNR_i^{zero}.
\]

동일 ROI와 동일 원 PAN 참조를 쓴다. `g>0`이면 learned가 유리하다. q_A–g 관계는 H3b로 보고한다. H1의 e와 H3의 g는 수학적으로 일부 항을 공유하므로 **change-score coupling**을 주의하고 baseline zero error·texture를 조건화한 보조 분석도 제공한다.

I-L>I-Z는 **학습된 이 U-Net에서 correction을 제거한 효과**다. 별도로 학습한 P0보다 정합이 우수하다는 전체 causal claim이나, 두 scalar만 학습한 모델보다 CNN이 낫다는 주장을 대신하지 않는다.

### D30-B: 독립 native 정합 proxy — 필수

- RR: 같은 해상도·phase에서 **GT↔PAN** 구조를 비교한다. LRMS/GT 생성 경로도 함께 확인한다.
- FR: **해상도를 맞춘 PAN↔관측 MS** 구조를 비교한다. FR GT를 만들어 넣지 않는다.
- Primary: 기존 Scharr–MAD–ZNCC + subpixel 구현. Secondary: 기존 census 또는 검증된 phase-correlation 구현. **기존 source hash와 단위를 확인**하며 없는 helper를 같은 이름의 임의 공식으로 대체하지 않는다.
- before/after 잔여 벡터, peak/ambiguity, boundary hit, 두 estimator 차이, band별 차이, low-confidence coverage를 기록한다.
- 전체 sample의 상태와 high-confidence subset의 수치를 따로 보고한다. 신뢰도가 낮은 sample을 숨기지 않는다.
- 실제 band마다 잔여가 다르면 “band별 물리적 misalignment”와 “spectral appearance에 의한 추정 차이”를 구분하지 못한 것으로 남긴다.

**완전 정합의 판정 금지:** 독립 proxy가 좋아지는 것은 native 위치 개선의 근거다. 물리적 ground-truth registration이 없는 한 “모든 misalignment 제거”라고 쓰지 않는다.

### D30-C: 동일 q인데 native 품질이 달라지는지 — bias 개입

L1E4 S2025 primary의 상세 subset에서 가상의 `A_v=A+v` wrapper를 만들고 v∈{(±.5,0),(0,±.5),(±1,0),(0,±1)}를 사용한다.

\[
(A(P_\epsilon,M)+v)+\epsilon-(A(P,M)+v)
=A(P_\epsilon,M)+\epsilon-A(P,M).
\]

따라서 **q는 수치 허용오차 내에서 동일**해야 한다. 반면 native PAN은 `W(P,c0+v)`이므로 e/HQNR/위치 proxy가 달라질 수 있다.

이 검사는 q의 **절대 기준 미식별성**을 확인하는 음성 대조다. q가 native 품질과 전혀 상관없음을 증명하는 실험도, v가 좋은 새 모델이라는 실험도 아니다.

### D30-D: 구조 최적 shift와 복원 최적 shift가 같은가? — 상세 subset

네 집단 각 최대16개 native64에서, 고정 U-Net에 대해 **c∈{−2,−1.5,…,2}² 81개**의 coarse correction grid를 평가한다. c0와 c_geo도 후보에 따로 추가한다.

동일 c마다 다음을 함께 저장한다.

- native reconstruction L1, band별 L1, edge/non-edge L1.
- 독립 구조 objective. 최적 c_geo와 다른 secondary objective도 기록.
- q는 **원래 A의 값**으로 유지한다. 적용 c를 바꾸는 개입과 A를 다시 학습하는 것을 혼동하지 않는다.

`c_rec_grid*=argmin_c L1`는 **GT를 사용한 고정-U-Net task oracle**이고, 센서 정합 GT가 아니다. Test에서 고른 c_rec_grid*를 공식 성능/추론 방법으로 승격하지 않는다.

**사용자의 의문에 대한 직접 분석:** c_geo에서 구조는 좋아지지만 c0 또는 c_rec_grid*보다 reconstruction이 나빠지는 sample이 있는지 확인한다. 그 차이가 특정 band, edge, interpolation energy와 연결되는지 D40에 넘긴다. Grid 경계가 최적이면 `search_boundary`, 전역 최적이라고 부르지 않는다.

---

<a id="d40-stress"></a>
## 10. D40 — E↓C↑와 E↑C↓를 설명할 출력·spectral·선명도 검사

### D40-A: 추가 PAN 이동에 대한 출력 stress — H2

Probe B 중 r={.5,1,2}, 4대각 방향을 주 stress로 사용한다. 다른 방향은 재현 subset에서 추가한다.

\[
\hat Y_i^\epsilon=M_i+F\big([W(W(P_i,\epsilon),c_i^\epsilon),M_i]\big).
\]

각 sample/ε에서 세 경로를 같은 두-warp sampler로 비교한다.

| 경로 | 두 번째 correction | 의미 |
|---|---|---|
| **Response** | `cε=A(Pε,M)` | 모델의 실제 반응 |
| **No-response** | 원래 `c0` | native 보정은 유지하지만 추가 변화에 반응하지 않음 |
| **Known-inverse** | `c0−ε` | 추가한 변위만 정확히 상쇄하는 좌표 기준선 |

Native는 `W(P,c0)` 한 번이다. Known-inverse도 두 번 raster interpolation의 손상이 남으므로 native와 같다고 가정하지 않는다. 가능한 경우 정수-shift + parent-context 대조를 추가하여 fractional interpolation과 분리한다.

\[
d_e(i,\epsilon)=L_1(\hat Y_i^\epsilon,Y_i)-L_1(\hat Y_i^0,Y_i),
\]
\[
d_Q(i,\epsilon)=Q(\hat Y_i^0;P_i,M_i)-Q(\hat Y_i^\epsilon;P_i,M_i),
\]
\[
s_Y(i,\epsilon)=\operatorname{mean}|\hat Y_i^\epsilon-\hat Y_i^0|.
\]

Native와 stress는 **같은 ROI**다. FR 참조는 **원래 P와 원래 MS**로 고정한다. ε나 예측 correction에 맞춰 평가 PAN을 움직이지 않는다. Aligned 점수는 별도 진단 열에만 둔다.

보고는 d의 signed mean, median, 양수 손실 평균, P90, 최악 방향을 포함한다. 추가 shift가 우연히 원래 오차를 줄이면 d가 음수일 수 있으므로 0으로 덮어써 버리지 않는다. **Native가 낮고 안 떨어지는 모델**을 가장 좋은 모델로 선언하지 않는다.

H2는 q_A가 낮은 sample일수록 **독립 방향 Probe B의 native 대비 저하**가 작은지 본다. 반경·native error·texture를 조건화한다. 작은 q가 단순히 작은 ε를 의미하는 것을 피하기 위해 반경별 분석을 먼저 한다.

### D40-B: 높은 q, 낮은 e 집단의 원인 분리

E↓C↑에서 다음을 함께 해석한다.

| 관측 조합 | 지지하는 설명 후보 | 아직 배제 못 하는 것 |
|---|---|---|
| q↑, s_Y↓, Response와 No-response 유사 | U-Net 내부 보상 또는 PAN 위치에 낮은 민감도 | 둘 중 어느 것인지는 추가 PAN 정보 개입 필요 |
| q↑, s_Y↑, native e↓ | native에서는 잘하지만 위치 변화에 취약 | synthetic interpolation 특이 반응 |
| zero/wrong-sign도 거의 무해, PAN detail 제거도 무해 | 해당 sample에서 PAN 의존성 낮음 | OOD 개입의 한계, MS로 쉬운 sample 효과 |
| zero/wrong-sign은 해로운데 q↑ | native correction 기준은 유효하나 추가 shift 반응은 약함 | 공통 bias 또는 제한된 입력 범위 fitting |

PAN 정보 민감도 검사에서는 **c0를 먼저 고정**한 채 PAN을 (i) zero-phase blur, (ii) 다른 scene PAN, (iii) intensity mean으로 바꾸고 U-Net만 검사한다. 그런 뒤 A까지 재추정하는 경로를 별도 열로 둔다. 앞단 변화와 뒷단 의존성의 효과를 섞지 않는다.

이 개입은 OOD일 수 있으므로 효과의 방향·크기를 설명하는 진단이지, “PAN을 사용하지 않음”의 완전한 증명은 아니다.

### D40-C: 낮은 q, 높은 e 집단의 원인 분리

E↑C↓에서:

1. absolute synthetic error와 native before/after proxy를 확인한다. q↓라도 native 위치 기준이 부정확할 수 있다.
2. q↓이고 proxy 개선도 있지만 e↑이면, GT band별 L1·SAM·edge/non-edge 오차를 본다.
3. c_rec_grid*에서도 e가 크면 해당 고정 U-Net의 복원 한계 또는 어려운 sample을 의심한다. **shift만 바꿔 해결 가능한 문제인지**를 구분한다.
4. c_geo와 c_rec_grid*가 band마다 다르면 cross-band 표현·sampling과 관련된 정황으로 기록한다. 동일한 물리적 edge의 이동이라고 확정하지 않는다.

### D40-D: 동일 기하 spectral 대조와 native band 분석

D20-B의 same-grid Z에서 **공간 좌표를 고정하고 spectral weights만 변경**한다. 적용 δ도 동일하게 유지한다. q, absolute synthetic error, descriptor 추정치를 비교한다.

실제 native RR에서는 PAN과 각 GT band/밴드집단의 구조 정합 추정치, band별 reconstruction error를 함께 본다. Band 명칭은 metadata를 따른다. 짝수/홀수 index를 임의로 가시광/NIR 집단이라고 부르지 않는다.

**해석 기준:** spectral 재조합만으로 q/추정 c가 달라지는 것은 appearance 의존성의 통제 근거다. Native의 모든 loss 불일치가 spectral 원인이라는 결론은 아니다. Spectral 대조와 native residual 패턴이 함께 반복될 때만 조건부 해석을 강화한다.

### D40-E: 위치 개선과 smoothing 분리

- RR GT가 정한 동일 band edge mask에서 L1·gradient error, 단순 edge profile의 **50% crossing 위치와 10–90% 폭**, double edge/overshoot를 측정한다.
- Profile을 못 찾거나 다중 crossing이면 실패로 기록하고 좋은 profile만 평균하지 않는다. Edge 전체 mask error는 항상 함께 보고한다.
- Native warp 전후 PAN energy와 출력 edge width를 연결한다. 입력 PAN energy 하나만으로 출력 8band 선명도를 결론짓지 않는다.
- Blur 대조는 A calibration에서 **위치가 0인 대칭 kernel**을 맞춘다. 원 warp가 sharpening/overshoot를 만들거나 주파수 특성이 맞지 않으면 다시 `unmatched_blur_control`로 남긴다. 억지로 Gaussian 하나를 골라 matched라고 쓰지 않는다.
- Zero-shift identity는 구현 검사이며 **fractional-warp smoothing을 배제하는 대조는 아니다.**

fSCC 하락이 실제 위치 이동 때문인지 detail 손실 때문인지, 둘이 공존하는지를 네 집단별로 보고한다.

---

<a id="d50-gradient"></a>
## 11. D50 — loss 크기 불일치와 gradient 충돌을 구분

### D50-A: 같은 sample의 gradient 진단

L1E4 primary와 L000 primary의 네 집단 각 최대32 patch에서 U-Net을 업데이트하지 않고 다음을 계산한다.

\[
g_r=\nabla_\phi L_{rec},\qquad
g_c=\nabla_\phi L_{off}^{SG},
\]
\[
\cos(g_r,g_c),\qquad
R_g=\frac{10^{-4}\|g_c\|_2}{\|g_r\|_2+\varepsilon_{num}}.
\]

Gradient가 사실상 0이면 cos는 NA로 둔다. 매우 작은 norm을 정상적인 방향 신호처럼 해석하지 않는다. 전체 A와 final head를 나누어 기록한다.

**높은 q·낮은 e가 반드시 gradient conflict라는 뜻은 아니다.** 같은 값에서도 방향이 정렬될 수 있다. 반대로 loss가 둘 다 낮아도 국소 gradient 방향이 반대일 수 있다.

### D50-B: 동일 snapshot에서 작은 A-only 국소 개입

각 sample의 원래 A를 복사하고 다음 네 방향을 비교한다. U-Net은 완전히 고정한다.

- J0: 변경 없음.
- JR: reconstruction gradient 방향.
- JC: SG offset gradient 방향.
- JRC: `g_r+1e-4 g_c` 방향.

방향별 step 크기 교란을 줄이기 위해 **diagnostic normalized SGD step**을 사용한다.

\[
\phi'=\phi-\alpha\|\phi\|_2\frac{g}{\|g\|_2+\varepsilon_{num}},
\quad \alpha\in\{10^{-5},3\cdot10^{-5}\}.
\]

이는 실제 AdamW 학습을 대체하지 않는 **국소 방향 검사**다. 원래 source로 매번 되돌리고 one-step 뒤 e_native, q_A, **q_B**, c0, 출력 변화를 재평가한다.

중요: SG surrogate의 gradient와, 두 예측을 모두 다시 계산한 q 값의 미분은 다르다. JC를 적용했다고 q가 반드시 감소한다고 보장하지 않는다. 실제 측정값을 보고한다. Probe A에만 맞고 B에 전이되지 않으면 이를 보존 능력 개선이라고 쓰지 않는다.

**분석 질문:** E↓C↑에서 q를 줄이는 방향이 e를 악화시키는가? E↑C↓에서 recon 방향은 q를 망가뜨리지 않고 e를 줄이는가? 이 관계가 네 집단과 source마다 다른가?

이 단계는 “두 loss가 왜 엇갈리는지”의 국소 기전 근거다. 장기 수렴이나 KD 효과를 직접 증명하는 결과는 아니다.

---
<a id="k10-utility"></a>
## 12. K10 — 네 집단에서 hard와 soft가 실제로 다른 가치를 갖는가?

**H4를 위한 필수 단기 실험**이다. e/q로 좋은 Teacher를 미리 정답화하지 않고, 같은 Student snapshot에서 **추가 hard와 추가 soft 중 어느 쪽이 별도 자료의 복원을 더 개선하는지** 직접 측정한다. 이는 최종 KD 성능 실험이 아니라, 감독 선택의 유용성에 대한 국소 학습 실험이다.

### 12.1 무엇을 비교하는가

같은 native sample에서:

\[
h_i=\operatorname{mean}|\hat Y_{S,i}-Y_i|,\qquad
 d_i=\operatorname{mean}|\hat Y_{S,i}-\operatorname{sg}(\hat Y_{T,i})|.
\]

모든 arm의 기본 목적은 현재 L1E4다.

\[
L_{base}=\bar h+m_t10^{-4}L_{off,S}.
\]

| Arm | 추가 감독 | Student aligner의 gradient | Student U-Net의 gradient |
|---|---|---|---|
| **K10-R0** | 없음 | 기본 Rec + Off | 기본 Rec |
| **K10-RH** | \(\gamma\bar h\) | **기본 Rec + Off만** | 기본 Rec + 추가 hard |
| **K10-RS** | \(\gamma\bar d\) | **기본 Rec + Off만** | 기본 Rec + 추가 soft |

**신규 진단 계수는 γ=0.1로 사전 고정**한다. 이는 검증된 최적 KD 계수가 아니다. 양의 신호를 얻기 위해 집단마다 γ를 다르게 조절하지 않는다. 결과가 없으면 이 계수·학습 길이에서의 미검출로 한정한다.

R0는 “그냥 조금 더 학습한 효과”를 분리한다. RH와 RS는 **같은 크기의 추가 감독 예산에서 GT와 Teacher 중 어느 target이 유리한가**를 비교한다. 계수의 크기가 같다고 실제 parameter gradient norm도 같다는 뜻은 아니므로 norm을 기록한다.

### 12.2 집단·Student 우위·반복

1. Primary Pair A와 반복 Pair B를 §2.3대로 고정한다.
2. 각 pair에서 **추가 학습 전** Teacher e/q로 네 집단을 만든다. 이어 `a_T>0` / `a_T<=0`를 나누어 최대 **8개 cell**로 분류한다.
3. B pool에서 각 cell당 최대 **4개의 fit pool**, pool당 목표 **48개 고유 patch**를 source 균형으로 뽑는다. 가능한 한 서로 겹치지 않게 한다.
4. 한 fit pool을 K10-R0/RH/RS 모두에 동일하게 사용하고 **64 optimizer updates**를 수행한다. 매 step의 batch·data 순서·Student ε RNG를 맞춘다.
5. 각 arm·각 trial 시작 시 **원 Student snapshot으로 완전히 복구**한다. RH 완료 모델을 RS의 초기값으로 넘기지 않는다.
6. C 중 입력/source 기준으로 사전 선택한 **128개 고정 validation patch**에서 before/after를 평가한다. C는 이 update에 사용하지 않는다. 각 arm의 train-fit 이득과 C 이득을 분리한다.

집단이 작으면 지원되는 trial 수만 수행한다. `48개가 없어 32~47개인 pool`은 실제 n을 기록하고 cycling으로 실효 batch48을 맞출 수 있지만, **이를 48개 독립 표본이라고 세지 않는다**. 32개 미만이면 해당 cell의 update를 보류한다. 겹치는 fit pool은 반복으로 기록하되 독립 trial로 취급하지 않는다.

최대 실행량은:

\[
2\ \text{pair}\times8\ \text{cell}\times4\ \text{trial}
\times3\ \text{arm}\times64
=12{,}288\ \text{optimizer updates}.
\]

**네 집단을 25%씩 강제로 맞추거나, 없는 a_T 부호 집단을 만들어 채우지 않는다.** Teacher가 이미 Student보다 거의 항상 좋거나 나쁘면 그 자체가 결과다.

### 12.3 국소 적응 설정

| 항목 | K10 신규 설정 |
|---|---|
| Source | 같은 기존 Student의 A+U snapshot |
| Teacher | 같은 기존 Teacher A+U, frozen·eval·target no-grad |
| Optimizer state | 각 trial/arm에서 **동일한 fresh AdamW state** |
| LR | **U-Net 1e-5 / aligner 1e-6**, 64 step 내 상수 |
| Scheduler | K10에는 추가 warmup/cosine 없음 |
| WD·betas·eps·clipping·precision | source resolved config 계승, 모든 arm 동일 |
| Augmentation | canonical native, 추가 flip/rotation 없음 |
| Student Off | 기존 b=2, SG, odd update, λ_off=1e-4 유지 |
| 추가 hard/soft | **U-Net 가중치만 수신** |
| 평가 시점 | source와 exact64, 사후 best 선택 없음 |

이는 **50K run의 optimizer를 이어가는 continuation이 아니라, 저장 모델의 단기 반응을 비교하는 새 protocol**이다. 비교군 모두 같은 재시작을 하므로 이 차이는 통제된다. 모든 arm에서 A는 기본 Rec/Off로 계속 학습하지만, K10의 **집단 label과 Teacher cue는 source 상태로 고정**한다.

### 12.4 H4의 유용성 label

C의 동일 validation 집합에서 다음을 산출한다.

\[
U_{soft-hard}=L_C(S_{RH}^{64})-L_C(S_{RS}^{64}),
\]
\[
U_{soft-base}=L_C(S_{R0}^{64})-L_C(S_{RS}^{64}),\qquad
U_{hard-base}=L_C(S_{R0}^{64})-L_C(S_{RH}^{64}).
\]

양의 `U_soft-hard`는 **이 fit pool과 이 Student에서 추가 soft가 추가 hard보다 유리**했다는 뜻이다. source C L1으로 나눈 상대 변화도 보고하되, 원래 L1 차이를 함께 보존한다.

**유용성의 정답을 `a_T>0`으로 정의하지 않는다.** 그것은 이미 a_T에 포함된 정보라 q의 추가 설명력을 검사할 수 없다. 실제 학습 후 별도 C 성능 차이가 이 단계의 label이다.

이 label은 **fit-pool / 국소 update의 유용성**이지, 각 픽셀·개별 sample의 참 유용성이나 uncertainty label은 아니다. 하나의 pool 이득을 그 안의 48개 patch에 복제해 48개 독립 label로 회귀하지 않는다.

### 12.5 네 집단에서 요구하는 분석

각 e/q 집단을 a_T 부호별로 나누어 다음을 보고한다.

- R0/RH/RS의 train-fit L1 및 C L1 변화, band/edge 변화.
- U_soft-hard의 평균·중앙값·분포, 긍정 trial 수, source별 변동.
- `q 낮음 → soft 유리`를 강제하지 않은 실제 순서.
- `E↓C↑`에서 soft가 유효한지, `E↑C↓`에서 hard가 유리한지. 반대 결과도 그대로 보존.
- 같은 e_T·a_T 범위에서 q가 다르면 utility가 달라지는지. Cell 평균뿐 아니라 연속값을 통제한 분석도 수행.

Pair A에서 관계를 발견해도 Pair B에서 같은 방향인지 확인한다. 서로 다른 Student learning stage가 fallback으로 섞였다면 stage별로 표를 나눈다. **한 번의 short-horizon 효과를 최종 50K KD 성공이라고 쓰지 않는다.**

---

<a id="k20-pilot"></a>
## 13. K20 — e/a 기준에 q를 더하면 실제 Student가 나아지는가? 조건부 pilot

이 단계는 앞선 분석을 실제 감독 선택으로 연결하는 **제한된 추가 학습 비교**다. D10–D50 또는 K10이 불충분하면 무리하게 gate를 만들지 않고, 그 원인을 정리하는 것으로 종료할 수 있다. **H1의 양의 상관이 반드시 있어야 실행하는 것은 아니다.** Native 품질과 q의 단순 상관이 약해도 H4의 조건부 정보는 남을 수 있다.

### 13.1 진입 조건

- e/q cache·checkpoint·unit·ROI의 의미가 검증됐다.
- q_A/q_B의 sample 순서와 지표 수치가 재실행 오차만으로 결정되지 않는다.
- K10의 주요 cell에 실제 fit pool이 있고, U_soft-hard를 원시 C 결과에서 재현할 수 있다.
- Teacher error·Student advantage만 쓰는 기준을 **같은 조건으로 구현**할 수 있다.
- D locked holdout이나 FR 결과를 보고 gate 방향·계수를 선택하지 않는다.

Cell이 희소하면 아래의 사전 지정 shrinkage를 적용한다. 전체 label이 사실상 수치 재현 오차와 같거나 source 지원이 부족하면 `pilot_not_identifiable`로 기록한다. 원하는 q 기여를 만들기 위해 임계값을 계속 바꾸지 않는다.

### 13.2 가장 단순한 신규 gate: 조건부 utility 표

처음에는 대형 gating CNN·pixelwise uncertainty head를 추가하지 않는다.

- **EA 표:** `(e_T low/high, a_T>0 여부)`의 4개 cell.
- **EAQ 표:** 위 4개 cell을 `q_T low/high`로 나눈 8개 cell.
- e/q의 threshold는 §3의 A calibration에서 고정한다. a 기준은 0이다.
- Cell값은 K10에서 측정한 **U_soft-hard의 상대 변화**를 사용한다. q가 낮다는 이유로 높은 utility를 미리 넣지 않는다.

희소 cell은 다음처럼 상위 표로 수축한다.

\[
\bar u^{EAQ}_g=
\frac{n_g\bar u_g+4\bar u^{EA}_{parent(g)}}{n_g+4},
\]

EA cell도 같은 규칙으로 전체 utility 평균에 수축한다. `n_g`는 고유 trial 수가 아니라 **독립 fit-pool 지지 수를 넘지 않도록** 잡는다. 겹침·동일 source 의존성이 있으면 보수적으로 낮추고 원래 trial 수도 병기한다.

수축 강도 4는 **이번 작은 pilot의 사전 지정값**이지 최적값이나 통계적 보증이 아니다. 지원이 없는 joint cell은 EA 값과 같아진다.

Soft 비율은 다음처럼 0~1의 유계값으로 바꾼다.

\[
\rho_g=\sigma\!\left(\operatorname{clip}(\bar u_g/s_u,-4,4)+b_{mass}\right).
\]

`s_u`는 K10 calibration utility의 robust scale과 재현 noise scale 중 큰 값이다. 모두 0이면 표 전체를 0.5로 두고 `no_resolved_utility_signal`을 기록한다. Noise scale은 같은 source·fit pool·RNG를 재실행한 소수 대조에서 측정하고 임의의 “유의한 효과”로 바꾸지 않는다.

`b_mass`는 A/B calibration에서 **평균 ρ=0.5**가 되도록 한 번 계산해 고정한다. 이 조정은 전체 KD 강도보다 sample 배분의 차이를 보기 위한 것이다. q를 쓰는 표와 쓰지 않는 표가 같은 형태의 두 table일 필요는 없으므로, **q shuffle 대조**로 추가 자유도 효과도 점검한다.

**중요:** 이 첫 pilot의 a_T는 **source Student 상태에서 계산한 값으로 고정**한다. 매 step a_T와 gate를 다시 바꾸는 온라인 적응은 이번에 하지 않는다. 먼저 고정 cue의 유용성을 확인한 뒤 dynamic policy를 별도 검증한다.

### 13.3 다섯 arm, 동일 source·동일 5K

Pair A의 같은 Student source에서 다음을 독립 실행한다.

| Arm | 추가 감독의 soft 비율 ρ | 구분하는 효과 |
|---|---|---|
| **K20-R** | 추가 감독 없음 | 단순 추가 fitting 기준 |
| **K20-U** | 모든 sample 0.5 | Uniform hard/soft의 효과 |
| **K20-EA** | e_T·a_T 조건부 utility 표 | 기존 error/advantage cue의 효과 |
| **K20-EAQ** | e_T·a_T·q_T 조건부 utility 표 | 정합 cue의 추가 정보 |
| **K20-EAQ-SHUF** | EAQ와 같은 표, q를 조건부 셔플 | 실제 sample–q 대응의 필요성 |

`K20-EAQ-SHUF`는 영상·Teacher 출력·MS·correction을 섞지 않는다. **Gate 계산용 q metadata만** e/a가 비슷한 B sample 사이에서 고정 derangement한다. 우선 EA cell 안에서 연속 `(rank e, rank a)`가 가까운 블록을 만들고 블록 내에서 섞는다. 블록 목표 크기는 16, 최소 8이다. 가능한 경우 source/texture 분포도 확인한다. 블록을 만들 수 없는 sample은 `shuffle_unmatched`로 기록한다.

e/a를 대략적으로만 맞췄다면 “e/a를 완전히 제거한 순수 q 효과”라고 말하지 않는다. 원래 q와 shuffled q의 주변 분포, hard/soft 계수 평균, 실제 변경 sample 비율을 보고한다.

### 13.4 Loss와 수신자

\[
L_{extra}=\frac\gamma B\sum_i[(1-\rho_i)h_i+\rho_i d_i],
\qquad\gamma=0.1.
\]

**기본 GT reconstruction은 항상 유지**한다. U-Net이 받는 계수는 hard `1+γ(1−ρ)`와 soft `γρ`이며 합은 항상 `1+γ`다. 그러나 실제 gradient 크기는 같지 않을 수 있어 별도 기록한다.

\[
g_{\phi_S}=\nabla_{\phi_S}(\bar h+m_t10^{-4}L_{off,S}),
\]
\[
g_{\theta_S}=\nabla_{\theta_S}(\bar h+L_{extra}).
\]

Teacher output, Teacher q/e, source Student a, ρ는 모두 detach된 metadata/target이다. **Student의 추가 hard/soft는 U-Net에만 직접 전달**한다. 기본 Rec→aligner는 유지한다.

### 13.5 적응·평가 계약

- 각 arm **5,000 optimizer updates**, fresh 동일 optimizer, U LR1e-5 / A LR1e-6.
- K20은 새 5K warmup100+cosine을 모든 arm에 동일하게 사용한다. Parent run의 끝난 cosine LR=0을 그대로 로드하지 않는다. Minimum LR 등 나머지 값은 기존 정책에서 비율로 계승하고 manifest에 남긴다.
- 학습은 B 전체, 동일 순서·실효 batch48·canonical native input. Student Off는 기존 odd update 규약 유지.
- Source / 2,500 / exact5,000을 저장한다. **H4 주 비교는 exact5,000**이며, 중간에 가장 좋은 결과를 골라 gate를 바꾸지 않는다.
- K10 C로 policy를 정했다면 C 성능은 development 결과다. **D locked holdout의 L1와 band/edge**가 이 pilot의 우선 유용성 endpoint다.
- 이후 RR20·FR20을 같은 endpoint에서 재평가한다. Raw-original HQNR/fSCC와 RR 손실을 모두 보고하되, 이미 반복 관찰한 FR20을 새로운 미사용 test라고 부르지 않는다.
- Optional Pair B 반복에서도 같은 gate 규칙·γ·5K·비교 arm 전체를 유지한다. Pair B를 pilot 보고 전에 보지 않았다면 cross-pair 확인으로, K10에 이미 사용했다면 **held-pair 검증이 아님**을 명시한다.

최대 기본 K20 학습량은 `5 arm × 5K = 25K update`다. 50K single run보다 update 수는 적지만 Teacher/cache·추가 gradient 비용은 profiling해야 한다. 2.5시간은 예약치다.

### 13.6 무엇이 H4의 긍정 근거인가?

**K20-EAQ > K20-EA**이고, **EAQ > EAQ-SHUF**이며, D에서의 개선이 source/Student pair와 RR/FR 진단에서 설명 가능한 방향으로 남는가를 본다. EAQ가 uniform만 넘고 EA를 넘지 못하면 q의 추가 기여라고 하지 않는다.

모든 arm의 평균 soft/hard 비율을 맞췄는데도 EAQ가 유리하면 cue 배분의 기여를 지지한다. 평균만 같고 source별 가중치가 크게 다르면 그 차이도 보고한다.

**이 결과가 좋아도 즉시 “q는 Teacher의 uncertainty”라고 이름 붙이지 않는다.** 허용되는 표현은 “현재 고정 Teacher/Student·데이터 조건에서 native error/advantage에 q를 결합한 감독 선택이 추가 이득을 보였다”이다. 최종 장기 KD·새 데이터 검증은 별도다.

---

<a id="gradient-routing"></a>
## 14. Gradient routing·캐시 구현 지시

### 14.1 단순 `loss.sum().backward()`만으로는 이번 수신자 분리가 되지 않는다

기본 forward의 `P_aligned`를 통째로 detach하면 Rec→A가 끊긴다. 반대로 추가 soft loss를 전체 graph로 backward하면 KD→A가 생긴다. **손실별 수신 parameter를 분리**해야 한다. [S3 §11.4]

아래는 FP32의 **의미 pseudocode**다. 실제 trainer·AMP·DDP·accumulation에 그대로 붙이는 완성 코드가 아니며, 기존 wrapper에 맞는 adapter와 unit test가 필요하다.

```python
# A_params and F_params are disjoint Student parameter lists.
# Teacher tensors and rho are frozen. They must not share live Student buffers.
optimizer.zero_grad(set_to_none=True)

pred, c0 = student_native_forward(P, M)  # Keep Rec -> F -> warp -> A.
h = (pred - Y).abs().flatten(1).mean(1)
base = h.mean()
if update_index % 2 == 1:
    off = student_offset_auxiliary(P, M, c0)  # c0 target detached only here.
    base = base + 1e-4 * off

with torch.no_grad():
    teacher_pred = frozen_teacher_native_or_exact_cache(P, M)
    rho = frozen_policy_metadata(sample_ids)  # Not learned from this backward.

d = (pred - teacher_pred).abs().flatten(1).mean(1)
extra = gamma * ((1.0 - rho) * h + rho * d).mean()

base_grads = autograd.grad(
    base, A_params + F_params, retain_graph=True, allow_unused=False
)
extra_F_grads = autograd.grad(extra, F_params, allow_unused=False)

# Assign base_grads for A; sum base+extra only for F.
# If a component is intentionally unused in a verified wrapper, handle that
# explicit contract; do not silently replace unexpected None gradients with 0.
assign_checked_gradients(A_params, base_grads[:len(A_params)])
assign_checked_gradients(
    F_params,
    [g0 + g1 for g0, g1 in zip(base_grads[len(A_params):], extra_F_grads)]
)
existing_gradient_clip_and_optimizer_step()
```

K10-RH는 extra=`γ*h.mean()`, RS는 extra=`γ*d.mean()`, R0/K20-R은 extra를 생성하지 않는다. 계수 0인 graph를 남긴 것만으로 동치 검증을 대체하지 않는다.

AMP/scaler, DDP synchronization, accumulation, gradient clipping의 적용 위치는 별도로 검증한다. 위의 `.grad` 수동 대입이 기존 분산 구현과 같다고 가정하지 않는다. s1 단일 GPU·기존 FP32 방침을 우선하고 불필요한 새 분산 경로는 추가하지 않는다.

### 14.2 필수 unit tests

- Base Rec만으로 A/F 양쪽에 gradient가 있고, Off만으로는 F에 gradient가 없다.
- Extra hard/soft를 켜도 **같은 source·같은 forward에서 A에 전달되는 직접 gradient는 Base만 쓴 경우와 같아야 한다.**
- Extra=0에서 원래 L1E4의 forward와 base gradient가 재현된다.
- `.eval()`만으로 Teacher가 동결됐다고 처리하지 않고, 모든 Teacher parameter/buffer가 update 전후 동일한지 검사한다.
- Source snapshot/RNG/optimizer를 매 trial 복원하고, 재실행 순서가 utility를 바꾸지 않는지 검사한다.
- Head/margin/z-score를 중복 적용하거나 M residual을 두 번 더하지 않는다.
- q/e cache는 실제 checkpoint·native preprocessing·sample ID·scale과 일치해야 한다. Cache miss를 비슷한 ID로 대체하지 않는다.

### 14.3 q와 reconstruction의 계산 비용

D10의 Teacher q는 작은 A의 16/32회 호출, native Teacher 출력은 F의 1회 호출로 계산한다. **모든 probe에 F까지 호출하는 것은 D40 stress 대상으로 제한**한다. K 단계는 canonical 입력이므로 native Teacher 출력과 q를 정확한 key로 cache할 수 있다.

저장량은 dtype·shape로 산정하고 모든 영상·모든 probe의 출력을 무제한 저장하지 않는다. 원칙적으로 모든 sample의 scalar/vector, 대표 panel의 영상, 재현 가능한 probe manifest를 남긴다. Teacher cache를 FP16으로 바꿨다면 수치 차이를 측정하고, 특정 대조군만 FP32에 남기지 않는다.

---

<a id="analysis-rules"></a>
## 15. 분석 지시 — 상관, 개입, KD 유용성을 분리해 결론 낸다

### 15.1 주 분석과 보조 분석

| 질문 | 주 분석 | 반드시 함께 볼 대조 |
|---|---|---|
| **H1 RR** | 같은 checkpoint의 q_A–native L1 연속 관계 | q_B 반복성, texture/edge 층화, source별 관계 |
| **H1 FR** | 같은 checkpoint의 q_A–raw-original HQNR | Dλ/Ds/fSCC, FR20 scene별, RR과 분리 |
| **H2** | q_A–Probe B의 signed native 대비 저하 | 반경·native error 조건화, Known-inverse·No-response |
| **H3a** | 같은 가중치의 learned−zero paired 차이 | wrong-sign·shuffle·constant·같은 ROI |
| **H3b** | q_A–sample별 correction 이득 | native e, texture, 보정 크기와의 관계; algebraic coupling 주의 |
| **H4 진단** | K10 U_soft-hard에 대한 EA vs EAQ 설명 | 별도 fit pool, 다른 Student pair, q 조건부 셔플 |
| **H4 실행** | K20 EAQ−EA, EAQ−SHUF의 D 이득 | 추가 학습량·평균 가중치·source state·endpoint 일치 |

**H3b의 주의:** `learned e`가 개입 이득의 한 항에 들어가므로, e와 gain의 상관 일부는 정의에서 생길 수 있다. 같은 sample의 change score 하나만으로 인과 설명을 선언하지 않고, zero-error·learned-error를 각각 보여 주고 여러 개입을 함께 비교한다.

### 15.2 통계 규칙

- 단위는 원본 scene/strip이다. Seed나 probe를 독립 scene 수로 세지 않는다. 같은 scene의 patch와 여러 ε는 **반복 측정**이다.
- 기술통계는 mean/median/IQR/P90, paired 차이, 긍정 sample·scene 수를 함께 보고한다. 3-seed 평균 하나만으로 결론 내리지 않는다.
- 불확실성은 가능한 경우 **source-group block bootstrap 2,000회**로 95% interval을 구한다. FR20은 scene block을 유지한다. 이 수치들은 신규 분석 규격이며, source 의존성이 모호하면 보수적으로 제한한다.
- Source group 수가 너무 작거나 확인할 수 없으면 CI의 독립성 가정이 성립하지 않는다고 쓰고, 원시 paired plot 중심의 descriptive 결론으로 남긴다.
- Continuous 관계는 우선 Spearman을 사용하되, 비선형 구간·U형 관계를 분위수 표로 확인한다. Median 이분화의 결론과 연속값 결론이 다르면 둘 다 적는다.
- H1, H2, H3b는 모델별로 적합한다. 서로 다른 Teacher의 q scale, checkpoint age, FR/RR을 무작정 합쳐 하나의 coefficient로 만들지 않는다.
- Threshold·probe·ROI·primary outcome은 A/G00에서 동결한다. D/FR 결과를 보고 radius나 quadrant 경계를 바꾸면 **새 exploratory 분석**으로 이름을 분리한다.

**0.0031은 현재 연구의 HQNR 방법 간 경험적 판정선이다.** L1, EPE, Spearman, 단기 utility의 유의성 기준이나 equivalence margin으로 재사용하지 않는다. 작은 차이가 통계적으로 구분되지 않았다는 결과와 “실질적으로 동일하다”는 주장은 다르다.

### 15.3 H4의 ‘추가 설명력’을 직접 검사한다

K10 trial/pool 단위에서 다음 두 모델을 비교하는 보조 분석을 한다.

- **B0:** pool의 e_T 평균/산포, a_T 평균/우위 비율, source Student 상태.
- **B1:** B0 + q_A 평균/산포 + `e_T×q_A`, `a_T×q_A` 항.

첫 구현은 작은 regularized linear model 또는 위 EA/EAQ table로 제한한다. 선택한 형식과 규제값은 **C calibration 내부**에서 결정한다. 동일 source patch가 validation fold 양쪽에 들어가지 않도록 **fit-pool/source 묶음 단위**로 나눈다. 겹침으로 group split이 불가능하면 out-of-fold 성능이라고 보고하지 않는다.

가능하면 Pair A에서 만든 관계를 Pair B에 적용한 성능을 별도로 제공한다. 그때 model/threshold/normalization을 Pair B의 utility 결과로 다시 맞추지 않는다. 두 pair를 모두 fit했다면 그 분석은 cross-pair validation이 아니다.

보고할 값은 held-pool utility 예측 MAE, 방향 판정 정확도, q를 추가했을 때의 개선, 조건부 q 셔플 대비 차이다. Utility sign이 재실행 noise 부근이면 **tie/undetermined**로 표시하고 강제로 hard/soft 정답을 만들지 않는다.

**중요한 한계:** K10은 작은 pool에 반복 적응한 단기 utility이고, EA/Q median cell은 e/a를 완벽히 통제하지 않는다. EAQ의 이득이 보이면 연속 e/a를 넣은 B0/B1과 matched shuffle에서도 남는지 확인한다. 이 추가 확인 없이 “e_T·a_T로 설명되지 않는 인과적 정보”라고 확대하지 않는다.

### 15.4 네 집단별 최종 해석표

| 집단 | 이번에 설명해야 할 내용 | 해석을 지지하는 조합의 예 | 해석을 바꾸게 하는 결과 |
|---|---|---|---|
| **E↓C↓** | 두 과제의 성공이 실질적 위치·KD 이득으로 이어지는가 | proxy 개선, stress 작음, Teacher 우위일 때 soft 이득 | q↓지만 absolute synthetic bias 큼, soft 무익 |
| **E↓C↑** | 복원은 좋은데 q가 큰 이유 | 출력 stress 작음+PAN 민감도 낮음 / native correction 유효+추가shift만 취약 | q 반복성이 없어 group 자체가 불안정 |
| **E↑C↓** | 기하 반응 외에 무엇이 어려운가 | q↓+proxy 개선에도 특정 band/edge error 큼, hard 우세 | correction grid만 바꿔 error 크게 감소 → native 기준 문제 |
| **E↑C↑** | 공통 난이도인지 두 독립 실패인지 | texture·band 문제가 동반, recon/offset 개입 각각 다른 항 개선 | GT label/preprocessing 오류 또는 낮은 측정 신뢰도 |

이 표는 결과를 억지로 맞추는 분류기가 아니다. **다중 원인이 공존하거나 설명되지 않는 sample은 `mixed/unresolved`**로 남긴다. 각 집단에서 대표적인 성공·실패·미설명 사례와 빈도를 함께 보고한다.

### 15.5 최종 가설 판정의 네 상태

1. **Supported in tested setting:** 사전 정의한 방향과 대응 대조가 일관되며, 측정·source 한계 안에서 지지됨.
2. **Opposed in tested setting:** 반대 방향의 재현 가능한 관측이 있음. 현재 조건에 한정함.
3. **Insufficient / unresolved:** 표본·측정 반복성·effect size가 부족하거나 결과가 혼합됨.
4. **Implementation invalid:** 부호·ROI·graph·reference·cache 불일치 등으로 결과 해석 불가.

“No significant difference”를 2번으로 자동 배치하지 않는다. 같은 평균이라도 적절한 equivalence 검정 없이 “q는 완전히 무용하다”고 선언하지 않는다.

---

<a id="outputs"></a>
## 16. 종료 시 반드시 전달할 표·그림·원시 파일

새 출력 root는 **제안 경로** `work_dir/_eqrec4_s1_campaign/`다. 기존 `_pals24_campaign`, `_palsv18_campaign`이나 run checkpoint를 덮어쓰지 않는다.

```text
_eqrec4_s1_campaign/
  protocol_resolved.yaml
  protocol_changes.md
  time_ledger.jsonl
  sources_manifest.json
  assets_manifest.csv
  data_manifest.csv
  probe_bank_A.json
  probe_bank_B.json
  roi_contract.json
  calibration_thresholds.json
  native_sample_metrics.parquet
  offset_probe_records.parquet
  quadrant_assignments.parquet
  response_matrices.csv
  geometry_controls.csv
  correction_interventions.parquet
  correction_landscape.parquet
  output_stress.parquet
  band_edge_profiles.parquet
  blur_control_status.json
  gradient_interventions.parquet
  k10_trial_manifest.csv
  k10_utility.csv
  k20_policy_tables.json
  k20_weight_mass.csv
  k20_endpoint_metrics.csv
  figures/
  report_EQREC4.md
```

Parquet을 지원하지 않으면 동일 schema의 CSV/JSONL로 저장한다. 형식 때문에 구현을 새 패키지에 의존시키지 않는다.

### 16.1 Core scalar/vector schema

필수 식별자는 다음과 같다.

```text
campaign_id, run_id, model_hash, aligner_hash, unet_hash,
checkpoint_kind, actual_update, model_role, teacher_student_pair,
source_seed, split_role, sample_id, source_group_id, scale_hw,
seen_in_pretraining, preprocessing_hash, probe_bank, probe_id,
epsilon_dy_hr, epsilon_dx_hr, kernel, padding, roi_id,
valid_support, invalid_reason,
c0_dy, c0_dx, ce_dy, ce_dx, residual_dy, residual_dx,
q_l1_component_mean, epe_l2, e_native_full, e_native_common,
e_student_source, advantage_teacher_source,
quadrant_id, threshold_hash
```

Probe별 record의 q와 sample집계 q를 같은 열 이름으로 덮어쓰지 않는다. `scalar_scope=probe/sample/scene/model`을 명시한다. Native full e, common ROI e, train-fit e, validation e를 구분한다.

K10에는 `trial_id`, 실제 unique fit sample/source 수, arm, update, restore hash, optimizer-reset 상태, C before/after, U_soft-hard와 모든 status를 기록한다. K20에는 ρ 원값/정규화값, 해당 gate cell, q shuffle partner, 평균 hard/soft 질량, extra/base gradient ratio를 남긴다.

### 16.2 보고서 필수 표

**Table A — 자산·재현:** 원 Sheet/run 선택시점, 실제 checkpoint, 현재 재평가, evaluator/manifest 차이. 원 보고서 수치를 조용히 교체하지 않는다.

**Table B — 네 집단:** n/source 수, e/q/q_B, c0, band/edge error, stress, learned−zero gain, K10 utility를 **같은 checkpoint** 기준으로 연결한다.

**Table C — 정합의 세 수준:** relative response, known absolute synthetic error, native before/after proxy. 세 수준을 하나의 ‘정합 정확도’로 합치지 않는다.

**Table D — 개입:** learned/zero/wrong/shuffle/constant/c_geo, native/stress, RR/FR을 분리하고 common ROI를 명시한다.

**Table E — cue 유용성:** K10의 EA/EAQ 조건부 관계 및 K20의 R/U/EA/EAQ/SHUF endpoint. Raw HQNR, RR L1, band/edge, Teacher–Student 단계 정보를 병기한다.

**Table F — H1/H2/H3a/H3b/H4/Hspec 판정:** 현재 답, 근거 파일·행, 반대 증거, 범위 제한, 다음에 필요한 자료를 한 행씩 적는다.

### 16.3 보고서 필수 그림

1. **e_T–q_T 4분면 scatter**, 입력 texture/source 표시와 q_A/q_B 반복성.
2. **PAN-only/MS-only/common의 signed response**, 두 축과 비대각 결합.
3. 네 집단별 **native→stress L1 변화**, Known-inverse 기준선 포함.
4. Learned/zero/wrong/constant의 **paired gain plot**, raw와 aligned 별도.
5. 네 집단 대표 patch의 **동일 band GT edge 위치·폭과 spectral error panel**.
6. **q–e–correction objective landscape**의 사례: 공통bias, geometry optimum, reconstruction optimum의 차이.
7. **K10 hard/soft utility와 a_T 조건부 비교**, K20 조건부 q 추가 이득.

시각 예시는 전체 표의 극단 우승 사례만 고르지 않는다. Source/texture 층화로 사전 선택한 예시와, 자동 규칙으로 추가한 실패 예시를 구분해 표시한다. FR의 보기 좋은 RGB overlay를 absolute GT처럼 설명하지 않는다.

---

<a id="execution"></a>
## 17. 실행 큐·운영·예산 조절

### 17.1 의존 관계

```text
G00
 ├─ D10: e/q atlas + thresholds + source support
 │    ├─ D20: relative / absolute synthetic / modality checks
 │    ├─ D30: same-weights correction intervention
 │    └─ D40: output stress / spectral / edge / blur
 ├─ D50: selected quadrant samples, gradient + one-step intervention
 └─ K10: same Teacher/Student snapshots, 3-arm microtrials
       └─ K20: fixed policy pilot only after provenance/routing/utility gate
             └─ Report + H1–H4 decisions
```

D10은 먼저 충분한 n의 전체 atlas를 만들고, D30/D40/D50은 그 안의 사전 지정 상세 subset으로 제한한다. Whole-scene RR/FR learned/zero core 검증은 subset 때문에 없어지지 않는다.

### 17.2 구현 helper의 의미

다음은 **신규 adapter 요구사항**이며 기존 저장소에 같은 이름의 실행 명령이 있다는 뜻이 아니다.

```text
resolve_assets_and_reproduce()
make_grouped_data_manifest()
cache_native_predictions_and_offsets()
measure_probe_bank(A_or_B, modality, fixed_roi)
assign_quadrants_from_calibration()
intervene_correction_with_frozen_unet()
measure_native_geometry_with_calibrated_estimators()
measure_output_stress_and_spectral_profiles()
run_resettable_gradient_interventions()
run_paired_hard_soft_microtrials()
fit_frozen_utility_tables()
run_matched_gate_pilot()
write_hypothesis_verdicts()
```

이름만 있는 기존 도구를 검증 완료라고 처리하지 않는다. `pa_diag.py`, `po10_diag`, `palsv18_report.py`, `align_after_training_diag.py`의 실제 구현과 출력 schema를 확인해 재사용하고, 없으면 새 helper임을 표시한다. 추정기 source가 없으면 그 검증은 NA로 둔다.

### 17.3 Profiling과 약 20시간 운영

첫 128 native patch + probeA, RR/FR 각1scene, K10 한trial의 실제 시간을 측정해 ETA를 갱신한다. CPU estimator·I/O·그림 생성과 GPU training을 따로 기록한다.

- 약 20h를 넘는다고 진행 중인 matched arm 하나를 잘라내지 않는다. **대응 묶음을 끝내고** 초과 사유를 남긴다.
- 속도가 빠르면 남는 시간을 새로운 λ 탐색에 쓰지 않는다. L1E4의 누락 집단/seed·Known-inverse·support 검증을 먼저 보완한다.
- K20까지 못 가도 D10–D50/K10이 가설의 원인을 설명하면 의미 있는 완료다. `H4 final_training_pilot_not_run`을 명시한다.
- D20의 측정이 틀렸으면 q 기반 KD 학습을 강행하지 않는다. 지표 검증을 먼저 끝낸다.
- 같은 data/asset을 불러 원시값을 재현할 수 없다면 `reuse_failed`로 기록한다. 결과보고서 표의 평균값으로 sample별 cache를 만들어 채우지 않는다.
- 새로운 파일은 campaign 단위로 저장한다. 원래 Teacher/Student checkpoint와 다른 서버 결과를 변경하지 않는다.

**종료 목표는 시계가 아니라 질문이다.** 네 loss 조합이 실제로 존재하는지, 각각 어떤 개입에 반응하는지, q가 기존 e/a를 넘어 soft/hard 유용성을 설명하는지의 답과 제한을 남기면 된다.

---

<a id="config"></a>
## 18. 실행 설정 template

아래 key는 의미 명세다. 서버의 실제 config schema에 맞춰 매핑하되 값을 조용히 재해석하지 않는다.

```yaml
campaign:
  id: EQREC4_S1_v1
  server: s1
  nominal_wall_hours: 20
  hard_time_limit: false
  mode: diagnostic_and_short_adaptation

baseline:
  case: L1E4
  width: 112
  depth: [1, 2, 3]
  input_channels: 9
  task: ms_only
  lambda_offset: 0.0001
  offset_every_updates: 2
  offset_on_remainder: 1
  jitter_radius_hr: 2.0
  jitter_distribution: uniform_disk_area
  jittered_pan_to_unet_during_training: false
  recon_grad_to_aligner: true
  offset_grad_to_unet: false

assets:
  primary_teacher_run: PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1
  primary_teacher_checkpoint: best_raw
  student_A_run: PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1
  student_A_checkpoint_rule: first_saved_update_at_or_above_20000
  secondary_teacher_run: PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1
  secondary_teacher_checkpoint: best_raw
  student_B_run: PALS24_L1E4_W112_D123_WV3_S7777_N2LAST_R200_v1
  student_B_checkpoint_rule: first_saved_update_at_or_above_20000
  actual_file_and_hash: resolve_before_run
  freeze_teacher: true
  share_teacher_student_objects: false

data:
  split_seed: 314159
  grouping: original_scene_or_strip
  calibration_target: 512
  adaptation_fit_target: 2048
  policy_validation_target: 512
  locked_holdout_target: 1024
  explicitly_track_pretraining_exposure: true
  canonical_views_only: true

quadrants:
  error: native_hrms_l1
  consistency: unweighted_offset_component_mean_l1
  threshold_source: calibration_A
  threshold_rule: median_per_teacher_checkpoint_and_scale
  tie_goes_to: low
  min_unique_patches_for_cell_analysis: 32
  min_source_groups_for_supported_group_claim: 5
  detailed_samples_per_quadrant_max: 64

probes:
  radii_hr: [0.25, 0.5, 1.0, 2.0]
  bank_A_angles_deg: [0, 90, 180, 270]
  bank_B_angles_deg: [45, 135, 225, 315]
  identity_counted_in_error_mean: false
  record_full_2d_response: true

k10:
  teacher_student_pairs: 2
  strata: [e_low_high, q_low_high, teacher_advantage_sign]
  trials_per_cell_max: 4
  unique_fit_samples_target: 48
  updates_per_arm: 64
  arms: [BASE, EXTRA_HARD, EXTRA_SOFT]
  gamma_extra: 0.1
  student_unet_lr: 0.00001
  student_aligner_lr: 0.000001
  lr_schedule: constant
  optimizer_state: fresh_identical_per_trial_arm
  reset_snapshot_per_trial_arm: true
  extra_supervision_grad_to_aligner: false
  extra_supervision_grad_to_unet: true

k20:
  conditional_on_valid_k10: true
  primary_pairs: [A]
  optional_pairs: [B]
  updates_per_arm: 5000
  arms: [BASE, UNIFORM, EA, EAQ, EAQ_CONDITIONAL_SHUFFLE]
  gamma_extra: 0.1
  gate_utility_source: k10_policy_validation_C
  mean_soft_fraction_calibration: 0.5
  source_student_advantage_frozen: true
  gate_table_frozen: true
  lr_schedule: warmup100_cosine_new_5k
  endpoint_rule: exact_5000
  locked_eval: D
  extra_supervision_grad_to_aligner: false

evaluation:
  official_fr: raw_original_paper_mat20
  official_aggregation: mean_of_scene_products
  selector_reference: best_raw_hqnr_then_fr_fscc
  hqnr_tie_band: 0.0001
  old_hqnr_effect_reference: 0.0031
  aligned_views_are_diagnostics: true
  stress_reference: original_native_pan_and_ms
  stress_roi: common_valid_fixed_before_outcomes
  bootstrap_repeats: 2000
  bootstrap_unit: source_scene_group
```

---

<a id="sources"></a>
## 19. 근거 문서·인계 자료

이 계획은 아래 제공 문서와 현재 대화의 결정에 근거한다. **이번 작성에서 s1 CUDA 코드·원시 tensor를 실행하거나 live Sheet의 새 결과를 재조회한 것은 아니다.** 수치 재현과 실제 checkpoint 접근은 G00의 작업이다. 새 case의 효과·runtime은 모두 실행 전 가설/예약이다.

| Ref | 문서 | 사용 범위 |
|---|---|---|
| **S1** | `2026-09-14_palsv18-lambda-confirmed-and-na104-null.md` | §1 λ 정밀화·반응·V4, §4 불확실성, §7 시간·운영 |
| **S2** | `2026-09-13_s1_pals24-lambda-sweep.md` | L1E4 채택·seed 확인·fSCC·checkpoint 격자 |
| **S3** | `PAN_Aligner_L1E4_TechnicalSpec_Evidence_KD_Handoff_2026-09-14.md` | 고정 구조·loss·좌표·재현·KD gradient·수치 차이 |
| **S4** | `2026-09-03_alignment-audit-s2-detail.md` | descriptor·phase·spectral 측정 하한·native 정합 proxy 한계 |
| **S5** | `2026-09-04_global-alignment-plan-review (1).md` | 과거 MS 조건 이동 sweep의 범위; 현재 PAN-only와 구분 |
| **S6** | `PAN_L1E4_Refinement_AlignmentValidation_S1_18GPUh_2026-09-13.md` | 기존 V0–V4 probe·metric·mask·stress 지시의 재사용 근거 |

S1/S2에서 `정합 능력`이라 부른 표는 우선 **추가 offset 대응 지표**로 보존해 기록한다. 이를 native 센서의 절대 정합 GT 검증으로 자동 승격하지 않는다. 보고서의 `기전 규명` 표현은 해당 보고서의 해석이며, 이 문서는 가능한 원인을 더 분리하는 후속 검증이다.

기존 Sheet 참조: [pan-cvpr27 / WV3-s1](https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit#gid=994031662). 행 번호는 이동할 수 있으므로 전체 run ID로 찾는다. **Sheet의 scalar 평균으로 H1–H4의 sample별 결과를 대신하지 않는다.**

기존 기술 명세의 §14 run registry와 report의 `checkpoint_metrics.csv`, `scene_metrics.csv`, `offset_response_*.csv`가 있으면 provenance를 확인해 재사용한다. 원시 자료가 없거나 정의가 다르면 현재 protocol로 재측정하고 구분해 저장한다.

---

### 작성 시 확인한 원문 SHA-256

아래 값은 **이 문서 작성 환경에 있는 첨부 원문 bytes**의 hash다. 서버 checkpoint·dataset·Sheet snapshot hash가 아니며, 서버 실행 전에 해당 자산은 별도로 계산한다.

| Ref | 원문 SHA-256 |
|---|---|
| S1 | `3a1c60381bcfcf01ac318ad93d94c4f95e3ed844bd62dfd2d2551f05657b6af1` |
| S2 | `344a77e7342de5ed84a529e5ba53a8474393124462829c9ab13e6b1bfa700679` |
| S3 | `42e074a22e6f7c7fa255e8fc59c014b062d680d5a308081ee2512df850a351d2` |
| S4 | `197bfab70d240dc381b79b69c33a4262076e23158b5d46b962a3446ec1cab296` |
| S5 | `e6392ea26516cad149530a4147375d4b8e7a051149993acd247aabc1469f277f` |
| S6 | `857314f06af7e12a444189c1f860dd366c622ab761dea747e86733e3ead98dcf` |

---

## 20. 이 캠페인이 끝났을 때의 한 문장 결론 형식

> **현재 L1E4의 offset consistency는 [추가 이동 반응 / native 위치 개선 / native 복원 / Student 감독 유용성] 중 어디까지를 설명했으며, reconstruction과 엇갈리는 sample에서는 [검증된 원인 / 남은 후보]가 관측됐다. e_T·a_T에 q_T를 추가한 soft/hard 선택은 [개선 / 악화 / 미확인]이었고, 그 판단은 [동일 checkpoint·대응 개입·source 분할·실제 KD endpoint]에 근거한다.**

빈칸은 실제 산출물로 채운다. 네 집단의 해석을 실험 전에 정답으로 채워 두지 않는다. 이 검증이 끝날 때까지 **L1E4의 채택과 q_T의 Teacher-quality gate 채택은 별개의 결정**으로 유지한다.
