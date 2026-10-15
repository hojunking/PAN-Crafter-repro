# PANDA GF2 G20 — 전 서버 20시간 실험 Case 명세

**작성일:** 2026-09-20  
**Campaign:** `PANDA_GF2_G20_ALL5_20260920_v1`  
**적용 서버:** **s1·s2·s3·s4·s5 모두 GF2**  
**고정 구조:** `P0 · W112D123 Teacher → PLH · W104D122 Student` + 기존 PAN Aligner  
**시간:** 공통 기동 시점부터 **20시간 경과시간**, 준비·전환·학습·calibration·평가·보존 포함  
**문서 상태:** 실험 설계 및 case 등록용 명세. 서버 기동·중단, 코드 배포, queue 수정, live Sheet 쓰기는 수행하지 않았다.

> **이번 방향은 모델을 키우는 것이 아니라, 동일한 PANDA에서 reference와 학습 강도의 균형을 바꾸는 것이다.** 현재의 높은 `D_s`를 곧바로 “공간 디테일 부족”으로 해석하지 않는다. 첨부 감사에서 관찰된 PAN 과잉 상관이 현재 QG40에도 남아 있는지 확인하면서, Teacher consistency·Student Aligner LR·GT edge·soft KD 계수를 유한한 한 축 대조로 시험한다.

## 0. 실행 결정 요약

| 구분 | 이번 결정 |
|---|---|
| 자원 | s1까지 GF2로 전환한다. 이전의 “s1은 QB 유지” 방침을 이번 계획에는 적용하지 않는다. |
| 아키텍처 | Teacher/Student의 폭·깊이·블록·입력 구성·Aligner·attention·modulation·단일 HRMS task 모두 고정한다. |
| Method | Teacher native reconstruction + A-only consistency, adaptive hard/soft KD, q-weighted GT edge, q-weighted hard만으로 Student A 조정을 모두 유지한다. |
| 주 실험 | **Teacher 4개 + Student 28개 = 32개 50K case**를 기본 실행 대상으로 정의한다. |
| 추가 확인 | 조건과 시간이 맞을 때만 새 Student seed의 대응 확인 20개, reference×단일 Student 계수 전이 확인 4개를 연다. **등록 가능한 최대 56개이지, 56개 완료 보장이 아니다.** |
| 주 목표 | **같은 정상 A_ON checkpoint에서 raw-original HQNR > 0.964 및 ERGAS < 0.552**. ERGAS < 0.522는 별도 strong flag다. |
| 선택점 | Exact50K와 RR-validation-selected를 주 분석에 사용한다. RAW_MAX/TARGET/E_MIN은 별도의 test-aware 개발 진단이다. |
| 일정 | 16h 이후 신규 학습 입장 금지. 18h까지 학습·주 평가 완료를 예약하고, 18–20h는 미완 평가·보존·readback에 우선 배정한다. |
| 변경 금지 | MARs 추가, W/D 변경, input-mode 변경, LP filter 변경, KD/edge/q/A 제거, benchmark MTF 변경, PAN reference 이동, 출력 blending/TTA/후처리로 목표 맞추기. |

**실행 우선순위:** 수치 무결성 → 같은 seed의 기본 대조 → 유망 조건의 새 seed 확인 → 제한된 조합 전이. 시간 부족 시 뒤의 묶음을 줄이며, 비교군 없이 단일 최고점만 생산하지 않는다.

---

## 1. 현재 결과에서 출발할 위치

### 1.1 이번에 다시 읽은 현재 cohort

기준은 첨부 워크북 `pan-cvpr27_results_reviewed_2026-09-20.xlsx`의 `GF2-s3(5090)` 9–14행과 `GF2-s5` 6–10행이다. `QG40 role/sensor`로 현재 cohort를 구분하고, 과거 MULTISET/B01/B02/B03는 합산하지 않았다. **현재 등록분은 Teacher 1개, Student 10개이며, Student 10개 전부 동일한 GF2_TA/TS91001 reference를 사용한다.** [S1]

이번 확인은 워크북의 값·metadata 재독해와 계산이다. 파일에 없는 최신 서버 결과나 아직 업로드되지 않은 결과까지 조회한 것은 아니다. 기동 전에 §5의 live inventory로 중복 여부를 다시 확인한다.

| 원본 위치 | Student seed | H50 ↑ | E50 ↓ | Dλ50 ↓ | Ds50 ↓ | H 적격/50 |
| --- | --- | --- | --- | --- | --- | --- |
| GF2-s3(5090):10 | 92001 | 0.957673 | 0.556494 | 0.019257 | 0.023503 | 0 |
| GF2-s3(5090):11 | 92002 | 0.948618 | 0.557150 | 0.021567 | 0.030443 | 0 |
| GF2-s3(5090):12 | 92004 | 0.947259 | 0.559507 | 0.020571 | 0.032829 | 0 |
| GF2-s3(5090):13 | 92005 | 0.948944 | 0.560150 | 0.020841 | 0.030836 | 0 |
| GF2-s3(5090):14 | 92008 | 0.946408 | 0.558659 | 0.020989 | 0.033275 | 0 |
| GF2-s5:6 | 92003 | 0.947985 | 0.560600 | 0.020725 | 0.031930 | 0 |
| GF2-s5:7 | 92006 | 0.942695 | 0.560287 | 0.021171 | 0.036886 | 0 |
| GF2-s5:8 | 92007 | 0.944308 | 0.560001 | 0.021115 | 0.035300 | 0 |
| GF2-s5:9 | 92012 | 0.953741 | 0.561040 | 0.019489 | 0.027280 | 0 |
| GF2-s5:10 | 92013 | 0.950172 | 0.556985 | 0.020562 | 0.029872 | 0 |

위 각 행은 **동일한 Exact50K A/U checkpoint**의 값이다. 마지막 열만 해당 run의 저장된 50개 후보 중 HQNR > 0.964인 수다. 이 50개 checkpoint를 서로 독립적인 반복 실험으로 세지 않는다. [S1]

| 기준점 | HQNR | ERGAS | 해석 |
|---|---:|---:|---|
| 현재 Teacher TA Exact50K | 0.945199752 | 0.572899669 | Student에 제공한 실제 reference |
| 현재 최고 RAW_MAX: S92001, step45450 | 0.957832644 | 0.558401158 | 최고 HQNR 개발 선택점 |
| 같은 S92001 Exact50K | 0.957673462 | 0.556494284 | 주 비교에 쓸 고정 update 결과 |
| 전체 최소 RR: S92002, step46460 | 0.948371953 | 0.555709255 | 다른 checkpoint이며 위 최고 HQNR와 합치면 안 됨 |
| PAN-Crafter 보고값의 기존 주 기준 | 0.964 | 0.552 | 본문 0.522와의 불일치는 그대로 보존 |

Student 10개의 Exact50K HQNR 중앙값은 **0.948301**, 표본 표준편차는 **0.004363**이고, ERGAS 중앙값은 **0.559754**다. RAW_MAX의 step 중앙값은 **42,420**, `RAW_MAX H − H50` 중앙값은 **0.000269**다. 이는 주로 후반까지 낮은 수준에 머무는 패턴이며, QB처럼 이른 높은 HQNR가 후반에 크게 붕괴하는 패턴과는 다르다. 표준편차는 이 동일-reference/복수-server cohort의 기술통계이지, GF2의 보편적인 유의성 기준은 아니다. [S1; 본 문서 재계산]

### 1.2 결과가 지지하는 것과 아직 지지하지 않는 것

**지지하는 것:** 동일 TA에서 Student seed만 추가하는 것보다, reference와 학습 강도를 통제해 바꾸는 실험의 우선순위가 높다. 현재 best의 Dλ50=0.019257은 보고값 0.020 근방이지만 Ds50=0.023503은 보고값 0.017보다 높다. 다만 cohort 전체의 Dλ까지 이미 해결됐다는 뜻은 아니다. [S1]

**아직 지지하지 않는 것:** “Teacher가 절대적 상한이다”, “KD가 실패했다”, “디테일이 모자라므로 edge를 늘려야 한다”, “GF2에서 PAN을 덜 따르게 만들면 무조건 좋다”는 결론은 내리지 않는다. 실제 best Student는 TA보다 H50가 **0.012474 높고**, E50는 약 **2.86% 낮다**. Student 학습 전체는 개선을 만들고 있다. Teacher/reference의 원인성은 별도의 대조가 필요하다. [S1; 본 문서 재계산]

지금의 H50 best는 목표까지 약 0.006327, E50 best는 0.552까지 약 0.004494 남았다. **작은 상대 ERGAS 개선과 함께, 공간 지표의 개선을 재현 가능한 방향으로 만들어야 한다.** 어느 한 지표만 좋아진 결과를 공동목표 달성으로 승격하지 않는다.

---

## 2. 첨부 감사 MD 검토: 채택할 내용과 구분할 내용

참고 문서: `2026-09-20_gf2-hqnr-gap-audit.md`. 그 문서의 관측·검증 결과를 참고하되, 이번에 원 GPU 출력이나 CPU 감사 스크립트를 직접 재실행한 것으로 표현하지 않는다. [S2]

### 2.1 감사 문서의 “현 mainline”은 현재 QG40 Student가 아니다

| 구분 | 감사의 B01_GF2_P0 | 이번 계획의 현재 PANDA |
|---|---|---|
| 시기/계열 | 과거 SMEC12 준비용 무정합 baseline | QGBASE/QG40 |
| Aligner | 없음, A-ID/NOALIGN | Teacher/Student PAN Aligner 존재 |
| 학습 | 단일 HRMS L1 baseline | Stage1 native+consistency → Stage2 adaptive KD/edge/A adjustment |
| 복원망 | B01 W112D123, 입력 P0 | Teacher W112D123/P0, Student W104D122/PLH |
| 대표 H/Ds | 약 0.9314/0.0433 | 현재 best Student 약 0.9578/0.0233 |

감사의 **0.026–0.033 격차와 “80/80 PAN 과잉 상관”은 B01 대조의 결과**다. 현재 QG40의 격차나 현재 Student 전체의 부호 진단으로 옮겨 쓰지 않는다. `P0`라는 이름이 같아도 입력 layout을 뜻할 뿐, Aligner 유무나 전체 Method의 동일성을 보장하지 않는다. [S2 §0, §4.1, 부록 A]

### 2.2 이번에 실제로 활용할 근거

| 감사의 내용 | 이번 계획에서의 활용 | 적용 한계 |
|---|---|---|
| GF2 bit depth·MTF·set identity·F-1/F-3에 대한 감사와 anchor 비교 | 지표 상수를 성능에 맞춰 변경하는 탐색은 하지 않는다. 현재 서버의 explicit sensor/shape/hash를 짧게 재확인한다. | 과거 경로의 PASS가 현재 qg40 경로의 무결성을 자동 보증하지 않는다. |
| B01의 `Q_high−Q_low`가 양수로 치우침 | 현재 TA와 Student에서도 같은 **부호 있는 D_s 구성량**을 export한다. | 현재 출력의 부호는 아직 제공되지 않았다. |
| B02는 best HQNR가 높지만 RR가 악화되고 best/last 차이가 큼 | H-only 선택과 과도한 PAN 이동에 대한 경고로 사용한다. | B02를 현재 reference로 재사용하거나 현재 학습의 원인으로 단정하지 않는다. |
| Native FR와 Wald RR의 흐림/offset 특성 차이 | train/RR/FR 크기별 c·q·잔차를 진단한다. | 추정 GNyq와 NCC offset은 GT가 아니다. 이번 window의 augmentation/LP/출력 정책은 바꾸지 않는다. |
| dual와 B01은 폭·입력·task·modulation이 동시에 다름 | 아키텍처 효과를 분리했다는 주장을 하지 않는다. | 감사 §7의 MARs×input-mode 2×2와 W168 시험은 **이번 case에서 제외**한다. |

F-3 PASS는 MS–GT 생성 위상 일치에 대한 근거다. 이를 PAN–MS의 절대 정합 완료로 확대하지 않는다. 감사 §3.2 자체도 PAN–MS 잔여 offset을 별도로 보고한다. [S2 §3.1–3.2, §4.5]

### 2.3 이번 연구 가설

이번에 검증할 가설은 **“PANDA의 고정 구조 안에서도, reference의 정합 반응과 Student의 reconstruction/edge/soft supervision 및 Aligner 이동량의 학습 균형을 조정하면 GF2의 FR 지표와 RR 품질을 함께 개선할 수 있다”**이다.

따라서 edge·KD·A LR는 작은 값과 큰 값을 모두 시험한다. 감사에서 과잉 상관을 봤다는 이유로 현재 GT edge 계수를 반드시 줄여야 한다고 정하지 않는다. **GT edge loss는 PAN edge를 직접 복사하는 loss가 아니기 때문**이며, 현재 loss별 gradient가 만드는 방향을 따로 측정한다. 이 문장의 실험 방향은 새 제안이며, 이미 검증된 최적 계수라는 뜻이 아니다.

---

## 3. 절대로 바꾸지 않는 구조·Method·입력·평가

### 3.1 Architecture 계약

| 항목 | 고정값 |
|---|---|
| Teacher | P0, W112D123, MS-only/단일 HRMS, LN, attention OFF, mode modulation OFF |
| Student | PLH, W104D122, 현재 확정 backbone과 블록 구성 |
| GF2 band | MS 4, HRMS output 4 |
| Teacher input | aligned PAN 1 + upsampled MS 4 = 5ch |
| Student input | aligned P/L/H 각 1 + upsampled MS 4 = 7ch |
| PAN Aligner | 현행 구조, sample-wise 2D global translation, native PAN/MS만 입력, margin4 유지 |
| 입력 좌표 | MS/GT는 이동하지 않고, PAN을 MS frame으로 보정 |
| 추론 | Student 자신의 A+U, 정상 A_ON, 기존 full-frame 경로 |

WV3→QB/GF2에서 band-dependent 입출력만 달라진 기존 포팅 원칙을 그대로 따른다. 이번에는 같은 GF2이므로 그 shape마저 모든 case에서 같다. 모델의 정규화 layer, 새 head, feature gating, MARs, attention, block depth, width, residual 주입 방식은 실험 변수가 아니다. [S3 §10; S4 §5]

### 3.2 Forward와 P/L/H 동기 정합

\[
M=U_4(MS),\quad L=U_4(LPAN),\quad c=A(P_{m4},M_{m4}),
\]
\[
\widetilde P=W(P,c),\quad \widetilde L=W(L,c),\quad
\widetilde H=\widetilde P-\widetilde L.
\]

Teacher는 \(Z_T=M+F_T([\widetilde P,M])\), Student는 \(Z_S=M+F_S([\widetilde P,\widetilde L,\widetilde H,M])\)다. 두 모델은 각각 자기 Aligner를 사용한다. [S4 §5.1]

Warp는 FP32/bicubic/border/align_corners=False, c는 `(dy,dx)` HR PAN pixel 단위다. c를 ±2로 clamp하지 않는다. LP는 기존 Gaussian σ1.98/k41/replicate/[2::4,2::4], float64 생성/float32 cache를 유지하고, P/L은 같은 grid로 warp한다. H는 signed 차이이며 abs/clip/별도 정규화하지 않는다. LP를 센서별로 다시 설계하거나 입력 HP amplitude를 scaling하는 case는 없다.

### 3.3 Teacher loss: 바꿀 수 있는 것은 λcon 값뿐

\[
L_T(t)=\operatorname{mean}|Z_T-Y|
+\mathbf1[t\bmod 2=1]\lambda_{\rm con}L_{\rm con},
\]
\[
L_{\rm con}=\frac1{2B}\sum_i
\|c_{\epsilon,i}+\epsilon_i-\operatorname{sg}(c_{0,i})\|_1.
\]

Native reconstruction은 A/U 양쪽을 갱신한다. Synthetic PAN은 A-only branch에만 들어가며, consistency의 native 기준만 stop-gradient다. Synthetic shift는 반경2 HR pixel 원판 면적균등, 0-based odd update에서 적용한다. 반경·빈도·sampler·RNG는 모든 Teacher 대조에서 같게 둔다. [S4 §5.2]

Teacher는 fresh U/A이며 A 마지막 Linear만 zero-init한다. WV3 donor나 옛 B02/B03를 넣지 않는다. Reference는 **exact50K A/U 전체**이며, test HQNR best Teacher를 골라 연결하지 않는다.

### 3.4 Student loss와 gradient routing

\[
e_T=\operatorname{mean}_b|Z_T-Y|,\qquad e_S=\operatorname{mean}_b|Z_S-Y|,
\]
\[
d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\qquad
 a_T=\operatorname{sg}\frac{[e_S-e_T]_+}{e_S+10^{-6}},
\]
\[
\ell_{H,i}=\operatorname{mean}_p[(1+\alpha d_T)e_S],\quad
\ell_{K,i}=\operatorname{mean}_p[\beta(1-d_T)a_T\operatorname{mean}_b|Z_S-Z_T|],
\]
\[
L_U=\operatorname{mean}_i[\ell_{H,i}+\ell_{K,i}+\lambda_Es_i\ell_{E,i}],
\qquad L_A=\operatorname{mean}_i[s_i\ell_{H,i}].
\]

\(\ell_E\)는 HRMS와 GT의 signed Scharr 차이이며 기존 방향별 0.5·경계1px 제외·band mean을 유지한다. U에는 hard+soft+edge, **A에는 q-weighted hard만** 보낸다. soft/edge를 A에 보내거나 Student A를 freeze하지 않는다. `backward(L_U+L_A)`로 전체 model을 갱신하지 않는다. Teacher는 frozen, Student A는 독립 clone/trainable, U는 fresh다. [S4 §5.4]

### 3.5 공통 optimizer와 학습 길이

| 항목 | Teacher | Student |
|---|---:|---:|
| 실제 optimizer updates | 50,000 | 50,000 |
| Batch size | 48 | 48 |
| U peak LR | 1e-4 | 1e-4 |
| A peak LR | 1e-5 | BASE 3e-6; 지정 A1/A9만 변경 |
| Optimizer | AdamW, β=(.9,.999), eps=1e-8, wd=.01 | 동일 |
| Schedule | warmup100 + 기존 cosine50K | 동일 |
| Precision | FP32, AMP OFF | 동일 |
| Student α | — | 1.0 고정 |

현행 scheduler의 update indexing/첫 LR/마지막 LR를 상속한다. 같은 seed 대응은 profile명을 제외한 name-keyed 초기화·data stream·augmentation RNG를 사용한다. 초기 weight hash, sample/view 순서 hash를 기록한다. A LR 변경에는 동일 AdamW 식에서 생기는 update 차이가 포함되며, weight decay를 별도로 보정하지 않는다.

**이번 기본/조건부 case에는 75K/100K 연장이나 재시작을 넣지 않는다.** 같은 예산 비교에서 reference·계수 효과를 먼저 가른다. 긴 schedule은 필요성이 남았을 때 후속 실험으로 분리하며, 20h가 남았다고 cosine 종료 checkpoint를 즉석 재가열하지 않는다.

### 3.6 센서·평가 계약

GF2는 explicit `C=4`, maxDN1023, 정규화 `2*DN/1023−1`, 역변환 `(y+1)*511.5`를 사용한다. 경로의 `qb`/`wv3` 문자열을 따라 DN을 추론하지 않는다. Q4, 기존 RR support 20:-21/Q block32 및 현행 RR metric 정의를 유지한다. FR은 원 mat20/native PAN512/reference LMS와 **raw-original**로 평가한다. [S4 §4, §10; S2 §2–3]

Dλ의 GF2 GNyq=0.3 경로는 첨부 감사에서 확인된 정의다. 현재 evaluator와 불일치하면 source/anchor 차이를 먼저 해결하고 revision을 기록한다. 낮은 Dλ가 나오도록 MTF 상수를 고르는 것은 금지한다. 알고리즘용 LP, Wald 생성 filter, FR metric filter를 혼동하지 않는다.

HQNR는 장면별 `(1−Dλ)(1−Ds)`의 평균이다. 평균 Dλ/Ds의 곱으로 대체하지 않는다. c를 따라 evaluation PAN이나 LMS를 이동하지 않는다. JQM은 SRF-substitute 등 variant를 표시하는 보조값이고 공식 공동목표 판정에는 쓰지 않는다. 기존 clamp 규약은 고정하며, clip/no-clip을 test 성능에 따라 선택하지 않는다.

---

## 4. Reference registry와 calibration

### 4.1 R0: 현재 GF2_TA 자산

기존 TA는 삭제하지 않고 고정 대조 reference로 사용한다. 아래는 source snapshot의 식별값이며 실제 local file readback으로 검증한 뒤 소비한다. [S1]

| 항목 | 기록값 |
| --- | --- |
| Teacher alias / seed | GF2_TA / 91001 |
| A/U exact50K SHA256 | `0fb973376f9f00947d295620d96062b621145607aee311e28c889caefb656dab` |
| Calibration ID | `45cc75ebdc29867bd904d69ecefd3f2a3de1a80daf7145e5a6227350754c5da1` |
| τR | 0.005695626139640808 |
| qref | 0.4086490869522095 |
| q-cache SHA256 | `ddbb74a2a97661098a10db7bd9d17f5b7f7c5cb92b65ce39dd35751ed6b6d8ac` |
| Numerical revision | `QG40_SYNC_FREQ_C4_v1` |
| Recorded source bundle SHA256 | `0cf7bf2af1a81036a115b3b9a181c4cc0d3f292e7a7927383f25c5819dee9d8c` |

SHA는 내용 식별자이지 파일 위치가 아니다. 실제 checkpoint/calibration/cache의 local path를 inventory에서 확인한다. 위 source bundle SHA를 Git commit SHA로 혼동하지 않는다.

### 4.2 새 reference: 이름과 부모를 명확히 분리

| ID | 별칭 | 생성 서버 | Teacher seed | λcon | 역할 |
|---|---|---|---:|---:|---|
| R0 | GF2_TA | 기존 s3 자산 | 91001 | 1e-4 | 모든 Student 계수 대조의 공통 reference |
| R1 | GF2_G20_S1_TB_C100 | s1 | 91002 | 1e-4 | 독립 Teacher seed에 대한 reference 대조 |
| R2 | GF2_G20_S2_TB_C100 | s2 | 91002 | 1e-4 | s2 λcon 실험의 **동일 서버 BASE Teacher** |
| R3 | GF2_G20_S2_TB_C030 | s2 | 91002 | 3e-5 | 낮은 consistency 강도 |
| R4 | GF2_G20_S2_TB_C300 | s2 | 91002 | 3e-4 | 높은 consistency 강도 |

**R1/R2의 같은 seed 반복은 의도된 서버 대조다.** s2에서 BASE Teacher까지 직접 학습하므로 R2/R3/R4의 λcon 비교가 Teacher training server 차이와 섞이지 않는다. R1/R2를 동일 reference라는 이름으로 합치지 않으며, 각자의 A/U·τ/q/cache를 묶는다. SHA가 우연히 같더라도 origin receipt를 보존한다.

s2의 세 Teacher는 fresh 초기화 hash, 학습 sample/view sequence, synthetic shift RNG sequence가 같고 λcon만 달라야 한다. 학생 결과는 전체 reference bundle 변경 효과이며, qref 변화 한 항목의 효과로 분해했다고 주장하지 않는다.

이전 INDEP1의 `GF2_TB/91002`가 이미 완료되어 있으면 R1의 재사용 후보로 검사한다. 구조·수치·seed·정확한 50K·데이터·calibration 규약이 일치해야 한다. 이름만 같은 자산이나 기존 40h의 공유 TA를 TB로 재명명하지 않는다. **R2는 s2의 같은-server λ 대조를 위한 별도 local baseline**이므로 R1을 무조건 대신 넣지 않는다.

### 4.3 새 Teacher마다 calibration을 새로 만든다

Train에서 seed1234로 고정한 **동일 3,072 base ID**를 공유하며, τR는 무증강/full-pixel 최종 HRMS의 band-mean L1 오차 전체에 대한 median으로 계산한다.

\[
\tau_R=\max(\operatorname{median}_{i\in I,p}e_T(i,p),10^{-6}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROT4 view다. qref는 전체 train median이 아니라 기존 규칙대로 **3,072 base ID×4 view**의 median이다.

\[
q_{i,r}=\frac1{2K}\sum_{j=1}^{K}
\|c_{i,r,j}+\epsilon_j-c_{i,r,0}\|_1,\quad K=16,
\]
\[
q_{ref}=\operatorname{median}_{i\in I,r}q_{i,r},\qquad
s_{i,r}=\operatorname{sg}\frac{q_{ref}}{q_{ref}+q_{i,r}}.
\]

AXIS16 반경은 .25/.5/1/2 HR pixel의 네 방향이며, 단위·분모2·augmentation key를 유지한다. τ/q는 최적화할 자유계수가 아니라 **정해진 규칙의 측정값**이다. Teacher/augmentation/source가 바뀌면 재생성한다. 같은 R0를 쓰는 A1/E1/K05 등의 Student마다 q-cache를 새로 만들지는 않는다. [S4 §5.3]

새 reference는 `checkpoint + exact config + architecture signature + source/data/LP hashes + calibration indices + tau_R + q_ref + full q-cache + augmentation identity + readback receipt`가 모두 유효할 때만 사용한다. 초기값은 null이며 측정 전에 R0 수치를 복사하지 않는다.

---

## 5. 기동 전 점검: 과거 감사를 반복하지 않고 현재 경로를 검증한다

### 5.1 공통 t0를 기록하기 전에 계획 상태를 실제 상태와 대조

실제 준비/전환 작업을 시작할 때 UTC `t0`를 하나만 정하고 `deadline=t0+20h`를 배포한다. 문서 생성시각은 t0가 아니다. 이 20시간은 과거 QG40 deadline의 연장이 아니라 새 campaign이다.

각 서버에서 active run·남은 admitted block·미업로드 output·local Teacher·queue/source pin·GPU/CPU/디스크 inventory를 읽는다. 기존 진행 중 run/block은 checkpoint와 결과를 보존하며 안전하게 마무리하고 **미시작 QB/WV3 block을 새로 입장시키지 않는다**. 전환 대기 시간도 이 20h에 포함한다. 기존 Student에 새 Teacher를 중간 삽입하지 않는다.

### 5.2 P0 점검 목록

| 검사 | 통과 조건 | 실패 시 |
|---|---|---|
| Sensor/data | explicit GF2/4/1023, train/val/RR/FR source와 LP identity 일치 | 영향받은 경로만 차단, QB/WV3 fallback 금지 |
| Architecture | 현재 C4 BASE와 state-dict key/shape·module/forward 설정 동일 | 수정된 architecture case를 실행하지 않음 |
| Profile diff | 등록된 단일 scalar 외 config 변경 없음. Teacher seed 대조는 seed 차이만 허용 | resolved config diff부터 수정 |
| Gradient routing | Teacher con→A only, Student soft/edge→U only, hard→정의된 U/A 경로 | 해당 release 신규 학습 차단 |
| Frozen/clone | Teacher eval/frozen, Student A 독립 storage/trainable | reference 공유 메모리나 freeze 오류 수정 |
| Pair identity | 동일 Student seed의 U init 및 sample/view 순서 동일; 같은 reference면 A init도 동일 | seed 숫자만 같은 불완전 pair로 실행하지 않음 |
| Calibration | 3072 IDs, full cache coverage, sample/view key, online/cache 소표본 일치 | 해당 reference 소비 차단 |
| Evaluator | 동일 checkpoint 재평가 및 available anchor가 현재 source 규약과 일치 | 공식 결과 표시 보류, 원인 기록 |
| Runtime | local reference readback, 저장공간, 평가 대기량, 보수적 남은시간 확인 | 완료 가능한 대응 묶음만 입장 |

현재 TA와 S92001 Exact50K를 공통 evaluator로 재평가하는 작은 cross-server parity 작업을 우선한다. 초기 운영 허용치는 **ΔHQNR≤1e-5, ΔERGAS≤1e-4**로 두고, 초과하면 tensor·inverse normalization·metric source 차이를 조사한다. 이것은 수치 경로 검사 기준이지 통계적 유의성 기준이 아니다. 서로 다른 GPU의 **재학습 결과**에 이 parity 허용치를 강요하지 않는다.

Source snapshot의 `"1"` 같은 옛 placeholder는 빈값/숫자/완료 여부와 구분한다. `no_eligible` TARGET은 빈칸을 유지한다. Teacher 행의 Student seed placeholder를 실제 seed로 해석하지 않는다.

### 5.3 측정이 없어도 무작위 BASE 반복으로 되돌아가지 않는다

P0 수치 오류는 차단하지만, 높은 q 또는 낮은 Teacher HQNR 자체는 실패 gate가 아니다. 현재 데이터의 signed-Ds/gradient 진단이 아직 없다는 이유로 모든 서버를 BASE seed 반복에 투입하지 않는다. 이번 계수 대조 자체가 그 미확인 가설을 검증한다.

과거 QG40의 `qref/.46875≥.90` C3 gate는 당시의 보수적 운영 heuristic이었다. 현재 TA는 약 .872로 그 조건을 만족하지 않는다. **이번 s2는 그 gate를 통과했다고 처리하는 것이 아니라, 새 G20에 사전 정의한 양방향 λcon sensitivity 연구**다. 부호/단위/경로가 정상인 한 C030/C300을 비교하고, gate나 q 정의를 바꿔 효과를 만들어내지 않는다. [S5 §6; 이번 설계 변경]

---

## 6. 서버별 핵심 질문과 profile

| 서버 | 주 질문 | 고정할 것 | 변화시킬 것 |
|---|---|---|---|
| **s1** | 현재 결과가 한 Teacher realization에 묶여 있는가? | architecture·loss coefficients·Student profile | TA/91001 → 새 TB/91002 전체 reference |
| **s2** | Teacher의 task-fitting과 shift consistency 균형을 바꾸면 Student까지 개선되는가? | 같은 서버·Teacher seed91002·초기화·data/ε stream | λcon = 3e-5 / 1e-4 / 3e-4 |
| **s3** | Student의 A가 너무 약하게 또는 과하게 조정되는가? | R0, α/β/λE/U LR | A peak LR = 1e-6 / 3e-6 / 9e-6 |
| **s4** | GT edge supervision의 강도가 RR–FR 균형에 맞는가? | R0, α/β/U/A LR | λE = .001 / .002 / .004 |
| **s5** | 현재 Teacher prediction을 어느 정도 따라야 하는가? | R0, α/λE/U/A LR | β = .05 / .10 / .20 |

같은 reference의 대조에서는 U/Aligner의 최종 출력 크기·학습 가능 parameter 수·forward 설정이 모두 같아야 한다. scalar 변경 때문에 architecture signature가 달라지면 해당 run은 무효다.

계수 값은 모두 **이번 실험을 위한 사전 정의 후보**다. GF2에서 좋다고 이미 관측된 값이 아니다.

### 6.1 Student profile registry

| Profile | α | β | λE | Student A peak LR | BASE 대비 변화 |
|---|---:|---:|---:|---:|---|
| BASE | 1 | .10 | .002 | 3e-6 | 없음 |
| A1 | 1 | .10 | .002 | 1e-6 | A LR ×1/3 |
| A9 | 1 | .10 | .002 | 9e-6 | A LR ×3 |
| E1 | 1 | .10 | .001 | 3e-6 | GT edge 계수 ×1/2 |
| E4 | 1 | .10 | .004 | 3e-6 | GT edge 계수 ×2 |
| K05 | 1 | .05 | .002 | 3e-6 | soft KD 계수 ×1/2 |
| K20 | 1 | .20 | .002 | 3e-6 | soft KD 계수 ×2 |

A1/A9는 **첫 update부터 전체 A LR 곡선의 peak를 변경**하는 실험이다. 과거 A24R의 24,240-step 이후 감속과 다른 profile이다. 현재 GF2의 late-collapse 근거가 약하므로, late-only 정책을 기본 선택하지 않는다. A는 모든 profile에서 trainable이며, 기존 cosine의 최종 LR=0만 예외다.

E1/E4는 PAN/HPAN target을 쓰지 않고 GT edge만 사용한다. K05/K20는 teacher-advantage gate를 그대로 둔다. Student가 Teacher보다 정확해 gate가 닫히는 영역에서 β 증가가 자동으로 유효한 supervision을 만드는 것은 아니다. gate 활성률과 weighted gradient를 반드시 함께 기록한다.

### 6.2 공통 seed와 대응 설계

Screen Student seed는 **93001, 93002**, 새 seed 확인은 **93011, 93012**로 고정한다. 과거 GF2 best seed92001만 골라 tuning하지 않는다. 새로운 campaign/run namespace를 써 기존 이름 충돌을 검사한다.

같은 seed의 profile별 초기 U와 data/view 순서를 맞춘다. 같은 reference에서는 Student A 초기값도 같다. 서로 다른 reference에서는 A clone 자체가 달라지는 것이 정상이며 전체 reference 차이의 일부다.

서버별 BASE는 **해당 서버의 계수 대조군**이라 의도적으로 반복한다. s3/s4/s5에서 seed93001을 각각 학습했다고 독립 seed가 3개라고 세지 않는다. 보고 단위는 `(reference, server, Student seed, profile)`이고, 효과는 각 local BASE와의 paired 차이다.

---

## 7. 기본 32개 50K case

### 7.1 Teacher 4개

| Case | 서버 | 생성 reference | TS | Profile | λcon |
| --- | --- | --- | --- | --- | --- |
| G20-T01 | s1 | R1 | 91002 | C100 | 1e-04 |
| G20-T02 | s2 | R2 | 91002 | C100 | 1e-04 |
| G20-T03 | s2 | R3 | 91002 | C030 | 3e-05 |
| G20-T04 | s2 | R4 | 91002 | C300 | 3e-04 |

Teacher case에는 exact50K 평가와 **각각 독립된 calibration/reference 완성**이 뒤따른다. calibration은 50K 학습 case 수에는 넣지 않지만 시간 예산에는 반드시 넣는다.

s1은 T01 완성 후 R1을 local 소비한다. s2는 **T02(C100) → T04(C300) → T03(C030)** 순을 기본으로 하고, 각 reference가 완성되면 해당 Student를 이어서 수행해도 된다. 이 순서는 시간 부족 시 BASE+한 대조부터 완결하기 위한 운영 우선순위이며 C300이 더 좋다는 판정은 아니다. s2는 s1의 Teacher 완료를 기다릴 필요가 없다.

### 7.2 s1 — 독립 Teacher reference

| Case | Student seed | Reference | Student profile |
| --- | --- | --- | --- |
| G20-S01 | 93001 | R0 | BASE |
| G20-S02 | 93001 | R1 | BASE |
| G20-S03 | 93002 | R1 | BASE |
| G20-S04 | 93002 | R0 | BASE |

각 seed에서 R0와 R1의 BASE Student를 대응시킨다. 위 네 개는 같은 서버에서 수행한다. R1 Teacher의 raw H가 R0보다 낮다는 이유만으로 Student 두 seed를 생략하지 않는다.

### 7.3 s2 — Teacher λcon 대조

| Case | Student seed | Reference | Student profile |
| --- | --- | --- | --- |
| G20-S05 | 93001 | R2 | BASE |
| G20-S06 | 93001 | R3 | BASE |
| G20-S07 | 93001 | R4 | BASE |
| G20-S08 | 93002 | R4 | BASE |
| G20-S09 | 93002 | R3 | BASE |
| G20-S10 | 93002 | R2 | BASE |

R2가 λcon 대조의 BASE다. R3/R4를 R0와만 비교해서 consistency 효과라고 부르지 않는다. 세 Teacher의 calibration이 준비되는 순서에 맞춰 각 reference의 Student를 먼저 실행할 수 있지만, paired BASE가 모두 갖춰지기 전에는 승격 판정을 하지 않는다.

### 7.4 s3 — Student Aligner LR

| Case | Student seed | Reference | Student profile |
| --- | --- | --- | --- |
| G20-S11 | 93001 | R0 | BASE |
| G20-S12 | 93001 | R0 | A1 |
| G20-S13 | 93001 | R0 | A9 |
| G20-S14 | 93002 | R0 | A9 |
| G20-S15 | 93002 | R0 | A1 |
| G20-S16 | 93002 | R0 | BASE |

첫 seed는 BASE→낮은 값→높은 값, 두 번째 seed는 역순으로 실행한다. 모든 결과는 같은 R0/같은 seed의 local BASE에 대한 차이로 보고한다.

### 7.5 s4 — GT edge 강도

| Case | Student seed | Reference | Student profile |
| --- | --- | --- | --- |
| G20-S17 | 93001 | R0 | BASE |
| G20-S18 | 93001 | R0 | E1 |
| G20-S19 | 93001 | R0 | E4 |
| G20-S20 | 93002 | R0 | E4 |
| G20-S21 | 93002 | R0 | E1 |
| G20-S22 | 93002 | R0 | BASE |

첫 seed는 BASE→낮은 값→높은 값, 두 번째 seed는 역순으로 실행한다. 모든 결과는 같은 R0/같은 seed의 local BASE에 대한 차이로 보고한다.

### 7.6 s5 — soft KD 강도

| Case | Student seed | Reference | Student profile |
| --- | --- | --- | --- |
| G20-S23 | 93001 | R0 | BASE |
| G20-S24 | 93001 | R0 | K05 |
| G20-S25 | 93001 | R0 | K20 |
| G20-S26 | 93002 | R0 | K20 |
| G20-S27 | 93002 | R0 | K05 |
| G20-S28 | 93002 | R0 | BASE |

첫 seed는 BASE→낮은 값→높은 값, 두 번째 seed는 역순으로 실행한다. 모든 결과는 같은 R0/같은 seed의 local BASE에 대한 차이로 보고한다.

### 7.7 Run ID와 machine-readable 등록

Teacher 형식:

`G20_GF2_T_{S1|S2}_TS91002_{C100|C030|C300}_P0_W112_D123_FRESH50_v1`

Student 형식:

`G20_GF2_S_{S1..S5}_{R0..R4}_TS{teacher_seed}_SS{student_seed}_{profile}_PLH_W104_D122_FRESH50_v1`

전체 case와 구체 run ID는 동봉 `PANDA_GF2_G20_ALL5_Cases_2026-09-20.csv`에 있다. Run ID의 대문자 server 표기는 식별용이며 init RNG의 입력으로 쓰지 않는다. CSV는 등록용 명세이고 기존 qg40 runner가 그대로 해석한다고 보장하는 YAML이나 실행 명령이 아니다. 실행자는 이 계약을 자기 runner의 명시적 config로 옮긴 뒤 config-diff/gradient 검사를 수행한다.

---

## 8. 진단: H/Ds가 낮아진 이유를 같이 남긴다

### 8.1 현재 출력의 PAN 과잉/부족 상관부터 구분

TA Teacher와 현재 S92001/S92002/S92006 Exact50K, 그리고 신규 모든 Teacher/Student의 주 선택점에서 평가기의 `Q_high`, `Q_low` 중간량을 장면×밴드로 export한다.

\[
\delta_{i,b}=Q^{high}_{i,b}(Z_b,P)-Q^{low}_{i,b}(LMS_b,P_{low}).
\]

여기서 Q는 **현재 D_s 구현이 실제로 사용하는 중간량**이다. 새 NCC/단순 correlation으로 대체하지 않는다. block aggregation, PAN_low 생성, 경계 처리를 evaluation source와 같게 유지하며, export한 중간량으로 복원한 Ds가 공식값과 일치하는지 확인한다. 이 \(\delta\)는 alignment-consistency error \(q\)와 다른 변수다.

장면별·밴드별 signed δ, |δ|, 양수 비율, Q_high/Q_low 각각을 저장한다. 감사의 B01에서 양수였다는 이유로 신규 값도 양수라고 채우지 않는다. [S2 §4.1을 현재 모델에 적용하는 신규 진단]

### 8.2 측정 패널

| 축 | 기록할 실제 값 | 피할 해석 |
|---|---|---|
| Teacher 정합 반응 | native c의 mean/median/p95, 반경별 Δc 대 −ε response gain, cross-axis leakage, q 분포 | q 감소 = native displacement GT 개선 |
| Teacher loss 균형 | weighted-con/rec의 A gradient norm 및 cosine, con-on/off update 구분 | scalar L1이 작으니 gradient도 작다 |
| Student A | A init 대비 c 변화, q-weighted hard의 A gradient, border sampling 변화 | 큰 c = 큰 실제 misalignment |
| Student U | hard/soft/weighted-edge gradient norm과 pairwise cosine | β/λE의 숫자 크기 = 실제 학습 지배력 |
| Adaptive weighting | eT/eS 분포, d/s 분포, advantage>0 비율, 평균 soft weight | q가 작음 = 복원이 쉬움 |
| Spectral/edge | bandwise RR bias/L1/RMSE, GT edge 오차, 입력과 출력 HP energy | Ds 하락 = 선명도 향상 |
| 크기 변화 | train64, crop128, RR256, FR512에서 c/q와 분포 변화 | crop-consensus 결과를 정식 inference 성능으로 보고 |

진단은 학습 optimizer를 step하지 않으며, 진단 전후 mode·RNG·gradient buffer를 복구해 원래 학습 경로를 바꾸지 않는다. 각 term의 gradient를 재려면 동일 minibatch와 동일한 parameter subset에서 계산하고, weighted/unweighted norm을 구분한다.

고정 train 진단은 128 base samples, gradient 진단은 그중 24개를 사용한다. 기존 calibration ID와의 관계를 기록하고 데이터 성능을 보고 표본을 고르지 않는다. 신규 run의 10K/24,240/50K에 같은 표본·별도 RNG로 측정한다. c 통계의 source support와 augmentation을 표시한다. FR에서 GT reconstruction error를 계산했다고 쓰지 않는다. [S4 §6; 이번 추가 signed-Ds 패널]

### 8.3 안전한 성능 해석

`H 상승 + Ds 하락`만으로 통과시키지 않는다. **같은 checkpoint의 ERGAS/Dλ/GT edge/band bias와 c 추세**를 함께 본다. H만 좋아지고 RR가 크게 나빠지면 `HQNR_ONLY_TRADEOFF`로 남기며, 별도 metric-gaming 의도를 추정하지 않는다.

PAN을 이동시키는 것 자체는 제안 Method의 동작이다. 따라서 비영 c를 벌점으로 삼지는 않는다. 다만 큰 공통 shift bias는 q 차분에서 소거되므로, synthetic response만 좋아졌다고 native 정합이 검증됐다고 할 수 없다. current q_const=.46875도 물리적 정합 오차 기준은 아니다.

Signed δ가 0에 가까워지고 RR/분광 품질이 유지될 때는 유망한 균형 개선으로 해석한다. δ가 음수로 크게 넘어가거나 GT edge 품질이 악화하면 부호/주파수 진단과 함께 trade-off를 보고한다. **공식 성능은 수정하지 않고**, 이런 진단을 성공 원인에 대한 과장을 막는 자료로 사용한다.

---

## 9. Screen·확인·조합의 사전 판정 규칙

아래 숫자는 **이번 20h 자원 배분을 위한 운영 기준**이다. 통계적 유의성이나 GF2의 알려진 noise floor라고 표현하지 않는다. WV3의 2σ=0.0031을 GF2에 복사하지 않는다.

### 9.1 주 분석점과 목표

각 run은 기존 50-point 격자 `1010,2020,…,49490,50000`를 보존한다. Exact50K를 공통 update 비교에 사용하고, RR_VAL_SELECTED는 **validation ERGAS 최소**의 사전 규칙으로 정한다. 동률 tie-break는 기존 evaluator 규칙을 먼저 확인하여 동일하게 고정한다. test H로 validation checkpoint를 재선택하지 않는다.

TARGET은 `H>.964` 후보 중 ERGAS 최소, 이후 동률 SCC/PSNR/낮은 step 순의 기존 개발 규칙이다. `RAW_MAX`, `TARGET`, `E_MIN_DIAG50`는 test-aware다. 성능보다 유리한 선택점을 임의로 섞지 않는다. 새 seed 확인도 같은 test set을 독립 test로 바꾸지 않는다.

### 9.2 두 seed screen의 paired 기준

각 seed k에서 **Exact50K**의 차이를 계산한다.

\[
\Delta H_k=H^{alt}_k-H^{base}_k,\quad
r_{E,k}=E^{alt}_k/E^{base}_k-1,\quad
\Delta D_{\lambda,k}=D^{alt}_{\lambda,k}-D^{base}_{\lambda,k}.
\]

기본 유망 조건 `PROMISING_PAIRED`:

1. 두 seed 모두 ΔH > 1e-5이고, ΔH 중앙값 ≥ **0.0015**.
2. rE 중앙값 ≤ **+0.5%**, 개별 seed rE ≤ **+1.0%**.
3. ΔDλ 중앙값 ≤ **+0.001**, 개별 seed ΔDλ ≤ **+0.002**.
4. P0/paired identity/공식 평가가 정상이어야 한다. RR_VAL_SELECTED에서도 rE 중앙값≤+0.5%, 개별 rE≤+1.0%, paired ΔH 중앙값≥−0.0015를 확인한다. VAL 결과를 누락해 PASS로 표시하지 않는다.

위 0.0015는 목표 gap을 메웠다는 뜻이 아니다. H와 E 공동목표는 별도로 판정한다. Ds 자체의 0.017/0.020 임계값을 공식 성공 기준으로 대체하지 않는다.

**단일 seed에서 실제 공동목표를 통과했으나 위 paired 조건이 불충분하면 `JOINT_SINGLE_RETEST`**로 표시하고 새 seed 확인 후보로만 남긴다. 즉시 default로 승격하지 않는다. 한 seed의 조기 TARGET이 좋다는 이유로 다른 seed의 Exact50K 악화를 숨기지 않는다.

### 9.3 서버별 하나만 확인 후보로 고른다

s1은 R1 vs R0, s2는 R3/R4 각각 vs R2, s3는 A1/A9 vs BASE, s4는 E1/E4 vs BASE, s5는 K05/K20 vs BASE다.

`PROMISING_PAIRED`를 우선하고, 없으면 `JOINT_SINGLE_RETEST` 중 한 개만 확인한다. 동급이면 **RR_VAL_SELECTED 공동목표 seed 수 → Exact50K 공동목표 seed 수 → paired ΔH 중앙값 → 낮은 E 중앙값** 순으로 고른다. 모두 미충족이면 새 BASE seed를 계속 추가하지 않는다. fallback profile을 새로 발명하지 않고 실패한 두 방향을 그대로 보존한다.

후보 선택 시 `screen_decision_sX.json`에 기준값·실제값·원본 file hash·선택시간을 기록하고, 이후 다른 서버의 좋은 결과가 나와도 진행 중 확인 pair를 바꾸지 않는다.

### 9.4 새 seed 확인: 최대 20개 조건부 case

각 서버에서 선택된 한 조건만 **93011,93012**로 local BASE와 대응한다. 각 run은 update0부터 fresh이며 profile명에 따라 init이 달라지지 않는다.

| Case 묶음 | 서버 | 기준/변경 | 새 seed | 학습 수 |
|---|---|---|---|---:|
| G20-C01–C04 | s1 | R0/BASE ↔ R1/BASE | 93011,93012 | 4 |
| G20-C05–C08 | s2 | R2/BASE ↔ 선택된 R3 또는 R4/BASE | 93011,93012 | 4 |
| G20-C09–C12 | s3 | R0/BASE ↔ 선택 A1 또는 A9 | 93011,93012 | 4 |
| G20-C13–C16 | s4 | R0/BASE ↔ 선택 E1 또는 E4 | 93011,93012 | 4 |
| G20-C17–C20 | s5 | R0/BASE ↔ 선택 K05 또는 K20 | 93011,93012 | 4 |

4-seed 기준의 `REPRODUCIBLE_GAIN` 운영 표시는 3/4 이상 ΔH>1e-5, 전체 paired ΔH 중앙값≥.0015, 새 두 seed 중 적어도 하나에서 양의 ΔH, 그리고 §9.2의 E/Dλ guard를 전체 4개 seed에도 만족할 때 사용한다. 이는 형식적인 통계적 유의성 주장이 아니다.

`JOINT_REPRODUCED`는 **새 두 seed의 RR_VAL_SELECTED가 둘 다 H>.964 및 E<.552**일 때 별도로 표시한다. `REPRODUCIBLE_GAIN`인데 공동목표에는 미달할 수 있고, TARGET 한 점만 통과하면 `JOINT_SINGLE/TEST_AWARE`일 수 있다. 구분을 유지한다.

이 확인은 Student seed 변화에 대한 것이다. 하나의 Teacher training seed91002에서 얻은 λcon 결과를 Teacher seed-robust하다고 부르지 않는다.

### 9.5 제한된 reference×계수 전이: 최대 4개 조건부 case

조건: s1 또는 s2에서 기술적으로 유효하고 유망한 reference R*가 나오고, s3–s5 중 한 축의 Student profile X*가 두 seed screen을 통과한 경우에만 연다. **Teacher reference 하나 + Student scalar 하나**만 결합한다. A1+E1+K05 등 여러 Student 계수를 동시에 바꾸지 않는다.

X*를 시험했던 원 서버(s3/s4/s5)에서 screen seed **93001/93002**를 그대로 사용해 다음 네 run을 수행한다.

| Case | Seed | Reference | Student profile |
|---|---:|---|---|
| G20-X01 | 93001 | R* | BASE |
| G20-X02 | 93001 | R* | X* |
| G20-X03 | 93002 | R* | X* |
| G20-X04 | 93002 | R* | BASE |

그 서버의 기존 `R0/BASE`와 `R0/X*` screen 결과를 재사용해 **R0/R* × BASE/X***의 2×2를 완성한다. 동일 seed/초기 U/data stream/source가 일치해야 하며, 다른 서버의 BASE를 가져와 local control처럼 처리하지 않는다.

변화량은 `(R*/X*−R*/BASE)−(R0/X*−R0/BASE)`도 함께 기록한다. reference 변경과 Student 계수의 효과를 각각 확인할 수 없는 단일 combined run만으로 “시너지”라고 주장하지 않는다.

운영은 local fresh-seed 확인이 기본 우선이다. 다만 X* 서버의 screen 완료 시점에 R*가 이미 준비되어 있고 **전이 4개 전체가 budget 내에 들어오는 경우**, 같은 서버의 아직 시작하지 않은 확인 4개보다 전이를 먼저 배정할 수 있다. 이 선택은 `transfer_receipt`에 기동 전에 고정한다. 그 뒤 시간이 모자라면 새-seed 확인은 생략하고, combined 결과를 exploratory/test-aware로만 표시한다.

R*/X*가 미정인 CSV 행은 `NOT_ADMITTED`다. reference SHA·profile의 단일 scalar·owner를 manifest에 resolve한 뒤에만 실행한다. 시간 부족 시 다른 서버로 임의 이동해 필요한 R0 controls를 생략하지 않는다.

---

## 10. 20시간 자원 운영

### 10.1 처리량의 근거와 계획용 예약

현재 시트의 GF2 Teacher train 시간은 약 **0.672h**, Student는 s3에서 약 **0.947–0.949h**, s5에서 약 **0.965–0.966h**로 기록돼 있다. 이것은 해당 칸의 train 시간이며 **전체 calibration/50후보 평가/전송/보존 시간이라고 가정하지 않는다**. s1/s2/s4의 현재 GF2 처리량은 이 첨부만으로 확인되지 않는다. [S1]

초기 예약값은 Teacher50K+주 평가 **1.0h**, Student50K+주 평가/진단 **1.5h**, reference calibration/cache **1.0h**로 잡는다. 이는 새 계획용 가정이다. 실제 첫 warmup/첫 후처리와 pending 평가량을 보고 상향 조정하며, 느린 서버 때문에 batch/precision/method를 몰래 바꾸지 않는다.

| 서버 | 기본 학습 구성 | 위 가정의 기본 소요, 공통 준비 약1h 포함 | 남는 시간의 우선 사용 |
|---|---|---:|---|
| s1 | T1 + calibration1 + S4 | 약9h | reference 새 seed 확인 4개 |
| s2 | T3 + calibration3 + S6 | 약16h | 주 대조 완결 우선; 처리량이 빠를 때만 새 seed 확인 |
| s3 | S6 | 약10h | A LR 후보 확인 또는 제한 전이 |
| s4 | S6 | 약10h | edge 후보 확인 또는 제한 전이 |
| s5 | S6 | 약10h | KD 후보 확인 또는 제한 전이 |

이 표가 s2의 최대 10개 Student까지 20h에 끝난다는 약속은 아니다. **모든 56 case를 위 상한 시간으로 전부 실행하는 것은 불가능한 서버가 있으므로**, 조건부 묶음은 admission 계산으로 줄인다. 5개 서버×20h=명목 100 server-slot-hour는 예산 설명이지 실제 GPU 사용시간 측정값이 아니다.

### 10.2 시간대별 배정

| 경과시간 | 실행 |
|---|---|
| 0–약1.5h | 기존 작업 안전 전환, source/seed/architecture/data/reference P0, R0 local readback, 공통 parity/현재 signed-Ds 회수 |
| 준비 통과 즉시–약10h | s1/s2 Teacher+calibration+Student chain, s3–s5 local 2-seed 계수 대조. 전 서버 공통 성능 barrier 없음. |
| 약10–16h | 끝난 서버부터 자기 screen 판정과 새 seed 확인. reference와 X*가 동시에 준비된 경우에만 제한 전이. |
| 16–18h | 신규 학습 입장 없음. 이미 예약된 학습/pair/평가를 완결하고 metadata를 고정. |
| 18–20h | 미완 공식 평가, 유효 선택점 확인, 저장/backup, Sheet readback, 종결 분석. |

준비·calibration이 늦어지면 위 구간을 그대로 뒤로 미는 것이 아니라 case 수를 줄인다. 20h는 상한이며, 유효한 질문 없이 시간을 채우려고 seed를 생성하지 않는다. 모든 서버가 GF2에 투입되더라도 무결성 문제나 유효 후보 부재로 쉬는 시간이 생길 수 있으며 이를 숨기지 않는다.

### 10.3 Atomic block admission

새 block은 `현재시간 < t0+16h`이고 다음 조건을 만족할 때만 입장한다.

`현재시간 + 보수적 block 잔여시간 + 이미 발생한 평가부채 <= t0+18h`

보수적 시간은 실제 측정 throughput에 최소 **1.25 안전계수**를 적용하고, 아직 미측정인 calibration/후처리에는 위 초기 예약값을 사용한다. 동일 서버 GPU 학습/진단/추론이 같은 슬롯을 요구하면 중복 예약하지 않는다. CPU 평가를 무제한 병렬화해 학습 I/O와 경쟁시키지 않는다.

Teacher block은 **Teacher50K + calibration + 해당 reference Student 두 seed + 필수 평가**를 포함해 예약한다. Student screen은 같은 seed의 local BASE/대조들이 빠지지 않도록 예약한다. 확인과 전이는 각각 **4개 대응 run 전체**의 시간을 확보해야 시작한다.

s2가 전체 세 λ를 완료할 시간이 부족하면 `R2 C100 + R4 C300`의 두-seed 비교 완결을 먼저 예약하고, R3 C030의 Teacher+calibration+Student2 묶음을 미입장 처리한다. 사전 순위일 뿐 좋은 결과만 고르는 취소가 아니다. Teacher만 학습하고 Student를 못 내려보낸 경우를 성능 실패로 기록하지 않는다.

예상치 못한 지연은 full-state checkpoint를 보존하고 `PARTIAL_TIME_LIMIT`로 표시한다. 50K 미완료를 Exact50K로 채우거나, 다음 campaign의 update를 이 20h의 성과에 합산하지 않는다. 정상적인 기존 run을 raw kill하지 않고 coordinator의 안전한 checkpoint 종료 절차를 사용한다.

### 10.4 서버 간 의존성과 실패 격리

s3–s5는 이미 존재하는 R0를 local 복사/검증한 뒤 시작하며 새 TB를 기다리지 않는다. s1의 R1과 s2의 R2/R3/R4는 독립 local chain이다. 새 reference 전이는 optional이므로 한 서버의 calibration 지연이 나머지 서버의 screen을 막지 않는다.

반대로 reference가 없거나 cache 검증이 실패하면 그 reference의 Student는 실행할 수 없다. TA를 임의 fallback으로 넣어 TB case 이름을 유지하지 않는다. 완료 자산 재사용은 동일 계약 확인 후 `REUSED_VERIFIED`로 기록하고, 재평가를 새 학습 결과처럼 두 번 세지 않는다.

---

## 11. 결과 저장·Sheet·논문용 해석

### 11.1 매 run의 필수 산출물

`resolved_config`, source pin/diff, architecture signature, U/A init hash, full-state checkpoint, data/view RNG identity, reference manifest, 50-candidate metrics, Exact50K/VAL/TARGET/RAW/E_MIN 선택 receipt, scene-wise RR/FR metrics, signed-Ds components, c/q/gradient diagnostic, train/eval/calibration 시간, 완료/미완료/기술실패 사유를 남긴다.

최소 공개 성능 행에는 `run_id / server / TS / SS / reference SHA / profile / selection / step / A+U SHA / H / E / Dλ / Ds / SAM / SCC / PSNR / Q4 / joint flags / status`가 있어야 한다. 서로 다른 checkpoint의 A와 U를 섞은 진단은 별도 `DIAGNOSTIC_ONLY`로 저장한다.

### 11.2 Sheet routing

| 서버 | 결과 tab |
|---|---|
| s1 | GF2-s1 |
| s2 | GF2-s2 |
| s3 | GF2-s3(5090) |
| s4 | GF2-s4 — 첨부 snapshot에는 없으므로 실제 live 존재 여부 확인 후 필요한 경우 생성 |
| s5 | GF2-s5 |

기존 QB/WV3 tab을 GF2 용도로 재명명하지 않는다. 과거 결과는 보존한다. 실제 header를 의미별로 읽고 `G20` identity/diagnostic 필드만 확장한다. tab마다 열 배치가 다르므로 고정 열번호를 복사하지 않는다. Upsert key는 `(sensor, campaign, run_id)`다.

`유의미한결과`에는 공동목표 통과뿐 아니라 **paired 개선, coefficient 두 방향 모두 무효, gradient 비활성, reference 민감도 등 다음 결정에 영향을 주는 결과**를 근거와 함께 남긴다. 단, 모든 낮은 run을 대표 성능으로 나열하거나 개발 중 test-aware 값을 독립 검증으로 표시하지 않는다. 이번 문서는 기록 규약만 정의하며 현재 live Sheet를 변경한 것이 아니다.

### 11.3 종료 상태 분류

| 상태 | 뜻 |
|---|---|
| JOINT_REPRODUCED | 새 두 seed의 validation-selected에서 공동목표 통과 |
| JOINT_SINGLE / TEST_AWARE | 일부 seed/개발 선택점에서만 공동목표 통과 |
| REPRODUCIBLE_GAIN | paired 운영 기준의 개선을 확인했으나 절대 공동목표와는 별개 |
| PROMISING_UNCONFIRMED | screen 개선은 있으나 새 seed 확인 미완 |
| HQNR_ONLY_TRADEOFF | H 개선에 RR/분광 품질 악화가 동반됨 |
| NO_GAIN | 유효한 동일조건 대조에서 우위가 확인되지 않음 |
| INACTIVE_TERM | 해당 weighted term의 영향이 실제 진단에서 거의 없거나 gate로 비활성; 구현 누락과 구분 |
| TECHNICAL_INVALID | 데이터/경로/routing/reference/평가 무결성 문제 |
| NOT_ADMITTED / PARTIAL_TIME_LIMIT | 예산상 미시작/미완, 성능 실패 아님 |

논문 보고에서는 architecture 고정 BASE 이식군과 sensor-tuned hyperparameter군을 분리한다. 같은 구조라는 사실이 모든 센서의 학습 계수까지 동일했다는 의미는 아니다. 선택된 계수와 test-aware 개발 이력을 공개하고, 유리한 단일 seed와 평균을 바꿔 쓰지 않는다.

---

## 12. 20시간 종료 시 남아야 할 답

**Reference:** 같은 PANDA에서 새 Teacher seed가 현재 TA보다 좋은 출발점을 제공했는가? s2의 같은-server λcon 대조가 Teacher와 Student에서 같은 방향을 보였는가? R1/R2 반복은 어느 정도 차이였는가?

**Student fitting:** A LR·GT edge·soft KD 중 어느 축이 H/Ds와 E/Dλ를 함께 개선했는가? 무엇이 실제 gradient를 바꾸었고 무엇은 사실상 비활성이었는가? 큰 값/작은 값의 두 방향에서 무효라면 그 사실도 결론이다.

**현재 감사 가설:** B01에서의 PAN 과잉 상관이 현재 QG40에서도 확인되는가? 새 profile의 Ds 개선은 signed δ의 균형 변화와 함께 나타나는가, 아니면 큰 shift 또는 RR 품질 저하와 함께 나타나는가?

**재현성과 목표:** 새 Student seed에서 개선이 반복되는가? H>.964/E<.552가 같은 정상 checkpoint에서 달성됐는가? 미달이면 남은 gap과 실패한 조정 축을 구체적으로 남긴다.

최종 산출물은 “최고점 하나”가 아니라, **동일 architecture에서 효과가 있는 reference/계수와 효과가 없는 축을 구분한 결과표, 보존된 재현 자산, 다음에 무엇을 반복하지 않을지에 대한 근거**다. 목표 달성 여부는 실행 후 결과로 판정한다.

---

## 부록 A. 전체 조건부 case의 resolve 계약

C01–C20과 X01–X04는 처음에는 `NOT_ADMITTED`다. 각 슬롯의 `R_CSTAR`, `S3_XSTAR`, `S4_XSTAR`, `S5_XSTAR`, `RSTAR`, `OWNER`를 실제 후보 중 하나로만 resolve한다. 후보 집합 밖의 계수/architecture를 넣지 않는다.

- s2 R_CSTAR ∈ {R3,R4}; Student profile=BASE.
- s3 XSTAR ∈ {A1,A9}; reference=R0.
- s4 XSTAR ∈ {E1,E4}; reference=R0.
- s5 XSTAR ∈ {K05,K20}; reference=R0.
- 전이 RSTAR ∈ {R1,R2,R3,R4}, XSTAR는 위에서 선택된 단일 Student profile이며 원래 XSTAR 소유 서버에서 수행한다.

CSV의 `RESOLVE_FROM_BUNDLE/PROFILE`은 실제 값이 아니라 미결 상태다. executor는 manifest를 고정하고 실제 run ID를 확정한 뒤 `PLANNED`로 바꾼다. unresolved 슬롯을 numerical default로 실행해서는 안 된다. profile에 없는 scalar는 BASE 값을 그대로 유지한다.

## 부록 B. 데이터/LP identity — source snapshot의 R0 기록

| 항목 | SHA256 |
| --- | --- |
| train data | `243a0bc8a4cc0a2740ed24409f9e87afe531d87e9d57524c6bcbec2b5f937621` |
| train LP | `af43c2aa96c26d92874e61052647e4cf3909e8fcb680a9c584d48cfe20e3fc3b` |
| val data | `ae15b19ccc6ece799ebfeb3fb374b1a333238226397441e3d523b0195dff4f43` |
| val LP | `aad58b4fd6435b695e1d00adc100c5a8d79450ea6e2d69ad33089e2fa7d9feae` |
| rr data | `709a9a53b2e0f29d3dcd5c6ca4410c2913b62cc03c104fed6c049911dbe1c8ea` |
| rr LP | `6535574fe8efb3dbfb72c1938ca2557fb868fd9f32af9598fb7ba367267d9489` |
| fr data | `e52d151f262a74f96e59f03124f86a27d80841073ce60376856ae679020d00dc` |
| fr LP | `f17335ea6c433c39081ad623a22a355a5f8b469e148c77d574e283c2f3165167` |

이 값은 현재 서버를 직접 읽은 결과가 아니라 첨부 Sheet 기록이다. 재직렬화 때문에 H5 file bytes가 달라졌다면 numerical equality와 sample-order/canonical tensor hash를 별도로 증명한다. 파일 hash 차이를 무조건 데이터 오류라고 하거나, 이름이 같다는 이유로 무조건 같은 데이터라고 하지 않는다.

## 부록 C. 출처와 검토 범위

**[S1] 현재 결과 snapshot**  
`pan-cvpr27_results_reviewed_2026-09-20.xlsx` — `GF2-s3(5090)` 9–14행, `GF2-s5` 6–10행의 QG40 fields와 선택점별 metric. 현재 T1/S10, reference 동일성, 50K/50후보/eligible, 본 문서의 기술통계는 이 파일에서 재독해했다.

SHA256: `030324ec15cf13d4e4a9c82e3e7b0ab681672808ffe5b65a1c2275ad5e25dee6`

**[S2] 사용자 첨부 감사**  
`2026-09-20_gf2-hqnr-gap-audit.md` — §2 metric audit, §3 데이터/정합, §4 signed-Ds/B02/recipe 차이, §7 제안 실험, 부록 A의 B01 정의. 과거 감사의 raw script/assets는 이번에 다시 실행하지 않았다. 이번 계획은 해당 문서의 architecture-changing 실험 제안을 채택하지 않는다.

SHA256: `54a951d58e7952e46c4dd63a6ed52b0b6438f79b3bd72c77a8a98853be3271a8`

**[S3] architecture 고정 근거**  
`PAN_WV3_Closeout_Review_FrozenRecipe_2026-09-20.md` — 주 조건 F1/P0/W112D123→PLH/W104D122, §10의 센서 이전 및 band-dependent shape 계약.

**[S4] 현재 수치 Method와 기본 학습 계약**  
`PAN_QG40_QB_GF2_40H_ExperimentPlan_2026-09-20.md` — §4 데이터, §5 forward/Teacher/calibration/Student routing, §6 진단, §7–8 과거 한 축 대조. 이번에는 그 architecture/수식은 유지하고 시간·서버배정·유한 profile/seed·분기 정책을 새 G20으로 정의한다.

**[S5] 인계된 결과 해석과 운영 불일치**  
`PAN_QG40_QB_GF2_Results_Review_NextDirection_2026-09-20.md`, `PAN_NEXT_SESSION_HANDOFF_2026-09-20_QB_GF2.md`, `PAN_QG40_INDEP1_LocalTeacher_Addendum_2026-09-20.md` — TA 집중, QB/GF2 시간 패턴 차이, old INDEP1와 실제 공유 TA 기록의 불일치. 이들 문서의 마지막 runtime 상태를 현재 live 상태로 가정하지 않는다.

**이번에 새로 정한 것:** all-five GF2/20h, case ID 및 새 seed, C030/C100/C300·A1/A9·E1/E4·K05/K20 후보, paired 운영 문턱, confirmation/transfer의 유한 조건, 시간 예약값이다. 이미 관측된 사실이나 최적값과 구분한다.

**미확인 항목:** 실제 서버의 새 결과/active process/INDEP1 적용 상태, raw checkpoint 파일 존재, 최신 code pin, 현재 모델의 signed-Ds/q/c/gradient 실측, 각 서버 calibration·evaluation 처리량. 실행자는 §5/§8로 확인하고, 미확인을 PASS로 채우지 않는다.
