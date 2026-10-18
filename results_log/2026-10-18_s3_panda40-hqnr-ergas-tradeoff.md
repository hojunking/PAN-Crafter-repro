# PANDA40 s3 Stage 2 결과 분석 — HQNR·ERGAS 상충의 원인과 균형 방법 (2026-10-18)

s3 의 `PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1` 캠페인(고정 donor + fresh W112·D123 Student, loss 변형 6종 × seed 2)이
10-17 22:08 KST 에 12 run 모두 끝났다(경과 25.46 h / 상한 40 h). 이 문서는 그 결과를 분석하고, **"HQNR 은 닿는데 ERGAS 는 안 된다"**
는 반복 관찰의 원인과 두 지표의 균형을 잡을 방법을 저장된 checkpoint 재평가로 검증한 것이다. 다음 캠페인을 구현할 서버가 참고하도록
권장 실험 명세(§7)와 재현 도구([`tools/panda40_tradeoff_diag.py`](../tools/panda40_tradeoff_diag.py))를 함께 둔다.
명세서는 [`research_log/PANDA_Stage2_Loss_Implementation_Spec_2026-10-16.md`](../research_log/PANDA_Stage2_Loss_Implementation_Spec_2026-10-16.md),
캠페인 계획은 [`research_log/PANDA_S13_40H_S2D_Stage2_2026-10-16.md`](../research_log/PANDA_S13_40H_S2D_Stage2_2026-10-16.md), 실행 WIP 는
[`2026-10-16_WIP_s3_donor24-launch.md`](2026-10-16_WIP_s3_donor24-launch.md) 의 Panda40 절이다.

| 대상 | 조건 | 최고 구성 | HQNR / ERGAS |
|---|---|---|---|
| 단일 Student — HQNR 기준 | 현재 선택 규칙(best-HQNR fallback) | GT-only seed R2 @18K | **.9574** / 2.1241 |
| 단일 Student — ERGAS 기준 | 전 checkpoint 중 ERGAS 최소 | 조건부 q·전체 픽셀 seed R2 @48K | .9497 / **2.0482** |
| 단일 Student — 균형 | 목표까지 상대거리 `D_inf` 최소 | GT-only seed R2 @42K | .9520 / 2.0540 |
| 출력 앙상블 (진단) | 50K Student 2개 평균 | GT-only seed R1 + R2 | .9529 / **2.0288** |
| 출력 앙상블 + Teacher (진단) | 50K Student 12개 평균 × 0.8 + donor Teacher × 0.2 | 전 run | **.9580 / 2.0351** |
| 공동 목표 (HQNR ≥ .959 & ERGAS ≤ 2.045) | — | **없음** | — |

> **수치 신뢰도 — 반드시 읽을 것**
> - 모든 수치는 **s3** 에서 쟀다. 실행은 이 캠페인의 Stage 2 run(배포본 `baseline`/`fixed` 구분과 무관한 PANDA 재구성 계열)이고, 측정은
>   **`py` 공식 evaluator**(DLPan MATLAB 프로토콜의 파이썬 재구현; `ta2.evaluation` 의 `rr_scene`/`fr_scene` 과 같은 함수)다.
>   RR = 20장 `dim_cut21`, FR = **논문 세트 `.mat` 20장(`fr_mat20`)** full512, D_s 기준은 원 PAN.
> - 재평가 도구가 기록값을 소수 5자리까지 재현한다(GT-only R2 @18K: HQNR .95738 / D_λ .02062 / D_s .02246 / ERGAS 2.1241).
>   JQM·Q8·SCC 는 진단 재평가에서 생략했다.
> - **§3–§5 의 진단은 전부 test 세트를 보고 고른 것(test-aware)이다.** 독립 holdout 검증이 아니며, 앙상블·Teacher 혼합·blur·보간 수치는
>   "기전 확인용" 이다. 그대로 논문 표나 시트의 공식 행으로 옮기지 않는다.
> - seed 는 case 당 2개다. CLAUDE.md 판정선(raw HQNR 2σ = .0031, ERGAS 0.8%)보다 작은 case 간 차이는 주장하지 않는다.
> - 서버 간 절대 HQNR 을 빼지 않는다. §2.2 의 s1 시트 값은 "어느 step 에서 최고점이 나오는가" 패턴 확인용이다.

---

## 약명 → 설정 대응 (이 문서 안에서만 쓴다)

| 약명 | 설정 | α / β / γ | loss 영역 |
|---|---|---|---|
| G | GT-only (`gt_only`) | 0 / 0 / 0 | MASK2 |
| E | Teacher 난이도 Hard 가중만 (`panda_e`) | .5 / 0 / 0 | MASK2 |
| ES | E + selective Teacher Soft (`panda_e`) | .5 / .05 / 0 | MASK2 |
| Q | ES + 조건부 q Hard 가중 (`panda_condq`) | .5 / .05 / .5 | MASK2 |
| Q_FULL | Q 와 같은 계수, 마스크 없는 전체 픽셀 + 전체 픽셀 τ_e | .5 / .05 / .5 | FULL |
| Q_WEAK | Q 의 계수 절반 | .25 / .025 / .25 | MASK2 |
| R1 / R2 | seed 261016101 / 261016102 (Student 초기화·데이터 순서·증강 stream 이 seed 별로 같다) | | |
| donor Teacher | `D24_WV3_S3_A1_a2_eps0.003_lrA1e-05_R3_M2_H50_R1_S261016101` step 7000 (sha `d5fd3d65…`), Aligner 와 Teacher 를 함께 동결 | | |
| best-HQNR ckpt | run 안에서 HQNR 최대 checkpoint(현재 선택 규칙이 실제로 고른 것) | | |
| `D_inf` | `max((.959−HQNR)₊/.959, (ERGAS−2.040)₊/2.040)` — panda40 evaluator 가 이미 계산하는 값 | | |
| signed D_s (밴드 k) | `Q(F_k, P) − Q(LMS_k, P_LP)` (32×32 블록 UQI 평균). 절댓값의 밴드 평균이 D_s 다 | | |

수식: `w_H = 1 + α·d_e + γ·d_e·r̄_q`, `L = ⟨w_H·e_S⟩ + β·⟨(1−d_e)·a_T·|S−T|⟩` (명세 §5, 구현 [`panda_stage2_condq/losses.py`](../panda_stage2_condq/losses.py)).

---

## 0. 구현 서버가 먼저 볼 요지

1. **단일 Student 로는 두 목표가 학습 궤적의 반대편 끝에 있다.** HQNR 정점은 7–24K(ERGAS 2.10–2.31), ERGAS 최저는 42–50K(HQNR .947–.952).
   D_λ 는 20K 이후 평평하고, HQNR 하락은 전부 **D_s 상승**(8K .0225 → 50K .0288) 때문이다.
2. **HQNR(D_s)은 디테일을 보수적으로 넣을수록 좋아진다.** 50K 출력의 잔차를 σ=0.7 로 흐리면 HQNR **.962**(목표 초과) / ERGAS 3.50.
   RR 에서는 **정답(GT) 자체의 D_s 가 0.070** 으로 Student(0.058)보다 나쁘다. D_s 는 GT 충실도와 같은 방향의 지표가 아니다.
3. **D_s 를 올리는 성분은 run 마다 다른, PAN 과 무관한 고주파 디테일이다.** 12 run 출력 평균에서 각 run 이 벗어난 성분은
   PAN 고주파와의 상관이 **0.000**(합의 성분은 0.515)이고, RR MSE 의 3.9% 를 차지한다. 출력 앙상블은 이 성분만 지워
   **ERGAS 와 HQNR 을 동시에** 개선한다(K=2: 2.0505→2.0285, .9497→.9527).
4. **loss 변형 E/ES/Q/Q_FULL/Q_WEAK 는 frontier 를 옮기지 못했다**(공통 회귀선 대비 잔차 ±.0015). Hard 가중은 20K 까지 gradient 방향을
   거의 바꾸지 않고(GT L1 과 cos ≥ .998), β=.05 Soft 는 gradient 의 약 1% 다.
5. **가중치 평균(soup)은 같은 초기화끼리도 실패한다**(ERGAS 2.87–3.94). 앙상블 이득을 단일 모델로 가져오려면 **distillation** 이 필요하다
   → §7.1 의 "앙상블 distillation Stage 2" 가 1순위 권장 실험이다.
6. 선택 규칙을 `D_inf` 최소로 바꾸면 34–44K 가 잡힌다(12 run 평균 .9504 / 2.056; 현재 규칙 평균 .9551 / 2.148). §7.3.

---

## 1. 캠페인 개요와 실행 상태

### 1.1 실행

| 항목 | 값 |
|---|---|
| 캠페인 | `PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1` / s3 / WV3, method revision `PANDA40-S2D-MASK2-CONDQ-v1` |
| 기동 → 종료 | 10-16 20:40 KST → 10-17 22:08 KST, `FINISHED_REGISTERED_QUEUE`, 경과 25.46 h |
| 그룹 | G → E → ES → Q → Q_FULL → Q_WEAK, 각 R1 → R2. 12/12 완료, 실패·재시도·NaN 없음 |
| 속도 | G 0.101 s/update(Teacher 없음), 그 외 0.136 s/update. run 당 G 1.6 h, 그 외 2.1 h(평가 포함) |
| Student | W112 · D(1,2,3) · P0 · BICUBIC_MS, fresh(`role="S"`), Teacher 복사 없음 |
| 학습 | AdamW lr 1e-4 · wd .01 · warmup 100 + cosine 50K → 0 · batch 48 = 12 × 4 · FP32 · TF32 off |
| calibration | MASK2 유효 픽셀 9714 패치 전체: τ_e = 0.0136397, n_e = 36,145,794, n_q = 22,381,056, K_eff = 5(병합 없음), bin 당 ≈4.47 M |
| 시트 | `WV3-ablations` 에 12 행 업로드(readback 검증 완료). 공동 목표 행 없음 → 전부 best-HQNR 대표 행 |

### 1.2 명세 대비 구현 점검

loss 수식·stop-grad·ROI8·exact ECDF·Student 전용 optimizer·동결 hash 검사는 명세 그대로다. G R1 과 E R1 의 step 1 학습 L1 이
0.0636000 으로 비트 동일해 paired 초기화·데이터 순서도 확인했다. calibration 위생도 정상이다 — ROI 안 r_q 분포가 균등하고
(평균 .510, 분위 .055/.268/.515/.756/.946), g bin 이 오를수록 q 중앙값이 0.0023 → 0.0256 으로 11 배 커져 g 조건화가 실제로 의미가 있다.
d_e 와의 상관은 r_q .25, raw q .55, g .57 이라 조건화가 edge 세기 성분을 걷어낸다.

명세와 다른 점(모두 캠페인 계획서가 정한 것):

| 항목 | 명세 | 실행 | 영향 |
|---|---|---|---|
| loss 영역 | 마스크 없는 전체 이미지 | MASK2 기본, Q_FULL 만 전체 | Aligner shift 크기가 0.82 px 이하라 MASK2 유효 영역이 **항상 61×61(90.8%)** — 사실상 고정된 3 px 테두리 제거다. Q 와 Q_FULL 차이는 seed 폭 안 |
| panda_e | α > 0, β > 0 | E 는 β = 0, ES 에서 β 추가 | 증분 분해가 더 잘다. 문제 없음 |
| 계수 | smoke 출발점 α=1, β=.1 | α=.5, β=.05, γ=.5 | §4.6 — 효과가 seed 폭보다 작은 원인 중 하나 |
| donor | 미결정 | donor24 A1 R1 의 **step 7000** | HQNR .959 기준을 넘는 checkpoint 중 ERGAS 최소. 같은 run 은 이후 D_s 붕괴로 50K 에서 HQNR .918 |

### 1.3 부수 관찰 — 증강은 회전 4종뿐이다

`panda_stage2_condq/stream.py` 는 `hflip_probability = vflip_probability = 1` 을 `random() < p` 로 해석해 **모든 microbatch 를 항상
좌우·상하로 함께 뒤집는다**(= 180° 회전). 여기에 rot90 이 더해지므로 실제 증강은 회전 4종이고 거울상은 한 번도 나오지 않는다.
원 배포 [`feeders/feeder.py`](../feeders/feeder.py) `PanFeeder.augment` 도 `if self.hflip:` 로 매번 뒤집으므로 **배포 코드를 충실히 따른 것**이며
Stage 2 버그는 아니다. 다만 KNOWN_ISSUES 에 없다. 증강 다양성이 run 간 분산(§4.4)에 관여할 가능성이 있어, 새 캠페인에서 바꾸려면
별도 축으로 다룬다.

---

## 2. 결과

### 2.1 run 별 요약 (s3, `py`)

| case | seed | best-HQNR ckpt: step · HQNR · ERGAS · D_s · D_λ | ERGAS 최소: step · HQNR · ERGAS · D_s · D_λ | 50K: HQNR · ERGAS |
|---|---|---|---|---|
| G | R1 | 20K · .9524 · 2.113 · .0239 · .0243 | 50K · .9487 · 2.052 · .0294 · .0226 | .9487 · 2.052 |
| G | R2 | 18K · .9574 · 2.124 · .0225 · .0206 | 50K · .9513 · 2.050 · .0278 · .0215 | .9513 · 2.050 |
| E | R1 | 15K · .9547 · 2.162 · .0248 · .0210 | 46K · .9478 · 2.058 · .0305 · .0223 | .9478 · 2.058 |
| E | R2 | 18K · .9552 · 2.123 · .0243 · .0209 | 50K · .9487 · 2.049 · .0300 · .0219 | .9487 · 2.049 |
| ES | R1 | 24K · .9555 · 2.104 · .0211 · .0239 | 48K · .9508 · 2.053 · .0277 · .0221 | .9509 · 2.054 |
| ES | R2 | 7K · .9566 · 2.309 · .0197 · .0242 | 48K · .9516 · 2.051 · .0279 · .0212 | .9516 · 2.052 |
| Q | R1 | 18K · .9532 · 2.157 · .0227 · .0247 | 48K · .9490 · 2.052 · .0284 · .0232 | .9490 · 2.052 |
| Q | R2 | 17K · .9568 · 2.130 · .0212 · .0225 | 50K · .9492 · 2.051 · .0269 · .0246 | .9492 · 2.051 |
| Q_FULL | R1 | 20K · .9534 · 2.115 · .0236 · .0236 | 42K · .9507 · 2.054 · .0266 · .0233 | .9506 · 2.054 |
| Q_FULL | R2 | 18K · .9548 · 2.113 · .0240 · .0217 | 48K · .9497 · 2.048 · .0270 · .0240 | .9498 · 2.048 |
| Q_WEAK | R1 | 13K · .9561 · 2.200 · .0209 · .0234 | 50K · .9467 · 2.051 · .0344 · .0196 | .9467 · 2.051 |
| Q_WEAK | R2 | 18K · .9550 · 2.130 · .0234 · .0221 | 50K · .9487 · 2.049 · .0299 · .0221 | .9487 · 2.049 |

- **HQNR ≥ .959 checkpoint 0개, ERGAS ≤ 2.045 checkpoint 0개.** 50K ERGAS 는 12 run 모두 2.048–2.058, 50K HQNR 은 .9467–.9516.
- 같은 서버 Stage 1 donor24 run 은 HQNR 최고점이 7–20K(.9569–.9591, ERGAS 2.18–2.36)이고, Aligner 를 같이 학습하므로 마지막 평가
  (B4 는 34K 에서 중단, 나머지 50K)에서 D_s 가 .032–.073 으로 무너진다(HQNR .911–.950). Stage 2 는 Aligner 를 동결해 이 붕괴가 없다(50K D_s .026–.034).

### 2.2 시트 전체에서 본 패턴 — "HQNR 달성" 은 전부 초반 step

`WV3-ablations` 에서 HQNR 이 .9599 이상인 행 다섯 개는 모두 s1 Stage 1 이고, **전부 6–8K step** 이다. 이 표는 패턴 확인용이며 서버 간 순위 비교가 아니다.

| 서버 | 실행 | step | HQNR | ERGAS |
|---|---|---:|---:|---:|
| s1 | DirectShift DF07 (abs 3 · ε .003), `DS3_WV3_S1_DF07_BASE_H50_R1` | 8K | .9608 | 2.3614 |
| s1 | panda40 S2D Aligner, A1 조건, `P40_WV3_S1_S_A1_R2` | 6K | .9612 | **3.1480** |
| s1 | donor24 A1 (abs 2 · ε .003), `D24_WV3_S1_A1_…_R1` | 7K | .9602 | 2.3764 |
| s1 | DirectShift DF05 (abs 3 · ε .001), `DS3_WV3_S1_DF05_BASE_H50_R1` | 6K | .95995 | 2.5295 |
| s1 | panda40 Direct Aligner, A5 조건, `P40_WV3_S1_D_A5_R2` | 6K | .9600 | 2.4631 |

ERGAS 가 3.15 인 checkpoint 가 HQNR .9612 를 받는다는 것 자체가 §4.1 의 "HQNR 은 덜 학습된 보수적 출력을 보상한다" 의 실례다.
ERGAS 2.045 이하는 시트 어디에도 없다(최저 2.048, 이 캠페인).

---

## 3. 학습 궤적 — 두 목표는 반대편 끝에 있다

![](assets/2026-10-18_s3_panda40_trajectory.png)
*12 run 평균. 좌: HQNR(18K 정점). 중: D_s(파랑)는 8K 최저 이후 계속 오르고 D_λ(주황)는 20K 이후 평평. 우: ERGAS 는 50K 까지 단조 감소하나 2.045(점선)에 닿지 않는다.*

| step | HQNR | D_λ | D_s | ERGAS | SAM | Q8 |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | .9429 | .0280 | .0301 | 2.713 | 3.568 | .8997 |
| 8K | .9507 | .0274 | **.0225** | 2.285 | 3.020 | .9136 |
| 18K | **.9544** | .0227 | .0234 | 2.133 | 2.864 | .9188 |
| 30K | .9518 | .0224 | .0264 | 2.073 | 2.804 | .9206 |
| 40K | .9499 | .0225 | .0282 | 2.057 | 2.785 | .9214 |
| 50K | .9494 | .0224 | .0288 | **2.052** | 2.780 | .9217 |

- 10K 이후 checkpoint 324개에서 corr(ERGAS, D_s) = −0.58, corr(ERGAS, D_λ) = +0.49. ERGAS 가 좋아질수록 D_λ 는 같이 좋아지고 D_s 는 나빠진다.
- **D_s 상승은 FR 20 scene 중 19개에서 12 run 중 11개 이상 일관된다.** scene 별 18K→50K 증가폭은 +.0032(fr13) ~ +.0088(fr16).
  특정 scene 의 이상이 아니라 전역 현상이다.
- ERGAS 개선도 RR 20 scene 전부에서 일어난다(−.04 ~ −.17). 50K ERGAS 는 상위 3 scene(rr0 3.39, rr13 2.62, rr10 2.50)이 평균을 끌어올린다 —
  이 셋을 빼면 1.91.

---

## 4. 왜 충돌하는가 — 기전

### 4.1 HQNR(D_s)은 보수적 디테일을 보상한다

| 출력 (50K, 3 run 평균) | HQNR | D_s | ERGAS |
|---|---:|---:|---:|
| 원본 | .9505 | .0274 | 2.052 |
| 잔차 Gaussian blur σ = 0.5 | .9572 | .0212 | 2.519 |
| 잔차 Gaussian blur σ = 0.7 | **.9618** | .0171 | **3.500** |

(3 run = GT-only R2, ES R1, Q R2. 잔차 = Student 출력 − bicubic MS base.)

같은 checkpoint 를 흐리게 하기만 해도 HQNR 이 목표를 넘는다. D_s 는 "HR 융합 밴드와 PAN 의 관계가 LR 에서의 관계와 같아야 한다" 는
**척도 불변 가정**을 재는데, 실제 영상은 이 가정을 지키지 않는다. RR 에서 GT 로 같은 식을 계산하면:

![](assets/2026-10-18_s3_panda40_signed_ds.png)
*좌: FR 에서 Student 의 밴드별 signed D_s — best-HQNR ckpt(파랑) 대비 50K(주황)에서 green–NIR1 이 더 음수로 간다(막대 = 12 run 평균, 수염 = 최소–최대).
우: RR 에서 정답 GT(초록)와 Student 50K(주황). GT 자체의 |D_s| = 0.070 으로 Student(0.058)보다 크다 — 정답을 완벽히 맞히면 D_s 가 오히려 나빠진다.*

| RR 에서 | \|D_s\| | 해석 |
|---|---:|---|
| GT (정답) | **0.0704** | 실제 영상의 척도 불변 위반 정도 |
| bicubic LMS (EXP) | 0.1286 | 디테일 없음 |
| Student G R2 @18K / @50K | 0.0587 / 0.0581 | GT 보다 "D_s 상 더 좋은" 출력 |
| donor Teacher | 0.0563 | |

그러므로 **GT 에 가까워지는 학습(ERGAS↓)이 D_s 를 낮추리라는 보장이 없다.** HQNR 최고점이 덜 학습된 초반에 오는 것도 이것으로 설명된다.

### 4.2 D_s 상승의 정체 — 주입량이 아니라 원 PAN 과의 상관 저하

FR 에서 UQI 를 상관 × 대비 × 밝기 항으로 분해하면(G R2), 18K → 50K 사이 변화는 대부분 **상관 항**이다.

| 밴드 | coastal | blue | green | yellow | red | red edge | NIR1 | NIR2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| signed D_s @18K | +.035 | +.019 | −.003 | −.009 | −.027 | −.011 | −.037 | +.032 |
| signed D_s @50K | +.031 | +.017 | −.014 | −.025 | −.042 | −.024 | −.051 | +.018 |
| Δ상관 항 (HR−LR) @18K | +.005 | −.006 | −.012 | −.001 | −.016 | −.006 | −.019 | +.020 |
| Δ상관 항 (HR−LR) @50K | −.018 | −.018 | −.023 | −.016 | −.029 | −.020 | −.031 | +.004 |
| σ 비 F/P (HR) @18K → @50K | .33→.34 | .53→.55 | .92→.94 | 1.20→1.21 | 1.26→1.28 | 1.11→1.11 | 1.45→1.47 | .94→.95 |

- 대비(σ 비, 즉 디테일 주입 gain)는 거의 변하지 않는다. **원 PAN 과의 블록 상관이 모든 밴드에서 떨어진다.**
- 밴드 2–6(green–NIR1)은 부호가 음수(PAN 과 덜 닮음)로 커지고, coastal·NIR2 는 양수(PAN 과 더 닮음)로 남는다. 밴드 평균 부호는 ≈0 이라
  "전역 과주입/과소주입" 이 아니라 **밴드별로 PAN 과 무관한 구조가 늘어나는 것**이다.
- 12 run 전부 같은 패턴이다(그림 좌; 원자료는 진단 JSON 의 `signed_D_s_band_best_vs_50K`).

### 4.3 정렬(warp)과의 관계 — shift 를 줄이면 D_s 는 줄지만 D_λ 로 옮겨갈 뿐

Student 는 Aligner 가 옮긴 PAN 을 입력으로 받지만 D_s 는 **원 PAN** 을 기준으로 잰다. FR 에서 Aligner shift 는 평균 0.70 px(RR 0.29 px)다.
추론 시 shift 에 배율을 곱해 봤다.

| run · step | shift × | HQNR | D_λ | D_s | ERGAS |
|---|---:|---:|---:|---:|---:|
| GT-only R2 · 18K | 1.0 | .9574 | .0206 | .0225 | 2.1241 |
| | 0.5 | .9542 | .0239 | .0225 | 2.1608 |
| | 0.0 | .9465 | .0277 | .0266 | 2.2517 |
| GT-only R2 · 50K | 1.0 | .9513 | .0215 | .0278 | 2.0499 |
| | 0.5 | .9516 | .0253 | .0236 | 2.0510 |
| | 0.0 | .9485 | .0291 | .0230 | 2.0933 |
| ES R1 · 24K | 1.0 | .9555 | .0239 | .0211 | 2.1043 |
| | 0.0 | .9417 | .0336 | .0256 | 2.1651 |
| ES R1 · 50K | 1.0 | .9509 | .0222 | .0275 | 2.0537 |
| | 0.5 | .9515 | .0273 | .0218 | 2.0570 |
| | 0.0 | .9464 | .0320 | .0223 | 2.0950 |

- **shift 0 이면 학습이 진행돼도 D_s 가 오르지 않는다**(GT-only R2: 18K .0266 → 50K .0230). 정렬된 상태에서만 오른다(.0225 → .0278).
  Student 가 정렬된 PAN 에서 점점 미세한 디테일을 가져오고, 그 디테일이 원 PAN 과 어긋나 상관을 떨어뜨린다는 해석과 맞는다.
- 그러나 shift 를 줄이면 **같은 크기만큼 D_λ 가 나빠진다**(50K × 0.5: D_s −.0042, D_λ +.0038 → HQNR .9513 → .9516). 정렬은
  D_λ 와 D_s 사이에서 오차를 옮길 뿐이고, 추론 shift 조정은 균형 수단이 아니다.

### 4.4 run 별 분산 — 두 지표를 함께 나쁘게 만드는 성분

12 run 의 50K 출력으로 분해했다(`tools/panda40_tradeoff_diag.py variance`).

| 지표 | 값 | 의미 |
|---|---:|---|
| RR MSE: 단일 run 평균 / 12-앙상블 / run 간 분산 (DN²) | 732.9 / 704.6 / 28.3 | 분산이 단일 run MSE 의 **3.9%** |
| FR 고주파(σ=1 Gaussian 차) 에너지 비중: run 편차 / 합의 잔차 | 0.351 / 0.203 | run 편차는 고주파에 몰려 있다 |
| FR 고주파의 PAN 고주파 상관: run 편차 / 합의 잔차 | **−0.000** / 0.515 | run 편차는 **PAN 과 무관** |
| FR 에너지 비: run 편차 / 합의 잔차 | 0.105 | 작지만 D_s 를 좌우한다 |

각 run 의 출력 = PAN 과 상관된 합의 성분 + **run 마다 다른 PAN 무관 고주파**다. 후자는 RR 에서는 오차(ERGAS↑), FR 에서는 PAN 상관
저하(D_s↑)로 나타난다. 그래서 출력 앙상블은 두 지표를 **동시에** 개선한다(§5.2). seed 가 같고 loss 만 다른 6 run 끼리 평균해도 이득이
나므로(ERGAS 2.017), 이 분산은 초기화 차이보다 학습 경로의 혼돈에서 온다.

### 4.5 단일 Student 의 ERGAS 하한 ≈ 2.05

12 run 의 50K ERGAS 가 2.048–2.058 로 좁다(case·seed 무관). 46K→50K 개선이 −0.0005 에 그쳐 cosine 끝에서 수렴했다.
ERGAS 목표만 놓고 봐도 이 Student 로는 경계선이다. 이전 기록상 2.040 아래는 7.17 M 논문 재구성본(`s1_A1`, 2.0351)뿐이다.

### 4.6 loss 변형이 frontier 를 옮기지 못한 이유

- 10K 이후 checkpoint 로 HQNR ~ ERGAS 공통 회귀선을 그으면, case 별 평균 잔차가 ES +.0015, G +.0002, Q .0000, Q_WEAK −.0004,
  Q_FULL −.0006, E −.0007 로 모두 판정선 .0031 안이다.
- train 48 패치 gradient probe(E R1·G R1 checkpoint):

| checkpoint | Student 학습 오차 / Teacher | a_T 평균 · Teacher 우위 픽셀 | β=.05 Soft ∥∇∥ / Hard ∥∇∥ | cos(Soft, Hard) | cos(E Hard, GT L1) | cos(Q Hard, GT L1) |
|---|---|---|---:|---:|---:|---:|
| E R1 @1K | .0246 / .0187 | .214 · 71% | 1.5% | +.76 | .9993 | .9985 |
| E R1 @5K | .0192 / .0187 | .102 · 54% | 1.0% | +.85 | .9997 | .9992 |
| G R1 @10K | .0181 / .0187 | .067 · 43% | 0.6% | +.40 | .9995 | .9991 |
| G R1 @20K | .0174 / .0187 | .057 · 38% | 0.7% | +.28 | .9990 | .9982 |
| G R1 @50K | .0168 / .0187 | .049 · 34% | 4.2% | **−.28** | .9688 | .9545 |

  - Hard 가중(E·Q)은 20K 까지 GT L1 gradient 와 방향이 사실상 같다. w_H 범위가 E 1.00–1.49, Q 1.00–1.97 이고, Adam 은 전체 배율에
    둔감해 실효 변화는 픽셀 간 상대 가중 ±15%(E) / −19%~+33%(Q)뿐이다. paired 학습 L1 비(E/G)도 1.0014 였다.
  - Student 의 학습 오차가 5–10K 사이에 Teacher(step 7000)보다 작아진다. 이후 selective Soft 는 gate 가 대부분 닫히고, 후반에는 GT 와
    **반대 방향**(cos −.28)으로 당긴다.
  - 정리하면, 현재 설정의 Teacher 신호는 너무 약하고 너무 일찍 꺼진다. 또 a_T gate 는 Teacher 가 GT 에 더 가까운 픽셀로만 영향을 제한하므로,
    Teacher 의 장점인 낮은 D_s(보수적이고 PAN 과 상관된 출력)를 전달하는 경로가 구조적으로 막혀 있다.

---

## 5. 균형 방법 실측 (test-aware 진단)

![](assets/2026-10-18_s3_panda40_frontier.png)
*점·회색선: 12 run 의 checkpoint 궤적(색 = step). 주황: 50K Student K 개 출력 평균(무작위 부분집합 평균). 초록: 12-앙상블에 donor Teacher 출력을
w 만큼 섞은 것. 빗금: 공동 목표 영역. 단일 run 궤적은 왼쪽 아래(ERGAS 2.05, HQNR .95)에서 오른쪽 위로 휘어 올라갈 뿐이고, 앙상블만 곡선 전체를 왼쪽 위로 옮긴다.*

### 5.1 한 run 안에서의 교환 (단일 모델 비용)

| 방법 (GT-only R2 기준; 출발점 50K = .9513 / 2.0499) | HQNR | ERGAS | ΔHQNR / ΔERGAS | 추론 비용 |
|---|---:|---:|---:|---|
| early stop 18K (현재 선택) | .9574 | 2.1241 | 0.08 | 1× |
| early stop 30K | .9543 | 2.0685 | 0.16 | 1× |
| 가중치 보간 18K:50K = .5:.5 | .9547 | 2.0782 | 0.12 | 1× |
| 가중치 보간 .25:.75 | .9531 | 2.0602 | 0.17 | 1× |
| 출력 보간 18K/50K = .5/.5 | .9560 | 2.0669 | 0.28 | 2× |
| + donor Teacher 출력 15% | .9546 | 2.0563 | 0.52 | 2× |
| + donor Teacher 출력 30% | .9572 | 2.0782 | 0.21 | 2× |

ES R1(24K/50K)·Q R2(17K/50K)에서도 순서가 같다 — Teacher 15% 혼합의 교환비가 0.65(ES R1)·1.1(Q R2)로 가장 좋고,
같은 run 안의 가중치 보간은 궤적과 거의 같은 선 위에 놓인다. 어느 방법도 출발점의 ERGAS 2.05 보다 왼쪽으로 가지 못하므로 공동 목표에는 닿지 않는다.

### 5.2 run 간 출력 앙상블 — frontier 자체가 움직인다

| 구성 (출력 평균) | HQNR | D_λ | D_s | ERGAS |
|---|---:|---:|---:|---:|
| 단일 50K (5 run 평균) | .9497 | .0221 | .0288 | 2.0505 |
| K=2 (무작위 7쌍 평균) | .9527 | .0218 | .0260 | 2.0285 |
| K=4 (무작위 8조 평균) | .9547 | .0209 | .0249 | 2.0182 |
| K=8 | .9553 | .0206 | .0246 | 2.0125 |
| K=12 | .9556 | .0205 | .0244 | 2.0103 |
| GT-only R1+R2 | .9529 | .0212 | .0264 | 2.0288 |
| ES R1+R2 | .9544 | .0206 | .0255 | 2.0294 |
| Q R1+R2 | .9527 | .0229 | .0249 | 2.0276 |
| seed 같은 6 run (R1) | .9544 | .0207 | .0255 | 2.0174 |
| 12-앙상블 @best-HQNR ckpt | .9579 | .0214 | .0212 | 2.0770 |
| 12 @50K + 12 @best-HQNR (24개) | .9574 | .0208 | .0223 | 2.0354 |
| 12 @50K + Teacher 10% | .9570 | .0205 | .0230 | 2.0195 |
| **12 @50K + Teacher 20%** | **.9580** | .0205 | .0219 | **2.0351** |
| 12 @50K + Teacher 30% | .9589 | .0207 | .0209 | 2.0570 |
| 12 @50K + Teacher 40% | .9595 | .0209 | .0201 | 2.0850 |
| 2 (GT-only R1+R2) + Teacher 10% | .9548 | .0210 | .0247 | 2.0354 |
| 2 (GT-only R1+R2) + Teacher 20% | .9563 | .0209 | .0233 | 2.0485 |

- K=2 에서 이득의 절반 이상이 나오고 K=4 이후 체감한다.
- ERGAS ≈ 2.05 지점의 HQNR: 단일 궤적 .9513 → 2-앙상블 + Teacher 20% .9563 → 12-앙상블 + Teacher 30% .9589.
  ERGAS ≈ 2.035 에서는 12-앙상블 + Teacher 20% 가 .9580 으로 **공동 목표에 HQNR .001 차이**까지 온다.
- donor Teacher 는 ERGAS 2.36 의 약한 모델이지만 D_s 가 낮아(.0174) 소량 혼합이 D_s 를 효율적으로 낮춘다.

### 5.3 가중치 평균(soup)은 실패

| 구성 (가중치 평균) | HQNR | ERGAS |
|---|---:|---:|
| GT-only R1 + E R1 @50K (같은 초기화·데이터 순서) | .9432 | 2.8694 |
| seed R1 6 run @50K | .9259 | 3.7116 |
| seed R2 6 run @50K | .9284 | 3.9410 |
| GT-only R1 + R2 @50K (다른 초기화) | .9188 | 6.5903 |
| seed R2 6 run @50K + Teacher 출력 30% | .9397 | 3.2637 |

초기화와 데이터 순서가 같아도 loss 가 다르면 서로 다른 basin 으로 간다(선형 연결성 없음). **앙상블 이득은 출력 공간에서만 얻어진다.**
같은 run 의 18K↔50K 보간만 성립한다(§5.1).

### 5.4 선택 규칙 — `D_inf` 최소

| case · seed | 현재 규칙: step · HQNR · ERGAS | `D_inf` 최소: step · HQNR · ERGAS |
|---|---|---|
| G R1 | 20K · .9524 · 2.113 | 36K · .9502 · 2.059 |
| G R2 | 18K · .9574 · 2.124 | 42K · .9520 · 2.054 |
| E R1 | 15K · .9547 · 2.162 | 38K · .9490 · 2.060 |
| E R2 | 18K · .9552 · 2.123 | 38K · .9501 · 2.057 |
| ES R1 | 24K · .9555 · 2.104 | 38K · .9520 · 2.057 |
| ES R2 | 7K · .9566 · 2.309 | 44K · .9518 · 2.052 |
| Q R1 | 18K · .9532 · 2.157 | 38K · .9504 · 2.054 |
| Q R2 | 17K · .9568 · 2.130 | 42K · .9498 · 2.055 |
| Q_FULL R1 | 20K · .9534 · 2.115 | 38K · .9512 · 2.057 |
| Q_FULL R2 | 18K · .9548 · 2.113 | 38K · .9507 · 2.055 |
| Q_WEAK R1 | 13K · .9561 · 2.200 | 38K · .9484 · 2.054 |
| Q_WEAK R2 | 18K · .9550 · 2.130 | 34K · .9495 · 2.060 |
| **평균** | .9551 · 2.148 (`D_inf` .053) | .9504 · 2.056 (`D_inf` .009) |

현재 규칙은 "HQNR ≥ .959 후보 안에서 ERGAS 최소, 없으면 best-HQNR" 이라 기준 미달 run 에서는 사실상 HQNR 최대 선택이 되고,
ES R2 처럼 7K(ERGAS 2.31)를 고르기도 한다. 시트 대표 행이 "HQNR 은 그럭저럭, ERGAS 는 크게 미달" 로 보이는 이유의 절반은 이 규칙이다.

---

## 6. 배제한 가설 / 실패 사례

| 가설 | 검증 | 결과 |
|---|---|---|
| 특정 FR scene 이 D_s 상승을 만든다 | scene 별 18K→50K Δ D_s | 20 scene 중 19개에서 일관 상승 — 전역 현상 |
| D_s 상승 = 디테일 과주입(gain) | UQI 대비 항 · σ 비 분해 | 대비는 거의 불변, 상관 항이 원인 |
| 추론 shift 를 줄이면 HQNR 이 오른다 | shift × 0.5 / 0 | D_s↓ 만큼 D_λ↑, HQNR 불변(50K) 또는 하락(18K) |
| 잔차를 흐리면 균형이 잡힌다 | blur σ .5 / .7 | HQNR .957–.962 이지만 ERGAS 2.52–3.50 — 지표 편향 확인용일 뿐 |
| Teacher 난이도 Hard 가중(E)이 frontier 를 옮긴다 | 대응 L1 비 · gradient cos · 회귀 잔차 | L1 비 1.0014, cos ≥ .999, 잔차 −.0007 |
| selective Soft(ES, β=.05)가 Teacher 의 낮은 D_s 를 옮긴다 | gradient 비 · 잔차 | gradient 약 1%, 5–10K 이후 gate 대부분 닫힘, 잔차 +.0015(판정선 안) |
| 조건부 q Hard(Q, Q_WEAK) | 잔차 | .0000 / −.0004 |
| MASK2 ↔ 전체 픽셀(Q ↔ Q_FULL) | 잔차 · 마스크 통계 | 잔차 −.0006; 마스크가 고정 3 px 테두리라 실질 차이 없음 |
| 같은 초기화 run 끼리 가중치 평균 | soup 6 종 | 전부 붕괴(ERGAS 2.87–6.59) |
| seed 를 늘리면 ERGAS ≤ 2.045 가 나온다 | 50K ERGAS 분포 | 12 run 2.048–2.058 — 단일 모델 하한 |

---

## 7. 구현 서버 권장안

### 7.1 1순위 — 앙상블 distillation Stage 2

**가설.** 출력 앙상블의 이득(§5.2)은 run 마다 다른 PAN 무관 고주파를 지운 데서 온다(§4.4). 이 합의 함수는 평균이므로 단일 Student 가
학습할 수 있는 더 매끄러운 target 이다. Student 를 앙상블 출력에 맞추면 추론 비용 1× 로 이득의 일부를 가져올 수 있다.

**확인할 것.** 단일 Student 의 50K(또는 `D_inf` 선택) 지점이 GT-only 단일 run(.949–.951 / 2.050) 대비 ERGAS 와 HQNR 을 **함께**
개선하는가. 상한 참고치는 같은 구성의 앙상블 자체(K=4: .9547 / 2.018; +Teacher 20% 는 별도 측정).

| 항목 | 권장 |
|---|---|
| namespace | 새 패키지(예: `panda_stage2_ensd/`). `panda_stage2_condq/`·`panda40/` 는 수정하지 않는다(실행 중 code-pin 대상) |
| 동결 모듈 | donor Aligner 1개(모든 member 와 Student 공유) + member Student K 개 + (선택) donor Teacher. 전부 `eval()`·`requires_grad_(False)` |
| target | `T_ens = (1−w)·mean_k S_k(P̃, M, L_A) + w·T_donor(P̃, M, L_A)` — 모두 같은 warp PAN `P̃`, no-grad |
| loss | `L = ⟨e_S⟩ + β·⟨\|S − T_ens\|⟩` (밴드 평균 L1, 영역은 member 와 같은 정책). **a_T gate 없음** |
| 격자 | β ∈ {0.5, 1, 2}, w ∈ {0, 0.2}. 대조군 = 같은 seed 의 GT-only(β=0). 순수 distillation(GT 항 없음)은 2순위 |
| member | K = 4 권장(K=2 에서 이득 절반, K=4 이후 체감). GT-only 가 가장 싸다(s3 RTX 5090 기준 run 당 1.6 h) |
| seed | Student seed 는 member seed 와 겹치지 않게(member 를 외운 효과와 구분). paired 2 seed 이상 |
| 학습 | 현 계약 유지: AdamW 1e-4 · wd .01 · warmup 100 + cosine 50K · 12×4 · FP32 · TF32 off |
| 증강 | target 은 **증강된 입력으로 online 계산**한다. 앙상블은 회전 등변이 아니므로 비증강 target 을 회전시켜 쓰면 다른 target 이 된다 |
| 비용 | member forward K 회(no-grad)가 step 마다 추가된다. 측정 전 추정 0.15–0.25 s/update — preflight 로 실측 |
| 데이터 | RR train 만. **등록 catalog 에 FR 학습 split 이 없다**(`train/val/rr/fr`, fr = test mat20). FR 입력 distillation 은 새 데이터가 필요하므로 이번에는 하지 않는다 |
| checkpoint | 2K 마다 Student 저장(사후 보간·EMA·앙상블 분석용). 현재 panda40 은 개선 시점과 50K 만 저장해 사후 분석이 제한됐다 |
| 선택 | 공식 규칙과 함께 `D_inf` 최소 checkpoint 를 별도 기록(§7.3) |
| 로그 | 기존 loss 항목 + `distill_raw = ⟨\|S−T_ens\|⟩`, Hard/Distill gradient norm·cos(100 update 마다), 평가마다 밴드별 signed D_s |
| 판정 | 같은 서버·같은 seed 대응 차이로 본다. HQNR .0031, ERGAS 0.8% 미만은 seed 3개 이상 같은 방향일 때만 주장한다 |

참조 pseudocode(구현 형태는 서버가 정한다):

```python
with torch.no_grad():
    u = aligner(pan, native_lms)                  # frozen donor Aligner, one warp
    p_al = warp_pan(pan, u)
    members = [m(p_al, ms, native_lms)['y'] for m in frozen_students]   # K frozen Stage 2 Students
    t_ens = torch.stack(members).mean(0)
    if w > 0:
        t_ens = (1 - w) * t_ens + w * donor_teacher(p_al, ms, native_lms)['y']
s = student(p_al, ms, native_lms)['y']           # fresh, trainable
e_gt = (s - gt).abs().mean(1, keepdim=True)
e_kd = (s - t_ens).abs().mean(1, keepdim=True)
loss = reduce(e_gt) + beta * reduce(e_kd)        # reduce = same MASK2 / full policy as members
```

**member 확보.** 같은 donor Aligner 위에서 학습된 Student 여야 한다.

- 구현 서버가 **자기 donor 로 GT-only member K 개를 새로 학습**하는 것이 가장 깔끔하다(서버 로컬 donor 규칙 유지).
- s3 의 12 checkpoint 를 쓰려면 s3 donor checkpoint(sha `d5fd3d65e0a4ba6da9452042a46cc261306ec4efe4aee433dfd48abe17351222`)와
  member `final_0050000.pt`(각 32.7 MB, trainer 전체 payload; Student 만 추출 가능)를 함께 옮겨야 한다. 위치는
  `work_dir/panda40/PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1/s3/WV3/runs/*/attempt_01/models/`. 이 경우 결과는 s3 donor 기준이다.

**명세와의 관계.** 명세 §0.1 은 "Stage 1 Teacher 하나" 를, §5.2 는 "Teacher 가 앞설 때만 켜지는 selective Soft" 를 규정한다.
앙상블 Teacher 와 gate 없는 distillation 은 둘 다 **명세 개정 사항**이므로 새 `method_version`·`loss_schema` 로 정의한다.

**위험.** distillation 은 보통 앙상블 이득을 전부 회수하지 못한다. Student 용량(W112)이 run 간 분산의 원인이라면 회수율이 낮을 수 있다.
그때는 §7.4 의 용량 축과 묶어 본다.

### 7.2 함께 돌릴 저비용 arm — 가중치 EMA

같은 run 안의 가중치 보간은 성립했다(§5.1). Student 가중치 EMA(decay 0.999 / 0.9999)를 유지하고 raw 와 EMA 를 같은 step 에서 둘 다
평가하면, 추가 학습 비용 없이 "시간 방향 평균" 의 효과를 잰다. run 간 분산(§4.4)을 지우는 것은 아니므로 기대치는 앙상블보다 작다.

### 7.3 선택 규칙 — `D_inf` 최소를 함께 기록

현재 규칙은 목표 미달 run 에서 best-HQNR 로 대체되어 초반 checkpoint 를 고른다(§5.4). 공식 규칙은 바꾸지 말고, `D_inf` 최소 checkpoint 를
`best_balanced` 로 **추가 저장·기록**하기를 권한다. panda40 evaluator 가 `D_inf` 를 이미 계산한다
([`panda40/evaluation.py`](../panda40/evaluation.py) `target_gap`). test 세트로 고르는 선택 편향은 기존 규칙과 같다.

### 7.4 별도 축 — ERGAS 하한

단일 W112·D123·P0 Student 의 ERGAS 하한 ≈ 2.05 는 loss·선택과 무관하다(§4.5). 2.040 아래를 단일 모델로 원하면 용량(폭·깊이)이나
학습 길이(50K 초과) 축이 필요하다. 이것은 균형 문제와 독립이며, 7.1 과 섞지 않고 대조한다.

### 7.5 하지 말 것

- Hard 가중 계수만 바꾼 변형 추가(E·Q 계열): gradient 방향이 거의 같아 seed 폭 안에 묻힌다.
- selective Soft 의 β 만 키우기: gate 가 5–10K 이후 닫혀 효과가 제한된다. Teacher 신호가 필요하면 gate 없는 형태로 바꾼다.
- 추론 shift 배율 조정, 잔차 blur: 오차를 D_λ↔D_s 로 옮기거나 ERGAS 를 무너뜨린다.
- run 간 가중치 평균(soup): 같은 초기화에서도 실패한다.

### 7.6 진단 도구 사용

[`tools/panda40_tradeoff_diag.py`](../tools/panda40_tradeoff_diag.py) — 읽기 전용, work_dir 에 쓰지 않는다. 공식 evaluator 함수를 그대로 쓴다.

```bash
PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python   # 서버별 경로
export PANCRAFTER_DLPAN=/path/to/DLPan-Toolbox
R=work_dir/panda40/PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1/s3/WV3      # --campaign: donor 를 적은 campaign_state.json 위치
M=$R/runs/P40_WV3_S3_G_R2_S261016102/attempt_01/models
$PY tools/panda40_tradeoff_diag.py eval --campaign $R --ckpt $M/step_0018000.pt                 # 지표 + 밴드별 signed D_s
$PY tools/panda40_tradeoff_diag.py eval --campaign $R --ckpt $M/final_0050000.pt --shift-scale 0.5
$PY tools/panda40_tradeoff_diag.py ensemble --campaign $R --ckpt A.pt B.pt --teacher-weight 0.2    # 출력 앙상블(+ donor Teacher)
$PY tools/panda40_tradeoff_diag.py variance --campaign $R --ckpt A.pt B.pt C.pt D.pt              # §4.4 분해
$PY tools/panda40_tradeoff_diag.py gt-ds --campaign $R                                            # RR GT 의 D_s
```

새 캠페인의 Student checkpoint 형식이 `panda_stage2_condq` trainer payload(`trainer.stage` 의 `student.*`)와 다르면 `Context.student()` 만 고친다.
s3 RTX 5090 기준 평가 한 번(RR20 + FR20)은 약 7 s 다.

---

## 8. 남은 판단

| 질문 | 필요한 결정 |
|---|---|
| 앙상블 Teacher·gate 없는 distillation 의 명세 개정 | 승인 여부(§7.1) |
| FR 입력 distillation | 등록 catalog 에 FR 학습 split 이 없다. 새 FR 학습 데이터를 만들지 결정해야 한다(test mat20 은 사용 불가) |
| 증강(회전 4종, 거울상 없음) | 배포 코드 그대로다. 바꾸려면 별도 축으로 다루고 KNOWN_ISSUES 기재 여부를 정한다(§1.3) |
| §4.4 분산 기전의 일반성 | s3 한 서버·한 donor 결과다. 다른 서버 run 으로 `variance` 를 재측정하면 확인된다 |

## 9. 산출물

| 파일 | 내용 |
|---|---|
| [`assets/2026-10-18_s3_panda40_frontier.png`](assets/2026-10-18_s3_panda40_frontier.png) | §5 그림 |
| [`assets/2026-10-18_s3_panda40_trajectory.png`](assets/2026-10-18_s3_panda40_trajectory.png) | §3 그림 |
| [`assets/2026-10-18_s3_panda40_signed_ds.png`](assets/2026-10-18_s3_panda40_signed_ds.png) | §4.1–4.2 그림 |
| [`assets/2026-10-18_s3_panda40_diag.json`](assets/2026-10-18_s3_panda40_diag.json) | 전 checkpoint 평가 622건(donor24 포함), 밴드별 signed D_s, shift·보간·Teacher·blur·앙상블·soup·분산·gradient probe 원자료 |
| [`../tools/panda40_tradeoff_diag.py`](../tools/panda40_tradeoff_diag.py) | 재현 도구(§7.6) |

원 checkpoint·평가 JSON 은 s3 로컬 `work_dir/panda40/PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1/s3/WV3/` 와
`work_dir/donor24/PANDA_WV3_DONOR_S13_24H_20261016_v1/s3/WV3/` 에 있다(git 밖).
