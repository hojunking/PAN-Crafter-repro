# Shift-robust conditioning 30h 결과 — jitter 는 후반 붕괴를 없애는 정규화다, best HQNR 은 동급 (2026-09-07)

계획: [s1_w168_d123_shift_robust_alignment_30h_plan.md](../research_log/s1_w168_d123_shift_robust_alignment_30h_plan.md) ·
검토·구현: [2026-09-06_shift-robust-plan-review.md](../research_log/2026-09-06_shift-robust-plan-review.md)
**6벌 완주(무장애): J1·J3·J4·J2·G1 + gate 가 연 J4 seed 1234.** 2026-09-06 03:25 → 09-07 03:40 (24.3h).
판정: **best checkpoint HQNR(12-19, 장면별 평균) → fSCC.** anchor `S1_T05_W168_D123_DUAL` = 0.9571@ep100 · fSCC 0.8785 · D_λ 0.0231 · D_s 0.0202.

---

## 1. 결과

| run | 세팅 | **best HQNR** | ep | fSCC | D_λ | D_s | plateau(≥100) | last50 | final(245) | 시간 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor | 원 bicubic, perturbation 없음 | 0.9571 | 100 | 0.8785 | 0.0231 | 0.0202 | 0.9524 | 0.9497 | 0.9496 | 3.46h |
| J1 | ±0.5px 무작위 jitter, MS·PAN 두 mode | 0.9565 | 125 | 0.8908 | 0.0222 | 0.0218 | 0.9549 | 0.9549 | 0.9548 | 3.47h |
| J2 | jitter MS mode 만 | 0.9574 | 155 | 0.8863 | 0.0200 | 0.0231 | **0.9560** | **0.9560** | **0.9560** | 3.47h |
| J3 | Gaussian blur σ1.225(MSE 매칭), 두 mode | 0.9491 | 100 | 0.8800 | 0.0244 | 0.0272 | 0.9390 | 0.9342 | 0.9338 | 3.47h |
| J4 | clean+jitter branch, consistency λ0.1 | 0.9577 | 120 | **0.8963** | 0.0227 | 0.0201 | 0.9558 | 0.9555 | 0.9554 | 4.93h |
| J4 seed1234 | (winner 반복) | **0.9580** | 160 | 0.8858 | 0.0202 | 0.0223 | 0.9556 | 0.9562 | 0.9560 | 4.94h |
| G1 | first-conv PAN 기여 global correlator | 0.9527 | 70 | 0.8458 | 0.0201 | 0.0278 | 0.9458 | 0.9447 | 0.9445 | 3.87h |

### 1.1 best HQNR 로는 전부 anchor 와 동급이다
J1 −0.0006 · J2 +0.0003 · J4 +0.0006 · J4-seed +0.0009. 같은 J4 두 seed 의 차이가 0.0003(best)·0.0002(plateau)·0.0006(final)이므로
이 차이들은 노이즈 안이다. 장면별 개선 수도 J4 5/8, J1 3/8, J2 1/8 로 방향이 없다.

### 1.2 그러나 후반 붕괴가 사라졌다 — 계획 §22 J1 성공 조건 B 충족
anchor 는 ep100 0.9571 → ep245 **0.9496 (−0.0075)** 로 내려간다. jitter 3벌은 ep125 이후 **0.955–0.956 에 평평**하다:

| | plateau − anchor | final − anchor | 후반 D_s |
|---|---:|---:|---|
| J1 | +0.0025 | +0.0052 | 0.0218→0.0243 평탄 |
| J2 | +0.0036 | +0.0064 | 0.0235→0.0250 평탄 |
| J4 | +0.0034 | +0.0058 | 0.0210→0.0241 평탄 |
| J4 seed | +0.0032 | +0.0064 | 재현 |

두 seed 에서 재현되고, 크기(+0.005~0.006)는 실측 노이즈 상한(0.0027)의 2배다. GA 캠페인의 C2(cache Δ jitter, W152, +0.005 final) 가
**noisy cache 없이 통제된 무작위 jitter 만으로 재현**됐다 — H1 성립.

### 1.3 H3: smoothing 이 아니다 (J3 대조)
J3(위치 이동 없는 등에너지 blur)는 best 0.9491 · final 0.9338 · RR ERGAS 3.09 로 무너진다. 사전 실측(bicubic warp 의 gradient-energy 비 1.0056)과
합쳐 "jitter 의 이득 = smoothing" 해석은 기각. 이득은 **위치 perturbation 에 대한 조건 입력 강건성**이다 (계획 §23 Case A).

### 1.4 H2: MS-only(J2) ≥ 두 mode(J1)
J2 가 best +0.0009 · plateau +0.0011 · final +0.0012 — seed 노이즈의 2~3배라 약한 신호. fSCC 는 J1 이 +0.005 높다.
"PAN mode 에 jitter 를 주면 PAN 입력과 충돌" 쪽으로 기운다. 미세조정에서 J2 를 기준으로 삼는다.

### 1.5 H4: consistency 는 작동하지 않았다
J4 의 λ·L_cons/L_MS EMA = **0.006 (<0.01, 계획 §9.4 "사실상 무효")**. J4 의 우위(J1 대비 best +0.0012, fSCC +0.0055)는 consistency 항이 아니라
clean+jitter 두 branch(실효 MS batch 2배, 표본의 절반이 clean)에서 왔을 가능성이 크다. → 미세조정 §4.

### 1.6 추론 jitter 강건성 (§22 "shift-robustness" 조건)
추론 시 조건 입력에 ±0.5 px 를 넣어도 J1 0.9565→0.9565~0.9573, J2 0.9574→0.9570~0.9578, J4 0.9577→0.9574~0.9588 — 하락 없음.
jitter=0 에서도 유지. 두 seed 재현. J1 > J3. → 네 조건 전부 충족.

### 1.7 G1 실패 — PAN 쪽을 흔들면 안 된다
- correlator 가 끝까지 날카로워지지 않았다: conf 0.04→0.08, boundary 확률 0.56→0.50(거의 균등 posterior), |Δ̂+ε_g| 0.39→0.31.
  gate(conf/0.3)≈0.2 라 보정이 거의 적용되지 않았고, β sweep 도 평평(0.9532/0.9530/0.9527), **wrong-sign 이 오히려 0.9542** — 보정은 노이즈.
- 그런데 β=0 인 G1 자체가 anchor 보다 −0.004 낮다. 학습 중 PAN feature 에 ±1 px synthetic shift(p 0.75)를 준 것이 GT 정렬을 깨뜨렸다:
  **PAN 은 공간 기준이라 흔들면 D_s·fSCC 가 곧바로 나빠진다** (fSCC 0.846). J 계열이 MS 조건만 흔들어 성공한 것과 정확히 대칭.
- §22 G1 기준 4항목 전부 불충족.

### 1.8 §12 local diagnostic — 종료
winner J4 에 audit 전역 shift 를 first-conv PAN 기여에 적용(L1)하면 HQNR 0.9577→**0.9277**(D_s 0.020→0.053), fSCC 0.896→0.801.
gated local(L2) 0.9218, ungated(L3) 0.8537, wrong-sign(L4) 0.9299(정방향보다 높음). §12.6 gate 5항목 중 4개 불충족 → **local alignment 종료.**
정렬 없이 학습된 모델에 추론 시 PAN 을 옮기는 것은 모두 해롭다 — GA anchor sweep·UVS teacher(RR +6.2%)·여기까지 세 번째 확인.

### 1.9 대조 run (W152, 원 bicubic + cache jitter) — 이득은 커널과 독립
| W152 | best | plateau | last50 | final |
|---|---:|---:|---:|---:|
| anchor(bicubic) | 0.9546 | 0.9516 | 0.9505 | 0.9502 |
| C2 α1(interp23tap+jitter) | 0.9553 | 0.9528 | 0.9549 | 0.9552 |
| **CTRL(bicubic+jitter)** | 0.9539 | 0.9523 | 0.9535 | 0.9534 |

원 커널 위에서도 final +0.0032·last50 +0.0030 — 방향 동일, 크기는 interp23tap 조합보다 작다. GA 문서의 "C2 이득은 커널 교체에 의존하는가" 에 대한 답: **아니다(독립), 단 크기의 일부는 커널 교체분이었다.**

---

## 2. 결론
1. **Shift-robust conditioning(MS 조건 입력의 ±0.5 px 무작위 jitter)은 후반 D_s 드리프트를 없앤다** — final HQNR +0.005~0.006, fSCC +0.01, 두 seed·두 backbone(W152/W168)·두 커널에서 재현. 이전 14벌 격자를 괴롭힌 "ep80~130 정점 뒤 하락"의 첫 해법이다.
2. **best-checkpoint HQNR 은 올리지 못한다** (동급). 판정 규칙상 "동급" 이고, 실용적 의미는 "언제 멈춰도 best 근처" — checkpoint 선택 편향에 덜 의존하는 모델.
3. PAN 을 옮기는 모든 접근(G1, local, 추론 시 정렬)은 실패. **정렬은 MS 조건 쪽 강건성으로만 다룬다.**
4. 다음: §4 미세조정(진행 중), 그 뒤 계획 §23 Case A — W96 student·W168 teacher 에 jitter 를 기본값으로 얹고 KD 재개.

## 3. 운영
6벌 24.3h 무장애. J4 는 4.9h(1.42×). 시트 SR 범주 구분행·Notes 자동. 진단 CSV: `results/sr_diag.csv`, `results/local_diag.csv`.

## 4. 미세조정 (2026-09-07 09:2x 기동, `config/queues/s1_shift_robust_refine.txt`, ≈16h)
| run | 무엇을 가르나 |
|---|---|
| `SR_J4_CJCONS_R050_L100` | consistency λ 1.0 (비율 0.006→≈0.06) — consistency 가 실제로 작동하면 J4 를 더 올리나 |
| `SR_J2_C2RAND_MSONLY_R050_P050` | jitter 확률 0.5 — J4 의 우위가 clean 노출 때문인지(consistency 없이) |
| `SR_J2_C2RAND_MSONLY_R075` | 반경 ±0.75 (§14: plateau D_s 0.024 가 anchor best 0.020 보다 남음) |
| `SR_J2_C2RAND_MSONLY_R025` | 반경 ±0.25 (spectral 쪽) |
