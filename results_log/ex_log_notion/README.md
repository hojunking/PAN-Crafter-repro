# 실험 보고서 (case 단위)

**KD 캠페인(2026-08-31)부터** case 하나가 끝날 때마다 여기에 보고서 하나를 쓴다.
그 이전 실험은 [`../README.md`](../README.md) 의 캠페인 단위 문서를 본다.

## 양식

| | |
|---|---|
| **1. 어떤 실험인가** | 무엇을 왜, 무엇과 무엇만 다른가 (단일 변인), 이 실험이 답할 질문 |
| **2. 어떤 결과가 나왔나** | 표와 그림. 해석 없이 수치만 |
| **3. 어떤 분석이 나왔나** | 결과가 뜻하는 것, 다음 case 에 미치는 영향, 배제한 가설 |
| **4. 참고 지표** | 판정에 쓰지 않는 것들. 기록용 |

## 판정 규약

**판정 지표는 HQNR(공식 FR 12-19), 보조는 SCC. 그게 전부다.**

- 결과표는 **HQNR → SCC 순으로 맨 위**. "동급 / 우위 / 열위" 판정 문장은 **HQNR 로만** 쓴다.
- **판정선 = HQNR 시드 2σ 1.18%.** 이보다 작으면 "동급".
- HQNR 이 동급이고 SCC 도 포화면 **"구분되지 않는다"로 끝낸다.**
  ERGAS 로 내려가 tie-break 하지 않는다.
- HQNR 해석이 필요하면 **D_λ·D_s 분해**로 설명한다 (`HQNR = (1−D_λ)(1−D_s)`).
  ERGAS·SAM 으로 설명하지 않는다.
- **ERGAS·SAM·PSNR·Q2n 은 §4 "참고 지표 — 판정에 쓰지 않음"** 에 몰아 둔다.
  기록·시트 업로드는 계속하되 결론의 근거로 앞세우지 않는다.
- **HQNR 판정에는 plateau(ep200–245) 동일 epoch 평균을 함께 낸다.**
  best 선택은 HQNR 로 하는데 HQNR plateau 가 평평(실행 내 sd ±0.0004)해 argmax 가
  튄다. 두 기준의 **부호가 다르면 "구분되지 않는다"**, 일치할 때만 방향을 주장한다.
  T1 에서 실제로 부호가 뒤집혔다 → [T1 §2.1](2026-08-31_T1_c6-uncertainty-teacher.md)

## 그 외

- 모든 수치에 **어느 실행 / 누가 쟀는가**를 명시한다 (`fixed`·`baseline` / `py`·`matlab`).
- 그림은 `assets/`. matplotlib 에 한글 글리프가 없어 그림 라벨은 ASCII 로 쓴다.
- 기존 문서는 고치지 않는다 ([`../CONVENTION.md`](../CONVENTION.md) §5).
  결론이 바뀌면 새 문서에 "무엇이 왜 바뀌었는지" 를 쓴다.
- 실험 약명을 단독으로 쓰지 않는다 — 항상 서술형 세팅을 병기한다.

---

## s1 — KD 캠페인 (Teacher = c6 계열 3.77M → Student = R4 2.13M) — **7/7 완주**

> **종합 결론: [KD 캠페인 종합](2026-09-01_KD-campaign-summary.md)** — KD 를 붙인 6벌 중
> **K0(Student 단독)를 넘은 것이 하나도 없다.** 원인은 기법이 아니라 전제다 —
> **Teacher(3.77M)의 HQNR 이 Student(2.13M)보다 낮다**(c6 −0.505%, T1 −0.434%).
> Teacher 를 ERGAS 로 골랐던 것이 어긋남의 출처. U-Know 가중은 이득 장치가 아니라
> **오염 방어 장치**로 작동해 K1B 의 손실 92% 를 걷어냈다(−0.507% → −0.038%).

| # | case | 설정 | 보고서 | HQNR 수렴 | vs K0 |
|---|---|---|---|---:|---:|
| 1 | `T1_c6_unc` | Teacher: c6 + unc NLL | [T1](2026-08-31_T1_c6-uncertainty-teacher.md) | 0.95121 | −0.434% |
| 2 | `T2_c6_unc_sis` | Teacher: T1 + SiS | [T2](2026-08-31_T2_c6-uncertainty-sis-teacher.md) | 0.94850 | −0.717% |
| 3 | `K0_R4_base` | **Student 단독, KD 없음** | [K0](2026-09-01_K0_R4-baseline.md) | **0.95536** | **기준선** |
| 4 | `K1A_R4_fullKD` | full-output KD, T=c6 | [K1A](2026-09-01_K1A_full-output-KD.md) | 0.95462 | −0.077% |
| 5 | `K1B_R4_specKD` | spectral KD, T=c6 | [K1B](2026-09-01_K1B_spectral-KD.md) | 0.95051 | **−0.507%** |
| 6 | `K1B_T1_specKD` | spectral KD, T=T1 (교란 분리) | [K1B_T1](2026-09-01_K1B-T1_spectral-KD-teacher-swap.md) | 0.95194 | −0.357% |
| 7 | `K2_R4_uknow` | + U-Know 가중 | [K2](2026-09-01_K2_uknow-weighting.md) | 0.95401 | −0.141% |
| 8 | `K3_R4_uknow_gtvar` | + GT 국소분산 | [K3](2026-09-01_K3_gt-variance.md) | **0.95499** | **−0.038%** (KD 최선) |
| 9 | `K4_R4_uknow_gtvar_sis` | + Student SiS | [K4](2026-09-01_K4_student-sis.md) | 0.95462 | −0.077% |
| — | `c6_c4d124` | **Teacher (3.77M)** | — | 0.95053 | −0.505% |

**case 별 한 줄:**

| case | 답 |
|---|---|
| T1 | Teacher 품질 보존(c6 와 구분되지 않음) + **θ calibration 압도적 통과** (Spearman 0.884 · 10분위 완전 단조 14.6× · risk–coverage excess 0.082) → K2~K4 게이트 개방 |
| T2 | SiS 는 D_λ −6.7% 를 D_s +17.7% 로 상쇄당해 HQNR 이득 없음 → K2+ teacher 는 **T1 확정** |
| K0 | **Student(2.13M)가 Teacher(3.77M)보다 HQNR 이 높다** — 캠페인 전제가 여기서 깨진다. 구 실행과 0.03% 재현 |
| K1A | 출력 전체 복제 → 이득 없음. D_λ 를 얻고 D_s 를 잃는 교환이 시작된다 |
| K1B | **선택적 전달 가설 기각.** 전체 복제보다 6.6배 더 잃고, 수렴 HQNR 이 **c6 Teacher 와 5자리 일치**(0.95051 vs 0.95053). 손상이 soft 램프 완료(ep74)에 정확히 시작 |
| K1B_T1 | 분해 완성 — K2 회복분 중 **가중 59% > teacher 교체 41%** |
| K2 | 가중은 작동(단일 항 최대 회복 +0.216%p)하나 K0 미달. **이득 장치가 아니라 방어 장치** |
| K3 | GT 분산이 θ 에 더할 것이 있었다(+0.103%p, D_s 회복 경유). **KD 최선이자 K0 와 동급** |
| K4 | Student SiS 도 이득 없음. **T2 와 부호 일치** → SiS 는 이 파이프라인에서 접는다 |

`K5`(feature KD)는 구현 결함(proj 스케줄러 미적용)으로 **No-Go 보류**.

## s1 — SE ablation (R4 위 SENet 2-Case) — **2/2 완주**

계획: [`../../research_log/se_ablation_two_experiments.md`](../../research_log/se_ablation_two_experiments.md)

> **결론: 계획 §9 의 "둘 다 무효 → SE 계열 종료".** 두 위치 모두 HQNR 이
> 기준선 2개 × 선택기준 2개 = **4가지 대조 전부에서 음수**다. 게이트는 살아 있으나
> (std 0.04~0.12, 포화 0%) **MS/PAN mode 를 구분하지 않는다**(cos 0.9937~0.9999) —
> 계획 §8 채택조건 3 불충족. 결합(SE1+SE2) ablation 은 열지 않는다.

| case | 설정 | 보고서 | HQNR 수렴 | vs K0 | D_λ | D_s |
|---|---|---|---:|---:|---:|---:|
| — | **R4 기준선 (K0)** | [K0](2026-09-01_K0_R4-baseline.md) | **0.95536** | — | 0.0235 | **0.0209** |
| `SE1_R4_btl_se` | bottleneck SE-ResBlock ×4 (2.1382 M) | [SE1](2026-09-01_SE1_bottleneck-SE.md) | 0.95325 | −0.221% | **0.0220** | 0.0243 |
| `SE2_R4_dec_h2_se` | H/2 skip-fusion SE ×1 (2.1309 M) | [SE2](2026-09-01_SE2_skip-fusion-SE.md) | 0.95250 | −0.299% | 0.0235 | 0.0237 |

| case | 답 |
|---|---|
| SE1 | 가설 절반 성공 — **D_λ 를 −6.4% 실제로 회복**(w128 c6 수준에 근접)했으나 D_s 를 +16.3% 잃어 HQNR 순손실. R4 를 "더 나은 R4" 가 아니라 **"작은 c6"** 로 만들었다 |
| SE2 | 경쟁 가설 기각 — **D_λ 개선 0%**, D_s 만 +13.4% 잃었다. 병목은 skip-fusion 채널 선택이 **아니다**. 게이트 범위도 SE1 보다 좁다(0.31~0.64 vs 0.13~0.87) |

**캠페인 공통 패턴**: `D_λ 를 사고 D_s 를 파는` 변경은 이 아키텍처에서 HQNR 순손실이다.
SE1·SE2·[T2 SiS](2026-08-31_T2_c6-uncertainty-sis-teacher.md)·[KD 6벌](2026-09-01_KD-campaign-summary.md)
전부 같은 방향이었다. **R4 의 D_s 0.0209 는 전 실행 최저이고, 무엇을 더해도 올라갔다.**

## s1 — 진행 중

| case | 상태 |
|---|---|
| `MS1_R4_msonly` · `MS1_w96_9ch_msonly` · `MS1_w128_msonly` · `MS1_w128_9ch_msonly` | 2026-09-01 16:32 기동, 1/4 진행 중 |

## s2 — Mutual 캠페인

| # | case | 상태 | 보고서 |
|---|---|---|---|
| 1~4 | `M0`→`M1`→`M2`→`M3` | 대기 | — |

---

명세: [`../../research_log/s1_mutual_and_kd_implementation_spec.md`](../../research_log/s1_mutual_and_kd_implementation_spec.md) ·
구현·검증: [`../../research_log/2026-08-31_kd-mutual-implementation-report.md`](../../research_log/2026-08-31_kd-mutual-implementation-report.md)

## 공통 진단 도구

```bash
python tools/uncertainty_diag.py <run>             # uncertainty 계열 필수 — 5종 진단 + 4패널 그림
python tools/check_calibration.py work_dir/<run>   # 게이트 판정만
```

`uncertainty_diag.py` 가 내는 것 (uncertainty 를 다루는 case 는 **반드시** 포함한다):

1. Spearman / Pearson (θ, |e|)
2. 조건부 오차 순서 `E[e|Top10] > E[e|Top20] > E[e|Top30] > E[e] > E[e|Bottom10]`
3. 10분위 error curve
4. risk–coverage curve (+ oracle / random 기준선, AURC, 정규화 excess)
5. θ vs GT 국소분산 상관 (+ GT 분산 자체의 오차 예측력 — θ 의 경쟁 가설)
