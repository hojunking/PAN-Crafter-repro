# 2026-09-14 — s1: PALSV18 결과(λ_off 3e-5·3e-4 × 3 seed) + PAKD50 통합 캠페인 기동

**요지.** L1E4(λ_off 1e-4) 양옆의 λ 는 3 seed 에서 L1E4 를 넘지 못했다 — 3e-5 는 세 seed 모두 낮고(평균 −0.0032), 3e-4 는 seed 1234 에서만 높고 나머지 둘에서 낮다(부호 불일치, best 가 전부 16160 update 의 초기 checkpoint). **working reference 는 L1E4 유지.** 정합 능력(반응·native 보정량)은 λ 에 단조라 09-13 결론과 같다. 이어서 L1E4 를 Teacher 로 쓰는 통합 캠페인 PAKD50 을 s1 에서 시작했다.

- 실행: `fixed`(재구성본 W112·D123), 평가 `py`+`matlab`(FR 논문 세트 .mat 20장, raw_original). 계획 [`PAN_L1E4_Refinement_AlignmentValidation_S1_18GPUh_2026-09-13.md`](../research_log/PAN_L1E4_Refinement_AlignmentValidation_S1_18GPUh_2026-09-13.md), 구현 [`2026-09-13_palsv18-implementation.md`](../research_log/2026-09-13_palsv18-implementation.md).
- 약명→세팅: `CTRLP0` aligner 없음(NF16 P0 정의) · `L000` λ 0(NF16 P2 정의) · `L3E5/L1E4/L3E4` λ_off 3e-5/1e-4/3e-4 (N2 R200 last aligner 를 T0 로 fine-tune, 홀수 update 에 P_ε 를 aligner 에만, b=2) · `L1E2` λ 0.01(NF16 P3).

## 1. 무엇을 돌렸나

| 항목 | 값 |
|---|---|
| 신규 학습 | 6벌 (L3E5·L3E4 × seed 1234·7777·2025, 각 50K), 2026-09-13 20:49 → 09-14 06:30, run 당 1.43–1.49 h + 진단 0.15 h, 실패·재개 없음 |
| 대조군 | P0/L000/L1E4 × 3 seed = NF16 P0/P2 + PALS24 (재사용 gate: G-M1 재평가 Δ 0.0, 초기 tensor·donor hash 일치) |
| 예산 | 18 GPU-h 중 학습 10.3 h(중단분 0.42 h 포함) + V-pre 0.13 + V-post 4.0(13:03 에 일시 중단, 11/23 checkpoint) = 14.3 h |
| 첫 run 중단 | 리뷰 반영(평가 checkpoint 보존 등 7건)으로 14.5K 에서 폐기·재실행 — 결과에 포함하지 않음 |

## 2. 공식 성능 (best_raw raw_original HQNR, matched grid = 25 후보 공통 격자; 대응 차이는 같은 seed)

| λ_off | seed 1234 | seed 7777 | seed 2025 | 3-seed 평균 ± sd | Δ vs L1E4 (seed 별) | Δ vs L000 | Δ vs P0 |
|---|---:|---:|---:|---:|---|---|---|
| **L1E4 1e-4** (기준) | 0.95552 | 0.95542 | 0.95698 | 0.95597 ± 0.00087 | — | +0.0016 / +0.0032 / +0.0015 | +0.0028 / +0.0046 / +0.0063 |
| L3E5 3e-5 | 0.95340 | 0.95161 | 0.95331 | 0.95277 ± 0.00101 | −0.0021 / −0.0038 / −0.0037 (3/3 음) | −0.0005 / −0.0006 / −0.0022 (3/3 음) | +0.0007 / +0.0008 / +0.0026 |
| L3E4 3e-4 | 0.95792 | 0.95340 | 0.94991 | 0.95374 ± 0.00402 | **+0.0024** / −0.0020 / −0.0071 (부호 불일치) | +0.0040 / +0.0012 / −0.0056 | +0.0052 / +0.0026 / −0.0008 |
| L000 0 | 0.95389 | 0.95222 | 0.95547 | 0.95386 | | | |
| P0 | 0.95272* | 0.95081 | 0.95068 | 0.95140 | | | |

\* P0 seed 1234 는 eval_epoch 5 run 이라 matched grid 값(원 selector 0.95316). 판정선 0.0031(HQNR 만). fSCC 는 L3E5 0.899/0.896/0.890 > L1E4 0.875/0.884/0.879 > L3E4 0.877/0.879/0.886 — HQNR 순위와 반대(09-13 §2.6 의 대가 그대로).

**판정.** performance_leader = L1E4(3-seed 평균 선두). L3E5 는 평균 차이가 판정선 밖(−0.0032, 3/3 음)이고 L000 보다도 낮다 — "약한 λ 가 P2 와 같은 동작으로 수렴" 하는 것도 아니고 그보다 낮다. L3E4 는 seed 1234 의 단일 자산(0.95792, 이 캠페인 전체 최고)만 L1E4 를 넘고 세 best 가 전부 16160 update 의 초기 checkpoint 다(L1E4 는 40400/18180/24240). 계획 §12.2 의 "L1E4 만 좋고 양옆에서 하락" 에 해당하며 좁은 최적 구간·선택 민감성으로 기록한다. 더 많은 λ 를 만들지 않는다.

## 3. 같은 checkpoint 의 정합 능력 (V1, last 50K, fr512 고정 probe 반경 {0.25,0.5,1,2}×8 방향; 진단이며 판정 아님)

| λ_off | native ‖ĉ‖ 중앙값 (px, seed 1234/7777/2025) | B_resp 대각 (이상 −1) | offset EPE (px; 무반응 0.9375) | donor 대비 drift |
|---|---|---|---|---|
| L000 0 | 0.34 / 0.35 / 0.34 | −0.20 | 0.75 | 1.17 |
| L3E5 3e-5 | 0.38 / 0.38 / 0.38 | −0.23 | 0.71 | 1.14 |
| L1E4 1e-4 | 0.56 / 0.54 / 0.60 | −0.51 ~ −0.53 | 0.44–0.48 | 0.93–0.98 |
| L3E4 3e-4 | 1.08 / 1.11 / 1.12 | −0.61 ~ −0.65 | 0.36 | 0.45–0.51 |
| L1E2 0.01 (seed 1234) | 1.43 | −0.79/−0.86 | 0.18 | 0.28 |

반응(B, EPE)과 native 보정량은 λ 에 단조·seed 간 일관이고, HQNR 은 1e-4 에서 정점 — 09-13 문서의 "정합을 잘하는 것과 HQNR 이 좋은 것이 λ>1e-4 에서 갈라진다" 가 양옆 λ 에서도 재현된다. 모달리티 검사(V2, L1E4 seed 1234 best): pan_only B ≈ −0.51 I, ms_only ≈ +0.49 I, common ≈ 0 (EPE 0.018) — aligner 는 PAN–MS 상대 위치에 반응한다(학습하지 않은 MS 이동에도 부호가 맞는다). V3.3: bicubic warp 는 Scharr energy 를 줄이지 않아(비 1.03) energy-match blur 대조는 `unmatched_blur_control`. V3.1 native proxy(Scharr-ZNCC, PAN_b←up(MS)): L1E4 seed 1234 best 에서 1.79 → 1.07 px(20/20 개선; 추정기 acceptance 는 0/20 이라 proxy 근거로만). V4 개입(L1E4 S1234 best): learned 0.9555 · zero 0.9495 · wrong_sign 0.9477 · scene_shuffle 0.9553 · constant 0.9479 — 학습된 보정은 유효하지만 장면별 차이는 작다(shuffle ≈ learned).

V-post 는 11/23 checkpoint 에서 멈췄다(PAKD50 에 GPU 양보). 나머지(신규 run 의 last·대조군 V3/V4) 는 캠페인 사이에 이어서 돌리고 표 B/C 를 `work_dir/_palsv18_campaign/final_report.md` 에 채운다.

## 4. PAKD50 통합 캠페인 기동 (WIP)

계획 [`PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md`](../research_log/PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md), 요약 [`PAN_Integrated_Method_Summary_2026-09-14.md`](../research_log/PAN_Integrated_Method_Summary_2026-09-14.md), 구현 [`2026-09-14_pakd50-implementation.md`](../research_log/2026-09-14_pakd50-implementation.md).
Teacher T0 = PALS24 L1E4 seed 2025 best_raw(step 24240) 의 A+U; Student 는 그 aligner 를 복사(J 공동 적응 / F frozen)하고 U-Net 은 새로 학습, backend N0/R1/Q12/X02. 세 서버 병렬 50h(서버당 학습 46h), seed s1 1234 · s2 777 · s3 2026, 목표 raw HQNR ≥ 0.959.

| 항목 | 값 |
|---|---|
| s1 기동 | **2026-09-14 13:22:31** (50h 시계 시작; 체인 마감 09-16 11:22), release cac23a6. 첫 run J0-1234 gate: 준비 0.02 + 1.1×4.0 + 4.0 = 8.4 ≤ 46 → RUN. smoke: J 40 ms/step(Teacher 없음 N0 도 eval-only Teacher 포함), F 19 ms, R1 22 ms(Teacher forward), peak 3.3 GB; 예상 run ≈ 1.9 h(50회 평가 포함) |
| stage 1 큐 | J0 → F0 → JR → FR (seed 1234); J0-1234 exact50K 가 λE pilot |
| stage 2 | λE 고정 뒤 JQ → FQ → XJ (gate `pakd50`) |
| T0 재현 | raw HQNR 0.956976 (기록과 Δ 0), τR 0.012464 |
| s2/s3 | 이 commit 을 pull 한 뒤 `./tools/pakd50_prepare.sh` (현재 NA104 20H 체인이 있으면 먼저 정리) |

## 5. 같은 날 추가 (14:04) — 구현 감사 반영·s1 큐 재편성

| 항목 | 내용 |
|---|---|
| 감사 | `research_log/PAN_Integrated_Implementation_Experiment_Audit_2026-09-14.md` F01–F10. J0 목적함수 오류 없음 → run 유지. F01–F05 반영(`research_log/2026-09-14_pakd50-implementation.md` §6), F06–F10 후속 |
| s1 큐 | 14:04 runner 교체(`tools/pakd50_requeue.sh`): 학습 중인 J0-1234 유지, 새 runner 가 J0 종료를 기다린 뒤 gate 편성 J0 → λE → **JQ** → F0 → FQ → JR → FR → XJ. §4 표의 "stage 1 큐 J0→F0→JR→FR / stage 2" 는 이 시점부터 무효 |
| 마감 | 공통 시계 `assets/pakd50/campaign_clock.json`: start 13:22:31 · 학습 마감 09-16 11:22:31 · 최종 15:22:31 — 체인·trainer·gate 가 같은 절대 시각(재기동으로 늘지 않음) |
| J0 진행 | 14:04 iter 18,000/50,000 (예상 완료 ≈ 15:20, 그 직후 λE 고정 → JQ) |
| s2/s3 | pull 뒤 `./tools/pakd50_prepare.sh`. λE 는 `assets/pakd50/calibration_resolved.json` 사본으로 전달 — pull 만으로 다음 pass 에 JQ 가 열린다 |
| λE 고정 (15:29) | J0-1234 exact50K(last, sha 82735386…) 에서 gate 가 λE = **0.0907517**(r_grad 0.05 × output-gradient RMS 비, τR 0.012464 그대로) 를 고정하고 `assets/pakd50/calibration_resolved.json` 사본에 mirror. 세 서버 stage 2 config 생성 → JQ-1234 15:29:44 시작. J0-1234 최종은 이 문서의 후속(09-15 s1 문서)에서 |
| s5 (16:30) | 배정 `research_log/PAN_S5_Timing_Routing_Experiment_Plan_2026-09-14.md` 반영: trainer 에 A 동결 일정·loss 별 A 수신 경로 구현(기본값이면 J 와 동일; K10–K12 + kdv/nf16/pals24 gate 통과, D0/PR GPU smoke OK). s1 의 JQ-1234 는 15:29 코드로 시작했고 그 다음 run 부터 새 코드(수치 경로 동일) |
