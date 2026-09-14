# EQREC4-S1-v1 결과 — offset consistency q 는 무엇을 말해 주는가 (s1, 2026-09-15 03:00)

계획 `research_log/PAN_S1_EQREC4_Alignment_Cue_Hypotheses_20h_2026-09-14.md` · 구현·운영 노트 `research_log/2026-09-14_eqrec4-implementation.md`(§6 감사 반영, 02:00–03:00 K10/REPORT 수정 포함) ·
자동 보고서 `work_dir/_eqrec4_s1_campaign/report_EQREC4.md`(표 A–F, `hypothesis_verdicts.json`, `verdict_evidence.csv`, 그림 `figures/`).
**전부 py**(`tools/eqrec4` 진단 + `tools/metrics` 평가기; MATLAB 실행 아님). 대상 checkpoint 는 PALS24/NF16 의 `fixed` recipe(W112·D123, 9ch, 새 U-Net + N2 R200 last aligner fine-tune):
**L1E4**(λ_off 1e-4; seed 1234·7777·2025, best_raw 와 last) 가 주 대상, L000(λ_off 0)·L1E2(0.01) 는 대조. 학습은 없고(K10/K20 의 단기 적응 제외) 고정 가중치의 진단이다.

기호: **e** = native(무변위) 입력에서의 64² patch L1 오차 · **q_A** = 축 방향 probe bank(반경 0.25/0.5/1/2 px × 4 축) 로 잰 offset consistency 잔차 |ĉ_ε + ε − ĉ_0| 의 평균(작을수록 aligner 가 자기 변위 예측에 일관) · q_B = 대각 bank ·
4분면 = e 중앙값·q_A 중앙값 기준 EdCd(오차↓·q↓)/EdCu/EuCd/EuCu · source block = 32 연속 patch index 의 proxy(실제 scene id 없음) · FR20 = 논문 세트 `.mat` 20장, HQNR 은 raw-original 전체 프레임.

## 1. 한 줄 결론

**q 는 "정합이 잘 된 sample" 의 표지가 아니다.** 같은 checkpoint 안에서 q_A 가 낮은 sample 은 native 오차가 오히려 높고(H1 native, 3 seed 전부 반대 방향), 추가 변위에도 더 크게 무너진다(H2, 3 seed 전부 반대).
그런데 scene 단위(FR20) 에서는 q_A 가 낮은 scene 이 raw HQNR 이 높다(H1 FR, 3 seed 전부 p<0.05). 즉 q 는 **scene 수준 난이도/텍스처와 함께 움직이는 지표**이지 patch 수준 정합 품질의 지표가 아니다.
learned correction 은 zero correction 보다 낫고(H3a, 3 seed 전부 양) 그 이득의 크기는 q_A 가 낮을수록 크다(H3b) — 이건 "정합이 보정할 것이 있는 patch 에서 보정이 작동한다" 로 읽힌다.
Student 감독 cue 로서의 q 는 **판정 불가**(H4 diag) 이고, 5K 단기 적응(K20) 에서 q 를 넣은 가중은 D-split L1 을 통계적으로 아주 조금 낮추지만 **HQNR 은 모든 arm 이 같이 떨어져(0.9555 → 0.951–0.952) HQNR 기준으로는 아무 arm 도 이득이 없다.**
→ L1E4 채택(기존 working reference) 은 그대로, **q_T 를 Teacher-quality gate 로 쓰는 안은 채택하지 않는다.**

## 2. 가설별 판정 (seed 를 단위로, best_raw·last 가 같은 방향 + source-block bootstrap CI 가 0 을 제외해야 "Supported/Opposed"; 하나라도 어긋나면 unresolved)

| 가설 (예측) | 판정 | L1E4 seed 1234 / 7777 / 2025 (best_raw; last 는 같은 방향) | 범위·주의 |
|---|---|---|---|
| H1 native: q_A↓ ↔ e↓ (ρ>0) | **Opposed** 3/3 | ρ(q_A, e) = −0.115 [−0.152, −0.077] / −0.122 / −0.149 (n 3,584 = B∪C∪D, 112 block) | L000 도 −0.21, **L1E2 는 +0.17/+0.25** — 부호가 λ_off 에 따라 뒤집힌다 |
| H1 FR: scene q_A↓ ↔ raw HQNR↑ (ρ<0) | **Supported** 3/3 | ρ = −0.683 (p 9e-4) / −0.898 (8e-8) / −0.782 (5e-5); last −0.67 / −0.67 / −0.47 (p 0.036) | 20 scene; FR20 은 개발에 반복 사용된 세트 |
| H2: q_A↓ 이면 bank B 추가 변위의 오차 증가 d_e 가 작다 (ρ>0) | **Opposed** 3/3 | ρ(q_A, d_e) r=1 px: −0.378 / −0.481 / −0.517 (r=2: −0.41 / −0.48 / −0.49; r=0.5 도 음) | D split 상세 subset, response 경로, paired valid set |
| H3a: learned correction > zero (paired gain>0) | **Supported** 3/3 | gain = +0.00083 [0.00080, 0.00087] / +0.00115 / +0.00092 L1, 양인 sample 95–96 % (n 4,096, 128 block) | 같은 U-Net 에서 c 를 0 으로 바꾼 것 — P0 와의 인과 비교가 아니다 |
| H3b: q_A 가 gain 크기를 예측 (ρ<0) | **Supported** 3/3 | ρ(q_A, gain) = −0.579 / −0.554 / −0.564 (last −0.59 / −0.52 / −0.63) | change-score coupling(e 와 gain 이 항을 공유) 주의 |
| H4 diag: q 를 더하면 K10 pool utility 예측이 좋아지나 | **Insufficient / unresolved** | Pair A LOO MAE B0 0.000901 → B1(+q) 0.000999 (**나빠짐**), Pair B 0.001289 → 0.001021(좋아짐), A→B 교차 불일치; direction acc 1.0 은 utility 가 전부 양이라 무의미 | source-disjoint 9 pool/pair(§4), 재실행 noise 4.1e-6 |
| H4 exec: K20 5K 적응에서 EAQ < EA, EAQ < EAQ-SHUF (D locked L1) | **Supported** (Pair A) | EAQ−EA = −2.75e-6 [−4.05e-6, −1.58e-6], EAQ−SHUF = −8.4e-7 [−1.19e-6, −5.0e-7]; U−R = −5.4e-5 (32 block) | **HQNR 은 아래 §5 — 모든 arm 이 하락**, 효과 크기는 L1 의 1e-4 % 수준 |
| Hspec: spectral 재조합만으로 q 가 변하나 | descriptive | MS 밴드 상수화/교환이면 q_A 0.28 → 0.49, EPE 0.53 → 0.94 px, ĉ_0 변화 0.48–0.60 px (bilinear 커널·padding·PAN affine 은 무변화) | 표 C; aligner 는 MS 내용에 강하게 의존 |

## 3. 네 집단 (primary L1E4 S2025 best_raw 의 4분면; 전수 4,096 patch, 집단당 882–1,152)

| 집단 | n | e 평균 | q_A 평균 | stress(bank B r=1, D subset 64): response / no_response / known_inverse d_e | PAN 민감도 d_e (blur σ1 / 다른 scene / 평균 상수) | learned−zero gain (core, 자기 4분면) · 양 비율 |
|---|---:|---:|---:|---|---|---|
| EdCd (e↓ q↓) | 920 | 0.0128 | 0.312 | 0.0205 / 0.0308 / 0.0011 | 0.0139 / 0.0483 / 0.0380 | +0.00087 · 0.990 |
| EdCu (e↓ q↑) | 1152 | 0.0099 | 0.338 | 0.0145 / 0.0211 / 0.0008 | 0.0102 / 0.0408 / 0.0258 | +0.00034 · 0.910 |
| EuCd (e↑ q↓) | 1142 | 0.0242 | 0.311 | 0.0413 / 0.0587 / 0.0043 | 0.0315 / 0.0871 / 0.0669 | +0.00167 · 0.992 |
| EuCu (e↑ q↑) | 882 | 0.0273 | 0.340 | 0.0260 / 0.0361 / 0.0026 | 0.0196 / 0.0570 / 0.0403 | +0.00073 · 0.893 |

읽기: (i) 집단 정의상 E↓ 의 e 가 낮은 것은 결과가 아니다. (ii) 어느 집단이든 aligner 가 반응하는 경로(response) 의 오차 증가가 무반응(no_response) 보다 작고 known_inverse 는 거의 0 — 보정 자체는 작동한다(H3a 와 일관).
(iii) q↓ 집단(EdCd·EuCd) 이 learned−zero gain 이 크고 양 비율도 99 % — "보정이 할 일이 있는 patch" 다. (iv) EuCd(오차 높고 q 낮음) 가 stress·PAN 민감도 모두 가장 크다: **q 가 낮은데 오차가 높은 집단은 PAN 내용(다른 scene 대체 d_e 0.087) 에 가장 민감한 집단**이며, 이것이 H1/H2 의 "반대 방향" 을 만든다. 원인(텍스처·경계·spectral) 은 자동 판정하지 않았다 — 표 B 의 texture 열(Scharr energy) 은 EuCd 0.0119 로 다른 집단(0.002–0.0065) 의 2–6 배다.

## 4. K10 — 같은 pool 에 hard 대 soft 추가 항 (Pair A: L1E4 S2025 best ← S1234 20K 후보 Student; Pair B 반대 방향; 64 update, fresh AdamW, C split 128 patch 로 평가)

| pair | cell | pool 수 | U_soft−hard (rel, C L1) 평균 · 양인 pool |
|---|---|---:|---|
| A | EdCd / EdCu / EuCd / EuCu (모두 a_T ≤ 0) | 2 / 3 / 2 / 2 | +0.0036 / +0.0055 / +0.0046 / +0.0072 · 9/9 |
| B | EdCd / EdCu / EuCd / EuCu (모두 a_T > 0) | 2 / 3 / 2 / 2 | +0.0038 / +0.0044 / +0.0049 / +0.0070 · 9/9 |

soft(Teacher 출력 쪽) 추가 항이 hard(GT 재가중) 추가 항보다 C 에서 항상 조금 낫고(재실행 noise 4.1e-6 대비 1,000 배), **cell(= q 4분면) 간 차이는 그 순서가 e/q 와 단조로 맞지 않는다**(EuCu 가 최대). Pair A 는 Teacher 가 Student 보다 나은 sample(a_T>0) 이 14개뿐이라 그 집단에 대한 주장은 불가(계획 §5 의 "Teacher 우위 집단에서도 효과" 는 검증 못 함).
pool 은 source block 을 cell 에 배타 배정해 만들었다(9 pool/pair, block 전역 분리; 02:00–02:12 두 번 수정 — 노트 §6). 다른 cell 에 있는 block 의 sample 은 버려서 pool 에 든 sample 은 432/2,048 이다.

## 5. K20 — 조건부 gate pilot (Pair A; gate 통과: q_A·q_B 재현성 ρ 0.926, label 일치 0.878, 9 pool, utility MAD 0.00145 ≫ noise 4e-6; 5,000 update, D split locked)

| arm (가중) | D L1 (0 → 5000) | raw HQNR FR20 (0 → 5000) | fSCC (0 → 5000) |
|---|---|---|---|
| R (재가중 없음) | 0.01789 → 0.01801 | **0.95552 → 0.95118** | 0.8755 → 0.8553 |
| U (균등 soft) | 0.01789 → 0.01796 | 0.95552 → 0.95193 | 0.8755 → 0.8565 |
| EA (e·a 셀 가중) | 0.01789 → 0.01795 | 0.95552 → 0.95199 | 0.8755 → 0.8567 |
| EAQ (e·a·q 셀 가중) | 0.01789 → 0.01795 | 0.95552 → 0.95210 | 0.8755 → 0.8569 |
| EAQ-SHUF (q 라벨 조건부 섞음) | 0.01789 → 0.01795 | 0.95552 → 0.95205 | 0.8755 → 0.8568 |

**HQNR 기준(판정 규칙 HQNR > SCC > ERGAS)**: 5K 단기 적응은 **모든 arm 에서 raw HQNR 을 0.0034–0.0043 떨어뜨렸고** arm 간 차이(≤0.0009) 는 seed 2σ(≈0.011) 훨씬 안이다. D L1 의 paired 차이(EAQ < EA < U ≪ R, CI 가 0 제외) 는 실재하지만 크기가 1e-6 수준이고 HQNR 로 이어지지 않는다.
따라서 H4 exec "Supported" 는 **D-split L1 에 한정된 진술**이고, 이 실험의 주 지표로는 q 가중이 무엇도 개선하지 않는다. 50K KD 성공 여부는 이 pilot 이 말하지 않는다(계획 §15.4).

## 6. 정합의 세 수준 (표 C 요약; primary S2025 best_raw · S1234 best_raw)

- relative response(bank A, r 평균): 반응 기울기 B_yy/B_xx = −0.32/−0.38 (S2025), −0.42/−0.57 (S1234) — 완전 반응(−1) 의 1/3–1/2. PAN-only/MS-only/common 분해와 bank B 는 보고서 표 C.
- known absolute synthetic error: 합성 변위의 역보정은 bank A r=1 에서 known_inverse d_e ≈ 1e-6(≈0) — 격자 내 보정은 정확하다(calibration OK, OOD 진단은 표 C).
- native before/after proxy(estimator 로 잰 PAN–MS 잔여 변위, 센서 GT 아님): FR20 에서 보정 전 중앙값 1.79 px → 후 1.21 px, 20/20 scene 감소 — 단 estimator 의 2차 독립 검증기가 없어(census 는 같은 estimator 의 gate) **판정에 쓰지 않는다**(`n_both_accepted 0`).

## 7. 소요·운영

| stage | G00 | D10 | D20 | D30 | D30B | D40 | D50 | K10 | K20 | REPORT | 합 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 분 | 0.5 | 22.2 | 0.9 | 75.0 | 4.3 | 63.3 | 1.5 | 6.6 | 45.5 | 0.0 | 220 (3.7 h; 계획 상한 20 h) |

23:07 기동 → 03:00 DONE. 중간에 runner 교체(D30B 삽입, 00:49) · K10 두 번 실패(pool 의 source block 분리 규칙; 02:00·02:05, run 1 산출물은 `_run1_badpools_k10_k20/` 보존) · REPORT 한 번 실패(죽은 줄) — 전부 노트 §6. 판정·표는 마지막 실행(K10 02:07–02:13, K20 02:14–02:59, REPORT 03:00) 의 것이다.

## 8. 한계

- source group 이 index-block proxy 라 bootstrap CI 는 보수적으로 읽는다(G00 `provenance_limited`). FR20 은 개발에 반복 사용된 benchmark. Pair A 의 a_T>0 sample 14개.
- H1/H2 의 "반대 방향" 은 **상관**이다 — q 가 낮은 patch 가 왜 PAN 내용에 더 민감한지(텍스처·경계·spectral) 는 개입으로 가르지 않았다. 표 B 의 texture 와 Hspec(MS 밴드 상수화가 q 를 0.28 → 0.49 로) 이 단서다.
- K20 은 Pair A 하나·5K·D split 고정이다.

## 9. 다음

- q_T gate 는 채택하지 않는다. PAKD50 의 Teacher T0(L1E4 S2025 best_raw) 선택은 이 결과와 무관하게 유지(정합 축 판정은 `2026-09-14_palsv18-lambda-confirmed-and-na104-null.md`).
- s3 에서 같은 캠페인을 돌리면(bundle 이식, 노트 §5) 서버 간 재현만 추가된다 — 새 가설은 없다.
