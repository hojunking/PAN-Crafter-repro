# EQREC4-S1-v1 구현 노트 (2026-09-14)

계획: `research_log/PAN_S1_EQREC4_Alignment_Cue_Hypotheses_20h_2026-09-14.md`(§번호는 그 문서). 사용자 지시(22:40): PAKD50 s1 체인을 중단하고 이 캠페인을 s1 에서 진행한다.
구현: `tools/eqrec4/`(패키지) + CLI `tools/eqrec4.py <stage>` + runner `tools/eqrec4_run.sh`(detached, stage 별 재개) + gate `tools/eqrec4_unit_tests.py`(E01–E10). 출력 root `work_dir/_eqrec4_s1_campaign/`.

## 1. PAKD50 s1 중단 (22:40)

- runner(`_run_cases.sh`) 와 감시자 cron 을 내렸다. 학습 중이던 `PAKD50_JR_…_S1234_FRESH50_v1`(22:02 시작) 은 끝까지 두고(≈00:05) 업로드는 수동(`./tools/_upload.sh`). s1 의 J0/JQ/F0/FQ 는 완료. s2–s5 의 PAKD50 은 영향 없음(λE0·stage 2 config 는 이미 배포).
- 두 캠페인은 이름·출력 root 가 분리돼 충돌하지 않는다(§6-8). gate token 은 `pakd50` 그대로(다른 서버 규약) — G00 이 상태만 기록.

## 2. 계획 → 구현 대응 (약명은 계획 §의 정의를 따른다)

| 계획 | 구현 | 비고 |
|---|---|---|
| §2.2 registry | `common.RUNS/CORE/EXTRA`: L1E4·L000 3 seed × best_raw(=저장소 `best_hqnr`)/last, L3E4 3 best, P0 3 best/last(aligner 없음 → q 없음), N2 last, L1E2 best/last | 모든 checkpoint 존재 확인(assets_manifest.csv) |
| §2.3 pair | Pair A: T=L1E4 S2025 best_raw(24240) / S=L1E4 S1234 ≥20K 최초 저장 = step **40400**(그 run 의 best_raw 와 같은 update) · Pair B: T=L1E4 S1234 best_raw / S=L1E4 S7777 step **30300** | 가상 checkpoint 없음; `late_student_fallback` False |
| §3.2 분할 | train_wv3.h5 9,714 patch 중 연속 index 32 블록 → seed 314159 → A 512 / B 2048 / C 512 / D 1024 (블록 단위 배정, 인접 patch 가 분할을 넘지 않음). 원본 scene id 없음 → `source_group_unknown`, 블록 = proxy | `data_manifest.csv`; seen_in_pretraining=True |
| §4.1 e·q | e = native full64 L1(정규화 단위; DN = ×1023.5), q_A = 1/(2K)Σ‖r‖₁, EPE = mean‖r‖₂, r = ĉε+ε−ĉ0, λ 없음 | E04 검사 |
| §4.2 probe | bank A r∈{.25,.5,1,2}×{0,90,180,270}°, bank B ×{45,…}°; identity 별도; q_const(bank) 기록 | `probe_bank_{A,B}.json` |
| §4.3 ROI | 64→[16:48], 256→[32:224], 512→V96; 두 warp support 검사 `two_stage_ok`; invalid_support 보고(clamp 없음) | `roi_contract.json` |
| §6 G00 | 자산 manifest·primary best_raw raw HQNR 재현(Δ 0.0)·RR L1 새로 계산·warp 부호/identity/ramp/z-score 검사·gradient 수신자(실제 wrapper)·gate token 상태 | `g00_*.json`, `protocol_resolved.yaml`, `protocol_changes.md` |
| §7 D10 | 24 model-ckpt × 4,096 patch(e, q_A/q_B/EPE/반경별, B fit, c0, 입력 통계) + RR20/FR20; A median threshold → 4분면(tie→low); occupancy·insufficient_support; H1 Spearman(block bootstrap)·decile·texture 층화·q_B 재현성; 상세 subset(64/집단, D 에서 source×texture 층화) | `native_sample_metrics.csv`, `offset_probe_records.csv.gz`(core), `quadrant_*.csv`, `h1_stats.json`, fig1 |
| §8 D20 | A: PAN-only 전수(D10 기록)·MS-only/common(상세 subset+RR+FR, `modality_response`) · B: RR GT 동일 격자 합성(P_syn=Σw_cZ_c, w 3종, Z/blur, δ 20개; 절대 오차 vs 상대 q) · C: MS swap/상수/PAN affine/bilinear/reflection | `response_matrices.csv`, `geometry_controls*.csv`, `geometry_cues*.csv` |
| §9 D30 | A: I-L/I-Z/I-W/I-C(A median c0)/I-S×10(집단 내 derangement + e-매칭)/I-G(estimator canonical, accepted 만)/I-BLUR(matched 일 때) — patch64 상세 subset(ROI L1)·RR20(L1+reduced)·FR20(raw views) · B: FR native proxy(`palsv18_validate.native_proxy` 재사용) + RR GT↔PAN proxy(GT-informed) · C: A+v (q 대수 항등 + 수치 확인, e/HQNR 변화) · D: 81 grid + c0 + c_geo landscape(L1·band·edge/flat·struct ZNCC), c_rec* 는 GT oracle | `correction_interventions.csv`, `native_geometry_proxy.csv`, `bias_intervention.csv`, `correction_landscape*.csv`, `h3_stats.json`, `blur_control_status.json` |
| §10 D40 | A: 두-warp stress(response/no_response/known_inverse; bank B r∈{.5,1,2} 대각 + bank A r=1 재현) patch64/RR/FR(참조 native 고정, V96) · B: PAN 민감도(c0 고정 → blur/other-scene/mean; A 재추정 열) · D: band 별 구조 정합 추정(GT-informed) · E: edge_profile_v1·energy·blur calibration 재사용 | `output_stress.csv`, `pan_sensitivity.csv`, `band_alignment.csv`, `band_edge_profiles.csv`, `h2_stats.json` |
| §11 D50 | g_r=∇φL_rec, g_c=∇φL_off^SG(ε r=1 4축 고정), cos·R_g(전체 A/fc2 head) · normalized SGD one-step JR/JC/JRC α∈{1e-5,3e-5} 뒤 e/q_A/q_B/c0/출력 재측정·복원 | `gradient_interventions.csv`, `gradient_summary.json` |
| §12·§14 K10 | 2 pair × 8 cell × ≤4 pool(48 고유, 32–47 은 cycling 기록) × R0/RH/RS × 64 update; fresh AdamW(U 1e-5/A 1e-6, WD config), 같은 batch·ε RNG; base(Rec+Off)→A+U, extra→U only (`k_step`: autograd.grad 분리, `.grad` 대입 뒤 step 한 번); C 128 고정 검증; 첫 pool 3회 반복으로 noise scale | `k10_utility.csv`, `k10_cells_*.csv`, `k10_h4_diag.json`(B0/B1 ridge LOO, A→B 교차) |
| §13 K20 | EA/EAQ 표(상대 U_soft-hard, shrink 4, n_g ≤ 독립 pool), s_u=max(MAD, noise), ρ=σ(clip+b_mass) 평균 0.5, EAQ-SHUF(EA cell 안 rank e/a 블록 16 derangement) · R/U/EA/EAQ/EAQ-SHUF × 5K(warmup100+cosine, U 1e-5/A 1e-6), source/2500/5000 저장, D L1(주)·RR20·FR20 | `k20_policy_tables.json`, `k20_weight_mass.csv`, `k20_endpoint_metrics.csv`, `k20_D_paired.json` |
| §15–16 report | 표 A–F, fig 1–7, H1–H4/Hspec 4-상태 판정(사전 규칙: 방향 + source-block bootstrap CI + 대조 일관성), §20 한 문장 | `report_EQREC4.md`, `hypothesis_verdicts.json` |

## 3. 계획과 다른 점 (protocol_changes.md 에도 기록)

- parquet 미지원(pyarrow 없음) → csv/csv.gz/jsonl.
- source group = index-block proxy(32). CI 의 독립성 가정이 약하다 — 보수적으로 읽는다.
- Pair A 학생 checkpoint(40400)는 그 run 의 best_raw 와 같은 update 다(규칙 "≥20K 최초 저장" 적용 결과).
- D30-A 의 blur 대조·I-C 의 상수는 `palsv18_validate.calibration_patches`(train 256 patch, seed 20260913) 의 정의를 계승 — 분할 A 가 아니라 별도 calibration patch.
- 두 warp support 검사는 보수적 규칙 `|ε|∞+|c|∞+4 ≤ margin`.
- D50 의 L_off 진단 ε 는 무작위 원판이 아니라 r=1 4 축 고정(결정적 대조).
- K10 "같은 batch" = fit pool 자체(48) 를 매 update 의 batch 로 쓴다(cycling 은 n 기록).
- 시간 ledger 는 wall time + GPU peak memory (GPU busy 시간은 따로 재지 않음).

## 4. 실행

- 23:07 `./tools/eqrec4_run.sh`(detached; T0 = `work_dir/_eqrec4_s1_campaign/T0.txt`) — G00(0.5 min, 재현 Δ 0.0, 수신자 검사 통과) → D10 → … stage 실패 시 그 자리에서 멈추고 재기동하면 이어 돈다(ledger done 기준).
- JR-1234(PAKD50) 학습이 GPU 를 같이 쓰는 동안(≈00:05 까지) 진단 속도는 느려질 수 있다.
