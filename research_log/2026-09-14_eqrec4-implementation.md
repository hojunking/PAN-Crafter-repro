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

## 5. s3 에서 실행 (2026-09-15 00:40 사용자 지시: "이 실험은 s3 에서 진행")

계획·코드는 서버 무관하게 돈다. s1 고유 의존은 두 가지였고 둘 다 제거했다.

| 의존 | 처리 |
|---|---|
| registry checkpoint 24벌 + pair 학생 2벌이 s1 `work_dir` 에만 있음 | `tools/eqrec4_bundle.py pack`(s1) → `work_dir/_eqrec4_bundle/`(14 run · 64 file · 272 MB, sha256 manifest). 전송은 사람이(rsync/scp; git 에는 넣지 않는다). s3 에서 `verify` → `install` 이 같은 `work_dir/<run>/…` 배치로 넣는다(다른 내용의 파일이 이미 있으면 덮어쓰지 않고 보고) |
| run config 의 s1 절대 dataroot (`setup_paths.sh` 는 `config/*.yaml` 만 고침) | `common.localize_cfg`: 없는 경로는 이 저장소의 `data/…` 로 |
| 출력 root·campaign id | `gspread/server.txt` 기준 `work_dir/_eqrec4_<server>_campaign`, `EQREC4_<SERVER>_v1` (s1 은 그대로) |

s3 절차: pull → bundle 전송 → `./tools/eqrec4_prepare.sh`(bundle verify/install → 데이터·DLPan → gate E01–E10 → G00(primary raw HQNR 재현 Δ ≤ 1e-4) → runner 기동). s3 의 PAKD50 체인이 돌고 있으면 GPU 를 같이 써 느려진다 — 중단 여부는 사람이 정한다(prepare 가 경고만).
s1 의 실행(23:07 기동) 은 그대로 두었다 — 같은 checkpoint·같은 분할이라 s3 결과와의 환경 교차(같은 sample 의 e/q·개입 재현) 로 쓸 수 있다. s1 을 멈추려면 `pkill -f eqrec4_run` 대신 `ps -eo pid,args | grep '[e]qrec4'` 로 PID 를 골라 kill.

## 6. 구현 감사(`research_log/PAN_EQREC4_Implementation_Experiment_Audit_2026-09-15.md`) 반영 — 2026-09-15 00:45–01:30

| # | 판단 | 반영 |
|---|---|---|
| Q01 K20 진입 조건 없음 | 맞다 | `k20.k20_gate`: G00 유효 · q_A/q_B 반복성(ρ≥0.8, label 일치 ≥0.75) · K10 pool ≥4·restore ok·source 분리 · 원시 C(npz) 에서 U_soft-hard 재현(≤1e-7) · 식별 가능 신호(noise 있으면 max|u|>2·noise, 없으면 MAD>0; 임의 floor 없음) · proxy source 지원 ≥5. 실패면 `k20_gate.json`(pilot_not_identifiable) 만 남기고 학습하지 않는다; report 는 `final_training_pilot_not_run` |
| Q02 pool 의 source 공유·held-pool | 맞다 | `k10.fit_pools` 가 cell 안에서 **source block 단위**로 pool 을 채운다(한 block 은 한 pool 에만; 48 을 넘는 나머지는 버림) → LOO 가 source 분리. `source_blocks`·`source_disjoint_across_pools` 기록. G00 `provenance_limited` 에 proxy 명시 |
| Q03 H2 에 bank A 혼합 | 맞다 | `d40.h2_stats` 를 bank 별로 나눠 **bank B 를 주**로, A r=1 은 재현 표; paired valid = 세 경로 모두 valid 인 (sample, probe) 만. D40 은 수정 뒤 시작(00:46:47; 파일 00:46:40) 이라 저장 자료가 새 규약 |
| Q04 자동 판정 규칙 | 맞다 | `report.verdicts` 를 증거표 기반으로: 단위 = seed(best/last 합의, CI 가 0 제외), 세 seed 모두 일치해야 Supported/Opposed, 그 밖은 Insufficient; H4_diag 는 source 분리·Pair A→B 교차·noise 기록 조건; H4_exec 는 gate 통과 시만. `verdict_evidence.csv`. H1_RR → H1_native64 |
| Q05 band d_e 가 절대값 | 맞다 | `l1_stress_band*` 와 native 를 뺀 `d_e_band*` 를 모두 저장 |
| Q06 native64 전수 개입·자기 4분면 | 맞다 | 새 stage **D30B**: L1E4 6 checkpoint 의 4,096 전수 I-L/I-Z/I-W/I-C + `quadrant_own`(자기 threshold) / `quadrant_primary` 병기, `h3_core_stats.json`(전체·D·own 4분면별). D40/D50 행에도 `quadrant_own` 추가 |
| Q07 blur calibration 이 A 밖 | 맞다 | D30B `blur_calibration_A`: 분할 A 512 에서 energy 2% + overshoot 0 + 고주파 power 5% 를 모두 만족해야 matched; D30 의 blur 행은 "PALSV18 calibration 재사용(exploratory)" |
| Q08 shuffle 이 1D·최소 2 | 맞다 | rank e 순 32 묶음을 rank a 로 반갈라 16 블록(2D 근접), 8 미만 unmatched; `k20_shuffle_pairs.csv`(partner·q 전후·e/a rank 거리·source) |
| Q09 estimator secondary·범위 | 부분 | `common.est_full`(secondary·차이·margin·boundary) 로 RR/FR proxy 전체 필드(`native_geometry_proxy_full.csv`, known-shift sign_ok 포함), band alignment, native64 I-G(`native64_ig.csv`, accepted 만), RR bias 개입 전후 위치 proxy(`bias_position_proxy_rr.csv`), D20 MS-only/common **bank B**, stress 의 ROI 민감도(`d_e_roi_plus8`), primary zero-correction edge profile. 미반영: parent-context(64² 고정 patch 라 불가, 기록), 출력 profile overshoot |
| Q10 K10 원시·K20 2500 | 맞다 | `k10_raw/*.npz`(C id·arm 별 before/after per-sample·fit id·restore/final hash·grad log); K20 은 0/2500/5000 state 저장 + D band endpoint |
| Q11 G00 exit·재개 | 맞다 | 단독 `g00` 도 invalid 면 exit 2; ledger 에 code fingerprint(git+eqrec4 py sha); runner `done_stage` = ledger done **and** 필수 산출물 존재(`tools/eqrec4.py check`) |
| Q12 보고서 원인 설명 | 맞다 | Table B 를 열마다 지원 n 을 두고 집단별 관측/미검증 목록으로; 자동 원인 배정 없음(mixed/unresolved), placeholder 제거; 해석은 results_log 에서 |
| §5 a_T>0 희소 | 관측 | Pair A 는 Aneg 4 cell 만 지원 — K10 이 그대로 기록; "Teacher 우위 집단에서도 효과" 주장 불가를 보고서에 반영 |

운영: 옛 runner(D30B 없음) 는 D40 시작 뒤 종료(00:49)하고 D40 python 은 계속; D40 종료를 기다려 새 runner(D30B → D50 → K10 → K20 → REPORT) 를 기동한다.

**02:00 K10 실패·수정.** D40(63 min) → D30B(4.4 min) → D50(1.5 min) 뒤 K10 이 `fit_pools` 의 전역 분리 assert("source block shared across pools") 로 즉시 실패했다.
원인: block(32 연속 index 의 proxy) 하나의 sample 이 4 Aneg cell 전부에 흩어져 있는데(cell 당 64 block 전부 등장, block·cell 당 평균 7.6 sample), 종전 코드는 **cell 마다** 전체 block 으로 pool 을 채워 같은 block 이 여러 cell 의 pool 에 들어갔다 —
E11 fixture 가 단일 cell 이라 잡지 못했다. 수정: block 을 먼저 pool 을 만들 수 있는 cell(≥32 sample) 에 **배타적으로 배정**(seed 로 섞고 누적 sample 이 가장 적은 cell 에) 한 뒤 cell 안에서 자기 block 의 sample 로만 pool 을 채운다(계획 §12.2 "fit-pool/source 묶음 분할" 그대로; 감사 Q02).
실제 표(Pair A, B split 2048): 4 cell 이 자격, 9 pool(EdCd 3(하나는 35 cycled)·EdCu 2·EuCd 2·EuCu 2), 55 block 전부 서로 다름, pool 에 든 sample 419/2048 — 다른 cell 에 있는 block 의 sample 은 버린다(그래서 cell 당 ≤4 가 아니라 2–3 pool). E11 에 다중 cell fixture 추가(ALL OK).
runner 는 02:00 재기동 — 완료 stage 는 ledger+산출물 검사로 건너뛰고 K10 부터.

**02:05 K10 2차 수정.** 1차 수정의 배정 규칙(누적 sample 이 가장 적은 cell 에 block) 은 Pair A 에서는 9 pool 이었지만 **Pair B 에서 pool 0** 이었다 — Pair B 는 큰 Apos 4 cell(447–539) 과 자격(≥32) 은 있으나 block 당 0.5 sample 인 작은 Aneg 2 cell(38·34) 인데,
작은 cell 이 누적 sample 이 늘지 않아 block 을 계속 받아 64 block 을 다 삼키고도 pool 을 못 만들었다. K10 은 그 상태로 "done" 이 됐고 K20 이 시작돼 runner·K20 을 멈췄다(02:05; run 1 의 K10/K20 산출물은 `_run1_badpools_k10_k20/` 에 보존).
새 규칙: **pool 단위 round-robin** — 매 round 에 pool 수가 적은 cell 부터(동률이면 block 당 기대 sample 큰 cell 부터) 다음 pool 을 채울 만큼 block 을 주고, 남은 block 으로 못 채우는 cell(기대 수율 n_c/64) 은 건너뛴다. 실제 표: Pair A 9 pool(EdCu 3·EuCd 2·EdCd 2·EuCu 2, 432 sample, 62 block), Pair B 9 pool(EdCu_Apos 3·EuCd/EdCd/EuCu_Apos 2, Aneg 소 cell 0, 63 block); 모두 48 고유, block 전역 분리.
E11 에 Pair B 형 불균형 fixture 추가. runner 02:12 재기동(K10 부터).
