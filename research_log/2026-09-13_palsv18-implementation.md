# 2026-09-13 — PALSV18 구현 노트: L1E4 근방 미세조정·정합 능력 검증, 18 GPU-h (s1)

계획: [`PAN_L1E4_Refinement_AlignmentValidation_S1_18GPUh_2026-09-13.md`](PAN_L1E4_Refinement_AlignmentValidation_S1_18GPUh_2026-09-13.md). 캠페인 `PALSV18_W112D123_N2LAST_R200_v1`, 학습 의미 `PALS_W112D123_N2LAST_R200_v1` 계승.
PALS24(`2026-09-12_pals24-implementation.md`) 위에 **λ 값·예산 상수·검증 도구만** 더했다. trainer·모델·loss·sampler·optimizer·평가·selector 는 그대로다.

## 1. 무엇을 돌리나

| 순번 | run | λ_off | seed | 예약 |
|---:|---|---:|---:|---:|
| R1-1 | `PALSV18_L3E5_W112_D123_WV3_S1234_N2LAST_R200_v1` | 0.00003 | 1234 | 1.7 h |
| R1-2 | `PALSV18_L3E4_W112_D123_WV3_S1234_N2LAST_R200_v1` | 0.0003 | 1234 | 1.7 h |
| R2-1 | `PALSV18_L3E4_…_S7777_…` | 0.0003 | 7777 | 1.7 h |
| R2-2 | `PALSV18_L3E5_…_S7777_…` | 0.00003 | 7777 | 1.7 h |
| R3-1 | `PALSV18_L3E5_…_S2025_…` | 0.00003 | 2025 | 1.7 h |
| R3-2 | `PALSV18_L3E4_…_S2025_…` | 0.0003 | 2025 | 1.7 h |

대조군 9벌은 재사용한다(§3.2): seed 1234 CTRL-P0/L000 = NF16 P0/P2, L1E4 = PALS24 L1E4; seed 7777·2025 의 CTRL-P0/L000/L1E4 = PALS24 stage 2. 진단 참조: N2 donor `last`, 강한 λ L1E2 = NF16 P3(seed 1234).
약명→세팅: `CTRLP0` aligner 없음(NF16 P0 정의) · `L000` λ 0(NF16 P2 정의) · `L3E5`/`L1E4`/`L3E4` λ 3e-5/1e-4/3e-4 (A-FT donor aligner, I-AEQ: 홀수 update 에 P_ε 를 aligner 에만, b=2 원판, ramp 없음) · `L1E2` λ 0.01(NF16 P3 정의).

## 2. 계획 → 코드

| 계획 | 구현 |
|---|---|
| §3.1 6회·순서 | `tools/gen_palsv18_configs.py` (PALS24 생성기를 캠페인 상수만 바꿔 호출; gate PV02: 생성 config 는 PALS24 L1E4 와 `kdv.aux.offset_weight` 하나만 다르다). 큐 `config/queues/palsv18_s1.txt` |
| §4.1–4.3 예산 | `kdv.budget`: ledger `work_dir/_palsv18_budget/ledger.json` 18 h, reserve 5.0(V-post 3.5 + report 0.5 + buffer 1.0), margin 1.1, run 예상 = max(1.7, smoke×1.1) = 1.7, `remaining_mandatory` = 같은 seed 의 나머지 λ (pair 단위 완결), required False. 큐 순서 1234 → 7777 → 2025 라 부족하면 2025 pair → 7777 pair 순으로 통째로 DEFERRED |
| §5.1 gate | G-A/G-C/G-M/G-S: `tools/pals24_metric_gate.py --campaign palsv18` (대조군 9벌 + 참조 2 의 G-M1 재평가, `selection_grid.json`(L1E4 S1234 의 25 평가 update + exact 50K), `initial_tensor_registry.json`(3 seed 초기 tensor = 대조군 기록)); G-G/G-W/G-R: `tools/pals24_unit_tests.py` PL03/PL04(실제 wrapper); PV01–PV07: `tools/palsv18_unit_tests.py`; G-T: smoke |
| §6 V0 | 기존 selector·view·`best_on_grid.py`(matched grid 10) — `tools/palsv18_report.py` 가 original 표와 matched-grid 표를 따로 낸다 |
| §7 V1 | `tools/po10_diag.py --probe-set palsv18`: zero + r{0.25,0.5,1,2}×8 방향 = 33 probe, 장면별 B_i·EPE_i 후 장면 동등 집계(`scene_level`), 반경별, 무반응 참조 EPE 0.9375, donor 대비 drift. legacy closure 는 이름 유지 |
| §8 V2 | `tools/palsv18_validate.py` `modality()`: pan_only/ms_only(upsampled M 8 band 동일 shift)/common, 기대 −e/+e/0, 장면별 fit·잔차. shortcut(MS swap·상수 MS·bilinear·padding) 은 기존 `po10_diag` last 산출물 재사용 |
| §9 V3 | `native_proxy()`: `align/estimator`(Scharr→median/MAD→top-30% edge→ZNCC→quadratic; census 는 같은 추정기의 gate) 로 PAN_b←up(MS) before/after 장면별 벡터·신뢰도·identity·known-shift. `edge_profile()` = **edge_profile_v1**(기존 Part B 와 다른 이름): GT Scharr top-30% mask 오차, 16×16 cell 최강 edge, ±4 px @0.25 profile, 50% crossing·10–90% 폭, missing/multiple 기록. `energy_fr()`+`calibration_patches()`: warp 전후 Scharr energy 비, identity·ramp, train 256 patch 에서 σ 후보 {0,…,1.0} energy-match(허용 0.02; 못 맞추면 `unmatched_blur_control`) |
| §10 V4 | `interventions()`: learned/zero/wrong_sign/scene_shuffle(고정 (i+1) mod 20)/constant_calibration(train 256 patch 원좌표 c0 중앙값)/blur_energy_match(σ*, correction 0) 의 FR view; zero 의 수치 동치(aligner_enabled=False vs delta_override 0). stress: 기존 `stress_hqnr_native`(원 P / 고정 N2 참조, V96) 를 r{0.5,1,2}×4 방향(12 probe) 로 |
| §11.2 gradient | PALS24 trainer 진단 그대로(홀수 step: ρ_g, cos ψ, raw/weighted L_off, off_unet_grad_absent) — 1/1001/…/10001/25001/49001 |
| §14.1 산출물 | `tools/palsv18_report.py` → `work_dir/_palsv18_campaign/` 의 csv 13종 + `campaign_budget_ledger.json` + `final_report.md` + `alignment_evidence_summary.md`; `campaign_manifest.yaml`·`metric_contract.json`·`reuse_registry.json`·`selection_grid.json`·`initial_tensor_registry.json`·`metric_gate_report.json` 은 prepare 가. `estimator_contract.json`·`probe_manifest.json` 은 validate 산출물 안(summary 의 `estimator`/`probe_manifest`) |

실행기: `./tools/palsv18_prepare.sh`(gate → 기동), `./tools/palsv18_validate.sh pre|post|run`(V1–V4 를 best_raw·last 에, GPU 시간을 ledger `vpre`/`vpost` 로). `_upload.sh` 는 PALSV18 run 완료 시 pals24 와 같은 진단(palsv18 probe) 을 last·best·10K·ep125 에 돌리고 ledger `diag_<run>` 에 더한다. 시트 범주 ㉔ PALSV18.

## 3. gate 결과 (2026-09-13 19:25–19:28, 학습 전)

- 기존 gate 전부 통과(pa/po10/kdv/nf16/pals24 PL01–PL12), PV01–PV07 25/25 OK.
- G-M1 재평가: 대조군 9벌 + L1E2 + N2 last 의 best_hqnr 저장 checkpoint 를 같은 evaluator 에 넣어 기록과 Δ 0.0(scene 단위도 0.0). G-M2 곱의 평균 = 기록. G-M5/G-M6/G-M7 통과. 재사용 승인 9/9. N2 는 po trainer 라 초기 tensor 기록이 없어 init 검사 N/A(donor hash 는 일치).
- G-S selection grid: L1E4 S1234 의 25 후보(10 epoch 격자) + exact 50000; 신규 config eval_epoch 10 동일. NF16 P0(S1234) 만 격자 5 → matched-grid 10 값 0.95275 병기.
- G-C: seed 1234/7777/2025 의 저장 U-Net 초기 tensor sha(c988a6c9…/90cb0372…/881bdb65…) 가 각 seed 대조군 3벌의 기록과 일치. donor file sha 9d4cbf21…, aligner tensors db638b55… 일치.
- G-T smoke: step ≈ 18 ms, peak 2.8 GB, 50K 예상 1.22 h → gate 예상은 규칙대로 1.7 h.

## 4. 실행 (2026-09-13)

- **기동 19:28:44** (체인 마감 09-15 01:28), 큐 6 run. 첫 run gate: used 0.055 + 1.1×(1.7 + 1.7) + 5.0 = 8.80 ≤ 18 → RUN. `campaign_gates_enabled.txt` 의 `pals24` token 은 `_palsv18_campaign/campaign_gates_enabled.pals24.closed.txt` 로 옮겨 닫았다(PALSV18 은 조건부 gate 없음).
- gate 실측 0.055 h(예약 0.8). PALS24 실측(1.41–1.46 h/run + 진단 0.15 h) 기준 6회 ≈ 9.5 h → 09-14 05:00 께 완료 예상.
- V-pre(대조군 11 checkpoint 쌍의 V1–V4) 는 aligner 전용 추론 + FR 20장 재추론이라 학습과 겹쳐 돌리고 벽시계 시간 전부를 ledger `vpre` 에 계상한다(계획 §4.1 예약 2.0).
- 검증 도구 실전 시험(PALS24 L1E4 S1234 best_raw, 학습과 겹쳐 실행): V2 fr512 — pan_only B ≈ diag(−0.51, −0.51), ms_only ≈ diag(+0.50, +0.47), common ≈ diag(−0.01, −0.01)·EPE 0.018 (무반응 참조 0.9375) → aligner 는 PAN–MS **상대** 위치에 반응한다(학습하지 않은 MS-only 변환에도 부호가 맞는다). native64 도 같은 방향(−0.44/−0.57, +0.37/+0.43, ≈0).
  V3.3 — bicubic warp 는 Scharr energy 를 **줄이지 않는다**(FR 비 1.028, calibration 1.005) → σ 후보가 target 을 맞추지 못해 `unmatched_blur_control`(계획 §9.3 의 예외 경로; blur 개입은 돌리지 않고 상태만 기록). identity 0.0, ramp +1.000. constant_calibration 벡터(train 256 patch c0 중앙값) = (+0.136, +0.077) px.

## 4.1 2026-09-13 저녁 리뷰(7건) 반영 — 첫 run 중단·재기동

사용자 리뷰(P1 2건, P2 5건)를 받아 19:54 에 V-pre 와 첫 run(`PALSV18_L3E5_…_S1234`, 14.5K update) 을 멈추고 아래를 고쳤다. 중단한 시도는 처음부터 다시 돈다(재개 아님); 소요 0.42 h 는 ledger 에 `KILLED_FOR_FIX` 로 계상했고 부분 산출물은 `work_dir/_palsv18_campaign/killed_PALSV18_L3E5_S1234_attempt1/` 에 두었다.

| # | 지적 | 반영 |
|---|---|---|
| P1-1 | 평가 checkpoint 가 selector 동률 밖에서 삭제됨(계획 §6.3 위반) | `kdv.select.retain_all_candidates: true`(PALSV18 config 전부) → `train_kdv._select` 가 후보를 지우지 않는다(`candidates/step-*` 25개 + exact 50K 보존; run 당 ≈ 0.8 GB). 첫 run 은 이미 2020·4040 을 잃어 폐기·재실행 |
| P1-2 | matched-grid 재선택이 selector·격자와 다름(`best_on_grid`: epoch%10, exact 50K 제외, HQNR 최대만) | `palsv18_report.matched_grid_best`: `selection_grid.json` 의 실제 25 update(50000 포함) 위에서 `pa.selector.BestSelector` 재생(HQNR 1e-4 → fSCC 1e-4 → 늦은 update). P0 S1234 → 38380 / 0.952723(리뷰 값과 일치). 대응 차이·performance_leader 는 matched-grid 값, original 은 병기 |
| P2-3 | stress 장면별 csv 덮어씀 · 실행기가 JSON 존재만 보고 생략 | `stress_hqnr_native(tag=…)` → `stress_hqnr_native_fr512_<tag>.csv`; 실행기는 checkpoint sha·`tool_version`·완료 parts 를 모두 확인할 때만 생략(`need_run`/`need_v1`), po10_diag 산출물에 `ckpt_sha256` 기록 |
| P2-4 | stress 4 방향, RR GT stress 없음, c0−ε 기준선 없음, best_raw shortcut 대조 없음, 상수 MS 가 전체 단일 평균, 개입 평균이 부적격 장면 포함 | stress 8 방향(HR px {0.5,1,2}) + `fr_stress_baseline`(c0−ε, native 참조 V96) + `rr_stress`(GT 고정, ROI margin 32, learned·기준선) · shortcut 대조를 best/last 둘 다 · 상수 MS = band 별 공간 평균 · 개입은 raw_original 전체·V64 view 는 적격 장면만(0 이면 None) + n_eligible 기록 |
| P2-5 | native proxy 벡터 방향이 모델 correction 과 반대 · identity/known-shift 가 JSON 에 없음 | `canon_* = −audit`(PAN→MS 보정 방향) 병기, 모델 c 와의 cos 기록, identity·known-shift(+1 px → audit +1.007 → canonical −1) 를 summary·`estimator_contract.json` 에 |
| P2-6 | 검증 포함 18 h 상한 미보장 · V-pre 병행 · V-pre 예약 미반영 · ledger 잠금 없음 | V-pre 를 학습 **전** 에 순차로 돌린다(계획 순서). 실행기 guard: `used(예약 포함) + 이 checkpoint 예상 + 1.7×미완 run + 1.5 ≤ 18` 아니면 시작하지 않음. `vpre_reserved`/`vpost_reserved` 항목으로 미수행 예약을 ledger 에 잡아 학습 gate 가 본다. ledger 갱신은 trainer(`_LedgerLock`)·`_upload.sh`·실행기 모두 `tools/_ledger_update.py`(flock) |
| P2-7 | exact resume 미충족 · gradient 진단이 현재 batch·마지막 active update 49999 미측정 | `kdv.diag.fixed_batch: true` → 첫 학습 batch 를 고정해 두고 diag step 마다 별도 graph 로 §9.3 분해(`gradient_diagnostics_fixed.jsonl`: ρ_g, cos ψ, off_unet_grad_absent, 합산 규칙 오차; ε 전용 generator, fork_rng, EMA/optimizer 불변 — dry run 으로 학습 loss 열 동일 확인) · 진단 step 에 num_iter−1(49999) 추가. **exact resume 은 미구현**: sampler 순서 복원이 없어 재개는 `resumed_nonexact` 로 남고, 보고서가 `resumed_nonexact` 열로 구분한다(재개된 run 은 무중단 run 과 같은 표에 두되 표시). P0 S1234 의 eval 5/10 차이도 matched grid 로 완전히 제거되지 않음을 표 D 에 남긴다 |

### 4.2 trainer 변경 검증 (dry run 24 update × 3)

`fixed_batch` on / off / off(재실행) 세 벌: 마지막 가중치 sha 가 셋 다 다르고, 진단 값의 run 간 차이는 **on–off 와 off–off 가 같은 크기**(step 1 에서 3e-8, 23 에서 1e-5~3e-5)다 — 이 GPU 에서 학습 자체가 bit-identical 하지 않으며(CUDA grid_sample backward 비결정성, 계획 §13.3), 고정 batch 진단이 그 잡음 위에 추가 편차를 만들지 않는다.
고정 batch 기록은 diag step 전부(0,1,4,5,…,num_iter−1) 에 남고 `sum_rule_max_abs_err` 0.0, `off_unet_grad_absent` True. 후보 checkpoint 는 selector 동률 밖이어도 남는다(`candidates/step-24`). ledger lock 파일 생성 확인.

### 4.3 재기동

V-pre(대조군 11 checkpoint 쌍의 V1+V2, 0.13 h; 이미 계산된 것은 sha·도구 버전 확인 뒤 생략) → `palsv18_prepare.sh` → **2026-09-13 20:49 재기동**(마감 09-15 02:49). 첫 run gate: used 0.61(gate 0.06 + 중단분 0.42 + V-pre 0.13) + 1.1×(1.7+1.7) + 5.0 = 9.35 ≤ 18 → RUN. prepare 는 `nf16_unit_tests` 의 stub 객체가 `args` 를 갖지 않아 두 번 실패했다(`is_diag_step` 을 stub-safe 로 고침).

## 5. 판정 규약·주의

- 주 판정 best_raw raw_original HQNR → fSCC(원 PAN 참조), 판정선 0.0031 은 raw HQNR 에만. `performance_leader`(3-seed 평균 선두)·`working_reference`(L1E4)·`alignment_evidence`(V1–V4 근거 범위) 를 분리한다(§12.1). 판정선 안이면 L1E4 를 바꾸지 않는다.
- 3 seed 는 같은 donor·같은 test 20장에 조건부인 downstream 반복이다. native proxy 는 센서 GT 가 아니며 secondary 독립 추정기(phase-correlation) 는 없다 → `not_available`.
- 결과 문서는 캠페인 종료 뒤 **09-14 s1 문서**로 쓴다.
