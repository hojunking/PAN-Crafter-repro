# s2 W112·D123 KD·variance·aligner 재사용 — 계획 검토와 구현 노트 (2026-09-10)

> **2026-09-10 저녁 사용자 결정: depth [1,2,3]** (계획 원안 [1,2,4]). 이 노트의 W112·D124 표기는 계획 원안·dry run 시점의 것이고, 실행 config·큐·이름(`S2W112D123_*`)은 D123 이다. backbone 2.6589 M (s1 PO10 R200 block 과 같은 골격).

계획 묶음: `research_log/PAN_S2_W112_KD_Variance_Plan_and_References_2026-09-10/`
(캠페인 문서 `PAN_S2_W112_D124_KD_Variance_Adaptive_Campaign_2026-09-10.md`, 참조 구현 `pan_gt_anchored_kd.py` · `pan_s2_w112_stat_kd_reference.py`, README, 검사 JSON).
사용자 지시: "현 workspace 기준으로" 검토·구현. 이 문서는 (1) 계획의 REQUIRED/가정을 저장소 실제와 대조한 검토, (2) 구현 지도(implementation map), (3) gate 결과, (4) s2 실행 절차다.

---

## 1. 검토 — 계획의 REQUIRED · 가정 ↔ 저장소 실제

| 계획 항목 | 저장소 실제 (2026-09-10) | 처리 |
|---|---|---|
| backbone class·폭·깊이 (§2.2 REQUIRED) | `model.pancrafter_paper.PANCrafterPaper`, `hidden_size 112`, `in_mode paper`(cat(PAN, ↑MS) 9ch), `mode_modulation false`, `attn_locations []`, norm **LN** → GN 나눗셈 문제 없음. backbone **D123 2.6589 M**(실행) / D124 2.8854 M(계획 원안). aligner GN4/GN8 은 폭과 무관 | `architecture_manifest.json` 을 run 마다 자동 기록 (M01) |
| **depth** | 계획 §0.2 "Depth [1,2,4] **유지**". 그러나 같은 날 사용자 결정으로 s1 PO10 R200 block 은 **W112·D123**(2.6589 M) 이다 | **사용자 결정(2026-09-10 저녁): D123.** 생성기 기본 `--depth 1,2,3`, init `work_dir/_kdv_init_w112_d123`, 이름 접두 `S2W112D123`(계획의 `S2W112` 에 depth 를 붙여 D124 와 구분). s1 PO10 R200 과 같은 골격 |
| W112 Teacher checkpoint (§3.2 REQUIRED) | **없다.** 저장소의 W112 run 은 s1 PO10 R200 block(진행 중, W112·D123, N 계열) 뿐 | Q01 에서 **새로 학습**: `T112_DONORFROZEN_REC`(donor aligner frozen + W112·D124 복원 supervised, §3.3). Student 와 seed 가 같으면 Q02 와 동일 모델이 되므로 Teacher 는 **seed 2025**, Student 는 **seed 1234** (계획 §17.2 "우선 1234"). teacher_id `T112_v01` |
| aligner donor (§3.1) | s1 `PA_A1_REC_W96_D124_9CH_S2025/best_hqnr`(step 34,340, HQNR 0.9492) 의 `aligner.*` 22 tensor·105,330 params. 독립 dual-stem 구조라 폭과 무관. s2 에는 없으므로 저장소에 asset 으로 넣었다: `assets/donor_aligner/PA_A1_REC_W96_D124_9CH_S2025_best_hqnr_aligner.pt`(0.43 MB, manifest json 동봉, sha256 기록) | strict load (key·shape). 알려진 성질(입력 무반응 상수 보정 ~0.2 px, 반응 기울기 −0.02~−0.09)을 manifest 에 적었다 — 계획 §3.4 "상수라는 것만으로 실패 확정 않음" |
| Teacher/donor 출처 분리 (§3.1) | donor = W96 A1 aligner, KD Teacher = Q01 W112 (같은 donor 를 frozen 으로 씀) | `init_and_teacher_hashes.json`·`teacher_registry.json` 에 별도 기록 |
| 입력 순서·잔차 base (§0.3) | `PAModel`: `y = bicubic↑MS + backbone(cat(P̃, ↑MS))`, base 정확히 1회 (gate M04) | 그대로 |
| sampler (§5.1) | `pa.warp.warp_pan`: bicubic/border/align_corners=False/FP32, Δ=0 도 sampling | 그대로. **A-ID 만** sampler 부재 (`PAModel(sampler=False)`, B0 와 같은 그래프 — gate M04 로 동치 확인) |
| training profile (§17.2 PROFILE-INHERIT) | PA/PO 와 같은 resolved 값: AdamW 1e-4 / wd 0.01 / betas 기본 (0.9,0.999) / eps 1e-8 / warmup 100 → cosine / batch 48 / 50K / **fp32**(mixed_precision 없음) / gradient clipping 없음 | 그대로 계승. A-FT/A-SC aligner LR = backbone LR |
| 평가·선택 (§18) | `PATrainer.test_full` 세 view(raw_original / raw_valid / aligned_valid, fixed V64) + `BestSelector`(running max, 1e-4 band, fSCC, later step) + `best_hqnr`(=best_raw alias)·`best_aligned`·`last` | 그대로 + **`best_rr_val`**(검증셋 valid_wv3.h5 1,080 patch 의 plain ERGAS 최소, 동률 1e-4 → later step) 추가. FR mat20 로 고른 best 는 계획 §18.1 대로 **test-adaptive exploratory** 로 표기 |
| 지표 (§19.1) | 시트 = `tools/eval_fr_paperset.py`(FR·paper mat20: HQNR↑·HQNR(V64)↑·JQM↑), RR 는 MATLAB 포팅. RR-val ERGAS 는 dim_cut 없는 plain 정의 — **선택 전용**, 시트 ERGAS 와 다른 값 | kdv trainer 를 evaluator·pa_diag·시트 비용 모델·범주(⑳ KDV)에 연결 |
| 참조 loss (§16.5) | 두 참조 파일을 **그대로** `kdv/losses_rec.py`·`kdv/losses_stat.py` 로 옮겼다(수식·검증 동일, unittest 는 `tools/kdv_unit_tests.py` 로). 22 검사 전부 통과(torch 2.4.0, s1) | T-only 통계는 별도 식(`stat_term(mode='T')` = ⟨\|S_V(S)−sg S_V(T)\|⟩), fixed_kd 를 T 라고 쓰지 않음 |
| intensity 단위 (§6.1 "[0,1] 가정") | feeder 는 **[−1,1]**(range 2). tau 는 데이터로 calibration 하므로 단위 자체는 문제 없음; `eps_R 1e-6`·`eps_V` 하한도 상대적으로 무시 가능 | `calibration_resolved.json` 에 단위 기록 |
| 이동량 covariance KD G1–G5·G-EQ·G-STRUCT (§11) | **구현(§9)** — 출처 eq_closure/geo_curvature/struct, native step·trainable aligner 만. 실행은 보류 큐 | G-CORR/XVIEW 만 resolver 가 막는다 |
| Teacher cache (§17.6) | online Teacher (cache 없음) | — |
| 24h 운영 (§14) | `campaign_start.sh --hours 36`(마감은 새 case 시작 억제만) — 24h 는 review 시점 | 요청 시 보고(상시 tracking 없음) |

**계획의 가정 중 확인된 것**: 참조 코드 22 검사 통과, aligner 105,330 일치, GN 문제 없음(LN), W96→W112 partial load 가 strict 로 구조적으로 불가(gate M02), 9ch 입력·8ch 출력·base 1회.
**계획이 채우지 못한 REQUIRED 를 채운 결정**: Teacher seed 2025 / Student seed 1234, donor = s1 A1 seed 2025 aligner, λ_V pilot = Q02 `last`, calibration set = train 3,072 patch(증강 없음, seed 1234 고정 부분집합), 평가 주기 = PA 와 같은 eval_epoch 5(≈1K update), depth D124(위 표).

---

## 2. 구현 지도 (implementation map, 계획 §16.2 제안 경로 → 실제)

| 계획 모듈 | 실제 파일 | 내용 |
|---|---|---|
| model_adapter | `pa/model.py` `PAModel(backbone, aligner=None, aligner_margin, sampler=True)` + `kdv/forward.py` `kdv_forward` | aligner 없음/ sampler 없음(A-ID) 확장(기본 인자에서는 이전과 동일). 학습 forward 와 평가 forward 가 같은 연산(gate E01). A-FR 은 aligner·warp 를 no_grad 로, Teacher 와 correction·P̃ 공유 |
| teacher_assets | `kdv/teacher_assets.py` | `load_donor_aligner`(strict), `load_run_model`(run config 로 skeleton → strict), `freeze`, `state_hash`, `assert_param_disjoint`, `skeleton_from_cfg`(evaluator·시트·smoke 공용) |
| input_protocol | `kdv/protocol.py` | I-A / I-N(홀수 update corrupt, ε 원판, 전용 generator, checkpoint 에 RNG 포함) / I-NATIVE-TRANSFER |
| losses_rec | `kdv/losses_rec.py` | `GTAnchoredReconstructionKD` (참조 그대로) |
| losses_statistics | `kdv/losses_stat.py` | `statistic_map`(IV/GV/GC/SC), `statistic_loss`(참조), `stat_term`(H/T/FIX/WH/AD, EDGE 는 trainer 에서 `pa.losses.output_edge_loss`) |
| losses_geometry | `pa/losses.py`(A2 edge, A3 geo) · `pa/offset.py`(N offset) 재사용 | frozen aligner 면 geo/offset 은 optimizer loss 에서 제외(resolver `disabled_terms`) |
| covariance_* | `kdv/alignment_kd.py` (§9) | eq_closure / geo_curvature / struct 출처, G1–G5·G-STRUCT, CovHead·KL |
| calibration | `kdv/calibration.py` | tau_R · tau_V/eps_V · λ_V(pilot, r_grad 0.05) · json 캐시 `work_dir/_kdv_calibration/<teacher_id>/` |
| evaluation_adapter | `train_pa.PATrainer` 상속 | 세 view·선택기·export 그대로 + `best_rr_val` + `fitting_bins.csv`(§19.2 Teacher error Q0–50/50–90/90–100 의 Δe·WinRate, pixel·stat) |
| phase_manager | `train_kdv.KDVTrainer._warm_start` + `main.py` warm-start 분기 | `kdv.phase{parent_run, parent_tag, parent_step, optimizer_state_policy}` → backbone/optimizer(backbone group 위치 대응)/scheduler 위치 복원, `parent_and_phase.yaml`(config diff·inherited cost) |
| run_registry | `kdv/registry.py` | case 등록·resolver(잘못된 조합 오류)·이름 규칙·서술 |
| trainer | `train_kdv.py` `KDVTrainer` (`trainer: kdv`, `--kdv` YamlAction) | 아래 §3 |
| config/queue | `tools/gen_kdv_configs.py` → `config/S2W112D123_*.yaml` 9벌, `config/queues/kdv_s2.txt` | §4 |
| gates | `tools/kdv_unit_tests.py`, `tools/smoke_cases.py`(kdv 분기), `tools/pa_unit_tests.py`(evaluator·selector) | §5 |
| 기동 | `tools/kdv_prepare.sh` | 환경 json → 데이터·donor·지표·gate → config → smoke(Q00–Q02) → 캠페인 manifest → `campaign_start.sh` + 감시자 |
| 시트·평가기 | `tools/eval_fr_paperset.py`, `tools/pa_diag.py`, `gspread/gspread_upload.py`(비용·descriptor·notes), `gspread/sheet_categories.py`(⑳ KDV), `tools/_upload.sh`(S2W112D123_* → pa_diag) | |

### 2.1 run 폴더 산출물 (계획 §21.1 대응)

`architecture_manifest.json` · `init_and_teacher_hashes.json` · `kdv_config_resolved.json` · `calibration_resolved.json` · `parent_and_phase.yaml` · `dataset_hashes.json` · `baseline_manifest.json` ·
`train_log.jsonl`(rec hard/soft, d/a 평균, soft-positive·student-win 비율, stat, λ_V, aux, Δ, grad norm, step 시간, peak) · `gradient_diagnostics.jsonl`(1K 마다: F_S 의 rec/stat gradient norm·cosine, aligner gradient, ∂L_rec/∂Δ) ·
`checkpoint_metrics.csv`(세 view + RR + RR-valid + **rr_val_ergas**) · `scene_metrics.csv` · `delta_predictions.csv` · `fitting_bins.csv` · `memory_and_throughput.json` ·
`best_hqnr/`(=best_raw) · `best_aligned/` · `best_rr_val/` · `last/` (+ `*_meta.json`) · `results/{reduced,full}_<tag>.mat` · `results/fr_mat20.json`(시트) · `results/pa_diag.json`.
캠페인 폴더 `work_dir/_kdv_campaign/S2_W112_KDV_20260910/`: `campaign_manifest.yaml`, `environment_<server>.json`, `teacher_registry.json`, `runs.csv`(상태 행 누적), `smoke_<server>.log`, `implementation_map.md`(이 문서 사본).

---

## 3. trainer 동작 (train_kdv.py)

한 update:
```
P_view, ε, corrupted = prepare_view(P, protocol, t)            # I-A: P 그대로
o = kdv_forward(student, teacher, P_view, M, ...)               # Ŷ_S (graph), Ŷ_T (no_grad; A-FR 이면 같은 Δ·P̃ 공유)
L_R   = GTAnchoredReconstructionKD(tau_R, α_R 1, β_R 0.1, mode)(Ŷ_S, Ŷ_T, Y)     # N0 이고 Teacher 불필요면 plain L1
L_V   = stat_term(Ŷ_S, Ŷ_T, Y, kind, window 5, mode, criterion(tau_V, eps_V))     # H/T/FIX/WH/AD
L_aux = λ_E(t)·edge + λ_G(t)·geo(trainable 만) + λ_off(t)·offset(I-N corrupt·trainable 만) + λ_GKD·G1(native·trainable 만)
L     = L_R + λ_V·L_V + L_aux
```
- Teacher forward 는 **필요할 때만**(R0–R3, T/FIX/WH/AD, G1). Q02 (A-FR·N0·OFF) 는 donor aligner 만 쓰고 F_T 를 돌리지 않는다(§6.3 wall-time 기록: `memory_and_throughput.json`).
- calibration 은 trainer 시작 시 자동: `tau_R = max(median e_T, 1e-6)`, `tau_V = max(median E_T^V, 1e-3·v_scale, 1e-12)`, `eps_V = max(1e-3·tau_V, 1e-12)`, `λ_V = 0.05·median(RMS ∂L_rec/∂Ŷ ÷ RMS ∂L_V^H/∂Ŷ)` — pilot = `stat.lambda_pilot`(Q02 `last`). 통계 gradient ≈ 0 이면 `CALIBRATION_DEGENERATE` 로 exit 4(체인은 재시도 안 함). 같은 representation 의 H/T/FIX/WH/AD 는 같은 λ_V(같은 캐시 key).
- 고정성 검사: 매 평가마다 Teacher state hash·A-FR aligner hash 불변 확인(다르면 RuntimeError). optimizer ∩ Teacher/frozen aligner = ∅ (시작 시).
- 실패: 비유한 loss → exit 3, |Δ| > 8 px 또는 geometry support 실패 → `support_fail.json` + exit 4.
- `runs.csv` 에 RUNNING / EVAL / FINISHED_TRAIN / NAN / SUPPORT_FAIL 행.

**A-ID(Q00)** 는 `PAModel(sampler=False)` — U-Net 이 raw PAN 을 그대로 받는 B0 와 같은 그래프이며 결과 폴더·평가·시트 경로는 다른 kdv run 과 같다(세 view 에서 Δ=0 이므로 aligned_valid = raw_valid).

---

## 4. 큐 (계획 §13.2 Q00–Q08) — 약명 → 세팅

이름 규칙 `S2W112D123_<recipe>_<input>_<aligner>_<rec>_<stat>_<geomKD>_s<seed>_<version>`(접두 = 서버+폭+depth, `kdv.registry.arch_prefix`). 공통: **W112·D123**(2.6589 M) · 9ch · I-A(native) · 50K · AdamW 1e-4 · batch 48 · fp32 · 평가 eval_epoch 5.

| Q | run | recipe / aligner | rec | stat | Teacher | 목적 |
|---|---|---|---|---|---|---|
| Q00 | `S2W112D123_NOALIGN_IA_AID_N0_OFF_G0_s1234_v01` | aligner·sampler 없음 | N0 (L1) | OFF | — | 새 폭 독립 baseline |
| Q01 | `S2W112D123_T112DFR_IA_AFR_N0_OFF_G0_s2025_v01` | donor frozen (T112_DONORFROZEN_REC) | N0 | OFF | — | **Teacher** (seed 2025) → `T112_v01` |
| Q02 | `S2W112D123_A1_IA_AFR_N0_OFF_G0_s1234_v01` | donor frozen | N0 | OFF | — | frozen aligner GT-only 기준 (Teacher 와 seed 만 다름) · λ_V pilot |
| Q03 | `S2W112D123_A1_IA_AFR_R1_OFF_G0_s1234_v01` | donor frozen | R1 hard-only | OFF | T112_v01 | Teacher-error hard 재가중 |
| Q04 | `S2W112D123_A1_IA_AFR_R3_OFF_G0_s1234_v01` | donor frozen | R3 adaptive | OFF | T112_v01 | adaptive rec 기준 |
| Q05 | `S2W112D123_A1_IA_AFR_R3_GVH_G0_s1234_v01` | donor frozen | R3 | GV-H (GT gradient-variance 5×5) | T112_v01 | GT 구조 통계 추가 |
| Q06 | `S2W112D123_A1_IA_AFR_R3_GVAD_G0_s1234_v01` | donor frozen | R3 | GV-AD | T112_v01 | **첫 주력 후보** |
| Q07 | `S2W112D123_A1_IA_AFT_R3_GVAD_G0_s1234_v01` | donor 초기화 후 학습 | R3 | GV-AD | T112_v01 | aligner fine-tune |
| Q08 | `S2W112D123_A1_IA_ASC_R3_GVAD_G0_s1234_v01` | 독립 초기화 학습 | R3 | GV-AD | T112_v01 | 초기값 제약 확인 |

Q07/Q08 의 `STAT-*` 는 계획 "Q04~06 중 고정 선택" → **GV-AD**(§1 첫 주력 후보) 로 정했다. 순서 의존: Q01 → Q03–Q08(Teacher), Q02 → Q05–Q08(λ_V pilot). 체인(`_run_cases.sh`)이 큐 순서대로 돌고 case 직전 smoke 가 Teacher 존재를 확인한다.
Q09–Q25(R0/R2, GV-T/FIX/WH, IV/GC/SC/EDGE, G1, I-N, seed 반복 …)는 `tools/gen_kdv_configs.py build_kdv()` 에 case 를 더해 만든다(resolver 가 조합을 검사).

예상 소요(s1 4090 실측 smoke 기준, s2 에서 다시 잰다): W96 A1 run 1h31m(학습 46m + 평가 45m) 대비 W112 학습 ≈ ×1.4, Teacher forward 가 있는 run 은 학습 +30–40% → run 당 약 2–2.5 h, 9 run ≈ 20 h.

---

## 5. Gate 결과 (s1, 2026-09-10)

| Gate | 내용 | 결과 |
|---|---|---|
| L01 | 참조 rec 10 검사 + 8-band gradient 방향 | `tools/kdv_unit_tests.py` 전부 OK |
| L02/L04 | 참조 stat 12 검사 + spectral cov row-chunking = explicit unfold, 음의 off-diagonal 보존 | OK |
| M01 | W112·D123 2.6589 M(캠페인 골격; D124 2.8854 M 참고), 첫 conv 9ch, GN 없음, 이름 접두 S2W112D123 | OK |
| M02 | donor strict(105,330·22 key) · W96→W112 strict load RuntimeError · non-aligner state KeyError | OK |
| M03 | Student step 뒤 Teacher hash 불변 · optimizer ∩ Teacher → RuntimeError | OK |
| M04 | y = base + residual(1회) · A-ID == B0 forward | OK |
| W01/W03/R01 | Δ=0 sampling identity · I-N 홀짝·순차 두 warp ≠ 단일 warp · corruption RNG roundtrip | OK |
| L03 | A-FR: Δ 공유·aligner/Teacher grad 없음·F_S grad 있음 · A-FT: sampler 통해 aligner grad · geo-only backward 에 F_S grad 없음 | OK |
| E01 | 학습 forward == 평가 forward(PAModel) | OK |
| resolver | 9개 잘못된 조합 오류, A-FR 의 geo 항 진단 전용 | OK |
| calibration | λ_V 규칙·degenerate 판정·v_scale | OK |
| P01 smoke | D124 dry config Q00/Q01/Q02: peak 2.69 GB, step 52–94 ms · **D123 실행 config Q00/Q01/Q02: peak 2.66 GB, step 53–58 ms** (PO10 학습과 GPU 공유 중 측정) | OK |
| dry run (300 updates) | §6 | 아래 |

---

## 6. s1 dry run (300 updates, 평가 1회, export 4 tag)

**dry run 은 depth 결정 전 D124 골격(`S2W112_*_dry`)으로 돌렸다.** 경로 검증 목적이라 D123 에서 반복하지 않았고, D123 실행 config 는 §5 P01 smoke 로 확인했다.

s1(RTX 4090, PO10 N1 학습과 GPU 공유 중) 에서 `tools/gen_kdv_configs.py --updates 300 --version dry` 로 만든 7 run 을 순서대로 돌렸다(15:03–15:21, run 당 2.5 min: 학습 20–60 s + FR 평가 1회 + export 4 tag ×2 mat).
전부 rc=0, `results/{reduced,full}_{best_hqnr,best_aligned,best_rr_val,last}.mat` 생성, `tools/eval_fr_paperset.py`(HQNR·V64·JQM)·`tools/pa_diag.py`·시트 `--dry-run`(범주 ⑳ KDV, 비용 2.8854 M, descriptor) 동작 확인.
**300 update 의 지표는 pipeline 검증값이지 성능이 아니다** (아래 HQNR 은 인용 금지).

| run(dry) | t_step | peak | 확인한 것 |
|---|---:|---:|---|
| NOALIGN A-ID N0 (Q00) | 57 ms | 3.1 GB | aligner·sampler 없음 → Δ=0, aligned_valid = raw_valid, 세 selector·4 tag export |
| T112DFR A-FR N0 s2025 (Q01) | 63 ms | 3.1 GB | donor frozen, Δ = donor 상수 (+0.228, −0.057) — s1 A1 결과와 같은 값. Teacher 산출(`best_hqnr`) |
| A1 A-FR N0 (Q02) | 60 ms | 3.1 GB | Teacher forward 없음(needs_teacher False), λ_V pilot(`last`) 제공 |
| A1 A-FR R3 GV-H (Q05) | 140 ms | 3.2 GB | **calibration**: τ_R 0.0199(e_T p50; p90 0.068), τ_V 5.14e-4(v_scale 7.8e-4, GT 통계 양수 비율 1.0), λ_V 0.94(g_rec/g_V 비 18.8 × 0.05, status OK) — 캐시 `work_dir/_kdv_calibration/T112_dry/`. rec adaptive 통계 d̄ 0.50·student-win 0.45·soft 4e-5, stat H 1.4e-3, `fitting_bins.csv`(pixel·stat Q0-50/50-90/90-100), `gradient_diagnostics.jsonl`(rec/stat F_S gradient cosine 0.66), correction 공유(share_correction True) |
| A1 A-FR R3 GV-AD (Q06) | 150 ms | 3.2 GB | 같은 calibration 캐시 재사용(from_cache), stat AD d̄_V 0.48, frozen aligner hash 불변 |
| A1 A-FT R3 GV-AD (Q07) | 157 ms | 3.4 GB | aligner 학습됨(hash 변경, Δ drift 0.14 px vs Teacher), Teacher 는 자기 aligner(share False) |
| A1 A-SC R3 GV-AD (Q08) | 144 ms | 3.4 GB | 독립 초기화 aligner(zero head → Δ (+0.04, −0.05) 로 이동), hash 변경 |

관찰: Teacher forward 가 있는 run 은 step 시간이 ≈2.3× (60 → 140–157 ms; GPU 공유 상태의 측정이라 s2 에서 다시 잰다). step 0 진단의 aligner gradient 는 0 이다 — U-Net 의 zero_module 출력 때문에 첫 step 에는 입력(P̃) 민감도가 없다(계획 §16.6 의 주의와 일치); 이후 step 에서 aligner 가 움직인 것은 hash·drift 로 확인했다.
dry run 산출물은 `work_dir/S2W112_*_dry/`(config 는 저장소에 넣지 않음) 에 남겨 두었다 — 지워도 된다.

---

## 7. s2 에서 실행

```bash
git pull
./tools/kdv_prepare.sh --no-start        # 환경 json·donor sha·지표·gate·config·smoke(Q00–Q02)·캠페인 manifest
./tools/kdv_prepare.sh                   # 위 + 체인 기동 (queue config/queues/kdv_s2.txt, 마감 36h, 감시자 cron)
```
`gspread/server.txt` 가 `s2` 여야 한다. 진행 상황은 요청 시 보고: `work_dir/cases_chain.log`, `work_dir/_kdv_campaign/S2_W112_KDV_20260910/runs.csv`, 각 run 의 `checkpoint_metrics.csv`.
Teacher 가 바뀌면(teacher_id) 새 cohort — 기존 run 의 target 을 덮어쓰지 않는다(§3.2). 중간 변경은 `kdv.phase`(parent_run/parent_step) 로 child 를 만들고 같은 parent 의 continuation control 을 큐에 남긴다(§15).

## 9. 이동량 covariance KD (계획 §11) — 구현 (2026-09-10 저녁 추가)

사용자 결정: G-STRUCT → G-EQ → G2/G3/G4 → G-GEO → G5 순으로 **구현은 지금**, **실행은 보류**(PO10 N2/N3 의 반응하는 aligner 와 Q07/Q08 결과 뒤). 코드 `kdv/alignment_kd.py`, calibration `kdv/calibration.py`(`calibrate_covariance`·`teacher_precision_fn`·`calibrate_covhead`), trainer 통합 `train_kdv.py`, 도구 `tools/kdv_teacher_covhead.py`.

### 9.1 정의 (좌표 μ=(dy,dx) HR px, 모든 2×2 대수 float64)

| 항목 | 식 | 구현 |
|---|---|---|
| mean-KD | d_μ = μ_S − sg(μ_T), L_G = ½·mean_b[valid_b · d_μᵀ M_b d_μ] (invalid 는 weight 0, 분모는 batch 전체 §11.3-5) | `mean_kd_loss`, `kd_matrix` |
| G1 | M = k0·I, k0 = median_train tr(Π_T)/2 (출처 calibration) 또는 `geom_kd.k0` 명시(`G1K`) | |
| G2 / G3 / G4 | M = (tr Π_T/2)·I / Π_T / diag(Π_T) — G4 는 Σ 의 off-diagonal 제거가 아니라 **precision 의 대각 유지** (§11.5) | |
| G-STRUCT | M = J_Y / k_struct, J_Y = mean_{p,c} ∇Y_c ∇Y_cᵀ (Scharr, 경계 1px 제외), k_struct = median_train tr(J_Y)/2 | `structure_tensor_mean` |
| G5 | KL(N(μ_S,Σ_S) ‖ sg N(μ_T,Σ_T)) = ½[dᵀΣ_T⁻¹d + tr(Σ_T⁻¹Σ_S) − 2 + log det Σ_T − log det Σ_S]; Σ = LLᵀ + 1e-4·I, L = [[softplus a, 0],[c, softplus b]] (초기 Σ=I), head 입력 = aligner 64-d GAP feature | `CovHead`, `gaussian_kl`, `PANGlobalAligner(return_features=True)` |

Π_T 출처 (`geom_kd.covariance_source`, 모두 frozen Teacher 에서 no_grad 로 매 native step 계산):

| 출처 | 식 | 안전 규칙 |
|---|---|---|
| `eq_closure` (§11.4) | probe ε_k 16개(8 방향 × 반경 {0.5, 1.0} px, 고정 격자): r_k = μ_{T,ε_k} + ε_k − μ_{T,0}; Q = mean_k r_k r_kᵀ (= C + bbᵀ); **Π = (Q + σ_min²I)⁻¹**, σ_min 0.05 px | 고유값 자동 cap 1/σ_min² = 400; probe support guard margin max(m_A, 4); Cov(μ,ε) 자체는 쓰지 않음. 반응 없는 aligner 는 r_k = ε_k → Q = 0.3125·I → Π ≈ 2.8 (신뢰 낮음). 절대 정합 posterior 아님 |
| `geo_curvature` (§11.2) | A3 residual r(δ) = vec(√(w_Y/Σw_Y)·(Q(g_P̃)−Q(g_Y))) (`pa.losses.geometry_residual`, ‖r‖² = L_geo 의 sample 값), J = ∂r/∂δ 를 중심 유한차분 h=0.05 px 로, H = JᵀJ, Σ_dir = s²/(λ+γ) + σ_min² | rank 규칙(§11.3-1): λ < max(τ_abs, τ_rel λ_max) 방향은 precision **0** (damping 확신 금지). τ_abs = 1e-3·median λ_max, γ = 1e-2·median λ_max, s² ∈ [p1, p99] clamp — train calibration manifest 에 고정. **C01**: FD Jᵀr ↔ autograd ∇½‖r‖² 상대오차 ≤ 0.1, h/2·2h 일관성 ≤ 0.2 아니면 `GEO_JACOBIAN_MISMATCH` 로 시작 거부 |
| `struct` (§11.7) | 위 J_Y | G-STRUCT 전용 |

λ_GKD: `outer_weight: calibrate` → **λ = r_gkd / k0** (r_gkd 0.1; struct 는 k0 정규화가 이미 들어가 λ = r_gkd). 같은 Teacher·출처의 G1/G2/G3/G4 는 같은 k0·λ 를 쓴다(§11.5). G5 도 같은 규칙(λ = r_gkd/k0) — KL 의 Mahalanobis 항이 Σ_T⁻¹(≈k0 규모) 이라서.
G5 Teacher head: Teacher aligner feature → Σ_head 를 출처 Σ_T = Π_T⁻¹ 에 KL(N(0,Σ_head)‖N(0,Σ_T)) 로 calibration set 3 epoch 학습 후 freeze (`work_dir/_kdv_calibration/<teacher_id>/covhead_<source>.pt`, trainer 가 없으면 자동 생성). 학습 중 hash 불변 검사.

### 9.2 gate·진단
- `tools/kdv_unit_tests.py` C01(‖r‖² = L_geo, FD↔autograd 상대오차 0.3%·h 일관성 3.8% on 합성 blob 영상, H PSD, 평탄 영상 → rank 0·precision 0, 구조 영상 → rank 2·cap), eq(probe 격자, 무반응 aligner 의 Q = mean εεᵀ 정확, closure 0 → cap), G-STRUCT PSD, G1–G4 대수, invalid sample gradient 0·Teacher μ gradient 없음, C02(초기 Σ=I·SPD, 같은 분포 KL 0, 방향, head gradient, 비대칭), resolver(이름 G3EQ/G3GEO/GSTRUCT/G5EQ/G1K, 잘못된 조합 7종).
- 학습 중 `covariance_diagnostics.jsonl`(diag_every): precision 고유값 median·anisotropy·rank-deficient·cap·valid 비율, q 평균, Teacher closure/bias(eq), FD↔autograd 상대오차(geo, 실제 배치), Teacher head vs 출처 KL(G5). `train_log.jsonl` 에 gkd_* EMA.
- calibration 결과 `calibration_resolved.json` → `cov`(k0, 임계, 분포, status) · `covhead`.

### 9.3 큐 (보류) — `config/queues/kdv_s2_geomkd.txt`, 생성 `tools/gen_kdv_configs.py --geomkd`

| Q(계획) | run | G 모드 / 출처 |
|---|---|---|
| Q19-0 | `S2W112D123_A1_IA_AFT_R3_OFF_G0_s1234_v01` | G0 대조 (같은 trainable 정책, alignment KD 없음) |
| Q19-1 | `S2W112D123_A1_IA_AFT_R3_OFF_G1GEO_s1234_v01` | G1, k0 from geo_curvature |
| Q19-2 | `S2W112D123_A1_IA_AFT_R3_OFF_G2GEO_s1234_v01` | G2 scalar, geo_curvature |
| Q19-3 | `S2W112D123_A1_IA_AFT_R3_OFF_G3GEO_s1234_v01` | G3 full, geo_curvature (계획 §11.5 의 1차 출처) |
| Q21-1 | `S2W112D123_A1_IA_AFT_R3_OFF_G4GEO_s1234_v01` | G4 diag, geo_curvature |
| Q21-2 | `S2W112D123_A1_IA_AFT_R3_OFF_G3EQ_s1234_v01` | G-EQ = G3 with Σ from probe closure (별도 행) |
| Q21-3 | `S2W112D123_A1_IA_AFT_R3_OFF_GSTRUCT_s1234_v01` | G-STRUCT |
| Q22 | `S2W112D123_A1_IA_AFT_R3_OFF_G5GEO_s1234_v01` / `…_G5EQ_…` | G5 Gaussian KL, geo / eq |

공통: A-FT + REC-R3 + STAT-OFF (Q04 와 짝) · Teacher `T112_v01`. **실행 보류 이유**: 현재 donor(A1 aligner)는 입력 무반응 상수라 G-EQ precision 이 낮고(closure ≈ probe), G-GEO 는 영상 구조의 날카로움을 잰다(§11.3). PO10 N2/N3 가 반응하는 aligner 를 주면 그 donor(I-N·margin 4)로 다시 생성한다. 미구현: G-CORR/G-XVIEW(§12), Teacher cache(§17.6).

### 9.5 적대적 검토(2026-09-10, 4 관점 × 반박 검증 2명) 와 수정
확정된 결함과 처리: (1) **G5 Student cov head 가 첫 평가 뒤 requires_grad 가 꺼진 채 학습** → `train()` 에서 재활성. (2) **G5/geo 에서 rank 부족 sample 이 분포 KD·Teacher head calibration 에 포함되고 Σ_T = inv(Π+1e-9 I)** → `sigma_from_precision` 으로 precision 0 방향은 invalid, 분포 KD 에서 제외(비율 `gkd_rank_excluded_frac`·`excluded_fraction` 기록, §11.3-3). (3) cov head 캐시 key 에 epochs/lr 누락 → 추가. (4) **G1K(명시 k0·출처 없음) 가 resolver 통과 뒤 calibration 에서 crash** → k0·I 경로 추가. (5) 보류 큐에 **같은 trainable 정책의 G0 대조** 없음 → `S2W112D123_A1_IA_AFT_R3_OFF_G0_s1234_v01` 추가. (6) 계획 §11.5 의 1차 Π_T 출처는 **G-GEO** 이고 G-EQ 는 별도 행인데 큐가 EQ 를 1차로 썼음 → G1/G2/G3/G4/G5 를 GEO 로, G-EQ(=G3EQ)·G5EQ 를 별도 행으로 재생성. (7) warm start 가 cov_head 를 계승하지 않음 → 계승. (8) A-FT 의 Student cov head 는 Teacher head 복사로 초기화(aligner 와 같은 원칙). (9) probes.radius_hr > 2.5 면 전 sample invalid 가 조용히 KD=0 → resolver 오류. (10) geo 의 μ±h probe 고정 support 검사(§11.2) → valid mask. (11) §12 유효 적용 빈도 `gkd_applied` EMA 기록.
반박된 지적(변경 없음): eq_closure 에 rank 규칙 없음(Q+σ²I 는 항상 full-rank — 의도), λ_GKD=r/k0 를 G5 KL 에 적용, G-STRUCT 의 ½·1/k 정규화, σ_min=0·k0=0 입력(resolver 가 막음), FD h 근처 정수 μ 의 O(h) 오차(h 일관성 gate 가 잡음). 검증 단계 21개 agent 가 세션 한도로 실패해 11개 지적은 미검증으로 남았다 — 그중 실행 빈도 기록·support 검사·radius 검사·A-FT head 초기화는 비용이 낮아 그대로 반영했고, "§11.4 interpolation/crop 단서 진단(MS swap·kernel·padding) 을 KDV run 에도" 는 후속(PO10 run 에는 `po10_diag` 가 이미 수행).

### 9.4 s1 dry run (D123, 300 updates)
s1 (D123, 300 updates; Teacher = `S2W112D123_T112DFR_..._s2025_dry`, A-FT + R3 + OFF 위):

| run(dry) | 출처 | calibration | 학습 중 값 (마지막 log) | step |
|---|---|---|---|---:|
| G3EQ | eq_closure | k0 4.33 (trΠ/2 median), closure p50 0.64 vs probe 평균 0.75 → **response proxy 0.85 (무반응에 가깝다)**, cap 0 %, valid 100 % | λ_GKD 0.0231, q̄ 0.046, drift 0.09 px | 214 ms |
| G3GEO | geo_curvature | k0 15.2, λ_max p50 0.075, **C01 FD↔autograd 1.6 %/0.7 %, h 일관성 4.8 %/4.2 % (통과)**, rank-deficient 0 % | λ_GKD 0.0066, q̄ 0.18 | 88 ms |
| GSTRUCT | struct | k_struct 3.8e-3 (intensity²/px²) → Π 정규화 trΠ/2 ≈ 1 | λ_GKD 0.1, q̄ 0.012 | 78 ms |
| G5EQ | eq_closure | Teacher cov head KL 0.112 → 0.004 → 0.003 (3 epoch), head vs 출처 KL 0.056 | KL 0.118 | 219 ms |
| G5GEO (검토 수정 뒤) | geo_curvature | k0 15.2, rank 부족 제외 0 %, head KL 1.25 → 0.25 → 0.23, Student head 는 Teacher head 복사(A-FT) | KL 0.057, 적용 빈도 1.0 | — |

읽기: (1) eq_closure 의 closure ≈ probe 는 **현재 donor 가 알려진 변위에 반응하지 않는다**는 §3.4 판정과 같다 — Π ≈ 4.3 ≪ cap 400, 즉 G-EQ 는 "Teacher 를 믿지 말라"는 값을 낸다(보류 이유). (2) geo 는 FD/autograd·h 일관성 gate 를 실데이터에서 통과했지만 H 는 영상 구조의 곡률이다. (3) 16 probe 의 eq 는 step 을 +130 ms 늘린다; geo(4 residual) 는 +10 ms.

## 10. TRI-A/B/C — GT 방향·구조 통계·정합 민감성 (addendum) 검토와 구현 (2026-09-10 저녁 추가)

문서: `research_log/PAN_S2_W112_D124_TGeo_ABC_Addendum_2026-09-10.md` (v0.1-proposal). 사용자 지시: 검토 + 구현, 타당성 판단.

### 10.1 검토 (타당성)

| 항목 | 판단 | 근거·구현 대응 |
|---|---|---|
| A: R3 soft 안의 band 별 방향 gate (sign / cap / cosine), hard 불변 | **타당, 비용 0.** 국소 L1 gradient 의 부호 논증(§4.3)이 맞다. 다만 문서 §3 의 지적대로 scalar a_T 위의 세분화라 효과가 작을 수 있어 **MASS·SHUFFLE 대조가 판정의 핵심** | `kdv/tri.py direction_mask·masked_l1`, `mass_kappa`, `shuffle_mask`, `component_weights`(BANDADV) — 참고 함수 §18.1 그대로 이식, mask=1 이면 R3 와 loss·gradient 동일(gate) |
| B: 통계 성분(GV 16 / GC 32 / SC 64) 별 같은 gate | **타당.** 기존 `stat_term` 은 성분 평균 뒤 gate 였으므로, 평균 전 성분 오차에 mask 를 곱하도록 경로를 분리했다. COMPADV 는 성분별 τ_V,j 캘리브레이션 필요 | trainer 통계 분기(TRI-B), `calibrate_component_tau` |
| C-SENS/C-DIAG: Teacher 출력의 correction 민감도 J_T (FD, U-Net 4회) 로 soft 감쇠 | **구현 가능하나 비용과 출처 한계.** step 당 Teacher forward 4회 추가 (s1 smoke 로 실측, §10.4). **현재 donor 는 무반응이라 G-EQ proxy Σ ≈ 0.31·I(등방)** → C-DIAG(EQ) 는 q = 0.31‖J‖² 로 C-SENS 와 같은 정보다(문서 §6.5 의 EQPROXY 경고와 일치). geo 출처는 영상 구조 곡률이다. 방향 정보의 가치는 반응하는 aligner 뒤에 검증 | `teacher_fd_jacobian·sens_risk·diag_risk`, Σ 는 §9 의 출처(eq/geo) 에서 `sigma_from_precision`(precision 0 방향 → invalid → r=1 fallback 기록) |
| C-FULL/QISO/QSCALAR (quadratic soft, Woodbury 2×2) | 타당·P3. λ_Q 는 QISO 의 soft gradient 를 R3 L1 soft 와 맞춘다(pilot Q02 `last`) | `lowrank_quad·iso_quad·scalar_quad`, `calibrate_lambda_q` |
| A 와 quadratic C 결합 | 문서에 정의 없음 → 허용하지 않음 | resolver |
| 통계 C (Φ = V_GV) | J_V 를 통계 출력의 FD 로 (Pixel q resize 금지 준수) | `teacher_fd_jacobian(phi='stat')`; quadratic 은 통계에 미구현 |
| 골격·이름 | 문서 D124 → **D123**(사용자 결정). 이름은 우리 규칙에 `TRI_A<mode>_B<mode>_C<mode><src>` 토큰 추가(앞단 A1/A2/A3 와 구분) | `kdv.registry.tri_tag` |
| 이중 합산 금지 (§2.1) | 준수: soft 만 교체, `L_R_base + L_R_A` 없음; B 는 GV-AD 를 교체 | trainer |

**결론: A·B 는 즉시 실행 가능하고 위험이 낮다. C 는 C-SENS 부터 의미가 있고, C-DIAG 의 방향 정보는 PO10 N2/N3 결과 뒤에야 검증된다.** 세 축 모두 "KD 제거 효과" 를 가려내는 R1(Q03)·GV-WH(45)·MASS·SHUFFLE 대조가 같은 큐에 있다.

### 10.2 구현
- `kdv/tri.py`: §18.1 참고 함수(direction_mask, masked_l1, fd_jacobian, diag_risk, lowrank_quad) 그대로 + mass/shuffle/component_weights/masked_l1_componentwise/routing_stats/teacher_eval/teacher_fd_jacobian/sens_q/sens_risk/iso_quad/scalar_quad/sigma_from_precision/sigma_control/linearization_check.
- config `kdv.tri: {a: {mode}, b: {mode}, c: {mode, phi, h, s_c, covariance_source, sigma_control, lambda_q}, shuffle_seed}`; resolver 는 soft 없는 rec(N0/R1) 에 A, 통계 OFF/H/WH 에 B, 출처 없는 C-DIAG/FULL/QSCALAR, A+quadratic C 를 오류로 막는다.
- trainer: rec soft = `masked_l1(y, y_T, G, w_H, w_K, mask_A, risk_C)` (w_H/w_K 는 R3 criterion 의 detached maps, β 포함); 통계 soft 도 같은 방식(mode T 는 w_H=0·w_K=1). C 는 Student graph 전에 Teacher probe 4회(순차·no_grad). calibration: s_sens(양의 q_sens median, FD h/2·2h 일관성·선형화 잔차 → `BLOCKED_NUMERICS`/`SOURCE_UNRESPONSIVE` 이면 exit 4), s_C = τ_R(identity)/τ_V(stat), 성분 τ, λ_Q. 로그 `routing_components.jsonl`(u/v 0·same-sign·충돌 비율, mask 평균, soft mass 비율, soft 0 원인(a_T vs mask), risk 분위), `fd_and_linearity_checks.jsonl`(J RMS·h/2 상대차·선형화 잔차·q 분위·Σ 고유값·probe 시간), `train_log.jsonl` 의 tri_* EMA.
- gate: `tools/kdv_unit_tests.py` TRI 절 — addendum §12.1–12.3 의 검사(손계산 §4.5, G=S, S=T≠G, cap<sign, cosine, mask=1 ⇒ R3 동일, mask=0 ⇒ hard-only, detach, BANDADV⊂sign, MASS κ, SHUFFLE 보존, B 성분 독립·gradient·constant, 선형 toy FD, diag 해석식, Σ=0/invalid/PSD 단조, (dy,dx) 뒤집힘 검출, HR/LR 단위, Woodbury=직접 inverse·W∈(0,1], J=0 fallback, precision 0 → invalid, rotate 대조, 실제 Teacher J·hash) 전부 통과.
- 큐 `config/queues/kdv_s2_triabc.txt` (`gen_kdv_configs.py --triabc`, 22 run): P1 A-SIGN(30)·B-SIGN(40)·C-SENS(50)·C-DIAG EQ(51) → 대조 A-MASS(34)·A-SHUFFLE(35)·GV-WH(45)·B-MASS(43)·B-SHUFFLE(44) → P2 B-CAP(41)·B-COMPADV(42)·A-COS(31)·A-CAP(32)·A-BANDADV(33)·C-DIAG GEO(52)·Σ rotate/shuffle(54/55)·AB(60)·AC(61) → P3 QISO/QSCALAR/FULL(56–58). 본 큐(Q01 Teacher·Q02 pilot·Q04/Q06 base) 뒤에 기동.

### 10.3 약명 → 세팅 (큐 22 run)

| run | 세팅 |
|---|---|
| `S2W112D123_A1_IA_AFR_R3_OFF_G0_TRI_ASIGN_BOFF_COFF_s1234_v01` | Q04(A-FR·R3·stat OFF) + A sign |
| `…_R3_OFF_G0_TRI_ACOS_…` / `…_ACAP_…` / `…_ABANDADV_…` / `…_AMASS_…` / `…_ASHUF_…` | Q04 + A cosine / sign_cap / band_adv / mass 대조 / shuffle 대조 |
| `S2W112D123_A1_IA_AFR_R3_GVAD_G0_TRI_AOFF_BSIGN_COFF_s1234_v01` | Q06(A-FR·R3·GV-AD) + B sign |
| `…_GVAD_G0_TRI_AOFF_BCAP_…` / `…_BCOMPADV_…` / `…_BMASS_…` / `…_BSHUF_…` | Q06 + B sign_cap / comp_adv / mass / shuffle |
| `S2W112D123_A1_IA_AFR_R3_GVWH_G0_s1234_v01` | GV-WH (B 의 기준, 계획 Q13) |
| `…_R3_OFF_G0_TRI_AOFF_BOFF_CSENS_…` | Q04 + C sens (s_sens calibration) |
| `…_CDIAGEQ_…` / `…_CDIAGGEO_…` / `…_CDIAGEQROT_…` / `…_CDIAGEQSHUF_…` | Q04 + C diag (Σ = eq proxy / geo) / Σ 회전 대조 / Σ sample-shuffle 대조 |
| `…_CQISO_…` / `…_CQSCALAREQ_…` / `…_CFULLEQ_…` | Q04 + quadratic soft W=I / tr W/D / 저랭크 full (λ_Q pilot Q02 last) |
| `…_GVAD_G0_TRI_ASIGN_BSIGN_COFF_…` / `…_OFF_G0_TRI_ASIGN_BOFF_CDIAGEQ_…` | AB / AC 결합 |

### 10.4 s1 dry run (D123, 300 updates)
s1 (D123, 300 updates; Teacher = D123 dry Teacher, pilot = Q02 dry `last`):

| run(dry) | step | probe | 확인한 것 |
|---|---:|---:|---|
| A-SIGN (Q04 위) | 75 ms | — | gate 평균 0.57(=soft 가 있는 band 중 same-sign), same-sign 0.76 / 충돌 0.24 (u,v 성분 기준), soft 4.2e-5 → 3.9e-5, soft mass ratio κ 0.93, soft 0 의 원인: a_T 0.26 / mask 0.00 |
| B-SIGN (Q06 위) | 80 ms | — | 통계 성분 same-sign 0.98 / 충돌 0.02 (GV 통계에서는 방향 충돌이 드물다 → B-SIGN 의 효과는 작을 전망), 통계 soft 6.7e-7 → 6.4e-7 |
| C-SENS | 252 ms | 167 ms | s_sens 1.1e-3 (양의 q_sens 중앙값, HRMS²/px²), FD h/2 상대차 4.4 %, 선형화 잔차 9.4 % (δ=(0.1,−0.05) px), r p50 0.56, soft 3.7e-5 → 1.6e-5 (경계에서 soft 절반 감쇠) |
| C-DIAG(EQ) | 170 ms | 97 ms | Σ = probe closure(등방 ≈ 0.23·I 이 run 의 Teacher), q p50 2.0e-4 vs s_C² = τ_R² 4.0e-4 → r p50 0.67, soft 3.9e-5 → 1.9e-5 |
| C-FULL(EQ) | 340 ms | 160 ms | λ_Q 1.04 (QISO soft gradient 를 R3 L1 soft 와 맞춤), Woodbury quadratic soft 유한 |

읽기: Teacher probe 4회는 step 을 +100–170 ms(Teacher forward 4회) 늘린다 — 50K 면 학습만 +1.4–2.4 h. C-DIAG(EQ) 와 C-SENS 의 r 분포가 비슷한 것은 예상대로(등방 Σ). J 의 FD 는 h/2 일관성 4–5 %, 선형화 잔차 9–10 % 로 문서 §6.9 기준(수치 실패 없음) 안.

## 8. 미결·후속

- depth: **D123 으로 확정**(사용자, 2026-09-10 저녁). Teacher seed 2025 / Student 1234 도 확정.
- 이동량 covariance KD G1–G5·G-STRUCT: **구현 완료(§9), 실행 보류**. G-CORR/XVIEW·Teacher cache·activation checkpointing: 미구현.
- A-FTW(warm freeze→unfreeze)는 `kdv.phase` 로 표현 가능하나 dry run 하지 않았다.
- RR-val ERGAS 는 plain 정의(선택 전용). 시트 수치는 기존 evaluator.
