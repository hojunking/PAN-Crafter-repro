# s2 W112·D124 KD·variance·aligner 재사용 — 계획 검토와 구현 노트 (2026-09-10)

계획 묶음: `research_log/PAN_S2_W112_KD_Variance_Plan_and_References_2026-09-10/`
(캠페인 문서 `PAN_S2_W112_D124_KD_Variance_Adaptive_Campaign_2026-09-10.md`, 참조 구현 `pan_gt_anchored_kd.py` · `pan_s2_w112_stat_kd_reference.py`, README, 검사 JSON).
사용자 지시: "현 workspace 기준으로" 검토·구현. 이 문서는 (1) 계획의 REQUIRED/가정을 저장소 실제와 대조한 검토, (2) 구현 지도(implementation map), (3) gate 결과, (4) s2 실행 절차다.

---

## 1. 검토 — 계획의 REQUIRED · 가정 ↔ 저장소 실제

| 계획 항목 | 저장소 실제 (2026-09-10) | 처리 |
|---|---|---|
| backbone class·폭·깊이 (§2.2 REQUIRED) | `model.pancrafter_paper.PANCrafterPaper`, `hidden_size 112`, `depth [1,2,4]`, `in_mode paper`(cat(PAN, ↑MS) 9ch), `mode_modulation false`, `attn_locations []`, norm **LN** → GN 나눗셈 문제 없음. backbone **2.8854 M** (W96·D124 2.123 M 대비 +35.9%). aligner GN4/GN8 은 폭과 무관 | `architecture_manifest.json` 을 run 마다 자동 기록 (M01) |
| **depth** | 계획 §0.2 "Depth [1,2,4] **유지**". 그러나 같은 날 사용자 결정으로 s1 PO10 R200 block 은 **W112·D123**(2.6589 M) 이다 | **계획대로 D124 로 구현**(생성기 `--depth 1,2,4` 기본). s1 PO10 R200(W112·D123) 과는 골격이 달라 직접 대응 비교가 아니다. D123 으로 맞추려면 `tools/gen_kdv_configs.py --depth 1,2,3` 한 번이면 된다 — **사용자 확인 필요 항목** |
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
| 이동량 covariance KD G2–G5·G-EQ·G-STRUCT (§11) | 미구현(계획도 P2/P3). G0(없음)·**G1**(scalar k0 mean-KD, native step, trainable aligner 만) 만 구현 | resolver 가 G2+ 를 **PENDING** 오류로 막는다 (조용히 0 으로 돌리지 않음) |
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
| covariance_* | **미구현** | resolver 가 G2+ 를 막음 |
| calibration | `kdv/calibration.py` | tau_R · tau_V/eps_V · λ_V(pilot, r_grad 0.05) · json 캐시 `work_dir/_kdv_calibration/<teacher_id>/` |
| evaluation_adapter | `train_pa.PATrainer` 상속 | 세 view·선택기·export 그대로 + `best_rr_val` + `fitting_bins.csv`(§19.2 Teacher error Q0–50/50–90/90–100 의 Δe·WinRate, pixel·stat) |
| phase_manager | `train_kdv.KDVTrainer._warm_start` + `main.py` warm-start 분기 | `kdv.phase{parent_run, parent_tag, parent_step, optimizer_state_policy}` → backbone/optimizer(backbone group 위치 대응)/scheduler 위치 복원, `parent_and_phase.yaml`(config diff·inherited cost) |
| run_registry | `kdv/registry.py` | case 등록·resolver(잘못된 조합 오류)·이름 규칙·서술 |
| trainer | `train_kdv.py` `KDVTrainer` (`trainer: kdv`, `--kdv` YamlAction) | 아래 §3 |
| config/queue | `tools/gen_kdv_configs.py` → `config/S2W112_*.yaml` 9벌, `config/queues/kdv_s2.txt` | §4 |
| gates | `tools/kdv_unit_tests.py`, `tools/smoke_cases.py`(kdv 분기), `tools/pa_unit_tests.py`(evaluator·selector) | §5 |
| 기동 | `tools/kdv_prepare.sh` | 환경 json → 데이터·donor·지표·gate → config → smoke(Q00–Q02) → 캠페인 manifest → `campaign_start.sh` + 감시자 |
| 시트·평가기 | `tools/eval_fr_paperset.py`, `tools/pa_diag.py`, `gspread/gspread_upload.py`(비용·descriptor·notes), `gspread/sheet_categories.py`(⑳ KDV), `tools/_upload.sh`(S2W112_* → pa_diag) | |

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

이름 규칙 `S2W112_<recipe>_<input>_<aligner>_<rec>_<stat>_<geomKD>_s<seed>_<version>`. 공통: W112·D124 · 9ch · I-A(native) · 50K · AdamW 1e-4 · batch 48 · fp32 · 평가 eval_epoch 5.

| Q | run | recipe / aligner | rec | stat | Teacher | 목적 |
|---|---|---|---|---|---|---|
| Q00 | `S2W112_NOALIGN_IA_AID_N0_OFF_G0_s1234_v01` | aligner·sampler 없음 | N0 (L1) | OFF | — | 새 폭 독립 baseline |
| Q01 | `S2W112_T112DFR_IA_AFR_N0_OFF_G0_s2025_v01` | donor frozen (T112_DONORFROZEN_REC) | N0 | OFF | — | **Teacher** (seed 2025) → `T112_v01` |
| Q02 | `S2W112_A1_IA_AFR_N0_OFF_G0_s1234_v01` | donor frozen | N0 | OFF | — | frozen aligner GT-only 기준 (Teacher 와 seed 만 다름) · λ_V pilot |
| Q03 | `S2W112_A1_IA_AFR_R1_OFF_G0_s1234_v01` | donor frozen | R1 hard-only | OFF | T112_v01 | Teacher-error hard 재가중 |
| Q04 | `S2W112_A1_IA_AFR_R3_OFF_G0_s1234_v01` | donor frozen | R3 adaptive | OFF | T112_v01 | adaptive rec 기준 |
| Q05 | `S2W112_A1_IA_AFR_R3_GVH_G0_s1234_v01` | donor frozen | R3 | GV-H (GT gradient-variance 5×5) | T112_v01 | GT 구조 통계 추가 |
| Q06 | `S2W112_A1_IA_AFR_R3_GVAD_G0_s1234_v01` | donor frozen | R3 | GV-AD | T112_v01 | **첫 주력 후보** |
| Q07 | `S2W112_A1_IA_AFT_R3_GVAD_G0_s1234_v01` | donor 초기화 후 학습 | R3 | GV-AD | T112_v01 | aligner fine-tune |
| Q08 | `S2W112_A1_IA_ASC_R3_GVAD_G0_s1234_v01` | 독립 초기화 학습 | R3 | GV-AD | T112_v01 | 초기값 제약 확인 |

Q07/Q08 의 `STAT-*` 는 계획 "Q04~06 중 고정 선택" → **GV-AD**(§1 첫 주력 후보) 로 정했다. 순서 의존: Q01 → Q03–Q08(Teacher), Q02 → Q05–Q08(λ_V pilot). 체인(`_run_cases.sh`)이 큐 순서대로 돌고 case 직전 smoke 가 Teacher 존재를 확인한다.
Q09–Q25(R0/R2, GV-T/FIX/WH, IV/GC/SC/EDGE, G1, I-N, seed 반복 …)는 `tools/gen_kdv_configs.py build_kdv()` 에 case 를 더해 만든다(resolver 가 조합을 검사).

예상 소요(s1 4090 실측 smoke 기준, s2 에서 다시 잰다): W96 A1 run 1h31m(학습 46m + 평가 45m) 대비 W112 학습 ≈ ×1.4, Teacher forward 가 있는 run 은 학습 +30–40% → run 당 약 2–2.5 h, 9 run ≈ 20 h.

---

## 5. Gate 결과 (s1, 2026-09-10)

| Gate | 내용 | 결과 |
|---|---|---|
| L01 | 참조 rec 10 검사 + 8-band gradient 방향 | `tools/kdv_unit_tests.py` 전부 OK |
| L02/L04 | 참조 stat 12 검사 + spectral cov row-chunking = explicit unfold, 음의 off-diagonal 보존 | OK |
| M01 | W112·D124 2.8854 M, 첫 conv 9ch, GN 없음 | OK |
| M02 | donor strict(105,330·22 key) · W96→W112 strict load RuntimeError · non-aligner state KeyError | OK |
| M03 | Student step 뒤 Teacher hash 불변 · optimizer ∩ Teacher → RuntimeError | OK |
| M04 | y = base + residual(1회) · A-ID == B0 forward | OK |
| W01/W03/R01 | Δ=0 sampling identity · I-N 홀짝·순차 두 warp ≠ 단일 warp · corruption RNG roundtrip | OK |
| L03 | A-FR: Δ 공유·aligner/Teacher grad 없음·F_S grad 있음 · A-FT: sampler 통해 aligner grad · geo-only backward 에 F_S grad 없음 | OK |
| E01 | 학습 forward == 평가 forward(PAModel) | OK |
| resolver | 9개 잘못된 조합 오류, A-FR 의 geo 항 진단 전용 | OK |
| calibration | λ_V 규칙·degenerate 판정·v_scale | OK |
| P01 smoke | Q00/Q01/Q02: peak 2.69 GB, step 52–94 ms (PO10 학습과 GPU 공유 중 측정) | OK |
| dry run (300 updates) | §6 | 아래 |

---

## 6. s1 dry run (300 updates, 평가 1회, export 4 tag)

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

## 8. 미결·후속

- **depth D124 vs D123**: s1 PO10 R200 은 W112·D123 — s2 를 D123 으로 맞출지 사용자 확인(§1 표).
- 이동량 covariance KD(G2–G5·G-EQ·G-STRUCT)·G-CORR/XVIEW·Teacher cache·activation checkpointing: 미구현(PENDING), 필요 시점에 구현.
- A-FTW(warm freeze→unfreeze)는 `kdv.phase` 로 표현 가능하나 dry run 하지 않았다.
- RR-val ERGAS 는 plain 정의(선택 전용). 시트 수치는 기존 evaluator.
