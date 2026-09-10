# 2026-09-10 — PO10 (PAN 추가 변위 + offset consistency, 10 GPU-h) 명세 검토와 구현 노트

명세: [`PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md`](PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md). 상위 A1–A3 구현 노트: [`2026-09-09_pa-a1-a3-implementation.md`](2026-09-09_pa-a1-a3-implementation.md).
결과는 `results_log/2026-09-10_pa-a1-a3-s1-results.md` 의 추기(WIP)와 이후 날짜 문서에 쓴다.

## 1. 검토 — 명세의 전제와 저장소 실제

| 명세 항목 | 저장소 실제 | 처리 |
|---|---|---|
| 규모 R (§3.1): audit 부록 E WV3 train 전역 \|δ\| P90 0.250 LR px × 4 | 부록 E 표는 "해상도 맞춤, **LR px**", WV3 n=199, 중앙 0.072 / P90 0.250 — 단위·출처 확인(G1). 09-10 §7 의 독립 측정(64² patch, HR px): 중앙 0.17(하한 0.16), p95 0.76 → 같은 규모 | **R = 1.0 HR px** 고정, `calibration_quality=train_aggregate_proxy` 로 manifest 기록 |
| A1 resolved config (§7) | AdamW 1e-4 · wd 0.01 · cosine · warmup 100 · batch 48 · 50K · AMP 없음(`mixed_precision` 기본 None → fp32) | 그대로. GradScaler 경로 없음 |
| A1 학습 소요 | 50K: 1 h 31 m (학습 0.055 s/step ≈ 46 min + 평가 50회) | PO10 smoke 실측 native 46–49 ms · corrupt 47–49 ms/step (batch 48) → run 당 ≈ 1.5 h 예상, N1+N2+N3 ≈ 4.7 h + gate·진단 < 10 h |
| 초기값 (§7-2) | `work_dir/_pa_init/init_{unet,aligner}_seed2025.pt` (A1 이 만든 것) | PATrainer 가 로드·hash 대조(T19) — N1/N2/N3 동일 |
| "train signed shift 평균 ≈ 0 이 flip/rot 을 되돌린 값인가"(§1.1) | 09-10 §7 의 train 측정은 h5 원본 patch(증강 없음)라 되돌림 불필요. aligner 학습 로그의 Δ 평균은 증강된 batch 위 값이라 부호가 섞인다 | 새 로그는 ĉ 의 **norm 중앙값·표준편차**를 함께 남긴다 |
| "같은 batch 를 두 번이 아니라 다음 batch 마다 교대"(§4.1) | update 홀짝으로 결정. grad accumulation 없음 | `po_step(update_index)` |
| 조건부 N3 (§11.2) | 체인은 exit 4 를 "사전 gate 불통과, 재시도 없음" 으로 다룬다 | trainer `_budget_gate`: ledger(`work_dir/_po10_budget/ledger.json`)의 used + 1.2×proj(N1·N2 평균 ×1.03) + reserve 1 h ≤ 10 h 일 때만 시작, 아니면 `DEFERRED_BUDGET` + exit 4 |
| 평가·선택 (§10) | A1–A3 의 세 view·dual selector·last 그대로 | `PATrainer` 상속. aligner 입력만 margin 4 crop (`PAModel.aligner_margin`), 512² → view 504², 256² → 248² |
| 과거 A1 과의 비교 (§0, §5.2) | A1 은 전체 view | 결과표에 배경 기준으로만 둔다. 핵심 대응은 N2−N1, N3−N2 |

**명세 자체에 대한 의견.**

- ĉ0 의 절대 anchor 는 native recon 뿐이라는 한계(§4.7)는 09-10 결과와 같은 상황이다. 반응(B≈−I)이 생겨도 절대 정합량은 검증되지 않는다 — §13.3 의 해석 규칙대로 두 축(반응·native 품질)으로만 판정한다.
- R=1 HR px 는 FR 어긋남(1.79)의 절반, 학습 patch 어긋남(≈0.17)의 6배다. 학습 분포에 "변동" 을 만드는 목적에는 충분하고, FR 규모 일반화는 §10.3 의 ±2R stress 로만 본다.
- N1 의 offset loss 값을 진단으로 남기려면 corrupted step 에 ĉ0 를 no-grad 로 한 번 더 계산해야 한다 — 명세 §4.4 대로 했다(비용 aligner forward 하나).

## 2. 명세 → 코드

| 명세 | 코드 |
|---|---|
| §3 원판 sampling, 전용 RNG | `pa/offset.sample_offsets` (R·sqrt(u), CPU generator seed = seed + 2000) |
| §4.1–4.6 native/corrupt 교대, 두 warp, offset loss(SG/no-SG), N1 계수 0 | `pa/offset.po_step` |
| §4.7 λ=0.01, 5K ramp, native 0 | `pa/offset.lambda_off`, `train_po.OffsetConsistencyTrainer` |
| §5 고정 내부 view m_A=4·ceil((R+2)/4)=4, crop 후 z-norm | `pa/offset.aligner_margin / valid_view / predict_c`; 평가는 `pa/model.PAModel(aligner_margin=4)` |
| §6.2 FP32·0 우회 없음·두 번째 warp gradient 유지 | `po_step` (`torch.autocast(enabled=False)`) |
| §6.3 mask 없음, border 사용 비율·명시적 failure | `train_po` 로그 `border ε / ĉ`, \|ĉ\| > 8 px → `SUPPORT_FAIL` exit 4, 비유한 → exit 3 |
| §7 초기화·RNG·첫 1000 ε hash | `_pa_init` 로드, `corruption_first1000.json` (sha256) |
| §9 gradient 진단 (1000 update 마다 고정 native/corrupt minibatch) | `train_po._diagnose` → `gradient_diagnostics.jsonl`: dLrec/dc, grad_rec_phi, grad_off_phi, weighted, cosine, grad_off_theta(=0 확인), head/stem 분리, c0.requires_grad, parameter_update_norm_phi, prediction_change_norm |
| §10.1–10.2 세 view·best_raw/best_aligned/last | `PATrainer` 그대로 |
| §10.3 반응 (64 valid / 256 RR / 512 FR, probe 0·±R/2·±R + 원판 8 + ±2R stress, B·b fit, closure P50/P90, native ĉ0 통계) | `tools/po10_diag.py` → `offset_response_{native64,rr256,fr512}.csv`, `results/po10_diag.json` |
| §10.4 대조 (padding 변경 불변, MS 교체/상수, 두 번 보간 바닥, bilinear kernel; zero/learned/wrong-sign 은 pa_diag) | `tools/po10_diag.py interpolation_controls` → `interpolation_controls.json`; `tools/pa_diag.py` (po run 지원, tile 진단도 view 적용) |
| §11.3 run ID · 큐 | `tools/gen_po10_configs.py` → `config/PO10_{N1_REC,N2_OFFSG,N3_OFFNOSG}_W96_D124_WV3_S2025.yaml`, `config/queues/po10_s1.txt` (`--updates 25000` 이면 `_SCREEN25K`) |
| §12 gate | `tools/po10_unit_tests.py`(G2–G7), `tools/pa_unit_tests.py`(G9 evaluator), `tools/smoke_cases.py` po 분기(G8 실배치·throughput), `tools/po10_prepare.sh`(G0·G1 + ledger + 기동) |
| §13.1 파일 | `corruption_scale_manifest.json`, `source_and_init_hashes.json`, `corruption_first1000.json`, `budget_status.json` + ledger, `train_log.jsonl`(rec/off/weight, ĉ0·ĉε 통계, \|ε\|, closure, border, step time), 나머지는 PATrainer 산출물 |
| 시트 | trainer `po` 라벨(`PO10 N2(corrupt + offset loss, sg) R=1.0`), 범주 ⑲ PO10, `_upload.sh` 가 pa_diag + po10_diag 자동 실행 |

## 3. Gate 결과 (s1, 2026-09-10)

- `tools/po10_unit_tests.py` **전부 통과**: G2(2만 표본 mean ≤0.002, 축 sd 0.499/0.502, max L2 ≤ R, 면적 균등 P(<R/2)=0.25), G3 부호·zero-warp gradient, G4(N2 c0 no-grad / N3 연결 / 같은 forward loss / gradient 상이 / native 연결), G5(L_off → aligner 만, L_rec → 둘 다), N1 계수 0, ramp, closure 항등, G6(모든 \|ε\|≤R 에서 [4:60] 4-tap 유효 · **padding border↔reflection 에 ĉ 불변 1.5e-8**), G7(정수 roundtrip 정확 · subpixel 은 무손실 아님 — 기록), PAModel margin 256/512 forward.
- `tools/smoke_cases.py` N1/N2/N3 통과: peak 2.43–2.55 GB, **t_native 46–49 ms · t_corrupt 47–49 ms** (batch 48). 예상 run 당 ≈ 1.5 h (A1 실측 1 h 31 m 과 같은 급).
- 40-update dry run(N2, 격리 ledger) + `po10_diag` + `pa_diag`: 아래 §5 에 기록.
- N3 예산 gate: ledger 에 N1 3.2 h·N2 3.4 h 를 넣으면 `DEFERRED_BUDGET` exit 4, 1.6/1.7 h 면 시작 — 아래 §5.

## 4. 미반영·주의

- §9 의 "매 1,000 update 고정 minibatch" 는 첫 native/corrupt batch 의 앞 16 sample 로 고정했다(메모리). 진단 backward 는 optimizer 를 건드리지 않는다.
- §10.3 의 64² 반응은 valid 세트(`valid_wv3.h5`) 64 patch, 고정 sample id(등간격). 학습 sample 과 분리.
- `budget_ledger.json` 은 run 폴더가 아니라 `work_dir/_po10_budget/ledger.json` 한 곳(세 run 이 공유해야 하므로); 각 run 폴더에는 `budget_status.json`.
- 25K screening 전환(§11.2)은 `gen_po10_configs.py --updates 25000` 으로 만들 수 있지만, 실측 throughput 상 50K 두 벌 + 진단이 10 h 안에 들어 사용하지 않는다.

## 5. Dry run · gate 결과

- 40-update dry run(N2, 격리 ledger, diag_every 10): native/corrupt 교대, offset loss·λ ramp, ĉ0·ĉε·\|ε\|·closure·border 로그, `gradient_diagnostics.jsonl`(dLrec/dc, grad_rec_phi head/stem, parameter_update_norm, prediction_change), 세 view 평가(view 504²)·두 선택기·best_hqnr/best_aligned/last 내보내기, manifest 3종·ledger 기록 — 전부 동작.
  초기 aligner 는 zero head 라 ĉ ≈ 0, closure ≈ mean\|ε\| = 0.65 (정의상). 학습 patch 위 border 사용 비율 0.0916 = 1 − (61/64)² 는 보수적 4-tap mask 의 바닥값.
- `tools/po10_diag.py`: 64/256/512 반응 CSV·B fit·closure·±2R stress·native ĉ0 통계, §10.4 대조(padding 불변 0.0e+0, MS swap/const, 두 번 보간 바닥 5e-3, bilinear kernel) 동작. `tools/pa_diag.py` po run 지원(교차표·RR-valid) 동작.
- 예산 gate: ledger 에 N1 3.2 h·N2 3.4 h → N3 `DEFERRED_BUDGET`(exit 4, 체인은 재시도 없음); 1.6/1.7 h → 시작. 실측 throughput(≈47 ms/step)이면 run 당 ≈ 1.5 h 라 N3 는 예산 안에 들어갈 전망.
- 기동: `./tools/po10_prepare.sh` (G0·G1 확인 → verify_metrics → po10/pa unit gate → smoke·throughput → ledger → `campaign_start.sh --queue config/queues/po10_s1.txt`). 큐 순서 N1 → N2 → N3(gate).

## 6. 2차 검토(사용자) 반영 — 2026-09-10 낮

| 지적 | 판단 | 반영 |
|---|---|---|
| 두 번 warp 보간 바닥 진단이 [8:-8] 고정 crop 이라 경계 복제값이 섞임 (ĉ0=(7,0) 에서 4.2%, (8,0) 에서 6.3%) | **맞음** | `pa/warp.two_stage_support_mask(ε, c)`: 1단계 P_ε 의 유효 직사각형 안에서 2단계 4-tap 이웃이 전부 유효한 픽셀만. 바닥은 두 경로 support 의 교집합 안에서만 계산하고, padding 을 바꿔도 그 안의 값이 같음을 함께 기록(`double_interp_padding_independence_inside_support_max_abs`). 테스트: 교집합 안 padding 차이 1.8e-7, 옛 [8:-8] 은 차이 있음 |
| corruption stress HQNR·전용 mask 미구현 | **맞음** | `po10_diag.stress_hqnr`: FR 20장에 ε∈{0, ±R 축} 을 넣은 P_ε 로 추론 → **stress ROI margin 96**(블록 3개; 저해상도 PAN support 54 + 두 단계 taps 4 → 적격 \|ε\|∞+\|ĉ\|∞ ≤ 38) 에서 raw_valid·aligned_valid(P̃ε = W(W(P_raw,ε),ĉε) float64). `stress_hqnr_fr512.csv`, `stress_roi_manifest.json`. 최종 diagnostic 만, 선택 미사용 |
| 평가용 P̃(float64 재warp)와 저장 P̃(FP32 forward + clip)가 다름 → 저장본 재평가가 기록을 재현 못 함 | **맞음** | 내보내기 규약 통일(`train_pa._collect/_savemat`): 참조 pan·lms·ms·gt 는 h5 원본 float64, `pan_aligned` 는 평가에 쓴 W(P_raw, Δ̂) float64(clip 없음), `pan_aligned_forward_fp32` 는 네트워크가 본 P̃ 의 정확한 역변환(clip 없음), `sr` 만 clip. `results/export_convention.json`. 검증: 저장 MAT 로 aligned_valid 재계산 = 기록값 (아래) |
| ① 재개 시 corruption RNG 가 처음부터 다시 시작 | **맞음** | `RNGState` 를 `accelerator.register_for_checkpointing` → `custom_checkpoint_0.pkl` 에 generator 상태 저장·복원(테스트: 10 회 draw 후 상태 = 저장 상태). DataLoader 는 근사 재개라 sample–ε 대응은 깨진다 → 재개 run 은 `resume_manifest.json`(`paired_epsilon_valid=false`) 로 표시. 실제로 뽑힌 첫 1000 ε 의 hash(`corruption_drawn_first1000.json`)를 학습 중 기록 — N1/N2/N3 대조는 이 값으로 |
| ② 진단의 weighted offset gradient 가 update_index=1 로 λ 를 2,500배 작게 기록 | **맞음** | `diag_update_index(step, parity)` 로 실제 step 의 λ(t) 사용, `lambda_used` 기록. N1 은 λ=0.01 을 가상으로 쓴다는 표시(`lambda_virtual_for_N1`) |
| ② prediction_change_norm 이 이전 진단 시점 대비 | **맞음** | 같은 update 전후의 `prediction_change_norm_1step`(고정 native batch) + `prediction_change_since_last_diag` 둘 다 기록 |

N1 은 11:33 에 옛 코드로 시작돼 있었다 — 세 run 이 같은 코드여야 하므로 N1 을 중단·삭제하고 고친 코드로 다시 시작한다(사용한 GPU 시간은 ledger 에 `aborted` 로 남긴다).

## 7. 변경 명세 R100 → R200 적용 (2026-09-10 오후) — 다음 실험부터

변경 명세: [`PAN_OffsetConsistency_ChangeNote_R100_to_R200_2026-09-10.md`](PAN_OffsetConsistency_ChangeNote_R100_to_R200_2026-09-10.md). 사용자 지시: **다음 실험부터 적용**, 노트에 구분.

| 구분 | R100 (옛 protocol `pan_offset_rel_10h_v1`) | R200 (새 protocol `pan_offset_rel_10h_v2_wv3_r200_frstat`) |
|---|---|---|
| 반경 b | 1.0 HR px (audit 부록 E train P90 0.250 LR px × 4, train_aggregate_proxy) | **2.0 HR px** (09-10 §7 WV3 FR \|δ\| P95 1.98 HR px, `fr_input_statistics_informed_development`) |
| 골격 | W96 · D124 (B0/A1 과 동일) | **W112 · D123** (사용자 결정 2026-09-10 오후, "변경된 실험은 width/depth 도 바꾼다"). backbone 2.6589 M (+aligner 0.1053 M). init snapshot 은 별도(`work_dir/_pa_init_w112_d123`) |
| run | `PO10_N1_REC_W96_D124_WV3_S2025` — 13:16 기동, **완주시켜 R100 기록으로 보존**. R100 N2/N3 는 **실행하지 않음**(큐에서 제거, 옛 체인 스케줄러 중지) | `PO10_{N1_REC,N2_OFFSG,N3_OFFNOSG}_W112_D123_WV3_S2025_R200_FRSTAT` — 새 골격의 **새 독립 학습**(restart_pure), N3 는 예산 gate |
| 큐 | `config/queues/po10_s1.txt` (기록용) | `config/queues/po10_s1_r200_frstat_w112_d123.txt` — 첫 줄은 R100 N1(완료분 업로드용). (W96·D124 R200 config 는 만들었다가 폐기) |
| 대응 비교 | — | 골격이 B0/A1/R100 과 다르므로 **그 run 들과의 직접 대응(Ak−B0, N−A1)은 하지 않는다**. block 안 N2−N1, N3−N2 만. W112·D123 의 B0(aligner 없음)는 이번 예산에 없다 — 필요하면 후속 |
| 그 밖 | 원판 uniform·1:1 교대·λ 0.01/5K·margin 4·loss·평가 전부 동일 | 동일 (변경 명세 §3 대조표) |
| manifest | `corruption_scale_manifest.json`: radius_profile R100_TRAINP90 | radius_profile R200_FRSTAT + `radius_provenance`(uses_evaluation_input_statistics **true**), training_regime restart_pure, parent_run_id null |
| 진단 | probe 0·±0.5·±1 (in-range) + stress ±1.5·±2 | probe 0·±0.5·±1·±1.5·±2 전부 in-range — 같은 절대 변위 구간에서 R100 과 비교. \|ε\| 0.5 px 구간별 closure 기록 |
| 예산 | 같은 10 GPU-h ledger 에 profile 표시. 사용: gate 0.3 + 폐기 1.65 + R100 N1 ≈1.55 | 남은 ≈ 6.5 h. W112·D123 은 step 이 길어져(smoke 실측 참조) run 당 ≈ 1.8~2 h → N1·N2 는 들어가고 **N3 는 gate 에서 DEFERRED_BUDGET 이 될 가능성이 높다**(3.5 + 2×1.9 + 1.2×1.9 + 1 ≈ 10.6 h). 그 경우 예산 증액은 사용자 결정 |

R100 N1 은 옛 코드가 아니라 2차 검토까지 반영된 코드(13:16 기동)로 완주하므로 R100 기록으로 유효하다. 다만 R100 은 N1 하나뿐이라 R100 안의 N2−N1 대응은 없다 — R200 block 이 주 비교다.

### HQNR 두 가지 (masking 유무) — 모든 run 의 시트 열

사용자 요청: masking 들어간 HQNR 과 들어가지 않은 HQNR 을 둘 다 뽑는다.

| 열 | 정의 | 어디서 |
|---|---|---|
| `HQNR↑` | 전체 프레임(논문 프로토콜, crop 없음). 논문 비교표는 이 값 | `tools/eval_fr_paperset.py` (기존) |
| `HQNR(V64)↑` | 가장자리 64 px(블록 2개)를 뺀 고정 영역 [64:H−64, 64:W−64] 의 장면별 (1−D_λ)(1−D_s) 평균. 필터는 전체 프레임에서 계산한 뒤 자른다 — A1–A3/PO10 의 raw_valid 와 같은 정의 | `pa/evalviews.raw_views` → `fr_mat20.json` 의 `hqnr_valid`(+ `d_lambda_valid`, `d_s_valid`, `fscc`, `fscc_valid`) |

evaluator 버전 `2026-09-10.5`. 기존 run 은 저장된 `full_*_mat20.mat` 에서 재추론 없이 V64 필드만 더한다(`--all`). 시트는 `--all --replace` 로 열을 추가한다.
09-09 문서 §7 에서 본 대로 CNN 출력의 D_λ 는 테두리 한 블록 몫이 21~64% 라 두 값의 차이가 크다 — 모델 간 비교는 각 열 안에서만 한다.
