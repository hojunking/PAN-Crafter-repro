# 2026-09-12 — PALS24 구현 노트: P2 기반 offset-consistency 가중치(λ_off) 비교, 24 GPU-h (s1)

계획: [`PAN_P2_P3_LambdaSweep_MetricAware_24GPUh_Plan_2026-09-12_v2.md`](PAN_P2_P3_LambdaSweep_MetricAware_24GPUh_Plan_2026-09-12_v2.md) (v2, 2026-09-12).
캠페인 `PALS24_W112D123_N2LAST_R200_v2` · protocol `PALS_W112D123_N2LAST_R200_v1` · s1 단일 GPU 순차 · 탐색 seed 1234 / 확인 seed 7777·2025.
NF16(`2026-09-11_nf16-implementation.md`) 의 P2/P3 recipe 를 **λ_off 만 바꿔** 그대로 쓴다 — trainer(`train_kdv.py`)·모델·loss·sampler·optimizer·평가는 수정하지 않았다.

## 1. 무엇을 돌리나

| 단계 | run (실행명) | 세팅 | 상태 |
|---|---|---|---|
| A1 | `PALS24_L1E3_W112_D123_WV3_S1234_N2LAST_R200_v1` | λ_off 0.001 | 신규 50K |
| A2 | `PALS24_L1E4_W112_D123_WV3_S1234_N2LAST_R200_v1` | λ_off 0.0001 | 신규 50K |
| A3 | `PALS24_L3E3_W112_D123_WV3_S1234_N2LAST_R200_v1` | λ_off 0.003 | 신규 50K |
| A4 | — | λ* 선택·고정 (`tools/pals24_select_lambda.py`) | 자동 (캠페인 gate) |
| B1–B3 | `PALS24_{CTRLP0,L000,<λ*>}_…_S7777_…` | seed 7777: P0 → λ 0 → λ* | 신규 50K × 3 (gate 가 연다) |
| C1–C3 | `PALS24_{L000,<λ*>,CTRLP0}_…_S2025_…` | seed 2025: λ 0 → λ* → P0 | 신규 50K × 3 (gate 가 연다) |
| 재사용 | `NF16_P0/P2/P3_W112_D123_WV3_S1234_N2LAST_v1` | seed 1234 의 CTRL-P0 / L000 / L1E2 | 재사용 gate 통과 (§5) |
| 배경 | `NF16_P3_W112_D123_WV3_S7777_N2LAST_v1` | λ 0.01 seed 7777 | 큐에 없음, 배경 대조만 |

**약명 → 세팅** (시트·문서에는 이 표로만 읽는다):

| 약명 | 세팅 |
|---|---|
| `CTRLP0` | W112·D123, aligner·sampler 없음 (A-ID, I-A) = NF16 P0 정의 |
| `L000` | λ_off 0 — A-FT donor aligner + native L_rec 만 (I-NATIVE-TRANSFER) = NF16 P2 정의 |
| `L1E4` / `L1E3` / `L3E3` | λ_off 0.0001 / 0.001 / 0.003 — A-FT, I-AEQ (홀수 update 에 P_ε 를 aligner 에만; \|ĉε+ε−sg ĉ0\| component-mean L1, ramp 없음, b=2 원판, view margin 4) |
| `L1E2` | λ_off 0.01 = NF16 P3 정의 |

공통: W112·depth[1,2,3](2.6589 M) 9ch 단일 HRMS, AdamW 1e-4(U-Net)/1e-5(aligner)/wd 0.01, cosine warmup 100, batch 48, 50,000 updates, eval_epoch 10,
donor = `PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT/last`(정확한 50K, file sha `9d4cbf218cd8f674…`, aligner tensors `db638b55c62f8c62`) 의 aligner 만 strict load,
U-Net 은 seed 별 저장 초기값 `work_dir/_kdv_init_w112_d123/init_unet_seed{1234,7777,2025}.pt` (donor U-Net·optimizer 미사용; seed 2025 초기값도 donor 와 무관한 새 tensor).

## 2. 계획 → 코드 대응

| 계획 | 구현 |
|---|---|
| §2.2 λ 외 변경 없음 | `tools/gen_pals24_configs.py` — NF16 템플릿·`build` 와 같은 recipe 에 `kdv.aux.offset_weight = λ` 만. gate PL02 가 생성 YAML 을 NF16 P3/P2/P0 config 와 dict 비교: 차이 없음(P0 는 eval_epoch 5→10 만) |
| §4–§5 loss·schedule | 기존 `train_kdv.py` I-AEQ 경로 그대로 (`step % 2 == 1` 에 P_ε → aligner, `offset_loss(stop_reference=True)`, `lambda_off(ramp 0)`). gate PL03: L_rec → 양쪽, L_off → aligner 만, sg target, 합산 gradient = g_rec + λ g_off (상대오차 2e-6), λ=0 ≡ P2, U-Net forward 1회 |
| §5.3 ε sampler | `pa/offset.sample_offsets` (원판 면적 uniform, b=2). PL04: E[r²]=2.00, 상수 예측기 component-mean L1 = 4b/(3π) 0.8488 (MC 0.8493) |
| §6.4 RNG 분리 | 기존 `corruption_seed_offset 2000` generator (data RNG 와 분리), calibration 은 `fork_rng` |
| §7 평가·checkpoint | 기존 4 view(raw_original / raw_valid / aligned_valid / aligned_fixed_v64), best_raw=best_hqnr(HQNR 1e-4 tie → fSCC 1e-4 → 늦은 update), last 정확한 50K |
| §8 G-M0–G-M8 | `tools/pals24_metric_gate.py` → `work_dir/_pals24_campaign/{metric_contract.json, reuse_registry.json, metric_gate_report.json}` (§5) |
| §9.2 반응 | `tools/po10_diag.py --probe-set pals24` — 0 + 반경 {0.5,1,2} × 8 방향(축 4·대각 4) 고정 probe 를 64²/256²/512² 에, `B_resp` 2×2·intercept·`offset_mae_component`·`offset_epe`·legacy `closure_*`(이름 유지)·‖ĉ0‖ 중앙값/P90·`drift_vs_reference`(donor 대비). csv `offset_response_<scale>_pals24_<ckpt>.csv` |
| §9.3 λ 전달량 | `train_kdv._diagnose`: 홀수 진단 step(1001, …, 10001, 25001, 49001) 에 `lambda_off, loss_off_raw, loss_off_weighted, grad_rec_A, grad_off_A, grad_off_A_weighted, rho_g, cos_psi`(norm 0 이면 None) — `autograd.grad(retain_graph)`, `.grad`·optimizer 불변 |
| §9.4 aligner-only 검사 | 진단 step 마다 `off_unet_grad_absent`(L_off → U-Net θ gradient 없음) 기록 + gate PL03 |
| §10.3 λ* 규칙 | `tools/pals24_select_lambda.py::select_lambda` — raw best HQNR → fSCC(1e-4) → 더 작은 양의 λ; 한 번 쓰면 `selected_lambda.json` 고정(재선택 없음) |
| §11.5 순서 | 큐 `config/queues/pals24_s1.txt` = A1–A3. stage 2 는 `tools/campaign_gate.py` 의 `pals24` gate(`work_dir/campaign_gates_enabled.txt`) 가 A1–A3 완료 뒤 λ* 선택 → `gen_pals24_configs.py --stage2` → B1–B3·C1–C3 순서로 연다 |
| §11.5 종료 조건 | 세 신규 λ 가 모두 P2(0.95389) 보다 0.0031 초과 낮으면 `selected_lambda.json: TERMINATE` — stage 2 를 열지 않는다. 열려면 `tools/pals24_select_lambda.py --policy continue` 뒤 `campaign_start.sh` |
| §11.6 예산 gate | `kdv.budget`: ledger `work_dir/_pals24_budget/ledger.json`, 24 GPU-h, reserve 4.0(최종 평가 3.0 + buffer 1.0), **margin 1.1**(새 키 `budget.margin`, NF16 기본 1.2 불변), `required: false`(초과면 DEFERRED_BUDGET exit 4 — 사용자 승인 없이 24h 를 넘기지 않는다), block 완결 검사 = `remaining_mandatory` 에 같은 block 의 나머지(P0·L000·λ*) |
| §13 산출물 | run 폴더 파일은 기존 이름 그대로(`campaign_manifest.yaml` 의 `file_map` 에 계획 이름 ↔ 실제 파일). 캠페인 수준: `campaign_manifest.yaml`, `reuse_registry.json`, `selected_lambda.json`, `campaign_budget_ledger.json`, `paired_seed_results.csv`, 표 A/B/C csv, `report.md` (`tools/pals24_report.py`) |
| 시트 | 범주 ㉓ PALS24(`gspread/sheet_categories.py`, KEEP), `tools/_upload.sh` PALS24 분기: last 에 pals24 probe + native-reference stress, best_raw·checkpoint-10000·epoch-125(=25,250 update, 25K 에 가장 가까운 저장본) 은 반응만; 진단 GPU 시간은 ledger `diag_<run>` |

## 3. 바뀐 파일

- `train_kdv.py` — `budget_decision(..., margin=1.2)` + `kdv.budget.margin`; `_diagnose` 의 §9.3/§9.4 필드. NF16·KDV gate 재실행 ALL OK (동작 불변).
- `tools/po10_diag.py` — `fixed_probes(R, probe_set)`, `--probe-set pals24`, `--response-only`, `drift_vs_reference`, fit 에 `offset_mae_component/offset_epe/native_c0_norm_*` 추가. 기본(po10) 경로·NF16 파일명 불변.
- 새 파일: `tools/gen_pals24_configs.py`, `tools/pals24_select_lambda.py`, `tools/pals24_metric_gate.py`, `tools/pals24_report.py`, `tools/pals24_unit_tests.py`(PL01–PL12, 50 검사), `tools/pals24_prepare.sh`, `config/PALS24_*_S1234_*_v1.yaml`(3), `config/queues/pals24_s1.txt`.
- `tools/campaign_gate.py` — `gate_pals24` (`GATES["pals24"]`; 기본 닫힘 유지). `tools/_upload.sh`, `gspread/sheet_categories.py`, `gspread/refile_sheet.py`.

## 4. 판정 규약 (변하지 않음)

주 판정은 **best_raw 의 raw_original HQNR(논문 세트 .mat 20장, 원 PAN 참조) → fSCC**(같은 checkpoint, 원 PAN 참조 full-frame Sobel SCC; RR GT-SCC 가 아니다 — `metric_contract.json` 에 고정). 경험적 판정선 0.0031 은 raw HQNR 에만.
aligned_self / fixedN2 / V64 / last 는 진단이며 낮은 raw 를 구제하는 대체 점수가 아니다. ERGAS·SAM 은 참고. λ* 는 seed 1234 결과로 한 번 정하고 확인 seed 결과로 바꾸지 않는다.

## 5. gate 결과 (2026-09-12, 학습 전)

- `tools/pals24_unit_tests.py` PL01–PL12: **50/50 OK**. 핵심: 생성 config ≡ NF16 P3/P2/P0(λ 외 차이 없음), 합산 gradient 선형성 상대오차 ≤ 2e-6, L_off→U-Net gradient 없음, 4b/(3π) 재현, 예산 gate(24.1h → DEFERRED), λ* 규칙 3 경우, 캠페인 gate 기본 닫힘.
- `tools/pals24_metric_gate.py` (G-M0–G-M8, `--skip-repro` 로 먼저; G-M1 재평가는 prepare 가 GPU 로):
  - G-M0: evaluator `460484e2227bc017`, FR eval 2026-09-10.5, selector best_raw HQNR tie 1e-4 → fSCC tie 1e-4. SCC 이름 모호성 해소: 선택의 보조 key 는 `fscc`(원 PAN 참조), RR SCC 는 참고.
  - G-M2 (best step, raw_original 20 scene, 중복 없음): 곱의 평균 = 기록값 (P0 0.953161, P2 0.953886, P3 0.946255, P1 0.932976; 평균의 곱과의 차 2e-5~1e-4 = scene 간 covariance, evaluator 오류 아님).
  - G-M3 P0 aligned_self ≡ raw_v64 (전 epoch, 1e-9) · G-M4 P1 self ≡ fixedN2 · G-M5 D_λ 세 view 동일 (전 run·전 epoch) — 통과.
  - G-M6: best 실제 update 보존 (P0 33330 · P2 40400 · P3 8080 · P1 16160), last 는 전부 정확한 50000, best≠last hash. **P0 는 eval_epoch 5** 로 돌았다 → 10 격자 재선택 0.95275@ep240 (기록 0.95316@165) 을 registry 에 병기.
  - G-M7: best step 20/20 scene 적격, invalid 없음. 자산: U-Net 초기값 tensors sha `c988a6c95b17f4fd`(seed 1234) 현재 파일과 일치, donor file sha 일치, FR eval version 현재와 동일.
  - G-M8 provenance: S1 §1 의 PO10 D_λ 0.0112–0.0116 은 **raw_valid/aligned_valid(V64 ROI)** 의 last 값(N2 0.01146), 시트/S4 의 0.0171–0.0175 는 같은 last 의 **raw_original(전체 프레임)** — view 차이로 해소. sweep 기준은 재현된 raw_original 프로토콜.
  - 재사용 승인: CTRLP0 ← NF16 P0 (APPROVED_WITH_NOTE: 격자 5), L000 ← NF16 P2, L1E2 ← NF16 P3 (APPROVED).

## 6. 예산 운영

`ledger.json`: `gate`(실측; 예약 2.0), run 마다 `PALS24_…`(실측 hours), `diag_<run>`(FR 평가 분담 + pa_diag + po10_diag). gate 식 `used + 1.1·(이 run + 같은 block 나머지) + 4.0 ≤ 24`.
예약표: gate 2 + A 6 + B 6 + C 6 + 최종 평가 3 + buffer 1 = 24. NF16 실측(같은 골격·eval_epoch 10) run 당 1.43–1.52 h + 진단 0.14 h 라 예약 2.0/run 은 보수적이다.
부족 시 순서: C(seed 2025) block → B(seed 7777) block 보류(DEFERRED_BUDGET, 체인은 다음으로). 50K 를 줄이거나 λ·LR 을 run 중 바꾸지 않는다. 남는 예산은 미사용으로 기록한다(새 λ 추가 없음).

## 7. 실행 (2026-09-12)

- `./tools/pals24_prepare.sh --hours 40` — donor 확인 → verify_metrics → pa/po10/kdv/nf16 gate → PALS24 gate(50 OK) → metric gate(G-M1: NF16 P0/P1/P2/P3 best_hqnr 재평가 = 기록과 Δ 0.0, scene 단위도 0.0; 0.94 min GPU)
  → 재사용 run 반응 진단(pals24 probe, last·best_raw, donor 대비 drift: 예 P2 last ‖ĉ0‖ 중앙값 0.344 px, B_resp ≈ −0.19 I, component MAE 0.571 px, drift 1.17 px) → smoke(3 config, step ≈ 18–21 ms, peak 2.8 GB) → ledger → manifest → `campaign_start.sh --queue config/queues/pals24_s1.txt --hours 40 --label pals24-s1` → 감시자 cron.
  prepare 는 두 번 실패 뒤 세 번째에 기동했다: (1) `po10_diag.py` 의 `@torch.no_grad()` 가 새로 끼운 `fixed_probes` 에 붙어 `response` 가 grad 를 켠 채 `.numpy()` — 데코레이터 위치 수정, (2) manifest 의 `torch.__version__`(TorchVersion 객체) YAML 직렬화 — `str()`. 그 사이 pipefail 함정(파이프 중간 grep/head 의 SIGPIPE)도 로그 파일 경유로 바꿨다.
- **기동 13:00:35** (체인 마감 2026-09-14 05:00). A1 `PALS24_L1E3_…_S1234_…` 예산 gate: used 0.1 + 1.1×(1.24 + 2.0 + 2.0) + 4.0 = 9.86 ≤ 24 → RUN. smoke 예상 1.22 h/run (NF16 실측 1.43–1.52 h + 진단 0.14 h).
- dry run(24 update, `_dry` 버전·별도 ledger, 삭제함)으로 확인: 홀수 step 진단에 `lambda_off 0.001 · loss_off_raw · grad_rec_A · grad_off_A · rho_g · cos_psi · off_unet_grad_absent True` 기록, `budget_status.json` 에 margin 1.1 · DEFERRED 규칙 기록.
- 예상: A block ≈ 5 h(~18:00) → gate 가 λ* 고정(또는 TERMINATE) → stage 2 6벌 ≈ 10 h(09-13 새벽). 캠페인이 끝나면 `work_dir/campaign_gates_enabled.txt` 를 지우고 `python tools/pals24_report.py` 로 표 A/B/C·paired 를 뽑아 **09-13 s1 결과 문서**에 쓴다.
- 주의: 재사용 NF16 run 의 `gradient_diagnostics.jsonl` 에는 ρ_g/cos ψ 가 없다(PALS24 이전 진단) — 표 C 에서 `grad_off_A × λ` 만 계산하고 cos 는 n/a 로 둔다. 25K 진단 checkpoint 는 `epoch-125`(update 25,250) 가 가장 가깝다.
