# NF16 — N2 정합 능력 보존 + native HRMS fitting (16 GPU-h): 명세 검토와 구현 노트 (2026-09-11)

명세: `research_log/PAN_N2_NativeFitting_16GPUh_W112_D123_2026-09-11.md` (protocol `NF16_W112D123_N2LAST_v1`). 사용자 지시: 구현 진행.
구현은 s2 KDV trainer(`train_kdv.py`, `kdv/`)를 그대로 쓰고 NF16 이 요구하는 것만 더했다 — 새 프로토콜 `I-AEQ`, aligner 별도 LR, donor step 검증, 고정 donor 참조 view, 16 GPU-h 예산 gate, native 참조 stress 진단.

## 1. 명세 ↔ 저장소 대조

| 명세 항목 | 저장소 실제 | 처리 |
|---|---|---|
| donor = `PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT` 의 **정확한 50K last** (§2.2) | `work_dir/…/last/model.safetensors`, `last_meta.json` step 50000, aligner.* 22 tensor 105,330 params, aligner sha `db638b55c62f8c62`, head bias (−0.010, −0.002) ≠ 0 | `kdv.donor.source = <run>/last`, `expected_step: 50000`, `expected_sha256`(파일) — trainer 가 `last_meta.json` 의 step 과 sha 를 대조하고 다르면 시작하지 않는다. head 재초기화 없음, donor optimizer state 미사용 |
| backbone W112·D123, 9ch, 단일 HRMS, init 은 seed 별 저장 초기값 (§2.2–2.3) | `work_dir/_kdv_init_w112_d123/init_unet_seed{1234,7777}.pt` (PATrainer._pair_init) — 모든 case 가 같은 tensor | 그대로 |
| optimizer: backbone 1e-4 / **aligner 1e-5**, wd 0.01, cosine+warmup100, batch 48, fp32 (§2.3) | PO10 R200 resolved config 와 동일 (확인) + `kdv.aligner_lr: 1e-5` (param group 별 lr, 같은 scheduler 배율) | 구현 |
| P0 aligner·sampler 없음 (§3) | KDV `A-ID` (`PAModel(sampler=False)` = B0 그래프) | 그대로 |
| P1 frozen: requires_grad False + optimizer 밖, hash·buffer 불변 (§2.3) | KDV `A-FR` (`freeze`, `assert_param_disjoint`, 매 평가 hash 검사) | gate G1 로 확인 |
| P2–P4: c0 를 detach 하지 않음, L_rec 이 U-Net·aligner 양쪽 (§4.1) | KDV `A-FT` (`kdv_forward(aligner_live=True)`) | gate G1 |
| P3/P4 **I-AEQ**: 복원은 매 update native, 홀수 update 에 P_ε 를 aligner 에만, L_off = mean\|ĉε+ε−sg(ĉ0)\|, λ_off 0.01 즉시(ramp 0), ĉ0 재사용, U-Net forward 1회 (§4.2, §5.2) | 신규 프로토콜 `I-AEQ` (`kdv/protocol.py`, `train_kdv._step`): `prepare_view` 는 native, 홀수 step 에 `sample_offsets(b=2)` → `W(P,ε)` no_grad → `predict_c`(live) → `offset_loss(…, stop_reference=True)`; `aux.offset_ramp_updates: 0` (`pa.offset.lambda_off` 가 ramp ≤ 0 이면 상수) | gate G1: U-Net 입력 native, forward 1회, L_off→aligner 만, target detach |
| ε ~ disk(b=2), 전용 RNG, P3/P4 동일 ε (§4.3) | `pa.offset.sample_offsets`, `RNGState` checkpoint, `corruption_seed_offset 2000` → seed 1234 면 3234 (P3·P4 같은 열; 홀수 update 에서만 소비) | gate |
| P4 A3 geometry: λ 0.01·min(1,(t+1)/5000), native P̃0 vs GT, aligner 에만 gradient (§4.4) | KDV `aux.geometry_weight 0.01, ramp_updates 5000` → `pa.losses.direct_geometry_loss` (A3 수식 그대로), support guard margin 4 | gate G2: aligner grad 있음·U-Net grad 없음, 알려진 변형 대조 통과 |
| HQNR 네 view: raw_original / raw_v64 / aligned_self_v64 / **aligned_fixedN2_v64** (§6.1) | 세 view 는 PATrainer; 네 번째는 `train_kdv._extra_views` (고정 donor 복사본 `ref_aligner` 의 native 예측 c_D 로 `W(P_raw, c_D)` 참조, 같은 V64, 매 평가 hash 검사). `checkpoint_metrics.csv` 의 `aligned_fixed_v64.*`, `scene_metrics.csv` view 행. P1 은 self == fixed | 구현 |
| checkpoint: last(정확한 50K)·best_raw·best_aligned·best_rr_val (§7.1) | KDV export 그대로 (`best_rr_val` = valid_wv3.h5 1,080 patch, 학습·RR test 와 분리된 split) | 그대로 |
| 예산 16 GPU-h, 시작 gate used + 1.2·proj + reserve ≤ 16 (§10) | `kdv.budget` → `train_kdv._budget_gate` (ledger `work_dir/_nf16_budget/ledger.json`; P0·P1 은 required, 나머지는 gate; DEFERRED_BUDGET → exit 4, 체인은 재시도 안 함), `_finish_ledger` | 구현 |
| 반응·closure·shortcut 대조·stress 를 **last** 에서, stress 참조는 native P / 고정 donor (§8) | `tools/po10_diag.py` 가 kdv run 을 읽고(`--ckpt last --out po10_diag_last`), `--native-reference --ref-run <donor> --ref-ckpt last` 로 `stress_hqnr_native_fr512.csv`(raw = 원 P, aligned = W(P, c_D), V96) 추가. `_upload.sh` 의 `NF16_*` hook 이 자동 실행 | 구현 |
| 이름 `NF16_P{0..4}_W112_D123_WV3_S{1234|7777}_N2LAST_v1` (§14) | `kdv.check_run_name: false` 로 명세 이름 사용; 시트 범주 ㉑ NF16 (`NF16_*`, KEEP) | 구현 |
| P0 재사용 조건 (§3) | s1 에 같은 block 의 W112·D123 NOALIGN 이 없다(s3 의 `BASE_W112_D123_MSPAN_*` 는 다른 서버·default trainer) → 새로 학습 | P0 학습 |

명세가 채우라고 한 값: donor 경로·step·sha(위), evaluator 2026-09-10.5·FR mat20 manifest(run 폴더 `baseline_manifest.json`·`selection_roi_manifest.json`), P0 신규 학습.

## 2. 구현 지도
- `kdv/protocol.py`·`kdv/registry.py`: `I-AEQ` (native 복원 + aligner 전용 연습; A-FT/A-SC + offset_weight>0 에서만), `offset_ramp_updates`, I-NATIVE-TRANSFER 에 offset 0.
- `train_kdv.py`: `aligner_lr`, donor `expected_step`, `eval.fixed_reference_from_donor` → `ref_aligner`/`_extra_views`, `budget` gate/ledger, I-AEQ offset 연습 분기, 진단에 `grad_off_A`·`grad_geo_A/F`.
- `train_pa.py`: `test_full` 에 extra-view hook (`extra_view_names`/`_extra_views`), 로그·`last_full_metrics` 에 포함.
- `pa/offset.lambda_off`: ramp ≤ 0 → 상수.
- `tools/po10_diag.py`: kdv run 지원, `stress_hqnr_native`, `--out`.
- `tools/gen_nf16_configs.py` → `config/NF16_*.yaml` 7벌, `config/queues/nf16_s1.txt`; `tools/nf16_unit_tests.py` (G0–G2); `tools/smoke_cases.py` (I-AEQ·geometry); `tools/nf16_prepare.sh`; `gspread/sheet_categories.py` ㉑ NF16; `tools/_upload.sh`.

## 3. 큐 (약명 → 세팅)

| run | case | aligner | 입력 | loss |
|---|---|---|---|---|
| `NF16_P0_W112_D123_WV3_S1234_N2LAST_v1` | P0 NOALIGN | 없음(sampler 없음) | 원 PAN + M | L_rec |
| `NF16_P1_W112_D123_WV3_S1234_N2LAST_v1` | P1 FROZEN | N2 last 고정 | native 보정 PAN + M | L_rec |
| `NF16_P2_W112_D123_WV3_S1234_N2LAST_v1` | P2 FT-REC | N2 last 학습(LR 1e-5) | native | L_rec |
| `NF16_P3_W112_D123_WV3_S1234_N2LAST_v1` | P3 FT-EQ | 학습 | native (P_ε 는 aligner 만, 홀수 update) | L_rec + 0.01·L_off |
| `NF16_P4_W112_D123_WV3_S1234_N2LAST_v1` | P4 FT-EQ-GEO | 학습 | native | L_rec + 0.01·L_off + λ_geo(t)·L_geo |
| `NF16_P3/P4_…_S7777_…` | 반복 pair | 같은 donor, U-Net init seed 7777 | | 예산 gate 통과 시 |

공통: W112·D123 · 50K · AdamW 1e-4/1e-5 · wd 0.01 · cosine · batch 48 · fp32 · 평가 eval_epoch 5(≈1K) + 정확한 50K.

## 4. Gate 결과 (s1, 2026-09-11)
| Gate | 내용 | 결과 |
|---|---|---|
| G0 | donor `…N2_OFFSG…/last`: step 50000, aligner 22 key·105,330 params (sha `db638b55c62f8c62`, 파일 `9d4cbf21…`), head bias ≠ 0; W112·D123 2.6589 M; margin 4 = aligner_margin(R=2) | `tools/nf16_unit_tests.py` OK |
| G1 | P0 sampler 없음·Δ=0 / P1 requires_grad False·optimizer 밖·step 뒤 weight+buffer hash 불변·영상별 Δ / P2–P4 L_rec→aligner grad(c0 detach 없음) / P3 I-AEQ U-Net 입력 native·L_off→aligner 만(U-Net grad 없음)·target detach·λ 즉시 0.01 / update 당 U-Net forward 1회 / ε disk(2) 통계(mean 0, std 1, 반경 4/3, ≤2) / P3·P4 같은 ε 열 | OK |
| G2 | P4 L_geo→aligner grad·U-Net grad 없음·유한, support guard 11→4, 알려진 변형 대조(GT 구조로 만든 PAN 을 (1,0)/(0,−1) 옮기면 −∇L_geo 가 되돌리는 부호) | OK |
| 기존 gate | pa_unit_tests · po10_unit_tests · kdv_unit_tests | 전부 통과 |
| G3 smoke | 7 config 실배치: step 52–59 ms(offset 연습 update 55–59 ms), peak 2.66–2.81 GB (s1 4090, GPU 유휴) → 50K ≈ 학습 0.75 h + 평가 ≈ 50 min ≈ **1.6–1.8 h/run** (명세 예약 1.6–2.0 h 와 일치) | OK |
| dry run (300 updates) | s1, 300 updates (P0·P1·P3·P4, 별도 ledger `_nf16_budget_dry`): 전부 rc=0, 4 tag export, `po10_diag_last`(P3: 반응·closure·shortcut 대조·stress + native 참조 stress, P0: native 참조 stress 만) 생성. 확인한 것 — P1: donor step 50000 검증 통과, `share_correction True`, **aligned_self == aligned_fixedN2 (0.9470 = 0.9470)**(명세 §6.1 sanity); P0: aligned_self == raw_v64, fixedN2 view 없음(aligner 없음). P3/P4: 홀수 update 만 offset 연습(`eq_exercise` 0.50), L_off 0.18·closure 0.49 px(300 update 시점), step 59–62 ms(P1 56 ms); P4: step 0 진단에서 L_geo→aligner grad 0.28, U-Net grad 0 (aligner 에만), λ_geo(300) = 6e-4. 예산 ledger RUNNING→FINISHED_TRAIN 기록. **관찰**: 300 update 만에 P3/P4 의 native Δ̂(FR) 가 donor 의 (+1.27, −0.79) 에서 (+0.42, −0.78) 로, 반응 기울기가 −0.98 에서 −0.60 으로 움직였다 — U-Net 이 초기(흐린 출력)라 L_rec 가 aligner 를 끌어당기는 초기 구간의 현상으로, 명세 H2(망각) 가 50K 에서 어떻게 되는지가 바로 P2/P3 비교의 내용이다. 이 값으로 판정하지 않는다. |
| 검토 반영 후 재검증 (2026-09-11 01:2x–01:5x) | 수정본으로 3건 dry 재실행 — **s2 A-FR**(`S2W112D123_A1_IA_AFR_N0_OFF_G0_s1234_dry`, `eval` 키 없음) rc=0 · `aligner_hash0 = 101a3ebb…`(donor) · `share_correction true` → 지적 1 의 회귀가 실제로 사라졌다. **P0** rc=0, `checkpoint_metrics.csv` 에 `aligned_fixed_v64.*` 4열 생김(참조 = N2 last, margin 4, step 50000). **P3** rc=0, `gradient_diagnostics.jsonl` 에 step 0(짝수) 과 **step 1(홀수, offset 연습)** 두 행 — `grad_off_A = 1.892` 로 지적 5 가 해결됐다. P0 의 native 참조 stress 진단은 `stress_hqnr_native` 안에서 forward 가 graph 를 남겨 `numpy()` 가 막혔던 것을 `torch.no_grad()` 로 감싸 재실행 → rc=0 (ε 5개 × 20장 전부 적격) |

## 4.1 반영 후 추가 검토 (다중 agent 재검증, 2026-09-11)
수정 5건을 각각 **반증**하도록 검증한 결과 4건은 확인, 예산 gate(2번)에서 다음이 남아 있어 함께 고쳤다.

| 남은 결함 | 왜 문제인가 | 수정 |
|---|---|---|
| 죽은 시도의 GPU 시간이 사라진다 | OOM·SIGKILL·재부팅이면 `_finish_ledger` 가 못 돌아 항목이 `RUNNING` 으로 남고, 재시작 gate 는 그것을 `run#k` 로 보존하지 않아 `used` 에서 빠졌다 (실측 1.3 h 가 0 으로) | `RUNNING` 으로 남은 이전 시도도 `PREV_CRASHED` 로 보존하고, 그 시간을 trainer 가 주기적으로 쓰는 `memory_and_throughput.json` 의 `elapsed_hours`(없으면 `started`→now)로 회수한다 (`_crashed_hours`) |
| 반복 pair 가 한쪽만 묶여 있었다 | `pair_with` 가 P3(7777) 에만 있어, P3 가 예산으로 DEFERRED 돼도 P4(7777) 는 단독 gate 를 통과할 수 있었다 (명세 §10 "두 run 한 묶음" 위반) | 양쪽 config 가 서로를 `pair_with` 로 가리킨다. 이중 계상은 `_projection` 이 **이미 FINISHED 인 상대는 0** 으로 보아 막는다 (그 실비는 `used` 에 있다) |
| 예상치가 case 차이를 뭉갠다 | 남은 필수·pair 의 예상이 "완료 run 평균"이라 P0(aligner 없음)과 P3/P4(offset 연습+geometry)를 같은 값으로 봤다 | `budget.projected_map` 에 명세 §10 예약(1.6/1.6/1.8/2.0/2.0)을 넣고, 지정 run 은 map → 완료 평균 → smoke throughput 순으로 본다 |
| 진단 GPU 시간 과소 계상 | `_upload.sh` 가 `po10_diag` 만 쟀다 — 앞선 `eval_fr_paperset`(≈35 s)·`pa_diag`(≈3.4 min) 가 빠져 run 당 ≈4 분(7 run ≈ 0.45 h)이 ledger 밖이었다. 재기동 시 같은 run 을 다시 진단하면 덮어써 사라졌다 | run 당 타이머를 `pa_diag` 앞으로 옮기고 `eval_fr_paperset` 시간을 NF16 run 수로 나눠 분담, `diag_<run>` 은 **누적**(`runs` 횟수 기록) |
| 큐 주석이 config 와 달랐다 | "P0·P1 만 필수" 라고 적혀 있었지만 생성기는 s1234 사슬 5개 전부 `required: true`(경고만) | 큐 머리말을 실제 동작(사슬 전부 required, 반복 pair 만 DEFERRED)과 gate 식·예상치 순서로 고쳤다 |
| `A-SC` + `fixed_reference_from_donor` 의 오류 메시지 | guard 가 `aligner is None` 이라 A-SC(scratch aligner)는 통과한 뒤 `donor_manifest=None` 으로 `TypeError` | guard 를 `self.donor_manifest is None` 으로 (뜻대로 A-ID·A-SC 둘 다 ValueError) |
| stress 적격 상한 주석의 산술 | `pa/evalviews.py`·`po10_diag.py` 주석이 `|ε|+|ĉ| ≤ 38` — 저해상도 support 54 가 이미 warp taps 2 를 포함하는데 두 단계 4 를 또 뺐다. 코드값은 96−8−44−4 = **40** | 주석을 40 과 분해(52 + 2×2)로 고치고, `stress_hqnr_native` 결과에 `rule`·`max_eligible_two_stage`·`max_eligible_reference_shift`·`aggregation` 을 남겨 예전 `stress_roi_manifest` 규칙과 구분한다 |

남은 한계(고치지 않음, 기록만): native stress 의 적격 판정은 **모델 입력**의 두 단계 support 와 고정 참조 support 를 보지만, 참조 저해상도 재생성 경로의 상한은 `scene_views` 의 `|c_D|∞ ≤ 8` 로만 걸린다(V64 기준이라 V96 보다 보수적). `_projection` 은 이 run 의 `projected_hours` 를 다른 run 에 적용하지 않는다(의도).

## 5. 실행
```bash
./tools/nf16_prepare.sh --no-start     # donor 확인·gate·config·smoke·ledger
./tools/nf16_prepare.sh                # + 체인 기동 (queue config/queues/nf16_s1.txt, 감시자 cron)
```
결과·진단: run 폴더 `checkpoint_metrics.csv`(네 view + RR + rr_val), `results/po10_diag_last.json`(반응·closure·shortcut 대조·stress native), `results/pa_diag.json`, `budget_status.json`; 예산 `work_dir/_nf16_budget/ledger.json`. 시트 ㉑ NF16.
판정(명세 §11): 인과 비교는 last 끼리; best_raw 는 초기 checkpoint 일 수 있음(기록용); 반응 보존은 donor 와 같은 정의의 closure/B; 위치 근거는 RR GT 와 독립 진단; 두 HQNR(raw / aligned) 의 trade-off 를 숨기지 않는다.

## 6. 사용자 검토(2026-09-11) 5건과 반영
| # | 지적 | 확인 | 반영 |
|---|---|---|---|
| 1 (P1) | 공통 trainer 의 fixed-reference 분기에 정책 `elif/else` 가 붙어, `fixed_reference_from_donor` 없는 기존 s2 KDV(A-FR/A-FT) 초기화에서 donor 를 None 으로 덮음 | 소스 확인 — 실제 회귀 | 정책 `if/elif/else` 를 먼저 닫고 참조 aligner 는 뒤에 별도 블록으로. `eval.reference_donor`(A-ID/A-SC 용 평가 전용 donor) 추가. gate: 블록 구조 검사 + s2 A-FR dry run 재실행 |
| 2 (P1) | 예산 gate 가 다음 run 하나만 검사(반복 pair 둘 다 50K 보장 안 됨), P0/P1 우회, smoke 예상치 미사용, export·진단 비용 누락, 재시작 시 항목 덮어씀 | 맞음 | `budget_decision(used, proj_this, proj_remaining, reserve, total, required, proj_pair)`: 예상 = used + 1.2·(이 run + 남은 필수 사슬 + pair) + reserve. 생성기가 s1234 사슬에 `remaining_mandatory`(뒤따르는 P0→P4), P3(7777) 에 `pair_with = P4(7777)` 을 넣는다. 필수 run 은 초과 시 **경고만**(명세: 필수 case 학습량 유지), 반복은 DEFERRED. 예상치: config → 완료 run 평균 → prepare 의 smoke `throughput.projected_run_hours_50k`. 재시작은 이전 시도를 `run#k` 로 보존해 used 에 포함. export 뒤 `hours_total`(학습+평가+export), `_upload.sh` 가 진단 시간을 `diag_<run>` 으로 더한다. 사용자 반례(13.5 h + 1 h 씩)는 pair 검사로 DEFERRED — gate 통과 |
| 3 (P1) | native 참조 stress 가 c_D 만 검사하고 두 단계(ε→ĉε) support 를 안 봄; 부적격 장면을 빼고 평균 | 맞음 | `native_stress_eligible(H,W,ε,ĉε,c_D)`: `two_stage_support_mask` 로 V96 안 관측만 읽는지 + 참조 W(P,c_D) support. 집계 `aggregate_native`: 전부 적격일 때만 값, 아니면 None·`eligible_all False`·n 유지(부분 평균은 `*_eligible_subset` 키). 반례(ĉε=200 px) 부적격 — gate 통과. 기존 `stress_hqnr` 집계에도 `eligible_all` 추가(기존 키는 유지) |
| 4 (P2) | P0 에 공통 N2 참조 평가 없음; donor 생성이 전역 RNG 를 소비해 P0 만 데이터 순서가 다름 | 맞음 | P0 config 에 `eval.reference_donor`(N2 last, margin 4) → `aligned_fixed_v64` 가 P0 에도 생김. donor/skeleton/CovHead 생성을 `fork_rng` 로 감싸 전역 RNG 불변 — gate: donor 로드 전후 `get_rng_state` 동일 |
| 5 (P2) | 진단(diag_every 배수, 짝수)과 offset 연습(홀수)이 겹치지 않아 `grad_off_A` 가 50K 동안 기록되지 않음 | 맞음 | `is_diag_step`: I-AEQ 면 diag_every 배수 다음 홀수 step 도 진단 → `grad_off_A` 기록. 로그 해석 주의(step 0 의 aligner grad 0 은 zero_module 때문)는 §4 dry run 에 적음 |
