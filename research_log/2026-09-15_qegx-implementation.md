# QEGX (s3·s4, W104·D121) 구현·검토 노트 — 2026-09-15 저녁

계획: `research_log/PAN_QEGX_S3_S4_W104D121_Experiment_Plan_2026-09-15.md` (+ `S3_Run_Handoff_QEGX_2026-09-15.md`, `S4_Run_Handoff_QEGX_2026-09-15.md`).
전제 구현: PAKD50 (`2026-09-14_pakd50-implementation.md`), s4 골격 이식 (`2026-09-15_pakd50-s4-w104d121-implementation.md`), QEDGE9 (`2026-09-15_qedge9-implementation.md` §1–§9).
작업 서버: s1 (개발·검사·smoke·40-update 실학습). s3/s4 기동은 사용자 조작(§7).

## 0. 결론 요약

- 계획 §4 의 신규 6 case(QX50 · QEC3 · QE50_B005 · LFQE50 · **QER50 · QERS**) 를 registry/trainer/generator 에 등록했고, §5·§6 의 명시 큐(s3 15 run, s4 14 run) config 29 벌을 만들었다. 계획 §11.3 의 12 항목 중 코드로 닫을 수 있는 것은 unit gate K34–K38 로 닫았다(총 136 검사 ALL OK). smoke 6 벌·QER50 40-update 실학습 통과(§6).
- **QER50/QERS 는 `kdv.edge_route`** 로 구현했다(계획 §4.5 "별도 명시 경로"). total 은 JQ 와 같은 all-edge 식이고, backward 뒤 A 의 `.grad` 에서 λE·mean((1−g)E_i) 의 A gradient 를 뺀다(`_apply_routing` 과 같은 보정 방식). registry 는 `edge_route` 와 `edge_gate`/`routing` 의 결합을 거부한다 — "gate 를 끄고 QER 이라 부르는" 우회가 없다.
- **s3 묶음은 15 run 전부 v2** 로 만들었다. s5 QEDGE9 의 `PAKD50_{J0,JQ,QE50}_W104_D121_WV3_T0_S2026_FRESH50_v1` config(QEDGE9 campaign·9 h ledger) 와 이름이 같아지기 때문이다. 학습 정의는 같고 캠페인·예산 metadata 만 다르다(계획 §12 "v1/v2 를 덮어쓰지 않는다"). s4 신규 14 run 은 v1(충돌 없음: s4 의 옛 QEDGE9 v1 config 는 17:20 에 제거됐고, s1 은 v2).
- 시간 정책: 캠페인 `QEGX_A104D121_S3S4_20260915_v1` / branch `A104D121_T0FIX_QEGX_v1`, 자체 ledger `work_dir/_qegx_budget/ledger.json`, `total_gpu_hours 1000`(도달 불가) + `required true`(경고만), `training_deadline` 없음, `time_policy{mode: no_hard_limit}`(기록용) — 계획 §12 "계획 필드를 trainer schema 에 그대로 넣지 말고 warning-only 규약으로 사상". gate 편성은 QEGX 항목을 admission 에서 제외한다.

## 1. 약명 → 세팅 (계획 §4; 보고서에는 이 표로만 읽는다)

| case | 정책 / rec | GT edge — U 가 받는 계수 / A 가 받는 계수 | 구현 키 | 비고 |
|---|---|---|---|---|
| `J0` / `JQ` / `XJ` / `J_R3_NOEDGE` / `J_QB005` / `JE0` / `LF0` / `LFQ` / `LFX` | 기존 정의 그대로 | JQ 1/1 · XJ 1/1(soft 없음) · JE0 1/0(`routing.qE 0`) · LF* 는 25K 뒤 A 동결 | — | QEGX branch 로 다시 만든 것(campaign/budget metadata 만 다름) |
| `QE50` / `QES` | J / R3 α1 β0.1 | g / g (low_q) · g_shuffle / g_shuffle | `edge_gate low_q` / `shuffle` | QEDGE9 정의 재사용 |
| **`QX50`** | J / **R1**(hard-only, β=0) | g / g | `edge_gate low_q` + rec R1 | "QE50 에서 β=0" 의 동치 — XJ 아님 |
| **`QE50_B005`** | J / R3 α1 **β0.05** | g / g | `edge_gate low_q` + `rec.kd_weight 0.05` | J_QB005 와 대응쌍 |
| **`QEC3`** | J / R3 | c_E3 / c_E3 | `edge_gate const`, `c_E_file work_dir/_qegx/qec3_cE.json`, pilot `PAKD50_J0_W104_D121_WV3_T0_S2026_FRESH50_v2`/last/50000 | s1 QEDGE9 QEC 의 `work_dir/_qedge9/qec_cE.json` 과 별도; pilot identity 를 trainer 가 대조 |
| **`LFQE50`** | LF(freeze_from 25000) / R3 | g / g(t<25000), 0(이후 A 동결) | `aligner_schedule{freeze_from 25000}` + `edge_gate low_q` | 0–24999 는 QE50 과 정의·ε RNG 동일(K37) |
| **`QER50`** | J / R3 | **1 / g** | **`edge_route low_q`** | total = JQ; A `.grad` −= ∇φ[λE·mean((1−g)E_i)] |
| **`QERS`** | J / R3 | **1 / g_shuffle** | **`edge_route shuffle`**(perm 51515) | QES 와 같은 셔플 라벨 |

g = 1[q_T < θq], q_T 는 고정 T0 aligner 의 AXIS16 probe(cue 자산 `assets/qedge9/cue_T0_AXIS16_v1`, θq 0.327613) — QEDGE9 노트 §2. hard/soft 의 e 배분(wH≥1)·λE0·τR·α·A LR 은 전부 그대로(계획 §2).

## 2. edge_route 의 수식과 구현 (계획 §4.5·§11.3-5/6)

계획: G_U = ∇θ[L_H + L_K + λE·E(1)], G_A = ∇φ[L_H + L_K + λE·E(g) + L_O], "두 식을 더해 하나의 total 로 backward 하면 reconstruction 이 이중 계산되므로 금지".

구현(`train_kdv.py`):
- `_step`: `stat_kind == edge` 이고 `self.edge_route` 가 있으면 `loss_stat_raw = output_edge_loss(y, gt)`(JQ 와 **같은 식**, bitwise) 그대로 total 에 넣고, per-sample `E_i = output_edge_loss_per_sample` 와 cue 표의 g_i(sg) 로 `info["_edge_route_hi_t"] = mean((1−g_i)E_i)` 를 남긴다. λ 를 곱한 `_edge_route_hi_w_t = λE·mean((1−g)E_i)` 가 "A 가 받지 않을 몫".
- 학습 loop: `routed_now = (self._routed or self._edge_routed) and aligner_active(step)` → `backward(total, retain_graph=routed_now)` → `_apply_edge_route`: `torch.autograd.grad(extra × scale, A params)` 를 A 의 `.grad` 에서 뺀다. scale = 1/accum × scaler.get_scale() (AMP 도 `_apply_routing` 과 동일 규약; 이 캠페인 config 는 mixed_precision 없음). **optimizer.step 은 한 번**(K36 이 소스 검사).
- 동결 구간(LF 결합은 이번 큐에 없지만) 은 `aligner_active` False → 보정 자체를 하지 않고 A `.grad` 도 None.
- registry(`kdv/registry.py`): `edge_route{mode low_q|shuffle, asset, perm_seed, theta_source}` — EDGE-H 위에서만 · trainable A · Teacher 필요 · asset 필수 · shuffle seed 51515 고정 · `edge_gate`/`routing`/TRI/extra/control/geomKD 와 결합 거부. spec 에 `edge_U 1.0, edge_A 'g'|'g_shuffle'`. 토큰 `EDGEHR50`/`EDGEHRS`, `describe` 서술.
- 자산 적재는 `EdgeGate.load` 재사용(Teacher aligner hash·margin·train h5 sha·feeder 계약·재개 asset_id 대조). `return_meta` 필수. manifests(`kdv_config_resolved.json`·`calibration_resolved.json`·architecture aligner) 에 `edge_route` 요약, 고정 진단 batch 의 per-term 분해에 `LEwA`(= LEw − hi; A 가 실제 받는 routed edge) 추가 — 계획 §13 "U all-edge 와 A routed-edge 및 gradient 규모 구별".

검사(K36, CPU 실제 `_step`+backward+보정, T0 복사 A + 잡음 U): g=1 → total 이 JQ 와 bitwise 같고 A/U `.grad` == JQ(상대 1e-6, 보정량 0) · g=0 → A `.grad` == JE0(routing qE=0; 상대 1e-5), U `.grad` == JQ 의 U · 혼합 g=(1,0) → A `.grad` == ∇φ[L_H+L_K+λE·mean(g E_i)+λoff L_O] 를 autograd 로 직접 구한 값(max|Δ| 1e-11 / ref 1e-4), U == ∇θ total · QERS 셔플 표 조회, gate 조회가 ε RNG 를 소비하지 않음(gen 상태 == JQ) · meta 없으면 ValueError.

## 3. 나머지 case 의 동치 검사 (계획 §11.3-2/3/4/8)

- K37: QX50 total == QE50(g=1) total − L_K · XJ == JQ − L_K · QX50 의 soft 항 정확히 0 · QE50_B005 의 L_K == 0.5 × QE50 의 L_K(β 선형) · QE50(g=0) total == J_R3_NOEDGE total(edge 0, hard/soft/offset 유지).
- K37 LFQE50: update 1 은 QE50 과 total·ε 열 동일 · `aligner_active` 24999 True / 25000 False · 동결 update 25001(odd) 은 offset 연습 생략(L_O 0)·Δ graph 없음·U 는 gated edge 를 계속 받음(λE L_E > 0)·AdamW step 뒤 A hash 불변(grad None → moment·WD 없음). 25K 전 경로 일치·경계·동결 뒤 불변은 QEDGE9/s5 의 K12·K14 와 같은 기전(D/LF 는 `aligner_active(step)` 순수 함수).
- 저오차 patch 의 wH≥1·a_T=0→soft 0 은 K24 그대로(§11.3-4). Teacher/GT/cue/gate detach 도 K24·K36.
- §11.3-9(zero-init 오판): K36/K37 의 stub 은 T0 A 복사 + U 잡음(비퇴화) — 0 gradient 로 인한 "경로 차단" 오판 없음. §11.3-10(AMP/clip 순서) 은 이 캠페인 config 가 mixed_precision·clip 없이 도는 것을 확인(스케일 규약은 `_apply_routing` 과 동일 코드). §11.3-12(resume) 은 QEDGE9 F04 의 `exact_resume`(K29) 을 QEGX config 전부에 켰다.

## 4. 캠페인·편성·시간 (계획 §5·§6·§9·§12)

- 생성기 `tools/gen_pakd50_configs.py`: `QEGX_*` 상수, BACKEND `QX50/QE50_B005/QEC3(const3)/QER50/QERS(edge_route)`, CASES 6, `PRIORITY_BY_SERVER["s3"]` = 15 run 이름(v2), `["s4"]` = E0 5 + 14 run 이름(v1), `QEGX_ITEMS_BY_SERVER`, `branch_for`(QEGX 전용 case 또는 서버 QEGX 목록 → "QEGX"; QEGX 가 QEDGE9 보다 먼저), `cue_ready`(QEC3 → `_qegx/qec3_cE.json`, QER/QX/LFQE50 → cue 자산), `measured_hours_all` 이 QEGX ledger 도 합침, `server_id('s3(5090)') → 's3'`(gate·스크립트가 사용; 시트 탭 이름은 그대로).
- **주의(계획 §12)**: bare 항목 `QE50@W104_D121` 은 기존 QEDGE9 자동 규칙(9 h ledger) 으로 간다. QEGX 큐·extra 에는 run 이름만 쓴다 — `qegx_switch.sh` 가 QEGX 항목이 run 이름인지 검사한다. `kdv_block` 은 QEGX 전용 case 를 다른 branch 로 만들면 SystemExit.
- 예약(§9.2, `reservation_h = 1.10×ref + 10/60`): s3 W104 는 미측정 → 같은 유형 W112 실측 대용(J0 1.16 / JQ 1.34 / XJ 1.33 / J_R3_NOEDGE 1.30; QEC3·J_QB005 는 JQ), cached-gate 1.50 h(*) → 15 run **25.2040 h**(계획 25.2039). s4 는 §9.1 의 W104 Sheet 실측(NA0 1.27 / J0 1.17 / JQ 1.38 / XJ 1.37 / F0 1.13; 종전 큰 골격 대용값 교체) + LF* 유형 대용 + gate 1.50(*) + routing 1.80(*) → 14 run **26.2803 h**(계획 26.2803). 확인 seed 4321/3407 은 이제 `allowed_seeds` 에 들어가(명시 표) confirm 가예약이 아니라 같은 기준값을 쓴다. 같은 서버 첫 실측이 나오면 `measured_same_case` 로 자동 교체.
- `--plan` 표는 branch 열을 추가했고, 마감 admission 제외 branch 의 합을 따로 보인다(s1 에서 s3/s4 표를 보면 E0 5 run 이 planned 로 나오지만 그 서버에서는 terminal).
- 시트: `integrated_label` 이 branch 에 QEGX 가 있으면 `PAKD50 / <case> / A104D121 / QEGX / FRESH50`. Notes 에 `edge_route=…; edge_U=1; edge_A=g|g_shuffle` 또는 gate run 의 `edge_U=g; edge_A=g` + `cE_pilot=…/last@50000; cE_pilot_sha=…`, 그리고 QEGX branch 공통 `campaign; beta; freeze_from; seed; time_policy`. 탭은 `gspread/server.txt` 그대로(`WV3-s4`; s3 는 그 서버의 server.txt 값).

## 5. 도구

- `tools/qegx_switch.sh` (s3·s4; `--dry-run` / `--pilot`(s3) / `--refresh-controls`(s4)): ① gate K01–K38 ② cue verify(그 서버 T0) ③ 명시 순서 config == 생성기 + 완료 run 의 `verified_complete`(s4 의 J0/JQ/XJ v1 재사용 전제 §6 — 불통과면 보고하고 `--refresh-controls` 로 **v3** 를 QEGX branch 로 extra_priority 에 추가; 옛 v1 값을 고르지 않는다) ④ `--plan` ⑤ (apply) mandatory(PAKD50 + `_qegx/mandatory_runs.txt`)·reservations, 비 QEGX extra 보존 분리 ⑥ (s3) c_E3 pilot(없을 때만) ⑦ 마감 파일 제거, runner 살아 있으면 그대로(다음 gate pass 부터 새 순서), 없으면 `config/queues/qegx_<srv>.txt` 로 기동, 대기자 기동.
- `tools/qegx_waiter.sh`: 5 분마다 (s3) J0 S2026 v2 의 exact50K + results 가 생기고 c_E3 가 없으면 `qedge9_cue.py pilot --branch qegx` · chain 없음 + 준비된 미완 run 있으면 재기동 · 상태 `work_dir/_qegx/status.json`.
- `tools/qedge9_cue.py pilot --branch qegx --run PAKD50_J0_W104_D121_WV3_T0_S2026_FRESH50_v2 --tag last`: QEDGE9 pilot 과 같은 식(c_E = Σ g E_pilot / Σ E_pilot, calibration view) 인데 identity(J0·W104·seed 2026·v2·exact50K) 와 출력 파일이 다르다. `status` 는 두 파일을 다 보인다.
- gate `tools/campaign_gate.py::gate_pakd50`: exempt = branch ∈ {QEDGE9, QEGX}, blocked = `cue_ready` 아님, 완료 run 의 `verified_complete` 를 두 branch 모두 기록, 대기 메시지에 c_E3.

## 6. 검증 기록 (s1, 2026-09-15 저녁)

- unit gate `tools/pakd50_unit_tests.py`: K01–K38 **136 검사 ALL OK**(K21/K22/K26 의 s3/s4 편성·예약 assertion 을 새 순서·§9.1 실측값으로 갱신; K22 의 옛 `--confirm` 묶음은 정의 그대로, 예약만 plan reference). 기존 config(s5/s1 QEDGE9, s4 E0) 는 생성기 출력과 그대로 같다(회귀 없음 — QEGX 약명 줄은 QEGX config 머리에만).
- smoke `tools/smoke_cases.py`(shared GPU): QER50 S1234 v1 · QERS S3407 v1 · QX50 S2026 v2 · LFQE50 S1234 v1 · QE50_B005 S2026 v2 · QE50 S4321 v2 — 전부 통과(1.9036 M, peak ≈3.05 GB, t_native 67–100 ms). QEC3 는 c_E3 파일이 없어 smoke 대상이 아니다(s3 pilot 뒤 trainer 가 적재 검사).
- 40-update 실학습(QER50 S1234 임시 config, 별도 work_dir/ledger, diag_every 10): §6.1.

### 6.1 QER50 40-update 실학습 (s1 shared GPU, `PAKD50_QER50_W104_D121_WV3_T0_S1234_FRESH50_v1` 을 num_iter 40·diag_every 10·별도 work_dir/ledger 로; 임시 산출물은 지웠다)

- 적재: `[kdv] edge_route low_q: U all-edge / A gated (θq 0.3276133416220546)`; `kdv_config_resolved.json` 의 `edge_route{mode low_q, edge_U 1.0, edge_A g, asset_id 526dbd71…}`, `architecture_manifest.aligner.edge_route = g`, `route_A [1,1,1]`.
- 매 update A 보정: `gradient_diagnostics.jsonl` 의 `edge_route_params_adjusted = 22`(A 의 requires_grad parameter 수) step 1 부터 끝까지 · gate 활성 비율 0.40–0.58(batch 48, θq = 중앙값) · ungated E 평균 0.027–0.038 vs A 가 받는 gated 평균 0.016–0.023.
- A 의 edge gradient 규모(fixed 아님, 매 진단 step): all-edge ‖∇φ λE E(1)‖ vs routed ‖∇φ λE E(g)‖ = step 10 4e-6/3e-6 · 20 1.4e-5/1.1e-5 · 30 1.3e-4/9.8e-5 · 39 1.7e-4/1.2e-4 — routed ≤ all. **step 0 은 둘 다 0**(U 출력 zero-init → edge 항의 A gradient 가 0; 계획 §11.3-9 의 "초기 0 을 경로 차단으로 오판하지 않는다" — 검사 K36 은 비퇴화 상태에서 한다).
- 고정 진단 batch(step 39) per-term: `L0_A 4.2e-3 · LD_A 2.6e-3 · LK_A 1.4e-4 · LEw_A 1.9e-4 · LEwA_A 2.1e-4 · LOw_A 2.0e-4`(LEwA = A 가 실제 받는 routed edge).
- 예산 gate: `RUN`(required → 경고; 임시 config 는 projection 을 뺐으므로 projected NaN 경고) · time_policy `no_hard_limit` 기록 · 최종 step 40 평가·후보 저장·manifests 정상(HQNR 0.9424 는 40 update 값 — 수치로 쓰지 않는다).

## 7. s3·s4 절차 (사용자 조작)

1. s1 에서 push → s3/s4 `git pull`.
2. `./tools/qegx_switch.sh --dry-run` → gate/verify/config/control 검증/예약 표 확인. s4 에서 J0/JQ/XJ v1 의 검증 불통과가 보고되면 `./tools/qegx_switch.sh --refresh-controls`(적용 시).
3. `./tools/qegx_switch.sh` → chain 이 돌고 있으면 그대로 두고(현재 run 은 원 설정으로 마무리) 다음 gate pass 부터 QEGX 순서; DONE 이면 큐로 기동. 대기자가 (s3) J0 v2 뒤 c_E3 를 만들고 QEC3 를 연다.
4. 첫 QE50/QER50 완료 뒤 예약은 실측으로 자동 교체(§4). 실패/NaN 은 자동 반복하지 않는다(runner 규약 그대로).

## 8. 남긴 것·한계

- QEC3 의 실제 c_E3 값·QEC3 config 의 적재 검사는 s3 pilot 뒤에야 확인된다(구조는 QEDGE9 QEC 와 같고 K38 이 파일 identity 를 검사).
- s3 의 W104 시간 기준값은 W112 대용이다(계획 §9.1) — 결과 수치로 인용하지 않는다.
- 계획 §8 의 후속 묶음(LF 다른 seed, e cutoff, θq 민감도, routing 평균 계수 대조) 은 미구현. 계획 §12 의 `case_registry.json` 은 계획용 명세로 두고 코드 schema 로 옮기지 않았다.
- s3 실행 결과와 s5 QEDGE9 의 같은 seed 2026 run 은 같은 정의의 환경·실행 반복이지 새 독립 seed 가 아니다(계획 §5) — 보고서에서 서로 대체하지 않는다.
