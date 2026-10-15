# EDGEBAL (s2·s5, W104·D121) 구현·검토 노트 — 2026-09-16

계획: `research_log/PAN_EDGEBAL_S2_S5_Experiment_Plan_2026-09-16.md` (§10 구현 인계). 동반 handoff/JSON/CSV 는 이 저장소에 없다(계획 §11 산출물은 사용자 측).
전제 구현: PAKD50 · s4 골격 이식 · QEDGE9(`2026-09-15_qedge9-implementation.md`) · QEGX(`2026-09-15_qegx-implementation.md`). 작업 서버: s1(개발·검사·smoke·40-update 실학습). s2/s5 기동은 사용자 조작(§7).

## 0. 결론 요약

- 계획 §4 의 EB case 14 종을 registry/trainer/generator 에 등록했고, §5·§6 의 명시 큐(s2 12 run **v1**, s5 14 run **v2**) config 26 벌을 만들었다. 새 trainer 키는 둘이다: **`kdv.edge_schedule`**(§3.3 DOWN/UP; 0-based update 25000 분기, A 는 계속 학습) 와 **`kdv.edge_weight`**(§3.4 FLOOR/SHUF/REVERSE; 기존 cue 자산의 0/1 표 위에 affine 계수 w = high + (low−high)·g). 상수 배수 r(§3.2) 은 기존 `lam_mult`(λE0 배율; `stat.outer_weight` 에 한 번만 곱) 로 표현한다 — λE0 와 r 을 중복 곱하지 않는다.
- 동치 검사(계획 §10.3) 를 unit gate K39–K43 으로 닫았다(총 152 검사 ALL OK): EB_N0≡J0 · EB_R3E100≡JQ · EB_R3E000≡J_R3_NOEDGE · EB_R1E100≡XJ · EB_N0E100≡J_N0_EDGE · EB_R3E050≡J_QE025 · EB_R3E200≡J_QE10 은 같은 batch 의 total 이 bitwise 같고 U/A gradient 가 같다(§3). 배수·경계·floor·shuffle·reduction 도 §3.
- 캠페인 `EDGEBAL_A104D121_S2S5_20260916_v1` / branch `A104D121_T0FIX_EDGEBAL_v1`, 시간 상한 없음(자체 ledger `work_dir/_edgebal_budget/ledger.json`, `total 1000 h` + `required`(경고만), `training_deadline` 없음, `time_policy{no_hard_limit}` 기록; gate admission 제외). PAKD50 50h·QEDGE9 9h·절대 마감을 상속하지 않는다(K39 가 config 로 검사).
- **주의**: registry spec 의 `edge_weight` 키는 이미 aux λ_edge(`aux.edge_weight`) 가 쓰고 있어, config 키 `kdv.edge_weight` 의 spec 은 **`edge_cue_weight`** 로 저장된다(`kdv_config_resolved.json`·`calibration_resolved.json` 도 같은 키). 계획 §10.2 의 이름은 config 키에서 그대로다.

## 1. 약명 → 세팅 (계획 §4; 보고서에는 이 표로만 읽는다)

| case | rec | GT edge 계수 w_i(t) (λE0 배) | 구현 | 동치 |
|---|---|---|---|---|
| `EB_N0` | N0 | — (edge 없음) | | J0 |
| `EB_R3E000` | R3 | 0 (stat OFF) | | J_R3_NOEDGE |
| `EB_R3E025` / `050` / `075` / `100` / `200` | R3 | 0.25 / 0.5 / 0.75 / 1 / 2 상수 | `lam_mult` → `stat.outer_weight = r·λE0` | 050 = J_QE025 · 100 = JQ · 200 = J_QE10 |
| `EB_N0E100` | N0 | 1 | Teacher eval_only(loss 에 없음; A 는 T0 복사) | J_N0_EDGE |
| `EB_R1E100` | R1 | 1 | soft 없음 | XJ |
| `EB_EDOWN` / `EB_EUP` | R3 | 1→0.5 / 0.5→1 (t<25000 / ≥25000) | `edge_schedule{before, after, switch 25000}` | 평균 0.75 (CONST075 대조) |
| `EB_QFLOOR` / `EB_QFREV` | R3 | 0.25 + 0.5·g / 0.75 − 0.5·g (g = 1[q_T<θq]) | `edge_weight{floor, low .75/.25, high .25/.75, asset}` | 평균 ≈ 0.5 (CONST050 대조) |
| `EB_QFSHUF` | R3 | 0.25 + 0.5·g_shuffle | `edge_weight{floor_shuffle, perm 51515}` | g 의 stratum 셔플 |

전부 J 정책(A-FT·I-AEQ·A LR 1e-5·offset 1e-4 홀수), α1·β0.1·τR·λE0 고정, W104·D121. g 는 QEDGE9 cue 자산(`assets/qedge9/cue_T0_AXIS16_v1`, θq 0.327613) 의 0/1 표 그대로(§3.4).

## 2. 구현 (계획 §10.2 최소 변경 지점)

- `tools/gen_pakd50_configs.py`: `EDGEBAL_*` 상수, BACKEND/CASES 14, `PRIORITY_BY_SERVER["s2"]`(12 run 이름 v1), `["s5"]` = `QEDGE9_S5_ITEMS`(6; 완료분 terminal·실행 중 QE50 S777 은 끝까지) + 14 run 이름 v2, `QEDGE9_ITEMS_BY_SERVER["s5"]` 는 앞 6 만(EDGEBAL 항목이 QEDGE9 ledger 로 가지 않게), `EDGEBAL_ITEMS_BY_SERVER`, `branch_for`(EB case → EDGEBAL), `EDGEBAL_CONTROLS`(§5/§6 재사용 대조: s2 777 → QEDGE9 v2 3 벌, s5 2026/777 → v1; 다른 seed 는 같은 큐의 EB_N0/EB_R3E100) → config `kdv.control_runs`(Notes `control_run_id`), `cue_ready`(QF* 는 cue 자산), `measured_hours_all`(EDGEBAL ledger 포함), `--plan` 문서, 옛 s2 순서 `PREVIOUS["s2_20260915"]`.
- `kdv/registry.py`: `edge_schedule{before, after, switch}`(정확히 세 키, [0,4], switch 양의 정수, before≠after, EDGE-H 위에서만, edge_gate/edge_route/routing/TRI 결합 거부) · `edge_weight{mode floor|floor_shuffle, low, high, asset, perm_seed}`(low≠high, asset 필수, Teacher 필요, shuffle seed 51515, schedule 과 결합 거부 — 계획 §6 "각각 검증 뒤 한 개의 후속 교차만"). 토큰 `EDGEHSD/EDGEHSU/EDGEHF/EDGEHFS/EDGEHFR`, `describe` 서술.
- `kdv/edge_gate.py`: **`AffineEdgeWeight`**(`EdgeGate` 위 wrapper; `load` 는 EdgeGate.load 와 같은 자산·Teacher·데이터·feeder·재개 대조, `weight_for(meta) = high + (low−high)·g` sg). 기존 `gate_low_q`/`gate_shuffle` 0/1 표·함수는 바꾸지 않았다(§10.2-4).
- `train_kdv.py`: `edge_schedule_factor(step)`(step 만의 순수 함수 → 재개 동일), `_step` 의 edge 분기 둘 추가(weight: `mean_i w_i E_i`; schedule: `w(t)·output_edge_loss`, w=1 이면 JQ 와 bitwise 같은 식), 경계 기록 `edge_schedule_events.jsonl`(step·factor·λ_eff·A 활성·A LR), 로그 `stat_edge_w_mean/lambda_eff/sched/raw/eff/gate_frac/E_qlow/E_qhigh`(per-step 진단·EMA), manifests(`edge_schedule`, `edge_cue_weight`). 기존 `stat.ramp_updates` 는 쓰지 않았다(§10.2 "의미 확인 없이 재사용 금지").
- gate/sheet/scripts: `campaign_gate` exempt ∋ EDGEBAL · X열 `PAKD50 / EB_* / A104D121 / EDGEBAL / FRESH50` · Notes `edge_mult, edge_schedule, edge_low/high, gt_hard_always=1, T0_sha, cue_asset_id, seed, release_sha(meta/git_commit), control_run_id, time_policy` · `tools/edgebal_switch.sh`(`--dry-run`; runner 를 죽이지 않는다, 재사용 대조 검증 보고) · `tools/edgebal_waiter.sh` · 큐 `config/queues/edgebal_{s2,s5}.txt`.

## 3. 검사 (계획 §10.3 표 ↔ K39–K43)

| §10.3 | 검사 | 결과 |
|---|---|---|
| EB_N0 / EB_R3E100 동치 · E0 / β0 | K41: 7 쌍(EB_N0/J0 · R3E100/JQ · R3E000/R3_NOEDGE · R1E100/XJ · N0E100/N0_EDGE · R3E050/J_QE025 · R3E200/J_QE10) 같은 batch 에서 total bitwise 동일, U/A gradient 상대 1e-6 | OK |
| 상수 배율 | K41: λE·L_E 와 U/A gradient 가 0.25/0.5/2 배(1e-6) · L_H/L_K 불변 · w_H ≥ 1 | OK |
| 경계 · A 계속 학습 | K41: w(24999)=1, w(25000)=.5(UP 반대) · 24999 는 JQ 와 total·ε 열 bitwise 동일 · 25001 은 total = JQ − .5·λE·L_E, hard/soft 동일, `aligner_active` True, A gradient 있음, offset 연습 그대로 | OK |
| Resume | w(t) 는 step 만의 순수 함수(두 stub 동일) + `kdv.exact_resume`(K29 의 batch 열 복원) + 경계 기록 파일 | OK(구조) |
| Cue floor · Shuffle | K42: g=1→.75, g=0→.25, reverse 반대, w 에 grad 없음, 혼합 (1,0) → (.75E₀+.25E₁)/2, mean w .5 · 실제 자산에서 floor/shuffle 의 active 수(=weight 분포) 동일·표 다름·reverse = 1−floor·평균 w ≈ .5 | OK |
| Reduction | K42: active 수/weight 합 나눗셈 없음(loss = mean w_i E_i), g 전부 1 → EB_R3E075 와 total 동일 · 전부 0 → EB_R3E025 · mean w = .5 라도 CONST050 의 loss 와 같지 않다 | OK |
| 경로 | K41/K42: rec/edge → A·U, offset → A(기존 K12/K18) · Teacher/GT detach | OK |
| Identity | 기존 K01/K22/K25/K30(T0·init·data·cue·evaluator·bridge) | OK |
| Queue | K39/K43: 명시 순서·run 이름·버전(s2 v1/s5 v2)·예약 합(s2 31.5900 h = 31h35m24s · s5 26.5663 h ≈ 26h33m59s; QEDGE9 6 = 11.12 h)·큐 파일 == 순서·config == 생성기·s5 QEDGE9 v1 config 회귀 없음 | OK |

기존 검사 갱신: K19(s2 옛 순서 → PREVIOUS; 9091 은 이제 s5 허용 seed) · K26(s5 앞 6 만 QEDGE9; s5 W104 기준값을 §7.1 실측 1.38/1.57/1.65 로 교체 → 6 run 11.12 h).

## 4. 시간 (계획 §7)

- s2: R3/edge 형 2.30 h · plain N0 1.95 h(§7.1 seed 777 실측 대용) → 12 run 예약 31.5900 h(A 16.18 · B 7.7051 · C 7.7051). s5: dense/schedule/GT+edge 1.57 · cached floor 1.65 · N0 1.38 → 14 run 26.5663 h(A 15.4136 · B 3.7874 · C 7.3658). 계획 §7.2 와 초 단위까지 같다. 실측이 아니며 같은 서버 첫 실측(case@arch) 이 나오면 `measured_same_case` 로 자동 교체.
- 옛 s5 기준값(W112 대용 1.35/1.36/1.50) 을 §7.1 의 W104 실측으로 바꿨으므로 s5 QEDGE9 잔여(QE50 S777) 의 예약도 1.9817 h 로 바뀐다(실측이 있으면 그것).

## 5. 검증 기록 (s1, 2026-09-16)

- unit gate K01–K43 **152 검사 ALL OK**. 기존 config(s5/s1 QEDGE9, s3/s4 QEGX, s4 E0) 는 생성기 출력과 그대로 같다.
- smoke `tools/smoke_cases.py`(shared GPU): EB_QFLOOR/QFSHUF/QFREV/EDOWN S2026 v2 · EB_R3E050/N0E100 S777 v1 · EB_N0 S9091 v1 — 7 벌 통과(1.9036 M, peak ≈3.05 GB; N0 계열 2.55 GB). (EUP S777 은 큐에 없어 smoke 대상 아님.)
- 40-update 실학습: §5.1.

### 5.1 EB_QFLOOR · EB_EDOWN 40-update 실학습 (임시 config: num_iter 40, diag_every 10, 별도 work_dir/ledger; EDOWN 은 switch 를 20 으로 바꿔 경계를 본다; 임시 산출물은 지웠다)

- EB_QFLOOR S2026 v2: 적재 `[kdv] edge_weight floor: w = 0.25 + (0.75 − 0.25)·g (θq 0.3276)`; `kdv_config_resolved.json`·`calibration_resolved.json` 의 `edge_cue_weight{asset_id 526dbd71…, base_gate low_q, low .75, high .25}`. 진단 step 별(batch 48): gate 비율 0.375–0.646 · **mean w 0.44–0.57**(≈ (low+high)/2) · λ_eff = λE0×mean w 0.040–0.052 · raw E 평균 0.027–0.034 → effective(w 가중) 0.014–0.020 · **q-low patch 의 E(0.031–0.044) 가 q-high(0.021–0.027) 보다 일관되게 크다**(QEDGE9 40-update 관측과 같은 방향 — c_E > 0.5 였던 이유). 고정 진단 per-term(step 39) `L0_A 4.6e-3 · LD_A 2.8e-3 · LK_A 1.3e-4 · LEw_A 1.9e-4 · LOw_A 2.2e-4`. 예산 RUN(required 경고; 임시 config 는 projection 을 빼 projected NaN). 최종 step 40 평가·후보·manifests 정상.
- EB_EDOWN S2026 v2 (switch 20): `edge_schedule_events.jsonl` = [{step 0, factor 1.0, λ_eff 0.0908, aligner_active True}, {step 20, factor 0.5, λ_eff 0.0454, aligner_active True, A LR 2e-6(warmup 중)}] — 경계가 **정확히 update 20(0-based)** 이고 A 는 계속 활성. 진단: step ≤ 11 은 sched 1.0·effective E = raw E, step ≥ 20 은 sched 0.5·effective E = raw E/2(0.032→0.016). 고정 진단 `LEw_A 1.0e-4`(QFLOOR 의 1.9e-4 의 절반 수준; 같은 batch 가 아니라 정확 비교는 아님).
- 두 run 의 40-update HQNR(0.9417 / 0.9467) 은 파이프라인 확인값이지 결과가 아니다.

## 6. s2·s5 절차 (사용자 조작)

1. s1 에서 push → s2/s5 `git pull`. **학습 worktree 는 run 중 pull 로 바꾸지 않는다**(계획 §10.5): 현재 run 은 그 코드로 끝난다.
2. `./tools/edgebal_switch.sh --dry-run` → gate/verify/config/재사용 대조 검증(§7.3)/예약 표. s2 는 자기 QEDGE9 v2 3 벌(J0/JQ/QE50 S777 v2) 의 `verified_complete` 를 보고한다 — 불통과면 refresh 여부는 사람이(§7.3: J0/JQ 새 버전 5h00m30s).
3. `./tools/edgebal_switch.sh` → chain 이 돌고 있으면 그대로 두고(현재 run 은 원 설정으로 마무리; **Sheet 만 보고 kill 하지 않는다**) 다음 gate pass 부터 EDGEBAL 순서; DONE 이면 큐로 기동. 대기자가 DONE-with-pending 을 다시 연다.
4. s2 에는 DCR12 인계(`dcr12_prepare.sh`) 도 걸려 있다 — 둘 다 s2 GPU 를 쓰므로 순서는 사용자 결정(DCR12 의 JK0 S777 config 는 PAKD50 마감 09-16 11:22 를 상속하므로 지금 기동하면 `required: true` 처리가 필요하다; DCR12 노트 §5).

## 7. 남긴 것·한계

- 계획 §9 의 조건부 확장(S2-R*/S2-GT-EDGE/S5-UP/S5-FLOOR/S5-COMBINE) 은 `extra_priority.txt` 에 run 이름으로 넣는 방식만 준비(EB case 는 branch_for 가 EDGEBAL 로 잡는다); S5-COMBINE(schedule × floor) 은 registry 가 아직 거부한다 — 계획대로 각각 반복 양성 뒤 별도 구현.
- `gt_hard_always` 는 loss 정의상 항상 1(K41 w_H ≥ 1); Notes 에는 상수로 적는다.
- s2 의 재사용 대조 v2 3 벌은 s1 에서 검증할 수 없다(s2 work_dir). switch 가 s2 에서 보고한다.
- 계획 §8.1 의 `plateau_30Kplus`(21 점 평균) 집계는 결과 도구가 아직 없다 — 결과 보고 때 `checkpoint_metrics.csv` 에서 계산한다(후보 격자 1010 간격은 30300…50000 을 포함).
