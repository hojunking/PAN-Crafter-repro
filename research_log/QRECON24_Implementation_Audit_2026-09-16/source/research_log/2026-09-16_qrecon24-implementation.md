# QRECON24 (s1–s5, W104·D121 q/e 기반 단순 KD 튜닝) 구현·검증 노트 — 2026-09-16

계획: `research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md` (§2 method · §4 profile · §5 큐 · §6 구현 계약 · §7 selector · §8 시간 · §9 인계). 동반 handoff/JSON/CSV 는 이 저장소에 없다.
전제 구현: PAKD50 · QEDGE9(cue 자산) · QEGX · EDGEBAL. 작업 서버: s1(개발·검사·smoke·실학습). **학습 기동·push·Sheet 변경은 하지 않았다**(계획 §0·§부록 B: 기동은 검증된 release 로 사용자가).

## 0. 결론 요약

- 계획 §2 의 고정 method 를 **새 trainer 키 `kdv.qrecon`** 하나로 등록했다(기존 edge_gate/edge_weight/edge_route/routing 을 조합해 검사만 끄는 방식 아님 — registry 가 결합을 거부). 연속 q 가중 `w_i = 2·qref/(qref + q_T(i))`(raw q, qref 0.3276133416220546) 를 A(hard-only) 와 U(GT edge) 가 공유하고, **L_U = mean(H+K+λE·w^E·E) 는 U 에만, L_A = mean(w^A·H) 는 A 에만** 같은 forward 에서 `autograd.grad` 로 따로 넣는다(단일 total backward 금지 §2.5·§6.1). Student offset·jitter 경로는 없다(I-NATIVE-TRANSFER, radius 0).
- profile 29(G 3×3 · A/E 대조 6 · H 3×3+2 · L 3) 와 서버별 사전 고정 큐 68 run(s1 12 · s2 12 · s3 18 · s4 14 · s5 12) 의 config 를 만들었다. 이름 `PAKD50_QRC24_<SRV>_<PROFILE>_W104_D121_WV3_T0_S<seed>_FRESH50_v1`. **다섯 서버의 편성 순서(PRIORITY_BY_SERVER) 를 전부 QRECON24 로 교체**했고 직전 QEDGE9/QEGX/EDGEBAL 순서는 `PREVIOUS_PRIORITY_BY_SERVER` 에 보존(superseded; 미완 항목은 필요 시 extra_priority 에 run 이름으로).
- 계획 §6.3 의 기동 전 검사를 unit gate K44–K48 로 닫았다(총 **171 검사 ALL OK**): reference gradient 와 trainer gradient 일치(상대 1e-6), A-only loss 의 U 누적 없음, U loss 의 A 혼입 없음, β/λE 변경에 A gradient 불변·α 변경에 변화, w(qref)=1·q=0→2·fail-fast, shuffle multiset 보존, λE 선형, frozen 대조 parameter·buffer 불변, G22≡H22≡L100, ALL_UNIF 의 L_U == 표준 total. smoke 8 profile · 실학습 G22 200 update / A_FREEZE 40 / E_SHUF 40 통과(§5).
- target selector `HQNR9585_RR_v1`(§7.2) 은 **후처리 도구 `tools/qrecon24_select.py`** 로 두었다(trainer 의 best_hqnr/raw-max/last 는 그대로 보존). `--official` 은 적격 후보 checkpoint 로 reduced 테스트셋을 다시 추론해 공식 RR 6 지표(업로더 `_rr` 와 같은 경로) 로 정렬한다 — s1 검증에서 legacy best 후보의 값이 업로더 값과 소수점 전 자리 일치. 업로더에는 selection JSON 이 있을 때만 `target_feasible/target_selected_step/target_official` 을 Notes 에 적고, 없으면 `target_selector=pending` (지원완료로 가장하지 않음 §9.3).

## 1. 약명 → 세팅 (계획 §4; 보고서에는 이 표로만 읽는다)

공통: W104·D121 Student U fresh(seed 별 저장 초기값) + A ← T0 aligner 복사, Teacher T0 frozen, rec R3(τR 0.012463942170143127), GT edge λE **절대값**, native 입력(offset·jitter 없음), AdamW WD .01 batch 48 warmup100 + 50K cosine, 두 목적함수 분리.

| profile | λE | rA (A LR = rA × U LR) | α / β | U LR | A weight / edge weight | 비고 |
|---|---:|---:|---|---:|---|---|
| G<ij> | {3e-4, 1e-3, 3e-3}[i] | {.003, .01, .03}[j] → A LR {3e-7, 1e-6, 3e-6} | 1 / .1 | 1e-4 | q / q | **G22 = 공통 기준** |
| A_UNIF · A_SHUF · A_FREEZE | 1e-3 | .01 · .01 · (동결, A-FR) | 1 / .1 | 1e-4 | 1 / q · shuffle / q · — / q | A 의 q 배분·연결·추가 적응 대조 |
| E_UNIF · E_SHUF · ALL_UNIF | 1e-3 | .01 | 1 / .1 | 1e-4 | q / 1 · q / shuffle · 1 / 1 | edge 의 q 대조 · q 없는 기준 |
| H<ij> | 1e-3 | .01 | {.5, 1, 1.5}[i] / {.05, .1, .2}[j] | 1e-4 | q / q | H22 = G22 |
| H_ALPHA0 · H_BETA0 | 1e-3 | .01 | 0 / .1 · 1 / 0 | 1e-4 | q / q | 성분 대조 |
| L100 · L070 · L050 | 1e-3 | .01 → A LR 1e-6 · 7e-7 · 5e-7 | 1 / .1 | 1e-4 · 7e-5 · 5e-5 | q / q | L100 = G22 |

shuffle = 연속 w 를 (Teacher e_roi32 decile × rot) stratum 안에서 고정 permutation 51515 로 교환(값 multiset 보존; 기존 binary gate_shuffle 아님). config 의 `kdv.qrc24` 에 profile·canonical·λE·rA·U/A LR·α/β·weight mode 가 그대로 적힌다.

## 2. 구현 (계획 §6.4 최소 변경 지점)

- `kdv/qrecon.py`(신규): `q_weight`(fail-fast: q<0/NaN/qref≤0), `shuffle_values`(stratum 내 연속값 permutation), `QWeight.load`(EdgeGate.load 와 같은 자산·Teacher A hash/margin·train sha·feeder 계약·재개 대조 + raw q → w 표 [N,4] 두 개(q/shuffle) + 통계·sha), `weight_for(meta, mode)`.
- `kdv/registry.py`: `qrecon{mode continuous_v1, q_ref, asset, a_weight, e_weight, perm_seed}` — rec R3 · stat EDGE-H 의 `outer_weight` 숫자 > 0(절대 λE) · I-NATIVE-TRANSFER radius 0 offset 0 · A-FT/A-FR · Teacher 필요 · edge_gate/edge_route/routing/edge_schedule/edge_weight/aligner_schedule/TRI/extra/control/geom 결합 거부 · shuffle seed 51515. 토큰 `EDGEHQRC[AU|AS][EU|ES]`, describe.
- `train_kdv.py`: `_step` 끝에서 per-sample H_i/K_i(같은 α·β·d_T·a_T 지도)·E_i 와 w^A/w^E → `L_U`, `L_A`(total = L_U 는 기록·NaN 검사용) · 학습 loop 는 `accelerator.backward(total)` 대신 **`_qrecon_backward`**(∇θL_U → U, ∇φL_A → A; retain_graph 한 번; scaler 거부; optimizer.step 은 한 번) · L_A 비유한이면 exit 3 · 진단 `qrc_*`(w 평균·H/K/E·λ·L_U/L_A) + gradient 규모(`grad_LU_U`, `grad_LA_A`, 제외되는 `grad_LA_U_excluded`/`grad_LU_A_excluded`) · 고정 진단 per-term `LA/LU` · manifests `qrecon` 요약(w/perm sha).
- `tools/gen_pakd50_configs.py`: `QRC24_PROFILES/QUEUES/REFERENCE_H/EXTRA`, `qrc24_run_name/parse/profile/items/extension_items`, POLICY `QRC`(A-FT native) / `QRCF`(A-FR), BACKEND `QRC24`, CASES `QRC24_<SRV>_<PROFILE>`(서버 토큰 ≠ 생성 서버면 SystemExit), kdv block(λE 절대값·aligner_lr = rA×U LR·`qrecon`·`qrc24`·campaign/branch/ledger/time_policy{no_hard_limit, min_operating_hours 24}·exact_resume·control_runs G22), render(`learning_rate` 를 `7.0e-05` 처럼 소수점 표기로 — PyYAML 은 `7e-05` 를 문자열로 읽는다), 예약 slack 1.20(§8.2), `plan_reference_qrc24`.
- gate/sheet/scripts: `campaign_gate` exempt ∋ QRECON24 · X열 `PAKD50 / QRC24 / <PROFILE> / A104D121 / FRESH50`(서버 토큰 제외) · Notes `method=qrecon_continuous_v1; A_loss=weighted_H_only; A_soft=0; A_edge=0; student_offset=0; profile; lambda_E_abs; rA; U_lr; A_lr; alpha; beta; qref; A/E_weight; w_sha; perm_sha; init_seed; T0_sha; cue_asset_id; release_sha; control_run_id; target_*` · `tools/qrecon24_switch.sh`(`--dry-run` / `--extend`) · `tools/qrecon24_waiter.sh` · 큐 `config/queues/qrecon24_s{1..5}.txt` · `tools/qrecon24_select.py`.

## 3. 검사 (계획 §6.3 ↔ K44–K48)

| §6.3 항목 | 검사 | 결과 |
|---|---|---|
| reference gradient 대조(≤1e-5 상대) | K46: 실제 `_step`+`_qrecon_backward`(CPU, T0 복사 A + 잡음 U, 비퇴화) — U .grad == ∇θL_U, A .grad == ∇φL_A (상대 1e-6) | OK |
| A-only loss 의 U 누적 없음 / U loss 의 A 혼입 없음 | K46: ∇θL_A ≠ 0 인데 U .grad 에 없음 · ∇φL_U ≠ ∇φL_A 인데 A .grad == ∇φL_A | OK |
| q/GT/Teacher detach · H live | K46: w·y_t 에 grad 없음, U/A 에 gradient 있음 | OK |
| β/λE 변경에 A gradient 불변, α 변경에 변화 | K46: H21(β .05)·G32(λ 3e-3) 의 ∇φL_A 가 G22 와 bitwise 같고 H12(α .5) 는 다르다 | OK |
| G22 ≡ H22 ≡ L100 | K44: core(rec/stat/aligner_lr/qrecon/protocol/policy/corruption/aux) 동일 | OK |
| w(qref)=1, q=0→2, 유한, shuffle multiset, negative/NaN fail-fast, qref>0 | K47 (+ 실제 자산: w 0.7944–1.0949, 평균 0.9963(≠1; 기록), 40 stratum, multiset 보존) | OK |
| λE 3e-4/1e-3/3e-3 선형 · A edge-gradient 부재 · 중앙값을 gate 로 쓰지 않음 | K46: λ·w·E 가 .3/1/3 배 · ∇φL_A 가 λE 와 무관 · 표는 연속값(고유값 > 100) | OK |
| Fresh A == T0 A hash · U 초기값 hash · A LR group 비율 | 기존 K01/K22 + K44(aligner_lr = rA×U LR; scheduler 는 두 group 을 같은 배율로 — 실학습 §5 에서 A LR 열 확인) | OK |
| 200–500 update smoke(G22/A_UNIF/A_FREEZE/E_UNIF·α/β 경계) | smoke 8 profile + 실학습 G22 200·A_FREEZE 40·E_SHUF 40 (§5) — 정식 결과 아님 | OK |
| interruption 뒤 정확 resume | `kdv.exact_resume`(K29) + QWeight 재개 대조(asset_id·w_sha·qref; K47) | OK(구조) |

## 4. 시간 (계획 §8)

R_s 대용값 s1 2.20 / s2 2.30 / s3 1.35 / s4 1.94 / s5 2.17 h, 예약 = 1.20×R_s + 10/60(다른 branch 의 1.10 과 다름) → s1 33.68 · s2 35.12 · s3 32.16 · s4 34.9253 · s5 33.248 h(계획 §8.2 초 단위 일치). 새 gradient 분리 경로의 실측은 아직 없다 — 첫 run 이 끝나면 같은 case@arch 실측이 자동 교체(`measured_same_case`). 24h 는 최소 운영구간이지 상한이 아니며(`time_policy.min_operating_hours 24`, admission 제외), 기본 큐를 마친 뒤 신규 완료 Train(h) 합 < 24h 이면 `./tools/qrecon24_switch.sh --extend`(§8.3 지정 seed 3 run; 이미 결과가 있으면 reserve seed 17041/26017).

## 5. 검증 기록 (s1, 2026-09-16)

- unit gate `tools/pakd50_unit_tests.py` K01–K48 **171 검사 ALL OK**(직전 순서를 PREVIOUS 상수로 옮기며 K19/K22/K26/K34/K39 갱신; 기존 QEDGE9/QEGX/EDGEBAL config 는 생성기 출력과 그대로).
- smoke `tools/smoke_cases.py`: G22 S1234 · A_FREEZE S1234 · A_SHUF S3407 · E_SHUF S777 · ALL_UNIF S777 · H_BETA0/H_ALPHA0 S1234 · L050 S1103 — 8 벌 통과(1.9036 M; A_FREEZE 2.92 GB, 나머지 3.05 GB).
- 실학습(임시 config·work_dir·ledger; 산출물은 지웠다): **G22 200 update** — `[kdv] qrecon continuous_v1: qref 0.3276 · w 0.7944–1.0949 평균 0.9963 · A weight q / edge weight q · λE 1e-3 · A trainable(hard-only)`; 진단 step 0/50/100/150: w 평균 0.99–1.00 · H 0.096→0.058 · K 0.0016→0.0006 · λ·w·E 3.3e-5→2.0e-5 · L_U/L_A 0.098/0.097 → 0.058/0.057 · ‖∇θL_U‖ 0.55–4.6 · ‖∇φL_A‖ 0(step 0; U zero-init) → 0.009 → 0.026 → 0.044 · 제외되는 ‖∇φL_U‖ 0.009–0.045(같은 규모; 대부분 H 경유) · A param 22 매 update 보정 · 최종 평가·후보 저장 정상. **A_FREEZE 40** — aligner_active False, A param 0, U 만 학습. **E_SHUF 40** — w^A 평균 ≠ w^E 평균(edge 만 셔플). 예산 gate 는 required(경고) 로 RUN.
- selector: FQ S1234(W112) 에 proxy 모드 → 적격 0(H 0.9537 < 0.9585), target_feasible false, h_consistency 0(CSV raw H == fr_mat20 H); `--threshold 0.9537 --official` 로 legacy best 후보(step 33330) 의 공식 RR(SCC .98782 · ERGAS 2.05964 · PSNR 37.92684 · SAM 2.79390 · Q8 .92062 · SSIM .97540) 가 업로더 `_rr(reduced_best_hqnr.mat)` 와 일치.

## 6. 서버 절차 (사용자 조작; 계획 §9.2·§10.1)

1. s1 에서 push → 각 서버 `git pull` (**실행 중 run 의 worktree 는 그대로**; 현재 run 은 그 코드로 끝난다).
2. `./tools/qrecon24_switch.sh --dry-run` → gate K01–K48 · cue verify · 큐 config == 생성기(서버 토큰 검사) · 예약 표.
3. `./tools/qrecon24_switch.sh` → chain 이 살아 있으면 그대로(현재 run 은 마지막 update 까지; gate 가 켜진 s2–s5 는 다음 pass 부터 QRECON24 순서, gate 가 꺼진 s1 은 chain DONE 뒤 대기자가 큐로 재기동), 없으면 큐로 기동. **Sheet 만 보고 kill 하지 않는다.** 대기자 `tools/qrecon24_waiter.sh`.
4. 완료 run 마다 `python tools/qrecon24_select.py <run> --official` → `results/qrecon24_target_selection.json`(target/legacy/raw-max/exact50K/late6; 적격 후보 없으면 `target_feasible=false`). 업로드는 기존 `tools/_upload.sh` 그대로(Notes 에 selector 상태).
5. 기본 큐를 마친 뒤 `./tools/qrecon24_switch.sh --extend`(§8.3).

## 7. 남긴 것·한계

- selector 의 대표 행 export(target 후보의 같은-checkpoint FR/RR 를 시트 행으로) 는 미구현 — 업로더는 legacy best 행에 selector 상태만 적는다(계획 §9.3 허용). 공식 RR 6 지표는 후보별 재추론이라 시간이 든다(적격 후보 수 × reduced 추론).
- s1 은 현재 idle(09-16 저녁; QEDGE9 5 벌 완료) — 기동은 사용자. 다른 서버의 진행 중 QEGX/EDGEBAL run 은 정상 종료 뒤 전환.
- w 의 실제 범위가 0.79–1.09 로 좁아(계획 §7.4 "q 의 변화폭이 작으면 결과 차이도 작을 수 있다") uniform/shuffle 대조 차이가 작을 수 있다 — 그대로 보고한다.
- 계획 §11 의 선택적 fine-tuning 확장(FT +10K) 은 미구현.
