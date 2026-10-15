# QEDGE9 — W104·D121 q-gated GT edge (s5 두 seed + s4) 검토·구현 노트 (2026-09-15)

계획: `research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md` · 부모: PAKD50(`2026-09-14_pakd50-implementation.md`) + s4 골격 이식(`2026-09-15_pakd50-s4-w104d121-implementation.md`) · 구현 commit 은 git log("QEDGE9").
이 문서는 실행 전 인계 노트다 — 결과 수치는 없다(결과는 s5/s4 의 `results_log/` 새 문서로).

## 1. 검토 — 계획과 저장소가 맞지 않던 곳 / 결정

| # | 계획 | 저장소 사실 / 결정 |
|---|---|---|
| 1 | §2.1 q = (1/2K) Σ‖ĉε + ε − ĉ0‖₁, K=16 "EQREC4 axis bank A" | `tools/eqrec4/common.py` bank A = r {.25,.5,1,2} × θ {0,90,180,270}, q = mean over (16 probe × 2 성분) — 같은 정의. 새 모듈 `kdv/edge_gate.py::teacher_q` 로 옮겼다(aligner 만 호출, 입력 adapter = `predict_c(view margin 4, bicubic↑4 M)` = Teacher forward 와 동일). |
| 2 | §7.2 feeder 의 유효 augmentation 상태 = 고정 flip + rot 4 | `feeders/feeder.py` 확인: hflip/vflip 은 조건 없이, rot 만 `random.randint(0,3)`. **feeder 를 고치지 않고** `return_meta` 옵션만 추가 — (index, rot, hflip, vflip) 를 하나 더 돌려주며 RNG 소비·순서는 그대로(K27 이 bitwise 검사). |
| 3 | §7.1 cue cache key(Teacher/A hash · dataset sha · index · aug state · bank · dtype) | 자산 manifest 가 T0 aligner state hash · train/_pan h5 sha256 · feeder 계약 · bank 정의 · 정규화 경로를 기록하고, trainer `EdgeGate.load` 가 Teacher aligner hash·train h5 sha·feeder 계약을 대조한다(다르면 ValueError → 시작 안 함). |
| 4 | §7.3 θq = calibration(seed 1234, 3072 base) × 4 state 의 중앙값 | `kdv.calibration.calibration_batches` 와 같은 randperm(seed 1234) 선택; index sha `02122e2554783774` = s1 `work_dir/_pakd50/calibration_resolved.json` 의 patch_manifest 와 같다. **θq = 0.327613** (calibration 12,288 view). |
| 5 | §3.2 `sum(g·E)/B`, 재정규화 금지, Q12 edge 항 **교체** | trainer 의 stat EDGE-H 분기에서 `output_edge_loss` 를 `(g·E_i).mean()` 으로 바꾼다(λE·ramp·routing 경로는 그대로). K24: g≡1 이면 JQ 와 total·U/A gradient 동일(1e-6), g≡0 이면 edge 항 정확히 0 이고 hard/soft/offset gradient 유지. |
| 6 | §6.1 QEC c_E = Σ g E_pilot / Σ E_pilot (s4 W104 J0 S1234 exact50K) | pilot 은 s4 에만 있다 → `tools/qedge9_cue.py pilot` 을 **s4 에서** 돌려 `work_dir/_qedge9/qec_cE.json` 을 만든다(calibration view 12,288; 분모 퇴화면 실패, 0.5 대체 금지). c_E 가 없으면 gate 가 QEC 를 그 pass 에서 건너뛴다(`cue_ready`). |
| 7 | §6.2 QES: (e decile, aug state) stratum 안 고정 permutation 51515, active 수 보존 | 자산 빌드 때 dataset-wide 로 확정(`gate_shuffle`; e = T0 native `[16:48]²` L1 `e_T_roi32`, decile 경계는 calibration view). 40 stratum 전부 셔플, 단일 라벨/소형 stratum 0, 라벨 변경 비율 0.3673. 학습 loss support 는 full-domain 그대로(§2.2). |
| 8 | §0.7·§9.3 시간 정책: 50h·09-16 마감 미상속, soft target 9h | trainer schema 를 늘리지 않고 `kdv.budget` 으로 표현: 자체 ledger `work_dir/_qedge9_budget/ledger.json`, `total_gpu_hours 9.0`, `required: true`(초과·마감은 **경고만**), `training_deadline` 없음, `time_policy{mode: soft_target_only, target 9, hard_deadline null, inherit_parent_deadline false}` (ledger 항목에 기록). gate 편성(`campaign_gate.gate_pakd50`) 도 QEDGE9 항목은 admission 에서 제외(`exempt`). NaN(exit 3)·smoke·중복 run 보호는 그대로. |
| 9 | §11.1 실행명은 PAKD50 prefix·골격 토큰 유지, 새 캠페인/branch id | `PAKD50_<case>_W104_D121_WV3_T0_S<seed>_FRESH50_v1` 그대로; kdv 블록만 `campaign_id QEDGE9_A104D121_20260915_v1 · parent_campaign_id PAKD50… · experiment_branch_id A104D121_T0FIX_QEDGE9_v1`. **s5 의 J0/JQ@W104 도 QEDGE9 branch** 로 만든다(예산 metadata 만 다르고 학습 정의는 s4 의 E0 J0/JQ 와 같다 — §8.3 "예산 metadata 와 수식 변경을 구분"). s4 의 JQ/XJ/F0@W104 는 기존 E0 allocation 그대로(마감 상속). |
| 10 | §5 s5 seed 2026 + 777 (777 은 s2 seed 이름) | 편성 항목을 전체 run 이름(`PAKD50_J0_W104_D121_WV3_T0_S777_…`) 으로 두고 `allowed_seeds(s5) = {2026, 777}` — 확인 seed(9091) 규칙과 구분. 예약은 case@arch 기준(§9.2 값), 실측도 case@arch 키(seed 무관). W104 초기값은 seed 별 새로(trainer init_dir `_kdv_init_w104_d121`; case 사이 공유), `assets/pakd50/init_hashes.json` 의 `unet@W104_D121` 2026/777 은 첫 run 뒤 채운다. |
| 11 | §8.2 "이상 aligner 에서 q=0" 검사 | PyTorch bicubic(a=−0.75) 은 선형 ramp 를 **정수·반정수 이동에서만** 정확히 재현한다(r=.25 는 오차). K25 는 정수/반정수 이동 잔차 < 1e-4, bank 전체 q_ideal < 0.01 ≪ 무반응 0.46875 ≪ 부호 반대 0.9375 로 부호·단위를 닫는다. |
| 12 | §11.3 θq/c_E 미산출이면 대기 | gate `cue_ready`(자산 json+npz+θq; QEC 는 c_E 파일) 가 없으면 편성하지 않는다. trainer 도 자산 없으면 시작 전에 멈춘다(EXIT_GATE 가 아니라 예외 → run.sh 실패 → 재시도 1회 뒤 ledger; 그래서 gate 차단이 1차 방어다). |

## 2. 구현

- `kdv/edge_gate.py`(신규): bank AXIS16 · `apply_view/to_feeder_tensor`(feeder 와 bitwise 같은 변환·정규화) · `teacher_q` · θq/gate/decile/셔플 · `EdgeGate.load`(자산 검증, (index, rot) 표) / `synthetic` / `gate_for` / `summary`.
- `pa/losses.py`: `output_edge_loss_per_sample` (배치 평균 == `output_edge_loss`, K24).
- `feeders/feeder.py`: `return_meta` (기본 False; 다른 trainer·평가 loader 불변).
- `kdv/registry.py`: `edge_gate{mode low_q|const|shuffle, asset, c_E|c_E_file, perm_seed 51515, theta_source}` 검증 — EDGE-H 위에서만, routing/TRI/stat.extra/rec.control/geomKD 와 결합 거부, Teacher 필요, asset 필수; 토큰 `EDGEHQ50/QC/QS`; `describe` 에 gate 서술(시트 Notes).
- `train_kdv.py`: calibration 끝에 `EdgeGate.load`(Teacher aligner hash·train h5 sha·feeder 계약 대조) → `calibration_resolved.json`/`kdv_config_resolved.json` 에 edge_gate 요약; 학습 loop 가 batch meta 를 `_step(…, meta=)` 로 전달(고정 진단 batch 에도); EDGE-H 분기: low_q/shuffle = `(g·E_i).mean()`, const = `c_E·L_E`; EMA `stat_edge_gate_frac / stat_edge_i_mean / stat_edge_i_active / stat_edge_ungated`(train_log.jsonl); ledger 항목에 `time_policy`.
- `tools/gen_pakd50_configs.py`: case QE50/QEC/QES(backend R3+EDGE-H+edge_gate), 상수 `QEDGE9_*`, `branch_for/allowed_seeds/cue_ready`, `kdv_block(branch="QEDGE9")`(캠페인·branch·예산·edge_gate; W104 전용, gate case 는 자동 branch), `render`(gated 면 `train_feeder_args.return_meta: true`, 머리 주석), `generate`(항목별 seed·branch), `schedule(exempt, blocked)`, `PRIORITY_BY_SERVER` s5 = QEDGE9 6 항목(옛 s5 순서 → `PREVIOUS['s5_20260915']`) · s4 = 기존 5 + QE50/QEC/QES, `REFERENCE_CASE_TRAIN_H` s5/s4(§9.2), `--plan` 문서 매핑.
- `tools/campaign_gate.py::gate_pakd50`: QEDGE9 항목 마감 제외 + cue 미준비 항목 대기 로그.
- `tools/qedge9_cue.py`(신규): `build`(s1; 38,856 view: q·c0·e_roi32·gate·QES → `assets/qedge9/cue_T0_AXIS16_v1.{json,npz}` 0.61 MB, git) · `verify`(어느 서버든 96 base × 4 재계산 대조 → `work_dir/_qedge9/verify_<srv>.json`) · `pilot`(s4, c_E) · `status`.
- `tools/qedge9_switch.sh`(s4/s5): gate → cue verify → (s4) pilot c_E → config == 생성기 → 로컬 mandatory(PAKD50 + `work_dir/_qedge9/mandatory_runs.txt`)·reservations → 예약 표 → requeue(runner 교체, chain 마감 72h, 감시자). `--pilot` 은 s4 에서 c_E 만.
- `gspread/gspread_upload.py`: X열 `PAKD50 / <case> / A104D121 / QEDGE9 / FRESH50`(branch 토큰), Notes `q_source=T0; q_bank=AXIS16; q_cut=median; theta_q=<실측>; edge_gate=…; hard_always=1; cue_sha=…; time_policy=…` (+QEC `cE`, QES `perm_seed`). `gspread/sheet_categories.py` 범주 설명.
- config 9 벌: s5 `J0/JQ/QE50 @W104 × S2026·S777`, s4 `QE50/QEC/QES @W104 S1234` (gated 만 `return_meta: true`).
- 검사 `tools/pakd50_unit_tests.py` K23–K28 (전체 110 ALL OK): case/branch/시간 정책·registry 거부 · E_i·trainer 경로(g=1/0/혼합/const, 저오차 patch wH≥1·a_T=0, Teacher/GT detach) · q 부호·단위·θq·셔플·실제 자산 · s5/s4 편성·schedule(blocked/exempt)·실측 키 · feeder return_meta(bitwise·RNG) · 생성 config(회귀 없음).

## 3. 검증 (s1, 2026-09-15 15:50–16:10)

- cue 자산: bench 534 view/s → 실제 **0.8 min** (835 view/s, RTX 4090, SMEC12 학습과 GPU 공유) = §7.4 의 T_cue. θq 0.327613 · 선택 비율 calib 0.5000 / 전체 0.4996 · 동일값 0 · q p50 0.3276(무반응 상수 0.4688 보다 낮다 = aligner 가 반응한다) · QES 40 stratum 전부 셔플, 라벨 변경 0.3673. `verify`(s1): max|Δq| 0, max|Δe| 1.2e-6, 라벨 차이 0, θq 근처(|q−θq|<1e-3) 21/384 — 다른 GPU 에서는 이 근처 라벨이 바뀔 수 있다(각 서버의 verify 가 기록; θq 는 재보정하지 않는다 §7.4).
- smoke(`tools/smoke_cases.py`): QE50 S2026 step ≈125 ms peak 3.05 GB · QEC/QES S1234 95/99 ms · J0 S777 101 ms — 전부 통과.
- 40-update 실학습(QE50 S2026 임시 config, 별도 work_dir/ledger): edge_gate 적재(θq 0.3276) → `stat 0.0204×λ0.0908` · EMA gate 비율 0.476 · 선택 patch 의 E_i 평균 0.0398 > 전체 0.0326(선택 patch 가 edge 오차가 더 크다 → §6.1 대로 c_E 는 0.5 보다 클 수 있다) · 고정 진단 batch(meta 포함) · 예산 RUN(required, 경고) · manifests 에 edge_gate 요약. 임시 산출물은 지웠다.

## 4. 시간 (§9.2 재현; 예약 = 1.10×ref + 10/60)

| 서버 | 순서 | 기준(h) | 예약(h) | 합 |
|---|---|---|---|---|
| s5 | J0 1.35 → JQ 1.36 → QE50 1.50* (S2026) → 같은 셋 (S777) | | 1.6517 · 1.6627 · 1.8167 ×2 | **10.2620** |
| s4 | (잔여 JQ 1.40 · XJ 1.39 · F0 1.14 @W104: 4.8231) → QE50 1.50* → QEC 1.40 → QES 1.50* | | 1.8167 · 1.7067 · 1.8167 | 5.3401 (+4.8231 = 10.1630) |

`*` cache-ready 임시 편성값(미실측). 시트 확인(2026-09-15 15:00): s4 는 NA0/J0/JQ@W104 완료(0.9580/0.9585/0.9565), XJ/F0@W104 진행·대기; s5 는 J0…RCQ(W112) 전부 완료 → s5 는 pull 즉시 시작. 실제 종료는 현재 run 잔여 + s4 의 pilot c_E 준비(수 분) + 예약 합.

## 5. s5 절차 (pull 뒤)

```bash
git pull
./tools/qedge9_switch.sh --dry-run   # gate(110) · cue verify(이 서버 T0) · 6 config == 생성기 · 예약 표 10.2620 h
./tools/qedge9_switch.sh             # runner 교체 → gate 가 J0→JQ→QE50 @W104 S2026 → J0→JQ→QE50 @W104 S777 (완료 run 은 건너뜀; 마감 admission 없음)
tail -f work_dir/cases_chain.log
```

## 6. s4 절차 (pull 뒤; 현재 XJ/F0@W104 는 그대로 끝까지)

```bash
git pull
./tools/qedge9_switch.sh --dry-run   # + QEC pilot c_E (W104 J0 S1234 exact50K 가 있으면 work_dir/_qedge9/qec_cE.json)
./tools/qedge9_switch.sh             # runner 교체 → 잔여 XJ/F0@W104 뒤 QE50 → QEC(c_E 있으면) → QES
./tools/qedge9_switch.sh --pilot     # c_E 를 나중에 만들 때 (gate 는 다음 pass 에 QEC 를 연다)
```

- 첫 W104 seed 2026/777(s5)·확인용 hash: 첫 run 의 `work_dir/<run>/initialization_hashes.json` `unet_init_sha256_16` 을 `assets/pakd50/init_hashes.json` `unet@W104_D121` 에 적으면 이후 config 가 `expect_init` 으로 fail-fast 한다.
- 결과 판정(§10.1): seed 별 같은 서버 안에서 ΔH_gate = H(QE50) − H(JQ), ΔH_total = H(QE50) − H(J0); s4 QE50−QEC / QE50−QES; JQ−XJ. selected raw HQNR · exact50K · plateau(45,450–50,000) 열 별도. 0.0031 을 유의성 검정으로 쓰지 않는다.

## 7. 남긴 것

- QEC 의 c_E 는 s4 에서만 산출된다(pilot 이 s4 에만 있음) — 값은 s4 의 `work_dir/_qedge9/qec_cE.json` 과 시트 Notes(`cE=`) 로 기록; 자산화(assets) 는 s4 가 커밋할 때.
- §10.3 연장(seed 9091 J0/JQ/QE50, β=0.05 대응쌍, LF+QE50) 은 이번 release 에 없다 — 확인 seed 는 기존 `pakd50_reallocate.sh --confirm` 규칙(9091 은 QEDGE9 목록의 허용 seed 가 아니므로 confirm 예약 1.80h 가예약) 을 그대로 쓸 수 있으나 QEDGE9 case 묶음 표는 만들지 않았다.
- q cache 는 W104 학습과 무관하게 T0 에 묶여 있어 다른 골격/seed 에 그대로 쓸 수 있다; feeder 계약(crop/flip/rot) 이 바뀌면 다시 만든다(§7.2).
