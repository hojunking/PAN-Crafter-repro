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

## 5. s5 절차 (pull 뒤; 17:20 갱신 — runner 를 죽이지 않는 전환)

```bash
git pull
./tools/qedge9_switch.sh --dry-run   # gate(117) · cue verify(이 서버 T0; 내부 일관성·margin·θq 밖 flip 0) · 6 config == 생성기 · 완료 control 검증 · 예약 표 10.2620 h — 운영 파일은 쓰지 않는다
./tools/qedge9_switch.sh             # 로컬 mandatory/reservations · 옛 extra_priority 보존 분리 · chain 마감 파일 제거(soft) · chain 이 없으면(DONE) campaign_start · 대기자(tools/qedge9_waiter.sh)
tail -f work_dir/cases_chain.log; cat work_dir/_qedge9/status.json
```

s5 의 옛 chain 은 시트상 전부 완료라 DONE 상태다 → switch 가 새 chain 을 열고 gate 가 J0→JQ→QE50 @W104 S2026 → 같은 셋 S777 을 편성한다(마감 admission 없음). 살아 있는 chain 이 있으면 손대지 않는다(현재 run·평가·업로드는 그 runner 가 끝내고, gate 는 매 pass 새 코드를 부르므로 다음 pass 부터 새 순서).

## 6. s1 절차 (17:20 사용자 결정: seed 1234 묶음은 s4 대신 s1)

```bash
./tools/qedge9_prepare_s1.sh --dry-run   # gate · cue verify · 5 config(v2) == 생성기 · 예약 표 8.5113 h
./tools/qedge9_prepare_s1.sh             # s1 로컬 예약/mandatory + waiter 둘: SMEC12 chain DONE 뒤 campaign_start(마감 파일 제거) · J0 v2 exact50K 가 생기면 c_E (QEC 는 큐 마지막)
tail -f work_dir/cases_chain.log · work_dir/_qedge9/launch_s1.log
```

- s1 에는 W104 control 이 없어 J0/JQ@W104 S1234 를 새로 학습한다(같은 정의·다른 호스트; s4 의 E0 config v1 과 이름 충돌을 피해 **v2**, 캠페인 metadata 는 QEDGE9). 순서 J0 → JQ → QE50 → QES → QEC(pilot = s1 의 J0 v2 exact50K). 큐 `config/queues/qedge9_s1.txt`, gate 'pakd50' 는 s1 에서 꺼져 있다.
- 기동 시점: SMEC12 준비 학습 chain(10 run, 12:34 시작, run 당 ≈1.3 h → 09-16 새벽) 이 DONE 된 뒤 자동. 먼저 돌리려면 SMEC12 chain 을 사람이 멈추고 `./tools/campaign_start.sh --queue config/queues/qedge9_s1.txt --hours 24 && rm -f work_dir/cases_deadline.txt`.
- seed 1234 의 대응 비교(QE50−JQ, QE50−J0, QE50−QEC/QES) 는 전부 s1 안에서; s4 의 W104 J0/JQ/NA0(0.9585/0.9565/0.9580) 는 다른 호스트라 절대값을 빼지 않는다(계획 §6.3).

## 7. 남긴 것

- QEC 의 c_E 는 s4 에서만 산출된다(pilot 이 s4 에만 있음) — 값은 s4 의 `work_dir/_qedge9/qec_cE.json` 과 시트 Notes(`cE=`) 로 기록; 자산화(assets) 는 s4 가 커밋할 때.
- §10.3 연장(seed 9091 J0/JQ/QE50, β=0.05 대응쌍, LF+QE50) 은 이번 release 에 없다 — 확인 seed 는 기존 `pakd50_reallocate.sh --confirm` 규칙(9091 은 QEDGE9 목록의 허용 seed 가 아니므로 confirm 예약 1.80h 가예약) 을 그대로 쓸 수 있으나 QEDGE9 case 묶음 표는 만들지 않았다.
- q cache 는 W104 학습과 무관하게 T0 에 묶여 있어 다른 골격/seed 에 그대로 쓸 수 있다; feeder 계약(crop/flip/rot) 이 바뀌면 다시 만든다(§7.2).

## 8. 감사 대응 (research_log/PAN_QEDGE9_Implementation_Audit_2026-09-15.md F01–F08; 2026-09-15 17:00–18:00)

| ID | 조치 | 검증 |
|---|---|---|
| F01 전환 시 runner 종료·dry-run 기록 | `qedge9_switch.sh` 재작성: **runner 를 죽이지 않는다** — 현재 run 의 학습·평가·업로드는 그 runner 가 끝내고, gate 는 매 pass `campaign_gate.py` 를 새로 부르므로 pull 뒤 다음 pass 부터 새 순서. chain 이 없을 때만 `campaign_start`. `--dry-run` 은 임시 경로만 쓴다(mandatory/reservations/extra/마감/c_E 미기록). | 스크립트 구조; s1 `qedge9_prepare_s1.sh --dry-run` 이 운영 파일을 만들지 않음을 확인 |
| F02 QEC pilot identity | `qedge9_cue.py pilot`: run 이름(`RUN_RE`: J0·W104_D121·S1234)·tag last·last_meta step 50000·모델 W104/D[1,2,1]/seed/case/step 을 전부 강제, 기존 c_E 가 있으면 출처만 대조하고 `--force` 없이는 덮어쓰지 않는다. config 의 `edge_gate.const` 에 `pilot_run/pilot_tag/pilot_step` 을 박고(registry 필수) trainer 가 c_E 파일의 pilot identity·checkpoint hash·cue asset_id·calibration view 표기를 대조 | K31 (잘못된 run/tag/step/asset_id/hash 없음/전체 view 전부 거부, 올바른 파일 통과) |
| F03 cue 식별·일관성 | manifest 에 **asset_id**(npz sha·Teacher A hash/파일/margin·데이터 sha·정규화·feeder 계약·bank·θq·calibration id·QES seed 의 sha256[:32]) 를 넣고(`stamp`), `check_asset` 이 bank 정의·θq 유한·`gate == 1[q<θq]`·coverage·calibration id 재계산·QES permutation 재현·정규화 문자열·asset_id 를 검사한다(`read_asset(strict)`, trainer 적재·verify·status 전부). `EdgeGate.load` 가 Teacher view margin 도 대조하고, 재개 시 이 run 의 이전 `kdv_config_resolved.json` 의 asset_id/c_E 와 대조한다. `verify` CLI 합격 조건: 내부 일관성 + |Δq|<1e-4 + |Δe|<1e-3 + θq 밖 라벨 flip 0(θq 근처 flip 은 기록) | K30 (θq/bank/QES seed/gate 반전/shuffle 위조/asset_id 위조 거부 · margin 0 거부 · 재개 asset_id 불일치 거부) · s1 verify PASS |
| F04 exact resume | `kdv/resume.py`: epoch 시작 시점의 전역 torch RNG(RandomSampler permutation·worker base_seed 의 출처) + 시작 step 을 accelerate checkpoint 에(`EpochState`), 재개 시 복원 → 같은 iterator → 소비 batch skip(worker augmentation RNG 도 같은 만큼 진행) → 전역 RNG 는 checkpoint 시점으로. `kdv.exact_resume: true`(QEDGE9 config 전부). `resume_events.jsonl` 에 exact/skipped 기록 | K29(tiny feeder·2 worker·rot 무작위: 연속 열 == 재개 열) · **실학습 e2e**: QE50 40 update 연속 vs 20 + checkpoint-20 재개 → step 20–39 의 (index, rot) 열 20/20 동일 |
| F05 QEC 미준비 시 DONE | `tools/qedge9_waiter.sh`(switch 가 기동): QEDGE9 mandatory 중 미완 run 이 있는데 chain 이 없으면 cue 준비 여부에 따라 재기동(READY_TO_RESTART) 또는 `WAITING_FOR_CUE`(`work_dir/_qedge9/status.json`); 전부 끝나야 DONE. s1 은 QEC 를 큐 마지막에 두고 pilot 대기자가 J0 v2 뒤 c_E 를 만든다 | 스크립트; status 판정은 `terminal`/`cue_ready` 재사용 |
| F06 완료 marker 만으로 재사용 | `gen_pakd50_configs.verified_complete(run)`: results .mat + exact-50K last + 후보 격자(step 50000 포함·≥45 행) + kdv manifest + Teacher 파일 sha == T0 + train h5 sha == cue + U init hash == seed 공유 init 파일. gate 는 QEDGE9 서버의 완료 run 이 불통과면 `COMPLETE_UNVERIFIED` 로 기록(재실행은 사람 결정), switch 가 표와 `control_verification.json` 을 남긴다 | K33 (marker 만 있는 가짜 run 불통과 사유 명시 · s1 PAKD50 FQ S1234 실제 완료 run 통과) |
| F07 chain 마감·실측 조회원 | switch/waiter/prepare_s1 이 `campaign_start` 뒤 `cases_deadline.txt` 를 지운다(`_run_cases.sh` 는 파일이 없으면 마감 없음 = `hard_deadline: null`; 사본 보존). `measured_hours_all` 이 PAKD50 + QEDGE9 ledger 의 같은 서버 실측(case@arch) 을 합쳐 gate·plan·switch 가 쓴다 | K32 (QEDGE9 ledger 의 2.75 h 가 예약에 반영; 확인 seed 제외) |
| F08 옛 extra_priority | switch 가 `extra_priority.txt` 를 `extra_priority.pre_qedge9_<ts>.txt` 로 보존 분리하고 빈 파일(주석) 을 둔다 | 스크립트 |

감사가 정상으로 확인한 범위(q 정의·Teacher 입력·hard/soft·edge 교체·gradient·gate·QES·feeder·config·예산)는 바꾸지 않았다. 감사 스크립트(`research_log/QEDGE9_Implementation_Audit_2026-09-15/verify_*.py`) 는 결함 재현을 전제로 짜여 있어 그대로 재실행하지 않았고, 같은 부정 입력을 K29–K33 으로 옮겼다. gate 는 117 검사 ALL OK.

## 9. 서버 변경 (2026-09-15 17:20 사용자 결정: s5 + **s1**)

seed 1234 묶음(QE50/QEC/QES) 을 s4 대신 s1 이 돈다. s1 에는 W104 control 이 없으므로 J0/JQ@W104 S1234 도 s1 이 새로 학습한다 — 이름은 **v2**(`PAKD50_<case>_W104_D121_WV3_T0_S1234_FRESH50_v2`; s4 의 E0 v1 config 와 파일·run 이름 충돌 방지, 학습 정의는 같고 캠페인 metadata·호스트만 다르다), 순서 J0 → JQ → QE50 → QES → QEC(pilot = s1 J0 v2 exact50K; `QEDGE9_PILOT_BY_SERVER`). s4 의 QEDGE9 항목·config 3 벌은 제거했고 s4 는 기존 E0 allocation(NA0/J0/JQ/XJ/F0@W104) 만 계속한다. generator 가 항목의 version 을 쓰도록 고쳤다(`generate`: run 이름 항목의 seed·version). 예약 합 8.5113 h(s4 관측·계획 §9.2 대용값; s1 W104 실측 없음). 기동은 SMEC12 chain 뒤 자동(§6). §5–§6 갱신, 검사 K26/K28 갱신. 시트 X열 `PAKD50 / <case> / A104D121 / QEDGE9 / FRESH50` (WV3-s1 탭; v2 는 실행명으로 구분).
