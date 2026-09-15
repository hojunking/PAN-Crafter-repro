# PAKD50 s4 아키텍처 이식(W104–D121 Student, T0 고정) — 검토·구현·검증 (2026-09-15)

계획: `research_log/PAN_PAKD50_S4_W104D121_Architecture_Allocation_2026-09-15.md` (이하 §). 기반: 재배정 노트 `research_log/2026-09-15_pakd50-derived-allocation-implementation.md`(예약식·reallocate 절차). 작성 s1 11:5x.

## 1. 검토 — 계획과 저장소가 맞지 않던 곳 / 결정

| # | 계획 | 저장소·판단 |
|---|---|---|
| 1 | §3/§9.1 Student U 만 W104·depth[1,2,1], "s3 성공 run 의 model_args 확인" (BASE_W104_D121 config 는 git 에 없음) | s3 run config 는 s1 에 없지만 **PO10 template(`model.pancrafter_paper.PANCrafterPaper`, in_mode paper, LN, attn_locations [], n_attn 3, mode_modulation false) 에서 hidden_size 104·depth [1,2,1] 만 바꾸면 backbone 이 정확히 1.9036 M** — s3 Sheet 의 backbone 보고값과 일치(K22 가 검사). 같은 lineage 라고 본다 |
| 2 | §9.2 `run_name/case_of/queue/config/donor·pilot` 의 `_W112_D123` 고정 가정 제거, `--width` 같은 CLI 를 가정하지 않음 | generator 를 골격 인지형으로: `ARCHS`(W112_D123 기본 · W104_D121), `run_name(..., arch)`, `RUN_RE` 파싱(`parse_item`: 'JQ' / 'JQ@W104_D121' / 전체 run 이름), `item_key`, `to_tag`, `case_of/arch_of/seed_of`. 편성 항목·`--cases` 는 `case@arch`. `_W112` 문자열 분할은 generator·gspread 에서 제거 |
| 3 | §9.1 init hash namespace `(architecture_signature, seed)`, W112 hash 를 W104 에 강제하지 않음, init_dir 분리 | `init_hash_for(seed, arch)`: `assets/pakd50/init_hashes.json` 의 `unet`(기본) / `unet@W104_D121` namespace(현재 null → `expect_init` 없음; s4 첫 run 뒤 기록). trainer 의 init_dir 는 이미 `work_dir/_kdv_init_w{hidden}_d{depth}` 라 `_kdv_init_w104_d121` 로 자동 분리(K22 가 규칙 확인). config 에 `architecture_signature`(골격·model class·in_mode·norm·attn·modulation) 기록 |
| 4 | §9.2 parent campaign_id 유지 + `experiment_branch_id=A104D121_T0FIX_E0_v1` | `kdv.experiment_branch_id`·`architecture_signature` 를 골격이 기본이 아닐 때만 추가(registry 가 통과). campaign_id·deadline·Teacher·calibration binding 그대로 |
| 5 | §9.1 Teacher 는 자기 config(W112) 로 load, Student model_args 를 Teacher loader 에 주지 않음; A 만 strict copy | trainer `load_run_model(t["run"], tag, type(model))` 은 run 의 meta/config.yaml 로 skeleton 을 만들어 W112 Teacher 가 된다(class 만 Student 와 공유). **단, trainer 와 smoke 가 "Teacher 폭 == Student 폭" 을 강제**(`kdv.teacher.bridge` 없으면 ValueError) → 골격 이식 config 는 `teacher.bridge: true` 로 명시(다른 동작 변화 없음; grep 확인). donor A strict copy 는 골격 무관(K22: hash == T0 A) |
| 6 | §4 NA0 = A-ID/NOALIGN, Teacher output 미사용, no-align 모델에 aligned PAN 을 넣지 않음 | 정책 `NA`: `aligner_policy A-ID · recipe NOALIGN · input_protocol I-A · na_protocol NA-STRICT · donor 없음 · aligner_lr 없음 · select.aligned_selector false · expect_arch.noalign true · teacher eval_only`(NA104 Q00 과 같은 계약). `eval.fixed_reference_from_donor` 는 넣지 않는다 |
| 7 | §8 예약: 큰 골격의 s4 관측값 대용(NA0/J0 1.18, JQ 1.40, XJ 1.39, F0 1.14) → 7.752333 h; 확인 4 run 6.342667 h; `gate_hours = reservation_h/margin` 유지; 4 h placeholder 금지 | `REFERENCE_CASE_TRAIN_H["s4"]` 를 `case@arch` 키로. `--plan --server s4` = 1.4647/1.4647/1.7067/1.6957/1.4207, 합 7.7523 h. 확인 `CONFIRM_REFERENCE_CASE_H["s4"] = {J0: 1.18, *: 1.40}` → 6.3427 h. 실측은 `measured_hours_from_ledger` 가 `case@arch` 로 나눠 W104 실측을 W112 에 덮어쓰지 않는다. 예약 파일의 gate_hours 규약 그대로(config 의 4.0 은 fallback) |
| 8 | §6 확인 seed 3407: J0@W104 · WIN@W104 · J0@W112 · WIN@W112 (WIN ∈ JQ/XJ/F0), 기존 s4 확인 예약 대체 | `CONFIRM_BUNDLE_BY_SERVER["s4"]`, `CONFIRM_MAX_RUNS_BY_SERVER["s4"] = 4`; `pakd50_reallocate.sh --confirm JQ` 가 4 config(seed 3407, 두 골격) 를 만들고 extra_priority/mandatory 에 넣는다. 표 밖 후보(RC0 등) 거부 |
| 9 | §10 시트: B열에 `W104_D121`, X열 `PAKD50 / JQ / A104D121 / FRESH50`, W열 Notes | run 이름에 토큰 포함; `integrated_label` 이 골격 토큰을 끼움(기본 골격은 종전 표기); KDV Notes 에 `Student=W104D121; Teacher=T0/W112D123; A_source=T0(NA0 는 none); branch=…; E0=fixed-transfer` 추가. 비용(params) 은 실제 추론 모델(A 포함/NA0 는 backbone 만) — 종전 `_cost_model` 이 처리 |
| 10 | §12.1 기존 s4 run 경계까지 대기, 완료 old-case 재편성 금지 | s4 의 파생 묶음(F0/RC0/RCQ/JR/XJ/J_R3_NOEDGE, Sheet 6 행 완료) 은 `PREVIOUS_PRIORITY_BY_SERVER["s4"]` 로 내리고 s4 명시 순서를 W104 5 case 로. 전환은 `./tools/pakd50_reallocate.sh`(runner 만 교체, 현재 run 은 끝까지). 골격이 다르면 다른 run 이라 W112 완료 run 과 겹치지 않는다 |
| 11 | §9.3 smoke 표 | `tools/smoke_cases.py` 5 벌 통과(backbone 1.9036 M, Teacher W112 bridge, NA0 aligner 없음). K22(8 검사): 골격 params · block 정의 · NA0 · 파싱 · 순서/예약 · 확인 묶음 · 생성 config(model_args 나머지 == W112, init_dir 규칙, W112 회귀 없음) · W104 Student forward(A == T0 A, Teacher 출력이 Student 골격과 무관, NA0 delta 0, L_rec → A/U gradient). **zero_module 함정**: 새 U 의 출력 conv 가 0 이면 A 로 가는 L_rec gradient 도 0 — 검사는 0 파라미터 난수화 뒤(CLAUDE.md 함정) |

§2–§5 의 해석(soft 효과·E0 이식 의미)은 실행에 영향이 없다. §5 대로 τR/λE0 는 재보정하지 않고 그대로 이식한다(config 값 동일; K22 가 rec/stat dict 동일성 검사).

## 2. 구현

- `tools/gen_pakd50_configs.py`: 골격 registry·파싱·이름·예약·확인 묶음(§1 표), 정책 `NA`·case `NA0`, `kdv_block(..., arch)`(expect_arch·bridge·branch id·signature·init namespace·NA 계약), `render(..., arch)`(hidden_size/depth/expect_params_m 치환), `generate` 가 `case@arch` 항목 처리, `--plan` 이 s4 계획 인용.
- `tools/pakd50_reallocate.sh`: 항목 → run 이름을 `to_tag` 로(골격 포함), 확인 묶음도.
- `gspread/gspread_upload.py`: `integrated_label` 골격 토큰, Notes 의 branch 설명. `gspread/sheet_categories.py`: DESC 에 s4 골격 토큰.
- `assets/pakd50/init_hashes.json`: `unet@W104_D121` namespace(1234/3407 null).
- config 5 벌 `PAKD50_{NA0,J0,JQ,XJ,F0}_W104_D121_WV3_T0_S1234_FRESH50_v1.yaml`.
- `tools/pakd50_unit_tests.py`: K22 8 검사 + K09/K19/K21 의 s4 묶음 참조 갱신 — 92 검사 ALL OK.

## 3. s4 절차 (pull 뒤)

```bash
git pull
./tools/pakd50_reallocate.sh --dry-run     # unit gate(92) · 5 config == 생성기 · mandatory_runs.txt/reservations.json(case@arch) · 예약 표 7.7523 h
./tools/pakd50_reallocate.sh               # runner 교체 → gate 가 NA0 → J0 → JQ → XJ → F0 (@W104_D121) 편성. 현재 run(있으면) 은 끝까지
./tools/pakd50_reallocate.sh --confirm JQ  # 외부 분석의 WIN 뒤: seed 3407 × {J0, JQ} × {W104_D121, W112_D123} 4 run
```

- 첫 W104 run 이 끝나면 `work_dir/<run>/initialization_hashes.json` 의 `unet_init_sha256_16` 을 `assets/pakd50/init_hashes.json` 의 `unet@W104_D121.1234` 에 적어 두면 이후 config 가 `expect_init` 으로 고정한다(같은 골격·seed 의 모든 case 는 같은 init 파일을 자동 공유).
- 첫 실측(NA0/JQ@W104) 이 나오면 gate 의 `measured_hours_from_ledger` 가 `case@arch` 키로 예약을 갱신한다(W112 실측과 섞이지 않음).

## 4. 남긴 것

- W104 전용 λE 재보정·같은 골격 Teacher/aligner 준비(§5·§11 후속) 는 이번 release 에 없다.
- `expect_params_m` 은 backbone 만(1.9036); 시트 params 는 A 포함(NA0 제외) 실측.
