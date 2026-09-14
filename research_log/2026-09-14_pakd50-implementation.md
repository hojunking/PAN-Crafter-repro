# 2026-09-14 — PAKD50 구현 노트 (C0 release): L1E4 aligner 재사용·적응 × Q12/R1 fitting, raw HQNR 0.959–0.960, 50h 병렬

계획: [`PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md`](PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md) · 요약: [`PAN_Integrated_Method_Summary_2026-09-14.md`](PAN_Integrated_Method_Summary_2026-09-14.md). 캠페인 `PAKD50_W112D123_WV3_20260914_v1`.
**기존 kdv trainer 로 전부 표현된다** — 새 loss·새 모듈 없음. 계획의 backend/정책이 저장소의 기존 이름과 어떻게 대응하는지가 이 노트의 핵심이다.

## 1. 계획 → 저장소 대응 (C0)

| 계획 | 저장소 |
|---|---|
| Teacher T0 = PALS24 L1E4 seed 2025 `best_raw` 의 A+U 한 쌍 (§3.1) | `kdv.teacher = {run: <T0 run dir>, tag: best_hqnr, expected_sha256}` → `load_run_model` 이 그 run 의 config 대로 A+U 를 strict load, frozen. **Teacher 는 자기 aligner 로 forward** (`kdv_forward(share_correction=False)`; A-FR 에서만 공유). T0 파일: selected update 24240, checkpoint sha `16b5cf78…`, aligner tensors `493aca18…`, 기록 raw HQNR 0.956976. s2/s3 용 사본 `assets/pakd50/T0_run/`(같은 layout, `teacher_assets.json` 에 sha) |
| Student A = T0 aligner 복사, U = seed 별 저장 초기값 (§3.2) | `kdv.donor = {source: <T0>/best_hqnr, view_margin_hr 4, expected_sha256, expected_step 24240}` + `aligner_policy A-FT`(J) / `A-FR`(F); U 는 trainer 의 `_pair_init` 가 `work_dir/_kdv_init_w112_d123/init_unet_seed<seed>.pt` 를 만들고 hash 를 `init_and_teacher_hashes.json` 에 남긴다 (seed 는 서버당 하나: s1 1234 · s2 777 · s3 2026) |
| 정책 J / F / AL (§5, §7.1, §7.5) | J: `A-FT` + `I-AEQ` + `aux.offset_weight 1e-4`(홀수 update, b=2, ramp 없음) · F: `A-FR` + `I-NATIVE-TRANSFER` + offset 0 (frozen A 는 optimizer 밖, L03 gate 가 hash 불변 검사) · AL: J 에 `aligner_lr 3e-6`. **P/JK0/JE0(routing)·D/LF(freeze schedule)·TCOPY/CONT 는 C1/C2** — trainer 가 아직 parameter group 별 loss routing 을 지원하지 않는다 |
| backend N0 / R1 / Q12 / X02 (§4.1) | `rec.case N0`(gt) / `R1`(hard_only: (1+αd)L1) / `R3`(adaptive: (1+αd)L1 + β(1−d)a\|S−T\|) + `stat EDGE/H`(signed Scharr /32, reflect1, 내부 1px, 0.5·x+0.5·y) / `R1` + `stat EDGE/H`. α 1, β 0.1, ε 1e-6. gate K03 이 수식 그대로임(재정규화 없음)·β=0 ⇒ R1·gt ⇒ plain L1·gate detach 를 확인. Teacher 가 loss 에 없는 N0 는 `teacher.eval_only`(bin 진단만) |
| τR (§5.1) | `tools/pakd50_calibrate.py --tau` = `kdv.calibration.calibrate_rec(T0)` (train 고정 3072 patch, seed 1234, batch 48, 증강 off; median e_T, floor 1e-6) → `work_dir/_pakd50/calibration_resolved.json` → config `rec.tau` 숫자 고정. **s1 값 0.012464**(기존 no-align 0.012367 을 복사하지 않았다). s2/s3 는 prepare 가 재계산해 1e-6 안에서 대조한 뒤 s1 값을 쓴다 |
| λE (§5.2) | `--lambda-e` = `calibrate_lambda(J0 S1234 exact50K/last, 'edge', r_grad 0.05)` = 0.05 × median RMS(∂L0/∂Z)/RMS(∂LE/∂Z) → `stat.outer_weight` 숫자 고정. J0-1234 완료 전에는 stage 2 config 를 만들지 않는다(생성기 fail-fast) |
| candidate grid GRID1K (§6.2) | **실행 대체 `GRID1010_50K_v1`**: eval_epoch 5 = 1010 update 마다(49) + exact 50000 = 50 후보. trainer 는 epoch 단위로만 평가하므로 1000 배수 대신 1010 배수를 쓴다(계획 §6.2 "실행상 이유가 있으면 정식 run 전에 protocol ID 를 바꾼다"). plateau45_50 = {45450, 46460, 47470, 48480, 49490, 50000}. `select.retain_all_candidates` 로 50 후보 전부 보존(run 당 ≈1.6 GB) |
| selector·저장 (§6.2) | best_raw = `best_hqnr`(running-max HQNR tie 1e-4 → fSCC) · last50k = `last` · best_rr_val 보조 · run max 는 보고 도구가 별도 열로 |
| 예산 (§9.1, §11) | 서버당 GPU slot 1 · `kdv.budget`: ledger `work_dir/_pakd50_budget/ledger.json` 46h(학습) + reserve 4h(감사), margin 1.1, run 예약 4h, required False. 50h 시계 `work_dir/_pakd50/ledger.json`(start = prepare 시각). stage 2 admission: 경과 + 1.1×예상×잔여 run ≤ 46h (gate) |
| 큐 (§9.4·§5.3) | stage 1 `config/queues/pakd50_<srv>_stage1.txt` = J0 → F0 → JR → FR (τR 만 필요). stage 2 (JQ → FQ → XJ) 는 s1 의 J0-1234 완료 뒤 `campaign_gate` 의 `pakd50` gate 가 λE 를 고정하고 config 를 만들어 연다; s2/s3 는 그 commit 을 pull 한 뒤 `./tools/pakd50_prepare.sh --stage 2` 로 잇는다 |
| 진단 (§14.3–14.4) | 기존 `gradient_diagnostics.jsonl`(live) + `gradient_diagnostics_fixed.jsonl`(고정 batch: ρ, cos, off_unet_grad_absent, 합산 규칙) · checkpoint_metrics 의 4 view(raw_original/raw_v64/aligned_self/aligned_fixed=T0 aligner) · fitting bins · `_upload.sh` 의 pals24/palsv18 진단은 PAKD50 에는 아직 연결하지 않았다(C2) |

약명→세팅: `J0` A trainable(L0+LO)/U N0 · `JQ` J+Q12 · `JR` J+R1 · `XJ` J+X02(Q12 의 soft 제거) · `F0/FQ/FR/XF` A frozen + N0/Q12/R1/X02 · `AL0/ALQ` J 인데 A LR 3e-6. run 이름 `PAKD50_<case>_W112_D123_WV3_T0_S<seed>_FRESH50_v1`.

## 2. 실행 전 gate (`tools/pakd50_unit_tests.py` K01–K05, 25 검사)

T0 binding(sha·step·aligner 동일), 정책/backend 매핑, Q12↔X02↔R1↔N0 수식 관계, edge 정의, gate detach, L_off 수신자(A 만), native L_rec → A/U, frozen A 불변, 후보 격자·큐·τR 고정·λE fail-fast·gate 격리. `tools/pakd50_prepare.sh` 가 각 서버에서 T0 파일 sha 와 **raw HQNR 재현**(허용 1e-4) 을 먼저 검사한다.

## 3. 배포 (§13.4)

s1 커밋 → 사용자 push → s2/s3 `git pull` 후 `./tools/pakd50_prepare.sh` (T0 는 `assets/pakd50/T0_run`, τR 고정값은 `assets/pakd50/calibration_resolved.json`). 실행 중인 checkout 을 pull 로 바꾸지 않는다 — 다음 release 는 새 worktree 또는 run 경계에서. s2/s3 에 다른 캠페인(NA104 20H) 체인이 돌고 있으면 prepare 가 거부한다 — 먼저 정리한다.

## 4. C1/C2 로 미룬 것

P/JK0/JE0 (A/U 별 loss routing), D0/DQ(첫 5K A frozen), LF(25K 이후 A frozen), QA05/QB005/QB02/QE025/QE10/QSF scalar variant(계수는 config 로 바로 가능하나 QSF schedule 은 미구현), TCOPY10/CONT10 protocol, T1/S3407/XSRV, PAKD50 전용 집계 도구(paired block summary·plateau·run max). J0/JQ 3 block 결과를 본 뒤 순서대로 붙인다.

## 5. 실행 (s1)

- 2026-09-14 13:22:31 `./tools/pakd50_prepare.sh --hours 46` → T0 재현 raw HQNR 0.956976(Δ 0.0) → 기존 gate 4종 + K01–K05 통과 → τR 재계산 0.012463942(고정값과 Δ 0) → stage 1 config → smoke 4벌 OK → 50h 시계 시작 → gate token `pakd50` → 체인 기동(마감 09-16 11:22). release cac23a6.
- 첫 run `PAKD50_J0_…_S1234_FRESH50_v1`(λE pilot) 예산 gate RUN(8.4 ≤ 46). smoke 처리량: J 40 ms/step(offset 연습 포함) · F 19 · R1 22 · peak 3.3 GB → run ≈ 1.9 h(평가 50회 포함) → stage 1 4벌 ≈ 8 h, J0-1234 완료 ≈ 15:20 → λE 고정 → stage 2.
- PALSV18 V-post(11/23 checkpoint) 는 13:03 에 일시 중단하고 GPU 를 이 캠페인에 양보했다 — 나머지는 캠페인 사이에 `./tools/palsv18_validate.sh post` 로 이어 돌린다(완료분은 sha 검사로 생략).
- s2/s3: 이 commit 을 pull → (NA104 20H 체인이 있으면 먼저 정리) → `./tools/pakd50_prepare.sh` (stage 1: J0 → F0 → JR → FR, seed 777/2026). stage 2 는 s1 이 λE 를 고정한 commit 을 다시 pull 한 뒤 `./tools/pakd50_prepare.sh --stage 2`.
