# PAKD50 s5 배정(timing/routing) 검토·구현 노트 (2026-09-14)

배정 문서: `research_log/PAN_S5_Timing_Routing_Experiment_Plan_2026-09-14.md`(§번호는 그 문서). 동반 문서로 적힌 `PAN_S5_S3_Result_and_Repository_Snapshot_2026-09-14.md`·`PAN_S5_S3_Observed_Row_2026-09-14.json` 은 저장소에 없어 본 문서만으로 검토했다.
상위 계획 `PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md` §7.2–7.6·§13.3, C0 노트 `2026-09-14_pakd50-implementation.md`(§6), s4 노트 `2026-09-14_pakd50-s4-review-and-implementation.md`.

## 1. 검토 — 현 실험과의 결

**결이 다르지 않다.** T0·W112·D123·FRESH50·GRID1010_50K_v1·raw-original HQNR 판정·공통 τR/λE0 를 그대로 두고, 상위 계획 §7.2–7.6 의 조건부 후보 **D(초기 5K A 동결)·P(A 는 L0+LO 만)·LF(25K 뒤 A 동결)·JK0/JE0(부분 차단)** 를 s5 의 기본 탐색으로 올린 것이다. s4(LR·계수) 와 축이 겹치지 않는다. 다만 배정 문서 자신이 적었듯 **이 후보들이 유효하다는 결과는 아직 없다**.

주의해서 읽을 점:

| 항목 | 판단 |
|---|---|
| 탐색 seed 2026 = s3 | 독립 seed 가 아니라 환경 교차(bridge). s5 의 신규 방법은 s5 내부 J0/JQ 와 비교(§2·§5) |
| D/P/LF 는 trainer 기능 | 상위 계획 C1 로 미뤄 두었던 것 — **이번에 구현**(§2). `policy_id` 만 적어 J 와 같은 실험이 도는 일이 없도록 registry 가 미지원 조합을 거부한다(§10.1) |
| PQ 의 정의 | "aligned PAN 전체 detach" 가 아니라 A 의 .grad 에서 (1−q)·∂L/∂φ 를 빼는 방식(§4.2·§10.3). L0→A 는 남는다. optimizer step 은 한 번 |
| D 경계 | 0-based next update index: 4999 동결, 5000 해제, 첫 offset 5001(§5 T0/T1·§10.2). A 의 AdamW moment/WD 는 동결 구간에 갱신되지 않는다(grad None) |
| LF 의 exact25K prefix 재사용 | GRID1010 에 exact 25K state 가 없어 **처음부터 50K**(§6.1). prefix 재사용은 미구현 |
| s3 J0 행(raw 0.9553) | 배정 문서의 해석대로 J0/N0 대조군 값이며 KD 판정이 아니다. 여기서 재확인하지 않았다 |
| 공통 시계 | s5 도 `assets/pakd50/campaign_clock.json`(학습 마감 09-16 11:22:31) 을 쓴다(§9.1). 서버 timezone 은 전부 KST |
| 배정 문서의 snapshot 2303e56 | 그 뒤 감사 반영(54915c5)·s4(2303e56)·s4 보고(7257106·76336ac) 까지 반영된 큐 규약(큐 J0 만 + gate 편성)을 따른다(§9.2) |

## 2. 구현 (s1; 기본값이면 기존 J 경로와 완전히 같다 — S5-G02)

| 구성 | 내용 |
|---|---|
| registry (`kdv/registry.py`) | 새 키 `kdv.aligner_schedule {freeze_until, freeze_from}`(0-based next update index) · `kdv.routing {qD, qK, qE}` ∈ [0,1]. 거부: A-FR/A-ID 위의 일정·routing, freeze_until ≥ freeze_from, 알 수 없는 키, N0 위 qD, soft 없는 rec 위 qK, EDGE-H 없는 qE, TRI/geomKD/stat.extra/rec.control 과의 결합. spec `aligner_freeze_until/from`, `route_A` |
| trainer 일정 (`train_kdv.py`) | `aligner_active(step)` = trainable ∧ step ≥ freeze_until ∧ step < freeze_from (step 만의 순수 함수 → 재개 안전). 매 update 앞에 `A.requires_grad_(active)`; 동결 구간은 forward 를 no_grad 로(Δ 에 graph 없음), backward 뒤 A.grad = None 방어 → AdamW 가 moment·WD 를 건드리지 않는다. 홀수 update 의 ε 는 동결 구간에도 같은 순서로 소비하고 jittered-A forward/backward 만 생략(LO 없음; `off_skipped_frozen`). 경계는 `aligner_schedule_events.jsonl`(step, active, A hash, 그 시점 A LR) 에 기록 |
| trainer routing | `accelerator.backward(total, retain_graph)` 뒤 `_apply_routing`: A 의 .grad 에서 (1−qD)(hard−L0) + (1−qK)·L_K + (1−qE)·λE L_E 의 ∂/∂φ 를 뺀다(AMP scaler·accumulation 배율 동일). U 의 .grad 는 전체 L_Q 그대로, step 은 한 번. `_l0_t`(plain L1)·`_edge_w_t`(λE·L_E) 를 `_step` 의 info 에 추가 |
| 진단 | 고정 batch 진단에 **loss 별 A/U gradient 분해**(L0·L_D·L_K·λE L_E·λ_off L_O 의 norm, native L0 와의 cosine; routing 전 원 gradient) — 배정 §7.1 / 감사 F08. 진단·live 진단은 `aligner_active` 인 step 에서만 A 항을 계산. run manifest 에 schedule·route_A |
| generator | 정책 D/LF/P/DP/JK0/JE0 (계수·LR 은 J 와 같음) → case `D0 DQ DR DX · PQ PR PX · DPQ DPX · LF0 LFQ · JK0 JE0`. routing 은 backend 에 있는 항만 적는다(PX: qD·qE, PR: qD; 아무 항도 안 바뀌면 생성 거부). no-KD control: D*→D0, P*/JK0/JE0→J0, LF*→LF0, DP*→D0. 서버 s5(seed 2026), 기본 묶음 `J0→JQ→D0→DQ→PQ`, stage 1 = J0/D0(τR 만), stage 2 = JQ/DQ/PQ(λE0). 조건부(LF pair·JK0/JE0·DPQ·soft-off) 와 seed 9091 확인은 `work_dir/_pakd50/extra_priority.txt` |
| 검사 (`tools/pakd50_unit_tests.py` K10–K12) | K10 registry 수용/거부 10건(S5-G07) · K11 case 매핑·s5 묶음 · K12 실제 `KDVTrainer._step` CPU: PQ/JK0/JE0 의 A grad = 수식(∇φ(L0+λ_off L_O [+유지 항])) 과 ≤4e-9, U grad 불변(S5-G05/G06); JQ 는 routing 비활성(S5-G02); D 4999/5000/49999·LF 24999/25000 경계(S5-G03/G04); 동결 update 에 offset 생략·Δ graph 없음·AdamW 뒤 A hash 불변·ε RNG 소비 동일(S5-G08); 해제 뒤 재개. kdv/nf16/pals24 gate 회귀 통과 |
| 시트 | `gspread/server.txt`=s5 → `WV3-s5` 탭 자동 생성, C열 캠페인 PAKD50, X열 `통합실험` = `PAKD50 / DQ / FRESH50` 등(s4 와 같은 경로; 새 spreadsheet·Y열 없음, §11) |

미구현(후속): S5-G09(동결 경계 직전 중단·resume 대조 — exact resume 자체가 감사 F06 후속), LF 의 exact25K prefix 재사용, D+P 결합 외 세 축 결합 금지는 문서 규약, §12.1 의 s5 산출물(manifest·bridge/paired csv·winner_lock) 자동 생성, CONT10/TCOPY10.

## 3. s5 절차

1. 저장소 pull(이 커밋 이후) → `echo s5 > gspread/server.txt` → 데이터·DLPan 배치 → `./tools/setup_paths.sh --apply`.
2. `./tools/pakd50_prepare.sh` — T0 sha·raw HQNR 재현 → gate(K01–K12 포함) → τR 대조(자산 고정값) → stage 1 config(J0/D0) → smoke → 공통 시계 → 체인 기동(큐 J0). 그 뒤 gate: λE0 사본이 있으면 JQ → D0 → DQ → PQ, 없으면 D0 한 벌 뒤 다시 확인(§9.2 "J0 뒤 λE0 가 없으면 D0 먼저").
3. 조건부·확인: `python tools/gen_pakd50_configs.py --server s5 --cases LF0,LFQ`(또는 JK0 / DPQ / DX·PX) 뒤 case id 를 `work_dir/_pakd50/extra_priority.txt` 에; seed 9091 은 `--seed 9091 --cases J0,JQ,<WIN>` 로 생성한 run 이름을 같은 파일에.
4. 업로드는 체인이 run 마다 한다(`WV3-s5`).

## 4. s5 보고(8건) 검토·반영 — 17:30

| # | 판단 | 반영 |
|---|---|---|
| 1 `_diagnose_fixed` graph 재사용 | **맞다.** sum-rule grad 가 `retain_graph=False` 로 graph 를 해제한 뒤 loss 별 분해가 다시 autograd.grad → 진단 step(홀수 diag step, offset 연습 포함) 에서 RuntimeError. s1 의 다음 stage-2 run 도 같은 경로였다 | `retain_graph=True`(해제는 `del total, info`). K13 이 stub 으로 `_diagnose_fixed(step 1)` 을 실제 호출해 재현·검사 |
| 2 s3/s5 config 네임스페이스 공유 | **맞다.** 같은 seed → 같은 config 파일. `gen --all` 이 s1→s4, s3→s5 순으로 덮어써 `remaining_mandatory` 가 마지막 서버 것으로 남았다(s1 의 JQ-1234 는 s4 묶음 [J0,AL0,ALQ] 으로 시작 — 예약량만 다르고 수치 경로는 같다) | **config 를 서버 공용으로**: `remaining_mandatory: []` + `remaining_mandatory_file: work_dir/_pakd50/mandatory_runs.txt`(prepare 가 `mandatory_for(server)` 로 쓴다; trainer `_remaining_mandatory()` 가 파일 우선, 자기 제외). 이제 s1/s4·s3/s5 의 J0/JQ config 는 byte 단위로 같다(K08). 헤더 주석도 "seed 를 쓰는 서버 s3/s5 공용" |
| 3 `init_unet_seed2026.pt` 부재 | 부분 동의. 초기값은 seed 로 결정적(main 의 `manual_seed` → 모델 생성 → `_pair_init` 저장) 이라 s5 가 같은 파일을 만든다. 각 run 의 `initialization_hashes.json`(`unet_init_sha256_16`) 으로 **사후 대조는 가능**했다. 다만 fail-fast 는 없었다 | `kdv.expect_init{unet_sha256_16}` (registry) + trainer `_check_init_hash` (`INIT_HASH_MISMATCH`). `assets/pakd50/init_hashes.json` 에 s1 이 가진 seed 1234/2025/7777 을 기록(1234 = c988a6c95b17f4fd, 감사와 일치); **777·2026 은 s2·s3 가 자기 run 의 `initialization_hashes.json` 값을 보내면 채운다** — 그때까지 s5 는 검사 없이 돌고 사후 대조 |
| 4 동결 구간 LR 감쇠 | **계획대로다.** 배정 §5 "A 는 1e-5 × g(global_update) 로 해제, warmup 을 새로 시작하지 않는다", §10.2 "기존 scheduler 규약 보존". 실제 cosine(warmup 100/50K) 의 5000 시점 배율은 **0.9764**(보고의 0.79 는 다른 산식) — `aligner_schedule_events.jsonl` 에 해제 시점 A LR 이 기록된다. "peak 에서 재시작" 은 다른 arm 이며 계획 작성자가 원하면 별도 case 로 |
| 5 동반 문서 2종 | s3/계획 작성자 몫. 노트 §1 에 부재를 적어 두었다 |
| 6 처리량 | J0 완주 뒤 ledger 의 완료 평균이 자동으로 우선한다(gate est) |
| 7 §12.1 산출물 | 미구현 그대로. `winner_lock.json` 은 CONF 전에 **수동 작성**(case·config·code sha·T0/calibration/evaluator hash·근거 run) |
| 8 S5-G09 | K14: 연속 실행(4998–5001) vs 4999 뒤 재개(모델·ε RNG 상태 복원) 에서 aligner_active·offset 유무·ε·Δ·loss 동일. optimizer/scheduler 복원은 accelerate `save_state` 의 몫이며 exact resume(F06) 은 여전히 후속 |

이번 변경 뒤 config 12+7+s4/s5 벌을 다시 생성했다(공용 파일 = 서버 무관). s1 의 다음 run 부터 적용.
