# QRECON24 ADJ-R1 — s1–s5 실행 순서·우선순위 조정 구현 노트 (2026-09-17)

계획: `research_log/PAN_QRECON24_S1_S5_Queue_Adjustment_2026-09-17.md` (§0–§13, 부록 A·B). 원 캠페인 구현 노트: `research_log/2026-09-16_qrecon24-implementation.md` (§9–§9.1 의 분자 2 제거·λE 환산·uniform .5 가 이번에도 그대로다).
이 노트는 **계획을 실행 가능하게 만든 코드·큐·도구·검사**를 기록한다. 결과·판정은 여기에 쓰지 않는다(results_log 새 문서).

## 0. 요약

- 수식·gradient 경로·Teacher·q 정의·λE 환산값(6e-4/2e-3/6e-3)·uniform .5·50K 는 **한 줄도 바꾸지 않았다** (kdv/·train_kdv.py 무변경). 바뀐 것은 편성(생성기 목록·큐 파일)·metadata(v2 config)·운영 도구·검사·문서다.
- 신규 profile 3 개(§4): `B20A03`(λE .002·rA .03·α 1·β .2) · `A03_UNIF` · `A03_SHUF`(G23 에서 A 의 q 연결만 상수 .5 / stratum 셔플). 기존 표의 숫자는 그대로.
- 활성 편성 = **등록 완료 39(원 순서 그대로; runner 가 건너뜀) + §5 남은 순서 42** = 81 run(s1 14 · s2 12 · s3 20 · s4 19 · s5 16). 추가 16 run 은 `_FRESH50_v2` id(부록 B 와 정확히 일치), 유지 26 run 은 v1 id 그대로 — **원계획 68 v1 config 는 바이트 불변**(보류 3 벌 포함; K50 이 매번 확인).
- 보류 3 run(s4 H11@3407 · s5 L070/L050@1103)은 `superseded_pending` 으로 편성·mandatory·활성 큐에서 빠진다. 이미 실행 중/완료면 원 정의로 끝나고 switch 가 그 사실을 보고한다(§9).
- 공식 RR selector(§7.2) 와 버전 감사(§8.1) 는 **case 경계**(run 뒤처리 `tools/_upload.sh` → `tools/qrecon24_postrun.py`) 에서 GPU 로 돈다 — 학습과 겹치지 않는다. s1 G23 S1234 자산은 §7.1 대로 보존했다.
- 24h 규칙은 원 campaign 의 누적(0 부터 다시 세지 않음) — `--extend` 의 판정 코드는 그대로이고 보류 run 이 mandatory 에 남지 않으므로 끝없이 기다리는 일이 없다(§11.1 지적).

## 1. 계획 ↔ 구현 대응

| 계획 | 구현 |
|---|---|
| §4 신규 profile 3 | `tools/gen_pakd50_configs.py` `QRC24_PROFILES` += B20A03/A03_UNIF/A03_SHUF (32 개), `_QRC_TXT` 설명, `CASES` 자동 등록(`QRC24_<SRV>_<PROFILE>`) |
| §5 서버별 남은 순서 | `QRC24_ADJ_ORDER[srv]` = (profile, seed, version) 목록(§5.1–5.5 표 순서 그대로). 원계획은 `QRC24_QUEUES_20260916` 으로 보존, 활성 `QRC24_QUEUES = 등록 완료 + ADJ_ORDER`, `QRC24_REGISTERED` = 부록 A 39 |
| §9 보류 3 run | `QRC24_HELD` (사유 포함) · `QRC24_HELD_STATUS = "superseded_pending"` · `qrc24_held_runs(srv)` — `priority_for/mandatory_for/qrc24_items` 밖. switch 가 완료/실행 중/보류를 판별해 `work_dir/_qrecon24/held_runs.json` 에 기록 |
| §11.2 새 id · gspread | v2 id `qrc24_run_name(..., "v2")`; config `kdv.qrc24{queue_revision: QRC24_ADJ_R1_20260917, adjustment: ADJ-R1, source_plan, block_2x2, numerator_factor: 1}`; X열 `PAKD50 / QRC24 / <PROFILE> / A104D121 / ADJ-R1 / FRESH50`(v2 만; `gspread/gspread_upload.py integrated_label`), Notes 에 `numerator_factor/uniform_weight/queue_revision/source_plan/block_2x2` 추가 |
| §11.2 control_run_id 실제 id | `qrc24_control_run(srv, seed, profile)` 이 **편성된 version** 으로 id 를 만든다(s4 3407 → H22 v1, s5 1103 → L100 v1, s3 4321 B20A03 → G22 v1·G23 v1·H23 v2). v2 config 의 `control_runs` 에 같은 서버·seed 의 G23/H23/B20A03 상대까지, `block_2x2` 에 2×2 네 칸(없으면 null) |
| §11.1-4 같은 revision | `config/queues/qrecon24_s{1..5}.txt` = `write_qrc24_queue_file()`(`--qrc24-queues`); switch ⑤ 가 mandatory(PAKD50+QRC24)·reservations·`queue_revision.json`·`held_runs.json`·**`queue_effective.txt`**(실제 실행 순서 = 시작된 보류 run + 미완 편성; 완료 run 제외) 을 쓰고, `queue_active.txt` 가 있으면(이전 `--extend`) 갱신; ⑥ 은 그 파일로 case 경계 인계 + **`work_dir/cases_queue.txt` 도 같은 순서로**(tmp + mv) — runner 가 재기동 시 읽는 영속 큐라 감시자·재부팅이 옛 편성·보류 run 을 돌리지 않는다. runner(`_run_cases.sh`) 도 인계를 적용할 때 영속 큐를 갱신하고 빈 인계 파일은 무시한다 |
| §7.1 자산 보존 | `tools/qrecon24_preserve.py <run>` → `work_dir/_qrecon24/preserved/<run>/` + `preserve_manifest.json`(sha256, selected step, release, T0/cue). s1 G23 S1234 보존 완료(§7) |
| §7.2 공식 selector | `tools/qrecon24_postrun.py <run> --backlog` — `qrecon24_select.py --official --threshold 0.9585 --device cuda` 를 이 run + 이 서버의 미완 backlog 에; 학습 중이면 건너뜀(다음 경계). `_upload.sh` 가 FR 평가 뒤·시트 업로드 전에 부른다(Notes 가 official 상태를 갖는다). GPU 시간은 QRC24 ledger `select_<run>`(kind diag) 별도 |
| §8.1 버전 감사 | `tools/qrecon24_version_audit.py` — 시작 manifest(kdv_config_resolved: numerator_factor·uniform·λE)·config 스냅샷·git·command(--resume)·ledger 이전 시도 → `single_definition_factor1 / factor2_or_old_definition / unverified`. switch ③-ADJ(dry-run 은 파일 안 씀) 와 postrun 이 부른다 |
| §10 시간 | **남은 모든 항목(v1·v2)** 이 `QRC24_ADJ_REFERENCE_H`(s1 2.42 · A_FREEZE 2.15 / s2 2.56 / s3 1.47 / s4 1.51 / s5 2.58 — 느린 시나리오) 로 예약(kind `plan_reference_qrc24_adj_r1`), ledger 실측이 있으면 그것이 우선. 원 `QRC24_REFERENCE_H` 는 **v1 config 의 `projected_map` 에만** 남는다(68 벌 바이트 불변). 예약식 1.20×ref + 10/60 한 번 |
| §11.3 회귀 검사 | K50 (§3) |
| §8.3 확장·reserve seed | `qrc24_control_profile` 이 확장 묶음의 alias(s4 H22 · s5 L100) 를 돌려준다 — 편성·확장 어디에도 없는 seed 는 SystemExit(가상 G22 id 금지, §11.2) |
| §9 '미시작일 때만' | `qrc24_held_state()` 가 terminal / running_original_definition / **started_interrupted**(checkpoint·epoch 디렉터리 존재 = runner 의 `latest_ckpt` 와 같은 규칙) / superseded_pending 을 구분하고, 시작된 보류 run 은 보류하지 않고 **활성 큐·mandatory 맨 앞**에 두어 원 정의로 끝낸다 |

## 2. 서버별 활성 편성 (등록 완료는 생략; 계획 §5 순서)

| 서버 | 남은 편성 (v2 는 ADJ-R1 추가) | 보류 |
|---|---|---|
| s1 | G22@3407 → G23@3407 → **A03_UNIF@1234 v2** → **A03_SHUF@1234 v2** → A_UNIF → A_FREEZE → A_SHUF → G21 @3407 | — |
| s2 | G13 → E_UNIF → E_SHUF → ALL_UNIF → G31 → G33 @777 | — |
| s3 | G23@4321 → **H23 v2** → **B20A03 v2** → G13 → G12 → G21 → G11 → G32 → G31 → G33 @4321 | — |
| s4 | H_BETA0@1234 → H22@3407 → **H23 v2** → H12 → **H13 v2** → **H32 v2** → **H33 v2** @3407 → **G23@1234 v2** → **B20A03@1234 v2** | H11@3407 |
| s5 | L070 → L050 @9091 → **G23/H23/B20A03 @9091 v2** → L100@1103 → **G23/H23/B20A03 @1103 v2** | L070/L050@1103 |

A LR × β 2×2(§6.1) 의 실제 id 는 각 v2 config 의 `kdv.qrc24.block_2x2` 에 있다(s3 4321 · s4 1234 · s5 9091 · s5 1103; s1 1234 는 rA 축만). s2 는 추가 없음.

## 3. 검사 (`tools/pakd50_unit_tests.py`)

- K44/K48b 갱신: profile 32, 원계획 개수는 `QRC24_QUEUES_20260916` 기준, 활성 81 config, 예약 합(measured 없이 v1 R_s / v2 ADJ R_s).
- **K50 문서 검사는 계획 원문(`research_log/PAN_*.md`) 의 존재를 요구하지 않는다** — 계획 원문은 저장소에 두지 않는 규약이라 s1 에만 있었고, 그대로 두면 s2–s5 에서 gate 가 막힌다(09-17 s3 보고). K48c·PL02 와 같은 방식으로 **있으면 내용 대조**(부록 B 16 id · 신규 profile 3 · 보류 profile · revision · 하한 .9585), **없으면 그 항목만 사유와 함께 건너뛴다**; 구현 노트·CLAUDE.md·generator 상수는 언제나 강한 검사. 두 갈래 모두 s1 에서 188 ALL OK 로 확인(원문을 잠시 치우고 재실행).
- **K50** (12 check): 신규 profile 계수·동치(B20A03 = G23+β / = H23+rA; A03_* = G23+a_weight) · §5 순서·부록 B 16 id·보류 3·등록 39·활성 81·큐 파일 일치 · **원계획 68 v1 config 바이트 불변**(생성기 재생성 filecmp)·v2 16 == 생성기·v2 metadata·control 실제 id·projected_map · kdv block 차이 집합(B20A03−G23 = rec.kd_weight 만, B20A03−H23 = aligner_lr 만, A03_*−G23 = qrecon.a_weight 만; expect_init 같은 seed) · **실제 trainer**(K46 harness): B20A03 == H23 손실 bitwise, B20A03 vs G23 의 L_A·∇φL_A bitwise 동일 + L_U 차 == mean(K)(β 2 배), A03_UNIF L_A = .5·mean H, A03_SHUF w_a = 셔플 표, 세 run 의 ∇θL_U 는 G23 과 같다 · 시트 X열/Notes·스크립트(bash -n, postrun hook, switch 항목) · 도구 fixture(버전 감사 4 종 판정·보존 sha256·postrun dry-run) + 실제 s1 G22 S1234 판정 · 문서.
- gate 결과(s1, 2026-09-17): `tools/pakd50_unit_tests.py` **188 ALL OK** (K01–K50 + heredoc import 검사; K32 는 이 서버의 실제 QRC24 ledger 실측이 fixture 에 섞이던 것을 격리). 로그 `work_dir/_qrecon24/gate_s1_adjr1.log`.

## 4. s1 에서 확인한 것 (2026-09-17)

- §8.1 버전 감사(s1 완료 6 run 전부): `single_definition_factor1` — numerator_factor 1.0 · uniform .5 · λE .002 == 스냅샷 outer_weight · release 337d6c2 · 재개 없음. G22 S1234 는 ledger 에 이전 시도 `…#1`(PREV_CRASHED 1.18 h; 옛 코드 15204de 로 시작했다가 09-16 17:31 사용자 지시로 버리고 새로 시작한 것; 작업 디렉터리는 `work_dir/_discarded/`) 이 있다 — 현재 결과와 무관, 유효한 새 방식 결과. 결론: **s1 seed 1234 의 새 방식 G22 를 새 id 로 다시 돌릴 필요 없음**(§8.1 '필요할 때만').
- §7.1 보존: `work_dir/_qrecon24/preserved/PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S1234_FRESH50_v1/` 28 파일(16 MB; selected step 31310, raw HQNR 0.959148, release 337d6c2), `--verify` 전부 일치. selector 결과·감사 json 은 다음 경계 뒤 다시 `qrecon24_preserve.py` 를 돌리면 덧붙는다(멱등).
- §7.2 backlog: 10:29 G22 S3407 완료 → 같은 경계(`_upload.sh` → `qrecon24_postrun.py --backlog`) 에서 GPU 로 7 run 처리(학습과 겹치지 않음): **G23 S1234 = official, 적격 2, target step 31310**(= 기존 best_raw 와 같은 step; raw H .959148, 공식 RR SCC .98758 / ERGAS 2.0712 / PSNR 37.877 / SAM 2.812 / Q8 .9194 / SSIM .9752 — 논문 표시 정밀도로 SCC 만 동률, 고정밀 strict pass 0/6; sidecar `reduced_candidate_step-31310.json` 에 checkpoint/config/h5/evaluator sha) · 나머지 6 run(1234 G22/A_UNIF/A_FREEZE/G21/A_SHUF, 3407 G22) 은 적격 후보 없음 → `no_eligible`(재추론 없음). ledger `select_<run>` 기록. 버전 감사 7/7 `single_definition_factor1`.
- 인계: 10:35:10 runner 가 case 경계에서 옛 큐 잔여 5 건을 버리고 ADJ-R1 큐 14 건으로 교체(`cases_queue_handover.txt.applied.0917-103510`). 등록 완료 7 run 을 '완료됨 — 업로드만 확인' 으로 다시 걸으며 FR 평가·pa_diag·업로드에 ≈29 분을 썼다 — 이것이 §6 의 '완료 run 을 실행 큐에서 뺀다' 수정의 계기다.
- 검토 수정 뒤 switch 재적용(11:00): `cases_queue.txt`·`queue_effective.txt` 를 미완 7 run 으로 교체 → 11:04:39 두 번째 인계(`…applied.0917-110439`, 남은 7 건 폐기 → 새 큐 7 건) 로 walk 가 끊기고 **G23@3407 학습 시작**(§5.1 1순위 G22@3407 은 10:29 완료). 이후 전환에서는 완료 run 재업로드가 없다.

## 5. 서버 절차 (사용자 조작)


1. `git pull` (실행 중 코드에 무조건 pull 하지 않는다 — 현재 run 은 그 코드로 끝난다; 이번 변경은 trainer 수학을 건드리지 않으므로 다음 case 부터 새 코드여도 정의가 같다).
2. `./tools/qrecon24_switch.sh --dry-run` — gate(K01–K50) · cue verify · 큐 == 생성기 · 보류 상태 · 버전 감사(파일 안 씀) · 예약 표.
3. `./tools/qrecon24_switch.sh` — mandatory/reservations/revision/held 기록 · chain 이 살아 있으면 **case 경계 인계**(현재 run 은 끝까지) · 없으면 큐로 campaign_start · 대기자 · GPU 가 비어 있으면 selector backlog 지금.
4. 확인: `tail -f work_dir/cases_chain.log` (인계 줄 `[cases] 큐 인계(case 경계 …)`), `work_dir/_qrecon24/queue_revision.json`, `held_runs.json`, `version_audit.json`, 각 run 의 `results/qrecon24_target_selection.json`.

## 6. 적대적 검토 대응 (2026-09-17, 구현 직후)

계획·코드·라이브 상태를 5 축(계획 부합·method 불변·운영·시트 metadata·문서/시간)으로 훑고 상위 지적을 두 관점으로 재검증했다. 확인된 5 건 전부 고쳤다.

| 지적 | 대응 |
|---|---|
| case 경계 인계는 메모리에만 적용돼 `work_dir/cases_queue.txt` 가 옛 편성으로 남는다 → 감시자·재부팅 재기동이 superseded 순서(보류 run 포함, v2 없음) 를 돌린다 | switch ⑥ 이 인계와 같은 순서를 `cases_queue.txt` 에 tmp+mv 로 쓰고, runner 의 HANDOVER 블록도 적용 시 영속 큐를 갱신(+ 빈 인계 파일은 `.empty.*` 로 무시). K49 가 두 경로를 실제 loop 로 검사 |
| switch ⑤ 가 백업 경로를 `keep` 에 재바인딩 → `extra_priority.txt` 가 한 글자씩 쪼개지고 블록이 죽는다(⑥⑦ 미실행) | `bak` 로 분리(qegx·edgebal switch 의 같은 버그도). K50 이 임시 ROOT 에서 ⑤ 블록을 **실제로 실행**해 확인 |
| §10 시간 대용이 v2 항목에만 적용돼 유지 v1 26 run 이 옛 R_s 로 예약된다 | `reference_hours` 가 QRC24 항목 전부에 §10 대용(+ s1 A_FREEZE 2.15) 을 쓴다. v1 config 의 `projected_map` 은 원값 유지(바이트 불변) |
| `--extend` 확장·reserve seed 의 대조 id 가 가상 `G22` 가 된다(§11.2 위반) | 확장 묶음의 alias 로 해석(s4 H22 · s5 L100), 없으면 SystemExit |
| 보류 run 이 '시작됐지만 중단' 상태면 §9 와 달리 버려진다 | `qrc24_held_state` 가 checkpoint 존재로 판정하고, 그런 run 은 활성 큐·mandatory 맨 앞에 남겨 원 정의로 끝낸다 |

부수로 고친 것: 완료 run 을 실행 큐에서 빼 전환마다 되풀이되던 FR 평가·`pa_diag`(GPU)·시트 재업로드 walk 제거(이력은 git 큐·`queue_revision.json`) · `postrun` 이 이미 공식 선택이 끝난 run 을 다시 돌리지 않음 + GPU busy 판정에 FR 평가·smoke·다른 selector 포함 + flock · v2 config 의 `lambda_E_plan` 을 ADJ-R1 정의값 .002 로(원계획 절반값 아님) · 학습 manifest(`kdv_config_resolved.json`) 에 `qrc24` 블록(queue_revision·block_2x2) · 시트 Notes 에 U 초기값 hash(`init_sha`) · switch heredoc 의 import 누락 회귀 검사.

## 7. 남긴 것·한계

- 실행 중인 chain 은 **옛 runner 코드**를 메모리에 들고 있다(스크립트는 새 inode 로 교체) — 영속 큐 갱신은 그 chain 에서는 switch ⑥ 이, 다음 chain 부터는 runner 자신이 한다.
- `qrecon24_postrun.py` 의 backlog 는 이 서버의 완료 run 만 본다 — 다른 서버 결과는 그 서버의 경계에서.
- 시간 대용 s5 는 느린 시나리오(2.58 h) 로 예약한다(빠른 1.46 h 는 큐 머리에 병기). ledger 실측이 case 별로 생기면 그것이 우선이다.
- `verified_complete`·selector·시트 범주는 v2 이름을 그대로 다룬다(RUN_RE `_v\d+`). `_FRESH50_v2` 는 편성 이력 표시이지 q 정의 변경이 아니다(§11.2).
- 보류 run 이 이미 실행 중/중단 상태인 서버에서는 그 run 이 큐 맨 앞에 남아 원 정의로 끝나므로 시간이 표(§10) 보다 늘어난다 — `held_runs.json` 의 `status`·`kept_in_queue`·`checkpoint_present` 가 그 기록이다.
- 예약의 실측 경로는 ledger `hours_total`(평가·export 포함) 을 쓰고 거기에 10 분 후처리를 다시 더한다 — QRECON24 이전부터의 동작이라 이번에 바꾸지 않았다(보수적 방향).
