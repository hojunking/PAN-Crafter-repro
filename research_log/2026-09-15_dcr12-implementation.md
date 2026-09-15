# DCR12 (offset consistency × reconstruction 불일치 검증) — s1 실행판 검토·구현·검증 (2026-09-15)

계획: `research_log/PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md` (이하 §). 사용자 지시 2026-09-15 09:2x: "s1 에서 돌릴 수 있게 구현/검증". 이 문서는 구현·운영 노트다(결과 문서는 캠페인 뒤 `results_log/` 새 날짜 문서).

## 1. 검토 — 계획과 저장소가 맞지 않던 곳 / 결정

| # | 계획 | 저장소·결정 |
|---|---|---|
| 1 | §0.2 B0=FQ · B1=JK0 은 generator 정의 있음, B2/B3 미구현 | 맞다. B0/B1 학습은 `tools/gen_pakd50_configs.py` 그대로(v1). B2/B3/CMASS/B1-NOOFF 는 **구현하지 않았다** — §9.2 gate 가 열릴 때만 별도 release(§9.4 "자동 실행하지 않는다") |
| 2 | §8.2 seed 1234·777, 같은 호스트에서 pair 완성; exact-matching asset 은 재사용 | s1 의 `PAKD50_FQ_…_S1234_FRESH50_v1`(완료, best_hqnr step 33330, HQNR 0.953712) 을 B0-1234 로 재사용. JK0-1234 · FQ-777 · JK0-777 은 s1 에서 새로 학습(큐 `config/queues/dcr12_s1.txt`, 순서 §8.2). **FQ S777 은 s2 의 같은 이름 run 과 별개 host 학습**(§8.2 "다른 서버에 FQ 만" 금지; 이름이 같아도 시트는 서버 탭이 다르다). seed 777 의 U 초기값은 s1 이 새로 만든다(`init_hashes.json` 의 777 은 null — s2 값과 같다고 주장하지 않음) |
| 3 | §12.2 `--version v1201` 충돌 회피 | 쓰지 않았다: v1 config 를 그대로 써야 B0-1234 완료본(v1) 과 같은 정의·이름 규약이고, s1 에는 같은 이름 run 이 없어 덮어쓰기가 없다. `analysis_campaign_id=DCR12_20260914_v1` 은 manifest.json sidecar 에 기록 |
| 4 | §12.2 DCR12 전용 budget ledger·gate 격리 | config 의 `kdv.budget` 은 PAKD50 ledger(50 h, 마감 09-16 11:22:31; s1 사용 10.6 h) 그대로 — 바꾸면 config identity 가 달라진다. 격리는 `work_dir/campaign_gates_enabled.txt` 에서 `pakd50` 을 비워(백업 `work_dir/_dcr12_gates_backup.txt`) 큐 뒤 gate 가 옛 우선순위(FR/XJ …)를 재주입하지 않게 했다(D00 manifest `execution_isolation`) |
| 5 | §4.2 필수 smoke | `tools/dcr12/d00.py` 가 실제 trainer 경로로 검사: warp 부호 · native task→A · soft routing(JK0 의 A .grad == ∇φ(L_H+λE L_E+1e-4 L_off), U .grad == JQ) · offset→U 없음 · frozen B0 A == T0 A · Teacher 출력 불변 · identity probe floor · forward 기준(총 loss 재계산 일치) · RNG 격리 · probe support. **routing 허용치 1e-4 상대**(GPU FP32 reduction; CPU K12 검사는 1e-5) — D00 첫 실행이 1.1e-6/0.0122 로 1e-6 허용치에 걸려 조정 |
| 6 | §5.1 panel: 원 scene id 없음 | EQREC4 와 같은 32 연속 index block proxy(`independence=patch_only`); CAL 32 block / DISC 16 / CONF 16 서로 다른 block, seed 314159. §5.1 대로 B2/B3 gate 개방은 보류 조건 |
| 7 | §5.2 probe 16 + confirm 16 | r ∈ {0.5, 1.0} × θ = kπ/4 (16) — EQREC4 의 bank A∪B 의 r 0.5/1.0 과 같은 집합; confirm = θ + π/8. 반경별 C 를 먼저 기록 |
| 8 | §5.3 Student bin | 같은 seed B0 best_hqnr(완료 run) 의 CAL median 을 B1 에도 적용(`cell_ref`), 각 pipeline 의 자기 median 은 `cell_own` 으로 분리 |
| 9 | §6.1 개입 domain | native64: 전체 64² + interior 32²(margin 16, 고정) 둘 다; RR: full + roi192; FR: raw_original(+valid views). ±0.25 px bias 는 고정 proposal 그대로, FR 에서는 FR8 고정 subset(partial) |
| 10 | §7 gradient 분해의 ε | 학습 sampler(radius 2 disk-uniform) 대신 **r=1 4 축 고정 ε** 의 SG offset loss 로 gC 를 정의(결정적 대조; EQREC4 D50 과 같은 선택). 실제 학습 gradient 는 홀수 update 의 1e-4·gC(무작위 ε) 라는 점을 summary 에 기록 |
| 11 | §7.3 학습 시점 5050/25250/45450/50000 | GRID1010 후보에 있다(5×1010, 25×1010, 45×1010, exact 50000) — B1 의 candidates 폴더에서 읽는다 |
| 12 | §9 D04 optimizer state | candidate 에 optimizer.bin 이 있지만 **fresh AdamW(그 step 의 cosine LR)** 로 양쪽 동일하게 — "continuation 이 아니다" 를 CSV 에 기록 |
| 13 | §13 산출물 | 이름 그대로(`manifest.json source_identity.json panel_ids.json probe_manifest.json quadrant_thresholds.json sample_metrics.csv correction_interventions.csv gradient_conflicts.csv checkpoint_metrics.csv quadrant_summary.csv micro_update_utility.csv paired_results.csv run_status.json report.md`) + `scene_metrics.csv closure_invariance.csv synthetic_composition.csv d0*_stats/summary.json routing_gate.json`; `routing_policy.json` 은 gate 통과 시만 |

## 2. 구현

| 파일 | 내용 |
|---|---|
| `tools/dcr12/common.py` | registry(T0 자산, B0/B1 × seed, tag best_hqnr/last/cand:<step>; **완료 run 만**), panel 생성, probe 집합, `consistency`(C·반경별·축별·identity·closure-bias wrapper), `reconstruct`(delta_override), `r_metrics`, `q12_terms`(학습과 같은 loss 객체), `trainer_stub`(실제 `KDVTrainer._step/_apply_routing`), `cosine_lr` |
| `tools/dcr12/d00.py` | manifest(host/GPU/torch/git/dirty, T0 file·A·U tensor sha, calibration sha, run 상태·init hash, 계약·격자·comparator, 격리 상태, 예상 시간) · source_identity · smoke(§4.2) |
| `tools/dcr12/d01.py` | sample_metrics(CAL/DISC/CONF × pipeline), thresholds(T0 CAL median; DEGENERATE_METRIC), cell_ref/cell_own, quadrant_summary(SMALL_CELL), d01_stats(Spearman+group bootstrap, Pearson, 반경별, confirm, texture), scene_metrics(RR20 R·reduced, FR20 raw views), GRAD64(T0 cell 균형) |
| `tools/dcr12/d02.py` | I0/IZ/IN/IB±y/IB±x on DISC128·RR20·FR20(FR8 partial) · closure 불변(§6.3) · seq/comp 합성(§6.4) · B_R/B_Q(group bootstrap), ±bias 평균/최선/최악, cell 별 B_R |
| `tools/dcr12/d03.py` | GRAD64 per-sample: g0·gH·gE·gK·gC 의 norm·cos·비율·NA_ZERO_GRAD(floor = 같은 gradient 반복차 ×10), dL0/dc0·dLE/dc0, virtual step(η 1e-5, SG 고정 closure vs 재계산 closure, 1차 예측 vs 실제 ΔL_task), B1 checkpoint 의 routing 검사 |
| `tools/dcr12/d04.py` | H-copy/K-copy micro-update(4×16 support, 8 update, query 16 같은 cell R_T 전 구간), DISC/CONF × B0 25250/45450, U = R_H − R_K, §9.2 gate(5 조건) → routing_gate.json(통과 시만 routing_policy.json; B2/B3 미실행) |
| `tools/dcr12/results.py` · `report.py` | run 별 selected/numerical max/plateau(45450–50000)/last HQNR·Dλ/Ds·시간, paired B1−B0(seed 별, scene bootstrap CI, DISC/CONF plain L1, C median 변화) · §13.2 7 질문 + §14.1 라벨 자동 적용 |
| `tools/dcr12.py` · `tools/dcr12_run.sh` | CLI(stage·check --of) · runner: pre(D00→D03) → 학습 큐 대기(실패면 INVALID 중단) → post(D01/D02/D03 증분 → D04 → RESULTS → REPORT) |
| `tools/dcr12_unit_tests.py` | X01–X13 (13 검사 ALL OK) |
| `config/PAKD50_JK0_…_S1234_v1.yaml`, `…_S777_v1.yaml`, `config/queues/dcr12_s1.txt` | B1 두 seed config(FQ S777 은 기존 파일) |

## 3. 실행 (s1)

- 09:32 D00 통과 → 09:33 학습 체인 기동(`campaign_start.sh --queue config/queues/dcr12_s1.txt --hours 20`, 감시자 cron) — JK0 S1234 → FQ S777 → JK0 S777 (예상 각 2.2–2.5 h).
- 09:4x D01(T0 + B0-1234: tC 0.319 px, tR 0.01706; T0 DISC cell A 137 / B 121 / C 140 / D 114, DEGENERATE 아님) → runner `tools/dcr12_run.sh` 가 D02·D03(pre) 뒤 학습 완료를 기다려 post 단계를 돈다. 진단은 학습과 GPU 를 같이 쓴다(가벼움).
- 시작 직후 한 번 실수: D01 이 **학습 중인 JK0-1234 의 best_hqnr** 을 집어 계산했다(디렉토리가 학습 중에도 생긴다). `available_students` 를 완료 run 만으로 고치고 그 행을 지운 뒤 다시 돌렸다(X12 가 검사).
- 확인: `tail -f work_dir/_dcr12_s1_campaign/run.log`, 학습은 `work_dir/cases_chain.log`.
- 11:46 **hold**: 사용자의 서버 교체 제안(§5) 을 검토하며 `work_dir/cases_deadline.txt` 를 과거 시각으로 돌렸다 — chain 은 JK0 S1234(≈12:10) 를 끝내고 FQ/JK0 S777 을 '마감 경과' 로 건너뛴 뒤 DONE(감시자 정지), runner 는 미완 run 을 보고 STOPPED 로 끝난다. 원래 마감 사본 `work_dir/_dcr12_s1_campaign/cases_deadline_before_hold.txt`. s1 에서 계속하려면 `./tools/campaign_start.sh --queue config/queues/dcr12_s1.txt --hours 20` 뒤 runner 를 다시 띄운다(완료 stage 는 건너뛴다).

## 4. 남긴 것

- B2/B3/CMASS/B1-NOOFF 미구현(§9.4·§10 조건부). D04 gate 가 열리면 별도 release 로.
- 계획 §11.2 의 300-update smoke 처리량 재산정은 하지 않았다 — 같은 골격·같은 trainer 의 s1 실측(FQ 2.15 h, JQ 2.22 h) 이 있어 그 값을 예상치로 썼다.
- FR 의 ±bias 는 FR8 만(partial), 합성 이동(§6.4) 은 DISC128 만.

## 5. s2 이관 준비 (2026-09-15 12:00; 사용자 제안 — 확인 대기)

**왜**: SMEC12 의 WV3 lane 자산(PALS24/NF16/PO10 run) 이 s1 에만 있어 SMEC12 는 s1 이 맞고, DCR12 의 B0/FQ S777 완료본은 s2 에 있으며 s2 의 파생 큐(JR/XJ/J_R3_NOEDGE/J_N0_EDGE, 777) 는 시트에 전부 올라와 있다(12:00 확인 → s2 GPU 가 비어 있거나 곧 빈다). s1 의 JK0 S1234(12:10 완료) 를 버리지 않도록 seed 1234 pair 는 **bundle 로 옮긴다** — 계획 §8.2 의 "host A: FQ1234+JK0 1234 / host B: FQ777+JK0 777" 배치(빠른 확인 배치; seed 와 host 가 결합되므로 REPORT 가 `SEED_HOST_COUPLED` 라벨을 붙이고 최종 재현 주장은 한 호스트 두 seed 또는 pair 복제로 보강한다).

**바꾼 것** (gate X01–X15 ALL OK, s1 `dcr12_prepare.sh --dry-run` 통과):

| 파일 | 내용 |
|---|---|
| `tools/dcr12_run.sh` | 서버 공용: 큐 `config/queues/dcr12_<server>.txt`(없으면 시작 안 함). train phase 가 **다른 큐의 chain 이 돌면 끝날 때까지 기다렸다가** gate 'pakd50' 를 비우고(백업 `work_dir/_dcr12_gates_backup.txt`) `campaign_start` 로 우리 큐를 기동; 미완 run 을 남기고 chain 이 끝나면 최대 3 회 재기동, `cases_failed.txt` 에 우리 run 이 있으면 INVALID. (s1 에서 돌던 runner 는 옛 inode 로 계속 돈다 — mv 로 교체) |
| `tools/dcr12_bundle.py` | `pack --seed 1234`(학습 호스트) → `verify` / `install`(분석 호스트). run 마다 meta/config.yaml(+started/finished/git) · best_hqnr/last model+meta · results/fr_mat20.json · checkpoint_metrics.csv · initialization_hashes.json · candidates step-{5050,25250,45450,50000}/model.safetensors(D03∪D04 격자; optimizer 는 D04 가 fresh AdamW 라 제외). 실측 127 MB (30 files; .mat 제외 — sha/size 만 manifest, `--with-mats` 로 포함 가능); 첫 pack(12:07) 은 .mat 포함 1.7 GB 였고 진단이 .mat 을 읽지 않아 뺐다 — install 된 run 의 완료 판정은 provenance `complete_on_source`. 미완 run 은 묶지 않는다(X12 와 같은 규칙). install 은 `work_dir/<run>/bundle_provenance.json` 을 남긴다 |
| `tools/dcr12/common.py` | `provenance / trained_on / host_pairing`(pair 안 같은 호스트 여부 · seed 간 호스트 결합) |
| `tools/dcr12/d00.py` · `results.py` · `report.py` | manifest run 항목에 `trained_on/provenance`, `host_pairing` 을 계산값으로(문자열 → dict); RESULTS 에 `trained_on/finished_at`; REPORT 머리에 학습 호스트 줄, `reused` 는 "campaign T0 이전에 끝난 B0", seed 간 호스트가 다르면 라벨 `SEED_HOST_COUPLED` |
| `config/queues/dcr12_s2.txt` | JK0 S777 v1 만(B0/FQ S777 은 s2 완료본 재사용; seed 1234 pair 는 bundle) |
| `tools/dcr12_prepare.sh` | ① 서버·큐·config·T0·calibration·이 서버 seed 의 B0 완료본 ② bundle verify→install ③ gate ④ registry·마감 여유 ⑤ runner detached (`--dry-run` 은 verify 까지) |
| `tools/dcr12_unit_tests.py` | X14 bundle 정의(필수 파일·optimizer 제외·trained_on) · X15 서버 큐(run 이름·config) |

**s1 (12:10 확인)**: JK0 S1234 12:04:59 완료(best_hqnr step 26260, raw HQNR 0.95693) → chain 은 FQ/JK0 S777 을 '마감 경과' 로 건너뛰고 12:09:54 DONE(hold 대로), bundle 은 12:07 에 pack 완료(`work_dir/_dcr12_bundle/`, 1.7 GB (34 files; results .mat 1.6 GB 포함 — 완료 판정·FR 재평가용, checkpoint 127 MB); 로그 `pack.log`). s1 의 `_dcr12_s1_campaign/`(D00–D03 pre, T0+B0-1234) 은 기록으로 두고 s2 결과와 합치지 않는다.

**s2 절차** (pull 뒤):

```bash
rsync -a s1:/home/knuvi/Desktop/song/PAN-Crafter/work_dir/_dcr12_bundle/ work_dir/_dcr12_bundle/   # 실측 127 MB, 30 files (bundle_manifest.json 포함)
./tools/dcr12_prepare.sh --dry-run      # 큐·T0·FQ S777 완료본·bundle verify·gate·registry·마감 여유
./tools/dcr12_prepare.sh                # bundle install → runner detached: D00–D03(pre: T0 + FQ S777 + 1234 pair) → 다른 chain 이 끝나면 JK0 S777 학습(≈2.3–2.7 h) → post D01–D04 → RESULTS → REPORT (≈1.5 h)
tail -f work_dir/_dcr12_s2_campaign/run.log
```

- **마감**: JK0 S777 config 는 PAKD50 generator 출력 그대로라 `kdv.budget.training_deadline` 2026-09-16 11:22:31 · `required: false` 를 갖는다 — 예상 종료(4.0 h × margin 1.1) 가 마감을 넘기면 trainer 가 DEFERRED(rc=4) 로 시작하지 않고 runner 는 INVALID 로 멈춘다. 09-16 07:00 이전 기동이면 문제없다. 그 뒤라면 그 config 의 `kdv.budget.required` 를 `true` 로 바꿔(DCR12 전용 run; PAKD50 순서 검사 대상이 아님) 돌리고 노트에 적는다.
- s2 의 chain 이 아직 돌고 있으면 runner 가 5 분 간격으로 기다렸다가 기동한다(그 chain 의 `[cases] DONE` 뒤). 시트 확인으로는 s2 의 4 run 이 전부 올라와 있어 곧 빈다.
- 교체하지 않기로 하면: s1 에서 §3 의 되돌리기 명령으로 chain 을 다시 열면 FQ/JK0 S777 을 이어 돈다(JK0 S1234 완료본 그대로).
