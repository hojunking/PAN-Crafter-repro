# PAKD50 s4 배정 검토·구현 노트 (2026-09-14)

배정 문서: `research_log/PAN_S4_Integrated_Experiment_Cases_2026-09-14.md`(§번호는 그 문서). 동반 문서로 적힌 `PAN_S4_Gspread_Implementation_Handoff_2026-09-14.md`·`PAN_S4_CaseRegistry_2026-09-14.yaml` 은 저장소에 없어 본 문서만으로 검토했다.
상위 계획 `PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md`, C0 구현 노트 `2026-09-14_pakd50-implementation.md`(§6 감사 반영 포함).

## 1. 검토 — 현 실험과의 결

**결이 다르지 않다.** T0(PALS24 L1E4 seed 2025 best_raw)·W112·D123·FRESH50·GRID1010_50K_v1·raw-original HQNR 판정·τR/λE 공통 package 를 그대로 쓰고, 상위 계획 §9.4 의 P2(AL0/ALQ, scalar 조정)·P4(같은 seed 서버 교차, seed 3407) 를 s4 로 떼어낸 것이다. s1–s3 의 P0/P1 큐는 건드리지 않는다.

주의해서 읽을 점:

| 항목 | 판단 |
|---|---|
| seed 1234 를 s4 에서 다시 돌림(B00/B01) | 독립 seed 가 아니라 **환경 교차(bridge)** 다. 배정 문서도 그렇게 적었다. 집계에서 n_unique_seed 에 넣지 않는다 |
| AL(A LR 3e-6) 을 첫 탐색으로 | 상위 계획 §9.4 는 AL0/ALQ 를 "명확한 drift 일 때 대체" 로 뒀는데 s4 는 기본 묶음으로 올렸다. 신규 가설이지만 s4 는 추가 용량이라 허용. **ALQ−AL0 이 0 이면 KD 순증분으로 쓰지 않는다**(§5) |
| Q12 scalar 5종 중 최대 2개 | 계수만 바꾸는 case(상위 계획 C1 목록의 QA05/QB005/QB02/QE025/QE10) — loss 정의 변경 없음. 근거 없이 grid 로 다 돌리지 않는다(§6.1) |
| seed 3407 | winner_lock 뒤에만. 확인 seed 를 보고 계수를 다시 고르면 개발용으로 표시(§9) |
| 50h 시계 | s4 도 **공통 절대 시계**(`assets/pakd50/campaign_clock.json`) — 이미 구현(감사 F05). 세 서버 모두 KST 라 timezone 변환은 없음 |
| 배정 문서의 저장소 snapshot 3d06885 | 오늘 감사 반영 커밋(54915c5) 으로 큐 규약이 바뀌었다: 큐 파일은 J0 만, 나머지는 gate 가 우선순위로 편성. s4 도 같은 기전을 쓴다(아래) |
| s4 하드웨어·처리시간 | 미실측. prepare 의 smoke 가 case 별로 잰다. 1 slot 1 run 기본(§2·§10.4) |

## 2. 구현 (s1 에서, 학습 경로 변경 없음)

| 구성 | 내용 |
|---|---|
| 서버 | `gen_pakd50_configs.SERVER_SEED["s4"] = 1234`; `pakd50_prepare.sh` 가 s4 허용 |
| T0 경로 | 모든 서버에서 `assets/pakd50/T0_run`(같은 sha 파일; trainer 가 `expected_sha256` 검사). s4 가 s1 과 같은 run id 를 쓰므로 config 가 서버와 무관하게 같아야 한다. s1 의 J0-1234 는 work_dir 원본 경로로 시작했고 이후 run(F0/JR/FR…) 은 자산 경로 — 수치 경로 동일 |
| s4 기본 묶음 | `PRIORITY_BY_SERVER["s4"] = [J0, JQ, AL0, ALQ]`(§5·§10.1 B0/B1/E0/E1), 예산 예약(`remaining_mandatory`) 도 이 넷. stage 1 config = J0/AL0, stage 2 = JQ/ALQ(λE0 필요) |
| 추가 편성 | `work_dir/_pakd50/extra_priority.txt` 에 case id(`J_QA05` …) 또는 전체 run 이름(seed 3407 등) 을 한 줄씩 — gate 가 기본 묶음 뒤에 그 순서로 편성(중복 제거, λE 없으면 λE case 는 미편성, 공통 마감 admission) |
| scalar case | `BACKEND` Q12_A05/Q12_B005/Q12_B02/Q12_E025/Q12_E10 → case `J_QA05 J_QB005 J_QB02 J_QE025 J_QE10` + `AL_*` 동형(§7 결합용). α = `rec.alpha`, β = `rec.kd_weight`, λE = λE0 × 배율(`stat.outer_weight`; 다시 calibrate 하지 않으며 λE0 없이는 생성 거부). anchor JQ/ALQ, no-KD J0/AL0 |
| 확인 seed | `python tools/gen_pakd50_configs.py --server s4 --cases J0,JQ,<WIN> --seed 3407` → run 이름을 extra_priority.txt 에 |
| 시트 | `gspread/server.txt` = `s4` 면 업로더가 `WV3-s4` 탭을 만든다(이미 손으로 만든 빈 탭이 있으면 그대로 쓰고 첫 업로드에서 B:W 헤더를 쓴다). 범주 ㉕ PAKD50(`sheet_categories`, KEEP·DESC 구분행). **X열 `통합실험`** = `PAKD50 / <case> / FRESH50` — B:W 열 배치 검사 밖이라 s1–s3 기존 탭에도 충돌 없이 붙는다. 새 spreadsheet 는 만들지 않는다(§12) |
| 진단 | `_upload.sh` 의 pa_diag 분기에 PAKD50 포함(정합 진단 json). PAKD50 전용 paired/plateau 집계·offset response 는 후속(감사 F10) |
| 검사 | `tools/pakd50_unit_tests.py` K09(scalar 계수·λE 배율 거부·s4 묶음·추가 편성·to_tag/case_of) |

## 3. s4 절차

1. 저장소 pull(이 커밋 이후) → `echo s4 > gspread/server.txt` → 데이터(`data/PanCollection/WV3`, DLPan) 배치 → `./tools/setup_paths.sh --apply`(config 절대경로).
2. `./tools/pakd50_prepare.sh` — T0 자산 sha·raw HQNR 재현(허용 1e-4) → gate → τR 재계산 대조 → stage 1 config(J0/AL0) → smoke(case 별 처리량) → 공통 시계 적용(학습 마감 09-16 11:22:31) → 체인 기동(큐 J0). 그 뒤 gate 가 λE0 사본이 있으면 JQ → AL0 → ALQ, 없으면 AL0 한 벌 뒤 다시 확인.
3. 진단을 보고 scalar 를 고르면 `python tools/gen_pakd50_configs.py --server s4 --cases J_QA05` + `echo J_QA05 >> work_dir/_pakd50/extra_priority.txt`. 체인이 이미 DONE 이면 `./tools/pakd50_prepare.sh --stage 2`(마감은 공통 시계).
4. winner_lock 뒤 seed 3407: `--seed 3407` 로 config 생성 → run 이름을 extra_priority.txt 에.
5. 업로드는 체인이 run 마다 `_upload.sh` 로 한다(`WV3-s4`, X열 통합실험). 손으로는 `python gspread/gspread_upload.py PAKD50_*`.

미구현: 배정 §13 의 s4 산출물(`s4_manifest.json`, `winner_lock.json`, paired/bridge/seed summary csv, upload_status) 자동 생성과 §11.4 진단 표 — 결과가 나오기 시작하면 `tools/pakd50_report.py` 로 묶는다. §12 의 실행 전용 worktree 분리는 감사 F07 과 같이 후속.
