# PAKD50 s3 추가 실험(J_R3_NOEDGE → J_N0_EDGE → LF0 → LFQ → LFX) — 검토·구현·검증 (2026-09-15)

계획: `research_log/PAN_PAKD50_Latest_Sheet_Analysis_and_S3_Experiments_2026-09-15.md` (이하 §). 재배정 공통 기반: `research_log/2026-09-15_pakd50-derived-allocation-implementation.md`(예약식·reallocate 절차). 작성 s1 10:5x.

## 1. 검토

| # | 계획 | 저장소·판단 |
|---|---|---|
| 1 | §4.3/§7.3 LFX = `CASES["LFX"] = ("LF", "X02")` 신규 등록 (설명·완료 판정·예약·unit test) | 등록. resolved config: `aligner_schedule{freeze_from: 25000}` + rec R1(hard-only α1 β0) + EDGE-H λE0 + offset 1e-4(I-AEQ), routing 없음, control `LF0`. **일정 키를 빼면 XJ 의 config 와 같다**(K21 이 dict 대조) — 0–24999 계약 동일 |
| 2 | §4 s3 명시 순서 5 case, 기존 J0/JQ/JR/XJ/F0/FQ/FR 은 control 재사용·재편성 금지 | `PRIORITY_BY_SERVER["s3"]`(= mandatory). 순서에 기존 case 가 없고 gate 의 완료 판정으로도 제외된다. s3 의 종전 기본 순서는 `PREVIOUS_PRIORITY_BY_SERVER["s3"]` 로 보존 |
| 3 | §6 reference: 1.34/1.34/1.16(J0)/1.34/1.33(XJ) → 예약 합 7.9943 h | `REFERENCE_TRAIN_H["s3"] = N0 1.16 / T 1.34` + case 대용값 `REFERENCE_CASE_TRAIN_H["s3"]["LFX"] = 1.33`(유형 표보다 우선). `--plan --server s3` 가 §6 표와 같은 1.6407/1.6407/1.4427/1.6407/1.6297, 합 7.9943 h |
| 4 | §5 확인 seed 4321, 후보별 묶음 표(≤3 run), 확인 reference 1.34(JQ) | `CONFIRM_SEED["s3"] = 4321`, `CONFIRM_BUNDLE_BY_SERVER["s3"]`(LFQ→JQ/LF0/LFQ · LFX→XJ/LF0/LFX · J_N0_EDGE→J0/XJ/J_N0_EDGE · J_R3_NOEDGE→J0/JR/J_R3_NOEDGE · XJ→J0/JR/XJ · JQ→J0/XJ/JQ; 표 밖 후보 거부), `CONFIRM_REFERENCE_H["s3"] = 1.34`. s2/s4/s5 는 종전 규칙(WIN·control·F0, 1.80 가예약) 유지. `pakd50_reallocate.sh --confirm <후보>` 가 서버 규칙을 고른다 |
| 5 | §4.4 prefix·초기화·RNG 계약: LF0/J0, LFQ/JQ, LFX/XJ 의 처음 25K 가 같아야 하고, exact25K 전체 state 가 없으면 full 50K | 코드 경로는 같다(`aligner_active(step)` 가 25000 전에는 항상 True; routing 없음; ε RNG 소비 동일 — K12/K14 가 CPU 에서 4999/5000 경계로 검증, freeze_from 도 같은 함수). exact 25000 전체 state 는 기존 run 에 없다(candidate 는 1010 격자·checkpoint-N 은 10000 격자; **§4.4 대로 25,250 을 대체하지 않는다**) → 계획대로 full 50K. **단, GPU 학습은 bitwise 재현이 아니다**: 같은 config·seed 의 s4 J0 v1/v2 가 0.9594 vs 0.9568 이었다. 따라서 LF 계열과 그 대응 recipe 의 처음 25K 는 "정의·RNG 가 같다" 이지 "같은 궤적" 이 아니며, §3.2 의 s5 LF0(0.9517) vs J0(0.9570) 차이도 이 run-to-run 폭(≈0.003) 안팎이라 Sheet 만으로는 버그로 볼 수 없다. 정확한 prefix 공유가 필요하면 exact25K 전체 state 를 저장하는 run 을 따로 만들어 분기해야 한다(이번 release 에 없음) |
| 6 | §7.4 `--cases` 로 config 만 만들고 편성이 끝났다고 가정하지 않는다 | config 5 벌은 s1 에서 만들어 커밋(LF0/LFQ 는 재배정 때 이미 존재). 편성은 s3 에서 `./tools/pakd50_reallocate.sh`(mandatory/reservations 로컬 파일 + requeue) |
| 7 | §7.5 skip/resume · §7.6 hot pull 금지 | s3 의 기본 순서(J0→…→XJ) 는 Sheet 상 7 행 전부 완료라 chain 이 끝나 있을 것이다(확인은 s3 에서 `ps`/`cases_chain.log`). 돌고 있다면 reallocate 가 runner 만 교체하고 현재 run 은 끝까지 간다. 완료·실행 중 run 은 config 검사에서 제외 |
| 8 | §7.7 시트 | 변경 없음(`WV3-s3(5090)` 탭, X열 `PAKD50 / <case> / FRESH50`); 범주 설명에 s3 토큰만 추가 |

계획 §2–§3 의 해석(soft 효과의 block 별 반전, LFQ +0.007 의 귀속, RC 의 작은 차이, D/P/AL 재스윕 보류)은 실행에 영향이 없어 그대로 둔다.

## 2. 구현

- `tools/gen_pakd50_configs.py`: `CASES["LFX"]`·PURPOSE, `PRIORITY_BY_SERVER["s3"]`, `REFERENCE_TRAIN_H["s3"]`, `REFERENCE_CASE_TRAIN_H`, `CONFIRM_SEED["s3"]`, `CONFIRM_BUNDLE_BY_SERVER`, `CONFIRM_REFERENCE_H`, `confirmation_cases(win, server)`, `reference_hours` 의 case 대용값·서버별 확인 reference, `--plan` 이 s3 계획을 인용.
- `tools/pakd50_reallocate.sh`: s3 허용, `--confirm` 이 서버 규칙 사용.
- config: `PAKD50_{J_R3_NOEDGE,J_N0_EDGE,LFX}_W112_D123_WV3_T0_S2026_FRESH50_v1.yaml` 신규(LF0/LFQ S2026 기존). smoke 3 벌 통과(s1 GPU 공유 중이라 시간은 참고만).
- `tools/pakd50_unit_tests.py` K21(3 검사) + K19 조정 — 81 검사 ALL OK.
- `gspread/sheet_categories.py` DESC 에 s3 토큰.

## 3. s3 절차 (pull 뒤)

```bash
git pull
./tools/pakd50_reallocate.sh --dry-run     # unit gate · config == 생성기 · mandatory_runs.txt/reservations.json · 예약 표(7.9943 h)
./tools/pakd50_reallocate.sh               # runner 교체 → gate 가 J_R3_NOEDGE → J_N0_EDGE → LF0 → LFQ → LFX 편성 (마감 09-16 11:22:31 안, 09-15 11:00 기준 24 h)
# 외부 분석의 후보 확정 뒤:
./tools/pakd50_reallocate.sh --confirm LFX   # seed 4321 · XJ/LF0/LFX config 생성 + extra_priority/mandatory 등록 (후보별 묶음은 §5 표)
```

- seed 4321 의 U 초기값은 첫 확인 run 이 s3 에서 만들고(`work_dir/_kdv_init_w112_d123/init_unet_seed4321.pt`) 같은 묶음의 run 이 공유한다. 그 hash(`work_dir/<run>/initialization_hashes.json`) 를 `assets/pakd50/init_hashes.json` 에 적어 두면 이후 config 가 `expect_init` 으로 고정한다.
- `work_dir/campaign_gates_enabled.txt` 에 `pakd50` 이 있어야 한다(prepare 가 넣었다; reallocate 가 검사).
