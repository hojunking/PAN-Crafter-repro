# PAKD50 파생 실험 재배정(s2/s4/s5) — 검토·구현·검증 (2026-09-15)

계획: `research_log/PAN_PAKD50_S2_S4_S5_Derived_Run_Allocation_2026-09-15.md` (이하 "재배정").
상위: `research_log/PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md`, 구현 노트 `research_log/2026-09-14_pakd50-implementation.md`,
s4/s5 노트 `research_log/2026-09-14_pakd50-s{4,5}-review-and-implementation.md`. 작성 s1, 2026-09-15 02:00 KST 부근.

## 1. 검토 — 계획과 저장소가 맞지 않던 곳

| # | 계획 | 저장소(재배정 전) | 처리 |
|---|---|---|---|
| 1 | §5 J_R3_NOEDGE · J_N0_EDGE · RC0 · RCQ 와 RC 정책 "생성기 등록 필요" | `tools/gen_pakd50_configs.py` 에 없음 (§9.3: 미등록 이름을 extra_priority 에 적는 것으로 구현했다고 치지 않는다) | 등록 (§2) |
| 2 | §5 RC = `A-FT + I-NATIVE-TRANSFER + aligner_lr 1e-5 + corruption.radius_hr 0 + offset 0`; "I-AEQ 에 weight 0 은 registry 계약과 맞지 않는다" | registry 가 정확히 그렇다 — `I-AEQ` 는 offset > 0 을 요구(`kdv/registry.py`), `I-NATIVE-TRANSFER` 는 radius > 0 을 거부 | RC 는 protocol 자체를 바꾼다. K17 이 두 거부를 검사 |
| 3 | §4 서버별 **명시 순서** (s2 JR→XJ→J_R3_NOEDGE→J_N0_EDGE · s4 F0→RC0→RCQ→JR→XJ→J_R3_NOEDGE · s5 PQ→F0→LF0→LFQ→RC0→RCQ), 완료 묶음 재편성 금지, s2 옛 꼬리(FR) 가 앞서지 않게 | s2 는 기본 PRIORITY(J0→JQ→F0→FQ→JR→FR→XJ), s4/s5 는 09-14 묶음 | `PRIORITY_BY_SERVER` 를 명시 순서로 교체(옛 묶음은 `PREVIOUS_PRIORITY_BY_SERVER` 로 보존). 완료 run 은 gate 의 `terminal()`(work_dir 완료 판정) 로 빠지고, FR 은 s2 순서에 없다 |
| 4 | §3 예약식 `reservation_h = 1.10 × reference_train_h + 10/60`, 기준값은 같은 서버 완료 case 의 Sheet Train(h) 대용값(N0 형 J0 / Teacher 형 JQ; s2 는 전부 2.33; PQ 1.80 가예약; D0 1.15 일반화 금지), §8 admission `finish ≤ training_deadline` | gate 는 완료 run 전체 평균 est 하나에 margin 1.1 (감사 F05) | `reference_hours / reservation_hours / reservation_for` + gate 가 run 별 예약을 누적해 admission. 같은 서버 같은 case 실측이 생기면 그것으로 교체(§3) |
| 5 | §3 trainer 예산 gate 도 같은 예약을 봐야 한다(구현 판단) | trainer 는 config `projected_map{me: 4.0h}` 보수값 | 서버 로컬 `work_dir/_pakd50/reservations.json`(gate 가 매 pass 씀) → config `budget.projection_file` → `train_kdv._projection` 이 우선 사용. config 는 서버 공용이라 값을 박지 않는다(s5 보고 #2 와 같은 원칙) |
| 6 | §9.2 "다음 run 경계에서" 전환, 진행 중 프로세스를 죽이지 않는다 | `tools/pakd50_requeue.sh` 가 이미 그 방식(runner bash 만 교체, 새 runner 는 `main.py` 가 끝날 때까지 대기) | `tools/pakd50_reallocate.sh` 가 검사·로컬 파일·예약 표 뒤 requeue 를 부른다 |
| 7 | §7 확인 seed(s4 3407 / s5 9091) 최대 3 run = WIN · 같은 policy no-KD control · F0 (F0 가 control 이면 2), 외부 WIN 뒤에만 | 09-14 절차(`--seed` 생성 → extra_priority) 는 있었으나 묶음 규칙이 없음 | `confirmation_cases(win)` + `pakd50_reallocate.sh --confirm <WIN>` (config 생성 · extra_priority · mandatory 추가; 예약 1.80 가예약, 같은 case 실측이 있으면 그 값) |
| 8 | §9.6 업로드·시트 그대로 | `_upload.sh` 는 `PAKD50_*` prefix, X열 `PAKD50 / <case> / FRESH50` 자동 | 변경 없음. 범주 설명(`gspread/sheet_categories.py` DESC) 에 새 토큰만 추가 |
| 9 | §2 s4 "J0 v2" 를 v1 과 합치지 않음 · §5 control hash 는 분석 담당자 확정 자산 | 생성기는 v1 만 만든다; T0/init/calibration hash 는 assets 고정 | 변경 없음 |

결이 다른 부분은 없다 — 손실·구조를 넓히지 않고(§6) 기존 policy/backend 축의 조합만 늘린다. 한 가지 이름 주의: `I-NATIVE-TRANSFER` 는 원래 frozen donor(F) 의 "domain-transfer 대조" 이름이지만 trainer 의 뜻은 "매 update native 입력, corruption 없음" 뿐이라 A-FT 와 결합해도 코드 경로는 그대로다(K18 로 gradient 확인).

## 2. 구현

| 파일 | 내용 |
|---|---|
| `tools/gen_pakd50_configs.py` | `POLICY["RC"]`(A-FT, I-NATIVE-TRANSFER, offset 0, A LR 1e-5, `corruption.radius_hr 0.0` 명시) · `BACKEND["R3_NOEDGE"]`(rec R3, edge 없음) · `BACKEND["N0_EDGE"]`(rec N0 + EDGE-H λE0; Teacher 는 loss 에 없어 `teacher.eval_only`) · case `J_R3_NOEDGE/J_N0_EDGE/RC0/RCQ`, `BASELINE_OF["RC"]="RC0"` · `PRIORITY_BY_SERVER`/`MANDATORY_BY_SERVER`(= 명시 순서) · `REFERENCE_TRAIN_H`, `ROUTING_PLACEHOLDER_H`, `CONFIRM_*`, `RESERVE_SLACK/POST_H` · `reference_kind/reference_hours/reservation_hours/reservation_for/measured_hours_from_ledger/write_reservation_file/confirmation_cases/plan_rows` · `schedule(est_hours=callable)` 이면 항목별 예약(여유 포함) 을 그대로 누적 · `kdv_block` 에 `budget.projection_file` · `--plan` dry-run |
| `tools/campaign_gate.py::gate_pakd50` | run 별 예약(`reservation_for`; 같은 서버 실측 `measured_hours_from_ledger` 우선) 으로 admission, 편성·밀린 run 을 `work_dir/_pakd50/reservations.json` 에 기록, 로그에 ref 출처·누적 |
| `train_kdv.py` | `_projection_file_hours`: `budget.projection_file` 의 `runs[run].gate_hours`(= reservation_h / margin) 가 있으면 config `projected_map`·ledger 평균보다 우선 (이 run 과 remaining mandatory 둘 다) |
| `tools/pakd50_reallocate.sh` | s2/s4/s5 전환 절차 (§3) · `--dry-run` · `--confirm <WIN>` |
| `tools/pakd50_unit_tests.py` | K17–K20 추가, K09/K11 의 묶음 검사는 `PREVIOUS_PRIORITY_BY_SERVER` 로 (78 검사 ALL OK) |
| `gspread/sheet_categories.py` | PAKD50 DESC 에 재배정 토큰 |
| `config/PAKD50_*` | 16 run 의 config (새 8: `J_R3_NOEDGE` S777/S1234 · `J_N0_EDGE` S777 · `RC0/RCQ` S1234/S2026 · `LF0/LFQ` S2026). 기존 config 는 머리 주석 한 줄과 `budget.projection_file` 키만 바뀌었다(학습 의미 동일; 완료·실행 중 run 의 기록은 `work_dir/<run>/meta/config.yaml`) |

약명→세팅 (재배정 §5; 시트 X열은 `PAKD50 / <case> / FRESH50`):

| case | 정책 (A) | backend (U 손실) | no-KD control | 시간 산정 유형 |
|---|---|---|---|---|
| J_R3_NOEDGE | J: T0 aligner 복사·trainable LR 1e-5, 홀수 update offset 연습 λ 1e-4 | R3 adaptive: (1+αd)L1 + β(1−d)a·\|S−T\|, **edge 없음** (JQ − λE L_E) | J0 | T |
| J_N0_EDGE | J | plain GT L1 + λE·GT edge, **Teacher 를 loss 에 쓰지 않음**(eval bin 만) | J0 | T |
| RC0 | RC: T0 aligner 복사·trainable LR 1e-5, **offset 연습 없음**(매 update native, radius 0) | N0 | RC0 | N0 |
| RCQ | RC | Q12 (R3 + λE edge; JQ 와 같은 계수) | RC0 | T |
| JR / XJ / F0 / PQ / LF0 / LFQ | 기존 등록 (09-14 노트) | | | JR/XJ/LFQ T · F0/LF0 N0 · PQ ROUTING |

시간 산정 (재배정 §3; **계획 기준값이지 실측이 아니다**): s2 N0 1.97 / T 2.33 → 이번 4 case 전부 2.33 · s4 N0 1.17 / T 1.39 · s5 N0 1.35 / T 1.36 · routing(PQ) 1.80 가예약 · 확인 seed 1.80 가예약.
같은 서버 같은 case 의 ledger 실측이 생기면 그것이 우선하고, routing 실측이 하나라도 생기면 다른 routing case 는 그 평균을 쓴다.
`--plan` 출력(측정값 없이) = 계획 §4 표와 같다: s2 4 run 10.9187 h · s4 6 run 9.6900 h · s5 6 run 10.4270 h (병렬 경과 = 최대 10.92 h).

## 3. 서버 절차 (s2/s4/s5; 각 서버에서 pull 뒤 한 번)

```bash
git pull                                   # config 16 벌 + 생성기/gate/trainer release
./tools/pakd50_reallocate.sh --dry-run     # ① λE 동기화 + unit gate(78) ② 순서의 config == 생성기 검사(완료·실행 중은 생략) ③ mandatory_runs.txt·reservations.json ④ 예약 표
./tools/pakd50_reallocate.sh               # ⑤ requeue: runner bash 만 교체 — 현재 학습(s2 JR / s5 PQ 등) 은 그대로 끝나고, 그 뒤 gate 가 명시 순서로 편성
tail -f work_dir/cases_chain.log           # "[cases] 대기 …" → 학습 종료 → "조건부 게이트(패스 1) 통과: <명시 순서>"
```

- 큐 파일(`config/queues/pakd50_<srv>_stage1.txt`) 은 그대로 J0 만이다 — J0 는 완료라 건너뛰고, 편성은 gate 가 매 pass 한다(runner 는 gate 출력 순서대로 돈다). 옛 pass 에 이미 편성돼 있던 FR 등은 runner 교체로 사라진다.
- 실행 중 run 이 명시 순서 안의 것(s5 PQ) 이면 gate 가 `_running` 으로 빼고 그 뒤부터 편성한다 — 같은 run 을 두 번 시작하지 않는다.
- 실패한 run(`work_dir/cases_failed.txt`) 은 재편성하지 않는다(재배정 §5 "PQ 실패면 원인 수정 version 으로 따로"): 원인을 고친 뒤 그 줄을 지우고 `pakd50_requeue.sh`.
- 확인 seed(§7): 외부 분석의 WIN 이 오면 `./tools/pakd50_reallocate.sh --confirm <WIN case>` — s4 3407 / s5 9091 로 WIN·control·F0(≤3) config 를 그 서버에서 만들고 extra_priority/mandatory 에 넣는다. 그 뒤 계수·release 를 바꾸지 않는다.
- 마감: 공통 `training_deadline` 2026-09-16 11:22:31 KST (assets/pakd50/campaign_clock.json). gate 는 예약 누적이 남은 시간을 넘는 run 을 밀고(로그 "admission"), trainer 는 같은 예약으로 예상 종료 ≤ 마감을 다시 검사한다.

## 4. 검증 (s1, 2026-09-15 01:40–02:10)

- `tools/pakd50_unit_tests.py` 78 검사 ALL OK. 새 검사: **K17** RC/J_R3_NOEDGE/J_N0_EDGE 의 registry 해석(정책·protocol·radius·offset·trainable·control·λE 의존·Teacher eval_only, I-AEQ+offset 0 거부, I-NATIVE-TRANSFER+radius 거부) ·
  **K18** 실제 `KDVTrainer._step`(CPU): RC0 update 1 에서 corrupt 없음·offset 연습 없음·ε RNG 미소비(J0 은 같은 update 에 연습)·Δ 에 graph·L_rec 가 A 와 U 양쪽으로, RCQ 는 hard+soft+λE edge 가 A 로도(routing 없음), J_N0_EDGE 는 Teacher forward 없음(y_t None)·edge > 0 ·
  **K19** 명시 순서·예약식 검산(10.9187 / 9.6900 / 10.4270)·기준값 출처·admission(s2 남은 5.0 h → JR 만)·실행 중 제외·확인 묶음 규칙 · **K20** trainer 가 예약 파일을 우선 보고(gate_hours×1.1 = 예약값) 없으면 종전 경로.
- `tools/smoke_cases.py` RC0/RCQ/J_R3_NOEDGE(S1234)·J_N0_EDGE(S777)·LFQ(S2026): 전부 통과 (2.6589 M, peak 2.8–3.3 GB, t_native 57–79 ms; s1 GPU 는 EQREC4 D40 과 공유 중이라 시간은 참고만).
- gate 시뮬레이션(임시 root 에 s2 상태: J0/F0/JQ/FQ 완료 + Sheet Train(h) ledger): 편성 `JR → XJ → J_R3_NOEDGE → J_N0_EDGE`, 예약 2.73 h 씩 누적 10.92 h, FR 없음, `reservations.json` 4 run 기록.
- `--plan --server s4` 를 s1 에서 돌리면 F0/JR 이 완료로 보인다 — s1 이 seed 1234 를 공유해 같은 run id 를 이미 돌렸기 때문이고, s4 의 work_dir 에서는 6 run 전부 planned 다.

## 5. 남긴 것

- 예약 기준값은 재배정 §2 의 Sheet 값(2026-09-15 00:05 read) 을 그대로 상수로 뒀다. 새 case 의 첫 완주가 나오면 gate 가 ledger 실측으로 바꾸지만, 표 자체를 갱신하는 것은 사람이 한다(결과 수치로 기록하지 않는다 — §3).
- 후처리 10 분 가예약이 실제로 초과되는지는 `_upload.sh` 시간을 따로 재지 않으면 알 수 없다(§3) — 큐 비용 갱신은 요청 시.
- s1/s3 는 신규 배정 없음(§1). s1 의 gate 는 켜져 있지만 체인이 멈춰 있어(EQREC4) 아무것도 편성하지 않는다.
