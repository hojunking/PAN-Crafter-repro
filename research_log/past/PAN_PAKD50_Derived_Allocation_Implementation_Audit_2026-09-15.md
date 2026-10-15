# PAKD50 s2/s4/s5 파생 배분 구현 감사

작성: 2026-09-15 KST. 검증 기준 commit: `2d0fc25e54aa190688031f2feba0ad2a22db506c`.

**판정: 기본 16run의 학습 정의·설정은 통과한다. 현재 인계 절차와 재배정 스크립트 그대로의 적용은 보류해야 한다.** 실제 손실/gradient 구현과, 그 실험을 정해진 범위·순서·마감 안에서 완주하고 업로드하는 운영 계약의 검증 결과가 다르다. 아래 P1 항목은 적용 전에 해결해야 한다.

대상은 [파생 배분 계획](PAN_PAKD50_S2_S4_S5_Derived_Run_Allocation_2026-09-15.md) §1–9이다. 앞선 Integrated/EQREC4 감사의 결론을 현재 코드에 그대로 적용하지 않았다. 새 case 등록, 현재 generator/registry/trainer, runner/gate/업로드 경로를 다시 읽고 확인했다.

**1. 검증 범위와 증거**

- 로컬 서버 표시는 `s1`이다. s2/s4/s5의 현재 checkout, PID, step, 완료 상태, 원장, 초기값 파일은 이번 감사에서 직접 조회하지 않았다. 이 문서는 원격 배포 완료 증명이나 실험 결과 분석이 아니다.
- 기존 단위검사 **78개 통과**. 실제 스크립트를 `/tmp`의 격리된 저장소에서 실행했다. K08이 `mandatory_runs.txt`를 잠시 덮어쓰는 검사이므로 활성 저장소에서 실행하지 않았다.
- 감사 스크립트의 적합성/fixture 검사 **56/56**, 운영 문제 재현 **10/10**, 초기값 pin 공백 확인 **1/1**. 문제 재현의 PASS는 결함이 재현됐다는 뜻이다. 56개 중 하나가 기존 단위검사 묶음의 성공을 확인하는 항목이다.
- 실제 WV3 학습 패치 2개, 배포 T0, 저장된 Student 4040-step checkpoint의 U 가중치를 이용해 CPU에서 실제 `KDVTrainer._step`을 호출했다. A는 T0에서 복사했다. 이미 학습된 U를 쓴 이유는 soft/edge→A gradient가 0인 자명한 검사에 그치지 않기 위해서다. 이것은 수식 검증용 상태이며 새 실험을 warm-start한 것이 아니다.
- LF는 24999/25000/25001 index의 세 update를 실제 AdamW로 계산했다. 25K 학습 완주나 checkpoint 재개를 수행한 검사는 아니다. scheduler의 전역 index와 실제 학습 루프 호출 위치도 대조했다.
- 셸 동작은 별도 임시 저장소에서 재현했다. 재배정 fixture에서 unit gate만 성공 stub으로 교체했고, 실제 generator/config 비교/`--dry-run`/`--confirm` 로직은 그대로 실행했다. 업로드는 Python 실행을 명령 기록 mock으로 바꿔 dispatch만 확인했다. runner 종료 검사는 감사가 만든 부모/자식 프로세스에만 수행했다.
- GPU 학습·원격 배포·활성 큐 변경·Sheet 변경은 수행하지 않았다. 구현 노트의 GPU smoke/시간 수치는 이번 감사에서 독립 재측정한 값으로 취급하지 않는다.

재현 자료:

- [검증 코드](PAKD50_Derived_Allocation_Audit_2026-09-15/verify_allocation.py)
- [검증 결과 JSON](PAKD50_Derived_Allocation_Audit_2026-09-15/verification.json), [전체 로그](PAKD50_Derived_Allocation_Audit_2026-09-15/verification.log), [기존 78개 검사 로그](PAKD50_Derived_Allocation_Audit_2026-09-15/existing_unit_suite.log)
- [코드·자산 SHA256와 검사 범위](PAKD50_Derived_Allocation_Audit_2026-09-15/provenance.json), [감사 시점 코드/설정 사본](PAKD50_Derived_Allocation_Audit_2026-09-15/snapshot/)

재실행 명령:

```bash
/home/knuvi/miniconda3/envs/pancrafter/bin/python -B \
  research_log/PAKD50_Derived_Allocation_Audit_2026-09-15/verify_allocation.py \
  /home/knuvi/Desktop/song/PAN-Crafter \
  /tmp/pakd50-derived-audit-recheck
```

초기 감사 harness에서 YAML `1e-4` 문자열을 float와 직접 비교해 설정 검사가 실패했다. 실제 `main.get_parser()`의 float 변환을 적용하도록 **감사 코드만** 수정하고 전체 검사를 재실행했다. 최종 결과는 모두 위 수치와 같으며, 첫 로그는 `initial_harness_type_check.log`에 보존했다. 이는 제품 설정 결함으로 세지 않았다.

**2. 계획대로 구현된 부분**

| 대상 | 확인 결과 |
|---|---|
| s2 seed777 | `JR → XJ → J_R3_NOEDGE → J_N0_EDGE`, 4개 설정/우선순위 일치 |
| s4 seed1234 | `F0 → RC0 → RCQ → JR → XJ → J_R3_NOEDGE`, 6개 일치 |
| s5 seed2026 | `PQ → F0 → LF0 → LFQ → RC0 → RCQ`, 6개 일치 |
| 공통 학습 | 16 YAML 모두 현재 생성기와 의미상 동일. W112/D123, 50K, batch48, U LR1e-4, 전역 cosine, eval_epoch5 일치 |
| 공통 계수 | R1/R3 `tau_R=0.012463942170143127`, R3 α1/β0.1, edge 사용 시 `lambda_E=0.09075170336956798` 유지 |
| JR | Teacher-error weighted GT만 사용. soft/edge 전체 제거 |
| XJ | weighted GT+GT edge. soft를 A/U 전체에서 제거 |
| J_R3_NOEDGE | weighted GT+adaptive soft. edge를 A/U 전체에서 제거 |
| J_N0_EDGE | plain GT+GT edge. Teacher 객체를 `None`으로 두어도 실제 step 정상. Teacher target 불필요 |
| RC0/RCQ | A-FT, I-NATIVE-TRANSFER, A LR1e-5, radius0, offset0. odd step에도 offset 연습·ε RNG 소비 없음. native 정합 및 reconstruction→A gradient 유지 |
| RCQ | Q12 hard/soft/edge 전부 A/U에 전달. soft→A와 edge→A가 실제로 0이 아님 |
| F0 | T0 A 동결 유지. AdamW 후 A hash 불변 |
| PQ | A gradient는 `∇(plain_GT+offset)`, U gradient는 Q12 전체와 일치. soft/edge 전체 제거 case와 구별됨 |
| LF0/LFQ | 24999에서 A/U 갱신. 25000/25001에서 A hash·AdamW step 상태 불변, offset 중단, U 갱신 지속. U LR은 약 5e-5에서 전역 cosine대로 감소 |
| native PAN 계약 | 10종 모두 Student reconstruction 입력이 원 PAN이며 학습 forward와 inference forward 일치 |
| T0/입력 보존 | 테스트 전후 Teacher hash 및 입력 tensor 불변 |

실제 loss와 직접 작성한 수식의 scalar 차이는 **0**이었다. A parameter gradient 최대 절대오차 **3.03e-8**, U **1.68e-8**이었다. PQ에서는 routing 후 gradient를 비교했다. 이는 해당 CPU 입력/상태에서의 검증이며 전체 학습의 수치 안정성이나 원격 GPU OOM 통과를 대신하지 않는다.

근거: [generator](../tools/gen_pakd50_configs.py:25), [설정 블록](../tools/gen_pakd50_configs.py:324), [실제 step](../train_kdv.py:760), [offset 경로](../train_kdv.py:920), [routing](../train_kdv.py:981), [LF 학습 루프](../train_kdv.py:1197).

fresh U 저장 초기값을 읽은 뒤 T0 donor A를 구성하는 순서도 유지된다([초기화](../train_kdv.py:106)). 16 YAML에는 phase/25K warm-start 지시가 없다. 다만 실패 후 자동 resume는 아래 D07의 별도 운영 문제다.

raw-original HQNR selector, exact50K 저장 및 최종 평가 경로도 유지된다([main](../main.py:346), [last 저장](../train_kdv.py:1270)). 현재 release의 평가 격자는 `GRID1010_50K_v1`이며, 고정 plateau 시점은 `[45450,46460,47470,48480,49490,50000]`이다([prepare manifest](../tools/pakd50_prepare.sh:105)). 실제 서버의 배치 수/완료 로그까지 확인한 것은 아니며, 일반 `plateau_report.py`의 기본 마지막 40%를 이 고정 plateau와 혼용해서는 안 된다.

초기 시간 계산도 맞는다:

| 항목 | 예약 시간 h |
|---|---:|
| s2 기본 4개 | 10.9186667 |
| s4 기본 6개 | 9.6900000 |
| s5 기본 6개 | 10.4270000 |
| 기본 합산 | 31.0356667 |
| 기본+확인 최대 합산 | 43.9156667 |
| 전체 기본 결과를 기다린 뒤 확인하는 병렬 경과시간 | 17.3586667 |

`1.10×ref+10/60`과 trainer `gate_hours×margin`의 연결은 맞다. 이 산술의 정확성과 실제 admission의 적합성은 구분해야 한다.

**3. 적용 전 해결할 P1 항목**

**D01 — 진행 중 runner 교체가 현재 run의 후처리/실패 처리를 잃는다.**

계획 §9.2는 현재 run을 유지하고 다음 경계에서 전환하도록 한다. 그러나 [requeue 30–34행](../tools/pakd50_requeue.sh:30)은 현재 `_run_cases.sh` 부모 bash를 즉시 종료한다. 학습 자식 `run.sh → main.py`는 남아 있어도 그 부모가 이후 실행해야 할 `eval_dlpan/_upload.sh`, rc 판정 및 `cases_failed.txt` 기록은 사라진다([runner](../tools/_run_cases.sh:109)).

새 큐에는 J0만 들어간다. 현재 JR/PQ가 정상 종료하면 새 gate가 완료된 run으로 제외하므로 그 run의 업로드를 다시 호출하지 않는다. 부모만 SIGTERM한 격리 bash 검사에서 **자식 완료=true, 후속 upload 실행=false**를 재현했다. `main.py` 안의 기본 .mat export까지 반드시 사라진다는 뜻은 아니다. 소실되는 것은 부모가 맡던 후처리/업로드/실패 처리다. 실패로 끝나는 경우에는 실패 원장 없이 같은 identity의 자동 재개로 이어질 수도 있다.

수정 완료 조건: 현재 `run_case`의 종료 판정과 후처리까지 마친 경계에서 새 목록을 읽게 하거나, 별도 완료/후처리 소유권을 안전하게 인계해야 한다. 학습 PID가 살아 있다는 검사만으로 인계 성공으로 판단하면 안 된다.

**D02 — `--dry-run`이 실행 제어 파일과 확인 실험을 실제 변경한다.**

[reallocate 19행](../tools/pakd50_reallocate.sh:19)에서 calibration을 쓰고, [66–67행](../tools/pakd50_reallocate.sh:66)에서 mandatory/reservations를 쓴 뒤 **71행에서야** dry-run 여부를 확인한다. 격리된 빈 서버에서 `--dry-run`만으로 세 파일이 생성됐다.

더 큰 문제는 `--confirm RCQ --dry-run`도 confirmation YAML과 `extra_priority.txt`를 실제 생성한다는 점이다. 이미 돌아가는 gate가 이를 읽으면 dry-run이 확인 실험을 등록하는 동작이 된다. 확인 모드에 들어가지 않는 dry-run도 기존 mandatory 목록을 바꿀 수 있다. 기존 unit K08의 임시 운영 파일 변경도 같은 검사 경로에 포함된다.

수정 완료 조건: dry-run의 모든 산출물은 임시 경로에서 계산하고, 활성 config/calibration/mandatory/reservations/extra 파일의 존재 여부·내용이 호출 전후 동일해야 한다.

**D03 — 확인 실험 예약 목록을 추가한 직후 지우며, WIN 고정과 총 3run 제한도 보장하지 않는다.**

`--confirm`은 [34–38행](../tools/pakd50_reallocate.sh:34)에서 확인 seed run을 mandatory에 추가하지만, 이후 `write_mandatory_file()`이 기본 seed 목록만 `w` 모드로 다시 쓴다([generator 86행](../tools/gen_pakd50_configs.py:86)).

`--confirm RCQ --dry-run` 후 s4 extra에는 `RCQ/RC0/F0@3407`이 있지만 mandatory에는 이 셋이 모두 없다. 따라서 trainer는 확인 run 자신의 비용은 보더라도 다음 확인 run들을 필수 잔여 묶음으로 예약하지 못한다. `reservations.json`에 항목이 존재하는 것만으로는 충분하지 않다. trainer가 비용을 합칠 run ID는 mandatory에서 읽기 때문이다.

이어 `--confirm LFQ --dry-run`을 호출하면 extra는 **RCQ/RC0/F0/LFQ/LF0@3407, 총 5개**가 된다. `confirmation_cases()`의 ≤3은 호출 한 번의 반환 길이만 검사한다. 누적 목록, 이미 고정한 WIN, config/계수/release 재작성은 차단하지 않는다.

수정 완료 조건: 기본/확인 예약을 합쳐 보존하고, 서버·확인 seed별 단일 WIN manifest에 release/계수/hash와 최종 ≤3개 identity를 고정해야 한다. 같은 요청 재호출은 멱등이어야 하며, 다른 WIN으로 바꾸는 요청은 이미 진행된 확인 결과와 섞이지 않아야 한다.

**D04 — 계획 §7–8의 전체 barrier와 전역 마감 admission이 없다.**

[campaign gate](../tools/campaign_gate.py:371)는 로컬 `hours_to_deadline`과 로컬 목록만 입력받는다. [schedule](../tools/gen_pakd50_configs.py:301)은 그 목록의 비용 누적만 검사한다. [trainer](../train_kdv.py:320)의 절대 마감 검사도 현재 run의 예상 종료만 검사한다. 다른 서버의 ready/기본 잔여 비용, 외부 WIN 대기, 확인 단계 예약을 연결한 `finish_all` 검사는 없다.

재현: s5 로컬 기본 결과가 있고 확인 목록을 등록한 상태, 학습 마감까지 7h이면 세 확인 run(6.44h)을 모두 emit한다. 다른 서버의 기본 작업이 아직 2h 남았다면 계획상 최소 `2+6.44=8.44h`가 필요해 부적합하다. 해당 원격 상태를 입력하거나 확인하는 경로 자체가 없다. `--confirm` 수동 호출이 WIN 전달 수단이라는 점은 인정하지만, 이것이 전역 barrier/종료 예측의 검증 기록까지 대신하지는 않는다.

확인 목록 등록 후에도 `--plan`은 기본 6개만 출력한다([plan_table](../tools/gen_pakd50_configs.py:414)). 또한 `plan_rows` 기본 완료 판정은 실행 중 PID를 보지 않아 실행 중 PQ를 full reservation으로 세며, 직접 호출한 gate는 실행 중 run을 아예 제외해 remaining을 예약하지 않는다. runner의 대기는 중복 학습 방지에는 도움이 되지만 전환 시점의 정확한 종료 시각 산정은 아니다.

수정 완료 조건: 기본 단계 전체 결과 barrier, 서버별 실제 가용 시각·진행 중 중복/비중복 run의 잔여 비용·후처리·WIN 대기·확인 묶음을 포함하는 단일 일정 상태와 admission 증거가 필요하다. 고정 training deadline **2026-09-16 11:22:31 KST** 뒤의 4h를 여기서 다시 빼면 안 된다. 본 감사는 현재 원격 실험이 실제로 마감을 넘긴다고 판정한 것이 아니다.

**D05 — 제공된 배포 지침이 학습 중 checkout을 pull하지 말라는 요구와 충돌한다.**

[구현 노트 49–55행](2026-09-15_pakd50-derived-allocation-implementation.md:49)은 `git pull` 직후 dry-run/reallocate를 안내하면서 기존 s2 JR/s5 PQ는 계속 실행한다고 설명한다. 이는 계획 §9.4와 맞지 않는다. 현재 스크립트에는 학습 run의 checkout을 격리하거나, 모든 학습/후처리가 끝난 뒤 release를 활성화하는 절차가 없다. 이미 import된 코드가 있다는 이유로 이후 import, export, evaluator, 업로드, 재개까지 같은 release라고 증명할 수 없다.

수정 완료 조건: 별도 release checkout/worktree에서 검증하고 run 경계에서 활성화하거나, 현재 run의 후처리 완료 후 checkout을 바꾸도록 인계 지침과 구현을 일치시켜야 한다. 실제 원격에서 이미 부적절한 pull이 일어났다고 단정하지는 않는다.

**D06 — 기본 업로드가 제외 대상 재평가/진단까지 수행하고, PAKD50 후처리 실측을 다음 비용에 반영하지 않는다.**

[업로드 38–39행](../tools/_upload.sh:38)의 `PAKD50_*` 분기는 항상 `pa_diag.py`를 호출한다. 실제 셸을 mock Python과 실행했을 때 `eval_fr_paperset → pa_diag → gspread_upload` 순서가 기록됐다([dispatch 로그](PAKD50_Derived_Allocation_Audit_2026-09-15/upload_mock_commands.log)).

[pa_diag](../tools/pa_diag.py:187)는 PAKD50 A-FT/A-FR을 정합 모델로 취급해 selected/last 등 여러 checkpoint의 FR/RR 재추론, learned/zero/wrong-sign, 25개 known-shift 반응, tile/full 및 독립 정합 진단을 수행한다. 이것은 별도 A/U 모델 교환 실험과 동일한 작업은 아니지만, 이번 계획 §1/§6에서 학습 서버에 한정한 기본 export/gspread 범위보다 넓다.

PAKD50에는 이 전체 후처리 경과시간을 기록·읽어 다음 예약에 반영하는 경로가 없다. 현재 비용식은 항상 `RESERVE_POST_H=10/60`이고, gate는 run ledger 시간만 읽는다. 구현 노트의 ‘후처리 초과 시 큐 갱신은 요청 시’도 계획의 즉시 갱신 요구와 다르다. 또한 gate가 한 pass에서 전체 목록과 예약 파일을 만들어 runner가 끝까지 소비하므로, 첫 실제 완주 비용 반영도 다음 run마다 이루어지지는 않는다.

수정 완료 조건: 이번 캠페인의 업로드 경로를 요구된 기본 export/기존 평가/Sheet 저장 범위로 제한하고, 필요한 후처리는 전체 경과시간을 측정해 다음 run admission 직전에 반영해야 한다. 기존 학습 중 정기 평가를 이 시간에 다시 통째로 더하지 않는다.

**4. 함께 닫아야 할 P2 항목과 배포 증거 공백**

**D07 — 완료/실패/재개 identity 검증이 계획보다 약하다.**

[gate.complete](../tools/campaign_gate.py:39)와 [runner.complete](../tools/_run_cases.sh:49)는 best_hqnr의 `.mat` 두 파일 존재만 본다. 두 빈 파일만 만들고 `last_meta`, 실제 50K, 정상 exit 기록이 없는 fixture도 complete=True였다. 실제 `main.py`는 학습 뒤 export하지만, best_hqnr export 후 뒤쪽 tag export에서 실패할 수 있고 `_finish_ledger(FINISHED_EXPORT)`도 각 tag 뒤에 찍힌다([trainer](../train_kdv.py:1478)). 따라서 파일/`FINISHED*` 어느 하나만으로 전체 정상 완료를 확정할 수 없다. 정상 runner가 살아 있으면 rc로 추가 판정하지만 D01에서는 그 주체가 사라진다.

또한 runner는 checkpoint가 있으면 같은 이름으로 자동 resume하고 실패하면 같은 이름으로 fresh retry한다([runner 84행](../tools/_run_cases.sh:84)). 코드가 `resumed_nonexact=True` 및 sampler 순서 미복원을 기록하는 점은 개선됐지만([trainer 638행](../train_kdv.py:638)), ‘fresh50 기본’과 ‘실패 PQ는 원인 수정 version으로 비용 별도 기록’까지 강제하지는 않는다. 구현 노트의 실패 원장 줄 삭제 후 requeue 안내도 새 version을 만들지 않는다.

완료는 정상 exit와 필수 export/last50K를 함께 확인해야 한다. 실패는 완료와 구별해 일정의 미충족 항목으로 남겨야 하며, 재시작/새 version/비정확 resume를 분석 control에 섞지 않도록 해야 한다.

**D08 — 이전 extra 꼬리가 남을 수 있고, 비용 부족 시 명시 순서가 바뀐다.**

reallocate는 기존 `extra_priority.txt`를 교체/검증하지 않는다. 이전 s5의 `DPQ`가 남아 있으면 새 기본 6개 뒤에도 DPQ가 편성되는 것을 재현했다. 이전 scalar/gradient 관련 후보가 있으면 계획의 확정 16run 범위를 넘어갈 수 있다. 실제 원격 extra 내용은 확인하지 않았으므로 현재 발생 사실로 단정하지 않는다.

`schedule()`은 비싼 앞 run을 drop한 뒤 뒤의 싼 run을 계속 넣는다. s5 남은 1.8h에서는 PQ를 밀고 F0를 실행했다. 계획에 없는 부분 배정/우선순위 변경을 묵시적으로 한다. 명시 순서를 유지하고, 전체 묶음이 맞지 않으면 그 사실을 admission 결과로 드러내야 한다.

s1/s3에는 재배정 명령을 거부하는 guard가 있지만, 일반 `gate_pakd50()`는 여전히 예전 기본 순서를 편성할 수 있다. 이번 배포에서 이 두 서버에 신규 작업을 주지 않는다는 별도의 운영 상태도 확인할 필요가 있다. 본 감사는 이 서버들의 체인/cron을 변경하지 않았다.

**G01 — 같은 서버 control 및 초기값의 확정 hash 증거가 부족하다.**

T0 파일 SHA와 selected step은 pin되어 있다. seed1234의 U hash도 `c988a6c95b17f4fd`로 pin된다. 그러나 [init_hashes](../assets/pakd50/init_hashes.json)는 **777/2026/3407/9091이 null**이며 해당 YAML에는 `expect_init`이 없다. 이는 s2/s5의 실제 초기값이 틀렸다는 뜻이 아니라, 분석 담당자가 확정한 초기값과 같다는 fail-fast 검증이 아직 없다는 뜻이다. 기본 16개 중 s2 4개+s5 6개가 이 공백에 해당한다.

`baseline_run`도 [generator 345행](../tools/gen_pakd50_configs.py:345)에서 동일 version의 이름을 자동 구성할 뿐 control checkpoint hash를 고정하지 않는다. s4 J0 v1/v2 중 분석 담당자가 선택한 release/자산의 확정 기록은 이번 제공 파일만으로 검증하지 못했다. 같은 seed를 공유하는 s1의 완료 결과가 s4에서 실제 수행됐다는 증거도 되지 않는다.

확인 WIN manifest에는 같은 서버 control version/hash, U 초기값 hash, T0/calibration hash가 포함되어야 한다. seed별 첫 초기값을 새로 만드는 경우에도 그 파일을 저장·고정한 뒤 해당 seed의 모든 비교에 재사용해야 한다.

**5. 적용 판정의 경계**

새 case의 수식 자체를 재설계할 이유는 이번 검증에서 발견하지 못했다. 새 학습을 적용하기 위한 주된 수정 대상은 **안전한 run 경계 전환, 읽기 전용 dry-run, 확인 묶음 고정/예약, 전역 마감 검사, 기본 후처리 범위/실측 갱신**이다.

현재 원격 진행 중 run을 중단하거나 실패로 판정할 근거는 없다. 원격별 실제 run identity/step/완료 exit·필수 export, 예정 release, 기존 extra 목록, 초기값/control hash를 확인하고 위 결함을 닫은 뒤 이 계획의 운영 적용을 승인할 수 있다. 이번 감사로 생성한 파일은 본 보고서와 증거 디렉터리뿐이며, 운영 코드·config·활성 원장·큐·Sheet는 수정하지 않았다.
