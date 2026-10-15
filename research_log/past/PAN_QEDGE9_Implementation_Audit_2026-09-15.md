**QEDGE9 구현 감사 — 2026-09-15**

**판정: 핵심 수식·현재 cue·기본 9개 config는 계획과 일치한다. 실행 전환, 자산 식별, 재개, 완료 판정에는 계획을 충족하지 못하는 결함이 있어 적용 준비 완료로 판정하지 않는다.** 현재 자산이 잘못 생성됐다는 결론은 아니다. 정상 입력에서의 수치 검증과 잘못된 입력·중단 상황에서의 검증을 구분했다.

기준 문서는 [QEDGE9 실험 계획](/home/knuvi/Desktop/song/PAN-Crafter/research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md), 구현은 `433ad67f4ae0da7516e3c1dcaa6845af1ec1e837`(2026-09-15 16:09:09 KST)이다. 부모 commit은 `1a3edb9`이다. 감사 중 로컬 `origin/main`도 `433ad67`로 갱신됐으며 reflog에 16:23:48 `update by push`가 있다. 이는 로컬 push 기록이고, s4/s5가 이 commit을 pull해 실제 실행 중이라는 증거는 아니다.

계획·구현 노트·변경 코드·기존 trainer/runner의 연결을 읽고 46개 기준 파일의 SHA256과 사본을 보존했다. [감사 기준 파일 목록](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/provenance.json)에 범위를 명시했다. 운영 코드와 config를 수정하지 않았고, 실제 학습·큐·cron·Sheet를 변경하지 않았다. CPU 계산, 임시 디렉터리, 감사 전용 모의 프로세스로 검증했다.

16:45:05 KST 종료 검사에서 기준 파일 46개가 모두 최초 SHA와 일치했고 tracked worktree 변경은 없었다. [최종 검사 집계](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/final_summary.json)에 commit·검사 수·원본 안정성 확인을 남겼다.

**우선 수정할 사항은 아래와 같다.** P1은 계획의 실험 동일성이나 필수 실행·후처리를 깨뜨릴 수 있는 문제, P2는 특정 운영 상태에서 편성·시간 정책을 어기는 문제다. 이미 현장에서 발생했다는 뜻은 아니며, 발생 조건과 재현 범위를 각 항목에 적었다.

| ID | 우선순위 | 확인한 문제 | 직접 영향 |
|---|---|---|---|
| F01 | P1 | 전환 시 현재 runner를 종료하고, dry-run도 운영 파일을 기록 | 현재 run 후처리 누락 및 검토 단계의 큐 상태 변경 |
| F02 | P1 | QEC pilot의 case·seed·depth·50K 식별을 완전하게 검사하지 않음 | 계획과 다른 상수 대조군을 QEC로 실행 가능 |
| F03 | P1 | cue의 의미·고정 hash·QES permutation 검증 부족 | 잘못된 gate 또는 재개 중 변경된 gate를 수용 가능 |
| F04 | P1 | 다음 batch까지 동일한 resume 미구현 | 중단된 paired run이 연속 실행과 다른 학습 경로로 재개 |
| F05 | P1 | 미준비 QEC를 건너뛰고 pending만 남아도 runner가 DONE | QES 선행 및 QEC 영구 미실행 가능 |
| F06 | P1 | 두 결과 파일 존재만으로 완료·재사용 판정 | 50K·평가 후보·control 동치 미확인 상태로 대조군 재사용 가능 |
| F07 | P2 | runner에 72시간 시작 마감이 남고 실측 조회는 옛 ledger 사용 | 무마감 정책 불완전, 새 run 실측의 예약 반영 실패 |
| F08 | P2 | 이전 extra_priority가 새 기본 묶음 뒤에 남음 | 계획에서 교체한 저우선순위 확장의 추가 실행 가능 |

**F01 — 전환은 현재 run의 정상 종료 경계를 보장하지 않는다.** 계획 §0.1·§1.3은 현재 run을 끝까지 진행한 뒤 전환하도록 요구한다. [qedge9_switch.sh:63](/home/knuvi/Desktop/song/PAN-Crafter/tools/qedge9_switch.sh:63)는 기존 `_run_cases.sh` PID를 찾고 65행에서 `kill`한다. 학습 Python을 직접 죽이지 않는 것은 맞지만, 기존 부모 runner가 돌아와 수행할 완료/실패 처리, 공식 평가, 진단·업로드가 사라질 수 있다. 그 후처리는 [_run_cases.sh:111](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:111)에 있다. 새 runner가 시작 시 확인하는 것은 학습 Python의 존재이며, 전체 후처리의 완료가 아니다.

감사 전용 부모/자식 프로세스만 보이는 `ps`와 임시 start/watchdog/cron을 사용해 실제 switch 스크립트를 실행했다. switch는 rc=0, 부모는 SIGTERM으로 rc=-15, 자식은 정상 종료했지만 부모의 후처리 marker는 생성되지 않았다. 특히 새 우선순위에 없는 이전 run이면 새 runner가 그 업로드를 대신 수행한다는 보장도 없다. 이는 실제 훈련 프로세스를 대상으로 한 테스트가 아니다. [전환 재현 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/owned_handoff.log)

또한 `--dry-run` 분기는 [59행](/home/knuvi/Desktop/song/PAN-Crafter/tools/qedge9_switch.sh:59)에 있지만, 17~58행에서 디렉터리·검증 로그·mandatory·reservations를 이미 기록한다. s4에서는 기존 pilot이 있으면 c_E도 다시 산출한다. 격리된 s4/s5 양쪽에서 `work_dir/_pakd50/mandatory_runs.txt`, `_qedge9/mandatory_runs.txt`, `_pakd50/reservations.json`의 기록을 확인했다. 현재 runner가 gate를 다시 읽으면 검토 명령만으로도 다음 편성이 바뀔 수 있다. 구현 노트가 dry-run에서 pilot 산출을 안내하더라도 실제 운영 목록 변경까지 안전한 사전 검토로 간주할 수는 없다.

수정 기준: 현재 run의 학습·결과 export·진단·업로드가 끝난 경계에서 새 편성으로 넘기는 인계 상태를 둔다. dry-run의 출력은 임시 경로에서 생성하고 운영 파일은 apply 단계에서 기록한다. 인계 테스트에서는 자식 완료뿐 아니라 기존 run의 후처리 완료와 새 run 시작 순서를 함께 검사해야 한다.

**F02 — QEC의 계산식은 맞지만 pilot 식별 검사가 불충분하다.** 계획 §6.1의 pilot은 **s4 W104D121 J0 S1234 exact50K**이다. [qedge9_cue.py:163](/home/knuvi/Desktop/song/PAN-Crafter/tools/qedge9_cue.py:163)는 tag가 `last`일 때만 step=50000을 확인하고, 모델을 읽은 뒤에는 width=104만 검사한다. case, seed, depth는 강제하지 않고 `best_hqnr` 같은 다른 tag는 50K 검사를 우회한다.

축소된 calibration fixture에서 실제 W104D123 모델과 실제 데이터 2개 patch × 4 view를 사용했다. 로더가 `QE50 / S777 / D123 / best_hqnr / step1010`이라는 잘못된 pilot을 제공하도록 제한했는데도 실제 `pilot()`이 c_E를 계산해 파일을 기록했다. 모델 로더와 calibration 모집단은 검증 조건을 분리하기 위해 대체한 것이며, 이 fixture의 c_E는 현장 QEC 값이 아니다. [pilot 검사 증거](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/pilot_guard.log)

학습 측 [EdgeGate.load:167](/home/knuvi/Desktop/song/PAN-Crafter/kdv/edge_gate.py:167)도 c_E의 유한성·범위와 **기록돼 있을 때만** cue NPZ hash를 확인한다. 잘못된 pilot identity가 적힌 파일도 수용했다. 따라서 switch의 기본 호출이 올바른 run/tag를 지정한다는 사실만으로, 복사·교체된 c_E 파일까지 검증됐다고 볼 수 없다. switch를 반복할 때 기존 c_E를 매번 다시 산출하는 동작도 §6.1의 산출 후 고정 요구를 보장하지 않는다.

수정 기준: 산출 및 학습 양쪽에서 pilot case/seed/전체 구조/tag/step/full checkpoint hash, 데이터·calibration view·cue hash를 필수 값으로 대조한다. 이미 고정된 c_E는 동일 출처인지 검증하고 재산출로 덮어쓰지 않는다. 현재 로컬에는 s4의 실제 pilot와 `qec_cE.json`이 없어, 그 자산의 실제 값과 출처는 이번 감사에서 확인하지 못했다.

**F03 — 현재 cue는 맞지만 잘못된 cue를 차단하는 검사가 충분하지 않다.** 계획 §7.1의 key와 §8.2의 resume 검사는 단순히 NPZ와 그 옆 JSON이 서로 일치하는 것보다 강한 조건이다. [edge_gate.py:113](/home/knuvi/Desktop/song/PAN-Crafter/kdv/edge_gate.py:113)는 JSON에 적힌 NPZ hash만 확인한다. 이어지는 `load()`는 Teacher A weight hash, train H5 hash, 네 augmentation flag를 대조하지만, margin·전처리/보간 구현·dtype·bank·theta와 gate의 관계·QES permutation의 고정성을 충분히 확인하지 않는다. 특히 QES의 e에는 Teacher U도 관여한다.

다음 부정 입력이 실제 로더에서 수용됐다. 현재 원본 자산은 바꾸지 않았고 임시 사본에서 검사했다.

| 입력 조건 | 결과 |
|---|---|
| theta가 없거나 `1e9`인데 gate 배열은 기존 값 | 로드 성공; theta 없음은 별도 campaign gate가 막지만 직접 trainer 로더는 수용 |
| 다른 bank ID/K가 manifest에 기록 | 로드 성공 |
| 같은 A weights에 margin=0, cue는 margin=4 | 로드 성공 |
| QES gate를 seed=7로 재생성, config는 51515 | 로드 성공 |
| **같은 config·같은 JSON/NPZ 경로**에서 gate를 반전하고 JSON의 자체 NPZ hash 갱신 | 재로드 성공, 반대 gate 사용 |

마지막 경우는 run/checkpoint에 기대 cue hash가 고정돼 있지 않음을 보여 준다. Trainer는 [train_kdv.py:566](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:566)에서 현재 경로의 자산을 읽고 resolved manifest를 다시 기록한 뒤, `main.py`가 accelerator state를 읽는다. 이전 checkpoint의 cue/permutation/c_E와 대조하는 단계가 없다.

온라인 `verify`도 [qedge9_cue.py:152](/home/knuvi/Desktop/song/PAN-Crafter/tools/qedge9_cue.py:152)에서 e 차이와 label flip을 **출력만 하고 합격 조건에서는 제외**한다. 실제 계산 함수를 통제한 부정 fixture에서 `Δq=0`, `Δe≈0.1`, threshold와 맞지 않는 label 8/16개가 나와도 `pass_=true`, rc=0이었다. 이는 현재 cache가 오염됐다는 증거가 아니라 검사 공백의 재현이다. 계획은 GPU 수치 차이로 생길 수 있는 threshold 인근 label 변화를 기록하도록 한다. 따라서 모든 미세한 flip을 무조건 실패시켜야 한다는 뜻은 아니다. theta·저장 q·저장 label의 내부 모순과 큰 e 차이는 별도 검사로 거부해야 한다.

수정 기준: q 정의·전처리·dtype·Teacher/A·데이터·calibration IDs·bank·theta·QES assignment를 포함한 불변 asset ID를 고정한다. 시작 시 의미 일치와 coverage를 확인하고, 재개 시 checkpoint의 기존 ID와 대조한다. QES e의 출처와 c_E 파일도 같은 방식으로 묶는다. [부정 검증 결과](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/verify_negative_evidence.json)

**F04 — 계획이 요구하는 exact resume은 구현되지 않았다.** 계획 [§8.2의 Resume 항목](/home/knuvi/Desktop/song/PAN-Crafter/research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md:381)은 cache/hash/permutation뿐 아니라 다음 batch/update 경로의 동일성까지 요구한다. [main.py:235](/home/knuvi/Desktop/song/PAN-Crafter/main.py:235)는 dataloader가 accelerator에 준비되지 않아 배치 순서를 복원하지 못하는 근사 재개라고 명시한다. [train_kdv.py:634](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:634)도 `resumed_nonexact=true`를 기록한다.

동일한 RandomSampler/DataLoader 구성에서 RNG만 복원한 재현 결과, 연속 실행의 다음 index는 `[94,18,65,77,83,123,82,37]`, 재개 후에는 `[116,4,58,108,53,69,117,33]`이었다. sampler permutation/cursor와 worker augmentation 상태가 복구되지 않기 때문이다. 이는 실제 50K checkpoint를 재개한 테스트는 아니며, 코드가 사용하는 방식의 next-batch 불일치 재현이다. 기존 단위검사의 optimizer/epsilon RNG 복구만으로 이 조건을 통과했다고 볼 수 없다.

[_run_cases.sh:84](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:84)는 중단된 run을 자동 재개한다. 따라서 중단이 발생하면 계획의 대응 실험 동일성이 저절로 유지되지 않는다. 처음부터 끝까지 중단 없이 수행한 같은 구조의 fresh run에 이 결함이 적용되는 것은 아니다.

수정 기준: sampler 진행 위치와 augmentation/epsilon RNG 및 cue ID를 포함한 실제 checkpoint-resume 비교를 통과시킨다. 그 전에는 근사 재개 결과를 연속 실행과 동일한 검증 완료 paired run으로 처리하면 안 된다.

**F05 — c_E 지연을 기다리지 않고 QEC 없이 종료할 수 있다.** 계획 §6은 s4 `QE50 → QEC → QES` 순서를 제시하고 §11.3은 c_E 미산출 run의 대기를 요구한다. [campaign_gate.py:387](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_gate.py:387)의 blocked 항목은 편성에서 빠지며 뒤 항목은 계속 열린다. c_E만 없는 조건에서 실제 schedule 출력은 `[QE50, QES]`였다.

QEC만 미완료·미준비로 남으면 gate 출력은 비게 된다. [_run_cases.sh:135](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:135)는 그 상태에서 loop를 종료하고 154행의 DONE을 기록한다. [_watchdog.sh:16](/home/knuvi/Desktop/song/PAN-Crafter/tools/_watchdog.sh:16)은 DONE이면 재시작하지 않는다. [qedge9_switch.sh:26](/home/knuvi/Desktop/song/PAN-Crafter/tools/qedge9_switch.sh:26)의 `--pilot`은 c_E 생성 후 종료하므로 그 뒤에 자동으로 생길 “다음 pass”가 보장되지 않는다. schedule의 빈 출력은 동적 검사, runner/watchdog/--pilot의 연결은 코드 추적으로 확인했다.

수정 기준: `WAITING_FOR_CUE`와 모든 필수 실행 완료를 구분한다. QEC가 미준비이면 지정 순서를 유지하거나 순서 변경을 명시적으로 기록하고, mandatory가 pending인 동안 DONE을 성공 종료로 쓰지 않는다. 준비 완료 이벤트 또는 watchdog이 다시 편성하도록 연결한다.

**F06 — 완료 control의 50K·출처 검증 없이 재사용할 수 있다.** 계획 §8.2·§8.3·§11.1은 완료 marker 외에 50K state·평가 후보·동일 Teacher/init/data/loss/evaluator와 기존 자산 identity를 확인하도록 한다. [campaign_gate.py:39](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_gate.py:39)와 [_run_cases.sh:49](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:49)는 두 `*_best_hqnr.mat` 파일의 존재만 확인한다. switch는 해당 run을 terminal로 분류하면 [40~42행](/home/knuvi/Desktop/song/PAN-Crafter/tools/qedge9_switch.sh:40)에서 config 검사도 생략한다.

임시 디렉터리에 두 빈 marker, step=1010인 last_meta, 다른 init hash를 두고 checkpoint와 평가 후보는 만들지 않았다. 실제 `complete()`와 `terminal()`이 모두 True였다. 현재 s4 결과 파일이 불완전하다는 주장은 아니다. 이미 존재하는 잘못된 run 디렉터리가 들어왔을 때 재사용을 거부할 검사 자체가 없다는 증거다. [완료 판정 재현](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/completion_verification.json)

수정 기준: s4 J0/JQ/XJ/F0 재사용 및 신규 run 완료 시 50K step과 state, selected/last/고정 평가 grid, 데이터·Teacher·init·loss·평가기의 identity를 확인한다. `FAILED`와 검증된 `COMPLETE`도 실험 묶음의 완결성 판단에서 구분해야 한다.

**F07 — soft time policy가 runner 전체에 반영되지 않았다.** 신규 config의 자체 ledger·required·deadline 미상속은 맞다. 하지만 [qedge9_switch.sh:67](/home/knuvi/Desktop/song/PAN-Crafter/tools/qedge9_switch.sh:67)의 `--hours 72`는 [campaign_start.sh:47](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_start.sh:47)에서 실제 `cases_deadline.txt`를 만든다. runner는 [68행](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:68)에서 그 이후의 새 run 시작을 거부한다. 이것은 진행 중 학습을 72시간에 강제 종료하는 동작은 아니지만, 계획의 `hard_deadline: null`과도 다르다. 기본 약 10시간 일정에서는 대개 드러나지 않고 긴 준비 지연·중단·연장 때 발현한다.

또한 QEDGE9 실측은 `_qedge9_budget/ledger.json`에 저장하지만 [measured_hours_from_ledger:293](/home/knuvi/Desktop/song/PAN-Crafter/tools/gen_pakd50_configs.py:293)의 기본 입력은 옛 PAKD50 ledger이다. [campaign_gate.py:376](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_gate.py:376)도 옛 ledger를 명시적으로 전달하고 switch 역시 기본 조회를 사용한다. QEDGE9 ledger에 QE50 완료 시간 2.75h를 둔 fixture에서 자동 조회 결과는 `{}`, 예약은 계속 reference 1.50h였다. 같은 함수를 QEDGE9 ledger와 직접 호출하면 2.75h를 정상 읽었다. 즉 키 파싱보다 조회원 연결의 문제다.

수정 기준: QEDGE9는 실제 runner의 시간 admission에서도 무마감으로 취급하고, 신규 ledger의 같은 서버·case·구조 실측을 예약 계산에 반영한다. 과거 ledger와 clock은 보존한다.

**F08 — 기존 extra_priority를 보존한 채 새 목록 뒤에 실행한다.** 계획 §1.3의 s5 역할은 완료한 이전 묶음 뒤 저우선순위 확장을 신규 queue로 교체하는 것이다. switch는 기본 목록과 mandatory는 새로 쓰지만 `work_dir/_pakd50/extra_priority.txt`는 그대로 둔다. [campaign_gate.py:380](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_gate.py:380)는 그 파일을 계속 읽어 새 기본 목록 뒤에 붙인다.

이전 `LFQ W112D123 S9091`을 extra에 둔 격리 fixture에서 switch 이후에도 항목이 남았고 schedule은 QEDGE9 기본 6개와 이 항목을 합쳐 7개를 반환했다. 예산·완료 상태에 따라 실제 시작은 달라지지만, 새 계획의 선택적 `W104 J0/JQ/QE50 S9091` 묶음과는 다른 실행이 열리는 경로다. 수정 기준은 이전 extra를 보존용 파일로 분리하고 새 branch에 적용할 목록을 명시하는 것이다. 완료 자산·실패 원장을 삭제할 필요는 없다.

**정상 구현으로 확인한 범위도 명확하다.** 다음 결과는 코드를 읽은 것 외에 unit, 실제 cue/data, 실제 trainer step을 함께 확인한 것이다.

| 계획 요구 | 확인 결과 |
|---|---|
| q 정의 | 고정 T0 A, 16개 axis probe, `mean(probe × 2 좌표의 L1)` 일치. 반응 없는 aligner의 q=0.46875. 독립 합산과 최대 차이 `5.82e-9`. λoff/EPE/epsilon norm으로 변경되지 않음 |
| Teacher 입력 | 실제 T0 full checkpoint SHA와 A hash, W112D123 own-config load, margin4를 확인. 추가 q probe는 A를 호출하며 Teacher U를 16회 반복하지 않음 |
| GT hard/soft | 기존 R3/Q12 full-domain 식과 detach 유지. low-e에서 hard를 지우지 않음. ROI32는 QES e 진단용 |
| GT edge | 같은 band의 signed Scharr·내부1px. 기존 edge 항을 `lambda_E * mean(g_i E_i)`로 교체. 활성 sample 수로 재정규화하지 않음 |
| gradient | 실제 QE50/QES/QEC trainer `_step`의 total·U·A gradient가 독립 계산과 일치. 비선택 edge gradient 0, hard/soft 유지, offset은 A 경로, Teacher 불변 |
| QE50 gate | calibration 3072 base × 4 view의 median 고정, `q < theta` strict 비교. minibatch top-k가 아님 |
| QES | calibration e decile × 실제 rot의 40 stratum, seed51515 dataset-wide shuffle을 전체 배열에서 재현. 각 active count 보존 |
| feeder | 기존 고정 hflip/vflip와 random rotation 4상태 유지. `return_meta` 전후 tensor/RNG 동일성 검사를 통과 |
| cue 준비의 RNG | cache load가 Python/NumPy/Torch 학습 RNG를 바꾸지 않는 것을 확인. 이 결과는 F04의 resume 동일성과 별개 |
| 신규 config 9개 | 생성기와 의미 일치. s5 2026/777의 J0/JQ/QE50, s4 1234의 QE50/QEC/QES. W104D121·50K·batch48·warmup100·U/A LR·fixed tau/lambda·평가 grid 확인 |
| 초기화 구조 | U는 구조별 namespace의 fresh init 공유, A는 T0 donor strict load, Teacher bridge 유지. 현재 비교는 같은 W104 구조이므로 과거 cross-architecture RNG 문제를 이 비교의 새 결함으로 판정하지 않음 |
| 필수 실행 예약 | switch가 QEDGE9 전용 mandatory를 실제 생성하며 s5 6개/s4 3개를 모두 포함. 이 기본 묶음에서 확인 seed 예약 누락이 재현된 것은 아님 |
| 기본 편성·시간 | 준비된 경우 지정 순서 일치. s5 10.262h, s4 기존 잔여 3개가 모두 남은 경우 10.163h, s4 신규 3개 5.340h. 합 20.425 서버·시간, 병렬 예약 약 10시간16분 |
| branch budget | QEDGE9 자체 ledger, soft9h, required=true, 부모 절대 마감 없음. 부모 마감이 지난 fixture에서도 신규 기본 run의 campaign admission 통과. runner 문제는 F07 |
| 기록·추론 | campaign/branch·edge_gate resolved metadata, gspread X열 QEDGE9/Notes 경로 연결 확인. 평가·추론은 Student A/U를 사용하고 q/Teacher/GT gate를 요구하지 않음. 실제 원격 Sheet 업로드는 검사하지 않음 |

QEC trainer 수식 검사는 의도적으로 주입한 `c_E=0.37`로 수행했다. 실제 s4 pilot에서 고정한 c_E 검증을 대체하지 않는다. gradient는 zero-output 초기화로 첫 A gradient가 사라지는 현상을 피하기 위해 비퇴화 weight를 사용했다.

**현재 cue 자산에 대한 실측은 다음과 같다.** 전체 자산의 배열·calibration·shuffle을 재계산했고, 온라인 모델 forward는 고정 부분집합에서 수행했다.

| 항목 | 값 |
|---|---|
| Train base patch / 실제 augmentation state | 9,714 / 4 |
| 전체 cache view / calibration view | 38,856 / 12,288 |
| Calibration base index hash16 | `02122e2554783774` |
| theta_q | `0.3276133416220546` |
| 선택 비율: calibration / 전체 | `0.5` / `0.49961395923409513` |
| QES stratum / shuffle 후 label 변경 비율 | 40 / `0.3672534486308421` |
| Cache NPZ SHA256 | `b83cc5d1a25e8766d1813c9d19d956efc24e0f152553e53c4894fa18dcdf563e` |
| 실제 T0 checkpoint SHA256 | `16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32` |
| Train H5 SHA256 | `f9e7ec381390ff36ddffd0714df7af07a3c5f456078b219a25f3e18372228f82` |
| Train PAN H5 SHA256 | `3ce697e7d57b04a02175b2f393fb762c897ed47c1a1e11e7f1a27018be5a5628` |
| CPU 온라인 재계산 vs 저장 GPU cache: 384 view | max Δq `1.45617e-5`, max Δe `2.15322e-5`, label flip **0** |
| threshold 근처 `abs(q-theta)<1e-3` | 21/384 |

384-view 비교는 q CLI 허용치 `1e-4` 이내이고, 감사에서 추가로 확인한 e도 `1e-4` 이내이며 label은 모두 일치했다. 첫 추가 검사에는 감사자가 e에만 `1e-5`를 적용해 1건 실패했다. 이를 운영 기본 q tolerance와 같은 `1e-4` 및 label 일치 조건으로 정리한 뒤 실제 재계산을 다시 수행했다. 초기 실패 로그도 `extra_verification.initial.*`로 보존했다. 따라서 최초 검사부터 전부 통과했다고 보고하지 않는다. 서버별 GPU 검증을 CPU 비교로 대신 완료했다고 간주하지도 않는다.

**검사 산출물과 해석.** 기존 단위검사는 **110/110 통과**, 주 감사의 적합성 검사는 **33/33 통과**, 추가 실데이터 적합성 검사는 **1/1 통과**했다. 주 감사 15개·추가 4개·완료 판정 1개의 부정/경계 시나리오도 재현됐다. 이 20개의 `defect_reproduction: passed=true`는 제품이 안전하다는 뜻이 아니라 **설명한 결함 조건이 관찰됐다는 뜻**이다. 중복 조건을 묶어 F01~F08로 보고했다. 33개 적합성 검사 중 1개가 기존 110개 suite 실행이므로 이를 독립 검사 144개라고 합산하지 않는다.

- [주 검사 코드](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/verify_qedge9.py), [결과 JSON](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/verification.json), [기존 unit 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/existing_units.log)
- [온라인·pilot·전환 추가 검사 코드](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/verify_extra.py), [추가 결과 JSON](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/extra_verification.json), [CPU 온라인 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/online_cpu.log)
- [완료 판정 검사 코드](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/verify_completion.py), [s4 dry-run 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/switch_s4.log), [s5 dry-run 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15/switch_s5.log)

재실행은 기준 commit과 실제 data/T0 자산이 같은 checkout에서 아래와 같이 할 수 있다. 스크립트는 자체적으로 임시 운영 경로와 CPU만 사용한다. `verify_extra.py`는 감사 전용 자식 프로세스를 생성·종료한다.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/knuvi/miniconda3/envs/pancrafter/bin/python \
  research_log/QEDGE9_Implementation_Audit_2026-09-15/verify_qedge9.py \
  /home/knuvi/Desktop/song/PAN-Crafter \
  /home/knuvi/Desktop/song/PAN-Crafter/research_log/QEDGE9_Implementation_Audit_2026-09-15
```

추가 검사는 같은 인자로 파일명을 `verify_extra.py`, `verify_completion.py`로 바꾼다. 이 감사에서는 기존 unit 스크립트를 운영 work_dir에 직접 실행하지 않았다. 일부 unit이 mandatory 파일을 쓰기 때문에 임시 fixture 안에서 실행했다.

**현장 확인이 필요한 범위는 남아 있다.** 로컬에는 QEDGE9 원본 cue 및 s1 verify 기록이 있으나 s4의 실제 QEC pilot/c_E, s4/s5 신규 9개 run의 50K 결과와 실행 manifest가 없었다. 따라서 원격 배포 commit, 두 서버의 온라인 cue 검증, 기존 small control의 init/data/eval 동치, 첫 seed별 U init hash, 실제 50K/candidate/plateau 완결성을 이번 감사에서 확정하지 않았다. [구현 노트](/home/knuvi/Desktop/song/PAN-Crafter/research_log/2026-09-15_qedge9-implementation.md)에 적힌 40-update GPU 실학습은 임시 산출물을 삭제했다고 되어 있어 원 로그를 재검토하지 못했다. 이번 감사의 실제 trainer step 검증은 이를 독립적으로 보완하지만 장시간 GPU 학습의 완주 증거는 아니다.

§10.3의 seed9091·beta0.05·LF 확장은 현재 구현 노트도 미구현으로 명시한다. 조건부 후속 실험이므로 기본 9개 구현의 즉시 결함으로 세지 않았다. 해당 확장을 실제로 선택할 때는 기존 confirm 명령의 묶음을 그대로 신뢰하지 말고 W104의 올바른 paired config/예약을 별도로 구현·검증해야 한다. seed별 selected/last/plateau 차이와 평균·표준편차는 학습 결과가 확보된 뒤 검증할 항목이며, 이번 수식 통과를 HQNR 개선의 증거로 해석하면 안 된다.
