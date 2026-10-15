**QEGX s3/s4 구현 감사 — 2026-09-15**

**판정: QX50·QEC3·QE50_B005·LFQE50·QER50·QERS의 핵심 학습 구현은 계획과 일치한다. 기본 config/queue 29개도 일치한다. 다만 epoch 끝 checkpoint 재개, QEC3 준비 대기, 기존 control 검증, control 추가 편성에 실제 결함이 있어 전체 실행을 검증 완료로 판정하지 않는다.** 현재 원본 cue가 잘못됐거나 진행 중인 모든 run이 무효라는 뜻은 아니다. 아래 조건이 발생했는지를 현장 로그와 연결해서 판단해야 한다.

기준 문서는 [전체 실험 계획](/home/knuvi/Desktop/song/PAN-Crafter/research_log/PAN_QEGX_S3_S4_W104D121_Experiment_Plan_2026-09-15.md), [s3 인계](/home/knuvi/Desktop/song/PAN-Crafter/research_log/S3_Run_Handoff_QEGX_2026-09-15.md), [s4 인계](/home/knuvi/Desktop/song/PAN-Crafter/research_log/S4_Run_Handoff_QEGX_2026-09-15.md)다. 코드 기준은 `ac232452e6b789225edad6817a04627afdbc7bd2`(19:39:08 KST)이며, QEDGE9 감사 대응 `c47cde6`도 포함해 확인했다. 로컬 origin/main도 동일 commit이고 reflog의 push 기록은 19:40:18이다. 원격 s3/s4가 이 commit으로 실행 중이라는 사실까지 의미하지는 않는다.

계획·코드·config·기반 trainer/runner 등 **86개 기준 파일**을 사본과 SHA256으로 보존했다. [검증 기준 목록](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/provenance.json). 운영 코드를 수정하지 않았으며 실제 학습·큐·cron·Sheet를 변경하지 않았다. CPU 검사와 임시 디렉터리를 사용했다. DataLoader의 로컬 IPC가 샌드박스에서 차단돼 해당 검사만 IPC가 허용되는 실행 환경에서 수행했다.

20:24:55 KST 종료 검사에서 최초 보존한 82개 파일은 변경이 없었고, 추가 기반 파일 4개도 HEAD 내용과 일치했다. tracked worktree 변경은 없었다. [최종 검사 집계](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/final_summary.json).

**발견 사항은 6개다.** P1은 필수 실행·복구·직접 대조의 완결성에 영향을 주는 문제이고, P2는 자산 교체·전달 시 계획과 다른 조건을 수용할 수 있는 검증 공백이다.

| ID | 우선순위 | 문제 | 발현 조건 |
|---|---|---|---|
| F01 | P1 | epoch 끝 checkpoint에서 exact resume 실패 | `epoch-25` 등 정상 저장된 epoch checkpoint에서 재개 |
| F02 | P1 | 직접 큐가 QEC3의 cue-ready 검사를 거치지 않음 | QEC3 차례까지 c_E3가 준비되지 않음 |
| F03 | P1 | verified_complete가 누락된 identity·잘못된 후보 grid를 합격 처리 | 기존 control의 증거가 불충분하거나 근사 재개 기록이 있음 |
| F04 | P1 | 이전 extra 정리 시 새 refresh control까지 편성에서 사라짐 | 옛 extra가 있는 s4에서 `--refresh-controls` 사용 |
| F05 | P2 | 계획에 고정된 cue ID/median을 시작 시 강제하지 않음 | 내부적으로 일관된 다른 threshold asset으로 교체 |
| F06 | P2 | c_E3에 적힌 pilot hash를 실제 pilot bytes와 대조하지 않음 | 기존 c_E3 파일의 출처가 잘못되거나 pilot 파일이 바뀜 |

**F01 — 정상 저장된 epoch checkpoint가 재개 함수의 허용 범위를 벗어난다.** [kdv/resume.py:32](/home/knuvi/Desktop/song/PAN-Crafter/kdv/resume.py:32)는 `skip = global_step - epoch_start_step`을 계산하고 `0 <= skip < n_batches`만 허용한다. 그런데 현재 29개 config는 `save_epoch: 25`, `save_iter: 10000`, train N=9714, batch48, drop_last=True이므로 epoch당 **202 update**다.

[main.py:349](/home/knuvi/Desktop/song/PAN-Crafter/main.py:349)는 한 epoch의 `train()`이 끝난 뒤 epoch checkpoint를 저장한다. [train.py:84](/home/knuvi/Desktop/song/PAN-Crafter/train.py:84)는 그대로 accelerator state를 `epoch-<n>`에 저장한다. `epoch-25`에는 마지막 epoch의 시작 step=4848, 종료 global_step=5050, n_batches=202가 들어가는 구조다. [main.py:254](/home/knuvi/Desktop/song/PAN-Crafter/main.py:254)는 재개 시 global_step=25×202=5050으로 복원한다.

이 실제 N/batch/worker 설정과 EpochState를 사용한 검사는 `skip=202`가 되어 RuntimeError를 재현했다. 따라서 epoch checkpoint를 저장하는 정상 동작과 resume의 경계 조건이 충돌한다. 반면 epoch 중간의 4-worker 재개는 다음 batch 3개가 연속 실행과 같았다. 기존 K29도 epoch 중간을 검사하므로 이 문제를 잡지 못한다.

[_run_cases.sh:85](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:85)는 최신 epoch/checkpoint를 골라 재개하고, rc=1이면 같은 run에서 처음부터 재시도한다. 이 때문에 중단 뒤 복구가 실패해 비용을 다시 쓰거나 같은 디렉터리에 새 실행 기록이 섞일 수 있다. 모든 fresh run의 정상 학습이 잘못됐다는 뜻은 아니다.

수정 기준: epoch 완료 상태를 별도로 저장하거나 `skip == n_batches`를 다음 epoch로 안전하게 전환한다. 해당 checkpoint는 평가 직전에 저장될 수 있으므로 누락된 평가와 RNG 진행도 함께 처리해야 한다. `epoch-25`, epoch 중간 checkpoint, LF 경계 전후를 각각 실제 저장→로드→다음 batch/optimizer/평가 상태까지 비교한다. [재현 결과](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/ops_results.json)

**F02 — s3 직접 큐의 QEC3는 c_E3를 기다리지 않고 실패 원장에 들어갈 수 있다.** [qegx_s3.txt:10](/home/knuvi/Desktop/song/PAN-Crafter/config/queues/qegx_s3.txt:10)에는 QEC3를 포함한 15개 run이 전부 들어 있다. runner가 없는 상태에서 switch/waiter가 기동하면 이 파일이 기본 ORDER가 된다. [_run_cases.sh:123](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:123)는 이 ORDER를 바로 `run_case()`에 전달한다.

`cue_ready()`를 사용하는 곳은 [campaign_gate.py:388](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_gate.py:388)의 추가 편성 단계다. 기본 ORDER의 [run_case:58](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:58)에는 그 검사가 없다. c_E3가 없으면 실제 const 로더는 FileNotFoundError를 낸다. 실패한 run은 재시도 후 [_run_cases.sh:117](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:117)의 `cases_failed.txt`에 기록된다.

실제 `run_case()` 함수를 임시 경로에서 실행하고 smoke/run만 대체해 admission을 분리했다. c_E3가 없는 조건에서도 trainer를 **2번 호출**했고 QEC3가 `terminal=True`가 됐다. [재현 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/missing_ce_runner.log). 실제 학습을 실패시키거나 운영 실패 원장을 수정한 검사는 아니다.

[qegx_waiter.sh:16](/home/knuvi/Desktop/song/PAN-Crafter/tools/qegx_waiter.sh:16)은 terminal인 run을 pending에서 빼므로 나중에 c_E3가 생겨도 이 QEC3를 다시 열지 않는다. waiter 추가 자체는 이전 QEDGE9의 DONE-with-pending 문제를 보완했지만, 직접 큐의 일시적인 자산 미준비까지 해결한 것은 아니다. J0 뒤에 여러 긴 run이 있어 일반적인 경우 pilot은 먼저 준비될 수 있으나, 준비 실패·지연·복구 경로는 여전히 잘못된다.

수정 기준: 직접 ORDER와 추가 gate에 같은 준비 검사를 적용한다. 자산 미준비는 `WAITING_FOR_CUE`로 남기고 실패 원장에 넣지 않는다. QEC3가 waiting인 동안 준비된 독립 case를 진행하는 것은 계획 §11.4가 허용하므로 순서를 무조건 막을 필요는 없다. 준비 완료 후 QEC3가 결국 실행되는지 확인해야 한다.

**F03 — 기존 control의 “검증 OK”가 계획의 재사용 전제를 충족하지 않는다.** 계획 §6은 W104 J0/JQ/XJ의 Teacher/init/data/evaluator·학습 경로, 50K state와 후보 50개, 재개 시 데이터열 계약을 확인하도록 한다. 현재 [verified_complete:387](/home/knuvi/Desktop/song/PAN-Crafter/tools/gen_pakd50_configs.py:387)는 이 요구보다 느슨하다.

- 후보는 exact grid가 아니라 `50000 in steps`와 CSV **45행 이상**만 검사한다. 중복 row도 세고 후보 checkpoint 파일의 완결성은 확인하지 않는다.
- Teacher/data/init 기록이 없거나 init 파일 읽기에 실패하면 `None`을 기록하고 합격 계산에서 제외한다. Teacher가 필요한 JQ도 이 누락을 합격할 수 있다.
- kdv manifest는 비어 있지 않은지만 확인한다. `resumed_nonexact`, evaluator, loss/optimizer/feeder/sampler 계약을 실제 대조하지 않는다.

임시 JQ 디렉터리에 두 결과 marker와 last marker, `step50000`을 45번 반복한 CSV, `resumed_nonexact=true` 및 다른 evaluator를 넣고 Teacher/data/init 증거는 두지 않았다. 실제 함수는 **`ok=true`**를 반환했다. 현재 s4 control이 이런 상태라고 주장하는 것이 아니라, 이 상태를 정상 대조군으로 판정하는 검증의 결함을 재현한 것이다.

또한 [qegx_switch.sh:47](/home/knuvi/Desktop/song/PAN-Crafter/tools/qegx_switch.sh:47)는 검사 불통과를 수집하더라도 60~63행에서 보고만 하고 새 14개 run 편성을 계속한다. 독립 실험을 먼저 진행하는 것은 가능하지만, 이 보고를 control 재사용 승인으로 해석하면 안 된다. 실행 완료와 비교에 필요한 대조군 확보 상태도 별도로 관리해야 한다.

수정 기준: 50개 고정 step 집합과 실제 candidate/last 자산을 확인하고, 필수 identity가 없으면 미검증으로 판정한다. 근사 재개 control은 데이터열 동치가 입증되지 않으면 새 control로 대체한다. 새로운 code 경로와의 동치 검사도 모델 parameter 이름만의 검사로 대신하지 않는다.

**F04 — `--refresh-controls`로 추가한 v3 control이 옛 extra와 함께 보관 파일로 빠진다.** [qegx_switch.sh:66](/home/knuvi/Desktop/song/PAN-Crafter/tools/qegx_switch.sh:66)는 J0/JQ/XJ v3를 config로 만들고 extra_priority에 추가한다. 이어 [90~93행](/home/knuvi/Desktop/song/PAN-Crafter/tools/qegx_switch.sh:90)은 비-QEGX 항목이 하나라도 있으면 extra 파일 전체를 보관 경로로 이동한다.

문제는 새 extra 파일을 `open(...,"w")`로 연 뒤 `G.extra_priority()`를 다시 읽는다는 점이다. 원래 내용은 이미 이동됐고 새 파일은 비어 있으므로, 함께 있던 QEGX v3 control도 복구되지 않는다. config만 남고 extra·mandatory에서 사라진다.

격리된 s4에 이전 W112 JQ S3407 extra 1개를 두고 실제 `--refresh-controls`를 실행했다. 명령은 rc=0이었지만 extra는 빈 목록, QEGX mandatory는 **14개**였고, J0/JQ/XJ v3 세 개는 보관 파일에만 남았다. [전환 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/switch_s4_refresh.log). 옛 extra가 없는 경우까지 항상 실패한다는 뜻은 아니다.

수정 기준: 이동·truncate 전에 전체 목록을 읽고 보관할 목록과 계속 실행할 QEGX 목록을 메모리에서 나눈다. 적용 후 요청한 refresh control이 config·extra·mandatory·예약 모두에 있는지 확인한다. 해당 조건에서는 기본 14개 + refresh 3개 = **17개**가 돼야 한다.

**F05 — 내부 일관성 검증은 개선됐지만 계획의 고정 cue를 강제하지 않는다.** 현재 원본 cue의 asset_id는 계획과 같은 `526dbd718af54c4a0c3eb5fd81778caa`, theta는 `0.3276133416220546`이다. 새 [check_asset](/home/knuvi/Desktop/song/PAN-Crafter/kdv/edge_gate.py:132)는 bank·label·calibration ID·QES shuffle·자체 asset_id를 검사한다. 이전 감사에서 지적한 단순 gate 반전·bank 변경·margin 불일치·이전 asset_id와 다른 재개는 상당 부분 차단됐다.

하지만 theta가 calibration q의 **실제 median인지** 확인하지 않으며, fresh run의 config가 계획의 기대 asset_id를 고정하지 않는다. 원본 q/e/calibration을 유지하고 theta만 0.4로 바꾼 뒤 label·shuffle·NPZ SHA·자체 asset_id를 일관되게 다시 계산한 임시 자산을 실제 `EdgeGate.load()`가 수용했다. calibration q의 median은 여전히 0.3276133416이다. 수용된 새 ID는 `afd5fe9d1064bb0252062ce2f7aa7e01`이었다.

이것은 현재 cache 생성 결과의 오류가 아니다. 이후 재생성·전달 과정에서 계획과 다른 자산을 같은 역할에 넣어도 자체 무결성 검사만으로 통과할 수 있다는 의미다. 기존 asset_id를 명시한 `previous`를 전달하면 변경 자산은 정상 거부됐다.

수정 기준: campaign의 expected asset_id/NPZ SHA를 config 또는 고정 release manifest에 넣고 trainer·verify에서 대조한다. median 및 decile 경계도 calibration 데이터에서 재계산해 의미를 확인한다. asset 자체의 hash와 계획이 요구하는 asset identity는 별도 조건이다.

**F06 — c_E3의 pilot identity는 일부 검사되지만 checkpoint hash는 여전히 선언값이다.** QEC3의 산출 경로는 이제 s3 J0/W104D121/S2026/v2/last/50000을 강제하고, s1 QEC 파일과 출력 경로도 분리했다. 이 수정은 확인했다.

그러나 [EdgeGate.load](/home/knuvi/Desktop/song/PAN-Crafter/kdv/edge_gate.py:246)는 c_E 파일의 run/tag/step와 cue ID를 대조한 뒤 pilot hash 두 개는 비어 있지 않은지만 확인한다. 실제 local pilot의 hash 또는 사전에 고정된 expected pilot hash와 비교하지 않는다. 올바른 run/tag/step에 임의의 `f...f` full hash와 임의 tensor hash를 적은 c_E3 파일도 실제 로더가 수용했다.

또한 [qedge9_cue.py:178](/home/knuvi/Desktop/song/PAN-Crafter/tools/qedge9_cue.py:178)의 기존 c_E 유지 분기는 cue ID·pilot run·tag만 같으면 종료하므로, 이 경로에서도 현재 checkpoint bytes를 재확인하지 않는다. 처음 자동 산출한 정상 파일이 곧바로 잘못됐다는 뜻은 아니다. 동일 이름의 pilot이나 전달 파일이 교체됐을 때 검출할 검사가 부족하다.

수정 기준: 최초 pilot의 full checkpoint hash를 고정해 c_E3와 묶고, 기존 c_E를 사용할 때 현재/기대 pilot 및 calibration view 수·index·데이터 계약을 확인한다. 동일 출처 검증과 계수의 재산출·덮어쓰기를 구분한다.

**정상으로 확인한 구현은 다음과 같다.** 기존 단위검사의 주장만 요약하지 않고, 실제 모델·실제 patch·독립 gradient·실제 train loop를 추가로 사용했다.

| 항목 | 확인 결과 |
|---|---|
| s3 큐 | S2026 J0/JQ/QE50/XJ/QX50/J_R3_NOEDGE/QEC3/QES/J_QB005/QE50_B005 + S4321 J0/JQ/XJ/QE50/QX50, 15개 |
| s4 큐 | S1234 QE50/LF0/LFQ/LFQE50/LFX/JE0/QER50/QERS + S3407 J0/JQ/QE50/JE0/QER50/QERS, 신규 14개 |
| version·branch | s3 v2는 s5 QEDGE9 v1과 config 충돌을 피하기 위한 구분. s4 신규 v1. 모두 QEGX campaign/branch로 생성 |
| 구조·초기화 정책 | Student W104D121, backbone 1,903,624 params. Teacher T0 W112D123 own-config/strict load, A hash 및 margin4 확인. U는 fresh init 정책 |
| 공통 학습 설정 | 50K/batch48/AdamW WD0.01/U LR1e-4/A LR1e-5/warmup100/cosine, 고정 tau/lambda/후보 grid 일치 |
| QX50 | Teacher-informed hard는 유지하고 soft만 0, q-gated GT edge 유지. XJ와 구분 |
| beta pair | J_QB005/QE50_B005만 beta0.05. 실제 gated soft loss가 기본의 정확히 절반 |
| QEC3 | 별도 c_E3 경로 및 s3 pilot binding. trainer의 상수 edge 수식 일치. 실제 현장 c_E3 값은 미확인 |
| QER50 | total의 U 경로는 all-edge, A gradient에서 q-high edge 부분만 제거. reconstruction 이중 계산 없음 |
| QERS | 동일 U all-edge, A에 기존 e-decile×rot shuffle label 사용. QES와 같은 assignment |
| LFQE50 | 24999에서 A 학습, 25000부터 A 동결. 실제 AdamW step 후 A hash 불변, offset=0, U gated edge는 유지 |
| Teacher·gate | frozen T0/GT/gate로부터 gradient 없음. q는 Student A의 현재 출력으로 재계산하지 않음 |
| 전환 | 현재 runner를 kill하는 이전 동작 제거. dry-run의 운영 mandatory/reservations/extra/deadline/queue는 불변. 검증용 디렉터리·로그는 기록할 수 있음 |
| 시간 정책 | QEGX ledger·required 경고 정책·deadline 제거. 새 QEGX 실측도 `measured_hours_all`에서 예약에 반영 |
| 시간 합계 | s3 25.2040h, s4 26.280333h. 준비 전후 비용은 별도. 계획 반올림과 일치 |
| metadata·Sheet | X열 QEGX, edge_U/edge_A, beta/freeze/c_E pilot/cue descriptor 연결 확인. 실제 원격 업로드는 미검사 |

실제 데이터 4개 base patch × 4 view 온라인 재계산은 원본 cache와 수치 허용오차 안에서 일치했다. 원본 NPZ SHA는 `b83cc5d1a25e8766d1813c9d19d956efc24e0f152553e53c4894fa18dcdf563e`, 실제 읽은 T0 checkpoint SHA는 `16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32`로 계획과 같다. 전체 cache 내부 일관성·shuffle은 기존 136개 검사와 실제 `read_asset()`에서도 확인했다. 이전 감사의 384-view 측정을 이번에 다시 수행했다고 세지는 않았다.

**Gradient 수치 결과.** 실제 W104 모델, T0 A/U, 실제 H5 patch를 사용했다. 단일 step 비교는 A gradient가 퇴화하지 않도록 U output을 비퇴화 상태로 만들었다. 독립 계산은 hard/soft 식, per-sample Scharr, U/A 수신식을 각각 작성했다.

| Case | A gradient 최대 절대 오차 | U gradient 최대 절대 오차 |
|---|---:|---:|
| QE50 | 0 | 0 |
| QX50 | 0 | 0 |
| QE50_B005 | 0 | 0 |
| QEC3 (`c_E3=0.37` 검사값) | 0 | 0 |
| QES | 0 | 0 |
| QER50 | 8.09e-11 | 0 |
| QERS | 6.91e-11 | 0 |

QEC3의 0.37은 수식 확인을 위해 주입한 검사값이며 실제 s3 calibration 결과가 아니다. QER50/QERS는 A parameter tensor 22개에 보정이 적용됐다. 이 수가 모든 tensor의 gradient가 비영이라는 뜻은 아니다. scale128/accumulation2 조건의 보정 대수도 통과했다. 현재 config는 mixed_precision=None이며, 이번에는 CUDA AMP 실행·GPU별 clipping/scaler 동치를 검증하지 않았다.

추가로 **실제 `KDVTrainer.train()` 6 update**를 CPU에서 실행했다. 진짜 zero-output 초기화에서 시작하고 실제 T0 및 실제 4개 patch를 반복 사용했다. 실제 Accelerator backward, gradient correction, AdamW, 50K cosine/warmup100 scheduler와 last checkpoint 저장을 실행했다. 진단·운영 ledger callback만 감사용으로 대체했다. 독립 U/A gradient로 같은 optimizer를 6회 진행한 기준과 비교해 최종 **U parameter 오차=0, A 최대 오차=2.33e-10**, Teacher 불변, last state 저장을 확인했다. 이 검사는 전체 main.py 평가·장시간 50K 완주를 대신하지 않는다. [실제 loop 검증 결과](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/train_loop_results.json)

**진행 중인 실험의 확인 범위.** 접근 가능한 현재 호스트의 `gspread/server.txt`는 s1이다. 20:14 KST의 실제 호스트 process에는 B03_GF2 학습과 기존 runner가 확인됐고, 이 자료로 원격 s3/s4의 상태를 판단하지 않았다. [로컬 process snapshot](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/local_process_snapshot.json). 같은 Python 명령행을 가진 DataLoader worker가 포함될 수 있으므로 여러 학습 중복 실행의 증거로 사용하지 않는다.

로컬 QEGX campaign CSV에는 `e2e_QER50`의 40-update 개발 실행 두 번과 평가 기록이 남아 있었다. 이는 [구현 노트](/home/knuvi/Desktop/song/PAN-Crafter/research_log/2026-09-15_qegx-implementation.md)의 개발 테스트와 부합하지만, s3/s4의 신규 29개 run 완주 기록은 아니다. 원래 임시 run 디렉터리는 삭제됐으므로 당시 전체 gradient 로그·checkpoint bytes를 재검토하지 못했다. [보존한 개발 실행 CSV](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/local_e2e_runs.csv)

s3/s4의 SSH 별칭이나 로컬 수집 로그 경로를 요청했으나 감사 시점에 해당 자료는 확보되지 않았다. 따라서 아래 항목은 **현장 미확인**이다.

- 실제 실행 commit 및 해당 worktree의 변경 여부, active run/config/step, 준비 검사 로그.
- s3 c_E3의 실측값·고정 pilot full hash 및 실제 QEC3 상태.
- s4 J0/JQ/XJ control의 원본 manifest·50K/candidate·재개 기록, refresh control 필요 여부 및 실제 편성 목록.
- 실행별 초기 U hash의 paired 일치, 서버별 cue verify, raw selected/last/plateau·후처리·Sheet 기록의 완결성.

현장 로그를 볼 때는 F01의 `exact resume ... n_batches 202` 오류 및 뒤이은 fresh 재시도, F02의 QEC3 실패 원장, F04의 v3 config 존재 대비 extra/mandatory 누락부터 대조하면 된다. 코드 결함이 재현됐다는 사실과 해당 서버에서 이미 발현했다는 사실은 구분한다.

**검사 산출물.** 기존 **136/136 단위검사**, 추가 **53개 적합성 검사**(운영/config 38 + 수식 14 + 실제 loop 1)가 통과했다. 별도 부정/경계 시나리오 **6개**는 위 결함이 재현되는 것을 확인했다. 결과 JSON의 `defect_reproduction: passed=true`는 그 결함을 관찰했다는 의미다.

- [기존 unit 결과](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/existing_units_host.log), [unit 실행 코드](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/run_units.py)
- [config·운영 검사 코드](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/verify_ops.py), [결과 JSON](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/ops_results.json)
- [독립 수식 검사 코드](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/verify_math.py), [수식 결과 JSON](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/math_results.json)
- [실제 train loop 검사 코드](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/verify_train_loop.py), [loop 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/train_loop.log)
- [s3 dry-run 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/switch_s3_dry.log), [s4 dry-run 로그](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QEGX_Implementation_Audit_2026-09-15/switch_s4_dry.log)

검사 초기 시행착오도 보존했다. 최초 단위/worker 검사는 샌드박스 IPC PermissionError로 중단돼 제품 실패로 세지 않았다. 수식 검사의 최초 parameter count는 감사 코드에 정확값을 잘못 적어 실패했으며, 계획이 명시한 1.9036M 정밀도로 수정 후 재실행했다. 실제 loop fixture도 로깅용 case 필드를 추가한 뒤 재실행했다. 초기 로그는 `*.initial.*` 및 `unit_launcher.log`, `ops.log`에 남아 있다. 위 통과 수는 수정한 감사 harness의 최종 결과다.

재실행은 기준 commit과 같은 data/T0를 가진 checkout에서 다음처럼 한다. production switch를 직접 실행하는 명령이 아니다. `run_units.py`와 `verify_ops.py`는 DataLoader 로컬 IPC가 가능한 환경이 필요하다.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/knuvi/miniconda3/envs/pancrafter/bin/python -B \
  research_log/QEGX_Implementation_Audit_2026-09-15/verify_ops.py
```

같은 방식으로 `run_units.py`, `verify_math.py`, `verify_train_loop.py`를 실행할 수 있다. 감사 스크립트의 ROOT는 이번 workspace로 고정돼 있다.

§8의 LF 추가 seed·beta 확인 pair·e cutoff·theta 민감도·routing 평균 계수 대조는 조건부 후속이며 현재 미구현으로 명시돼 있다. 이들을 기본 29개 구현의 누락으로 세지 않았다. 현재 검증은 학습 명세 적합성과 실행 신뢰성에 대한 것이며, HQNR 개선이나 방법 채택의 증거는 50K 대응 결과가 확보된 뒤 별도로 판단해야 한다.
