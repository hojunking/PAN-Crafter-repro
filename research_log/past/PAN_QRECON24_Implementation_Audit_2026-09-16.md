**QRECON24 구현 감사 — 2026-09-16, s1 로컬 검증**

**판정: 핵심 학습 수식과 68개 편성은 통과했다. 캠페인 전환·정확 재개·24시간 보충·공식 결과 선택은 수정이 필요하다.** 현재 관측한 s1 G22의 loss/gradient/초기화는 계획에 맞는다. 이 판정은 50K 성능 달성이나 다섯 서버 전체의 정상 완주를 보증하는 판정은 아니다.

기준 문서는 [QRECON24 계획](/home/knuvi/Desktop/song/PAN-Crafter/research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md) 전체다. 제목의 S1/S5와 달리 실제 범위는 s1–s5의 **68 FRESH50 run**이다. 선택적 §11 tail은 기본 68개 구현 완료의 필수 조건으로 취급하지 않았다.

최초 검증 release는 `15204de6c29ae5c7bc1ad29f6a2220a60eae31c6`이다. 검토 중 다른 작업에서 추가된 `48c0fad`(예약 표 머리글)와 `8a50aec`(K48c 서버 의존 검사 수정)도 반영했다. 두 후속 커밋은 학습 loss·gradient·68개 config·switch/waiter/selector의 실행 로직을 바꾸지 않았다. 최초 소스 309개와 추가 의존 소스 20개, 변경된 두 파일을 각각 보존했다. 출처와 SHA는 [source_manifest.json](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/source_manifest.json), [source_update_manifest.json](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/source_update_manifest.json)에 있다.

감사에서는 운영 코드·config를 수정하거나 학습을 기동/중단하지 않았다. Shell 재현은 `/tmp` 사본에서 실행했고, 학습 연산은 CPU를 사용했다. 기존/진행 중 실험 산출물은 읽기만 했다. 업로드·Sheet 변경·commit/push는 수행하지 않았다.

**통과한 구현**

| 계획 조건 | 검증 결과 |
|---|---|
| W104 D121, plain 9ch, fresh U + T0 A | backbone 1,903,624 parameter. T0 SHA와 A hash 일치. 실제 s1 U 초기 파일과 기록 hash 일치 |
| 고정 T0 자기 정합 경로 | 실제 trainable-A run의 `share_correction=false`, Teacher gradient 없음·state 불변 |
| 연속 raw q | 38,856 view 전수에 `2*qref/(qref+raw_q)` 대조. binary gate를 함수 입력으로 쓰지 않음 |
| qref 고정 | `0.3276133416220546`. 실제 w 범위 `0.7944487–1.0949190`, 평균 `0.99631455`; 배치 평균 1 정규화 없음 |
| shuffle | Teacher e decile × rot의 40 stratum 각각 연속 w multiset 보존. 고정 seed51515 |
| U/A 목적함수 | `U←mean(H+K+λE*wE*E)`, `A←mean(wA*H)`를 독립 수식/gradient로 대조 |
| 29 profile | 실제 T0/W104 모델·H5 patch·서로 다른 rot state로 **29개 모두** gradient 검사 통과. 최대 U 상대오차 약 `1.1e-8`, A 상대오차 `0` |
| edge 계수 | `3e-4/1e-3/3e-3`를 절대 계수로 적용. 기존 λE0 배율 미적용. FP32 Scharr를 독립 구현해 대조 |
| A 직접 gradient | hard-only, β/λE 영향 배제. α는 A의 H에도 반영. U에 A 목적함수 gradient를 더하지 않음 |
| offset/jitter | native step, offset=0, 전용 corruption RNG 미소비 |
| A_FREEZE | U만 optimizer update, A parameter 불변. 기존 단위검사에서는 buffer 불변도 검사 |
| 실제 trainer loop | G22/A_FREEZE/E_SHUF 각각 8 update를 실제 `KDVTrainer.train`·Accelerator·AdamW·cosine·last 저장으로 실행. 독립 AdamW reference와 최대 parameter 차이 U `7.45e-9`, A `1.86e-9` |
| 68 config/queue | 문서 §5의 표를 직접 파싱해 서버·순서·seed·λE·rA·α/β·LR 전수 대조. 68개 모두 생성기 출력과 byte 일치 |
| 고정 학습량 | 50K, batch48, warmup100, WD .01, exact_resume 요청, 모든 후보 보존 설정 확인 |
| 예약 산식 | s1 33.68h / s2 35.12h / s3 32.16h / s4 34.925333h / s5 33.248h. 모두 `1.20*N*R+N*10/60` 일치 |
| target 정상 분기 | H=.9584999 제외, .9585 포함. 공식 RR의 SCC→ERGAS→PSNR→SAM→Q8→SSIM→H 순서 확인 |

독립 수식/학습 경로 검사는 **37/37** 통과했다. 기존 K01–K48은 과거 s1 fixture를 제공한 환경에서 **171/171** 통과했다. 최신 K48c 수정의 재검증은 아래 별도 기록을 따른다. 이 수치는 실패 경로를 빠뜨리지 않았다는 뜻은 아니다. 다음 항목은 정상 경로 검사와 별도로 재현했다.

증거: [math_results.json](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/math_results.json), [ops_results.json](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/ops_results.json), [기존 검사 결과](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/units_with_history.json).

**F01 · P1 — epoch 끝 checkpoint에서 exact resume가 실패한다. 기존 감사 문제가 남아 있다.**

위치: [kdv/resume.py:33](/home/knuvi/Desktop/song/PAN-Crafter/kdv/resume.py:33), [main.py:353](/home/knuvi/Desktop/song/PAN-Crafter/main.py:353), [tools/_run_cases.sh:85](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:85).

실제 train set 9,714개, batch48/drop_last로 epoch당 202 batch다. `save_epoch=25`가 만드는 `epoch-25`는 global_step=5,050인데, 저장된 EpochState의 epoch_start_step은 4,848이다. 재개 시 `skip=202`가 되고 현재 조건 `skip<n`에 걸려 RuntimeError를 발생시킨다. 이 숫자로 현재 helper에서 그대로 재현했다. K29의 중간-epoch 재개 검사로는 이 문제를 검출하지 못한다.

runner는 최신 epoch/checkpoint를 우선 재개하고, 이 오류처럼 exit3/4가 아닌 실패에는 **같은 run ID로 처음부터 재시도**한다. 따라서 계산 낭비뿐 아니라 이전 후보/로그와 새 학습의 혼합 가능성이 있다. 계획 §6.3의 정확한 중단 재개를 충족하지 못한다. epoch 경계를 별도 처리하고, 재개 불일치 시 동일 ID로 자동 fresh 재실행하는 경로도 막아야 한다. 단순히 `skip<=n`으로 바꾸어 빈 iterator를 반환하는 것만으로는 epoch 번호·평가 격자까지 맞는 재개를 입증할 수 없다.

**F02 · P1 — 실행 중인 old run 하나가 끝난 직후 새 큐로 전환되지 않는다.**

위치: [qrecon24_switch.sh:83](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_switch.sh:83), [_run_cases.sh:32](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:32), [_run_cases.sh:124](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:124).

switch는 살아 있는 runner를 두고 다음 gate pass를 기다린다. 기존 runner는 시작할 때 `ORDER`를 배열로 읽고 그 배열 전체를 소진한 뒤 gate를 호출한다. 이미 진행 중인 gate의 `EXTRA`도 그 pass 동안 같은 배열이다. 현재 run 다음에 남은 old pending이 있으면 그 작업까지 실행한다. 계획 §9.1의 “진행 중 run 정상종료 후 다음 큐 교체”보다 늦다.

격리 환경에서 실제 switch와 원본 runner의 ORDER loop를 조합했다. 첫 case 동안 switch를 적용해도 실행 순서는 `OLD_CURRENT → OLD_PENDING_1 → OLD_PENDING_2`였다. 실제 학습 대신 표식만 쓰는 run_case를 사용했다. 진행 중 GPU 작업을 중단하지 않으면서 **case 경계에서 큐 세대를 다시 확인하는 인계**가 필요하다. s1은 관측 시 idle에서 새 큐를 시작했으므로 이 문제를 겪었다는 증거는 없다. 남은 old queue가 있는 서버에 해당한다.

증거: [handoff_results.json](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/handoff_results.json).

**F03 · P1 — `--extend`가 보충 3run을 생성해도 실행으로 이어지지 않는다.**

위치: [qrecon24_switch.sh:35](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_switch.sh:35), [qrecon24_waiter.sh:13](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_waiter.sh:13).

`--extend`는 config와 `extra_priority.txt`만 갱신하고 즉시 종료한다. QRECON24 mandatory 목록·실제 queue·reservation을 갱신하지 않고 runner/waiter도 재기동하지 않는다. waiter는 mandatory 목록만 읽으며 기본 큐 완료 시 종료한다. 특히 pakd50 gate가 꺼진 s1에서는 새 extra 목록을 실행할 경로가 없다.

기본 12개 완료·신규 시간 18h인 fixture에서 `--extend`는 rc0으로 보충 config 3개를 만들었다. mandatory는 여전히 기본 12개였고, **실제 waiter를 다시 실행해도 DONE으로 종료**했다. 계획 §8.3의 시간 보충이 이 명령만으로 성립하지 않는다. 보충 목록을 실행 큐/mandatory에 반영하고, 완료된 체인의 안전한 재기동까지 연결해야 한다.

**F04 · P2 — 이전 extra와 QRC24 extra가 함께 있으면 전환 시 QRC24 extra도 편성에서 빠진다.**

위치: [qrecon24_switch.sh:71](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_switch.sh:71).

기존 extra 파일을 `os.replace`로 archive한 후, 새 파일을 `open(...,"w")`로 열고 그 안에서 `G.extra_priority()`를 다시 읽는다. 이때 읽는 파일은 이미 비어 있다. 이전 항목 1개 + QRC24 보충 3개를 넣고 실제 내장 Python block을 실행했더니 활성 extra가 `[]`가 되었다. archive에는 남아 있으므로 영구 삭제는 아니지만 자동 편성에서 사라진다. 기존 내용을 메모리에 읽은 뒤 보존할 QRC 항목을 결정하고 원자적으로 써야 한다.

**F05 · P2 — 최소 24h 집계가 학습과 export를 분리하지 않는다.**

위치: [qrecon24_switch.sh:27](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_switch.sh:27), [train_kdv.py:1677](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:1677).

`hours_total`을 `hours`보다 우선 합산한다. trainer에서 `hours_total`은 학습·평가·export까지 포함한 시간이다. 학습 종료 시점 `hours`와 다르며, 두 값 모두 초기 준비 시간도 완전히 분리하지 않는다. 별도로 요구된 `new_train_h / extra_official_rr_h / postprocess_h / setup_h / wait_h` 기록도 없다. 따라서 계획 §8.3의 시간 구분을 구현했다고 볼 수 없다.

12run의 학습 종료 기준 합이 23.4h이고 export 포함 합이 24.6h인 fixture에서 “확장 불필요”로 종료했다. 추가로 정상적인 불필요 상태를 문자열 `sys.exit(...)`로 반환하므로 shell은 rc1/`--extend 실패`로 표시한다. 시간 경계를 명시하고 완료된 신규 학습 시간만으로 보충 여부를 결정해야 한다. 과거 Sheet Train(h)를 예약 대용값으로 사용한 사실과 새 실행의 분리 집계는 구분해야 한다.

**F06 · P2 — `--official`이 일부 후보의 proxy와 공식 RR를 섞어 선택한다.**

위치: [qrecon24_select.py:88](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_select.py:88).

적격 후보 checkpoint가 없으면 메모만 남기고 CSV proxy RR가 든 그 후보를 순위에 유지한다. 재현에서 checkpoint가 없는 step1010의 proxy SCC=.999가 공식 RR가 계산된 step2020의 SCC=.989보다 앞서 target으로 선택되었다. 반환은 rc0, `target_feasible=true`, `official=false`였다.

`official=false`를 남기는 점은 확인했다. 다만 계획 §7.2가 금지한 서로 다른 evaluator 값의 혼합 정렬은 실제로 발생한다. `--official`은 적격 후보 전부의 공식 평가가 준비될 때까지 target 확정을 보류하거나 명시적인 불완전 상태로 종료해야 한다. proxy 탐색 결과와 공식 target을 별도 필드/파일로 두는 편이 안전하다.

**F07 · P2 — 공식 RR 캐시가 checkpoint·dataset·evaluator hash에 묶여 있지 않다.**

위치: [qrecon24_select.py:47](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_select.py:47), [qrecon24_select.py:101](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_select.py:101).

`reduced_candidate_step-N.mat`이 있으면 재추론 없이 사용한다. checkpoint 파일은 존재 여부만 확인한다. 실제 완료 W104 checkpoint로 `/tmp`에서 캐시를 만든 뒤, 해당 사본 checkpoint를 유효하지 않은 bytes로 교체해도 **같은 RR를 `official`로 반환**했다. 운영 checkpoint는 수정하지 않았다.

selector JSON에도 selected checkpoint full SHA, RR dataset/evaluator SHA가 없다. F01처럼 같은 ID/step의 내용이 바뀌거나 결과를 다른 폴더에 복사한 경우 오래된 RR를 새 checkpoint와 연결할 수 있다. cache sidecar의 checkpoint/config/data/evaluator hash 검증과 선택 결과에 대한 동일 hash 기록이 필요하다.

**F08 · P2 — 완료 판정이 계획의 50개 후보·자산 동일성을 강제하지 않는다. 기존 감사 문제가 남아 있다.**

위치: [gen_pakd50_configs.py:495](/home/knuvi/Desktop/song/PAN-Crafter/tools/gen_pakd50_configs.py:495), [campaign_gate.py:39](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_gate.py:39).

QRC24 이름의 fixture에 빈 result mat 2개, 빈 last checkpoint, `last_meta.step=50000`, **50000만 45번 중복한 CSV**, `resumed_nonexact=true`만 있는 manifest를 두었다. T0/data/init 증거가 전혀 없어도 `verified_complete.ok=true`였다. 필수 identity를 `None`으로 두고 all()에서 제외하며, 후보 조건도 고유 50 step/실제 checkpoint 존재를 확인하지 않기 때문이다.

실제 scheduling의 `complete()`는 두 mat 존재만 보고, `terminal()`은 실패 원장도 포함한다. waiter의 DONE은 유효한 68개 완주와 같은 뜻이 아니다. QRC24에서는 정확한 격자 50개, 실제 checkpoint, fresh/init/T0/cue/data/evaluator/release 및 exact 재개 상태를 필수로 검증하고, 실패/불완전과 정상 완료를 별도 상태로 전달해야 한다.

**F09 · P2 — s4/s5의 26개 run에서 control 링크가 실제 편성에 없는 G22를 가리킨다.**

위치: [gen_pakd50_configs.py:750](/home/knuvi/Desktop/song/PAN-Crafter/tools/gen_pakd50_configs.py:750).

모든 profile의 `control_runs`와 `baseline_run`을 같은 서버/seed의 `G22` ID로 만든다. s4는 H22, s5는 L100이 편성되어 있고 별도 G22는 편성되지 않았다. 따라서 s4 14개+s5 12개가 존재하지 않는 control ID를 Notes로 전달한다. 수학적으로 G22=H22=L100이라는 canonical 정의는 올바르지만 **run ID는 alias가 아니다**. 실제 대응 H22/L100 run ID로 매핑해야 한다.

**F10 · P2 — 재현성 manifest의 필수 수치·full hash가 일부 빠져 있다.**

위치: [train_kdv.py:161](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:161), [train_kdv.py:699](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:699), [kdv/qrecon.py:46](/home/knuvi/Desktop/song/PAN-Crafter/kdv/qrecon.py:46), [qrecon24_select.py:101](/home/knuvi/Desktop/song/PAN-Crafter/tools/qrecon24_select.py:101).

실제 s1 checkpoint의 optimizer는 betas=(.9,.999), eps=1e-8, WD=.01, U/A group 74/22개, 초기 LR=1e-4/1e-6으로 올바르다. 그러나 resolved training manifest는 betas/eps, parameter 이름별 group·decay 제외, cosine 최저값, accumulation/clip, TF32를 기록하지 않는다. 일부 값은 라이브러리 기본값으로만 남아 있다. 현재 수치가 잘못됐다는 판정이 아니라 계획 §3·§6.4의 명시적 고정/기록 계약이 미완료라는 판정이다.

cue NPZ의 full SHA는 기록한다. w/shuffle에는 16자리 hash만 있고 permutation mapping의 full SHA는 없다. selector의 `display_tie`와 추가 공식 RR 소요시간 기록도 미구현이다. 이 항목들은 학습 loss를 바꾸지 않고 보완할 수 있다.

**검토 중 수정된 항목 — 최초 K48c 서버 의존 기동 차단**

`15204de`의 K48c는 `PAKD50_FQ_W112_D123_WV3_T0_S1234_FRESH50_v1` 결과를 무조건 요구했다. 해당 결과가 없는 격리 환경에서 170개 통과/1개 실패를 재현했고, switch의 필수 gate였으므로 다른 서버 기동을 막을 수 있었다. 동일 환경에 그 fixture의 읽기용 결과 3개를 복사하면 171개 모두 통과했다.

감사 중 다른 작업에서 커밋된 `8a50aec`는 로컬 다른 결과를 사용할 수 있게 바꾸고, 결과가 전혀 없으면 이유를 적어 K48c를 SKIP한다. 따라서 최초 결함을 현재 미수정 항목으로 세지 않는다. 수정 후 전체 gate를 다시 실행한 결과, 과거 결과 없는 환경은 rc0/171 OK(그중 K48c는 명시적 SKIP), 다른 W104 완료 결과를 둔 환경은 rc0/171 OK였다. 실제 재검증 로그/집계는 [updated_units_no_history.json](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/updated_units_no_history.json)과 [updated_units_other_history.json](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/updated_units_other_history.json)에 보존한다. SKIP을 selector 검증 완료로 읽어서는 안 된다. 서버 데이터와 무관한 작은 합성 fixture를 단위검사 안에서 생성하는 방식이 더 재현 가능하다.

**현재 s1 실행과 검증 범위**

감사 중 별도 작업에서 `PAKD50_QRC24_S1_G22_W104_D121_WV3_T0_S1234_FRESH50_v1`가 기동되었다. chain 시작 기록은 16:21:24 KST, run의 source SHA는 `15204de`다. 16:25:44 관측 시 step1700이며 step1010 후보가 보존되어 있었다. 감사자가 시작한 실행이 아니다.

실행 manifest에서 fresh/exact_resume, AMP=`no`, margin4, Teacher SHA, cue asset ID, continuous q, A_soft=0/A_edge=0/student_offset=0을 확인했다. U 초기값 hash는 `3965232db289bf84`, A 초기값 hash는 `493aca181f93ae68`이다. step1010 저장 optimizer의 U LR `9.991796434444349e-5`, A LR `9.991796434444349e-7`로 비율 .01이다. step1000 진단에서 λE=.001, weighted edge 기여 약 `1.33688e-5`, A gradient norm 약 `.01154`, U/A gradient parameter 수 74/22를 확인했다. 무한값·잘못된 gradient 합산을 발견하지 않았다.

증거: [live_s1](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/live_s1), [selector_live_results.json](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/selector_live_results.json).

s2–s5의 실제 프로세스·checkout SHA·asset 배포·50K 완료 상태는 직접 접속해 확인하지 않았다. 로컬 config/스크립트 감사로 원격 배포까지 검증 완료라고 쓰지 않는다. 계획 §9.2의 동일 SHA 별도 worktree 사용 여부도 원격 확인이 남아 있다. 구현 노트의 일반적인 “git pull 후 전환” 지시만으로 그 계약이 보장되지는 않는다.

**공식 RR 수치 재현에서 남은 확인 사항**

W104 완료 run의 step27270을 `/tmp`로 복사하여 CPU로 20장 재추론하고 공식 RR를 계산했다. selector와 실제 trainer `_infer`, 학습용 `kdv_forward`는 **동일 CPU·동일 checkpoint·동일 입력에서 출력 차이 0**이었다. 후보와 best_hqnr checkpoint의 파일 SHA도 같다.

다만 기존 GPU export와 비교한 강한 수치 일치 기준은 통과하지 못했다. PSNR 차이 `0.00118597`, ERGAS `0.00029841`, SAM `0.00051870`, SCC `2.6254e-6`, Q8 `6.1743e-5`, SSIM `9.6482e-6`이었다. SR의 최대 차이는 10.2785 DN, RMSE는 .380893 DN이다. 장치/커널 정밀도 등 원인을 이 검사만으로 확정하지 않는다. **공식 RR가 전 자리 동일하다는 주장을 W104/CUDA까지 확대 검증하지 못했다.** threshold에 매우 가까운 결과를 판정하기 전에 실제 운영 GPU·동일 precision 정책에서 기존 export와 selector 재추론을 대조할 필요가 있다. 이번 감사는 진행 중 GPU 학습에 추가 추론을 넣지 않았다.

증거: [inference_paths.json](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/inference_paths.json), [selector_live_results.json](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16/selector_live_results.json). 후자의 `official_RR_W104_export_equivalence=false`는 의도적으로 남긴 미통과 검사다. 이를 확인된 selector 알고리즘 결함으로 집계하지 않았다.

계획 §6.3의 G22/A_UNIF/A_FREEZE/E_UNIF/αβ 경계 **각각 200–500 update smoke**도 전부 완료되었다고 인증할 근거는 부족하다. 구현 노트는 G22 200, A_FREEZE/E_SHUF 40 및 1-step smoke를 기록하고 있다. 이번 감사는 실제 G22의 더 긴 진행을 관측했고, 독립 29-profile gradient 및 3-profile 8-update loop를 추가했지만 전체 지정 smoke를 대신했다고 주장하지 않는다. target 대표 행 export는 구현 노트대로 미지원이며, 계획 §9.3이 허용하는 pending 표시를 유지하는 점은 문제로 세지 않았다.

**수정 후 재검증 기준**

1. epoch-25와 mid-epoch checkpoint 모두 연속 실행과 같은 다음 batch·update·scheduler·후보 격자로 재개한다. 불일치 시 동일 ID fresh 재시작을 하지 않는다.
2. old current 1개만 마친 뒤 다음 실행이 QRC24 첫 pending이 된다. 기존 runner 종료를 강제하지 않고 case 경계에서 인계한다.
3. 기본 큐 완료/<24h 상태에서 `--extend` 한 번으로 지정 seed 3개가 실제 편성되고, waiter가 pending을 추적한다. old/new extra가 섞여도 QRC 항목이 보존된다.
4. 공식 RR 미완료 후보·hash가 다른 캐시·불완전 50개 격자를 최종 target/유효 완료로 승인하지 않는다.
5. s4 H22·s5 L100의 실제 control ID, full hash/optimizer/time manifest를 기록한다. 현재 CPU/GPU RR 차이는 동일 운영 환경에서 다시 검증한다.

재현 스크립트와 원시 로그는 [감사 자료 폴더](/home/knuvi/Desktop/song/PAN-Crafter/research_log/QRECON24_Implementation_Audit_2026-09-16)에 모았다. `defect_reproduction`의 `passed=true`는 결함을 재현했다는 뜻이며 구현 통과 수치에 더하지 않았다.
