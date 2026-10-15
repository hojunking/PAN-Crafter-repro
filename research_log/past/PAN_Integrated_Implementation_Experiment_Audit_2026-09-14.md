**PAN 통합 방법·50시간 실험 구현 검증 — 2026-09-14**

검증 대상은 `PAN_Integrated_Method_Summary_2026-09-14.md`와 `PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md`다. 아래 §번호는 상세 계획을 가리킨다. 실행 직전 작성된 구현 노트는 변경사항의 근거로 읽되, 테스트 통과 여부는 별도로 확인했다.

**판정: J/F × N0/R1/Q12/X02의 핵심 수학·gradient 구현은 검증한 범위에서 맞다. 그러나 배포·calibration 전환·우선순위·예산·정확한 resume 계약이 충족되지 않아, 현재 캠페인 전체를 ‘계획대로 구현·운영되고 있다’고 승인할 수 없다.**

현재 s1 J0의 학습 목적함수가 잘못됐다는 증거는 발견하지 않았다. 이 run을 즉시 폐기할 근거도 없다. 우선 수정할 대상은 새 서버 준비와 J0→λE→JQ 전환 경로다. P/D 등의 후속 정책이 아직 없는 사실과, 지금 필요한 C0 운영 결함은 구분해야 한다.

**검증 범위와 시점**

검토 HEAD는 `3d06885`, 실제 s1 run 기록은 `cac23a6efb168e47714d9194aee843df484d5b36`이다. 관찰·재현 검사는 2026-09-14 13:32–13:43 KST에 수행했다. 파일 증거는 13:43:16 KST에 [snapshot](PAKD50_Audit_2026-09-14/snapshot)에 복사했다. 실행 파일을 여러 개 순서대로 읽은 snapshot이므로 원자적인 전체 학습 상태 저장점은 아니다.

- 두 계획 문서 전체, C0 구현 노트, config 생성기, 12개 stage 1 config, 큐, prepare/gate/runner, trainer/forward/loss/calibration/selector/평가·export 경로를 대조했다.
- s1 실제 학습 PID `2559101`과 RTX 4090 GPU 사용을 읽기 전용으로 확인했다. 관찰 당시 GPU 사용률은 85%, 사용 메모리는 6,731 MiB였다. 다른 같은 명령의 PID 4개는 그 학습 프로세스의 DataLoader 자식이었다.
- s2/s3의 실제 프로세스·수신 파일·run manifest에는 접근하지 못했다. 공통 배포 코드와 s2/s3용 config 검증을 그 서버의 실제 기동 확인으로 표현하지 않는다.
- 학습 코드·큐·config·calibration·가중치를 수정하지 않았고, 학습을 중단·재기동하지 않았다. 외부 업로드도 실행하지 않았다. 이번에 추가한 것은 감사 보고서와 검증 증거뿐이다.

| 실제 확인 항목 | 13:43 snapshot 기준 |
|---|---|
| 실행 중 | `PAKD50_J0_W112_D123_WV3_T0_S1234_FRESH50_v1` |
| 학습 진행 | throughput 기록 completed update 9,090 / 50,000 |
| 당시 공식 selected | update 8,080, raw HQNR **0.9372703866544423**, RR ERGAS **2.2193615423616593** |
| T0 | PALS24 L1E4 seed2025, 같은 `best_hqnr`의 A/U, selected update 24,240 |
| T0 checkpoint SHA-256 | `16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32` |
| T0/Student 초기 A tensor hash | `493aca181f93ae68` |
| Student U 초기 tensor hash | `c988a6c95b17f4fd` |
| τR | **0.012463942170143127** |
| λE | 아직 없음. J0-1234 exact50K가 없으므로 이 시점에는 정상 |
| 현재 확보한 통합 paired 결과 | 없음. J0도 진행 중이며 JQ는 미실행 |

위 Student 지표는 초기 학습 관측값이다. Teacher 최종 성능과 직접 비교해 방법의 성공·실패를 결정할 시점이 아니다. raw 0.959/0.960 달성이나 3-block 반복 이득은 아직 판정할 수 없다.

**확인된 결함과 우선순위**

P1은 다음 배포·실험 단계 전에 수정할 문제, P2는 재현성·해석에 필요한 보완이다. ‘조건부’는 코드의 결함은 확인했지만 해당 장애가 현재 run에서 발생했다는 뜻은 아니다.

| ID | 우선순위 | 판정 | 영향 |
|---|---|---|---|
| F01 | P1 | 새 서버 bootstrap 순서 오류, 재현 | 준비 명령만으로 stage 1을 시작할 수 없는 경로 |
| F02 | P1 | pilot 완료 이벤트와 큐 연결 오류 | JQ가 F0/JR/FR 뒤로 밀림 |
| F03 | P1 | 공통 calibration 전달·갱신 오류, 재현 | s2/s3 JQ/FQ가 닫히거나 오래된 package 사용 가능 |
| F04 | P1 | unit test가 live gate 호출, 재현 | 검사 자체가 calibration/config를 변경하거나 단계 전환 검사 실패 |
| F05 | P1 | 단일 deadline·core 예약 미보장 | 50h와 필수 paired run 우선 확보 규약 불충족 |
| F06 | P1, 장애 시 | exact resume 미구현, 재현 | 재개 run의 데이터 대응 및 checkpoint grid 손상 |
| F07 | P2 | 실행 worktree 미분리 | 후속 release 시 실행 source 고정 보장 없음 |
| F08 | P2 | loss별 A gradient 진단 미구현 | P/JK0/JE0를 고르는 핵심 근거 부족 |
| F09 | P2 | fitting bin 경계의 자료원이 계획과 다름 | train calibration bin으로 설명하면 부정확 |
| F10 | P2 | package·환경 고정 및 결과 인계 미완성 | 서버 비교·재현·최종 주장 근거가 아직 부족 |

**F01 — 새 서버가 필요로 하는 τR 파일을 만들기 전에 파일 존재를 검사한다.**

근거: [prepare:43](/home/knuvi/Desktop/song/PAN-Crafter/tools/pakd50_prepare.sh:43), [unit test:86](/home/knuvi/Desktop/song/PAN-Crafter/tools/pakd50_unit_tests.py:86). 계획 §5.1, §5.4, §13.5.

`prepare`는 `pakd50_unit_tests.py`를 먼저 실행하고, 그 뒤 45행에서 `assets/pakd50/calibration_resolved.json`을 로컬 `work_dir/_pakd50/`로 복사한다. 그런데 K05는 바로 그 로컬 파일에 τR가 있는지 필수 검사한다. 새 서버에 로컬 calibration이 없으면 `calj={}`이고 검사 결과는 false다. `prepare`는 실패 즉시 종료하므로 복사·τR 재계산 단계까지 도달하지 못한다.

실제 K05 구문을 추출해 ‘로컬 calibration 없음’ 상태에서 실행한 결과 `None / 'None'`으로 실패함을 재현했다. s1의 통과는 이미 로컬 calibration이 있던 상태의 증거다. s2/s3에서 실제로 실패했는지는 원격 로그가 없어 미확인이다.

수정 기준: 먼저 신뢰할 공통 τR package를 설치·대조한 뒤 package 의존 검사를 수행해야 한다. 빈 `work_dir`에서 `prepare --no-start`가 통과하는 검사를 추가해야 한다. 옛 캠페인 gate도 로컬 옛 donor 자산 유무와 무관하게 새 T0 package만으로 통과할 수 있는지 함께 확인해야 한다.

**F02 — J0가 끝나도 calibration gate가 호출되지 않는다.**

근거: [runner:123](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:123), [generator:26](/home/knuvi/Desktop/song/PAN-Crafter/tools/gen_pakd50_configs.py:26), [gate:335](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_gate.py:335). 계획 §5.3, §9.3–9.4.

실제 runner는 `ORDER`의 모든 run을 처리한 뒤에만 `campaign_gate.py`를 호출한다. stage 1은 `J0 → F0 → JR → FR`이므로 실제 순서는 다음과 같다.

```text
J0 → F0 → JR → FR → λE calibration → JQ → FQ → XJ
```

구현 노트의 ‘J0-1234 완료 ≈15:20 → λE 고정 → stage 2’와 맞지 않는다. 구현 자체의 1.90h/run 가정만 적용해도 pilot 이후 약 5.7h가 더 지난 뒤에야 λE를 계산한다. 이는 예측값이며 현재 실제 완료시간은 아니다. R1/F0를 λE 대기 중 실행하는 계획의 허용과, s1에서 λE 계산 자체를 3개 run 뒤로 미루는 것은 다른 동작이다.

수정 기준: s1 pilot 완료·exact50K 확인을 독립 dependency로 연결해 즉시 calibration을 수행하고 JQ를 다음 필수 run으로 편성한다. FR은 frozen 유력성 또는 잔여 예산 근거가 있을 때 추가한다. 현재 진행 중인 J0의 수치 경로는 바꿀 필요가 없다.

**F03 — s1에서 계산한 새 λE가 s2/s3 로컬 package에 도달하는 경로가 완성되지 않았다.**

근거: [calibrator:16](/home/knuvi/Desktop/song/PAN-Crafter/tools/pakd50_calibrate.py:16), [prepare:45](/home/knuvi/Desktop/song/PAN-Crafter/tools/pakd50_prepare.sh:45), [generator:20](/home/knuvi/Desktop/song/PAN-Crafter/tools/gen_pakd50_configs.py:20), [gate:344](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_gate.py:344). 계획 §5.2–5.3, §13.5.

calibrator가 쓰는 위치는 s1의 `work_dir/_pakd50/calibration_resolved.json`이다. gate는 stage 2 config를 생성하지만, `assets/pakd50/calibration_resolved.json`에 새 package를 반영하는 절차는 없다. 별도 수동 반영이 필요하며 자동화 완료로 볼 수 없다.

더 직접적인 문제는 s2/s3가 stage 1을 거친 경우다. 로컬에는 τR만 있는 JSON이 이미 존재한다. 새 λE가 들어 있는 assets 파일을 받았다고 가정해도 prepare의 `[ -f local ] || cp ...`는 복사를 건너뛴다. 생성기는 assets가 아닌 로컬 JSON만 읽어 stage 2를 거부한다. 실제 copy 구문을 임시 디렉터리에서 실행해, 새 assets에 λE가 있어도 로컬에 τR만 남는 것을 재현했다.

또한 gate의 config 생성은 `if not cal.get('lambda_E')` 안에만 있다. 수동으로 λE만 먼저 계산했거나 config 생성이 한 번 실패하면, 다음 호출은 λE가 있다는 이유로 생성 단계를 건너뛴다. `emit`은 config가 없어 run을 열지 않는다. ‘calibration 완료’와 ‘config 생성 완료’를 각각 검사해야 한다.

수정 기준: 공통 package를 immutable revision/hash로 전달하고, 로컬 기존 파일과 비교해 명시적으로 갱신·검증해야 한다. 생성기는 그 package hash를 config에 연결하고, 이미 계산된 λE가 있어도 누락 config를 다시 생성할 수 있어야 한다. 생성 실패도 prepare에서 즉시 처리해야 한다. 현재 `set -uo pipefail` 및 55행은 생성 실패 직후 종료를 보장하지 않는다.

**F04 — ‘gate가 기본으로 닫혔는지 검사’하는 코드가 실제 캠페인 gate를 실행한다.**

근거: [unit test:90](/home/knuvi/Desktop/song/PAN-Crafter/tools/pakd50_unit_tests.py:90), [enabled_gates:376](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_gate.py:376). 계획 §14.1의 검사 격리, §6.3.

테스트는 환경변수 `PANCRAFTER_CAMPAIGN_GATES=''`로 subprocess를 실행한다. 하지만 `enabled_gates()`는 환경변수가 빈 문자열이면 `work_dir/campaign_gates_enabled.txt`를 읽는다. 현재 이 파일에는 `pakd50`이 있다. 실제 함수 호출에서도 반환값이 `['pakd50']`임을 확인했다.

현재는 pilot 미완주라 stdout이 비어 우연히 ‘닫힘’ 검사가 통과한다. pilot 완료 이후에는 이 테스트가 λE 계산·config 생성을 유발할 수 있고, stage 2 tag가 출력되면 test가 실패한다. 따라서 동일 검사가 실행 시점에 따라 부작용과 결과가 바뀐다. `prepare --stage 2`도 이 검사에 걸린다. 기존 PALS24 테스트에도 같은 빈 환경변수 호출이 있다.

이번 감사에서는 unit test를 읽기 전용으로 재실행하기 위해 **그 subprocess만 존재하지 않는 gate 이름 `audit_disabled`로 격리**했다. 그래서 얻은 24개 검사 통과를 ‘원본 prepare 전체의 무수정 통과’로 보고하지 않는다.

수정 기준: gate 테스트에 독립 fixture root/token 파일을 주거나 순수 판정 함수만 검사해야 한다. 테스트 실행이 live calibration·config·큐를 쓰지 않는 것을 확인해야 한다.

**F05 — 50시간 공통 마감과 J0/JQ 대응 block 예약을 코드가 보장하지 않는다.**

근거: [prepare:69](/home/knuvi/Desktop/song/PAN-Crafter/tools/pakd50_prepare.sh:69), [campaign_start:47](/home/knuvi/Desktop/song/PAN-Crafter/tools/campaign_start.sh:47), [generator:72](/home/knuvi/Desktop/song/PAN-Crafter/tools/gen_pakd50_configs.py:72), [budget gate:274](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:274). 계획 §9.1, §9.4, §11.2.

각 서버 prepare가 자기 실행 시각을 campaign start로 기록한다. s1의 동일한 절대 start/deadline을 받는 필드가 없다. 따라서 s2/s3가 나중에 준비되면 서로 다른 50시간 창을 운영한다. s1에서는 준비가 끝난 13:22:31을 start로 잡아 앞선 구현·검사 시간이 본문 계획의 0–4h 구간에 포함되지 않았다. 이 시점부터 시계를 재정의하려면 명시적인 캠페인 변경으로 기록해야 한다.

stage 2를 `prepare --stage 2`로 시작하면 로컬 ledger start는 유지되지만 `campaign_start.sh`가 마감을 다시 `현재 시각 + 46h`로 쓴다. stage 2 명시 큐의 run은 개별 시작 시 wall-clock 종료 예상 검사를 받지 않는다. 뒤에서 호출되는 조건부 gate가 과거 ledger의 46h를 보더라도 이미 늘어난 명시 큐 실행을 소급 차단하지 못한다.

생성된 모든 config는 `remaining_mandatory: []`, `pair_with: null`, `required: false`이고 자기 run의 4h만 예약한다. J0/JQ 3-block 확보, F0/FQ 대응 완결, 최종 확인 묶음은 예약하지 않는다. `required=true`로 바꾸는 것만으로 해결되지 않는다. 현재 trainer에서 그것은 예산 초과를 강제로 통과시키는 의미이므로, 필요한 것은 전체 남은 core와 대조의 예약이다.

처리량 추정에도 별도 보완이 필요하다. prepare는 전체 smoke 로그의 마지막 `t_native`와 마지막 `t_corrupt`를 합쳐 쓴다. 현재 값은 **FR native 67ms와 JR offset 73ms**로 서로 다른 case다. 평가시간은 실측값을 읽지 않고 67초를 고정한다. 200–500 update 및 data wait를 포함한 case별 실측이라는 §11.1과 다르다. `total_gpu_hours=46`에서 다시 reserve4를 빼는 trainer와, 46h까지 학습·이후4h 감사라고 설명하는 wall ledger의 예산 의미도 맞춰야 한다.

수정 기준: s1에서 하나의 절대 `campaign_start/training_deadline/final_deadline`을 확정·배포하고 모든 run 직전에 `now + 남은 학습·평가 예상시간 ≤ training_deadline`을 검사한다. 단계 변경으로 deadline을 연장하지 않는다. case별 실측과 mandatory/control 예약을 동일 ledger에서 관리한다.

**F06 — 자동 resume는 계획의 ‘정확한 이어 학습’이 아니다.**

근거: [main:252](/home/knuvi/Desktop/song/PAN-Crafter/main.py:252), [loader:159](/home/knuvi/Desktop/song/PAN-Crafter/main.py:159), [runner:84](/home/knuvi/Desktop/song/PAN-Crafter/tools/_run_cases.sh:84), [train step:1125](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:1125). 계획 §4.3, §13.6, §14.1.

코드는 이미 ‘배치 순서는 복원되지 않는 근사 재개’라고 로그에 명시한다. model/optimizer/scheduler 및 epsilon generator를 복원해도 RandomSampler의 epoch permutation·cursor와 worker augmentation 상태를 복원하지 않는다. 중간 checkpoint 뒤에는 새 loader iterator를 시작한다. 같은 DataLoader 규약으로 저장 RNG만 복원하는 최소 재현에서 다음 batch ID가 실제로 달라졌다.

게다가 epoch 안의 10,000-update checkpoint에서 시작하면 `last_epoch = 10000 // 202`이고 202개 batch를 새로 처리한다. 원래 10,100 update에서 해야 하는 epoch50 평가는 재개 경로에서 10,202 update가 될 수 있다. 이는 데이터 대응뿐 아니라 공통 후보 grid도 바꾼다. selector 상태 역시 선택 checkpoint 자체가 아니라 run 루트의 현재 JSON에서 복구한다.

`global_step`은 optimizer call 뒤 무조건 증가한다. AMP overflow로 step이 skipped되어도 성공 update와 epsilon parity를 유지하는 구현은 없다. **현재 s1 resolved AMP는 `no`이고 accumulation 기본1이므로 overflow 문제를 현재 발생한 결함으로 간주하지 않는다.** 다만 다른 서버의 AMP 설정을 허용한다면 현재 counter로 계획의 성공-update 정의를 충족할 수 없다.

수정 기준: 실제 loader 순서·worker/augmentation·epsilon·selector·completed update를 같은 저장점에서 복원하고, 다음 batch 및 optimizer update를 연속 실행과 비교해야 한다. 그 전까지 자동 재개된 run은 exact paired 결과와 분리하고 새 run으로 재실행하거나 비정확 resume 상태를 명시해야 한다. 현재 J0 snapshot의 `resumed`는 false다.

**F07 — s1은 실행 전용 고정 worktree에서 학습하지 않는다.**

근거: `git worktree list`, [실행 command](PAKD50_Audit_2026-09-14/snapshot/meta/command.txt), [campaign manifest](PAKD50_Audit_2026-09-14/snapshot/_pakd50/campaign_manifest_s1.yaml). 계획 §13.1, §13.4.

worktree는 메인 작업 디렉터리 하나뿐이다. 실제 명령도 이 디렉터리의 config를 사용한다. manifest에는 `git_dirty: true`가 있으며 source worktree 강제 검사나 code hash 검증 없이 시작한다. run 직후 HEAD는 `cac23a6`에서 `3d06885`로 바뀌었다.

두 revision 사이의 변경은 문서·결과 기록이며 현재 수치 source가 변경됐다는 증거는 없다. 따라서 이 사실만으로 현재 J0를 무효화하지 않는다. 그러나 앞으로 C1/C2를 개발하거나 pull하면 후속 subprocess와 다음 queued run이 다른 source를 읽을 수 있다. ‘다음 release는 run 경계에서 바꿔도 된다’는 구현 노트의 표현은 계획의 release별 고정 worktree 규약을 충분히 보장하지 않는다.

수정 기준: 다음 run/release부터 고정 SHA의 별도 실행 worktree를 사용하고, 별도 개발 checkout에서 수정한다. 현재 live run의 디렉터리를 이동하거나 중단하는 방식은 피한다. 현재 run의 code SHA와 수치 source bytes는 이번 감사의 [provenance](PAKD50_Audit_2026-09-14/provenance.json)에 기록했다.

**F08 — JQ의 정합 적응 원인을 구분하는 gradient 진단이 부족하다.**

근거: [live diagnose:935](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:935), [hard/soft 분해:1014](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:1014), [fixed diagnose:1043](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:1043). 계획 §7.2–7.3, §14.3–14.4.

현재 live 진단에서 A의 reconstruction gradient는 `loss_rec=L0+LD+LK` 전체다. hard/soft 분리는 backbone에 대해서만 계산한다. A의 LD/LK 각각의 norm·`cos(g0, gj)`, weighted edge와 g0의 cosine, `∂L0/∂cS`, `∂LD/∂cS`, `∂LK/∂cS`, `∂LEw/∂cS`의 y/x 성분은 없다. 현재 `dLrec_dDelta_rms` 한 값으로 대체할 수 없다.

고정 batch 진단은 reconstruction과 offset만 다룬다. JQ의 edge/soft/hard 추가항을 각각 기록하지 않는다. 이번 CPU 실행에서도 실제 fixed JSON keys가 이 범위에 한정됨을 확인했다. J0에는 학습 forward에서 Teacher가 없으므로 현재 fixed/live cS−cT drift도 기록되지 않는다. 평가의 fixed-reference HQNR는 이 초기 학습 drift의 대체 자료가 아니다.

현재 scalar·coefficient/loss/일부 backbone gradient 비율은 유용하다. 그러나 이 로그만으로 ‘soft가 aligner와 충돌하므로 JK0’, ‘edge가 원인이므로 JE0’라고 결정할 근거는 확보되지 않는다.

수정 기준: 작은 고정 batch에서 L0/LD/LK/LEw/LO를 A/U/cS별로 분해하고, 근처 norm0이면 cosine을 undefined로 남긴다. 진단이 학습 RNG·gradient·optimizer를 바꾸지 않는 검사는 유지한다. 초기 시점 근거가 필요하므로 최종 결과 보고 단계까지 미루면 놓치는 정보가 있다.

**F09 — fitting bin은 train calibration이 아니라 RR test에서 다시 정한다.**

근거: [test_reduced:1208](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:1208), [fitting_bins:1240](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:1240). 계획 §14.4.

계획은 고정 train calibration 자료의 Teacher error q50/q90를 공통 경계로 사용한다. 실제 구현은 RR20의 eT에서 매 평가 `torch.quantile`을 다시 계산한다. 실제 pixel bin은 q50 **0.0113905845**, q90 **0.0296976101**이고, train calibration은 q50 **0.0124639422**, q90 **0.0385872602**다.

같은 stationary Teacher·같은 RR20이므로 중단 없는 같은 조건의 run 사이 경계가 대체로 같다는 장점은 유지된다. 하지만 계획에서 정의한 train difficulty bin은 아니다. 학습 목적함수의 τR가 틀렸다는 뜻도 아니다.

수정 기준: calibration package에 q50/q90와 자료원 hash를 저장해 모든 run의 bin에 적용하거나, RR-test-relative bin으로 protocol 변경을 명시하고 계획의 bin과 구분한다.

**F10 — 서버 간 package 고정과 최종 인계 체계는 아직 검증 완료 상태가 아니다.**

근거: [config 생성기:45](/home/knuvi/Desktop/song/PAN-Crafter/tools/gen_pakd50_configs.py:45), [calibration batches:22](/home/knuvi/Desktop/song/PAN-Crafter/kdv/calibration.py:22), [run manifests:559](/home/knuvi/Desktop/song/PAN-Crafter/train_kdv.py:559), [후처리:39](/home/knuvi/Desktop/song/PAN-Crafter/tools/_upload.sh:39). 계획 §5.4, §13.5, §15.

현재 잘 보존하는 항목은 T0 file hash·selected update, A hash, U init hash, 데이터별 file hash, run config snapshot, AMP resolved 값이다. 이번 감사에서 실제 optimizer checkpoint를 읽어 s1 Student와 T0가 betas `(0.9,0.999)`, eps `1e-8`, WD `.01`, U/A initial LR `1e-4/1e-5`를 사용함을 추가 확인했다.

그러나 다음 항목은 보완이 필요하다.

- calibration subset은 index hash와 처음 8개 index만 기록한다. 전체 patch ID/order 목록은 저장하지 않는다. seed와 데이터 길이로 재생 가능하지만 명시적 목록 저장이라는 계획과 다르다.
- 생성기는 package 안의 τR/λE 숫자를 읽으며 현재 T0/pilot의 hash와 package source를 매번 대조하지 않는다. 양수가 있는지만 확인하는 것은 package 검증과 다르다. N0는 τR를 config에 pin하지 않아 평가 bin용 τR를 run 초기화에서 다시 계산한다. s1에서는 공통값과 같았지만 서버별 고정 정책으로는 통일되지 않는다.
- optimizer beta/eps·minimum LR·AMP/accumulation이 완전한 resolved config로 pin되지 않았다. 특히 `mixed_precision: null`은 환경변수에 영향을 받을 수 있다. 현재 s1의 값은 확인했으나 다른 서버도 같다는 증거는 없다.
- Teacher U hash는 전체 checkpoint hash로 간접 확인할 수 있지만 독립 component hash/공통 초기 U 파일 배포·cross-server 검증 기록은 없다. 데이터 hashes도 각 run에서 저장할 뿐 공통 예상값과의 launch 비교는 없다.
- `run_key.json`의 grid ID는 `GRID1010_50K_v1` 대신 기존 일반 문자열이다. campaign manifest 및 frozen config로는 추적 가능하나 자동 집계가 그 필드를 신뢰하면 정식 grid 식별이 흐려진다.
- PAKD50 전용 paired summary·last/plateau·target-hit 검증·RR trade-off 경고는 아직 없다. `_upload.sh`에서 PAKD50은 일반 FR 재평가 대상에는 포함되지만 PA/offset response 전용 진단 분기에 포함되지 않는다. signed2×2 response, selected/last 동일 자산 평가·hash 인계는 추후 구현해야 한다.

이는 지금 없는 완주 결과를 실패로 판정하는 근거가 아니다. 다만 C0가 끝났다는 이유만으로 ‘최종 보고까지 자동 완료’라고 간주할 수 없다. s2/s3의 실제 full SHA, T0/calibration/data/init hash, GPU/AMP/worker 및 run 시작 기록을 받기 전에는 3개 server×seed block의 대응성을 승인할 수 없다.

**맞게 구현됐거나 허용되는 변경**

| 요구사항 | 검증 결과 및 근거 |
|---|---|
| W112–D123, 9ch→8ch residual | Student/T0 model_args 일치. U 2,658,888 parameters, A 105,330. W104/D122로 변경되지 않음 |
| 같은 T0 checkpoint의 A/U 사용 | Teacher와 donor가 같은 file SHA와 update24,240에 연결. 실제 A tensor hash 일치. N2 donor를 다시 Student A로 싣지 않음 |
| Teacher A/U frozen, 자기 정합 경로 | J의 `share_correction=False`; F는 A hash·margin·sampler가 같을 때만 공유. no-grad 및 별도 parameter 객체 확인 |
| Student는 A 복사, U 독립 초기화 | 저장된 seed1234 U init 사용. 실제 optimizer step4,040에서 moment counter도4,040으로 새 Student update 기준이며 Teacher step24,240을 이어받지 않음 |
| PAN 한 번 보정, M-frame, base 한 번 | `kdv_forward`와 `PAModel.forward` 출력 bitwise 일치. 입력 GT/MS/PAN 불변 검사 통과 |
| margin4→z-score, yx/xy·warp | 성공 Teacher와 같은 `PANGlobalAligner`, `PAModel`, `warp_pan`; crop 후 bandwise norm, bicubic/border/align_corners=False |
| Native + odd offset | I-AEQ에서 U는 native만 사용. index1의 offset은 component mean, radius2 disk, native cS만 stop-grad. offset→U gradient 없음 |
| N0/R1/Q12/X02 | 실제 `_step`의 scalar와 A/U parameter gradient를 별도 직접 수식과 대조. 8개 J/F case 모두 통과 |
| GT signed-edge | 최종 8band와 GT, Scharr /32, reflect1, interior1px, 0.5(x)+0.5(y). Teacher/PAN edge나 variance loss로 대체되지 않음 |
| J 전체 감독·F 고정 | JQ gradient에 LD/LK/edge 포함. F는 A를 optimizer에서 제외하고 AdamW step 후 hash·grad 불변 |
| τR | 새 T0의 train-only 3,072 patch, seed1234, batch48, augmentation/crop OFF. 과거 no-align 값을 복사하지 않음 |
| λE 공식·의존성 | calibrator는 J0-1234 `last_meta.step==50000` 확인 후 출력 gradient RMS ratio의 median×0.05 사용. 현재 미산출은 정상 |
| 50K schedule | 50K cosine horizon, warmup100, U/A LR10:1. 실제 scheduler step과 optimizer state가 해당 update와 일치. 현 경로의 cosine 최종 multiplier는0 |
| Raw-original FR20 | 실제 H5 20장. 정상 A+U 출력에 원 PAN reference 사용. 장면별 product 평균. raw/V64/aligned별 열 분리 |
| 선택 comparator | running max−1e-4 band, fSCC tie 규칙 계승. 저장된 초반7개 checkpoint에서 scene별 값을 재집계한 오차 ≤2.22e-16, selector replay 확인 |
| exact50K와 후보 보존 | final update에서 last 저장 및 grid 밖 최종 평가가 코드상 연결. 50K는 아직 도달 전이므로 실제 최종 산출물 검증은 남음 |
| 독립 진단의 영향 | JQ fixed diagnostic 실행 전후 CPU torch RNG, epsilon RNG, model hash, parameter `.grad` 불변 확인 |

T0 raw HQNR **0.9569763995761281** 재현은 이번 launch의 `t0_reproduction.json`과 과거 같은 SHA 재현 기록에서 확인했다. 이번 감사에서 GPU FR20 전체를 다시 추론한 결과로 과장하지 않는다. 새로 수행한 수치 검사는 CPU FP32의 실제 trainer loss/gradient 및 저장된 장면 지표 재집계다.

**평가 grid 변경은 숨겨진 버그로 분류하지 않는다.** 원 계획은 1,000배수 50개이고, C0는 `GRID1010_50K_v1={1010,…,49490,50000}`로 변경했다. 이유는 기존 epoch 단위 평가이며, 시작 전 release 노트/config/campaign manifest에 새 grid ID가 기록돼 있다. §6.2는 실행상 이유에 따른 사전 cohort-wide grid 변경을 허용한다. 실제 초반 평가도1010배수였다. 다만 수정된 plateau `[45450,46460,47470,48480,49490,50000]`는 원 계획의 exact45–50K가 아니므로 최종 표에 실제 시점을 적고, 모든 서버에서 이 grid가 같고 resume로 깨지지 않았는지 검사해야 한다.

**C1/C2 미구현 자체는 현재 C0의 수식 오류가 아니다.** P/JK0/JE0/D/LF/QSF/TCOPY/CONT, T1/XSRV, 최종 paired reporting은 구현 노트에도 후속 작업으로 분리돼 있다. 상세 계획 §13.1도 순차 release를 허용한다. 이 case 이름을 현재 trainer에 붙여 실행하거나 전체 정책이 이미 검증됐다고 보고하면 안 된다. AL0/ALQ의 LR config는 생성 가능하지만 실험 우위를 확인한 것은 아니다.

**검사 방법과 한계**

[검증 스크립트](PAKD50_Audit_2026-09-14/verify_readonly.py)는 실제 `KDVTrainer._step`을 호출하되 학습 runner의 전체 생성·예산·manifest 쓰기는 수행하지 않는다. s1 T0와 보존된 Student step4,040 상태, 실제 train patch 2개를 CPU에서 사용했다. 수식 비교용 λE=0.3은 **합성 테스트 상수**이고 정식 calibration이 아니다. 실제 캠페인의 λE는 변경하지 않았다.

8개 case `J0/JR/JQ/XJ/F0/FR/FQ/XF`의 독립 수식과 실제 loss/gradient 비교에서 최대 parameter gradient 절대오차는 **1.4901161193847656e-08**이었다. frozen AdamW·offset ownership·forward 동치·Teacher 불변·진단 격리도 통과했다. 기존 PAKD50 gate는 live gate subprocess만 격리한 상태에서 **24개 check**가 통과했다. 구현 노트의 ‘25 검사’와 실제 출력 개수에는1개 차이가 있으나 핵심 결함으로 분류하지 않았다.

감사 harness의32개 check 중 일부는 **결함이 재현되는 것을 통과 조건으로 둔 검사**다. 따라서 `32 PASS`가 계획 준수32개 통과라는 의미는 아니다. bootstrap 실패, stale calibration, 빈 gate 환경변수의 live token 사용, resume 배치 차이를 재현한 결과가 포함된다. [verification.json](PAKD50_Audit_2026-09-14/verification.json)에 각 이름과 실제 수치가 있다. bootstrap/live-token 검사는 첫 실행 후 별도 추출 구문 검사로 JSON에 보충했으며, 최종 스크립트에는 같은 검사가 포함돼 있다.

CUDA에서 모든 case의 여러 step 및 정확한 resume, 세 서버의 대응 데이터·augmentation 열, 독립 RR validation의 scene 분리까지 검증한 것은 아니다. `valid_wv3.h5` 1,080 patch와 train9,714 patch가 서로 다른 파일이라는 사실만으로 scene 수준 독립성을 증명할 수 없다. 현재 AMP는 `no`여서 CPU FP32 검사가 수치 경로를 상당 부분 확인하지만 CUDA kernel별 동치나 장기 수렴을 보장하지 않는다.

T0의 metadata ERGAS **2.0562435097**와 요약의 Sheet 표시 ERGAS **2.0694**는 여전히 다른 기록이다. raw 재현이 통과했다는 이유로 RR 표도 해결됐다고 보지 않는다. 새 Student의 ERGAS 비교는 이번 캠페인의 같은 evaluator·같은 selected/last/plateau로 수행하고, 기존 Sheet 값과 섞기 전에는 동일 checkpoint의 RR export 재평가로 차이를 해소해야 한다.

**실행자에게 필요한 조치 순서**

1. 현재 s1 J0의 학습을 보존하고, F01/F03/F04의 package 준비·갱신·검사 부작용을 먼저 수정한다. 빈 work_dir와 stage1→stage2 두 경로를 GPU 학습 없이 fixture로 통과시킨다.
2. F02를 수정해 J0-1234 exact50K 직후 λE를 고정·검증하고 JQ를 우선 실행한다. F0/JR/FR 전체 완료를 기다리는 큐는 주력 우선순위로 다시 구성한다.
3. 하나의 절대 deadline과 J0/JQ 대응 block·필수 control 예약을 배포한다. 단계 재시작으로 마감이 늘어나지 않는지 확인한다.
4. 다음 run은 고정 실행 worktree로 옮기고, s2/s3의 full SHA/T0/data/init/calibration/AMP 증거를 모아 같은 cohort임을 확인한다.
5. exact resume를 구현하거나 재개 run을 공식 paired 결과에서 분리한다. 초기 loss별 A gradient 진단은 JQ 정책 분기에 사용하기 전에 보강한다.
6. 선택된 하나의 method family에 대해 3-block paired summary, X02 귀속 대조, 실제 수정 grid의 last/plateau, selected/last FR20·RR 재평가를 완성한 뒤 target-hit·repeated-gain·method-level target을 각각 판정한다.

이 순서라면 현재 J0에서 얻고 있는 학습 자산을 유지하면서, 이후 실험이 계획의 질문에 답할 수 있도록 운영과 증거 수집을 바로잡을 수 있다.
