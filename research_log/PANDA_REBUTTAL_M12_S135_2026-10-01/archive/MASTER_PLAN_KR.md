# PANDA rebuttal 대비 — 제출본 고정, s1·s3·s5 6-repeat 단계별 실험 계획

작성일: 2026-09-28  
상태: **실험 진행 제안 및 case 명세. 서버 기동, queue 변경, 학습·추론은 수행하지 않음.**

## 1. 기준과 보관 방침

기준 문서는 이미 제출된 `59391_PANDA_PAN_Alignment_Awar.pdf`다. 제출본, Tables 1–3, 기존 `paper` 표 및 과거 실험 기록은 수정하지 않는다. 본 작업은 제출본의 주장을 질문별로 검토할 **추가 통제실험·재측정·대비 자료**를 확보하는 것이다. 원 제출값을 새 실험의 목표 수치로 삼거나 새 측정값을 과거 결과로 바꾸어 기록하지 않는다.

이전 감사 MD와 CSV, 원 ZIP은 `archive/`에 원문 그대로 복사했고 SHA256을 대조했다. 그 문서의 표 수정·표시 수정 권고는 과거 감사 내용으로 보존하되, **현재 실행 범위에는 포함하지 않는다.** 운영 기준은 이 문서와 `B01_RB01_RB02_DETAILED_KR.md`다. 본 묶음은 대화에서 회수 가능한 파일 보존본이며, 별도로 Drive/Project/GitHub에 저장했다고 주장하지 않는다.

PDF 및 XLSX의 SHA256은 `archive/archive_manifest.json`에 남겼다. 두 원본은 별도 첨부를 유지하며 이 ZIP에는 중복 포함하지 않았다.

새 결과는 `Submitted reference`와 `New rebuttal measurement`를 분리해 관리한다. 재현 출처·cost 계산 규약·학습 budget·선택점 대비는 내부 증거 목록으로 정리한다. 제출본 수정이나 rebuttal 문구 작성은 이번 단계의 작업이 아니다.

## 2. “두 step씩”과 “case당 6개”의 정의

여기서 `RB01`, `RB02`는 **실험 업무 단계**다. 논문 Method의 Stage 1/2 또는 Table 3의 Step I/II와 혼동하지 않는다. 두 단계를 한 batch로 진행하고, 전체 자료를 확인한 뒤 다음 두 단계를 정한다.

- 서버: **s1, s3, s5**. s2/s4는 건드리지 않는다.
- 같은 dataset·같은 case를 모든 서버에서 **서로 다른 Student seed로 2회씩** 수행한다.
- 따라서 **case당 총 6개의 Student 학습 결과**가 생긴다. 서버별로 case나 dataset을 나누지 않는다.
- 같은 서버·같은 repeat의 대조군은 **초기 U, 초기 A, Teacher, 입력/augmentation stream, optimizer, budget, selector**를 공유한다. 바꾸는 것은 등록된 case의 항목뿐이다.
- server는 Student seed와 완전히 교차된 통계 요인이 아니다. 서버별 paired delta도 확인하지만, 이 6개만으로 순수한 hardware 효과를 추정하지 않는다.
- 이미 등록된 과거 run은 대비용 자료로 보존한다. 새 여섯 슬롯을 과거 best run으로 채우지 않는다.

| paired block | server | repeat | Student seed |
|---|---|---:|---:|
| B1 | s1 | 1 | 9281101 |
| B2 | s1 | 2 | 9281102 |
| B3 | s3 | 1 | 9281301 |
| B4 | s3 | 2 | 9281302 |
| B5 | s5 | 1 | 9281501 |
| B6 | s5 | 2 | 9281502 |

이 seed는 이번 계획에서 미리 지정한 신규 seed이며 성능에 따라 바꾸지 않는다. 기술적 실패는 같은 seed·같은 case로 재개/재시도하고 attempt를 별도로 기록한다. 동일 seed 재시도는 추가 독립 반복이 아니다. 수치 발산·실패도 남기고 결과가 나쁜 run을 대신할 seed를 새로 추첨하지 않는다.

## 3. 첫 batch B01: WV3에서 RB01 + RB02

첫 batch는 논문 Table 3와 Fig. 5의 주장이 집중된 **WV3 한 센서**에서 시작한다. 이는 WV3만 최종적으로 보고하겠다는 뜻이 아니다. sensor를 추가할 때도 각 case를 세 서버 모두에서 두 번씩 수행한다.

한 번에 세 센서를 넣으면 RB01만 72개 Student run이다. 먼저 WV3의 24개 run으로 구현·반복·증거 형식을 확정하고, QB/GF2 확장 및 다음 batch는 검토 후 진행하는 제안이다. 다른 센서의 결과는 이번 WV3 여섯 반복에 섞지 않는다.

### RB01 — q의 sample별 정보와 e와의 상보성

질문: Fig. 5 및 Eq. (7), (10)–(12)의 q가 단지 감독 강도를 줄이는 역할을 넘어서 sample에 맞는 정보를 제공하는가?

| case | geometry weight | 기존 FULL과의 변경 | case당 결과 |
|---|---|---|---:|
| QFULL | 원래 `s(n,v)` | 없음 | 6 |
| QMEAN | train의 전체 sample×view 평균 `mean(s)` | sample별 정보를 없애고 평균 강도 유지 | 6 |
| QSHUF | 같은 augmentation view·e quintile 안에서 s의 sample 대응을 섞음 | weight 값들의 분포는 유지, q–sample 연결 변경 | 6 |
| QESUR | patch-mean e로 s의 순위를 배정 | s와 같은 주변분포를 갖는 reconstruction-error 대체 신호 | 6 |

네 조건 모두 기존 e-guided hard/soft fitting을 그대로 사용하고, geometry에 들어가는 weight만 바꾼다. QESUR의 e 대체는 **geometry 경로**에 한정되며 원래 pixel-level e, d, a 항을 변경하지 않는다.

**총 4 case × 2 repeat × 3 server = 24회 fresh Student 50K. 서버당 8회.**

### RB02 — 추가 PAN shift에 대한 정합 반응과 복원 성능

RB01의 **각 case에서 생성된 여섯 개 독립 Student**를 그대로 사용해 추가 학습 없이 평가한다. 같은 checkpoint를 두 번 추론해 Student seed 반복이라고 세지 않는다.

| 평가 조건 | 내용 | 해석 |
|---|---|---|
| A_ON | 정상 입력-level PAN correction | 정상 모델의 shift 대응 |
| A_ZERO_INFERENCE_ONLY | 같은 checkpoint에서 correction만 0으로 override | 추론 시 correction 의존성 진단 |

각 trained case마다 A_ON 6개 곡선과 A_ZERO 6개 곡선을 얻는다. **4 trained case × 2 mode × 6 model = 48개 curve record**, 서버당 16개다. A_ZERO는 별도의 학습 run이나 “alignment 없이 학습한 baseline”이 아니다. scratch/no-align 및 consistency 자체의 인과 대조는 후속 RB04/RB05에서 맡는다.

제안 stress grid는 0을 포함한 49개 shift다: HR pixel 반경 0.25, 0.5, 1, 2, 3, 4와 방향 8개. 원 학습 및 q probe E와 겹치는 구간을 기록하고, 대각선이라는 이유만으로 학습 분포 밖이라고 부르지 않는다. 정상 native RR/FR 지표와 stress RR 지표를 별도 표로 보존한다.

## 4. 고정할 학습 기준

논문용 WV3 모델과 연결된 **FH20R1 / PLH / W104D122 / F1**을 기준으로 삼는다. 기존 G23/P0/W104D121/T0, 최신 sensitivity 구현 또는 PAN-Crafter reproduction을 자동 대체하지 않는다.

- Student U: PLH, W104, depth `[1,2,2]`, LN, 기존 attention/mode modulation off.
- reference: `FH12_S1_T_P0_W112_D123_WV3_S71001_FRESH50_v1`, **F1 exact50K**.
- 세 서버의 reference weight, calibration, raw q, data/LP identity는 같아야 한다. 서버마다 새 Teacher를 임의 선택하지 않는다.
- Student는 모두 fresh U이며, A는 F1의 A를 독립 복사한 뒤 학습한다. 제출된 Student checkpoint에서 fine-tune하지 않는다.
- batch48, 50K update, AdamW, U LR1e-4, A LR3e-6, warmup100/cosine.
- alpha=1, beta=.1, lambdaE=.002. tau_rec와 q_ref는 원 F1 train calibration 원값을 사용한다.
- 원 native64/16 data, augmentation, LP/HP frontend, normalization, clamp 및 gradient routing을 유지한다.

**이 여섯 반복은 고정 F1에 대한 Student-seed 반복이다.** 여섯 개의 독립 Teacher–Student 전체 파이프라인 반복은 아니다. Teacher consistency/Teacher 일반화는 별도의 여섯 Teacher seed pair를 만드는 RB05에서 다룬다.

서버로부터 원 F1의 SHA, 실제 resolved config, reference/calibration manifest를 읽고 binding을 고정하기 전에는 `READY_TO_TRAIN`으로 표시하지 않는다. 현재 bundle은 이 원격 파일이 존재하거나 서버 환경이 검증되었다고 주장하지 않는다.

## 5. 실행 동등성 점검은 짧은 사전 검사로 처리

제출본의 표를 바꾸는 감사 작업을 새로 진행하지 않는다. 대신 재실험 자체가 다른 구현이 되지 않도록 다음만 확인한다.

1. 같은 초기 tensor·같은 고정 batch에서 원 FH20R1 baseline과 새 QFULL의 prediction, hard/soft/edge, U/A gradient, 1회 optimizer update를 대조한다.
2. initialization과 data stream의 digest는 동일 block 내 네 case가 같아야 한다. q 변형은 별도 RNG를 사용해 sampler RNG를 소비하지 않는다.
3. pause/resume 시 모델, optimizer, scheduler, sampler, RNG를 복원한다. 실제 CUDA smoke와 identity 결과를 기록한다.
4. 코드/수치 차이가 발견되면 새 측정 결과를 원 recipe 반복으로 합치지 않는다. 차이를 기록하고 정리한 뒤 시작한다. 기존 원고의 최고 점수에 맞게 숫자를 조정하는 검사가 아니다.

세 서버가 동일 물리 GPU일 필요는 없지만 runtime/precision 정책을 맞추고 실제 GPU·library 정보를 남긴다. 학습 난수는 고정해도 GPU 간 bit-identical 결과를 자동 보장하지 않는다.

## 6. 평가와 통계

신규 기여 대조의 primary는 **EXACT_50000**, secondary는 같은 run의 **RR validation ERGAS 최소**다. FR test 최고점을 골라 비교하지 않는다. 제출 모델의 기존 TARGET·선택 기록은 별도 역사적 결과로 보존하며 변경하지 않는다.

RB02는 EXACT_50000 checkpoint만 사용한다. secondary가 exact와 다른 경우 native RR/FR 결과만 추가하고 6개 독립 반복 수를 늘리지 않는다. 같은 SHA인 selection은 alias 처리한다.

기본 지표는 HQNR, D_s, D_lambda, ERGAS, SAM, PSNR, SCC, SSIM, Q8, 보조 RMSE/CC/JQM이다. stress에서는 추가 shift 크기별 RR ERGAS/PSNR/SAM/edge error 및 relative correction response를 저장한다.

주 결과는 각 case의 여섯 run 평균±표준편차와 **같은 seed의 QFULL 대비 paired delta**, 개선한 block 수/6, 서버별 두 delta를 함께 기록한다. 20개 장면×6 seed 또는 49개 shift를 독립 학습 반복 120회/294회로 부풀리지 않는다. scene 표본과 학습 seed 변동은 분리한다. 고정 F1과 기존 test 데이터 조건의 결론임을 유지한다.

판정은 양의 결과에만 “성공” 표시를 붙이는 방식이 아니다. n, 기술적 완료 상태, 효과 방향, RR/FR trade-off, 대표적 실패 양상을 모두 정리하고 질문별로 어떤 설명을 지지하는지 기술한다. complete 6개가 확보되지 않으면 실제 n과 실패 사유를 보고한다.

## 7. 운영: 같은 실험 묶음을 세 서버에 복제

각 서버에서 repeat1의 네 case와 그 RB02 평가를 마친 뒤 repeat2를 진행한다. 원시 초기값은 미리 고정하므로 QFULL이 항상 첫 번째일 필요는 없다. case 순서는 `planning/server_schedule.csv`에 미리 정의했다.

각 run의 native 평가 후 RB02를 수행하고 다음 case로 넘어가는 구성을 기본으로 한다. 가벼운 metric/ZIP 처리는 GPU를 해제한 뒤 실행할 수 있다. 다른 서버의 완료를 기다리는 공유 GPU lock이나 중앙 runtime dependency는 두지 않는다.

**서버별 8개 학습 + 16개 stress curve record가 끝나면 `STOP_FOR_REVIEW`다.** 이번 요청을 무한 반복으로 해석하지 않는다. 다음 seed cohort나 RB03/RB04는 검토 후 별도 실행 지시가 있어야 한다.

현재 s1/s3/s5 가용성은 사용자의 전달 사항이며 직접 프로세스를 점검한 것은 아니다. 기동 담당자는 실 GPU owner를 먼저 확인한다. 알려지지 않은 학습/runner를 강제 종료하거나 s2/s4 queue를 바꾸지 않는다. 기존 작업과 충돌하면 자원 인계 상태를 기록하고 승인된 경로로만 전환한다.

시간은 이번 환경에서 측정하지 않았으므로 고정 소요시간을 약속하지 않는다. 한 서버당 기본 예상치는 `8 × (fresh50K + native 평가 + 두 stress mode)`이고, 첫 완료 run의 실제 시간으로 잔여 예산을 산출한다. 목표 score나 경과시간으로 불리한 case를 조기 종료하지 않는다.

## 8. 후속은 두 step씩, 지금 자동 실행하지 않음

| batch | step | 질문과 대비 결과 | 여섯 반복의 구현 |
|---|---|---|---|
| B01 | RB01 | q 정보성: FULL/mean/shuffle/e-surrogate | 4 case×6 fresh Student |
| B01 | RB02 | correction response와 shift robustness | RB01 각 case의 6개 Student에서 paired inference |
| B02 | RB03 | advantage gate와 q-edge/q-A, A gradient routing | no-a, q-edge-only, q-A-only, A에 soft+edge gradient 허용; 각 새 case×6 |
| B02 | RB04 | input-align vs feature-align vs trained no-align | 용량·frontend·loss 대조를 먼저 고정; 각 trained case×6 |
| B03 | RB05 | shift consistency와 Teacher 조건 변화의 효과 | T_CON_ON/OFF 각6개의 실제 Teacher training, 같은 Teacher seed pair 비교; 필요한 Student arms도 각6 |
| B03 | RB06 | WV3→WV2 zero-shot 및 native/shift 일반화 | 사전 고정한 각 case의 6개 WV3 Student를 s1/s3/s5 각2개씩 전이 |
| B04 | RB07 | q_ref/tau_rec/r_A scale sensitivity | 필요한 OAT 변형 각각6; alpha/beta/edge 기존 자료는 보존 |
| B04 | RB08 | 성능·비용·budget 비교 대비 | 동일 profiler의 모델당 서버별2 session; 필요 동일-budget 새 학습은 case당6 |

QB/GF2 확장에서도 server별 담당 센서를 나누지 않는다. 같은 sensor-case를 s1/s3/s5에 모두 배치한다. **한 센서의 여섯 결과가 다른 센서의 반복을 대신하지 않는다.** 센서 확장 범위와 실행 순서는 B01 검토 후 고정한다.

RB03에서 QFULL/QMEAN이 완전히 같은 source·Teacher·stream·budget이면 B01의 각 6개 결과를 재사용한다. 검증 후 재사용할 수 있는 baseline을 의미 없이 재학습하지 않는다. 제출 표의 한 모델만 세 서버에서 복제 실행하는 것은 이러한 6-seed baseline을 대신하지 못한다.

RB04는 feature-align을 구현하는 과정에서 MS frame이나 backbone이 달라질 수 있다. 단순 A_ZERO inference를 trained no-align 또는 late-alignment baseline으로 표기하지 않는다. shared frontend 변경이 필요하면 그 변경을 공유한 early/no-align baseline도 **각6개 새로** 만든다. 지금 1종 추가만으로 정당한 비교가 된다고 가정하지 않는다.

RB05의 Teacher 반복과 B01의 fixed-F1 반복은 통계적으로 다른 집합이다. 이미 있는 여러 Teacher의 결과는 보조로 정리하되, 새 case의 server별 2회 조건을 충족하지 않는 결과를 억지로 새 여섯 슬롯에 넣지 않는다.

RB08의 profiling session 6개는 **측정 반복**이고 서로 다른 학습 seed 6개가 아니다. 이미 있는 GF2 CTRL/MIX, 비교 모델 및 제출 checkpoint 재평가 자료는 그대로 보존한다. raw 재측정·예측 확보는 학습이 필요 없는 항목으로 구분한다.

## 9. 보존할 결과와 다음 판단

새 결과 namespace 제안: `work_dir/_panda_rb/20260928/B01/` 및 별도 `RB-B01-WV3` 결과 표. 기존 `paper`, `ablations`, 원 run의 summary는 덮어쓰지 않는다. 실제 Sheet 생성/업로드는 기동 지시에서 승인된 범위만 수행한다.

각 run에 initial U/A hash, Teacher/reference/calibration/data/stream SHA, q mapping SHA, resolved config, full-state resume, EXACT50K 및 val-selected identity, native per-scene metrics, stress per-scene metrics, raw prediction, timing, failures를 보존한다.

B01 검토 산출물:

1. q 네 조건의 n=6 mean±SD 및 paired delta 표.
2. q-shuffle/e-surrogate의 분포 보존 검사와 실제 consumed sample weight 통계.
3. A_ON/A_ZERO의 shift 곡선, native 성능과 stress 성능의 분리 보고.
4. e 조건부 q 진단, gate 활성 비율 및 U/A gradient/update 진단.
5. 서버별 일관성, 실패/발산·out-of-range shift 사례, 실제 계산시간.

자료가 완성되면 **B02(RB03/RB04)를 그대로 갈지, 먼저 QB/GF2에서 동일 대조를 확장할지** 결정한다. 특정 결과를 보고 수식·grid·seed를 바꾸는 경우에는 기존 cohort를 보존하고 새 revision으로 분리한다. 이번 bundle에는 실제 학습 runner나 실행 완료 결과가 포함되어 있지 않다.

## 10. 문헌/코드 근거와 제안의 구분

- 제출본 p2–3: align-first, 두 reliability cue의 역할. p5 Eq.(2): relative-shift consistency. p6 Eq.(7): q는 절대 정합오차가 아님. p7 Eq.(9)–(12): hard/soft/edge 및 U/A 분리. p8: 50K·batch48·LR·계수.
- 이전 감사 MD §6: q 정보성, controlled shift, routing, early/late, WV2, calibration/비용 검증 제안. 원본은 `archive/` 참조.
- 코드 확인 snapshot: `59abeff91df4c55acd0a9a029027c6925c62bcc8`, `fh12/model.py`, `fh12/losses.py`, `fh12/training.py`, `pa/aligner.py`, 원 FH20R1 config. 이는 조회한 snapshot이지 원 학습 release를 직접 인증한 것이 아니다.
- 서버 배치, seed, case 순서, stress grid, EXACT50K primary는 **이번에 제안한 새 실험 protocol**이다. 제출본에 이 protocol이 이미 있다고 주장하지 않는다.
