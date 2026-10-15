# 최근 2시간 Markdown 전체 구현 감사 — 2026-09-15

**판정: 기본 학습 수식과 새 W104 모델 경로는 대체로 계획대로 구현됐다. DCR 사분면 판정과 s3/s4 재배정 절차는 그대로 사용하기에 부적합하다. W104의 원본 구조 확인 및 아키텍처 간 data RNG 분리도 적용 전에 해결해야 한다.** 현재 s1의 JK0 학습 자체가 잘못됐다고 판정한 것은 아니다. 학습 결과를 얻는 경로와 그 결과로 과학적 결론을 내리거나 다음 실험을 편성하는 경로에서 발견한 문제를 구분한다.

검증 기준은 최종 commit `d616740e4b9188218f93b736394d6224b674534e`. 감사 시작은 **2026-09-15 10:37:07 KST**, 대상 시간창은 **08:37:07–10:37:07 KST**로 고정했다. 작업 도중 추가된 s4 구현 commit까지 반영해 재검증했다. 운영 코드·활성 큐·원장·Sheet를 수정하지 않았으며, GPU 학습을 시작하거나 중단하지 않았다.

## 1. 대상 문서와 commit/push 증거

`rg --files --hidden --no-ignore research_log -g '*.md'`로 하위 폴더까지 열거한 뒤 파일 시간을 조사했다. 해당 시간창의 문서는 다음 **5개 전부**다. 이 파일시스템은 birth time을 제공하지 않아 생성 시각을 별도로 증명할 수 없으며, 아래는 실제 수정 시각 기준이다. commit 시각을 파일 생성 시각으로 대체하지 않았다.

| 대상 문서 | 수정 시각 KST | 구현 검증 결과 |
|---|---:|---|
| [DCR12 사분면 검증 계획](PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md) | 09:17:45 | B0/B1 학습 정의 통과. D01 fixed bin, D04 gate/통계, 판정·운영 계약 불일치 |
| [DCR12 구현 노트](2026-09-15_dcr12-implementation.md) | 09:43:55 | 일부 축소 구현은 명시했으나, patch-only gate 차단·고정 bin·완료 검사 주장은 실제 코드와 다름 |
| [s3 추가 실험 계획](PAN_PAKD50_Latest_Sheet_Analysis_and_S3_Experiments_2026-09-15.md) | 10:19:07 | 5개 recipe·순서·예산 산술 통과. 인계/확인 예약/부족 예산 처리 불일치 |
| [s3 추가 구현 노트](2026-09-15_pakd50-s3-additions-implementation.md) | 10:24:07 | LFX 및 후보별 확인 묶음 구현됨. mandatory 등록 및 안전한 runner 전환 주장은 불충분 |
| [s4 W104–D121 계획](PAN_PAKD50_S4_W104D121_Architecture_Allocation_2026-09-15.md) | 10:34:59 | 감사 도중 구현됨. 5개 모델/설정·4개 확인 묶음 통과. source config, RNG, init identity, 기존 확인 예약 대체에 공백 |

감사 도중 생성된 [s4 구현 노트](2026-09-15_pakd50-s4-w104d121-implementation.md)는 위 5개에 대한 **추가 구현 증거**로 읽었다. 이 문서를 원래 시간창에 소급 포함하지 않았다. 기존 Integrated 계획/요약, EQREC4 및 파생 배분 감사는 상위 정의·공통 구현을 확인하는 자료로 사용했다.

| commit | commit 시각 KST | 내용 | 로컬 origin/main reflog |
|---|---:|---|---|
| `61ca248` | 09:44:10 | DCR12 및 JK0 두 seed 구현 | 아래 `4ea38e3`의 조상으로 포함 |
| `4ea38e3` | 10:24:50 | s3 LFX·편성·4321 확인 묶음 | 10:26:17 `update by push` |
| `d616740` | 10:48:56 | s4 W104–D121·NA0·두 구조 확인 묶음 | 10:50:18 `update by push` |

이는 **로컬 Git의 push 기록**이다. s3/s4 서버가 pull했고 해당 release로 실행 중이라는 증거는 아니다. 세 계획 MD 자체는 감사 시점 untracked였고, 관련 구현 노트는 tracked였다. 최종 소스 사본과 SHA256을 보존했다. 도중에 generator와 unit test가 서로 다른 시점으로 복사된 최초 검사는 폐기 판정하고, commit 고정 뒤 다시 실행했다. 초기 W104 미구현 판단 및 중간 unit 실패를 최종 결함으로 세지 않았다.

마무리 시점(11:07 이후)에 다른 작업의 SMEC12 구현이 `gspread/sheet_categories.py`와 `tools/_upload.sh`를 추가 수정했다. 원래 시간창 이후의 별도 계획이다. 변경 diff를 따로 보존하고 PAKD config들의 범주 분류 동일성 및 셸 구문을 확인했다. PAKD 실행명의 기존 매칭 분기는 바뀌지 않았고 본문의 학습/gate/재배정 지적에도 영향을 주지 않았다. 본문의 전체 수치 검사는 **10:52–10:56의 d616740 고정본** 기준이며, 이후 SMEC12 전체를 검증했다는 뜻은 아니다. [마지막 변경 확인](Recent_MD_2H_Audit_2026-09-15_103707/closing_delta/check.json).

## 2. 수행한 검사와 한계

| 검사 | 결과 | 의미 |
|---|---:|---|
| 기존 PAKD 단위검사 | 92/92 통과 | 최종 W104 추가분과 기존 recipe 회귀 검사 포함 |
| 기존 DCR 단위검사 | 13/13 통과 | 별도 출력 폴더에서 실행 |
| 추가 DCR/s3 적합성 검사 | 23/23 통과 | 위 두 suite의 성공 확인 2항목도 이 숫자에 포함 |
| 추가 W104 적합성 검사 | 22/22 통과 | config·예산·실제 모델·gradient·source 고정 확인 |
| 의도적으로 부적합한 입력/운영 상태의 재현 | 18/18 + 5/5 재현 | **PASS는 결함이 재현됐다는 뜻**. 중복 원인을 묶은 지적은 아래에 기술 |
| median 동률로 cell 소실 추가 검사 | 재현 | R가 일정하면 2개 cell만 남아도 degenerate=false |

따라서 “105개 기존 unit가 모두 통과했다”는 사실은 계획 전체 적합성을 보장하지 않는다. 실제 데이터/손실 경로의 양성 검사와 계획이 금지한 상태를 거부하는 음성 검사를 따로 수행했다.

- CPU만 사용했다. WV3 실제 학습 패치 2개, 고정 T0, 기존 Student 4040-step U를 이용해 `KDVTrainer._step` 및 routing 후 A/U gradient를 직접 수식과 대조했다. 비영(非零) gradient를 확인하기 위한 진단용 상태이며 새 run을 warm-start하지 않았다.
- W104는 실제 생성 config로 모델을 만들고 64/256/512 입력을 실행했다. 새 U의 zero output head 때문에 A gradient 검사가 자명해지는 것을 피하려고 gradient 검사에서만 zero parameter를 난수화했다. 이는 repository smoke와 같은 검사 방식이다.
- LF prefix 검사는 같은 상태에서 index 0/24999의 두 국소 update를 맞춘 뒤 25000/25001 전환을 확인했다. **25K 전체 궤적 또는 GPU bitwise 재현을 검증한 것은 아니다.**
- 기존 PAKD unit는 실제 mandatory 파일을 잠시 쓰는 코드가 있으므로 `/tmp`의 격리된 저장소에서 실행했다. 셸 재배정 검사도 별도 저장소에서 수행했다. 그 fixture의 unit 실행은 이미 별도 검증한 결과를 대신하는 stub으로, requeue는 실행 방지 stub으로 교체했다. 실제 config 생성·비교·extra/mandatory/reservation 로직은 그대로 실행했다.
- s3/s4 원격 PID·GPU OOM·현재 queue·완료 결과·Sheet 셀을 직접 조회하지 않았다. 300-update 실측이나 전체 50K 재학습도 하지 않았다. 문서의 GPU smoke·Sheet 수치를 독립 실측으로 표시하지 않는다.

## 3. 계획대로 확인된 구현

| 대상 | 확인한 계약 |
|---|---|
| DCR B0/B1 | B0=FQ, B1=JK0. W112/D123, seed1234/777, T0 donor A와 fresh U, full50K |
| JK0 routing | U에는 Q12 hard+soft+edge 전체. A에는 hard+edge+offset만 전달. 직접 계산한 scalar loss 및 A/U gradient와 일치 |
| Offset 연습 | I-AEQ의 odd update에서 radius2 ε를 A에만 적용. reconstruction은 native PAN. offset→U gradient 0 |
| 고정 자산·계수 | T0 step24240, checkpoint full SHA `16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32`; τR=0.012463942170143127, λE=0.09075170336956798 |
| s1 초기값 | 현재 FQ1234와 JK01234의 U init hash 모두 `c988a6c95b17f4fd`. FQ의 A hash `493aca181f93ae68`은 T0 A와 같음 |
| D01 기본 측정 | CAL1024/DISC512/CONF512, seed314159. 0.5/1.0 반경 16개 probe와 회전 confirm probe. zero probe는 주 평균에서 제외. FR에 GT R/사분면을 꾸며 넣지 않음 |
| D02 개입 | learned/zero/wrong-sign/±0.25 bias가 같은 원 PAN에서 한 번의 warp로 구성됨. B_R=R_zero−R_learned, B_Q=HQNR_learned−HQNR_zero 부호 일치. closure 공통 bias와 seq/comp 보간 차이 분리 |
| D03 기본 도구 | per-sample A gradient, λE를 포함한 edge, raw/weighted offset norm, native dL/dc, virtual step 전후 고정 SG target 및 재계산 closure 분리. 원 자산 보존 |
| s3 5개 recipe | J_R3_NOEDGE=weighted GT+adaptive soft; J_N0_EDGE=plain GT+GT edge; LF0/LFQ/LFX=각각 N0/Q12/X02에 freeze_from25000 |
| LF 전환 | 25000 전에는 대응 J0/JQ/XJ와 같은 국소 loss·update·ε 상태. 이후 A hash 불변, offset0, U update 지속. 25250 checkpoint로 25000 분기를 대체하지 않고 full50K |
| s3 편성·산술 | J_R3_NOEDGE→J_N0_EDGE→LF0→LFQ→LFX. core 7.994333h, 4321 확인 4.922h, 합12.916333h. 후보 6종의 고정 확인 표 일치 |
| W104 모델 | Student hidden104/depth[1,2,1], backbone **1,903,624** params(표시1.9036M). Teacher는 자기 config의 W112/D123을 strict load. A만 T0에서 복사 |
| W104 입력·residual | 64/256/512에서 HRMS8 shape·finite 통과. wrapper가 MS base를 한 번 더함 |
| W104 정책 | NA0은 aligner/sampler 없음·원 PAN·Teacher loss 없음. F0는 A 동결. J0/JQ/XJ는 지정한 A/U gradient, offset은 A만. Teacher 불변. Q의 β=0은 X와 동일 |
| W104 설정·예산 | NA0→J0→JQ→XJ→F0@W104_D121, seed1234, 50K/batch48/warm100/grid1010 유지. branch=A104D121_T0FIX_E0_v1. core7.752333h |
| W104 확인 묶음 | WIN∈{JQ,XJ,F0}, seed3407의 J0/WIN@W104 다음 J0/WIN@W112. 한 번 호출의 4개 정의 및6.342667h, 총14.095h 일치 |
| 예약 전달·시트 코드 | case@arch별 reservation과 measured key 분리, gate_hours×margin 규약 유지. X열 A104D121 토큰 및 Notes의 Student/Teacher/branch/E0 표기 구현됨. 실제 Sheet 반영은 미검증 |

## 4. 적용 또는 자동 판정 전에 해결할 문제

### D01 — patch-only 및 불충분한 검증 조합에서도 D04 gate가 열린다 [P1]

현재 [panel 생성](../tools/dcr12/common.py:192)은 원 scene ID가 없어 `independence=patch_only`라고 기록한다. 계획 §5/§9.2는 이 경우 routing을 보류하도록 한다. 그러나 [gate](../tools/dcr12/d04.py:40)는 이 메타데이터를 검사하지 않으며 인자 `d01`도 사용하지 않는다. 실제 caller는 `None`을 전달한다.

같은 코드에서 다음 반례가 모두 `passed=true`를 반환했다.

- `independence=patch_only`를 명시한 자료.
- DISC/CONF는 있지만 checkpoint가 **25250 하나뿐**인 자료.
- **DISC 없이 CONF만** 두 checkpoint 있는 자료.
- 독립 group이 **2개뿐**인 자료.

`n_contexts>=2`는 DISC/CONF×25250/45450의 필수 조합을 보장하지 않는다. condition1은 C 변동성만 확인하고 cell의 n/group support를 검사하지 않는다. 또 z* 선택은 `groupby(panel,step)`의 첫 원소를 사용해 **CONF의 순위**를 따른다. DISC에서 B가 최악이고 CONF에서 A가 최악인 반례에서 A를 골랐다. 이는 “DISC에서 한 cell을 고정하고 CONF로 확인”과 반대다.

**해결 조건:** scene/group 신뢰도와 cell support를 먼저 검사하고, 필요한 panel/checkpoint 조합을 명시적으로 요구한다. z*는 DISC만으로 저장한 뒤 CONF에서 변경하지 않는다. 현재 B2/B3를 자동 학습시키는 코드는 없지만, `routing_policy.json`과 후보 판정 자체가 잘못 생성될 수 있다.

### D02 — 추가정보 검증과 bootstrap의 관측 단위가 계획을 충족하지 않는다 [P1]

계획 §9.2는 R_T·Student 우위·texture가 유사한 조건에서도 C가 KD utility를 구분하는지 요구한다. [현재 gate](../tools/dcr12/d04.py:44)는 low/high R라는 큰 구간 안에서 A/C 또는 B/D의 U 평균만 비교한다. 연속 R_T와 a_T, texture를 맞추거나 조건부 분석하지 않는다. D01에도 sample별 a_T가 저장되지 않는다. 따라서 gate가 유의해도 C의 독립적인 추가정보라고 판정할 근거가 부족하다.

[episode 생성](../tools/dcr12/d04.py:10)은 여러 source group의 patch를 섞어 support/query를 뽑고 episode끼리 재사용할 수 있다. 저장되는 `group_key`는 그 episode가 포함한 group 중 **첫 번째 하나**다([저장](../tools/dcr12/d04.py:95)). 이 임의 key로 CONF group-bootstrap을 하면 실제 공유 표본/그룹 의존성이 반영되지 않는다. fixture에서 support/query의 patch ID는 겹치지 않지만 source group은 겹치는 것도 확인했다. 이 겹침만을 별도의 계획 위반으로 단정하는 것이 아니라, 현재 bootstrap이 독립 단위를 보장하지 못한다는 증거다.

**해결 조건:** sample 및 episode 구성 ID·전체 group 목록을 보존하고, 실제 독립 단위에서 split/resampling한다. R_T/a_T/texture 조건을 맞춘 utility 분석이 없으면 condition2를 통과시키지 않는다. 원 scene ID가 없는 현재 자료는 D01의 보류가 우선한다.

### D03 — B1이 B0의 고정 sample bin을 유지하지 않는다 [P1]

[assign_cells](../tools/dcr12/d01.py:66)는 B0의 **threshold 숫자만** 고정한 뒤 각 pipeline 자신의 C/R로 `cell_ref`를 다시 정한다. 동일 sample의 B0 C/R=.1/.1, B1=.9/.9, threshold=.5/.5이면 `cell_ref`가 A→D로 이동했다.

계획 [§5.3](PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md:301)이 피하려는 것은 method 변화에 따라 표본이 cell 사이로 이동하는 직접 대조의 선택 편향이다. 현재 `cell_ref`는 이를 막지 못한다. **해결 조건:** `(seed,panel,sample_id)`별 B0 reference label을 B1에 join하고, B1 자신의 위치 변화는 별도 동적 label로 보고한다.

### D04 — 부적합/미실행 상태가 자동 보고에서 합격·해석 라벨로 바뀐다 [P1]

[REPORT verdicts](../tools/dcr12/report.py:18)는 D04 `passed=true`만으로 `QUADRANT_ROUTING_CANDIDATE`를 출력한다. B2/B3 결과를 입력받지도 않는다. 계획 §14.1의 이 라벨은 실제 B2/B3 개선까지 요구한다. “B2/B3 확인 필요”라는 부연으로 라벨 정의의 불일치가 해소되지 않는다.

또 native C–R 상관만 있고 D02 synthetic/D04 자료가 전혀 없어도 `ROBUSTNESS_ONLY?`를 붙인다. 미실행은 KD utility 부재의 증거가 아니며, native 상관은 synthetic-only 설명의 증거도 아니다. 두 경우 모두 직접 재현했다.

하위 검증도 충분히 차단하지 않는다. [stage check](../tools/dcr12.py:16)는 파일 존재만 확인해 D00 `implementation_invalid=true`, D03 routing `ok=false`, 빈 CSV/JSON도 통과시킨다. [D03](../tools/dcr12/d03.py:70)은 routing 불일치를 기록해도 stage 실패로 반환하지 않는다. REPORT는 읽어 놓은 d00 결과를 판정에 사용하지 않는다. 최초 D00 CLI의 invalid 종료 경로는 있으나 **resume/후속 판정**의 차단과는 별개다.

**해결 조건:** 내용·row coverage·identity·hard failure를 검증하고, `not_run / inconclusive / invalid / measured_negative`를 분리한다. full routing 후보 라벨은 계획의 관측 요건을 모두 만족한 뒤에만 사용한다.

### O01 — s3/s4 runner 전환이 현재 run의 후처리를 잃을 수 있다 [P1, 기존 지적 미해결]

[requeue](../tools/pakd50_requeue.sh:30)는 진행 중 `_run_cases.sh`를 즉시 kill한다. 학습 자식이 살아 있어도 부모가 맡은 rc 판정·실패 원장·eval_dlpan·업로드는 유실될 수 있다([후처리](../tools/_run_cases.sh:109)). 새 gate가 완료 run을 제외하면 이를 자동 복구하지 않는다. s3/s4 구현 노트의 “runner만 교체, 현재 run 끝까지”만으로는 안전한 인계가 보장되지 않는다.

이 부모/자식 동작은 [이전 감사 D01](PAN_PAKD50_Derived_Allocation_Implementation_Audit_2026-09-15.md)에서 격리 프로세스로 재현했고, 관련 requeue/runner 코드가 이후 두 추가 commit에서 수정되지 않은 것을 재확인했다. 이번에는 활성 프로세스를 종료하지 않았다. 이미 모든 run과 후처리가 끝난 서버라면 이 특정 손실 경로는 발생하지 않는다.

**해결 조건:** 종료·후처리까지 마친 run 경계에서 목록을 교체하거나 후처리 소유권을 확실히 인계한다. 실행 중 checkout에 대한 무조건 `git pull` 절차도 경계/격리 조건을 명시해야 한다.

### O02 — dry-run이 활성 실행 파일을 쓰고 확인 실험까지 등록한다 [P1]

[reallocate](../tools/pakd50_reallocate.sh:19)는 calibration을 쓰고, confirm YAML·extra/mandatory·reservations를 만든 뒤 마지막에 dry-run 여부를 확인한다. s3와 W104 s4 fixture에서 `--dry-run --confirm ...`만으로 실제 extra에 신규 run이 등록됐다. 활성 gate가 이 파일을 읽는 환경에서는 조회 명령이 실행 범위를 바꾼다.

**해결 조건:** 모든 dry-run 산출물을 임시 경로에서 계산하고 활성 파일의 내용/존재 여부가 전후 동일해야 한다. 이번 감사에서는 이 명령을 운영 저장소에서 실행하지 않았다.

### O03 — 확인 run이 mandatory에서 지워지고 기존 확인 묶음은 남는다 [P1]

[confirm 등록](../tools/pakd50_reallocate.sh:23) 직후 같은 스크립트가 [write_mandatory_file](../tools/gen_pakd50_configs.py:110)을 호출해 기본 묶음만 `w` 모드로 다시 쓴다. **extra에는 확인 run이 있지만 mandatory에는 없는** 상태가 된다.

서버별 반례:

- s3: `--confirm LFX` 뒤 4321의 XJ/LF0/LFX가 mandatory에서 모두 사라짐. 이어 `--confirm J_N0_EDGE`를 호출하면 기존 묶음과 합쳐 **확인 5개**가 남음. “묶음 최대3”은 개별 함수 반환에만 적용됨.
- s4: 기존 3407 RCQ/RC0/F0가 등록된 상태에서 새 `--confirm JQ`를 호출하면 이전3+새4=**7개**가 extra에 남음. 새4는 mandatory에서 모두 사라짐. 계획 §6의 “기존 3407 예약을 대체”가 구현되지 않음.

**해결 조건:** branch/seed별로 승인된 확인 묶음 하나를 원자적으로 대체하고 mandatory와 reservations를 함께 갱신한다. 완료된 과거 결과의 파일을 삭제하라는 뜻은 아니다. 계획 순서만 바꾸고 extra를 유지하면 중복 신규 실행을 막을 수 없다.

### O04 — s3 부족 예산에서 확인 대응묶음을 먼저 예약하지 않는다 [P1]

계획 §6은 시간이 부족하면 확인 대응묶음을 먼저 확보하도록 한다. [schedule](../tools/gen_pakd50_configs.py:365)은 기본 priority를 모두 소비한 뒤 extra를 고려한다. 남은9h, core7.994333h, LFX 확인4.922h의 반례에서 core5개를 전부 넣고 확인3개는 전부 제외했다. 단순 예약 합 산술은 맞지만 계획의 admission 우선순위는 다르다.

**해결 조건:** 확인 대응묶음을 예약한 잔여 예산으로 exploration을 승인하고, 묶음 일부만 남지 않게 처리한다. 미래의 실제 시작 시각·남은시간은 다시 계산해야 한다.

### A01 — s3의 실제 W104 source config를 확인하지 않고 구조가 같다고 선언했다 [P1, 이식 근거 미확인]

계획 [§3](PAN_PAKD50_S4_W104D121_Architecture_Allocation_2026-09-15.md:76)은 실제 s3 run의 `meta/config.yaml`과 source revision을 전달받으라고 명시한다. 새 구현 노트도 그 파일이 s1에 없다고 인정하면서, 기존 PO10 template에서 width/depth만 바꿔 parameter 수가1.9036M이라는 이유로 같은 lineage라고 판단했다. [render](../tools/gen_pakd50_configs.py:437)도 실제로 두 항목을 치환한다.

**parameter 수 일치는 구조·연산 동일성의 증거가 아니다.** 예를 들어 dropout 변경은 parameter 수가 같다. 이번 감사는 새 W104 모델이 잘 실행됨을 확인했지만, 그것이 s3에서 성공한 바로 그 구조라는 사실은 확인하지 못했다. **해결 조건:** 원본 resolved config·소스 revision/hash를 확보해 decoder/norm/input/task 등 전체 설정과 대조한다. 없으면 “검증된 구조 이식”이 아니라 별도 새 구조 가설로 식별해야 한다.

### A02 — 모델 초기화가 cross-architecture data RNG를 바꾼다 [P1]

계획 [§6](PAN_PAKD50_S4_W104D121_Architecture_Allocation_2026-09-15.md:152)은 두 구조의 data/augmentation/epsilon RNG를 모델 초기화 소비와 분리하라고 요구한다. [main.load_data](../main.py:156)의 train DataLoader는 `generator`가 없고, 그 뒤 [load_model](../main.py:186)이 전역 Torch RNG를 소비한다. `_pair_init`은 모델 값을 맞출 뿐 이 RNG 상태를 맞추지 않는다.

실제 `main.load_data/load_model`을 사용하고 dataset I/O만 정수 index dataset으로 바꾼 재현에서 seed3407의 첫 batch는 다음처럼 달랐다.

| 구조 | 첫 4개 sample ID | 모델 생성 직후 RNG hash |
|---|---|---|
| W104_D121 | 450, 11, 422, 775 | `41c0109631c58bf2` |
| W112_D123 | 307, 928, 724, 681 | `feefc541cb4ab9aa` |

**해결 조건:** DataLoader/worker augmentation의 seed를 독립 generator로 관리하고, 두 구조에서 sample ID·augmentation·ε 열의 대응을 확인한다. ε 자체는 기존 별도 Generator를 사용한다. 같은 구조 안의 control/treatment pairing은 이번 검사에서 유지됐다. 이 문제는 서로 다른 구조의 비교 계약에 해당한다.

## 5. 분석 완결성·재개·초기값의 추가 문제

| 항목 | 근거와 영향 | 필요한 처리 |
|---|---|---|
| DCR 선택 D04가 필수 보고 앞을 막음 [P1] | [runner 마지막 줄들](../tools/dcr12_run.sh:35)이 D04를 무조건 실행한 뒤 RESULTS/REPORT. candidate가 없으면 [D04](../tools/dcr12/d04.py:99)가 없는 CSV를 읽어 FileNotFoundError; 직접 재현. 계획은 D04 선택·확장을 전부 자동 실행하지 않음 | D04를 명시적 조건부 단계로 두고 skip/failure를 기록해도 필수 B0/B1 report는 생성 |
| DCR 실행 예산 격리 미구현 [P2, 운영 계약] | 현재 전용 분석 ledger만 있고 training budget은 PAKD50 원장. s1 mandatory는 이전 J0/JQ만, DCR pending JK01234/FQ777/JK0777 미포함. campaign gate 파일은 비어 있음. 계획 §12의 DCR 전용 pending/budget와 다름 | 필요한 대응쌍과 진단 비용을 담는 전용 admission 상태 구성. 기존 절대 deadline/안전 검사는 보존. 현재 run이 이 때문에 이미 잘못됐다는 증거는 없음 |
| 완료 판정이 두 MAT 존재뿐 [P2] | [common.run_complete](../tools/dcr12/common.py:119), runner, results가 exit/last/grid/FR20 identity를 확인하지 않음. 빈2파일로도 완료. metadata 없는 last를50000으로 간주하는 반례 통과 | actual exit, exact50000, 50후보·plateau6점, export20scene 및 선택 checkpoint 연결 확인. 현재 재사용 FQ1234는 실제50후보·last50000으로 별도 확인됨 |
| profile/partial 산출물이 본 분석을 막음 [P2] | [D01](../tools/dcr12/d01.py:116), D02/D03가 pipeline row 하나라도 있으면 전체를 skip. profile 축소본과 본 분석이 같은 파일명. stage check도 row수/identity 미검사 | expected sample×panel×checkpoint coverage와 source/config/panel/probe hash 검증. profile 출력 분리 |
| D03 zero gradient NA 처리 [P2] | [per_sample](../tools/dcr12/d03.py:43)의 `norm < 10*floor`: floor=0, norm=0이면 false. 실제 T0=self-student에서 soft norm0을 NA 대신 cos0으로 보고하는 반례 재현 | `<=` 및 적절한 절대/상대 floor, 유효 sample 수와 분모 명시. 저장 T0의 미세 soft gradient는 batch64 Teacher 대 batch1 재계산 오차 가능성을 분리해 해석 |
| D00/D03 “relative1e-4” 검사식 [P2] | 실제식 `err <= 1e-4*(1+ref)`라 작은 gradient에서는 사실상 절대1e-4 허용. 현재 저장 D00 A error1.135e-6/ref.01221은 실제 상대9.30e-5로 의도한 기준도 통과 | 명시적 atol+rtol 계약으로 표시·검사. 현재 D00 통과를 오판이라고 주장하지 않음 |
| D04 저장 optimizer 무시 [P2, 명시된 이탈] | [micro_update](../tools/dcr12/d04.py:21)는 항상 fresh AdamW. 현재 FQ1234 candidate25250에 optimizer.bin이 실제 존재. 계획은 저장 state가 없을 때 fresh 대안 허용 | 두 arm에 같은 저장 optimizer 상태를 clone하거나 실험 정의 변경을 명시. 현재 노트에 fresh/non-continuation 표시가 있는 것은 적절함 |
| D03 gC의 의미 축소 [P2, 명시된 이탈] | 고정 r1 4축 평균이고 실제 odd-update의 r2 disk ε가 아님. 기존 note에 명시됨 | 해당 값은 고정 probe 진단으로만 해석. 실제 학습 gradient 분포 주장에는 같은 ε 정의/로그 추가 필요 |
| D02·D01 일부 산출물 누락 [P2] | synthetic composition은 DISC128만; 계획의 RR20/FR synthetic 없음. D01 C–R scatter 생성 없음. a_T 조건부 분석 없음. ±bias FR8 partial과는 별개 | 누락/partial을 개별 표기. synthetic-only 및 C의 추가정보 결론에 사용하지 않음 |
| degenerate/empty cell 처리 [P2] | [thresholds](../tools/dcr12/d01.py:52)가 C IQR만 검사. R 일정·C변동128sample 반례에서 A64/C64만 남아도 degenerate=false. summary는 빈 cell 행도 생략 | median 동률로 cell 소실 감지, A/B/C/D의 empty/small row 보존, gate 차단 |
| W104 init full signature 미검증 [P2] | [init_dir](../train_kdv.py:85)는 width/depth뿐; asset key는 unet@W104_D121. 문자열 architecture_signature를 기록하지만 init 선택/로드에 사용하지 않음. dropout0→.2 변경도 같은 파일로 조용히 초기화되는 반례 재현 | 전체 resolved model/task/decoder signature로 cache/hash namespace 구성하고 mismatch 거부 |
| W104 최초 U pin 공백 [P2] | asset의1234/3407 값이 null이라5개 config 모두 expect_init 없음. 이후 첫 run의 hash를 채우라는 노트만 있음 | 학습 전 fresh init identity를 생성·동결하고 config/manifest 연결. 현재 _pair_init의 동일 폴더 내 공유 보호까지 없는 것은 아님 |
| 새 throughput 실측 미수행 [제한] | DCR 계획300-update-equivalent profile 대신 이전 PAKD smoke 값을 재사용. W104도 현재 코드는 계획 대용값으로 예약 | 실측 확보 전에는 계획치로 표기. 같은 서버의 실제 throughput/평가/저장시간으로 갱신 |

s3 구현 노트의 “같은 config·seed의 v1/v2 HQNR 차이가 GPU run-to-run 폭”이라는 해석도 아직 근거가 충분하지 않다. 두 실제 run의 resolved config·init hash·Teacher·selector·중단/재개 이력을 맞추기 전에는 GPU 비결정성으로 귀속할 수 없다. 이번 CPU prefix 검사는 같은 정의와 국소 전환을 지지하며 그 귀속 주장을 검증한 것은 아니다.

## 6. 현재 진행 상태를 직접 확인한 범위

**s1, 10:55:42 KST host process 조회:** 학습 PID1268078이 `PAKD50_JK0_W112_D123_WV3_T0_S1234_FRESH50_v1`을09:33:13부터 실행 중이며 GPU6680MiB를 사용했다. 부모 `_run_cases.sh` PID1267201과 DCR runner PID1328592가 존재했다. 같은 명령처럼 보이는 추가4개 PID는 학습 프로세스의 DataLoader 자식들이며 별도 실험4개가 아니다.

10:57:25에 보존한 파일 기준:

- B0/FQ1234는 재사용 가능한 완료 run이며 U init·A hash가 현재 대조 설정과 일치한다. 50개의 평가 후보와 exact50000이 존재한다.
- DCR pre D00–D03가 완료됐고 DCR runner는 JK01234→FQ777→JK0777 학습 완료를 기다리고 있다. sample_metrics에는 T0/B0-best/B0-last, gradient_conflicts에는 T0/B0-best만 있다. B1 두 seed 비교가 끝난 상태가 아니다.
- 현재 JK0는 계획된 config로 진행 중이다. 이 확인만으로 final HQNR·두 seed 재현·D04 결과를 선행 판정할 수 없다.
- s3/s4 원격 실행 완료 또는 신규 queue 적용은 확인하지 않았다. Git push 기록, 로컬 생성 config, 문서의 Sheet 표는 원격 실행 상태를 대신하지 않는다.

우선 해결 순서는 **DCR gate/label의 잘못된 양성 판정 방지 → fixed reference/utility 통계 → 재배정 인계·dry-run·확인 예약 → W104 source/RNG/init identity**다. 기존 B0/B1 기본 학습의 수식 검증 통과와, 아직 충족되지 않은 분석·운영 조건을 함께 유지해야 한다.

## 7. 재현 자료

- [시간창/5개 파일 목록](Recent_MD_2H_Audit_2026-09-15_103707/scope.json)
- [최종 commit·SHA256 목록](Recent_MD_2H_Audit_2026-09-15_103707/provenance_final.json), [Git/push 기록](Recent_MD_2H_Audit_2026-09-15_103707/git_history_final.log), [최종 소스 사본](Recent_MD_2H_Audit_2026-09-15_103707/snapshot_final/)
- [DCR/s3 감사 코드](Recent_MD_2H_Audit_2026-09-15_103707/verify_recent.py), [JSON](Recent_MD_2H_Audit_2026-09-15_103707/verification.json), [로그](Recent_MD_2H_Audit_2026-09-15_103707/verification.log)
- [W104 감사 코드](Recent_MD_2H_Audit_2026-09-15_103707/verify_w104.py), [JSON](Recent_MD_2H_Audit_2026-09-15_103707/w104_verification.json), [로그](Recent_MD_2H_Audit_2026-09-15_103707/w104_verification.log)
- [PAKD92 검사 로그](Recent_MD_2H_Audit_2026-09-15_103707/pakd_unit.log), [DCR13 검사 로그](Recent_MD_2H_Audit_2026-09-15_103707/dcr13_unit.log), [동률 반례](Recent_MD_2H_Audit_2026-09-15_103707/degenerate_cell_reproduction.json)
- [s3 셸 재현](Recent_MD_2H_Audit_2026-09-15_103707/s3_reallocation_fixture.log), [s4 셸 재현](Recent_MD_2H_Audit_2026-09-15_103707/w104_reallocation_fixture.log), [시점 고정 runtime 사본](Recent_MD_2H_Audit_2026-09-15_103707/runtime_final/metadata.json)

모든 검사는 CPU·별도 출력·격리 fixture를 사용했다. 재검증은 보존된 release와 파일 SHA를 확인한 뒤 수행해야 한다. 코드가 바뀐 working tree에서 이전 결과를 재사용하면 안 된다. 이 감사가 생성한 것은 본 보고서와 위 감사 자료뿐이다.
