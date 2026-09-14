**EQREC4-S1-v1 구현·실행 검증 — 2026-09-15**

**판정: 핵심 forward·q·gradient routing은 계획과 일치한다. 그러나 현재 구현 전체를 ‘계획대로 검증 완료’로 승인할 수 없다. H2 집계, H4의 source 단위 검증·진입 조건, 최종 가설 판정에 수정이 필요하다. 이미 완료된 D10 및 D20의 확인된 정상 데이터를 폐기할 이유는 발견하지 못했다.**

대상 계획은 `PAN_S1_EQREC4_Alignment_Cue_Hypotheses_20h_2026-09-14.md`다. 앞선 PAKD50 감사와 다른 프로토콜로 검토했다. EQREC4는 frozen 진단과 신규 단기 적응이며, 20시간은 엄격한 종료 시각이 아니다. 코드·실제 산출물·실행 프로세스·CPU 재현 검사를 대조했다. 캠페인의 소스, checkpoint, optimizer, 실행 상태는 변경하지 않았다. 이번에 추가한 파일은 이 보고서와 `EQREC4_Audit_2026-09-15/`의 감사 자료다.

감사 기준 HEAD는 `a0bff24ff345a67922e1689482a3fade7a538cdb`. 원 계획 SHA-256은 `2721172d91e498e75f7b0e19b84882e94ddf41c53aaa16d6fa91aa61f02f0e48`. 코드와 주요 산출물의 snapshot/hash는 [provenance.json](EQREC4_Audit_2026-09-15/provenance.json)에 있다. 진행 중인 파일의 내용은 이후 달라질 수 있다.

**1. 실제 진행 상태와 이미 검증된 범위**

T0는 **2026-09-14 23:06:57 KST**다. 최초 runner는 D10 완료 표시와 실제 필수 파일이 불일치하여 D20 진입 시 실패했고, 23:07:40 재기동 후 D10부터 정상 실행됐다. 현재 ledger에는 아래 진행이 남아 있다.

| 단계 | 관측된 실행 상태 | 검증 판단 |
|---|---|---|
| G00 | 23:05:41–23:06:09 완료 | Primary HQNR·기본 수신자 검사 통과. provenance 분류와 실패 전파는 불완전 |
| D10 | 23:07:43–23:29:56 완료 | 24 model/checkpoint 전수 atlas, A threshold, q/EPE 수치 확인 |
| D20 | 23:30:46–23:31:40 완료 | FR 입력 수정 반영 확인. MS/common은 Probe A만 실행한 범위 제한 |
| D30 | 23:31:45 시작, 감사 중 실행 중 | 원시 intervention 전체 CSV는 단계 종료 때 저장. 현재 모든 결과를 확인한 상태는 아님 |
| D40 / D50 | 아직 시작 기록 없음 | 코드와 작은 실제 checkpoint CPU 검사로 검증 |
| K10 / K20 | 아직 시작 기록 없음 | 구현·예정 pool·합성 실패 사례·짧은 CPU reset 검사로 검증. 캠페인 성능 결과 아님 |
| REPORT | 아직 시작 기록 없음 | 생성 코드의 판정 오류와 누락을 확인 |

00:05 무렵 host 확인에서 runner PID `2190020`, D30 PID `2196665`, RTX 4090 사용량 약 9,031 MiB, GPU utilization 84%였다. **감사 종료 확인 00:21:30 KST에도 같은 D30 프로세스가 실행 중**이었다. primary 한 모델의 작업 완료 로그는 709초로 기록됐고 다음 모델로 진행했다. 당시 GPU 순간 사용량은 4,631 MiB/3%였다. CPU estimator 작업과 GPU 작업이 섞이며 로그가 model 단위로 기록되므로, 순간 GPU 사용률이나 로그 간격만으로 정지라고 판정하지 않았다.

중요하게, D20 FR MS 인덱스 수정은 **commit 시각 00:04:49보다 먼저인 23:30:43에 파일에 반영됐고, 23:30:46에 D20이 시작**됐다. 저장된 10 model × FR20 = **200건**의 PAN-only EPE를 D10과 대조하면 최대 차이가 `1.1921e-7 px`다. 따라서 이 수정이 현재 저장 결과에 반영되지 않았다는 의심은 배제했다. 수정 전 `fr[i][2]`는 1채널 lPAN이며, 현재의 `fr[i][1]`가 올바른 8채널 MS다.

**2. 정상 구현으로 확인한 내용**

| 계획 계약 | 확인 결과 |
|---|---|
| L1E4/L000 3 seed best+last, P0·L3E4·N2 등 registry | atlas 24개, Student source 포함 자산 26개 존재. 26개 모델 파일의 현재 SHA가 G00 manifest와 모두 일치 |
| Primary T 고정, A/U 동일 checkpoint | S2025 best_raw update 24,240, G00 raw HQNR **0.9569763995761281**, 기록값과 차이 **0** |
| Student 선별 | Pair A S1234의 현재 저장된 최초 ≥20K는 40,400; Pair B S7777은 30,300. 가상 checkpoint 없음. 늦은 학습 상태라는 해석 한계는 남음 |
| 데이터 역할 | A/B/C/D = 512/2,048/512/1,024, ID 중복 없음, 한 proxy block은 한 역할에만 배정 |
| D10 coverage | native64 **98,304행**, RR **480행**, FR **480행** = 24 × (4,096+20+20) |
| q/EPE | core 12개 × 4,096 × 32 = **1,572,864 probe 행**의 residual로 q/EPE 전수 재계산. 최대 차이 `1.3833e-7` |
| Probe/threshold | A 축·B 대각 각 16개, identity 제외, yx 부호·component 평균 맞음. A에서 계산한 e/q_A/q_B threshold 및 전체 quadrant label 재현 |
| P0 | q를 0으로 넣지 않고 결측으로 유지 |
| Forward | M을 한 번 더하고 native PAN만 warp. 현재 `kdv_forward`와 신규 R0의 forward/base gradient가 실제 Student에서 일치, CPU gradient 최대 차이 **0** |
| 손실별 수신자 | 실제 R0/RH/RS/K20에서 Rec→A/U, Off→A only, 추가 hard/soft/mixed→U only. 같은 source의 직접 A gradient 차이 **0**, 첫 AdamW A update도 동일 |
| Teacher/Student | 별도 parameter storage, frozen Teacher 및 원 Student hash 불변 |
| D40 기하 | 실제 작은 검사에서 response/no-response/known-inverse 경로 존재, known-inverse = c0−ε. FR은 원 PAN/MS 참조 고정, native/stress 모두 V96 사용 |
| D50 | A-only normalized step, 원 상태 복원 및 재계산한 q_B 측정 확인 |
| K10 기본 설정 | 64 updates, γ=.1, U/A LR=1e-5/1e-6, fresh AdamW, canonical input, odd 0-based update Off. 현재 source의 기본 optimizer·precision 정책과 일치. 실제 run_trial을 감사용 2-step으로 arm 순서를 바꿔 실행해 source hash·C per-sample 결과의 동일성 확인 |
| K20 기본 비교 | 5개 arm, 고정 source cue, 동일 batch order, 5K 신규 warmup100+cosine, γ=.1, EA/EAQ 평균 ρ=.5 계산은 구현됨 |

**검증 assertion 30개: 정상성 확인 20개, 결함 재현 10개. 모두 예상한 결과를 재현했다.** 결함 재현 assertion의 성공은 구현의 정상 통과를 의미하지 않는다. [verification.json](EQREC4_Audit_2026-09-15/verification.json), [재현 스크립트](EQREC4_Audit_2026-09-15/verify_readonly.py), [실행 로그](EQREC4_Audit_2026-09-15/verification.log)에 각각의 입력·수치가 있다. CSV threshold hash는 `float_precision='round_trip'`로 원 float를 복원해 비교했다. CPU 검사로 전체 64/5,000 update의 GPU 효과나 최종 성능까지 검증했다고 주장하지 않는다.

**3. 우선 수정해야 할 문제**

P1은 주요 가설 판정·비교 유효성에 직접 영향을 주는 항목, P2는 필수 실험 범위·측정량·재현성을 충족하기 위해 필요한 항목이다. 아직 실행되지 않은 단계의 결함을 이미 발생한 캠페인 결과처럼 서술하지 않았다.

**Q01 · P1 — K20이 조건부 pilot으로 동작하지 않는다.**

- 계획 §13.1, §17은 cache/unit/ROI 검증, q 반복성, K10 실제 지원 및 재현 가능한 C utility, noise/source 지원을 진입 조건으로 요구한다.
- [k20.py:100](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4/k20.py:100)의 `main`은 K10 CSV를 읽은 뒤 조건 검사 없이 5개 arm 학습으로 진행한다. G00 status, q_A/q_B 반복성, `restore_ok`, 독립 source 지원 등을 읽거나 검사하지 않는다.
- `utility_tables`의 resolved 조건은 `max(abs(utility)) > 1e-9`뿐이다(:26–29). 실제 재실행 noise와 식별 가능성을 비교하지 않는다. `MAD=noise=0`인데 utility가 일정한 비영값인 경우에도 계획의 `.5/no_resolved_utility_signal` 처리 대신 임의 floor `1e-12`를 사용한다.
- 재현 1: utility 전체를 0으로 하고 source 지원도 확인하지 않은 fixture에서 `no_signal=true`를 출력한 뒤 실제 학습 함수 호출 지점에 도달했다. 감사용 stub이 첫 update 전에 차단했다.
- 재현 2: utility 최대 `4e-6`, noise `1e-3`에서도 resolved로 분류했다.
- 조치: K20 실행 전에 명시적 gate 파일을 만들고 실패/미식별 사유를 저장해야 한다. 조건 실패 시 K20만 `pilot_not_identifiable` 또는 `final_training_pilot_not_run`으로 남기고 보고서로 진행할 수 있어야 한다. **현재 실행 중인 D30이 잘못됐다는 의미는 아니다.**

**Q02 · P1 — K10의 ‘held-pool’ 검증과 K20 수축에서 source 의존성을 통제하지 않는다.**

- 계획 §12.2/§15.3은 source 균형 pool 및 fit-pool/source 묶음 분할을 요구한다. 그룹 분할 불가능 시 out-of-fold 성능으로 보고하지 말라고 명시한다.
- [k10.py:93](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4/k10.py:93)는 cell 안에서 patch index를 단순 shuffle하여 48개씩 자른다. sample 중복은 없지만 source가 다른 trial에 반복된다.
- `h4_diag`(:126–145)는 한 pool만 빼고 회귀한다. 실제 Pair A source checkpoint가 D10의 S1234 best와 byte-identical임을 확인하고 예정 pool을 복원했다. **16개 모든 pool에서 해당 pool의 proxy source block 전부가 다른 pool에도 등장**한다. held pool별 공유 block은 29–39개다.
- K20의 `n=min(len(trials), sum(n_source_groups))`는 각 trial에 최소 한 source가 있으면 항상 `len(trials)`다. 동일 source 중복을 줄이는 기능이 없다. 동일 source 1개를 가진 4 trial fixture에서 `n=4`였다.
- 더 근본적으로 `blkNNN`은 32개 연속 index를 묶은 proxy다. 실제 scene ID가 없는데 G00 `provenance_limited=[]`, occupancy의 `insufficient_support=false`를 정상 source 지원처럼 사용한다.
- 감사에서 4,096개 input의 완전 중복 및 nonflat PAN 32×32 tile의 역할 간 완전 일치는 발견하지 못했다. 거친 특징 근접성 검사도 수행했으나 원 scene 독립성을 입증할 수 없다. [data_provenance.json](EQREC4_Audit_2026-09-15/data_provenance.json) 참조. 이는 캠페인의 빠진 provenance 검사를 보충한 감사 자료이며, `blkNNN`을 실제 scene으로 인증한 결과가 아니다.
- 조치: source 의존성에 따라 연결된 pool을 묶어서 검증하거나, 분리가 불가능하면 descriptive fit-pool 분석으로 제한한다. K20의 effective n도 동일 의존 구조로 계산해야 한다. `provenance_limited`와 확인 가능한 source 수를 별도로 기록해야 한다.

**Q03 · P1 — H2의 ‘독립 Probe B’ 집계에 Probe A가 섞인다.**

- `STRESS_PROBES`에는 B의 .5/1/2와 A의 r=1 재현 probe가 함께 있다. 추가 재현 probe 자체는 문제없다.
- 그러나 [d40.py:82](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4/d40.py:82)의 `h2_stats`는 `probe_bank`를 구분하지 않고 radius로만 묶는다. 따라서 native64/RR의 r=1은 **B 4방향 + A 4방향 평균**이다.
- 합성 fixture에서 B의 signed degradation을 +1, A를 −1로 두면 H2 r=1 보고값은 **0**, 사전 지정된 B-only 값은 **+1**이다. 이 경로를 직접 호출해 재현했다.
- `report.py:88,114`도 같은 혼합 자료를 사용하며 Fig3는 이를 bank B라고 표기한다. 실제 FR은 수정된 D40에서 B-only이므로 이 특정 혼합 문제는 native64/RR에 해당한다.
- 조치: 주 H2는 B-only, A는 독립 재현 표로 집계한다. 유효 support도 path마다 따로 고르지 말고 같은 sample/probe의 세 경로가 모두 유효한 paired 집합을 사용한다. 현재 보정 크기에서는 support가 충분할 가능성이 높지만, 현 집계 코드가 그 계약을 보장하지는 않는다.

**Q04 · P1 — 최종 가설 판정이 근거보다 강한 결론을 자동 생성한다.**

- [report.py:30](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4/report.py:30)의 `verdicts`는 계획에 없는 `4/6 checkpoint`, `60%`, `절반` 규칙을 사용한다. 이 규칙은 실행 protocol_changes에도 동결 근거가 없다. 같은 데이터에서 best/last·seed를 반복한 결과의 표 개수가 독립 지지 수는 아니다.
- 특히 H2의 반대 판정(:49)은 음의 **점추정치**가 60%면 충분하다. 양의 지지 판정에 사용한 CI 기준을 반대 판정에는 적용하지 않는다.
- 모든 상관을 `rho=-.001`, 모든 CI를 `[-.8,.8]`로 둔 fixture도 **Opposed in tested setting**이 됐다. 계획상 이 사례는 미검출/불충분으로 남아야 한다.
- H4_diag는 Pair A LOO MAE·direction accuracy가 조금만 개선돼도 Supported로 분류하며 Q02 source 의존성, 재실행 noise, 조건부 shuffle, Pair B 반증을 gate로 반영하지 않는다. H1_RR이라는 이름에는 실제로 native64 B∪C∪D 결과가 들어간다.
- 조치: 먼저 checkpoint/scale별 evidence table을 만들고 source 한계·CI·반대 대조·noise를 함께 판정한다. 경험적 투표 비율로 자동 결론을 내리지 않는다. 현재 H1의 음의 native64 상관처럼 직관과 다른 관측은 그대로 보존해야 하며, 그 자체를 구현 오류로 취급하지 않는다.

**4. 측정·실험 범위·재현성에서 필요한 수정**

**Q05 · P2 — band별 `d_e`가 실제로는 absolute error다.**

- [d40.py:22](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4/d40.py:22)에서 `eb=l1_roi(y_stress,GT,per_band=True)`를 구하고 그대로 `d_e_band0…7`로 저장한다(:26). native band error를 빼지 않았다.
- 실제 Primary Teacher 한 sample의 known-inverse 경로에서 band 열 평균은 `0.006408445`, `l1_stress_roi=0.006408443`, 실제 `d_e=0.0000625830`이었다.
- 조치: native band L1을 같은 ROI에서 빼서 delta를 저장하거나 열을 `l1_stress_band*`로 바꾸고 native/delta를 별도 저장한다. 현 scalar `d_e`는 맞으므로 spectral band 결과만 수정하면 된다.

**Q06 · P2 — D30 필수 전수 intervention과 각 checkpoint 자체의 quadrant 분석이 빠져 있다.**

- 계획 §9 D30-A는 L1E4 3 seed best/last의 learned/zero/wrong-sign을 core 전수로 명시한다. [d30.py:160](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4/d30.py:160)는 native64 **256개 primary 상세 subset**만 사용한다. RR20/FR20 전수는 구현돼 있다. native64에서 계획의 4,096개 중 3,840개가 core intervention에서 빠진다.
- [d10.py:131](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4/d10.py:131)의 subset은 Primary S2025 best의 D quadrant에서만 생성하고, D20/D30/D40/D50이 이를 공용으로 사용한다.
- 같은 256개에서 자기 모델의 quadrant와 primary label이 다른 sample은 S1234 best **59개**, S7777 best **17개**, L000 S2025 best **29개**다. D50 L000의 실제 예정 grad128에서도 **9개**가 다르다.
- `quadrant_primary`라고 명시한 것은 투명하다. 같은 sample의 모델 간 대응 비교로는 유효하다. 그러나 계획이 요구한 “각 checkpoint의 E↓C↑/E↑C↓” 비교를 대체하지 못한다. D50 L000의 quadrant별 해석에도 직접 영향을 준다.
- 조치: L/I-Z/I-W native 전수 결과를 보충하고, 자기 checkpoint quadrant로 rejoin/restratify한다. 고정 primary subset은 별도 대응 비교로 보존한다. 4분면 균형 추출 256개에서 얻은 평균을 4,096개 모집단 평균으로 확대하지 않는다.

**Q07 · P2 — blur calibration이 계획의 A split 밖에서 수행된다.**

- 계획 §10-E는 A calibration에서 zero-phase blur를 맞추도록 한다. `d30.py:169`는 기존 [palsv18_validate.py:177](/home/knuvi/Desktop/song/PAN-Crafter/tools/palsv18_validate.py:177)을 그대로 호출한다. 이 함수는 seed 20260913으로 원 train 전체에서 256개를 뽑는다.
- 실제 선택을 재현하면 A 18개, B 56개, C 12개, **D 25개**, 캠페인 밖 145개다. A512 calibration과 다른 집합이다.
- 이 함수는 GT score로 sigma를 선택하지 않으므로 이를 곧바로 supervised utility label 유출이라고 부르지는 않는다. 다만 계획의 고정 calibration/holdout 계약과 달라졌고 변경 이력이 없다.
- 또한 단일 평균 Scharr energy의 2% 오차만으로 matched를 선언한다. 계획이 요구한 주파수 특성/overshoot 일치 검사는 없다. `unmatched_blur_control` 처리를 가진 점은 올바르다.
- 조치: A ID를 명시적으로 전달하고 sigma·support·주파수/overshoot의 적합 여부를 기록한다. 기존 결과는 “이전 PALSV18 calibration을 재사용한 exploratory 대조”로 분리할 수 있다.

**Q08 · P2 — 조건부 q shuffle이 a의 연속값과 최소 block 크기를 구현하지 않는다.**

- [k20.py:47](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4/k20.py:47)는 `sort_values(['rank e','rank a'])`를 사용한다. e rank가 고유하면 정렬은 e만으로 결정되며 a는 이웃 선택에 영향을 주지 않는다. 이는 사전 지정된 `(rank e, rank a)` 근접 block과 다르다.
- 최소 크기는 계획의 8 대신 코드의 2다. 3개 fixture가 모두 `ok`였고, 실제 Pair A에서 Ed/Apos cell은 7개여서 이 문제가 현실적으로 발생한다.
- q-label 수와 EA cell을 유지하고 영상/Teacher target을 바꾸지 않는 부분은 맞다. 현재 Pair A의 q-label 변경 비율은 **45.3125%**다. 하지만 partner ID·continuous q_shuffled·e/a 차이가 저장되지 않아 조건부 matching 품질을 검토할 수 없다.
- 조치: 2D rank 거리로 block을 만들고 8 미만은 unmatched로 남긴다. partner·q 전후·e/a 거리·source/texture 분포 및 실제 hard/soft 질량을 저장한다. EAQ−SHUF를 연속 e/a까지 완전히 제거한 효과로 해석하지 않는다.

**Q09 · P2 — native geometry/공간 문맥/선명도 진단이 계획의 필수 범위를 다 채우지 못한다.**

- D20 MS-only/common은 `modality_response(..., bank='A')`만 호출한다. 계획 §8-A의 .5/1 반경 8방향(A/B) 확인에서 대각 B 방향이 빠져 있다. PAN-only의 D10 A/B 전수 계산은 정상이다.
- 기존 estimator를 재사용한 점, canonical=−audit 부호, confidence 미달 I-G를 0으로 대체하지 않는 점은 올바르다. 그러나 `_est`(:79–81)는 실제 `estimate_shift`가 반환하는 **secondary_dy/dx, primary_secondary_diff**를 버린다. RR proxy는 margin/boundary도 최종 row에서 빠진다. 계획 §9-B의 두 estimator 차이와 신뢰도 coverage 분석이 불가능하다.
- native_proxy는 identity/known-shift 결과를 저장하지만 D30 caller는 `sign_ok`를 확인하지 않고 accepted sample의 I-G를 진행한다. 현재 저장된 primary known-shift는 통과하므로 관측된 부호 오류라는 뜻은 아니다.
- D30 상세 native64의 I-G correction/잔여 proxy 및 bias 개입 전후 위치 proxy는 완성되지 않았다. landscape 일부 c_geo 계산만으로 전체 상세 개입을 대체하지 못한다.
- 계획 §4.3/§8의 parent-context·ROI sensitivity 대조는 없다. two_stage_ok는 warp 좌표 tap 범위 검사이며 U-Net 전체 문맥 안정성 검사가 아니다. D40의 edge profile은 RR scene/band별 native만 측정해 primary native64 quadrant나 learned/zero/blur의 출력 폭 차이와 직접 대응하지 않는다. overshoot도 출력 profile 지표로 저장하지 않는다.
- 조치: 필수 작은 subset의 문맥/ROI·secondary vector·band/profile 대응 대조를 추가한다. 이 자료가 없으면 Hspec와 위치 개선/blur 분리는 부분 검증 상태로 남겨야 한다.

**Q10 · P2 — K10 원시 C endpoint와 K20 중간 checkpoint가 저장되지 않는다.**

- [k10.py:109](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4/k10.py:109)의 `run_trial`은 `C_per_sample_after`를 반환하지만 `main`(:167–180)은 이를 버린다. 실제 C128 ID, 각 arm before/after per-sample·band 값, restore hash 자체, optimizer reset 상태도 trial manifest에 남기지 않는다. `restore_ok` bool와 평균만으로 줄인다.
- 따라서 K20 진입 조건인 “원시 C 결과에서 U_soft-hard 재현”을 저장 자료만으로 검증할 수 없다. trial 최종 state도 저장되지 않아 사후 per-sample 재평가 대신 trial 재학습이 필요하다. 이 문제는 아직 시작 전인 K10에서 예방할 수 있다.
- K20은 2,500에서 `endpoint`를 호출하지만 [k20.py:82](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4/k20.py:82)의 save_file은 0/5,000에서만 실행한다. 2,500의 D metric/npy와 **모델 checkpoint 저장**은 다르다. D band endpoint도 없다.
- 조치: trial/arm/step/C sample ID가 있는 원시 검증표와 실제 restore hash를 저장하고, K20 0/2,500/5,000 state·band endpoint를 완성한다. FP32 Teacher target은 현재 매번 source에서 직접 재계산하므로 현재 persistent 영상 cache 충돌이 발생했다고 단정하지 않는다.

**Q11 · P2 — G00 실패가 runner에 전파되지 않고 재개가 파일 의미를 확인하지 않는다.**

- [eqrec4.py:23](/home/knuvi/Desktop/song/PAN-Crafter/tools/eqrec4.py:23)는 `stage=='all'`일 때만 G00 implementation_invalid를 exit 2로 바꾼다. 실제 shell runner는 `python tools/eqrec4.py g00`처럼 단계별 실행한다. 독립 g00 호출에서는 invalid dict를 반환해도 exit 0이다. fixture로 재현했다.
- `Stage.__exit__`는 예외가 없으면 G00도 done으로 기록한다. shell의 `done_stage`는 과거 done 문자열 존재만 확인하고 output/plan/code hash, 이후 error, 파일 완결성을 보지 않는다. 초기 D10 skip→detail_subsets 없음→D20 실패 로그가 실제로 남아 있다. 이후 재기동에서 회복됐지만 안전한 자동 재개 계약은 아니다.
- sources_manifest는 EQREC 구현 commit 전의 `4054135 + dirty=true`만 남겨 당시 uncommitted 코드 내용을 고정하지 않는다. 단계별 code/preprocessing/data/evaluator hash pin도 없다. 이번 감사 snapshot으로 현재 소스는 고정했으나, 이것이 원 실행자가 남긴 provenance를 대신하지는 않는다.
- 조치: standalone G00도 실패 exit/status를 전파하고, stage done은 필수 산출물·입력/코드 fingerprint를 검증한 뒤 인정한다. 현재 관측된 G00 수치 검사는 통과했으므로 기존 D10을 무효로 만들 필요는 없다.

**Q12 · P2 — 자동 보고서만으로 네 집단의 원인 설명을 마무리할 수 없다.**

- `report.py:76–108`은 D50 gradient/one-step, PAN sensitivity, native band alignment를 최종 원인 분석에 읽어 들이지 않는다. 최종 문장에 `[D30/D40/D50 표의 검증된 원인 / 남은 후보: mixed/unresolved로 표시]`라는 작성용 placeholder가 그대로 남는다.
- Table B의 n은 전수 atlas, stress/gain은 D 상세 subset, K10은 B fit-pool utility인데 각 열의 실제 지원 n·분할·trial 수를 구분하지 않는다. 같은 행에 놓는 것은 가능하나 같은 관측 단위로 보이게 하면 안 된다.
- 대표 영상·실패 예·band profile의 위치/폭·구조/복원 landscape 연결도 계획 §16의 수준으로 완성되지 않았다. ledger는 wall time과 GPU 유무/peak memory를 기록하지만 별도 GPU elapsed time/ETA 갱신은 없다.
- 조치: 자동 판정을 수정한 뒤 실제 개입·반증·coverage를 사용해 네 집단별 설명을 작성한다. 자료가 부족한 원인은 `mixed/unresolved`로 남기되 무엇이 관측됐고 무엇이 미검증인지 구체적으로 연결해야 한다.

**5. 예정 K10의 지원 범위에서 주의할 실제 관측**

Pair A의 B 2,048 patch에서 source Student가 Teacher보다 나쁜 `a_T>0`은 **14개**뿐이다. 각 e/q quadrant의 Apos count는 2/5/2/5로 32 미만이다. Aneg는 448/583/562/441개다. 따라서 현재 계획의 최소 고유 sample 규칙을 적용하면 Pair A는 **Aneg 4개 cell × 4 pools = 16개 pool**만 지원한다. 이는 `fit_pools`가 없는 cell을 억지로 채우지 않아 올바르게 처리하는 부분이다.

이 관측 자체가 KD 무효를 의미하지 않는다. 계획처럼 실제 RH/RS의 C utility를 측정해야 한다. 다만 “Teacher가 유리한 네 집단에서도 효과가 확인됐다”는 주장은 Pair A에서 할 수 없다. Pair B utility는 아직 없으며 이 감사에서 대체 생성하지 않았다.

**6. 권장 처리 순서와 보존할 결과**

1. **K20 전에 Q01/Q02/Q08/Q10을 해결**한다. 현재 구성이면 식별 불가능한 경우에도 25K pilot updates를 자동 실행하고, H4의 검증·재현 근거가 부족해질 수 있다.
2. **H2/보고서 전에 Q03/Q04/Q05를 해결**한다. raw probe row가 정상 저장되면 scalar H2는 B-only로 재집계할 수 있다. 잘못 저장된 band delta는 native per-band error를 같은 sample/ROI로 보충하면 전체 stress 학습을 반복할 필요는 없다.
3. D30의 L1E4 native64 I-L/I-Z/I-W 전수를 보충하고, primary 고정 subset 분석과 각 checkpoint 자체 quadrant 분석을 구분한다. Q07/Q09의 누락은 작은 필수 subset부터 보충한다.
4. runner/G00 fail 상태와 stage fingerprint를 보완한 뒤 REPORT를 생성한다. 조건부 K20을 실행하지 않는 결론도 이 계획에서 허용된다.

**보존 가능:** 확인된 checkpoint, D10 atlas·probe residual·A threshold, 실행 전 수정이 반영된 D20 FR 반응, 정상적인 correction/known-inverse 및 loss routing 구현. D30/D40의 이미 계산한 올바른 sample-level 원시 자료도 실험 범위·분할·모델 label을 명시해 보존한다.

**현재 승인 불가:** H2의 A/B 혼합 통계, source를 분리하지 않은 K10 LOO를 독립 검증으로 부르는 주장, 진입 조건 없는 K20의 계획 준수 주장, 현재 자동 report의 Supported/Opposed 판정, 미완성 spectral/blur/문맥 검증을 마친 것으로 해석하는 결론.

감사 재실행 예: `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/knuvi/miniconda3/envs/pancrafter/bin/python research_log/EQREC4_Audit_2026-09-15/verify_readonly.py`. 스크립트 내부에서 CUDA를 비활성화하며 캠페인 파일을 쓰지 않는다. 합성 실패 테스트는 캠페인 성능 실험이 아니고, 작은 actual-checkpoint 검사는 장기 수렴의 보증이 아니다.
