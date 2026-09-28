# PANDA rebuttal 대비: 주장–증거 공백 감사 및 추가 실험 우선순위

작성일: 2026-09-28  
상태: **원고·업로드된 XLSX 분석 및 live Sheet 일부 readback 완료. 실험 실행·Sheet 수정·서버 queue 변경은 하지 않음.**

## 1. 판단의 요약

추가로 필요한 것은 많은 seed나 더 넓은 alpha/beta/edge grid가 아니다. 가장 먼저 필요한 것은
**논문 표에 실제로 연결되는 실험 identity를 복구하고, 이미 완료된 대조군이 주장을 지지하는지 확인하는 것**이다.
그 후 연구 기여에 가장 직접적인 두 질문, 즉 **“q가 평균 가중치 이상의 정보를 주는가?”**와
**“학습된 정합이 실제 기하 오차·미관측 shift를 줄이는가?”**에 계산을 배정해야 한다.

현재 결과는 전부 나쁘다는 뜻이 아니다. 예를 들어 WV3에서 FULL(C07)은 고정 0.5 가중치(C06)보다
5/5 block에서 HQNR이 높고, GF2에서는 FULL이 Student alignment 제거(C09)보다 5/5 block에서
RR ERGAS가 낮다. 그러나 평균 reliability를 사용하는 C15, uniform KD C04, no-consistency Teacher 등
더 직접적인 대조군에서는 주장에 불리한 결과도 반복된다. 이런 결과를 먼저 설명해야 한다.

## 2. 근거와 집계 규약

분석 대상은 `59391_PANDA_PAN_Alignment_Awar.pdf`(12쪽)와 `pan-cvpr27.xlsx`다.
PDF는 p1–9 본문, p10–12 참고문헌, 마지막에 내용 없는 Appendix 제목이 있다.
외부 최신 문헌 조사나 ICLR 제출/답변 정책 검토는 이번 범위가 아니다.

Sheet는 `paper`, `ablations`, `WV3-main`, `QB-main`, `GF2-main`, archive, `PC-Repro`,
숨김 원 실험 탭 및 MAIN-A sensitivity를 확인했다. 아래 일부 핵심 범위는 live Sheet에서도 재조회해
업로드된 snapshot과 일치함을 확인했다.

- `ablations!L7:T12`: 원고 Table3에 대응하는 소형 표.
- `WV3-s1!B144:U144`: C05/P02 실제 원 기록.
- `WV3-s4!B61:U61`: 위 소형 표의 숫자와 일치하는 과거 G23 결과.
- `SENS-MAINA-WV3-s4/s5` 끝부분: 최신 반복 block.
- `paper!B115:N123`: PAN-Crafter replica 선택과 비용 규약.

원시 집계는 업로드 XLSX 셀 값을 사용했다. main 탭에서 run별 `ABLR2 RR_VAL_SELECTED`를
정확히 한 번만 사용했다. SOURCE_SUMMARY, EXACT50K, RAW_MAX, E_MIN_DIAG50을 독립 run으로 세지 않았다.
대조군 비교는 **같은 sensor·같은 P block이 양쪽 모두 있는 경우만** 사용했다.
P block은 Teacher/Student seed 묶음이며, 서로 다른 Teacher를 고정한 Student-only 반복과 구분한다.
검증셋으로 checkpoint를 골랐다는 이유만으로 캠페인 전체를 untouched test라 부르지 않는다.

이 보고서의 평균/표준편차는 기술통계다. p-value나 통계적 유의성을 계산/주장하지 않았다.
전체 20장 평균만 있는 Sheet로는 장면별 paired significance를 확정할 수 없다.
`paired_component_deltas.csv`에 실제 대조한 run, source row, 각각의 값 및 delta를 보존했다.

## 3. 이미 확보한 결과: 재실험보다 재집계가 먼저

| 묶음 | 중복 selection을 제외한 현재 기록 | 실험 계획상의 의미 |
|---|---:|---|
| WV3 ABLR2 | 93개 = Student 83 + Teacher 10 | 5개 block 중 C00–C17+Teacher의 일부만 비어 있음 |
| QB ABLR2 | 91개 = Student 81 + Teacher 10 | P01–P04는 완전, P05 일부 미등록 |
| GF2 ABLR2 | 100개 = Student 90 + Teacher 10 | P01–P05 모두 C00–C17+Teacher 완료 |
| MAIN-A alpha/beta/edge | 98개 run / 196개 selection 행 | 13개 완전 7-case block, anchor 제외 12개 새 seed block |
| PC-Repro | 19개 run/evaluation ID | 학습15개와 WV2 평가4개; selection/동일seed 서버복제를 독립 seed로 세면 안 됨 |

### 현재 미등록된 component case

- WV3 P03: C14.
- WV3 P05: C10, C11, C12, C13, C14, C15.
- QB P05: C07, C08, C09, C10, C11, C12, C13, C14, C15.
- GF2: 없음.

합계16개다. **미등록 = 미실행은 아니다.** 서버의 현재 run, 완료 checkpoint, 평가/업로드 debt를 확인한 후에만
부족한 학습을 배정한다. 완전 5-block 표가 목적이면 이 범위가 최대 보완 대상이다.
당장 q 비교에 필요한 핵심 보완은 WV3 P05 C15, QB P05 C07/C15 등이며,
이미 완료된 나머지 전 case를 다시 돌릴 필요는 없다.

현재 `ablations` 상단의 “3 complete fresh-seed blocks” 표기는 원본 sensitivity 전체보다 적은 집합이다.
완전한 새로운 seed block은 s4 cycle1–5 및 s5 cycle0–6, 합계12개다.
s4 cycle0/seed73101은 anchor로 분리했고, s4 cycle6·s5 cycle7 부분 block은 완전-block 평균에서 제외했다.

## 4. P0: 새 실험 전에 고쳐야 할 증거 연결

### 4.1 Table3는 현재 상태로는 controlled cumulative ablation이 아니다

원고 p9 Table3는 구성요소를 순차 추가하는 표지만 출처는 다음과 같이 섞여 있다.

| 표의 역할 | 연결된 기록 | 문제 |
|---|---|---|
| baseline / frequency / scratch A | ABLR2 P01 C00/C01/C02 | 같은 block이라 이 부분은 비교 가능 |
| “+ self-supervised shift consistency” | ABLR2 P01 C03 | 실제 의미는 TPLUS에서 pretrained A를 받아 초기화; C02에서 consistency만 켠 비교가 아님 |
| “+ reconstruction-reliability guided distillation” | 표기는 C05/P02 | P01에서 P02로 바뀌며 표시 숫자가 실제 C05/P02와 다름 |
| FULL Ours | FH20R1/F1/seed73101 TARGET38380 | ABLR2 P01/P02 C07이 아니라 다른 Teacher·seed·campaign·selector |

특히 C05로 적힌 행은 HQNR **0.9582**, Ds **0.0205**, ERGAS **2.0560**이다.
하지만 `WV3-s1!row144`의 실제 C05/P02 primary는
HQNR **0.9564700275**, Ds **0.0281829499**, ERGAS **2.0459064299**다.
표시된 성능 숫자 묶음은 `WV3-s4!row61`의
`PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S41019_FRESH50_v1`과 일치한다.
이 과거 G23은 **edge와 q routing까지 켜진 W104D121 전체 방법**이며, “e-guided fitting만 추가” 대조가 아니다.
해당 행의 비용은 또 C05 source에서 가져온 상태다.
의도나 복사 경위를 추정하지 않지만 **현 상태의 source label·성능·cost 연결은 성립하지 않는다.**

조치:
1. 각 논문 행을 `run_id, Teacher, Student seed, input layout, selector, step, checkpoint SHA`에 연결한다.
2. ABLR2 표에는 ABLR2 C07을 쓰고 FH20R1 대표 모델은 별도 main-result 행으로 분리한다.
3. 동일 P의 C00–C07 결과, 그리고 각 핵심 contrast의 paired mean±SD를 보고한다.
4. consistency 단독 효과는 C03−C17 또는 TPLUS−TZERO로 본다.
5. 데이터가 내려가는 결과도 포함한다. 단조 증가하는 숫자 사다리를 목표로 다시 선별하지 않는다.

### 4.2 nominal MAIN-A와 원 FH20R1의 실행 동등성부터 확인

MAIN-A s4 cycle0은 seed73101/F1/PLH W104D122 anchor인데,
EXACT50K HQNR은 **0.9532954270**이다.
원 논문 source FH20R1 seed73101/F1의 EXACT50K는 **0.9581666434**다.
이 차이만으로 버그라고 단정하지 않는다. 그러나 둘을 “같은 recipe의 재실행”으로 묶기 전
initial U/A SHA, Teacher SHA, calibration/LP, batch stream, gate detach,
loss reduction, source/runtime flags를 대조해야 한다.

새 학습보다 먼저 고정 batch에서 forward·loss·U/A gradient·optimizer update와 resume parity를 검사한다.
의도적인 recipe 차이가 있으면 별도 cohort로 남긴다.

### 4.3 checkpoint 선택 효과와 학습 효과를 분리

현재 주 WV3/QB는 TARGET 계열의 test-aware 선택을 사용한다.
반면 component primary는 RR_VAL_SELECTED이고 PAN-Crafter replica는 FINAL/VAL 선택이다.

동일 논문 source의 재계산 없이 확보된 비교:
- WV3 TARGET38380: HQNR 0.9587325530 / ERGAS 2.056060194.
- WV3 RR_VAL_SELECTED49490: HQNR 0.9581686530 / ERGAS 2.050996543.
- QB TARGET47470: HQNR 0.9251135522 / ERGAS 3.561287322.
- QB VAL=EXACT50K: HQNR 0.9250839490 / ERGAS 3.564849561.

따라서 checkpoint 선택을 통일한 표는 지금 바로 재구성할 수 있다.
MAIN-A 새 seed12개 BASE 평균도:
- EXACT50K: HQNR **0.9532908651**, ERGAS **2.0492160596**.
- HQNR_MAX50: HQNR **0.9572082891**, ERGAS **2.1160805299**.

HQNR 선택의 이득과 RR 손해를 함께 보고해야 한다. 기존 캠페인 selector를 소급해 변경했다고 표현하지 않는다.
후속 실험은 시작 전에 selector를 고정한다. 추천은 native RR validation selection primary와
EXACT50K secondary이며, 기존 HQNR_MAX50은 development diagnostic으로 분리한다.
이미 개발에 사용한 FR20을 다시 나누어 “새 독립 test”라고 부르지 않는다.

### 4.4 GF2 학습량 및 PANMIX는 본문 설정과 다르다

원고 p8은 각 stage50K라고 적는다. 현재 GF2 main 결과의 source는
`GFP40_GF2_S4_B09_MIX_G050_FT20K_RS98202_v1`이고,
Teacher R4_100은100K, Student는 parent100K+추가20K=**누적120K**이며 G050 MIX를 사용했다.

따라서 추가 학습과 PANMIX 효과를 PANDA 기본 recipe 효과와 분리해야 한다.
이미 P40 s4 B07/B08, B10/B09의 두 matched CTRL/MIX pair가 있으므로 우선 이들을 재활용한다.
특히 B10 CTRL과 B09 MIX는:
- CTRL: HQNR0.9546301994 / ERGAS0.5483030588.
- MIX: HQNR0.9650681941 / ERGAS0.5510757826.
- Δ: HQNR +0.0104379947, ERGAS +0.0027727238.

같은 endpoint에서 공간/주파수 처리의 추가 효과가 크다. 본문에 이 절차를 명시하거나,
메인 표를 논문에 기술한 native50K recipe 결과로 맞추고 long/MIX를 별도로 분리한다.
새 실험은 이런 분리 후 정말 부족한 동일-budget comparator만 수행한다.

### 4.5 PAN-Crafter replica와 FLOPs의 공정한 비교

원고의 PAN-Crafter 행은 저자 reported 수치가 아니라 Selected PC-Repro 수치다.
`reported`, `official-code run`, `local Eq11 reconstruction`을 서로 다른 출처로 표시해야 한다.
동일20장·band/DN·metric protocol·checkpoint selection인지 확인한다.
공식 코드의 문제를 제기하더라도 현재 local reconstruction이 저자 원 논문 모델과 동일함을 자동 보증하지 않는다.

`paper!B117`은 PC-Repro FLOPs가 **partial FLOPs=2×MACs**이며 legacy THOP 값과 직접 정규화되지 않았다고 적는다.
Ours는 THOP MAC-scale 계열이다. 따라서313.93 대57.10만으로 계산한 **81.8% FLOPs 감소**는
현재 서로 다른 convention을 나눈 수치다. 단순2배 보정만으로 모든 누락 연산까지 해결됐다고 주장하지 않는다.

같은 RTX5090/runtime/batch1/FP32 및 동일 RR256·FR512에서
params, MACs/FLOPs 정의, 입력 frontend를 포함한 end-to-end latency, peak memory를 다시 측정한다.
warmup·CUDA synchronization·repeats·포함 연산·cache/preprocessing 비용을 명시한다.
이것은 긴 재학습이 아니라 profiling/정리 작업이며 rebuttal에서 방어 가치가 크다.

## 5. 실제 component 결과가 말하는 공백

### 5.1 q의 sample-wise 정보성은 아직 입증되지 않았다

C15는 단순 q 제거가 아니라 **train에서 측정한 mean reliability**를 넣는 대조다.
C07(FULL)−C15의 같은 P-block primary 차이는 다음과 같다.

| Dataset | Matched blocks | ΔHQNR (FULL − mean-q) | HQNR wins | ΔERGAS |
|---|---:|---:|---:|---:|
| WV3 | 4 | -0.0011515 | 1/4 | -0.0013727 |
| QB | 4 | -0.0022435 | 3/4 | +0.0038822 |
| GF2 | 5 | -0.0013703 | 2/5 | +0.0008371 |

양의 HQNR, 음의 ERGAS가 각각 유리한 방향이다.
이 수치만으로 유의한 차이가 있다고 주장하지 않는다. 그러나 “q의 일관된 우월성”을 주장할 자료도 아니다.
QB는3/4 block에서 HQNR이 높지만 P04의 큰 음의 차이로 평균이 음수다. 해당 block을 임의 제외하지 않는다.

반대로 C07−C06(고정0.5)은 WV3에서5/5 HQNR이 높고 평균 +0.0017362다.
**고정0.5 대조와 실제mean-s 대조에서 결론이 달라지는 것 자체가** 평균강도·최적화 drift·sample 대응을
분리할 실험이 필요하다는 근거다.

### 5.2 reconstruction-error fitting도 묶음 기여를 다시 봐야 한다

C05(adaptive hard/soft)−C04(uniform KD), 같은 P01–P05:

| Dataset | Paired blocks | mean ΔHQNR | mean ΔERGAS | ERGAS improvement blocks |
|---|---:|---:|---:|---:|
| WV3 | 5 | -0.0009597 | +0.0065506 | 0/5 |
| QB | 5 | -0.0077334 | +0.0187458 | 0/5 |
| GF2 | 5 | +0.0001196 | +0.0049620 | 0/5 |

세 센서15/15 block에서 RR ERGAS가 증가했다.
이것을 곧바로 모든 형태의 KD가 무효라는 뜻으로 해석해서는 안 된다.
C04와C05의 exact config/routing을 대조하고, hard emphasis와 soft gating을 분리해서
어떤 항이 어느 목적을 개선/악화시키는지 확인해야 한다.
균일 KD보다 우월하다는 주장은 현재 그대로 방어하기 어렵다.

### 5.3 alignment 및 consistency는 native 평균만으로 설명되지 않는다

- C03−C17은 consistency를 사용한 Teacher의 A prior와 사용하지 않은 prior의 비교다.
- TPLUS−TZERO는 reference 학습에서의 consistency 대조다.
- C07−C09는 최종 Student alignment 존재, C07−C16은 그 추가 적응을 묻는다.
- WV3 TPLUS는TZERO보다5/5 block에서 native HQNR이 낮다.
- GF2 C07은C09보다5/5 block에서 ERGAS가 낮고, 평균 개선은0.0170553이다.
- 반면 native HQNR의 alignment/consistency 기여는 sensor·block에 따라 다르다.

즉, “해당 항을 넣으면 native HQNR이 단조 증가”보다
**native reconstruction과 controlled-shift behavior를 함께 측정하는 증거**가 필요하다.

## 6. 신규 증거의 우선순위와 최소 설계

### R2 / P1 — q 정보성과 e와의 상보성

원고 Fig5의 네 가지 예시는 e와q가 완전히 같지 않음을 예시할 뿐,
q가 geometry-sensitive supervision의 유효한 신호임을 증명하지 않는다.

먼저 추가학습 없이 train/val의 고정 reference에서:
- per-sample q와 patch-mean e를 집계하되 픽셀을 독립 sample처럼 세지 않는다.
- e가 비슷한 bin 안에서 q가 held-out shift의 reconstruction degradation 또는 독립적인 geometry 지표를 예측하는지 본다.
- rank correlation만 보고 “독립적”이라고 하지 않는다. e-conditioned explanatory value가 중요하다.
- qref·q·s의 min/quantile/mean/std, loss/gradient norm 및 실제 optimizer update norm을 기록한다.

최소 학습군:
1. FULL(C07 의미).
2. train mean-s(C15 의미).
3. q-shuffle: weight 값들의 multiset을 유지하고 sample 연결만 e-stratum/실제 augmentation view 안에서 섞는다.
4. e-surrogate: sample-level e를 사용하되 q 가중치와 같은 marginal distribution을 갖도록 rank-matching한 geometry weight.

원 q/e/Teacher는train에서만 계산한다. shuffle/rank mapping은 시작 전에 고정한다.
각 비교는 Teacher, init U/A, stream, LR, update budget, selector를 공유한다.
기존 FULL/mean이 exact recipe상 재사용 가능하면 새2조건×3사전고정block=6 Student run이 최소 시작점이다.
WV3에서 먼저 실시하고 GF2 또는QB에서 결정적인 contrast를 확인한다.

### R3 / P1 — 정합과 미관측 perturbation에 대한 inference-only 검사

기존 C03/C17/C07/C09/C16, TPLUS/TZERO 및 비교 모델 checkpoint를 사용한다.
추가훈련 없이 다음 패널을 만든다.

- 추가 PAN shift: HR pixel 단위0,±0.25,±0.5,±1,±2,±4; 축 방향과 대각/미관측 방향.
- 알려진 training/probe range 안과 밖을 명시한다. 위 값은 제안이며 실제 aligner 범위와 E를 확인해 확정한다.
- correction response: `||u(P_eps,M)+eps-u(P,M)||`.
- downstream RR: ERGAS/PSNR/SAM/edge error 대 shift magnitude.
- 비교 가능한 고정 common valid support, 같은 보간·padding. shift가 클수록 다른 유리한 mask를 적용하지 않는다.
- native pair의 before/after: GT/MS geometry에 대응하는 독립 landmark/edge 위치, 실패 scene 및 보정량 분포.
- 영변위·평탄/texture-poor 영역과 국소변형/회전은 한계 진단으로 분리한다.

**기존 native PAN–MS pair의 true displacement를0으로 가정하지 않는다.**
추가shift의 상쇄 검사는 relative equivariance이며 absolute native registration accuracy와 다르다.
원고도 q가 absolute registration accuracy가 아니라고 명시하므로, q와 같은 수식 하나만으로 q의 유효성을
다시 증명하는 순환 논리를 피한다. 독립적인 reconstruction degradation/geometry 결과를 함께 본다.
global2D 모델에 조밀한 local flow의 완전한 복구를 요구하는 대신 적용 범위를 명확히 한다.

### R4 / P1 — soft advantage와 gradient route의 직접 대조

C04/C05/C12/C13부터 재집계하고 다음 진단을 기존 저장 시점에서 먼저 얻는다.
`a_T>0` 비율, soft weight 평균, hard/soft/edge gradient 비, Teacher보다Student가 좋은 영역 비율,
U/A update norm, aligner drift 및 q에 따른 분포. raw loss scalar만으로 기여도를 결론내리지 않는다.

그 다음 우선순위대로 최소 변형을 추가한다.
- `a_T` 제거: (1−d_T)는 유지하고 advantage gate만 제거.
- q를 edge에만 사용 / A-hard에만 사용 / 둘 다 / 둘 다mean으로 두는2×2.
- U의 objective와 A-hard의 q 가중치를 그대로 두고, A에 soft/edge gradient를 추가로 허용하는 대조.
  단순히 `total.backward()`로 바꿔 A-hard 가중치까지 1로 바꾸면 서로 다른 효과가 섞인다.
  soft와 edge를 함께 허용한 실험은 “두 경로를 함께 연 효과”로만 해석한다.

FULL과mean controls를 재사용할 수 있으면 no-a, q-edge-only, q-A-only, joint-gradient의4조건이 신규다.
각 실험의 변경 항목을 사전에 고정하고 3개 이상의 matched block으로 시작한다.
no-a는 soft 감독의 평균 강도도 바꾸므로, 결과 해석에 필요하면 평균 soft budget을 맞춘 보조 대조를 추가한다.
Student A frozen(C16)은 joint-gradient 대조를 대신하지 못한다.

### R5 / P1 — alignment를 “먼저” 하는 위치의 직접 비교

C09는 alignment가 없는 모델이어서 early가late보다 낫다는 증거를 주지 않는다.
같은 reconstruction backbone, global2D correction의 유사 capacity, loss와seed를 유지하고
no-align / input-align / feature-align을 비교한다. 신규late arm만 최소3개 matched block으로 시작할 수 있다.
native 성능과R3의shift stress를 같이 보고, cost차이도 기록한다.

PAN→MS와MS→PAN 방향까지 비교하려면 최종 출력과GT의 좌표계를 같게 만들어야 한다.
MS를warp한 모델을 원GT에 그대로 비교해 생긴 불이익을 align-first의 증거로 사용하지 않는다.
구현·시간이 제한되면 우선early/late 위치 비교가 더 단순하다.

### R6 / P2 — 현재 paper 모델의 WV2 zero-shot

과거 WV2 archive와 PC-Repro WV2는 현재 FH20R1/PANDA source의 결과를 대신할 수 없다.
정리본에는 현재 paper의 WV3 checkpoint로 평가한 완료 WV2 결과가 확인되지 않았다.
서버에 이미 결과가 있으면 회수·등록하고, 없으면 기존 checkpoint로 RR/FR inference-only를 수행한다.
WV2용 MTF와 sensor metadata를 사용하며 WV3 상수가 하드코딩된 FR evaluator를 그대로 쓰지 않는다.
WV2를 보고 checkpoint, weight 또는 calibration scale을 바꾸지 않는다.
동일 family의 no-align 또는 mean-q 모델도 전이하면 일반화와 mechanism을 연결할 수 있다.
이 작업은 P2이지만 추가 학습이 없어 다른 P1 학습과 병렬로 먼저 완료해도 된다.

### R7 / P2 — 최종 calibration scale 및 reference 조건부 강건성

현재 MAIN-A의 alpha/beta/edge는 이미 충분한 반복이 있다. 다음 항목이 필요하면 좁게 보완한다.
- qref 및 tau_rec: train calibration 원값의 ×0.5/1/2.
- aligner LR 비율: .01/.03/.06.
- 가능하면 reference 2개에서 가장 중요한 contrast만 확인.

과거 G23의 P0/W104D121/T0 결과를 현재 PLH/W104D122/F1 결과로 합산하지 않는다.
lambda_con 및 probe radius는 R3에서 행동상의 문제가 확인된 경우에만 추가한다.
s4/s5의 현재 alpha/beta/edge cycle 확대보다 R2–R5가 먼저다.

### R8 / P2 — 가장 가까운 비교 및 공정한 정성 패널

원고가 관련연구로 논의하는 U-Know-DiffPAN과 SIPSA 계열은 단일 reliability와 정합의 가까운 비교축이다.
released weight/raw 또는 기존 결과가 있다면 동일 평가로 보완할 가치가 있다.
모든 기존 비교 모델을 재학습하는 일을 rebuttal의 선행조건으로 두지는 않는다.

정성적으로는 C07/C15/C09 등 우리 대조군의 raw와 동일 GT를 사용해 같은 오차 수식·색상 범위로 새 패널을 만든다.
기존 PNG renderer의 역복구나 calibrate는 사용자 방침대로 재개하지 않는다.
수치 범위를 모르는 타 방법 PNG와 색을 비교해 오차가 낮다고 주장하지 않는다.
표본은 e/q strata 또는 사전 지정 scene으로 선정하고 실패 사례도 함께 제시한다.

## 7. 실제 계산 자원 배정 권고

**자동 queue 변경 지시가 아닌 우선순위 제안**이다.

첫 묶음 — 추가 학습이 없거나 매우 작은 작업:
- R0 표·selector·recipe audit와 기존 pair 재집계.
- R1 동일 profiler와 GF2 CTRL/MIX·native50K 대비.
- R3 기존 checkpoint의 shift tests.
- R6 WV2 현재 source 결과 회수 또는 inference.
- q/e 및 gradient 진단.

두 번째 묶음 — 제한된 조건의 신규 학습:
- q-shuffle 및 e-surrogate.
- 필요한 경우 route-only/no-a 대조.
- late-alignment 한 baseline.
- 미등록 ABLR2는 같은 P의 불완전 대조를 채우는 범위에서만 보완.

후순위:
- alpha/beta/edge에서 더 많은 seed를 무기한 반복.
- 더 넓은 width/depth/attention grid.
- 논문이 claim하지 않는 범용 backbone 확장 전면 실험.
- renderer 복구나 ROI 편집을 핵심 증거 확보보다 먼저 수행하는 작업.

실제 GPU 시간은 현재 runtime에서 추정하지 않았다. 현재 checkpoint 보유 여부, raw/gradient log,
서버 실행 상태를 확인하지 않았기 때문이다. 동일 run의 재시도를 독립 seed로 세거나
성능이 나쁜 case를 삭제하지 않는다.

## 8. rebuttal에 준비할 최소 증거 패키지

1. **출처가 정확한 paired ablation 표**: e-guided, q-vs-mean, no-A, no-consistency, frozen-A 각각의 n과 mean±SD.
2. **q 정보성과 route 표**: mean/shuffle/e-surrogate 및 최소 route-only 대조로 평균 강도와 sample 정보를 분리.
3. **controlled-shift 곡선**: relative response와 RR degradation을 함께, 미관측 범위와 실패 조건을 명시.
4. **공정한 comparison/compute 표**: reported와 reproduced를 구분하고 학습 budget, selector, MAC 정의를 통일.
5. **WV2 zero-shot 표**: 현재 source를 미리 고정한 결과.

전체 case 표, run/seed/Teacher/source/step 목록과 per-scene 통계는 대응용 원자료로 유지한다.
짧은 rebuttal에는 평가자의 핵심 질문에 직접 답하는 contrast를 제시하되,
불리한 결과를 숨기기 위한 축약은 하지 않는다.

## 9. 논문 기술만으로 보완 가능한 재현성 항목

lambda_con, synthetic shift 분포·주기·범위, probe 집합 E, qref/tau calibration 표본,
gate의 stop-gradient 규칙, warp 부호·padding·align_corners·보정 범위,
Teacher/Student 폭·depth 및 LP/HP recipe를 명시해야 한다.
Fig4의 114채널 표기와 현재 모델명 112/104의 관계도 정리 대상이다.
이들은 새 실험 부족과 혼동하지 않고 implementation/protocol 표로 정리한다.

## 10. 출처 위치와 산출물

- 원고 p2–3: align-first와 두 cue 주장. p5: global translation과 consistency.
  p6: Fig5와 Eq7. p7: Eq9–12. p8: 구현, Table1, 비용. p9: Table2–3.
- Live Sheet:
  https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit
- XLSX source SHA256: `d120bc815728ea5842eecf282e4d492235ceac9908c56cbb20065115b8d370b2`.
- C05 실제 source: WV3-s1 row144. 과거 동일 tuple: WV3-s4 row61 / WV3-archive row927.
- GF2 extended recipe: GF2-P40-s4 row10 및 CTRL row11; GF2-main rows1033–1035.
- Component: WV3-main/QB-main/GF2-main의 ABLR2 RR_VAL_SELECTED.
- MAIN-A: SENS-MAINA-WV3-s4/s5 원행; 12개 완전 새 seed block으로 계산.
- Case 정의 보조 확인: 저장소
  `ablr2/legacy/PANDA_ABL_S1_WV3_S2_QB_AdaptiveRepeat_ExperimentPlan_2026-09-21_v2.md`.
  C06의 sE=sA=.5, C03의 Teacher A transfer, fresh run 관계를 확인했다.
- 원격 프로세스, checkpoint/raw의 현재 존재, 서버의 현재 진행 단계는 검증하지 않았다.

동봉 CSV:
`component_primary_records.csv`, `paired_component_deltas.csv`, `paired_component_summary.csv`,
`missing_component_cases.csv`, `maina_12block_summary.csv`, `maina_source_selections.csv`,
`rebuttal_priorities.csv`.

결론: **핵심은 새로운 best score가 아니라 현재 핵심 주장의 원인을 분리해 검증하는 것**이다.
정확한 paired 대조 후에도 q 또는 adaptive fitting의 이득이 지속되지 않으면
조건부 효과나 trade-off로 주장을 좁혀야 한다.
