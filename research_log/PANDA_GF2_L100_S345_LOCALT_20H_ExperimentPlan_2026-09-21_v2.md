# PANDA GF2 L100 LOCAL-T — s3·s4·s5 독립 Teacher 우선 20시간 실험
**작성일:** 2026-09-21  
**Campaign:** `PANDA_GF2_L100_S345_LOCALT_20H_20260921_v2`  
**수정:** INDEP1 — 서버별 local Teacher → local calibration → local Student  
**대상:** s3·s4·s5, 사용자 지정 5090 서버. s1·s2는 신규 기동·변경 대상에서 제외.  
**고정:** P0/W112D123 Teacher → PLH/W104D122 Student, 현행 PAN Aligner·단일 HRMS·gradient routing.  
**문서 상태:** MD/CSV/설계 registry/로컬 queue 수정 완료. 원격 코드 배포·학습 기동·중단·live Sheet 쓰기는 수행하지 않았다. 이 번들은 실행된 결과나 drop-in runner config가 아니다.

## 0. 이번 변경의 결정

**각 서버는 자기 Teacher를 먼저 학습한다. 다른 서버 Teacher의 완료·전송·성능 판정을 기다리지 않는다.** Teacher endpoint와 자기 calibration의 무결성이 확인되면 로컬 Student로 바로 넘어간다. 낮은 Teacher HQNR, 높은 유효 q, 다른 서버의 미완 상태는 core Student를 막는 성능 gate가 아니다.

| 서버 | 먼저 수행하는 local Teacher | 다음 local Student | 다른 서버 참조 의존성 |
|---|---|---|---|
| **s3** | TS94001 fresh100K → calibration → 같은 TS fresh50K → calibration | T50/S50, T50/S100, T100/S100의 BASE 대조 ×2 Student seed | 없음 |
| **s4** | **TS94002 fresh100K → calibration** | 자기 R100으로 BASE↔H010, S100 ×2 seed | 없음 |
| **s5** | **TS94003 fresh100K → calibration** | 자기 R100으로 BASE↔E1, S100 ×2 seed | 없음 |

s3의 T50는 기존에 합의한 학습 길이 분리용 로컬 대조로만 남는다. s4·s5는 이를 기다리지 않는다. 세 서버의 주 Teacher는 모두100K다. Teacher seed를 다르게 정한 것은 독립 reference를 명시하기 위한 식별 결정이지, 좋은 seed를 선별한 결과가 아니다.

- **기본 실행: Teacher4 + Student14 = 18개.** Teacher는100K×3+50K×1, Student는100K×12+50K×2다. Calibration4개는 학습 case 수와 별도로 비용·상태를 기록한다.
- **ARW/E02 4개는 등록만 보존하고 기본 queue에서 제외한다.** 상태는 `DEFERRED_USER_RELEASE`; 사용자 재지시 없이 자동 개방하지 않는다. 총22개 등록이22개 실행이라는 뜻은 아니다.
- 20h 창, 16h 신규 block cutoff, 18h optimizer 종료, 18–20h 보존·평가는 유지한다. 공통 clock은 시간 기준이지 서버 간 성능 barrier가 아니다.
- 새 진단의 핵심을 반영해 **정합 억제보다 충분한 복원 학습과 로컬 재현성 확인을 우선**한다. A LR를 일괄 낮추거나 A를 freeze하지 않는다.
- 이전 shared-reference v1의 **미시작 queue만** 이 설계로 대체한다. 이미 시작된 run은 reference를 중간 교체하거나 새 ID로 재명명하지 않는다.

## 1. 추가 진단을 반영해 수정한 해석

### 1.1 근거의 범위

이번 수정은 기존 L100 MD/CSV와 사용자가 추가한 Q1/Q2/Q3 진단 요약을 읽어 작성했다. 새 진단의 원 JSON·estimator·GPU 출력과 live Sheet는 이번에 다시 조회하거나 재실행하지 않았다. 아래 수치는 제공된 표본/run에 한정된 보고값이다. 원 추정기 사이의 부호·norm·cos 정의가 다르므로 다른 문서의 절대값을 조용히 합치지 않는다. [S1,S3]

| 추가 관측 | 이번 판단 | 실행 변경 |
|---|---|---|
| GF2 Teacher clone 일치, cos(initial,current)≈.995, 같은 방향으로13–19% 보정 증폭; 추정기 기준 잔여 감소 | 관측한 run에서는 유효한 정합이 훼손됐다는 증거보다 유지·소폭 개선된다는 근거가 강함 | **초기 A 보호 ARW를 자동 실행하지 않음. BASE A LR 유지** |
| 비슷한 c에서도 Student seed별 Ds 차이가 남고 후반 큰 붕괴 없음 | U의 복원 학습·seed 민감도를 우선 점검할 단서. U 단독 원인으로 확증한 것은 아님 | 100K 학습과 로컬 BASE 대응 유지 |
| d_T 분포가 GF2/QB에서 유사하고 최난 decile의 hard 질량 변화가 약3%p | GF2에서 α 재가중이 유독 강하다는 기존 가설은 약화 | H010은 **민감도 대조**로 유지하되 주요 원인 해결이라고 미리 설명하지 않음 |
| soft/hard≈.009, weighted edge/hard≈3e-4 | 해당 측정에서 U gradient는 hard가 지배. 작은 gradient도 장기 경로 효과가 없다는 증명은 아님 | E1 재검정은 유지; E02의10배 감소는 **사용자 재지시 전 보류** |
| GF2 patch 경계 오차/강조 증가 | 기전 미분리; 보정·padding·receptive field 등이 함께 작용할 수 있음 | border/interior 기록만 추가. GT·loss border mask 변경 없음 |
| s_i가 약.5 부근의 좁은 범위 | 강한 선택적 A weighting이라고 설명하지 않음 | q 정의·calibration 유지. 임의 temperature/uniform 치환 없음 |
| 관측 train patch e_S/e_T≈1.011, Student 우위 patch27–40% | 현재 sample의 Student 복원 학습량을 늘려 볼 근거 | Teacher·Student의100K를 주축으로 유지 |

### 1.2 해석상 경계

`c`는 모델의 보정값이며 displacement GT가 아니다. PAN↔LS-fit GT의 전역 이동 추정치는 등록 GT가 아니다. 비슷한 c와 서로 다른 Ds는 U를 우선 볼 단서지만, 작은 c 차이의 비선형 효과까지 배제하지 않는다.

Teacher보다 Student가 나은 화소라고 GT fitting이 끝난 것은 아니다. d_T와 advantage의 음의 상관만으로 hard 강조가 잘못됐다고 판정하지 않는다. τ가 median이라는 사실은 중앙 부근의 정규화 구조를 설명하지만 전체 d_T 분포의 동일성을 수학적으로 보장하지 않는다.

보고된 train L1, RR ERGAS, FR Ds/Dλ는 서로 다른 표본·해상도·목적이다. FR GT가 없으므로 무참조 이득만으로 실제 정합·복원 정확도를 확정하지 않는다. 기존 과거 DUAL의 높은 HQNR도 폭·task·입력·modulation이 함께 달랐던 참고 관측이며 PAN reconstruction 단독 효과가 아니다. [S3,S4]

## 2. WV3 및 이전 GF2 설정과의 관계

비교 기준 WV3는 **최종 F1→PLH/W104D122/BASE**다. 그 이전의 W168 DUAL·PAN reconstruction 실험을 WV3의 최종 설정으로 혼동하지 않는다. 아래 수치는 source 문서가 보장하는 범위이며, actual GPU/TF32 설정은 원 runtime receipt가 필요하다. [S2–S4]


| 항목 | WV3 최종 고정 | 기존 GF2 QG40/G20 | 이번 LOCAL-T v2 | 구분 |
|---|---|---|---|---|
| Teacher backbone / task | P0 · W112 · D[1,2,3] · 단일 HRMS | 동일 | 동일; 50K/100K Teacher만 분리 | 구조 유지 [S1,S2] |
| Student backbone / task | PLH · W104 · D[1,2,2] · 단일 HRMS | 동일 | 동일 | 구조 유지 [S1,S2] |
| PAN 재구성 / MARs / mode modulation | OFF | OFF | OFF | 옛 s1 DUAL만 ON; 현재 센서 간 차이가 아님 [S4] |
| Attention / backbone norm | OFF / LN | 동일 | 동일 | 구조 유지 |
| PAN Aligner / 좌표 | 기존 2D global translation; MS/GT 고정 | 동일 | 동일; 보정 gain·phase 변경 없음 | 구조·추론 유지 |
| MS bands / bit / maxDN | 8 / 11 / 2047 | 4 / 10 / 1023 | GF2 그대로 | 센서별 입력·출력 차원 |
| T / S / A-MS 입력 채널 | 9 / 11 / 8 | 5 / 7 / 4 | 5 / 7 / 4 | band-dependent tensor shape |
| 정규화 / 역변환 | 2DN/2047−1 / (y+1)×1023.5 | 2DN/1023−1 / (y+1)×511.5 | GF2 그대로 | 단위 통일; 동일 DN 수치를 그대로 비교하지 않음 |
| train / validation patch 수 | 9,714 / 1,080 | 19,809 / 2,201 | GF2 그대로; actual N 확인 | 데이터 규모 차이 [S4] |
| 학습 crop / 배율 | PAN64, MS16 / ×4 | 동일 | 동일 | 공간 scale 유지 |
| Teacher 초기화 | fresh U/A; 마지막 A head zero | 동일; GF2 자체 학습 | s3 TS94001 T50/T100; s4 TS94002 T100; s5 TS94003 T100 | donor·기존50K 이어학습 아님 |
| Student 초기화 | fresh U + Teacher A 독립 clone | 동일 | 동일; SS95001/95002 | reference/seed마다 식별 |
| Teacher / Student updates | 50K / 50K | 50K / 50K | T50+T100 대조; S100 중심, S50 대조 | 이번 주 변경 |
| 50K 명목 표본 노출 | 약247.07회 | 약121.16회 | 100K는 약242.31회 | N과 batch로 계산; 실제 독립 관측/최적 epoch 아님 |
| Batch / optimizer / wd | 48 / AdamW(.9,.999), eps1e−8 / .01 | 동일 | 동일 | longer schedule의 누적 update·decay는 증가 |
| U peak LR (T/S) | 1e−4 / 1e−4 | 동일 | 동일 | LR 크기는 고정 |
| A peak LR (T/S BASE) | 1e−5 / 3e−6 | 동일; G20 A1/A9는 별도 대조 | 동일; ARW는 사용자 재지시 전 보류 | 상시 A LR sweep 반복 안 함 |
| Scheduler / warmup | cosine50K / 100 | 동일 | cosine50K 또는 cosine100K / 100 | 100K의 중간50K는 fresh50K와 다른 경로 |
| Teacher λcon / synthetic shift | 1e−4 / radius2, odd update A-only | BASE 동일; C030/C300 대조 이력 | 1e−4 고정; radius·빈도 유지 | Teacher 길이와 λcon를 동시에 바꾸지 않음 |
| Student α / β / λE BASE | 1 / .1 / .002 | 동일 | BASE 유지; H010=.1/.1/.002; E1=1/.1/.001 | 한 번에 하나의 Student 축 |
| τR: 대표 reference 측정값 | 0.012118559330701828 | R0:0.005695626139640808 | 로컬 reference 4개 각각 재측정, 미측정=null | tuning knob 아님 |
| qref: 대표 reference 측정값 | 0.4532603621482849 | R0:0.4086490869522095 | 로컬 reference 4개 각각 재측정, 미측정=null | native displacement GT 오차 아님 |
| Calibration 규칙 | train3072, q AXIS16·ROT4 | 동일 규칙; GF2 전체 q-cache 재생성 | 같은 GF2 calibration base IDs, 로컬 local reference별 τ/q/cache | 정의는 고정, 수치는 reference-dependent |
| LP / HP / warp | Gaussianσ1.98·k41·offset2; signedHP; synchronized bicubic | 동일 레시피; GF2 PAN으로 재생성 | 동일 | 데이터 phase·MTF 튜닝 제외 |
| Gradient routing | U: hard+soft+q-edge; A:q-hard만 | 동일 | 동일; H010는 기존 α를 바꿈 | 새 loss·PAN target·direct shift KD 없음 |
| Precision / GPU 환경 | FP32, AMP OFF; runtime flags 개별 기록 | 동일 방침, 서버별 실제 환경 확인 필요 | s3–s5만, 사용자 지정5090; flags/init/stream 검증 | 같은 GPU명만으로 수치 동등성 주장 안 함 |
| RR / FR 평가 | Q8 / WV3 MTF·nativePAN | Q4 / GF2 MTF·nativePAN | GF2 공식 규약 고정 | Q/MTF/bit 차이는 센서 대응; 성능맞춤 변경 없음 |
| 대표 checkpoint 의미 | 50K 학습의 TARGET38380; test-aware | Exact50K·VAL 주 분석, TARGET 별도 | EXACT_FINAL·VAL 주 분석; RAW/TARGET 별도 | WV3 선택step을 GF2 학습종료로 복제하지 않음 |

**독립 Teacher로 바뀌는 것은 서버 간 reference 동일성이지 제안 architecture가 아니다.** 공통 d_T/q 계산식은 유지하되 값·cache·clone 초기 A는 각 local Teacher에 따라 달라진다. s4 H010과 s5 E1의 raw 평균 차이를 α 대 edge의 효과로 해석하지 않는다.

GF2 train19,809와 WV3 train9,714에서 batch48·50K의 명목 노출은 약121.2회와247.1회다. GF2 100K는 약242.3회로 WV3의 기존 명목 노출에 근접한다. 이것은100K 최적성이나 FR 개선 보장이 아니다. 실제 N·sampler·view 노출은 실행 시 확인한다. [S1,S2,S4]

## 3. Method와 optimizer — 변경하지 않는 것

### 3.1 입력·출력

\[
M=U_4(MS),\quad L=U_4(LPAN),\quad c=A(P_{m4},M_{m4}),
\]
\[
\widetilde P=W(P,c),\quad\widetilde L=W(L,c),\quad\widetilde H=\widetilde P-\widetilde L,
\]
\[
Z_T=M+F_T([\widetilde P,M]),\quad
Z_S=M+F_S([\widetilde P,\widetilde L,\widetilde H,M]).
\]

Teacher P0/W112/D123, Student PLH/W104/D122, HRMS4-band, LN, attention OFF, mode modulation OFF, PAN image reconstruction OFF다. Aligner는 기존 sample-wise2D translation·margin4를 유지한다. 각각 자기 A를 사용하며 Student inference에 Teacher c를 쓰지 않는다.

MS/GT/output frame은 고정한다. PAN/upsampled LP는 같은 FP32 bicubic grid, border, align_corners=False로 warp하고 signed H를 뺄셈으로 만든다. LP는 native PAN의 Gaussianσ1.98/k41/replicate/[2::4,2::4], float64 생성→FP32 cache다. gain/clamp/TTA/blending/HP scaling/MS phase 변경을 넣지 않는다.

### 3.2 Teacher loss

\[
L_T(t)=\operatorname{mean}|Z_T-Y|+
\mathbf1[t\bmod2=1]10^{-4}\frac1{2B}\sum_i
\|c_{\epsilon,i}+\epsilon_i-\operatorname{sg}(c_{0,i})\|_1.
\]

Native reconstruction은 A/U 모두 갱신하고 synthetic PAN은 A-only branch다. ε는 radius2 HR-pixel 원판 면적균등, 독립 CPU RNG를 사용한다. native reconstruction의 c는 live다. Teacher100K는 더 많은 update·synthetic exposure를 포함하며 이를 λcon 변화라고 부르지 않는다.

### 3.3 Student loss와 routing

\[
e_T=\operatorname{mean}_b|Z_T-Y|,\ e_S=\operatorname{mean}_b|Z_S-Y|,\quad
 d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},
\]
\[
a_T=\operatorname{sg}\frac{[e_S-e_T]_+}{e_S+10^{-6}},\quad
s_i=\operatorname{sg}\frac{q_{\rm ref}}{q_{\rm ref}+q_i},
\]
\[
\ell_{H,i}=\operatorname{mean}_p[(1+\alpha d_T)e_S],\quad
\ell_{K,i}=\operatorname{mean}_p[\beta(1-d_T)a_T\operatorname{mean}_b|Z_S-Z_T|],
\]
\[
L_U=\operatorname{mean}_i[\ell_{H,i}+\ell_{K,i}+\lambda_Es_i\ell_{E,i}],
\qquad L_A=\operatorname{mean}_i[s_i\ell_{H,i}].
\]

Teacher A/U는 frozen. Student U는 fresh, A는 local Teacher의 독립 clone/trainable이다. U에는 hard+soft+q-edge, A에는 q-hard만 전달한다. 두 gradient를 구한 뒤 step한다. soft/edge를 A에 직접 보내거나 전체 parameter에 backward(L_U+L_A)를 하지 않는다. Edge는 HRMS와 GT의 signed Scharr, 기존 방향별0.5·경계1px 제외·band mean이다. Hard의 border mask는 바꾸지 않는다.

### 3.4 공통 optimizer

| 항목 | Teacher | Student |
|---|---|---|
| Batch / precision | 48 / FP32, AMP OFF | 동일 |
| Optimizer | AdamW β=(.9,.999), eps=1e-8, wd=.01 | 동일 |
| U peak LR | 1e-4 | 1e-4 |
| A peak LR | 1e-5 | BASE3e-6 |
| Warmup | 100 updates | 동일 |
| 초기화 | fresh A/U, A 마지막 head만0 | fresh U + local Teacher A clone |
| λcon | 1e-4, odd update | 직접 consistency 추가 없음 |

TF32/cuDNN benchmark/driver/CUDA/worker/RNG flags는 배포된 공통 수치 설정 파일로 명시하고 각자 local readback한다. 이를 위해 다른 서버의 성능이나 Teacher 완료를 기다리지 않는다. 기존 receipt와 다르면 변경점을 기록한다. 같은5090만으로 bitwise 동일성을 주장하지 않는다. 새 dropout/gradient clip/augment/optimizer 변경은 없다.

## 4. Fresh50K / Fresh100K scheduler

N∈{50,000,100,000}, t=다음 update 직전 completed updates로 정의한다.
\[
f_N(t)=\begin{cases}t/100,&0\le t<100,\\
\tfrac12[1+\cos(\pi(t-100)/(N-100))],&100\le t\le N.
\end{cases}
\]

\[
\eta_U=10^{-4}f_N(t),\quad \eta_A^T=10^{-5}f_N(t),\quad
\eta_A^S=3\times10^{-6}f_N(t)m(t).
\]

Core BASE/H010/E1은 m(t)=1이다. t=0/100/N의 indexing을 기존 scheduler와 일치시킨다. T100의 중간50K를 freshT50 reference로 쓰지 않는다. S100의 중간50K도 freshS50가 아니다. 기존50K 종료 checkpoint를 무단 재가열하거나 LR0인 update만 붙이지 않는다.

긴 schedule은 update 수·LR 경로·표본 노출·누적 AdamW decay가 함께 늘어난 처치다. 순수 iteration 수 하나만의 효과라고 표현하지 않는다. 같은 local pair의 공통 초기 U와 data/view/RNG prefix는 run/profile/N 문자열 때문에 달라지면 안 된다.

## 5. Local reference registry와 전환

| Reference ID | Teacher case | 서버 | Seed | Endpoint | 소비 서버 |
|---|---|---|---:|---:|---|
| L100I1_R100_S3_TS94001 | L100I1-T01 | s3 | 94001 | 100K | **s3만** |
| L100I1_R50_S3_TS94001 | L100I1-T02 | s3 | 94001 | 50K | **s3만** |
| L100I1_R100_S4_TS94002 | L100I1-T03 | s4 | 94002 | 100K | **s4만** |
| L100I1_R100_S5_TS94003 | L100I1-T04 | s5 | 94003 | 100K | **s5만** |

**불변식: `Student.server == Teacher.producer_server == reference.owner_server`.** 다른 서버의 A/U·τ/q/cache를 local 이름으로 재명명하거나 fallback으로 소비하지 않는다.

s3 T50/T100만 Teacher seed94001·초기 A/U·공통 first50K data/corruption prefix를 맞춘다. s4/s5 Teacher는 독립 fresh seed다. Teacher endpoint는 사전 지정한 최종N의 A/U 전체이고 bestHQNR reference를 선별하지 않는다.

### 5.1 각자 끝낸 뒤 각자 calibration

Calibration 작업은 `CAL-S3-100`, `CAL-S3-50`, `CAL-S4-100`, `CAL-S5-100`의4개다. 정확한 ID는 registry에 `L100I1-` prefix로 고정했다.

같은 GF2 train3072 base ID(seed1234의 원 선택 규약)를 사용하되, **각 endpoint의 실제 출력으로**
\[
\tau_R=\max\{\operatorname{median}_{i,p}\operatorname{mean}_b|Z_T-Y|,10^{-6}\}
\]
를 계산한다. frozen/eval/no_grad/FP32, unaugmented full-pixel을 유지한다.

q는 AXIS16, [.25,.5,1,2]의4방향 probe 및 실제 fixed-HV/ROT4 view로
\[
q_{i,r}=\frac1{2K}\sum_{j=1}^{K}\|c_{i,r,j}+\epsilon_j-c_{i,r,0}\|_1,\quad K=16
\]
를 계산한다. qref는 calibration3072×4 view의 median이고 전체train median으로 대체하지 않는다. 전체train×4 q-cache를 만든다. N19809이면79,236 view이며 실제 N/ID/순서로 검증한다.

미측정 τ/q/cache SHA는 null이다. A/U endpoint·data/LP·calibration base IDs·view mapping·source·τ/q/cache를 하나의 manifest로 pin한다. 같은 서버라도 T50와T100 자산을 섞지 않는다. 자기 reference 오류/NaN/잘못된 data는 정지 사유지만 낮은 유효 성능은 그렇지 않다.

### 5.2 빠른 Student 전환의 조건

필수는 자기 Teacher의 exact endpoint 보존·재로딩 일치, architecture/data 검증, finite 출력, 새 τ/q/full-cache 및 online q16×4 검증이다. 이 조건을 통과하면 **다른 서버의 calibration·성능·요약표를 기다리지 않고** Student로 전환한다.

Teacher의 모든50개 후보 평가를 Student 전환의 성능 gate로 묶지 않는다. Endpoint와 필요한 기술 검증을 먼저 수행하고, 나머지 Teacher 후보 평가는 평가부채에 명시해 CPU/로컬 유휴시간에 완료할 수 있다. 미평가이면 `OFFICIAL_EVAL_PENDING`을 유지하며 완료로 업로드하지 않는다. 이 순서 변경으로 전체 평가비용을 예산에서 빼지는 않는다.

### 5.3 기존 실행이 이미 있는 경우

이 revision에서 actual runtime은 미확인이다. 이전 L100 미시작이면 이 v2 queue를 사용한다. 이전 L100가 이미 시작됐다면 actual t0/deadline/소비시간을 그대로 승계하고, 미시작 부분만 바꾼다. 새 revision을 핑계로20h를 재시작하지 않는다.

s3의 이미 생성된 정확한 동일 seed/N fresh Teacher는 init/data/source/objective/schedule/endpoint/calibration의 동등성을 local receipt로 확인한 경우에만 중복 학습을 피할 수 있다. 원 run ID·원 campaign·reuse receipt를 그대로 남긴다. **s4/s5의 타 서버 Teacher 재사용은 이번 정책과 다르므로 허용하지 않는다.**

이미 shared Teacher로 시작한 Student의 reference를 중간 교체하지 않는다. 사용자가 명시해 full-state 안전 중단하거나 기존 run으로 완료시켜 이력을 분리한다. 진행 중 optimizer를 강제 kill하지 않는다.

## 6. Profile와 실제 기본 실행 범위

| Profile | α | β | λE | A schedule | 이번 상태 |
|---|---:|---:|---:|---|---|
| BASE | 1 | .1 | .002 | 기본 cosine | core 로컬 대조 |
| H010 | **.1** | .1 | .002 | BASE | s4 core |
| E1 | 1 | .1 | **.001** | BASE | s5 core |
| ARW | 1 | .1 | .002 | 초기0.1배→1배 | **DEFERRED_USER_RELEASE** |
| E02 | 1 | .1 | **.0002** | BASE | **DEFERRED_USER_RELEASE** |

H010는 기본 GT loss를 제거하지 않고 weight 범위를 [1,2)→[1,1.1)로 줄인다. 최신 요약의 약한 재가중을 감안하면 “α가 GF2 실패의 주원인”이라는 전제를 두지 않는다. U/A의 두 hard 항 모두 α가 바뀌므로 전체 hard 강조의 효과다.

E1은 과거4개 local pair의 작은 공간 이득과 RR 비용을 새 long/local reference에서 다시 시험한다. 아주 작은 단일 시점 edge gradient는 장기 효과가 없다는 증명은 아니지만, E1의 수치 guard 통과만으로 더 큰 감소 E02를 자동 개방하지 않는다.

ARW/E02의 계수 정의는 향후 재지시용 registry에만 보존한다. ARW m(t)는 t<5K에서.1, 5K≤t<10K에서.1+.9(t−5K)/5K, 이후1이다. 기존 진단의 초기 c 증폭이나 norm 증가 자체를 harmful drift로 판정해 이를 자동 실행하지 않는다. `α=0`, PAN auxiliary task, A freeze, correction gain, q temperature, GT border 변경은 추가하지 않는다.

## 7. 서버별 정확한 기본 queue

Student seed95001/95002는 이전 계획과 동일하다. 각 서버에서 동일 SS의 BASE/ALT는 동일한 local Teacher와 초기 U·data/view stream을 사용한다. 다른 Teacher를 쓰는 Student의 초기 A가 다른 것은 의도한 조건이다.

### 7.1 s3 — 두 local Teacher를 먼저 완료한 뒤 길이 대조

| Case | 역할 | 로컬 reference | Updates | Seed | Profile | 순서 |
|---|---|---|---:|---:|---|---:|
| L100I1-T01 | T | s3/R100/TS94001 | 100,000 | 94001 | C100 | 10 |
| L100I1-T02 | T | s3/R50/TS94001 | 50,000 | 94001 | C100 | 20 |
| L100I1-S01 | S | s3/R50/TS94001 | 50,000 | 95001 | BASE | 30 |
| L100I1-S02 | S | s3/R50/TS94001 | 100,000 | 95001 | BASE | 31 |
| L100I1-S03 | S | s3/R100/TS94001 | 100,000 | 95001 | BASE | 32 |
| L100I1-S04 | S | s3/R100/TS94001 | 100,000 | 95002 | BASE | 40 |
| L100I1-S05 | S | s3/R50/TS94001 | 100,000 | 95002 | BASE | 41 |
| L100I1-S06 | S | s3/R50/TS94001 | 50,000 | 95002 | BASE | 42 |

실제 순서: **T01→CAL-S3-100→T02→CAL-S3-50→S01→S02→S03→S04→S05→S06**. 이 모든 의존성은 s3 안에만 있다. s4/s5는 별도로 Teacher를 시작하므로 s3의 T50 및 길이 대조가 그 서버의 출발을 지연시키지 않는다.

| 비교 이름 | 조건 | 역할 |
|---|---|---|
| C0 | local T50 / fresh S50 / BASE | 이번 환경의 짧은 학습 대조 |
| C1 | 같은 local T50 / fresh S100 / BASE | Student 학습 길이 확대 |
| C2 | local T100 / fresh S100 / BASE | Student100K에서 Teacher 길이 확대 |

- SS95001: S02−S01 = Student 길이, S03−S02 = Teacher 길이.
- SS95002: S05−S06 = Student 길이, S04−S05 = Teacher 길이.
- S03−S01/S04−S06은 전체 long-chain 효과다. T100/S50가 없어 완전2×2 interaction은 측정하지 않는다.
- 위 S번호는 모두 `L100I1-` prefix다. 두 번째 seed 길이3조건은 하나의 atomic block이다.

### 7.2 s4 — 자기 T100 후 hard 강조 대조

| Case | 역할 | 로컬 reference | Updates | Seed | Profile | 순서 |
|---|---|---|---:|---:|---|---:|
| L100I1-T03 | T | s4/R100/TS94002 | 100,000 | 94002 | C100 | 10 |
| L100I1-S07 | S | s4/R100/TS94002 | 100,000 | 95001 | BASE | 30 |
| L100I1-S08 | S | s4/R100/TS94002 | 100,000 | 95001 | H010 | 31 |
| L100I1-S09 | S | s4/R100/TS94002 | 100,000 | 95002 | H010 | 32 |
| L100I1-S10 | S | s4/R100/TS94002 | 100,000 | 95002 | BASE | 33 |

**T03→CAL-S4-100→S07(BASE95001)→S08(H01095001)→S09(H01095002)→S10(BASE95002)**. TS94002의 local R100만 소비한다. α 외의값은 동일하다. S08−S07, S09−S10이 local paired 효과다.

### 7.3 s5 — 자기 T100 후 edge 대조

| Case | 역할 | 로컬 reference | Updates | Seed | Profile | 순서 |
|---|---|---|---:|---:|---|---:|
| L100I1-T04 | T | s5/R100/TS94003 | 100,000 | 94003 | C100 | 10 |
| L100I1-S11 | S | s5/R100/TS94003 | 100,000 | 95001 | BASE | 30 |
| L100I1-S12 | S | s5/R100/TS94003 | 100,000 | 95001 | E1 | 31 |
| L100I1-S13 | S | s5/R100/TS94003 | 100,000 | 95002 | E1 | 32 |
| L100I1-S14 | S | s5/R100/TS94003 | 100,000 | 95002 | BASE | 33 |

**T04→CAL-S5-100→S11(BASE95001)→S12(E195001)→S13(E195002)→S14(BASE95002)**. TS94003의 local R100만 소비한다. λE 외의값은 동일하다. S12−S11, S13−S14가 local paired 효과다.

### 7.4 기본 queue에서 빠지는 기존 추가 case

| Case | 서버 / SS | Profile | 참조 | 상태 |
|---|---|---|---|---|
| L100I1-X01 | s4 / 95001 | ARW | s4 local T100 | 사용자 재지시 전 미기동 |
| L100I1-X02 | s4 / 95002 | ARW | s4 local T100 | 사용자 재지시 전 미기동 |
| L100I1-X03 | s5 / 95001 | E02 | s5 local T100 | 사용자 재지시 전 미기동 |
| L100I1-X04 | s5 / 95002 | E02 | s5 local T100 | 사용자 재지시 전 미기동 |

Core가 끝나면 자동으로 새 seed·새 Teacher·추가 coefficient를 만들지 않는다. 결과를 보존·보고하고 사용자의 다음 조정을 기다린다. 사용자 재지시가 있더라도 기본20h deadline을 자동 연장하지 않는다.

## 8. 시간·admission·안전 전환

### 8.1 공통20h, 로컬 독립 admission

신규 창의 t0는 실제 준비/전환 시작 시점이다. 이미 시작된 L100가 있다면 그 actual clock을 승계한다. 한 서버가 늦게 준비돼도 별도20h를 다시 부여하지 않는다. 기존 G20는 별도 campaign이고 그 완료·미완 이력을 보존한다. s1/s2의 기존 실행은 변경하지 않는다.

| 작업 | 초기 예약(h) | 비고 |
|---|---:|---|
| Setup/local P0 | .75 | 같은 checkpoint/환경 확인 등 local 기본 검증 포함 |
| Teacher100K+평가 | 1.70 | 원안의 계획 가정, 이번 실측값 아님 |
| Teacher50K+평가 | 1.00 | s3 control만 |
| Local calibration각각 | 1.00 | τ+전체q-cache+local integrity |
| Student100K+평가·P1 | 2.40 | epoch/time만이 아니라 subprocess 전체 |
| Student50K+평가·P1 | 1.30 | s3 controls |
| 종결 보존 | 2.00 | 18–20h 확보 |

**Teacher 전송/import .25h와 s3 Teacher 대기시간은 제거했다.** 대신 s4·s5의 local Teacher100K와 calibration 비용을 각각 포함한다.

| 서버 | Core 구성 | 계획상 primary 완료(h) | 종결 포함(h) |
|---|---|---:|---:|
| s3 | T100+T50+CAL2+S100×4+S50×2 | **17.65** | **19.65** |
| s4 | T100+CAL1+S100×4 | **13.05** | **15.05** |
| s5 | T100+CAL1+S100×4 | **13.05** | **15.05** |

이 표는 초기 예약의 산술합이고 실제 walltime 완료 보장이 아니다. 특히 s3 여유가 작다. 세 서버60 slot-hour는 명목 자원 상한이지 실제 학습시간 합계가 아니다. ARW/E02는 이 core 예산에 넣지 않았으며, 시간 여유가 있어도 자동 입장하지 않는다.

### 8.2 입장 기준

새 block B는 현재시간<t0+16h이며, `현재+block train/cal/eval 잔여 예약+누적 평가부채<t0+18h`이고, 보존까지20h 이내에 들어올 때만 입장한다. 이미16h 전에 전체 예약한 block의 후속 case는18h 안에 끝낼 수 있을 때 시작할 수 있다. 미입장 case를 기존 block에 붙여 cutoff를 피하지 않는다.

Teacher 입장 전 각자 필요한 첫 Student block까지의 자원 여유를 확인한다. s3는 T2+CAL2+첫 길이3조건, s4/s5는 localT100+CAL+첫BASE/ALT pair를 예약한다. 이것은 다른 서버를 기다리는 gate가 아니라 자기 chain이 Student까지 도달하도록 하는 시간 확인이다.

실측은 train subprocess 전체 walltime, validation/eval, 저장, calibration, 재시도, 기존 spent time을 합산한다. 최근 대표 실측과 warmup 외삽 중 보수적인 쪽에1.15 여유를 적용하고, 초기 예약을 줄이려면 근거를 남긴다. Local P0 오류는 해당서버만 차단하며 유효한 다른 서버 core를 막지 않는다.

### 8.3 시간이 모자라면

우선 ARW/E02는 계속미기동이다. s3의 두 번째 seed 길이3조건(S04–S06)을 먼저 제외한다. 더 부족하면 해당 서버의 아직 미입장 두 번째 pair(s4 S09/S10, s5 S13/S14)를 전체로 제외한다. 한쪽만 완료했으면 `PAIR_INCOMPLETE`로 남기며 다른서버/seed 값을 대조로 가져오지 않는다.

축소 core는 s3 T2+S3, s4 T1+S2, s5 T1+S2의 **11개 학습**이다. 이조차 시간·자원이 없으면 미완을 기록한다. batch·precision·metric·update목표를 몰래 줄이지 않는다. 기존 실행을 강제 kill하지 않으며 사용자 재지시는 completed optimizer boundary에서 full-state를 남긴 뒤 새 recipe revision으로 적용한다.

## 9. 평가와 보고

각 fresh50K/100K run은 50개의 official candidate를 갖는다. G50은 `1010, 2020, …, 49490, 50000`이다. G100은 `2×G50`에서 50500을 50000으로 바꾼 50개 step이며 마지막 100000을 포함한다. 정확한 배열은 registry에 있다. 추가 1K/5K/10K 진단 checkpoint를 RAW/TARGET 후보에 더하지 않는다.

| 선택점 | 의미 |
|---|---|
| EXACT_FINAL | 해당 run의 정확한 N=50K 또는 100K, 주 분석 |
| RR_VAL_SELECTED | 사전 50개 candidate의 validation ERGAS 최소, 동률이면 낮은 step |
| MID50_OF100 | 100K horizon의 중간 50K이며, 독립 fresh50K와 다름 |
| RAW_MAX / TARGET / E_MIN_DIAG | test-aware 개발 진단으로 주 분석과 분리 |

GF2 C4/maxDN1023, 정규화 `2DN/1023−1`, RR20/support20:-21/Q4 block32, FR20/full512/native PAN·original LMS·기존 GF2 MTF를 유지한다. signed Ds의 Qhigh/Qlow 20×4를 보존하고 공식 값을 재구성한다. HQNR는 장면별 `(1−Dλ)(1−Ds)`의 평균이다. 보정 PAN을 공식 evaluation reference로 쓰지 않는다. JQM은 기존 SRF-substitute 보조값이다.

Teacher의 exact endpoint/VAL을 먼저 평가해 빠르게 보고할 수 있지만, 나머지 평가 미완을 숨기지 않는다. 100K 지표를 옛 `Exact50K` 열에 쓰지 않는다. 실제 run/step/Teacher SHA/profile와 paired baseline을 함께 기록한다. 하나의 RAW H와 다른 checkpoint의 E를 합치지 않는다.

목표는 동일한 정상 A_ON checkpoint에서 **H > .964 및 E < .552**다. E < .522는 별도 strong flag다. 정확한 최종값, VAL, test-aware TARGET을 분리한다. Correction norm이나 q는 native displacement GT가 아니다.

## 10. 이번 설계에서 가능한 비교와 불가능한 비교

**가능한 비교:** s3 안에서 Student 길이 효과(C1−C0), s3 안에서 Teacher 길이 효과(C2−C1), s4의 local H010−BASE, s5의 local E1−BASE다. 각각 두 Student seed에서 효과 방향과 E/H trade-off를 본다.

**분리할 수 없는 비교:** s4가 s5보다 HQNR가 높다는 이유로 α 감소가 edge 감소보다 우월하다고 단정할 수 없다. 서로 다른 Teacher의 Student를 동일 reference의 seed 평균으로 합칠 수도 없다. s3의 Teacher 길이 효과를 s4/s5에 자동으로 일반화하지 않는다. Teacher에 따른 예측·초기 A·τ/q/cache 차이는 이번 독립 실행에서 의도적으로 남겨 둔 조건이다.

기존 운영 기준은 결과 분석용으로 유지한다. Core를 시작하기 전에 다른 서버의 성능 판정을 기다리게 만드는 gate로 사용하지 않는다.

| 분석 분류 | 수치 조건 |
|---|---|
| **H_PROMISING** | 두 seed 모두 ΔH > 1e−5, ΔH 중앙값 ≥ .0015. 상대 E 악화 중앙값 ≤ .5% 및 개별 ≤ 1%. ΔDλ 중앙값 ≤ .001 및 개별 ≤ .002. VAL에서도 E guard와 ΔH 중앙값 ≥ −.0015. |
| **RR_GAIN_WITH_H_PRESERVED** | 두 seed 모두 E 개선, 상대 개선 중앙값 ≥ .5%, 각 ΔH ≥ −.001. VAL에서도 같은 방향. |
| **JOINT_TWO_SEED** | 두 seed 각각의 VAL이 동일 checkpoint에서 H > .964 및 E < .552를 충족. |

위 수치는 운영용 분류이지 통계적 유의성이나 전역 최적성의 인증이 아니다. 모든 결과를 원값으로 보고하고, recipe 승격과 추가 튜닝은 사용자의 조정 후 별도 revision으로 남긴다.

## 11. 최소 진단 — core 학습을 막는 대규모 원인 감사로 확대하지 않는다

공통 train128 ID와 그중 gradient24 ID를 사전에 고정하고 calibration ID와의 관계를 기록한다. Teacher는 init/10K/25K/50K/75K/100K, Student는 init/1K/5K/10K/25K/50K/75K/100K 중 해당 N 이내의 시점에서 기록한다. 진단 RNG와 training stream을 격리하고 상태를 복원한다.

| 영역 | 기록 | 새 진단이 요구하는 해석 |
|---|---|---|
| Clone/정합 | 초기 cT=cS, hash, cS−cT, dy/dx·norm·cos·p95 | 같은 방향의 증폭을 곧바로 실패로 표시하지 않음 |
| 복원 학습 | eS/eT, GT L1/ERGAS, 우위 patch 비율, 밴드·edge·밝기별 오차 | train/RR/FR의 향상이 같은 현상인지 분리 |
| 경계 | 1px ring과 interior의 오차, α 추가항 질량/면적 | crop64의 경계가 FR512 전체를 설명한다고 단정하지 않음 |
| 가중치 | dT/s 분포, advantage 활성, hard/soft/edge 원값·weighted 값 | s≈.5이면 강한 선별이라고 서술하지 않음 |
| 최적화 | weighted gradient norm·cosine, 실제 U/A parameter update norm | 작은 단일 시점 gradient를 장기 효과 부재로 일반화하지 않음 |
| FR 관계 | signed Qhigh−Qlow 20×4 및 Ds 재구성 | 양수 편향만으로 시각적 과선명화나 평가기 오류를 확정하지 않음 |
| 노출량 | 실제 sample/view count, unique IDs, prefix hash | 100K의 실제 추가 학습 확인 |

원 추정기나 미업로드 진단 JSON이 없어도 `P1_UNAVAILABLE`로 남기고 자기 Teacher/Student core를 진행한다. Clone/reference 경로, 데이터 shape, gradient routing, NaN 등 실제 P0 오류는 별개로 차단한다. 추가 GPU 진단은 예약 한도 내에서만 수행하고, 같은 pair에 같은 정책을 적용한다. 경계 mask·학습 정답·정규화 변경은 사용자의 별도 승인 전에는 추가하지 않는다.

## 12. 배포자가 적용할 변경과 회귀검사

이 자료는 설계이며 현재 원격 runner를 수정한 것은 아니다. 이전 문서가 참조한 G20의 50K/profile 검사를 무시하고 `num_iter`만 바꾸지 않는다. 실제 checkout에서 새 L100I1 registry·scheduler·reference·postrun/upload schema의 지원 여부를 확인한다. 기존 G20/FH12 수치 코드와 완료 결과는 소급 변경하지 않는다.

| 검사 | 통과 조건 |
|---|---|
| Local Teacher graph | 모든 Student의 Teacher owner가 자기 서버이며 cross-server edge는 0개 |
| 기본 queue | s3는 T2+CAL2 후 S6, s4/s5는 각각 T1+CAL1 후 S4. ARW/E02 자동 입장 없음 |
| 정확한 N | T100/S100 endpoint는 100000. 중간 50K와 독립 50K 구분. 각 50 candidates |
| 초기화 | s3 T50/T100 초기 A/U 동일. Local BASE/ALT의 U 동일, A clone 동일 |
| Calibration 무결성 | Local endpoint A/U와 τ/q/cache/ID/hash 일치. Online/cache q 검사 |
| Method | C4/5ch/7ch, P/L 동기 warp, 기존 분리 routing 유지. H010에서 α만 변경 |
| Runtime | Local GPU/flags/driver/precision 기록. 공통 설정 파일의 hash readback |
| Clock | 기존 active L100의 t0/deadline 승계 또는 신규 기동 시 한 번 생성. 16/18/20h 경계 검사 |
| 이력 | Shared reference로 시작한 Student의 hot-swap 금지. 재사용 Teacher의 원 ID·reuse receipt 유지 |
| Upload | s3–s5의 새 L100I1 행에 local Teacher seed/owner/SHA 및 EXACT_FINAL을 정확히 기록 |

Teacher50K/100K·Student50K/100K 각각의 경계 LR와 resume 연속성을 시험한다. 실행 확인 증거는 각 서버의 `teacher_started`, `teacher_complete`, `reference_ready`, `student_started` receipt와 실제 timestamp다. 설계 검사 PASS를 실제 GPU 학습이나 원격 배포 PASS로 보고하지 않는다.

## 13. 파일·출처·수정 이력

**원 v1:** `PANDA_GF2_L100_S345_20H_20260921_v1`의 미시작 shared-Teacher 분기만 대체한다. 같은 suffix/ID로 과거 자산을 덮어쓰지 않도록 case prefix는 `L100I1-`, run prefix는 `L100I1_GF2_`, suffix는 `_v2`다.

| 파일 | 용도 |
|---|---|
| 본 MD | 수정된 전체 운영 계약 |
| `PANDA_GF2_L100_S345_LOCALT_20H_Cases_2026-09-21_v2.csv` | 기본 18개와 보류 4개의 정확한 case/run/reference/profile |
| `PANDA_GF2_L100_S345_LOCALT_20H_DesignRegistry_2026-09-21_v2.json` | Local reference 4개, dependency graph, grid, 예산의 설계 registry |
| `queues/{s3,s4,s5}_local_teacher_first.json` 및 `.txt` | **기본 18개만** 포함한 서버별 Teacher 우선 순서. Runner 입력이 아닌 설계 queue |
| `validate_design.py` / `DESIGN_VERIFICATION.json` | 로컬 의존성·수량·profile·순서 검증. GPU 실행 검증은 아님 |
| `sources/User_Diagnostic_Update_2026-09-21.md` | 출처와 한계를 명시한 사용자 추가 진단 요약 |
| `sources/`의 원안 MD·CSV 및 기존 근거 문서 | 변경 전 기록 보존 |

[S1] 원안 `PANDA_GF2_L100_S345_20H_ExperimentPlan_2026-09-21.md` 및 case CSV. Source Sheet snapshot SHA `1d87f92ee7d6de241a60dffcebd9ad7b217800936601f25538ca3ffacaa80690`는 원안 작성 시점의 기록이다. 이번 수정의 live 조회값이 아니며, 원본 XLSX는 이전 v1 번들에 보존돼 있다.

[S2] `PAN_WV3_Closeout_Review_FrozenRecipe_2026-09-20.md`, §7–8. 최종 architecture와 학습 규약.

[S3] 이번 대화에서 사용자가 추가한 Q1/Q2/Q3, 정정 사항, 열린 질문. 원 JSON/estimator를 재실행하지 않고 사용자 제공 진단으로 구분해 반영했다.

[S4] `2026-09-20_gf2-hqnr-gap-audit.md`, §1/§4/§7. 데이터 개수와 과거 DUAL 참고. 이 문서의 “현 mainline”은 옛 B01 시점이지 현재 PANDA가 아니다. PAN 재구성 단독 효과는 그 감사에서도 미분리다.

[S5] `GF2_PAN_Shift_Ds_Historical_s3_Review_2026-09-21.md`, §4–6. E1의 기존 4-pair와 긴 schedule의 근거. 이번 local T100 성능을 보장하지 않는다.

**작성 시 actual t0, future Teacher checkpoint SHA, τ/q/cache 값은 모두 미확정이다.** 이 수정에서는 문서와 등록 자료를 만들었고, 원격 학습을 기동하지 않았다.
