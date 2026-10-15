# PAN 통합 실험 — s5 배정안
## Aligner 학습 시점·loss별 gradient routing / raw HQNR 0.959–0.960

**작성일:** 2026-09-14  
**상위 캠페인:** `PAKD50_W112D123_WV3_20260914_v1`  
**배정 ID:** `S5_TIMING_ROUTING_v1`  
**검토한 Git snapshot:** `2303e56acc44abeaf84695c1df6e7d83fb78932f`  
**문서 상태:** 실험 설계·s1 구현 인계. 학습 기동, 서버 접속, Git push, Sheet 편집을 수행한 기록이 아니다.  
**주력 protocol:** `FRESH50` = 독립 초기화 U-Net의 **50,000 update**. 50시간이라는 뜻이 아니다.

> **s5의 역할:** s4의 aligner LR·Q12 scalar 탐색을 반복하지 않고, Teacher에서 학습한 aligner를 **언제 업데이트하며 어떤 loss의 gradient를 받을지**를 최적화한다. 기본은 **J0/JQ → D0/DQ → PQ → winner 확인**이다. LF·부분 routing·D+P 결합은 근거가 생겼을 때 최대 2개 run만 추가한다.

## 0. 실행 요약

| 항목 | s5 결정 / 제안 |
|---|---|
| 목표 | **raw-original HQNR ≥0.959**, 확장 목표 ≥0.960. V64는 진단 |
| 탐색 seed | **2026**: s3와 같은 seed로 서버 교차 및 s5 내부 대응 비교 |
| 확인 seed | **9091**: 이번 배정에서 사전 지정. winner lock 전에는 실행하지 않음 |
| 고정 Teacher | T0의 같은 best_hqnr A/U, frozen, 자체 정합 경로 |
| Student 초기화 | T0 최종 aligner 복사 + seed별 저장 fresh U-Net |
| 공통 구조 | W112 · depth=[1,2,3] · 9ch → residual8 + clean MS base 한 번 |
| 기본 5개 | J0, JQ, D0, DQ, PQ |
| 기본 확인까지 | 신규 후보 WIN이면 대응 control/anchor/WIN 최대 3개 추가: **총8run** |
| 확장 상한 | 원인 확인 또는 결합 최대2run: **총10run**; 전수 탐색 아님 |
| 구현 위치 | s1에서 C1 timing/routing 구현·검증 → immutable release → s5 |
| 업로드 | 기존 gspread 경로 → `pan-cvpr27 / WV3-s5`, X열 `통합실험` |
| 마감 | 상위 공통 시계 그대로. s5 투입으로 50시간을 다시 시작하지 않음 |

Core 5는 연구상 우선순위다. 미구현 capability·미배포 λE·예산 부족 상태에서 강제로 실행하라는 뜻은 아니다. 짧은 예산에서는 먼저 J0/JQ와 완전한 D0/DQ 대응쌍을 확보하고, 미완료 대비 후보만 늘리지 않는다.

## 1. s3의 새 결과를 어떻게 반영했는가

### 1.1 확인된 것은 J0의 업로드 결과 1건

`WV3-s3(5090)`의 B3:X200을 `PAKD50`으로 검색했을 때 다음 run 1개가 확인됐다. 숫자는 B81:W81의 plain values다. 이후 추가 업로드가 있을 수 있으므로 실행 전 전체 run ID로 재검색한다. [S3]

```text
PAKD50_J0_W112_D123_WV3_T0_S2026_FRESH50_v1
```

| Raw HQNR | V64 | Dλ | Ds | ERGAS | SAM | PSNR | Train(h), 저장값 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.9553** | 0.9607 | 0.0196 | 0.0256 | 2.0860 | 2.8085 | 37.8095 | 1.16 |

**해석:** Teacher-final aligner를 복사한 뒤 native GT reconstruction과 offset으로 학습한 **J0 / N0 대조군**이다. Teacher는 평가 bin에만 쓰이며, Q12 output-soft·GT edge와 R1 재가중을 학습 loss로 사용하지 않는다. 따라서 “KD 통합으로 0.9553을 얻었다” 또는 “Q12가 실패했다”는 해석은 하지 않는다.

Raw 0.959까지 약 +0.0037, 0.960까지 약 +0.0047이 남는다. 이는 표시 정밀도의 산술 gap이며 달성 가능한 향상량의 예측은 아니다. V64 0.9607은 raw 목표 달성으로 세지 않는다.

### 1.2 아직 모르는 것

행에 `50K`가 적혀 있어도 selected update는 미확인이다. **exact50K·plateau·FR fSCC·shift drift·loss별 gradient**는 이번 Sheet 행에서 확보되지 않았다. s3의 실제 프로세스와 raw checkpoint도 조회하지 않았다. 단일 J0로 초기 정합 손상, 후반 drift, Q12 gradient 충돌 중 하나를 원인으로 확정하지 않는다.

s5 시작 전에 s3 실행자가 전달할 최소 결과 묶음은 `resolved config + 실행 commit + init/Teacher hash + best_hqnr metadata + checkpoint_metrics + exact50K export + 고정 진단`이다. 없는 산출물을 임의로 완성하지 않고 null 상태로 넘긴다. s5의 자체 대조 실험은 s3의 모든 진단이 준비될 때까지 막을 필요는 없다.

### 1.3 s5의 질문을 좁힌 이유

s3는 fresh Student 아래 no-KD 기준값을 제공한다. s4는 이미 LR와 loss 강도를 다룬다. 따라서 s5에서는 **계수는 고정한 채 학습의 시점·수신 경로만 조정**하는 것이 새 정보를 준다. D/P/LF는 새로 추가하는 대형 모델이 아니라 상위 50시간 계획의 §7.2–7.6 후보를 배정한 것이다. 상위 계획에서 조건부였던 D/P를 s5의 기본 탐색으로 올린 것은 이번 배정의 신규 결정이며, 효용이 이미 입증됐다는 뜻은 아니다. [P, R4]

## 2. 서버 간 역할 분담

| 서버 | 유지할 역할 | s5와의 연결 |
|---|---|---|
| s1 | 코드·공통 T0/τR/λE package·seed1234 core | timing/routing C1 구현·회귀 검사·release 주관 |
| s2 | seed777 core J/F/Q12/R1 | 유력 s5 WIN의 후속 별도 seed 반복에 활용 가능 |
| s3 | seed2026 core; J0 업로드 확인 | s5 seed2026과 동일-seed 환경 교차; 이후 JQ 대응 확인 |
| s4 | seed1234의 J/AL·scalar, winner 후3407 | LR·계수 탐색 담당. s5에서 다시 넓게 돌리지 않음 |
| **s5** | **D/P/LF: 업데이트 시점·gradient 수신 경로** | seed2026 탐색, winner 후9091 확인 |

이 표는 배정이다. 각 서버의 현재 GPU process와 모든 run 완료를 확인했다는 뜻은 아니다. `s3=seed2026, s5=seed2026` 두 결과는 **독립 seed 2개가 아니다**. s5의 신규 방법은 s5의 같은 seed control과 비교한다.

## 3. 바꾸지 않는 공통 계약

### 3.1 모델·좌표·Teacher

- Aligner는 native PAN/MS에서 current HR pixel 단위 `(dy,dx)`를 예측한다. 기존 margin4·crop 후 bandwise z-score·bicubic/border/align_corners=False를 유지한다.
- 원 PAN만 한 번 sampling한다. MS condition·GT·MS residual base·출력 frame은 움직이지 않는다. 추가 jitter PAN은 offset branch의 aligner에만 들어간다.
- Teacher T0의 A/U는 frozen이다. Student는 별도 A/U 객체를 갖는다. Student A가 달라져도 Teacher는 자기 고정 A로 target을 만든다.
- Student U에는 residual에 base를 한 번 더한 최종 HRMS로 Q12/R1을 계산한다. Variance/covariance, feature KD, shift KD, geometry, PAN task를 추가하지 않는다. [A §1–4, K §3–6]

T0의 확인된 checkpoint file SHA256:

```text
16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32
```

출처는 `assets/pakd50/calibration_resolved.json`의 `tau_source`, selected update=24240다. 실제 s5의 T0 파일을 이 hash와 대조하고 A/U를 같은 checkpoint에서 로드한다. [CAL]

### 3.2 학습·평가

| 항목 | 공통값 |
|---|---|
| U / A peak LR | 1e-4 / 1e-5; 동결 구간 A 업데이트 없음 |
| Optimizer | 기존 AdamW, WD0.01, warmup100 + cosine50K |
| 나머지 수치 옵션 | 기존 성공 run의 beta/eps/최소LR/AMP/clip/accumulation 규약 그대로 |
| Batch / updates | 48 / 50,000 |
| Offset | weight1e-4, radius2 면적 uniform disk, 0-based 홀수 update |
| 주 selector | raw-original FR20 HQNR → 기존 running-max tie1e-4 / fSCC 규칙 |
| 공통 candidate grid | **GRID1010_50K_v1**: 1010,2020,…,49490,50000 |
| Plateau | 45450,46460,47470,48480,49490,50000의 동일6개 시점 |
| Raw target | full original PAN reference. V64/align-self는 진단 |

추가로 5K/25K 경계의 state를 저장하더라도 **공식 selector 후보에 포함하지 않는다**. s5만 평가 기회를 늘려 최고값을 올리는 변경을 금지한다. 위 grid는 원안 GRID1K가 아니라 실제 C0 구현에서 채택한 격자다. [R0]

### 3.3 Calibration readiness

조회한 asset의 τR는 **0.012463942170143127**이다. 동일 T0·data·intensity package를 s5에서 재현하고 공통값을 고정한다.

**조회한 commit의 calibration asset에는 λE가 아직 없다.** 이는 로컬 미산출을 의미하지는 않는다. s1의 J0_S1234 exact50K에서 계산하여 배포한 공통 λE0를 받은 뒤에 JQ/DQ/PQ 및 다른 edge case를 실행한다. s3의 J0가 먼저 완료됐다는 이유로 pilot을 바꾸거나 λE=0.05를 넣지 않는다. [CAL, P §5]

D/P/LF의 목적은 routing/schedule 비교이므로 **같은 λE0**를 쓴다. 각각 다시 calibrate해 loss 크기까지 바꾸지 않는다. Teacher 경로를 고정했으므로 Student A 업데이트만으로 τR를 online 재계산하지 않는다.

## 4. 목적함수와 gradient 수신자

### 4.1 공통 정의

최종 HRMS 출력 S,T와 GT Y에 대해 다음을 사용한다. p는 batch와 공간 위치다.

\[
e_T=\operatorname{mean}_c|T-Y|,\quad e_S=\operatorname{mean}_c|S-Y|,\quad k=\operatorname{mean}_c|S-\operatorname{sg}(T)|,
\]
\[
d=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\qquad
 a=\operatorname{sg}\!\left(\operatorname{clip}_{[0,1]}\frac{[e_S-e_T]_+}{e_S+10^{-6}}\right).
\]
\[
L_0=\langle e_S\rangle,\quad L_D=\langle d\,e_S\rangle,\quad
L_K=\langle0.1(1-d)a\,k\rangle,\quad L_{Ew}=\lambda_{E0}L_E.
\]

LE는 GT의 같은 band와 비교한 **signed Scharr/32**, reflect1, 내부1px, x/y 각각 L1 평균의 0.5배 합이다. Variance loss가 아니다. [K §4–5]

\[
L_Q=L_0+L_D+L_K+L_{Ew},\qquad L_R=L_0+L_D.
\]

Offset 항 LO는 A가 trainable인 홀수 update에만 학습에 사용한다.

\[
L_O=10^{-4}\operatorname{mean}_{B,2}\big|A_S(\mathcal W(P,\epsilon),M)+\epsilon-\operatorname{sg}(c_S)\big|.
\]

cS target은 **현재 Student native prediction**이다. Teacher cT로 대체하지 않는다. Teacher/GT와 gate는 detach하되 실제 eS/k와 Student edge는 live다. Reconstruction/edge를 앞단 wrapper와 이 식에 중복 합산하지 않는다.

### 4.2 손실을 분리하는 것은 scalar 계수를 바꾸는 것과 다르다

A가 trainable일 때 routing vector `qA=(qD,qK,qE)`로 다음 gradient를 구성한다.

\[
g_A=\nabla_{\phi_S}(L_0+q_D L_D+q_K L_K+q_E L_{Ew}+L_O),
\]
\[
g_U=\nabla_{\theta_S}L_Q.
\]

| 정책 | qA | A가 직접 받는 항 | U가 직접 받는 항 |
|---|---|---|---|
| JQ | (1,1,1) | L0+LD+LK+LEw+LO | LQ |
| **PQ** | **(0,0,0)** | **L0+LO** | **LQ** |
| JK0 | (1,0,1) | L0+LD+LEw+LO | LQ |
| JE0 | (1,1,0) | L0+LD+LK+LO | LQ |
| J0 | 추가 항 없음 | L0+LO | L0 |

PQ는 A를 동결하는 방법이 아니다. 또한 PQ의 A 궤적이 J0와 완전히 같다는 뜻도 아니다. Q12로 바뀐 U를 거쳐 이후 L0가 A에 전달되므로 **간접 영향은 남는다**. 분리하는 것은 각 step의 직접 gradient 수신 경로다.

## 5. 기본 실험 5개

### S5-B0 / J0 — s3와의 bridge 및 s5의 no-KD control

| 항목 | 설정 |
|---|---|
| Case / seed | J0 / 2026 |
| 초기화 | T0 A 복사, s3와 같은 저장 U 초기 state; hash 대조 |
| 학습 | A: L0+LO, U: L0, 즉시 joint50K |
| 참조 | s3 J0_S2026 업로드 및 실제 동일 자산 |
| 필수 산출 | raw-selected/last/plateau, data/init hash, shift 및 fitting 진단 |

같은 seed 숫자만 맞추지 않고 s3의 초기 U state를 hash로 확인한다. s3/s5가 같은 평가 대상 checkpoint를 추론하는 검사는 환경 재현성 검사이고, 다시50K 학습한 점수 차이는 학습 궤적의 환경 민감성까지 포함한다.

두 서버의 J0 숫자가 다르다는 이유만으로 s5 새 방법을 탈락시키지 않는다. **s5 control 대비 paired 차이**가 이후 판단 기준이다. 구현 회귀 검사로 기존 J0가 보존됨을 확인하고, 소스 revision 차이를 누락하지 않는다.

### S5-B1 / JQ — 기존 주력 통합의 동일 조건 대조

| 항목 | 설정 |
|---|---|
| Case / seed | JQ / 2026 |
| 학습 | A: LQ+LO, U: LQ, 즉시 joint50K |
| 변경 | J0 대비 Q12 backend만 |
| 시작 조건 | 공통 λE0 package, Q12 graph tests 통과 |
| 주 비교 | JQ−J0; s3에서 JQ가 완료되면 같은 seed의 Δ도 병기 |

s3의 J0를 s5의 JQ에서 빼서 KD 순증분이라고 하지 않는다. 먼저 s5 내부 JQ−J0를 얻고, s3 내부 JQ−J0가 있으면 **두 대응 효과를 교차 비교**한다. 같은 seed 두 서버를 평균할 때는 환경 반복임을 표시한다.

### S5-T0 / D0 — 초기 5K A 동결, GT-only

| 구간, 0-based update t | A | U |
|---|---|---|
| 0 ≤ t < 5000 | T0 복사 상태 frozen, 학습 LO 없음 | L0 |
| 5000 ≤ t < 50000 | L0+LO, native sampler graph 유지 | L0 |

**가설:** 학습되지 않은 U의 초기 reconstruction gradient로부터 Teacher-final A를 잠시 보호하는 것이 유리할 수 있다. 아직 s3에서 입증한 원인은 아니다.

D0−J0는 **KD 없는 warm-start 일정의 효과**다. D0가 좋아져도 distillation 이득이라고 하지 않는다. 5K는 기존 상위 계획의 값이며 1K/3K/5K/10K 전수 탐색으로 넓히지 않는다.

### S5-T1 / DQ — 초기 5K A 동결, U는 처음부터 Q12

| 구간, t | A | U |
|---|---|---|
| 0 ≤ t < 5000 | frozen, 학습 LO 없음 | **LQ** |
| 5000 ≤ t < 50000 | **LQ+LO** | LQ |

DQ는 “처음 5K KD를 끈다”가 아니다. **A 업데이트만 늦춘다.** Q12는 첫 update부터 U에 적용한다.

필수 비교:

\[
\Delta_Q^J=H(JQ)-H(J0),\qquad
\Delta_Q^D=H(DQ)-H(D0),
\]
\[
I_D=\Delta_Q^D-\Delta_Q^J.
\]

DQ−JQ는 Q12 아래 일정 차이, DQ−D0는 같은 지연 일정 아래 backend 순증분이다. ID는 이2×2 조건의 상호작용을 기술하는 값이며, 한 seed의 양수만으로 일반적인 시너지를 확정하지 않는다.

**D0/DQ 공통 경계 구현:** U의 optimizer와 50K scheduler를 재시작하지 않는다. A는 `1e-5 × g(global_update)`로 해제하며 warmup을 새로 시작하지 않는다. t=4999는 동결, t=5000은 해제 후 even update여서 L0 또는 LQ만, 첫 offset은 t=5001이다. A의 Adam moment/weight decay는 frozen 구간에 갱신되지 않아야 한다. [P §7.4]

### S5-R0 / PQ — U는 Q12, A는 plain reconstruction+offset

| 항목 | 설정 |
|---|---|
| Case / seed | PQ / 2026 |
| A 일정 | 처음부터 trainable50K; D와 결합하지 않음 |
| A 직접 gradient | **L0+LO** |
| U 직접 gradient | **LQ 전체** |
| 계수/LR | JQ와 동일; qA만 (0,0,0) |
| no-KD control | **J0 재사용**; 동일 의미 P0를 새50K로 만들지 않음 |
| 주 비교 | PQ−JQ, PQ−J0 |

PQ가 JQ보다 좋으면 “추가 감독을 U에 집중하고 A는 기본 목표로 적응시키는 정책”의 양성 패턴이다. 어떤 추가 항이 원인인지는 아직 구분되지 않는다. PQ가 불리하면 Q12의 A 직접 전달이 유용했을 가능성을 남긴다.

입력 PAN 전체 detach로 구현하지 않는다. 그 방식은 L0→A도 제거해 frozen-like 다른 실험이 된다. **loss별 gradient를 합한 뒤 optimizer step은 기존과 같이 한 번**이다. AMP/accumulation/clip을 중복 적용하지 않는다.

## 6. 조건부 확장 — 기본적으로 최대2run

확장은 아래 묶음을 모두 실행하는 것이 아니라 **자료가 지지하는 한 방향**을 선택한다. 추가 확인 seed 예산을 먼저 예약한다.

### 6.1 LF0 / LFQ — 초기 공동 적응 후 25K부터 A 고정, 2run

**Trigger:** J0/JQ의 후반 shift 변화와 raw HQNR 하락이 함께 반복되고, last/plateau가 선택값에 비해 후퇴하는 경우. 단순히 best가25K 이전이라는 이유만으로 실행하지 않는다.

| Case | 0–24999 | 25000–49999 | no-KD control |
|---|---|---|---|
| LF0 | J0 | A frozen/LO OFF, U=L0 | LF0 자체 |
| LFQ | JQ | A frozen/LO OFF, U=LQ | LF0 |

LR/optimizer U는50K 전역 일정 유지. LFQ−LF0, LFQ−JQ, LF0−J0를 함께 읽는다. 후반 25K의 A를 고정하는 것이지 T0 A로 되돌리는 것이 아니다.

**현재 grid 주의:** GRID1010에는 exact25K가 없다. 기존24240/25250 snapshot을25K로 부르지 않는다. exact25K의 A/U/optimizer/scheduler/AMP/data RNG 상태가 있을 때만 prefix를 재사용해 tail25K 비용으로 계산한다. 없으면 LF0/LFQ를 처음부터50K 학습한다. 새로운 J0/JQ에 exact25K state를 별도 저장할 수 있지만 selector 후보는 그대로50개다.

### 6.2 JK0 또는 JE0 — PQ 이득을 한 경로로 좁힘, 우선1run

**Trigger:** PQ가 유리하거나 고정-batch에서 특정 extra loss의 A gradient 충돌이 반복된다.

| Case | JQ에서 차단하는 경로 | 유지하는 경로 | 대조 |
|---|---|---|---|
| JK0 | LK→A | U의 soft, A의 weighted hard·edge·offset | JQ, J0, PQ |
| JE0 | LEw→A | U의 edge, A의 hard·soft·offset | JQ, J0, PQ |

먼저 유력한 **하나**를 실행한다. 작은 soft loss 비율만 보고 JK0를 고르지 않는다. JK0 성공은 U의 soft가 불필요하다는 뜻이 아니며 JE0 성공은 GT edge 자체가 불필요하다는 뜻이 아니다.

### 6.3 DPQ — delay+protected 결합, 조건부1run

**Trigger:** DQ−JQ와 PQ−JQ가 각각 유리하고, 해당 이득이 큰 RR 비용만으로 나타난 것이 아닌 경우.

0–4999 A frozen/U=LQ, 5000 이후 A=L0+LO/U=LQ. A LR은1e-5이며 AL의3e-6을 함께 섞지 않는다. no-KD control은D0, anchor는DQ, 추가 비교는PQ다.

DPQ−DQ는 delayed schedule 아래 routing 차이다. 단독 두 이득을 더해 예상 HQNR을 만들지 않는다. 선택 단계의 조합 비용을 별도로 기록한다.

### 6.4 선택된 정책의 soft 제거 대조, 조건부1run

최종 기여에 output distillation을 주장하려면 선택된 routing/schedule에서 LK만 제거한 대조를 확보한다.

| 선택 정책 | Soft-OFF case | U | A |
|---|---|---|---|
| JQ | 기존 XJ | L0+LD+LEw | 해당 U loss+LO |
| DQ | 신규 DX | 위 식 | D 일정 아래 위 식+LO |
| PQ | 신규 PX | 위 식 | L0+LO |
| DPQ | 신규 DPX | 위 식 | D 일정 아래 L0+LO |

β=0으로 만들되 나머지 계수·λE0·init·data order를 고정한다. 이 대조의 효과가 없으면 Q12 전체의 이득을 모두 output soft 기여로 쓰지 않는다. 이 묶음도 최대 2개 조건부 run 예산 안에서 우선순위를 정한다.

### 6.5 R1 fallback과 추가 fitting의 우선순위

Q12가 여러 대응 조건에서 불리하고 R1의 유효성 근거가 생기면 **JR → DR 또는 PR 중 하나**로 전환한다. DR/PR은 각각 D/P에서 U=R1, edge/soft OFF다. DR은D0, PR은J0가 no-KD control이다. 단일 s3 J0만으로 이 fallback을 선행시키지는 않는다.

`TCOPY10/CONT10`은 상위 계획의 별도 초기화·tail protocol이다. s3의 완료 J0를 활용할 수 있어도 **main s5 질문과 다른 축**이므로 이번 기본8/확장10run에 넣지 않는다. Timing/routing이 좁혀지고 실제 잔여시간·full parent state·구현이 있을 때만 별도 승인된 부록 캠페인으로 편성한다. Fresh50 평균에 섞지 않는다.

## 7. 진단과 결과 판정

### 7.1 측정은 loss별·모듈별로 분리한다

고정 train diagnostic batch, 독립 RNG에서 L0/LD/LK/LEw/LO 각각의 A/U gradient norm을 기록하고 native L0에 대한 cosine도 기록한다. 진단을 실제 학습 gradient에 두 번 합산하지 않는다.

A가 내는 값이2개이므로 native loss의 `∂Lj/∂cS`도 기록한다. Offset target의 native cS는 SG이므로 LO의 직접 native-gradient를 억지로 복원하지 않고 cε 경로·A parameter에서 측정한다.

공통 probe에서 native shift 분포, `cS−cT`, signed2×2 response matrix, offset EPE를 같은 checkpoint에 연결한다. **Teacher와의 shift 거리가 작은 것 자체를 정답으로 채택하지 않는다.** EPE는 synthetic 관계 오차이며 native absolute alignment GT 오차가 아니다.

일반 평가·diagnostic의 RNG를 분리하고, frozen 기간의 offset forward 생략으로 U 데이터 순서나 해제 후 epsilon 순서가 변하지 않게 한다. 권장 구현은 기존 전용 offset RNG를 전역 odd update마다 동일하게 소비하되 frozen 구간에서는 jittered-A forward/backward만 생략하는 것이다.

### 7.2 ERGAS와 seed

기존 계획의 부호·경고선을 유지한다. `ΔH=Hcandidate−Hcontrol`은 양수가 개선, `ΔE=Ecandidate−Econtrol`은 음수가 개선이다. ERGAS는 raw-selected / exact50K / 지정 plateau를 별도 보고한다. [P §10]

| 관측 | 처리 |
|---|---|
| 한 seed의 작은 ERGAS 악화 | 즉시 탈락하지 않고 같은 시점과 추가 seed를 확인 |
| 같은 checkpoint 재평가 불일치 | seed 설명보다 export/scale/clip/평가기/hash 검사 우선 |
| 여러 seed의 last/plateau 반복 악화 | 실제 RR trade-off로 남김; seed 탓으로 면제하지 않음 |
| n=1 | 표준편차=0이 아니라 NA |
| s3/s5 같은 seed 반복 | 별도 환경 replicate, n_unique_seed 증가 없음 |

상위 계획의 운영 경고(+2% 반복, +5% 큰 단일-block 비용)는 통계 유의성이나 허용 화질의 증명이 아니다. 3block 기준의 `최소2/3` 규칙을 s5 탐색 1seed에 적용하지 않는다. 확인 2seed만 확보되면 그 두 결과를 그대로 보고한다.

### 7.3 목표 달성과 방법 기여를 나눈다

**Asset hit:** 검증된 한 selected checkpoint의 full-precision raw HQNR ≥0.959 또는 ≥0.960. Sheet의4자리 반올림만으로 경계값을 판정하지 않는다.

**Repeated gain:** 같은 정책 control 대비 여러 독립 Student seed의 대응 개선이 일관되는가. 지금의2026+9091 두 seed만 실행한 신규 WIN은2-seed 결과다.

**Method-level target:** 그 동일 method의 실행된 seed 평균이 목표를 넘는가. 평균, sample sd(n−1), worst, hit count, selected/last/plateau를 함께 제시한다. s1–s3의 다른 case 결과를 WIN의 seed에 편입하지 않는다.

20scene의 장면 변동은 학습 seed 변동을 대신하지 않는다. FR20은 이미 method와 checkpoint 개발에 사용한 benchmark이므로 독립 미사용 test 검증으로 표현하지 않는다. V64와 raw 중 큰 것을 고르지 않는다.

### 7.4 선택·중단 원칙

J0/JQ 및 시작한 D0/DQ 대응쌍은 계약 실패 외에는50K 완주를 우선한다. 10K는 correctness/안정성 확인, 25K는 다음 신규 후보 admission과 drift 확인이지 최종 방법 판정 시점이 아니다.

선택적 신규 후보의 추가 seed 보류는 기존 계획의 사전 정의 기준과 남은 예산으로 결정한다. 나쁜 run을 기록에서 삭제하거나, 중단 run을50K로 집계하지 않는다. **한 번 raw0.959를 넘으면 새 계수 탐색보다 winner 고정과 확인을 우선**한다.

## 8. 확인 seed 9091 묶음

Winner lock에 case·전체 config·code·Teacher/initialization/calibration/evaluator hash·발견에 사용한 run들을 기록한다. s4 확인 seed3407을 s5 탐색에 사용하지 않는다.

| WIN | seed 9091에서 필요한 묶음 | 확인 가능한 효과 |
|---|---|---|
| JQ | J0,JQ | 기본 KD의 별도 Student seed |
| DQ | **D0,JQ,DQ** | D 아래 KD 순증분, Q12 아래 delay 효과 |
| PQ | **J0,JQ,PQ** | KD 순증분, Q12 아래 protected routing |
| LFQ | **LF0,JQ,LFQ** | LF 아래 KD 순증분, Q12 아래 late-freeze |
| JK0/JE0 | J0,JQ,WIN | 부분 routing의 추가 이득 |
| DPQ | **D0,DQ,DPQ** | D 아래 backend 이득, D 아래 protected 추가 이득 |
| DR | D0,JR,DR | R1 아래 delay 및 D 아래 R1 이득 |
| PR | J0,JR,PR | R1 아래 protected 및 R1 이득 |

DQ 확인 묶음에는 J0가 없으므로 9091에서 전체 2×2 interaction ID까지 확인한 것은 아니다. DPQ 묶음에도 JQ가 없으므로 9091의 J→D 효과는 별도다. 필요한 주장을 넘어 확장하지 않는다.

9091 결과를 보고 계수·freeze 시점을 다시 수정하면 이 seed는 그 다음부터 개발에 사용된 것으로 표시한다. 추가 독립 확인은 새 seed를 사전 지정해야 한다. 여유가 생기면 **새 case보다 WIN의 세 번째 대응 seed 반복**을 우선하되 별도 예산표를 갱신한다.

## 9. 실제 큐와 공통 마감

### 9.1 절대 시계

조회한 상위 clock은 다음과 같다. [CLOCK]

```text
start             2026-09-14 13:22:31 Asia/Seoul
training_deadline 2026-09-16 11:22:31 Asia/Seoul
final_deadline    2026-09-16 15:22:31 Asia/Seoul
```

s5의 로컬 timezone이 다르면 naive local time으로 해석하지 말고 timezone-aware 값으로 변환한다. 이 파일을 투입 시각으로 다시 쓰지 않는다. GPU slot 1/run 1을 가정하며 s5 실제 사양이 확인되면 throughput만 갱신한다. 여러 GPU가 있어도 batch나 분산 방식을 조용히 바꾸지 않는다.

### 9.2 Readiness-aware 편성

```text
S5-G0  환경 / T0 / init hash / 공통 data·evaluator / GPU 처리량 확인
S5-B0  J0_S2026
S5-B1  JQ_S2026                         [공통 λE0 필요]
S5-T0  D0_S2026                         [C1 timing gate 필요]
S5-T1  DQ_S2026                         [timing gate + λE0]
S5-R0  PQ_S2026                         [C1 routing gate + λE0]
LOCK   선택 후보·대응 control·code·hash 고정
CONF   seed 9091의 control / anchor / WIN [예산을 탐색 전에 예약]
AUDIT  selected/last export·집계·gspread 기록 검증
```

J0 뒤 λE0가 없고 timing 구현은 준비되었다면 D0를 먼저 돌린다. 두 조건 모두 미준비이면 correctness 검사·진단을 우선하고, λE 대신 임의 상수를 넣지 않는다. 단순히 GPU를 채우려고 s4 scalar를 복제하지 않는다.

조건부 최대 2run(LF pair / 부분 routing / DPQ / soft-off)은 **CONF 실행 전에** 할 수 있지만, 그 시간을 빼고도 확인 묶음이 끝나는 경우에만 admission한다. CONF 시작 후에는 WIN을 다시 고르지 않는다.

현재 저장소는 큐 파일에 J0만 두고 gate가 다음 run을 편성한다. s5도 그 구조를 확장한다. 구버전의 `stage1 J0→F0→JR→FR 전부`를 고정 큐로 되살리지 않는다. [R0, R4]

### 9.3 예산 산식

`R_train`은 현재부터 공통 training_deadline까지 남은 시간, `h_case`는 **train+정기 평가+필수 export/진단** 처리시간 추정이다. 설정시간 `h_setup`도 남아 있으면 포함한다.

\[
h_{setup}+1.1\sum_{r\in\{현재 승인할 run,남은 대응·확인 run\}} h_r\le R_{train}.
\]

final_deadline까지의 시간으로 계산하면 마지막 4시간 audit를 먼저 뺀다. 이미 training_deadline을 사용한 산식에서 4시간을 다시 차감하지 않는다. 가중치·data 복사, 반복 진단, upload 재시도 시간도 비용 ledger에 남긴다. s3의 Train(h)=1.16은 s5 예상치의 근거로 쓰지 않는다.

### 9.4 처리시간 가정에 따른 예시 — 실제 예약값이 아님

일반 50K=2h, PQ형 routed 50K=2.5h, 준비 2h, 마지막 audit 4h, safety1.1이라고 **가정**한다. WIN=PQ일 때:

| 실행 묶음 | 학습 run | 계산한 필요 총시간 | 판단 예시 |
|---|---:|---:|---|
| J0/JQ | 2 | 2+1.1×4+4 = 10.4h | 최소 bridge·backend 대응 |
| J0/JQ/D0/DQ | 4 | 2+1.1×8+4 = 14.8h | timing2×2까지; 새 seed 확인 전 |
| Core 5 | 5 | 2+1.1×10.5+4 = 17.55h | timing+routing 탐색 1seed |
| Core 5 + 9091 control/anchor/PQ | 8 | 2+1.1×17+4 = 24.7h | **기본 권장 완료 단위** |
| 위8 + 일반 50K 조건부2 | 10 | 2+1.1×21+4 = 29.1h | 남은 30h와 가정 처리량일 때 가능 |

WIN이 다른 경우의 실제 run별 비용으로 다시 계산한다. Prefix 재사용 가능한 LF tail은 실측한 추가 25K만 비용으로 세되, 원 prefix 비용도 전체 계보 보고에 남긴다. 처리시간이 길면 조건부 탐색을 먼저 줄이고, 이미 시작한 대응쌍과 최종 감사를 보존한다.

## 10. s1 구현 인계

### 10.1 현재 구현과 새로 필요한 것

검토 snapshot `2303e56acc44abeaf84695c1df6e7d83fb78932f`의 generator에는 J/F/AL/scalar와 s4까지 있다. **s5, D/P/LF 기능은 이 문서 작성만으로 구현된 것이 아니다.** 아래는 변경 요구다. [R0, R4, GEN]

| 파일/구성 | 요구 변경 |
|---|---|
| `tools/gen_pakd50_configs.py` | s5 seed2026 등록, s5 전용 priority·mandatory·stage 목록, D/P/LF/부분 routing case와 지원 여부 검증 |
| `tools/pakd50_prepare.sh` 및 서버 검사 | s5 허용, local path·data·Teacher·환경 대조. 알 수 없는 서버의 거부는 유지 |
| `train_kdv.py` / `kdv/registry.py` | loss별 수신자와 A 동결 일정 해석. 미지원 key/case는 즉시 거부 |
| `kdv/forward.py` | Teacher 자체 경로·native graph·offset branch의 의미 보존. 필요한 부분만 수정 |
| `tools/pakd50_unit_tests.py` | timing/routing 신규 검사 및 기존 J/F/AL/Q12 회귀 검사 |
| Dynamic gate | λE와 **코드의 기능 지원 여부**를 모두 확인. 미완료 대응쌍·확인 run의 예산 예약 |
| 산출물 | run별 A 활성 상태, routing, 경계 state, checkpoint provenance, server/case/seed 식별 |

`policy_id=D` 또는 `P`를 YAML에 적는 것만으로 기능이 생기지 않는다. Parser가 무시하는 key를 추가해 실제로는 J와 같은 실험을 실행하는 오류를 막아야 한다. **동봉한 YAML은 계획용 registry이지 trainer config가 아니다.**

### 10.2 Freeze 구현의 필수 조건

A parameter group은 기존 optimizer와 scheduler 규약을 보존한다. Frozen 구간에는 `grad=None`로 두어 실제 Adam update·weight decay·moment step이 발생하지 않게 한다. **단순히 LR=0으로 설정하는 것만으로 동결을 대체하지 않는다.** A를 처음부터 optimizer에서 제외한 구현이라면, 해제 시 group 추가와 base LR 처리를 별도로 검사한다.

A 해제 시 U optimizer·moment·scaler·data iterator를 reset하지 않는다. Offset의 전역 parity와 전용 RNG 순서를 유지한다. Frozen 구간에 LO를 진단용으로 계산한다면 `diagnostic_only`로 기록하고 학습 목적에 넣지 않는다.

동결 경계는 **0-based next update index**로 구현한다. 예를 들어 D에서 5,000회의 업데이트를 완료한 상태의 다음 index는 5000이고, 이 update부터 A가 학습된다. `completed_updates`와 `next_update_index`를 혼동해 한 step 빠르거나 늦게 해제하지 않는다.

### 10.3 Routing 구현의 필수 조건

A/U gradient를 선별해 같은 scale로 합산하고, 기존 optimizer step을 한 번 수행한다. Loss별로 optimizer를 여러 번 step하거나, aligned PAN 전체를 detach하거나, 다른 경로까지 지우는 불명확한 gradient hook을 사용하지 않는다.

`qA=(1,1,1)`의 routed J가 원래 JQ와 같은 gradient·1-step 결과를 내는지 먼저 검사한다. N0에는 추가 항이 없으므로 routed N0와 기존 J0도 동치여야 한다. 실제 AMP·accumulation·clip·resume를 포함한 검사 전에는 CUDA 재현 검증을 완료했다고 표시하지 않는다.

선택된 추가 항을 U에만 보내기 위해 별도의 detached-input U forward를 쓰는 구현도 가능하지만, 이때는 추가 forward 비용과 stochastic layer·buffer의 동작을 확인한다. 단순히 계산되는 scalar loss가 같다는 사실만으로 parameter update 동치가 검증된 것은 아니다. [P §13.3]

### 10.4 최소 회귀 검사

| ID | 통과해야 하는 의미 |
|---|---|
| S5-G01 | T0 A/U hash 고정, Student와 parameter 객체 분리, T0 raw 재현이 기존 허용오차 이내 |
| S5-G02 | J0/JQ의 기존 forward/loss/gradient/1-step 동치 |
| S5-G03 | D의 4999/5000/5001에서 A 상태·LR·offset parity, frozen 구간 A hash/moment 불변 |
| S5-G04 | LF의 24999/25000 경계, U 상태 보존, A/LO 정지 |
| S5-G05 | PQ의 LD/LK/LEw→A 없음, L0/LO→A 있음, U의 Q12 경로 유지 |
| S5-G06 | JK0/JE0에서 지정한 항 하나만 A 전달 차단. U 경로는 모두 유지 |
| S5-G07 | 미배포 λE·미지원 case·부정확한 parent/full state는 즉시 거부 |
| S5-G08 | freeze/RNG/diagnostic 유무가 공통 data 순서와 epsilon sequence를 바꾸지 않음 |
| S5-G09 | 동결 경계 직전 중단·resume와 연속 실행의 상태 및 다음 step 대조 |
| S5-G10 | 공식 candidate 50개 유지. exact25K 추가 저장은 selector 밖 |
| S5-G11 | N0/soft-off 해제 관계, GT/Teacher/gate stop-gradient, MS base 한 번 |
| S5-G12 | WV3-s5 분리, PAKD50/X열 표기, metadata 업로드가 metric 값을 바꾸지 않음 |

s1에서 새 release를 검증한 뒤 push하고, s5는 고정 SHA를 pull/fetch한다. 학습 중 checkout을 pull로 바꾸지 않고 새 worktree 또는 run 경계의 고정 release로 전환한다. Baseline 실행 후 C1을 배포했다면 구 commit과 새 commit의 J0/JQ 회귀 통과 여부와 실제 실행 SHA를 함께 남긴다.

## 11. gspread 인계 — 직접 Sheet 수정 없음

기존 파이프라인을 유지한다.

```text
run 완료/검증 → tools/_upload.sh → gspread/gspread_upload.py
            → pan-cvpr27 / WV3-s5
```

| 위치 | 값/규약 |
|---|---|
| `gspread/server.txt` | `s5`. Service-account credential은 기존 안전한 방식으로 배치하고 Git에 추가하지 않음 |
| Worksheet | `WV3-s5`, 기존 업로더의 생성·재사용 경로 |
| C열 캠페인 | `PAKD50` |
| B:W | 기존 metric 열과 Notes 위치 유지 |
| N / O | raw HQNR / V64 각각 기존 정의 |
| X열 통합실험 | `PAKD50 / DQ / FRESH50`, `PAKD50 / PQ / FRESH50` 등 |
| Sidecar manifest | server, case, protocol, init/Teacher/code/checkpoint/calibration hash, timing/routing |

서버가 늘었다고 새로운 Y열을 추가하지 않는다. 이미 추가한 X열을 s5에도 적용한다. 같은 run 이름을 s3와 s5에서 사용할 수 있으므로 전체 집계 키는 `server + logical_run_id + execution_id/config_hash`로 구별한다. 평균의 독립 seed 수는 별도 필드다.

s3의 조회 행은 C=MISC/X빈값이었다. 현재 repo에는 PAKD50 분류가 반영되어 있으므로, 필요 시 s3의 실행자가 새 업로더로 해당 완료 run을 다시 올려 metadata를 갱신한다. 이관 때 metric·checkpoint를 바꾸지 않고 전후를 대조한다. 이 배정서는 업로더를 통한 처리를 명세하며, 직접 Sheet 생성이나 셀 수정을 수행하지 않는다.

## 12. 결과 인계와 최종 판독

### 12.1 반드시 남길 파일의 의미

```text
s5_assignment_manifest.json      # code/Teacher/calibration/clock/seed/case/capability
s5_bridge_summary.csv            # s3↔s5 동일-seed, 독립 seed 집계에서 제외
s5_paired_summary.csv            # s5 내부 control↔candidate, selected/last/plateau
s5_gradient_routes.jsonl         # loss×module 및 cS 진단, frozen 상태
s5_timing_boundaries.json        # next_update_index / A 활성 / LR / optimizer state
s5_winner_lock.json              # 확인 seed 실행 전에 고정
s5_confirmation_summary.csv      # 9091 결과, 독립 seed와 환경 반복 분리
s5_upload_status.json            # worksheet/run/metric provenance/실패 재시도
```

위 파일이 현재 존재한다는 뜻은 아니다. 같은 의미의 기존 파일을 재사용해도 되며, 누락 여부를 manifest에 기록한다.

### 12.2 다음 결정을 위한 표

| 관측 | 우선 결정 |
|---|---|
| DQ>JQ, DQ>D0이며 후반도 유리 | Delayed joint를 확인 seed로 보냄 |
| D0만 개선하고 DQ−D0가 작거나 음수 | 일정 효과와 KD 효과를 분리. 정합 보호만 기여했을 가능성 |
| PQ>JQ, PQ>J0 | Protected routing 확인. 여유가 있으면 JK0/JE0 또는 soft-off |
| JQ가 PQ/DQ보다 좋음 | 추가 제약 없이 joint 유지. 신규 방법을 억지로 선택하지 않음 |
| JQ best는 좋고 후반 shift·raw가 후퇴 | LF0/LFQ 검토. exact25K state가 없으면 처음부터 재실행 |
| D/P 단독이 모두 유리 | 최대 1개 DPQ 결합. 가산 이득을 미리 예측하지 않음 |
| Q12가 불리하고 R1의 반복 근거가 있음 | JR→DR/PR 중 하나의 대안. 대응 control 유지 |
| raw≥0.959인 단일 자산 확인 | 코드·방법 lock 후9091 확인. 추가 탐색 감축 |

## 13. 출처와 설계의 지위

**[S3]** 동반 `PAN_S5_S3_Result_and_Repository_Snapshot_2026-09-14.md` 및 `PAN_S5_S3_Observed_Row_2026-09-14.json`. Native Google Sheets range API로 읽은 s3 행이다. 일반 workbook export는 s3 행을 식별하는 근거로 사용하지 않았다.

**[A]** `PAN_Aligner_L1E4_TechnicalSpec_Evidence_KD_Handoff_2026-09-14.md`, §1–6, §11.4–11.7.

**[K]** `PAN_KD_Q12_R1_Spec_and_Aligner_Interface_2026-09-14.md`, §3–7, §9–10.

**[P]** `PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md`, §7.2–7.6(D/P/LF/JK0/JE0), §8.3–8.4(tail protocol), §10(seed/ERGAS), §13.3(routing).

**[R0]** [PAKD50 구현 노트@2303e56](https://github.com/hojunking/PAN-Crafter-repro/blob/2303e56acc44abeaf84695c1df6e7d83fb78932f/research_log/2026-09-14_pakd50-implementation.md), §1/§4/§6. 초기 본문과 후속 수정이 충돌하면 §6의 동적 큐·공통 마감을 우선한다.

**[R4]** [s4 배정 구현 노트@2303e56](https://github.com/hojunking/PAN-Crafter-repro/blob/2303e56acc44abeaf84695c1df6e7d83fb78932f/research_log/2026-09-14_pakd50-s4-review-and-implementation.md).

**[GEN]** [생성기@2303e56](https://github.com/hojunking/PAN-Crafter-repro/blob/2303e56acc44abeaf84695c1df6e7d83fb78932f/tools/gen_pakd50_configs.py), server/case/policy/priority 및 calibration 연결.

**[CAL]** [Calibration asset@2303e56](https://github.com/hojunking/PAN-Crafter-repro/blob/2303e56acc44abeaf84695c1df6e7d83fb78932f/assets/pakd50/calibration_resolved.json).

**[CLOCK]** [공통 clock@2303e56](https://github.com/hojunking/PAN-Crafter-repro/blob/2303e56acc44abeaf84695c1df6e7d83fb78932f/assets/pakd50/campaign_clock.json).

**기존 계약:** T0·W112D123·M-frame·L1E4 offset·Q12/R1 수식·공통 grid·calibration·50시간 마감.

**이번 새 배정:** s5 전용 timing/routing 역할, 탐색 seed2026/확인9091, 기본5+확인최대3, 조건부최대2, 구현·자산 준비 여부를 반영한 실행 순서. D/P/LF의 유효성은 아직 결과가 아니다.

**최종 핵심 질문:** Teacher가 학습한 정합을 사용하는 동일 Student에서, **정합 업데이트의 시점과 손실 수신자를 조정하면 기본 joint Q12보다 raw HQNR을 더 높이고 그 개선을 다른 seed에서도 유지할 수 있는가?**
