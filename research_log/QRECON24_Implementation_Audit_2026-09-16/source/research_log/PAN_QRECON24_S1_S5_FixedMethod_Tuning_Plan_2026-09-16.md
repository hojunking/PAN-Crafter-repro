# QRECON24 — q/e 기반 단순 KD의 전 서버 하이퍼파라미터 튜닝
## s1–s5 통합 실행 인계 / 2026-09-16

**방법 정의는 사용자 합의로 고정했다. 성능 우위가 검증됐다는 뜻은 아니다.**

- **주 방법:** T0 aligner를 Student에 복사한 뒤, **고정 Teacher q로 가중한 GT hard reconstruction만으로 Student aligner를 작게 업데이트**한다. U-Net은 기존 adaptive KD와 약한 q-weighted GT edge로 학습한다.
- **대체 대상:** 아직 실행하지 않았다고 확인한 기존 QETUNE24의 86run 계획. 이 계획은 거기에 추가되는 큐가 아니다. 같은 파일명/실험명을 재사용하지 않는다.
- **캠페인 제안 ID:** `QRECON24_A104D121_S1S5_20260916_v1`; run 토큰 `QRC24`.
- **기본량:** **FRESH50 68run**, 모든 run 50,000 updates. Student U는 fresh 초기값, A만 T0에서 이전한다. 이것이 이번 ‘aligner fine-tuning’의 뜻이다. 기존 완료 U까지 잇는 FT20는 기본 68run에 섞지 않는다.
- **시간:** 각 서버 신규 학습의 최소 운영구간을 **24시간**으로 둔다. 24시간에 학습을 끊지 않고 대응 묶음을 끝낸다. 현재 기록을 대용하면 학습 기준 합 24.30–27.60h/서버, 가예약 약 32–35h/서버다. 새로운 gradient 분리의 실측 전 예측이며, 그 이후 실제 시간으로 갱신한다.
- **목표:** raw-original HQNR **≥0.9585**, 통과 자산 안에서 **RR SCC → ERGAS → PSNR → SAM → Q8 → SSIM**. V64/fSCC와 혼동하지 않는다.
- **이 문서의 상태:** 실행 설계·구현 인계다. 학습 기동, repository push, Google Sheet 변경을 수행하지 않았다.

> 확정한 방법을 다시 복잡하게 바꾸지 않는다. 주력의 자유 조정값은 **직접 edge 계수 λE, aligner LR 비율 rA, 기존 α·β, optimizer peak LR, Student seed**다. q 함수와 qref는 고정한다.

## 0. 서버별 역할과 전체 규모

| 서버 | 핵심 질문 | 신규 run | Student seed | 학습 대용값 합 | 여유 포함 가예약 |
| --- | --- | --- | --- | --- | --- |
| s1 | Aligner 추가 적응·q 배분의 필요성, 작은 A LR | 12 | 1234, 3407 | 26.40h | 33시간 40분 48초 |
| s2 | 극소 edge × A LR 조합 탐색 + edge의 q 대조 | 12 | 777 | 27.60h | 35시간 07분 12초 |
| s3 | s2의 3×3 조합을 두 다른 seed에서 반복 | 18 | 2026, 4321 | 24.30h | 32시간 09분 36초 |
| s4 | e 기반 hard/soft 계수와 RR 절충 | 14 | 1234, 3407 | 27.16h | 34시간 55분 31초 |
| s5 | 같은 방법의 학습 속도 축소와 4-seed 반복 | 12 | 2026, 777, 9091, 1103 | 26.04h | 33시간 14분 53초 |

기본 큐는 사전 고정되어 서버 간 승자 결정을 기다리지 않는다. 초기 구현/자산 준비가 끝난 서버부터 시작하며, 정상 진행 중인 과거 run은 마지막 update까지 마친 뒤 큐 경계에서 전환한다. 대기·설치·과거 캠페인 시간을 신규 24시간 학습량으로 세지 않는다.

## 1. 현재 결과에서 무엇을 반영했는가?

### 1.1 이번 조회 범위와 숫자의 성격

이번에 `pan-cvpr27`의 s1–s5 WV3 탭을 다시 읽었다. 분석 범위는 **PAKD50 계열의 W104–D121 관련 57행, 동일 server+run ID 중복 2행을 제외한 55run**이다. s1 5 / s2 7 / s3 14 / s4 18 / s5 11건이다. 다른 구조·별도 진단·과거 archive를 전부 합친 개수가 아니다.

s5의 QE50 seed2026 19/23행과 seed777 22/24행은 같은 run ID이며 이 표에서 지표·Train(h)가 같아 각각 한 번만 집계했다. Sheet 원문은 수정하지 않았다. 아래 숫자는 **원래 selected checkpoint의 raw HQNR와 같은 행의 RR**다. 최신 last·plateau·선택 step·서버 프로세스 상태를 이번 Sheet 조회로 확인한 것은 아니다. 부록 A에 전체 관련 run과 원본 행을 남겼다. [SHEET]

### 1.2 새 방법을 설계할 때 사용한 관측

| 근거 | 이번 확인값 | 이번 계획에 반영한 판단 |
| --- | --- | --- |
| s2 seed777, edge 배율 | 0배 0.9588 / .25배 0.9548 / .5배 0.9561 / 1배 0.9577 / 2배 0.9557 | 강한 edge를 주력으로 되돌리지 않는다. 단, 단조 감소 법칙이 아니므로 3e-4/1e-3/3e-3의 작은 양수끼리 비교한다. |
| s1 seed1234, binary q 대조 | QE50 0.9574 / QES 0.9574 / QEC 0.9564 | 기존 이진 q 선택이 반드시 필요한 것으로 전제하지 않는다. 새 연속 q도 uniform/shuffle로 확인한다. |
| s3, 다른 seed | QE50−JQ: S2026 +0.0007 / S4321 −0.0011 | 같은 개입의 부호가 seed에 따라 달라져 3×3 조합을 3개 초기값에서 반복한다. |
| s4, 다른 seed | QE50−JQ: S1234 −0.0022 / S3407 +0.0032 | 새 q 방식도 좋은 한 seed로 채택하지 않는다. S3407 QE50는 H=0.9585, SCC=.9876, ERGAS=2.0746으로 표시됨. |
| s4 S1234, joint/late freeze | JQ .9565 / LFQ .9586 / LFX .9589 | 정합 업데이트 강도·경로가 중요한 후보축이다. LF가 새 hard-only q 업데이트의 성공을 입증하지는 않는다. |
| s5 S2026, 상수/시간/완만한 q | 상수 .75배 .9576 / DOWN .9571 / UP .9567 / FLOOR .9567 | 추가 함수나 schedule을 만들지 않는다. 단순화된 식과 작은 계수에 집중한다. |
| s1 J0의 높은 H와 RR 비용 | H .9595 / SCC .9771 / ERGAS 2.6221 | HQNR만 높고 RR가 무너진 자산을 최종 성능 방법으로 택하지 않도록 같은-checkpoint RR 목표를 유지한다. |

**자료의 결론과 새 가설은 구분한다.** 기존 s2 보고서는 QE50−JQ를 selected −0.000032, ≥30K plateau +0.001228로 기록하며, 단일 seed에서 차이를 확인하지 못한 것이 동등성 증명은 아니라고 제한한다. 보고서의 “edge 자체 양성”이라는 표현은 원문 해석으로 보존한다. 다만 이 계획은 JQ−J0에서 hard/soft/edge가 함께 바뀐다는 점 때문에 그 전체 차이를 edge 단독 기여로 재인용하지 않는다. [S2 §2·§6]

q↓·e↑ 집단이 PAN 위치 변화에 민감하고 보정 이득이 컸다는 기존 분석은 **q-weighted aligner reconstruction의 동기**다. 현재 보정 vs zero의 이득이며, 현재 보정에서 더 튜닝한 추가 이득의 증거는 아니다. q는 절대 정합 정확도나 Teacher-quality 확률이 아니다. [ANALYSIS §2–4]

**새 방법(연속 q + hard-only A + Student offset 제거)은 아직 결과가 없다.** 기존 J/F/LF/PQ/RC/QER 결과를 그 방법의 실행 결과로 기록하지 않는다. 과거 case는 설계의 배경이고, 새 비교는 같은 release·같은 초기값에서 다시 만든다.

## 2. 고정 method — 이 정의를 모든 서버가 공유한다

### 2.1 Forward와 초기화

Teacher `T0`는 `PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1`의 `best_hqnr`, step 24240이며, A/U가 같은 weight package다. 기대 full SHA256:

`16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32`

Teacher A_T/F_T는 frozen/eval. Fresh Student는 A_S←clone(A_T), U-Net F_S는 W104–D121의 seed별 독립 저장 초기값이다. 원 Teacher U를 Student에 잘라 복사하지 않는다. A head를 다시 zero-init하지 않는다.

\[
M=\operatorname{bicubic}_{\times4}(MS),\quad
c_T=A_T(P,M),\quad c_S=A_S(P,M),
\]
\[
Z_T=M+F_T([\mathcal W(P,c_T),M]),\qquad
Z_S=M+F_S([\mathcal W(P,c_S),M]).
\]

PAN만 sampling한다. M·Y·최종 출력의 목표 frame은 고정이다. Teacher는 Student의 c_S를 사용하지 않으므로 target과 q는 Student 학습에 따라 바뀌지 않는다. GT는 forward 입력이 아니라 loss target이다. 기존 validated wrapper·정규화·sampler를 그대로 재사용한다. [KD §3·§9, ALIGN §2–4]

### 2.2 e 기반 hard/soft — 기존 R3 정의

\[
e_T(i,p)=\operatorname{mean}_c|Z_T-Y|,\quad
e_S(i,p)=\operatorname{mean}_c|Z_S-Y|,\quad
k(i,p)=\operatorname{mean}_c|Z_S-\operatorname{sg}(Z_T)|.
\]
\[
d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\qquad
a_T=\operatorname{sg}\operatorname{clip}_{[0,1]}
\frac{[e_S-e_T]_+}{e_S+10^{-6}}.
\]
\[
H_i=\operatorname{mean}_p[(1+\alpha d_T)e_S],\qquad
K_i=\operatorname{mean}_p[\beta(1-d_T)a_Tk].
\]

**τR=0.012463942170143127**를 고정한다. GT hard의 바닥 계수는 항상 1이다. α는 추가 GT 재가중의 강도이고 β는 이미 K_i에 포함된 output soft 강도다. Student가 더 정확하면 해당 pixel의 soft는 0이다. 가중치 계산만 detach하고 실제 e_S/k의 학습 경로는 유지한다. Pixel map을 patch scalar와 혼동하지 않으며, soft-active 수나 가중치 합으로 재정규화하지 않는다. [KD §4–5]

### 2.3 q 함수 하나를 A와 edge가 함께 쓴다

\[
\boxed{w_i=\operatorname{sg}\frac{2q_{\mathrm{ref}}}{q_{\mathrm{ref}}+q_T(i)}}.
\]

**qref=0.3276133416220546**: 기존 T0/train/AXIS16 calibration 중앙값을 기준 scale로 재사용한다. q가 낮을수록 큰 양수, 중앙값에서 1이며, q≥0일 때 0<w≤2다. 평균 w가 반드시 1은 아니다. Batch 정규화, threshold/temperature/δ/low-high 계수/추가 clip을 넣지 않는다.

q는 `assets/qedge9/cue_T0_AXIS16_v1.json` 및 연결 NPZ의 **raw q**를 sample index·실제 augmentation state별로 조회한다. AXIS16은 .25/.5/1/2 HR px의 4축 probe다. 기존 `gate_low_q` 0/1 값에 연속 함수를 적용하면 다른 방법이다. Teacher probe는 기존 cache 준비 과정에만 있고, Student 학습 매 batch에 16번 probe하거나 Teacher U를 16번 호출하지 않는다.

### 2.4 Edge는 GT target, 단일 직접 계수

\[
E_i=\tfrac12\operatorname{mean}|D_xZ_{S,i}-D_xY_i|
+\tfrac12\operatorname{mean}|D_yZ_{S,i}-D_yY_i|.
\]

기존 signed Scharr 3×3 /32, x/y 동등 평균, 8밴드별 비교, reflect1, 함수 입력 1px 내부 support, FP32를 유지한다. Teacher edge·PAN edge·variance loss가 아니다. [KD §5.2]

**λE는 절대 계수다.** 예: `lambda_edge=0.001`이면 그대로 `0.001*mean(w*E)`이며, 이전 λE0=0.09075나 rE를 다시 곱하지 않는다. 새 gradient-pilot calibration을 요구하지 않는다. 이번 기본 68run 모두 λE>0이며, edge를 끈 실험을 q-edge 방법으로 집계하지 않는다.

### 2.5 가장 중요한 고정: 모듈별 gradient를 분리한다

\[
\boxed{L_U=\frac1B\sum_i[H_i+K_i+\lambda_E w_iE_i]},
\qquad
\boxed{L_A=\frac1B\sum_iw_iH_i}.
\]
\[
G_U=\nabla_{\theta_S}L_U,\qquad
G_A=\nabla_{\phi_S}L_A.
\]

| 신호 | Student U-Net | Student aligner |
|---|---|---|
| GT hard H_i | 기존 H_i | **w_i·H_i만** |
| Teacher output soft K_i | 사용 | **직접 전달 없음** |
| GT edge | **λE·w_i·E_i** | **직접 전달 없음** |
| Student offset consistency | 없음 | **없음** |

`L_A`의 H_i는 plain GT L1이 아니라 **같은 α,d_T를 쓰는 재가중 hard**다. 따라서 q-low·e_T-high일 때 A에 전달되는 데이터 gradient의 계수는 w_i(1+αd_T)다. β와 λE는 해당 update의 A gradient에 직접 들어가지 않는다. 다만 U가 달라진 다음 update의 A 학습에는 간접 영향을 줄 수 있다.

**L_U+L_A를 합쳐 전체 모델에 backward하면 잘못된 구현이다.** U hard가 중복되고, A가 받아서는 안 되는 soft/edge도 들어갈 수 있다. 두 목적함수는 같은 Student forward/같은 parameter 상태에서 각 parameter 집합의 gradient만 구한다. A를 먼저 step하고 U를 다시 계산하는 alternating 최적화가 아니다. q를 전체 loss에 곱하거나 이미 합친 batch A gradient에 평균 q를 곱하지 않는다.

### 2.6 Student 단계에서 제거하는 항목

Student offset loss, λoff, 학습용 PAN jitter 경로, radius·격번 주기·ramp, geometry/flow KD, 예측 shift shrinkage, A freeze schedule, soft/edge→A routing 계수는 **없다**. Stage 1에서 T0를 만든 L1E4 학습은 삭제하지 않는다. Student에 no-offset을 적용하는 것은 이미 검증된 Stage 1을 다시 정의하는 일이 아니다.

Teacher q는 고정되므로 Student의 consistency 붕괴를 자동으로 막는 regularizer가 아니다. 작은 A LR과 실제 shift/응답 기록으로 변화를 관찰하되, 성능이 나쁘다는 이유로 일부 run에만 offset loss를 조용히 복구하지 않는다.

## 3. 조정 범위 — 방법 변경이 아니라 숫자 조정

| 항목 | 기준 | 이번 값 | 담당 |
|---|---:|---|---|
| **직접 edge λE** | **1e-3** | **3e-4 / 1e-3 / 3e-3** | s2·s3 |
| **A/U LR 비율 rA** | **.01** | **.003 / .01 / .03** | s1·s2·s3 |
| 추가 GT hard α | 1 | .5 / 1 / 1.5 | s4 |
| Output soft β | .1 | .05 / .1 / .2 | s4 |
| U peak LR | 1e-4 | 1e-4 / 7e-5 / 5e-5 | s5 |
| qref·q 함수 | 위 고정값/유리함수 | 탐색하지 않음 | 공통 |
| Student seed | 별도 초기값 | 1234,3407,777,2026,4321,9091,1103 | 서버별 대응 |

rA=.003은 기존 .01보다 작은 적응 범위를 보려는 **이번의 낮은-LR 제안**이며, 성능이 확인된 최적값은 아니다. rA=.003/.01/.03에서 U LR1e-4이면 A LR은 **3e-7 / 1e-6 / 3e-6**이다. 주력에 A LR1e-5를 다시 올리는 실험은 넣지 않는다.

s5는 rA=.01을 고정하므로 U LR을 바꾸면 A LR도 **1e-6 / 7e-7 / 5e-7**로 함께 바뀐다. 따라서 s5의 개입은 ‘U만의 LR’이 아니라 **두 모듈의 전체 학습 속도 조정**이다.

**공통 optimizer:** AdamW, WD=.01, batch48, warmup100+기존 50K cosine. β1/β2·Adam epsilon·decay exclusion·cosine 최저 LR·AMP/scaler·clip·accumulation은 같은 성공 run의 resolved config에서 숫자/parameter 목록까지 고정한다. 새 라이브러리 기본값으로 추정하지 않는다. rA는 param-group LR에만 적용한다. AdamW decay가 남아 있으므로 ‘A의 데이터 gradient는 hard-only’와 ‘모든 parameter 변화가 데이터 gradient만으로 발생’은 다른 말이다.

### 3.1 기본은 FRESH50만

68run 모두 A는 T0에서, U는 architecture+seed별 같은 저장 초기 tensor에서 시작한다. 각 profile은 독립 run이다. 앞 profile의 완료 가중치를 다음 profile에 이어 주지 않는다. **사전학습부터 pipeline 전체를 7번 독립 학습한 실험이 아니라, 하나의 T0에 조건부인 Student 초기값 반복**이다.

강한 이전 LFQ/R3/E075 완료 U를 부모로 쓰는 FT20는 이번 기본량에 포함하지 않는다. 새 경로 자체의 비교를 먼저 정리하며, 이전 부모의 정합·optimizer 이력과 현재 method의 성과를 섞지 않는다. 추가 tail은 §11의 선택적 확장에만 정의한다.

## 4. Profile registry

아래 ID는 **신규 계획용 명세**다. 현재 trainer에 해당 이름이 이미 등록되었다는 뜻이 아니다. 모든 값을 resolved manifest에 내리고, 기존 registry의 금지 조합을 무시하는 우회 플래그로 실행하지 않는다.

### 4.1 G: 극소 edge × 작은 aligner LR

모두 α=1, β=.1, U peak LR1e-4, A hard에 실제 연속 q, U edge에도 같은 실제 연속 q다. **G22가 공통 기준**이다.

| Profile | λE (절대값) | rA | A peak LR |
| --- | --- | --- | --- |
| G11 | 3e-04 | 0.003 | 3e-07 |
| G12 | 3e-04 | 0.01 | 1e-06 |
| G13 | 3e-04 | 0.03 | 3e-06 |
| G21 | 1e-03 | 0.003 | 3e-07 |
| G22 | 1e-03 | 0.01 | 1e-06 |
| G23 | 1e-03 | 0.03 | 3e-06 |
| G31 | 3e-03 | 0.003 | 3e-07 |
| G32 | 3e-03 | 0.01 | 1e-06 |
| G33 | 3e-03 | 0.03 | 3e-06 |

### 4.2 A/E 대조: 주력 method를 검증하기 위한 최소 예외

다음은 G22의 λE=1e-3, α=1, β=.1, U LR1e-4를 유지한다. Trainable A의 rA는 .01이다. Uniform/shuffle/frozen은 **채택 method의 다른 본체가 아니라 그 효용을 분리하는 대조**다.

| Profile | A가 받는 hard 가중 | U의 edge 가중 | 목적 |
| --- | --- | --- | --- |
| G22 | 실제 w(q) | 실제 w(q) | 확정 주력 기준 |
| A_UNIF | 1 | 실제 w(q) | 추가 적응은 같고 A의 q 배분만 제외 |
| A_FREEZE | 업데이트 없음 | 실제 w(q) | T0 A를 처음부터 동결한 대응 |
| A_SHUF | w(q)의 조건부 permutation | 실제 w(q) | A에서 q와 sample 연결의 추가 가치 |
| E_UNIF | 실제 w(q) | 1 | edge에서 q의 추가 가치 |
| E_SHUF | 실제 w(q) | w(q)의 조건부 permutation | edge에서 q와 sample 연결의 추가 가치 |
| ALL_UNIF | 1 | 1 | q가 없는 동일 loss 종류·직접 λE 기준 |

Frozen에서도 Student U는 학습한다. A의 `requires_grad=False`, optimizer group 제외, grad=None을 확인해 weight-decay 이동도 없게 한다. 나머지 모든 trainable-A 대조 역시 **offset 없음·hard-only A**다. 예전 F0/JQ/QER와 동일 case가 아니다.

**중요: shuffle은 raw q 또는 연속 w 값을 섞는다.** 기존 binary `gate_shuffle`을 재사용해 0/1 weight로 만들지 않는다. 기존 Teacher e의 10분위 × 실제 augmentation state별로 sample을 정렬하고, 고정 permutation51515를 한 번 만든 후 모든 case/seed에서 저장해 재사용한다. A_SHUF는 A만, E_SHUF는 edge만 바꾼다. qref는 같은 값이다. Shuffle은 각 stratum의 연속 가중 분포를 보존하나 sample 오차와의 연결을 바꾼다. Degenerate stratum·permutation hash도 기록한다.

Uniform=1이 실제 q의 평균 강도와 정확히 같지는 않다. 평균 w·가중 H/E·gradient를 같이 남긴다. q 고유 효용을 주장할 때는 uniform 비교만이 아니라 분포가 같은 shuffle도 함께 본다. 필요 시 전체 train w 평균을 고정한 상수 대조를 추가할 수 있으나, batch별 재정규화를 주력 식에 추가하지 않는다.

### 4.3 H: e 기반 hard/soft 강도

λE=1e-3, rA=.01, U LR1e-4, A/U edge의 q는 실제 연속값으로 고정한다. α는 U와 A의 H_i에 함께 적용한다. β는 U의 K_i에만 직접 적용한다.

| Profile | α | β | 지위 |
| --- | --- | --- | --- |
| H11 | 0.5 | 0.05 | 튜닝 |
| H12 | 0.5 | 0.1 | 튜닝 |
| H13 | 0.5 | 0.2 | 튜닝 |
| H21 | 1 | 0.05 | 튜닝 |
| H22 | 1 | 0.1 | G22와 동일 |
| H23 | 1 | 0.2 | 튜닝 |
| H31 | 1.5 | 0.05 | 튜닝 |
| H32 | 1.5 | 0.1 | 튜닝 |
| H33 | 1.5 | 0.2 | 튜닝 |
| H_ALPHA0 | 0 | .1 | 추가 hard 재가중 제거 대조; GT hard는 1 |
| H_BETA0 | 1 | 0 | 직접 soft 제거 대조; 최종 KD 주력으로 포장하지 않음 |

### 4.4 L: 전체 업데이트 속도

λE=1e-3, rA=.01, α=1, β=.1, 실제 q 함수·두 모듈 경로는 동일하다.

| Profile | U peak LR | A peak LR | 학습량 |
| --- | --- | --- | --- |
| L100 | 1e-04 | 1e-06 | 50K |
| L070 | 7e-05 | 7e-07 | 50K |
| L050 | 5e-05 | 5e-07 | 50K |

**G22 = H22 = L100**은 같은 수학적 설정이다. 서버/큐 식별을 위한 별칭이며 같은 server+seed에서 중복 편성하지 않았다. 같은 설정의 다른 서버 반복은 host bridge로 기록한다.

## 5. 서버별 실행 큐 — 순서와 seed를 사전 고정

모든 표의 ID는 §4에서 정의한 정확한 profile이다. 모든 신규 run은 50K, batch48이며 같은 서버에서는 GPU 한 개에 직렬로 수행한다. 초기 구현 검사는 s1에서 하되 검증 후에는 s1도 학습에 참여한다.

### 5.1 s1 — Aligner 추가 적응·q 배분의 필요성, 작은 A LR / 12run

**질문:** Teacher A를 그대로 두는 것보다 미세조정이 필요한가? 필요하다면 q 가중이 uniform GT gradient보다 유리한가? q를 쓴 상태에서 A LR을 얼마나 작게 둘 것인가?

두 seed에서 G22/A_UNIF/A_FREEZE를 먼저 만들고 작은 LR과 shuffle을 확인한다. 방법은 이미 q-recon으로 고정했으므로 frozen을 전체 campaign의 기본으로 되돌리는 것이 아니다. **G22−A_UNIF**가 A용 q 배분의 직접 비교, **A_UNIF−A_FREEZE**가 동일 U loss 아래 추가 적응의 비교다. G21/G22/G23은 A LR만 다르다.

| 순서 | Profile | Seed | λE | rA | α / β | U LR | 묶음 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | G22 | 1234 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 2 | A_UNIF | 1234 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 3 | A_FREEZE | 1234 | 1e-03 | 0 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 4 | G21 | 1234 | 1e-03 | 0.003 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 5 | G23 | 1234 | 1e-03 | 0.03 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 6 | A_SHUF | 1234 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 7 | G22 | 3407 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 8 | A_UNIF | 3407 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 9 | A_FREEZE | 3407 | 1e-03 | 0 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 10 | G21 | 3407 | 1e-03 | 0.003 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 11 | G23 | 3407 | 1e-03 | 0.03 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |
| 12 | A_SHUF | 3407 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | A 전달 기준·작은 LR |

### 5.2 s2 — 극소 edge × A LR 조합 탐색 + edge의 q 대조 / 12run

**질문:** 과거 no-edge R3가 강했던 seed777에서도, 이제는 A에 soft/edge를 직접 주지 않는 상태에서 작고 양수인 edge를 어느 정도 남기는 것이 좋은가?

3×3 전체를 완료한다. 각 λ에서 rA를 고정한 비교와 각 rA에서 λ를 고정한 비교를 사용한다. 마지막 E_UNIF/E_SHUF/ALL_UNIF는 같은 seed의 G22와 비교한다. 기존 s2 no-edge R3(0.9588)는 강한 역사적 참고점이지만, 그 모델은 offset·A gradient 경로가 다르므로 새 G와의 차이를 λ의 단독 효과라고 부르지 않는다.

| 순서 | Profile | Seed | λE | rA | α / β | U LR | 묶음 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | G22 | 777 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | edge×A-LR 3×3 |
| 2 | G12 | 777 | 3e-04 | 0.01 | 1/0.1 | 1e-04 | edge×A-LR 3×3 |
| 3 | G32 | 777 | 3e-03 | 0.01 | 1/0.1 | 1e-04 | edge×A-LR 3×3 |
| 4 | G21 | 777 | 1e-03 | 0.003 | 1/0.1 | 1e-04 | edge×A-LR 3×3 |
| 5 | G23 | 777 | 1e-03 | 0.03 | 1/0.1 | 1e-04 | edge×A-LR 3×3 |
| 6 | G11 | 777 | 3e-04 | 0.003 | 1/0.1 | 1e-04 | edge×A-LR 3×3 |
| 7 | G13 | 777 | 3e-04 | 0.03 | 1/0.1 | 1e-04 | edge×A-LR 3×3 |
| 8 | G31 | 777 | 3e-03 | 0.003 | 1/0.1 | 1e-04 | edge×A-LR 3×3 |
| 9 | G33 | 777 | 3e-03 | 0.03 | 1/0.1 | 1e-04 | edge×A-LR 3×3 |
| 10 | E_UNIF | 777 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | q-edge·전체 q 대조 |
| 11 | E_SHUF | 777 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | q-edge·전체 q 대조 |
| 12 | ALL_UNIF | 777 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | q-edge·전체 q 대조 |

### 5.3 s3 — s2의 3×3 조합을 두 다른 seed에서 반복 / 18run

**질문:** s2가 탐색하는 λ×rA 조합의 순위와 raw/RR 절충이 seed2026·4321에서도 유지되는가?

s2 seed777에 더해 G11–G33 각각을 세 Student 초기값에서 얻는다. S2026/S4321을 모두 같은 release·동일 T0/cue로 실행한다. 각 seed 안에서는 같은 초기값과 데이터 순서를 사용한다. 서버 간 절대 최고값 대신 각 seed/서버 내부의 G22 대비 차이를 먼저 읽는다. 이전 binary QE50의 seed4321 음성 결과를 새 연속 q 방법의 실패라고 가져오지 않는다.

| 순서 | Profile | Seed | λE | rA | α / β | U LR | 묶음 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | G22 | 2026 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 2 | G12 | 2026 | 3e-04 | 0.01 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 3 | G32 | 2026 | 3e-03 | 0.01 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 4 | G21 | 2026 | 1e-03 | 0.003 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 5 | G23 | 2026 | 1e-03 | 0.03 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 6 | G11 | 2026 | 3e-04 | 0.003 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 7 | G13 | 2026 | 3e-04 | 0.03 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 8 | G31 | 2026 | 3e-03 | 0.003 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 9 | G33 | 2026 | 3e-03 | 0.03 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 10 | G22 | 4321 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 11 | G12 | 4321 | 3e-04 | 0.01 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 12 | G32 | 4321 | 3e-03 | 0.01 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 13 | G21 | 4321 | 1e-03 | 0.003 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 14 | G23 | 4321 | 1e-03 | 0.03 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 15 | G11 | 4321 | 3e-04 | 0.003 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 16 | G13 | 4321 | 3e-04 | 0.03 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 17 | G31 | 4321 | 3e-03 | 0.003 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |
| 18 | G33 | 4321 | 3e-03 | 0.03 | 1/0.1 | 1e-04 | edge×A-LR 대응 반복 |

### 5.4 s4 — e 기반 hard/soft 계수와 RR 절충 / 14run

**질문:** 고오차 GT 추가 강조 α와 Student 대비 우위로 제한되는 Teacher soft β를 조정하면 raw 하한을 지키면서 RR SCC/ERGAS/PSNR을 개선할 수 있는가?

S1234에서 3×3과 성분 대조를 수행하고 S3407에서 H22/H12/H11을 사전 고정 반복한다. H12(.5,.1), H11(.5,.05)는 강한 고오차 강조를 완화한 조건의 반복이며 승자라는 전제가 아니다. 다른 H가 더 좋으면 현재 단일-seed screening 지위를 남긴 뒤 동일 설정을 다른 seed에 추가한다. α와 β를 동시에 바꾼 비교로 한 항의 독립 효과를 주장하지 않는다.

| 순서 | Profile | Seed | λE | rA | α / β | U LR | 묶음 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | H22 | 1234 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | e 기반 α×β 3×3 |
| 2 | H12 | 1234 | 1e-03 | 0.01 | 0.5/0.1 | 1e-04 | e 기반 α×β 3×3 |
| 3 | H21 | 1234 | 1e-03 | 0.01 | 1/0.05 | 1e-04 | e 기반 α×β 3×3 |
| 4 | H11 | 1234 | 1e-03 | 0.01 | 0.5/0.05 | 1e-04 | e 기반 α×β 3×3 |
| 5 | H23 | 1234 | 1e-03 | 0.01 | 1/0.2 | 1e-04 | e 기반 α×β 3×3 |
| 6 | H32 | 1234 | 1e-03 | 0.01 | 1.5/0.1 | 1e-04 | e 기반 α×β 3×3 |
| 7 | H13 | 1234 | 1e-03 | 0.01 | 0.5/0.2 | 1e-04 | e 기반 α×β 3×3 |
| 8 | H31 | 1234 | 1e-03 | 0.01 | 1.5/0.05 | 1e-04 | e 기반 α×β 3×3 |
| 9 | H33 | 1234 | 1e-03 | 0.01 | 1.5/0.2 | 1e-04 | e 기반 α×β 3×3 |
| 10 | H_ALPHA0 | 1234 | 1e-03 | 0.01 | 0/0.1 | 1e-04 | e 재가중·soft의 성분 기여 |
| 11 | H_BETA0 | 1234 | 1e-03 | 0.01 | 1/0 | 1e-04 | e 재가중·soft의 성분 기여 |
| 12 | H22 | 3407 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | 완만한 e 감독의 사전 고정 반복 |
| 13 | H12 | 3407 | 1e-03 | 0.01 | 0.5/0.1 | 1e-04 | 완만한 e 감독의 사전 고정 반복 |
| 14 | H11 | 3407 | 1e-03 | 0.01 | 0.5/0.05 | 1e-04 | 완만한 e 감독의 사전 고정 반복 |

### 5.5 s5 — 같은 방법의 학습 속도 축소와 4-seed 반복 / 12run

**질문:** 같은 작은 q-recon/edge 방법을 더 느린 전체 LR로 처음부터 학습하면 H–RR 절충과 seed 안정성이 좋아지는가?

각 seed에서 L100→L070→L050의 대응을 만든다. 이는 완료 U의 추가학습이 아니라 fresh50K의 peak LR 비교다. 네 개 고유 Student seed로 반복하되, S2026/S777은 다른 서버와 host bridge도 형성한다. rA를 고정해 U/A가 함께 느려지므로 효과를 ‘U LR만의 효과’로 해석하지 않는다. q/α/β/λ/schedule horizon은 고정한다.

| 순서 | Profile | Seed | λE | rA | α / β | U LR | 묶음 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | L100 | 2026 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | 전체 업데이트 속도·seed |
| 2 | L070 | 2026 | 1e-03 | 0.01 | 1/0.1 | 7e-05 | 전체 업데이트 속도·seed |
| 3 | L050 | 2026 | 1e-03 | 0.01 | 1/0.1 | 5e-05 | 전체 업데이트 속도·seed |
| 4 | L100 | 777 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | 전체 업데이트 속도·seed |
| 5 | L070 | 777 | 1e-03 | 0.01 | 1/0.1 | 7e-05 | 전체 업데이트 속도·seed |
| 6 | L050 | 777 | 1e-03 | 0.01 | 1/0.1 | 5e-05 | 전체 업데이트 속도·seed |
| 7 | L100 | 9091 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | 전체 업데이트 속도·seed |
| 8 | L070 | 9091 | 1e-03 | 0.01 | 1/0.1 | 7e-05 | 전체 업데이트 속도·seed |
| 9 | L050 | 9091 | 1e-03 | 0.01 | 1/0.1 | 5e-05 | 전체 업데이트 속도·seed |
| 10 | L100 | 1103 | 1e-03 | 0.01 | 1/0.1 | 1e-04 | 전체 업데이트 속도·seed |
| 11 | L070 | 1103 | 1e-03 | 0.01 | 1/0.1 | 7e-05 | 전체 업데이트 속도·seed |
| 12 | L050 | 1103 | 1e-03 | 0.01 | 1/0.1 | 5e-05 | 전체 업데이트 속도·seed |

## 6. 학습·구현 계약

### 6.1 모듈별 gradient 계산의 의미 명세

다음은 **실행 코드가 아니라 구현 계약의 의사코드**다.

```text
Teacher forward: frozen A_T + frozen U_T, no_grad
Student forward: cloned A_S + U_S, native inputs, exactly once
raw q cache -> fixed w_i
H_i, K_i, E_i -> per-sample tensors; no sample reduction before w_i
L_U = mean(H + K + lambda_E * w_edge * E)
L_A = mean(w_align * H)  # w_align=q / 1 / shuffled-q by control
G_U = partial_gradient(L_U, U_parameters)
G_A = partial_gradient(L_A, A_parameters)  # through U input/sampler
assign G_U only to U; G_A only to A
one optimizer update of both parameter groups after both gradients are ready
```

같은 forward graph에서 parameter 집합을 지정한 `autograd.grad` 등으로 구현할 수 있다. A gradient를 얻을 때 U의 input Jacobian은 통과하지만 U parameter에 A loss의 gradient를 누적하면 안 된다. U를 완전히 `no_grad`로 감싸면 A까지 끊기므로 금지한다. `create_graph=False`의 1차 미분이면 충분하며 고차 gradient를 만들 필요가 없다.

단일 scalar를 일반 backward하는 기존 trainer와 같지 않다. 합쳐진 gradient를 나중에 일부 빼는 구현은 cancellation/AMP 경로까지 검증해야 하므로, 초기 실행은 명시적인 두 parameter 집합의 gradient 추출을 기준으로 한다. 더 효율적인 구현을 쓰면 이 reference gradient와 동치여야 한다.

두 gradient 계산 사이에 optimizer step, 재forward, 다른 dropout/RNG view를 넣지 않는다. A/U gradient를 모두 얻은 뒤 업데이트한다. Shared parameter가 없어야 하고, 같은 tensor가 두 optimizer group에 들어가면 실패로 처리한다. AMP를 쓰면 같은 loss scaling 규약·unscale·finite 검사 후 두 group이 함께 step/skip하도록 한다. 이전 일반 `.backward()`로 쌓인 A grad를 남기지 않는다.

### 6.2 이번 설정의 invariant

1. Teacher A/U hash와 q/e/cue는 step 전후 동일하다. `.grad`는 없다.
2. q는 A용 H와 U용 E에 각각 한 번만 곱한다. U의 H/K에는 곱하지 않는다.
3. A의 직접 gradient는 weighted H에서만 온다. 고정된 같은 Student 출력에서 β/λE를 바꾸어도 G_A는 변하지 않는다. α를 바꾸면 G_A는 바뀔 수 있다.
4. Native M·GT·출력은 warp하지 않는다. HRMS output을 GT에 사후 정합해 loss를 낮추지 않는다.
5. Student offset auxiliary를 아예 실행하지 않는다. 0을 곱하면서 jitter+forward 비용과 RNG를 남기는 구현을 기본으로 하지 않는다. 새 캠페인 안에서 paired data/augmentation RNG를 일치시키며, 과거 I-AEQ의 궤적과 같다고 주장하지 않는다.
6. q weight는 [B]에서 명시적으로 broadcast한다. edge/hard의 sample별 평균을 먼저 확보하고 마지막에 batch 평균한다. 모든 주력 λE>0이다.
7. Frozen control은 gradient만 0이 아니라 parameter·buffer가 불변이어야 한다.
8. Tiny edge 계산과 gradient는 FP32 경로에서 검증한다. 문서상의 1e-3가 실제 loss에서 다른 배율과 곱해져 거의 0이 되지 않게 한다.

### 6.3 기동 전 검사 — 성능 탐색 전에 한 번만

- CPU/FP32 작은 tensor에서 두 목적함수의 reference gradient와 실제 trainer gradient를 대조한다. 신호가 존재하는 parameter에 relative error ≤1e-5, 작은 값의 absolute tolerance는 기존 수치 정책에 명시한다. 모든 float/GPU 환경의 bitwise 동치를 강제하는 규칙은 아니다.
- 동일 forward에서 A-only loss의 U `.grad` 누적이 없고, U loss의 A `.grad`가 섞이지 않음을 검사한다. q/GT/Teacher의 detach 및 H의 live gradient를 각각 검사한다.
- β와 λE를 인위적으로 바꾼 같은-step 테스트에서 A data gradient가 그대로인지 확인한다. G22와 H22/L100 동치도 검사한다.
- w(qref)=1, q=0에서2, q>=0에서유한, shuffle stratum별 value multiset 보존, negative/NaN q fail-fast, qref>0을 검사한다.
- λE=3e-4/1e-3/3e-3의 실제 loss 선형 배율 및 A edge-gradient 부재를 검사한다. 중앙값을 hard gate로 쓰지 않는지 확인한다.
- Fresh A와 T0 A, 동일 seed의 U 초기 tensor hash 일치. A LR group이 warmup와 cosine 중 정확한 rA를 유지하는지 확인한다.
- 200–500update 실제 학습 smoke로 G22/A_UNIF/A_FREEZE/E_UNIF 및 α/β 경계 profile을 확인한다. Smoke는 별도 폴더이고 정식 50K 결과·24시간 학습량으로 세지 않는다.
- interruption 이후 해당 run의 optimizer/scheduler/scaler/data RNG를 정확히 resume한다. 다른 profile의 state를 가져오지 않는다.

### 6.4 현재 코드와 새 구현의 경계

이번에 조회한 main은 `5eeeb3de952772dc7f443319550d0eca1bf54d89`로 EDGEBAL의 scalar/affine edge·시간배분 구현이다. [REPO] 이 사실로 **연속 rational-q + weighted-hard-only A의 조합이 이미 검증됐다고 간주하지 않는다.**

기존 `kdv/edge_gate.py`의 raw q cache·per-sample edge, `kdv/losses_rec.py`의 hard/soft map, `kdv/registry.py`의 validation을 재사용할 수 있다. 새 전용 method spec을 등록하고 기존 `edge_gate/edge_weight/edge_route/routing`을 여러 개 조합해 검사만 끄는 방식으로 구현하지 않는다. Student input은 native transfer이며 I-AEQ가 요구하는 양의 offset을 억지로 채우지 않는다.

새 schema의 의도는 다음과 같다. 키 이름은 구현자가 기존 schema와 매핑할 **의미 명세**이며 바로 실행 가능한 config가 아니다.

```yaml
method: qrecon_continuous_v1
student_arch: W104_D121
teacher: T0
init: clone_T0_A_and_fresh_U
student_offset_aux: disabled
student_pan_jitter: disabled
aligner_update: weighted_gt_hard_only
q_source: frozen_T0_cache_raw_q
q_weight: twice_median_over_median_plus_q
q_ref: 0.3276133416220546
reconstruction:
  alpha: 1.0
  beta: 0.1
  tau_R: 0.012463942170143127
  eps: 1.0e-6
edge:
  lambda_absolute: 1.0e-3
  target: GT
  U_weight: continuous_q
  A_gradient: none
optimizer:
  U_peak_lr: 1.0e-4
  A_lr_ratio: 0.01
  updates: 50000
  batch: 48
  warmup_updates: 100
  scheduler: inherit_validated_fresh50_cosine
```

최소한 optimizer betas/eps, cosine minimum, decay 제외, AMP/scaler/TF32, augmentation 구현·확률, dataset/evaluator/Teacher/cue full hash를 실행 manifest에 잠근다. 과거 문서에 없는 숫자는 새 기본값으로 조용히 채우지 않는다. [ALIGN §4.4]

## 7. 목표 selector와 결과 해석

### 7.1 목표는 바꾸지 않는다

| 순서 | 지표 | 목표 |
|---|---|---:|
| 1차 하한 | **raw-original HQNR** | **≥0.9585** |
| RR 1 | **SCC** | **>0.988** |
| RR 2 | **ERGAS** | **<2.040** |
| RR 3 | **PSNR** | **>37.956** |
| RR 4 | SAM | <2.787 |
| RR 5 | Q8 | >0.922 |
| RR 6 | SSIM | >0.976 |

Reported RR 기준은 PAN-Crafter Table1, WV3, p.6의 표시값이다. [PAPER] 표시 숫자를 넘는 `numeric_pass`와 원 논문 정밀도에서의 `display_tie`를 구분한다. 출판된 미반올림 값·seed 불확실성을 우리가 확보한 것은 아니므로 통계적 우월성을 자동 주장하지 않는다.

### 7.2 같은 checkpoint에서 선택한다

기존 `best_hqnr`와 raw-max 기록은 보존한다. QRECON24의 별도 `HQNR9585_RR_v1`는 공식 raw H 하한 통과 후보 안에서 **SCC 큰 순 → ERGAS 작은 순 → PSNR 큰 순 → SAM 작은 순 → Q8 큰 순 → SSIM 큰 순 → H 큰 순**으로 정렬한다. 여섯 RR를 모두 넘는 후보가 있으면 그 집합의 최상도 별도로 인계한다. [TARGET]

- H=0.9584999는 미달, H=0.9585는 통과다. 1e-4 legacy tie band로 하한을 완화하지 않는다.
- 적격 후보가 없으면 `target_feasible=false`와 legacy best를 함께 기록한다. RR만 좋은 모델을 목표 성공으로 올리지 않는다.
- FR는 원 PAN/MS 전체 frame의 mat20, 장면별 `(1-Dλ_i)(1-Ds_i)`의 평균이다. 평균 Dλ·Ds의 곱으로 대체하지 않는다.
- **RR SCC는 GT 기준이며 FR fSCC가 아니다.** Q8은 8밴드 전체이고 Q4_first4가 아니다. 학습 proxy와 공식 evaluator 값을 합쳐 하나의 결과로 쓰지 않는다.
- FR/RR 필터·padding·border·intensity·aggregation·clipping을 바꿔 성능을 올리지 않는다. 목표를 이유로 SCC loss·PAN reconstruction·새 feature loss를 추가하지 않는다.

### 7.3 후보 격자·반복

50K의 기존 `GRID1010_50K_v1`: 1,010…49,490 및 exact50,000을 보존한다. **원 selected / target selected / exact50K / 고정 후반6점**을 별도로 기록한다. 후반6점은 45,450·46,460·47,470·48,480·49,490·50,000이다. s2 보고서의 ≥30K plateau와 섞지 않는다.

새 target selector의 공식 RR 값이 학습 중 후보마다 없으면, checkpoint를 먼저 보존하고 공식 FR 적격 후보의 RR를 보완한다. Proxy H만으로 checkpoint를 버려 공식 적격 후보를 놓치지 않는다. 필요한 추가 공식 RR 비용은 별도 측정한다.

**Asset board와 Method board를 분리한다.** 가장 높은 seed의 자산은 실용 목표에 사용할 수 있지만, method는 profile별 지정된 모든 seed의 평균·표준편차·하한 통과 개수로 평가한다. 성공 seed만 평균하지 않는다. 같은 T0를 쓰는 조건부 반복이며, 같은 seed의 여러 서버 실행은 host 반복이지 독립 초기값이 아니다. 과거 0.0031은 현재 W104·새 routing의 p-value나 무시해도 되는 차이의 자동 기준이 아니다.

### 7.4 각 비교에서 할 수 있는 주장

| 비교 | 확인하는 것 | 확대하면 안 되는 해석 |
|---|---|---|
| G22 − A_UNIF | 같은 U q-edge 아래 A용 q 가중의 가치 | q가 절대 정합 정확도를 안다는 주장 |
| A_UNIF − A_FREEZE | 이 새 loss에서 추가 A 적응의 가치 | 모든 backbone에서 동결이 나쁘다는 주장 |
| G22 − A_SHUF | A에서 실제 q–sample 연결의 가치 | 단일 seed 차이를 일반화 |
| G22 − E_UNIF/E_SHUF | 실제 q-edge 가중의 가치 | edge 종류/총 계수까지 달라진 old JQ와의 단독 인과 귀속 |
| G22 − ALL_UNIF | 두 q 적용 경로의 합친 차이 | A와 edge 각각의 순증분을 단순 합산 |
| H22 − H_BETA0 | 이 조건에서 output soft의 추가 기여 | H/e/q 모든 효과를 soft에 귀속 |
| H22 − H_ALPHA0 | GT 추가 재가중의 U/A 결합 효과 | pixel hard 가중만의 U 단독 효과 |

q의 변화폭이 작으면 결과 차이도 작을 수 있다. 그때 q를 더 복잡하게 증폭하지 말고 uniform/shuffle 대비 관측된 범위대로 보고한다. 새로운 신호가 없으면 방법 정의는 유지한 실험 결과를 정직하게 남기고, 성능 목표 자산과 주장 범위를 분리한다.

## 8. 시간 산정 — 기존 실측과 신규 예측을 분리

### 8.1 이번 Sheet의 실제 시간

| 서버 | 관측된 관련 50K 시간 | 편성 대용 R_s | 선택 이유 |
|---|---|---:|---|
| s1 | JQ 2.20h; QE50/QES 2.15h; QEC 2.13h | **2.20h** | 현재 W104 KD 기록 상단 |
| s2 | JQ 2.30h; QE50 .25/.5/2배 2.29h; no-edge 2.26h | **2.30h** | 같은 서버의 현재 KD 시간 |
| s3 | W104 KD 1.32–1.35h; 새 S4321도 1.33–1.34h | **1.35h** | 현재 W104 KD 시간 상단 |
| s4 | 기존 KD 1.35–1.54h; routing 1.53–1.54h, **최근 JE0 S3407 1.94h** | **1.94h** | 최근 느린 routing 기록도 포함해 보수적으로 편성 |
| s5 | JQ 1.55–1.57h; E075 1.66h; UP 1.86h; **FLOOR 2.17h** | **2.17h** | 최근 q-weighted run 지연을 반영 |

s4의 1.94h나 s5의 2.17h가 새 method의 필연적 처리시간이라는 뜻은 아니다. 서버 경합·구현 차이·후처리 포함 여부의 원인은 Sheet만으로 확정하지 않았다. 이전 1.38/1.66h만 사용해 최신 지연을 무시하지 않았다. 1.94h는 1시간56분24초, 2.17h는 2시간10분12초다. [SHEET]

### 8.2 새 gradient 분리는 아직 실측이 없다

Student offset 보조 forward가 없어져 빨라질 수 있지만, hard-only A와 U의 gradient를 별도로 계산하는 비용은 늘 수 있다. **둘이 상쇄된다고 가정하지 않는다.** 기존 routing 시간도 정확히 같은 알고리즘은 아니므로 new runtime으로 둔갑시키지 않는다.

초기 가예약은 아래 하나의 식만 사용한다.

\[
T_{\mathrm{예약},s}=1.20\,N_sR_s+N_s\times10/60\quad\mathrm{hours}.
\]

**20%는 새 경로와 처리량 불확실성에 대한 임시 여유**이지 실측 overhead나 상한이 아니다. run당10분은 기본 export/gspread/전환 가예약이다. 이전 10% 여유를 또 곱하지 않는다. 기존 Train(h)에 들어 있는 정기 평가 50회도 다시 더하지 않는다. 새 공식 RR 보완·source/cue 검증·초기 구현·기존 작업 잔여시간은 별도다.

| 서버 | Run 수 | 산식: N×R | 학습 대용값 합 | 20%+run당10분 예약 |
| --- | --- | --- | --- | --- |
| s1 | 12 | 12×2.20 | 26.40h | 33시간 40분 48초 |
| s2 | 12 | 12×2.30 | 27.60h | 35시간 07분 12초 |
| s3 | 18 | 18×1.35 | 24.30h | 32시간 09분 36초 |
| s4 | 14 | 14×1.94 | 27.16h | 34시간 55분 31초 |
| s5 | 12 | 12×2.17 | 26.04h | 33시간 14분 53초 |

**합산 학습 대용값:** 131.50 서버·시간. **합산 가예약:** 169.133333 서버·시간. 다섯 슬롯이 동시에 준비되어 독립 실행하면 **병렬 경과시간 가예약은 합계가 아니라 최대값인 35시간 07분 12초**다. 사전 구현·추가 공식 RR·기존 실행 잔여시간은 이 시간 밖이다.

### 8.3 실제 24시간을 채우는 규칙

각 서버의 첫 G22 및 대응 control을 완주하면 `new_train_h`, `extra_official_rr_h`, `postprocess_h`, `setup_h`, `wait_h`를 별도 기록한다. 5K까지의 처리율과 최소 두 정기평가가 있으면 잠정 ETA를 갱신하고, 첫 full50K가 끝나면 그 실제 값으로 R_s/신규 overhead 대용을 교체한다.

- 12시간/24시간은 중간 확인 시점이다. 정상 진행 중 run을 24h에 중단하지 않는다.
- 기본 큐를 모두 마치고 실제 **신규 완료 run의 Train(h) 합 <24h**이면 아래 사전 지정 3run 확인 묶음을 추가한다. 반복하려고 같은 run을 이름만 바꿔 재실행하거나 GPU를 idle로 놓지 않는다.
- 기존 R_s가 보수적이라 실제가 빨랐으면 위 확장이 필요할 수 있다. 특히 s4가 예전 1.54h 수준으로 돌아가면 14run=21.56h이므로 이를 명시적으로 보완한다.
- 새 경로가 예상보다 느리면 완료 예상시간을 늘린다. 같은 50K 방법 비교를 20K로 잘라 시간을 맞추지 않는다.

| 서버 | 최소시간 미달 시 첫 추가 seed | 추가 3run |
|---|---:|---|
| s1 | 9091 | G22 / A_UNIF / A_FREEZE |
| s2 | 3407 | G12 / G22 / G32 |
| s3 | 1103 | G21 / G22 / G23 |
| s4 | 2026 | H22 / H12 / H11 |
| s5 | 2909 | L100 / L070 / L050 |

이미 같은 server+profile+seed+release의 유효 결과가 있으면 중복 실행하지 않고 다음 사전 reserve seed17041(그것도 있으면26017)로 같은 묶음을 수행한다. 선호 성능을 보고 seed 번호를 바꾸지 않는다. 추가량은 기본68run 밖으로 집계한다.

## 9. 실행 파일·Git·gspread 인계

### 9.1 새 ID와 old pending 분리

```text
PAKD50_QRC24_<SERVER>_<PROFILE>_W104_D121_WV3_T0_S<seed>_FRESH50_v1
```

예: `PAKD50_QRC24_S3_G22_W104_D121_WV3_T0_S2026_FRESH50_v1`.

서버 토큰은 파일 충돌을 막기 위한 것이다. `canonical_profile`에 수학적 정의를 기록하고 H22/L100은 G22와 같은 것으로 집계한다. 기존 PAKD50_JQ/QE50/QETUNE24 이름을 재사용하지 않는다.

사용자는 QETUNE24가 아직 기동되지 않았다고 확인했다. **그 미실행 86run은 superseded로 보류**하며 새68run과 합쳐154run으로 돌리지 않는다. 예전 QEDGE9/QEGX/EDGEBAL의 진행 중 run은 정상종료하고, 다음 큐를 교체한다. 실제 프로세스 상태를 확인하지 않고 Sheet 완료 행만 보고 idle로 판단하지 않는다.

### 9.2 릴리스 고정

s1에서 새 mode의 정확한 gradient 검사를 끝내고 release SHA를 고정한다. s2–s5는 동일 SHA의 별도 worktree를 사용한다. 실행 중 코드에 pull하지 않는다. 진행 중 결함 수정은 새 release/영향받은 대조군으로 구분한다. Frozen와 route별 off 규약, scaler 정책이 달라지면 같은 v1로 덮어쓰지 않는다.

T0/cue와 U 초기값은 checksum으로 전달한다. A의 clone 초기값은 모든 fresh case에서 동일한 T0 hash다. Git config와 대용량 weight 전달은 구분한다. 이 문서는 원격 실행 명령이나 현재 경로를 임의로 만들어 ‘바로 실행 가능’하다고 주장하지 않는다.

### 9.3 기존 업로드 구조 유지

`tools/_upload.sh → gspread/gspread_upload.py → WV3-s1…s5`를 유지한다. 기존 B:W와 X열 `통합실험`을 사용한다. 새 worksheet나 직접 Sheet 편집은 필요 없다.

X 예: `PAKD50 / QRC24 / G22 / A104D121 / FRESH50`.

Notes/manifest에는 적어도 아래를 기록한다.

- `method=qrecon_continuous_v1; A_loss=weighted_H_only; A_soft=0; A_edge=0; student_offset=0`.
- λE **절대값**, rA, U/A LR, α/β, qref, A/E weight mode, raw-q/permutation full hash.
- init_seed, architecture, T0/initU/data/evaluator/release SHA, actual50K.
- selector/feasible 여부, selected step/hash, legacy/raw-max/target/last의 연결.
- 실제 train_h/추가 공식 RR/후처리 시간.

신규 run의 대표 행은 target selector가 지원되면 같은 checkpoint FR/RR를 그 경로에서 export한다. 적격 후보가 없으면 legacy-best를 내보내되 실패 flag를 숨기지 않는다. 기존 결과 행의 metric은 덮어쓰지 않는다. 업로더가 target selector를 아직 지원하지 않으면 local 공식 결과를 먼저 보존하고 ‘지원완료’로 가장하지 않는다.

## 10. 24시간 관측 후 의사결정

우선순위는 **raw 하한 통과 자산 확보 → 같은 checkpoint의 RR 개선 → 고정 방법의 seed 재현성**이다. 이것은 q/e method를 성능이 확인된 것으로 미리 고르는 규칙이 아니다.

1. 먼저 G22와 직접 대조를 읽는다. q-weighted A가 uniform/frozen보다 반복적으로 유리한지 확인하고 λ×rA/αβ/LR의 숫자 조정을 종합한다. 어떤 계수가 모든 서버에서 같아야 한다는 결론은 아니지만, 최종 method 보고에서는 seed마다 유리한 숫자를 따로 쓰지 않는다.
2. G11–G33의 3seed 결과에서 λ와 rA를 하나씩 선택한다. raw 적격 수·RR 방향·후반 유지와 seed dispersion을 함께 본다. s2 한 seed의 최고 λ를 전체 승자로 부르지 않는다.
3. H 계열의 승자가 H22/H12/H11 밖이면, 우선 그 숫자를 같은 λ/rA 조건의 다른 두 seed에서 확인한다. H 승자와 G 승자의 결합은 자동 합산하지 않는다.
4. λ/rA를 고른 G_WIN, αβ를 고른 H_WIN, 두 값을 합친 COMB를 **같은 미사용 seed17041와26017**에서 비교하는 확인 묶음을 다음 우선 확장으로 둔다. 새 loss/function은 만들지 않는다. 그때의 exact profile을 확인 결과 전에 고정한다.
5. s5의 낮은 LR이 유리하면 COMB와 동일한 loss에서 peak LR만 비교한다. 학습량/schedule 최저값까지 동시에 바꾸지 않는다.

Frozen/constant가 가장 좋다고 주력의 q 성공으로 포장하지 않는다. 사용자가 고정한 q/e method의 관측 성능과 전체 최고 성능 자산을 분리해 인계한다. q가 작은 계수로만 거의 무영향이면 그 크기를 그대로 보고한다.

### 10.1 안전·중단

NaN/Inf, 잘못된 teacher/cue/초기값, 두 objective의 잘못된 gradient 합산, GT frame 위반, optimizer param 누락은 즉시 INVALID로 중단한다. 낮은 성능이라는 이유만으로 후보 하나만 중간 종료하지 않는다. 모든 정상 run은 동일50K를 완주하고 selected와 last를 모두 보존한다.

A의 native shift 분포 및 T0 대비 Δc를 고정 train 진단 batch에서 기록한다. q_T가 고정인 것은 drift를 검출하지 못하므로 q_T 자체의 값이 안정적이라는 이유로 Student A가 안정적이라고 쓰지 않는다. 필요한 Student probe는 **정해진 진단 시점에 no_grad로 측정**할 수 있지만 학습 loss·추가 계수로 되돌리지 않는다.

## 11. 선택적 fine-tuning 확장 — 기본68run/24시간량 밖

이 항목은 처음부터 기존 고성능 모델의 서로 다른 학습 경로를 기본 실험에 섞지 않기 위한 분리다. **이번 method의 완료 run**에서 목표에 가까운 같은 A/U checkpoint를 선택한 뒤에만 실행하는 것을 우선한다.

동일 parent에서 별도로 두 fork를 만든다: U peak LR1e-5 vs3e-6, 각각 **추가10K**, rA=.01, 양의 λE/αβ/연속 q/hard-only A/no-offset은 parent에서 유지한다. 두 arm 모두 새 AdamW, 같은 WD·나머지 optimizer policy, local warmup100+cosine(end=peak의0.1배)을 쓰는 **후속 제안**이다. +0,+1K…+10K를 같은 격자로 보존한다. 이 schedule이 이미 좋은 것으로 검증된 것은 아니다.

부모는 선택된 같은 checkpoint의 A/U를 함께 가져온다. A만 T0로 다시 교체하지 않는다. parent_init_seed와 tail_rng_seed를 별도 기록한다. Tail RNG는 `70000+parent_init_seed`로 사전 결정해 두 fork에서 같게 한다. 실제 총학습량은 parent_step+10000이며 best@20K에 tail10K를 붙였다고 무조건60K라 부르지 않는다. +0이 최상이면 기존 자산 유지이지 미세조정의 개선이 아니다.

기존 LFQ/R3/E075의 부모를 재사용하려면 **OLD_PARENT_ADAPT**로 별도 분류하고 provenance 및 A/U pair를 보존한다. 그 부모들의 준비비용과 이전 method 효과를 새 fresh 결과에 합쳐 같은 반복으로 집계하지 않는다. 추가 tail의 시간은 새 경로50K 실측 후 측정 밀도/고정비를 반영해 재산정한다. 과거 0.4배 FT20 추정치를 이번10K에 그대로 사용하지 않는다.

기존 K20에서 모든 적응 arm이 HQNR을 낮춘 보고가 있으므로, 성능 하락을 숨기지 않고 원본 parent를 보호한다. [ANALYSIS §5, EQREC §B.5]

## 12. 근거·출처와 해석의 경계

이 문서는 **사용자 확정 method → 새로운 튜닝 설계**와 **자료에서 실제 관측한 결과**를 구분했다. 자료의 오래된 추천이 사용자 최신 결정과 충돌하면 그 원문을 바꾸지 않고 새 결정으로 명시했다. 예를 들어 과거 보고서의 plateau 우선·fSCC 우선·q-quality gate 미채택은 당시 범위의 기록이며, 이번 target selector와 새 q-hard A 방법이 이미 시험됐다는 근거가 아니다.

| ID | 출처 | 사용 범위 |
|---|---|---|
| USER-FIX | 직전 사용자 합의 | 연속 q 공유, U=H+K+q-edge / A=q-H only, Student offset 제거 |
| SHEET | `pan-cvpr27`의 이번 live native 조회, 부록 A | 원 selected H/RR와 실제 Train(h), 과거 완료 case |
| TARGET | `PAN_QETUNE24_S1_S5_HQNR9585_RR_Integrated_Plan_2026-09-16.md` §1 | 목표/selector만 계승. old method·86run은 대체 |
| KD | `PAN_KD_Q12_R1_Spec_and_Aligner_Interface_2026-09-14.md` §4–5·§9 | 기존 e/a·hard/soft·GT edge·Teacher fixed path |
| ALIGN | `PAN_Aligner_L1E4_TechnicalSpec_Evidence_KD_Handoff_2026-09-14.md` §2–6 | T0 앞단·PAN sampler·coords·원 학습 경로/단위 |
| S2 | `2026-09-16_s2_qedge9-seed777-cross.md` §2·3·6 | 고정밀 selected/plateau·실제 이진 gate 작동·한계 |
| S3 | `2026-09-15_s3_pakd50-results.md` §2–5 | 기존 W112에서 policy/backend 상호작용·단일seed 제한 |
| ANALYSIS | `PAN_Aligner_Analysis_Summary_2026-09-15.md` §2–5 | q 정의·보정 이득과 추가 적응을 구분 |
| EQREC | `2026-09-15_s1_eqrec4-results.md` §A·§B.5 | q/e 집단과 상관/개입 구분·K20 악화 |
| PAPER | `pancrafter.pdf`, arXiv2505.23367v2, Table1 p.6 | WV3 reported RR 표시값 |
| REPO | main `5eeeb3de952772dc7f443319550d0eca1bf54d89`, 이번 조회 | EDGEBAL 기반과 새 gradient 구현의 경계 |

FR20과 RR benchmark를 반복적으로 보고 선택하는 것은 benchmark 개발이다. 목표 숫자 획득과 untouched 데이터의 일반화 검증을 동일시하지 않는다. 예측되는 목표 달성량을 임의로 +0.001처럼 산출하지 않는다.

### 12.1 조회 provenance

- 문서 생성 시각(KST): `2026-09-16T15:18:01+09:00`.
- Spreadsheet ID: `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`.
- export 파일 SHA256: `e181909cbdbdb0054d85d860bf86d383cbd906ab2d5fe0d34b580192cce1d23c`.
- live 조회 범위: s1 B3:X198 및 U61:U65 / s2 B3:X200 / s3 B90:X200 / s4 B15:X198 / s5 B15:X200; query `W104_D121`.
- export와 이어진 native 조회에는 시차가 있으므로 부록 A의 값은 native 조회를 기준으로 한다. export SHA가 서버 checkpoint/evaluator hash를 대신하지 않는다.
- 각 탭 header3: D=ERGAS, F=PSNR, H=RR SCC, N=raw HQNR, U=Train(h).
- 실제 서버의 마지막 source revision·프로세스 상태·checkpoint step/weight 파일은 여기서 접속 검증하지 않았다. Sheet 완료를 그대로 학습 중/idle 판정으로 쓰지 않는다.

## 부록 A. 이번에 참고한 W104–D121 결과 55개

아래는 **새 QRECON24 결과가 아니라 그 이전 완료 결과**다. H/SCC/ERGAS/PSNR은 Sheet 저장 정밀도대로 표시했다. run ID는 `PAKD50_<Case>_W104_D121_WV3_T0_S<Seed>_FRESH50_<v>`이며, s3 탭 이름은 `WV3-s3(5090)`, 나머지는 `WV3-s1/2/4/5`다. 표의 숫자로 HQNR 경계값 통과를 최종 판정하지 않고 원 JSON을 확인한다. [SHEET]

| 서버:행 | Case | Seed/v | raw H | RR SCC | ERGAS | PSNR | Train(h) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s1:61 | J0 | 1234/v2 | 0.9595 | 0.9771 | 2.6221 | 36.0120 | 1.97 |
| s1:62 | JQ | 1234/v2 | 0.9567 | 0.9874 | 2.0840 | 37.8299 | 2.20 |
| s1:63 | QE50 | 1234/v2 | 0.9574 | 0.9864 | 2.1378 | 37.5936 | 2.15 |
| s1:64 | QES | 1234/v2 | 0.9574 | 0.9873 | 2.0870 | 37.8056 | 2.15 |
| s1:65 | QEC | 1234/v2 | 0.9564 | 0.9872 | 2.0911 | 37.8011 | 2.13 |
| s2:78 | J0 | 777/v2 | 0.9524 | 0.9873 | 2.0866 | 37.8263 | 1.95 |
| s2:79 | JQ | 777/v2 | 0.9577 | 0.9867 | 2.1187 | 37.6896 | 2.30 |
| s2:80 | QE50 | 777/v2 | 0.9577 | 0.9865 | 2.1343 | 37.6292 | 2.29 |
| s2:81 | EB_R3E000 | 777/v1 | 0.9588 | 0.9873 | 2.0857 | 37.8224 | 2.26 |
| s2:82 | EB_R3E050 | 777/v1 | 0.9561 | 0.9876 | 2.0715 | 37.8836 | 2.29 |
| s2:83 | EB_R3E025 | 777/v1 | 0.9548 | 0.9875 | 2.0736 | 37.8595 | 2.29 |
| s2:84 | EB_R3E200 | 777/v1 | 0.9557 | 0.9875 | 2.0747 | 37.8639 | 2.29 |
| s3:95 | J0 | 2026/v2 | 0.9555 | 0.9867 | 2.1125 | 37.7087 | 1.15 |
| s3:96 | JQ | 2026/v2 | 0.9551 | 0.9872 | 2.0933 | 37.7910 | 1.33 |
| s3:97 | QE50 | 2026/v2 | 0.9558 | 0.9873 | 2.0824 | 37.8245 | 1.35 |
| s3:98 | XJ | 2026/v2 | 0.9548 | 0.9873 | 2.0868 | 37.8200 | 1.33 |
| s3:99 | QX50 | 2026/v2 | 0.9556 | 0.9876 | 2.0732 | 37.8707 | 1.34 |
| s3:100 | J_R3_NOEDGE | 2026/v2 | 0.9573 | 0.9869 | 2.1039 | 37.7468 | 1.32 |
| s3:101 | QEC3 | 2026/v2 | 0.9560 | 0.9876 | 2.0739 | 37.8852 | 1.34 |
| s3:102 | QES | 2026/v2 | 0.9554 | 0.9873 | 2.0857 | 37.8131 | 1.35 |
| s3:103 | J_QB005 | 2026/v2 | 0.9549 | 0.9874 | 2.0801 | 37.8453 | 1.34 |
| s3:104 | QE50_B005 | 2026/v2 | 0.9556 | 0.9872 | 2.0929 | 37.7937 | 1.35 |
| s3:105 | J0 | 4321/v2 | 0.9554 | 0.9871 | 2.0925 | 37.7871 | 1.14 |
| s3:106 | JQ | 4321/v2 | 0.9574 | 0.9872 | 2.0968 | 37.7911 | 1.33 |
| s3:107 | XJ | 4321/v2 | 0.9561 | 0.9871 | 2.0947 | 37.7855 | 1.34 |
| s3:108 | QE50 | 4321/v2 | 0.9563 | 0.9865 | 2.1310 | 37.6366 | 1.34 |
| s4:20 | NA0 | 1234/v1 | 0.9580 | 0.9872 | 2.0879 | 37.8170 | 1.27 |
| s4:21 | J0 | 1234/v1 | 0.9585 | 0.9862 | 2.1416 | 37.5879 | 1.17 |
| s4:22 | JQ | 1234/v1 | 0.9565 | 0.9874 | 2.0804 | 37.8386 | 1.38 |
| s4:23 | XJ | 1234/v1 | 0.9571 | 0.9869 | 2.1079 | 37.7301 | 1.37 |
| s4:24 | F0 | 1234/v1 | 0.9574 | 0.9871 | 2.0964 | 37.7872 | 1.13 |
| s4:25 | QE50 | 1234/v1 | 0.9543 | 0.9873 | 2.0838 | 37.8250 | 1.38 |
| s4:26 | LF0 | 1234/v1 | 0.9577 | 0.9870 | 2.0985 | 37.7706 | 1.14 |
| s4:27 | LFQ | 1234/v1 | 0.9586 | 0.9873 | 2.0866 | 37.8175 | 1.35 |
| s4:28 | LFQE50 | 1234/v1 | 0.9571 | 0.9873 | 2.0837 | 37.8323 | 1.35 |
| s4:29 | LFX | 1234/v1 | 0.9589 | 0.9864 | 2.1322 | 37.6252 | 1.35 |
| s4:30 | JE0 | 1234/v1 | 0.9548 | 0.9865 | 2.1323 | 37.6398 | 1.53 |
| s4:31 | QER50 | 1234/v1 | 0.9558 | 0.9874 | 2.0833 | 37.8282 | 1.54 |
| s4:32 | QERS | 1234/v1 | 0.9583 | 0.9873 | 2.0833 | 37.8280 | 1.54 |
| s4:33 | J0 | 3407/v1 | 0.9568 | 0.9863 | 2.1392 | 37.6036 | 1.15 |
| s4:34 | JQ | 3407/v1 | 0.9553 | 0.9868 | 2.1142 | 37.7020 | 1.37 |
| s4:35 | QE50 | 3407/v1 | 0.9585 | 0.9876 | 2.0746 | 37.8790 | 1.54 |
| s4:36 | JE0 | 3407/v1 | 0.9555 | 0.9865 | 2.1309 | 37.6305 | 1.94 |
| s4:37 | QER50 | 3407/v1 | 0.9560 | 0.9864 | 2.1321 | 37.6200 | 1.54 |
| s5:17 | J0 | 2026/v1 | 0.9552 | 0.9870 | 2.1017 | 37.7591 | 1.38 |
| s5:18 | JQ | 2026/v1 | 0.9560 | 0.9875 | 2.0734 | 37.8693 | 1.57 |
| s5:19 | QE50 | 2026/v1 | 0.9557 | 0.9875 | 2.0760 | 37.8473 | 1.65 |
| s5:20 | J0 | 777/v1 | 0.9536 | 0.9875 | 2.0694 | 37.8924 | 1.30 |
| s5:21 | JQ | 777/v1 | 0.9549 | 0.9875 | 2.0729 | 37.8674 | 1.55 |
| s5:22 | QE50 | 777/v1 | 0.9559 | 0.9876 | 2.0701 | 37.8882 | 1.57 |
| s5:25 | EB_R3E050 | 2026/v2 | 0.9552 | 0.9867 | 2.1141 | 37.7097 | 1.56 |
| s5:26 | EB_R3E075 | 2026/v2 | 0.9576 | 0.9876 | 2.0705 | 37.8807 | 1.66 |
| s5:27 | EB_EDOWN | 2026/v2 | 0.9571 | 0.9876 | 2.0779 | 37.8677 | 1.58 |
| s5:28 | EB_EUP | 2026/v2 | 0.9567 | 0.9872 | 2.0941 | 37.7876 | 1.86 |
| s5:29 | EB_QFLOOR | 2026/v2 | 0.9567 | 0.9875 | 2.0745 | 37.8746 | 2.17 |

## 부록 B. 실행자 최종 확인

- [ ] 기존 미실행 QETUNE24가 아니라 **QRECON24 신규68run**으로 교체했다.
- [ ] 모든 primary에서 T0 A clone + W104D121 fresh U, λE>0, 고정 Teacher q/e다.
- [ ] U는 H+K+λEwE, A는 wH만 받고 Student offset·jitter는 없다.
- [ ] L_U+L_A를 단일 total backward하지 않는다. 두 gradient는 같은 forward 상태에서 계산한다.
- [ ] A의 direct soft/edge gradient가 0인 테스트, Teacher hash·frozen control 불변 검사를 통과했다.
- [ ] qref와 q 함수는 고정이며 old δ/rE/λE0/threshold pilot을 추가하지 않는다.
- [ ] q shuffle은 연속값의 조건부 permutation이며 old binary gate가 아니다.
- [ ] G22/H22/L100은 별칭, 같은 Student seed의 host 실행은 독립 seed가 아니다.
- [ ] 목표는 raw H≥.9585와 같은-checkpoint RR이며 기존 best와 마지막 값을 모두 보존한다.
- [ ] 현재 s4/s5의 느린 기록과 새 gradient 비용의 미측정을 시간표에 남겼다.
- [ ] 각 서버 actual 신규 완료 Train(h)≥24를 확인하고 부족하면 지정 추가 묶음을 실행한다.
- [ ] 학습 기동·중간 profile 변경·고정 seed 변경을 자동으로 수행하지 않으며 검증된 release만 배포한다.
