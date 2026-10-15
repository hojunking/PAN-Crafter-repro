# QEDGE9 — W104–D121에서 q 기반 edge gating 및 다중 seed 실험

**작성일:** 2026-09-15  
**주 실행 서버:** s5  
**병행 서버:** s4 — 기존 W104–D121 비교를 마친 뒤 동일 구조의 gate 대조  
**s3:** 현재 LFQ/LFX 중심의 기존 실험 유지  
**목표:** raw-original HQNR 0.959–0.960에 접근하면서, 기존 Q12보다 q 기반 edge 선택이 추가 이득을 만드는지 확인  
**시간:** 약 9시간을 편성 기준으로 삼되 hard stop은 두지 않음. 준비 완료 후 기본 학습·후처리 예약은 병렬 약 10시간 16분이며, 신규 cue 준비·구현 시간은 별도  
**산출물 지위:** 실험·구현 인계 명세. 이 문서를 작성하면서 서버 실행/중단, Git push, Sheet 수정은 하지 않았다.

---

## 0. 결정 요약

1. **새 gating 실험은 s5로 옮긴다.** Sheet에서 s5의 J0/JQ/D0/DQ/PQ/F0/LF0/LFQ/RC0/RCQ가 확인됐다. 반면 s3에는 LF0까지만 올라와 있고 LFQ/LFX는 아직 보이지 않는다. s3의 비교를 끊는 것보다 s5의 후속 우선순위를 바꾸는 편이 낫다는 편성 판단이다. Sheet가 실제 프로세스 상태를 알려 주지는 않으므로 현재 run은 끝까지 진행하고 다음 run 경계에서 전환한다. [S1]
2. **Student는 W104–D121, Teacher와 aligner source는 기존 T0로 고정한다.** s4의 W104–D121 NA0=0.9580, J0=0.9585가 후보 근거다. 아직 작은 구조의 Q12나 q gate의 효과가 검증됐다는 뜻은 아니다. [S1]
3. **e가 작아도 GT hard를 유지한다.** e에 따른 hard/soft 배분은 기존 Q12의 연속식을 계승하고, q는 기존 GT edge loss의 적용 patch만 선택한다. q로 hard/soft를 다시 배분하지 않는다. [S2]
4. **q가 낮은 patch에서만 GT edge를 적용하는 `QE50`을 주력으로 둔다.** θq는 고정 Teacher의 train-only q 중앙값이다. 새로운 Scharr-energy feature, 의미적 건물/식생 분류기, uncertainty head는 넣지 않는다.
5. **세 개의 고유 Student seed에서 동일 gate를 비교한다.** s5의 2026/777에 각각 J0/JQ/QE50를 실행하고, s4의 1234에는 기존 J0 및 예정 JQ와 QE50를 연결한다. 동일 seed의 서버 반복은 독립 seed 수로 늘리지 않는다.
6. **s4에서 상수 edge 감소와 e-조건부 gate 셔플을 추가한다.** 단순 edge 강도 감소, 잘못된 sample 재가중과 q의 정보 기여를 분리한다.
7. **이 신규 branch에는 과거 PAKD50의 50시간·9월 16일 절대 마감을 자동 상속하지 않는다.** 사용자가 이번 실험에는 hard time limit이 없다고 변경했다. 기존 캠페인 기록을 고치지 않고 새 branch의 시간 정책으로 분리한다.

**기본 편성:** s5 6run + s4 최대 6run(이 중 3run은 기존 architecture 큐의 잔여분). 12run이 모두 신규 가설인 것은 아니다.

---

## 1. 최신 Sheet 확인과 아키텍처 선정의 근거

### 1.1 s4의 W104–D121 결과

이번 조회는 `pan-cvpr27 / WV3-s4`, B3:AK198 범위에서 `PAKD50` 행을 읽었다. 아래 숫자는 **N열 HQNR, 원래 selected checkpoint의 raw-original 점수**다. V64·exact50K·plateau가 아니다. [S1]

| Sheet 행 | Run 핵심 | Raw HQNR | ERGAS | Params(M) | Train(h) |
|---:|---|---:|---:|---:|---:|
| 20 | NA0 · W104–D121 · S1234 | 0.9580 | 2.0879 | 1.9036 | 1.27 |
| 21 | J0 · W104–D121 · S1234 | 0.9585 | 2.1416 | 2.0090 | 1.17 |

같은 s4·seed1234의 small J0−small NA0는 표시값 기준 **+0.0005**다. Aligner를 추가한 복원 후보가 높게 시작했다는 근거이지만, 단일 seed·selected 비교다. ERGAS는 +0.0537로 높아졌다.

같은 Sheet의 W112–D123 J0는 v1=0.9594, v2=0.9568이다. 따라서 W104의 0.9585가 모든 큰 모델 결과를 이긴 것은 아니다. **새 구조를 주력 후보로 넓혀 시험할 근거는 있으나, 아키텍처의 보편적 우위를 확정하지 않는다.** 큰 모델 v1/v2 중 편리한 쪽만 택해 size 효과를 만들지 않는다.

이 조회 시점에 W104–D121의 JQ/XJ/F0는 아직 업로드되지 않았다. 실제로 진행 중일 수 있으므로 해당 run을 중복 시작하지 않는다.

### 1.2 기존 결과를 반영해 유지하거나 보류할 파라미터

| 관측 또는 명세 | 이번 결정 |
|---|---|
| s4 W112: XJ=0.9562, JQ_v1=0.9558, J_R3_NOEDGE=0.9524 | Edge는 유지하며 적용 patch를 바꾼다. 작은 구조에서도 XJ를 남겨 soft의 필요성을 확인한다. |
| s3 W112: XJ=0.9578, R3_NOEDGE=0.9575, JQ=0.9554 | Soft와 edge를 모두 일괄 적용하는 조합이 모든 조건에서 우세하지 않다. Gate 가설의 동기이지 q gate의 성공 증거는 아니다. |
| s5 W112: DQ=0.9549, PQ=0.9569, JQ=0.9576 | 첫 q gate에 초기 동결·protected routing을 함께 섞지 않는다. |
| s5 W112: LFQ=0.9587, LF0=0.9517 | LF는 s3에서 별도 확인한다. 새 구조+q gate+LF를 첫 run부터 동시에 적용하지 않는다. |
| s5 W112: RC0=0.9577, RCQ=0.9576 | Offset 제거와 q gate의 동시 변경은 우선 보류한다. |
| 기존 Q12 α=1, β=0.1; fixed-transfer λE0 | 이 값으로 대응 비교한다. β=0.05 등은 후속 paired variation이지 검증된 최적값이 아니다. |

각 서버의 absolute HQNR을 합쳐 하나의 대응 효과로 해석하지 않는다. 최신 last/plateau가 없는 행은 별도 분석에서 받기 전까지 미확보로 둔다. [S1, S2]

### 1.3 서버 우선순위

| 서버 | 이번 역할 | 변경 범위 |
|---|---|---|
| s5 | **W104–D121 q-edge gate의 두 seed 대응 실험** | 완료된 기존 묶음 뒤의 저우선순위 확장 대신 신규 queue |
| s4 | 작은 구조의 기존 비교 마무리 + 세 번째 seed 및 원인 대조 | 미완료 JQ/XJ/F0는 보존; QE50/QEC/QES 추가 |
| s3 | W112 LFQ/LFX 및 기존 대조 유지 | 이번 문서에서 새로운 run을 추가하거나 취소하지 않음 |
| s1/s2 | 별도 진행 중인 작업 유지 | 이번 allocation의 대상 아님 |

여기서 “s5의 우선순위가 낮다”는 것은 **앞으로 추가할 기존 확장의 한계 효용**에 대한 판단이다. s5의 결과나 하드웨어가 나쁘다는 뜻이 아니다. [S1]

---

## 2. q와 e — 고정할 정의

### 2.1 q: 고정 Teacher aligner의 추가 변위 반응 잔차

\[
c^T_{i,0}=A_T(P_i,M_i),\qquad
c^T_{i,k}=A_T(\mathcal W(P_i,\epsilon_k),M_i),
\]

\[
q_T(i)=\frac{1}{2K}\sum_{k=1}^K
\left\|c^T_{i,k}+\epsilon_k-c^T_{i,0}\right\|_1.
\]

- **K=16**, radius {0.25, 0.5, 1, 2} HR pixel × {+y, −y, +x, −x}. EQREC4 axis bank A를 재사용한다.
- 두 보정 좌표 성분과 probe를 모두 평균한다. EPE/L2나 epsilon norm으로 나눈 값으로 바꾸지 않는다.
- 단위는 현재 PAN grid pixel이며 λoff를 곱하지 않는다.
- q가 **낮을수록** 추가 변위에 대한 반응이 일관적이다. 건물 확률·Teacher 정확도 확률·절대 정합 참값이 아니다.
- q는 **고정 T0 aligner**에서 계산한다. 현재 업데이트되는 Student aligner의 q로 gate를 만들지 않는다.
- 추가 probe PAN은 Teacher aligner에만 넣는다. Jittered Teacher U-Net을 16번 호출하는 방식은 아니다.
- Student의 offset 보조학습과 Teacher q 진단은 별개다. 기존 Student offset target을 Teacher c0로 교체하지 않는다. [S3, S4]

### 2.2 e와 기존 hard/soft

\[
e_T(i,p)=\frac1C\sum_c|Z_{T,i,c}(p)-Y_{i,c}(p)|,
\quad e_S(i,p)=\frac1C\sum_c|Z_{S,i,c}(p)-Y_{i,c}(p)|,
\]
\[
k(i,p)=\frac1C\sum_c|Z_{S,i,c}(p)-\operatorname{sg}(Z_{T,i,c}(p))|.
\]

\[
d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\qquad
a_T=\operatorname{sg}\frac{[e_S-e_T]_+}{e_S+10^{-6}},
\]
\[
L_H=\left\langle(1+\alpha d_T)e_S\right\rangle,
\qquad
L_K=\left\langle\beta(1-d_T)a_Tk\right\rangle.
\]

**wH≥1이다. e가 낮거나 Teacher가 Student보다 좋아도 GT hard를 삭제하지 않는다.** Student가 Teacher보다 정확한 위치에서는 aT=0이어서 soft는 꺼진다.

이번 핵심 실험에는 이진 e cutoff를 새로 넣지 않는다. 사용자와 합의한 “e↑는 GT 중심, e↓는 Teacher 참조 허용”을 기존 Q12의 연속식으로 유지한다. q는 edge 선택에만 사용한다. [S2]

EQREC4의 patch 진단 e는 중앙 `[16:48]²` ROI 평균이다. 위 reconstruction loss는 **기존 full-domain support**를 유지한다. 보고서의 중앙 ROI를 학습 loss 전체에 조용히 적용하지 않는다. 셔플 대조에 사용할 patch e만 별도 이름 `e_T_roi32`로 저장한다. [S2, S3]

---

## 3. 주력 변경: QE50

### 3.1 q gate

\[
\theta_q=\operatorname{median}_{i\in\mathcal D_{cal,q}}q_T(i),
\qquad
 g_i=\operatorname{sg}\bigl(\mathbf1[q_T(i)<\theta_q]\bigr).
\]

- θq는 train-only calibration의 고정값이다. 그림에서 읽은 0.31/0.32 같은 값을 직접 입력하지 않는다.
- θq와 같은 값은 `g=0`으로 처리한다. 동일값이 많아 선택 비율이 50%와 다르면 그 사실을 기록하고, batch별 top-k로 조용히 바꾸지 않는다.
- `50`은 calibration 중앙값 gate라는 뜻이다. 매 minibatch의 정확히 절반을 강제로 선택한다는 뜻이 아니다.
- q는 patch별 scalar다. 이것을 pixel별 alignment confidence map이라고 부르지 않는다.

### 3.2 기존 GT edge를 sample별로 계산한 뒤 선택

기존 signed Scharr edge loss의 sample별 값을 Ei라고 둔다.

\[
E_i=\frac12\operatorname{mean}_{c,p\in V_E}|D_xZ_{S,i,c}-D_xY_{i,c}|
+\frac12\operatorname{mean}_{c,p\in V_E}|D_yZ_{S,i,c}-D_yY_{i,c}|.
\]

`VE`는 기존 edge 계산의 내부 1px 제외 영역이다. 기존 Scharr /32, reflect padding, MS band별 signed 비교를 유지한다. **Scharr energy를 gate feature로 사용하는 것이 아니라, 기존 Q12의 edge 연산만 유지하는 것이다.** [S2, S5]

\[
\boxed{L_{QE50}=L_H+L_K+\lambda_E\frac1B\sum_i g_iE_i
+m_t\lambda_{off}L_{off,S}.}
\]

**기존 Q12의 edge 항을 교체한다. 기존 edge에 gated edge를 더하는 방식이 아니다.**

`sum(g_i*E_i)/B`를 사용한다. `sum(g_i*E_i)/sum(g_i)`나 active-patch 평균으로 재정규화하지 않는다. 선택 비율이 낮아질수록 총 edge 압력이 낮아질 수 있으며, 그 부분은 QEC 대조로 구분한다.

### 3.3 네 경우의 동작

| 측정 상태 | GT reconstruction | Teacher output soft | GT edge |
|---|---|---|---|
| q↓ · e↑ | 항상 유지, e에 따라 강화 | 기존 d/a에 따라 제한적 | 적용 |
| q↑ · e↑ | 항상 유지, e에 따라 강화 | 기존 d/a에 따라 제한적 | 미적용 |
| q↑ · e↓ | 항상 유지 | Teacher가 더 나으면 허용 | 미적용 |
| q↓ · e↓ | 항상 유지 | Teacher가 더 나으면 허용 | 적용 |

이는 **edge가 중요한 데이터–모델 상호작용을 q로 선택할 수 있는지 시험하는 가설**이다. 네 칸을 건물/식생의 정답 라벨로 취급하지 않는다. 기존 보고서는 raw q를 Teacher-quality gate로 채택하지 않았다. 이번 변경은 그 결론을 뒤집어 기정사실화하는 것이 아니라, **GT edge 선택이라는 다른 용도**를 새로 시험한다. [S3]

### 3.4 Gradient 경로

| 항 | Student U-Net | Student aligner | Teacher |
|---|---|---|---|
| GT hard | 유지 | native sampler 경로 유지 | 없음 |
| Output soft | 유지 | 기존 joint 경로 유지 | 없음 |
| Gated GT edge | q↓ patch만 | q↓ patch만 | 없음 |
| Student offset consistency | 직접 없음 | 기존 격번 경로 | 없음 |
| q·θq·g 계산 | 없음 | 없음 | frozen/no_grad |

이번 기본 묶음에서는 loss gating과 routing-only를 동시에 바꾸지 않는다. `gE`가 0이면 해당 edge의 U/A gradient가 함께 0이다. 여기에 같은 이진 gate를 한 번 더 곱해 독립적인 추가 기여로 세지 않는다.

---

## 4. 고정 파라미터

아래는 새 구조에 대한 “최적값 확정”이 아니라 **기존 성공 명세를 고정 이식하는 기준값**이다. S4 small-J0가 높다는 이유만으로 soft·edge 계수를 함께 바꾸지 않는다. [S2, S4, S6]

| 항목 | 값 |
|---|---|
| Student backbone | `model.pancrafter_paper.PANCrafterPaper`, hidden_size=104, depth=[1,2,1] |
| Student 나머지 model_args | 현재 s4 W104 config 그대로: norm=ln, in_mode=paper, attn_locations=[], mode_modulation=false 등 |
| Input/output | PAN1 + MS8 → HRMS8, M-frame, MS residual base 한 번 |
| Teacher | T0, W112–D123, A/U 모두 frozen, 자기 aligner로 native forward |
| Teacher asset | `PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1/best_hqnr`, step24240 |
| Config에 기록된 T0 file SHA256 | `16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32` — 실제 local 파일 대조 필수 |
| Student aligner | T0의 최종 A만 strict copy; J 계열에서 trainable |
| Student U 초기화 | W104–D121 × seed별 새 초기값, 대응 case 사이에 동일 tensor 사용 |
| Teacher/Student 용량 차이 | `teacher.bridge: true`; 동일 용량 KD로 보고하지 않음 |
| Optimizer / U LR / A LR | AdamW / 1e-4 / 1e-5 |
| Weight decay / scheduler | 0.01 / warmup100 + cosine |
| Updates / effective batch | 50,000 / 48 |
| α / β / ε | 1.0 / 0.1 / 1e-6 |
| τR | 0.012463942170143127 |
| λE0 | 0.09075170336956798, fixed-transfer; Small-U gradient로 새 보정했다는 뜻 아님 |
| λoff / 빈도 | 1e-4 / 0-based 홀수 update |
| Student perturbation | 반경2 HR pixel의 원판 면적 uniform; Teacher q의 축 probe bank와 구분 |
| q threshold | train-only median, 값은 신규 산출; FR20으로 fit 금지 |
| q gate direction | 낮은 q만 선택 |
| q source | fixed T0 A; Student q 사용 안 함 |
| 선택·평가 | raw-original HQNR, 기존 fSCC tie 처리, GRID1010_50K_v1; exact50K·고정 plateau 별도 |

**이번에는 변경하지 않는 것:** GT hard를 없애는 switch, e 이진화, 새 edge filter, output alignment, GT 이동, Teacher 재학습, LF/RC/AL와의 동시 결합, 학습량 축소, adaptive λoff.

---

## 5. s5 — 두 seed의 gate 순증분과 no-KD 기준: 6run

**실행 순서:** `J0_S2026 → JQ_S2026 → QE50_S2026 → J0_S777 → JQ_S777 → QE50_S777`  
**전부 W104–D121. 기존 s5 W112 결과는 이 여섯 run의 대조군을 대신하지 못한다.**

| 순서 | Case | Seed | 구성 | 필수 비교 |
|---:|---|---:|---|---|
| 1 | J0 | 2026 | GT-only joint + offset | 작은 구조·s5의 no-KD 기준 |
| 2 | JQ | 2026 | 기존 Q12, 모든 patch edge | JQ−J0 |
| 3 | **QE50** | 2026 | 기존 e hard/soft + q↓ edge | **QE50−JQ**, QE50−J0 |
| 4 | J0 | 777 | 1번의 다른 U seed | 독립 Student seed의 no-KD 기준 |
| 5 | JQ | 777 | 2번의 다른 U seed | JQ−J0 |
| 6 | **QE50** | 777 | 같은 θq/계수/gate 규칙 | **QE50−JQ**, QE50−J0 |

세 가지 변형을 같은 seed에서 각각 fresh50K로 시작한다. J0에서 JQ로, JQ에서 QE50로 이어 학습하는 체인이 아니다.

S2026은 기존 s3/s5와 이름이 같지만, **W104 초기값은 W112 초기값과 다르다.** Architecture별 hash namespace를 사용한다. Seed777도 W104 초기값을 새로 생성하고 case 사이에 공유한다.

첫 seed 결과만 보고 두 번째 seed의 β·θq를 바꾸지 않는다. 바꾸면 새 variant와 새 대응군으로 분리한다.

---

## 6. s4 — 현재 architecture 묶음 보존 + 세 번째 seed와 원인 대조: 최대 6run

NA0와 J0의 W104·seed1234는 이미 Sheet에 있으므로 재실행하지 않는다. **JQ/XJ/F0는 기존 미완료 allocation을 그대로 완료**한다. 이 세 건이 실제로 이미 완료/진행 중이면 해당 상태를 재사용하고 중복하지 않는다.

**순서:** `JQ_S1234 → XJ_S1234 → F0_S1234 → QE50_S1234 → QEC_S1234 → QES_S1234`.

| 순서 | Case | 구성 | 역할 |
|---:|---|---|---|
| 1 | JQ | 기존 Q12 | Small-U Q12 anchor; 기존 예정 run |
| 2 | XJ | Teacher-error hard + GT edge, soft 없음 | 작은 구조에서 output soft의 필요성; 기존 예정 run |
| 3 | F0 | T0 A frozen + GT-only | 단순 정합 재사용 기준; 기존 예정 run |
| 4 | **QE50** | q↓ edge gate | 세 번째 고유 Student seed의 gate 비교 |
| 5 | **QEC** | 모든 patch edge × 고정 상수 cE | 단순 전체 edge 감소 효과의 대조 |
| 6 | **QES** | e가 유사한 patch 사이에서 q gate 셔플 | q와 sample의 올바른 연결이 필요한지 확인 |

### 6.1 QEC — 고정 상수 대조

s4의 **W104–D121 J0 S1234 exact50K**를 frozen pilot으로 사용한다. Train-only calibration에서 얻은 sample별 edge 오차를 `E_pilot(i)`라 할 때:

\[
c_E=\frac{\sum_i g_i E_{pilot}(i)}{\sum_i E_{pilot}(i)}.
\]

\[
L_{QEC}=L_H+L_K+\lambda_E c_E\operatorname{mean}_i E_i+L_{off,weighted}.
\]

cE는 산출 후 고정한다. 분모가 퇴화하면 임의 상수를 채우지 않고 calibration 실패로 처리한다. 이것은 **고정 pilot에서 edge loss 평균을 맞춘 대조**이며, 전 학습 구간의 parameter gradient까지 일치하는 대조가 아니다. 그 차이는 기록한다.

단순히 중앙값 gate라는 이유로 cE=0.5라고 놓지 않는다. 선택된 patch에서 edge 오차가 더 크면 평균 weighted loss는 50%보다 클 수 있다.

### 6.2 QES — e-조건부 셔플

- Frozen T0의 native `e_T_roi32`를 train-only 자료에서 계산한다. 실제 reconstruction 학습 support를 ROI32로 바꾸지는 않는다.
- Train calibration의 e decile 경계를 고정한다.
- **같은 e bin과 같은 실제 augmentation state 안에서** binary gate를 고정 permutation으로 교환한다.
- Permutation seed는 51515로 고정하고 dataset-wide assignment를 asset으로 저장한다. 각 stratum의 active count는 보존한다.
- Student seed/예측/GT loss를 보고 셔플을 다시 뽑지 않는다. Batch 순서에 따라 label을 재생성하지 않는다.
- 작은 stratum·동일 gate만 있는 stratum은 보존하고 해당 비율을 기록한다.

QES가 QE50와 비슷하면 q 자체보다 선택 비율·e와의 공통 관계·학습 변동 등의 대안 설명이 남는다. QE50가 QEC/QES보다도 좋아야 q 선택의 추가 정보 기여를 더 구체적으로 주장할 수 있다.

### 6.3 seed 회계

기본 묶음 이후 QE50−JQ의 고유 seed는 다음 세 개다.

| Seed | 대응 비교가 이루어지는 서버 |
|---:|---|
| 1234 | s4 |
| 2026 | s5 |
| 777 | s5 |

각 seed 안에서는 같은 서버의 차이를 먼저 계산한다. 서버별 absolute HQNR을 곧바로 빼지 않는다. 모두 같은 Teacher T0를 쓰므로 **고정 Teacher에 조건부인 Student 반복**이지, Teacher 학습부터 독립인 세 파이프라인이 아니다.

---

## 7. q 계산 비용: 16 probe를 50K 내내 반복하지 않는다

### 7.1 학습용 cue cache

q는 Teacher가 고정되어 있고 현재 crop=False이므로 같은 입력 변환 상태에서 반복 계산할 필요가 없다. 다음 key로 저장한다.

`Teacher/A hash + dataset hash + sample_id + actual augmentation state + PAN/MS 전처리/보간 hash + probe bank + dtype`

Student backbone·seed·학습 update는 key에 필요하지 않다. 다만 P/M 자체를 바꾸는 다른 전처리/augmentation은 다른 key다.

**권장 구현:** 학습 시작 전 또는 별도 준비 단계에서 train의 실제 유한 augmentation 상태를 열거하고 q cache를 만든다. 학습 loader는 sample_id/aug_state를 RNG를 추가 소비하지 않는 방식으로 함께 전달한다. cache q와 online 계산의 동일성을 고정 sample로 검사한다.

### 7.2 현재 feeder를 그대로 보존할 것

조회한 `feeders/feeder.py`에서는 `hflip=True`와 `vflip=True`가 각각 조건 없이 적용되고, `rot=True`에서만 `random.randint(0,3)`을 뽑는다. 따라서 이 config의 유효 영상 상태는 **고정 flip 조합 + 네 rotation, 즉 4개**다. 이름만 보고 8개 무작위 dihedral 상태라고 가정하지 않는다. [S7]

이 사실을 이유로 flip을 새 Bernoulli 방식으로 “수정”하지 않는다. 그러면 gate 실험과 augmentation 변경이 섞인다. 실제 서버가 다른 feeder revision을 사용한다면 그 revision의 상태 수와 입력 경로를 manifest에 기록하고, cache도 다시 맞춘다.

Dataset 생성자가 전역 random seed를 설정하는 코드도 있다. Cue 준비를 별도 프로세스로 실행하거나 RNG state를 엄격히 복원해 기존 data/augmentation/epsilon 순서를 바꾸지 않는다.

### 7.3 θq calibration과 전체 cache의 범위

1. 기존 train calibration seed1234로 최대 3,072개의 **base patch ID**를 고정한다. 학생 학습 seed와 구분한다.
2. 각 ID의 실제 네 augmentation state를 동일 규약으로 생성한다. 현재 config라면 최대 12,288개 view다.
3. 해당 view에서 frozen T0 q를 계산해 θq를 고정한다. FR20·RR test로 threshold를 정하지 않는다.
4. 같은 실행 입력·bank로 전체 train view의 q cache를 채운다. N은 로컬 H5에서 읽어 기록하며, 논문의 train 개수를 무조건 하드코딩하지 않는다.
5. QES용 e와 QEC용 pilot error는 별도 필드/asset으로 보존한다. q 자체에는 GT나 Teacher U 출력이 필요 없다.

Teacher A는 입력 margin4·표준화·sampling 규약을 기존 adapter와 동일하게 사용한다. `M`을 임의의 새 bicubic 또는 제공 lms로 대체하지 않고 **실제 Teacher forward에 사용되는 M**과 맞춘다.

### 7.4 비용 산식

현재 네 상태라면 q cache의 논리적 sample-forward 수는 대략:

\[
N_{A\text{-}views}=4N(1+16)=68N.
\]

이는 68N번 Python 함수를 개별 호출하라는 뜻이 아니다. Batch/vectorization과 적절한 microbatch로 처리한다. 네 rotation의 q가 자동 불변이라고 가정해 하나만 계산하지 않는다.

실제 처리시간은 먼저 n개 view를 benchmark해:

\[
T_{cue}\approx T_{bench}\frac{N_{views}}{n}+T_{I/O}+T_{e/pilot}
\]

로 산정하고 실제 총시간으로 교체한다. **아직 측정하지 않은 Tcue를 0 또는 임의의 몇 분으로 확정하지 않는다.** 전체 online bank를 매 step 계산하는 fallback을 선택하면 아래 cached-run 시간표는 사용할 수 없으며 다시 실측한다.

Cue asset은 한 서버에서 검증해 s4/s5에 hash로 전달할 수 있다. 서버 간 고정 sample에서 q 및 gate label을 대조하고, threshold 인근 label 차이를 기록한다. 서버별로 θq를 몰래 재보정하지 않는다.

---

## 8. 구현 연결 및 최소 검사

### 8.1 확인한 저장소 상태

조회 기준 commit은 `1a3edb9df34edeb07571989a926248bdec8cda85`다.

- W104–D121 architecture registry와 `case@arch` 파싱, `teacher.bridge=true`, architecture별 초기값 namespace는 이미 반영됐다. [S6]
- `QE50`, `QEC`, `QES`는 **이번 계획의 새 case 이름**이다. 생성기·loss·cache·metadata에 등록해야 한다. 문서가 존재한다고 구현 완료로 취급하지 않는다.
- `pa.losses.output_edge_loss`는 현재 global scalar를 반환한다. q gate를 적용하려면 per-sample Ei를 제공하는 경로가 필요하다. 기존 기본 경로는 보존하고 g=1에서 수식·gradient 동치를 검사한다. [S5]

참고 구현 논리:

```python
# 기존 pa.losses.scharr를 재사용한다. 실제 호출부에서 기존 FP32 정책을 유지한다.
gx_s, gy_s = scharr(student_hrms)
gx_y, gy_y = scharr(gt_hrms)
ex = (gx_s[..., 1:-1, 1:-1] - gx_y[..., 1:-1, 1:-1]).abs()
ey = (gy_s[..., 1:-1, 1:-1] - gy_y[..., 1:-1, 1:-1]).abs()
edge_i = 0.5 * ex.flatten(1).mean(1) + 0.5 * ey.flatten(1).mean(1)
edge = (edge_weights.detach().reshape(-1) * edge_i).mean()
# total = 기존 L_H + 기존 L_K + lambda_E * edge + 기존 Student offset 보조항
```

이는 연산 의미를 보여 주는 조각이며 trainer에 바로 붙여 실행이 끝나는 완성 patch가 아니다.

### 8.2 필수 검사

| 검사 | 합격 의미 |
|---|---|
| q 부호·단위 | ideal cε=c0−ε에서 q=0; λoff 포함 안 함 |
| q 출처 | T0 hash 고정; Student A를 바꿔도 같은 P/M의 q cache는 불변 |
| g=1 | 기존 Q12 edge 값·gradient와 수치 허용오차 내 일치 |
| g=0 | Edge만 0, GT hard와 기존 soft/offset gradient는 유지 |
| 저오차 patch | eT가 작아도 wH≥1, eS≤eT이면 aT=0 |
| Teacher·GT detach | parameter 및 target에 gradient 없음 |
| Native 재구성 | hard/soft는 Student A/U에 기존 경로로 전달 |
| Edge target | GT의 같은 band·signed filter·내부1px, PAN/Teacher edge target 아님 |
| Cache 동일성 | sample ID·실제 augmentation 상태가 바뀌면 올바른 cache 값 선택 |
| RNG 분리 | q/cache/셔플 준비를 켜도 data/epsilon 학습 순서가 변하지 않음 |
| Resume | cache/hash/permutation 및 다음 batch/update 경로 동일 |
| Architecture | Student W104D121, Teacher W112D123; U weight 부분 로드·채널 자르기 금지 |
| 학습 완료 | 기존 완료 marker·50K state·평가 후보를 확인; finished timestamp만으로 성공 판정 안 함 |

현재 Student U가 zero-output 초기화라면 첫 step의 A gradient가 0일 수 있다. 비퇴화 고정 batch/weight에서 경로를 검사하고, 이를 신규 gate가 A를 끊은 증거로 오해하지 않는다. [S6]

### 8.3 이전 control의 재사용

s4의 기존 small J0/JQ/XJ/F0는 Teacher·U init·data/epsilon 규약·loss·평가기 동치가 확인되면 재사용한다. 예산 metadata만 새로 잡는 것과 학습 수식을 바꾸는 것을 구분한다.

동치가 깨지는 실제 code 변경이 발견되면 새 baseline을 같은 release로 수행하고 기존 asset은 보존한다. 이 추가 control 시간은 아래 기본 예약에 포함되어 있지 않으므로 명시적으로 더한다.

---

## 9. 시간 계산 — 약 9시간을 목표로 하되 50K 완료를 우선

### 9.1 실제 Sheet 시간과 새 추정의 분리

| 서버 / 구조 | 완료 case의 Train(h) | 사용 원칙 |
|---|---|---|
| s4 / W104D121 | NA0 1.27, J0 1.17 | 실제 작은 구조 관측. 작아졌으니 모든 run이 더 빠를 것이라고 가정하지 않음 |
| s4 / W112D123 | JQ 1.39–1.40, XJ 1.39, F0 1.14 | 아직 small counterpart가 없을 때 **동일 서버의 대용값** |
| s5 / W112D123 | J0 1.35, JQ 1.36, PQ 1.53 | s5 small 모델은 아직 미측정. J0/JQ의 초기 대용값 |
| QE50/QES | 완료 실측 없음 | **cache-ready 1.50h/run을 임시 편성값**으로 둠. 보장된 상한 아님 |

NA0 1.27h는 1시간 16분 12초, J0 1.17h는 1시간 10분 12초다. 소수 부분을 분으로 읽지 않는다.

### 9.2 예약식

\[
T_{reserve}(run)=1.10T_{reference}(run)+10/60\ \text{hour}.
\]

10%는 실행 변동 여유, 10분은 종료 후 기본 export/gspread/큐 전환 가예약이다. 정기 평가 비용이 포함된 기존 run 시간에 정기 평가 50회 비용을 다시 더하지 않는다. q cache 준비는 이 표 밖의 **일회성 실제 시간**이다.

| 서버 | 순서 | Case · seed | 기준(h) | 예약(h) |
|---|---:|---|---:|---:|
| s5 | 1 | J0 · 2026 | 1.35 | 1.6517 |
| s5 | 2 | JQ · 2026 | 1.36 | 1.6627 |
| s5 | 3 | QE50 · 2026 | 1.50* | 1.8167 |
| s5 | 4 | J0 · 777 | 1.35 | 1.6517 |
| s5 | 5 | JQ · 777 | 1.36 | 1.6627 |
| s5 | 6 | QE50 · 777 | 1.50* | 1.8167 |
| **s5 합** | **6** | | **8.42** | **10.2620** |
| s4 | 1 | JQ · 1234 | 1.40 | 1.7067 |
| s4 | 2 | XJ · 1234 | 1.39 | 1.6957 |
| s4 | 3 | F0 · 1234 | 1.14 | 1.4207 |
| s4 | 4 | QE50 · 1234 | 1.50* | 1.8167 |
| s4 | 5 | QEC · 1234 | 1.40 | 1.7067 |
| s4 | 6 | QES · 1234 | 1.50* | 1.8167 |
| **s4 합** | **6** | | **8.33** | **10.1630** |

`*`는 신규 cue 경로의 미측정 계획값이다. 반올림한 행 예약을 다시 합하지 말고 원래 소수로 계산한다.

- **s5:** 10.262h = **10시간 15분 43초**.
- **s4:** 10.163h = **10시간 9분 47초**.
- 두 서버 예약 합: **20.425 서버·시간**.
- 각 서버 한 slot, 준비 완료 후 동시 시작 기준 **병렬 경과시간은 최대값인 약 10시간 16분**. 20.425시간이 아님.
- s4의 첫 3건이 이미 완료되면 제거하고, 진행 중이면 잔여시간만 산입한다.

실제 종료 예상은 다음 항을 추가해야 한다.

`현재 run 잔여시간 + 구현/배포 준비 + cue asset 준비/전달 의존성 + 해당 서버의 남은 예약 합`

s4는 cue가 없어도 기존 JQ/XJ/F0를 진행할 수 있다. s5에서 cue를 준비해 전달하면 s4의 뒤 3건이 시작된다. cue가 늦으면 그 대기만 실제 critical path에 추가한다.

**9시간을 맞추려고 50K를 40K로 줄이거나 확인 seed를 중간에 끊지 않는다.** 사용자가 초과를 허용했으므로 약 10시간대의 기본 편성으로 잡고, 실제 cue 비용이 크면 해당 비용을 공개한 채 연장한다.

### 9.3 새로운 시간 정책

기존 config에는 `2026-09-16T11:22:31`과 50h budget이 들어 있다. [S6]

이번 신규 branch는 다음 의도를 명시해야 한다.

```yaml
# 계획용 필드. 기존 trainer가 이 schema를 지원한다고 가정하지 않는다.
time_policy:
  mode: soft_target_only
  target_elapsed_hours: 9
  hard_deadline: null
  inherit_parent_deadline: false
```

생성기/runner의 새 branch 처리에서 기존 절대 deadline·누적50h를 자동 상속하지 않도록 검사한다. 단, NaN/오류/자원 부족/중복 run 보호는 유지한다. 과거 campaign_clock·완료 ledger를 삭제하거나 덮어쓰지 않는다.

---

## 10. 결과 해석 및 후속 파라미터

### 10.1 첫 판정

고유 seed별로 같은 서버에서:

\[
\Delta H_{gate,s}=H(QE50_s)-H(JQ_s),
\]
\[
\Delta H_{total,s}=H(QE50_s)-H(J0_s)
\]

를 먼저 계산한다. Selected raw HQNR, exact50K, 고정 45,450/46,460/47,470/48,480/49,490/50,000 plateau를 다른 열로 보존한다. q 감소·RR L1 감소로 raw HQNR 성공을 대체하지 않는다.

- **Gate의 추가 가치:** QE50−JQ의 seed별 방향·평균·표준편차.
- **전체 backend의 추가 가치:** QE50−J0. JQ가 이미 작은 J0보다 낮다면 “Q12 회복”과 “GT-only 초과”를 구분.
- **q의 sample 선택 가치:** s4 QE50−QEC 및 QE50−QES.
- **Soft의 필요성:** s4 JQ−XJ. q gate를 붙인 soft 필요성은 후속 β=0 대조가 별도로 필요.
- **최종 비용:** 실제 2.009M 내외 Student+A parameter와 추론 시간. q/GT/Teacher는 추론에서 제거.

0.0031 같은 과거 변동선을 자동 유의성 검정으로 사용하지 않는다. 세 고유 Student seed의 기술통계를 우선 보고, source-scene별 결과와 별도 분석을 연결한다. 서로 다른 branch의 최고 seed만 선택해 평균을 만들지 않는다. [S4]

### 10.2 선택 예시

| 결과 | 다음 조치 |
|---|---|
| QE50가 JQ와 J0를 여러 seed에서 높이고 QEC/QES보다 좋음 | gate 후보 유지, 새 확인 seed로 진행 |
| QE50≈QEC이고 둘 다 JQ보다 좋음 | q의 정보 기여보다 edge 총강도 감소를 우선 해석 |
| QE50≈QES | 실제 q–sample 연결의 필요성이 아직 약함 |
| QE50가 JQ를 높이지만 J0보다 낮음 | KD 회복은 가능하지만 최종 fitting 방법의 채택 근거는 부족 |
| 한 seed에서만 높은 selected, last/plateau는 반대 | 해당 자산의 성능과 안정적인 방법 개선을 분리 |
| 작은 구조에서 XJ가 반복적으로 유리 | 다음 paired β=0/0.05/0.1 중 한 축만 검토; gate의 주장을 soft KD의 성공으로 확대하지 않음 |

### 10.3 선택적 연장 — hard limit 없음

**첫 우선순위: 새 seed9091의 `J0/JQ/QE50` 3run.** θq·계수·정합 정책·코드를 먼저 고정한다. s5 같은 대용값으로 기준 4.21h, 예약 **5.131h = 약 5시간 8분** 추가다. 기본 s5 6run 뒤에 붙이면 예약 15.393h이며, 준비시간은 여전히 별도다.

**둘째 우선순위: β만 줄인 대응쌍.** 기본 결과가 soft 강도 조정을 요구할 때만 `JQ_B005/QE50_B005`를 같은 seed2026에서 실행한다. β=0.05만 바꾸고 α, λE, θq는 고정한다. 이것은 새 실험값이며 현재 최적값이 아니다. 한 gate run만 β를 바꿔 기존 β=0.1 JQ와 비교하지 않는다.

**그다음:** s3에서 LFQ/LFX의 대응 효과가 정리되면, small LFQ와 small LF+QE50를 하나의 쌍으로 확장할 수 있다. 새 구조·새 q gate·새 freeze schedule을 처음부터 묶는 것은 이번 기본 실험에서 제외했다.

**이진 e gate와 routing-only**도 별도 후보다. 이번 문서의 주력은 기존 Q12 e 배분 유지 + q edge 선택이며, low-e GT hard 유지 결정은 모든 확장에서 보존한다.

---

## 11. 실행/배포·gspread 인계

### 11.1 식별자

신규 논리 캠페인: `QEDGE9_A104D121_20260915_v1`  
기존 Teacher/architecture parent: `PAKD50_W112D123_WV3_20260914_v1`  
신규 branch: `A104D121_T0FIX_QEDGE9_v1`

업로더 호환을 위해 실행명은 기존 PAKD50 prefix와 architecture token을 유지한다. 예:

```text
PAKD50_QE50_W104_D121_WV3_T0_S2026_FRESH50_v1
PAKD50_QEC_W104_D121_WV3_T0_S1234_FRESH50_v1
PAKD50_QES_W104_D121_WV3_T0_S1234_FRESH50_v1
```

동일 이름이 로컬에 있으면 완료 asset/hash를 확인하고 재사용하거나 새 version을 부여한다. 기존 run 디렉터리를 덮어쓰지 않는다. 서로 다른 서버에서 같은 run ID를 사용한 경우 결과 merge key에 server/environment ID를 포함한다.

### 11.2 배포

기존 운영 방식대로 별도 개발 worktree에서 구현·unit/smoke를 통과한 **고정 commit**을 s4/s5에 push/pull로 전달한다. 실행 중인 worktree를 pull로 변경하지 않는다. s1/s2/s3의 현재 작업을 이 문서로 재배정하지 않는다.

Registry에 새 case만 넣는 것으로 완료 처리하지 않는다. Edge per-sample 경로, cue cache 연결, reducer, seed/init hash, 새 budget policy, queue, gspread descriptor까지 연결한다.

### 11.3 Sheet

- `WV3-s4`, `WV3-s5` 기존 worksheet를 사용한다. 직접 Sheet를 수정하지 않는다.
- B:W 기존 format과 N열 raw HQNR/O열 V64 위치를 유지한다.
- X열 예: `PAKD50 / QE50 / A104D121 / QEDGE9 / FRESH50`.
- Notes: `q_source=T0; q_bank=AXIS16; q_cut=median; theta_q=<actual>; edge_gate=low_q; hard_always=1; branch=...; cue_sha=...`.
- QEC는 cE, QES는 permutation seed/hash를 추가한다.
- Training time은 실제 started/finished 기록을 업로드한다. 문서의 1.50h 가예약을 실측 열에 넣지 않는다.
- θq/cE가 아직 미산출이면 해당 gate run은 대기하고 placeholder 0으로 학습을 진행하지 않는다.

### 11.4 인계 완료 체크

`W104 model/Teacher bridge 확인 → 기존 low-e GT hard 유지 → train q cache/θq 고정 → g=1/0 및 RNG/resume 검사 → 새 soft-time budget 확인 → s5 대응 queue → s4 잔여+대조 queue → 기존 gspread 업로드`

**최종 의도:** 작은 backbone의 높은 출발점을 보존하면서, Q12의 GT edge 감독만 q로 선택해 추가 raw HQNR 이득을 얻을 수 있는지 시험한다. q를 Teacher 정확도로 간주하거나 건물/식생 분류기로 바꾸지 않는다.

---

## 12. 출처와 확인 범위

**[S1] 현재 live Sheet 읽기, 2026-09-15.** `pan-cvpr27`, spreadsheet ID `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`. s4 B3:AK198, s3 B3:AN200, s5 B3:AN200의 PAKD50 검색. 주요 원값과 행 위치는 동봉 `source_snapshot.json`. 본 문서에서 프로세스 상태나 checkpoint 수치를 새로 재평가한 것은 아니다.

**[S2]** `PAN_KD_Q12_R1_Spec_and_Aligner_Interface_2026-09-14.md`, §4–5(e/d/a 및 hard/soft), §5.2(edge), §7(calibration/reduction), §9(aligner 경로).

**[S3]** `2026-09-15_s1_eqrec4-results.md`, §A.1–A.3 및 §B 기호·K10/K20; `PAN_Aligner_Analysis_Summary_2026-09-15.md`, §2·5. q cue의 의미·한계와 새 gate 가설의 구분.

**[S4]** `PAN_Aligner_L1E4_TechnicalSpec_Evidence_KD_Handoff_2026-09-14.md`, §2–4(보정/offset), §6(raw HQNR/seed 변동), §5(A/U asset).

**[S5] 코드:** `pa/losses.py`, pinned commit `1a3edb9df34edeb07571989a926248bdec8cda85`, `scharr`와 `output_edge_loss`. Blob `8f97708b10091587f09976f72abcf66aa6d6ab33`.

**[S6] 코드·구현 노트:** 같은 commit의 `research_log/2026-09-15_pakd50-s4-w104d121-implementation.md` 및 `config/PAKD50_JQ_W104_D121_WV3_T0_S1234_FRESH50_v1.yaml`. Small backbone, Teacher bridge, architecture init namespace, 고정 계수, 기존 absolute budget을 확인.

**[S7] 코드:** 같은 commit의 `feeders/feeder.py`, `augment` 및 `np2tensor`. Blob `be81736ad90e65ff795b69bfdda9d98fc4dbe680`. 현재 flags의 실제 동작·4 rotation·[-1,1] 처리를 확인. 다른 local revision은 별도 대조가 필요.

Repository: `https://github.com/hojunking/PAN-Crafter-repro`.

새롭게 제안한 내용은 QE50/QEC/QES, train q threshold/cache의 구체화, 신규 다중 seed allocation, QEDGE9의 soft-time policy다. **기존 source에 이 방법들의 성능이 이미 기록돼 있다고 주장하지 않는다.**
