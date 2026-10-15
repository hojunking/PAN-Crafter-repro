# QEGX — W104–D121의 q-edge 선택·soft 상호작용·aligner 전달 경로
## s3·s4 신규 실험 배정 및 구현 인계 — 2026-09-15

**상태:** 실행 설계. 이 파일 및 동반 registry는 학습 완료 보고나 trainer에 바로 넣는 config가 아니다.  
**대상:** s3와 s4. s1·s2·s5의 현재 작업은 변경하지 않는다.  
**기본 규모:** s3 신규 15run + s4 신규 14run = **신규 29run**, 각 run 50K.  
**시간:** 상한 없음. 두 서버 각각 학습 기준값 합이 12시간을 넘는 계획이다. 기존 PAKD50의 50시간·9월 16일 마감과 QEDGE9의 9시간 목표를 새 실험에 상속하지 않는다.  
**공통 Student:** W104–D121, `depth=[1,2,1]`. Teacher는 T0/W112–D123을 그대로 고정한다.  
**주 지표:** 기존 selected **raw-original HQNR**. V64와 구별한다. Exact50K·고정 plateau·fSCC·RR 지표는 함께 보관한다.

## 0. 실행자가 먼저 볼 결정

**s3는 “어떤 loss를 어느 patch에 줄 것인가”, s4는 “그 loss를 어느 모듈까지 보낼 것인가”를 담당한다.**

- s3: 기존 Q12 / soft 제거 XJ / q-edge QE50 / q-edge+soft 제거 QX50를 같은 작은 구조·같은 seed에서 비교한다. 상수·shuffle 대조와 β=0.05 대응쌍도 포함한다.
- s4: QE50에 더해 **U-Net에는 모든 patch의 edge를 주고, aligner에만 q-low edge를 주는 QER50**을 시험한다. 같은 q gate의 late-freeze 조합도 별도 확인한다.
- GT hard는 모든 patch·모든 pixel에서 항상 유지한다. q를 Teacher 전체의 정확도나 soft/hard의 품질 점수로 쓰지 않는다.
- Scharr energy, 건물/식생 분류기, 새 feature/variance KD, 새 Teacher, 새 backbone 탐색을 이 묶음에 섞지 않는다. 기존 signed Scharr **edge loss 연산**은 유지한다.
- 기본 후보·계수·확인 seed를 지금 고정한다. 단일 seed의 최고값에 따라 기본 확인표를 바꾸지 않는다.
- 모든 case는 독립 fresh50K다. `J0 → JQ → QE50`는 실행 순서이지 가중치를 연속 fine-tune하는 과정이 아니다.

### 새 계획이 이전 배정을 대체하는 범위

이번 계획은 s3의 완료된 W112 후속 큐와 s4의 완료된 W104 5개 비교 **이후**의 새 큐다. 진행 중인 run이 있으면 그 run을 마친 뒤 경계에서 넘긴다.

조회한 main `c47cde68aad2a110800b243c1cf8efe3b8028690`에는 QEDGE9 seed1234 대조가 s4에서 **s1으로 이동**한 기록이 있다. 따라서 이전 대화의 “s4에서 QE50/QEC/QES를 수행 중”을 현재 사실로 쓰지 않는다. 이 계획은 s1·s5의 QEDGE9 배정을 다시 이동시키지 않으며, s4의 사용 가능 여부는 이번 사용자 지시를 따른다.

## 1. 근거 snapshot과 최신 판단

### 1.1 조회 범위

이 대화에서 Google Drive로 `pan-cvpr27`을 새로 export해 읽었다. WV3의 s1–s5 PAKD50 결과는 **55행**이다.

| 서버 | 결과 run 행 수 | 이번 설계에서의 사용 |
|---|---:|---|
| s1 | 6 | 기존 Q12·routing의 참고 결과 |
| s2 | 9 | s3와 다른 soft/edge 상호작용의 참고 |
| s3 | 12 | 첨부 고정밀 보고서와 연결 |
| s4 | 18 | W104–D121 완료 5개와 실제 소요시간 |
| s5 | 10 | 기존 LF·routing 결과 참고 |

원값·행 위치·전체 run ID는 `live_sheet_55runs.csv`, workbook hash와 조회 정보는 `source_snapshot.json`에 있다. 이 snapshot에서는 **QE50/QEC/QES의 완료 결과 행이 없다.** 구현 및 cue 자산 생성은 확인되지만, q-edge 방식의 HQNR 개선이 이미 확인된 것은 아니다. Sheet에 행이 없다는 것만으로 서버가 idle이라고 단정하지 않는다.

### 1.2 s4의 작은 구조는 출발점이 높지만, KD 추가 이득은 아직 없다


| Case · seed1234 | Raw selected HQNR | ERGAS | Params(M) | Train(h) | Sheet 행 |
|---|---:|---:|---:|---:|---:|
| NA0 | 0.9580 | 2.0879 | 1.9036 | 1.27 | 20 |
| J0 | 0.9585 | 2.1416 | 2.0090 | 1.17 | 21 |
| JQ | 0.9565 | 2.0804 | 2.0090 | 1.38 | 22 |
| XJ | 0.9571 | 2.1079 | 2.0090 | 1.37 | 23 |
| F0 | 0.9574 | 2.0964 | 2.0090 | 1.13 | 24 |

출처: 새 export의 `WV3-s4!B20:X24`. 위 수치는 Sheet의 4자리 저장값이다.

같은 seed의 작은 J0 대비:
- JQ: **−0.0020**
- XJ: **−0.0014**
- F0: **−0.0011**
- J0−NA0: **+0.0005**

따라서 “아키텍처를 줄이니 KD 효과도 커졌다”는 결론은 아직 틀리다. **작은 구조의 GT-only 기준은 유력하지만, 현 Q12/soft/edge 조합을 그 위에 그대로 더하면 오히려 낮아진 상태**다. 새 gate는 JQ만 이기는 것이 아니라 **같은 구조의 J0와 단순 대조에도 가까워지거나 이를 넘어야** 한다.

W112의 J0 v1/v2, JQ v1/v2를 유리한 쪽으로 사후 선택해 W104와 비교하지 않는다. 이번 s3 신규 W104 block과 s4의 같은 release 대조가 새로운 판단 단위다.

### 1.3 s3 보고서: 단독 loss보다 상호작용이 크다

첨부 `2026-09-15_s3_pakd50-results.md`의 고정밀 값은 전부 W112–D123, Student seed2026의 단일 run이다.

| Case | Raw selected HQNR |
|---|---:|
| J0 | 0.95529 |
| JR | 0.95517 |
| JQ = R3 + all-edge | 0.95536 |
| XJ = R1 + all-edge | **0.95778** |
| J_R3_NOEDGE | **0.95750** |
| J_N0_EDGE | 0.95648 |
| LF0 | 0.95491 |
| LFQ | **0.95759** |
| LFX | 0.95737 |

관측된 변화:
- EDGE 추가: N0 위 **+0.00119**, R1 위 **+0.00261**, R3 위 **−0.00214**.
- LF: Q12 위 보고서 기록 **+0.00224**, N0 위 −0.00038, X02 위 −0.00040.
- 표시값으로 계산한 soft×edge 상호작용은 약 **−0.00475**다:
  `H(JQ) − H(XJ) − H(J_R3_NOEDGE) + H(JR)`.

이 때문에 q-edge 선택을 **“edge가 무조건 좋으니 더 준다”가 아니라, soft와 edge를 항상 같은 patch에 동시에 주는 것이 필요한가**라는 질문에 연결한다. 상호작용이 원인이라고 확정한 것은 아니며, 새 2×2 비교에서 직접 검증한다.

보고서는 일부 완료 run의 `best_hqnr == max_hqnr`이며 궤적 변동이 중요함을 확인했다. 따라서 이전 중간 문건의 “효과 부호가 다른 것은 selector 때문”이라는 설명을 그대로 사용하지 않는다. Selected·exact50K·plateau는 각각 보존하고, 서로 다른 질문으로 읽는다.

### 1.4 분석 보고서가 새 방법에 허용하는 범위

`q = offset-consistency residual`, `e = Teacher–GT native reconstruction L1`이다. 낮은 q는 추가 변위에 대한 상대 반응이 일관적이라는 뜻이며, native 절대 정합이나 Teacher output 정확도를 보장하지 않는다.

기존 EQREC4 K20에서는 q 배분을 포함한 모든 5K 적응 arm의 HQNR이 출발점보다 낮아졌다. 그 결과를 성공으로 인용하지 않는다. 이번에는 q로 soft 신뢰도를 결정하지 않고 **GT edge의 적용 대상 또는 수신 모듈을 선택**한다. 새 가설을 새 50K 대응 실험으로 검증하는 것이다.

## 2. 고정할 공통 조건

| 항목 | 이번 값 |
|---|---|
| Student | W104–D121, hidden_size 104, depth [1,2,1] |
| 구조 계보 | `model.pancrafter_paper.PANCrafterPaper`, 9ch, plain MS-only, `in_mode=paper`, LN, `attn_locations=[]`, mode modulation 없음 |
| T0 | W112–D123, L1E4 S2025 `best_hqnr`, step24240 |
| T0 checkpoint SHA256 | `16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32` |
| Teacher aligner hash | `493aca181f93ae68` |
| Teacher path | A/U 모두 frozen·eval. 자신의 고정 A로 native PAN을 보정해 forward |
| Student 초기화 | A는 T0 최종 A strict copy. U는 architecture+seed별 저장 fresh 초기값 |
| 복원 좌표 | PAN만 warp. MS base·GT·출력 좌표는 M-frame 유지 |
| Updates / batch | 50,000 / 48 |
| Optimizer | AdamW, WD0.01; 나머지 실제 기존 resolved 설정 유지 |
| LR | U 1e−4 / A 1e−5 |
| Scheduler | warmup100 + 전역 50K cosine |
| α / β / ε | 1.0 / 0.1 / 1e−6; 명시한 β=0.05 case만 예외 |
| τ_R | 0.012463942170143127 |
| λ_E | 0.09075170336956798 |
| λ_off | 1e−4, A trainable인 0-based 홀수 update에 적용 |
| 추가 변위 | b=2 HR px 원판 면적 uniform, PAN 보조 경로만 |
| GT hard | 전 pixel·전 patch에 항상 유지 |
| 후보 격자 | GRID1010_50K_v1, 1010마다 + exact50000, 기존 50후보 |
| 공식 선택 | raw_original HQNR → 기존 fSCC tie 규약 |
| 별도 후반 기록 | exact50000, {45450,46460,47470,48480,49490,50000} 평균 |

T0와 Student의 용량이 다르다. 이 결과를 “동일 용량 fitting”의 검증으로 쓰지 않는다. W104 Teacher를 새로 준비하는 실험은 후속 별도 축이다.

NA0/F0는 기존 s4 W104의 단순 기준으로 보관하지만, 이번 새 routing/q gate 비교의 직접 대조는 **같은 서버·seed·구조의 JQ/QE50/JE0**다. 외부 서버의 높은 점수를 직접 분모로 쓰지 않는다.

τ_R·λ_E를 새 gate나 새 seed마다 재보정하지 않는다. λ_E0의 W112 pilot에서 W104로의 **의도적 고정 이식**이다. QEC3의 c_E3만 별도 상수 대조 계수이며, λ_E calibration을 다시 하는 것이 아니다.

## 3. q·e·gate의 실행 정의

### 3.1 고정 Teacher의 q

\[
q_T(i)=\frac{1}{2K}\sum_{k=1}^{K}
\|c^T_{i,\epsilon_k}+\epsilon_k-c^T_{i,0}\|_1,\qquad K=16.
\]

Bank는 `EQREC4_A_AXIS16`: 반경 {0.25,0.5,1,2} HR pixel × 각도 {0,90,180,270}도다. λ_off를 곱하지 않으며 EPE/L2로 바꾸지 않는다.

\[
g_i=\operatorname{sg}\bigl(\mathbf1[q_T(i)<\theta_q]\bigr),\qquad
\theta_q=\mathbf{0.3276133416220546}.
\]

동일값은 0이다. θq는 기존 train-only calibration 3,072 base×4 augmentation view의 중앙값이며, 이 캠페인에서 FR20 결과를 보고 바꾸지 않는다. q가 낮은 것이 좋은 “건물 classifier”임을 전제하지 않는다.

### 3.2 자산을 그대로 재사용

- Manifest: `assets/qedge9/cue_T0_AXIS16_v1.json`
- asset_id: `526dbd718af54c4a0c3eb5fd81778caa`
- NPZ SHA256: `b83cc5d1a25e8766d1813c9d19d956efc24e0f152553e53c4894fa18dcdf563e`
- 9,714 base patch × 4 실제 augmentation 상태 = 38,856 view.
- 전체 gate 활성 비율: 약 0.499614; calibration에서는 0.5.
- QES: `(e_T_roi32 decile, rot state)` 안에서 permutation seed51515. 각 stratum의 활성 수 보존.
- e_T_roi32는 Teacher native HRMS–GT의 중앙 `[16:48]²`·8밴드 L1이다. 훈련 reconstruction support를 이 ROI로 줄이지 않는다.

현재 feeder는 hflip/vflip 옵션이 켜지면 항상 뒤집고, rot만 0~3 중 무작위인 구현이다. 따라서 현재 cache는 **4상태**다. 이를 일반적인 8개 D4 augmentation으로 조용히 바꾸거나 q를 회전 불변으로 가정하지 않는다. Feeder 개선 자체는 이 캠페인의 변경 항목이 아니다.

### 3.3 기존 e 기반 hard/soft는 유지

\[
e_T(i,p)=\mathrm{mean}_c|T-Y|,\quad
e_S(i,p)=\mathrm{mean}_c|S-Y|,\quad
k(i,p)=\mathrm{mean}_c|S-T|.
\]

\[
d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\qquad
a_T=\operatorname{sg}\frac{[e_S-e_T]_+}{e_S+\epsilon}.
\]

\[
L_H=\langle(1+\alpha d_T)e_S\rangle,\qquad
L_K=\langle\beta(1-d_T)a_T k\rangle.
\]

GT hard가 low-e에서 없어지지 않는다. q를 w_H 또는 w_K에 직접 곱하지 않는다. Teacher target·d_T·a_T·q·g는 detach하고 Student의 실제 hard/soft residual 경로는 유지한다.

### 3.4 Edge와 offset의 소유권

기존 signed Scharr /32, reflect pad, 내부1px, 8밴드 대응으로 구한 sample별 edge를 E_i라 한다.

\[
E(g)=\frac1B\sum_i g_i E_i,\quad E(1)=\frac1B\sum_iE_i.
\]

`sum(g*E)/sum(g)`로 재정규화하지 않는다. 기존 λ_E E(1)과 λ_E E(g)를 동시에 더하지 않는다. q mask는 patch별 scalar이지 pixel별 confidence map이 아니다.

\[
L_O=m_t\,10^{-4}L_{\mathrm{off},S}.
\]

Student offset target은 현재 Student의 `sg(c0_S)`다. 고정 Teacher c0_T로 대체하지 않는다. q_T probe는 학습용 Student offset와 별도이고, q_T의 계산·cache에서 Student parameter에 gradient가 흐르지 않는다.

## 4. 신규 후보의 정확한 수식

### 4.1 기존 Q12와 q-edge 계열

| Case | Reconstruction | Edge | 의미 |
|---|---|---|---|
| JQ | L_H + L_K | λ_E E(1) | all-edge Q12 |
| XJ | L_H | λ_E E(1) | all-edge, soft 없음 |
| QE50 | L_H + L_K | λ_E E(g) | q-low edge |
| **QX50** | **L_H** | **λ_E E(g)** | **같은 q-edge에서 soft만 제거** |
| J_R3_NOEDGE | L_H + L_K | 0 | edge 자체가 없는 단순 KD |

모든 J case에는 L_O를 같은 규칙으로 더한다. `QE50에서 β=0`의 동치는 **QX50**이지 XJ가 아니다. QX50도 d_T를 쓰므로 Teacher-informed hard가 남는다.

s3에서 계산할 핵심 대비:

\[
\Delta_{\mathrm{gate},Q}=H(QE50)-H(JQ),\quad
\Delta_{\mathrm{gate},X}=H(QX50)-H(XJ).
\]

\[
I_{\mathrm{gate}\times\mathrm{soft}}
=[H(QE50)-H(QX50)]-[H(JQ)-H(XJ)].
\]

JQ만 나쁘고 QE50이 회복돼도 그것만으로 최종 최적이라고 하지 않는다. J0/XJ/J_R3_NOEDGE 대비 최종 HQNR을 함께 본다.

### 4.2 QEC3 — s3의 강도 대조

s3의 **J0/W104D121/seed2026/exact50K**를 공통 고정 pilot으로 사용한다.

\[
c_{E3}=
\frac{\sum_{i\in\mathcal D_{\rm cal}}g_iE_i^{\rm pilot}}
{\sum_{i\in\mathcal D_{\rm cal}}E_i^{\rm pilot}},\qquad
L_{\rm QEC3}=L_H+L_K+\lambda_E c_{E3}E(1)+L_O.
\]

여기서 calibration view와 gate는 기존 asset과 같다. Pilot hash·tag·step을 c_E3 자산에 넣고 확인한다. denominator 퇴화 시 중단하며 0.5로 대신하지 않는다.

**QEC3는 s1의 QEDGE9 QEC pilot/c_E 파일을 덮어쓰지 않는다.** 새 `work_dir/_qegx/qec3_cE.json` 등 별도 경로를 사용한다. 이 대조는 pilot의 평균 edge loss를 맞추는 것이며, 모든 학습 시점의 A/U gradient 크기를 같게 보장하지 않는다.

### 4.3 QES

QE50의 g만 기존 조건부 shuffled label `g_shuffle`로 바꾼다. e decile 및 실제 augmentation state를 보존한다. QES가 존재한다는 사실로 특정 의미적 scene type을 통제했다고 쓰지 않는다.

### 4.4 β=0.05 대응쌍

- `J_QB005`: JQ에서 β만 0.1→0.05.
- `QE50_B005`: QE50에서 β만 0.1→0.05.

α·τ_R·λ_E·θq·A LR·학습량은 같다. “soft가 작아 보여서 늘리자”가 아니라, s3 및 s4 일부 block의 all-edge+soft 열세를 근거로 **완전 제거(β=0), 절반(0.05), 기본(0.1)**을 좁게 비교한다. 0.05가 최적이라고 사전 확정하지 않는다.

### 4.5 QER50 — loss 선택과 모듈 선택을 구별하는 핵심 신규안

QE50은 U-Net과 aligner 모두 q-low patch의 edge만 받는다. QER50은 **U에는 모든 patch의 GT edge, A에는 q-low edge만** 전달한다.

\[
G_U=\nabla_{\theta_U}[L_H+L_K+\lambda_E E(1)],
\]

\[
G_A=\nabla_{\phi_A}[L_H+L_K+\lambda_E E(g)+L_O].
\]

이는 parameter block별 gradient 명세다. 두 식을 그대로 더해 하나의 total loss로 backward하면 reconstruction이 이중 계산되므로 금지한다.

| Case | U가 받는 edge 계수 | A가 받는 edge 계수 |
|---|---:|---:|
| JQ | 1 | 1 |
| JE0 | 1 | 0 |
| QE50 | g | g |
| **QER50** | **1** | **g** |
| QERS | 1 | g_shuffle |

비교:
- QER50−JQ: q-high patch에서 A로 가는 edge gradient를 막는 효과.
- QER50−JE0: q-low patch의 A edge gradient를 허용하는 효과.
- QER50−QE50: A의 edge gate는 같게 두고 q-high patch의 U edge 감독을 복구하는 효과.
- QER50−QERS: 실제 q-sample 연결을 쓰는 효과.

**QER50는 q_T를 A의 직접 정답으로 모방하지 않는다.** 여전히 GT reconstruction과 GT edge이며, q_T는 수신 경로만 고른다. 고정 Teacher input/output은 변경하지 않는다.

현재 registry는 `edge_gate + routing`을 거부한다. QER50를 이 보호 검사를 끄고 우회해서는 안 된다. **`edge_route` 등 별도 명시 경로**를 구현하고, U의 edge는 all, A의 edge는 g라는 동치를 검사한 뒤 등록한다. 기존 scalar `routing.qE`에 배치 tensor를 억지로 넣지 않는다.

### 4.6 LFQE50

0–24,999는 QE50와 동일한 학습, 25,000부터 A를 완전히 동결한다. U는 **같은 gated edge와 Q12 reconstruction으로 50K까지** 학습한다.

- Freeze 이후 A parameter `requires_grad=False`, `grad=None`; AdamW weight-decay drift가 없어야 한다.
- Offset 보조학습은 freeze 이후 중단하되 ε RNG의 소비 규약은 기존 LF와 같게 유지한다.
- U optimizer·scheduler를 재시작하지 않는다.
- q_T와 θq는 그대로다. Student A의 동결 시점에 q gate를 다시 만들지 않는다.
- Exact25K를 다른 checkpoint로 대신하지 않는다. 기본은 독립 full50K다.

s4에서 다음 상호작용을 본다.

\[
I_{\rm LF\times gate}
=[H(LFQE50)-H(LFQ)]-[H(QE50)-H(JQ)].
\]

LF0는 no-KD schedule 대조이고 LFX는 LF 안에서 soft를 제거한 대조다. 첫 seed의 LF 파트는 탐색이며, 채택하려면 §8의 LF 확인 묶음을 추가한다.

## 5. s3 고정 실행 큐 — 15run

**역할:** W104 위의 soft×q-edge 효과와 좁은 β 조정을 확인한다.  
**탐색 seed:** 2026. **사전 고정 확인 seed:** 4321.  
**모든 신규 run:** W104–D121 / T0 / fresh50K. s3의 기존 W112 12개는 재실행하지 않는다.


| 순서 | 단계 | Seed | Case | U / A의 edge 계수 | 시간 기준(h) | 예약(h) | 질문 |
|---:|---|---:|---|---|---:|---:|---|
| 1 | A | 2026 | `J0` | `0` / `0` | 1.16 | 1.4427 | 동일 구조·seed의 joint GT-only 기준 |
| 2 | A | 2026 | `JQ` | `1` / `1` | 1.34 | 1.6407 | 기존 Q12 기준 |
| 3 | A | 2026 | `QE50` | `g` / `g` | 1.50 | 1.8167 | q-low patch만 edge 감독 |
| 4 | A | 2026 | `XJ` | `1` / `1` | 1.33 | 1.6297 | all-edge에서 output soft 제거 대조 |
| 5 | A | 2026 | `QX50` | `g` / `g` | 1.50 | 1.8167 | 같은 q-edge gate에서 output soft 제거 |
| 6 | A | 2026 | `J_R3_NOEDGE` | `0` / `0` | 1.30 | 1.5967 | edge 없는 reconstruction KD 기준 |
| 7 | A | 2026 | `QEC3` | `c_E3` / `c_E3` | 1.34 | 1.6407 | s3 고정 pilot의 평균 edge loss 강도에 맞춘 상수 대조 |
| 8 | A | 2026 | `QES` | `g_shuffle` / `g_shuffle` | 1.50 | 1.8167 | e-decile/augmentation 조건부 q-label shuffle |
| 9 | B | 2026 | `J_QB005` | `1` / `1` | 1.34 | 1.6407 | all-edge에서 beta 0.05 |
| 10 | B | 2026 | `QE50_B005` | `g` / `g` | 1.50 | 1.8167 | 같은 beta 0.05에서 q-edge 선택 |
| 11 | C | 4321 | `J0` | `0` / `0` | 1.16 | 1.4427 | 동일 구조·seed의 joint GT-only 기준 |
| 12 | C | 4321 | `JQ` | `1` / `1` | 1.34 | 1.6407 | 기존 Q12 기준 |
| 13 | C | 4321 | `XJ` | `1` / `1` | 1.33 | 1.6297 | all-edge에서 output soft 제거 대조 |
| 14 | C | 4321 | `QE50` | `g` / `g` | 1.50 | 1.8167 | q-low patch만 edge 감독 |
| 15 | C | 4321 | `QX50` | `g` / `g` | 1.50 | 1.8167 | 같은 q-edge gate에서 output soft 제거 |

### 단계의 의미

- A의 첫 6개가 핵심 방법 대조다. QE50를 일찍 넣어 새 cue의 작동을 바로 확인한다.
- QEC3는 J0 exact50K pilot 생성 후 실행 가능하다. QES는 공통 cache만 있으면 된다.
- B의 두 run은 대응 β 조정이다. QE50_B005 하나만 실행하지 않는다.
- C의 다섯 run은 결과를 보고 case를 고르는 WIN 확인이 아니라 **사전에 고정한 factorial 반복**이다. QX50의 새 seed에서도 XJ가 있어야 output soft 효과를 비교할 수 있다.

s5의 seed2026·777 QEDGE9 결과가 나오더라도 그것을 s3 로컬 control로 대체하지 않는다. 같은 seed2026의 s3/s5는 새 독립 seed가 아니라 환경·실행 반복이다. 비교 가능성의 근거는 같은 seed라는 이름뿐 아니라 init/Teacher/data/actual augmentation/metric/config에 있다.

## 6. s4 고정 실행 큐 — 신규 14run + 기존 검증된 대조 재사용

**역할:** q를 edge loss의 적용 영역에 써야 하는지, A gradient의 수신 제한에 써야 하는지 구분한다. LF 결합도 별도 탐색한다.  
**탐색 seed:** 1234. **사전 고정 확인 seed:** 3407.  
기존 작은 구조 NA0/J0/JQ/XJ/F0는 완료 기록이 있으므로 먼저 검증해 재사용한다.


| 순서 | 단계 | Seed | Case | U / A의 edge 계수 | 시간 기준(h) | 예약(h) | 질문 |
|---:|---|---:|---|---|---:|---:|---|
| 1 | A | 1234 | `QE50` | `g` / `g` | 1.50 | 1.8167 | q-low patch만 edge 감독 |
| 2 | A | 1234 | `LF0` | `0` / `0` | 1.17 | 1.4537 | 작은 모델의 25K 이후 A 동결 GT-only 기준 |
| 3 | A | 1234 | `LFQ` | `1` / `1 if t<25000 else 0` | 1.38 | 1.6847 | 작은 모델의 LF Q12 기준 |
| 4 | A | 1234 | `LFQE50` | `g` / `g if t<25000 else 0` | 1.50 | 1.8167 | q-edge gating × late-freeze 상호작용 |
| 5 | A | 1234 | `LFX` | `1` / `1 if t<25000 else 0` | 1.37 | 1.6737 | LF 안에서 output soft 제거 |
| 6 | A | 1234 | `JE0` | `1` / `0` | 1.80 | 2.1467 | U에는 all-edge, A에는 edge를 전혀 전달하지 않는 routing 대조 |
| 7 | A | 1234 | `QER50` | `1` / `g` | 1.80 | 2.1467 | U all-edge, A q-low edge만 전달 |
| 8 | A | 1234 | `QERS` | `1` / `g_shuffle` | 1.80 | 2.1467 | routing의 실제 q-sample 연결 대조 |
| 9 | C | 3407 | `J0` | `0` / `0` | 1.17 | 1.4537 | 동일 구조·seed의 joint GT-only 기준 |
| 10 | C | 3407 | `JQ` | `1` / `1` | 1.38 | 1.6847 | 기존 Q12 기준 |
| 11 | C | 3407 | `QE50` | `g` / `g` | 1.50 | 1.8167 | q-low patch만 edge 감독 |
| 12 | C | 3407 | `JE0` | `1` / `0` | 1.80 | 2.1467 | U에는 all-edge, A에는 edge를 전혀 전달하지 않는 routing 대조 |
| 13 | C | 3407 | `QER50` | `1` / `g` | 1.80 | 2.1467 | U all-edge, A q-low edge만 전달 |
| 14 | C | 3407 | `QERS` | `1` / `g_shuffle` | 1.80 | 2.1467 | routing의 실제 q-sample 연결 대조 |

### s4 기존 control의 재사용 전제

재사용할 최소 직접 control은 W104 seed1234의 J0와 JQ다. XJ는 LFX와의 비교에 필요하고, NA0/F0는 단순 최종 기준으로 보존한다.

검증할 항목:
1. T0 A/U checkpoint와 cue binding, Student U init, data/feeder/sampler/normalization, LR·50K·candidate grid가 일치한다.
2. `return_meta` 활성화가 입력 tensor와 augmentation RNG를 바꾸지 않는다.
3. 기존 run이 완주했고 expected_step=50K 및 후보 50개가 있으며, resumed run이었다면 데이터열 계약을 확인한다.
4. 코드 차이가 loss/gradient/optimizer 경로를 바꾸지 않는지 현재 구현 검사와 연결한다.

전제가 충분하지 않으면 **J0/JQ를 같은 새 release에서 v2 등 새 이름으로 다시 수행**하고 둘을 control로 쓴다. 높은 옛 값만 고르지 않는다. 이 경우 §9의 **약 3시간 8분**을 추가한다. XJ도 직접 비교의 전제가 맞지 않으면 같은 방식으로 별도 추가하며, 실제 case 시간으로 계산한다.

s4의 C에서는 J0/JQ부터 새 seed3407로 실행하므로 기본 routing 판정은 두 seed에 걸쳐 닫을 수 있다. LFQ/LFQE50의 새 seed 확인은 별도 §8에 있다. LF의 단일 seed 결과를 최종 채택으로 보고하지 않는다.

## 7. s3·s4가 같은 질문을 중복 실행하지 않는 이유

| 질문 | 주 담당 | 필요한 직접 대조 |
|---|---|---|
| q-low에서만 edge를 주는 것이 유효한가? | s3, s4 local anchor | QE50−JQ |
| Gate 안에서도 soft가 필요한가? | **s3** | QE50−QX50, JQ−XJ |
| q가 단순 edge 강도 감소 이상을 주는가? | **s3** | QE50−QEC3, QE50−QES |
| β를 절반으로 줄이면 soft/edge 결합이 달라지는가? | **s3** | J_QB005/JQ 및 QE50_B005/QE50 |
| q-high의 edge를 U에서까지 끌 필요가 있는가? | **s4** | QER50−QE50 |
| q가 A gradient routing의 cue로 유효한가? | **s4** | QER50−JQ/JE0/QERS |
| q-edge와 후반 A 동결이 대체재인가 보완재인가? | **s4** | LFQE50−LFQ 대 QE50−JQ |

공통 QE50/JQ/J0를 두 서버에 두는 것은 각 서버 내부의 causal contrast를 만들기 위한 대조다. 서버가 다르다는 이유로 같은 seed를 독립 seed로 늘려 세지 않는다.

고유 Student seed는 기본 계획에서 **2026·4321·1234·3407**다. T0는 한 package를 공유하므로 전체 Teacher 사전학습을 독립 반복한 결과가 아니다. QX50와 QER50는 각각 해당 서버의 두 seed에서 평가되며, 모든 신규 방법을 4seed로 평가했다고 쓰지 않는다.

## 8. 결과에 따른 후속 묶음 — 자동 전수 sweep 아님

기본 29run은 고정이며, 아래는 추가 시간 상한 없이 선택해 확장한다. 성능을 억지로 만들기 위해 모든 계수를 동시에 바꾸지 않는다.

### 8.1 LF의 다른 seed 반복 — 우선 선택 가능한 확장

s4 seed1234에서 LFQE50가 LFQ/QE50보다 유력하면, **seed3407의 LF0/LFQ/LFQE50 3run**을 추가한다. 같은 seed의 J0/JQ/QE50는 기본 C에서 확보된다. 기준 4.05h, 예약 **4.955h = 4시간 57분 18초**다.

LFX만 유력하면 LF0/LFQ/LFX로 검증하며, q gate 효과와 혼동하지 않는다.

### 8.2 β=0.05 확인

s3의 β=0.05 대응쌍이 유력하면 seed4321의 **J_QB005/QE50_B005 2run**을 추가한다. 기존 seed4321 JQ/QE50와 비교한다. 기준 2.84h, 예약 **3.4573h ≈ 3시간 27분**.

### 8.3 e cutoff를 추가하는 간단한 변형

기본 e 연속 가중은 그대로 두고, patch별 Teacher 오차가 낮을 때만 output soft를 허용하는 별도 가설:

\[
s_i=\mathbf1[\bar e_{T,\mathrm{ROI32}}(i)\le\theta_e],
\quad \theta_e=0.016400586813688278.
\]

- `ES50`: JQ의 L_K에만 s_i를 곱한다. All-edge.
- `ESQE50`: QE50의 L_K에만 s_i를 곱한다. q-low edge.

**GT hard는 기존 L_H 그대로 모든 곳에 남는다.** e threshold는 픽셀 τ_R와 다르다. soft cutoff와 hard 이진 재가중을 동시에 바꾸지 않는다.

첫 pair는 s3 seed2026에서, 유력하면 동일 pair를 4321에서 반복한다. “e가 높으면 soft를 끄는 것이 맞다”가 이미 입증된 것으로 쓰지 않는다. β 감소만의 효과와 구별해야 한다.

### 8.4 q의 방향·범위와 threshold 민감도

기본 θq=median을 바꾸기 전에 QE50/QEC3/QES의 차이를 먼저 본다.

- `QEHI50`: edge gate를 정확히 `1-g`로 바꾼 방향 반증 대조.
- `QE25`: 기존 q cache의 train calibration 25분위 아래에서만 edge.
- `QE75`: 75분위 아래에서 edge.

세 case는 q값을 다시 추정하는 것이 아니라 새 threshold/child-manifest로 선택 범위를 바꾼다. 기존 asset의 θq나 gate 배열을 덮어쓰지 않는다. Low-q 방향이 항상 맞다는 전제를 두지 않는다.

### 8.5 Routing의 평균 계수 대조

`QERC05`: U edge=1, A edge=0.5. QER50와 비교해 단순한 A edge coefficient 감소의 가능성을 확인한다. 0.5는 calibration gate 활성 비율과 같은 **평균 계수 대조**이지 parameter-gradient norm을 정확히 맞춘 값은 아니다. QERS는 별도의 내용 대응 대조로 유지한다.

### 8.6 이 계획에서 뒤로 미루는 것

새 Teacher, W104 Teacher 학습, 다른 센서, attention 재도입, dense aligner, λ_off 광범위 sweep, online Student-q gate, q를 soft 품질 점수로 사용, 새 output variance/feature KD는 기본·직접 후속 묶음에서 제외한다. 작은 모델+선택적 edge의 효용이 확인된 다음 별도 질문으로 연다.

## 9. 시간 계산 — 실제 값과 대용값을 분리

### 9.1 이번에 읽은 실제 Train(h)

| 서버 | Case | 구조 | Sheet Train(h) |
|---|---|---|---:|
| s3 | J0 / JQ / XJ | W112–D123 | 1.16 / 1.34 / 1.33 |
| s3 | J_R3_NOEDGE / J_N0_EDGE | W112–D123 | 1.30 / 1.16 |
| s3 | LF0 / LFQ / LFX | W112–D123 | 1.13 / 1.31 / 1.31 |
| s4 | NA0 / J0 / JQ / XJ / F0 | **W104–D121** | **1.27 / 1.17 / 1.38 / 1.37 / 1.13** |
| s5 | PQ | W112–D123 | 1.53 — routing 비용 참고, s3/s4 실측 아님 |

s3 W104는 이 snapshot에서 미측정이다. s3의 같은 유형 W112 시간을 대용하고, 모델 크기가 줄었다는 이유로 미리 시간을 줄이지 않는다. s4는 가능한 항목에서 **이미 측정된 W104 시간**을 사용한다.

신규 cached-gate run은 **1.50h**, s4 routing run은 **1.80h**를 임시 기준으로 사용한다. 둘 다 관측값이나 보장 상한이 아니다. 첫 해당 run 완료 후 `case@architecture@server`별 실제 시간으로 갱신한다.

### 9.2 예약식

\[
T_{\rm reserve}(r)=1.10\,T_{\rm ref}(r)+10/60\ {\rm h}.
\]

10%는 실행 변동, run당10분은 export/기본 후처리/gspread/큐 전환을 위한 가예약이다. 정기 평가가 들어 있는 Train(h)에 평가50회 비용을 다시 더하지 않는다.

1.38h는 1시간38분이 아니라 **1시간22분48초**다. 아래 값은 소수 시간으로 합산한 뒤 시·분·초로 변환했다.


| 서버 | 신규 core | 학습 기준값 합 | 여유 포함 예약 | 단계별 예약 |
|---|---:|---:|---|---|
| **s3** | **15run** | **20.64h** | **25시간 12분 14초** | A 13:24:01 + B 03:27:26 + C 08:20:47 |
| **s4** | **14run** | **21.77h** | **26시간 16분 49초** | A 14:53:07 + C 11:23:42 |
| 합산 | **29run** | **42.41 서버·시간** | **51.4843 서버·시간** | 병렬 경과시간과 다름 |

동시 준비된 두 서버의 **병렬 경과시간 예약은 최대값인 약 26시간17분**이다. 합계 51시간29분이 아니다. s4 J0/JQ의 대조 새로고침이 필요하면 약 **3시간8분**을 s4에 더해 전체 병렬 예약은 약 **29시간25분**이 된다.

**각 서버 모두 운영 여유를 빼고도 학습 기준값 합이 12시간을 넘는다.** 실제 실행이 더 빨라졌다고 12시간을 채우려고 idle·중복 run을 만들지 않으며, 의미 있는 다음 등록 pair를 선택한다. 시간 상한으로 50K를 자르거나 seed 반복을 생략하지 않는다.



### 9.3 표 밖의 일회성 비용과 현재 작업

- 구현/통합 unit·smoke, local cue verify, s3 QEC3 pilot 계산.
- 현재 실행 중인 기존 run의 **남은 시간**.
- 실패 후 복구·재실행, 외부 분석의 추가 해석 대기.

기존 q 자산 manifest는 s1에서 전체 38,856 view의 빌드가 약 **46.55초**였다고 기록한다. 이는 s1의 관측이지 s3/s4의 비용 보장이 아니다. 자산은 이미 존재하므로 원칙적으로 배포/verify만 하고, 16probe를 50K마다 새로 계산하는 시간표를 사용하지 않는다.

새 campaign에는 hard deadline이 없다. 기존 `PAKD50 campaign_clock.json`은 보존한다. 새 ledger 및 새 branch의 admission만 무제한/경고 정책으로 설정해 다른 서버의 원래 캠페인 시간 규약을 바꾸지 않는다.

## 10. 결과 판정 — 사전에 고정

### 10.1 두 수준을 구분한다

1. **최종 성능:** 같은 server/seed/architecture의 J0·JQ·단순 강한 대조 대비 raw selected HQNR.
2. **방법 귀속:** 같은 gate 아래 soft 제거, 같은 U edge 아래 A routing, 상수/shuffle 대비 차이.

목표는 raw HQNR 0.959–0.960이다. 단일 selected 자산의 달성, 여러 seed의 재현, 방법 평균의 달성을 서로 다르게 기록한다. V64 값으로 목표를 달성했다고 하지 않는다.

### 10.2 항상 보고할 항목

- 원래 selected raw HQNR, selected step, exact50K, 고정 plateau.
- 같은 checkpoint의 공식 Dλ·Ds·FR fSCC와 RR ERGAS/SAM/PSNR.
- 같은 seed의 Δ와 고유 Student seed별 평균·표본sd·개별 부호.
- Full soft loss와 edge loss의 크기, A/U gradient 규모와 gate 활성 비율.
- 비교 run의 release·init·Teacher·cue·dataset/evaluator hash 및 실제 학습시간.

FR HQNR는 scene별 `(1-Dλ_i)(1-Ds_i)`를 계산해 20장 평균한다. 평균 Dλ·Ds를 먼저 곱하지 않는다. Q8 헤더나 RR SCC를 FR fSCC로 바꾸어 인용하지 않는다.

### 10.3 성능의 소폭 차이를 과장하지 않는 규칙

- 두 seed가 모두 개선돼도 작은 표본이다. 0.0031 경험선을 새 방법의 통계적 유의성 검정으로 쓰지 않는다.
- 20 scene·50 checkpoint·여러 서버가 있다고 seed n을 곱하지 않는다.
- FR20은 개발·checkpoint 선택에 이미 반복 사용됐다. 확인 seed가 추가됐어도 untouched test set 결과로 표현하지 않는다.
- ERGAS는 seed와 선택 시점의 변동을 함께 본다. 단일 seed의 작은 악화만으로 즉시 탈락시키지 않되 반복 악화를 삭제하지 않는다.
- q·e·edge L1 감소가 있어도 HQNR로 이어지지 않으면 진단상의 개선으로 남긴다.
- QER50/LFQE50가 JQ만 회복하고 J0/XJ/단순 대조를 넘지 못하면 “취약한 조합의 회복”과 “최종 파이프라인의 추가 개선”을 구분한다.

### 10.4 다음 결정

| 관측 | 결정 |
|---|---|
| QE50−JQ 양수, QX50−XJ도 양수 | q-edge 선택 자체의 효과 후보. 상수/shuffle 차이 확인 |
| QE50−JQ만 양수, QX50−XJ는 null/음수 | soft×edge 결합에 조건부인 효과로 좁힘 |
| QE50≈QEC3, QE50≈QES | q cue보다 강도 감소/일반 mask 효과 가능성. q 신규성 주장 보류 |
| QER50>QE50 및 JQ/JE0/QERS | U의 all-edge를 유지하고 A만 제한하는 정책이 유력 |
| QER50≈JE0 | q 선택보다 A의 edge 차단 자체가 원인일 가능성 |
| LFQE50>LFQ 및 QE50 | late-freeze와 q 선택 결합 후보. 다른 seed LF pair 실행 |
| 작은 J0/NA0가 모든 KD보다 높음 | 작은 구조 baseline을 성능 기준으로 유지. 복잡한 KD를 강제로 채택하지 않음 |
| 방향이 seed마다 반전 | 평균 최고 하나를 고르지 말고 추가 대응 seed 또는 원인 대조로 이동 |

## 11. 구현 인계 — 현재 지원과 새로 필요한 것

### 11.1 이미 확인한 재사용 요소

조회 snapshot `c47cde68aad2a110800b243c1cf8efe3b8028690` 기준:
- W104 registry·Teacher bridge·architecture별 init namespace.
- QE50/QEC/QES 및 `kdv/edge_gate.py`.
- `pa.losses.output_edge_loss_per_sample`.
- `feeders.feeder.PanFeeder(return_meta)`와 q cache lookup.
- J_QB005, JE0, LF0/LFQ/LFX.
- exact resume, verified_complete, cue asset identity 검증 관련 업데이트.

저장소의 “검사 통과” 기록을 s3/s4 CUDA 환경에서 직접 재검증한 결과로 쓰지 않는다. 실행 전 해당 서버에서 다시 통과시킨다.

### 11.2 새 등록과 코드가 필요한 항목

| 항목 | 필요한 변경 |
|---|---|
| QX50 | R1+EDGE-H+low_q backend binding, soft 정확히 0 |
| QEC3 | 기존 const 방식 + s3 J0S2026 exact50K pilot·c_E3 binding |
| QE50_B005 | 기존 low_q edge + rec β=0.05 |
| LFQE50 | LF 일정 + QE50 loss 결합, 경계/RNG/동결 검사 |
| QER50/QERS | **별도 per-sample edge routing 구현**. U full-edge / A selected-edge |
| QEGX campaign | s3/s4 명시 큐, seed 허용, branch/예산/재개/완료 판정 연결 |
| 결과 업로드 | 기존 탭 유지, X열 QEGX 구분, 정확한 architecture/seed/policy/gate/pilot 기록 |

Case 이름을 `extra_priority.txt`에 추가하는 것만으로 구현됐다고 처리하지 않는다. Generator가 지원하지 않는 항목은 실제 config를 만들기 전에 명시적으로 실패해야 한다.

### 11.3 Gradient와 수치 검사

필수 회귀 검사:
1. `g=1`: QE50의 forward loss 및 U/A gradient가 JQ와 일치.
2. `g=0`: QE50가 J_R3_NOEDGE와 일치. GT hard/soft/offset 경로는 살아 있음.
3. `β=0`: QE50→QX50. All-edge에서는 JQ→XJ.
4. Low-e sample에서도 w_H≥1; a_T=0이면 soft=0.
5. QER50에서 `g=1`은 JQ, `g=0`은 JE0와 U/A gradient 각각 동치.
6. Mixed g에서 QER50의 U gradient는 all-edge, A gradient는 gated-edge 기준 autograd와 일치.
7. QERS의 셔플이 gate 개수·strata를 보존하고 training/epsilon RNG를 소비하지 않음.
8. LF의 25K 이전 경로 일치, 정확한 freeze boundary, freeze 이후 A hash 불변·weight decay 없음.
9. A head와 U output의 zero-init 때문에 gradient가 0인 초기화에서 “경로 차단”으로 오판하지 않음. 비퇴화 synthetic state와 실제 몇 update 후 상태 둘 다 검사.
10. CPU/CUDA/AMP dtype, scaler의 scale/unscale, clipping, gradient 합산 순서 동치.
11. Teacher·GT·cue tensors·gate에 gradient 없음. Student HRMS에 gradient 유지.
12. Resume 전후 sample/augmentation/epsilon 열과 optimizer·scheduler·scaler가 연속 실행과 같음.

QER 구현은 선택한 모듈 gradient를 `autograd.grad`로 따로 얻거나 검증된 gradient correction으로 합산할 수 있다. 어느 방식을 쓰든 **optimizer.step은 한 update에 한 번**이다. AMP scale이 다른 gradient를 혼합하거나, clip 전후 순서를 case마다 다르게 하지 않는다.

### 11.4 준비·기동 순서

1. s3/s4 로컬 active pid와 완료 manifest 확인. 현재 run을 죽이지 않고 run 경계 인계.
2. 같은 commit의 고정 worktree로 실행 release 준비. s1/s5의 기존 코드를 pull로 바꾸거나 큐를 강제로 이동하지 않음.
3. T0·cue·data·evaluator·초기값 hash 검사. `python tools/qedge9_cue.py verify`는 현 repo에 존재하는 검증 도구다.
4. 신규 case 등록·unit·smoke. 기존 baseline의 loss/gradient 동치 확인.
5. s4 control 재사용 여부 결정. 불충분하면 대응 J0/JQ 새 버전 추가.
6. s3 J0가 끝나면 c_E3 pilot 계산. QEC3 외 case는 이 pilot을 기다릴 필요가 없음.
7. 별도 새 큐를 건다. 신규 실행용 helper/CLI는 구현 후 명칭·옵션을 확정한다. 이 문서는 존재하지 않는 launch 명령을 실행 가능한 것으로 제시하지 않는다.
8. 첫 QE50/QER50 실제 처리시간으로 reservation 갱신. 실패나 NaN은 예산 무제한이어도 재실행을 자동 반복하지 않음.

QER 구현 준비가 늦으면 s4의 QE50/LF 묶음 및 seed3407 J0/JQ/QE50처럼 준비된 독립 case를 먼저 수행할 수 있다. 이는 성능에 따른 사후 case 변경이 아니라 기술적 의존성에 따른 순서 변경이다. Gate를 끈 상태로 QER라는 이름을 붙여 우회하지 않는다.

## 12. 실행명·큐·gspread·시간 정책

새 campaign id 제안:
`QEGX_A104D121_S3S4_20260915_v1`

부모 PAKD50/QEDGE9는 lineage 정보로 보관하되 admission deadline은 상속하지 않는다. 모든 control도 같은 시간 정책을 받는다.

예시 run ID:
```
PAKD50_QX50_W104_D121_WV3_T0_S2026_FRESH50_v1
PAKD50_QER50_W104_D121_WV3_T0_S1234_FRESH50_v1
PAKD50_LFQE50_W104_D121_WV3_T0_S1234_FRESH50_v1
```

동일 server에서 이미 같은 logical run이 있으면 verified-complete 또는 actual config를 읽어 중복을 막는다. 서로 다른 서버에서 같은 run 이름을 썼다면 `server_id`를 포함한 compound key로 구분한다. v1/v2를 덮어쓰지 않는다.

현재 generator는 QE50 등을 QEDGE9로 자동 분기할 수 있으므로, 신규 계획의 QEGX branch가 **이 자동 규칙에 의해 9시간 ledger/옛 큐로 되돌아가지 않도록** 검사한다. 옛 `--confirm`은 서버별 다른 확인 묶음을 만들 수 있으므로 이번 명시 표를 무조건 대신하지 않는다.

- s3 Sheet: `WV3-s3(5090)`
- s4 Sheet: `WV3-s4`
- 서버 문자열 `s3(5090)`는 내부 server_id s3로 안전하게 정규화하되 표시 탭명은 유지.
- 기존 B:W와 N열 raw HQNR/O열 V64는 유지.
- X열 예: `PAKD50 / QX50 / A104D121 / QEGX / FRESH50`
- Notes: hard_always, q-source/bank/threshold/asset_id, edge_U/edge_A, β, freeze_from, pilot hash, seed, release, time-policy.
- 새로운 Sheet를 직접 만들거나 기존 행을 편집하지 않는다. 기존 gspread uploader를 통해 완료·검증된 결과만 업로드한다.

`case_registry.json`은 이 문서의 **계획용 구조화 명세**다. `time_policy.mode=no_hard_limit` 같은 계획 필드를 기존 trainer schema에 그대로 넣지 말고, 실제 warning-only budget/admission 규약으로 명시적으로 매핑한다. Batch·update·loss 규약은 시간 편성을 위해 바꾸지 않는다.

## 13. 남길 산출물과 최종 인계 양식

각 run에서 최소 저장:
- resolved config, git commit, architecture signature, environment, init/Teacher/A/cue/evaluator/data hash.
- 시작/종료 시각과 Train(h), exact50K checkpoint, selected와 후보 50개.
- scene별 공식 raw HQNR/Dλ/Ds/fSCC, RR 동반 지표.
- q gate 활성 비율, ungated/gated edge loss, hard/soft loss.
- QER는 U all-edge와 A routed-edge 및 각 parameter gradient 규모를 구별.
- 같은 server/seed의 주요 대조 Δ.
- `failure_reason`·`resume_history`·`control_refresh_reason` 등 예외 기록.

결론은 아래 순서로 작성한다:
1. 최종 raw HQNR이 J0/단순 기준보다 개선됐는가?
2. q gate가 Q12 및 상수/shuffle보다 유리한가?
3. 그 효과가 soft 유무·β·A 수신 정책에 따라 달라지는가?
4. 다음 seed에서 같은 방향인가?
5. ERGAS·fSCC·last/plateau 비용은 무엇인가?

## 14. 출처와 해석 범위

| ID | 출처 | 이번 사용 |
|---|---|---|
| S0 | `pan-cvpr27` 이번 Google Drive export, WV3 s1–s5 | 55 run의 selected 지표·실제 Train(h); 동반 CSV/JSON에 row·run mapping |
| S1 | `2026-09-15_s3_pakd50-results.md` §2–5, §7 | 12개 고정밀 HQNR·상호작용·단일 seed 한계·LF/RNG 실행 검증 |
| S2 | `PAN_Aligner_Analysis_Summary_2026-09-15.md` §2–5 | q/e 의미·절대 정합 및 KD 품질과의 구분 |
| S3 | `2026-09-15_s1_eqrec4-results.md` §A, §B K10/K20 | q/e 집단과 보완 분석·negative HQNR pilot |
| S4 | `PAN_KD_Q12_R1_Spec_and_Aligner_Interface_2026-09-14.md` §4–7 | hard/soft·GT edge·reduction·calibration |
| C1 | repo `c47cde68aad2a110800b243c1cf8efe3b8028690`, `tools/gen_pakd50_configs.py` | 기존 cases, 현재 s1/s5 QEDGE9와 s3/s4 큐, 새 branch 연결 필요 |
| C2 | 같은 repo, `assets/qedge9/cue_T0_AXIS16_v1.json` | 실측 θq·cue hash·view 계약·1회 빌드 비용 |
| C3 | 같은 repo, `kdv/edge_gate.py`, `kdv/registry.py`, `pa/losses.py` | mask·shuffle·per-sample edge·routing 동시사용 거부 |
| C4 | 같은 repo, `research_log/2026-09-15_qedge9-implementation.md` | 구현·감사·exact resume·서버 배정 변경 |
| C5 | 같은 repo, `config/PAKD50_QE50_W104_D121_WV3_T0_S2026_FRESH50_v1.yaml` | T0/Student·계수·time policy·feeder·50K 규약 |

별도 서버의 GPU·process·로컬 work_dir·checkpoint bytes를 이 문서 작성 중 직접 확인하지 않았다. 완료 Sheet 행과 코드의 구현 기록을 구분한다. s3 첨부 보고서의 W112 관측을 W104에서 이미 검증된 효과로 옮기지 않는다.

---

**최종 실행 요약:** s3 15run으로 q-edge×soft를, s4 14run으로 q-edge×aligner 수신을 비교한다. 각 서버 12시간 이상이며, 현재 기준 예약은 각각 약 25시간12분·26시간17분이다. Small backbone 자체의 높은 J0를 유지한 채, q가 어느 감독의 어디에 쓰여야 최종 raw HQNR에 도움이 되는지 확인한다.
