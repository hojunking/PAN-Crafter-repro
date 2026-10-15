# s2 · W112–D124: GT 방향·구조 통계·정합 민감성 기반 KD — A/B/C 추가 실험·구현 명세

**작성일:** 2026-09-10  
**문서 버전:** `v0.1-proposal`  
**추가 캠페인 ID:** `S2_W112_TRIABC_20260910`  
**상위 문서:** `PAN_S2_W112_D124_KD_Variance_Adaptive_Campaign_2026-09-10.md`  
**범위:** 앞선 개념 검토의 A·B·C를 구현 가능한 후속 실험으로 구체화한다. 기존 캠페인을 덮어쓰지 않는다.  
**실행 상태:** s2 접속·실제 W112 네트워크 통합·위성영상 학습·성능 검증은 수행하지 않았다. 문서의 초기값과 새 수식은 검증할 설계안이다. 실제 실행자는 아래 계약을 저장소에 구현한 후 진행한다.

> **핵심 질문**  
> 특정 센서·데이터의 GT를 더 깊게 fitting할 때, Teacher의 예측 전체를 따라가는 것이 아니라 **GT에 필요한 수정 방향·구조 관계만 전달**할 수 있는가? 또한 **정합 불확실성에 민감한 Teacher 성분을 불필요하게 강제하지 않을 수 있는가?**

**A:** Student→GT와 Student→Teacher의 수정 방향을 밴드별로 구분한다.  
**B:** 같은 판단을 GT·Teacher·Student의 gradient variance와 공분산 성분에 적용한다.  
**C:** PAN 보정량 변화가 Teacher 출력에 미치는 영향을 측정하고, 그 영향에 민감한 soft 감독만 조절한다.

**계승:** s2, W112–D124, WV3 9ch, M-frame, 연속 가중 hard/soft, Teacher aligner 재사용 우선, frozen/미세조정/scratch 선택지.  
**미계승:** 과거 W96 폭, 11ch, PAN reconstruction, 고정되지 않은 geometry loss의 자동 추가, 독점적인 hard/soft 이진 switch.

**바로가기:** [A: 영상값 방향](#case-a) · [B: 통계 방향](#case-b) · [C: 정합 민감성](#case-c) · [실험 우선순위](#registry) · [검사 목록](#tests) · [설정](#config) · [참고 함수](#reference)

---

## 0. 문서 사용법과 출처의 지위

### 0.1 무엇이 기존 결정이고 무엇이 새 제안인가?

| 구분 | 내용 | 지위 |
|---|---|---|
| 상위 캠페인 | W112·9ch·R3 reconstruction, 통계 표현, aligner 모드, 평가·분기 규칙 | 기존 설계 계승 [S0] |
| 선행연구의 발상 | S/T/GT의 관계를 함께 사용, soft 감독의 이득과 제약을 구분 | 문헌의 개념 [R1–R3] |
| A/B의 sign·cap gate | 회귀값과 구조 통계에 적용하는 구체적 수식 | **이번 구현용 제안** |
| C의 Jacobian·대각 감쇠·저랭크 soft | PAN 보정량의 출력 영향을 KD에 연결 | **이번 구현용 제안** |
| 초기 계수·유한차분 간격·우선순위 | 첫 구현과 대응 실험을 위한 선택 | 검증된 최적 설정 아님 |
| s2의 실제 성능·가중치·처리량 | 현재 자료로 확인되지 않음 | 실행 manifest/로그로 채울 것 |

A/B/C는 **TGeo-KD 또는 Weighted Soft Labels의 공식 팬샤프닝 구현이 아니다.** 본문에서 `TRI-A*` 등의 ID를 사용하고, 논문의 재현 결과로 표기하지 않는다. 기존 앞단 A1/A2/A3와 혼동하지 않도록 새 실험에는 반드시 `TRI-` 접두어를 붙인다.

### 0.2 선행연구에서 가져오는 범위

**TGeo-KD — Less or More From Teacher: Exploiting Trilateral Geometry For Knowledge Distillation, ICLR 2024.** Teacher·Student·GT의 예측 차이와 예측값, 같은 클래스의 Teacher 평균을 이용해 샘플별 융합 비율을 학습한다. 가중치 모델에는 validation GT loss를 외부 목적함수로 하는 bilevel 학습을 사용한다. 여기서 geometry는 **예측 사이의 관계**이지 PAN–MS 공간 정합이 아니다. 본 추가안은 그 관계 관점만 차용하며, 원 논문의 gate network·분류 softmax·class-mean·bilevel 절차를 자동으로 도입하지 않는다. [R1]

**Weighted Soft Labels — Rethinking Soft Labels for Knowledge Distillation: A Bias–Variance Tradeoff Perspective, ICLR 2021.** Soft 감독이 샘플에 따라 bias·variance에 다르게 작용할 수 있다는 문제의식을 참고한다. 공식 분류 코드의 Student/Teacher GT 오차 비율 가중치를 MS 회귀에 그대로 옮기지 않는다. 원 연구의 bias–variance와 영상의 국소 gradient variance는 다른 개념이다. [R2, R3]

**당위성의 범위:** 두 논문은 팬샤프닝의 강한 fitting이 필요하다는 직접적인 증거가 아니다. 본 연구의 가설은 **Teacher가 틀린 성분에 대한 모방 제약을 줄이고, 관측 GT에 필요한 복원 성분을 더 fitting한다**는 것이다. 이 가설의 효용과 신규성은 아래 대조군으로 확인해야 한다.

### 0.3 우선 확인할 상위 자료

- [S0] s2 W112 통합 계획: §§3–6 모델·입력·R3, §§8–11 통계·정합, §§15–18 분기·구현·평가.
- [S1] W96 GT-anchored 원안: R3와 이동량 covariance의 출발점. **폭·서버·입력 프로토콜은 [S0]가 우선**한다.
- [S2] A1–A3 v2: 선택된 aligner의 warp·정규화·A2/A3 감독 계약.
- [S3] Offset consistency 계획: N계열을 선택했을 때의 crop·corruption·상대 offset 관계.
- 이전 대화의 A/B/C 설명: 개념의 범위. 이 문서는 그 설명에 구현 세부를 추가한 것이며 실험 성공을 전제하지 않는다.

---

## 1. 전체 고정 조건과 실험 축

| 항목 | 이번 추가안 |
|---|---|
| 서버 | **s2**. 실제 GPU/VRAM/소프트웨어 버전은 현재 manifest 확인 |
| 복원망 | Teacher·Student 모두 **W112–D124**가 정식 비교 기준 |
| 입력·출력 | WV3 PAN 1ch + bicubic MS 8ch, 최종 HRMS 8ch |
| Residual base | 이동시키지 않은 `M = U4(LRMS)`를 정확히 한 번 더함 |
| 좌표 | 입력 MS 기준, GT·최종 출력 M-frame 고정 |
| Teacher | 정합기와 복원망을 포함한 고정 asset. 모든 대응 run에서 hash 동일 |
| Student aligner | `A-FR`: Teacher 재사용·고정 / `A-FT`: 복사 후 학습 / `A-SC`: scratch |
| 첫 비교 | 같은 frozen aligner로 **loss 효과를 분리**. 최종 frozen 확정 아님 |
| 기존 입력 프로토콜 | `I-A`, `I-N`, `I-NATIVE-TRANSFER` 중 현재 cohort의 것을 계승 |
| GT 감독 | 모든 위치에서 reconstruction hard 유지 |
| Teacher 입력 | Student와 같은 raw PAN/MS·증강·corruption. FT/SC에서는 Teacher 자신의 warp 사용 |
| 제외 | 새 PAN task, LPAN/HPAN 채널, MS band softmax, EMA/mutual, 새 gate MLP |
| 시간 | 상위 계획의 약 24h 중간 검토·연장 허용. 추가안 완주를 24h 내로 보장하지 않음 |

**Width 변경이나 Teacher asset 교체는 본 loss ablation과 분리한다.** W96 Teacher만 있다면 [S0]의 W112 준비/명시적 bridge cohort 분기를 따르고, 묵시적으로 W96→W112 KD를 같은 용량 비교로 보고하지 않는다.

### 1.1 Forward 계약

원 LRMS를 `S_LR`, GT를 `G`, 각 복원 결과를 `S_hat`, `T_hat`으로 표기한다. `S`라는 문자가 LRMS와 Student를 동시에 가리키지 않도록 코드 변수명을 분리한다.

$$
M=U_4(S_{LR}),\qquad \mu_i=A_i(P_{view},M),\qquad
P_i^a=\mathcal W(P_{view},\mu_i),\quad i\in\{T,S\}.
$$
$$
\hat Y_i=M+F_i([P_i^a,M]).
$$

`I-N`의 corrupted step에서는 `P_view = W(P, epsilon)`이다. 현재 설계의 sequential warp를 유지한다. Teacher와 Student에 같은 `epsilon`을 적용하되 각각의 learned correction은 다를 수 있다.

`A-FR`이고 동일 donor·동일 view일 때만 `P_T^a = P_S^a`를 한 번 계산해 공유한다. `A-FT/A-SC`에서는 Student correction을 Teacher에 넣어 Teacher target을 바꾸지 않는다.

### 1.2 Aligner 학습 상태와 A/B/C 적용 가능성

| 모듈 | A-FR | A-FT / A-SC |
|---|---|---|
| A 출력 방향 gate | Student U-Net 학습 | U-Net 및 sampling을 통해 Student aligner 학습 |
| B 출력 통계 loss | Student U-Net 학습 | U-Net 및 Student aligner 학습 |
| C Teacher 민감도 기반 output KD | **사용 가능** | 사용 가능 |
| 기존 직접 이동량 KD | 같은 aligner면 차이 0, 제외 | 별도 선택 가능 |
| A3 PAN–GT geometry / N offset | frozen 입력끼리의 상수이므로 optimizer에서 제외 | 선택된 recipe에 맞춰 유지 |

**C와 기존 이동량 covariance KD는 다르다.** C의 Student 학습 대상은 출력이므로 frozen aligner에서도 가능하다. Teacher의 correction을 수치적으로 perturb해 출력 민감도를 계산하는 것은 Teacher 파라미터를 업데이트하는 작업이 아니다.

---

## 2. 공통 기준: 승인한 R3 reconstruction 유지

위치 `p`는 batch와 공간 위치를 포함하고, `c`는 MS band이다.

$$
e_T(p)=\operatorname{mean}_c|\hat Y_{T,c}(p)-G_c(p)|,
\quad e_S(p)=\operatorname{mean}_c|\hat Y_{S,c}(p)-G_c(p)|.
$$
$$
e_{ST}(p)=\operatorname{mean}_c|\hat Y_{S,c}(p)-\operatorname{sg}(\hat Y_{T,c}(p))|.
$$
$$
d_T=\operatorname{sg}\!\left(\frac{e_T}{e_T+\tau_R}\right),\qquad
a_T=\operatorname{sg}\!\left(\frac{[e_S-e_T]_+}{e_S+\varepsilon_R}\right).
$$
$$
w_H=1+\alpha_Rd_T,\qquad w_K=\beta_R(1-d_T)a_T.
$$
$$
\boxed{L_R^{base}=\langle w_H e_S+w_K e_{ST}\rangle_p.}
$$

[S0]의 초기 제안 `alpha_R=1`, `beta_R=0.1`, `epsilon_R=1e-6`과 native train calibration의 `tau_R`를 계승한다. 실제 parent가 이미 다른 값을 명시적으로 사용하면 그 값을 유지하고 manifest에 적는다. 본 추가안에서 동시에 재조정하지 않는다.

- Teacher/GT/가중치 계산은 detach. 실제 Student–GT 및 Student–Teacher loss는 live.
- Hard는 모든 픽셀·밴드에서 유지한다. 큰 Teacher 오차는 hard 증가, soft 감소로 작용한다.
- Soft는 전체 픽셀 수로 평균한다. gate가 꺼진 위치를 제외한 평균이나 `sum(w_K)` 정규화는 사용하지 않는다.
- 두 모델이 같은 오답이면 soft가 없어도 hard가 남는다.
- Student가 이미 평균적으로 더 정확하면 `a_T=0`이다. 후속 component gate가 이 기본 조건을 저절로 해제하지 않는다.

### 2.1 새 항을 이중으로 넣지 않는 원칙

A와 C는 기본적으로 **기존 reconstruction soft 항을 변경**한다. `L_R_base + L_R_A`처럼 full reconstruction을 두 번 더하지 않는다.

B는 기존 `STAT-GV-AD` 등의 **통계 soft 항을 변경**한다. 기존 GV-AD 전체를 더한 뒤 B 전체를 또 더하지 않는다.

$$
L_{total}=L_{R,\,chosen}+\lambda_VL_{V,\,chosen}+L_{aux,inherited}+L_{shiftKD,optional}.
$$

마지막 이동량 KD는 원래 캠페인의 별도 축이다. A/B/C의 독립 효과를 먼저 볼 때에는 OFF를 권하며, 이미 ON인 parent를 이어가면 모든 대조에 같은 항을 유지한다.

---

## 3. 공통 삼각관계: 방향과 크기를 해석하는 법

어떤 동일 좌표의 표현 `z`에 대해 다음을 계산한다.

$$
u=z_G-\operatorname{sg}(z_S),\qquad
v=\operatorname{sg}(z_T)-\operatorname{sg}(z_S),\qquad
t=z_G-\operatorname{sg}(z_T)=u-v.
$$

`u`는 GT가 요구하는 수정, `v`는 Teacher가 요구하는 수정, `t`는 Teacher가 남긴 오차다. **세 벡터가 대수적으로 독립인 것은 아니다.** 관계를 보존해 사용하는 것이 목적이며, 차이 세 개를 단순 concat했다는 사실을 신규성으로 삼지 않는다.

설명용 제곱거리에서는:

$$
\|z_S+\eta v-z_G\|_2^2-\|z_S-z_G\|_2^2
=-2\eta u^Tv+\eta^2\|v\|_2^2.
$$

`uᵀv>0`이면 작은 연속 이동에서 GT에 가까워질 수 있지만, 너무 큰 `eta`는 지나칠 수 있다. 실제 구현은 L1과 공유 네트워크 파라미터를 사용하므로, 이 식을 Adam 업데이트의 개선 보장으로 읽지 않는다.

**두 가지 중복을 미리 배제한다.**

1. 같은 L2 거리로 Teacher가 더 정확하면 `2uᵀv > ||v||²`가 이미 성립한다. 이 경우 벡터 cosine의 부호만 보는 gate는 상당 부분 중복이다.
2. 한 scalar 성분에서 `|z_T-z_G| < |z_S-z_G|`이면 `u*v>0`이다. 따라서 **성분별 상대오차 우위 gate에 동일 성분 sign gate를 다시 곱한 독립 기여**는 보통 없다. 이를 별도 신규 case로 중복 집계하지 않는다.

이번 주력은 **여러 밴드·구조 성분을 평균한 우위 판단이 숨기는 부분적인 충돌**을 구분하는 것이다.

---

<a id="case-a"></a>

## 4. Case A — GT 방향에 합치하는 reconstruction KD

### 4.1 목적과 변경 위치

기존 `w_H(p), w_K(p)`는 그대로 두고, soft 안에서 어느 band를 따라갈지 추가로 판단한다. U-Net 입력·출력·Teacher·Student 구조는 바꾸지 않는다.

$$
u_{pc}=G_{pc}-\operatorname{sg}(\hat Y_{S,pc}),\quad
v_{pc}=\operatorname{sg}(\hat Y_{T,pc})-\operatorname{sg}(\hat Y_{S,pc}).
$$

### 4.2 TRI-A-COS — 평균 방향 대조

$$
m_{cos}(p)=\left[\frac{u_p^Tv_p}{\max(\|u_p\|\|v_p\|,\varepsilon_{cos})}\right]_{[0,1]}.
$$

norm이 0이면 `m_cos=0`. 두 벡터는 8-band 값이며, scalar gate를 8개 band에 broadcast한다.

$$
L_R^{Acos}=\langle w_H e_S+w_Km_{cos}e_{ST}\rangle.
$$

이는 **방향 정보를 하나의 scalar로 요약한 저비용 대조군**이다. 일부 band의 충돌까지 해결한다고 주장하지 않는다. `epsilon_cos`는 squared-intensity 단위의 수치 하한으로 `epsilon_R²`를 첫 제안으로 두고 FP32에서 계산한다.

### 4.3 TRI-A-SIGN — 주력 최소안

$$
\boxed{m_A(p,c)=\mathbf1[u_{pc}v_{pc}>0].}
$$

수치 구현에서는 곱셈 대신 `(u>0 & v>0) | (u<0 & v<0)`로 같은 부호인지 확인해도 된다. `u=0` 또는 `v=0`이면 0이다. gate는 FP32/no-grad에서 만들고 그 값에 역전파하지 않는다.

$$
\boxed{
L_R^{Asign}=\left\langle w_H e_S+
 w_K\operatorname{mean}_c\{m_A(p,c)|\hat Y_{S,pc}-\hat Y_{T,pc}|\}\right\rangle_p.
}
$$

**이진 hard/soft switch로 되돌아간 것이 아니다.** 이진값은 soft의 특정 band를 허용하는 mask이고, hard는 계속 모든 위치에서 적용된다.

L1 soft의 출력 성분에 대한 음의 gradient는 Teacher 방향의 부호를 가진다. 허용된 성분에서는 그 부호가 GT 방향과 같다. 따라서 이 항 자체가 이미 맞은 band를 Teacher 쪽으로 끌어당기는 상황을 줄일 수 있다. 다만 파라미터 공유·다른 loss·optimizer momentum 때문에 최종 모든 band의 개선을 보장하지는 않는다.

### 4.4 TRI-A-CAP — Teacher가 GT를 지나친 성분의 압력 제한

같은 방향이라도 Teacher가 해당 성분에서 GT보다 훨씬 멀리 지나가 있을 수 있다. 다음을 별도 후보로 둔다.

$$
\boxed{m_{cap}=m_A\min\left(1,\frac{|u|}{|v|+\varepsilon_R}\right).}
$$

A-SIGN의 `m_A`를 이 값으로 교체한다. Gate 전체를 detach한다.

이는 **soft gradient의 계수를 낮추는 방식**이지, 실제 optimizer 이동량을 GT 앞에서 물리적으로 잘라 주는 방식이 아니다. L1에서 soft target을 GT–Teacher 사이로 바꾸는 것과도 다르다. 본 최소안은 target 자체를 바꾸지 않는다.

### 4.5 손계산으로 확인할 예

`G=(0.50,0.50), S=(0.60,0.50)`에 대해:

| Teacher | 기존 평균 Teacher 오차 | m_A | 해석 |
|---|---:|---|---|
| `(0.55,0.50)` | 0.025 | `(1,0)` | 첫 band의 유용한 수정만 전달 |
| `(0.50,0.55)` | 0.025 | `(1,0)` | 첫 band를 참고하되 이미 맞은 두 번째 band는 모방하지 않음 |
| `S`와 같은 값 | 0.050 | `(0,0)` | 같은 오답에 합의하지 않고 GT hard로 학습 |

첫 두 경우의 기본 `d_T,a_T`는 같다. 새 성분 gate는 두 번째 경우의 불필요한 band 수정만 분리한다.

### 4.6 필요한 대조군

| ID | 비교 |
|---|---|
| `TRI-A-BASE` | 기존 R3 그대로 |
| `TRI-A-COS` | R3 + pixel-level cosine soft 감쇠 |
| `TRI-A-SIGN` | R3 + band별 sign soft 감쇠 |
| `TRI-A-CAP` | R3 + band별 sign·크기비 감쇠 |
| `TRI-A-BANDADV` | 기존 scalar w_K를 band별 Teacher 오차·우위로 바꾸는 대조 |
| `TRI-A-MASS` | A-SIGN/CAP과 soft 총계수량만 맞춘 scalar 감쇠 |
| `TRI-A-SHUFFLE` | 새 mask의 공간 대응만 흐트러뜨리는 부정 대조 |

`A-BANDADV`에서는 hard `w_H`를 원래대로 두어 soft의 세분화만 본다. Band별 `tau_R,c`는 native train Teacher 오차에서 별도로 정한다. Scalar 우위가 0이어서 막혔던 유용한 band를 다시 허용할 수 있으므로 A-SIGN과 목적이 다르다. 여기에 sign을 다시 곱한 별도 run은 위의 scalar 중복 관계 때문에 기본 큐에서 제외한다.

**논리적 한계:** A-SIGN/CAP은 원래 `a_T=0`이면 아무 band도 새로 살리지 않는다. 이를 “Teacher가 전체적으로 나빠도 좋은 band를 찾아 전달한다”는 확장과 혼동하지 않는다. 그 질문은 A-BANDADV 또는 B의 성분 우위 비교가 담당한다.

---

<a id="case-b"></a>

## 5. Case B — 구조 통계의 성분별 hard/soft

### 5.1 목적

현재 통계 KD는 band·방향의 오차를 평균해 위치별 gate 하나를 만든다. B는 그 gate를 유지한 상태에서 **Teacher가 특정 band/방향의 구조를 부족하게 또는 과도하게 복원한 성분**을 구분한다. [S0 §§8–9]

처음에는 `STAT-GV`로 시작하고, 같은 메커니즘을 `STAT-GC`, `STAT-SC`로 확장한다. GT brightness variance, output gradient covariance, 이동량 covariance를 같은 변수로 사용하지 않는다.

### 5.2 표현의 정확한 계승

- 최종 HRMS `G/T_hat/S_hat`에서 계산한다. Residual 통계와 동등하지 않다.
- 국소 창 `5×5`, stride1, population variance(`correction=0`). 현재 parent의 확정값이 있으면 그 값 계승.
- Scharr valid convolution 후 local pooling. 기본 통계 margin은 3 HR px이며 train64²에서 통계 중심은58².
- IV/SC도 동일한 통계 중심을 사용한다. Recon full-domain과 통계 ROI를 혼동하지 않는다.
- 각 output을 독립 z-score해 분산 차이를 지우지 않는다. 음수 roundoff만 clamp하고 공분산의 실제 음수 off-diagonal은 보존한다.

$$
V^{GV}(Z)=\{\operatorname{Var}_{5}(K_y*Z_c),\operatorname{Var}_{5}(K_x*Z_c)\}_{c=1}^{8}.
$$

`[dy의 8개 band, dx의 8개 band]` 순서, shape `[B,16,Hv,Wv]`를 유지한다.

GC는 band별 중심화 2×2 gradient covariance의 모든 원소 `[cyy,cyx,cxy,cxx]`, 총32채널을 사용한다. SC는 local 8×8 spectral covariance의 row-major 64채널이다. 기존 all-entry mean L1과 다른 Frobenius reduction으로 몰래 바꾸지 않는다.

### 5.3 통계 hard/soft 기준

`z_G=V(G), z_T=V(T_hat), z_S=V(S_hat)`로 두고:

$$
E_T^V=\operatorname{mean}_j|z_T-z_G|,\quad
E_S^V=\operatorname{mean}_j|z_S-z_G|,\quad
E_{ST}^V=\operatorname{mean}_j|z_S-z_T|.
$$
$$
d_T^V=\frac{E_T^V}{E_T^V+\tau_V},\qquad
a_T^V=\frac{[E_S^V-E_T^V]_+}{E_S^V+\varepsilon_V}.
$$
$$
w_H^V=1+\alpha_Vd_T^V,\qquad
w_K^V=\beta_V(1-d_T^V)a_T^V.
$$

Hard/soft의 기본 가중치와 `tau_V/epsilon_V/lambda_V`는 [S0 §9]의 representation별 calibration을 계승한다. `tau_R`를 통계에 재사용하지 않는다.

### 5.4 TRI-B-SIGN / TRI-B-CAP

$$
u^V=z_G-\operatorname{sg}(z_S),\qquad v^V=\operatorname{sg}(z_T)-\operatorname{sg}(z_S).
$$
$$
m_B(p,j)=\mathbf1[u^V_{pj}v^V_{pj}>0],
\quad
m_{Bcap}=m_B\min\left(1,\frac{|u^V|}{|v^V|+\varepsilon_V}\right).
$$
$$
\boxed{
L_V^B=\left\langle w_H^V E_S^V+
 w_K^V\operatorname{mean}_j\{m_B|z_S-z_T|\}\right\rangle_p.
}
$$

`B-CAP`에서는 `m_B`를 `m_Bcap`으로 바꾼다. **GT 통계 hard는 그대로 유지**하고 Teacher 통계의 일부 성분만 제한한다. R3의 영상값 hard/soft도 함께 유지한다.

$$L_{total}^{B}=L_R^{base}+\lambda_VL_V^B+L_{aux,inherited}.$$

### 5.5 통계 공간에서 기대하는 동작

| 관계: scalar 구조 성분 | 의미 | B의 soft 동작 |
|---|---|---|
| `V_T < V_S < V_G` | Teacher가 필요한 경계를 더 평탄화 | 방향 mask0, GT로 fitting |
| `V_S < V_T < V_G` | Teacher가 유용한 중간 구조를 보유 | 기본 우위 gate가 허용하면 전달 |
| `V_G < V_S < V_T` | Teacher가 더 과장 | 방향 mask0, GT 수준으로 감소 |
| `V_S = V_G ≠ V_T` | Student 통계가 이미 정확 | 해당 soft0 |
| `V_S = V_T ≠ V_G` | 같은 통계 오답 | soft0이어도 GT hard 유지 |

이 표만으로는 기존 scalar AD를 넘어서는 기여가 되지 않는다. **여러 band/방향이 서로 다른 관계에 놓인 실제 패치에서 평균 gate와 다른 유효한 동작이 발생하는지**가 핵심이다.

### 5.6 유용한 부분 지식의 복구: TRI-B-COMPADV

기존 평균 통계 우위 `a_T^V=0`인 위치에서도 일부 성분의 Teacher가 더 좋을 수 있다. 이를 별도로 확인하려면:

$$
d_{T,j}^V=\frac{|z_{T,j}-z_{G,j}|}{|z_{T,j}-z_{G,j}|+\tau_{V,j}},
\quad a_{T,j}^V=\frac{[|z_{S,j}-z_{G,j}|-|z_{T,j}-z_{G,j}|]_+}{|z_{S,j}-z_{G,j}|+\varepsilon_{V,j}}.
$$
$$
L_V^{comp}=\left\langle w_H^VE_S^V+
\operatorname{mean}_j\{\beta_V(1-d_{T,j}^V)a_{T,j}^V|z_{S,j}-z_{T,j}|\}\right\rangle.
$$

Hard는 scalar parent 그대로 두고 soft만 바꾼다. Component calibration manifest가 추가된다. 이 상대우위 조건 자체가 같은 성분의 방향 일치를 이미 내포하므로 sign을 다시 곱한 것을 추가적인 두 번째 메커니즘으로 세지 않는다.

**주력 시작은 B-SIGN, 다음 후보는 B-CAP 또는 COMPADV**다. Component tau 변경과 sign 변경을 동시에 했다고 한 가지 요인의 효과로 해석하지 않는다.

### 5.7 Gradient/분광 공분산으로 확장

`TRI-B-GC-*`, `TRI-B-SC-*`는 동일 수식을 서로 다른 `V(Z)`에 적용한다.

- GC: 가로·세로 구조 변화의 공동 성분.
- SC: 서로 다른 MS band가 국소 공간에서 함께 변하는 관계.
- 대칭 off-diagonal을 두 번 펼치는 기존 reduction을 유지한다. 중복 원소의 sign gate도 동일해야 한다.
- 각 원소의 target을 독립적으로 재조립한 새 covariance matrix를 만들지 않는다. 실제 영상에서 계산된 `V_S`와 원래 `V_G/V_T`를 loss로 비교한다.
- 성분별 weighting이 좌표축에 의존한다는 한계가 있다. 이를 주성분 방향에 대한 불변 감독이라고 부르지 않는다.

GT eigenbasis를 이용한 mode-wise 비교는 후속 선택지다. 이 경우 basis는 GT에서만 계산·detach하고 S/T/G를 같은 basis로 변환해야 한다. 중복 고유값·약한 eigengap에서는 basis가 불안정하므로 그대로 mode gate를 사용하지 않는다. **이번 첫 구현에는 eigenbasis 추가를 요구하지 않는다.**

### 5.8 B에서 필수로 점검할 실패 형태

- Noise/ringing으로 variance만 맞추고 실제 경계 위치가 틀림.
- 평탄한 Student 출력에서 variance의 미분도0에 가까워 통계 감독이 처음에는 작동하지 않음. 이때 plain reconstruction이 탈출 신호이며 거대한 lambda로 해결하지 않음.
- 공분산의 off-diagonal 음수를 잘못 clamp하여 band 관계를 잃음.
- 큰 창에서 서로 다른 물체가 섞여 잘못된 평균 구조를 학습.
- GT 통계만으로 충분한데 Teacher 통계가 불필요한 제한으로 작용.

**B는 별도의 정합 정답이 아니다.** 통계 위치·방향은 관련이 있지만, variance만으로 올바른 고해상도 배치를 유일하게 결정할 수 없다.

---

<a id="case-c"></a>

## 6. Case C — 정합 민감성에서 비롯되는 soft 감독 위험을 구분

### 6.1 대상과 주장 범위

C는 Teacher–Student 출력 차이 중 **PAN 보정량이 조금 달라졌을 때 쉽게 변할 수 있는 성분**에 강한 모방을 덜 요구한다. GT hard를 약화하지 않는다.

그러나 민감한 방향과 실제 오차 원인이 같다는 보장은 없다. 따라서 “정합 때문에 생긴 오류를 완전히 분리했다”보다 **“Teacher의 정합량에 민감한 출력 성분을 별도로 취급한다”**로 주장한다.

공유 frozen aligner일 때 S/T가 다른 실제 이동량을 쓰는 것은 아니다. 그래도 그 공통 이동량이 부정확할 가능성과 두 복원망의 민감도 차이는 남는다. C는 이를 고려하는 **Teacher target의 안정성 제약**이지 Student aligner 재학습이 아니다.

### 6.2 Teacher 출력 함수: PAN sampling만 독립적으로 변화

현재 raw view와 MS를 고정하고:

$$
z_T(\delta)=\Phi\!\left(M+F_T([\mathcal W(P_{view},\delta),M])\right).
$$

첫 `Phi`는 identity, 즉 HRMS 값이다. 후속 `Phi=V_GV`는 완전한 통계 추출까지 포함한 함수다.

기준점은 해당 view에서 frozen aligner가 예측한 `mu_T`다. `I-N` corrupted step이면 기준 PAN은 이미 생성된 `P_epsilon`이고, epsilon은 모든 probe에서 동일하다.

**금지:** Teacher 최종 HRMS를 직접 shift한 결과를 위 함수의 대용으로 쓰기. U-Net이 PAN과 MS에 비선형적으로 반응하므로 같은 미분이 아니다.

### 6.3 두 방향의 Jacobian

$$
J_T(p)=\left.\frac{\partial z_T(p;\delta)}{\partial(\delta_y,\delta_x)}\right|_{\mu_T}
\in\mathbb R^{D\times2}.
$$

`D=8`이면 pixel의8band, GV에서는 `D=16`이며 통계의 고정 중심을 사용한다. 단위는 `(해당 표현 단위)/HR pixel`이다.

첫 구현은 중앙 유한차분을 권한다.

$$
J_y\approx\frac{z_T(\mu_T+h e_y)-z_T(\mu_T-h e_y)}{2h},\quad
J_x\approx\frac{z_T(\mu_T+h e_x)-z_T(\mu_T-h e_x)}{2h}.
$$

- **새 pilot 간격:** `h=0.05 HR px`; `0.025/0.10`과 비교. 기존 curvature 수치 pilot과 같은 출발값일 뿐 최적값은 아니다.
- h는 미분 간격이지 training jitter 반경이나 실제 misalignment의 표준편차가 아니다.
- Aligner를 ±h 입력마다 다시 호출해 보정하게 만들지 않는다. `mu_T.detach()`에 h를 직접 더해 Teacher sampling grid를 만든다.
- 원 `P_view`에서 각 probe의 PAN을 생성한다. 이미 보정한 PAN에 추가 h warp를 누적하지 않는다.
- Student graph 생성 전에 **Teacher probe를 순차 no-grad forward**하고 결과를 detach한다.
- 기본 Teacher forward1회 외에 **Teacher 복원 U-Net4회가 추가**된다. 작은 aligner4회만의 비용이 아니다.
- GT는 Teacher forward에 들어가지 않는다. GT 기반 covariance/calibration은 loss-side의 별도 작업이다.

### 6.4 Covariance가 있을 때: 출력 공간으로 전파

Teacher 이동량 covariance가 해당 sample/view에서 유효하다고 판단됐다는 조건에서:

$$
\boxed{C_T^{align}(p)\approx J_T(p)\Sigma_{\Delta,T}J_T(p)^T.}
$$

`Sigma_delta`는2×2 HR pixel², `C_align`은 D×D 표현단위²다. **최대 rank2인 국소 linearized conditional covariance**다. 영상 전체에 대한 `(CHW)×(CHW)` 행렬을 만들지 않는다.

이 항은 평균적인 Teacher bias, radiometric 오류, 모델 학습의 seed variance를 모두 설명하지 않는다. 작은 변위와 chosen covariance 아래의 위치 민감성만 근사한다.

### 6.5 Covariance 출처와 적격성

| Source | 사용 조건 | 명칭/해석 |
|---|---|---|
| `G-GEO` 구조비용 국소 근사 | 현재 descriptor·normalization·rank·오차 scale 검증 | 조건부 위치 covariance 근사 |
| 정확한 기하 label이 있는 별도 calibration | 실제 보유한 경우만, train/calibration 분리 | label 기반 평가가 가능한 추정 uncertainty |
| `G-EQ` closure error 이차 모멘트 | offset 반응 통계로만 해석 | 절대 위치 covariance가 아닌 **robustness proxy** |
| `sigma² I` 임의 지정 | 확률적 근거 없음 | uncertainty가 아닌 isotropic stress-scale 대조 |
| covariance 없음 | 아래 C-SENS 사용 또는 C-COV 보류 | 누락을 covariance0으로 처리하지 않음 |

G-GEO의 출발식은 [S0 §11]의 `s²(J_geoᵀJ_geo+gammaI)^−1+floorI`다. 여기서 **J_geo는 PAN–GT 구조 residual의 미분**, 본 C의 **J_T는 복원 U-Net 출력의 미분**이다. 같은 J가 아니다.

특히 구조정보가 없는 방향의 precision을0으로 처리한 것을 역으로 covariance0, 즉 완전한 확신으로 바꾸면 안 된다. **Pseudoinverse에서 null direction이0으로 나타났다는 이유로 uncertainty 전파에서 그 방향을 확실하다고 처리하지 않는다.**

**G-EQ를 전파하는 경우는 `TRI-C-EQPROXY`라는 별도 탐색 ID로 둔다.** 이를 G-GEO covariance와 같은 uncertainty 결과 열로 합치지 않는다.

첫 C-COV에서는 source가 수치적으로 유한하고, 출처·단위·view가 맞고, 요구한 방향이 식별되며, 최소 진단을 통과한 sample만 `cov_valid=true`로 사용한다. 원리상 정확히 알려진 zero covariance와 관측 정보 부족을 구분한다. Rank/허용 residual/scale 기준은 train calibration manifest에 기록한다.

Invalid sample에서는 C의 추가 감쇠를1로 두어 **기존 R3/A soft로 돌아간다**. GT hard에는 영향이 없다. 이는 unknown=certainty라는 주장이 아니라 C를 적용하지 않은 fallback이며, 해당 비율을 반드시 별도로 보고한다. 전부 invalid면 `COV_BLOCKED`이지 C 성공 run이 아니다.

### 6.6 TRI-C-SENS — covariance가 없어도 가능한 민감도 대조

$$
q_{sens}(p,j)=J_{p,j,y}^2+J_{p,j,x}^2.
$$

고정 native train calibration의 양의 q_sens 중앙값을 `s_sens`로 두고:

$$
\boxed{r_{sens}(p,j)=\frac{s_{sens}}{s_{sens}+q_{sens}(p,j)}.}
$$

둘의 단위가 동일하므로 비율은 무차원이다. 양의 민감도가 전혀 없다면 source가 무반응인 것으로 표시하고 감쇠1로 fallback한다. 통계0을 분모의 작은 상수로 바꿔 모든 입력을 high confidence라고 부르지 않는다.

**이 case에는 Teacher uncertainty라는 이름을 붙이지 않는다.** 어떤 band의 Teacher 출력이 correction 변화에 민감한지를 측정하는 deterministic sensitivity 대조다. 미분 구현과 C의 가능성을 먼저 확인하는 데 사용한다.

### 6.7 TRI-C-DIAG — 먼저 구현할 covariance-aware L1 감쇠

전체 C 대신 대각만 계산한다.

$$
q_{pj}=J_{pj,:}\Sigma_{\Delta,T}J_{pj,:}^T\ge0.
$$

표현의 soft 감쇠 scale을 `s_C>0`라고 할 때:

$$
\boxed{r_C(p,j)=\frac{s_C^2}{s_C^2+q_{pj}}\in(0,1].}
$$

HRMS identity 표현의 첫 제안은 `s_C=tau_R`다. 이는 기존 복원 오차 scale을 사용하는 **감쇠 민감도 파라미터**이며, 전체 Teacher noise의 표준편차라고 주장하지 않는다. GV에서는 해당 통계 공간의 `tau_V`를 사용한다. Source와 s_C를 모두 고정해야 C의 강도를 해석할 수 있다.

$$
\boxed{
L_R^{Cdiag}=\left\langle w_H e_S+
 w_K\operatorname{mean}_c\{r_C(p,c)|\hat Y_{S,pc}-\hat Y_{T,pc}|\}\right\rangle.
}
$$

**Hard는 불확실성으로 나누지 않는다.** Teacher의 정합량을 믿기 어렵다는 이유로 GT의 경계 fitting까지 약화하지 않는다. R3의 `w_K`에 이미 Teacher 오차·Student 우위가 있으므로 C는 그 위의 별도 risk attenuation이다.

A와 결합할 때에는 soft 내부에 `m_A*r_C`를 곱한다. B와 결합할 때에는 **통계 출력 자체를 perturb해 얻은 J_V**에서 계산한 `r_C^V`를 B의 통계 soft에 곱한다. Pixel q를 그냥 resize해서 gradient-variance의 propagated covariance라고 부르지 않는다.

### 6.8 TRI-C-FULL — 저랭크 방향 정보까지 사용하는 후속안

D×D 작은 행렬을 매 pixel inverse하기보다 다음을 사용한다.

$$
\Sigma_{\Delta,T}=LL^T,\qquad U_p=J_pL/s_C,\qquad
W_p=(I_D+U_pU_p^T)^{-1}.
$$
$$
W_pd=d-U_p(I_2+U_p^TU_p)^{-1}U_p^Td.
$$

2×2 solve만 필요하다. `J/Sigma/U/W`는 detach하고 `d=Y_S-Y_T`만 live로 둔다.

$$
\ell_Q(p)=\frac{d_p^TW_pd_p}{D\,s_C}.
$$

이는 기존 L1 soft와 **다른 quadratic soft**다. 따라서:

$$
L_R^{CQ}=\langle w_H e_S+\lambda_Q w_K\ell_Q\rangle
$$

와 **같은 quadratic loss에 W=I인 `C-QISO`**, 그리고 W의 trace/D만 쓰는 `C-QSCALAR`를 반드시 비교한다. `lambda_Q`는 공통 train pilot에서 QISO의 soft gradient 규모를 기준 R3 soft와 맞추는 제안이며, FULL에만 따로 유리하게 맞추지 않는다.

L1과 비교해 FULL만 좋아졌다고 covariance의 방향성 효과를 주장하지 않는다. Off-diagonal mixing과 quadratic soft 때문에 기본 R3의 출력 성분별 GT-dominance 설명을 이 전체 loss에 그대로 적용하지 않는다.

첫 구현에서는 **C-DIAG와 C-SENS가 우선**이고 FULL은 독립 대조가 준비됐을 때 진행한다.

### 6.9 C의 최소 검증

1. **미분 간격:** h/2·h·2h의 Jacobian 방향·RMS와 finite 여부를 기록한다. h가 너무 작아 FP32 cancellation이 발생하면 C를 시작하지 않는다.
2. **국소 선형성:** 진단용 작은 delta에서 `z(mu+delta)-z(mu)`와 `J delta`를 비교한다. 원래 GT를 움직이거나 해당 probe로 Student를 추가 학습하지 않는다.
3. **공분산 전파:** 구현은 선형 toy에서 `J Sigma Jᵀ`와 직접 샘플 통계가 일치하는지 검사한다. 실영상에서는 근사 오차를 별도 보고한다.
4. **해상도·단위:** HR/LR 단위 변경에서 `mu_HR=r mu_LR`, `Sigma_HR=r² Sigma_LR`, 그에 맞는 J 단위가 일관돼야 한다.
5. **Source independence:** Teacher weight와 correction source, covariance, J에 Student gradient가 없어야 한다.
6. **Hard 보존:** Covariance invalid·매우 큰 q·모든 soft0 상태에서도 GT hard가 남아야 한다.
7. **실제 원인 분리:** q가 큰 경계가 단순히 모든 고주파 영역인지, Teacher 오차·Student–Teacher 차이와 어떤 관계인지 보고한다. “민감=틀림”으로 단정하지 않는다.

---

## 7. A/B/C 조합: 무엇을 대체하고 무엇을 추가하는가?

### 7.1 통일된 표기

$$
L_R^{A,C}=\left\langle w_H e_S+
 w_K\operatorname{mean}_c\{m_A r_C|\hat Y_S-\hat Y_T|\}\right\rangle,
$$
$$
L_V^{B,C}=\left\langle w_H^VE_S^V+
 w_K^V\operatorname{mean}_j\{m_B r_C^V|V_S-V_T|\}\right\rangle.
$$

해당 모듈이 OFF면 mask를1로 둔다. C가 covariance 미검증으로 fallback한 경우도1이지만 **OFF와 실패/불확실 fallback을 로그에서 구별**한다.

$$
\boxed{L_{total}=L_R^{A,C}+\lambda_VL_V^{B,C}+L_{aux,inherited}.}
$$

- A-only: 통계 parent가 OFF면 계속 OFF. A 효과와 새 GV 도입을 한 비교로 섞지 않는다.
- B-only: R3 영상값 loss는 unchanged. 기존 GV-AD를 B로 교체한다.
- C-only: R3 soft에 r_C만 적용한다. 먼저 A·B 없이 별도 효과를 확인한다.
- AB: A의 영상값 방향 판단과 B의 통계 방향 판단 결합.
- ABC: 앞의 단독 및 주요 조합이 유효할 때만 선택. Pixel C만 쓸지 통계 C도 쓸지 명시한다.

### 7.2 여러 gate가 KD를 사실상 없앨 수 있다

기본 `w_K` 자체가 작고 A/B/C가 모두1 이하이므로 결합 후 soft가 거의 사라질 수 있다. 이때 개선은 복잡한 지식 전달보다 **KD 제거** 때문일 수 있다.

그래서 항상 다음을 비교한다.

- Reconstruction `R1`: weighted GT-only.
- Statistics `WH`: weighted GT-stat-only.
- 새 gate와 평균 soft 총계수량만 맞춘 `MASS` 대조.

Soft active 평균으로 재정규화해 억제를 상쇄하지 않는다. 거의0이면 그대로 기록하고 더 단순한 hard-only와 비교한다.

---

<a id="registry"></a>

## 8. 실험 registry와 우선순위

### 8.1 이름 규칙

예시:

```text
S2_W112_TRIABC_AFR_IA_R3_GVOFF_ASIGN_COFF_seed<seed>_v01
S2_W112_TRIABC_AFR_IA_R3_GV_BSIGN_COFF_seed<seed>_v01
S2_W112_TRIABC_AFT_IN_R3_GVAD_AOFF_CDIAG_GGEO_seed<seed>_v01
```

`seed`, full update N, optimizer/LR/batch는 **실제 s2 parent manifest**를 계승한다. s2라는 이름만으로 seed1234나 GPU4090을 자동 확정하지 않는다. 한 단계에서 width·seed·aligner 정책·loss를 모두 바꾸지 않는다.

### 8.2 최소 독립 비교

| 우선 | ID | 변경 | 필수 대응 기준 |
|---|---|---|---|
| P0 | `TRI-DIAG` | 학습 없이 S/T/GT 관계, mask, J 기능 검사 | 현재 frozen assets |
| P1 | `TRI-A-BASE` | R3, 새 mask 없음 | 기존 cohort 재사용은 hash/조건 일치 시만 |
| P1 | `TRI-A-SIGN` | 영상 band별 방향 mask | A-BASE, R1 |
| P1 | `TRI-B-BASE` | R3 + GV-AD | 기존 STAT 구현 |
| P1 | `TRI-B-SIGN` | GV 통계 성분별 방향 mask | B-BASE, GV-WH/GV-H |
| P1 | `TRI-C-SENS` | HRMS 민감도 감쇠 | A-BASE 및 matched mass |
| 조건부 P1 | `TRI-C-DIAG` | 검증된 이동 covariance 전파 | C-SENS, A-BASE |
| P2 | `TRI-A-COS/CAP` | scalar 방향 / magnitude cap | A-SIGN과 구분 |
| P2 | `TRI-A-BANDADV` | band 우위로 soft 세분화 | A-BASE, 성분 tau 기록 |
| P2 | `TRI-B-CAP/COMPADV` | 통계 cap / 성분 우위 | B-SIGN, B-BASE |
| P2 | `TRI-B-GC/SC-SIGN` | 공분산 표현의 방향 mask | **같은 GC/SC-AD** |
| P2 | `TRI-AB` | A-SIGN + B-SIGN | A-only, B-only, 공통 base |
| P2 | `TRI-AC` | A-SIGN + C-DIAG 또는 C-SENS | 각각 단독 |
| P3 | `TRI-C-QISO/QSCALAR/FULL` | 동일 quadratic soft에서 방향성 비교 | 세 case의 동일 scale |
| P3 | `TRI-BC/ABC` | 통계 J 전파까지 결합 | Pixel C와 구분 |
| P3 | `TRI-FT/SC` | 유력 loss의 aligner 미세조정/scratch | 같은 loss frozen 대조 |
| 확인 | `REPEAT/RESTART` | 주요 쌍 seed 반복·고정 설정 재시작 | adaptive path와 분리 |

**구현 순서:** A-SIGN → B-SIGN → C-SENS/J 검사 → C-DIAG → 세부·조합.  
**방법론 집중:** B의 구조 성분 구분을 중심으로 A가 보완하는지 확인하고, covariance가 준비되면 C를 연결한다.

C-COV가 막혀도 A/B와 C-SENS는 진행 가능하다. 다만 C-SENS를 covariance 검증 완료 사례로 바꿔 부르지 않는다.

### 8.3 대조군의 정확한 의미

**MASS 대조:** 새 mask m 또는 r가 단순히 soft를 줄인 것인지 확인한다. 같은 batch에서 기본 soft 계수 `w_K`를 feature축에 broadcast한 `b_j`에 대해:

$$
\kappa=\frac{\sum b_j m_j}{\sum b_j},\quad \sum b_j>0;
\qquad \kappa=0\ \text{otherwise}.
$$

이 κ를 detach하고 모든 soft 성분에 동일하게 적용한다. **계수 총량만 맞추며 실제 loss값·gradient 크기까지 같지는 않다.** Production normalization이 아니라 별도로 이름 붙인 인과 대조다. Gradient accumulation에서는 optimizer update 전체의 matching을 할지 microbatch별 matching인지 고정·기록한다.

**SHUFFLE 대조:** 새 mask만 고정 RNG로 공간 순서를 섞는다. Teacher/GT/출력·기본 w_H/w_K는 그대로 둔다. Band permute는 다른 질문이므로 따로 표시한다.

**C 방향 대조:** Sigma의 eigenvalue는 유지하고 축 방향만 회전하거나 sample 간 Sigma를 섞는다. **J는 해당 sample의 실제 J**를 유지한다. C-DIAG의 scalar 감쇠 대비 방향 효과는 제한적일 수 있으므로 FULL 방향성은 QSCALAR와 비교한다.

**같은 계산비 대조:** C의 추가 Teacher forward 시간을 기록하고, 그 시간만큼 hard-only/R3를 더 학습한 결과를 별도 보고한다. 진단 probe만 동일하게 돌리고 loss에는 쓰지 않는 compute-matched run도 선택할 수 있다.

**단순 band gate 대조:** A/B의 개선이 고차 기하학 때문인지, 단순한 세분화 때문인지 A-BANDADV 및 기존 component-wise 방식과 비교한다. 단순한 gate를 이겼다는 실험 없이 복잡한 trilateral geometry의 고유 효과로 확대하지 않는다.

---

## 9. 구현 모듈과 데이터 인터페이스

다음 경로는 제안이다. 실제 저장소 layout을 확인해 adapter를 만든다.

```text
experiments/s2_w112_kdv/triabc/
  contracts.py                 # tensor shape·좌표·asset manifest
  routing_direction.py         # COS / SIGN / CAP, 모든 gate detach
  losses_reconstruction.py     # 기존 wH/wK를 받아 soft만 교체
  losses_statistics.py         # 기존 V extractor + 동일 soft 교체
  teacher_shift_response.py    # frozen F_T의 correction ±h probe
  covariance_adapter.py        # source별 validity·unit·provenance
  output_risk.py               # diag JΣJᵀ, sensitivity, low-rank solve
  calibration_triabc.py        # train-only scale/FD 진단
  diagnostics_triabc.py        # 삼각관계·softmass·방향/구조/위치 분석
  tests/                       # 아래 단위·통합 검사
```

### 9.1 Tensor 계약

| 이름 | shape | grad |
|---|---|---|
| `pred_s` | `[B,8,H,W]` | live |
| `pred_t`, `gt` | 같은 shape | detached target |
| `w_h`, `w_k` | `[B,1,H,W]` | detached, w_k는 beta_R 포함 |
| `mask_a`, `risk_r` | `[B,8,H,W]` 또는 명시적 broadcast | detached |
| `stat_s` | GV16 / GC32 / SC64 channels | live |
| `stat_t`, `stat_g` | 같은 통계 shape | detached |
| `mask_b` | 통계와 동일 shape | detached |
| `mu_t`, `mu_s` | `[B,2]`, 순서 `(dy,dx)` | Teacher detached, trainable Student live |
| `jac_t` | `[B,D,2,H_phi,W_phi]`, `axis=dy,dx` | detached |
| `sigma_delta` | `[B,2,2]`, HR pixel² | detached |
| `cov_valid` | `[B]` 또는 명시적 granular source mask | bool, detached |
| `q_diag`, `r_c` | `[B,D,H_phi,W_phi]` | detached |

실행 전 dtype/device/shape/finite/broadcast를 assert한다. MS band의 실제 순서와 z-space 순서를 metadata로 남긴다. Teacher probe와 Student loss의 ROI 중심이 같아야 한다.

### 9.2 기존 reconstruction criterion 연결

기존 `GTAnchoredReconstructionKD(..., return_maps=True)`에서 `hard_weight`, `soft_weight`를 얻을 수 있다. `soft_weight`에는 이미 `kd_weight`가 포함된다.

```python
base = rec_criterion(pred_s, pred_t, gt, return_maps=True)
w_h = base.maps['hard_weight']
w_k = base.maps['soft_weight']
# base.loss에 새 full loss를 더하지 않는다.
# 동일 hard를 사용하고 soft만 새 mask로 다시 계산한다.
```

통계 wrapper도 equivalent한 detached maps를 반환하도록 adapter를 확장한다. 기존 코드 내부의 scalar soft 평균을 먼저 내버리면 band/component mask를 나중에 정확하게 적용할 수 없으므로 **평균 전 성분 오차**를 사용한다.

### 9.3 주 training step 의사코드

```python
# 실제 optimizer/AMP/accumulation/loader API에 맞춰 연결할 의사코드다.
P_view, M, G, view_meta = prepare_current_parent_protocol(batch)

with no_grad_teacher_context():
    t_native = teacher_forward_own_alignment(P_view, M)
    pred_t, mu_t = t_native.pred, t_native.mu

# C probe는 Student activation graph를 만들기 전에 순차 계산한다.
if cfg.c.enabled:
    J = teacher_shift_jacobian_fd(
        P_view=P_view, M=M, correction=mu_t.detach(),
        frozen_restorer=teacher.restorer, phi=chosen_phi, h=cfg.c.h,
    )
    risk = build_C_mask_or_fallback(J, covariance_asset, cfg.c)
else:
    risk = None

pred_s, mu_s = student_forward_for_aligner_policy(P_view, M)
rec_maps = frozen_R3_routing(pred_s.detach(), pred_t, G)
mask_a = direction_gate(G, pred_s.detach(), pred_t, cfg.a)
rec_loss = hard_plus_masked_soft(pred_s, pred_t, G, rec_maps, mask_a, risk)

stat_loss = zero_scalar()
if cfg.statistics.enabled:
    stat_s = existing_stat_extractor(pred_s)  # live, not no_grad
    with no_grad_teacher_context():
        stat_t = existing_stat_extractor(pred_t)
        stat_g = existing_stat_extractor(G)
    stat_maps = frozen_stat_AD_routing(stat_s.detach(), stat_t, stat_g)
    mask_b = direction_gate(stat_g, stat_s.detach(), stat_t, cfg.b)
    # 통계 C에는 통계 Phi의 J로 만든 risk만 넣는다.
    stat_loss = hard_plus_masked_soft(stat_s, stat_t, stat_g,
                                     stat_maps, mask_b, stat_risk)

aux = inherited_auxiliary_loss_with_correct_grad_paths(...)
loss = rec_loss + cfg.lambda_v * stat_loss + aux
backward_with_parent_accumulation_and_scaler(loss)
```

주의: `no_grad_teacher_context`는 설명용 이름이다. 실제 PyTorch `no_grad`/autocast 설정과 저장소 함수를 사용한다. 전체 Student forward나 `stat_s` 계산을 no-grad로 감싸면 안 된다.

### 9.4 C probe 구현 의사코드

```python
# Forward만 하는 수치 미분; teacher parameter gradient는 필요하지 않다.
@torch.no_grad()
def fd_response(p_view, ms_base, mu_t, teacher_restorer, phi, h):
    cols = []
    for axis in (0, 1):  # 0: dy, 1: dx
        step = zeros_like(mu_t)
        step[:, axis] = h
        p_plus  = warp_pan(p_view, mu_t + step)
        z_plus  = phi(ms_base + teacher_restorer(cat(p_plus, ms_base)))
        p_minus = warp_pan(p_view, mu_t - step)
        z_minus = phi(ms_base + teacher_restorer(cat(p_minus, ms_base)))
        cols.append((z_plus.float() - z_minus.float()) / (2.0*h))
    return stack(cols, dim=2).detach()  # [B,D,2,Hphi,Wphi]
```

`cat`, `zeros_like`, `stack`은 명시적으로 구현할 placeholder다. 실제 wrapper가 base를 더한다면 위에서 중복하지 않는다. Warp 부호·bicubic·align_corners·padding·input channel order는 parent와 같아야 한다. ±probe는 full P_view에서 각각 생성하고 teacher_restorer는 eval 고정 상태다.

---

## 10. Calibration과 계수 선택

### 10.1 바꾸지 않을 것

- R3의 `tau_R/alpha_R/beta_R`.
- 같은 표현 안의 `tau_V/epsilon_V/lambda_V`.
- 기존 Teacher와 Student 초기값, 데이터 순서, optimizer와 schedule.
- A/N recipe의 계수·crop·view·warp 횟수.

A/B/C mask를 추가하면서 hard를 더 키우거나 soft를 추가 증폭하지 않는다. 유리한 coefficient를 case별로 달리 찾으면 그 효과가 섞인다.

### 10.2 새로 필요한 것

| 항목 | 초기 규칙 | 범위 |
|---|---|---|
| A/B sign | strict same-sign, equality0 | 새 loss-side gate |
| A/B cap epsilon | 현재 R/V의 수치 epsilon | 같은 단위의 분모 |
| Cos epsilon | `epsilon_R²` 시작, norm0이면0 | vector direction 대조 |
| C FD h | 0.05 HR px; h/2·2h 확인 | 수치 미분 간격 |
| C-SENS scale | train positive `q_sens` median | 민감도 정규화, uncertainty 아님 |
| C-DIAG s_C | identity는 tau_R, GV는 tau_V로 시작 | 감쇠 규모, 재확인 필요 |
| C covariance validity | source별 rank/scale/support gate | 실제 calibration 결과로 확정 |
| C quadratic scale | QISO와 기준 soft gradient의 공통 pilot | FULL/QISO/QSCALAR 동일 |

이 값은 새 제안이다. 수치변경은 config version과 paired control로 기록한다. s_C를 변화시키는 후속 후보는 `×0.5/1/2`로 제한해 먼저 sensitivity를 본다. Student 현재 오차를 s_C 분모에 넣어 온라인으로 변하게 만드는 방식은 기본에 포함하지 않는다.

### 10.3 Calibration 데이터

Native **train** sample ID를 고정한다. 비용상 subset을 사용하면 ID·seed·개수·source hash를 저장한다. 

FR benchmark나 보고용 test에서 threshold·covariance scale·h·계수를 선택하지 않는다. N corrupted condition도 확인하되 native/corrupt 통계를 따로 기록한다. Shape64/RR256/FR512에서 발생하는 민감도 차이를 같은 단위로 보고, image size 비율로 shift나 J를 임의로 rescale하지 않는다.

---

## 11. 메모리·처리시간·수치 안정성

### 11.1 비용 구분

| Case | 기본 대비 주요 추가 비용 |
|---|---|
| A | 출력 차이와 작은 mask 계산. 추가 U-Net 없음 |
| B | 기존 통계 추출 + component mask. 새로운 Student forward 없음 |
| C-SENS/C-DIAG | 기본 Teacher 외 **4번의 Teacher 복원 forward**, J map·2×2 연산 |
| C-GV | 같은4개의 Teacher 영상에서 통계 Phi까지 계산 |
| C-FULL | J map + 저랭크2×2 solve. 거대한 output covariance 없음 |

실제 시간은 s2에서 warmup 제외 최소200 update를 profile한다. `A/B가 거의 공짜`, `모두24시간 내 완주`처럼 실측 없는 주장은 하지 않는다. 계산비는 Teacher forward, Student forward/backward, stat, J, Sigma, evaluator로 나눠 기록한다.

### 11.2 실행 순서와 graph 수명

1. 입력 전처리 및 frozen Teacher 기준 출력.
2. C가 켜졌으면 Teacher probe4개를 **순차** 생성·축약하고 activation을 보존하지 않음.
3. Student forward를 한 번 수행.
4. FP32 loss/gate/stat/risk 계산 후 backward.
5. 필요 diagnostic은 별도 주기에 실행하고 optimizer step을 추가하지 않음.

Teacher 기준 출력과 FD 출력은 같은 radiometric precision/forward 규칙을 사용한다. 작은 h에서 FP16 차분을 하면 cancellation이 커질 수 있으므로 C source는 FP32 우선이다. Teacher precision을 바꾸면 target/calibration 조건 변경으로 기록한다. Student U-Net AMP는 parent를 계승한다.

### 11.3 OOM 대응

- C probe를 병렬 batch concat하지 않고 순차 실행.
- covariance는2×2, 출력은 대각 또는 rank2 factor만 저장.
- SC는 band-pair chunk 방식으로 계산; 전체 unfold 금지.
- 필요하면 microbatch를 줄이고 parent effective batch를 accumulation으로 유지.
- 큰 장면 probe가 필요하면 inference diagnostic에서만 적절한 tiling을 검토하되, global shift·halo·full-scene 동치 검증 없이 교체하지 않음.
- GPU 메모리 정리만으로 해결되는 것처럼 반복 `empty_cache`에 의존하지 않음. 실제 graph와 tensor 보유량을 확인.

### 11.4 Cache

Frozen Teacher라고 sample ID만으로 cache할 수 있는 것은 아니다. Key에 최소 다음을 포함한다.

```text
teacher checkpoint/hash, source code/hash, image ID, crop/flip/rotation,
MS base/upsampler, intensity, input protocol, exact epsilon,
aligner crop/normalization, correction, warp config,
Phi/stat window/ROI, finite-difference h, covariance source/version.
```

Teacher가 완전히 augmentation-equivariant하다고 가정해 canonical output/J/covariance를 임의 회전하여 쓰지 않는다. 실제 온라인 값과 동치 확인 후에만 최적화한다.

C를 일부 update에만 계산하는 최적화는 가능하나 `C-PERIODIC`로 별도 ID를 부여한다. 비계산 update에 오래된 다른 sample risk를 사용하지 않는다. 첫 정확성 비교는 매 update 계산을 기준으로 두고, 이후 비용/빈도 효과를 분리한다.

---

<a id="tests"></a>

## 12. 단위·통합 검사 명세

아래는 **실행자가 통합 전후에 수행할 검사 목록**이다. 본 문서 작성 시 실제 s2 데이터/model/CUDA 검사가 끝났다는 뜻이 아니다.

### 12.1 A: 방향과 GT 보존

| 검사 | 기대 결과 |
|---|---|
| G=S, T만 다름 | A mask0; GT error0, 불필요한 soft0 |
| S=T≠G | soft0이어도 hard gradient 존재 |
| G−S와 T−S 같은 부호 | sign1 |
| 반대 부호 | sign0 |
| 일부 band만 GT와 동일 | 해당 band soft0, 다른 band 정상 |
| T가 GT를 크게 지나침 | cap<sign |
| mask=ones | 기존 R3 loss/gradient와 일치 |
| Teacher/GT/mask leaf requires_grad | gradient 없음; Student loss live |
| all mask0 | weighted GT-only와 동일 |
| BANDADV에 sign 추가 | 수치 경계 외에는 동등, 중복을 확인 |

### 12.2 B: 통계와 좌표

- 기존 통계 extractor와 값·shape·순서 동치; k5 margin3/58² 확인.
- Scharr valid vs reflect+interior 동치.
- Population variance, GT·Student 동일영상에서 hard0.
- Covariance 대칭성, off-diagonal 음수 보존, raw covariance PSD 수치 확인.
- GT/S/T의 상수 밝기 이동에서 centered statistic의 기대 불변성.
- 동일 variance지만 위치가 다른 패턴에서 통계0이어도 reconstruction이 오차를 검출함.
- Constant Student의 zero variance gradient를 정상 현상으로 확인하고 recon은 살아 있는지 검사.
- Component gate reshape로 dy/dx·band·covariance 원소가 섞이지 않음.
- Frozen aligner일 때 U-Net grad 존재, aligner grad 없음; FT/SC에서는 loss 경로 연결.

### 12.3 C: 미분·covariance·fallback

- 선형 toy `z(delta)=b+J delta`의 FD가 J와 일치.
- 분석적 `diag(J Sigma Jᵀ)`와 구현이 일치, q≥0.
- Sigma=0이 **명시적으로 유효**인 수학적 toy에서는 r=1; 실제 미확인 source와 구분.
- covariance invalid이면 r=1 parent fallback, invalid 로그 증가.
- 같은 J에서 Sigma를 PSD 순서로 키우면 diag q 증가, r 감소.
- `(dy,dx)` 순서와 `grid_sample (x,y)` 순서 불일치 탐지.
- h의 단위변환과 covariance r² 규칙 확인.
- Woodbury2×2와 직접 D×D inverse의 toy 동치, W eigenvalue (0,1].
- J=0에서 sensitivity 반응없음을 감지; 잘못된 certainty claim 금지.
- Teacher parameters frozen이면서 probe값은 delta에 따라 실제로 변함.
- Corrupt view의 epsilon 고정, learned correction만 perturb.
- Phi=GV의 FD는 `V(Yplus)−V(Yminus)`로 계산했는지 확인.
- Support 실패를 clamp/동적 crop으로 숨기지 않음.

### 12.4 s2 integration smoke

실제 W112·실제 batch·현재 AMP/optimizer·native/corrupt 경로에서 유한값, 올바른9ch, base1회, Student gradient, Teacher freeze, peak allocated/reserved memory, 시간, evaluator/selector를 검사한다. 참고 helper의 단위검사 통과가 이 과정을 대체하지 않는다.

---

## 13. 분석·로그: 어떤 결과가 개념을 지지하는가?

### 13.1 공통 품질과 fitting

상위 평가를 유지한다. RR ERGAS/SAM/PSNR/Q8와 unweighted train/validation MAE를 보고한다. FR HQNR/Dlambda/Ds와 기존 fSCC를 함께 보고한다. Teacher 고정오차 bin(예: 하위50%,50–90%,상위10%)에서 Student가 얼마나 개선했는지 동일 픽셀 집합으로 비교한다.

**Overfitting의 작동과 효용을 분리한다.** Train error 감소만 있으면 강한 fitting이 일어난 것이며, validation/FR까지 좋아졌다고 표현하지 않는다. Teacher 실패가 noise/센서 차이/정합 한계에서 왔는지도 함께 본다.

### 13.2 A/B 기전 로그

```text
plain_hard / weighted_hard / original_soft / masked_soft
teacher_error, student_error, teacher_student_error
wK_positive_fraction, mask_mean, weighted_soft_mass_ratio
u_zero / v_zero / same_sign / conflicting_components fraction
pixel_good_teacher_but_bad_bands fraction
masked_components_Student_GT_error_before_after
GT-stat/Teacher-stat/Student-stat raw maps
under_sharp / over_sharp bins; signed-edge error
soft actually zero due to aT vs due to new mask
```

동적인 gate 조건만으로 성능을 평균하면 학생마다 평가 대상이 달라진다. 동적 로그는 mechanism 설명에 쓰고 최종 성능은 고정 영역과 전체 장면에서 비교한다.

### 13.3 C 기전 로그

```text
Sigma source/id/validity/reason, units, eigenvalues/rank
J dy/dx RMS, finite_difference_step, h-sensitivity
linearization residual, probe support
q per-band/P50/P90, r_C histogram, invalid fallback fraction
mean_soft_attenuation, covariance-vs-sensitivity-only difference
Teacher error × q bins, GT edge strength × q bins
runtime teacher_base / probes / student / covariance / stat
```

Teacher 출력이 변화에 민감한 경계를 학생이 덜 복원하는 부작용을 반드시 확인한다. 전체 경계가 약해져 Ds가 좋아지는 상황을 자동 성공으로 보지 않는다.

### 13.4 판정표

| 관찰 | 해석 |
|---|---|
| A-SIGN > R3 및 MASS | 단순 KD 감쇠 외에 band 방향 정보의 가치 |
| A-SIGN≈R1 | Teacher target을 빼는 것이 주효과일 수 있음 |
| B-SIGN > GV-AD 및 GV-WH | Teacher 통계의 성분별 선택이 추가 정보를 줄 가능성 |
| B가 signed edge와 유사 | 일반 경계 감독 대비 variance 특유 효과는 약함 |
| C-DIAG > C-SENS/MASS | covariance 정보가 단순 민감도·약한KD 이상일 가능성 |
| C-FULL > QISO/QSCALAR | 동일 quadratic 조건에서 방향 간 관계의 가치 |
| shuffled mask/covariance도 동일 | 제안한 기하학적 의미보다 가중치 배율 효과 의심 |
| 통계만 개선되고 실제 edge/RR 악화 | moments fitting의 지름길·noise/ringing 확인 |
| A-FT만 개선 | Student에 맞는 정합 adaptation이 필요할 가능성 |
| A-SC가 FT보다 개선 | Teacher 초기정합에 대한 불리한 제약 가능성 |
| Train만 개선 | fitting 성공과 활용 성능 성공을 구분 |

“가능성”이라는 해석은 단일 case의 확정 인과 결론이 아니다. 대응 seed와 부정 대조로 재확인한다.

---

## 14. 평가·checkpoint·데이터 누수 방지

[S0 §18]의 다음 계약을 그대로 유지한다.

1. `Raw-original`: 기존 PAN reference와 원 evaluator 영역.
2. `Raw-valid`: 같은 원 PAN, 고정 공통 ROI.
3. `Aligned-valid`: 해당 Student forward에서 실제 사용한 PAN, 같은 공통 ROI.

동일 checkpoint의 HRMS는 세 view에서 동일하다. Raw-original은 aligner OFF 결과가 아니다. Aligned PAN의 저해상도 reference도 같은 evaluator의 경로로 다시 생성한다. Dlambda가 PAN 비의존인 구현이라면 같은 ROI의 raw-valid/aligned-valid Dlambda가 같아야 한다.

최근 계획의 FR mat20·metric v2 py와 `best_raw/best_aligned/last/best_rr_val` 저장을 실제 s2 설정과 대조해 계승한다. 과거12–19로 되돌리지 않는다. **FR mat20이 benchmark test이고 방법·checkpoint 선택에 반복 사용됐다면 test-adaptive exploratory라고 표시한다.** 최종 확인은 독립 validation 기준 또는 사전 선언 final-N로 분리한다.

A/B/C는 training GT로 gate를 만든다. 추론에는 GT·Teacher·A/B/C gate·J·통계 loss가 필요하지 않는다. 최종 추론에는 해당 run의 aligner와 Student 복원망만 남는다.

**GT로 teacher output을 정렬해 loss를 계산하거나, metric이 좋아지는 추가shift를 탐색해 보고하지 않는다.** 어려운 위치가 mask에서 빠져 점수가 좋아지는 우회로를 막는다. Metric ROI와 통계 loss ROI와 aligner input crop은 별개다.

---

## 15. 약 24시간 운영과 중간 변경

이번 문서는 기존 s2 작업 위에 추가되는 backlog다. **전체 A/B/C가 추가24시간 안에 끝난다는 예측은 아니다.** 첫24시간 시점에는 다음을 정리한다.

- A/B의 구현·smoke 상태와 최소 대응 결과.
- C의 J 구현, covariance source 가용성, C-SENS 가능 여부.
- 각 case의 실측 처리량과 남은 확인 우선순위.
- 명확한 실패를 바탕으로 바꿀 한 가지 요인과 필요한 continuation control.

시간이 넘으면 예산 때문에 판별 가능한 대조를 삭제하기보다, 낮은 우선순위의 확장을 뒤로 둔다. 충분한 결과 없이 많은 half-run만 늘리지 않는다.

### 15.1 Running campaign에 연결

현재 run을 마구 덮어쓰지 않고 `CONTROLLED`와 `ADAPTIVE-PATH`를 분리한다.

- 같은 parent checkpoint에서 기존 설정을 유지하는 child와 A/B/C를 바꾼 child를 만든다.
- Model/optimizer/scheduler/scaler/data order/RNG/global update를 보존한다.
- Frozen→FT이면 Student aligner 복사본만 열고 optimizer group을 추가한다. Teacher 객체를 공유·수정하지 않는다.
- Scratch로 전환하면 Student aligner와 해당 optimizer state만 reset한다. 전체 Student scratch 여부는 별도 기록한다.
- Teacher·covariance·precision source 변경은 새 cohort/phase로 분리하고 cache/calibration 재생성.
- 유력한 adaptive path는 고정 설정 재시작 또는 사전선언 warm protocol로 재확인한다.

### 15.2 권장 분기 판단

| 관찰 | 다음 작업 |
|---|---|
| A-SIGN 대부분1 | 새로운 정보가 적음. R3와 MASS/COMPADV를 먼저 확인 |
| A/B mask 대부분0 | hard-only와 비교. coefficient 정규화로 숨기지 않음 |
| GV variance 감소·경계흐림 | direct edge, 작은lambda, GC/SC별 단독 대조 |
| Teacher covariance 준비 안 됨 | C-SENS/J 진단만 진행, C-DIAG는 BLOCKED |
| C 감쇠가 모든 고주파에 집중 | Hard 경계 복원 유지 여부와 MASS 확인 |
| C FD 불안정 | FP32/간격/identity sampler/source support를 수정 후 새 버전 |
| Aligner frozen이 병목 | 유력 A/B loss를 고정한 FT 대조, 이후scratch |
| FT+KD가 Teacher offset에 묶임 | 이동량 KD OFF를 별도 비교; scratch만으로 해결됐다고 가정하지 않음 |

---

<a id="config"></a>

## 16. Config template

다음은 **실제 trainer key와 연결할 명세 template**이다. 그대로 실행되는 config라고 주장하지 않는다.

```yaml
campaign_id: S2_W112_TRIABC_20260910
version: v0.1-proposal
server: s2
parent:
  resolved_config: REQUIRED_ACTUAL_PATH
  checkpoint: REQUIRED_OR_FROM_INIT
  source_hash: REQUIRED
  teacher_id: REQUIRED
  teacher_checkpoint_hash: REQUIRED
  controlled_pair: REQUIRED_CONTROL_ID
  kind: CONTROLLED  # or ADAPTIVE_PATH
model:
  width: 112
  depth: [1, 2, 4]
  input_channels: 9
  output_channels: 8
  frame: M
  pan_reconstruction: false
  aligner_policy: A-FR  # A-FT / A-SC are separate factors
input:
  protocol: INHERIT_PARENT
  teacher_view: same_raw_view_own_alignment
  upsampler: INHERIT_PARENT
  augmentation: INHERIT_PARENT
  warp: INHERIT_PARENT
reconstruction:
  mode: REC-R3
  alpha: INHERIT_PARENT
  beta: INHERIT_PARENT
  tau_manifest: REQUIRED_NATIVE_TRAIN_CALIBRATION
  epsilon: INHERIT_PARENT
tri_a:
  enabled: true
  mode: sign  # off / cosine / sign / sign_cap / band_adv
  detached: true
  hard_changed: false
statistics:
  representation: GV  # OFF / GV / GC / SC
  window: 5
  stride: 1
  correction: 0
  roi: INHERIT_STAT_VALID_CENTERS
  mode: AD
  lambda_v: INHERIT_MATCHED_REPRESENTATION_CALIBRATION
  tau_manifest: REQUIRED_IF_ENABLED
tri_b:
  enabled: false
  mode: sign  # sign_cap / component_adv are separate cases
  hard_changed: false
tri_c:
  enabled: false
  mode: diag_cov  # sensitivity / diag_cov / full_quad
  phi: identity  # GV requires differentiation through entire statistic
  h_hr_px: 0.05
  h_check: [0.025, 0.05, 0.10]
  correction_order: [dy, dx]
  jacobian_shape: BD2HW
  probes: central_two_axes_four_forwards
  detach_teacher_jacobian: true
  covariance_source: REQUIRED_FOR_DIAG_COV
  covariance_manifest: REQUIRED_FOR_DIAG_COV
  covariance_units: current_PAN_HR_pixel_squared
  invalid_policy: base_loss_fallback_r1_and_log
  s_c_rule: tau_R_for_identity_tau_V_for_statistics
  hard_attenuation: false
  online_every_update: true
loss:
  reduction: full_domain_mean
  normalize_active_soft: false
  add_parent_full_loss_twice: false
  inherited_aux: INHERIT_PARENT_BY_ALIGNER_POLICY
  shift_kd: INHERIT_FIXED_CONTROL_OR_OFF
training:
  seed: INHERIT_S2_PARENT
  updates: INHERIT_PARENT_FULL_N
  optimizer: INHERIT_PARENT
  scheduler: INHERIT_PARENT_GLOBAL_STEP
  effective_batch: INHERIT_PARENT
  amp: INHERIT_PARENT_STUDENT_ONLY
  teacher_probe_precision: fp32
  review_hours: 24
  hard_timeout: null
evaluation:
  evaluator_manifest: REQUIRED
  views: [raw_original, raw_valid, aligned_valid]
  checkpoint_policy: INHERIT_S2_MANIFEST
  keep_exact_final_N: true
  fixed_roi: true
  test_adaptive_status: REQUIRED_TRUE_OR_FALSE_WITH_PROVENANCE
```

`statistics=OFF`일 때 B가 true이거나, covariance source가 없는 C-DIAG를 켜거나, A-FR에 이동량 KD를 optimizer 항으로 넣는 설정은 resolver가 오류로 차단한다. 단순 경고를 출력한 뒤 조용히 다른 방법으로 실행하지 않는다.

---

## 17. 실행·결과 파일 계약

```text
work_dir/<run_id>/
  config_resolved.yaml
  parent_and_phase_manifest.json
  source_hashes.json
  teacher_assets.json
  calibration_R.json / calibration_V.json / calibration_C.json
  covariance_source_manifest.json
  train_log.jsonl
  routing_components.jsonl
  fd_and_linearity_checks.jsonl
  gradient_diagnostics.jsonl
  compute_budget.json
  checkpoint_metrics.csv
  scene_metrics.csv
  fixed_teacher_error_bins.csv
  stat_under_overfit_bins.csv
  unit_test_results.json
  s2_smoke_results.json
  best_raw.pt / best_aligned.pt / best_rr_val.pt
  last.pt / last_NFULL.pt
```

Status는 `PENDING → IMPLEMENTED → UNIT_PASS → SMOKE_PASS → SCREENED → CONFIRMED/NEGATIVE/REVISE_REQUIRED`로 남긴다. C source 부재는 `BLOCKED_COVARIANCE`, FD 실패는 `BLOCKED_NUMERICS`, 기준 없이 source를 대신한 sensitivity run은 별도 `C-SENS` ID다.

성공한 사례만 보존하지 않는다. 사라진 gate, 무효 covariance, train만 개선된 결과, budget상 미완주 run도 같은 registry에 남긴다.

---

<a id="reference"></a>

## 18. 구현 참고 함수와 확인 범위

다음 부록의 짧은 함수들은 **loss-side 수식의 reference**다. 기존 trainer, W112 model factory, covariance source estimator, data loader, 평가기를 구현하지 않는다. 실제 코드에서는 현재 criterion의 반환 interface와 연결하고 dtype/AMP/accumulation을 계승해야 한다.

문서 작성 시의 로컬 검사 여부와 범위는 부록 끝에 따로 적는다. 로컬 tensor 검사는 s2에서의 위성영상 학습이나 C의 실제 covariance calibration을 대체하지 않는다.

### 18.1 Tensor-only reference

아래 함수는 기존 criterion의 detached weight map을 받아 soft를 다시 계산한다. 코드는 `triabc_math_reference.py`로 옮길 수 있다. **CUDA trainer에서는 이 loss/geometry 블록을 `torch.autocast(device_type="cuda", enabled=False)`로 감싸고, Student U-Net의 AMP와 분리한다.** CPU 검사는 ambient autocast 없이 수행했다.

`diag_risk`의 PSD 검사는 수학적 입력 검증에 불과하다. 실제 정합 정보의 rank·covariance provenance·calibration을 통과시켜 `valid`를 만드는 작업은 별도 adapter 책임이다. `q_report`의 invalid NaN은 진단 표시용이며 optimizer loss에 사용하지 않는다.

```python
"""Tensor-only references for the TRI-ABC addendum; not a W112 trainer."""
from __future__ import annotations
import math
from typing import Callable
import torch
from torch import Tensor


def _dtype(*xs: Tensor) -> torch.dtype:
    return torch.float64 if any(x.dtype == torch.float64 for x in xs) else torch.float32


def _images(s: Tensor, t: Tensor, g: Tensor) -> None:
    if s.ndim != 4 or s.shape != t.shape or s.shape != g.shape or min(s.shape) <= 0:
        raise ValueError("Expected equal, nonempty [B,D,H,W] tensors.")
    if not (s.device == t.device == g.device):
        raise ValueError("Device mismatch.")
    if not all(x.is_floating_point() for x in (s, t, g)):
        raise TypeError("Floating tensors required.")
    if not all(bool(torch.isfinite(x).all()) for x in (s, t, g)):
        raise FloatingPointError("Non-finite prediction/target.")


@torch.no_grad()
def direction_mask(s: Tensor, t: Tensor, g: Tensor, *,
                   mode: str = "sign", eps: float = 1e-6) -> Tensor:
    """Gate only the soft term. D can be bands or statistic components."""
    _images(s, t, g)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite.")
    dt = _dtype(s, t, g)
    u, v = g.to(dt) - s.to(dt), t.to(dt) - s.to(dt)
    if mode == "off":
        return torch.ones_like(u)
    same_sign = (((u > 0) & (v > 0)) | ((u < 0) & (v < 0))).to(dt)
    if mode == "sign":
        return same_sign
    if mode == "sign_cap":
        return same_sign * (u.abs() / (v.abs() + eps)).clamp(max=1)
    if mode == "cosine":
        norm_product = u.norm(dim=1, keepdim=True) * v.norm(dim=1, keepdim=True)
        value = (u * v).sum(dim=1, keepdim=True) / norm_product.clamp_min(eps**2)
        value = torch.where(norm_product > 0, value, torch.zeros_like(value))
        return value.clamp(0, 1).expand_as(u)
    raise ValueError(f"Unknown mode {mode!r}.")


def masked_l1(s: Tensor, t: Tensor, g: Tensor, wh: Tensor, wk: Tensor,
              *, mask: Tensor | None = None, risk: Tensor | None = None) -> dict:
    """Use parent detached weights. wk ALREADY includes the soft coefficient."""
    _images(s, t, g)
    shape = (s.shape[0], 1, *s.shape[-2:])
    if wh.shape != shape or wk.shape != shape:
        raise ValueError("wh/wk must have shape [B,1,H,W].")
    if wh.device != s.device or wk.device != s.device:
        raise ValueError("Weight device mismatch.")
    dt = _dtype(s, t, g, wh, wk)
    s, t, g = s.to(dt), t.detach().to(dt), g.detach().to(dt)
    wh, wk = wh.detach().to(dt), wk.detach().to(dt)
    if not bool(torch.isfinite(wh).all() & torch.isfinite(wk).all()):
        raise FloatingPointError("Non-finite weights.")
    if bool((wh < 0).any() | (wk < 0).any()):
        raise ValueError("Nonnegative weights required.")
    gate = torch.ones_like(s)
    for extra in (mask, risk):
        if extra is not None:
            if extra.device != s.device:
                raise ValueError("Gate device mismatch.")
            extra = torch.broadcast_to(extra.detach().to(dt), s.shape)
            if not bool(torch.isfinite(extra).all()) or bool(((extra < 0) | (extra > 1)).any()):
                raise ValueError("Gates must be finite and in [0,1].")
            gate = gate * extra
    hard = (wh * (s - g).abs().mean(1, keepdim=True)).mean()
    soft = (wk * (gate * (s - t).abs()).mean(1, keepdim=True)).mean()
    return {"loss": hard + soft, "hard": hard, "soft": soft}


@torch.no_grad()
def fd_jacobian(evaluate: Callable[[Tensor], Tensor], mu: Tensor,
                *, h: float = .05) -> Tensor:
    """evaluate(delta) must rerender the ORIGINAL current PAN view through F_T."""
    if mu.ndim != 2 or mu.shape[1] != 2 or not bool(torch.isfinite(mu).all()):
        raise ValueError("mu must be finite [B,2], order dy/dx.")
    if not math.isfinite(h) or h <= 0:
        raise ValueError("Positive finite h required.")
    cols = []
    for axis in (0, 1):
        step = torch.zeros_like(mu)
        step[:, axis] = h
        plus, minus = evaluate(mu.detach() + step), evaluate(mu.detach() - step)
        if plus.ndim != 4 or plus.shape != minus.shape or plus.shape[0] != mu.shape[0]:
            raise ValueError("evaluate must return fixed [B,D,H,W] shape.")
        if not bool(torch.isfinite(plus).all() & torch.isfinite(minus).all()):
            raise FloatingPointError("Non-finite probe outputs.")
        dt = _dtype(plus, minus)
        cols.append((plus.to(dt) - minus.to(dt)) / (2 * h))
    return torch.stack(cols, dim=2)  # [B,D,2,H,W]


@torch.no_grad()
def diag_risk(jac: Tensor, sigma: Tensor, valid: Tensor, *, s_c: float) -> dict:
    """Numerical propagation only. Source calibration/identifiability is external."""
    if jac.ndim != 5 or jac.shape[2] != 2:
        raise ValueError("jac must be [B,D,2,H,W].")
    b = jac.shape[0]
    if sigma.shape != (b, 2, 2) or valid.shape != (b,) or valid.dtype != torch.bool:
        raise ValueError("sigma [B,2,2] and bool valid [B] required.")
    if not (jac.device == sigma.device == valid.device):
        raise ValueError("Device mismatch.")
    if not math.isfinite(s_c) or s_c <= 0 or not bool(torch.isfinite(jac).all()):
        raise ValueError("Finite jac and positive finite s_c required.")
    dt = _dtype(jac, sigma)
    j = jac.to(dt)
    safe_sigma = torch.where(valid[:, None, None], sigma.to(dt), torch.zeros_like(sigma, dtype=dt))
    if not bool(torch.isfinite(safe_sigma).all()):
        raise FloatingPointError("Valid covariance contains non-finite entries.")
    if not torch.allclose(safe_sigma, safe_sigma.transpose(-1, -2), atol=1e-8, rtol=1e-5):
        raise ValueError("Symmetric covariance required.")
    safe_sigma = .5 * (safe_sigma + safe_sigma.transpose(-1, -2))
    if bool((torch.linalg.eigvalsh(safe_sigma) < -1e-8).any()):
        raise ValueError("Covariance is not positive semidefinite.")
    # Autocast must be disabled by the calling geometry/loss block on CUDA.
    q = torch.einsum("bdihw,bij,bdjhw->bdhw", j, safe_sigma, j).clamp_min(0)
    r = s_c**2 / (s_c**2 + q)
    valid_map = valid[:, None, None, None]
    r = torch.where(valid_map, r, torch.ones_like(r))
    # NaN is deliberate in diagnostics: invalid is not zero uncertainty.
    q_report = torch.where(valid_map, q, torch.full_like(q, float("nan")))
    return {"risk": r, "q_report": q_report, "valid": valid.detach()}


def lowrank_quad(d: Tensor, jac: Tensor, sigma: Tensor, *, s_c: float) -> Tensor:
    """[B,1,H,W] quadratic soft; compare against SAME quadratic with W=I."""
    if d.ndim != 4 or jac.shape != (d.shape[0], d.shape[1], 2, *d.shape[-2:]):
        raise ValueError("d [B,D,H,W] and matching jac [B,D,2,H,W] required.")
    if sigma.shape != (d.shape[0], 2, 2) or not math.isfinite(s_c) or s_c <= 0:
        raise ValueError("Bad covariance shape/scale.")
    dt = _dtype(d, jac, sigma)
    with torch.no_grad():
        ss = sigma.detach().to(dt)
        if not torch.allclose(ss, ss.transpose(-1, -2), atol=1e-8, rtol=1e-5):
            raise ValueError("Symmetric covariance required.")
        eig, vec = torch.linalg.eigh(ss)
        if not bool(torch.isfinite(eig).all()) or bool((eig < -1e-8).any()):
            raise ValueError("Finite PSD covariance required.")
        root = vec * eig.clamp_min(0).sqrt().unsqueeze(-2)
        jp = jac.detach().to(dt).permute(0, 3, 4, 1, 2)
        u = (jp @ root[:, None, None, :, :]) / s_c
        small = torch.eye(2, device=d.device, dtype=dt) + u.transpose(-1, -2) @ u
    dp = d.to(dt).permute(0, 2, 3, 1).unsqueeze(-1)
    z = u.transpose(-1, -2) @ dp
    energy0 = (dp * dp).sum(dim=(-2, -1))
    correction = (z * torch.linalg.solve(small, z)).sum(dim=(-2, -1))
    energy = energy0 - correction
    if bool((energy.detach() < -1e-5 * energy0.detach().clamp_min(1e-12)).any()):
        raise FloatingPointError("Woodbury subtraction unstable; do not silently clip.")
    return (energy.clamp_min(0) / (d.shape[1] * s_c)).unsqueeze(1)
```

### 18.2 이번 문서 작성 중 수행한 로컬 확인

**CPU PyTorch 2.10.0+cpu에서 tensor-only 검사 22개를 실행했고, 22개 통과·실패0·오류0을 확인했다.** 검사는 다음 범위다.

- mask=1일 때 기존 reconstruction criterion과 loss·Student gradient 동치.
- 같은 오답/정확한 GT/밴드별 충돌/반대방향/cap/cosine0 사례.
- Teacher·GT·gate detach, Student live gradient, mask0의 hard-only 동치.
- 성분별 우위와 sign 조건의 중복 관계, 통계 표현을 통한 Student gradient.
- 선형 toy의 FD Jacobian, 분석적 공분산 대각, invalid fallback, PSD 증가에 대한 감쇠 단조성.
- Woodbury2×2 solve와 직접8×8 solve의 동치, Teacher J/Sigma의 gradient 차단.
- 잘못된 shape·비PSD covariance 탐지, L2 설명식의 대수적 동치.

**검사하지 않은 것:** 실제 PAN/MS 영상의 covariance calibration, W112 복원망과 aligner의 전체 forward, s2 코드 연결, CUDA/AMP·실배치·처리시간, 실제 학습 및 RR/FR 성능. §12의 integration 검사는 실행자가 별도로 수행해야 한다.

기존 상위 문서의 22개 검사와 같은 실행을 재인용한 것이 아니라, 이 추가안의 참고 함수에 대해 새로 수행한 22개 수학 검사다. 전체 trainer가 준비됐다는 의미는 아니다.

---

## 19. 출처·재현 정보

### 19.1 내부 근거

| Ref | 자료 | 사용한 범위 |
|---|---|---|
| [S0] | `PAN_S2_W112_D124_KD_Variance_Adaptive_Campaign_2026-09-10.md` | W112/s2·R3·통계 계약·aligner 정책·평가·adaptive path |
| [S1] | `PAN_W96_D124_GT_Anchored_KD_Experiment_Plan_2026-09-09.md` | 원래 reconstruction 및 이동량 covariance의 의미. 폭/서버는 superseded |
| [S2] | `PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2 (1).md` | 원래 PAN sampler, aligner, A2/A3 geometry/gradient 경로 |
| [S3] | `PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md` | N계열의 상대 offset, view crop, sequential sampling |
| [S4] | `2026-09-03_alignment-audit-s2-detail.md` | spectral 차이의 정합 측정 하한, 해상도/위상·단위 주의 |
| [S5] | `pan_gt_anchored_kd.py` | 현재 criterion의 weight 반환 계약과 이번 parent 동치 검사 |

이번 사용자 요청은 **A/B/C를 추가 실행할 수 있는 명세로 작성**하는 것이므로, 이전의 컨셉 검토보다 구체적인 수식·게이트·후속 대조를 제안했다. 이 새 제안을 선행 논문이나 기존 실험에서 이미 검증된 것으로 인용하지 않는다.

### 19.2 외부 1차 자료

**[R1] TGeo-KD.** Less or More From Teacher: Exploiting Trilateral Geometry For Knowledge Distillation. ICLR 2024. 삼각관계와 bilevel 기반 융합비율의 개념 출처다. 본 문서의 sign/cap/C 전파 수식의 출처는 아니다.

- 공식 논문 기록: `https://proceedings.iclr.cc/paper_files/paper/2024/hash/6aa4967920e495e90aeeaa3acf18d019-Abstract-Conference.html`
- 확인한 arXiv v3 HTML: `https://arxiv.org/html/2312.15112v3`
- 사용자 제시 PDF: `https://arxiv.org/pdf/2312.15112`

**[R2] Weighted Soft Labels.** Rethinking Soft Labels for Knowledge Distillation: A Bias–Variance Tradeoff Perspective. ICLR 2021. Soft 감독의 역할을 샘플별로 구분하는 문제의식의 출처다.

- 공식 발표 기록: `https://iclr.cc/virtual/2021/poster/2807`

**[R3] 저자 공개 구현.** Weighted-Soft-Label-Distillation의 `knowledge_distiller.py`. 분류용 Teacher/Student GT loss 관계를 참고했지만 그 코드를 본 회귀 loss에 복사하지 않았다.

- `https://raw.githubusercontent.com/bellymonster/Weighted-Soft-Label-Distillation/master/knowledge_distiller.py`

**[R4] Ceres Solver 공식 Covariance Estimation 문서.** Least-squares covariance의 residual scaling·rank 조건을 확인하는 배경 자료다. 본 PAN–MS descriptor에서 올바른 covariance를 얻었다는 보장은 아니다.

- `https://ceres-solver.readthedocs.io/latest/nnls_covariance.html`

**[R5] PyTorch 공식 API.** `grid_sample`, `no_grad`의 sampling 및 autograd 동작 확인용이다. s2에 문서의 최신 버전이 설치됐다고 가정하지 않는다. 이번 참고 검사는 위에 명시한 로컬 CPU 버전에서만 수행했다.

- `https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grid_sample.html`
- `https://docs.pytorch.org/docs/stable/generated/torch.no_grad`

공개 자료 확인일: 2026-09-10. 외부 연구의 성능 결과를 본 팬샤프닝 추가안의 결과로 옮기지 않는다.

### 19.3 참조한 로컬 원본의 SHA256

상위 파일을 수정하지 않고 다음 bytes를 참고했다. 실제 s2의 동일 파일·코드라고 가정하지 말고 server manifest와 대조한다.

| 파일 | SHA256 |
|---|---|
| `PAN_S2_W112_D124_KD_Variance_Adaptive_Campaign_2026-09-10.md` | `482303d6adbfbdab0171c32755eb3f0333d191bcb1aa77dd1c3d41ad440ea96e` |
| `PAN_W96_D124_GT_Anchored_KD_Experiment_Plan_2026-09-09.md` | `32d3907fcfa0a89e1fb5424a006c7596d88d8b728dd9120c0d64aaf197a3ac63` |
| `pan_gt_anchored_kd.py` | `36a69ad5516a635a86e895c8af291d3f8253fc49e6f3326d4862edaa3fc0f2a9` |

---

## 20. 실행자용 최종 인계

**먼저** 현재 s2 W112 Teacher·donor·parent loss·input-view·평가 manifest를 연결한다. Aligner가 아직 미확정이면 Teacher donor를 frozen으로 재사용해 loss 효과를 분리하고, 동일 loss의 FT/scratch를 후속 대응 비교한다.

**A는** 기존 R3의 hard/soft 배분을 유지하되, Student→GT와 Student→Teacher의 방향이 충돌하는 band의 soft만 줄인다. 첫 구현은 `TRI-A-SIGN`, 이어 COS/CAP과 scalar-mass·band-adv 대조다.

**B는** 기존 GV-AD의 평균 gate를 그대로 두고 통계 성분별 sign/cap을 추가한다. GT hard-stat은 유지한다. GC/SC 확장은 같은 표현의 AD 기준과 비교하며, 통계 우위의 세분화는 별도 COMPADV다.

**C는** Teacher correction을 직접 ±h 변화시켜 **전체 복원망의 출력 Jacobian**을 구한다. Covariance가 없으면 C-SENS로 민감도만 평가한다. 유효한 Sigma가 있으면 `J Sigma Jᵀ`의 대각으로 soft만 감쇠한다. 이것은 GT를 약화하거나 Teacher의 HRMS를 정답에 재정렬하는 방법이 아니다.

**마지막으로** mask들이 KD를 거의 없애서 얻은 효과인지 weighted GT-only·통계 WH·MASS 대조로 구분한다. 문서의 새 수식을 원 논문의 그대로인 구현 또는 이미 확보한 신규성으로 부르지 않는다. 효과가 확인되면 어떤 Teacher 제약을 제거하고 어떤 구조 지식을 살렸는지에 근거해 방법을 정리한다.
