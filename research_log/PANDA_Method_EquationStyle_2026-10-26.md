# PANDA — Method 수식 전개 통일안

> 2026-10-26 · 논문 Method 서술용 정리. 기호는 현재 합의된 PANDA 표기를 유지하고, 식의 전개를 **입력·출력 정의 → 핵심 감독식 → 최종 objective** 순서로 단순화했다. 이 문서는 수정된 연구 설계의 **표현안**이며, 기존 학습 체크포인트의 구현 상태를 소급해 변경하지 않는다.

## 1. 공통 Notation과 평균 규약

- $\mathbf I_{\mathrm{pan}}$: 원본 PAN, $\mathbf I_{\mathrm{ms}}$: 원본 저해상도 MS.
- $\mathbf I_{\mathrm{ms}}^{\mathrm{lr}}$: $\mathbf I_{\mathrm{ms}}$를 4배 업샘플한 MS. 이 문서에서는 사용자가 선택한 $\mathrm{lr}$ 표기를 유지한다.
- $\mathbf I_{\mathrm{ms}}^{\mathrm{LMS}}$: PAN 격자에 맞춰 데이터셋이 제공한 Stored LMS. **Aligner가 받는 8밴드 MS 참조**로, 원본 $\mathbf I_{\mathrm{ms}}$ 및 $\mathbf I_{\mathrm{ms}}^{\mathrm{lr}}$와 구분한다.
- $\mathbf I_{\mathrm{ms}}^{\mathrm{GT}}$: RR 학습의 HRMS GT.
- $\mathcal A_\phi$: PAN Aligner, $\mathcal P_T$: Teacher U-Net, $\mathcal P_S$: Student U-Net.
- $\mathbf u_0$: 원본 PAN의 예측 보정, $\mathbf u_\epsilon$: 합성 이동 PAN의 예측 보정. 각 벡터는 $(y,x)$ 순서의 **두 실수 성분**이다.
- $\mathcal W$: PAN warp, $\tilde{\mathbf I}_{\mathrm{pan}}$: 보정 PAN, $\mathbf I_{\mathrm{pan}}^{\mathrm{res}}$: 보정 PAN과 원본 PAN의 **부호 있는 차영상**.
- $\hat{\mathbf I}_{\mathrm{ms}}^T$, $\hat{\mathbf I}_{\mathrm{ms}}^S$: Teacher와 Student의 HRMS 예측.
- $B$: 미니배치의 이미지 수; $n\in\{1,\ldots,B\}$: 이미지 인덱스; $C=8$: MS 밴드 수; $p$: 영상 픽셀 위치.
- $\operatorname{sg}(\cdot)$: stop-gradient; $m^{(1)},m^{(2)}$: 각각 Stage 1 및 Stage 2 유효 PAN 관측 support.

원본 MS에 대한 업샘플링은

$$
\mathbf I_{\mathrm{ms}}^{\mathrm{lr}}=\mathcal U_{23}(\mathbf I_{\mathrm{ms}}).
\tag{1}
$$

밴드별 평균 절대오차 지도는 다음처럼 정의하고, 두 Stage에서 같은 형식을 사용한다.

$$
\begin{aligned}
e_T(p)&=\frac1C\sum_{c=1}^C\left|\hat I_{\mathrm{ms},c}^T(p)-I_{\mathrm{ms},c}^{\mathrm{GT}}(p)\right|,\\
e_S(p)&=\frac1C\sum_{c=1}^C\left|\hat I_{\mathrm{ms},c}^S(p)-I_{\mathrm{ms},c}^{\mathrm{GT}}(p)\right|.
\end{aligned}
\tag{2}
$$

명료성을 위해 식 (2) 등에서 배치 인덱스 $n$은 생략한다. 픽셀별 단일 채널 지도 $f$의 학습 평균은

$$
\mathcal R_m[f]=\frac1B\sum_{n=1}^B\frac{\sum_{p}m_n(p)f_n(p)}{\sum_{p}m_n(p)}.
\tag{3}
$$

논문 본문에서는 이 평균을 한 번 정의한 뒤 반복해서 모든 유효 픽셀의 합을 펼칠 필요가 없다.

## 2. Stage 1 — PAN Alignment Learning

Stage 1에서는 PAN Aligner $\mathcal A_\phi$와 Teacher $\mathcal P_T$를 공동 학습한다. Aligner는 원본 PAN과 **Stored LMS 8밴드**로부터 영상별 이동량을 예측하고, PAN을 보정한다.

$$
\boxed{\begin{aligned}
\mathbf u_0&=\mathcal A_\phi(\mathbf I_{\mathrm{pan}},\mathbf I_{\mathrm{ms}}^{\mathrm{LMS}}),\\
\tilde{\mathbf I}_{\mathrm{pan}}&=\mathcal W(\mathbf I_{\mathrm{pan}},\mathbf u_0).
\end{aligned}}
\tag{4}
$$

$\mathcal W$의 sampling 좌표는 $(y+u^y,x+u^x)$로 정의한다. 이동량은 PAN 격자 픽셀 단위를 사용하며, RR·FR에서 임의의 추가 배율을 적용하지 않는다.

Teacher는 보정 PAN과 업샘플 MS를 이용해 HRMS 잔차를 추정한다.

$$
\boxed{\hat{\mathbf I}_{\mathrm{ms}}^T=\mathbf I_{\mathrm{ms}}^{\mathrm{lr}}+\mathcal P_T\!\left([\tilde{\mathbf I}_{\mathrm{pan}},\mathbf I_{\mathrm{ms}}^{\mathrm{lr}}]\right).}
\tag{5}
$$

### 2.1 Reconstruction supervision

$$
\boxed{\mathcal L_{\mathrm{rec}}=\mathcal R_{m^{(1)}}[e_T].}
\tag{6}
$$

이는 **전체 $C=8$밴드**의 HRMS 정답에 대한 평균 절대오차로, Aligner와 Teacher에 함께 gradient를 제공한다.

### 2.2 RGB-based structural alignment

두 단계의 구조 감독 참조는 **동일한 GT RGB 평균**으로 통일한다.

$$
\mathbf I_{\mathrm{ms}}^{\mathrm{RGB}}=\frac13\left(\mathbf I_{\mathrm{ms},R}^{\mathrm{GT}}+\mathbf I_{\mathrm{ms},G}^{\mathrm{GT}}+\mathbf I_{\mathrm{ms},B}^{\mathrm{GT}}\right).
\tag{7}
$$

고정 Scharr 연산자 $S_x,S_y$를 사용하여

$$
\boxed{\mathcal L_{\mathrm{abs}}=\frac12\,\mathbb E_{p\in\Omega_8}\!\left[\sum_{k\in\{x,y\}}\left|\,|S_k*\tilde{\mathbf I}_{\mathrm{pan}}|(p)-|S_k*\mathbf I_{\mathrm{ms}}^{\mathrm{RGB}}|(p)\,\right|\right].}
\tag{8}
$$

여기서 $\mathbb E_{p\in\Omega_8}$는 배치의 영상 및 고정 ROI8 위치에 대한 평균이다. 구조 감독의 참조만 RGB 평균이며 Aligner 입력 MS는 8밴드 Stored LMS 그대로다.

### 2.3 Controlled PAN shift consistency

영상별 알려진 합성 이동 $\boldsymbol\epsilon=(\epsilon^y,\epsilon^x)$을 PAN에만 적용한다.

$$
\boxed{\begin{aligned}
\mathbf I_{\mathrm{pan}}^\epsilon&=\mathcal W(\mathbf I_{\mathrm{pan}},\boldsymbol\epsilon),\\
\mathbf u_\epsilon&=\mathcal A_\phi(\mathbf I_{\mathrm{pan}}^\epsilon,\mathbf I_{\mathrm{ms}}^{\mathrm{LMS}}).
\end{aligned}}
\tag{9}
$$

$\boldsymbol\epsilon$은 RR PAN 격자에서 반경 $R_\epsilon=3$픽셀인 원판에 **면적 균일**하게 분포한다. $\mathcal W$는 입력 좌표 $(y+\epsilon^y,x+\epsilon^x)$를 sampling하므로 이상적인 Aligner의 상대 응답은 $\mathbf u_\epsilon+\boldsymbol\epsilon\approx\mathbf u_0$이다. 이 조건의 loss는

$$
\boxed{\mathcal L_\epsilon=\frac1{2B}\sum_{n=1}^{B}\left\|\mathbf u_{\epsilon,n}+\boldsymbol\epsilon_n-\operatorname{sg}(\mathbf u_{0,n})\right\|_1.}
\tag{10}
$$

**분모 해설**: $n$은 배치 속 $n$번째 영상이고, $B$는 영상 개수다. $\|\cdot\|_1$은 해당 영상의 $(y,x)$ **두 성분 절댓값의 합**이므로 $2B$로 나누면 모든 배치의 두 축에 대한 MAE가 된다. 예를 들어 $B=12$이면 24개의 좌표 오차를 평균한다. 정규화는 식의 scale만 정하며 상대 이동 제약의 의미를 바꾸지 않는다. $\operatorname{sg}$는 $\mathcal L_\epsilon$의 native 기준 출력에만 적용한다.

마지막으로 Stage 1의 전체 목적함수는

$$
\boxed{\mathcal L_{\mathrm{S1}}=\mathcal L_{\mathrm{rec}}+\lambda_{\mathrm{abs}}\mathcal L_{\mathrm{abs}}+\lambda_\epsilon\mathcal L_\epsilon.}
\tag{11}
$$

Stage 1 완료 후 같은 checkpoint에서 추출한 Aligner와 Teacher를 Stage 2에서 동결한다.

## 3. Stage 2 — Adaptive Reconstruction with PAN Correction Residual

Stage 2에서 $\mathcal A_\phi$, $\mathcal P_T$는 고정하고, $\mathcal P_S$만 학습한다. 보정 PAN은 식 (4)를 재사용하며, **새 Aligner나 $\tilde{\mathbf I}_{\mathrm{pan}}^T$를 추가 정의하지 않는다.**

### 3.1 PAN correction residual 입력

$$
\boxed{\mathbf I_{\mathrm{pan}}^{\mathrm{res}}=\tilde{\mathbf I}_{\mathrm{pan}}-\mathbf I_{\mathrm{pan}}.}
\tag{12}
$$

이 차이는 절댓값을 취하지 않는 *signed difference*다. 입력에는 영상 강도 차이의 양·음 부호가 유지된다. Student의 HRMS 예측은

$$
\boxed{\hat{\mathbf I}_{\mathrm{ms}}^S=\mathbf I_{\mathrm{ms}}^{\mathrm{lr}}+\mathcal P_S\!\left([\tilde{\mathbf I}_{\mathrm{pan}},\mathbf I_{\mathrm{pan}}^{\mathrm{res}},\mathbf I_{\mathrm{ms}}^{\mathrm{lr}}]\right).}
\tag{13}
$$

8밴드 WV3의 Student 입력은 $1+1+8=10$채널이다.

### 3.2 Reference reconstruction difficulty

Teacher error $e_T$는 식 (2), 식 (5)의 **고정 Teacher 예측**에서 계산한다. 이를 GT 학습 가중치로 변환한다.

$$
\boxed{d_e(p)=\operatorname{sg}\!\left(\frac{e_T(p)}{e_T(p)+\tau_e}\right).}
\tag{14}
$$

$\tau_e$는 augmentation 없이 수집한 RR train의 Teacher error 중 Stage 2 유효 support 위치에 대한 전역 중앙값이다. 이 지도는 Teacher 출력의 모방 목표가 아닌 **GT 감독의 상대 중요도**를 제공한다.

### 3.3 Conditional spatial discrepancy

GT RGB 구조 참조 식 (7)을 공유하되, Stage 2에서는 다음 두 공간 신호를 정의한다.

$$
\boxed{\begin{aligned}
g(p)&=\frac12\sum_{k\in\{x,y\}}|S_k*\mathbf I_{\mathrm{ms}}^{\mathrm{RGB}}|(p),\\
q(p)&=\frac12\sum_{k\in\{x,y\}}\left|\,|S_k*\tilde{\mathbf I}_{\mathrm{pan}}|(p)-|S_k*\mathbf I_{\mathrm{ms}}^{\mathrm{RGB}}|(p)\,\right|.
\end{aligned}}
\tag{15}
$$

$g$는 GT edge strength이고, $q$는 보정 PAN과 GT의 Scharr 절대응답 차이이다. 강한 GT edge에서 $q$도 커질 수 있으므로 전체 RR train의 $g$ 분위수로 $K=5$개 그룹을 구성하고 그룹 $j(p)$에서 $q$의 경험적 누적분포를 계산한다.

$$
\boxed{\begin{aligned}
\widehat F_j(t)&=\frac{1}{N_j}\sum_{(i,p')\in\mathcal C_j}\mathbf1[q_i(p')\le t],\\
r_q(p)&=\widehat F_{j(p)}(q(p)),\\
\bar r_q(p)&=\begin{cases}r_q(p),&p\in\Omega_8,\\0,&p\notin\Omega_8.\end{cases}
\end{aligned}}
\tag{16}
$$

$\mathcal C_j$는 유사한 GT edge strength를 가지는 RR train calibration 픽셀들의 집합이고 $N_j=|\mathcal C_j|$이다. $r_q$는 상대 discrepancy의 **조건부 순위**이며 실제 정합이 틀렸을 확률이 아니다. ECDF 경계·동점·빈 그룹 처리 등 재현 세부사항은 Appendix에 둔다.

### 3.4 Adaptive reconstruction objective

Teacher difficulty와 conditional spatial discrepancy를 결합한 공간 가중치는

$$
\boxed{w_{\mathrm{rec}}(p)=1+\alpha d_e(p)+\gamma d_e(p)\bar r_q(p).}
\tag{17}
$$

Student의 유일한 학습 목표는 원본 HRMS GT를 복원하는 것이다.

$$
\boxed{\mathcal L_{\mathrm{S2}}=\mathcal L_{\mathrm{AR}}=\mathcal R_{m^{(2)}}[w_{\mathrm{rec}}e_S].}
\tag{18}
$$

$\mathcal P_S$로만 gradient가 전달되며 Teacher–Student Soft imitation, 추가 ε-consistency 및 별도 Student spatial loss는 적용하지 않는다. $\alpha,\gamma$는 spatial reweighting 강도이고, $\mathcal R_{m^{(2)}}$는 가중치 합이 아닌 **유효 픽셀 수**로 나누므로 손실 scale도 영향을 받는다.

## 4. Inference

추론에서 Teacher, GT, $\epsilon$, $g$, $q$, $r_q$ 및 calibration은 사용하지 않는다.

$$
\boxed{\begin{aligned}
\mathbf u_0&=\mathcal A_\phi(\mathbf I_{\mathrm{pan}},\mathbf I_{\mathrm{ms}}^{\mathrm{LMS}}),\\
\tilde{\mathbf I}_{\mathrm{pan}}&=\mathcal W(\mathbf I_{\mathrm{pan}},\mathbf u_0),\\
\mathbf I_{\mathrm{pan}}^{\mathrm{res}}&=\tilde{\mathbf I}_{\mathrm{pan}}-\mathbf I_{\mathrm{pan}},\\
\hat{\mathbf I}_{\mathrm{ms}}^S&=\mathbf I_{\mathrm{ms}}^{\mathrm{lr}}+\mathcal P_S\!\left([\tilde{\mathbf I}_{\mathrm{pan}},\mathbf I_{\mathrm{pan}}^{\mathrm{res}},\mathbf I_{\mathrm{ms}}^{\mathrm{lr}}]\right).
\end{aligned}}
\tag{19}
$$

## 5. 논문 표기와 실제 구현을 구분해야 하는 두 가지

1. **Stage 1 RGB 구조 참조는 새 Method 결정사항이다.** 과거 `panda_band30/aligner.py` / `panda_mature30/stage1.py`에서 훈련된 G8 donor는 GT 8밴드 평균 Scharr supervision을 사용한다. 논문식 RGB Stage 1을 실증하려면 해당 Loss를 바꾸고 Stage 1 재학습, donor 재선택 및 Stage 2 재학습이 필요하다.
2. **공통 $\mathcal W$는 논문에서 합의한 단일 Warp 표기다.** 현재 `panda_mature30` Stage 1 / 고정 Teacher는 bicubic, `panda_follow30` Student는 Lanczos-4이므로 *기존 코드의 실제 kernel이 동일하다*고 주장하면 안 된다. 하나의 연산으로 구현까지 통일하려면 해당 경로 변경과 재실험이 필요하다. 반면 수식에서 인터페이스만 추상화할 목적이라면 공통 $\mathcal W$를 쓰되 구현 세부에서 실제 operator 차이를 공개해야 한다.

## 6. 논문 본문과 Appendix의 역할 분리

- **Method 본문:** 식 (1), (4)–(18) 중 핵심 관계와 최종 두 Stage objective만 배치. Batch 인덱스는 epsilon과 재현성 정의에서만 명시.
- **Appendix / Implementation:** 식 (3)의 MASK support 평균, exact conditional ECDF의 세부 정렬·동점 정책, Scharr 커널/정규화, interp23 augmentation phase, 이동량 support 검사, Teacher/Student 초기화와 checkpoint 선택 규약.
