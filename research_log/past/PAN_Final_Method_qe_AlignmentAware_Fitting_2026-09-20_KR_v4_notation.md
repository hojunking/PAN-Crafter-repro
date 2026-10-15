# 최종 방법론 정리 — q/e 기반 Alignment-Aware Student Fitting

**Target:** CVPR 2027 Pansharpening  
**Method status:** 논문 작성용 현재 formulation (v4, notation을 PAN-Crafter 기준으로 정렬)  
**핵심 개념:** PAN–MS 정합의 불안정성과 HRMS 복원의 어려움을 각각 **alignment-consistency error \(q\)**와 **reconstruction error \(e\)**로 분리하고, 두 신호를 서로 다른 supervision 경로에 사용함.

> **Notation policy.** 본 문서의 기호는 PAN-Crafter [Do et al., CVPR-style preprint 2025]의 표기를 기준으로 정렬함.
> 1. 영상 변수는 \(\mathbf I^{\text{tag}}_{\text{modality}}\) 형식 (\(\mathbf I_{\mathrm{pan}}\), \(\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\), \(\mathbf I^{\mathrm{hr}}_{\mathrm{ms}}\)), 예측값은 hat (\(\hat{\mathbf I}^{\mathrm{hr}}_{\mathrm{ms}}\)).
> 2. 네트워크는 calligraphic + parameter subscript (\(\mathcal P_\theta\), \(\mathcal A_\phi\)).
> 3. 결정적 연산은 roman operator 이름 (\(\operatorname{Conv}\), \(\operatorname{Warp}\), \(\operatorname{Up}_4\)), channel-wise concatenation은 \([\,\cdot\,|\,\cdot\,]\).
> 4. loss weight는 모두 \(\lambda_\bullet\), loss는 \(\mathcal L_\bullet\), L1은 \(\|\cdot\|_1\).
>
> v3에서 사용하던 \(P,S,M,Y,Z,\mathbf c,\alpha,\beta\) 등은 8절 대응표에 따라 교체함.

---

# 1. Problem Definition

## 1.1 입력 구조와 기준 좌표계

Pansharpening은 고해상도 PAN 영상의 **spatial detail**과 저해상도 MS 영상의 **spectral information**을 결합하여 HRMS 영상을 복원하는 문제임. Paired dataset을 PAN-Crafter와 동일하게

\[
\mathcal D=\left\{\left(\mathbf I_{\mathrm{pan}},\ \mathbf I_{\mathrm{ms}},\ \mathbf I^{\mathrm{hr}}_{\mathrm{ms}}\right)\right\},
\qquad
\mathbf I_{\mathrm{pan}}\in\mathbb R^{4H\times4W\times1},
\quad
\mathbf I_{\mathrm{ms}}\in\mathbb R^{H\times W\times C_{\mathrm{ms}}}
\]

으로 두고, MS 입력을 PAN 해상도로 4배 확대하여

\[
\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}
=
\operatorname{Up}_4\!\left(\mathbf I_{\mathrm{ms}}\right)
\in\mathbb R^{4H\times4W\times C_{\mathrm{ms}}}
\]

으로 둠. HRMS 정답 \(\mathbf I^{\mathrm{hr}}_{\mathrm{ms}}\), residual base \(\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\), 최종 출력은 모두 **MS reference frame**에 고정함. 따라서 본 연구에서는 MS/GT/output을 PAN 좌표계로 이동시키지 않고, PAN의 spatial information을 MS frame에 맞추는 방향으로 문제를 구성함.

> **용어 주의.** PAN-Crafter 규약에서 \(\mathbf I_{\mathrm{ms}}\)는 native 해상도(\(H\times W\)) MS이고, **LRMS(\(\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\))는 이미 ×4 up-sampling된 \(4H\times4W\) 영상**임. v3에서 "LRMS 입력 \(S\)" / "upsampled MS \(M\)"로 부르던 것과 용어 대응이 바뀌므로 본문 전체에서 이 규약으로 통일함.

**Teacher / Student 입력 구분.** 본 문서의 최종 구성은 **Teacher P0: 보정된 PAN + LRMS**, **Student PLH: 보정된 PAN + LPAN + HPAN + LRMS**임. Teacher reconstruction network는 LPAN·HPAN을 추가 입력으로 받지 않으며, Stage 2의 frozen Teacher도 동일한 P0 입력을 유지함. Student reconstruction network에는 4.1절에서 정의하는 정합된 저주파·고주파 PAN을 함께 제공함. 입력 채널 수는 각각 \(C_{\mathrm{ms}}+1\), \(C_{\mathrm{ms}}+3\)이며, **Teacher와 Student의 PAN Aligner는 모두 PAN과 LRMS만 입력받음**.

---

## 1.2 PAN–MS residual misalignment

PAN과 MS는 서로 다른 센서 경로와 해상도에서 획득되므로, 전처리 이후에도 작은 spatial displacement가 잔존할 수 있음. 아래 예시는 WV3 full-resolution 영상에서 PAN–MS 사이의 local displacement가 존재하며, 단순 translation만으로 두 영상의 구조적 상관이 크게 증가하는 경우를 보여줌.

![Observed PAN–MS displacement example](./fig1_pan_ms_misalignment_example.png)

이러한 위치 오차를 고려하지 않고 PAN의 고주파 구조를 직접 주입하면 GT와 다른 위치의 edge가 강화되어 double-edge 또는 blur가 발생할 수 있음. 반대로 PAN detail을 일괄적으로 약화하면 spatial fidelity가 저하됨. 따라서 **PAN-derived geometry를 어느 정도 신뢰할 수 있는지 판단하는 정합 관련 신호가 필요함.**

---

## 1.3 정합 오차와 재구성 오차의 분리

정합 상태와 복원 난이도는 동일하지 않음. 실제 patch 분석에서는 정합 반응은 안정적이지만 복원이 어려운 영역, 또는 복원은 잘 되지만 정합 반응은 불안정한 영역이 모두 관찰됨.

![q/e patch quadrants](./fig2_qe_patch_quadrants.png)

본 연구에서는 이를 두 종류의 오류 신호로 분리함. 아래 quadrant 그림에서 \(e\)는 patch 수준에서 집계한 reconstruction error를 의미하며, 실제 학습에서는 Teacher와 Student의 위치별 오류 \(e_{\mathrm T}\), \(e_{\mathrm S}\)를 사용함.

- **\(q\): alignment-consistency error**  
  알려진 PAN perturbation에 대해 Teacher PAN Aligner의 보정 예측이 얼마나 일관되게 변하는지 측정하는 **정합 일관성 오차**
- **\(e\): reconstruction error**  
  Teacher 또는 Student의 HRMS prediction이 GT와 얼마나 차이나는지 측정하는 **재구성 오차**

핵심 원리는 다음과 같음.

> **\(e\)는 GT hard supervision과 Teacher soft supervision의 상대적 비중을 조절하고, \(q\)는 geometry-sensitive edge supervision과 Student Aligner adjustment의 신뢰도를 조절함.**

즉 하나의 uncertainty 값으로 모든 학습 신호를 제어하는 것이 아니라, **복원 오류와 정합 오류를 명시적으로 분리하여 서로 다른 역할에 사용**함.

---

# 2. Overall Framework

전체 학습은 두 단계로 구성함.

```text
Stage 1. Teacher / PAN Aligner Training
Native PAN + LRMS
      ↓
Teacher Aligner A_{phi_T} → Aligned PAN → Teacher P_{theta_T} → HRMS
      │                                        ↓
      └──────────────── Reconstruction supervision

Shifted PAN
      ↓
Same Aligner A_{phi_T}
      ↓
Relative-shift consistency

                ↓ Teacher P_{theta_T} / Teacher Aligner A_{phi_T} freeze

Stage 2. Student Training
Frozen Teacher   P_{theta_T} → Teacher prediction
Frozen Aligner   A_{phi_T}   → alignment-consistency cue q

PAN + LRMS → Student Aligner A_{phi_S} → correction u^S
                                          ↓
PAN + Up_4(LPAN) → Shared Warp → Aligned PAN / LPAN; HPAN = Aligned PAN − Aligned LPAN
                                          ↓
[Aligned PAN | Aligned LPAN | HPAN | LRMS] → Student P_{theta_S} → Student prediction
                                                                    │
                                              e-guided reconstruction supervision
                                              q-guided edge supervision
                                              q-guided Aligner adjustment
```

- **Stage 1:** PAN Aligner \(\mathcal A_{\phi_{\mathrm T}}\)가 HRMS 복원에 유효한 native correction과 알려진 PAN shift에 대한 일관된 반응을 학습함.
- **Stage 2:** Stage 1에서 얻은 \(\mathcal P_{\theta_{\mathrm T}}\)와 \(\mathcal A_{\phi_{\mathrm T}}\)를 고정하고, \(e\)와 \(q\)를 이용해 \(\mathcal P_{\theta_{\mathrm S}}\)와 \(\mathcal A_{\phi_{\mathrm S}}\)를 서로 다른 supervision 경로로 fitting함.

추론 시에는 **\(\mathcal A_{\phi_{\mathrm S}}\) + \(\mathcal P_{\theta_{\mathrm S}}\)**만 사용하며 Teacher, GT, \(q\), \(e\), probe bank는 제거됨.

---

# 3. Stage 1 — Teacher / PAN Aligner Training

![Stage 1 — Teacher / PAN Aligner training](./fig3_stage1_teacher_aligner_framework.png)

**흐름 요약.** 먼저 native PAN–MS pair를 이용해 Teacher reconstruction을 학습함. PAN Aligner가 예측한 보정량으로 PAN을 MS frame에 맞춘 뒤, 보정된 PAN과 \(\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\)를 Teacher에 입력하여 HRMS를 복원함. Reconstruction supervision은 Teacher와 PAN Aligner 양쪽을 함께 갱신함. 이후 동일 PAN에 알려진 synthetic shift를 추가하고 같은 Aligner를 다시 통과시켜, native pair와 shifted pair의 예측 보정량 사이의 상대 관계를 consistency supervision으로 학습함. 이 consistency branch는 PAN Aligner만 직접 갱신하며 shifted PAN은 Teacher reconstruction 입력으로 사용하지 않음.

## 3.1 Native reconstruction

Teacher PAN Aligner를 \(\mathcal A_{\phi_{\mathrm T}}\), Teacher reconstruction network를 \(\mathcal P_{\theta_{\mathrm T}}\)라 정의함. Native pair에서 PAN Aligner는 sample-wise global translation을 예측함.

\[
\mathbf u^{\mathrm T}_{0}
=
\mathcal A_{\phi_{\mathrm T}}\!\left(\mathbf I_{\mathrm{pan}},\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\right).
\]

예측된 보정량을 differentiable sampling에 적용하여 PAN을 MS frame 방향으로 이동시킴.

\[
\tilde{\mathbf I}^{\mathrm T}_{\mathrm{pan}}
=
\operatorname{Warp}\!\left(\mathbf I_{\mathrm{pan}},\mathbf u^{\mathrm T}_{0}\right).
\]

Teacher prediction은 PAN-Crafter와 동일한 residual 형식으로

\[
\hat{\mathbf I}^{\mathrm{hr},\mathrm T}_{\mathrm{ms}}
=
\mathcal P_{\theta_{\mathrm T}}\!\left(\tilde{\mathbf I}^{\mathrm T}_{\mathrm{pan}},\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\right)
+
\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}
\]

으로 정의함. 여기서 \(\mathcal P_{\theta_{\mathrm T}}\)는 두 입력을 channel-wise로 concatenate한 \(\bigl[\tilde{\mathbf I}^{\mathrm T}_{\mathrm{pan}}\,\big|\,\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\bigr]\)를 받음. Native reconstruction loss는 전체 batch, spectral band, spatial location에 대한 평균 절대 오차로 둠.

\[
\boxed{
\mathcal L_{\mathrm{rec}}
=
\left\|
\hat{\mathbf I}^{\mathrm{hr},\mathrm T}_{\mathrm{ms}}
-
\mathbf I^{\mathrm{hr}}_{\mathrm{ms}}
\right\|_1
}
\]

\(\operatorname{Warp}\)는 학습 parameter를 갖지 않는 sampling 연산이지만, 입력 보정량 \(\mathbf u^{\mathrm T}_{0}\)에 대해 미분 가능함. 따라서 \(\mathcal L_{\mathrm{rec}}\)의 gradient는 \(\mathcal P_{\theta_{\mathrm T}}\)뿐 아니라 \(\mathcal A_{\phi_{\mathrm T}}\)까지 전달됨.

---

## 3.2 Relative-shift consistency

Native PAN–MS pair의 absolute displacement GT는 존재하지 않음. 대신 PAN에 알려진 synthetic translation \(\boldsymbol\epsilon\)을 적용하여 shifted PAN을 생성함.

\[
\mathbf I^{\epsilon}_{\mathrm{pan}}
=
\operatorname{Warp}\!\left(\mathbf I_{\mathrm{pan}},\boldsymbol\epsilon\right),
\qquad
\mathbf u^{\mathrm T}_{\epsilon}
=
\mathcal A_{\phi_{\mathrm T}}\!\left(\mathbf I^{\epsilon}_{\mathrm{pan}},\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\right).
\]

Aligner가 입력 shift에 대해 일관되게 반응한다면 shifted pair의 보정량은 native pair의 보정량에서 알려진 shift만큼 상쇄되는 관계를 가져야 함.

\[
\mathbf u^{\mathrm T}_{\epsilon}+\boldsymbol\epsilon
\approx
\mathbf u^{\mathrm T}_{0}.
\]

이에 따라 relative-shift consistency loss를

\[
\boxed{
\mathcal L_{\mathrm{con}}
=
\operatorname{mean}_{n}
\left\|
\mathbf u^{\mathrm T}_{\epsilon,n}
+\boldsymbol\epsilon_{n}
-\mathbf u^{\mathrm T}_{0,n}
\right\|_1
}
\]

로 정의함. 여기서 \(n\)은 sample index임. Consistency branch에서 native prediction \(\mathbf u^{\mathrm T}_{0}\)는 reference target으로만 사용하며, 해당 항을 통해 reference 자체를 갱신하지 않음.

Shifted PAN은 **consistency branch에만 사용**하며 Teacher reconstruction network에는 전달하지 않음. 따라서 이 항은 HRMS output consistency가 아니라, **알려진 geometric perturbation에 대한 PAN Aligner의 relative response consistency를 학습하는 항**임.

---

## 3.3 Stage 1 objective

Stage 1의 기본 objective는 reconstruction과 relative-shift consistency의 결합으로 표현함.

\[
\boxed{
\mathcal L_{\mathrm{S1}}
=
\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{con}}\mathcal L_{\mathrm{con}}
}
\]

Consistency supervision을 사용하지 않는 update에서는 \(\mathcal L_{\mathrm{rec}}\)만 적용함. 이때 두 항의 역할은 다음과 같이 구분됨.

- \(\mathcal L_{\mathrm{rec}}\): 최종 HRMS 복원에 유효한 native PAN correction을 학습
- \(\mathcal L_{\mathrm{con}}\): 알려진 PAN displacement 변화에 대한 Aligner의 반응 일관성을 학습

중요하게도 \(\mathcal L_{\mathrm{con}}\)는 native PAN–MS의 물리적 absolute shift GT를 제공하지 않으며, **알려진 추가 변위에 대한 상대적 반응만 감독**함.

---

# 4. Stage 2 — q/e-guided Student Fitting

![Stage 2 — q/e-guided Student training](./fig4_stage2_student_framework.png)

**흐름 요약.** Stage 1에서 학습된 \(\mathcal P_{\theta_{\mathrm T}}\)와 \(\mathcal A_{\phi_{\mathrm T}}\)를 고정 reference branch로 사용함. Student PAN Aligner \(\mathcal A_{\phi_{\mathrm S}}\)는 \(\mathcal A_{\phi_{\mathrm T}}\)의 가중치로 초기화한 뒤 trainable하게 두고, Student reconstruction network \(\mathcal P_{\theta_{\mathrm S}}\)는 경량 backbone으로 학습함. Frozen Teacher는 Student 학습에 사용할 reconstruction reference를 제공하며, Frozen Teacher Aligner는 정합 일관성 오차 \(q\)를 제공함. Reconstruction error \(e\)는 GT hard / Teacher soft supervision을 조절하고, \(q\)는 GT edge supervision과 Student Aligner adjustment의 신뢰도를 조절함.

## 4.1 Teacher / Student forward

Stage 2에서는 \(\mathcal P_{\theta_{\mathrm T}}\)와 \(\mathcal A_{\phi_{\mathrm T}}\)를 고정함. Teacher와 Student는 동일한 원본 PAN–MS pair를 사용하되, reconstruction network의 입력 구성은 각각 P0와 PLH로 다름.

Teacher branch는

\[
\mathbf u^{\mathrm T}_{0}
=
\mathcal A_{\phi_{\mathrm T}}\!\left(\mathbf I_{\mathrm{pan}},\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\right),
\qquad
\tilde{\mathbf I}^{\mathrm T}_{\mathrm{pan}}
=
\operatorname{Warp}\!\left(\mathbf I_{\mathrm{pan}},\mathbf u^{\mathrm T}_{0}\right),
\]

\[
\hat{\mathbf I}^{\mathrm{hr},\mathrm T}_{\mathrm{ms}}
=
\mathcal P_{\theta_{\mathrm T}}\!\left(\tilde{\mathbf I}^{\mathrm T}_{\mathrm{pan}},\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\right)
+
\mathbf I^{\mathrm{lr}}_{\mathrm{ms}},
\]

Student branch에서는 원본 PAN을 저역통과 필터링한 뒤 ×4 축소한 LPAN \(\mathbf I_{\mathrm{lpan}}\in\mathbb R^{H\times W\times1}\)을 사용함. LPAN은 기존 전처리/cache 규약으로 생성한 PAN-derived 입력이며, GT나 Teacher prediction으로 만들지 않음. Student Aligner의 보정량과 정합된 PAN은

\[
\mathbf u^{\mathrm S}
=
\mathcal A_{\phi_{\mathrm S}}\!\left(\mathbf I_{\mathrm{pan}},\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\right),
\qquad
\tilde{\mathbf I}^{\mathrm S}_{\mathrm{pan}}
=
\operatorname{Warp}\!\left(\mathbf I_{\mathrm{pan}},\mathbf u^{\mathrm S}\right),
\]

로 정의함. LPAN을 PAN 해상도로 확대한 뒤 **PAN과 동일한 \(\mathbf u^{\mathrm S}\) 및 sampling grid**로 보정하고, HPAN은 두 정합 영상의 차이로 구성함.

\[
\tilde{\mathbf I}^{\mathrm S}_{\mathrm{lpan}}
=
\operatorname{Warp}\!\left(\operatorname{Up}_4(\mathbf I_{\mathrm{lpan}}),\mathbf u^{\mathrm S}\right),
\qquad
\tilde{\mathbf I}^{\mathrm S}_{\mathrm{hpan}}
=
\tilde{\mathbf I}^{\mathrm S}_{\mathrm{pan}}
-
\tilde{\mathbf I}^{\mathrm S}_{\mathrm{lpan}}.
\]

Student prediction은 PAN·LPAN·HPAN·LRMS를 channel-wise로 concatenate하여

\[
\hat{\mathbf I}^{\mathrm{hr},\mathrm S}_{\mathrm{ms}}
=
\mathcal P_{\theta_{\mathrm S}}\!\left(
\bigl[\tilde{\mathbf I}^{\mathrm S}_{\mathrm{pan}}\,\big|\,
\tilde{\mathbf I}^{\mathrm S}_{\mathrm{lpan}}\,\big|\,
\tilde{\mathbf I}^{\mathrm S}_{\mathrm{hpan}}\,\big|\,
\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\bigr]
\right)
+
\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}
\]

로 정의함. Teacher branch의 출력은 학습 reference로만 사용하며 \(\theta_{\mathrm T},\phi_{\mathrm T}\)는 Stage 2에서 갱신하지 않음.

LPAN·HPAN은 Student reconstruction network의 추가 입력이며 Aligner 입력은 확장하지 않음. PAN·LPAN warp와 HPAN 차분에서 \(\mathbf u^{\mathrm S}\)를 detach하지 않으므로 세 PAN-derived 경로 모두 보정량에 대해 미분 가능함. 실제 Student Aligner parameter에 전달하는 gradient는 5.3–5.4절의 q-weighted hard reconstruction 규칙을 그대로 따름.

---

## 4.2 \(q\): Alignment-Consistency Error

\(q\)는 Stage 1의 relative-shift consistency 관계를 **frozen Teacher Aligner의 진단 신호**로 재사용한 값임.

Sample \(n\)에 대해 임의의 translation \(\boldsymbol\epsilon\)을 가한 뒤의 Teacher correction을

\[
\mathbf u^{\mathrm T}_{n}(\boldsymbol\epsilon)
=
\mathcal A_{\phi_{\mathrm T}}\!\left(
\operatorname{Warp}\!\left(\mathbf I_{\mathrm{pan},n},\boldsymbol\epsilon\right),
\ \mathbf I^{\mathrm{lr}}_{\mathrm{ms},n}
\right),
\qquad
\mathbf u^{\mathrm T}_{n,0}
=
\mathbf u^{\mathrm T}_{n}(\mathbf 0)
\]

라 하고, 고정된 probe bank \(\mathcal E=\{\boldsymbol\epsilon\}\), \(|\mathcal E|=N_q\)에 대해 sample-wise alignment-consistency error를

\[
\boxed{
q_{n}
=
\operatorname{mean}_{\boldsymbol\epsilon\in\mathcal E}
\left\|
\mathbf u^{\mathrm T}_{n}(\boldsymbol\epsilon)
+\boldsymbol\epsilon
-\mathbf u^{\mathrm T}_{n,0}
\right\|_1
}
\]

로 정의함.

- **Low \(q_{n}\)**: 알려진 PAN shift에 대해 Teacher Aligner의 correction response가 일관적임
- **High \(q_{n}\)**: perturbation에 대한 correction response가 불안정함

\(q_{n}\)은 absolute misalignment GT, edge probability, semantic class probability가 아님.

Train data의 \(q\) 분포로부터 reference scale을

\[
q_{\mathrm{ref}}
=
\operatorname{median}_{n\in\mathcal D}q_{n}
\]

로 정하고, alignment reliability weight를

\[
\boxed{
s_{n}
=
\frac{q_{\mathrm{ref}}}{q_{\mathrm{ref}}+q_{n}}
}
\]

로 정의함. 따라서 정합 일관성이 높을수록 \(s_{n}\)이 커지고, geometry-sensitive supervision을 더 신뢰함.

Teacher Aligner는 고정되어 있으므로 \(q_{n}\)과 \(s_{n}\)은 학습 전에 계산하여 cache할 수 있음.

---

## 4.3 \(e\): Reconstruction Error

위치 \(p\)에서 Teacher와 Student의 GT reconstruction error를 각각

\[
e_{\mathrm T}(n,p)
=
\operatorname{mean}_{c}
\left|
\hat{\mathbf I}^{\mathrm{hr},\mathrm T}_{\mathrm{ms},n,c}(p)-\mathbf I^{\mathrm{hr}}_{\mathrm{ms},n,c}(p)
\right|,
\]

\[
e_{\mathrm S}(n,p)
=
\operatorname{mean}_{c}
\left|
\hat{\mathbf I}^{\mathrm{hr},\mathrm S}_{\mathrm{ms},n,c}(p)-\mathbf I^{\mathrm{hr}}_{\mathrm{ms},n,c}(p)
\right|
\]

로 정의함. 여기서 \(c\in\{1,\dots,C_{\mathrm{ms}}\}\)는 spectral band index임.

Teacher–Student prediction discrepancy는

\[
\Delta_{\mathrm{TS}}(n,p)
=
\operatorname{mean}_{c}
\left|
\hat{\mathbf I}^{\mathrm{hr},\mathrm S}_{\mathrm{ms},n,c}(p)-\hat{\mathbf I}^{\mathrm{hr},\mathrm T}_{\mathrm{ms},n,c}(p)
\right|
\]

로 정의함.

Teacher reconstruction difficulty는

\[
\boxed{
d_{\mathrm T}(n,p)
=
\frac{e_{\mathrm T}(n,p)}{e_{\mathrm T}(n,p)+\tau_{\mathrm{rec}}}
}
\]

로 정규화함. \(\tau_{\mathrm{rec}}\)는 train data에서 고정하는 reconstruction-error reference scale임.

Teacher가 Student보다 실제로 더 정확한 위치만 soft supervision에 활용하기 위해 Teacher-advantage weight를

\[
\boxed{
a_{\mathrm T}(n,p)
=
\frac{
\max\!\left(e_{\mathrm S}(n,p)-e_{\mathrm T}(n,p),\,0\right)
}{
e_{\mathrm S}(n,p)+\delta_{\mathrm{rec}}
}
}
\]

로 정의함. \(\delta_{\mathrm{rec}}>0\)는 수치 안정성을 위한 작은 상수이며 학습되는 parameter가 아님.

따라서 다음과 같이 동작함.

- Teacher reconstruction error가 클수록 → GT hard fitting 강화
- Teacher reconstruction error가 작을수록 → Teacher soft target 활용 가능
- Student가 Teacher보다 이미 정확한 위치에서는 → \(a_{\mathrm T}=0\), soft imitation 비활성화

중요하게도 **GT hard supervision은 모든 위치에서 유지**함.

---

# 5. q/e-guided Supervision

## 5.1 Reconstruction supervision

### GT hard fitting

Sample \(n\)의 GT hard reconstruction term은

\[
\boxed{
\ell^{\mathrm{hard}}_{n}
=
\operatorname{mean}_{p}
\left[
\left(1+\lambda_{\mathrm{hard}}\,d_{\mathrm T}(n,p)\right)e_{\mathrm S}(n,p)
\right]
}
\]

로 정의함. \(\lambda_{\mathrm{hard}}\)는 Teacher reconstruction difficulty가 GT fitting 강도에 미치는 정도를 조절함. Base hard supervision은 항상 유지되므로 Teacher가 어려워하는 영역에서만 추가적으로 GT fitting이 강화됨.

### Teacher soft supervision

Teacher soft reconstruction term은

\[
\boxed{
\ell^{\mathrm{soft}}_{n}
=
\operatorname{mean}_{p}
\left[
\lambda_{\mathrm{soft}}
\left(1-d_{\mathrm T}(n,p)\right)
a_{\mathrm T}(n,p)
\Delta_{\mathrm{TS}}(n,p)
\right]
}
\]

로 정의함. \(\lambda_{\mathrm{soft}}\)는 Teacher soft supervision의 상대 강도를 조절함.

즉 soft supervision은 Teacher reconstruction error가 상대적으로 작고, Teacher가 Student보다 실제로 더 정확한 위치에서 주로 활성화됨. Teacher prediction을 모든 영역에서 동일하게 모방하는 conventional output distillation과 구분됨.

---

## 5.2 q-guided edge supervision

PAN-sharpening에서 edge는 spatial detail 복원에 중요하지만, 위치가 어긋난 상태에서 edge를 강하게 맞추면 잘못된 고주파 구조를 강화할 수 있음. 따라서 GT edge supervision의 강도를 \(q\)-derived reliability \(s_{n}\)로 조절함.

Student prediction과 GT 사이의 sample-wise signed edge error를

\[
\boxed{
\ell^{\mathrm{edge}}_{n}
=
\operatorname{mean}_{c,\,p\in\Omega_{\mathrm{edge}}}
\Bigl(
\bigl|D_x\hat{\mathbf I}^{\mathrm{hr},\mathrm S}_{\mathrm{ms},n,c}(p)-D_x\mathbf I^{\mathrm{hr}}_{\mathrm{ms},n,c}(p)\bigr|
+
\bigl|D_y\hat{\mathbf I}^{\mathrm{hr},\mathrm S}_{\mathrm{ms},n,c}(p)-D_y\mathbf I^{\mathrm{hr}}_{\mathrm{ms},n,c}(p)\bigr|
\Bigr)
}
\]

로 정의함. 여기서 \(D_x\), \(D_y\)는 두 축의 signed edge operator이며, \(\Omega_{\mathrm{edge}}\)는 edge loss를 계산하는 valid spatial support임.

q-guided edge term은

\[
\boxed{
\mathcal L_{\mathrm{edge}}
=
\operatorname{mean}_{n}
\left[s_{n}\,\ell^{\mathrm{edge}}_{n}\right]
}
\]

로 정의함. 최종 objective에서는 \(\lambda_{\mathrm{edge}}\)로 이 항의 전체 기여도를 조절함.

- Low \(q_{n}\) / high \(s_{n}\): geometry response가 안정적 → edge supervision을 상대적으로 강하게 적용
- High \(q_{n}\) / low \(s_{n}\): alignment response가 불안정 → edge supervision을 완화

따라서 **\(e\)는 reconstruction supervision의 구성 방식을, \(q\)는 geometry-sensitive supervision의 신뢰도를 결정**함.

---

## 5.3 Student Aligner adjustment

Student Aligner \(\mathcal A_{\phi_{\mathrm S}}\)는 Teacher soft loss나 GT edge loss의 gradient를 직접 받지 않음. \(\mathcal A_{\phi_{\mathrm S}}\)는 **q-weighted GT hard reconstruction**으로만 조정함.

\[
\boxed{
\mathcal L_{\mathcal A}
=
\operatorname{mean}_{n}
\left[
s_{n}\,\ell^{\mathrm{hard}}_{n}
\right]
}
\]

즉 Teacher Aligner의 shift response가 안정적인 sample일수록 Student Aligner를 더 적극적으로 fitting함. 반대로 high-\(q\) sample에서는 정합 cue 자체가 불안정하므로, reconstruction error가 크더라도 Student Aligner를 추가로 강하게 움직이지 않음.

---

## 5.4 Student objective와 gradient routing

Student reconstruction network의 objective는

\[
\boxed{
\mathcal L_{\mathcal P}
=
\operatorname{mean}_{n}
\left(
\ell^{\mathrm{hard}}_{n}
+
\ell^{\mathrm{soft}}_{n}
\right)
+
\lambda_{\mathrm{edge}}\mathcal L_{\mathrm{edge}}
}
\]

이며 Student Aligner의 objective는 앞서 정의한 \(\mathcal L_{\mathcal A}\)임.

두 objective를 단순 합산하여 전체 Student parameter에 동일하게 backward하지 않음. 같은 forward state에서 recipient를 분리하여

\[
\theta_{\mathrm S}
\leftarrow
\theta_{\mathrm S}-\eta_{\mathcal P}\nabla_{\theta_{\mathrm S}}\mathcal L_{\mathcal P},
\]

\[
\phi_{\mathrm S}
\leftarrow
\phi_{\mathrm S}-\eta_{\mathcal A}\nabla_{\phi_{\mathrm S}}\mathcal L_{\mathcal A}
\]

로 갱신함.

| Supervision | Student \(\mathcal P_{\theta_{\mathrm S}}\) | Student \(\mathcal A_{\phi_{\mathrm S}}\) | Teacher \(\mathcal P_{\theta_{\mathrm T}},\mathcal A_{\phi_{\mathrm T}}\) |
|---|:---:|:---:|:---:|
| GT hard reconstruction | ✓ | ✓, \(\mathcal L_{\mathcal A}\)를 통해 | Frozen |
| Teacher soft reconstruction | ✓ | – | Frozen |
| q-weighted GT edge | ✓ | – | Frozen |
| q computation | – | – | Frozen |

이와 같이 reconstruction knowledge와 geometry adaptation의 gradient 경로를 분리하여, Teacher output distillation이나 edge supervision이 Student Aligner까지 무분별하게 전달되는 것을 방지함.

---

# 6. q/e Quadrant Interpretation

q/e 네 영역은 실제 training에서 사용하는 binary class가 아니라, **alignment consistency와 reconstruction difficulty가 서로 다른 축임을 설명하기 위한 분석적 해석**임.

| 상태 | 해석 | Reconstruction supervision | Geometry-sensitive supervision | 관찰 예시 |
|---|---|---|---|---|
| **q low / e low** | 정합 반응 안정적, 복원도 양호 | GT hard 유지, 유효한 Teacher soft 활용 | edge/aligner를 상대적으로 신뢰 | 단순하거나 이미 잘 복원되는 영역 |
| **q low / e high** | 정합 반응 안정적, 복원은 어려움 | **GT hard 강화**, Teacher soft 약화 | **edge/aligner 적극 활용** | edge가 많은 도시·건물 영역 |
| **q high / e low** | 정합 반응 불안정, 복원은 양호 | GT hard 유지, Teacher가 더 좋으면 soft 활용 | edge/aligner 완화 | edge가 적은 단순 영역 |
| **q high / e high** | 정합과 복원 모두 어려움 | **GT hard 우선**, Teacher soft 제한 | 불확실한 geometry pressure를 보수적으로 완화 | 복잡한 산지·텍스처 영역 등 관찰 사례 |

핵심은 다음 두 질문을 분리하는 것임.

```text
복원이 어려운 이유가 HRMS mapping 자체의 난이도인가?
                           ↓ e

아니면 PAN–MS geometry에 대한 반응이 불안정하기 때문인가?
                           ↓ q
```

---

# 7. Training–Inference Separation

## Training-only

- Frozen Teacher reconstruction network \(\mathcal P_{\theta_{\mathrm T}}\)
- Frozen Teacher PAN Aligner \(\mathcal A_{\phi_{\mathrm T}}\)
- Teacher prediction \(\hat{\mathbf I}^{\mathrm{hr},\mathrm T}_{\mathrm{ms}}\)
- Alignment-consistency cue \(q_{n}\), reliability weight \(s_{n}\)
- HRMS GT \(\mathbf I^{\mathrm{hr}}_{\mathrm{ms}}\)
- Reconstruction-error cues \(e_{\mathrm T},e_{\mathrm S}\)
- Hard / soft / edge supervision
- PAN-shift probe bank \(\mathcal E\)

## Inference

```text
I_pan + I_ms
    ↓
I^lr_ms = Up_4(I_ms), LPAN = 원본 PAN의 저역통과·×4 축소
    ↓
Student Aligner  A_{phi_S}(I_pan, I^lr_ms) → u^S
    ↓
동일 grid로 Warp(I_pan, u^S), Warp(Up_4(LPAN), u^S)
    ↓
HPAN = Aligned PAN − Aligned LPAN
    ↓
[Aligned PAN | Aligned LPAN | HPAN | I^lr_ms]
    ↓
Student network  P_{theta_S}
    ↓
HRMS  Î^{hr,S}_ms
```

추론 시에도 Student의 LPAN·HPAN 입력 구성을 유지하며, 두 입력은 원본 PAN에서 구성함. Teacher, GT, \(q\), \(e\), probe bank \(\mathcal E\)는 필요하지 않음.

---

# 8. Notation

## 8.1 Symbol table

아래 표는 본문 수식에 등장하는 기호를 모두 포함함. PAN-Crafter의 표기 규약(영상 = \(\mathbf I^{\text{tag}}_{\text{modality}}\), 네트워크 = \(\mathcal X_{\text{param}}\), 연산 = roman operator, loss weight = \(\lambda_\bullet\))을 따르며, 동일 문자를 서로 다른 의미로 재사용하지 않음.

### 입력 · 영상 변수

| Symbol | Definition |
|---|---|
| \(\mathbf I_{\mathrm{pan}}\in\mathbb R^{4H\times4W\times1}\) | PAN input |
| \(\mathbf I_{\mathrm{ms}}\in\mathbb R^{H\times W\times C_{\mathrm{ms}}}\) | Native-resolution MS input |
| \(\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}\in\mathbb R^{4H\times4W\times C_{\mathrm{ms}}}\) | ×4 up-sampled MS (LRMS); MS-frame reconstruction base |
| \(\mathbf I_{\mathrm{lpan}}\in\mathbb R^{H\times W\times1}\) | 원본 PAN의 저역통과·×4 축소로 얻은 LPAN; Student PLH 입력의 원천 |
| \(\tilde{\mathbf I}^{\mathrm S}_{\mathrm{lpan}}\in\mathbb R^{4H\times4W\times1}\) | ×4 확대 후 Student correction으로 PAN과 함께 정합한 LPAN |
| \(\tilde{\mathbf I}^{\mathrm S}_{\mathrm{hpan}}\in\mathbb R^{4H\times4W\times1}\) | 정합된 PAN − 정합된 LPAN으로 구성한 Student HPAN 입력 |
| \(\mathbf I^{\mathrm{hr}}_{\mathrm{ms}}\) | Training HRMS ground truth |
| \(\mathbf I^{\epsilon}_{\mathrm{pan}}\) | Synthetic translation이 적용된 PAN |
| \(\tilde{\mathbf I}^{\mathrm T}_{\mathrm{pan}},\ \tilde{\mathbf I}^{\mathrm S}_{\mathrm{pan}}\) | Teacher / Student 경로에서 보정된 PAN |
| \(\hat{\mathbf I}^{\mathrm{hr},\mathrm T}_{\mathrm{ms}},\ \hat{\mathbf I}^{\mathrm{hr},\mathrm S}_{\mathrm{ms}}\) | Teacher / Student HRMS prediction |
| \(\mathcal D\) | Paired training dataset \(\{(\mathbf I_{\mathrm{pan}},\mathbf I_{\mathrm{ms}},\mathbf I^{\mathrm{hr}}_{\mathrm{ms}})\}\) |
| \(H,W,C_{\mathrm{ms}}\) | MS 공간 크기와 spectral band 수 |

### 네트워크 · 연산자

| Symbol | Definition |
|---|---|
| \(\mathcal P_{\theta_{\mathrm T}},\ \mathcal P_{\theta_{\mathrm S}}\) | Teacher / Student reconstruction network |
| \(\mathcal A_{\phi_{\mathrm T}},\ \mathcal A_{\phi_{\mathrm S}}\) | Teacher / Student PAN Aligner |
| \(\theta_{\mathrm S},\ \phi_{\mathrm S}\) | Student reconstruction network / Aligner parameter set |
| \(\operatorname{Up}_4(\cdot)\) | MS 또는 LPAN을 PAN 해상도로 확대하는 고정 ×4 upsampling |
| \(\operatorname{Warp}(\cdot,\cdot)\) | Predicted 또는 synthetic shift를 적용하는 differentiable sampling; Student PAN·확대 LPAN은 동일 grid 사용 |
| \([\,\cdot\,|\,\cdot\,]\) | Channel-wise concatenation |
| \(\|\cdot\|_1\) | Batch·band·spatial 평균 절대 오차 (PAN-Crafter와 동일 의미) |

### Alignment 관련

| Symbol | Definition |
|---|---|
| \(\mathbf u^{\mathrm T}_{0},\ \mathbf u^{\mathrm T}_{\epsilon}\) | Stage 1 native / shifted pair의 predicted PAN correction |
| \(\mathbf u^{\mathrm S}\) | Stage 2 Student Aligner의 predicted PAN correction |
| \(\mathbf u^{\mathrm T}_{n}(\boldsymbol\epsilon)\) | Sample \(n\)에 probe \(\boldsymbol\epsilon\)을 적용했을 때의 frozen Teacher Aligner correction |
| \(\mathbf u^{\mathrm T}_{n,0}=\mathbf u^{\mathrm T}_{n}(\mathbf 0)\) | Sample \(n\)의 frozen Teacher Aligner native correction |
| \(\boldsymbol\epsilon\) | 알려진 synthetic PAN translation |
| \(\mathcal E,\ N_q=|\mathcal E|\) | Stage 2 \(q\) 계산에 사용하는 고정 PAN-shift probe bank와 그 크기 |
| \(q_{n}\) | Sample \(n\)의 alignment-consistency error |
| \(q_{\mathrm{ref}}\) | Train \(q\) distribution에서 고정한 reference scale |
| \(s_{n}\) | \(q\)-derived alignment reliability weight |

### Reconstruction 관련

| Symbol | Definition |
|---|---|
| \(e_{\mathrm T}(n,p),\ e_{\mathrm S}(n,p)\) | 위치 \(p\)의 Teacher / Student GT reconstruction error |
| \(\Delta_{\mathrm{TS}}(n,p)\) | 위치 \(p\)의 Teacher–Student prediction discrepancy |
| \(d_{\mathrm T}(n,p)\) | Normalized Teacher reconstruction difficulty |
| \(a_{\mathrm T}(n,p)\) | Teacher가 Student보다 정확한 정도를 나타내는 soft-supervision weight |
| \(\tau_{\mathrm{rec}}\) | Reconstruction-error reference scale |
| \(\delta_{\mathrm{rec}}\) | \(a_{\mathrm T}\) 계산의 numerical stabilizer |
| \(D_x,D_y\) | Horizontal / vertical signed edge operators |
| \(\Omega_{\mathrm{edge}}\) | Edge loss를 계산하는 valid spatial support |

### Loss · 최적화

| Symbol | Definition |
|---|---|
| \(\mathcal L_{\mathrm{rec}}\) | Stage 1 native HRMS reconstruction loss |
| \(\mathcal L_{\mathrm{con}}\) | Stage 1 relative-shift consistency loss |
| \(\mathcal L_{\mathrm{S1}}\) | Stage 1 전체 objective |
| \(\ell^{\mathrm{hard}}_{n},\ \ell^{\mathrm{soft}}_{n},\ \ell^{\mathrm{edge}}_{n}\) | Sample \(n\)의 GT hard / Teacher soft / GT edge term |
| \(\mathcal L_{\mathrm{edge}}\) | \(q\)-weighted GT edge loss |
| \(\mathcal L_{\mathcal P},\ \mathcal L_{\mathcal A}\) | Student reconstruction network / Student Aligner objective |
| \(\lambda_{\mathrm{con}},\lambda_{\mathrm{hard}},\lambda_{\mathrm{soft}},\lambda_{\mathrm{edge}}\) | Consistency / hard / soft / edge supervision weight |
| \(\eta_{\mathcal P},\ \eta_{\mathcal A}\) | Student reconstruction network / Aligner learning rate |

### Index

| Symbol | Definition |
|---|---|
| \(n\) | Sample index |
| \(c\) | MS spectral-band index, \(c\in\{1,\dots,C_{\mathrm{ms}}\}\) |
| \(p\) | Spatial position index |

> **Index 규약.** PAN-Crafter는 \((i,j)\)를 local attention의 query position, \((m,n)\)을 그 receptive field offset, \(k\)를 attention kernel size로 사용함. 본 문서의 supervision 수식은 그 범위와 겹치지 않도록 sample index \(n\), spatial position \(p\), band index \(c\)만 사용하고 \((i,j,m,k)\)를 쓰지 않음. 동일 논문에서 attention 수식을 함께 제시할 경우 \(n\)의 중복이 신경 쓰이면 sample index를 \(b\)(batch)로 바꾸는 것만으로 충돌 없이 분리 가능함.

## 8.2 v3 → v4 대응표

| v3 (기존) | v4 (PAN-Crafter 정렬) | 비고 |
|---|---|---|
| \(P\) | \(\mathbf I_{\mathrm{pan}}\) | |
| \(S\) | \(\mathbf I_{\mathrm{ms}}\) | v3의 "LRMS 입력" → PAN-Crafter에서는 native MS |
| \(\mathcal U\) | \(\operatorname{Up}_4\) | roman operator 표기 |
| \(M=\mathcal U(S)\) | \(\mathbf I^{\mathrm{lr}}_{\mathrm{ms}}=\operatorname{Up}_4(\mathbf I_{\mathrm{ms}})\) | PAN-Crafter에서 LRMS는 이미 ×4 up-sampled |
| \(Y\) | \(\mathbf I^{\mathrm{hr}}_{\mathrm{ms}}\) | |
| \(Z_T,\ Z_S\) | \(\hat{\mathbf I}^{\mathrm{hr},\mathrm T}_{\mathrm{ms}},\ \hat{\mathbf I}^{\mathrm{hr},\mathrm S}_{\mathrm{ms}}\) | 예측은 hat, tag는 superscript |
| \(F_T,\ F_S\) | \(\mathcal P_{\theta_{\mathrm T}},\ \mathcal P_{\theta_{\mathrm S}}\) | PAN-Crafter의 \(\mathcal P_\theta\) 규약 |
| \(A_T,\ A_S\) | \(\mathcal A_{\phi_{\mathrm T}},\ \mathcal A_{\phi_{\mathrm S}}\) | |
| \(\mathcal W\) | \(\operatorname{Warp}\) | 폭 \(W\)와의 혼동 제거 |
| \(\operatorname{Concat}(\cdot,\cdot)\) | \([\,\cdot\,|\,\cdot\,]\) | |
| \(\widetilde P_0,\ \widetilde P_S\) | \(\tilde{\mathbf I}^{\mathrm T}_{\mathrm{pan}},\ \tilde{\mathbf I}^{\mathrm S}_{\mathrm{pan}}\) | |
| \(P_\epsilon\) | \(\mathbf I^{\epsilon}_{\mathrm{pan}}\) | |
| \(\mathbf c_0,\ \mathbf c_\epsilon\) | \(\mathbf u^{\mathrm T}_{0},\ \mathbf u^{\mathrm T}_{\epsilon}\) | band index \(c\)와의 충돌 제거 |
| \(\mathbf c^T_{i,0},\ \mathbf c^T_{i,j}\) | \(\mathbf u^{\mathrm T}_{n,0},\ \mathbf u^{\mathrm T}_{n}(\boldsymbol\epsilon)\) | probe index \(j\) 제거, probe bank 평균으로 표기 |
| \(\alpha\) | \(\lambda_{\mathrm{hard}}\) | PAN-Crafter의 \(\boldsymbol\alpha\)(Select 학습 파라미터)와 충돌 |
| \(\beta\) | \(\lambda_{\mathrm{soft}}\) | PAN-Crafter의 \(\boldsymbol\beta\)(Modulate 파라미터)와 충돌 |
| \(\lambda_E\) | \(\lambda_{\mathrm{edge}}\) | loss weight 표기 통일 |
| \(E_i\) | \(\ell^{\mathrm{edge}}_{n}\) | hard/soft/edge를 동일한 per-sample \(\ell\) 계열로 통일 |
| \(\tau_R,\ \delta_R\) | \(\tau_{\mathrm{rec}},\ \delta_{\mathrm{rec}}\) | roman subscript 통일 |
| \(\Omega_E\) | \(\Omega_{\mathrm{edge}}\) | |
| \(\mathcal L_U,\ \mathcal L_A\) | \(\mathcal L_{\mathcal P},\ \mathcal L_{\mathcal A}\) | 네트워크 기호와 일치 |
| \(\eta_U,\ \eta_A\) | \(\eta_{\mathcal P},\ \eta_{\mathcal A}\) | |
| \(\theta_S,\ \phi_S\) | \(\theta_{\mathrm S},\ \phi_{\mathrm S}\) | subscript upright |
| \(e_T,e_S,d_T,a_T,\Delta_{TS}\) | \(e_{\mathrm T},e_{\mathrm S},d_{\mathrm T},a_{\mathrm T},\Delta_{\mathrm{TS}}\) | subscript upright (PAN-Crafter의 \(\mathrm{ms},\mathrm{pan}\) 방식) |
| \(i\) (sample), \(j\) (probe) | \(n\) (sample), probe index 미사용 | \((i,j)\)는 attention query position 용으로 보류 |
| \(\mathcal D_{\mathrm{train}}\) | \(\mathcal D\) | PAN-Crafter와 동일 |
| \(q_i,\ s_i\) | \(q_{n},\ s_{n}\) | |

---

# 9. Claim Scope

1. **\(q\)는 absolute misalignment GT가 아님.**  
   Controlled PAN shift에 대한 frozen Teacher Aligner \(\mathcal A_{\phi_{\mathrm T}}\)의 response consistency를 측정하는 error임.

2. **\(e\)는 predictive uncertainty가 아님.**  
   Training GT \(\mathbf I^{\mathrm{hr}}_{\mathrm{ms}}\)를 이용해 계산하는 reconstruction error임.

3. **q/e quadrant는 해석용임.**  
   도시·산지·단순 영역 등의 예시는 qualitative observation이며 semantic class label로 사용하지 않음.

4. **Stage 1 consistency는 physical alignment correctness를 직접 보장하지 않음.**  
   Known perturbation에 대한 relative response를 학습하며, native alignment는 reconstruction objective에 의해 task-oriented하게 anchor됨.

5. **최종 method는 continuous weighting임.**  
   q/e 네 영역은 설명을 위한 구분이며 실제 training에서 binary q/e switch를 사용하지 않음.

6. **High-q/high-e 영역에서는 conservative geometry routing을 사용함.**  
   GT hard reconstruction은 강화하되, 신뢰도가 낮은 geometry-sensitive supervision을 추가로 강화하지 않음.

---

# 10. Paper-level Method Narrative

실제 PAN–MS pair에는 residual spatial displacement가 존재하며, 동시에 영상 영역별 HRMS reconstruction difficulty 또한 크게 달라짐. 그러나 두 난이도는 동일하지 않음. 정합 반응은 안정적이지만 복원이 어려운 영역이 존재하는 반면, 복원 오차는 작지만 PAN–MS geometry에 대한 반응이 불안정한 영역도 존재함. 본 연구는 이러한 차이를 **alignment-consistency error \(q\)**와 **reconstruction error \(e\)**로 분리함.

먼저 Stage 1에서는 PAN Aligner \(\mathcal A_{\phi_{\mathrm T}}\)와 Teacher reconstruction network \(\mathcal P_{\theta_{\mathrm T}}\)를 native HRMS reconstruction objective로 공동 학습함. 이후 알려진 synthetic PAN translation \(\boldsymbol\epsilon\)을 이용한 relative-shift consistency를 추가하여, Aligner가 입력 displacement 변화에 일관되게 반응하도록 학습함. 이때 shifted PAN \(\mathbf I^{\epsilon}_{\mathrm{pan}}\)은 Aligner supervision에만 사용하고 Teacher reconstruction branch에는 입력하지 않음.

Stage 2에서는 Stage 1에서 얻은 \(\mathcal P_{\theta_{\mathrm T}}\)와 \(\mathcal A_{\phi_{\mathrm T}}\)를 고정하고 경량 Student를 학습함. Reconstruction error \(e\)는 GT hard fitting과 Teacher soft distillation의 상대적 강도를 조절하며, alignment-consistency error \(q\)는 GT edge supervision과 Student Aligner adjustment의 신뢰도를 제어함. \(\mathcal P_{\theta_{\mathrm S}}\)와 \(\mathcal A_{\phi_{\mathrm S}}\)는 동일한 forward를 공유하지만 서로 다른 objective를 통해 갱신되어, Teacher output distillation과 geometry adaptation이 불필요하게 뒤섞이지 않도록 함.

최종적으로 본 방법은 **복원 오류와 정합 오류를 분리한 supervision routing**을 통해, Teacher의 오류를 무조건 모방하거나 불안정한 PAN geometry를 강제하지 않으면서 Student를 입력 데이터에 보다 정밀하게 fitting하는 것을 목표로 함.

---

## One-line Method Summary

> **재구성 오차 \(e\)로 GT와 Teacher 사이의 reconstruction supervision을 조절하고, 정합 일관성 오차 \(q\)로 geometry-sensitive supervision과 Student Aligner adaptation의 신뢰도를 조절함.**
