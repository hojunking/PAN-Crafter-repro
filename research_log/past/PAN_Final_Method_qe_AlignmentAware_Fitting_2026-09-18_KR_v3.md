# 최종 방법론 정리 — q/e 기반 Alignment-Aware Student Fitting

**Target:** CVPR 2027 Pansharpening  
**Method status:** 논문 작성용 현재 formulation  
**핵심 개념:** PAN–MS 정합의 불안정성과 HRMS 복원의 어려움을 각각 **alignment-consistency error \(q\)**와 **reconstruction error \(e\)**로 분리하고, 두 신호를 서로 다른 supervision 경로에 사용함.

---

# 1. Problem Definition

## 1.1 입력 구조와 기준 좌표계

Pansharpening은 고해상도 PAN 영상의 **spatial detail**과 저해상도 MS 영상의 **spectral information**을 결합하여 HRMS 영상을 복원하는 문제임. LRMS 입력 \(S\)를 PAN 해상도로 확대하여

\[
M=\mathcal U(S)
\]

으로 두며, HRMS 정답 \(Y\), residual base \(M\), 최종 출력은 모두 **MS reference frame(M-frame)**에 고정함. 따라서 본 연구에서는 MS/GT/output을 PAN 좌표계로 이동시키지 않고, PAN의 spatial information을 M-frame에 맞추는 방향으로 문제를 구성함.

---

## 1.2 PAN–MS residual misalignment

PAN과 MS는 서로 다른 센서 경로와 해상도에서 획득되므로, 전처리 이후에도 작은 spatial displacement가 잔존할 수 있음. 아래 예시는 WV3 full-resolution 영상에서 PAN–MS 사이의 local displacement가 존재하며, 단순 translation만으로 두 영상의 구조적 상관이 크게 증가하는 경우를 보여줌.

![Observed PAN–MS displacement example](./fig1_pan_ms_misalignment_example.png)

이러한 위치 오차를 고려하지 않고 PAN의 고주파 구조를 직접 주입하면 GT와 다른 위치의 edge가 강화되어 double-edge 또는 blur가 발생할 수 있음. 반대로 PAN detail을 일괄적으로 약화하면 spatial fidelity가 저하됨. 따라서 **PAN-derived geometry를 어느 정도 신뢰할 수 있는지 판단하는 정합 관련 신호가 필요함.**

---

## 1.3 정합 오차와 재구성 오차의 분리

정합 상태와 복원 난이도는 동일하지 않음. 실제 patch 분석에서는 정합 반응은 안정적이지만 복원이 어려운 영역, 또는 복원은 잘 되지만 정합 반응은 불안정한 영역이 모두 관찰됨.

![q/e patch quadrants](./fig2_qe_patch_quadrants.png)

본 연구에서는 이를 두 종류의 오류 신호로 분리함. 아래 quadrant 그림에서 \(e\)는 patch 수준에서 집계한 reconstruction error를 의미하며, 실제 학습에서는 Teacher와 Student의 위치별 오류 \(e_T\), \(e_S\)를 사용함.

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
Native PAN + MS
      ↓
PAN Aligner → Aligned PAN → Teacher → HRMS
      │                         ↓
      └──────── Reconstruction supervision

Shifted PAN
      ↓
Same PAN Aligner
      ↓
Relative-shift consistency

                ↓ Teacher / Teacher Aligner freeze

Stage 2. Student Training
Frozen Teacher        → Teacher prediction
Frozen Teacher Aligner→ alignment-consistency cue q

PAN + MS → Student Aligner → Aligned PAN → Student → Student prediction
                                               │
                         e-guided reconstruction supervision
                         q-guided edge supervision
                         q-guided Aligner adjustment
```

- **Stage 1:** PAN Aligner가 HRMS 복원에 유효한 native correction과 알려진 PAN shift에 대한 일관된 반응을 학습함.
- **Stage 2:** Stage 1에서 얻은 Teacher와 Teacher Aligner를 고정하고, \(e\)와 \(q\)를 이용해 Student U-Net과 Student Aligner를 서로 다른 supervision 경로로 fitting함.

추론 시에는 **Student PAN Aligner + Student U-Net**만 사용하며 Teacher, GT, \(q\), \(e\), probe bank는 제거됨.

---

# 3. Stage 1 — Teacher / PAN Aligner Training

![Stage 1 — Teacher / PAN Aligner training](./fig3_stage1_teacher_aligner_framework.png)

**흐름 요약.** 먼저 native PAN–MS pair를 이용해 Teacher reconstruction을 학습함. PAN Aligner가 예측한 보정량으로 PAN을 M-frame에 맞춘 뒤, 보정된 PAN과 upsampled MS를 Teacher에 입력하여 HRMS를 복원함. Reconstruction supervision은 Teacher와 PAN Aligner 양쪽을 함께 갱신함. 이후 동일 PAN에 알려진 synthetic shift를 추가하고 같은 Aligner를 다시 통과시켜, native pair와 shifted pair의 예측 보정량 사이의 상대 관계를 consistency supervision으로 학습함. 이 consistency branch는 PAN Aligner만 직접 갱신하며 shifted PAN은 Teacher reconstruction 입력으로 사용하지 않음.

## 3.1 Native reconstruction

Teacher PAN Aligner를 \(A_T\), Teacher reconstruction network를 \(F_T\)라 정의함. Native pair에서 PAN Aligner는 sample-wise global translation을 예측함.

\[
\mathbf c_0=A_T(P,M).
\]

예측된 보정량을 differentiable sampling에 적용하여 PAN을 M-frame 방향으로 이동시킴.

\[
\widetilde P_0=\mathcal W(P,\mathbf c_0).
\]

Teacher prediction은

\[
Z_T
=
M+F_T\!\left(\operatorname{Concat}(\widetilde P_0,M)\right)
\]

으로 정의함. Native reconstruction loss는 전체 batch, spectral band, spatial location에 대한 평균 절대 오차로 둠.

\[
\boxed{
\mathcal L_{\mathrm{rec}}
=
\operatorname{mean}\left(|Z_T-Y|\right)
}
\]

\(\mathcal W\)는 학습 parameter를 갖지 않는 sampling 연산이지만, 입력 보정량 \(\mathbf c_0\)에 대해 미분 가능함. 따라서 \(\mathcal L_{\mathrm{rec}}\)의 gradient는 Teacher reconstruction network뿐 아니라 PAN Aligner까지 전달됨.

---

## 3.2 Relative-shift consistency

Native PAN–MS pair의 absolute displacement GT는 존재하지 않음. 대신 PAN에 알려진 synthetic translation \(\boldsymbol\epsilon\)을 적용하여 shifted PAN을 생성함.

\[
P_{\epsilon}=\mathcal W(P,\boldsymbol\epsilon),
\qquad
\mathbf c_{\epsilon}=A_T(P_{\epsilon},M).
\]

Aligner가 입력 shift에 대해 일관되게 반응한다면 shifted pair의 보정량은 native pair의 보정량에서 알려진 shift만큼 상쇄되는 관계를 가져야 함.

\[
\mathbf c_{\epsilon}+\boldsymbol\epsilon
\approx
\mathbf c_0.
\]

이에 따라 relative-shift consistency loss를

\[
\boxed{
\mathcal L_{\mathrm{con}}
=
\operatorname{mean}_{i}
\left\|
\mathbf c_{\epsilon,i}
+\boldsymbol\epsilon_i
-\mathbf c_{0,i}
\right\|_1
}
\]

로 정의함. Consistency branch에서 native prediction \(\mathbf c_0\)는 reference target으로만 사용하며, 해당 항을 통해 reference 자체를 갱신하지 않음.

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

**흐름 요약.** Stage 1에서 학습된 Teacher U-Net과 Teacher PAN Aligner를 고정 reference branch로 사용함. Student PAN Aligner는 Teacher Aligner의 가중치로 초기화한 뒤 trainable하게 두고, Student U-Net은 경량 backbone으로 학습함. Frozen Teacher는 Student 학습에 사용할 reconstruction reference를 제공하며, Frozen Teacher Aligner는 정합 일관성 오차 \(q\)를 제공함. Reconstruction error \(e\)는 GT hard / Teacher soft supervision을 조절하고, \(q\)는 GT edge supervision과 Student Aligner adjustment의 신뢰도를 조절함.

## 4.1 Teacher / Student forward

Stage 2에서는 Teacher와 Teacher Aligner를 고정함. Student PAN Aligner를 \(A_S\), Student reconstruction network를 \(F_S\)라 정의함.

Teacher와 Student는 동일한 PAN–MS pair를 입력으로 사용함.

\[
Z_T=\operatorname{Teacher}(P,M),
\]

\[
\mathbf c_S=A_S(P,M),
\qquad
\widetilde P_S=\mathcal W(P,\mathbf c_S),
\]

\[
Z_S
=
M+F_S\!\left(\operatorname{Concat}(\widetilde P_S,M)\right).
\]

Teacher branch의 출력은 학습 reference로만 사용하며 Teacher parameter는 Stage 2에서 갱신하지 않음.

---

## 4.2 \(q\): Alignment-Consistency Error

\(q\)는 Stage 1의 relative-shift consistency 관계를 **frozen Teacher Aligner의 진단 신호**로 재사용한 값임.

Sample \(i\)의 native Teacher correction을

\[
\mathbf c^{T}_{i,0}=A_T(P_i,M_i)
\]

라 하고, 고정된 probe bank \(\mathcal E=\{\boldsymbol\epsilon_j\}_{j=1}^{N_q}\)의 각 translation을 적용한 뒤

\[
\mathbf c^{T}_{i,j}
=
A_T\!\left(\mathcal W(P_i,\boldsymbol\epsilon_j),M_i\right)
\]

를 계산함. Sample-wise alignment-consistency error는

\[
\boxed{
q_i
=
\operatorname{mean}_{j}
\left\|
\mathbf c^{T}_{i,j}
+\boldsymbol\epsilon_j
-\mathbf c^{T}_{i,0}
\right\|_1
}
\]

로 정의함.

- **Low \(q_i\)**: 알려진 PAN shift에 대해 Teacher Aligner의 correction response가 일관적임
- **High \(q_i\)**: perturbation에 대한 correction response가 불안정함

\(q_i\)는 absolute misalignment GT, edge probability, semantic class probability가 아님.

Train data의 \(q\) 분포로부터 reference scale을

\[
q_{\mathrm{ref}}
=
\operatorname{median}_{i\in\mathcal D_{\mathrm{train}}}q_i
\]

로 정하고, alignment reliability weight를

\[
\boxed{
s_i
=
\frac{q_{\mathrm{ref}}}{q_{\mathrm{ref}}+q_i}
}
\]

로 정의함. 따라서 정합 일관성이 높을수록 \(s_i\)가 커지고, geometry-sensitive supervision을 더 신뢰함.

Teacher Aligner는 고정되어 있으므로 \(q_i\)와 \(s_i\)는 학습 전에 계산하여 cache할 수 있음.

---

## 4.3 \(e\): Reconstruction Error

위치 \(p\)에서 Teacher와 Student의 GT reconstruction error를 각각

\[
e_T(i,p)
=
\operatorname{mean}_{c}
\left|Z_{T,i,c}(p)-Y_{i,c}(p)\right|,
\]

\[
e_S(i,p)
=
\operatorname{mean}_{c}
\left|Z_{S,i,c}(p)-Y_{i,c}(p)\right|
\]

로 정의함.

Teacher–Student prediction discrepancy는

\[
\Delta_{TS}(i,p)
=
\operatorname{mean}_{c}
\left|Z_{S,i,c}(p)-Z_{T,i,c}(p)\right|
\]

로 정의함.

Teacher reconstruction difficulty는

\[
\boxed{
d_T(i,p)
=
\frac{e_T(i,p)}{e_T(i,p)+\tau_R}
}
\]

로 정규화함. \(\tau_R\)는 train data에서 고정하는 reconstruction-error reference scale임.

Teacher가 Student보다 실제로 더 정확한 위치만 soft supervision에 활용하기 위해 Teacher-advantage weight를

\[
\boxed{
a_T(i,p)
=
\frac{
\max\!\left(e_S(i,p)-e_T(i,p),\,0\right)
}{
e_S(i,p)+\delta_R
}
}
\]

로 정의함. \(\delta_R>0\)는 수치 안정성을 위한 작은 상수이며 학습되는 parameter가 아님.

따라서 다음과 같이 동작함.

- Teacher reconstruction error가 클수록 → GT hard fitting 강화
- Teacher reconstruction error가 작을수록 → Teacher soft target 활용 가능
- Student가 Teacher보다 이미 정확한 위치에서는 → \(a_T=0\), soft imitation 비활성화

중요하게도 **GT hard supervision은 모든 위치에서 유지**함.

---

# 5. q/e-guided Supervision

## 5.1 Reconstruction supervision

### GT hard fitting

Sample \(i\)의 GT hard reconstruction term은

\[
\boxed{
\ell^{\mathrm{hard}}_i
=
\operatorname{mean}_{p}
\left[
\left(1+\alpha d_T(i,p)\right)e_S(i,p)
\right]
}
\]

로 정의함. \(\alpha\)는 Teacher reconstruction difficulty가 GT fitting 강도에 미치는 정도를 조절함. Base hard supervision은 항상 유지되므로 Teacher가 어려워하는 영역에서만 추가적으로 GT fitting이 강화됨.

### Teacher soft supervision

Teacher soft reconstruction term은

\[
\boxed{
\ell^{\mathrm{soft}}_i
=
\operatorname{mean}_{p}
\left[
\beta
\left(1-d_T(i,p)\right)
 a_T(i,p)
\Delta_{TS}(i,p)
\right]
}
\]

로 정의함. \(\beta\)는 Teacher soft supervision의 상대 강도를 조절함.

즉 soft supervision은 Teacher reconstruction error가 상대적으로 작고, Teacher가 Student보다 실제로 더 정확한 위치에서 주로 활성화됨. Teacher prediction을 모든 영역에서 동일하게 모방하는 conventional output distillation과 구분됨.

---

## 5.2 q-guided edge supervision

PAN-sharpening에서 edge는 spatial detail 복원에 중요하지만, 위치가 어긋난 상태에서 edge를 강하게 맞추면 잘못된 고주파 구조를 강화할 수 있음. 따라서 GT edge supervision의 강도를 \(q\)-derived reliability \(s_i\)로 조절함.

Student prediction과 GT 사이의 sample-wise signed edge error를

\[
\boxed{
E_i
=
\operatorname{mean}_{c,p\in\Omega_E}
\Bigl(
|D_x Z_{S,i,c}(p)-D_xY_{i,c}(p)|
+
|D_y Z_{S,i,c}(p)-D_yY_{i,c}(p)|
\Bigr)
}
\]

로 정의함. 여기서 \(D_x\), \(D_y\)는 두 축의 signed edge operator이며, \(\Omega_E\)는 edge loss를 계산하는 valid spatial support임.

q-guided edge term은

\[
\boxed{
\mathcal L_{\mathrm{edge}}
=
\operatorname{mean}_{i}
\left[s_iE_i\right]
}
\]

로 정의함. 최종 U-Net objective에서는 \(\lambda_E\)로 이 항의 전체 기여도를 조절함.

- Low \(q_i\) / high \(s_i\): geometry response가 안정적 → edge supervision을 상대적으로 강하게 적용
- High \(q_i\) / low \(s_i\): alignment response가 불안정 → edge supervision을 완화

따라서 **\(e\)는 reconstruction supervision의 구성 방식을, \(q\)는 geometry-sensitive supervision의 신뢰도를 결정**함.

---

## 5.3 Student Aligner adjustment

Student Aligner는 Teacher soft loss나 GT edge loss의 gradient를 직접 받지 않음. Student Aligner는 **q-weighted GT hard reconstruction**으로만 조정함.

\[
\boxed{
\mathcal L_A
=
\operatorname{mean}_{i}
\left[
 s_i\ell^{\mathrm{hard}}_i
\right]
}
\]

즉 Teacher Aligner의 shift response가 안정적인 sample일수록 Student Aligner를 더 적극적으로 fitting함. 반대로 high-\(q\) sample에서는 정합 cue 자체가 불안정하므로, reconstruction error가 크더라도 Student Aligner를 추가로 강하게 움직이지 않음.

---

## 5.4 Student U-Net objective와 gradient routing

Student U-Net의 objective는

\[
\boxed{
\mathcal L_U
=
\operatorname{mean}_{i}
\left(
\ell^{\mathrm{hard}}_i
+
\ell^{\mathrm{soft}}_i
\right)
+
\lambda_E\mathcal L_{\mathrm{edge}}
}
\]

이며 Student Aligner의 objective는 앞서 정의한 \(\mathcal L_A\)임.

두 objective를 단순 합산하여 전체 Student parameter에 동일하게 backward하지 않음. 같은 forward state에서 recipient를 분리하여

\[
\theta_S
\leftarrow
\theta_S-\eta_U\nabla_{\theta_S}\mathcal L_U,
\]

\[
\phi_S
\leftarrow
\phi_S-\eta_A\nabla_{\phi_S}\mathcal L_A
\]

로 갱신함.

| Supervision | Student U-Net | Student Aligner | Teacher |
|---|:---:|:---:|:---:|
| GT hard reconstruction | ✓ | ✓, \(\mathcal L_A\)를 통해 | Frozen |
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

- Frozen Teacher U-Net
- Frozen Teacher PAN Aligner
- Teacher prediction \(Z_T\)
- Alignment-consistency cue \(q_i\)
- HRMS GT \(Y\)
- Reconstruction-error cues \(e_T,e_S\)
- Hard / soft / edge supervision

## Inference

```text
PAN + LRMS
    ↓
MS Upsampling
    ↓
Student PAN Aligner
    ↓
Aligned PAN + MS
    ↓
Student U-Net
    ↓
HRMS
```

추론 시 Teacher, GT, \(q\), \(e\), probe bank는 필요하지 않음.

---

# 8. Notation

아래 표는 본문 수식에 등장하는 기호를 모두 포함함. 동일 문자를 서로 다른 의미로 재사용하지 않도록 정리함.

| Symbol | Definition |
|---|---|
| \(P\) | PAN input |
| \(S\) | LRMS input |
| \(\mathcal U\) | LRMS를 PAN 해상도로 확대하는 고정 upsampling operator |
| \(M=\mathcal U(S)\) | Upsampled MS이자 M-frame reconstruction base |
| \(Y\) | Training HRMS ground truth |
| \(A_T,A_S\) | Teacher / Student PAN Aligner |
| \(F_T,F_S\) | Teacher / Student reconstruction network |
| \(\mathcal W\) | Predicted 또는 synthetic shift를 적용하는 differentiable PAN sampling operator |
| \(\operatorname{Concat}(\cdot,\cdot)\) | Channel-wise concatenation |
| \(\mathbf c_0\) | Stage 1 native pair의 predicted PAN correction |
| \(\mathbf c_{\epsilon}\) | Stage 1 shifted pair의 predicted PAN correction |
| \(\boldsymbol\epsilon\) | Stage 1 synthetic PAN translation |
| \(P_{\epsilon}\) | Synthetic translation이 적용된 PAN |
| \(\widetilde P_0,\widetilde P_S\) | Stage 1 Teacher / Stage 2 Student 경로에서 보정된 PAN |
| \(Z_T,Z_S\) | Teacher / Student HRMS prediction |
| \(\mathcal L_{\mathrm{rec}}\) | Stage 1 native HRMS reconstruction loss |
| \(\mathcal L_{\mathrm{con}}\) | Stage 1 relative-shift consistency loss |
| \(\lambda_{\mathrm{con}}\) | Stage 1 consistency loss weight |
| \(\mathcal E\) | Stage 2 q 계산에 사용하는 고정 PAN-shift probe bank |
| \(N_q\) | q 계산에 사용하는 probe 개수 |
| \(\mathbf c^T_{i,0}\) | Sample \(i\)의 frozen Teacher Aligner native correction |
| \(\mathbf c^T_{i,j}\) | Sample \(i\)의 \(j\)번째 probe 적용 후 Teacher Aligner correction |
| \(q_i\) | Sample \(i\)의 alignment-consistency error |
| \(q_{\mathrm{ref}}\) | Train q distribution에서 고정한 reference scale |
| \(s_i\) | q-derived alignment reliability weight |
| \(e_T(i,p),e_S(i,p)\) | 위치 \(p\)의 Teacher / Student GT reconstruction error |
| \(\Delta_{TS}(i,p)\) | 위치 \(p\)의 Teacher–Student prediction discrepancy |
| \(d_T(i,p)\) | Normalized Teacher reconstruction difficulty |
| \(a_T(i,p)\) | Teacher가 Student보다 정확한 정도를 나타내는 soft-supervision weight |
| \(\tau_R\) | Reconstruction-error reference scale |
| \(\delta_R\) | \(a_T\) 계산의 numerical stabilizer |
| \(\alpha\) | GT hard difficulty weighting coefficient |
| \(\beta\) | Teacher soft supervision coefficient |
| \(\ell_i^{\mathrm{hard}}\) | Sample \(i\)의 GT hard reconstruction term |
| \(\ell_i^{\mathrm{soft}}\) | Sample \(i\)의 Teacher soft reconstruction term |
| \(D_x,D_y\) | Horizontal / vertical signed edge operators |
| \(\Omega_E\) | Edge loss를 계산하는 valid spatial support |
| \(E_i\) | Sample \(i\)의 GT edge error |
| \(\lambda_E\) | Edge supervision coefficient |
| \(\mathcal L_{\mathrm{edge}}\) | q-weighted GT edge loss |
| \(\mathcal L_U\) | Student U-Net objective |
| \(\mathcal L_A\) | Student Aligner objective |
| \(\theta_S,\phi_S\) | Student U-Net / Student Aligner parameter set |
| \(\eta_U,\eta_A\) | Student U-Net / Student Aligner learning rate |
| \(\mathcal D_{\mathrm{train}}\) | Training dataset used for q reference calibration |
| \(i\) | Sample index |
| \(j\) | q probe index |
| \(c\) | MS spectral-band index |
| \(p\) | Spatial position index |

---

# 9. Claim Scope

1. **\(q\)는 absolute misalignment GT가 아님.**  
   Controlled PAN shift에 대한 frozen Teacher Aligner의 response consistency를 측정하는 error임.

2. **\(e\)는 predictive uncertainty가 아님.**  
   Training GT를 이용해 계산하는 reconstruction error임.

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

먼저 Stage 1에서는 PAN Aligner와 Teacher reconstruction network를 native HRMS reconstruction objective로 공동 학습함. 이후 알려진 synthetic PAN translation을 이용한 relative-shift consistency를 추가하여, Aligner가 입력 displacement 변화에 일관되게 반응하도록 학습함. 이때 shifted PAN은 Aligner supervision에만 사용하고 Teacher reconstruction branch에는 입력하지 않음.

Stage 2에서는 Stage 1에서 얻은 Teacher와 Teacher Aligner를 고정하고 경량 Student를 학습함. Reconstruction error \(e\)는 GT hard fitting과 Teacher soft distillation의 상대적 강도를 조절하며, alignment-consistency error \(q\)는 GT edge supervision과 Student Aligner adjustment의 신뢰도를 제어함. Student U-Net과 Student Aligner는 동일한 forward를 공유하지만 서로 다른 objective를 통해 갱신되어, Teacher output distillation과 geometry adaptation이 불필요하게 뒤섞이지 않도록 함.

최종적으로 본 방법은 **복원 오류와 정합 오류를 분리한 supervision routing**을 통해, Teacher의 오류를 무조건 모방하거나 불안정한 PAN geometry를 강제하지 않으면서 Student를 입력 데이터에 보다 정밀하게 fitting하는 것을 목표로 함.

---

## One-line Method Summary

> **재구성 오차 \(e\)로 GT와 Teacher 사이의 reconstruction supervision을 조절하고, 정합 일관성 오차 \(q\)로 geometry-sensitive supervision과 Student Aligner adaptation의 신뢰도를 조절함.**
