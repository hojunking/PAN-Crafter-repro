# PANDA Stage 1 — Edge-Guided Residual PAN Alignment

**최종 방법론·구현 인계서 / 2026-10-13**  
**범위:** Stage 1 Teacher reconstruction + PAN Aligner 학습만 포함. Stage 2의 KD, `e/q` 가중치, Student 학습, 비교 실험 case 및 ablation 설계는 포함하지 않는다.  
**상태:** 아래 `[확정]`은 이번 대화에서 합의한 설계이며, 이미 서버에 구현되거나 성능이 검증되었다는 의미가 아니다. `[구현 확인]`은 코드·데이터에서 확인할 항목, `[열린 항목]`은 합의되지 않은 세부 수치/운영 정책을 뜻한다.

> **핵심:** Scharr 기반의 간단한 **비학습 EdgeMatch**가 매 입력의 PAN→MS 초기 이동량 `b`를 구한다. 기존 PANDA의 경량 CNN은 이 초기 이동으로 미리 보정된 PAN을 보고 **연속값 residual translation `Δu`**만 예측한다. 최종 이동량 `u=b+Δu`는 **현재 입력 PAN pixel 단위**이며, 원본 PAN을 `u`로 **한 번만 warp**하여 Teacher reconstruction에 제공한다.

---

## 0. 이번 Stage 1에서 확정된 변경

| 요소 | 최종 명세 |
|---|---|
| 학습 데이터 | PanCollection 계열 **RR train**: PAN `[B,1,64,64]`, LRMS `[B,C,16,16]`, HRMS GT `[B,C,64,64]` |
| Aligner 외부 관측 | **원본 PAN + 관측 LRMS**. Aligner 내부에 전달할 MS reference `L_A`는 PAN grid로 위상 정합되게 생성한 LMS(또는 검증된 저장 `lms`) |
| 초기 shift | **입력 쌍별** Scharr edge-matching `b_i`; source-scene 매핑 불필요. 센서/데이터셋 고정 `b_global` 및 `b_s`는 사용하지 않음 |
| MS를 grayscale로 만드는 순서 | **RGB 세 밴드를 먼저 평균**, 그 결과에 Scharr를 적용 |
| Edge operator | **수평·수직 Scharr absolute gradient** `G(I)=[|S_x*I|,|S_y*I|]`; `EdgeMatch`와 `L_abs`가 같은 연산 사용 |
| Edge normalization | **영상/edge map별 추가 정규화 없음**: z-score, min–max, NCC, gradient amplitude normalization 및 learned descriptor 없음 |
| EdgeMatch 비교 | 정수 PAN-pixel 후보 변위별 두 축 absolute-gradient **L1 평균**; 최적 후보를 초기 `b_i`로 채택. 초기 subpixel fitting 없음 |
| Learned correction | `Δu_i = A_res(W(P_i,b_i),L_A)`; **직접 Cartesian `(Δdy,Δdx)`** 회귀. `α/β`, direction/perpendicular basis 및 scene-magnitude head conditioning 없음 |
| Aligner backbone | 기존 PANDA dual stem + joint CNN + **U-Net과 동일한 LN 기반 64ch ResBlock** + GAP + FC `64→32→2` |
| Aligner 초기화 | residual FC2 weight/bias zero-init → 시작 시 `Δu=0`, 총 shift `u=b` |
| Reconstruction | `W(P,b+Δu)`(원본 PAN에서 최종 **한 번** warp) + 기존 bicubic-upsampled LRMS → Teacher U-Net → HRMS |
| Stage 1 loss | `L_S1 = L_rec^RR + L_abs^RR + λ_ε L_ε^RR`; FR 학습 consistency 없음 |
| Gradient | `L_rec`→Teacher U-Net **및** PAN Aligner; `L_abs`·`L_ε`→PAN Aligner **만**; hard EdgeMatch는 학습하지 않음 |
| RR/FR 추론 | **동일한 입력별 EdgeMatch + CNN residual**. `b`, `Δu`, `u`는 **각 입력 PAN grid**의 pixel 단위. FR 출력에 임의 ×4 없음 |

**버전 차이 주의:** 이 문서는 기존 `PANDA_Method_Implementation_Validation_Plan_2026-10-13_v1_3.md`의 **scene-prior-conditioned** Stage 1 및 `PANDA_bGlobal_RR_Implementation_Changes_2026-10-13.md`의 **sensor-global** Stage 1을 **대체하는 새 Stage 1 명세**다. 원래의 `b_s`, `b_global`, `concat(h,b)`, `αd+βn`, Stage 1 `L_ε^FR`을 이번 구현에 섞지 않는다.

---

## 1. 입력·격자·참조 좌표계

### 1.1 입력과 출력

관측 PAN 및 LRMS:

\[
P\in\mathbb R^{B\times1\times H\times W},\qquad
M\in\mathbb R^{B\times C\times H/4\times W/4}.
\]

* 학습(RR): `PAN 64×64`, `LRMS 16×16`, `HRMS GT 64×64`.
* RR 평가: `PAN 256×256`, `LRMS 64×64`, `HRMS GT 256×256`.
* FR 평가: `PAN 512×512`, `LRMS 128×128`, **HRMS GT 없음**.
* WV3는 `C=8`, QB/GF2는 `C=4`. `PAN 1 + WV3 MS 8 = 9`는 결합 입력 채널 수이지 HRMS GT 밴드 수가 아니다.

`L_A`는 PAN grid의 **Aligner 전용** MS reference다:

\[
L_A=U_{\mathrm{phase}}(M)\in\mathbb R^{B\times C\times H\times W}.
\]

PanCollection의 검증된 `lms`가 있으면 그것을 사용한다. 없으면 원본 `ms`에서 **저장 `lms`와 동일한 sampling phase가 확인된 연산**으로 재생성한다. 무조건 4× bicubic 결과를 Aligner 기준으로 대체하지 않는다. 기존 audit에서 bicubic과 데이터셋 native LMS 간 phase 차이가 관찰된 바 있다.

Teacher reconstruction에 사용하는 base는 **별도**로 유지한다:

\[
B_M=U_{\mathrm{bicubic}}(M).
\]

PAN을 움직이고 MS 좌표는 고정한다. `L_A`와 `B_M`는 서로 다른 목적이므로 동일 변수로 혼동하지 않는다.

### 1.2 Warp 부호·단위

기존 `pa/warp.py` 규약 그대로:

\[
\boxed{W(P,\mathbf u)(y,x)=P(y+u_y,x+u_x)},\quad \mathbf u=(dy,dx).
\]

`+dx`는 **sampling 좌표가 오른쪽으로 이동**한다는 뜻이며 결과 영상 내용은 왼쪽으로 이동한다. EdgeMatch가 반환하는 `b` 역시 별도 sign 변환 없이 이 `W`에 적용하는 **PAN→MS correction convention**이어야 한다. 이미지 정합 라이브러리가 `MS→PAN` 이동량을 반환하면 방향을 검증해 변환한다.

`b`, `Δu`, `u`, `ε`는 모두 현재 PAN의 pixel 단위다. FR에서 Aligner/EdgeMatch가 **FR PAN grid를 직접 본다면** 예측에 4배를 곱하지 않는다. 입력 해상도를 낮춰서 matching하는 별도 좌표계 변환은 이번 기본 Stage 1에 없다.

---

## 2. 하나의 공통 spatial-edge 정의

### 2.1 RGB 평균 → Scharr → 절댓값

모든 MS-derived RGB grayscale은 **밴드별 edge 평균이 아니라 RGB 값 평균 후 edge 추출**이다.

\[
\operatorname{Gray}_{RGB}(X)=\frac{X_R+X_G+X_B}{3}.
\]

밴드 순서는 sensor H5의 고정 band mapping을 따른다. 예를 들어 현재 repo의 WV3 순서는 `Coastal,B,G,Y,R,RE,NIR1,NIR2`, QB/GF2는 `B,G,R,NIR`이며, 앞의 임의 세 채널을 RGB로 취급하지 않는다. 코드의 indexing을 검증한다.

고정 Scharr 필터:

\[
S_x=\frac1{32}\begin{bmatrix}-3&0&3\\-10&0&10\\-3&0&3\end{bmatrix},
\qquad S_y=S_x^\top.
\]

\[
\boxed{G(I)=\big[|S_x*I|,\ |S_y*I|\big]}.
\]

**스케일 `1/32`는 Scharr 커널의 한 번 정한 고정 coefficient convention**이며, 영상이나 edge map을 보고 통계적으로 정규화하는 과정이 **아니다**. 같은 커널을 EdgeMatch와 `L_abs` 양쪽에서 사용한다. 현재 repository의 `ta2/registration.py`도 `cv2.Scharr(..., scale=1/32)`를 사용하지만 **그 코드의 NCC·Gaussian·bandwise matching 전체를 가져오는 것은 아니다**.

* Edge operator에는 영상별 `mean/std`, min–max, NCC, adaptive normalization, learned normalization **추가하지 않는다**.
* Scharr convolution의 경계는 정해진 reflection padding을 사용하고, 각 candidate score에서는 공통 유효 영역만 비교한다.
* **기존 PAN Aligner CNN stem의 sample/channel-wise `znorm`은 별도 신경망 입력 처리이며 그대로 유지**한다. 이는 **Scharr edge map normalization이 아니다**. 데이터 로더가 원래 사용하던 intensity scaling도 변경하지 않는다.
* Scharr의 **두 방향을 유지**한다. `|dx|+|dy|`로 먼저 하나의 scalar edge map을 만드는 정의와는 다르다.

### 2.2 기존 연구와의 관계

* **원래 PANDA Fig. 2:** 블러된 PAN의 **Otsu iso-contour**를 이용해 불일치를 보여주는 **시각화**다. EdgeMatch 학습 알고리즘이 아님.
* **SIPSA-Net 공개 구현:** 이름은 `sobelfilter2XY()`이지만 핵심 연산은 표준 Sobel/Scharr가 아닌 **간단한 인접 2D 차분**, reflect padding이다. RGB 평균 후 절댓값 gradient 간 L1을 사용하는 **원리**를 참고하되, 이번 PANDA의 이산 미분 연산자는 **Scharr로 구체화**한다.
* 새 `L_abs`는 **원래 PANDA 논문에 있던 loss가 아니라 이번 연구에서 추가한 SIPSA-inspired RR absolute supervision**이다.

---

## 3. 비학습 EdgeMatch — 입력별 초기 translation

EdgeMatch의 역할은 **거친 initial guidance 제공**으로 제한한다. 복잡한 learned correlation, Otsu binarization, feature descriptor, NCC, subpixel peak fitting, adaptive weighting, 추가 edge normalization을 넣지 않는다.

### 3.1 계산

원본 PAN과 위상 정합된 LMS의 RGB 평균으로 edge를 만든다.

\[
E_P=G(P),\qquad E_M=G\!\left(\operatorname{Gray}_{RGB}(L_A)\right).
\]

현재 PAN grid의 **정수 translation 후보** `δ=(dy,dx)∈D`에 대해, 동일한 **공통 내측 영역** `Ω`에서:

\[
S_i(\boldsymbol\delta)=
\operatorname{mean}_{p\in\Omega,\ k\in\{x,y\}}
\left|E_{P,i}^{(k)}(p+\boldsymbol\delta)-E_{M,i}^{(k)}(p)\right|.
\]

\[
\boxed{\mathbf b_i=\underset{\boldsymbol\delta\in\mathcal D}{\operatorname{argmin}}\ S_i(\boldsymbol\delta)}.
\]

이 수식은 정수 shift, 공통 interior에서는 `G(W(P,δ))`의 위치 비교와 같은 취지다. 실제 구현은 **미리 계산한 Scharr map을 정수 indexing/slicing**하여 candidate scores를 구할 수 있다. 이때 후보마다 평가 pixel 개수가 달라지거나 padding 경계가 점수에 들어가면 안 된다.

**[확정]** `b_i`는 **매 입력 쌍**에 대해 새로 구한다. train 집합 전체 또는 source-scene 전체에서 구한 고정 `b_global`/`b_s`를 사용하지 않는다. 탐색 결과 `b_i`는 no-gradient tensor이며 전체 RR/FR 입력 patch/이미지에 대해 **2D 벡터 하나**다. Pixel별 flow field를 예측하지 않는다.

**[열린 항목 / 구현 전 결정]** 후보 grid의 정수 탐색 반경 및 실패/동점 처리 규칙의 세부 수치는 대화에서 정해지지 않았다. 임의의 반경이나 confidence threshold를 논문 확정값처럼 쓰지 않는다. 최소한 검색 범위를 **현재 PAN px** 단위로 선언하고 RR/FR 및 회전/flip에서 실제 후보 coverage, low-texture, `no valid region`을 명시적으로 처리해야 한다. 성능 평가용 test/FR GT를 이용해 탐색 반경을 최적화하지 않는다.

**의도된 단순성:** EdgeMatch 자체가 0.1 px 수준으로 정합할 필요는 없다. 정수 픽셀의 대략적 일치 후보를 제공하며, subpixel 잔차는 CNN이 연속값으로 학습한다. 다만 cross-modal contrast 차이로 raw edge L1의 최솟값이 잘못될 수 있다는 한계는 인정한다.

### 3.2 초기 PAN 정합

\[
\boxed{P_i^{pre}=W(P_i,\operatorname{sg}(\mathbf b_i))}.
\]

`P_pre`는 **Residual CNN 입력에만 사용하는 내부 영상**이다. 이 사전 warp의 interpolation이 최종 HRMS에 누적되도록 두지 않는다. LMS/GT는 정합을 위해 이동하지 않는다.

---

## 4. PANDA 경량 Residual CNN — 레이어 증설 없이 잔차만 학습

### 4.1 네트워크 흐름

```text
P (PAN) ─────────────────┐
                         ├─ Scharr EdgeMatch ── b (integer dy,dx, nonlearned)
M (LRMS) ─> L_A ──────────┘              │
                                        ▼
                             P_pre = W(P, b)
                                        │
                      ┌─────────────────┴────────────────┐
                      │                                  │
                PAN stem 1→16                     LMS stem C→16
                      └─────────── Concat 32 ─────────────┘
                                        │
                             joint Conv 32→32 (s2)
                                        │
                             joint Conv 32→64 (s2)
                                        │
                          common U-Net ResBlock(64)
                                        │
                                      GAP64
                                        │
                             FC 64→32 → SiLU → FC 32→2
                                        │
                             Δu = (Δdy, Δdx)  [real-valued]
                                        │
                               u = b + Δu
                                        │
                             W(P_original, u)  # exactly once
                                        │
                            Teacher Reconstruction U-Net
```

**[확정]** 기존 PANDA와 동일한 규모의 feature extractor를 유지한다.

| 레이어 | 세부 |
|---|---|
| PAN stem | `ReflectionPad(1) → Conv3×3(1→16) → GN(4) → SiLU` |
| LMS stem | `ReflectionPad(1) → Conv3×3(C→16) → GN(4) → SiLU` |
| Joint1 | `Conv3×3(32→32, stride2) → GN(8) → SiLU` |
| Joint2 | `Conv3×3(32→64, stride2) → GN(8) → SiLU` |
| ResBlock | **Reconstruction U-Net과 공통의 pre-LN residual block**: LN→SiLU→Conv3×3→LN→SiLU→Dropout(0)→zero-init Conv3×3, identity addition 이후 activation 없음 |
| Head | GAP(64) → FC(64→32) → SiLU → FC(32→2), 최종 FC weight/bias 0 초기화 |

기존 Aligner `_Res(64,GN,post-SiLU)`는 공통 U-Net ResBlock으로 대체하되 **stem/joint GN은 유지**한다. `model/pancrafter_paper.py`의 공통 `ResBlock` 구현을 재사용하거나 정확히 같은 연산을 공유한다. 기존 ResBlock의 `forward(x,s)` 호출 형식도 점검한다. Teacher U-Net 본체 아키텍처는 변경하지 않는다.

새 입력은 `[P_pre,L_A]`, 출력은 **`Δu=(Δdy,Δdx)`**다. 과거 `concat(h, b_s)` / `concat(h,b_global)` → FC66, 방향 성분 `α,β`, magnitude/vector basis 및 learnable prior는 **모두 제거**한다.

### 4.2 학습 초깃값과 최종 warp

\[
\boxed{\Delta\mathbf u_i=A_\phi^{res}(P_i^{pre},L_{A,i})},
\quad
\boxed{\mathbf u_i=\mathbf b_i+\Delta\mathbf u_i}.
\]

최종 FC2 zero-init이면 처음에는 `Δu=0`, `u=b`다. Aligner ResBlock의 마지막 Conv도 zero-init이므로 초반 gradient 전달이 단계적으로 발생할 수 있다. 이는 구조적 특성이지 성능 향상이 보장된다는 뜻은 아니다.

**반드시 최종 reconstruction PAN은 원본 `P`에서 한 번만 warp**한다:

\[
\boxed{\widetilde P_i=W(P_i,\mathbf u_i)}.
\]

`W(P_pre,Δu)`를 Teacher 입력으로 사용하면 interpolation이 누적되므로 **금지**한다. `P_pre`는 residual head의 입력용이다.

---

## 5. Teacher reconstruction 경로

`L_A`와 독립적인 기존 reconstruction base:

\[
B_i=U_{\mathrm{bicubic}}(M_i).
\]

\[
\boxed{\hat Y_{T,i}=B_i+F_{T,\theta}([\widetilde P_i,B_i])}.
\]

기존 reconstruction U-Net의 입력/폭/depth/Residual 구조는 유지한다. Teacher reconstruction은 **native RR branch**에서만 계산하며, 인위적으로 ε-shift된 PAN을 Teacher U-Net에 추가 입력하거나 별도 reconstruction loss를 걸지 않는다.

RR HRMS GT를 `Y_i`라 하면:

\[
\boxed{L_{\rm rec}^{RR}=\operatorname{mean}_{i,c,p}|\hat Y^c_{T,i}(p)-Y_i^c(p)|}.
\]

`L_rec`의 gradient는 **Teacher U-Net과 Residual PAN Aligner** 양쪽으로 흐른다. Hard EdgeMatch는 비미분 고정 초기값이므로 업데이트되지 않는다.

---

## 6. RR absolute spatial loss — 동일한 Scharr

RR GT의 **RGB 평균 후 Scharr**:

\[
Y_{\mathrm{lum},i}=\operatorname{Gray}_{RGB}(Y_i)
=\frac{Y_i^R+Y_i^G+Y_i^B}{3}.
\]

\[
\boxed{
L_{\mathrm{abs}}^{RR}
=\operatorname{mean}_{i,p,k\in\{x,y\}}
\left|
G_k\big(W(P_i,\mathbf u_i)\big)(p)
-G_k\big(Y_{\mathrm{lum},i}\big)(p)
\right|
}.
\]

`G_x(I)=|S_x*I|`, `G_y(I)=|S_y*I|`. 추가 gradient normalization, NCC, edge detector 학습, Otsu 임계화는 사용하지 않는다.

**차이의 핵심:** EdgeMatch는 `PAN↔L_A의 RGB 평균`으로 **초기 `b`를 찾아내고**, `L_abs`는 `최종 corrected PAN↔GT의 RGB 평균` 사이의 **reduction-free edge 오차를 L1로 지도**한다. 양쪽 모두 cross-modal edge 비교다. GT는 초기 EdgeMatch 또는 FR inference의 입력이 아니다.

`L_abs`는 loss 경로에 Teacher U-Net을 통과시키지 않는다. **Residual PAN Aligner만 업데이트**한다. PAN/GT의 spectral response 차이 때문에 `L_abs=0`이 물리적인 정합의 필수 조건이거나 `L_abs`가 displacement GT인 것은 아니다.

---

## 7. RR synthetic-shift consistency

### 7.1 ε 생성

매 학습 update, sample마다 **현재 RR PAN pixel 단위의 2D translation ε 하나**를 독립적으로 만든다.

\[
u_1,u_2\overset{iid}{\sim}U(0,1),\quad
r=R_\epsilon\sqrt{u_1},\quad
\theta=2\pi u_2,
\]

\[
\boxed{\epsilon=(r\sin\theta,r\cos\theta)=(\epsilon_y,\epsilon_x)}.
\]

원판 면적 균등, 영 중심. 현재 논의에서 이어받는 **초기 반경 설정 `R_ε=2.0` RR PAN px**, **매 update 적용**, **초기 계수 `λ_ε=1e-3`**는 *제안된 학습 설정*이다. 과거 완료 실험에서 이 조합이 최적이라고 검증된 사실은 아니며, 반경 2의 역사적 산정에는 FR benchmark 입력 통계가 포함되어 있었다. 논문에서는 이 반경을 train-only로 정당화했다고 잘못 쓰지 않는다. `R_ε`는 EdgeMatch의 정수 탐색 반경과 **전혀 다른 quantity**다.

### 7.2 Native branch와 shifted branch

Native branch:

\[
\begin{aligned}
b_0 &= \operatorname{EdgeMatch}(P,L_A),\\
\Delta u_0 &= A_\phi^{res}(W(P,b_0),L_A),\\
u_0 &= b_0+\Delta u_0.
\end{aligned}
\]

Shifted branch는 **PAN만** synthetic warp하고, `L_A`, `M`, `Y` 및 MS coordinate frame은 유지한다:

\[
\begin{aligned}
P_\epsilon &= W(P,\epsilon),\\
b_\epsilon &=\operatorname{EdgeMatch}(P_\epsilon,L_A),\\
\Delta u_\epsilon &=A_\phi^{res}(W(P_\epsilon,b_\epsilon),L_A),\\
u_\epsilon &=b_\epsilon+\Delta u_\epsilon.
\end{aligned}
\]

**중요:** `b_ε`는 `P_ε`에서 **동일한 EdgeMatch를 다시 실행**해 구한다. `b_0`를 강제로 고정하거나 `b_ε=b_0-ε`을 정답처럼 주입하지 않는다. EdgeMatch는 각 branch에서 계산되는 non-learned guidance다.

\[
\boxed{
L_\epsilon^{RR}
=\operatorname{mean}_{i,j\in\{y,x\}}
\left|u_{\epsilon,i,j}+\epsilon_{i,j}-\operatorname{sg}(u_{0,i,j})\right|
}.
\]

`sg`는 **ε branch가 참조하는 `u0`에만** 적용한다. Native `u0`는 여전히 `L_rec`, `L_abs`를 통해 gradient를 받는다. `P_ε`를 만든 synthetic warp에는 gradient를 흘릴 필요가 없다. Shifted branch에는 Teacher reconstruction을 수행하지 않는다.

### 7.3 학습되는 부분과 주의

EdgeMatch가 ε에 맞춰 `b_ε-b_0≈-ε`로 변하면, ε loss는 이미 그 부분이 충족되어 CNN residual에 큰 gradient를 제공하지 않을 수 있다. 반대로 discrete integer EdgeMatch에서 남은 subpixel/quantization error는 residual CNN이 처리하도록 학습될 수 있다. CNN의 절대 정합 성질은 reconstruction과 `L_abs`가 함께 지도한다.

`L_ε`는 **알려진 추가 translation에 대한 반응**을 제약한다. 이것만으로 native PAN–MS absolute shift 정답이나 FR absolute alignment 정확성이 보장되는 것은 아니다.

---

## 8. Stage 1 전체 objective 및 gradient 계약

\[
\boxed{
L_{\rm Stage1}
=
L_{\rm rec}^{RR}
+
L_{\rm abs}^{RR}
+
\lambda_\epsilon L_\epsilon^{RR}
}.
\]

`L_rec`, `L_abs`의 앞 계수 1은 이번 방법론의 초기 구성이다. Scharr 계수·영상 DN/intensity scaling에 따라 실제 loss 규모가 변하므로 구현은 두 항의 크기와 gradient를 기록해야 한다. 현재 `λ_ε=1e-3` 역시 초기 설정이며 최적 성능을 주장하지 않는다.

| 손실 | Teacher U-Net gradient | PAN Aligner residual CNN gradient | EdgeMatch gradient |
|---|---|---|---|
| `L_rec^RR` | **있음** | **있음**: `u0→warp→U-Net` | 없음 (`b0` stop) |
| `L_abs^RR` | 없음 | **있음**: `u0→warp→Scharr` | 없음 (`b0` stop) |
| `L_ε^RR` | 없음 | **있음**: shifted `uε` branch | 없음 (`bε` stop) |

**FR training image가 없으므로** `L_ε^FR`과 FR HRMS GT-based training loss는 이번 Stage 1에서 **없다**. 평가용 FR 영상으로 학습 update를 하지 않는다. FR에서는 동일한 PAN+LRMS 입력 처리 규칙으로 `EdgeMatch`와 trained residual CNN을 적용하는 **inference**만 수행한다.

### 8.1 참고 실행 흐름(구현 의사코드)

```python
# Shapes: P[B,1,H,W], M[B,C,H/4,W/4], Y[B,C,H,W] (RR train only)
# L_A: checked native LMS / phase-consistent interpolation, [B,C,H,W]
# B_M: independent bicubic(M), [B,C,H,W]

def predict_shift(P, L_A):
    with torch.no_grad():
        b = edge_match_scharr_integer(P, L_A)  # [B,2], (dy,dx), current PAN px
    P_pre = warp_pan(P, b)                   # CNN input guidance only
    delta = residual_aligner(P_pre, L_A)     # [B,2], learnable continuous residual
    u = b + delta                           # [B,2], current PAN px
    return u, b, delta

u0, b0, delta0 = predict_shift(P, L_A)
P_aligned = warp_pan(P, u0)                 # ORIGINAL PAN, exactly one final warp
Yhat = B_M + teacher_unet(P_aligned, B_M)   # keep original U-Net layout
L_rec = (Yhat - Y).abs().mean()
L_abs = (scharr_abs_xy(P_aligned) -
         scharr_abs_xy(rgb_mean(Y))).abs().mean()

with torch.no_grad():
    epsilon = uniform_disk(batch=P.size(0), radius=2.0)  # independent RNG
    P_eps = warp_pan(P, epsilon)
u_eps, b_eps, delta_eps = predict_shift(P_eps, L_A)
L_eps = (u_eps + epsilon - u0.detach()).abs().mean()

L_stage1 = L_rec + L_abs + lambda_eps * L_eps
L_stage1.backward()  # U-Net only rec; residual_aligner all three
```

이 코드는 **계산 그래프와 입력/출력 계약을 표현하는 pseudocode**다. `rgb_mean`은 센서별 밴드 순서를 반드시 사용하며, `teacher_unet`의 실제 시그니처는 기존 구현에 맞춘다. Scharr 비교에서 경계가 점수/학습 결과를 왜곡하지 않도록 동일 padding 및 interior-support 규칙을 적용한다.

---

## 9. RR 학습과 FR 추론의 연결 — Stage 1 범위에서의 원칙

1. **원본 PAN/LRMS의 4:1 관계를 유지**한다. FR을 불필요하게 RR-equivalent GSD로 전부 다운샘플링하지 않는다.
2. EdgeMatch는 **각 입력 영상(또는 처리 patch)에서 다시 수행**한다. 결과 `b`는 입력 PAN pixel 단위다. FR에서 RR `b_global`을 가져오거나 CNN output을 무조건 ×4하지 않는다.
3. FR의 displacement가 RR보다 큰 경우 EdgeMatch 검색 후보 범위가 실제 displacement를 포함해야 한다. 이 범위는 사전에 선언된 물리적·입력 pixel 단위 규칙이어야 한다. FR metric을 보고 검색 반경을 튜닝하지 않는다.
4. 한 forward의 residual CNN은 **하나의 global 2D shift**를 출력한다. FR 512×512 전체 forward는 하나의 `u`이며, 별도로 여러 patch를 처리한다면 각 patch마다 `u`가 생성된다. Global correction 하나가 목표이면 aggregation 규칙은 별도 정의해야 한다. GSD에 맞춘 spatial crop 크기를 사용해도 CNN의 pixel-grid receptive field까지 자동으로 같아지는 것은 아니다.
5. 원해상도 FR에 training GT가 없으므로, **FR에서 residual의 통계가 완전히 같거나 정합이 개선된다고 가정하지 않는다**. EdgeMatch가 큰 shift를 초기 보정할 수는 있지만 PAN–MS spectral response 차이, interpolation phase, 고주파 구조 차이 및 움직이는 객체로 인한 국소 변형은 남는다.
6. 평가 input 자체에서 고정 EdgeMatch를 재실행하는 것은 **입력 기반 결정적 전처리**다. 신경망 weight를 FR test에서 업데이트하지 않으며 전처리 비용을 보고해야 한다.

---

## 10. 구현 일관성 및 오류 방지 체크

- [ ] `b`는 **입력별** non-learned EdgeMatch 산출값인가? `b_s`·`b_global` 상수 자산이 연결되지 않았는가?
- [ ] EdgeMatch에서 **PAN/LMS의 GT-독립 RGB 평균**을 사용하고, `L_abs`만 **RR HRMS GT RGB 평균**을 사용하는가?
- [ ] Scharr의 **두 방향 absolute gradient**와 고정 커널 스케일·padding이 EdgeMatch 및 `L_abs`에서 같고, edge map에 **추가 normalization이 없는가**?
- [ ] WV3, QB, GF2 **RGB 실제 밴드 인덱스**가 확인되었는가? WV3 8밴드의 앞 3개를 RGB라고 가정하지 않는가?
- [ ] EdgeMatch는 **정수 PAN-pixel shift**를 같은 유효 pixel 집합에서 비교하며, 반환값의 `(dy,dx)`와 warp 부호가 맞는가?
- [ ] `b_0`와 `b_ε`는 각 PAN으로부터 **별도로 재계산**되며, ε injection 후 MS/LMS/GT는 움직이지 않는가?
- [ ] Aligner는 `[W(P,b),L_A]`를 받고 `[B,2]` **연속 `(Δdy,Δdx)`**를 출력하는가? `α/β`, scene basis, head `66→32`가 제거됐는가?
- [ ] Aligner의 ResBlock은 **U-Net 공통 LN pre-activation block**, stem/joint는 **기존 GN**인가? Teacher U-Net 자체는 변경되지 않았는가?
- [ ] FC2 zero-init으로 `Δu(init)=0`, `u(init)=b`가 되고, 초반 residual head gradient가 올바르게 전달되는가?
- [ ] Reconstruction에서 **`W(P_original,b+Δu)` 한 번만** 적용하고 `W(W(P,b),Δu)`를 Teacher 입력으로 사용하지 않는가?
- [ ] Native `u0`는 `L_rec+L_abs` gradient를 받되 ε anchor에서만 stop-gradient되고, U-Net은 `L_rec`만 받는가?
- [ ] FR test 영상은 **추론/평가로만** 사용되며 Stage 1 `L_ε^FR`로 update하지 않는가?
- [ ] EdgeMatch의 빈/동점/저텍스처 및 유효영역 미충족 처리를 침묵 없이 기록하고, 검색 반경 등 미확정 수치를 구현·논문에 실제 사용값으로 고정해 명시했는가?

---

## 11. 출처 및 구현 상태의 구분

**원래 방법으로 확인된 내용**

* 기존 PANDA 초안 `59391_PANDA_PAN_Alignment_Awar.pdf`, §3.2: PAN을 MS frame으로 이동하는 **sample-wise 2D Aligner**, RR GT reconstruction과 synthetic shift consistency. **새 EdgeMatch나 `L_abs`는 원래 PANDA 구현이 아니다.**
* 같은 PANDA 초안 Fig. 2: 블러링된 PAN의 **Otsu iso-contour를 시각화에 사용**. 정합 추정기로 정의하지 않음.
* [SIPSA-Net 공식 코드 `helperTF.py` — sobelfilter2XY](https://github.com/brachiohyup/SIPSA/blob/main/helperTF.py#L341-L360): 2D 인접 차분 + reflect padding, **Scharr가 아님**.
* [SIPSA-Net 공식 코드 `main_train.py` — edge loss](https://github.com/brachiohyup/SIPSA/blob/main/main_train.py#L215-L226): RGB 평균 후 절댓값 2D gradient L1 원리.
* [PANDA 재현 저장소 `pa/aligner.py`](https://github.com/hojunking/PAN-Crafter-repro/blob/main/pa/aligner.py): 기존 dual stem/GN/joint/GAP/FC 및 `znorm`; **이 문서의 EdgeMatch+공통 LN ResBlock 수정이 이미 반영된 것은 아님**.
* [기존 `ta2/registration.py`](https://github.com/hojunking/PAN-Crafter-repro/blob/main/ta2/registration.py): 별도 Scharr/32 + NCC matching 구현의 존재만 참고. **이번 단순 EdgeMatch의 raw-absolute-gradient L1과 동일하지 않음**.
* [기존 `pa/warp.py`](https://github.com/hojunking/PAN-Crafter-repro/blob/main/pa/warp.py): `W(P,u)(y,x)=P(y+dy,x+dx)`, bicubic grid sampling, border, `align_corners=False`.
* 기존 `PAN_Epsilon_Offset_Method_2026-10-13.md`: ε 원판 면적 균등/손실/warp/기존 진단 기록.

**이번 Stage 1의 새 설계 선택**: Scharr-based deterministic input-specific EdgeMatch, RGB-average-first shared Scharr `G`, integer candidate L1, residual CNN pre-alignment input, `b+Δu`, common U-Net LN ResBlock for Aligner, RR-only `L_abs` 및 `L_ε`. 검증 완료 사실로 기술하지 않는다.
