# PANDA Stage 1 — Prior-Free Direct PAN Alignment

**방법론·구현 기술서 (Stage 1 only) | 2026-10-14**  
**버전:** `S1-DirectShift-Scharr-EpsilonR3-v1.0`  
**범위:** PAN Aligner와 Teacher Reconstruction U-Net의 Stage 1 학습, RR–FR 추론 연결, 구현 일관성 점검. **Stage 2, Student/KD, ablation 및 실험 case 설계는 다루지 않는다.**  
**해석 기준:** `[확정]` = 이번 대화에서 채택한 설계, `[유지]` = 직전 Stage 1 명세에서 바꾸지 않는 부분, `[검증 필요]` = 구현·데이터와 대조해야 할 조건, `[미확정]` = 이 대화에서 결정하지 않은 설정. 설계 사항을 검증된 성능 결과로 서술하지 않는다.

> **핵심 요약:** 부정확한 정수 EdgeMatch로 생성되는 초기 shift `b`를 **완전히 제거**한다. 기존 PANDA처럼 경량 Aligner CNN이 PAN–MS 사이의 **전체 2D global translation `(dy,dx)`를 직접 예측**한다. RR HRMS reconstruction, RGB 평균 후 Scharr absolute-gradient supervision, PAN-only synthetic shift consistency의 **세 학습 손실은 유지**한다. ε의 최대 길이 **`R_ε=3.0` RR PAN pixel**로 확장한다(기존 2.0 → 3.0). **Sampling 분포는 반경 확정과 별개 결정**이므로, 이 문서의 재현 가능한 기본 명세는 기존 **원판 면적 균등 `r=R√u` 유지**다. 지난 논의의 **반경 균등 `r=Ru`는 아직 확정된 변경으로 취급하지 않는다.**

---

## 0. 변경 이력 및 대체 범위

| 항목 | 직전 Edge-Guided Stage 1 | 이번 Direct-Shift Stage 1 |
|---|---|---|
| 비학습 초기 EdgeMatch | 매 native/ε branch마다 수행 | **삭제** |
| 고정/입력별 prior `b` | `b=EdgeMatch(P,L_A)` | **삭제** (`b_s`, `b_global`, `b_i` 모두 없음) |
| Aligner CNN 입력 PAN | 사전 정합 `W(P,b)` | **원본 PAN `P`** |
| CNN 회귀 목표 | `Δu=(Δdy,Δdx)` residual | **전체 shift `u=(dy,dx)` 직접 회귀** |
| Head 입력 | `GAP64→FC64→32→2` | **유지**; `b` concat 없이 FC64 |
| FC2 초기화 | residual 0 → 총 correction `b` | **출력 0 → 총 correction 0** |
| Reconstruction PAN | `W(P,b+Δu)` | **`W(P,u)`** |
| Edge operator | `G=[|Scharr_x|,|Scharr_y|]` | **`L_abs`에서만 유지**; EdgeMatch 목적은 삭제 |
| RR objective | `L_rec + L_abs + λ_ε L_ε` | **그대로 유지** |
| ε native/shifted | 두 branch에서 각각 EdgeMatch 후 CNN | **같은 CNN `A(P,L_A)` / `A(P_ε,L_A)`** |
| ε 최대 반경 | `R_ε=2.0` RR PAN px | **`R_ε=3.0` RR PAN px 확정** |
| ε sampling 분포 | 원판 면적 균등 | **기존 방식 유지**. 반경 균등은 별도 변경안 |
| Stage 1 FR 학습 | 없음 | **없음** |
| RR/FR inference | EdgeMatch + residual CNN | **직접 회귀 CNN만**; 현재 PAN grid shift 적용 |

**[적용 우선순위]** 이 파일은 다음 구형 Stage 1 정의를 **대체**한다.

- `PANDA_Stage1_EdgeGuided_Residual_Aligner_Final_2026-10-13.md`: EdgeMatch, prewarp, `b+Δu` 관련 절을 폐기한다.
- `PANDA_bGlobal_RR_Implementation_Changes_2026-10-13.md`: sensor-global `b_global`, `αd+βn`, head 66차원을 폐기한다.
- `PANDA_Method_Implementation_Validation_Plan_2026-10-13_v1_3.md`: scene prior `b_s`, head `[h;b_s]`, RR+FR ε 학습 경로는 더 이상 최신 Stage 1이 아니다.

이전 자료 중 **RR 데이터 구성, native LMS 및 bicubic reconstruction base 분리, warp 부호/precision, U-Net과의 공통 ResBlock, Scharr loss, gradient routing**만 아래와 같이 유지한다.

---

## 1. 문제 설정과 공간 좌표계

### 1.1 관측 데이터

PAN을 `P`, 저해상도 MS를 `M`, HRMS GT를 `Y`로 쓴다.

\[
P\in\mathbb{R}^{B\times1\times H\times W},\quad
M\in\mathbb{R}^{B\times C\times H/4\times W/4}.
\]

| Split/사용 | PAN | LRMS | HRMS GT | 역할 |
|---|---:|---:|---:|---|
| RR train/val | 64×64 | 16×16 | 64×64 | Stage 1 지도 학습/검증 |
| RR test | 256×256 | 64×64 | 256×256 | 평가 전용 |
| FR test | 512×512 | 128×128 | 없음 | native GSD 추론·평가 전용 |

WV3는 `C=8`, QB와 GF2는 `C=4`. **PAN+8-band MS가 9채널 입력인 것**과 `HRMS GT` 자체의 밴드 수(8)는 다르다. 학습 64²와 RR 평가 256²는 동일한 RR GSD 계층의 **서로 다른 H5 split**이며, train patch의 source-scene ID는 제공되지 않는 것으로 확인됐다. 이번 방법론은 이 ID를 필요로 하지 않는다.

### 1.2 Aligner 전용 MS 기준 `L_A`와 복원 base `B_M`

\[
\boxed{L_A=U_{\mathrm{phase}}(M)\in\mathbb R^{B\times C\times H\times W}}
\]

- **[유지]** Aligner는 PAN과 **같은 grid에 존재하는 MS reference**를 사용한다. `L_A`는 저장된 PanCollection `lms`가 검증돼 있으면 그 값을 우선 사용하며, 없으면 동일한 interpolation phase가 검증된 연산으로 생성한다.
- **[유지]** Reconstruction U-Net은 기존 bicubic base `B_M=U_bicubic(M)`를 사용한다. `L_A`와 `B_M`의 생성 방식을 무심코 통일하지 않는다.
- Aligner 함수의 외부 wrapper는 `(P,M,lms_if_available)`를 받을 수 있지만, **CNN stem에 실제 공급되는 쌍은 `(P,L_A)`**다. LMS는 관측 LRMS로부터 생성되는 기준 영상이며, 별도 관측 HRMS GT가 아니다.
- MS의 위치는 고정하며 PAN만 움직인다. GT가 Aligner 입력으로 들어가면 안 된다.

### 1.3 Shift convention 및 warp

\[
\boxed{W(P,\mathbf u)(y,x)=P(y+u_y,x+u_x),\qquad \mathbf u=(dy,dx).}
\]

- `+dx`는 **오른쪽 좌표에서 샘플링**하므로 결과 영상의 내용은 왼쪽으로 이동한다.
- `u`는 **그 시점에 입력된 PAN grid의 signed pixel 단위**다. RR train이면 RR PAN px, FR inference이면 FR PAN px다.
- `grid_sample` bicubic, `padding_mode='border'`, `align_corners=False`, grid/warp FP32(필요시 검증된 FP64 경로)는 기존 처리 규약을 유지한다. 좌표 normalization은 `gx=2(x+dx+0.5)/W-1`, `gy=2(y+dy+0.5)/H-1`이다.
- 최종 보정에 `×4`, `tanh`, `clamp`, 정수 반올림, `b` 추가, per-image confidence gate를 **도입하지 않는다**.

---

## 2. Stage 1 전체 흐름

```text
                    P : original PAN [B,1,H,W]
                    M : observed LRMS [B,C,H/4,W/4]
                                     │
                         Phase-correct LMS reference
                                     │ L_A [B,C,H,W]
            ┌────────────────────────┴───────────────────────┐
            │                                                │
       Native PAN P                                Known shift ε (RR only)
            │                                                │
            │                                         P_ε = W(P, ε)
            │                                                │
            ▼                                                ▼
       A_φ(P,L_A)                                     A_φ(P_ε,L_A)
            │ u_0 [B,2]                                     │ u_ε [B,2]
            │                                                │
            ▼                                                ▼
       P̃ = W(P,u_0)                           L_ε = |u_ε + ε - sg(u_0)|
            │                                           (Aligner only)
            ├─────────────────────┐
            │                     │
      Teacher U-Net          Scharr spatial edge
        + B_M base          G(P̃) ↔ G(Gray_RGB(Y))
            │                     │
            ▼                     ▼
       HRMS Ŷ_T                 L_abs
            │
            ▼
        L_rec(Ŷ_T,Y)

        Total: L_S1 = L_rec + L_abs + λ_ε L_ε
```

- ε branch에는 **Teacher Reconstruction U-Net을 연결하지 않는다**.
- 최종 native reconstruction에서 PAN은 **원본 P로부터 한 번만** `W(P,u_0)`한다. 삭제된 EdgeMatch 사전 warp 경로를 남기지 않는다.

---

## 3. PAN Aligner 아키텍처 — 기존 경량 CNN 직접 회귀

### 3.1 계산

\[
\boxed{\mathbf u_0=A_\phi(P,L_A)\in\mathbb R^{B\times2}.}
\]

각 sample의 입력 전체에서 **한 개의 global 2D translation**을 예측한다. Dense flow나 pixel-wise field를 출력하지 않는다. 학습 patch 64×64 하나에 correction 한 개이며, FR 512×512 전체 forward에서도 image-level correction 한 개다. 추론 시 별도 tiling/aggregation 알고리즘을 이번 기본 방법에 추가하지 않는다.

### 3.2 레이어 구성

| 순서 | 모듈 | 유지/변경 |
|---|---|---|
| PAN 입력 전처리 | PAN 입력 sample/channel-wise spatial z-normalization; mean/variance detach | **기존 stem 입력 정규화 유지** |
| LMS 입력 전처리 | LMS 각 밴드 sample/channel-wise spatial z-normalization | 유지 |
| PAN stem | `ReflectionPad2d(1)→Conv3×3(1→16,bias=False)→GN(4)→SiLU` | 유지 |
| LMS stem | `ReflectionPad2d(1)→Conv3×3(C→16,bias=False)→GN(4)→SiLU` | 유지 |
| Joint | Concat `[16+16]` → `Conv3×3(32→32,stride=2)→GN(8)→SiLU` → `Conv3×3(32→64,stride=2)→GN(8)→SiLU` | 유지 |
| Residual block | **Reconstruction U-Net과 공통 설계의 LN 기반 ResBlock(64)** | 유지 (기존 Edge-Guided에서 합의한 블록) |
| Pooling | Global Average Pooling over spatial coordinates → `[B,64]` | 유지 |
| Regression | `FC(64→32)→SiLU→FC(32→2)` | **총 shift `(dy,dx)` 출력** |

**공통 LN ResBlock(64) 내부 규약:** channel-wise pre-LN → SiLU → bias 포함 Conv3×3 → channel-wise pre-LN → SiLU → dropout(0) → **zero-initialized** bias 포함 Conv3×3 → identity residual addition, **합산 후 activation 없음**. Aligner stem/joint의 GN(4/8)은 LN으로 바꾸지 않는다. Teacher U-Net의 본체는 구조 변경하지 않는다. 기존 `model/pancrafter_paper.py` 공통 ResBlock을 재사용할 때 `forward(x,s)` 호출과 `mode_mod=False` 등의 호환성을 확인한다.

### 3.3 초기화와 checkpoint

- Regression 최종 FC2의 `weight`, `bias`를 **모두 zero-init**한다. 따라서 최초 `u_0=(0,0)`이며 native warp는 displacement 0에서 시작한다.
- ResBlock의 최종 Conv도 zero-init이므로 초기 gradient가 FC2에서 앞단으로 단계적으로 전파될 수 있다. **초기 수십~수백 update의 각 layer gradient 전달을 확인**해야 한다. 이는 성능 안정성이 증명됐다는 뜻은 아니다.
- 이전 `EdgeMatch + residual` 모델의 checkpoint를 `fc1`/`fc2` 차원 일치만 보고 직접 shift 모델이라고 재사용하지 않는다. `FC` 출력 의미, 입력 이미지, zero-init 기준이 바뀌었다.
- 구형 `b_s`, `b_global`, 입력별 EdgeMatch, `α/β` direction basis, `[h;b]` 66차원 FC, `W(P,b)` prewarp 및 hard matching cache 사용을 **전부 금지**한다.

---

## 4. Teacher reconstruction과 L_rec

Native corrected PAN:

\[
\boxed{\widetilde P=W(P,\mathbf u_0)}.
\]

독립적 bicubic reconstruction base:

\[
B_M=U_{\mathrm{bicubic}}(M).
\]

Teacher U-Net은 기존 PANDA/PAN-Crafter 계열의 입력 및 residual prediction 구조를 유지한다.

\[
\boxed{\hat Y_T=B_M+F_{T,\theta}([\widetilde P,B_M])}.
\]

RR HRMS GT `Y`에 대하여:

\[
\boxed{L_{\mathrm{rec}}^{RR}=\operatorname{mean}_{b,c,y,x}|\hat Y_{T,b,c,y,x}-Y_{b,c,y,x}|}.
\]

- 복원 손실은 **native correction** `u_0`에 대해서만 계산한다.
- `L_rec` gradient는 Teacher U-Net `θ`와 Aligner `φ` **모두**로 흐른다. `u_0.detach()`를 reconstruction 경로에 넣지 않는다.
- U-Net 입력은 실행 서버의 기존 채널 구성과 residual base를 따르며, 별도 LPAN/HPAN 입력 등을 새로 추가하지 않는다.

---

## 5. RR absolute spatial supervision — RGB 평균 + Scharr

### 5.1 Grayscale 정의

RR GT의 실제 RGB 밴드 세 개를 **먼저 평균**한다.

\[
\boxed{Y_{\mathrm{lum}}=(Y_R+Y_G+Y_B)/3}.
\]

WV3는 H5 band order `Coastal,B,G,Y,R,RE,NIR1,NIR2`에서 RGB indices가 `R=4,G=2,B=1`(0-based), QB/GF2가 `B,G,R,NIR`이면 `R=2,G=1,B=0`이다. 실제 H5 밴드 계열/메타데이터 일치 여부는 loader manifest에서 확인한다. **앞의 임의 세 MS 밴드를 RGB라고 가정하지 않는다.**

### 5.2 고정 Scharr 2D absolute gradient

\[
S_x=\frac1{32}\begin{bmatrix}-3&0&3\\-10&0&10\\-3&0&3\end{bmatrix},\qquad
S_y=S_x^{\top}.
\]

\[
\boxed{G(I)=\big[|S_x*I|,|S_y*I|\big]}.
\]

- 두 gradient 방향을 개별적으로 유지해 `mean` reduction한다. `|dx|+|dy|`로 먼저 합쳐서 하나의 채널을 만드는 정의와 다르다.
- Scharr의 `1/32`는 **고정 커널 계수**이며, 영상별 gradient normalization이 아니다.
- `reflect` padding/고정된 convolution 위치와 비교 영역은 동일하게 사용한다. 경계 warp의 border extrapolation이 supervision을 왜곡하지 않도록 유효 support 처리 정책을 명시·검증한다.
- **추가 edge z-score, min–max, NCC, adaptive normalization, threshold/Otsu, 학습 가능한 descriptor 없음.** Aligner 내부의 기존 입력 `znorm`과 혼동하지 않는다.
- Scharr는 **학습 loss용 고정 spatial operator**이며, 이번 prior-free Aligner의 입력에는 별도 Scharr edge map을 연결하지 않는다. 두 stem은 원본 intensity PAN과 LMS를 받는다.

### 5.3 Absolute spatial loss

\[
\boxed{
L_{\mathrm{abs}}^{RR}
=\operatorname{mean}_{b,p,k\in\{x,y\}}
\left|
G_k\big(W(P,\mathbf u_0)\big)(p)-G_k(Y_{\mathrm{lum}})(p)
\right|
}.
\]

- `L_abs`의 gradient는 `W(P,u_0)→u_0→A_φ` 경로로 **Aligner만** 업데이트한다. Teacher U-Net은 이 loss 경로에 없다.
- 이 loss는 **cross-modal absolute-gradient discrepancy**다. PAN과 RGB GT의 spectral response가 다르므로 실제 물리적 displacement GT나 반드시 0이 되는 정합 정확도 지표로 해석하지 않는다.
- 원래 PANDA 논문의 loss가 아니라 본 연구에서 추가한 **SIPSA-inspired** absolute spatial supervision이다. SIPSA-Net 공개 코드에서는 `sobelfilter2XY`라는 이름의 실제 **인접 픽셀 2D 차분**을 쓰며, 본 변경안은 명시적으로 **Scharr**로 정의한다.

---

## 6. RR ε-consistency — 최대 반경 3 px

### 6.1 목적과 고정 조건

**[확정]** Synthetic shift는 **PAN만** 이동시켜 relative response를 학습시키며, `M`, `L_A`, `Y`, GT reference 좌표계는 고정한다. 하나의 sample마다 한 개의 global 2D ε를 뽑아 모든 spatial pixel에 동일하게 적용한다. 학습 update마다 새로 샘플링하고, flip/rotation augmentation **이후** PAN에 적용하며 누적하지 않는다. 전용 난수 generator를 사용한다.

\[
\boxed{R_\epsilon=3.0\ \text{RR PAN pixels}.}
\]

이 반경은 **합성으로 주입하는 이동 크기의 최대값**일 뿐, Aligner가 출력할 수 있는 `u`의 제한값은 아니다. CNN 출력에 `tanh`, `clamp(±3)`, `round`를 넣지 않는다.

**이력의 정확한 해석:** 옛 `R=2`는 WV3 FR 입력으로부터 구한 추정 P95 약 1.98 px를 참고했고 QB FR 추정 P95 약 2.98 px는 포괄하지 못했다. 다만 train의 PAN↔GT와 FR의 PAN↔LMS 통계는 동일한 기하 정답이 아니다. **R=3**은 사용자가 이번에 선택한 **새 학습 설정**이며 센서별 최대 정합량을 측정해 그대로 입력한 값은 아니다. 이전 실험의 FR benchmark 통계를 언급한 사실을 지우고 새 반경을 train-only 근거로 포장하지 않는다.

### 6.2 ε sampling — 분포 변경은 아직 미확정

**이번 기술서의 구체적인 baseline은 이전 합의에서 사용하던 원판 면적 균등을 그대로 유지한다.** 변경된 값은 `R=2→3`이다.

\[
a,b\overset{iid}{\sim}U(0,1),\quad
r=3\sqrt{a},\quad
\theta=2\pi b,\quad
\boxed{\boldsymbol\epsilon=(r\sin\theta,r\cos\theta)=(\epsilon_y,\epsilon_x)}.
\]

반경 3에서 원판 면적 균등이면 `Pr(||ε||≤1)=1/9≈11.1%`, `Pr(||ε||≤2)=4/9≈44.4%`, `2<||ε||≤3`가 `5/9≈55.6%`다. **작은 subpixel shift의 비중은 기존 R=2보다 감소**한다. 이 사실은 반드시 진단해야 한다.

지난 논의에서 `r=3a`인 **반경 균등 sampling**이 작은 이동의 비중을 유지하기 위한 **대안**으로 제안되었으나, 사용자가 명시적으로 동의한 것은 **최대 반경 3 px**다. 따라서 `r=3a`를 확정안에 조용히 혼합하지 않는다. 분포를 바꿀 때는 별도 방법론 변경 및 manifest 기록이 필요하다. **반경 균등, 혼합 분포, 센서별 반경 등은 이번 문서의 확정 구현값이 아니다.**

### 6.3 ε 생성·forward·loss

Native correction:

\[
\boxed{\mathbf u_0=A_\phi(P,L_A)}.
\]

Synthetic PAN:

\[
\boxed{P_\epsilon=W(P,\boldsymbol\epsilon)},\qquad\text{PAN only}.
\]

Shifted correction:

\[
\boxed{\mathbf u_\epsilon=A_\phi(P_\epsilon,L_A)}.
\]

Consistency loss:

\[
\boxed{
L_\epsilon^{RR}
=\operatorname{mean}_{b,j\in\{y,x\}}
\left|u_{\epsilon,b,j}+\epsilon_{b,j}-\operatorname{sg}(u_{0,b,j})\right|
}.
\]

- Native `u_0`는 `L_rec` 및 `L_abs`를 통해 정상적으로 gradient를 받는다. **Stop-gradient는 오직 ε loss에서 비교하는 anchor 값에 적용**한다.
- `P_ε` 생성은 고정된 synthetic perturbation이며 ε 자체는 parameter가 아니다. `P_ε`에 teacher reconstruction loss를 추가하지 않는다.
- 기존 `u=b+Δu`였을 때와 달리 ε-relative response를 만드는 주체는 **전적으로 CNN Aligner 자체**다.
- `L_ε`는 **상대적 translation response만** 지도하고 native PAN–MS의 절대 정답 displacement는 제공하지 않는다.

### 6.4 Interpolation artifact 및 유효 영역

`P_ε`는 bicubic warp와 border padding으로 생성된다. ε가 커질수록 경계 단서 및 interpolation artifact 문제가 커질 수 있고, 기존 실험에서 CNN의 ε 반응에 interpolation-induced blur와 연관된 성분이 관찰됐다.

**[검증 필요 / 구현 필수]** Aligner가 경계 extrapolation 패턴만으로 ε를 추측하지 않도록 native/ε branch의 **같은 고정 spatial view/유효영역 규칙**을 사용한다. 구체적인 crop margin은 입력 크기, 최대 ε=3, convolution의 실제 유효 receptive field, bicubic 4-tap support, 두 stride-2 layer 등을 근거로 결정하고 manifest에 기록해야 한다. **근거 없는 margin 숫자를 이 문서에서 새 hyperparameter로 확정하지 않는다.**

`L_abs`에서도 warped PAN의 유효 support와 Scharr 경계 처리 규칙을 명시한다. 마스크를 predicted shift에 따라 임의로 줄여 loss를 회피하지 않도록 같은 비교 domain을 유지하고, support violation을 통계/오류로 기록한다. 마스킹과 gradient 처리의 실제 구현은 수치 검증을 거쳐 확정한다.

### 6.5 매 update 수행 및 loss weight

- **[유지]** 매 optimizer update에서 native loss와 함께 ε branch를 계산한다.
- **[유지/초기 구현값]** `λ_ε=1e-3`은 이전 합의에서 제안된 초기 수치다. 새 prior-free + R3 아키텍처의 검증된 최적값으로 주장하지 않는다.
- **[유지/초기 구현값]** `λ_abs=1` (식의 앞 계수 1). Scharr 고정 계수/영상 DN normalization에 따라 `L_abs` 크기가 달라질 수 있으므로 loss/gradient magnitude는 기록한다.
- `λ_ε`에 어떤 값을 쓰든 ε branch가 **실제로 CNN Aligner의 gradient를 생성**해야 한다. 이전 EdgeMatch 모델의 ε gain≈1은 대부분 matcher 자체 반응이었으므로 이번 직접 회귀 결과의 예측 근거로 쓰지 않는다.

---

## 7. Stage 1 objective와 parameter update

\[
\boxed{
L_{S1}=L_{\mathrm{rec}}^{RR}+L_{\mathrm{abs}}^{RR}+\lambda_\epsilon L_\epsilon^{RR}
}.
\]

| 손실 | Teacher U-Net `θ` | PAN Aligner `φ` |
|---|---|---|
| `L_rec` | **gradient 있음** | **gradient 있음** (differentiable PAN warp 경유) |
| `L_abs` | 없음 | **gradient 있음** (Scharr 및 PAN warp 경유) |
| `L_ε` | 없음 | **gradient 있음** (shifted Aligner branch; native anchor stop-grad) |

- Aligner와 Teacher U-Net을 **동시에 최적화**한다. `L_abs`, `L_ε` 때문에 Teacher U-Net에 gradient가 새로 흘러가서는 안 된다.
- Native `u_0`는 `L_rec+L_abs`의 gradient를 받으며 ε anchor에서만 detach한다.
- `L_ε^FR` 항은 **없다**. 학습 가능한 native-FR train split이 없으므로 FR evaluation PAN/MS로 Aligner의 weight를 업데이트하지 않는다.
- 이 문서는 Stage 1에 한정한다. Stage 2 optimizer, Student branch, e/q weighting, KD는 구현 명세 범위 밖이다.

---

## 8. RR→FR inference 계약과 남는 한계

\[
\begin{aligned}
\mathbf u^{FR}&=A_\phi(P^{FR},L_A^{FR}),\\
\widetilde P^{FR}&=W(P^{FR},\mathbf u^{FR}),\\
\hat Y^{FR}&=U_{\mathrm{bicubic}}(M^{FR})
+F_{T,\theta}(\widetilde P^{FR},U_{\mathrm{bicubic}}(M^{FR})).
\end{aligned}
\]

- **[확정]** FR에서 **추론만** 한다. FR PAN/MS의 실제 원해상도 4:1 입력 비율 유지. CNN은 PAN 512×512에서도 2D shift 하나를 출력한다.
- FR shift의 의미는 **FR PAN pixel**이며, RR에서 학습했다고 **4를 곱하지 않는다**.
- synthetic ε는 **훈련에서만** 사용하며, FR inference에는 주입하지 않는다.
- 공간해상도 차이, sensor PSF/MTF, MS interpolation phase, sample-wise global translation 가정 및 local parallax 등은 여전히 남을 수 있다.
- **중요:** RR에서 알려진 ε에 잘 반응해도 FR의 absolute registration이 개선된다는 보장은 없다. FR에는 물리적 displacement GT가 없다. HQNR 및 D_s, D_λ는 최종 복원 평가이며 기하 정합의 절대 정답 지표가 아니다.

---

## 9. 실행용 의사코드 — 경로 및 detach 확인

```python
# P: [B,1,H,W], M:[B,C,H//4,W//4], Y:[B,C,H,W] (RR train only)
# L_A: verified PAN-grid native LMS, B_M: independent bicubic(M)
# A_phi: dual-stem direct (dy,dx) regressor, fc1(64,32), fc2(32,2) zero-init
# W: differentiable bicubic PAN warp with (dy,dx) sampling-coordinate convention
# scharr_abs_xy: [B,1,H,W] -> [B,2,H,W] (abs(dx),abs(dy)), no extra normalization
# rgb_mean_by_sensor: actual per-sensor RGB band indices

u0 = aligner(P, L_A)               # [B,2], DIRECT total correction; NO EdgeMatch
P_aligned = warp_pan(P, u0)        # original P, one warp
B_M = bicubic_upsample(M)          # shape [B,C,H,W]
Yhat = B_M + teacher_unet(P_aligned, B_M)  # adapt signature to existing U
L_rec = (Yhat - Y).abs().mean()
L_abs = (scharr_abs_xy(P_aligned) -
         scharr_abs_xy(rgb_mean_by_sensor(Y))).abs().mean()

# Dedicated CPU generator; sample AFTER spatial augmentations.
with torch.no_grad():
    eps = uniform_disk_area(batch=P.shape[0], radius=3.0, generator=rng).to(P.device)
    P_eps = warp_pan(P, eps)

# Native/shifted Aligner use exactly the SAME verified LMS reference and crop policy.
u_eps = aligner(P_eps, L_A)       # [B,2]
L_eps = (u_eps + eps - u0.detach()).abs().mean()

L_total = L_rec + L_abs + 1e-3 * L_eps
L_total.backward()
# teacher params: L_rec gradient only
# aligner params: L_rec + L_abs + L_eps gradient
```

**주의:** pseudocode는 parameter routing을 표현하며 실행 가능한 전체 training script가 아니다. Boundary validity/cropping, AMP 제어, RNG checkpoint, optimizer configuration, normalization 및 기존 Teacher input layout은 server implementation에 맞춰 검증한다.

---

## 10. 구현 확인 및 품질 진단 체크리스트 — 실험 설계가 아님

### 10.1 입력·모델

- [ ] PAN/LRMS 4:1 spatial shape, 저장 `lms` 및 reconstruction bicubic base를 구분한다.
- [ ] 모든 `b_s`, `b_global`, `b_i`, offline EdgeMatch, prewarp, `α/β`, head 66차원, ε branch의 matcher 재실행 코드를 완전히 제거한다.
- [ ] Aligner `forward(P,L_A)`가 `[B,2]` 직접 total `(dy,dx)`를 출력한다.
- [ ] `fc2` zero-init에 의해 `u_init=0`; 최종 native warp는 P 원본에서 한 번만 수행한다.
- [ ] 공통 ResBlock 구현과 Aligner stem/joint GN 유지, Teacher U-Net 본체 무변경을 확인한다.
- [ ] 이전 Edge-Guided checkpoint/optimizer state의 묵시적 재사용 금지. 새 모델 정의 및 학습 초기화 근거를 manifest에 남긴다.

### 10.2 Scharr/absolute supervision

- [ ] Scharr 축별 absolute 2채널, 고정 계수 1/32, padding·계산 precision을 검사한다.
- [ ] RGB 평균 **후** Scharr이며, 센서별 RGB indices가 맞다.
- [ ] Scharr edge map에 추가 통계적 정규화나 Otsu/NCC가 없다.
- [ ] `L_abs` gradient는 Aligner에 흐르고, Teacher U-Net에는 흐르지 않는다.
- [ ] `L_abs`가 물리 displacement GT가 아니라 **RR 구조 대리 loss**임을 지표/문서에 명시한다.

### 10.3 ε

- [ ] `R_ε=3` RR PAN px; norm(ε)≤3, sample마다 전역 벡터 1개, augmentation 후 샘플링, 독립 RNG 및 resume 상태 보존.
- [ ] **분포는 원판 면적 균등 `r=3sqrt(u)`**; 명시적 후속 승인 없이 `r=3u`로 변경하지 않는다.
- [ ] PAN만 shifted, `L_A/M/Y`는 고정, native anchor에 `sg`만, shifted branch에 reconstruction 없음.
- [ ] 모든 ε·native forward에서 고정된 동일 interior/support 규칙; border interpolation 단서 점검.
- [ ] Gradient path가 실제로 CNN Aligner에 도달하는지 및 `L_rec`, `L_abs`, `L_ε`의 규모를 분리 기록한다.
- [ ] ε 크기 0.25/0.5/1/2/3 PAN px에서 synthetic-shift response를 분리 점검한다. 이는 **진단**이지 별도 training loss나 실험 case를 새로 정하는 것이 아니다.

### 10.4 FR

- [ ] FR 평가 자료로 optimizer update, ε 학습, per-image GT label-fitting을 하지 않는다.
- [ ] FR 결과의 `(dy,dx)`는 **FR PAN px**; ×4, 별도 matching prior, evaluation GT 종속 preprocess 없음.
- [ ] 입력 크기가 달라질 때 GAP 출력·예측 correction 분포·ε-response의 일반화 가능성을 확인하되, 이를 FR absolute displacement 정답으로 오해하지 않는다.

---

## 11. 출처 및 측정과 설계의 구분

### 기존 소스에서 확인한 부분

- `59391_PANDA_PAN_Alignment_Awar.pdf` §3.2 및 Fig. 4: Aligner의 직접 global 2D translation 회귀, `L_rec`를 통한 공동 학습, PAN-only ε translation consistency. **본 문서의 `L_abs`와 LN 공통 ResBlock 변경은 원래 PANDA 원문의 명세가 아니다.**
- 기존 Stage 1 문서 `PANDA_Stage1_EdgeGuided_Residual_Aligner_Final_2026-10-13.md`: Scharr RGB 평균 후 loss 정의, verified LMS와 reconstruction bicubic reference 분리, 공통 LN ResBlock 결정. **해당 문서의 EdgeMatch pathway는 폐기한다.**
- `PAN_Epsilon_Offset_Method_2026-10-13.md`: 기존 `R=2` 원판 면적 균등, warp convention, stop-grad, 주기/계수 및 scale-response 한계. 이번 `R=3`은 **새 결정**이다.
- SIPSA-Net 공식 `helperTF.py` / `main_train.py`: RGB 평균 후 2D absolute-gradient 간 L1 loss 원리. 공식 `sobelfilter2XY()`의 실제 커널은 인접 픽셀 차분이며, **Scharr 선택은 이번 연구의 구현 결정**이다.
- 기존 구현 규약: `pa/warp.py`의 bicubic·border·align_corners=False, `pa/aligner.py`의 dual-stem/GCNN/GAP/FC/znorm, `model/pancrafter_paper.py`의 공통 LN ResBlock. **기존 코드가 자동으로 본 문서와 일치한다는 의미가 아니며 현재 Stage 1 구현과 대조해야 한다.**

### 이번에 변경하고 아직 성능으로 검증되지 않은 부분

- EdgeMatch와 모든 `b` prior 제거, 사전 정합 PAN 입력 제거.
- CNN이 `(dy,dx)`를 직접 회귀하도록 다시 정의; 공통 LN ResBlock 및 Scharr spatial supervision은 유지.
- `R_ε=3.0`으로 확대; 원판 면적 균등은 이전 방식 유지, `λ_ε=1e-3`은 초기 수치이며 새 모델의 최적값이라는 근거 없음.
- FR에서 높은 정합·복원 품질을 낸다는 결과는 **실측 전까지 주장하지 않는다**.
