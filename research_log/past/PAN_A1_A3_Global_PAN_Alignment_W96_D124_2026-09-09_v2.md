# PAN–MS 앞단 정합 실험: A1–A3 실행 명세 v2
## 원본·정합 반영 평가 및 dual-best checkpoint 개정본

**최초 작성일:** 2026-09-09  
**개정일:** 2026-09-09  
**문서 버전:** v2 — 학습 구성 유지, 평가·checkpoint·로그 명세 확장  
**기준 모델:** W96–D124 U-Net · MS+PAN 입력 · HRMS 단일 복원  
**대상:** 우선 WV3, 8-band MS + 1-band PAN = 9ch  
**상태:** B0 완료와 A1–A3 진행은 사용자 확인·승인 사항이다. v1의 모듈·loss·서버 배치를 유지하고, 이후 대화에서 승인된 원본/정합 반영 평가와 dual-best 저장 방식을 통합했다. 실제 서버 실행 또는 신규 evaluator 검증을 완료한 보고서는 아니다.
**개정 기준:** 첨부 v1 실행 명세와 직전 대화의 평가 확장 합의. 이번 개정에서 새로운 Sheet 결과를 조회하거나 B0의 미확인 설정을 추정하지 않았다.

> **핵심 구성**  
> 입력 PAN과 upsampled MS를 보고 경량 CNN이 전역 변위 `(dy, dx)`를 예측한다. PAN만 그 변위로 sampling한 뒤 기존 W96–D124 U-Net에 넣는다. MS base·GT·최종 HRMS는 원래 MS 기준으로 유지한다.  
> A1–A3의 forward와 파라미터 수는 같고 **학습 loss만 다르다**.  
> 서버별 case 전담 대신, **각 서버에서 A1·A2·A3를 모두 실행**한다. 한 서버 안에서는 같은 seed와 같은 초기 가중치로 대응 비교한다.  
> **같은 checkpoint의 같은 HRMS 예측**에 대해 `Raw-original`, `Raw-valid`, `Aligned-valid`를 모두 계산한다. `best_raw`를 주 비교 모델로, `best_aligned`를 정합 반영 탐색 모델로 각각 저장하고 `last`도 보존한다.

---

## 0. 결정의 우선순위와 이번 실험 범위

### 0.1 사용자 고정 결정

| 항목 | 적용 |
|---|---|
| 복원 backbone | **width=96, depth=[1,2,4]** |
| 입력 | **MS+PAN만 사용**; WV3 9ch |
| PAN reconstruction / MARs PAN task | **없음** |
| LPAN / HPAN 입력 | **없음** |
| 기본 task | HRMS reconstruction |
| 이번 실행 | **A1, A2, A3** |
| B0 | 완료된 기준 실험을 재사용 |
| 평가 확장 | 원본 PAN 기준 및 예측 shift·유효 영역을 반영한 기준을 함께 기록 |
| Checkpoint | `best_raw`·`best_aligned`·`last`를 분리 저장 |
| KD / uncertainty / GT variance | 후속 방향; 이번 캠페인에서 제외 |

### 0.2 이번 실행안에서 추가로 고정할 사항

- PAN 앞단의 **global translation 정합 모듈 하나**를 사용한다. Dense flow·affine·rotation·scale은 이번 비교에서 제외한다.
- 학습 가능한 정합량은 **관측 입력에서 예측**한다. 무작위 PAN/MS jitter, jitter 반경 `a`, 작은 patch의 shift cache label은 사용하지 않는다.
- B-2 보조 복원, clean/jitter 두 U-Net forward, EMA Teacher, KD를 결합하지 않는다.
- 기본 MS upsampling, intensity scaling, 데이터 split, augmentation은 완료된 **새 B0의 실제 코드·config**를 유지한다.
- MS residual base와 GT는 움직이지 않는다. PAN만 MS target 좌표로 맞추도록 학습한다.
- GT는 loss에만 사용한다. 정합 모듈 및 U-Net의 forward 입력으로 넣지 않는다.
- **원본 평가 경로를 보존하면서 정합 반영 평가 경로를 추가**한다. v1의 “HQNR 정합 반영 평가 보류”는 이번 개정으로 대체한다.
- 원본 평가에서는 기존 PAN reference와 B0 evaluator를 유지한다. 정합 반영 평가에서는 모델이 실제 사용한 `P̃`를 PAN reference로 재사용하고 유효 영역만 평가한다.
- `Raw-valid`를 추가하여 영역 변경과 PAN reference 변경의 영향을 분리한다.
- 학습 `L_rec`에 예측 shift 기반 moving mask를 새로 적용하지 않는다. **평가 mask와 학습 loss 영역은 별개**이다.
- 주 checkpoint는 `best_raw`, 보조 탐색 checkpoint는 `best_aligned`이다. 정합 반영 HQNR을 원본 HQNR과 같은 benchmark 숫자로 혼합하지 않는다.

기존 W168–D123/11ch/dual-MARs 문서는 역사적 근거이다. 그 문서의 `고정`, `유지`를 새 모델에 자동 적용하지 않는다. [S1–S3]

### 0.3 이번 실험이 답할 수 있는 것

**A1:** 복원 loss만으로 앞단 정합이 유효하게 학습되는가?  
**A2:** 최종 MS 출력의 경계 supervision을 추가하면 도움이 되는가?  
**A3:** 보정된 PAN과 GT의 구조를 직접 비교해 정합 모듈을 감독하는 것이 도움이 되는가?

이 실험만으로 센서의 물리적 정합 오차가 완전히 제거됐다고 주장하지 않는다. 결과가 좋아지는 원인은 정합·PAN 활용도·보간 효과·regularization 중 하나 또는 여러 가지일 수 있다.


### 0.4 v1 → v2 변경 범위

| 항목 | v1 | v2 |
|---|---|---|
| Aligner/U-Net/학습 loss | A1 recon / A2 recon+edge / A3 recon+geometry | **그대로 유지** |
| 원본 평가 | B0 evaluator만 유지 | `Raw-original`로 보존 |
| 영역 대조 | 선택적 diagnostic | `Raw-valid` 필수 |
| 정합 반영 평가 | 보류 | `Aligned-valid` 필수 |
| PAN reference | 원래 P만 | 원래 P와 실제 forward의 P̃를 구분 |
| 경계 | 기존 평가 영역 | 원본 영역 + 고정된 공통 유효 영역 |
| Best 저장 | 하나의 best + final | **best_raw + best_aligned + last** |
| 결과 해석 | 정합 진단과 복원 성능 | 이에 더해 영역·reference·checkpoint 선택 효과를 분리 |

§10의 ROI·mask 전파·checkpoint 적격성·저장 형식은 합의된 평가 원칙을 실행 가능하게 풀어 쓴 명세이다. B0 evaluator의 실제 필터 support와 scene ID처럼 원문에 없는 값은 `REQUIRED_FROM_B0` 또는 사전 확정 필드로 남긴다. 숫자를 임의로 보충하지 않는다.

---

## 1. 데이터 관계와 기준 좌표

기호를 다음과 같이 고정한다.

| 기호 | 의미 | WV3 train shape 예시 |
|---|---|---|
| `P` | 현재 학습 pair의 PAN | `[B,1,64,64]` |
| `S` | 현재 학습 pair의 LRMS | `[B,8,16,16]` |
| `M=U₄(S)` | B0의 기존 bicubic-upsampled MS | `[B,8,64,64]` |
| `Y` | HRMS GT | `[B,8,64,64]` |
| `Δ̂` | PAN sampling 위치의 전역 보정량 | `[B,2]`, `(dy,dx)` |
| `P̃` | 정합 모듈로 sampling한 PAN | `[B,1,64,64]` |
| `Ŷ` | 최종 HRMS | `[B,8,64,64]` |

위 shape는 기존 audit의 train 크기에 근거한 예시이다. 실제 B0에서 크기가 바뀌었다면 그 값을 사용하고 기록한다. [S2 §I.0, §I.5]

RR에서는 LRMS가 GT에서 합성되므로 둘의 생성 관계가 일관적이다. PAN은 별도 모달리티이며 해상도가 같다고 자동 정합되는 것은 아니다. 다만 audit은 RR의 작은 정합 측정값이 spectral 차이로 인한 측정 하한에 가깝다고 경고했다. **모든 RR sample에 큰 실제 shift가 있다는 전제는 두지 않는다.** [S2 §II.2, §II.8.5]

또한 B0 bicubic과 데이터셋 제공 `lms`의 phase 차이는 실제 PAN–MS 이동과 구분한다. 이번 캠페인 중 upsampler를 변경하지 않는다. 알려진 interpolation phase는 별도 diagnostic에 기록한다. `lms`를 정합 모듈 입력이나 residual base로 몰래 대체하지 않는다. [S2 §II.4; S3 §1.1]

---

## 2. A1–A3 공통 파이프라인

```text
원래 LRMS S ── B0 bicubic ×4 ── M ───────────────────────────────┐
                               │                                │
PAN P ─────────────────────┐    │                                │
                          ▼    ▼                                │
                   GlobalAligner Aφ                             │
                       Δ̂ = (dy,dx)                              │
                          │                                     │
PAN P ── differentiable sampling W(P,Δ̂) ── P̃                   │
                                             │                 │
                                             └─ concat(P̃,M)    │
                                                    │          │
                                           U-Net Fθ, W96–D124   │
                                                    │          │
                                               residual R      │
                                                    └── + M ◀──┘
                                                         │
                                                      최종 Ŷ
```

\[
M=U_4(S),\qquad \hat\Delta=A_\phi(P,M)
\]

\[
\widetilde P=\mathcal W(P,\hat\Delta)
\]

\[
\boxed{\hat Y=M+F_\theta(\operatorname{Concat}(\widetilde P,M))}
\]

- **모달리티 입력은 P와 S뿐이다.** Aligner 내부에서 M을 사용하는 것은 S를 기존 B0 경로로 확대한 표현을 사용하는 것이다.
- Warp 대상은 **PAN 영상 1채널**이다. MS condition, MS base, GT를 옮기지 않는다.
- U-Net은 기존 B0를 그대로 사용한다. 정합 모듈 때문에 depth나 width를 바꾸지 않는다.
- 출력에는 inverse warp를 붙이지 않는다.
- PAN에 대한 새로운 reconstruction output/head는 없다.

Localization CNN → grid → sampler → downstream task라는 미분 가능한 구성은 STN 원리를 따른다. 아래 channel 수와 loss는 본 실험을 위한 설계안이며, STN 논문의 그대로인 구조나 검증된 최적 구성이 아니다. [R1, R2]

---

## 3. 정합 모듈 권고안: dual-stem 경량 global shift CNN

### 3.1 왜 작은 CNN과 두 개의 출력인가?

처음부터 두 번째 U-Net이나 dense flow를 사용하면 PAN 정보를 변형하는 자유도와 학습 비용이 함께 커진다. 지금은 **앞단의 간단한 전역 위치 조정과 loss의 효과**를 분리해야 한다.

따라서 다음을 첫 구현으로 선택한다.

> **PAN/MS 별도 얕은 stem → 공간 위치를 유지한 결합 → joint convolution → global average pooling → 2개 변위 회귀**

이 설계는 표현의 단순성과 해석 가능성을 우선한 제안이다. Local displacement까지 설명할 수 있다는 뜻은 아니다. Global 이후 local 구조가 필요한지는 후속 실험이다. Audit은 global 보정의 효과와 일부 장면의 local residual을 모두 보고했다. [S2 §II.5–7]

### 3.2 Aligner만 사용하는 입력 정규화

입력 `P`, `M`은 먼저 B0의 intensity scaling을 적용한 tensor이다. 그 뒤 **aligner에 들어가는 복사본만** sample별·band별 공간 평균과 분산으로 표준화한다.

\[
Z(I_c)=\frac{I_c-\mu_c}{\sqrt{v_c+10^{-6}}}
\]

`μ`, `v`는 같은 영상의 공간축에서 계산한다. MS 8개 band를 평균하지 않는다. 표준화용 통계에는 gradient를 주지 않는다.

**실제 warp와 U-Net에는 B0 intensity scale의 원래 P와 M을 사용**한다. Z-normalized PAN을 최종 U-Net 입력으로 넘기지 않는다. 이 분리는 radiometric 복원 경로를 바꾸지 않으면서 aligner의 밝기 규모 의존성을 줄이기 위한 설계이다.

### 3.3 Layer 명세

| 순서 | 연산 | 출력 shape: train 64×64 기준 |
|---|---|---|
| PAN stem | Conv 3×3, `1→16`, stride1, reflect pad1 → GN(4 groups) → SiLU | `[B,16,64,64]` |
| MS stem | Conv 3×3, `8→16`, stride1, reflect pad1 → GN(4 groups) → SiLU | `[B,16,64,64]` |
| 결합 | PAN/MS stem을 채널 concat | `[B,32,64,64]` |
| Joint stage 1 | Conv 3×3, `32→32`, stride2, reflect pad1 → GN(8) → SiLU | `[B,32,32,32]` |
| Joint stage 2 | Conv 3×3, `32→64`, stride2, reflect pad1 → GN(8) → SiLU | `[B,64,16,16]` |
| Joint residual | Conv 3×3 `64→64` → GN(8) → SiLU → Conv 3×3 `64→64` → GN(8) → skip add → SiLU | `[B,64,16,16]` |
| 공간 집계 | Global average pooling | `[B,64]` |
| MLP | Linear `64→32` → SiLU | `[B,32]` |
| Shift head | Linear `32→2` | `[B,2]`: `(dy,dx)` |

모든 convolution은 `bias=False`, GN은 affine=True이다. Linear는 bias=True이다. PAN/MS stem의 가중치는 서로 공유하지 않는다. Aligner에는 BatchNorm, dropout, attention, fixed candidate offset bank, confidence gate를 넣지 않는다.

**공간 결합 전에 PAN과 MS를 각각 global pooling하면 안 된다.** 서로 다른 위치의 대응을 읽을 수 있도록 두 stem의 feature를 같은 위치에서 먼저 결합하고 joint convolution을 수행한 뒤 pooling한다.

이 명세의 WV3 aligner 파라미터 수는 **105,330개 = 약 0.1053M**이다. v1에서 reference implementation을 CPU로 구성해 `sum(p.numel())`로 계산한 기록이다. 이번 문서 개정에서 재측정한 것은 아니다. 기존 B0 backbone의 실제 파라미터 수는 별도로 읽어 합산한다. 과거 11ch 모델의 파라미터 수를 새 B0 수치로 대신하지 않는다.

### 3.4 변위 표현

- `Δ̂[:,0]=dy`, `Δ̂[:,1]=dx`.
- 단위는 **현재 입력 PAN과 같은 HR pixel**이다.
- Sample마다 하나의 전역 벡터이다. 픽셀마다 다르거나 MS band마다 다른 field가 아니다.
- Linear head의 출력을 HR pixel 값으로 직접 읽는다. `H`, `W`를 곱해 displacement 크기를 키우지 않는다.
- 마지막 출력에 `tanh × a`나 정해진 25개 후보의 soft-argmax를 사용하지 않는다.
- 따라서 **jitter 반경 a가 이 방법에 존재하지 않는다.** 학습률·loss 계수·모듈 width·경계 안전 규칙 등 다른 설계 선택까지 사라진다는 뜻은 아니다.

Image size가 64에서 512로 바뀌어도 `dy=0.5`는 현재 영상 격자의 반 픽셀이다. GAP가 variable shape 처리를 가능하게 하지만, 큰 scene의 변위 추정이 잘 일반화된다는 보장은 아니다. Full-scene과 tile별 추정의 차이를 진단한다.

### 3.5 초기화

- Stem·joint encoder·첫 Linear는 seed로 정한 일반 초기화를 사용한다.
- **마지막 Linear `32→2`의 weight와 bias만 0**으로 초기화한다.
- 시작 시 모든 sample에서 `Δ̂=(0,0)`이므로 앞단은 identity이다.
- 첫 backward에서는 마지막 Linear에 gradient가 생기고, 그 weight가 바뀐 다음부터 앞 layer에도 gradient가 전달된다. 첫 step의 stem gradient가 0이라는 이유만으로 버그로 판정하지 않는다.
- Aligner 전체의 weight를 모두 0으로 만들지 않는다.

Identity initialization은 STN reference에서 쓰는 원칙과 연결되지만, 본 모델은 affine 6개 값이 아니라 translation 2개만 출력한다. [R2]

### 3.6 왜 correlation volume은 이번 첫 버전에서 제외하는가?

Correlation 기반 정합은 이후 후보로 남긴다. 지금 넣으면 offset 탐색 범위, 격자 간격, softmax temperature, descriptor별 정규화까지 추가되어 loss 비교와 모듈 설계 비교가 섞인다. 이번 9개 run에서는 **동일한 작은 shift CNN만 사용**한다.

이 CNN이 알려진 변위를 따라가지 못하거나 변위가 거의 상수에 고정된다면, A1–A3의 실패를 정합이라는 연구 방향 전체의 기각으로 읽지 않는다. 그때는 correlation 기반 모델을 별도 버전으로 시험한다.

---

## 4. PAN warp: 단위·부호·gradient 규약

### 4.1 정의

\[
\widetilde P[b,y,x]=P[b,y+\hat d_y,x+\hat d_x]
\]

이것은 **sampling 위치**의 부호이다. 화면에 보이는 물체 내용의 이동 방향과는 반대이다.

`align_corners=False`에서 output pixel `(y,x)`가 사용하는 normalized grid는 다음과 같다.

\[
g_x=2\frac{x+0.5+\hat d_x}{W}-1,
\qquad
 g_y=2\frac{y+0.5+\hat d_y}{H}-1
\]

`grid_sample` 마지막 축의 순서는 `(gx,gy)`임에 주의한다. Aligner 출력 `(dy,dx)`와 순서가 다르다. [R3]

```python
pan_aligned = F.grid_sample(
    pan_fp32,
    grid_fp32,
    mode="bicubic",
    padding_mode="border",
    align_corners=False,
)
```

PAN은 한 번만 warp한다. `P` 자체를 보간하며, HPAN을 만들거나 down-up PAN을 U-Net 입력으로 추가하지 않는다.

### 4.2 이번 구현에서 가장 중요한 금지 사항

```python
# trainable aligner에서 사용 금지
if torch.all(delta == 0):
    return pan
```

**0 초기화된 Δ에서 위처럼 sampler를 우회하면, loss가 Δ에 의존하지 않게 되어 aligner의 gradient가 끊긴다.** 과거 frozen/random jitter 구현의 zero-shift shortcut을 trainable 앞단에 그대로 복사하면 안 된다.

학습 중에는 Δ=0이어도 sampler를 항상 실행한다. `delta.detach()`, `.item()`으로 새 grid를 만들기, NumPy/scipy warp 역시 금지한다. GT loss가 grid까지 미분되어야 한다.

### 4.3 정밀도와 재현성

Aligner, grid, warp, edge/geometry loss는 FP32로 계산한다. B0 U-Net이 AMP를 사용했다면 그 정책만 유지한다. 실제 trainer에서는 해당 계산을 `with torch.autocast(device_type="cuda", enabled=False):` 블록으로 감싼다. Tensor에 `.float()`만 호출해도 상위 autocast가 convolution을 다시 저정밀로 바꿀 수 있으므로 명시적으로 분리한다. Bicubic overshoot가 생길 수 있으므로 training 중 PAN/HRMS 출력을 clamp해 gradient를 지우지 않는다. 내보내기와 metric clipping은 B0 evaluator 규칙을 따른다. [R3]

PyTorch는 platform/version 간 완전한 재현을 보장하지 않으며, CUDA `grid_sample` backward는 비결정적일 수 있다. 같은 seed만으로 서버 간 동치를 가정하지 않는다. 실제 설치된 버전의 deterministic 동작을 smoke에서 확인한다. [R3, R4]

---

## 5. Loss를 case별로 정확하게 정의한다

### 5.1 공통 reconstruction: B0와 같은 영역을 비교한다

\[
L_{\mathrm{rec}}=\frac{1}{BCHW}\sum_{b,c,y,x}|\hat Y-Y|
\]

**B0와 같은 full-image 평균 L1을 유지**하는 것이 이 실행안의 기본이다. PAN sampling의 바깥 부분은 border padding을 사용하지만, GT와 MS base 자체가 사라진 것은 아니다. 모델이 displacement를 키워 어려운 GT 영역을 reconstruction loss에서 제거할 수 없도록 `L_rec`의 mask를 Δ에 따라 바꾸지 않는다.

완료된 B0가 다른 loss 영역을 사용했다면 정확히 같은 규칙을 복사한다. 새로운 moving mask를 A-case에만 적용한 결과를 B0와 직접 비교하지 않는다.

### 5.2 A2의 출력–GT edge loss

MS 각 band에 동일한 Scharr 미분 필터를 적용한다.

\[
K_x=\frac{1}{32}
\begin{bmatrix}
-3&0&3\\-10&0&10\\-3&0&3
\end{bmatrix},\qquad K_y=K_x^T
\]

\[
L_{\mathrm{edge}}=
\frac12\operatorname{mean}_{V_e,c}|K_x*\hat Y_c-K_x*Y_c|
+
\frac12\operatorname{mean}_{V_e,c}|K_y*\hat Y_c-K_y*Y_c|
\]

`Ve`는 미분 경계 1픽셀만 제외한 고정 영역이다. 입력 convolution은 reflect padding을 사용한다.

- 같은 spectral band끼리 비교하므로 gradient의 **부호를 유지**한다.
- MS를 평균해서 1채널로 만들지 않는다.
- 이 loss는 U-Net과 aligner 모두로 미분된다.
- 출력의 위치·detail을 감독하지만, 앞단 aligner가 정확한 위치를 배웠음을 독립적으로 보장하지는 않는다.

### 5.3 A3의 보정 PAN–GT 직접 구조 loss

PAN과 MS GT는 spectral response가 다르므로 밝기 L1을 직접 적용하지 않는다. Aligner를 직접 감독하기 위해 **대비 반전에 불변인 normalized gradient outer-product 표현**을 비교한다.

아래는 본 실행안의 정확한 수식이다. NGF 계열의 normalized gradient 비교에서 동기를 얻었지만, 특정 NGF 논문의 식을 그대로 구현한 것으로 주장하지 않는다. [R6]

#### (a) Loss용 공통 저주파 비교

\[
I_Y=\frac1C\sum_cY_c
\]

PAN 원본에서 계산한 detached 평균·표준편차로 `P`와 `P̃`를 동일하게 표준화한다. `IY`는 자체 통계로 표준화한다. `P̃`의 통계를 다시 계산해 normalization 자체가 변위를 따라 바뀌지 않게 한다.

두 영상을 공통 Gaussian `G_r`로 평활화한다.

\[
\sigma_c=r/2,\quad R_c=\lceil3\sigma_c\rceil
\]

WV3 r=4이면 sigma=2 HR px, radius=6, 13×13 separable kernel이다. **이 필터는 loss용 공통 smoothing이라는 설계 선택이다. 실제 PAN/MS MTF가 정확하게 일치한다는 주장은 아니다.** B0의 데이터 생성·upsampling을 교체하지 않으며, 이 영상은 U-Net이나 aligner 입력 채널에 추가되지 않는다.

\[
g_P=\nabla G_r(Z_P(\widetilde P)),\qquad
 g_Y=\nabla G_r(Z(I_Y))
\]

`∇`는 위 Scharr이다. 비교를 LR 16×16까지 decimate하지 않고, **현재 HR 격자를 유지**한다. 작은 train patch에서 위치 단서를 과도하게 줄이지 않으려는 설계이다.

#### (b) 낮은 gradient의 분모 안정화

PAN은 **unwarped P**의 loss용 gradient energy, GT는 GT의 energy로 sample별 척도를 정한다.

\[
\eta_P^2=\operatorname{mean}_{V_g}\|\nabla G_r(Z_P(P))\|^2+10^{-12}
\]

\[
\eta_Y^2=\operatorname{mean}_{V_g}\|g_Y\|^2+10^{-12}
\]

이 척도들은 detach한다. Warp한 PAN의 energy로 척도를 재추정하지 않는다.

#### (c) Sign-invariant 구조 표현

\[
Q(g;\eta^2)=
\frac{(g_x^2,\sqrt2g_xg_y,g_y^2)}{g_x^2+g_y^2+\eta^2}
\]

이는 정규화된 2×2 gradient outer product를 3개 독립 원소로 표현한 것이다. `Q(g)=Q(-g)`이므로 PAN과 MS에서 경계의 밝기 부호가 반대여도 같은 구조로 취급한다. `√2`는 대칭 off-diagonal 두 원소의 Frobenius norm을 맞춘다.

GT 기반의 soft edge weight는 다음과 같다.

\[
w_Y=\operatorname{sg}\left(\frac{\|g_Y\|^2}{\|g_Y\|^2+\eta_Y^2}\right)
\]

\[
\boxed{
L_{\mathrm{geo}}=
\operatorname{mean}_b\frac{
\sum_{x\in V_g}w_Y(x)\|Q(g_P(x);\eta_P^2)-Q(g_Y(x);\eta_Y^2)\|_2^2
}{\sum_{x\in V_g}w_Y(x)+10^{-8}}
}
\]

GT의 gradient가 없어 weight 합이 수치적으로 0인 sample은 geometry 항만 0으로 처리하고 그 수를 로그에 기록한다. `L_rec`에서는 sample을 제거하지 않는다. 예측된 confidence로 weight를 줄이는 head는 넣지 않는다.

#### (d) Gradient가 흐르는 위치

`L_geo`는 `P̃ → warp → Δ̂ → aligner`로 직접 전달된다. U-Net의 출력 feature를 사용하지 않으므로 **이 항 자체는 U-Net θ에 gradient를 주지 않는다**. `L_rec`는 θ와 φ 모두를 학습시킨다.

GT 평균과 정규화는 학습 loss에서만 사용한다. 추론 forward에는 GT 없이 P와 S만 있어야 한다.

#### (e) 의미와 한계

이 항은 PAN을 MS target 위치로 sampling하도록 감독한다. PAN을 생성하는 decoder나 PAN reconstruction target을 도입한 것이 아니다.

하지만 band 평균은 PAN의 정확한 spectral response가 아니며, smoothing이 그 차이를 모두 제거하지 않는다. Spectral mismatch, flat region, 반복 패턴, parallax 때문에 잘못된 gradient도 생길 수 있다. 따라서 **`L_geo` 감소 = 물리적 정합 성공**으로 해석하지 않는다. [S2 §II.8.5]

### 5.4 세 case — 누적 조합이 아님

| Case | 구조 | Loss |
|---|---|---|
| **A1** | Aligner + PAN warp + B0 U-Net | `L_rec` |
| **A2** | A1과 동일 | `L_rec + λE(t) L_edge` |
| **A3** | A1과 동일 | `L_rec + λG(t) L_geo` |

**A3에 A2의 edge loss를 함께 넣지 않는다.** 두 항의 결합은 A2/A3의 독립 결과가 나온 뒤 후속 ablation이다.

```text
lambda_edge(t) = 0.10 * min(t / 5000, 1)
lambda_geo(t)  = 0.01 * min(t / 5000, 1)
```

이 계수와 5K ramp는 이번 실행을 위한 **초기 제안값**이며 실험으로 최적화된 값이 아니다. 이전 J4 coefficient를 새 방법의 검증된 설정으로 승계한 것이 아니다. 모든 서버와 seed에 같은 값을 사용하고, 결과를 본 뒤 서버별로 바꾸지 않는다.

`λL_aux / L_rec`와 aligner에 대한 `||∇φ λL_aux|| / ||∇φ L_rec||`를 주기적으로 기록한다. 값만 보고 winner를 고르지 않는다. 계수 수정이 필요하면 새 version의 campaign으로 분리한다.

---

## 6. Border·mask·coverage: loss를 쉽게 만드는 우회로를 막는다

### 6.1 공통 원칙

- Reconstruction은 B0와 동일한 고정 domain을 사용한다.
- A2 edge loss는 고정 1px interior를 사용한다.
- A3 geometry loss는 blur, Scharr, bicubic 보간의 support를 고려한 **고정 interior Vg**를 사용한다.
- `Vg`가 model 예측에 따라 계속 줄어들도록 하지 않는다.
- **학습 loss에서는** shift 때문에 MS/GT reference를 이동시키거나 GT를 inverse warp하지 않는다. 평가용 PAN reference 변경과 유효 영역은 §10에서 별도로 정의한다.

### 6.2 Geometry support 제안

WV3 r=4의 geometry loss margin은 다음으로 둔다.

```text
Gaussian radius 6 + Scharr radius 1 + bicubic support allowance 2
+ alignment-support slack 2 = margin 11 HR px
```

64×64에서는 고정된 `[11:53,11:53]`, 즉 42×42 영역이다. **이 2px slack과 margin은 구현용 안전 설계값이며 무하이퍼파라미터라고 주장하지 않는다.** jitter의 sampling 반경이나 모델 출력의 clamp 값은 아니다.

매 batch에서 `Vg`의 모든 loss용 원본 sampling 이웃이 실영상 안에 들어오는지 확인한다. 현재 delta에서 support 조건이 깨지면 **그 sample의 loss를 조용히 0으로 만들거나 ROI를 줄이지 말고**, run을 `SUPPORT_FAIL`로 중단·기록한다. 모든 A-case에서 동일 guard를 적용해 허용 조건이 같게 한다. 입력이 작은 64px patch에서 이러한 문제가 반복되면, full-run 결과 판정 전에 support/변환 안전 설계를 별도 버전으로 수정한다.

이 guard 때문에 전역 이동이 작은 범위에 있어야 하는 operational constraint가 남는다는 점을 명시한다. 데이터의 최대 misalignment가 2px라는 뜻은 아니다. FR에서는 geometry training loss를 계산하지 않으며, full-frame prediction을 저장한다.

### 6.3 추가 로그

```text
mean / p95 / max abs(dx,dy), displacement norm
sampling-center in-bounds fraction
geometry-support-valid flag
GT edge-weight sum, geometry-zero-weight sample count
PAN before/after gradient energy
```

Mask 면적 축소로 **학습 loss**가 작아지지 않게 한다. 평가에서는 §10의 사전 고정된 유효 영역을 사용하며, output이 불리하다는 이유로 영역을 선택하거나 checkpoint마다 ROI를 축소하지 않는다. `Raw-original`은 B0의 원래 영역을 유지하고, `Raw-valid`와 `Aligned-valid`는 동일한 영역에서 비교한다. 이 평가 확장은 §5의 A1–A3 학습 loss를 바꾸지 않는다.

---

## 7. 학습 설정: B0와의 비교 가능성을 유지한다

### 7.1 기본은 from-scratch 대응 실험

A1/A2/A3는 같은 block seed에서 **동일한 초기 U-Net과 동일한 초기 aligner**로 각각 독립 학습한다. A1의 학습된 결과를 A2로 이어 학습하거나 A2의 결과를 A3로 이어 학습하지 않는다.

B0의 **학습 완료 checkpoint를 A1–A3 초기 가중치로 자동 사용하지 않는다.** 그렇게 하면 B0 대비 추가 학습량의 효과가 섞인다. Warm-start가 필요하다면 `B0 continuation` 대조군을 같은 추가 budget으로 별도 두는 새 캠페인으로 정의한다.

### 7.2 실제 B0 config가 source of truth

다음은 완료된 새 B0의 `config_resolved`, code commit, dataset manifest를 복사한다.

```text
normalization / MARs-free backbone implementation
input order and intensity normalization
train split / sample list / feeder / augmentation
batch and gradient accumulation
optimizer / betas / weight decay
learning rate / scheduler / update count
AMP / clipping / validation interval
original evaluator / raw checkpoint-selection manifest
평가 확장 설정은 위 원본 설정과 분리한 v2 evaluation manifest
```

**과거 11ch·dual-MARs의 batch, 50K, 1e-4를 새 B0의 확인된 값이라고 단정하지 않는다.** 실제 B0가 50K이면 A1–A3도 50K이다. Budget은 wall-clock 시간이 아니라 동일 optimizer update/sample 노출량으로 맞춘다.

Aligner는 B0 optimizer에 parameter group으로 추가하고, 첫 실행에서는 **U-Net과 같은 learning-rate/scheduler**를 사용한다. 새로운 pretrain·freeze·aligner 전용 LR sweep은 넣지 않는다. Auxiliary loss는 §5.4 ramp만 사용한다.

### 7.3 입력과 초기 가중치의 대응

한 block 안에서 A1–A3가 같은 초기 tensor를 사용하는지 hash로 검증한다. Aligner 생성 때문에 random-number 소비 순서가 바뀌어 U-Net 초기값이나 DataLoader 순서가 바뀌지 않게, initialization RNG와 data/augmentation RNG를 분리한다.

`init_unet_seed<seed>.pt`, `init_aligner_seed<seed>.pt`를 CPU에서 한 번 생성해 저장하고 세 case가 로드한다. B0 학습 시작 시의 tensor 또는 seed manifest가 있으면 그와도 대응시킨다.

### 7.4 이번 문서에서 확인하지 못한 B0 정보

B0 완료 자체는 사용자 확인이다. 이번 Google Sheet 읽기에서는 새 B0를 유일하게 식별하는 run ID, 서버별 seed, 실제 resolved training/evaluation config를 확보하지 못했다. 보이는 과거 `MS1_w96_9ch_msonly` 행은 mode conditioning 유지 기록이 있어 새 B0로 임의 간주하지 않는다.

**따라서 새 B0 manifest만 실행 전에 연결하면 된다.** 이 때문에 B0를 무조건 다시 돌리라는 뜻은 아니다. 여러 server×seed에 대응하는 B0가 이미 있으면 그대로 재사용한다.

---

## 8. 서버·seed 배치 권고

### 8.1 선택: 각 서버가 A1–A3를 모두 수행

**권고하지 않는 배치**

```text
s1: A1 × 3 seeds
s2: A2 × 3 seeds
s3: A3 × 3 seeds
```

이 경우 method와 server가 완전히 묶인다. 서버 자체의 training 효과가 있으면 A3–A1 차이가 방법 때문인지 환경 때문인지 분리하기 어렵다. 한 서버 내부에서 seed 반복이 늘어도 그 confounding은 사라지지 않는다.

**권고 배치**: `(server, seed)`를 block으로 두고 block 안에서 세 방법을 비교한다. 이는 통제 가능한 nuisance factor 안에서 treatment를 비교하는 blocking 원칙이다. [R5]

| 서버 | Block seed — 제안값 | 1순위 | 2순위 | 3순위 |
|---|---:|---|---|---|
| **s1** | **2025** | A1 | A2 | A3 |
| **s2** | **1234** | A2 | A3 | A1 |
| **s3** | **777** | A3 | A1 | A2 |

총 **9개 신규 full run**, case마다 3회, 서버마다 3회이다. 첫 wave에서 A1·A2·A3를 하나씩 확인하고 이후 회전시키는 순서이다. Order는 결과를 보기 전에 고정한다. `s1/s2/s3`의 이름만으로 실제 GPU나 CUDA version을 추정하지 않는다.

B0의 실제 seed set이 다르면, **실행 전** 표의 seed를 기존 B0 seed와 맞춰 대체하고 공통 manifest를 고정한다.

### 8.2 이 배치가 해결하는 것과 못 하는 것

- 같은 서버·같은 seed에서 A2–A1, A3–A1을 비교할 수 있다.
- 서버별 기본 수준과 sample order/초기값의 영향을 대응 비교에서 줄인다.
- 그러나 server와 seed는 여전히 block 안에서 묶여 있으므로 **server effect와 seed effect 자체를 별도로 추정할 수는 없다**.
- Case별 3개 결과의 SD는 **server–seed block 사이 변동**이다. 같은 GPU에서 얻은 순수 seed variance로 부르지 않는다.
- 최종 논문에서 seed 안정성을 별도 주장하려면 winner와 대조군을 같은 기준 서버에서 동일한 3 seeds로 추가 확인한다. 이번 9개 결과가 이미 있는 cell은 중복 실행하지 않아도 된다.

### 8.3 B0가 일부 block에만 있는 경우

- 대응되는 B0가 있는 block에서만 paired `Ak−B0`를 계산한다.
- 다른 서버의 B0 숫자 하나를 모든 A-run에서 빼서 3-seed improvement라고 쓰지 않는다.
- B0가 하나뿐이어도 A1–A3의 대응 비교는 진행할 수 있다. B0 대비 보편적인 향상 주장은 matching control을 보완한 뒤 한다.
- 같은 B0 checkpoint를 모든 서버에서 inference하는 것은 evaluator 동치 검증이다. **각 서버에서 B0를 학습한 대조군을 대체하지 않는다.**

### 8.4 소요 시간

새 aligner의 실제 iteration time을 아직 측정하지 않았다. 과거 W168/dual-MARs 시간을 환산해 확정하지 않는다.

각 서버에서 warm-up 제외 200 update를 측정하여 다음을 산출한다.

\[
T_{run}\approx t_{step}N_{update}+T_{eval}+T_{export}
\]

서버마다 같은 full-run 완료 기준을 사용한다. 빠른 GPU에서 batch를 임의로 키우거나 느린 서버만 학습 step을 줄이지 않는다. 서버당 하나의 학습 job을 권고하고, 정확한 wall-clock quota는 smoke 후 기록한다.

---

## 9. 실행 전 gate와 검증

### 9.1 코드·데이터·평가 동치

- 세 서버의 code commit, dependency/container, train/test 파일 hash 확인.
- 입력은 WV3 9ch, PAN reconstruction 호출 0회, LPAN/HPAN input concat 0회.
- 새 B0 forward를 wrapper의 **disabled aligner** 경로에서 동일 tensor로 재현.
- Aligner가 enabled되고 Δ=0일 때 값이 B0와 정밀도 범위에서 일치하는지 별도 확인. 이 경로는 반드시 gradient가 살아 있어야 한다.
- B0의 MARs mode-dependent parameter가 제거됐는지, 또는 실제 B0가 어떤 single-task 구현인지 확인. Backbone 정리는 이 비교 중간에 변경하지 않는다.

### 9.2 Warp unit test

```text
T01: dx/dy 순서, grid 마지막 축 x/y 순서
T02: 동일 global shift가 PAN의 모든 위치에 적용
T03: zero shift forward identity (float32 tolerance <=1e-5)
T04: zero shift에서도 d(loss)/d(delta) 존재
T05: impulse / ramp / known translation의 부호
T06: LR->HR 단위 변환이 필요할 때 정확히 ×4, 중복 적용 없음
T07: MS base와 GT가 warp되지 않음
T08: empty/invalid support를 조용히 제거하지 않음
T09: NaN/Inf shift를 fail 처리
T10: 64² / 256² / 512² / 직사각형 shape 처리
```

### 9.3 Loss/gradient unit test

```text
T11: A1에는 L_edge/L_geo가 역전파되지 않음
T12: A2 L_edge -> backbone, aligner 두 경로에 gradient
T13: A3 L_geo -> aligner; backbone에는 이 항의 직접 gradient 없음
T14: GT를 requires_grad=False로 고정
T15: 동일 영상의 edge loss 0
T16: Q(g) == Q(-g), sign reversal 검증
T17: synthetic aligned PAN의 geometry loss < uncorrected < wrong-sign
T18: 처음에는 fc2만 gradient, update 후 stem도 gradient
T19: 세 case 같은 seed의 초기 state_dict hash 일치
T20: train/val/inference 모두 aligner 입력은 P,M; GT 필요 없음
```

### 9.4 짧은 synthetic module 사전 검증

알려진 정수·소수점 shift를 가진 **같은 영상에서 만든 통제 pair**로 estimator가 input-dependent shift를 학습 가능한지 검사한다. 이 검사는 modality mismatch가 없는 구현 검증이지, PAN–MS 성능 검증이 아니다.

이 test로 학습된 aligner checkpoint는 실제 A1–A3에 사용하지 않고 버린다. Full run은 정의된 zero-head 초기화부터 시작한다. 검증에서 임의 shift를 가하더라도 이는 **training augmentation을 추가했다는 뜻이 아니다**.

### 9.5 실제 batch smoke

각 A-case에서 B0 실효 batch로 forward/backward/optimizer step과 validation/export를 수행한다. 특히 A3의 extra gradient maps 메모리를 측정한다. OOM 때 batch를 몰래 줄이지 않고 B0와 같은 effective batch를 유지하는 accumulation을 동일하게 적용한다.

Technical gate 실패는 method 성능 실패와 구분한다. 구현을 수정하면 version/commit을 올리고 세 case를 같은 수정본으로 시작한다. 낮은 metric을 이유로 특정 seed만 제외하거나 재추출하지 않는다.


### 9.6 v2 evaluator·selector gate — 신규 구현 후 실행

```text
E01: 같은 forward의 SR/delta/P_aligned를 세 view가 재사용함
E02: raw_original이 B0의 기존 evaluator 결과를 허용 정밀도에서 재현함
E03: 실제 delta=0, 동일 V에서 raw_valid == aligned_valid
E04: 고해상도/저해상도 PAN reference가 모두 같은 P 또는 P_aligned에서 생성됨
E05: aligned 경로가 원본 PAN의 stale low-resolution cache를 사용하지 않음
E06: 같은 SR/MS/V에서 PAN 미참조 D_lambda 경로의 값이 두 valid view에서 같음
E07: known integer/subpixel warp의 in-bounds 및 nominal bicubic support mask가 맞음
E08: border padding과 all-ones border warp를 실제 관측 유효성으로 오인하지 않음
E09: PAN filtering/decimation/window/block support까지 mask가 전파됨
E10: raw_valid와 aligned_valid의 HR/LR ROI·window·scene 집합이 같음
E11: 필터/decimation phase와 crop origin이 원본 grid 기준으로 유지됨
E12: RR GT/예측을 shift하지 않으며 동일 V의 GT 기반 RR metric은 reference PAN과 무관함
E13: invalid selection scene을 빼고 평균하지 않고 aligned 후보 전체를 INVALID로 처리함
E14: checkpoint마다 fixed ROI가 바뀌지 않고 protocol/ROI/scene hash가 유지됨
E15: raw/aligned running maximum·동률 후보·later-step 선택이 독립적으로 작동함
E16: 동률 연쇄 하락 방지, 과거 후보 재선택, resume 이후 선택 결과가 일치함
E17: 정확한 last 평가와 best_raw/best_aligned/last의 전체 view export가 동작함
E18: raw evaluation과 delta=0 재추론 ablation을 서로 다른 결과로 저장함
E19: v2 평가 mask가 학습 L_rec domain 또는 gradient에 연결되지 않음
E20: 전체 평가 후보가 무효이면 no_valid_candidate를 남기고 best_aligned를 위조하지 않음
```

실수 오차 tolerance는 B0 evaluator의 dtype·수치 정밀도를 확인해 검증 설정에 기록한다. 현재 문서에는 이 gate의 통과 결과가 없다. Pure evaluator 지원 문제로 실패한 aligned view를 방법 성능의 실패 점수로 변환하지 않는다.

---

## 10. 평가·mask·checkpoint: 원본과 정합 반영 버전을 함께 기록

### 10.1 평가 목표와 source of truth

이번 개정은 **원본 평가를 대체하는 것이 아니라, 별도의 정합 반영 평가를 추가**하는 것이다.

- **주 비교:** 원래 PAN·MS reference와 B0 evaluator를 사용한 `Raw-original`.
- **영역 대조:** 같은 원래 PAN을 공통 유효 영역에서 비교한 `Raw-valid`.
- **정합 반영 진단:** 실제 모델의 보정 PAN을 같은 유효 영역에서 비교한 `Aligned-valid`.
- **Checkpoint:** `best_raw`, `best_aligned`, `last`를 각각 보존하고 세 평가를 교차 보고한다.

원본 metric의 정의, intensity scale, clipping, MTF/필터, decimation phase, window/block 규칙은 B0의 실제 evaluator를 기준으로 한다. 정합 반영 버전에서는 **PAN reference와 명시한 평가 영역만** 바꾸며, 다른 metric 식을 몰래 도입하지 않는다.

과거 대화에는 FR 12–19와 mat20의 서로 다른 범위가 등장했다. 본 개정에서 어느 하나를 새로 선택하지 않는다. 실행 전에 B0에서 다음을 가져와 고정한다.

```text
selection_scene_ids, selection_data_manifest
report_scene_ids, report_data_manifest
raw_metric_code_hash, evaluator_config
intensity_and_clipping_policy
PAN_low_resolution_reference 생성 경로와 source hash
MTF/filter/window/block support, decimation phase
```

`best_raw`와 `best_aligned`는 **동일한 selection scene set과 평가 주기**를 사용한다. Report용 전체 장면 집합이 다르면 별도 split ID로 기록한다. 최종 보고 전용 장면에서 별도로 checkpoint를 고르지 않는다. B0가 benchmark test subset을 selection에 사용한 이력은 그대로 명시하며, 이를 독립 hold-out 검증으로 부르지 않는다.

### 10.2 한 번의 forward에서 예측·변위·보정 PAN을 함께 확보

```text
inputs P,S
-> M = B0_upsample(S)
-> delta = Aφ(P,M)
-> P_aligned = W(P,delta)
-> SR = M + Fθ(cat[P_aligned,M])
-> cache {SR, delta, P_aligned, source-coordinate/support information}
-> 같은 SR로 세 evaluation view를 계산
```

평가는 `eval()`·`no_grad`에서 수행하지만, 모델의 learned alignment는 켜 둔다. 정합 반영 reference는 **이 forward에서 U-Net에 실제 공급한 `P_aligned`**이다.

- 평가용으로 delta를 재추정하지 않는다.
- metric이 가장 좋아지는 shift를 추가 탐색하지 않는다.
- PAN을 두 번 warp하거나, 반대 부호로 다시 움직이지 않는다.
- MS·GT·SR은 이동시키지 않는다. Output inverse warp도 없다.
- Aligner의 HR pixel 출력을 다시 4배 곱하지 않는다.
- B0 intensity domain에서 생성된 `P_aligned`를 evaluator domain으로 변환할 때 원본 PAN과 같은 radiometric 규칙을 사용한다. 별도 min–max stretching이나 reference별 histogram matching을 추가하지 않는다.
- Tiling을 쓰는 실제 B0 inference가 있다면 이를 명시한다. 이번 모듈은 scene별 global shift 하나를 전제로 하므로 tile별 서로 다른 shift를 자동 도입하지 않는다.

**`Raw-original`은 정합 모듈을 끄는 실험이 아니다.** 모델이 정합을 사용해 만든 같은 SR을 원래 PAN reference로 평가한다. `delta=0` 재추론은 §11의 별도 ablation이다.

B0에는 aligner가 없으므로 비교 기록에서 `delta=(0,0)`, `P_aligned=P`로 둔다. 다른 A-case의 shift를 B0의 주 평가 reference에 이식하지 않는다.

### 10.3 세 evaluation view

`Ω0`는 B0 evaluator의 원래 평가 영역이다. “original”은 전체 배열을 강제한다는 뜻이 아니라, **기존 border crop까지 포함한 B0 영역을 보존**한다는 뜻이다. `V`는 §10.6에서 고정하는 공통 유효 영역이다.

| View ID | PAN reference | PAN 저해상도 reference | 영역 | 용도 |
|---|---|---|---|---|
| `raw_original` | 원래 P | B0의 원래 경로 | Ω0 | 기존 B0와 주 비교 |
| `raw_valid` | 원래 P | 같은 원본 경로 | V | 경계·영역 변경 대조 |
| `aligned_valid` | 실제 forward의 P̃ | **P̃에서 동일 evaluator 경로로 재생성** | **동일한 V** | 모델 정합 관계를 반영한 평가 |

세 view의 SR tensor와 MS reference는 동일하다. `raw_valid`와 `aligned_valid`의 mask/ROI, window/block 집합, 장면별 집계 가중치도 동일하게 둔다.

동일 checkpoint 안에서 다음 차이를 기록한다.

\[
\Delta Q_{\mathrm{region}}=Q_{\mathrm{raw\_valid}}-Q_{\mathrm{raw\_original}}
\]

\[
\Delta Q_{\mathrm{reference}}=Q_{\mathrm{aligned\_valid}}-Q_{\mathrm{raw\_valid}}
\]

첫 차이는 평가 영역의 변경 효과, 두 번째는 PAN reference 변경 효과이다. **두 번째 차이를 모델의 순수 성능 향상량이나 실제 misalignment error라고 부르지 않는다.** 보간과 reference 변화가 포함된 평가상 차이이다.

`Aligned-valid HQNR`은 모델 의존 정합 reference를 사용한 **별도 변형 지표**이다. 원본 HQNR 열에 덮어쓰거나, 기존 논문 보고 HQNR과 동일 조건으로 비교하지 않는다.

### 10.4 HQNR의 PAN 경로를 일관되게 변경

정합 반영 평가에서는 PAN을 사용하는 모든 evaluator 입력을 같은 reference에서 만든다.

\[
P_{\mathrm{ref}}^{\mathrm{aligned}}=\widetilde P,
\qquad P_{\mathrm{low,ref}}^{\mathrm{aligned}}=\mathcal D_{\mathrm{eval}}(\widetilde P)
\]

`D_eval`은 B0 evaluator가 사용하는 저역통과·해상도 처리 연산의 표기이다. 실제 구현이 downsampling 뒤 upsampling된 PAN을 받는다면 **그 전체 기존 경로**를 같은 방식으로 적용한다. 특정 필터·sigma·phase를 이 문서에서 추정해서 대입하지 않는다.

고해상도 비교에는 P̃, 저해상도 비교에는 원래 P의 cache를 쓰는 혼합을 금지한다. 원본 PAN의 cached low-resolution reference는 `aligned_valid`에 재사용할 수 없다. 캐시를 만들면 checkpoint·scene·delta·PAN source·evaluator hash를 key에 포함한다.

**Identity gate:** 원본 P를 기존 evaluator 경로로 처리했을 때 B0가 실제 사용한 PAN 저해상도 reference와 일치하는지 검증한다. 원본 evaluator가 제공 배열을 읽는다면 그 생성 경로를 확인한다. 일치하지 않는데 aligned view에서만 새 필터로 바꾸지 않는다. 이 gate가 실패하면 aligned evaluator를 미검증으로 남기고 원본 경로는 보존한다.

MS spectral reference, 제공 `lms` 사용 여부, SR 저해상도 투영 규칙은 B0 그대로이다. **Bicubic MS base를 evaluator의 제공 `lms`와 임의 교환하지 않는다.**

PAN을 참조하지 않는 기존 Dλ 경로라는 전제하에, 같은 SR·MS·V를 쓰는 다음 값은 수치적으로 같아야 한다.

\[
D_\lambda^{\mathrm{raw\_valid}}=D_\lambda^{\mathrm{aligned\_valid}}
\]

이는 evaluator 의존성 검사와 unit test로 확인한다. 다르다면 mask, spectral input, cache 또는 metric 경로가 함께 바뀌었는지 조사한다. Dλ 경로가 실제로 PAN에 의존하는 구현이면 이를 조용히 수정하지 말고 의존성을 명시한 뒤 명세를 검토한다.

장면별 HQNR은 각 view에서 독립적으로 계산한다.

\[
HQNR_i^{(v)}=(1-D_{\lambda,i}^{(v)})(1-D_{s,i}^{(v)})
\]

\[
HQNR^{(v)}=\frac{1}{N}\sum_{i=1}^{N}HQNR_i^{(v)}
\]

`(1-mean Dλ)(1-mean Ds)`로 대체하지 않는다. 장면 크기나 유효 픽셀 수로 scene 평균의 가중치를 바꾸지 않는다. Distortion 자체의 집계 방식은 B0 evaluator를 유지한다. [S3 §7]

fSCC도 각 view에 해당하는 PAN reference와 영역에서 계산한다. SR의 luminance/high-frequency 표현 정의는 B0와 동일하게 유지한다.

### 10.5 Mask: 복제된 경계값은 실제 관측이 아니다

`padding_mode=border`로 값이 채워졌더라도, 그 값이 실제 영상 밖에서 관측된 것은 아니다. 평가 mask는 **영상값이 아니라 sampling 좌표와 연산 support**로 만든다.

\[
q_y=y+\hat d_y,\qquad q_x=x+\hat d_x
\]

`Vwarp(y,x)`는 그 위치의 bicubic 보간에 필요한 원본 이웃이 영상 안에 있는지를 표시한다. 보수적인 구현은 각 축에서 `floor(q)-1, floor(q), floor(q)+1, floor(q)+2`의 nominal 4-tap support를 검사한다. 정확한 정수 위치에서 가중치 0인 tap도 제외하지 않는 보수적 규칙을 선택했다면 모든 case에 같은 규칙을 적용한다.

**실제 좌표를 검증할 때 border clamp 이전의 좌표를 사용**한다. Clamp된 sampling 위치를 검사하면 영상 밖 값도 전부 유효하다고 잘못 판정할 수 있다.

Mask는 이후 연산까지 전파한다.

```text
PAN warp support
  -> evaluator PAN filtering support
  -> 저해상도 decimation/upsampling support
  -> Q/SSIM/SCC 등의 window 또는 block support
  -> 유효한 공통 통계 집계 영역
```

공통으로 적용할 원칙:

- 무효 픽셀을 0으로 곱한 영상을 일반 metric에 넣지 않는다.
- 검은 픽셀이나 `P̃==0` 여부로 mask를 만들지 않는다.
- All-ones 영상을 `border`로 warp한 결과만으로 mask를 만들지 않는다. Border 복제로 영상 밖도 1이 될 수 있다.
- Metric 중심 픽셀만 유효하고 window의 나머지 부분이 무효라면 그 window를 사용하지 않는다.
- HR/LR 영역은 **같은 물리 영역**을 가리키게 대응시킨다. Mask를 단순 resize하여 가중 평균하지 않는다.
- Filtering·decimation은 가능한 한 원래 full-frame 좌표에서 먼저 수행한다. Crop 이후 좌표 원점이 바뀌어 phase가 달라지지 않게 한다.
- Crop형 adapter를 사용해야 하면 경계 halo와 HR/LR phase, metric block 시작 위치까지 보존하고 그 동치성을 검증한다.

기존 evaluator가 pixel mask를 받지 않는다면, 함수에 임의 mask를 넘겼다고 완료로 간주하지 않는다. 유효한 window/block 선택 또는 검증된 crop adapter를 구현하고, 그 adapter의 version/hash를 기록한다. 정확한 support와 mask reduction은 기존 evaluator code 확인이 필요한 구현 항목이다. [S2 §III.1–3]

### 10.6 Checkpoint용 V는 사전에 고정하고, 실제 support는 매번 검사

**Checkpoint마다 예측된 shift에 맞춰 ROI를 줄인 점수로 best를 선택하지 않는다.** 평가 면적 감소로 어려운 영역을 없애는 선택 효과를 방지하기 위해 다음 규칙을 사용한다.

#### (a) Selection용 고정 공통 영역

첫 selection 평가 전에 `selection_roi_manifest`를 저장한다. Scene별 fixed interior rectangle 또는 동치인 고정 유효 window/block 집합을 담는다. 이 영역은 모든 case·seed·checkpoint에서 공통으로 사용한다.

고정할 내용:

```text
scene_id, H/W, ratio, HR/LR ROI coordinates
original evaluator crop
warp support rule
PAN/MS filtering and window/block support
decimation phase and coordinate origin
유효 window/block index 목록 또는 재생성 규칙
ROI 결정 규칙과 hash
```

Interior margin은 evaluator support와 사전에 선언한 평가 안전 영역으로 정한다. **원문에는 실제 evaluator kernel/window의 전체 support가 없으므로 픽셀 수는 여기서 임의 지정하지 않는다.** 실행 전에 B0 evaluator에서 확인해 고정한다. §6의 training geometry margin 11px를 evaluation margin으로 자동 복사하지 않는다.

이 평가 안전 영역은 무작위 jitter 반경도, 실제 센서 misalignment의 최대값 주장도, 모델 output clamp도 아니다. ROI 결정은 unit/synthetic support 검사에 근거하며 **높은 HQNR을 얻는 crop을 찾아 선택하지 않는다.**

#### (b) 매 checkpoint의 적격성 검사

실제 Δ와 모든 metric support를 적용했을 때 고정 V의 비교가 유효한지 장면별로 확인한다.

- 선택 장면 **전부**에서 고정 V가 유효해야 `aligned_valid` 평균이 checkpoint 선택 후보가 된다.
- 하나라도 support 실패·NaN·정의 불가능 metric이 있으면 그 checkpoint의 aligned selection을 `INVALID`로 기록한다.
- 실패 장면만 빼고 더 작은 N으로 평균하지 않는다.
- ROI를 줄이거나 shift를 clamp해서 점수를 복구하지 않는다.
- SR과 원본 metric이 유한하고 B0 조건을 만족하면 `raw_original` 평가는 별도로 수행할 수 있다. 학습 중 `SUPPORT_FAIL`로 중단된 run은 정상 완주 run과 구분한다.
- Invalid가 반복되면 evaluator/support 설계를 새 protocol version으로 검토한다. 이미 얻은 점수에 맞춰 기존 manifest를 덮어쓰지 않는다.

Aligned 후보가 하나도 없으면 `best_aligned.pt`를 가짜로 만들지 않는다. `best_aligned_meta.json`에 `status=no_valid_candidate`와 실패 사유를 남긴다.

#### (c) Dynamic mask는 진단용으로만 보존

매 scene의 실제 최대 유효 영역과 coverage를 별도로 저장할 수 있다. 그러나 그 가변 영역의 metric은 `dynamic_valid_diagnostic`처럼 이름을 달리하고 **best selection에 사용하지 않는다.**

#### (d) 최종 비교의 공통 영역

원칙적으로 최종 비교도 같은 fixed V에서 수행한다. 추가로 정확한 공통 overlap 평가가 필요하면, **보고 대상 B0/A1/A2/A3의 best_raw·best_aligned·last 전체**에 대해 scene별 유효 영역의 교집합을 한 번 만든다.

이 보조 영역은 `report_common`이라는 별도 `roi_scope`로 저장하고 모든 보고 모델을 동일하게 재평가한다. 점수는 보지 않고 support만으로 교집합을 만든다. **사후 report_common 점수로 기존 best를 다시 고르지 않는다.** 영역이 달라진 값은 selection-fixed 결과와 섞지 않는다.

### 10.7 RR GT 지표와 학습 loss는 별도로 유지

RR에서 `Ŷ`와 `Y`는 모두 M-frame이다. 따라서 ERGAS·SAM·PSNR·SSIM·GT 기반 SCC/Q2n 등의 비교에서 GT를 shift하지 않는다.

- `RR-original`: B0와 동일한 영역·규칙.
- `RR-valid`: 동일한 공통 V에 대응한 GT 기반 평가.
- 같은 V에서 PAN reference만 바꿔도 **GT 기반 RR metric은 바뀌지 않아야 한다.** 중복 계산할 필요 없이 같은 값을 연결한다.
- PAN을 사용하는 fSCC 같은 별도 구조 지표는 해당 reference view를 명시한다.

이 평가 mask는 `L_rec`에 연결하지 않는다. **A1은 계속 B0와 같은 영역의 reconstruction loss만 사용**한다. A2의 고정 edge domain과 A3의 고정 geometry domain도 §5–6 그대로이다.

### 10.8 Best checkpoint 두 개와 last

| 파일 | 선택 기준 | 성격 |
|---|---|---|
| `best_raw.pt` | `raw_original` HQNR → 같은 view의 fSCC → later step | B0 대비 주 비교 |
| `best_aligned.pt` | 적격한 `aligned_valid`, fixed V의 HQNR → 같은 view의 fSCC → later step | 모델 정합 reference 반영 탐색용 |
| `last.pt` | 정확한 마지막 optimizer update | 동일 학습 budget의 결과 |

HQNR 동률 허용 범위는 `1e-4`를 사용한다. fSCC 동률 허용 범위도 기존 실행 규칙인 `1e-4`로 기록하되 B0 manifest와 차이가 있으면 실행 전에 명시한다. 기존 B0 selection과 달라지는 설정을 숨기지 않는다.

각 selection stream은 독립적으로 다음처럼 관리한다.

1. 지금까지 관측한 적격 checkpoint HQNR의 **running maximum**을 보존한다.
2. 그 maximum에서 `1e-4` 이내인 checkpoint만 HQNR 동률 후보로 남긴다.
3. 동률 후보 중 fSCC의 maximum에서 허용 오차 이내인 집합을 구한다.
4. 그 집합에서 가장 늦은 optimizer step을 선택한다.

현재 선택된 best와의 차이만 비교해 동률 교체가 누적되면서 HQNR 기준이 내려가지 않게 한다. Running max 상승 시 과거 후보 중 새로운 best가 될 수 있으므로 **동률 후보의 가중치와 메타데이터를 보존**한다. 후보 band에서 영구히 제외된 checkpoint만 정책에 따라 정리한다. 반올림한 Sheet 값이 아니라 저장된 원 정밀도의 metric을 사용한다.

Selection state에는 metric record뿐 아니라 `protocol_id`, ROI hash, scene manifest, eligible 상태를 포함하고 resume에서 검증한다. 원본과 aligned 기준의 running maximum을 공유하지 않는다.

정확한 최종 step도 세 view로 평가하고, 두 selector의 후보에 포함한 뒤 `last.pt`를 별도로 보존한다. 기존 코드가 `best.pt`를 요구하면 **명시적으로 best_raw의 alias**로만 사용하고, 어떤 기준의 파일인지 metadata에 기록한다.

### 10.9 왜 aligned best를 유일한 주 결과로 삼지 않는가?

\[
HQNR_{\mathrm{aligned},t}=\mathcal H(\hat Y_t,S,\mathcal W(P,\hat\Delta_t);V)
\]

V를 고정해도 checkpoint t에 따라 **예측 영상과 PAN reference가 함께 바뀐다.** 복원 향상과 reference 조정의 효과가 섞이므로 aligned score만으로 방법의 주 우열을 선언하지 않는다. 이는 본 평가 정의에서 따라오는 한계이며 `no_grad`만으로 해결되지 않는다.

따라서 `best_raw`를 주 비교로 유지하고, `best_aligned`는 원본 평가와 함께 해석한다. 이후 모델 독립적인 입력 정합 절차를 고정해 모든 방법이 동일한 reference를 쓰는 평가를 만들 수 있지만, **이번 개정에는 그 추가 estimator를 도입하지 않는다.**

### 10.10 모든 저장 모델의 교차 평가

동일 run에서 다음 표를 모두 채운다. 같은 step이 두 selector에 선택되면 계산 결과를 공유하되 두 selector의 선택 이력은 보존한다.

| Checkpoint | Raw-original | Raw-valid, fixed V | Aligned-valid, 동일 V | RR-original / RR-valid |
|---|---|---|---|---|
| best_raw | 필수 | 필수 | 필수 또는 명시적 INVALID | 필수 |
| best_aligned | 필수 | 필수 | 필수 | 필수 |
| last | 필수 | 필수 | 필수 또는 명시적 INVALID | 필수 |

`best_raw`에는 raw score만, `best_aligned`에는 aligned score만 골라 제시하는 보고를 금지한다. 표마다 `selection_kind`, `evaluation_view`, `roi_scope`, scene set을 표시한다.

B0도 같은 evaluator·공통 V로 재평가하되, 이는 완료된 B0 checkpoint의 **inference 재평가**이며 baseline 재학습 요구가 아니다. B0의 과거 checkpoint 이력이 없으면 새로운 기준으로 B0 best를 다시 선택했다고 주장하지 않는다.

### 10.11 평가 확장의 완료 조건

신규 evaluator의 실제 repo 구현, mask 전파, ROI manifest, dual selector는 아래 §9.6 gate를 통과한 뒤 사용한다. **본 Markdown 개정만으로 구현·검증이 끝난 것은 아니다.** 기존 reference ZIP의 evaluator/설정은 v1 기준이므로 v2 전체 평가를 지원한다고 간주하지 않는다.

---

## 11. 정합을 배웠는지 확인하는 진단

지표 개선과 실제 정합을 구분하기 위해 다음을 모든 A-case에서 동일하게 실시한다.

### 11.1 Aligner 사용 여부 대조

동일 trained checkpoint에서 `learned Δ`, `Δ=0`, `−Δ`의 세 inference를 저장한다. 이는 **서로 다른 SR을 만드는** diagnostic이다. 같은 SR에서 평가 reference만 바꾸는 §10의 세 view와 혼동하지 않는다. 원본 reference·공통 V 등 동일한 평가 조건에서 먼저 비교하고, 이 control로 얻은 최고 점수로 `best_raw`/`best_aligned` 결과를 교체하지 않는다.

- learned > zero는 정합 모듈의 기여 가능성을 지지한다.
- wrong-sign까지 같이 좋아지면 smoothing/보간 효과를 의심한다.
- learned와 zero가 같으면 U-Net이 PAN을 무시하거나 Δ가 사실상 0인지 확인한다.
- zero-warp interpolation 자체가 B0 입력을 바꾸지 않는지 먼저 검증한다.

### 11.2 추가된 shift에 대한 반응

기본 P와 별도로 `P_e=W(P,e)`를 만드는 diagnostic을 수행한다. 우리의 sampling 부호 규약에서는 원래 pair가 미정합이어도 이상적으로 다음 차분 관계를 기대한다.

\[
A_\phi(W(P,e),M)-A_\phi(P,M)\approx-e
\]

이는 원래 shift가 0이라는 잘못된 가정을 피한다. 미리 정한 통제 집합의 `e`에 대해 회귀 오차, 축별 부호, 기울기를 측정한다. 예: 각 축 `{-1,-0.5,0,0.5,1}` HR px의 조합. **이 범위는 diagnostic sweep이지 학습 jitter 반경이 아니다.**

관측 FR audit shift는 노이즈가 있는 참고 추정값이다. 기하 GT라고 부르지 않고, moving/reference와 부호가 다른 경우 정확히 변환한다. [S2]

### 11.3 별도의 구조 진단

A3의 training loss와 동일한 Q-loss만 평가하면 그 목적함수에 적응한 결과를 다시 확인하는 데 그친다. RR에서는 GT와의 band별 edge error·시각 overlay를, input alignment는 비교 해상도를 맞춘 Scharr-ZNCC와 독립 descriptor로 확인한다. Spectral 차이에 따른 측정 하한을 함께 기록한다. [S2 §II.8.5]

### 11.4 Full scene과 patch의 차이

학습 PAN64와 FR PAN512의 크기가 다르다. Global average pooling은 shape만 해결한다. 한 FR scene에 서로 다른 local displacement가 섞이면 global Δ 하나가 타협값이 될 수 있다.

Full-scene Δ와 고정 크기 tile들의 Δ 분포를 비교하되, main inference를 tile-flow로 바꾸지 않는다. Tile별 서로 다른 warp는 다음 캠페인의 local 모델이다.

### 11.5 A2의 주장 범위

A2만 좋아지면 **출력 edge loss의 일반적인 복원 효과**일 수도 있다. 정합 모듈과의 상호작용을 확정하려면 이후 `B0 + 같은 output edge loss, aligner 없음`을 matched seed로 추가해야 한다. 이 추가 control은 이번 승인된 A1–A3 9개 run에 자동 포함하지 않는다.

---

## 12. 결과 비교와 판정 규칙

### 12.1 대응 차이를 먼저 본다

Metric Q가 클수록 좋은 지표라면 block `b=(server,seed)`마다:

\[
d_b^{2-1}=Q(A2,b)-Q(A1,b),\qquad
 d_b^{3-1}=Q(A3,b)-Q(A1,b)
\]

그 뒤 세 block의 차이를 평균한다. 작을수록 좋은 지표는 개선의 부호를 별도로 표시한다. 서버 간 raw 절대값으로만 방법을 순위 매기지 않는다.

대응 비교는 **같은 `selection_kind`·`evaluation_view`·`roi_scope`·scene manifest**끼리 수행한다. 기본 주 비교는 `best_raw / raw_original`이고 `last / raw_original`도 함께 제시한다. `best_aligned` 결과와 모든 valid-view 결과는 별도 표로 구분한다. A1의 best_raw 원본 점수와 A3의 best_aligned 정합 반영 점수를 직접 빼지 않는다.

B0가 대응되는 block에서만 `Ak−B0`를 계산한다. Case별 raw mean±SD와 **paired difference 3개 원값**을 함께 저장한다. 3개 block만으로 통계적 유의성을 확정하거나 seed와 server 효과를 분리했다고 말하지 않는다.

### 12.2 최소 결과 표

**Run 요약 — 행마다 checkpoint 종류를 명시**

| server | seed | case | selection_kind | step | HQNR raw-original | HQNR raw-valid | HQNR aligned-valid | Δraw vs paired A1 | paired B0 유무 | RR-original / valid | median shift norm | fixed coverage | aligned eligible |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| s1 | 2025 | A1 | best_raw | | | | | 0 | | | | | |
| s1 | 2025 | A1 | best_aligned | | | | | | | | | | |
| s1 | 2025 | A1 | last | | | | | | | | | | |
| s1/s2/s3 | 해당 seed | A2/A3 및 대응 B0 | 각 선택 종류별 | | | | | | | | | | |

**Metric 상세 — long format**

```text
run_id, server, seed, checkpoint_id, selection_kind, step,
split_id, scene_set_id, scene_id, evaluation_view, roi_scope,
HQNR, D_lambda, D_s, fSCC,
RR_ERGAS, RR_SAM, RR_PSNR, RR_SSIM, RR_SCC, RR_Q2n,
dy_hr, dx_hr, delta_norm_hr,
valid_fraction, num_valid_windows_or_blocks,
metric_valid, selection_eligible, invalid_reason,
protocol_hash, roi_hash, prediction_hash, pan_reference_hash
```

Scene별 metric과 aggregate를 구분해 보존한다. Scene 수를 training-seed 반복 수로 세지 않는다. GT 기반 RR 값이 두 valid view에 공통이면 중복값임을 명시한다. 없는 GT 또는 미지원 metric은 `NA`이며 0으로 채우지 않는다.

### 12.3 해석 기준

| 관찰 | 우선 해석 |
|---|---|
| A1부터 개선, learned>zero, synthetic shift 반응 정상 | task loss만으로 front-end alignment가 유효할 가능성 |
| A2 개선, Δ 변화/정합 지표는 미약 | output edge regularization 기여 가능성; B0+edge 후속 필요 |
| A3 개선, 보정 PAN의 독립 정합 지표도 개선 | 직접 geometry supervision의 기여를 지지 |
| L_geo만 감소, 최종 metric/복원은 악화 | spectral proxy에 과도하게 맞춤 또는 task 충돌 |
| 세 case 모두 Δ≈상수, 입력 shift에 반응 안 함 | 고정 phase 보정 또는 bias만 학습했을 가능성 |
| 세 case 모두 실패 | loss, regressor capacity, train-FR distribution 차이 분리; 정합 자체를 즉시 기각하지 않음 |
| aligned-valid만 개선, raw-valid는 그대로/악화 | 평가 PAN reference 변경 효과 가능성; 독립 복원 향상으로 주장하지 않음 |
| raw-valid만 raw-original보다 개선 | 영역 변경 효과부터 분리; 정합 모듈의 효과로 환산하지 않음 |
| best_aligned의 aligned 점수 상승, raw/RR 품질 저하 | 모델 의존 reference에 유리한 checkpoint 선택 가능성 |
| mask/support 문제 반복 | 학습 support 실패와 aligned evaluator 부적격을 구분하고 사유 기록; 정상 결과처럼 혼합하지 않음 |

방향이 3개 block에서 일관적인지와 효과 크기를 우선 보고한다. 2/3이 양수라는 이유만으로 통계적 성공으로 확정하지 않는다. 유망 case는 matching B0와 같은 기준 서버의 추가 seed 검증으로 이어간다.

---

## 13. 필수 로그와 산출물

```text
work_dir/<run_id>/
  config_resolved.yaml
  baseline_manifest.json
  environment.json
  code_commit.txt
  dataset_hashes.json
  initialization_hashes.json
  train_log.jsonl
  gradient_diagnostics.jsonl
  evaluation_manifest_v2.json
  selection_roi_manifest.json
  report_common_roi_manifest.json          # 보조 공통 overlap 평가를 수행했을 때만
  metric_adapter_version.json
  checkpoint_metrics.csv
  scene_metrics.csv
  selector_state_raw.json
  selector_state_aligned.json
  best_raw.pt
  best_raw_meta.json
  best_aligned.pt                         # 적격 후보가 있을 때만
  best_aligned_meta.json
  last.pt
  last_meta.json
  candidates/{raw,aligned}/               # 동률 band의 재선택에 필요한 checkpoint
  delta_predictions.csv
  support_diagnostics.csv
  predictions/<checkpoint_id>/<split>/<scene>/
    sr                                      # canonical full-size M-frame prediction
    pan_aligned                             # 실제 forward PAN, evaluator 변환 명세 포함
    delta.json
    warp_support_mask
    evaluation_roi_metadata.json
  evaluation_views/{raw_original,raw_valid,aligned_valid}/
  overlays/
  controls/{zero_shift,wrong_sign,known_shift}/
```

`best.pt` 호환 alias가 필요하면 best_raw를 가리키게 한다. 파일을 세 개 보존한다는 것은 **한 checkpoint의 output frame을 세 개 만든다는 뜻이 아니다.** 각 모델의 SR은 M-frame full-size로 저장하고 metric view만 구분한다.

`baseline_manifest.json`에는 B0 run ID, checkpoint hash, init/seed, train config, 원본 evaluator와 selection scene ID를 포함한다. Environment에는 GPU 모델, PyTorch/CUDA/cuDNN, precision/TF32, deterministic flags, driver, batch/accumulation을 기록한다.

`evaluation_manifest_v2.json`에는 원본 evaluator hash와 확장 adapter hash를 분리하고 다음을 저장한다.

```text
protocol_id, original_protocol_id
selection/report scene manifests
intensity transform 및 clipping 순서
PAN low-resolution reference 생성 규칙, support, phase
metric window/block/grid origin
selection fixed ROI와 report_common ROI의 구분
mask propagation 및 invalid 처리 규칙
best_raw/best_aligned 선택 기준, 허용 오차
model_predicted_reference=true인 view 식별
```

Step 로그는 기존처럼 유지한다.

```text
loss_rec, loss_edge, loss_geo, lambda_edge, lambda_geo
lambda_aux * loss_aux / loss_rec
aligner_grad_from_rec, aligner_grad_from_aux
backbone_grad_norm, aligner_grad_norm
dy/dx mean std min max, delta_norm percentiles
sampling_valid_fraction, geometry_support_valid
PAN gradient energy before/after warp
train step time, peak memory
```

Checkpoint/scene 평가 로그를 추가한다.

```text
HQNR, D_lambda, D_s, fSCC × evaluation_view
raw/aligned selector eligibility와 선택 사유
fixed ROI coverage, dynamic support coverage, valid window/block 수
원본/정합 PAN high/low reference hash
SR hash, 실제 적용 delta, warp count
raw-valid minus raw-original, aligned-valid minus raw-valid
invalid scene IDs와 원인, 동일 N으로 평균했는지
평가 시간·메모리, reference 재사용 여부
```

A1에서도 edge/geo 진단값을 `no_grad`로 계산할 수 있으나, 그 항을 역전파해서는 안 된다. 같은 주기로 기록하고 추가 진단 비용은 training throughput과 분리한다. 추가 evaluator와 dual selector의 비용도 forward 학습 비용과 구분한다.

Resume에서 evaluation/ROI/scene/metric hash가 달라지면 기존 selector state를 이어 쓰지 않는다. 변경이 필요하면 새 protocol ID로 기록하고 모든 비교 결과를 해당 규칙으로 재평가한다. 원본 로그와 v1 상태를 덮어쓰지 않는다.

---

## 14. 실행 설정 예시

```yaml
campaign: PAN_GLOBAL_ALIGN_W96D124_v2
status: execution_spec_with_approved_evaluation_extension
baseline:
  run_id: REQUIRED_FROM_COMPLETED_B0
  config_policy: inherit_exact_training_and_raw_evaluation
  initialization: from_scratch_paired_with_seed
  use_trained_b0_as_initial_weights: false
model:
  backbone: existing_B0_UNet
  width: 96
  depth: [1, 2, 4]
  ms_bands: 8
  input_order: [PAN, upsampled_MS]
  input_channels: 9
  pan_reconstruction: false
  lpan_hpan_input: false
aligner:
  type: dual_stem_joint_CNN_global_translation
  pan_stem_channels: 16
  ms_stem_channels: 16
  joint_channels: [32, 64]
  residual_blocks_at_quarter_grid: 1
  mlp_hidden: 32
  output: [dy, dx]
  unit: current_HR_pixel
  head_init: zeros
  output_activation: linear
  offset_search_grid: null
  jitter: false
  training_zero_shift_bypass: false
warp:
  target: PAN_only
  mode: bicubic
  padding_mode: border
  align_corners: false
  compute_dtype: float32
loss:
  reconstruction: B0_full_domain_mean_L1
  output_edge: bandwise_signed_Scharr_mean_L1
  geometry: normalized_gradient_outer_product
  geometry_sigma_hr: ratio_div_2
  geometry_fixed_margin_hr: 11
  geometry_support_fail: stop_and_record_invalid
  ramp_steps: 5000
cases:
  A1: {lambda_edge: 0.0, lambda_geo: 0.0}
  A2: {lambda_edge: 0.1, lambda_geo: 0.0}
  A3: {lambda_edge: 0.0, lambda_geo: 0.01}
servers:
  s1: {seed: 2025, order: [A1, A2, A3]}
  s2: {seed: 1234, order: [A2, A3, A1]}
  s3: {seed: 777, order: [A3, A1, A2]}
evaluation:
  protocol_id: PAN_GLOBAL_ALIGN_EVAL_v2
  raw_protocol: REQUIRED_FROM_COMPLETED_B0
  selection_scene_ids: REQUIRED_FROM_COMPLETED_B0
  report_scene_ids: REQUIRED_FROM_COMPLETED_B0
  same_selection_scenes_for_raw_and_aligned: true
  same_evaluation_schedule_for_all_cases: true
  use_same_prediction_for_all_views: true
  ms_gt_output_warp: false
  test_time_shift_optimization: false
  reuse_forward_pan_aligned: true
  pan_low_reference_operator: REQUIRED_FROM_B0_EVALUATOR
  recompute_low_pan_from_aligned_pan: true
  views:
    raw_original:
      pan_reference: original_P
      domain: original_B0_domain
    raw_valid:
      pan_reference: original_P
      domain: selection_fixed_common_V
    aligned_valid:
      pan_reference: actual_forward_P_aligned
      domain: selection_fixed_common_V
      model_dependent_reference: true
      official_raw_benchmark_equivalent: false
  validity:
    rule: original_sampling_coordinates_and_full_operator_support
    zero_fill_then_metric: false
    propagate_filter_and_window_support: true
    fixed_roi_manifest: REQUIRED_BEFORE_SELECTION
    roi_numeric_bounds: REQUIRED_AFTER_EVALUATOR_SUPPORT_AUDIT
    metric_support_manifest: REQUIRED_FROM_B0_EVALUATOR
    shrink_roi_per_checkpoint: false
    clamp_predicted_shift_for_evaluation: false
    invalid_scene_policy: invalidate_aligned_selection_without_dropping_scene
    dynamic_support_metrics: diagnostic_only
    report_common_intersection: optional_separate_report_not_for_selection
  checkpoints:
    primary: best_raw
    save: [best_raw, best_aligned, last]
    best_raw_metric: raw_original.HQNR
    best_raw_tiebreak: raw_original.fSCC
    best_aligned_metric: aligned_valid.HQNR
    best_aligned_tiebreak: aligned_valid.fSCC
    best_aligned_requires_all_selection_scenes_valid: true
    aggregate: mean_of_scene_level_products
    hqnr_tolerance: 0.0001
    fscc_tolerance: 0.0001
    final_tiebreak: later_optimizer_step
    anchor: independent_running_maximum_per_stream
    retain_tie_band_candidates: true
    evaluate_exact_last_update: true
    no_valid_aligned_candidate: write_status_without_fake_checkpoint
    best_pt_compatibility_alias: best_raw
  reporting:
    cross_evaluate_all_saved_checkpoints: true
    rr_metrics: [original_domain, valid_domain]
    rr_gt_shift: false
    save_protocol_roi_prediction_reference_hashes: true
```

Seed는 B0 대응을 확인한 뒤 실행 전에 대체 가능하다. Loss·architecture를 결과를 보고 server별로 바꾸는 것은 허용하지 않는다.

Run ID 제안:

```text
PA_A1_REC_W96_D124_9CH_S2025_s1
PA_A2_OUTEDGE_W96_D124_9CH_S2025_s1
PA_A3_GEO_W96_D124_9CH_S2025_s1
```

다른 block에서는 seed/server만 바꾼다. 이전 W168의 A1, GA_C1 등과 이름이 겹치지 않게 `PA_` prefix를 사용한다.

---

## 15. Reference implementation과 확인한 범위 — v1 기록과 v2 미검증 사항

기존 v1 보조 ZIP의 `pan_global_aligner_reference.py`에는 다음 함수가 포함되어 있다. 이번 요청의 산출물은 **개정 Markdown**이며, 그 ZIP의 코드·YAML·CPU 검증 파일을 갱신한 것이 아니다.

```text
PANGlobalAligner
warp_pan
output_edge_loss
common_blur
orientation_tensor
direct_geometry_loss
```

**B0 U-Net, 데이터 로더, 실제 trainer, checkpoint evaluator와 연결된 완제품 코드는 아니다.** 실제 repository code에 통합하면서 §9의 full gate를 통과해야 한다. Fixed geometry support guard는 trainer에 구현해야 하며, reference loss 함수를 단독으로 호출해 그 조건을 우회하면 안 된다.

다음은 **v1 원문에 기록된 CPU reference checks 결과를 보존**한 것이다. 이번 v2 문서 개정에서 해당 test를 재실행한 결과가 아니다:

| 검사 | 결과 |
|---|---|
| Aligner param count, WV3 | **105,330** |
| Zero head 출력 | `(0,0)` |
| 64px FP32 zero warp의 최대 차이 | **0.0**, 이 로컬 test에 한정 |
| Known integer shift의 interior 회복 | 최대 차이 **0.0**, synthetic test |
| Synthetic structure loss | uncorrected **0.120923**, correct **0.000002587**, wrong-sign **0.296999** |
| Zero-head backward | 첫 step head gradient 존재, stem 0; 다음 step stem gradient 존재 |
| Shape | 64², 64×80, 256², 512² 통과 |
| Output edge identity / sign-invariant Q | 통과 |

v1의 기록 환경은 `torch 2.10.0+cpu`였다. **GPU 속도, CUDA backward 재현성, 실제 PanCollection 정합 성능, A1–A3 최종 품질은 해당 CPU 기록으로 검증되지 않는다.** 위 synthetic 결과는 구현 sanity check이지 연구 성능 결과가 아니다. 상세 수치는 기존 ZIP의 `reference_cpu_checks.json`에 있다.

**v2 신규 미검증 범위:** 원본/정합 PAN reference adapter, HR/LR mask support 전파, fixed ROI 평가, dual-best selector, 해당 evaluator의 B0 동치 및 실제 데이터 export. §9.6 테스트를 구현·실행하기 전에는 검증 완료로 표기하지 않는다. 기존 ZIP의 `campaign_spec.yaml`은 평가 확장을 반영하지 않으므로 §14에 맞춰 별도로 갱신해야 한다.

---

## 16. 실행 전 남은 연결 정보 — 미확인 값은 추정하지 않는다

1. 완료된 **새 B0 run ID와 resolved training/raw-evaluation manifest**를 연결한다.
2. B0의 실제 server×seed 집합을 확인하고 §8의 block seed를 대응시킨다.
3. B0 forward가 입력 9ch·PAN loss 없음·W96–D124임을 wrapper smoke로 확인한다.
4. B0와 같은 초기화·학습 budget, selection/report scene IDs를 고정한다.
5. B0 evaluator의 PAN high/low reference 생성 경로, filter/window/block support, decimation phase, dtype 및 clipping을 확인한다.
6. **Selection용 fixed common ROI manifest**를 unit/synthetic support 검사에 따라 사전 고정한다. Margin의 숫자는 evaluator 확인 없이 임의 기입하지 않는다.
7. 원본 evaluator 재현, zero-shift valid-view 동치, fixed ROI·mask 전파, 두 selector와 resume, 세 checkpoint의 cross-evaluation gate를 통과한다.
8. 실제 GPU에서 full gradient·training support·v2 export gate를 통과한다.

모듈 layer, loss 식, case 구분, 서버 배치는 v1과 동일하다. 변경 대상은 평가·선택·로그이다. 일부 evaluator 정보가 원문에 없다는 이유로 B0를 무조건 재학습하거나, 과거 W168/dual-MARs 값을 새 B0에 자동 적용하지 않는다.

---

## 17. 근거와 참고자료

### 사용자 결정 및 내부 자료

- **[S1]** `PAN_research_baseline_W96_D124_2026-09-09.md`: 새 W96–D124·MS+PAN·PAN reconstruction 제거 기준.
- **[S2]** `2026-09-03_alignment-audit-s2-detail.md`: §I 데이터·단위, §II.4 interpolation phase, §II.5 global/local 효과, §II.8.5 spectral 하한, §III 교정·부호 검증.
- **[S3]** `2026-09-04_global-alignment-plan-review (1).md`: 작은 train patch의 cache 신뢰도 한계, bicubic/lms 차이, gradient/평가 구현 이슈. 과거 사용한 interp23tap·dual frame 등은 이번 구조로 승계하지 않음.
- **[S4]** `s1_w168_d123_shift_robust_alignment_30h_plan.md`: 과거 random jitter의 zero-shift shortcut과 fixed radius의 역사적 정의. **이번 trainable Δ 경로에서는 그 shortcut을 사용하지 않음.**
- **[S5]** 현재 대화: B0 완료, A1–A3 진행 승인, 세 서버·seed 배치 질문, 상세 방법·정합 모듈 설계 요청.
- **[S6]** 이후 대화의 평가 개정 합의: A1은 reconstruction만 유지; 원본 PAN 및 예측 shift를 반영한 PAN+mask 평가를 함께 산출; Raw-valid 대조; best_raw·best_aligned·last 저장과 교차 평가. 사용자의 “좋아 반영해서 md 파일로 다시 만들어줘” 요청을 반영.
- **[S0]** 개정 원문 `PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09.md`, SHA256: `cd30889c1f37187d161c984b7446d6025dd47121c152841cda74de31816649c7`. v2는 원본 파일을 덮어쓰지 않고 별도 저장한다.
- **v1의 조회 이력:** Google Sheet `pan-cvpr27`의 메타데이터와 WV3 서버 탭에서 B0/W96/9ch 관련 행을 읽었으나 새 B0를 유일하게 확정하지 못함. 이번 v2에서는 Sheet를 재조회하지 않음. 이 문서에는 추정한 B0 성능을 기록하지 않음. 조회 경로: `https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit`.

### 외부 1차 자료 — v1의 참고 목록 보존

아래는 v1에서 사용한 설계 참고 목록이다. 이번 문서 개정에서 웹 재조사나 최신 API 검증을 수행한 것은 아니며, A1–A3 또는 v2 evaluator 성능 근거도 아니다.

- **[R1]** Jaderberg et al., *Spatial Transformer Networks*, NeurIPS 2015. Localization → grid → differentiable sampling과 task loss 학습 원리. `https://proceedings.neurips.cc/paper/2015/hash/33ceb07bf4eeb3da587e268d663aba1a-Abstract.html`
- **[R2]** PyTorch official, *Spatial Transformer Networks Tutorial*. Identity initialization과 end-to-end 연결. `https://docs.pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html`
- **[R3]** PyTorch official, `torch.nn.functional.grid_sample`. Grid 정의, align_corners, bicubic/border, CUDA backward 비결정성 안내. 조회 시 2.13 문서로 연결됨. `https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.grid_sample.html`
- **[R4]** PyTorch official, *Reproducibility*. Version/platform/seed 및 DataLoader randomness 제한. `https://docs.pytorch.org/docs/stable/notes/randomness.html`
- **[R5]** NIST/SEMATECH e-Handbook, *Randomized block designs*. Nuisance factor를 block 안에서 통제하는 비교 설계. `https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm`
- **[R6]** Haber & Modersitzki, *Intensity Gradient Based Registration and Fusion of Multi-modal Images*, MICCAI 2006, DOI `10.1007/11866763_89`. Normalized gradient를 통한 cross-modality similarity의 선행 근거. 이 문서의 outer-product loss는 별도 명세. `https://pubmed.ncbi.nlm.nih.gov/17354837/`

---

## 18. 한 페이지 인계 요약 — v2 우선 적용

**방법:** PAN/MS 입력을 보고 작은 모델이 global `(dy,dx)`를 예측해 PAN만 MS-frame으로 sampling하고, 같은 W96–D124 U-Net으로 HRMS를 복원한다. MS base·GT·최종 SR은 이동하지 않는다.

**정합 모듈:** PAN/MS stem 각각 16ch → concat32 → Conv32(stride2) → Conv64(stride2) → ResBlock64 → GAP → MLP64→32→2. WV3 105,330 params는 v1 reference 기록. 마지막 Linear만 zero-init. Output은 현재 HR pixel 단위 linear translation.

**세 실험:** A1 recon만 / A2 recon+출력–GT edge / A3 recon+보정 PAN–GT 직접 구조 loss. A3는 A2 loss를 포함하지 않는다. No random jitter, no B-2, no PAN reconstruction, no LPAN/HPAN inputs, no KD.

**서버:** s1(seed2025) A1→A2→A3 / s2(seed1234) A2→A3→A1 / s3(seed777) A3→A1→A2. 총9run. 같은 server–seed에서 paired 비교. Seed는 B0 실기록에 맞춰 실행 전 고정한다.

**평가 — 동일 checkpoint의 동일 SR:**

| View | PAN | 영역 |
|---|---|---|
| Raw-original | 원래 P | B0 원래 영역 |
| Raw-valid | 원래 P | 고정 공통 V |
| Aligned-valid | 실제 forward에서 사용한 P̃ | 같은 V |

Aligned view의 저해상도 PAN도 P̃에서 기존 evaluator 경로로 재생성한다. MS/GT는 원본 그대로이다. Raw 평가는 aligner를 끄는 것이 아니며, zero-shift 재추론은 별도 ablation이다.

**Mask:** border 복제값을 실제 관측으로 간주하지 않는다. Sampling·보간·필터·window support를 추적한다. 무효 픽셀을 0으로 채워 metric을 계산하지 않는다. Selection용 V는 모든 checkpoint/case에서 고정하고, 실제 shift가 V의 support를 깨면 aligned selection 부적격으로 기록한다. 실패 scene을 평균에서 빼거나 ROI를 줄이지 않는다. 평가 mask를 학습 L_rec에 연결하지 않는다.

**Checkpoint:** `best_raw`를 주 비교로, `best_aligned`를 모델 의존 정합 reference의 탐색용으로 저장하며 `last`도 보존한다. 각각 HQNR 우선·동일 view의 fSCC 동률 처리·later-step을 적용한다. 두 selector는 독립 running maximum을 관리한다. 세 저장 checkpoint 모두 세 view와 RR-original/valid를 교차 보고한다.

**해석:** Raw-valid−Raw-original은 영역 변경, Aligned-valid−Raw-valid는 PAN reference 변경 효과를 보여 준다. Aligned HQNR 상승만으로 실제 복원·정합 향상을 주장하지 않는다. 독립 구조 진단과 learned/zero/wrong-sign·known-shift 대조를 함께 본다.

**유지하는 구현 주의:** Δ=0이어도 training에서 warp를 우회하지 않는다. GT는 predictor 입력이 아니라 loss에만 사용한다. 큰 shift로 geometry loss 영역을 숨기지 않는다. 학습 loss 영역은 v1 그대로이다.

**실행 전 연결:** B0 run/seed/train/raw-eval manifest, evaluator support/phase, 사전 fixed ROI, v2 adapter·dual-selector gate. 원문에 없는 수치는 추정하지 않는다. 기존 v1 ZIP은 v2 evaluator를 구현한 패키지가 아니다. 이 문서는 평가 확장을 반영한 실행 명세이며 실제 서버 실행·신규 evaluator 검증을 수행한 보고서가 아니다.
