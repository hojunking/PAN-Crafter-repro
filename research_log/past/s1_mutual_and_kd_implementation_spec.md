> **[확정 변경 — 2026-08-31, 본문과 다르면 이 블록을 따른다]**
> 1. **서버 배정 교체: s1 = KD(Part B), s2 = mutual(Part A)** — 2-peer 학습이
>    VRAM 을 2배 쓰므로 큰 쪽(s2)이 mutual 을 맡는다 (사용자 확정).
> 2. **checkpoint 선택은 공식 HQNR 로 불변** — §0.4-5·§24 의 "FR HQNR 금지 /
>    best val-ERGAS" 는 채택하지 않는다 (사용자 확정, 기존 전 실행과의 비교성 유지).
>    best_state.json 에 best epoch 의 SCC/ERGAS 를 병기해 실행 간 판정(HQNR band
>    → SCC → ERGAS)에 쓴다.
> 3. K5(feature KD)는 구현 결함(proj 스케줄러 미적용) 판정으로 **No-Go 보류**.
> 구현·검증 상세: [2026-08-31_kd-mutual-implementation-report.md](2026-08-31_kd-mutual-implementation-report.md)

# R4–R4 Mutual Learning 및 c6→R4 Two-Stage KD 구현 설계서

> **목적**: s1에서 공통 코드를 구축하고, 동일 커밋을 두 서버에 배포해 다음 두 연구 축을 병렬 검증한다.
>
> - **s1**: 동일 구조 `R4 ↔ R4` 상호학습
> - **s2**: `c6 Teacher → R4 Student` 2-stage knowledge distillation
>
> 이 문서는 단순 실행 계획이 아니라 **모델 구성, loss API, 학습 루프, 설정 파일, unit test, 평가와 중단 기준까지 포함한 구현 명세**다.

---

## 0. 최종 결정 요약

### 0.1 서버별 연구 질문

| 서버 | 주 실험 | 핵심 질문 |
|---|---|---|
| **s1** | `R4_A ↔ R4_B` mutual learning | 동일한 CNN U-Net 두 개도 서로 다른 초기화·최적화 경로 및 loss specialization만으로 단독 학습보다 좋아지는가? |
| **s2** | `c6 → R4` two-stage KD | c6의 spectral/reconstruction 강점을 R4에 선택적으로 전달하면서 R4의 낮은 spatial distortion 특성을 보존할 수 있는가? |

### 0.2 고정 architecture

| 모델 | Width | Depth | Attention | Input | MARs | Params |
|---|---:|---|---|---:|---|---:|
| **c6** | 128 | `[1, 2, 4]` | 없음 | 11ch | 사용 | 약 3.772M |
| **R4** | 96 | `[1, 2, 4]` | 없음 | 11ch | 사용 | 약 2.129M |

두 모델은 topology와 depth가 같고 **width만 다르다**. 따라서 stage별 feature 위치가 대응되며, c6→R4 feature KD에 유리하다.

### 0.3 현재 metric 역할 해석

- c6는 R4보다 **ERGAS, SAM, SCC, Dλ**가 좋다.
- R4는 c6보다 **Ds와 HQNR**이 좋다.
- 따라서 c6를 **spectral/reconstruction teacher**, R4를 **compact student**로 둔다.
- c6의 전체 출력을 무조건 복제하는 KD보다, **spectral 저주파와 신뢰 가능한 영역만 선택적으로 전달**하는 접근을 우선한다.

### 0.4 공통 원칙

1. **MARs는 항상 anchor로 유지**한다.
2. SiS는 HRMS/PAN GT를 대체하지 않고 **misalignment-robust auxiliary spectral loss**로 사용한다.
3. Mutual/KD loss는 **MS mode의 HRMS 출력에만 적용**한다.
4. PAN mode는 기존 PAN back-reconstruction hard loss로 유지한다.
5. FR HQNR로 checkpoint를 고르지 않는다. `final 50K` 또는 `best validation ERGAS`를 사용한다.
6. 두 서버 절대값을 직접 비교하지 않고 **각 서버 local baseline 대비 delta**를 비교한다.

---

## 1. 논문 기반 구현 요구사항

### 1.1 PAN-Crafter에서 유지할 것

PAN-Crafter의 MARs는 한 네트워크가 MS mode에서는 HRMS를, PAN mode에서는 반복된 PAN을 복원하게 한다.

\[
L_{\mathrm{MARs}}
=
\|\hat Y_{\mathrm{HRMS}}-Y_{\mathrm{HRMS}}\|_1
+
\lambda_{\mathrm{PAN}}
\|\hat Y_{\mathrm{PAN}}-Y_{\mathrm{PAN}}\|_1
\]

구현 규칙:

- `lambda_pan = 1.0`을 기본값으로 유지한다.
- MS/PAN mode는 같은 sample을 batch 방향으로 복제해 처리한다.
- Mutual/KD/SiS/GT-variance는 **MS mode**에만 적용한다.
- PAN mode는 auxiliary spatial anchor이며, teacher/student 간 soft target을 만들지 않는다.

### 1.2 SIPSA-Net에서 가져올 것

SIPSA-Net의 SiS loss는 출력과 여러 shifted MS 후보 사이의 차이 중 최소값을 사용한다. 원 논문은 spectral loss와 PAN edge loss를 분리하고, shift 범위는 데이터셋의 misalignment에 맞춰 조절해야 한다고 명시한다.

현재 연구에서 사용할 핵심:

- `SiS`: 출력 HRMS를 LRMS scale로 degradation한 뒤 shifted LRMS 후보와 비교
- `Edge`: 출력 luminance의 절대 gradient와 PAN의 절대 gradient 비교
- 원 논문의 `9×9`을 바로 복사하지 않고 `3×3 → 5×5` 순서로 검증
- WV3 8-band에서는 **bandwise shift**와 **shared spectral-vector shift**를 분리해 ablation

### 1.3 U-Know-DiffPAN에서 가져올 것

U-Know의 기본 구조는 다음이다.

\[
L_{\mathrm{U-Know}}
=
L_{\mathrm{hard}}
+
\lambda_s L_{\mathrm{soft}}
+
\lambda_f L_{\mathrm{feat}}
\]

- Teacher uncertainty가 높은 곳: GT hard loss 강화
- Teacher uncertainty가 낮은 곳: teacher soft target 강화
- 중간 feature도 별도 loss로 전달

현재 연구에서는 diffusion을 사용하지 않고, uncertainty weighting 원리만 CNN teacher/student에 이식한다.

---

## 2. 저장소와 브랜치 구조

실제 저장소 이름에 맞춰 경로는 조정하되, 역할은 아래처럼 분리한다.

```text
project/
├── model/
│   ├── pancrafter_reconstructed.py
│   ├── uncertainty_head.py
│   └── feature_hooks.py
├── loss/
│   ├── mars.py
│   ├── sis.py
│   ├── edge.py
│   ├── mutual.py
│   ├── uknow.py
│   ├── feature_kd.py
│   └── weighting.py
├── operators/
│   ├── sensor_mtf.py
│   ├── gradients.py
│   ├── local_variance.py
│   └── shift_candidates.py
├── trainer/
│   ├── train_single.py
│   ├── train_mutual.py
│   ├── train_teacher_uncertainty.py
│   └── train_student_kd.py
├── evaluator/
│   ├── evaluate_rr.py
│   ├── evaluate_fr.py
│   ├── evaluate_shift_robustness.py
│   └── diagnostics.py
├── configs/
│   ├── common/
│   ├── mutual/
│   ├── teacher/
│   └── kd/
└── tests/
    ├── test_sis.py
    ├── test_uncertainty.py
    ├── test_mutual_grad.py
    ├── test_feature_shapes.py
    └── test_resume_determinism.py
```

### 2.1 Git 운용

```text
main
└── feat/mutual-kd-loss-framework     # s1에서 공통 구현
    ├── exp/s1-r4-r4-mutual
    └── exp/s2-c6-r4-kd
```

배포 절차:

1. s1에서 공통 모듈 구현 및 unit test 완료
2. `git tag ml-kd-v1`
3. s2에서 동일 tag checkout
4. 각 결과 폴더에 `git_commit.txt`, `pip_freeze.txt`, `config_resolved.yaml` 저장

---

## 3. 공통 데이터 계약

DataLoader가 반환할 권장 dictionary:

```python
batch = {
    "pan": pan,              # [B, 1, 64, 64]
    "lrms": lrms,            # [B, 8, 16, 16]
    "upms": upms,            # [B, 8, 64, 64]
    "hrms_gt": hrms_gt,      # [B, 8, 64, 64]
    "lpan": lpan,            # 필요 시 [B, 1, 64, 64]
    "sample_id": sample_id,
}
```

11ch 입력 구성은 현재 코드와 동일하게 유지한다.

```text
8ch: upsampled MS
1ch: PAN
1ch: upsampled low-pass PAN
1ch: PAN - upsampled low-pass PAN
```

### 3.1 augmentation 규칙

- `crop=False`
- horizontal/vertical flip, 90° rotation은 PAN/LRMS/GT에 동일하게 적용
- Mutual peer A/B는 **동일한 augmented batch**를 사용
- A/B에 서로 다른 geometric transform을 적용하면 output mutual loss 좌표가 맞지 않으므로 금지
- 서로 다른 최적화 경로는 initialization seed와 loss specialization으로 만든다

---

## 4. 모델 forward API

모든 trainer가 공통으로 사용할 출력 contract를 정의한다.

```python
@dataclass
class ModelOutput:
    residual: torch.Tensor        # [B, 8, H, W]
    hrms: torch.Tensor            # upms + residual
    pan_recon: torch.Tensor | None
    uncertainty: torch.Tensor | None   # [B, 1, H, W]
    features: dict[str, torch.Tensor]
```

권장 forward:

```python
def forward(
    self,
    pan: Tensor,
    lrms: Tensor,
    upms: Tensor,
    mode: Literal["MS", "PAN"],
    return_features: bool = False,
    return_uncertainty: bool = False,
) -> ModelOutput:
    ...
```

Feature key는 c6/R4에서 동일하게 유지한다.

```text
enc_h
enc_h2
bottleneck_h4
dec_h2
dec_h
```

초기 feature KD는 `bottleneck_h4` 하나만 사용한다.

---

## 5. 공통 loss return API

모든 loss는 total scalar뿐 아니라 logging용 구성 항목과 map을 반환한다.

```python
@dataclass
class LossResult:
    loss: torch.Tensor
    scalars: dict[str, torch.Tensor]
    maps: dict[str, torch.Tensor]
```

예:

```python
result = sis_loss(pred_hrms, lrms)
# result.loss
# result.scalars["sis_mean"]
# result.maps["offset_index"]
# result.maps["offset_magnitude"]
```

이 구조를 쓰면 TensorBoard/W&B 로깅과 ablation 분석이 단순해진다.

---

## 6. 공통 operator 구현

## 6.1 Sensor MTF degradation

SiS와 spectral KD는 HRMS를 LRMS scale로 내려 비교한다.

```python
class SensorMTFDownsampler(nn.Module):
    def forward(self, x_hr: Tensor) -> Tensor:
        # band-wise MTF blur
        # stride=4 downsample
        # output: [B, C, H/4, W/4]
        ...
```

필수 조건:

- 기존 평가/데이터 생성에서 사용하는 sensor MTF kernel과 동일하게 구성
- 임의 bicubic downsample을 기본값으로 사용하지 않음
- band별 kernel이 존재하면 grouped convolution 사용
- `requires_grad=False`, gradient는 입력 prediction으로만 흐름

## 6.2 Absolute edge operator

SIPSA의 edge loss 취지를 따라 edge 방향이 아니라 절대 gradient 크기를 비교한다.

```python
class AbsoluteGradient(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        gx = conv2d(x, sobel_x)
        gy = conv2d(x, sobel_y)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)
```

WV3 8-band HRMS luminance baseline:

```python
lum = hrms.mean(dim=1, keepdim=True)
```

추후 sensor PAN spectral response가 확보되면 weighted luminance를 별도 ablation으로 추가한다.

## 6.3 GT local variance

```python
class LocalVarianceMap(nn.Module):
    def __init__(self, kernel_size: int = 5): ...

    def forward(self, gt: Tensor) -> Tensor:
        # gt: [B, 8, H, W]
        z = normalize_per_band(gt)
        mean = avg_pool2d(z, k, stride=1, padding=k//2)
        mean2 = avg_pool2d(z*z, k, stride=1, padding=k//2)
        var = (mean2 - mean*mean).clamp_min(0)
        var = var.mean(dim=1, keepdim=True)
        return robust_normalize_01(var.detach())
```

권장 robust normalization:

```text
q05 = batch/patch 5 percentile
q95 = batch/patch 95 percentile
v = clamp((v_raw - q05) / (q95 - q05 + eps), 0, 1)
```

Variance map은 gradient를 받지 않으며 hard GT loss weighting에만 우선 적용한다.

---

## 7. SiS loss 구현 명세

## 7.1 기본 정의

현재 adaptation에서는 HRMS를 LRMS scale로 degradation한 뒤 shifted LRMS와 비교한다.

\[
\hat Y_{lr}=D_{\mathrm{MTF}}(\hat Y_{hr})
\]

\[
L_{\mathrm{SiS}}(x)
=
\min_{\delta\in\Omega}
 d_{\mathrm{spec}}
 \left(
 \hat Y_{lr}(x),
 LRMS(x+\delta)
 \right)
\]

기본 shift radius:

```text
radius=0 : 1×1, conventional spectral target
radius=1 : 3×3, ±1 LRMS pixel
radius=2 : 5×5, ±2 LRMS pixels
```

원 SIPSA의 9×9은 현재 16×16 LRMS patch에 비해 너무 넓을 수 있으므로 후순위다.

## 7.2 shifted candidate 생성

```python
def build_shift_candidates(lrms: Tensor, radius: int) -> Tensor:
    # input : [B, C, h, w]
    # output: [B, C, K, h, w], K=(2r+1)^2
    padded = F.pad(lrms, [r, r, r, r], mode="reflect")
    patches = F.unfold(padded, kernel_size=2*r+1)
    ...
```

경계의 artificial shift를 피하려면 loss 계산 시 `r` pixel border를 제외한다.

## 7.3 두 종류의 SiS를 모두 구현

### A. `bandwise` — SIPSA 원식에 가까운 baseline

각 band가 독립적으로 가장 가까운 shift를 선택한다.

```python
cost = abs(pred_lr.unsqueeze(2) - candidates)   # [B,C,K,h,w]
loss_map, offset_idx = cost.min(dim=2)
loss = loss_map[..., r:-r, r:-r].mean()
```

### B. `shared_vector` — WV3 8-band 제안 확장

모든 band가 하나의 shift를 공유한다.

```python
l1_cost = abs(pred_lr.unsqueeze(2) - candidates).mean(dim=1)
# [B,K,h,w]

sam_cost = spectral_angle_per_candidate(pred_lr, candidates)
total_cost = l1_cost + eta_sam * sam_cost

min_cost, offset_idx = total_cost.min(dim=1)
loss = min_cost[..., r:-r, r:-r].mean()
```

이 방식은 서로 다른 위치의 band가 한 spectral vector로 조합되는 것을 막는다.

## 7.4 optional soft-min

Hard min이 불안정할 경우에만 사용한다.

\[
\operatorname{softmin}_{\tau}(c_k)
=
-\tau\log\sum_k e^{-c_k/\tau}
\]

기본 실험은 SIPSA 취지와 가까운 hard min으로 시작한다.

## 7.5 SiS 필수 diagnostics

매 epoch 또는 1K iteration마다 기록:

```text
center_shift_ratio
mean_abs_dx
mean_abs_dy
mean_offset_magnitude
offset_histogram
sis_loss
sis_vs_gt_l1_ratio
```

경고 조건:

- center shift가 거의 0%: 모델이 임의 이웃으로 target을 회피할 가능성
- 최대 radius 경계 offset이 과도하게 많음: shift 범위가 작거나 loss가 잘못됨
- SiS는 감소하지만 GT ERGAS가 빠르게 악화: target avoidance 가능성

---

## 8. Teacher uncertainty 구현

## 8.1 head 위치

MS mode의 마지막 decoder feature에서 scalar uncertainty를 예측한다.

```python
class UncertaintyHead(nn.Module):
    def __init__(self, channels: int):
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 1, 1),
        )

    def forward(self, feat: Tensor) -> Tensor:
        return F.softplus(self.net(feat)) + 1e-6
```

초기에는 8-band별 uncertainty가 아니라 pixel당 1-channel map을 사용한다.

## 8.2 uncertainty training loss

U-Know 식을 CNN 복원에 적용한다.

\[
e_T(x)=\frac{1}{B}\sum_b |\hat Y_{T,b}(x)-Y_b(x)|
\]

\[
L_{\mathrm{unc}}^T
=
\operatorname{mean}
\left[
\frac{e_T}{2\theta_T}
+
\frac{1}{2}\log\theta_T
\right]
\]

Teacher total:

\[
L_T
=
L_{\mathrm{unc}}^T
+
\lambda_{PAN}L_{PAN}
+
\lambda_{SiS}L_{SiS}
+
\lambda_{edge}L_{edge}
\]

## 8.3 uncertainty validation

Teacher가 Student loss를 제어하기 전에 uncertainty가 실제 error와 연결되는지 확인한다.

필수 통계:

```text
Spearman(theta, abs_error)
MAE by uncertainty quintile
risk-coverage curve
mean theta on high-GT-variance region
mean theta on low-GT-variance region
```

`Spearman <= 0` 또는 quintile별 error 순서가 뒤집히면 uncertainty KD를 진행하지 않고 head/loss를 수정한다.

## 8.4 Student weighting용 normalization

Paper-faithful mode와 robust mode를 모두 구현한다.

```yaml
uncertainty_weight_mode: paper_raw | robust_normalized
```

### `paper_raw`

```python
w_hard = tau + theta
w_soft = torch.clamp(tau - theta, min=0.0)
```

### `robust_normalized` — 권장 adaptation

```python
u = robust_normalize_01(theta.detach())
w_hard = 1.0 + alpha_u * u
w_soft = 1.0 - u
```

각 weight map은 평균 1로 정규화해 loss scale 변화를 분리한다.

```python
w = w / (w.mean(dim=(-2,-1), keepdim=True) + 1e-6)
```

---

# Part A. s1 — R4 ↔ R4 Mutual Learning

## 9. s1 연구 목적

동일 구조 R4 두 개를 이용해 다음을 단계적으로 분리한다.

1. 서로 다른 initialization/optimization path만으로 DML 효과가 있는가?
2. 구조는 같지만 edge loss와 SiS loss를 달리하면 specialization이 형성되는가?
3. 전체 output consistency보다 component-wise mutual transfer가 더 적합한가?

두 모델은 teacher/student가 아니라 `peer_a`, `peer_b`로 명명한다.

---

## 10. s1 model 초기화

```python
set_global_seed(2025)
peer_a = build_r4()

set_global_seed(2026)
peer_b = build_r4()
```

DataLoader seed는 고정하고 같은 mini-batch를 두 모델에 넣는다.

- 모델 parameter seed만 다르게 설정
- augmentation 결과는 공유
- optimizer state는 각각 독립

```python
opt_a = AdamW(peer_a.parameters(), ...)
opt_b = AdamW(peer_b.parameters(), ...)
```

---

## 11. Mutual loss 종류

## 11.1 Vanilla residual mutual

전체 HRMS보다 injected residual을 맞춘다.

\[
R_i=\hat Y_i-Up(MS)
\]

\[
L_{A\leftarrow B}^{res}
=
\rho(R_A-\operatorname{sg}(R_B))
\]

\[
L_{B\leftarrow A}^{res}
=
\rho(R_B-\operatorname{sg}(R_A))
\]

`rho`는 Charbonnier 권장:

```python
def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps).mean()
```

## 11.2 Spectral component mutual

SiS/spectral peer에서 spatial peer로 전달한다.

\[
L_{M\rightarrow P}^{spec}
=
\|D_{MTF}(\hat Y_P)-\operatorname{sg}(D_{MTF}(\hat Y_M))\|_1
+
\eta\,SAM
\]

## 11.3 Spatial component mutual

Edge peer에서 spectral peer로 전달한다.

\[
L_{P\rightarrow M}^{edge}
=
\left\|
|\nabla Lum(\hat Y_M)|
-
\operatorname{sg}(|\nabla Lum(\hat Y_P)|)
\right\|_1
\]

처음에는 raw feature mutual loss를 사용하지 않는다.

---

## 12. s1 experiment matrix

## M0 — matched independent control

```text
peer_a: R4 + MARs
peer_b: R4 + MARs
mutual: 없음
```

목적:

- 같은 trainer, 같은 batch, 같은 총 compute에서 mutual loss만 없는 대조군
- 기존 R4 단일 결과보다 더 정확한 비교 기준

## M1 — homogeneous vanilla mutual

```text
peer_a: R4 + MARs
peer_b: R4 + MARs
mutual: bidirectional residual consistency
```

\[
L_A=L_{MARs}^A+\lambda_m(t)L_{A\leftarrow B}^{res}
\]

\[
L_B=L_{MARs}^B+\lambda_m(t)L_{B\leftarrow A}^{res}
\]

이 실험은 동일 architecture의 optimization diversity만 검증한다.

## M2 — loss specialization, mutual 없음

```text
peer_p: R4 + MARs + Edge
peer_m: R4 + MARs + SiS
mutual: 없음
```

목적:

- edge/SiS 자체가 specialization을 만드는지 확인
- M3의 개선이 mutual 때문인지, auxiliary loss 때문인지 분리

## M3 — component-wise mutual

```text
peer_p: R4 + MARs + Edge + spectral knowledge from peer_m
peer_m: R4 + MARs + SiS  + edge knowledge from peer_p
```

\[
L_P
=
L_{MARs}^P
+
\alpha_{edge}L_{edge}^P
+
\lambda_{M\rightarrow P}(t)L_{M\rightarrow P}^{spec}
\]

\[
L_M
=
L_{MARs}^M
+
\beta_{SiS}L_{SiS}^M
+
\lambda_{P\rightarrow M}(t)L_{P\rightarrow M}^{edge}
\]

---

## 13. Mutual weight schedule

```python
def mutual_weight(step, total_steps=50_000, max_w=0.05):
    if step < 5_000:
        return 0.0
    if step < 15_000:
        return max_w * (step - 5_000) / 10_000
    if step < 40_000:
        return max_w
    return max_w * (50_000 - step) / 10_000
```

초기 search set:

```yaml
lambda_mutual: [0.02, 0.05]
lambda_edge_self: 0.05
lambda_sis_self: 0.10
lambda_m_to_p_spec: 0.05
lambda_p_to_m_edge: 0.02
```

이 값은 고정 정답이 아니다. 1K smoke에서 weighted gradient norm을 확인한다.

권장 gradient budget:

```text
weighted mutual grad / MARs grad: 0.05 ~ 0.15
weighted SiS grad    / MARs grad: 0.10 ~ 0.30
weighted edge grad   / MARs grad: 0.05 ~ 0.20
```

---

## 14. Mutual trainer pseudo-code

```python
for batch in loader:
    batch = move_to_device(batch)

    # 같은 augmentation batch를 두 peer가 공유
    out_a_ms = peer_a(..., mode="MS", return_features=True)
    out_a_pan = peer_a(..., mode="PAN")

    out_b_ms = peer_b(..., mode="MS", return_features=True)
    out_b_pan = peer_b(..., mode="PAN")

    mars_a = mars_loss(out_a_ms, out_a_pan, batch)
    mars_b = mars_loss(out_b_ms, out_b_pan, batch)

    aux_a, aux_b = build_self_losses(...)
    mut_a, mut_b = build_mutual_losses(
        out_a_ms,
        out_b_ms,
        detach_targets=True,
    )

    loss_a = mars_a + aux_a + mut_a
    loss_b = mars_b + aux_b + mut_b
    total = loss_a + loss_b

    opt_a.zero_grad(set_to_none=True)
    opt_b.zero_grad(set_to_none=True)
    scaler.scale(total).backward()
    scaler.step(opt_a)
    scaler.step(opt_b)
    scaler.update()
```

반드시 상대 peer target을 detach한다.

```python
target_b = out_b_ms.hrms.detach()
target_a = out_a_ms.hrms.detach()
```

그렇지 않으면 `loss_a`가 peer_b까지 업데이트해 방향 분리가 무너진다.

---

## 15. s1 memory 운영

R4는 작지만 MARs dual mode와 two-peer 때문에 activation 수가 증가한다.

초기 설정:

```yaml
batch_size_per_peer: 24
grad_accum_steps: 2
effective_batch_per_peer: 48
amp: true
grad_checkpoint: false
```

VRAM 여유가 있으면 batch 48로 복원한다.

1K smoke에서 기록:

```text
peak_vram
iter_time
samples_per_second
loss scale
NaN/Inf
```

---

## 16. s1 additional diagnostics

매 validation에서 기록:

```text
peer_a RR/FR metrics
peer_b RR/FR metrics
ensemble RR/FR metrics
residual disagreement L1
spectral disagreement at LR scale
edge disagreement
pixel error correlation
center/non-center SiS shift ratio
```

DML 성공은 ensemble만 좋아지는 것이 아니라 **각 단일 peer 또는 두 peer 평균이 M0보다 좋아지는 것**이다.

---

# Part B. s2 — c6 → R4 Two-Stage KD

## 17. s2 연구 목적

1. c6를 uncertainty-aware teacher로 만들 수 있는가?
2. c6의 spectral/reconstruction knowledge가 R4의 ERGAS/SAM/Dλ를 개선하는가?
3. R4의 낮은 Ds 특성을 유지할 수 있는가?
4. GT variance와 SiS가 uncertainty weighting에 추가 정보를 제공하는가?

---

## 18. Stage 1 — Teacher experiments

## T0 — existing c6 MARs baseline

```text
c6 W128 d124 no-attn 11ch nocrop
loss = MARs
```

현재 checkpoint를 재사용할 수 있으나, 최종 비교는 동일 commit/evaluator로 재평가한다.

## T1 — c6 + uncertainty

```text
loss = uncertainty-weighted HRMS reconstruction + PAN MARs
```

\[
L_{T1}=L_{unc}^T+\lambda_{PAN}L_{PAN}
\]

Teacher 선택 전 uncertainty calibration을 통과해야 한다.

## T2 — c6 + uncertainty + SiS

```text
loss = T1 + lambda_sis * SiS
```

우선 `radius=1`, `shared_vector`를 사용한다.

Ablation 순서:

```text
T2-a: bandwise, radius=1
T2-b: shared_vector, radius=1
T2-c: winner, radius=2
```

## T3 — edge 추가, 조건부

T2가 spectral robustness는 좋아지지만 edge/SCC가 약해질 때만 실행한다.

```text
loss = T2 + lambda_edge * SIPSA-style absolute edge loss
```

PAN-Crafter의 PAN-mode supervision과 중복될 수 있으므로 필수 실험이 아니다.

---

## 19. Teacher 선택 기준

Teacher는 HQNR 하나가 아니라 **전달할 지식의 신뢰성**으로 고른다.

우선순위:

1. validation ERGAS / SAM
2. shift robustness curve
3. Dλ
4. uncertainty calibration
5. SCC/HQNR non-regression

Teacher 후보가 기존 c6보다 HQNR이 낮더라도 spectral/GT reconstruction과 uncertainty calibration이 좋으면 KD teacher로 사용할 수 있다.

---

## 20. Stage 2 — Student KD experiments

Teacher는 완전히 freeze한다.

```python
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
```

Teacher output/feature/uncertainty는 같은 augmented batch에서 online으로 계산한다.

```python
with torch.no_grad():
    t_out = teacher(..., mode="MS", return_features=True, return_uncertainty=True)
```

개발 단계에서는 teacher output cache를 사용하지 않는다. 캐시와 augmentation 좌표가 어긋날 수 있다.

## K0 — R4 hard baseline

```text
R4 + MARs only
```

동일 code path에서 다시 측정한 local baseline이다.

## K1-A — vanilla full-output KD

\[
L_{soft}^{full}
=
\|\hat Y_S-\operatorname{sg}(\hat Y_T)\|_1
\]

이 실험은 필수 대조군이지만 최종 방법으로 예상하지 않는다.

## K1-B — spectral-only KD

\[
L_{soft}^{spec}
=
\|D_{MTF}(\hat Y_S)-\operatorname{sg}(D_{MTF}(\hat Y_T))\|_1
+
\eta\,SAM
\]

c6가 R4보다 강한 spectral component만 전달한다.

K1-A와 K1-B를 비교해 full-output 복제가 R4의 Ds 장점을 해치는지 확인한다.

## K2 — U-Know spectral KD

\[
L_{hard}^{U}
=
\|w_{hard}(U_T)\odot(\hat Y_S-Y)\|_1
\]

\[
L_{soft}^{U}
=
\|w_{soft}(U_T)\odot(D(\hat Y_S)-D(\hat Y_T))\|_1
\]

\[
L_{K2}
=
L_{hard}^{U}
+
\lambda_{PAN}L_{PAN}^{S}
+
\lambda_sL_{soft}^{U}
\]

## K3 — GT variance 추가

\[
w_{hard}
=
1+\alpha U_T+\beta V_Y
\]

```text
U_T: teacher reliability
V_Y: local content complexity
```

GT variance는 hard loss에만 우선 사용한다.

```python
w_hard = normalize_mean(1 + alpha_u * u + beta_v * gt_var)
w_soft = normalize_mean(1 - u)
```

## K4 — Student SiS 추가

\[
L_{K4}=L_{K3}+\lambda_{SiS}L_{SiS}^{S}
\]

SiS는 teacher–student 사이가 아니라 Student HRMS와 shifted LRMS observation 사이에 적용한다.

Ablation:

```text
K4-a: Teacher SiS X / Student SiS O
K4-b: Teacher SiS O / Student SiS X
K4-c: Teacher SiS O / Student SiS O
```

## K5 — feature KD, 조건부

초기 stage는 bottleneck 하나만 사용한다.

c6 width 128, R4 width 96이므로 공통 projection dimension 64를 사용한다.

```python
proj_t = Conv1x1(128, 64)
proj_s = Conv1x1(96, 64)
```

\[
L_{feat}
=
\|
W_{soft}^{btl}
\odot
(N(P_S(F_S))-\operatorname{sg}(N(P_T(F_T))))
\|_1
\]

\[
L_{K5}=L_{K4}+\lambda_fL_{feat}
\]

U-Know 논문의 시작값을 참고해:

```yaml
lambda_soft: 0.1
lambda_feat: 0.001
```

로 시작하되 gradient norm을 확인한다.

---

## 21. Student trainer pseudo-code

```python
for batch in loader:
    batch = move_to_device(batch)

    with torch.no_grad():
        t_ms = teacher(
            batch["pan"], batch["lrms"], batch["upms"],
            mode="MS",
            return_features=True,
            return_uncertainty=True,
        )

    s_ms = student(..., mode="MS", return_features=True)
    s_pan = student(..., mode="PAN")

    u = normalize_uncertainty(t_ms.uncertainty.detach())
    v = gt_variance(batch["hrms_gt"]).detach()

    hard = hard_gt_loss(
        s_ms.hrms,
        batch["hrms_gt"],
        weight=build_hard_weight(u, v),
    )

    pan_loss = l1(s_pan.pan_recon, repeated_pan_gt)

    soft = spectral_kd_loss(
        s_ms.hrms,
        t_ms.hrms.detach(),
        weight=build_soft_weight(u),
    )

    sis = sis_loss(s_ms.hrms, batch["lrms"])
    feat = feature_kd(s_ms.features, t_ms.features, u)

    total = (
        hard
        + lambda_pan * pan_loss
        + lambda_soft * soft
        + lambda_sis * sis
        + lambda_feat * feat
    )

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(total).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 22. KD loss schedule

초기부터 teacher/SiS/feature loss를 모두 강하게 켜지 않는다.

```text
0–5K    : MARs hard loss only
5–15K   : soft spectral KD ramp-up
10–20K  : SiS ramp-up
15–40K  : main KD stage
40–50K  : soft/feature loss decay, hard GT 중심 fine-tuning
```

권장 함수:

```python
lambda_soft_t = ramp_then_decay(step, start=5k, full=15k, decay=40k)
lambda_sis_t  = ramp_then_decay(step, start=10k, full=20k, decay=45k)
lambda_feat_t = ramp_then_decay(step, start=15k, full=25k, decay=40k)
```

---

## 23. s2 memory 운영

Teacher는 `eval + no_grad`로 실행하므로 두 trainable peer보다 메모리 부담이 작다.

초기 설정:

```yaml
student_batch: 32
grad_accum_steps: 1
amp: true
teacher_no_grad: true
```

GPU VRAM이 부족하면:

```yaml
student_batch: 16
grad_accum_steps: 2
```

s2의 64GB가 system RAM인 경우 GPU VRAM과는 별개이므로 1K smoke에서 실제 VRAM을 확인한다.

---

## 24. 공통 checkpoint 정책

저장:

```text
last.pt
best_val_ergas.pt
best_val_sam.pt          # 선택적 보조
```

금지:

```text
best FR HQNR checkpoint
FR test image로 early stopping
```

최종 보고는:

- `best_val_ergas.pt`를 주 결과
- `last.pt`를 stability check
- FR은 checkpoint 확정 후 1회 평가

Resume 저장 항목:

```text
model state
optimizer state
scheduler state
AMP GradScaler
iteration
global RNG
CUDA RNG
DataLoader sampler state 가능 시
```

---

## 25. 평가 프로토콜

## 25.1 RR

- 20 test images
- ERGAS, SAM, SCC, Q8, PSNR, SSIM
- per-image metric array 저장

## 25.2 FR

- 기존 논문 비교 가능한 WV3 index 12–19, 8 images
- repaired `lpan`
- Dλ, Ds, HQNR
- FR metric은 높은 variance를 가지므로 단독 판정 금지

## 25.3 Mutual-specific

```text
peer A/B individual metrics
A+B ensemble metrics
error correlation
output disagreement
spectral disagreement
edge disagreement
mutual loss curve
```

## 25.4 KD-specific

```text
teacher/student metric gap
Dλ gap recovery ratio
Ds preservation
uncertainty-error correlation
GT variance bin metrics
feature similarity
```

## 25.5 SiS-specific shift robustness

LRMS에 synthetic shift를 적용한다.

```text
(0,0)
(±1,0), (0,±1)
(±1,±1)
(±2,0), (0,±2)   # 조건부
```

각 shift에서:

```text
ERGAS
SAM
SCC
shift-aware spectral error
local high-variance MAE
```

SiS의 핵심 성공은 clean metric 하나가 아니라 **shift 증가에 따른 성능 저하 곡선이 완만해지는 것**이다.

---

## 26. Unit tests

## 26.1 SiS

### Test 1: exact shift recovery

```text
LRMS를 오른쪽 1 pixel 이동한 synthetic target 생성
radius=1 SiS에서 최적 offset이 (-1,0)으로 회복되는지 확인
```

### Test 2: center equivalence

```text
radius=0 SiS == conventional L1 at LR scale
```

### Test 3: shared vector

```text
8 band가 같은 위치를 선택하는지 확인
bandwise mode와 결과가 다른 synthetic example 구성
```

### Test 4: gradient

```text
pred_hrms.grad finite
LRMS target에는 grad 없음
```

## 26.2 Uncertainty

```text
theta > 0
no NaN / Inf
high error synthetic region에서 theta 증가 방향 확인
soft weight non-negative
weight map mean normalization 확인
```

## 26.3 Mutual gradient isolation

```text
loss_A만 backward했을 때 peer_B parameter grad == None 또는 0
loss_B만 backward했을 때 peer_A parameter grad == None 또는 0
```

## 26.4 Feature KD

```text
c6 feature 128ch → 64ch
R4 feature 96ch  → 64ch
spatial size 일치
teacher feature detach 확인
```

## 26.5 Resume determinism

```text
100 step 연속 실행
50 step save + resume + 50 step
metric/loss 허용 오차 내 동일
```

---

## 27. Smoke test checklist

각 experiment 실행 전 200–1,000 iteration smoke:

```text
[ ] model build
[ ] forward MS mode
[ ] forward PAN mode
[ ] backward
[ ] AMP
[ ] NaN / Inf
[ ] peak VRAM
[ ] iteration time
[ ] RR inference output shape
[ ] FR 512×512 inference
[ ] loss component magnitude
[ ] gradient norm ratio
[ ] checkpoint save/resume
```

---

## 28. Logging schema

TensorBoard/W&B scalar:

```text
train/total
train/mars_ms
train/mars_pan
train/sis
train/edge
train/mutual_residual
train/mutual_spectral
train/mutual_edge
train/kd_hard
train/kd_soft
train/kd_feature
train/uncertainty_nll
train/grad_norm_anchor
train/grad_norm_aux
train/grad_cosine_anchor_aux
```

Map/image logging:

```text
prediction RGB
GT RGB
absolute error
PAN
uncertainty
GT variance
SiS offset magnitude
edge map
teacher-student difference
peer A-B difference
```

결과 폴더:

```text
outputs/{server}/{experiment_id}/
├── config_resolved.yaml
├── git_commit.txt
├── checkpoints/
├── metrics/
│   ├── rr_summary.json
│   ├── rr_per_image.json
│   ├── fr_summary.json
│   ├── fr_per_image.json
│   └── diagnostics.json
├── curves/
└── visuals/
```

---

## 29. 실험 ID와 실행 순서

# s1 queue — Mutual

```text
S1-M0-R4-R4-INDEPENDENT
S1-M1-R4-R4-RESIDUAL-MUTUAL-L002
S1-M1-R4-R4-RESIDUAL-MUTUAL-L005   # 첫 결과가 약할 때
S1-M2-R4-EDGE__R4-SIS-INDEPENDENT
S1-M3-R4-EDGE__R4-SIS-COMPONENT-MUTUAL
```

권장 우선순위:

1. M0
2. M1 λ=0.02
3. M2
4. M3
5. M1 λ=0.05는 M1 λ=0.02가 under-coupled일 때만

# s2 queue — KD

```text
S2-T0-C6-MARS
S2-T1-C6-UNCERTAINTY
S2-T2-C6-UNCERTAINTY-SIS-R1-SHARED
S2-K0-R4-MARS
S2-K1A-R4-FULL-OUTPUT-KD
S2-K1B-R4-SPECTRAL-KD
S2-K2-R4-UKNOW-SPECTRAL-KD
S2-K3-R4-UKNOW-GTVAR
S2-K4-R4-UKNOW-GTVAR-SIS
S2-K5-R4-UKNOW-GTVAR-SIS-FEAT   # 조건부
```

---

## 30. 대략적인 실행 시간

최근 단일 50K 기준을 바탕으로 한 운영 추정이며, 1K smoke 후 갱신한다.

| 실험 | 예상 |
|---|---:|
| R4 single 50K | 약 2h |
| R4–R4 mutual 50K | 약 4–5h |
| c6 uncertainty teacher 50K | 약 2.5–3h |
| R4 online KD 50K | 약 3–4h |

두 서버를 병렬 사용하면 mutual baseline/제안과 teacher/student ablation을 같은 기간에 확보할 수 있다.

---

## 31. 성공·중단 기준

## 31.1 Mutual

M0 대비 M1/M3에서 다음을 본다.

성공:

- 두 peer 평균 ERGAS/SAM 개선 또는 non-inferior
- SCC 유지
- HQNR/Dλ/Ds 중 적어도 하나가 의미 있게 개선
- 한 peer만 좋아지고 다른 peer가 크게 나빠지지 않음
- ensemble뿐 아니라 단일 peer도 개선

부분 성공:

- HQNR 개선, ERGAS/SAM 소폭 악화
- spectral/spatial trade-off 이동만 확인

중단:

- mutual disagreement가 초기에 거의 0으로 붕괴
- 두 peer 모두 M0보다 악화
- mutual gradient가 MARs gradient를 지속적으로 역행

## 31.2 KD

목표는:

```text
R4의 낮은 Ds 유지
+ c6의 낮은 Dλ / ERGAS / SAM 획득
```

운영 목표:

- R4 local baseline 대비 ERGAS ≥1% 개선
- SAM 개선
- Dλ gap을 c6 방향으로 최소 30% 이상 회복
- Ds 악화는 0.002 이내를 우선 목표
- HQNR non-regression

K1-A full KD가 Ds를 크게 악화하고 K1-B spectral KD가 보존하면 선택적 KD 가설이 지지된다.

## 31.3 SiS

성공:

- synthetic shift에서 성능 저하 감소
- moving-object/edge artifact 감소
- center/offset histogram이 합리적
- clean RR metric이 크게 무너지지 않음

중단/수정:

- offset이 radius boundary에 몰림
- GT loss는 악화되는데 SiS만 급감
- large-radius에서 임의 target avoidance 발생

---

## 32. 주요 실패 모드와 대응

| 실패 | 원인 | 대응 |
|---|---|---|
| Mutual peers가 빠르게 동일해짐 | λ 과대, 동일 loss | λ 감소, warm-up 증가, M3 specialization 사용 |
| ERGAS는 악화되고 HQNR만 상승 | spatial injection 감소 또는 trade-off | Dλ/Ds 분리, edge/SCC/시각 결과 확인 |
| SiS offset이 항상 바깥쪽 | radius 부족 또는 target avoidance | radius/penalty 재설계, offset distance penalty 추가 |
| SiS가 spectral vector를 깨뜨림 | bandwise 독립 shift | shared-vector shift 사용 |
| U-Know soft weight 음수 | raw θ > τ | clamp 또는 robust normalization |
| uncertainty와 error 상관 없음 | head/loss calibration 실패 | Student KD 이전에 teacher uncertainty 재설계 |
| GT variance와 uncertainty가 중복 | 같은 edge만 강조 | 상관 분석 후 GT variance 제거/축소 |
| Feature KD가 Student를 과도하게 묶음 | λf 과대 | bottleneck 한 곳, λf 축소, late start |
| s1/s2 metric 차이 | server/config/checkpoint 차이 | local baseline delta만 비교, commit/config 고정 |

---

## 33. 구현 완료 정의

다음 항목을 모두 충족하면 s1 공통 코드 구축 완료로 본다.

```text
[ ] c6/R4 model config가 하나의 builder에서 생성됨
[ ] ModelOutput contract 구현
[ ] MARs loss 분리 모듈화
[ ] MTF downsampler 구현 및 기존 pipeline과 일치 확인
[ ] SiS bandwise/shared-vector 구현
[ ] SiS offset diagnostics 구현
[ ] absolute edge loss 구현
[ ] uncertainty head 및 loss 구현
[ ] GT local variance map 구현
[ ] residual/spectral/edge mutual loss 구현
[ ] U-Know hard/soft weighting 구현
[ ] bottleneck feature KD 구현
[ ] mutual trainer 구현
[ ] teacher trainer 구현
[ ] student KD trainer 구현
[ ] unit tests 통과
[ ] 1K smoke 통과
[ ] config와 output logging 저장
[ ] s2에서 동일 commit 실행 확인
```

---

## 34. 연구 메시지와 직접 연결되는 최소 실험 세트

시간이 제한될 때 반드시 남길 실험은 다음이다.

### Mutual 최소 세트

```text
M0: R4 + R4 independent
M1: R4 ↔ R4 vanilla residual mutual
M2: R4-edge / R4-SiS independent
M3: R4-edge ↔ R4-SiS component mutual
```

### KD 최소 세트

```text
T0: c6 baseline
T1: c6 uncertainty
T2: c6 uncertainty + SiS
K0: R4 baseline
K1A: full-output KD
K1B: spectral-only KD
K2: uncertainty-aware spectral KD
K3: K2 + GT variance
K4: K3 + Student SiS
```

Feature KD는 K4까지 효과가 확인된 뒤 추가한다.

---

## 35. 최종 의사결정 트리

```text
R4↔R4 M1이 M0보다 좋음?
├─ Yes → 동일 구조 DML 자체가 restoration에서도 유효
│        └─ M3가 더 좋으면 loss specialization을 최종 method로 사용
└─ No  → M3가 M2보다 좋음?
         ├─ Yes → reconstruction에서는 component specialization이 필요
         └─ No  → mutual route는 baseline/negative result로 두고 KD 집중

c6→R4 K1B가 K1A보다 좋음?
├─ Yes → selective spectral KD 채택
└─ No  → full-output KD도 유지 가능하나 Ds 보존 확인

K2가 K1B보다 좋음?
├─ Yes → uncertainty routing 채택
└─ No  → teacher uncertainty calibration 재검토

K3가 K2보다 좋음?
├─ Yes → GT variance가 uncertainty와 다른 정보 제공
└─ No  → GT variance 제거

K4가 shift robustness를 높임?
├─ Yes → SiS를 최종 loss에 포함
└─ No  → shift range/정의 재검토 또는 SiS 제외
```

---

## 36. 참고 문헌과 구현 기준

1. **Deep Mutual Learning**, CVPR 2018  
   동일 구조와 이질 구조 peer가 서로의 예측을 학습하는 기본 개념.

2. **SIPSA-Net: Shift-Invariant Pan Sharpening with Moving Object Alignment for Satellite Imagery**, CVPR 2021  
   Sec. 3.3, Eq. (2)–(6): edge detail loss와 shifted MS 후보 최소값 기반 SiS loss.

3. **PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening**  
   Eq. (4): HRMS/PAN dual reconstruction MARs loss. MS/PAN mode-specific reconstruction이 현재 모든 실험의 anchor.

4. **U-Know-DiffPAN: An Uncertainty-aware Knowledge Distillation Diffusion Framework with Details Enhancement for PAN-Sharpening**  
   Eq. (16)–(19): uncertainty-weighted hard/soft/feature distillation. 현재 연구에서는 diffusion을 제외하고 loss routing 원리만 사용.

---

## 37. 즉시 구현 순서

```text
1. ModelOutput / common loss API
2. MTF downsampler
3. SiS + unit tests
4. edge / variance operators
5. mutual loss + gradient isolation test
6. s1 mutual trainer M0/M1
7. uncertainty head + calibration
8. U-Know weighting
9. s2 teacher/student trainer
10. M2/M3, T2, K2–K4 순서로 확장
```

핵심은 처음부터 모든 loss를 한 번에 결합하지 않는 것이다. **M0→M1→M2→M3**, **T0→T1→T2**, **K0→K1→K2→K3→K4** 순서를 유지해야 각 성능 변화의 원인을 분리할 수 있다.
