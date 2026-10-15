# s1 — Global Alignment 40시간 구현·실험 설계서

**작성 목적:** PanCollection/WV3에서 확인된 전역 sub-pixel PAN–LRMS 오차와 bicubic sampling-phase 오차를, 현재 선택한 `S1_T05_W152_D123_DUAL` backbone에 단계적으로 적용한다.  
**서버:** s1  
**총 실행 예산:** 최대 40시간  
**기준 모델:** `S1_T05_W152_D123_DUAL (50K) · w152 d123 11ch attn:0 nocrop`  
**연구 범위:** 이번 40시간에는 **global alignment만** 다룬다. Local flow, SIPSA FAM/PWPAC, SiS, KD, mutual learning, uncertainty, SE, Swin, CM3A는 모두 비활성화한다.

---

## 0. 이번 캠페인의 고정 결정

### 0.1 반드시 유지할 사항

```text
Dataset                 : WV3
Backbone                : width=152, depth=[1,2,3]
Attention / CM3A / Swin : 없음
Input                   : 11ch
Crop / scale jitter     : 없음
Normalization           : 기존 S1_T05와 동일
Task                    : dual MARs
MS mode                 : HRMS reconstruction
PAN mode                : PAN back-reconstruction
Mode modulation         : γMS, βMS, γPAN, βPAN 유지
PAN-loss weight         : 기존값 1.0 유지
Iteration               : 50,000
Optimizer               : AdamW
Base LR                 : 1e-4
Weight decay            : 0.01
Scheduler               : cosine
Warmup                   : 100 steps
Seed                     : 2025
Checkpoint priority      : 1) HQNR, 2) fSCC
```

### 0.2 이번 실험에서 바꾸지 않는 것

- PAN mode에는 output inverse transform을 적용하지 않는다.
- PAN reconstruction target은 기존 입력 PAN의 repeated tensor를 그대로 사용한다.
- local alignment는 구현하지 않는다.
- 별도의 spatial/edge/SiS loss를 추가하지 않는다.
- architecture width, depth, activation, normalization, skip path를 변경하지 않는다.
- shift estimator가 GT를 보지 않도록 한다.
- input과 output에 서로 독립적인 두 shift estimator를 두지 않는다.
- trainable case에서도 input shift와 output inverse는 **반드시 같은 \(\Delta\)** 와 \(-\Delta\) 관계를 사용한다.

---

## 1. 기준 모델과 현재 성능

### 1.1 기준 아키텍처

```text
Run       : S1_T05_W152_D123_DUAL
Width     : 152
Depth     : [1, 2, 3]  # full / H2 / H4
Attention : 0
Input     : 11ch
Crop      : False
Task      : dual MARs
Params    : 4.8935M
FLOPs     : 121.6G
Infer     : 9.39ms
Train     : 3.18h / 50K
```

11개 입력 채널은 다음과 같이 유지한다.

```text
8ch : upsampled LRMS
1ch : PAN
1ch : upsampled low-pass PAN
1ch : PAN - upsampled low-pass PAN
```

### 1.2 현재 기준 수치

| ERGAS↓ | SAM↓ | SCC↑ | Q2n↑ | D_lambda↓ | D_s↓ | HQNR↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 2.0893 | 2.8190 | 0.9907 | 0.9206 | 0.0227 | 0.0232 | 0.9546 |

이 행은 기존 bicubic 경로를 사용한 역사적 anchor다. 이후 모든 global-alignment case의 직접 비교 anchor는 새로 학습하는 **P0 phase-corrected baseline**으로 한다.

---

## 2. 실험이 답해야 하는 질문

이번 캠페인은 다음 질문을 순서대로 분리한다.

1. **Sampling phase만 바로잡아도 성능이 달라지는가?**
2. **PAN frame에서 정렬된 MS를 내부 처리한 뒤 최종 출력을 다시 MS frame으로 돌리면 이득이 있는가?**
3. **최종 출력은 PAN frame에 유지하고, GT loss용 copy만 MS frame으로 역변환하는 것이 더 나은가?**
4. **전체 shift의 일부만 input conditioning에 적용했을 때 HQNR–fSCC 절충점이 존재하는가?**
5. **사전 계산 shift 대신 작은 trainable global estimator로 같은 효과를 재현할 수 있는가?**

이번 결과로 local alignment 필요성까지 결론 내리지 않는다.

---

# 3. 좌표계와 부호 규약

## 3.1 두 좌표계

```text
M-frame : LRMS와 RR GT가 놓인 좌표계
P-frame : PAN이 놓인 좌표계
```

Audit에서 RR의 `LRMS ↔ GT`는 거의 정합되어 있었으므로, 기본 GT는 M-frame으로 본다. 최종 spatial reference는 PAN이므로, P-frame output을 별도로 정의한다.

## 3.2 Shift 부호

모든 코드와 cache는 다음 규약을 사용한다.

\[
\delta_{P\leftarrow M}=(d_y,d_x)
\]

이는 MS를 옮겨 PAN에 맞추는 양이다.

```python
aligned[y, x] = moving[y + dy, x + dx]
```

따라서:

```text
M → P forward transform : +Δ
P → M inverse transform : -Δ
```

Scale factor가 4이므로:

\[
\Delta_{HR}=4\Delta_{LR}
\]

이다.

## 3.3 절대 금지

- `scipy.ndimage.shift`와 `grid_sample`의 부호를 추정으로 섞지 않는다.
- input shift와 output inverse에 각기 다른 sign convention을 사용하지 않는다.
- augmentation 후 원래 shift vector를 그대로 사용하지 않는다.
- \((d_y,d_x)\)와 \((d_x,d_y)\)를 혼용하지 않는다.

모든 실행 전에 impulse test로 부호를 검증한다.

---

# 4. Phase-corrected bicubic: 모든 case의 공통 기본값

## 4.1 수정 이유

PanCollection의 MS 생성은 4배 격자에서 `phase=2` 중심을 사용한다. 기존:

```python
F.interpolate(ms, scale_factor=4, mode="bicubic", align_corners=False)
```

는 1.5 중심을 가정하므로 제공 `lms` 대비 약 \((-0.5,-0.5)\) output-pixel 편향이 발생한다.

이번 캠페인에서는 phase correction을 별도 옵션이 아니라 **항상 활성화된 기본값**으로 둔다.

## 4.2 직접 sampling 식

LRMS 크기를 \(H\times W\), HR/PAN 크기를 \(4H\times4W\)라고 한다. HR output coordinate를 \((v,u)\)라 하면, shift를 포함한 LR source coordinate는 다음과 같다.

\[
x_{src}=\frac{u-2}{4}+\alpha d_x
\]

\[
y_{src}=\frac{v-2}{4}+\alpha d_y
\]

- phase center: 2
- \(\alpha=0\): phase correction만
- \(\alpha=1\): full global MS→PAN shift
- \(0<\alpha<1\): partial shift

## 4.3 구현

`grid_sample`을 한 번만 호출하여 phase correction, 4× scaling, global shift를 합친다.

```python
def phase_shift_upsample(
    lr: Tensor,              # [B,C,H,W]
    delta_lr: Tensor,        # [B,2], order=(dy,dx)
    alpha: float | Tensor,
) -> Tensor:
    # out[v,u] = lr[(v-2)/4 + alpha*dy, (u-2)/4 + alpha*dx]
    B, C, H, W = lr.shape
    HH, WW = 4 * H, 4 * W

    yy, xx = meshgrid_hr(HH, WW, device=lr.device, dtype=lr.dtype)

    dy = delta_lr[:, 0].view(B, 1, 1)
    dx = delta_lr[:, 1].view(B, 1, 1)

    src_y = (yy[None] - 2.0) / 4.0 + alpha * dy
    src_x = (xx[None] - 2.0) / 4.0 + alpha * dx

    grid_y = 2.0 * src_y / max(H - 1, 1) - 1.0
    grid_x = 2.0 * src_x / max(W - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    return F.grid_sample(
        lr,
        grid,
        mode="bicubic",
        padding_mode="border",
        align_corners=True,
    )
```

### 구현 규칙

- `align_corners=True`는 위에서 명시한 integer source-coordinate를 그대로 normalized coordinate로 변환하기 위해 사용한다.
- 기존 `F.interpolate` 호출과 새 helper를 case마다 혼용하지 않는다.
- input MS는 native LRMS에서 HR로 직접 한 번 sampling한다.
- global warp 후 다시 bicubic upsampling하는 두 단계 처리를 금지한다.
- output inverse는 별도의 HR-grid warp이므로 Case 1/3/4에서는 두 번째 interpolation이 존재한다. 이를 별도 진단한다.

## 4.4 PAN-derived channels

PAN 쪽 3개 채널은 global shift하지 않는다.

```python
lpan_lr = mtf_downsample_phase2(pan)
lpan_hr = phase_shift_upsample(lpan_lr, zeros, alpha=0)
pan_hf  = pan - lpan_hr
```

11ch input은:

```python
x11 = cat([
    ms_condition_hr,  # 8ch; case별 shift 적용
    pan,              # 1ch; P-frame
    lpan_hr,          # 1ch; phase-corrected, shift 없음
    pan_hf,           # 1ch; P-frame
], dim=1)
```

이다.

## 4.5 Phase gate

학습 전에 다음을 통과해야 한다.

```text
phase-corrected upsample vs dataset lms:
  ZNCC                  >= 0.9999
  residual shift        <= 0.01 output pixel
  mean absolute diff    audit 재현 범위 이내

alpha=0:
  매 실행에서 완전히 같은 grid 생성
```

통과하지 않으면 어떤 50K run도 시작하지 않는다.

---

# 5. Global shift cache

## 5.1 Shift source

Case 1–3은 학습 가능한 aligner를 쓰지 않는다. 검증된 alignment-audit estimator의 결과를 sample별로 cache한다.

Estimator 입력:

```text
reference : MTF↓ PAN
moving    : native LRMS structural map
GT        : 사용 금지
```

Primary estimator:

```text
Scharr gradient magnitude
→ median/MAD normalization
→ top-30% edge mask
→ ZNCC search
→ 3×3 quadratic subpixel refinement
```

Secondary estimator:

```text
Census 5×5 + Hamming
```

## 5.2 WV3 train cache

가능하면 parent scene/tile ID를 사용한다.

```text
우선순위 1: parent scene 또는 큰 tile에서 Δ를 계산하고 소속 patch에 공유
우선순위 2: scene ID가 없으면 각 16×16 LR patch에서 계산
```

Per-patch fallback에서는 작은 patch로 인한 오추정을 막기 위해 아래 gate를 적용한다.

```text
search range            : ±1.0 LR pixel
boundary hit            : False
primary-secondary diff  : <= 0.25 LR pixel
peak margin             : >= 0.05
magnitude               : <= 1.0 LR pixel
```

하나라도 실패하면:

```text
accepted=False
delta_applied=(0,0)
```

으로 둔다. 실패 sample에 임의의 이웃 shift를 적용하지 않는다.

## 5.3 RR/FR evaluation cache

- RR/FR에서는 patch가 아니라 원본 test scene 전체에서 shift를 추정한다.
- 모든 20개 scene에 대해 cache한다.
- official HQNR selection은 index 12–19을 사용한다.
- cache 생성에는 PAN과 LRMS만 사용한다.
- 사전에 보고된 shift를 코드에 hard-code하지 않는다.

## 5.4 Cache schema

```text
outputs/global_shift_cache/
├── wv3_train.parquet
├── wv3_rr.parquet
├── wv3_fr.parquet
└── cache_meta.json
```

필수 column:

```text
split
sample_id
scene_id
dy_lr_raw
dx_lr_raw
magnitude_raw
peak_margin
primary_secondary_diff
boundary_hit
accepted
dy_lr_applied
dx_lr_applied
estimator_version
source_file_hash
```

`cache_meta.json`:

```json
{
  "sign_convention": "aligned[y,x]=moving[y+dy,x+dx]",
  "scale": 4,
  "search_radius_lr": 1.0,
  "mtf_sigma": 1.98,
  "decimation_phase": 2,
  "primary": "scharr_zncc",
  "secondary": "census5_hamming"
}
```

모든 run의 resolved config에 cache SHA256을 기록한다.

---

# 6. Augmentation과 shift vector

현재 `nocrop`은 유지하되 flip/90° rotation이 있으면 shift vector도 같이 변환한다.

Shift order는 `(dy, dx)`다.

| Augmentation | 변환된 shift |
|---|---|
| none | \((dy,dx)\) |
| horizontal flip | \((dy,-dx)\) |
| vertical flip | \((-dy,dx)\) |
| rot90 CCW, k=1 | \((-dx,dy)\) |
| rot90, k=2 | \((-dy,-dx)\) |
| rot90 CCW, k=3 | \((dx,-dy)\) |

여러 augmentation이 연속 적용되면 영상과 같은 순서로 vector transform을 합성한다.

필수 unit test:

```text
1. PAN/MS impulse pair에 known shift 부여
2. flip/rotation 적용
3. 변환된 delta로 alignment
4. impulse 위치가 정확히 일치하는지 확인
```

---

# 7. HR inverse warp

## 7.1 정의

HR tensor \(Y\)에 대한 warp operator는:

```python
out[y,x] = src[y + dy_hr, x + dx_hr]
```

로 정의한다.

```python
def warp_hr(x_hr: Tensor, delta_hr: Tensor) -> Tensor:
    # delta_hr order=(dy,dx)
    ...
```

P-frame → M-frame inverse는:

```python
y_ms = warp_hr(y_pan, -4.0 * delta_lr)
```

이다.

## 7.2 Border mask

Inverse warp가 있는 MS reconstruction loss는 invalid/border extrapolation을 제외한다.

Per-sample margin:

\[
m=\left\lceil4\max(|d_x|,|d_y|)\right\rceil+2
\]

- `+2`: bicubic support margin
- loss는 valid mask의 합으로 정규화한다.
- PAN loss에는 이 mask를 적용하지 않는다.
- RR/FR official metrics는 기존 `dim_cut` 규칙을 유지한다.

```python
loss = (abs(pred - gt) * mask).sum() / (mask.sum() * channels + eps)
```

---

# 8. 공통 모델 API

Architecture 코드를 case별로 복제하지 않는다. Alignment는 model 앞/뒤의 wrapper에서 처리한다.

```python
@dataclass
class AlignmentContext:
    delta_lr: Tensor       # [B,2]
    accepted: Tensor       # [B]
    alpha: Tensor | float
    frame: str             # "M" or "P"

@dataclass
class ForwardViews:
    ms_phase_hr: Tensor
    ms_cond_hr: Tensor
    ms_base_hr: Tensor
    lpan_hr: Tensor
    pan_hf: Tensor
    y_pan: Tensor | None
    y_ms: Tensor | None
    y_final: Tensor
```

Core model API:

```python
residual = model(
    x11,
    mode="MS" | "PAN",
)
```

Core U-Net 내부는 수정하지 않는다.

---

# 9. Dual MARs 처리 규칙

## 9.1 MS mode

Case별로 `ms_cond_hr`와 `ms_base_hr`만 달라진다.

```python
res_ms = model(x11, mode="MS")
pred   = ms_base_hr + res_ms
```

## 9.2 PAN mode

PAN mode는 모든 case에서 동일하다.

```python
res_pan = model(x11, mode="PAN")
pred_pan_rep = repeat(lpan_hr, bands) + res_pan
loss_pan = L1(pred_pan_rep, repeat(pan, bands))
```

### PAN mode에 하지 않는 것

- output inverse
- GT-frame transform
- global inverse loss
- separate shift prediction
- separate aligner

Input의 MS conditioning은 해당 case의 aligned/partial MS를 그대로 쓰지만, PAN loss의 output과 target은 P-frame에 유지한다.

## 9.3 Batch duplication

Shift는 duplication 이전에 sample당 한 번 얻고, 두 mode에 복제한다.

```python
delta_dual = torch.cat([delta_lr, delta_lr], dim=0)
mode       = ["MS"] * B + ["PAN"] * B
```

Trainable aligner case에서는 PAN mode로 들어가는 shift에 대해 aligner gradient를 차단한다.

```python
delta_pan_mode = delta_pred.detach()
```

PAN loss가 shift estimator를 임의의 방향으로 움직이지 않도록 하기 위함이다.

---

# 10. 40시간 실행 순서

## 10.1 전체 queue

| 순서 | Run ID | 핵심 설정 | 예상 시간 |
|---:|---|---|---:|
| 0 | 구현·cache·unit test | 공통 infrastructure | 4.5h |
| 1 | `GA_P0_PHASEFIX_W152_D123_DUAL` | phase correction만 | 3.3h |
| 2 | `GA_C1_FROZEN_RT_A100_W152_D123_DUAL` | frozen full shift + final inverse | 3.5h |
| 3 | `GA_C3_FROZEN_DUALFRAME_A100_W152_D123_DUAL` | frozen full shift + inverse loss view | 3.5h |
| 4 | `GA_C2_INPUTONLY_A050_W152_D123_DUAL` | input-only partial shift 0.50 | 3.3h |
| 5 | `GA_C2_INPUTONLY_A100_W152_D123_DUAL` | input-only full shift 1.00 | 3.3h |
| 6 | `GA_C2_INPUTONLY_A025_W152_D123_DUAL` | input-only partial shift 0.25 | 3.3h |
| 7 | `GA_C2_INPUTONLY_A075_W152_D123_DUAL` | input-only partial shift 0.75 | 3.3h |
| 8 | `GA_C4_TRAIN_RT_W152_D123_DUAL` | trainable shift + final inverse | 3.7h |
| 9 | `GA_C4_TRAIN_DUALFRAME_W152_D123_DUAL` | trainable shift + inverse loss view | 3.7h |
| 10 | 전 case 최종 평가·표·샘플 | HQNR/fSCC 및 frame 진단 | 3.1h |
| 11 | buffer | 실패 복구·runtime 편차 | 1.5h |

예상 총합은 약 40시간이다. 첫 1K iteration에서 실측 시간을 다시 계산한다.

```python
estimated_50k_hours = elapsed_1k_hours * 50
```

### 40시간 초과 시 drop 순서

아래 순서대로만 제거한다.

```text
1. C2 alpha=0.75
2. C2 alpha=0.25
3. C4 trainable round-trip
```

다음 case는 반드시 보존한다.

```text
P0
C1 frozen round-trip
C3 frozen dual-frame
C2 alpha=0.50
C2 alpha=1.00
C4 trainable dual-frame
```

---

# 11. P0 — Phase-corrected baseline

## 11.1 목적

기존 `S1_T05_W152_D123_DUAL` 대비 sampling phase correction 자체의 영향을 분리한다.

## 11.2 Forward

```python
delta = zeros([B,2])

ms_phase_hr = phase_shift_upsample(ms, delta, alpha=0)
lpan_hr     = phase_shift_upsample(lpan_lr, delta, alpha=0)
pan_hf      = pan - lpan_hr

x11 = cat([ms_phase_hr, pan, lpan_hr, pan_hf], dim=1)

res_ms = model(x11, mode="MS")
y_ms   = ms_phase_hr + res_ms

res_pan = model(x11, mode="PAN")
p_hat   = repeat(lpan_hr, bands) + res_pan
```

## 11.3 Loss

\[
L_{MS}=\|\hat Y_M-Y_{GT}\|_1
\]

\[
L_{PAN}=\|\hat P-P^{rep}\|_1
\]

\[
L=L_{MS}+L_{PAN}
\]

## 11.4 Output frame

```text
Final output : M-frame
RR metric    : y_ms vs GT
FR HQNR      : y_ms
fSCC         : y_ms vs PAN
```

## 11.5 판정

P0와 기존 S1_T05를 비교해 bicubic phase correction의 순수 효과를 기록한다. 이후 모든 case는 P0를 직접 anchor로 사용한다.

---

# 12. Case 1 — Frozen full global round-trip

**Run:** `GA_C1_FROZEN_RT_A100_W152_D123_DUAL`

## 12.1 질문

> MS를 PAN frame으로 정렬한 상태에서 U-Net이 feature를 처리하면, 최종 출력을 다시 원래 M-frame으로 돌리더라도 이득이 남는가?

## 12.2 Forward

```python
delta = load_cached_delta(sample_id)     # [B,2], P <- M
alpha = 1.0

ms_pan_hr = phase_shift_upsample(ms, delta, alpha=1.0)
lpan_hr   = phase_shift_upsample(lpan_lr, zeros, alpha=0)
pan_hf    = pan - lpan_hr

x11 = cat([ms_pan_hr, pan, lpan_hr, pan_hf], dim=1)

res_ms = model(x11, mode="MS")
y_pan  = ms_pan_hr + res_ms

y_ms   = warp_hr(y_pan, -4.0 * delta)
```

PAN mode:

```python
res_pan = model(x11, mode="PAN")
p_hat   = repeat(lpan_hr, bands) + res_pan
```

## 12.3 Loss

\[
L_{MS}=\operatorname{MaskedL1}(\hat Y_M,Y_{GT})
\]

\[
L_{PAN}=\|\hat P-P^{rep}\|_1
\]

\[
L=L_{MS}+L_{PAN}
\]

## 12.4 Output frame

```text
Deployed/final output : y_ms, M-frame
Internal diagnostic   : y_pan, P-frame
```

## 12.5 필수 기록

```text
HQNR(y_ms)
fSCC(y_ms, PAN)
HQNR_internal(y_pan)
fSCC_internal(y_pan, PAN)
round-trip interpolation control
```

## 12.6 해석

- P0보다 좋아지면: aligned internal processing 자체가 유효하다.
- internal `y_pan`은 좋아지지만 final `y_ms`가 나쁘면: output inverse가 spatial gain을 되돌린다.
- P0와 같으면: global shift가 backbone이 이미 흡수 가능한 수준이거나 round-trip 이득이 없다.
- 더 나쁘면: 두 번째 resampling 손실 또는 frame 왕복이 해롭다.

---

# 13. Case 3 — Frozen dual-frame loss

**Run:** `GA_C3_FROZEN_DUALFRAME_A100_W152_D123_DUAL`

## 13.1 질문

> Final HRMS는 PAN frame에 유지하면서, GT reconstruction loss만 inverse-warped M-frame view에서 계산하면 spatial/spectral 좌표 충돌을 줄일 수 있는가?

## 13.2 Forward

Case 1과 동일하게 full shift된 MS를 input과 residual base로 사용한다.

```python
delta = load_cached_delta(sample_id)

ms_pan_hr = phase_shift_upsample(ms, delta, alpha=1.0)
x11       = cat([ms_pan_hr, pan, lpan_hr, pan_hf], dim=1)

res_ms = model(x11, mode="MS")
y_pan  = ms_pan_hr + res_ms

# loss 전용 view
y_ms_loss = warp_hr(y_pan, -4.0 * delta)
```

## 13.3 Loss

\[
L_{MS}=\operatorname{MaskedL1}(\hat Y_{M,\text{loss}},Y_{GT})
\]

\[
L_{PAN}=\|\hat P-P^{rep}\|_1
\]

\[
L=L_{MS}+L_{PAN}
\]

별도 PAN edge loss는 이번 run에 넣지 않는다. PAN mode의 기존 auxiliary supervision만 유지한다.

## 13.4 Output frame

```text
Deployed/final output : y_pan, P-frame
GT-loss view          : y_ms_loss, M-frame
PAN-mode output       : P-frame, inverse 없음
```

## 13.5 평가

```text
FR HQNR / fSCC  : y_pan으로 계산
RR ERGAS/SAM    : y_ms_loss vs GT를 primary로 기록
RR diagnostic   : y_pan vs GT도 별도 suffix로 저장
```

파일 저장:

```text
predictions/
├── rr_msframe/
├── rr_panframe/
├── fr_panframe/
└── fr_msframe_diagnostic/
```

## 13.6 해석

- C1보다 HQNR/fSCC가 좋으면: final inverse가 spatial gain을 지웠다는 근거다.
- C1과 RR reconstruction은 비슷하지만 C3 FR이 좋으면: dual-frame loss 설계가 유효하다.
- \(D_\lambda\)만 악화하고 fSCC가 크게 좋아지면: 공식 metric의 좌표 충돌을 정량적으로 확인한 것이다.
- HQNR도 fSCC도 악화하면: full PAN-frame output이 현재 데이터/metric에 맞지 않는다.

---

# 14. Case 2 — Input-only partial global adjustment

## 14.1 핵심 구조

함수/역함수 쌍을 사용하지 않는다. Shift는 **conditioning MS에만** 적용하고, residual base와 GT frame은 기존 M-frame에 둔다.

\[
M_{\text{cond}}=\mathcal A(M;\alpha\Delta)
\]

\[
M_{\text{base}}=\mathcal A(M;0)
\]

\[
\hat Y_M=M_{\text{base}}+R_\theta(P,M_{\text{cond}})
\]

## 14.2 Forward

```python
delta = load_cached_delta(sample_id)

ms_cond_hr = phase_shift_upsample(ms, delta, alpha=ALPHA)
ms_base_hr = phase_shift_upsample(ms, zeros, alpha=0)

x11 = cat([ms_cond_hr, pan, lpan_hr, pan_hf], dim=1)

res_ms = model(x11, mode="MS")
y_ms   = ms_base_hr + res_ms
```

PAN mode:

```python
# input conditioning은 동일 alpha를 사용
res_pan = model(x11, mode="PAN")
p_hat   = repeat(lpan_hr, bands) + res_pan
```

## 14.3 Loss

\[
L=\|\hat Y_M-Y_{GT}\|_1+\|\hat P-P^{rep}\|_1
\]

Inverse transform과 border mask는 없다.

## 14.4 Alpha sweep

P0가 \(\alpha=0\) 역할을 한다. 신규 run:

| 실행 순서 | Run | \(\alpha\) |
|---:|---|---:|
| 1 | `GA_C2_INPUTONLY_A050...` | 0.50 |
| 2 | `GA_C2_INPUTONLY_A100...` | 1.00 |
| 3 | `GA_C2_INPUTONLY_A025...` | 0.25 |
| 4 | `GA_C2_INPUTONLY_A075...` | 0.75 |

이 순서는 0.5와 1.0에서 먼저 경향을 확인하고, 이후 양옆을 채우기 위함이다.

## 14.5 Output frame

```text
Final output : M-frame
Conditioning : alpha만큼 P-frame 쪽으로 이동
Residual base: M-frame
```

## 14.6 해석

- 최적 \(\alpha=0\): input global shift가 필요하지 않다.
- 최적 \(\alpha=1\): full aligned conditioning이 유리하다.
- 최적 \(0<\alpha<1\): metric/optimization이 부분 정합을 선호한다.
- 중간 \(\alpha\)가 좋더라도 이를 물리적 정답 위치라고 주장하지 않는다.
- alpha curve는 `HQNR`, `fSCC`, `D_lambda`, `D_s` 네 값으로 그린다.

---

# 15. Case 4 — Trainable global shift

Case 4는 frozen shift가 유효하다는 전제 없이도 실행하되, 최종 해석은 Case 1/3과 함께 한다.

## 15.1 ShiftNet 입력

Raw intensity를 직접 맞추지 않고 audit과 동일하게 구조 map을 사용한다.

```python
pan_lr = mtf_downsample_phase2(pan)
g_pan  = robust_norm(scharr_magnitude(pan_lr))
g_ms   = robust_norm(mean_band_scharr(ms))

shift_input = cat([g_pan, g_ms], dim=1)  # 2ch
```

## 15.2 ShiftNet 구조

```python
class GlobalShiftNet(nn.Module):
    def __init__(self, max_shift_lr=1.0):
        self.body = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.max_shift_lr = max_shift_lr

    def forward(self, x):
        z = self.body(x).flatten(1)
        return self.max_shift_lr * torch.tanh(self.head(z))
```

Output order는 반드시 `(dy,dx)`다.

예상 추가 파라미터는 수만 개 수준이며 build 후 실측한다.

## 15.3 ShiftNet pseudo-label pretraining

C4 두 run에 동일 pretrained ShiftNet checkpoint를 사용한다.

```text
Data          : WV3 train cache
Target        : accepted sample의 cached Δ
Rejected      : shift target으로 사용하지 않음
Batch         : 128
Optimizer     : AdamW
LR            : 1e-3
Weight decay  : 1e-4
Steps         : 2,000
Validation    : sample_id hash 기준 10%
Loss          : SmoothL1(beta=0.05 LR pixel)
```

통과 조건:

```text
accepted validation median error <= 0.10 LR pixel
P90 error                         <= 0.25 LR pixel
sign accuracy                     >= 95%
```

통과하지 못해도 C4를 무조건 실행하지 않는다. 원인을 수정한 뒤 한 번 재시도하고, 재실패하면 C4를 취소하고 남은 시간을 evaluation에 사용한다.

## 15.4 Joint training loss

Accepted sample:

\[
L_{\text{shift}}=
\operatorname{SmoothL1}(\Delta_\phi,\Delta_{\text{cache}})
\]

Rejected sample:

\[
L_{\text{zero}}=\|\Delta_\phi\|_1
\]

전체:

\[
L=L_{MS}+L_{PAN}
+0.1L_{\text{shift}}
+0.01L_{\text{zero}}
\]

Optimizer param groups:

```yaml
backbone:
  lr: 1.0e-4
  weight_decay: 0.01

shift_net:
  lr: 1.0e-5
  weight_decay: 1.0e-4
```

- reconstruction gradient는 MS mode에서 ShiftNet까지 전달한다.
- PAN mode에서는 `delta_pred.detach()`를 사용한다.
- independent output shift head를 만들지 않는다.
- correlation loss는 이번 캠페인에서 비활성화한다.

---

## 15.5 C4-A — Trainable round-trip

**Run:** `GA_C4_TRAIN_RT_W152_D123_DUAL`

```python
delta_pred = shift_net(shift_input)

ms_pan_hr = phase_shift_upsample(ms, delta_pred, alpha=1.0)
y_pan     = ms_pan_hr + model(x11, mode="MS")
y_ms      = warp_hr(y_pan, -4.0 * delta_pred)

loss_ms   = masked_l1(y_ms, gt)
```

Final output:

```text
y_ms, M-frame
```

PAN mode inverse 없음.

---

## 15.6 C4-B — Trainable dual-frame

**Run:** `GA_C4_TRAIN_DUALFRAME_W152_D123_DUAL`

```python
delta_pred = shift_net(shift_input)

ms_pan_hr = phase_shift_upsample(ms, delta_pred, alpha=1.0)
y_pan     = ms_pan_hr + model(x11, mode="MS")
y_ms_loss = warp_hr(y_pan, -4.0 * delta_pred)

loss_ms   = masked_l1(y_ms_loss, gt)
```

Final output:

```text
y_pan, P-frame
```

PAN mode inverse 없음.

## 15.7 C4 필수 진단

```text
median |Δpred - Δcache|
P90 |Δpred - Δcache|
corr(dy_pred, dy_cache)
corr(dx_pred, dx_cache)
zero/collapse ratio
boundary saturation ratio: |delta| >= 0.95
mean predicted magnitude
HQNR / fSCC
```

판정 경고:

```text
median error > 0.25 LR pixel          -> aligner 의미 불안정
saturation ratio > 10%                -> max range 또는 loss 문제
predicted shift가 거의 전부 0         -> aligner collapse
품질은 개선하지만 audit shift와 무관   -> alignment 외 shortcut 가능성
```

---

# 16. 공통 training config

```yaml
experiment:
  dataset: WV3
  iterations: 50000
  seed: 2025

model:
  width: 152
  depths: [1, 2, 3]
  attn_locations: []
  input_channels: 11
  mars: dual
  mode_modulation: true
  pan_loss_weight: 1.0

data:
  crop: false
  horizontal_flip: true
  vertical_flip: true
  rotation_90: true
  scale: 4

optimizer:
  name: AdamW
  lr: 1.0e-4
  weight_decay: 0.01

scheduler:
  name: cosine
  warmup_steps: 100
  total_steps: 50000

alignment:
  local_enabled: false
  sis_enabled: false
  phase_corrected_bicubic: true
  phase: 2
  padding_mode: border
  max_global_shift_lr: 1.0
```

각 case는 아래 5개 key만 달라져야 한다.

```yaml
alignment.delta_source: zero | cache | trainable
alignment.alpha: 0.0 | 0.25 | 0.50 | 0.75 | 1.0
alignment.output_frame: M | P
alignment.inverse_location: none | final_output | loss_branch
alignment.trainable_shift_net: true | false
```

---

# 17. Checkpoint와 best model 선정

사용자가 정한 우선순위를 그대로 따른다.

## 17.1 후보 checkpoint

```text
10K, 15K, 20K, 25K, 30K, 35K, 40K, 45K, 50K
```

각 checkpoint에서 FR index 12–19의 official HQNR와 fSCC를 계산한다.

## 17.2 선정 규칙

```python
HQNR_TIE_EPS = 1e-4
FSCC_TIE_EPS = 1e-4

1. unrounded mean HQNR 최대
2. HQNR 차이가 <= 1e-4이면 fSCC 최대
3. fSCC도 <= 1e-4 동률이면 더 나중 iteration
```

파일:

```text
best_hqnr.pt
best_hqnr_meta.json
last.pt
```

`best_hqnr_meta.json` 필수 항목:

```json
{
  "iteration": 45000,
  "hqnr": 0.0,
  "fscc": 0.0,
  "d_lambda": 0.0,
  "d_s": 0.0,
  "final_frame": "P",
  "shift_source": "cache",
  "alpha": 1.0,
  "cache_sha256": "..."
}
```

---

# 18. Frame별 평가 규칙

| Case | Final FR output | RR GT metric view | fSCC view |
|---|---|---|---|
| P0 | M-frame | M-frame | final M-frame |
| C1 frozen RT | M-frame | M-frame | final M-frame |
| C2 input-only | M-frame | M-frame | final M-frame |
| C3 frozen dual-frame | P-frame | inverse M-frame copy | final P-frame |
| C4 train RT | M-frame | M-frame | final M-frame |
| C4 train dual-frame | P-frame | inverse M-frame copy | final P-frame |

Dual-frame case는 반드시 두 view를 모두 저장한다.

## 18.1 필수 metric

```text
Primary selection:
  HQNR
  fSCC

Diagnostic:
  D_lambda
  D_s
  ERGAS
  SAM
  SCC
  Q2n
  PSNR
  SSIM
```

## 18.2 Latency

Frozen cache case는 두 시간을 분리한다.

```text
network_only_latency
external_shift_estimator_latency
end_to_end_latency
```

Trainable case는 ShiftNet을 포함한 end-to-end latency를 기록한다.

---

# 19. 시각 샘플

각 case의 best checkpoint에서 동일한 FR scene/crop을 저장한다.

## 19.1 고정 scene

```text
WV3 FR index 12–19 전체
```

## 19.2 필수 panel

```text
PAN
original LRMS RGB
phase-corrected LRMS
globally aligned LRMS
model final HRMS
inverse/loss-view HRMS (해당 case)
PAN–output edge overlay
shift vector (dy,dx)
```

## 19.3 대표 샘플

세 유형을 각각 최소 3개 저장한다.

```text
small shift   : |Δ| bottom 25%
typical shift : median 부근
large shift   : |Δ| top 25%
```

C4는 같은 샘플에:

```text
cached Δ
predicted Δ
difference
```

를 함께 표기한다.

---

# 20. Round-trip interpolation control

Case 1/3/4의 inverse warp가 blur를 만드는 정도를 네트워크와 분리해 측정한다.

```python
m0 = phase_shift_upsample(ms, zeros, alpha=0)
mp = phase_shift_upsample(ms, delta, alpha=1)
mr = warp_hr(mp, -4*delta)
```

기록:

```text
PSNR(mr, m0)
fSCC(mr, m0)
gradient-energy ratio
mean absolute difference
```

이 값이 case 성능 변화와 같은 크기라면, network alignment 이득보다 interpolation loss가 지배적일 수 있다.

---

# 21. Logging

각 step의 EMA를 기록한다.

```text
loss_total
loss_ms
loss_pan
loss_shift
loss_zero
delta_dy_mean/std
delta_dx_mean/std
delta_mag_p50/p90
accepted_ratio
valid_mask_ratio
```

Checkpoint evaluation:

```text
hqnr
fscc
d_lambda
d_s
rr_ergas
rr_sam
rr_scc
final_frame
```

Run directory:

```text
work_dirs/<run_id>/
├── config_resolved.yaml
├── code_commit.txt
├── shift_cache_meta.json
├── train_log.jsonl
├── checkpoint_metrics.csv
├── scene_metrics.csv
├── best_hqnr.pt
├── last.pt
├── predictions/
└── visualizations/
```

Resume 시 다음을 모두 복원한다.

```text
backbone
shift_net
optimizer
scheduler
AMP scaler
iteration
RNG states
best HQNR / fSCC
cache hash
```

---

# 22. Unit / smoke test

## 22.1 필수 unit test

```text
T01 phase mapping
  output center 4j+2가 LR pixel j를 정확히 참조

T02 provided-lms reproduction
  ZNCC >= 0.9999
  residual shift <= 0.01 HR px

T03 sign convention
  known dx/dy impulse가 forward shift 후 일치

T04 inverse
  +Δ 후 -Δ가 원위치로 복귀

T05 augmentation
  hflip/vflip/rot90 후 vector 변환 검증

T06 alpha=0
  alignment path가 P0와 동일

T07 alpha=1
  cache delta가 그대로 sampling grid에 반영

T08 PAN mode
  output inverse 호출 횟수 0

T09 mode duplication
  MS/PAN 두 half가 동일 delta를 사용

T10 gradient
  frozen case: delta/cache에 gradient 없음
  trainable case: MS loss와 L_shift에서 ShiftNet gradient 존재
  PAN loss에서는 ShiftNet gradient 차단
```

## 22.2 1K smoke

각 run 시작 전:

```text
NaN / Inf 없음
output shape 정상
loss_ms, loss_pan non-zero
phase helper 사용 여부 확인
delta sign sample visual 확인
peak memory
iteration time
pred range
```

Smoke 실패 시 50K를 진행하지 않는다.

---

# 23. 결과 해석 표

| 비교 | 답하는 질문 |
|---|---|
| old S1_T05 vs P0 | bicubic phase correction 효과 |
| P0 vs C1 | aligned internal processing + round-trip 효과 |
| C1 vs C3 | final inverse가 spatial gain을 없애는가 |
| P0 vs C2 alpha curve | partial conditioning shift의 절충점 |
| frozen vs trainable RT | ShiftNet이 external estimator를 재현하는가 |
| frozen vs trainable dual-frame | trainable alignment가 최종 method로 성립하는가 |

## 23.1 가능한 결론

### P0만 개선

```text
주된 문제는 learnable/global alignment보다 bicubic phase였다.
```

### C1 개선, C3 비개선

```text
내부 정합은 유효하지만 final output은 M-frame으로 복귀하는 것이 metric상 유리하다.
```

### C3 개선

```text
PAN-frame output과 MS-frame GT loss를 분리하는 dual-frame formulation이 유효하다.
```

### C2 중간 alpha가 최적

```text
공식 HQNR/fSCC는 full physical alignment보다 partial conditioning을 선호한다.
```

단, 이를 물리적 정답 shift라고 해석하지 않는다.

### C4가 frozen case를 재현

```text
입력 기반 adaptive global shift estimation을 model 내부에 통합할 수 있다.
```

### C4 품질은 좋지만 audit shift와 불일치

```text
ShiftNet이 실제 registration 대신 reconstruction shortcut을 학습했을 가능성이 있다.
```

---

# 24. 이번 캠페인에서 주장하지 않을 것

- local object motion을 해결했다고 주장하지 않는다.
- SIPSA FAM을 대체했다고 주장하지 않는다.
- predicted global shift를 physical sensor offset으로 확정하지 않는다.
- HQNR 최적 alpha를 true geometry라고 주장하지 않는다.
- C3의 P-frame output을 원래 misaligned LRMS와 직접 비교한 \(D_\lambda\)만으로 실패 판정하지 않는다.
- trainable shift가 audit pseudo-label과 무관해도 단순 성능 향상만으로 alignment 성공이라 부르지 않는다.

---

# 25. Definition of Done

40시간 종료 시 다음 산출물이 있어야 한다.

```text
[ ] phase-corrected upsampler와 10개 unit test
[ ] WV3 train/RR/FR global shift cache와 hash
[ ] P0, C1, C3 best checkpoint
[ ] C2 alpha curve 최소 0/0.5/1.0
[ ] C4 trainable dual-frame checkpoint
[ ] 모든 run의 HQNR→fSCC best selection
[ ] D_lambda / D_s 분해
[ ] frame별 RR/FR output 저장
[ ] round-trip interpolation control
[ ] frozen vs predicted shift 오차
[ ] 대표 scene panel
[ ] 최종 comparison CSV와 markdown summary
```

최종 우선순위는 다음과 같다.

```text
1. HQNR 최대
2. HQNR 동률 시 fSCC 최대
3. 나머지 지표는 원인 해석용
```

---

# 26. 최종 실행 요약

```text
P0  : bicubic phase correction
C1  : frozen global alignment → U-Net → final inverse
C3  : frozen global alignment → U-Net → PAN-frame final
      └─ inverse copy에서만 GT loss
C2  : alpha × frozen global shift를 input conditioning에만 적용
      └─ residual base와 final output은 M-frame
C4A : trainable global shift → U-Net → final inverse
C4B : trainable global shift → U-Net → PAN-frame final
      └─ inverse copy에서만 GT loss
```

PAN mode는 모든 case에서:

```text
aligned/adjusted input
→ shared U-Net PAN mode
→ PAN back-reconstruction
→ input PAN target
```

으로 유지하며, **output inverse를 적용하지 않는다.**
