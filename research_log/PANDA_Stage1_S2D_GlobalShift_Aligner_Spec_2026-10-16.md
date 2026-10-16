# PANDA — S2D Global-Shift PAN Aligner 구현 기술서 (A안)

- **기준일:** 2026-10-16
- **상태:** 신규 아키텍처 **제안 / 미구현 / 미검증**. 기존 `DirectPANAligner`를 이미 교체했다는 의미가 아님.
- **목표:** 기존 PANDA의 **sample-wise global translation** `[B,2]`를 유지하고, 기존 `stride-2 Conv ×2` 기반 특징 추출을 `S2D ×4 + stride-1 Conv`로 치환한다. **D2S 및 displacement map 없음.**
- **원본 기준:** `hojunking/PAN-Crafter-repro`의 `directshift48/method.py`, `pa/aligner.py`, `pa/warp.py`, `model/pancrafter_paper.py`.

## 1. 입력과 출력

| 표기/변수 | 크기 | 정의 |
|---|---|---|
| `B` | 정수 | Batch 내 패치 수 |
| `C` | 정수 | MS 밴드 수: WV3=8, GF2/QB=4 |
| `H, W` | 정수 | Crop 이전 현재 PAN 해상도에서의 영상 높이/너비, **각각 4의 배수** |
| `P` | `[B,1,H,W]` | 원본 PAN. aligner 관측 입력이자 최종 warp 원본 |
| `L_A` | `[B,C,H,W]` | PAN-grid에 **저장된 native/stored LMS**. Aligner 입력; 새로 bicubic 생성한 MS와 혼동 금지 |
| `u=(dy,dx)` | `[B,2]` | 샘플당 하나의 2D shift. **현재 PAN 그리드 픽셀 단위** |
| `P_tilde` | `[B,1,H,W]` | 보정 PAN: `warp_pan(P,u)`; 영상 크기 유지 |
| `F_S2D` | - | `torch.nn.functional.pixel_unshuffle(x, downscale_factor=4)` |

**알고리즘 출력:** `u=A_{S2D}(P,L_A)`.  
**Warp 정의:** `P_tilde[b,y,x] = P[b,y+dy_b,x+dx_b]`를 FP32 bicubic `grid_sample`(border, `align_corners=False`)로 계산한다. **PAN 전체를 단 한 번 warp**하며 LMS/MS는 이동하지 않는다. FR에서 shift에 `×4`를 적용하지 않는다. Shift를 `tanh`, `clamp`, `[-3,3]`에 가두지 않는다.

## 2. 고정할 아키텍처 (구현 선택값)

**설계 결정:** 두 입력의 정규화된 원본 강도에 **각각** S2D를 먼저 적용하고, 그 뒤 기존 Dual Stem → Concat → Conv → ResBlock → GAP → FC 방식을 사용한다. S2D의 정확한 위치를 이 문서에서 새로 고정한다. 과거 별도 displacement-map 설계의 세부 레이어를 재현했다는 주장은 하지 않는다.

```text
PAN P [B,1,H,W]                          Stored LMS L_A [B,C,H,W]
       │                                             │
    Crop8                                         Crop8
       │                                             │
Detached-stat z-norm                       Detached-stat z-norm
       │                                             │
S2D ×4 (pixel_unshuffle)                    S2D ×4 (pixel_unshuffle)
       │                                             │
[B,16,(H-16)/4,(W-16)/4]                  [B,16C,(H-16)/4,(W-16)/4]
       │                                             │
Reflect Conv3×3 16→16, GN4, SiLU           Reflect Conv3×3 16C→16, GN4, SiLU
       └───────────────────┬─────────────────────────┘
                           │ channel concat → 32
                    Reflect Conv3×3 32→32, stride1, GN8, SiLU
                           │
                    Reflect Conv3×3 32→64, stride1, GN8, SiLU
                           │
                    Common pre-LN ResBlock(64) ×1
                    (mode_mod=False, se=False, dropout=0)
                           │
                    Global Average Pooling (H',W' axes)
                           │ [B,64]
                    Linear 64→32, SiLU
                           │ [B,32]
                    Linear 32→2, zero-initialized W/b
                           │ [B,2]
                        u=(dy,dx)
                           │
             Warp **original full** PAN P with u
                           │
                     P_tilde [B,1,H,W]
```

- `Crop8`: 4변에서 각각 8픽셀을 **Z-Norm 및 S2D 이전**에 제거. **CNN의 관측창 한정**이지 GT reconstruction loss mask가 아니다.
- `z-norm`: batch 내 각 sample, 입력 원래 channel의 공간 평균/분산으로 정규화; 통계는 `detach`. **S2D 이후 16개 phase-channel별로 따로 z-norm하지 않는다.**
- `S2D`: **비학습** 공간-채널 재배열. PAN 1→16ch, LMS C→16Cch. Factor=4, PyTorch `pixel_unshuffle`과 동일한 위상 순서로 고정.
- `Conv3×3`: 1px reflection padding, convolution bias 없음. `GN4/GN8`은 각각 GroupNorm group 수가 4/8임을 의미. `SiLU` 활성화 사용.
- `ResBlock`: **현재 DirectShift의** `model.pancrafter_paper.ResBlock(64, out_channels=64, norm='ln', dropout=0., mode_mod=False, se=False)` 재사용. 기존 custom GN `_Res`로 되돌리지 않는다.
- `FC`: `Linear(64,32)→SiLU→Linear(32,2)`. 마지막 FC 가중치/bias만 0 초기화해 시작 shift=0. Head 추가 출력·flow decoder·D2S 없음.

### Shape 검산

| 단계 | RR 학습: H=W=64 | FR 추론: H=W=512 |
|---|---|---|
| 원본 PAN / LMS | `[B,1,64,64]` / `[B,C,64,64]` | `[B,1,512,512]` / `[B,C,512,512]` |
| Crop8 | 공간 `48×48` | 공간 `496×496` |
| S2D4 PAN | `[B,16,12,12]` | `[B,16,124,124]` |
| S2D4 LMS | `[B,16C,12,12]` | `[B,16C,124,124]` |
| Stems → Concat | `[B,32,12,12]` | `[B,32,124,124]` |
| Joint Conv×2 → ResBlock | `[B,64,12,12]` | `[B,64,124,124]` |
| GAP → FC | `[B,64] → [B,2]` | `[B,64] → [B,2]` |
| PAN warp output | `[B,1,64,64]` | `[B,1,512,512]` |

`H-16, W-16`은 각각 4로 나누어져야 한다. 입력/중간 feature는 FP32, 동일 device, finite를 검사하고 유효하지 않으면 optimizer update 전에 실패한다.

## 3. 최소 구현 로직 (참조 의사 코드)

새 네임스페이스/클래스(예: `s2dshift48/aligner.py::S2DGlobalPANAligner`)를 생성한다. 기존 `directshift48` 파일·체크포인트는 수정/덮어쓰기하지 않는다. 실제 Python 구현 시 기존 타입 검증·checkpoint identity 규약과 연결한다.

```python
# forward 핵심 흐름 — 현존 _conv/znorm/ResBlock/warp_pan 재사용
# __init__:
#   pan_stem = _conv(16, 16, stride=1, groups=4)
#   ms_stem  = _conv(16*C, 16, stride=1, groups=4)
#   joint1   = _conv(32, 32, stride=1, groups=8)
#   joint2   = _conv(32, 64, stride=1, groups=8)
#   res      = 기존 DirectShift pre-LN ResBlock(64)
#   fc1      = nn.Linear(64, 32)
#   fc2      = nn.Linear(32, 2)  # weight/bias zero-init

def forward(pan, stored_lms, *, return_details=False):
    # [B,1,H,W], [B,C,H,W] 및 FP32/device/finite/4배수 검증
    p = pan[..., 8:-8, 8:-8]
    m = stored_lms[..., 8:-8, 8:-8]
    p = F.pixel_unshuffle(znorm(p), 4)
    m = F.pixel_unshuffle(znorm(m), 4)
    x = torch.cat((self.pan_stem(p), self.ms_stem(m)), dim=1)
    x = self.res(self.joint2(self.joint1(x)), None)
    features = x.mean(dim=(2, 3))
    hidden = F.silu(self.fc1(features))
    u = self.fc2(hidden)
    if return_details:
        return u, {"features": features,
                   "head_input": features, "head_hidden": hidden}
    return u

# 학습/추론 호출 측: P_tilde = warp_pan(original_pan, u)
# PAN crop view가 아니라 original_pan을 warp해야 한다.
```

`forward` 시그니처·return_details의 key를 기존과 동일하게 유지할 것. 새 aligner의 `manifest()`/checkpoint revision 및 fingerprint에 **S2D4, before-stem, both-modalities, normalized-before-S2D, joint stride1**을 명시하고, 기존 DirectShift weight를 이 모델에 silently load하는 행위를 금지한다. 기존 `DirectShiftStage1`의 `isinstance(DirectPANAligner)` 검사, `MaskedDonorStage1` wrapping, donor binding은 새 클래스에서 명시적으로 호환 처리해야 한다. 단순히 class 이름만 바꾸면 type guard 또는 stale calibration에서 실패할 수 있다.

## 4. Stage 1 학습 및 Stage 2 연결: 변경하지 않는 부분

Stage 1은 새 A와 **동일한 W112/D(1,2,3)/P0 Teacher**를 fresh로 학습하는 독립 donor 실험으로 취급한다. 다음 기존 손실 정의와 계수/스케줄은 비교군과 동일하게 고정한다.

\[
\mathcal L_{\mathrm{S1}}=\mathcal L_{\mathrm{rec}}+
\lambda_{\mathrm{abs}}\mathcal L_{\mathrm{abs}}+
\lambda_{\epsilon}\mathcal L_{\epsilon}.
\]

- `L_rec`: 원래 Stage1 기준 전체 밴드·전체 픽셀 GT L1 (선택한 비교 baseline이 MASK2이면 **양쪽 모델 모두 같은 MASK2 정책** 적용).
- `L_abs`: 보정 PAN과 GT RGB 평균의 Scharr absolute 응답 차이, 고정 ROI8; 기존 연산자/스케일/gradient routing 유지.
- `L_eps`: PAN에만 합성 ε 이동을 준 `u_eps + eps ≈ stopgrad(u_0)` 일관성, ε는 반경 3px 원판 내부 면적 균등 샘플링. shifted branch에 Teacher 재구성 없음.
- `L_rec`은 Teacher+Aligner를, `L_abs/L_eps`는 Aligner를 갱신한다. Aligner의 입력은 stored LMS; Teacher의 residual base는 별도 bicubic MS.
- Stage 2에서는 **새로 학습한** Aligner+Teacher를 동결하고 Student만 학습한다. 기존 PANDA Hard/Soft 및 q ECDF를 재사용할 수 있으나 `tau_e`, q 분포와 `r_q` ECDF는 **새 donor에 대해 train-only로 재보정**해야 한다. 기존 donor30 checkpoint/calibration 재사용 금지.

## 5. 기존 구조 대비 효과/우려 (가설, 미검증)

| 비교 항목 | 기존 DirectShift | A안 S2D Global Shift | 해석 |
|---|---|---|---|
| 4배 공간 축소 | 학습 가능한 stride-2 Conv 두 번 | 비학습 S2D4 한 번 | S2D는 입력 재배열 자체로 정보를 버리지 않음. **이후 압축 stem은 정보를 버릴 수 있음** |
| 초반 특징 추출 | 48²→24²→12² | 48²→12², 이후 12² 고정 | Joint 첫 Conv의 실행 공간이 작아짐; 실제 속도는 측정 필요 |
| PAN 입력 채널 | stem에서 1→16 | S2D로 16→stem 16 | 4×4 phase 정보를 명시적으로 분리 |
| LMS 입력 채널(WV3) | stem에서 8→16 | S2D로 128→stem 16 | Stem의 파라미터 증가·압축 부담 |
| Shift 예측 | GAP→FC→2 | 동일 | output·warp·Stage 1 목표 유지 |
| 예측 구조 | global (dy,dx) | global (dy,dx) | dense flow/D2S와 별개 |

**기대 효과(가설):** S2D 단계에서 4×4 내 pixel phase를 채널로 옮겨 **초기 stride-conv sampling에 의한 세부 위치 정보 손실 가능성을 줄인다**. 작은 PAN–MS 위치차 추정에 도움이 될 수 있다. CNN의 주요 spatial 연산을 12×12에서 수행하여 MACs가 줄 여지도 있다.

**주요 우려:**

1. **Phase 민감성:** PAN이 1px 움직일 때 S2D 채널 내 phase 배치가 크게 바뀌어, 기존 stride-conv보다 오히려 shift-equivariance를 학습하기 어려울 수 있다. **ε-consistency 오차**로 검증.
2. **Stem 압축:** WV3 LMS가 **128→16 채널**로 바로 축소된다. S2D는 무손실이지만 stem은 아니므로 spectral/edge 정보가 소실될 수 있다.
3. **넓어진 실제 receptive field:** S2D 후 3×3 Conv는 원영상의 4px 간격 grid 이웃을 보므로 동일 3×3 커널 수라도 기존 모델과 유효 수용영역이 달라진다. 미세 정합/국소 물체에는 오히려 불리할 수 있다.
4. **모델 크기/비용 혼선:** 같은 ResBlock·FC를 유지한다는 가정의 계산상 WV3 파라미터 수는 **기존 약 105,458 → A안 약 124,898 (+18.4%)**이다. 이는 코드 기반 계산 예상치이며 실측 `sum(p.numel())`와 FLOPs/latency를 별도 기록해야 한다. 다른 모델과의 품질 개선을 S2D의 효과만으로 단정하지 않는다.
5. **RR–FR 일반화:** RR 12×12 feature와 FR 124×124 feature에서 GAP의 통계가 달라질 수 있다. 원래 모델도 입력 크기 차이가 있지만 S2D phase-channel의 분포 차이가 추가된다. FR HQNR의 `D_s, D_lambda`를 같이 확인.

## 6. 구현 완료 조건 및 최소 검증

1. **형상/순서:** WV3/GF2, RR64/FR512에서 위 Shape 정확히 일치. `pixel_shuffle(pixel_unshuffle(x,4),4)==x` 검산. **Crop→z-norm→S2D** 순서 unit test.
2. **인터페이스:** `u.shape == [B,2]`, finite FP32; `return_details` 하위 key 동일; FC2 zero-init 출력 `u=0`; 원본 PAN warp 크기 불변.
3. **동결/gradient:** `L_rec` 경로에서 A와 T 양쪽 gradient, `L_abs/L_eps`에서 A만; Stage 2는 frozen A/T, Student만. 입력 LMS 이동/GT 입력 누설 없음.
4. **경계/좌표:** Crop8 유지; Stage1 Scharr ROI8 및 warp support guard 유지. Shift `(dy,dx)` 부호, PAN pixel 단위, bicubic/border/`align_corners=False`, RR·FR 무스케일 동일.
5. **체크포인트:** 새 버전 identity, old/new strict cross-load 실패; donor hash, train split fingerprint, sensor, architecture, calibration binding 확인.
6. **비교 실험:** 기존 DirectShift vs A안을 **동일 Teacher 구조, 학습 update 수, seed/data order, ε 분포, loss 계수, mask 정책**에서 Stage 1부터 각각 독립 재학습. 각각 동일 checkpoint의 RR ERGAS와 FR HQNR·`D_s`·`D_lambda`을 기록. `u` 분포와 ε consistency 및 latency 동시 보고.

### 비범위

- D2S, `[B,2,H,W]` displacement map, trainable Student Aligner, LPAN/HPAN branch 추가, Stage2 loss 공식을 이번 변경과 동시에 재설계하는 일은 **포함하지 않는다**.
- **이 기술서는 구현 계약이며 아직 실험으로 S2D 우위가 확인된 것은 아니다.**
