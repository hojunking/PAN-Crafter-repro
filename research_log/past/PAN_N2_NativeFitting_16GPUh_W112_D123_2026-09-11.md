# N2 정합 능력 보존 + Native HRMS fitting: 16 GPU-hour 실행 제안

- 작성일: 2026-09-11
- Protocol 제안: `NF16_W112D123_N2LAST_v1`
- 예산: **추가 총 16 GPU-hours**. 단일 GPU 순차 실행 기준이며, 서버 3대에 각각 16시간을 배정하는 뜻이 아니다.
- 주 서버: **s1**. 기본 복원 학습 seed **1234**, 확인 반복 seed **7777**. N2 aligner donor는 기존 s1 seed 2025의 정확한 50K `last`로 고정한다.
- 현재 backbone: **W112–D123**, MS+PAN 9채널, HRMS 단일 복원. W96–D124는 과거 대조이며 이번 새 run의 backbone으로 되돌리지 않는다.
- 문서 상태: **다음 실행을 위한 신규 설계**. 서버 코드·큐·Sheet·checkpoint를 실제 변경한 기록이 아니다. 계수, 시간, gate는 아래에서 명시한 설계값이며 성능 보장은 없다.

> **핵심 방향**: 이미 반응을 배운 N2 aligner를 재사용한다. 복원 U-Net은 모든 update에서 추가 jitter가 없는 native 관측으로 학습한다. 추가 이동 PAN은 작은 aligner의 상대 offset 학습에만 사용한다. 그 위에 native PAN–GT 구조 감독이 실제 위치 기준을 개선하는지 분리한다.
>
> 핵심 비교: **P1−P0: 사전학습 정합 모듈 재사용 / P2−P1: 공동 미세조정 / P3−P2: 반응 보존 / P4−P3: GT 구조 감독**.

---

## 0. 출발 근거와 이번에 하지 않는 주장

### 0.1 최신 자료에서 확인된 출발점 [S1 §8, §10]

| 관측 | 확인된 값 또는 내용 | 이번 계획에 반영 |
|---|---|---|
| N2 SG R200 last의 추가 변위 반응 | native64 대각 기울기 −0.98/−0.99, FR512 −0.91/−0.95 | CNN을 새로 크게 만들지 않는다 |
| N2 last의 보고서 closure | native64 0.06 px, FR512 0.10 px | 동일 진단 코드로 donor를 재확인하고 출발 기준으로 보존 |
| N3 No-SG | closure는 비슷하거나 작지만 aligned-valid와 native 결과는 더 나쁨 | No-SG를 주력에서 제외; 상대 반응과 절대 위치를 구분 |
| R200 세 run의 best_raw | 첫 평가 1,010 update가 선택됨 | `best_raw`만으로 donor/방법의 품질을 판정하지 않음 |
| N2 last HQNR | raw 0.9329 / aligned-valid 0.9574 | 두 참조 방식과 ROI를 분리해 보고 |
| RR 수렴 ERGAS | R200 약 2.09–2.11, 학습 로그 정의 | 동일 export/evaluator로 다시 확인; 다른 정의를 합치지 않음 |
| R200 backbone | W112–D123, 사용자 결정으로 변경됨 | 이번 신규 대조군도 전부 같은 구조 |

현재 읽은 `pan-cvpr27`의 `WV3-s1`에도 N1/N2/N3 R200 결과가 등재되어 있다. Sheet의 N2 RR ERGAS 3.8380은 선택 checkpoint 결과이고, 보고서의 50K 수렴값과 다른 시점이다. 자료의 작은 수치 차이를 임의로 반올림/정밀도 차이라고 설명하지 않는다.

`WV3-s2`에서 `S2W112D123` 행을 검색했으나 이번 조회에서는 반환되지 않았다. 이는 **확인 가능한 등재 결과가 없었다는 뜻**이지, 서버에서 작업이 시작되지 않았다는 확인이 아니다. 기존 s2 KDV 큐는 수정하지 않는다.

### 0.2 해석상의 한계

1. 위 반응 수치는 알려진 추가 변형에 대한 검증이다. 실제 native PAN–MS의 물리적 정합 참값을 확인한 것은 아니다.
2. 보고서는 raw HQNR 하락을 정합·좌표 관계로 설명한다. 본 계획은 그 해석을 가설로 유지하며, 하락 전부가 지표 문제라고 전제하지 않는다.
3. Aligned-self HQNR은 모델마다 참조가 달라진다. 이를 동일한 benchmark 점수로 서로 순위 매기지 않는다.
4. 작은 closure를 절대 정합 uncertainty/covariance로 변환하지 않는다. 특히 `1/(closure² + 상수)`를 검증된 precision으로 사용하지 않는다.
5. b=2는 이미 FR 입력 통계를 참고한 개발 설정이다. 이번 재사용 실험도 그 이력을 물려받는다. train-only, test-untouched 결과라고 기술하지 않는다.

### 0.3 이번 16시간에서 제외

새 attention/correlation/local flow, LPAN/HPAN 입력, PAN reconstruction, MS/GT/output warp, 새 radius sweep, No-SG 재탐색, uncertainty head, GT variance KD, 대규모 KD 큐를 넣지 않는다.

GT 구조 감독은 아래 P4의 **정합용 보조 loss**이며 GT variance loss와 다르다. 이번 단계에서 유효한 donor/학습 정책을 정한 뒤 별도 KDV cohort로 연결한다.

---

## 1. 이번 연구 질문

- **H1 — 재사용**: N2에서 배운 aligner를 고정하고 native 입력으로 새 U-Net을 학습해도 복원과 shift robustness가 유효한가?
- **H2 — 망각**: N2를 native reconstruction만으로 미세조정하면 반응이 줄거나 작은 상수로 돌아가는가?
- **H3 — 분리 학습**: 복원은 native에서 수행하고, 알려진 추가 shift는 aligner에만 감독하면 반응을 유지하면서 복원을 개선할 수 있는가?
- **H4 — 위치 기준**: 상대 consistency에 native PAN–GT 구조 loss를 더하면 절대 위치의 진단과 복원 품질이 함께 좋아지는가?

H4의 loss는 PAN의 spectral response와 정확히 같은 GT 영상을 제공하지 않는다. 따라서 “정확한 절대 offset GT를 얻었다”가 아니라 **GT 좌표의 구조 감독을 추가한 가설**이다.

---

## 2. 공통 모델·초기화·데이터

### 2.1 입력과 출력

- P: 현재 학습/추론 PAN, `[B,1,H,W]`.
- S: 원래 LRMS, `[B,8,H/4,W/4]`.
- M: 기존 bicubic으로 4배 확대한 MS, `[B,8,H,W]`.
- Y: HRMS GT, M-frame.
- Aφ: 기존 dual-stem global aligner, 출력 `[B,2]`, 순서 `(dy,dx)`, 단위 현재 PAN/HR pixel.
- Fθ: W112–D123 U-Net, 9채널 입력에서 8채널 residual 출력.

```text
M = 기존 bicubic(S)
c0 = Aφ(P의 고정 내부 view, M의 동일 view)
P0_aligned = W(P, c0)
Yhat0 = M + Fθ(concat[P0_aligned, M])
```

MS condition·residual base·GT·출력에 추가 warp를 적용하지 않는다. Attention/CM3A, mode γ/β, PAN reconstruction은 없다. 이들 설정은 실제 PO10 R200 resolved config와 대조한다.

### 2.2 재사용할 donor

```text
source_run:
  PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT
source_checkpoint:
  정확한 optimizer update 50,000의 last
```

실제 checkpoint 파일 경로·확장자·state_dict prefix는 서버에서 확인하고 manifest에 기록한다. 이름만 `last`인 파일을 신뢰하지 말고 update, weight hash를 확인한다. `best_hqnr`나 첫 평가 1,010 step 가중치를 잘못 가져오면 실행하지 않는다.

**P1–P4는 aligner만 재사용한다. N2 U-Net 가중치는 가져오지 않는다.** 복원 U-Net은 모든 case에서 같은 seed의 동일한 random-init tensor에서 시작한다. P2/P3/P4는 서로의 학습 완료 결과를 이어 받지 않는다.

이 설계의 의미:

- 기존 N2 전체 모델과의 비교는 학습 프로세스가 다른 배경 비교이다.
- P1−P0에는 aligner의 사전학습 비용도 포함된 방법 차이가 있다. 완전히 같은 총 훈련 비용의 비교라고 주장하지 않는다.
- P2−P1, P3−P2, P4−P3는 동일한 donor와 새 U-Net 초기화를 공유하는 대응 비교다.
- P1은 가중치가 frozen이어도 **영상마다 다른 변위를 예측**한다. “모든 영상에 같은 상수 shift”를 쓰는 case가 아니다.

### 2.3 공통 설정

| 항목 | 실행 제안 |
|---|---|
| 데이터셋 | WV3만 |
| Backbone | W112, depth=[1,2,3] |
| 복원 학습량 | case당 50,000 optimizer updates |
| 기본 block | s1 / U-Net seed 1234 |
| 반복 block | 같은 s1 / U-Net seed 7777 |
| Donor | 모든 case·반복에서 동일 N2 s2025 last |
| Optimizer | 실제 PO10 R200의 AdamW 설정 계승 |
| U-Net LR | 기존 R200 1e-4를 manifest 확인 후 유지 |
| Trainable aligner LR | **1e-5**: 이번 미세조정의 새 설계값, P2–P4 동일 |
| Weight decay | 기존 R200 0.01을 manifest 확인 후 유지 |
| Scheduler | 기존 cosine + warmup100, 50K 끝까지; 각 parameter group에 같은 배율 |
| Batch | 기존 실효 48; 변경 시 전 case 동일 accumulation 규칙 |
| 정밀도 | 현재 R200 정책 확인; geometry/warp/offset은 FP32 |
| Aligner view | P/M 양쪽 margin4; 64²→56², 512²→504² |
| 전체 U-Net domain | 원래 전체 크기 유지 |

새 optimizer를 만들며 donor optimizer state는 가져오지 않는다. **로드한 donor의 마지막 head를 0으로 재초기화하지 않는다.**

P1은 aligner를 `requires_grad_(False)`로 하고 optimizer에서 제외한다. `.eval()`만 호출해서는 동결되지 않는다. Frozen module hash와 buffer까지 전후 비교한다. [R1]

---

## 3. 핵심 case 5개

| ID | Aligner | U-Net 입력 | 학습 loss | 직접 비교 |
|---|---|---|---|---|
| **P0 — NOALIGN** | 없음, sampler 우회 | 원 PAN + M | Lrec | 현재 W112–D123의 같은 block 기준 |
| **P1 — FROZEN** | N2 last 고정 | native PAN을 donor로 보정 + M | Lrec | P1−P0: 재사용 효과 |
| **P2 — FT-REC** | N2 last에서 학습 | native PAN을 학습된 aligner로 보정 + M | Lrec | P2−P1: 공동 미세조정·망각 |
| **P3 — FT-EQ** | N2 last에서 학습 | **native 보정 PAN** + M | Lrec + λoff Loff | P3−P2: 반응 보존 |
| **P4 — FT-EQ-GEO** | P3와 같은 시작점에서 독립 학습 | **native 보정 PAN** + M | Lrec + λoff Loff + λgeo Lgeo | P4−P3: native GT 구조 감독 |

**P3/P4의 corrupted PAN은 작은 aligner 입력에만 사용한다. U-Net에 전달하지 않는다.** 이는 기존 PO10과의 핵심 변경이다.

P0는 과거 W96 B0로 대체하지 않는다. 동일 s1·W112–D123·seed·학습 설정·평가 버전으로 이미 완료된 것이 확인되면 중복 학습 없이 재사용할 수 있으나, 실제 resolved config/hash 확인이 선행되어야 한다.

---

## 4. Forward와 loss를 구현하는 방법

### 4.1 모든 update의 기본 reconstruction

P1–P4에서:

\[
\hat c_0=A_\phi(P,M),\quad
\widetilde P_0=\mathcal W(P,\hat c_0),\quad
\hat Y_0=M+F_\theta([\widetilde P_0,M]).
\]

\[
L_{rec}=\operatorname{mean}_{b,c,y,x}|\hat Y_0-Y|.
\]

P0는 `P0_aligned=P`이고 aligner·warp를 호출하지 않는다.

**P2–P4의 c0는 reconstruction 경로에서 detach하지 않는다.** `Lrec`는 U-Net과 trainable aligner 양쪽을 학습시킨다. P1만 aligner가 고정된다.

학습용 reconstruction domain을 예측 shift에 따라 줄이지 않는다. MS와 GT의 관측이 없어지는 것이 아니고, 어려운 영역을 밖으로 밀어 loss를 낮추는 우회로를 만들지 않기 위해서다. 경계와 내부의 오차는 진단으로 따로 기록한다.

### 4.2 P3/P4: 두 update 중 한 번 작은 aligner에만 추가 변형

optimizer update index가 0부터 시작할 때:

```text
짝수 update: native reconstruction만 (+ P4 native geometry)
홀수 update: 위 native reconstruction + aligner offset 연습 (+ P4 native geometry)
```

이는 **기존 native/corrupted U-Net step 1:1을 유지하는 것이 아니다.** 복원은 이제 100% native이다. 1:1로 유지하는 것은 aligner의 추가 offset 연습 빈도다.

홀수 update에서:

\[
P_\epsilon=\mathcal W(P,\epsilon),\qquad
\hat c_\epsilon=A_\phi(P_\epsilon,M).
\]

\[
\boxed{L_{off}=\operatorname{mean}_{b,k}
|\hat c_\epsilon+\epsilon-\operatorname{sg}(\hat c_0)|.}
\]

- `k`는 dy,dx 두 성분이다. loss는 batch×2 성분 평균 L1이고 단위는 HR px다.
- `c0`는 이미 native forward에서 계산한 값을 재사용한다.
- offset target에서만 `c0.detach()`한다. clean aligner 호출 전체를 no-grad로 감싸지 않는다.
- ε는 loss와 영상 변형 생성에만 사용한다. CNN/U-Net의 입력 채널로 주지 않는다.
- 별도 Teacher/EMA 네트워크는 없다.
- 큰 U-Net은 **매 update 한 번**만 실행한다.

P3/P4에서는 `λoff=0.01`을 offset 연습 update의 시작부터 사용한다. **이것은 기존 PO10의 0→0.01/5K ramp와 다른 새 선택**이다. 이미 반응을 배운 donor의 미세조정이므로 반응 보존 항을 처음부터 활성화하기 위한 것이다. P2와 P3의 비교가 이 감독 추가의 효과를 분리한다. 반경별·sample별 ε로 정규화하지 않는다.

P0/P1/P2는 offset loss 계수 0이다. 별도 고정 진단 batch에서 같은 offset response를 측정할 수는 있다.

### 4.3 ε의 분포: b=2를 유지

\[
u_1,u_2\sim U(0,1),\quad
\rho=2\sqrt{u_1},\quad \theta=2\pi u_2,\quad
\epsilon=(\rho\sin\theta,\rho\cos\theta).
\]

- 0 중심 원판 면적 uniform, `||ε||₂≤2` 현재 PAN/HR px.
- 한 sample에 하나의 global vector; 모든 위치에 공통 적용.
- 매 연습 update·sample마다 새로 뽑고 원 PAN에서 만든다.
- 기존 공통 flip/rotation·전처리 이후 적용한다.
- 전용 RNG를 쓰며 P3/P4가 같은 sample/update에 같은 ε를 받게 한다.
- FR 추론에서는 synthetic ε=0이지만 aligner는 입력 기반 native correction을 예측한다.
- Aligner correction의 출력 범위를 ±b로 clamp/tanh 제한하지 않는다.

현재 b는 이미 분석한 FR 입력 규모를 참고한 설정을 계승한다. 이번 16시간에서는 b의 최적성이나 다른 센서 적응을 검증하지 않는다.

### 4.4 P4: GT 좌표의 구조 loss로 native 위치를 직접 감독

**신규 목적은 native 위치 감독을 추가하는 것**이다. 다만 descriptor를 새로 설계하지 않고, 기존 A3의 normalized-gradient outer-product loss [S2 §5.3]를 그대로 재사용한다.

\[
I_Y=\frac1C\sum_cY_c.
\]

- PAN 원본 P의 detached 평균/표준편차로 P와 보정 PAN P̃0를 동일하게 표준화한다.
- IY는 자체 통계로 표준화한다.
- 공통 Gaussian `σ=2 HR px`, radius6의 13×13 separable kernel과 정규화된 Scharr를 쓴다. 이는 loss용 smoothing이며 LPAN 입력을 재도입하지 않는다.

\[
g_P=\nabla G_2(Z_P(\widetilde P_0)),\quad
g_Y=\nabla G_2(Z(I_Y)).
\]

\[
\eta_P^2=\operatorname{sg}\{\operatorname{mean}_{V_g}
\|\nabla G_2(Z_P(P))\|^2+10^{-12}\},\quad
\eta_Y^2=\operatorname{sg}\{\operatorname{mean}_{V_g}\|g_Y\|^2+10^{-12}\}.
\]

\[
Q(g;\eta^2)=\frac{(g_x^2,\sqrt2g_xg_y,g_y^2)}{g_x^2+g_y^2+\eta^2},\qquad
w_Y=\operatorname{sg}\left(\frac{\|g_Y\|^2}{\|g_Y\|^2+\eta_Y^2}\right).
\]

\[
L_{geo}=\operatorname{mean}_b
\frac{\sum_{x\in V_g}w_Y(x)
\|Q(g_P(x);\eta_P^2)-Q(g_Y(x);\eta_Y^2)\|_2^2}
{\sum_{x\in V_g}w_Y(x)+10^{-8}}.
\]

`Vg`는 기존 고정 margin11, 64²에서 42² interior를 계승한다. Gaussian+Scharr+bicubic source support가 실제 영상 안인지 점검한다. GT edge weight가 없는 sample은 geometry만 0으로 하고 개수를 기록하며 rec에서는 제외하지 않는다.

```text
λgeo(t) = 0.01 × min(1, (t+1)/5000)
```

P4에서만 매 update 적용한다. 이 계수는 이전 A3의 초기 설정을 재사용한 것이지, donor 재사용 상황의 최적값으로 검증된 것은 아니다.

**이 항이 직접 업데이트하는 것은 aligner이며, U-Net θ에는 직접 gradient를 주지 않는다.** GT는 loss에만 사용한다. 추론에는 PAN/MS만 사용한다.

이전 A3가 거의 상수 상태에 머문 사실을 숨기지 않는다. 이번 P4는 **이미 shift response를 배운 N2에서 시작하여 그 반응을 유지하면서** 같은 geometry 감독이 유용해지는지 시험한다. 잘못된 spectral proxy를 더 강하게 따를 가능성도 있으므로 반드시 P3 대조와 독립 위치 진단이 필요하다.

### 4.5 역할 요약

| 항 | 감독하는 것 | 직접 gradient |
|---|---|---|
| Lrec | GT HRMS 복원 | Fθ, trainable Aφ |
| Loff | 추가 shift에 대한 correction 변화 | Aφ의 corrupted 경로 |
| Lgeo | native 보정 PAN과 GT의 구조 관계 | Aφ의 native 경로 |

P4에는 A2의 출력 edge loss, PAN reconstruction, pseudo-offset label, Teacher output KD를 추가하지 않는다.

---

## 5. 실행 코드에 반영할 핵심 차이

### 5.1 이전 PO10과 새 NF16의 구분

```text
이전 corrupted step:
P → random warp → aligner correction → 두 번째 warp → U-Net → GT loss

새 NF16 P3/P4:
[P, M] → aligner c0 → native PAN 한 번 warp → U-Net → GT loss
[Pε, M] → 같은 aligner cε → 알려진 ε로 offset loss
                               └ U-Net으로 보내지 않음
```

복원용 PAN은 새 case에서 한 번만 보간된다. Offset 연습용 Pε는 별도의 copy이고 한 번 보간된다. 따라서 NF16은 PO10과 동일한 augmentation ablation이 아니라 **native fitting으로 학습 경로를 분리한 후속 설계**이다.

### 5.2 P3/P4의 pseudocode

```python
# Existing preprocessing, augmentations and MS residual base are unchanged.
# This is integration pseudocode, not a complete trainer.
M = existing_bicubic(S)
c0 = aligner(valid_view(P, 4), valid_view(M, 4))
P0 = differentiable_warp(P, c0)        # gradient retained through c0
pred = M + unet(cat(P0, M))           # one restoration forward
loss = mean_l1(pred, Y)

if update_index % 2 == 1:
    eps = sample_uniform_disk(B, radius_hr=2.0, dedicated_rng=offset_rng)
    with no_grad():
        Pe = warp(P, eps)              # eps is known, not learned
    ce = aligner(valid_view(Pe, 4), valid_view(M, 4))
    Loff = abs(ce + eps - c0.detach()).mean()
    loss = loss + 0.01 * Loff

if case == 'P4':
    Lgeo = existing_A3_geometry(P0, P, Y, fixed_margin=11)
    loss = loss + 0.01 * min(1.0, (update_index + 1) / 5000) * Lgeo

loss.backward()
optimizer.step()
```

각 변환의 세부는 기존 구현을 사용한다.**이 pseudocode의 helper 함수는 실제 repository API 이름이 아니다.** 실제 model wrapper가 이미 M을 더한다면 residual base를 중복 합산하지 않는다.

### 5.3 Warp 규약

\[
\mathcal W(P,c)[y,x]=P[y+c_y,x+c_x].
\]

`grid_sample`의 grid 마지막 축은 `(x,y)`이고 aligner 출력은 `(dy,dx)`다. `bicubic / border / align_corners=False`를 계승한다. 학습되는 c가 0이어도 sampling을 생략하지 않는다. `.item()`, NumPy 변환, c.detach()로 grid를 재구성하지 않는다. [R2]

Pε 전체를 먼저 만든 다음 내부 crop을 취한다. 먼저 crop한 작은 영상을 warp해 인공 테두리를 재도입하지 않는다. Crop 후 normalization을 수행하고, donor와 같은 normalization 정의를 유지한다.

동결/학습 switch 때 gradient 여부뿐 아니라 optimizer parameter group과 `model.train()` 호출 뒤 aligner 상태를 점검한다. `.eval()`과 gradient 차단은 다른 기능이다. [R1]

---

## 6. HQNR: 두 참조 방식을 모두 기록하되, 평가 영역 효과를 분리

### 6.1 Native 평가 기본 열

동일 checkpoint의 **동일한 예측 HRMS**에서 아래를 계산한다.

| 열 이름 제안 | PAN reference | ROI | 의미 |
|---|---|---|---|
| `hqnr_raw_original` | 원 PAN P | 기존 전체 domain | 기존 benchmark·Sheet HQNR |
| `hqnr_raw_v64` | 원 PAN P | 고정 V64 | mask/경계 제외 대조 |
| `hqnr_aligned_self_v64` | 실제 model이 사용한 W(P,c0) | **동일 V64** | 자기 정합 참조 안의 일관성 |
| `hqnr_aligned_fixedN2_v64` | 고정 N2 donor가 만든 W(P,cD) | **동일 V64** | 모든 case에 동일한 보정 참조를 적용한 추가 진단 |

사용자가 동의한 두 방식은 **raw PAN vs aligned PAN**이다. Raw-V64는 이 두 방식 사이의 crop 효과를 분리하는 control이다. FixedN2는 aligned 계열 안의 공통 참조 진단이며, 별도의 출력 좌표계가 아니다.

FR V64는 `P.shape=512²`일 때 `[64:448,64:448]`, 384²이다. 예측값마다 ROI를 줄이거나 이미지 내용을 보고 바꾸지 않는다. 이전 보고서의 `aligned_valid`가 정확히 이 ROI였는지는 evaluator manifest로 확인한다. 다르면 과거 0.9574와 새 V64 값을 직접 비교하지 말고 donor도 재평가한다.

**공통 참조 frozen N2는 참 정합 GT가 아니다.** 다만 참조와 ROI를 모델별로 바꾸지 않고 출력 차이를 비교할 수 있다. N2 계열에 유리한 편향 가능성은 남는다. 공통 참조 점수 하나만으로 절대 정합 성공을 주장하지 않는다.

P1(FROZEN)에서는 model c0와 frozen reference cD가 같으므로 self/fixed aligned 점수가 수치적으로 일치해야 한다. P0의 self reference는 P라 raw_v64와 같아야 한다.

### 6.2 계산 규칙

1. 최신 `fr_mat20`의 **20장 전체**와 실제 manifest를 계승한다. 예전 FR 12–19 subset으로 돌아가지 않는다.
2. 장면별 `Hi=(1−Dλ,i)(1−Ds,i)`를 계산한 뒤 평균한다. 평균 Dλ,Ds의 곱으로 대체하지 않는다.
3. Shifted-PAN 평가의 저해상도 PAN reference도 **그 shifted PAN에서 기존 evaluator 경로로 다시 생성**한다. 높은 해상도 참조만 교체하지 않는다.
4. MTF/down-up/필터 연산은 전체 영상에서 수행한 뒤 정해진 ROI로 집계한다. ROI를 먼저 잘라 downsample phase를 바꾸지 않는다.
5. MS reference·GT는 이동하지 않는다. 자기 출력에 맞춰 GT를 최적 정렬하지 않는다.
6. 같은 예측·ROI·MS 참조에서 PAN reference만 교체하면 Dλ는 같아야 한다. 다르면 평가 경로를 검사한다.
7. V64 내부까지 padding/interpolation/filter support가 침범하면 그 장면을 빼 평균하지 않는다. eligibility 실패를 기록하고 해당 aggregate를 부적격으로 표시한다.
8. Raw-original은 과거 정의 그대로 기록한다. 지원 범위를 넘는 실패를 숨기기 위해 view 자체를 바꾸지 않는다.
9. 각 view의 fSCC, Dλ, Ds와 유효 면적을 함께 기록한다. RR SCC와 FR fSCC를 혼동하지 않는다.

### 6.3 Sheet 표시 제안 — 기존 행은 덮어쓰지 않음

`checkpoint_kind`, `optimizer_update`, `output_frame`, `reference_policy`, `roi_id`, `metric_version`을 함께 기록한다. 최소한 50K last 행에는 raw_original, raw_v64, aligned_self_v64를 나란히 제공한다. FixedN2는 별도 진단 표로 제공해도 된다.

HQNR(V64)를 aligned 점수라고 이름 바꾸지 않는다. Aligned 점수는 논문 raw HQNR과 다른 프로토콜임을 Notes/헤더에 명시한다. 이번 문서 작성만으로 Sheet에는 아무 변경도 수행하지 않았다.

---

## 7. Checkpoint 선택: 측정과 주 판정을 분리

### 7.1 저장

- `last`: **정확한 50K**. 이번 방법 비교·donor 후보 분석의 기본 시점.
- `best_raw`: 기존 raw_original HQNR 최대. 기존 benchmark 보고와 조기 선택 현상의 기록용.
- `best_aligned`: aligned_self_v64 HQNR 최대. 정합 참조 내부의 탐색용. self-referenced selection이라고 표시.
- `best_rr_val`: **실제로 분리된 RR validation split이 확인될 때만** 저장 가능. RR test 20장을 val이라고 이름 바꾸지 않는다.

각 selector는 독립 상태다. HQNR running max로 tie band 1e-4를 관리하고 같은 view의 fSCC로 동률을 처리한다. 둘 다 같으면 더 늦은 update를 선택한다. 실제 구현의 exact comparator를 manifest에 기록한다.

새 native 평가 주기는 기존 R200을 계승하되 exact50K를 누락하지 않는다. FixedN2 전체 평가와 큰 stress 진단은 최소 last에서 실행하고, 필요 시 저장된 best에서도 실행한다.

### 7.2 주 판정

**공식 best_raw 선택 기준을 폐기하지 않는다.** 다만 그 checkpoint가 미학습 초기일 수 있으므로 본 캠페인의 인과 비교는 같은 50K last끼리 먼저 한다.

Raw HQNR과 aligned-self HQNR을 가중합해 새로운 단일 점수를 만들지 않는다. Self-aligned 하나로 winner를 고르지 않는다. 이후 donor를 정할 때에는 반응 보존, 독립 위치 진단, RR 복원, 두 HQNR의 trade-off를 명시해 판단한다.

R200 donor의 이전 학습 50K와 이번 복원 50K를 감추지 않는다. P1–P4의 비용은 `shared donor pretraining + incremental fitting`으로 분리 기록한다.

---

## 8. 상대 반응, 절대 위치, 복원 품질을 별도로 진단

### 8.1 상대 반응

고정 probe에서:

\[
r_\epsilon=A(P_\epsilon,M)+\epsilon-A(P,M).
\]

- 기존 `po10_diag` closure 정의를 확인해 **동일 정의로 donor와 새 모델을 재측정**한다.
- 추가로 `offset_mae_component=mean|rε|`, `offset_epe=mean||rε||₂`를 각각 기록한다. 둘을 같은 closure로 섞지 않는다.
- `cε−c0=Bε+a`의 전체 2×2 B와 intercept를 기록한다. 대각뿐 아니라 교차축 항도 본다.
- 각 크기 구간의 평균/P90, 원본 예측 분산, bias drift를 별도 기록한다.

Probe 제안은 train64의 고정 비학습 진단 subset, RR256, FR512에서 공통으로 `r={0.5,1,2}`, 8방향과 native 0을 사용한다. 기존 donor 비교에 필요한 이전 probe도 보존한다. 작은 입력에서는 두 warp support를 점검하고 고정 공통 interior에서 계산한다.

### 8.2 Native 절대 위치의 근거

- **RR**: 예측 HRMS–GT를 같은 band/같은 좌표에서 비교한다. 원본/보정 PAN–GT의 구조 차이는 추가 proxy다.
- **FR**: 관측 MS와 선명도를 맞춘 PAN의 정합을 Scharr-ZNCC와 기존 독립 descriptor(census/phase-correlation 중 검증된 것)로 재측정한다.
- Aligner 예측, audit 추정, 보정 후 잔여 추정값을 scene별로 저장한다. 부호·moving/ref 순서·HR/LR 단위를 manifest에 명시한다.
- PAN과 MS의 spectral 차이에 따른 audit 하한이 남는다. audit값을 정확한 GT offset이라고 부르지 않는다.
- 학습에 사용한 Q-geometry loss만 평가하지 않는다. P4가 학습 descriptor만 개선하고 다른 진단은 악화하면 성공으로 판단하지 않는다.
- GT-based diagnosis는 학습/검증/RR에만 있다. FR HRMS GT는 존재한다고 가정하지 않는다.

입력 크기별 보정 median과 crop/tile 변화를 기록하되, H=64→512라고 shift를 8배 하지 않는다.

### 8.3 복원과 artifact

동일 evaluator로 last의 ERGAS/SAM/PSNR/SSIM/SCC/Q8을 재산출한다. 학습 로그와 export 평가의 차이는 각각 이름을 남기고 원인을 확인한다. 고정된 건물·도로 edge crop에서 Yhat–GT overlay와 gradient profile을 저장한다. FR은 미리 정한 장면/좌표로 원 PAN, MS, 보정 PAN, output을 함께 확인한다.

Zero correction·wrong-sign correction은 **같은 가중치의 진단**으로만 실행한다. 이 진단이 새로 훈련한 P0 대조를 대체하지 않는다.

### 8.4 FR synthetic stress의 참조를 고정

Stress에서 모델 입력은 Pε로 바뀌지만, M과 원하는 출력 좌표는 그대로다. 강건성 비교의 기준 PAN/reference는 **native 원 PAN P와 native fixedN2 reference**로 고정한다. ε마다 평가 reference를 따라 옮겨 손실이 줄어든 것처럼 만들지 않는다.

기존 stress의 raw_input(Pε 참조)·aligned_self(W(Pε,cε) 참조)는 재현용 추가 열로 남길 수 있으나 native-reference 점수와 구분한다. ROI는 기존 stress margin96을 재검증해 고정하고, native도 같은 V96에서 계산한다. Native V64와 stress V96을 직접 빼지 않는다.

RR stress는 GT Y를 움직이지 않고 복원 오차를 직접 평가한다. 전체 synthetic stress 결과를 checkpoint 선택에 사용하지 않는다.

### 8.5 Last에서 shortcut 대조를 다시 한다

보고서 §10.3의 MS-swap/상수-MS/bilinear/padding 대조는 주로 **초기 best checkpoint**에서 기록되었다. 그 결과를 last의 shortcut 검증으로 그대로 승계하지 않는다.

기본 donor N2 last와 최종 후보에서 같은 MS-swap/상수-MS/다른 보간 kernel 대조를 다시 수행한다. 원래 모달리티 대응을 제거했을 때 반응이 어떻게 바뀌는지를 기록한다. 검사 통과를 완전한 물리적 정합 증명으로 과장하지 않는다.

---

## 9. 사전 gate

### G0 — 자산·설정 확인

Donor가 정확한 50K인지, aligner submodule weight와 normalization/view가 맞는지 확인한다. 새로운 W112 U-Net parameter shape와 donor aligner I/O를 점검한다. 최신 evaluator와 데이터 manifest를 고정한다. 핵심 보고 수치 재현 오차는 원래 코드의 tolerance를 적용하고, code/version 불일치 시 진행하지 않는다.

### G1 — 학습·gradient 검사

- P0 sampler 없음; P1 aligner hash/buffer가 optimizer update 이후에도 불변.
- P2–P4 native Lrec→aligner gradient 존재.
- P3 Loff→corrupted aligner gradient 존재; target 경로만 stop.
- P4 Lgeo→aligner gradient 존재, Lgeo만으로 U-Net θ gradient 없음.
- 각 update U-Net forward 호출 1회; P3/P4 U-Net 입력이 Pε가 아님.
- Native/corrupt view 둘 다 crop4, crop 후 normalization.
- P3/P4 동일 ε와 데이터 순서 hash.
- b=2 sampler는 mean0, 축 std≈1, mean radius≈4/3. 이론 분포와의 통계 검사는 충분한 sample에서 수행.

`eval()`만으로 frozen을 구현하거나, clean c0를 전부 detach해 native recon의 gradient를 끊는 오류를 반드시 검사한다. [R1]

### G2 — P4 geometry의 가용성 검사

기존 A3 구현과 수식이 같은지 확인한다. source support/정규화/flat-region 처리와 gradient finite 여부를 검증한다. GT 구조로 만든 알려진 변형 대조에서 올바른 위치 주변에 유효한 gradient가 생기는지 확인하되, 실제 cross-modal 참값까지 검증한 것으로 간주하지 않는다.

기존 잘못된 phase나 GT 입력 유출을 발견하면 수정·별도 version화 후 진행한다. Source support가 실패하면 sample을 조용히 제외하거나 loss를 0으로 만드는 대신 실패를 기록한다.

### G3 — 비용 profiling

실제 batch로 native·offset·geometry update를 각각 측정하고 evaluation 비용을 포함한 50K 예상 시간을 낸다. 기존 PO10의 1.62h 등은 예약 근거일 뿐 새로운 경로의 확정 소요 시간이 아니다. CUDA `grid_sample` backward 비결정성은 실제 설치 환경을 기록한다. [R2]

---

## 10. 16 GPU-hour 배분

| 작업 | 예약 GPU 시간 | 상태 |
|---|---:|---|
| Donor 재평가·계측·구현 gate | 1.5h | 필수 |
| P0 NOALIGN / s1234 | 1.6h | 필수 |
| P1 FROZEN / s1234 | 1.6h | 필수 |
| P2 FT-REC / s1234 | 1.8h | 대응 대조 |
| P3 FT-EQ / s1234 | 2.0h | 주 후보 |
| P4 FT-EQ-GEO / s1234 | 2.0h | geometry gate 통과 시 |
| P3 + P4 / s7777 대응 반복 | 4.0h | 예산·gate 통과 시, 두 run 한 묶음 |
| 최종 동일 evaluator 평가·표·진단 | 1.0h | 필수 |
| Buffer | 0.5h | 필수 |
| **합계** | **16.0h** | |

시간은 예약 예산이다. 실시간 시계 기준 16시간 안에 CPU 작업까지 반드시 완료된다는 보장은 없다. GPU ledger에는 학습·GPU 평가·폐기된 gate run도 포함한다.

기본 순서: `gate → P0 → P1 → P2 → P3 → P4 → (P3,P4 반복) → 최종 평가`.

시작 gate:

```text
used_gpu_h + 1.2 * profiled_remaining_mandatory_h + final_eval_reserve_h <= 16
```

반복 pair는 둘 다 같은 50K까지 갈 예산이 있을 때만 시작한다. P3는 50K, P4는 20K로 종료하고 같은 수준으로 비교하지 않는다. 예산이 초과하면 먼저 두 번째 seed 반복을 다음 캠페인으로 넘기고, 필수 case의 학습량은 유지한다. P4의 geometry gate가 실패하면 P4를 억지로 기동하지 않고 해당 실패를 문서화하며, 반복 대상은 사전에 정한 대체 pair **P1/P3**로 바꿀 수 있다.

s2에서 동일한 이름의 baseline이 있어도 s1 수치와 직접 대응시키지 않는다. 다른 서버로 옮길 때는 비교 pair 전체를 같은 block에 배치한다. 이 계획을 수행해도 모든 case의 3-seed 통계가 확보되는 것은 아니다.

---

## 11. 무엇을 성공으로 볼 것인가

### 11.1 지표의 역할

1. **반응 보존**: donor와 동일 정의의 closure/B를 비교한다. 작은 self-consistency가 절대 정확성을 뜻하지는 않는다.
2. **Native 위치 근거**: RR의 GT 좌표와 FR의 독립적인 입력 정합 proxy를 함께 본다.
3. **복원 품질**: 같은 50K·evaluator의 RR 지표와 시각 결과를 확인한다.
4. **원본/정합 HQNR**: 두 참조의 결과와 차이를 숨기지 않는다. Raw가 내려갔다면 benchmark trade-off로 남긴다.

파일럿 경고값으로 `FR offset_epe > donor_recomputed + 0.05 HR px`, `RR ERGAS > matched_control × 1.02`를 사용할 수 있다. 이는 **이번 조기 경고용 설계값**이며 통계적 동등성/비열등성 검정이나 물리적 기준이 아니다. 최종 채택은 대응 반복·scene별 결과를 포함해 판단한다.

### 11.2 결과별 해석

| 관측 | 다음 해석 |
|---|---|
| P1이 P0보다 의미 있게 좋음 | 사전학습 정합이 새로운 복원 모델에 재사용 가능함을 지지 |
| P2에서 raw만 상승하고 반응이 작아짐 | 정합을 덜 수행해 원본 참조에 가까워졌을 가능성; 성공으로 단정하지 않음 |
| P3가 P2보다 반응을 지키며 RR/공통 참조 결과 유지·개선 | native fitting + offset 보존의 유효성 지지 |
| P4가 P3보다 independent 정합·RR를 개선, 반응 유지 | GT 구조 감독을 포함한 후속 주력 후보 |
| P4는 Lgeo/self-aligned만 좋고 독립 진단 악화 | spectral proxy 편향 또는 잘못된 위치 감독; P4 미채택 |
| P1이 가장 안정적 | aligner frozen 전략을 다음 KDV의 기본으로 사용 |
| 모든 P1–P4가 P0 대비 불리 | 반응 학습의 성공은 보존하되 복원 이득 주장을 보류; 실제 적용 경로/정합 기준을 재검토 |

P4가 반드시 성공해야 하는 계획이 아니다. “반응만 학습된 모듈”에서 “복원에 유효한 정합 모듈”로 넘어가는 데 필요한 근거를 얻는 것이 목적이다.

---

## 12. s2 KD로 연결하는 기준

이번 16시간에서는 KD loss를 추가하지 않는다. s2의 기존 A1 donor cohort를 덮어쓰지 않는다.

후속 KDV cohort를 만들 때:

- 이번에 선택한 donor의 exact checkpoint, weight/view/normalization hash를 고정한다.
- Teacher·Student 모두 W112–D123 및 같은 M-frame 기준을 사용한다.
- Frozen aligner 재사용은 weight transfer이고, 그것만으로 별도의 distillation loss가 수행된 것은 아니다.
- 절대 shift teacher loss와 covariance weighting은 독립적으로 검증한 뒤 추가한다.
- closure 0.06을 GT variance/uncertainty로 간주해 큰 precision을 바로 부여하지 않는다.
- 동일한 frozen donor/reference 조건 안에서 GT-only, output KD, GT variance의 효과를 분리한다.

이번 반복은 복원 초기화 seed를 바꾸는 검사다. donor 자체는 같은 N2 s2025이므로 donor 사전학습 seed의 불확실성까지 측정한 것은 아니다.

---

## 13. Config 통합 template

기존 repository key의 실제 이름은 확인해야 한다. 아래는 실행 semantics를 표현하는 template이며 바로 실행되는 완제품 config가 아니다.

```yaml
protocol_id: NF16_W112D123_N2LAST_v1
server: s1
budget_gpu_hours: 16
primary_seed: 1234
repeat_seed: 7777
full_updates: 50000

assets:
  source_run: PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT
  donor_checkpoint: REQUIRED_EXACT_LAST_PATH
  donor_optimizer_update: 50000
  donor_weight_sha256: REQUIRED_ACTUAL_HASH
  preload_backbone: false
  donor_calibration_status: inherited_FR_input_statistics_informed

model:
  width: 112
  depth: [1, 2, 3]
  input: [PAN, bicubic_MS]
  pan_reconstruction: false
  lpan_hpan_channels: false
  mode_modulation: false
  attention: false
  output_frame: M
  aligner_view_margin_hr: 4
  aligner_output_order: [dy, dx]
  aligner_output_units: current_PAN_HR_pixel

training:
  reconstruction_route: native_every_update
  restoration_forward_count_per_update: 1
  backbone_lr: 0.0001
  trainable_aligner_lr: 0.00001
  weight_decay: 0.01
  batch_effective: 48
  scheduler: inherited_cosine_warmup100
  native_reconstruction_domain: inherited_full_GT_domain
  copy_donor_optimizer: false
  reinitialize_loaded_shift_head: false

offset_exercise:
  enabled_cases: [P3, P4]
  every_n_optimizer_updates: 2
  active_parity: odd_zero_based
  radius_hr: 2.0
  distribution: uniform_disk_area
  send_corrupted_pan_to_unet: false
  epsilon_input_to_network: false
  dedicated_rng: true

loss:
  rec: mean_L1
  offset: mean_abs_c_eps_plus_eps_minus_detached_c0
  offset_lambda: 0.01
  offset_ramp_updates: 0
  geometry: existing_A3_normalized_gradient_outer_product
  geometry_sigma_hr: 2.0
  geometry_kernel_radius: 6
  geometry_margin_hr: 11
  geometry_lambda_max: 0.01
  geometry_ramp_updates: 5000
  geometry_on_native: true
  case_overrides:
    P0: {aligner: absent, offset: false, geometry: false}
    P1: {aligner: frozen_N2_last, offset: false, geometry: false}
    P2: {aligner: finetune_N2_last, offset: false, geometry: false}
    P3: {aligner: finetune_N2_last, offset: true, geometry: false}
    P4: {aligner: finetune_N2_last, offset: true, geometry: true}

evaluation:
  fr_scene_manifest: REQUIRED_FR_MAT20_MANIFEST
  evaluator_commit: REQUIRED_ACTUAL_COMMIT
  hqnr_aggregation: mean_of_scene_products
  native_roi: fixed_V64
  stress_roi: fixed_V96
  views: [raw_original, raw_v64, aligned_self_v64, aligned_fixedN2_v64]
  fixed_aligned_reference: source_N2_last_native_input_prediction
  mask_adapts_to_prediction: false
  reference_generated_from_full_frame_before_roi: true
  methods_primary_checkpoint: exact_last_50K
  save_checkpoints: [best_raw, best_aligned, last]
  best_rr_val: only_if_genuine_validation_split_exists
  select_on_synthetic_stress: false
  posthoc_output_or_GT_registration: false

repeat:
  default_pair: [P3, P4]
  fallback_if_P4_preflight_fails: [P1, P3]
  pair_must_complete_same_budget: true
  donor_fixed_across_seeds: true
```

---

## 14. 필수 산출물

```text
config_resolved.yaml
asset_manifest.json                 # donor/update/hash/view, F init hash
training_data_manifest.json
native_eval_manifest.json           # scene IDs, filter/phase, ROI, reference IDs
stress_eval_manifest.json
train_log.jsonl
gradient_diagnostics.jsonl          # rec/off/geo norm + cosine, c0 drift
checkpoint_metrics.csv              # checkpoint_kind와 exact update 포함
scene_metrics.csv                   # view/ROI/reference policy 포함
offset_response_last.csv            # full B, intercept, MAE/EPE
absolute_alignment_diagnostics.csv  # audit는 proxy임을 표시
paired_seed_comparisons.csv
budget_ledger.json
last/ best_raw/ best_aligned weights and export metadata
```

매 run은 별도 새 이름으로 보존한다. 제안 이름은 `NF16_P{0..4}_W112_D123_WV3_S{1234|7777}_N2LAST_v1`이다. Exact filenames는 서버 convention에 맞춘다.

---

## 15. 출처와 검증 범위

### 내부 근거

- **S1** 최신 결과 보고서 §8, §9, §10.2, §10.5, §10.6. 오래된 §1–§7의 상수/초기 반응 설명으로 마지막 모델을 재판정하지 않는다.
- **S2** 기존 A1–A3 v2: §5.3 geometry 수식과 gradient 경로, §6 border/support, 평가 원칙.
- **S3** 기존 PO10 10시간 계획: offset 식·sampler·view·native/corrupted 경로.
- **S4** R100→R200 변경 명세: 반경·provenance. 이후 S1 §8에 기록된 W112–D123 변경을 현재 구조로 우선 적용한다.
- **라이브 Sheet**: `pan-cvpr27`, ID `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`, 이번 응답 중 `WV3-s1!A1:V198`에서 PO10/W112_D123, `WV3-s2!A1:U200`에서 S2W112D123를 검색했다. 런타임 성능·서버 큐를 직접 검사한 것은 아니다.

| ID | 입력 파일 | 이 문서 작성 시 로컬 원문 SHA-256 |
|---|---|---|
| S1 | `2026-09-10_pa-a1-a3-s1-results.md` | `3fd15614bf37ebecd9a96af0d726302cedc13ae7d4696f30ff2559819e6a539e` |
| S2 | `PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md` | `4ea1f157151a1b8a25033afa175f9923e33d4ee385c6643c46c2c86d6d94d64b` |
| S3 | `PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md` | `334a13fdae48faa177d839d5b04e938b283083c97edc56b67c430152f70077f3` |
| S4 | `PAN_OffsetConsistency_ChangeNote_R100_to_R200_2026-09-10.md` | `cff0daa5ff8ff0bd985d96b5fefeebc48bff088c4d42dd38737c14fed42836f1` |

### 외부 구현 참고

- R1: PyTorch 공식 문서, **Autograd mechanics — Setting requires_grad / Grad Modes**. `https://docs.pytorch.org/docs/main/notes/autograd.html`
- R2: PyTorch 공식 문서, **torch.nn.functional.grid_sample**. `https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html`

이 문서의 새로운 학습 정책·예산·계수는 본 연구의 제안이다. 위 외부 문서가 그 성능을 보장하지 않는다. API 세부는 실행 서버의 실제 PyTorch/CUDA 버전에서 확인한다.

### 작성 시 수행한 범위

원문·라이브 Sheet 열람, 수식/설정의 일관성 점검, 예산 합계 확인, Markdown 파일 생성까지 수행했다. 실제 W112 U-Net 학습, GPU profiling, donor checkpoint 로드, 신규 loss의 서버 unit test는 수행하지 않았다. 위 gate를 통과한 것으로 기록하지 않는다.
