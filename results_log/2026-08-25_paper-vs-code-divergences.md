# 논문 서술과 배포 코드의 불일치 (2026-08-25)

배포 코드를 논문 본문·Figure 3·수식과 한 줄씩 대조한 목록이다.
**방침: 불일치가 있으면 논문을 따른다.** 재구성본은 `model/pancrafter_paper.py` 다.

| | 개수 |
|---|---:|
| 불일치 확인 | **13** |
| 재구성본에 반영 | **11** |
| 미반영 | **2** |
| 판정 보류 (논문이 명시하지 않음) | 1 |

---

## A. 구조 — Figure 3 / §3.2

| # | 항목 | 논문 | 기존 코드베이스 | 재구성본 | |
|---|---|---|---|---|:--:|
| 1 | spatial scale | Fig 3 도식 **3개** | **4개** | 3개 | ✅ |
| 2 | DownConv / UpConv | Fig 3 **2 / 2** | **3 / 3** | 2 / 2 | ✅ |
| 3 | AttnBlock 수 | Fig 3 **3개** | **5개** | 3개 | ✅ |
| 4 | AttnBlock 배치 | *"low- and mid-resolution stages incorporate both ResBlock and AttnBlock, while **high-resolution stages use only ResBlock** to reduce computational overhead"* | H/2(고해상도) encoder·decoder 에도 CM3A 배치 | H/2 enc · H/4 bottleneck · H/2 dec | ✅ |

4번이 1–3번의 근거다. 4단계를 high / mid / low 로 나누면 H/2 는 high 이므로 CM3A 가 없어야 하고,
그러면 자연히 scale 3 · AttnBlock 3 이 된다.

---

## B. 블록 내부 — Eq (5)–(6)

논문 ResBlock:

```
x ← Conv(SiLU(LN(x)))
x ← x + Conv(SiLU(Modulate(LN(x); mode)))

Modulate(x; MS)  : x ← (1 + γ_ms)  ⊙ x + β_ms
Modulate(x; PAN) : x ← (1 + γ_pan) ⊙ x + β_pan
```

| # | 항목 | 논문 | 기존 코드베이스 | 재구성본 | |
|---|---|---|---|---|:--:|
| 5 | ResBlock 정규화 | **LayerNorm** (Eq 5 의 `LN`) | `GroupNorm32(32, C)` | ChannelLayerNorm | ✅ |
| 6 | mode modulation | **γ, β ∈ R^C 를 직접 학습** (Eq 6). 블록당 2×2×C = **512** | mode token 을 블록마다 `Linear(128, 2C)` 로 사영. 블록당 **33,024** | 직접 학습 | ✅ |
| 7 | dropout | 수식에 없음 | `dropout: 0.2` | 0.0 | ✅ |

- **5번은 파라미터가 중립이다.** LayerNorm 과 GroupNorm32 는 둘 다 2C 라, 파라미터 수를 맞추는
  방식의 대조로는 드러나지 않는다. 수식을 직접 읽어야 발견된다.
- **6번이 파라미터 격차의 최대 몫**이다. ResBlock 12 + AttnBlock 3 = 15 블록에서 약 **0.49 M**.

---

## C. CM3A — Eq (10)–(12) / §3.3

| # | 항목 | 논문 | 기존 코드베이스 | 재구성본 | |
|---|---|---|---|---|:--:|
| 8 | `[K_pan \| V_pan]` | **결합 conv 1개** — `Conv([I_pan^rep,↓ \| x])` (Eq 11) | `k_pan`, `v_pan` **별도 conv 2개**. 입력도 다르다 — `k_pan` 은 `(x, lpan)`, `v_pan` 은 `(x, lpan, pan−lpan)` 로 고주파 채널이 더 들어간다 | **기존 코드 그대로** | ❌ |
| 9 | local attention k | **k = 3** (§4.2, 전역) | bottleneck 만 **k = 1** (`_attn(3, 1, 1)`) | 전 위치 k = 3 | ✅ |
| 10 | key 별칭 | `x_pan = LocalAttn(Q, K_pan, V_pan)`, `x_ms = LocalAttn(Q, K_ms, V_ms)` (Eq 10/11) | `q, k_pan, k_pan = ...` — 좌변에 `k_pan` 이 두 번. **PAN 브랜치가 MS key 로 attention** 하고, `k_ms` 는 재할당되지 않아 뒤섞인다. 죽은 파라미터 **611,584개**(전체의 6.13%) | 수식대로 | ✅ |
| 11 | LocalAttn | §3.3: *"attention scores within the k × k local receptive field"* | `reset_parameters()` **호출부가 저장소에 없다.** `dep_conv` 가 shift kernel 이 아니라 랜덤 초기화로 학습된다 — softmax 가 고르는 9개가 "3×3 이웃 9픽셀" 이 아니라 "3×3 패치의 학습된 선형결합 9개" | 호출 + bias 0 고정 | ✅ |

### 일치가 확인된 것 (대조 완료)

- `Q = Conv([I_ms^lr,↓ | x])` (MS mode) / `Conv([I_pan^rep,↓ | x])` (PAN mode) —
  코드의 `cond = pan_·(1−s) + ms·s` 가 mode 별로 같은 동작을 한다
- `[K_ms | V_ms] = Conv([I_ms^lr,↓ | x])` (Eq 10) — 결합 conv 로 일치
- `x_attn = α1 ⊙ x_ms + α2 ⊙ x_pan`, α ∈ R^C 를 mode 별로 학습 (Eq 8)
- `x ← x + x_attn`, `x ← x + FFN(LN(x))` — AttnBlock 의 LN 위치
- MARs 손실 (Eq 4)과 λ = 1.0
- 하이퍼파라미터 전부 — 50K iteration, warmup 100, AdamW lr 1e-4 / wd 0.01, cosine,
  batch 48(실효 96), **seed 2,025**, C = 128, PAN 64×64 / MS 16×16

---

## D. 입력 — §3.2

| # | 항목 | 논문 | 기존 코드베이스 | 재구성본 | |
|---|---|---|---|---|:--:|
| 12 | 입력 채널 | *"channel-wise concatenation of I_pan and I_lrms"* → **9 ch** | `PAN, ↑LPAN, PAN−↑LPAN, ↑MS` → **11 ch** | 9 ch | ✅ |

---

## E. 학습 — §4.2

| # | 항목 | 논문 | 기존 코드베이스 | 재구성본 | |
|---|---|---|---|---|:--:|
| 13 | 증강 | *"random horizontal/vertical flips, 90-degree rotations, and **random cropping**"* | crop 후 **`cv2.resize` 로 원크기 복원** = scale jitter | **기존 코드 그대로** | ❌ |

```python
ratio = (1 - 0.75) * random() + 0.75         # 0.75 ~ 1.0
# gt / lms / ms / lpan / pan 을 ratio 만큼 자른 뒤
gt = cv2.resize(gt, (64, 64), INTER_CUBIC)   # ← 원래 크기로 되돌린다
ms = cv2.resize(ms, (16, 16), INTER_CUBIC)
```

깨지는 것이 둘이다.

1. `ms` 와 `gt` 가 **각각 독립적으로** bicubic 리샘플되어, MS→GT 열화관계(MTF + decimation)가
   학습 중에만 왜곡된다. 테스트에는 그 왜곡이 없다.
2. PAN : MS = 4 : 1 이라는 고정 비율이 최대 1.33 배까지 흔들린다.

---

## 반영하지 않은 2건과 그 이유

| # | 항목 | 반영하지 않은 이유 |
|---|---|---|
| **8** | `[K_pan\|V_pan]` 결합 | 논문대로 결합 conv 로 바꾸면 파라미터가 **7.1707 M → 7.2122 M** 이 되어, 논문이 주장하는 7.170 M 과의 일치(+0.01%)가 **+0.59%** 로 벌어진다. 방침(논문 우선)에 따르면 적용 대상이지만, 그 경우 **파라미터 일치가 우연이었을 가능성**을 함께 받아들여야 한다 |
| **13** | 증강 crop | 논문 서술을 **그대로 구현할 수 없다.** 학습 패치가 이미 최소 단위(PAN 64×64 / MS 16×16)라 잘라낼 여지가 없다. 선택지는 `crop: False`(증강에서 제외)와 현행 유지 둘뿐이고, 어느 쪽도 "논문대로" 가 아니다 |

---

## 판정 보류 — 논문이 명시하지 않은 것

| 항목 | 논문 | 기존 코드베이스 |
|---|---|---|
| 잔차 기준선 | Eq (1): *"up-sample it by a factor of 4"* — **보간 방법 미명시** | `F.interpolate(ms, scale_factor=4, mode="bicubic")`. 데이터셋이 제공하는 `lms` 와 다르다 |
| ResBlock depth 배분 | **stage 별 반복 수 미기재** | `[2, 2, 2, 2]` (총 14개) |
| `mlp_ratio` | 미기재 | 4.0 |
| `s_embed_size` | 미기재 | 128 |

논문이 적지 않았으므로 **불일치로 단정할 수 없다.** 재구성본은 파라미터 7.17 M 제약에서
ResBlock 총 12개가 되도록 `(2, 2, 4)` 를 골랐으나 배분 자체는 특정되지 않는다 —
`(1,3,4)`, `(3,2,2)`, `(1,1,8)` 이 전부 같은 파라미터 수다.

---

## 추가 이슈 — 논문 불일치는 아니지만 코드에 있는 문제

`KNOWN_ISSUES.md` 에 상세가 있다.

### 배포 데이터 결함

| | |
|---|---|
| **F-1** | 배포 `pan_h5.zip` 의 **WV3·QB full-resolution `lpan` 이 다른 장면이다.** 그대로 쓰면 full-res 평가가 무효다. `tools/repair_lpan.py` 로 재생성해야 한다 |

### 운용 — 50K 실행 전에 손봐야 하는 것

| | |
|---|---|
| **C-1** | `accelerator.prepare` 에 model 만 등록해 optimizer·scheduler state 가 checkpoint 에 없다. **중간 재개가 불가능**했다 (5~6시간 실행 중 죽으면 처음부터) |
| **C-2** | `save_epoch: 500` 이 한 번도 발동하지 않는다 |
| **C-3** | 마지막 학습 로그가 `nan` 으로 찍힌다 |
| **C-4** | `--res` 가 `type=bool` 이라 항상 True 다 |

### 지표 — 학습 중 수치가 논문과 다른 구현

| | |
|---|---|
| **D-1** | best 선택 기준이 파이썬 자체 구현이라 논문 Table 과 비교 불가 |
| **D-2** | `ERGAS` 가 참조(GT)가 아니라 **예측의 밴드 평균으로 나눈다.** 예측이 어두울수록 값이 커지는 방향으로 편향된다 |
| **D-3** | 로그의 `Q4` 는 8밴드에서도 앞 4밴드만 쓴다 |
| **D-4** | full-res 의 `ERGAS`/`SCC` 는 PAN 을 참조로 삼는 비표준 진단값이다 |
| **D-5** | 논문은 HQNR 인데 코드는 QNR 이다 |

### 평가 방법론

| | |
|---|---|
| **E-1** | **검증셋이 로드만 되고 쓰이지 않는다.** best 를 테스트셋으로 골라 테스트 수치가 낙관적으로 부풀려진다 |

### 잠재 버그 (현 config 에선 발동하지 않음)

| | |
|---|---|
| **B-1 ~ B-5** | `max_pixel` 을 feeder 인자로 넘기면 AttributeError, 미지원 센서명이면 `max_pixel=1.0` 으로 조용히 진행, split 토큰 없으면 미정의, `augment_without_gt` 에서 `lpan` 만 crop 되지 않음, `tensor2img` 가 batch=1 을 가정 |

### 논문 수치 중 설명되지 않는 것

| | |
|---|---|
| FLOPs | 논문 주장 **79.03 G**. 재구성본은 161.9 G 이고, AttnBlock 을 전부 빼도 125.9 G 라 닿지 않는다. 논문이 측정 도구도 MACs/FLOPs 규약도 명시하지 않고 ablation 표에 FLOPs 열이 없어 내부 대조도 불가능하다 |
| Params 계수 | 논문의 계수 방법 자체는 정확하다 — Table 10 의 CANConv 행 0.79 M 은 배포 가중치 실측 0.787 M 과 0.4% 안에서 맞는다 |
