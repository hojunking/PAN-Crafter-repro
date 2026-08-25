# 논문 서술과 배포 코드의 불일치 목록 (2026-08-25)

## 요약

배포 코드를 논문 본문·Figure 3·식과 한 줄씩 대조해 **14곳**의 불일치를 찾았다.
방침은 **불일치가 있으면 논문을 따른다** 이고, 재구성본(`model/pancrafter_paper.py`)에
반영 상태를 함께 적는다.

| 분류 | 개수 | 반영됨 | 미반영 |
|---|---:|---:|---:|
| 구조 (Figure 3 / §3.2) | 4 | 4 | 0 |
| 블록 내부 (Eq 5–6) | 3 | 3 | 0 |
| CM3A (Eq 10–12 / §3.3) | 4 | 3 | **1** |
| 입력 (§3.2) | 1 | 1 | 0 |
| 학습 (§4.2) | 1 | 0 | **1** |
| 판정 보류 | 1 | — | — |

**효과는 확인됐다.** 반영분만으로 WV3 ERGAS 가 배포본 재현 2.1643 → **2.1205**
(−2.02%, p=0.005)로 개선됐고, 논문과의 격차가 **6.09% → 3.94%** 로 줄었다.
[재구성 보고서](2026-08-24_paper-faithful-rebuild.md) 참고.

---

## A. 구조 — Figure 3 / §3.2

| # | 항목 | 논문 | 배포 코드 | 반영 |
|---|---|---|---|:--:|
| 1 | spatial scale | **3** (Fig 3) | 4 | ✅ |
| 2 | Down/UpConv | **2 / 2** (Fig 3) | 3 / 3 | ✅ |
| 3 | AttnBlock 수 | **3** (Fig 3) | 5 | ✅ |
| 4 | AttnBlock 배치 | *"low- and mid-resolution stages incorporate both ResBlock and AttnBlock, while **high-resolution stages use only ResBlock**"* | H/2(고해상도)에도 CM3A 배치 | ✅ |

4번이 1–3번의 근거다. 4단계를 high/mid/low 로 나누면 H/2 는 high 이므로 CM3A 가 없어야 한다.

---

## B. 블록 내부 — Eq (5)–(6)

논문 ResBlock:

```
x ← Conv(SiLU(LN(x)))
x ← x + Conv(SiLU(Modulate(LN(x); mode)))
Modulate(x; MS) : x ← (1 + γ_ms) ⊙ x + β_ms
```

| # | 항목 | 논문 | 배포 코드 | 반영 |
|---|---|---|---|:--:|
| 5 | ResBlock 정규화 | **LayerNorm** (Eq 5) | `GroupNorm32(32, C)` | ✅ |
| 6 | mode modulation | **γ, β ∈ R^C 직접 학습** (Eq 6). 블록당 2×2×C = 512 | mode token + 블록별 `Linear(128, 2C)`. 블록당 **33,024** | ✅ |
| 7 | dropout | 수식에 없음 | `dropout: 0.2` | ✅ (0.0) |

**5번은 파라미터가 중립이라 늦게 발견했다.** LayerNorm 과 GroupNorm32 는 둘 다 2C 이므로,
params 7.17 M 을 맞추는 방식의 탐색으로는 드러나지 않는다. 식을 직접 대조해서야 찾았다.
효과는 작지 않다 — 이것만으로 ERGAS −1.34% (p=0.040).

6번이 파라미터 격차의 최대 몫이다. 15개 블록에서 약 **0.49 M**.

---

## C. CM3A — Eq (10)–(12) / §3.3

| # | 항목 | 논문 | 배포 코드 | 반영 |
|---|---|---|---|:--:|
| 8 | `[K_pan \| V_pan]` | **결합 conv 1개** — `Conv([I_pan^rep,↓ \| x])` (Eq 11) | `k_pan`, `v_pan` **별도 conv 2개**. 입력도 다르다 (`v_pan` 에만 고주파 `pan−lpan` 채널) | ❌ |
| 9 | local attention k | **k = 3** (전역, §4.2) | bottleneck 만 **k = 1** | ✅ |
| 10 | key 별칭 | Eq (10)/(11): `x_pan = LocalAttn(Q, K_pan, V_pan)` | `q, k_pan, k_pan = ...` — 좌변에 `k_pan` 이 두 번. **PAN 브랜치가 MS key 로 attention** 하고 `k_ms` 는 뒤섞인다 | ✅ |
| 11 | LocalAttn | Sec 3.3: *"attention scores within the k×k local receptive field"* | `reset_parameters()` **호출부 없음** → `dep_conv` 가 shift kernel 이 아니라 랜덤 초기화로 학습된다 | ✅ |

10번은 죽은 파라미터가 611,584개(전체의 6.13%), 11번은 7,200개다.

**8번만 미반영이다.** 논문대로 결합 conv 로 바꾸면 params 가 7.1707 → **7.2122 M**
(논문 주장 대비 +0.59%)이 되어 지금의 +0.01% 일치가 깨진다. 그래서 보류했으나,
방침(논문 우선)에 따르면 적용 대상이다 — §E 참고.

### 일치하는 것 (대조 완료)

- `Q = Conv([I_ms^lr,↓ | x])` (MS mode) / `Conv([I_pan^rep,↓ | x])` (PAN mode) — 코드의
  `cond = pan_·(1−s) + ms·s` 가 mode 별로 같은 동작을 한다
- `[K_ms | V_ms] = Conv([I_ms^lr,↓ | x])` (Eq 10) — 결합 conv 로 일치
- `x_attn = α1 ⊙ x_ms + α2 ⊙ x_pan`, α ∈ R^C mode 별 (Eq 8)
- `x ← x + x_attn`, `x ← x + FFN(LN(x))` — AttnBlock 의 LN 위치
- MARs 손실 (Eq 4), λ = 1.0

---

## D. 입력 — §3.2

| # | 항목 | 논문 | 배포 코드 | 반영 |
|---|---|---|---|:--:|
| 12 | 입력 채널 | *"channel-wise concatenation of I_pan and I_lrms"* → **9 ch** | `PAN, ↑LPAN, PAN−↑LPAN, ↑MS` → **11 ch** | ✅ |

---

## E. 학습 — §4.2

| # | 항목 | 논문 | 배포 코드 | 반영 |
|---|---|---|---|:--:|
| 13 | 증강 | *"random horizontal/vertical flips, 90-degree rotations, and **random cropping**"* | crop 후 **`cv2.resize` 로 원크기 복원** = scale jitter | ❌ |

논문은 random cropping 을 한다고 쓴다. 문제는 **구현이 crop 이 아니라는 것**이다.

```python
ratio = (1 - 0.75) * random() + 0.75        # 0.75 ~ 1.0
# gt/lms/ms/lpan/pan 을 ratio 만큼 자른 뒤
gt = cv2.resize(gt, (64, 64), INTER_CUBIC)  # 원래 크기로 되돌린다
ms = cv2.resize(ms, (16, 16), INTER_CUBIC)
```

두 가지가 깨진다.

1. `ms` 와 `gt` 가 **각각 독립적으로** bicubic 리샘플되어 MS→GT 열화관계(MTF + decimation)가
   학습 중에만 왜곡된다. 테스트에는 그 왜곡이 없다.
2. PAN:MS = 4:1 이라는 고정 비율이 최대 1.33 배까지 흔들린다.

학습 패치가 이미 최소 단위(PAN 64×64 / MS 16×16)라 **진짜 crop 을 할 여지가 없다.**
따라서 논문 서술을 그대로 구현하는 것은 불가능하고, 선택지는 둘이다 —
`crop: False`(증강에서 제외) 또는 현행 유지. **아직 어느 쪽도 측정하지 않았다.**

---

## F. 판정 보류

| # | 항목 | 논문 | 배포 코드 |
|---|---|---|---|
| 14 | 잔차 기준선 | Eq (1): *"up-sample it by a factor of 4"* — **방법 미명시** | `F.interpolate(ms, scale_factor=4, mode="bicubic")`. 데이터셋이 제공하는 `lms` 와 다르다 |

`lms` 는 단독 기준선 ERGAS 7.1220, `bicubic(ms,×4)` 는 7.3035 로 **2.5% 차이**가 난다.
논문이 보간 방법을 적지 않아 불일치로 단정할 수 없다. 실험(`d2_lmsbase`)으로만 가릴 수 있다.

---

## 남은 격차와 다음

반영분만으로 **6.09% → 3.94%**. 아직 논문 2.040 에 닿지 않았고, 미반영 2건이 남는다.

| 우선순위 | 항목 | 근거 | 비용 |
|---|---|---|---|
| 1 | **#13 crop** | 학습 파이프라인에서 서술과 코드가 어긋나는 유일한 지점 | 50K 1벌, 5.5h |
| 2 | **#8 `[K_pan\|V_pan]` 결합** | 구조에서 유일한 미반영. params 가 7.2122 M 로 +0.59% 어긋나지만 방침상 논문 우선 | 구현 + 50K 1벌 |
| 3 | #14 잔차 기준선 | 판정 보류지만 2.5% 차이라 확인 가치 있음 | 50K 1벌 |

> **주의.** 25K 에서 잰 시드 2σ 가 1.32% 다. 위 항목들의 예상 효과가 그보다 작을 수 있으므로,
> 유망한 것이 나오면 **시드 3벌**로 확인해야 한다. 지금까지의 개선(−2.02%, −1.34%)도
> 시드 반복 없이 얻은 값이다.

## full-resolution 은 반대로 갔다

| (12–19, 8장) | D_λ↓ | D_s↓ | HQNR↑ |
|---|---:|---:|---:|
| 논문 | 0.016 | 0.027 | **0.958** |
| 재구성 LN | 0.0309 | 0.0342 | 0.9360 |
| 배포본 재현 | 0.0249 | 0.0283 | **0.9475** |

reduced 가 좋아지는 동안 HQNR 이 0.9475 → 0.9360 으로 떨어졌다. 논문은 양쪽 다 좋은데
(2.040 / 0.958) 재구성본은 트레이드오프가 생긴다. **아직 설명되지 않는다.**
