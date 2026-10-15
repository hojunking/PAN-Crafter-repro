# PAN-Crafter 후속 24시간 아키텍처 탐색 계획 (2026-08-28)

## 0. 문서 상태와 목적

> **실행 전 검토용 계획서.** 아직 실험 결과가 아니다. 실행 구성과 우선순위를 검토한 뒤 확정한다.

최신 [`WV3-s1`·`WV3-s2` 결과 시트](https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit?gid=5032275#gid=5032275)를 기준으로,
약 24시간 동안 추가로 탐색할 아키텍처를 정한다.

이번 단계의 목표는 논문 구조를 더 충실하게 재현하는 것이 아니다.

> **성능을 조금 허용 가능한 범위에서 낮추더라도, 비용 대비 기여가 낮은 모듈과 고해상도 연산을 과감히 없애고 새로운 Pareto 지점을 찾는다.**

집중 질문은 다음 세 가지다.

1. 현재 최상 Pareto 후보인 `c6_c4d124`가 다른 서버에서도 재현되는가?
2. full-resolution stage와 대칭 decoder를 더 없앨 수 있는가?
3. 9ch에서도 같은 제거 결론이 유지되며, 거의 새로 설계한 저해상도 backbone이 더 좋은 효율을 내는가?

---

## 1. 최신 결과에서 확정된 판단

### 1.1 두 서버 공통 실행

`WV3-s1`과 `WV3-s2`의 공통 실행 평균이다. 추론시간은 서버 간 절대값이 다르므로 평균 자체보다
각 서버 내부 speedup을 우선한다.

| 구성 | Params | 평균 ERGAS↓ | 평균 HQNR↑ | 평균 추론 | 판단 |
|---|---:|---:|---:|---:|---|
| `c0_hqnr` 전체 구조 | 7.173M | 2.0811 | 0.9539 | 20.95ms | 기준 |
| `c1_nopan` PAN K/V 제거 | 6.225M | **2.0749** | 0.9539 | 16.04ms | **PAN branch 삭제 가능** |
| `c3b_btl` bottleneck attention 1개 | 4.985M | 2.0906 | **0.9541** | 11.80ms | 보수적 후보 |
| `c3e_enc` encoder attention 1개 | 4.985M | 2.0870 | 0.9532 | 13.26ms | `c3b`보다 느림 |
| `c2_encbtl` attention 2개 | 5.605M | 2.0880 | 0.9519 | 14.05ms | 단일 attention 대비 실익 불명 |
| `c4_noattn` attention 0개 | 4.364M | 2.1142 | 0.9537 | **11.22ms** | 성능 약 −1.6%, 속도 약 1.87배 |
| `m1_single` PAN-mode 제거 | 7.173M | 2.1180 | 0.9552 | 21.42ms | 추론 효율 이득 없음 |

### 1.2 현재 최상 Pareto 후보

`c6_c4d124`는 아직 s1 한 번만 측정됐다.

| 구성 | ERGAS↓ | HQNR↑ | Params | 추론 | 메모리 | 학습 |
|---|---:|---:|---:|---:|---:|---:|
| `c0_hqnr` | 2.0734 | 0.9542 | 7.173M | 16.76ms | 864.8MB | 5.21h |
| **`c6_c4d124`** | **2.0826** | 0.9536 | **3.772M** | **6.63ms** | **283.3MB** | **2.40h** |

`c6`의 s1 기준 변화:

- ERGAS: **+0.44%**
- params: **−47.4%**
- 추론: **2.53배**
- 메모리: **−67.2%**
- 학습시간: **2.17배 단축**

더 중요한 점은 `c4_noattn`의 depth를 `[2,2,4] → [1,2,4]`로 줄였을 때
ERGAS가 **2.1228 → 2.0826**으로 좋아졌다는 것이다. full-resolution ResBlock은 단순 잉여를 넘어
최적화를 방해할 가능성이 있다.

### 1.3 더 투자하지 않을 갈래

다음 구성은 새 결과가 없으면 추가 반복하지 않는다.

- `enc+btl` attention 2개
- `c5_c2d124`
- `c7_c1w96`
- `c8_c4w96`의 기존 depth
- MARs PAN-mode 제거의 추가 탐색

특히 `c8_c4w96`은 2.462M이지만 ERGAS 2.2004, 추론 6.62ms다. `c6`은 추론시간이
6.63ms로 같으면서 ERGAS 2.0826이므로, **depth를 줄이기 전에 width를 줄이는 방식은 비효율적**이다.

따라서 제거 순서는 다음으로 고정한다.

```text
PAN K/V 제거 및 attention 최소화
→ full-resolution depth 제거
→ decoder 비대칭화
→ width 축소
```

---

## 2. 수치 해석 규칙

### 2.1 서버·seed 변동

공통 구성의 s1–s2 ERGAS 차이는 약 0.7~1.1%까지 나타난다. 따라서:

- 0.3~0.5% 차이로 세부 순위를 만들지 않는다.
- 같은 서버 안에서 기준선과 비교한다.
- 1차 탐색은 50K·seed 2025 한 벌로 한다.
- 최종 후보 2개만 다른 서버 또는 seed 2026으로 확인한다.
- 대응표본 `p<0.05`만으로 seed·환경 변동을 이겼다고 주장하지 않는다.

### 2.2 공통 학습·평가 조건

새 실험은 다음을 고정한다.

```text
norm             = LayerNorm
crop             = False
dropout          = 0.0
iterations       = 50K
seed             = 2025 (1차 탐색)
optimizer        = AdamW, lr 1e-4, wd 0.01
scheduler        = cosine + warmup 100
checkpoint 선택 = val-ERGAS
RR 평가          = 20장, DLPan 프로토콜
FR 평가          = index 12–19, 복구 lpan
```

모든 실험은 params뿐 아니라 같은 서버에서 다음을 측정한다.

```text
ERGAS / SAM / Q2n / PSNR / SSIM / SCC
D_lambda / D_s / HQNR
inference latency / peak memory / training time
```

HQNR 또는 D_s 단독 개선은 채택 근거로 쓰지 않는다. PAN 주입을 줄이면 자동으로 좋아질 수 있으므로,
주 판정은 **ERGAS·SAM + 실측 추론시간**이다.

---

## 3. 9ch 검증

11ch가 추가하는 것은 `↑LPAN`과 `PAN−↑LPAN` 두 채널이며 params 증가는 2,304개뿐이다.
따라서 9ch는 효율 손잡이라기보다 구조 단순화·논문 조건·입력과 attention의 상호작용을 확인하는 축이다.

| ID | 입력 | 구조 | 확인 질문 | 우선순위 |
|---|---|---|---|---:|
| **`N3_9_d124_noattn`** | 9ch | W128·d124·attention 0 | 현재 최상 `c6`가 9ch에서도 유지되는가 | **1** |
| `N2_9_d224_noattn` | 9ch | W128·d224·attention 0 | attention 제거와 9ch의 직접 상호작용 | 2 |
| `N1_9_nopan` | 9ch | W128·d224·PAN K/V 제거·attention 3 | 명시적 PAN 고주파 없이 PAN K/V를 제거해도 되는가 | 3 |

기존 `s1_A0` 9ch 전체 구조가 ERGAS 2.0436을 기록했으므로 9ch 자체의 성능 잠재력은 충분하다.
다만 11ch는 FR의 `D_lambda`를 개선하는 경향이 있고 비용이 사실상 없으므로, 9ch가 더 낫거나 동급이라는
증거가 없으면 최종 배포 입력은 11ch를 유지한다.

---

## 4. 기존 U-Net 뼈대에서의 과감한 축소

### 4.1 full-resolution stage 제거

| ID | Width | Depth | Attention | Input | 예상 Params | 목적 |
|---|---:|---|---:|---:|---:|---|
| **`R1_w128_d024_noattn`** | 128 | `[0,2,4]` | 0 | 11ch | 약 3.18M | full-resolution ResBlock 완전 제거 |
| **`R2_w128_d014_noattn`** | 128 | `[0,1,4]` | 0 | 11ch | 약 2.6M | mid-resolution도 축소, bottleneck 집중 |
| `R1_9_w128_d024_noattn` | 128 | `[0,2,4]` | 0 | 9ch | 약 3.18M | 최소 입력과 stage 제거 결합 |

`depth[0]=0`을 기존 코드가 허용하지 않으면 full-resolution ResBlock list를 `Identity`로 만드는 옵션을
추가한다. bottleneck depth는 비용이 싸므로 우선 4를 유지한다.

### 4.2 depth 축소 후 width 탐색

LayerNorm은 GroupNorm의 32배수 width 제약이 없다. 따라서 기존에 못 본 W112와 W80을 포함한다.

| ID | Width | Depth | Attention | Input | 예상 Params | 성격 |
|---|---:|---|---:|---:|---:|---|
| **`R3_w112_d124_noattn`** | 112 | `[1,2,4]` | 0 | 11ch | 약 2.9M | 가장 유력한 중간 폭 |
| **`R4_w96_d124_noattn`** | 96 | `[1,2,4]` | 0 | 11ch | 약 2.1M | 공격적 균형 후보 |
| `R5_w80_d124_noattn` | 80 | `[1,2,4]` | 0 | 11ch | 약 1.5M | 극단 폭 지점 |
| `R6_w96_d024_noattn` | 96 | `[0,2,4]` | 0 | 11ch | 약 1.8M | stage·width 동시 축소 |

기존 W96 결과는 full-resolution depth가 큰 상태에서 나왔으므로, W96 자체를 종결한 것이 아니다.
이번에는 반드시 d124 또는 d024 위에서 다시 본다.

---

## 5. 대칭 U-Net을 버리는 새 구조 — `LiteU-Asym`

기존 attention 제거에서 encoder가 decoder보다 중요했다. 현재 depth 설정은 encoder와 decoder를
대칭으로 줄이므로, 다음 단계에서는 양쪽 depth를 분리한다.

### 5.1 기본 형태

```text
Input: PAN + upsampled MS, 9ch 또는 11ch
Attention: 없음

Encoder
  full-resolution block: 0 또는 1
  H/2 block:            1

Bottleneck
  block:                4

Decoder
  H/2 block:            0 또는 1
  full-resolution block: 0

Output
  3×3 residual head
  + upsampled MS residual base
```

### 5.2 비교 case

| ID | Encoder | Bottleneck | Decoder | 목적 |
|---|---|---:|---|---|
| **`A1_asym_114_10`** | full 1, H/2 1 | 4 | H/2 1, full 0 | 보수적 비대칭 |
| **`A2_asym_014_10`** | full 0, H/2 1 | 4 | H/2 1, full 0 | full-resolution body 제거 |
| `A3_asym_014_00` | full 0, H/2 1 | 4 | H/2 0, full 0 | single-head 극단 |

세 case를 모두 처음부터 돌리지 않는다. `A1`을 먼저 실행하고 결과가 유망할 때 `A2`, `A3`로 진행한다.

---

## 6. 거의 새로 설계하는 초경량 구조 — `LR-Fuse`

고해상도 feature map에서 backbone을 돌리지 않고, PAN을 MS grid로 접어 대부분의 연산을
1/16 면적에서 수행한다.

### 6.1 구조

```text
PAN 256×256×1
  → PixelUnshuffle ×4
  → 64×64×16

LRMS 64×64×8
  → PAN feature와 concatenate

Low-resolution backbone
  width 64 또는 96
  ResBlock 6개
  LayerNorm + SiLU
  attention 없음

Output
  → PixelShuffle ×4
  → HRMS residual 256×256×8
  → upsampled MS에 더함
```

### 6.2 입력 variant

| ID | 입력 | Backbone | 목적 |
|---|---|---|---|
| **`L1_9_lr_fuse_w64`** | raw PAN + MS만 사용 | W64·6 blocks | 9ch 철학의 초경량 모델 |
| **`L1_11_lr_fuse_w64`** | PAN low/high-frequency 정보 추가 | W64·6 blocks | explicit HF 효과 |
| `L2_11_lr_fuse_w96` | 11ch 계열 | W96·6 blocks | 저해상도 backbone 용량 상한 |

초기 목표선:

```text
Params  : 1~2M
추론    : 3~4ms 이하
ERGAS   : 2.20 이하
```

`L1`이 ERGAS에서 `c6`보다 5% 정도 나빠도 3ms 이하라면 별도의 ultra-lite Pareto 지점으로 인정한다.

---

## 7. 24시간 실행 큐

아래는 **s1과 s2를 병렬로 약 24시간 사용**하는 기준이다. 실제 종료시각은 각 서버의 실측 속도로 갱신한다.

### 7.1 s1 — 탐색 서버

| 순서 | 실행 | 예상 학습 | 분기 |
|---:|---|---:|---|
| 1 | `N3_9_d124_noattn` | 약 2.4h | 9ch 최상 후보 확보 |
| 2 | `R1_w128_d024_noattn` | 약 2h | full-resolution stage 제거 판정 |
| 3 | `R3_w112_d124_noattn` | 약 2.2h | 중간 width 탐색 |
| 4 | `R4_w96_d124_noattn` | 약 2h | 공격적 width 탐색 |
| 5 | `R2_w128_d014_noattn` | 약 2h | bottleneck 집중 확인 |
| 6 | `A1_asym_114_10` | 약 2h | 비대칭 U-Net 판정 |
| 7 | `L1_9_lr_fuse_w64` | 추정 1~2h | 새 구조 9ch |
| 8 | `L1_11_lr_fuse_w64` | 추정 1~2h | 새 구조 11ch |

구현·smoke test를 포함해 약 20~24시간을 예상한다.

### 7.2 s2 — 검증 서버

| 순서 | 실행 | 목적 |
|---:|---|---|
| 1 | **`c6_c4d124` 동일 설정** | 현재 Pareto 선두의 환경 재현 |
| 2 | `N2_9_d224_noattn` | 9ch×attention 제거 기준점 |
| 3 | `N3_9_d124_noattn` | 9ch 최종 후보의 서버 재현 |
| 4 | s1에서 먼저 끝난 `R1` 또는 `R3` | radical 후보 재현 |
| 5 | 남은 6~8시간에 s1 최상위 후보 seed 2026 | seed 확인 |

s2는 넓은 탐색이 아니라 **기준선·최종 후보 독립 확인** 역할을 유지한다.

### 7.3 총 24 GPU-hour만 가능한 경우

두 서버 합산 예산이 24 GPU-hour라면 다음 여섯 개만 실행한다.

```text
1. s2에서 c6 재현
2. N3: 9ch d124 no-attention
3. R1: W128 d024 no-attention
4. R3: W112 d124 no-attention
5. R4: W96 d124 no-attention
6. A1: asymmetric LiteU
```

---

## 8. 조기 중단과 채택 기준

### 8.1 smoke test

새 구조는 1K 이전에 다음을 확인한다.

```text
입출력 shape
residual base 일치
NaN/Inf
params/FLOPs
peak VRAM
iteration time
초기 output이 zero_module 때문에 공허하게 동일하지 않은지
```

### 8.2 50K 결과 판정

| 등급 | 조건 |
|---|---|
| **주력** | ERGAS ≤2.12이며 추론 ≤4ms 또는 기준 대비 ≥3배 speedup |
| **균형** | ERGAS ≤2.10, 2~3M, 추론 5~6ms |
| **초경량** | ERGAS ≤2.25, ≤2M, 추론 ≤4ms |
| **탈락** | 기존 후보보다 품질·params·latency가 모두 열세 |

추가 규칙:

- 같은 서버에서 ERGAS 차이 1% 미만이면 성능 동급으로 본다.
- 동급이면 실측 latency와 메모리가 낮은 쪽을 택한다.
- 25K와 50K를 가로 비교하지 않는다.
- 최종 후보 2개만 seed 2026 또는 다른 서버에서 확인한다.
- 새 초경량 구조는 주력 모델과 별도의 Pareto 지점으로 평가한다.

---

## 9. 예상 의사결정

현재 가장 가능성이 높은 결과는 다음 세 지점이다.

| 용도 | 예상 후보 |
|---|---|
| 품질 우선 경량 | `c3b_btl` 또는 s2에서 재현된 `c6` |
| 균형형 | `R1_w128_d024_noattn` 또는 `R3_w112_d124_noattn` |
| 초경량 | `R6_w96_d024_noattn` 또는 `L1_lr_fuse_w64` |

이번 24시간의 가장 중요한 질문은 다음이다.

> **full-resolution stage를 완전히 제거해도 되는가?**

그다음 질문은 다음이다.

> **대칭 decoder를 버리고 대부분의 용량을 저해상도 bottleneck에 둘 수 있는가?**

CM3A 위치와 PAN K/V를 더 세분하는 단계는 종료한다. 새 결과가 이를 명확히 뒤집지 않는 한,
이후 탐색은 no-attention·early-fusion을 기본 구조로 진행한다.


---

## 10. (추기 2026-08-28) 검토 확정 사항 — 이 절이 위 본문과 다르면 이 절을 따른다

1. **§2.2 의 `checkpoint 선택 = val-ERGAS` 는 폐기한다. best 선택은 HQNR(공식 12-19) 유지.**
   기존 11개 캠페인 수치가 전부 HQNR 선택 체크포인트(중반부 epoch)라, 선택 기준을 바꾸면
   신구 비교가 새 실험 쪽으로 기울어진다. 해석 축(ERGAS·SAM + 실측 추론시간)은 본문대로.
2. **R2(d014) 삭제.** `dec_depth` 옵션 도입으로 R-계열과 A-계열이 config 차이로 통일됐고,
   R2 는 A2 에 full-res 융합 블록 1개를 얹은 것에 불과하다. 그 여유로 조건부였던
   **A2·R6 을 s1 큐 꼬리에 무조건 편성** (§9 의 R6 예상후보 누락 모순도 해소).
3. **`depth[0]=0` 의 의미가 바뀌었다.** 종전 코드는 decoder 에 융합 블록 1개가 강제로
   남았지만, `dec_depth` 미러 시맨틱에서는 양쪽 모두 0 — R1 은 이제 문자 그대로
   full-res ResBlock 완전 제거다.
4. **LR-Fuse 는 dual MARs 로 학습한다.** 출력이 항상 8ch 라(PAN mode 는 8밴드 broadcast
   target) train.py 무수정으로 성립. §6 검토에서 우려한 m1 단일모드 교란(ERGAS +1.4%)을 배제.
5. 실측 params (build 확인): R1 2.999M · R3 2.892M · R4 2.129M · R6 1.693M ·
   A1 2.703M · A2 2.407M · A3 1.930M · L1 0.544M · L2 1.148M · N3 3.770M.
   본문 추정치(§4)와 다르면 이 값을 인용한다. L-계열은 §6 목표선(1~2M)보다 작다.
6. N1_9_nopan 은 제외 확정. N2 는 s2 몫. R5(w80)·A3·L2 는 config 만 두고 큐 결과로 결정.
