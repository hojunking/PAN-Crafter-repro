# s1 — 4–6M Teacher Architecture 탐색 계획 (2026-09-01)

> 상태: 실행 전 계획  
> 서버: **s1 전용**  
> 목표: 4–6M 범위에서 강한 teacher 골격을 찾는다. **s2는 동일 구조를 별도로 학습하며 s1 checkpoint를 사용하지 않는다.**

| 대상 | 확정 구성 | 목적 | 예상 시간 |
|---|---|---|---:|
| 첫 teacher 후보 | **Case 0: W160 · d122 · clean MS-only** | 약 5M의 넓고 얕은 teacher를 가장 먼저 확인 | 2.2–2.5h |
| architecture search | Res-U-Net 한 계열의 width/depth matched-budget 비교 | 모듈 효과가 아니라 용량 배분을 측정 | 15–19h |
| 최종 대조 | architecture winner의 dual MARs 1회 | PAN reconstruction 필요성 최종 확인 | 3.6–4.7h |

이 문서의 파라미터와 시간은 기존 실측에서 외삽한 **실행 전 추정값**이다. 실제 모델 build의 전체 trainable parameter와 첫 1K throughput을 최종 기준으로 사용한다.

---

## 1. 최신 결과가 지지하는 방향

동일한 `depth=[1,2,4]`에서 width를 늘리면 RR(reduced-resolution) reconstruction이 일관되게 좋아졌다.

| Width | ERGAS↓ | HQNR↑ | Params |
|---:|---:|---:|---:|
| 96 | 2.1418 | 0.9561 | 2.1285M |
| 112 | 2.1166 | 0.9540 | 2.8918M |
| 128 | 2.0826 | 0.9536 | 3.7719M |

반대로 W128에서 bottleneck을 늘린 `d126`은 ERGAS 2.0839/HQNR 0.9537로 `d124`보다 좋아지지 않았다. `d122`는 params를 약 15.7% 줄이고도 ERGAS 2.0884/HQNR 0.9539로 거의 유지됐다. 따라서 이번 탐색의 주가설은 다음이다.

> 포화된 bottleneck block에 쓰던 파라미터를 전체 width로 옮기면 4–6M teacher의 성능이 더 좋아진다.

또한 W128에서 dual MARs와 MS-only의 결과는 사실상 같았다.

| 학습 | ERGAS↓ | HQNR↑ | Train |
|---|---:|---:|---:|
| dual MARs | 2.0826 | 0.9536 | 2.40h |
| MS-only | 2.0812 | 0.9539 | 1.48h |

따라서 architecture 탐색은 빠른 **clean MS2 plain**으로 통일하고, 최종 winner에 대해서만 dual MARs를 한 번 비교한다.

Swin, CM3A, SE, `d126` 이상 증설은 이미 효과가 없거나 비용 대비 열세였으므로 이번 queue에서 제외한다.

---

## 2. 전 case 공통 설정

```yaml
input_channels: 11
crop: false
attention_locations: []
pan_input: true

task: ms_only_plain
pan_reconstruction: false
pan_mode: false
mode_modulation: false

iterations: 50000
seed: 2025
optimizer: AdamW
lr: 1.0e-4
weight_decay: 0.01
scheduler: cosine
warmup_steps: 100
checkpoint_selection: val_hqnr
```

`MS-only`는 PAN 입력을 제거한다는 뜻이 아니다. PAN은 spatial guidance로 계속 입력되며, HRMS만 출력·복원한다. 제거하는 것은 PAN reconstruction branch/loss와 mode-conditioning이다.

Architecture search 중에는 다음을 넣지 않는다.

```text
uncertainty joint loss
KD / feature KD
SiS / edge loss
GT-variance loss
PAN reconstruction
```

---

## 3. Case 0 — 첫 s1 teacher 후보

### `S1_T00_W160_D122_MS2`

```yaml
width: 160
depth: [1, 2, 2]
attention_locations: []
input_channels: 11
crop: false
task: ms_only_plain
```

| 항목 | 예상 |
|---|---:|
| Params | 약 4.958M |
| Clean MS2 50K | 2.2–2.5h |
| Smoke·RR/FR 평가 | 0.3–0.5h |

이전 계획의 0번이었던 W96 student control은 `S1_C0`으로 분리하고, 첫 실제 teacher 후보인 W160-d122를 0번으로 둔다. s1에서는 architecture 탐색만 수행하며 uncertainty head는 붙이지 않는다.

s2는 이 checkpoint를 복사하지 않는다. 같은 모델 정의와 config를 사용해도 initialization, optimizer state, 학습 데이터 순서와 checkpoint는 s2에서 독립적으로 생성한다.

---

## 4. Architecture cases

### 4.1 핵심 네 case

| 순서 | ID | Width | Depth | 예상 Params | MS2 50K | 질문 |
|---:|---|---:|---|---:|---:|---|
| 0 | `S1_T00_W160_D122_MS2` | 160 | `[1,2,2]` | 4.958M | 2.2–2.5h | 약 5M wider-shallow teacher의 첫 확인 |
| 1 | `S1_T01_W144_D124_MS2` | 144 | `[1,2,4]` | 4.769M | 1.8–2.0h | 같은 5M급에서 bottleneck depth가 유리한가 |
| 2 | `S1_T02_W160_D124_MS2` | 160 | `[1,2,4]` | 5.883M | 2.3–2.5h | W160에서 bottleneck 두 블록의 추가 가치 |
| 3 | `S1_T03_W176_D122_MS2` | 176 | `[1,2,2]` | 5.994M | 2.7–3.0h | 6M 한계에서 width 확대의 성능 상한 |

Case 3은 build 실측이 6.000M을 넘으면 실행하지 않는다. 이때는 `W168-d122` 약 5.464M으로 대체한다.

### 4.2 조건부 중간점 — 둘 중 하나만

| ID | 구조 | 예상 Params | 실행 조건 |
|---|---|---:|---|
| `S1_T04A_W152_D123_MS2` | W152 · `[1,2,3]` | 4.894M | T00과 T01이 서로 다른 지표 방향을 보일 때 |
| `S1_T04B_W168_D123_MS2` | W168 · `[1,2,3]` | 5.973M | T02와 T03이 서로 다른 지표 방향을 보일 때 |

중간점 두 개를 모두 돌리지 않는다. 어느 matched-budget 구간에서 depth/width 결론이 불명확한지를 본 뒤 하나만 선택한다.

### 4.3 Capacity scaling 실패 시 fallback

T00과 T01이 W128 anchor를 모두 개선하지 못하면, 나머지 Res-U-Net 세분화보다 backbone family 대조가 중요해진다.

```text
S1_TF_NAFU_MS2
- 11ch early fusion
- standard NAFBlock encoder–middle–decoder
- residual to upsampled MS
- build-only width 조정으로 4.8–5.5M
- clean MS-only
```

이 경우 T04 중간점 하나를 생략하고 NAF-U 한 벌을 실행한다. 구현과 smoke가 1시간을 넘기면 fallback 자체를 다음 캠페인으로 넘긴다.

---

## 5. Controls

아래 두 control은 teacher 후보가 아니므로 4–6M 제한 밖이다.

| ID | 구조 | 이유 | 예상 |
|---|---|---|---:|
| `S1_C0_W96_D124_MS2` | 최종 student와 동일 | 같은 s1에서 teacher headroom 계산 | 1.2–1.4h |
| `S1_A0_W128_D124_MS2` | 기존 c6 골격의 clean MS2 | width scaling의 정확한 anchor | 1.4–1.6h |

이는 seed 반복 실험이 아니다. 두 clean MS2 control이 이미 동일 commit/evaluator로 존재하면 checkpoint를 재사용하고 queue에서 생략한다.

---

## 6. 최종 PAN reconstruction 대조

Architecture winner 한 개만 동일 구조로 dual MARs 재학습한다.

```text
S1_T05_<WINNER>_DUAL
- winner와 width/depth/11ch/crop 조건 동일
- MS/PAN reconstruction과 mode-conditioning만 복원
- uncertainty/KD/SiS 없음
```

예상 학습 시간은 3.6–4.7h다.

판정:

- dual의 HQNR이 plain보다 `+0.011` 이상이면 dual 채택
- HQNR 동급에서 dual이 ERGAS를 0.5% 이상 개선하고 SCC/SAM도 같은 방향이면 dual 채택
- 그 외에는 더 단순하고 빠른 clean MS-only teacher 채택

s2는 동일 winner 여부와 관계없이 자체 queue에서 W160-d122 teacher를 독립 학습한다. 따라서 s1 결과 대기나 checkpoint 전달이 없다.

---

## 7. 실행 순서와 예상 시간

```text
0.0–2.5h    T00 W160-d122 clean MS2
2.5–5.5h    C0/A0 controls — 기존 exact checkpoint가 있으면 생략
5.5–10.0h   T01 → T02
10.0–13.0h  T03
13.0–15.5h  필요한 T04 중간점 또는 NAF fallback 한 벌
15.5–20.2h  architecture winner의 dual MARs 대조
20.2–23.5h  RR/FR 평가, paired scene 분석, 실패 복구
23.5–24.0h  buffer
```

| 범위 | 예상 총시간 |
|---|---:|
| exact controls 재사용, NAF 없음 | 17–20h |
| controls 신규 + 중간점 1벌 | 20–23h |
| NAF 구현 지연 또는 재실행 포함 | 23–26h |

---

## 8. 판정 규칙

HQNR을 최우선으로 하되 확정된 변동 범위를 무시하지 않는다.

1. `|ΔHQNR| > 0.011`: HQNR로 결정
2. `|ΔHQNR| ≤ 0.011`: 동급으로 보고 SCC → ERGAS → SAM/Q2n
3. 전 지표 동급: 더 작은 모델

최종 KD teacher로 사용할 최소 조건:

- 같은 s1의 W96 clean-MS2 student보다 HQNR이 0.011 이상 나쁘지 않음
- HQNR 동급이면 ERGAS 최소 1% 개선, 2% 개선 권장
- SCC 비열화 없음
- RR 20장 중 최소 12장 이상에서 student보다 낮은 reconstruction error

이 조건을 만족하지 못하면 4–6M teacher의 capacity가 커졌더라도 **KD에 전달할 headroom 부족**으로 기록한다.

---

## 9. 기록할 항목

```text
RR: ERGAS, SAM, PSNR, SSIM, SCC, Q2n, RMSE, CC
FR: HQNR, D_lambda, D_s
Cost: params, FLOPs, inference, memory, train time
Reproducibility: commit, full config, checkpoint hash, evaluator version
```
