# c6 기반 Swin·CM3A 대조 및 압축 24시간 계획 (2026-08-29)

> **실행 전 검토용 계획서.** 2026-08-28 계획을 새 결과에 맞게 교체한 스냅샷이다.
> 이전 문서는 당시 판단을 보존하며, 실제 실행·결과가 확정되면 별도 결과 로그를 작성한다.

[WV3-s1·WV3-s2 결과 시트](https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit?gid=5032275#gid=5032275)의
최신 결과를 기준으로 앞으로 약 24시간 동안 실행할 case를 정한다.

이번 캠페인은 seed 반복이나 PAN-Crafter 재현이 목적이 아니다.

> **HQNR을 최우선으로 보되, HQNR의 확인된 변동 범위 안에서는 SCC·ERGAS와 실제 비용을 함께 보며
> 기여가 낮은 연산을 제거한다. s1과 s2는 서로 다른 질문을 맡는다.**

---

## 0. 이번 수정의 핵심

2026-08-28 계획에서 다음 네 항목을 폐기하거나 교정한다.

| 기존 항목 | 새 결정 | 근거 |
|---|---|---|
| HQNR 0.001 차이로 분기 | **동급 band를 절대값 0.011로 확대** | 구조가 크게 다른 모델도 0.9475~0.9561에 집중됨 |
| ERGAS는 5% 악화 때만 경고 | **HQNR 동급 시 SCC 다음 tie-break로 사용** | R4의 명목 HQNR 상승과 ERGAS +2.84%가 동시에 발생 |
| LR-TinySwin SW2→SW4 | **SW2 반증 실험 한 벌만 유지** | LR-Fuse 11ch·9ch가 모두 전면 붕괴 |
| HQNR 동급이면 D024 자동 선택 | **SCC→ERGAS로 구조 선택 후 비용 판단** | R1에서 full-resolution 제거의 숨은 품질 비용 확인 |

추가로 c6 bottleneck에 CM3A 하나만 붙인 대조군을 넣어,
Swin의 효과와 단순한 bottleneck attention의 효과를 구분한다.

---

## 1. 최신 결과가 바꾼 전제

아래 수치는 모두 50K 결과다. 추론시간은 서버마다 절대값이 다르므로 같은 서버 안에서만 비교한다.

| Case | Server | ERGAS↓ | SCC↑ | HQNR↑ | Params | Infer | 판단 |
|---|---|---:|---:|---:|---:|---:|---|
| c6 d124 no-attn | s1 | 2.0826 | 0.9908 | 0.9536 | 3.7719M | 6.63ms | 공통 기준 |
| c6 d124 no-attn | s2 | **2.0716** | **0.9909** | 0.9528 | 3.7719M | 10.35ms | s2 anchor 완료 |
| N3 9ch d124 | s1 | 2.0810 | 0.9908 | 0.9475 | 3.7696M | 6.63ms | 9ch도 HQNR 동급 band |
| R1 d024 | s1 | 2.1301 | 0.9901 | 0.9540 | 2.9989M | 3.23ms | 빠르지만 ERGAS +2.28% |
| R3 w112 d124 | s1 | 2.1166 | 0.9903 | 0.9540 | 2.8918M | 5.52ms | 중간 폭 후보 |
| R4 w96 d124 | s1 | 2.1418 | 0.9900 | **0.9561** | 2.1285M | 4.91ms | 명목 HQNR 최고 착시 |
| R6 w96 d024 | s1 | 2.2180 | 0.9889 | 0.9477 | 1.6932M | 3.12ms | 극단 축소 하한 |
| LR-Fuse 11ch | s1 | 4.0221 | 0.9479 | 0.9293 | 0.5439M | 2.28ms | LR-only 실패 |
| LR-Fuse 9ch | s1 | 4.1485 | 0.9437 | 0.8985 | 0.5341M | 2.20ms | LR-only 실패 |

### 1.1 c6는 서버를 넘어 유지됐다

c6의 s1·s2 차이는 HQNR 0.0008, ERGAS 약 0.53%다. 따라서 s2에서 c6를 다시 돌리지 않는다.
앞으로 s2의 모든 case는 이미 완료된 c6-s2를 서버 내부 anchor로 사용한다.

### 1.2 HQNR의 소수점 세 자리 분기는 무효다

R4는 c6보다 HQNR이 0.0025 높지만 ERGAS는 2.84% 나쁘고 SCC도 낮다.
이 차이를 HQNR 개선이라고 부르지 않는다. 확인된 동급 band 0.011 안에서는 다른 품질지표와
비용을 반드시 함께 본다.

### 1.3 full-resolution 제거는 실패가 아니라 trade-off다

R1은 c6보다 ERGAS가 2.28% 나쁘지만 추론이 약 2.05배 빠르고 파라미터가 20.5% 적다.
따라서 R1/D024 계열은 품질 최우선 모델로 자동 선택하지 않되, 효율 Pareto 후보로 유지한다.

### 1.4 LR-only 주력 탐색은 종료한다

LR-Fuse는 1/16 면적 연산으로 속도와 크기를 얻었지만 품질 손실이 너무 크다.
LR-TinySwin은 Swin의 전역 정보 교환이 이 패러다임을 구할 수 있는지 확인하는
반증 case 한 벌만 실행한다. 같은 캠페인에서 SW4로 확대하지 않는다.

---

## 2. 공통 기준 구조와 Swin 정의

### 2.1 c6 backbone

~~~text
Input        : 11ch, 별도 표기 시 9ch
Width        : 128
Depth        : [1, 2, 4]
Attention    : 없음
MARs         : dual

Encoder
  H          : ResBlock ×1
  H/2        : ResBlock ×2

Bottleneck H/4
  ResBlock ×4

Decoder
  H/2        : ResBlock ×2
  H          : ResBlock ×1

Output
  3×3 residual head
  + upsampled MS
~~~

### 2.2 표준 Swin block

외부 의존성을 추가하지 않고 표준 pre-norm Swin block을 직접 사용한다.

~~~text
삽입 위치       : c6 bottleneck ResBlock 뒤
embed dim       : backbone width와 동일
heads           : 4
window          : 8×8
shift           : 0 / 4 교대
MLP ratio       : 2
dropout         : 0
attention drop  : 0
drop path       : 0
normalization   : LayerNorm
position        : relative-position bias
patch merge     : 없음
class token     : 없음
~~~

한 pair는 다음 두 block이다.

~~~text
SW2 = W-MSA → SW-MSA
SW4 = W-MSA → SW-MSA → W-MSA → SW-MSA
SW6 = W-MSA → SW-MSA → W-MSA → SW-MSA → W-MSA → SW-MSA
~~~

C=128, MLP ratio 2일 때 Swin pair의 예상 파라미터는 약 0.2668M이다.
따라서 SW2-ADD 전체는 약 4.0387M으로, PAN-Crafter 7.173M보다 약 43.7% 작다.
최종 파라미터 표에는 예상값이 아니라 build 실측값을 기록한다.

---

## 3. 판정 규칙

### 3.1 학습 중 best checkpoint

~~~text
best checkpoint = HQNR 최대
val-ERGAS 기반 checkpoint 선택은 사용하지 않음
seed 반복       = 하지 않음
iteration       = 50K
~~~

### 3.2 최종 품질 순위

서버별 anchor와 비교하며 다음 순서를 사용한다.

1. HQNR 차이가 절대값 0.011보다 크면 HQNR로 판정한다.
2. HQNR 차이가 0.011 이내면 HQNR 동급으로 본다.
3. HQNR 동급이면 SCC를 비교한다.
4. SCC도 포화되어 방향이 불분명하면 ERGAS로 가른다.
5. 품질이 동급이면 실측 추론시간, 파라미터, FLOPs, peak memory 순으로 작은 쪽을 택한다.

SCC의 0.000x 차이 하나만으로 승자를 선언하지 않는다. 가능하면 장면별 방향 일관성을 함께 확인하고,
불분명하면 ERGAS와 비용까지 내려간다.

### 3.3 품질 승자와 효율 Pareto를 분리한다

하나의 최종 순위만 만들면 c6 같은 품질형 모델과 R1 같은 효율형 모델을 잘못 섞게 된다.
따라서 결과는 두 종류로 남긴다.

| 분류 | 기준 |
|---|---|
| **Quality winner** | HQNR band → SCC → ERGAS 순으로 가장 좋은 모델 |
| **Efficiency Pareto** | HQNR band 안이며, 허용 가능한 품질 손실로 latency·params를 뚜렷하게 줄인 비지배 모델 |

축소 다음 단계를 여는 운영 gate는 다음과 같다.

~~~text
HQNR drop       ≤ 0.011
ERGAS 악화      ≤ 3%일 때만 공격적 축소를 계속 검토
그리고
latency 또는 params가 ≥ 15% 감소
~~~

3%는 품질 동급의 정의가 아니라 추가 축소를 시도할 수 있는 비용-품질 예산이다.
최종 승자 판정에서는 여전히 SCC와 ERGAS를 순서대로 사용한다.

---

## 4. s1 — attention의 존재·종류·깊이

s1은 c6 CNN을 고정하고 bottleneck attention만 바꾼다.
여러 종류의 새로운 attention을 나열하지 않고 Swin 하나와 필수 CM3A 대조군만 실행한다.

### 4.1 확정 case

| 순서 | ID | 입력 | Bottleneck | 예상 Params | 확인 질문 |
|---:|---|---:|---|---:|---|
| 1 | **S1_SW2_ADD_11CH** | 11ch | Res×4 + Swin×2 | **4.0387M** | Swin pair 자체가 유효한가 |
| 2 | **S1_CM3A_BTL_NOPAN_D124** | 11ch | Res×4 + CM3A×1 | 약 **4.392M** | 효과가 Swin 고유인지, btl attention 일반 효과인지 |
| 3 | **S1_SW4_ADD_11CH** | 11ch | Res×4 + Swin×4 | **4.3054M** | Swin depth 증가가 유효한가 |
| 4 | **S1_SW2_ADD_9CH** | 9ch | Res×4 + Swin×2 | 약 **4.0364M** | 9ch에서도 Swin 효과가 유지되는가 |

S1_SW2_ADD_9CH는 기존 N3_9_d124_noattn과 같은 s1 환경에서 직접 비교한다.
9ch는 파라미터 절약 목적이 아니라 입력 구성과 attention의 상호작용을 확인하는 case다.

### 4.2 CM3A 대조군 정의

주 대조군은 c3b의 유리했던 no-PAN-K/V 조건을 d124 위로 옮긴다.

~~~yaml
width: 128
depths: [1, 2, 4]
attn_locations: [btl]
cm3a_pan_branch: false
swin_depth: 0
~~~

이 선택의 이유:

- 기존 c3b는 d224 기반이어서 c6와 직접 비교되지 않았다.
- PAN K/V는 앞선 실험에서 비용 대비 기여가 낮았다.
- Swin과 비교할 때 backbone과 attention 위치를 같게 두고, attention mechanism 차이를 읽을 수 있다.

원본 PAN-Crafter에 충실한 PAN K/V-on CM3A는 이번 24시간의 필수 case가 아니다.
논문 서사상 반드시 필요할 때만 잔여 시간 예비 case로 둔다.

### 4.3 조건부 case

| ID | Bottleneck | 예상 Params | 실행 조건 |
|---|---|---:|---|
| **S1_SW6_ADD_11CH** | Res×4 + Swin×6 | 4.5722M | SW4가 SW2보다 종합 품질에서 실제 우세할 때만 |

SW6 gate:

~~~text
실행:
  - SW4가 SW2보다 HQNR > 0.011 개선
  또는
  - HQNR 동급이고 SCC·ERGAS가 같은 방향으로 개선

중단:
  - HQNR 동급이며 SCC·ERGAS 개선 없음
  - ERGAS 악화와 latency 증가가 동시에 발생
  - SW4가 SW2에 지배됨
~~~

SW8·SW10은 삭제한다. SW4에서 정체되면 그 자체를 depth 포화 결론으로 사용한다.

### 4.4 s1 예상 시간

| Case | 예상 학습 |
|---|---:|
| SW2 11ch | 2.7~3.0h |
| CM3A-btl noPAN | 2.6~3.1h |
| SW4 11ch | 3.0~3.4h |
| SW2 9ch | 2.7~3.0h |
| 조건부 SW6 | 3.3~3.8h |

확정 case 약 11~12.5h, SW6까지 약 14.5~16.3h다.
구현·smoke·평가 시간을 포함해 24시간 안에 충분히 들어간다.

---

## 5. s2 — Swin을 유지한 pipeline 축소

s2는 이미 완료된 c6-s2를 no-attention anchor로 사용한다.
먼저 SW2를 실행한 뒤 Swin pair를 고정하고 CNN 연산 배치를 바꾼다.

### 5.1 확정 case

| 순서 | ID | CNN 구조 | Swin | 예상 Params | 확인 질문 |
|---:|---|---|---:|---:|---|
| 1 | **S2_SW2_ADD** | d124 | 2 blocks | 4.0387M | s2에서 Swin 효과 |
| 2 | **S2_SW2_D024** | d024 | 2 blocks | 약 **3.2657M** | full-resolution body 제거 |
| 3 | **S2_SW2_BTL2** | d122 | 2 blocks | 약 **3.44~3.45M** | 고해상도 경로를 남기고 btl CNN 축소 |
| 4 | **S2_LR_SW2_W64** | LR-only 신규 pipeline | 2 blocks | 약 **0.20~0.30M** | LR-only 패러다임 최종 반증 |

### 5.2 D024와 BTL2 비교의 해석

~~~text
S2_SW2_D024
  H encoder ResBlock : 1 → 0
  H decoder ResBlock : 1 → 0
  Bottleneck         : ResBlock ×4 + Swin ×2

S2_SW2_BTL2
  H/H2 경로          : c6 유지
  Bottleneck         : ResBlock 4 → 2 + Swin ×2
~~~

새 R1 실측을 반영하면 두 case는 완전히 같은 파라미터가 아니다.
D024 약 3.27M, BTL2 약 3.45M으로 예상되므로 이를 exact matched-budget이라고 부르지 않는다.
대신 비슷한 3M대 예산에서 용량을 고해상도 경로와 bottleneck 중 어디에 둘지 비교하는
near-budget allocation 실험으로 기록한다.

선택 규칙:

1. HQNR band로 먼저 판정한다.
2. 동급이면 SCC, 이어서 ERGAS로 quality parent를 정한다.
3. 다른 한쪽이 훨씬 빠르면 별도 efficiency Pareto로 남길 수 있지만, width 축소의 parent는 quality parent 하나만 사용한다.

### 5.3 조건부 width 축소

| 순서 | ID | Width | 실행 조건 |
|---:|---|---:|---|
| 5 | **S2_WINNER_W112** | 112 | D024/BTL2 quality parent 확정 후 |
| 6 | **S2_WINNER_W96** | 96 | W112가 축소 gate를 통과한 경우만 |

W112는 한 번 실행해 중간 폭 Pareto를 측정한다. W96은 다음 조건을 모두 만족할 때만 실행한다.

~~~text
W112 HQNR drop ≤ 0.011
W112 ERGAS 악화 ≤ 3%
W112 latency 또는 params 감소 ≥ 15%
W112가 parent에 품질·비용 모두 지배되지 않음
~~~

W96에서 같은 조건을 다시 적용한다. 기존 R4/R6 결과처럼 ERGAS와 SCC가 함께 악화되면
더 작은 width로 진행하지 않는다.

### 5.4 LR-TinySwin 반증 case

~~~text
PAN 256×256×1
  → PixelUnshuffle ×4
  → 64×64×16

LRMS 64×64×8
  → concatenate
  → Conv 24→64
  → W-MSA
  → SW-MSA
  → residual Conv
  → Conv 64→128
  → PixelShuffle ×4
  → HRMS residual
  + bicubic-upsampled MS
~~~

판정:

| 결과 | 결정 |
|---|---|
| HQNR < 0.940 | LR-only 패러다임 종료 |
| HQNR ≥ 0.940이나 기존 후보에 지배됨 | 결과만 기록하고 종료 |
| HQNR ≥ 0.940이고 2~3ms ultra-lite Pareto 성립 | 별도 후보로 보존하되 이번 큐에서 SW4는 실행하지 않음 |

LR_SW4는 어떤 결과에서도 이번 24시간 큐에 자동 추가하지 않는다.

### 5.5 s2 예상 시간

| Case | 예상 학습 |
|---|---:|
| SW2 anchor | 2.7~3.0h |
| SW2-D024 | 1.6~2.0h |
| SW2-BTL2 | 2.3~2.8h |
| LR-SW2 | 0.8~1.2h |
| winner-W112 | 2.0~2.6h |
| 조건부 winner-W96 | 1.8~2.3h |

확정 case 약 7.4~9.0h, width 조건부까지 약 11.2~13.9h다.
남는 시간은 새 계열 추가가 아니라 실패 복구·평가·측정에 우선 사용한다.

---

## 6. 실제 실행 큐

### 6.1 s1

~~~text
1. S1_SW2_ADD_11CH
2. S1_CM3A_BTL_NOPAN_D124
3. S1_SW4_ADD_11CH
4. S1_SW2_ADD_9CH
5. [gate] S1_SW6_ADD_11CH
~~~

### 6.2 s2

~~~text
1. S2_SW2_ADD
2. S2_SW2_D024
3. S2_SW2_BTL2
4. S2_LR_SW2_W64
5. [gate] S2_WINNER_W112
6. [gate] S2_WINNER_W96
~~~

s1과 s2의 SW2 중복은 seed 검증이 아니다. 각 서버의 후속 case를 같은 서버 SW2와 비교하기 위한
local anchor다. seed는 추가로 확인하지 않는다.

---

## 7. 구현 및 smoke 요구사항

### 7.1 구현 범위

~~~text
PANCrafterPaper:
  swin_depth
  swin_window_size
  swin_num_heads
  swin_mlp_ratio

SwinBlock:
  window partition / reverse
  cyclic shift
  shifted-window mask
  relative-position bias
  pre-norm residual

LR-TinySwin:
  별도 model/lr_tiny_swin.py
~~~

기존 config에서 swin_depth가 0이거나 미지정이면 c0·c6 파라미터와 checkpoint 동작이 변하지 않아야 한다.

### 7.2 학습 전 smoke

모든 case에서 학습 전에 다음을 확인한다.

~~~text
model build
forward / backward
NaN / Inf
RR·FR output shape
window partition → reverse 왕복
shifted-window mask
PAN·MS 양쪽 non-zero output gradient
512×512 FR 입력
params / FLOPs 실측
peak memory
iteration time
~~~

LR-TinySwin은 추가로 다음을 검사한다.

~~~text
PixelShuffle(PixelUnshuffle(x, 4), 4) == x
단일 PAN impulse의 출력 좌표
64² / 128² LR grid의 window divisibility
~~~

smoke 실패 case는 학습하지 않고 FAILED로 기록한 뒤 다음 case로 넘어간다.

---

## 8. 결과 기록 형식

각 case는 같은 서버 anchor 대비 다음을 함께 기록한다.

~~~text
HQNR / ΔHQNR
D_lambda / D_s
ERGAS / ΔERGAS(%)
SCC
SAM / Q2n / PSNR / SSIM
Params / FLOPs
Inference latency / speedup
Peak memory
Training time
checkpoint selection metric
baseline 또는 fixed
py 또는 matlab 평가 주체
~~~

최종 표는 최소한 다음 네 비교를 분리한다.

1. c6 vs SW2: Swin 존재 효과
2. SW2 vs CM3A-btl: 표준 Swin과 기존 attention 비교
3. SW2 vs SW4/SW6: Transformer depth 효과
4. SW2 vs D024/BTL2/width 축소: 연산 배치와 압축 효과

---

## 9. 캠페인 종료 시 내려야 할 결론

| 질문 | 필요한 비교 | 가능한 결론 |
|---|---|---|
| Swin을 붙일 가치가 있는가 | c6 vs SW2 | 품질 개선 / 비용 대비 무효 |
| 개선이 Swin 고유 효과인가 | SW2 vs CM3A-btl | 표준 attention 우세 / attention 일반 효과 / CM3A 우세 |
| Swin depth가 필요한가 | SW2 vs SW4, 조건부 SW6 | pair 하나 충분 / 깊이 증가 유효 |
| 9ch에서도 유지되는가 | N3 vs SW2-9ch | 입력 단순화 가능 / 11ch 유지 |
| CNN 용량은 어디에 남겨야 하는가 | D024 vs BTL2 | 고해상도 경로 / bottleneck 우선 |
| 더 줄일 수 있는가 | W112, 조건부 W96 | 중간 폭 Pareto / 폭 축소 종료 |
| LR-only를 살릴 수 있는가 | LR-Fuse vs LR-SW2 | 최종 기각 / ultra-lite 예외 |

캠페인 종료 후에는 단일 최고 모델만 선언하지 않고 다음 두 지점을 별도로 남긴다.

~~~text
1. Quality winner
2. Efficiency Pareto winner
~~~

현재 가장 보수적인 예상은 SW2 또는 CM3A-btl이 품질형 후보가 되고,
D024 계열이 효율형 후보가 되는 것이다. 그러나 이 예상은 실행 우선순위를 정하기 위한 가설일 뿐,
결과 판정에는 사용하지 않는다.

