# s2 서버 계획 — Teacher–Teacher Mutual Learning Validation

## 1. 목적

s2에서는 architecture를 변경하지 않고, **동일한 강한 PAN-Crafter peer 두 개 사이의 상호학습 효과**만 검증한다.

검증 질문은 다음 하나다.

> 동일 architecture의 두 peer가 output-level mutual learning을 받을 때, 각 단일 peer가 matched independent control보다 좋아지는가?

이번 단계에서 ensemble 향상만으로는 성공으로 판정하지 않는다.

---

## 2. 하드웨어 — 실측 완료 (2026-08-25)

원안의 "RAM 64GB 가 시스템인지 VRAM 인지" 질문은 실측으로 닫혔다.

| | |
|---|---|
| GPU | **NVIDIA CMP 170HX, VRAM 63.5 GiB** (시스템 RAM 은 별개로 94 GB) |
| 두 peer 동시 학습 peak allocated | **30.49 GiB** (reserved 32.92) |
| 처리량 | **1.56 it/s** (단일 peer 2.77 대비 56%) |
| 25K 소요 | **약 4.5시간** (원안 추정 5h50m 보다 빠르다) |

→ **10절의 "GPU VRAM 이 충분한 경우" 분기를 쓴다.** peer 당 nominal batch 48,
gradient accumulation 불필요. **AMP 도 쓰지 않는다** — 여유가 충분하고, 재현 설정에서
벗어나는 변수를 늘릴 이유가 없다.

### 1K smoke test 에서 여전히 확인할 것

```text
peak allocated / reserved VRAM
iteration time, samples per second
anchor loss / mutual loss
peer disagreement
NaN/Inf 여부
```

`gradient ratio r_g` 는 학습 루프에서 재지 않는다 — `retain_graph` 로 두 backward 동안
그래프를 살리면 peak 이 30 → 34 GiB 를 넘어 **실제로 OOM 났다**. r_g 는 스텝마다 필요한
값이 아니라 λ 를 정하기 위한 상수이므로 `tools/dml_calibrate.py` 에서 작은 배치로 따로 잰다.

> smoke config 는 `mutual_warmup` 을 100 으로 낮춰 둔다. 원안대로 2500 이면
> 1K smoke 안에서 λ 가 끝까지 0 이라 mutual 경로가 한 번도 실행되지 않는다.

## 3. 고정 architecture

s1의 최종 Teacher 결과를 기다리지 않고, 현재까지 검증된 동일 구조로 pure DML 효과를 먼저 본다.

```text
Architecture : T7, 약 7.17M
Depth        : [2,2,4]
AttnBlock    : 3
Input        : 9ch
Crop         : False (nocrop)
Norm         : LayerNorm   (2026-08-26 변경. 아래 근거 표 참고)
Dropout      : 0.0
Loss anchor  : 기존 MARs
Peer A seed  : 2025
Peer B seed  : 2026
Total schedule: 50K
Screening     : 25K checkpoint
```

11ch, uncertainty head, SCC loss, SWT feature loss는 이번 최초 검증에 넣지 않는다.

### 이 구조 선택의 근거 (2026-08-25 결과 기준)

나중에 "왜 GroupNorm 인가 / 왜 9ch 인가" 를 다시 파헤치지 않도록 근거를 남긴다.
수치는 `WV3-s1` · `WV3-s2` 시트다.

| 항목 | 선택 | 근거 |
|---|---|---|
| `crop: False` | **유지** | 50K 에서 ERGAS −2.56% (20/20, p<10⁻⁶), HQNR 0.9430→0.9467. **RR·FR 을 동시에 올린 유일한 변경** |
| `dropout: 0.0` | **유지** | 0.2 는 0/20, ERGAS +9.11%. 결정적으로 기각됨 |
| `depth (2,2,4)` / `AttnBlock 3` | 유지 | 논문 params 7.170 M 일치 지점 (실측 7,170,664, +0.009%) |
| **`LayerNorm`** | **GroupNorm → LayerNorm 으로 정정 (2026-08-26)** | 논문 Eq (5) 는 LN 이다. s1 50K 에서 **ERGAS −1.34%**(시드 잡음 0.8% 초과 = 실재), **HQNR −0.24%**(FR 8장, D_λ 표준편차가 평균의 46% = 잡음 안). **잡음 수준의 FR 손실을 근거로 잡음을 넘는 RR 개선을 포기할 수 없다.** 앞선 개정에서 "HQNR 중심이니 GN 이 유리" 라고 적었으나, 그것은 이미 정해진 선택을 사후 정당화한 것이었다 |
| **입력 `9ch`** | **유지 — 단 한계로 기록한다** | `p25_in11` 은 25K 인데 D_λ 0.0260 으로 s2 전체 최저이고 HQNR 0.9441 로 50K `paper_wv3`(0.9430)를 넘는다. **11ch 가 FR 축의 가장 강한 손잡이다.** 그럼에도 9ch 로 가는 것은 pure DML 효과를 분리하기 위해서이며, **HQNR 헤드룸이 낮은 지점에서 재게 된다**는 점을 결과 해석 시 명시해야 한다 |

> **GN 이 들어온 경위 (기록).** ① 계획서 §3 이 `Norm: GroupNorm` 으로 고정했고 ② config 를
> `paper_nocrop.yaml` 에서 파생시켰는데 그 파일의 `norm: gn` 은 "이미 돌린 실행은 norm 옵션
> 도입 전 코드였다" 는 **사후 기록용 주석**이지 앞으로의 선택이 아니었다. ③ 검토 때 이를
> 열린 질문으로 다루지 않고 HQNR 논거로 유지했다. ③이 잘못이다.

> **부수 효과.** `LN + nocrop` 은 아무도 학습한 적이 없는 조합이다. M0 는 λ=0 인 두 peer 이므로
> **M0 자체가 이 조합을 시드 2벌 돌리는 것**과 같다. DML 판정과 별개로 그 기준선이 확보된다.

**지표 축에 대한 확인** — 13벌 순위상관에서 ERGAS·SAM·SCC·Q2n 은 서로 ρ ≥ 0.92 로 한 덩어리이고,
**HQNR 만 ρ 0.33~0.40 으로 독립**이다. 즉 실질 축은 둘뿐이다.
**SCC 는 ERGAS 와 ρ=0.96 이고 전체 폭이 0.52% 라 독립 정보가 거의 없다** — 게다가
우리 최고 SCC 0.9908 은 이미 논문 0.988 을 넘었다. 집중 지표는 **HQNR + ERGAS(또는 SAM)** 로 두고,
SCC 는 보조로만 본다.

---

## 4. 비교 실험

### M0 — Two-peer independent control

```text
동일 two-peer trainer 사용
Peer A와 B 모두 학습
mutual weight λ_m = 0
```

M0는 “두 모델을 같은 trainer에서 동시에 돌린 것” 자체의 영향을 통제한다.

### M1 — Symmetric mutual learning

```text
M0와 동일한 초기 weights
M0와 동일한 data order
M0와 동일한 augmentation
M0와 동일한 optimizer / scheduler
차이는 mutual loss의 활성화뿐
```

### 공정한 paired initialization

M0와 M1을 시작하기 전에 다음을 저장한다.

```text
initial Peer A weights
initial Peer B weights
optimizer 초기 상태
scheduler 초기 상태
RNG state
sampler/data-loader state
```

---

## 5. 최초 Mutual Loss

각 peer의 HRMS residual을 다음처럼 둔다.

\[
r_i = \hat I_i^{HRMS} - I_{MS}^{up}
\]

Peer A와 B의 loss는 다음과 같다.

\[
L_A = L_{MARs}^{A}
+ \lambda_m(t)\left\|r_A-\operatorname{sg}(r_B)\right\|_1
\]

\[
L_B = L_{MARs}^{B}
+ \lambda_m(t)\left\|r_B-\operatorname{sg}(r_A)\right\|_1
\]

여기서 `sg`는 stop-gradient다.

### 적용 범위

- Mutual loss는 우선 **MS mode의 HRMS residual**에만 적용
- PAN mode는 기존 MARs anchor로만 학습
- 두 peer는 같은 sample과 같은 geometric augmentation을 사용
- peer별 parameter initialization은 다르게 유지

두 peer에 서로 다른 rotation/flip을 적용하면 출력 좌표가 달라지므로, 직접적인 output mutual loss를 계산하면 안 된다.

---

## 6. λ 스케줄

초기 권장값:

```text
λ_max = 0.05
```

스케줄:

\[
\lambda_m(t)=
\begin{cases}
0, & 0 \le t < 2.5K \\
\text{linear ramp}, & 2.5K \le t < 5K \\
\lambda_{max}, & 5K \le t \le 25K
\end{cases}
\]

### Gradient-ratio calibration

1K smoke test에서 다음을 계산한다.

\[
r_g = \frac{\|\nabla L_{mut}\|}{\|\nabla L_{MARs}\|}
\]

실제 mutual 기여인 `λ_m × r_g`가 대략 `0.1–0.2`가 되도록 조정한다.

```text
너무 강함 : λ_max = 0.02
적정     : λ_max = 0.05
너무 약함 : λ_max = 0.10 이하로 상향
```

최초 overnight run에서는 0.1을 상한으로 둔다.

---

## 7. 50K 완주로 변경 (2026-08-26 개정)

원안은 50K scheduler 위에서 **25K screening** 이었다. 이를 **50K 완주**로 바꾼다.

```text
max_iterations        = 50K
scheduler_total_steps = 50K      (= num_iter. 분리가 불필요해졌다)
save/evaluate         = eval_epoch 10 (248 epoch 기준 25회)
```

### 왜 바꾸는가

**25K 에서는 판정이 어렵다.** 이 저장소의 25K 값은 일관되게 **3.6% 비관적**이고
([8/24 요약 §1.2](../results_log/2026-08-24_wv3-summary-and-protocol.md)), 그 편향이
소형·미수렴 구성에 유리하게 걸린다. DML 효과는 그보다 훨씬 작을 것으로 예상되므로,
25K 에서 잡히는 차이가 DML 때문인지 수렴도 차이인지 분리하기 어렵다.

또 25K 는 이 저장소의 다른 50K 결과들과 가로로 비교할 수 없어, DML 결과를
`paper_nocrop`(2.0875) 같은 기준선 옆에 놓으려면 어차피 50K 가 필요하다.

### 비용

| | |
|---|---|
| 50K 1벌 (두 peer 동시) | **약 9.3시간** (학습 8.9h + 평가 25회 26분) |
| M0 + M1 순차 | **약 18.6시간** |

원안 8절의 16시간 계획은 25K 기준이므로 성립하지 않는다. **M0 완주 후 조기 게이트로
M1 진행 여부를 정하는 것이 더 중요해졌다** — 게이트에 걸리는 비용이 4.5h 에서 9.3h 로 늘었다.

## 8. 권장 16시간 계획

실제 시작 시각을 `T0`로 둔다.

| 구간 | 작업 | 예상 소요 | 누적 |
|---|---|---:|---:|
| T0–T0+0:30 | 1K smoke test, VRAM·throughput·gradient ratio 확인 | 30분 | 0:30 |
| +5:50 | M0: λ=0, 25K checkpoint | 5시간 50분 | 6:20 |
| +0:10 | M0 checkpoint·sanity evaluation | 10분 | 6:30 |
| +5:50 | M1: calibrated λ, 25K checkpoint | 5시간 50분 | 12:20 |
| +0:40 | A/B/ensemble RR·FR 전체 평가 | 40분 | 13:00 |
| +3:00 | 결과에 따른 adaptive continuation | 3시간 | **16:00** |

실제 처리량은 1K smoke test에서 다시 계산한다.

---

## 9. 25K 이후 분기

### Case A — M1이 긍정적

조건:

- HQNR·SCC가 control보다 상승
- ERGAS·SAM 비열화
- peer A/B 중 최소 하나가 matched control보다 개선
- 다른 peer가 명확히 악화되지 않음

조치:

```text
M1을 25K → 40K까지 resume
mutual 효과의 지속성 확인
이후 별도 run에서 50K 완성
```

### Case B — Peer collapse

징후:

- residual disagreement가 5–10K 사이에 거의 0
- 두 출력은 같아지지만 metric은 개선되지 않음
- HQNR 또는 SCC 하락
- ERGAS·SAM 동반 악화

조치:

```text
λ_max ← λ_max / 2
동일 초기 상태에서 10K 재실험
```

### Case C — 효과가 거의 없고 diversity는 남아 있음

징후:

- M0와 M1 metric 차이가 거의 없음
- peer disagreement가 충분히 유지
- mutual gradient가 anchor에 비해 매우 작음

조치:

```text
λ_max ← 2 × λ_max
최대 0.10
동일 초기 상태에서 10K 재검증
```

### Case D — Ensemble만 향상

A/B 단일 성능은 그대로이고 ensemble만 향상하면, 이는 일반적인 ensemble 효과일 수 있다.

```text
DML 성공으로 판정하지 않음
single-peer improvement가 필요
```

---

## 10. 메모리 운영

PAN-Crafter의 MARs는 MS/PAN mode를 함께 처리하므로 두 peer 학습은 activation 부담이 크다.

### GPU VRAM이 충분한 경우

```text
peer당 nominal batch = 48
AMP                  = on
상대 출력             = detach
각 peer loss          = 별도 backward
learning rate         = 기존 값 유지
```

### GPU VRAM이 부족한 경우

우선 적용:

```text
peer당 batch          = 24
gradient accumulation = 2
peer당 effective batch= 48
learning rate         = 기존 값 유지
```

그래도 부족하면:

1. 상대 peer target을 `no_grad`로 먼저 계산
2. Peer A forward/backward
3. Peer B forward/backward
4. optimizer step을 동기화

이 경우 메모리는 줄지만 wall-clock time은 증가할 수 있다.

---

## 11. 성공 기준 — 대응표본 검정으로 판정한다 (2026-08-25 개정)

### 원안의 문제

원안은 절대 임계였다.

```text
ΔHQNR ≥ +0.002     ΔSCC ≥ +0.0002
ERGAS 악화 ≤ 0.5%   SAM 악화 ≤ 0.5%
```

**이 값들은 잡음 아래다.**

| 기준 | 상대 크기 | 대비 |
|---|---:|---|
| ΔHQNR ≥ +0.002 | 0.21% | 시드 잡음 0.8% 의 **1/4** |
| ΔSCC ≥ +0.0002 | 0.02% | 13벌 SCC **전체 폭 0.52% 의 1/26** |

운영 gate 라 해도 이 크기면 잡음에 그냥 통과한다.

### 개정 기준

이 저장소가 이미 쓰는 **대응표본 t-검정**으로 바꾼다. 같은 20장(FR 8장)이라 장면 간 변동이
상쇄되어 대응 SE 가 0.006~0.043 (비대응 0.129) 으로 검정력이 실제로 있다.
`p25_nocrop` 이 20/20 · p<10⁻⁵ 로 잡힌 것이 그 예다.

```text
필수 (M1 을 M0 와 대응표본으로 비교)
  HQNR      : p < 0.05 이고 개선 방향
  ERGAS·SAM : 유의한 악화가 없을 것 (p < 0.05 인 악화가 없어야 한다)
  방향 일관성: peer A 와 peer B **양쪽 모두**에서 같은 방향
보조
  scene 별 승패 (20/20 같은 형태)
  ensemble 은 별도 보고 — 단일 peer 개선이 없으면 DML 성공으로 보지 않는다 (원안 Case D 유지)
```

**0.8% 미만 차이는 시드 3벌 이상에서 방향이 일관될 때만 주장한다** (`CLAUDE.md` 판정 규칙).
이번 M0/M1 은 각 1벌이므로, 유망하면 시드를 늘려 확인한 뒤에만 결론에 넣는다.

### M0 자체가 조기 게이트다 (신설)

M0(λ=0)는 대조군이면서 동시에 **DML 헤드룸 측정**이다. 8/20 go/no-go 와 같은 정의로
peer 오차 상관과 오라클 이득을 낸다.

```bash
python tools/dml_analyze.py --m0 work_dir/dml_m0        # M0 만으로 다양성 진단
```

| M0 진단 | 해석 | 조치 |
|---|---|---|
| 상관 ≥ 0.97, 오라클 < +5% | 두 peer 가 사실상 같은 오차를 낸다. **DML 로 얻을 것이 거의 없다** | M1 에 4.5h 를 쓰기 전에 λ 상향 또는 다양성 설계(다른 증강·다른 데이터 순서)를 먼저 검토 |
| 상관 0.85~0.97 | 원안대로 진행 | M1 실행 |
| 상관 ≤ 0.85, 오라클 ≥ +15% | 8/20 기준 통과 — 상보성 있음 | M1 실행, 기대치 상향 |

**왜 이 게이트를 넣는가.** 8/20 은 Teacher–Student 쌍에서 **용량비가 1 에 가까울수록 오차 상관이
높아지는 것**을 관측했다 (1.2배 비율에서 ρ=0.9733 으로 최고, 오라클 +5.4%).
동일 구조 두 peer 는 비율 1.0 이므로 외삽하면 ρ ≥ 0.97 이다.
다만 8/20 이 잰 것은 **구조 다양성**이고 **시드 다양성은 측정된 적이 없다** — 그래서 열린 질문이
맞지만, M0 가 그 답을 4.5시간 안에 주므로 M1 전에 확인하는 것이 싸다.

## 12. 필수 진단 로그

### 학습 중

```text
L_MARs_A
L_MARs_B
L_mut_A
L_mut_B
λ_m
anchor grad norm
mutual grad norm
peer residual L1 disagreement
peer output correlation
learning rate
peak VRAM
iteration time
```

### 25K 평가

```text
Peer A single
Peer B single
A+B ensemble
M0 average
M1 average
M1 - M0 delta
```

### Metric

```text
RR : ERGAS, SAM, SCC, PSNR, SSIM, Q8
FR : HQNR-paper, HQNR-all, D_s, D_λ
```

---

## 13. s2 전용 Sheet 구성

권장 sheet 이름:

```text
s2_mutual_learning
```

권장 열:

| 구분 | 열 |
|---|---|
| 실행 정보 | Run ID, 날짜, 서버, GPU, Git commit, config path |
| Pair | architecture, Peer A seed, Peer B seed, initial checkpoint ID |
| Mutual 설정 | loss type, target type, λ_max, warmup, ramp, stop-gradient |
| 메모리 | batch/peer, grad accumulation, peak VRAM, runtime |
| Peer A RR/FR | ERGAS, SAM, SCC, HQNR-paper, HQNR-all, D_s, D_λ |
| Peer B RR/FR | 동일 항목 |
| Ensemble | 동일 항목 |
| 진단 | disagreement@5K/10K/25K, mutual/anchor grad ratio |
| 판정 | collapse, positive, neutral, negative, continuation action |

### Run ID 예시

```text
s2_M0_T7xT7_independent_seed2025_2026
s2_M1_T7xT7_mutual_lam005_seed2025_2026
s2_M1b_T7xT7_mutual_lam0025_seed2025_2026
```

---

## 14. 이번 단계에서 제외할 항목

Pure mutual-learning 효과를 분리하기 위해 다음은 넣지 않는다.

```text
uncertainty head
NLL predictive distribution
encoder feature matching
decoder feature matching
SWT mutual loss
SCC 직접 최적화 loss
hysteresis controller
different-size teacher–student
static KD
```

M1이 유효한 경우에만 다음 순서로 확장한다.

```text
output-only mutual
→ uncertainty-gated mutual
→ SWT/frequency relation
→ dynamic λ
```

---

## 15. 이번 s2 실험의 산출물

1. Two-peer independent matched control
2. Output-residual symmetric DML의 25K 결과
3. Peer A, Peer B, ensemble 성능 분리
4. Mutual loss와 peer disagreement 곡선
5. λ가 강한지·약한지에 대한 근거
6. 40K/50K continuation 여부

### 다음 단계 결정

| 결과 | 후속 방향 |
|---|---|
| Mutual 효과 명확 | 선정 Teacher로 50K Teacher–Teacher DML |
| 효과 없음 | Teacher–Student asymmetric learning 또는 static KD로 이동 |
| collapse | λ 감소, late mutual 또는 diversity 유지 설계 |
| 한쪽만 개선 | asymmetric λ 또는 one-way online KD 검토 |
