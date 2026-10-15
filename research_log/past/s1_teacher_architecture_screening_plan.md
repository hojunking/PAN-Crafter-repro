# s1 서버 계획 — Teacher Architecture Screening

## 1. 목적

s1의 RTX 4090을 사용해 **상호학습 이전에 고정할 Teacher architecture**를 선정한다.

이번 단계에서 최적화할 우선순위는 다음과 같다.

1. **HQNR 상승**
2. **SCC 유지 또는 상승**
3. ERGAS·SAM 비열화
4. 성능이 유사하면 더 작은 파라미터 수와 낮은 연산량 우선

> 이 단계에서는 architecture와 입력 구성만 비교한다. Mutual-learning loss, uncertainty head, feature distillation은 포함하지 않는다.

---

## 2. 고정 조건

모든 후보는 아래 설정을 동일하게 사용한다.

```text
Dataset              : PanCollection WV3
Crop                  : False (nocrop)
Geometric augmentation: H/V flip + 90° rotation 유지
Normalization         : LayerNorm  (논문 Eq (5). 50K 실측 ERGAS 2.1494 -> 2.1205, p=0.040)
Dropout               : 0.0
Loss                  : 기존 MARs loss 유지
CM3A                  : 수정·검증된 동일 구현
LPAN                   : repaired LPAN 사용
Optimizer / scheduler : 동일 설정
Total schedule        : 50K iterations
Screening checkpoint  : 25K
Seed                  : 2025
Evaluation protocol   : 동일 RR / FR evaluator
```

### 25K 운용 원칙

25K는 별도의 짧은 학습이 아니라 **50K trajectory의 중간 checkpoint**로 사용한다.

```text
max_iterations        = 50K
scheduler_total_steps = 50K
save/evaluate         = 25K
selected runs         = optimizer·scheduler state를 유지한 채 50K까지 resume
```

25K에서 cosine schedule이 끝나는 별도 run은 사용하지 않는다.

---

## 3. 비교 후보

| ID | Architecture | 입력 | Params 예상 | 비교 목적 |
|---|---|---:|---:|---|
| **A0** | T7: width 128, depth `[2,2,4]`, AttnBlock 3 | 9ch | 약 7.17M | s1 local baseline |
| **A1** | T7: width 128, depth `[2,2,4]`, AttnBlock 3 | 11ch | 약 7.17M | 11ch 입력 효과 분리 |
| **A2** | A1 + depth `[4,4,4]` | 11ch | 9.5425M | capacity 증가 효과 |

> A2 를 배포 구조(T10, 9.97M)로 두면 4-scale·GroupNorm·mode-token 까지 함께 달라져
> **capacity 단독 효과가 되지 않는다.** 같은 재구성 구조 안에서 depth 만 키워
> 9.5425M 로 맞춘다 (배포본 9.97M 에 근접).

핵심 비교는 다음과 같다.

```text
A0 ↔ A1 : 동일 architecture에서 9ch 대 11ch
A1 ↔ A2 : 동일 11ch에서 7.17M 대 9.97M
```

6.69M 모델은 이번 Teacher 선정 queue에서 제외하고, 이후 capable peer 또는 student 후보 탐색 단계에 사용한다.

---

## 4. 권장 15시간 45분 계획

실제 시작 시각을 `T0`로 둔다. 아래 시간은 현재 실험표의 기존 속도를 바탕으로 한 운영 예상치다.

| 구간 | 작업 | 예상 소요 | 누적 |
|---|---|---:|---:|
| T0–T0+2:45 | A0를 25K까지 학습 | 2시간 45분 | 2:45 |
| +0:10 | A0 RR·FR 평가 및 checkpoint 확인 | 10분 | 2:55 |
| +2:45 | A1을 25K까지 학습 | 2시간 45분 | 5:40 |
| +0:10 | A1 평가 | 10분 | 5:50 |
| +3:05 | A2를 25K까지 학습 | 3시간 5분 | 8:55 |
| +0:20 | 25K 자동 ranking | 20분 | 9:15 |
| +2:55 | 1위 후보 25K→50K resume | 2시간 55분 | 12:10 |
| +2:55 | 2위 후보 25K→50K resume | 2시간 55분 | 15:05 |
| +0:40 | 최종 RR·FR 평가 및 sheet 기록 | 40분 | **15:45** |

### Resume checkpoint에 반드시 포함할 상태

```text
model weights
optimizer state
scheduler state
AMP GradScaler
current iteration
RNG state
가능하면 sampler/data-loader state
```

---

## 5. 25K 자동 선별 기준

A0의 25K 결과를 s1 local baseline으로 사용한다.

### 5.1 Hard guard

후보가 아래 조건을 하나라도 크게 위반하면 50K resume 대상에서 제외한다.

```text
ERGAS ≤ A0_ERGAS × 1.004     # 재구성본 시드 2σ = 0.11% 기준 약 4σ
SAM   ≤ A0_SAM   × 1.004
# SCC 는 guard 에서 뺀다 — 전 구성이 0.9887~0.9902 로 포화라 판별력이 없다
```

### 5.2 통과 후보 ranking

1. HQNR이 높은 후보 우선
2. HQNR 차이가 `0.002` 미만이면 ERGAS가 낮은 후보 우선
3. ERGAS도 유사하면 SCC가 높은 후보 우선
4. 모든 차이가 작으면 파라미터가 적은 후보 우선

이 기준은 통계적 유의성 판정이 아니라 **overnight successive-halving 운영 기준**이다.

---

## 6. 50K Teacher 확정 기준

50K에서는 A0의 50K 결과를 local baseline으로 사용한다. 다른 서버의 절대 수치와 직접 비교하지 않는다.

최종 Teacher 후보는 다음을 만족해야 한다.

```text
1. HQNR이 A0보다 상승하거나 최소 동등
2. SCC 하락 ≤ 0.0003
3. ERGAS 악화 ≤ 0.5%
4. SAM 악화 ≤ 0.5%
5. HQNR(12-19) 기록
6. T10 선택 시, 증가한 모델 크기를 정당화할 명확한 이득 필요
```

### T10 선택 기준

T10은 T7보다 모델 크기가 크므로 단순 동률이면 선택하지 않는다. 다음 중 하나가 필요하다.

- HQNR이 반복적으로 명확히 상승
- SCC가 상승하면서 ERGAS·SAM도 비열화
- full-resolution의 `D_s` 또는 `D_λ`가 구조적으로 개선

### 11ch 결과 해석

A1이 HQNR만 높이고 RR 지표를 악화시키면 다음처럼 분리한다.

```text
General Teacher : A0 또는 RR 균형 후보
FR-oriented Teacher : A1
```

11ch를 모든 후속 실험의 기본값으로 즉시 고정하지 않는다.

---

## 7. 평가 항목

### Reduced-resolution

```text
ERGAS ↓
SAM   ↓
SCC   ↑
PSNR  ↑
SSIM  ↑
Q8    ↑
```

### Full-resolution

```text
HQNR ↑ : index 12-19 (8장). 논문과 대조 가능한 유일한 프로토콜
D_s  ↓
D_λ  ↓

전체 20장 기준은 쓰지 않는다. 어떤 논문도 그렇게 보고하지 않고, 0-11 이 12-19 보다
크게 어려운 장면이라(D_lambda 2.4배) idx 1 한 장이 평균을 -0.02 끌어내려 방향이 뒤집힌다.
```

### 기록 원칙

- s1 결과는 s1 local baseline과만 비교한다.
- 25K와 50K를 같은 단계의 수치처럼 직접 비교하지 않는다.
- 최종 구조 확정 전 seed 반복이 필요하다.
- test scene별 metric도 보존한다.

---

## 8. s1 전용 Sheet 구성

권장 sheet 이름:

```text
WV3-s1     (gspread_upload.py 가 <데이터셋>-<서버> 로 자동 생성한다)
```

권장 열:

| 구분 | 열 |
|---|---|
| 실행 정보 | Run ID, 날짜, 서버, GPU, Git commit, config path |
| 구조 | width, depth, AttnBlock 수, input channels, params, FLOPs |
| 고정 조건 | crop, norm, dropout, MARs λ, seed |
| 학습 | max steps, eval step, runtime, peak VRAM, batch, grad accumulation |
| RR | ERGAS, SAM, SCC, PSNR, SSIM, Q8 |
| FR | HQNR-paper, HQNR-all, D_s, D_λ |
| 판정 | hard guard pass, 25K rank, 50K selected, 비고 |

### Run ID 예시

```text
s1_A0_T7_9ch_nocrop_seed2025
s1_A1_T7_11ch_nocrop_seed2025
s1_A2_T10_11ch_nocrop_seed2025
```

---

## 9. 무인 실행 대안

중간 ranking 자동화가 어렵다면 다음 고정 queue를 사용한다.

```text
A0 50K
→ A1 50K
→ A2 25K checkpoint
```

예상 시간은 약 14시간 이상이다. 정보 효율은 successive halving보다 낮지만, 운영 안정성은 높다.

---

## 10. 실패 및 복구 기준

### 즉시 중단

- NaN/Inf 발생
- loss 폭주
- peak VRAM 지속 초과
- evaluation pipeline 또는 LPAN protocol 불일치
- 예상 처리량 대비 30% 이상 저하가 지속

### 자동 복구

1. 마지막 정상 checkpoint에서 resume
2. AMP 문제면 GradScaler 초기화 후 재개
3. OOM이면 batch를 절반으로 줄이고 gradient accumulation으로 effective batch 유지
4. scheduler step이 50K 기준인지 재확인

---

## 11. 이번 s1 실험의 산출물

1. T7 9ch nocrop의 s1 기준선
2. T7 11ch nocrop의 입력 구성 효과
3. T10 11ch nocrop의 capacity 효과
4. 상위 두 후보의 50K 결과
5. Teacher architecture 고정 또는 추가 검증 필요 여부

### 다음 단계

```text
Teacher 확정
→ 동일 Teacher 두 개의 mutual-learning 검증
→ capable peer / smaller student 탐색
→ Teacher–Student asymmetric mutual learning
→ static KD
```
