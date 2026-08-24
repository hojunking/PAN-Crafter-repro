# PAN-Crafter 기반 Teacher–Student 재설계 전략

## 0. 문서 목적

현재 목표는 원본 PAN-Crafter를 그대로 Teacher로 사용하는 것이 아니라 다음 순서로 재설계하는 것이다.

```text
공개 PAN-Crafter 분석
        ↓
중복성이 큰 CM3A 경로 제거
        ↓
효율적인 PAN-Crafter Core 확보
        ↓
성능을 실제로 높이는 모듈만 추가
        ↓
새로운 고성능 Teacher 설계
        ↓
Teacher에서 경량 Student 도출
        ↓
Teacher–Student Mutual Learning
```

핵심 판단은 다음과 같다.

1. 공개 PAN-Crafter는 논문 보고 수치와 구조·계산량·성능이 일치하지 않는다.
2. 공개 모델의 여러 CM3A와 PAN K/V branch에는 상당한 중복성이 있다.
3. 현재 6.694M 구조는 원 모델 대비 성능이 통계적으로 구분되지 않으면서 약 2배 빠르다.
4. 따라서 6.694M 구조를 곧바로 최종 Student로 확정하기보다, **새 Teacher를 설계하기 위한 정제된 core backbone**으로 사용하는 것이 타당하다.
5. 새 Teacher가 확정된 뒤 depth와 고비용 모듈을 줄여 Student를 도출한다.

---

# 1. 용어 정리

## 1.1 PAN-Crafter의 전체 구조

PAN-Crafter는 U-Net형 encoder–decoder 구조다.

```text
Full Resolution
    ResBlocks
        ↓
H/2 Encoder
    ResBlocks + CM3A
        ↓
H/4 Encoder
    ResBlocks + CM3A
        ↓
H/8 Bottleneck
    ResBlocks + CM3A
        ↑
H/4 Decoder
    ResBlocks + CM3A
        ↑
H/2 Decoder
    ResBlocks + CM3A
        ↑
Full Resolution
    ResBlocks
```

CM3A는 모든 ResBlock 내부에 포함되는 것이 아니라, 중·저해상도 stage에 배치된 별도의 AttnBlock 내부에 존재한다.

## 1.2 CM3A의 두 branch

각 CM3A는 하나의 query로 두 종류의 local attention 결과를 만든다.

\[
x_{\mathrm{MS}}
=
\operatorname{LocalAttn}(Q,K_{\mathrm{MS}},V_{\mathrm{MS}})
\]

\[
x_{\mathrm{PAN}}
=
\operatorname{LocalAttn}(Q,K_{\mathrm{PAN}},V_{\mathrm{PAN}})
\]

그리고 mode-dependent coefficient로 두 출력을 결합한다.

\[
x_{\mathrm{attn}}
=
\alpha_1 x_{\mathrm{MS}}
+
\alpha_2 x_{\mathrm{PAN}}
\]

이 문서에서 말하는 **PAN branch 제거**는 다음을 의미한다.

```text
K_pan projection 제거
V_pan projection 제거
PAN output projection 제거
x_pan attention path 제거
```

다음 구성은 제거하지 않는다.

- PAN 입력
- PAN high-frequency residual input
- MARs의 PAN reconstruction mode
- PAN-conditioned query
- ResBlock mode modulation
- shared feature 안에 포함된 PAN 정보

따라서 정확한 명칭은 다음과 같다.

> **CM3A PAN K/V branch removal**

이는 MARs의 PAN mode 제거와 다른 실험이다.

---

# 2. 공개 PAN-Crafter와 재현 기준

## 2.1 공개 코드 기준 Teacher

현재 내부 기준 Teacher는 공개 PAN-Crafter 배포 코드다.

```yaml
width: 128
depth: [2, 2, 2, 2]
CM3A: 5 blocks
CM3A_branch: MS + PAN
MARs: true
ResMod: true
PAN_HF_stem: true
```

측정값:

| 항목 | 공개 코드 Teacher |
|---|---:|
| Parameters | 9.969M |
| FLOPs | 약 158–168G |
| Inference | 약 18.0ms |
| 25K ERGAS | 2.2598 |
| 50K ERGAS | 약 2.1643 |

논문은 7.17M, 79.03G, ERGAS 2.040을 보고하지만, 공개 코드와 동일한 설정에서는 해당 수치가 재현되지 않았다.

또한 재현된 PAN-Crafter는 CANConv와 통계적으로 구분되지 않았다.

| 모델 | ERGAS |
|---|---:|
| 재현 PAN-Crafter | 2.1643 |
| CANConv 배포 모델 | 2.1716 |
| 대응표본 검정 | \(p=0.667\) |

따라서 논문 수치가 아니라 **공개 코드 재현 Teacher**를 내부 기준으로 사용한다.

---

# 3. 기존 구조와 경량화 구조 비교

## 3.1 기존 PAN-Crafter

```text
Input stem
[PAN | ↑MS | ↑LPAN | PAN-HF]
        │
        ▼
Full-res Encoder ResBlocks
        │
        ▼
H/2 Encoder
Dual-branch CM3A
        │
        ▼
H/4 Encoder
Dual-branch CM3A
        │
        ▼
H/8 Bottleneck
Dual-branch CM3A
        │
        ▼
H/4 Decoder
Dual-branch CM3A
        │
        ▼
H/2 Decoder
Dual-branch CM3A
        │
        ▼
Full-res Decoder ResBlocks
        │
        ▼
HRMS residual
```

## 3.2 현재 가장 강한 경량화 구조

현재 6.694M 모델은 다음 구조다.

```text
Input stem
[PAN | ↑MS | ↑LPAN | PAN-HF]
        │
        ▼
Full-res Encoder ResBlocks
        │
        ▼
H/2 Encoder
ResBlocks only
        │
        ▼
H/4 Encoder
MS-K/V-only CM3A
        │
        ▼
H/8 Bottleneck
MS-K/V-only CM3A
        │
        ▼
H/4 Decoder
ResBlocks only
        │
        ▼
H/2 Decoder
ResBlocks only
        │
        ▼
Full-res Decoder ResBlocks
        │
        ▼
HRMS residual
```

비교:

| 구성 | 기존 Teacher | 현재 6.694M 구조 |
|---|---|---|
| H/2 encoder CM3A | Dual branch | 제거 |
| H/4 encoder CM3A | Dual branch | MS K/V만 유지 |
| Bottleneck CM3A | Dual branch | MS K/V만 유지 |
| H/4 decoder CM3A | Dual branch | 제거 |
| H/2 decoder CM3A | Dual branch | 제거 |
| MARs | 유지 | 유지 |
| ResMod \(\beta,\gamma\) | 유지 | 유지 |
| Mode gate \(\alpha\) | 유지 | 유지 |
| PAN-HF stem | 유지 | 유지 |
| Parameters | 9.969M | 6.694M |
| Inference | 18.0ms | 8.4ms |
| 25K ERGAS | 2.2598 | 2.2527 |
| Teacher 대비 \(p\) | — | 0.114 |

정확한 해석은 다음과 같다.

> PAN K/V branch와 중복 CM3A를 제거해도 reduced-resolution 성능은 기존 Teacher와 통계적으로 구분되지 않았고, 모델 크기와 추론시간은 크게 감소했다.

성능이 실제로 향상됐다고 단정하지 않는다. 현재 데이터가 지지하는 것은 **성능 유지와 효율 개선**이다.

---

# 4. 현재 실험에서 확인된 최소 구조

현재 가장 중요한 경계는 다음 비교다.

| 구조 | 남은 CM3A | Params | ERGAS | Teacher 대비 |
|---|---|---:|---:|---|
| Core-2 | H/4 encoder + bottleneck | 6.694M | 2.2527 | 구분 불가 |
| Core-1 | bottleneck만 | 6.041M | 2.2693 | 유의하게 악화 |

따라서 현재 결과가 지지하는 최소 골격은 다음이다.

\[
\boxed{
\text{H/4 encoder CM3A}
+
\text{bottleneck CM3A}
+
\text{MARs}
+
\text{ResMod}
+
\text{PAN-HF stem}
}
\]

H/4 encoder CM3A가 중요한 이유는 다음과 같다.

1. native MS spatial scale과 대응한다.
2. local cross-modality alignment를 수행한다.
3. 출력이 bottleneck 경로와 decoder skip 양쪽으로 전달된다.

```text
H/4 encoder CM3A output
          ├─ downsample → bottleneck
          └─ skip → decoder
```

Bottleneck CM3A는 계산 비용이 작고 coarse/global context를 제공하므로 당분간 유지한다.

---

# 5. PAN K/V branch 제거 후에도 PAN 정보가 남는 이유

PAN K/V branch를 제거해도 PAN 정보는 다음 경로로 계속 전달된다.

## 5.1 Input stem

```text
PAN
LPAN
PAN − LPAN
```

이 세 입력이 처음부터 shared feature에 포함된다.

## 5.2 Shared feature

남은 MS branch의 K/V는 다음처럼 생성된다.

\[
[K_{\mathrm{MS}},V_{\mathrm{MS}}]
=
\operatorname{Conv}([x,I_{\mathrm{MS}}])
\]

여기서 \(x\) 자체가 이미 PAN 정보를 포함한다.

## 5.3 MARs PAN mode

PAN mode에서는 shared backbone 전체가 PAN high-frequency residual을 복원한다.

## 5.4 PAN-conditioned query

PAN mode에서는 query가 PAN 또는 LPAN 조건으로 형성된다.

따라서 현재 구조는 PAN 정보를 제거한 것이 아니라 다음을 수행한 것이다.

> **중복 explicit PAN K/V 경로를 제거하고, stem·shared feature·MARs·query를 통한 PAN 정보는 유지한다.**

---

# 6. 원 PAN-Crafter를 그대로 Teacher로 쓰지 않는 이유

Teacher는 단순히 가장 큰 모델이어서는 안 된다.

Teacher가 만족해야 하는 최소 조건:

- Student보다 평균 성능이 명확히 높음
- Student가 학습할 가치가 있는 추가 표현을 가짐
- 여러 seed에서 안정적임
- error 또는 uncertainty가 신뢰 가능함
- mutual learning에서 기준점 역할을 할 수 있음

현재 공개 PAN-Crafter는 다음 문제가 있다.

1. 논문 성능이 재현되지 않음
2. CANConv보다 유의하게 우수하지 않음
3. 공개 코드의 params/FLOPs가 논문과 크게 다름
4. 여러 CM3A와 PAN K/V branch가 성능에 기여하지 않음
5. 작은 구조가 거의 같은 성능을 보임

따라서 다음 접근이 더 타당하다.

> **6.694M 구조를 Clean PAN-Crafter Core로 재정의하고, 그 위에 성능 기여가 명확한 모듈만 추가해 새로운 Teacher를 설계한다.**

---

# 7. 새로운 Teacher 설계 전략

## 7.1 설계 원칙

기존 PAN-Crafter는 여러 scale에서 유사한 CM3A를 반복한다.

새 Teacher는 위치별 역할을 분리한다.

| 위치 | 담당 역할 |
|---|---|
| H/4 encoder | local PAN–MS alignment |
| H/8 bottleneck | global/frequency representation |
| H/4 decoder | explicit detail reconstruction |
| MARs | PAN spatial auxiliary supervision |
| ResMod | mode-specific representation |

목표:

```text
Parameters: 약 7–9M
Single-pass inference
공개 Teacher보다 작거나 유사한 비용
재현 PAN-Crafter보다 명확히 우수한 성능
CANConv보다 통계적으로 우수
```

---

# 8. Teacher 후보

## T0 — Clean PAN-Crafter Core

현재 6.694M 구조를 Teacher 설계 baseline으로 사용한다.

```yaml
width: 128
depth: [2, 2, 2, 2]

CM3A:
  H/4_encoder: MS-KV-only
  bottleneck: MS-KV-only

MARs: true
ResMod: true
PAN_HF_stem: true
```

목적:

- 공개 모델의 중복 경로 제거
- 이후 module contribution을 명확히 측정
- 새로운 Teacher의 clean baseline 제공

---

## T1 — Sparse Dual-Branch Alignment Teacher

H/4 encoder 한 곳에서만 PAN K/V branch를 복원한다.

```yaml
CM3A:
  H/4_encoder: dual-branch
  bottleneck: MS-KV-only
```

의도:

- 모든 scale의 PAN branch는 제거
- alignment가 가장 중요한 H/4에서만 explicit PAN structure 사용
- misalignment robustness 회복 여부 확인
- parameter 증가 최소화

주 평가:

- standard WV3 ERGAS/SAM
- synthetic shift stress test
- WV2 zero-shot
- building/car double-edge artifact

---

## T2 — Frequency Bottleneck Teacher

H/4 encoder는 MS-K/V-only CM3A를 유지하고 bottleneck에 frequency module을 추가한다.

```yaml
H/4_encoder:
  local_CM3A: true

bottleneck:
  Fourier_channel_attention: true
```

후보 모듈:

- FTCA-lite
- FFT channel gate
- frequency-domain channel attention
- global channel attention

구조적 역할:

```text
H/4 encoder: local geometric alignment
H/8 bottleneck: global/frequency context
```

Bottleneck은 해상도가 작으므로 frequency operation의 계산 부담이 제한적이다.

---

## T3 — Decoder Frequency-Fusion Teacher

H/4 decoder에서 PAN high-frequency와 MS low-frequency를 명시적으로 융합한다.

기본 condition:

\[
S\text{-Cond}
=
[L_{\mathrm{MS}},
H_{\mathrm{PAN}},
V_{\mathrm{PAN}},
D_{\mathrm{PAN}}]
\]

가능한 구현:

- SWT/DWT condition + channel gate
- PAN-HF/MS-LF depthwise projection
- lightweight cross-attention
- high-frequency residual injection
- wavelet-guided feature modulation

```yaml
H/4_decoder:
  PAN_high_frequency: true
  MS_low_frequency: true
  fusion: lightweight_gated_fusion
```

의도:

- decoder가 실제 detail을 복원하는 위치에서만 PAN 고주파 사용
- 모든 CM3A에서 PAN branch를 반복하는 비효율 제거
- spectral fidelity와 spatial detail의 역할 분리

---

## T4 — Alignment–Frequency Teacher

T1, T2, T3 중 개별적으로 효과가 확인된 모듈만 결합한다.

권장 최종 형태:

```text
H/4 Encoder:
  sparse local CM3A

H/8 Bottleneck:
  frequency/global channel module

H/4 Decoder:
  PAN-HF + MS-LF gated fusion

Other scales:
  plain ResBlocks
```

주의:

- T1, T2, T3를 한 번에 결합하지 않는다.
- 각 단일 모듈의 독립 효과를 먼저 측정한다.
- 개별 효과가 없는 모듈은 T4에 포함하지 않는다.

---

# 9. 권장 Teacher 개발 순서

## Phase 0 — 학습 파이프라인 진단 완료

현재 진행 중인 두 항목을 먼저 확인한다.

```text
d1_nocrop:
배포 crop의 scale-jitter 성격 검증

d2_lmsbase:
residual baseline을 bicubic MS 대신 dataset LMS로 변경
```

둘 중 하나가 유의하게 개선되면 이후 모든 Teacher/Student 실험의 공통 학습 pipeline으로 채택한다.

## Phase 1 — T0 재검증

```text
T0 Clean Core
50K iterations
최소 2 seeds
```

확인 항목:

- ERGAS
- SAM
- PSNR
- reduced/full-resolution
- inference time
- peak training memory
- seed variance

## Phase 2 — 단일 Teacher module 실험

```text
T1: H/4 sparse PAN branch
T2: bottleneck frequency module
T3: H/4 decoder frequency fusion
```

각 후보는 T0에서 정확히 한 요소만 변경한다.

## Phase 3 — 유효 모듈 결합

개별적으로 유의한 모듈만 결합해 T4를 구성한다.

## Phase 4 — Teacher 확정

최종 Teacher는 아래 조건을 만족해야 한다.

---

# 10. Teacher 선정 기준

## 필수 성능 조건

- 재현 PAN-Crafter보다 ERGAS가 유의하게 낮음
- CANConv보다 ERGAS 또는 SAM이 유의하게 우수
- 여러 seed에서 방향이 일관됨
- reduced-resolution 성능 개선이 full-resolution artifact 악화와 동반되지 않음

## 구조적 조건

- single-pass inference
- 약 7–9M parameters
- 공개 Teacher보다 계산량이 작거나 유사
- 24GB 환경에서 Student와 공동 학습 가능

## robustness 조건

- PAN/MS shift stress test
- local translation/affine test
- WV2 zero-shot
- building, car, road edge 분석
- spectral band별 error
- high-frequency region error

## mutual-learning 조건

Teacher가 Student보다 평균적으로 우수해야 하지만 모든 위치에서 항상 우수할 필요는 없다.

---

# 11. Teacher에서 Student 도출

새 Teacher가 확정된 뒤 Student를 만든다.

## 권장 Student 구조

| 구성 | Teacher | Student |
|---|---|---|
| Width | 128 | 128 |
| Depth | D2222 | D1121 또는 D1111 |
| H/4 alignment | enhanced CM3A | MS-K/V-only CM3A |
| Bottleneck | frequency module | lite module 또는 plain ResBlock |
| H/4 decoder fusion | full module | depthwise gate 또는 제거 |
| H/2 CM3A | 제거 | 제거 |
| MARs | 유지 | 유지 |
| ResMod | 유지 | 유지 |
| PAN-HF stem | 유지 | 유지 |

예상 크기:

```text
Teacher+: 7–9M
Student D1121: 약 4.8–5.5M
Student D1111: 약 4.0–4.7M
```

## Student에서 유지할 요소

- MARs
- H/4 encoder alignment
- mode modulation
- PAN-HF stem
- residual topology
- local window \(k=3\)

## Student에서 먼저 줄일 요소

1. Teacher 전용 frequency module
2. decoder frequency fusion
3. ResBlock depth
4. bottleneck block complexity

Channel width는 마지막 축으로 둔다.

---

# 12. Mutual learning 적합성 평가

독립 학습된 Teacher와 Student에 대해 다음을 측정한다.

## 12.1 Error correlation

\[
\rho(e_T,e_S)
\]

두 모델의 오류 패턴이 지나치게 같으면 mutual learning의 추가 이득이 작을 수 있다.

## 12.2 Student-win ratio

\[
P(e_S<e_T)
\]

Student가 Teacher보다 정확한 pixel 또는 patch 비율이다.

이 값이 거의 0이면 Student→Teacher 방향의 학습 근거가 약하다.

## 12.3 Oracle gain

\[
e_{\mathrm{oracle}}(p)
=
\min(e_T(p),e_S(p))
\]

위치별로 더 좋은 모델을 선택했을 때 Teacher 단독보다 얼마나 개선되는지 본다.

## 12.4 영역별 분석

- edge / flat region
- high / low local variance
- PAN–MS misalignment region
- object boundary
- spectral band별 error
- low/high-frequency error
- urban / vegetation / water region

---

# 13. 최종 Teacher–Student 구조

```text
                    New Teacher+
          Sparse Alignment + Frequency Refinement
                             │
             ┌───────────────┴───────────────┐
             │                               │
       H/4 local CM3A                 Bottleneck frequency
             │                               │
             └───────────────┬───────────────┘
                             │
                    H/4 decoder fusion
                             │
                         HRMS_T
                             ⇅
                       Mutual Learning
                             ⇅
                    Lite Student
      Sparse Alignment + Direct Residual Reconstruction
                             │
                 H/4 MS-K/V-only CM3A
                             │
                   Lite/plain bottleneck
                             │
                         HRMS_S
```

---

# 14. OOM 및 24GB 운영 원칙

목표 크기에서는 두 모델의 공동 학습이 24GB 안에 들어갈 가능성이 높지만, MARs가 batch를 mode별로 복제하므로 peak activation을 반드시 측정해야 한다.

권장 시작점:

```yaml
precision: bf16_or_fp16
physical_batch: 4_to_8
gradient_accumulation: adjust_to_target_batch
feature_mutual_locations:
  - H/4
  - H/8
```

OOM 발생 시 축소 순서:

1. all-stage feature mutual 제거
2. physical batch 축소
3. Teacher frequency block checkpointing
4. mutual feature를 H/8 한 곳으로 제한
5. strict simultaneous backward를 alternating update로 변경
6. Student depth D1121 → D1111
7. 마지막에 Student width 축소

Alternating update:

```text
Step A:
Teacher no_grad
Student backward
T → S update

Step B:
Student no_grad
Teacher backward
S → T update
```

같은 minibatch를 사용하되 한 번에 하나의 backward graph만 유지한다.

---

# 15. 실험 관리 원칙

## 비교 기준

모든 결과는 논문 수치가 아니라 다음 내부 기준으로 비교한다.

```text
공개 코드 Teacher
동일 profiler
동일 dataset split
동일 evaluation code
동일 seed policy
동일 training iterations
```

## 주 지표

- ERGAS
- SAM
- PSNR
- paired statistical test
- seed variance

## 보조 지표

- HQNR
- \(D_s\)
- \(D_\lambda\)
- Q8/Q4
- SSIM
- SCC

Q8, SSIM, SCC는 현재 축소 범위에서 포화되어 있으므로 Student 또는 Teacher 선정을 위한 주 근거로 사용하지 않는다.

\(D_s\)와 HQNR은 PAN detail injection이 줄어들 때 좋아질 수 있으므로 reduced-resolution GT 성능과 함께 해석한다.

---

# 16. 최종 권장 전략

```text
1. 공개 PAN-Crafter를 최종 Teacher로 확정하지 않는다.

2. 6.694M 구조를 Clean PAN-Crafter Core로 정의한다.

3. Clean Core에서 다음을 개별적으로 시험한다.
   - H/4 sparse dual branch
   - bottleneck frequency module
   - H/4 decoder frequency fusion

4. 유효한 모듈만 결합해 7–9M의 새로운 Teacher를 만든다.

5. 새 Teacher가 재현 PAN-Crafter와 CANConv보다
   통계적으로 우수한지 검증한다.

6. 새 Teacher에서 D1121 또는 D1111 Student를 도출한다.

7. Teacher–Student의 성능뿐 아니라 오류 상보성을 측정한다.

8. 상보성이 확인될 때 uncertainty-gated mutual learning으로 진행한다.
```

최종 구조 방향:

\[
\boxed{
\text{Clean PAN-Crafter Core}
+
\text{Sparse H/4 Alignment}
+
\text{Bottleneck Frequency Modeling}
+
\text{Decoder Detail Fusion}
}
\]

그리고 Student는 다음과 같이 정의한다.

\[
\boxed{
\text{Sparse Alignment}
+
\text{Reduced Depth}
+
\text{Lite Frequency Path}
}
\]

이 접근은 원 PAN-Crafter를 단순히 줄이는 방식이 아니라, 공개 구조에서 불필요한 중복을 제거한 후 **성능 기여가 검증된 기능만 Teacher에 추가하고, 그 Teacher로부터 목적에 맞는 Student를 도출하는 전략**이다.

---

## Source Basis

이 문서는 다음 자료를 기준으로 작성했다.

- `2026-08-24_reproduction-audit.md`
  - 공개 코드와 논문 수치의 불일치
  - PAN-Crafter와 CANConv 비교
  - 6.694M/6.041M 구조 경계
  - 다중 지표 및 통계 검정
- `2026-08-24_WIP_running.md`
  - crop 및 LMS residual baseline 진단 진행 상황
- `2026-08-20_student-architecture-sweep.md`
  - width/depth/CM3A 스윕
  - CM3A 축소와 width 축소의 효율 차이
- `PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening`
  - MARs, CM3A, ResMod 및 multi-scale architecture
- `U-Know-DiffPAN`
  - FTCA, SWTCA, PAN-HF/MS-LF frequency conditioning의 설계 근거

진행 중인 crop 및 residual baseline 진단 결과에 따라 Phase 0 이후의 공통 학습 pipeline은 변경될 수 있다.
