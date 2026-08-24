# PAN-Crafter Student 경량화: 다음 실험 계획

## 1. 현재 결론

Teacher는 원본 PAN-Crafter인 `W128-D2222-A5`로 고정한다.

현재 스윕 결과에서는 **channel width보다 CM3A 개수와 ResBlock depth를 먼저 줄이는 전략**이 가장 효율적이다.

- `CM3A 5 → 3`은 W128에서 성능 차이가 통계적으로 구분되지 않았지만, 추론시간과 학습 메모리는 크게 감소했다.
- `width 96 → 64`는 파라미터를 크게 줄였으나 속도 이득은 거의 없고 ERGAS가 크게 악화됐다.
- 따라서 mutual learning용 Student는 **width 128 또는 최소 96을 유지**하는 것이 적절하다.
- Depth 축은 validation과 test 결과가 엇갈렸으므로 50K 학습 전까지 판정을 보류한다.

---

## 2. 유지할 구성

Student에서도 다음 요소는 유지한다.

- **MARs**
  - MS mode
  - PAN back-reconstruction mode
  - mode-dependent `α, β, γ`
- **Residual reconstruction**
- **Local attention window `k=3`**
- **최소 1개 이상의 CM3A**
- **U-Net multi-scale topology와 skip connection**
- **channel width 128 또는 최소 96**

PAN-Crafter 논문에서도 MARs가 가장 큰 단독 성능 향상을 보였고, CM3A는 MARs와 결합할 때 효과가 커졌다. 따라서 MARs 제거 또는 CM3A 완전 제거는 첫 Student 후보에 적합하지 않다.

---

## 3. 먼저 줄일 부분

경량화 우선순위는 다음과 같다.

```text
H/2 CM3A 제거
→ CM3A 개수 및 위치 축소
→ ResBlock depth 축소
→ channel width 축소
```

특히 encoder와 decoder의 H/2 CM3A 두 개는 우선 제거 대상으로 본다.

### 제외할 방향

- `W64`, `W32`를 mutual-learning 주 Student로 사용
- CM3A 전체 제거
- MARs 제거
- width를 먼저 크게 줄이는 방식

---

## 4. 주 Student 후보

### Main candidate

```text
W128-D1121-A1
```

권장 설정:

```yaml
width: 128
depth: [1, 1, 2, 1]
CM3A_count: 1
MARs: true
local_window: 3
alpha_beta_gamma: true
```

동일한 `D1121-A1` 구조에서 parameter가 대체로 width 제곱에 비례한다고 가정하면:

\[
P_S \approx 2.509
\left(\frac{128}{96}\right)^2
\approx 4.46\text{M}
\]

따라서 Teacher 9.97M 대비 예상 비율은 약 45%다.

> 이 수치는 설계 추정치이며 실제 구현 후 parameter와 FLOPs를 다시 측정해야 한다.

---

## 5. 보조 후보

| 역할 | 구성 | 현재 의미 |
|---|---|---|
| Main | `W128-D1121-A1` | 목표 크기 약 45%, width 보존형 |
| Balanced | `W96-D1121-A3` | alignment capacity와 효율의 절충 |
| Efficient | `W96-D1121-A1` | 2.51M, 4.7ms, 4.67GB의 최경량 실용 후보 |
| Upper bound | `W128-D2222-A3` | H/2 CM3A 제거의 안전성 기준, Teacher-lite에 가까움 |

---

## 6. 다음 최소 실험

### Phase 1 — 핵심 후보 50K 재학습

```text
Teacher: W128-D2222-A5
Main:    W128-D1121-A1
Balanced: W96-D1121-A3
Efficient: W96-D1121-A1
```

동일 seed, 동일 split, 동일 MARs 설정으로 50K까지 학습한다.

### Phase 2 — A1 위치 실험

`W128-D1121`을 고정하고 CM3A 위치만 비교한다.

| 설정 | CM3A 위치 |
|---|---|
| A1-E4 | H/4 encoder |
| A1-B | H/8 bottleneck |
| A1-D4 | H/4 decoder |
| A2-Sym | H/4 encoder + H/4 decoder |
| A3 | H/4 encoder + H/8 + H/4 decoder |

A1의 위치가 명확하지 않으면 “CM3A 한 개로 충분하다”는 결론을 내리기 어렵다.

### Phase 3 — Depth 비교

주 후보가 결정된 뒤 다음만 비교한다.

```text
D1121: 중간 해상도 표현력 유지
D1111: 최대 경량화
D2111: full-resolution detail 강화
```

---

## 7. Mutual learning용 Student 선정 기준

Student는 단독 성능과 효율만으로 선택하지 않는다.

\[
\text{Quality}
+
\text{Efficiency}
+
\text{Complementarity}
\]

필수 측정:

### Error correlation

\[
\rho(e_T,e_S)
\]

낮을수록 Teacher와 Student의 오류 패턴이 다르다.

### Student-win ratio

\[
P(e_S<e_T)
\]

Student가 Teacher보다 정확한 pixel 또는 patch 비율이다.

### Oracle gain

\[
e_{oracle}(p)=\min(e_T(p),e_S(p))
\]

위치별로 더 정확한 모델을 선택했을 때 Teacher 단독 대비 얼마나 개선되는지 측정한다.

추가로 다음 영역별 분석을 수행한다.

- edge / flat region
- high / low GT local variance
- PAN–MS misalignment가 큰 patch
- spectral band별 error
- low/high-frequency error
- 건물, 차량, 도로 등 고주파 객체

---

## 8. 결과 해석 시 주의사항

- 현재 스윕은 25K 학습이며, 원 Teacher는 25K에서 50K로 갈 때 ERGAS가 약 3.6% 개선됐다.
- 작은 모델이 더 빨리 수렴할 수 있어 25K 비교는 소형 Student에 다소 유리할 수 있다.
- Depth 축은 validation 1,080장과 test 20장에서 방향이 달라 아직 결론을 내릴 수 없다.
- Full-resolution HQNR은 모델 크기에 둔감했으므로 Student 선택의 주 지표로 사용하지 않는다.
- 주 기준은 reduced-resolution GT 성능, Teacher와의 오류 상보성, 그리고 실제 속도·메모리다.

---

## 9. 최종 권장

```text
살릴 것:
MARs, α/β/γ, k=3, residual topology, width 128 또는 96,
H/4 또는 H/8의 최소 CM3A

먼저 줄일 것:
H/2 CM3A → 전체 CM3A 개수 → ResBlock depth

마지막에 줄일 것:
channel width
```

현재 가장 우선적으로 시험할 구조는 다음이다.

\[
\boxed{\texttt{W128-D1121-A1}}
\]

이 모델을 50K까지 학습하고, `W96-D1121-A3`, `W96-D1121-A1`과 함께 성능·시간·메모리·오류 상보성을 비교한 뒤 mutual learning용 Student를 최종 결정한다.
