# s2 단일-task HRMS 재구성 실험 및 Uncertainty 검증 계획

> 작성일: 2026-08-31  
> 대상 서버: **s2**  
> 기본 backbone: **R4 (`width=96`, `depth=[1,2,4]`, attention 없음, 11ch, nocrop)**

---

## 0. 목적

현재 c6/R4 계열은 CM3A attention은 제거됐지만, 기본 설정에서는 PAN-Crafter의 **MARs dual-task 학습**이 남아 있다.
즉 하나의 네트워크가 같은 입력으로 다음 두 작업을 번갈아 수행한다.

1. **MS mode:** HRMS 재구성
2. **PAN mode:** 반복된 PAN 영상 back-reconstruction

이번 s2 실험은 PAN mode와 PAN reconstruction loss를 제거했을 때를 명확하게 분리하여 확인한다.

핵심 질문은 다음 두 가지다.

1. PAN auxiliary task를 제거하면 R4의 HRMS 복원 성능이 어떻게 변하는가?
2. PAN mode뿐 아니라 mode-switching 파라미터까지 제거한 **순수 single-task residual U-Net**에서 mutual/KD loss 연구를 진행하는 것이 더 적절한가?

---

## 1. 용어 정리: “MS-only”가 정확히 의미하는 것

이 문서에서 `MS-only`는 **입력에서 PAN을 제거한다는 뜻이 아니다.**

### 유지되는 입력

- 고해상도 PAN
- 업샘플된 LRMS
- LPAN
- PAN high-frequency (`PAN - upsampled LPAN`)
- 현재 11ch 입력 구성

### 제거되는 것

- PAN mode forward
- PAN back-reconstruction target
- PAN reconstruction loss
- batch의 MS/PAN mode 복제
- PAN-mode 전용 modulation 파라미터
- 최종 clean case에서는 모든 mode-switching 파라미터

### 출력과 supervision

```text
Input  : PAN + LRMS-derived 11ch input
Output : HRMS only
Target : GT HRMS only
Loss   : L1/Charbonnier(output HRMS, GT HRMS)
```

따라서 정확한 명칭은 다음이 적절하다.

> **single-task HRMS reconstruction network**  
> 또는  
> **MS-mode-only pansharpening network**

PAN은 여전히 입력 guidance로 사용되므로, `MS-input-only network`는 아니다.

---

## 2. PAN-Crafter에서 완전히 벗어난 clean baseline의 조건

PAN-Crafter supplementary의 `w/o MARs` 정의에 맞추려면 단순히 `lambda_pan=0`만 설정해서는 부족하다.

다음 요소를 모두 제거해야 한다.

```text
1. PAN mode 제거
2. PAN target 생성 제거
3. batch duplication 제거
4. mode argument 제거
5. beta_ms, gamma_ms 제거
6. beta_pan, gamma_pan 제거
7. alpha 관련 파라미터 제거
8. PAN reconstruction loss 제거
```

현재 R4에는 attention이 없으므로 `alpha`는 사실상 존재하지 않을 수 있다. 그래도 파라미터 이름과 forward graph를 검사한다.

### Clean ResBlock

기존 mode-conditioned block:

```text
x = Conv(SiLU(LN(x)))
x = x + Conv(SiLU(Modulate(LN(x), mode)))
```

clean single-task block:

```text
x = Conv(SiLU(LN(x)))
x = x + Conv(SiLU(LN(x)))
```

즉 mode-dependent affine modulation을 제거한다.

---

## 3. 반드시 실행할 두 실험

기존 dual-MARs R4 결과를 기준점으로 사용하되, 아래 두 case를 신규 실행한다.

### Case S2-MS1 — PAN task만 제거, MS modulation 유지

**목적:** PAN auxiliary supervision의 효과만 우선 분리한다.

```yaml
id: S2_MS1_R4_MS_ONLY_MODED
model:
  width: 96
  depth: [1, 2, 4]
  attention: none
  input_channels: 11
  mode: fixed_ms
  keep_ms_modulation: true
  keep_pan_modulation: false
training:
  duplicate_batch_for_modes: false
loss:
  hrms_reconstruction: true
  pan_reconstruction: false
```

#### Forward

```python
pred_hrms = model(input_11ch, mode="MS") + upsampled_ms
```

#### Loss

\[
L_{MS1}=\rho(\hat Y, Y_{GT})
\]

여기서 `rho`는 기존 R4와 동일한 L1 또는 Charbonnier를 사용한다.

#### 해석

- 이 case가 dual MARs보다 나빠지면 PAN auxiliary task의 regularization 이득이 존재한다.
- 이 case가 좋아지면 PAN back-reconstruction이 현재 compact R4에서는 negative transfer를 만들 가능성이 있다.
- 아직 MS modulation은 남아 있으므로 완전한 일반 U-Net은 아니다.

---

### Case S2-MS2 — 완전한 single-task plain R4

**목적:** PAN-Crafter의 MARs 요소를 제거한 순수 HRMS reconstruction backbone을 확립한다.

```yaml
id: S2_MS2_R4_MS_ONLY_PLAIN
model:
  width: 96
  depth: [1, 2, 4]
  attention: none
  input_channels: 11
  mode_argument: false
  all_mode_modulation: false
training:
  duplicate_batch_for_modes: false
loss:
  hrms_reconstruction: true
  pan_reconstruction: false
```

#### Forward

```python
pred_residual = model(input_11ch)
pred_hrms = pred_residual + upsampled_ms
```

#### Loss

\[
L_{MS2}=\rho(\hat Y, Y_{GT})
\]

#### 해석

이 case가 본 연구에서 말하는 가장 깨끗한 baseline이다.

> PAN과 LRMS를 입력받지만, HRMS 한 가지 출력만 생성하고 HRMS GT 한 가지로만 supervision받는 residual U-Net.

이 모델을 이후 SiS, edge, mutual learning 또는 KD의 공통 backbone으로 사용하면 PAN-Crafter의 MARs 기여와 새 loss 기여가 섞이지 않는다.

---

## 4. 기준선 및 비교표

| ID | PAN mode | PAN loss | MS modulation | PAN modulation | 출력 | 의미 |
|---|---:|---:|---:|---:|---|---|
| `R4_DUAL_MARS` | O | O | O | O | HRMS / PAN | 기존 PAN-Crafter-derived R4 |
| `S2_MS1_R4_MS_ONLY_MODED` | X | X | O | X | HRMS | PAN task 효과 분리 |
| `S2_MS2_R4_MS_ONLY_PLAIN` | X | X | X | X | HRMS | clean single-task baseline |

가능하면 세 case를 동일 seed, 동일 data order, 동일 50K scheduler로 비교한다.

---

## 5. 구현 요구사항

### 5.1 Dataset / batch

Dual MARs 기존 방식:

```python
batch_ms  = duplicate(batch)
batch_pan = duplicate(batch)
combined_batch = concat(batch_ms, batch_pan)
```

MS-only 방식:

```python
combined_batch = batch
```

즉 effective batch가 기존 dual MARs의 절반이 될 수 있다.

공정한 비교를 위해 두 가지 중 하나를 명시한다.

#### 권장안 A — nominal batch 유지

```text
R4 dual MARs : nominal B, internal 2B
MS-only      : nominal B, actual B
```

이는 실제 학습 비용과 auxiliary-task 효과를 함께 비교한다.

#### 보조 control B — sample-exposure matched

```text
MS-only nominal batch를 2B로 증가
```

VRAM이 허용되는 경우에만 추가한다. 주 결과는 권장안 A로 유지한다.

---

### 5.2 Model API

#### 기존

```python
pred = model(x, mode="MS")
pred_pan = model(x, mode="PAN")
```

#### clean single-task

```python
pred = model(x)
```

config에서 다음을 명확히 분리한다.

```yaml
training_mode: dual_mars | ms_only_modulated | ms_only_plain
```

---

### 5.3 제거 여부 검사

`ms_only_plain` build 후 다음 조건을 모두 만족해야 한다.

```python
for name, _ in model.named_parameters():
    assert "beta_ms" not in name
    assert "gamma_ms" not in name
    assert "beta_pan" not in name
    assert "gamma_pan" not in name
    assert "alpha_ms" not in name
    assert "alpha_pan" not in name
```

또한:

```text
- forward signature에 mode 없음
- PAN output head 없음
- PAN target 생성 없음
- pan_loss 로그 없음
- batch duplication 없음
```

---

## 6. Smoke test

각 신규 case는 50K 실행 전 100 iteration smoke test를 수행한다.

### 필수 검사

```text
1. forward/backward 정상
2. NaN/Inf 없음
3. output shape = GT HRMS shape
4. PAN input에 non-zero gradient가 존재
5. LRMS input에 non-zero gradient가 존재
6. PAN reconstruction graph가 생성되지 않음
7. model parameter names에 PAN/MS mode modulation이 예상대로 존재/부재
8. 실제 batch size가 의도와 일치
9. loss log에 hrms_recon만 존재
```

### 중요

`ms_only_plain`에서도 PAN input gradient가 0이면 네트워크가 사실상 multispectral super-resolution만 수행하는 것이므로 실패다.

---

## 7. 학습 설정

```yaml
iterations: 50000
optimizer: AdamW
lr: 1.0e-4
weight_decay: 0.01
scheduler: cosine
warmup_steps: 100
seed: 2025
input_channels: 11
crop: false
augmentation:
  horizontal_flip: true
  vertical_flip: true
  rotation_90: true
checkpoint_selection: final_50k_or_val_ergas
```

FR HQNR로 checkpoint를 선택하지 않는다.

---

## 8. 평가 지표

### RR

- ERGAS
- SAM
- SCC
- PSNR
- SSIM
- Q8

### FR

- HQNR
- D_lambda
- D_s

### 추가 진단

- output gradient magnitude
- PAN–output high-pass correlation
- edge-region MAE
- smooth-region MAE
- training/validation gap

### 핵심 판정

PAN mode 제거 후 다음과 같은 패턴이 예상될 수 있다.

```text
ERGAS/SAM 악화 + D_s 개선
```

이는 PAN auxiliary supervision을 제거하면서 output이 보수적·평활해진 경우일 수 있으므로 HQNR만으로 성공 판정하지 않는다.

---

## 9. s2 실행 순서와 예상 시간

현재 R4 single run이 약 2시간 내외, two-peer run이 약 4시간 내외였다는 실측을 기준으로 한다.

| 순서 | Case | 예상 시간 |
|---:|---|---:|
| 1 | `S2_MS1_R4_MS_ONLY_MODED` | 1.8–2.2h |
| 2 | 평가 및 config audit | 0.3h |
| 3 | `S2_MS2_R4_MS_ONLY_PLAIN` | 1.8–2.2h |
| 4 | 평가 및 비교표 생성 | 0.5h |

총 예상 시간은 약 4.4–5.2시간이다.

남는 시간에는 아래 optional case를 순차적으로 수행한다.

---

## 10. Edge와 SiS의 실험 순서 수정

SIPSA-Net은 edge loss와 SiS loss를 공간·분광 보존을 위한 상보적 손실로 함께 사용한다.
또한 SIPSA의 `w/o SiS` 비교는 edge loss가 남은 상태에서 SiS만 제거한 비교에 가깝다.

따라서 source-faithful한 순서는 다음이 더 적절하다.

```text
E0: plain HRMS reconstruction
E1: E0 + edge
E2: E0 + SiS            # 독립 효과 확인용
E3: E0 + edge + SiS     # 주 방법 후보
```

예산이 제한되면 우선순위는:

```text
E0 → E1 → E3 → E2
```

이다.

### 이유

- Edge loss는 PAN의 공간 구조를 명시적으로 anchor한다.
- SiS는 주변 shifted MS 중 최소 오차를 선택하므로 spectral target의 위치 제약을 완화한다.
- PAN mode를 제거한 single-task pipeline에서는 edge loss가 없으면 명시적 PAN spatial loss가 완전히 사라진다.
- 따라서 SiS를 주 방법으로 평가할 때는 edge가 먼저 또는 동시에 존재하는 편이 안정적이다.

### 단, pure baseline은 edge 없이 유지

`S2_MS2_R4_MS_ONLY_PLAIN`은 반드시 HRMS GT reconstruction loss만 사용한다.
Edge와 SiS는 그 이후 method ablation이다.

---

## 11. Optional loss ablation queue

### E1 — Plain + Edge

\[
L_{E1}=L_{HRMS}+\lambda_{edge}L_{edge}
\]

\[
L_{edge}=\left\|\,|\nabla \operatorname{Lum}(\hat Y)|-|\nabla PAN|\,\right\|_1
\]

### E2 — Plain + SiS

\[
L_{E2}=L_{HRMS}+\lambda_{SiS}L_{SiS}
\]

### E3 — Plain + Edge + SiS

\[
L_{E3}=L_{HRMS}+\lambda_{edge}L_{edge}+\lambda_{SiS}L_{SiS}
\]

초기 가중치는 gradient norm을 측정하여 HRMS anchor 대비 각 auxiliary gradient가 10–20% 수준이 되도록 맞춘다.

---

## 12. Clean single-task 기반 mutual learning 확장

`S2_MS2_R4_MS_ONLY_PLAIN`이 안정적으로 학습된 뒤, MARs가 없는 상태에서 동일 구조 DML을 재검증할 수 있다.

### P-M0 — Plain two-peer independent

\[
L_A=L_{HRMS}^A,\qquad L_B=L_{HRMS}^B
\]

### P-M1 — Plain residual mutual

\[
L_A=L_{HRMS}^A+\lambda_m\rho(R_A-\operatorname{sg}(R_B))
\]

\[
L_B=L_{HRMS}^B+\lambda_m\rho(R_B-\operatorname{sg}(R_A))
\]

이 비교는 기존 R4–R4 DML 효과가 MARs에 의존하는지 분리한다.

---

# Appendix A. T1 Teacher uncertainty 상위 10/20/30% 오류 검증

## A.1 현재 결과로 바로 가능한가?

**T1 checkpoint, RR prediction, uncertainty map, GT HRMS가 서버에 남아 있다면 재학습 없이 바로 가능하다.**

현재 sheet에 있는 ERGAS/SAM/HQNR 집계값만으로는 계산할 수 없다.
필요한 것은 pixel-level tensor다.

```text
pred_hrms : [B, C, H, W]
uncertainty: [B, 1, H, W] 또는 [B, C, H, W]
gt_hrms   : [B, C, H, W]
```

FR에는 HRMS GT가 없으므로 percentile-error 검증은 RR validation/test에서 수행한다.

---

## A.2 실제 오류 정의

기본 pixel error:

\[
e(x)=\frac{1}{C}\sum_{c=1}^{C}|\hat Y_c(x)-Y_c(x)|
\]

추가로 다음을 별도 기록한다.

- spectral angle error
- gradient/high-frequency error
- local ERGAS proxy

---

## A.3 Percentile 분석

각 이미지 안에서 uncertainty percentile을 계산한 뒤 장면별 결과를 평균한다.

```text
Top 10%: U >= q90
Top 20%: U >= q80
Top 30%: U >= q70
```

보고 항목:

| 영역 | 평균 MAE | 전체 대비 error lift | pixel 비율 |
|---|---:|---:|---:|
| All |  | 1.0 | 100% |
| Top 10% U |  |  | 10% |
| Top 20% U |  |  | 20% |
| Top 30% U |  |  | 30% |
| Bottom 10% U |  |  | 10% |

\[
\text{Error Lift}_{top-k}=\frac{E[e\mid U\in top-k]}{E[e]}
\]

좋은 uncertainty map이라면 일반적으로:

```text
Top10 error > Top20 error > Top30 error > All error > Bottom10 error
```

형태가 나타나야 한다.

중첩 percentile뿐 아니라 다음 disjoint bin도 함께 본다.

```text
0–10, 10–20, ..., 90–100 percentile
```

---

## A.4 필수 통계

```text
1. Spearman rho(U, absolute error)
2. Pearson correlation은 보조
3. error lift at top 10/20/30%
4. risk–coverage curve
5. uncertainty와 GT variance의 상관
6. edge 영역과 smooth 영역에서 각각의 correlation
```

### 판정 예시

```text
유효:
  Spearman rho > 0
  Top10 error lift가 명확히 1보다 큼
  percentile이 높아질수록 error가 단조 증가

무효 또는 재검토:
  rho ≈ 0
  Top10과 Bottom10 error가 유사
  uncertainty가 texture brightness만 반영
  uncertainty 방향이 반대로 나타남
```

주의: 구현이 `log variance`, `precision`, `confidence` 중 무엇을 출력하는지 먼저 확인한다.
값이 클수록 uncertainty가 높은 정의인지 반드시 검증한다.

---

## A.5 권장 실행 명령 형식

```bash
python tools/analyze_uncertainty.py \
  --config configs/t1_c6_unc.yaml \
  --checkpoint checkpoints/T1_c6_unc_50k.pt \
  --split val \
  --percentiles 10 20 30 \
  --per-image-quantile \
  --save-csv outputs/t1_uncertainty_percentile.csv \
  --save-fig outputs/t1_uncertainty_calibration.png
```

---

## 13. 최종 의사결정

| 결과 | 해석 | 다음 단계 |
|---|---|---|
| MS1/Plain 모두 dual MARs보다 악화 | PAN auxiliary task가 유효 | MARs 유지, 새 loss는 dual 구조에서 검증 |
| MS1은 동급, Plain은 악화 | MS modulation이 유효 | PAN task 제거 가능, MS modulation 유지 고려 |
| Plain이 dual MARs와 동급 | PAN-Crafter 요소 불필요 | clean single-task backbone으로 전환 |
| Plain이 더 우수 | MARs가 compact model에서 negative transfer | clean backbone을 mutual/KD 공통 기반으로 채택 |
| Edge가 Plain 개선 | PAN spatial anchor 필요 | Edge를 SiS보다 먼저 유지 |
| Edge+SiS가 Edge보다 개선 | SiS 기여 성립 | 최종 loss 후보로 채택 |
| SiS가 metric은 낮추지만 artifact를 줄임 | SIPSA와 유사한 trade-off | shift robustness·local visual metric으로 판단 |

---

## 14. 요약

가장 중요한 신규 case는 다음 두 개다.

```text
S2_MS1_R4_MS_ONLY_MODED
S2_MS2_R4_MS_ONLY_PLAIN
```

두 번째 case가 성공적으로 구현되면 현재 pipeline은 다음처럼 정리된다.

> **PAN과 LRMS를 입력받아 HRMS 하나만 복원하고, HRMS GT 하나로만 학습하는 standard residual U-Net pansharpening pipeline.**

이는 MARs, PAN back-reconstruction, mode switching이 제거됐으므로 PAN-Crafter의 제안 구조에서 깔끔하게 분리된 baseline이다.
