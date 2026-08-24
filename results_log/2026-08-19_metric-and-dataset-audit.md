# 지표·데이터셋 현황 감사 (2026-08-19)

두 논문(`../2505.23367v2.pdf` PAN-Crafter, `../uknowdiff.pdf` U-Know-DiffPAN)이 보고하는
**모든 지표를 낼 수 있는가**, **데이터셋에 빠진 것이 있는가**, **측정이 왜 어려웠는가** 를 정리한다.

**결론: 두 논문의 지표는 전부 산출 가능하고, 데이터도 전부 갖춰져 있다.**
오늘 SCC 정의 오류를 바로잡고 SSIM 을 추가해 마지막 구멍을 메웠다.

---

## 1. 지표 — 전부 가능하다

DLPan-Toolbox `Tools/indexes_evaluation.m` 이 반환하는 **표준 RR 프로토콜은 5개뿐**이다.

```matlab
[Q_index, SAM_index, ERGAS_index, sCC, Q2n_index] = indexes_evaluation(...)
  → q2n(), Q(), SAM(), ERGAS(), SCC()   % PSNR 과 SSIM 은 호출하지 않는다
```

FR 은 `Tools/indexes_evaluation_FS.m` → `D_lambda`, `D_S`, `QNR/HQNR`.

| 지표 | 표준 프로토콜 | 우리 도구 | WV3 baseline 실측 | 논문 | 차이 |
|---|---|---|---:|---:|---|
| ERGAS↓ | O | `eval_dlpan.py` | 2.1633 | 2.040 | +6.0% |
| SAM↓ | O | `eval_dlpan.py` | 2.9093 | 2.787 | +4.4% |
| **SCC↑** | O | `eval_dlpan.py` **(오늘 교정)** | **0.9900** | 0.988 | **+0.2%** |
| Q2n (Q8/Q4)↑ | O | `eval_dlpan.py` | 0.9165 | 0.922 | −0.6% |
| Q↑ | O | 미구현 | — | (두 논문 모두 미보고) | — |
| D_λ↓ | O | `eval_dlpan_fr.py` | 0.0245 | 0.016 | +53% |
| D_s↓ | O | `eval_dlpan_fr.py` | 0.0277 | 0.027 | +2.6% |
| HQNR↑ | O | `eval_dlpan_fr.py` | 0.9486 | 0.958 | −1.0% |
| **SSIM↑** | **X** | `eval_dlpan.py` **(오늘 추가)** | **0.9754** | 0.976 | **−0.06%** |
| PSNR↑ | **X** | `eval_dlpan.py` | 37.5135 | 37.956 | −1.2% |
| Params | — | 측정 가능 | 9.969 M | 9.97 M | 일치 |
| Memory / Time | — | 측정 가능 | 1.81 GB / — | 1.711 GB / 0.009 s | — |
| FLOPs | — | 미측정 (`fvcore`/`thop` 필요) | — | — | — |

**SSIM 과 SCC 는 논문과 0.2% 이내로 맞는다.** ERGAS·SAM 이 4~6% 뒤지는 것은 지표 구현이 아니라
**모델 성능 차이**다(시드 1개, 50k 1회). D_λ 가 크게 벌어지는 것은 4절의 FR 테스트셋 이슈 때문이다.

## 2. 데이터셋 — 두 논문이 요구하는 것은 전부 있다

| 센서 | train | valid | test reduced | test full | 밴드 | lpan 상태 |
|---|---:|---:|---:|---:|---:|---|
| WV3 | 9,714 | 1,080 | 20 | 20 | 8 | 배포본 O / **full-res 손상 → 복구본 O** |
| QB | 17,139 | 1,905 | 20 | 20 | 4 | 배포본 O / **full-res 손상 (미복구)** |
| GF2 | 19,809 | 2,201 | 20 | 20 | 4 | 배포본 O (전부 정상) |
| WV2 | **없음** | **없음** | 20 | 20 | 8 | **직접 생성 (검증 불가)** |

- **PAN-Crafter** 요구: WV3/QB/GF2 학습+평가, WV2 zero-shot 평가 → **전부 충족**
  (WV2 train/valid 는 원래 존재하지 않는다. zero-shot 전용이라 결손이 아니다.)
- **U-Know-DiffPAN** 요구: WV3/QB/GF2 학습+평가 → **전부 충족**. WV2 는 쓰지 않는다.

### `lpan`(`*_pan.h5`)이 필요한 쪽은 PAN-Crafter 뿐이다

| | lpan 필요? | 근거 |
|---|---|---|
| PAN-Crafter | **필요** | 입력 concat(`↑4 lpan`, `pan − ↑4 lpan`)과 CM3A value 경로 |
| U-Know-DiffPAN | **불필요** | 입력이 `[X_t \| I_PAN \| I_MS^LR]` 뿐 (Eq 3, 15) |

**U-Know-DiffPAN 재현에는 `lpan` 결함(4절 ②)이 아예 영향을 주지 않는다.**

### 두 논문 밖의 자산 (`../CANConv`)

`cas500`(train/valid/test, scale 4095), `vantor`(finetune/valid/test, scale 16383).
붙이려면 feeder 의 `max_pixel` 문자열 추론을 먼저 고쳐야 한다([B-1~B-3](../KNOWN_ISSUES.md)).

---

## 3. 측정이 어려웠던 이유 — 확인된 것 7가지

### ① MATLAB 런타임이 없다 (해결)

README 는 "All evaluation metrics were measured using the official MATLAB code from the
DLPan-Toolbox" 라고 명시한다. 툴박스 소스는 `../DLPan-Toolbox` 에 있으나
**`matlab`·`octave` 모두 설치돼 있지 않다.** 데이터 문제가 아니라 도구 문제였다.

→ `../CANConv/tools/eval_rr.py` / `eval_fr.py` 의 파이썬 포팅을 재사용해 우회했다.
CANNet 논문 대비 reduced 0.7%, full HQNR 0.2% 이내로 검증된 구현이다.

### ② 표준 프로토콜에 PSNR·SSIM 이 없다 (구조적 한계)

`indexes_evaluation.m` 은 Q/SAM/ERGAS/sCC/Q2n 만 반환한다. 그런데 **두 논문 모두
PSNR 과 SSIM 을 보고한다.** 즉 각자 따로 구현했고, 논문에 정의가 없다.

PSNR 은 관례에 따라 **1.5 dB** 이 갈린다.

| 관례 | WV3 baseline |
|---|---:|
| 전 밴드 통합 MSE, peak=2^L | **37.513** ← 논문(37.956)과 일관 |
| 밴드별 PSNR 평균, peak=2^L | 39.066 |
| peak = 영상별 GT 최대값 | 37.195 |

밴드별 평균을 쓰면 **다른 모든 지표가 논문보다 나쁜데 PSNR 만 좋아지는 모순**이 생긴다.
그래서 통합 MSE 를 기본으로 채택했다. 다만 **어느 쪽도 확정할 수 없다.**

### ③ 같은 이름, 다른 정의 — SCC (오늘 발견·교정)

| 구현 | 정의 | WV3 값 |
|---|---|---:|
| DLPan `SCC.m` | Sobel 기울기 크기의 **전역 코사인 유사도** (평균 미차감) | **0.9900** |
| CANConv `eval_rr.py` | 3×3 Laplacian + **밴드별 Pearson 상관** | 0.8779 |
| 논문 | — | 0.988 |

두 배 가까이 차이나는 **서로 다른 지표**였다. 그동안 "SCC 는 정의가 달라 비교 불가" 로
처리했는데, 원인이 이것이었고 DLPan 정의로 계산하면 논문과 맞는다.

흥미롭게도 PAN-Crafter 자체 `utils.py` 의 `SCC_numpy` 는 **DLPan 정의를 정확히 구현하고 있었다.**
`tools/eval_dlpan.py` 를 오늘 그쪽으로 교체했다.

### ④ 우리가 가진 FR 테스트셋이 논문이 쓴 것과 다른 장면이다 (미해결)

**"논문 시점" 이라고 줄여 썼던 것의 정확한 의미**: 우리 손에 있는 full-resolution 테스트
20장이 논문 저자가 평가에 쓴 20장과 **같은 장면이 아니다.**

증명은 **EXP 기준선**으로 한다. EXP 는 모델을 전혀 쓰지 않고 `lms`(보간 입력) 자체를
결과물로 넣어 평가한 값이다. 특히 `D_λ = 1 − Q2n(msexp, MTF(msexp))` 는 **`lms` 에만 의존**하며
모델도, 우리 `.mat` 도, 학습도 개입하지 않는다. 그래서 이 값이 논문과 다르면
원인은 코드가 아니라 **데이터**일 수밖에 없다.

| WV3 EXP (모델 무관) | D_λ↓ | D_s↓ | HQNR↑ |
|---|---:|---:|---:|
| 장면 0–11 | 0.0597 | 0.1461 | 0.8023 |
| **장면 12–19** | **0.0246** | **0.0811** | **0.8965** |
| 전체 0–19 | 0.0456 | 0.1201 | 0.8399 |
| **논문(CANNet) EXP** | **0.0232** | **0.0813** | **0.897** |

**12–19 만 쓰면 논문과 소수점 셋째 자리까지 맞는다.** 0–11 을 넣으면 어긋난다.
장면별로 보면 #1(0.1502), #2(0.1039) 같은 이상치가 앞쪽에 몰려 있다.

센서별로 보면 lpan 결함(⑤)과 같은 센서에서 나타난다.

| 센서 | 배포 `lpan` 과 PAN 의 상관 | EXP D_λ 0–11 / 12–19 |
|---|---:|---:|
| WV3 | 0.011 (어긋남) | 0.0597 / 0.0246 = **2.4배** |
| QB | −0.001 (어긋남) | 0.0672 / 0.0469 = 1.4배 |
| GF2 | **1.000 (일치)** | 0.0151 / 0.0192 = 0.8배 (분할 없음) |

**GF2 만 두 가지가 모두 정상이다.**

### PanCollection 이 FR 테스트셋 개정을 명시하고 있다

[PanCollection README](https://github.com/liangjiandeng/PanCollection) 에 다음이 적혀 있다.

> *Latest Update (Dec. 11, 2022): we updated **full-resolution test examples** that contain
> more different image scenes.*
>
> *Latest Update (Mar. 20, 2023): one testing example in reduce-resolution format for **WV3**
> sensor is not consistent with the one in full-resolution format, we have fixed it.*

즉 **full-resolution 테스트셋이 실제로 교체된 이력이 있고**, WV3 는 reduced/full 정합
문제로 추가 수정까지 받았다. "FR 셋 버전이 다를 수 있다" 는 것은 추측이 아니라 **문서화된 사실**이다.
RR 은 전체 20장으로 논문과 맞고 FR 만 어긋나는 우리 관측과도 부합한다.

다만 **어느 논문이 어느 버전을 썼는지는 확정할 수 없다.** 배포 h5 에 버전 필드가 없고,
PanCollection 은 과거 버전을 따로 내려받을 수 있게 제공하지 않는다.
QB 는 lpan 이 어긋났는데 난이도 분할은 약해(1.4배) 완전히 일관되지도 않는다.

**왜 그냥 전체 20장으로 논문과 비교하면 안 되는가**: 비교 대상이 서로 다른 장면이기 때문이다.
어려운 장면 12개가 우리 쪽에만 들어 있으면 우리 수치가 나쁘게 나오는데,
그건 모델이 나빠서가 아니다. 그래서 `--indices 12-19` 로 **같은 장면끼리** 비교한다.
반대로 **우리 실험끼리 비교할 때는 전체 20장을 쓴다** — 모두 같은 데이터를 보므로 공정하다.

### ⑤ 배포 `lpan` 이 손상돼 있었다 (WV3/QB, 복구)

[KNOWN_ISSUES F-1](../KNOWN_ISSUES.md). full-res HQNR 0.3768 → 0.8446.
PAN-Crafter 전용 문제이며 U-Know 계열에는 영향 없다.

### ⑥ WV2 `lpan` 이 존재하지 않는다 (생성, 검증 불가)

zero-shot 평가에 필요한데 저자 배포본이 없다. 다른 센서 10개 파일에서 역추정한
레시피(`Gaussian σ=1.98, N=41, REPLICATE → [2::4,2::4]`)로 만들었으나
**대조할 원본이 없어 검증할 수 없다.** WV2 수치를 인용할 때 반드시 명시할 것.

### ⑦ Q2n 구현이 까다롭다 (해결됨)

DLPan 공식 파이썬 포트의 `q2n` 은 1 을 초과하는 값을 반환해 쓸 수 없다.
CANConv 가 MATLAB 원본에서 직접 이식한 `tools/q2n.py` 를 쓴다.

---

## 4. 지금 상태에서 할 수 있는 것 / 없는 것

| | 상태 |
|---|---|
| PAN-Crafter Table 1/2 (WV3·QB·GF2 reduced+full) | **전부 산출 가능** |
| PAN-Crafter Table 3 (WV2 zero-shot) | 산출 가능. 단 `lpan` 이 자체 생성물 |
| U-Know Table 2/3 (WV3·QB·GF2) | **전부 산출 가능**, `lpan` 불필요 |
| U-Know Table 4 (Params/Time/Memory) | 측정 가능. **FLOPs 만 미구현** (`fvcore` 추가하면 됨) |
| U-Know Table 5/6 (ablation) | 산출 가능 |
| 논문 절대값과의 1:1 대조 | PSNR·SSIM 은 구현 미명시로 근사만 가능. FR 은 `--indices 12-19` 필요 |

**빠진 데이터는 없다.** 남은 제약은 (a) PSNR/SSIM 의 구현 불확정성, (b) FR 테스트셋 버전 차이,
(c) WV2 `lpan` 미검증 — 셋 다 데이터 확보가 아니라 **문서화로 관리할 사항**이다.

## 5. 오늘 변경

- `tools/eval_dlpan.py`
  - SCC 를 DLPan `SCC.m` 정의로 교체 (0.878 → 0.990, 논문 0.988)
  - SSIM 추가 (0.9754, 논문 0.976)
  - PSNR 을 전 밴드 통합 MSE 기준으로 변경, 밴드별 평균도 참고 출력
  - 비표준 지표(PSNR/SSIM)에 `*` 표기와 경고 문구 추가
- QB full-res `lpan` 은 아직 복구하지 않았다 (`tools/repair_lpan.py --sensor qb` 로 가능)
- FLOPs 측정은 미구현
