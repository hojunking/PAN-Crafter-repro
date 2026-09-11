# 2026-08-19 — 지표·데이터 감사 · WV3 4벌 완결 · v0.3 아키텍처 검토

이 날의 세 문건을 하나로 합쳤다.

| 갈래 | 결론 |
|---|---|
| **지표·데이터 감사** | **두 논문 지표 전부 산출 가능, 데이터 결손 없음.** SCC 정의 오류 교정(0.878 → 0.990, 논문 0.988), SSIM 추가(0.9754 vs 논문 0.976). **PSNR·SSIM 은 DLPan 표준에 없어 구현 불확정** |
| **WV3 4벌 완결** | **재현 성립** — ERGAS 2.164 / HQNR 0.948. A-1/A-2 수정은 in-domain 중립~손해이나 **WV2 zero-shot 에서 HQNR +2.0%**. 선택 편향 +0.14~0.21%. full-res 복구 0.377 → 0.949 |
| **v0.3 아키텍처 검토** | 문서 채널 스펙이 **논문의 1/4**(2.27M vs 9.115M), **OOM 위험 과장**(batch32 에 1.8GB) → alternating 불필요 |

> **후속 정정**: SCC 는 2026-09-07 지표 v2 에서 **`SCC.m` zero-padding 관례**로 다시 교정됐다
> (이 시점 값이 약 +0.004 높다). SSIM 도 Gaussian 11×11 로 교정(약 +0.002 높았다) —
> [2026-09-07](2026-09-07_alignment-shift-robust-and-metric-v2.md).

---

## 지표·데이터셋 현황 감사 (2026-08-19)

> 원문: `2026-08-19_metric-and-dataset-audit.md`

두 논문(`../2505.23367v2.pdf` PAN-Crafter, `../uknowdiff.pdf` U-Know-DiffPAN)이 보고하는
**모든 지표를 낼 수 있는가**, **데이터셋에 빠진 것이 있는가**, **측정이 왜 어려웠는가** 를 정리한다.

**결론: 두 논문의 지표는 전부 산출 가능하고, 데이터도 전부 갖춰져 있다.**
오늘 SCC 정의 오류를 바로잡고 SSIM 을 추가해 마지막 구멍을 메웠다.

---

### 1. 지표 — 전부 가능하다

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

### 2. 데이터셋 — 두 논문이 요구하는 것은 전부 있다

| 센서 | train | valid | test reduced | test full | 밴드 | lpan 상태 |
|---|---:|---:|---:|---:|---:|---|
| WV3 | 9,714 | 1,080 | 20 | 20 | 8 | 배포본 O / **full-res 손상 → 복구본 O** |
| QB | 17,139 | 1,905 | 20 | 20 | 4 | 배포본 O / **full-res 손상 (미복구)** |
| GF2 | 19,809 | 2,201 | 20 | 20 | 4 | 배포본 O (전부 정상) |
| WV2 | **없음** | **없음** | 20 | 20 | 8 | **직접 생성 (검증 불가)** |

- **PAN-Crafter** 요구: WV3/QB/GF2 학습+평가, WV2 zero-shot 평가 → **전부 충족**
  (WV2 train/valid 는 원래 존재하지 않는다. zero-shot 전용이라 결손이 아니다.)
- **U-Know-DiffPAN** 요구: WV3/QB/GF2 학습+평가 → **전부 충족**. WV2 는 쓰지 않는다.

#### `lpan`(`*_pan.h5`)이 필요한 쪽은 PAN-Crafter 뿐이다

| | lpan 필요? | 근거 |
|---|---|---|
| PAN-Crafter | **필요** | 입력 concat(`↑4 lpan`, `pan − ↑4 lpan`)과 CM3A value 경로 |
| U-Know-DiffPAN | **불필요** | 입력이 `[X_t \| I_PAN \| I_MS^LR]` 뿐 (Eq 3, 15) |

**U-Know-DiffPAN 재현에는 `lpan` 결함(4절 ②)이 아예 영향을 주지 않는다.**

#### 두 논문 밖의 자산 (`../CANConv`)

`cas500`(train/valid/test, scale 4095), `vantor`(finetune/valid/test, scale 16383).
붙이려면 feeder 의 `max_pixel` 문자열 추론을 먼저 고쳐야 한다([B-1~B-3](../KNOWN_ISSUES.md)).

---

### 3. 측정이 어려웠던 이유 — 확인된 것 7가지

#### ① MATLAB 런타임이 없다 (해결)

README 는 "All evaluation metrics were measured using the official MATLAB code from the
DLPan-Toolbox" 라고 명시한다. 툴박스 소스는 `../DLPan-Toolbox` 에 있으나
**`matlab`·`octave` 모두 설치돼 있지 않다.** 데이터 문제가 아니라 도구 문제였다.

→ `../CANConv/tools/eval_rr.py` / `eval_fr.py` 의 파이썬 포팅을 재사용해 우회했다.
CANNet 논문 대비 reduced 0.7%, full HQNR 0.2% 이내로 검증된 구현이다.

#### ② 표준 프로토콜에 PSNR·SSIM 이 없다 (구조적 한계)

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

#### ③ 같은 이름, 다른 정의 — SCC (오늘 발견·교정)

| 구현 | 정의 | WV3 값 |
|---|---|---:|
| DLPan `SCC.m` | Sobel 기울기 크기의 **전역 코사인 유사도** (평균 미차감) | **0.9900** |
| CANConv `eval_rr.py` | 3×3 Laplacian + **밴드별 Pearson 상관** | 0.8779 |
| 논문 | — | 0.988 |

두 배 가까이 차이나는 **서로 다른 지표**였다. 그동안 "SCC 는 정의가 달라 비교 불가" 로
처리했는데, 원인이 이것이었고 DLPan 정의로 계산하면 논문과 맞는다.

흥미롭게도 PAN-Crafter 자체 `utils.py` 의 `SCC_numpy` 는 **DLPan 정의를 정확히 구현하고 있었다.**
`tools/eval_dlpan.py` 를 오늘 그쪽으로 교체했다.

#### ④ 우리가 가진 FR 테스트셋이 논문이 쓴 것과 다른 장면이다 (미해결)

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

#### PanCollection 이 FR 테스트셋 개정을 명시하고 있다

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

#### ⑤ 배포 `lpan` 이 손상돼 있었다 (WV3/QB, 복구)

[KNOWN_ISSUES F-1](../KNOWN_ISSUES.md). full-res HQNR 0.3768 → 0.8446.
PAN-Crafter 전용 문제이며 U-Know 계열에는 영향 없다.

#### ⑥ WV2 `lpan` 이 존재하지 않는다 (생성, 검증 불가)

zero-shot 평가에 필요한데 저자 배포본이 없다. 다른 센서 10개 파일에서 역추정한
레시피(`Gaussian σ=1.98, N=41, REPLICATE → [2::4,2::4]`)로 만들었으나
**대조할 원본이 없어 검증할 수 없다.** WV2 수치를 인용할 때 반드시 명시할 것.

#### ⑦ Q2n 구현이 까다롭다 (해결됨)

DLPan 공식 파이썬 포트의 `q2n` 은 1 을 초과하는 값을 반환해 쓸 수 없다.
CANConv 가 MATLAB 원본에서 직접 이식한 `tools/q2n.py` 를 쓴다.

---

### 4. 지금 상태에서 할 수 있는 것 / 없는 것

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

### 5. 오늘 변경

- `tools/eval_dlpan.py`
  - SCC 를 DLPan `SCC.m` 정의로 교체 (0.878 → 0.990, 논문 0.988)
  - SSIM 추가 (0.9754, 논문 0.976)
  - PSNR 을 전 밴드 통합 MSE 기준으로 변경, 밴드별 평균도 참고 출력
  - 비표준 지표(PSNR/SSIM)에 `*` 표기와 경고 문구 추가
- QB full-res `lpan` 은 아직 복구하지 않았다 (`tools/repair_lpan.py --sensor qb` 로 가능)
- FLOPs 측정은 미구현

---

## WV3 재현 완결 · WV2 zero-shot · full-res 복구 (2026-08-19)

> 원문: `2026-08-19_wv3-four-runs-and-zeroshot.md`

[2026-08-14 보고서](2026-08-14_wv3-baseline-vs-fixed.md)의 후속이자 완결편이다.
목적이 재현 확인에서 **새 연구의 baseline 만들기**로 옮겨갔고, 그 과정에서 배포 데이터 결함을
찾아 복구했다. WV3 를 **설정 2종 × 선택기준 2종 = 4벌** 학습했다(각 약 5~6시간, 총 22시간).

| 실행 | 코드 | best 선택 | WV3 ERGAS↓ | WV3 HQNR↑ | WV2 ERGAS↓ | WV2 HQNR↑ |
|---|---|---|---:|---:|---:|---:|
| wv3_baseline | 배포본 | test | 2.1633 | 0.9486 | 4.3162 | 0.9125 |
| wv3_baseline_valsel | 배포본 | **val** | 2.1643 | 0.9475 | 4.3251 | 0.9155 |
| wv3_fixed | A-1/A-2 수정 | test | 2.1765 | 0.9496 | 4.3364 | 0.9305 |
| wv3_fixed_valsel | A-1/A-2 수정 | **val** | 2.1804 | 0.9508 | 4.3362 | 0.9345 |
| **논문** | | | **2.040** | **0.958** | **4.169** | **0.942** |
| lms 보간 입력 (기준선) | | | 7.1220 | 0.8964 | 6.7679 | 0.9042 |

> **수치 신뢰도**
> 1. MATLAB 이 없어 `../CANConv` 의 DLPan-Toolbox 파이썬 포팅을 재사용했다
>    (`tools/eval_dlpan.py`, `tools/eval_dlpan_fr.py`). CANNet 논문 대비 reduced 0.7% /
>    full HQNR 0.2% 이내 검증본이며, EXP 기준선이 CANConv 측정치와 0.0001 이내로 일치한다.
>    **PSNR·SCC 는 정의가 달라 논문과 비교하지 말 것.**
> 2. WV3 HQNR 은 논문 대조가 가능한 **12–19번 8장** 부분집합, WV2 는 전체 20장이다.
> 3. **설정당 2벌은 완전한 시드 반복이 아니다.** 초기화와 epoch 5까지는 동일하고,
>    검증 dataloader 의 RNG 소비 때문에 epoch 10 부터 데이터 순서가 갈라진다.
>    "우연한 데이터 순서" 는 배제되지만 초기화 다양성은 포함하지 않는다.
> 4. WV2 의 `lpan` 은 저자 배포본이 없어 직접 생성했다. **대조 검증 불가.**

---

### 1. 재현 판정 — reduced 성립, full-res 성립

| | 우리 (baseline 2벌 평균) | 논문 | 차이 |
|---|---:|---:|---|
| WV3 ERGAS | 2.1638 | 2.040 | +6.1% |
| WV3 SAM | 2.911 | 2.787 | +4.4% |
| WV3 Q8 | 0.9165 | 0.922 | −0.6% |
| WV3 D_s | 0.0280 | 0.027 | +3.7% |
| WV3 HQNR | 0.9481 | 0.958 | −1.0% |
| WV2 ERGAS (zero-shot) | 4.3207 | 4.169 | +3.6% |
| WV2 HQNR (zero-shot) | 0.9140 | 0.942 | −3.0% |

Q8 과 D_s 는 사실상 일치하고 나머지가 1~6% 뒤진다. 시드 다양성이 없고 학습이 50k
iteration 1회라는 점을 감안하면 **재현은 성립한다**고 본다.

![](assets/qual_wv3_reduced.png)
*입력 LRMS(2열)의 뭉개짐이 출력(3·4열)에서 사라지고 GT(5열)와 육안 구분이 어렵다.*

### 2. 핵심 발견 — A-1/A-2 수정은 in-domain 과 zero-shot 에서 방향이 반대다

![](assets/tradeoff_indomain_vs_zeroshot.png)
*점이 개별 실행(설정당 2벌), 막대는 평균.*

| | baseline (2벌) | fixed (2벌) | 차이 |
|---|---:|---:|---|
| WV3 ERGAS↓ | 2.1633 / 2.1643 | 2.1765 / 2.1804 | **fixed 가 +0.68% 나쁨** |
| WV3 Q2n↑ | 0.9165 / 0.9165 | 0.9177 / 0.9178 | fixed 가 +0.13% 좋음 |
| WV3 HQNR↑ | 0.9486 / 0.9475 | 0.9496 / 0.9508 | fixed 가 +0.23% 좋음 |
| **WV2 HQNR↑** | 0.9125 / 0.9155 | 0.9305 / 0.9345 | **fixed 가 +2.0% 좋음** |
| **WV2 D_s↓** | 0.0322 / 0.0313 | 0.0246 / 0.0251 | **fixed 가 −21% 좋음** |

WV2 의 격차는 설정 내 편차(0.003~0.004)의 **약 5배**이고 2벌 모두에서 같은 방향이다.
반면 WV3 ERGAS 는 fixed 가 일관되게 나쁘다(역시 2벌 모두).

**해석 (가설)**: A-1 수정은 CM3A 의 PAN key 경로를 논문 Eq (10)/(11) 대로 되살리고
죽어 있던 파라미터 6.13% 를 학습에 참여시킨다. A-2 는 LocalAttn 을 실제 k×k 이웃 gather 로
고정한다. 이 둘이 **학습 도메인에 대한 과적합을 줄이고 모달리티 정합 자체를 더 일반적인
형태로 학습하게 만든다**는 설명이 관측과 부합한다. 즉 배포본의 버그는 WV3 안에서는
손해가 없고(오히려 ERGAS 는 유리하고) 도메인이 바뀌면 대가를 치른다.

**한계**: 초기화 다양성이 없는 2벌이고, WV2 `lpan` 이 검증 불가한 생성물이다.
**결론으로 쓰려면 시드 3개 이상과 QB/GF2 확인이 필요하다.**

### 3. 검증셋 선택 — 편향은 실재하나 무시할 수준

배포본은 검증셋을 로드만 하고 쓰지 않으며 best 를 테스트셋 49회 평가 중 최고로 고른다
([KNOWN_ISSUES E-1](../KNOWN_ISSUES.md)).

| 실행 | val 선택 epoch | 그때의 test ERGAS | test 선택 최고 | 편향 | corr(val,test) |
|---|---:|---:|---:|---:|---:|
| baseline_valsel | 240 | 2.1538 | 2.1507 @215 | +0.14% | 0.9927 |
| fixed_valsel | 240 | 2.1670 | 2.1625 @215 | +0.21% | 0.9934 |

**검증셋이 테스트셋의 거의 완벽한 대리지표(corr 0.993)** 라 어느 쪽으로 골라도 사실상 같은
체크포인트에 도달한다. 배포본의 선택 방식은 **방법론적으로 잘못됐지만 이 데이터셋에서는
수치를 거의 부풀리지 않았다.** 수렴 후 곡선이 평탄하기 때문이다.

그래도 `select_on: val` 을 새 연구의 기본값으로 둔다. 편향이 작다는 것은 사후 측정된 사실이지
설계 근거가 아니며, 곡선이 평탄하지 않을 설정에서는 크기가 달라진다.

![](assets/curve_wv3.png)
*우측 2칸(full-res)은 손상된 `lpan` 으로 측정된 값이라 무의미하다 — 4절 참고.*

### 4. 배포 데이터 결함과 복구

`pan_h5.zip` 의 WV3·QB full-res `lpan` 이 짝이 되는 PAN 과 무관한 다른 장면이다
(상관 0.011 / −0.001, GF2 만 1.000). [KNOWN_ISSUES F-1](../KNOWN_ISSUES.md).

![](assets/lpan_mismatch.png)

| WV3 full-res (20장) | D_λ↓ | D_s↓ | HQNR↑ |
|---|---:|---:|---:|
| 손상된 배포본 `lpan` | 0.5489 | 0.1645 | **0.3768** |
| 재생성한 `lpan` | 0.0849 | 0.0827 | **0.8446** |

복구 레시피: `Gaussian(sigma=1.98, N=41, REPLICATE)` → `[2::4, 2::4]`.
배포 12개 파일 중 10개를 RMSE 0.23~0.53(값 범위 0~2047)으로 재현하고, 어긋나는 2개가
정확히 문제의 파일이다. `tools/repair_lpan.py`, 원본은 보존한다.

### 5. 배제한 가설

| 가설 | 검증 | 결과 |
|---|---|---|
| full-res 가 나쁜 건 모델·평가기 문제 | `lpan` 재생성 후 재측정 | 기각. 데이터 결함 (0.377 → 0.949) |
| full-res `lpan` 이 순서만 뒤바뀜 | 20×20 교차상관 + 타 센서 대조 | 기각. 맞는 원본이 없음 |
| 테스트셋 선택이 수치를 크게 부풀린다 | val 선택 2벌과 대조 | 기각. +0.14~0.21% |
| A-1/A-2 수정 효과는 전부 노이즈 | 설정당 2벌, in/out domain 분리 측정 | **부분 기각.** WV2 에서 편차의 5배 |
| MATLAB 없이는 논문 지표 측정 불가 | CANConv 파이썬 포팅 재사용 | 기각. EXP 기준선 0.0001 이내 일치 |
| WV2 lpan 을 못 만들어 zero-shot 불가 | 세 센서 배포본에서 필터 역추정 | 기각 |

### 6. 확정된 설정

| 항목 | 값 | 근거 |
|---|---|---|
| iteration / batch / optimizer | 50,000 / 48(실효 96) / AdamW 1e-4 wd 0.01 cosine warmup 100 | 논문 Sec 4.2 |
| seed | 2025 | 논문 명시 |
| `save_epoch` | 25 | 500 은 총 epoch(248) 안에서 발동하지 않는다 |
| `select_on` | **val** (새 연구 기본값) | 3절 |
| `lpan` 재생성 | Gaussian σ=1.98, N=41, REPLICATE, `[2::4,2::4]` | 배포본 10/12 파일 RMSE ≤0.53 |
| 평가 | `tools/eval_dlpan.py` / `eval_dlpan_fr.py` | CANConv 포팅, 검증본 |
| 학습 1회 | 4h55m ~ 6h (RTX 4090, peak 16.5GB) | 실측 |

### 7. 남은 것

1. **시드 3개** — 2절의 in/out domain 역전을 결론으로 쓰려면 필요하다. 실행당 5~6시간.
2. **QB / GF2** — QB 는 `repair_lpan.py --sensor qb` 선행. 설정당 2벌이면 각 12시간.
3. **WV2 `lpan` 검증** — 저자 배포본을 구하거나 다른 경로로 대조.
4. **CAS500 / Vantor** — feeder 의 `max_pixel` 문자열 추론 수정 선행
   ([B-1~B-3](../KNOWN_ISSUES.md)).
5. 신규 연구 방향은 [별도 검토 문서](2026-08-18_review_variance-regularized-mutual-overfitting.md) 참고.

---

## 검토 — method concept v0.3 (아키텍처 · Student-first · OOM) (2026-08-19)

> 원문: `2026-08-19_review_v0.3_architecture_student_first.md`

대상: [`research_log/method_concept_v0.3_architecture_training_oom_student_first.md`](../research_log/method_concept_v0.3_architecture_training_oom_student_first.md)
기준 논문: `../uknowdiff.pdf` (U-Know-DiffPAN)

문서의 추정 대신 **FSA-S 규격 diffusion student 를 실제로 구현해 측정**했다.
결론부터: **문서의 가장 중요한 개념 교정은 옳고, 아키텍처 규격과 OOM 분석은 틀렸다.**

---

### 1. 문서에서 옳은 것

| 주장 | 검증 |
|---|---|
| 원 FSA-S 는 일반 CNN 이 아니라 **diffusion student** 다 | **맞다.** 논문 Eq (15): `X̃₀ = ψ([X_t \| I_PAN \| I_MS^LR]; t)` — `X_t` 와 `t` 를 입력으로 받는다. 추론은 DDIM 25-step |
| Table 4 규모 (FSA-T 25.492M/5.910GB, FSA-S 9.115M/2.136GB) | 논문 Table 4 와 일치. "이 값은 joint training backward peak 가 아니다" 는 지적도 정확 |
| peer 는 `no_grad`/`detach`, mutual gate 에도 stop-gradient | 옳은 원칙 |
| Student-first 개발 순서를 권장하되 warm-start 와 from-scratch 를 구분 | 옳다. 포지셔닝 리스크를 정확히 짚었다 |
| 두 모델에 같은 `X_t`, `t`, `ε` 를 쓴다 | 필수다. 다른 timestep 이면 uncertainty·error 비교가 성립하지 않는다 |

Part I 의 "diffusion 과 CNN 은 배타적 분류가 아니다" 는 교정이 이 문서의 가장 큰 기여다.
이걸 놓쳤으면 원 논문과 비교 불가능한 모델을 만들 뻔했다.

### 2. 틀린 것 ① — 아키텍처 규격이 논문의 1/4 이다

문서 §4 는 `inner_channel=32, channel_mults=(1,2,2,4)` → 채널 `[32,64,64,128]` 을 baseline 으로 제시한다.
그 규격으로 ResBlock U-Net(depth 2, 8×8 self-attention, mean+logvar head)을 구현해 파라미터를 셌다.

| 채널 | depth | 파라미터 | 논문 대비 |
|---|---:|---:|---|
| **[32, 64, 64, 128] (문서 스펙)** | 2 | **2.266 M** | FSA-S 9.115M 의 **25%** |
| [32, 64, 64, 128] | 3 | 2.953 M | 32% |
| [48, 96, 96, 192] | 2 | 4.934 M | 54% |
| **[64, 128, 128, 256]** | 2 | **8.636 M** | **95% — 여기가 맞다** |
| [64, 128, 128, 256] | 3 | 11.264 M | 124% |
| [96, 192, 192, 384] | 2 | 19.143 M | (FSA-T 25.5M 의 75%) |

문서 스펙대로 만들면 **논문 FSA-S 의 1/4 짜리 모델**이 된다. `inner_channel=32` 라는
"공개 구현" 근거는 이 저장소에서 확인할 수 없고, 파라미터 수와도 맞지 않는다.

> **수정**: Student 는 `[64,128,128,256]`, depth 2 로 시작한다.
> Teacher 는 `[96,192,192,384]` 계열 + FFA/HQFE 로 25M 대를 맞춘다.

### 3. 틀린 것 ② — OOM 위험이 크게 과장돼 있다

같은 구현으로 RTX 4090 에서 학습 1 step(forward+backward+AdamW step) 을 실측했다.
64×64 patch, FP32, WV3 8밴드.

| 배치 | peak VRAM | s/iter | 300K iteration 환산 |
|---:|---:|---:|---:|
| 8 | 0.27 GB | 0.026 | 2.2 h |
| 32 | 0.90 GB | 0.027 | 2.3 h |
| 128 | 3.37 GB | 0.048 | 4.0 h |

논문 규격(`[64,128,128,256]`)에서도 **배치 32 에 1.81 GB, 0.029 s/iter, 300K ≈ 2.4시간**이다.

문서 §18 의 `Student-only batch 8부터 시작` 은 **약 30배 보수적**이다.
24 GB 면 배치 256 이상도 들어간다. 비교 기준으로, 이 저장소의 PAN-Crafter 는
실효 배치 96 에서 **16.5 GB** 를 쓴다 — CM3A 가 key/value 를 k²=9 배로 펼치기 때문이다.
plain ResBlock U-Net 은 그런 항이 없다.

#### 그래서 alternating update 는 근거를 잃는다

문서 §19 는 alternating one-stage update 를 **main implementation 으로 권장**하며
그 이유를 "24 GB 에서 두 backward graph 를 동시에 유지할 수 없다" 로 든다.
측정치는 그 전제를 지지하지 않는다. Student 9M + Teacher 25M 을 동시에 학습해도
plain 부분만 보면 배치 32 에서 5 GB 안팎이다.

alternating 은 공짜가 아니다.

- forward 계산이 2배 (문서도 인정)
- peer target 이 **반 스텝 stale** — mutual 신호의 의미가 흐려진다
- 구현·디버깅 복잡도 증가

> **수정**: strict simultaneous update 로 시작한다. **실제로 OOM 이 나면 그때** alternating 으로 간다.
> 선제적으로 만들지 말 것.

**단, 측정하지 않은 부분이 있다.** Teacher 의 FFA/FTCA/SWTCA 는 FFT·SWT·cross-attention
중간 텐서를 만든다. 이건 plain ResBlock 이 아니므로 위 수치로 외삽할 수 없다.
**Teacher-only 프로파일이 alternating 채택 여부를 결정하는 유일한 근거**이며,
문서 §31 의 3~4번(Teacher-only profile → FSA-T-Lite)을 **alternating 설계보다 먼저** 해야 한다.

### 4. Student-first 계획의 핵심 결함 — 비교 기준이 잘못됐다

사용자 계획은 "Student 먼저 학습 → 논문 기준으로 성능 평가 → 아키텍처 타당성 판단" 이다.
방향은 옳지만 **무엇과 비교할지가 문제다.**

논문 Table 3 의 FSA-S (WV3 reduced ERGAS **2.046**, PSNR 37.930, Q8 0.922) 는
**teacher KD 로 학습된 값**이다. Teacher 없이 학습한 student 와 직접 비교하면 불공정하고,
"내 구현이 논문보다 나쁘다" 는 잘못된 결론에 이른다.

**논문이 student-only 를 보고한 곳은 단 하나다** — Table 6, `L₁` 행.

| FSA-S 학습 손실 | D_λ↓ | D_s↓ | HQNR↑ | 설명 |
|---|---:|---:|---:|---|
| `L₁` | 0.026 | 0.040 | **0.935** | **teacher 없음 = student-only** |
| `L_KD` | 0.025 | 0.038 | 0.938 | uncertainty 없는 일반 KD |
| `L_U-know` | 0.018 | 0.037 | **0.944** | 논문 최종 |

**GF2 full-resolution** 이다. 따라서:

> **첫 검증 타깃 = GF2 full-resolution HQNR 0.935**

부수적 이점이 크다. GF2 는 배포 `lpan` 이 **유일하게 멀쩡한 센서**다
([KNOWN_ISSUES F-1](../KNOWN_ISSUES.md)) — 복구 절차 없이 바로 쓸 수 있다.

KD 의 기여가 HQNR 0.935 → 0.944, 즉 **+0.9%** 라는 점도 기억할 것.
mutual learning 이 넘어야 할 문턱이 이 정도라는 뜻이다.
[8월 18일 검토](2026-08-18_review_variance-regularized-mutual-overfitting.md)에서 측정한
오라클 상한(+7.9%)과 함께 보면, 현실적 목표 구간은 **1~3%** 다.

### 5. 전략적 질문 — diffusion 이 정말 필요한가

| | 파라미터 | WV3 ERGAS↓ | Q8↑ | 추론 시간 |
|---|---:|---:|---:|---:|
| U-Know FSA-S (diffusion, DDIM 25-step) | 9.115 M | 2.046 | 0.922 | **12.287 s** |
| PAN-Crafter (deterministic, 1-pass) | 9.969 M | 2.040 | 0.922 | **0.009 s** |

**정확도가 사실상 동일한데 추론이 1365배 차이난다.**
(두 논문 모두 RTX 3090 · 256×256×8 기준이나, 측정 조건이 완전히 같다는 보장은 없다.)

제안 방법의 기여 — variance-guided mutual overfitting — 은 **diffusion 과 직교한다.**
uncertainty head, GT-detail-variance 가중, reliability-gated mutual 은 deterministic
regressor 에서 그대로 성립한다. diffusion 을 택하면 추가로 치르는 비용은:

- 학습 300K iteration (PAN-Crafter 는 50K)
- 추론 25 forward
- q_sample / DDIM sampler / timestep embedding 구현

얻는 것은 "원 논문과 같은 프레임워크" 라는 비교 가능성뿐이다. 그건 **공개 수치와 비교**해도 된다.

게다가 이 저장소에는 **PAN-Crafter 재현본이 이미 있다** — baseline/fixed 4벌,
검증셋 선택, DLPan 평가 도구, `lpan` 복구까지. deterministic 위에 올리면 그 자산을 그대로 쓴다.

> **권고**: Student-first 실험을 **두 갈래로 동시에** 돌려 결정한다.
> 둘 다 2~3시간이면 끝난다.
>
> - (a) FSA-S-UQ (diffusion, `[64,128,128,256]`) — GF2 student-only
> - (b) PAN-Crafter + variance head (deterministic) — GF2 student-only
>
> (b) 가 (a) 에 근접하면 **diffusion 을 버리고 deterministic 으로 간다.**
> 논문 포지셔닝은 "U-Know 의 uncertainty 설계를 deterministic PS 로 이식하고 확장" 이 된다.

### 6. 그 밖의 지적

- **§13 reliability gate 는 §7(v0.1) 대비 개선이다.** `g = σ((E_S−E_T)/τ_e)·σ((s_S−s_T)/τ_s)` 는
  uncertainty 뿐 아니라 **실제 local error** 를 함께 본다. v0.1 의 `(1−θ̄_T)θ̄_S` 보다 낫다.
  다만 `E_m` 은 GT 를 쓰므로 **학습 시에만 가능**하다 — 문서에 명시할 것.
- **1-channel log-variance 권장은 타당하다.** 자유도를 줄여 "error 를 variance 로 우회" 하는
  현상을 억제한다는 근거가 옳다. 원 FSA-T 는 밴드별이므로 ablation 유지도 맞다.
- **§30 의 검증 지표(`P(E_T<E_S)`, oracle, routing accuracy)는 정확히 필요한 것들이다.**
  [8월 18일 검토](2026-08-18_review_variance-regularized-mutual-overfitting.md)에서 PAN-Crafter vs CANNet 으로
  이미 재어 봤고 결과는 부정적이었다(오차 상관 0.94, 승률 47:53, 오라클 +7.9%).
  **Teacher–Student 쌍에서 다시 재야 한다** — 같은 데이터·같은 `X_t` 를 공유하므로
  상관이 더 높을 가능성이 크다.
- **§0-6 의 warm-start 구분은 지키되**, Student-only 체크포인트는 어차피 GF2 검증용이므로
  main joint training 은 random init 으로 시작하면 된다. 충돌 없다.
- Part VII(deterministic hybrid)을 "정말 원하면" 수준의 부록으로 둔 것은 5절 관점에서 재고 대상이다.

### 7. 수정된 실행 순서

```text
P0. Teacher-only 프로파일  ← alternating 설계보다 먼저
    full FSA-T 의 FFA/FTCA/SWTCA peak VRAM 측정.
    여기서만 OOM 여부가 결정된다. (반나절)

P1. Student-only, GF2, 두 갈래 동시            ← 사용자 계획, 규격만 수정
    (a) FSA-S-UQ diffusion [64,128,128,256]
    (b) PAN-Crafter + variance head
    타깃: GF2 full-res HQNR 0.935 (논문 Table 6 L1 행)
    각 2~3시간

P2. P1 결과로 backbone 확정 (diffusion vs deterministic)

P3. 쌍의 상보성 측정  ← 여기가 진짜 go/no-go
    P(E_T<E_S), oracle gain, routing accuracy.
    오라클 +15% 를 못 넘기면 mutual 축을 접고
    단일 모델 uncertainty 축으로 논문을 재구성한다.

P4. strict simultaneous joint (OOM 나면 그때 alternating)

P5. feature mutual (8×8 부터)
```

### 8. 요약

| | |
|---|---|
| **살릴 것** | FSA-S = diffusion student 라는 교정, peer detach 원칙, Student-first 순서, warm-start 구분, reliability gate(§13), 1-ch log-variance |
| **고칠 것** | 채널 `[32,64,64,128]` → `[64,128,128,256]` (논문 규모의 1/4 이었다) |
| **뺄 것** | Part VI 의 OOM 대책 대부분. alternating update 를 main 으로 선택하는 것 |
| **바꿀 것** | 비교 타깃을 Table 3 FSA-S(2.046, KD 포함) → **Table 6 L₁ 행 GF2 HQNR 0.935**(student-only) |
| **결정할 것** | diffusion 을 쓸 것인가. 정확도 동일에 추론 1365배 차이다 |
