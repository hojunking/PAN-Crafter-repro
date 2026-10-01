# PANDA ablation 결과 분석 및 추가 표 후보
작성일: 2026-10-01  
기준: 제출본 고정. 본 문서는 추가 근거 검토용이며 기존 Table 3 또는 paper/ablations 시트를 교체하는 지시가 아니다.  
대상 Sheet: pan-cvpr27 / WV3-ablations / GF2-ablations / B01 원본·stress.

## 1. 먼저 구분할 실제 분석 범위

| 묶음 | 확인한 서버·센서 | 측정량 | 반복 단위 |
|---|---|---:|---|
| ABLR2 component | s1 / WV3 | 고유 93 run = Student83 + Teacher10 | P01–P05. case별 3–5개 |
| ABLR2 component | s3 / GF2 | 고유 100 run = Student90 + Teacher10 | P01–P05. case별 5개 |
| B01 q cue substitution | s1 / WV3 | Student8, native16관측 | 고정 F1에서 case당 Student seed2개 |
| B01 stress | s1 / WV3 | 16곡선, 784 shift point | 각 Student의 A_ON/A_ZERO paired inference |
| B01 s3·s5 | 아직 수집된 측정값 없음 | 원본 탭은 header만 존재 | 파일 미제공/실행 상태 UNKNOWN |

**s3의 GF2 component 실험은 확인되지만, s3의 신규 WV3 B01 결과는 이번 live readback에서 확인되지 않았다.**
따라서 B01을 s1+s3 네 반복 또는 전체 여섯 반복으로 제시하지 않는다.
등록 부재가 서버에서의 미실행을 의미하지는 않는다.

계산은 2026-10-01에 live Sheet를 export한 스냅샷에서 수행했다.
원문 source/row/Result_ID는 각 CSV에 보존했다. 원 서버의 모델이나 raw 결과의 file hash를 이번에 직접 재검증한 것은 아니다.
기존 Sheet·서버·학습 queue는 수정하지 않았다.

## 2. 집계 규칙

- ABLR2: 기존 계획의 primary인 `ABLR2 RR_VAL_SELECTED`만 사용한다. `ABLR2 EXACT50K`는 방향성 교차검사.
- B01: `EXACT_50000` primary, `RR_VAL_ERGAS_MIN` secondary.
- SOURCE_SUMMARY, RAW_MAX, E_MIN, 선택점 alias를 추가 독립 run으로 세지 않는다.
- Component 대조는 동일 sensor/server/P block의 두 case를 짝지어 비교한다. 서로 다른 n의 전체 평균을 그대로 빼지 않는다.
- B01은 동일 server/repeat/seed의 variant−QFULL. 고정 F1 조건부 Student 반복이다.
- 평균 ± 표준편차는 sample SD(ddof=1). 관측 장면 20개·방향 8개·selector2개를 seed 수에 합산하지 않는다.
- 여러 sensor/Teacher/runtime/cohort 결과는 별개로 다룬다. s1/WV3와 s3/GF2로 서버 효과를 추정하지 않는다.
- n=2/3/4/5에서 방향 일치 횟수를 보되 통계적 유의성을 확정하지 않는다.

## 3. 핵심 판단

1. **LPAN/HPAN frequency inputs의 기여가 가장 일관되다.** C01−C00은 WV3·GF2 각각 5/5에서 HQNR와 ERGAS가 함께 개선된다. FULL C07−C08에서도 RR 이득이 각각 5/5다.
2. **GF2의 pretrained Aligner/Student alignment는 RR 복원에 도움이 되지만 FR 이득은 동일하지 않다.** Native 정합의 필요성을 모든 센서·모든 지표에 일반화할 근거는 부족하다.
3. **reliability와 fitting의 핵심 주장에는 아직 반례·상충 결과가 있다.** 특히 adaptive fitting C05−C04, hard emphasis C07−C12, q vs mean C07−C15.
4. **B01에서 q는 constant mean보다 낫지만, e-surrogate보다 필요하다는 결론은 나오지 않는다.** 현재 두 seed는 preliminary evidence다.
5. **RB02는 correction 사용의 작은 이득을 보여주지만 robust shift cancellation 증거는 약하다.** 추가 shift가 커지면 reconstruction이 크게 악화되며 radius4의 coverage 경고도 남아 있다.

## 4. 추가 표 후보 A — Frequency inputs and alignment initialization

목적: frequency input과 Student Aligner 초기화의 역할을 분리한다.
기존 Table 3의 교체가 아니라, 구체적인 초기화 대조를 보여주는 추가 표다.
C03은 “consistency loss만 추가”가 아니라 TPLUS에서 pretrained A를 복사한 조건이다.
C17은 같은 종류의 Student에 TZERO의 A를 초기화한 조건이다.

| Case | n / sensor | WV3 HQNR↑ | WV3 ERGAS↓ | GF2 HQNR↑ | GF2 ERGAS↓ |
|---|---|---|---|---|---|
| C00 PAN+MS baseline | 5 | 0.95109 ± 0.00108 | 2.05336 ± 0.00136 | 0.93343 ± 0.00228 | 0.59922 ± 0.00282 |
| C01 + LPAN/HPAN | 5 | 0.95370 ± 0.00169 | 2.04583 ± 0.00498 | 0.95536 ± 0.00285 | 0.57367 ± 0.00198 |
| C02 + scratch Aligner | 5 | 0.95420 ± 0.00218 | 2.04741 ± 0.00459 | 0.95371 ± 0.00451 | 0.57053 ± 0.00265 |
| C03 + TPLUS Aligner initialization | 5 | 0.95549 ± 0.00109 | 2.04729 ± 0.00251 | 0.94970 ± 0.00456 | 0.55783 ± 0.00220 |
| C17 TZERO Aligner initialization | 5 | 0.95503 ± 0.00305 | 2.04674 ± 0.00137 | 0.95142 ± 0.00591 | 0.55880 ± 0.00095 |

### 해석

- C01−C00: WV3 ΔHQNR=+0.002614, ΔERGAS=−0.007524; GF2 +0.021933 / −0.025553. 네 축 모두 각 센서 5/5 방향 일치.
- WV3에서는 이때 Ds가 약 +0.003101 악화되고 Dλ는 약 −0.005816 개선된다. “frequency inputs가 모든 spatial/spectral 지표를 개선”한다고 쓰지는 않는다.
- C03−C02: GF2 ERGAS가 −0.012700, 5/5 개선되나 HQNR는 −0.004008. WV3는 평균 HQNR +0.001285, RR 변화는 작다.
- C03−C17: WV3 ΔHQNR +0.000460, ΔERGAS +0.000553; GF2 −0.001718 / −0.000973. consistency prior의 native 성능 이득은 일관적이지 않다.
- C17은 그림·표에서 C03 뒤에 순차적으로 추가한 단계가 아니라 **초기화의 대체 조건**으로 배치한다.

**판정:** frequency inputs 표는 가장 준비도가 높다. Alignment initialization은 센서별 RR/FR 차이를 설명하는 표로는 유효하나, 보편적 성능 향상 증거로는 제한적이다.

## 5. 추가 표 후보 B — Leave-one-component-out and matched contrasts

아래는 **FULL C07−제거/대체군**의 같은 block 내 차이 평균이다.
HQNR는 양수, ERGAS는 음수가 FULL에 유리하다. 표준편차·개별 block·승패 수는 CSV에 있다.
이 표의 n은 metric 비교에 실제 들어온 paired block 수다.

| 대조: 왼쪽−오른쪽 | WV3 n | ΔHQNR↑ | ΔERGAS↓ | GF2 n | ΔHQNR↑ | ΔERGAS↓ |
|---|---|---|---|---|---|---|
| C07-C08: LP/HP 사용−제거 | 5 | +0.002613 | -0.003317 | 5 | +0.011349 | -0.022036 |
| C07-C09: Student 정합 사용−학습부터 제거 | 5 | +0.000037 | +0.000506 | 5 | -0.001769 | -0.017055 |
| C07-C11: 정상 reference−consistency 없는 reference | 4 | -0.001904 | +0.000418 | 5 | +0.001607 | -0.000815 |
| C07-C12: hard emphasis 사용−α=0 | 4 | +0.000191 | +0.003645 | 5 | -0.002462 | +0.000897 |
| C07-C13: soft KD 사용−β=0 | 4 | -0.000090 | -0.001740 | 5 | +0.000612 | +0.000368 |
| C07-C14: GT edge 사용−edge=0 | 3 | +0.002353 | -0.002086 | 5 | -0.002746 | -0.000605 |
| C07-C15: sample q−train mean weight | 4 | -0.001152 | -0.001373 | 5 | -0.001370 | +0.000837 |
| C07-C16: A refinement−frozen A | 5 | -0.000298 | -0.000235 | 5 | -0.000676 | -0.000327 |

### 강한 근거와 약한 근거

**Frequency input:** C07−C08의 ERGAS는 WV3/GF2 모두 5/5 개선이다. baseline 계열과 FULL 계열에서 같은 방향을 보인다.

**Student alignment:** C07−C09는 GF2 ERGAS −0.017055, 5/5 개선이다. 그러나 HQNR는 평균 −0.001769이며 WV3 native 차이는 매우 작다.
이는 GF2 RR에서의 정합 기여를 뒷받침하지만, early alignment가 late feature alignment보다 우수하다는 대조는 아니다.

**Teacher consistency:** C07−C11은 WV3에서 HQNR −0.001904, GF2에서 +0.001607이다.
TPLUS−TZERO Teacher 자체도 WV3 HQNR 평균 −0.001032(0/5 개선), GF2 +0.004253(4/5 개선)로 센서 의존적이다.
C11은 reference bundle 전체를 바꾸므로 q, prediction, initialization 효과를 모두 포함한다.

**Hard emphasis:** C07−C12의 ERGAS는 WV3 +0.003645(0/4 개선), GF2 +0.000897(1/5 개선).
또한 C05−C04의 ERGAS는 WV3 +0.006551, GF2 +0.004962로 각각 5/5 악화한다.
Hard weighting·soft trust·advantage를 동시에 바꾼 C05만으로 원인을 특정할 수는 없지만,
C12까지 함께 보면 hard emphasis의 효용과 최적화 효과를 우선 분해할 이유가 있다.

**Soft KD:** C07−C13은 WV3 RR에서 평균 개선하지만 GF2에서는 개선이 일정하지 않다.
따라서 “KD 자체가 필요 없다”는 결론도, “selective KD가 항상 우수하다”는 결론도 현재 자료로 확정할 수 없다.

**GT edge:** WV3에서 C07−C14는 HQNR/ERGAS 모두 3/3 개선이다.
GF2에서는 HQNR가 −0.002746이고 ERGAS 개선도 2/5에 불과하다.
WV3의 누락된 C14 두 반복부터 회수해야 한다.

**Sample-wise q:** 기존 ABLR2의 C07−C15는 WV3/GF2 모두 평균 HQNR가 떨어진다.
WV3는 1/4, GF2는 2/5만 HQNR 개선이다. B01의 q>mean 결과와 다른 reference/seed 묶음이다.
이를 합쳐 큰 n으로 만들거나 유리한 묶음만 q의 일반적 근거로 제시하지 않는다.

**Student A refinement:** C07−C16의 평균 RR 차이는 작고 HQNR는 두 센서 모두 소폭 낮다.
현재 native 결과만으로 frozen A보다 업데이트가 반드시 필요하다고 주장하기 어렵다.

이 주요 방향은 EXACT50K 교차검사에서도 대체로 유지된다.
따라서 validation selector만 바꾸면 핵심 결론이 전부 반전되는 양상은 아니다.

**판정:** 논문 검토용으로 가장 정보량이 높은 표다. 다만 모든 구성요소의 일관적 이득을 보여주는 표가 아니라,
어떤 항이 어디서 유효하고 어디서 절충/불확실성이 있는지 보여주는 표다.

## 6. 추가 표 후보 C — Choice of the geometry-supervision weight

제출본 3.3절, Fig.5, Eq.(10)–(12)의 “별도 geometric reliability cue”에 직접 대응한다.
모든 조건에서 reconstruction-error 기반 hard/soft 설정은 유지하고,
edge와 A-hard에 들어가는 geometry weight만 바꾼다.

| 조건 | weight | 질문 |
|---|---|---|
| QFULL | 원 sample의 q에서 얻은 reliability | 기준 |
| QMEAN | train-view 전체 평균 weight | 전체 감독 강도만으로 충분한가? |
| QSHUF | e strata/view를 맞추고 sample 연결을 섞은 q weight | sample과 q의 대응이 필요한가? |
| QESUR | e 기반 순위에 원 q weight의 분포를 대응 | 별도 q 대신 reconstruction error로 충분한가? |

### B01 native EXACT50K — s1, case당 두 seed

| Case | n | HQNR↑ mean ± SD | Ds↓ | Dλ↓ | ERGAS↓ mean ± SD | PSNR↑ |
|---|---|---|---|---|---|---|
| QFULL | 2 | 0.95363 ± 0.00427 | 0.03074 | 0.01613 | 2.04913 ± 0.00251 | 37.97134 |
| QMEAN | 2 | 0.95102 ± 0.00217 | 0.03290 | 0.01664 | 2.05069 ± 0.00195 | 37.96878 |
| QSHUF | 2 | 0.95280 ± 0.00222 | 0.03080 | 0.01693 | 2.04809 ± 0.00235 | 37.97578 |
| QESUR | 2 | 0.95474 ± 0.00270 | 0.02950 | 0.01625 | 2.05005 ± 0.00068 | 37.97102 |

QFULL 대비 variant의 paired 차이:
| 대조 | ΔHQNR | ΔERGAS | 해석 |
|---|---:|---:|---|
| QMEAN−QFULL | −0.002611 | +0.001565 | QFULL이 두 seed 모두 HQNR·ERGAS 우세 |
| QSHUF−QFULL | −0.000828 | −0.001040 | 평균 HQNR는 FULL 우세, RR은 shuffled가 우세. seed별 HQNR 승패 1:1 |
| QESUR−QFULL | +0.001108 | +0.000923 | 평균 HQNR는 e-surrogate 우세, RR는 FULL 소폭 우세 |

QESUR의 HQNR−QFULL는 seed9281101에서 +0.002221667,
seed9281102에서 −0.000005101로 두 번째는 사실상 매우 작은 차이다.
Validation-selected 보조 결과에서도 QESUR 평균 HQNR가 높으며 두 seed 모두 값은 높다.
따라서 selector 변화가 QFULL의 e-surrogate 대비 우위를 만들어주지는 않는다.

**현재 뒷받침되는 표현:** 고정 F1, 두 Student 반복에서 원래 q weighting이 uniform train-mean weighting보다 유리했다.

**아직 뒷받침되지 않는 표현:** q와 sample 연결이 반드시 필요하다 / e로 대체할 수 없다 /
두 cue 분리가 모든 지표에서 최적이다.

QFULL의 HQNR SD는 약0.004275, QESUR와 평균 차이는 약0.001108이다.
단순히 SD와 mean 차이를 비교해 유의성 여부를 판정하지는 않지만, 두 seed로 강한 결론을 내릴 근거가 부족하다는 점은 명확하다.
Paired 결과를 우선 사용하고 모든 사전 등록 seed를 수집한다.

**판정:** 설계상으로는 가장 직접적인 rebuttal 표이지만 현재는 n=2 preliminary다.
기존 ABLR2 C15 결과와의 불일치까지 포함해 해석해야 한다.

## 7. 추가 표/그림 후보 D — Inference correction under controlled PAN translations

QFULL의 동일 exact50K 모델에서 A_ON과 A_ZERO_INFERENCE_ONLY를 비교했다.
20개 장면을 각 방향별로 집계하고, 반경의 8방향을 Student 안에서 평균낸 후 두 Student를 평균했다.
Native RR 표의 crop과 다른 **fixed192 ROI**이므로 숫자를 서로 직접 섞지 않는다.

| r (HR px) | A_ON ERGAS↓ | A_ZERO ERGAS↓ | ΔERGAS ON−ZERO | ΔPSNR ON−ZERO | 상태 |
|---|---|---|---|---|---|
| 0 | 2.05723 | 2.06671 | -0.00947 | +0.04626 | 집계 coverage=1 |
| 0.25 | 2.22614 | 2.26994 | -0.04380 | +0.16915 | 집계 coverage=1 |
| 0.5 | 2.72963 | 2.81565 | -0.08601 | +0.26150 | 집계 coverage=1 |
| 1 | 4.15330 | 4.31670 | -0.16341 | +0.32933 | 집계 coverage=1 |
| 2 | 7.62102 | 7.85780 | -0.23678 | +0.27242 | 집계 coverage=1 |
| 3 | 9.28751 | 9.34230 | -0.05479 | +0.05153 | 집계 coverage=1 |
| 4 | 9.41779 | 9.41349 | +0.00430 | -0.00419 | coverage 경고 |

### 가능한 주장

0–2px 구간에서 correction을 사용하는 쪽이 같은 checkpoint의 correction=0 조건보다 RR 지표가 좋다.
특히 r=1에서 PSNR +0.32933 dB, r=2에서 +0.27242 dB다.
그러나 zero-shift ERGAS 2.05723이 r=1에서4.15330, r=2에서7.62102로 악화한다.
따라서 이것은 **정합 모듈의 기여 진단**이지 큰 shift에 대한 안정적인 복원 증거는 아니다.

A_ZERO는 처음부터 no-align으로 학습한 모델이 아니다. 학습된 모델에 대한 inference intervention이다.
별도로 학습한 no-align C09와 역할이 다르다.

### 보정 반응은 왜 추가로 확인해야 하는가?

source의 response metric은 `abs(dy)+abs(dx)`의 L1 sum이다.
알려진 shift가 있어도 correction이 native값에서 전혀 변하지 않는 상수 반응이면,
8방향 평균 response error의 기준은 `r(1+sqrt(2))/2`가 된다. 이 값은 수학적 reference이며 실제 학습 baseline의 측정치가 아니다.

| r | 측정 QFULL response error | 상수 반응 reference | reference 대비 error 감소 |
|---|---:|---:|---:|
| 1 | 1.15735 | 1.20711 | 4.12% |
| 2 | 2.34341 | 2.41421 | 2.93% |
| 3 | 3.54868 | 3.62132 | 2.01% |

이 숫자는 “오프셋의 몇 %를 정확히 보정했다”는 물리적 비율은 아니다.
현재 전역 shift response가 상수 반응 reference와 가깝다는 진단이다.
또한 source는 A_ZERO에서도 **estimated correction을 계산**하여 response를 기록하고 적용만0으로 바꾸므로,
A_ON/A_ZERO의 response 값이 같은 것은 실제 correction 적용 효과가 같다는 뜻이 아니다.

q는 frozen Teacher의 안정성 cue이고, 여기서는 최종 Student 반응을 본다.
따라서 이 곡선만으로 Teacher q가 의미 없다고 결론내릴 수 없다.
F1/초기 A/학습 후 A 및 patch size별 반응을 비교해야 원인을 분리할 수 있다.

### Coverage 경고

각 곡선에 invalid geometry40건, 전체16곡선에640건이 있다. Numerical failure는0이다.
업로드된 point를 보면 r=4의 D041(+dx=4)·D043(+dy=4)에서만 mean coverage<1이 확인된다.
0–3 반경의 point 평균 coverage는1이다. 단, 원 per-scene coverage를 이번에 별도 회수한 것은 아니다.
Coverage는 PAN/LP/HP frontend support에 대한 검사이며 U의 전체 receptive field를 보장하지 않는다.

r=4 값을 빼고 좋은 범위만 “전체 stress 성능”으로 제시하지 않는다.
원49점 결과를 보존하고 affected positions를 명시한다.
필요하면 성능을 보지 않고 정한 공통 support/ROI protocol v2로 **모든 case/seed/mode 전체를 재평가**한다.
이는 추가 학습이 아니라 inference-only이며 기존 v1 결과를 덮어쓰지 않는다.

**판정:** 현재는 내부 진단으로 가치가 크다. 강한 robustness 그림으로 넣기 전에는 coverage 및 relative-response 점검이 필요하다.

## 8. 제출본 주장과 추가 증거의 연결

| 제출본의 논점 | 직접적인 대조 | 현재 판정 |
|---|---|---|
| Student LP/HP frequency inputs | C01−C00, C07−C08 | 가장 일관된 근거 |
| PAN correction의 복원 기여 | C03−C02, C07−C09, A_ON−A_ZERO | GF2 RR 이득, WV3 native는 작은 효과 |
| shift consistency로 correction 안정화 | C03−C17, C07−C11, Teacher TPLUS−TZERO + response | native 이득 혼재; response 직접 점검 필요 |
| q를 e와 구분할 필요 | QFULL/QMEAN/QSHUF/QESUR, C07−C15 | uniform 대비 이득은 있으나 e 대체불가 주장 미입증 |
| adaptive reconstruction fitting | C04/C05, C12/C13, advantage-only controls | hard emphasis 효과가 불리한 경우가 많음 |
| 특정 gradient 경로 분리의 필요 | q-edge-only / q-A-only / route-to-A | 현재 표로 직접 분리되지 않음 |
| alignment를 feature extraction 전에 해야 함 | capacity-matched early / feature / no-align | 기존 C09·A_ZERO만으로는 비교 불충분 |

제출본의 Table3 행을 수정하는 제안이 아니다.
기존 제출본을 유지한 채, 향후 추가 표/그림에 어떤 설명이 가능한지의 판단 자료다.

## 9. 다음 작업 우선순위

### 우선 두 step: 새 학습을 최소화

**Step 1 — 완료 증거 회수/누락 보완**
- s3/s5의 B01 원 서버 report와 evidence package를 회수한다. 이미 완료되어 있으면 재학습하지 않는다.
- case별4/6 또는6/6으로 정확히 반영하고 같은 F1/data/evaluator/runtime/map cohort인지 확인한다.
- WV3 ABLR2 누락은 P03/C14 및 P05/C10–C15 총7개다. 기존 파일 우선 확인하고 없을 때만 원래 등록 seed로 보완한다.
- 추가6seed 규칙은 새 신규 대조에 적용한다. 옛 ABLR2 Teacher/Student 반복과 B01 고정F1 반복을 합쳐 n을 만들지 않는다.

**Step 2 — controlled-shift 원인 진단**
- F1 Teacher, Teacher에서 복사한 초기 Student A, 학습 후 A를 같은 입력/정의에서 평가한다.
- 64·128·256 크기에서 같은 HR-pixel 단위와 shift를 사용해 patch-size 일반화 여부를 진단한다. 원 q probe 범위도 대조한다.
- `A_NATIVE_FIXED`: shifted 입력에서도 native correction을 고정 적용하여 dynamic response의 이득과 native correction의 이득을 분리한다.
- `KNOWN_SHIFT_INVERSE`: native correction−epsilon을 적용하는 알려진 추가 shift 상쇄 기준을 둔다. 이것은 native 정합 GT를 사용하는 절대 oracle이 아니며 보간 오차가 남을 수 있다.
- r=4 support 경고를 수학적/합성 영상 검사로 점검하고, protocol 변경 시 전조건을 같은 v2로 재평가한다.
- 기존24 Student가 모두 확보되면 새 inference mode도 각 case의6개 모델에서 평가한다. 동일 checkpoint 반복 실행을 독립6회로 세지 않는다.

### 그 다음 신규 학습

| 우선순위 | 추가 대조 | 최소 신규 run | 이유 |
|---|---|---:|---|
| P1 | q-edge-only, q-A-only | 2case×6=12 | FULL·MEAN 재사용 가능할 때 두 경로2×2를 완성 |
| P1 | advantage a_T만 제거, hard weight를 train-mean으로 정규화한 대조 | 2case×6=12 | C05/C12의 악화가 selective transfer인지 scale/gradient 영향인지 분리 |
| P2 | 같은 capacity의 input-align / feature-align / trained no-align | 새 feature case6, 기준 재사용 불가 시3case×6=18 | align-first라는 위치 주장의 직접 대조 |
| P2 | 공통 fixed recipe로 QB/GF2 q substitution 확장 | sensor당4case×6=24 | WV3/F1 조건부 결과의 일반화 |
| P3 | 더 넓은 coefficient sweep·추가 Teacher seed | 진단 후 결정 | 원인 미분리 상태의 반복 확대는 우선순위 낮음 |

신규case는 모두 s1/s3/s5 각각2회, case당6 Student로 진행한다.
동일 repeat 내 초기 U/A·Teacher·stream·학습량을 맞춘다.
여기서 unit-mean hard weighting은 진단용 새 대조이며 제출된 method를 조용히 바꾸는 것이 아니다.
재사용 기준이 다르면 FULL/MEAN baseline도 같은 protocol로 다시 만들어야 한다.

## 10. 논문 투입 검토 결론

- **준비도가 높음:** frequency inputs의 paired ablation; GF2 RR의 alignment/initialization 효과.
- **조건부 사용:** sensor별 차이를 명시하는 leave-one-out 표; WV3 edge 제거 대조는 미등록2개 회수 후.
- **아직 핵심 주장 근거로 부족:** q의 e 대비 고유성, adaptive hard fitting의 일관적 이득, Student refinement의 필수성.
- **먼저 진단할 그림:** 큰 shift robustness. A_ON의 작은 이득과 absolute degradation을 동시에 보여야 한다.
- **B01 결론의 범위:** s1 두 seed만의 결과다. s3 신규 B01은 아직 분석자료에 들어 있지 않다.

## 11. 자료·재계산 경로

- `source_B01_native_16.csv`: 원본s1 16 selection행.
- `source_B01_stress_784.csv`: point별49-grid·metric·coverage·provenance.
- `source_B01_status_24.csv`: s1/s3/s5의 실제 수집 상태.
- `source_ABLR2_WV3_s1_GF2_s3_386.csv`: 두 센서193run의 primary/endpoint 관측.
- `tables/`: 전체case 평균·SD, paired per-block·요약, 표 후보, 누락slot.
- `analysis_validation.json`: 스냅샷SHA·집계수·cached formula 반올림 허용 범위.
- 스냅샷의 formula cache는 일부10자리 유효숫자로 반환되어 직접 source 재계산과 미세한 차이가 있다. 실제 source값으로 재집계했으며 화면표시 자릿수의 결론에는 영향이 없다.
- `source_pan-cvpr27.xlsx` SHA256: `cd1760636384a004fc9100d4d3c857caceb13770d0ab83a0caedc1c8671d06ec`.

## 12. 설명의 근거

- 제출본: 6쪽 Fig.5 및3.3절의 distinct cue 동기, 7쪽 Eq.(10)–(12)의 q routing, 9쪽 Table3의 component 서술.
- Sheet: https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit
- case catalog: https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/research_log/PANDA_ABLR2X_S123_Continuous_C17_GF2_Bundle_2026-09-23/registries/ABLR2X_ComponentCatalog_18.csv
- B01 수집기록: https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/results_log/2026-09-30_b01-normalized-sheets-backfill.md
- stress 정의: https://github.com/hojunking/PAN-Crafter-repro/blob/ea71b68be637a1d1f4d61bf89cb4e023cbf00385/panda_rb/stress.py

수치표는 source-derived calculations, 유효성 판정·추가실험 우선순위는 그 결과에 대한 분석자의 판단/제안이다.
