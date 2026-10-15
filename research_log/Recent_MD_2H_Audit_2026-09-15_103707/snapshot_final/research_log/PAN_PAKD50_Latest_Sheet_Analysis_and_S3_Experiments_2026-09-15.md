# PAKD50 최신 Sheet 검토와 s3 추가 실험
## 2026-09-15 오전 조회 · raw-original HQNR 기준

**조회 기준 시각:** 2026-09-15T09:32:07+09:00  
**근거:** `pan-cvpr27`의 WV3-s1/s2/s3(5090)/s4/s5 live values 및 같은 조회에서 export한 workbook.  
**완료 결과 행:** 41행 = s1 5 / s2 7 / s3 7 / s4 12 / s5 10. 결과 행의 존재는 checkpoint·서버 프로세스까지 재검증했다는 뜻이 아니다.  
**코드 검토:** `2d0fc25e54aa190688031f2feba0ad2a22db506c`.  
**작업 범위:** 읽기·계산·추가 실험 설계. Sheet 수정, Git push, 서버 학습 실행은 하지 않았다.

> **권고:** s3에는 `J_R3_NOEDGE → J_N0_EDGE → LF0 → LFQ → LFX`의 5개 파생 실험을 배정한다.
> s3의 XJ(raw 0.9578)가 Q12(raw 0.9554)보다 높다는 관측과,
> s5의 LFQ(raw 0.9587)가 현재 새 파생 후보 중 목표에 가장 가깝다는 관측을 연결한다.
> “soft를 제거하면 항상 좋아진다” 또는 “후반 동결만으로 0.007 개선됐다”라고 해석하지 않는다.
> 기본 예약 약 **8시간**, 방법 고정 후 확인 최대 3run까지 포함하면 약 **12시간 55분**이다.

## 1. 읽는 규칙

1. 아래 실험 성능표는 **N열 HQNR↑ = 기존 selector로 선택된 checkpoint의 raw-original 값**이다.
   O열 HQNR(V64)로 목표 0.959–0.960 도달을 판정하지 않는다. `(50K)`는 run의 학습량이며
   선택된 checkpoint가 exact50K라는 뜻이 아니다.
2. Sheet는 네 자리로 저장된 값을 제공한다. 아래 차이는 그 값의 산술 차이다.
   원시 JSON의 고정밀 차이나 matched-grid 재선택 차이를 새로 얻었다고 표현하지 않는다.
3. 이번 조회에서 Y열 이후는 비어 있고, 최신 last·plateau·selected step은 확인되지 않았다.
   예전 16-run 보고서의 plateau를 새 LF/RC/XJ 행에 채워 넣지 않는다.
4. 비교는 가능한 같은 서버·같은 Student seed·같은 recipe/version끼리 한다.
   실제 실행 commit·초기 state·데이터 RNG·Teacher/pilot hash 동치 여부는 별도 분석 workflow가 제공한다.
5. Student seed는 현재 {1234,777,2026,7777}의 네 값이다. seed7777은 s4 J0 한 행만 있어
   그 seed의 KD 효과는 아직 계산할 수 없다. s1↔s4(seed1234), s3↔s5(seed2026), v1/v2 반복은
   독립 Student seed를 늘리지 않는다. 모든 Student는 현재 같은 T0에 조건부다.
6. HQNR은 원래 주 기준을 유지한다. ERGAS와 같은 RR 지표의 변화를 숨기지 않되,
   하나의 seed·서로 다른 selected step의 차이를 순수 seed 효과 또는 방법의 전반적 악화로 단정하지 않는다.

## 2. 최신 현황

| server | result rows | highest selected record | raw HQNR | source range |
| --- | --- | --- | --- | --- |
| s1 | 5 | F0 / seed1234 / v1 | 0.9558 | 'WV3-s1'!B56:X56 |
| s2 | 7 | JQ / seed777 / v1 | 0.9583 | 'WV3-s2'!B70:X70 |
| s3 | 7 | XJ / seed2026 / v1 | 0.9578 | 'WV3-s3(5090)'!B88:X88 |
| s4 | 12 | J0 / seed1234 / v1 | 0.9594 | 'WV3-s4'!B7:X7 |
| s5 | 10 | LFQ / seed2026 / v1 | 0.9587 | 'WV3-s5'!B14:X14 |

전체 41행의 값·run ID·원본 위치·Notes는 `live_PAKD50_41runs.csv`에 보존했다.
s4 J0_v1의 0.9594는 기존 보고서가 중간 최고점으로 해석한 자산이다. 이번 검토에서 그 숫자를 삭제하지 않으며,
단일 selected 기록과 반복적으로 목표를 달성하는 방법을 구분한다. 같은 s4 seed1234의 J0_v2는 0.9568이다.
v1/v2 사이의 실행 변경을 확인하기 전까지 그 차이를 순수 난수·GPU·선택자의 영향 중 하나로 귀속하지 않는다.

### 2.1 s3의 현재 기준

| case | raw HQNR | ERGAS | Dλ | Ds | Train(h) | Sheet row |
| --- | --- | --- | --- | --- | --- | --- |
| J0 | 0.9553 | 2.0860 | 0.0196 | 0.0256 | 1.16 | 81 |
| F0 | 0.9540 | 2.0411 | 0.0233 | 0.0233 | 1.12 | 83 |
| JR | 0.9552 | 2.0875 | 0.0221 | 0.0233 | 1.31 | 84 |
| JQ | 0.9554 | 2.0758 | 0.0214 | 0.0237 | 1.34 | 85 |
| FQ | 0.9551 | 2.1917 | 0.0233 | 0.0222 | 1.28 | 86 |
| FR | 0.9533 | 2.0907 | 0.0214 | 0.0258 | 1.27 | 87 |
| XJ | 0.9578 | 2.1004 | 0.0178 | 0.0249 | 1.33 | 88 |

s3에서는 **XJ가 0.9578로 현재 통합 run 중 최고**다. XJ는 `R1 + GT signed Scharr edge`이며 output soft는 없다.
대응 차이는 다음과 같다.

- XJ−JQ = **+0.0024**
- XJ−JR = **+0.0026**
- XJ−J0 = **+0.0025**
- XJ−F0 = **+0.0038**

이는 s3/seed2026/현재 selected 규약에서의 관측이다. XJ가 모든 seed의 우승 방법이라는 결론은 아니다.
Q12 대비 XJ의 Dλ는 0.0214→0.0178로 낮아졌지만 Ds는 0.0237→0.0249로 높아졌다.
따라서 “모든 성분이 개선됐다” 또는 “공간 정합이 더 정확해졌다”라고 말하지 않는다.

## 3. 새 결과가 바꾼 해석

### 3.1 Output soft의 효과는 조건에 따라 반전된다

| 비교 block | JQ | XJ (soft OFF) | JQ−XJ |
|---|---:|---:|---:|
| s2 / seed777 / v1 | 0.9583 | 0.9538 | **+0.0045** |
| s3 / seed2026 / v1 | 0.9554 | 0.9578 | **−0.0024** |
| s4 / seed1234 / v1 | 0.9558 | 0.9562 | **−0.0004** |

s4의 JQ_v2는 0.9541로 XJ_v1과의 표시값 차이가 −0.0021이지만, 이것을 같은 버전의 대조로 합치지 않는다.
원래 v1 대조와 v2 참조를 구분한다.

s2에는 soft×edge의 네 조건이 처음으로 모였다.

| s2 seed777 | soft | GT edge | raw HQNR |
|---|---|---|---:|
| JR | OFF | OFF | 0.9543 |
| J_R3_NOEDGE | ON | OFF | 0.9546 |
| XJ | OFF | ON | 0.9538 |
| JQ | ON | ON | 0.9583 |

표시값에서 `JQ−XJ−J_R3_NOEDGE+JR = +0.0042`다.
**이 block에서는 soft와 edge를 함께 쓸 때의 이득이 단독 추가의 합보다 큰 패턴**이다.
서로 다른 selected checkpoint 및 단일 seed의 결과이므로 보편적 상호작용의 증명으로 쓰지 않는다.

그와 반대로 s3는 soft를 없앤 XJ가 더 좋다. 따라서 지금 β를 일괄 0으로 만들거나 확대하는 결정은 이르다.
s3에 J_R3_NOEDGE를 추가해 같은 네 조건을 채우고, J_N0_EDGE로 Teacher-error hard의 추가 가치도 확인한다.
Q12는 variance distillation이 아니라 adaptive output KD + GT signed-edge supervision이라는 정의를 유지한다.

### 3.2 s5 LFQ는 유력하지만, +0.007을 순수 후반 동결 효과로 읽으면 안 된다

| s5 seed2026 | raw HQNR | 비교 |
|---|---:|---|
| J0 | 0.9570 | 일반 joint N0 |
| JQ | 0.9576 | 일반 joint Q12 |
| LF0 | 0.9517 | 25K부터 A 동결, N0 |
| **LFQ** | **0.9587** | 25K부터 A 동결, Q12 |
| RC0 | 0.9577 | Stage2 offset OFF, N0 |

- LFQ−LF0 = **+0.0070**: 같은 LF 일정에서 backend 전체를 바꾼 차이.
- **LFQ−JQ = +0.0011**: 같은 Q12에서 LF 일정을 적용한 표시값 차이.
- LFQ−RC0 = **+0.0010**: 다른 유력 정책의 GT-only 모델 대비 차이.

LFQ는 raw 0.959까지 표시값 기준 약 **0.0003** 남아 있어 반복 가치가 있다.
하지만 새 LFQ의 last/plateau는 이번 Sheet에 없으므로 “후반 안정화가 입증됐다”로 확대하지 않는다.

선택 checkpoint의 비용도 남긴다. JQ→LFQ에서 ERGAS는 2.0601→2.1549(+0.0948),
Dλ는 0.0190→0.0219, Ds는 0.0239→0.0198이다. 현재 HQNR 상승은 이 선택값들에서 공간 성분 개선이
분광 성분 악화보다 크게 나타난 균형이다. 선택 시점이 다를 수 있으며, 후반 동일시점 비교는 별도다.

**LF 비교의 실행 전제에 관한 검토 추론:** 과거 16-run 보고서에서 s5 J0의 selected step은 22,220,
raw 0.9569710이다. LF0가 J0와 처음 25K까지 완전히 동일한 학습 경로와 후보 이력을 갖는다면 이 후보도
LF0에 남아야 한다. 그런데 LF0의 Sheet selected는 0.9517이다. 이 차이만으로 버그를 확정할 수는 없지만,
현재 LF0의 저하 전부를 25K 이후 동결로 설명할 수 없다는 뜻이다. 이번 새 LF 실험은 아래 prefix 계약을 지킨다.
이 확인은 새 분석 캠페인을 s3에 배정하라는 뜻이 아니라, 이미 진행 중인 분석의 자산·설정 대조와
실행 release의 동일성 검사를 받아 사용하는 전제다.

### 3.3 Offset OFF에서는 Q12의 추가 이득이 작다

| block | RC0 | RCQ | RCQ−RC0 |
|---|---:|---:|---:|
| s4 seed1234 | 0.9551 | 0.9554 | **+0.0003** |
| s5 seed2026 | 0.9577 | 0.9576 | **−0.0001** |

“offset를 끄면 KD가 더 좋아진다”는 기대는 현재 selected 표에서 지지되지 않는다.
반면 s5 RC0=0.9577은 강한 no-KD 기준이므로, LFQ나 새 KD 후보의 추가 가치를 읽을 때 보존한다.
이 두 차이를 통계적 동등성의 증명으로 읽지 않는다.

### 3.4 무조건 동결·초기 동결·protected routing으로 바꿀 근거도 부족하다

- FQ−F0: s1 −0.0021 / s2 −0.0017 / s3 +0.0011. Frozen Q12가 보편적으로 좋지 않다.
- s4 ALQ−AL0: −0.0008. A LR 축소와 Q12가 자동으로 더해지지 않는다.
- s5 DQ−D0: +0.0019지만 **DQ−JQ는 −0.0027**. 초기 동결이 최종 최고 성능을 높이지 못했다.
- s5 PQ−JQ: −0.0007. 추가 감독을 A에서 전부 차단한 방식의 우위는 아직 관측되지 않았다.

따라서 s3에서 D/P/AL을 넓게 다시 스윕하지 않고, **XJ의 좋은 구조 감독과 LF 일정의 결합**을 좁게 시험한다.

### 3.5 역사적 plateau 결과는 별도로 보존한다

첨부 `2026-09-14-pakd50-campaign-results.md`는 16-run 시점에 joint Q12의 plateau 이득을
3개 고유 seed로 접어 +0.004557 ± 0.000750으로 보고했다. 이는 그 시점의 후반 지표 결과이며,
오늘 추가된 XJ/RC/LF/N0-edge 실험의 후반 기록은 아니다. 본문 selected 비교와 숫자를 섞지 않는다.
보고서의 “plateau를 판정 기준으로 바꾸자”는 권고는 보고서의 제안으로 보존한다.
이번 인계는 기존 raw selector를 유지하고 exact50K/고정 plateau를 별도 필드로 받는다.

## 4. s3 추가 실험 — 5개 기본 묶음

**개발 seed: 2026.** s3의 기존 J0/JQ/JR/XJ/F0/FQ/FR을 control로 재사용한다.
단, 기존 control과의 구현·초기화 계약이 맞지 않으면 그대로 인과 대조로 쓰지 않는다.
추가 control이 꼭 필요하면 그 비용을 별도로 계산하고 기존 완료 run을 덮어쓰지 않는다.

| order | case | main control | reference(h) | reserved(h) |
| --- | --- | --- | --- | --- |
| 1 | J_R3_NOEDGE | JR, JQ, XJ | 1.34 | 1.6407 |
| 2 | J_N0_EDGE | J0, XJ | 1.34 | 1.6407 |
| 3 | LF0 | J0, F0 | 1.16 | 1.4427 |
| 4 | LFQ | JQ, LF0, F0 | 1.34 | 1.6407 |
| 5 | LFX | XJ, LF0, LFQ | 1.33 | 1.6297 |

### 4.1 J_R3_NOEDGE
- 정책 J, backend R3, GT edge OFF.
- weighted hard와 adaptive output soft의 정의·α=1·β=0.1은 Q12와 같다.
- 비교: J_R3_NOEDGE−JR(soft 단독), JQ−J_R3_NOEDGE(edge 추가),
  JQ−XJ−J_R3_NOEDGE+JR(표시값 상호작용).
- 기존에 등록된 case다. 같은 이름의 config가 존재한다는 것과 완료됐다는 것은 다르다.

### 4.2 J_N0_EDGE
- 정책 J, plain GT L1 + 동일 λE의 GT signed Scharr edge.
- Teacher-error hard와 output soft는 모두 없다.
- T0는 aligner의 초기 자산으로 남고, 필요 시 평가 bin에만 사용한다. Stage2 학습 loss는 Teacher를 사용하지 않는다.
- 비교: J_N0_EDGE−J0(GT edge의 가치), XJ−J_N0_EDGE(같은 edge 위의 Teacher-informed hard 가치).

### 4.3 LF0 / LFQ / LFX
세 case 모두 **t=0…24,999에서 A/U joint**, **t=25,000…49,999에서 A frozen**, U는 50K까지 학습한다.

| case | t<25K의 대응 recipe | U loss 전체 | A trainable 구간의 loss |
|---|---|---|---|
| LF0 | J0 | plain GT | plain GT + offset |
| LFQ | JQ | weighted hard + soft + edge | Q12 + offset |
| **LFX** | **XJ** | **weighted hard + edge; output soft 없음** | **X02 + offset** |

A 동결 후에는 offset 연습을 실행하지 않고 A의 gradient·AdamW weight decay·moment 업데이트를 멈춘다.
U의 optimizer/moments/global cosine은 연속 유지한다. U의 학습률을 재시작하지 않는다.

**LFX는 새 방법 블록을 추가하는 것이 아니라 기존 LF policy와 X02 backend를 조합한 새 case 등록이다.**
확인한 main에는 LF0/LFQ는 있지만 LFX는 없다. 등록·설명·완료 판정·예약·unit test를 함께 추가한다.

핵심 비교:
- LFQ−JQ: Q12에서 후반 A 동결의 추가 효과.
- LFX−XJ: output soft 없이도 같은 일정이 좋은가.
- LFQ−LFX: 같은 LF 일정에서 output soft의 추가 기여.
- LFQ/LFX−LF0: LF의 GT-only 대조 대비 전체 backend 효과.
- 새 후보의 F0 대비 성능도 보존한다. s5 RC0의 절대값은 참조이며 s3 후보와 직접 paired delta로 쓰지 않는다.

### 4.4 Prefix·초기화·RNG 계약
- LF0/J0, LFQ/JQ, LFX/XJ의 처음 25K는 loss, parameter group, lr, optimizer 설정,
  A/U 초기 state, batch/augmentation/epsilon RNG, 평가·진단의 RNG 소비 규칙이 같아야 한다.
- 가능한 경우 exact25K의 **전체 state**에서 분기한다. 가중치만 복사하고 optimizer를 reset하면 다른 방법이다.
- 기존 격자의 24,240 또는 25,250을 25K로 대체하지 않는다. exact25K complete state가 없으면 새 full50K run으로 수행한다.
- 이번 시간 계산은 prefix 공유에 따른 시간 절감 없이 full50K 기준이다.
- 각 case는 기존 성공 class/state를 재사용한다. Head zero-init, backbone 크기 변경, input crop/normalization 변경은 없다.
- T0는 자기 frozen A/U로 forward한다. Student의 변화하는 aligned PAN을 T0에 넣어 target을 바꾸지 않는다.
- τR=0.012463942170143127, λE=0.09075170336956798를 공통 package에서 읽는다.
  s3만 재보정하거나 λE=0.05로 넣지 않는다.

## 5. 확인 단계 — 방법을 고정한 뒤 최대 3run

**새 Student 확인 seed 4321을 제안**한다. 현재 Sheet에서 해당 seed는 관측되지 않았으며,
s4의 예약 seed3407, s5의 예약 seed9091과 분리한다. 결과를 본 뒤 더 유리한 seed로 바꾸지 않는다.
새 초기 state를 생성·hash 고정한 뒤 해당 묶음의 모든 case가 공유한다.

| 선택된 후보 | s3 seed4321 확인 묶음 |
|---|---|
| LFQ | JQ / LF0 / LFQ |
| LFX | XJ / LF0 / LFX |
| J_N0_EDGE | J0 / XJ / J_N0_EDGE |
| J_R3_NOEDGE | J0 / JR / J_R3_NOEDGE |
| 기존 XJ | J0 / JR / XJ |
| 기존 JQ | J0 / XJ / JQ |

- 별도 분석에서 방법·계수·정책·version을 먼저 고정한다.
- 약한 control 대비 큰 증가만으로 승격하지 않는다. 기존 해당 backend의 joint anchor 및 F0 등 강한 기준을 함께 읽는다.
- LFQ가 s3 seed2026에서 좋아져도 s5 seed2026과 합쳐 독립 2seed라고 하지 않는다.
  실제 확인 범위는 같은 seed의 환경 반복이다. seed4321까지 완료하면 해당 candidate는 최대 2개의 고유 Student seed다.
- 확인 단계 후에는 새 scalar/architecture 조합을 같은 캠페인에서 또 고르지 않는다.
- 전체 pipeline의 Teacher 재학습 독립 반복이 아니라 같은 T0 아래 Student 반복이다.

## 6. 소요시간 — s3 실측에서 계산

현재 s3의 기록은 J0 1.16h / F0 1.12h / JR 1.31h / JQ 1.34h / FQ 1.28h / FR 1.27h / XJ 1.33h다.
이는 각각 약 67–80분 범위다. s2의 약 2.3h나 s5의 PQ 1.53h를 s3의 일반 case 시간으로 쓰지 않는다.

미실행 case의 시간은 실측이 아니라 **같은 서버의 관련 완료 run을 대용한 reference**다.
Loss/offset/A update를 제거해 빨라질 가능성을 미리 차감하지 않는다.
기본 5run의 reference 합:

`1.34 + 1.34 + 1.16 + 1.34 + 1.33 = 6.51h`

기존 예약 규칙을 한 번만 적용:

`1.10 × 6.51 + 5 × 10/60 = 7.994333…h = 7시간 59분 40초`

- 10%는 runtime 변동 여유.
- run당 10분은 종료 후 기본 export/gspread/큐 전환에 대한 가예약이며 실측값이 아니다.
- Train(h)에 포함된 정기 평가 시간을 다시 더하지 않는다.
- 학습 중 원래 실행되는 기록/정기 평가와, 별도로 진행되는 분석 workflow를 구분한다.
- 실제 후처리나 LF 첫 run의 시간이 reference를 넘으면 잔여 queue 예약을 갱신한다.

확인 3run에는 모두 s3 JQ reference 1.34h를 상한 보장이 아닌 **공통 대용값**으로 사용한다:

`1.10 × (3 × 1.34) + 3 × 10/60 = 4.922h = 4시간 55분 19초`

따라서 **최대 8run** 예약 합은:

`7.994333… + 4.922 = 12.916333…h = 12시간 54분 59초`

정밀한 초 표시는 예약식의 산술 검산이며 실제 완료시간의 초 단위 예측이 아니다.
표현상 기본 약 8h / 확인 포함 약 12h55m로 사용한다.

**제외된 시간:** 코드 구현·배포 대기, 현재 실행 중인 다른 작업의 잔여시간, 외부 분석의 WIN 결정 대기,
기본 10분을 초과하는 큰 후처리. 이 항목이 생기면 실제 시계에 더한다. 새 슬롯이 비어 있어야 첫 run을 시작한다.

### 공통 마감
- 학습 마감: **2026-09-16 11:22:31 KST**
- 최종 마감: **2026-09-16 15:22:31 KST**
- 조회 기준 09-15 09:32:07에는 학습 마감까지 **25시간 50분 24초** 남았다.
- 지금부터 50시간을 다시 시작하지 않는다. 최종 4시간도 학습 마감 뒤에 이미 예약되어 있어 다시 차감하지 않는다.
- 실제 시작 시각과 대기시간으로 admission을 재계산한다. 부족하면 확인 대응묶음을 우선 예약하고
  후순위 확장을 줄이며, 한 쌍 중 처치군만 실행해 인과 대조처럼 쓰지 않는다.

## 7. 구현·배포 인계

1. 이번에는 s3만 추가한다. s1은 분석 전용, s2/s4/s5는 기존 파생 queue를 유지한다.
2. 현재 main의 J_R3_NOEDGE/J_N0_EDGE/LF0/LFQ 등록을 활용한다.
3. 새 등록은 `CASES["LFX"] = ("LF", "X02")`의 의미를 따른다.
   Teacher flags/edge lambda/offset schedule/description/control=`LF0`/run name 등을 resolved config로 검사한다.
4. s3의 priority와 server-local mandatory/reservations를 이번 다섯 case로 명시한다.
   `--cases`로 config만 만들고 queue 편성이 끝났다고 가정하지 않는다.
5. 기존 완료 J0/JQ/JR/F0/FQ/FR/XJ를 다시 넣지 않는다. 새 case라도 local artifact·process가 이미 있으면
   skip/resume 규약을 적용한다. Sheet 행이 없다는 이유만으로 실행 중이 아니라고 판단하지 않는다.
6. 실행 중인 checkout에 hot pull을 하지 않는다. 검증된 release를 다음 run 경계 또는 독립 worktree에서 사용한다.
7. gspread는 기존 `WV3-s3(5090)` 및 X열 `PAKD50 / <case> / FRESH50`을 사용한다.
   정식 완료 자산과 해당 metric만 업로드한다. Sheet를 수동으로 고치지 않는다.
8. Training/candidate/evaluator/Teacher/init/calibration hash를 남긴다.
   확인할 selected/last/plateau/ERGAS/실제 Train(h) 필드 형식은 기존 분석 workflow로 전달한다.

## 8. 이번에 보류하는 확장

- W104–D121 등 architecture 변경: 여전히 후보지만 이번 s3 묶음에는 섞지 않는다.
- α/β/λE 전수 sweep: 현재 soft 효과가 block에 따라 반전되므로 먼저 loss×timing의 작은 교차를 끝낸다.
- D/P/AL의 전수 재실행: 현재 selected에서 JQ보다 높은 일관된 신호가 없다.
- 새 Teacher·feature KD·variance/covariance·moving reference: 현재 두 기술의 정의와 비교자산을 바꾸므로 후순위다.

## 9. 결과에 따른 다음 선택

| 결과 | 해석·다음 선택 |
|---|---|
| LFQ가 JQ보다 좋고 LFX도 XJ보다 좋음 | 후반 A 동결의 양성이 두 backend에서 관측됨. 최종 raw HQNR이 높은 후보를 확인 seed로 |
| LFQ만 좋고 LFX는 개선 없음 | Soft×LF 결합 후보. 전체 soft 기여는 LFQ−LFX로 확인 |
| LFX가 LFQ와 XJ보다 좋음 | X02+LF를 후보로. 이 결과를 output soft distillation 성과라고 부르지 않음 |
| J_N0_EDGE가 XJ와 비슷하거나 더 좋음 | 이 조건에서 Teacher-error hard의 추가 가치가 약함. GT edge 중심의 단순 기준 보존 |
| s3 R3_NOEDGE도 좋아짐 | s2와 다른 soft/edge 균형. 계수 변경 전 현재 네 조건에서 비교 |
| LFQ−LF0만 크고 LFQ−JQ는 작거나 음수 | 약한 LF0의 회복을 timing의 큰 이득으로 과장하지 않음 |
| raw selected만 개선, last/plateau 미개선 | 선택 자산 성능 개선과 후반 안정화 주장을 분리 |
| 신규 후보 모두 현재 XJ보다 낮음 | XJ를 유지하고 확인에 자원 사용. 조합 수를 자동 확대하지 않음 |

## 출처 및 동반 산출물

- Live Sheet: https://docs.google.com/spreadsheets/d/1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0/edit
- 최신 코드: https://github.com/hojunking/PAN-Crafter-repro/blob/2d0fc25e54aa190688031f2feba0ad2a22db506c/tools/gen_pakd50_configs.py
- 이전 보고서: 사용자 제공 `2026-09-14-pakd50-campaign-results.md`의 16-run snapshot. 본문 §3.5 및 LF prefix 검토에만 구분 사용.
- 정의: `PAN KD 두 방법 명세 — Q12 / R1 및 PAN aligner 결합 인터페이스` §4–7,
  `PAN Aligner L1E4 — 확정 기술 명세·실험 근거·KD 통합 인계` §4·6.
- `live_PAKD50_41runs.csv`: 모든 현재 결과행, source URL·row·run ID·Notes·runtime.
- `selected_HQNR_paired_differences.csv`: selected raw 표시값의 대응 차이. 원시 고정밀 재평가가 아님.
- `S3_primary_queue_and_runtime.csv`: 다섯 case의 순서·reference·예약 계산.
- `S3_case_registry.yaml`: 계획용 명세. Trainer에 직접 넣는 실행 config가 아님.
- `live_snapshot_manifest.json`: 조회시각·source hash·scope·집계 규칙.

**최종 결정:** s3는 현재 높은 XJ를 기준으로 soft/edge 대조를 완성하고,
기존 LF policy와 X02 backend를 합친 LFX를 LFQ/LF0와 비교한다.
s2/s4/s5의 실험을 그대로 중복시키는 대신, 서로 다른 backend에서 후반 동결의 효용을 확인하는 작은 교차 실험이다.
