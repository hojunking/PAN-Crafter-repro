# s4 실행 인계 — QEGX / W104–D121
**2026-09-15 · 신규 14run · 학습 기준 21.77h · 예약 26시간 16분 49초 · 시간 상한 없음**

## 이번 역할
q-low edge를 U/A 양쪽에서만 사용할지(QE50), **U에는 전부 주고 A에만 q-low를 적용할지(QER50)** 구분한다.
LF25K와 q-edge의 결합도 탐색한다. s1/s5의 QEDGE9 작업은 변경하지 않는다.

## 기존 결과
s4 seed1234의 W104 NA0/J0/JQ/XJ/F0는 각각 raw 0.9580/0.9585/0.9565/0.9571/0.9574다.
작은 구조 자체의 출발점이 높지만 KD 추가 이득은 아직 양성이 아니다. W112의 v1/v2를 유리하게 골라 대조하지 않는다.

## 공통
W104/D121 Student, frozen T0(W112/D123), copied A, fresh50K/batch48.
U/A LR1e-4/1e-5, α1 β0.1 τR0.012463942170143127 λE0.09075170336956798.
q<0.3276133416220546에서 g=1, GT hard 항상 유지. q는 frozen T0의 상대 반응 residual이다.

## 명시 큐
| 순서 | 단계 | Seed | Case | U / A의 edge 계수 | 시간 기준(h) | 예약(h) | 질문 |
|---:|---|---:|---|---|---:|---:|---|
| 1 | A | 1234 | `QE50` | `g` / `g` | 1.50 | 1.8167 | q-low patch만 edge 감독 |
| 2 | A | 1234 | `LF0` | `0` / `0` | 1.17 | 1.4537 | 작은 모델의 25K 이후 A 동결 GT-only 기준 |
| 3 | A | 1234 | `LFQ` | `1` / `1 if t<25000 else 0` | 1.38 | 1.6847 | 작은 모델의 LF Q12 기준 |
| 4 | A | 1234 | `LFQE50` | `g` / `g if t<25000 else 0` | 1.50 | 1.8167 | q-edge gating × late-freeze 상호작용 |
| 5 | A | 1234 | `LFX` | `1` / `1 if t<25000 else 0` | 1.37 | 1.6737 | LF 안에서 output soft 제거 |
| 6 | A | 1234 | `JE0` | `1` / `0` | 1.80 | 2.1467 | U에는 all-edge, A에는 edge를 전혀 전달하지 않는 routing 대조 |
| 7 | A | 1234 | `QER50` | `1` / `g` | 1.80 | 2.1467 | U all-edge, A q-low edge만 전달 |
| 8 | A | 1234 | `QERS` | `1` / `g_shuffle` | 1.80 | 2.1467 | routing의 실제 q-sample 연결 대조 |
| 9 | C | 3407 | `J0` | `0` / `0` | 1.17 | 1.4537 | 동일 구조·seed의 joint GT-only 기준 |
| 10 | C | 3407 | `JQ` | `1` / `1` | 1.38 | 1.6847 | 기존 Q12 기준 |
| 11 | C | 3407 | `QE50` | `g` / `g` | 1.50 | 1.8167 | q-low patch만 edge 감독 |
| 12 | C | 3407 | `JE0` | `1` / `0` | 1.80 | 2.1467 | U에는 all-edge, A에는 edge를 전혀 전달하지 않는 routing 대조 |
| 13 | C | 3407 | `QER50` | `1` / `g` | 1.80 | 2.1467 | U all-edge, A q-low edge만 전달 |
| 14 | C | 3407 | `QERS` | `1` / `g_shuffle` | 1.80 | 2.1467 | routing의 실제 q-sample 연결 대조 |

## 직접 대조
| Case | U edge | A edge |
|---|---:|---:|
| JQ | 1 | 1 |
| JE0 | 1 | 0 |
| QE50 | g | g |
| QER50 | 1 | g |
| QERS | 1 | g_shuffle |

Reconstruction hard/soft와 offset는 같은 정책에서 유지한다. QER는 두 total loss를 더해 reconstruction을 이중 계산하지 않는다.
현재 registry의 `edge_gate + routing` 금지를 없애는 것으로 구현하지 말고, per-sample `edge_route`를 별도로 검증한다.

## 기존 control
W104 seed1234 J0/JQ/XJ의 verified state·init/Teacher/data/evaluator/학습경로 동치를 확인해 재사용한다.
J0/JQ가 불충분하면 같은 새 release의 새 버전으로 둘을 다시 실행한다: **약3h8m 추가**.
LF0의 25K 이전은 같은 seed J0와, LFQ는 JQ와, LFQE50는 QE50와 같은 계약이다.
동결 시 A.grad=None, weight decay drift 없음. U scheduler는 재시작하지 않는다.

## 확인 범위
Seed3407의 core는 J0/JQ/QE50/JE0/QER50/QERS다. q routing은 두 seed에서 평가된다.
LF 단일 seed만으로 채택하지 않는다. 유력하면 seed3407 LF0/LFQ/LFQE50 3run **약4h57m** 추가.

## 시간·진행
실측 W104 J0/JQ/XJ=1.17/1.38/1.37h를 사용. 새 q gate1.50h·routing1.80h는 가예약이며 실측이 아니다.
현재 run을 죽이지 않고 경계 인계. routing 구현이 늦으면 먼저 QE50/LF 및 S3407 J0/JQ/QE50 같은 준비된 독립 항목을 수행할 수 있다.
이전 캠페인의 절대 마감/50h/9h 중단을 상속하지 않는다.
기존 WV3-s4 탭, gspread 경로, X열 QEGX 구분을 유지한다. 직접 Sheet 편집 없음.
전체 수식·판정·구현 규약은 `Experiment_Plan.md`를 따른다.
