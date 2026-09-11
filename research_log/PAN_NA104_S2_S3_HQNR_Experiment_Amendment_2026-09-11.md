# s2·s3 전달: NA104 실험 설계 조정 및 HQNR 중심 판정 지침

**작성일:** 2026-09-11  
**대상:** NA104 · WV3 · W104–D122 · no-align · 동일 용량 Teacher–Student  
**문서 성격:** 현재 실험의 변경 지시서 및 운영 인계. 기존 전체 방법 명세를 대체하지 않는 추가 문서.  
**결정:** 기본 아키텍처·기존 loss 정의·진행 중 run은 유지한다. **HQNR 비교 조건, 진단, 대기 case의 순서와 확장 조건은 조정한다.**  
**적용 범위:** 아래 순서와 절차를 운영자가 적용한다. 이 문서 작성 과정에서 서버 프로세스, GitHub 파일, Google Sheet를 직접 변경하거나 새 학습을 실행하지 않았다.

> **공통 전달 요약**  
> 1. 주 성능·선택·case 승급 기준은 **원본 FR 논문 세트 20장의 HQNR**이다. ERGAS·학습 L1은 보조 진단이다.  
> 2. 정상 진행 중인 run은 원래 설정으로 완료한다. 완료 run과 원래 best는 보존한다.  
> 3. s2는 **N0/R0/R1/R2/R3 → GT 통계·edge → GV-H/WH/FIX/AD** 순서로 핵심 비교를 완성한다.  
> 4. s3는 빠져 있는 **R0/R1/R2의 서버 내부 대조**를 보완하고, **Q35/Q36/Q37의 N0+GT 통계**를 앞당긴다.  
> 5. R3가 검증되기 전에 R3 기반의 모든 통계×gate×window를 자동 소진하지 않는다. 정의는 보존하고 후속 실행만 조건부로 둔다.  
> 6. 공통 평가 간격의 HQNR best, 같은 checkpoint의 Dλ/Ds, 후반 HQNR, 장면별 차이, 대응 seed를 함께 기록한다.  
> 7. Teacher 신호를 크게 만드는 것 자체가 목표가 아니다. **그 신호를 썼을 때 N0보다 HQNR이 좋아지는가**가 성능 질문이다.

---

## 0. 근거·결정·추론의 구분

문서 내 표시는 다음과 같다.

| 표시 | 의미 |
|---|---|
| **[사용자 결정]** | 최신 대화에서 확정한 조건. 이전 계획의 충돌 항목보다 우선한다. |
| **[자료 확인]** | 첨부 보고서 또는 이번에 읽은 저장소 파일에 실제 기록된 내용. |
| **[검토 판단]** | 자료에 대한 이번 해석. 보고서 원문의 결론과 구분한다. |
| **[운영 조정]** | 앞으로 적용할 절차·우선순위. 이미 실행됐다는 뜻이 아니다. |
| **[현장 확인]** | 실행 서버의 manifest·산출물·프로세스에서 운영자가 확인해야 하는 사항. |

근거는 첨부 통합 분석 [S1], s3 fitting 검토 [S2], 기존 NA104 계획 [S3], 이번 조회의 s2/s3 큐 [S4–S5], Q00 config [S6]다. 정확한 파일과 식별 정보는 §15에 둔다.

**시점 주의:** S1은 s2 3/48 완료·R1 진행, S2는 s3 T00/Q00/Q04 완료·Q06 진행 당시의 보고서다. 이 숫자는 보고서 스냅샷이지 현재 서버 진행률이 아니다. 이번에는 큐와 config를 읽었으며, 실제 work_dir 전체와 현재 GPU 프로세스를 실시간 감사하지 않았다. 따라서 아래의 “추가·앞당김”은 **이미 완료·실행 중인 case를 제외한 잔여 작업**에 적용한다.

---

## 1. 변경 결정표

| 항목 | 결정 | 운영자가 적용할 내용 |
|---|---|---|
| 연구 목적 | **유지** | 압축이 아니라 동일 용량 학습 파이프라인의 fitting 강화. Teacher는 참조 신호원이다. |
| Backbone | **유지** | Teacher·Student 모두 W104, depth=[1,2,2]. |
| 입력·전방 계산 | **유지** | PAN 1ch+보간 MS 8ch, HRMS residual 복원. Aligner·sampling 보정 없는 NA-STRICT. |
| 기존 기본 R-case | **유지** | N0/R0/R1/R2/R3의 수식·α·β·τ 규칙을 첫 대응 비교 도중 변경하지 않는다. |
| 주 지표 | **HQNR로 명시 확정** | best_rr_val 또는 ERGAS를 주 판정으로 적은 과거 문구는 이번 캠페인의 지침에서 대체한다. 원문은 역사 자료로 보존한다. |
| Teacher checkpoint | **기존 학습 target 유지** | 실행 manifest의 실제 Teacher hash와 tag를 고정. 평가용 재선택으로 Teacher 파일을 교체하지 않는다. |
| 평가 간격 | **비교 조건 정비** | 기존 평가 이력을 감사하고, 비교 대상에 공통인 10-epoch 시점 또는 동일 update 격자를 사용한다. |
| 최고 HQNR 저장 | **원본 보존+파생 결과 추가** | 원래 best와 공통 격자의 best를 별도 이름으로 기록한다. 덮어쓰지 않는다. |
| Loss 진단 | **보완** | scalar coefficient/loss/gradient를 구분하고 hard·soft를 따로 측정한다. |
| N0 진단 | **실제 산출물 확인·필요 시 복구** | eval-only Teacher를 통해 공통 pixel/stat fitting bin을 남긴다. 학습에는 Teacher를 넣지 않는다. |
| s2 순서 | **핵심 비교 우선** | 기본 R 사다리와 GV 정보원 분리 먼저. |
| s3 순서 | **부분 수정** | 기존 R0/R1/R2 config를 해당 서버의 대조에 추가. Q35–37을 통계 모드 전수 확장보다 앞에 둔다. |
| A/B/C·window·다중 통계 | **보존하되 조건부 진행** | 전부 폐기하지 않는다. 부모 방법·표현의 HQNR 결과가 나온 뒤 우선순위를 갱신한다. |
| G계열 정합 KD | **이번 캠페인 제외 유지** | 현재 모델에는 학습할 이동량 출력이 없다. 별도 정합 연구로 유지한다. |
| 장기 학습·반복 | **유지** | 캠페인 시간 상한 없음. 대응 방법끼리 학습량·평가 기회·schedule을 맞춘다. |

**지금 새로 확정하는 loss 수식은 없다.** 핵심 수정은 “평가와 실험 순서”다. 현재 R3를 R0/R2로 일괄 바꾸거나, β를 전 run에서 올리는 변경은 하지 않는다.

---

## 2. 이번 조정의 수치 근거

### 2.1 s2 NA104: R0는 공통 평가 간격에서 작은 HQNR 개선 후보

**[자료 확인, S1 §9.2]**

| 비교 | Q00 N0 | Q01 R0 | R0−N0 |
|---|---:|---:|---:|
| 원래 기록된 best HQNR | 0.95326 | 0.95300 | −0.00026 |
| 보고서의 공통 10-epoch 격자 재선택 HQNR | **0.95211** | **0.95300** | **+0.00089** |

S1은 Q00을 5 epoch마다, Q01을 10 epoch마다 평가했다고 기록한다. Q00의 원래 best는 epoch 105이고, 공통 격자에서는 epoch 140이 선택된다.

**[검토 판단]** 작은 개선 후보가 생긴 것이지, R0의 효과가 확정된 것은 아니다. 동시에 “R0가 ERGAS에서 좋아졌으니 우선”이라는 종전의 해석은 사용하지 않는다. 여기서 우선 검증할 근거는 **공통 격자의 HQNR +0.00089**다.

**[현장 확인]** 재선택 HQNR의 evaluator·view·scene list와 실제 checkpoint를 연결한다. S1 표의 Dλ/Ds도 재선택 checkpoint에서 나온 것인지는 별도로 확인해야 한다. HQNR만 새 값으로 바꾸고 다른 지표는 원래 best 값을 유지하지 않는다.

### 2.2 s3 NA104: 현재 R3는 N0보다 HQNR이 낮다

**[자료 확인, S2 §6]**

| Case | HQNR | Dλ | Ds | 기록된 best epoch |
|---|---:|---:|---:|---:|
| T00 Teacher | 0.9497 | 0.0218 | 0.0291 | 210 |
| **Q00 N0** | **0.9544** | 0.0217 | **0.0245** | 200 |
| **Q04 R3** | **0.9523** | **0.0214** | 0.0270 | 110 |

R3−N0의 기록상 HQNR 차이는 **−0.0021**이다. Dλ는 0.0003 감소했지만 Ds는 0.0025 증가했다.

**[검토 판단]** 같은 모델 크기의 KD 전체가 실패했다는 의미가 아니다. **현재 R3의 설계가 HQNR에 유리하다는 근거가 없다**는 의미다. R3를 모든 후속 실험의 확정된 기반으로 취급하지 않는다. 이 표도 실제 공통 평가 격자·동일 evaluator를 확인한 뒤 새 비교표에 편입한다.

### 2.3 과거 W112: 통계 항이 분광·공간 균형을 바꾼 단서

**[자료 확인, S1 §9.1; 현재 W104와 다른 캠페인]**

| W112 구성 | HQNR | Dλ | Ds |
|---|---:|---:|---:|
| Frozen aligner+N0 | 0.9547 | 0.0219 | 0.0239 |
| R3 | 0.9486 | 0.0214 | 0.0306 |
| R3+GV-H | 0.9516 | 0.0261 | 0.0229 |
| R3+GV-AD | 0.9489 | 0.0238 | 0.0280 |

**[검토 판단]** GV-H는 해당 R3 대비 HQNR을 0.0030 회복했으나 N0를 넘지 못했다. GV-AD는 GV-H보다 HQNR이 0.0027 낮았다. 따라서 W104에서도 **GT 통계 H → Teacher-error 통계 WH → 통계 AD**를 분리할 이유가 있다. 이것을 W104 통계 방식의 성능 결과로 전이하지 않는다.

### 2.4 Loss 관측에서 가져올 것과 가져오지 않을 것

| 자료의 관측 | 이번 운영에 적용하는 범위 |
|---|---|
| S1: 후반 soft/hard loss 비율이 R0 약 2.354%, R3 약 0.056% | 신호 규모의 진단 근거. 두 숫자는 NA104와 W112이므로 같은 아키텍처의 완전한 대조로 해석하지 않는다. |
| S2: s3 GV-H의 weighted stat/rec은 약 1.75–2.0% | GT 통계 항이 scalar objective에 들어간다는 근거. gradient나 HQNR 기여율은 아니다. |
| S1: Teacher-error 가중치가 calibration p10~p99에서 약 1.26~1.88 | 배치 평균이 일정하다고 픽셀 가중치까지 동일한 것은 아니다. 실제 run 분포도 진단한다. |
| S1과 S2에서 β 증폭·재가중의 해석이 다름 | β 또는 공간 재가중 효과가 이미 기각됐다고 처리하지 않는다. 기존 대조군으로 확인한다. |
| Train L1이 비슷하거나 R0에서 조금 증가 | HQNR 성능을 자동 기각하지 않는다. 반대로 이를 fitting 강화의 증거로 포장하지도 않는다. |

**자료 해석의 한계:** 동일 크기 Teacher의 평균 HQNR이 낮다는 것만으로 참조 신호의 가치를 0으로 간주하지 않는다. 반대로 일반적인 정규화 가능성만으로 KD의 유효성을 인정하지 않는다. 판단은 대응 N0 대비 HQNR 결과로 한다.

---

## 3. 공통 평가 계약: HQNR을 바꾸지 말고, 비교 기회를 맞춘다

### 3.1 주 평가의 정의

**[사용자 결정+현행 경로 유지]**

- 대상: 원본 FR 논문 세트 `.mat` 20장.
- View: `raw_original`, 원 PAN·원 MS reference, 현재 evaluator의 원래 영역.
- 집계: 장면별 HQNR을 구한 뒤 평균.
- Checkpoint 주 선택: `best_hqnr`. 실제 기존 tie-break도 manifest에서 확인해 동일하게 유지한다.
- `best_rr_val`, ERGAS, SAM, JQM은 보조 기록이다. HQNR 순위를 뒤집는 숨은 선택 규칙으로 쓰지 않는다.
- `HQNR(V64)`나 다른 ROI의 점수로 원본 HQNR을 대체하지 않는다.
- 현재 no-align 입력·전처리는 그대로 유지한다. HQNR에 맞추기 위해 output/PAN을 사후 정합하지 않는다.

$$
Q_i=(1-D_{\lambda,i})(1-D_{s,i}),\qquad
Q=\frac{1}{20}\sum_{i=1}^{20}Q_i.
$$

평균 Dλ와 평균 Ds의 곱은 위 장면별 곱의 평균과 일반적으로 다르다. [S2 §10]

### 3.2 세 가지 HQNR을 역할에 맞게 기록

| 이름 | 정의 | 역할 |
|---|---|---|
| `hqnr_best_original` | 해당 run이 원래 평가 일정에서 선택한 최고점 | 기존 결과 보존·추적 |
| **`hqnr_best_common_grid`** | 대응 run들이 공통으로 가진 평가 시점에서 같은 selector로 선택한 최고점 | **이번 대응 비교의 주 HQNR** |
| `hqnr_plateau`, `hqnr_last` | 사전 고정 후반 구간의 평균과 마지막 checkpoint의 HQNR | 최고점의 지속성·학습 경로 설명 |

**[운영 조정]** 기본 50K block의 plateau는 `40,000 ≤ update ≤ 50,000`의 공통 평가 시점으로 정한다. 새로 정하는 운영 규칙이지 과거 파일이 이 구간으로 작성됐다는 뜻은 아니다. 시간 평균의 산포는 trajectory 변동이며 독립 seed의 표준오차가 아니다.

Best와 plateau가 엇갈리면 `PEAK_ONLY` 또는 `TRAJECTORY_TRADEOFF`로 명시한다. **Plateau를 사후 주 지표로 교체해 best-HQNR 개선을 삭제하지 않는다.** 다만 안정성이 미확인된 최고점 하나만으로 모든 후속 조합을 확대하지 않는다.

### 3.3 공통 격자 적용 절차

1. 각 run의 실제 `epoch`, `global_update`, 평가 성공 여부, metric ID를 읽는다. config 주석만 보지 않는다.
2. 비교할 block의 동일한 학습 horizon 안에서 공통 후보 집합을 정한다. epoch당 update가 동일하면 10-epoch 격자를 사용한다. 다르면 global update 기준으로 대응 가능성을 다시 확인한다.
3. 5-epoch run은 10-epoch 시점만 남긴다. 누락된 점수를 선형 보간하거나, 한 run에만 유리한 시점을 추가하지 않는다.
4. 같은 HQNR scorer와 동일 tie-break로 각 run의 common-grid best를 선택한다.
5. 선택한 checkpoint의 hash와 출력 파일을 연결한다. 그 checkpoint에서 HQNR·Dλ·Ds·보조 지표를 함께 export한다.
6. 원래 `best_hqnr`와 그 파일·symlink·manifest는 그대로 둔다. 공통 격자 결과는 별도의 분석 산출물이다.

큐에 기록된 기존 유틸리티 호출 예시는 아래와 같다. **먼저 해당 서버의 스크립트가 읽기 전용 분석인지, 어느 scorer를 쓰는지, 어떤 산출물을 저장하는지 확인한 뒤 사용한다.** 이 문서는 스크립트를 실행하거나 수정한 결과가 아니다. [S4–S5]

```bash
python tools/best_on_grid.py --grid 10 <run>
```

로그에 점수만 있고 해당 checkpoint가 없다면 `score_only`로 표시한다. 그 epoch의 새 영상·분해 지표를 생성했다고 기록할 수 없다. 필요한 weight가 없으면 그 한계를 남기고, 공통으로 보존된 checkpoint 비교 또는 별도 재실행으로 해결한다.

**평가 간격 조정은 선택 기회만 맞춘다.** 평가가 전역 RNG·data order를 소비했다면 학습 궤적 차이까지 제거되는 것은 아니다. 평가·diagnostic RNG가 학습에 영향을 주지 않는지 별도로 감사한다.

### 3.4 Teacher를 평가 재선택과 함께 바꾸지 않는다

현재 확인한 Q00 config는 Teacher `T104_v1`, `T00/best_hqnr`, `eval_only: true`를 명시한다. [S6]

- 기존 KD run은 실제 학습에 사용한 Teacher hash를 계속 기준으로 삼는다.
- 공통 격자 재선택은 Student 결과를 비교하는 **평가 조치**다.
- Teacher도 새 공통 격자로 다시 고르고 싶다면 **새 Teacher cohort**로 분리한다. τ·통계 calibration과 대응 baseline을 다시 연결한다.
- Teacher의 `best_hqnr`를 `best_rr_val`로 바꾸는 것은 이번 지시가 아니다.
- TCOPY/CONT의 parent tag도 별도로 확인한다. 큐 설명과 actual config가 다르면 기존 run 이력을 유지하고 차이를 기록한다.

### 3.5 탐색과 최종 검증의 구분

현재처럼 같은 FR 20장으로 checkpoint 선택과 보고를 하면 그 세트에 적응한 탐색 결과다. 원문은 이 한계를 명시한다. [S2 §6, §10]

HQNR을 계속 주 지표로 사용하되 최종 유력안에는 별도 FR 자료·센서 또는 추가 장면에서 같은 HQNR 프로토콜의 검증을 붙인다. 이 조치는 ERGAS selector 도입을 뜻하지 않는다. 현재 20장의 원본 결과는 삭제하지 않는다.

---

## 4. 유지할 기본 loss와 정확한 대조

기호는 최종 HRMS의 밴드 평균 L1 map이다.

$$
e_T=\operatorname{mean}_c|\hat Y_T-Y|,\quad
e_S=\operatorname{mean}_c|\hat Y_S-Y|,\quad
K=\operatorname{mean}_c|\hat Y_S-\operatorname{sg}(\hat Y_T)|.
$$

$$
d_T=\frac{e_T}{e_T+\tau_R},\qquad
a_T=\frac{[e_S-e_T]_+}{e_S+\varepsilon}.
$$

Routing용 Teacher·Student 오차와 gate는 detach한다. 실제 hard/soft의 Student 오차는 live tensor로 유지한다. [S1 §3; S3 §4]

| NA104 ID | 방식 | 목적함수 | 우선 대조 |
|---|---|---|---|
| Q00 | N0 | `mean(eS)` | 공통 기준 |
| Q01 | R0 | `mean(eS + β K)` | Q01−Q00: 일반 output KD |
| Q02 | R1 | `mean((1+αdT)eS)` | Q02−Q00: 실패 지도 기반 hard |
| Q03 | R2 | `mean((1+αdT)eS + β(1−dT)K)` | Q03−Q02: Teacher-error soft 추가 |
| Q04 | R3 | `mean((1+αdT)eS + β(1−dT)aT K)` | Q04−Q03: 상대우위 gate; Q04−Q02: output KD 추가 |

**기존 α=1, β=0.1, train-only Teacher error 기반 τ 규칙을 기본 비교에서 유지한다.** 수치를 개선하려고 기존 ID의 의미를 바꾸지 않는다. R0→R1은 한 요소만 바뀌는 대조가 아니다.

R1의 soft=0, 통계 H의 soft=0은 정상이다. R1은 Teacher error를 쓰지만 Teacher output을 soft target으로 모방하지 않는다. R3의 soft가 작아도 0이라고 기록하지 않는다.

---

## 5. s2에 전달할 실행 조정

### 5.1 s2의 역할

**기본 reconstruction의 작동 원리와 GV의 정보원을 분리한다.** 현재 s2 큐에는 필요한 R 사다리와 GV 대조가 이미 있다. 새 학습식을 추가하기보다 우선 분석 block을 만든다. [S4]

### 5.2 다음 run을 선택할 순서

아래는 전체 큐를 무조건 재시작하는 명령이 아니다. **DONE은 평가를 보완하고, RUNNING은 완료까지 보존하며, PENDING만 다음 순서로 배치한다.**

| 단계 | Case | 할 일 |
|---|---|---|
| S2-0 | T00·Q00 및 완료 run | §3의 HQNR 격자·hash·산출물 확인. 현재 학습과 별개로 가능한 사후 분석 진행. |
| **S2-1** | **Q00→Q01→Q02→Q03→Q04의 잔여분** | 기본 R 사다리를 같은 50K 조건으로 완성. R0·R3만 보고 결론내리지 않는다. |
| **S2-2** | **Q05 N0+GV-H, Q11 N0+EDGE-H** | GT gradient variance의 고유 이득을 일반 경계 loss와 분리. |
| **S2-3** | **Q06 GV-H, Q07 GV-WH, Q10 GV-AD, Q09 GV-FIX** | 기존 R3 위에서 통계의 GT 목표·실패 지도·Teacher 목표를 분리하는 대표 비교. 이미 시작한 순서는 유지해도 된다. |
| S2-4 | Q08 GV-T, Q12 R3+EDGE-H | Teacher-only 통계와 직접 edge의 대응을 보완. 핵심 결과 뒤로 둘 수 있다. |
| **판정 회의점 R** | Q00–Q04 | §9의 R-branch 판정. 이후 모든 확장을 R3로 고정할지 결정하지 말고 결과를 반영. |
| **판정 회의점 V** | Q05/11 및 대표 통계 block | §9의 통계 판정. 유력한 감독 형태만 후속 확대. |

S2-3은 R3 기반 원안을 해석할 최소 block이다. R3가 불리하다는 이유로 진행 중인 Q06/Q10을 버리지 않는다. 반면 이 대표 block 뒤의 모든 복합 case를 자동으로 열 필요도 없다.

### 5.3 s2에서 조건부로 올릴 대조군

| 조건 | 우선할 기존 case | 해석 시 주의 |
|---|---|---|
| R1이 N0와 다른 HQNR 결과를 보임 | `CTLHSCALE` | 공간 위치별 weighting과 전체 loss 배율을 분리. |
| R3의 위치별 routing 효과를 주장하려 함 | `CTLRSHUF` | d/a의 위치 의미를 검사. R1만의 순수 shuffle 대조라고 부르지 않는다. |
| R0/R2와 R3 차이가 큼 | `CTLBETA03/05`, 필요 시 `CTLTAU05/20` | 각각 강도 또는 scale만 바꾼 기존 사전 목록. 모든 인자를 동시에 변경하지 않는다. |
| 유력 통계 H/WH/AD가 강도에 민감해 보임 | `CTLLAMV03/30` | 실제 config의 multiplier를 확인. ID 문자열에서 계수를 추측하지 않는다. |
| A 방향 gate의 HQNR 개선 후보가 생김 | `CTLAMASS`, `CTLASHUF` | 단순 soft 총량 감소와 올바른 방향 선택을 분리. |
| B 성분 gate의 개선 후보가 생김 | `CTLBMASS`, `CTLBSHUF` | 같은 방식으로 통계 soft를 분해. |

β 증가가 이미 무효라고 확정하지 않는다. 다만 증가한 soft loss가 아니라 **HQNR의 변화**로 판정한다.

### 5.4 s2에 보내는 짧은 실행 메시지

```text
[NA104 / s2]
골격·Teacher·기존 loss는 유지. 진행 중 run은 원래 설정대로 완료.
1) T00/Q00 및 완료 결과의 HQNR 공통10격자·checkpoint 연결을 먼저 감사.
2) Q00~Q04 잔여분을 완료하고 HQNR, Dλ, Ds, plateau를 한 표로 보고.
3) Q05(N0+GVH), Q11(N0+EDGEH)를 먼저 확인.
4) R3 기반 GV-H/WH/AD/FIX 대표 block까지만 우선 완성.
5) 이후 CTL·A/B/C·장기 확장은 R/V 판정 후 승인 목록으로 진행.
ERGAS는 보조. 원래 best·Teacher hash를 재선택 분석으로 덮어쓰지 말 것.
```

---

## 6. s3에 전달할 실행 조정

### 6.1 s3의 역할

**GT-only 통계 표현의 다양성은 유지하고, 아직 검증되지 않은 R3 위의 전수 조합은 늦춘다.** 또한 s3 자체에서 일반 KD와 gate 효과를 판단할 수 있도록 R 사다리의 빈 부분을 메운다.

이번에 읽은 s3 큐에는 T00/Q00/Q04와 Q06/Q10 등은 있지만 **Q01(R0), Q02(R1), Q03(R2)**가 없다. Q35–Q37은 IV/GC/SC의 R3 기반 15개 모드 뒤에 있다. [S5]

### 6.2 추가·앞당김 구분

| 조치 | Case | 이유 |
|---|---|---|
| **해당 서버 큐에 추가** | **Q01 R0, Q02 R1, Q03 R2** | 새 알고리즘이 아니라 기존 config를 s3에서 대응 실행. R3의 손해가 hard 재가중인지 soft/gate인지 분리. |
| **앞당김** | **Q35 N0+IV-H, Q36 N0+GC-H, Q37 N0+SC-H** | R3에 의존하지 않고 통계 표현 자체를 비교. |
| **필요 시 지역 대조 추가** | **Q07 R3+GV-WH** | s3 Q06↔Q10의 H→AD에서 hard 재가중과 Teacher 통계 전달을 분리. |
| 선택적 지역 기준 보완 | Q05 N0+GV-H, Q11 N0+EDGE-H | s3 내에서 GV·edge와 IV/GC/SC를 직접 비교할 때 사용. s2 값을 s3 대조군으로 무단 대체하지 않음. |
| 후순위 이동 | Q20–Q34 중 미시작 모드, Q38–Q47·VX 묶음 | 정보원·표현의 대표 비교 이후 필요한 조합만 실행. |

Q01/Q02/Q03은 s2에 존재하는 동일 NA104 ID를 사용하되 **server_id를 포함한 결과 key**로 분리한다. 동일 서버에서 같은 run ID의 결과가 이미 있다면 중복 실행하지 않는다. 경로·Teacher·학습 조건이 다르면 같은 case ID를 쓰더라도 cohort/version을 구분한다.

### 6.3 권장 pending 순서

```text
[현재 RUNNING은 그대로 완료]
→ T00/Q00/Q04 기존 산출물 점검
→ Q01(R0) → Q02(R1) → Q03(R2) [미실행분만]
→ Q35(N0+IVH) → Q36(N0+GCH) → Q37(N0+SCH)
→ Q07(GVWH) [Q06/GV-H 및 Q10/GV-AD와 local attribution이 필요한 경우]
→ R/V 결과 검토
→ 선택된 표현의 H/WH/FIX/AD 대표 비교
→ 승인된 방향 gate·window·결합·반복
```

운영 상황상 s2에서 R 사다리 결과가 곧 나오고 s3의 통계 block 준비가 끝났다면 Q35–Q37을 먼저 실행해도 된다. **변경되는 것은 pending dispatch 순서이며 각 run의 loss·horizon은 아니다.** 실제 순서는 적용 회신에 남긴다.

### 6.4 s3 통계 표현별 재개 대조

| 표현 | Teacher-free anchor | 기존 R3 기반 H | WH | T | FIX | AD |
|---|---|---|---|---|---|---|
| IV | Q35 | Q20 | Q21 | Q22 | Q23 | Q24 |
| GC | Q36 | Q25 | Q26 | Q27 | Q28 | Q29 |
| SC | Q37 | Q30 | Q31 | Q32 | Q33 | Q34 |

우선 `N0+H−N0`로 통계 표현의 HQNR 기여를 본다. 그 다음 유력 표현에서 H/WH/FIX/AD를 비교한다. T는 Teacher 통계가 직접적인 target으로 유용한지를 따로 보는 대조다.

**GT-only가 무효라고 Teacher 통계까지 수학적으로 무효인 것은 아니다.** 다만 유용성의 단서가 전혀 없는 표현에 모드 5종과 window·gate를 모두 자동 적용하지 않는다. 기본 표현 하나의 부정 결과도 variance 접근 전체의 기각으로 확대하지 않는다.

### 6.5 s3에 보내는 짧은 실행 메시지

```text
[NA104 / s3]
진행 중 Q06 등은 유지·완료. 현재 R3의 HQNR은 N0보다 낮으므로 R3를 확정 winner로 두지 말 것.
1) 기존 Q00/Q04의 공통 평가 격자와 동일 evaluator를 점검.
2) 기존 config Q01(R0), Q02(R1), Q03(R2)를 s3 대조에 추가.
3) Q35/Q36/Q37(N0+IVH/GCH/SCH)을 R3 통계 전수 확장보다 앞당김.
4) Q06↔Q10의 통계 soft 해석에는 Q07(GVWH) local 대조를 보완.
5) 결과 검토 전 Q38~Q47·다중 window·복합 gate는 자동 소진하지 않음.
최종 채택 기준은 HQNR. ERGAS 개선만으로 채택하거나 악화만으로 기각하지 말 것.
```

---

## 7. R3 기반 통계가 불리할 때의 재기반화 규칙

### 7.1 기존 run을 수정하지 않고 새 mini-block으로 분리

**[운영 조정]** R 사다리에서 유력한 reconstruction을 `R*`라고 쓰자. 아직 R0·R1·R2·R3 중 어느 것도 확정하지 않는다. N0도 유효한 기준으로 남는다.

R3보다 다른 R*의 HQNR이 일관되게 유리하면, 유력 통계 1–2종만 새 R* 위에서 비교한다.

| 새 비교의 역할 | Loss 구성 |
|---|---|
| REC anchor | R*만 사용 |
| GT 통계 | R* + STAT-H |
| 실패 지도 | R* + STAT-WH |
| Teacher 통계 | R* + STAT-AD |
| Gate 대조 | 필요할 때 R* + STAT-FIX 또는 T |

이것은 **조건부 조정안**이다. 이번 지시로 전체 Q번호의 R3를 일괄 치환하는 것이 아니다. 새 ID/version을 발급하고 `derived_from_case`, `changed_fields`, `teacher_hash`, `calibration_hash`를 남긴다. `Q20_R3`를 수정해 놓고 동일 Q20으로 다시 업로드하지 않는다.

### 7.2 Calibration을 동시에 바꾸지 않는다

첫 재기반화 비교에서는 가능한 한 기존의 표현별 train-only calibration과 λ를 그대로 사용해 reconstruction 변경 효과를 분리한다. Teacher·domain·창·통계 normalization이 바뀌지 않았는지 확인한다.

새 R*에서 다른 calibration이 필요하다고 판단하면 다음처럼 **별도 단계**로 처리한다.

- 원래 calibration을 쓴 결과를 보존한다.
- 새 pilot과 산출 규칙을 명시한다.
- 해당 표현의 H/WH/FIX/AD 대응 block에 같은 λ 규칙을 적용한다.
- AD에만 유리한 재calibration을 해 놓고 H보다 좋다고 해석하지 않는다.
- 결합 GV+SC에는 단독에서 쓴 두 항의 scale을 먼저 사용하고, 총량 대조는 별도 case로 둔다.

Train calibration은 loss scale 설정이다. FR 20장의 장면별 결과로 픽셀 gate나 calibration target을 만들지 않는다.

---

## 8. A/B/C·기타 case는 어디까지 유지할 것인가?

| 계열 | 현재 상태 | 다음 실행 조건 |
|---|---|---|
| A-SIGN/COS/CAP | **정의 보존, 전수 확대 후순위** | 기본 output KD에 HQNR 이득 또는 유해 성분의 단서가 있을 때 대표 gate+총량 대조. |
| A-BANDADV | **조건부 유지** | 평균 밴드 gate가 놓친 Teacher 우세 성분을 살리는 효과를 확인할 때. 단순 추가 감쇠와 구분. |
| B-SIGN/CAP | **유력 통계 위에서 실행** | 해당 STAT-AD/FIX가 유효하거나 통계 모방의 방향 충돌이 의심될 때. |
| B-COMPADV | **조건부 유지** | 통계 평균보다 성분별 선택의 가치가 있을 때. |
| C-NASENS / CS 계열 | **후순위·분리 유지** | output KD의 가치와 입력 위치 민감성 관련성이 확인된 뒤. NA-TSENS를 표준 NA-STRICT와 구분. |
| G-STRUCT/G-EQ/G-GEO/G1–G5 | **이번 no-align 실험 제외** | 별도 정합 Teacher와 trainable aligner가 준비된 캠페인에서 검토. |
| IV/GV/GC/SC | **표현의 다양성 유지** | 먼저 GT-only 대표 비교, 그 뒤 유력 표현의 모드 확대. |
| M2·STD·LOG·residual·window3/7/multi | **후보로 보존** | 표현·정보원 효과가 확인된 뒤 같은 부모 조건으로 비교. |
| GV+SC, AB/AC/BC/ABC | **단독 효과 뒤로** | 각 단독의 대조가 확보되고 조합 질문이 명확할 때. |
| TCOPY·CONT·LONG | **유지하되 유력 방법 중심** | 같은 parent/추가 budget의 GT-only 대조를 포함. |
| 과거 LR spectral KD | **후속 아이디어로 보존** | 기본 사다리·통계가 좁혀진 뒤 필요할 때 별도 명세. 현재 큐에 새로 자동 추가하지 않음. |

**중요:** soft가 작은 상황에서 감쇠 gate가 개선을 보이면 “새 지식을 더 전달했다”가 아니라 “유해한 모방을 줄였다”는 설명일 수 있다. 반대로 작은 soft가 HQNR에 도움이 될 수도 있다. 계수 크기만으로 효과를 미리 판정하지 않는다.

과거 LR spectral KD의 작은 HQNR 우세는 다른 캠페인의 관측이다. 현재 SC 통계와 동일한 방법으로 취급하거나, 현재 W104의 검증된 성과로 인용하지 않는다. [S2 §7]

---

## 9. Case 승급·보류 판정 규칙

### 9.1 성능과 작동 원리를 따로 판정

| 구분 | 질문 | 증거 |
|---|---|---|
| **성능 판정** | N0 또는 정확한 부모 대조보다 HQNR이 높은가? | 공통 격자 best HQNR, 동일 evaluator, 대응 seed |
| 안정성 | 개선이 특정 최고점·장면에만 의존하는가? | plateau/last, 장면별 ΔHQNR, 재평가 |
| Fitting 원리 | Teacher가 놓친 부분이나 구조 통계를 더 학습했는가? | 공통 teacher bin, 비가중 L1·통계/edge 오차 |
| KD 기여 | hard 재가중 외에 Teacher target이 추가 역할을 하는가? | R1↔R2/R3, STAT-WH↔AD, hard/soft gradient |

HQNR이 좋지만 L1이 같거나 나쁘면 **HQNR 개선 후보로 유지**한다. 다만 논문에서 train fitting 강화가 입증됐다고 쓰지는 않는다. HQNR이 낮지만 L1이 좋아지면 **fitting 진단 후보이지 현재의 성능 우승안은 아니다.**

### 9.2 분기표

| 확보된 관측 | 유지·승급할 방향 | 당장 확대하지 않을 방향 |
|---|---|---|
| R0/R2가 N0보다 좋고 R3만 낮음 | 고정/Teacher-error KD, 필요 시 상대우위 감쇠의 제한적 대조 | 현재 R3 기반 복합 gate 전수 실행 |
| R1이 N0보다 좋고 R3는 R1보다 낫지 않음 | Teacher-error hard fitting 및 CTLHSCALE | 직접 imitation의 세부 mask 다수 |
| R3가 R1/R2보다 HQNR이 좋음 | R3 및 A의 대표 방향/성분 비교 | 근거 없이 gate 전체 폐기 |
| STAT-H가 부모보다 좋고 AD가 WH를 넘지 못함 | GT 통계 또는 통계 실패 지도 | Teacher 통계 target을 전제로 하는 복합 B/C |
| 특정 STAT-AD가 H/WH/FIX보다 좋음 | 그 표현의 성분 gate·필수 총량 대조·반복 | 다른 모든 표현의 같은 조합 전수 실행 |
| Dλ 개선과 Ds 악화가 상쇄되어 HQNR 이득 없음 | 교환의 원인 진단·선택적 scale 대조 | 한 지표만 좋아졌다는 이유로 성능 winner 처리 |
| Best만 소폭 좋고 plateau/last는 불리함 | `PEAK_ONLY`, HQNR 기준 재현성 확인 | 안정적 개선으로 간주해 광범위 결합 |
| 여러 공정한 반복에서 기본 N0를 넘지 못함 | 해당 설정을 보류, target/학습 단계 재설계 검토 | 동일 target의 미세 mask 변형 반복 |

단일 seed에서 차이가 작다는 이유로 모든 방법이 같다고 선언하지 않는다. 반대로 소수점상 양수 하나를 확정 우승으로 부르지도 않는다. **과거 W96의 σ를 W104의 고정 유의성 문턱으로 사용하지 않는다.**

### 9.3 반복의 최소 단위

운영 제안은 **유력 방법+정확한 대조군을 최소 3 Student seed로 대응 반복**하는 것이다. 기존 계획의 예시는 `[1234, 2025, 777]`이며, 실제 시작 전 집합을 manifest에 고정한다. [S3 §3.3]

- 같은 Teacher를 쓰는 첫 반복과, Teacher seed를 바꾸는 다음 반복을 분리한다.
- s2·s3에서 같은 seed 1234를 한 번씩 돌렸다고 2-seed 결과라고 세지 않는다.
- 각 seed에서 초기 tensor와 data/augmentation 조건이 대응되도록 확인한다.
- 장면 20개에 대한 bootstrap 또는 장면별 승률은 장면 변동 분석이며, 학습 seed의 변동을 대신하지 않는다.
- 동일 FR 20장을 탐색에 반복 사용한 경우 신뢰구간도 선택에 조건부인 분석임을 명시한다.

---

## 10. 지금 보완할 진단과 구현 계약

### 10.1 필수 산출물: 기존 파일은 보존하고 누락만 보완

| 산출물 | 필수 내용 | 용도 |
|---|---|---|
| 실행 inventory | server, run ID/version, 상태, 현재/최종 update, commit, config·init·Teacher hash | 진행상태와 대조 자격 확인 |
| 공통 격자 HQNR 표 | original/common best, candidate 집합, 선택 epoch/update, checkpoint hash | 공정한 주 지표 |
| 장면별 FR 표 | scene ID, HQNR, Dλ, Ds, evaluator/view/hash | 공간·분광 교환 분석 |
| 후반 HQNR 요약 | 구간과 실제 평가 시점, mean/std/min/max/last | 지속성 |
| fitting bins | 공통 Teacher error 구간의 N0와 모든 비교군 결과 | 원래 fitting 논지 |
| loss 신호 분해 | hard/soft/통계의 계수·loss·gradient 분리 | 작동 원리 |
| calibration 기록 | source split/sample ID, Teacher hash, τ/λ/통계 창·domain | scale·정보원 혼동 방지 |

아래 파일 이름은 **새 보조 산출물 제안**이다. 기존 trainer가 이 이름을 자동 생성한다는 뜻은 아니다.

```text
analysis/hqnr_revision_20260911/
  run_inventory.csv
  common_grid_manifest.json
  hqnr_comparison.csv
  hqnr_per_scene.csv
  loss_signal_audit.csv
  dispatch_decisions.md
```

### 10.2 N0·GT-only의 Teacher 평가를 확인

[S1 §12]는 N0 bin 누락을 보고하지만, [S6]의 현재 Q00 config에는 `teacher.eval_only: true`가 있다. **보고 시점과 코드 준비 상태를 구분한다.** 현장에서는 `fitting_bins.csv`의 존재·행 수·Teacher hash·split을 확인한다.

없다면 완료 checkpoint를 고정한 상태로 평가만 추가한다. N0의 optimizer·loss에는 Teacher를 넣지 않는다. 통계 학습 OFF run에도 공통 진단용 GV window5 등을 동일하게 계산할 수 있으나, 이는 loss 추가가 아니다.

모든 run의 bin은 같은 Teacher, 같은 scene/pixel 집합, 같은 band reduction과 thresholds를 사용한다. Teacher가 바뀐 bin끼리 숫자를 직접 비교하지 않는다. 평가 데이터로 만든 진단 threshold는 학습 가중치로 되돌리지 않는다.

### 10.3 세 종류의 비율을 분리

$$
r_{coef}=\frac{\operatorname{mean}(w_K)}{\operatorname{mean}(w_H)},\quad
r_{loss}=\frac{L_K}{L_H+\varepsilon},\quad
r_{grad}=\frac{\|\nabla_\theta L_K\|}{\|\nabla_\theta L_H\|+\varepsilon}.
$$

세 숫자는 서로 다르다. 로그의 ratio-of-means와 mean-of-ratios도 동일하지 않으므로 집계 정의를 필드에 기록한다. `Teacher 몫`이라고 쓸 때 denominator가 hard인지 total인지 명시한다.

통계도 `λV·LH_stat`, `λV·LK_stat`을 나눠 측정한다. H mode의 soft=0은 정상이다.

### 10.4 Gradient 진단의 안전한 방식

**[운영 조정]** 대표 checkpoint가 보존된 초기·중기·후기에서 고정 train 진단 batch를 사용한다. 예시 시점은 1K/5K/10K/30K/50K이며, 저장된 시점만 사용할 수 있다. 없는 시점은 복원한 것처럼 기록하지 않는다.

- Teacher는 frozen/eval. `d`, `a`, 통계 gate는 detach.
- Student의 실제 hard·soft residual은 live.
- 파라미터 집합은 동일한 Student backbone으로 고정한다.
- hard/soft/stat 각각의 norm, hard–soft cosine을 기록한다. zero gradient이면 cosine은 정의 불가로 둔다.
- 진단에서 optimizer step을 하지 않는다. `.grad` 버퍼·model mode·RNG를 원 학습에 누출시키지 않는다.
- 완료 run은 offline 진단을 우선한다. 진행 중 run에 계측 코드를 넣을 때에는 학습 경로의 동일성을 검사하고 계측 버전을 기록한다.
- 파라미터 gradient는 최적화 입력의 진단이며 AdamW의 실제 update 기여율과 같다고 부르지 않는다.

추가 계측으로 학습 중인 모든 run을 강제 재시작하지 않는다. 비용은 training cost와 별도로 기록한다.

### 10.5 Difficulty 분포

Teacher-error `d`와 `a`의 평균뿐 아니라 pixel/std 및 p10/p50/p90/p99를 동일 aggregation 단위에서 기록한다. `wH` batch 평균의 시간 변동과 batch 내부 pixel 분산을 혼동하지 않는다. Teacher 우세 비율과 우세 마진을 분리한다.

$$
A_T=\operatorname{mean}[(e_S-e_T)_+],\qquad
A_S=\operatorname{mean}[(e_T-e_S)_+].
$$

이 값은 학습 신호의 상보성 진단이지 HQNR 성능의 대체 판정식이 아니다.

---

## 11. HQNR 분해: 같은 checkpoint·같은 장면으로 계산

대조군 b와 후보 m의 같은 장면에서

$$
\delta_\lambda=D_{\lambda,m}-D_{\lambda,b},\qquad
\delta_s=D_{s,m}-D_{s,b}
$$

라 두면 다음은 위 HQNR 식의 대수 전개다.

$$
Q_m-Q_b
=-(1-D_{s,b})\delta_\lambda
 -(1-D_{\lambda,b})\delta_s
 +\delta_\lambda\delta_s.
$$

**[운영 조정]** 장면별 세 항을 계산한 뒤 각각 평균해 `spectral_contribution`, `spatial_contribution`, `interaction`으로 기록한다. 평균·반올림된 Dλ/Ds만으로 정확한 HQNR 기여도를 복원하지 않는다.

이 분석은 **왜 점수가 변했는지**를 설명한다. Ds 악화만으로 blur·과도한 sharpness·물리적 오정합 중 하나가 확정됐다고 말하지 않는다. 같은 고정 scene/ROI의 경계·ringing·band bias 진단을 함께 남긴다.

원인 분석은 재선택된 checkpoint끼리 할 수 있고, 같은 update끼리 추가 진단할 수도 있다. 두 분석의 checkpoint 정책을 명확히 구분한다. 어느 경우에도 한 run의 epoch 105 Dλ/Ds와 epoch 140 HQNR을 섞지 않는다.

---

## 12. 안전한 큐 변경·재개·결과 관리

### 12.1 상태에 따른 처리

| 현재 상태 | 처리 |
|---|---|
| DONE | 원 결과 보존. 비교 자격 감사·누락 평가만 수행. 필요 없는 중복 학습 금지. |
| RUNNING | 원 설정·Teacher·schedule로 완료. 명백한 오류 외에는 이 문서를 이유로 중단하지 않음. |
| PENDING | 위 우선순위로 이동·보류. 원래 config의 알고리즘 정의는 유지. |
| FAILED/INVALID | 오류와 영향 범위 기록 후 새 attempt/version. 실패 흔적 삭제 금지. |
| BLOCKED | Teacher·pilot·대조·evaluator 등 dependency를 명시. 무조건 전체 캠페인을 막지 않음. |

원래 큐를 보존한 채 `pending_hqnr_priority`와 `deferred_after_review`를 분리하는 방식을 권한다. 다만 실제 worker가 시작 시 큐를 통째로 읽는지, 한 줄씩 재조회하는지 먼저 확인한다. 파일 순서만 바꿨다고 running worker의 다음 작업까지 바뀌었다고 가정하지 않는다.

현재 자식 학습 프로세스를 종료하는 명령을 쓰지 않는다. 필요하면 현 run 완료 경계에서 dispatcher만 교체한다. 변경 전후 queue diff와 적용 시각을 기록한다.

### 12.2 결과 key와 중복 방지

최소한 다음을 합친 식별을 사용한다.

```text
server_id / campaign_id / run_id / version / student_seed
teacher_checkpoint_sha / init_sha / calibration_sha
selection_policy_id / evaluator_hash / dataset_hash / view / grid_id
```

`T104_v1`이라는 이름이 같아도 s2·s3의 weight가 같은 것은 아니다. local Teacher hash와 baseline을 먼저 연결한다. 같은 이름의 시트 행을 서버 간 결과로 덮어쓰지 않는다.

### 12.3 시간 제한 없음의 의미

- 총 캠페인 wall-clock 상한은 두지 않는다.
- 기본 비교는 각 50K와 동일한 schedule로 한다.
- 2N/4N·continuation은 원래 50K block과 분리한다.
- 100K에서 얻은 최고 HQNR을 50K의 N0 best와만 비교해 KD 효과로 부르지 않는다.
- schedule horizon 변경, optimizer reset/restore, teacher-copy와 scratch를 각각 기록한다.
- KD의 추가 teacher forward·진단 비용을 측정하고, 유력안에는 같은 추가 시간의 GT-only 대조를 남긴다.

### 12.4 중간 변경 정책

수식·Teacher·α/β/τ·통계 표현·λ·입력 프로토콜을 변경하면 새 run 또는 명시적인 새 phase로 기록한다. 부모 checkpoint의 hash와 optimizer/scheduler 정책을 남기고 같은 parent에서 변하지 않은 continuation 대조를 붙인다.

**단순한 queue 순서 변경·사후 평가 보완은 새 학습 방법이 아니다.** 기존 학습의 결과를 무효 처리하지 않는다.

---

## 13. 적용 완료 회신 형식

각 서버에서 다음 항목을 채운 짧은 회신을 받으면 된다. 빈 값은 추측하지 말고 `미확인`으로 둔다.

```text
[NA104 HQNR 조정 적용 회신 / s2 또는 s3]
적용 시각:
저장소 commit / pending queue hash:
현재 RUNNING / update:
완료 case:
원본 best 보존 여부:
실제 evaluator / FR dataset / scene list hash:
기존 eval 간격과 적용할 common grid:
Teacher run / tag / checkpoint hash:
Student init / calibration hash:
Q00 fitting_bins 존재 및 공통 Teacher 확인:
공통10격자 HQNR 표 생성 여부:
재선택 checkpoint의 Dλ/Ds 연결 여부:
추가한 local 대조 case:
다음 실행 case(최대 5개):
후속 review 전 보류 묶음:
확인된 문제 또는 재학습이 필요한 예외:
```

다음 판정 회의에서는 **HQNR 결과표 하나와 기여도 진단표 하나**를 분리해서 전달한다.

### HQNR 결과표 권장 열

| 서버/버전 | Case | Teacher hash | Common-grid best | ΔHQNR 대조 | 선택 ep/update | Dλ | Ds | Plateau | Last | Scene 승/패/동률 | Seed |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| 미입력 | 미입력 | 미입력 | — | — | — | — | — | — | — | — | — |

### 기여도 진단표 권장 열

| Case/시점 | Plain GT L1 | Hard | Soft | λV·Stat-H | λV·Stat-K | Coef ratio | Loss ratio | Grad ratio | Hard/soft cosine | d/a 분위 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 미입력 | — | — | — | — | — | — | — | — | — | — |

두 표의 숫자를 같은 목적의 지표로 읽지 않는다. 학습 신호가 커진 것은 진단 결과이고, HQNR이 좋아진 것은 성능 결과다.

---

## 14. 이번 문서에서 바꾸지 않는 결론·아직 말할 수 없는 결론

### 유지

- 동일 용량 Teacher–Student라는 연구 설계는 유지한다.
- GT hard는 유지하고, binary hard/soft 양자택일로 돌아가지 않는다.
- 이번 no-align 캠페인에 아직 최적화되지 않은 aligner를 끼워 넣지 않는다.
- HQNR을 주 기준으로 사용한다. ERGAS의 개선만으로 case를 승급하지 않는다.
- Variance의 비교 대상은 출력 통계와 이동 uncertainty를 구분한다. 현재는 출력 통계가 실행 대상이다.
- 유력한 case가 좁혀져도 보류한 정의·코드·부정 결과는 보존한다.

### 아직 말할 수 없음

- R0의 +0.00089가 seed를 바꿔도 재현된다는 결론.
- R3의 작은 soft가 모든 학습 시점에서 무의미했다는 결론.
- Teacher가 같은 크기여서 KD가 불가능하다는 결론.
- β/τ 대조가 이미 무효라는 결론.
- 배치 평균 hard weight가 일정하므로 공간 재가중이 없었다는 결론.
- 과거 W96의 2σ 안에 들어가므로 W104의 모든 방법이 동등하다는 결론.
- Train L1과 HQNR이 반대로 움직였으므로 메커니즘이 반드시 정규화라는 결론.
- HQNR이 ERGAS와 불일치하므로 HQNR selector가 고장났다는 결론.

**최종 실행 방침:** 정상 run은 유지하되, 전체 조합의 무조건 소진은 하지 않는다. **s2의 R 사다리와 GV 정보원 분리, s3의 local R 대조와 GT-only 통계 표현 선별을 먼저 완료하고, HQNR 결과가 지지하는 1–2개 reconstruction과 1–2개 통계 표현에 후속 실험을 집중한다.**

---

## 15. 출처·식별 정보

아래 보고서는 원문을 대체하거나 수정하지 않았다. 본문의 `[검토 판단]`과 `[운영 조정]`이 원문의 관측 또는 결론과 다른 경우 그 차이를 해당 절에 명시했다. 외부 문헌 조사 없이 이번 대화의 결정, 첨부 분석, 기존 계획, 읽기 전용 저장소 조회를 근거로 작성했다.

### [U1] 최신 사용자 결정

현재 대화: W104·D122 no-align, 동일 용량 KD로 fitting 강화, 시간 제한 없는 우선순위 탐색. 특히 **HQNR을 중심으로 비교하고 ERGAS는 보조로 둔다**는 최신 지시. 이 결정은 과거 계획의 RR-ERGAS 우선 제안보다 우선한다.

### [S1] 첨부 통합 분석

`2026-09-11_kd-integrated-analysis.md`

사용 절: §3 loss 수식, §4 loss/weight 분해, §8 gate 해석, §9.1 W112 결과, §9.2 NA104 공통10격자 결과, §10 R2의 예측 상태, §12 한계. 원자료가 train_log·metrics·fr_mat20라고 기록된 보고서이며, 본 문서 작성 중 그 서버 원시 파일을 다시 계산한 것은 아니다.

### [S2] 첨부 s3 fitting 검토

`2026-09-11_kd-fitting-signals-review.md`

사용 절: §2 scalar loss 분해, §4 difficulty 분포의 계측 문제, §6 NA104 HQNR 결과, §7 과거 K 사다리, §9 대조군 제안, §10 평가 프로토콜·시점. 본 문서는 원문이 제시한 가설을 모두 검증 완료 사실로 채택하지 않는다.

### [S3] 기존 NA104 전체 계획

`PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md`

사용 절: §3 Teacher/Student 초기화·반복, §4–6 기본 loss, §11 case registry, §12 대조군, §13 장기 학습, §19 변경 관리. **§14의 RR validation ERGAS 우선 제안은 이번 HQNR 중심 결정으로 대체한다.** 원본 문서는 보존한다.

### [S4] 이번 조회의 s2 큐

Repository: `hojunking/PAN-Crafter-repro`  
Path: `config/queues/na104_s2.txt`  
조회 기준: 2026-09-11, default branch `main`의 읽기 전용 조회  
Git blob SHA: `ecd2c23935488c0c234de866a3936e0eb89a5d88`

큐 정의와 실제 서버 process 상태는 다르다. 큐의 “NA104 미시작이므로 eval 간격 비대칭 없음”이라는 주석은 S1의 실제 5/10-epoch 기록과 다르므로, 비교에는 actual run manifest를 우선한다.

### [S5] 이번 조회의 s3 큐

Repository: `hojunking/PAN-Crafter-repro`  
Path: `config/queues/na104_s3.txt`  
조회 기준: 2026-09-11, default branch `main`의 읽기 전용 조회  
Git blob SHA: `32bc0fbdafa02978dbed78b322539b88439a43b2`

이번 조회에서 Q01/Q02/Q03/Q07이 없는 것과, Q35–Q37이 R3 기반 표현별 15개 뒤에 있는 것을 확인했다. 실제 서버에서 이미 별도로 실행됐을 가능성은 inventory로 확인한다.

### [S6] 이번 조회의 Q00 config

Path: `config/NA104_Q00_W104_D122_WV3_N0_OFF_S1234_v1.yaml`  
조회 범위: 1–90행  
Git blob SHA: `e06b9a0e5622e465490dac965e44d306cfd1fafa`

확인 항목: W104/D122/no-align, primary best_hqnr, 보조 best_rr_val·last, Teacher T104_v1/best_hqnr, eval_only=true, 50K/seed1234, config 주석과 과거 val 선택 설명의 공존. 이 config가 모든 실제 run에서 그대로 사용됐다는 검증은 아니다.

### 첨부 원본의 로컬 SHA-256

| 파일 | SHA-256 |
|---|---|
| `2026-09-11_kd-integrated-analysis.md` | `07a306dc920891c761d7b7d089ce20fd1dc651829aa4a3c8e28f98f05f0ae69b` |
| `2026-09-11_kd-fitting-signals-review.md` | `60fa2d8e9a1678e7b288d5abf81d743dcc567d56f21f39eb346aff326bfb7a23` |
| `PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md` | `e0c67b190e05a2649bd7af05441997ad0a9bc7359b04ba86bff3da22e8f52959` |

---

**운영 결론 한 줄:** **기본 실험은 유지하고, HQNR 비교 계약·누락 진단·s2/s3 pending 순서만 지금 조정한다. 그 다음의 loss 변경은 대응 결과가 나온 뒤 새 ID로 수행한다.**
