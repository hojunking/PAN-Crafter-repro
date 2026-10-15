# EDGEBAL — q-edge 추가 이득의 불확실성 이후: GT edge 강도·시간배분·완만한 cue 검증
## s2·s5 신규 실험계획 / 구현 인계 — 2026-09-16

**상태:** 실행 설계. 이 문서 작성으로 학습·repository push·Sheet 변경을 실행하지 않았다. 동반 JSON/CSV는 계획용이며 trainer에 직접 넘기는 config가 아니다.  
**대상:** s2와 s5. s1·s3·s4의 실행과 QEGX 방법 대조는 그대로 둔다.  
**규모:** s2 신규 12run + s5 신규 14run = **기본 신규 26run**, 각 run 독립 fresh50K.  
**시간 정책:** 상한 없음. 각 서버의 첫 방향성 묶음부터 학습 기준값 합 ≥12h. 이전 50h·9h·절대 마감 미상속.  
**공통 모델:** Student W104–D121, Teacher T0/W112–D123 고정, Student A는 T0 최종 aligner 복사 후 joint 학습.  
**주 목표:** 기존 selected **raw-original HQNR 0.959–0.960**. V64·aligned 값으로 대체하지 않는다.

## 0. 이번 방향과 서버 분담

**q gate를 더 복잡하게 만드는 것보다, 먼저 edge를 얼마나·언제 적용해야 하는지 확인한다.**

s2는 **edge 강도와 성분별 기여**를 맡는다. Q12의 λ_E를 0/0.25/0.5/1/2배로 비교하고, Teacher를 학습 loss에 쓰지 않는 GT+edge, output soft 제거 대조를 만든다. λ_E=0.5배를 주 반복 후보로 미리 고정해 3개 Student seed에서 확인한다.

s5는 **같은 edge 강도를 시간에 어떻게 배분하는가**와 **q-high 영역의 edge를 완전히 없애지 않는 간단한 대안**을 맡는다. 25K 전후 edge 1→0.5배, 반대 순서 0.5→1배, 상수 0.75배를 비교한다. 별도 소규모 cue 후보는 q-low/high에 각각 0.75/0.25배를 주며, 상수·shuffle·반전과 비교한다.

새로운 variance/feature KD, 식생/건물 분류, Scharr energy 조건화, Teacher 재학습, backbone 탐색, aligner LR/offset 계수 변경은 이 기본 묶음에 넣지 않는다. 기존 GT hard는 항상 남긴다.

### 기본 큐가 답할 질문

| 질문 | 직접 대조 |
|---|---|
| 기존 Q12의 개선 중 edge만의 효과가 있는가? | R3+edge ↔ 같은 R3의 edge 없음 |
| Teacher 없이 GT edge만으로도 충분한가? | N0+edge ↔ J0; R1+edge ↔ N0+edge |
| q로 절반을 고르기보다 모든 patch에 절반 강도를 주면 어떤가? | Q12_E050 ↔ 기존 QE50 ↔ JQ |
| 같은 평균 edge 계수라도 후반 약화가 좋은가? | DOWN ↔ UP ↔ CONST075 |
| q-high를 버리지 않고 약하게 감독하면 다른가? | FLOOR ↔ CONST050 ↔ FLOOR_SHUF ↔ FLOOR_REVERSE |

## 1. 근거 자료: 관측·보고서 해석·새 제안을 분리한다

### 1.1 최신 s2 보고서 — q gate의 추가 우위는 아직 확인되지 않았다

첨부 `2026-09-16_s2_qedge9-seed777-cross.md` §2의 원값이다. W104–D121, seed777, s2/170hx, J0/JQ/QE50 3벌 완주. 보고서의 plateau는 **≥30K**이며 이전 문서의 45–50K 6점 평균이 아니다.

| Case | Selected raw HQNR | 선택 update | 보고서 plateau ≥30K | Exact50K | 보고서 fSCC | 보고서 RR ERGAS | Train(h) |
|---|---:|---:|---:|---:|---:|---:|---:|
| J0 | 0.952367 | 26260 | 0.951056 | 0.950775 | 0.8687 | 2.0735 | 1.95 |
| JQ | 0.957728 | 18180 | 0.954902 | 0.954328 | 0.8829 | 2.1068 | 2.30 |
| QE50 | 0.957696 | 16160 | 0.956131 | 0.955744 | 0.8919 | 2.1200 | 2.29 |

보고된 차이는 **QE50−JQ = selected −0.000032 / plateau +0.001228**다. 원표의 반올림된 평균값을 다시 빼면 마지막 자리가 다를 수 있으므로 보고서의 delta를 유지한다. Exact50K 차이는 표에서 계산하면 +0.001416이다. 둘의 selected 성능은 사실상 비슷하고, 후반은 QE50 쪽이 높았지만 한 seed로 동등성이나 보편적 안정화 기전을 증명한 것은 아니다. [R1 §2·§4·§6]

Gate 자체는 작동했다. 다섯 학습 시점에서 QE50의 edge 항은 JQ의 평균 **0.491배**였고, s2의 cue 재계산에서는 gate label 차이가 0이었다. 따라서 현재 추가 이득 미확인을 구현상 gate 미적용으로 설명할 근거는 없다. 이 0.491은 서로 다르게 학습된 모델의 관측 loss 비율이지 고정 gradient 비율도, 새 λ_E의 정답도 아니다. [R1 §3]

### 1.2 보고서의 ‘edge 자체 양성’ 해석과 이번 대조 설계

보고서는 JQ−J0 = **+0.005361 selected, +0.003846 plateau**를 근거로 ‘edge 항 자체는 양성’이라고 해석하고 λ_E 조정을 제안한다. 그 해석과 원값은 그대로 인용한다. [R1 요지·§7]

**이번 설계자의 추가 구분:** J0→JQ에서는 Teacher-error hard 재가중, output soft, GT edge가 함께 바뀐다. 이 세 run만으로 직접 입증된 것은 **Q12 전체 패키지의 효과**다. Edge 단독 귀속에는 `R3_NOEDGE↔JQ`, `N0+edge↔J0`가 필요하다. 이를 이유로 s2 첫 묶음에 해당 대조를 넣었다. 이것은 원 보고서의 문장을 조용히 수정한 것이 아니라 새 실험의 식별 범위를 좁힌 것이다.

R1+edge↔N0+edge는 hard 재가중의 효과이고 JQ↔R1+edge는 output soft의 효과다. 모든 비교는 같은 정합 policy·λ_E·구조·seed·서버에서 한다.

### 1.3 이번 live Sheet: gate의 부호가 조건별로 다르다

이번 Google Drive export 및 native Sheet search에서 읽은 **selected raw**다. 전체 캠페인의 전수 재평가가 아니라 본 계획에 필요한 W104–D121 결과를 추렸다. 원본 행과 run ID는 동반 `live_relevant_results.csv` 및 `source_snapshot.json`에 있다.

| 서버·seed | J0 | JQ | QE50 | QE50−JQ | 출처 행 |
|---|---:|---:|---:|---:|---|
| s2·777 | 0.9524 | 0.9577 | 0.9577 | 0.0000 (4자리) | WV3-s2 78–80 |
| s5·2026 | 0.9552 | 0.9560 | 0.9557 | −0.0003 | WV3-s5 17–19 |
| s5·777 | 0.9536 | 0.9549 | **완료 행 미확인** | — | WV3-s5 20–21 |
| s3·2026 | 0.9555 | 0.9551 | 0.9558 | +0.0007 | WV3-s3(5090) 95–97 |
| s4·1234 | 0.9585 | 0.9565 | 0.9543 | −0.0022 | WV3-s4 21/22/25 |

s3의 QX50−XJ도 **0.9556−0.9548=+0.0008**이고, s4에서는 **LFQ 0.9586 / LFQE50 0.9571 / LFX 0.9589**다. ‘q gate는 무조건 유효’도 ‘모든 조건에서 무효’도 현재 자료의 결론이 아니다. 단일 조건의 작은 이득을 이유로 q threshold를 더 많이 탐색하기보다, 단순 강도·시간배분을 먼저 분리한다. [SHEET]

s2/s5의 같은 seed777 JQ는 selected가 0.9577/0.9549로 다르다. 이는 같은 seed의 호스트 반복이지 새 독립 seed 둘이 아니다. QE50의 s5·777 완료 후에는 같은 정의의 host 비교에 연결하되, 그 run을 중복 시작하지 않는다. Sheet 미등재는 실행 중/미실행의 구분 정보가 아니다.

### 1.4 숫자 출처 불일치는 그대로 남긴다

s2 보고서 ERGAS는 2.0735/2.1068/2.1200, 현재 Sheet는 2.0866/2.1187/2.1343이다. 동일 필드라고 합치지 않고 `ergas_report`, `ergas_sheet`로 보존한다. 정확한 차이 원인은 이번 자료만으로 확정하지 않는다.

보고서의 ±0.0097/±0.0109는 FR 장면 간 분산 표기이며 3-seed 표준편차가 아니다. 보고서가 쓰는 0.0031은 기존 경험적 seed 변동선으로 기록하되, 새 실험의 p-value·동등성 허용폭·보편적 채택선으로 자동 승계하지 않는다.

### 1.5 정합은 유지한다

보고서의 같은 checkpoint Δ 교체에서는 learned→zero 시 HQNR이 J0/JQ/QE50 각각 약 0.0111/0.0070/0.0091 낮아졌다. 이는 해당 복원망이 그 보정을 유용하게 사용한다는 근거다. 별도로 no-align으로 학습한 모델과의 인과 비교는 아니다. [R1 §5.5]

이번에는 PAN 이동을 끄거나 관측된 중앙 Δ를 목표로 강제하지 않는다. ‘Δ가 작아진 모델의 HQNR이 높았다’를 근거로 모든 Δ에 0.8을 곱하는 조정도 넣지 않는다. Aligner policy·LR·offset는 고정하고 edge 감독만 조정한다.

## 2. 공통 파이프라인과 불변 조건

| 항목 | 고정값 / 계약 |
|---|---|
| Student | W104–D121, depth=[1,2,1], HRMS 8ch residual, PAN+MS 9ch |
| Teacher | T0/W112–D123; 자신의 A/U 모두 frozen·eval, 자신의 PAN 보정 경로 유지 |
| T0 source | L1E4 S2025 best_hqnr, step24240 |
| T0 file SHA256 | `16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32` |
| A state hash | `493aca181f93ae68` |
| Student 초기화 | T0의 A 전체 strict copy + architecture/seed별 fresh U. 다른 case와 동일 U 초기 tensor 공유 |
| 좌표 | PAN만 warp; MS base·GT·출력의 M-frame 유지 |
| Policy | J: A와 U 계속 학습. Freeze·gradient routing·새 geometry 없음 |
| Update / batch | 50,000 / 48, 모든 run 독립 학습 |
| Optimizer/LR | AdamW WD0.01; U 1e−4 / A 1e−5; warmup100 + 전역 50K cosine |
| 기본 α / β | 1.0 / 0.1. N0/R1 대조만 명시적으로 변경 |
| τ_R | 0.012463942170143127 |
| λ_E0 | 0.09075170336956798 |
| Offset | 기존 b=2 HR px 원판 jitter, 0-based 홀수 update에 λ_off=1e−4 |
| GT hard | 낮은 e/q 포함 모든 위치에 항상 유지 |
| R3 gate | d_T/a_T 및 Teacher target detach; Student residual gradient는 유지 |
| λ_E calibration | E0를 고정 이식한 뒤 **배수만** 튜닝. case·seed마다 재calibration하지 않음 |
| q 자산 | 기존 T0 AXIS16, train-only θq=0.3276133416220546 |
| Inference | Student A+U만. GT·Teacher·q probe·loss 선택 필요 없음 |

Teacher와 Student의 크기는 다르므로 ‘동일 용량 KD의 검증’으로 부르지 않는다. 최종 방향이 안정된 후 동일 구조 Teacher를 만드는 작업은 별도 캠페인이다.

## 3. Loss 정의

### 3.1 Reconstruction은 기존 정의를 유지한다

\[
e_T=\mathrm{mean}_c|T-Y|,\quad e_S=\mathrm{mean}_c|S-Y|,\quad k=\mathrm{mean}_c|S-T|,
\]
\[
d_T=\operatorname{sg}\frac{e_T}{e_T+\tau_R},\qquad
 a_T=\operatorname{sg}\operatorname{clip}_{[0,1]}\frac{[e_S-e_T]_+}{e_S+\epsilon}.
\]
\[
L_H=\langle(1+\alpha d_T)e_S\rangle,\quad
L_K=\langle\beta(1-d_T)a_T k\rangle.
\]

N0는 plain GT만, R1은 L_H만, R3는 L_H+L_K다. 저오차에서도 GT를 제거하지 않는다. Soft를 활성 pixel 수나 weight 합으로 재정규화하지 않는다.

기존 sample별 signed Scharr GT edge를 E_i라 하면 새 목적함수는:

\[
\boxed{L=L_R+\lambda_{E0}\frac1B\sum_i w_i(t)E_i +m_t10^{-4}L_{\mathrm{off},S}.}
\]

w_i(t)만 case마다 다르다. E_i는 최종 HRMS–GT의 같은 밴드 signed Scharr /32, reflect pad, 내부1px, x/y 0.5 평균이다. 기존 edge와 새 edge를 중복해서 더하지 않는다. Gate/계수는 고정 자산 또는 update에서 결정하고 gradient를 받지 않는다.

활성 loss는 U/A 양쪽에 전달한다. Offset만 A에 직접 전달한다. 이 캠페인은 QER50처럼 U와 A의 edge 수신자를 다르게 만들지 않는다.

### 3.2 상수 edge 강도 — s2 주력

w_i(t)=r. 배수 r은 기존 λ_E0 대비다.

| 배수 r | 실제 λ_E | 의미 |
|---:|---:|---|
| 0 | 0 | R3 reconstruction만; edge 계산 경로 비활성 |
| 0.25 | 0.0226879258423919950 | 기존 edge의 1/4 |
| 0.5 | 0.045375851684783990 | 기존 edge의 1/2 |
| 0.75 | 0.0680637775271759850 | schedule 평균 계수 대조 |
| 1 | 0.09075170336956798 | 기존 Q12 |
| 2 | 0.18150340673913596 | 기존 edge의 2배 |

고정 계수를 줄이는 것은 q gate와 같은 방법이 아니다. 예컨대 r=0.5는 모든 patch에서 절반 gradient를 허용하고 QE50는 일부 patch의 edge를 제거한다. 둘의 평균 edge loss나 parameter gradient가 같다는 보장도 없다. 관측 0.491을 새 coefficient로 맞추지 않는다.

### 3.3 시간배분 — s5 주력

0-based optimizer update t를 기준으로:

\[
w^{DOWN}(t)=\begin{cases}1,&0\le t<25000,\\0.5,&25000\le t<50000,\end{cases}
\quad
w^{UP}(t)=\begin{cases}0.5,&0\le t<25000,\\1,&25000\le t<50000.\end{cases}
\]

`CONST075`는 전 구간 0.75다. 세 방식 모두 50K update의 **계수 산술평균 0.75**다. 그러나 실제 residual·gradient·Adam 상태·LR의 가중 효과는 다르며, 세 방식의 총 학습 효과가 같다는 뜻은 아니다. 그 시간배분 차이가 질문이다.

DOWN의 가설은 초기 구조 학습을 유지하면서 후반의 과도한 구조 감독 비용을 낮출 수 있는가다. UP은 반대 방향 대조다. 보고서가 보여준 selected/후반 차이만으로 DOWN이 우승한다는 보장은 없다.

**Aligner는 25K 이후에도 계속 학습한다.** U/A optimizer와 cosine을 재시작하지 않는다. 기존 LF 실험은 A 업데이트를 중단하지만 이번 실험은 edge 계수만 바꾼다는 차이를 명시한다.

### 3.4 완만한 q 가중 — s5의 제한적 탐색

기존 frozen T0의 g_i=1[q_T(i)<θq]를 그대로 쓴다.

\[
w_i^{FLOOR}=0.25+0.5g_i.
\]

q-low는 0.75, q-high는 0.25다. q-high 영역의 edge를 0으로 끄지 않는 새 후보다. q gate가 일반적으로 우월하다고 전제하지 않고, **‘제외’보다 ‘약한 차등 감독’이 나은가**를 한 seed에서 묻는다.

| Case | q-low edge | q-high edge | 대조 |
|---|---:|---:|---|
| CONST050 | 0.5 | 0.5 | cue 없는 평균 계수 기준 |
| FLOOR | 0.75 | 0.25 | low-q 쪽 강조 |
| FLOOR_SHUF | stratum 내 교환 | stratum 내 교환 | 같은 계수 분포에서 q–sample 연결 제거 |
| FLOOR_REVERSE | 0.25 | 0.75 | 방향 반전 |

Shuffle은 기존 `(Teacher e_ROI32 decile, 실제 rotation state)` 안에서 고정 permutation51515를 사용한다. Calib gate 비율이 0.5이므로 세 cue 방식의 calibration 평균 계수는 0.5이고, full train에서는 거의 0.5다. **Mean weight 일치≠mean edge loss/gradient 일치**이므로 관련 로그도 보존한다.

FLOOR가 CONST050만 이기고 SHUF/REVERSE보다 낫지 않다면, low-q 방향 cue의 독립 효과를 주장하지 않는다. 이 floor는 soft가 아니라 **GT edge 계수**다. q나 e를 새로운 Teacher confidence로 해석하지 않는다.

## 4. Case registry와 기존 코드 대응

새 case는 `EB_` prefix를 사용한다. 기존 QEDGE9/QEGX run을 새 이름으로 재표시하거나 덮어쓰지 않는다.

| 새 Case | 정확한 구성 | 기존 정의와의 관계 |
|---|---|---|
| EB_N0 | J + plain GT | J0와 학습 동치 |
| EB_R3E000 | J + R3, edge 없음 | J_R3_NOEDGE와 동치 |
| EB_R3E025 | J + R3 + 0.25λ_E0 edge | 새 상수 배수 |
| EB_R3E050 | J + R3 + 0.5λ_E0 edge | 기존 J_QE025와 동치. 기존 이름의 025는 실제 0.25배가 아님 |
| EB_R3E075 | J + R3 + 0.75λ_E0 edge | 새 상수 배수 |
| EB_R3E100 | J + R3 + λ_E0 edge | JQ와 학습 동치 |
| EB_R3E200 | J + R3 + 2λ_E0 edge | 기존 J_QE10과 동치 |
| EB_N0E100 | J + plain GT + λ_E0 edge | J_N0_EDGE와 동치 |
| EB_R1E100 | J + 재가중 GT + λ_E0 edge | XJ, output soft 제거 |
| EB_EDOWN / EB_EUP | R3 + §3.3 edge schedule | 새 시간배분 |
| EB_QFLOOR / EB_QFSHUF / EB_QFREV | R3 + §3.4 affine cue weight | 새 가중 방식 |

EB_N0E100은 **학습 loss에 Teacher 출력이 안 들어가는 것**이지 T0와 무관한 전체 방법이 아니다. Student A는 여전히 T0에서 복사한다. 평가용 Teacher error bin은 loss에 섞지 않는다.

## 5. s2 실행 큐 — 신규 12run

**우선순위:** 상수 강도와 성분별 기여를 좁혀 본다. 탐색 seed777, 고정 반복 seed2026·9091.

재사용 전제의 기존 대조: `PAKD50_{J0,JQ,QE50}_W104_D121_WV3_T0_S777_FRESH50_v2`. 새 case는 독립 fresh50K다. 큐 순서대로 checkpoint를 이어받는 것이 아니다.

| 순서 | 단계 | Seed | Case | 바꾸는 것 / 대조 | 시간 기준(h) | 예약(h) |
|---:|---|---:|---|---|---:|---:|
| 1 | A | 777 | `EB_R3E000` | JQ − R3E000: edge만의 효과 | 2.30 | 2.6967 |
| 2 | A | 777 | `EB_R3E050` | 상수 절반 vs JQ / 기존 QE50 | 2.30 | 2.6967 |
| 3 | A | 777 | `EB_R3E025` | edge 강도를 더 줄여도 되는가 | 2.30 | 2.6967 |
| 4 | A | 777 | `EB_R3E200` | 강도 증가의 이득·비용 | 2.30 | 2.6967 |
| 5 | A | 777 | `EB_N0E100` | N0+edge − J0: Teacher 없는 edge 효과 | 2.30 | 2.6967 |
| 6 | A | 777 | `EB_R1E100` | JQ − R1E100: output soft 효과 | 2.30 | 2.6967 |
| 7 | B | 2026 | `EB_N0` | 두 번째 seed의 no-KD 기준 | 1.95 | 2.3117 |
| 8 | B | 2026 | `EB_R3E100` | 두 번째 seed의 기존 Q12 | 2.30 | 2.6967 |
| 9 | B | 2026 | `EB_R3E050` | 사전 고정 half-edge 반복 | 2.30 | 2.6967 |
| 10 | C | 9091 | `EB_N0` | 세 번째 seed의 no-KD 기준 | 1.95 | 2.3117 |
| 11 | C | 9091 | `EB_R3E100` | 세 번째 seed의 기존 Q12 | 2.30 | 2.6967 |
| 12 | C | 9091 | `EB_R3E050` | 사전 고정 half-edge 반복 | 2.30 | 2.6967 |

A의 6run부터 학습 기준값 합 **13.80h**다. B/C는 A 결과를 보고 한 승자만 고르는 방식이 아니라 **기존 Q12 vs 0.5배 Q12 vs no-KD의 사전 고정 반복**이다. 그래서 새 coefficient의 효과를 3개 Student seed에서 읽을 수 있다.

0.25/2배 또는 edge 없음이 가장 유력해지면 B/C의 J0/JQ는 그대로 마치고 해당 r을 같은 두 seed에 추가한다(§9). 사후에 기본 확인표를 바꾸어 불리한 half-edge 결과를 숨기지 않는다.

### s2에서 계산할 핵심 대비

- Edge 단독: `H(JQ)-H(EB_R3E000)` — hard/soft 고정.
- Teacher 없는 edge: `H(EB_N0E100)-H(J0)`.
- Hard 재가중: `H(EB_R1E100)-H(EB_N0E100)`.
- Output soft: `H(JQ)-H(EB_R1E100)`.
- 강도 조정: `H(EB_R3E050)-H(JQ)`를 seed777/2026/9091에서 각각 계산.
- q 선택: 같은 server/seed777의 기존 QE50와 EB_R3E050 비교. 이는 nominal 강도 대조이지 완전한 gradient-matched 대조가 아니다.

s2의 다른 W112 또는 W104–D122 no-align run을 분모로 가져오지 않는다.

## 6. s5 실행 큐 — 신규 14run

**우선순위:** 같은 평균 edge 계수의 시간배분을 확인하고, q-floor의 필요성을 제한적으로 시험한다.

기존 QEDGE9의 `QE50/W104/S777/v1`은 이번 Sheet에서 완료 행이 없었다. 실행 중이면 끝까지 유지한 뒤 새 큐로 넘긴다. 이미 완료되어 뒤늦게 업로드된 경우는 그 완료 자산을 그대로 사용한다. 새 실험 시간을 채우기 위해 재시작하지 않는다.

재사용할 기존 대조는 W104 seed2026·777의 J0/JQ, seed2026의 QE50다. s2와 같은 seed777이라는 이유로 s2 JQ를 s5의 control로 대체하지 않는다.

| 순서 | 단계 | Seed | Case | 바꾸는 것 / 대조 | 시간 기준(h) | 예약(h) |
|---:|---|---:|---|---|---:|---:|
| 1 | A | 2026 | `EB_R3E050` | 상수 절반 edge, q-floor 기준 | 1.57 | 1.8937 |
| 2 | A | 2026 | `EB_R3E075` | 시간 스케줄의 평균 계수 대조 | 1.57 | 1.8937 |
| 3 | A | 2026 | `EB_EDOWN` | 후반 edge 1→0.5 | 1.57 | 1.8937 |
| 4 | A | 2026 | `EB_EUP` | 반대 순서 0.5→1 | 1.57 | 1.8937 |
| 5 | A | 2026 | `EB_QFLOOR` | q-high에도 edge 0.25를 남김 | 1.65 | 1.9817 |
| 6 | A | 2026 | `EB_QFSHUF` | floor-gate 조건부 shuffle | 1.65 | 1.9817 |
| 7 | A | 2026 | `EB_QFREV` | q-low/q-high 강조 방향 반전 | 1.65 | 1.9817 |
| 8 | A | 2026 | `EB_N0E100` | 같은 서버의 GT+edge 단순 기준 | 1.57 | 1.8937 |
| 9 | B | 777 | `EB_R3E075` | 두 번째 seed의 평균 계수 대조 | 1.57 | 1.8937 |
| 10 | B | 777 | `EB_EDOWN` | 두 번째 seed의 DOWN 반복 | 1.57 | 1.8937 |
| 11 | C | 9091 | `EB_N0` | 세 번째 seed의 no-KD 기준 | 1.38 | 1.6847 |
| 12 | C | 9091 | `EB_R3E100` | 세 번째 seed의 기존 Q12 | 1.57 | 1.8937 |
| 13 | C | 9091 | `EB_R3E075` | 세 번째 seed의 평균 계수 대조 | 1.57 | 1.8937 |
| 14 | C | 9091 | `EB_EDOWN` | 세 번째 seed의 DOWN 반복 | 1.57 | 1.8937 |

A의 8run부터 학습 기준값 합 **12.80h**다. B/C는 **DOWN과 동일 평균 계수 CONST075를 seed777·9091에 반복**한다. JQ·J0와의 비교도 해당 서버·seed의 기존 또는 신규 기준으로 남긴다.

FLOOR/SHUF/REVERSE는 기본적으로 seed2026의 스크리닝이다. 이 한 seed의 최고값을 방법 채택으로 쓰지 않는다. FLOOR가 두 대조보다 유력할 때만 §9의 6run 확인을 수행한다. UP이 유력하면 UP을 seed777·9091에 추가한다. 기본 DOWN의 결과를 지우지는 않는다.

### s5에서 계산할 핵심 대비

- `DOWN−CONST075`: 같은 평균 계수에서 시점 배분의 효과.
- `DOWN−UP`: 적용 순서의 효과.
- `DOWN−JQ`와 `CONST050−JQ`: 고정 강도보다 schedule이 필요한지.
- `FLOOR−CONST050`, `FLOOR−SHUF`, `FLOOR−REVERSE`: 총강도·선택 연결·방향의 구분.
- `N0E100−J0`: Teacher 없이 GT edge가 만드는 효과.

새 loss schedule과 q-floor를 한 run에서 곧바로 결합하지 않는다. 둘을 각각 검증한 뒤 한 개의 후속 교차만 허용한다.

## 7. 시간표: 실측과 편성용 대용값을 분리

### 7.1 현재 실제 Train(h)

| 서버 | 현재 구조 | 실제 기록 |
|---|---|---|
| s2 | W104–D121·seed777 | J0 **1.95h**, JQ **2.30h**, QE50 **2.29h** |
| s5 | W104–D121·seed2026 | J0 **1.38h**, JQ **1.57h**, QE50 **1.65h** |
| s5 | W104–D121·seed777 | J0 **1.30h**, JQ **1.55h**, QE50 **완료시간 미확인** |

이번에는 s5의 옛 1.34–1.36h를 계속 사용하지 않고 최신 W104의 시간을 반영했다. s2의 2.30h를 s5에 일괄 적용하지도 않는다.

### 7.2 신규 case 편성 원칙

s2 R3/edge형은 **2.30h**, plain N0는 **1.95h**를 사용한다. N0+edge와 no-edge가 빨라질 가능성을 사전에 차감하지 않는다. s5 dense/schedule/GT+edge형은 **1.57h**, cached floor형은 기존 QE50의 **1.65h**, plain N0는 큰 쪽의 **1.38h**를 대용한다.

**이 값들은 신규 run의 실측이나 보장 상한이 아니다.** q-floor는 기존 cue cache lookup이므로 매 batch 16 probe를 새로 돌리는 비용을 포함하지 않는다. 그렇게 구현하면 시간 모델을 다시 세워야 한다.

\[
T_{reserve}=1.10\sum T_{reference}+n\times10/60\ {\rm hours}.
\]

10%는 변동 여유, run당 10분은 기본 export·업로드·큐 전환 가예약이다. 정기 평가 시간은 기존 Train(h)에 들어 있으므로 다시 50회분을 더하지 않는다. 학습 총량을 줄이거나 대기시간을 넣어 ‘최소12h’를 맞추는 것은 목적이 아니다.

| 서버·단계 | 신규 run | 학습 기준값 합 | 여유 포함 예약 |
|---|---:|---:|---|
| s2 A — 첫 방향성 묶음 | 6 | 13.80h | 16시간 10분 48초 |
| s2 B — seed 반복 1 | 3 | 6.55h | 7시간 42분 18초 |
| s2 C — seed 반복 2 | 3 | 6.55h | 7시간 42분 18초 |
| **s2 전체** | **12** | **26.90h** | **31시간 35분 24초** |
| s5 A — 첫 방향성 묶음 | 8 | 12.80h | 15시간 24분 48초 |
| s5 B — seed 반복 1 | 2 | 3.14h | 3시간 47분 14초 |
| s5 C — seed 반복 2 | 4 | 6.09h | 7시간 21분 56초 |
| **s5 전체** | **14** | **22.03h** | **26시간 33분 59초** |

**총 신규 26run: 학습 기준값 합 48.93 서버·시간 / 예약 합 58.1563 서버·시간.**  
두 서버가 동시에 시작하면 병렬 경과시간 예약은 합계가 아니라 **31시간 35분 24초**다. A 단계만 먼저 완료하는 최초 결과 묶음은 s2/s5 중 최대 **16시간 10분 48초**다.

구현·배포·smoke, 현재 진행 중인 작업의 남은 시간, 정밀 재평가/분석, 다음 절의 확장, 검증 실패 시 control refresh는 별도다. 신규 case가 끝나면 `host×arch×case` 실측으로 다음 예약을 갱신하며, 반복 동치가 유지되는 조건에서 여러 seed의 시간을 함께 참고한다.

### 7.3 대조군 refresh가 필요한 경우

기존 run과 새 release의 Teacher/A·data·U 초기값·feeder/RNG·실제 loss·evaluator 동치를 확인할 수 없으면 비교 control을 새 버전으로 확보한다. 버전 문구만 다르다고 무조건 재학습하지도 않고, 같은 이름이라고 무조건 재사용하지도 않는다.

- s2의 J0/JQ 둘을 새로 해야 하면 **5시간 00분 30초 추가 예약**. QE50는 먼저 cross-host 완료를 기다리고, 추가 q 비교가 필요할 때만 별도 refresh한다.
- s5의 seed2026·777 J0/JQ 네 개 모두 새로 해야 하면 **7시간 09분 24초 추가 예약**.

조건부 refresh는 위 신규26run 합에 넣지 않았다. 실행 manifest에 필요 여부와 선택한 control hash를 확정한다.

## 8. 판정과 기록 — selected를 사후에 plateau로 바꾸지 않는다

### 8.1 서로 다른 시간 집계는 다른 이름으로 보관

| 필드 | 새 실험의 정의 | 용도 |
|---|---|---|
| selected_raw_original | 기존 GRID1010_50K_v1 후보와 best_hqnr comparator, raw FR20 | 기존 주 성능 비교 |
| exact50K_raw | 정확한 optimizer update 50000 | 동일 학습량 종료 상태 |
| plateau_30Kplus | 30300,31310,…,49490,50000 **21개** HQNR 평균 | 첨부 보고서 ≥30K와 맞추는 별도 후반 집계 |
| plateau_45_50 | 45450,46460,47470,48480,49490,50000 **6개** 평균 | 이전 PAKD/QEGX의 고정 후반 집계 |
| V64/aligned | 기존 진단 규약 | 공식 raw와 혼합 금지 |

이번 보고서는 ≥30K를 사용했다. 45–50K 값이 없는데 이를 같은 값으로 채우지 않는다. 최종 비교에서 selected와 후반이 갈리면 ‘selected 효과’와 ‘후반 유지 효과’를 나누어 보고하며, 유리한 지표만으로 성공을 재정의하지 않는다.

### 8.2 반복 단위와 해석

s2 half-edge의 3seed는 777/2026/9091, s5 DOWN의 3seed는 2026/777/9091이다. 두 서버를 합쳐 6개 독립 seed로 세지 않는다. 같은 seed의 host 차이는 별도 열로 남기고, method delta는 같은 host에서 계산한다.

각 seed의 대응 Δ, 그 평균·표준편차(n−1), 최소/최대, 개선 seed 수를 제시한다. 3개 seed는 여전히 작은 표본이다. 0.0031보다 작다는 이유만으로 무효라고 확정하거나, 크다는 이유만으로 통계적 유의라고 자동 판정하지 않는다. 장면별 bootstrap을 붙여도 그것이 학습 seed 변동까지 포함하지는 않는다.

본 자료의 FR20은 이미 반복 개발·선택에 쓰였다. 따라서 목표 달성은 개발 기준의 달성으로 기록하고, 최종 주장을 위해서는 방법을 고정한 이후 가능한 독립 평가를 별도로 해야 한다. 현재 제공되지 않은 held-out 세트를 사용했다고 쓰지 않는다.

### 8.3 다음 선택 규칙

1. 우선 같은 host·seed의 J0/JQ보다 **raw selected**가 좋아지는지 본다. last와 두 plateau에서 심한 공통 후퇴가 없는지도 함께 보고한다.
2. 일정이 좋아도 CONST075와 차이가 없으면 더 단순한 상수 계수를 우선 후보로 남긴다.
3. FLOOR가 SHUF/REVERSE/CONST050보다 일관되게 좋지 않으면 q 가중을 최종 방법의 기여로 주장하지 않는다.
4. N0+edge가 Teacher 기반 방법을 대체하면, 결과에 맞춰 ‘GT 구조 감독’ 효과로 논지를 좁힌다. Soft를 썼다는 사실을 성능 기여로 대신하지 않는다.
5. 어느 후보도 반복 이득이 없으면 기준 Q12/J0를 유지하고 이 branch를 음성/불확실 결과로 종료한다. 새 계수를 반드시 채택하지 않는다.

‘q가 좋다/나쁘다’보다 **좋은 최종 raw HQNR을 만드는 가장 단순한 확인된 설정**을 택하는 것이 목표다.

## 9. 조건부 확장 — 기본26run와 구분

| 확장 | 조건 | 추가 run | 시간 예약 |
|---|---|---:|---|
| S2-R* | r=0.25/2/0이 유력하면 같은 r을 seed2026·9091에 추가 | 2 | **5시간 23분 36초** (2.30h 대용) |
| S2-GT-EDGE | N0+edge가 유력하면 seed2026·9091에서 반복; 기존 N0와 비교 | 2 | **5시간 23분 36초** |
| S5-UP | UP이 유력하면 seed777·9091에 추가; 기존 CONST075/DOWN과 비교 | 2 | **3시간 47분 14초** (1.57h 대용) |
| S5-FLOOR | seed777·9091 각각 CONST050/FLOOR/SHUF | 6 | **11시간 42분 50초** (상수1.57, cue1.65 대용) |
| S5-COMBINE | constant/schedule와 FLOOR가 각각 반복 양성이면, 정확한 단독 대응을 보존한 조합 1개를 두 seed에 추가 | 2 | 실제 조합 실측 후 계산; 이번 합계 제외 |

모든 확장을 전수 실행하라는 뜻은 아니다. 큰 단일 seed 값 하나로 확장을 자동 선택하지 않고, 관련 대조의 방향과 반복 결과를 본다. 기본 half-edge/DOWN 반복은 ‘WIN 선택 후 유리한 비교만 실행’하지 않도록 사전 고정했다.

## 10. 구현 인계

### 10.1 새 branch·이름·기존 자산 보존

- 새 campaign: `EDGEBAL_A104D121_S2S5_20260916_v1`.
- `experiment_branch_id`: `A104D121_T0FIX_EDGEBAL_v1`.
- 새 이름 예: `PAKD50_EB_R3E050_W104_D121_WV3_T0_S777_FRESH50_v1`.
- 서버별 이름 충돌 방지를 위해 본 문서의 s2는 v1, s5는 v2로 잡았다. 이 버전 차이는 실행 identity이며 학습 수식 차이가 아니다. full ID가 이미 존재하면 내용을 대조하고 새 버전을 명시적으로 발급한다.
- 코드 snapshot 근거: `ac232452e6b789225edad6817a04627afdbc7bd2`. 실제 실행 release SHA는 구현 후 고정하며, 학습 worktree를 run 중 pull로 바꾸지 않는다.
- 동반 `case_registry.json`의 `implemented:false`는 이 문서가 구현을 실행하지 않았다는 표시다. 기존과 수학적으로 동치인 case도 branch/설정/이름 등록은 필요하다.

### 10.2 최소 변경 지점

1. `tools/gen_pakd50_configs.py`: EB case와 계수 registry, explicit seed 큐, branch별 time policy, reference/runtime 키, parser roundtrip, description.
2. `kdv/registry.py`: 새 `edge_schedule` / `edge_weight`를 명시 검증한다. 기존 routing·edge_route·stat.extra와 섞지 않는다. R3E000은 stat off, N0E100은 eval-only Teacher다.
3. `train_kdv.py`/loss: per-sample E_i에 w_i(t)를 곱한 뒤 **전체 batch mean**. λ_E0/r를 중복 곱하지 않는다. DOWN/UP는 global optimizer update를 읽고 resume에서 동일 값이 재현되어야 한다.
4. `kdv/edge_gate.py`: 기존 binary low-q/shuffle 배열은 바꾸지 않는다. 새 affine weight를 caller에서 명시 파생하거나 별도 버전 모드로 구현한다. 기존 low_q 함수가 내는 0/1을 몰래 0.25/0.75로 바꾸지 않는다.
5. cue verification·feeder meta: 같은 Teacher/A·train H5·실제 augmentation key. 현재 hflip/vflip은 옵션이 켜지면 항상 적용, rot4 상태 cache라는 기존 계약을 유지한다. 일반적인 랜덤 flip으로 고치는 작업을 이번에 섞지 않는다.
6. budget/gate/waiter: 자체 ledger·queue, no hard limit; 예전 deadline/9h/50h admission이 새 branch에 적용되지 않도록 planner와 trainer 양쪽에서 검사한다. 숫자 12를 예산 상한으로 넣지 않는다.

기존 `stat.ramp_updates`를 의미 확인 없이 후반 감소 스케줄로 재사용하지 않는다. 원래 ramp와 이번 piecewise 배율은 다른 규칙이다. code에서 정확한 25K 분기와 boundary test를 추가한다.

### 10.3 실행 전 검사

| 검사 | 통과 기준 |
|---|---|
| EB_N0 / EB_R3E100 동치 | 기존 J0/JQ와 같은 batch의 output·loss·U/A gradient·1-step update 일치 |
| E0 / β0 | EB_R3E000은 no-edge R3, EB_R1E100은 XJ와 동치 |
| 상수 배율 | 동일 output에서 edge loss와 그 gradient가 지정 배수로 변함 |
| Hard/soft 불변 | edge 조정 때문에 d/a 또는 L_H/L_K 정의가 변하지 않음; GT hard≥1 |
| 경계 | t=24999/25000의 DOWN/UP 배수가 정확함, 25K 후 A가 계속 업데이트됨 |
| Resume | 분기 직전 중단·재개해도 batch/RNG·optimizer·계수열이 연속 실행과 동일 |
| Cue floor | g=1→0.75, g=0→0.25; reverse는 반대; Teacher/q/weight grad 없음 |
| Shuffle | 기존 stratum 활성 수와 weight 분포 보존, dataset-wide 고정 permutation |
| Reduction | active 수/weight 합 나눗셈 없음, GT edge 두 번 더하지 않음 |
| 경로 | 모든 active rec/edge는 A/U; offset는 A. NaN gradient·detach 손실 없음 |
| Identity | T0/hash/init/data/evaluator/architecture 검증; small Student bridge 명시 |
| Queue | dry-run 무기록, 실행 중 run 중단 안 함, 완료 ID 덮어쓰기 없음 |

### 10.4 기록·gspread

기존 `tools/_upload.sh → gspread/gspread_upload.py`를 유지한다. s2는 `WV3-s2`, s5는 `WV3-s5`에 올린다. 새 Sheet 직접 생성·수정은 하지 않는다.

- N열: raw-original HQNR, O열: V64 그대로 구분.
- X열 예: `PAKD50 / EB_EDOWN / A104D121 / EDGEBAL / FRESH50`.
- W Notes: `edge_mult`, `edge_schedule`, `edge_low/high`, `gt_hard_always=1`, `T0_sha`, `cue_asset_id`, `seed`, `release_sha`, `control_run_id`.
- 기존 레이아웃 마지막 열 가정을 새로 만들지 않는다. EB branch 분류·archive keep·모든 upload/reorder 경로가 run을 빠뜨리지 않는지 검사한다.
- train log: plain_GT / weighted_hard / weighted_soft / raw_E / effective_E / effective λ / q-low-high별 E·recon / wE mean / A correction 통계.
- 고정 진단 시점만 hard·soft·edge·offset의 U/A gradient norm과 cosine을 기록한다. loss ratio와 gradient ratio는 다른 필드다.
- 결과 후 selected step와 exact50K, 두 plateau의 후보 목록/count, official Dλ/Ds/fSCC, RR ERGAS/SAM/PSNR를 동일 자산에 연결한다. RR SCC/JQM을 fSCC 대용으로 쓰지 않는다.

### 10.5 큐 전환

s2 완료된 QEDGE9 3벌은 재사용 검증부터 한다. s5는 진행 중인 QE50 seed777 또는 다른 승인 run을 마친 뒤 경계에서 새 큐로 넘긴다. Sheet만 보고 프로세스를 kill하지 않는다.

새 코드·config·큐를 별도 worktree에 배포하고 smoke 통과 후 지정 서버에서 실행한다. s1·s3·s4의 큐나 기존 cue/pilot/ledger를 변경하지 않는다. 신규 실험은 새 hard deadline이 없으며, 필요한 실패 재개·완주 판정은 그대로 유지한다.

## 11. 실행자가 받는 산출물

- 본 master MD: 근거·수식·전체 계획.
- `S2_EDGEBAL_Run_Handoff_2026-09-16.md`, `S5_EDGEBAL_Run_Handoff_2026-09-16.md`: 서버별 큐와 주요 주의.
- `case_registry.json`: 계획용 loss/policy/seed registry. 실행 config 아님.
- `s2_queue.csv`, `s5_queue.csv`: run별 ID·시간·대조.
- `runtime_budget.json`: 단일 예약식으로 계산한 합계와 refresh 여유.
- `source_snapshot.json`, `live_relevant_results.csv`, `paired_selected_differences.csv`: 이번 사용 원값·출처·계산.

## 12. 출처와 해석 경계

**[R1]** 사용자 첨부 `2026-09-16_s2_qedge9-seed777-cross.md`, §2(고정밀 결과·≥30K plateau), §3(gate 작동), §4(선택 spike에 대한 보고서 해석), §5/5.5(공간·분광 및 같은 checkpoint Δ 개입), §6(단일 seed·host·소요시간), §7(λ_E tuning 제안).

**[R2]** 사용자 첨부 `2026-09-15_s3_pakd50-results.md`, W112–D123 seed2026의 backend×edge 및 LF 상호작용. 새 W104의 효과값으로 이전하지 않는다.

**[R3]** `PAN_KD_Q12_R1_Spec_and_Aligner_Interface_2026-09-14.md`, §4–7 및 §9–10. 수식·GT hard·edge 정의·reduction·calibration 계약.

**[R4]** `PAN_Aligner_L1E4_TechnicalSpec_Evidence_KD_Handoff_2026-09-14.md`, §2–6. A/U·좌표·offset·raw HQNR와 seed 변동선의 의미.

**[SHEET]** 이 대화의 Google Drive live export `pan-cvpr27.xlsx` 및 s2/s5/s3/s4 native row search. export 파일 SHA와 사용한 23개 run을 source_snapshot에 보존. Sheet는 latest selected 값만 있으며 자동으로 last/plateau를 보충하지 않는다.

**[CODE]** `hojunking/PAN-Crafter-repro` main snapshot `ac232452e6b789225edad6817a04627afdbc7bd2`: `tools/gen_pakd50_configs.py`, QEDGE9/QEGX 구현·큐. 기존 scalar `.5×`/`2×`와 q gate/routing 구현은 확인했지만, 본 EDGEBAL의 새 schedule/floor 구현과 학습 성능은 아직 없다.

**새 제안의 지위:** half-edge, DOWN/UP, q-floor는 이번 자료를 근거로 만든 실험 후보다. 최소12h는 탐색·반복의 규모 요구로 충족하며, 시간이나 loss 증가 자체를 연구 성과로 취급하지 않는다. 결과가 기존보다 불리하면 그 결과를 유지하고 방향을 다시 좁힌다.
