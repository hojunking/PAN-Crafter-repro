# PANDA GF2 긴 학습 결과 분석과 우선순위
**기준 시점:** 2026-09-21 21:19 KST에 확보한 live Sheet snapshot  
**분석 범위:** GF2 QG40/G20/L100; ablation은 WV3/QB 완료 등록 현황만 정리  
**원본:** `pan-cvpr27`, ID `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`  
**Snapshot SHA256:** `bb3d100564eb413df2502a232c115b746ef01d9b6e9a7c9223f1a2ae5985ae29`  
**조회한 저장소 최신 commit:** `6bb78e6d61183a7ed1cd1da992527e7207d48ba3`  
**수행 범위:** Sheet export와 핵심 native cell readback, 결과 재집계, 공개된 실행 규약 확인. 원격 GPU/process/full-state 재평가·queue·Sheet 수정 없음.

## 1. 핵심 결론

1. s3의 첫 동일-seed 길이 대조가 완성됐다. Student 50K→100K는 ERGAS −1.325%, HQNR −0.002618, Ds +0.002223이다. 그 상태에서 Teacher까지 100K로 바꾸면 ERGAS −0.371%, HQNR −0.003642, Ds +0.003703이다. **긴 학습은 이 한 seed에서 RR를 개선했지만 FR 공간 지표를 악화했다.**
2. s4의 H010(alpha 1→0.1)은 같은 R100/SS95001 BASE 대비 ERGAS −1.523%, HQNR +0.000176이다. 최종 ERGAS 0.544284는 현재 비교한 FULL 계열의 새로운 RR 최저 endpoint다. 그러나 Ds는 +0.000628로 악화하고, Dλ가 −0.000815로 개선됐다. **H010은 현재 RR 개선 후보이지 Ds 해결책이 아니다.**
3. H010의 두 번째 Student는 E=0.546269로 RR 성능 수준을 반복했으나 H=0.946669다. 같은 seed BASE가 아직 없어 alpha 효과의 반복성은 미확정이다.
4. s5의 E1은 첫 100K pair에서 HQNR −0.000672, ERGAS +0.276%, Ds +0.001257로 불리했다. 50K에서의 작은 공간 이득이 새 R100/100K에 자동 전이되지 않았다.
5. 현행 GF2 Student 66개 run(QG40 23 + G20 34 + L100 9)에서 저장된 RAW_MAX HQNR는 모두 0.964 미만이다. 최고 HQNR는 기존 S92001 그대로다. **RR frontier는 확장됐고 spatial frontier는 갱신되지 않았다.** 이 66개는 독립 seed 66개를 뜻하지 않는다.

## 2. 진행과 학습시간

| 서버 | 확인한 L100 Student | 정의된 Student | 완료된 비교 | 미등록 핵심 |
| --- | --- | --- | --- | --- |
| s3 | S01–S03, 3개 | 6개 | TS94001/SS95001의 T50/S50→T50/S100→T100/S100 | SS95002 S04–S06 |
| s4 | S07–S09, 3개 | 4개 | SS95001 BASE↔H010 | SS95002 BASE S10 |
| s5 | S11–S13, 3개 | 4개 | SS95001 BASE↔E1 | SS95002 BASE S14 |

직전 16:20 리뷰 대비 새 GF2 Student는 7개다. 현재 L100 Student 9개 모두 `OFFICIAL_EVAL_COMPLETE`, `READBACK_VERIFIED`, 후보 50개 평가로 등록돼 있다. 이 가운데 8개가 actual/horizon/exact endpoint 100,000이며 1개(S01)만 의도된 50K 대조다.

s3 50K train 0.948h, 100K 약 1.893–1.894h; s4 100K 약 2.027–2.030h; s5 100K 약 1.930h다. 이번에는 세 서버 모두 100K Student 완료 등록이 확인된다. `Train(h)`는 해당 모델 update 구간 누적 시간이며 Teacher·calibration·평가를 합한 전체 chain 비용은 아니다.

Teacher 자체 완료행은 L100에서 아직 없다. Student metadata에는 s3 T50/T100, s4 T100, s5 T100의 네 endpoint와 각 calibration이 연결돼 있다. L100 규약상 Teacher full-grid RR/FR 평가를 뒤로 미룬 채 endpoint/calibration 후 Student로 진행할 수 있으므로 Teacher 행의 부재를 Teacher 미학습으로 읽지 않는다. 다만 원격 Teacher checkpoint bytes를 이번에 직접 읽은 것은 아니다.

미등록 S04–S06/S10/S14의 현재 학습 step, 평가/업로드 대기, admission 여부는 알 수 없다. 등록된 실행 순서에서 '다음 차례'라는 것과 실제 실행 중이라는 것은 구분한다. 공통 t0와 admission ledger를 읽지 않아 남은 20시간 budget도 계산하지 않았다.

## 3. L100 전체 endpoint

모든 행은 같은 A/U checkpoint의 H/E/Dλ/Ds다. RAW_MAX와 섞지 않았다.

| Case | 서버 탭 | 행 | T/S | S seed | 조건 | HQNR | ERGAS | Dλ | Ds | Train h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S01 | GF2-s3(5090) | 34 | 50K/50K | 95001 | BASE | 0.952923 | 0.560808 | 0.019998 | 0.027607 | 0.948 |
| S02 | GF2-s3(5090) | 35 | 50K/100K | 95001 | BASE | 0.950305 | 0.553379 | 0.020451 | 0.029830 | 1.894 |
| S03 | GF2-s3(5090) | 36 | 100K/100K | 95001 | BASE | 0.946663 | 0.551329 | 0.020473 | 0.033533 | 1.893 |
| S07 | GF2-s4 | 14 | 100K/100K | 95001 | BASE | 0.953483 | 0.552704 | 0.020265 | 0.026773 | 2.030 |
| S08 | GF2-s4 | 15 | 100K/100K | 95001 | H010 | 0.953660 | 0.544284 | 0.019450 | 0.027401 | 2.027 |
| S09 | GF2-s4 | 16 | 100K/100K | 95002 | H010 | 0.946669 | 0.546269 | 0.020063 | 0.033919 | 2.027 |
| S11 | GF2-s5 | 21 | 100K/100K | 95001 | BASE | 0.947716 | 0.550621 | 0.020819 | 0.032101 | 1.930 |
| S12 | GF2-s5 | 22 | 100K/100K | 95001 | E1 | 0.947044 | 0.552141 | 0.020243 | 0.033359 | 1.930 |
| S13 | GF2-s5 | 23 | 100K/100K | 95002 | E1 | 0.951072 | 0.550514 | 0.020509 | 0.028998 | 1.930 |

100K Student 8개 중 5개가 E<0.552, H>0.964는 0개다. 서로 다른 reference/profile/seed의 관측 수이며 성공확률이나 독립 반복 추정으로 사용하지 않는다.

비교 기준:
- 기존 QG40 S92001 exact50K: H=0.9576734619644258, E=0.5564942836244443, Dλ=0.019257199518842714, Ds=0.023502848055251335 (`GF2-s3(5090)!10`).
- 같은 run RAW_MAX(45,450): H=0.9578326435185687, E=0.5584011583082027.
- 새 H010 S08 exact100K: H=0.9536595553795489, E=0.5442843598694014.
- 새 H010 S08 RAW_MAX(78,780): H=0.9543967932877783, E=0.5442889212579196.
- 새 H010 S08 E_MIN_DIAG(80,800): E=0.5426196172762351, H=0.9534033037908667.

RAW_MAX와 E_MIN_DIAG는 test-aware 개발 진단이다. 특히 E_MIN만 떼어 RR 대표나 validation 성과라고 하지 않는다. H010의 RR_VAL_SELECTED는 step98,980, E=0.5442658152031326/H=0.953624869083057으로 endpoint와 같은 해석이다.

## 4. 동일 조건 대응 차이

| 비교 | 선택 | ΔHQNR | ΔERGAS | ERGAS 변화 | ΔDs | ΔDλ |
| --- | --- | --- | --- | --- | --- | --- |
| Student 50K->100K | T50 | EXACT_FINAL | -0.002618 | -0.007428 | -1.325% | +0.002223 | +0.000453 |
| Student 50K->100K | T50 | RR_VAL_SELECTED | -0.002612 | -0.007258 | -1.294% | +0.002208 | +0.000462 |
| Teacher 50K->100K | S100 | EXACT_FINAL | -0.003642 | -0.002051 | -0.371% | +0.003703 | +0.000022 |
| Teacher 50K->100K | S100 | RR_VAL_SELECTED | -0.003644 | -0.002053 | -0.371% | +0.003705 | +0.000022 |
| T50/S50->T100/S100 | EXACT_FINAL | -0.006260 | -0.009479 | -1.690% | +0.005926 | +0.000474 |
| T50/S50->T100/S100 | RR_VAL_SELECTED | -0.006256 | -0.009310 | -1.660% | +0.005914 | +0.000484 |
| H010 alpha1->.1 | EXACT_FINAL | +0.000176 | -0.008420 | -1.523% | +0.000628 | -0.000815 |
| H010 alpha1->.1 | RR_VAL_SELECTED | +0.000168 | -0.008475 | -1.533% | +0.000638 | -0.000816 |
| E1 edge.002->.001 | EXACT_FINAL | -0.000672 | +0.001520 | +0.276% | +0.001257 | -0.000576 |
| E1 edge.002->.001 | RR_VAL_SELECTED | -0.000690 | +0.001454 | +0.264% | +0.001277 | -0.000577 |

위 paired 비교들은 시트의 local reference 연결, source/numerical revision, 데이터 split SHA를 대조했다. S01/S02는 같은 T50 reference, S07/S08 및 S11/S12도 각 local reference SHA가 같다. T50/T100 변경 비교는 reference 전체가 의도적으로 다르다.

초기 U tensor와 sample stream receipt는 로컬 raw manifest를 직접 읽지 않아 이번에 별도 재검증하지 않았다. 따라서 같은-seed·같은-source·같은-data 설계의 등록 결과 비교이며 서버 실행의 완전한 bitwise 인증은 아니다. 모든 효과의 완성된 pair는 현재 각 1개 seed뿐이다.

### 4.1 긴 학습

s3 T50/S50→T100/S100의 합산 변화:
- ERGAS 0.560808→0.551329, −1.690%.
- HQNR 0.952923→0.946663, −0.006260.
- Ds 0.027607→0.033533, +21.467%.
- Dλ 0.019998→0.020473.

이 결과는 기존에 서로 다른 서버/Teacher를 비교하던 상황보다 학습 길이 질문에 직접적이다. 다만 50K와100K는 각각 처음부터 다른 cosine horizon으로 학습했으므로 '기존 50K 가중치에서 50K만 더 이어간 효과'는 아니다. 스케줄을 포함한 긴 학습 recipe의 효과다.

따라서 지금 모든 run을 150K/200K로 더 늘리는 것은 우선순위가 낮다. 100K의 RR 이득은 보존하되 FR을 유지하는 optimization 경로를 별도로 찾아야 한다. 한 seed의 결과만으로 긴 학습이 모든 경우 FR을 악화한다고 일반화하지는 않는다.

s3 Teacher calibration도 T50→T100에서 tau_R 0.005708117→0.005406213(−5.289%), q_ref 0.415759563→0.394490957(−5.116%)로 낮아졌다. 그럼에도 Student Ds는 나빠졌다. 이 값들은 각각 Teacher train error와 relative-shift response의 reference scale이지 Student FR 품질 보증이 아니다. q_ref 감소만으로 native 정합 정확도나 q-weighting selectivity 향상을 주장할 수 없다.

### 4.2 H010

새로운 중요한 후보는 alpha=.1이다. 첫 pair에서 E와 Dλ가 개선되고 H가 거의 유지된다. E=0.544284는 운영 기준 .552까지 약1.398% 여유가 있다.

그러나 Ds는 0.026773→0.027401이다. HQNR의 작은 개선은 Ds 개선이 아니라 Dλ 개선과 Ds 악화의 합성 결과다. 'alpha를 줄여 PAN 과잉 상관이 해결됐다'고 해석하지 않는다.

같은 H010, 같은 local R100에서 SS95001→SS95002는 H가 −0.006990, Ds가 +0.006518 달라진다. RR E는 0.544284/0.546269로 모두 좋은 수준이다. **RR 이득 후보와 FR seed 변동은 분리해야 한다.** 두 번째 local BASE가 없으므로 second-seed treatment effect는 미확정이다.

### 4.3 E1

첫 100K pair에서 E1은 H/E/Ds가 모두 불리하며 Dλ만 낮다. 두 번째 E1 S95002는 E=0.550514/H=0.951072이나, 같은 seed BASE가 없어 성공 pair로 세지 않는다.

과거 G20의 E1 3/4 H 개선은 특정 TA/50K 조건의 결과다. 새 reference와 긴 학습에서는 별도로 확인해야 한다. E1의 전역적인 열등함을 한 pair로 확정하지 않지만, 'edge를 더 크게 줄이면 해결된다'며 E02를 자동 개방할 근거도 없다.

## 5. 남아 있는 병목

100K L100 8개 모두 final signed-Ds positive_fraction=1.0으로 기록됐다. 장면20×밴드4의 high-minus-low Q 차이 80개가 전부 양수라는 의미다. 재구성된 Ds와 공식 Ds의 max absolute error는 모두0으로 등록돼 있다. 본 검토에서 원80개 operand를 재계산한 것은 아니다.

따라서 낮은 ERGAS를 확보한 모델에서도 PAN–output의 고해상도 품질지수 관계가 저해상도 reference 관계보다 높은 쪽의 차이가 남는다. 이는 순수 Pearson correlation이나 시각적인 과선명화의 직접 증명이 아니며, warp가 과도하다는 증명도 아니다.

H010/S95001의 Dλ=0.0194497을 고정하고 평균 값의 곱으로 H=.964를 근사하면 필요한 Ds는 약0.0168785다. 현재0.0274011 대비 약38.4% 감소가 필요하다. 실제 HQNR는 장면별 값의 평균이므로 이 계산은 개선량 감각을 위한 근사이며 정확한 달성 보장선이 아니다.

이전 사용자 제공 진단에서는 50K GF2의 Teacher→Student correction이 유지·증폭되고 추정 잔여 정합도 개선됐다. 새100K의 높은 Ds를 보자마자 그 해석을 뒤집어 'A가 잘못 움직인다'고 결론내릴 수 없다. 이번에는 c 분포나 U/A 기여를 직접 읽지 않았다.

## 6. 권고하는 다음 순서 — 논의안, 실행 승인/queue 변경 아님

### P0. 이미 거의 완성된 대응 실험을 먼저 마무리

- s4 SS95002 BASE(S10), s5 SS95002 BASE(S14)를 먼저 회수한다. 이미 ALT가 있으므로 두 BASE의 정보 가치가 높다.
- s3 SS95002의 길이 triple(S04–S06)은 1seed 역전의 재현성 검정이다.
- 이미 실행/평가 중인 작업을 중복 시작하지 않는다. 로컬 status/admission/evaluation-debt를 먼저 확인하고, 시간 허용 시 남은 완결 block을 끝낸다.
- 현재 bundle의 T50/T100과 잘 나온 Student를 보존한다. 성능만 보고 Teacher를 반복 교체하지 않는다.

### P1. 기존 R50 reference와 H010의 조합을 통제된 방식으로 검정

가장 먼저 새로 열어볼 질문은 **'T100의 FR 비용 없이, H010의 RR 이득을 유지할 수 있는가?'**다.

s3에는 local T50/T100과 같은 seed의 S100 BASE 대조가 이미 있다. 해당 조건에서 Student100K H010을 추가하면 다음 2×2를 만들 수 있다.

| Teacher endpoint | alpha=1 | alpha=.1 |
|---|---|---|
| T50 | S02: 완료 | 새 대조 |
| T100 | S03: 완료 | 새 대조 |

기존 BASE 재사용은 source/data/reference/initialization 규약이 동일한 경우로 제한한다. 이는 Teacher length × hard emphasis의 상호작용 검정이며 s4 H010의 이득을 그대로 숫자로 더하는 예측이 아니다. 가능한 경우 같은 추가 조건을 두 Student seed로 비교한다.

이 대조에서 R50+H010이 좋으면 더 긴 Teacher를 의무화할 이유가 없다. R100+H010이 더 좋으면 그 조합을 유지할 근거가 생긴다. 어느 경우든 같은 architecture와 모든 nonzero FULL component를 유지한다.

### P2. 반복 수보다 Student U의 LR 경로를 변경

P1이 RR만 개선하고 Ds를 못 낮추면, U의 peak LR 또는 decay timing 한 축을 바꾼다. 전체100K와 동일한 A 스케줄·reference·loss를 유지하면서 U가 초반에 더 빨리 낮은 LR로 진입하도록 하는 schedule 대조가 후보이다.

예시 설계 방향: U peak를 BASE의0.5배로 낮춘100K 또는, 총100K는 유지하되 U의 LR를50K 부근에서 낮춘 뒤 낮은LR 구간을 길게 두는 것. 두 축을 동시에 바꾸지 않는다. 구체 수치는 다음 case 승인 때 고정하며 이번에 queue를 추가하지 않았다.

이것은 '더 오래 돌리면 해결된다'는 주장이 아니라, RR 정확도와 FR 관계를 만드는 학습 경로가 달라질 수 있다는 후속 가설이다. 결과가 나쁘면 해당 가설을 보류한다.

### P3. Teacher 새 학습보다 기존 endpoint 진단 우선

- 같은 고정 train/validation 표본에서 T50/T100과 최종 Student의 correction·GT 오차를 비교하고, 저장된 FR output의 signed Ds 변화를 분리한다.
- q_ref 하나를 조절하거나 correction을 임의로 줄여 공식 점수를 맞추지 않는다.
- 필요한 A/U 교차나 correction-gain 개입은 원인 진단으로만 수행하고, 재학습 ablation이나 공식 성능으로 취급하지 않는다.
- 과거 C300의 두 Student H 개선은 보조 후보로 보존하지만, 한 Teacher seed의 관측이다. 지금의 local Teacher를 중간 교체하거나 네 Teacher를 다시100K로 학습하는 것이 우선은 아니다.
- 데이터셋 특성/평가 방식 검증은 사용자가 진행 중인 별도 감사와 구분한다. 그 검정에서 새 근거가 나오면 training distribution 문제를 별도 revision으로 시험한다.

### 당장 우선순위를 낮출 방향

- TA/BASE의 무작위 Student seed만 계속 추가하기.
- T/S 모두150K·200K로 일괄 연장하기.
- 전체 학습 동안 A LR를 줄이기: 이전 A1의4seed 대응에서 반복 이득이 없었다.
- KD를 크게 줄이거나 늘리기, edge를10배 줄이기: 지금까지 해당 방향의 일관된 공간 이득이 없다.
- 아키텍처 확장, PAN image reconstruction task 추가, native 평가 reference 변경. 현재 승인된 method 범위를 넘거나 평가 조건을 바꾼다.

## 7. Ablation — 현황만

s1/WV3는 첫 P01의 Teacher2개와 C00/C01/C02 Student3개가 등록됐다. s2/QB는 현재 ABLR2 완료행이 없다. 이 부재는 idle이나 미실행의 증거가 아니다.

| Case | 시트 행 | HQNR | ERGAS | Dλ | Ds |
| --- | --- | --- | --- | --- | --- |
| TPLUS | WV3-s1!106 | 0.949305 | 2.047293 | 0.025582 | 0.025778 |
| TZERO | WV3-s1!107 | 0.950266 | 2.044604 | 0.021935 | 0.028439 |
| C00 | WV3-s1!108 | 0.950720 | 2.054384 | 0.022933 | 0.026986 |
| C01 | WV3-s1!109 | 0.951106 | 2.044009 | 0.018367 | 0.031127 |
| C02 | WV3-s1!110 | 0.951848 | 2.049955 | 0.017931 | 0.030783 |

모두 main=RR_VAL_SELECTED다. C00→C01은 LP/HP 추가, C01→C02는 scratch Aligner 추가이다. 첫1seed에서 H는 조금 오르지만 E는 2.054384→2.044009→2.049955로 단조롭지 않다. Teacher 사용 Student(C03 이후)나 FULL(C07)의 완료 결과가 없어 Teacher/KD/component 전체 기여를 판단할 수 없다. Teacher TPLUS/TZERO 자체 점수만으로 consistency의 Student 효과를 확정하지 않는다. 최초5sweep은 전혀 완성된 상태가 아니다.

## 8. 근거와 재현 파일

- [S1] live Google Sheets `pan-cvpr27`, 2026-09-21 21:19 KST export. 이 폴더의 snapshot을 읽고 raw 수치를 계산했다.
- [S2] native readback: `GF2-s3(5090)!B34:N36`, `GF2-s4!B14:N16`, `GF2-s5!B21:N23`.
- [S3] 저장소 commit `6bb78e6d61183a7ed1cd1da992527e7207d48ba3`, `l100/README.md`의 local queue·수치·평가 규약.
- [S4] 이전 `/mnt/data/results_training_time_review_20260921/GF2_selection_metrics.csv`와16:20 분석을 이전 cutoff로 사용했다.
- [S5] PAN-Crafter 첨부 PDF Table7의 H=.964/E=.552를 기본 운영 기준으로 유지한다. 본문Table2의 E=.522는 다른 보고값으로 별도 보존하며 임의로 통일하지 않는다.
- [S6] 앞선 사용자 제공 Q1–Q3 정합/hard diagnostic 요약은 당시50K 표본의 관측으로만 사용했다.

산출물:
- `GF2_L100_all_selection_metrics.csv`: 9개 L100의 endpoint/VAL/RAW/E_MIN/MID 선택점44개.
- `GF2_L100_paired_deltas.csv`: 같은-seed 핵심 비교5개×선택2개. 합산 비교는 개별효과와 별도 표기.
- `Ablation_completed_summary.csv`: 완료 ablation5개, FULL claim 없음.
- `pan-cvpr27_snapshot_20260921_2119KST.xlsx`: 원본 무수정 copy.
- `raw_records.json`: 읽은 현행 campaign metadata, provenance 포함.

분석 도중 workbook import의 comment/person metadata 오류가 발생해, 원본xlsx의 저장 cell값을 OOXML 읽기 전용으로 추출했다. 주요 비교값은 native Sheets readback으로 대조했다. 원본 workbook의 formula·서식·값을 수정하거나 재계산한 것은 아니다.
