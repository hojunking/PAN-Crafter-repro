# S2 최종 실험 계획 — NA104 전체 보류 case 포함

**작성일:** 2026-09-12 · **버전:** FINAL-v1  
**서버:** s2 · **Backbone:** W104·D122 · **입력:** WV3 9ch · **Aligner:** 없음  
**목적:** 동일 용량 Teacher–Student의 학습 신호로 fitting 및 HQNR을 개선한다. 경량화 실험이 아니다.  
**주 지표:** 원본 FR 논문 세트 20장의 HQNR. ERGAS는 보조 진단.  
**시간:** 캠페인 시간 상한 없음. case별 50K/25K tail/100K horizon은 고정.  
**전달 상태:** 실행 계획서이며 서버 queue·코드·시트는 변경하지 않았다.

> 이 파일은 s2에 단독 전달할 수 있도록 해당 서버의 전체 case·순서·대조와 공통 계약을 포함한다. 원 ID와 완료 결과는 보존한다. 기존 보류군의 첫 탐색은 전부 편성하며, 후보 선발은 추가 확장·최종 채택에 적용한다. 모든 조합이 효과적이라는 가정은 하지 않는다.

## S2-0. 전달 결론

**s2는 reconstruction 신호·구조 감독의 상호작용, 방향/성분 gate와 그 대조군, C 민감도, 초기화·장기 fitting을 담당한다.**

가장 먼저 할 일은 새 모델을 만드는 것이 아니다. 기존 Q09(R3+GV-FIX, HQNR 0.9566), Q12(R3+EDGE-H, 0.9560)의 공통 격자·장면별 재평가를 정리하고, 아래 X01–X08로 어느 감독이 필요한지 분리한다. s3의 Q09/Q12 결과를 기다리는 동안 s2는 자체 대조·seed 반복을 진행한다. 서로의 완료를 기다리며 두 서버가 모두 대기하지 않는다.

**원래 보류 34개를 전부 단계 P3–P7에 넣었다.** 기존 case는 재기반화하지 않고 원 R3/GV-AD 정의대로 1회 탐색한다. 그와 별도로 현재 유력한 GV-FIX/EDGE의 장기 경로를 PX/CX/LX 새 ID로 추가한다. 따라서 “R3 또는 AD가 단독으로 나빠서 그 위 실험을 전부 취소”하지 않으면서, 후보 구성요소도 분해할 수 있다.

## S2-1. 실행 순서 요약

1. **P0:** 완료 결과·hash·공통 HQNR 평가·fitting/gradient 산출물을 확인한다. 정상 완료 run은 재학습하지 않는다.
2. **P1:** Q36(N0+GC-H)을 s2에서 확인해 s3의 약한 긍정 후보에 같은 서버 대조를 만든다.
3. **P2:** X01/X02/X03/X04, 통계 모드 분해 X05/X06, T 강도 대응 X08.
4. **R1 반복:** core10을 Student seed777에서 대응 실행한다.
5. **P3–P4:** hard 재가중·β/τ/λ 대조, TRI-A/B와 MASS/SHUF를 모두 수행한다.
6. **R2 반복:** 같은 core10을 Student seed2026에서 대응 실행한다.
7. **P5:** C 진단만→민감도 감쇠→총량 대조→A/B와의 결합. no-align 입력은 유지한다.
8. **P6–P7:** 기존 TCOPY/CONT/LONG 전부 + GV-FIX/EDGE의 새 경로 + 실제 시간 대응 COSTMATCH.
9. **원 반복 보존:** Q00/Q04/Q10-S2025, Q10-S777. seed2025의 Teacher 재현 여부를 별도 표시한다.

위 번호는 queue 편성 순서다. 완료·실행 중인 것은 중복 제외하고, 기술적 blocker가 있는 묶음만 건너 다음 준비된 묶음을 진행한다. 과학적 결과가 낮다는 이유로 원 보류 정의를 삭제하지 않는다.

## S2-2. 현재 완료 보고분 — 재학습보다 산출물 검증

| ID | 구성 | HQNR original best | 해당 서버 N0 대비 | 처리 |
|---|---|---|---|---|
| T00 | N0 / OFF | 0.9498 | Teacher | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q00 | N0 / OFF | 0.9533 | +0.0000 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q01 | R0 / OFF | 0.9530 | -0.0003 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q02 | R1 / OFF | 0.9519 | -0.0014 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q03 | R2 / OFF | 0.9546 | +0.0013 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q04 | R3 / OFF | 0.9520 | -0.0013 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q05 | N0 / GVH | 0.9484 | -0.0049 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q11 | N0 / EDGEH | 0.9511 | -0.0022 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q06 | R3 / GVH | 0.9529 | -0.0004 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q07 | R3 / GVWH | 0.9531 | -0.0002 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q10 | R3 / GVAD | 0.9514 | -0.0019 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q09 | R3 / GVFIX | 0.9566 | +0.0033 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q08 | R3 / GVT | 0.9516 | -0.0017 | SHEET_REPORTED_50K → local 검증 후 재사용 |
| Q12 | R3 / EDGEH | 0.9560 | +0.0027 | SHEET_REPORTED_50K → local 검증 후 재사용 |

이 표는 2026-09-12 시트 snapshot이다. 새 결과·완료 상태가 생기면 실행 전 inventory에서 차감한다. 값이 같은 다른 서버 run을 복사해 자기 서버 결과로 만들지 않는다. 같은 이름의 결과도 server·Teacher·init·calibration hash가 다른 별개의 block이다. [S2]

## S2-3. 실제 실행 전 P0 확인

| 검사 | 완료 조건 | 실패 시 |
|---|---|---|
| Backbone/no-align | width104/depth122/9→8ch, 원 PAN byte-identity, base 1회 | 학습 시작하지 않음 |
| Teacher·pilot | local T00/best_hqnr·Q00 S1234/last hash와 실제 가중치 일치 | 해당 family만 정리 후 재개 |
| 초기값·데이터 | 같은 seed의 모든 대조가 동일 init·data policy, calibration RNG 격리 | 과거 run 오염 여부를 분리 기록 |
| HQNR | 동일 evaluator/raw_original/mat20, 원 best 및 common-grid 연결 | 새 결과 선발 보류, 기존 결과 보존 |
| 계측 | N0 포함 공통 bin, hard/soft/stat 분리 gradient, real cost | 가능하면 저장 checkpoint 사후 분석 |
| Run 상태 | 목표 update·마지막 checkpoint·평가 metadata 검증 | 폴더 존재만으로 완료 처리 금지 |
| 신규 case | X/PX/CX/LX가 실제 registry·config에 존재하고 직접 대조와 diff 확인 | `NEEDS_IMPLEMENTATION` 유지 |

읽기 전용 분석 도구는 이미 저장소에 있다. 대상은 **완료된 같은-horizon 비교 묶음**으로 명시한다. 전체 `NA104_*` glob에 초기 checkpoint나 진행 중 run이 섞이면 공통 구간이 잘릴 수 있다.

```bash
# 형식 예. 실제 존재하는 완료 run ID만 넣고, 결과는 새로운 분석 디렉터리로 저장한다.
python tools/na104_hqnr_report.py --grid 10 \
  --baseline NA104_Q00_W104_D122_WV3_N0_OFF_S1234_v1 \
  --out analysis/final_s2_20260912_core \
  NA104_Q00_W104_D122_WV3_N0_OFF_S1234_v1 \
  NA104_Q04_W104_D122_WV3_R3_OFF_S1234_v1
```

원래 best와 공통 격자 점수·checkpoint 보존 여부를 같이 기록한다. 다른 timestamp의 Dλ·Ds를 붙이지 않는다.

## S2-4. P1 — 상호 서버 후보 보강

| ID | 구성 | 질문/변경 | 직접 대조 | 추가 update | 구분 |
|---|---|---|---|---|---|
| Q36 | N0 / GCH | Teacher-free 학습: GC GT 통계 표현 자체의 효과 | Q00 | 50,000 | 기존 정의 |

Q36은 신규 loss가 아니라 s3에 이미 있는 기존 정의의 s2 반복이다. s2 Q00와 같은 초기값·데이터 정책, s2의 기존 λ pilot을 쓴다. s3의 λ 숫자를 그대로 복사하지 않는다.

## S2-5. P2 — 구성요소·통계 hard/soft 분리

| ID | 구성 | 질문/변경 | 직접 대조 | 추가 update | 구분 |
|---|---|---|---|---|---|
| X01 | R1 / GVH | R1 + GT gradient variance | Q02; Q06; Q05 | 50,000 | 신규 정의 |
| X02 | R1 / EDGEH | R1 + signed GT edge | Q02; Q12; Q11 | 50,000 | 신규 정의 |
| X03 | N0 / GVFIX | 일반 GT reconstruction + 고정 GT/Teacher 통계 KD | Q05; Q09 | 50,000 | 신규 정의 |
| X04 | R1 / GVFIX | Teacher-error hard reconstruction + 고정 통계 KD | X01; Q09; X03 | 50,000 | 신규 정의 |
| X05 | R3 / GVHAD | 통계 hard는 plain, soft만 AD: HV + βV(1-dV)aV KV | Q06; Q09; Q10 | 50,000 | 신규 정의 |
| X06 | R3 / GVWFIX | 통계 hard는 weighted, soft는 fixed: (1+αVdV)HV + βV KV | Q07; Q09; Q10 | 50,000 | 신규 정의 |
| X08 | R3 / GVTMATCH | Teacher-only 통계 계수를 FIX와 같은 βV=0.1로 대응 | Q08; Q09 | 50,000 | 신규 정의 |

**첫 번째 비교 묶음:** Q02(R1)↔X01(R1+GV-H)↔Q06(R3+GV-H), Q02↔X02(R1+EDGE-H)↔Q12(R3+EDGE-H). R3 조합의 개선에 output soft가 필요한지 확인한다.

**두 번째 묶음:** Q05(N0+GV-H)↔X03(N0+GV-FIX), X01↔X04, Q06↔Q09. 같은 reconstruction마다 GT 통계 위에 Teacher 통계를 추가하는 효과를 나눈다.

**세 번째 묶음:** Q06(H), Q09(FIX), Q07(WH), Q10(AD), X05(H+adaptive soft), X06(weighted H+fixed soft). hard 가중과 soft gating을 분리한다. X08은 T의 강도 차이를 보완한다.

이 묶음에서 λ를 후보별로 다시 맞추면 한 요인 대조가 깨진다. 표현 GV/w5/최종 HRMS가 같으면 s2의 고정 λ를 공유한다. 결과가 좋다고 X05를 AD라는 이름으로 업로드하지 않는다.

**이 단계 다음:** 아래 seed777 core10을 실행한 뒤 다음 P단계로 넘어간다. 준비되지 않은 새 모드 때문에 다른 기존 case를 함께 기다리게 하지 않는다.

## S2-6. P3 — 재가중·계수 대조

| ID | 구성 | 질문/변경 | 직접 대조 | 추가 update | 구분 |
|---|---|---|---|---|---|
| CTLHSCALE | R1HSCALE / OFF | R1의 공간 hard weight를 batch 평균 scalar로 치환 | Q02 | 50,000 | 기존 정의 |
| X07 | R1RSHUF / OFF | R1의 d 지도만 위치 shuffle; soft는 0 유지 | Q02; CTLHSCALE | 50,000 | 신규 정의 |
| CTLRSHUF | R3RSHUF / OFF | R3의 d·a를 같은 공간 permutation으로 섞음 | Q04 | 50,000 | 기존 정의 |
| CTLBETA03 | R3 / OFF | βR=0.3, 나머지 R3 유지 | Q04 | 50,000 | 기존 정의 |
| CTLBETA05 | R3 / OFF | βR=0.5, 나머지 R3 유지 | Q04 | 50,000 | 기존 정의 |
| CTLTAU05 | R3 / OFF | τR를 train-calibrated 값의 0.5배 | Q04 | 50,000 | 기존 정의 |
| CTLTAU20 | R3 / OFF | τR를 train-calibrated 값의 2배 | Q04 | 50,000 | 기존 정의 |
| CTLLAMV03 | R3 / GVAD | λV×0.3, 통계 AD 유지 | Q10 | 50,000 | 기존 정의 |
| CTLLAMV30 | R3 / GVAD | λV×3, 통계 AD 유지 | Q10 | 50,000 | 기존 정의 |

CTLHSCALE은 hard weight의 공간 배치와 평균 배율을 분리한다. X07은 R1 전용 shuffle이며, CTLRSHUF는 원 R3의 d/a 공동 shuffle이다. 두 case는 다른 질문이다.

β/τ/λ는 한 번에 하나만 바꾸며 기본 α=1을 유지한다. CTLLAMV03/30은 **GV-AD에 대한 계수 민감도**이지 Q09/GV-FIX의 최적 계수 탐색이라고 쓰지 않는다. FIX 계수 탐색이 나중에 필요하면 별도 ID로 만든다. τ 중앙값 보정은 확률 신뢰도 calibration이 아니다.

## S2-7. P4 — 픽셀 방향 및 통계 성분 선택

| ID | 구성 | 질문/변경 | 직접 대조 | 추가 update | 구분 |
|---|---|---|---|---|---|
| Q13 | R3 / OFF / ASIGN_BOFF_COFF | 방향이 GT와 같은 밴드의 soft만 남김 | Q04; CTLAMASS; CTLASHUF | 50,000 | 기존 정의 |
| CTLAMASS | R3 / OFF / AMASS_BOFF_COFF | A-SIGN과 soft 계수 총량만 대응 | Q13 | 50,000 | 기존 정의 |
| CTLASHUF | R3 / OFF / ASHUF_BOFF_COFF | A mask 위치만 섞음 | Q13 | 50,000 | 기존 정의 |
| Q14 | R3 / OFF / ACOS_BOFF_COFF | 벡터 cosine 방향 감쇠 | Q04; Q13 | 50,000 | 기존 정의 |
| Q15 | R3 / OFF / ACAP_BOFF_COFF | SIGN + 필요한 수정 대비 크기비 감쇠 | Q04; Q13 | 50,000 | 기존 정의 |
| Q16 | R3 / OFF / ABANDADV_BOFF_COFF | 밴드별 τ·상대우위로 soft를 재판단 | Q04; Q13 | 50,000 | 기존 정의 |
| Q17 | R3 / GVAD / AOFF_BSIGN_COFF | 통계 원소의 GT 방향 soft 선택 | Q10 | 50,000 | 기존 정의 |
| CTLBMASS | R3 / GVAD / AOFF_BMASS_COFF | GV B-SIGN과 soft 계수 총량만 대응 | Q17 | 50,000 | 기존 정의 |
| CTLBSHUF | R3 / GVAD / AOFF_BSHUF_COFF | GV B mask 위치만 섞음 | Q17 | 50,000 | 기존 정의 |
| Q18 | R3 / GVAD / AOFF_BCAP_COFF | 통계 SIGN + 크기비 감쇠 | Q10; Q17 | 50,000 | 기존 정의 |
| Q19 | R3 / GVAD / AOFF_BCOMPADV_COFF | 통계 원소별 τ·상대우위로 soft 재판단 | Q10; Q17 | 50,000 | 기존 정의 |

A-SIGN/COS/CAP과 BANDADV는 같은 유형의 감쇠가 아니다. BANDADV는 평균 gate가 놓친 밴드의 Teacher 우위를 복구할 수 있다. B-COMPADV도 같은 차이를 갖는다.

MASS는 각 run의 현재 출력에서 계산한 mask와 기본 coefficient로 **계수 총량**을 맞추는 대조다. 서로 다른 학습 궤적의 실제 loss·gradient 총량까지 같다고 주장하지 않는다. SHUF의 permutation은 별도 generator를 사용하고 위치 분포를 보존한다.

부정적인 R3/GV-AD 기반도 원 정의로 한 번씩 완료한다. gate가 단순히 KD를 줄여 좋아진 것인지 확인하려면 대응 MASS·SHUF까지 같이 보고한다.

**이 단계 다음:** 아래 seed2026 core10을 실행한 뒤 나머지 P단계로 넘어간다.

## S2-8. P5 — 입력 민감도 C 분기

| ID | 구성 | 질문/변경 | 직접 대조 | 추가 update | 구분 |
|---|---|---|---|---|---|
| CS00 | R3 / OFF / AOFF_BOFF_CSENSDIAG | C 진단만. 학습 loss는 Q04와 동일 | Q04 | 50,000 | 기존 정의 |
| CS01 | R3 / OFF / AOFF_BOFF_CSENS | raw PAN 입력 민감도로 output soft만 감쇠 | CS00; CTLCMASS | 50,000 | 기존 정의 |
| CTLCMASS | R3 / OFF / AOFF_BOFF_CSENSMASS | C-NASENS와 감쇠 총량만 대응 | CS01 | 50,000 | 기존 정의 |
| CS02 | R3 / OFF / ASIGN_BOFF_CSENS | A-SIGN + C-NASENS | Q13; CS01 | 50,000 | 기존 정의 |
| CS03 | R3 / GVAD / AOFF_BSIGN_CSENSV | 통계 Jacobian을 직접 재계산해 B soft 감쇠 | Q17 | 50,000 | 기존 정의 |

이 묶음은 `NA-TSENS`다. CS00은 감쇠 없이 민감도만 계산하므로 Q04와 학습 수식은 같고, 새로운 성능 방법으로 세지 않는다. 진단이 RNG·배치 순서·optimizer를 바꾸지 않는지 확인한다.

CS01/02는 pixel Jacobian, CS03은 **통계 Jacobian**을 쓴다. ROI 밖은 r=1로 원 soft에 복귀하며 GT hard는 전역 유지한다. h=0.05에 대해 h/2·2h 수치 대조와 same-ROI calibration을 통과한 뒤 full run을 시작한다.

정합 module이 없으므로 covariance를 identity로 만들어 C-COV나 G-KD를 억지로 실행하지 않는다. C가 blocked여도 다른 단계는 계속한다.

## S2-9. P6 — Teacher-copy·공통 parent continuation

| ID | 구성 | 질문/변경 | 직접 대조 | 추가 update | 구분 |
|---|---|---|---|---|---|
| TCOPYN0 | N0 / OFF | Teacher best_hqnr에서 가중치만 복사; fresh optimizer로 추가 50K | 그룹 기준 | 50,000 | 기존 정의 |
| TCOPYR1 | R1 / OFF | Teacher best_hqnr에서 가중치만 복사; fresh optimizer로 추가 50K | TCOPYN0 | 50,000 | 기존 정의 |
| TCOPYR3 | R3 / OFF | Teacher best_hqnr에서 가중치만 복사; fresh optimizer로 추가 50K | TCOPYN0 | 50,000 | 기존 정의 |
| PX01 | R3 / GVFIX | 유력 구조 감독 GVFIX의 Teacher-copy | TCOPYN0 | 50,000 | 신규 정의 |
| PX02 | R3 / EDGEH | 유력 구조 감독 EDGEH의 Teacher-copy | TCOPYN0 | 50,000 | 신규 정의 |
| CONTN0 | N0 / OFF | 공통 Q00/last에서 fresh optimizer로 동일 tail 25K | 그룹 기준 | 25,000 | 기존 정의 |
| CONTR3 | R3 / OFF | 공통 Q00/last에서 fresh optimizer로 동일 tail 25K | CONTN0 | 25,000 | 기존 정의 |
| CONTGVAD | R3 / GVAD | 공통 Q00/last에서 fresh optimizer로 동일 tail 25K | CONTN0 | 25,000 | 기존 정의 |
| CX01 | R3 / GVFIX | 유력 구조 감독 GVFIX의 공통 parent continuation | CONTN0 | 25,000 | 신규 정의 |
| CX02 | R3 / EDGEH | 유력 구조 감독 EDGEH의 공통 parent continuation | CONTN0 | 25,000 | 신규 정의 |

Teacher-copy는 원 Teacher **가중치만** 복사한다. 새 optimizer·schedule로 시작하며 Teacher 자신은 frozen이다. 시작에서 Student=Teacher라 R3 soft=0인 것은 정상이다. TCOPYN0를 빼고 독립 scratch만 기준으로 비교하지 않는다.

CONTN0/CONTR3/CONTGVAD/CX01/CX02는 모두 **같은 Q00/last**에서 분기한다. 각각의 방법별 best를 parent로 쓰지 않는다. 기본 legacy tail은 생성기의 fresh optimizer·25K recipe를 유지한다. metadata의 parent_step은 실제 parent에서 읽는다.

## S2-10. P7 — 100K horizon 및 시간 대응

| ID | 구성 | 질문/변경 | 직접 대조 | 추가 update | 구분 |
|---|---|---|---|---|---|
| LONG2NN0 | N0 / OFF | 독립 초기화, 처음부터 100K cosine horizon | 그룹 기준 | 100,000 | 기존 정의 |
| LONG2NR1 | R1 / OFF | 독립 초기화, 처음부터 100K cosine horizon | LONG2NN0 | 100,000 | 기존 정의 |
| LONG2NR3 | R3 / OFF | 독립 초기화, 처음부터 100K cosine horizon | LONG2NN0 | 100,000 | 기존 정의 |
| LONG2NGVAD | R3 / GVAD | 독립 초기화, 처음부터 100K cosine horizon | LONG2NN0 | 100,000 | 기존 정의 |
| LX01 | R3 / GVFIX | 유력 구조 감독 GVFIX의 100K horizon | LONG2NN0 | 100,000 | 신규 정의 |
| LX02 | R3 / EDGEH | 유력 구조 감독 EDGEH의 100K horizon | LONG2NN0 | 100,000 | 신규 정의 |
| COSTMATCH | N0 / OFF | 참조 KD의 사전 측정 시간 budget과 대응한 GT-only 학습 | 확정 후보의 동일 서버 run | 실측 budget으로 결정 | 생성기만 존재 |

100K는 처음부터 100K cosine이다. 완료된 50K run 뒤에 LR=0인 채 update만 추가해 LONG이라고 부르지 않는다.

COSTMATCH는 현재 생성기에서만 확인된 정의다. 기준 KD와 예산 범위를 정하고 실제 throughput을 측정한 뒤 N을 확정한다. 아직 N이 없으면 실행하지 말고 `BLOCKED_COST_MEASUREMENT`로 기록한다. 미정 숫자를 50K로 채워 완료 처리하지 않는다.

GV-FIX/EDGE가 첫 반복에서 좋지 않더라도 원보류 LONG 비교는 유지한다. 새 PX/CX/LX는 현재 유력 후보의 학습 경로를 놓치지 않기 위한 사전 지정 확장이며 결과를 전부 남긴다.

## S2-R. 반복·최종 확인 block

### R1/R2: 사전 지정 core10

`Q00, Q01, Q02, Q03, Q04, Q05, Q06, Q09, Q11, Q12`

- P2 다음 Student seed777, P4 다음 Student seed2026에서 각각 10개를 실행한다.
- 기존 seed1234와 합쳐 3개의 Student seed 조건을 비교한다. 매 seed에서 Q00을 먼저 실행한다.
- 같은 seed의 Student 초기값과 데이터 순서는 대응 case 사이에 동일하게 유지한다.
- Teacher는 기존 local seed2025 T00/best_hqnr hash를 계속 사용한다. 반복 Student와 함께 새 Teacher를 무심코 생성하지 않는다.
- λ pilot은 기존 Q00 S1234 v1/last로 고정한다. 해당 seed의 N0 baseline과 혼동하지 않는다.
- 기존 generator의 `--repeat`만으로 core10이 전부 생성되지 않는다. 명시적 목록을 구현하고 출력 개수·resolved config를 확인한다.

### R3: 원 반복안 보존

| Case | 추가 seed | 이유 |
|---|---|---|
| Q00, Q04, Q10 | 2025 | 기존 생성기의 반복 제안을 삭제하지 않음. Teacher 재현 여부를 별도 기록 |
| Q10 | 777 | 원 반복안의 Q10-S777이 core10에 없으므로 추가 |

seed2025가 Teacher 초기값·데이터 순서를 재현하면 `TIED_TO_TEACHER_SEED`로 표시한다. 그 run은 흥미로운 sensitivity/sanity 대조지만, 독립 T/S 참조가 유효하다는 증거로 과장하지 않는다.

### 최종 후보 확인

전수 1-seed 탐색과 core 반복 후, HQNR 기준 후보 **최대 2개 family**를 사전 고정하고 그 직접 대조도 같은 Student seed777/2026에서 확인한다. 후보명·parent·Teacher·λ·주 평가를 `confirmation_manifest`에 기록한 뒤 실행한다. 새 seed 결과를 보고 매번 후보를 바꿔 최고점만 취하지 않는다.

이 후속 후보 확인의 정확한 run 수는 winner와 필요한 직접 대조에 따라 달라지므로 아래 고정 슬롯 합계에 넣지 않았다. 기존 R3/GV-AD의 부정 결과도 기록한다.

## S2-C. 작업량과 완료 기준

| 항목 | 계획 슬롯 |
|---|---:|
| 이미 시트에 보고된 local run | 14 |
| 새로 실행 또는 local 상태 확인할 고정 편성 | 74 |
| 합계 | 88 |

이 숫자는 **서로 다른 방법 수가 아니라 server×case×seed×version의 편성 수**다. COSTMATCH 1개는 실제 update를 정해야 하므로 측정 전에는 시작하지 않는다. 새 결과가 이미 완료됐으면 동일 identity를 확인한 뒤 차감한다. 정상 완료분을 큐 목록에 있다는 이유만으로 다시 실행하지 않는다. 추가 confirmation·기술적 재실행은 별도 집계한다.

시간 예산은 제한하지 않는다. 예상 완료일은 새 case별 실제 초/step과 평가·진단 시간을 측정한 뒤 갱신한다. s3의 GPU가 빠르다는 이유로 같은 원리의 대조 하나만 다른 서버에 보내지 않는다. 대응 묶음을 함께 옮기거나 계획대로 local 완료한다.

완료는 `모든 legacy 첫 탐색이 ANALYZED 또는 사유가 있는 BLOCKED_TECHNICAL`이고, 사전 지정 core 반복이 끝나며, 최종 유력 후보와 단순 대조의 HQNR 관계가 설명되는 상태다. 기술적으로 막힌 것을 성능 0으로 채우거나 “효과 없음”으로 기록하지 않는다.

## S2-A. 원래 보류 목록 전수 대응표

| 원 보류 ID | 현재 배치 | 처리 |
|---|---|---|
| Q13 | P4 | 원 정의 보존·1회 full-budget 탐색 |
| Q14 | P4 | 원 정의 보존·1회 full-budget 탐색 |
| Q15 | P4 | 원 정의 보존·1회 full-budget 탐색 |
| Q16 | P4 | 원 정의 보존·1회 full-budget 탐색 |
| Q17 | P4 | 원 정의 보존·1회 full-budget 탐색 |
| Q18 | P4 | 원 정의 보존·1회 full-budget 탐색 |
| Q19 | P4 | 원 정의 보존·1회 full-budget 탐색 |
| CTLHSCALE | P3 | 원 정의 보존·1회 full-budget 탐색 |
| CTLRSHUF | P3 | 원 정의 보존·1회 full-budget 탐색 |
| CTLAMASS | P4 | 원 정의 보존·1회 full-budget 탐색 |
| CTLASHUF | P4 | 원 정의 보존·1회 full-budget 탐색 |
| CTLBMASS | P4 | 원 정의 보존·1회 full-budget 탐색 |
| CTLBSHUF | P4 | 원 정의 보존·1회 full-budget 탐색 |
| CTLTAU05 | P3 | 원 정의 보존·1회 full-budget 탐색 |
| CTLTAU20 | P3 | 원 정의 보존·1회 full-budget 탐색 |
| CTLBETA03 | P3 | 원 정의 보존·1회 full-budget 탐색 |
| CTLBETA05 | P3 | 원 정의 보존·1회 full-budget 탐색 |
| CTLLAMV03 | P3 | 원 정의 보존·1회 full-budget 탐색 |
| CTLLAMV30 | P3 | 원 정의 보존·1회 full-budget 탐색 |
| CS00 | P5 | 원 정의 보존·1회 full-budget 탐색 |
| CS01 | P5 | 원 정의 보존·1회 full-budget 탐색 |
| CS02 | P5 | 원 정의 보존·1회 full-budget 탐색 |
| CS03 | P5 | 원 정의 보존·1회 full-budget 탐색 |
| CTLCMASS | P5 | 원 정의 보존·1회 full-budget 탐색 |
| TCOPYN0 | P6 | 원 정의 보존·1회 full-budget 탐색 |
| TCOPYR1 | P6 | 원 정의 보존·1회 full-budget 탐색 |
| TCOPYR3 | P6 | 원 정의 보존·1회 full-budget 탐색 |
| CONTN0 | P6 | 원 정의 보존·1회 full-budget 탐색 |
| CONTR3 | P6 | 원 정의 보존·1회 full-budget 탐색 |
| CONTGVAD | P6 | 원 정의 보존·1회 full-budget 탐색 |
| LONG2NN0 | P7 | 원 정의 보존·1회 full-budget 탐색 |
| LONG2NR1 | P7 | 원 정의 보존·1회 full-budget 탐색 |
| LONG2NR3 | P7 | 원 정의 보존·1회 full-budget 탐색 |
| LONG2NGVAD | P7 | 원 정의 보존·1회 full-budget 탐색 |

**원 보류 34개 중 누락 0개.** full run name은 동봉 `NA104_S2_FINAL_STAGE_PLAN.json`에 수록했다. 기존 run은 v1 이름을 유지하고, 신규 정의/추가 seed는 v2로 구분한다.

---
# 부록: 단독 전달용 공통 규약

아래는 공통 문서의 loss·평가·seed·구현 규약을 같은 내용으로 포함한 것이다. 서버별 표와 충돌하면 새 결과에 따라 임의 해석하지 말고 버전 변경을 기록한다.

### 3. 공통 forward와 loss 계약

#### 3.1 Forward

$$M=U_4(MS),\quad \hat Y_i=M+F_i([P,M]),\qquad i\in\{T,S\}.$$

Teacher는 eval/frozen/no-grad, Student만 optimizer에 들어간다. GT는 loss·calibration·분석용이며 추론 입력이 아니다. 원 PAN을 직접 전달한다. `warp(P,0)`을 identity로 대신 호출하지 않는다. 기본 GT reconstruction은 전체 기존 학습 영역을 유지한다.

#### 3.2 Reconstruction: 기존 정의 그대로

밴드 평균을 픽셀 p에서 계산한다.

$$H(p)=\operatorname{mean}_c|\hat Y_{S,c}-Y_c|,\quad K(p)=\operatorname{mean}_c|\hat Y_{S,c}-\operatorname{sg}(\hat Y_{T,c})|.$$
$$e_T=\operatorname{mean}_c|\hat Y_T-Y|,\quad e_S=\operatorname{sg}(H),\quad d_T=\frac{e_T}{e_T+\tau_R},\quad a_T=\frac{[e_S-e_T]_+}{e_S+\epsilon_R}.$$

| REC | 목적함수 | 주 비교 |
|---|---|---|
| N0 | ⟨H⟩ | GT-only 기준 |
| R0 | ⟨H+βR K⟩ | R0−N0: 일반 output KD |
| R1 | ⟨(1+αR dT)H⟩ | R1−N0: Teacher 실패 지도 |
| R2 | ⟨(1+αR dT)H+βR(1−dT)K⟩ | R2−R1: 난이도 기반 soft |
| R3 | ⟨(1+αR dT)H+βR(1−dT)aT K⟩ | R3−R2: 상대우위 마진, R3−R1: soft 추가 |

기본 αR=1, βR=0.1, εR=1e−6. τR는 해당 frozen Teacher와 정규화·train calibration에서 얻은 값이다. 과거 W112의 τ나 λ를 숫자 그대로 복사하지 않는다. gate는 detach하고 H/K의 Student 경로는 live로 유지한다. R1 soft=0은 정상이며 R1을 N0의 재실행으로 분류하지 않는다. [S1, S5]

#### 3.3 통계 표현과 모드

$$V_G=V(Y),\quad V_T=V(\hat Y_T),\quad V_S=V(\hat Y_S).$$
$$H_V(p)=\operatorname{mean}_j|V_S-V_G|,\quad K_V(p)=\operatorname{mean}_j|V_S-V_T|.$$

`dV`, `aV`는 Teacher/Student의 GT 통계 오차에서 REC와 같은 형태로 구하되, **해당 표현의 τV·εV**를 쓴다. 공간·성분 축을 구분하고 정규화·평균 방식은 기존 구현과 맞춘다.

| 모드 | 추가 통계 loss |
|---|---|
| H | ⟨HV⟩ |
| T | ⟨KV⟩. 원 정의의 Teacher 계수는 1이다. |
| FIX | ⟨HV + βV KV⟩ |
| WH | ⟨(1+αV dV)HV⟩ |
| AD | ⟨(1+αV dV)HV + βV(1−dV)aV KV⟩ |

$$L=L_{REC}+\lambda_V L_{STAT};\qquad \alpha_V=1,\ \beta_V=0.1.$$

**T와 FIX는 Teacher 항의 계수도 다르다.** 따라서 T↔FIX를 GT 추가의 순수 효과로 해석하지 않는다. X08은 Teacher-only 항에 0.1을 곱해 이 혼동을 분리한다. H와 FIX의 대응 비교에서는 같은 λV를 유지한다. [S5]

| 표현 | 정의·WV3 성분 수 | 지켜야 할 것 |
|---|---|---|
| IV | 국소 Var(Zc), 8 | 주변 밝기 대비이며 uncertainty가 아님 |
| GV | Var(∂yZc), Var(∂xZc), 16 | `[dy 8, dx 8]`, signed Scharr/32 후 국소 분산 |
| GC | Cov((∂yZc,∂xZc)), 32 | band별 2×2 모두 사용. 대칭 비대각 두 원소도 기존 mean L1에 포함 |
| SC | Cov((Z1,…,Z8)), 64 | 8×8 row-major, 음의 비대각 원소를 지우지 않음 |
| M2 | E[ggᵀ], 32 | 중심화하지 않음. GC와 같지 않음 |
| EDGE | GT와 signed Scharr gradient 직접 비교 | variance 없음. 원 edge 구현의 support·reduction 그대로 |

기본 window5, stride1, population variance. Scharr valid+국소창의 margin은 k5에서 3이다. IV/SC는 공통 중심을 맞추기 위한 1px interior 처리 후 pooling한다. **통계 loss ROI와 공식 HQNR 평가 ROI를 혼동하지 않는다.** EDGE는 기존 별도 유효영역을 따른다. Student 통계는 FP32 이상의 live tensor, Teacher/GT 통계는 detach한다. [S5, S7]

#### 3.4 Calibration

동일 서버·같은 표현·창·domain의 H/T/FIX/WH/AD 비교는 **같은 Q00 S1234 v1/last pilot에서 얻은 λV**를 공유한다. 반복 seed에서도 이 λ를 유지하여 Student seed 효과와 loss scale 변화를 한꺼번에 바꾸지 않는다. 해당 seed의 비교 baseline은 별도의 Q00이지만, λ pilot은 seed1234로 고정한다.

- λV: 고정 train pilot에서 plain GT reconstruction과 STAT-H의 **출력 gradient RMS** 비를 초기 목표 0.05에 맞춘 기존 방식. 파라미터 gradient 비 또는 전체 학습에서 항상 5%라는 의미가 아니다.
- τV·εV: 해당 Teacher·표현·창·domain의 train 통계로 정한다. VX에서 창/변환/domain이 달라지면 따로 calibration한다.
- GC/SC/M2는 전체 원소 평균 L1이다. eigenvalue loss·Gaussian KL·역행렬 loss로 바꾸지 않는다.
- calibration의 데이터·augmentation·RNG·캐시 key·hash를 기록하고 전역 학습 RNG를 소비하지 않는다.
- 반복의 `--seed`만 바꾸어 새 Q00/last가 자동으로 λ pilot이 되는 것을 방지한다.
- 이 항목의 실제 캐시가 과거 run과 다르면, v1 결과를 덮어쓰지 않고 새 calibration block을 명시한다.

#### 3.5 TRI-A/B/C의 역할

A는 REC soft를 **대체**, B는 STAT soft를 **대체**한다. 기존 전체 loss에 A/B 전체식을 다시 더하지 않는다. SIGN/CAP은 soft mask이며 GT hard를 없애는 binary switch가 아니다. BANDADV/COMPADV는 원소별 우위를 재판단하므로 추가 감쇠와 다른 실험이다.

C는 이동량 KD가 아니다. δ=0에서 raw PAN을 ±0.05 HR px씩 probe하여 Teacher 출력 Jacobian을 중앙 유한차분으로 구하고 soft만 감쇠한다. 기본 Teacher forward 외에 **Teacher 복원 U-Net 4회**를 순차 no-grad로 추가한다. 통계 C에서는 V(T(δ))의 Jacobian을 직접 계산한다. finite difference·ROI·민감도 scale의 전제를 통과한 뒤 실행한다. G0–G5, C-COV/FULL/EQPROXY는 이번 no-align 전수 범위 밖이다. [S7]

### 4. 이번에 추가한 18개 정의 — 기존 case를 몰래 바꾸지 않는다

#### 4.1 구성요소 분해 X01–X08: 양쪽 서버에서 대응

| ID | 수식/설정 | 대조 | 구현 범위 |
|---|---|---|---|
| X01 | R1 + λV HV | Q02, Q06, Q05 | 기존 loss 조합, registry/config 추가 |
| X02 | R1 + λE LE | Q02, Q12, Q11 | 기존 loss 조합 |
| X03 | N0 + λV(HV+βV KV) | Q05, Q09 | 기존 loss 조합 |
| X04 | R1 + λV(HV+βV KV) | X01, Q09, X03 | 기존 loss 조합 |
| X05 | R3 + λV[HV+βV(1−dV)aV KV] | Q06, Q09, Q10 | **통계 plain hard + adaptive soft 새 모드** |
| X06 | R3 + λV[(1+αVdV)HV+βV KV] | Q07, Q09, Q10 | **통계 weighted hard + fixed soft 새 모드** |
| X07 | R1의 dT만 공간 shuffle. hard=1+αR dπ, soft=0 | Q02, CTLHSCALE | 기존 shuffle 재사용 가능 여부 검사 후 R1 전용 정의 |
| X08 | R3 + λV βV KV, βV=0.1 | Q08, Q09 | Teacher-only 통계 강도 대응. λV×0.1 구현이라면 base λ·multiplier를 따로 기록 |

X05/X06의 모드 토큰 `GVHAD/GVWFIX`는 **이번 계획의 새 이름**이다. 현재 resolver에 존재하는 것처럼 실행하지 않는다. 구현자가 명시적으로 추가하고 unsupported config는 오류로 막는다. 모든 X의 기본 α·β·τ·pilot·ROI는 대응 기존 block과 같다.

X07은 sample마다 공간 위치만 바꾸고 모든 밴드에 같은 permutation을 공유한다. GT나 입력을 shuffle하지 않는다. permutation은 분포를 보존하고 별도 RNG를 쓴다. 기존 CTLRSHUF는 R3의 d·a를 함께 섞는 다른 대조로 그대로 보존한다.

#### 4.2 표현별 방향 대조 X09–X12: s3 담당

| ID | 설정 | 대조 |
|---|---|---|
| X09 / X10 | GC-AD의 B-MASS / B-SHUF | Q38 GC-B-SIGN |
| X11 / X12 | SC-AD의 B-MASS / B-SHUF | Q39 SC-B-SIGN |

GV용 CTLBMASS/SHUF 결과를 GC·SC의 고유 대조로 대신하지 않는다. 같은 표현에서 mask와 기본 soft coefficient를 만든 후 총계수량/위치만 조절한다. 이 네 정의는 기존 연산을 재사용하더라도 새로운 config/ID이다.

#### 4.3 현재 후보의 장기·초기화 대조: s2 담당

| ID | 방식 | Parent / update | 필수 대조 |
|---|---|---|---|
| PX01 / PX02 | Teacher-copy 후 R3+GV-FIX / R3+EDGE-H | T00/best_hqnr → 추가 50K | TCOPYN0, TCOPYR1, TCOPYR3 |
| CX01 / CX02 | 공통 parent에서 R3+GV-FIX / R3+EDGE-H | Q00/last → 추가 25K | CONTN0, CONTR3, CONTGVAD |
| LX01 / LX02 | 처음부터 R3+GV-FIX / R3+EDGE-H | 독립 init → 100K | LONG2NN0, LONG2NR1, LONG2NR3, LONG2NGVAD |

부진한 GV-AD만 장기 비교하고 현재 유력한 GV-FIX/EDGE를 빠뜨리지 않기 위한 추가다. 이미 있는 TCOPY/CONT/LONG 정의는 삭제하지 않는다. 모든 새 ID는 `v2`로 분리하고 Teacher·calibration은 명시적으로 기존 local source를 지정한다.

### 5. HQNR 평가와 판정 규칙

#### 5.1 주 평가

`FR paper mat20`, `raw_original`, 같은 evaluator·MTF·정규화·export/clipping·원래 입력 PAN reference를 사용한다.

$$Q=\frac1{20}\sum_{i=1}^{20}(1-D_{\lambda,i})(1-D_{s,i}).$$

평균 Dλ와 평균 Ds를 곱해 Q를 재구성하지 않는다. HQNR(V64), 과거 H5 부분 세트, aligned reference, 구 버전 metric을 섞지 않는다. 주 selector는 **HQNR → fSCC → 후기 checkpoint**의 기존 구현 규칙이며 exact tolerance도 같은 코드로 맞춘다. `best_rr_val`는 보관하지만 선발 기준으로 바꾸지 않는다.

#### 5.2 선택 기회 대응

- 새 50K run은 기존 최신 기본값 **10 epoch 평가 간격**을 쓴다. 서로 다른 시점의 평가 기회를 추가하지 않는다.
- 과거 5-epoch run과 비교할 때는 실제 성공한 평가 시점 중 같은 10-epoch 후보의 교집합에서 재선택한다. 보간하지 않는다.
- 모든 unfinished run을 한꺼번에 넣어 공통 구간을 짧게 만들지 않는다. **완료·동일 horizon·같은 대조 block**별로 분석한다.
- epoch만 같다고 update가 같다고 가정하지 말고 실제 sample/update를 검증한다. 데이터 크기·drop_last·accumulation이 다르면 비교 block을 분리한다.
- 원래 best_hqnr를 덮어쓰지 않는다. 공통 선택 시점의 가중치가 없으면 `score_only`로 표기한다. 존재하지 않는 checkpoint 영상이나 장면별 분해를 생성했다고 쓰지 않는다.
- 같은 선택 checkpoint의 Dλ·Ds·fSCC와 장면별 결과를 붙인다. 서로 다른 checkpoint의 값으로 한 행을 조립하지 않는다.
- 별도로 **40K–50K plateau HQNR 및 last HQNR**을 기록한다. plateau는 안정성 설명이지 HQNR best를 몰래 대체하는 selector가 아니다.

100K horizon은 plateau 80K–100K, tail 25K는 tail 후반 20K–25K를 별도로 기록한다. 전체 누적 update와 tail update를 모두 남기고 50K plot과 같은 구간이라고 표시하지 않는다.

#### 5.3 paired 분석

모든 방법은 **같은 서버·Teacher hash·Student init/data seed·calibration protocol**의 N0 및 직접 부모 대조와 비교한다. s2의 최선 방법을 s3의 N0 숫자와 직접 빼지 않는다.

장면별 변화는 다음처럼 분해한다. 0은 해당 비교의 baseline이다.

$$\Delta Q_i=-(1-D_{s,i}^0)\Delta D_{\lambda,i}-(1-D_{\lambda,i}^0)\Delta D_{s,i}+\Delta D_{\lambda,i}\Delta D_{s,i}.$$

두 서버의 같은 seed1234 run은 서버 간 재현 진단이다. 이를 Student 독립 seed 2개로 세지 않는다. 장면 20개는 모델 20회 재학습도 아니다.

#### 5.4 실험 완료와 후보 채택을 분리

**이번 목록의 첫 1-seed 탐색은 원칙적으로 모두 실행한다.** 낮은 단독 R3/GV-AD HQNR을 이유로 조합을 자동 취소하지 않는다. 판정은 그 다음 추가 반복과 최종 모델 선택에 적용한다.

- `EXPLORATORY_POSITIVE`: 직접 대조 대비 common-grid HQNR이 양수. 단일 seed 후보일 뿐이다.
- `CONFIRMATION_READY`: 구현·산출물 검증 완료, 동일 조건 seed 대조가 준비됨.
- `REPLICATED_CANDIDATE`: 사전 지정한 3 Student seed의 평균 ΔHQNR이 양수이고 최소 2개 seed에서 양수. 양쪽 서버 block을 함께 주장하려면 양쪽에서 이 조건을 따로 확인한다. **이는 이번 계획의 운영상 기준이지 통계적 유의성 증명은 아니다.**
- `INCONCLUSIVE`: 부호 반전, 극소 차이, artifact 또는 큰 protocol 차이. 과거 W96 sd로 등가라고 단정하지 않는다.
- `NEGATIVE_AT_TESTED_SETTING`: 현재 계수·Teacher·horizon의 첫 탐색에서 음수. 정의/결과를 보존한다. KD 전체 불가능으로 확대하지 않는다.

HQNR 개선의 실용적 결론과 “Teacher 실패 영역을 더 fitting했다”는 기전 주장은 별개다. 후자는 공통 error-bin·통계/edge 오차와 gradient 진단이 있어야 한다. 결과를 억지로 overfitting 성공이나 regularization 성공 중 하나로 고정하지 않는다.

### 6. Seed·Teacher·초기값·비용 계약

#### 6.1 사전 지정 seed 반복

양쪽 서버에서 핵심 집합을 다음과 같이 고정한다.

`Q00, Q01, Q02, Q03, Q04, Q05, Q06, Q09, Q11, Q12`

기존 seed1234 결과를 보존하고 **777, 2026**을 추가한다. seed2026은 Teacher seed2025와 동일 초기 궤적이 되는 문제를 피하려는 이번 계획의 새 선택이다. 새 seed777 반복은 P2 다음, seed2026 반복은 P4 다음에 배치한다. 각 seed에서 반드시 Q00부터 실행하고 같은 init 및 데이터 순서를 공유한다.

기존 생성기의 반복 대상 Q00/Q04/Q10, seed2025·777도 없애지 않는다. 위 core와 겹치지 않는 **Q10-S777, Q00/Q04/Q10-S2025**를 말미의 `LEGACY_SEED_CHECK`로 남긴다. seed2025의 N0가 Teacher 학습을 그대로 재현하면 그 사실을 기록하며 **독립 Teacher–Student 관계의 추가 증거로 세지 않는다.** 이 함정은 생성기 자체도 경고한다. [S5]

초기화 hash·optimizer/data seed를 확인하고 seed 숫자만으로 독립성을 주장하지 않는다. 최신 결과를 보고 새로 선택한 X/표현/gate winner는 최종적으로 그 직접 대조까지 함께 777/2026에서 확인한다. 이 후속 승자 확인은 정의 catalog와 분리된 제한적 confirmation block이다.

#### 6.2 Teacher·pilot 보존

각 서버의 기존 Teacher를 유지한다. 아래 두 identity를 분리한다.

- 비교 baseline: `Q00`의 **해당 Student seed** run.
- λ pilot: 기존 `Q00_S1234_v1/last`의 **고정 hash**.

`--version v2` 때문에 Teacher 경로가 존재하지 않는 `T00...v2`로 바뀌거나, `--seed 777` 때문에 calibration pilot이 자동 교체되지 않도록 generator를 수정/검증한다. Teacher 준비 비용은 이미 지출했다고 숨기지 않고 누적 cost manifest에 남긴다. N0의 Teacher는 **평가 bin 전용**이며 학습 loss에 섞지 않는다.

두 서버의 R1/R2 부호 반전이 반복되면, 별도 Teacher-swap 확인 block을 수행할 수 있다. 이는 같은 크기를 유지한 채 참조 신호를 바꾸는 **후속 조건부 설계**다. 원래 local-Teacher 전수 결과를 덮어쓰지 않으며, swap 전에 송신·수신 SHA와 τ 재calibration 정책을 고정한다. 본 문서의 고정 107개 정의에 swap을 암묵적으로 포함하지 않는다.

#### 6.3 추가 학습 경로와 horizon

| 경로 | 초기값 | optimizer·schedule | 비교 |
|---|---|---|---|
| 일반 50K | seed별 저장된 독립 init | 기존 AdamW 1e-4, wd0.01, cosine, warmup100 | 같은 50K끼리 |
| TCOPY/PX | 동일 T00 best_hqnr의 가중치만 복사 | fresh AdamW, 기존 50K recipe | TCOPYN0와 같은 시작에서 비교 |
| CONT/CX | 동일 Q00/last의 가중치만 복사 | fresh optimizer, 같은 25K schedule | CONTN0와 비교 |
| LONG/LX | 같은 독립 init | 처음부터 100K horizon | LONG2NN0와 비교 |
| COSTMATCH | 같은 독립 init 또는 사전 선언한 공통 parent | 참조 KD의 실제 비용에서 N을 사전 결정 | update 대응 결과와 별도 보고 |

기존 생성기는 CONT를 25K로 줄이되 LR template을 그대로 사용한다. **현재 파일의 CONT는 “아주 낮은 LR 미세조정”으로 입증된 설정이 아니라, 공통 parent에서 fresh optimizer로 다시 학습하는 restart/tail**이다. 이번 legacy pass는 그 정의를 보존한다. 낮은 tail LR(예: 1e-5)을 검토하려면 별도 버전과 GT-only pair를 만든다. [S5]

Parent의 실제 update는 parent checkpoint metadata에서 읽는다. 생성기 phase의 기본 `parent_step: 0`을 실제 parent 학습량으로 기록하지 않는다.

100K의 중간 50K는 50K cosine run의 끝과 LR이 다르다. 같은 prefix라고 합치지 않는다. Teacher-copy에서 초기 soft=0인 것은 정상이다. optimizer를 유지하는 run과 재시작하는 run을 같은 continuation 효과로 비교하지 않는다.

#### 6.4 COSTMATCH

이 정의는 생성기에 있으나 실제 N이 미정이다. s2에서 Q09/Q12 등 재검증된 후보 중 기준 대상 하나를 사전 선택하여 `reference_run_key`로 고정한다.

1. 평가·진단을 포함한 Student 온라인 wall time과 학습-only GPU time을 모두 측정한다.
2. 동일 서버 N0 throughput으로 같은 budget의 N을 사전 산정한다. 평가 빈도와 평가 비용을 일관되게 반영한다.
3. budget 정의·예측 N·스케줄을 **시작 전에** 고정하고 최종 실제 비용 차이를 보고한다. 실제 일치가 안 되면 “정확한 시간 대응”이라고 쓰지 않는다.
4. Teacher 사전학습·calibration을 포함한 총비용과, Teacher를 공유했을 때의 상각 비용을 별도로 보고한다.
5. 데이터가 없는데 `--cost-match-updates 50000`을 임의 기본값으로 넣지 않는다. `BLOCKED_COST_MEASUREMENT`는 수치 확보 후 해제한다.

### 7. 실행·로깅·안전 규칙

#### 7.1 P0에서 수집할 산출물

`run_inventory.csv`, `run_key.json`, 실제 resolved config, Teacher/init/calibration hash, `common_grid_manifest.json`, `hqnr_comparison.csv`, `hqnr_per_scene.csv`, `fitting_bins.csv`, `gradient_diagnostics.jsonl`, cost와 restart 기록을 확인한다.

- code에 진단 기능이 있다는 사실과 해당 run에 파일이 생성됐다는 사실을 구분한다.
- N0·모든 GT-only 표현에도 공통 Teacher를 평가에서만 사용해 같은 bin을 계산한다.
- 과거 완료 checkpoint로 재평가가 가능하면 학습을 다시 하지 않고 분석 파일을 추가한다.
- `hqnr_best_original`, `hqnr_best_common_grid`, `hqnr_plateau`, `hqnr_last`를 서로 다른 열로 저장한다.
- parser가 빈 문자열/NaN을 0점으로 채우지 않도록 한다. 누락은 누락으로 기록한다.

#### 7.2 세 종류의 신호 비율

$$r_{coef}=\frac{\langle w_K\rangle}{\langle w_H\rangle},\qquad r_{loss}=\frac{\langle w_K K\rangle}{\langle w_H H\rangle+\epsilon},\qquad r_{grad}=\frac{\|\nabla_\theta L_K\|_2}{\|\nabla_\theta L_H\|_2+\epsilon}.$$

REC hard/soft, STAT hard/soft를 분리하고 STAT에는 λV를 포함한 값도 남긴다. hard–soft gradient cosine, d/a/weight 분위수·표준편차, Teacher 우위 영역의 상대 마진을 같은 진단 batch에서 기록한다. 진단 단계의 추가 autograd는 optimizer step·scheduler step·데이터 순서·전역 RNG를 바꾸지 않아야 한다. 기존 진단기가 처리한다면 재구현하지 말고 산출물을 확인한다.

#### 7.3 기술적 gate와 완료 정의

- `TECH_READY`: no-align byte-identity, shape·GT anchor·Teacher freeze·gate detach·Student gradient·통계 FP32·factor별 config diff가 통과.
- `RUNNING`: local 상태 확인. 디렉터리 존재만으로 완료 또는 실행 중이라고 확정하지 않는다.
- `TRAINED_N`: 실제 optimizer update가 목표 N에 도달했고 정상 checkpoint가 있음.
- `EVALUATED`: 주 HQNR 평가와 메타데이터가 연결됨.
- `ANALYZED`: 직접 대조·공통 격자·장면별 결과·주요 로그가 함께 정리됨.
- `BLOCKED_TECHNICAL`: config 미지원, SPD가 필요 없는 곳의 잘못된 연산, FD 불일치, OOM, NaN, 데이터·Teacher 불일치 등. 성능 실패와 별도로 기록.

C의 FD 오류가 발생해도 R/STAT queue 전체를 멈추지 않는다. C만 고친다. 일반 loss case는 기술적으로 정상일 때 원래 N까지 실행하고, 초반 낮은 HQNR로 자동 중단하지 않는다.

#### 7.4 OOM 및 재개

Teacher는 순차 no-grad, 통계는 row/pair별 계산, C는 네 probe를 순차로 계산한다. 배치48을 못 받는 새 case에서는 먼저 불필요한 graph·activation·동시 probe를 줄인다. microbatch/accumulation·precision을 바꾸면 직접 대조에도 같은 방식이 필요한지 확인하고 새 runtime block을 기록한다. 서로 다른 update/sample budget을 같은 50K로 부르지 않는다.

재개 시 데이터 순서까지 exact restore인지 확인한다. 지원되지 않으면 `resumed_nonexact`로 표시한다. 같은 이름에 새 학습을 덮어쓰지 않는다. 수행 중인 shell queue 파일을 바꾸지 말고 run 경계에서 새 pending manifest를 적용한다.

### 8. 실제 전달 파일과 적용 방법

- `01_S2_FINAL_EXPERIMENT_PLAN.md`: s2 담당 전체 순서·직접 대조·기존 보류 34개 전부.
- `02_S3_FINAL_EXPERIMENT_PLAN.md`: s3 담당 전체 순서·직접 대조·기존 보류 39개 전부.
- `NA104_FINAL_REGISTRY.json`: 기존89+신규18 정의, 출처 queue snapshot, 결과 snapshot, 전수 coverage.
- `NA104_S2_FINAL_STAGE_PLAN.json`, `NA104_S3_FINAL_STAGE_PLAN.json`: server/seed별 실행 **계획**. `execute:false`. trainer에 그대로 넣는 config가 아니다.
- `check_plan_coverage.py`: 이 묶음의 정의·배정·중복 누락을 검사한다. 학습·네트워크·시트 쓰기를 하지 않는다.

기존 generator의 `--repeat`는 Q00/Q04/Q10만 확장한다. 이번 core10을 자동으로 다 생성한다고 가정하지 않는다. 기존 generator는 queue/config를 덮어쓸 수 있으므로 별도 output staging에서 새 registry를 생성하고 diff 검증 후 적용한다. **본 JSON의 새 case를 기존 `--only`로 요청하면 조용히 빠질 수 있다. 새 ID가 실제 생성·resolve됐는지 개수를 검사한다.**

### 9. 출처와 해석 지위

[S0] 최신 사용자 결정: W104·D122 no-align, 동일 용량 KD의 fitting 강화, HQNR 중심, 시간 상한 없음, 보류 case까지 포함한 서버별 최종 계획.

[S1] 첨부 `2026-09-11_kd-fitting-signals-review.md` §서두; `2026-09-11_kdv-s2-w112-results.md` 개정본 §3–5; `2026-09-11_kd-integrated-analysis.md` §3–5. 과거 W112/NA104 초기 관측과 수식을 구분했다. 첨부 초판의 해석을 현재 사실로 대체하지 않는다.

[S2] live `pan-cvpr27`, spreadsheet ID `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`; `WV3-s2`, `WV3-s3(5090)`, `A1:W160`의 `NA104_` 행, 2026-09-12 조회. 열 L/M/N은 Dλ/Ds/HQNR. 실제 server 상태를 대신하지 않는다.

[S3] `hojunking/PAN-Crafter-repro` main, `config/queues/na104_s2.txt`, blob `8ce2e5e107f2df0aad8ef08701dc7bcf3cae2384`; `na104_s3.txt`, blob `eb0c7273f34636a9fe412cd27bb39b7c22783e2b`.

[S4] 같은 repo main, `na104_s2_deferred.txt`, blob `261b26af42c1f72d03453f167cf6bb088411a38e`; `na104_s3_deferred.txt`, blob `321176bde4330821e8b88a1d618ef60aec6ab156`. 2026-09-12 조회.

[S5] 같은 repo main, `tools/gen_na104_configs.py`, blob `4cbc525ce2dc590327f07fd1441009c2ccec3217`. case 정의·seed·horizon·COSTMATCH 생략 규칙·생성 방식 확인. config/함수의 존재와 s2/s3 실학습 검증 완료를 구분한다.

[S6] `PAN_NA104_S2_S3_HQNR_Experiment_Amendment_2026-09-11.md`, 이전 전달안. 주 metric·기존 결과 보존 규칙은 유지하되, 이번 사용자의 전수 포함 결정으로 **기존 첫 탐색의 performance gate는 단계적 편성으로 대체**한다.

[S7] `PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md` §§5–9·13; 기존 통계·TRI·C-NASENS·장기학습 규약. 실제 생성기와 다른 당시 selector/미확정 budget은 최신 사용자 지시·조회 config를 우선한다고 명시했다.

**설계자가 추가한 부분:** X/PX/CX/LX 18개, seed2026 확인 block, 모든 보류 첫 탐색의 유한 편성, 후보 운영 판정 기준. 이들은 기존 문서에서 실험 완료된 사실이 아니라 이번 결과를 바탕으로 제안한 실행 설계이다.
