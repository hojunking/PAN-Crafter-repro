# Stage 1 RGB / 전체 밴드 평균 정합 참조 비교

2026-10-26. 기존 공식 평가 JSON과 구현을 조사했다. 새로운 학습·GPU 평가·Sheet 게시 없이 현재 실행 큐를 유지했다. 원본 경로·평가 SHA·전체 35-step 곡선·장면별 차이는 [분석 JSON](assets/2026-10-26_stage1-mean8-target-audit.json)에 있다.

## 조사 당시 구현 — SENSOR5 v1 및 기존 WV3

| 항목 | WV3 | QB / GF2 |
|---|---|---|
| Aligner 입력 MS | native LMS 8채널 | native LMS 4채널 |
| Stage 1 absolute alignment loss의 참조 | GT 8밴드 평균 → Scharr | GT 4밴드 평균 → Scharr |
| Stage 2 q/g 참조 | GT RGB [4,2,1] 평균 → Scharr | GT RGB [2,1,0] 평균 → Scharr |
| Teacher / Student reconstruction error | 8밴드 전체 | 4밴드 전체 |

인덱스는 0-based다. Aligner는 PAN 1채널과 전체 LMS를 별도 CNN stem으로 처리한다. 입력을 먼저 RGB 또는 한 채널 평균으로 줄이지 않는다. 입력 LMS는 제공된 PAN 격자의 native LMS이며 재구성기의 interp23 `I_lrms`와 구분한다.

Stage 1의 대응 항은 Stage 2의 q가 아니라 `L_abs`다. mean-C GT로 만든 한 채널 영상에 Scharr를 적용한 다음 corrected PAN의 Scharr 절대응답과 비교한다. 밴드별 Scharr를 각각 구해 평균하는 방식이 아니다. 이 loss는 Aligner에 gradient를 준다. Stage 2 q는 RGB 평균의 구조 차이이며 frozen reference로 Hard 가중치에만 사용된다. q 자체가 Student 출력에 대한 별도 luminance reconstruction loss는 아니다.

근거: `directshift48/method.py::DirectPANAligner`, `panda_band30/aligner.py::AllBandScharr`, `panda_sensor5/aligner.py::AllBandScharr`, `panda_stage2_condq/losses.py::reference_maps`.

## 비교 조건

- RGB/BIC: donor24 s1 `A1`, seed 261016101.
- mean8/BIC: band30 s1 `G8_S1_BIC`, 같은 seed.
- mean8/interp23: band30 s1 `G8_S1_U23`, 같은 seed.
- 공통: WV3, W112/D123, global2 Aligner, 50K warmup-cosine, LR A=1e-5/T=1e-4, λ_abs=2, λ_ε=.003, batch48/micro12, MASK2, PAN warp bicubic.
- baseline JSON과 공통 데이터·증강·ε seed가 동일하다. 초기 Aligner 24개 parameter tensor와 Teacher 90개 parameter tensor가 bitwise 동일함을 CPU에서 확인했다. source registry SHA와 공식 evaluator protocol도 동일하다.
- 별도 band augmentation RNG는 mean8 구현의 예약 상태이며 이 global2 조건에서 auxiliary policy는 NONE이다.
- **BIC/interp23는 MS 재구성 base의 보간 방식**이다. 여기서는 세 조건 모두 PAN warp는 bicubic이다.

## 공식 Stage 1 Teacher 결과

| 설정 / 50K | HQNR | ERGAS | D_s | D_lambda |
|---|---:|---:|---:|---:|
| RGB 평균 + bicubic MS base | .915865 | 2.073584 | .068681 | .016715 |
| 8밴드 평균 + bicubic MS base | .917363 | 2.064035 | .068308 | .015488 |
| 8밴드 평균 + interp23 MS base | .920739 | 2.040115 | .068904 | .011201 |

RGB→mean8의 50K 관측 차이는 HQNR **+.001498**, ERGAS **−.009549**(약 −.46%)다. RR20 중 14장 ERGAS 개선, FR20 중 14장 HQNR 개선이다. 각 세트 20장의 변화를 새 seed 반복 검증으로 취급하지 않는다.

| 같은 step의 mean8/BIC − RGB/BIC | ΔHQNR | ΔERGAS |
|---:|---:|---:|
| 7K | −.007439 | −.031748 |
| 20K | +.006290 | −.027848 |
| 30K | +.003182 | −.021417 |
| 40K | +.001253 | −.012124 |
| 50K | +.001498 | −.009549 |

초반 HQNR 최대는 RGB **.960203 @7K**(ERGAS 2.376434), mean8/BIC **.956295 @9K**(ERGAS 2.327291)다. 따라서 전 구간의 HQNR 우위는 아니다. 같은 44K에서 최소 ERGAS는 RGB 2.071371, mean8/BIC 2.061201이다.

mean8을 유지한 채 MS base를 bicubic→interp23로 바꾼 50K 차이는 HQNR **+.003376**, ERGAS **−.023920**다. RGB/BIC→mean8/interp23의 총 차이를 전부 8밴드 평균 효과라고 귀속할 수 없다. RGB/interp23 조건까지 포함한 완전한 2×2 상호작용 검증은 이 비교에 없다.

## 정합량 변화와 해석

50K FR20의 축별 shift 중앙값은 RGB/BIC `(1.1842, −1.3785)`, mean8/BIC `(1.0827, −1.1147)`이다. 장면별 두 shift 사이 거리의 중앙값은 **.3463 FR PAN pixel**이다(RR은 .1346 pixel). 채널 평균의 정의를 바꾸면 실제 Aligner가 택하는 정합점도 달라진다. 이것을 물리적 정합 정확도의 개선으로 단정할 displacement GT는 없다.

관측상 mean8은 후반 Teacher 복원에 소폭 유리했다. 다만 단일 seed, 과거 Stage 1 구현 비교이며 현재 L4·DIFF·UBIL Student의 최종 성능 향상을 입증하는 대조는 아니다. 특히 G8_PAN_BIC Stage 2는 band30 완료 목록에 없어 직접적인 Stage 2 target-only 비교가 없다. 기존 MS base 변경·donor 선정·Student 구조 효과가 함께 섞인 결과로 mean8의 독립 기여를 크게 주장하지 않는다.

전체 밴드 평균은 구조 정합용 대리 영상이다. RGB 단순 평균 역시 표준 휘도 가중식 또는 센서 응답으로 보정된 PAN 합성과 동일하지 않다. WorldView-3의 PAN 명목 대역은 450–800nm이며 NIR2는 857–1039nm다. 따라서 mean8을 물리적 PAN luminance라고 부를 근거도 없고, PAN 정합에는 RGB만이 유일한 정답이라고 단정할 근거도 없다. [제조사 WorldView-3 자료](https://www.edpmedia.com/assets/maxar_brochure/WorldView3.pdf).

## 후속 방향

향후 새 캠페인의 warp는 사용자 결정대로 Stage 1·Stage 2에 Lanczos-4를 통일한다. 구조 참조를 RGB로 통일할 경우 Aligner 입력의 MS 전체 채널은 그대로 유지할 수 있다. 이는 입력 채널 변경과 별개의 loss-reference 변경이다. 현재 성능에서 이를 판단하려면 동일 L4/interp23 조건의 Stage 1 RGB 대 mean-C와 각각의 동일 BASE Student를 비교하는 것이 직접적이다. 이 비교 감사 자체는 실행 큐를 변경하지 않았다. 이후 사용자 결정으로 QB/GF2는 RGB/Lanczos-4 v2로 전환했다. 최신 구현·전환 상태는 [QB 감사 및 v2 적용 기록](2026-10-26_qb-low-hqnr-audit.md)과 [인계문](../research_log/PANDA_Sensor5_Stage12_Handoff_2026-10-26.md)을 따른다. 위 구현 표는 전환 전 v1을 설명한다.
