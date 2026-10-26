# QB SENSOR5 저조한 HQNR 원인 감사

2026-10-26. 대상은 기존 `PANDA_S13_SENSOR5_STAGE12_20261026_v1`의 QB Stage 1 및 `QB_BASE` Student다. 이 문서의 추가 측정은 **학습 없는 진단**이며 공식 결과, 원본 평가 참조, checkpoint, Sheet를 변경하지 않는다.

## 1. 결론

**저조한 FR HQNR은 실제이며, 가장 강하게 확인된 원인은 큰 PAN 보정과 원본 PAN 기준 공간 지표 사이의 불일치다. 현재 증거는 Aligner가 FR에서 엉뚱한 방향으로 이동한다는 설명을 지지하지 않는다.**

- 동일 Student 12K에서 적용 shift만 0으로 바꾸면 FR HQNR이 **.795451 → .921862**로 회복된다. 반대로 RR ERGAS는 **3.566785 → 3.929512**로 악화한다. PAN 보정은 RR 복원에 도움이 되지만 현 FR 공식 지표에는 큰 비용을 유발한다.
- 독립적인 **입력만 사용한** FR PAN↔RGB-LMS 정합 추정에서는 원래 오프셋의 중앙값이 **2.6568 px**, Aligner 보정 뒤 잔차는 **.1818 px**다. 20장 모두 개선된다. 큰 보정량 자체를 오작동이라고 판단하면 안 된다.
- 같은 native prediction을 그대로 두고 PAN 평가 참조만 동일하게 보정하는 별도 진단에서는 Student HQNR이 **.795451 → .878966**, Teacher는 **.732061 → .903711**로 오른다. 좌표 기준의 영향은 크지만 전체 성능 저하를 전부 설명하지는 않는다.
- 현재 확인한 범위에서 DN 정규화, 채널 수, LRMS 생성, interp23 증강 위상, 평가 수치 재현의 명백한 구현 오류는 발견되지 않았다. **이것은 모든 버그의 부재 증명이 아니다.**
- RGB/Lanczos-4 통일은 정의와 구현의 일관성을 개선하지만, 이 QB 현상을 해결한다고 보장할 근거는 없다. RGB와 mean4의 독립 정합 추정은 이번 자료에서 거의 같다.

## 2. 실제 학습 궤적

Stage 1은 50K까지 완료됐고 RR validation RRE 기준으로 38K가 donor로 선택됐다. Student는 마지막 공식 평가가 12K다. FR shift 크기는 장면별 벡터 norm의 중앙값이다.

| 모델 / step | 공식 FR HQNR | 공식 RR ERGAS | FR D_s | FR shift 중앙값, px |
|---|---:|---:|---:|---:|
| Teacher 5K | .928827 | 3.772555 | .03346 | .3489 |
| Teacher 10K | .791691 | 3.693427 | .18004 | 1.1289 |
| Teacher 20K | .743663 | 3.585640 | .23718 | 2.1849 |
| Teacher 38K, 선택 donor | .732062 | 3.514094 | .24800 | 2.4839 |
| Teacher 50K | .732632 | 3.509882 | .24718 | 2.4933 |
| Student 9K, 관측 최고 HQNR | .800997 | 3.597361 | .17071 | 2.4839, 고정 |
| Student 12K | .795451 | 3.566785 | .17377 | 2.4839, 고정 |

38K donor의 shift 축별 중앙값은 RR **(.1651, −.3763)**, FR **(1.7664, −1.7521)**이다. RR norm 중앙값은 .4326 px다. RR/FR은 같은 장면을 정확히 짝지은 자료가 아니므로 이 비율을 그대로 해상도 배율이나 물리적인 변위 비율로 해석하지 않는다.

Stage 1에서 이미 HQNR 하락이 발생했다. 따라서 Student의 q 보정값이나 Hard 계수만을 최초 원인으로 지목할 수 없다. D_lambda는 Teacher 5K .03913에서 38K .02642로 개선되지만 D_s 악화가 훨씬 크다.

원본: `work_dir/panda_sensor5/PANDA_S13_SENSOR5_STAGE12_20261026_v1/s1/QB/runs/` 아래 `PSN5_QB_S1_G4_MATURE_U23_NOSOFT_R1_S261016101` 및 `PSN5_QB_S1_QB_BASE_NOSOFT_R1_S261018101`의 `attempt_01/evaluation_*.json`.

## 3. 동일 checkpoint의 shift 제거 진단

보존한 v1 코드로 동일 A/T/S checkpoint를 다시 읽었다. Teacher는 기존 bicubic, Student는 기존 Lanczos-4를 유지했다. 공식 평가 함수와 RR20/FR20 원본 참조를 그대로 사용했다. zero 조건은 **그 자리에서 적용 shift만 0으로 설정**한 것으로, zero 조건으로 학습된 모델이 아니다.

| 모델 | 적용 shift | HQNR | D_s | D_lambda | ERGAS |
|---|---|---:|---:|---:|---:|
| Teacher 38K | 원래 예측 | .732061 | .248005 | .026423 | 3.514094 |
| Teacher 38K | 진단용 0 | .882871 | .084599 | .035269 | 3.676240 |
| Student 12K | 원래 예측 | .795451 | .173769 | .036911 | 3.566785 |
| Student 12K | 진단용 0 | .921862 | .030464 | .049343 | 3.929512 |

Student의 FR HQNR은 20장 모두 개선(+.04387∼+.16071), RR ERGAS는 20장 모두 악화했다. Teacher도 FR 20장 모두 HQNR이 개선됐고 RR은 19/20장에서 악화했다.

원래 적용 shift 조건 재평가가 공식 기록과 일치한다: RR ERGAS 차이 0, FR HQNR 차이 최대 **4.1e−8**. 새 평가 코드의 스케일 또는 지표 함수 혼용 때문에 .79가 나온다는 설명은 이 재현 결과와 맞지 않는다.

**zero 조건은 원인 분리를 위한 진단이다. 최종 방법이나 목표 달성 결과가 아니며, scale 1로 Aligner를 활용한다는 요구는 그대로 유지한다.**

근거: [진단 JSON](assets/2026-10-26_qb-shift-counterfactual.json), [재현 스크립트](assets/2026-10-26_qb-shift-counterfactual.py).

## 4. 큰 FR shift가 잘못된 정합인가?

기존 `ta2.registration.register_band`의 독립 gradient-NCC 추정으로 원본 PAN과 보정 PAN을 각각 비교했다. FR은 **GT 없이 native LMS**를 참조했고, RR은 **진단용 GT oracle**만 사용했다. RGB 단순 평균과 mean4를 둘 다 확인했다.

| 자료 / 참조 | 보정 전 잔차 norm 중앙값 | 보정 후 | 개선 장면 |
|---|---:|---:|---:|
| FR / RGB native LMS | 2.6568 px | .1818 px | 20/20 |
| FR / mean4 native LMS | 2.6543 px | .1718 px | 20/20 |
| RR / RGB GT, oracle 진단 | .5049 px | .0900 px | 20/20 |
| RR / mean4 GT, oracle 진단 | .4987 px | .0798 px | 20/20 |

FR RGB 참조에서 독립 추정한 보정 전 shift 축별 중앙값은 **(1.9817, −1.7748)**이다. Aligner의 **(1.7664, −1.7521)**과 방향 및 크기가 비슷하다. 따라서 현재 데이터상 큰 FR 출력은 기존 PAN–LMS 오프셋에 반응하는 것으로 해석하는 편이 타당하다.

설정은 FR PAN sigma1.98/LMS sigma.7, RR PAN·GT sigma.8, gradient, bound8, margin16이다. 이것은 **영상 기반 정합 proxy이며 물리적인 변위 정답은 아니다.** 서로 다른 분광 응답·평활화·LMS 보간의 영향을 포함한다. FR HRMS GT가 없으므로 보정 뒤 실제 HRMS 기하가 완전히 옳다고 증명하지 못한다. 다만 RGB/mean4 및 20장 전부의 일관된 개선은 무시할 수 없는 증거다.

근거: [정합 proxy JSON](assets/2026-10-26_qb-registration-proxy.json), [스크립트](assets/2026-10-26_qb-registration-proxy.py).

## 5. PAN 평가 참조 좌표를 바꾼 진단

**공식 평가에 사용할 수 없는 진단**이다. 같은 prediction을 고정하고, HQNR 계산에 넣는 PAN 참조만 해당 모델의 원래 donor shift와 warp로 보정했다. LMS/MS 참조는 그대로다. Teacher에는 bicubic, Student에는 Lanczos-4를 사용했다.

| 동일 native prediction | 원본 PAN 참조 HQNR | 보정 PAN 참조 HQNR | 원본 D_s → 보정 참조 D_s |
|---|---:|---:|---:|
| Teacher 38K | .732061 | .903711 | .248005 → .071648 |
| Student 12K | .795451 | .878966 | .173769 → .087082 |

D_lambda는 그대로다. 반대로 zero-input Student prediction에 보정 PAN 참조를 적용하면 HQNR은 .921862에서 .703343으로 내려간다. 따라서 어떤 PAN 좌표를 기준으로 평가하는지가 결과에 큰 영향을 준다는 점은 직접 확인된다.

하지만 보정 참조에서도 Student가 목표 HQNR .9315에 이르는 것은 아니다. Student는 corrected PAN과 차영상을 함께 받아 원본 PAN 정보를 활용할 수 있고, RR의 작은 보정량에서 학습한 결합 방식이 FR의 큰 보정량에서 동일하게 작동한다고 보장되지 않는다. **출력의 일부 구조가 두 입력 좌표 사이에 남는 현상, 디테일 전달, 센서의 분광 차이 등이 추가 원인 후보이며 아직 분리 검증되지 않았다.**

이 수치는 `SHIFTED_PAN_REFERENCE_DIAGNOSTIC_INELIGIBLE`로 기록했다. 공식 기록·업로드·모델 선택·목표 달성 판정에 사용하지 않는다. PAN을 보간한 참조이므로 좌표 이동 외에 보간에 따른 변화도 포함한다.

근거: [보정 참조 진단 JSON](assets/2026-10-26_qb-shifted-reference-diagnostic.json). 재현은 위 counterfactual 스크립트의 `--splits fr --reference shifted` 옵션을 사용한다.

## 6. Aligner가 이동에 정상적으로 반응하는가?

같은 38K Aligner에 PAN만 ±1 px 축 방향으로 이동시켰다. 원래 v1의 bicubic injection을 사용했고, 기대 반응은 `A(W(P,eps),LMS)−A(P,LMS)=−eps`다. RR20/FR20 각각 80개 반응을 측정했다.

| 자료 | equivariance 오차 평균 | 반응 gain 평균, 이상값 1 |
|---|---:|---:|
| RR20 | .1195 px | 1.0084 |
| FR20 | .2391 px | .7860 |

FR 반응이 RR보다 약하고 부정확한 것은 사실이다. 하지만 head가 죽거나 부호가 반전됐거나 4배 단위를 잘못 적용한 현상은 아니다. native 절대 정합 proxy는 앞 절처럼 실제로 개선된다. epsilon 반응 오차와 native 절대 정합 정확도는 별개로 봐야 한다.

근거: [반응 JSON](assets/2026-10-26_qb-aligner-response.json), [스크립트](assets/2026-10-26_qb-aligner-response.py).

## 7. 데이터·입력·해상도 감사

별도 CPU 감사에서 확인했다.

- QB 정규화는 `2*DN/2047−1`이며 GF2의 1023과 섞이지 않는다. QB의 Aligner MS 입력, 복원 출력은 모두 4밴드다.
- QB 원본 train/val LMS와 repaired LRMS를 함께 사용하는 것은 기존 명시적 정책이다. 원본 LMS를 repaired MS의 interp23 결과로 바꾼 입력 진단에서 Aligner shift 차이 중앙값은 train **.0231 px**, val **.0521 px**다. 이 차이만으로 FR 약2.5 px 보정을 설명하기 어렵다.
- 공식 RR/FR stored LMS는 현재 interp23(MS)와 평균 절대차 약 **1.3e−5 DN**으로 일치한다.
- train40개 표본의 msfix 생성 재현은 MS 차이 0이었다. 16개 flip/rotation 조합의 interp23 위상 일관성 검사 최대 차이는 **.000488 DN**이었다. 전체 corpus를 새로 생성하거나 수정하지 않았다.
- FR 전체512 입력의 shift norm 중앙값2.484 px는 center64 crop에서2.073, 9개64 patch에서는2.174 px로 여전히 크다. 따라서 전체 영상의 크기/GAP만으로 생긴 현상이라고 볼 수 없다. 입력 축소 시 출력이 달라지는 것은 관측되지만, 원래 오프셋과 해상도를 같이 바꾸므로 그대로 대체 추론 정책을 정당화하지 않는다.

근거: [데이터·해상도 감사 JSON](assets/2026-10-26_qb-data-integrity-audit.json).

## 8. Donor 선택 및 향후 해석

현재 selector는 RR validation RRE50을 최소화한다. 5K **.25614**, 20K **.16134**, 38K **.13970**로 개선되므로 38K 선택은 선언된 규칙대로다. 하지만 **RR 정합 proxy 개선은 FR 공식 HQNR 보장이 아니다.** 선택 함수 구현 오류와 선택 목적의 한계를 구분해야 한다.

즉 현재 결과를 “더 학습하면 해결되는 미성숙 Student” 또는 “mean4/RGB 불일치 때문에 전부 망가진 run”으로 단정할 수 없다. 원래 PAN–MS 오프셋이 큰 FR에서, MS/GT 좌표로 정합해 학습하는 목적과 원본 PAN에 대한 공간 평가 사이의 관계를 먼저 이해해야 한다.

사용자 결정인 **신규 QB/GF2의 RGB 참조 및 Lanczos-4 통일, 전체 MS 입력·출력 유지, scale1 사용, beta0**는 별도로 적용한다. 이 통일을 위 문제의 검증된 해결책으로 주장하지 않는다. 기존 자료와 새 campaign을 분리하며 공식 평가에는 원본 PAN/LMS를 계속 사용한다.

후속 검증은 신규 통일 Stage 1에서도 RR validation 정합 개선과 native FR shift·공식 D_s가 어떻게 함께 움직이는지 확인하는 것이 우선이다. 이후 Student의 두 PAN 입력 경로가 큰 보정량에서 어떤 좌표의 구조를 출력에 전달하는지 진단할 가치가 있다. 공식 지표 참조 변경, shift 제거·임의 축소, test-GT 정합을 최종 방법으로 추가하는 것은 이번 감사의 결론이 아니다.

## 9. DIFF 입력의 RR 학습–FR 분포 차이

동일 38K Aligner와 기존 Student Lanczos-4로 `corrected PAN − original PAN`의 크기를 측정했다. train은 균등 간격으로 뽑은 **128개 미증강 patch 표본**, RR/FR은 각각 공식20장이다. 영상 크기 차이를 줄이기 위해 모두 중앙64 영역의 내부48×48에서 측정했다. Aligner는 각 자료의 원래 입력 크기를 받았다.

| 자료 | shift norm 중앙값 | 차영상 절대값 평균의 중앙값, DN | 모델 입력 단위 | 차영상 L1 / 같은 PAN 영역 std 중앙값 |
|---|---:|---:|---:|---:|
| RR train128 표본 | .4640 px | 9.3725 | .00916 | .1493 |
| RR test20 | .4326 px | 13.1365 | .01283 | .1377 |
| FR test20 | 2.4839 px | 54.3245 | .05308 | .4734 |

FR 차영상 크기는 train 표본보다 중앙값 기준 **DN에서5.80배, PAN 표준편차로 나눈 뒤에도3.17배**다. 관측한 FR shift 최소값2.091 px는 train 표본 최대값1.217 px보다 크다.

이는 “작은 PAN 차영상으로 학습한 Student가 FR에서 훨씬 큰 차영상을 받는다”는 설명을 뒷받침한다. **현재 분포 차이는 관측 사실이며, 이것이 성능 저하의 어느 비율을 유발하는지는 아직 검증되지 않았다.** train 전체·모든 augmentation을 조사한 결과가 아니고 장면의 구조·주파수 특성도 다르므로, 이 표만으로 학습 분포 밖이라는 엄밀한 판정이나 특정 구조 변경의 효과를 주장하지 않는다.

근거: [DIFF 분포 JSON](assets/2026-10-26_qb-diff-distribution.json), [CPU 재현 스크립트](assets/2026-10-26_qb-diff-distribution.py).


## 10. 구현 반영 및 검증

사용자 결정에 따라 별도 `PANDA_S13_SENSOR5_RGB_L4_STAGE12_20261026_v2`를 등록했다. Stage 1 정합 loss와 Stage 2 q/g의 참조를 RGB 평균으로, native correction·ε 주입·동결 Teacher·Student의 PAN warp를 Lanczos-4로 통일했다. 전체 4밴드 MS 입력/출력/복원 손실, scale1, DIFF+UBIL, β0 및 기존 다섯 Student 하이퍼파라미터 case는 유지한다. validation도 RGB 참조와 L4 support에 맞추고 calibration은 신규 donor로 다시 만든다.

기존 QB_BASE는 12,480 update에서 완전한 checkpoint를 남기고 안전 종료했다. v1 결과는 변경하지 않았다. s1 v2는 새 Stage 1부터 실행을 시작했다. 두 데이터셋의 source check, **57개 자동 검증**, 실제 QB/GF2 GPU에서 A/T 학습 및 완전한 checkpoint 복원을 통과했다. GPU 재개 후 다음 update의 최대 파라미터 차이는 QB 9.31e−10, GF2 1.86e−9로 설정된 FP32 오차 범위 안이다. [GPU 검증 기록](assets/2026-10-26_sensor5-rgb-l4-gpu-preflight.json).

Sheet의 기존 run별 대표 및 공동 목표 통과 step 게시 규칙을 보존한다. v2의 새 prefix로 원래 v1 행과 분리하며 RGB/L4 설정을 설명에 표시한다. Stage 1 숫자 행은 50K 종료 후 validation-selected donor가 확정될 때 게시된다. s3 실행은 [안전 전환 인계문](../research_log/PANDA_Sensor5_Stage12_Handoff_2026-10-26.md)을 따른다. 기존 WV3 TUNE30/HARD36은 이 새 센서 큐 뒤에 이어진다.

이 재시작은 **사용자가 승인한 참조·warp 통일의 적용**이며, HQNR 원인 해결이 입증되어서 수행하는 조치는 아니다. 공식 평가의 PAN/LMS 참조를 바꾸거나 shift를 제거하지 않는다.

재현 스크립트는 v1 모듈을 별도 경로에서 import해야 한다. 로컬 archive `work_dir/panda_sensor5_launchers/s1/sensor5_v1_code_before_rgb_l4.tar.gz`를 `/tmp/panda_sensor5_v1_qb_audit`에 풀면 기록된 스크립트 기본 경로와 맞는다. Git 기준 구현은 이전 commit `6a01bb8`의 `panda_sensor5`다. 실제 checkpoint·데이터·공식 evaluator는 로컬 등록본이 필요하며 Git에 포함하지 않는다.
