# s3 — 앞으로 20시간 후보 확인 우선순위
작성일: 2026-09-13
상위 규약: `00_20H_CRITERIA_AND_RETIREMENT.md`
이 문서는 적용안이다. 실제 서버 큐/프로세스는 변경하지 않았다.

## 1. 후보를 세 이름으로 제한
1. **Q36 / N0+GC-H**: GT variance·covariance 중심 주력.
2. **Q12 / R3+EDGE-H**: 반복 개선이 남은 성능 경쟁 후보.
3. **CF01 / N0+GC-FIX**: Q36이 재현될 때만 확인할 유일한 통계 KD 확장.
Q00는 baseline, X02는 Q12의 soft 제거 대조다. 이 둘은 새로운 성능 family가 아니다.
독립 R1·GV-FIX·A/B/C·IV/SC·광범위 gate·장기학습 탐색은 이번 개발 목록에서 종료한다.
완료/negative 결과는 보존하고 미실험 family를 실패로 쓰지 않는다.

## 2. P0: 0–2시간 목표
- 현재 run은 원 설정으로 마무리하고 해당 남은 시간을 예산에 반영한다.
- 로컬 완료/진행중 작업을 확인해 아래 whitelist와 중복되는 것은 차감한다.
- Q00/Q36/Q12의 common-grid best HQNR와 plateau/last, 같은 checkpoint의 D_lambda/D_s를 확보한다.
- Teacher/init/pilot/data/evaluator/config hash를 고정한다.
- 평가 간격이 다르면 공통 시점에서 재선택한다. 원래 best는 보존한다.
- 잘못 남은 best_rr_val 설명을 실제 HQNR 기준과 구분한다. 선택 기준을 ERGAS로 변경하지 않는다.

## 3. P1: 필수 4개
| 순서 | case | Student seed | budget | 계획 run ID |
|---:|---|---:|---:|---|
| 1 | Q36 | 777 | 50K | `NA104_Q36_W104_D122_WV3_N0_GCH_S777_v2` |
| 2 | Q00 | 2026 | 50K | `NA104_Q00_W104_D122_WV3_N0_OFF_S2026_v2` |
| 3 | Q36 | 2026 | 50K | `NA104_Q36_W104_D122_WV3_N0_GCH_S2026_v2` |
| 4 | Q12 | 2026 | 50K | `NA104_Q12_W104_D122_WV3_R3_EDGEH_S2026_v2` |

이 ID의 실제 config/실행 존재 여부는 운영자가 확인한다. 생성 완료를 주장하는 표가 아니다.
현재 시트 Train(h): Q36 0.88h, Q12 0.94–0.95h, Q00 최근 0.81h.
필수 4개 학습의 계획량은 약 3.5–3.7h다. export/추가 평가·준비시간은 별도로 남긴다.
새 seed도 같은 방법이다. Student baseline만 해당 seed로 연결하고 Teacher와 lambda pilot은 기존 v1 hash를 유지한다.
이 20시간에는 계수·window·LR을 바꾸지 않는다.

## 4. P2: CF01 조건부 검증
본 서버 Q36이 세 seed 기준을 통과하면 CF01-S777과 CF01-S1234를 50K로 실행한다.
두 pilot 각각 Q36 대비 full-precision common-grid Delta HQNR>0이면 CF01-S2026을 실행하고 s2에 교차 확인을 요청한다.
한 pilot이 비양성이면 이 분기를 종료한다. 반복을 유리한 seed로 교체하거나 beta를 곧바로 바꾸지 않는다.
CF01 config는 Q36의 Teacher eval_only 설정을 그대로 복사하지 않는다. 학습에 Teacher covariance target이 필요하다.

CF01 수식:
`GT_L1 + lambda_C * (mean_abs(C_S-C_GT) + 0.1 * mean_abs(C_S-C_T))`.
rec=N0, stat=GC/FIX, window5. Student 통계만 gradient를 유지한다.
Q36의 lambda_C를 재사용하고 새로운 hard 재가중·adaptive gate를 넣지 않는다.
현재 CF01은 제안 ID이며 결과/실행시간 실측이 없다.
최대 3개 학습의 계획량은 약 3.0–3.6h다.
통과하지 않은 분기를 나중에 자동 재개할 보류 큐로 옮기지 않는다.

## 5. P3: Q12가 남을 때만 X02 보강
X02(R1+EDGE-H)-S777/S2026 각 50K를 실행한다.
Q12와 동일 Teacher/lambda_E/seed/update/eval 기회를 유지한다.
예상 학습량은 약 1.9h이며, Q12가 탈락하면 실행하지 않는다.
X02와 차이가 없으면 Q12 family에서 직접 output soft가 필수라는 주장을 버린다.
구조 감독과 Teacher-error hard는 남길 수 있지만 이것은 새 독립 R1 스윕을 시작한다는 뜻이 아니다.

## 6. 18–20시간 판정
공통 3 Student seed 1234·777·2026에서 이 서버의 N0 대비 평균 Delta HQNR>0, 최소 2/3 양수를 확인한다.
CF01은 N0뿐 아니라 Q36 대비 같은 기준도 확인한다.
양 서버 일반성을 주장하려면 두 서버 각각 통과해야 한다.
plateau/last는 안정성 설명이고 ERGAS는 보조다. 임의로 판정축을 바꾸지 않는다.

모든 조건부가 열려도 새 학습은 최대 9개다. 조건부가 닫히면 조기에 끝내도 된다.
완료 시간을 보장하는 schedule이 아니다. 실측 처리량으로 남은 시간을 갱신한다.
시간이 부족하면 P3, 다음으로 CF01 교차 확인을 이월하고 P1 full budget을 우선한다.
남은 20h를 채우려고 폐기한 방법을 다시 실행하지 않는다.

## 7. 폐기와 미세조정 준비
- 실제 pending whitelist는 P1과 승인된 P2/P3뿐이다.
- 이전 CX02/CONTN0·TCOPY·LONG 목록도 이번에는 실행하지 않는다.
- 후보가 남으면 이후 Q36 lambda_C×{0.5,1,2}, Q12 lambda_E×{0.5,1,2}부터 한 축씩 조정한다.
- CF01이 Q36을 반복해서 넘을 때만 beta_C 국소 조정을 연다.
- 같은 parent 추가 fine-tuning은 동일 LR/optimizer/추가 update 대조로 별도 기록한다.
- 미래 tuning 결과와 이번 baseline-coefficient 확인을 하나의 동일 방법 seed 평균에 섞지 않는다.

회신: 현재 run, 차감한 완료분, 적용 queue hash, core 결과, CF01/X02 진입 여부,
폐기 목록, Teacher/pilot/config hash, 후보별 다음 tuning 축.
