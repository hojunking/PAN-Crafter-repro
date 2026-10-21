# ZERO30: 성숙 Aligner · ZERO · 회전 일관성, 서버별 30시간

사용자 승인: 2026-10-21, “30시간 기준 실험 case를 짜서 실험 진행”.
이번 세션의 실행 대상은 **s1**. s3용 동일 명령도 제공하되 원격 시작은 하지 않는다.

Campaign: `PANDA_S13_ZERO_ROTCONS_HARDONLY_30H_20261021_v1`

## 질문과 우선순위

1. 성숙 all-band donor + Hard-only에서 Student 입력 PAN 이동을 완전히 없애면
   기존 LOW/DOWN보다 HQNR–ERGAS 공동 성능이 나아지는가?
2. 회전 두-view 계산량 자체와 구분하여, 일관성 loss가 단일 추론 모델의
   방향별 변동과 후기 D_s 상승을 줄이는가?

이번 탐색 목표는 사용자 최신 분석의 **HQNR ≥ .959, ERGAS ≤ 2.045**.
이는 통과 여부를 표시하는 기준이며 Stage2 checkpoint/Sheet 선정은 계속
**HQNR 최대 → ERGAS 최소 → 이른 step**이다. 이전 Mature30의 .960 기준과
과거 checkpoint 보관 기준은 소급 변경하지 않는다. Test-aware 탐색이며 독립
확증 실험 또는 통계적 유의성을 주장하지 않는다.

## 큐

| 우선순위 | Case | Seed | Student views | 일관성 λ | 업데이트 | 예상 시간 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | ZERO R1 | 261018101 | 1 | 0 | 50K | 약 3.6h |
| 2 | ZERO R2 | 261018102 | 1 | 0 | 50K | 약 3.6h |
| 3 | PAIR0 R1 | 261018101 | 2 | 0 | 50K | 약 5–6h |
| 4 | ROTCONS R1 | 261018101 | 2 | .05 | 50K | 약 5–6h |
| 5·예비 | PAIR0 R2 | 261018102 | 2 | 0 | 50K | 실측 |
| 6·예비 | ROTCONS R2 | 261018102 | 2 | .05 | 50K | 실측 |

- ZERO 두 seed를 먼저, 그 다음 R1 두-view 한 쌍을 진행한다.
- R2 한 쌍은 두 case 전체가 남은 예산에 들어올 때만 함께 허용한다.
  예산 부족 시 둘 다 skip한다. 실제 시간 초과에는 안전 checkpoint를 남긴다.
- 30시간은 cost 측정·준비·calibration·학습·평가를 포함한 지속 wall-clock.
  마지막 30분은 종료 처리에 남긴다. 재시작해도 마감이 갱신되지 않는다.
- single/paired 속도를 분리하고 최근 1,000 update의 p90, 평가·저장 시간과
  12% 여유를 반영한다. 이전 실행의 일시적 최악 속도를 영구 사용하지 않는다.
- FULL/HALF/CAP 추가, q 계수 탐색, D4 별도 case, Teacher 평균 distillation,
  추론 TTA/ensemble은 이번 큐에 없다. TTA의 4회전/8변환 출처가 불명확하므로
  D4 augmentation 변경은 섞지 않는다.

## 공통 고정 조건

- WV3, W112 D123, MS8+PAN1, 기존 FP32 AdamW/LR 1e-4/50K 스케줄.
- effective batch 48, microbatch 12×4. 대응 case끼리 초기화·데이터·증강 seed 동일.
- 완료된 로컬 Mature30의 **원본 Val-RRE50 → RRE90 → val-ERGAS → 이른 step**
  receipt를 그대로 재사용한다. Stage1을 다시 학습하거나 test로 donor를 고르지 않는다.
- s1 donor: G8_S1_U23 50K,
  SHA `62f1c445b8500070429f46503a68fb972a9144a4321fc9da10ccd45891186b8d`.
  s3는 자체 Mature30 donor/receipt를 사용한다. 서버 간 기다림·donor 공유 없음.
- Student의 PAN은 **원본 텐서 그대로**. 0-shift grid_sample도 실행하지 않는다.
  Aligner forward는 기존 A+S 측정 경로와 진단을 위해 유지하되 적용 shift는 0.
  따라서 Aligner를 생략한 최적화 배포 cost라고 주장하지 않는다.
- Teacher는 기존 full-u PAN과 interp23 base를 받아 고정된 error map을 만든다.
  Teacher 출력에 맞추는 Soft loss는 계속 **0/없음**이다.
- base: 증강 후 위상을 반영한 interp23tap. Teacher의 임의 소수 shift warp는
  기존 bicubic. GT·MS·최종 출력의 정합 좌표 이동 없음.
- 현재 canonical augmentation은 C4 회전 분포다. 고정 H/V flip + random90은
  8종 D4가 아니라 4종 회전과 동치다. 모든 case에서 이를 유지한다.
- 동일한 ZERO geometry의 full RR-train ECDF/tau_e를 세 case 모두 공유한다.

## loss와 계산량 대조

기존 `w = 1 + .25 d_e + .5 d_e r_q_bar`를 그대로 유지한다.
q는 기존 RGB [4,2,1] 기준이며 이번에 all-band q로 바꾸지 않는다.

`ZERO: L = mean_MASK2(w · mean_bands(|y − GT|))`

두-view는 canonical 입력 x와 정확한 90° CCW 회전 R(x)를 사용한다.

`PAIR0: L = 0.5 [Hard(y, GT, w, M) + Hard(y_R, R(GT), R(w), R(M))]`

`ROTCONS: L = L_PAIR0 + 0.05 mean_M(mean_bands(|y − R⁻¹(y_R)|))`

- 두 Student branch에 모두 gradient. stop-gradient, EMA, Teacher 출력 target 없음.
- .05는 새 탐색의 고정 계수이며, 기존 결과로 검증된 최적 계수라는 주장은 하지 않는다.
- PAIR0와 ROTCONS는 동일 샘플·동일 두 forward·같은 Hard 감독으로 비교한다.
  ZERO 대비 두-view의 계산량/유효 view 수 증가 효과는 PAIR0로 분리한다.
- Aligner/Teacher/error/q/ECDF/가중치는 canonical에서 한 번만 계산하고 정확히
  회전한다. Teacher 자체의 비등변성까지 규제에 섞지 않는 명시적 대조 설계다.
- 이미 만들어진 phase-aware interp23 base를 회전한다. 회전한 LRMS에 잘못된
  위상 메타데이터로 다시 보간하지 않는다.
- mask는 inverse rotation으로 같은 좌표에 옮긴 후 교집합. companion mask도
  canonical mask를 정확히 회전한 것이므로 이 구현에서는 원 support를 보존한다.
- inference는 모든 case에서 single forward. FR/RR에 다른 배율·후처리 없음.

## 평가·기록·업로드

- 기존 공식 원본 참조 RR20/FR20, FR full512 unmasked, RR dim_cut21 유지.
  학습 MASK2를 공식 HQNR 평가 마스크로 사용하지 않는다.
- 기존 50K 평가 시점 35회 유지. 매 optimizer step 시간/step 기록,
  100 step마다 loss/gradient 상세 기록. HQNR·SCC·ERGAS 출력 유지.
- `hard_canonical`, `hard_rotated`, `consistency_raw/weighted`, λ, views 및
  주기적 consistency gradient norm/Hard-gradient cosine을 추가 기록한다.
- 모든 raw 평가·장면별 수치·SHA·config·CSV/JSON은 로컬에 보존한다.
- 새 workbook의 **WV3-ablations**, case/seed/attempt별 **지금까지 HQNR 최고 1행**.
  best 갱신 시 평가주기와 동일하게 반영하며 5K 단위 제한이나 별도 final 행 없음.
  모든 metric과 cost는 같은 checkpoint identity에 결합한다. 비용은 새 실제
  A+S deployment를 GPU에서 측정하며 두-view 학습을 추론 cost에 넣지 않는다.
- 이전 캠페인 weight의 3역할 선정을 검토했지만, publisher가 원본 `.pt`에
  의존하므로 아직 삭제하지 않는다. `results_log/2026-10-21_zero30-retention-preview.md`
  참고. 이를 이유로 새 학습을 지연시키지 않는다.

## 실행

```bash
bash scripts/start_panda_zero30_costed.sh s1
# s3 세션에서 같은 코드와 자체 Mature30 donor가 있을 때:
bash scripts/start_panda_zero30_costed.sh s3
```

자동 순서: 고립 GPU cost 측정 → 로컬 원본 donor/SHA 확인 → 실데이터 CUDA
update/resume 및 RR/FR 추론 검증 → fresh Student 본 큐 → HQNR-best publisher.
검증 Student는 본 실험에 재사용하지 않고 Sheet에도 올리지 않는다.

CUDA 검증은 checkpoint 상태(model/optimizer/scheduler/RNG/stream)의 **정확한 복원**과
후속 update의 부동소수점 재현성을 구분한다. 기존 nondeterministic CUDA 설정에서
동일 상태 반복 update의 차이 2.33e-10을 확인했고, 격리 deterministic 진단에서는
차이가 0이었다. 생산 환경은 기존 그대로 유지한다. update replay는 최대 절대 오차
1e-8, 실제 update L2 대비 오차 1e-4 이하를 요구하며, 복원 상태/stream은 허용오차 없이
검사한다. 준비 실패 기록은 별도 보존하고 30시간 시작 시각을 초기화하지 않았다.

root: `work_dir/panda_zero30/PANDA_S13_ZERO_ROTCONS_HARDONLY_30H_20261021_v1/<server>/WV3`.
중간 변경은 아직 시작하지 않은 그룹에 대한 명시적 skip 또는 새 버전 캠페인으로 한다.
실행 중 소스·loss·seed를 바꾸지 않는다. 결과 해석은 서버 내 대응 비교를 우선한다.
