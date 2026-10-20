# PANDA: s1/s3 성숙 Aligner·감소 배율 비교 — 서버별 30시간

## 확정 범위

- s1은 현재 `G8_S1_BIC` case를 50K까지 완료하고 최종 checkpoint·공식 평가를 저장한 뒤 기존 큐를 정상 종료한다. 기존 case 중간 중단/기존 source 수정 없음.
- s3은 해당 서버 세션에서 명시적으로 실행한다. 원격 실행이나 서버 간 donor 전달·대기는 하지 않는다.
- 새 캠페인 ID: `PANDA_S13_MATURE_ALIGN_SCHEDULE_30H_20261020_v1`.
- 두 서버 모두 WV3, W112/D123, G8 global2 Aligner, 9-channel U-Net. Stage1 Scharr target은 **8밴드 평균 GT**이다. 밴드별 16-output/16-input 구조는 이번 기본 큐에서 제외한다.
- RR train loss만 사용. Aligner 입력은 제공 LMS. 복원기 LRMS base는 flip/rotation 이후 phase-aware `interp23` 업샘플링. 소수 픽셀 shift warp는 기존 bicubic이다.
- Stage2에서 PAN만 이동. MS·GT·최종 출력 좌표는 원래 MS 좌표로 유지한다. RR/FR 모두 동일한 현재 PAN 화소 단위 규칙이며 ×4 변환을 하지 않는다.

## 질문과 한계

성숙한 Aligner를 고정한 상황에서 Student 자체 보정 증가를 **PAN 적용 배율 감소**로 완화할 수 있는가?

기존 관측의 `출력 위치 ≈ Aligner 적용분 + g × 잔여분`은 진단적 1차 근사이다. `g`는 추가 오정합 크기, donor, 학습 정책에 따라 달라지므로 고정 상수로 취급하지 않는다. FR 최적 위치와 cap 0.85는 이미 본 test 진단의 영향을 받은 탐색적 선택이며, 독립 test 검증·새로운 물리적 정합 정답으로 주장하지 않는다.

고정 0.25 대조군은 감소 스케줄의 개선이 단순히 마지막 배율에서 오는 것인지 구분한다. HALF/DOWN의 두 번째 seed는 **Student 초기화·데이터 순서 반복**이며 Stage1 donor를 새로 학습하는 전체 파이프라인 반복이 아니다.

## Stage1: 정합 기준 donor 선정

- s1: 로컬 `PB30_WV3_S1_G8_S1_U23_R1_S261016101`의 **완료된 50K run, 저장된 35개 평가 checkpoint 전체**를 재선정한다. 기존 HQNR 선택 파일·모델·평가를 덮어쓰지 않는다.
- s3: 같은 G8/interp23 Stage1을 seed `261016101`, 50K까지 새로 학습한 뒤 전체 checkpoint를 선정한다. 기존 RGB donor로 조용히 대체하지 않는다.
- 우선순위: **Val-RRE50 최소 → Val-RRE90 최소 → Val-ERGAS 최소 → 이른 step**.
- RRE는 validation에서 실제 `W(PAN,u)`와 8밴드 평균 GT 간 남은 translation을 독립적인 고정 registration으로 추정한 크기이다. RRE50=중앙값, RRE90=90백분위수. 물리적 ground-truth EPE나 표준 논문 지표가 아니다.
- validation 1,080개 중 GT texture만으로 고정 적격 집합을 만든다. 후보별로 성공 패치만 남겨 선택 편향을 만들지 않는다. 적격 패치 registration 실패는 고정 큰 벌점으로 집계한다. source SHA·sample ID·ROI·등록 설정·실패율·epsilon equivariance를 receipt에 보존한다.
- FR HQNR/RR20 ERGAS는 **선택된 정확히 그 checkpoint의 기존 공식 평가**를 보고할 뿐 Stage1 순위에 사용하지 않는다.
- 새 Stage1의 `models/selection.json`은 전체 저장 목록과 HQNR 모니터만 관리한다(`best_selected=null`). 실제 best는 캠페인 `donors/rre_selection.json`과 `donors/G8_MATURE_U23_best_val_rre.json`이며 Stage2는 이것만 사용한다. 과거 s1 run의 HQNR pointer는 원본 보존을 위해 그대로 둔다.
- 두 서버의 donor와 수치는 따로 해석한다. 서로 다른 서버의 절대값으로 승자를 정하지 않는다.

## Stage2 고정 조건

같은 서버의 모든 case는 동일 RRE-selected A/T pair를 사용한다.

- Aligner·Teacher 고정, Student fresh 50K.
- 기존 conditional-q, MASK2, `alpha=.25 / beta=.05 / gamma=.5`, learning rate `1e-4` 및 기존 100-step warmup 유지.
- Teacher는 Stage1과 같은 **full-u PAN**을 항상 받는다. Student에만 아래 적용 정책을 바꾼다.
- q는 실제 Student PAN geometry 기준으로 calibration한다. full RR train, K=5 exact ECDF. `s100/s050/s0375/s025/cap085`별 불변 artifact를 만들고 같은 geometry끼리 공유한다.
- Stage1 정합 목표는 mean8이지만 Stage2 conditional-q 목표는 기존 RGB mean `[4,2,1]` 그대로이다. 이번에 loss까지 바꾸지 않는다.
- Teacher/GT 자체는 고정이지만 Student PAN의 valid support와 Teacher support의 교집합은 정책별로 달라질 수 있다. 따라서 calibration의 `tau_e`/실효 hard·soft 가중치까지 완전히 동일하다고 주장하지 않는다.
- optimizer/LR/data/RNG를 phase 경계에서 재시작하지 않는다. 스케줄·단계별 calibration SHA·완료 update를 checkpoint에 함께 저장한다.
- Stage2 best는 **공식 FR HQNR → RR ERGAS → 이른 step** 유지. SCC 포함 나머지 지표는 계속 평가·기록한다.

| 정책 | Student PAN 적용 | 목적 |
|---|---|---|
| FULL | `u` | 성숙 donor 완전 적용 대조 |
| HALF | `0.5u` | 기존 유망한 단방향 절반 대조 |
| DOWN | 1–10K: `0.5u`; 10,001–30K: `0.375u`; 30,001–50K: `0.25u` | 감소 후 마지막 20K 유지 |
| LOW | `0.25u` 전 구간 | DOWN의 최종 배율 대조 |
| CAP085 | `u × min(1,0.85/||u||₂)` | 배율이 아닌 절대 이동량 상한 |

DOWN의 10K checkpoint는 0.5, 30K checkpoint는 0.375로 평가·배포한다. 다음 optimizer update부터 새 배율이다. 50K는 0.25다. 임의로 test 평가 시 배율만 바꾸지 않는다.

## 우선순위 큐

| 서버 | 준비 | 핵심 R1 | 조건부 R2 쌍 | 버퍼 |
|---|---|---|---|---|
| s1 | 기존 G8 full50K 전체 RRE 재선정 | FULL → HALF → DOWN → LOW | HALF → DOWN | CAP085 R1 |
| s3 | fresh G8 Stage1 50K + RRE 선정 | FULL → HALF → DOWN → LOW | HALF → DOWN | CAP085 R1 → CAP085 R2 |

Student R1 seed=`261018101`, R2 seed=`261018102`. 동일 seed case끼리 초기화와 입력 순서를 맞춘다.

예상 시간은 보장 시간이 아니다. s1 실측 Stage2 약 3.61h/case이므로 핵심 4벌 약 14.4h, 반복 2벌을 합치면 약 21.7h + 준비·평가·calibration이다. s3 prior는 Stage1 4h, Stage2 2.1h로 아직 새 구성 실측 전이다.

실행기는:

1. 새 캠페인 시작부터 setup/선정/calibration/학습/평가를 포함해 **30h wall time**을 지속 저장한다. 기존 case 종료 대기는 이 시간에 포함하지 않는다.
2. 업데이트 p90·평가·저장 시간 실측에 1.2 safety factor를 적용한다.
3. 30분 마감 여유를 남긴다. 재시작해도 deadline을 초기화하지 않는다.
4. HALF/DOWN R2는 두 벌 전체가 들어갈 때에만 진입한다. cap은 후순위이며 시간이 부족하면 자동 제외한다.
5. 부족한 donor를 다른 서버나 다른 case로 대체하지 않는다.

## 중간 판단과 변경

FULL/HALF 후 정합 잔여·HQNR peak 시점·50K 지표를 확인한다. DOWN/LOW 완료 후에는 동일 seed에서 다음을 비교한다.

- 공식 HQNR/ERGAS 동시 통과 유무와 시점, best와 50K의 차이.
- DOWN이 HALF보다 공동 지표를 개선하는지, LOW보다 개선해야 스케줄 자체의 근거가 된다.
- 후기 HQNR 하락이 남으면 배율 감소가 Student 자체 보정을 충분히 상쇄하지 못한 것이다. 성공을 미리 가정하지 않는다.
- PAN 배율/cap은 Student가 내부에서 더하는 정렬량 `g`나 최종 출력 HF 위치를 직접 제한하지 않는다.
- 보조 위치 분석은 별도 진단이며 공식 metric/reference를 바꾸지 않는다.

중간 변경은 **아직 시작하지 않은 group만 skip** 가능하다. 각 캠페인 root의 `pending_control.json`:

```json
{"request_id":"interim-01","reason":"중간 결과에 따른 명시적 결정","skip_groups":["cap_screen","cap_repeat"]}
```

`cap_repeat`은 s3에만 있으므로 s1 파일에는 넣지 않는다. 이미 시작한 group이나 배율·loss를 수정하면 실행기가 거부한다. 새로운 정책은 새 case ID/새 계획으로 명시적으로 등록해야 한다. 실행 중 Python/config를 직접 수정하지 않는다.

## 평가·Sheet·로컬 결과

- 기존 공식 RR20/FR20 원본 reference, unmasked 평가를 그대로 사용한다. loss용 valid mask를 공식 평가에 전용하지 않는다.
- HQNR/SCC/ERGAS뿐 아니라 RMSE/CC/JQM 등 기존 전체 metric, scene별 값, SHA를 동일 checkpoint에서 저장한다.
- 새 workbook의 `WV3-ablations` 사용. s1/s3는 별도 Run_ID, 기존 행 보존.
- Stage1: `BEST_VAL_RRE` 1행. s1 재사용 source는 원래 run/학습 날짜/선택 step과 재사용 사실을 보존하며 새 학습 50K로 위장하지 않는다.
- Stage2: `HQNR ≥ .960 AND ERGAS ≤ 2.045` 통과 checkpoint 전부, 없으면 HQNR→ERGAS 대표 1행. 추가 `FINAL50K`는 endpoint 비교용이며 이미 같은 checkpoint를 게시했다면 중복하지 않는다. 매 1K 무조건 게시하지 않는다.
- 전체 raw 평가는 로컬 run에 모두 저장한다. 게시된 selected row의 로컬 projection/CSV도 저장한다. 비용을 측정하지 못했으면 수치를 만들지 않고 공란이다.

## 실행

```bash
# s3: 파일 전달 후 해당 서버에서, 기존 GPU 작업이 끝난 상태로 실행
bash scripts/run_panda_mature30_docker.sh s3

# 읽기 전용 확인 / CPU regression
python -m panda_mature30 check --server s3
python -m panda_mature30 smoke
```

s1에는 완료 후 전환 watcher가 별도 등록된다. watcher는 현재 case의 완료 상태와 최종 평가/저장을 확인하고 정확히 지정한 trainer에 정상 종료 신호만 보낸다. 신규 코드 검증 완료 receipt의 파일 SHA가 맞아야 새 Docker를 시작한다. 이 파일을 수동으로 먼저 만들지 않는다.

결과 root: `work_dir/panda_mature30/<campaign_id>/<server>/WV3/`.
주요 기록: `campaign_state.json`, `donors/rre_selection.json`, `calibration/`, `runs/`, `publication/`.
