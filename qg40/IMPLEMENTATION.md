# QG40 구현·검토 인계 (2026-09-20)

## 반영 범위

`research_log/0920`의 40h 운영 MD·Cases CSV·Baseline11 CSV와 사용자가 지정한
DistributionAware Preparation MD를 SHA로 고정했다. 제공되지 않은 companion
JSON 대신 이 네 파일의 계약에서 89개 config와 registry를 생성했다.
89개 전체를 무조건 학습하지 않으며, 기본 11개·선택 profile·조건부 C3·유한 예비 큐를 구분한다.
이번 반영은 Preparation 기준이며 INDEP1 delta는 제외했다. s4→s2 QB_TA와
s3→s5 GF2_TA를 공유하고, 40h/36h 운영정책은 유지한다. Registry는
`QG40_REGISTRY_20260920_v2_PREPARATION`, 수치 Method revision은 기존 v1이다.

| 계획 항목 | 구현 경로 |
|---|---|
| 서버·센서·seed·reference 계약, 89개 YAML | `plan.py`, `config/qg40/` |
| C4 모델·P0/PLH·독립 Student A 복제·gradient routing | `model.py`, `losses.py`, `training.py` |
| QB 2047 / GF2 1023, native PAN LP, source·band·split 검증 | `data.py`, `preflight.py` |
| split별 입력 분포·QB raw residual·phase ramp/impulse | `input_distribution.py`, `phase_audit.py` |
| exact50K calibration3072, 전체 train×4-view q, 전달·실제 online parity | `calibration.py`, `references.py`, `reference_parity.py` |
| Teacher eT 분포·band/texture 잔차·train-cal/validation 비교 | `error_distribution.py` |
| train128/grad24·A gradient·cross-axis/border·크기 비교 | `diagnostics.py`, `alignment_diagnostics.py`, `p1.py` |
| A/U 교차, A24R/B20/E10 및 C3 증거 | `branch_evidence.py`, `policy.py` |
| 완료 update별 정확한 sample/view exposure·재개 | `exposure.py`, `training.py` |
| 공통40h·36h 신규 차단·atomic 예약·평가부채·정확 재개 | `controller.py`, `training.py` |
| local GPU/디스크 inventory·checkpoint 저장 예약·block 보류 | `resources.py`, `controller.py` |
| RR20/Q4, native FR20/raw HQNR, 선택점 5종 | `evaluation.py`, `postrun.py` |
| RMSE·CC·JQM 포함, 기존행 보호·재전송·숫자 표시4자리 | `upload.py`, `sheet_helpers.py` |
| 명시적 시작·기존 WV3 종료 대기·HEAD 보존 배포 | `tools/qg40_runner.py`, `deployment.py` |

기존 `fh12/`, `fh20r1/`, `model/`, `pa/`, 기존 metric/runner 파일은 수정하지 않았다.
현재 WV3의 수치 identity를 바꾸지 않도록 별도 모듈·진입점으로 추가했다.
`RAW_MAX`는 HQNR 최대점이며 TARGET/VAL/E_MIN은 별도 이름·용도로 저장한다.
공식 FR에 masking·PAN reference 재이동을 추가하지 않았다.

## Preparation 반영에서 보완한 사항

- q cache는 checksum 외에 실제 frozen Teacher의 고정 16 ID×4 view AXIS16
  재계산도 통과해야 한다. CAL 발행·import·Student load에 연결했으며 동일
  source/data/reference와 atol=3e-6/rtol=2e-5를 검증한다. 합성 receipt는 실제
  reference에 허용하지 않는다. 높은 positive q 자체는 차단 사유가 아니다.
- 입력 통계는 split별 최대 128 patch 표본이다. 분위수의 픽셀 subsampling과
  covariance의 측정 범위를 명시했다. τ calibration은 원래 3072 base의
  full-pixel pooled median이며, 추가 error report의 표본/시간 제한과 분리한다.
- P1 분석은 bounded budget·공통 deadline 안에서 실행하고 불완전하면 기록한다.
  분석 실패·진단 전용 파일 IO 실패가 BASE checkpoint/학습을 막지 않도록 했다.
  기존 evidence SHA나 model/Teacher state 변경은 계속 무결성 오류다.
- QB phase 한 칸을 HR 1px/LR spacing 0.25로 확인하는 독립 검사를 추가했다.
  원 repair 코드의 과거 설명을 실제 PAN–MS 물리적 shift 측정으로 사용하지 않는다.
- 완료 update의 base/view count를 fullstate에 저장한다. 진단 전후 전체 RNG·
  하위 module mode·requires_grad·parameter gradient를 복원한다.
- PAN–MS common-band proxy는 동일한 box-filter/stride4 후 밴드별 고정 3×3
  LR 상관만 비교한다. low-texture·경쟁 peak·밴드별 불일치·polarity를 기록하고
  물리적 shift 정답으로 해석하지 않는다. SRF/MTF fitting이나 입력 보정은 없다.
- 첫 run 이후 저장한 timing observation을 다음 block admission이 읽을 때
  `profile`/memory 부가 필드로 TypeError가 나던 경로를 수정했다. 실제 controller
  완료→state 직렬화/복원→다음 admission 회귀 테스트를 추가했다.

결과 위치와 제한은 `README.md`의 Preparation 추가 진단을 따른다. 불완전한
P1 증거로 조건부 분기를 승인하지 않는다. 원 MD/CSV 내용은 수정하지 않았다.

## 검증 범위와 한계

새 테스트는 실제 수식·C4/C8 초기화·gradient 분리·합성 소규모 exact-resume와
임시 디렉터리 기반 controller/reference/업로드 시나리오를 검사한다.
업로드 테스트는 mock Sheet이며 실제 원격 Sheet 갱신 검증은 아니다.
즉시 시작 경로까지 반영 후 QG40 테스트 161개와 기존 FH12/FH20R1/reporting_extra
회귀 테스트 297개를 통과했다. 총 458개이며 실제 50K 본 학습이나 원격 서버
실행 횟수가 아니다. 합성 Teacher/Student에서는 P1 timeout을 강제로 넣어도
연속 학습과 fullstate 재개의 model·optimizer·scheduler·전체 RNG·sampler·노출량이 일치했다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  python -m unittest discover -s qg40 -t . -p 'test_*.py' -q
```

CPU에서 실제 크기 Teacher W112 d123(P0)와 Student W104 d122(PLH)의
`1×4×64×64` 출력을 확인했다. A+U parameter 수는 각각 2,755,574 / 2,198,070이다.
이 값은 GPU latency·VRAM·실데이터 전체 평가 또는 성능 목표 달성의 검증을 의미하지 않는다.

리뷰에서 보완한 항목은 기본 Teacher의 CAL 시간 예약, 미완 평가/업로드 상태 분리,
분기 진단의 checkpoint·배열·Q4 대각 재현성, reference 전송 마감시간·경로 보호,
배포물의 symlink/누락 파일 차단이다. 실패를 임의 PASS나 fresh 재학습으로 전환하지 않는다.

## 즉시 시작 지시 반영

사용자의 후속 지시에 따라 공통 t0를 2026-09-20 05:12:28 KST로 고정했다.
`campaign_window.json`을 커밋/배포에 포함하며 별도 시간 입력은 필요 없다.
`sensor_sources.json`은 확인한 기존 배포본의 상대 경로·전체 SHA·출처를 담고,
`bootstrap.py`가 이를 로컬 경로로 연결한다. 수동 sensor JSON이 있으면 우선한다.
실제 QB/GF2에 대해 이 bootstrap으로 전체 source SHA를 다시 확인하고 bound
SensorSpec 생성까지 통과했다. 이는 전체 P0 preflight/GPU smoke의 완료가 아니다.
밴드 순서는 제공 코드/설치 데이터 기록의 규약으로 명시하며 H5 내장 메타데이터로
확인했다는 주장은 하지 않는다. 근거와 제한은 `SENSOR_SOURCE_EVIDENCE.md`에 있다.

## 실제 실행 시 유지하는 확인

1. 모든 서버가 같은 window를 읽는다. pull만으로 자동 실행하지 않으며
   `bash tools/qg40_start.sh` 한 명령으로 등록·preflight·큐 진행을 요청한다.
2. 실제 파일이 source catalog와 달라지면 차단한다. 이후 전체 QB msfix·LP·
   DN/geometry 검사 및 실제 GPU smoke를 수행하고 나서 LOCAL_READY를 기록한다.
3. s4→s2 QB_TA, s3→s5 GF2_TA archive를 지정 incoming 경로로 전달한다.
   export/import 및 자동 대기는 구현했으나 원격 주소 없이 전송 설정을 생성하지 않았다.

Preparation 검증 시점에는 실제 학습·campaign clock·cron·hold·Sheet를 활성화하지 않았다.
후속 즉시 시작 요청의 실제 등록/대기 상태는 `work_dir/_qg40/<server>/`에 별도 기록한다.
각 서버의 실데이터 preflight, Teacher50K/CAL, 원격 parity, live Sheet readback은
실행 후에만 완료로 기록한다. 시작·배포 명령과 파일 위치는 `README.md`를 따른다.

## Git 인계

s1에 기존 FH20R1 학습이 실행 중인 것을 확인했으므로 체크아웃된 `main`/HEAD는
유지하고 `qg40-20260920-preparation` 브랜치에만 커밋한다. 원 index/작업 내용도
변경하지 않는다. 사용자가 다음 명령으로 원격 main에 fast-forward push할 수 있다:

```bash
git push origin qg40-20260920-preparation:main
```

진행 중인 FH12/FH20R1/QG40이 있는 체크아웃의 HEAD를 임의로 바꾸면 source pin이
깨질 수 있다. 해당 서버는 overlay를 사용하거나 기존 작업 종료 후 갱신한다.
현재 s1 역시 등록 후에는 캠페인이 끝날 때까지 기존 HEAD를 유지한다.
