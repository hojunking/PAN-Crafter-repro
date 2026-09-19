# QG40: QB/GF2 공통 40시간 캠페인

원본은 `research_log/0920`의 40h 운영 MD·Cases CSV·Baseline11 CSV와 사용자가
선택한 `PAN_QB_GF2_DistributionAware_Preparation_2026-09-20.md`다. 누락된
companion JSON을 제공받았다고 가정하지 않고, 이 네 파일의 SHA를 고정한다.
Preparation의 Method·센서·분포 진단·TA 공유 규약을 반영하고, 이미 정의한
40h 시간정책·조건부 큐는 유지한다. **INDEP1 delta CSV는 적용하지 않는다.**
Registry revision은 `QG40_REGISTRY_20260920_v2_PREPARATION`이며 수치 Method는
`QG40_SYNC_FREQ_C4_v1` 그대로다.
89개는 정의 전체이며 전부 mandatory queue가 아니다. 기본 11개, 조건부 주군
31–39개, C3 최대 6개와 유한 BASE 예비를 포함한 활성 상한은 73개다.

| 서버 | 센서·주 reference | 최초 경로 |
|---|---|---|
| s1 | QB TB / TS81002 | fresh Teacher → CAL → SS82001/82002 |
| s2 | QB TA / TS81001 | s4 reference import → SS82003 |
| s3 | GF2 TA / TS91001 | fresh Teacher → CAL → SS92001/92002 |
| s4 | QB TA / TS81001 | fresh Teacher → CAL → SS82001/82002 |
| s5 | GF2 TA / TS91001 | s3 reference import → SS92003 |

## 구현과 실제 실행 상태의 구분

코드 생성·CPU 테스트는 학습 실행이나 `LOCAL_READY` 인증이 아니다.
이번 배포에는 공통 시작시각과 검증한 배포본의 source catalog가 포함된다.
기본 경로의 파일 SHA가 catalog와 같으면 실제 로컬 경로를 자동 연결하므로
별도 window/binding 명령이 필요하지 않다. 출처는 `SENSOR_SOURCE_EVIDENCE.md`에
기록했으며, 물리적 밴드 순서를 파일명/채널 수만으로 인증한 것은 아니다.
τR·qref·q cache는 fresh Teacher exact50K 이후 실제로 계산하며 null을 WV3
상수로 채우지 않는다. 다른 센서 성능이나 전체 서버 barrier는 없다.

## 생성·읽기 전용 확인

환경에 맞는 Python을 사용한다. s1 기본은
`/home/knuvi/miniconda3/envs/pancrafter/bin/python`이다.

```bash
python tools/qg40_runner.py build
python tools/qg40_runner.py inventory --sensor QB
python tools/qg40_runner.py binding-template --sensor QB
bash tools/qg40_start.sh --dry-run
```

`build`는 `config/qg40/`에 89개 YAML과 registry만 생성한다. `check/status`는
실제 next-two, 단일 선택 profile, reference, tab, 남은 시간, 평가부채를 읽는다.
unbound window는 남은 시간을 꾸미지 않고 null로 출력한다. `binding-template`은
실제 절대 경로·SHA와 **미확인 provenance null**을 stdout에만 출력한다.
다른 배포본을 명시적으로 연결할 때만 템플릿을 검토하여
`config/qg40/QB_sensor.json` 또는 `GF2_sensor.json`으로 저장한다. 명시 설정은
기본 catalog보다 우선하며 서버별 경로가 달라도 source/LP identity를 보존해야 한다.

필수 binding은 `band_order`, 각 split의 원본 `source_identity`, `source_provenance`의
band-order/MTF 근거·DN 단위·no-data/saturation 정책·split 대응 근거다.
QB에는 raw train/val의 경로·SHA·msfix recipe 근거도 필요하다. 없는 근거를
`PASS` 문자열로 채우지 않는다. runtime에서 전체 raw GT/PAN 불변, MS/LMS
레시피와 phase, DN·유한값·geometry 및 전체 LP correspondence를 다시 검사한다.

## 공통 시계와 명시적 시작

사용자의 즉시 시작 지시에 따라 `qg40/campaign_window.json`에 공통 t0를
**2026-09-20 05:12:28 KST**로 고정했다. 36h 신규 입장 마감은
**09-21 17:12:28 KST**, 40h 종료는 **09-21 21:12:28 KST**다.
준비·현재 case 종료 대기 시간도 포함한다. 늦게 시작한 서버나 재개는 시계를
리셋하지 않는다. runtime과 배포 window가 다르면 자동 덮어쓰기 없이 차단한다.

각 서버에서 코드와 기존 데이터가 배치된 뒤 다음 한 명령으로 등록·시작한다:

```bash
bash tools/qg40_start.sh
```

`git pull` 자체가 학습을 자동 실행하지는 않는다. 위 명령은 source catalog의
실제 파일·SHA부터 확인하며, 데이터 누락/변경을 임의 우회하지 않는다.

기존 `gspread/server.txt`로 서버를 인식한다. start는 WV3의 현재 case를 kill하지
않고, 다음 WV3 case 입장을 hold한 뒤 기존 runner lock과 local GPU가 자연스럽게
풀리기를 기다린다. 원 WV3 source·HEAD·checkpoint·ledger는 수정하지 않는다.
이후 local preflight → 자기 Teacher/CAL/import → Student로 진행한다.

Docker에서는 **기존 WV3 controller가 종료되면 같이 사라질 컨테이너 안에
background child를 띄우지 않는다.** 동일 이미지·GPU·bind mount의 독립 영속
컨테이너에서 `bash tools/qg40_start.sh --foreground`를 실행한다. 원 컨테이너를
강제 종료하지 않는다. host의 Python 환경이 검증되어 있으면 native start도 가능하다.

start는 가능할 때 QG40 전용 cron을 추가하며 기존 cron·watchdog는 보존한다.
cron 부재/권한 오류는 경고와 `recovery.json`에 남기고 자동 복구를 인증하지 않는다.
무결성 오류는 자동 재학습하지 않는다. 신호 중단은 검증된 fullstate로만 재개한다.

## Reference 전달

s4/s3는 기본 Teacher exact50K→calibration을 마치면 각각 아래 파일을 자동 생성한다.

- s4: `work_dir/_qg40/s4/outgoing/QB_TA.tar.gz`
- s3: `work_dir/_qg40/s3/outgoing/GF2_TA.tar.gz`

이를 소비 서버의 다음 경로에 완전히 복사한 뒤 원자적 rename으로 게시한다.
미완성 전송 파일은 `.part` 등의 별도 이름을 쓴다.

- s2: `work_dir/_qg40/s2/incoming/QB_TA.tar.gz`
- s5: `work_dir/_qg40/s5/incoming/GF2_TA.tar.gz`

소비 서버는 `WAIT_REFERENCE`로 기다리다 import를 진행한다. 전송에는 원 manifest,
exact50K A/U·fullstate·config·calibration·전체 q cache·train probe가 포함된다.
경로 relocation은 별도 bridge에 기록하고 원 수치 source는 덮어쓰지 않는다.
SHA·센서·band·DN·source·실제 output/c parity 실패 시 그 reference만 차단한다.
CAL 발행 전·consumer import·Student load에서 고정 train ID 16개×네 HV/rotation
view의 AXIS16 q를 frozen Teacher로 다시 계산한다(atol=3e-6, rtol=2e-5).
Receipt는 실제 cache/online 값과 ID/view/source/data identity에 묶인다. 이 표본
검사는 전체 cache SHA 검증을 보완하며, 전체 N×4를 온라인 재계산했다는 뜻은 아니다.
원격 주소/자격증명이 제공되지 않아 서버 간 파일 전송 자체를 임의 설정하지 않는다.

## 시간·분기·결과

- 36h 이후 새 학습 금지, 40h 안전 update 경계 fullstate 저장. 이미 입장한
  pair라도 36h 이후 남은 **fresh** run을 새로 시작하지 않는다.
- Teacher 입장에는 calibration까지, C3는 Teacher+CAL+Student2 전체를 예약한다.
  현재 평가부채+종결4h를 빼며 같은 서버/센서/역할/구조 실측 p90×1.25로 갱신한다.
- block 입장 시 실제 디스크 여유·GPU inventory를 기록한다. 체크포인트·진단·
  calibration 저장 예약이 부족하면 해당 block을 보류한다. 같은 구성의 실측 peak가
  없으면 VRAM 필요량을 미측정으로 표시하며 충분하다고 인증하지 않는다.
- train128·gradient24를 10000/24240/50000에 기록한다. A24R의 A/U 교차는
  별도 진단이며 공식 TARGET을 덮어쓰지 않는다. 증거 부족/예산 부족이면 BASE.
- s2/s5 한 축만 선택하고 screen 탈락 후 다른 후보로 교체하지 않는다.
  s1/s3의 C3는 별도 reference이며 원 QB_TA/GF2_TA를 대체하지 않는다.
- 실제 Q4, QB2047/GF21023, RR20 `20:-21`, native full512 FR20·raw-original
  mean-per-scene HQNR. masking이나 정답/reference PAN shifting을 추가하지 않는다.
- RAW_MAX/TARGET/EXACT50K/RR_VAL_SELECTED/E_MIN_DIAG50을 같은 checkpoint로
  구분하고, strict H>.920/.964 판정에 반올림값을 쓰지 않는다.
- HQNR·SCC·ERGAS를 평가 때 출력한다. RMSE·CC·JQM도 평가/업로드하며 JQM은
  명시한 SRF 추정 variant이다. SIPSA-equivalent라고 주장하지 않는다.
- Sheet는 기존 탭 의미별 header를 읽어 upsert한다. 수식·validation cell과 과거행을
  보호하며 숫자는 원 precision, 표시만 소수4자리다. 미측정 값은 공란이다.
- 실패한 업로드는 로컬 완료 결과에서 upload-only로 재시도한다. 완결 후에도
  재전송 가능하며 학습·40h clock을 다시 시작하지 않는다.

## Preparation 추가 진단

- P0 phase 좌표 검사: `work_dir/_qg40/<server>/phase_coordinates.json`.
  QB decimation phase 한 칸은 HR 1px, LR sampling 간격의 0.25라는 것을
  ramp/impulse로 확인한다. 실제 장면의 물리적 PAN–MS shift를 측정했다는 뜻은
  아니며, 기존 repair 코드나 데이터 recipe를 바꾸지 않는다.
- 입력 분포: `work_dir/_qg40/<server>/diagnostics/input_distribution.json`.
  split별 최대 128개 patch, 표본 분위수·공분산·PAN/LP/HP·edge·저texture를
  기록한다. sample ID·픽셀 추출 방식·실제 범위를 남기고 정규화 상수는 fit하지 않는다.
- Teacher error 분포: reference의 `reconstruction_distribution.json`.
  train-cal/validation의 eT 분위수·band별 L1/MSE/bias·texture strata·τ floor를
  기록한다. 최대 60초 내 측정분만 기록하며 τ/q calibration을 다시 fit하지 않는다.
- 학습 진단: `work_dir/<run>/diagnostics/update_<step>.json/.npz`에
  10K/24,240/50K의 loss·U/A gradient·d/advantage/q·native c·cross-axis slope·
  border 비율을 기록한다. `alignment_sizes_<step>.json`에는 최대 두 RR/FR
  scene의 paired crop64/128 및 full256/512 A-only 비교를 남긴다.
  동일 box-filter/stride4의 PAN–MS 밴드별 3×3 LR 상관 proxy도 함께 기록한다.
  낮은 texture·반복 peak·밴드별 불일치·분광 polarity를 표시하며, 이 proxy의
  peak를 물리적 shift 정답이나 A의 보정량으로 사용하지 않는다. 모델 AXIS16
  radius·입력·공식 평가 reference는 그대로다.
- 노출량: `meta/training_status.json`의 `sample_exposure`와
  `last/training_state.pt`의 `exposure_counts`에 실제 완료 update의 base/view
  횟수를 기록한다. 재개 시 정확히 복원하고 `updates×batch` 합을 검증한다.

입력·학습·크기 분석은 각각 기본 120/60/30초의 cooperative budget과 공통
deadline을 함께 적용한다. P1 timeout/진단 저장 실패는 warning/INCOMPLETE이며
BASE 학습을 막지 않는다. 기록 자체가 불가능하면 runner log에 경고한다.
불완전한 증거로 ALT/C3 분기를 승인하지 않으며, 기존 증거 변조나 모델 weight
변경은 계속 fatal이다. 진단 중 RNG·mode·gradient를 복원한다.

## 배포와 검증

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  python -m unittest discover -s qg40 -t . -p 'test_*.py' -q
python tools/qg40_runner.py bundle
```

`-t .`는 최상위 `model/`과 `qg40/model.py`의 테스트 discovery 충돌을 막는다.
배포물은 `work_dir/_qg40/deployment/qg40-overlay.tar.gz`. 신규 QG40 경로·명세·
생성 config·공통 시계·source catalog/근거만 포함하며 원 trainer/watchdog/실제
데이터/credential은 포함하지 않는다.
활성 WV3의 HEAD를 바꾸는 pull/checkout 대신 새 파일만 전달한다. QG40 등록
후에는 이 코드도 수치 identity에 묶이므로 임의 overwrite나 identity 수정은 금지다.
이전 QG40 release로 이미 등록/학습한 서버를 이 버전으로 자동 migration하지
않는다. 기존 reference/fullstate를 새 revision으로 이름만 바꾸어 사용하지 않는다.

데이터/실제 GPU preflight, 센서별 Teacher·τ/q, 원격 import parity, live Sheet
readback은 해당 단계가 실제 실행된 뒤에만 PASS로 기록한다.
