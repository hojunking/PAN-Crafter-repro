# PANDA GF2 G20 — fixed architecture, all five servers

계획/CSV: `research_log/PANDA_GF2_G20_ALL5_*_2026-09-20.*`.
Campaign: `PANDA_GF2_G20_ALL5_20260920_v1`.

기존 QG40 수치 모델을 변경하지 않고 별도 `g20/`, `config/g20/`,
`tools/g20_runner.py`를 추가했다. 기존 QG40/FH12/FH20R1 runtime ledger,
Teacher, 학습 중 config와 checkpoint를 구현 단계에서 수정하지 않는다.

## 범위와 선택 규칙

- 전 서버 GF2/C4/DN1023. Teacher P0 W112D123, Student PLH W104D122 고정.
- 기본 Teacher 4 + Student 28의 실행 config 32개. 조건부 확인 20 + 전이 4는
  registry의 `NOT_ADMITTED` 슬롯일 뿐, BASE로 대체한 실행 config가 아니다.
- s1 R0/R1, s2 동일 Teacher seed의 R2/R4 우선 후 R3, s3 A1/A9,
  s4 E1/E4, s5 K05/K20. 전 서버 같은 Student seed의 대응 비교를 유지한다.
- Student U에는 hard+soft+q-weighted edge, A에는 q-weighted hard만 미분한다.
  A1/A9는 전체 cosine schedule의 peak LR 변경이며 late-LR 분기가 아니다.
- 이 계획의 주 분석은 Exact50K와 RR_VAL_SELECTED다. RAW_MAX, TARGET,
  E_MIN_DIAG50도 보존하지만 test-aware 진단으로 명시한다. 기존 캠페인의
  checkpoint 선택 규칙은 바꾸지 않는다. 모든 학습 로그에 HQNR/SCC/ERGAS를 출력한다.
- RR20/Q4/crop20:-21 및 FR20/native PAN/full512/raw-original을 유지한다.
  PAN reference shift/masking은 하지 않는다. Signed Ds는 실제 공식 Ds의
  Q_high/Q_low 20×4를 저장하고 절댓값 평균으로 원래 Ds가 재구성되는지 검증한다.
- Sheet는 서버별 GF2 탭에 5개 선택점의 RMSE/CC/JQM까지 올린다.
  JQM variant를 명시하며 표시만 소수점 4자리, 판단/원본 JSON은 full precision이다.

## 비활성 상태 확인

아래 명령은 clock, 학습, hold, cron, Sheet를 시작하지 않는다.

```bash
python tools/g20_runner.py build
python tools/g20_runner.py status --server s1
python tools/g20_runner.py inventory
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m unittest discover -s g20 -t . -q
```

`build`는 32개 YAML과 56개 슬롯 registry만 생성한다. 서버별 경로·실측
reference 값은 시작 후 검증된 local assets로 resolved config에 결합한다.
같은 run의 기존 config/원본 결과가 다르면 덮어쓰지 않는다.

## 실제 실행 — 별도 명시적 작업

실제 준비/전환을 시작할 때 한 서버에서만 공통 UTC t0를 만든다. 서버마다
자기 현재시각을 넣거나 QG40의 기존 deadline을 재사용하면 안 된다.

```bash
python tools/g20_runner.py window --t0 "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

생성된 `work_dir/_g20/campaign_window.json`을 모든 서버에 동일하게 전달하고:

```bash
bash tools/g20_start.sh --server s1 --window /absolute/path/campaign_window.json
# 다른 서버는 같은 window 파일에 --server s2 / s3 / s4 / s5
```

지속 실행 컨테이너에서는 `--foreground`를 추가한다. 시작 요청이 실제
학습 개시를 의미하지는 않는다. 기존 admitted block을 안전하게 마무리하고
GPU/runner가 비며 GF2 데이터와 reference P0가 통과해야 한다. 임의 kill은 없다.
대기·준비 시간도 20시간에 포함된다. 기존 hold owner가 예상과 다르면 보존한다.

신규 **블록**은 16h 전에, 블록 전체와 평가부채를 18h까지 끝낼 수 있을 때만
예약한다. 이미 예약된 pair의 후속 run은 16–18h에 마무리할 수 있다.
18h 이후 optimizer update는 금지하고 20h까지 평가·보존·readback만 수행한다.
미완 학습은 `PARTIAL_TIME_LIMIT`; Exact50K/공식 완료로 위장하지 않는다.

## Reference 전달과 P0

R0는 계획에 핀된 기존 GF2_TA 전체 bundle이다. local manifest를 발견하면
원본을 읽기 전용으로 검증하여 bridge를 만들며, 다른 서버에서는
`work_dir/_g20/<server>/incoming/R0.tar.gz`로 전달할 수 있다.
Teacher weights만 보내거나 τ/qref를 복사해 적는 방식은 금지한다.
R1–R4도 자체 exact50K calibration 이후 `outgoing/R*.tar.gz`로 내보낸다.

TA와 S92001 exact50K의 공통 evaluator 재현도 필요하다. 원본 checkpoint,
config, identity, full-precision report와 dataset provenance가 없으면
`WAIT_REFERENCE`/`WAIT_PARITY`로 남는다. 자료·CLI/API와 evidence 형식은
[REFERENCES.md](REFERENCES.md)에 있다. CPU unit test 통과는 실제 데이터/GPU
parity 통과를 대신하지 않는다.

## 조건부 실험

두 seed의 완전한 local BASE/candidate 결과와 초기 U/A·data stream·source·
reference identity를 확인한 뒤 후보 하나를 immutable decision으로 고정한다.
PROMISING_PAIRED가 없으면 엄격한 JOINT_SINGLE_RETEST만 별도로 허용한다.
유효 후보가 없으면 BASE를 더 반복하지 않는다.

새 seed 확인은 4개 전체를 예약한다. 전이는 원래 X* 서버에만, R* 하나와
Student scalar 하나로 4개 전체를 예약하며 global single-transfer 승인도
필요하다. 기본 순서는 확인→전이이며, 확인이 시작되지 않았고 budget이
충분할 때만 명시적 receipt로 전이를 먼저 둘 수 있다. 다른 서버의 BASE는
local 2×2 control로 사용할 수 없다.

선택된 local screen의 `work_dir/_g20/<server>/screens/<candidate>.json`과
해시가 일치하는 원본 근거 파일을
s1 coordinator에 전달한 뒤 전이 하나만 고정한다.

```bash
python tools/g20_runner.py transfer-select --server s1 --reference R1 \
  --reference-screen /path/reference_paired_screen.json \
  --student-screen /path/student_paired_screen.json
# 아직 확인 block을 시작하지 않은 경우에만 --before-confirmation 허용
```

생성된 `work_dir/_g20/transfer_selection.json`을 원래 Student 축 서버에
전달하고 `transfer --server s3 --global-receipt /path/transfer_selection.json`
으로 등록한다(예시는 X*가 s3인 경우). 러너가 이미 실행 중이면 immutable
incoming receipt를 남기고 다음 block 경계에서 검증·입장한다. 로컬 결과,
reference SHA 또는 네 run 전체의 시간이 맞지 않으면 입장하지 않는다.
서버별 절대 경로가 다르면 근거 파일을 s1의
`work_dir/_g20/incoming/evidence/<해당 파일의 SHA256>`에 전달할 수 있다.
원본 receipt의 경로/해시는 수정하지 않으며 복사된 실제 bytes를 검증한다.

## 배포와 보존

```bash
python tools/g20_runner.py bundle
```

`work_dir/_g20/deployment/g20-overlay.tar.gz`는 source/config/계획과 파일 SHA를
담는다. 데이터·weights·credentials·기존 QG40 clock/ledger는 넣지 않는다.
이미 공통 G20 clock을 정했다면 그 clock만 포함한다. 기존 FH12/PA/metric
파일 및 외부 evaluator는 manifest의 `required_existing_files`와 일치해야 한다.
QG40의 frozen dependency가 다른 서버에서 다르면 무작정 덮어쓰지 말고 검증한다.
이 명령은 배포·git push·학습 시작을 수행하지 않는다.

주요 저장 위치: `work_dir/<run>/meta`, `candidates`, `restart_fullstates`,
`last`, `diagnostics`, `official`; campaign 상태는 `work_dir/_g20/<server>`.
기존 train/eval/Sheet 파일 규약을 유지하면서 campaign을 분리한다.
