# QB / GF2 Stage 1 + 5 case 실행 인계

2026-10-26 최신 사용자 결정: **현재 QB/GF2부터 Stage 1 구조 참조와 Stage 2 q/g 참조를 RGB 평균으로 통일하고, PAN warp도 Stage 1·2 모두 Lanczos-4로 통일한다.** 기존 v1 학습의 중간 설정 변경이나 checkpoint 이어학습으로 처리하지 않는다. s1은 QB, s3는 GF2에서 **새 Stage 1 50K + fresh Student 다섯 case**를 실행한다. 완료 즉시 해당 서버의 남은 WV3 TUNE30과 HARD36을 순서대로 재개한다.

이 결정은 이전에 기록했던 “향후 별도 캠페인에서만 변경하고 현재 SENSOR5는 유지” 방침을 대체한다. 기존 WV3의 과학 설정은 계속 보존한다.

- 새 campaign: `PANDA_S13_SENSOR5_RGB_L4_STAGE12_20261026_v2`
- method: `PANDA-SENSOR5-RGB-L4-DIFFUBIL-HARDONLY-v2`
- run prefix: `PSN5R2`
- fresh Stage 1 donor case: `RGB_MATURE_U23`

전체 설정, 데이터 경로 정책, 게시 규칙과 운영 절차는 [SENSOR5 README](../panda_sensor5/README.md)에 있다.

| 서버 | 공동 목표 | Stage 2 case |
|---|---|---|
| s1 / QB | HQNR ≥ .9315, ERGAS ≤ 3.5 | BASE, LR050, LR200, LONG 50K, HARD4 |
| s3 / GF2 | HQNR ≥ .9645, ERGAS ≤ .548 | BASE, LR050, LR200, LONG 60K, HARD4 |

기준은 WV3 유력 설정 scale1 DIFF+UBIL W112/D123 C30 LR1e-4 α.25/β0/γ.5다. LR을 .5×/2×, 추가 Hard 계수를 함께 4×로 넓혀 확인한다. 모든 Student는 같은 seed의 fresh 학습이며 센서별 validation RRE로 선택한 동일 donor를 공유한다. **Aligner에는 전체 MS 4밴드를 입력**하며 Stage 1 정합 참조와 Stage 2 q/g는 GT의 **RGB 단순 평균 `(R+G+B)/3`**을 사용한다. Teacher·Student의 출력 및 reconstruction 손실은 전체 4밴드를 유지한다. Student는 `[PANc, PANc-PAN, I_lrms]`의 6채널 입력이며, 센서별 정규화·공식 Q4/MTF 평가를 적용한다.

Stage 1 native PAN 보정·ε 주입·validation 경로, Stage 2 동결 Teacher·Student 경로를 같은 Lanczos-4 규약으로 맞춘다. validation donor의 RGB 참조와 support 검사도 함께 갱신한다. 기존 v1 donor·calibration은 재사용하지 않으며 새 Stage 1 완료 후 donor 선정과 전체 RR-train calibration을 다시 수행한다.

새 센서 큐는 전부 실행하며 시간 상한을 두지 않는다. WV3의 남은 등록 case도 원래 예산 때문에 다시 skip하지 않도록 명시적 continuation override를 기록한다. 기존 원본 설정·시작 시각·예산·체크포인트는 보존한다.

## v1 보존 및 현재 로컬 전환 상태

기존 campaign `PANDA_S13_SENSOR5_STAGE12_20261026_v1`의 work_dir, checkpoint, 평가 및 게시 행은 그대로 보존한다. v1 prefix `PSN5`와 v2 prefix `PSN5R2`, campaign ID 및 게시 namespace를 구분하므로 v1 결과를 v2 결과로 고쳐 쓰지 않는다.

s1에서는 v1 Stage 1 50K가 완료됐고, 진행 중이던 QB_BASE를 **12,480 update에서 안전 종료**했다. 상태는 `STOPPED_SAFE / STOP_REQUESTED`, run은 `STOPPED`이며 완전한 resume checkpoint SHA 검증을 통과했다. CLI는 캠페인 전체 완료가 아닌 안전 종료를 exit code 1로 반환하므로, code 1만으로 학습 실패라고 판정하지 않는다. migration은 실제 안전 종료 상태·완전한 checkpoint·SHA를 함께 확인한다.

- 기존 컨테이너: `panda-sensor5-stage12-qb-s1-v1` — 종료 상태 유지
- 기존 코드 archive: `work_dir/panda_sensor5_launchers/s1/sensor5_v1_code_before_rgb_l4.tar.gz`
- 중단 checkpoint: `work_dir/panda_sensor5/PANDA_S13_SENSOR5_STAGE12_20261026_v1/s1/QB/runs/PSN5_QB_S1_QB_BASE_NOSOFT_R1_S261018101/attempt_01/models/stop_0012480_1793002631167412449.pt`
- checkpoint SHA256: `7a9bc6bee366e556e94173384d6d9bb04dbc2c32da61a7af502be5f387974ddc`

s1 v2 migration을 적용했고 2026-10-26 17:33 KST에 controller를 시작했다. 새 컨테이너 `panda-sensor5-stage12-qb-s1-v2`에서 `RGB_MATURE_U23` Stage 1이 실제 optimizer update를 진행 중이다. `WAIT_SENSOR5` controller와 `QB-ablations` publisher의 readback 검증도 확인했다. 57개 테스트 및 QB/GF2 실제 GPU Stage 1 학습·checkpoint 복원 검증을 통과했다. 이후 진행 상태는 아래 receipt와 campaign_state로 확인한다. s3에서의 실제 migration·재시작은 별도 에이전트가 이 commit을 반영해 수행해야 한다.

QB 낮은 HQNR의 진단은 [QB 구현·성능 분석](../results_log/2026-10-26_qb-low-hqnr-audit.md)에 기록한다. 진단상 FR 정합 proxy는 Aligner 보정 후 개선되므로 “Aligner 구현이 잘못됐다”는 결론은 아직 성립하지 않는다. 원본 PAN을 사용하는 공식 지표와 보정 좌표의 관계, RR/FR 보정량 차이를 구분해서 검토한다. **RGB/Lanczos 통일이 낮은 HQNR을 해결한다는 보장은 없다.** scale 0, 임의 shift cap, 보정 PAN으로 공식 평가 참조 교체를 생산 실험에 추가하지 않는다.

## s3 코드 갱신 전 안전 종료와 v2 인계

현재 trainer와 watcher는 소스를 hash로 고정한다. **실행 중인 checkout에서 먼저 git pull/checkout/cherry-pick하면 안 된다.** 실제 remote와 전달된 commit을 지정해 fetch만 수행하고, 그 commit의 standalone 종료 도구를 `/tmp`로 읽어 실행한 뒤 checkout을 갱신한다. 아래 `<...>`는 s3 저장소의 실제 값으로 바꾼다. remote 이름이나 branch를 가정하지 않는다.

```bash
set -euo pipefail
sensor5_remote='<실제 remote 이름>'
sensor5_commit='<전달받은 v2 commit SHA>'
git fetch "$sensor5_remote"
git show "${sensor5_commit}:scripts/stop_panda_sensor5_before_update.py" > /tmp/pan-stop-sensor5-before-update.py
python /tmp/pan-stop-sensor5-before-update.py --root "$PWD" --server s3
```

종료 도구가 `SAFE_TO_UPDATE_CHECKOUT`을 출력한 후에만 기존 저장소 운영 방식에 맞춰 해당 commit을 반영한다. 예를 들어 별도 서버 branch로 전달받는 방식이면 다음처럼 cherry-pick한다. checkout 또는 fast-forward로 동기화하는 환경은 그 방식을 사용하며, 아래 명령을 중복 실행하지 않는다.

```bash
git cherry-pick "$sensor5_commit"
```

종료 도구는 해당 root/server의 기존 SENSOR5 queue·publisher와 v1 sensor trainer만 식별해 종료한다. trainer에는 SIGTERM만 보내고 완전한 checkpoint 저장을 기다리며 SIGKILL로 강제 종료하지 않는다. 코드 archive는 `work_dir/panda_sensor5_launchers/s3/legacy_source_before_rgb_l4_<timestamp>.tar.gz`에 남긴다. 기존 결과와 WV3 학습 컨테이너는 변경하지 않는다. s3가 이미 `RESUME_TUNE30` 또는 `RESUME_HARD36` 단계로 넘어갔다면 도구가 중단하므로, 현재 WV3 상태를 먼저 점검한다.

코드 반영 후에는 **torch와 기존 uploader 의존성이 설치된 학습 Python 환경**에서 실행한다.

```bash
export PYTHONDONTWRITEBYTECODE=1
python -m panda_sensor5.migration --server s3
python -m panda_sensor5 check --server s3
python -m panda_sensor5 smoke
python -m panda_sensor5.migration --server s3 --apply
python -m panda_sensor5.queue --server s3 --enqueue
```

첫 migration 명령은 dry-run이다. `--apply`는 v1 안전 종료 checkpoint와 기존 WV3 완료 증거를 확인하고, 기존 chain을 백업한 뒤 `WAIT_SENSOR5 → START_SENSOR5`로 바꾼다. 기존 `initial_cutover.json` 및 WV3 finish-current receipts를 다시 만들지 않는다. migration은 반복 실행 가능하며 새 Stage 1의 가중치·optimizer·donor·calibration을 v1에서 이어받지 않는다.

이 명령은 fresh GF2 Stage 1, GF2 다섯 Student, 남은 WV3 재개까지 연결한다. 새 GF2 학습을 먼저 별도로 실행하거나 기존 HARD36 watcher를 다시 enqueue하지 않는다. 새 controller가 순서를 소유한다. `gspread/server.txt=s3`, 기존 자격정보·데이터·Docker 이미지·WV3 work_dir가 필요하며 Git은 이를 운반하지 않는다. v1 센서 학습을 아직 시작하지 않은 서버는 기존 `CUTOVER` 단계를 보존하므로 앞서 활성화돼 있던 WV3 run만 완료한 후 새 센서 캠페인으로 넘어간다.

아직 최초 `CUTOVER`를 완료하지 않은 서버는 실행 직전에 기존 HARD36이 본 학습 컨테이너 생성 전의 cost/preflight 준비 단계인지 확인한다. 준비 subprocess만 실행 중인 경우에는 이를 중복 실행하거나 강제로 종료하지 말고 준비가 끝난 뒤 enqueue한다. 중간 실패한 준비 상태는 해당 로그부터 확인한다. 최초 `CUTOVER` 단계의 TUNE30/HARD36 본 학습 또는 정상 종료 상태는 기존 큐 제어가 처리한다. v1 센서 완료 후 이미 WV3를 재개한 상태는 위 migration의 자동 처리 범위가 아니다.

## 확인할 경로와 게시 규칙

- 전체 순서: `work_dir/panda_sensor5_launchers/s3/status.json`
- 전환 receipt: `work_dir/panda_sensor5_launchers/s3/sensor5_rgb_l4_v2_migration.json`
- 원래 chain 백업: `work_dir/panda_sensor5_launchers/s3/chain_before_rgb_l4_v2.json`
- 학습: `work_dir/panda_sensor5/PANDA_S13_SENSOR5_RGB_L4_STAGE12_20261026_v2/s3/GF2/campaign_state.json`
- 컨테이너: `panda-sensor5-stage12-gf2-s3-v2`
- 게시: 같은 캠페인 디렉터리의 `sheet_publish_panda_sensor5/last_poll.json`

s1은 위 경로의 서버 표시 `s3`를 `s1`, 센서 표시 `GF2`를 `QB`로 바꾸며 컨테이너는 `panda-sensor5-stage12-qb-s1-v2`다. 상태가 `BLOCKED`이면 launcher 오류와 로그를 확인하고, admission이나 checkpoint를 삭제해 강제로 통과시키지 않는다.

Sheet는 `GF2-ablations`, `QB-ablations` 각각에 센서 목표의 공동 통과 step 전부와 run별 대표를 유지한다. 대표는 HQNR 목표 충족 중 최소 ERGAS, 없으면 최대 HQNR다. Stage 1은 validation RRE 선택 모델을 게시한다. run 이름과 메모에 RGB 참조·Teacher/Student/ε의 L4 통일·scale1·DIFF/UBIL·하이퍼파라미터를 함께 남긴다. 기존 v1 게시 행은 별도 campaign 기록으로 보존한다.

## WV3 보존 및 해석 범위

남은 WV3 TUNE30/HARD36은 취소하거나 새 RGB/L4 설정으로 바꾸지 않는다. 센서 v2 전체 완료 뒤 원래 checkpoint·설정·donor·평가 규약으로 이어간다. 이미 기록된 제어 어댑터 `panda_sensor5/wv3.py`의 SHA256은 `daf6c5d3d57e74166b9ce4cc2ffc02ebb16332158ae1039d3582298457c13ca7`로 유지하며, HARD36 continuation bridge도 보존한다.

향후 신규 캠페인도 RGB 구조 참조와 Lanczos-4 통일 규약을 사용한다. 이는 참조 정의와 warp 경로를 일관되게 하는 결정이며 성능 개선의 확증은 아니다. v1과 v2는 구조 참조와 Teacher/ε warp가 동시에 바뀌므로 차이를 RGB 단독 효과라고 해석하지 않는다. 과거 mean8 비교 근거는 [Stage 1 mean8 비교](../results_log/2026-10-26_stage1-mean8-target-audit.md)에 보존한다.
