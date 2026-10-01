# PANDEP: s2 전용 Shared PLH U-Net

계획: `PANDEP_S2_Shared_PLH_UNet_Experiment_Cases_2026-10-01_v4.md`.
Campaign: `PANDEP_S2_SHARED_PLH_20261001_v4`.

이 디렉터리는 기존 PAN-Crafter 저장소와 **물리적으로 분리된 독립 프로젝트**입니다. 기존 trainer/controller/uploader를 수정하거나 import하지 않습니다. 기존 repo에서 `git pull`하는 것만으로 이 프로젝트가 배포되지는 않습니다. s2에는 이 프로젝트 전체를 별도로 복사해야 합니다. 기존 실험을 자동 중단하지 않으며, GPU가 사용 중이면 `WAIT_RESOURCE`입니다.

## s2에서 시작

먼저 이 프로젝트를 s2의 기존 repo 밖에 복사합니다. `src/`, `tests/`, `vendor_reference/`, `launch.py`, 프로젝트 manifest를 포함하고, 다른 서버의 `work_dir/`, 가중치, credential은 복사하지 않습니다. 기존 s2 credential을 읽기 전용으로 참조합니다. 실제 경로는 다음 두 변수에 지정합니다.

```bash
cd /absolute/path/pan_deploy_shared
PANDEP_LEGACY_ROOT=/absolute/path/PAN-Crafter
PANDEP_CREDENTIALS=/absolute/path/PAN-Crafter/gspread/account.json
python3 launch.py --legacy-root "$PANDEP_LEGACY_ROOT" --credentials "$PANDEP_CREDENTIALS" start --microbatch 48
```

이 한 명령은 독립 Docker 컨테이너에서 `preflight → verify --all-gates → register → run`을 진행하도록 `pan_shared.cli start`를 호출합니다. 별도 수동 setup 순서를 반복할 필요가 없습니다. gate 실패를 우회하거나 기존 작업을 종료하지 않습니다. **컨테이너 ID 출력은 학습 시작 성공 증거가 아닙니다.** 실제 `RUNNING`/`WAIT_RESOURCE`/`BLOCKED_*` 상태와 gate receipt를 확인합니다.

- `${PANDEP_LEGACY_ROOT}/gspread/server.txt`가 실제로 `s2`여야 합니다. s1 등 다른 서버에서는 GPU 실행 전에 거부합니다. 이 파일을 s2로 위조해 우회하지 마십시오.
- source catalog는 `vendor_reference/model_source/ablr2/sensor_sources.json`입니다. 지정 12개 source와 QB raw train/val의 실제 절대경로를 해석해 **모두 read-only mount**합니다. symlink target이 기존 repo 밖에 있어도 같은 절대경로로 별도 read-only mount합니다.
- 고정 image ID는 `sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887`입니다. 로컬에 이 이미지가 없거나 identity가 다르면 차단합니다. 자동 pull, pip install, conda 변경, driver 변경은 없습니다. 이미지 전달은 별도 승인된 운영 절차로 처리합니다.
- 기존 repo 전체도 read-only mount합니다. `PYTHONPATH`는 신규 프로젝트 `src`만 지정합니다. source vendor 파일은 검증 근거이며 legacy trainer를 실행하지 않습니다.
- 컨테이너 hostname은 실제 host hostname으로 고정하고 `--pid host`를 사용합니다. GPU inventory에서 own PID와 기존 host process를 구분합니다.
- 신규 campaign만 고유 lock과 고유 `pandep-shared-s2-<timestamp>-<pid>` 컨테이너를 사용합니다. 기존 컨테이너를 `kill`, `stop`, `rm`하지 않습니다. 실행용 컨테이너도 완료 후 자동 삭제하지 않아 로그를 보존합니다.
- 공식 실행 중 microbatch를 자동 변경하지 않습니다. 최초 검증 전에 필요한 경우 `--microbatch 24` 또는 `12`로 effective batch48을 유지합니다. 이미 등록된 run의 설정 변경은 identity 충돌입니다.

쓰기 없이 명령·mount를 먼저 확인하려면:

```bash
python3 launch.py --legacy-root "$PANDEP_LEGACY_ROOT" --credentials "$PANDEP_CREDENTIALS" --dry-run start --microbatch 48
```

`--dry-run`은 Docker를 실행하지 않고 새 디렉터리도 만들지 않습니다. 실제 image/GPU 검증을 완료했다고 표시하지 않습니다.

## 상태·일시정지·중단·재개

```bash
python3 launch.py --legacy-root "$PANDEP_LEGACY_ROOT" status
python3 launch.py --legacy-root "$PANDEP_LEGACY_ROOT" pause
python3 launch.py --legacy-root "$PANDEP_LEGACY_ROOT" stop
python3 launch.py --legacy-root "$PANDEP_LEGACY_ROOT" --credentials "$PANDEP_CREDENTIALS" resume
python3 launch.py --legacy-root "$PANDEP_LEGACY_ROOT" --credentials "$PANDEP_CREDENTIALS" sync-sheet
```

`status`/`pause`/`stop`는 GPU 없이, network-disabled 일회성 컨테이너에서 신규 campaign control만 처리합니다. raw 데이터가 사라졌더라도 이 제어 명령은 14개 데이터 존재검사를 요구하지 않습니다. `stop`은 `--after-current-update`를 전달합니다. 실제 저장과 정지는 안전 optimizer 경계에서 이루어집니다. 긴 validation/benchmark가 진행 중이면 반환 후 정지할 수 있습니다.

`resume`는 사용자가 명시한 재개이며 새 고유 컨테이너에서 기존 logical run의 fullstate를 이어갑니다. 완료되지 않은 기존 controller가 lock을 보유하면 두 번째 학습을 시작하지 않습니다. start와 resume는 `--until-operator-stop`을 명시적으로 전달합니다. STOP 상태를 crash retry로 자동 해제하지 않습니다. 상태·로그는 아래 campaign에 남습니다.

`resume`에서 microbatch를 생략하면 등록된 값을 보존합니다. **정식 등록·gate PASS 이전 Q00가 OOM으로 실패한 경우에만**, 사용자가 `resume --microbatch 24` 또는 `12`를 명시해 preflight 정책을 변경하고 gate를 다시 실행할 수 있습니다. 이전 Q00·실패 기록은 보존하며, formal run 등록 후에는 이 변경을 거부합니다.

```text
work_dir/PANDEP_S2_SHARED_PLH_20261001_v4/s2/
  campaign_manifest.json / dataset_manifest.json / source_manifest.json
  subset_manifest.json / seed_registry.json
  cases.json / queue.json / control.json / controller.lock
  preflight/ / runs/ / outbox/ / reports/ / exports/
```

모델별 `architecture_manifest.json`은 각 `runs/<Run_ID>/`에 저장합니다.

사용 중인 프로젝트·source·data bytes를 바꾼 뒤 같은 run으로 재개하지 마십시오. 재시작 시 identity를 검증합니다. source/config 변경은 새 계획 또는 attempt로 명시해야 합니다.

## 고정 실험 정의

| Case | 구조 | train subset | 최초 matched-exposure |
| --- | --- | --- | --- |
| C00 | Shared PLH W104/D122 | WV3+GF2+QB 각각100% | global150K |
| C11 | Shared PLH W128/D122 | 세 센서 각각100% | global150K |
| C01 | Shared PLH W104/D122 | 각 센서 고정50% | global150K |
| S01 | Single WV3 W104/D122 | 100% | local50K |
| S02 | Single GF2 W104/D122 | 100% | local50K |
| S03 | Single QB W104/D122 | 100% | local50K |

R01/R02/R03의 master seeds는271001/271002/271003, 최초 logical runs는18개입니다. case 번호는 과거 ABLR2 case 의미와 다릅니다. 모든 case가 **실제 PLH 입력**을 사용하며 Aligner/Teacher/KD/attention/modulation은 생성하지 않습니다.

Shared는 하나의 trunk와 WV311→W/GF2·QB7→W stem, WV38/GF2·QB4 출력 head를 사용합니다. 출력은 `MS_up + residual` 한 번입니다. 동일 repeat의 같은 canonical tensor name/shape는 Shared/Single/Full/Half에서 동일 초기값입니다. 새 stem의 L/H kernel만0으로 시작하며, L/H 입력 tensor 자체는0이 아닙니다.

W104 Shared의 총 parameter는2,121,616, 활성 WV3 경로2,100,808, 활성 GF2/QB 경로2,093,316입니다. W128 Shared 총 parameter는3,201,040입니다. 시트 Params는 **전체 저장 모델** 수이며 active path 값과 구분합니다.

학습은 FP32/TF32 off, effective48, 단일 센서 batch, 균형 shuffled triplets, 정규화 출력의 unclipped mean-L1, AdamW입니다. 50K는 양보·저장 block이고 종료 기한이 아닙니다. shared150K/single50K 이후 LR1e−5로 같은18개 run을 계속 학습합니다. wall-clock 제한/성능 기반 early stop/새 seed 자동생성은 없습니다.

## 데이터와 평가 경계

- WV3/GF2/QB 지정 benchmark만 사용합니다. 사내 데이터와 대체 FR source를 탐색하지 않습니다.
- 12개 source byte SHA, geometry, 개수, 전체 tensor finite/range/zero/saturation, QB msfix와 원본 GT/PAN 보존을 검사합니다. byte SHA 불일치를 내용 유사성으로 승인하지 않습니다.
- Native PAN의 DN에서 float64 Gaussian σ1.98/kernel41/replicate/[2::4,2::4] LP를 생성하고 float32 cache로 저장합니다. Gaussian LP는 공식 평가의 sensor MTF와 별개입니다.
- 정규화는 WV3/QB DN2047, GF2 DN1023의 `2*DN/max−1`. finite zero와 MS/LMS ringing을 보존합니다. fixed Hflip→Vflip→rot90의 동기화 four-view를 사용합니다.
- C01 subset은 source SHA와 sample ID의 고정 hash 순위로 선택하며 세 repeat에서 동일합니다. train probe는 그 공통 subset 안에서 선택합니다.
- Primary는 same exposure의 EXACT입니다. Secondary는 full validation의 normalized mean-L1: shared는 sensor macro mean, single은 해당 sensor mean. HQNR/ERGAS test 값으로 선택하거나 LR/queue를 조정하지 않습니다. 이는 이번 v4의 독립 selector 규약입니다.
- 공식 RR/FR은 센서별20장 전체입니다. FR에는 native PAN/original LMS를 사용하고 shift/masking하지 않습니다. FR GT 지표를 만들지 않습니다. shared의 세 sensor 행은 동일 checkpoint SHA를 사용합니다.
- Source의 train/test 지리적 독립성이 인증됐다는 주장은 하지 않습니다. JQM variant와 metric protocol을 실제 결과 provenance에 명시합니다.

## 시트 충돌 방지

Spreadsheet `1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0`, 기존 **`배포용 모델`**, sheetId **1198707876**만 대상입니다. A1:BL5 헤더·기존 notes/formula를 보존하고 검증된 데이터행에만 씁니다.

`paper`, `ablations`, `WV3-main`, `_records`, 숨김 source, 기존 uploader는 수정하지 않습니다. 신규 탭/clear/duplicate/기존 행 재정렬이 없습니다. 실제 checkpoint×sensor×selector×evaluator Result_ID로 중복 재시도를 막고, readback 일치 후 VERIFIED로 전환합니다. Review 수기값은 uploader가 덮지 않습니다.

학습 원장은 로컬 artifact이며 네트워크 실패는 outbox에 남습니다. Sheet 장애 때문에 loss/seed/precision을 바꾸거나 학습을 재시작하지 않습니다. unknown row/formula spill/다른 writer와 충돌하면 쓰기를 차단합니다. Q00나 가짜 metric은 실제 Sheet 결과행으로 올리지 않습니다.

## 검증과 라이선스

정식 시작에는 G0–G5 receipt가 필요합니다. isolated source parity, three-sensor Q00, source/LP audit, sampling/exact-resume, 공식 metric/Sheet contract를 확인합니다. 로컬 단위 테스트 성공과 **실제 s2 데이터·GPU·시트 gate 성공**은 별개입니다. 기존 trained checkpoint가 없어도 synthetic nonzero state로 parity를 검사하지만 이를 trained checkpoint 검증으로 부르지 않습니다.

검증용 코드는 `tests/`, 고정 수치 근거·SHA·매핑은 `vendor_reference/`에 있습니다. 테스트를 실행한다고 공식 실험을 시작하거나 시트에 가짜 성능을 쓰지 않습니다. host에 torch 등을 설치하지 않고 고정 image 또는 이미 검증된 독립 환경을 사용합니다.

원 PAN-Crafter의 LICENSE에는 **NON-COMMERCIAL RESEARCH AND EDUCATIONAL USE ONLY** notice가 포함되어 있습니다. 이 구현이나 PoC export가 상업 배포 권한을 부여하지 않습니다. DLPan-derived 평가/데이터 audit 코드의 GPL-3.0 출처도 보존하며 inference bundle에 포함하지 않습니다. candidate export를 최종 production 승인이나 laptop/tiled inference 검증 완료로 표현하지 않습니다.
