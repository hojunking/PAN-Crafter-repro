# B01 결과의 현행 Google Sheets 등록 — 코드 변경·백필 실행 지시서

작성일: **2026-09-30**  
작업 ID: `PANDA_RB_B01_SHEETS_BRIDGE_v1`  
대상: `pan-cvpr27`, PANDA rebuttal **B01 = RB01 + RB02**, 원 실행 서버 **s1·s3·s5**  
산출물: 독립 uploader 구현, 기존 완료 결과의 백필, 자동 표시 연결, 실제 readback receipt  
문서 상태: **live Sheet 구조·수식과 저장소 코드를 확인해 작성한 실행 지시서. 구현·업로드 완료 보고가 아니다.**

## 1. 작업 목표와 금지 범위

서버에 이미 생성된 B01의 학습·native·stress 결과를 검증해 현재 정리된 Sheet 규칙으로 등록한다. 새 학습, checkpoint 재선택, metric 재계산 없이 **증거 읽기 → schema 변환 → 업로드 → 원시값 및 표시 결과 readback**만 수행한다. 실험 완료 여부는 추정하지 말고 파일별로 판정한다.

**제출본을 기준으로 유지한다.** `paper`, `ablations`, `유의미한결과`의 기존 값·수식·note·표시 형식·모델 표기·순서를 변경하지 않는다. Table 3 수정, 메인 값 교체, reproduced 표시 추가는 이 작업에 포함하지 않는다. 기존 수치의 재선별·교정도 하지 않는다.

기존 학습 controller, queue, watchdog, frozen Docker, model/optimizer/RNG/checkpoint, native/stress 원본 JSON·CSV·NPY를 수정하지 않는다. 업로드 실패 때문에 training/evaluation을 재실행하지 않는다. B02 이상의 실험을 새로 시작하지 않는다.

이 문서에서 **2절은 실제 확인한 현행 규칙**, **3절 이후의 새 탭·schema·CLI는 구현할 확장 설계**다. 새 탭이 이미 생성되었거나 uploader가 현재 존재한다고 해석하지 않는다.

## 2. 실제 확인한 현행 Sheet 규칙

### 2.1 대상과 데이터 흐름

```text
spreadsheet_id: 1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0
spreadsheet_title: pan-cvpr27
locale: ko_KR
timeZone: Asia/Tokyo

기존 숨김 원본 탭 33개
    ↓ _records!A2의 ARRAYFORMULA / LET / parse / VSTACK
숨김 통합 관측표 _records!A:BL — 64열
    ↓ Dataset / Bucket / Approach에 따른 FILTER 및 SORT
WV3-main / QB-main / GF2-main / archive / PC-Repro
```

2026-09-30 live 조회에서 **46개 탭**을 확인했다. 주요 sheet ID와 성격은 다음과 같다. 실행 시 ID와 제목을 다시 대조한다.

| 탭 | sheetId | 현재 성격 / 쓰기 정책 |
|---|---:|---|
| paper | 1923997092 | 제출 관련 수동 표. **수정 금지** |
| ablations | 1697831403 | 기존 수동 요약. **수정 금지** |
| 유의미한결과 | 410652761 | 기존 수동 요약. **수정 금지** |
| _records | 260928001 | 숨김 12000×64. A1:BL1 헤더, **A2가 통합 배열 수식** |
| _case_labels | 260928002 | 숨김 기존 C00–C17/Teacher 설명표. 이번 작업에서는 수정하지 않음 |
| WV3-main | 260928010 | 3000×64 표시 탭. **A5는 헤더 수식, A6는 본문 배열 수식** |
| QB-main / GF2-main | 260928011 / 260928012 | 동일 구조. B01 때문에 수정할 필요 없음 |
| WV3-archive / QB-archive / GF2-archive / WV2-archive | 260928020 / 260928021 / 260928022 / 260928023 | 기존 archive 표시. 수정 금지 |
| PC-Repro | 260928030 | 기존 reproduction 표시. 수정 금지 |

### 2.2 직접 append하면 안 되는 곳

`_records`는 값만 모아둔 database 탭이 아니다. **A2 하나의 수식이 하단 64열로 전개된다.** `WV3-main` 역시 A6의 수식이 그룹 제목과 관측행을 만든다.

따라서 다음 작업은 금지한다.

```text
_records.append_rows(...)
WV3-main.append_rows(...)
_records 전체 clear / 값으로 다시 채우기
WV3-main A6 아래에 결과값 직접 쓰기
과거 WV3-s1/s3/s5 원본에 B01 행을 구 schema인 것처럼 끼워 넣기
```

배열 전개 범위에 값이 있으면 수식이 막히거나 기존 관측이 사라질 수 있다. `_records`에 연결되지 않은 새 탭만 만들어도 현재 main에는 표시되지 않는다.

### 2.3 현행 parser와 grouping

`_records!A2`는 **기존 탭 목록을 명시적으로 나열**한다. 자동으로 모든 새 worksheet를 발견하지 않는다. `Run`, `run_id`, selection prefix, checkpoint SHA, metric 헤더의 정규식으로 원본을 읽고, selection별 관측행으로 펼친다. B01 전용 source와 RB01 family 분기는 현재 목록에 없다.

분류는 score가 아니라 method family 기준이다. 현재 대표적인 `Approach`는 `01 | FULL…`, `02 | Component…`, `03 | MAIN-A…`, `04 | GFB20/GFP40…`, `05 | Teacher…`다.

`WV3-main!A6`의 조건은 **Dataset=WV3, Bucket=Main**이다. Approach별 그룹을 만든 뒤 `01` 그룹은 selector/HQNR, 그 밖의 그룹은 다음 순서로 정렬한다.

```text
Sort_case(63) → Repeat(32) → Server(2) → Run_ID(30) → Selector(43)
```

`WV3-main!A5`는 `=ARRAYFORMULA('_records'!$A$1:$BL$1)`이다. 1–5행, 처음 3열을 고정하고 기술 metadata는 오른쪽에 유지하는 기존 표시 원칙을 따른다. 새 B01 접근 번호는 **06**으로 제안한다.

**관측행과 독립 run은 다르다.** EXACT와 validation-selected가 같은 SHA라도 selector 두 행은 유지할 수 있지만, 독립 학습 수는 1이다. seed, SHA, source를 설명 문자열에서 제거하지 말고 별도 열에 보존한다.

### 2.4 기존 B01 코드의 상태

확인한 저장소 snapshot은 `af1f69e79835268157ccd6cb183d222c280fd31a`다. 서버 실제 수치 runtime을 이 commit으로 교체하라는 뜻이 아니다.

- `tools/panda_rb_runner.py`: `plan/status/run/report/package`만 지원하고 **Sheet를 upload하지 않는다**.
- `panda_rb/reporting.py`: 로컬 요약·패키지 생성. `sheets_uploaded=False`는 고정 반환값이며 업로드 시도/실패 이력이 아니다.
- `panda_rb/common.py`: `panda_rb/*.py`, `tools/panda_rb_*.py` 등을 수치 source identity에 포함한다.

**기존 `panda_rb` 안에 upload 함수를 추가하는 방식은 피한다.** 완료된 frozen source의 identity를 바꿔 기존 증거 검증이나 재개를 깨뜨릴 수 있다. 별도 bridge가 원 보고서를 읽게 한다.

## 3. 구현할 연결 구조

```text
원 서버 s1 / s3 / s5의 검증된 B01 결과
        ↓ 읽기 전용 보고서·증거 manifest 수집
독립 reporting bridge / durable spool
        ↓ 단일 조정 writer(s1 권장)
┌─ _rb01_s1 / _rb01_s3 / _rb01_s5   [신규 숨김, 각 64열]
│       ↓ 기존 _records의 마지막 union에만 추가
│     _records → WV3-main의 06 | Rebuttal B01 그룹
│
├─ _rb02_points                     [신규 숨김, stress 전용 schema]
│       ↓ 표시 수식
│     RB02-curves                   [신규 공개, 49-point 곡선 원자료]
│
└─ _rb_b01_status                   [신규 숨김, 24개 계획 run의 실제 상태]
        ↓ native·stress·status를 연결
      RB-B01                       [신규 공개, 완료율·6-seed·paired 요약]
```

새 탭 이름이 이미 사용 중이면 제목만 보고 재활용하지 않는다. 소유 schema/version/header/campaign을 검증해 일치할 때만 사용하고, 다르면 충돌로 중단한다. 기존 탭을 rename/delete/hide 변경하지 않는다. 새 공개 탭은 기존 탭의 상대 순서를 유지하며 뒤쪽에 추가한다.

**Native 결과만 기존 64열 `_records`에 넣는다.** RB02의 fixed192 ROI ERGAS는 native RR의 20:-21 지표와 다른 조건이다. shifted 결과나 A_ZERO 지표를 같은 native metric 열에 섞지 않는다. RB02는 별도 표에서 비교한다.

## 4. 코드 변경 범위와 두 단계 실행

### 4.1 신규 모듈 — 제안 파일명

```text
reporting_bridge/
    rb_b01_contract.py       # live 64열·고정 ID·JSON schema·숫자 변환
    rb_b01_collect.py        # sealed 결과와 검증 receipt 읽기; GPU 작업 없음
    rb_b01_sheets.py         # snapshot/setup/upsert/readback/rollback
    rb_b01_summary.py        # Student-seed 단위 통계·paired 차이
    tests/...
tools/rb_b01_sheets.py       # 신규 명시적 CLI
```

이 이름들은 **구현 요구사항**이다. 기존에 존재하는 명령으로 보고하지 않는다. 실제 프로젝트 전체 source-identity 규칙을 한 번 더 확인하고, bridge 파일을 기존 frozen 수치 checksum 대상 밖에 둔다. `panda_rb/*`, `tools/panda_rb_*`, 기존 gspread uploader/cleanup 파일을 수정하지 않아도 되는 독립 구조를 우선한다.

기존 `pcrepro/upload.py`의 **RAW 쓰기, spool, idempotent upsert, 숫자 readback** 패턴은 참고할 수 있다. 단, PCREPRO의 row schema·selection grid·recipe validator·자동 탭 생성 함수를 B01에 그대로 적용하지 않는다. 인증도 import 시 접근하지 말고 명시적 live 단계에서만 사용한다.

### 4.2 단계 1 — 수집·검증·dry-run

세 서버는 자기 결과만 원 frozen 환경에서 검증하고, 조정 writer에 보고서/소형 증거 package를 전달한다. 기존 학습을 중단하거나 원 runtime에 git pull하지 않는다.

각 서버에서 **이미 존재하는** 명령은 다음이다. 실제 원 프로젝트 루트·container mount를 확인해 사용한다. `SERVER`만 해당 서버로 바꾼다.

```bash
SERVER=s1   # s3 / s5에서는 각 서버명
python3 tools/panda_rb_runner.py status --server "$SERVER"
python3 tools/panda_rb_runner.py report --server "$SERVER" \
  --output "work_dir/B01_${SERVER}_report_readback.json"
```

`report`는 기존 `summarize(..., verify=True)`를 실행하며 원 평가의 payload/file hash를 검사한다. 이 단계에서 원 reader가 요구하는 파일이 없으면 검증 실패를 기록한다. `status`의 파일 존재 표시만으로 성공 처리하지 않는다.

추가로 선택 manifest, checkpoint identity 및 실제 weight SHA, config/case/seed 일치도 읽기 전용으로 대조한다. `report`가 모든 source/checkpoint 조건을 자동 재검사한다고 가정하지 않는다. 이 추가 검사는 hash/JSON 대조이며 모델 forward나 optimizer 생성이 아니다.

**조정 writer는 s1 한 곳**으로 고정하는 것을 권장한다. s3/s5는 collect만 하고 공유 `_records!A2`나 요약 탭을 동시에 변경하지 않는다. 로컬 `flock`만으로 세 서버의 공유 Sheet 경쟁을 막았다고 주장하지 않는다. 기존 다른 Sheet writer와 구조 변경 시간이 겹치지 않는지 확인한다.

### 4.3 단계 2 — setup → 백필 → readback

1. live headers/수식/보호 범위 snapshot과 원정밀도 baseline을 저장한다.
2. 신규 source/status 탭을 준비한다. 모든 source 탭은 먼저 빈 상태로 존재해야 한다.
3. 검토 가능한 dry-run diff를 만든 후 `_records!A2`에 최소 union patch를 1회 적용한다.
4. source에 verified native/stress를 idempotent upsert한다.
5. source 원시값, `_records`, `WV3-main`, `RB02-curves`, `RB-B01`을 다시 읽어 검증한다.
6. receipt와 누락/실패 목록을 반환한다. 재실행은 검증된 동일 payload에서 **0개 신규 관측**이어야 한다.

완료된 서버부터 등록할 수 있다. 단, 2/6 또는 4/6 결과를 6/6으로 표시하지 않는다. 모든 증거가 없다는 이유로 이미 검증된 결과까지 보류할 필요는 없다.

## 5. 읽을 원본과 완료 조건

### 5.1 실제 campaign과 고정 registry

```text
CAMPAIGN_ID = PANDA_REBUTTAL_B01_WV3_S135_20260928_v1
ROOT/work_dir/_panda_rb/20260928/B01/
    control/<server>/status.json
    control/<server>/summary.json
    RB01/<server>/R<repeat>/<case_id>/
        meta/config.resolved.yaml
        meta/training_status.json
        meta/controller_failure.json          # 존재 시
        checkpoints/selection_manifest.json
        checkpoints/exact50000/identity.json
        checkpoints/exact50000/model.safetensors
        checkpoints/val_selected/...          # alias/symlink 가능
        native/metrics.json
        native/per_scene.csv
        stress/A_ON/completion.json
        stress/A_ZERO_INFERENCE_ONLY/completion.json
        stress/<mode>/curve_summary.json
        stress/<mode>/per_scene.csv
```

경로는 원 코드 규약이다. 실제 repo root·파일 존재는 서버에서 확인한다. `native/metrics.json`과 stress completion의 `payload_sha256`, `file_hashes`, case/source/data identity를 유지한다. 원 report의 `sheets_uploaded` 값을 True로 덮어쓰지 않는다. 업로드 기록은 **별도 `work_dir/_rb_sheet_upload/`**에 남긴다.

| 서버 | Repeat 1 seed | Repeat 2 seed | 학습 case |
|---|---:|---:|---|
| s1 | 9281101 | 9281102 | QFULL / QMEAN / QSHUF / QESUR 전부 |
| s3 | 9281301 | 9281302 | 동일 |
| s5 | 9281501 | 9281502 | 동일 |

Case별로 Student 6개, 서버별로 Student 8개다. 다른 seed, 다른 F1, 다른 dataset 또는 다른 campaign을 부족한 슬롯에 대체하지 않는다. report와 registry가 다르면 별도 상태로 차단한다.

### 5.2 읽을 schema

| 파일 | 확인된 schema / 용도 |
|---|---|
| native/metrics.json | `PANDA_RB01_NATIVE_v1`; selections의 native RR/FR·raw identity |
| stress/<mode>/completion.json | `PANDA_RB02_CURVE_v1`; exact50K의 49점·980관측 |
| report JSON | `PANDA_RB_B01_LOCAL_REPORT_v1`; 로컬 완료 수·실패·summary |

검증된 native report 하나에는 `EXACT_50000`, `RR_VAL_ERGAS_MIN` 두 selection이 있다. 같은 SHA이면 secondary의 `alias_of=EXACT_50000`를 유지한다. 이때 **관측 2행, Student 1개**다.

Stress는 `source_selection=EXACT_50000`, `update=50000`, 동일 native exact SHA, modes `A_ON` / `A_ZERO_INFERENCE_ONLY`, 49개 등록 shift, scene 0–19를 검사한다. val-selected checkpoint로 stress를 바꾸지 않는다.

### 5.3 모두 수집된 경우의 기대량 — 실제 성공을 의미하지 않음

| 항목 | 서버당 | 전체 |
|---|---:|---:|
| 등록 Student | 8 | **24** |
| Native selection 관측행 | 16 | **48** |
| Stress curve report | 16 | **48** |
| Stress shift-point 행 | 784 | **2352** |
| 원 per-scene stress 관측 | 15680 | **47040** |

47040개 per-scene 관측을 처음부터 Sheet에 전부 펼치지 않는다. 서버 원 CSV·검증 package에 보존하고, Sheet에는 2352개 shift-point와 파일 출처를 제공한다. 원시 영상·모델 weight·credential을 Sheet 셀에 넣지 않는다.

`STOP_FOR_REVIEW`만으로 성공 판정하지 않는다. 학습 실패, 기술 오류, 빠진 native/curve, stress의 nonfinite/invalid geometry를 각각 기록한다. Stress의 `complete=True`는 전체 grid 처리가 끝났다는 뜻일 수 있으며 **모든 관측이 유효하다는 뜻이 아니다**.

## 6. RB01 native 원본 schema와 고정 행

신규 `_rb01_s1`, `_rb01_s3`, `_rb01_s5`의 **A1:BL1을 현행 `_records!A1:BL1`과 정확히 일치**시킨다. 64열 뒤에 추가 열을 붙이지 않는다. richer provenance는 BI 및 외부 manifest로 연결한다.

이번 bridge는 이미 64열로 정규화한 source를 공급하므로, 기존 레거시 `parse(...)`의 정규식을 바꾸지 않는다. `_case_labels`의 고정 C00–C17 lookup도 건드리지 않고 새 case 설명을 A열에 직접 넣는다.

| 열 | 현행 헤더 | B01 입력 규칙 |
|---|---|---|
| A | Experiment | 예: `QFULL \| original q \| R1`. case 설명과 repeat만 표시하며 성능으로 이름을 바꾸지 않는다. |
| B | Server | 등록된 원 학습 서버 `s1`, `s3`, `s5`. 업로드 조정 서버와 구분한다. |
| C | Selection@Step | `EXACT_50000@50000` 또는 `RR_VAL_ERGAS_MIN@<실제 update>`. |
| D | HQNR↑ | 해당 selection의 `fr.hqnr` 원정밀도. |
| E | D_s↓ | 해당 selection의 `fr.d_s`. `ds` 별칭은 실제 source에서 확인한 경우에만 허용한다. |
| F | D_lambda↓ | 해당 selection의 `fr.d_lambda`. `dlambda` 별칭 사용 시 값 충돌을 검사한다. |
| G | JQM↑ | `supplemental_fr.jqm`. 없으면 빈칸. JQM variant를 BC에 함께 보존한다. |
| H | ERGAS↓ | 해당 selection의 `rr.ergas`. |
| I | SCC↑ | 해당 selection의 `rr.scc`. |
| J | SAM↓ | 해당 selection의 `rr.sam`. |
| K | PSNR↑ | 해당 selection의 `rr.psnr`. |
| L | SSIM↑ | 해당 selection의 `rr.ssim`. |
| M | Q4/Q8↑ | WV3이므로 `rr.q8`. `q2n` 별칭은 동일 정의·일치 확인 시만 사용. Q4를 가져오지 않는다. |
| N | RMSE↓ | `supplemental_rr.rmse`. 없으면 빈칸. |
| O | CC↑ | `supplemental_rr.cc`. 없으면 빈칸. |
| P | Infer(ms) | 이번 run·해당 프로파일 조건의 실측만. B01에 없으면 빈칸; native 전체 elapsed를 inference latency로 쓰지 않는다. |
| Q | Mem(MB) | 원 실측과 단위를 확인할 수 있을 때만. bytes/2^20 사용 시 MiB임을 BD에 명시한다. 타 run에서 차용 금지. |
| R | Params(M) | 실제 저장된 profile/parameter count가 있으면 total/1e6. 없으면 빈칸. |
| S | FLOPs(G) | 실측값과 계산 규약이 확인될 때만. 없으면 빈칸. MACs·FLOPs·partial count 혼합 금지. |
| T | Train(h) | 원 run의 optimizer 학습 seconds/3600, 확인된 필드만 사용. 두 selection에 반복 표시되더라도 합산하지 않는다. |
| U | Eval(h) | `native.elapsed_seconds/3600`. 두 selection 평가의 run-level 총시간이면 BD에 명시하고 selection별 합산 금지. |
| V | Wall(h) | 원 run의 실제 시작·종료·scope가 확인될 때만. 업로드 시간으로 대체하지 않는다. |
| W | HQNR(V64)↑ | B01에서는 측정하지 않았으므로 빈칸. |
| X | Signed D_s | 별도의 signed 통계가 source에 있을 때만. D_s를 복사하지 않는다. |
| Y | Positive D_s fraction | source에 측정된 경우만. B01 기본 보고서에 없으면 빈칸. |
| Z | Dataset | `WV3`. |
| AA | Bucket | `Main`. 기존 method-family 분류에 맞춘 제안이며, 최고 성능 또는 제출 대표라는 뜻이 아니다. |
| AB | Approach | `06 \| Rebuttal B01 \| RB01 q reliability`로 고정. 기존 01–05 접근과 분리한다. |
| AC | Campaign_ID | `PANDA_REBUTTAL_B01_WV3_S135_20260928_v1`. |
| AD | Run_ID | registry의 원 `RB01_WV3_...` ID를 그대로 사용. |
| AE | Case_ID | `QFULL`, `QMEAN`, `QSHUF`, `QESUR`. |
| AF | Repeat | 숫자 1 또는 2. P01/P02 또는 cycle로 재해석하지 않는다. |
| AG | Seed | registry의 원 seed 숫자. 아래 6-seed 표와 대조한다. |
| AH | Attempt | source에 별도 논리적 실험 attempt가 있을 때만. resume/업로드 재시도 횟수는 넣지 않는다. |
| AI | Role | `S`. |
| AJ | Model / inputs | `PLH / W104 / D[1,2,2] / A_ON` 등 원 resolved config를 확인한 표시. |
| AK | Teacher / reference | `F1 exact50K`와 짧은 SHA 표시; 전체 Teacher SHA는 BI의 provenance에 보존. |
| AL | Parent_run | fresh Student이므로 빈칸. F1 Teacher를 Student parent로 오기입하지 않는다. |
| AM | Updates | 원 학습 완료 update 50000. |
| AN | Lifetime_updates | fresh50K임이 확인된 경우 50000. val-selected step과 혼동하지 않는다. |
| AO | Selected_step | 해당 selection의 실제 `update`. |
| AP | Checkpoint_SHA | 해당 selection의 전체 64자 SHA256. |
| AQ | Selector | `EXACT_50000` 또는 `RR_VAL_ERGAS_MIN` 원 label. |
| AR | q_ref | 실제 F1 binding/resolved config에 보존된 q_ref. 문서 숫자를 default로 채우지 않는다. |
| AS | tau_R | 실제 calibration의 tau_R. source 경로/alias를 검증한 값만. |
| AT | alpha | 검증된 resolved config의 값. |
| AU | beta | 검증된 resolved config의 값. |
| AV | lambda_E | 검증된 resolved config의 값. |
| AW | U_peak_lr | 검증된 원 U peak LR. |
| AX | A_peak_lr | 검증된 원 A peak LR. |
| AY | Status | 성공한 native evidence이면 `NATIVE_EVAL_COMPLETE`. RB02·전체 campaign 완료와 구분한다. |
| AZ | Readback | 최초 `UPLOAD_PENDING`, 원시값 대조 후 `READBACK_VERIFIED`. evidence의 complete를 그대로 복사하지 않는다. |
| BA | Eval_scope | `RB01_NATIVE_RR20_FR20;RR20:-21;FR_FULL512;A_ON`. |
| BB | Test_aware | B01 selection이 고정 endpoint 또는 validation-only임을 확인한 경우 boolean false. 테스트셋의 과거 개발 사용과는 별개이며 BI에 설명한다. |
| BC | JQM_variant | 해당 selection의 `jqm_variant` 원문. SRF-substitute를 SIPSA-equivalent로 바꾸지 않는다. |
| BD | Cost_scope | 비용 필드의 실제 단위·범위·selection alias 비용 중복 여부와 `NOT_PROFILED` 등을 명시. |
| BE | Source_sheet | 원 서버에 대응하는 `_rb01_s1`, `_rb01_s3`, `_rb01_s5`. |
| BF | Source_row | 해당 신규 원본 탭의 실제 고정 행 번호. |
| BG | Source_URL | 새 원본 탭의 실제 gid와 해당 행을 가리키는 Google Sheets URL. 로컬 파일 경로와 구분한다. |
| BH | Result_ID | 이 문서의 결정적 native observation ID. checkpoint/selection 구분을 보존한다. |
| BI | Original description / notes | 간결한 설명 + compact JSON provenance. alias_of, input/report/source/runtime/evaluator/Teacher/map SHA, 원 파일 상대경로, payload SHA, 원 완료 시각을 포함. |
| BJ | Review | 검토 필요 사유만 기록. 성능 저하를 오류로 표시하지 않는다. alias는 `SELECTION_ALIAS` 표기 가능. |
| BK | Sort_case | QFULL=1, QMEAN=2, QSHUF=3, QESUR=4. |
| BL | Date | 원 native report의 `completed_at_utc`를 보존. 업로드 시각은 별도 receipt에 기록. |

### 6.1 결정적 observation ID와 source row

```text
Result_ID = RB01|<campaign_id>|<run_id>|<selection_id>|<checkpoint_sha256>|NATIVE_RR20_FR20_v1
logical_key = (<campaign_id>, <run_id>, <selection_id>, NATIVE_RR20_FR20_v1)
```

ID에 업로드 시각·재시도 횟수·Sheet 행 번호를 넣지 않는다. 같은 logical_key인데 checkpoint/evaluator/metric payload가 다르면 **충돌**이다. 다른 SHA를 새 성공 반복으로 조용히 추가하지 않는다. 기존 Result_ID를 수정하거나 legacy 방식의 ID를 일괄 변경하지 않는다.

원본 행은 repeat → case → selection 순서의 고정 슬롯을 권장한다.

```text
case_index: QFULL=0, QMEAN=1, QSHUF=2, QESUR=3
selection_index: EXACT_50000=0, RR_VAL_ERGAS_MIN=1
source_row = 2 + (repeat-1)*8 + case_index*2 + selection_index
```

서버별 row2–17이다. 아직 verified evidence가 없는 슬롯은 관측행을 만들지 않고 status 탭에서만 설명한다. 원본 탭을 성능순으로 sort하지 않는다. 표시 순서는 main 수식이 처리한다. 같은 SHA alias도 두 고정 슬롯을 유지한다.

### 6.2 provenance와 숫자

BI 또는 row note에 다음을 보존하고, 전체 내용이 긴 경우 별도 manifest path/hash로 연결한다.

```text
schema_version, alias_of, native_payload_sha256, native_file_sha256,
selection_manifest_sha256, config_sha256, binding_common_sha256,
data_content_identity_sha256, numerical_source_sha256, evaluator_sha256,
teacher_sha256, q_weight/map identity, source_file_relative_path,
rr_scene_count, fr_scene_count, native_coordinate_frame,
cost_scope, runtime/device, source_completed_at_utc, verification_receipt_sha256
```

JSON 파일의 byte SHA와 `object_sha`/payload SHA는 서로 다른 규약이다. 같은 `sha` 이름으로 섞지 않는다. null/미측정은 빈칸, 실제 0은 숫자 0, booleans는 실제 bool로 기록한다. Metric은 float 원정밀도로 쓰고 표시 서식만 4자리로 제한한다. 숫자처럼 보이는 문자열을 일부러 만들지 않는다.

문자열은 **RAW/stringValue**로 써서 `=`, `+`, `-`, `@` 등 시작 문자가 수식으로 실행되지 않게 한다. 실제 설치할 수식만 formulaValue/명시적 formula 경로를 사용한다. SHA/Run_ID는 텍스트로 보존한다.

## 7. `_records!A2`의 최소 연결 patch

### 7.1 변경할 부분

현재 A2 수식의 **legacy `parse` 정의와 `allrows=VSTACK(parse(...))`는 그대로 둔다.** 마지막 반환식에서 새 64열 source만 추가한다. 전체 거대 수식을 기억·하드코딩한 복사본으로 덮어쓰지 않는다.

현재 마지막 반환식은 다음이다.

```text
FILTER(allrows,CHOOSECOLS(allrows,1)<>"")
```

아래는 그 반환식만 대체하는 **구현 예시**다. 독립적인 전체 A2 수식이 아니며, 기존 LET 안의 `allrows`를 참조한다.

```text
LET(
  rbload,LAMBDA(tab,
    IFNA(
      FILTER(
        INDIRECT("'"&tab&"'!A2:BL"),
        INDIRECT("'"&tab&"'!BH2:BH")<>""
      ),
      MAKEARRAY(1,64,LAMBDA(rr,cc,""))
    )
  ),
  joined,VSTACK(
    allrows,
    rbload("_rb01_s1"),
    rbload("_rb01_s3"),
    rbload("_rb01_s5")
  ),
  FILTER(joined,CHOOSECOLS(joined,1)<>"")
)
```

BH는 `Result_ID`다. 완전히 빈 새 source는 64열 빈 행으로 처리한다. 존재하지 않는 탭이나 잘못된 header를 `IFERROR`로 전부 숨기지 않는다. source가 먼저 생성되어야 한다.

### 7.2 migration 안전장치

- live A2의 **수식 문자열과 note**, 헤더64열, 주요 view 수식을 byte/hash snapshot으로 저장한다.
- 구조를 파싱하거나 명확한 마지막 반환식 1개만 치환한다. 문자열의 `allrows` 전체를 global replace하지 않는다.
- 재실행 시 세 rbload 연결이 이미 정확히 한 번 있으면 no-op한다. 중복 VSTACK 금지.
- 수식 적용 직전에 재조회해 snapshot과 다르면 다른 편집이 발생한 것이므로 중단한다. 이 비교를 Google API의 원자적 CAS라고 부르지 않는다. 구조 migration은 단일 writer/짧은 편집 창에서 수행한다.
- 먼저 빈 신규 source로 migration 후 기존 관측이 모두 유지되는지 검사한다. 실패하면 **자신이 변경한 A2만** 원래 수식으로 복구한다. 다른 사람이 수정한 값은 덮어쓰지 않는다.
- 그 후 native rows를 올리고 expected ID가 `_records`, `WV3-main`에 각 1회 나타나는지 검사한다.
- `_records`/main grid가 실제 spill 크기보다 작으면 하단 row capacity만 확장한다. 기존 셀·헤더·표 형식 삭제는 금지한다.
- `WV3-main!A5/A6`, 다른 main/archive/PC-Repro의 수식은 변경하지 않는다. 신규 `06` 그룹은 기존 수식으로 자동 표시된다.

새 source는 이미 canonical이므로 기존 parser에 `parse("_rb01_s1",...)`를 중복 추가하지 않는다. 이 bridge와 legacy parser 경로를 동시에 연결하면 두 번 집계된다.

## 8. RB02 stress 전용 등록

### 8.1 `_rb02_points` — 신규 source schema

한 행은 **한 Student·한 inference mode·한 shift에 대해 20 scene을 집계한 점**이다. 독립 Student 행이나 native FR 평가행이 아니다. 다음 헤더를 신규 v1 schema로 고정한다.

```text
Record_ID, Schema, Campaign_ID, Run_ID, Case_ID, Server, Repeat, Seed,
Checkpoint_SHA, Selector, Mode, Shift_ID, Radius_HR, Angle_deg, Dy, Dx,
N_scenes, N_failures, ERGAS, PSNR, SAM, Edge_error_DN,
Relative_response_L1_sum, Coverage_all_pan_paths,
Delta0_ERGAS, Delta0_PSNR, Delta0_SAM, Delta0_Edge_error_DN,
ERGAS_scene_SD, PSNR_scene_SD, SAM_scene_SD, Edge_error_scene_SD,
Curve_numerical_failures, Curve_invalid_geometry,
ROI, Grid_SHA, Evidence_payload_SHA, Source_completed_at_utc,
Status, Readback, Source_artifact, Provenance, Review
```

43열 A:AQ이다. 저장 규칙은 다음과 같다.

- `Schema=RB02_SHEET_POINT_v1`, `Selector=EXACT_50000`.
- `Mode`는 원 enum `A_ON`, `A_ZERO_INFERENCE_ONLY`를 그대로 사용한다.
- `Record_ID=RB02|campaign|run|checkpointSHA|mode|gridSHA|shiftID`.
- curve의 `id/radius_hr/angle_deg 또는 angle_degrees/dy/dx`는 실제 schema를 확인해 명시적으로 변환한다. 별칭 둘이 있으면 값 일치 검사.
- 4개 metric은 원 `curve` 값에서 가져온다. `delta_from_zero_*`, `*_scene_std`도 원 배열과 대조한다.
- `n_failures` 및 coverage를 그대로 유지한다. invalid sampling·nonfinite 결과를 버려서 좋은 곡선을 만들지 않는다. null은 빈칸이며 Review/상태를 남긴다.
- `ROI=32:-32 / fixed192` 및 metric 정의를 note/provenance에 기록한다. 0-shift에서도 native의 crop과 다르므로 native ERGAS로 대체하지 않는다.
- A_ZERO는 추론 시 correction을 0으로 놓은 모드다. independently trained no-align model이라고 부르지 않는다. 상대 response는 원 코드가 측정한 값을 보존한다.
- 위 reference source 코드의 completion은 curve-level invalid geometry 합계를 제공한다. shift별 invalid geometry 수가 필요하면 원 per_scene.csv에서 계산하고 별도 schema revision으로 기록한다. 전체 합계를 각 shift의 실패 수처럼 표시하지 않는다.

같은 logical curve/shift에 다른 payload가 오면 충돌로 중단한다. 업로드 재시도가 새로운 곡선을 만들지 않게 한다. writer 한 곳에서만 Record_ID 기준으로 upsert한다.

### 8.2 `RB02-curves` — 신규 표시 탭

원본 source를 formula로 표시하고 case → server → repeat → mode → radius → angle/shift 순서로 정렬한다. 위 header와 metadata를 보존하며 판독용 열을 앞쪽에 둔다. RAW 데이터와 source link를 수동으로 고쳐 그리는 표로 만들지 않는다.

곡선은 49-point 전체 또는 radius별 7점으로 제공할 수 있다. radius 평균은 **같은 Student 안에서 방향을 먼저 평균한 뒤 Student 사이에서 평균/표준편차**를 계산한다. 8개 방향·20개 scene을 독립 seed로 세지 않는다. 0-radius는 1방향, 나머지는 등록된 8방향을 유지한다.

## 9. 상태와 6-seed 요약

### 9.1 `_rb_b01_status`

24개 registry run을 먼저 ledger에 모두 보존한다. 최소 필드는 campaign/run/case/server/repeat/seed, training actual_updates/status, native status와 selector 수, 두 mode의 curve 상태·shift 수·failure 수, source 검증 상태, upload 상태·오류, source hash, 마지막 확인 시각이다.

등록 상태는 **파일을 못 찾음**, **검증 실패**, **학습 완료**, **native 완료**, **curve 처리 완료**, **업로드 대기**, **readback 완료**를 구분한다. 파일 없음만으로 실험 미실행을 단정하지 않는다. 계획상 expected 값과 measured 값을 별도 열로 둔다.

기술 실패 후 재개된 동일 seed를 두 개 run으로 만들지 않는다. 과거 오류 이벤트는 append-only log로 남기고 현재 슬롯 상태만 갱신한다. 수치 실패도 삭제하지 않는다.

### 9.2 `RB-B01` 신규 요약 탭

제출 관련 수동 탭을 건드리지 않고 이 탭에 대비 자료만 제공한다. 요구 패널은 다음이다.

| 패널 | 내용 |
|---|---|
| 상태 | s1/s3/s5 각 Student 8, curve16의 measured/expected, 오류·누락·업로드 debt |
| Native | case별 EXACT50K mean/sample SD/n, validation-selected는 별도 구역 |
| Paired | 같은 server/repeat/seed의 `case−QFULL`, paired n·평균·SD·개선/동률 수 |
| Stress | case/mode/radius별 n_students·mean/SD, A_ON−A_ZERO paired 차이 |
| Provenance | fixed F1, source/runtime/data/map/evaluator cohort, last verified import 및 source 링크 |

Primary native 표본 수는 case당 최대 6이다. 동일 SHA의 secondary selector는 독립 표본으로 합산하지 않는다. 세 서버 결과를 합산할 때 source/runtime/common F1/data/evaluator가 호환되는지 먼저 확인한다. 다르면 별도 cohort로 보여주고 강제로 pooled6를 만들지 않는다.

원 `summarize`의 통계를 그대로 쓸 경우에도 **세 서버의 평균이나 SD를 단순 평균하지 않는다**. 전부 모인 per-seed rows로 재계산하거나 검증된 통합 summary를 사용한다. 평균은 source와 일치하는 metric 정의에서만 계산한다. 표준편차는 Student sample SD(ddof=1)이며, n=1이면 빈칸이다.

Sheet 내 집계는 새 source를 참조하는 수식을 우선 사용하고, readback에서 원 report/per-seed 재집계값과 대조한다. cross-server report가 없는 동안은 부분 결과와 n을 표시한다. 각 point가 유효하지 않은 Student는 임의로 제외한 평균 대신 missing/failed n을 함께 표시하고, full-six 완료 여부를 별도 표현한다.

전체 curve가 complete이어도 일부 point가 invalid인 상태는 `COLLECTED_WITH_FAILURES` 등으로 표시한다. 구현의 기존 `report.complete` 하나만으로 clean evidence라고 인증하지 않는다.

최소 표시 형식은 현재 main과 맞춘다: 제목·설명·헤더, 고정 header/identity 열, 기본 metric 4자리, 비용 3자리, 회색 기술 metadata, wrap/폭 제한. 원 source 숫자는 반올림하지 않는다. 새 탭의 형식만 바꾸며 기존 탭 전체 copy/format/reset은 하지 않는다.

## 10. 업로드 안전성·재시도

### 10.1 증거와 전달 상태 분리

원 sealed 결과는 불변이다. uploader의 상태는 별도 spool/receipt에서 관리한다.

```text
DISCOVERED → EVIDENCE_VERIFIED → SPOOLED → SOURCE_WRITTEN
           → SOURCE_READBACK_VERIFIED → VIEW_READBACK_VERIFIED
```

실패 시 `UPLOAD_PENDING`, `SCHEMA_CONFLICT`, `EVIDENCE_INVALID`, `FORMULA_REVIEW_REQUIRED` 등 구체적 이유와 마지막 성공 단계를 남긴다. **학습 실패와 업로드 실패는 다른 상태**다.

행의 Run_ID·Checkpoint_SHA·provenance와 숫자를 모두 확인한 뒤 `Readback` 필드를 올린다. 데이터 payload hash와 전달상태 hash를 구분하여 `READBACK_VERIFIED` 문자열 변경이 원 증거 hash를 바꾸게 하지 않는다. 재시도 때 receipt만 믿지 말고 실제 Sheet의 ID/값을 재확인한다.

### 10.2 쓰기 허용 범위

- 매번 spreadsheet **ID로** 접속한다. 같은 제목의 다른 문서를 열지 않는다.
- 현재 승인된 기존 인증 수단을 재사용한다. 공개 공유, 새 계정/credential 생성, token 기록을 하지 않는다.
- source64의 header·ID가 다르면 강제 덮어쓰지 않는다. B01 전용 새 source 밖에 row-values를 쓰지 않는다.
- 구조 setup만 `_records!A2`의 최소 patch를 쓸 수 있다. 일반 `upload`에는 이 권한을 주지 않는다.
- `_rb02_points`, `_rb_b01_status`, 신규 요약/표시 탭은 자신의 schema 소유 범위만 갱신한다.
- 오류·rate limit·timeout은 spool을 유지하고 bounded retry/backoff한다. HTTP 성공 후 응답을 잃었을 때는 key readback으로 완료 여부를 확인한 후 재시도한다.
- 전 서버 병렬 append 또는 `len(rows)+1` 경쟁을 허용하지 않는다. 원 writer/기존 uploader와의 협조되지 않은 동시 구조변경을 피한다.

### 10.3 사전·사후 readback

사전 snapshot은 다음을 포함한다.

```text
모든 tab ID/title/hidden/행열 bounds
_records A1:BL1, A2 formula/note
각 기존 main/archive/PC-Repro A5/A6 및 필요한 header 수식
paper / ablations / 유의미한결과 userEnteredValue·수식·note·형식·merges
기존 _records의 Result_ID → 원정밀도 observation mapping
신규 source의 소유 schema/version
```

Google Sheets 숫자 readback은 unformatted numeric으로 비교한다. 권장 직렬화 오차 한도는 `abs(a-b) <= 1e-12*max(1,abs(b))`; 이것은 **업로드 직렬화 검사 기준**이지 모델 재평가 허용 오차가 아니다. 문자열·SHA·ID·bool·정수는 정확히 비교한다.

완료 후 검사:

1. 각 native Result_ID가 해당 source, `_records`, `WV3-main`에 정확히 1회 나타난다.
2. complete 시 selection행48개/고유 Student24개이며 case마다 primary6개·server마다2개다.
3. stress Record_ID2352개/curve48개·curve당49점이며 native exact SHA와 연결된다.
4. 역사적 source/논문표 원값·수식·note가 유지된다. 기존 관측은 행 위치가 아니라 Result_ID로 비교한다.
5. 주요 수식에 새 `#REF!`, spill overwrite, `#VALUE!`, circular reference가 없다. 원래 있던 무관한 오류를 몰래 고치지 않는다.
6. `RB-B01`의 n·mean·SD·paired 값과 실제 per-seed 데이터가 일치한다. 빈값을 0으로 계산하지 않는다.
7. 같은 payload를 다시 전달하면 신규 row/curve/Student가 0개 증가한다.

기존 source가 같은 시간에 legitimate append되었다면 원 snapshot의 기존 ID는 유지되어야 하며 증가분을 B01과 구별한다. 동시 변경으로 회귀 판정을 할 수 없으면 업로드는 `VIEW_REVIEW_REQUIRED`로 남긴다.

## 11. 신규 CLI 계약과 실행 순서

아래는 **새로 구현할 CLI의 계약**이다. 아직 설치된 기존 명령이 아니다. 단위·통합 테스트 통과 후 실행한다. 소유권·schema 충돌이 없고 dry-run 검사 통과 시 이번 작업 범위의 setup/backfill을 진행하되, 다른 데이터나 기존 표의 교정을 묶어서 실행하지 않는다.

```bash
# 아래 변수는 실제 검증 package 위치로 지정한다.
SHEET_ID=1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0
INCOMING=/actual/path/to/verified/B01_s1_s3_s5_evidence
OUT=work_dir/_rb_sheet_upload/B01

# 1) read-only: live schema/수식 snapshot
python3 tools/rb_b01_sheets.py inspect \
  --spreadsheet-id "$SHEET_ID" --output-dir "$OUT/inspect"

# 2) offline mapping/검증: 부족한 서버·case도 명시
python3 tools/rb_b01_sheets.py prepare \
  --evidence-root "$INCOMING" --output-dir "$OUT/prepared"

# 3) dry-run: 실제 변경 범위/신규 ID/누락/기존값 회귀 계획만 출력
python3 tools/rb_b01_sheets.py plan \
  --spreadsheet-id "$SHEET_ID" --prepared "$OUT/prepared" \
  --snapshot "$OUT/inspect" --output-dir "$OUT/plan"

# 4) 1회 structural setup: 승인된 작은 연결 patch와 새 탭만
python3 tools/rb_b01_sheets.py setup \
  --spreadsheet-id "$SHEET_ID" --plan "$OUT/plan" --apply

# 5) 기존 완료 결과 백필: 학습/추론 호출 없음
python3 tools/rb_b01_sheets.py upload \
  --spreadsheet-id "$SHEET_ID" --prepared "$OUT/prepared" \
  --output-dir "$OUT/receipts" --apply

# 6) source 및 표시/통계/보호된 기존 표 readback
python3 tools/rb_b01_sheets.py verify \
  --spreadsheet-id "$SHEET_ID" --prepared "$OUT/prepared" \
  --snapshot "$OUT/inspect" --output-dir "$OUT/verification"
```

`inspect/prepare/plan/verify`에는 Sheet 변경 권한을 사용하지 않는다. `setup/upload`는 `--apply` 없으면 읽기 전용이어야 한다. `plan`의 숫자와 실제 write 수가 달라지면 원인을 기록한다. 네트워크 업로드 작업이 trainer를 import하여 GPU forward에 진입하는 경로를 만들지 않는다.

이번 B01은 일회성 backlog 등록을 우선한다. 이후 자동 전달이 필요하면 **명시적으로 기동한 별도 bridge worker**가 완료 artifact만 감시하게 하고, frozen controller는 바꾸지 않는다. cron 설치나 무한 uploader를 이번 setup의 숨은 부작용으로 만들지 않는다.

## 12. 필수 테스트 — 아직 수행된 것으로 보고하지 말 것

| 테스트 | 기대 동작 |
|---|---|
| Header drift | 64열 이름/순서 불일치 시 중단, 기존 헤더 교체 없음 |
| Empty source | 세 source가 비어 있어도 기존 `_records` 결과 보존 |
| One native run | source와 view에 selector2행, Student수1 |
| Same checkpoint alias | metric은 일치, selector는 보존, 독립 n 증가 없음 |
| Partial campaign | s1만 완료하면 case n=2/6, 없는 서버를 complete로 처리하지 않음 |
| Duplicate/retry | 동일 payload 두 번 업로드해도 row 수 동일 |
| Evidence conflict | 같은 logical key의 SHA 또는 metric 다르면 overwrite 대신 차단 |
| Tampered report/raw | payload/file hash 오류가 있는 값은 성공 metric으로 등록하지 않음 |
| Missing optional metrics/cost | 빈칸으로 유지, 과거 run 값이나 숫자0으로 채우지 않음 |
| Formula injection | `+ description`, `=text` 등이 문자열로 저장됨 |
| Frozen source | bridge 설치 전후 원 numerical identity 및 결과 파일 SHA 불변 |
| Stress protocol | 49점/20scene/native exact SHA/mode 검증, native지표와 혼합 금지 |
| Stress failures | null·invalid geometry 보존, 정상관측만으로 n을 부풀리지 않음 |
| Statistics | selection alias, 20scene, 8방향을 Student 반복으로 세지 않음 |
| Units | 초→ms/시간, byte 단위가 한 번만 변환되고 scope와 일치 |
| Transport failure | source write 후 timeout이어도 재실행 시 중복 없이 회수 |
| Migration rollback | 빈 source 설치 후 회귀가 생기면 자신이 바꾼 formula만 복구 |
| Historical regression | paper/ablations/유의미한결과 및 기존 IDs·수식·note 불변 |
| Already migrated | union이 이미 있으면 setup은 no-op |
| New version/cohort | 다른 F1/source/evaluator가 섞이면 pooled6 대신 cohort 분리 |

테스트 fixture는 합성 값임을 표시하고 실제 결과 sheet에 올리지 않는다. 로컬 단위 테스트 통과와 실제 live write/readback 성공을 다른 항목으로 보고한다.

## 13. 완료 보고에 반드시 포함할 것

```text
구현 commit / bridge schema revision
검증한 기존 numerical source와 input evidence manifest SHA
대상 spreadsheet ID 및 신규 실제 탭 gid
서버별 registered / verified native / verified curves / missing / failure
업로드된 native 관측 수 / 고유 Student 수 / case별 n
업로드된 curve 수 / point 수 / invalid·numerical failure 수
신규/변경/동일/no-op/충돌 row 수
source readback와 formula-view readback 성공 여부
_records formula 변경 전·후 SHA와 rollback 파일 위치
보호된 기존 표·source 회귀검사 결과
실제 RB-B01 및 RB02-curves 링크
pending 목록과 재시도 명령
```

전부 미완료여도 verified evidence만 정확히 등록했다면 그 범위를 보고한다. **“코드 작성 완료”, “source 업로드 완료”, “화면 반영 확인”, “24 Student·48 curve 전체 검증 완료”는 각각 다른 상태**다. 부족한 실험을 새로 실행하는 것은 다음 사용자 판단으로 남긴다.

## 14. 확인 근거

### Live Sheet readback — 2026-09-30

- metadata: 기존46탭, 숨김 원본과 동적 표시 구조.
- `_records!A1:BL2`: 64열 header, A1 note, A2의 실제 통합 parser/탭 목록/분류.
- `WV3-main!A1:H8`, `QB-main!A1:D6`, `PC-Repro!A1:D6`: A5/A6 표시 수식과 grouping.
- `_case_labels!A1:C40`: 기존 component 설명 lookup. 새 q-case 목록은 없음.
- `WV3-main!A1:A5/D5:D7/P8:S8`: 현재 서식과 raw numeric readback. 당시 WV3-main 표시 관측784개/접근5개는 **시점값**이며 코드 상수로 사용하지 않는다.

Sheet의 현재 수식이 운영 규칙의 기준이다. 저장소 검색만으로 모든 과거 migration 스크립트를 확보한 것은 아니다. 실행 시 live snapshot을 다시 읽고 사용자나 다른 writer의 변경을 보존한다.

### 확인한 저장소 파일 — 수치 실행 commit으로 대체하지 말 것

```text
repository: hojunking/PAN-Crafter-repro
inspected_ref: af1f69e79835268157ccd6cb183d222c280fd31a

panda_rb/common.py       — 실제 campaign/root, source identity 포함 범위
panda_rb/evaluation.py   — native/stress schema, seal, 선택점, crop, 실패 집계
panda_rb/reporting.py    — 로컬 summary, Student 단위 통계, sheets_uploaded=False
tools/panda_rb_runner.py — 기존 report/package CLI, Sheet upload 없음
pcrepro/upload.py       — RAW/spool/readback 패턴 참고; B01에 validator 복제 금지
```

정확한 원문은 아래 pinned 경로에서 확인한다.

```text
https://github.com/hojunking/PAN-Crafter-repro/blob/af1f69e79835268157ccd6cb183d222c280fd31a/panda_rb/common.py
https://github.com/hojunking/PAN-Crafter-repro/blob/af1f69e79835268157ccd6cb183d222c280fd31a/panda_rb/evaluation.py
https://github.com/hojunking/PAN-Crafter-repro/blob/af1f69e79835268157ccd6cb183d222c280fd31a/panda_rb/reporting.py
https://github.com/hojunking/PAN-Crafter-repro/blob/af1f69e79835268157ccd6cb183d222c280fd31a/tools/panda_rb_runner.py
https://github.com/hojunking/PAN-Crafter-repro/blob/af1f69e79835268157ccd6cb183d222c280fd31a/pcrepro/upload.py
```

## 부록 A. 실제 현행 64열 헤더 — exact order

```json
[
  "Experiment",
  "Server",
  "Selection@Step",
  "HQNR↑",
  "D_s↓",
  "D_lambda↓",
  "JQM↑",
  "ERGAS↓",
  "SCC↑",
  "SAM↓",
  "PSNR↑",
  "SSIM↑",
  "Q4/Q8↑",
  "RMSE↓",
  "CC↑",
  "Infer(ms)",
  "Mem(MB)",
  "Params(M)",
  "FLOPs(G)",
  "Train(h)",
  "Eval(h)",
  "Wall(h)",
  "HQNR(V64)↑",
  "Signed D_s",
  "Positive D_s fraction",
  "Dataset",
  "Bucket",
  "Approach",
  "Campaign_ID",
  "Run_ID",
  "Case_ID",
  "Repeat",
  "Seed",
  "Attempt",
  "Role",
  "Model / inputs",
  "Teacher / reference",
  "Parent_run",
  "Updates",
  "Lifetime_updates",
  "Selected_step",
  "Checkpoint_SHA",
  "Selector",
  "q_ref",
  "tau_R",
  "alpha",
  "beta",
  "lambda_E",
  "U_peak_lr",
  "A_peak_lr",
  "Status",
  "Readback",
  "Eval_scope",
  "Test_aware",
  "JQM_variant",
  "Cost_scope",
  "Source_sheet",
  "Source_row",
  "Source_URL",
  "Result_ID",
  "Original description / notes",
  "Review",
  "Sort_case",
  "Date"
]
```
