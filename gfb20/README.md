# GFB20 — GF2 s3/s4/s5 구현·운영

원본: `research_log/PANDA_GF2_B20_S345_20H_Bundle_2026-09-22/`.
Campaign: `PANDA_GF2_B20_S345_20260922_v1`. 원본 bundle은 수정하지 않는다.

## 범위

- s3/s4/s5 전용. 필수 FT 18개, 조건부 포함 최대 36개(s3 16 / s4 10 / s5 10).
- 기존 Teacher 및 Student 부모의 **자기 U/A**를 검증·로드한다. 새 Teacher를 학습하거나 부모 A를 Teacher A로 치환하지 않는다.
- FT는 가중치만 초기화하고 optimizer/scheduler를 새로 만든다. FT 도중의 재개만 해당 run의 전체 상태를 복원한다.
- s1/WV3, s2/QB 및 기존 L100/G20/QG40 실행·결과에는 쓰지 않는다.
- 이번 bundle 기준 primary는 `EXACT_FINAL`, secondary는 `RR_VAL_SELECTED`, 최고 HQNR은 `RAW_AUX`이다. 기존 캠페인의 checkpoint 선택 규칙은 바꾸지 않는다.
- FT 후보는 1K 간격 20개, fresh 후보는 2020 간격과 마지막 100K의 50개이다. **50500은 후보, 진단용 50000은 후보가 아니다.**
- 공식 평가는 native RR20/FR20, DN1023, 기존 RR crop/Q4 및 FR full-PAN/no-mask 경로다. HQNR/SCC/ERGAS를 출력하고 RMSE/CC/JQM도 보존한다. JQM은 기존 SRF-substitute 변형임을 기록한다.

## 배포와 시작

먼저 새 `gfb20/`, `tools/gfb20_runner.py`, `tools/gfb20_start.sh`, 원본 bundle 전체를 커밋·배포해야 한다. 관련 numerical source가 미커밋이거나 변경됐으면 시작하지 않는다. 다른 작업의 변경을 stash/삭제하지 않는다.

각 서버에서 기존 부모를 생성했던 **동일 RTX 5090 환경**의 Python을 사용한다. 이 launcher는 Docker를 자동 생성하지 않으며 CPU/4090 fallback도 없다. 기존 환경의 패키지를 임의로 교체하지 않는다.

자산 발견만 확인하려면 다음을 실행한다. 이는 경로 검색이며 SHA 검증 완료를 뜻하지 않는다. clock이나 학습을 만들지 않는다.

```bash
python tools/gfb20_runner.py inspect-assets --server s3
```

세 서버의 **공통 실제 시작 시각 하나**를 확정하고 같은 timezone-aware 값을 사용한다. 아래 시각 문자열은 실제 값으로 교체한다. 서버마다 별도로 `date`를 실행해서 서로 다른 t0를 만들지 않는다.

```bash
bash tools/gfb20_start.sh --server s3 --t0 'YYYY-MM-DDTHH:MM:SS+00:00'
bash tools/gfb20_start.sh --server s4 --t0 'YYYY-MM-DDTHH:MM:SS+00:00'
bash tools/gfb20_start.sh --server s5 --t0 'YYYY-MM-DDTHH:MM:SS+00:00'
```

각 명령은 해당 서버에서 한 번만 실행한다. `PYTHON=/absolute/path/to/python`으로 interpreter를 지정할 수 있다. 기본은 background 실행이며 `--foreground`도 지원한다. 시작 시 고정된 sibling Git worktree를 만들고 `data`, `work_dir`, credential 파일만 공유한다. 이후 main checkout의 pull이 실행 중 numerical source를 바꾸지 않는다.

대신 한 번 만든 `work_dir/_gfb20/campaign_window.json`을 다른 서버에 복사하여 `--clock-file /absolute/campaign_window.json`으로 동일 clock을 가져올 수 있다. clock 생성만 하려면 `python tools/gfb20_runner.py clock --t0 '실제 공통 시각'`을 쓴다. 미래 t0 이전에 비용을 숨긴 준비 실행은 거부한다. clock은 재시작해도 연장·초기화하지 않는다.

자동 자산 검색은 registry에 등록된 정확한 run ID만 사용한다. 다른 디렉터리에 있다면 `--bindings /absolute/parents.json`을 전달한다. 형식 예시는 다음과 같다(실제 파일의 절대 경로로 교체).

```json
{
  "parents": {
    "P3HI": {
      "origin_root": "/absolute/original/repository",
      "parent_config": "/absolute/run/meta/config.resolved.yaml",
      "parent_checkpoint": "/absolute/run/candidates/100000/model.safetensors",
      "parent_identity": "/absolute/run/candidates/100000/identity.json",
      "parent_fullstate": "/absolute/run/candidates/100000/training_state.pt",
      "reference_manifest": "/absolute/original/reference_manifest.json"
    }
  }
}
```

자산 선택은 경로만으로 인정되지 않는다. 원본 checkpoint/fullstate/config/reference/cache/data의 내용과 SHA, 학습 진행·sampler·노출·끝 LR, source/runtime, native q parity를 검사한다. P3OLD가 없으면 A_OLD block만 보류한다. 다른 부모로 대신 실행하거나 Teacher/cache를 유리하게 골라 바꾸지 않는다. 구 QG40 P3OLD data manifest는 원본을 보존한 별도 lane wrapper로 읽는다.

## 시간·중단·재개

- 공통 t0부터 기존 작업 대기, 준비, calibration, 진단, 저장, 평가를 모두 20시간에 포함한다.
- 16시간 이후 새 block 입장 금지, 18시간 이후 optimizer 금지, 18–20시간은 보존·평가·업로드 마감이다.
- pair/triple 전체 비용과 평가 부채를 예약한다. 바닥 비용 및 동일 서버 실측 총시간 ×1.15 중 큰 값을 사용한다.
- 기존 worker가 있으면 종료까지 기다린다. **기존 L100 controller를 안전-drain시키는 별도 명령은 구현하지 않았다.** 기존 큐를 중지시키거나 kill하지 않으며, 이 대기도 공통 20시간에 포함된다. 충분한 실행 시간을 원하면 기존 큐 인계가 끝나는 시점에 공통 시작을 조율해야 한다.

```bash
python tools/gfb20_runner.py status --server s3
python tools/gfb20_runner.py stop --server s3 --command STOP_NOW_SAFE
python tools/gfb20_runner.py stop --server s3 --command STOP_AFTER_BLOCK
```

학습 중 `STOP_NOW_SAFE`는 optimizer 경계에서 해당 B20 run의 전체 상태를 저장한다. 준비·mixed cache·평가 subprocess는 control 파일을 직접 감시하지 않으므로, 그 단계에서는 즉시 중단하지 않고 준비가 끝난 뒤 다음 scheduling 경계까지 반응이 지연될 수 있다. `STOP_AFTER_BLOCK`은 이미 입장한 block을 마친 뒤 멈춘다. 다른 캠페인을 건드리지 않는다. 안전 중단 후 재개:

```bash
python tools/gfb20_runner.py stop --server s3 --command CONTINUE
bash tools/gfb20_start.sh --server s3
```

새 t0나 임의 train `--resume`를 주지 않는다. 같은 release/부모/stream/clock으로 controller가 재개한다. 완료된 학습은 재실행하지 않고 남은 평가만 처리한다. 인프라 오류의 학습 재시도는 동일 fullstate로 최대 2회다. FT val-ERGAS가 부모의 1.10배 이상인 후보가 연속 두 번이면 양 arm에 동일한 발산 규칙을 적용한다. HQNR에 의한 유리한 조기중단은 하지 않는다.

## s4/s5 해석 게이트

s4 fresh 선택에는 공식 grid 완료뿐 아니라 core 6개 run의 t=0/5K/20K gradient 진단과 checkpoint/source/config 대응이 필요하다. 누락되면 조건부 승격하지 않는다. §7.2의 LOW_ACTIVE_MASS 우선 규칙은 **K1 양 부모 모두** 해당 진단일 때 E100 fresh-path를 우선하는 것으로 구현했다. 이때 K1의 양성 신호를 E100의 성공으로 이월하지 않는다. β 추가 자동 탐색도 없다.

s5는 전체 19809×4×3 q-cache, gamma1 전체 native parity, calibration3072의 weighted pooled-pixel τ 및 qref를 사용한다. 작은 subset으로 본 cache를 대체하지 않는다. gamma=1은 기존 native PAN/LP를 그대로 사용한다. PAN gain 후 해당 LP 재생성, 이후 HV/ROT 순서를 지키며 MS/GT와 평가 입력은 바꾸지 않는다. mixed reference가 2시간을 넘으면 두 번째 fresh pair를 제외한다.

부모 native 평가 equality guard는 full-precision bundle 값에 대해 `atol=1e-6, rtol=0`으로 고정했다. 이는 **구현상의 보수적 사전 허용치**이며 RTX5090 반복실측으로 추정한 오차폭이라는 주장이 아니다. 불일치 시 중단·기록하며 자동 완화하지 않는다. fresh trunk 비교는 변경 가능한 요약 파일 전체가 아니라 고정 checkpoint와 수치 선택 identity를 사용한다.

선택적 P1 sensitivity replay 및 독립 정합 추정기의 새 구현은 이번 범위에 포함하지 않았다. 기존 검증 추정기를 연결하지 않았으므로 관련 분석은 `ESTIMATOR_NOT_AVAILABLE`로 취급한다. 필수 B gradient 및 C view/q 검증과는 구분한다.

## 출력·업로드

- 캠페인 상태: `work_dir/_gfb20/<server>/`.
- run: 기존 규약대로 `work_dir/<run_id>/`.
- 공식 결과: `official/raw_grid.json`, `official/summary.json`, `official/upload_receipt.json`.
- 별도 Google Sheet 탭: `GF2-B20-s3`, `GF2-B20-s4`, `GF2-B20-s5`. 기존 GF2 탭은 덮지 않는다.
- 실제 학습 시작일 UTC/KST, local/lifetime updates, 부모 계보·비용, 세 selector의 **각각 같은 checkpoint에서 나온 모든 지표**를 업로드한다. 원 수치는 보존하고 metric 표시 형식은 소수 4자리로 설정한 후 raw 값을 readback 검증한다.
- `--no-upload`는 시작 시 live 업로드를 끈다. 테스트는 mock worksheet만 사용한다.

```bash
python tools/gfb20_runner.py report --server s3
python tools/gfb20_runner.py retry-uploads --server s3
```

업로드 재시도는 고정 release로 전달되며 학습을 재시작하지 않는다. 전체 비교는 계획서대로 test-aware development이고 fresh seed가 untouched test set을 만드는 것은 아니다.

## 로컬 검증

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest discover -s gfb20 -t . -v
bash -n tools/gfb20_start.sh
```

실제 서버 자산·GPU 사전검증, 전체 mixed cache 생성, 실제 Sheet 권한/readback은 CPU unit test로 완료됐다고 간주하지 않는다.
