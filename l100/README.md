# GF2 L100 LOCAL-T v2 — s3 / s4 / s5

이 구현은 `PANDA_GF2_L100_S345_LOCALT_20H_ExperimentPlan_2026-09-21_v2.md`
전용입니다. 기존 G20/FH12 실행·수치 코드·완료 결과를 변경하지 않습니다.
일반 실행 진입점은 아래 `start`입니다. `git pull`, `build`, `status`는
학습이나 공통 20h clock을 시작하지 않습니다.

## 실행

먼저 각 서버에서 이 커밋을 pull하고, 기존 GF2 실행에 사용하던 **5090 지원
Python/PyTorch 환경**을 활성화합니다. 데이터·DLPan-Toolbox·Sheet credential은
기존 서버의 로컬 자산을 사용합니다. 설치나 환경 변경을 자동 수행하지 않습니다.

중앙 제어에서 **실제 준비/전환을 시작할 때 한 번만** 아래 명령을 실행합니다.
아직 실행하지 않을 때 미리 clock을 만들지 마세요.

```bash
python tools/l100_runner.py window --t0 now
```

출력의 `t0_utc` **동일한 문자열**을 세 서버에 전달합니다. 이후 각 서버에서:

```bash
# s3: <공통 t0_utc>를 위에서 출력된 실제 ISO timestamp로 교체
bash tools/l100_start.sh --server s3 --t0 '<공통 t0_utc>'
# s4
bash tools/l100_start.sh --server s4 --t0 '<공통 t0_utc>'
# s5
bash tools/l100_start.sh --server s5 --t0 '<공통 t0_utc>'
```

별도의 case 승인·Teacher 전송·수동 calibration 절차는 없습니다. 각 서버가
자기 Teacher → calibration → Student 순서로 진행합니다. `PYTHON=/path/to/python`
환경변수로 실행 환경을 지정할 수 있습니다. 지속 실행 컨테이너에서는
`--foreground`를 추가합니다. 호스트 PID가 보이지 않는 격리 컨테이너에서 기존
GPU queue와 경쟁시키지 마세요. 먼저 기존 전용 컨테이너의 큐 종료를 확인합니다.

시작 시 현재 커밋의 detached runtime checkout을 자동 생성합니다:
`../PAN-Crafter-runtime-l100-sN-<commit12>/`. 데이터·work_dir·credential은 로컬
자산에 연결하며, 이후 주 checkout에서의 pull이 실행 중 수치 소스를 바꾸지 않습니다.
다른 실험을 kill하거나 기존 queue를 수정하지 않습니다. 기존 로컬 runner/GPU가
사용 중이면 기다리며 **대기 시간도 이미 시작된 공통 창에 포함**됩니다.

```bash
python tools/l100_runner.py status --server s3
tail -f work_dir/_l100/s3/runner.log
```

학습별 로그는 `work_dir/_l100/sN/logs/`에 있습니다. 중단된 controller 재개는
**같은 frozen checkout**에서 `python tools/l100_runner.py start --server sN --in-place`
로 수행합니다. 이미 실행 중인 controller가 있으면 중복 기동하지 않습니다.
새 커밋이나 다른 t0로 기존 실행을 재개할 수 없습니다.

## 등록된 유한 큐

| 서버 | 순서 | 기본 학습 수 |
|---|---|---:|
| s3 | T01(100K) → CAL → T02(50K) → CAL → S01/S02/S03 → S04/S05/S06 | 8 |
| s4 | T03(100K) → CAL → S07(BASE)/S08(H010) → S09(H010)/S10(BASE) | 5 |
| s5 | T04(100K) → CAL → S11(BASE)/S12(E1) → S13(E1)/S14(BASE) | 5 |

ARW/E02의 4개 X-case는 registry에만 보존하며 config·자동 실행 경로가 없습니다.
첫 local chain과 두 번째 seed의 pair/triple을 각각 원자적 block으로 예약합니다.
신규 block은 16h 이전, 남은 전체 block+평가부채는 **엄격히 18h 미만**, 18–20h는
평가·보존에 사용합니다. 시간 부족은 `PAIR_INCOMPLETE`/미입장으로 남깁니다.
계획상의 19.65/15.05/15.05h는 실측 완료 보장이 아닙니다.

## 수치·평가 계약

- Teacher P0/W112/D123, Student PLH/W104/D122, GF2 C4/DN1023, batch48/FP32/AMP OFF.
- BASE/E1·Teacher objective·warp·LP 생성·분리 gradient routing은 기존 구현 재사용.
  H010은 hard weighting의 alpha만 1 → 0.1로 변경합니다.
- `runtime_policy.json`에 TF32/cuDNN/worker 설정을 명시하고 local readback·SHA를 저장합니다.
  동일 GPU명만으로 bitwise 동등성을 주장하지 않습니다.
- 각 local endpoint의 전체 A/U·optimizer·scheduler·RNG·샘플 노출을 검증합니다.
  Student는 자기 서버 Teacher의 A만 독립 clone하고 U는 fresh입니다.
- calibration train3072/seed1234, AXIS16, 전체 train×ROT4 q-cache를 유지합니다.
  **생성과 온라인 검증의 batch16을 통일**하고 manifest에 기록합니다.
- Teacher는 후보별 val-ERGAS와 weights를 보존하고 전체 RR/FR 평가를 명시적 부채로
  이월할 수 있습니다. 로컬 endpoint 검증·calibration이 전체 50-FR 평가를 기다리지 않습니다.
- Student 평가와 endpoint/debt 평가에서 HQNR·SCC·ERGAS를 출력합니다. 아직 측정하지
  않은 Teacher 후보 HQNR/SCC/RR-ERGAS는 `NOT_EVALUATED`이며 val-ERGAS와 혼동하지 않습니다.
- 각 50K/100K run의 공식 후보는 정확히 50개. 진단 step은 공식 선택에 추가하지 않습니다.
  `EXACT_FINAL`, `RR_VAL_SELECTED`, `MID50_OF100`을 구분하고, 별도의
  `best_hqnr_meta.json`은 HQNR → SCC → ERGAS 우선순위를 유지합니다.
- 기존 GF2 RR20/support20:-21/Q4-32와 FR20/full512/native PAN·original LMS·무마스크
  평가 함수를 그대로 사용합니다. RMSE/CC/JQM도 업로드하며 JQM은 SRF-substitute 보조값입니다.
- Sheet 주 열은 이 계획의 EXACT_FINAL이고 선택 종류/학습 길이/Teacher owner·seed·SHA를
  명시합니다. 100K를 기존 Exact50K 열에 넣지 않습니다. 저장은 full precision,
  metric 표시만 소수점 4자리입니다. 50개 평가가 모두 검증되기 전에는 공식 완료 행을 올리지 않습니다.

## 제공 자료와 검증 범위

계획서가 참조한 원본 Cases CSV와 DesignRegistry JSON은 제공되지 않았습니다.
`config/l100/L100_Cases_MD_Derived.csv`와 `L100_DesignRegistry_MD_Derived.json`은
**MD에서 생성한 구현용 정의**이고 `MD_DERIVED_MISSING_DESIGN_ASSETS`로 표시했습니다.
원본 파일을 복구했다고 주장하지 않습니다. 18개 config와 서버별 queue도 함께 커밋합니다.

확장 P1 원 추정기/gradient·border 분석은 `P1_UNAVAILABLE`로 기록하고 진단 시점의
checkpoint를 보존합니다. clone/reference/data/routing/NaN 같은 P0 오류는 무시하지 않습니다.
과거 L100 v1 자산의 자동 재사용·reference 교체는 지원하지 않습니다. 발견 시 기존 clock과
이력을 보존한 명시적 migration 검토가 필요하며 새 20h로 덮어쓰지 않습니다.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m unittest discover -s l100 -t . -p 'test_*.py' -q
python tools/l100_runner.py build
python tools/l100_runner.py start --server s3 --dry-run
```

CPU 회귀·dry-run은 원격 GPU 학습 PASS가 아닙니다. 실제 GPU·데이터는 각 서버 시작 시,
calibration은 각 Teacher 완료 후 자동 검증하며 결과를 `work_dir/_l100/sN/`에 남깁니다.
