# FH12 구현 및 실행 인계

기준 계획: `PAN_FH12_WV3_LPAN_HPAN_FreshTeacher_12H_2026-09-18.md`.
구현 상태: 로컬 코드·서버별 config/queue·CPU/모의 통합 테스트 완료. 실제 CUDA preflight, 신규 학습, 원격 배포, Sheet 쓰기는 이 구현 작업에서 실행하지 않았다.

## 실행

동일 코드를 배포한 각 서버의 저장소에서 아래 명령 하나를 실행한다. 기존 `gspread/server.txt`로 서버를 식별한다.

```bash
bash tools/fh12_start.sh
```

서버 식별 파일이 없는 경우만 `--server s1`처럼 지정한다. 기존 식별 파일과 충돌하는 지정은 거부한다. 명시적 시작 전에는 코드 배포/pull만으로 실험이 시작되지 않는다.

- 시작 시 서버별 12시간 창을 한 번 기록한다. 기존 작업 대기·준비·calibration·평가 시간이 모두 포함된다.
- 기존 작업을 강제 종료하지 않는다. 확인된 M20의 다음 case 진입을 보류하고 현재 작업 및 후처리 종료를 기다린다. 안전한 drain을 확인할 수 없는 구형 실행기는 자동으로 가로채지 않는다.
- 별도 수동 gate 없이 데이터/LP 준비, CPU 수식 검사, GPU 실 batch48 smoke, 새 Teacher, exact50K calibration, Student, 공식 평가와 업로드를 이어간다.
- 남은 시간이 기본 대조를 수용하지 못하면 `DEFERRED_BUDGET`; 진행 중 마감이면 full-state를 보존하고 `PAUSED_DEADLINE`이다. 부분 학습을 50K 결과로 올리지 않는다.
- 마감 이후 기존 큐로 자동 복귀하지 않는다. 재시작으로 12시간 창을 초기화하지 않는다.

선택적인 확인/상태/업로드 재시도 명령:

```bash
bash tools/fh12_start.sh --dry-run
python tools/fh12_runner.py status
python tools/fh12_runner.py retry-upload
```

실행 인터프리터는 기본 pancrafter 환경이며 다른 위치에서는 `PYTHON=/path/to/python bash tools/fh12_start.sh`로 지정한다. `retry-upload`는 학습·추론을 다시 실행하지 않는다.

## 서버별 확정 배정

| 서버 | fresh Teacher W112D123 | 기본 Student W104D121 순서 | 시간 허용 예비 |
|---|---|---|---|
| s1 | P0 / 71001 | P0 → PLH / 72001 | 없음 |
| s2 | PL / 71001 | PL → P0 / 72001 | 없음 |
| s3 | PH / 71001 | PH → P0 / 72001 | PH W104D122 → PH W112D121 |
| s4 | PLH / 71001 | PLH → P0 → PL → PH / 72001 | PLH W104D122 → PLH W112D121 |
| s5 | PLH / 71002 | PLH / 72002 | PLH W104D122 |

학습 config는 Teacher 5 + 기본 Student 11 + 예비 Student 5 = 21개다. 서버별 preflight/calibration을 포함한 registry는 31개 stage다. 각 Student는 다른 서버를 기다리지 않고 자기 서버의 Teacher만 참조한다.

## 구현 범위

- `fh12/model.py`, `losses.py`: P0/PL/PH/PLH 9/10/10/11ch, FP32 shared PAN/LP warp, signed HP, native MS base 1회, margin4 PAN1/MS8 aligner. 공통 fresh 초기화와 추가 L/H kernel=0. Teacher odd-update A-only consistency, Student U/A 분리 gradient.
- `fh12/data.py`, `calibration.py`: 원본 H5를 바꾸지 않는 새 native-PAN LP cache. Gaussian σ1.98/k41/replicate/offset2 고정. Train3072 seed1234 τ_R, 전 train×4 view AXIS16 q, 양의 q_ref, local exact50K SHA 검증. phase 측정값도 기록하며 보정하지 않는다.
- `fh12/training.py`: 별도 FP32 50K/batch48 trainer. 모델·optimizer·scheduler·scaler·Python/NumPy/Torch/CUDA 및 sampler/corruption RNG 재개. checksum과 상태를 함께 atomic publish한다. 후보 저장도 완료된 디렉터리만 공개한다.
- `fh12/evaluation.py`, `postrun.py`: raw-original FR20/full512와 공식 RR20/full256 추론 후 RR 지표용 `[20:-21]` support. aligned-PAN 참조나 shift-dependent mask를 적용하지 않는다. clipping 후 DN 변환, 임의 반올림 없음(Q8 자체 정수 처리 제외).
- `fh12/upload.py`: 기존 WV3 서버 탭/GID 및 헤더명을 사용한다. `(campaign, run)` upsert와 readback 검증. 기존 행 및 NOA 열은 보존한다. 네트워크 실패 시 local 공식 결과를 유지한다.
- `tools/fh12_start.sh`, `fh12_runner.py`, `fh12_preflight.py`, `fh12_calibrate.py`: 자동 준비 및 서버별 유한 큐. Calibration은 별도 프로세스여서 종료 후 runner 자신의 CUDA context를 기다리는 문제가 없다.
- 기존 `main.py`/훈련 loop는 수정하지 않았다. 기존 `_watchdog.sh`/`_run_cases.sh`에는 명시적 FH12 등록이 있을 때만 동작하는 분기만 추가했다.

## 평가·저장 규칙

후보는 `{1010*k, k=1..49} ∪ {50000}` 50개다. 평가 로그에는 반드시 **HQNR/SCC/ERGAS**를 출력한다.

- 본 Sheet RR/FR: 같은 checkpoint의 `RAW_MAX`(최대 raw HQNR).
- `TARGET`: HQNR≥.9585 중 ERGAS 최소, 동률 SCC→PSNR→낮은 step. 해당 후보가 없으면 Target 수치는 빈칸.
- `EXACT50K`: update 50000. Student reference는 항상 해당 로컬 Teacher의 이 가중치다.
- `RR_VAL_SELECTED`: 독립 validation 1080 patch의 최소 ERGAS 후보.
- `E_MIN_DIAG50`: RR test-aware 개발 진단이며 독립 test 성능으로 표시하지 않는다.

각 선택점의 HQNR/ERGAS/SCC/PSNR/SAM/Q8/SSIM/Dλ/Ds는 모두 같은 checkpoint에 속한다. 완전한 50개 후보 및 공식 RR 확인 전 Target을 인증하지 않는다.

구현상 매 후보의 기존 RR 모니터링 추론 출력을 재사용해 6개 공식 RR 지표를 모두 계산한다. 따라서 E_MIN용 추가 GPU 전수 추론 없이 50개 ERGAS 진단을 얻는다. Q8 등의 CPU 계산도 포함한 evaluator 실측 시간을 매 1010 update projection에 반영한다. 실제 12시간 내 모든 case 완료를 보장하는 것은 아니다.

디스크 중복을 줄이기 위해 모든 후보의 모델 가중치·identity는 보존하되 full optimizer/RNG는 최신 재개점과 exact50K에 저장한다. 모든 후보의 대용량 SR MAT를 중복 저장하지 않는다. 공식 지표와 장면별 값은 JSON에 보존한다.

RR256 비용 측정은 전체 A+frequency frontend+U를 포함한다. Sheet FLOPs(G)는 기존 THOP MAC 스케일에 frontend의 명시적 MAC-equivalent 추정치를 더한다. 별도 arithmetic FLOPs 추정치와 계산 convention도 저장한다. LP cache 생성은 offline 준비 비용이며 inference에서 제외한다. 계산량은 하드웨어 instruction의 정확한 총량으로 주장하지 않는다.

## 검증 및 남은 실행 시 확인

검증 대상은 model, assets/calibration, sampler/full-state resume, official evaluator/profile, runner/admission, uploader, preflight다. **총 107개 테스트가 통합 실행에서 모두 통과**했다(model16 + assets17 + training15 + evaluation5 + runner27 + upload14 + preflight13). 개별 실행 파일은 `tools/fh12_*_tests.py`이며 GPU를 사용하지 않고 실행했다. Config generator `--check`, shell syntax, `git diff --check`, 실제 starter `--dry-run`도 통과했다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  /home/knuvi/miniconda3/envs/pancrafter/bin/python -m unittest discover -s tools -p 'fh12_*_tests.py'
```

로컬 `--check-only`로 train9714/val1080/RR20/FR20 및 canonical source SHA 일치를 확인했다. 새 LP manifest는 아직 생성하지 않아 `CHECK_ONLY_NEEDS_PREPARATION`이며 이는 시작 명령이 자동 처리하는 준비 단계다. `preflight_pass=true` 또는 실제 GPU 검증 완료로 오인하지 않는다.

실제 기존 저장 출력 20장을 재계산한 대조에서는 신규 RR의 6개 지표 및 FR의 HQNR/Dλ/Ds가 기존 공식 평가 함수와 모두 동일했다. 이는 Python evaluator 간 동일성 확인이지 MATLAB bitwise 동일성 주장과는 다르다.

중단/재개 테스트는 stochastic tiny backbone으로도 학습 sample/rotation 순서와 최종 가중치의 bit-identical 재현을 확인했다. 후보 저장 중 실패, checksum 손상, source/config drift, 무적격 Target, 잘못된 서버/Sheet GID, 업로드 readback 실패도 검사한다.

실제 각 서버 GPU의 속도·VRAM·라이브 Sheet 인증은 아직 실행하지 않았다. 본 시작 명령이 GPU 배치 검사를 먼저 수행하고, 실패하면 50K 학습을 시작하지 않는다. 이 검사는 소모성 smoke 모델을 사용하며 실제 Teacher/Student 초기 가중치로 재사용하지 않는다.

실험 시작 전 동일 release로 배포하고 진행 중 소스/config를 바꾸지 않아야 한다. exact resume/calibration은 소스·런타임·데이터·reference identity가 달라지면 조용히 계속하지 않고 거부한다.
