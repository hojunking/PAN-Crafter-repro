# 2026-10-30 QB / GF2 checkpoint 보존 감사

## 1. 결론과 확인 범위

**현재 캠페인은 목표 달성 여부와 무관하게 매 1K 공식 평가의 전체 checkpoint를 저장한다.**
목표를 달성한 뒤 후속 학습의 성능이 낮아져도 해당 step의 파일을 덮어쓰거나 지우지 않는다.
Sheet 대표 선택과 파일 보존은 별개다.

- s1의 완료 SENSOR_FOCUS 6개 및 QB_REFIT 3개 run에서 **253개 `.pt` / 12,065,621,442 bytes**를 실제로 읽어 SHA256, sidecar, 원본 evaluation, 동일 모델 상태를 대조했다. 오류 0개.
- 이 253개에는 QB HQNR≥.9315 / ERGAS≤3.5 동시 통과가 없었다. 목표 미달 모델도 저장됨을 실제 파일로 확인했다.
- 각 run의 현행 Sheet 규칙상 대표 9개를 CPU에서 역직렬화하여 Student/Aligner/Teacher, optimizer, scheduler, stream, RNG가 들어 있음을 확인했다. GPU나 학습 프로세스는 사용하지 않았다.
- frozen QB donor @44K와 두 캠페인의 calibration NPZ도 실제 SHA가 등록값과 일치했다.
- **s3 GF2는 직접 파일 접근이 없어 Sheet 게시 당시의 hash 검증 근거까지 확인했다. 현재 시점의 파일 존재와 별도 보관은 s3에서 다시 확인해야 한다.**

감사 시작: 2026-10-30 16:59:35 KST. 진행 중인 QB_REFIT은 이 스냅샷 이후 평가 파일이 더 늘어날 수 있다.

원본 감사: [local JSON](assets/2026-10-30_checkpoint-retention-local-audit.json).
GF2 근거: [live Sheet snapshot](assets/2026-10-30_qb-gf2-closeout-live.json).

## 2. GF2 최종 달성 모델을 정확히 지정해야 한다

| 항목 | 값 |
|---|---|
| Campaign | PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1 |
| Run | PSFC_GF2_S3_GF2P_C60LR200_NOSOFT_R2_S261018102 |
| Step | **47,000** |
| HQNR | **0.9646694883162039** |
| ERGAS | **0.5148830158510865** |
| PSNR | **45.084138095671655** |
| Checkpoint SHA256 | `9027206526d343b5a82aa735057f13de7ed0778ec2ad8fadaec73d27760182d6` |
| Relative path | `work_dir/panda_gf2_psnr/PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1/s3/GF2/runs/PSFC_GF2_S3_GF2P_C60LR200_NOSOFT_R2_S261018102/attempt_01/models/step_0047000.pt` |

Sheet 114행의 원본 정밀도 수치가 같은 checkpoint에서 세 목표를 충족한다.
게시 note에는 weights_saved, 전체 resume 상태, SAME_STATE_FILE_SHA256 검증 근거가 있다.
단, 이는 **게시 시점 s3 publisher의 검증**이며 이번 세션에서 s3 디스크를 직접 hash한 것은 아니다.

같은 run의 **H/E 대표 43K**는 ERGAS가 약 .514804로 더 낮지만 PSNR은 약 45.075620으로 새 목표를 통과하지 않는다.
따라서 `best` 또는 Sheet 대표 하나만 보관하면 안 된다. **47K를 최종 3목표 달성 모델로 명시하여 보존**해야 한다.

## 3. 실제 저장 경로와 내용

`panda_sensor_focus/runtime.py`의 평가 루프는 평가 → `_save` → evaluation JSON → selection/latest_resume → Sheet record 순서로 진행한다.
`evaluation_steps`는 매 1,000 update다. `retention_steps=[horizon]`이라는 config 필드가 있어도 현재 runtime은 그 필드로 중간 모델을 삭제하지 않는다.

`panda30r/engine.py::save_checkpoint`는 새 경로에 원자적으로 저장하고 저장한 전체 payload를 다시 읽어 roundtrip을 확인한다.
동일 경로가 이미 존재하면 덮어쓰기를 거부한다. 이후 저장된 파일 SHA와 모델 상태 SHA를 sidecar에 기록한다.

checkpoint에는 다음이 있다.

- `trainer.stage`: **Student + frozen Aligner + frozen Teacher** 가중치와 stage 상태.
- `trainer.optimizer`: AdamW moment 및 step.
- `trainer.scheduler`, completed update, 설정과 과학적 계약.
- 데이터 순서/cursor/epoch/augmentation RNG, Python/NumPy/PyTorch CPU/CUDA RNG.
- 당시 같은 상태의 RR/FR 평가와 code/evaluator pin.

대표 payload 9개에서 위 항목을 실제로 확인했다. 평가의 tuple이 JSON에서 list로 직렬화되는 차이를 정규화한 뒤 payload와 sidecar/evaluation 내용도 일치했다.

**재개에는 `.pt` 하나만 보관하는 것보다 의존성 묶음을 유지해야 한다.** 현재 loader는 먼저 원본 Stage1 donor와 선택 receipt를 검증하고 calibration을 구성한다.
ECDF 배열 자체는 Student state_dict에 들어 있지 않고, calibration identity/tau_e/binding만 들어 있다. 전체 학습 재현에는 calibration `.npz`와 donor/selector/config/code/data/evaluator 근거를 유지해야 한다.
추론 자체는 Aligner+Student를 사용하며 Teacher와 calibration은 필요하지 않지만, 공식 결과 재현·재학습 경로에서는 해당 근거를 보존한다.

## 4. 자동 삭제와 목표 판정의 차이

- 해당 캠페인 runtime에 `step_*.pt` 자동 삭제 코드가 없다.
- `checkpoint_retention`은 **읽기 전용 preview**다. 삭제·이동·apply 기능이 없고, 현재 학습 artifact를 변경하지 않는다.
- `tools/prune_workdir.py`의 삭제 대상은 완료된 구형 run의 top-level `epoch-*` / `checkpoint-*` 디렉터리이며, 이 캠페인의 중첩 `models/step_*.pt`는 대상이 아니다.
- GF2 publisher는 기존 규칙대로 H/E 동시 통과를 전부 게시하고 대표를 하나 고른 뒤, **PSNR 포함 세 목표 통과 여부를 추가 표시**한다. PSNR 실패가 weight 삭제 조건이 되는 구조가 아니다.
- 향후 보존 정책을 바꿀 때 주의할 점: 현재 preview의 BEST_HQNR/BEST_ERGAS/BEST_BALANCE 역할에는 PSNR이 없다. 이를 곧바로 삭제 허용 목록으로 쓰면 안 된다. 최종 47K 모델과 donor/재개 의존성을 별도로 보호해야 한다.

## 5. s3 종료 전 최소 확인

아래는 읽기 전용 확인 예시다. 저장소 루트에서 실행한다. 목표 충족 여부를 소수점 반올림으로 판정하지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from pathlib import Path
import hashlib, json
root = Path.cwd()
run = root / ('work_dir/panda_gf2_psnr/'
    'PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1/s3/GF2/runs/'
    'PSFC_GF2_S3_GF2P_C60LR200_NOSOFT_R2_S261018102/attempt_01')
p = run / 'models/step_0047000.pt'
meta = json.loads(p.with_suffix('.json').read_text())
ev = json.loads((run / 'evaluation_0047000.json').read_text())
expected = '9027206526d343b5a82aa735057f13de7ed0778ec2ad8fadaec73d27760182d6'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
assert sha(p) == meta['artifact']['file_sha256'] == expected
assert meta['artifact']['weights_saved'] is True
assert meta['artifact']['campaign_resume_complete'] is True
assert meta['evaluation'] == ev
assert ev['state_before'] == ev['state_after'] == ev['rr_state'] == ev['fr_state']
assert ev['state_before']['completed_update'] == 47000
h, e, psnr = ev['fr']['metrics']['HQNR'], ev['rr']['metrics']['ERGAS'], ev['rr']['metrics']['PSNR']
assert h >= .9645 and e <= .522 and psnr >= 45.08
print(json.dumps({'verified': True, 'path': str(p), 'sha256': expected,
                  'HQNR': h, 'ERGAS': e, 'PSNR': psnr}, indent=2))
PY
```

공식 closeout은 위 검증과 함께 다음을 수행한다.

1. 현재 GF2 controller/container identity를 확인한 뒤 추가 run을 예약하지 않도록 전환한다. 진행 중 run을 정지한다면 SIGTERM 기반 안전 checkpoint 저장과 정상 종료 상태를 확인한다.
2. **47K `.pt`와 sidecar/evaluation/config를 최종 선정 manifest에 고정**한다. donor checkpoint·selector receipt·calibration NPZ·admission·데이터/evaluator pin도 경로와 hash로 묶는다.
3. 별도 보관 위치에 복사한다면 원본은 유지하고, 복사본 SHA를 원본과 다시 대조한다. 신규 QB 큐도 이 경로들을 삭제하지 않도록 한다.
4. GF2의 기존 H/E 게시와 세 목표 표시, 최종 상태의 Sheet readback을 확인한다. 성능 목표 달성과 queue 전체 완료는 구분하여, 미실행 케이스를 완료로 표시하지 않는다.
5. GF2 최종 모델 보존·종료 증거를 기록한 뒤 s3 QB 큐를 시작한다.

이번 감사에서는 실험 source, 학습 프로세스, checkpoint 파일을 변경하지 않았다. 보고서와 감사 JSON만 새로 작성했다.
