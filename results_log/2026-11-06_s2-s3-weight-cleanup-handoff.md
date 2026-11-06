# s2 / s3 weight 정리 인계 — 2026-11-06

사용자가 s1에서 적용한 weight 정리 기준을 **s2와 s3에도 전달해 각 서버에서 정리**하도록 요청했다.
전달은 기존 방식대로 **사용자 push → 각 서버 pull → 해당 서버 에이전트 실행**이다.
이 문서는 원격 삭제 완료 보고가 아니다. s1에서 s2/s3 파일을 읽거나 삭제하지 않았다.

## 1. 적용 기준

| 대상 | 삭제 기준 | 보존 기준 |
|---|---|---|
| Main / 하이퍼파라미터 튜닝 Student | 완료 또는 검증된 HQNR 조기 종료 run 중, 단 한 checkpoint도 공동 목표를 달성하지 못했다면 weight 전부 | 한 checkpoint라도 목표를 달성한 run은 이번 정리에서 유지 |
| 30K component ablation | **step ≤20,000** (20K 포함) | 21K 이후, 아래 예외 |
| 50K component ablation | **step ≤30,000** (30K 포함) | 31K 이후, 아래 예외 |
| Main의 Stage 1 Aligner/Teacher | Student 목표 미달 기준으로 제거하지 않음 | donor와 그 선택·calibration 증거 유지 |
| 실행 중 / 대기 / 중단 후 재개 가능 / 실패 복구 대상 | 아직 최종 미달로 판정하지 않음 | 현재 실행·재개·후속 큐가 필요로 하는 weight 유지 |

목표는 **동일 checkpoint의 반올림 전 원본 값**으로 판정한다. 서로 다른 step의 최고 HQNR과 최저 ERGAS를 합치지 않는다.

| 데이터셋 | HQNR 최소 | ERGAS 최대 | PSNR 최소 |
|---|---:|---:|---:|
| WV3 | 0.9585 | 2.040 | 없음 |
| QB | 0.9315 | 3.5 | 없음 |
| GF2 | 0.9645 | 0.522 | 45.08 |

GF2는 최신 최종 목표인 PSNR까지 함께 확인한다. 이미 확정한 GF2 최종 모델과 논문 지정 결과는 반드시 유지한다.
표시상 3자리 수치나 Sheet 탭 이름만으로 삭제하지 않는다.

**예외 우선순위:** 실제 후속 학습/추론/재개/사전 검증이 읽는 donor·parent·baseline,
Stage 1의 validation-selected 모델, 논문·그림에 지정한 원본과 보관본을 먼저 보존 목록에 넣는다.
단순히 main 결과의 대표행이라는 이유로 목표 미달 weight를 보존할 필요는 없다.
Ablation의 초기 HQNR 대표도 삭제 대상이지만, 선택 donor나 논문 사용 모델이면 예외다.
삭제 예외의 run·step·SHA·소비하는 코드/후속 case를 보고서에 기록한다.

50K의 앞 30K를 제거하면 마지막 20K가 남으므로 후기/최종 결과 비교에는 타당하다.
그러나 Stage 1은 최고 검증 RRE 모델이 30K 이전일 수 있다. 이 모델을 제거하고 후기 모델로
재선택하면 실험 정의가 달라지므로 **원래 선택 모델을 예외로 보존**한다.
초기 궤적은 평가 JSON/로그로 계속 분석할 수 있고, 삭제한 weight로 재추론하는 것은 불가능하다.

## 2. s2: QB main 튜닝

확인할 주요 경로:

- `work_dir/panda_qb_s2/PANDA_S2_QB_STAGE1_BALANCE_20261102_v1/s2/QB`
- `work_dir/panda_qb_s2_frozen/PANDA_S2_QB_FROZEN_S2_20261104_v1/s2/QB`

`QB6`, `QFS2`는 `QB-ablations`에 게시되지만 **main 튜닝**이다.
30K ablation의 20K cutoff를 적용하지 않고, 완료된 Student run의 공동 목표 달성 여부를 확인한다.
실제 현재 큐 상태는 s2의 campaign/run state와 프로세스로 확인한다. s1에서 완료 여부를 추정하지 않는다.

다음 네 weight는 [frozen plan](../config/panda_qb_s2_frozen/plan.json)에 고정되어 있으며 보존한다.

| 역할 | 원본 run (`PQS2_QB_S2_` 접두사) | Step |
|---|---|---:|
| D4 Aligner/Teacher | `QB6_S1_A400_E030_D301_NOSOFT_R1_S261016301` | 48K |
| D2 Aligner/Teacher | `QB6_S1_A200_E030_D301_NOSOFT_R1_S261016301` | 48K |
| Q0 R1 기존 Student baseline | `QB6_S2_A400_E030_D301_NOSOFT_R1_S261018101` | 26K |
| Q6 R1 기존 Student baseline | `QB6_S2_A200_E030_D301_NOSOFT_R1_S261018101` | 26K |

마지막 두 모델도 현재 [donor.verify_baselines](../panda_qb_s2_frozen/donor.py)가 실제 파일 SHA를 읽는다.
목표 미달이더라도 이를 삭제하면 다음 실행의 사전 검증이 실패한다. 비교용이라는 막연한 이유가 아니라
**현재 실행 의존성** 때문에 보존하는 예외다. 나머지 미달 Student 파일은 의존성 확인 후 삭제한다.
`rre_selection.json`, `models/selection.json`, config/run state, calibration JSON/NPZ도 그대로 둔다.
다른 서버의 donor로 대체하거나 새 Stage 1을 학습할 필요는 없다.

## 3. s3: WV3 Stage 1 ablation 및 과거 GF2/QB

확인할 주요 경로:

- `work_dir/panda_wv3_ablation/PANDA_WV3_COMPONENT_S13_20261102_v1/s3/WV3`
- `work_dir/panda_wv3_ablation_rgb/PANDA_WV3_STAGE1_RGB_ABL_S3_20261103_v1/s3/WV3`
- s3에 실제 존재하는 GF2/QB main 튜닝 캠페인

mean8 및 `(RGB)` Stage 1 component ablation은 각각 독립된 결과다.
완료된 50K run에 **30K 이하 삭제**를 적용하되, 각 run의 기존 validation 대표를 보존한다.
Stage 1의 정합 성능을 Student의 HQNR/ERGAS 목표로 잘라 전체 삭제하면 안 된다.
현재 게시 선택은 validation RRE50 → RRE90 → Val ERGAS → 이른 step 순서이며,
테스트 RRE로 donor를 새로 고르지 않는다. 기존 receipt가 있으면 그 선택을 우선 대조한다.

각 A0–A5 × 반복의 후기 모델과 선정 모델을 유지한다. 아직 진행 중인 반복은 정리하지 않는다.
과거 mean8 run이 RGB 전환 때문에 STOPPED라면, 단순 중단을 완료로 간주하지 말고 재개 의존성을 검토한다.
GF2/QB main에는 위 공동 목표 규칙을 적용하며 GF2 최종 모델/보관본을 유지한다.

## 4. 해당 서버 에이전트 실행 절차

사용자가 push한 커밋을 pull한다. 실행 중인 학습이 읽는 기존 source/registry/release 파일은 바꾸지 않는다.
이번 커밋은 별도 정리 패키지와 문서만 추가하며 학습 큐·하이퍼파라미터를 변경하지 않는다.

먼저 저장소 루트의 기존 Python 환경에서 해당 서버 명령 하나를 실행한다.

```bash
python -m checkpoint_cleanup.handoff --server s2
```

```bash
python -m checkpoint_cleanup.handoff --server s3
```

이 명령은 **읽기 전용 범위 점검**이다. 서버별 campaign/run 상태와 분류를 기록하고,
s2에서는 위 네 파일의 SHA도 확인한다. 결과는
`work_dir/_operator_storage/20261106_<server>_cleanup/inventory.json`이다.
다시 점검하려면 `--output work_dir/_operator_storage/<새_이름>.json`으로 새 파일을 지정한다.
`deletion_ready: false`는 파일별 삭제 manifest가 아직 준비되지 않았다는 뜻이다.
미분류 캠페인이나 누락 파일을 main 미달로 자동 해석하지 않는다.

이후 서버 에이전트는 사용자가 승인한 위 범위에서 아래를 끝까지 수행한다.
정상적인 파일 대조·삭제를 위해 추가 사용자 승인을 요청할 필요는 없다.

1. 로컬 state·실제 프로세스·전체 평가 JSON을 대조한다. 완료 horizon 또는 조기 종료 증거,
   run/attempt/seed, 평가 누락 여부를 확인하고 원본 정밀도로 공동 목표를 판정한다.
2. 기존/후속 큐의 checkpoint·parent·donor·baseline·논문 경로를 수집해 보존 목록을 만든다.
   삭제 목록에는 각 **정확한 로컬 파일 경로**, run·step·삭제 사유·파일 SHA·크기,
   보존할 config/evaluation/sidecar의 SHA를 기록한다.
3. 정리 대상의 기존 Sheet 업로드가 완료됐는지 확인하고 현재 행을 보관한다.
   **원래 publisher는 weight 존재/SHA를 요구한다.** 삭제 전에 아래 재게시 처리를 검증한다.
4. 서버별 manifest를 새로 만들고, 삭제될 모든 파일의 SHA를 검증한다.
   `checkpoint_cleanup.ledger` / `apply` / `publication`의 s1 구현을 참고하되,
   **s1의 `prepare`나 s1 manifest를 s2/s3에서 그대로 실행하지 않는다.**
   `prepare.py`는 s1 audit 경로/범위 전용이고 Stage 1을 제외하므로 s3에 직접 사용할 수 없다.
   서버 에이전트가 로컬 audit와 보호 목록으로 manifest 생성부를 맞추고 검증한다.
5. 미리 검증한 파일만 삭제한다. 파일이 바뀌거나 evidence가 맞지 않으면 그 항목을 삭제하지 않는다.
   Dataset/HDF5, JSON/JSONL, 평가·학습 로그, selection/calibration, outbox/게시 receipt는 제거하지 않는다.
   미달 완료 run의 optimizer/resume weight도 소비자가 없으면 정리 가능하다.
6. 재게시 수치/대표행 불변, 보호 모델 SHA, 현 학습·후속 시작 검증, disk 여유를 다시 확인한다.
   삭제 파일 수/용량, 보존 예외, 실제 처리하지 않은 항목, 오류 여부를 서버별로 보고한다.

Git에는 소스/정책/요약만 전달한다. 실제 weight, dataset, credential,
s1의 로컬 삭제 manifest 및 저널을 s2/s3에 복사해서 실행하지 않는다.

## 5. Sheet와 재게시

삭제는 실험 결과 삭제가 아니다. 기존 대표행/통과행의 수치·선택 step·원본 SHA·비용·3자리 표시를 유지한다.
삭제한 weight에 대해서만 A열 note 및 `Original description / notes`에
`checkpoint_retention.status=REMOVED_BY_USER_POLICY`, `weights_currently_available=false`,
삭제 manifest 경로/SHA, 삭제 전 파일 SHA 검증 사실을 추가한다.
`weights_saved=true` 등의 원래 로그는 저장 당시 사실이므로 수정하지 않고 현재 보유 상태를 별도로 명시한다.

원본 평가 기록과 삭제 저널을 검증하는 wrapper로 재게시해야 한다.
현재 [publication.py](../checkpoint_cleanup/publication.py)는 s3의 두 WV3 ablation publisher 및
s2의 QB legacy-style publisher 구조를 처리할 수 있게 작성되어 있으나,
**s2/s3 실데이터 검증은 해당 서버에서 해야 한다.**
삭제 전 preview와 삭제 후 projection의 모든 수치·행 ID·선택 step이 동일해야 한다.
누락 weight를 무조건 무시하거나, 남은 후기 모델만으로 대표를 재선택하는 방식은 사용하지 않는다.
이 기능을 쓰려면 서버별 manifest의 family/root/server 및 retained-evidence binding이 정확해야 한다.
기존 publisher를 계속 주기 실행하는 프로세스가 있다면 대상 캠페인의 publisher만 wrapper로 전환하고
다른 학습·다른 캠페인의 게시를 중단하지 않는다.

## 6. s1에서 완료된 작업 참고

2026-11-06 s1에서 2,199 weight / 82.72 GiB를 삭제했고 JSON·로그는 유지했다.
WV3 main 720개, QB main 1,039개, 30K ablation의 20K 이하 440개였다.
Stage 1 donor, 목표 달성 main run, 논문 지정 모델, 실행 중인 W128은 보존했다.
9개 캠페인 재게시 검증과 Sheet 71행의 보유 상태 갱신/readback을 완료했다.
이 숫자를 s2/s3 예상 삭제량으로 사용하지 않는다.

패키지 단위 검증:

```bash
python -m unittest discover -s checkpoint_cleanup/tests -v
```
