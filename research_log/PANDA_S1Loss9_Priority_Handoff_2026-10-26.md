# S1LOSS9: Stage 1 손실 계수 탐색을 세부 LR 및 WV3보다 먼저 실행

> **2026-10-28 후속 지시로 대체됨:** 현재 유지·제외 목록, 우선순위, s3 설치·실행은 [SensorRefine R3/R4 인계](PANDA_SensorRefine_R34_Handoff_2026-10-28.md)를 따른다. 아래의 9조합 전체 실행 순서와 `stage1-loss9-priority` 브랜치·bundle·인증 불가 서술은 현재 전달 상태를 보증하지 않는 과거 기록이다. 실제 전달 commit과 push 결과를 확인하며, 기존 실행 중 학습 소스는 변경하지 않는다.

2026-10-26 사용자 지시. 표기는 [Method 수식 통일안](PANDA_Method_EquationStyle_2026-10-26.md)을 따른다. 이 추가 큐는 시간 상한 없이 실행한다. s1은 QB, s3는 GF2를 담당한다.

## 목적과 기존 큐와의 차이

기존 SENSOR5는 Stage 1의 λ_abs=2, λ_epsilon=.003을 고정하고 Stage 2 LR·학습 길이·Hard 강조를 비교한다. 이 두 Stage 1 계수의 튜닝은 기존 다섯 case에 포함되지 않았다. 신규 캠페인은 다음 목적함수의 계수를 먼저 비교한다.

$$
\mathcal L_{\mathrm{S1}}=\mathcal L_{\mathrm{rec}}+
\lambda_{\mathrm{abs}}\mathcal L_{\mathrm{abs}}+
\lambda_\epsilon\mathcal L_\epsilon.
$$

λ_abs는 RGB 구조 정합과 reconstruction의 상대 비중이고, λ_epsilon은 합성 PAN 이동에 대한 Aligner 응답 일관성이다. 둘 다 shift norm penalty는 아니다. 특히 λ_epsilon을 낮춘다고 native 보정량이 작아지는 것으로 해석하지 않는다. ε 경로에는 Teacher/Student recon을 추가하지 않는다.

공통 notation은 유지하되 GF2·QB에서는 C=4, Aligner 입력은 전체 4밴드 Stored LMS, Teacher 입력은 보정 PAN+업샘플 MS 5채널, Student 입력은 `[보정 PAN, 보정 PAN−원본 PAN, I_ms^lr]` 6채널이다. 문서의 WV3 C=8·10채널을 이 두 센서에 그대로 적용하지 않는다.

## 9개 조합과 실행 순서

| 순서 | pair | λ_abs | λ_epsilon | 비교 목적 |
|---:|---|---:|---:|---|
| 1 | A200_E003 | 2 | .003 | 기존 계수의 새 캠페인 기준점 |
| 2 | A020_E003 | .2 | .003 | 구조 정합 가중 1/10 |
| 3 | A050_E003 | .5 | .003 | 구조 정합 가중 1/4 |
| 4 | A200_E0003 | 2 | .0003 | 이동 일관성 가중 1/10 |
| 5 | A200_E030 | 2 | .03 | 이동 일관성 가중 10배 |
| 6 | A020_E0003 | .2 | .0003 | 두 약화 축의 상호작용 |
| 7 | A020_E030 | .2 | .03 | 약한 절대 정합·강한 이동 일관성 |
| 8 | A050_E0003 | .5 | .0003 | 중간 절대 정합·약한 이동 일관성 |
| 9 | A050_E030 | .5 | .03 | 중간 절대 정합·강한 이동 일관성 |

각 조합은 **fresh Stage 1 50K → 해당 조합의 validation-selected A/T 동결 → fresh Student 30K** 순서로 끝낸 뒤 다음 조합으로 간다. 서버당 18 run, 720K update다. 두 서버 합계 36 run이다. 시간 상한으로 case를 skip하지 않는다. 실제 소요 시간에는 모든 validation selector·calibration·공식 평가·cost 측정도 포함한다.

기준점도 새 namespace에서 동일 seed로 학습한다. 이는 구현·캠페인 재현 대조이며 독립 seed 반복이 아니다. 기존 SENSOR5 checkpoint를 신규 캠페인 결과로 재표기하지 않는다. 조합별 donor receipt, source config, checkpoint, calibration을 분리한다.

## 고정 조건

- Aligner scale=1, 전역 2성분 이동, Stage 1/2 RGB 평균 구조 참조, native/ε/Teacher/Student PAN warp는 Lanczos-4.
- Stage 1 LR_A=1e−5, LR_T=1e−4, warmup100+cosine50K. reconstruction 계수1, ε는 RR PAN 반경3의 면적 균일 원판이다.
- Stage 2 LR_S=1e−4, warmup100+cosine30K, α=.25, γ=.5, β=0. 손실은 문서의 adaptive reconstruction objective이고 Soft 모방은 없다.
- W112/D123 Teacher, DIFF+UBIL Student, effective batch48/microbatch12, weight decay .01, FP32.
- 모든 pair에 동일 Stage 1 seed261016101, Student seed261018101을 사용한다. 실제 random stream 초기화·데이터 순서도 같은 seed 규약을 유지한다.
- 각 Stage 1의 50K 완료 후 고정 RR validation 집단의 RRE50→RRE90→validation ERGAS→이른 update로 donor를 선택한다. test HQNR로 Stage 1 donor를 고르지 않는다.
- 공식 RR20/FR20은 원본 참조·단일 forward, rotation consistency loss/TTA/SWA/EMA/밴드 가중 loss는 제외한다.

## 우선순위 및 기존 학습 보존

```text
진행 중 SENSOR5 run + 기준 BASE Student 완료
  → S1LOSS9: 9 × (새 Stage 1 + 해당 Student)
  → SENSOR5 남은 LR050 / LR200 / LONG / HARD4
  → WV3 남은 TUNE30
  → WV3 남은 HARD36
```

이미 BASE 이후 run을 시작한 서버라면 그 진행 중 run도 완료한 뒤 경계에서 넘어간다. 완료 run은 반복하지 않는다. 신규 큐에 맞추기 위해 기존 모델을 처음부터 다시 학습시키지 않는다.

기존 SENSOR5/WV3 학습 소스와 admission·checkpoint를 유지한 제어 어댑터를 사용한다. 기존 SENSOR5 자동 watcher만 인계하고, 필요하면 현재 trainer를 완전한 checkpoint로 안전 저장한 뒤 같은 optimizer/RNG/data stream으로 재개하여 BASE 경계까지 마친다. 시간 중간에 SIGKILL하거나 모델만 저장하는 중단은 하지 않는다. old queue가 먼저 WV3를 시작하는 일이 없도록 신규 watcher가 순서를 소유한다.

## 코드 설치와 실행

새 파일은 `panda_s1loss9/` 아래에 격리했다. **실행 중인 기존 `panda_sensor5/` 또는 WV3 source를 덮어쓰지 않는다.** 이전 GF2 목표 수정 commit은 별도 publisher 설치 절차가 있으며 기존 training checkout에 통째로 적용하면 source pin이 달라질 수 있다.

현재 서버에서 준비된 코드 기준 실행은 다음과 같다. Python은 기존 학습·게시 환경의 실행 파일을 사용한다.

```bash
export PYTHONDONTWRITEBYTECODE=1
python -m panda_s1loss9 catalog --server s1 --csv
python -m panda_s1loss9 smoke
python -m panda_s1loss9.queue --server s1 --enqueue
```

s3에서는 `--server s3`로 바꾼다. s3가 아직 mean4/bicubic v1 SENSOR5를 실행 중이면 먼저 [RGB/L4 전환 인계](PANDA_Sensor5_Stage12_Handoff_2026-10-26.md)의 안전 전환을 완료한다. v1 방법의 donor나 calibration을 이번 RGB/L4 계수 탐색에 섞지 않는다. 신규 큐는 등록된 v2 baseline을 필요로 하며 잘못된 버전을 조용히 가져오지 않는다.

활성 v2 checkout에는 전달 commit에서 신규 파일만 추출해 설치할 수 있다. 기존 tracked code를 전환하는 `git checkout`/`pull`은 사용하지 않는다. 최초 설치 대상 `panda_s1loss9`가 이미 있다면 기존 큐 실행 여부와 파일 SHA를 먼저 확인하고, 실행 중 파일에 덮어쓰지 않는다.

```bash
git archive <전달_commit> panda_s1loss9 \
  research_log/PANDA_S1Loss9_Priority_Handoff_2026-10-26.md \
  research_log/PANDA_Method_EquationStyle_2026-10-26.md | tar -x
```

## Sheet 규칙과 판정

- GF2: `GF2-ablations`, HQNR≥.9645 AND ERGAS≤.522.
- QB: `QB-ablations`, HQNR≥.9315 AND ERGAS≤3.5.
- Stage 1은 각 pair의 validation-selected donor 대표. Stage 2는 공동 목표 통과 step 전부와 run별 대표 하나를 게시한다.
- 대표는 HQNR 목표 충족 중 ERGAS 최소, HQNR 충족이 없으면 HQNR 최고다. ERGAS 미달만으로 대표를 없애지 않는다.
- PSL9 namespace로 기존 SENSOR5 결과와 분리한다. 설명에 λ_abs·λ_epsilon, Stage 1 donor pair, full scale, RGB/L4, Stage 2 LR/horizon·α/β/γ가 나타나야 한다.
- 원본 평가·checkpoint SHA·같은 상태의 RR/FR·native cost를 검증한다. 기존 캠페인 행과 원본 기록을 덮어쓰지 않는다.

기존 SENSOR5 GF2 결과의 `.548 → .522` 게시 전환은 [별도 목표 변경 인계](PANDA_GF2_Target0522_Handoff_2026-10-26.md)를 먼저 적용한다. S1LOSS9 publisher는 새 캠페인의 `.522` 판정을 담당하며 기존 v1/v2 통과 행을 대신 정리하지 않는다.

상태와 실행 증거는 `work_dir/panda_s1loss9_launchers/<server>/`의 `status.json`, `chain.json`, `takeover.json`, `through-base.json`, `loss9.json`, `resume-sensor5.json`에 기록한다. 새 실험 기록은 `work_dir/panda_s1loss9/PANDA_S13_S1LOSS9_RGB_L4_STAGE12_20261026_v1/<server>/<dataset>/campaign_state.json`이며, 학습 종료 후에도 남은 SENSOR5/WV3의 원래 결과는 각각의 원래 디렉터리에 남는다.

판단은 Stage 1 RRE 개선만이 아니라 **해당 Student의 두 목표 동시 달성**으로 한다. 실제 RR/FR 보정량, D_s·D_lambda, Stage 1 각 손실과 gradient 관계도 함께 본다. 동일 seed 1개의 탐색이므로 작은 ERGAS 차이를 통계적으로 확정된 우열로 해석하지 않는다. 계수 후보를 좁힌 뒤 해당 donor 조건에서 LR 등 세부 튜닝을 설계한다. 뒤에 보존된 기존 SENSOR5 LR 실험은 원래 donor의 대조이며 신규 최적 donor의 LR 튜닝을 대신하지 않는다.

## 이번 등록 검증과 Git 전달

실제 loss/autograd·낮은 계수 checkpoint 복원·donor/calibration 오연결 거부·Sheet 목표/대표 판정·큐 경계/실제 파일 잠금 경합을 포함한 31개 CPU 테스트를 통과했다. QB와 GF2의 등록 데이터 SHA·train/val 개수 검사도 통과했다. 새 각 run은 시작 전에 실제 GPU update/resume preflight를 별도로 수행하며, 아직 시작하지 않은 run의 GPU preflight를 완료했다고 주장하지 않는다.

s1에서는 48,729 update의 완전한 기존 Stage 1 checkpoint를 확인하고 동일 학습 상태로 재개했다. 재개 후 49,150 update까지 새 train 기록이 이어진 것을 확인했다. 신규 priority watcher는 `WAIT_INITIAL_PAIR`에서 Stage 1 및 BASE 완료를 기다린다. 최초 watcher 인계에서 종료 프로세스의 파일 잠금 해제가 약간 늦는 경합을 발견해 제한된 대기와 회귀 테스트를 추가했고, 첫 시도에서는 trainer에 신호를 보내지 않았다. 검증 상세는 [실행 증거](../results_log/assets/2026-10-26_s1loss9-priority-validation.json)에 보존한다.

전달 브랜치는 `stage1-loss9-priority`다. 현재 GitHub push 인증/권한이 없어 원격 업로드는 수행하지 못한다. 대신 `work_dir/handoffs/2026-10-26_stage1-loss9-priority.bundle`을 함께 준비하며, s3에서 이 파일을 받은 뒤 다음처럼 가져온다. 이 bundle은 기존 v1 commit `6a01bb8`을 전제로 RGB/L4 통일·GF2 목표 변경·이번 추가 큐의 Git 객체를 포함한다.

```bash
git bundle verify /실제/전달경로/2026-10-26_stage1-loss9-priority.bundle
git fetch /실제/전달경로/2026-10-26_stage1-loss9-priority.bundle \
  refs/heads/stage1-loss9-priority:refs/remotes/delivery/stage1-loss9-priority
```

위 신규 파일 설치의 `<전달_commit>`에는 `refs/remotes/delivery/stage1-loss9-priority`를 사용할 수 있다. 기존 활성 checkout을 전환하지 않는다. 인증된 환경에서 브랜치를 push할 수 있다면 bundle 대신 해당 remote branch를 fetch해도 된다. s3에서 실제 적용·학습·Sheet readback을 확인하기 전에는 s3 실행 완료로 기록하지 않는다.
