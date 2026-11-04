# s2 QB Stage2-only 튜닝 구현·인계 — 2026-11-04

승인된 [설계](2026-11-04_qb-s2-frozen-stage2-plan.md)를 `panda_qb_s2_frozen`으로 구현했다.
**기존 s2 Stage1 D4/D2의 validation-selected 48K Aligner·Teacher만 고정 재사용한다.**
새 Stage1 학습, s1/s3 weight 전송, source checkpoint 재선택은 없다.

실행·상태·재개 명령은 [README](../panda_qb_s2_frozen/README.md),
정확한 순서는 [22-run CSV](../config/panda_qb_s2_frozen/cases_s2.csv),
모델·선택 receipt·calibration SHA는 [plan.json](../config/panda_qb_s2_frozen/plan.json)에 있다.

8개 recipe × 3개 Student seed의 24개 비교 중 Q0/Q6 R1 두 baseline은 원본 기록으로 연결한다.
R1의 Q3→Q7→Q1→Q2→Q4→Q5 이후 R2/R3는 각 8개를 순회한다.
22회 모두 fresh Student이며 Stage1 학습 job은 존재하지 않는다.

## 실제 실행 경로

사용자가 push한 뒤 s2에서 **전체 커밋**을 받아 실행 README를 따른다.
기존 s2 출력은 read-only mount, 새 출력은 `work_dir/panda_qb_s2_frozen`이다.
다른 학습이 같은 checkout을 읽는 중이면 별도 worktree에서 실행하며,
원본 donor 디렉터리는 동일 s2 디스크 경로를 연결한다.

```bash
docker build -f Dockerfile.env -t pancrafter-env:qb-s2 .
python3 -m panda_qb_s2_frozen.enqueue --server s2 --image pancrafter-env:qb-s2 --wait
```

실행 컨테이너는 `panda-qb-s2-frozen-v1`이다. 동일 명령은 중복 실행을 피하며,
재개는 `python3 -m panda_qb_s2_frozen.enqueue --server s2 --resume --wait`를 쓴다.
launcher가 source/model/calibration hash와 기준 환경을 검증하고, 각 recipe의 첫 실행 전에
s2 실제 CUDA update/save-resume와 RR/FR 추론을 검사한다.
**이 문서는 s2 원격 학습을 시작했다는 기록이 아니다.**

## 결과·비용·보존

`QB-ablations`, server=s2, 별도 `QFS2` run namespace로 게시한다.
같은 checkpoint HQNR≥.9315 AND ERGAS≤3.5 전부와 run별 대표 하나를 합쳐 게시한다.
대표는 HQNR 충족 중 ERGAS 최저, 없으면 HQNR 최고다.
소수점 3자리 표시와 원본 정밀도 판정을 분리했다.
Approach에는 frozen s2 donor/48K, 두 단계 seed, Student 폭·α·γ·LR·batch를 남긴다.
기존 baseline을 새 결과로 중복 게시하지 않는다.

실제 s2의 W112/W128 배포 구조 비용 측정과 선택 step까지 Train/Eval/Wall 기록을 유지한다.
비용은 같은 배포 구조의 초기화 모델 측정이며, 정확한 학습 checkpoint 측정으로 오기하지 않는다.
모든 1K 평가 weights와 optimizer/RNG/stream 재개 상태를 보관한다.
QB 8·9·10K HQNR이 모두 .90 미만이면 기존 조기 중단을 적용한다.

## 검증 완료 범위

[validation.json](assets/2026-11-04_qb-s2-frozen-implementation/validation.json),
[15개 unittest 원본 출력](assets/2026-11-04_qb-s2-frozen-implementation/unittest.txt)에 기록했다.

- Git index의 커밋 대상 파일만 분리한 환경에서 **15개 테스트 통과**.
- 기존 s2 baseline과 새 consumer의 실제 loss·gradient 정확히 일치. 원본 calibration binding/NPZ 재사용 검증.
- 8개 recipe의 실제 graph/loss 및 frozen A/T 확인. W128 optimizer/RNG/sampler 완전 재개와 A/T hash 불변 확인.
- 전체 22개 순회·완료 skip·잘못된 donor/step/SHA·source 누락·예외 처리·컨테이너 중복 실행/재개 검증.
- 원본 정밀도 경계값, 대표/통과 중복 제거, 3자리 표시 및 게시기 lifecycle 검증.
- s2에 게시된 D4/D2 baseline 설정과 현재 원본 코드의 설정 전 항목 일치 확인.
- clean index export에서 CSV와 registry 일치, **509개 파일 release SHA 검증**.
- 기준 Docker 이미지에서 **GPU와 network 없이** source/environment/evaluator 검증 통과.

실제 s2 파일 존재·CUDA 실행·30K 장시간 성공 여부는 여기서 확인하지 않았다.
s2 launcher/preflight가 해당 검사를 수행한다. 현재 s1 GPU·s1/s3 큐·기존 s2 실행 코드는 변경하지 않았다.
