# WV3 s1 재시작 / QB s2 새 main 실험 인계 — 2026-11-13

## 사용자 확정 범위

- s1은 현재 WV3 실험을 중단하고 해당 캠페인의 결과·Sheet 기록을 삭제한 뒤 새 구현으로 2회 반복한다.
- s2는 RGB Aligner 1회 + RGBN Aligner 1회, 이후 각 그룹에 같은 Student 5설정씩 총 10회를 실행한다.
- s2는 자기 서버에서 처음부터 Stage 1을 학습한다. 다른 서버 weight 전달·기존 donor 재사용 없음.
- Git commit까지 준비하며 push는 사용자가 수행한다.

## 완료한 이전 캠페인 정리

`PANDA_WV3_RGBN_L4_S1_20261113_v1`의 컨테이너를 정상 정지 후 제거했다. 해당 캠페인 폴더 185파일 / 1,896,349,683 bytes를 삭제했다.

`WV3-ablations` 9966–9971행의 A:BL 값과 메모를 삭제하고 빈 행을 숨겼다. 타 서버의 게시 행 번호를 유지하려고 물리적 행 이동은 하지 않았다. 인접 s3 캠페인과 과거 결과는 보존했다.

삭제 영수증: `results_log/assets/2026-11-13-unified-restart/deletion_receipt.json`.
과거 소스 패키지는 그대로 남겼으며 직전 실행의 소스 snapshot도 `results_log/assets/2026-11-13-architecture-notation-review/active_source_ea29df0e5f9d.tar.gz`에 보존했다.

## 새 구현

실행 패키지: `panda_unified_campaign`; 공통 연산: `panda_unified/core.py`.

1. Teacher와 Student는 같은 width/D123/UBIL backbone을 사용한다. Down은 3×3 stride2 convolution, up은 bilinear×2(`align_corners=False`)+3×3 convolution이다. WV3 W112, QB W128.
2. Teacher 입력은 corrected PAN + interp23 MS base, Student는 corrected PAN + (corrected PAN−original PAN) + interp23 MS base다. WV3 9/10채널, QB 5/6채널 차이는 방법의 의도된 입력 차이다.
3. Stage 1 correction·ε injection·validation, Stage 2 Teacher/Student·calibration·RR/FR 추론의 PAN warp는 모두 동일 Lanczos-4이다. Scale1 유지, shift clamp 없음. MS interp23와 U-Net UBIL은 PAN warp와 구분한다.
4. 모든 pixel loss 및 τ_e·g·q calibration은 **Lanczos 8-tap 관측 support를 3×3 erosion한 공통 영역**을 사용한다. 예시 64×64에서 55×55=3025 pixels. 실제 보간과 마스크가 모두 `int64 pixel + floor(shift).int64`로 참조 범위를 결정한다.
5. L_spa·q/g·validation/test RRE의 edge reference를 그룹 내 통일한다. WV3 RGBN=[4,2,1,6](NIR1), QB RGB=[2,1,0], RGBN=[2,1,0,3]. 동일 가중 평균 후 Scharr. Aligner 입력과 복원 error는 전체 MS 밴드를 유지한다.
6. β=0, scale1, 회전 consistency/TTA/SWA 없음. 기존 paired flip/rot90 augmentation은 유지한다.

Aligner 입력의 clean crop8과 RRE 탐색·Gaussian stencil에 필요한 metric margin은 학습 loss ROI와 목적이 다르며 보존한다. 공식 RR20/FR20 품질 지표의 비교 영역도 바꾸지 않는다. Validation ERGAS의 tie-break는 새 공통 mask에서 계산하므로 새 selector protocol로 기록한다.

기존 ROI8 support 검사로 생기던 예측 shift의 암묵 범위 제한은 새 학습 영역에 적용하지 않는다. 빈 유효 마스크는 오류로 처리한다. RRE의 고정 탐색 support를 벗어난 샘플은 검증에서 실패 penalty 12px로 계산하며 샘플을 임의로 빼지 않는다.

## s1 WV3

| 순서 | 실행 | Stage | seed | steps | 설정 |
|---:|---|---|---:|---:|---|
| 1 | RGBN R1 | 1 | 261016101 | 50K | λ_spa=2, λ_ε=.003, T LR1e-4/A LR1e-5 |
| 2 | Ours R1 | 2 | 261018101 | 30K | α=γ=.25, LR1.1e-4 |
| 3 | RGBN R2 | 1 | 261016102 | 50K | R1과 동일 |
| 4 | Ours R2 | 2 | 261018102 | 30K | R1과 동일 |

각 Student는 자기 반복의 새 Stage 1을 사용한다. 목표 HQNR≥.9585, ERGAS≤2.040.

## s2 QB

성공 기준: `QB S2 Frozen Q2_W128 R1`, 23K, HQNR **.9389469340058131**, ERGAS **3.495269912500685** (`QB-ablations` 105행, 2026-11-13 확인). 원본 설정 근거는 `assets/2026-11-13-unified-restart/qb_baseline_evidence.json`에 저장했다. 기존 R2/R3는 목표 동시 통과가 아니었으므로 seed 안정성이 확정된 결과라고 주장하지 않는다.

Stage 1 RGB·RGBN 각각 W128/UBIL, λ_spa=4, λ_ε=.03, ε radius3, 50K, seed261016301, T LR1e-4/A LR1e-5. 두 Stage 1 완료 후 다음 순서를 RGB→RGBN 짝으로 순회한다.

| 우선순위 | Stage 2 설정 | α | γ | LR | 실행 수 |
|---:|---|---:|---:|---:|---:|
| 1 | Q2BASE | .25 | .50 | 1e-4 | RGB/RGBN 각1 |
| 2 | H050 | .125 | .25 | 1e-4 | RGB/RGBN 각1 |
| 3 | G025 | .25 | .25 | 1e-4 | RGB/RGBN 각1 |
| 4 | LR090 | .25 | .50 | 9e-5 | RGB/RGBN 각1 |
| 5 | LR110 | .25 | .50 | 1.1e-4 | RGB/RGBN 각1 |

모든 Student W128/UBIL, 30K, seed261018101. 위 10개는 독립 seed 10반복이 아니라 **5개 설정의 paired RGB/RGBN 비교**다. 목표 HQNR≥.9315, ERGAS≤3.5. 8·9·10K HQNR이 모두 .90 미만이면 10K에서 해당 Student를 중단하고 다음 case로 간다.

공통 AdamW WD=.01, β=(.9,.999), optimizer ε=1e-8, batch48/micro12, warmup100, cosine→0. 기존 성공 Q2의 Teacher W112/BASE에서 새 W128/UBIL 및 공통 mask로 바뀌었으므로 성공 레시피에 기반한 재검증이며 동일 수치의 재현 보장은 아니다.

## 보관·게시·실행

- 모든 1K 평가 모델을 보관하고, optimizer/stream/RNG가 포함된 complete resume은 최신2개를 유지한다. 이번 새 캠페인 초기 결과를 예전 목표 미달 정리 규칙으로 자동 삭제하지 않는다.
- Stage 1 donor는 같은 서버·mode·반복의 validation RRE50→RRE90→ValERGAS→빠른 step 순으로 선택한다. Test는 donor 선택에 사용하지 않는다.
- Exact calibration은 전체 unaugmented RR train의 공통 유효 픽셀을 사용한다. QB용 CPU 예산8GiB, GPU 통계1GiB로 명시하며 cache는 donor·reference·mask·source hash로 분리한다.
- 게시: `WV3-ablations` / `QB-ablations`, 소수점4자리. HQNR 기준을 충족한 checkpoint 중 ERGAS 최소 대표, 미충족이면 HQNR 최고; 목표 동시 통과 step과 고정29K/50K 비교도 기록한다. 새 graph의 미측정 inference cost는 다른 모델 값으로 채우지 않는다.
- 새 캠페인 ID 및 전체 목록은 `config/panda_unified_campaign/plan.json`, `cases_s1.csv`, `cases_s2.csv`가 원본이다.

s2에서 사용자 push 이후:

```bash
git pull --ff-only
python3 -m panda_unified_campaign.release
python3 -m panda_unified_campaign catalog --server s2
python3 -m panda_unified_campaign.enqueue --server s2 --wait
```

구버전 저장소라도 commit 전체를 pull하고 release가 검증되면 동일한 소스가 사용된다. 기존 로컬 변경으로 pull이 충돌하면 해당 변경을 보존해 통합하며, 강제 reset하지 않는다. Dataset과 Sheet 인증은 서버 로컬 자원이다. 기본 이미지 `hojunqueen/pancrafter-env:torch2.4.0-cu118`; 필요할 때 `PANCRAFTER_DLPAN`으로 local toolbox 경로를 전달한다. 타 GPU 작업은 `--wait`로 기다린다. 재개는 동일 컨테이너에 `--resume`을 사용한다.

## 검증 및 실행 확인

- CPU regression 19개 통과: corrected support indexing·최대 공통 mask·shift 미분·4/8band·UBIL 초기화·RGB/RGBN binding·가변 calibration·실제 donor 로딩·frozen backward·complete resume.
- s1 및 s2 설정의 Docker environment/data/source/Sheet dry run 검증 통과. s2 검증도 이 서버에서 disposable하게 수행했으며 s2 실제 학습 실행을 뜻하지 않는다.
- CUDA Stage 1의 full batch update, 정확한 checkpoint 경계 복원, 다음 update replay 및 RR/FR native Lanczos 경로 검증 통과.
- 실제 s1 실행 상태 및 최종 source SHA는 완료 시 아래에 덧붙인다.

### 실행 확인

- 2026-11-13 11:50 KST: 컨테이너 `panda-unified-s1-v1`에서 s1 Stage 1 R1 실행 중, 확인 시 400 step. 나머지 3개는 QUEUED.
- 새 s1 Sheet 행: `WV3-ablations` **9975–9978**. 최초 게시 및 원격 readback 검증 완료. 첫 정량 평가는 1K에서 게시된다.
- 최종 source SHA: `04b1e84c7b694b59ed5e918d08a5d73c2ed45f8d50ccfcd63fa58258a39f9542`. 167개 소스/설정 파일의 Git index 내용까지 hash 일치를 확인한다.
- s2는 원격 학습을 시작하지 않았다. 위 push/pull/launch 절차로 실행한다.
- 전체 검증·게시 영수증: `results_log/assets/2026-11-13-unified-restart/verification.json`.
