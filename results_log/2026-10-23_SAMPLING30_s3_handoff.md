# 2026-10-23 Sampling30 s3 실행 인계

사용자가 Student 다운·업샘플 구조 실험을 승인했다. **밴드 가중 loss는 제외한다.** 새 캠페인은 `PANDA_S13_SAMPLING_HARDONLY_20261023_v1`, prefix는 `PSA30`이다. 현재 `panda_schedule30` 전체 큐가 정상 완료될 때까지 기다렸다가 후속 실험을 실행한다. 기존 실험의 중단·skip·deadline 초기화는 이번 인계에 포함하지 않는다.

## s3 등록 큐

| 순서 | Case | 반복 | 변경 |
|---:|---|---|---|
| 1 | DAVG | R1 | down 두 곳을 2×2 평균 pooling + 3×3 conv로 변경 |
| 2 | UBIL | R1 | up 두 곳을 bilinear ×2 + 3×3 conv로 변경 |
| 3 | DAVG_UBIL | R1 | 두 변경 모두 적용 |
| 4 | DAVG | R2 | 두 번째 seed |
| 5 | UBIL | R2 | 두 번째 seed |
| 6 | DAVG_UBIL | R2 | 두 번째 seed |

s1도 같은 6개를 실행하되 각 서버의 donor를 사용한다. R1/R2 seed는 `261018101`/`261018102`다. 공통 조건: WV3, fresh Student·AdamW, **30K**, warmup **100**, peak LR **1e-4 → 0**, Hard **α=.25·γ=.5·β=0**, PAN Aligner **scale=.2**. PAN warp는 기존 bicubic, Teacher·Aligner는 동결이다. rotation loss·밴드 가중 loss·TTA 없이 단일 Student view와 기존 C4 입력 증강을 유지한다.

대조군은 s3 Schedule30의 **S020C30 R1/R2**다. 두 대조군이 30K까지 완료되어야 한다. BASE를 중복 학습하지 않고, 대조군의 가중치도 가져오지 않는다. 전체 등록 설정과 동일 donor·데이터·증강 stream을 검증하며 checkpoint·평가 SHA를 새 admission에 고정한다. s1 파일로 대체하지 않는다. 정확한 구조·비용·게시 규칙은 [Sampling30 README](../panda_sampling30/README.md)에 있다.

## Git 전달 후 s3에서 실행

이 문서가 포함된 commit을 받아 새 파일과 공유 의존성이 일치하는지 확인한다. 기존 로컬 수정·실행 소스·credential을 덮어쓰지 않는다. 기존 환경의 `gspread/server.txt=s3`, WV3 데이터, DLPan, mature RRE donor receipt와 checkpoint, Schedule30 결과를 유지한다. `work_dir`와 credential은 Git으로 전달하지 않는다.

기존 torch·google-auth·requests Python을 사용한다. 필요하면 `PAN_HOST_PYTHON`과 `PAN_SAMPLING30_IMAGE`를 기존 로컬 Python·검증된 이미지로 지정한다. 이미지 자동 갱신은 하지 않는다.

```bash
python -m panda_sampling30 catalog --server s3 --csv
python -m panda_sampling30 smoke
bash scripts/start_panda_sampling30_transition.sh s3
```

마지막 명령은 wait-only watcher를 등록한다. 아직 Schedule30이 시작되지 않았으면 시작을 기다리고, 실행 중이면 전체 큐 완료를 기다린다. 안전 정지 상태의 기존 큐는 별도로 같은 코드·설정으로 재개해야 한다. watcher는 기존 trainer에 신호를 보내거나 pending control을 쓰지 않는다.

완료 후 **대조군 검증 → GPU idle → 서버별·구조별 실제 cost 측정 → 실제 데이터 CUDA 학습·복원·추론 검증 → 새 학습과 게시기**를 실행한다. CUDA 검증은 기존 큐가 끝난 뒤 진행되므로 이 인계만으로 GPU 검증 통과를 선언하지 않는다. 두 서버 모두 WV3이며 mount CLI 내부의 역사적 `s1=WV3` 선택은 물리 서버 s3와 무관하다.

전환 대기·실행 중에는 pin 대상 소스·스크립트·registry·평가기 파일을 수정하거나 Git으로 갱신하지 않는다. code pin·SHA 불일치는 기록을 유지하고 원인을 확인한다.

## 확인과 재개

```bash
docker logs --tail 40 panda-sampling30-hardonly-wv3-s3-v1
```

- 전환: `work_dir/panda_sampling30_launchers/s3/transition_receipt.json`
- 캠페인: `work_dir/panda_sampling30/PANDA_S13_SAMPLING_HARDONLY_20261023_v1/s3/WV3/`
- 검증: 캠페인 root의 `preflight/passed.json`, `admission.json`, `campaign_state.json`
- 게시: root의 `sheet_publish_panda_sampling30/last_poll.json`, `readback_verified=true`

새 캠페인이 안전 정지한 후 같은 코드·설정·이미지이고 GPU가 비어 있을 때만 기존 컨테이너를 재개한다.

```bash
docker start panda-sampling30-hardonly-wv3-s3-v1
bash scripts/start_panda_sampling30_best_publisher.sh s3
```

게시기만 재시작하려면 `bash scripts/restart_panda_sampling30_publisher.sh s3`. 실행 중인 컨테이너를 중복 생성·삭제하지 않는다.

## Sheet 규칙

`WV3-ablations`에서 동일 checkpoint의 전체 정밀도로 **HQNR ≥ .9585 AND ERGAS ≤ 2.040**인 결과를 모두 게시한다. 각 run/seed/attempt 대표는 **HQNR ≥ .9585 중 최소 ERGAS**이며 대표에는 ERGAS 상한이 없다. HQNR 충족 후보가 없으면 **최고 HQNR → 최소 ERGAS → 이른 step**으로 대표를 유지한다. 대표·통과 행이 겹치면 하나만 게시하며 별도의 final30K 행은 추가하지 않는다. Stage1 BEST_VAL_RRE는 유지한다.

새 정책은 `PANDA_SAMPLING30_RRE_STAGE2_H9585_PASS_ALL_OR_BEST_FALLBACK_V1`이다. 새 `PSA30` 소유 행의 A:BL만 갱신하고 기존 대조군·다른 서버 결과·추가 열은 유지한다. 비용은 s3에서 실제 A+S 배포 경로를 구조별로 측정한 sidecar만 사용하며, 미측정값을 추정하거나 s1 비용으로 채우지 않는다. 대조군·donor 학습 시간을 새 30K 학습에 합산하지 않는다.

실행 후 s3 에이전트는 사용 commit, transition 상태, 실제 첫 학습 update, GPU preflight, Sheet readback을 실행 기록에 남긴다. 이 인계에는 아직 s3 새 학습 시작이나 목표 달성 결과가 없다.
