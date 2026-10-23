# Sampling30: Student 다운·업샘플 구조 비교

> 후속 사용자 결정으로 미시작 구조 큐는 [NEXT30](../panda_next30/README.md)으로 통합했다. 아래는 이전 bicubic 계획의 기록이다.

캠페인은 `PANDA_S13_SAMPLING_HARDONLY_20261023_v1`이며 run prefix는 `PSA30`이다. 현재 Schedule30 큐가 정상 완료된 뒤, 양의 PAN Aligner 보정을 유지하면서 Student의 다운·업샘플 연산만 비교한다. **서버별 6개 유한 큐이며 총 시간 제한은 없다.** 기존 실행을 중단하거나 skip하지 않고, 현재 캠페인의 코드·설정·기록도 변경하지 않는다.

## 고정 조건과 큐

두 서버 모두 WV3, W112·D123 Student, `interp23`, FP32, AdamW, 유효 batch 48을 사용한다. 동결 PAN Aligner의 예측 `u`를 **scale 0.2**로 적용하며 PAN warp는 기존 bicubic이다. Teacher와 Aligner는 각 서버의 기존 mature BEST_VAL_RRE donor를 재사용한다. 모든 Student·optimizer는 새로 초기화하고 **30,000 update**, warmup **100**, peak LR **1e-4**, 종료 LR **0**의 cosine을 사용한다. 기존 Hard 계수는 **α=0.25, γ=0.5, β=0**이다.

밴드 가중 loss와 rotation consistency loss는 **사용하지 않는다**. Student 단일 view, 기존 C4 입력 증강, 원본 RR20/FR20 평가를 유지하며 공식 평가에 TTA·앙상블을 사용하지 않는다. FFT 등 warp 변경은 별도 검토 대상으로, 이번 구조 비교에 섞지 않는다.

| 순서 — s1·s3 공통 | Case | 반복 | 변경 연산 | Student 파라미터 |
|---:|---|---|---|---:|
| 1 | DAVG | R1 | down 두 곳: 2×2 평균 pooling → 3×3 stride-1 conv | 2,658,888 |
| 2 | UBIL | R1 | up 두 곳: bilinear ×2 → 3×3 conv | 2,784,328 |
| 3 | DAVG_UBIL | R1 | 위 down·up 변경 모두 적용 | 2,784,328 |
| 4 | DAVG | R2 | DAVG 반복 | 2,658,888 |
| 5 | UBIL | R2 | UBIL 반복 | 2,784,328 |
| 6 | DAVG_UBIL | R2 | DAVG_UBIL 반복 | 2,784,328 |

R1/R2 seed는 각각 `261018101`/`261018102`다. 기본 up은 2×2 stride-2 transposed conv, 기본 down은 3×3 stride-2 padding-1 conv다. 새 bilinear는 `align_corners=False`다. Teacher 구조는 바꾸지 않는다. 학습 가능한 convolution은 방향별로 자유로우므로 이 변경이 네트워크 전체의 회전 등변성을 보장하지 않는다.

대조군은 **동일 서버·동일 seed**의 기존 `PANDA_S13_SCHEDULE_HARDONLY_20261023_v1/S020C30 R1·R2`다. 새 BASE 학습을 중복 실행하지 않으며 대조군 가중치도 가져오지 않는다. 전환 시 두 대조군의 30K 완료, 전체 등록 설정, checkpoint·평가 SHA, 동일 donor 및 데이터·증강 stream을 검증한다. 시작 후 실제 로드한 donor의 A/T tensor와 학습 inventory 해시도 대조군에 맞춰 확인한다. 대조군 행은 기존 Sheet에 유지한다. 서버마다 donor가 다르므로 s1과 s3의 절대 수치를 직접 차감하지 않는다.

## 실행

해당 서버의 기존 torch·google-auth·requests Python 환경에서 실행한다. 필요하면 `PAN_HOST_PYTHON`, `PAN_SAMPLING30_IMAGE`, `PANCRAFTER_DLPAN`을 기존 로컬 환경으로 지정한다. 기본적으로 직전 Schedule30의 검증된 이미지를 재사용하며 image pull은 하지 않는다. 인증 파일은 출력하거나 Git에 넣지 않는다.

```bash
python -m panda_sampling30 catalog --server s1 --csv
python -m panda_sampling30 smoke
bash scripts/start_panda_sampling30_transition.sh s1
```

s3에서는 다음 명령을 **s3 에이전트가 해당 서버에서** 실행한다. SSH 원격 실행을 전제로 하지 않는다.

```bash
python -m panda_sampling30 catalog --server s3 --csv
python -m panda_sampling30 smoke
bash scripts/start_panda_sampling30_transition.sh s3
```

전환 watcher는 기존 Schedule30 전체 큐가 정상 완료될 때까지 기다린다. 중단 신호나 pending skip을 보내지 않는다. 기존 실행이 안전 정지 상태이면 재개를 기다리고, 비정상 종료·불완전 이력·대조군 누락은 자동 우회하지 않는다. 정상 완료 후 대조군 증거 검증 → GPU idle 확인 → 격리 cost 측정 → 실제 데이터 CUDA 학습·checkpoint 복원·추론 preflight → 새 trainer·publisher 시작 순서다.

**실제 데이터 CUDA 검증과 새 학습은 기존 큐 완료 뒤 실행된다.** CPU 검증이나 watcher 등록을 GPU preflight 통과로 해석하지 않는다. 등록·대기·학습 중에는 pin 대상 코드·스크립트·registry·평가기 파일을 수정하지 않는다. 기존 소스가 바뀌면 혼합 실행을 거부한다. 두 서버 모두 WV3이므로 mount CLI는 역사적 sensor helper의 `s1=WV3` 선택을 사용한다. 해당 helper에 물리 서버 이름 `s3`를 넘기면 GF2가 선택되므로 바꾸지 않는다.

## 상태와 재개

- 전환 기록: `work_dir/panda_sampling30_launchers/<server>/transition_receipt.json`
- 캠페인 root: `work_dir/panda_sampling30/PANDA_S13_SAMPLING_HARDONLY_20261023_v1/<server>/WV3`
- 실행·검증: root의 `campaign_state.json`, `admission.json`, `preflight/passed.json`, 각 run의 `train.jsonl`
- 게시 확인: root의 `sheet_publish_panda_sampling30/last_poll.json`에서 `readback_verified=true`

```bash
docker logs --tail 40 panda-sampling30-hardonly-wv3-s3-v1
```

안전 정지 후 같은 코드·설정·이미지이고 GPU가 비어 있음을 확인한 경우 기존 컨테이너를 유지한 채 재개한다. 이미 실행 중이면 중복 실행하지 않는다.

```bash
docker start panda-sampling30-hardonly-wv3-s3-v1
bash scripts/start_panda_sampling30_best_publisher.sh s3
```

게시기만 재시작할 때는 `bash scripts/restart_panda_sampling30_publisher.sh s3`를 사용한다. s1은 인자를 바꾼다. 실패·poisoned state·SHA 불일치 시 원본을 덮어쓰거나 새 초기화로 우회하지 않는다.

## Sheet와 비용

기존 `WV3-ablations` 탭에 동일 checkpoint의 반올림 전 지표로 게시한다.

1. **HQNR ≥ 0.9585와 ERGAS ≤ 2.040**을 함께 만족하는 모든 checkpoint를 게시한다.
2. run/seed/attempt별 대표는 HQNR ≥ 0.9585 중 **최소 ERGAS**다. 대표에는 ERGAS 상한을 요구하지 않는다.
3. HQNR 조건을 만족한 후보가 없으면 **최대 HQNR → 최소 ERGAS → 이른 step** 순으로 대표를 유지한다. 자격을 만족한 대표의 ERGAS 동률은 HQNR 최고 → 이른 step 순이다.
4. 대표와 공동 통과 행이 같은 checkpoint이면 중복하지 않는다. 고정 final30K 행을 별도로 추가하지 않으며, Stage1 BEST_VAL_RRE 게시도 유지한다.

정책 ID는 `PANDA_SAMPLING30_RRE_STAGE2_H9585_PASS_ALL_OR_BEST_FALLBACK_V1`이다. `PSA30` 캠페인·서버 소유 행의 A:BL만 갱신하고 기존 결과·다른 서버·오른쪽 추가 열은 보존한다. 실제 A+S 배포 경로의 비용을 각 서버에서 구조별로 격리 측정하고 signature가 일치하는 결과만 연결한다. 미측정 값이나 다른 서버의 비용으로 대체하지 않는다. 구조 변경이 포함되므로 기존 BASE 비용을 UBIL 등에 그대로 재사용하지 않는다. 이번 fresh 학습량은 run당 30K이며 원 donor·대조군 학습 비용을 다시 계산하지 않는다.

이 문서는 실행 인계이며 실제 시작 또는 목표 달성 확인서는 아니다. [s3 인계](../results_log/2026-10-23_SAMPLING30_s3_handoff.md), [등록 정의](contract.py), [큐](plan.py)를 함께 참조한다.
