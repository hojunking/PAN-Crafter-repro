# Stage 2: GT-always Hard + selective Teacher Soft + conditional-q

구현 근거: `research_log/PANDA_Stage2_Loss_Implementation_Spec_2026-10-16.md`.
현재 `donor24` Stage 1을 수정하지 않는 독립 namespace다. 패키지 자체의 diagnostic은
정식 50K 캠페인이 아니다. 사용자 승인 40h 캠페인은 별도 실행기/계획을 따른다.

40시간 정식 실행: `bash scripts/run_panda40_docker.sh s3 --replace-donor24`.
`panda40` 실행기가 로컬 donor 선택/재평가, calibration, 두 seed, 비용 및 시트
업로드를 연결한다. 상세 큐는
`research_log/PANDA_S13_40H_S2D_Stage2_2026-10-16.md`를 따른다.

## 2026-10-16 사용자 승인: MASK2 주 비교 + full-image 대조

아래 기존 명세의 full-image 구현은 기본값으로 보존한다. 새로운 40h 실험에서는
`Stage2Spec(..., mask_policy=MASK2_POLICY)`로 **별도 버전**을 명시한다.

- `MASK2_POLICY = "DETACHED_BICUBIC4_SUPPORT_PER_SAMPLE_MEAN_V1"`: Stage 1과 같은
  `warp_support_mask`/`donor30.masking.masked_mean`을 재사용한다. correction은 detach;
  bicubic의 네 sampling tap이 원본 PAN 안에 있는 픽셀만 사용한다.
- **Hard와 Soft 모두** 밴드 평균 → 샘플별 valid-pixel 평균 → batch 평균이다.
  valid 픽셀이 많은 샘플의 학습 비중이 커지지 않으며 microbatch accumulation도 같다.
- MASK2의 `tau_e`는 전체 unaugmented RR train의 **유효 픽셀 전역 중앙값**이다.
  샘플별 중앙값을 평균하지 않는다. full-image `tau_e`와 별도 artifact를 생성한다.
- q/g는 동일 fixed ROI8이다. ROI8의 Scharr stencil 전체가 유효해야 하며, 이를
  만족하지 못하면 실패한다. 유효하지 않은 ROI 부분을 조용히 삭제하지 않는다.
- `PANDA_STAGE2_CONDQ_MASK2_V1` / `PANDA_S2_CONDQ_TRAIN_VALID_SUPPORT_V1`로 식별하며
  full-image calibration 및 checkpoint와 교차 사용하면 실패한다. 기존 donor30의
  옛 E/EQ용 calibration 또한 사용할 수 없다.
- **공식 평가와 추론은 마스킹하지 않고 그대로 유지**한다. MASK2도 GT-only는
  Teacher/calibration을 만들지 않으며, 다른 두 variant와 같은 frozen Aligner를 쓴다.

Python API: `runtime.calibrate(..., mask_policy=MASK2_POLICY)` 후
`CalibrationStats.load(..., expected_mask_policy=MASK2_POLICY)`;
`ConditionalQStage2`가 loss/calibration policy 일치를 재검증한다.
기존 아래 CLI는 full-image 기본 diagnostic이다. MASK2 캠페인은 새 실행기가
위 명시적 API를 사용한다.

## 현재 Stage 1과 연결

- 같은 저장 checkpoint에서 DirectPANAligner A / W112-D123-P0 Teacher T를
  SHA256 및 실제 tensor hash로 검증하여 읽는다. checkpoint 자동 선정 없음.
- A에는 원본 PAN + **stored LMS**가 들어간다. PAN을 current-PAN-pixel 단위
  correction으로 **한 번** bicubic/border warp한다. RR/FR 모두 shift ×4 없음.
- T/S는 동일 aligned PAN, LRMS, stored LMS 객체를 받는다. residual base는
  stored LMS가 아니라 Reconstructor 내부 **bicubic LRMS**다.
- Student는 W112-D123-P0/BICUBIC_MS, `role=S` fresh seed로 만든다.
  Teacher 복사·파라미터 공유·LPAN/HPAN·trainable Student Aligner가 없다.
- 현재 Stage 1 donor는 MASK2다. 원래 Stage 2 명세의 full-image 대조는
  **Hard/Soft만 전체 픽셀 감독**으로 바뀐다. MASK2 주 비교는 위 승인 규약이다.
  어느 경우에도 Stage 1의 학습 이력/마스크를 소급 변경하지 않는다.
- 기존 `donor30` E/EQ loss·`tau_e/tau_q` artifact·W104 Student를 재사용하지 않는다.

## 손실과 분기

다음 수식은 원래 **full-image 대조**다. MASK2에서는 두 `mean_full_BHW`를
위의 같은 샘플 비중을 갖는 valid-pixel reduction으로 치환한다.

`e_s=mean_C|S-GT|`, `e_t=mean_C|T-GT|`, `delta_ts=mean_C|S-T|`.

```
d_e = stopgrad(e_t / max(e_t + tau_e, FP32_tiny))
a_T = stopgrad(max(e_s - e_t, 0) / (e_s + delta))
w_H = 1 + alpha*d_e + gamma*d_e*r_q_bar
Hard = mean_full_BHW(w_H * e_s)
Soft = mean_full_BHW((1-d_e) * a_T * delta_ts)
loss = Hard + beta*Soft
```

`r_q_bar`는 ROI8 밖에서 0이다. **GT/Soft는 경계에서도 살아 있고**, warp-valid
mask가 reduction의 분자/분모에 들어가지 않는다. ROI8 Scharr stencil이 border
extension을 읽으면 세 variant 모두 update 전에 실패한다. adaptive mask나 shift
clamp로 숨기지 않는다.

| variant | Teacher | calibration | q/g/CDF |
|---|---|---|---|
| `gt_only` | 객체/forward 없음 | 없음 | 없음 |
| `panda_e` | frozen | 새 full-image tau_e | 실행 없음 |
| `panda_condq` | frozen | 새 tau_e + exact conditional ECDF | Hard weight에만 사용 |

추론은 **A+S만** 호출한다. GT·Teacher·calibration·CDF에 접근하지 않는다.

## Calibration 계약

`RegisteredRRTrain`은 등록된 train H5 전체 inventory, 실제 파일 SHA, stored LMS,
sensor·band·radiometry를 묶는다. validation/test/FR, 중복/누락 sample, online
augmentation은 calibration에 들어갈 수 없다.

- `tau_e`: 모든 train patch의 **전체** 픽셀, 짝수 중앙값은 두 중앙값 FP64 평균.
- `g/q`: FP32 RGB-mean-first absolute Scharr `/32`, reflect1, ROI8만 수집.
- g 전역 linear quantile 경계는 FP64. 중복 경계를 제거하고 빈 bin을 결정적으로
  병합한다. 경계와 같은 g는 상위 bin(`right=True`)이다.
- bin별 **전체 sorted FP32 q 표본**을 저장한다. ECDF는 `count(q_i <= q)/N`.
  zero-q ties가 많으면 rank가 0이 아닐 수 있다.
- NPZ 하나에 metadata+배열을 원자적·exclusive 저장한다. content SHA와 file SHA를
  구분하며 읽을 때 둘 다 검증할 수 있다. 이전 artifact를 덮어쓰지 않는다.
- 메모리 예산 부족은 오류다. LUT/부분 표본/CPU-GPU fallback을 암묵적으로 하지 않는다.
  GPU 이동은 `stage.move_calibration(device, memory_budget_bytes=...)`로 명시한다.

현재 WV3 train 9,714 patches 기준 `n_e=39,788,544`, `n_q=22,381,056`.
exact q 테이블 약 85.38 MiB, builder의 보수적 host 예산 계산 약 1.64 GiB다.
이는 메모리 **계산값**이며 최종 donor의 전체 calibration/GPU 처리량 실측은 아직 없다.

## 학습·재개·평가

`StudentTrainer`는 Student 90개 parameter tensor만 AdamW에 넣는다. TF32/AMP를
사용하지 않는다. optimizer 값·moment 누락/변조·NaN·scheduler/seed/source/calibration
불일치·옛 checkpoint schema를 거부한다. microbatch 실패 시 누적 gradient를 버리고,
optimizer mutation 후 실패하면 trainer를 poisoned 상태로 만들어 완전 복구를 요구한다.

기본값은 현재 실제 Stage 1과 같은 LR1e-4, AdamW WD.01, warmup100+cosine50K,
**microbatch12 × accumulation4 = 48**, H/V flip probability **1.0**, random rot90다.
문서의 microbatch4 × 12 예시는 수학적으로 유효하지만 현재 Stage 1 기본값과 다르다.
다른 그룹 크기나 augmentation은 명시적으로 바꾸고 공통 실험조건으로 기록해야 한다.

`TrainStream`은 paired data/augmentation seed, shuffled no-drop 순서, cursor/RNG를
저장·검증한다. 완전 resume은 `load_checkpoint(..., trainer)`가 반환한 `stream_state`를
동일 계약의 `TrainStream.load_state_dict()`에 복구해야 한다.

학습 로그에는 loss 분해, detach weight 통계, ROI rank/bin, correction/support,
Student gradient norm 및 주기적 Hard 대 beta-Soft gradient norm/cosine을 기록한다.
전체 유효 batch의 p95/quantile을 계산하며 microbatch별 p95의 평균을 쓰지 않는다.

`evaluation.official_evaluate`는 기존 metric 함수를 그대로 사용한다:
RR20/dim_cut21, FR20/full512, **original PAN/native LMS** 참조. 같은 S/A state의
RR/FR를 함께 측정하고 HQNR·SCC·ERGAS를 출력한다. 2026-10-16 사용자 수정에 따라
**HQNR 목표를 충족한 후보 중 ERGAS 최소**로 선택한다. 현재 WV3 목표는 **HQNR ≥ 0.959**.
목표 미달 후보만 있으면 HQNR 최대를 탐색용으로 선택한다. SCC는 기록만 하고 순위에
절대 사용하지 않는다. 목표는 반올림하지 않은 scene-mean HQNR에 적용한다.
동률은 통과군에서 HQNR, 미달군에서 ERGAS, 이후 이른 update로 결정한다.
`checkpoint_rank`는 최대값 선택이고 서로 다른 목표를 비교할 때는
`expected_policy=trainer.selection_policy`를 전달해야 한다. 목표는 checkpoint·평가·resume
계약에 고정되며 변경 시 기존 resume은 거부된다. 비-WV3 목표는 명시적으로 입력한다.
공식 test-aware selection임을 기록하며 독립 holdout으로 주장하지 않는다.
synthetic metric 결과는 실제 checkpoint 선정에 사용할 수 없다.

## 명령

Python은 기존 `pancrafter` 환경을 사용한다. 아래 점검은 GPU/클라우드/실제 데이터에
접근하지 않는 synthetic CPU tests다.

```bash
python -m panda_stage2_condq smoke --report-dir work_dir/panda_stage2_condq/validation
```

아래 `<...>` 값은 **승인된 donor/설정으로 직접 대체**해야 한다. `inspect-donor`는
파일 연결 확인일 뿐 donor를 선정하거나 Stage 1을 변경하지 않는다.

```bash
python -m panda_stage2_condq inspect-donor --sensor WV3 \
  --checkpoint '<trusted local donor.pt>' --checkpoint-sha256 '<full file SHA256>'

python -m panda_stage2_condq calibrate --sensor WV3 \
  --checkpoint '<same donor.pt>' --checkpoint-sha256 '<same full SHA256>' \
  --kind condq --k '<approved K>' --batch-size 12 --memory-gib 2 \
  --device cpu --output '<new calibration.npz>'
```

tau-only artifact는 `--kind e`. condq artifact를 `panda_e`에도 읽을 수 있지만 q/g는
실행하지 않는다. `gt_only`는 calibration 인수를 주면 오류다.

```bash
python -m panda_stage2_condq diagnostic --sensor WV3 \
  --checkpoint '<same donor.pt>' --checkpoint-sha256 '<same full SHA256>' \
  --variant panda_condq --alpha '<approved alpha>' --beta '<approved beta>' \
  --gamma '<approved gamma>' --delta '<approved positive delta>' \
  --seed '<paired Student seed>' --data-seed '<paired order seed>' \
  --augmentation-seed '<paired augmentation seed>' \
  --calibration '<new calibration.npz>' --calibration-file-sha256 '<archive file SHA256>' \
  --updates 2 --microbatch 12 --effective-batch 48 --memory-gib 2 \
  --hqnr-target 0.959 --device cpu --output '<new diagnostic directory>'
```

`diagnostic`은 **최대 2 update**, manifest `official_experiment=false`, 별도 경로에만
기록한다. 50K 학습을 자동으로 시작하는 `run` 명령이나 sheet uploader는 추가하지
않았다. GPU 명령은 기존 GPU 점유 시 실패하며 기존 학습을 중단하지 않는다.

## 구현 검증 / 남은 실행 단계

- 실제 donor24 A0/step8000을 호환성 샘플로 읽었을 때 기존 코드 대비 A correction,
  aligned PAN, T prediction이 `torch.equal=True`, 최대 차이 0이었다.
- 실제 WV3 train 입력으로 CPU GT-only 1-update smoke 및 checkpoint 저장 완료.
  출력: `work_dir/panda_stage2_condq/validation_20261016/real_gt_only_1step/`.
  이 checkpoint는 구현 점검용 Student이며 공식 donor/실험 결과가 아니다.
- Synthetic 전체 calibration/ECDF, 실제 W112 gradient isolation, B48 대 B4×12,
  RR256/FR512 A+S 추론, 같은-state 평가 어댑터, strict resume와 실패 복구 테스트가 있다.
- 2026-10-16 검증: 신규 테스트 **73/73**, 기존 `donor24 smoke` **23/23** 통과.
  신규 테스트 결과와 source pin은
  `work_dir/panda_stage2_condq/validation_20261016/unit_tests_1792141528969974997.json`
  에 보존했다. 평가 테스트의 합성 metric 수치는 실제 모델 성능이 아니다.
- 같은 날 checkpoint 선택 규칙 수정 후 **79/79** 통과. HQNR 목표 경계·ERGAS 우선·
  SCC 무관성·목표 변경 resume 거부를 추가 검증했다. 최신 결과:
  `work_dir/panda_stage2_condq/validation_20261016/unit_tests_1792142513967946040.json`.
- 공식 RR/FR **수치 자체를 새로 측정하거나 개선을 입증한 것은 아니다**.
- 현재 Stage 1 source pin 105개와 기존 `donor24`/`donor30`/평가/시트 코드는 수정하지 않았다.

정식 실험 전에 하나의 공통 donor SHA, paired seed/optimizer, alpha/beta/gamma/delta/K,
실험 case 및 시간 예산을 확정하고 **그 donor의 전체 train calibration**을 먼저 만들어야
한다. 이후 GPU smoke/처리량 측정과 공식 RR20/FR20 평가를 수행하고 별도 신규 campaign
ID/outbox로 연결한다. 현재 Stage 1의 시트 규칙이나 결과 ID를 재사용하지 않는다.
