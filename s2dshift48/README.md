# S2D4 Global-Shift PAN Aligner — isolated implementation

근거: `research_log/PANDA_Stage1_S2D_GlobalShift_Aligner_Spec_2026-10-16.md`.
기존 `directshift48`, `donor30`, `donor24` 또는 `panda_stage2_condq`를 교체하지 않는다.
**구현 검증용 신규 namespace이며, 기존 학습 중단·새 캠페인 시작·donor 자동 선택·시트
업로드를 수행하지 않는다.**

정식 40시간 실행은 별도 `panda40` namespace에 연결되었다.
`bash scripts/run_panda40_docker.sh s1 --replace-donor24`를 사용하며,
이 패키지 자체의 diagnostic 명령은 여전히 학습 캠페인을 시작하지 않는다.
상세 큐는 `research_log/PANDA_S13_40H_S2D_Stage2_2026-10-16.md`를 따른다.

## 구조와 유지한 계약

`S2DGlobalPANAligner`는 기존 타입 검사와 인터페이스를 만족하는 명시적 subclass다.
기존 레이어를 만들었다 버리는 대신 새 레이어만 초기화한다.

```
original PAN / stored LMS
  -> crop8 -> original-channel detached-stat z-norm
  -> each pixel_unshuffle4
  -> PAN 16->16 / LMS 16C->16 reflect3x3 GN4 SiLU
  -> concat32 -> joint32->32->64 (both stride1, GN8 SiLU)
  -> existing common pre-LN ResBlock64
  -> GAP -> FC64->32->2 (last weight/bias zero)
  -> (dy,dx), current PAN pixels
```

- global `[B,2]`만 출력한다. D2S, dense flow, LPAN/HPAN, shift clamp 없음.
- 호출 측에서 **원본 전체 PAN을 한 번** warp. LMS/MS는 이동하지 않으며 FR ×4 없음.
- Crop8은 입력 관측창이다. Stage 1 reconstruction MASK2/UNMASKED와 다른 개념이다.
- native64, RR256, FR512 및 4/8 bands 지원. 각 변은 4의 배수이며 최소24.
- FP32, device, finite 입력/중간 feature 검증. normalization은 S2D 전에 원래 밴드별로 실행.
- `return_details=True`: `features`, `head_input`, `head_hidden`은 기존 인터페이스와 동일.
- 새 method/revision4001 및 manifest identity를 기록. old/new weight, 잘못된 shape/dtype,
  nonfinite weight, 누락 state 및 직접 `strict=False` load를 거부한다.

## Stage 1 연결

`stage1.build_s2d_stage1(...)`은 W112/D123/P0/BICUBIC_MS Teacher를 fresh로 만든다.
`teacher_seed`, `aligner_seed`, `epsilon_seed`, `mask_policy`를 명시해야 한다.
`mask_policy`는 `MASK2` 또는 `UNMASKED`; 현재 donor24와 비교하려면 **양쪽 MASK2**다.
loss 계수, optimizer, scheduler, effective batch/data/augmentation은 비교군과 동일하게
맞춰야 하며 이 모듈이 임의의 실험값을 확정하지 않는다.

- `S2DShiftStage1`: 기존 `DirectShiftStage1.forward`를 그대로 상속.
- `S2DMaskedStage1`: 기존 `MaskedDonorStage1.forward`를 그대로 상속.
- 기존 `DirectStage1Trainer`의 optimizer/gradient 진단·epsilon RNG·누적 배치 API 사용.
- reconstruction, Scharr `/32` RGB mean-first ROI8, radius3 uniform-area epsilon,
  native-target detach, support fail-before-update 정의를 변경하지 않는다.
- 같은 Teacher seed/epsilon seed라면 Teacher 초기 weight, epsilon sequence, zero-head
  초기 loss가 기존 baseline과 정확히 같다. Aligner의 서로 다른 구조에 같은 seed를
  주는 것은 모든 공통 레이어의 초기 weight가 같다는 뜻은 아니다.
- 기존 Teacher output convolution도 zero-init이므로 첫 update의 `L_rec -> A`
  gradient는 0이다. `L_abs/L_eps -> A`는 살아 있고 Teacher output layer가 움직인 후
  `L_rec -> A/T`가 연결된다. S2D가 새로 만든 문제는 아니다.

**이 패키지 자체는 정식 campaign runner가 아니다.** `panda40`가 별도 실행을 담당한다.
기존 donor24 launcher/evaluator의 case identity는
S2D와 다르므로 기존 run ID·checkpoint를 재사용하지 않는다. 신규 case가 확정되면
별도 campaign의 scheduler/stream/global RNG, 공식 same-state RR20/FR20, checkpoint
보존·비용·sheet outbox를 연결해야 한다. 현재 공식 metric 수치를 측정한 것은 아니다.

## Stage 2 donor handoff

```
save_donor_checkpoint(path, trainer, registered, *, run_id,
                      purpose="IMPLEMENTATION_DIAGNOSTIC")
load_donor(path, expected_sha256, registered, *, include_teacher=True, device="cpu")
```

두 함수는 `s2dshift48.donor` 소속이다. 새 schema의 같은 checkpoint에서 A/T를 읽고,
파일·A/T tensor·train inventory/catalog/H5·구조·구현 파일 SHA를 검증한다. 저장은
원자적·exclusive이며 0-update 모델이나 old donor30/Stage2 checkpoint를 허용하지 않는다.

반환된 `LoadedDonor`는 기존 `ConditionalQStage2`에 연결되며 loss는 바뀌지 않는다.
**새 donor로 전체 train calibration을 다시 만들어야 한다.** 기존 donor의 `tau_e`,
q ECDF를 재사용하면 정확한 binding 비교에서 실패한다. GT-only의
`include_teacher=False`는 Teacher 객체를 생성하지 않는다.

- Stage 2 손실은 현재 `panda_stage2_condq`의 full-image Hard/Soft 계약 그대로다.
  이번 Aligner 변경으로 Stage 2 masking을 동시에 변경하지 않았다.
- 기존 `python -m panda_stage2_condq ...` CLI는 old loader를 사용하므로 S2D 파일을
  받지 않는다. 새 loader/API로 연결하고 새로운 실행기의 source pin에는
  `s2dshift48.runtime.code_pin()`을 사용해야 한다.
- handoff에는 trainer 감사용 상태가 포함되지만 scheduler/stream/global RNG와 공식
  평가 선택 증거는 없다. **완전한 campaign resume나 qualified/best donor를 뜻하지 않는다.**

## 비용과 방법론 검토

아래 수치는 실제 모듈 parameter 수와 실행된 Conv/Linear hook에서 얻었다. MAC은
**Aligner 1회 forward, batch1**이며 normalization/activation/warp/Teacher/Student는 제외한다.

| 항목 (WV3) | 기존 DirectShift | S2D4 |
|---|---:|---:|
| Parameters | 105,458 | 124,898 (+18.4%) |
| native64 Conv/Linear MAC | 21.568M | 17.586M (−18.5%) |
| RR256 Conv/Linear MAC | 539.138M | 439.605M |
| FR512 Conv/Linear MAC | 2.303G | 1.878G |
| local convolution RF side | 25px | 44px |

RF44는 normalization을 제외한 convolution 경로의 범위다. z-norm/GN과 GAP은 여전히
관측창 전체에 의존한다. Crop8을 RF22로 자동 확대하면 입력 계약을 바꾸게 되므로
그렇게 하지 않았다. warp 외삽은 입력 정규화 전에 crop8로 제거하고, CNN padding은
기존과 동일한 reflection/zero padding을 유지한다.

CPU latency 원자료는 `work_dir/s2dshift48/validation_20261016/profile_wv3_cpu.json`에 있다.
기존 학습과 기타 프로세스를 정지시키지 않은 관측값이다. CPU backend·검증 overhead가
포함되므로 GPU 속도나 전체 학습시간의 개선 근거로 삼지 않는다. GPU profile은 미실행.

방법론 판단:

- S2D 자체는 가역적인 phase 재배열이지만 **128->16 LMS stem은 무손실이 아니다**.
- PAN-only translation에서 필요한 것은 출력 불변성이 아니라
  `u(W(P,epsilon),LMS) + epsilon ≈ u(P,LMS)`라는 상대이동 반응이다.
  1px 이동은 phase 채널을 바꾸므로 이 반응이 자동 보장되지 않는다.
- `diagnostics.epsilon_response`로 quarter/subpixel/integer shift별 반응을 측정할 수 있다.
  이 값은 실제 씬 displacement GT 오차가 아니라 synthetic consistency 진단이다.
  zero-head constant predictor가 이 검사를 통과한 것으로 표시되지 않도록 테스트했다.
- 단일 global translation 한계, PAN/RGB 분광 차이, RR–FR 일반화, HQNR–ERGAS trade-off는
  그대로 남는다. 구조 변경만으로 해결됐다고 해석하지 않는다.
- 우선 기존 vs S2D를 동일 MASK2/Teacher/data/seed/update/loss 조건으로 비교한다.
  Stage 2 새 loss와 동시에 변경하지 않는다. 개선이 있으면 크기/RF 증가 대조군으로
  "S2D만의 효과"와 용량/문맥 효과를 분리한다.
- 본선 판단은 같은-state HQNR 목표 통과 후보 중 ERGAS 최소다. 그 외 `D_s/D_lambda`,
  epsilon response, shift 분포, support 실패율, 비용도 함께 기록한다.

관련 선행 근거는 [SPD-Conv](https://arxiv.org/abs/2208.03641)와
[downsampling/shift sensitivity 연구](https://proceedings.mlr.press/v97/zhang19a.html)다.
이들 연구는 현재 PAN/MS 정합에서 S2D가 우월하다는 실험 증거는 아니다.

## 검증 실행

기존 `pancrafter` 환경에서:

```bash
python -m s2dshift48 smoke --report-dir work_dir/s2dshift48/validation
python -m s2dshift48 profile --bands 8 --sides 64 256 512 --threads 2
```

둘 다 synthetic CPU 전용이다. `run` 명령·실험 자동 실행·시트 uploader는 없다.
테스트는 shape/phase/전처리 순서, gradient 경로, 단일 warp/좌표,
old-new checkpoint 분리, paired donor export/import, 전체 synthetic train calibration,
Student-only update, MAC 집계, epsilon 반응을 검증한다. 품질 개선 검증은 별도 학습이 필요하다.

2026-10-16 검증 완료:

- 신규 S2D **33/33** 통과(9.62초).
- 기존 `directshift48.test_method` + `donor30.test_masking` **23/23** 통과.
- 기존 신규 Stage 2 패키지 `panda_stage2_condq smoke` **79/79** 통과.
- S2D 결과/source pin:
  `work_dir/s2dshift48/validation_20261016/unit_tests_1792145601136219461.json`.
- Stage 2 회귀 결과:
  `work_dir/s2dshift48/validation_20261016/stage2_regression/unit_tests_1792145641655286577.json`.
- 실행 중인 donor24 Stage 1 source pin **105개 불변**, 동일 컨테이너/PID 실행 유지 확인.
- 아직 실제 데이터 S2D 학습, GPU throughput/메모리, 공식 RR20/FR20 품질 측정은 하지 않았다.
