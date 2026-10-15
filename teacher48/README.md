# Teacher48H: Stage 1 Teacher 목표 탐색

기준일: 2026-10-13 (Asia/Seoul)  
Campaign: `PANDA_S1_M13_TEACHER48H_S123_20261013_v1`  
Method revision: `MethodPlan-v1.3`  
Parent campaign: `PANDA_S1_M12_TEACHER48H_S123_20261013_v1` (계획 출처만 보존)

이 패키지는 [Stage 1 실험 계획](../research_log/PANDA_Stage1_Teacher_TargetSearch_S123_48H_2026-10-13_v1.md)의 case·학습·평가·기록·업로드를 기존 코드에 **추가**한다. 기존 TA2의 loss·Aligner 가중치나 selection 규칙을 새 방법으로 재명명하지 않는다. Stage 2/KD/calibration은 실행하지 않는다.

2026-10-13 사용자 지시에 따라 [MethodPlan-v1.3](../research_log/PANDA_Method_Implementation_Validation_Plan_2026-10-13_v1_3.md)을 적용했다. head 입력은 `[GAP h; sg(b_s)]`이며, `b_s=(dy,dx)`는 검증된 grid 변환과 증강 후의 **raw signed current-PAN-grid prior**다. 크기 정규화·clamp·추가 scale은 없으며 native/ε branch에 같은 prior를 넣는다. 출력 `b_s + αd_s + βn_s`, zero-init, U-Net, loss 및 case/seed/시간 예산은 그대로다.

v1.2와 FC shape가 같아도 weight 의미가 다르다. A의 persistent revision marker, method SHA, inference/calibration 계약으로 묵시적 재사용을 차단한다. v1.3 실행 ID는 `S1M13_48H_`로 시작한다. 과거 M12 결과·outbox·48h clock을 새 방법의 결과나 이어받을 clock으로 취급하지 않으며, 승인된 별도 migration 없이 **fresh 실행**한다. marker나 문자열만 바꾸는 migration은 지원하지 않는다.

## 현재 상태: 구현과 실제 실행 준비는 다르다

CPU 합성 테스트를 통과한 구현이다. **실제 실험을 시작하거나 Google Sheet에 결과를 쓴 상태가 아니다.** 실제 데이터 admission 및 GPU throughput/VRAM 검증도 아직 완료되지 않았다.

기존 등록 자산을 읽어 확인한 결과, 세 센서 모두 다음 정보가 새 방법에 필요한 수준으로 등록되어 있지 않다.

- training-only **native FR** PAN/MS/LMS와 원천 증거.
- 기존 RR training patch의 실제 source scene/acquisition/crop 연결.
- train/eval의 겹침·근접 crop 분리 증거.
- 승인된 source-scene prior, RR/FR grid 변환 및 confidence 처리 증거.

현재 발견된 학습 H5는 이미 RR PAN `64×64`/MS `16×16`이고, 등록된 FR `512×512`/MS `128×128`은 **benchmark 20장**이다. RR을 확대하거나 FR test를 학습에 넣어서 위 항목을 채우면 다른 protocol이 된다. H5 row 번호를 실제 source scene ID라고 선언해서도 안 된다.

이 판단의 범위는 기존 등록 source catalog와 로컬 TA2 run metadata다. 서버 전체에 다른 원본 자산이 절대로 없다고 증명한 것은 아니다. [자산 조사 JSON](../results_log/2026-10-13_teacher48_existing_assets.json)과 [구현 검토 보고서](../results_log/2026-10-13_teacher48_implementation_review.md)를 참고한다.

`config/teacher48/*.baseline.draft.json`은 **초안**이다. 미확정 항목을 근거 없이 채우거나 `draft` 파일을 launch-ready baseline으로 취급하지 않는다.

이 원천 자료를 준비하는 별도 offline 도구를 추가했다. [Native FR·scene prior 준비 안내](PREPARATION.md)의 `python -m teacher48.prepare`는 inventory, native crop, 정확 GT 매핑 후보, prior 후보를 생성한다. 공식 FR20은 이미 존재하지만 평가 자료이며, 준비 도구가 이를 학습으로 전용하거나 누락된 원본 PAN을 RR에서 복원하지는 않는다.

## 1. 서버·case·선정 정책

| 서버 | 센서 | HQNR 하한 | ERGAS 상한 | R1 / R2 / R3 |
|---|---|---:|---:|---|
| s1 | WV3 | 0.958 | 2.040 | 261013101 / 261013102 / 261013103 |
| s2 | QB | 0.923 | 3.500 | 261013201 / 261013202 / 261013203 |
| s3 | GF2 | 0.965 | 0.522 | 261013301 / 261013302 / 261013303 |

- F00–F15: 계획의 16개 `(λ_eps, λ_abs)` 조합과 센서별 우선순위를 그대로 등록한다. 두 계수는 항상 양수다.
- T00–T08: 선택한 F recipe의 loss만 상속한다. LR/weight decay 배율은 **원 baseline 기준**이며, 직전 T의 값에 누적하지 않는다. WD0=0이면 T05/T06는 `SKIP_DUPLICATE_CONFIG`다.
- H50/H100/H150: 50K/100K/150K horizon을 처음부터 선언한 **fresh** 학습이다. H50 weight를 불러와 H100이라고 하지 않는다.
- 공식 평가: `{1000} ∪ {2000, 4000, …, horizon} ∪ {25000}`. 각각 **27/52/77개**이며 endpoint 중복은 없다.
- R1 탐색 후 최대 두 recipe를 후속 optimizer/긴 학습/반복 대상으로 삼는다. 전체 catalog를 48시간 내 모두 끝내는 약속은 아니다.
- 같은 A/T snapshot의 **HQNR ≥ 하한 AND ERGAS ≤ 상한**을 full precision으로 만족해야 weight와 정성 결과를 저장한다.
- 통과 후보 대표는 HQNR 내림차순 → ERGAS 오름차순 → step 오름차순 → run ID다. 통과 후보 중 ERGAS 최소 예비도 따로 기록한다.
- 미달·중단·실패 수치도 보존한다. `HQNR_MAX_OBSERVED_NUMERIC`은 숫자 selector이지 미달 weight 저장 허가가 아니다.
- 이 탐색은 benchmark RR/FR 성능을 보고 설정을 고르는 **test-aware 개발**이다. 독립 test 검증이라고 표현하지 않는다.

서버별 t0/48h 예산은 독립이다. 46h부터 새 학습은 시작하지 않으며, 실제 P90 update·전체 평가·probe·artifact 예상 비용과 20% 안전 여유로 admission한다. F00 및 확인 반복은 일반 성능 pruning으로 자르지 않는다. 다른 R1 탐색 후보도 최소 25K, 같은 seed/horizon의 두 선행 recipe·세 공통 평가점·10K 이후 gap 무개선 조건을 확인한다.

## 2. 추가된 코드의 책임

| 파일 | 책임 |
|---|---|
| `plan.py` | registry, seed, baseline-relative resolve, cadence, target gate, 순위/pruning/예산 |
| `data.py` | hash-pinned source manifest, 실제 scene/crop 연결, RR/FR train/eval 분리, 증강/prior 변환 |
| `training.py` | RR mean + FR mean 누적, shared A/T optimizer, 독립 RNG, 고정 warmup/cosine |
| `evaluation.py` | paused same-state RR20/FR20, 전체 metric, qualified-only checkpoint/정성 export |
| `diagnostics.py` | 고정 AXIS16·구조 discrepancy·추론 개입·training sample gradient 진단 |
| `runtime.py` | 로컬 campaign clock, case 선택, local-first journal, 예산·상태·요약 |
| `reporting.py` | 새 fixed64 전용 destination 검증, outbox, idempotency, RAW/readback |
| `asset_audit.py` | 기존 H5 header와 과거 config의 read-only 조사; admission이나 데이터 생성 아님 |
| `__main__.py` | `catalog`, `inspect-assets`, `evaluator-id`, `admit`, `run`, `publish` CLI |

모델·loss 본체는 `panda_method/`를 사용한다. RR reconstruction은 A/T를, RR absolute gradient 및 RR/FR ε는 공유 A를 update한다. shifted PAN을 T에 넣거나 FR GT/reconstruction loss를 만드는 경로는 없다. ε는 radius 2의 면적 균등 원판이며 branch별 mean을 **합한다**.

## 3. 실제 CLI

저장소 root에서 기존 `pancrafter` Python 환경을 사용한다. 명령이 존재한다는 것과 실행 준비가 완료됐다는 것은 다르다.

```bash
python -m teacher48 --help
python -m teacher48 inspect-assets
python -m teacher48 evaluator-id
python -m teacher48 catalog --server s1 --output config/teacher48
```

`inspect-assets --output PATH`는 조사 JSON만 저장한다. 이 파일은 training source manifest가 아니다. `catalog`도 recipe metadata만 만들고 학습/Google 접근을 하지 않는다. s2/s3 catalog는 `--server`만 바꾼다.

검증된 실데이터와 확정 baseline을 확보한 뒤 실행하는 예시는 다음과 같다. 아래 `baseline.resolved.json`, `sources.resolved.json`은 **사용자가 근거를 갖고 준비할 파일 이름의 예**이며 현재 존재하는 자산이라고 주장하지 않는다.

```bash
python -m teacher48 run \
  --server s1 \
  --baseline config/teacher48/s1.baseline.resolved.json \
  --sources /absolute/path/to/s1.sources.resolved.json \
  --output work_dir/teacher48 \
  --device cuda
```

`run`은 admission을 내부 수행하므로 별도 사전 명령을 반드시 거칠 필요는 없다. 실패하면 `BLOCKED_DATA_PROTOCOL`을 로컬/업로드 대기열에 기록하고 full-method training을 시작하지 않는다. 사전 진단만 필요할 때는 동일 인수로 `run` 대신 `admit`를 사용할 수 있다. `admit`도 로컬 evidence/outbox를 만들지만 optimizer update나 publisher 시작은 하지 않는다.

이미 시작된 campaign 프로세스가 중단됐다면 같은 `run` 명령에 `--resume-campaign`을 명시한다. 이는 **기존 t0/48h 마감을 유지하는 장부 복구**다. 완료·pruned run을 반복하지 않고 실패/중단 run은 같은 seed의 새 attempt로 fresh 재학습한다. 미달 weight를 디스크에서 복원하는 exact resume가 아니며, 이미 만료된 48h를 새 48h로 초기화하지 않는다. 기존 baseline/source admission과 충돌하는 변경은 조용히 이어 붙이지 않는다.

`run`은 별도 publisher 프로세스를 시작한다. 인증 문제가 있어도 학습 update가 네트워크 응답을 기다리지 않고 outbox를 남긴다. `--credentials`를 생략하면 현재 transport는 저장소 `gspread/account.json`을 사용한다. credential 내용은 코드·보고서·commit에 넣지 않는다.

이미 로컬에 남은 이번 campaign 수치를 다시 전송할 때:

```bash
python -m teacher48 publish --server s1 --output work_dir/teacher48
python -m teacher48 publish --server s1 --output work_dir/teacher48 --watch
```

이 명령은 **원격 쓰기를 수행할 수 있다**. 실제 결과·권한을 확인하고 실행한다. 다른 campaign의 과거 결과를 새 시트로 backfill하는 명령이 아니다. 프로세스 정보와 로그는 해당 campaign의 `publisher_process_*.json`, `publisher.log`, `publisher_errors.jsonl`에 남는다.

## 4. baseline 계약: 과거 근거와 새 결정의 구분

로컬 WV3 TA2 실제 config에서 확인된 것은 W112/D123/P0, `BICUBIC_MS`, AdamW, betas `(0.9, 0.999)`, eps `1e-8`, U LR `1e-4`, A LR `1e-5`, WD `0.01`, warmup 100, RR effective batch 48/microbatch 12/accumulation 4다. **QB/GF2의 이 서버 내 실제 TA2 run config가 확인됐다는 뜻은 아니다.** 과거 공통 계획 근거와 센서별 실제 실행 증거를 구분한다.

`baseline_resolved.json` 필수 항목:

- `method_revision: "MethodPlan-v1.3"`, `method_sha256: "122472a5c89b2b1e827bbb5c46d19d02821efdae264fb3b311ec2bd5f07f64c0"`, `head_conditioning: "raw_signed_current_pan_grid_dy_dx"`.
- `teacher_architecture`: `{bands,width,depth}`, `teacher_input_layout`, `teacher_base`.
- `aligner_architecture`: `{name: "SceneAwarePANAligner", view_margin}`. 새 A는 공통 LN ResBlock/scene residual 구조이며 old GN A checkpoint를 이식하지 않는다.
- `optimizer`, `optimizer_betas`, `optimizer_eps`, `lr_U0`, `lr_A0`, `weight_decay0`.
- `warmup_updates0`, `min_lr`(scalar), `scheduler_family: "warmup_cosine"`, `scheduler_indexing`.
- `effective_batch_RR/FR`, `microbatch_RR/FR`, `accumulation_RR/FR`: 각 branch에서 effective = microbatch × accumulation.
- `RR_train_HW: [64,64]`, 실제 native support로 확정한 `FR_train_HW`, `precision: "fp32"`.
- `augmentation`: `hflip_probability`, `vflip_probability`, `rot90`(boolean).
- `gradient_kernel_roi_normalization`: `sensor`, `rgb_indices`, `operator`, `padding`, `roi_margin`, `radiometry: "normalized_minus1_plus1"`.
- `boundary_policy`: `report` 또는 `require_spatial_support` 중 명시적으로 승인된 선택.
- `evaluator_revision`: `evaluator-id`의 실제 digest.
- `qual_export`: `rgb_indices`, `display_range_DN: [0, sensor_peak]`, `roi: "FULL_SCENE"`, `edge_gain`, `error_gain`.
- `probes`: `rr_eval_ids`, `fr_eval_ids`, `rr_train_sample_id`, `fr_train_sample_id`, `max_samples`(branch별 양의 상한).
- `artifact_reserve_seconds`, `probe_reserve_seconds`: 근거가 있는 양의 시간 allowance. 관측된 실제 시간으로 보수적으로 갱신한다.

새 scheduler의 정확한 convention은 `ONE_BASED_WARMUP_LINEAR_PEAK_THEN_COSINE_EXACT_ENDPOINT_MIN`이다. warmup이 있으면 update u에서 peak×u/warmup, warmup endpoint가 peak다. warmup이 없으면 첫 update가 peak다. 등록 horizon의 마지막 update가 min LR이다. 과거 zero-based scheduler의 궤적과 비트 단위로 같다고 주장하지 않는다. 이 convention을 baseline에 명시적으로 승인해야 한다.

FR batch/shape, absolute-gradient ROI, prior acceptance, 새 scheduler index는 과거 config만으로 자동 확정하지 않는다. `MUST_PIN`, null 또는 누락은 실행 가능한 default가 아니라 오류다.

## 5. source manifest 계약

`Sources`는 데이터 전처리/scene estimator가 아니라 **이미 준비하고 검증한 native 자산을 읽는 adapter**다. manifest schema는 `TEACHER48_SOURCES_V1`이다.

상위 필드:

- `schema`, `dataset`(`WV3/QB/GF2`), `radiometry: "native_DN"`, `ratio: 4`.
- `files`: file ID → `{path,sha256}`. 상대 path는 manifest 디렉터리 기준이다.
- `h5_keys`: `pan/ms/lms/gt` 및 layout이 요구하면 `lpan`의 실제 H5 key.
- `uses_lpan`: T layout의 사용 여부와 일치하는 boolean.
- `samples`: 정확히 `rr_train`, `fr_train`, `rr_eval`, `fr_eval` 네 group. train은 비어 있으면 안 되고, RR/FR eval은 각각 **20개**다.
- `scene_priors`, `grid_transforms`, `split_separation_margin_native_pan`.
- `{path,sha256}`로 연결된 여섯 evidence: `scene_mapping_evidence`, `native_fr_training_evidence`, `lms_provenance_evidence`, `prior_acceptance_evidence`, `split_separation_evidence`, `evaluation_protocol_evidence`.

각 sample에는 `sample_id`, `source_scene_id`, `acquisition_id`, `file_id`, `row`, `grid_name`, `geometry_evidence`, `source_crop_native_pan_yxhw`, `preparation_augmentation`가 필요하다. `sample_id`는 전체 group에서 유일해야 한다. crop은 실제 공통 native PAN 좌표의 정수 `[y,x,h,w]`다. 준비 단계 증강은 `{hflip: bool,vflip: bool,rot90: 0..3}`로 기록한다. adapter는 저장된 영상을 다시 증강하지 않고 prior vector만 준비 단계의 방향으로 변환한다. 학습 시 추가 증강은 영상과 prior에 함께 적용한다.

eval sample은 고정 평가 목록의 유일한 `scene_id`가 있어야 한다. fr_train sample은 `native_fr_training: true`와 실제 근거가 필요하다. 문자열/boolean을 채우는 것만으로 원천 사실이 검증되지는 않는다.

`scene_priors`는 `panda_method.geometry.ScenePriorRecord`의 모든 필드를 가진다:

```text
source_scene_id, prior_shift_dy_dx, prior_grid_name, pixel_scale, origin_phase,
n_windows, n_valid, inlier_fraction, direction_concentration, dispersion,
confidence, estimator_version, reference_type, scene_input_hash, mapping_rule,
accepted, acceptance_evidence
```

prior는 sampling displacement `(dy,dx)`이며 방향과 크기를 모두 포함한다. exact zero/nonfinite/미승인 prior는 조용한 fallback 없이 실패한다. low-confidence/near-zero의 과학적 수용 여부는 외부 증거와 승인 정책으로 확정한다.

grid가 다르면 `grid_transforms`에 `source_grid`, `target_grid`, `matrix_dy_dx`, `phase_dy_dx`, `evidence`를 준다. RR/FR 영상 크기만 보고 ×4/×0.25를 자동 적용하지 않는다.

split separation evidence JSON은 `accepted: true`, `exact_duplicates: 0`, `near_duplicates: 0`, `protocol`, 현재 `samples`의 `sample_index_sha256`를 포함한다. 코드도 동일 acquisition의 crop 교차/버퍼 위반, eval의 중복 file-row/내용, train/eval 정확 중복을 추가 검사한다. 외부 near-duplicate audit의 진위를 코드가 대신 입증하는 것은 아니다.

입력은 native DN으로 보존하며 network 경로에서만 FP32 `[-1,1]`로 변환한다. 공식 metric의 원본 GT/PAN/LMS/MS는 저장 precision을 유지한다. 새 LMS를 bicubic으로 즉석 생성하지 않는다.

## 6. 공식 평가·보존·진단

학습을 optimizer 경계에서 멈추고 A/T state를 고정한다. RR 전후·FR 전후의 run/attempt/seed/update/state hash/inference/evaluator/prior-rule identity를 비교한다. RR/FR의 numerical prior cache가 같을 필요는 없지만 변환 규칙은 같아야 한다.

- RR: ERGAS, SCC, SAM, PSNR, SSIM, Q8(WV3)/Q4(QB/GF2), **RMSE, CC**.
- FR: HQNR, D_s, D_lambda, **JQM**.
- 공식 RR20/FR20 고정 목록 전체, 중복/누락/NaN, aggregate와 per-scene mean 일치를 검사한다.
- 기존 canonical evaluator를 pin하고 FR은 **original PAN/native LMS, full-frame unmasked** 기준으로 평가한다. training warp support mask를 official HQNR에 전달하지 않는다.
- JQM은 `SRF-substitute (NNLS-normalized), MTF41 phase2, global CMSC v1=0.5; not SIPSA-equivalent` variant다. SIPSA의 SRF 기반 결과와 완전히 동일하다고 표기하지 않는다.
- 로그에는 HQNR/SCC/ERGAS를 함께 출력한다. 목표 판단은 표시 문자열이 아닌 원정밀도 수치로 한다.

`STRICT_TARGET_ONLY`에서 미달 모델은 `last/best/resume/temp_eval` weight나 prediction image/NPY/MAT를 디스크에 쓰지 않는다. 현재 학습 state와 transient 평가 예측은 RAM에만 존재한다. 실패하면 해당 미달 weight의 exact resume는 불가능하다.

통과 state는 local numeric journal 뒤에 A/T·optimizer/RNG/config·state hash·파일 SHA를 atomic 저장하고 load/tensor 검증한다. 같은 평가 예측의 raw all-band DN와 고정 display RGB/GT·LMS·PAN/corrected PAN/edge overlay를 저장한다. 숫자상 통과와 `QUAL_COMPLETE`를 구분하며, export가 실패하면 `QUAL_PENDING`이다.

10K/25K/50K 및 긴 horizon endpoint의 고정 probe는 모두 numeric-only다. AXIS16 gain/MAE/EPE, before/prior/current 구조 discrepancy, 같은 T에 대한 current/prior/zero inference intervention을 기록한다. 이는 재학습 ablation도 물리적 displacement GT도 아니다. gradient probe만 실제 training sample을 사용하며 평가 장면을 backward에 넣지 않는다. 모든 probe는 A/T parameter·mode·training/global RNG·`.grad`를 보존한다.

v1.3의 RR/FR 학습 수치에는 raw prior의 dy/dx 범위와 FC1 preactivation·SiLU hidden의 min/max/RMS/finite를 별도로 남긴다. 기존 native forward 결과를 재사용하므로 진단을 위한 A forward를 추가하지 않는다. microbatch 범위는 실제 최솟값/최댓값, RMS는 제곱 평균 기준으로 합친다. 이 계측이나 단위 테스트는 raw prior conditioning의 실제 성능 향상을 증명하지 않는다.

## 7. Google Sheet: 최신 fixed64 지시가 우선

계획서 §15의 **구 workbook/`analysis` 152열은 사용하지 않는다.** 이후 사용자 지시와 [gspread handoff](../research_log/gspread_rules/FIXED_COLUMNS_AND_GSPREAD_HANDOFF_KR.md), [고정 schema](../research_log/gspread_rules/PAN_FIXED64_V1.schema.json)가 목적지를 대체한다.

Workbook: **pan-cvpr27-new**  
ID: `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE`

사용자가 재전달한 URL과 workbook ID가 일치한다. 그 URL의 `gid=261013013`은 제공된 로컬 schema에서 `WV2-main`이며, 이번 WV3/QB/GF2 캠페인의 목적지를 그 탭으로 바꾸지 않는다. 아래 `*-ablations` 규칙을 유지한다. 실제 원격 접근·schema/readback 확인은 publisher 실행 시 별도 검증 대상이다.

| 서버 | 자동 업로드 탭 | sheetId |
|---|---|---:|
| s1 | WV3-ablations | 261001010 |
| s2 | QB-ablations | 261001011 |
| s3 | GF2-ablations | 261001012 |

공통 `ablations` 탭, `paper`, `유의미한결과`, `*-main`에는 쓰지 않는다. 구 결과 backfill, 새 숨김 raw 탭 생성, 헤더 재설치/확장도 하지 않는다.

- A5:BL5 literal header, columnCount=64, workbook/sheet ID, A2 note의 `PAN_FIXED64_V1`/reset epoch `2026-10-13`을 쓰기 전에 검증한다.
- 데이터는 6행부터 A:BL 64열에 직접 기록한다. 부족하면 행만 확장한다.
- 성공/미달에 관계없이 공식 전체 평가와 RR/FR **각 scene 행**, 진단·상태를 기록한다. 모든 update 상세는 로컬에 두고 약 1시간 요약을 전송한다.
- 값은 **원정밀도 숫자를 RAW로 업로드**한다. metric·시간 등 표시만 소수점 네 자리다. LR은 여섯 자리, count/seed는 정수 표시다.
- 미측정 cost·지표는 공란이다. RMSE/CC/JQM을 생략하거나 0으로 대체하지 않는다.
- 같은 checkpoint의 여러 selector는 서로 다른 관측행이지 독립 반복이 아니다.
- method-specific 계수는 BI JSON에 기록한다. 기존 q_ref/tau_R/alpha/beta/lambda_E 열을 새 의미로 재활용하지 않는다.

먼저 `records.jsonl`과 `outbox/<Result_ID>.json`을 durable 저장한다. publisher가 별도로 Result_ID 조회 → RAW append 또는 동일 payload no-op → 실제 범위 readback → VERIFIED 확인 → receipt를 처리한다. 동일 ID에 다른 수치가 오면 overwrite하지 않고 conflict로 남긴다. 네트워크 timeout 뒤 blind append하지 않으며 permission/rate-limit/schema 오류를 구분한다. 원격 변경 성공은 **`receipts/*.json`의 `READBACK_VERIFIED`**로 확인한다.

현재 CLI publisher는 한 pass에서 최대 32행을 처리하고 watch 간격은 30초다. 실패 record는 원본과 retry 시각을 보존한다. 처리량 때문에 남은 pending은 학습 실패가 아니며, 종료 후에도 publisher/replay로 전송 상태를 따로 확인한다.

서버 간 Teacher/selection 공유 lock은 없다. 서버 자신의 중복 campaign 및 목적지 publisher 중복만 host-local lock으로 막는다. 같은 dataset destination을 두 물리 서버에서 동시에 운영하는 구성은 승인된 세 lane의 범위가 아니다.

## 8. 로컬 결과와 검증

기본 root:

```text
work_dir/teacher48/PANDA_S1_M13_TEACHER48H_S123_20261013_v1/<server>/<dataset>/
  campaign.json
  baseline_resolved.json
  admission/
  records.jsonl
  outbox/<Result_ID>.json
  receipts/<Result_ID>.json
  upload_events.jsonl
  runs/<run_id>/attempt_<n>/
    config_resolved.json
    initialization_hashes.json
    train_metrics.jsonl
    evaluation_progress.jsonl
    evaluations.jsonl
    evaluations_per_scene.jsonl
    diagnostics_numeric.jsonl
    final_status.json
  qualified/<run_id>/attempt_<n>/step_<u>/<state_sha>/
    teacher_aligner.pt
    source_manifest.json
    model_manifest.json
    metrics.json
    qual/
  summaries/current.json
```

`initialization_hashes.json`은 hash/manifest이지 초기 weight snapshot이 아니다. `qualified/`는 통과한 state에만 생긴다. 로컬에 원정밀도 평가·per-scene 수치·정확한 64열 payload·업로드 receipt가 함께 남아 Sheet 미접속 상태에서도 결과를 재구성할 수 있다. outbox 원본은 전송 후에도 자동 삭제하지 않는다.

테스트:

```bash
python -m unittest discover -s teacher48 -t . -q
python -m unittest discover -s panda_method -t . -q
```

테스트는 합성 CPU tensor·임시 파일·fake transport를 사용한다. 실제 source provenance, 실제 dataset metric 재계산, GPU 실행 성능, 원격 Sheet 권한/header/readback이 검증됐다는 뜻은 아니다. 실제 admission/실험/업로드는 자산이 준비된 뒤 위 절차로 별도 수행한다.
