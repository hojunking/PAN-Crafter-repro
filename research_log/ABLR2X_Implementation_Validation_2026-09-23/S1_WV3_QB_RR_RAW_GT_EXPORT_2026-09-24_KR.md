# s1 실행 지시서 — WV3·QB RR 전체 밴드 예측 + GT 추출·반환

작성일: 2026-09-24  
작업 ID: `PANDA_S1_WV3_QB_RR_RAW_GT_v2`  
범위: **기존 지정 모델의 RR20 전체 밴드 prediction과 대응 GT 확보. 렌더링은 수행하지 않는다.**  
문서 상태: 서버 실행 담당자에게 전달할 지시서. 이 문서 자체는 추론·파일 전송 완료 보고가 아니다.

## 1. 이번 지시가 우선하는 변경 사항

기존 `S1_WV3_QB_EXPORT_AND_DELIVERY_KR.md`의 대상 checkpoint는 유지하되, 이번 작업의 필수 범위를 **RR prediction + RR GT**로 좁힌다. 사용자 메시지의 “WV3-QP”는 앞서 지정한 **WV3·QB**를 대상으로 정리했다.

- 기존 PNG 생성 스크립트는 존재하지 않는다. 찾거나 복구하는 작업을 하지 않는다.
- 기존 CANConv PNG와의 렌더링 규칙 검증, `calibrate`, 오차 수식·vmax 역추정은 **생략**한다.
- 비교 모델 raw 확보, ROI 선정, 포함 모델 수, RGB stretch, colormap 결정은 이 작업의 선행 조건이 아니다.
- PNG/error map 생성 없이 raw와 GT를 검증·압축해 먼저 반환한다. 렌더링 부재를 이유로 `RENDER_PENDING` 대기 상태에 두지 않는다.
- **FR 신규 추론은 이번 필수 작업이 아니다.** 이미 동일 checkpoint의 FR raw가 있으면 별도 선택 자료로 반환할 수 있지만, RR 반환을 지연시키지 않는다. WV2·GF2도 이번 s1 작업 대상이 아니다.

이번 지시는 **렌더링 검증만 생략**한다. checkpoint 확인, prediction–GT 장면 대응, 전체 밴드·축·단위·파일 무결성 검사는 유지한다.

## 2. 고정 추출 대상

아래는 앞서 paper 표와 대조해 고정한 대상이다. 이번 실행에서 더 좋은 모델을 다시 검색하거나 checkpoint를 재선택하지 않는다. 서버의 실제 파일 존재는 실행 담당자가 확인한다.

| 항목 | WV3 | QB |
|---|---|---|
| 원 실행 서버 | s1 | s1 |
| 선택 | TARGET | TARGET |
| 선택 update | **38380** | **47470** |
| 전체 학습 완료량 | 50000 | 50000 |
| 밴드 수 | 8 | 4 |
| RR 장면 수 | 20 | 20 |
| 저장할 scene별 pred/GT shape | `(8,256,256)` | `(4,256,256)` |
| 저장 단위 | DN, maxDN 2047 | DN, maxDN 2047 |
| 기존 원결과 위치 | WV3-s1 96행 | QB-s1 16행 |
| 기존 paper 연결 | 17행 | 34·71행, 같은 결과 |

### 2.1 WV3

```text
run_id: FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1
selection: TARGET
step: 38380
checkpoint_sha256: 3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50

checkpoint:
work_dir/FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1/candidates/38380/model.safetensors
identity:
work_dir/FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1/candidates/38380/identity.json
config:
work_dir/FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1/meta/config.resolved.yaml
selection_record:
work_dir/FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1/official/target_selection.json
```

### 2.2 QB

```text
run_id: QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1
selection: TARGET
step: 47470
checkpoint_sha256: 548d0ed2ccdc968d319e850e1470d7068f9321a9f0cc18e1794807f5cfd3399d

checkpoint:
work_dir/QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1/candidates/47470/model.safetensors
identity:
work_dir/QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1/candidates/47470/identity.json
config:
work_dir/QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1/meta/config.resolved.yaml
selection_record:
work_dir/QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1/official/target_selection.json
```

각 상대 경로는 해당 run의 실제 repository/work_dir 루트를 기준으로 해석한다. 동일 SHA의 검증된 복제본을 사용할 수 있다. 경로가 없으면 알려진 해당 run/export 위치에서만 찾고, 현재 `best_hqnr`·`last`·EXACT50K로 대신하지 않는다.

**WV3 G23/W104D121, PAN-Crafter reproduction, QB TA/S82009는 이번 대상이 아니다.** 두 모델 모두 동일 선택 checkpoint의 학습된 Student Aligner + frequency frontend + reconstruction network 전체를 정상 `A_ON`으로 사용한다.

## 3. 실행 순서

### 3.1 s1의 현재 작업과 원본 자산 보존

현재 실행 중인 학습·평가 프로세스와 GPU 점유를 먼저 확인한다. s4에 대한 이전 일시정지 지시는 s1에 적용하지 않는다. 기존 학습의 강제 종료·임의 중단·queue 변경·watchdog 해제·checkpoint 덮어쓰기를 하지 않는다.

검증된 기존 raw가 있으면 GPU 없이 재사용한다. 신규 추론은 사용 가능한 GPU에서 독립 프로세스로 수행한다. GPU가 비어 있지 않으면 기존 loader가 지원하는 CPU 추론의 실행 가능성을 확인하고, 실행 자원을 확보할 수 없으면 `WAITING_FOR_RESOURCE`와 필요한 조치를 명시한다. 다른 학습에 무조건 병렬로 붙이지 않는다.

### 3.2 기존 raw를 먼저 확인

각 정확한 run 폴더와 기존 export 폴더에서 `.npy`, `.npz`, `.mat`, `.h5`의 예측 배열을 확인한다. 확장자는 달라도 좋다. 다음이 연결되는 원시 수치만 재사용한다.

- 지정한 checkpoint SHA와 step, 정상 전체 Student `A_ON`.
- 실제 RR 평가 데이터 및 LP 입력의 identity, 20개 scene의 순서.
- 전체 8/4밴드, 값의 단위와 배열 축, 기존 inference의 DN 변환·clip 규약.

예측 provenance가 없는 파일은 임의 채택하지 않는다. RGB PNG를 full-band raw로 복원하거나 GT로부터 예측을 만드는 우회는 하지 않는다. `metrics.json`만 있는 경우에는 예측 파일이 없는 것으로 처리한다.

`.mat` 등이 이미 있으면 key와 axis를 확인하고 NPY로 변환한다. 원본은 보존하고 변환 전후 수치가 같은지 확인한다. 예를 들어 `sr`이라는 key를 확인 없이 가정하지 않는다.

### 3.3 없으면 inference-only

기존 `export_worker.py`가 서버에 있으면 실제 코드/도움말을 읽고 아래 계약을 만족하는지 확인해 재사용한다. 경로·CLI 인자·지원 sensor를 추정해 실행하지 않는다. renderer/calibration을 필수로 요구한다면 raw export 부분만 분리한다. 기존 worker가 없으면 별도의 얇은 export wrapper를 만들되 학습 코드나 기존 frozen run의 내용을 수정하지 않는다.

앞선 감사에서 확인한 함수 경로는 다음이다. **독립 CLI 명령이 아니며**, 실제 run과 호환되는 frozen checkout에서 signature와 데이터 loader를 확인해 사용한다.

```text
WV3: fh20r1.common.load_checkpoint_model -> fh12.evaluation.infer
QB:  qg40.common.load_checkpoint_model   -> qg40.evaluation.infer
```

추론 조건은 `FP32`, `model.eval()`, no-grad/inference mode, `shuffle=False`, 원 native RR20 및 원 LP/frontend다. RGB 변환 전 모든 MS 밴드를 보존한다. GT는 prediction 생성 forward에 넣지 않는다. Aligner 제거, training augmentation, synthetic shift, test-time augmentation, fine-tuning, Teacher 출력 대체는 하지 않는다.

기존 inference와 호환되는 별도 프로세스를 사용한다. 읽어야 하는 것은 선택된 `model.safetensors`와 identity/config/data이며, 학습 optimizer를 만들거나 training loop를 다시 시작할 필요는 없다.

### 3.4 GT는 같은 RR 데이터의 원본에서 읽는다

선택 checkpoint의 원 평가에 사용한 RR H5/MAT에서 대응 GT를 가져온다. 장면 이름이 없으면 `source file SHA + dataset key + original index`로 식별한다. `testimgN`은 그 대응표에 따라 부여하고, prediction과 GT에 **동일한 scene mapping**을 적용한다.

GT가 없으면 PAN, upsampled MS, Teacher prediction 또는 다른 scene을 GT로 대체하지 않는다. 그 sensor는 `BLOCKED_MISSING_GT`로 보고하고, 다른 sensor의 완료 자료는 먼저 반환한다.

CANConv PNG와의 외부 scene 대응 검증은 이번에 요구하지 않는다. 하지만 **우리 prediction과 우리 GT가 같은 장면인지**는 반드시 확인한다. 외부 PNG 대응은 `NOT_CHECKED_BY_SCOPE`라고 적고, 전체 배열을 맞추기 위해 image registration/warp를 새로 적용하지 않는다.

## 4. 저장 규약

필수 저장 형식은 scene별 **float32 NPY, CHW, DN 단위**다. `raw`는 여기서 RGB/stretch 전의 full-band 출력이라는 뜻이며, 기존 evaluator의 출력 clamp를 해제한다는 뜻이 아니다.

| 검사 항목 | 규칙 |
|---|---|
| 파일명 | `testimg0.npy`부터 `testimg19.npy`, 0-based |
| WV3 pred와 GT | 각각 20개, 각 `(8,256,256)` |
| QB pred와 GT | 각각 20개, 각 `(4,256,256)` |
| dtype | `float32`, object array 금지 |
| 값의 단위 | 두 sensor 모두 원 evaluator와 같은 DN, maxDN 2047 |
| 밴드 순서 | 원 source 순서와 변환 여부를 명시. RGB 3밴드 선택 금지 |
| 공간 범위 | 256×256 전체 영상. resize/crop/재정합 금지 |
| 값 검사 | NaN/Inf 없음. min/max 기록 |
| 후처리 | percentile stretch, gamma, 색보정, sharpen, 추가 rounding/quantization 금지 |

기존 추론 함수가 이미 DN으로 반환한다면 `2047`을 다시 곱하지 않는다. `[-1,1]` 또는 `[0,1]` 원값에서 변환해야 하는 경우는 그 사실과 변환을 명시하고 **한 번만** 적용한다. GT도 같은 단위를 사용한다. GT는 원본 값을 float32로 보존하며 불일치를 숨기기 위해 임의 clipping하지 않는다.

기존 evaluator의 prediction clamp는 그대로 유지하고 manifest에 기록한다. 진단용 unclipped 출력이 이미 있으면 별도 부가 파일로 분리하되, `pred/`에는 정량평가와 연결되는 출력만 둔다.

원시 source가 HWC 또는 NHWC라면 의미를 확인하고 CHW로 transpose한다. reshape로 축을 바꾸지 않는다. 기존 GF2 package의 축이 다르더라도 새 package의 `axis_order=CHW`를 명시해 downstream에서 구분 가능하게 한다.

## 5. 반환 폴더와 파일 수

실제 생성 시각을 사용해 다음 package를 만든다. 경로 규약은 이전 s1 인계서의 `Ours_raw/`와 최상위 `references/`를 유지한다.

```text
Ours_s1_WV3_QB_RR_raw_gt_<timestamp>/
├── README.md
├── manifest.json
├── scene_manifest.json
├── Ours_raw/
│   ├── WV3/reduced/pred/testimg0.npy ... testimg19.npy
│   └── QB/reduced/pred/testimg0.npy ... testimg19.npy
├── references/
│   ├── WV3/reduced/gt/testimg0.npy ... testimg19.npy
│   └── QB/reduced/gt/testimg0.npy ... testimg19.npy
├── provenance/
│   ├── WV3/                 # identity, config, data/source 기록, 선택 기록
│   └── QB/
├── validation/
│   ├── raw_array_checks.json
│   └── rr_metric_comparison.json
└── checksums.sha256
```

**필수 NPY는 합계 80개: WV3 pred20 + GT20, QB pred20 + GT20.** float32 배열 본체만 계산하면 WV3 80 MiB, QB 40 MiB, 합계 120 MiB다. NPY header·metadata·reference 추가분과 ZIP 압축률은 이 계산에 포함하지 않는다.

PAN/MS/LMS의 해당 RR scene slice가 바로 확보되어 있으면 `references/<sensor>/reduced/` 아래에 부가 자료로 포함할 수 있다. 이것 때문에 필수 pred/GT 반환을 지연시키지 않는다. FR은 필수 80개에 포함하지 않는다.

실제 데이터셋의 재배포 조건을 준수하고 사용자 소유/승인된 저장소에 전달한다. train dataset 전체, optimizer state, 모델 weight 자체, SSH key, credential, 서비스 계정 JSON, 전체 환경변수는 넣지 않는다. config·로그도 credential이 없는 내용만 포함한다.

## 6. 꼭 남길 provenance

`manifest.json`에 각 sensor별 다음을 기록한다.

```text
sensor, origin_server, run_id, selection, selected_step
checkpoint_sha256, checkpoint_identity, source_identity
resolved_config_sha256, raw_dataset_sha256, LP_input_identity
reuse_or_inference, source_prediction_path_or_new_export
scene_count=20, axis_order=CHW, dtype=float32, units=DN, max_dn=2047
band_order, spatial_shape, normalization_and_clamp_policy
execution_device, runtime_version, exported_at
raw_export_status, rendering_status, cross_method_png_validation, delivery_status
```

`scene_manifest.json`에는 scene마다 export index, pred 상대 경로, GT 상대 경로, 원 source의 file/key/index 또는 실제 scene ID, shape·dtype·min/max·SHA256을 기록한다. band 명칭을 source에서 확인하지 못하면 임의로 부여하지 말고 원 band index와 미확정 사실을 적는다.

`raw_export_status=COMPLETE`는 아래 raw 검사 통과 후에만 사용한다. `rendering_status=SKIPPED_BY_USER`, `cross_method_png_validation=NOT_CHECKED_BY_SCOPE`로 적고, renderer가 없다는 이유로 raw 완료를 보류하지 않는다.

## 7. 검증 — raw 무결성만 필수, PNG 검증 없음

### 7.1 필수 검사

모든 NPY를 `np.load(..., allow_pickle=False)`로 다시 열어 총수·번호·shape·float32·finite를 확인한다. prediction/GT scene mapping과 단위·밴드 순서를 대조한다. 단순히 파일 수가 20개라는 이유만으로 장면 대응이 맞다고 판정하지 않는다.

NPY 재읽기 값은 exporter가 저장한 float32 배열과 정확히 같아야 한다. GT도 원본 scene slice를 float32·CHW로 변환한 값과 대조한다. 각 파일 SHA와 ZIP CRC를 검사하고, ZIP 안에 실제 배열이 들어 있는지 확인한다.

checkpoint SHA 불일치, prediction–GT pairing 불일치, shape/밴드/단위 불명, NaN/Inf는 `COMPLETE`를 선언할 수 없는 오류다. 값을 보정하거나 다른 checkpoint로 대체하지 말고 sensor별 원인을 반환한다.

### 7.2 RR 수치 대조

원 선택 결과 JSON을 동봉한다. 기존 RR evaluator를 바로 사용할 수 있으면 **저장 후 다시 읽은 pred와 GT**로 1회 재계산해 아래 원정밀도 값과 차이를 기록한다. 이 재계산은 오차맵 renderer 검증과 다른 작업이다.

| RR metric | WV3 TARGET 38380 | QB TARGET 47470 |
|---|---:|---:|
| ERGAS | 2.056060193721197 | 3.5612873217098007 |
| SAM | 2.78725977076841 | 4.382716698791735 |
| PSNR | 37.95123548109508 | 38.32335566176771 |
| SSIM | 0.9755018106766457 | 0.9627539129448743 |
| SCC | 0.9878930864916938 | 0.9834211183043315 |
| Q8 / Q4 | 0.921030341439477 | 0.937071392649955 |

기존 RR metric crop은 `[20:-21,20:-21]`, Q block은 32를 따르되, **NPY 저장 영상은 256×256을 유지**한다. HQNR·D_lambda·D_s는 FR 지표이므로 이번 RR-only 작업의 필수 대조 항목이 아니다.

동일 runtime 재평가와 장치 변경 재평가를 구분한다. 허용 오차를 미리 정의하지 못하면 차이를 그대로 기록하고 `PARITY_REVIEW_REQUIRED`로 표시한다. 목표 숫자를 맞추려고 예측·GT·checkpoint를 바꾸지 않는다.

외부 metric 의존성 부족 등으로 재계산하지 못해도 **출처와 raw pairing·무결성이 확인된 배열은 먼저 반환**하고 `metric_parity=NOT_RECOMPUTED`와 이유를 적는다. 오류가 없는 raw export와 정량 재현성 인증은 별도 상태다. 큰 수치 차이가 있으면 숨기지 말고 `metric_parity=REVIEW_REQUIRED`로 보고한다.

## 8. 실제 파일 전달과 완료 보고

검사한 package를 ZIP으로 묶는다. 두 sensor 중 하나만 끝났으면 sensor별 ZIP을 먼저 보내고 나머지는 미완료로 보고한다. 문서·manifest만 있는 ZIP을 실제 prediction package로 보고하지 않는다.

이 대화에서 회수할 수 있도록 다음 중 이미 승인된 경로로 전달한다.

1. ZIP을 이 대화에 첨부한다.
2. 기존 인증된 사용자 Drive/파일 저장소에 업로드하고 실제 파일 링크·byte 수·SHA256을 반환한다.

로컬 파일을 만들었다는 사실만으로 ChatGPT에서 바로 접근 가능하다고 보고하지 않는다. 업로드 권한/수단이 없으면 `READY_LOCAL_NOT_DELIVERED`로 적고 정확한 로컬 경로와 SHA256을 반환한다. 새로운 공개 공유 권한이나 credential을 만들지 않는다.

최종 보고 형식:

```text
WV3: TARGET 38380 / checkpoint SHA 검증 / pred20 + GT20 / reused 또는 inferred
QB:  TARGET 47470 / checkpoint SHA 검증 / pred20 + GT20 / reused 또는 inferred
raw 검사: shape / dtype / DN / scene pairing / finite / 파일 SHA / ZIP CRC
metric parity: PASS 또는 REVIEW_REQUIRED 또는 NOT_RECOMPUTED + 실제 차이/이유
rendering: SKIPPED_BY_USER (PNG 생성·calibration·renderer 탐색 없음)
package: 실제 ZIP 경로 / bytes / SHA256 / 접근 가능한 파일 링크 또는 전송 필요 상태
s1 기존 학습: 변경 없음, 추론 실행 장치와 실제 시각
미완료 항목: sensor별로 구분
```

**이번 작업의 완료 조건은 “두 지정 모델의 RR full-band prediction과 대응 GT가 실제 배열로 확보되어 전달됨”이다. 기존 PNG와의 색상 일치나 오차맵 완성은 완료 조건이 아니다.**

## 9. 자료 근거와 적용 범위

고정 run·step·SHA·밴드 수·DN·원 수치는 다음 기존 인계 자료에서 가져왔다.

- `OURS_QUALITATIVE_AUDIT_2026-09-24/ours_export_targets.json`
- `PANDA_QUAL_EXPORT_HANDOFF_2026-09-24/S1_WV3_QB_EXPORT_AND_DELIVERY_KR.md`
- `OURS_QUALITATIVE_AUDIT_2026-09-24/QUALITATIVE_EXPORT_AUDIT_KR.md`

대상 명세 원파일 SHA256: `f4ea887f51ed8f71d905068be74de9a67c9fd4ea39e66d655e0b1be8b981a1e5`.

기존 코드 감사 snapshot은 `6e746d43804c3a78ce781a7ce57adedb85b6431a`이며, 이것이 모든 run의 실제 학습 commit이라는 뜻은 아니다. 서버에서는 선택 checkpoint의 source/config/data identity와 호환되는 runtime을 사용한다.

본 문서의 **RR-only, rendering/calibration 생략, raw 완료 상태 분리**는 이번 사용자 지시를 반영한 운영 변경이다. 원 논문/공식 코드의 rendering 규칙을 새로 확정한 것이 아니다.
