# s1 추가 지시서 — WV3·QB FR 예측 20장씩 추출·반환

작성일: 2026-09-25  
작업 ID: `PANDA_S1_WV3_QB_FR_RAW_SUPPLEMENT_v1`  
범위: **같은 논문용 checkpoint의 FR full-band prediction 40개만 추가 확보**  
상태: 실행 지시서. 이 문서는 실제 추론·서버 파일 존재·업로드 완료를 증명하지 않는다.

## 1. 이번 추가 작업의 이유

초기 `S1_WV3_QB_EXPORT_AND_DELIVERY_KR.md`는 RR20/FR20을 함께 요청했다. 이후 `S1_WV3_QB_RR_RAW_GT_EXPORT_2026-09-24_KR.md`에서는 RR prediction+GT를 먼저 반환하도록 범위를 줄였고, FR 신규 추론을 필수에서 제외했다. 따라서 RR 패키지가 준비되어 있어도 FR prediction이 포함되어 있다고 가정하지 않는다.

이번에는 아래 두 checkpoint의 FR raw만 추가한다. RR 재추론·GT 재추출·오차맵 생성·renderer 탐색/검증은 하지 않는다. 기존 RR 파일과 이전 결과를 덮어쓰지 않는다.

사용자는 기존 preview에서 PAN-Crafter와 FR 순서가 일치함을 확인했다고 전달했다. 그 대응을 유지하며 **FR 원본 H5의 행 index 0–19를 testimg0–testimg19에 그대로 대응**한다. 새 장면 재정렬이나 외부 PNG 기반 재정합은 하지 않는다.

## 2. 모델은 RR 추출 때 지정한 동일한 두 개

| 항목 | WV3 | QB |
|---|---|---|
| 원 실행 서버 | s1 | s1 |
| 선택 | TARGET | TARGET |
| 선택 update | 38380 | 47470 |
| 밴드 수 | 8 | 4 |
| FR scene 수 | 20 | 20 |
| NPY 한 개의 shape | `(8,512,512)` | `(4,512,512)` |
| 저장 | float32 / CHW / DN | float32 / CHW / DN |
| 값 범위 | 0–2047 | 0–2047 |

### WV3

```text
run_id: FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1
checkpoint_sha256: 3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50
checkpoint:
work_dir/FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1/candidates/38380/model.safetensors
config:
work_dir/FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1/meta/config.resolved.yaml
```

### QB

```text
run_id: QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1
checkpoint_sha256: 548d0ed2ccdc968d319e850e1470d7068f9321a9f0cc18e1794807f5cfd3399d
checkpoint:
work_dir/QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1/candidates/47470/model.safetensors
config:
work_dir/QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1/meta/config.resolved.yaml
```

각 checkpoint 폴더의 `identity.json`, 원 run의 `official/target_selection.json`을 함께 대조한다. 상대 경로의 실제 루트와 frozen checkout은 서버의 기존 runtime 기록을 따른다. 별칭 `best_hqnr`, `last`, EXACT50K로 바꾸지 않으며 G23, Teacher, PAN-Crafter reproduction 결과로 대체하지 않는다.

## 3. FR 입력 데이터의 위치 확인

### WV3 — 저장소 정의에 구체적인 입력 경로가 있다

```text
data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5
```

위는 `config/FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1.yaml`의 `test_full_feeder_args.dataroot`다. 같은 config는 `fh20r1.dataset_manifest`를 아래로 지정한다.

```text
work_dir/_fh12/s1/dataset_manifest.json
```

**이 H5는 모델에 넣을 원본 FR 입력이며 Ours prediction NPY가 아니다.** 서버에 저장된 run별 resolved config, FR manifest, 실제 raw/LP 파일과 SHA를 따라 입력을 결정한다. 저장소 경로 조회만으로 현재 디스크 존재를 확인한 것으로 보고하지 않는다.

### QB — 반드시 서버의 resolved config를 읽는다

저장소 `config/qg40/QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1.yaml`는 실행 전 정의로, `test_full_feeder_args.dataroot`와 `qg40.dataset_manifest`가 `null`이다. 이 파일만 보고 FR 입력이 없다고 결론내리거나 WV3 경로를 문자열 치환해 사용하지 않는다.

대상 run의 `meta/config.resolved.yaml`에서 **실제 `qg40.dataset_manifest` 또는 실행 시 바인딩된 sensor/data 기록**을 읽고, 해당 manifest의 FR split이 지정한 raw H5와 원 LP 입력을 사용한다. 최근 RR export가 사용한 같은 run/source 기록을 출발점으로 하되, 데이터 split은 FR로 바꾼다. 실제 경로가 해석되지 않으면 무엇이 빠졌는지 구체적으로 보고한다.

## 4. 실행 절차

### 4.1 검증된 기존 FR raw가 있으면 재사용

정확한 run 폴더와 해당 run의 알려진 export 폴더에서 `.npy`, `.npz`, `.mat`, 예측 `.h5` 및 manifest를 확인한다. 지정 checkpoint SHA, FR input identity, scene 순서, full-band 값이 연결되는 경우에만 재사용한다.

기존 배열이 `[20,C,512,512]`면 scene별로 나누어 저장한다. HWC/NHWC 원본은 의미를 확인한 뒤 transpose한다. 메트릭 JSON, RGB PNG, 다른 선택점의 결과는 raw prediction을 대신하지 못한다.

### 4.2 저장된 FR prediction이 없으면 FR-only inference

확인된 함수 경로:

```text
WV3: fh20r1.common.load_checkpoint_model -> fh12.evaluation.infer
QB:  qg40.common.load_checkpoint_model   -> qg40.evaluation.infer
```

이들은 독립 실행 명령이 아니라 wrapper가 호출할 함수다. 실제 frozen checkout에서 signature와 데이터 loader를 확인한다. **각각 FR dataset만 만들어 `infer`를 호출**하며, RR/validation을 함께 계산하는 전체 `evaluate_model`이나 원 campaign controller를 다시 기동하지 않는다.

서버에 `export_worker.py`가 있으면 먼저 코드/도움말을 확인해 full/FR split export를 지원하는지 확인하고 재사용한다. 확인하지 않은 경로·CLI 인자를 가정하지 않는다. 필요하면 기존 worker에 별도 FR-only wrapper를 추가하되 원 모델/학습 파일은 바꾸지 않는다.

추론 조건은 FP32, eval, no-grad, batch_size=1, shuffle=False, drop 없음이다. 같은 checkpoint의 Student Aligner+frequency frontend+U 전체를 정상 A_ON으로 사용한다. 512×512 native FR 입력을 처리하며, RR 예측의 upsampling은 FR inference가 아니다. GT/새로운 warp/augmentation/TTA/fine-tuning을 추가하지 않는다.

현재 확인한 `infer`는 `(prediction_array, deltas)`를 반환하고 prediction은 이미 DN 단위다. 이를 받아 파일로 저장해야 한다. **HQNR 결과가 기록되어 있다고 FR NPY도 저장된 것으로 가정하지 않는다.** DN 출력에 2047을 다시 곱하지 않는다. 원 evaluator의 clamp/역정규화를 유지하며 별도의 rounding, RGB 선택, stretch, gamma, sharpen을 적용하지 않는다.

### 4.3 s1의 현재 실험은 보존

현재 실행/GPU 점유를 먼저 확인한다. 기존 학습 중단·queue 변경·watchdog 해제 권한은 이번 문서에 없다. 기존 raw 재사용을 우선하고 신규 추론은 비어 있는 GPU 또는 별도 허용된 자원에서 진행한다. CPU 추론을 사용했다면 장치 차이를 기록한다. GPU를 쓰기 위해 다른 학습을 무조건 병렬 실행하거나 강제 종료하지 않는다.

## 5. 반환 형식 — 필수는 NPY 40개

ZIP을 풀었을 때 아래 상대 경로가 유지되도록 한다.

```text
Ours_raw/
  WV3/full/pred/testimg0.npy ... testimg19.npy
  QB/full/pred/testimg0.npy ... testimg19.npy
manifest.json
scene_manifest.json
provenance/
  WV3/checkpoint_identity.json
  WV3/original_selection.json
  QB/checkpoint_identity.json
  QB/original_selection.json
validation/raw_array_checks.json
checksums.sha256
```

각 NPY는 float32, CHW, DN(0–2047)이며 object array가 아니다. 파일명은 0-based이고 scene order는 FR raw H5의 행 순서다. 원본이 동일한지 확인하는 file SHA와 각 source index를 scene manifest에 기록한다. 사용자가 확인한 cross-method 순서는 `USER_CONFIRMED_PREVIEW_ORDER`로 적으며 이 작업에서 다시 검증했다고 주장하지 않는다.

PAN/LMS 참조는 **전달 패키지에서 선택**이다. 없으면 downstream에서 사용자가 확보한 동일 장면의 PAN-Crafter 참조를 사용하는 것으로 처리하고 export를 지연시키지 않는다. 다만 이것은 **우리 모델 forward의 원 PAN/MS/LP 입력을 임의로 생략·교체해도 된다는 뜻이 아니다.** 모델에는 원 run이 사용한 입력 및 frontend 규약이 필요하다.

FR GT와 errormap은 요청하지 않는다. 후속 PNG는 `full/output`과 `full/compare_output`만 만들면 되며, 이번 raw export에서 PNG 생성이나 renderer 검증을 완료 조건으로 두지 않는다. 비교 모델 raw·기존 PNG 스크립트·ROI 설정을 기다리지 않는다.

## 6. 최소 검증과 보고

40개 NPY를 `np.load(..., allow_pickle=False)`로 재읽어 파일 번호 0–19, dtype, shape, NaN/Inf 없음, min/max, SHA256을 검사한다. 기록한 원 FR H5 index와 출력 index가 동일해야 한다. 저장 후 수치가 inference 반환 float32 배열과 같은지 확인한다.

manifest에 dataset/run/step/checkpoint SHA, 실제 FR source와 SHA, band order, axis/dtype/units, 장치, 생성 시각, 재사용 또는 새 추론 여부를 남긴다. 상태는 `FR_RAW_COMPLETE`와 `RENDERING_SKIPPED_BY_SCOPE`로 분리한다.

FR evaluator를 바로 재사용할 수 있으면 저장된 배열의 HQNR/D_lambda/D_s를 원 선택 기록과 1회 대조한다. 이것은 renderer 검증이 아니다. 추가 평가 의존성이 없어 raw 전달을 막지는 않으며, 미재계산이면 `NOT_RECOMPUTED`를 기록한다. 불일치를 발견하면 그대로 기록하고 다른 checkpoint로 바꿔 맞추지 않는다.

| 원 선택 FR metric | WV3 / 38380 | QB / 47470 |
|---|---:|---:|
| HQNR | 0.9587325529698706 | 0.9251135521548385 |
| D_lambda | 0.01579909375823136 | 0.04423648652689379 |
| D_s | 0.025886448898951097 | 0.03217561500686554 |

실제 생성 시각으로 `Ours_s1_WV3_QB_FR_raw_<timestamp>.zip`을 만들고 CRC, byte 수, SHA256을 확인한다. 배열 본체만 240 MiB(WV3 160 MiB + QB 80 MiB)이며 압축 파일 크기는 실제 측정한다. train data, optimizer state, credential/SSH key/서비스 계정 JSON, 모델 weight는 포함하지 않는다.

사용자가 이미 승인한 저장소/연결된 Drive로 전송하고 실제 다운로드 링크를 반환한다. 인증된 업로드 수단이 없으면 실제 로컬 ZIP 경로와 SHA를 먼저 반환하고 전송 미완료를 명시한다. 서버 경로만 있는 상태를 여기서 다운로드 가능한 상태라고 부르지 않는다. 한 sensor만 먼저 완료되면 그 20장부터 별도 ZIP으로 반환한다.

최종 보고에는 다음을 적는다.

```text
WV3: TARGET38380 / checkpoint SHA 확인 / FR20 / reuse or inference
QB: TARGET47470 / checkpoint SHA 확인 / FR20 / reuse or inference
shape=float32 CHW / DN0–2047 / FR H5 index order 유지
각 archive: 실제 경로 / byte 수 / SHA256 / 사용자 접근 가능한 링크
s1 기존 학습: 변경 없음 또는 별도 승인된 변경 내역
```

## 7. 근거와 확인 한계

- 고정 대상: 기존 `ours_export_targets.json`과 `S1_WV3_QB_RR_RAW_GT_EXPORT_2026-09-24_KR.md`. 모델을 재선택한 문서가 아니다.
- 2026-09-25 연결 저장소 조회: WV3 config의 FR H5/manifest 경로, QB 저장소 config의 미바인딩 null 상태, `fh12/evaluation.py`와 `qg40/evaluation.py`의 FR infer 반환 규약을 확인했다.
- 저장소 코드가 조회된 것과 해당 s1 디스크 파일·export가 현재 존재하는 것은 별개다. 실제 raw가 이미 만들어졌는지는 s1 실행 담당자가 확인한다.
- 관련 코드: `config/FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1.yaml`, `config/qg40/QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1.yaml`, `fh12/evaluation.py`, `qg40/evaluation.py`.
