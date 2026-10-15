# s1: 현재 paper의 WV3·QB Ours 결과를 다운로드 가능한 파일로 반환

작성일: 2026-09-24  
문서 유형: s1 실행 담당자에게 전달할 결과 추출·전달 지시서. **실제 결과 ZIP을 이 문서가 포함하는 것은 아니다.**

## 1. 현재 접근 상태와 목표

현재 대화에는 모델 지정 및 실험 기록이 있으나, 아래 s1 모델의 실제 full-band 예측 파일이나 검증된 정성 결과 ZIP은 회수되지 않았다. 연결된 Drive의 결과 파일 검색 및 GitHub 결과 검색/Release 목록에서도 대응 다운로드 package를 찾지 못했다. 이것은 s1 로컬 파일이 없다는 뜻은 아니다.

s1에서 출처가 검증된 기존 결과를 우선 찾아 묶는다. 지표 JSON만 있다면 지정 checkpoint로 inference-only를 수행해 WV3·QB 각각 RR20/FR20의 full-band 예측을 저장한다. 이후 **실제 ZIP과 SHA256 및 사용자가 접근 가능한 파일 링크**를 반환한다. 서버 내 절대 경로만 적는 것으로 전달 완료를 선언하지 않는다.

이번 사용자 요청의 학습 일시정지 대상은 **s4**다. s1의 현재 학습을 임의로 중단하지 않는다. 기존 예측 재사용 또는 실제로 비어 있는 GPU/승인된 실행 시간에 추론한다. 여기서 s1 학습 중단 권한을 확장하지 않는다.

## 2. 고정 대상 두 개

### WV3

```text
server: s1
run_id: FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1
selection: TARGET
step: 38380
checkpoint_sha256: 3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50
source: WV3-s1 row96 / paper row17
num_bands: 8
max_dn: 2047
```

### QB

```text
server: s1
run_id: QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1
selection: TARGET
step: 47470
checkpoint_sha256: 548d0ed2ccdc968d319e850e1470d7068f9321a9f0cc18e1794807f5cfd3399d
source: QB-s1 row16 / paper row34 and row71
num_bands: 4
max_dn: 2047
```

두 모델 모두 원 실행 루트에서 다음 상대 경로로 확인한다.

```text
work_dir/<run_id>/candidates/<step>/model.safetensors
work_dir/<run_id>/candidates/<step>/identity.json
work_dir/<run_id>/meta/config.resolved.yaml
work_dir/<run_id>/official/target_selection.json
```

원 frozen runtime/source/config/data identity를 검증한다. 각 run의 `best_hqnr`, `last`, EXACT50K를 대신 사용하지 않는다. WV3의 역사적 G23 모델, PAN-Crafter reproduction 모델, QB TA/S82009는 이번 대상이 아니다. QB paper의 중복 두 행은 export 한 개로 충분하다.

## 3. 기존 결과 검색 및 재사용 기준

정확한 run 디렉터리와 서버의 기존 export 위치에서 `.npy`, `.npz`, `.mat`, PNG/ZIP 및 manifest를 찾아본다. 처음부터 서버 전체 홈·다른 프로젝트·credential 디렉터리를 재귀 검색하지 않는다.

다음이 모두 연결되는 기존 결과만 재사용한다.

- 위 checkpoint SHA와 selected step 및 정상 A_ON의 전체 A/U model.
- 원 config/source와 native dataset/LP manifest 및 scene 순서.
- RR20/FR20 전체 scene, full-band 값·축·단위.
- 같은 배열에서 계산된 original metric 또는 재평가 결과.

`metrics.json`은 prediction file이 아니며, 같은 run에서 나온 PNG라는 사실만으로 선택 step을 증명하지 않는다. provenance가 없는 이미지는 참고/검토용으로 분리한다.

## 4. 예측이 없을 때의 inference-only

신규 학습·finetuning·Teacher 재학습·checkpoint selection은 하지 않는다. 기존 코드의 다음 함수와 원 dataset manifest를 재사용하는 작은 wrapper를 만든다. 함수 signature는 실제 frozen checkout에서 확인한다. 이들은 단독 CLI 이름이 아니다.

```text
WV3: fh20r1.common.load_checkpoint_model → fh12.evaluation.infer
QB:  qg40.common.load_checkpoint_model → qg40.evaluation.infer
```

FP32, eval, no-grad, shuffle=False, 전체 Student Aligner+frontend+U의 정상 A_ON. GT는 forward에 넣지 않는다. training augment/shift/masking/다른 checkpoint의 A/U 조합은 금지한다.

RGB 변환 전에 full-band prediction을 scene별 float32 `(C,H,W)`, DN 단위로 보존한다.

| dataset | reduced | full |
|---|---|---|
| WV3 | 20 × `(8,256,256)` | 20 × `(8,512,512)` |
| QB | 20 × `(4,256,256)` | 20 × `(4,512,512)` |

기존 infer의 DN 변환 및 clamp 정책을 유지하고 manifest에 명시한다. metric 계산의 RR border crop은 전체 raw 저장이나 PNG 크기를 바꾸는 지시가 아니다.

## 5. 반환 package

두 데이터셋을 분리하면 파일 회수와 검증이 쉽다. 이름의 timestamp는 실제 생성 시각으로 기록한다.

```text
Ours_s1_WV3_raw_<timestamp>.zip
Ours_s1_QB_raw_<timestamp>.zip
```

각 ZIP은 다음 구성을 갖춘다.

```text
Ours_s1_<dataset>_raw_<timestamp>/
  README.md
  manifest.json
  scene_manifest.json
  Ours_raw/<dataset>/
    reduced/pred/testimg0.npy ... testimg19.npy
    full/pred/testimg0.npy ... testimg19.npy
  references/<dataset>/
    reduced/gt/...          # RR error 재계산에 필요
    reduced/pan/...
    reduced/ms/...
    full/pan/...
    full/ms/...
  provenance/
    checkpoint_identity.json
    config.resolved.yaml
    dataset_manifest.json
    source_identity.json
    checkpoint.sha256
  metrics/
    original_selection.json
    re_evaluation.json
    metric_comparison.json
  rendering_requirements.json
  checksums.sha256
```

reference는 해당 test scene slice만 포함한다. 재배포가 허용되지 않는 reference는 제외하고 dataset ID/SHA 및 제외 이유를 기록한다. 별도 요청 없는 train dataset, optimizer state, credential, SSH key, 서비스 계정 JSON, 전체 환경변수는 넣지 않는다.

필수 manifest: run/step/SHA, 데이터 및 LP SHA, 원 scene ID 또는 correspondence 미확정 표시, H5/MAT key/index, 저장 파일 hash, band order, axis, dtype, DN 단위, 정규화/clip 정책, 원 runtime, 실제 실행 장치/시각, 추론 여부 또는 재사용 출처.

단순 출력 배열 용량만 float32 기준으로 계산하면 WV3 약 200 MiB, QB 약 100 MiB다. 이 수치는 reference/metadata/압축을 포함하지 않는 계산값이며 ZIP 크기 예측이 아니다.

### 대조할 원 수치

| dataset | HQNR | ERGAS | D_lambda | D_s |
|---|---:|---:|---:|---:|
| WV3 | 0.9587325529698706 | 2.056060193721197 | 0.01579909375823136 | 0.025886448898951097 |
| QB | 0.9251135521548385 | 3.5612873217098007 | 0.04423648652689379 | 0.03217561500686554 |

나머지 수치는 함께 전달한 `ours_export_targets.json`에 있다. 원 배열 기반 evaluator와 사전에 정한 재현성 허용 오차로 비교하고, 허용 오차 근거가 없으면 차이를 그대로 적어 `PARITY_REVIEW_REQUIRED`로 보고한다. 그림을 보고 유리한 step/seed로 갈아끼우지 않는다.

## 6. PNG와 기존 시각화 코드

첨부 CANConv는 각 dataset마다 full의 output/compare_output, reduced의 output/compare_output/errormap/compare_errormap, 총 120 PNG다. WV3·QB 모두 만들면 240 PNG다.

동일 renderer와 scene/ROI 설정이 확보되면 다음 최상위 형식으로 별도 visual package를 만든다.

```text
Ours/WV3/{full,reduced}/...
Ours/QB/{full,reduced}/...
```

raw만 확보된 상태에서도 반환을 지연시키지 않는다. RGB band indices, shared stretch/gamma, colormap·error limits, ROI/inset/보간 설정은 기존 비교 결과의 원 스크립트/config에서 가져온다. 원 설정이 없다면 `RAW_READY_RENDER_PENDING`으로 표시하고 임의의 PNG를 논문 비교 완성본으로 표시하지 않는다. 원 renderer 파일과 설정이 있으면 같이 전달하되 credential과 무관한 코드만 넣는다.

WV2는 이번 두 archive의 완료 조건이 아니다. 추가 작업은 현재 WV3 step38380 checkpoint의 zero-shot으로 별도 수행한다.

## 7. ChatGPT에서 회수 가능한 전달 방법

**현재 세션은 s1 로컬 파일시스템과 자동으로 공유되지 않는다.** 실제 ZIP을 아래 둘 중 하나의 경로로 전달해야 이 대화에서 검사·정리할 수 있다.

1. ZIP을 이 대화에 파일로 첨부한다.
2. 사용자의 연결된 Google Drive 또는 이미 승인된 파일 저장소에 ZIP을 올리고 정확한 파일 링크와 SHA256을 전달한다. 링크는 수신자가 실제 raw bytes에 접근할 수 있어야 한다.

서버에서 기존 인증된 업로드 수단이 없다면 신규 credential/공개 권한을 만들지 말고 로컬 ZIP 경로와 SHA를 먼저 반환한다. 사용자 측 전송이 필요한 상태를 명시한다. GitHub 코드 경로만으로 서버 weight/prediction을 다운로드할 수 있다고 보고하지 않는다.

파일을 전송한 뒤에는 ZIP CRC, 파일 수·크기·finite, scene manifest, model/data SHA, package SHA 및 renderer 출처를 확인해야 한다. 이 검증 전에는 paper-ready 또는 다운로드 확인 완료로 표시하지 않는다.

## 8. 마지막으로 반환할 실행 보고

```text
WV3: run / 38380 / expected SHA / RR20+FR20 / reuse or fresh inference
QB:  run / 47470 / expected SHA / RR20+FR20 / reuse or fresh inference
지표 대조 및 미확정 scene/renderer 사항
각 ZIP: 실제 절대 경로 / byte 수 / SHA256 / 실제 접근 가능한 파일 링크
s1 기존 학습 상태: 변경 없음 또는 별도로 승인된 변경의 정확한 기록
```

두 archive 중 하나만 완료되었으면 그것부터 전달하고 나머지는 미완료로 적는다. manifest·MD·검사 결과만 들어 있는 작은 bundle을 실제 prediction package라고 부르지 않는다.

## 9. 근거

고정 추출 대상과 숫자는 `OURS_QUALITATIVE_AUDIT_2026-09-24/ours_export_targets.json` 및 동일 감사 MD를 따른다. 저장소에서 확인된 원 디렉터리/loader 규약은 다음이다.

- `fh20r1/common.py`, `fh20r1/postrun.py`, `fh12/evaluation.py`
- `qg40/common.py`, `qg40/postrun.py`, `qg40/evaluation.py`

조회 snapshot: `6e746d43804c3a78ce781a7ce57adedb85b6431a`.
https://github.com/hojunking/PAN-Crafter-repro/tree/6e746d43804c3a78ce781a7ce57adedb85b6431a

현재 실행 runtime 및 실제 서버 파일 존재는 실행 담당자가 다시 확인한다. 이 문서만으로 s1 추론·파일 업로드가 수행되었다는 뜻은 아니다.
