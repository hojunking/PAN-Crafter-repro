# Native FR·scene prior 준비 도구

이 도구는 **원본 관측 자료로부터 준비물을 만드는 단계**다. 학습 실행, 모델/metric 변경, Sheet 업로드, 기존 데이터 덮어쓰기를 하지 않는다. 생성물은 `STAGED_NOT_ADMITTED` 또는 `accepted=false`이며, 원본·매핑 검증이 끝나기 전에 실행 가능한 training manifest로 위장하지 않는다.

## 필요한 실물과 현재 확인 사항

FR 평가 자료는 있다. WV3/QB/GF2 각각 공식 20장의 PAN512/MS128/LMS512를 보유한다. 이는 `data/PanCollection/<sensor>/full_examples_mat20/` 아래 파일이다. 별도 native FR training source는 아니다.

기존 training H5의 PAN64/MS16/GT64에는 GT64와 대응하는 **다운샘플 전 PAN256**이 들어 있지 않다. `_pan.h5`도 LPAN16이지 원본 PAN256이 아니다. RR64 PAN의 확대는 native 관측의 복원이 아니며, LRMS를 다시 확대하는 것으로 대체할 수 없다.

새 방법을 기존 benchmark split과 함께 실행하려면 다음 사실이 필요하다.

1. 해당 센서의 별도 native PAN/MS 학습 원본과 실제 acquisition/source scene 식별 근거.
2. PAN/MS 공통 FOV·좌표 원점·ratio·기존 준비 변환. 이번 도구는 PAN을 미리 정렬하지 않는다.
3. 기존 RR GT/patch와 native MS의 연결, RR PAN 생성 recipe와 좌표 phase 검증.
4. 학습 영역과 공식 평가 영역의 분리 및 필터가 참조하는 영역의 분리.

사용자가 새 JSON을 직접 작성할 필요가 있다는 뜻이 아니다. 원본 파일 위치와 원천 metadata가 확인되면 준비 코드와 담당자가 hashes/catalog/좌표 index/진단을 만든다. 정보가 없는 H5 row에 scene 이름을 붙이는 것으로 acquisition 증거가 생기지는 않는다.

## 구현

| 파일 | 기능 |
|---|---|
| `prepare_inventory.py` | 입력 경로의 H5 header/작은 attrs 조사, inode 중복·symlink 순환 방지, 역할 후보 분류 |
| `prepare_assets.py` | read-only SHA-pinned raster, native FR crop H5/index, global LMS 보간, 정확 GT 매칭 후보 |
| `prepare_prior.py` | native LMS/PAN structural matching과 robust scene prior 후보·신뢰도 진단 |
| `prepare.py` | 별도의 offline CLI; 기존 trainer 진입점은 변경하지 않음 |

파일 목록의 shape는 provenance가 아니다. inventory는 학습 승인이나 이미지 전수 검사 없이 metadata를 보고 후보만 분류한다. 데이터 읽기 단계는 명시된 H5/NPY, CHW/HWC/HW/NCHW layout과 SHA를 사용한다. NCHW index는 필요하지만 original scene ID로 해석하지 않는다. TIFF/MAT 원본은 geometry와 layout을 보존하는 별도 adapter가 필요하며 자동 변환하지 않는다.

## CLI

```bash
python -m teacher48.prepare --help
python -m teacher48.prepare inventory \
  --root data/PanCollection \
  --output results_log/native_inventory.json
```

아래 `sources.native.json`, `prior.config.json`은 원본 확인 후 만드는 입력 파일의 예시다. 현재 launch-ready 자산이 존재한다는 뜻이 아니다. 출력은 새 경로를 사용하며 기존 파일/디렉터리 덮어쓰기를 거부한다.

```bash
python -m teacher48.prepare crop-native \
  --catalog sources.native.json --output data/prepared/native_fr_v1 \
  --patch 256 256 --stride 256 256 --split-margin 0

python -m teacher48.prepare match-rr \
  --rr-h5 data/PanCollection/WV3/train_wv3.h5 --gt-key gt --row 0 \
  --catalog sources.native.json --max-scene-pixels 16777216 \
  --output results_log/rr0.mapping-candidates.json

python -m teacher48.prepare prior \
  --catalog sources.native.json --scene-id VERIFIED_SCENE_ID \
  --config prior.config.json --max-pixels 16777216 \
  --output results_log/scene.prior-proposal.json
```

예시의 256 patch와 메모리 제한은 명령 설명용이며 실제 실험 설정을 확정한 값이 아니다. `--max-pixels`는 PAN pixel 수, `--max-scene-pixels`는 native MS pixel 수 기준이다.

위 `--split-margin 0`은 추가 여유를 0으로 설정한 예시이며 **보간 support 검사를 끄지 않는다**. 실제 센서의 지리적 분리 buffer를 이 예시로 자동 승인하지 않는다. 큰 scene은 명시된 크기 제한을 넘으면 실패한다. 메모리를 줄이려고 자동 축소하거나 일부 scene을 건너뛰어 유일 매칭인 것처럼 기록하지 않는다.

## Native source catalog 계약

상위 필드:

```text
schema = TEACHER48_NATIVE_SCENES_V1
dataset = WV3 | QB | GF2
split_basis = 실제 train/eval 구분의 출처 설명
heldout_crops = [{acquisition_id, source_crop_native_pan_yxhw}, ...]
scenes = [...]
```

각 scene 필수 필드:

```text
source_scene_id, acquisition_id, source_identity_basis, geometry_basis
native_pan_grid, pan_origin_native_yx
role = native_training_source
radiometry = native_DN, ratio = 4
native_source_evidence = {path, sha256}
geometry_evidence = {path, sha256}
pan, ms = {path, sha256, key, layout[, index]}
lms_mode = provided_global | REPO_INTERP23TAP_R4_FLOAT64_GLOBAL_WRAP_v1
```

`provided_global`이면 `lms` array reference와 `lms_support={radius_native_pan,boundary,basis}`를 추가한다. boundary는 `wrap` 또는 `bounded`다. 주어진 LMS를 자르는 것과 그 LMS가 생성될 때 참조한 MS support를 구분한다. prior CLI에는 `pixel_scale=[sy,sx]`, `origin_phase=[py,px]`도 명시한다. 값을 영상 크기만으로 추정하지 않는다.

`heldout_crops=[]`도 split 근거가 있을 때만 의미가 있다. 빈 목록을 독립 평가와의 지리적 분리 증명으로 바꾸지 않는다. 준비 결과는 전체 official eval index·prior·RR 변환·near-duplicate audit가 결합된 `TEACHER48_SOURCES_V1`이 아니며 자동 admission 경로가 없다.

## LMS와 필터 support

- 제공 LMS가 있으면 pixels/dtype를 그대로 보존한다. PAN/MS는 보간·warp·clip하지 않는다.
- 생성 모드를 명시하면 기존 `align.resample.interp23tap`의 ratio4, float64, global circular 경계 recipe를 사용한다. bicubic으로 대체하지 않는다.
- 작은 crop에 독립 보간을 적용하지 않는다. whole-scene에서 16 LR pixel halo를 읽고 보간한 뒤 중앙 crop을 남긴다. scene 경계는 같은 global wrap을 읽는다.
- crop의 native 위치와 LMS의 전체 참조 support를 heldout 좌표와 대조한다. 생성 모드는 보수적인 64 PAN pixel radius를 적용하고, wrap으로 반대쪽 scene 가장자리를 참조하는 경우도 검사한다.
- 이 recipe는 명시적 생성 선택이다. 출처 불명의 제공 데이터 LMS와 비트 동일하다고 주장하지 않는다. 별도 실측 일치 검증이 필요하다.

## RR 매핑과 prior의 의미

RR 매핑은 GT와 native MS의 모든 밴드/화소가 **정확히 일치**하는 좌표를 8개 flip/rotation에 대해 찾는다. 근사 일치는 승인하지 않고 다중/대칭 매칭은 `AMBIGUOUS`다. 유일 매칭도 `EXACT_GT_ONLY_CANDIDATE`이며, native PAN 대응 및 RR degradation/phase까지 증명한 것으로 취급하지 않는다. 기존 RR과 원본이 실제로 연결되지 않으면 원본에서 RR을 재구성하는 별도 data revision이 필요하다.

prior는 PAN이 moving, native LMS 평균이 reference인 Scharr/ZNCC+census 추정값을 사용한다. 부호는 `aligned_pan[y,x]=pan[y+dy,x+dx]`다. legacy estimator의 실패 시 적용값 0을 쓰지 않고 raw 값과 gate 사유를 보존한다. 새 deterministic weighted-median translation consensus는 과거 RANSAC 감사와 bitwise 동일한 구현이라고 주장하지 않는다.

window/stride/texture/consensus/confidence 등 숫자는 전부 기록된 명시적 config다. `audited_hr_local_gates()`는 기존 HR 진단의 검색 범위·local gate만 제공하며 scene acceptance 정책을 자동 결정하지 않는다. 모든 prior 출력은 `accepted=false`이고, 수치 기준 통과는 `ready_for_review=true`일 뿐이다. GT를 보지 않으며 해당 acquisition에 heldout crop이 있으면 현재 whole-scene prior CLI는 사용을 거부한다. 이 경우 train-only window를 통한 별도 estimator 확장이 필요하다.

## 검증과 한계

합성 테스트는 native pixels 보존, global interp23tap 전체-image parity, edge wrap/heldout support, 해시·외부 H5 dependency 거부, 좌표·8방향 변환 복구, prior 부호·outlier·zero/flat 거부를 확인한다. 실물 FR20이 존재하는 사실과 별개로, 검증된 native training source가 없으면 실제 FR train H5를 만들지 않는다.

원본 H5의 external link/VDS/external storage는 바깥 파일 SHA만으로 내용이 고정되지 않으므로 이 최소 adapter에서 거부한다. 준비 실패 시 생성한 부분 파일과 `FAILED.json`을 남겨 복구할 수 있게 하고 성공 manifest를 쓰지 않는다. 기존 파일은 제거하지 않는다.
