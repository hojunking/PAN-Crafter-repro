# GF2 저장 결과 선별 — s3 CPU 후처리

학습과 export가 완료된 뒤 실행한다. **재학습·GPU 추론·weight 이동·Sheet 게시가 없다.**
입력 파일을 읽기만 하며 결과는 새로운 `selections/` 폴더에 쓴다. 기존 학습/replay
release와 독립된 명령이므로, 이 작업을 위해 `enqueue`나 `run`을 다시 실행하지 않는다.

## 실행

사용자가 push한 뒤 s3에서 전체 커밋을 pull한다. NumPy, SciPy, Pillow가 있는 기존
`pancrafter` Python 환경을 사용한다. Torch/CUDA는 이 선별 코드에서 import하지 않는다.

```bash
git pull --ff-only
python -m panda_gf2_sharpness.select --verify
python -m panda_gf2_sharpness.select
```

기본 입력: `work_dir/panda_gf2_sharpness/exports/` 아래 모든 `manifest.json`.
기본 출력: `work_dir/panda_gf2_sharpness/selections/YYYYMMDD_HHMMSS/`.
재실행할 때 새 출력 폴더를 사용한다. 기존 결과 폴더를 덮어쓰지 않는다.

서로 다른 추가 실험 폴더를 함께 비교할 경우:

```bash
python -m panda_gf2_sharpness.select \
  --exports work_dir/panda_gf2_sharpness/exports /실제/추가/export/폴더 \
  --output work_dir/panda_gf2_sharpness/selections/all_steps_review
```

`panda_sensor_visuals.export` 형식의 manifest, provenance, SHA ledger, FR20 NPY,
PAN PNG, render_metadata가 필요하다. 표준 형식의 하위 폴더를 재귀 검색하므로 step
목록을 다시 지정할 필요가 없다. 임의 구조의 NPY 폴더를 추측하여 연결하지 않는다.
기존 export의 선택 checkpoint weight가 삭제됐더라도 저장 출력과 근거가 온전하면 읽을 수 있다.

Docker를 사용한다면 학습용 GPU 옵션 없이 다음처럼 실행할 수 있다. 입력이 symlink이면
실제 경로를 mount하고, output은 입력 폴더 밖의 별도 writable 경로로 둔다.

```bash
mkdir -p work_dir/panda_gf2_sharpness/selections
docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$(pwd),dst=/workspace,readonly" \
  --mount "type=bind,src=$(realpath work_dir/panda_gf2_sharpness/exports),dst=/inputs,readonly" \
  --mount "type=bind,src=$(realpath work_dir/panda_gf2_sharpness/selections),dst=/outputs" \
  --workdir /workspace pancrafter-env:gf2-sharpness \
  python -m panda_gf2_sharpness.select --exports /inputs --output /outputs/all_steps_review
```

## 기준 모델과 검증

- 정성 기준: `GF2_C60LR150 R1 @47K`.
- 정량 기준: `GF2P_C60LR200 R2 @47K` (HQNR .964669, ERGAS .514883, PSNR 45.084138).
- 기본적으로 정성 기준의 실제 export를 사용한다. 과거 weight 정리 등으로 이 export가
  없으면 **고정된 정량 기준 C60LR200 R2@47K**를 paired 비교 기준으로 사용하고
  `QUANTITATIVE_REFERENCE_FALLBACK_QUALITATIVE_EXPORT_MISSING`을 보고서에 명시한다.
  이 경우 개선율은 이전 정성 Ours가 아닌 정량 기준 대비다. 둘 다 없으면 임의로 성능이
  낮은 후보를 골라 기준으로 삼지 않는다. `--baseline /실제/export/폴더`로 명시해야 한다.
  정량 기준 자체가 누락되면 그 참조 ID는 `null`로 기록한다.
- run/step/config/checkpoint SHA, 동일 상태 RR20·FR20 평가, NPY 파일/배열 SHA를 검증한다.
  같은 run/step에 서로 다른 checkpoint SHA가 있으면 양쪽을 제외한다.
- 모든 후보의 PAN 이미지와 shared-reference RGB bounds가 장면별로 동일해야 한다.
- 손상·누락 출력은 `exclusions.json`에 남기며 평가 실패를 낮은 성능으로 대체하지 않는다.
  기준 모델이 손상됐으면 중단한다. 동일 checkpoint의 중복 export는 RR도 있는 쪽을 우선한다.

## 세 측정 경로

1. `legacy_*`: 원래 각 출력별 1–99% PNG 표시를 NPY에서 재생성해 기존 지표를 계산한다.
   U-Know 경계 1.150082, PAN-Crafter 경계 1.227738, 기존 Ours 평탄부 .009428과
   비교하는 것은 이 경로뿐이다. 외부 방법 값은 전달받은 기록이며 재추론한 값이 아니다.
2. `shared_*`: stored LMS에서 정한 동일한 RGB bounds로 모든 모델을 변환한다.
   float64 [0,1]에서 계산해 PNG 양자화 영향을 피한다. 위 legacy 절대 임계값을 적용하지 않는다.
3. `raw_*`: 원본 RGB 평균을 1023으로 나눈 값에서 계산한다. clipping하지 않는다.
   범위 이탈률도 별도로 기록한다. PAN mask는 동일한 보관 PAN에서 고정한다.

모두 FR20 장면별 지표의 산술평균이다. `flat_grain`은 local std이므로 순수 잡음만
측정하는 값이 아니다. PAN은 FR GT가 아니며, 기울기/고주파 양만으로 실제 구조 복원을
확정하지 않는다. RR NPY와 GT 참조가 있는 경우 4밴드 Sobel gradient 오차도 계산하지만,
이는 공식 ERGAS/SAM/PSNR을 대체하는 지표가 아니다.

## 후보 규칙

- `visual`: legacy 경계≥U-Know, legacy 평탄부≤기존 Ours이고,
  shared/raw 모두 기준 대비 경계가 감소하지 않고 평탄부가 증가하지 않으며,
  **같은 장면에서 shared/raw 경계가 함께 개선된 장면이 14/20 이상**인 후보.
- `quantitative`: HQNR≥.9645, ERGAS≤.522, PSNR≥45.08을 모두 만족하는 후보를
  shared 경계 순으로 정렬한다. 정성 규칙을 통과했다고 간주하지 않는다.
- `compromise`: 정성 최상 후보의 shared 경계 95% 이상을 유지하면서 raw/shared 잡음
  조건과 장면 일관성을 만족하는 joint Pareto 후보 중 ERGAS가 낮은 순이다.
  정량 세 목표를 모두 달성했다고 간주하지 않는다.
- `provisional_frontier`: shared 경계↑/평탄부↓ 양쪽에서 다른 모델에 밀리지 않는 후보.
  정성 통과가 하나도 없으면 `visual=NONE`으로 두고 이 목록만 검토용으로 제공한다.
- 선택 규칙과 14/20·95%는 실무용 휴리스틱이며 검증된 통계적 유의성 기준이 아니다.
  임의 가중합 점수는 사용하지 않는다. 모든 목표 판정은 반올림 전 지표를 사용한다.
- 같은 run의 인접 저장 step과 통과 여부도 기록한다. 인접 step이 멀리 떨어져 있으면
  이를 시간적 안정성의 증거로 과대 해석하지 않는다.

기본값 변경은 `--min-wins`, `--grain-ratio-max`, `--edge-retention`으로 가능하며
보고서에 실제 값이 기록된다. 결과를 보고 바꾼 기준은 탐색적 선별이라는 점을 유지한다.

## 결과 확인

1. `index.html` / `FR_blind.png`: 후보 이름을 가린 공통 대비 비교. PAN에서만 고른
   경계/평탄부 8개 ROI와 모든 FR20 장면을 제공한다. 특정 후보의 출력으로 crop을 고르지 않는다.
2. `FR_named.png`, `blind_key.json`: 확인 후 후보 이름을 공개한다.
3. `RR_all20/`: 해당 후보에 RR NPY가 있으면 같은 표시 범위로 비교한다.
   GT NPZ가 있으면 GT도 함께 표시한다. 미보관 후보는 `RR_availability.json`에 명시한다.
   RR·FR 동일 scene 번호가 같은 지리적 장면이라는 가정은 하지 않는다.
4. `ranking.csv`, `per_scene.csv`: 지표, 목표 충족, 개선 장면 수, 최악 장면, 인접 step.
5. `shortlist.json`: 세 범주의 예비 대표와 각 top-k 목록. `final_selection`은
   **PENDING_HUMAN_REVIEW**다. ringing/이중 경계/색 번짐 검토를 자동 통과로 표시하지 않는다.
6. `report.md`, `ranking.json`: 수치·정의·프로토콜·모델 및 출력 경로·소스 버전.
   원본 `FR_sharpness.json`, 기존 `ranking.csv`, Sheet의 대표 모델은 바꾸지 않는다.

사용자가 고정한 crop을 쓰려면 `--rois /path/rois.json`을 지정한다:

```json
[{"scene": 0, "x": 32, "y": 64, "size": 96, "label": "fixed ROI"}]
```

최종 선정은 **한 checkpoint의 RR/FR 전체 결과**로 한다. 장면별 best step을 혼합하지 않는다.
이 코드에서는 새 GF2 결과에 대한 실제 우열을 미리 주장하지 않는다. 실제 후보는 s3 실행으로 결정된다.
