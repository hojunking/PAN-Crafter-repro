# GF2 정성 후보 선별 — s3 인계

사용자 요청: s3에 저장된 추가 GF2 실험의 모든 step 정성 출력/NPY에서 후보를 선별하고,
해당 서버에서 실행할 수 있도록 코드를 커밋한다.

## 실행

사용자가 push한 뒤 s3에서 전체 커밋을 pull한다. 기존 NumPy/SciPy/Pillow 환경을 사용한다.

```bash
git pull --ff-only
python -m panda_gf2_sharpness.select --verify
python -m panda_gf2_sharpness.select
```

기본 입력은 `work_dir/panda_gf2_sharpness/exports/` 아래 모든 표준 export다.
추가 경로가 있다면 `--exports <기존 exports> <추가 exports>`로 함께 전달한다.
기본 출력은 `work_dir/panda_gf2_sharpness/selections/<실행시각>/`이다.
별도 실행 예제와 CPU Docker 명령은 [SELECTION.md](../panda_gf2_sharpness/SELECTION.md)에 있다.

**재학습·추론·GPU·checkpoint 이동·Sheet 변경은 없다.** 이 작업을 위해 기존
`panda_gf2_sharpness.enqueue` / `run`을 실행하지 않는다. 원본 export나 기존 순위표를
덮어쓰지 않으며, 이미 존재하는 선택 결과 폴더에는 쓰지 않는다.

## 제공하는 판정

- 기존 렌더링 지표와 동일한 `legacy_*`를 NPY에서 다시 계산한다.
- `shared_*`: 같은 장면의 stored LMS bounds를 모든 후보에 공통 적용한 float 진단.
- `raw_*`: clipping하지 않은 원본 RGB 평균/1023의 진단.
- 선명도·평탄부 변동·20장 개선 일관성을 함께 보고 정성 후보를 구한다.
- HQNR≥.9645, ERGAS≤.522, PSNR≥45.08의 동일 checkpoint 목표 판정을 별도 표시한다.
- 정성 우선 / 정량 충족 / 절충 후보를 따로 제공한다. 정성 기준 미달이면 `NONE`이다.
- 같은 run의 인접 저장 step, 최악 장면과 개선 장면 수를 기록한다.
- PAN만 보고 정한 고정 ROI와 FR20 전체 비교판을 만든다. RR NPY가 있는 후보는
  RR20도 비교하고, GT NPZ가 있으면 함께 표시한다. 누락은 명시한다.
- `index.html` → `FR_blind.png`로 이름을 가린 비교부터 진행한다. `blind_key.json`과
  `FR_named.png`로 실제 run/step을 확인한다. `shortlist.json`은 예비 후보이며
  `final_selection=PENDING_HUMAN_REVIEW`를 유지한다.

정성 기준은 C60LR150 R1@47K, 정량 기준은 C60LR200 R2@47K다. 정성 기준 export가
없으면 고정 정량 기준으로 paired 비교하고 그 사실을 명시한다. 둘 다 없으면
임의 후보를 기준으로 고르지 않고 `--baseline <명시한 export>`를 요구한다.

## 검증과 범위

- 선택기 7개 테스트 + 기존 사용자 지표 일치 테스트 1개: **8/8 통과**.
- 원본 NPY 변조/장면 순서 오류/공통 대비 불일치 검출, 반올림 전 목표 판정,
  잡음이 높은 선명도 후보 제외, 통과 후보 부재 처리, 기본 기준 누락 시 고정 fallback을 확인했다.
- 합성 FR20/RR20 export 두 개로 전체 실행과 비교판 생성을 확인했다.
  처리 전후 모든 입력 파일 SHA가 동일했다. 이 합성 수치는 과학적 실험 결과가 아니다.
- 선택기 import에서 Torch가 로드되지 않음을 확인했다.
- Git index의 9개 의존 소스 SHA가 manifest와 일치하며, 이 파일들만 꺼낸 별도 경로에서
  source 검증과 합성 export 전체 선별 실행도 통과했다. 미추적 로컬 코드에 의존하지 않는다.
- 기존 학습/replay release 파일과 그 소스 pin은 변경하지 않았다.
  선택기는 `config/panda_gf2_sharpness/selection_release.json`의 별도 source manifest를 검증한다.
- 실제 s3 GF2 출력은 s1에서 접근하지 않았다. 새 후보의 실제 성능과 순위는 s3 실행 결과로 판단한다.

후보 선정은 테스트 출력 기반의 탐색적 분석이다. 한 checkpoint의 전체 RR/FR 결과를
사용하며 장면별 best step을 혼합하지 않는다. PAN은 FR GT가 아니므로 ringing·이중 경계·
색 번짐은 공통 대비 비교판에서 별도로 확인한다.
