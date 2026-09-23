# G23 sensitivity — s4 / s5

새 캠페인은 **WV3 / G23 / P0 / W104D121 / 고정 T0(step24240)** 전용이다.
기존 QRECON24·PCREPRO·ABLR2·GF2 학습 코드는 수정하지 않는다.
정의의 기준은 `research_log/PANDA_G23_SENS_S45_UNLIMITED_2026-09-23`의
해시 고정 번들과 그 `casegen.py`다.

## 실행

커밋을 push/pull한 뒤 **해당 서버에서 한 번만** 실행한다.
push/pull만으로 학습이 시작되지는 않는다. 로컬에 이미 설치된 Docker 이미지를
content ID로 고정하며 자동 pull·기존 컨테이너 삭제·강제 종료는 하지 않는다.

```bash
# s4에서
bash tools/g23sens_docker_start.sh --server s4

# s5에서
bash tools/g23sens_docker_start.sh --server s5
```

기본 이미지: `hojunqueen/pancrafter-env:torch2.4.0-cu118`.
필요하면 `--image <installed-image>` / `--gpu <index-or-UUID>`를 지정한다.
호스트 Python은 기존 pancrafter 환경을 사용한다. 셸 래퍼는
`PANCRAFTER_PYTHON=/path/to/python`도 지원한다.
Docker 미사용 시 `bash tools/g23sens_start.sh --server s4`로 같은 고정 release를 실행한다.

명령은 전용 release → 기존 실행의 current-run 인계 → 자산 검증 → 실제 CUDA
smoke/분리 gradient·resume/전체 native RR·FR 평가 → 첫 BASE 순서로 진행한다.
필요한 검증은 자동 연결되어 있으며 별도 승인 파일이나 시간 lease는 없다.
현재 지원하는 자동 인계는 PCREPRO의 `STOP_AFTER_CURRENT_RUN`이다.
알 수 없는 watchdog/구형 러너는 강제 종료하지 않고 안전 중지 사유를 출력한다.
다른 GPU를 사용하는 s1–s3 또는 다른 서버 lane은 변경하지 않는다.

첫 실행에 필요한 자산은 기존 T0 패키지, cue JSON/NPZ, 원본 WV3 8개 H5/LP,
고정 calibration JSON, 외부 DLPan 평가 코드다. Teacher 재학습이나 데이터/LP
재생성 fallback은 없다. 다른 서버에서도 원본 SHA가 같아야 한다.
해당 자산은 Git에 대용량으로 추가하지 않았다.

## 운영

```bash
python tools/g23sens_runner.py preview --server s4 --cycle 0
python tools/g23sens_runner.py status --server s4
python tools/g23sens_runner.py stop --server s4
# 긴급 안전 checkpoint 중단
python tools/g23sens_runner.py stop --server s4 --safe-now
# 네트워크 실패 outbox만 재업로드 (학습/평가 재실행 없음)
python tools/g23sens_runner.py upload --server s4
```

상태·로그·설정·초기값·전체 stream·결과는 `work_dir/g23sens/<server>/`에 저장된다.
기존 `STOP_AFTER_RUN`/`STOP_NOW_SAFE`는 start가 임의로 지우지 않는다.
재개하려면 해당 서버의 의도한 중단 marker만 확인 후 제거하고 start한다.
run은 `runs/<run_id>/attemptNNN/`에 보존되며 잘못된 source/binding으로 덮어쓰지 않는다.
BASE 실패, 재시도 초과, 디스크 부족은 중지한다. variant의 수치 발산은 실패로
기록하고 다음 조건으로 진행한다. 계수 변경·시드 재추첨·성능 기준 조기 종료는 없다.
서버별 BASE+6조건 뒤 새 시드로 무기한 반복하며 서버 간 완료 대기는 없다.

## 계산·평가·보고 계약

- 기존 G23 U 생성 순서와 T0 A 독립 복사. 같은 cycle은 실제 initial U/A와
  **50K 전체 sample/rotation/flip stream**을 공유한다.
- 기존 feeder의 양방향 flip=True는 확률 0.5가 아니라 항상 flip이다. 이 동작을
  변경하지 않고 원본 메서드로 실행한다. 원본 4-worker sampler와 수치 대조한다.
- `U ← H+K+λE·wE`, `A ← wH`의 두 `autograd.grad` 경로를 분리한다.
  qraw는 고정, q_ref만 배율 적용, tau는 criterion에서 정확히 한 번 배율 적용한다.
- 이번 명시적 민감도 계획의 primary는 **EXACT_50000**, secondary는 고정
  GRID1010_50K의 **val-ERGAS 최소 / 동률 시 더 이른 step**이다.
  다른 캠페인의 HQNR 선택 규칙을 변경한 것이 아니다. FR test HQNR로 선택하지 않는다.
- 학습 validation은 native64 RR ERGAS/SCC이며 HQNR은 `N/A(val-only)`라고 명시한다.
  최종 평가 로그에는 실제 HQNR·SCC·ERGAS를 출력한다. val과 최종 RR를 혼합하지 않는다.
- RR 20장·FR 20장 전부. 기존 공식 RR crop `20:-21`, Q8, native original-PAN HQNR를
  재사용하며 FR shift/masking을 추가하지 않는다. RMSE·CC·JQM도 포함한다.
- JQM은 기존 **SRF-substitute** 구현으로 명시한다. SIPSA 동일 구현으로 주장하지 않는다.
  파일 고정과 별개로 논문 scene correspondence는 **PAPERSET_IDENTITY_UNVERIFIED**다.
- 전용 탭은 `SENS-G23-WV3-s4`, `SENS-G23-WV3-s5`뿐이다. 원값은 full precision,
  metric 표시 형식만 소수점 4자리다. 실제 학습일·학습시간·seed·case·선택·SHA·attempt와
  같은 서버/cycle BASE 대비 Δ를 분리 기록한다. raw-value readback 후에만 업로드 완료다.
- 정확히 같은 checkpoint인 두 selection은 alias로 표시한다. 다른 서버 BASE나
  재시도를 독립 seed로 합산하지 않는다. 결과는 고정 T0 조건부 Student-seed 민감도다.

## 검증

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m unittest discover -s g23sens -t . -p 'test_*.py'
python -m unittest discover \
  -s research_log/PANDA_G23_SENS_S45_UNLIMITED_2026-09-23/tests
```

`-t .`를 유지한다. 이를 생략하면 테스트 discovery가 `g23sens/model.py`를
저장소의 최상위 `model` 패키지보다 먼저 찾을 수 있다.
CPU mocked-device persistence 검증은 실제 GPU smoke로 보고하지 않는다.
자산 검증 근거: `research_log/2026-09-23_g23sens_asset_audit.json`.
