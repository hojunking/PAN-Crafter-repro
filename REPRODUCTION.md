# PAN-Crafter 재현 실행 메모

논문 요지와 코드 구조는 [INTRO.md](INTRO.md), 코드 문제는 [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

옆 저장소 `../CANConv` 에서 이미 받아 둔 PanCollection 데이터를 그대로 재사용하도록
경로를 맞춰 둔 상태다. 데이터 복사는 하지 않고 심볼릭 링크로 연결한다.

## 1. 데이터 배치

PAN-Crafter 는 PanCollection 원본 h5 외에 저자들이 별도 배포한 `*_pan.h5`
(= `lpan`, PAN 을 4배 다운샘플해 MS 해상도로 맞춘 것)를 **같은 디렉터리에서**
`dataroot.replace(".h5", "_pan.h5")` 로 찾는다 ([feeders/feeder.py:43](feeders/feeder.py#L43)).
그래서 원본 h5 는 링크, `_pan.h5` 는 실제 파일로 나란히 배치했다.

```
data/PanCollection/
├── WV3/
│   ├── train_wv3.h5              -> ../CANConv/data/datasets/wv3/train_wv3.h5   (symlink)
│   ├── train_wv3_pan.h5                                                          (pan_h5.zip)
│   ├── valid_wv3.h5              -> ...                                          (symlink)
│   ├── valid_wv3_pan.h5                                                          (pan_h5.zip)
│   ├── reduced_examples_h5/
│   │   ├── test_wv3_multiExm1.h5 -> ...                                          (symlink)
│   │   └── test_wv3_multiExm1_pan.h5
│   └── full_examples_h5/
│       ├── test_wv3_OrigScale_multiExm1.h5 -> ...                                (symlink)
│       └── test_wv3_OrigScale_multiExm1_pan.h5
├── QB/    (동일 구조)
└── GF2/   (동일 구조)
```

`reduced_examples_h5` / `full_examples_h5` 디렉터리 이름은 장식이 아니다.
feeder 가 경로 문자열에서 `train`/`valid`/`reduced`/`full` 토큰을 찾아 split 을
결정하고, `wv3`/`qb`/`gf2` 토큰으로 `max_pixel`(2047 / 2047 / 1023)을 추론한다
([feeders/feeder.py:24-30, 53-62](feeders/feeder.py#L24-L62)). 경로를 바꿀 때 이 토큰이
사라지면 조용히 잘못된 정규화로 학습된다.

재배치가 필요하면:

```bash
python tools/check_data.py          # 3개 config 전부 검증
```

## 2. 환경

```bash
conda env create -f requirements.yaml   # env 이름: pancrafter (python 3.9.19 / torch 2.4.0+cu118)
conda activate pancrafter
```

## 3. 학습 — 축 두 개, 실행 네 벌

두 가지가 독립적으로 걸려 있다.

| 축 | 값 | 무엇을 바꾸나 |
|---|---|---|
| `model_args.fix_key_alias` / `fix_local_attn` | False / True | 모델 동작 ([A-1](KNOWN_ISSUES.md), [A-2](KNOWN_ISSUES.md)) |
| `select_on` | `test` / `val` | best 체크포인트 선택 기준 ([E-1](KNOWN_ISSUES.md)) |

```bash
./tools/run.sh wv3                    # baseline, select_on=test  (배포본 동작 그대로)
./tools/run.sh wv3_fixed              # fixed,    select_on=test
./tools/run.sh wv3_baseline_valsel    # baseline, select_on=val   (권장)
./tools/run.sh wv3_fixed_valsel       # fixed,    select_on=val

# 죽었으면 이어서 (C-1 적용으로 optimizer/scheduler state 가 복원된다)
./tools/run.sh wv3 --resume work_dir/wv3_baseline/checkpoint-20000
```

**새 연구의 출발점으로는 `*_valsel` 을 쓴다.** `select_on=test` 쪽은 배포본 동작을 보존한
재현 기준선이다. 실행당 약 5시간(valsel 은 검증 평가 때문에 +16분).

`tools/run.sh` 는 실행 조건을 `work_dir/<실험>/meta/` 에 남긴다
(config 사본, git commit·status·diff, conda 환경, nvidia-smi, 시작/종료 시각).

| 경로 | 내용 |
|---|---|
| `meta/` | 실행 조건 스냅샷 |
| `train_log.txt` / `test_log.txt` | iteration 별 loss / eval 별 지표 |
| `metrics.csv` | eval 별 지표 (val 열 포함). 곡선·사후 재선택의 원자료 |
| `best_val/` 또는 `best_reduced/`,`best_full/` | 선택된 체크포인트 |
| `epoch-*/`, `checkpoint-*/`, `lastest/` | 주기적 체크포인트 |
| `results/reduced_<tag>.mat`, `results/full_<tag>.mat` | 평가 입력 |

## 3-1. 평가 — MATLAB 없이 논문 프로토콜로 잰다

이 머신에 MATLAB 이 없다. 옆 저장소의 DLPan-Toolbox 파이썬 포팅을 재사용한다
(CANNet 논문 대비 reduced 0.7% / full HQNR 0.2% 이내 검증됨).
같은 평가기를 쓰므로 `../CANConv` 결과와 직접 비교된다.

```bash
# reduced (정답 있음): PSNR / SAM / ERGAS / SCC / Q2n
python tools/eval_dlpan.py work_dir/wv3_baseline/results/reduced_best_reduced.mat \
       --preset wv3 --baseline

# full-resolution (정답 없음): D_lambda / D_s / HQNR
python tools/eval_dlpan_fr.py --preset wv3 --baseline --indices 12-19 \
       --mat "baseline=work_dir/wv3_baseline/results/full_frrepair.mat"

# 임의 checkpoint 를 .mat 으로 내보내기
python tools/export_mat.py --config config/pancrafter_wv3.yaml \
       --ckpt work_dir/wv3_baseline/epoch-100 --tag ep100

# 학습 곡선 · 보고서 그림
python tools/make_report_figures.py
```

`--indices 12-19` 는 논문 수치와 대조할 때 쓴다. 이 PanCollection 배포본은 full-res
테스트 20장 중 0-11 번이 크게 어려워, 논문 표는 12-19 와 맞는다(`../CANConv` RUNBOOK 8절).

## 3-2. full-resolution 을 쓰기 전에 — 데이터 복구가 필요하다

배포 `pan_h5.zip` 의 WV3·QB full-res `lpan` 이 손상돼 있다([F-1](KNOWN_ISSUES.md)).
그대로 쓰면 HQNR 0.38 이 나온다(정상은 0.84).

```bash
python tools/repair_lpan.py --sensor wv3     # full_examples_h5_repaired/ 생성
python tools/setup_wv2.py                    # WV2 zero-shot 용 데이터 배치
```

원본은 건드리지 않는다. 복구본을 보는 평가 전용 config 가 `config/eval_wv3_frrepair_*.yaml` 이다.

실험 기록은 [results_log/](results_log/) 에 `CONVENTION.md` 규약대로 남긴다
(`../CANConv/results_log` 와 동일 형식).

## 4. 재현 설정 (논문 하이퍼파라미터 그대로)

- 50,000 iteration, AdamW lr 1e-4 / wd 0.01, cosine + warmup 100, seed 2025
- `batch_size: 48` 이지만 학습 스텝마다 배치를 2배로 복제해
  (PAN 재구성 branch + MS 재구성 branch) 실효 배치는 96 이다
  ([train.py:137-148](train.py#L137-L148)).
- `eval_epoch: 5` — 5 epoch 마다 reduced/full 평가. 매 평가 결과가 `metrics.csv` 에 한 행씩 쌓인다
- 총 epoch: WV3 248 / QB 141 / GF2 122

경로와 아래 5절에 적은 항목 외에는 저자 배포본에서 바꾸지 않았다.

## 5. 저장소에 적용한 수정

[KNOWN_ISSUES.md](KNOWN_ISSUES.md) 에서 확인한 것 중 **운용(C)·지표(D)·평가 방법론(E)·데이터(F)** 를 적용했다.
논문 불일치(A-1, A-2)와 선택 기준(E-1)은 고치지 않고 **config 스위치**로 만들어 양쪽을 모두 보존했다.
데이터 확장 시 문제가 되는 잠재 버그(B)는 문서화만 하고 손대지 않았다.

| 적용 | 무엇이 달라지는가 |
|---|---|
| C-1 | optimizer·scheduler 를 accelerator 에 등록 → checkpoint 가 완전해지고 `--resume` 가능 |
| C-2 | `save_epoch` 가드 추가, config 값 500 → 25 (총 epoch 안에서 발동하도록) |
| C-3 | 마지막 학습 로그의 `nan` 제거 |
| C-4 | `--res` 에 `str2bool` 연결 |
| D-1 | `metrics.csv` 추가, `.mat` 파일명 태그화(덮어쓰기 제거), 학습 종료 후 일괄 내보내기 |
| D-2 | ERGAS 를 참조(GT) 밴드 평균으로 정규화 (표준 정의) |
| D-3~D-5 | 로그 라벨 정정 — `Q4(first4)`, `ERGAS(vs PAN)`, `SCC(vs PAN)`, QNR≠HQNR 주석 |
| D-6 | `scipy.ndimage` 로 import 정정, 미사용 import 제거 |
| E-1 | `select_on: test\|val` 스위치. `val` 이면 검증셋으로 best 를 고르고 테스트는 기록만 |
| F-1 | `tools/repair_lpan.py` — 손상된 full-res `lpan` 재생성 (원본 보존) |

**D-2 때문에 ERGAS 값의 스케일이 바뀐다.** 수정 전 로그와 직접 비교하지 말 것
(같은 스모크 조건에서 23.00 → 7.21).

C 항목은 학습 수학을 건드리지 않는다. D 항목은 파이썬 지표 계산만 바꾸며,
논문 비교용 최종 수치는 어차피 DLPan-Toolbox 로 낸다.

## 6. VRAM

24GB 카드에서 OOM 나지 않는다. 논문의 RTX 3090 서술과 일관된다.
Table 1 의 `Memory 1.711 GB` 는 256×256 reduced-resolution **추론** 수치이지 학습 수치가 아니다.

| 단계 | peak allocated |
|---|---|
| 학습 B=96(48×2), PAN 64² | 16.51 GB |
| 추론 reduced, B=1, PAN 256² | 0.86 GB |
| 추론 full, B=1, PAN 512² | 3.29 GB |

`nvidia-smi` 기준 19.8 GB 는 여기에 allocator reserve 와 CUDA context 가 더해진 값이다.
메모리를 먹는 곳은 CM3A 의 k²=9 확장이며 학습 배치에 정확히 비례한다.
자세한 분해와 감축 선택지는 [KNOWN_ISSUES.md 의 G절](KNOWN_ISSUES.md) 참고.
