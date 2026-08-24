# 다른 서버에 옮기기

이 저장소는 [KAIST-VICLab/PAN-Crafter](https://github.com/KAIST-VICLab/PAN-Crafter)
(ICCV 2025) 의 코드에 **재현 실험과 분석**을 얹은 것이다. 원 저작권과 `LICENSE`
(MIT, **비상업 연구·교육 목적 한정**)는 그대로 유지한다.

## 0. 한 번에 준비하기 (권장)

clone · 이미지 pull · 데이터 내려받기 · 재배치 · lpan 복구 · 검증을 한 번에 한다.

```bash
# 빈 디렉터리에서
./tools/bootstrap.sh --repo <이 저장소 URL>

# 이미 clone 했다면 저장소 안에서
./tools/bootstrap.sh
```

| 옵션 | |
|---|---|
| `--sensors wv3,qb,gf2` | 배치할 센서 (기본 `wv3`) |
| `--skip-data` / `--skip-docker` | 해당 단계 생략 |
| `--keep-archive` | 받은 zip(12 GB)을 지우지 않는다 |

**디스크는 35 GB 이상** 필요하다 — zip 12 GB + 압축해제 19 GB + 이미지 9.4 GB.
WV3 만 배치하면 데이터는 약 6.5 GB 다.

아래 1–6절은 각 단계를 손으로 할 때의 설명이다.

---

## 1. clone

```bash
git clone <이 저장소 URL> PAN-Crafter
cd PAN-Crafter
```

## 2. 환경 — 셋 중 하나

### (a) Docker — 클라우드·새 서버에 가장 빠르다

**빌드할 필요 없다.** Docker Hub 에 올려둔 환경 이미지를 받아 코드를 마운트한다.

```bash
git clone <이 저장소> PAN-Crafter && cd PAN-Crafter
docker run --gpus all -it --rm \
    -v "$PWD":/workspace \
    -v /path/to/PanCollection:/workspace/data/PanCollection \
    hojunqueen/pancrafter-env ./tools/run.sh wv3
```

| | |
|---|---|
| 이미지 | `hojunqueen/pancrafter-env:latest` (= `:torch2.4.0-cu118`), 약 9.4 GB |
| 담긴 것 | PyTorch 2.4.0 / CUDA 11.8 + 의존성 16개 + DLPan-Toolbox |
| **담기지 않은 것** | **코드·데이터·`work_dir`** — 전부 마운트한다 |

**코드를 마운트하는 구조라 코드가 바뀌어도 이미지를 다시 만들 필요가 없다.**
이미지는 의존성이 바뀔 때만 갱신한다. 저장소가 비공개이므로 연구 기록이 이미지에
섞여 들어가지 않는 이점도 있다.

**`config/*.yaml` 의 절대경로는 한 번 맞춰줘야 한다.** 컨테이너 기준으로 바꾼다.

```bash
docker run --rm -v "$PWD":/workspace hojunqueen/pancrafter-env ./tools/setup_paths.sh --apply
```

`tools/bootstrap.sh` 는 이 단계를 포함한다. 컨테이너 안에서
`python tools/verify_metrics.py` 로 지표 구현을 확인할 수 있다.

직접 빌드하려면:

```bash
docker build -f Dockerfile.env -t pancrafter-env .    # 환경만 (권장)
docker build -t pancrafter .                          # 코드까지 포함 (비공개 배포용)
```

full-res 지표가 필요 없으면 `--build-arg WITH_DLPAN=0` 으로 줄일 수 있다.

### (b) pip — 기존 PyTorch 환경 위에 얹을 때

```bash
pip install -r requirements.txt
```

실제로 import 되는 16개만 담았다. torch/torchvision 이 이미 있으면 그 두 줄은 빼도 된다.

### (c) conda — 원 개발 환경 그대로

```bash
conda env create -f requirements.yaml     # env 이름: pancrafter, 약 7.7 GB
conda activate pancrafter
```

`requirements.yaml` 은 conda env 전체 export(155행)라 ffmpeg·intel-openmp 등 무관한 것도
섞여 있다. 재현성이 중요한 게 아니면 (a)나 (b)가 빠르다.

> `requirements.yaml` 은 Anaconda 기본 채널(`defaults`)의 빌드를 고정한다. 최근 conda 는
> 그 채널의 이용약관을 수락하지 않으면 `CondaToSNonInteractiveError` 로 멈춘다.
> **수락 여부는 조직의 라이선스 판단이 필요한 사항이다** (200인 이상 조직은 유료).
>
> ```bash
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
> ```

## 3. 경로 맞추기

`config/*.yaml` 에는 원 개발 환경의 절대경로가 박혀 있다. 한 번 실행하면 현재 위치로 바뀐다.

```bash
./tools/setup_paths.sh            # 무엇이 바뀌는지 확인만
./tools/setup_paths.sh --apply    # 반영
```

> **서버가 둘 이상이면 브랜치를 나눈다.** 치환된 경로를 `main` 에 커밋하면 다른 서버가 깨진다.
> 또 `results_log/README.md` 는 양쪽이 **맨 위에** 행을 추가하는 구조라 커밋마다 충돌한다.
>
> | 브랜치 | 담는 것 |
> |---|---|
> | `main` | 서버 중립. 코드·문서·`results_log` |
> | `server/<호스트명>` | 그 서버의 작업 전부 |
>
> `server/*` 브랜치는 **통째로 `main` 에 병합하지 않는다.** 공유 가능한 커밋만 cherry-pick 한다.
> 그래서 커밋을 섞지 말고 `[공유]` / `[서버전용]` 으로 나눠 쌓는다.

## 4. 데이터 (저장소에 포함되지 않음)

[PanCollection](https://github.com/liangjiandeng/PanCollection) 에서 받아 아래 구조로 둔다.
심볼릭 링크도 된다.

```
data/PanCollection/WV3/
├── train_wv3.h5
├── valid_wv3.h5
├── reduced_examples_h5/test_wv3_multiExm1.h5
├── full_examples_h5/test_wv3_OrigScale_multiExm1.h5
└── full_examples_h5_repaired/          # tools/repair_lpan.py 로 생성
```

`pan_h5/pan_h5.zip` 은 저자 배포본이며 저장소에 포함되어 있다. 압축을 풀어 `data/` 로 옮긴다.

**PanCollection 배포본의 파일명은 위 구조와 다르다** (`train_wv3_9714.h5`, `valid_wv3_9714.h5`,
`test_data/` 한 폴더에 전 센서 혼재). feeder 가 경로 문자열에서 split·센서를 추론하므로
([KNOWN_ISSUES.md B-2/B-3](KNOWN_ISSUES.md)) 이름을 반드시 맞춰야 한다. 자동화해 두었다.

```bash
unzip -o pan_h5/pan_h5.zip -d pan_h5/extracted
unzip -o <PanCollection zip> -d data/_extracted
./tools/setup_data_layout.sh          # 링크 배치 + *_pan.h5 복사 (멱등)
```

실제 h5 20 GB 는 `<song>/datasets/PanCollection/` 한 곳에만 두고, PAN-Crafter 와 CANConv 가
각자 기대하는 이름으로 심볼릭 링크만 건다.

> **주의.** 배포 `pan_h5.zip` 의 WV3·QB **full-resolution `lpan` 이 다른 장면이다**
> (`KNOWN_ISSUES.md` F-1). 그대로 쓰면 full-res 평가가 무효다. 반드시 복구본을 만든다.
>
> ```bash
> python tools/repair_lpan.py
> ```

## 5. 평가 지표

지표 구현은 **이 저장소 안에 있다** (`tools/metrics/`). 별도 저장소를 clone 할 필요가 없다.

| | |
|---|---|
| `tools/metrics/eval_rr.py` | reduced: SAM / ERGAS / Q2n |
| `tools/metrics/eval_fr.py` | full-res: D_λ / D_s / HQNR |
| `tools/metrics/q2n.py` | Q2n 코어 |
| PSNR / SSIM / SCC | `tools/eval_dlpan.py` 안에 자체 구현 |

### full-resolution 만 외부 의존이 있다

`eval_fr.py` 는 DLPan-Toolbox 의 `wald_utilities.py`(MTF / interp23tap)를 런타임에
import 한다. **DLPan-Toolbox 는 GPL-3.0 이라 이 저장소에 넣지 않았다** — MIT 인 이 저장소에
포함시키면 결합 저작물이 GPL 로 끌려간다. 별도로 clone 해서 경로만 알려준다.

```bash
git clone https://github.com/liangjiandeng/DLPan-Toolbox.git
export PANCRAFTER_DLPAN=$PWD/DLPan-Toolbox
```

reduced-resolution 지표(SAM/ERGAS/Q2n/PSNR/SSIM/SCC)는 **이것 없이도 전부 동작한다.**
필요한 건 파이썬 파일 하나뿐이고 MATLAB 은 쓰지 않는다.

### 이식 확인

지표는 순수 numpy/scipy 연산이라 **입력이 같으면 서버가 달라도 값이 같아야 한다.**
고정 시드 난수로 확인한다 (저장소에 데이터가 필요 없고 영상 라이선스 문제도 없다).

```bash
python tools/verify_metrics.py
```

여섯 지표가 상대오차 0 으로 일치하면 이식이 정상이다. `PANCRAFTER_DLPAN` 이 없으면
reduced 세 개만 검사한다.

`tools/env.sh` 에 이 export 가 들어 있고 `~/.bashrc` 에 등록해 두면 매번 쓰지 않아도 된다.
`PANCRAFTER_CANCONV` 는 지표에는 더 이상 쓰이지 않고, `tools/setup_wv2.py` 가 WV2 테스트셋을
찾는 데만 쓴다.

## 6. 동작 확인

```bash
python tools/verify_metrics.py                   # 지표 구현 이식 확인
python tools/check_data.py                       # 데이터 경로·shape 점검
python tools/repair_lpan.py --sensor wv3         # F-1 복구 (qb 도 함께). 상관 0.01 -> 1.000 이 나와야 한다
python tools/setup_wv2.py                        # WV2 zero-shot 배치 (선택)
./tools/run.sh _smoke                            # 250 iter + 평가 1회 + mat 내보내기까지 전 경로
```

환경이 제대로 섰는지는 **수치 두 개**로 확인한다. 둘 다 서버와 무관하게 같아야 한다.

| 확인 | 기대값 |
|---|---|
| `pancrafter_wv3.yaml` 파라미터 수 | **9,968,808** |
| `repair_lpan.py --sensor wv3` 의 배포본 상관 | **+0.011** (손상) → 재생성 **+1.000** |

## 7. 학습

```bash
./tools/run.sh wv3                                       # 배포본 그대로
./tools/run.sh wv3_fixed                                 # KNOWN_ISSUES A-1/A-2 적용
setsid nohup ./tools/run.sh wv3 > /dev/null 2>&1 &       # SSH 끊겨도 유지
```

`work_dir/<실험>/meta/` 에 config·git commit·환경이 함께 남는다.

---

## 저장소에 포함되지 않는 것

| | 이유 |
|---|---|
| `data/` | 데이터셋. PanCollection 에서 별도로 받는다 |
| `work_dir/` | 학습 산출물 40 GB. 체크포인트·`.mat` 포함 |
| `*.mat` | 평가 산출물. 필요하면 `tools/export_mat.py` 로 재생성 |

## 결과를 구글시트로 올리기

```bash
echo s2 > gspread/server.txt          # 이 서버의 식별자. 한 번만 해두면 된다
python gspread/gspread_upload.py paper_ln --profile
```

`gspread/account.json`(서비스 계정 키)이 필요하다. **저장소에 없으므로 별도로 복사**한다.

**서버 식별자는 필수다.** 시트가 `<데이터셋>-<서버>` 로 갈리므로(`WV3-s1`, `WV3-s2`)
서버를 모르면 어느 시트에 쓸지 정할 수 없다. 지정하지 않으면 업로드를 거부한다.
같은 config 를 두 서버가 돌려도 시트가 달라 충돌하지 않는다.

`--server s2` 나 `PANCRAFTER_SERVER=s2` 로도 줄 수 있다.

---

## 먼저 읽을 것

| 문서 | |
|---|---|
| [INTRO.md](INTRO.md) | 논문 요지와 코드 구조 |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 논문 불일치·잠재 버그·데이터 결함과 적용 현황 |
| [REPRODUCTION.md](REPRODUCTION.md) | 재현 절차 |
| [results_log/README.md](results_log/README.md) | 실험 결과 문서 색인 (최신순) |
