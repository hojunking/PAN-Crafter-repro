# 다른 서버에 옮기기

이 저장소는 [KAIST-VICLab/PAN-Crafter](https://github.com/KAIST-VICLab/PAN-Crafter)
(ICCV 2025) 의 코드에 **재현 실험과 분석**을 얹은 것이다. 원 저작권과 `LICENSE`
(MIT, **비상업 연구·교육 목적 한정**)는 그대로 유지한다.

## 1. clone

```bash
git clone <이 저장소 URL> PAN-Crafter
cd PAN-Crafter
```

## 2. 환경

```bash
conda env create -f requirements.yaml     # env 이름: pancrafter
conda activate pancrafter
```

## 3. 경로 맞추기

`config/*.yaml` 에는 원 개발 환경의 절대경로가 박혀 있다. 한 번 실행하면 현재 위치로 바뀐다.

```bash
./tools/setup_paths.sh            # 무엇이 바뀌는지 확인만
./tools/setup_paths.sh --apply    # 반영
```

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

## 6. 동작 확인

```bash
python tools/verify_metrics.py                   # 지표 구현 이식 확인
python tools/check_data.py                       # 데이터 경로·shape 점검
./tools/run.sh wv3 --num-iter 100                # 짧은 스모크 런
```

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

## 먼저 읽을 것

| 문서 | |
|---|---|
| [INTRO.md](INTRO.md) | 논문 요지와 코드 구조 |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 논문 불일치·잠재 버그·데이터 결함과 적용 현황 |
| [REPRODUCTION.md](REPRODUCTION.md) | 재현 절차 |
| [results_log/README.md](results_log/README.md) | 실험 결과 문서 색인 (최신순) |
