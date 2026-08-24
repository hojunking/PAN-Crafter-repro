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

## 5. 평가 도구용 형제 저장소

`tools/eval_dlpan*.py`, `tools/collect_*.py` 는 외부 저장소 두 개를 참조한다.
**환경변수로 위치를 알려준다** (없으면 원 개발 환경 경로를 기본값으로 쓴다).

```bash
export PANCRAFTER_CANCONV=/path/to/CANConv          # 지표 구현(eval_rr.py) 재사용
export PANCRAFTER_DLPAN=/path/to/DLPan-Toolbox      # MTF/Q2n 공식 구현
```

학습만 할 것이라면 없어도 된다. **논문과 비교 가능한 수치를 내려면 둘 다 필요하다.**

## 6. 동작 확인

```bash
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
