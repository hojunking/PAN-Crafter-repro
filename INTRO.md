# PAN-Crafter — 논문 요지와 코드 구조

처음 이 저장소를 보는 사람이 30분 안에 전체 그림을 잡는 것이 목적이다.
실행 절차는 [REPRODUCTION.md](REPRODUCTION.md), 코드 문제는 [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

---

## 1. 논문 요지

> Do et al., *PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening*, ICCV 2025
> ([arXiv 2505.23367](https://arxiv.org/abs/2505.23367), 로컬 사본 `2505.23367v2.pdf`)

### 무엇을 푸는가

PAN-sharpening 은 **고해상도 흑백 PAN 영상**과 **저해상도 컬러 MS 영상**을 합쳐
고해상도 컬러(HRMS) 영상을 만드는 작업이다. 위성이 두 센서를 따로 싣기 때문에
PAN 은 공간해상도가 4배 높고, MS 는 분광 정보가 풍부하다.

문제는 **두 영상이 픽셀 단위로 정확히 겹치지 않는다**는 것이다. 센서 위치, 촬영 시각,
해상도 차이 때문에 국소적으로 어긋난다. 기존 방법은 완벽 정합을 가정하고 픽셀별 L1/L2
손실을 쓰므로, 어긋난 구간에서 색 번짐·이중 윤곽·블러가 생긴다.

### 두 가지 아이디어

**MARs (Modality-Adaptive Reconstruction)** — 하나의 네트워크가 mode 스위치에 따라
HRMS 를 만들기도 하고 PAN 을 만들기도 한다. PAN 재구성은 정답이 필요 없는 보조
자기지도 과제이고, PAN 의 고주파 성분을 네트워크에 주입하는 역할을 한다.
학습 시 배치를 2배로 복제해 절반은 PAN mode, 절반은 MS mode 로 동시에 학습한다.
추론에서는 MS mode 로 고정한다.

**CM3A (Cross-Modality Alignment-Aware Attention)** — 정합을 attention 으로 맞춘다.
MS mode 에서는 MS 텍스처를 PAN 구조에 맞추고, PAN mode 에서는 그 반대로 맞춘다.
전역 attention 대신 k×k 국소 윈도(k=3)만 본다. 데이터가 이미 대략 정합되어 있어
큰 변위를 찾을 필요가 없기 때문이다. 위치 임베딩을 쓰지 않고, 상대 모달리티의 특징을
Q/K 에 직접 넣어 국소 어긋남에 적응한다.

### 결과 (논문 기준)

WV3 / QB / GF2 에서 SOTA, 학습에 쓰지 않은 WV2 에 zero-shot 으로도 최고 성능.
CANConv 대비 추론 **50.11× 빠르고** 메모리 0.63×(1.711 GB, 256×256 추론 기준).
확산모델 기반(PanDiff, TMDiff) 대비 300~1100× 빠르다.

| | 파라미터 | 학습 | 실효 배치 | 최적화 |
|---|---|---|---|---|
| | 9.97 M | 50,000 iter | 96 (48×2 복제) | AdamW 1e-4, cosine, warmup 100 |

---

## 2. 데이터가 흐르는 모양

```
입력          I_pan   [4H, 4W, 1]      고해상도 PAN
              I_ms    [ H,  W, C]      저해상도 MS      (C=8 WV3 / 4 QB·GF2)
              I_lpan  [ H,  W, 1]      PAN 을 4배 축소한 것  ← 저자 배포 *_pan.h5

네트워크      P(pan, lpan, ms, mode) -> 잔차 [4H, 4W, C]

출력          MS  mode:  HRMS = 잔차 + bicubic(ms, x4)                 ... Eq (1)
              PAN mode:  PAN' = 잔차 + bicubic(lpan, x4).repeat(C)     ... Eq (3)

손실          L = |HRMS - GT|_1  +  λ·|PAN' - I_pan.repeat(C)|_1       ... Eq (4),  λ=1
                 └ 배치 뒤 절반 ┘     └────── 배치 앞 절반 ──────┘
```

`lpan` 이 별도 파일로 배포되는 이유가 여기 있다. PAN mode 의 잔차 기준선이자
CM3A 의 PAN 쪽 key/value 재료라서, MS 와 같은 해상도의 PAN 이 필요하다.

---

## 3. 핵심 코드

```
main.py                진입점. config+argparse 병합 → 학습 루프 → best 선택 → .mat 내보내기
train.py               Trainer. 학습 스텝, reduced/full 평가, .mat 저장
model/pancrafter.py    ★ 모델 전부. PANCrafter / AttnBlock / CMAAA / ResBlock
feeders/feeder.py      PanFeeder. h5 로드 + 증강 + [-1,1] 정규화
utils.py               지표(ERGAS·SAM·Q4·D_s·QNR 등)와 로깅
```

나머지(`config/`, `tools/`, `results_log/`, `data/`)는 실행·분석용이다.
[4절](#4-저장소-구조)에 정리했다.

### 3.1 `model/pancrafter.py` — 여기만 읽으면 모델은 다 본 것이다

4단계 U-Net 이다. 각 스케일에서 `ResBlock` 이 특징을 다듬고, 중·저해상도 단계에만
`AttnBlock`(=CM3A) 이 붙어 모달리티 정합을 맞춘다. 고해상도 단계는 연산량 때문에 뺐다.

```
PANCrafter.forward(pan, lpan, ms, s)          s: 0=PAN mode, 1=MS mode
│
├ c = s_token[s]                              mode 임베딩. 모든 블록의 조건 입력
├ input  Conv3x3( [pan | ↑4 ms | ↑4 lpan | pan−↑4 lpan] )     4H×4W, C=128
│
├ encoder1 ─ ResBlock×2                       4H×4W   ← res1 (skip)
├ down1
├ encoder2 ─ ResBlock×2 → cond2_e (AttnBlock)  2H×2W  ← res2
├ down2
├ encoder3 ─ ResBlock×2 → cond3_e (AttnBlock)   H× W  ← res3
├ down3
├ middle   ─ ResBlock×2 → cond4  (AttnBlock)   H/2
├ up3  → cat(res3) → decoder3 → cond3_d (AttnBlock)
├ up2  → cat(res2) → decoder2 → cond2_d (AttnBlock)
├ up1  → cat(res1) → decoder1
└ output GroupNorm → SiLU → Conv3x3(zero-init)   →  잔차 [4H,4W,C]
```

출력 conv 가 0 으로 초기화되어 있어, **학습 시작 시점의 예측은 정확히 bicubic 업샘플**이다.
거기서부터 잔차를 배워 나간다.

| 클래스 | 하는 일 | 논문 |
|---|---|---|
| `PANCrafter` | U-Net 본체. mode 임베딩 `s_token` 을 모든 블록에 뿌린다 | Fig. 3 |
| `ResBlock` | GroupNorm → SiLU → Conv, mode 로 scale/shift 변조 | Eq (5)(6) |
| `AttnBlock` | CM3A 호출 → 두 브랜치 출력을 mode 별 gate 로 합산 → FFN | Eq (7)(8) |
| `CMAAA` | ★ CM3A 본체. 국소 attention 두 갈래(PAN·MS) | Eq (9)~(12) |

### 3.2 `CMAAA` — 한 블록 안에서 벌어지는 일

```python
cond  = lpan.repeat(C_ms) if PAN mode else ms      # mode 에 따라 query 재료가 바뀐다  Eq (9)/(12)
q     = Conv([x | cond])
k_pan = Conv([x | lpan])                           # PAN 쪽 key         Eq (11)
v_pan = Conv([x | lpan | pan | pan−lpan])          # PAN 쪽 value: 고주파를 명시적으로 넣는다
k_ms, v_ms = Conv([x | ms]).chunk(2)               # MS 쪽 key/value    Eq (10)

k, v  = dep_conv(각각)                             # k×k(=3×3) 이웃 9개로 펼친다
attn  = softmax over 9 neighbors( q · k )
x_pan, x_ms = (attn·v).sum()                       # 두 브랜치 결과를 AttnBlock 이 합친다
```

`v_pan` 에 `pan − ↑4 lpan`(PAN 의 고주파 성분)이 직접 들어가는 것이 핵심이다.
MS 쪽 query 가 이 값을 국소 이웃에서 골라 가져오는 것이 "정합 보정" 의 실체다.

> ⚠ 이 함수에 논문과 어긋나는 지점이 둘 있다. `k_pan` 이 `k_ms` 로 덮이고,
> `dep_conv` 를 shift kernel 로 고정하는 `reset_parameters()` 가 호출되지 않는다.
> config 의 `model_args.fix_key_alias` / `fix_local_attn` 로 켜고 끌 수 있게 해 두었다.
> 상세: [KNOWN_ISSUES.md A-1, A-2](KNOWN_ISSUES.md)

### 3.3 `train.py` — MARs 가 실제로 구현된 곳

```python
# 배치 복제: 앞 절반 PAN mode(s=0), 뒤 절반 MS mode(s=1)
gt, lms, ms, lpan, pan = [t.repeat(2,1,1,1) for t in ...]
switch = cat(zeros(B), ones(B))

recon = model(pan, lpan, ms, switch)
recon = recon + ↑4ms·switch + ↑4lpan.repeat(C)·(1−switch)      # Eq (1)/(3)

loss_pan = |pan[:B].repeat(C) − recon[:B]|.mean() * w_off      # Eq (4) 앞항
loss_ms  = |gt[B:]           − recon[B:] |.mean()              # Eq (4) 뒷항
```

평가(`test_reduced`, `test_full`)에서는 `switch=1` 로 고정한다. 논문의 "추론 시 MS mode 고정".

### 3.4 `feeders/feeder.py` — 경로 문자열에 의미가 있다

h5 전체를 RAM 에 올리고(`train_wv3.h5` 는 약 5.5 GB), `dataroot` 옆의 `*_pan.h5` 에서
`lpan` 을 따로 읽는다. 값은 `2x/max_pixel − 1` 로 [-1, 1] 정규화한다.

**경로 문자열에서 split(`train`/`valid`/`reduced`/`full`)과 `max_pixel`(`wv3`→2047 등)을
추론한다.** 디렉터리 이름을 바꾸면 조용히 잘못된 정규화로 학습된다.
새 센서를 붙일 때 특히 주의할 것 ([KNOWN_ISSUES.md B-1~B-3](KNOWN_ISSUES.md)).

---

## 4. 저장소 구조

```
PAN-Crafter/
├── INTRO.md               이 문서 — 논문 요지 + 코드 구조
├── REPRODUCTION.md        데이터 배치·환경·실행 절차
├── KNOWN_ISSUES.md        논문 불일치 / 잠재 버그 / 개선안 (ID 로 코드 주석과 연결)
├── README.md              저자 원본
├── 2505.23367v2.pdf       논문
│
├── main.py                진입점 · 학습 루프 · best 선택 · 최종 .mat 내보내기
├── train.py               Trainer (학습 스텝 / 평가 / .mat 저장)
├── utils.py               지표 · 로깅 · metrics.csv
├── model/pancrafter.py    ★ 모델 전부
├── feeders/feeder.py      데이터 로더
│
├── config/
│   ├── pancrafter_wv3.yaml         baseline — 배포본 그대로 (fix_* 모두 False)
│   ├── pancrafter_wv3_fixed.yaml   fixed    — A-1/A-2 적용 (fix_* 모두 True)
│   └── ... qb / gf2 동일 쌍
│
├── tools/
│   ├── run.sh             학습 실행 + 실행 조건 스냅샷을 work_dir/meta/ 에 기록
│   ├── check_data.py      3개 config 의 h5·샘플수·해상도·경로 추론 검증
│   ├── export_mat.py      임의 checkpoint → DLPan-Toolbox 입력 .mat
│   └── plot_metrics.py    metrics.csv → 학습 곡선 · 실험 간 비교표
│
├── data/PanCollection/    ../CANConv 데이터로의 심볼릭 링크 + 저자 배포 *_pan.h5  (gitignore)
├── pan_h5/pan_h5.zip      저자 배포 lpan 원본
│
├── work_dir/<실험>/       (gitignore)
│   ├── meta/              config·git commit·환경·nvidia-smi·시작/종료 시각
│   ├── train_log.txt      iteration 별 loss
│   ├── test_log.txt       eval 별 지표 (사람이 읽는 형식)
│   ├── metrics.csv        eval 별 지표 (기계가 읽는 형식) ← 곡선·사후 best 재선택
│   ├── console.log
│   ├── best_reduced/  best_full/  epoch-*/  checkpoint-*/  lastest/
│   └── results/
│       ├── reduced_best_reduced.mat  full_best_reduced.mat     ← MATLAB 입력
│       ├── reduced_best_full.mat     full_best_full.mat
│       └── reduced/*.png  full/*.png                            ← 시각 확인용
│
└── results_log/           실험 기록 (../CANConv/results_log 와 동일 규약)
    ├── CONVENTION.md      작성 규약
    ├── PURPOSE.md         무엇을 위해 쓰는가
    ├── README.md          인덱스
    └── assets/            그림
```

---

## 5. 실험을 왜 두 벌 돌리는가

배포 코드가 논문과 다르게 동작하는 지점이 둘 있어([A-1](KNOWN_ISSUES.md), [A-2](KNOWN_ISSUES.md)),
어느 한쪽만 돌리면 결과를 해석할 수 없다.

| 실행 | config | `fix_key_alias` | `fix_local_attn` | 무엇을 답하는가 |
|---|---|---|---|---|
| baseline | `pancrafter_wv3.yaml` | False | False | 논문 Table 1 수치가 배포 코드로 재현되는가 |
| fixed | `pancrafter_wv3_fixed.yaml` | True | True | 논문대로 고치면 더 좋아지는가 |

두 실행의 파라미터 총량은 9,968,808 개로 같다(토글은 배선과 초기화만 바꾼다).
따라서 성능 차이를 모델 크기 차이로 설명할 수 없다 — 비교가 성립한다.

```bash
./tools/run.sh wv3          # baseline
./tools/run.sh wv3_fixed    # fixed
python tools/plot_metrics.py work_dir/wv3_baseline work_dir/wv3_fixed \
       --out results_log/assets/curve_wv3.png
```

수치는 파이썬 지표로 후보를 좁힌 뒤 `results/*.mat` 을 DLPan-Toolbox(MATLAB)로 평가해
확정한다. 파이썬 지표와 논문 지표는 정의가 다르다 ([KNOWN_ISSUES.md D절](KNOWN_ISSUES.md)).
