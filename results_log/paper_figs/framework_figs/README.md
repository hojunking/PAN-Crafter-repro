# Framework 그림 (Stage 1) 회색 박스용 샘플 영상 — 2026-09-16

Stage 1 (Teacher / PAN Aligner Training) 도식의 회색 박스에 넣을 실제 영상이다.
**합성이 아니라 실제 T0 Teacher 를 돌려서 나온 산출물**이다.

전체를 한눈에 보려면 `_overview.png`.

## 파일 → 도식의 박스

| 파일 | 도식의 박스 | 내용 |
|---|---|---|
| `P_pan.png` | **PAN  P** (Step 1-a, 1-b 양쪽) | 원 PAN, 흑백 |
| `M_ms_up.png` | **MS↑  M** | bicubic ×4 로 키운 MS (`F.interpolate(ms, scale_factor=4, mode="bicubic")`), RGB |
| `P_aligned.png` | **aligned  P′** | `W(P, c_0)` — aligner 가 낸 c₀ 로 warp 한 PAN |
| `YT_teacher.png` | **Teacher  Y_T** | U-Net 출력 `M + F_θ([P′, M])`, RGB |
| `Y_gt.png` | **HRMS GT  Y** | 정답 HRMS, RGB |
| `P_shifted.png` | **shifted  P_ε** (Step 1-b) | `W(P, ε)`, ε = (+1.6, −1.2) HR px (\|ε\| = 2.0) |
| `S_ms_native.png` | (도식에 없음) | 원 해상도 MS `S`. MS↑ 앞에 박스를 하나 더 두고 싶을 때 쓴다. **표시 128×128** = 32² 네이티브 ×4 — `M` 의 1/4 로 두어 배율 관계가 보이게 한 판 |
| `S_ms_native_512.png` | 같은 위 | **같은 내용의 512×512 판**(32² ×16 nearest). 도식의 다른 박스와 크기를 맞출 때. 128 판을 ×4 한 것과 픽셀 단위로 동일하고 32² 격자도 그대로다 |
| `ET_teacher_err.png` | **e_T** (Stage 2 의 cue) | `\|Y_T − Y\|` 의 **밴드 평균**. `kdv/losses_rec.py` 의 `e_t` 와 같은 식. 변형 `_magma` `_gray` |
| `G_gt_edge.png` | **∇Y** (ℒ_edge 타깃) | GT 의 signed Scharr `(gx,gy)` 크기, 밴드 평균. `pa/losses.py` 의 `scharr` — `output_edge_loss` 가 쓰는 그 커널. 변형 `_inv`(흰 배경) `_cividis` |

### 도식 박스에 넣을 이동 표현 (2026-09-20, `tools/framework_figs_shift_schematic.py`)

edge 오버레이는 **도식 크기(1 인치 남짓)로 줄이면 뭉개진다** — 선이 너무 많아서다.
박스에 넣을 것은 이쪽이다: **PAN 에서 실제로 추출한 윤곽선을 몇 개만 굵게** 그리고 원본/이동본을
두 색으로 겹친다.

| 파일 | 크롭 | 선 | 도식 크기에서 |
|---|---|---|---|
| **`SHIFT_schem_zoom_eps_x1.png`** | 32² 확대 | 3 개 + 화살표 | **또렷하다. 권장** |
| `SHIFT_schem_wide_eps_x1.png` | 128² 전체 | 14 개 | 읽히지만 가늘다 |
| `*_x3`, `*_x6` | 위와 같음 | | **과장판** — 캡션에 배율 필수 |

**`SHIFT_schem_zoom_eps_x1.png` 을 쓰면 과장 없이 해결된다.** ε = 2.0 px 그대로인데도
작은 박스에서 cyan/red 가 또렷이 갈라진다 — 비결은 이동을 부풀리는 게 아니라 **좁게 확대하고
선을 줄여 굵게** 그린 것이다. 기본 출력은 `shift 2.00 px` 라벨을 포함한다.
`3_city/SHIFT_schem_zoom_eps_x1.png`는 원본 데이터에서 라벨만 생략해 다시 출력했다
(`--only 3_city --exaggerate 1 --view zoom --no-label`). 이동량은 동일한 2.0 px다.

- cyan = 원본 PAN 윤곽 · red = 이동한 PAN 윤곽 · 흰 화살표 = 이동 방향
- 윤곽선은 `skimage.measure.find_contours` 로 **실제 PAN 에서 추출**한 것이다(그려 넣은 것이 아니다)
- 배경은 원 PAN 을 alpha 0.35 로 깔아 위성영상임이 보이게 했다
- `--vec c0` 로 aligner 실측 보정(0.11–0.21 px)판도 만들 수 있다 — 그건 ×1 로는 거의 겹쳐 보인다
- `*_x3`/`*_x6` 는 **이동량을 그림에서만 부풀린 것**이다. 쓸 거면 캡션에 배율을 적는다

### PAN 이동을 보이게 하는 edge 오버레이 (2026-09-20, `tools/framework_figs_edge_shift.py`)

`P` 와 `P′`(또는 `P_ε`)를 나란히 놓아도 **눈으로는 구분되지 않는다**(아래 "쓸 때 주의" 참조).
그래서 PAN 의 **edge 만 뽑아 두 색으로 겹친 판**을 따로 둔다 — 겹치면 흰색, 어긋나면 색이 갈라진다.

| 파일 | 내용 | 이동량 |
|---|---|---|
| `EDGE_pan.png` | PAN edge 만 (흑백) | — |
| **`EDGE_shift_eps.png`** | `P`(cyan) vs `W(P, ε)`(red) | **ε = (+1.6, −1.2), \|ε\| = 2.0 px — 실제값** |
| `EDGE_shift_c0.png` | `P`(cyan) vs `W(P, c₀)`(red) | c₀ = aligner 실측(0.11–0.21 px) — **실제값** |
| `EDGE_shift_c0_x8.png` | `P` vs `W(P, 8·c₀)` | **과장 ×8** — 개념 설명용 |
| `*_zoom.png` | 위 셋의 32² 부분 확대 | 콜아웃용 |

**도식에 쓸 것은 `EDGE_shift_eps.png` 다.** ε = 2.0 px 는 학습에 실제로 주입하는 변위이고,
edge 오버레이에서 cyan/red 가 뚜렷이 갈라져 **과장 없이** 보인다.
`EDGE_shift_c0.png` 는 거의 흰색이다 — 그게 사실이다(aligner 가 내는 보정이 0.2 px 수준).
`_x8` 판을 쓸 거면 **캡션에 배율을 반드시 적는다**(과장을 숨기지 않는다는 뜻).

edge 는 128² 네이티브에서 Scharr 로 계산한 뒤 ×4 nearest 로 키웠다 — **한 블록이 PAN 1 픽셀**이라
확대판에서 이동량을 눈으로 셀 수 있다. 약한 텍스처는 gradient 0.18 이하로 잘라 선만 남겼다.

> **2026-09-20 추가** (`tools/framework_figs_maps.py`). 위 두 장은 같은 `t0_rr.npz`·같은 장면·크롭·
> ×4 nearest 규약이라 기존 일곱 장과 픽셀 단위로 겹친다. 기존 PNG 는 건드리지 않았다.
> Scharr 는 **256² 전체에서 계산한 뒤 크롭**한다 — 크롭 경계의 reflect pad 인공물을 피하려는 것.
> 표시 스케일은 0 = 검정(오차 0), 상한은 크롭의 **99 퍼센타일**(소수 화소가 스케일을 먹지 않게).
> 실측값은 `index_maps.csv` — 3_city 기준 e_T 중앙 **18.5 DN** · p99 **74.3 DN**(모델 단위 0.0725),
> edge 중앙 96.5 DN · p99 358.7 DN. **`e_T` 는 DN 으로 저장했고 학습 코드는 [−1,1] 이라 값이 R/2 = 1023.5 배
> 차이난다 — 지도 자체는 같다.**

세 묶음: `1_buildings` (관공서형 건물·주차) · `2_cars` (도로와 차량 행렬) · `3_city` (조밀한 도심 격자).

## 출처와 재현

| | |
|---|---|
| 모델 | **T0 Teacher** = `PALS24_L1E4_W112_D123_WV3_S2025_N2LAST_R200_v1` best_hqnr (step 24240), 사본 `assets/pakd50/T0_run` |
| 자료 | WV3 **reduced-resolution** 20 장면 `data/PanCollection/WV3/reduced_examples_h5_regen/test_wv3_multiExm1.h5` (GT 가 있는 세트라 이걸 쓴다 — FR 세트에는 `Y` 가 없다) |
| 장면·크롭 | `index.csv` 의 `scene / crop_row / crop_col`. 256² 장면에서 **128² 크롭** |
| 저장 크기 | 512×512 = 128² 를 **×4 nearest** 로 확대(보간 없음). 실제 정보량은 128² 그대로다 |
| RGB | WV3 밴드 (4, 2, 1). 스트레치는 **GT 기준 밴드별 1–99 %**, 같은 묶음의 `M`·`Y_T`·`Y`·`S` 에 같은 값을 쓴다(셋을 나란히 비교할 수 있게). PAN 계열은 PAN 의 1–99 % |
| 생성 | `tools/framework_figs_run_t0.py` (T0 forward → npz) → `tools/framework_figs_export.py` (크롭·저장) |

Teacher 출력은 기록된 `work_dir/PALS24_.../results/reduced_best_hqnr.mat` 와 **평균 0.005 DN** 차이다
(1 DN 넘는 화소 0.014 %). autocast 비결정성 수준이고 같은 forward 경로다.

## 도식의 값 박스에 넣을 실측값

`c_0`, `c_ε`, `ε`, `c_ε + ε`, `sg(c_0)` 는 영상이 아니라 숫자다. 같은 크롭의 실측값:

| 묶음 | c₀ (dy, dx) | \|c₀\| | ε | c_ε | **c_ε + ε − sg(c₀)** |
|---|---|---:|---|---|---|
| 1_buildings | (−0.013, +0.108) | 0.109 | (+1.6, −1.2) | (−0.635, +0.925) | (+0.978, −0.383) |
| 2_cars | (+0.212, +0.023) | 0.213 | (+1.6, −1.2) | (−0.266, +0.546) | (+1.122, −0.678) |
| 3_city | (+0.180, +0.023) | 0.181 | (+1.6, −1.2) | (−0.296, +0.551) | (+1.124, −0.672) |

마지막 열이 `L_off` 가 줄이려는 잔차다. 0 이 아니다 — aligner 가 ε 에 반응은 하지만 완전히
상쇄하지는 못한다(관측된 성질이지 결함 표시가 아니다).

## 쓸 때 주의

- **`P` 와 `P′` 는 눈으로 구분되지 않는다.** c₀ 가 0.11–0.21 px 라 128 px 크롭에서 0.1 % 수준의
  이동이다. 도식에서 "aligned" 를 강조하려면 화살표·라벨로 표시하고, **영상이 눈에 띄게 달라
  보이도록 과장하지 말 것.**
- **`P_ε` 도 마찬가지로 ε = 2 px 이라 크게 보이지 않는다.** 원 도식처럼 빨간 점선 박스로
  이동을 표시하는 방식이 맞다.
- 세 묶음 모두 **reduced-resolution 프로토콜**이다 — PAN 256², MS 64², GT 256². 논문 본문의
  full-resolution 수치와 같은 자료가 아니다.
- 위성영상 라이선스 제약이 있다. 외부 서비스에 업로드하지 않는다.
