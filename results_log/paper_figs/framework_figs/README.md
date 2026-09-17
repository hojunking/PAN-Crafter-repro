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
| `S_ms_native.png` | (도식에 없음) | 원 해상도 MS `S`. MS↑ 앞에 박스를 하나 더 두고 싶을 때 쓴다 |

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
