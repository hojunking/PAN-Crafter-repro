# GF2 데이터셋 실험 분석 — HQNR 격차의 원인 감사 (2026-09-20)

GF2 는 세 센서 중 유일하게 논문 FR HQNR(0.964) 에 못 미친다. 현 mainline recipe(단일 HRMS · 4-band · W112·D123, `B01_GF2_P0`) 가
**0.9314(D_s 0.0433)** 인데, 같은 recipe 계열이 QB 에서는 논문(0.92) 과 동률이고 WV3 에서는 논문과 같은 수준이다(QRECON24 G23 S1234 0.9591 vs 논문 0.958 — 판정선 0.0031 안). 사용자 질문은 "지표 구현이나
데이터 취급이 GF2 에 맞지 않은 부분이 있는가" 였다. 이 문서는 **지표 경로 · 데이터 · recipe · 문헌** 네 갈래를 CPU 만으로 전수 점검한 결과다
(학습·GPU 추론 없음 — s1 은 QRECON24 seed 학습 중). 스크립트는 `tools/gf2_audit/`, 수치 파일은 `results_log/assets/gf2audit/`.

| 대상 | 조건 | 구성 | 정량 (FR·paper mat20, raw_original, `matlab`·EVAL_VERSION 2026-09-10.5) |
|---|---|---|---|
| 논문 PAN-Crafter (reported) | GF2 FR 20 장 | 7.17 M, dual MARs + CM3A | HQNR **0.964±0.015** · D_λ 0.020±0.013 · D_s **0.017±0.007** |
| 옛 dual recipe (s1, 3 seed) | `ARCH_W168_D123_DUAL_GF2_S{1234,2025,7777}` best_hqnr | W168·D123 · dual MARs · mode modulation · 7ch 입력 | HQNR 0.9625 / 0.9576 / 0.9594 · D_s 0.0181 / 0.0234 / 0.0200 · D_λ 0.0197 / 0.0194 / 0.0209 |
| **현 mainline** (s1) | `B01_GF2_P0_W112_D123_S2025_BOOTSCRATCH` best_hqnr | W112·D123 · 단일 HRMS · 5ch(4 MS+PAN) · aligner 없음 | HQNR **0.9314** · D_λ 0.0264 · **D_s 0.0433** · RR ERGAS 0.5854 |
| aligner 계열 (s1) | `B03_GF2_L1E4` / `B03_GF2_L000` / `B04_GF2_L1E4_S1234` | P0 + WV3 donor aligner 재학습 | 0.9526 / 0.9468 / 0.9500 (D_s 0.0225 / 0.0292 / 0.0261) |
| 평가기 anchor | `_ref_cannet_gf2` (CANConv 배포 가중치) | 우리 평가기 | 0.9189 / 0.0194 / 0.0629 — CANConv 논문 GF2 행 0.919 / 0.019 / 0.063 과 **셋째 자리 일치** |

**결론: 지표·데이터는 GF2 에 맞게 이식돼 있고 anchor 5 종이 전부 맞는다. 격차 0.026–0.033 은 recipe 효과다** — 단일-HRMS 4-band recipe 의
출력이 GF2 에서 PAN 과 **과잉 상관**(4 밴드 × 20 장면 전부) 해 D_s 가 dual recipe 의 2.4 배로 벌어진다. 격차의 74% 가 D_s, 26% 가 D_λ 다.
다만 dual 대조군은 폭(W168)·입력 채널(7ch)·task(dual MARs)·mode modulation 이 **한꺼번에** 다르고 단일 변수 run 이 없어, 어느 요소가 얼마인지는
이 감사에서 분리하지 못했다(§7).

수치 신뢰도: FR 은 전부 같은 평가기(`tools/eval_fr_paperset.py`, 논문 세트 .mat 20 장, raw_original 전체 프레임) 이고 RR 은 `tools/eval_dlpan.py` 규약
(crop 20:-21, peak 1023, SCC=SCC.m, SSIM Gaussian 11×11) 이다. 단일 seed 비교(S2025 대응) 가 대부분이며, HQNR 판정선 2σ = 0.0031 은 WV3 실측값이라
GF2 에서는 참고치다 — 그러나 여기서 다루는 차이(0.026–0.033) 는 그 10 배다.

---

## 1. 질문의 구조 — 무엇이 GF2 에서 "다른가"

| | WV3 | QB | GF2 |
|---|---|---|---|
| 밴드 · 비트 | 8 · 11-bit (2047) | 4 · 11-bit (2047) | **4 · 10-bit (1023)** |
| 학습 patch (MS 16² → PAN 64²) | 9,714 | 17,139 | **19,809** |
| D_λ MTF (genMTF.m) | 센서표 [0.325…0.315] | 센서표 [0.34 0.32 0.30 0.22] | **`otherwise` 0.3** (MATLAB 에 GF2 항목 없음) |
| 논문 FR HQNR / D_s | 0.958 / 0.027 | 0.920 / 0.039 | 0.964 / **0.017** |
| 현 mainline P0 HQNR / D_s | 0.9507 / 0.0261 (`PALS24_CTRLP0_…S2025`) | 0.9149 / 0.0257 (`B01_QB_P0`) | **0.9314 / 0.0433** |

GF2 만 특이한 것은 10-bit 와 MTF 기본값 분기다. 이 둘이 지표·데이터 경로에서 제대로 처리되는지가 §2·§3 이고, 그래도 남는 격차가 §4 다.

## 2. 지표 경로 — GF2 에 맞게 이식됐는가

### 2.1 anchor 다섯 개가 전부 맞는다

| anchor | 우리 값 | 문헌 값 | 출처 |
|---|---|---|---|
| CANConv 배포 가중치 GF2 **FR** | HQNR 0.9189 · D_λ 0.0194 · D_s 0.0629 | 0.919±0.011 · 0.019±0.010 · 0.063±0.009 | `work_dir/_ref_cannet_gf2/results/fr_mat20.json` (09-08, `tools/make_cannet_reference.py`) vs PAN-Crafter Table 7 |
| CANConv 배포 가중치 GF2 **RR** | SAM 0.7128 · ERGAS 0.6362 · Q4 0.9824 | (시트 `GF2-s1` □ CANConv 행) | 시트 |
| EXP(lms) GF2 **RR** | SAM 1.823 · ERGAS 2.366 · Q4 0.809 | 1.820±0.403 · 2.366±0.554 · 0.812±0.051 | CANConv arXiv 2404.07543 Table 3 EXP 행 |
| EXP(lms) WV3 **FR** (대조군) | D_λ 0.0232 · D_s 0.0813 · HQNR 0.8975 | 0.0232±0.0066 · 0.0813±0.0318 · 0.897±0.036 | CANConv Table 1 EXP 행 — **넷째 자리 일치** |
| 옛 dual recipe 3 seed | HQNR 0.958–0.963 · D_s 0.018–0.023 | 논문 Table 12 CM3A×·MARs× baseline 0.959 · 0.021 | `ARCH_W168_D123_DUAL_GF2_S*` |

EXP GF2 FR 은 우리 값이 D_λ 0.0180±0.0081 · D_s 0.0957±0.0209 · HQNR 0.8881±0.0228 인데 **문헌에 GF2 EXP FR 행이 없다** — DLPan 벤치마크 논문
(Deng 2022) 은 GaoFen-2 를 다루지 않고(WV2/WV3/WV4/QB/IKONOS), CANConv 는 GF2 EXP 를 RR 만 준다. 대조는 WV3 EXP 로 대신했고 넷째 자리까지 맞는다.

**시트 `GF2-s1` 의 CANConv 행 FR 칸이 비어 있는 것은 평가기 문제가 아니라 표시 게이트다** — `gspread/gspread_upload.py` L150 이
`eval_version != 2026-09-10.5` 인 JSON 을 버리는데 참조 4 개(`_ref_cannet_*`) 가 전부 2026-09-08.4 다(WV3·QB 탭도 같은 이유로 비어 있다).
`.4 → .5` 는 V64 view 필드 추가뿐이라 D_λ/D_s/HQNR 은 3e-4 이내에서 같다. 참조 디렉터리에 `tools/eval_fr_paperset.py` 를 한 번 더 돌리면 채워진다.

### 2.2 상수 감도 — 어느 상수도 격차를 만들지 못한다

저장된 `full_best_hqnr*.mat` 을 상수만 바꿔 재평가했다(`tools/gf2_audit/sweep.py` → `assets/gf2audit/sweep_results.json`).

| 설정 | B01_GF2_P0 HQNR / D_λ / D_s | ARCH_W168 S1234 | CANConv |
|---|---|---|---|
| 기준 (무클립, GNyq 0.3 = genMTF `otherwise`) | 0.9314 / 0.0264 / 0.0433 | 0.9625 / 0.0197 / 0.0181 | 0.9189 / 0.0194 / 0.0629 |
| 클립 [0, 2¹⁰] (L=10, th_values=1) | 0.9314 / 0.0264 / 0.0433 | 0.9625 | 0.9189 |
| 클립 [0, 2¹¹] (WV3 처럼 L=11) | 0.9314 / 0.0264 / 0.0433 | 0.9625 | 0.9189 |
| 클립 [0, 1023] | 0.9314 | 0.9625 | 0.9189 |
| 정수 DN 반올림 | 0.9316 / 0.0264 / 0.0431 | 0.9626 | 0.9189 |
| GNyq 0.20 | 0.9333 / 0.0245 / **0.0433** | 0.9641 | 0.9203 |
| GNyq 0.25 | 0.9329 / 0.0248 / 0.0433 | 0.9639 | 0.9202 |
| GNyq 0.35 | 0.9291 / 0.0289 / 0.0433 | 0.9601 | 0.9167 |
| GNyq 0.40 | 0.9259 / 0.0322 / 0.0433 | 0.9570 | 0.9137 |
| GNyq QB 표 [0.34 0.32 0.30 0.22] | 0.9318 / 0.0261 / 0.0433 | 0.9628 | 0.9191 |
| GNyq GeoEye1 0.23 | 0.9332 / 0.0245 / 0.0433 | 0.9642 | 0.9204 |

- 클리핑은 어느 L 에서도 **0.0000** — 출력이 이미 [0, 1023] 안에 있다. 반올림 +0.0002.
- GNyq 는 D_λ 만 움직이고 **D_s 는 어떤 상수에도 반응하지 않는다**(D_s.m 에 센서 상수가 없다). 0.20–0.40 전 범위에서 B01 HQNR 은 0.9259–0.9333 —
  최대 **+0.0019**. 격차 0.033 중 D_s 몫 0.026 은 지표 상수로 설명할 수 없다.
- 10-bit 스케일 1023 은 `feeders/feeder.py` L28-29(dataroot 의 `gf2` → max_pixel 1023) · config `max_pixel: 1023.0` · `tools/eval_dlpan.py` L88 `SCALE gf2=1023` ·
  `tools/eval_fr_paperset.py` R=1023 · `train.py` L315 · `train_kdv.py` L1681 전 구간에서 일관된다. RR PSNR 44.06 이 논문 45.08 과 같은 자리인 것이
  그 증거다(2047 이었다면 +6 dB).
- D_λ 의 MTF 는 `tools/metrics/eval_fr.py` L68 `SENSOR_NAME["gf2"] = None` → L214-217 GNyq 0.3·ones. MATLAB `genMTF.m` 에 GF2 case 가 없어 `otherwise
  GNyq = 0.3` 을 타므로 **논문의 MATLAB 경로와 같은 분기**이고, PanCollection GF2 의 `ms` 자체가 정확히 그 커널로 만들어졌다(§3.1).

부수 지적(수치 경로 밖): `feeders/feeder.py` L25-32 의 max_pixel 경로 추론이 `wv3`/`qb`/`wv2` 토큰을 `gf2` 보다 먼저 검사하고 토큰이 없으면 print 만 하고 1.0 으로
진행한다 — s1 의 GF2 경로에는 다른 토큰이 없어 1023 이 맞았지만(저장 sr max 1023.00), 경로에 `qb` 가 섞인 서버·복사본에서는 조용히 깨진다(잠재 버그 B-2). JQM 의 PAN MTF 커널이 GF2 에서 QB 기본값 0.15 로 조용히 대체된다(`tools/metrics/jqm.py` L33·L66; JQM 은 판정 지표가 아님 —
인용 시 "PAN GNyq 0.15 기본값" 명시). `eval_fr.py`/`eval_rr.py` 의 독립 CLI 는 없는 `tools/presets.json` 을 읽어 깨져 있고 docstring 이 옛 12-19 프로토콜을
서술한다(시트·selector 경로는 함수 호출이라 무관).

## 3. 데이터 — F-1 · F-3 · 세트 동일성 · 정합

### 3.1 결함 검사 (WV3·QB 대조 포함)

| 검사 | GF2 | 대조 | 스크립트 |
|---|---|---|---|
| **F-3**(ms 가 gt 의 MTF·(2,2) 위상 데시메이션인가) | train 400/400 · valid 400/400 · RR test 20/20 이 (2,2), MAD **0.0000 DN** | QB 원본(양성 대조) 63% 가 (1,2)/(2,1), MAD 3.54 DN · QB msfix 0.0 · WV3 0.0 | `f3_check.py` |
| **F-1**(lpan 이 다른 장면인가) | 배포 lpan vs 복구 레시피(Gaussian σ1.98·41탭·[2::4]) 상관 **≥0.999997**(RMSE 0.25–0.45 DN, FR·train) | WV3 **배포** lpan 은 상관 0.011(F-1 재현, 양성 대조); 복구된 mat20 은 1.000 | `f1_reg_check.py` |
| FR 세트 정체성 | `full_examples_mat20` = PanCollection `Test(HxWxC)_gf2_data_fr1..20.mat`(provenance sha256 20 건) = 배포 H5 20 장, **화소 단위 동일**(max\|diff\| 0.0) | — | `inspect_gf2.py` |
| 값 범위·포화 | FR PAN max 1023 · MS max 1004 · lms max 1043(interp23tap overshoot) · ≥1023 픽셀 0.000% | WV3 2047/2047/2307 | `h5dump.py` |
| MTF 생성값 | GNyq 0.25/0.35 로 바꾸면 MAD 1.01/0.97 DN — **0.3 이 곧 데이터 생성값** | — | `f3_check.py` |
| 세트 크기 | RR 20×256²·FR 20×512² (논문 §4.1 과 동일) · train 19,809 · valid 2,201 | WV3 9,714 / 1,080 | `inspect_gf2.py` |

QB 를 무효로 만들었던 F-3 도, WV3·QB 에서 복구가 필요했던 F-1 도 GF2 에는 없다. `*_msfix` 같은 복구본이 GF2 에 없는 것은 필요가 없기 때문이다.
GF2 run 들이 읽은 파일(`work_dir/<run>/meta/config.yaml`) 은 전부 위 원본이다.

### 3.2 GF2 고유의 데이터 성질 (관측 — 원인으로 확정하지 않는다)

PAN↔MS 정합 오프셋을 PAN 저역 vs 업샘플 MS intensity 의 NCC 정점으로 쟀다(HR px, (dy, dx) 평균; `f1_reg_check.py`).

| 세트 | GF2 | WV3 | QB |
|---|---|---|---|
| FR mat20 (원본) | (+1.44, −0.20) | (+1.54, −0.53) | (+2.36, −1.19) |
| RR test (Wald) | **(+0.44, −0.15)** | (−0.17, +0.01) | (+0.21, −0.42) |
| train 400 (Wald) | **(+0.33, +0.27)** | (−0.06, −0.03) | (+0.06, −0.08; msfix) |

- 원본 FR 에서 GF2 는 WV3·QB 사이에 있다 — **GF2 고유의 정합 결함이 아니다.** 검증 단계가 독립 방법(PAN 저역 LR 격자 + NNLS 강도 + phase correlation ×100) 으로 다시 재도
  |Δ| 평균 GF2 1.05 HR px · WV3 0.89 · QB 2.26 으로 순위가 같다. 오프셋이 더 큰 QB 에서 같은 P0 recipe 가 논문보다 좋은 D_s(0.0257 vs 0.039) 를 낸다.
- 대신 GF2 의 **Wald 학습·RR 쌍이 약 0.4 HR px 의 잔여 오프셋**을 갖는다(WV3 ≈ 0.06). 데이터 감사자는 NIR 이 가시 밴드 대비 ~0.9 HR px 더 어긋난다고 관측했다(검증 단계에서 재측정하지 않음).
- GF2 원본 MS 는 PAN 대비 가장 흐리다 — 유효 GNyq 가 native FR 에서 밴드별 0.04–0.11(가정 0.3), Wald RR 에서 0.23–0.27(`eff_gnyq_exp.py`, `ms_sharpness.json`).
  WV3 native 는 0.18–0.27. 즉 RR 학습 ↔ FR 시험의 MS 흐림 불일치가 GF2 에서 가장 크다.

이 둘은 "GF2 가 recipe 변화에 왜 더 민감한가" 의 후보이지 격차의 원인이 아니다 — **같은 데이터로 학습한 dual recipe 가 D_s 0.018 을 낸다**(§4). B01 의
장면별 D_s 도 20 장 전부 균일하게 높고(0.028–0.061) 장면별 정합 크기와의 상관은 +0.25 에 그친다.

## 4. recipe — 격차는 어디서 오나

### 4.1 D_s 를 밴드별로 분해하면 "PAN 과잉 상관" 이다

D_s 는 밴드별 q(fused_b, PAN) 와 q(lms_b, PAN_lp) 의 차이다. 부호까지 보면(`ds_decomp.py` → `assets/gf2audit/ds_decomp.txt`):

| run | q_high − q_low (B, G, R, NIR) | q_high > q_low 비율 | D_s |
|---|---|---|---|
| **B01_GF2_P0** (단일 HRMS) | **+0.026, +0.044, +0.047, +0.056** | **1.00** | 0.0433 |
| ARCH_W168_DUAL_GF2_S2025 | +0.006, +0.021, +0.011, +0.051 | 0.91 | 0.0234 |
| B03_GF2_L1E4 (aligner) | +0.015, +0.017, +0.021, +0.031 | 0.89 | 0.0225 |
| B02_GF2_DONN2 (WV3 donor aligner) | −0.002, −0.008, −0.009, +0.014 | 0.50 | 0.0119 |
| B01_QB_P0 | +0.017, +0.009, −0.006, −0.052 | 0.42 | 0.0257 |
| ARCH_W168_DUAL_QB_S2025 (미완 run) | +0.025, +0.021, +0.008, −0.033 | 0.64 | 0.0235 |
| PALS24_CTRLP0 WV3 (단일 HRMS) | 혼재 (8 밴드) | 0.67 | 0.0261 |
| S1_T05 W168 dual WV3 | 혼재 | 0.54 | 0.0211 |

GF2 P0 만 **네 밴드·20 장면 전부에서 fused 가 PAN 을 더 따른다**(q_high > q_low). 이 성질은 09-09 정렬 분석에서 기록된 "GF2 에서는 PAN 을 덜 따를수록
D_s 가 내려간다" 와 같다. 장면별 Ds(P0) − Ds(DUAL) 는 평균 +0.0198(최소 +0.0113, 20/20 양), Dλ 차는 +0.0070(20/20 양).

격차 분해(S2025 대응, HQNR = (1−D_λ)(1−D_s)): P0 의 D_s 만 dual 의 0.0234 로 두면 0.9508 → **0.0262 중 +0.0194(74%) 가 D_s, +0.0068(26%) 가 D_λ.**

### 4.2 선택 편향이 아니다

| run | eval 횟수 | best HQNR | last | D_s min / median / max |
|---|---|---|---|---|
| B01_GF2_P0 | 61 | 0.9315 (ep 66; 실제 선택은 tol 1e-4 안 fSCC tie-break 로 ep 116 = step 47,792, 0.93145) | 0.9314 | **0.0433 / 0.0457 / 0.0739** |
| ARCH_W168_DUAL_GF2_S2025 | 25 | 0.9576 (ep 85) | 0.9562 | 0.0234 / 0.0257 / 0.0537 |

B01 은 61 회 eval 전부에서 D_s ≥ 0.0433 이고, `full_last` · `full_best_rr_val` · `best_hqnr` 세 checkpoint 를 같은 평가기로 재계산해도 **셋 다 HQNR 0.9314 / D_s 0.0433** 이다 — 어느 checkpoint 를 골라도 격차가 있다(`traj.py`; 검증 단계 재계산).

### 4.3 두 recipe 는 무엇이 다른가 (config diff)

`config/B01_GF2_P0_…yaml` vs `config/ARCH_W168_D123_DUAL_GF2_S2025.yaml` 에서 다른 키 전부:

| 키 | 현 mainline P0 | 옛 dual |
|---|---|---|
| `mars` (task) | `ms` — 단일 HRMS | `dual` — HRMS + PAN 재구성(dual MARs) |
| `model_args.in_mode` (입력 조립) | `paper` — **5ch** (4 MS + PAN) | `released` — **7ch** (PAN, up(LPAN), PAN−up(LPAN), up(MS)) |
| `model_args.hidden_size` | 112 | 168 |
| `model_args.mode_modulation` | False | True |
| `trainer` | `kdv` (A-ID/NOALIGN, rec N0, stat off) | legacy |
| `eval_epoch` / `save_iter` | 2 / 25000 | 5 / 10000 |

같은 recipe 의 GF2 ↔ QB config(`B01_GF2_P0` vs `B01_QB_P0`) 는 센서 키(`max_pixel` 1023/2047, dataroot, 후보 격자 id, eval_epoch) 만 다르다 —
recipe 는 센서에 대해 올바르게 파라미터화돼 있고 GF2 에서만 다른 코드 경로를 타지 않는다. WV3 T0 의 보정 상수(τR·λE0) 는 P0 에 쓰이지 않는다.

**분리할 수 없는 것**: task · 입력 채널(lpan/HPAN 유무) · 폭 · mode modulation 이 함께 바뀐다. W112 dual 이나 W168 단일 run 이 GF2 에 하나도 없다.

### 4.4 다른 센서에서도 같은 방향이다 — GF2 가 가장 크고 D_s 로 나타난다

| 센서 | 단일 HRMS P0 | dual | Δ(단일−dual) | 어디서 |
|---|---|---|---|---|
| WV3 | 0.9507 (`PALS24_CTRLP0 S2025`) | 0.9612 (`S1_T05_W168_D123_DUAL`) | −0.011 | D_s +0.005, D_λ +0.006 |
| QB | 0.9149 (`B01_QB_P0`) | 0.9355 (`ARCH_W168_DUAL_QB_S2025`, **약 21K step 에서 중단된 미완 run** — console.log 마지막 Iter 21,420, metrics.csv 마지막 eval ep 55 = step 19,635, 11 회) | −0.021 | D_λ +0.019 (D_s 는 오히려 +0.002) |
| GF2 | 0.9314 (`B01_GF2_P0`) | 0.9576 (`ARCH_W168_DUAL_GF2_S2025`) | **−0.026** | **D_s +0.020**, D_λ +0.007 |

QB 가 "논문과 동률" 인 것은 D_λ 악화(0.0611 vs 논문 —)와 D_s 개선(0.0257 vs 0.039)의 상쇄다. 따라서 "QB 는 괜찮고 GF2 만 나쁘다" 는 전제는
절반만 맞다 — dual → 단일 하락은 세 센서 공통이고, GF2 에서 가장 크며 D_s 로 나타난다.

### 4.5 B02 의 0.967 은 이득이 아니다

`B02_GF2_DONN2`(WV3 N2 donor, R200 — 이식된 것은 radius b=2 HR px 뿐이고 가중치는 GF2 4-band init) 의 HQNR 0.9670 은 aligner 가 FR 에서 PAN 을
dy 중앙값 +1.73 px 옮겨 raw_original D_s 가 0.0119 로 내려간 것(best step 22,248) + test-set 선택(last step 50,000 은 0.9479 / D_s 0.0299; 선택 폭 0.019 ≫ 판정선 0.0031) 이다. 완벽히 정렬된 Wald RR 에서도 PAN 을 ~0.9 px 옮겨 ERGAS 가 0.6461 로
**P0(0.5854) 보다 나빠진다.** B02−B01 = +0.0356 중 D_s 몫 +0.0306, last 기준이면 +0.0165. 시트 HQNR 만 보면 오독한다.

## 5. 문헌 — 논문의 GF2 D_s 0.017 은 예외인가

PAN-Crafter Table 7 (GF2 FR, HQNR / D_s / D_λ): PanNet .929/.052/.020 · MSDCNN .898/.079/.026 · FusionNet .865/.105/.034 · LAGNet .895/.078/.030 ·
S2DBPN .935/.046/.020 · PanDiff .936/.045/.020 · DCPNet .953/.024/.024 · TMDiff .942/.030/.029 · CANConv .919/.063/.019 · **PAN-Crafter .964/.017/.020**.
다른 논문: SSDiff .9573/.0267/.0164 (arXiv 2404.11537) · U-Know-DiffPAN FSA-T .953/.030/.017 (Table 2/8) · RAFNet .9615/.0223 (WFANet arXiv 2502.04903 표
2차 인용). 논문 값은 문헌 최상위이지만 상위군과 0.003–0.007 차이라 **프로토콜 이탈로 볼 근거가 없고**, 우리 파이프라인도 같은 평가기로
0.958–0.967 을 실측했다(§4). 논문 자체 ablation(Table 12) 의 CM3A×·MARs× baseline 이 .959/.021 이고 우리 dual 3 seed 가 .9576–.9625/.018–.023 이므로
**D_s ≈ 0.02 가 이 모델군의 정상 범위**다.

부수: 논문 본문 Table 2 의 GF2 RR ERGAS 0.522 와 보충 Table 7 의 0.552±0.093 이 불일치한다(다른 9 개 방법은 일치). 시트는 0.552 를 채택.

## 6. 배제한 가설

| 가설 | 검증 | 결과 |
|---|---|---|
| 10-bit 스케일(1023) 이 어딘가에서 2047 로 처리된다 | feeder·config·RR/FR 평가기·selector·mat 내보내기 경로 추적 + RR PSNR 자리 | 전 구간 1023. **기각** |
| FR 클리핑 L 이 틀렸다 | L=10/11/무클립/1023 재평가 | 0.0000 차이. **기각** |
| GF2 용 MTF(GNyq) 가 틀렸다 | genMTF.m `otherwise` 확인 · 데이터 생성 커널 MAD · 0.20–0.40 sweep | 0.3 = MATLAB = 생성값; sweep 최대 +0.0019, D_s 불변. **기각** |
| 4-band Q2n/D_λ 에 8-band 가정이 남아 있다 | CANConv anchor D_λ 0.0194 vs 0.019 | **기각** |
| FR 세트가 논문 세트가 아니다 | mat20 sha256 20 건 · H5 와 화소 동일 | **기각** |
| F-1 (lpan 다른 장면) | 상관 1.000 | **기각** |
| F-3 (ms 1 px 위상 어긋남) | 820 장 (2,2) MAD 0.0 · QB 양성 대조 재현 | **기각** |
| GF2 원본이 유독 어긋나 있다 | NCC 오프셋 세 센서 비교 | WV3·QB 사이. **기각** |
| best_hqnr 선택이 나쁜 checkpoint 를 잡았다 | 61 회 eval 궤적 | D_s 최소 0.0433, best≈last. **기각** |
| WV3 T0 보정 상수(τR·λE0·b=2) 를 GF2 에 잘못 이식했다 | P0 config | P0 에 쓰이지 않음; L1E4 는 이식됐으나 L1E4−L000 = +0.006(양). **기각** |
| 평가기 자체가 GF2 에서 낮게 나온다 | dual 3 seed 0.958–0.963 · B02 0.967 | 상한 문제 아님. **기각** |
| 논문 D_s 0.017 이 비정상 프로토콜 | 문헌 10 편 대조 | 상위 정상 범위. **기각** |

## 7. 남은 것 — 다음에 돌릴 것

1. **단일 변수 2×2 ablation (GF2 S2025, W112·D123, 4 run × ~1.4 h ≈ 5.5 h)**: `mars {ms, dual}` × `in_mode {paper 5ch, released 7ch}`, 나머지는 B01 config 그대로
   (select_on hqnr, fr_select_indices 0-19). D_s +0.02 과잉 상관이 PAN 재구성 task 인지 lpan/HPAN 입력 채널인지를 가른다. W168·`ms` 1 run 을 더하면 폭 몫까지 분리.
   s1 의 QRECON24 seed 큐(~18 h) 뒤에 편성한다.
2. **B01 seed 반복 2 run(1234/7777, ~2.8 h)**: dual 3 seed 폭이 0.0049 라 격차 0.026 은 seed 밖일 것으로 예상되지만 확정은 실측으로.
3. **시트 anchor 칸 채우기(CPU 1 분)**: `python tools/make_cannet_reference.py --sensor gf2 --rr ../CANConv/data/datasets/gf2/sr_cannet_rr.h5 --fr ../CANConv/data/datasets/gf2/sr_cannet_mat20.h5`
   (wv3/qb/wv2 동일; torch 없이 저장 h5 만 읽어 현재 EVAL_VERSION 으로 다시 찍는다 — work_dir 쓰기라 이번 감사에서는 실행하지 않았다). 값은 0.9189/0.0194/0.0629 그대로일 것.
4. **B02 류 aligner run 의 선택 폭 확인(CPU)**: B02 의 best−last 0.019 가 aligner run 공통인지 `checkpoint_metrics.csv` 로 — aligner 를 GF2 해법으로 보기 전 필요.
5. (선택, GPU) B01 에 추론 모드 `A_CROP64_MED`(`kdv/eval_modes.py`; 장면당 상수 Δ 로 PAN 한 번 warp) 를 적용해 D_s 가 얼마나 내려가는지 — B02 의 D_s 0.012 가 학습이 아니라
   '추론 시 PAN 전역 이동' 만으로 나오는지 판별. 학습·추론 정책을 자동으로 바꾸지 않는 원칙은 유지.
6. §3.2 의 Wald 잔여 오프셋(~0.4 px)·MS 흐림 불일치·NIR 어긋남이 recipe 민감도의 기전인지는 1 의 결과가 나온 뒤 판단한다.

## 8. 검증 방법

감사는 독립 4 갈래(지표·데이터·recipe·문헌) 를 병렬로 돌린 뒤 적대 검증 단계가 각 주장을 CPU 재계산으로 다시 확인하는 절차로 했다.
적대 검증의 판정은 §8.1 에 그대로 옮긴다.

### 8.1 적대 검증 판정

검증자는 네 감사자의 핵심 수치를 전부 CPU 로 다시 계산했다(`verify_fr.py` · `verify_rr.py` · `verify_data.py` · `verify_reg.py`; 세션 scratch 에 남았고 결과는 본문 수치와 같다).

| 주장 | 판정 | 근거 |
|---|---|---|
| 격차는 recipe 효과 — 단일-HRMS 5ch W112 recipe 가 GF2 에서 fused 를 PAN 과 과잉 상관시켜 D_s 0.0433 | **CONFIRMED** | B01 / ARCH 3 seed / 장면별 Δ 재계산 일치; 4 밴드 × 20 장면 80/80 양수; CANConv 배포 가중치도 GF2 에서 100% 양수(D_s 0.063 = 논문값) — "PAN 을 그대로 따를수록 D_s 벌점" 인 데이터 |
| 지표 구현이 GF2 에 맞지 않는다(bit depth·클리핑·L·반올림·GNyq·D_s 상수·Q4·집계) | **REFUTED** | anchor 재계산 = JSON = 논문(평균·N−1 σ 까지); sweep 최대 +0.0019; D_s 무반응; 저장 sr max 1023.00 |
| GF2 데이터 결함(F-1 · F-2 · F-3 · MTF 생성값) | **REFUTED** | F-2 max\|diff\| 0.0 · F-1 상관 ≥0.999997(WV3 양성대조 0.011) · F-3 (2,2) MAD ≤3.8e-13(QB 양성대조 57% 어긋남) · GNyq 0.3 = 생성값 |
| best checkpoint 선택 편향 | **REFUTED** | last / best_rr_val / best_hqnr 셋 다 0.9314 / 0.0433; 61 회 eval D_s 최소 0.0433 |
| GF2 FR 세트의 정합 오프셋이 GF2 특유 결함 | **REFUTED** | 세 방법 모두 GF2 가 WV3·QB 사이; 오프셋 더 큰 QB 에서 같은 recipe 가 논문보다 좋음 |
| recipe 안의 어느 요소(task · 입력 채널 · 폭 · mode modulation)가 D_s 를 잡는가 | PLAUSIBLE(미분리) | 단일 변수 run 0 건 — §7 의 2×2 ablation 필요. 논문 Table 12 는 MARs 단독 .945/.032 · 둘 다 .964/.017 로 방향만 준다 |
| GF2 고유 데이터 성질(MS 가장 흐림 · NIR 추가 어긋남 · patch 2 배)이 민감도를 설명 | PLAUSIBLE(미재측정) | 데이터 감사자 관측; 격차 기전과의 연결 미정량 |

감사자 오류 정정 2 건: (1) recipe 감사자의 "B01 best ep 66" — 실제 선택 checkpoint 는 ep 116(step 47,792; tol 1e-4 안 fSCC tie-break), 효과 1e-4 로 결론 불변.
(2) QB dual 의 중단 지점 — console.log Iter 21,420 / metrics.csv 마지막 eval step 19,635; 어느 쪽이든 50K 미완주.

## 부록 A — 약명 → 세팅

| 약명 | 세팅 |
|---|---|
| P0 (`B01_*_P0_W112_D123_S2025_BOOTSCRATCH`) | SMEC12 준비 학습 무정합 baseline: aligner 없음(A-ID/NOALIGN), 새 U-Net W112·D123 (4-band 2.6508 M), 단일 HRMS L1, 50K, batch 48, AdamW 1e-4 wd 0.01 cosine warmup 100, 입력 `paper` 5ch, seed 2025 |
| DONN2 (`B02_*_DONN2_*_R200_*`) | P0 + WV3 N2(R200) donor aligner 를 붙여 공동 학습 |
| L000 / L1E4 (`B03_*`, `B04_*`) | B02 의 aligner 를 donor 로 λ_off 0 / 1e-4 (WV3 PALS24 값 그대로) |
| dual (`ARCH_W168_D123_DUAL_GF2_S*`) | 09-09 다중 데이터셋 3-seed: W168·D123, dual MARs(HRMS + PAN 재구성), mode modulation, 입력 `released` 7ch, attn 없음, nocrop |
| CANConv (`_ref_cannet_gf2`) | CANConv 배포 가중치 `cannet_gf2.pth` 출력을 우리 평가기로 잰 참조 |
| EXP | 23-tap 업샘플 MS(`lms`) 를 그대로 출력으로 본 무처리 기준 |

## 부록 B — provenance

- FR: `work_dir/<run>/results/fr_mat20.json` (`eval_version 2026-09-10.5`, best_hqnr; `_ref_cannet_gf2` 만 2026-09-08.4), 입력 `data/PanCollection/GF2/full_examples_mat20/test_gf2_OrigScale_mat20.h5`.
- RR: `work_dir/<run>/results/reduced_best_hqnr.mat` → `tools/eval_dlpan.py` 규약(`gspread_upload._rr`), `reduced_examples_h5/test_gf2_multiExm1.h5`.
- 학습 궤적: `work_dir/<run>/metrics.csv` (`hqnr_official`, `d_lambda_official`, `d_s_official`).
- 스크립트 `tools/gf2_audit/`(README 에 절별 대응) · 수치 파일 `results_log/assets/gf2audit/`.
- 문헌 원문은 저장소 밖(arXiv 2505.23367v2 · 2404.07543 · 2404.11537 · 2502.04903, DLPan 벤치마크 GRSM 2022).
