# Global alignment 캠페인 — 가설·계획 대비 구현의 차이, 이슈, 결과 (상세 기록, 2026-09-05)

이 문서는 보고서가 아니라 **작업 기록**이다. 계획서에 세운 가설과 설계가 구현·실행·결과 단계에서
어디서 어떻게 달라졌는지, 왜 달라졌는지, 그 결과 무엇이 확인됐는지를 순서대로 남긴다.

- 가설·계획: [s1_w152_d123_global_alignment_40h_plan.md](s1_w152_d123_global_alignment_40h_plan.md)
- 사전 검토·구현 보고 + 실행 후 지적 9건 조치: [2026-09-04_global-alignment-plan-review.md](2026-09-04_global-alignment-plan-review.md)
- 진행 상태·중간 표: [results_log/2026-09-04_WIP_global-alignment.md](../results_log/2026-09-04_WIP_global-alignment.md)
- 코드: `align/` · `train_align.py` · `feeders/feeder_align.py` · `tools/{build_shift_cache,pretrain_shiftnet,test_alignment,align_infer_sweep,align_diag}.py` · `config/GA_*.yaml`

판정 규칙: **best checkpoint 의 공식 HQNR(FR 12-19, 장면별 (1−D_λ)(1−D_s) 평균) → fSCC(12-19)**.
ERGAS·SAM 은 참고 지표로만 표기한다. "동급" 밴드는 기존 0.011 이 이 지표에서 측정된 값이 아님이 확인돼
(2026-09-04 문서 §2) 실측 상한 ~0.0027 을 함께 쓴다.

작성 시점: 9벌 중 7벌 종료(C2 α0.75 는 ep105 진행 중, C4 2벌은 gate 로 미학습). α0.75 완주 시 §5 표만 갱신한다.

---

## 0. 한 페이지 요약

| 계획의 가설 | 결과 |
|---|---|
| H1. `lms` 는 phase-2 bicubic 이며 기존 bicubic(phase 1.5)의 0.5px 오차를 고치면(P0) 이득이 있다 | **커널이 틀렸다** — `lms` 는 DLPan `interp23tap`. 고쳐서 돌린 P0 는 anchor 대비 **−0.0003(무효과)**. HQNR 은 이 0.5px 위상에 둔감하다 |
| H2. PAN 과 LRMS 사이에 전역 sub-pixel shift 가 있고, 이를 추정해 넣으면 정렬이 좋아진다 | shift 는 **FR 에서만 실재**(12-19 ≈ 0.9 HR px). train 16² patch 의 추정값은 **추정기 노이즈**(오차 0.27 px > shift 0.06) — 학습 시 "정렬" 은 실제로는 sd 0.076 LR px 의 무작위 jitter 였다 |
| H3. 정렬된 MS 를 내부 처리하고 최종 출력을 M-frame 으로 되돌리면(C1) 이득이 남는다 | **붕괴.** best 가 epoch 5, plateau 0.916. inverse warp 의 저역통과를 네트워크가 과선명화로 보상(내부 P-frame fSCC 0.963, D_s 0.076) |
| H4. P-frame 출력 + M-frame GT loss(C3)가 좌표 충돌을 줄인다 | fSCC 는 최고급(0.901)이지만 공식 HQNR 0.9245. **같은 모델의 두 뷰에서 D_λ 0.0399(P) vs 0.0220(M)** — 공식 지표가 M-frame `lms` 를 기준 삼는 좌표 충돌 0.018 을 정량화. 이 지표로는 P-frame 출력이 이길 수 없다 |
| H5. 조건 입력에만 부분 shift α 를 주면(C2) HQNR–fSCC 절충점이 있다 | **유일한 양성.** α0.5/1.0 best 0.9553(+0.001, 밴드 안), 그러나 **plateau/final +0.006~0.009** 로 후반 D_s 붕괴가 사라졌다. 단, 추론 시 α 를 0 으로 둬도 HQNR 이 같다(0.9555/0.9560) → **이득은 학습 중 조건 입력 jitter 의 정규화 효과이지 정렬이 아니다.** 추론 시 정렬은 fSCC 만 +0.01 |
| H6. 작은 trainable ShiftNet 이 외부 추정기를 재현한다(C4) | pseudo-label 이 노이즈라 pretrain gate(sign ≥95%) 가 **구조적으로 불통과**(0.678). 계획 §15.3 규칙대로 C4 2벌 미학습 |

캠페인 실행 자체의 이슈: 기동 3시간 뒤 **LR 증강 × phase-2 오정렬(치명)** 이 발견돼 P0 를 무효화하고
9건을 수정해 재기동했다 (§3).

---

## 1. 계획 대비 구현이 달라진 곳 (실행 전)

### 1.1 upsampler — phase-2 bicubic 이 아니라 interp23tap (계획 §4)

계획 §4.5 의 phase gate 는 "phase-corrected upsample vs dataset `lms` ZNCC ≥ 0.9999" 였다. 실측:

| split | `F.interpolate` bicubic(기존, phase 1.5) | phase-2 bicubic(계획 원안) | `interp23tap` |
|---|---:|---:|---:|
| train | ZNCC 0.9828 · MAD 17.8 DN | 0.9906 · 9.3 | **1.000000 · 0.000** |
| RR | 0.9926 · 7.7 | 0.9955 · 3.6 | **1.000000 · 0.000** |
| FR | 0.9937 · 9.2 | 0.9986 · 3.3 | **1.000000 · 0.000** |

- `lms` 는 DLPan `wald_utilities.interp23tap`(23-tap CDF 보간, 2단 zero-stuffing) 이다. LR 샘플 j 가
  HR 4j+2 에 놓이는 **phase 는 계획이 맞았고, 커널이 bicubic 이 아니었다.** bicubic 은 어떤 phase 로도
  gate 를 못 넘어 계획대로면 run 을 시작할 수 없었다.
- 구현: `align/resample.py::interp23tap` (torch 이식, float64 로 1e-13 이내 일치 — T02). shift 는 그 위에
  HR bicubic warp 를 **한 번** 건다(`warp_hr`, 4·α·Δ). α·Δ 가 상수 0 이면 warp 를 호출하지 않아 P0 와
  비트 동일(T06). 계획 §4.3 "warp 후 재 upsampling 금지" 의 취지(이중 보간 blur)는 §20 round-trip control
  로 쟀다: FR 12-19 PSNR 61.9 dB · grad-energy 비 1.028 · MAD 0.001 — 무시할 수준.
- 계획 원안은 `alignment.upsampler: bicubic_phase2`(+`phase`) 로 남겼다. phase 1.5 로 두면 기존
  `F.interpolate` 와 3e-12 이내 동일 — 이것으로 기존 checkpoint 에 추론 sweep 을 걸 수 있었다 (§4.1).

### 1.2 shift 의 실체 — train/RR 은 노이즈, FR 만 실재 (계획 §5)

캐시(`tools/build_shift_cache.py`, GT 미사용, lpan=MTF↓PAN phase 2 · ms):

| split | n | accepted | \|δ\| p50 / p90 (LR px) | dy mean ± sd | sign(dy)>0 |
|---|---:|---:|---|---|---:|
| train (16² patch) | 9714 | 92.5% | 0.091 / 0.212 | +0.013 ± 0.187 | **55%** |
| RR (64²) | 20 | 100% | 0.064 / 0.096 | +0.041 ± 0.041 | 75% |
| FR (128²) | 20 | 60% | **0.335 / 0.487** | +0.151 ± 0.260 | 60% |

추정기(Scharr+MAD+top30%+ZNCC ±2 정수+3×3 quadratic / Census5+Hamming) 의 자체 정확도(합성, 알려진 shift):
128² p50 **0.044** px · 64² 0.069 · **16² 0.271**. 즉 train patch 에서는 추정 오차가 shift 보다 크다.
92.5% "accepted" 는 계획 §5.2 게이트가 노이즈를 걸러내지 못한다는 뜻이다.

- FR 판정 subset 12-19 는 8/8 accepted, Δ ≈ (−0.16, +0.18) LR px ≈ 0.9 HR px 로 일관. 0-11 은 8/12 기각인데
  이유가 전부 **peak margin < 0.05** — |δ| 가 0.40~0.45 면 정수 격자 0 과 +1 의 ZNCC 가 비슷해져 margin 이
  작아진다. **계획의 이 게이트는 shift 가 큰 장면을 골라서 버린다.** 판정 subset 은 무관해 그대로 뒀다.
- 결과적으로 frozen-cache case 의 학습 시 Δ 는 적용값 기준 sd 0.076 LR px(≈0.30 HR px) 의 **무작위 jitter**
  (학습 로그 EMA: dy +0.010±0.076, |d| p50 0.081 — 전 run 동일). 의미 있는 Δ 는 추론(FR)에만 들어간다.
  이것이 §4.3 의 "jitter vs 정렬" 분리 실험이 필요했던 이유다.

### 1.3 C4 pseudo-label pretrain gate — 구조적으로 불통과 (계획 §15.3)

계획 규칙 그대로(accepted train Δ 라벨, 2,000 step, SmoothL1 β 0.05):

| 조건 | 기준 | 실측 |
|---|---|---:|
| val median 오차 | ≤ 0.10 | 0.075 ✓ |
| P90 오차 | ≤ 0.25 | 0.147 ✓ |
| **sign accuracy** | ≥ 0.95 | **0.678** ✗ (|target|≥0.15 만: 0.78) |
| (참고) corr dy / dx | — | +0.54 / +0.46 |

라벨 부호가 55:45 동전이라 95% 는 도달 불가. 계획 규칙("재실패 시 C4 취소")을 코드가 집행한다 —
`AlignTrainer` 가 시작 시 gate json 을 읽고 실패면 rc=4 로 종료, 러너가 "사전 gate 불통과" 로 원장에 기록.
합성 라벨 옵션(`--synthetic`)도 구현했으나 검증 라벨이 cache 라 역시 FAIL.

### 1.4 그 밖의 설계 차이

| 항목 | 계획 | 구현 | 이유 |
|---|---|---|---|
| cache 형식 | parquet | CSV + `cache_meta.json`(SHA256, 게이트, 지문) | pyarrow 부재. 열은 동일 + `source_pan_file_hash` |
| 정수 탐색 | ±1 | ±2 (boundary=±2), 크기 게이트 ≤1.0 | ±1 이면 |δ|>0.5 를 quadratic 이 못 잡는다 |
| 평가 격자 | 10K…50K 9점 | eval_epoch 5 → 49점 + **종료 시 50K 1회 추가** | 더 촘촘·기존 14벌과 동일 조건, 50K 도 후보 |
| 11ch 순서 | (MS, PAN, LPAN, HF) | (PAN, LPAN, PAN−LPAN, MS) | backbone 입력 conv 가 기대하는 순서 |
| HQNR 집계 | mean((1−D_λᵢ)(1−D_sᵢ)) | 처음엔 (1−mean D_λ)(1−mean D_s) → **계획식으로 통일** | 차 1.27e-5 = tie band 13% (§3.2) |
| best 선택 | HQNR(1e-4)→fSCC(1e-4)→나중 iteration | 동일 + tie 기준을 **running max** 에 고정 | 동률 교체 누적 하락 방지 (§3.3) |
| 코어 U-Net | 수정 없음 | 수정 없음 + `forward(x_in=)` 우회 입구 | 기본 경로 비트 동일 확인 |
| C4 gate 실패 | 수정 후 1회 재시도 | rc=4 즉시 종료 | 자동 "수정" 은 없다 |

---

## 2. 학습 없이 먼저 본 것 — 기존 checkpoint 추론 sweep

`tools/align_infer_sweep.py` 로 anchor(`S1_T05_W152_D123_DUAL`, bicubic phase 1.5 학습) 에 추론 시 cache Δ 를 적용:

| 추론 case | HQNR | D_λ | D_s | fSCC |
|---|---:|---:|---:|---:|
| 없음 (=anchor 재현) | **0.9546** | 0.0227 | 0.0232 | 0.885 |
| 조건 입력 α=0.5 | 0.9527 | 0.0226 | 0.0252 | 0.896 |
| α=1.0 | 0.9503 | 0.0231 | 0.0272 | 0.906 |
| round-trip(C1형) | 0.9279 | 0.0207 | 0.0525 | 0.800 |
| dual-frame(C3형) | 0.9231 | 0.0394 | 0.0390 | 0.915 |

이때의 예상: "정렬할수록 fSCC↑ HQNR↓(D_s), 이 캠페인이 HQNR 을 올릴 가능성은 낮다". **이 예상은 학습 후
C2 에서 뒤집혔다** (§4.2). shift 없이 학습한 모델에 shift 를 넣는 것(OOD)과 shift 로 학습하는 것은 다르다.

---

## 3. 실행 중 발견된 이슈 (기동 3시간 뒤 검증 지적 9건)

### 3.1 치명 — LR 증강 × phase-2 오정렬

- 원본 feeder 는 LR(ms, lpan)과 HR(gt, lms, pan) 을 각각 flip/rot 한다. 그 LR 을 phase-2 interp23tap 으로
  올리면 HR 을 flip 한 것과 **축마다 1 HR px 어긋난다**: LR j→HR 4j+2 를 flip 하면 62−4j, HR flip 은 61−4j.
  원본 코드의 bicubic(phase 1.5)은 flip 대칭이라 없던 문제 — **P0 가 phase 를 "고친" 것이 이 버그를 만들었다.**
- 재현: hflip+vflip 고정(모든 config) + rot 0/1/3 에서 MAD 22.7 / 17.0 / 14.8 DN, 1px roll 로 0.000. rot 2 만 정상.
  **표본 75%** 오정렬, 오차 크기가 후반 MS loss(0.018) 와 같은 급.
- 조치: feeder 는 **LR 을 원본 격자로** 주고 HR 만 증강 → wrapper 가 upsample(+shift, 원본 frame Δ) **뒤 HR 에서**
  같은 flip/rot(`augment_hr`) → inverse/mask 는 증강된 출력 frame 에서 `transform_delta(Δ)`. ShiftNet 입력·
  pseudo-label 은 원본 frame 이라 변환이 사라진다.
- 검증: T11(실데이터 16 증강 조합에서 `ms_base_hr` ≡ 증강 `lms`, 2.5e-8; 옛 경로 ~2e-2 대조) · T12(shift→증강 ==
  증강→shift(변환 Δ), 1e-9). 기존 T02(무증강 재현)·T05(동일 해상도 벡터 변환)로는 못 잡던 것.
- 처리: 00:30 체인 중단, 3h 학습한 P0 를 `work_dir/_INVALID_augphase_GA_P0_…` 로 격리, 00:42 재기동.
  통합 210 iter 에서 수정본 P0 0.8744 vs 오염본 0.8687, 같은 step 의 MS loss 0.0381 vs 0.0390.

### 3.2 HQNR 집계식 불일치
trainer 는 (1−mean D_λ)(1−mean D_s), 사후 평가기·계획은 mean((1−D_λᵢ)(1−D_sᵢ)). S1_T05_W152 best mat 에서
0.9546072 vs 0.9546199, 차 1.27e-5 = tie band 의 13% → fSCC tie-break 결과를 바꿀 수 있다. 두 trainer 모두
계획식으로 통일, 로그에 두 값 병기. 과거 14벌 `best_state` 값은 옛 식(차 ~1e-5, 판정 불변).

### 3.3 tie band 기준 누적 하락
기준이 "현재 best 의 HQNR" 이면 동률 교체가 반복될 때 기준이 내려가 최대값보다 1e-4 넘게 낮은 checkpoint 도
best 가 될 수 있다. 기준을 running max 로 고정, `best_state.json` 에 `max_hqnr` 저장·resume 복구.

### 3.4 정확한 50K 미평가
마지막 eval 이 ep245(49,490 step)라 50K 모델은 후보에 못 들었다. 마지막 epoch 이 격자 밖이면 종료 시 1회 추가 평가.
(이번 캠페인 best 는 전부 245 이전이라 결과엔 영향 없음.)

### 3.5 나머지
- resume 이 config/commit/cache 를 확인하지 않고 `run.sh` 가 meta 를 덮어씀 → `meta/prev_<시각>/` 보존 + `AlignTrainer` 가
  cache SHA·upsampler 불일치 시 rc=4 거부.
- cache builder 가 추정기 입력 `*_pan.h5` 를 해시하지 않음 → `source_pan_file_hash` 추가, 캐시 재생성.
- 큐 drop 순서가 주석뿐, deadline 은 case 시작 전에만 검사 → 큐 순서 = 계획 §10.1 우선순위.
- C4 gate 종료(exit 3)가 러너에 "NaN 중단" 으로 기록 → rc=4 분리.
- 실배치 OOM smoke 가 bare backbone → wrapper 로 실배치(48×2) forward+backward+AdamW.

---

## 4. 결과

### 4.1 본 표 (best checkpoint · 공식 12-19)

| run | 세팅 | **HQNR** | ep | fSCC | D_λ | D_s | plateau(≥100) | final(245) | 시간 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor `S1_T05_W152_D123_DUAL` | bicubic phase1.5 · shift 없음 | 0.9546 | 110 | 0.885 | 0.0227 | 0.0232 | 0.9516 | 0.9502 | 3.18h |
| P0 | interp23tap · shift 없음 | 0.9543 | 80 | 0.886 | 0.0206 | 0.0257 | 0.9492 | 0.9465 | 3.26h |
| C1 frozen round-trip | cache α1 · P 내부 · M 출력(inverse) | 0.9383 | **5** | 0.856 | 0.0249 | 0.0378 | 0.9159 | 0.9124 | 3.35h |
| C3 frozen dual-frame | cache α1 · **P 출력** · GT loss 만 inverse | 0.9245 | 235 | 0.901 | 0.0399† | 0.0371 | 0.9224 | 0.9241 | 3.35h |
| C2 α0.25 | 조건 입력만 shift | 0.9534 | 145 | 0.887 | 0.0192 | 0.0279 | 0.9518 | 0.9514 | 3.28h |
| C2 α0.50 | | **0.9553** | 120 | 0.892 | 0.0205 | 0.0248 | 0.9529 | 0.9525 | 3.28h |
| C2 α0.75 | | (ep105 진행: 0.9513, D_s 0.029↓) | | | | | | | |
| C2 α1.00 | | **0.9553** | 225 | 0.900 | 0.0213 | 0.0240 | 0.9528 | **0.9552** | 3.28h |
| C4 dual-frame / RT | trainable ShiftNet | gate FAIL → 미학습 | | | | | | | |

† P-frame 출력을 M-frame `lms` 와 비교한 값. 같은 모델의 M-frame(inverse) 뷰: HQNR 0.9160 · D_λ 0.0220 · D_s 0.0635 · fSCC 0.789.
C1 의 내부 P-frame 뷰: HQNR 0.8835 · D_λ 0.0439 · **D_s 0.0762 · fSCC 0.963**.

장면별 HQNR(12-19), P0 대비 차이:

| | s12 | s13 | s14 | s15 | s16 | s17 | s18 | s19 | 개선 장면 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 − anchor | +.001 | −.002 | −.000 | −.000 | −.001 | −.001 | −.000 | +.001 | 2/8 |
| C2 .25 − P0 | +.006 | −.002 | −.002 | −.002 | −.001 | −.002 | −.001 | −.002 | 1/8 |
| C2 .50 − P0 | +.001 | +.001 | +.001 | +.002 | +.001 | −.000 | +.001 | +.001 | **7/8** |
| C2 1.0 − P0 | −.008 | +.003 | +.002 | +.003 | +.003 | −.000 | +.000 | +.005 | 6/8 |
| C1 − P0 | −.014 | −.012 | −.011 | −.015 | −.010 | −.013 | −.023 | −.028 | 0/8 |
| C3(P) − P0 | −.036 | −.027 | −.014 | −.024 | −.021 | −.036 | −.050 | −.030 | 0/8 |

곡선(공통 epoch, HQNR):

| epoch | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 245 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor | .9395 | .9506 | .9499 | .9516 | .9530 | .9515 | .9517 | .9508 | .9505 | .9502 |
| P0 | .9435 | .9489 | .9521 | .9493 | .9521 | .9506 | .9494 | .9471 | .9467 | .9465 |
| C2 .25 | .9377 | .9451 | .9487 | .9515 | .9519 | .9527 | .9520 | .9510 | .9516 | .9514 |
| C2 .50 | .9355 | .9468 | .9467 | .9501 | .9520 | .9541 | .9533 | .9524 | .9526 | .9525 |
| C2 1.0 | .9270 | .9373 | .9438 | .9481 | .9484 | .9530 | .9534 | .9542 | **.9553** | .9552 |
| C1 | .9243 | .9225 | .9307 | .9144 | .9186 | .9191 | .9121 | .9159 | .9125 | .9124 |
| C3 | .9120 | .9197 | .9168 | .9174 | .9223 | .9233 | .9244 | .9238 | .9238 | .9241 |

공식 D_s 곡선: P0 는 ep75 0.0277 → ep245 0.0353 으로 **상승**(anchor·이전 14벌과 같은 후반 붕괴), C2 1.0 은
0.0508 → 0.0241 로 **단조 감소**. C1 은 0.054 → 0.069.

### 4.2 결정 실험 — C2 의 이득은 어디서 오는가

완료 checkpoint 에 **추론 시 α 만** 바꿔 잰 값 (`align_infer_sweep`, base interp23tap):

| 학습 checkpoint | 추론 α=0 | α=0.5 | α=1.0 |
|---|---:|---:|---:|
| C2 α1.0 | **0.9555** / fSCC 0.891 / ERGAS 2.076 | 0.9551 / 0.897 / 2.072 | 0.9553 / 0.900 / 2.072 |
| C2 α0.5 | **0.9560** / 0.883 / 2.095 | 0.9552 / 0.892 / 2.091 | 0.9558 / 0.897 / 2.095 |
| P0 | 0.9543 / 0.886 / 2.112 | 0.9524 / 0.899 / 2.214 | 0.9521 / 0.908 / 2.577 |

- C2 모델은 **추론 시 shift 를 빼도 HQNR 이 같거나 높다.** HQNR 이득은 학습 중 조건 입력의 무작위 sub-pixel
  jitter(§1.2)가 후반 D_s 드리프트를 억제한 **정규화 효과**다. 추론 시 정렬은 HQNR 에 중립·fSCC 에 +0.01.
- shift 없이 학습한 P0 에 추론 시 shift 를 주면 HQNR 이 내려간다(§2 의 anchor 와 같은 현상). C2 모델은
  추론 shift 에 강건하다(ERGAS 도 불변 — 참고).
- 따라서 계획의 Q4("부분 shift 의 절충점")는 α 의 문제가 아니라 **jitter 유무**의 문제로 답이 바뀐다.

### 4.3 각 case 의 기전 해석

- **P0**: 입력·잔차 base 를 `lms` 와 정확히 맞추면 D_λ 는 좋아지고(0.0227→0.0206) D_s 는 나빠져(0.0232→0.0257)
  상쇄된다. 학습 loss 는 동일(0.01812 vs 0.01819). HQNR 은 이 0.5px 위상 차이에 둔감하다. 기존
  `residual_base: lms` 기록(RR ERGAS 2.5% 이득)이 HQNR 로는 옮겨오지 않았다.
- **C1**: 최종 출력을 −4Δ 로 되돌리는 bicubic warp 가 저역통과다. GT 를 warp 뒤에 맞추려면 그 앞단(P-frame 출력)이
  역필터처럼 과선명화된다 — 내부 뷰 fSCC 0.963(정상 0.886), D_s 0.076. warp 뒤에도 D_s 0.038~0.069 가 남는다.
  학습할수록 심해져 best 가 epoch 5. §20 이 잰 보간 blur 자체(62 dB)는 작았지만 학습이 그것을 증폭했다.
  또 border mask 가 |Δ|≈0.08 에도 margin 3px 를 잘라 MS loss 화소의 17.5% 를 버린다(계획 §7.2 식의 +2 상수).
- **C3**: P-frame 출력은 PAN 에 잘 맞아 fSCC 0.901 이지만, 공식 D_λ 는 MTF↓(출력) 을 M-frame `lms` 와 비교하므로
  Δ 만큼 어긋난 채 계산된다 → 0.0399. 같은 모델의 inverse 뷰는 D_λ 0.0220 이나 이번엔 D_s 가 0.0635(inverse blur).
  **어느 frame 으로 봐도 P0 에 못 미친다** — 공식 지표는 M-frame 을 기준으로 삼는다.
- **C2**: 조건 입력만 흔들고 base·GT 는 M-frame 에 둔다. 네트워크는 "MS 조건이 약간 어긋나도 M-frame 출력을 내는"
  불변성을 배우고, 그것이 후반 D_s 상승(=PAN 구조를 과하게 주입하는 경향)을 억제한다. α 는 jitter 진폭(0.076·α LR px)일
  뿐이며 α0.5 와 1.0 이 같다.

### 4.4 계획의 질문 §2 에 대한 답

1. sampling phase 만 바로잡으면 달라지는가 — **아니오** (P0 −0.0003).
2. PAN frame 내부 처리 후 M-frame 복귀(C1)에 이득이 남는가 — **아니오, 크게 해롭다** (−0.016, 과선명화).
3. P-frame 출력 + M-frame GT loss(C3)가 나은가 — **공식 HQNR 로는 아니오** (좌표 충돌 0.018 을 정량화). fSCC 는 최고급.
4. 부분 shift 의 절충점이 있는가 — **절충이 아니라 jitter 정규화**. α0.5~1.0 에서 HQNR·fSCC 동반 상승, 후반 붕괴 소멸.
5. trainable estimator 가 외부 추정기를 재현하는가 — **검증 불가** (라벨이 노이즈, gate 불통과).

---

## 5. 운영 기록

- 1차 기동 09-04 21:27 → 09-05 00:30 중단(§3.1) → 00:42 재기동. run 당 3.18~3.35h, 7벌 ≈ 23h.
- 시트: GA 범주 구분행(⑫) + 캠페인 설명이 첫 업로드에 삽입됐고, 각 행 Notes 에 case 질문·세팅·판정·anchor 서술.
- 산출물: `work_dir/GA_*/{best_hqnr_meta.json, metrics.csv(공식 D_λ/D_s·fSCC·alt 열), results/*_msframe|panframe.mat}`,
  추론 sweep CSV(`work_dir/S1_T05_W152_D123_DUAL/results/infer_sweep.csv`, scratchpad 의 C2/P0 sweep).
- 무효본: `work_dir/_INVALID_augphase_GA_P0_PHASEFIX_W152_D123_DUAL` (참고용, 시트 미반영).

## 6. 남는 것과 다음

- **C2 의 효과가 캐시 없이 재현되는지** — 무작위 sub-pixel jitter(sd ≈ 0.08 LR px, 등방)만으로 같은 곡선이 나오면
  추정기·캐시·정렬은 전부 불필요하고 "jitter augmentation" 하나로 정리된다. 1벌(3.3h)로 판정 가능.
- jitter 가 **다른 backbone(W96 student, W168·d123 teacher)** 에서도 후반 D_s 붕괴를 막는지 — 이전 14벌의 공통 병이었다.
- 밴드: best 차 +0.001 은 어떤 밴드로도 동급이다. 주장할 수 있는 것은 plateau/final(+0.006~0.009, 7/8 장면)과
  곡선 형태다. 공식 HQNR 의 시드 반복이 여전히 없다.
- C4 를 살리려면 라벨을 바꿔야 한다(FR 급 크기의 합성 shift, 또는 fSCC 를 직접 최대화). 이번 캠페인 범위 밖.

---

## 7. 객관성 검증 (2026-09-05 22:40) — P0 의 변경이 C 계열 평가를 오염시켰는가

질문: P0 에서 baseline 파이프라인을 바꿨고(커널 교체) 그 과정에 구현 오류(증강-phase 버그)가 있었다.
그 변경이 C1/C2/C3 의 결과에 스며들어 비교가 객관적이지 않을 가능성.

### 7.1 버그는 새지 않았다 (provenance)

| run | 시작 commit | 시작 시각 | 수정(9b324b4) 포함 |
|---|---|---|---|
| 1차 P0 (격리, 시트 미반영) | db22f14 | 09-04 21:27 | **이전** |
| P0 (재기동) | 9b324b4 | 09-05 00:42 | 포함 |
| C1 · C3 · C2 ×4 · C4(gate) | a17aea3 | 09-05 03:58 ~ 20:33 | 포함 |

resume 흔적 없음(전부 처음부터). 시트에 올라간 GA 행은 모두 수정 이후 실행이다.

### 7.2 P0 는 원 baseline 과 "커널 하나" 만 다르다 (등가 실험)

원 `Trainer`(S1_T05 config) 와 `AlignTrainer`(P0 config + `upsampler: bicubic_phase2, phase: 1.5` = 옛 `F.interpolate` 와
3e-12 이내 동일) 를 같은 seed·`num_worker=0`(데이터 순서·증강 난수 고정)으로 210 iter 돌렸다.

| iter | 원 Trainer loss | AlignTrainer+옛 커널 | 차이 |
|---:|---:|---:|---:|
| 10 | 0.114551 | 0.114551 | **0** |
| 50 | 0.079634 | 0.079619 | 1.5e-5 |
| 100 | 0.053489 | 0.053463 | 2.7e-5 |
| 200 | 0.046482 | 0.046318 | 1.6e-4 |
| ep1/ep2 HQNR | 0.86552 / 0.86551 | 0.86580 / 0.86584 | 2.8e-4 / 3.3e-4 |

iter 10 까지 완전히 같고 이후 부동소수 비결정성(cuDNN, 커널 구현 3e-12 차) 만큼 벌어진다. 즉 AlignTrainer·
feeder_align·HR 증강·masked L1(마스크 1)·PAN mode 처리는 원 코드와 **동작이 같고**, P0 의 실질 변경은
interp23tap 커널뿐이다. (부수 정보: 같은 설정의 두 run 이 210 iter 뒤 HQNR 3e-4 차이 — 1e-3 이하 차이는 애초에
신뢰할 수 없다.)

### 7.3 그러나 커널 교체는 기준선 자체를 움직였다 — C2 이득은 P0 기준으로 부풀려 보인다

| | best | plateau(≥100) | last50(200-245) | final(245) |
|---|---:|---:|---:|---:|
| anchor (원 bicubic 1.5) | 0.9546 | 0.9516 | 0.9505 | 0.9502 |
| P0 (interp23tap) | 0.9543 | **0.9492** | 0.9468 | **0.9465** |
| C2 α0.5 | 0.9553 | 0.9529 | 0.9526 | 0.9525 |
| C2 α1.0 | 0.9554 | 0.9528 | 0.9549 | 0.9552 |

| C2 이득 | vs **원 anchor** | vs P0 |
|---|---|---|
| α0.5 | best +0.0006 · plateau **+0.0012** · final +0.0024 | best +0.0010 · plateau +0.0036 · final +0.0061 |
| α1.0 | best +0.0008 · plateau **+0.0011** · final **+0.0050** | best +0.0011 · plateau +0.0036 · final +0.0087 |

P0 커널 교체가 후반 plateau 를 −0.0024(final −0.0037) 내려놓았고, 이것이 C2-vs-P0 이득의 약 2/3 를 차지한다.
**원 anchor 기준에서 C2 의 이득은 plateau +0.001 수준으로 노이즈 상한(0.0027) 안이며**, 남는 증거는
α1.0 의 final +0.005 와 "후반 하락이 없는 곡선 형태" 다. 계획이 P0 를 anchor 로 삼으라 했지만, 객관적 비교 기준은
원 baseline 이어야 한다 — 이 문서의 §4 수치는 P0 기준이므로 그만큼 할인해 읽어야 한다.

### 7.4 남은 확인 — 커널 교체 없이 C2 를 돌리면?

`GA_CTRL_C2_BICUBIC15_A100_W152_D123_DUAL`: 원 커널(bicubic phase 1.5) 위에 C2 α1 jitter 만 얹은 대조 run.
현재 체인 DONE 직후 자동 기동(≈ 09-06 00:00, 3.3h). 판정:
- 이 run 의 plateau/final 이 anchor(0.9516/0.9502) 를 C2(interp23tap) 만큼 넘으면 → jitter 효과는 커널과 독립.
- anchor 와 같으면 → C2 의 "이득" 은 interp23tap 이 만든 후반 하락을 jitter 가 되돌린 것일 뿐, 원 baseline 대비 이득이 아니다.
