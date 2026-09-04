# Global alignment 40h 계획 — 검토와 구현 보고 (2026-09-04)

대상: [s1_w152_d123_global_alignment_40h_plan.md](s1_w152_d123_global_alignment_40h_plan.md)
구현: `align/` · `train_align.py` · `feeders/feeder_align.py` · `tools/{build_shift_cache,pretrain_shiftnet,test_alignment,align_infer_sweep,align_diag}.py` · `config/GA_*.yaml` 9벌 · `config/queues/s1_global_alignment.txt`
판정: **best checkpoint HQNR(공식 12-19) → fSCC(12-19)**. ERGAS·SAM 은 참고 지표.

---

## 0. 요약

- **구현은 끝났고 검증을 통과했다.** 단위 검사 T01–T10 10/10, smoke 9/9, 실제 trainer 통합 실행
  (P0·C3·C4B, 210 iter — 학습·평가·best 선택·mat 내보내기) 완주.
- **계획의 전제 두 가지가 데이터와 어긋나 설계를 바꿨다** (§1). 어느 쪽도 계획의 질문을 바꾸지는
  않지만, 그대로 구현했다면 계획 자신의 gate 에 막혀 한 run 도 시작하지 못했을 것이다.
- **학습 없이 이미 답이 하나 나왔다** (§3): 기존 W152·d123 checkpoint 에 추론 시 전역 shift 를
  넣으면 **fSCC 는 단조 상승, HQNR 은 단조 하락**한다. 하락은 전부 D_s 다.
- **실행 전에 사용자가 정할 것 3가지** (§5): C4 처리(pretrain gate 가 구조적으로 실패), 큐
  규모, P0 결과를 보고 C 계열을 계속할지의 2단계 진행 여부.

---

## 1. 계획 전제 검증 — 두 가지가 틀렸다

### 1.1 `lms` 는 phase-2 bicubic 이 아니라 `interp23tap` 이다

계획 §4 는 "PanCollection 의 MS 는 4배 격자 phase 2 중심, bicubic 으로 재현" 을 전제로 phase gate
(ZNCC ≥ 0.9999 vs 제공 `lms`)를 뒀다. 실측:

| split | `F.interpolate` bicubic (기존 코드, phase 1.5) | phase-2 bicubic (계획 원안) | **DLPan `interp23tap`** |
|---|---:|---:|---:|
| train | ZNCC 0.9828 · MAD 17.8 | 0.9906 · 9.3 | **1.000000 · 0.000** |
| RR | 0.9926 · 7.7 | 0.9955 · 3.6 | **1.000000 · 0.000** |
| FR | 0.9937 · 9.2 | 0.9986 · 3.3 | **1.000000 · 0.000** |

**phase 는 계획이 맞고(LR j → HR 4j+2), 커널이 틀렸다.** bicubic 은 어떤 phase 로도 gate 를 못 넘는다.
그래서:

- **기본 upsampler = `interp23tap`** (torch 이식, float64 로 1e-13 이내 일치 — T02). shift 는 그 위에
  HR bicubic warp 를 **한 번** 건다 (`warp_hr`, 4·α·Δ). α·Δ 가 상수 0 이면 warp 를 호출하지 않아
  **P0 와 비트 동일**하다 (T06).
- 계획 §4.3 "warp 후 다시 upsampling 금지" 는 두 단계 blur 를 막으려는 것이다. 우리 경로는
  정확 upsample → HR warp 1회이고, blur 는 §20 round-trip control 로 쟀다:
  **FR 12-19 PSNR 61.9 dB · grad-energy 비 1.028 · MAD 0.001([-1,1] 단위)** — 무시할 수준.
- 계획 원안은 `alignment.upsampler: bicubic_phase2` 로 남겨 뒀다 (phase 인자 포함. phase 1.5 로
  두면 기존 `F.interpolate` 와 3e-12 이내 동일 — 이것으로 기존 checkpoint 에 sweep 을 걸 수 있다, §3).
- 기존 코드의 sampling-phase 오차(ZNCC 0.983)는 **실재한다**. `residual_base: lms` 옵션이 잔차
  기준선만 바꿔도 2.5% 낫다는 기존 기록과 정합한다. **P0 는 입력 conditioning 까지 고친다.**

### 1.2 전역 shift 는 FR 에서만 실재한다 — train/RR 은 추정기 노이즈

캐시(`tools/build_shift_cache.py`, GT 미사용) 통계:

| split | n | accepted | \|δ\| p50 / p90 (LR px) | dy mean±sd | sign(dy)>0 |
|---|---:|---:|---|---|---:|
| train (16² patch) | 9714 | **92.5%** | 0.091 / 0.212 | +0.013 ± 0.187 | **55%** |
| RR (64² scene) | 20 | 100% | 0.064 / 0.096 | +0.041 ± 0.041 | 75% |
| FR (128² scene) | 20 | **60%** | **0.335 / 0.487** | +0.151 ± 0.260 | 60% |

추정기 자체의 정확도(합성 영상, 알려진 shift): 128² p50 **0.044** px · 64² 0.069 · **16² 0.271**.
즉 train patch 에서는 **추정 오차(0.27)가 shift(0.06–0.09)보다 크다.** 92.5% 가 accepted 인 것은
게이트가 노이즈를 못 거른다는 뜻이지 shift 가 있다는 뜻이 아니다.

FR 은 다르다. **판정 subset 12-19 는 8/8 accepted, Δ ≈ (−0.16, +0.18) LR px ≈ 0.9 HR px 로 일관**된다.
기각된 8 scene 은 전부 0–11 이고 이유가 **peak margin < 0.05** 다 — |δ| 가 0.40–0.45 면 정수 격자 0 과
+1 의 ZNCC 가 비슷해져 margin 이 작아진다. **계획의 이 게이트는 shift 가 큰 scene 을 골라서 버린다.**
판정 subset 에는 영향이 없어 캐시는 계획대로 뒀고, 규칙을 바꾸려면 새 캠페인에서 한다.

**결과적으로:**
- frozen-cache case(C1·C2·C3)의 **학습 시 정렬은 sd 0.076 LR px(≈0.30 HR px) 의 sub-pixel jitter** 이고
  (적용값 기준 — 원시 sd 0.187 은 기각된 7.5% 의 큰 |δ| 가 부풀린 것. 학습 로그 EMA 0.075 와 일치),
  의미 있는 Δ 는 **추론 시(FR)에만** 들어간다. "정렬을 학습한다" 기보다 "정렬된 입력에 대한 강건성을
  jitter 로 익히고, 추론 때 실제 정렬을 준다" 가 정직한 서술이다.
- C4 의 pseudo-label 은 노이즈다. §4 참고.

---

## 2. 구현

### 2.1 구조 (코어 U-Net 무수정)

```
feeders/feeder_align.py   PanFeeder + meta[sample_id, split, hflip, vflip, rot]  (cache 조회·§6 벡터 변환용)
align/resample.py         interp23tap · phase_shift_upsample · warp_hr · border_mask · masked_l1 · transform_delta
align/estimator.py        Scharr+MAD+top30% mask+ZNCC(±2 정수, 3x3 quadratic) / Census5+Hamming / §5.2 게이트
align/cache.py            CSV cache + cache_meta.json(SHA256) + lookup(split, sample_id)
align/shiftnet.py         GlobalShiftNet(§15.2) + 구조맵 입력
align/model.py            AlignCfg(5 key 검증) + AlignedModel(build_views / residual / finalize_ms)
train_align.py            AlignTrainer — dual 복제·delta 복제(PAN half 는 detach)·masked L1·공식 HQNR+fSCC·alt-frame 뷰·mat 2종
model/pancrafter_paper.py forward(..., x_in=None) — 11ch 를 밖에서 주는 우회 입구. 기본 경로는 비트 동일(검증)
main.py                   --alignment · trainer=align · best 선택 HQNR(1e-4)→fSCC(1e-4)→나중 iteration (§17.2)
```

부호 규약은 `align/resample.py` 한 곳에만 있다: `δ=(dy,dx)`, `aligned[y,x]=moving[y+dy,x+dx]`,
M→P `+Δ`, P→M `−Δ`, `Δ_HR = 4Δ_LR`. `scipy.ndimage.shift` 는 쓰지 않는다.

### 2.2 case ↔ config

| run | delta_source | alpha | output_frame | inverse_location | 잔차 base | 최종 출력 | RR 1차 지표 뷰 |
|---|---|---:|---|---|---|---|---|
| `GA_P0_PHASEFIX_W152_D123_DUAL` | zero | 0 | M | none | M | M | 최종 |
| `GA_C1_FROZEN_RT_A100_…` | cache | 1 | M | final_output | P | M(inverse) | 최종 |
| `GA_C3_FROZEN_DUALFRAME_A100_…` | cache | 1 | P | loss_branch | P | **P** | inverse 뷰 |
| `GA_C2_INPUTONLY_A{025,050,075,100}_…` | cache | α | M | none | **M** | M | 최종 |
| `GA_C4_TRAIN_RT_…` | trainable | 1 | M | final_output | P | M(inverse) | 최종 |
| `GA_C4_TRAIN_DUALFRAME_…` | trainable | 1 | P | loss_branch | P | **P** | inverse 뷰 |

`AlignCfg.validate()` 가 이 조합 외를 거부한다. PAN mode 는 모든 case 에서 inverse 없음·마스크
없음·target = 입력 PAN 복제 (T08 로 inverse 호출 0회 확인).

산출물: `full_best_hqnr.mat` 의 `sr` = **최종(배포) frame**, `reduced_best_hqnr.mat` 의 `sr` = **M-frame 뷰**
(GT 좌표계). full-shift case 는 `*_msframe.mat` / `*_panframe.mat` 로 다른 frame 도 저장한다.
`best_hqnr_meta.json` 에 iteration·hqnr·fscc·d_lambda·d_s·frame·shift_source·alpha·cache_sha256.

### 2.3 계획과 달리 한 것

| 항목 | 계획 | 구현 | 이유 |
|---|---|---|---|
| upsampler | phase-2 bicubic | **interp23tap** + HR warp | §1.1. 원안은 옵션으로 유지 |
| cache 형식 | parquet | CSV | pyarrow 부재. 열은 동일 |
| 정수 탐색 | ±1 | ±2 (boundary = ±2), 크기 게이트 ≤1.0 | ±1 이면 |δ|>0.5 를 quadratic 이 못 잡는다 |
| 평가 격자 | 10K…50K 9점 | eval_epoch 5 → **49점** | 더 촘촘하고 기존 14벌과 동일 조건 |
| 11ch 순서 | (MS, PAN, LPAN, HF) | (PAN, LPAN, PAN−LPAN, MS) | backbone 의 입력 conv 가 기대하는 순서 |
| C4 gate 실패 | 수정 후 1회 재시도 | exit 3 (체인 FAILED, 재시도 없음) | 자동 "수정" 은 없다. 사람이 §5 에서 정한다 |
| 시트 | — | run 명에 `GA:src/aα/frame/inverse`, Notes 에 서술 | 약명 단독 금지 규약 |

### 2.4 검증

- `tools/test_alignment.py` **10/10**: 위상(T01) · lms 재현(T02, 1e-13) · 부호(T03, 추정기 포함) ·
  inverse(T04, 정수 왕복 정확) · augmentation 벡터 변환(T05, 16 조합 impulse) · α=0 ≡ P0(T06, 비트) ·
  α=1 cache 반영(T07) · PAN mode inverse 0회(T08) · mode 복제(T09) · gradient 경로(T10).
- `tools/smoke_cases.py` 9/9 (wrapper forward 학습 형상 dual + FR 512²).
- 통합 실행 210 iter: P0·C3·C4B — `[핵심]` 로그, `fSCC(12-19)`, `[alt frame]`, `metrics.csv` 신규 열
  (`d_lambda_official, d_s_official, fscc_official, hqnr_alt, fscc_alt, …`), tie-break, mat 2종.
- `metrics.csv` 에 **공식 D_λ/D_s 열**을 기본 Trainer 에도 추가했다 — 어제 발견한 "proxy 열은 분해가
  아니다" 공백을 메운다.

---

## 3. 학습 없이 얻은 결과 — 기존 checkpoint 추론 sweep

`tools/align_infer_sweep.py` 로 `S1_T05_W152_D123_DUAL/best_hqnr` (기존 코드 = bicubic phase 1.5 로
학습) 에 추론 시 cache Δ 를 적용했다. α=0 이 기록된 best HQNR **0.9546 을 그대로 재현**한다(도구 검증).

| 추론 case | HQNR | D_λ | D_s | fSCC | RR ERGAS(참고) |
|---|---:|---:|---:|---:|---:|
| 없음 (=기존) | **0.9546** | 0.0227 | **0.0232** | 0.8846 | 2.075 |
| 조건 입력만 α=0.25 | 0.9537 | 0.0226 | 0.0242 | 0.8903 | 2.088 |
| α=0.50 | 0.9527 | 0.0226 | 0.0252 | 0.8959 | 2.135 |
| α=0.75 | 0.9516 | 0.0228 | 0.0262 | 0.9012 | 2.218 |
| α=1.00 | 0.9503 | 0.0231 | 0.0272 | **0.9057** | 2.344 |
| round-trip(C1형, OOD) | 0.9279 | 0.0207 | 0.0525 | 0.8002 | 2.386 |
| dual-frame(C3형, OOD) | 0.9231 | 0.0394 | 0.0390 | 0.9150 | 2.386 |

**조건 입력을 PAN 쪽으로 옮길수록 fSCC 는 오르고 HQNR 은 내린다. 하락은 전부 D_s 다** (D_λ 불변).
D_s 는 `Q(fused, PAN)` 과 `Q(lms, PAN↓)` 의 차이인데 `lms` 가 M-frame 이라, **출력이 PAN 에 더 잘 맞을수록
D_s 는 나빠진다.** 이는 어제 확인한 "D_s 가 HQNR 순위를 가른다(r=−0.78)·full-res depth 가 손해" 와 같은
기전이다. C1/C3 형은 학습 분포 밖(shift 된 base 를 본 적 없음)이라 수치는 참고만.

**함의**: 이 캠페인이 **HQNR 을 올릴 가능성은 낮다.** 학습이 이 추세를 뒤집으려면 네트워크가 정렬된
조건 입력을 받고도 M-frame 스타일 출력을 내야 하는데, 그러면 C2 ≈ P0 가 된다. **P0 는 예외**다 —
phase 수정은 D_s 와 무관하게 GT 정합을 고치므로 이득이 날 수 있다.

---

## 4. ShiftNet pretrain gate — 예측대로 FAIL

계획 §15.3 규칙 그대로 돌렸다 (`tools/pretrain_shiftnet.py`, accepted train Δ 라벨, 2,000 step):

| 조건 | 기준 | 실측 | |
|---|---|---:|---|
| val median 오차 | ≤ 0.10 | 0.075 | 통과 |
| P90 오차 | ≤ 0.25 | 0.147 | 통과 |
| **sign accuracy** | ≥ 0.95 | **0.678** | **실패** (|target|≥0.15 만: 0.78) |
| (참고) corr dy/dx | — | +0.54 / +0.46 | 노이즈 라벨에서도 뭔가는 배운다 |

원인은 §1.2 — 라벨 부호가 55:45 로 동전 던지기라 95% 는 도달 불가. 계획 규칙("재실패 시 C4 취소")을
따르면 **C4 두 run(≈7.4h) 은 취소**다. 구현은 gate 를 그대로 두고 실패 시 exit 3 으로 체인이 FAILED
처리하게 했다. 대안이 필요하면 §5.

---

## 5. 실행 전 결정 사항

1. **C4 처리.** (a) 계획 규칙대로 취소 (권고 — 라벨이 노이즈인데 학습시켜 봐야 해석이 안 된다),
   (b) sign 게이트를 |target|≥0.15 부분집합·corr≥0.5 로 완화, (c) 합성 라벨 pretrain
   (`--synthetic`, 구현돼 있으나 검증 라벨이 cache 라 현재 FAIL).
2. **큐 규모.** 9벌 × 3.2–3.5h ≈ 31h + cache/pretrain 완료분 → 40h 안. C4 취소 시 7벌 ≈ 24h.
3. **2단계 진행 여부.** 체인은 큐를 일렬로 돈다. P0(3.3h) 가 끝나면 WIP 에 P0 vs 기존 S1_T05 를 바로
   남길 테니, 그 시점에 C 계열 계속 여부를 판단할 수 있다. §3 의 추세를 볼 때 **P0 만 돌리고 C 계열은
   P0 결과와 추론 sweep 으로 갈음**하는 것이 24h 를 아끼는 길이다 — 결정은 사용자 몫.

---

## 6. 남은 리스크·주의

- align run 의 checkpoint 는 `backbone.*` prefix 다. `tools/export_mat.py`·`align_infer_sweep.py` 는
  처리하지만, bare 모델을 기대하는 다른 도구는 prefix 를 벗겨야 한다.
- val split 에는 cache 가 없다(zeros). `select_on: hqnr` 이라 사용되지 않는다.
- RR 의 shift 는 0.06 px 라 RR 지표는 정렬로 거의 안 움직인다 — RR 차이는 학습 분포 변화(jitter)의 효과다.
- C3/C4B 의 공식 HQNR 은 P-frame 출력을 M-frame `lms` 와 비교한다 (계획 §13.5·§24 가 인정). D_λ 가
  뛰는 것은 좌표 불일치의 표현이지 스펙트럼 붕괴가 아니다 — `hqnr_alt`(inverse 뷰)를 같이 본다.
