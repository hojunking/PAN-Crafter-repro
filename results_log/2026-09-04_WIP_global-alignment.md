# [WIP] Global alignment 40h 캠페인 — s1 (2026-09-05 00:42 재기동)

계획: [s1_w152_d123_global_alignment_40h_plan.md](../research_log/s1_w152_d123_global_alignment_40h_plan.md) ·
검토·구현: [2026-09-04_global-alignment-plan-review.md](../research_log/2026-09-04_global-alignment-plan-review.md) ·
계획 대비 차이·이슈·결과 상세: [2026-09-05_global-alignment-plan-vs-implementation.md](../research_log/2026-09-05_global-alignment-plan-vs-implementation.md)
판정: **best checkpoint HQNR(공식 12-19) → fSCC(12-19)**. 밴드 0.011 은 미검증 상한(어제 문서 §2).

## 상태

- **2026-09-05 00:30 체인 중단·P0 무효화** — 검증 지적: LR 을 flip/rot 한 뒤 phase-2 interp23tap 하면 HR 과 1px
  어긋남(표본 75%). 검토서 §7 의 9건을 수정하고 재기동한다. 무효 P0 는 `work_dir/_INVALID_augphase_…` 에 격리.
- **2026-09-05 00:42:38 재기동** (9건 수정 커밋 9b324b4 · T01–T12 12/12 · smoke 9/9 · 통합 P0/C3 완주). 큐 순서는 계획 §10.1 우선순위(P0·C1·C3·C2 a0.5·C2 a1.0·C4 dual-frame → C2 a0.25·C2 a0.75·C4 RT). P0 는 ≈3.7h 뒤 끝난다.

- 구현·검증 완료 (T01–T10 10/10 · smoke 9/9 · 통합 210 iter 완주).
- **2026-09-04 21:27:22 기동**, 마감 2026-09-06 13:27 (40h). 큐 `config/queues/s1_global_alignment.txt`
  9벌, 계획 §10.1 순서. 사용자 지시: "lms 안 맞는 부분은 알아서 맞추고(→ interp23tap 기본), 실험은 진행".
- C4 두 벌은 ShiftNet pretrain gate 가 FAIL(sign 0.678) 이라 계획 §15.3 규칙대로 **시작 즉시 exit 3 →
  체인이 FAILED 로 기록하고 넘어간다.** 실효 7벌 ≈ 24h. P0 는 ≈ 3.3h 뒤(9/5 01:00 경) 끝난다.
- 진행 확인: `ps -eo pid,ppid,args | grep '[_]run_cases'` · `tail -n +1 -f work_dir/cases_chain.log`

## 이미 확보된 중간 산출물

| 항목 | 값 | 어디 |
|---|---|---|
| shift cache | train 9714 / RR 20 / FR 20, SHA `8d136157…` | `outputs/global_shift_cache/` |
| FR 12-19 Δ | (−0.16, +0.18) LR px, 8/8 accepted | `wv3_fr.csv` |
| round-trip control (§20) | FR PSNR 61.9 dB · grad-energy 1.028 | `tools/align_diag.py` |
| 추론 sweep (기존 W152·d123) | α↑ → fSCC↑ HQNR↓ (D_s) | `work_dir/S1_T05_W152_D123_DUAL/results/infer_sweep.csv` |
| ShiftNet pretrain gate | **FAIL** (sign 0.678) | `outputs/global_shift_cache/shiftnet_pretrained.json` |

- **객관성 검증(22:40)**: 1차 P0 버그는 C 계열에 새지 않음(commit 확인). AlignTrainer+옛 커널 == 원 Trainer(210 iter 등가).
  단 **P0 커널 교체가 plateau 를 −0.0024 내려** C2-vs-P0 이득이 부풀려 보임 — 원 anchor 기준 C2 plateau +0.001. 대조 run
  `GA_CTRL_C2_BICUBIC15_A100`(원 커널 + C2 jitter) 이 체인 DONE 직후 자동 기동된다. 상세: 상세 기록 §7.

## 결과 표 (2026-09-05 21:20 중간 — 6/9 완료, C2 α0.75 진행 중, C4 는 gate FAIL rc=4)

best checkpoint HQNR(공식 12-19, 장면별 곱 평균) → fSCC. plateau = ep≥100 평균, final = ep245.

| run | 세팅 | **best HQNR** | ep | fSCC | D_λ | D_s | plateau | final | alt-frame HQNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor `S1_T05_W152_D123_DUAL` | bicubic phase1.5 · shift 없음 | 0.9546 | 110 | 0.8846 | 0.0227 | 0.0232 | 0.9516 | 0.9502 | — |
| `GA_P0_PHASEFIX` | interp23tap · shift 없음 | 0.9543 | 80 | 0.8864 | 0.0206 | 0.0257 | 0.9492 | 0.9464 | — |
| `GA_C1_FROZEN_RT_A100` | cache α1 · M 출력(inverse) | 0.9383 | **5** | 0.8555 | 0.0249 | 0.0378 | 0.9159 | 0.9125 | P-frame 0.8835 (fSCC 0.963) |
| `GA_C3_FROZEN_DUALFRAME_A100` | cache α1 · P 출력 · loss 만 inverse | 0.9245 | 235 | 0.9008 | 0.0399† | 0.0371 | 0.9224 | 0.9241 | M-frame 0.9160 (D_λ 0.022) |
| `GA_C2_INPUTONLY_A025` | 조건 입력만 α0.25 | 0.9534 | 145 | 0.8874 | 0.0192 | 0.0279 | 0.9518 | 0.9514 | — |
| `GA_C2_INPUTONLY_A050` | α0.5 | **0.9553** | 120 | 0.8915 | 0.0205 | 0.0248 | 0.9529 | 0.9525 | — |
| `GA_C2_INPUTONLY_A100` | α1.0 | **0.9553** | 225 | 0.9003 | 0.0213 | 0.0240 | 0.9528 | **0.9552** | — |
| `GA_C2_INPUTONLY_A075` | α0.75 | (진행 중) | | | | | | | |
| `GA_C4_TRAIN_*` 2벌 | trainable ShiftNet | gate FAIL (sign 0.678) → rc=4, 미학습 | | | | | | | |

† C3 의 D_λ 는 P-frame 출력을 M-frame `lms` 와 비교한 값 — 같은 모델의 M-frame 뷰는 0.0220. 좌표 충돌 ≈ 0.018.

**추론 시 α 분리 실험** (`tools/align_infer_sweep.py`, 완료 checkpoint 에 추론 Δ 만 변경):

| 학습 | 추론 α=0 | α=0.5 | α=1.0 |
|---|---:|---:|---:|
| C2 α1.0 checkpoint | **0.9555** / fSCC 0.891 | 0.9551 / 0.897 | 0.9553 / 0.900 |
| C2 α0.5 checkpoint | **0.9560** / 0.883 | 0.9552 / 0.892 | 0.9558 / 0.897 |
| P0 checkpoint | 0.9543 / 0.886 | 0.9524 / 0.899 | 0.9521 / 0.908 |

→ **C2 의 HQNR 이득은 추론 시 정렬이 아니라 학습 중 조건 입력 jitter(Δ 노이즈 sd 0.076 LR px)에서 온다.**
추론 시 정렬은 HQNR 을 바꾸지 않고 fSCC 만 올린다(+0.01). shift 없이 학습한 P0 에 추론 시 shift 를 주면 HQNR 이 내린다.

## 볼 곳

- 학습 로그 매 eval: `[핵심] HQNR / SCC / ERGAS` + `fSCC(12-19)` + `[alt frame] HQNR fSCC`
- `work_dir/<run>/best_hqnr_meta.json` (frame·alpha·cache hash), `metrics.csv` 의
  `d_lambda_official / d_s_official / fscc_official / hqnr_alt / fscc_alt / delta_mag_mean`
- 시각 패널(완주 후): `python tools/align_diag.py --panels --run work_dir/<run>` → `visualizations/fr12..19.png`
