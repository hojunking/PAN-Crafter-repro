# [WIP] Global alignment 40h 캠페인 — s1 (2026-09-04 21:27 기동)

계획: [s1_w152_d123_global_alignment_40h_plan.md](../research_log/s1_w152_d123_global_alignment_40h_plan.md) ·
검토·구현: [2026-09-04_global-alignment-plan-review.md](../research_log/2026-09-04_global-alignment-plan-review.md)
판정: **best checkpoint HQNR(공식 12-19) → fSCC(12-19)**. 밴드 0.011 은 미검증 상한(어제 문서 §2).

## 상태

- **2026-09-05 00:30 체인 중단·P0 무효화** — 검증 지적: LR 을 flip/rot 한 뒤 phase-2 interp23tap 하면 HR 과 1px
  어긋남(표본 75%). 검토서 §7 의 9건을 수정하고 재기동한다. 무효 P0 는 `work_dir/_INVALID_augphase_…` 에 격리.

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

## 결과 표 (완주 시 채운다)

| run | 세팅 | best HQNR | fSCC | D_λ | D_s | alt-frame HQNR | best ep |
|---|---|---:|---:|---:|---:|---:|---:|
| 기존 anchor `S1_T05_W152_D123_DUAL` | bicubic phase1.5 · shift 없음 | 0.9546 | 0.8846 | 0.0227 | 0.0232 | — | 110 |
| `GA_P0_PHASEFIX_W152_D123_DUAL` | interp23tap · shift 없음 | | | | | — | |
| `GA_C1_FROZEN_RT_A100_…` | cache α1 · M 출력(inverse) | | | | | | |
| `GA_C3_FROZEN_DUALFRAME_A100_…` | cache α1 · P 출력 · loss 만 inverse | | | | | | |
| `GA_C2_INPUTONLY_A050_…` | 조건 입력만 α0.5 | | | | | — | |
| `GA_C2_INPUTONLY_A100_…` | α1.0 | | | | | — | |
| `GA_C2_INPUTONLY_A025_…` | α0.25 | | | | | — | |
| `GA_C2_INPUTONLY_A075_…` | α0.75 | | | | | — | |
| `GA_C4_TRAIN_RT_…` | trainable · M 출력 | (gate FAIL 시 미실행) | | | | | |
| `GA_C4_TRAIN_DUALFRAME_…` | trainable · P 출력 | (gate FAIL 시 미실행) | | | | | |

## 볼 곳

- 학습 로그 매 eval: `[핵심] HQNR / SCC / ERGAS` + `fSCC(12-19)` + `[alt frame] HQNR fSCC`
- `work_dir/<run>/best_hqnr_meta.json` (frame·alpha·cache hash), `metrics.csv` 의
  `d_lambda_official / d_s_official / fscc_official / hqnr_alt / fscc_alt / delta_mag_mean`
- 시각 패널(완주 후): `python tools/align_diag.py --panels --run work_dir/<run>` → `visualizations/fr12..19.png`
