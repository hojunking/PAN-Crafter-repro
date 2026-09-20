# GF2 HQNR 격차 감사 스크립트 (2026-09-20)

`results_log/2026-09-20_gf2-hqnr-gap-audit.md` 의 수치를 만든 CPU 전용 스크립트. 전부 **저장된 출력·데이터만 읽는다**(GPU·학습 없음).
산출물은 스크립트와 같은 폴더에 쓴다(`OUT = 스크립트 폴더`) — 보고서에 인용한 결과 파일은 `results_log/assets/gf2audit/` 에 복사해 두었다.

| 스크립트 | 무엇을 재는가 | 보고서 절 |
|---|---|---|
| `sweep.py` | 저장된 `full_best_hqnr*.mat` 을 클리핑 L=10/11·무클립·정수 반올림·GNyq 0.20–0.40·QB/GeoEye1 표로 바꿔 FR 재평가 (지표 상수 감도) | §2 |
| `exp_anchor.py` | EXP(= lms) 의 FR D_λ/D_s/HQNR 을 GF2/WV3/QB/WV2 에서 계산 (WV3 는 CANConv 논문 EXP 행과 대조) | §2 |
| `eff_gnyq_exp.py` | 원 MS 의 PAN 대비 유효 GNyq (native vs Wald) | §3 |
| `f3_check.py` | KNOWN_ISSUES F-3 검사: `ms` 가 genMTF(GNyq)·`gt` 의 (2,2) 위상 데시메이션과 맞는가 (QB 원본 = 양성 대조) | §3 |
| `f1_reg_check.py` | KNOWN_ISSUES F-1 검사(lpan 장면 상관) + PAN↔MS NCC 정합 오프셋(HR px) | §3 |
| `reg_lrgrid.py` | NNLS intensity 기반 PAN↔MS 오프셋(보조 측정) | §3 |
| `ds_decomp.py` | D_s 를 밴드별 q(fused_b, PAN) − q(lms_b, PAN_lp) 로 분해 — "PAN 과잉 상관" 진단 | §4 |
| `traj.py` | `metrics.csv` 의 HQNR/D_λ/D_s 궤적 · best vs last (선택 편향 여부) | §4 |
| `inspect_gf2.py`, `h5dump.py`, `frjson.py` | h5 구조·값 범위 덤프, `fr_mat20.json` 요약 | §3 |

실행 예: `PANCRAFTER_DLPAN=/path/to/DLPan-Toolbox python tools/gf2_audit/sweep.py`
