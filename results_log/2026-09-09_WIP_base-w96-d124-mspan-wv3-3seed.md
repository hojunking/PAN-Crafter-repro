# 2026-09-09 [WIP] 새 baseline W96·D124 · MS+PAN 9ch · 단일 HRMS task — WV3 3-seed (s1)

> **진행 중.** 기동 2026-09-09 19:58 (s1), 마감 09-10 19:58. 큐 `config/queues/base_w96_d124_mspan_wv3_3seed.txt` 3건 순차.
> 기준 문서 `research_log/PAN_research_baseline_W96_D124_2026-09-09.md` · 실행 준비 `research_log/2026-09-09_w96-d124-mspan-wv3-3seed-launch.md`.
> 진행 확인: `tail -n +1 -f work_dir/cases_chain.log | grep --line-buffered '\[cases\]\|핵심'` · 결과는 시트 `WV3-s1` 범주 ⑰.
> 실행: 재구성본(`fixed`) · 측정: 지표 v2(`py`, DLPan 프로토콜 재구현) · FR: 논문 `.mat` 20장(`fr_mat20`), best 선택도 같은 세트.

## 무엇을 돌리는가

사용자 결정(2026-09-09)으로 mainline 이 바뀌었다. **U-Net W96 · depth [1,2,4] · 입력 MS+PAN 만(9ch = concat(PAN, bicubic↑MS)) ·
LPAN/HPAN 채널 없음 · PAN reconstruction task·loss 없음(`mars: ms`, batch 복제·PAN forward 자체 없음) · MARs mode γ/β 제거
(`mode_modulation: false`) · attention 없음 · crop=False · bicubic 잔차 base(M-frame) · 50K AdamW 1e-4/wd0.01 cosine · batch 48.**
params 2.123 M. seed 2025 · 1234 · 7777.

| 실행명 | seed | 상태 |
|---|---:|---|
| `BASE_W96_D124_MSPAN_WV3_S2025` | 2025 | 19:58 시작 — 18 it/s, 248 epoch. run 당 약 1 h 예상(같은 골격 `MS1_w96_9ch_msonly` 가 1 h 15 m) |
| `BASE_W96_D124_MSPAN_WV3_S1234` | 1234 | 대기 |
| `BASE_W96_D124_MSPAN_WV3_S7777` | 7777 | 대기 |

이 3벌은 새 기준의 **일반 HRMS 복원 baseline** 이다. 이후 정합/fitting 후보(기준 문서 §4)는 이 3벌과 비교한다.
과거 W168·d123 dual(`S1_T05_W168_D123_DUAL`, ARCH 캠페인)은 직접 대조군이 아니다. 판정 HQNR → SCC; 정렬 축은
`2026-09-09_alignment-after-training-analysis.md` §3 에 따라 fSCC·δ_out 병기.

## 이 캠페인을 위해 멈춘 것

아키텍처 고정 다중 데이터셋 캠페인(`config/queues/arch_w168_multiset_3seed.txt`, 09-09 09:04 기동)을 19:57 에 중지했다.

| run | 상태 |
|---|---|
| `ARCH_W168_D123_DUAL_GF2_S{2025,1234,7777}` | **완료** — HQNR 0.9576 / 0.9625 / 0.9594 (논문 세트 20장, 시트 GF2-s1 업로드됨) |
| `ARCH_W168_D123_DUAL_QB_S2025` | 중단 (약 80분, epoch ckpt 2개 남음; finished_at 없음 → 시트·`--all` 에 안 잡힌다) |
| QB S1234·S7777, WV3 ×3 | 미실행 |

재개하려면 이 캠페인이 `[cases] DONE` 을 찍은 뒤 `./tools/campaign_start.sh --queue config/queues/arch_w168_multiset_3seed.txt --hours 30 --label arch-multiset-resume`.
체인은 완료분(GF2 ×3)을 건너뛰고 QB S2025 는 최신 체크포인트에서 `--resume` 을 먼저 시도한다.

## 결과 (채워 넣는다)

run 이 끝나면 여기에 HQNR / D_λ / D_s / fSCC / SCC / ERGAS 와 학습 시간을 적고, 3 seed 의 평균·N−1 표준편차를 낸다.

## 추기 (같은 날 20:52) — 다음 캠페인: A1–A3 PAN 앞단 전역 정합, B0 체인 뒤 자동 기동

사용자 지시 정정: 진행할 실험은 [`research_log/PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md`](../research_log/PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md) 다.
위 B0 3벌은 그 대조군이므로 그대로 완주시키고, 끝나면 `config/queues/pa_s1.txt`(seed 2025: A1 → A2 → A3) 가 자동 기동된다
(`tools/_chain_after.sh`, 로그 `work_dir/chain_after.log`). 구현·검토 노트: [`research_log/2026-09-09_pa-a1-a3-implementation.md`](../research_log/2026-09-09_pa-a1-a3-implementation.md).

| 실행명 | case | loss | 상태 |
|---|---|---|---|
| `PA_A1_REC_W96_D124_9CH_S2025` | A1 | L_rec | 대기 (B0 체인 뒤) |
| `PA_A2_OUTEDGE_W96_D124_9CH_S2025` | A2 | L_rec + 0.1·L_edge (5K ramp) | 대기 |
| `PA_A3_GEO_W96_D124_9CH_S2025` | A3 | L_rec + 0.01·L_geo (5K ramp) | 대기 |

s2(seed 1234: B0 → A2→A3→A1) · s3(seed 7777: B0 → A3→A1→A2) 는 `./tools/pa_prepare.sh` 한 줄. 시트 HQNR = best_raw 의 raw_original(원 PAN, 전체 프레임).
raw_valid / aligned_valid / best_aligned / Δ̂ 통계는 run 폴더 `checkpoint_metrics.csv` 와 `results/pa_diag.json`. 판정 규칙(§12): 같은 (server, seed) block 안의 대응 차이 A2−A1, A3−A1, Ak−B0.
