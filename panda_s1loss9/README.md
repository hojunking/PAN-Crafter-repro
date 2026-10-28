# PANDA Stage 1 loss grid (S1LOSS9)

**2026-10-28 queue update:** This package preserves the original experiment registration and resume code. Its original nine-pair order below is historical. For the currently authorized retained cases, new R4/tuning runs, and s3 deployment, use [the SensorRefine handoff](../research_log/PANDA_SensorRefine_R34_Handoff_2026-10-28.md) and `scripts/enqueue_panda_sensor_refine.sh`.

The user-specified notation is `L_S1 = L_rec + lambda_abs L_abs + lambda_epsilon L_epsilon` in [the method specification](../research_log/PANDA_Method_EquationStyle_2026-10-26.md). Here QB and GF2 have C=4; the document's C=8 describes WV3. `L_epsilon` constrains the relative response to a known PAN translation; it is not a shift-magnitude penalty.

Server s1 owns QB (HQNR >= .9315, ERGAS <= 3.5), s3 owns GF2 (HQNR >= .9645, ERGAS <= .522). No time limit. Every pair runs fresh Stage 1 50K, selects matching A/T from all retained checkpoints by registered RR-validation RRE50/RRE90/ERGAS/earlier-step, recalibrates on RR train, then trains a fresh BASE Student 30K. Test metrics never select the Stage 1 donor.

| Order | Pair | lambda_abs | lambda_epsilon |
|---:|---|---:|---:|
| 1 | A200_E003 | 2 | .003 |
| 2 | A020_E003 | .2 | .003 |
| 3 | A050_E003 | .5 | .003 |
| 4 | A200_E0003 | 2 | .0003 |
| 5 | A200_E030 | 2 | .03 |
| 6 | A020_E0003 | .2 | .0003 |
| 7 | A020_E030 | .2 | .03 |
| 8 | A050_E0003 | .5 | .0003 |
| 9 | A050_E030 | .5 | .03 |

This is a complete 3x3 factorial screen. First comes the fresh baseline, then one-factor changes, then interactions. The same Stage 1 seed 261016101 and Student seed 261018101 are paired across coefficients. The fresh baseline reproduces the implementation; it is **not an independent-seed confirmation** of the preceding Sensor5 baseline.

All other settings are fixed: LR_A=1e-5, LR_T=LR_S=1e-4, 100-update warmup then cosine, AdamW weight_decay=.01, epsilon radius3 uniform-area disk, full PAN Aligner scale1, RGB structural reference, Lanczos-4 for all PAN paths, all4-band MS input/output, interp23 MS base. Student W112D123 uses UBIL and `[corrected PAN, corrected PAN - original PAN, I_ms^lr]` (6 input channels), alpha=.25, gamma=.5, beta=0. No rotation loss, SWA, TTA, or band-weighted loss.

Every loss pair has its own run identity, donor checkpoint/selection receipt, and exact full-training calibration. Pair coefficients are enforced at Stage 1 construction, donor loading, evaluation, and publication. Existing Sensor5/WV3 source code is not changed.

The queue finishes the existing Sensor5 Stage 1 and BASE Student first, runs these nine pairs next, then returns to the remaining Sensor5 LR/horizon/Hard cases and finally WV3 TUNE30/HARD36. See the queue handoff for controller installation.

Read-only catalog: `python -m panda_s1loss9 catalog --server s1 --csv` (or s3). Explicit run: `python -m panda_s1loss9 run --server s1`. The launch controller prepares measured cost profiles and the publisher. Official rows go to QB-ablations / GF2-ablations: all joint target passes plus one representative per Student run (minimum ERGAS among HQNR-qualified states, else maximum HQNR); Stage 1 rows retain validation-RRE selection. Visible approach descriptions include both Stage 1 coefficients and Stage 2 settings.
