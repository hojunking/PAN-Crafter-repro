# s3 QB deepen — independent local Stage1, eight Student comparisons

Campaign `PANDA_S3_QB_DEEPEN_OWNSTAGE1_20261030_v1`. Each server trains its own
Stage1 independently. s1 retains its independently trained Stage1 and current QB queue;
s3 trains its own QB Aligner/Teacher. No model transfer or imported donor is
required. No automatic WV3 resumption follows this campaign.

## Queue: one Stage1 + eight Students

First run `S1_A200_E030 R1`: **50K**, seed **261016301**, lambda_abs2,
lambda_epsilon.03, epsilon radius3, Aligner LR1e-5, Teacher LR1e-4,
warmup100/cosine, AdamW decay.01. All random streams derive from this new seed.
Stage1 is independently initialized and trained entirely on s3's QB train set.

After all50K and all50 scheduled checkpoints are retained, the existing
validation selector ranks RRE50, then RRE90, then validation ERGAS, then earlier
step. The selected step is unknown before training; there is no44K or SHA
preset, no test-HQNR selection, and no imported A/T pair. The selected Aligner
and Teacher come from the same checkpoint and remain fixed for all eight
Students. Full-training calibration is generated from this local pair and
reused across their existing coefficient/optimizer comparisons.

All Students: 30K warmup100/cosine, LR1e-4, W112, gamma.5, beta0, full Aligner
scale1, RGB structure reference, Lanczos4, UBIL, `[PANc, PANc-PAN, I_lrms]`,
single Student forward, and original Hard loss definition.

| Order | Case | Replicas | AdamW decay | Alpha | Purpose |
|---|---|---|---:|---:|---|
| 1 | S1_A200_E030 | R1, seed261016301 | .01 | — | Independent local Aligner/Teacher,50K |
| 2–3 | QB3_CONTROL | R1/R2 | .01 | .25 | Local selected-donor controls |
| 4–5 | QB3_WD000 | R1/R2 | 0 | .25 | Remove existing decay |
| 6–7 | QB3_WD100 | R1/R2 | .1 | .25 | Tenfold existing decay |
| 8–9 | QB3_A075 | R1/R2 | .01 | .75 | Increase existing Teacher-error weighting |

Total290K updates before early stops:50K Stage1 +240K Stage2. Time cap:none.
Student seeds are261018101/261018102; matched Student comparisons share their
s3 donor and all sampling/initialization streams. Historical s1 comparisons
also include an independently trained donor and hardware difference; they are
not pure Student hyperparameter ablations.

Existing QB stop rule applies **only to Stage2**: all8K/9K/10K HQNR strictly
below.90 causes an authenticated safe stop. Low Teacher HQNR never stops
Stage1. Stopped Students are recorded as early-stopped, not fully completed.

No SWA/EMA/TTA/rotation loss/new band loss or Aligner/input architecture changes.

## Inputs and sources

s3 needs the six QB files pinned by `ablr2/sensor_sources.json` and
`config/bglobal_rr/qb_original_lms.json`: msfix train/val, original train/val LMS,
RR20, FR20. The launcher adds read-only mounts for resolved QB symlink targets.
SHA/population checks enforce B/G/R/NIR, peak2047, train17139/val1905.
The existing Stored-LMS + msfix-MS input policy is retained; the separately
discussed common-interp23-input change is not silently introduced here.

## Launch

After Git pull, use the existing torch-enabled host Python with
`gspread/server.txt=s3`. Preserve the GF2 goal weights and stop its active job
to free the local GPU. `panda_gf2_closeout` is a separate helper for this work;
QB does not require its receipt, other queues' completion, or their Sheet readback.

```bash
python -m panda_qb_deepen catalog --csv
mkdir -p work_dir/panda_qb_deepen/launcher_s3
nohup python -u -m panda_qb_deepen.enqueue \
  > work_dir/panda_qb_deepen/launcher_s3/controller.log 2>&1 < /dev/null &
```

No donor archive import step is required. The launcher clones the original s3
runtime image/environment, adds QB data mounts, checks input SHA/populations,
profiles both A+T and A+S deployments, and starts the single registered queue.
The launcher checks only local GPU availability. Optional `--wait` waits for
that GPU; it does not wait for GF2/WV3 status files or another server.
There is no waiting-controller source pin or queue-edit approval requirement.
An unlaunched old plan is refreshed automatically, retaining its history.

The original S1LOSS9 runtime's Stage1 build, update, checkpoint restore,
full-horizon selector, and same-checkpoint donor loader are reused in an
isolated process. Only registered case/lane configuration and queue control
are adapted. Existing source files remain unchanged. Checkpoint/model integrity
and exact-resume checks are retained, as is protection against duplicate local
writers to the same output directory.

Actual-data preflight checks Stage1 updates to both A/T and exact resume, then
Stage2 update/resume before each new case. Every scheduled evaluation retains
its complete checkpoint. A signal or disk guard saves exact optimizer, RNG,
and sampling state. Failures are not automatically restarted or substituted.

## Publication and retained models

Both stages publish to **QB-ablations**. Stage1 retains its validation-RRE
selection receipt and posts that selected checkpoint. Stage2 posts all
HQNR≥.9315/ERGAS≤3.5 joint passes plus one representative per run (minimum
ERGAS among HQNR qualifiers, otherwise maximum HQNR). Approach text reports
LR, alpha/gamma, AdamW decay and independently trained Stage1 seed.

All evaluated weights are saved, including intermediate goal passes. The
launcher requires verified final Sheet readback and all nine runs truthfully
accounted for before completion. It also checks the local Stage1 complete50K
history and selected checkpoint; no missing Stage1 is replaced by an s1 model.

Output:
`work_dir/panda_qb_deepen/PANDA_S3_QB_DEEPEN_OWNSTAGE1_20261030_v1/s3/QB/`

Launcher status: `work_dir/panda_qb_deepen/launcher_s3/status.json`.

Read-only final donor verification: `python -m panda_qb_deepen verify-donor`.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest panda_qb_deepen.tests.test_campaign -v
```

CPU tests cover Stage1 A/T updates and exact resume, independent stream seeds,
complete50K selection requirements, Stage2 actual AdamW updates and exact
resume for all four settings, lane isolation, QB Sheet thresholds, independent
startup without other campaign records and QB data mounts. These are implementation checks;
actual s3 data/GPU validation occurs before production training.
