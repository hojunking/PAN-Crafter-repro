# s2 QB / s3 GF2 — corrected-only Stage 2, dual PAN reference

This adds the method confirmed for s1 WV3 without modifying any active historical
package. Source and checkpoints stay local to their server. No weight transfer.

## Queue

| Server | Order | Dataset / run | Donor | Updates | LR |
|---|---:|---|---|---:|---:|
| s2 | prerequisite | finish existing unified RGBN Stage1 if incomplete | existing s2 run | through50K | original |
| s2 | 1 | QB Stage2 R1 | selected from that same local RGBN Stage1 | 40K | 1e-4 |
| s2 | 2 | QB Stage2 R2 | identical frozen donor; new Student seed | 40K | 1e-4 |
| s3 | 1 | GF2 RGBN Stage1 R1 | fresh local Aligner/Teacher | 50K | T1e-4 / A1e-5 |
| s3 | 2 | GF2 RGBN Stage1 R2 | independent fresh local Aligner/Teacher | 50K | T1e-4 / A1e-5 |
| s3 | 3 | GF2 Stage2 R1 | selected Stage1 R1 | 40K | 2e-4 |
| s3 | 4 | GF2 Stage2 R2 | selected Stage1 R2 | 40K | 2e-4 |

QB excludes the RGB group. It reuses only
`UNIFIED_QB_S2_S1_RGBN_FULL_R1_S261016301` from
`PANDA_UNIFIED_WV3_QB_20261113_v1/s2/QB` on s2. The launcher checkpoints/stops
that exact old container, then its unchanged original runtime finishes only the
RGBN donor. Old unfinished Stage2 jobs become SUPERSEDED_NOT_STARTED or
SUPERSEDED_PARTIAL, preserving their true step and all evidence. Completed old
results remain. Do not restart the old unified s2 container alongside this queue.

GF2 starts fresh because its historical selected donor is RGB/W112, whereas the
new campaign uses RGBN and W128/UBIL in both Teacher and Student. It does not
reinterpret an old RGB checkpoint as RGBN. Stage1 lambda_spa=2, lambda_epsilon=.003,
epsilon radius3; same reconstruction+spatial+controlled-shift equations. Stage1
seeds261016301/261016302; Student seeds261018101/261018102. GF2 LR2e-4 is inherited
from the previous successful GF2 LR200 recipe; shortening to40K and changing the
method is a new experiment, not a promise of reproducing its old result.

## Common method

- Teacher/Student W128/D123/UBIL, FP32; correctedPAN + interp23 I_lrms4,
  input5 channels and output4. No PAN correction-residual input. The ordinary
  predicted HRMS residual added to the MS base remains.
- Student weight `1 + 0.25*d_e*r_q`: alpha0, gamma.25, beta0. Aligner scale1.
- Frozen Aligner and Teacher during Student fitting; no new Stage1 in QB.
- RGBN mean[2,1,0,3] before Scharr for L_spa, g/q, calibration and RRE. Aligner
  receives all4 stored LMS bands; preserve the registered QB original-LMS path.
- Lanczos4 reflection in every PAN/epsilon path. All pixel losses and full-train
  calibration use the same maximal tap support eroded by3x3 Scharr support.
- AdamW WD.01, betas(.9,.999), eps1e-8, batch48/micro12, warmup100 then cosine0
  at the declared horizon. Existing paired flips/rot90 augmentation unchanged;
  no rotation consistency, soft loss, TTA, SWA, EMA or band-weighted loss.
- QB's existing early stop remains: at10K, stop if original-PAN HQNR at8/9/10K
  is strictly below.90 in all three evaluations.

## Explicit donor selection

For each completed local Stage1, audit all50 saved1K checkpoints and their RR20 /
original-PAN FR20 results. No selection using corrected-PAN HQNR.

- QB: require ERGAS<=3.5, then highest PSNR, then highest original-PAN HQNR;
  deterministic remaining ties use lowerERGAS and earlierstep. No PSNR floor.
  Both Student seeds share this one selected A/T and full-train calibration.
- GF2: require ERGAS<=.522 AND PSNR>=45.08, then highest original-PAN HQNR;
  remaining ties use lowerERGAS, higherPSNR, earlierstep. Each repetition selects
  its own donor. If a donor has no qualified checkpoint, its Student is explicitly
  BLOCKED_DONOR, and the other independent GF2 Student can still run. No silent
  threshold relaxation, other-repeat substitution or cross-server copy.

These are requested test-aware selection criteria, not validation-only choices.
Selection receipt, full candidate list, all checkpoint SHA256s and calibration
identity are retained. Validation RRE is a separate diagnostic. GF2 full RR/FR
remains every1K; full validation RRE runs at30/40/50K (2201 validation patches).
The training equations and RNG trajectory do not depend on validation frequency.

## FR comparison and Sheet

Every1K, generate one prediction for each of20 native FR scenes. Evaluate it
with originalPAN and then with the frozen Aligner's correctedPAN. Output and
storedLMS stay unchanged; D_lambda must be exactly identical. The unchanged D_s
implementation regenerates its low-resolution PAN from whichever PAN is supplied.
Both HQNR/D_s/D_lambda triplets are saved. Corrected-PAN metrics are diagnostic,
not directly comparable to methods evaluated against the originalPAN.

Native RR20 still uses original GT and unchanged official metrics. No masks,
posthoc shifts or crops are added to official RR/FR metrics. Save all1K evaluated
weights plus the latest2 complete optimizer/data/RNG resume generations.

Automatic destinations: QB-ablations (s2), GF2-ablations (s3). Each selected
checkpoint gets labeled `[original PAN]` and `[corrected PAN]` companion rows.
Report4 decimals, retain full precision and provenance in JSON/notes. Existing
Student publication policy remains: minimumERGAS among original-HQNR-qualified
checkpoints, otherwise maxoriginalHQNR; all H/E passes plus fixed40K. Mirror the
same selected steps for the correctedPAN companion; never choose it separately.
GF2 Stage1 representatives use the explicit donor criterion (label an unqualified
interim lowestERGAS candidate as such), plus fixed50K.

Final targets: QB originalHQNR>=.9315 and ERGAS<=3.5; GF2 originalHQNR>=.9645,
ERGAS<=.522, PSNR>=45.08. GF2 notes/labels explicitly distinguish H/E passes
from all3-target passes. Do not copy unmatched inference-cost values.

## Handoff and launch

After the user pushes, pull this complete commit on the corresponding server.
Retain local datasets, service account and `gspread/server.txt` (`s2` or `s3`).
The host needs Python3 and Docker; training uses the pinned image environment.

```bash
# s2
python3 -m panda_sensor_eq40 catalog --server s2
python3 -m panda_sensor_eq40.enqueue --server s2 --check-only
python3 -m panda_sensor_eq40.enqueue --server s2 --wait

# s3
python3 -m panda_sensor_eq40 catalog --server s3
python3 -m panda_sensor_eq40.enqueue --server s3 --check-only
python3 -m panda_sensor_eq40.enqueue --server s3 --wait
```

Use tmux for a launcher waiting on another workload. s3 waits for existing GPU
work and does not interrupt Table3 or delete its results. s2 only gracefully
stops the authenticated old `panda-unified-s2-v1`, then resumes its RGBN Stage1
from its complete checkpoint. Unrelated workloads are not stopped. No SIGKILL.

Default image `hojunqueen/pancrafter-env:torch2.4.0-cu118`. If unavailable, build
`docker build -f Dockerfile.env -t pancrafter-env:sensor-eq40 .` and pass
`--image pancrafter-env:sensor-eq40`. Set `PANCRAFTER_DLPAN` when its checkout
is external to the image. Do not regenerate release.json to bypass a stale pull.

Detached containers: `panda-sensor-eq40-s2-v1`, `panda-sensor-eq40-s3-v1`.
Check with `docker logs --tail 50 NAME` and enqueue `--status`. Use enqueue
`--resume --wait` for an authenticated stopped container. Full checkpoint restore
preserves optimizer, model, data stream, augmentation RNG and exact calibration.

Output: `work_dir/panda_sensor_eq40/PANDA_S23_QB_GF2_EQ40_20261113_v1/SERVER/DATASET`.
Read-only check mode can run on s1; training refuses a mismatched server marker.
The s2 original donor/state is checked on s2, not copied to s1. s1 validation
therefore does not claim that a remote experiment has started.

## Verification

```bash
python -m unittest discover -s panda_sensor_eq40/tests -v
python -m panda_sensor_eq40.release --environment
python -m panda_sensor_eq40.native_probe --server s2  # operator test, no Sheet writes
python -m panda_sensor_eq40.native_probe --server s3
```

Synthetic integration checks cover four-band native paths, Stage1 gradients,
calibration save/load binding, immutable frozen A/T, exact Student restart,
queue dependencies, selection gates and Sheet routes. Native probes use real
FR data and a fixed fractional shift with an interp23 baseline to check both
metric references; they are not trained-model performance results. Actual CUDA
preflight on each remote server uses disposable updates and exact checkpoint
restoration before production, with full source/data/evaluator checks.
