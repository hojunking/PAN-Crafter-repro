# QB s2/s3: late-validation donor selection, unchanged Stage 2

Campaign: `PANDA_QB_S23_LATEVAL_20261115_v1`.
This is main tuning. Each server selects **one existing local donor** and runs
**one Student for 50K**. No new Stage 1, no transfer between servers, and no
automatic A2/epsilon/LR/gamma expansion.

| Server | Existing RGBN Stage 1 | Candidates | New Student |
|---|---|---|---|
| s2 | `UNIFIED_QB_S2_S1_RGBN_FULL_R1_S261016301` | 40K / 48K / 50K | `LV404850_LR100_G025`, seed261018101 |
| s3 | `QBS3_QB_S3_S1_RGBN_FULL_R1_S261016303` | 40K / 48K / 50K | Same recipe and Student seed |

Both source runs must have completed50K. All three checkpoint files, sidecars,
source configurations, hashes, evaluator and dataset identities are verified
**before the existing QB queue is stopped**. Missing candidates cause an explicit
error; no fallback to RGB, a different server, another step, or new Stage 1.

Selection is lexicographic **validation RRE50, RRE90, validation ERGAS, earlier
step**, restricted to these three candidates. No test metric or Teacher target
gate selects the donor. Existing matching validation records are checked against
their per-sample aggregates and source evaluation. A missing validation record
(normally s3@48K) is computed from the retained checkpoint, without learning.
All candidate evidence and the choice live in `donor/` of the new campaign.

## Method contract

- Existing Stage1: lambda_spa (`lambda_abs`) **4**, lambda_epsilon **.03**,
  Teacher LR1e-4, Aligner LR1e-5, epsilon radius3,50K.
- **RGBN everywhere**: QB bands [R,G,B,NIR]=[2,1,0,3], equal mean before Scharr
  for L_spa, q/g, calibration and validation/test RRE. Aligner still receives
  all four stored-LMS bands. Original QB stored LMS/repaired LRMS paths remain.
- **Lanczos-4 everywhere PAN is warped**: correction, epsilon injection,
  Teacher, Student, calibration, validation and RR/FR inference.
- **Common valid support**: true integer-index Lanczos8-tap support eroded by
  Scharr's one-pixel neighborhood. Same domain for pixel losses and calibration.
  Aligner input crop8 and RRE search margins19/24 remain distinct diagnostics.
- **Teacher/Student W128/D123/UBIL**, same backbone: bilinear upsampling + conv.
- Student input **[corrected PAN, interp23 I_lrms]**, five channels; no correction
  residual. Output4bands. Aligner scale1, both A and paired Teacher frozen.
- `w=1+0.25*d_e*r_q`, alpha=beta0, exact q=0 -> r_q=0.
- Student LR1e-4, warmup100, cosine to zero at50K, AdamW WD.01,
  betas(.9,.999), eps1e-8, batch48/micro12, FP32. Existing paired augmentation.
- No performance floors, early stopping, new loss, TTA, rotation consistency,
  weight averaging, architecture changes or post-processing.

The only intended experimental variable relative to local MAIN50 Q1 / PARALLEL
P1 is the donor selection. Same Student/data/augmentation seeds. Different
Stage1 seeds across servers are not repetitions with the same frozen model.
Exact full-train calibration is bound to the selected paired A/T checkpoint.

If the selected donor equals the local completed Q1/P1 donor, the scientific
recipe, all50 evaluations/weights, data/evaluator and exact calibration tables
are checked. An identical trajectory is explicitly reused (zero new updates,
not an independent repeat). Different or incomplete trajectories are trained
fresh; the new runner never continues an old Student with changed conditions.

## Handoff and safe replacement

The user pushes the commit from s1. On each target server pull the **whole
commit**, then use its matching command. These commands are not run on s1.
Do not discard local changes with a forced reset; resolve any pull conflict.

```bash
# s2
git pull --ff-only
python3 -m panda_qb_lateval.release
python3 -m panda_qb_lateval catalog --server s2
python3 -m panda_qb_lateval.enqueue --server s2 --check-only
python3 -m panda_qb_lateval.enqueue --server s2

# s3
git pull --ff-only
python3 -m panda_qb_lateval.release
python3 -m panda_qb_lateval catalog --server s3
python3 -m panda_qb_lateval.enqueue --server s3 --check-only
python3 -m panda_qb_lateval.enqueue --server s3
```

Launch verifies the server marker and authenticates only s2's old MAIN50 QB
or s3's old PARALLEL QB container (campaign/source/checkout/command). It sends
SIGTERM and waits for the final complete resume checkpoint, **never SIGKILL**.
Unfinished old jobs become CANCELLED_BY_USER with a replacement-campaign link;
completed jobs and all weights/evaluations are preserved. Other GPU workloads
are not terminated; the new runner waits for them. The old queue is not
automatically resumed. Stopped container restart policies are disabled.

New containers: `panda-qb-lateval-s2-v1`, `panda-qb-lateval-s3-v1`.
After inspecting a stopped new container, resume with the same enqueue command
plus `--resume`. Optimizer, data stream, RNG and donor/calibration binding restore
exactly; original source directories are mounted read-only. Runtime CUDA
preflight verifies native FR and exact restoration before real training.

Default image `hojunqueen/pancrafter-env:torch2.4.0-cu118`; optional `--dlpan`
or `PANCRAFTER_DLPAN` supplies the existing external evaluator toolbox.

## Evaluation and Sheet

Goal at one checkpoint: **corrected HQNR>=.9315 and RR ERGAS<=3.5**; no PSNR
floor. Original-PAN metrics remain auxiliary. No metric gates training.
Publish to existing **QB-ablations**, Bucket=main, four decimals, original FR
in D:F and corrected FR in H:J. Representative = H-qualified minimum ERGAS,
otherwise maximum corrected HQNR. All joint passes, unrestricted minimum
ERGAS/maximum PSNR, and fixed50K remain visible. Selection uses full precision.
The previous campaigns' upload thresholds and source files are not modified.

The **actually selected Stage1 Teacher** is also published with its RR and both
FR references, distinct from Student results. s3 can reuse its verified dual-FR
record; s2's missing corrected FR is measured on the exact selected Teacher,
checking reproduction of original FR first. A transient Sheet failure is retried
without deleting weights. No cost measurement is fabricated or copied from an
incompatible graph.

New output root:
`work_dir/panda_qb_lateval/PANDA_QB_S23_LATEVAL_20261115_v1/<server>/QB`.
Inspect `campaign_state.json`, `donor/selection.json`, `reuse_decision.json`,
`preflight/verified.json`, `stage1_reference/`, `publication/last_poll.json`, and
`runs/*/evaluation_0050000.json`. All1K model weights and two complete resume
generations are retained during this campaign.

CPU regression uses the production Docker image and serialized four-band
checkpoints on both source paths. Remote launch still checks actual local
weights/data/GPU/Sheet; local tests do not claim the remote jobs have started.
