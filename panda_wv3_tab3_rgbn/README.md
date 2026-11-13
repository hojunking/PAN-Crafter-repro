# WV3 Table 3 — s3, RGBN, Stage 1 only

User-approved 2026-11-13: five Table 3 configurations × three repeats,
50,000 updates each, round-major; stop the old s3 experiment immediately and
preserve its results. This package does not launch or stop any s1/s2 experiment.
It does not train Stage 2 or transfer existing weights.

## Cases and paired repetitions

| Order within a round | Table 3 | Case | Alignment | λ_spa (`lambda_abs`) | λ_ε |
|---:|---|---|---|---:|---:|
| 1 | (a) | A_SCHARR | Nonlearned image-specific Scharr global translation | 0 | 0 |
| 2 | (b) | B_REC | Learned Aligner, reconstruction only | 0 | 0 |
| 3 | (c) | C_REC_SPA | Learned Aligner, reconstruction + spatial | 2 | 0 |
| 4 | (d) | D_REC_EPS | Learned Aligner, reconstruction + controlled shift | 0 | 0.003 |
| 5 | Ours | OURS | All three Stage 1 losses | 2 | 0.003 |

The queue runs `(a)→(b)→(c)→(d)→Ours` for R1, then R2, then R3.
Paired seed roots are **261016101, 261016102, 261016103**. Within a round,
Teacher/Aligner initialization and data/augmentation streams are matched.
Every run starts fresh. Total: **15 runs / 750,000 optimizer updates**.
Case (a) trains the Teacher but has no learned Aligner parameters. A raw-PAN
U-Net baseline is not a row of the referenced Table 3.

## Common method

- **WV3 RGBN = stored indices [4,2,1,6] (Red, Green, Blue, NIR1)**, equal arithmetic
  mean before Scharr. L_spa and validation/test RRE all use this definition.
  The learned Aligner still receives PAN + **all eight stored LMS bands**.
  There is no Stage 2 q/calibration computation in this Stage1-only campaign;
  the recorded future q reference is also RGBN.
- Teacher **W112/D123/UBIL**, matching the current unified main architecture:
  bilinear ×2 (`align_corners=False`) + 3×3 conv at both upsampling levels.
  Reconstruction base is `interp23 I_lrms`; Teacher receives corrected PAN + MS base.
- **Lanczos-4, whole-sample reflection, FP32** for native correction, controlled
  shifts, validation and RR/FR prediction. Shared numerical implementation is
  imported unchanged from `panda_unified.core`.
- Every pixel loss uses **the full 8-tap observed support, eroded by a 3×3
  Scharr stencil**. Support indices use integer pixel coordinates + floor(shift).
  No fixed ROI8 loss crop. The Aligner input crop8 and RRE metric search margins
  serve different purposes and remain as in the unified main implementation.
- Teacher peak LR **1e-4**; learned Aligner **1e-5**. AdamW β=(.9,.999), ε=1e-8,
  weight decay .01; batch48/microbatch12; warmup100 + cosine to zero at50K.
- Controlled-shift disk radius3 PAN pixels; scale1; no soft loss, rotation
  consistency, TTA or weight averaging. Existing paired image augmentation remains.
- A disabled loss is not computed; e.g. cases (a)/(b)/(c) do not invoke epsilon
  injection during training. Learned-case epsilon validation remains a diagnostic.

### Nonlearned (a)

One `(dy,dx)` per input image is estimated against **observed stored LMS RGBN**,
never GT. Search: each axis ±5 px, integer coarse search, quarter-pixel local
refinement ±1 px. Both scoring and final Teacher loss use the same Lanczos/Scharr
support rule; each candidate score is normalized by its valid pixel count.
Ties prefer smaller displacement norm, then L1 norm, then dy/dx lexicographically.
The ±5 search extent is retained from the previous Scharr baseline; it is not
the synthetic epsilon radius or a learned-shift bound.

An in-memory cache stores **exact PAN+LMS content → computed shift**. It never
approximates a shift or rotates a precomputed shift; a new augmentation is
evaluated on its actual tensors. It contains no GT or fitted parameters.
Cache eviction/restart recomputes the same matcher, and does not alter the model.

## Evaluation, retention and Sheet

- **Only 30K, 40K, 50K**: save full validation records, official RR20/FR20
  evaluations, and immutable Aligner+Teacher model weights. No 1K evaluations.
- Latest **two complete resume generations** are retained, saved every250 steps
  (plus clean stopping boundaries), including optimizer and RNG/stream states.
  They are recovery files, not additional validation checkpoints.
- **Only 50K**: upload **one row per completed run** to `WV3-ablations` in
  spreadsheet `1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE`, gid `261001010`.
  Experiment names contain the case, repeat and `(RGBN)`. No queued placeholders,
  30K/40K rows, threshold selection, or validation-best substitution.
  Numbers are displayed to four decimals; raw metrics and full provenance remain
  in the row notes. Other campaigns' rows and extra columns are preserved.
- Automatic local summaries: `summary/table3_fixed50.{json,csv,md}`.
  Five rows with **all three repeats' mean ± sample std (ddof=1)**; incomplete
  triplets remain blank, with individual available results retained in JSON.
  Each run's RRE50 is the median over its20 scenes; the table averages those
  three medians, not a pooled60-scene median. No chosen subset of seeds.
- RRE reference: RR GT RGBN, FR original stored LMS RGBN. FR is a proxy, not
  geometric ground truth. Gaussian σPAN/σref=.8/.8 (RR),1.98/.7 (FR), search±8,
  margins19/24 as in the current unified method. Invalid test registrations are
  recorded explicitly; no zero substitution or successful-scene-only median.
  Thus an incomplete RRE metric is left blank even if reconstruction metrics exist.
  Official RR/FR reconstruction metrics retain their benchmark crops/references.
- Old campaign results/weights/Sheet rows are preserved. This isolated package
  does not edit the source closure of the currently active unified s1/s2 main.

## s3 handoff and immediate switch

User handles `git push`; s3 pulls this commit before the commands below. The
reference production image is `hojunqueen/pancrafter-env:torch2.4.0-cu118`.
No checkpoint transfer is needed. WV3 dataset symlinks/registered hashes and
`gspread/account.json` must be available on s3 as for previous WV3 experiments.

```bash
git pull --ff-only
python3 -m panda_wv3_tab3_rgbn.release
python3 -m panda_wv3_tab3_rgbn catalog --server s3
python3 -m panda_wv3_tab3_rgbn.enqueue --server s3 --check-only
docker ps --format '{{.Names}}\t{{.Command}}'
```

After identifying the **actual old s3 experiment container**, replace the name
below (do not type the placeholder literally):

```bash
python3 -m panda_wv3_tab3_rgbn.enqueue --server s3 --stop-container OLD_S3_CONTAINER_NAME
docker logs -f panda-tab3-rgbn-s3-v1
```

The launcher checks code/environment, data and Sheet read-only access **before
sending SIGTERM** to the explicitly named old container. Its server and checkout
must be verified. It preserves the old container/results, waits at most300s for
graceful exit, and never sends SIGKILL. It then requires an idle GPU, runs a
disposable CUDA preflight (all5 losses, exact saved-boundary restoration,
Lanczos/native inference parity, Scharr cache parity), checks disk space and
launches the new queue. It does not wait for the old case to finish50K.

If the old process is a host Python process or another supervisor, the s3 agent
must inspect that exact process/supervisor and send its graceful stop first;
then run the same launcher without `--stop-container`. Do not use broad pkill.
No matching s3 identity or other remaining GPU work causes a refusal, not a kill.

Resume an interrupted **new** container only after inspecting its state:

```bash
python3 -m panda_wv3_tab3_rgbn.enqueue --server s3 --resume
python3 -m panda_wv3_tab3_rgbn.enqueue --server s3 --status
docker exec panda-tab3-rgbn-s3-v1 python -m panda_wv3_tab3_rgbn aggregate --server s3
```

Artifacts are under
`work_dir/panda_wv3_tab3_rgbn/PANDA_WV3_TAB3_RGBN_S3_20261113_v1/s3/WV3/`.
Persistent source/data/evaluator admission hashes prevent silent mixed-version
resume. Numerical failure is recorded and independent runs can continue;
infrastructure/evidence errors stop for repair. Failed/missing runs remain visible
and never become selectively excluded three-run summaries.

## Local verification

```bash
python -m unittest discover -s panda_wv3_tab3_rgbn/tests -v
```

The numerical tests need PyTorch and the repository's normal dependencies.
The launcher deliberately mounts WV3 sources via the historical mount helper's
`s1` dataset mapping; its old `s3` mapping points to GF2 and must not be used here.

Implementation verification on s1: 10 automated tests passed, plus real WV3
batch48/microbatch12 optimizer updates and exact CPU checkpoint continuation
for all five cases. Reference Docker environment, all four data splits, the full
RGBN validation inventory, and Sheet dry-run passed (zero row writes). The
committed-source-only import closure is verified separately. The new campaign's
CUDA preflight runs on s3 at launch; no s3 process was stopped from s1.
