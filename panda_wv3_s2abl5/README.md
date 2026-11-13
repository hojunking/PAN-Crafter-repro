# WV3 Stage 2 component ablation — five cases, three paired Student seeds

Campaign: `PANDA_WV3_S2ABL5_S1_20261114_v1`. Runs locally on **s1**, after the
currently running ZEROQ main R2 completes. No Stage 1 training, donor transfer,
global-rank case, correction-residual channel, soft loss, or rotation loss.

| Case | Student PAN input | Pixel fitting weight |
|---|---|---|
| A_RAW_L1 | Original PAN | `1` |
| B_ALIGN_L1 | Corrected PAN | `1` |
| C_DE_ONLY | Corrected PAN | `1 + 0.25*d_e` |
| D_RQ_ONLY | Corrected PAN | `1 + 0.25*r_q` |
| E_OURS | Corrected PAN | `1 + 0.25*d_e*r_q` |

All cases also receive the same eight-band interp23 MS reconstruction base.
`r_q` is the existing edge-strength-conditional training-distribution rank;
`q == 0` explicitly produces `r_q == 0`. C uses only the teacher-difficulty
factor as an ablation; E retains no independent alpha term. The internal
`gamma` field stores the 0.25 coefficient for C, D, and E; the full weight
formula is explicitly recorded in each config and Sheet note.

## Controlled comparison

- Frozen Stage 1 reference: `UNIFIED_WV3_S1_S1_RGBN_FULL_R1_S261016101`, **27K**.
  Donor checkpoint SHA-256:
  `bf8e9f5624bb49e6d2f5aa437505029b3d03a09dbd3fde7a319734c83c05d0d8`.
  The selection receipt and original calibration are independently pinned in
  `config/panda_wv3_s2abl5/plan.json`; no new donor selection or recalibration.
- Student W112/D123, bilinear upsampling + convolution (UBIL), nine input
  channels, full alignment scale 1, Lanczos-4, RGBN reference throughout.
- Common eroded Lanczos support for every training case. A uses the frozen
  reference Aligner only to define this shared support and evaluation PAN;
  its Student receives original PAN and inference requires only the Student.
- Every FR output is evaluated against **the same fixed donor's corrected PAN**.
  Original-PAN FR metrics are auxiliary. Different cases never define their
  own reference PAN. RR/FR use all 20 native test samples and the unchanged
  metric implementation and evaluation crops.
- 40K optimizer updates, LR `1.1e-4`, warmup 100, cosine to 0, AdamW weight
  decay 0.01, effective batch 48 / microbatch 12. Frozen main's remaining
  optimizer, data, augmentation, precision, and stream settings are inherited.
- Student seeds `261018101`, `261018102`, `261018103`. Within each seed, all
  five cases use identical initialization and data/augmentation streams.
  These are three Student repeats conditional on one donor, not independent
  repetitions of the complete two-stage pipeline.

## Queue and existing-result reuse

Queue is round-major: A/B/C/D/E R1, then A/B/C/D/E R2, then A/B/C/D/E R3.
There are **15 result slots and 14 new training runs**. E R1 reuses the completed
`ZQ40_WV3_S1_R1_S261018101` exactly: same seed, donor, calibration, architecture,
inputs, loss, training schedule, data streams, and metric definitions.
The original config, evaluation provenance, all 40 checkpoint file hashes,
and forward/loss equivalence are verified. Its artifacts remain at their
original paths, with reuse explicitly disclosed in the Sheet and summary.
The current main R2 uses a different donor and is not reused for E R2.

**Retention exception:** preserve the source main R1 directory under
`work_dir/panda_eq40_zeroq/PANDA_S123_EQ40_ZEROQ_20261113_v1/s1/WV3/runs/`
`ZQ40_WV3_S1_R1_S261018101`, including its evaluated models and JSON records.
It is now an ablation source. New ablation runs retain every evaluated 1K
checkpoint, including models below the main target; two complete optimizer
resume generations are also retained. Do not apply main-only pruning here.

## Results and reporting

The paper comparison is **fixed 40K for every run**, with all three registered
seeds included in each mean and sample standard deviation (`ddof=1`).
`summary/fixed40_mean_std.json` and `.csv` disclose partial counts and only mark the
table complete when all five cases have three 40K results. No best-step
replacement, seed selection, or mixed-donor main R2 is used in this table.

Existing publication rules remain additional views: corrected HQNR-qualified
minimum ERGAS representative (otherwise highest corrected HQNR), all joint
passes, and fixed 40K. Targets remain corrected HQNR >= 0.9585 and ERGAS <=
2.040. Sheet: `WV3-ablations`, experiments `WV3-S2ABL5-<case> R<n> (RGBN)`.
Metrics use four decimals; original-PAN columns D–G and corrected-PAN H–J
remain explicitly distinct. Unmeasured inference costs are left blank.

## Operation

```bash
python3 -m panda_wv3_s2abl5 catalog
python3 -m panda_wv3_s2abl5.enqueue --check-only
python3 -m panda_wv3_s2abl5.enqueue
docker logs --tail 30 panda-wv3-s2abl5-s1-v1
```

The detached container first publishes the queue and verified E R1 records,
then waits without GPU training until main R1/R2 both finish 40K and the GPU
is free. It runs a disposable native-data CUDA preflight before the first
new optimizer update. Preflight checks matching initialization, shared input
reference/mask, Ours numerical parity, full optimizer/RNG/stream restoration,
and native dual-PAN evaluation. A restart uses
`python3 -m panda_wv3_s2abl5.enqueue --resume`; the same source/data pins and
registered configs are required.

State is under
`work_dir/panda_wv3_s2abl5/PANDA_WV3_S2ABL5_S1_20261114_v1/s1/WV3/`:
`queue_wait.json`, `campaign_state.json`, `preflight/verified.json`,
`reuse/ours_r1.json`, `publication/`, and `summary/`.
Publication failures are recorded and retried; training artifacts persist.
The frozen main numerical and publication packages are unchanged.
