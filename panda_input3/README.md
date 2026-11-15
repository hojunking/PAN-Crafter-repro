# INPUT3 — current priority on s1 / s2 / s3

The previous QB donor×schedule grid is **on hold**. Do not automatically resume
it after this campaign. This campaign trains only Stage 2, using each server's
existing local main Aligner/Teacher. No new Stage 1 training or weight transfer.

| Case | Student input, including the common upsampled MS base |
|---|---|
| A_RAW | `[original PAN, interp23 I_lrms]` |
| B_RAW_CORRECTED | `[original PAN, corrected PAN, interp23 I_lrms]` |
| C_CORRECTED_DIFF | `[corrected PAN, corrected PAN-original PAN, interp23 I_lrms]` |

All three use the same frozen Teacher, corrected-PAN q, shared calibration,
common Lanczos support eroded for Scharr, and loss `1+.25*d_e*r_q` (α=β=0).
Exact q=0 gives r_q=0. RGBN applies to L_spa/q/g/calibration/RRE, and Lanczos-4
applies throughout. Only the Student input layout changes. The RAW case still
uses Aligner/Teacher guidance while training; it is not a uniform-L1 baseline.
Both original- and corrected-PAN FR metrics use the same output and fixed donor.

| Server | Data | Local Stage 1 donor | Student width | LR | Updates |
|---|---|---|---:|---:|---:|
| s1 | WV3 | R1@27K, UNIFIED seed261016101 | 112 | 1.1e-4 | 40K |
| s2 | QB | R1@32K, UNIFIED seed261016301 | 128 | 1e-4 | 50K |
| s3 | GF2 | R2@42K, SEQ40 seed261016302 | 128 | 2.5e-4 | 50K |

One run per case, order A→B→C on each server: **9 runs total**. Paired Student
seed261018101, warmup100 + cosine to zero at the registered horizon, AdamW
WD .01, effective batch48/microbatch12, D123/UBIL, Aligner scale1. GF2 uses the
prior successful G3 LR250 recipe. No metric early stop. All evaluated1K models
and the last two complete optimizer/RNG restart generations are retained.

B and C contain the same two PAN images in invertible linear representations.
Their comparison tests parameterization/optimization, not extra information.
Shared Student weights use the existing named initialization. Extra PAN weights
start at zero: B starts with `[0,W_pan,W_ms]`, C with `[W_pan,0,W_ms]`.
They therefore start from the same corrected-PAN function. A starts from the
same shared weights applied to original PAN. Checkpoint identity records the
actual input order, and rejects B↔C model substitution despite equal shapes.

## Launch

After the user pushes this commit, each server pulls and starts its local queue:

```bash
git pull --ff-only
python3 -m panda_input3.release
python3 -m panda_input3.enqueue --server s1
# s2: --server s2 ; s3: --server s3
```

The launcher first verifies the release, data/evaluator, exact donor SHA and
Sheet. It then requests graceful stop of the explicitly registered prior queue,
verifies its retained restart checkpoint, marks unfinished jobs HELD_BY_USER,
disables automatic container restart, and starts INPUT3. Historical results and
weights are preserved. Unrelated GPU workloads are not stopped.
`--check-only` performs read-only admission/Sheet checks. `--resume` restarts
this campaign's authenticated stopped container. Never use it for an old held
campaign without a new user instruction.

## Evidence and Sheet

- Native RR20/FR20 every1K; no TTA. Original FR D:F and corrected FR H:J are on
  one row, four decimal places, in `WV3-ablations`, `QB-ablations`, or
  `GF2-ablations`. Bucket=ablation. Experiment names include the input case;
  Approach includes LR/horizon/width/guidance/donor step.
- Representative: minimum ERGAS among corrected-HQNR-qualified checkpoints;
  otherwise maximum corrected HQNR. Fixed endpoint, unrestricted min-ERGAS and
  max-PSNR slots are also retained. GF2 notes report the PSNR45.08 goal as well.
- Goals unchanged: WV3 H≥.9585/E≤2.040; QB H≥.9315/E≤3.5;
  GF2 H≥.9645/E≤.522/PSNR≥45.08, using corrected-PAN HQNR.
- Checkpoints: `work_dir/panda_input3/PANDA_S123_INPUT3_20261115_v1/<server>/<sensor>/runs/`.
  One frozen donor receipt and calibration per server live under `donor/`.
- Runtime CUDA preflight checks real input contents, common guidance/mask,
  matched B/C initial predictions, native dual-FR evaluation, and exact
  checkpoint restoration before production updates.
- CPU integration suite: `python -m unittest panda_input3.test_input3 -v` in
  `hojunqueen/pancrafter-env:torch2.4.0-cu118`.
