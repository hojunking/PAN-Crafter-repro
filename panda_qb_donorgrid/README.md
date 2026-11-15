# QB: existing local donor checkpoints × Stage 2 schedules

This is the current s2/s3 QB plan. It supersedes the single-donor
`panda_qb_lateval` queue. **No new Stage 1 training. No weight transfer.**

Each server uses 40K, 48K and 50K checkpoints from its own completed Stage 1:

| Server | Existing Stage 1 run |
|---|---|
| s2 | `UNIFIED_QB_S2_S1_RGBN_FULL_R1_S261016301` |
| s3 | `QBS3_QB_S3_S1_RGBN_FULL_R1_S261016303` |

These are three checkpoints of one local training trajectory, not three
independent Aligner seeds. All three are evaluated as fixed donors; there is
no validation/test ranking to remove a candidate before Stage 2.

## Queue: nine Stage 2 runs per server

| Order | Cases | Schedule | Total updates |
|---|---|---|---|
| 1–3 | `D40_C50`, `D48_C50`, `D50_C50` | warmup100; cosine to 0 at 50K | 50K each |
| 4–6 | `D40_C15_HOLD`, `D48_C15_HOLD`, `D50_C15_HOLD` | warmup100; cosine to 1e-6 at 15K; hold through 50K | 50K each |
| 7–9 | `D40_C25_HOLD`, `D48_C25_HOLD`, `D50_C25_HOLD` | warmup100; cosine to 1e-6 at 25K; hold through 50K | 50K each |

Peak LR 1e-4, AdamW WD .01, effective batch48/microbatch12, α=β=0,
γ=.25; Student seed261018101 is paired across all cases and both servers.
The hold is a constant learning rate, not a model averaging operation.
Early decay is an unverified scheduling hypothesis: it may lower RR error
before late FR deterioration, but does not guarantee the joint target.

Method remains RGBN for L_spa/q/g/calibration/RRE, W128/D123/UBIL for Teacher
and Student, Lanczos-4 for every PAN warp, common Lanczos-support/Scharr1
valid area, Aligner scale1, corrected PAN + interp23 I_lrms input,
`w=1+.25*d_e*r_q`, exact q=0 ⇒ r_q=0. No residual input or new loss.
The Stage 1 source recipe remains λ_spa=4, λ_ε=.03, epsilon radius3.

## Execute locally on each server

After the user pushes the commit, on **s2**:

```bash
git pull --ff-only
python3 -m panda_qb_donorgrid.release
python3 -m panda_qb_donorgrid.enqueue --server s2
```

On **s3**, use the same commands with `--server s3`.
`--check-only` verifies release/environment/data, all three local checkpoint
hashes and existing Sheet layout without replacing the queue. `--resume`
restarts this campaign's authenticated stopped container.

The launcher replaces only the declared prior QB MAIN50/PARALLEL and LATEVAL
queues on that physical server. It requests graceful stop, preserves existing
results/weights, cancels remaining jobs and disables the old containers'
restart policy. Other GPU workloads are left running; the new runner waits.
No local donor or checkpoint means fail before cutover, with no automatic
retraining or donor substitution. Source work directories are mounted read-only.

## Evidence and reporting

- Calibration is separate for each donor and shared by its three schedules.
- Runtime performs an initial native CUDA preflight and exact checkpoint
  restoration check. Complete restart states every250 steps; evaluated models
  and official RR20/FR20 results every1K. No metric early stop; every run50K.
- Goal on the **same checkpoint**: corrected-PAN HQNR≥.9315, ERGAS≤3.5.
  No PSNR gate. Selection uses full precision, not rounded Sheet values.
- `QB-ablations`, Bucket `main`, four decimal places, original FR D:F and
  corrected FR H:J on the same row. Experiment names encode donor step and
  schedule. Representative, unrestricted min-ERGAS/max-PSNR, every joint pass,
  and fixed50K slots are retained. Each actual donor Teacher also gets its
  original/corrected FR reference row, with exact checkpoint identity.
- Output root: `work_dir/panda_qb_donorgrid/PANDA_QB_S23_DONORGRID_20261115_v1/<server>/QB/`.
  Donor receipts/calibration: `donors/step_0040000`, `step_0048000`, `step_0050000`.
- Runs are paired tuning comparisons, not 9 independent repetitions. Official
  RR/FR trajectories are test-aware diagnostics; final selection must be reported
  accordingly. Corrected-PAN HQNR is not directly comparable to original-PAN HQNR.

CPU integration suite: `python -m unittest panda_qb_donorgrid.test_donorgrid -v`
in the production `hojunqueen/pancrafter-env:torch2.4.0-cu118` image.
