# GF2 W128 Stage 2 — s3

Six recipes × two paired Student seeds, in round-major order (G1–G6 R1, then
G1–G6 R2). All runs start fresh and train for 60K updates without a time cap.

The reference is **GF2P_C60LR200 R2 @47K**, W112, LR2e-4,
alpha0.25/gamma0.50: HQNR0.9646694883, ERGAS0.5148830159, PSNR45.0841380957.
The width candidate comes from QB Q2_W128: W128, alpha0.25/gamma0.50.
GF2's existing W128/LR1e-4 R1/R2 runs already completed; their best posted PSNRs
were about45.00/45.02, so this queue explores higher LR without repeating them.

| Case | Student width | Peak LR | Alpha | Gamma | Horizon |
|---|---:|---:|---:|---:|---:|
| G1_LR200 | 128 | 2e-4 | 0.25 | 0.50 | 60K |
| G2_LR150 | 128 | 1.5e-4 | 0.25 | 0.50 | 60K |
| G3_LR250 | 128 | 2.5e-4 | 0.25 | 0.50 | 60K |
| G4_LR200_H050 | 128 | 2e-4 | 0.125 | 0.25 | 60K |
| G5_LR150_H050 | 128 | 1.5e-4 | 0.125 | 0.25 | 60K |
| G6_LR250_H050 | 128 | 2.5e-4 | 0.125 | 0.25 | 60K |

G1 changes only Student width relative to the existing W112/LR2e-4 scientific
recipe. The other comparisons test LR and a joint halving of alpha/gamma; H050
does not separate their individual effects. Keep seed-paired results and late
checkpoint trajectories when judging the width effect. The previous three-goal
success was one seed, not a demonstrated seed average.

## Local donor and method

- Reuse **s3's own** GF2 `RGB_MATURE_U23` validation-selected **42K** Aligner and
  paired W112 Teacher from `PANDA_S13_SENSOR5_RGB_L4_STAGE12_20261026_v2`.
  No new Stage 1 and no weights copied from s1 or s2.
- Source checkpoint SHA:
  `937ac56b559b76a151df661bbb853119512d6f9cab283346c61186db27ff7478`.
  Source Stage1 used lambda_abs2, lambda_epsilon0.003, epsilon radius3px.
- GF2 has four MS bands; the Student receives six channels:
  `[corrected PAN, corrected PAN-original PAN, interp23 LRMS4]` and predicts four.
  Depth[1,2,3], UBIL, Aligner scale1.0, RGB edge reference[2,1,0], and Lanczos4
  Teacher/Student warps retain the existing GF2 implementation.
- AdamW, weight decay0.01, batch48/microbatch12, warmup100, cosine to0 at60K.
  Seeds R1=261018101, R2=261018102. Beta0, no rotation loss/TTA/SWA/EMA/band loss.
  Existing paired augmentation and evaluation stay unchanged.
- The original donor and successful47K Student must still exist on s3.
  Their file/receipt/evaluation hashes are authenticated before starting.
  Calibration uses the existing full RR-train procedure in the new output root.
  The old campaigns are mounted read-only.

## s3 launch

After the user pushes this commit, pull the complete commit on s3. The launcher
needs host Python3 and Docker; the ML environment runs in the container. It
does not depend on any old container remaining installed.

```bash
docker build -f Dockerfile.env -t pancrafter-env:gf2-w128 .
python3 -m panda_gf2_w128.enqueue --server s3 --image pancrafter-env:gf2-w128 --wait
```

`--wait` waits for existing GPU workloads to finish, including an active s3
ablation queue. It never interrupts them. Run the waiting launcher in a persistent
terminal such as tmux. There is no automatic resumption of old WV3/QB queues.
For status and a stopped container:

```bash
python3 -m panda_gf2_w128.enqueue --server s3 --status
python3 -m panda_gf2_w128.enqueue --server s3 --resume --wait
docker logs --tail 40 panda-gf2-w128-s3-v1
```

The source release, exact environment/evaluator, data inventory, local donor,
successful reference, and read-only Sheet preflight are checked first. W128
inference costs are measured on s3 before training. The durable worker owns one
trainer and one CPU publisher and requires final Sheet readback before reporting
completion. Resumption restores model/optimizer/scheduler/data stream/RNG.

Output:
`work_dir/panda_gf2_w128/PANDA_S3_GF2_W128_FIXEDDONOR_20261106_v1/s3/GF2`.
Every1K official evaluation saves the full checkpoint; originals and the prior
GF2 successful model are never overwritten. Retention cleanup is a separate
completed-run operation. Active new runs must not be pruned.

## Sheet and targets

Publish to **GF2-ablations**, server=s3, Experiment=`GF2 W128 <case> <repeat>`.
Target: same-checkpoint **HQNR>=0.9645, ERGAS<=0.522, PSNR>=45.08**.
Preserve the existing H/E-pass-all plus one representative posting rule:
minimum ERGAS among HQNR-qualified steps, otherwise maximum HQNR.
Explicitly label whether all three goals pass. Do not replace the prior W112
rows or equate the H/E representative with the three-goal best model.
Approach/config notes retain width, LR, alpha/gamma, scale, source donor,
cost protocol and measured timing.

## Validation and portability

```bash
python -m unittest discover -s panda_gf2_w128/tests -v
python -m panda_gf2_w128.release --environment
python -m panda_gf2_w128 catalog --csv
```

Tests exercise all six CPU optimizer updates, unchanged frozen Aligner/Teacher,
four-band/six-channel graphs, exact checkpoint continuation, source isolation,
queue ordering, target boundaries and writable-mount scope. They are synthetic
correctness checks, not GF2 performance measurements. Real s3 model/data/GPU
execution is checked on s3; this package does not claim a remote run has started.
