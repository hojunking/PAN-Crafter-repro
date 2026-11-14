# s3: stop GF2 and run independent QB experiments

User decision, 2026-11-14: GF2 has passed its goals; stop its active run and
remaining queue on s3 and use s3 for QB alongside s2. Preserve existing GF2
results/checkpoints. s2's five QB MAIN50 cases and s1's WV3 three-seed ablation
remain unchanged. **This package must execute on s3; it cannot stop a remote
server merely by being committed on s1.**

## Evidence behind the plan

The latest GF2 G3@30K is corrected HQNR .9720, ERGAS .5063, PSNR45.2337
(all targets pass). QB Q1@25K reaches ERGAS3.4866 but corrected HQNR .8883;
its early4K representative has H .9375 but E3.7281. This is still an FR/RR
tradeoff, not a solved QB recipe. Existing s2 sweeps LR.5/1/1.5e-4 and gamma
.1/.25/.5. s3 adds an independent donor plus complementary LR/gamma controls.

## Execution order

One **fresh local s3 Stage1**, 50K, seed261016303, then five fresh 50K Students.
Stage1 matches the current s2 RGBN recipe: W128/D123 UBIL Teacher, lambda_spa
(`lambda_abs`)=4, lambda_epsilon=.03, Aligner LR1e-5, Teacher LR1e-4,
epsilon radius3, AdamW WD.01, batch48/micro12,100-step warmup/cosine to zero.
Only the independent Stage1 seed differs. No weight transfer or old RGB/
bicubic/transposed-upsample donor fallback.

All Stage1 1K checkpoints get RR20 and native full512 FR20 using **both**
original and corrected PAN. Validation diagnostics are retained at30/40/50K.
Once50K finishes, choose the local donor from E<=3.5 by descending PSNR,
then corrected HQNR. If none qualifies, choose minimum ERGAS, then PSNR/H;
there is **no performance admission gate**. This is explicitly test-aware
selection and is recorded as such. All five Students use the same chosen
checkpoint and calibration.

| Priority | Stage2 case | Peak LR | Gamma |
|---|---|---:|---:|
|1|P1_LR100_G025|1e-4|.25|
|2|P2_LR075_G025|7.5e-5|.25|
|3|P3_LR100_G000|1e-4|0|
|4|P4_LR100_G0125|1e-4|.125|
|5|P5_LR075_G0125|7.5e-5|.125|

P1 is a cross-server control, not a clean donor-only ablation (the independently
trained donor also changes calibration). P3 is a uniform-L1 diagnostic with the
Aligner still fully used; gamma0 is not advertised as the adaptive fitting
method. The other cases probe intermediate LR and weaker adaptive strength.

Stage2 remains corrected PAN + interp23 LRMS (five input channels), alpha=beta0,
scale1, RGBN reference everywhere, Lanczos4 on every path, common eroded support,
q=0 => r_q=0, and `w=1+gamma*d_e*r_q`. No residual input, global rank, TTA,
rotation consistency, EMA/SWA or new loss. Student seed261018101 is paired
across the five recipes. Every run reaches50K regardless of metric floors;
errors in bytes/data/numerics still fail explicitly. One seed per recipe is
screening, not a three-seed robustness result.

## Handoff / launch

After the user pushes from s1, pull the complete commit on s3. Do not copy
individual files. The launcher verifies the frozen shared source, pinned
environment, QB repaired MS/original stored LMS and live Sheet before cutover.

```bash
python3 -m panda_qb_s3_parallel.enqueue --check-only
python3 -m panda_qb_s3_parallel.enqueue
```

Optional `--dlpan /path/to/DLPan-Toolbox` and `--image ...` match existing launchers.
The second command authenticates only s3's MAIN50 GF2 containers, sends
**SIGTERM**, waits for graceful checkpoint save (never SIGKILL), records
`CANCELLED_BY_USER` for unfinished GF2 runs, and publishes those statuses.
Completed results and all weights remain. It posts the GF2 Stage1 reference,
then starts `panda-qb-s3-parallel-v1`. Unrelated GPU jobs are not stopped.
A retry resumes the same QB container/config/optimizer; it does not create a
second run. No old GF2 queue is automatically restarted.

```bash
docker logs --tail 40 panda-qb-s3-parallel-v1
```

Output: `work_dir/panda_qb_s3_parallel/PANDA_QB_S3_PARALLEL_20261114_v1/s3/QB`.
Existing experiment destinations are retained: `QB-ablations`, Bucket=main.
Original FR in D:F, corrected FR in H:J, four decimals. Representatives,
unrestricted RR extrema, passes and fixed50K are recorded. Teacher results
are explicitly Stage1 and never mislabelled Student results.

## Stage1 reference reporting on all servers

See `panda_stage1_reference/README.md`. s3 automatically posts its selected QB
Teacher when the Stage2 donor is prepared. The former GF2 Teacher remains in
GF2-ablations for comparison. On s2, run the CPU publication command separately;
no restart or modification of the active QB training is required.

Tests: five real synthetic Student optimizer paths, exact CPU resume,
frozen A/T integrity, source-method Stage1/Stage2 numerical parity, zero-q
handling, same initialization, no floors, donor selection, and own Sheet rows.
Native CUDA preflight additionally runs on s3 before production.
