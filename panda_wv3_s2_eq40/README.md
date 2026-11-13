# WV3 Stage 2 corrected-PAN / interaction-only follow-up

Scope: s1 only. Finish the existing unified Stage 1 R2 to 50,000 updates with
its original model/config/optimizer/RNG/data stream. Do not run the old Stage 2
R2. Preserve old R1 Stage 1/2 results and every Stage 1 checkpoint. s2 QB and s3
Table 3 packages remain unchanged.

| Run | Frozen donor | Student input | alpha | gamma | Peak LR | Updates |
|---|---|---|---:|---:|---:|---:|
| EQ40 R1 | original local unified Stage 1 R1 | corrected PAN + interp23 I_lrms | 0 | .25 | 1.1e-4 | 40,000 |
| EQ40 R2 | original local unified Stage 1 R2 | corrected PAN + interp23 I_lrms | 0 | .25 | 1.1e-4 | 40,000 |

Each Student is freshly initialized with the existing S seed, not Teacher weights.
W112/D123/UBIL, 9 input channels, 8 output bands; scale1; beta0; FP32 AdamW
WD.01, batch48/microbatch12, warmup100 then cosine to zero at40K. No TTA,
rotation loss, SWA, output transport, or additional loss. Existing rotation/flip
input augmentation is unchanged. Only the correction-residual input is removed;
the U-Net still predicts an HRMS residual added to the interp23 MS base.

All existing shared numerical settings remain: Lanczos4 reflection for Stage1,
Teacher, Student and epsilon; common detached full8tap support eroded by3x3
Scharr for every pixel loss/calibration; RGBN mean indices[4,2,1,6] for
L_spa, g, q, validation and testRRE. Aligner receives all8 storedLMS bands.

## Stage 1 donor selection (explicitly test-aware)

Wait until each local repetition completes50K; audit all50 saved1K evaluations
and checkpoint SHAs. Keep candidates satisfying RR ERGAS<=2.04 AND PSNR>=38;
choose maximum original-PAN FR HQNR, ties by lowerERGAS/higherPSNR/earlierstep.
If none qualifies, stop with NO_QUALIFIED_STAGE1_DONOR rather than silently
substitute a weaker checkpoint or another repetition. These are test-set
criteria requested for this campaign, not validation-only selection.

R1 selects27K: HQNR0.9228283873940223, ERGAS2.0334644364358057,
PSNR38.03433329225183. The previous validationRRE selector chose41K. Since
both donor selection and horizon change, this is not a controlled two-component
ablation against old Stage 2. Recompute full-train calibration for the new donor.

## Evaluation and Sheet

Every1K: RR20 and FR20, unchanged full-size official evaluator. Evaluate the
same prediction with originalPAN and with that donor's Lanczos4-correctedPAN.
Keep storedLMS/output coordinates unchanged. D_lambda must be identical;
D_s recomputes its low-resolution PAN from the supplied PAN. Corrected-PAN
HQNR/D_s are a separate diagnostic reference, not official comparable values.

`WV3-ablations`: two explicitly labeled companion rows per selected checkpoint,
`[original PAN]` and `[corrected PAN]`, four decimal places. Select checkpoints
using originalPAN only under existing rules: HQNR>=.9585, minimumERGAS among
qualified checkpoints; otherwise maximumHQNR. Include all HQNR>=.9585 &
ERGAS<=2.04 passes plus fixed40K. Mirror the same chosen steps for correctedPAN.
Keep original precision and checkpoint/data/evaluator/selection hashes in local
JSON and row notes. New measurements only; do not copy unmatched inference cost.

## Execute / resume

```sh
python3 -m panda_wv3_s2_eq40.enqueue --check-only
python3 -m panda_wv3_s2_eq40.enqueue
# If this exact campaign container stopped:
python3 -m panda_wv3_s2_eq40.enqueue --resume
```

The launcher verifies the current original container's campaign/source/checkout,
requests SIGTERM, waits for its complete checkpoint without force killing, runs
a disposable CUDA integration check, then starts `panda-eq40-s1-v1`. The
coordinator resumes only original Stage 1 jobs through the unchanged original
runtime. It records old Stage 2 R2 as SUPERSEDED_NOT_STARTED, then runs EQ40
R1/R2. Do not restart the old `panda-unified-s1-v1` container alongside it.

Output: `work_dir/panda_wv3_s2_eq40/PANDA_WV3_S2_EQ40_20261113_v1/s1/WV3`.
Complete restart checkpoints retain optimizer, sampler, augmentation/RNG and
calibration identities. All1K evaluated weights remain, plus latest2 complete
resume generations. Preflight uses scratch weights; production begins at0.
