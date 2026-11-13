# Shared ZEROQ Stage2 restart — s1 WV3 / s2 QB / s3 GF2

**Current selection policy (2026-11-14): corrected-PAN HQNR.** Use
[panda_corrected_hqnr](../panda_corrected_hqnr/README.md) and
`python3 -m panda_corrected_hqnr.enqueue --server s1` (or s2/s3) to resume
this same campaign with the new selection/stopping policy. The launch commands
below document the original rollout; the numerical training implementation is
still the one in this package.

The user requested an explicit zero-discrepancy exception and fresh Stage2
training. The frozen historical source files and their manifests are preserved.
The new campaign is `PANDA_S123_EQ40_ZEROQ_20261113_v1`; new experiments contain
`EQ40-QZERO` in their Sheet name. Earlier Stage2 weights are never resumed into
this campaign. Each new run starts from its original named initialization seed.

## Boundary rule

For valid pixels, the spatial rank is now:

```
r_q(i) = 0                         if q(i) == 0
         F_hat[g_bin(i)](q(i))      if q(i) > 0
w(i)   = 1 + 0.25 * d_e(i) * r_q(i)
```

`weighting.conditional_rank` is the one implementation used by all three
sensors. Exact positive values, including very small positives and positive
ties, retain the previous right-inclusive ECDF. There is no new epsilon cutoff.
Negative/nonfinite queries still fail validation. An all-zero calibration bin
therefore gives rank 0 for q=0, even when its raw right-inclusive ECDF is 1.

This is separate from spatial masking: invalid pixels are already excluded
before the lookup. The new condition handles *valid* pixels with zero mismatch.
The raw calibration population, tau_e, group boundaries, and original CDF
artifacts remain unchanged and are reused with their hashes verified. Every
training step logs `q_zero_fraction` on the common valid domain; calibration's
population/zero count is recorded in `preflight/verified.json`. Absence of zeros
in one calibration is not a claim that zeros never occur in training or another
sensor. This guard imposes the requested semantic boundary regardless.

## Common method and intentional recipe differences

| Setting | s1 WV3 | s2 QB | s3 GF2 |
|---|---|---|---|
| Stage1 | Existing local R1/R2, unchanged | Existing local RGBN R1, unchanged | Existing/new local unified R1/R2, unchanged |
| Student repeats / horizon | 2 x 40K | 2 x 40K | 2 x 40K |
| Width / LR | W112 / 1.1e-4 | W128 / 1.0e-4 | W128 / 2.0e-4 |
| MS bands / Student input channels | 8 / 9 | 4 / 5 | 4 / 5 |
| RGBN indices (zero based) | 4,2,1,6 | 2,1,0,3 | 2,1,0,3 |

All new Students use exactly `panda_eq40_zeroq.model.Stage2`, the same shared
weighting function, D123 / UBIL, full Aligner scale 1, corrected PAN + interp23
MS without correction residual, alpha=0 / gamma=.25 / beta=0. Existing Stage1
and both Stage2 frozen-Teacher/Student paths use the same `panda_unified.core`
Lanczos-4 warp and eroded full-tap support. RGBN is used for L_spa, q/g,
calibration and RRE. Official metric functions are unchanged.

Dataset-specific band counts, normalization/data catalog and original-LMS QB
repair provenance remain as registered. They are not changes to the loss or
geometric method. Donor gates are unchanged: WV3 E<=2.04 and P>=38 then max H;
QB E<=3.5 then max P, then H; GF2 E<=.522 and P>=45.08 then max H. These existing
donor gates explicitly use test metrics; the new guard does not change them.

## Pull and execute

Pulling updates files; it does **not** replace a running Python process. After
the user pushes and each server pulls the complete commit, run:

```bash
# s1, already launched by the implementing agent
python3 -m panda_eq40_zeroq.enqueue --server s1

# s2
python3 -m panda_eq40_zeroq.enqueue --server s2

# s3
python3 -m panda_eq40_zeroq.enqueue --server s3 --wait
```

`--check-only` performs read-only source/data/environment/Sheet checks. The
launcher gracefully stops only registered same-server EQ40 predecessors (and
the original unified QB predecessor on s2), preserving results. An unrelated
GPU workload is left alone; `--wait` waits for it. Old containers are retained.
The new run verifies the complete source manifest and the server marker.

If Stage1 is incomplete, the coordinator finishes only the required unchanged
Stage1 jobs first. GF2 therefore still has its two 50K Stage1 runs before the
two new Students. No weights are moved between servers. Old Stage2 queue entries
are marked superseded; their source, weights, logs and evaluations are retained.

The first new run migrates its own Sheet tab to FR67 if needed. The publisher
retains original-PAN FR in D:F and corrected-PAN FR in H:J on the same row,
four decimals, same checkpoint verified. Non-evaluated values remain blank.
Use this launcher for the new main method, not the previous EQ40/FR67 launcher.
Independent s3 Table3 Stage1 ablations are outside this restart; if still active,
finish/stop them explicitly or leave the new GF2 launcher waiting behind them.

```bash
python -m unittest panda_eq40_zeroq.test_zeroq panda_fr_columns.test_reporting -v
```

Tests exercise real exact calibration classes on all three sensor bindings,
all-zero/mixed/positive distributions, valid-vs-masked pixels, positive-query
parity, serialized four/eight-band inputs, actual optimizer steps, frozen donor
integrity, complete checkpoint restoration, and the shared numerical paths.
The production launcher also requires a disposable CUDA preflight and native
dual-PAN FR metric probe before the fresh Student starts.
