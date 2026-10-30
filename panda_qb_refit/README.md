# QB fixed-donor refit, 2026-10-30

Ten fresh Students (five settings × R1/R2), 290K updates total, with no wall-time cap.
Targets remain **HQNR ≥ .9315 and ERGAS ≤ 3.5 at the same checkpoint**.

| Order | Setting | Updates | LR | α | γ | Seeds |
|---:|---|---:|---:|---:|---:|---|
| 1–2 | QB2_C30LR080 | 30K | 8e−5 | .25 | .5 | R1, R2 |
| 3–4 | QB2_C30LR050 | 30K | 5e−5 | .25 | .5 | R1, R2 |
| 5–6 | QB2_C25 | 25K | 1e−4 | .25 | .5 | R1, R2 |
| 7–8 | QB2_C30AG000 | 30K | 1e−4 | 0 | 0 | R1, R2 |
| 9–10 | QB2_C30G025 | 30K | 1e−4 | .25 | .25 | R1, R2 |

The comparison controls are seed-matched original W112, 30K, LR 1e−4,
α .25, γ .5 Students. R1 is `panda_s1loss9 / QB_A200_E030 / R1`;
R2 is `panda_sensor_refine / QB_BASE / R2` in
`PANDA_S13_SENSOR_REFINE_R34_20261028_v1`. The original initialization and data
stream seeds are preserved. No Student weights or optimizer state are reused.
AG000 is explicitly a **two-weight ablation**. It uses the existing `panda_e`
path with α=γ=0, so the Hard weight is one on the same valid training mask.
The Aligner, input geometry and loss support remain the same.

Every case reuses the authenticated `S1_A200_E030 @44K` Stage 1 donor
(λ_abs=2, λ_epsilon=.03, epsilon radius 3). Its checkpoint SHA is
`7fa4491f0c0e7b8c5d559263c50fff18f84abd7a820852a0d24666928191ed34`.
The architecture remains W112 UBIL, four-band Ilrms, corrected PAN plus
`corrected PAN − original PAN`, Aligner scale 1, RGB reference, Lanczos-4,
β=0, AdamW, warmup 100 and cosine minimum LR 0. Rotation losses, TTA,
SWA/EMA and new/band-weighted losses remain excluded.

The [CSV](cases_s1.csv) and [manifest](experiment_manifest.json) are part of
the source pin. The process-local adapter reuses the existing focus runtime,
calibration, official RR/FR evaluation and Sheet publisher without modifying
any previous campaign files or processes.

## Checks and enqueue

On s1, with the existing environment and authenticated original SENSOR5 Docker
container available:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m panda_qb_refit catalog --csv
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s panda_qb_refit/tests -v
PYTHONDONTWRITEBYTECODE=1 python -m panda_qb_refit.enqueue --wait
```

Keep the enqueue command in a persistent host session. It waits for the existing
focus controller to report `COMPLETED`, including the unchanged WV3 TUNE30 and
HARD36 queues. It authenticates the preceding six QB runs, performs read-only
data/donor checks and cost profiling, then starts the new training container and
the publisher. It never interrupts another trainer. Control files and logs are
under `work_dir/panda_qb_refit/launcher_s1/`; campaign artifacts are under
`work_dir/panda_qb_refit/PANDA_S1_QB_REFIT_FIXEDDONOR_20261030_v1/s1/QB/`.

The QB gate remains unchanged: only complete official 8K, 9K and 10K evaluations,
**all HQNR strictly below .90**, permit an authenticated safe stop. The completed
checkpoint, optimizer, stream and RNG state remain available. Terminal reporting
lists completed and `EARLY_STOPPED_HQNR` runs separately; a safely stopped run is
never represented as having completed its planned horizon.

Sheet publication still targets `QB-ablations`: publish every same-checkpoint
H/E joint pass, plus one minimum-ERGAS representative among HQNR-qualified
checkpoints (otherwise maximum HQNR), without duplicates. Approach/config notes
include each new setting. Completion requires verified terminal Sheet readback.

## Revising or recovering the queue

There is no additional automatic pruning. When the user requests a review, use
the new results to decide what to retain, replace or add. Review and implement a
versioned change at a checkpoint/run boundary. **Do not edit this registry or its
source files while its waiting controller or trainer is active.** The waiting
controller pins both source and registry before waiting and refuses stale launches.

The launcher saves a nonce-labelled Docker intent and a container-ID receipt.
It refuses foreign containers, unreceipted exited containers and environment
changes. A failed container is not restarted automatically. For manual recovery,
first inspect the failure, authenticate the saved container/image/mounts and
unchanged code pin, and verify the runtime's retained exact-resume checkpoint.
If the failure is transient and the same container can resume safely, explicitly
restart that authenticated container and rerun the enqueue command. Do not remove
or overwrite receipts to bypass an identity, source or checkpoint mismatch;
such a change requires a reviewed migration to a new campaign version.
