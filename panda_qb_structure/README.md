# QB Student input and upsampling comparisons, 2026-10-31

Campaign: `PANDA_S13_QB_STRUCTURE_LOCALSTAGE1_20261031_v1`.

QB still misses the joint target **HQNR ≥ .9315 and ERGAS ≤ 3.5**. Lower LR,
25K cosine, removing Hard weights, and reducing gamma did not produce a paired
improvement. All previous mature QB Students used CORRECTED_DIFF + UBIL.
This queue fills the three missing cells of an input × upsampling comparison
and checks the most promising remaining optimizer setting, AdamW weight decay 0.

## Method and local Stage 1

Each server uses its **own independently trained Aligner/Teacher**:

- s1: completed S1LOSS9 `S1_A200_E030`, validation-selected 44K.
- s3: completed QB_DEEPEN `S1_A200_E030`, currently validation-selected 39K.
  The receipt is read from the local completed Stage 1; the code does not impose
  s1's selected step or weights on s3.

No donor transfer is required. Stage 1 is already complete on both servers, so
these Student comparisons do not retrain it and introduce a second changing
variable. Donor, source-data, and saved-model integrity are checked locally.
There are no GF2/WV3 completion prerequisites for the standalone QB launcher,
no waiting-controller code pin, and no additional queue-edit approval rule.

All new Students start fresh: **scale 1, W112, 30K updates, LR 1e-4,
warmup 100/cosine to zero, alpha .25, gamma .5, beta 0, batch 48**.
AdamW decay is .01 except WD000, where it is 0. RGB structure reference,
Lanczos-4 warp, the existing four-band MS inputs, GT Hard loss, and existing
paired input augmentation remain unchanged. Rotation consistency, TTA,
SWA/EMA, soft loss, and new/band-weighted losses are excluded.

Input labels:

- `DIFF`: `[corrected PAN, corrected PAN − original PAN, I_lrms]`, six channels.
- `P0`: `[corrected PAN, I_lrms]`, five channels. Aligner scale remains **1**.
- `UBIL`: existing bilinear + 3×3 convolution upsampling blocks.
- `BASE`: original 2×2 transposed-convolution upsampling blocks.

Downsampling is unchanged. BASE versus UBIL changes the whole upsampling block,
including 125,440 parameters at W112; it is not a pure interpolation-kernel test.
The additional DIFF stem column has 1,008 parameters and starts at zero.
Matched-seed P0/DIFF runs preserve their shared initialization and initially
compute the same function for a given upsampling variant.

## Registered queue

R1/R2/R3 use Student seeds 261018101/261018102/261018103.

| Server | Order | Case | Replicas | Input | Upsampling | Weight decay |
|---|---:|---|---|---|---|---:|
| s1 | 1–2 | QB4_DIFFBASE | R1/R2 | DIFF | BASE | .01 |
| s1 | 3–4 | QB4_P0UBIL | R1/R2 | P0 | UBIL | .01 |
| s1 | 5–6 | QB4_P0BASE | R1/R2 | P0 | BASE | .01 |
| s1 | 7–8 | QB4_WD000 | R1/R2 | DIFF | UBIL | 0 |
| s1 | 9 | QB4_CONTROL | R3 | DIFF | UBIL | .01 |
| s1 | 10 | QB4_WD000 | R3 | DIFF | UBIL | 0 |
| s3 | 1–2 | QB4_P0UBIL | R1/R2 | P0 | UBIL | .01 |
| s3 | 3–4 | QB4_DIFFBASE | R1/R2 | DIFF | BASE | .01 |
| s3 | 5–6 | QB4_P0BASE | R1/R2 | P0 | BASE | .01 |
| s3 | 7 | QB4_CONTROL | R3 | DIFF | UBIL | .01 |
| s3 | 8 | QB4_WD000 | R3 | DIFF | UBIL | 0 |

Total: **18 new Students, 540K updates**, without a wall-time limit. R1/R2
DIFF+UBIL controls already exist on each server; s3 WD0 R1/R2 also already
exist. These are reused as historical controls after the configuration,
initialization, data-stream, and donor equivalence checks. They are not counted
as newly executed runs. Compare within each server because its donor differs.

Fresh full-training calibration tables are also compared numerically against
that server's historical R1/R2 controls. The comparison covers tau_e, population,
edge cutpoints, all conditional ECDF arrays, and physical data/A/T bindings;
it does not rewrite provenance. Results are retained in
`calibration/numeric_parity_*.json` and referenced from each run configuration.
If reference evidence is unavailable or differs, training continues but the
historical-control comparison is explicitly unverified. Do not interpret it as
a clean paired comparison until that discrepancy is resolved.

QB's existing early-stop rule remains: official **8K, 9K, and 10K HQNR all
strictly below .90** permits a saved-state early stop. No 1K screening rule is
added. Early-stopped runs remain explicitly distinguished from 30K completions.

## s1 handoff

The current WV3 run completes before the new QB queue. Its checkpoint, optimizer,
scheduler, sampling position, and RNG state are retained during the controller
handoff. After all new s1 QB runs, remaining WV3 TUNE30 and HARD36 resume with
their original scientific settings.

```bash
python -m panda_qb_structure.priority plan
python -m panda_qb_structure.priority apply
```

`apply` installs the detached local priority watcher. Use the existing watcher
if it is already installed; the command records and validates ownership.
Status: `work_dir/panda_qb_structure/priority_s1/status.json`.

## s3 handoff after Git pull

Use the existing torch-enabled host Python and `gspread/server.txt=s3`.
The completed local QB_DEEPEN Stage 1 and its validation selection files must
remain in `work_dir`. No GF2 closeout receipt or s1 checkpoint archive is needed.

```bash
python -m panda_qb_structure catalog --server s3 --csv
python -m panda_qb_structure verify-donor --server s3
mkdir -p work_dir/panda_qb_structure/launcher_s3
nohup python -u -m panda_qb_structure.enqueue --server s3 --wait \
  > work_dir/panda_qb_structure/launcher_s3/controller.log 2>&1 < /dev/null &
```

`--wait` waits only for local GPU occupancy. The launcher checks the QB dataset
and donor, profiles the actual architectures, runs training preflights, starts
the registered queue, and maintains its publisher. No automatic WV3 resumption
is scheduled on s3. Status is in `launcher_s3/status.json`; each campaign's
state is under `work_dir/panda_qb_structure/<campaign>/<server>/QB/`.

## Sheet and model retention

Both servers publish to **QB-ablations**, with input, upsampling, LR, alpha/gamma,
weight decay, and local Stage 1 provenance in their description/configuration.
Every same-checkpoint joint H/E pass is posted, plus one representative per run:
minimum ERGAS among HQNR-qualified checkpoints, otherwise maximum HQNR.

Every scheduled 1K evaluation retains its checkpoint, including intermediate
goal passes. Complete resume state and frozen Aligner/Teacher provenance remain
available; calibration files are retained alongside the campaign. Completion
is recorded after all registered runs and final Sheet readback are verified.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s panda_qb_structure/tests -v
```

CPU tests cover both server lanes, genuine P0/DIFF and BASE/UBIL models,
paired initialization, actual optimizer updates and exact resume, local donor
provenance, publication metadata, container ownership, and the WV3 handoff.
The Docker runtime performs additional actual-data preflight before training.

Evidence and detailed interpretation:
[QB refresh plan](../results_log/2026-10-31_qb-structure-refresh-plan.md).
