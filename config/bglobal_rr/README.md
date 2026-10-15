# bGlobal RR — method implementation, not an experiment launch

Scope: `research_log/PANDA_bGlobal_RR_Implementation_Changes_2026-10-13 (1).md`
and the user's subsequent decisions: no FR training loss; provided LMS; QB
original-distribution LMS. No new case/queue, training launch, Sheet write or
checkpoint selection change is included. HQNR-first selection remains unchanged.

## Implemented path

| Component | Entry | Contract |
|---|---|---|
| Prior preparation | `teacher48.prepare_global_prior` | Every registered train PAN64/LMS64 row; no GT/MS or evaluation reads |
| Frozen prior | `panda_method.global_prior.GlobalPriorAsset` | Dataset/sensor-specific signed `(dy,dx)`, source/settings/report hashes, reviewed confidence gates |
| Actual H5 adapter | `panda_method.global_data.GlobalRRDataSources` | Stored LMS without reinterpolation/clipping; explicit source identity, no scene mapping |
| Aligner / stages | `panda_method.global_rr` | Same LN common ResBlock and 66→32→2 head; raw signed b; incompatible checkpoint identity |
| RR update engine | `teacher48.global_training` | One RR batch, `rec + abs_rr + .001*eps_rr`; caller supplies architecture, optimizer and spatial policy |

The old `teacher48` RR+FR runner and scene-aware method are intentionally kept
for old experiments. They are **not** an entrypoint for the new method. The new
engine is an additive integration API; its checkpoint includes A/T, epsilon RNG,
optimizer state and ordered parameter-name binding. A future experiment runner
must also save its data-order/augmentation state, scheduler and any global RNG
states. This small engine does not claim complete campaign restart by itself.

PAN sampling convention is `aligned[y,x] = PAN[y+dy,x+dx]`. Match inputs are
unchanged PAN64 and the band mean of provided LMS64. Historical estimator
fields named `*_lr_raw` are therefore **RR PAN pixels in this adapter**, not
16×16 MS pixels; no sign reversal or factor four is applied. Scharr/ZNCC plus
census uses the existing `align.estimator` with exact settings/code hashes.

Aggregation uses weighted 2D translation consensus, then a **component-wise
weighted median**, not an average of magnitudes or an unnamed geometric median.
Every row, raw estimate, rejection reason, confidence weight, consensus result
and source hash is retained in the proposal. Invalid vectors have zero weight,
not a replacement zero displacement. Consensus examines all candidates (O(N²));
there is no hidden subset/cap. Confidence is a diagnostic score, not a probability.

Online augmentation applies H flip → V flip → CCW rot90 to all observations and
the same frozen b. Unknown offline augmentation is recorded as unknown; no
source-scene identity or reverse augmentation is fabricated. An isotropic or
multimodal aggregate may legitimately fail the direction/near-zero gates.
This prior describes the **stored training distribution**, not verified physical
sensor calibration.

Stage 1 makes two A forwards (native/epsilon) and one native Teacher forward.
Native/epsilon share b; the native correction is detached only in epsilon's
target. Radius-2 offsets are uniform in disk area. Stage 2 retains one shared
frozen A/T, complete train-pixel e/q median calibration and Student-only updates.
The new semantic checkpoint guards reject old scene-prior weights even when
their tensor shapes happen to match.

## Source policy, especially QB

`*_train_source.json` register the original train H5's PAN/LMS. Source bytes and
row count are checked, not inferred from file names. These are registrations,
**not measured prior assets**.

| Sensor | Prior / Aligner LMS | Reconstruction LRMS |
|---|---|---|
| WV3 | Existing original stored LMS | Unchanged registered MS |
| QB | **Original `train_qb.h5` / `valid_qb.h5` LMS** | Existing `*_qb_msfix.h5` MS |
| GF2 | Existing original stored LMS | Unchanged registered MS |

QB `qb_original_lms.json` supplies explicit `splits` overrides. The two sources
are hashed separately; actual PAN and GT values for every consumed row must be
identical before the original LMS is used. No original MS is silently restored.
Finite negative LMS interpolation lobes are preserved through `2*DN/peak-1`.
Original LMS is **not** asserted equal to an interpolation of repaired MS.

Aligner reference and reconstruction base are separate choices. The adapter
always supplies stored LMS; it does not change an explicitly selected
`BICUBIC_MS` U-Net base to `NATIVE_LMS`. The current data-to-Teacher builder
supports P0; PL/PH/PLH fail at build time until a separately pinned LPAN adapter
is supplied. The method core itself supports explicitly supplied LPAN.

## Preparation API (requires numerical decisions first)

There are **no approved real-data matcher/consensus thresholds yet**, and no
WV3/QB/GF2 frozen b has been generated. Do not copy synthetic test settings into
production. `panda_method.global_prior.SETTINGS_KEYS` specifies the complete
required policy:

```text
search_int, max_magnitude, min_peak_margin, max_primary_secondary_diff,
edge_top_frac, match_window (=64), min_reference_std, min_pan_std,
min_peak_zncc, min_valid_candidates, min_valid_fraction, consensus_tolerance,
min_inlier_fraction, min_direction_concentration, max_dispersion,
min_confidence, min_prior_magnitude, rationale
```

All thresholds are explicit; nonfinite/missing settings or a zero/near-zero b
fail instead of choosing a fallback axis. Given an agreed policy, preparation is:

```bash
python tools/panda_bglobal_prior.py propose \
  --source config/bglobal_rr/QB_train_source.json \
  --settings /path/to/reviewed-qb-settings.json --root . \
  --output /path/to/new-qb-proposal.json

python tools/panda_bglobal_prior.py freeze \
  --proposal /path/to/new-qb-proposal.json \
  --review /path/to/proposal-review.json \
  --output /path/to/new-qb-frozen.json
```

The review JSON records schema `PANDA_B_GLOBAL_RR_POLICY_REVIEW_V1`,
`accepted: true`, `reviewed_by`, `rationale`, `settings_sha256` and
`proposal_sha256`. This is provenance for a numerical decision, **not a server
lock or external approval service**. Proposal and settings hashes are canonical
JSON SHA256; frozen `asset.fingerprint()` is also canonical, not a pretty-printed
file-byte SHA. Outputs never overwrite existing files. Freeze recomputes the
reported aggregation; changed implementation/report requires regeneration.

Example API wiring after a real prior is prepared (not runnable defaults):

```python
import json
from panda_method.global_prior import GlobalPriorAsset
from panda_method.global_data import GlobalRRDataSources

asset = GlobalPriorAsset.from_file(frozen_path, sensor="QB")
override = json.loads(open("config/bglobal_rr/qb_original_lms.json").read())["splits"]
sources = GlobalRRDataSources("ablr2/sensor_sources.json", root=".",
    sensor="QB", prior_asset=asset, lms_overrides=override)
rr = sources.batch("train", [0, 1], augmentation=(True, False, 1))
# Supply explicit Teacher/spatial/seed policies to build_global_rr_stage1;
# construct optimizer and GlobalRRTrainer, then trainer.step(rr).
```

## FR loss versus FR evaluation

FR **training loss is removed**, and FR data cannot reach the new optimizer.
Existing official RR/FR metric formulas, reference images, HQNR/SCC/ERGAS
priority, clipping and mask conventions were not changed by this implementation.

RR256 inference is whole-image, one residual shift per image, no automatic
64-tile decomposition. It uses the same RR pixel units as training.

New-method FR512 inference still needs its **input b policy**, independently of
removing FR loss. Specification §7.2 forbids silently supplying either b_RR or
4*b_RR; scaling would also change FC1's raw-b conditioning distribution. Until
that separate decision is made, the new adapter/model rejects FR inference.
This does **not** delete or disable official FR evaluation of existing models.
It does mean the new method is not yet ready for an HQNR-selected campaign.

## Remaining decisions, not silently filled in

1. Train matcher/consensus acceptance settings, diagnostic review of the full
   distribution, and the zero/near-zero policy. Current handling is rejection.
2. Spatial ROI / Aligner view margin / boundary policy for an actual run.
   These remain explicit; no shift-dependent loss mask is invented.
3. FR inference b coordinate/conditioning policy before the new model's HQNR
   evaluation. No decision about FR loss remains.
4. Later case definitions: Teacher/Student capacity, P0 vs PLH, base, optimizer,
   augmentation sampling, schedule, data-order resume and evaluation/upload.
   These are deliberately not new experiments in this method-only change.

See `results_log/2026-10-13_bglobal-rr-implementation-review.md` for the verification
record and source audit. CPU synthetic correctness tests do not establish the
real prior value, improved alignment, GPU stability or quality improvement.
