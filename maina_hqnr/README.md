# MAIN-A WV3 coefficient campaign — s4 / s5

Plan: `research_log/PANDA_WV3_TableA_S45_HQNR_Continuous_ExperimentPlan_2026-09-25.md`.
This package adds an independent campaign; the existing FH12/FH20R1 numerical code,
s1–s3 experiments, historical G23 runs and representative Sheet tabs are unchanged.

## Start after pulling the implementation commit

Run on the actual target server, with the existing `pancrafter` Python environment
and Docker installed. The host environment reads/checks assets; production training
uses the original pinned Docker image, not the host's Torch installation.

```bash
# s4 only
bash tools/maina_hqnr_docker_start.sh --server s4 --gpu 0

# s5 only
bash tools/maina_hqnr_docker_start.sh --server s5 --gpu 0
```

These are local commands, not remote dispatch. Neither server waits for the other.
`--dry-run` validates local assets and prints the Docker command without stopping
the old campaign, creating a runtime checkout, or starting training.

The launcher validates F1/data first, uses a committed frozen checkout, requests the
old local G23 campaign's supported safe stop, verifies full-state preservation and
GPU release, then launches MAIN-A. If safe-update stop is unsupported, it waits for
the current **run** boundary, not the entire cycle. Unknown owners/watchdogs or
unverified mounts cause a safe pause; no `pkill`, guessed PID, or forced kill is used.
The old stop markers remain set so the old queue cannot silently restart.

The new service performs actual-GPU numerical and save/resume acceptance automatically.
`START_REQUESTED` is not evidence of training: `meta/training_start_receipt.json`
and a positive actual update count are required. No manual acceptance token is needed.

The image must already be local and resolve to:
`sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887`.
No implicit image pull or environment substitution occurs.

## Required original F1 assets

Git transfers code/specifications, **not** checkpoints, H5 data, calibration or LP caches.
Both s4 and s5 use the original **s1 F1**, never F4/F5 or a newly trained Teacher.
The default bridge is:
`work_dir/_fh20r1/s1/imported_refs/F1/bridge_manifest.json`.

Copy its original referenced files byte-for-byte if they are absent on a target.
The ten artifact keys are `teacher_checkpoint`, `teacher_training_state`,
`teacher_config`, `teacher_checkpoint_identity`, `calibration_path`, `q_cache_path`,
`dataset_manifest_path`, `lpan_manifest_path`, `origin_manifest`, and `init_manifest`.
The original WV3 train/val/RR/FR native H5 and four LP H5 files are also required.
Canonical JSON/YAML files must **not** be edited to relocate paths: that changes
their prescribed hashes. Use a separate path-only map when necessary:

```json
{
  "bridge_path": "/local/copied/F1/bridge_manifest.json",
  "artifacts": {
    "teacher_checkpoint": "/local/copied/F1/model.safetensors"
  },
  "data": {
    "fr": {
      "dataroot": "/local/data/test_wv3_OrigScale_multiExm1.h5",
      "lpan_path": "/local/data/original_fr_lpan.h5"
    }
  }
}
```

Only changed paths need overrides; all resolved files must match the original hashes.
Supply the map as `--asset-map /absolute/path/map.json` to the launcher. Missing or
different bytes pause before the old experiment is asked to stop. Nothing is
recalibrated, regenerated, downloaded, or substituted. External DLPan can be located
with `PANCRAFTER_DLPAN`; its original MTF source hash is checked separately.

## Preserved experiment semantics

- Fixed PLH/W104/D122 Student, original F1 P0/W112/D123, FP32, batch48, 50,000 actual updates.
- BASE plus six one-factor α/β/λE variants; same actual initialization and full sample/view
  stream within a cycle, independent optimizer/scheduler per run. No trained BASE reuse.
- Fresh deterministic nonoverlapping server seeds and rotated case order. Unlimited cycles;
  s4 cycle0/73101 is an historical anchor excluded from fresh-seed aggregates.
- All 50 A/U candidates retained at 1010, 2020, …, 49490, 50000.
- Primary `HQNR_MAX50`: all20 original full-frame FR scenes, raw PAN, mean per-scene HQNR,
  full-precision maximum, lower step on an exact tie. No shifted/masked support, ERGAS tie,
  or target threshold. Secondary `EXACT_50000` shares evaluation if checkpoint SHA matches.
- Selection is explicitly **FR20 test-aware**, not an independent test-set claim. Native
  RR20 data are hash-bound; paper-scene correspondence stays `PAPERSET_IDENTITY_UNVERIFIED`.
- Selected RR/FR metrics include HQNR, Dλ, Ds, ERGAS, SAM, PSNR, SCC, SSIM, Q8, RMSE, CC,
  and JQM. Original main evaluation kernels/cropping are reused; JQM is explicitly labeled
  SRF-substitute, not claimed SIPSA-equivalent. Candidate/JQM evaluation does not update weights.

## Output, Sheets and recovery

All local state lives under `work_dir/maina_hqnr/<server>/`. Each run has immutable
`attemptNNN/config.json`, full-state `last/`, all `candidates/`, and `official/` evidence.
No completion is accepted until all50 candidate evaluations, exact50K full-state
evidence, and selected full20-scene RR/FR metrics are verified.

Dedicated tabs are `SENS-MAINA-WV3-s4` and `SENS-MAINA-WV3-s5`, with a separately owned
MAIN-A section in `Hyper parameter experiements all`. To avoid concurrent row-overwrite
races, only s4 creates the section's live sorted projection of both raw tabs; s5 only
reads that projection. Neither server waits for the other to train. If s4 has not yet
created the section, s5 raw rows are retained and collection readback stays pending.
Keep the live projection's spill area free of manual entries; a collision or changed
formula is reported rather than overwritten. Numeric values retain full
precision and display four decimals. Upload/readback failures remain in a local
durable outbox and never cause retraining. `--no-upload` retains the same local evidence.
No timing/FLOPs number from another experiment is substituted into unmeasured cells.

Useful commands (use the campaign's frozen runtime/image for evaluation or upload
recovery, so source and package identities remain unchanged):

```bash
python tools/maina_hqnr_runner.py preview --server s4 --cycle 1
python tools/maina_hqnr_runner.py status --server s4
python tools/maina_hqnr_runner.py stop --server s4 --safe-now
python tools/maina_hqnr_runner.py retry --server s4 --phase postrun
python tools/maina_hqnr_runner.py upload --server s4
python tools/maina_hqnr_runner.py report --server s4
```

Safe stop preserves optimizer/scheduler/RNG/sampler. Stop markers are never silently
cleared: inspect the local status and explicitly remove only the intended marker
before resuming with the launcher. `retry --phase postrun` reuses trained candidates;
`retry --phase train` is only for an explicitly acknowledged technical failure and
creates a new attempt with the same case/seed. BASE failures pause the lane; a variant
has at most two automatic technical retries, and numerical divergence never rerolls a seed.

`reporting/paired_results.json` separates anchors, failed/partial blocks, same-cycle BASE
deltas, and balanced low/default/high seed sets. Three/six fresh completed blocks trigger
summary readiness only, not early stopping. No cross-server absolute performance ranking.

## Local tests (no production launch)

```bash
python -m unittest discover -s maina_hqnr -p 'test_*.py'
```

CPU tests are not production GPU acceptance. The launch-time CUDA check must still
pass on each target GPU; the implementation commit itself does not start any experiment.
