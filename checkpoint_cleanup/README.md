# Explicit checkpoint retirement (s1, 2026-11-06)

This package implements the user's one-time cleanup request. It does not change
the running experiment's pinned training or publication sources.

For **s2/s3**, start with the [server handoff](../results_log/2026-11-06_s2-s3-weight-cleanup-handoff.md)
and `python -m checkpoint_cleanup.handoff --server s2` (or `s3`). This read-only
command inventories local campaign scopes and checks s2's four required source
weights. It does not authorize individual paths or delete files. The `prepare`
command below is **s1-specific** and requires its local audit; never reuse s1's
manifest on another server. Local manifests and deletion journals are not in Git.

## Scope and evidence

- Completed or explicitly HQNR-early-stopped main/QB Stage 2 runs with no
  checkpoint meeting both dataset targets: remove their weights.
- WV3 target: HQNR >= 0.9585 and ERGAS <= 2.040. QB target: HQNR >= 0.9315
  and ERGAS <= 3.5. Selection uses unrounded metrics at the same checkpoint.
- Completed 30K Stage 2 component ablations: remove checkpoints <= 20K.
- A 50K ablation cutoff of <= 30K is reasonable for late-checkpoint comparisons,
  provided selected donors and paper sources are protected first. No local 50K
  ablation weights were included. This tool does not prune Stage 1 or s3 files.
- Preserve Stage 1, successful main runs, active/incomplete training, explicitly
  archived paper sources/copies, configurations, evaluations, sidecars and logs.

The exact deletion set and SHA-256 checks are recorded in
`work_dir/_operator_storage/20261106_target_cleanup_v1/manifest.json`.
The append-only `events.jsonl` records each unlink; `applied.json` and
`verified.json` report completion. A removed model cannot be loaded or resumed
from these records. Existing checkpoint sidecars describe historical artifacts,
not present-day availability; consult the retirement ledger.

## Commands

Run from the repository root with the pancrafter Python environment. The prepared
manifest is immutable. Do not prepare it again or edit it to widen the scope.

```bash
python -m checkpoint_cleanup project --family panda_wv3_main_rgb --preview
python -m checkpoint_cleanup confirm-prepublication
python -m checkpoint_cleanup apply --apply
python -m checkpoint_cleanup project --family panda_wv3_main_rgb
python -m checkpoint_cleanup verify
```

Pre- and post-deletion projections must be generated for every family listed in
the manifest. Run each family in a fresh process because legacy adapters are
process-wide. Verification compares every publication cell except notes, where
retirement information is added. File and retained-evidence hashes are checked
before deleting; uncommitted missing files or changed evidence stop execution.

For future republication of a retired campaign, use its
`CHECKPOINT_RETIREMENT.json` command, for example:

```bash
python -m checkpoint_cleanup publish --family panda_wv3_main_rgb --apply
```

This wrapper accepts only missing checkpoints authenticated by the manifest and
committed deletion journal. It retains evaluation/selection checks and explicitly
reports that weights are unavailable. Original entrypoints requiring all weight
files cannot republish these retired campaigns. Sheet metrics and representatives
remain historical results and are not recalculated from the remaining weights.

```bash
python -m unittest discover -s checkpoint_cleanup/tests -v
```
