# Corrected-PAN HQNR selection — 2026-11-14

The user now prioritizes corrected-PAN HQNR. This control/reporting overlay
resumes the existing ZEROQ campaign and preserves its frozen numerical source,
configs, donor receipts, optimizer, LR schedule, sampler/RNG and checkpoint IDs.
It does not retrain completed runs or transfer weights between servers.

## Current Stage2 policy

| Sensor | Corrected-PAN HQNR minimum | RR ERGAS maximum | PSNR goal |
|---|---:|---:|---:|
| WV3 | 0.9585 | 2.040 | 38.0 |
| QB | 0.9315 | 3.500 | No floor |
| GF2 | 0.9645 | 0.522 | 45.08 |

Selection uses the existing rule, with corrected HQNR substituted: among
HQNR-qualified checkpoints choose minimum ERGAS, then maximum corrected HQNR,
then earliest step. If none qualifies, choose maximum corrected HQNR. Publish
all corrected-HQNR/ERGAS passing steps plus the representative and fixed40K,
deduplicated. GF2 retains its separately annotated PSNR goal. WV3 PSNR38 is an
existing donor/quality goal, not a new filter silently added to the H/E uploader.

Physical Sheet columns stay D:F original PAN, H:J corrected PAN, K:R RR.
Original metrics remain available as reference. Notes/labels explicitly say
corrected-PAN selection. Thresholds use unrounded evidence, not displayed cells.
Missing corrected evaluation is an error, never an original-PAN fallback.
Historical campaigns without paired evaluation remain historical records.

For QB, stop at10K only if **corrected** HQNR at8K/9K/10K is strictly below .90
in all three. Before restarting an already-stopped run, the launcher verifies
all three local evaluation files. If the corrected rule also fails it remains
stopped; otherwise it resumes the exact complete stopped checkpoint. The
previous decision is retained in `corrected_hqnr_stop_review.json`.

Old configs/evaluation fields named `official_reference` and `joint_pass`
describe the historical original-PAN protocol. They are retained for provenance.
The authoritative current control policy is `corrected_hqnr_policy.json`, with
explicit selection reference in Sheet notes and future evaluations. Model and
metric functions do not change. Corrected-PAN HQNR is a different evaluation
reference from published original-PAN benchmarks; describe that reference in
tables. Existing selected Stage1 donors and Stage1-only ablation selectors are
preserved; this switch does not silently reselect or retrain those donors.

## Each server, after pull

```bash
python3 -m panda_corrected_hqnr.enqueue --server s1
python3 -m panda_corrected_hqnr.enqueue --server s2
python3 -m panda_corrected_hqnr.enqueue --server s3 --wait
```

Use the command for the physical server only. `--check-only` is read-only.
The old same-server ZEROQ process is gracefully checkpoint-stopped and replaced
by a `-corrected` container sharing the same campaign directory. Completed R1
is only re-published; an unfinished R2 resumes. Unrelated work is not stopped.
GF2's existing/new Stage1 preparation remains under the original coordinator.
This package has its own manifest; frozen training release hashes stay unchanged.

```bash
python -m unittest panda_corrected_hqnr.test_policy -v
```
