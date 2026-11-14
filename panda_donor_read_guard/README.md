# Donor read-time compatibility guard

On 2026-11-14, the s1 WV3 five-case Stage 2 ablation stopped before
`B_ALIGN_L1` repeat 2. Six of fifteen result slots were complete (including
the explicitly reused Ours repeat 1); nine training runs remained.

The pinned donor SHA-256 still matched. The frozen loader compared the
entire `stat()` object before and after reading, including access time.
Linux `relatime` refreshed access time during this read, producing a
false `Donor checkpoint changed while loading` exception. An aged-file
test on this filesystem reproduces that behavior.

This separate IO overlay verifies the donor SHA-256 and stable identity
(device, inode, size, modification time and change time), reads the file
before the legacy snapshot to refresh access time, then invokes the
unchanged loader with all its original checkpoint and model checks.
An access-time-only legacy failure is eligible for one retry; other
exceptions, content changes and file replacements remain errors.

The scientific source manifests, checkpoint pins, calibration, seeds,
optimizer state, model, losses and schedule remain unchanged. The overlay
has its own source hash recorded in the Docker label and
`donor_read_guard/active.json`; successful loads are logged in
`donor_read_guard/loads.jsonl`. Existing completed results are retained.

Resume the s1 ablation:

```bash
python3 -m panda_donor_read_guard.enqueue --package panda_wv3_s2abl5 --server s1
```

The original container must be stopped. The launcher authenticates its
checkout, campaign, server, command and source pins, runs the original
source and Sheet checks, and starts a separate `-readguard` container on
the same campaign state. It never stops another experiment automatically.
The original stopped container is retained for diagnosis.

The same wrapper supports `--package panda_sensor_main50 --server s2`
or `s3` if that campaign needs this IO fix. This does not launch a remote
experiment from s1.

Verification:

```bash
python3 -m unittest panda_donor_read_guard.test_guard -v
```
