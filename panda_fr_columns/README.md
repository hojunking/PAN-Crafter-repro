# Corrected-PAN FR columns (reporting only)

The current EQ40 publications use one row per selected checkpoint. The same
prediction, checkpoint and FR20 evaluation have two reference triplets:

| Physical columns | Meaning |
|---|---|
| D:F | HQNR, D_s, D_lambda against original PAN (official comparison/selection) |
| G | Original-PAN JQM, preserved |
| H:J | HQNR, D_s, D_lambda against corrected PAN (diagnostic comparison) |
| K:R | Existing RR metrics, shifted right without changing values |
| S:BO | Existing cost, configuration and provenance, shifted right |

Numbers are displayed to four decimal places. H-cell notes retain the original
precision, checkpoint SHA, step and evaluation-file SHA. D_lambda must agree
between both references because reconstruction and LMS are unchanged. Missing
evaluations remain blank, never zero. Historical models without corrected-PAN
evaluation are not assigned invented values. Legacy EQ40 corrected companion
rows remain intact and are hidden after their original row is populated.

This package does not modify any frozen training/evaluation/configuration file
or source manifest. It installs a process-local reporting adapter. Model,
optimizer, sampler, RNG, seeds, selection and evaluation are unchanged. Existing
fixed64 publishers see a reversible logical layout and write only A:G + K:BO;
they cannot overwrite H:J. EQ40 publishers write both triplets atomically when
the selected checkpoint changes, then verify readback and provenance.

## Launch after pulling the complete commit

Use these launchers instead of the old package's direct enqueue command:

```bash
# s1 WV3: graceful resume of the current EQ40 experiment
python3 -m panda_fr_columns.enqueue --server s1

# s2 QB: local RGBN donor, two existing EQ40 Stage2 cases
python3 -m panda_fr_columns.enqueue --server s2

# s3 GF2: existing fresh unified Stage1 x2 + EQ40 Stage2 x2 plan
python3 -m panda_fr_columns.enqueue --server s3 --wait
```

`--check-only` validates sources/data/environment/Sheet without stopping work or
migrating the header. The first actual run atomically inserts H:J in its own
dataset tab if still fixed64, backs up the native grid, and verifies preservation
of existing values. Repeated starts detect FR67 and do not insert again. Unknown
headers are rejected. QB/GF2 headers are not changed merely by pulling code.

Only the exact same-package old container is checkpoint-stopped automatically.
The s2 launcher also retains the existing coordinator's targeted original-QB
continuation. Unrelated GPU jobs are never stopped; `--wait` queues behind them.
Already-stopped containers/results are preserved. A stopped FR67 container is
resumed automatically after verifying both training and reporting identities.

**Shared WV3 tab:** s3 Table3 also writes WV3-ablations. If it is still active,
resume that same Table3 campaign through the adapter after pulling:

```bash
python3 -m panda_fr_columns.enqueue --package panda_wv3_tab3_rgbn --server s3
```

This only adapts its existing columns; Table3 does not have dual-PAN evidence,
so H:J remain blank. Until restarted, an old publisher rejects the new header;
its local evaluation records remain available. Do not restart legacy direct
uploaders against an already migrated tab.

Publication backups and receipts are under each campaign's
`publication/fr_columns/`. The initial s1 migration also has a local backup in
`work_dir/panda_fr_columns/`. No credentials or weights are committed.

```bash
python3 -m unittest panda_fr_columns.test_reporting -v
```
