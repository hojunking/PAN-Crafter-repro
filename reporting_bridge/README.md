# B01 normalized Sheets bridge

Reporting only: reads existing sealed B01 artifacts, verifies them, and publishes
normalized native observations and separate stress curves. It never trains,
selects a new checkpoint, recomputes image metrics, starts a controller, or
modifies original experiment artifacts. Importing the modules does not access
credentials or Google Sheets. The package and `tools/rb_b01_sheets.py` are outside
the B01 numerical source-identity globs.

## Environment and ownership

Use an **existing** Python environment containing `gspread` and `PyYAML` (PyYAML
is needed only for local collection). Do not upgrade the experiment environment.
On the current s1, this is `/home/knuvi/miniconda3/envs/pancrafter/bin/python`.
System Python may resolve the repository's `gspread/` directory rather than the
installed SDK; that is not an authenticated Sheets client.

Authentication uses the existing `gspread/account.json`, lazily and only for live
commands. Never copy this file into an evidence package, commit it, or place it
in a Sheet. The spreadsheet is opened by its pinned ID, never by title.

Only the coordinating **s1** may run `setup --apply` and `upload --apply`. The
local B01 owner record is checked as well as the CLI writer argument. This is a
single-writer operating contract, not a claim of a distributed Google lock.
Confirm there is no other structural Sheet writer before applying setup.

## Original servers: collect, do not upload

On each original s1/s3/s5, first run the existing `panda_rb_runner.py report`
from that experiment's **frozen source and pinned Docker**, with the original
repository root and a new output path under `work_dir/_rb_sheet_upload/`.
Mount original experiment assets read-only; do not pull into a running frozen
release. The report command hashes existing results; it does not run inference.

Then use the new reporting code outside that immutable runtime:

```bash
BRIDGE_PYTHON=/path/to/existing/python
"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py collect \
  --server s3 \
  --frozen-report /actual/path/to/B01_s3_report_readback.json \
  --output-dir work_dir/_rb_sheet_upload/B01/outgoing
```

Change `s3` only to the **actual original server**. Collection independently
checks the immutable release and original artifact validator, config, selection
manifest, checkpoint file hashes, original data/map identity, native CSV and
curve evidence. The original aggregate report is cross-checked when supplied;
its changing creation time is not part of per-observation identity.

Transfer only the produced `s3.evidence.json` to the s1 writer's `incoming/`
folder using the already approved transfer route. It contains reports and
verification provenance, not model weights, imagery or credentials. Keep one
current verified package per source server. Missing packages mean **unknown /
not collected**, not proof that an experiment did not run.

## s1: explicit inspect → prepare → plan → setup → upload → verify

```bash
BRIDGE_PYTHON=/home/knuvi/miniconda3/envs/pancrafter/bin/python
OUT=work_dir/_rb_sheet_upload/B01

"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py inspect --output-dir "$OUT/inspect"
"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py prepare \
  --evidence-root "$OUT/incoming" --output-dir "$OUT/prepared"
"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py plan \
  --prepared "$OUT/prepared" --snapshot "$OUT/inspect" --output-dir "$OUT/plan"

# Review plan.json. Only these two commands can write to Sheets.
"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py setup \
  --plan "$OUT/plan" --output-dir "$OUT/setup" --apply
"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py upload \
  --prepared "$OUT/prepared" --output-dir "$OUT/upload_dry_run"
"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py upload \
  --prepared "$OUT/prepared" --output-dir "$OUT/receipts" --apply

"$BRIDGE_PYTHON" -B tools/rb_b01_sheets.py verify \
  --prepared "$OUT/prepared" --snapshot "$OUT/inspect" --output-dir "$OUT/verification"
```

Without `--apply`, setup/upload are dry-runs. Inspect/prepare/plan/verify reject
`--apply`. Content-addressed immutable payloads, full append-only receipt history,
successful-stage events, and latest error/receipt JSON remain in
the reporting workspace. They never overwrite `panda_rb`'s `sheets_uploaded`
field or original training status.

Prepared input must be **cumulative**: keep the existing s1 package when adding
s3 or s5. Upload refuses to replace a public summary with a subset that omits
already registered observations. Identical payload retries create zero new
observations; a different metric/checkpoint/evaluator occupying the same logical
slot is a conflict, not another seed. Do not delete old source rows to bypass it.

## Data and display contract

- `_rb01_s1/s3/s5`: 64-column canonical native sources, 16 fixed slots each.
  Two selectors remain two observations even when their checkpoint is an alias.
- `_records!A2`: only its final return expression is extended. The legacy parser
  and existing `WV3-main!A5/A6` formulas are preserved.
- `_rb02_points`: 43-column source, 49 points × two inference modes × up to 24
  Students. Fixed192 stress metrics never enter native RR/FR metric columns.
- `_rb_b01_status`: all 24 registered Students, expected and measured counts
  separate; absent, invalid, completed, and delivery states are distinct.
- `RB02-curves` and `RB-B01`: new presentation surfaces. Existing `paper`,
  `ablations`, and `유의미한결과` are not edited.

Raw values retain numeric precision and nulls remain blank. All descriptive
strings use RAW/string cells, never formula interpretation. Statistics use
Student sample SD (`ddof=1`), not scenes/directions/aliases as extra samples.
Within a radius, directions are averaged within each Student first. Source,
runtime, common F1, data and evaluator mismatch produces separate cohorts.
Invalid geometry/numerical failures are retained and surfaced; a processed
49-point curve is not automatically clean evidence.

Snapshot preserves the original formula (including note), protected tables and
ID-indexed observations. Setup rechecks before changing A2 and rolls back only
its own unchanged patch when regression is detected. This is not atomic CAS;
concurrent human edits trigger review instead of overwrite.

## Tests

```bash
python3 -B -m unittest discover -s reporting_bridge/tests -v
```

Tests are synthetic and must never be uploaded. Local tests, source readback,
formula-view readback, and complete six-seed campaign coverage are separate
claims; consult the actual receipts for each.
