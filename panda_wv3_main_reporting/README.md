# WV3 RGB main reporting extension

The training release `aa63715e4e14edced38199d7ae9b112189d59debd9c0d920c095730e79e6cb8c`
and all original model/optimizer/stream/resume checks remain unchanged. This
entry point replaces only `panda_wv3_main_rgb.publication.publish` in the running
process. Do not run a second publisher or restart the original entry point after
the Sheet has migrated to rounded values: its full-precision row verifier will
correctly reject the new display representation.

Run in the existing pinned PAN-Crafter image, with the same mounts/environment:

```bash
python -m panda_wv3_main_reporting run --server s1 --device cuda:0
```

For a read-only publication check or a manual backfill while training is stopped:

```bash
python -m panda_wv3_main_reporting publish
python -m panda_wv3_main_reporting publish --apply
```

If `reporting/cost.json` is absent, stop the trainer gracefully, verify that no
other CUDA compute process is active, and use the same image with `--pid=host`:

```bash
python -m panda_wv3_main_reporting profile --case M0_OURS --repeat 1 --step 24000
```

The checkpoint must already exist. Profiling refuses concurrent GPU work.
Resume training with the reporting entry point immediately afterward. This
campaign and this profile are s1-specific; do not copy GPU cost values to s3.

## Sheet policy and units

- Actual numeric values in D:Y are rounded to three decimal places, with number
  format `0.000`. Raw immutable evaluations and metric copies in row notes retain
  full precision. Selection and pass/fail remain based on the raw values.
- IDs, steps, seeds, LR, and configuration parameters retain their original
  precision. No foreign rows, columns beyond BL, or unrelated formatting are
  changed by this publisher.
- Infer(ms): **milliseconds per image**, RR PAN 256×256, batch 1, FP32, no TTA.
  Aligner + Lanczos-4 + correction residual + interp23 + Student are timed using
  synchronized wall time, 10 warmups and 50 repetitions, reporting the median.
  Inputs already reside on the GPU. Data reading, normalization/H2D, evaluation
  metrics, frozen training Teacher and optimizer are excluded. The full native
  inference methods are the same functions used by the actual main model.
- One actual evaluated A+S checkpoint supplies the architecture cost. All 18
  recipes have the same deployed graph; LR/loss/horizon/repeat do not alter it.
  Shared architecture cost is explicitly identified, not represented as 18
  independently benchmarked checkpoints. FR 512×512 cost is also measured and
  retained in the profile and each measured row's notes.
- Mem(MB): the fixed schema's legacy header means **peak allocated MiB**
  (`bytes / 2^20`), including model, resident inputs, output and workspace.
  This is not CUDA reserved memory, process RSS or the training peak.
- FLOPs(G): **partial arithmetic GFLOPs estimate, 1 MAC = 2 FLOPs**. Conv/MatMul
  and the stated resampling estimates are covered; normalization, activation,
  indexing, interpolation-weight generation and other listed operations are
  omitted. Legacy TUNE30 used GMAC under this header; do not compare its raw
  unconverted values directly. The copied historical row in WV3-main was
  separately converted from 71.790663744 GMAC to 143.581327488 GFLOPs and its raw
  value retained in the cell note. Original TUNE30 logs are unchanged.
- Train(h): sum of committed Stage2 optimizer-step durations through the selected
  checkpoint. Eval(h): sum of all scheduled official evaluations through that
  checkpoint, rather than just one evaluation. Wall(h): elapsed time from the
  first optimizer step to the selected evaluation record, including I/O and any
  pause/downtime. Stage1, initial calibration and setup before the first optimizer
  step are excluded. These are not whole-method training time or necessarily the
  completed run's final time.

Evidence and publication/readback receipts are stored under the existing main
campaign's `reporting/` directory. The original `publication/last_poll.json` also
points to the latest successful reporting receipt for existing monitoring.

Validation:

```bash
python -m unittest panda_wv3_main_reporting.test_reporting -v
python -m panda_wv3_main_rgb.release
```
