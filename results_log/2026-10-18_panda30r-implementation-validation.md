# PANDA30R implementation / pre-launch validation — 2026-10-18

Scope: requested s1/s3/s4 WV3 local-donor/schedule campaigns, each wall30h.
Plan: `research_log/PANDA_S134_LocalDonor_Schedule_30H_2026-10-18.md`.
Implementation is additive under `panda30r/` plus two launchers. Existing campaign,
metric, data, model and sheet-schema files were not modified.

## Evidence actually checked

- `python -m panda30r smoke`: **99 tests passed**, 46.296 seconds, CPU/synthetic data.
- `python -m unittest discover -s panda_stage2_condq/tests -t . -q`:
  **93 existing tests passed**, 54.753 seconds.
- Shell syntax check on both new launchers: passed.
- `git diff --cached --check`: passed after EOF whitespace cleanup.
- Actual read-only `panda30r check --server s1`: passed. WV3 train count9714,
  registered train catalog/data checks plus validation/RR/FR access and evaluator pins.
- Actual local s1 A1/R1 inventory: early7K and same-run final50K files present,
  file SHA, A/T tensor hashes, official full40 evaluation identity, sensor and
  training manifest agree. The late file is an actual retained resume checkpoint,
  not an assumption that every historical evaluation saved weights.
- Actual early7K donor CPU import: original Teacher Adam moments (90 parameter
  states), H50 scheduler rate and RR sampler position preserved.
- s3/s4 checkpoint files are not on this server. Their inventory rules were
  exercised with synthetic fixtures; actual local replay is required at launch.

## Scientific / infrastructure tests

- Original DirectPANAligner + W112/D123/P0, no LPAN/HPAN, no new inference model.
- Teacher continuation freezes exact early A, preserves T moments/order/RNG,
  and reaches cumulative50K with the original global cosine schedule.
- Each tail starts at its own same-seed M_Q20K/40K state, with identical S
  weights/moments/stream; only registered differences are enabled.
- Aligner gradients reach fc2 only; after local5K, gradients are None and
  optimizer weight decay cannot move the refrozen head.
- Current-shift MASK2 is detached, sample-normalized; native warp occurs once;
  epsilon native anchor is detached and synthetic ROI8 support is checked.
- No Teacher or stale q/e calibration is constructed in adaptation tails.
- Native checkpoint roundtrip and exact next-update resume tested; changed
  donor/config/data/code/evaluator provenance is rejected.
- Synthetic official20+20 evaluation checks same-state weights/optimizer/RNG/
  trainable flags before and after evaluation, including the5K freeze boundary.
  These synthetic metric values are not performance evidence and are not uploaded.
- GPU cost deployment graph agrees bitwise on CPU with actual native A+T/A+S
  inference; Stage2 training-only Teacher is excluded.
- Paired admission, durable clock including downtime, disk reserve, partial
  checkpoint recovery, stable outbox timestamps and contiguous timing logs tested.
- Same-campaign other-server rows and historical sheet rows remain protected;
  append/readback recovery is tested offline with fake transports.

## Integration errors found and corrected before commit

1. Profiler construction with zero *training* loss coefficients failed the old
   constructor. The separate inference-only profile graph now uses harmless
   positive constructor placeholders; deployed predictions are unchanged.
2. Full-update timing rows are necessary for Train(h). The controller now writes
   every local step's measured duration, with detailed gradient trace every100.
3. Re-emitting a saved evaluation with a new timestamp conflicted with immutable
   outbox IDs. The first evaluation timestamp is now persisted and reused.
4. Derived Student-init seed, not only the experiment seed, must match at tail import.
5. Restored final checkpoints must update persisted completion step immediately.
6. Optional20K G/EPS/ABS controls now reserve their whole six-run block.

## Remaining verification at execution

No production GPU training, official real-data full40 replay, GPU latency/memory
profiling, live Sheet write, push, or remote-server operation was performed in
this task. Actual launcher runs the read-only data check and then GPU pilots,
full40 replay, calibration and cost measurement inside the30h campaign.

Runtime estimates are priors until those local pilots finish. Optional cases may
be skipped, and an unexpectedly slow server may refuse a core queue that cannot
fit its remaining budget. CPU tests do not establish the scientific hypothesis
or guarantee the joint HQNR/ERGAS target.

All formal evaluated checkpoints are retained. This costs more disk than
best-only retention; admission uses measured checkpoint sizes and never deletes
older experiments to make space. User data, credentials, outputs and unrelated
untracked files are excluded from the implementation commit.
