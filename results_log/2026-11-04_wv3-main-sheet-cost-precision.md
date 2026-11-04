# WV3-main: three decimals and inference/training costs

## Findings

The RGB main publisher populated metrics and Params only. Infer, memory, FLOPs,
Train and Wall were absent. Eval was one evaluation duration instead of the
cumulative evaluation cost through the selected checkpoint.

The historical TUNE30 row's 9.98679647454992 was a valid **ms/image** measurement:
RTX 4090, RR PAN 256×256, batch 1, FP32, 3 warmups and 10 repetitions, median of
synchronized resident-input A + warp + Student forwards. Its FR 512×512 result
was 33.08472101343796 ms/image. This was a fresh architecture profile rather than
timing the 29K checkpoint itself. File IO/H2D/metrics/Teacher were excluded.

Original source:
`work_dir/panda_tune30/PANDA_S13_TUNE_HARDONLY_30H_20261025_v1/s1/WV3/cost_profiles/9ed3fda867d94771dc67c49cbdc3ad9aa9d481239eb98695186a93ade06273a4.json`.

The historical `FLOPs(G)=71.790663744` was actually GMAC, a partial operation
estimate. The copied WV3-main historical row now shows **143.581 GFLOPs**, using
1 MAC = 2 FLOPs. Its old numeric value is preserved in a cell note; original
experiment evidence remains unchanged. All metrics/cost cells in this main tab
now show three decimals. Target selection retains full precision.

## New isolated measurement

Measured actual `M0_OURS R1 @24,000` A+S weights, same pinned image and RTX 4090,
FP32, batch 1, 10 warmups, 50 repetitions. Teacher and optimizer were not loaded
onto the profiling GPU. Identical native inference functions and strict A/S
state-dict loading preserve the evaluated model's graph.

| Domain | PAN size | Median ms/image | Peak allocated MiB | Params (M) | Partial GFLOPs, 2 FLOPs/MAC |
|---|---|---:|---:|---:|---:|
| RR | 256×256 | 10.049 | 292.652 | 2.891 | 143.581 |
| FR | 512×512 | 33.336 | 1090.527 | 2.891 | 574.618 |

The timing includes Aligner, Lanczos-4 warp, residual input construction,
interp23 base and Student. It excludes data reading/H2D and evaluation metrics.
Memory includes model, input, output and workspace, not reserved memory or
training memory. FLOPs omit activation, normalization and other documented
elementwise/indexing/weight-generation arithmetic; these operations **are**
included in measured latency.

Profile SHA-256 (canonical JSON excluding its digest field):
`1e02e9863f5c889bf8b5954a438fe6bba8d3dd31459b502192155e736968fdb3`.

Full profile and migration/readback evidence:
`work_dir/panda_wv3_main_rgb/PANDA_WV3_MAIN_RGB_S1_20261103_v1/s1/WV3/reporting/`.

Costs are shared across identical deployment graphs and identified as such in
row notes. Every measured main row receives cost values; queued rows retain
blank result/cost cells until their first official evaluation. FR costs are
retained in notes; the visible cost columns use RR for continuity.

Train, Eval and Wall columns refer to Stage2 through **Selected_step**, not the
whole Stage1+Stage2 method or necessarily the run's terminal step. Train sums
committed update durations; Eval sums all official evaluations; Wall includes
IO and any pause time. Stage1 and initial calibration are excluded and this
scope is recorded in both header notes and row provenance.

## Runtime migration and checks

M3_C27 R1 was gracefully saved at step **12,909**. The profiling process ran with
no competing CUDA compute process. The same image, same campaign, same weights,
optimizer, data stream, random state and original 436-file source release were
resumed through `panda_wv3_main_reporting run`. Only the publication entry point
is replaced. Original release digest remains
`aa63715e4e14edced38199d7ae9b112189d59debd9c0d920c095730e79e6cb8c`.

The stopped old container is retained as
`panda-wv3-main-rgb-s1-v1-before-reporting-20261104` for audit. Do not start it
alongside the active container. The active container retains the original name
`panda-wv3-main-rgb-s1-v1`.

Unit checks cover raw-versus-rounded selection/evidence, migration of original
rows, rejection of changed metrics, cumulative timing, missing/noncontiguous
trace rejection, and row-scoped formatting. Live readback confirmed all 18 owned
rows and populated costs. Training advanced through 13K after resumption.
