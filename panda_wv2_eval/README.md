# s1: frozen WV3 models evaluated on WV2 after the active queue

This is evaluation only: no Stage 1/2 training, calibration fitting, finetuning,
TTA, scale search or WV2 checkpoint selection. All three checkpoints are fixed
before observing any WV2 result, and every result is posted.

| Order | WV3 source | Step | Width | WV3 HQNR | WV3 ERGAS |
|---|---|---:|---:|---:|---:|
| 1 | M5-LR110 R3 (RGB) | 29K | 112 | 0.9606112 | 2.0394382 |
| 2 | W128 W1-G025-LR100 R1 (RGB) | 30K | 128 | 0.9597951 | 2.0327496 |
| 3 | M0-OURS R2 (RGB) | 28K | 112 | 0.9591662 | 2.0393692 |

The first is user-selected. The second has the lowest currently observed ERGAS
among preserved target-passing main checkpoints. The third checks the original
W112 recipe with a separately trained R2 donor/Student. These are exploratory
WV3 test-aware selections; they are not three statistical repetitions of one recipe.

## Scheduling

The waiting Docker container is `panda-wv2-zero-shot-after-wv3-s1-v1`.
It requires **all 18 runs** of `PANDA_WV3_W128_FROZEN_S1_20261106_v1` to be
`COMPLETED` at 30K, with final evaluation evidence, then waits for the GPU to be
idle. A stopped/failed/partially completed queue never triggers WV2 evaluation.
It does not interrupt, resume or modify the training container. Waiting uses no
CUDA context. Do not start a separate GPU job once this queue begins evaluation.

```bash
python -m panda_wv2_eval.enqueue
python -m panda_wv2_eval.enqueue --status
docker logs --tail 30 panda-wv2-zero-shot-after-wv3-s1-v1
# After inspecting an interrupted/failed worker:
python -m panda_wv2_eval.enqueue --resume
```

Original checkpoints/evaluations/configs are read-only. Verified copies of the
three checkpoints are preserved under the new campaign's `sources/` folder on
**s1**, with no cross-server transfer. Preserve these until the evaluation is
complete. Per-scene predictions/receipts allow an interrupted evaluation to
continue without mixing models/data/code. Source/environment hashes are checked.

## Data and metrics

- Existing PanCollection WV2 RR20 (256×256) and FR20 (512×512), 8 bands, peak2047.
- FR H5 PAN/MS/LMS arrays were verified exactly equal to the repository's MAT20
  conversion for every scene. No scene subset is selected.
- Stored band order: Coastal, Blue, Green, Yellow, Red, Red Edge, NIR1, NIR2.
  Preserve this benchmark convention; no band permutation or bandwise fit.
- Input normalization `2*DN/2047-1`; Aligner receives original PAN and stored LMS;
  Student receives corrected PAN, corrected-original PAN and interp23 LRMS base.
- Frozen Aligner scale1, Lanczos4 reflection, original D123/UBIL backbone,
  full native inference. Teacher and adaptive fitting losses are not used at inference.
- RR GT is excluded from the inference batch. It enters RR metrics only.
  Use original RR `[20:-21,20:-21]` crop, Q8 block32, global PSNR, SAM/ERGAS/SCC/SSIM/CC/RMSE.
- FR uses original PAN/stored LMS/LRMS, full512 without a mask or reference warp.
  **WV2** spectral MTF (`[.35]*7+[.27]`) is used for D_lambda;
  JQM uses WV2 PAN GNyq .11 and retains the disclosed NNLS SRF-substitute variant.
- WV2 sensor registration is process-local; running WV3 source files stay unchanged.
- No supplied WV2 target: publish all three results, without applying WV3 thresholds.

## Output / Sheet

Root: `work_dir/panda_wv2_eval/PANDA_WV2_ZEROSHOT_S1_20261106_v1/s1/WV2`.
Each `runs/<case>/evaluation.json` contains aggregate and per-scene FR/RR results.
`rr/prediction/scene_00.npy` through `scene_19.npy` and the corresponding `fr/`
contain unclipped float32 CHW eight-band predictions.

Destination is **WV2-ablations**, sheetId261001013 in the existing workbook.
There are exactly three owned rows. Pending rows have blank quality metrics;
full RR20+FR20 completion updates that same row, with four-decimal results and
full-precision evidence in notes. The source WV3 run/step/hash and hyperparameters
are recorded. Existing rows from other campaigns are preserved.

Infer(ms), peak allocated MiB and partial GFLOPs are measured on the actual frozen
model using native WV2 scene0 after the GPU becomes idle. This native timing
includes interp23 and validation checks, so it has a different scope from the
prepared CUDA-Graph timing in WV3-main; the scope is explicit in the Sheet.
Train(h)=0 denotes no WV2 adaptation, not zero cost for original WV3 training.

## Validation

`python -m unittest discover -s panda_wv2_eval/tests -v` checks all three loaded
frozen models on synthetic WV2 batches, no RR-GT leakage, WV2 metric routing,
complete-queue gating, and owned-row publication. `python -m panda_wv2_eval check`
authenticates the real local source files and metric environment without inference.
`check-sheet` verifies credentials/header/row handling without writing.
