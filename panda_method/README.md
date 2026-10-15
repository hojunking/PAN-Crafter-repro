# PANDA MethodPlan-v1.3 — method-only implementation

**Latest 2026-10-13 update: Edge-Guided Stage1.** The current Stage1 specification
replaces both scene-prior and bGlobal conditioning with **per-input integer
Scharr EdgeMatch + Cartesian residual CNN**. See
[config/edge_stage1/README.md](../config/edge_stage1/README.md). Stage2 is not part
of this new change. New RR/FR inference estimates b on the current input grid;
no frozen prior or FR ×4 policy is needed. Old APIs below remain for old runs.

**Previous 2026-10-13 bGlobal/RR update (preserved, not the active Stage1):** the separate
`global_prior.py` / `global_data.py` / `global_rr.py` path, documented in
[config/bglobal_rr/README.md](../config/bglobal_rr/README.md). It uses the stored
LMS, a train-only dataset-global prior and **no FR training loss**. The scene
mapping and RR+FR instructions below describe the preserved legacy v1.3 API,
not the new bGlobal path. Do not launch an old Teacher48 campaign expecting the
new method. No existing checkpoint or experiment is automatically migrated.

Scope: `research_log/PANDA_Method_Implementation_Validation_Plan_2026-10-13_v1_3.md`.
The user's 2026-10-13 instruction excludes experiment plans. This package has
**no training entrypoint, experiment cases, queue, launch, automatic dataset
selection, sheet upload, or checkpoint-ranking change**. HQNR selection remains
the existing runner's responsibility. Nothing executes on import.

## Implemented numerical contract

| Part | Implementation |
|---|---|
| Scene prior | `geometry.ScenePriorRecord/ScenePriorRegistry`: explicit accepted `(dy,dx)` PAN sampling correction; immutable provenance; no learned prior or sensor-average fallback |
| Prior identity/grid | `data.PriorBinding`: binds sample→source-scene mapping, registry and explicit grid transforms by SHA256; unknown scene/grid or wrong binding fails |
| Augmentation | `data.augment_batch`: hflip → vflip → CCW rot90 of PAN/MS/LMS/GT/LPAN and the corresponding vector, in that order |
| Aligner | `SceneAwarePANAligner`: legacy GN stems/joints; **exact existing** `PANCrafterPaper.ResBlock(64, norm='ln', dropout=0, mode_mod=False, se=False)`; **`[GAP h(64); detached raw b(dy,dx)]`** → 66→32→2 head; `c=b+αd+βn`; no output clamp |
| Initialization | Common ResBlock second Conv and final FC zero-init; initially `c=b`, not zero displacement |
| Reconstruction | `Reconstructor`: existing U-Net and named initialization unchanged; explicit capacity/layout/base; native LMS is separate from reconstruction base |
| Existing LP/HP inputs | Native LR-grid LPAN is upsampled and warped with PAN on the same grid; HP=corrected PAN−corrected LP; no change to MS frame/base |
| Stage 1 | RR native `L_rec + L_abs + .001*Lε_RR`, plus FR `.001*Lε_FR`; one Teacher forward, four shared Aligner forwards per objective |
| Epsilon | `pa.offset.sample_offsets`: radius2 uniform disk on each current PAN grid; independent CPU generator; fresh RR/FR samples every call; native PAN warped once and detached; only ε target `c0` detached |
| Stage 2 | One shared frozen A, frozen T, trainable S only; identical corrected PAN tensor; Teacher-error continuous blend plus q-weighted GT RGB edge L1 |
| Calibration | Exact **global pixel** training medians, not median of patch medians; complete caller-supplied sample inventory required; checkpoint/preprocessing/scene mapping/source hashes bound |
| Deployment | `StudentInference`: A+S only; no Teacher, GT, calibration, e/q or epsilon inputs |

Legacy `pa/aligner.py`, `model/pancrafter_paper.py` and campaign paths are not
modified. The old A remains available for old checkpoints; the **new method A**
really contains the common LN ResBlock. Old A checkpoints are incompatible with
the new residual/head, and no permissive migration is performed.

### v1.3 head semantics and checkpoint boundary

The last two FC1 inputs are the **raw signed scene shift on the current PAN
grid**, after the caller's verified grid conversion and image augmentation.
They are not the normalized direction, shift norm, epsilon vector, or learned
embedding. There is no extra magnitude normalization, scaling or clamp.
`d=b/||b||` and `n=(-d_x,d_y)` are still used only for residual synthesis:
`c=b+alpha*d+beta*n`. Native and synthetic-epsilon forwards use the same prior.
The final zero-initialized FC still gives `c=b` on a fresh model.

FC1's shape remains 66→32, so shape/parameter-count equality cannot establish
v1.2/v1.3 compatibility. The Aligner carries a persistent non-trainable int64
`_conditioning_revision=13` buffer and rejects missing/incompatible markers
even under nested or permissive state-dict loading. Pipeline manifests pin
the exact conditioning semantics; method version and calibration bindings
also change. **Do not add a marker to an old checkpoint to bypass the guard.**
No weight migration is implemented: fresh training is required unless a
separate, explicitly documented migration/retraining protocol is approved.

`return_details=True` provides head input/preactivation/hidden tensors for
numeric diagnostics. `Stage1.native_prediction(..., return_aligner_details=True)`
reuses the same Aligner forward, without adding another geometry prediction.
Teacher48 logs native RR/FR prior and FC activation ranges without changing
the forward values or the loss.

Teacher/Student width/depth/input layout/base are required arguments. The plan's
112/D123 and 104/D122 are candidates, **not automatically selected cases**.
`build_reconstructor` preserves FH12/QG40 named U initialization. A's new
initializer is the PyTorch module constructor; a future runner must seed/pin it
explicitly and must not reset its intentional zero projections afterwards.

## Spatial operator: verified official SIPSA detail

Official [helperTF.py, `sobelfilter2XY`](https://github.com/brachiohyup/SIPSA/blob/main/helperTF.py)
uses adjacent-pixel differences, despite the function name: center minus next
row/column, unscaled, reflection padding. It is **not conventional 3×3 Sobel**.
[main_train.py](https://github.com/brachiohyup/SIPSA/blob/main/main_train.py) takes
RGB arithmetic mean, absolute derivative components, then mean absolute
component discrepancy. No external runtime or copied training code is imported.

Use `losses.sipsa_spatial_spec(sensor=..., rgb_indices=..., roi_margin=...,
radiometry=...)` for this method. It selects `forward_difference` and `reflect`;
the derivative sign is opposite before absolute value, making the descriptor
mathematically identical. A direct difference/reflected-boundary golden test
checks this independently of convolution. Other explicitly named kernels in the
low-level utility are not the canonical SIPSA method or selected experiments.

Still explicit: actual RGB band indices, common PAN/GT intensity units, fixed
spatial ROI. No radiometric normalization or clipping is silently introduced.
Default operator import is not a claim that PANDA's coordinate frame, target,
weights and complete training pipeline are identical to SIPSA.

Stage 1 absolute term, new `q` and Student GT edge term reuse the **same object**.
Reconstruction/e/fit use all image pixels. q calibration and spatial terms use
the declared fixed ROI. Weighted losses divide by the fixed number of ROI
pixels, **not by weight sum** and not by a shift-dependent mask.

`w=x/(x+tau)` uses FP32 minimum, preserves FP64, and floors denominators at the
computation dtype's smallest normal positive number (`finfo.tiny`). The explicit
0/0 result is zero; positive x with zero tau has weight one. This is numerical
handling, not a tuned loss parameter. No nonfinite/empty ROI becomes zero loss.

## Integration order (no runnable experiment defaults)

1. Supply real sample→source-scene mapping, accepted canonical priors, verified
   scale/phase transforms and hashes. `bind_scene_priors` resolves vectors;
   `augment_batch` keeps imagery and vectors synchronized. Raw `ObservationBatch`
   is a lower-level numerical API; typed strings/hashes alone cannot prove that
   external provenance is truthful.
2. Supply explicit Aligner `view_margin`, spatial `roi_margin`, `boundary_policy`,
   RGB/radiometry and a `calibration_protocol` describing population/augmentation.
   `report` logs insufficient warp/ε-view support without modifying the
   objective; `require_spatial_support` rejects unsupported spatial ROI plus
   derivative stencil. Neither silently masks pixels. A full-field ε view can
   contain padding cues; choosing it is **not proof those cues are harmless**.
3. Wrap selected unchanged Teacher U and initialize new A. Construct `Stage1`
   with explicit `TrainingSources` and epsilon seed. Its call computes one full
   RR+FR objective; the later trainer owns optimizer steps, batching and logs.
   RR and FR each use batch/component mean, then their ε terms are added.
4. Stage 1 `state_dict` includes method/pipeline/source contracts and ε RNG state.
   Save it **plus optimizer/data-loader/global RNG states** in the future runner.
   `forward_calls` counts objectives computed, not successful optimizer steps.
   Restore strictly. This package does not choose a best checkpoint.
5. On the selected fixed A/T, explicitly call `calibrate_training(...,
   expected_sample_ids=...)` with the complete training population once. The
   calibration loader must implement the declared augmentation protocol; it
   must not use shuffled repeated samples, `drop_last`, validation or test GT.
   Even population medians use the midpoint of the two central values.
6. `Stage2(stage1, student, scales)` transfers A/T **by reference**, freezes and
   clears their gradients. Do not continue Stage 1 with these same modules.
   Use `stage2.trainable_parameters()` and `assert_optimizer()`; parameter
   aliasing among A/T/S is rejected. `train()` leaves A/T in eval mode.
7. Use `stage2.inference_model()` for A+S deployment. Preserve architecture,
   priors/mapping and preprocessing metadata alongside exported weights.

Calibration currently retains exact detached FP64 e/q values in host memory,
not GPU memory. Budget RAM before a full population pass (GF2 19,809×64² is
~81M values per map; both maps plus concatenation/median work buffers can take
several GB). No approximate quantile or sampled subset silently substitutes
for the full-population median.

## Deliberately not decided/claimed

- No source scenes, scene IDs, priors, MTF phase offsets or FR training sources
  are invented from the existing independent benchmark patches.
- No new off-line registration algorithm/threshold is picked. The old audit's
  Scharr/ZNCC+census components live in `align/estimator.py`; its scene-level
  aggregation/acceptance, correct sign and phase still need a source-data-bound
  adapter. This package **consumes verified priors**, not unverified test-patch
  estimates. A benchmark row ID is not accepted as proof of an acquisition ID.
- Scientific near-zero/low-confidence fallback remains unresolved. Registry
  records require explicit reviewed acceptance; exact zero/nonfinite/subnormal
  vectors fail rather than inventing a direction. No arbitrary confidence
  threshold is added.
- No RR↔FR direct consistency loss, MIND training loss, q_old, teacher-advantage
  gate, extra unconditional GT term, trainable Student A, or shifted U training.
- No actual training, real-data calibration, source leakage audit, GPU/AMP
  throughput validation, or quality claim was performed by these unit tests.
- The supplied plan references §20 but ends at §19. Its linked
  `PANDA_Stage1_Stage2_Final_Method_2026-10-13.md` was not available at the named
  repository path. This implementation follows the provided v1.3 plan itself.

## Validation

2026-10-13 verification: **56 new method tests PASS**, **19 existing TA2
model/loss regression tests PASS**, Python compilation PASS. CPU environment:
PyTorch 2.4.0, existing `pancrafter` environment. No GPU or real training data
was used. These are correctness checks, not evidence of task-quality gains.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s panda_method -t . -p 'test_*.py' -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest ta2.test_model_loss -v
```

CPU tests cover warp/augmentation sign, uniform-disk moments/area CDF and RNG
replay, initial correction=prior, unclamped residual, staged zero-init gradient
flow, shared common block vs legacy GN stems, strict checkpoints, native LMS/base
separation, C4/C8×all four layouts bitwise legacy U parity, one native U/four A
calls, independent RR/FR gradient routes, complete train-only global calibration,
frozen A/T state hashes, exact shared PAN, Student-only graph and invalid inputs.

Zero-init caveat: fresh U output Conv initially blocks rec→A gradients; fresh A
FC2 blocks feature extractor gradients. Tests separately check initial FC2
gradient and later feature/ResBlock gradients; they do not remove zero-init to
make an invalid first-step test pass.

Measured A counts/cost, batch1 and explicit full-field `view_margin=0`:

| Bands | Parameters | 64² GFLOPs | 256² GFLOPs | 512² GFLOPs |
|---|---:|---:|---:|---:|
| 4 | 104,946 | 0.071963 | 1.151341 | 4.605350 |
| 8 | 105,522 | 0.076681 | 1.226838 | 4.907340 |

Cost is **2×Conv2d/Linear MACs only**; excludes norm, bias adds, activations,
GAP, validation, vector arithmetic, warp/interpolation and off-line prior cost.
It is not end-to-end inference FLOPs or a speed/quality measurement. The new A
has 192 more parameters than the corresponding old A.
