# Edge-Guided Residual Aligner — Stage1 implementation

Current specification:
`research_log/PANDA_Stage1_EdgeGuided_Residual_Aligner_Final_2026-10-13.md`.
This supersedes scene/global-prior **Stage1** semantics. Previous APIs and
checkpoints are preserved, never silently migrated. Stage2, new experiment
cases, campaigns, training launches and Sheet changes are outside this change.

## Entry points

| Module | Responsibility |
|---|---|
| `panda_method/edge_match.py` | Shared fixed Scharr, per-input integer EdgeMatch, explicit failure diagnostics |
| `panda_method/edge_guided.py` | FC64 Cartesian residual CNN, native/epsilon paths, RR Stage1 objective and RR/FR prediction |
| `panda_method/edge_data.py` | Actual stored-LMS source adapter, augmentation, RR/FR split boundaries; no prior asset |
| `teacher48/edge_training.py` | Explicit builder and optimizer engine, losses/gradient/shift logging and checkpoint resume |

No `b_s`, `b_global`, 66-dimensional head, alpha/beta basis, global-prior
confidence gates or Stage2 calibration is connected to this method.

## Numerical contract

1. RGB mean first: WV3 `(R,G,B)=(4,2,1)`, QB/GF2 `(2,1,0)`.
2. Shared `G=[abs(Sx*I),abs(Sy*I)]`, Scharr fixed `/32`, reflect padding.
   No adaptive edge normalization, magnitude reduction, NCC/census, learned
   descriptor or subpixel peak fit. The CNN's old channelwise `znorm` remains.
3. Matching searches the explicitly declared **square integer** lattice
   `[-r,r]²` on the current PAN grid. Score is two-axis raw absolute-gradient
   L1 on the same common support for every candidate. The margin is `r+1`,
   including Scharr's halo. No sign reversal and no automatic ×4.
4. `P_pre=W(P,b)` is CNN guidance only. GN stems/joints + the unchanged common
   LN ResBlock + GAP64 + FC64→32→2 produce continuous Cartesian `delta`.
   Final FC zero-init gives `delta=0`, `u=b`. Valid `b=(0,0)` is allowed.
5. Teacher always receives `W(P_original,b+delta)`; it does not receive
   `W(P_pre,delta)`. Its unchanged P0 input uses independent bicubic LRMS base.
6. Native and epsilon PAN independently run the **same** EdgeMatch. No reused
   b0 or injected `b0-epsilon`. The shared LMS and MS coordinate frame stay fixed.
7. RR `L=rec + abs + .001*eps`; epsilon radius2 area-uniform disk, every update.
   These are the specification's initial settings, not a train-only validated
   optimum. Their historical radius rationale included FR benchmark statistics.
8. Teacher has rec gradients only. Residual CNN has rec/abs/epsilon gradients.
   Only the epsilon target `u0` is detached, not native rec/abs geometry.
9. RR256 and FR512 prediction are whole-image, one shift per image. Each reruns
   EdgeMatch with its declared current-grid radius. No scale multiplication.
   FR/held-out predictions disable gradients; FR cannot enter optimizer steps.

Epsilon introduces known synthetic padding. `warp_support_mask(epsilon)` is
eroded by the Scharr footprint, then intersected across **all candidates**.
This yields one common scoring support per sample, with identical score
denominators for every candidate. No candidate-dependent valid pixel count and
no confidence weighting are used. This is matcher geometry handling, **not an
HQNR evaluation mask or a dynamic training-loss mask**.

`L_abs` has a separately declared fixed ROI. `loss_boundary_policy='report'`
records unsupported warp/stencil pixels without changing the objective;
`'require_spatial_support'` rejects that update. Neither silently crops by the
predicted shift. CNN preliminary warp support (including both epsilon/prewarp
samplers) is logged; a full-field CNN may still observe border cues.

## Data and reference policy

`EdgeDataSources('ablr2/sensor_sources.json', root=..., sensor=...)` uses existing
registered train/val/RR20/FR20 arrays and actual file SHA checks. Provided LMS
is scaled by the existing `2*DN/peak-1`, not reinterpolated or clipped. Images
are augmented before matching; no cached vector is transformed. No scene
mapping, train-wide prior scan, frozen b asset or native-FR training data is needed.

For QB, pass the `splits` dictionary from
`config/bglobal_rr/qb_original_lms.json` as `lms_overrides`. This JSON is only
the earlier **user-selected LMS source registration**, not a bGlobal asset.
Original train/validation LMS is combined with unchanged msfix LRMS; PAN/GT
for each consumed row must be exact matches between the two files. The hybrid
source is recorded. We do **not** claim to have reproduced original LMS as
`U_phase(msfix MS)` or independently proved their phase identity.

## Explicit policies before a real run

`policy.template.json` lists unresolved settings and deliberately contains
invalid nulls. There are no runnable production defaults. Construct:

```python
EdgeMatchSpec(sensor, radius_rr, radius_fr, tie_policy,
              low_texture_threshold, low_texture_policy, boundary_policy)
```

- Radius is a nonnegative integer in **current PAN pixels**, separately RR/FR.
  A radius of zero explicitly means no search.
- Tie is exact equality of minimum scores. Policies: `error` or
  `min_norm_then_lexicographic`. Nonunique tie resolution is deterministic but
  is **not generally D4-equivariant**; unique winners are tested under D4.
- Low texture is PAN or RGB-LMS mean absXY at/below the explicit threshold,
  in the supplied intensity units. Policies: `error` or explicit `zero`.
  Zero fallback is logged and overrides tie error for that low-texture sample.
- A winner at the search boundary is recorded or rejected (`report`/`error`).
  It is not a certified interior optimum. No valid common region always fails.
- Independently choose `aligner_view_margin`, fixed loss `roi_margin`, loss
  boundary policy and per-term gradient logging cadence.

These decisions must not be selected by maximizing FR test metrics. The new
specification resolves the old FR b-conversion question: **run the matcher on
FR directly**. Only search coverage/handling remains to be supplied.

## Training API, not an automatic launch

Build an explicit P0/BICUBIC_MS `ReconstructionSpec`, `EdgeMatchSpec`, source
adapter and seeds; call `build_edge_stage1(...)` from `teacher48.edge_training`.
Construct the optimizer on that stage and `EdgeStage1Trainer(...,
gradient_log_interval=N)`. `trainer.step(rr_batch)` performs one optimizer update.

Returned JSON-serializable metrics include losses, combined A/T gradient norms,
per-term A/T norms at the selected cadence, native/epsilon b/delta/u, tie/texture/
boundary flags, and support diagnostics. A caller must persist these records
through its run logger; this method engine does not select a log destination or
upload a run by itself. Per-term gradient logging costs extra backwards on
diagnostic steps and does not alter the actual gradient/update.

Checkpoint includes method/matcher/source/ROI semantics, A/T, epsilon RNG,
optimizer class and **ordered parameter names**, diagnostic cadence and update
count. It rejects scene/global/legacy GN64 Aligner state. The future campaign
must additionally persist its scheduler and data-order/augmentation RNG.
HQNR-first checkpoint ranking and official metric implementations are unchanged.

## Verification

```bash
python -m unittest panda_method.test_edge_match panda_method.test_edge_guided \
  panda_method.test_edge_data -v
python -m unittest discover -s panda_method -t . -q
python -m unittest discover -s teacher48 -t . -q
python -m unittest ta2.test_model_loss -q
```

See `results_log/2026-10-13_edge-guided-stage1-implementation-review.md` for the
audit. Tests use synthetic tensors/disposable H5 files; this is not a quality
claim or an approval of the synthetic matcher policy for real data.
