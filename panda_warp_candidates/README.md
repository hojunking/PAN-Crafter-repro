# Warp candidates: isolated CPU research

These operators are **not connected to any training, official evaluation, or
Sheet publisher**. `pa.warp.warp_pan`, frozen donors, and active campaigns are
unchanged. The purpose is to make alternatives concrete and test their numerical
properties before selecting a separate train/inference-matched experiment.

```bash
python -m unittest panda_warp_candidates.tests.test_ops -v
python tools/warp_candidate_audit.py \
  --out results_log/assets/2026-10-23_warp_candidate_audit.json
```

The audit uses CPU FP64 synthetic signals. Its result is not a new PAN model
score. It also checks arithmetic against the supplied
`2026-10-23_s1_warp_error_probe.json` without editing that evidence.

## Operators

`warp_global(image, delta, method=..., fft_padding=32)` accepts `B,C,H,W` images
and per-image `B,2` displacements `(dy,dx)`, using
`output[y,x] = input[y+dy,x+dx]`. Output preserves FP32/FP64 image dtype. Both
image and displacement gradients are tested; no Python-float conversion detaches
the shift. Reflection does not repeat the boundary pixel.

| Method | Definition | Main limitation |
|---|---|---|
| `fft` | Reflect-pad, complex 2D phase shift, real projection, crop | Global boundary dependence; Nyquist and crop/re-pad information loss |
| `lanczos4` | Normalized separable Lanczos, 8 taps per axis | High-frequency attenuation and ringing |
| `lanczos6` | Normalized separable Lanczos, 12 taps per axis | More cost/ringing; some passband amplification |
| `kaiser12_beta6` | Normalized 12-tap sinc with Kaiser beta 6 | Finite-band transition; beta/width tradeoff |

The audit also considers Kaiser 6/8/12 taps and prefiltered quintic spline.
Spline is an offline SciPy reference, not a torch training operator. Increasing
Kaiser beta can suppress ripple while worsening near-Nyquist attenuation; Kaiser
is not automatically better than Lanczos.

## Findings from the supplied evidence

- Bicubic's half-pixel response at `.375/.45` cycles/pixel is `.62766/.27089`,
  as reported. It also **amplifies** `.10/.15` to `1.01917/1.02874`; its effect
  is not only blur.
- At diagonal magnitude `2.0`, the axis magnitude is `1.41421`, not an integer.
  FFT model ERGAS increases from `2.0482224325` to `2.0499293788`, **+0.08334%**.
  At diagonal `1.5`, bicubic increases by **+0.06418%**, small but nonzero.
- The source field `hf_energy_retained` is a norm ratio, not squared energy.
  The source FFT's FR half-pixel round-trip relative RMSE is **0.00258264**.
- Inference-only FFT replacement establishes sensitivity to a changed input
  distribution. It does not prove that scale-dependent HQNR changes come only
  from geometry, nor that blur compensation is the only learned mechanism.
- The source probe converts deltas to Python floats. The mathematical FFT shift
  is differentiable, but that diagnostic wrapper does not preserve shift
  autograd. These isolated operators do.

## Why FFT is a candidate, not a lossless guarantee

Unit-modulus Fourier phases preserve a **complex periodic transform**. Real
finite images followed by reflection padding, real projection, cropping, and
re-padding do not satisfy that exact inverse model. Padding reduces some boundary
effects but does not remove them or provide a finite observed-support mask.

At Nyquist, `(-1)^x` shifted by half a pixel has an imaginary quadrature
component. Taking `.real` destroys it; the tested real-valued round trip has
RMSE 1 for a unit-amplitude checkerboard stripe. This is an explicit edge case,
not a claim that natural PAN images have large Nyquist energy.

There is also a 2D convention choice: the diagnostic's joint phase and `.real`
projection retains `cos(pi*(dy+dx))` at the Nyquist corner. Axis-wise real
projection would retain `cos(pi*dy)*cos(pi*dx)`. Consequently the implemented
probe-compatible FFT policy has a C4 counterexample at that corner. A future
training implementation must choose and validate its Nyquist policy; ordinary
phase-shift intuition alone is insufficient.

The synthetic unit step also rings: observed overshoot is about 9.38% for
bicubic, 11.89% for Lanczos4, and 13.66% for this FFT policy. These are diagnostic
step responses, not measured artifacts or metrics on the PAN dataset.

## Selection and experiment design

1. Screen one-way analytic targets and round trips separately, over fractional
   phases, signs, axes/diagonals, frequencies, and relevant image sizes. Compare
   full frame and ROI8/16/32, padding16/32/64, ringing and DC preservation.
2. Keep bicubic as control. Test FFT with an explicit Nyquist/boundary policy,
   Lanczos4, and a 12-tap candidate. Record real PAN spectra and boundaries;
   synthetic frequency response alone cannot select a winning model.
3. For a first Stage 2 ablation, train a fresh Student with the candidate and use
   that same candidate at inference. Keep positive Aligner scale, same seed,
   schedule, architecture and losses. Keep the frozen Teacher's trained bicubic
   input intact. Recompute train-only conditional-q calibration for the candidate
   Student PAN warp and its support policy; do not reuse bicubic q tables under a
   new warp label. Changing Stage 1 warp requires separately trained donors.
4. Finite-sinc support uses its actual 8/12 taps. FFT has global support: a
   bicubic four-tap support mask cannot certify that FFT's interior contains no
   reflected-boundary influence. Preserve the comparison ROI but record this
   change of boundary assumption explicitly.
5. For future synthetic misregistration, consider composing injection and
   correction before resampling the original PAN once: `d_total=d_inject+d_correct`
   under this sign convention. The Aligner still sees the injected image. This
   changes the augmentation's interpolation degradation and must be identified
   as a separate case, not silently substituted for a two-warp pipeline.

Beyond these operators, DCT/even-extension sinc methods are worth comparing if
FFT boundary artifacts are material. Learned warps or deblurring filters add
extra trainable choices and should follow the controlled interpolation cases.

The integer-attraction hypothesis for Stage 1 remains unverified. A fractional
shift histogram alone is insufficient: real sensor offsets, zero initialization,
and global bias can also produce peaks. Compare residual error against known
uniform-phase synthetic shifts and independent data-based offsets, then test
matched Stage 1 warp retraining before attributing a peak to bicubic loss.

## Primary references

- [Meijering et al.: 84 sinc-approximating kernels](https://research.tue.nl/en/publications/quantitative-comparison-of-sinc-approximating-kernels-for-medical/)
- [Theußl et al.: Kaiser/Gaussian window tradeoffs](https://www.cg.tuwien.ac.at/research/vis/vismed/Windows/)
- [Yaroslavsky: DCT-domain sinc interpolation and boundaries](https://opg.optica.org/ao/abstract.cfm?uri=ao-42-20-4166)
- [SciPy: prefiltered spline interpolation and boundary modes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html)
