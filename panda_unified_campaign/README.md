# Unified WV3 / QB fresh main experiments — 2026-11-13

This campaign starts from fresh weights. It preserves historical experiment packages.

## Shared implementation

- Teacher and Student: same sensor-specific width, D123, stride-2 convolution downsampling, bilinear x2 (`align_corners=False`) + 3x3 convolution upsampling (UBIL).
- Teacher input: corrected PAN + interp23 MS base. Student input: corrected PAN + (corrected PAN − original PAN) + interp23 MS base. This intended input-channel difference remains.
- Stage 1 correction, controlled ε injection, validation, frozen Teacher, Student, calibration and native RR/FR inference all use the same Lanczos-4 operator. PAN Aligner scale = 1.
- All pixel losses and calibration use `erode3x3(full eight-tap Lanczos support)`, with outside values false. Warp support uses exactly the integer indices used by the interpolation implementation. No fixed ROI8 remains in these losses or calibration.
- The Aligner clean input crop8 is input preprocessing. The RRE metric/search margins and official benchmark crops are separate metric definitions and are retained.
- The selected edge-reference bands are averaged equally before Scharr: WV3 RGBN=[4,2,1,6]; QB RGB=[2,1,0], RGBN=[2,1,0,3]. L_spa, q/g, calibration and RRE all use that group's reference. Aligner/reconstruction inputs and reconstruction error retain every MS band.
- β=0, no rotation consistency, no TTA/SWA. Existing paired flip/rot90 augmentation remains.

## s1 WV3

Two repetitions: fresh Stage 1 50K then Ours Stage 2 30K, repeated with seeds 261016102/261018102 after 261016101/261018101. W112 throughout. Stage 1 λ_spa=2, λ_ε=.003, R_ε=3, T LR1e-4/A LR1e-5. Stage 2 α=γ=.25, LR1.1e-4. Total four runs. Representative target HQNR≥.9585 and ERGAS≤2.040.

## s2 QB

Two fresh local Stage 1 runs, one RGB and one RGBN, both seed261016301; W128 throughout, λ_spa=4, λ_ε=.03, R_ε=3, T LR1e-4/A LR1e-5, 50K. No weight transfer. Complete both donors, then execute paired RGB/RGBN Students in this order:

| Priority | Recipe | α | γ | Peak LR |
|---|---|---:|---:|---:|
| 1 | Q2BASE | .25 | .50 | 1e-4 |
| 2 | H050 | .125 | .25 | 1e-4 |
| 3 | G025 | .25 | .25 | 1e-4 |
| 4 | LR090 | .25 | .50 | 9e-5 |
| 5 | LR110 | .25 | .50 | 1.1e-4 |

All ten Students use seed261018101 and 30K. These are five paired recipe comparisons, not five independent repeats. QB target HQNR≥.9315 and ERGAS≤3.5. Existing QB early stop remains: stop Student at 10K if 8K, 9K and 10K HQNR are all strictly below .90.

The prior QFS2 Q2_W128 R1 23K result passed: HQNR .9389469340, ERGAS 3.4952699125 (`QB-ablations` row105, verified 2026-11-13). Its recipe supplies the reference. The unified Teacher W128/UBIL, warp and common loss domain are changes, so the new campaign is not an identical reproduction or a guaranteed target pass. Historical R2/R3 did not jointly pass.

## Runtime and publication

AdamW β=(.9,.999), ε=1e-8, WD=.01, batch48/micro12, warmup100 then cosine to zero. Full RR20/FR20 every1K. Same-server, same-mode, same-repeat Stage 1 donor is chosen by validation RRE50, RRE90, ValERGAS then earlier step; test metrics do not choose the donor. Every evaluated model is retained, with the two latest complete optimizer/stream/RNG resume checkpoints.

Automatic publication uses `WV3-ablations` / `QB-ablations`, four decimal places, one representative (minimum ERGAS among HQNR-qualified checkpoints, otherwise max HQNR), all joint passes and fixed29K/50K rows. No unmeasured inference-cost numbers are copied from a different graph. Notes include the full recipe and geometry/reference contract.

## Run after pull

```bash
python3 -m panda_unified_campaign catalog --server s2
python3 -m panda_unified_campaign.release
python3 -m panda_unified_campaign.enqueue --server s2 --wait
```

Use `s1` for the WV3 queue. The launcher checks data, reference Docker environment, source hashes, Sheet access, CUDA numerical/resume preflight and available storage. It waits for another GPU job without stopping it. `--check-only` performs read-only/data/Sheet checks. `--resume` restarts the same stopped campaign container from complete checkpoints. A running old-code campaign cannot resume into this new campaign.

Image: `hojunqueen/pancrafter-env:torch2.4.0-cu118`. If DLPan is not present inside the image, set `PANCRAFTER_DLPAN` to its local checkout. Dataset H5s and local Sheet service account remain local resources. Source identity includes the transitive local Python imports and registered config dependencies; the committed release must verify after pull. Do not regenerate the release to conceal missing or changed code on s2.
