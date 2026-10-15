# QB actual raw predictions and preview images

Run: QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1; TARGET step 47470.

This package contains real inference outputs, not just metadata.
- PNG predictions: `Ours/QB/reduced/output/` and `Ours/QB/full/output/`.
- Full-band float32 DN predictions: `Ours_raw/QB/`, 20 RR + 20 FR.
- References include only the corresponding 20+20 test slices, unchanged source values.
- Standalone PNGs retain the entire native frame; panels add labels outside it.
- RGB uses shared per-scene reference 1–99% per-band stretch, gamma=1.
- PNGs are inspection previews; original CANConv renderer/ROI was unavailable.
- Frozen code and package versions match the original checkpoint. CPU export differs from original CUDA execution.
- See metrics/metric_comparison.json for differences: no predeclared parity tolerance, so review is required.
- Original full-band arrays are available for independent rendering/evaluation.
- No training, Sheets upload, or checkpoint selection was performed.
