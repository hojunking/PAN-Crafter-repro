# Sensor refinement, 2026-10-28

Separate campaign `PANDA_S13_SENSOR_REFINE_R34_20261028_v1` preserves all legacy training source files and WV3 queues. Launch through `scripts/enqueue_panda_sensor_refine.sh s1|s3`; the controller authenticates retained LOSS9 completion before entering this package.

Each server has ten new runs: four fresh Students using the original same-server A200_E030 R3 donor, then three fresh Stage1/Student pairs. QB Students: BASE R2, LR120, C40, H050. GF2: BASE R2, LR150, C60, H050. Fresh pairs: A200_E030_R4, A200_E010_R3, A100_E030_R3. GF2 center is a transfer hypothesis.

Radius changes only the existing RR PAN translation injection. Uniform disk R3/R4, input margin8, native Scharr ROI8, unchanged scale1/Lanczos4/RGB/DIFF+UBIL/Hard-only, beta0. No clipping, adaptive ROI, rotation consistency, SWA or band-weighted loss.

Known support failure preserves a verified unpoisoned stop checkpoint, holds dependent Student and continues independent cases. A disposable preflight support failure has a separate authenticated zero-production-update receipt. Unknown exceptions remain fatal. A failed external donor holds only its four dependent Students. `FINISHED_WITH_EXPLICIT_FAILURES` is never full experimental success.

Sheets: QB-ablations `.9315/3.5`; GF2-ablations `.9645/.522`; every joint pass plus minimum-ERGAS HQNR-qualified representative, otherwise highest HQNR. Visible descriptions include radius and coefficients. All official evaluations remain same-checkpoint RR20/FR20 with original references.

CPU tests: `python -m unittest discover -s panda_sensor_refine/tests -v`. Production runs also require real-data disposable exact-resume preflight, source hashes and independent cost measurement.
