# Verified QB/GF2 qualitative exports

`python -m panda_sensor_visuals.export --run-dir <attempt> --step <step> --output <new-directory> --zip`

Authenticates checkpoint SHA, config and same-state RR20/FR20 evaluation. Rebuilds
the original Aligner+Student inference graph without Teacher, retraining, TTA or
post-warp. Every raw float32 DN prediction must match the original per-scene SHA.
NPY is CHW4 with B/G/R/NIR bands; display RGB is R/G/B, indices2/1/0. Default
prediction PNGs use the independent sorted1–99% display convention from the
user's sharpness analysis. Shared-reference and fixed-DN images are also saved.

The ZIP includes original predictions, reference images/NPZ, metrics, provenance,
README and a browsable index. It excludes weights and credentials. ZIP CRC and
all archived NPY hashes are verified. Existing complete exports are reused only
after their file hashes match; incomplete outputs are preserved, not overwritten.

The `.txt` source snapshot is the user's `51_sharpness_qb_gf2.py`, stored only as
metric provenance and an independent golden reference for tests. It is never
executed as a script. The core metrics do not require the comparison-model images.
Reference-model metrics on s3 come from the supplied JSON, not a new measurement
of their absent images. This diagnostic never publishes to Sheets.
