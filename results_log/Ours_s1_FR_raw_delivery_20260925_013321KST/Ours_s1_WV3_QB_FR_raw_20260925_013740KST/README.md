# WV3 + QB FR raw supplement

FR_RAW_COMPLETE / RENDERING_SKIPPED_BY_SCOPE / READY_LOCAL_NOT_DELIVERED

- WV3 TARGET38380:20 float32 CHW(8,512,512) predictions.
- QB TARGET47470:20 float32 CHW(4,512,512) predictions.
- Ours_raw/<sensor>/full/pred/testimg0..19.npy at ZIP root.
- DN0..2047, native512x512, all bands, unchanged source order.
- Source FR H5 row index0..19 preserved. USER_CONFIRMED_PREVIEW_ORDER, not externally reverified here.
- Byte-identical reuse of verified CPU predictions; no new inference or postprocessing.
- No RR, GT, PNG, error maps, training, checkpoint reselection, or reference arrays.
- Source PAN/MS/LP inputs were verified; their arrays are omitted from delivery only.
- Original selection metadata is retained, including historical RR fields, without RR arrays/re-evaluation.
- CPU vs original CUDA metric differences are recorded, not certified as bit-identical.
- See validation/fr_metric_comparison.json and raw_array_checks.json.
- WV3 names unconfirmed: original band indices0..7; QB band labels from source manifest.
- Existing training/previous exports unchanged. Local ZIP requires user retrieval or separately authorized transfer.
