# WV3 + QB RR raw prediction / GT

Raw export COMPLETE. Local ZIP ready; not externally delivered.

- WV3 TARGET38380, QB TARGET47470: verified reused predictions, no new inference.
- Ours_raw/<sensor>/reduced/pred/testimg0..19.npy: 40 predictions.
- references/<sensor>/reduced/gt/testimg0..19.npy: 40 matched native GT arrays.
- All arrays float32 CHW DN, maxDN2047; full256x256, all8/4bands.
- GT comes from original RR H5, converted only to float32; no clipping/registration.
- Scene mapping uses original H5 SHA + gt key + index; no cross-method PNG matching.
- No PNG, rendering, calibration, FR inference, training, selection, or upload.
- Original CPU predictions differ from historical CUDA metrics; see validation/rr_metric_comparison.json.
- WV3 band names are not asserted: unchanged source indices0..7; QB labels come from source manifest.
- checksums.sha256 covers package files; sibling ZIP receipt gives archive SHA/CRC and size.
