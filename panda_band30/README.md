# s1 all-band / bandwise, 30-hour campaign

Implements [the registered case plan](../research_log/PANDA_S1_AllBand_Warp_Bandwise_30H_2026-10-19_v1.md)
in an isolated namespace. No existing s3/shared training implementation is modified.

```bash
python -m panda_band30 catalog --server s1 --csv
python -m panda_band30 smoke
bash scripts/launch_panda_band30_selected.sh s1
```

The explicit launcher reuses the local s1 Docker image, refuses an occupied GPU
or existing container, and starts the host-side fixed64 Sheet publisher.
It does not stop other work, pull an image, import an old donor, or wait for another server.

Queue: G8 Stage1 U23 → PAN / MID / MIDRET Stage2 → B8A Stage1 / Stage2 →
G8 Stage1 BIC / Stage2. Half-strength, zero/quarter-strength and the whole three-frame
R2 group are optional, admitted only with measured complete-run time remaining.
Every run retains its 50K horizon. The durable campaign clock allows 29.5h training
and reserves 30 minutes for closeout; restarting does not reset it.

Global Stage1 targets Scharr(mean of 8 GT bands), not mean of eight Scharr losses.
Bandwise Stage1 outputs 8×(dy,dx), supplies eight separately warped PAN views to a
16-channel U-Net, and uses the explicitly registered HR LMS/GT-only auxiliary
augmentation. The raw LRMS and reconstruction branch are not auxiliary-warped.
Fractional warp remains bicubic. LRMS interp23tap runs after shared augmentation
with its existing phase correction. No ×4/¼ inference displacement conversion.

Checkpoints and calibration bind the exact head, stem, upsampler, loss reduction,
augmentation RNG, sources, geometry and evaluator. A/T are frozen in Stage2.
HQNR is always primary; ERGAS breaks exact ties. Official RR20/FR20 references
remain original and unmasked. All 35 official evaluation steps remain saved locally.
The restored Sheet policy publishes every step with HQNR >= 0.960 AND ERGAS <= 2.045;
if none pass, exactly one HQNR-best row is updated per run (ERGAS breaks ties).
The display-only override lives in `panda_band30_publish`, outside the live training
code pin. The original all-step publisher/launcher and registered JSON are frozen
historical code; do not use the old launcher. See
[the publisher README](../panda_band30_publish/README.md) for a running-campaign cutover.

Input-only per-band residual registration runs on selected states, caches identical
corrections/source/geometry, and excludes invalid zero fallbacks from summaries.
It does not affect checkpoint selection. Disposable CUDA smoke artifacts under
`work_dir/panda_band30/preflight` are not experiment results or eligible donors.
