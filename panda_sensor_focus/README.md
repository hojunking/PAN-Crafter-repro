# Fixed-donor SensorFocus, 2026-10-29

This isolated successor registers six fresh QB Students on s1 and five fresh GF2 Students on s3. It does not modify the running SENSOR_REFINE implementation or train new Stage 1 models. All models use full Aligner correction (scale 1), corrected PAN plus corrected-minus-original residual plus four-band Ilrms, Lanczos-4, RGB reference maps, UBIL, beta 0, and one native inference view. Rotation consistency, TTA, weight averaging and band-weight loss remain excluded.

The exact prior Stage 1 selection is authenticated by original validation receipt, completed 50K donor lineage, selected step and checkpoint SHA. QB reuses LOSS9 `S1_A200_E030 @44K` (lambda_abs 2, lambda_epsilon .03, epsilon radius 3). GF2 reuses SENSOR5 v2 `RGB_MATURE_U23 @42K` (2, .003, 3); this is the donor of the published LONG result, not the newer LOSS9 A200_E030 reference. Original source Teacher/Aligner classes, checkpoint tensor identities and source binding fields are preserved. W128 changes only the Student; the Teacher remains W112.

The finite registry is in [cases_s1.csv](cases_s1.csv), [cases_s3.csv](cases_s3.csv) and [experiment_manifest.json](experiment_manifest.json). Each run starts from the original named initialization/stream seed protocol; there is no Student checkpoint continuation or inherited optimizer. Horizons are 30K QB and 60K GF2, warmup 100, minimum cosine LR 0.

Read-only checks and catalog:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m panda_sensor_focus catalog --server s1 --csv
PYTHONDONTWRITEBYTECODE=1 python -m panda_sensor_focus check --server s1
PYTHONDONTWRITEBYTECODE=1 python -m panda_sensor_focus smoke
```

The server launcher first accounts for the explicitly retained/retired predecessor queue, waits for its safe current-run boundary, prepares native W112/W128 cost measurements, then runs disposable real-data training/resume preflight before each production case. Do not run the training CLI directly while another campaign owns the GPU. Control and handoff instructions are documented by the queue/launcher files and the research-log handoff.

`Campaign(server, output='work_dir/panda_sensor_focus', device='cuda')` and CLI `run --server s1|s3 --output ...` (`--root` alias) use a separate immutable campaign root. Exact optimizer/RNG/stream resume and source-pin checks are retained. The QB Student gate requires complete authenticated official 8K, 9K and 10K HQNR records, all strictly below .90; after 10K evidence is saved it stops without waiting another 1K. It never applies to GF2 or Stage 1.

Sheet publication uses QB-ablations and GF2-ablations. Targets are QB HQNR >= .9315 and ERGAS <= 3.5; GF2 HQNR >= .9645 and ERGAS <= .522. It publishes all joint-pass steps plus one per-run representative: minimum ERGAS among HQNR-qualified steps, or maximum HQNR if none qualify, with duplicate suppression. The Approach column includes original Stage 1 coefficients, radius, full scale, Student width, inputs, alpha/gamma/beta, LR and horizon; immutable source/receipt provenance is retained in notes. Cost profiling and publication remain separate processes.

Student adapters are isolated copies of the existing strict namespace-bound implementation. Shared original architectures, donor loaders, datasets, optimization, RR/FR metrics and Lanczos numerical kernels are reused and included in the source pin. Synthetic CPU tests verify original QB/GF2 donor class compatibility, W128 parameter counts, hard-only updates, exact checkpoint resume, finite registration, goal/tab routing and authenticated early-stop behavior; synthetic values are never experiment evidence.
