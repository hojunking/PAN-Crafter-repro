# PANDA30R — s1 / s3 / s4, WV3, each ≤30h

Plan: [experiment rationale and cases](../research_log/PANDA_S134_LocalDonor_Schedule_30H_2026-10-18.md).

This additive package keeps the original Direct Aligner, W112/D123/P0 model,
MASK2 training support, and unmasked official RR20/FR20 evaluation.

- s1: early vs final donor vs frozen-early-A/mature-T; paired GT/KD controls.
- s3: matched Stage2 forks; frozen A vs fc2-only5K adaptation then refreeze.
- s4: paired soft-KD strength, Student LR, and mirror augmentation controls.

Each server uses its own named A1/R1 lineage (fresh local fallback if absent).
No checkpoint transfer, ensemble, metric change, existing campaign mutation,
or automatic training on `git pull`.

```bash
bash scripts/run_panda30r_docker.sh s3  # replace s3 with this server's s1/s4
```

The launcher starts a separate container and host sheet watcher. Existing busy
GPUs/containers are not stopped. Existing WV3 data, DLPan evaluator, Docker image
and local upload credential must be available. Network failure retains outbox.

```bash
python -m panda30r catalog --server s3 --csv
python -m panda30r check --server s3   # read-only actual data/evaluator check
python -m panda30r smoke              # CPU synthetic tests, no Sheets writes
```

The 30h deadline includes donor preparation, calibration, profiling, evaluation,
checkpoints and downtime. Optional pairs are admitted together using local
measured throughput and disk space. No promise that all optional cases fit.

All formal evaluated states are retained with full optimizer/sampler/RNG state.
Local updates and inherited lifetime updates are distinct. Resume reuses exact
pins; truncated post-checkpoint trace is preserved in audit backups, not erased.
Do not update code while this campaign is running.

Checkpoint choice: HQNR≥.959 then minimum ERGAS; otherwise best HQNR.
Existing sheet rule is different and unchanged: all .960/2.045-qualified steps,
otherwise one best-HQNR row per run. Costs measure deployed A+S, not KD Teacher.
