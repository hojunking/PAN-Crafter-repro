# s2 QB / s3 GF2 corrected-only Stage 2 handoff — 2026-11-13

User decisions confirmed in this session:

- QB: RGBN only; reuse s2's original unified RGBN Stage1, then two fresh Student
  seeds at40K. ERGAS<=3.5 candidates ranked by PSNR descending, then originalPAN
  HQNR descending. No added PSNR threshold.
- GF2: train fresh unified RGBN Stage1 twice on s3 (50K each), then corresponding
  two Students (40K each). Gate donors on ERGAS<=.522 and PSNR>=45.08, then maximize
  originalPAN HQNR. No cross-server weights.
- Both: no PAN correction-residual input; alpha0/gamma.25/beta0, scale1. Same
  prediction evaluated against originalPAN and correctedPAN, with all FR HQNR /
  D_lambda / D_s recorded and clearly separated.

| Server | Execution order | Stage2 LR | Architecture |
|---|---|---:|---|
| s2 | finish existing RGBN Stage1 to50K → S2 R1 40K → S2 R2 40K | 1e-4 | W128/D123/UBIL, input5/output4 |
| s3 | new S1 R1 50K → new S1 R2 50K → S2 R1 40K → S2 R2 40K | 2e-4 | W128/D123/UBIL, input5/output4 |

GF2 Stage1 uses lambda_spa2/lambda_epsilon.003, epsilon radius3, T/A LR1e-4/1e-5.
RGBN mean [2,1,0,3], Lanczos4 throughout, shared eroded full-tap valid mask.
Other optimizer/sampling settings follow the unified WV3 method. Full RR/FR
is evaluated every1K; GF2 validation RRE at30/40/50K. Frozen references and
calibration are checked against exact source/data/weight identities.

The explicit ranking above selects the Stage1 donor and is test-aware. Student
publication keeps the prior official-HQNR-qualified minimumERGAS representative
plus all H/E passes and fixed40K; correctedPAN mirrors the same chosen steps.
Final goals remain QB H>=.9315/E<=3.5; GF2 H>=.9645/E<=.522/P>=45.08.

Sheet destinations: QB-ablations (gid261001011), GF2-ablations (gid261001012).
Each selected state gets originalPAN and correctedPAN companion rows;4 decimals,
raw precision in notes/local JSON. No cloud rows were written from s1 for these
new remote queues. Automatic writing/readback occurs when the remote worker runs.

## Execute after the user pushes and each server pulls

```bash
# On s2 (gspread/server.txt must contain s2)
python3 -m panda_sensor_eq40.enqueue --server s2 --check-only
python3 -m panda_sensor_eq40.enqueue --server s2 --wait

# On s3 (gspread/server.txt must contain s3)
python3 -m panda_sensor_eq40.enqueue --server s3 --check-only
python3 -m panda_sensor_eq40.enqueue --server s3 --wait
```

Use a persistent terminal while --wait is waiting. s2 gracefully stops only its
identified original unified container and preserves the complete Stage1 restart
state; unfinished old Stage2 jobs are superseded without deleting results. s3
waits for current GPU work, preserving the Table3 queue/results. No automatic
interruption of unrelated experiments. Do not restart the old s2 unified queue.

Container names: panda-sensor-eq40-s2-v1 / panda-sensor-eq40-s3-v1.
Stopped containers resume with the same enqueue command plus --resume.
The existing s1 WV3 process/source release remains unchanged.

## Validation completed on s1

- Nine tests passed in pinned Torch2.4.0/cu118 Docker: native4-band loss paths,
  Stage1 gradient, full synthetic calibration save/load, frozen A/T identity,
  exact Student optimizer/model/data/RNG continuation, queues, ranking and Sheet
  targets. Real source/dataset/environment checks passed for QB and GF2.
- Native512 FR operator probes with registered QB/GF2 data confirmed identical
  D_lambda, unchanged originalPAN arrays, and correctly recomputed HQNR/D_s for
  a fractional Lanczos4 reference. These are operator checks with an interp23
  baseline, not trained-model performance results.
- Read-only Sheet preflights passed for both destinations:4 planned QB rows,
  8 planned GF2 rows, before training. No new remote result was claimed.
- External source dependencies match committed HEAD; active unified/WV3 EQ40/
  s3 Table3 source releases verify unchanged. Remote CUDA/donor checks execute
  at launch, because s1 does not hold s2's Stage1 weights.

See [full implementation and recovery instructions](../panda_sensor_eq40/README.md).
