# GF2 PSNR 45.08 followup

Eight fresh Students after the existing s3 queues. H≥.9645, E≤.522, P≥45.08 simultaneously.
Frozen original Stage1, full Aligner scale1, RGB/Lanczos4, correction residual, β0, no additional method.

See [analysis](../results_log/2026-10-30_sensor-results-and-gf2-psnr.md) and
[s3 run instructions](../research_log/PANDA_GF2_PSNR4508_Handoff_2026-10-30.md).

```bash
python -m unittest discover -s panda_gf2_psnr/tests -v
python -m panda_gf2_psnr catalog
python -m panda_gf2_psnr.enqueue --wait
```

The registry adapter runs only in a new dedicated process and leaves all existing campaign files unchanged.
The launcher waits for the original focus controller (including WV3) to finish, validates predecessors,
profiles in the original Docker environment, starts training, and verifies terminal Sheet readback.
It does not stop existing processes. Execute on s3 after the user's Git push/pull.
