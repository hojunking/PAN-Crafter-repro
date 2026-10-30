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

Each R1/R2 case compares with the matching seed of the previous focus campaign's
`GF2_C60LR150`. The original `RGB_MATURE_U23 @42K` Stage1 donor remains unchanged.

The waiter records its PID, source hashes and run registry in
`work_dir/panda_gf2_psnr/launcher_s3/launcher_source_pin.json`. It checks those
hashes while waiting, before each check/profile/launch command, and while
monitoring training and publication. Source changes stop the waiter with
`BLOCKED_SOURCE_CHANGED`; it never launches revised code using a stale registry.
For a user-requested queue revision before admission, stop the verified waiter
process, apply the revision, then restart it. Once `admission.json` exists, a
changed queue requires an explicit versioned migration; do not edit the admitted
registry or delete its admission/checkpoint files.

If training stops unexpectedly, the launcher reports `FAILED_TRAINING` and does
not restart it automatically. Before manual recovery, inspect
`launcher_s3/launch.json`, the Docker state/logs, the campaign/run states,
`admission.json`, and the affected run's `models/latest_resume.json`. Confirm the
container ID, image, command and mounts still match the launch receipt, the
source pin is unchanged, and the checkpoint file SHA and its metadata match.
Resolve the interruption first; a poisoned run or an incomplete optimizer step
requires investigation rather than restart. Confirm the GPU is idle.

Only after that review, restart the **same existing container** and the launcher:

```bash
docker start panda-gf2-psnr4508-s3-v1
PYTHONDONTWRITEBYTECODE=1 python -m panda_gf2_psnr.enqueue --wait
```

Training restores its complete checkpoint and verifies the frozen configuration,
code and donor evidence. Do not remove/recreate the container or erase receipts
to bypass a failed identity check. The launcher reuses the publisher and verifies
terminal Sheet readback after the queue completes.
