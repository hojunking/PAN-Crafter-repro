# s2 G20 preflight recovery — 2026-09-20

At 22:42 KST, G20 on s2 was BLOCKED_INTEGRITY before admitting any G20 case. The preflight subprocess failed with `mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library`. GPU utilization was 0%. This was a runtime threading-library conflict before data/model preflight completed.

The prior QG40 seed82007 BASE had completed all 50,000 updates and official evaluation. Its result was READBACK_VERIFIED in pan-cvpr27 / QB-s2 row13 at 22:40:06 KST. The legacy QG40 controller's lost-hold message reflected the intentional G20 takeover after this final case; it did not indicate a failed checkpoint.

## Repair and verification

- Backed up blocked status, runtime command, registration, campaign window and preflight error log under s2 `.git/codex-g20-activation-20260920/mkl-repair-20260920/`.
- Ran the shipped G20 preflight in the same frozen image/worktree with `MKL_THREADING_LAYER=GNU`.
- Official preflight exited0: all64 method checks, real GF2 data validation and real CUDA forward passed. Its complete numerical source/runtime identity exactly matched the existing registration.
- Preserved registration, sensor specification, shared window and takeover hold byte-for-byte, verified by SHA256 before resuming. Model/loss/training/metric sources were not changed. No clock reset or checkpoint identity rewrite.
- Retained the failed container with suffix `-mkl-failure-20260920`; recreated the production container `pan-g20-s2-20260920` with the verified GNU setting. Host recovery reads the persisted corrected command.
- Cleared only this diagnosed pre-training failure after successful official preflight. No existing G20 learning had started; no training was interrupted or restarted fresh.
- At 22:46:52 KST, actual log updates had reached500 in `G20_GF2_T_S2_TS91002_C100_P0_W112_D123_FRESH50_v1`, GPU93%, no new error. Training status JSON is checkpoint/evaluation paced and still showed0 before the first1010 evaluation; the live optimizer log confirmed progress.

At 22:48:12 KST, the optimizer log reached1500 updates and the first scheduled official evaluation at1010 completed: rawHQNR0.90044547, SCC0.98216491, ERGAS0.93284770, valERGAS0.84399828. GPU utilization remained93%. This verifies learning and the real evaluation path after recovery.

The campaign still stops optimizer updates Sep21 16:05:13 KST and closes Sep21 18:05:13 KST. Existing per-run official upload/readback behavior is unchanged.

## s1 cross-check

s1 was also checked following the preceding user report: QG40 seed82008 progressed from32320 to34500 while GPU utilization remained about90%. Its G20 process was correctly waiting for the admitted QB pair82008/82009 to finish. No s1 failure was observed in this check.

The accompanying JSON and s2 repair directory retain actual runtime, data, source, clock and training evidence. No GitHub push was performed.
