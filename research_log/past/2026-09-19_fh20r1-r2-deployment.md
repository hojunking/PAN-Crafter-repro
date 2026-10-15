# FH20R1 R2 deployment — 2026-09-19

User-directed delivery: fetch fh20r1-r2, archive only campaign_r2 + tools/r2_{runner.py,start.sh,bundle.py} + supplied results_log plan, then bash tools/r2_start.sh. No pull/checkout/merge was performed. Overlay commit: 6a995ae876589067cd21fbbf617caa6634c73635.

Verified snapshot around 20:47 KST:

| Server | Original HEAD retained | R2 state | Current update | Existing effective hours | Completed/upload pending |
|---|---|---|---:|---:|---|
| s1 | cccedee | WAITING_CASE_BOUNDARY | 44000 | 6.385 | 4/0 |
| s2 | 0ec1132 | WAITING_CASE_BOUNDARY | 40300 | 5.753 | 3/0 |
| s3 | cccedee | WAITING_CASE_BOUNDARY | 10600 | 7.529 | 7/0 |
| s4 | cccedee | WAITING_CASE_BOUNDARY | 38300 | 6.868 | 6/0 |
| s5 | cccedee | APPLIED | 1000 | 7.714 | 7/0 |

All five passed 95 R2 tests, original source/config/branch integrity checks, and existing ledger validation. Original budgets and campaign start times are unchanged. s2's explicitly authorized GPU variance exception remains intact. s5 has already handed over and begun the next run under R2. Other controllers remain alive and will take over after the current case finishes training, postrun evaluation, and upload. Already admitted pairs/quartets retain priority and internal order.

s3 will finish admitted B04 before B06; s4/s5 will finish admitted B03 before B06. s1/s2 retain their existing block order. The plan changes comparison priority and TARGET-first local summaries; it does not change the numerical method or rewrite existing Sheet RAW_MAX/TARGET columns.

Automatic uploads remain enabled. Counts above are verified against existing local readback receipts, not a new live Sheet scan. Additional scientific diagnostics retain their actual TO_VERIFY/NOT_IMPLEMENTED states.

s1 uses its original native Python environment. s2-s5 run R2 in separate persistent Docker containers cloned from the original image/environment/binds/user, so the retiring original controller container cannot terminate the new R2 controller. Each container runs the requested r2_start.sh after tests. The original trainer remains untouched until the cooperative case boundary.

The containers lack crontab. Host cron recovery was verified/installed with the package's own preserving installer. An unhashed watchdog prefix routes host recovery to .git/codex-r2-docker-watchdog.sh, which ensures a running R2 container or starts the existing stopped container. The original R2 hook remains byte-exact and passes its verifier. Host recovery commands returned ALREADY_RUNNING on all four Docker servers. Reboot recovery has not been physically tested.

Evidence on every server:
- .git/codex-r2-deploy-20260919/sN/ (archive delivery record, test/start logs, watchdog backup)
- work_dir/_fh20r1/sN/priority_r2/registration.json
- work_dir/_fh20r1/sN/priority_r2/deployment_verification.json
- work_dir/_fh20r1/sN/priority_r2/applied_readback.json (created at actual handoff)
- work_dir/_fh20r1/sN/priority_r2/verification_status.json
- work_dir/_fh20r1/sN/priority_r2/host_docker_recovery.json (s2-s5)

Do not overwrite an already registered overlay with different code. Original source identity and HEAD remain authoritative for active checkpoints.
