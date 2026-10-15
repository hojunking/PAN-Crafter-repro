# WIP: s1 TA2 Teacher-only launch (2026-10-06)

Incomplete campaign; this is a launch record, not a completed-result claim.

- Requested: pull and start the registered s1 experiment in Docker.
- Source commit: `77e5cfee3e09e26142a802d5edfa856e677e63eb`. Pre-existing local changes are preserved.
- Dataset: WV3; supplied 52 cases x 3 seeds = 156 runs.
- Plan and operations: `ta2/OPERATIONS.txt` and `research_log/PAN_Aligner_TeacherOnly_v2/`.
- Command: `bash ta2/start.sh s1 run --publish`.
- Runtime: immutable source snapshot, Docker image digest and actual CUDA probe recorded by the launcher.
- Outputs: `work_dir/aligner_teacher_lms_v2/s1/`.
- Status at this record: pull complete, launch requested; verify preflight, smoke receipt and increasing optimizer updates before claiming training.
- Reporting: background numeric publishing enabled; durable local outbox and readback receipts distinguish generated results from uploaded results.
- Scientific settings and queue are unchanged. No previous results were deleted and no unrelated job was stopped.
- Storage: existing 5 GiB free-space guard preserved; full-lane minimum exceeds available storage on some hosts.

Purpose: the supplied Teacher-only NativeLMS/GTStructure comparisons; assess registered paired and three-seed results under the fixed FR20 selector and RR20 protocol after completion.

Prepared: 2026-10-06T16:27:27.348281+09:00
