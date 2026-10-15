# s2 FH20R1 GPU parity exception — 2026-09-19

User approval: "진행해 참고기록으로 남기고" (proceed, leave a reference record).
Scope: this s2 FH20R1 deployment, consumer Git 0ec113218f60f861dab18ab84f3083b730890979 and the exact local F2 teacher below. No numerical source, TF32 policy, or comparison tolerance was changed.

## Observed evidence
- GPU historical/current reader parity failed at the existing atol=3e-6 and rtol=2e-5.
- Independent current/current GPU readers also differed: fr_y max_abs=0.00040912628173828125, 23063/4194304 elements outside tolerance; rr_y max_abs=0.0001220703125, 1/1048576 outside tolerance.
- Historical/current CPU readers matched exactly in the diagnostic.
- cuDNN deterministic=True alone did not resolve the GPU comparison.
- A separate strict deterministic diagnostic stopped because reflection_pad2d_backward_cuda has no deterministic implementation. This does not establish the cause of the forward fr_y variance.
- F2 teacher SHA256: d2832f5a2d40536a859369c011063fe65794c0b7159c93ab23e9976bcb98a1b2.
- Origin content SHA256: c3c1ade0d8f43f29edeab61b0584cc0ab866c45947ee9fb816629fe35cf6c5ec.
- Detailed diagnostic: .git/codex-fh20r1-s2-parity-diagnosis.json.

## Approved behavior
The original GPU failure is retained as USER_ACCEPTED_GPU_VARIANCE; the imported reference is ACCEPTED_WITH_PARITY_EXCEPTION, never PASS. Before accepting the exception the runner requires a freshly repeated, exactly matching CPU parity check, identical initial model-state hashes, identical output keys/shapes, finite arrays, and all existing source/checkpoint/config/data/q integrity checks. Actual CUDA batch48 smoke, donor checks, loss routing, resource policy and disk checks remain required.

The authorization JSON and its SHA256 are embedded in the F2 bridge. Bridge consumers verify this authorization. Readiness retains origin_forward_step0_parity=false and separately records user_authorized_gpu_variance=true. Every s2 FH20R1 result (including reuse and N2 arms) appends the exception to the existing Sheet Notes field.

Existing low-memory idle GPU services remain running under the previously authorized identity-pinned resource policy. No prior result or historical source identity is rewritten. These results must be interpreted with the GPU variance exception.

## Runtime evidence
- work_dir/_fh20r1/s2/parity_exception_authorization.json
- work_dir/_fh20r1/s2/imported_refs/F2/bridge_manifest.json
- work_dir/_fh20r1/s2/readiness_report.json
- work_dir/_fh20r1/s2/deployment_blocker.json
- work_dir/_fh20r1/s2/runner.log

The code changes are local to s2 and have not been pushed upstream. The bridge records the full modified consumer source identity. Do not edit these runtime source files after training starts.

## Verified restart
- 2026-09-19T13:30:14.962065+09:00: real CUDA training started; run FH20R1_S2_S_PL_W104_D122_WV3_TF2_TS71001_SS72001_BASE_FRESH50_v1.
- 134 FH20R1 tests plus 6 resource-policy tests passed. Generator, read-only origin checks and actual GPU batch48 preflight passed.
- Fresh historical/current GPU fr_y max_abs=0.000032782554626464844; 9/4194304 elements outside the unchanged tolerance. Fresh CPU max_abs=0.
- Consumer source content SHA256: a74a454979a17191778bc7daa8c9b85b42361ca4e8fb1c01ea3941426339c18b.
- Readiness: origin_forward_step0_parity=false; user_authorized_gpu_variance=true; launch_ready=true.
- N2 initialization donor checkpoint is absent locally; the predefined S2_DONOR_OR_REFERENCE_UNAVAILABLE branch runs the F2 PL versus PLH comparison. This is separate from the GPU parity exception.
- Automatic uploads remain enabled, with the exception in every s2 FH20R1 row's existing Notes field. No full-run result is complete at this restart snapshot.

- First official evaluation verified at 2026-09-19T13:31:01.170905+09:00: [FH20R1 official step=1010] HQNR=0.90207146 SCC=0.95797358 ERGAS=3.46005720 valERGAS=3.35423994. Training continued to update 1100/50000, GPU 97 %, 5674 MiB, 65536 MiB, no fatal training errors.
