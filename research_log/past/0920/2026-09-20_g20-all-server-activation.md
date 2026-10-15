# G20 all-server activation — 2026-09-20

## Authorization and current campaign

User requested starting the new experiment on all five servers. Campaign: PANDA_GF2_G20_ALL5_20260920_v1. Existing admitted QG40 blocks are drained by the shipped G20 transition protocol without interrupting trainers or rewriting checkpoints.

One common window was created by the official window command: 2026-09-20 22:05:13 KST through 2026-09-21 18:05:13 KST. New-block admission ends Sep21 14:05:13; optimizer updates end Sep21 16:05:13. Preparation and drain time count within this same window. The JSON SHA256 is 9a06b3f8f616ceb9482243c1f897bd9755b34687fa2032f98e6e14359b54d985 on all five servers. No clock reset.

## Source and runtime baseline

All 77 G20 source/config/plan files on every frozen runtime are byte-identical to e2d838a3f9ea45170fb71052710b38acc9992c61. Executions use locked detached worktrees mounted read-only at the original repository paths. Original work_dir remains writable and data/credentials remain on their servers. Development HEAD changes cannot change the executing HEAD.

| Server | Frozen runtime HEAD | G20 numerical content SHA | Runtime |
|---|---|---|---|
| s1 | 8a63d10604901609facfd22b2b195724ea0122f6 | cdb98d2c18dcfa3225102aab1846cdc5db812b283de15a73c7244924fd65bdab | Original native conda Python, Torch 2.4.0/CUDA11.8, inside dedicated container |
| s2 | 0ec113218f60f861dab18ab84f3083b730890979 | 4e18b1b6ec1c93616ce2211aea02752b68cf5f2995a1789e82f55d45e3e89e72 | Existing Torch2.8/CUDA12.8 image; retains prior QG40 validation-only portability patch |
| s3/s4/s5 | cccedeeeffd7ed19686e5ea684489ceee23cf313 | cdb98d2c18dcfa3225102aab1846cdc5db812b283de15a73c7244924fd65bdab | Existing Torch2.8/CUDA12.8 images |

The s2 content difference is recorded explicitly; it is not relabelled as identical. Future imported references must pass the shipped compatibility checks. No numerical model/loss/training/metric source was modified in this operation. No GitHub push.

## Reference and data preparation

The original s3 GF2_TA export was staged as incoming/R0.tar.gz on every server. Ten unmodified files from TA and S92001 (config, exact50K weights, identity, full-precision official report, original dataset manifest) were staged under parity_inputs. Tau, qref, checkpoint hashes and calibration IDs were never manually supplied or rewritten.

s3 and s5 use their already verified QG40 GF2 sensor bindings. Their default GF2 FR H5 serializations do not match the pinned file hash. s4 received the exact original FR H5 under work_dir/_g20/s4/incoming/GF2_FR.h5, verified against the original SHA e52d151f262a74f96e59f03124f86a27d80841073ce60376856ae679020d00dc. All other s4 split hashes were checked before binding. Existing data files were preserved.

## Operational launcher correction

After reference adoption/replay, the controller retained its own CUDA context. The shipped exclusive GPU check then detected the controller's own PID and waited indefinitely. A separate operational launcher filters only its own PID from the GPU inventory; unknown/foreign GPU processes remain blocking. Six boundary checks passed. On s2 only, it also calls the previously authorized exact-PID/start-time idle-service filter, with its original free-memory/utilization limits.

Launcher SHA256: 4ba34147fc7339b00c39638c0e760fcdc7a1a06535388b4199ad48bb1a207ca2. Registration/source hashes and existing receipts were not rewritten. Only G20 controllers with no training runs were restarted to install this launcher. QG40 trainers remained running.

## Verification at 22:22 KST

- s5: G20 BASE seed93001 RUNNING, 3030 optimizer updates, 3 scheduled evaluations; GPU96%. R0 TA/S92001 replay PASS, exact same HQNR/ERGAS as originals.
- s4: QG40 finite queue finished; G20 LOCAL_READY and R0 replay PASS; first BASE seed93001 submitted for training.
- s1: G20 registered and running in WAIT_ADMITTED_LEGACY_BLOCK; QG40 seed82008 at21210, then paired82009 must finish.
- s2: G20 registered, hold DRAINING_FINAL_LEGACY_CASE; QG40 final seed82007 at41410. Approved service coexistence applies once the trainer finishes.
- s3: G20 registered and running in WAIT_ADMITTED_LEGACY_BLOCK; QG40 seed92020 at18180, then paired92021 must finish.

s1-s3 have not yet passed G20 production data/GPU parity; the official runner performs these after the safe drain. No readiness PASS is inferred from registration.

## Assigned GF2 axes and upload

s1: R0/R1; s2: R2/R4 then eligible R3; s3: A1/A9; s4: E1/E4; s5: K05/K20. BASE controls and seed93001/93002 order follow the shipped registry. Conditional confirmation/transfer remain evidence and budget gated.

Official G20 completion invokes the shipped upload immediately, then checks Sheet readback; failed uploads remain pending and are retried. Target tabs are GF2-s1, GF2-s2, GF2-s3(5090), GF2-s4, GF2-s5 in pan-cvpr27. G20 has no completed/uploaded run at this snapshot. Scheduled within-training metrics already exist locally on s5.

## Recovery and evidence

Each host has .git/codex-g20-activation-20260920/{runtime.json,g20_launch.py,g20_host_recovery.py,activation-verification.json,host-recovery.json}. Host recovery runs every5min and after reboot. It preserves BLOCKED_INTEGRITY and closed queues instead of repeatedly resetting them. It checks the expected container image and command. Containers are pan-g20-sN-20260920.

Campaign state: work_dir/_g20/sN. Learning log: work_dir/<run>/g20_train.log. The accompanying JSON retains per-server actual source/window/registration/cron/state evidence. All original QG40 campaign clocks and numerical outputs remain unchanged.

