# G20 source synchronization and s3 existing reference handoff

Checked: 2026-09-20 21:25 KST.

Remote release: e2d838a3f9ea45170fb71052710b38acc9992c61 (origin/main).
G20 campaign activation was not requested in this step and was not performed. No common G20 window was created, no training/evaluation/calibration was started, and no tau/qref/SHA value was manually populated.

## Repository update

- s1: git pull --ff-only origin main succeeded with Already up to date. HEAD e2d838a.
- s2: both origin/main and the configured server branch rejected fast-forward due divergent history. HEAD 0ec1132 preserved, including the local-only QG40 portability recovery.
- s3/s4: normal main pull refused local results_log/README.md changes and existing untracked QG40 source/config collisions. HEAD cccedee preserved.
- s5: normal main pull refused existing untracked QG40 source/config collisions. HEAD cccedee preserved.
- For s2–s5, fetched origin and applied ONLY the 77 new G20 files from the exact release: g20/, config/g20/, tools/g20_runner.py, tools/g20_start.sh and the two G20 plan/CSV files. This follows the existing HEAD-preserving deployment approach, but is not a full Git merge/pull to main.
- On every server, all 77 deployed files match the remote release byte-for-byte; Python sources parsed successfully. QG40 file hashes and local HEAD were checked unchanged during each overlay. Each server retains a receipt and tracked-change backup in .git/codex-g20-sync-20260920-sN/.
- No GitHub push was performed.

## Accessible s3 original paths

SSH alias: song. Host filesystem and running pan-qg40-s3-20260920 container both see the repository at /home/user/Desktop/song/PAN-Crafter.

Reference:
```text
/home/user/Desktop/song/PAN-Crafter/work_dir/_qg40/s3/references/GF2_TA/
```

Original Teacher TA:
```text
/home/user/Desktop/song/PAN-Crafter/work_dir/QGBASE_GF2_TA_P0_W112_D123_TS91001_FRESH50_v1/
```

Original Student S92001:
```text
/home/user/Desktop/song/PAN-Crafter/work_dir/QGBASE_GF2_S_PLH_W104_D122_TA_TS91001_SS92001_BASE_FRESH50_v1/
```

Additional original data provenance:
```text
/home/user/Desktop/song/PAN-Crafter/work_dir/_qg40/s3/dataset_manifest.json
```

All three directories exist and are readable. Reference has 5 files (about 2.21 MB); Teacher directory has 242 files (about 800.55 MB); Student directory has 240 files (about 666.17 MB). Whole-directory transfer is unnecessary for the requested next selection/verification step.

## Verified useful files

Reference contains reference_manifest.json, calibration.json, q_cache.npz, q_cache_parity.json, and reconstruction_distribution.json.
All eight original-manifest-linked artifacts (calibration, q-cache, q-cache parity, dataset manifest, Teacher checkpoint, Teacher identity, Teacher resolved config, Teacher candidate-50000 training full state) are present, readable, and have matching SHA256 values.

Both original run directories contain:
- meta/config.resolved.yaml
- meta/training_start_manifest.json and meta/training_status.json
- candidates/50000/model.safetensors and identity.json
- official/exact50k.json and postrun_status.json

Both original runs report actual_updates=50000, training_complete=true, official_complete=true, sheet_uploaded=true. Both candidate-50000 model hashes match their respective stored identity.json. Teacher full state is available at candidates/50000/training_state.pt.

The deployed G20 read-only inventory CLI executed successfully on s3 and found GF2_TA with artifacts_present=true and status DISCOVERED_NOT_VALIDATED. This is an availability/byte-integrity check, not G20 reference adoption or a common-evaluator parity PASS. GPU replay and cross-server transport remain separate later steps.

Machine-readable availability, file paths, and computed checks: 2026-09-20_g20-sync-s3-reference-availability.json.
