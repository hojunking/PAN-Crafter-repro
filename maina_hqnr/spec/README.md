# MAIN-A implementation-derived specification

These files were generated from the user-supplied experiment MD; the companion CSV/JSON files mentioned in §13 were not present. They are not represented as supplied originals.

`First3Cycles_PerServer_42Runs_PREVIEW.csv` is a preview, not a stopping limit. `spec_validation.json` reports static checks, not CUDA acceptance, training completion, or Sheet upload.

`original_source_identity.json` preserves the historical F1 bridge consumer identity (release `cccedeeeffd7ed19686e5ea684489ceee23cf313`). Startup verifies all 42 original files still match before using the original main numerical path. New campaign code has its own separate source identity.

F1 artifact paths may be explicitly relocated through `--asset-map` JSON. Canonical original manifests are copied byte-for-byte, never rewritten; the separate runtime binding retains original and resolved paths. No Teacher/q-cache/LP regeneration or F4/F5 fallback is implemented.
