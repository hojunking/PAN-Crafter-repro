# G20 reference and evaluator-parity handoff

All paths below are local to the consuming server. Reading inventory does not
adopt a reference, mark it valid, allocate a GPU, or start training.

## Reference inputs

`g20.references.inventory_references(root)` searches the existing `_qg40` and
new `_g20` reference manifests without copying or changing the original assets.
`DISCOVERED_NOT_VALIDATED` is deliberately not a readiness result.

- R0 is exactly the pinned s3 `GF2_TA/91001` bundle from plan §4.1. Its old
  manifest, run name, source identity and calibration ID remain intact.
- R1 belongs to s1; R2/R3/R4 belong to s2. New Teachers use seed91002 and their
  own exact50K A/U, config, full state, calibration and full train×4 q-cache.
  Identical seeds or coincidentally identical weights never merge R1 and R2.
- `calibrate(teacher_run, root, server, device, deadline_utc, batch_size)` obeys
  the immutable campaign window and publishes the manifest last. It measures
  the shared seed1234 train3072 pooled-pixel median and all train×4 AXIS16 values;
  qref is the train3072×4 median, not a full-train median. No R0 τ/q fallback.
  Frozen exact50K calibration may finish during 18–20h closeout; it never resumes
  optimizer updates and cannot run beyond the absolute 20h deadline.

Explicit local adoption uses:

```python
from g20.references import adopt_reference
adopt_reference(original_manifest_path, 'R0', server, root,
                device='cuda', deadline=campaign_deadline)
```

Portable transport uses `export_reference(reference_id, server, root, device,
deadline=...)` and `import_reference(archive, reference_id, server, root, device,
deadline=...)`. Imports accept a complete checksummed G20 export or the original
QG40 `GF2_TA` export for R0. Archive aliases, path traversal, symlinks, duplicated
entries, payload checksums and incomplete exact50K full-state provenance are
checked. Do not send only `model.safetensors` or rename GF2_TA into R1/R2.

R0's old complete package source hash is **not** compared to G20's new package
hash. Instead its original plan pins and unchanged numerical-core file hashes
are verified, followed by actual frozen-Teacher q-cache/online checks on 16
fixed train IDs×4 views and output/native-c probe readback. A new bridge receipt
binds the consumer release; it does not overwrite the original provenance.
Local output probing is explicitly labelled an immutable-core readback, not a
historical GPU replay; the common-evaluator replay below provides that separate
same-checkpoint metric check.

H5 byte equality is sufficient but is not the only permitted proof. Repacked
source data needs exact ordered canonical MS/LMS/PAN/GT tensor equality against
the pinned original or an original-manifest-bound checksummed export receipt.
For LP only, the authenticated original dataset manifest's full float32 LP
canonical SHA plus sample order can prove equality after reserialization.
Every local split file SHA is rechecked. Names, approximate statistics and
unbound user-entered canonical strings are not proof.

## Common-evaluator replay evidence

Call `g20.parity.verify_r0_parity(root, server, device, deadline,
evidence_path=None)`. The default evidence path is
`work_dir/_g20/<server>/r0_parity_evidence.json`. Both original TA and original
S92001 exact50K assets are required; rounded Sheet cells are not accepted.

`discover_parity_evidence(root, server)` can fill the paths and SHA values
automatically from verified original local QG40 runs. On another server, retain
the original `meta/config.resolved.yaml`, `candidates/50000/{model.safetensors,
identity.json}`, `official/exact50k.json`, and unmodified original
`dataset_manifest.json` under
`work_dir/_g20/<server>/parity_inputs/{TA,S92001}/`. Configs and original
manifests are copied unchanged, not rewritten to local paths. This is separate
from the R0 reference archive: that archive does not include the S92001 model
or original official reports. Discovery is read-only and returns
`EVIDENCE_READY_NOT_REPLAYED` plus the evidence document only after artifact
validation. It never claims that metric replay already passed; ambiguous
different S92001 checkpoint copies require an explicit choice.
When the default evidence file is absent, `verify_r0_parity` invokes this
discovery itself and saves only a fully validated evidence document before
replay. A missing explicitly specified evidence path never silently falls back
to a different asset.

The evidence document has this shape; replace each placeholder with the actual
local path and file SHA256, not a guessed value:

```json
{
  "schema": "G20_R0_EVALUATOR_EVIDENCE_v1",
  "assets": {
    "TA": {
      "checkpoint_path": ".../candidates/50000/model.safetensors",
      "checkpoint_sha256": "actual SHA256",
      "config_path": ".../meta/config.resolved.yaml",
      "config_sha256": "actual SHA256",
      "identity_path": ".../candidates/50000/identity.json",
      "identity_sha256": "actual SHA256",
      "prior_report_path": ".../EXACT50K.json",
      "prior_report_sha256": "actual SHA256",
      "dataset_manifest_path": ".../original_dataset_manifest.json",
      "dataset_manifest_sha256": "actual SHA256"
    },
    "S92001": {
      "checkpoint_path": ".../candidates/50000/model.safetensors",
      "checkpoint_sha256": "actual SHA256",
      "config_path": ".../meta/config.resolved.yaml",
      "config_sha256": "actual SHA256",
      "identity_path": ".../candidates/50000/identity.json",
      "identity_sha256": "actual SHA256",
      "prior_report_path": ".../EXACT50K.json",
      "prior_report_sha256": "actual SHA256",
      "dataset_manifest_path": ".../original_dataset_manifest.json",
      "dataset_manifest_sha256": "actual SHA256"
    }
  }
}
```

Original reports must include their full checkpoint identity and unrounded
official RR20/Q4/20:-21 and native FR20/full512/raw-original metrics. The current
evaluator actually infers both models. Each must satisfy |ΔHQNR|≤1e-5 and
|ΔERGAS|≤1e-4. This is numerical replay, not a retraining-quality threshold.
The receipt is `diagnostics/r0_evaluator_parity.json` under the local campaign.
Successful measured receipts are sealed by a separate immutable SHA receipt.
Subsequent cases reuse PASS only after rechecking the original evidence and
artifact hashes, every local source/LP file, data equivalence, the current
release, and both stored measurements. They do not repeat 40-scene inference
for every Student. A changed/corrupt sealed receipt is rejected.
Missing evidence/assets produce `WAIT_PARITY`, never fabricated PASS. An actual
out-of-tolerance replay produces `BLOCKED_PARITY`. R0 consumers remain gated;
independent fresh Teacher chains need not wait for the R0 replay.

No actual R0 or S92001 replay was performed while implementing these modules.
Unit tests use temporary CPU fixtures and do not certify production assets.
