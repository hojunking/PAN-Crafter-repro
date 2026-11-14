# Actual Stage1 reference / Teacher report

Report the Teacher paired with the **actual Stage2 donor**, not a separately
selected best Teacher or automatically the50K endpoint. No retraining and no
cross-server weights. A Stage1 reconstruction may outperform the Student;
these rows make that comparison visible without relabelling Stage1 as Ours
Stage2 or changing checkpoint selection.

Current registered references:

|Server|Sensor|Stage1 checkpoint|
|---|---|---|
|s1|WV3|UNIFIED RGBN R1@27K, the shared current ablation donor|
|s2|QB|UNIFIED RGBN R1@32K, current MAIN50 donor|
|s3|GF2|SEQ40 RGBN R2@42K, the former MAIN50 donor|
|s3|QB|New s3-local PARALLEL donor, chosen after its50K Stage1 completes|

The same Teacher output and original stored LMS are used in both FR triplets.
Only the PAN reference changes to the same Aligner's Lanczos4-corrected PAN.
No mask/crop/TTA or alternate spectral reference is added to native FR20.
D_lambda should be identical up to CPU/GPU numerical tolerance. RR uses the
verified original evaluation for that same checkpoint.

Native dual-FR evidence is reused when present (GF2 and new s3 QB). For legacy
QB evidence without corrected FR, a CPU-only FR20 re-evaluation verifies original
metrics and weights before publishing. s1 reuses the already verified CPU FR20
probe, checked against its exact checkpoint and original evaluation SHA.

Commands, executed on the matching server after pull:

```bash
# s1, does not interrupt WV3 training
python3 -m panda_stage1_reference.enqueue --server s1 --sensor WV3 --apply
# s2, does not interrupt QB training
python3 -m panda_stage1_reference.enqueue --server s2 --sensor QB --apply
# s3, also performed automatically by the GF2 -> QB cutover
python3 -m panda_stage1_reference.enqueue --server s3 --sensor GF2 --apply
# s3, automatically performed before Student cases once new Stage1 completes
python3 -m panda_stage1_reference.enqueue --server s3 --sensor QB --apply
```

Omit `--apply` for a dry run. Reference identities, unrounded scene metrics,
Sheet request/readback receipts are stored under
`work_dir/panda_stage1_reference/<server>/<sensor>/<run_id>_<step>/`.
Each dataset's `*-ablations` tab gets a distinct `Stage1-reference / Teacher`
row, Bucket=main, four decimals, original D:F/corrected H:J and RR K:R.
An exact result ID makes reruns idempotent; existing experiment rows are not
replaced. Unmeasured inference costs remain blank. The new s3 QB reference
appears only once its local Stage1 has completed and a donor is selected.
