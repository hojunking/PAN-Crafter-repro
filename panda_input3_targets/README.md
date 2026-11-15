# INPUT3 publication: HQNR-qualified representative and all-target detail rows

This reporting-only release keeps the HQNR-qualified representative, removes
the HQNR-failing fallback and unrestricted ERGAS/PSNR diagnostic rows, and
requires all dataset goals for other result rows. It does not change training,
the original release SHA, immutable run configs, donor, optimizer/RNG resume,
input layouts, RGBN, Lanczos-4, masks, widths, or schedules.

| Dataset | corrected-PAN FR HQNR | RR ERGAS | RR PSNR |
|---|---:|---:|---:|
| WV3 | >= .9585 | <= 2.040 | no gate |
| QB | >= .9315 | <= 3.500 | no gate |
| GF2 | >= .9645 | <= .522 | >= 45.08 dB |

Use unrounded evaluation evidence. The representative is minimum ERGAS among
checkpoints meeting the corrected-HQNR goal; ties prefer PSNR, corrected HQNR,
then earlier step. This representative may miss ERGAS and/or GF2 PSNR, which
are explicitly labelled rather than called full goal achievement. All other
step rows, including endpoints, require every dataset goal simultaneously at
the same checkpoint. There is no best-HQNR fallback when HQNR itself misses,
and no unrestricted diagnostic upload. A case with no HQNR-qualified checkpoint
retains a status-only row with empty metric cells. All local evaluations and
weights remain available for analysis.

## Rollout on each server after the user's push

```bash
git pull --ff-only
python3 -m panda_input3_targets.release
python3 -m panda_input3_targets.enqueue --server s2
# s1 uses --server s1; s3 uses --server s3.
```

The launcher verifies the new reporting code and old immutable training release,
checks the local donor and Sheet, gracefully stops only this server's old INPUT3
container, checks the complete saved checkpoint, and resumes the same run in
`panda-input3-targets-<server>-v1`. Completed cases are not retrained. Do not
restart the old INPUT3 container or use its old publisher after migration.

The first publication backs up and removes obsolete INPUT3 result slots without
leaving empty rows, then replaces HQNR-ineligible representatives with status
rows. Foreign campaigns and all local models/evaluations remain intact. This
migration is performed only after stopping the old publisher on that server.
If a retired row has external annotations to the right of BO, it is preserved
and migration reports the conflict instead of deleting them.

Remote old publishers may keep running until rollout. An INPUT3-only Sheet
filter can temporarily hide their ineligible metric rows while preserving their
unchanged evidence, queued status rows, and all other campaigns. This display
filter keeps HQNR-qualified representatives even when RR targets are missed.
It is not a replacement for running the new launcher on s2/s3.

Validation:
`python -m unittest panda_input3_targets.test_reporting -v` in the production
container; original `panda_input3.test_input3` runs separately without the new
reporting monkey patches. New policy ID: `INPUT3_HQNR_REP_AND_ALL_TARGET_DETAILS_V3`.
