# WV3-main four-decimal publication

User change on2026-11-06: numeric result/cost columns D:Y use **four decimal
places**. Selection still uses unrounded evaluation evidence. LR, step, seed,
configuration, target definitions, metrics and cost scopes are unchanged.

This overlay wraps the existing local `panda_wv3_main_reporting` and
`panda_wv3_w128` publishers in a separate process. It does not edit the running
campaign's469 pinned source files or invalidate optimizer/RNG continuation.
It restores the fourth digit from original metrics/cost/timing evidence, rather
than appending a zero to a previously rounded three-digit cell. Both old3dp rows
and new4dp rows are authenticated against immutable evidence.

The s1 W128 trainer moved from `panda-wv3-w128-s1-v1` to
`panda-wv3-w128-s1-4dp-v1` after saving its complete20,442-step resume checkpoint.
Use the new container/entrypoint for this active campaign:

```bash
python -m panda_wv3_reporting4.cutover --status
python -m panda_wv3_reporting4.cutover --resume --apply
docker logs --tail 30 panda-wv3-w128-s1-4dp-v1
```

The two original model packages and their local data must already be present;
this reporting-only overlay is not a replacement for the training packages.
Do not restart the old3dp trainer alongside the4dp trainer.

For historical W112 main rows, including removed checkpoints authenticated by
the s1 retirement ledger:

```bash
python -m panda_wv3_reporting4 project --kind rgb --output work_dir/_operator_storage/rgb4.json
python -m panda_wv3_reporting4 publish --kind rgb --apply
```

This supersedes the old three-decimal RGB republication command. Historical
metrics, row identities and checkpoint-retirement notes remain intact. The
existing cleanup manifest/evaluation/deletion journal must be present locally.
Other dataset/ablation publishers are unaffected.
