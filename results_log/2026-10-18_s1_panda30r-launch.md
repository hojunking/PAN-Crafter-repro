# s1 PANDA30R launch — 2026-10-18

- Campaign: `PANDA_S134_DONOR_SCHEDULE_30H_20261018_v1`, s1 / WV3.
- Container: `panda30r-wv3-s1`; original start 2026-10-18 17:59:09 KST.
- Deadline: original start +30h, approximately 2026-10-19 23:59 KST;
  closeout reserve30min. Recovery downtime remains inside this budget.
- Existing experiments had already exited; no prior job was stopped or deleted.

## Launch-time compatibility fix

The late donor24 50K safe-resume checkpoint has a minimal identity without a
`dataset` key. Inventory correctly proved its A/T tensors against official50K
evaluation, but the reused strict donor loader additionally required that key.
The first preparation attempt therefore stopped before any registered training.

Added isolated `panda30r/source_donor.py`: only a `resume_only=True`,
`evaluation=None` payload may omit identity.dataset. Full config sensor,
native training source sensor/manifest, actual A/T tensor hashes, run/step and
file SHA remain mandatory. An explicitly wrong dataset is never normalized.
Source checkpoint files and old campaign code were not modified.

`recover_preflight.py` preserves a complete audit of the failed admission and
allows re-admission only before registered runs/pilots have completed. It does
not permit post-training code-pin changes or restart the30h clock.

Actual CPU imports passed for both exact source files. On the GPU, full40 replay
matched early7K and late50K source metrics. The first100-update disposable mature
Teacher pilot passed full40 evaluation and complete checkpoint resume checks.

Measured pilot: update p90 .142034s; full RR20+FR20 evaluation19.920s;
complete checkpoint32,800,168 bytes. A+T deployment cost profile was produced on
the actual GPU. These are preparation timings, not final method performance.

## Confirmed production state

At 2026-10-18 18:04:57 KST, `MATURE` was running at local235 / lifetime7235.
Early A is frozen, original T optimizer/sampler/RNG/LR continue toward lifetime50K.
The100-update pilot is not included in that lineage and is not uploaded.
After maturity, registered s1 Student cases proceed automatically under the
remaining wall-time/disk gates.

Host publisher restarted successfully after the preparation repair. Google
Sheets readback was verified with zero numerical rows before the first official
production evaluation; this is not a claim that a Student result already exists.

Regression coverage added: valid minimal resume import; rejection of missing
dataset in non-resume, conflicting dataset/source/hash; preparation-only repair,
foreign-code/config refusal and preserved deadline.

After the repair, the complete `panda30r smoke` suite passed **102 tests** in
42.970s with CUDA hidden from tests; the production GPU job continued separately.
