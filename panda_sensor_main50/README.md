# QB / GF2 main follow-up: five recipes per server, 50K without metric stops

Campaign `PANDA_S23_MAIN50_20261114_v1`. **s2 runs QB; s3 runs GF2.**
This is main hyperparameter tuning, not a component ablation or a three-seed
report. Each server has five fresh Stage 2 runs with paired seed 261018101.
No new Stage 1 training and no transfer of weights between servers.

## Evidence and purpose

The latest QB ZEROQ R1/R2 were stopped at 20K under the remote corrected-HQNR
rule. Their Sheet representatives at 4K/5K had corrected HQNR .9413/.9445 and
ERGAS 3.7124/3.6709. They do not establish the full 50K trajectory. Retain the
same local donor and test schedule length, LR, and interaction strength.

GF2's completed unified Stage 1 R2@42K has corrected HQNR .9746, ERGAS .516366,
and PSNR45.059975. It passes the corrected H/E goals; PSNR is about .0200 dB
short of45.08. The old Stage 1 admission gate incorrectly prevents trying a
Student until the Teacher already passes the final Student PSNR goal. This new
campaign explicitly pins the available local Teacher/Aligner and does not apply
performance gates to Stage 2 admission. This does not claim the Student will
necessarily improve on the Teacher.

An older GF2 W112/residual-input/RGB recipe at LR2e-4 reached PSNR45.084138.
It motivates the LR neighborhood only; its different method and original-PAN
HQNR do not prove that the current RGBN/corrected-only recipe will pass.

## Registered cases, execution order

| Server | Case | Peak LR | Gamma | Purpose |
|---|---|---:|---:|---|
| s2 QB | Q1_BASE_LR100_G025 | 1e-4 | .25 | Full50K reference |
| s2 QB | Q2_LR050_G025 | 5e-5 | .25 | Smaller optimization steps |
| s2 QB | Q3_LR150_G025 | 1.5e-4 | .25 | Higher LR/convergence comparison |
| s2 QB | Q4_LR100_G010 | 1e-4 | .10 | Weaker adaptive weighting |
| s2 QB | Q5_LR100_G050 | 1e-4 | .50 | Stronger adaptive weighting |
| s3 GF2 | G1_BASE_LR200_G025 | 2e-4 | .25 | Student baseline from current local donor |
| s3 GF2 | G2_LR150_G025 | 1.5e-4 | .25 | Lower LR |
| s3 GF2 | G3_LR250_G025 | 2.5e-4 | .25 | Higher LR |
| s3 GF2 | G4_LR200_G0125 | 2e-4 | .125 | Half adaptive strength |
| s3 GF2 | G5_LR200_G050 | 2e-4 | .50 | Double adaptive strength |

All run 50,000 optimizer updates, warmup100 then cosine to zero **at50K**.
These are fresh schedules from step0, not 40K runs relabelled or resumed with
an incompatible schedule. W128/D123 UBIL, AdamW WD.01, batch48/micro12,
FP32, full Aligner scale1, alpha=beta=0, and `w=1+gamma*d_e*r_q` throughout.
RGBN arithmetic mean for L_spa/q/g/RRE, Lanczos-4 for Teacher/Student,
common eroded valid support, and the exact-zero rank guard remain unchanged.
Input is `[corrected PAN, interp23 I_lrms4]`, five channels; output four bands.
No correction residual, global rank, rotation loss/TTA, EMA/SWA, or new loss.

Only LR or gamma differs from that sensor's Q1/G1 reference. Same donor,
initial Student parameters, data/augmentation streams, and schedule for all
five cases. One seed per setting supports screening, not seed-robust claims.

## Local fixed donor and calibration

- QB: `UNIFIED_QB_S2_S1_RGBN_FULL_R1_S261016301`, **32K**, same donor used by
  the stopped ZEROQ campaign. SHA-256
  `37c7d718233e6d85c89fd96301c31ca8ff93fe7b2bd6d3f767b61e3b86d84724`.
- GF2: `SEQ40_GF2_S3_S1_RGBN_FULL_R2_S261016302`, **42K**, SHA-256
  `b69fe5619123f2c8cc4d3bb5f184f5108a6253eec3dd45f8547fc3fec7badd06`.
- Source Stage1 run must have finished50K. Its original config, admission data
  and evaluator pins, checkpoint sidecar/hash, and evaluation hash are checked.
  The new receipt records the explicitly chosen checkpoint and discloses that
  its selection used reported test results. No silent fallback to another
  donor, RGB model, other server, or previous method.
- Full-train calibration is computed once per pinned local donor in the new
  output directory with the unchanged shared calibration implementation;
  all five cases reuse that artifact. Original campaigns are mounted read-only.

## No performance early stopping

HQNR, ERGAS, PSNR, and donor target misses cannot stop or skip any run.
The former QB10K and remote20K rules are absent from this runner. Numerical
errors, invalid support, missing/corrupt evidence, and hardware failures still
raise explicit errors; these are not converted into successful50K runs.
Resumption restores complete optimizer, RNG, and data-stream state.

Unchanged reporting goals:

| Sensor | Corrected-PAN HQNR | RR ERGAS | RR PSNR |
|---|---:|---:|---:|
| QB | >= .9315 | <= 3.5 | no floor |
| GF2 | >= .9645 | <= .522 | >= 45.08 |

Each1K saves native RR20/FR20 metrics and an evaluated checkpoint. FR records
both original and same-Aligner corrected PAN references for the same output;
corrected PAN drives selection. Final50K is always reported, even on failure
to meet all goals. Every checkpoint remains during this active campaign for
trajectory inspection; main retention cleanup is a separate later operation.

## Sheet

Continue the existing experiment-log destinations **QB-ablations / GF2-ablations**,
with `Bucket=main` and Experiment `QB-MAIN50-...` / `GF2-MAIN50-...`. The tab
name is historical; these are main runs. Four decimals, original FR D:F,
corrected FR H:J, registered recipe and donor in notes.

Preserve H-qualified minimum-ERGAS representative, else maximum corrected H;
publish all H/E passes and fixed50K. Also publish labelled **unrestricted
minimum ERGAS** and **unrestricted maximum PSNR** diagnostic slots, so low-H
late convergence is visible. Slots may point to the same checkpoint and are
labelled by purpose, not counted as separate runs. GF2 always annotates
three-goal pass/fail. Unmeasured inference costs stay blank.

## Remote launch / handoff

Push the commit from s1, then pull it on each server. Use the same release;
`release.py` verifies the complete shared code and data-independent plan.
Do not restore an old modified early-stop policy onto this package. Its
independent runner never installs `panda_corrected_hqnr` policy hooks.

```bash
# s2 (local marker gspread/server.txt must be s2)
python3 -m panda_sensor_main50 catalog --server s2
python3 -m panda_sensor_main50.enqueue --server s2 --check-only
python3 -m panda_sensor_main50.enqueue --server s2

# s3 (local marker must be s3)
python3 -m panda_sensor_main50 catalog --server s3
python3 -m panda_sensor_main50.enqueue --server s3 --check-only
python3 -m panda_sensor_main50.enqueue --server s3
```

Default image: `hojunqueen/pancrafter-env:torch2.4.0-cu118`; an equivalent pinned
image can be supplied with `--image`, external toolbox with `--dlpan`.
The launcher checks local source donor and Sheet schema before creating a
detached container. It publishes the queue, waits for current GPU work to
finish, then runs a disposable native CUDA preflight and starts training.
It never stops unrelated work or automatically restarts old queues.

```bash
docker logs --tail 40 panda-sensor-main50-s2-v1
docker logs --tail 40 panda-sensor-main50-s3-v1
# On the corresponding server, after inspecting a stopped container:
python3 -m panda_sensor_main50.enqueue --server s2 --resume
python3 -m panda_sensor_main50.enqueue --server s3 --resume
```

Output root: `work_dir/panda_sensor_main50/PANDA_S23_MAIN50_20261114_v1/<server>/<sensor>`.
Check `campaign_state.json`, `queue_wait.json`, `preflight/verified.json`,
`donor/selection.json`, `publication/last_poll.json`, and per-run evaluations.

Local validation: all ten synthetic four-band optimizer paths, frozen
Aligner/Teacher integrity, exact CPU resume, shared initialization, base-method
forward/loss parity, zero-discrepancy guard, full50K/no-stop schedules, Sheet
selection semantics, and read-only source mounts. These tests do not claim
the remote donor files exist or that remote GPU training has started; those
checks execute on s2/s3 when the above command runs.
