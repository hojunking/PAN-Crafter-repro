# s4 WV3: fresh Direct donor trio → fixed-donor Stage 2

Independent 40-hour campaign. No changes to running s1/s3 code or registries.
No training starts merely by pulling the commit.

On **s4**, after the commit is pushed and pulled:

```bash
git pull --ff-only
bash scripts/run_pandas4_docker.sh
```

Uses the existing Docker image, four registered WV3 H5 files, DLPan checkout,
and server-local `gspread/account.json`. These assets/credentials are not in Git.
`PAN_HOST_PYTHON` can name the existing publisher environment; the launcher also
detects the usual `pancrafter` conda environment. The launcher never stops other
experiments and refuses to run if the GPU is occupied.

Sequence: `D_A1/R1 → D_A1/R2 → D_A3/R1`, all fresh 50K updates, then choose **one**
saved A/T pair. HQNR ≥ 0.959: lowest ERGAS; otherwise highest HQNR, explicitly
exploratory. It must pass checkpoint SHA and official full40 replay verification.
All Stage 2 cases freeze that same A/T pair and freshly initialize the student.
Students use PAN + MS only, **no LPAN/HPAN**.

Stage 2: `G/R1,R2 → Q/R1,R2 → E/R1 → ES/R1 → Q_ALOW/R1 → Q_BHIGH/R1 →
Q_GHIGH/R1 → Q_FULL/R1`. G/Q are paired two-seed anchors. Other cases are
single-seed screening, not confirmed gains. Optional cases can be skipped for
budget. The deadline includes preflight, training, calibration, evaluation,
saving and downtime; restarting does not grant another 40 hours. Admission
reserves all three donors and four Stage 2 anchors using local timing estimates.
H50 is not silently shortened to fit a slow GPU.

MASK2 loss support is inherited from Stage 1; `Q_FULL` is the explicit full-image
control with its own consistent calibration. Official RR20/FR20 metrics still
use original references and **no evaluation masking**. Prints HQNR/SCC/ERGAS.
Deployment A+T/A+S cost is measured, excluding the frozen training-only teacher.

The host publisher uses existing WV3-ablations append/upsert ownership rules.
Display selection (HQNR ≥ 0.960 **and** ERGAS ≤ 2.045, otherwise best HQNR)
remains distinct from donor/checkpoint selection. All evaluations remain local.
No cloud write happens from tests or catalog/check commands.

```bash
python -m pandas4 catalog --csv
python -m pandas4 check                  # data/evaluator/import checks only
python -m pandas4 smoke                  # CPU tests
docker logs -f pandas4-wv3-s4
```

Artifacts: `work_dir/pandas4/PANDA_S4_DONOR_STAGE2_40H_20261017_v1/s4/WV3/`.
See [full plan](../research_log/PANDA_S4_WV3_Donor3_Stage2_40H_2026-10-17.md)
for data-supported motivation, coefficients, budget caveats and interpretation.
