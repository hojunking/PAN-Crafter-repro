"""Recompute descriptive EQREC4 correlations from stored metrics, without inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path('/home/knuvi/Desktop/song/PAN-Crafter')
OUT = Path(__file__).resolve().parent
src = ROOT / 'work_dir/_eqrec4_s1_campaign/native_sample_metrics.csv'
cols = ['model_key', 'scale', 'sample_id', 'source_group_id', 'split_role',
        'q_A', 'e_native_roi', 'pan_scharr_energy', 'raw_original_hqnr']
d = pd.read_csv(src, usecols=cols)
keys = [f'{family}_S{seed}_best_raw'
        for family in ('L1E4', 'L000', 'L1E2') for seed in (1234, 7777, 2025)]
d = d[d.model_key.isin(keys)]
d.to_csv(OUT / 'eqrec4_selected_input_columns.csv', index=False)
rows = []
for key, g in d[d.scale == 'native64'].groupby('model_key'):
    g = g.dropna(subset=['q_A', 'e_native_roi', 'pan_scharr_energy'])
    rx, ry, rz = [rankdata(g[k]) for k in ('q_A', 'e_native_roi', 'pan_scharr_energy')]
    z = np.c_[np.ones(len(g)), rz]
    ex = rx - z @ np.linalg.lstsq(z, rx, rcond=None)[0]
    ey = ry - z @ np.linalg.lstsq(z, ry, rcond=None)[0]
    rows.append(dict(model_key=key, n=len(g), source_blocks=g.source_group_id.nunique(),
                     raw_rho=float(spearmanr(g.q_A, g.e_native_roi).statistic),
                     report_texture_adjusted_rho=float(spearmanr(ex, ey).statistic),
                     conventional_partial_rank_rho=float(np.corrcoef(ex, ey)[0, 1])))
pd.DataFrame(rows).to_csv(OUT / 'eqrec4_texture_correlations.csv', index=False)
scene_rows = []
for key, g in d[(d.scale == 'fr512') & d.model_key.str.startswith('L1E4_')].groupby('model_key'):
    g = g.dropna(subset=['q_A', 'raw_original_hqnr'])
    scene_rows.append(dict(model_key=key, n=len(g),
                           rho_q_hqnr=float(spearmanr(g.q_A, g.raw_original_hqnr).statistic)))
pd.DataFrame(scene_rows).to_csv(OUT / 'eqrec4_scene_correlations.csv', index=False)
manifest = dict(source=str(src.relative_to(ROOT)), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                selected_columns=cols,
                methods={'report': 'Spearman of residuals after regressing each variable rank on texture rank and intercept',
                         'conventional': 'Pearson of those residuals (partial correlation of ranks)'},
                native=rows, fr=scene_rows,
                limitations=['Descriptive correlations only; no causal inference or independent-patch significance tests',
                             'Native figures use all 4096 patches, unlike original H1 BCD or D-only subsets',
                             'FR texture-adjusted coefficient is not recomputed here'])
(OUT / 'eqrec4_recheck.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
