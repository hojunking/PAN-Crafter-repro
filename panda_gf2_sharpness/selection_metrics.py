"""CPU diagnostics for saved predictions; no model loading or image modification."""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from panda_sensor_visuals.metrics import gradient, local_std, measure, rgb_display


def shared_rgb(raw, bounds):
    lo, hi = np.asarray(bounds, np.float64).T
    rgb = np.asarray(raw, np.float64)[[2, 1, 0]].transpose(1, 2, 0)
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def diagnostics(lum, pan):
    g = gradient(pan)
    edges = g >= np.quantile(g, .90)
    flat_stat = local_std(pan, 9)
    flat = flat_stat <= np.quantile(flat_stat, .20)
    out_g = gradient(lum)
    hp = pan - gaussian_filter(pan, 1.5)
    hf = lum - gaussian_filter(lum, 1.5)
    denom = np.sqrt(np.sum((hf-hf.mean())**2) * np.sum((hp-hp.mean())**2))
    corr = float(np.sum((hf - hf.mean()) * (hp - hp.mean())) / denom) if denom > 1e-12 else None
    return dict(edge_pm2=float(maximum_filter(out_g, 5)[edges].mean()),
                edge_exact=float(out_g[edges].mean()), hf=float(np.abs(hf).mean()),
                flat_grain=float(local_std(lum, 5)[flat].mean()), hf_corr_pan=corr)


def measure_saved(raw, pan_rgb, bounds, peak):
    """Legacy PNG protocol plus unquantized shared-reference and raw-DN views."""
    pan = np.asarray(pan_rgb, np.float64).mean(2) / 255
    legacy = measure(rgb_display(raw, peak), pan_rgb)
    shared = diagnostics(shared_rgb(raw, bounds).mean(2), pan)
    # Raw prediction is deliberately not clipped; numerical overshoots are visible.
    raw_diag = diagnostics(np.asarray(raw, np.float64)[[2, 1, 0]].mean(0) / peak, pan)
    result = {f'{mode}_{k}': v for mode, values in [('legacy', legacy), ('shared', shared), ('raw', raw_diag)]
              for k, v in values.items()}
    result['out_of_range_fraction'] = float(np.mean((raw < 0) | (raw > peak)))
    return result


def pareto(rows, objectives):
    """Return indices not dominated in every specified (key, maximize) objective."""
    good = []
    for i, a in enumerate(rows):
        av = np.array([a[key] * (1 if maximize else -1) for key, maximize in objectives])
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            bv = np.array([b[key] * (1 if maximize else -1) for key, maximize in objectives])
            if np.all(bv >= av) and np.any(bv > av):
                dominated = True
                break
        if not dominated:
            good.append(i)
    return good


def classify(rows, scene_rows, baseline_id, reference, *, min_wins=14, grain_ratio_max=1.0,
             edge_retention=.95, top_k=5):
    """Explicit gates and trade-offs, never a weighted opaque overall score."""
    base = next(r for r in rows if r['candidate_id'] == baseline_id)
    by_scene = {r['scene']: r for r in scene_rows if r['candidate_id'] == baseline_id}
    if len(by_scene) != 20 or any(b[k] <= 1e-12 for b in by_scene.values()
                                  for k in ('shared_edge_pm2', 'raw_edge_pm2')):
        raise ValueError('Baseline requires 20 nondegenerate scenes for paired comparisons')
    for row in rows:
        scenes = sorted((r for r in scene_rows if r['candidate_id'] == row['candidate_id']), key=lambda r: r['scene'])
        changes = []
        wins = 0
        for s in scenes:
            b = by_scene[s['scene']]
            s['shared_edge_change_pct'] = 100 * (s['shared_edge_pm2'] / b['shared_edge_pm2'] - 1)
            s['raw_edge_change_pct'] = 100 * (s['raw_edge_pm2'] / b['raw_edge_pm2'] - 1)
            s['joint_edge_win'] = (s['shared_edge_pm2'] > b['shared_edge_pm2'] + 1e-10
                                   and s['raw_edge_pm2'] > b['raw_edge_pm2'] + 1e-10)
            changes.append(s['shared_edge_change_pct'])
            wins += int(s['joint_edge_win'])
        row.update(edge_change_pct=100 * (row['shared_edge_pm2'] / base['shared_edge_pm2'] - 1),
                   raw_edge_change_pct=100 * (row['raw_edge_pm2'] / base['raw_edge_pm2'] - 1),
                   shared_grain_ratio=row['shared_flat_grain'] / base['shared_flat_grain'],
                   raw_grain_ratio=row['raw_flat_grain'] / base['raw_flat_grain'],
                   improvement_scenes=wins, median_scene_change_pct=float(np.median(changes)),
                   worst_scene_change_pct=min(changes),
                   worst_scene=min(scenes, key=lambda s: s['shared_edge_change_pct'])['scene'])
        row['legacy_gate'] = (row['legacy_edge_pm2'] >= reference['U-Know']['edge_pm2']
                              and row['legacy_flat_grain'] <= reference['Ours']['flat_grain'])
        row['legacy_strong_gate'] = (row['legacy_gate'] and
                                     row['legacy_edge_pm2'] >= reference['PAN-Crafter (s5)']['edge_pm2'])
        row['raw_shared_guard'] = (row['edge_change_pct'] >= -1e-9 and row['raw_edge_change_pct'] >= -1e-9
                                   and row['shared_grain_ratio'] <= grain_ratio_max + 1e-9
                                   and row['raw_grain_ratio'] <= grain_ratio_max + 1e-9)
        row['visual_gate'] = row['legacy_gate'] and row['raw_shared_guard'] and wins >= min_wins
        row['quantitative_gate'] = row['HQNR'] >= .9645 and row['ERGAS'] <= .522 and row['PSNR'] >= 45.08
        row['human_review'] = 'PENDING'
        row['pareto_visual'] = False
        row['pareto_joint'] = False
    for i in pareto(rows, [('shared_edge_pm2', True), ('shared_flat_grain', False)]):
        rows[i]['pareto_visual'] = True
    for i in pareto(rows, [('shared_edge_pm2', True), ('shared_flat_grain', False),
                           ('HQNR', True), ('ERGAS', False), ('PSNR', True)]):
        rows[i]['pareto_joint'] = True
    order = lambda r: (-r['shared_edge_pm2'], r['shared_flat_grain'], r['candidate_id'])
    visual = sorted((r for r in rows if r['visual_gate']), key=order)
    numerical = sorted((r for r in rows if r['quantitative_gate']), key=order)
    provisional = sorted((r for r in rows if r['pareto_visual']), key=order)
    compromise = []
    if visual:
        limit = edge_retention * visual[0]['shared_edge_pm2']
        compromise = sorted((r for r in rows if r['raw_shared_guard'] and r['pareto_joint']
                             and r['improvement_scenes'] >= min_wins and r['shared_edge_pm2'] >= limit),
                            key=lambda r: (r['ERGAS'], -r['HQNR'], -r['PSNR'], r['candidate_id']))
    lists = dict(visual=visual, quantitative=numerical, compromise=compromise,
                 provisional_frontier=provisional)
    return {key: [r['candidate_id'] for r in value[:top_k]] for key, value in lists.items()}
