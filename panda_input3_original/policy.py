"""Original PAN HQNR gates; unchanged sensor targets and representative rule."""
import math
from panda_input3 import plan

POLICY = 'INPUT3_ORIGINAL_PAN_HQNR_REP_ALL_TARGET_DETAILS_V4'


def hqnr_passes(record, cfg):
    if not record: return False
    h = record['fr']['metrics']['HQNR']
    return math.isfinite(h) and h >= plan.targets(cfg['sensor'])['HQNR_min']


def passes(record, cfg):
    if not hqnr_passes(record, cfg): return False
    t = plan.targets(cfg['sensor']); rr = record['rr']['metrics']
    e, p = rr['ERGAS'], rr.get('PSNR')
    return (math.isfinite(e) and e <= t['ERGAS_max'] and
            ('PSNR_min' not in t or (p is not None and math.isfinite(p) and p >= t['PSNR_min'])))


def allowed(record, cfg, selector):
    return hqnr_passes(record, cfg) if selector == 'HQNR_QUALIFIED_MIN_ERGAS' else passes(record, cfg)


def choose(records, stage, cfg):
    if stage != 'stage2': raise ValueError('INPUT3 publishes Stage 2 only')
    unique = {}
    for r in records:
        if r['step'] in unique and unique[r['step']] != r:
            raise ValueError('Conflicting evaluations of the same checkpoint')
        unique[r['step']] = r
    qualified = [unique[k] for k in sorted(unique) if hqnr_passes(unique[k], cfg)]
    if not qualified: return []
    rep = min(qualified, key=lambda r: (r['rr']['metrics']['ERGAS'],
        -r['rr']['metrics']['PSNR'], -r['fr']['metrics']['HQNR'], r['step']))
    result = [(rep, 'HQNR_QUALIFIED_MIN_ERGAS', 'ROW0')]
    for r in qualified:
        if passes(r, cfg):
            selector = 'PASS_ALL_STEPS'
            if r['step'] == cfg['primary_step']: selector += f"|FIXED{cfg['primary_step']//1000}K"
            result.append((r, selector, f"STEP_{r['step']:07d}"))
    return result
