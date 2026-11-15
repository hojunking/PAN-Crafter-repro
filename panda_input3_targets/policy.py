"""HQNR-gated representative; all dataset targets for other checkpoint rows."""
import math
from panda_input3 import plan

POLICY = 'INPUT3_HQNR_REP_AND_ALL_TARGET_DETAILS_V3'


def hqnr_passes(record, cfg):
    if not record:
        return False
    h=record['fr_corrected_pan']['metrics']['HQNR']
    return math.isfinite(h) and h >= plan.targets(cfg['sensor'])['HQNR_min']


def allowed(record, cfg, selector):
    return hqnr_passes(record,cfg) if selector=='HQNR_QUALIFIED_MIN_ERGAS' else passes(record,cfg)


def passes(record, cfg):
    if not record:
        return False
    target = plan.targets(cfg['sensor'])
    h = record['fr_corrected_pan']['metrics']['HQNR']
    e = record['rr']['metrics']['ERGAS']
    p = record['rr']['metrics'].get('PSNR')
    return (math.isfinite(h) and math.isfinite(e)
            and h >= target['HQNR_min'] and e <= target['ERGAS_max']
            and ('PSNR_min' not in target or
                 (p is not None and math.isfinite(p) and p >= target['PSNR_min'])))


def choose(records, stage, cfg):
    if stage != 'stage2':
        raise ValueError('INPUT3 publishes Stage 2 only')
    unique = {}
    for record in records:
        step = record['step']
        if step in unique and unique[step] != record:
            raise ValueError('Conflicting evaluations of the same checkpoint')
        unique[step] = record
    qualified = [unique[k] for k in sorted(unique) if hqnr_passes(unique[k], cfg)]
    if not qualified:
        return []
    rep = min(qualified, key=lambda r: (r['rr']['metrics']['ERGAS'],
              -r['rr']['metrics']['PSNR'], -r['fr_corrected_pan']['metrics']['HQNR'], r['step']))
    result = [(rep, 'HQNR_QUALIFIED_MIN_ERGAS', 'ROW0')]
    for record in qualified:
        if not passes(record,cfg):
            continue
        selector = 'PASS_ALL_STEPS'
        if record['step'] == cfg['primary_step']:
            selector += f"|FIXED{cfg['primary_step']//1000}K"
        result.append((record, selector, f"STEP_{record['step']:07d}"))
    return result
