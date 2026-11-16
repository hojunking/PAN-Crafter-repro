"""Always retain one original-PAN representative when finite evidence exists."""
import math
from panda_input3_original.policy import hqnr_passes, passes

POLICY = 'QB_ORIGINAL_PAN_REP_WITH_HQNR_FALLBACK_V5'
REPRESENTATIVES = ('HQNR_QUALIFIED_MIN_ERGAS', 'BEST_HQNR_FALLBACK')


def finite(record):
    return bool(record and all(math.isfinite(record[split]['metrics'][key])
        for split, key in [('fr', 'HQNR'), ('rr', 'ERGAS'), ('rr', 'PSNR')]))


def allowed(record, cfg, selector):
    if cfg['sensor'] != 'QB':
        raise ValueError('This reporting correction is QB only')
    if selector == 'BEST_HQNR_FALLBACK':
        return finite(record) and not hqnr_passes(record, cfg)
    return hqnr_passes(record, cfg) if selector == 'HQNR_QUALIFIED_MIN_ERGAS' else passes(record, cfg)


def choose(records, stage, cfg):
    if stage != 'stage2' or cfg['sensor'] != 'QB':
        raise ValueError('QB Student results only')
    unique = {}
    for r in records:
        if r['step'] in unique and unique[r['step']] != r:
            raise ValueError('Conflicting evaluations of the same checkpoint')
        if not finite(r):
            raise ValueError('Nonfinite representative evidence')
        unique[r['step']] = r
    records = [unique[k] for k in sorted(unique)]
    if not records:
        return []
    qualified = [r for r in records if hqnr_passes(r, cfg)]
    if qualified:
        # Preserve the existing qualified representative and tie-break exactly.
        rep = min(qualified, key=lambda r: (r['rr']['metrics']['ERGAS'],
            -r['rr']['metrics']['PSNR'], -r['fr']['metrics']['HQNR'], r['step']))
        selector = 'HQNR_QUALIFIED_MIN_ERGAS'
    else:
        rep = min(records, key=lambda r: (-r['fr']['metrics']['HQNR'],
            r['rr']['metrics']['ERGAS'], r['step']))
        selector = 'BEST_HQNR_FALLBACK'
    result = [(rep, selector, 'ROW0')]
    for r in records:
        if passes(r, cfg):
            tag = 'PASS_ALL_STEPS'
            if r['step'] == cfg['primary_step']:
                tag += f"|FIXED{cfg['primary_step']//1000}K"
            result.append((r, tag, f"STEP_{r['step']:07d}"))
    return result
