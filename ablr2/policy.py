"""Pure ABLR2 operating statistics and finite adaptive choices (not significance).

The caller verifies same-checkpoint, local-reference and panel identities before
passing metrics here. Missing or invalid observations cannot be resampled away.
"""
from __future__ import annotations

import math
from statistics import median

from .plan import COMPARISON_GRAPH, GRAPH, MAIN_CASES, RECIPES, sensor_spec

IMPROVEMENTS = ('JOINT_GAIN', 'H_GAIN_SAFE', 'RR_GAIN_SAFE')
THRESHOLD_REVISION = 'thresholds_v1'


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _metrics(value, *, fitting=False):
    keys = ('HQNR', 'ERGAS', 'E_val', 'D_lambda') if fitting else ('HQNR', 'ERGAS')
    if not isinstance(value, dict) or not all(_finite(value.get(key)) for key in keys):
        raise ValueError('Finite same-checkpoint metric observations are required')
    if not 0 <= value['HQNR'] <= 1 or value['ERGAS'] <= 0:
        raise ValueError('Invalid HQNR/ERGAS domain')
    if fitting and (value['E_val'] <= 0 or not 0 <= value['D_lambda'] <= 1):
        raise ValueError('Invalid validation ERGAS/D_lambda domain')
    return value


def paired_deltas(pairs):
    out = []
    for pair in pairs:
        p, c = _metrics(pair['parent']), _metrics(pair['child'])
        out.append(dict(delta_H=c['HQNR'] - p['HQNR'], rE=c['ERGAS'] / p['ERGAS'] - 1.))
    return out


def _mad(values):
    center = median(values)
    return median(abs(v - center) for v in values)


def calibrate_thresholds(sensor, panels):
    """Exactly five complete 17-case BOOT5 panels, per sensor, frozen once."""
    sensor_spec(sensor)
    if len(panels) != 5 or any(set(panel) != set(MAIN_CASES) for panel in panels):
        raise ValueError('PILOT_INCOMPLETE: thresholds require all 17 cases in all five BOOT5 panels')
    for panel in panels:
        for value in panel.values(): _metrics(value)
    rows = []
    for relation in COMPARISON_GRAPH:
        pairs = [dict(parent=panel[relation['parent']], child=panel[relation['child']]) for panel in panels]
        deltas = paired_deltas(pairs)
        rows.append(dict(relation_id=relation['relation_id'], MAD_H=_mad([r['delta_H'] for r in deltas]),
            MAD_E=_mad([r['rE'] for r in deltas]), paired_deltas=deltas))
    raw_h = .25 * median(row['MAD_H'] for row in rows)
    raw_e = .25 * median(row['MAD_E'] for row in rows)
    eps_h, eps_e = min(max(raw_h, 1e-5), 1e-3), min(max(raw_e, 1e-4), .005)
    return dict(schema='ABLR2_THRESHOLDS_v1', threshold_revision=THRESHOLD_REVISION, sensor=sensor,
        source='FIRST_COMPLETE_BOOT5_PER_SENSOR', scope='SENSOR_GLOBAL_NOT_EDGE_SPECIFIC',
        epsilon_H=eps_h, epsilon_E=eps_e, delta_H=max(.001, 2 * eps_h), delta_E=max(.003, 2 * eps_e),
        raw_epsilon_H=raw_h, raw_epsilon_E=raw_e, H_cap_applied=raw_h != eps_h, E_cap_applied=raw_e != eps_e,
        support_fraction=.8, panel_count=5, relation_count=17, per_relation=rows,
        frozen=True, statistical_significance_claim=False, independent_relation_count_claim=False)


def validate_thresholds(value):
    if not isinstance(value, dict) or any(not _finite(value.get(k)) for k in ('epsilon_H', 'epsilon_E', 'delta_H', 'delta_E')):
        raise ValueError('Measured frozen operating thresholds are required, never defaults from another sensor')
    h, e = value['epsilon_H'], value['epsilon_E']
    if not 1e-5 <= h <= .001 or not 1e-4 <= e <= .005 or value['delta_H'] != max(.001, 2*h) or value['delta_E'] != max(.003, 2*e):
        raise ValueError('Changed threshold formula requires an operator-approved revision')
    return value


def _gain(deltas, thresholds):
    eh, ee, dh, de = (thresholds[k] for k in ('epsilon_H', 'epsilon_E', 'delta_H', 'delta_E'))
    n = len(deltas)
    support = math.ceil(.8*n)
    h, e = median(row['delta_H'] for row in deltas), median(row['rE'] for row in deltas)
    if h >= dh and e <= -de and sum(row['delta_H'] >= -eh and row['rE'] <= ee for row in deltas) >= support:
        return 'JOINT_GAIN'
    if h >= dh and sum(row['delta_H'] > eh for row in deltas) >= support and e <= ee and all(row['rE'] <= .01 for row in deltas):
        return 'H_GAIN_SAFE'
    if e <= -de and sum(row['rE'] < -ee for row in deltas) >= support and h >= -eh and all(row['delta_H'] >= -max(.003, 2*eh) for row in deltas):
        return 'RR_GAIN_SAFE'
    return None


def classify_relation(pairs, thresholds, exact_pairs=None):
    thresholds = validate_thresholds(thresholds)
    if len(pairs) < 5:
        return dict(classification='INCOMPLETE', reason='PILOT_INCOMPLETE', n=len(pairs), flags=[], valid=False)
    try:
        deltas = paired_deltas(pairs)
        inverse = paired_deltas([dict(parent=r['child'], child=r['parent']) for r in pairs])
    except (ValueError, KeyError, TypeError) as exc:
        return dict(classification='INVALID', reason=str(exc), n=len(pairs), flags=[], valid=False)
    h, e = median(row['delta_H'] for row in deltas), median(row['rE'] for row in deltas)
    gain, reverse = _gain(deltas, thresholds), _gain(inverse, thresholds)
    if gain:
        status = gain
    elif reverse:
        status = 'REVERSAL'
    elif (h > thresholds['epsilon_H'] and e > thresholds['epsilon_E']) or (h < -thresholds['epsilon_H'] and e < -thresholds['epsilon_E']):
        status = 'TRADEOFF'
    elif abs(h) <= thresholds['epsilon_H'] and abs(e) <= thresholds['epsilon_E']:
        status = 'NEAR_ZERO'
    else:
        status = 'UNSTABLE'
    result = dict(classification=status, n=len(pairs), support_required=math.ceil(.8*len(pairs)),
        median_delta_H=h, median_rE=e, paired_deltas=deltas, inverse_deltas=inverse,
        reverse_gain_rule=reverse, flags=[], valid=True, statistical_significance_claim=False)
    if exact_pairs is not None:
        exact = classify_relation(exact_pairs, thresholds)
        result['exact50k'] = exact
        # MD does not quantify "large opposite". Use its frozen improvement
        # rules in the opposite direction, not an invented or tunable margin.
        if gain and exact['classification'] == 'REVERSAL':
            result['flags'].append('CHECKPOINT_SENSITIVE')
        result['checkpoint_sensitivity_definition'] = 'VAL_IMPROVEMENT_AND_EXACT50K_INVERSE_REGISTERED_GAIN'
    return result


def choose_relation(classifications, rechecked_relation_ids=(), last_measurements=None):
    """One finite recheck per exact relation/recipe; caller scopes the history."""
    done, last = set(rechecked_relation_ids), last_measurements or {}
    ranked = []
    for relation_id, report in classifications.items():
        if relation_id not in GRAPH: raise ValueError('Unknown comparison graph relation')
        if relation_id in done: continue
        status = report['classification']
        if status in ('INVALID', 'INCOMPLETE'): continue  # separate technical repair, never seed replacement
        adverse_full = relation_id in ('E00', 'D10') and (report.get('median_delta_H', 0) < 0 or report.get('median_rE', 0) > 0)
        if status == 'REVERSAL' or adverse_full: priority = 0
        elif 'CHECKPOINT_SENSITIVE' in report.get('flags', ()) or status == 'UNSTABLE': priority = 1
        elif status == 'TRADEOFF': priority = 2
        elif status == 'NEAR_ZERO': priority = 3
        else: continue
        ranked.append((priority, last.get(relation_id, ''), relation_id))
    return min(ranked)[2] if ranked else None


_RECIPE_PRIORITY = {'L04': 'R01', 'D13': 'R01', 'L05': 'R03', 'D12': 'R03',
    'L06': 'R02', 'D14': 'R02', 'D11': 'R04', 'L03': 'R05', 'D16': 'R05'}


def choose_recipes(relation_id, tried_recipes=()):
    if relation_id is not None and relation_id not in GRAPH: raise ValueError('Unknown relation')
    first = _RECIPE_PRIORITY.get(relation_id)
    bank = ([first] if first else []) + [r for r in ('R01', 'R02', 'R03', 'R04', 'R05') if r != first]
    return tuple(recipe for recipe in bank if recipe not in tried_recipes)[:2]


def screen_candidate(recipe_id, pairs, thresholds):
    """A two-block FULL-only absolute quality screen; no ablation-gap scoring."""
    if recipe_id not in RECIPES: raise ValueError('Unregistered recipe')
    thresholds = validate_thresholds(thresholds)
    if len(pairs) != 2: return dict(recipe_id=recipe_id, eligible=False, status='SCREEN_INCOMPLETE')
    try:
        rows = []
        for pair in pairs:
            parent, child = _metrics(pair['parent'], fitting=True), _metrics(pair['child'], fitting=True)
            rows.append(dict(delta_H=child['HQNR'] - parent['HQNR'], rE_val=child['E_val']/parent['E_val']-1,
                delta_D_lambda=child['D_lambda']-parent['D_lambda'], E_val=child['E_val'], HQNR=child['HQNR']))
    except (ValueError, KeyError, TypeError) as exc:
        return dict(recipe_id=recipe_id, eligible=False, status='SCREEN_INVALID', reason=str(exc))
    h, ev, dl = (median(row[k] for row in rows) for k in ('delta_H', 'rE_val', 'delta_D_lambda'))
    dguard = dl <= .001 and all(row['delta_D_lambda'] <= .002 for row in rows)
    eval_path = (all(row['rE_val'] < 0 for row in rows) and ev <= -.003 and dguard
        and all(row['delta_H'] >= -max(.003, 2*thresholds['epsilon_H']) for row in rows))
    h_path = (all(row['delta_H'] > thresholds['epsilon_H'] for row in rows) and h >= thresholds['delta_H']
        and ev <= .005 and all(row['rE_val'] <= .01 for row in rows) and dguard)
    return dict(recipe_id=recipe_id, eligible=eval_path or h_path, eval_path=eval_path, h_path=h_path,
        both_paths=eval_path and h_path, status='PROVISIONAL_RECIPE' if eval_path or h_path else 'REJECTED_DEV',
        median_E_val=median(row['E_val'] for row in rows), median_H=median(row['HQNR'] for row in rows),
        paired_rows=rows, test_aware=True, objective='FULL_ABSOLUTE_QUALITY_NOT_ABLATION_GAP',
        statistical_significance_claim=False)


def choose_recipe(incumbent_id, candidate_reports):
    if incumbent_id not in RECIPES: raise ValueError('Unregistered incumbent')
    if len(candidate_reports) > 2: raise ValueError('At most two candidates may be screened per cycle')
    valid = [r for r in candidate_reports if r.get('eligible') is True]
    if any(r['recipe_id'] not in RECIPES for r in candidate_reports): raise ValueError('Unregistered candidate')
    if len({r['recipe_id'] for r in candidate_reports}) != len(candidate_reports): raise ValueError('Duplicate candidate')
    if not valid: return incumbent_id
    return min(valid, key=lambda r: (not r['both_paths'], r['median_E_val'], -r['median_H'], r['recipe_id']))['recipe_id']


def recheck_outcome(original, recheck):
    a, b = original['classification'], recheck['classification']
    if b in ('INVALID', 'INCOMPLETE'): return b
    if a == 'REVERSAL' and b == 'REVERSAL': return 'REPEATED_NEGATIVE_DEV'
    if (a == 'REVERSAL' and b in IMPROVEMENTS) or (a in IMPROVEMENTS and b == 'REVERSAL'):
        return 'SEED/REFERENCE_SENSITIVE_DEV'
    if b in IMPROVEMENTS: return 'SUPPORTED_DEV'
    if b == 'NEAR_ZERO': return 'SMALL_OR_UNDETECTED_DEV'
    if b == 'TRADEOFF': return 'PERSISTENT_TRADEOFF_DEV'
    return 'SEED/REFERENCE_SENSITIVE_DEV'


def flow_dashboard(classifications):
    if set(classifications) != set(GRAPH): raise ValueError('All 17 graph relations are required')
    valid = all(r['classification'] not in ('INVALID', 'INCOMPLETE') for r in classifications.values())
    def counts(kind):
        records = [classifications[r['relation_id']] for r in COMPARISON_GRAPH if r['kind'] == kind]
        return dict(improvement=sum(r['classification'] in IMPROVEMENTS for r in records),
            reversal=sum(r['classification'] == 'REVERSAL' for r in records),
            tradeoff=sum(r['classification'] == 'TRADEOFF' for r in records),
            undetected=sum(r['classification'] in ('NEAR_ZERO', 'UNSTABLE') for r in records))
    flow = valid and classifications['E00']['classification'] in IMPROVEMENTS and not any(
        classifications[r['relation_id']]['classification'] == 'REVERSAL' for r in COMPARISON_GRAPH
        if r['kind'] in ('LADDER', 'FULL_MINUS'))
    return dict(ladder=counts('LADDER'), full_minus=counts('FULL_MINUS'), E00=classifications['E00'],
        alert='FLOW_CANDIDATE_DEV' if flow else None, early_stop=False, automatic_promotion=False,
        monotonicity_proven=False, statistical_significance_claim=False)
