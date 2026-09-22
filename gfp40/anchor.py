"""Fail-closed parent re-evaluation against frozen, full-precision anchors.

1e-6 absolute / zero relative tolerance is a pre-launch engineering equality
assumption, NOT a performance gate or a measured 5090 reproducibility bound.
It is never relaxed automatically when a server fails this check.
"""
from __future__ import annotations

import math
from pathlib import Path

from gfp40.common import ROOT, immutable_json, object_sha, read_config, read_json, resolved_path
from gfp40.plan import CAMPAIGN_ID, case_for
from gfp40.assets import REGISTRY_PATH, registry
from fh12.common import sha256
from g20.postrun import validate_record

ATOL = 1e-6
RTOL = 0.
METRICS = ('HQNR', 'ERGAS', 'D_lambda', 'D_s')
RULE = 'PRELAUNCH_ENGINEERING_EQUALITY_ATOL_1e-6_RTOL_0_v1'


def anchor_metrics(record):
    return dict(HQNR=record['fr']['hqnr'], ERGAS=record['rr']['ergas'],
                D_lambda=record['fr']['d_lambda'], D_s=record['fr']['d_s'])


def compare_metrics(expected, actual):
    """JSON-safe failure evidence also for NaN/Inf; never drop a bad metric."""
    rows = {}
    for metric in METRICS:
        try:
            target, measured = float(expected[metric]), float(actual[metric])
            finite = math.isfinite(target) and math.isfinite(measured)
        except (ValueError, TypeError, KeyError):
            target, measured, finite = math.nan, math.nan, False
        delta = measured - target if finite else None
        rows[metric] = dict(expected=target if math.isfinite(target) else None,
            actual=measured if math.isfinite(measured) else None, finite=finite,
            delta=delta, abs_delta=abs(delta) if finite else None, atol=ATOL, rtol=RTOL,
            passed=bool(finite and abs(delta) <= ATOL))
    return dict(rule=RULE, passed=all(row['passed'] for row in rows.values()), metrics=rows,
        interpretation='Engineering equality check, not a paper performance threshold',
        tolerance_basis='Fixed before launch; actual 5090/runtime reproducibility remains to be measured',
        automatic_relaxation=False, substitute_parent_allowed=False)


def expected_anchor(case, parent, root=ROOT):
    case = case_for(case)
    if not case.is_ft or parent.get('parent_id') != case.parent_id or parent.get('server') != case.server:
        raise ValueError('Anchor parent/case/local-server identity differs')
    if parent.get('parent_step') != case.parent_step:
        raise ValueError('Anchor is not the registered exact parent endpoint')
    parent_id = case.parent_id
    PARENT_ASSETS = registry()
    if parent_id in PARENT_ASSETS['parents']:
        reported = PARENT_ASSETS['parents'][parent_id]
        if (parent['parent_run_id'] != reported['source_run_id']
                or parent['parent_step'] != reported['parent_step']
                or (reported['published_model_sha256'] is not None
                    and parent['parent_model_sha256'] != reported['published_model_sha256'])):
            raise ValueError('Re-evaluated parent differs from published run/step/SHA')
        return reported['reported_metrics'], dict(kind='DERIVED_VERIFIED_B20_REGISTRY_FULL_PRECISION',
            path=REGISTRY_PATH, sha256=sha256(Path(ROOT) / REGISTRY_PATH),
            provenance=PARENT_ASSETS['provenance'], source_cell=reported['source_cell'],
            published_model_sha256=reported['published_model_sha256'],
            resolved_parent_model_sha256=parent['parent_model_sha256'])
    if parent.get('parent_kind') != 'GFP40_FRESH_TRUNK':
        raise ValueError('Unregistered parent has no permissible anchor')
    trunk = case_for(parent_id)
    if trunk.is_ft or trunk.updates != 100000 or trunk.server != case.server:
        raise ValueError('Only registered same-server fresh100K trunk anchors are permissible')
    cfg = read_config(resolved_path(parent['parent_config'], root))
    folder = resolved_path(cfg['work_dir'], root)
    source = folder / 'official/summary.json'
    summary = read_json(source)
    row = summary['selections']['EXACT_FINAL']
    identity = read_json(resolved_path(parent['parent_identity'], root))
    if (parent['parent_run_id'] != trunk.run_id or folder.name != trunk.run_id
            or summary.get('campaign_id') != CAMPAIGN_ID or summary.get('run_id') != trunk.run_id
            or summary.get('case_id') != trunk.case_id or summary.get('official_complete') is not True
            or row.get('selection_id') != 'EXACT_FINAL' or row.get('update') != 100000
            or row.get('checkpoint_identity') != identity
            or identity.get('model_sha256') != parent['parent_model_sha256']
            or identity.get('state_hash') != parent['parent_tensor_hashes']['full']
            or summary.get('config_sha256') != object_sha(cfg)):
        raise ValueError('Fresh parent anchor must be its same-checkpoint exact100K trunk result')
    validate_record(row, 'GF2')
    # The summary also contains mutable reporting costs/timestamps. They are
    # not the scientific anchor and cannot invalidate an identical FT resume.
    # Bind only the checked numerical selection and full checkpoint identity.
    numerical_selection = {key: row[key] for key in
        ('update', 'selection_id', 'checkpoint_identity', 'rr', 'fr', 'val_ergas')}
    return anchor_metrics(row), dict(kind='VERIFIED_FRESH_TRUNK_EXACT_FINAL', path=str(source),
        numerical_selection_sha256=object_sha(numerical_selection),
        parent_model_sha256=parent['parent_model_sha256'],
        parent_checkpoint_identity_sha256=object_sha(identity), source_case_id=trunk.case_id)


def verify_and_save_parent_anchor(case, parent, anchor, output_path, root=ROOT):
    """Save the pass/failure comparison before permitting the first update."""
    expected, source = expected_anchor(case, parent, root)
    result = compare_metrics(expected, anchor_metrics(anchor))
    errors = []
    if (anchor.get('model_state_hash') != parent['parent_tensor_hashes']['full']
            or anchor.get('parent_step') != parent['parent_step']
            or anchor.get('candidate_eligible') is not False):
        errors.append('Parent full tensor identity/step/candidate exclusion mismatch')
    try:
        validate_record(dict(anchor, update=parent['parent_step'],
            checkpoint_identity={'update': parent['parent_step']}), 'GF2')
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    # A NaN anchor cannot be canonically JSON-serialized; the comparison still
    # records every finite/nonfinite field without fabricating an input hash.
    try:
        anchor_sha = object_sha(anchor)
    except (ValueError, TypeError):
        anchor_sha = None
    result.update(schema='GFP40_PARENT_ANCHOR_COMPARISON_v1', campaign_id=CAMPAIGN_ID,
        case_id=case_for(case).case_id, parent_id=parent['parent_id'],
        parent_run_id=parent['parent_run_id'], parent_step=parent['parent_step'],
        parent_model_sha256=parent['parent_model_sha256'], anchor_sha256=anchor_sha,
        expected_source=source, integrity_errors=errors, passed=result['passed'] and not errors,
        re_evaluation='native RR20/FR20 gamma1; full512 original PAN; no masking',
        evaluated_before_first_optimizer=True)
    immutable_json(output_path, result)
    if not result['passed']:
        raise ValueError('BLOCKED_INTEGRITY: parent native anchor equality failed; no tolerance relaxation or substitution')
    return result
