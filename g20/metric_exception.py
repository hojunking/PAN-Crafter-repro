"""User-authorized s1/R0 metric variance, never a strict parity PASS.

The original evaluator, reports, thresholds and failed replay are unchanged.
This separately seals bounded acceptance of both exact50K asset replays on
the existing registered environment. Every reuse authenticates the actual
original asset/data chains and recomputes comparisons from measured reports.
"""
from pathlib import Path

from g20.common import (CAMPAIGN_ID, camp, check_deadline, immutable_json,
                        object_sha, read_json, sha256, source_identity)


AUTH_SCHEMA = 'G20_S1_R0_METRIC_EXCEPTION_AUTH_v1'
RECEIPT_SCHEMA = 'G20_S1_R0_METRIC_EXCEPTION_v1'
SEAL_SCHEMA = 'G20_S1_R0_METRIC_EXCEPTION_SEAL_v1'
APPROVAL_TEXT = '아니다 그냥 진행해. 환경을 맞추지 않아도 괜찮으니 실험 진행해'
POLICY = dict(abs_difference=dict(hqnr=2e-5, ergas=1e-4),
              strict_tolerance=dict(hqnr=1e-5, ergas=1e-4),
              labels=['TA', 'S92001'], training_changed=False,
              environment_changed=False, clock_changed=False,
              origin_artifacts_changed=False)


def authorization_path(root, server='s1'):
    return camp(root, server) / 'metric_exception_authorization.json'


def authorization_fields(root, server='s1', current_source=None):
    """Exact required authorization fields; this does not write approval."""
    from g20.references import R0_PINS
    if server != 's1':
        raise ValueError('Metric exception is authorized only for s1/R0')
    registration_path = camp(root, server) / 'registration.json'
    registration = read_json(registration_path)
    identity = registration['identity']
    current = source_identity(root) if current_source is None else current_source
    if (identity.get('campaign_id') != CAMPAIGN_ID or identity.get('server_id') != 's1'
            or identity.get('device') != 'cuda'):
        raise ValueError('Metric exception requires the original s1 CUDA registration')
    mutable = {'files', 'content_sha256', 'git_release'}
    if ({key: value for key, value in identity['source_identity'].items() if key not in mutable}
            != {key: value for key, value in current.items() if key not in mutable}):
        raise ValueError('Metric exception cannot change the registered execution environment')
    return dict(schema=AUTH_SCHEMA, campaign_id=CAMPAIGN_ID, server_id='s1',
                reference_id='R0', user_approved=True, approval_text=APPROVAL_TEXT,
                original_registration_sha256=sha256(registration_path),
                consumer_source_identity=current,
                teacher_checkpoint_sha256=R0_PINS['teacher_checkpoint_sha256'],
                q_cache_sha256=R0_PINS['q_cache_sha256'],
                origin_manifest_sha256=R0_PINS['calibration_id'], policy=POLICY)


def authorization(root, server='s1'):
    expected = authorization_fields(root, server)
    auth = read_json(authorization_path(root, server))
    if (auth.get('user_approved') is not True
            or any(auth.get(key) != value for key, value in expected.items())):
        raise ValueError('Metric exception authorization identity/policy mismatch')
    return auth


def accept_or_validate(root, server='s1', deadline=None):
    """Accept a fresh normal replay, or verify its existing immutable seal.

    Call the unchanged ``verify_r0_parity(..., device='cuda')`` first. This
    function performs no inference and cannot turn missing/partial/stale replay
    data into acceptance. Cached acceptance still revalidates actual asset and
    dataset bytes, the full measured protocol, and both strict/capped results.
    """
    from g20.parity import EVIDENCE_SCHEMA, _prior, _metric_pair, compare_metrics
    check_deadline(deadline)
    root = Path(root)
    auth = authorization(root, server)
    release = auth['consumer_source_identity']
    folder = camp(root, server)
    auth_path = authorization_path(root, server)
    evidence_path = folder / 'r0_parity_evidence.json'
    raw_path = folder / 'diagnostics/r0_evaluator_parity.json'
    receipt_path = folder / 'diagnostics/r0_evaluator_metric_exception.json'
    seal_path = receipt_path.with_suffix('.seal.json')
    # A strict PASS is handled by the original verifier, never this exception.
    if raw_path.with_suffix('.seal.json').exists():
        raise ValueError('Metric exception cannot replace or reuse a strict parity PASS seal')
    hashes = {str(path): sha256(path) for path in (auth_path, evidence_path, raw_path,
              folder / 'registration.json', folder / 'dataset_manifest.json')}
    evidence, raw = read_json(evidence_path), read_json(raw_path)
    local = read_json(folder / 'dataset_manifest.json')
    if (evidence.get('schema') != EVIDENCE_SCHEMA
            or set(evidence.get('assets', {})) != {'TA', 'S92001'}):
        raise ValueError('Metric exception requires both original TA/S92001 evidence chains')
    if (raw.get('schema') != 'G20_R0_EVALUATOR_PARITY_v1'
            or raw.get('status') != 'BLOCKED_PARITY' or raw.get('passed') is not False
            or raw.get('source_identity') != release
            or raw.get('evidence_sha256') != hashes[str(evidence_path)]
            or raw.get('local_data_sha256') != object_sha(local)
            or set(raw.get('assets', {})) != {'TA', 'S92001'}):
        raise ValueError('Metric exception requires a complete current-source strict FAILED replay')
    rows = {}
    for label in ('TA', 'S92001'):
        check_deadline(deadline)
        _, _, prior, equality, _ = _prior(evidence['assets'][label], label, local,
                                         root, deadline=deadline)
        measured = raw['assets'][label]
        comparison = compare_metrics(prior, _metric_pair(measured['measured']))
        if (comparison['tolerance'] != POLICY['strict_tolerance']
                or measured.get('passed') is not comparison['passed']
                or any(measured.get(key) != value for key, value in comparison.items())
                or measured.get('checkpoint_sha256') != evidence['assets'][label]['checkpoint_sha256']
                or measured.get('prior_report_sha256') != evidence['assets'][label]['prior_report_sha256']
                or measured.get('data_equality') != equality):
            raise ValueError('Metric exception replay measurement binding differs: ' + label)
        if any(comparison['abs_difference'][key] > cap
               for key, cap in POLICY['abs_difference'].items()):
            raise ValueError('Metric exception approved variance cap exceeded: ' + label)
        rows[label] = dict(comparison, accepted=True,
                           checkpoint_sha256=measured['checkpoint_sha256'],
                           prior_report_sha256=measured['prior_report_sha256'],
                           data_equality=equality)
    if all(row['passed'] for row in rows.values()):
        raise ValueError('Metric exception requires at least one actual strict parity failure')
    if (source_identity(root) != release
            or any(sha256(Path(path)) != digest for path, digest in hashes.items())):
        raise ValueError('Metric exception source or input artifacts changed during validation')
    check_deadline(deadline)
    receipt = dict(schema=RECEIPT_SCHEMA, campaign_id=CAMPAIGN_ID, server_id='s1',
                   reference_id='R0', status='ACCEPTED_WITH_METRIC_EXCEPTION',
                   accepted=True, passed=False, strict_passed=False, policy=POLICY,
                   source_identity=release, local_data_sha256=object_sha(local),
                   original_registration_sha256=hashes[str(folder / 'registration.json')],
                   authorization_path=str(auth_path), authorization_sha256=hashes[str(auth_path)],
                   raw_receipt_path=str(raw_path), raw_receipt_sha256=hashes[str(raw_path)],
                   evidence_path=str(evidence_path), evidence_sha256=hashes[str(evidence_path)],
                   assets=rows)
    if seal_path.exists() and (not receipt_path.is_file()
            or read_json(seal_path).get('receipt_sha256') != sha256(receipt_path)):
        raise ValueError('Immutable metric exception receipt changed')
    immutable_json(receipt_path, receipt)
    seal = dict(schema=SEAL_SCHEMA, receipt_sha256=sha256(receipt_path),
                raw_receipt_sha256=hashes[str(raw_path)],
                authorization_sha256=hashes[str(auth_path)],
                evidence_sha256=hashes[str(evidence_path)], source_sha256=object_sha(release))
    immutable_json(seal_path, seal)
    return dict(receipt, receipt_path=str(receipt_path), receipt_sha256=sha256(receipt_path),
                seal_path=str(seal_path), seal_sha256=sha256(seal_path))
