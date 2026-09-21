"""Pure LOCAL-T clock, whole-block reservations and descriptive paired analysis.

Performance classifications never authorize, postpone or expand the core queue.
The local controller alone validates runtime evidence and persists admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from statistics import median

from .plan import (ADMISSION_CUTOFF_HOURS, CALIBRATION_HOURS, CAMPAIGN_ID,
    CLOSE_HOURS, TRAIN_FINISH_HOURS, WINDOW_HOURS, Block, Case, block_for,
    blocks_for, case_for, registry_sha256, validate_case)


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('A timezone-aware timestamp is required')
    return value.astimezone(timezone.utc)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hours(value, name):
    if not _finite(value) or value < 0:
        raise ValueError(name + ' must be finite and nonnegative')
    return float(value)


@dataclass(frozen=True)
class CampaignWindow:
    t0_utc: datetime | str

    def __post_init__(self):
        object.__setattr__(self, 't0_utc', _utc(self.t0_utc))

    @property
    def deadline_utc(self):
        return self.t0_utc + timedelta(hours=WINDOW_HOURS)

    @property
    def admission_cutoff_utc(self):
        return self.t0_utc + timedelta(hours=ADMISSION_CUTOFF_HOURS)

    @property
    def train_finish_utc(self):
        return self.t0_utc + timedelta(hours=TRAIN_FINISH_HOURS)

    @property
    def training_deadline_utc(self):
        return self.train_finish_utc

    def remaining_hours(self, now):
        return max(0., (self.deadline_utc - _utc(now)).total_seconds() / 3600.)

    def to_dict(self):
        return dict(campaign_id=CAMPAIGN_ID, t0_utc=self.t0_utc.isoformat(),
            deadline_utc=self.deadline_utc.isoformat(), admission_cutoff_utc=self.admission_cutoff_utc.isoformat(),
            train_finish_utc=self.train_finish_utc.isoformat(), close_hours=CLOSE_HOURS)

    @classmethod
    def from_dict(cls, value):
        result = cls(value['t0_utc'])
        for key, expected in result.to_dict().items():
            if value.get(key) != expected:
                raise ValueError('Changed/incomplete LOCAL-T shared clock: ' + key)
        return result


def _canonical_block(block):
    if isinstance(block, str):
        return block_for(block)
    if not isinstance(block, Block) or block != block_for(block.block_id):
        raise ValueError('Admission requires one complete canonical atomic block and order')
    return block


def _matching_observations(case, observations, component):
    result = []
    for row in observations:
        if not isinstance(row, dict):
            row = vars(row)
        if (row.get('server_id') != case.server_id or row.get('sensor', 'GF2') != 'GF2'
                or row.get('role') != case.role or row.get('updates') != case.updates
                or row.get('width') != case.width or tuple(row.get('depth', ())) != case.depth
                or row.get('component') != component or not _finite(row.get('hours')) or row['hours'] < 0):
            continue
        if component == 'WARMUP_EXTRAPOLATED':
            if row.get('warmup_excluded') is not True or row.get('includes_eval_and_io') is not True:
                continue
        elif row.get('completed') is not True or row.get('evaluation_valid') is not True:
            continue
        result.append(float(row['hours']))
    return result[-5:]


def reservation_estimate(case, observations=(), *, calibration=False):
    """Initial reservation is a floor; larger recent/warmup evidence gets ×1.15.

    No unsupported downward extrapolation, optimizer-only timing, or measurements
    from another server/horizon can silently cheapen a local reservation.
    """
    case = validate_case(case_for(case) if isinstance(case, str) else case)
    if calibration and case.role != 'T':
        raise ValueError('Calibration reservation belongs to a local Teacher endpoint')
    initial = CALIBRATION_HOURS if calibration else case.reservation_hours
    component = 'CALIBRATION' if calibration else 'TRAIN_EVAL'
    recent = _matching_observations(case, observations, component)
    warmup = [] if calibration else _matching_observations(case, observations, 'WARMUP_EXTRAPOLATED')
    observed = max(recent + warmup, default=0.)
    return dict(initial_hours=initial, recent_hours=recent, warmup_hours=warmup,
        safety_factor=1.15, evidence_hours=observed,
        hours=max(initial, observed * 1.15), downward_revision=False,
        basis='max(initial, 1.15 * max(last5 representative full-subprocess/eval, warmup-extrapolated))')


def estimate_case_hours(case, observations=()):
    return reservation_estimate(case, observations)['hours']


def estimate_calibration_hours(teacher, observations=()):
    return reservation_estimate(teacher, observations, calibration=True)['hours']


def estimate_block_hours(block, observations=(), completed_run_ids=(), calibrated_reference_ids=()):
    block = _canonical_block(block)
    completed, calibrated = set(completed_run_ids), set(calibrated_reference_ids)
    if not completed.issubset(block.run_ids):
        # Callers must scope completed work to the exact block, avoiding an
        # unrelated server/reference receipt accidentally cancelling a cost.
        raise ValueError('Completed run IDs must belong to the reserved local block')
    expected_refs = {c.reference_id for c in block.cases if c.role == 'T'}
    if not calibrated.issubset(expected_refs):
        raise ValueError('Calibration IDs must be local Teacher references in this block')
    return sum(estimate_case_hours(c, observations) for c in block.cases if c.run_id not in completed) + sum(
        estimate_calibration_hours(c, observations) for c in block.cases if c.role == 'T' and c.reference_id not in calibrated)


def _validate_admitted(receipt, block, window):
    if (not isinstance(receipt, dict) or receipt.get('allowed') is not True
            or receipt.get('reason') != 'ADMITTED' or receipt.get('campaign_id') != CAMPAIGN_ID
            or receipt.get('server_id') != block.server_id or receipt.get('block_id') != block.block_id
            or receipt.get('run_ids') != list(block.run_ids) or receipt.get('window') != window.to_dict()
            or receipt.get('registry_sha256') != registry_sha256()):
        raise ValueError('Continuation lacks the original complete immutable admission receipt')
    admitted = _utc(receipt['at_utc'])
    if not window.t0_utc <= admitted < window.admission_cutoff_utc:
        raise ValueError('Original block was not admitted before the cutoff')
    required = _hours(receipt.get('required_hours'), 'Original reservation')
    if admitted + timedelta(hours=required) >= window.train_finish_utc:
        raise ValueError('Original reservation did not fit before the optimizer deadline')


def evaluate_admission(window, now, block, remaining_hours, evaluation_debt_hours=0,
                       admitted_receipt=None, *, p0_ready=True):
    """Admit a whole local chain/pair or finish its already-reserved suffix.

    ``remaining_hours`` includes unfinished training *and calibration* for the
    entire canonical block. Deferred Teacher grid work is additionally retained
    in ``evaluation_debt_hours``; fast local Student transition is not free eval.
    """
    window = window if isinstance(window, CampaignWindow) else CampaignWindow.from_dict(window)
    now, block = _utc(now), _canonical_block(block)
    remaining = _hours(remaining_hours, 'Remaining whole-block reservation')
    debt = _hours(evaluation_debt_hours, 'Evaluation debt')
    if now < window.t0_utc:
        raise ValueError('Cannot admit before the actual common start')
    continuation = admitted_receipt is not None
    if continuation:
        _validate_admitted(admitted_receipt, block, window)
        if now < _utc(admitted_receipt['at_utc']):
            raise ValueError('Continuation precedes original admission')
    elif remaining < estimate_block_hours(block):
        raise ValueError('New block reservation cannot omit cases/calibration or silently lower initial estimates')
    required = remaining + debt
    if p0_ready is not True:
        reason = 'BLOCKED_LOCAL_INTEGRITY'
    elif now >= window.train_finish_utc:
        reason = 'OPTIMIZER_CLOSED'
    elif not continuation and now >= window.admission_cutoff_utc:
        reason = 'NOT_ADMITTED_CUTOFF'
    elif (now + timedelta(hours=required) >= window.train_finish_utc
          or now + timedelta(hours=required + CLOSE_HOURS) >= window.deadline_utc):
        reason = 'NOT_ADMITTED_BUDGET'
    else:
        reason = 'CONTINUE_ADMITTED_BLOCK' if continuation else 'ADMITTED'
    return dict(campaign_id=CAMPAIGN_ID, server_id=block.server_id, block_id=block.block_id,
        registry_sha256=registry_sha256(), window=window.to_dict(), at_utc=now.isoformat(),
        run_ids=list(block.run_ids), allowed=reason in ('ADMITTED', 'CONTINUE_ADMITTED_BLOCK'), reason=reason,
        remaining_hours=remaining, evaluation_debt_hours=debt, required_hours=required,
        train_finish_utc=window.train_finish_utc.isoformat(), deadline_utc=window.deadline_utc.isoformat(),
        original_admission_at_utc=admitted_receipt['at_utc'] if continuation else None,
        performance_gate=False, deferred_auto_release=False)


CONTRASTS = {
    'S3_STUDENT_LENGTH': (('S01', 'S02'), ('S06', 'S05')),
    'S3_TEACHER_LENGTH': (('S02', 'S03'), ('S05', 'S04')),
    'S3_LONG_CHAIN': (('S01', 'S03'), ('S06', 'S04')),
    'S4_H010': (('S07', 'S08'), ('S10', 'S09')),
    'S5_E1': (('S11', 'S12'), ('S14', 'S13'))}


def _metrics(value):
    if not isinstance(value, dict) or not all(_finite(value.get(k)) for k in ('HQNR', 'ERGAS', 'D_lambda')):
        raise ValueError('Finite same-checkpoint HQNR/ERGAS/D_lambda are required')
    if not 0 <= value['HQNR'] <= 1 or value['ERGAS'] <= 0 or not 0 <= value['D_lambda'] <= 1:
        raise ValueError('Metric value outside its accepted domain')
    return value


def classify_pairs(contrast, results):
    """MD §10 analysis only, never an execution/admission gate.

    The caller supplies proof-validated reports keyed by exact case ID. Each
    report has ``exact_final`` and ``rr_val_selected`` three-metric dictionaries
    and explicit true ``official_complete/same_checkpoint_verified/pair_verified``.
    Missing one half/seed yields PAIR_INCOMPLETE, not a cross-server substitute.
    """
    if contrast not in CONTRASTS:
        raise ValueError('Unregistered local paired contrast')
    assignments = [('L100I1-' + a, 'L100I1-' + b) for a, b in CONTRASTS[contrast]]
    if any(identifier not in results for pair in assignments for identifier in pair):
        return dict(contrast=contrast, classification=['PAIR_INCOMPLETE'], execution_gate=False)
    rows = []
    for baseline_id, candidate_id in assignments:
        base_case, alt_case = case_for(baseline_id), case_for(candidate_id)
        if base_case.server_id != alt_case.server_id or base_case.student_seed != alt_case.student_seed:
            raise ValueError('Only same-server same-Student-seed local pairs are valid')
        base, alt = results[baseline_id], results[candidate_id]
        for value, case in ((base, base_case), (alt, alt_case)):
            if (value.get('case_id') != case.case_id or value.get('server_id') != case.server_id
                    or value.get('actual_updates') != case.updates or value.get('reference_id') != case.reference_id
                    or any(value.get(k) is not True for k in ('official_complete', 'same_checkpoint_verified', 'pair_verified'))):
                raise ValueError('Paired report identity/completion/evidence mismatch')
        selections = {}
        for selection in ('exact_final', 'rr_val_selected'):
            a, b = _metrics(base[selection]), _metrics(alt[selection])
            selections[selection] = dict(delta_H=b['HQNR'] - a['HQNR'],
                relative_E_percent=100 * (b['ERGAS'] / a['ERGAS'] - 1),
                delta_D_lambda=b['D_lambda'] - a['D_lambda'],
                joint=b['HQNR'] > .964 and b['ERGAS'] < .552,
                strong_joint=b['HQNR'] > .964 and b['ERGAS'] < .522)
        rows.append(dict(seed=alt_case.student_seed, baseline=baseline_id, candidate=candidate_id, **selections))
    exact, val = [[row[selection] for row in rows] for selection in ('exact_final', 'rr_val_selected')]
    def e_guard(items):
        return median(r['relative_E_percent'] for r in items) <= .5 and all(r['relative_E_percent'] <= 1 for r in items)
    def rr_gain(items):
        return (all(r['relative_E_percent'] < 0 and r['delta_H'] >= -.001 for r in items)
                and median(r['relative_E_percent'] for r in items) <= -.5)
    classes = []
    if (all(r['delta_H'] > 1e-5 for r in exact) and median(r['delta_H'] for r in exact) >= .0015
            and e_guard(exact) and all(r['delta_D_lambda'] <= .002 for r in exact)
            and median(r['delta_D_lambda'] for r in exact) <= .001
            and e_guard(val) and median(r['delta_H'] for r in val) >= -.0015):
        classes.append('H_PROMISING')
    # §10 asks for the same VAL direction, not a second ≥0.5% magnitude test.
    val_direction = all(r['relative_E_percent'] < 0 and r['delta_H'] >= -.001 for r in val)
    if rr_gain(exact) and val_direction:
        classes.append('RR_GAIN_WITH_H_PRESERVED')
    if all(r['joint'] for r in val):
        classes.append('JOINT_TWO_SEED')
    return dict(contrast=contrast, classification=classes or ['NO_REGISTERED_GAIN'], paired_rows=rows,
        execution_gate=False, statistical_significance_claim=False, automatic_recipe_promotion=False)
