"""Pure local branch decisions and conservative atomic QG40 reservations.

No function starts a clock, writes a receipt, or launches training. Callers
persist the returned evidence before activating a branch or admitting a block.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math
from statistics import median

from .plan import (ADMISSION_CUTOFF_HOURS, C3_CUTOFF_HOURS, CALIBRATION_HOURS,
                   CAMPAIGN_ID, CLOSE_HOURS, STUDENT_PROFILES, WINDOW_HOURS,
                   Case, case_for, cases_for, sensor_spec)


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Explicit timezone-aware campaign UTC timestamps required")
    return value.astimezone(timezone.utc)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _range(value, lower=None, upper=None, lower_strict=False, upper_strict=False):
    if not _finite(value):
        return False
    return ((lower is None or (value > lower if lower_strict else value >= lower)) and
            (upper is None or (value < upper if upper_strict else value <= upper)))


def _pair(values, predicate):
    return isinstance(values, (list, tuple)) and len(values) == 2 and all(predicate(v) for v in values)


@dataclass(frozen=True)
class CampaignWindow:
    t0_utc: datetime | str

    def __post_init__(self):
        object.__setattr__(self, "t0_utc", _utc(self.t0_utc))

    @property
    def deadline_utc(self):
        return self.t0_utc + timedelta(hours=WINDOW_HOURS)

    @property
    def admission_cutoff_utc(self):
        return self.t0_utc + timedelta(hours=ADMISSION_CUTOFF_HOURS)

    @property
    def c3_cutoff_utc(self):
        return self.t0_utc + timedelta(hours=C3_CUTOFF_HOURS)

    def remaining_hours(self, now):
        return max(0., (self.deadline_utc - _utc(now)).total_seconds() / 3600)

    def to_dict(self):
        return dict(campaign_id=CAMPAIGN_ID, t0_utc=self.t0_utc.isoformat(),
                    deadline_utc=self.deadline_utc.isoformat(),
                    admission_cutoff_utc=self.admission_cutoff_utc.isoformat(),
                    c3_cutoff_utc=self.c3_cutoff_utc.isoformat(), close_hours=CLOSE_HOURS)

    @classmethod
    def from_dict(cls, value):
        result = cls(value["t0_utc"])
        expected = result.to_dict()
        for key in ("campaign_id", "deadline_utc", "admission_cutoff_utc", "c3_cutoff_utc", "close_hours"):
            if key in value and value[key] != expected[key]:
                raise ValueError(f"Shared QG40 window changed: {key}")
        return result


def choose_student_trial(server, evidence, selected_at_utc, previous=None):
    """Choose once from §7.1; missing evidence, malformed values or budget => BASE.

    Array measurements are ordered [completed update 24240, completed update
    50000]. A24R deltas compare A24240+U50000 against normal A50000+U50000.
    Evidence files are provenance references whose contents callers validate.
    """
    if server not in ("s2", "s5"):
        raise ValueError("Only s2/s5 may choose a Student trial")
    timestamp = _utc(selected_at_utc).isoformat()
    if previous is not None:
        if (previous.get("campaign_id") != CAMPAIGN_ID or previous.get("server_id") != server or
                previous.get("profile") not in STUDENT_PROFILES or previous.get("decision") != "STUDENT_TRIAL"):
            raise ValueError("Invalid existing branch receipt")
        # This prevents a failed candidate being replaced by another candidate.
        return dict(previous)
    ready = all(evidence.get(key) is True for key in
                ("p0_passed", "baseline_complete", "diagnostics_complete", "budget_available"))
    ready = ready and bool(evidence.get("evidence_files"))
    a = evidence.get("a24r") or {}
    b = evidence.get("b20") or {}
    e = evidence.get("e10") or {}
    criteria = dict(
        A24R=(all(a.get(key) is True for key in ("native_c_changed", "diagonal_reproduced",
                                                "asset_sha_verified", "lp_synchronized")) and
              _range(a.get("delta_h"), .001) and _range(a.get("delta_ds"), upper=-.001) and
              _range(a.get("relative_delta_e"), upper=.005)),
        B20=(b.get("has_h_eligible") is True and b.get("e_goal_missed") is True and
             b.get("gate_implementation_verified") is True and
             _pair(b.get("advantage_positive_fraction"), lambda v: _range(v, .10, 1.)) and
             _pair(b.get("soft_hard_gradient_ratio_median"), lambda v: _range(v, 0., .20, lower_strict=True))),
        E10=(all(e.get(key) is True for key in ("lp_phase_valid", "band_order_valid", "metric_valid")) and
             _range(e.get("late_delta_ds"), .001) and _range(e.get("late_delta_dlambda"), .001) and
             _pair(e.get("edge_hard_gradient_cosine"), lambda v: _range(v, -1., 0., upper_strict=True)) and
             _pair(e.get("weighted_edge_hard_gradient_ratio"), lambda v: _range(v, .10))))
    profile = next((name for name in ("A24R", "B20", "E10") if ready and criteria[name]), "BASE")
    return dict(campaign_id=CAMPAIGN_ID, server_id=server, decision="STUDENT_TRIAL", profile=profile,
                selected_at_utc=timestamp, criteria=criteria, prerequisites_passed=bool(ready),
                reason="FIRST_QUALIFYING_AXIS" if profile != "BASE" else "NO_SUPPORTED_AFFORDABLE_TRIAL",
                evidence=evidence, test_aware=True)


def c3_eligible(server, evidence):
    if server not in ("s1", "s3"):
        raise ValueError("Only s1/s3 may admit C3")
    expected_seeds = (82001, 82002) if server == "s1" else (92001, 92002)
    completed = evidence.get("parent_student_seeds_completed", ())
    criteria = dict(
        parent_complete=(evidence.get("parent_teacher_50k_complete") is True and
                         isinstance(completed, (list, tuple)) and tuple(sorted(completed)) == expected_seeds and
                         evidence.get("parent_students_official_complete") is True),
        local_valid=all(evidence.get(key) is True for key in
                        ("p0_passed", "sign_valid", "units_valid", "cache_valid",
                         "augmentation_valid", "lp_ms_phase_valid")),
        qref=_range(evidence.get("q_ref"), .90 * .46875),
        response_gain=_range(evidence.get("response_gain_median"), upper=.20),
        gradient=_pair(evidence.get("weighted_consistency_rec_a_gradient_ratio"),
                       lambda v: _range(v, 0., .10, upper_strict=True)),
        evidence=bool(evidence.get("evidence_files")))
    allowed = all(criteria.values())
    return dict(campaign_id=CAMPAIGN_ID, server_id=server, decision="C3_EVIDENCE",
                eligible=allowed, criteria=criteria, evidence=evidence,
                reason="SUPPORTED_C3_SENSITIVITY" if allowed else "NOT_ADMITTED_CONDITION",
                requires_full_package_reservation=True)


@dataclass(frozen=True)
class SeedResult:
    seed: int
    target_eligible: bool
    target_e: float | None
    target_h: float | None
    raw_max_h: float
    e50: float
    actual_updates: int = 50000
    n_evaluated: int = 50
    official_complete: bool = False
    same_checkpoint_verified: bool = False

    def valid_for(self, sensor):
        spec = sensor_spec(sensor)
        if (self.actual_updates != 50000 or self.n_evaluated != 50 or
                not self.official_complete or not self.same_checkpoint_verified or
                not _range(self.raw_max_h, 0., 1.) or not _range(self.e50, 0., lower_strict=True)):
            return False
        if self.target_eligible:
            return (_range(self.target_h, spec.hqnr_threshold, 1., lower_strict=True) and
                    _range(self.target_e, 0.))
        return self.target_e is None and self.target_h is None and self.raw_max_h <= spec.hqnr_threshold


def _screen_summary(results):
    eligible = [result for result in results if result.target_eligible]
    return dict(eligible_count=len(eligible), eligible_seeds=[result.seed for result in eligible],
                target_e_median=median(result.target_e for result in eligible) if eligible else None,
                raw_max_h_median=median(result.raw_max_h for result in results),
                e50_median=median(result.e50 for result in results))


def promote_screen(server, profile, base_results, candidate_results):
    """Rank two completed paired seeds; retain BASE on tie/invalid/incomplete."""
    if server not in ("s2", "s5") or profile not in STUDENT_PROFILES:
        raise ValueError("Invalid screen assignment")
    sensor = "QB" if server == "s2" else "GF2"
    expected = (82006, 82007) if server == "s2" else (92006, 92007)
    result = dict(campaign_id=CAMPAIGN_ID, server_id=server, screen_profile=profile,
                  confirm_profile="BASE", promoted=False, decision="SCREEN_PROMOTION",
                  reason="INVALID_OR_INCOMPLETE_SCREEN", different_eligible_seeds=False,
                  base=None, candidate=None, test_aware=True)
    if profile == "BASE":
        result["reason"] = "NO_ALTERNATIVE_SELECTED"
        return result
    try:
        base = tuple(row if isinstance(row, SeedResult) else SeedResult(**row) for row in base_results)
        candidate = tuple(row if isinstance(row, SeedResult) else SeedResult(**row) for row in candidate_results)
    except (TypeError, ValueError):
        return result
    if (tuple(sorted(row.seed for row in base)) != expected or
            tuple(sorted(row.seed for row in candidate)) != expected or
            not all(row.valid_for(sensor) for row in (*base, *candidate))):
        return result
    b, c = _screen_summary(base), _screen_summary(candidate)
    result.update(base=b, candidate=c,
                  different_eligible_seeds=set(b["eligible_seeds"]) != set(c["eligible_seeds"]))
    if b["eligible_count"] == c["eligible_count"] == 0:
        promoted = (c["raw_max_h_median"] - b["raw_max_h_median"] >= .001 and
                    c["e50_median"] / b["e50_median"] - 1 <= .005)
        reason = "NO_ELIGIBLE_RAW_H_AND_E50_RULE" if promoted else "NO_ELIGIBLE_KEEP_BASE"
    else:
        # First compare eligible counts; None is never filled with a numeric sentinel.
        if c["eligible_count"] != b["eligible_count"]:
            promoted = c["eligible_count"] > b["eligible_count"]
        else:
            promoted = (c["target_e_median"], -c["raw_max_h_median"], c["e50_median"]) < (
                b["target_e_median"], -b["raw_max_h_median"], b["e50_median"])
        reason = "STRICT_SCREEN_RANK_WIN" if promoted else "TIE_OR_BASE_RANK_WIN"
    result.update(promoted=promoted, confirm_profile=profile if promoted else "BASE", reason=reason)
    return result


@dataclass(frozen=True)
class RuntimeObservation:
    server_id: str
    sensor: str
    role: str
    width: int
    depth: tuple[int, ...]
    component: str  # TRAIN, EVAL or CALIBRATION: never pool these timings.
    hours: float
    completed: bool
    evaluation_valid: bool
    # Controller persists capacity evidence with timing observations. Preserve
    # these optional fields; old timing-only receipts remain readable.
    profile: str | None = None
    peak_training_memory_bytes: int | None = None
    memory_scope: str | None = None
    gpu_uuid: str | None = None


def _p90(values):
    values = sorted(values)
    position = .9 * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def estimate_case_hours(case, observations=()):
    values = {"TRAIN": [], "EVAL": []}
    for observation in observations:
        if isinstance(observation, dict):
            observation = RuntimeObservation(**observation)
        if (observation.completed is True and observation.evaluation_valid is True and
                (observation.server_id, observation.sensor, observation.role, observation.width,
                 tuple(observation.depth)) ==
                (case.server_id, case.sensor, case.role, case.width, case.depth) and
                observation.component in values and _range(observation.hours, 0., lower_strict=True)):
            values[observation.component].append(observation.hours)
    # Initial contract combines 50K + official evaluation. Do not invent a split
    # or drop evaluation when only TRAIN observations exist.
    if not all(values.values()):
        return case.reservation_hours
    return 1.25 * sum(_p90(component) for component in values.values())


def estimate_calibration_hours(teacher, observations=()):
    values = []
    for observation in observations:
        if isinstance(observation, dict):
            observation = RuntimeObservation(**observation)
        if (observation.completed is True and observation.evaluation_valid is True and
                (observation.server_id, observation.sensor, observation.role, observation.width,
                 tuple(observation.depth), observation.component) ==
                (teacher.server_id, teacher.sensor, "T", teacher.width, teacher.depth, "CALIBRATION") and
                _range(observation.hours, 0., lower_strict=True)):
            values.append(observation.hours)
    return 1.25 * _p90(values) if values else CALIBRATION_HOURS[teacher.server_id]


def valid_reuse(case, receipt):
    """A caller's verified, content-backed equivalence receipt, never ID alone."""
    return (receipt.get("run_id") == case.run_id and receipt.get("actual_updates") == 50000 and
            receipt.get("official_complete") is True and
            all(receipt.get(key) is True for key in
                ("checksum_verified", "reference_equivalent", "source_equivalent",
                 "initialization_equivalent", "evaluation_equivalent")))


def admission(window, now, block_cases, *, p0_ready, evaluation_debt_hours,
              observations=(), reuse_receipts=None, c3_evidence_valid=False):
    """Strict > reservation gate for an entire finite block (C3 includes CAL).

    Prerequisite readiness must describe this local dependency path only.
    Existing admitted-block continuation is handled by the controller, not by
    weakening the new-admission cutoff here.
    """
    if not isinstance(window, CampaignWindow):
        window = CampaignWindow.from_dict(window)
    now = _utc(now)
    if now < window.t0_utc:
        raise ValueError("Cannot admit work before the shared campaign start")
    if not _range(evaluation_debt_hours, 0.):
        raise ValueError("Evaluation debt must be finite and nonnegative")
    cases = tuple(case_for(case) if isinstance(case, str) else case for case in block_cases)
    if not cases or any(case_for(case.run_id) != case for case in cases):
        raise ValueError("Atomic block must contain registered cases")
    if len({case.run_id for case in cases}) != len(cases):
        raise ValueError("Duplicate run in block")
    if len({(case.server_id, case.block_id) for case in cases}) != 1:
        raise ValueError("An atomic block cannot mix servers or block IDs")
    profiles = {case.profile for case in cases if case.role == "S"} - {"BASE"}
    if len(profiles) > 1:
        raise ValueError("At most one Student alternative may be admitted")
    expected_ids = {
        case.run_id for case in cases_for(cases[0].server_id)
        if case.block_id == cases[0].block_id and
        (not case.tier.endswith("_ALT") or case.profile in profiles)
    }
    if {case.run_id for case in cases} != expected_ids:
        raise ValueError("Admission must reserve the entire selected atomic block")
    is_c3 = cases[0].tier == "P2_TEACHER_CONDITIONAL"
    if is_c3:
        if len(cases) != 3 or sum(case.role == "T" for case in cases) != 1:
            raise ValueError("C3 admission requires Teacher and both paired Students")
        teacher = next(case for case in cases if case.role == "T")
        expected = (82001, 82002) if teacher.server_id == "s1" else (92001, 92002)
        if (teacher.profile != "C3" or tuple(sorted(case.seed for case in cases if case.role == "S")) != expected or
                any(case.reference_id != teacher.reference_id for case in cases)):
            raise ValueError("Invalid C3 package")
    elif profiles and (len(cases) != 2 or sum(case.profile == "BASE" for case in cases) != 1 or
                       len({case.seed for case in cases}) != 1):
        raise ValueError("A Student trial must reserve BASE plus alternative at the same seed")
    reuse_receipts = reuse_receipts or {}
    reused = []
    for case in cases:
        if case.run_id in reuse_receipts:
            if not valid_reuse(case, reuse_receipts[case.run_id]):
                raise ValueError("Existing run is not verified equivalent; cannot reuse")
            reused.append(case.run_id)
    train_eval = sum(estimate_case_hours(case, observations) for case in cases if case.run_id not in reused)
    # BASE Teachers also calibrate immediately before releasing their reference;
    # their published initial Teacher timing includes evaluation, not CAL.
    calibration = sum(estimate_calibration_hours(case, observations)
                      for case in cases if case.role == "T")
    block_hours = train_eval + calibration
    required = block_hours + evaluation_debt_hours + CLOSE_HOURS
    remaining = window.remaining_hours(now)
    if p0_ready is not True:
        reason = "BLOCKED_INTEGRITY"
    elif is_c3 and c3_evidence_valid is not True:
        reason = "NOT_ADMITTED_CONDITION"
    elif now >= window.deadline_utc:
        reason = "INCOMPLETE_AT_DEADLINE"
    elif now >= window.admission_cutoff_utc or (is_c3 and now >= window.c3_cutoff_utc):
        reason = "NOT_ADMITTED_CUTOFF"
    elif remaining <= required:
        reason = "NOT_ADMITTED_BUDGET"
    else:
        reason = "ADMITTED"
    return dict(campaign_id=CAMPAIGN_ID, server_id=cases[0].server_id, block_id=cases[0].block_id,
                allowed=reason == "ADMITTED", reason=reason, at_utc=now.isoformat(),
                run_ids=[case.run_id for case in cases], reused_run_ids=reused,
                block_hours=block_hours, train_eval_hours=train_eval, calibration_hours=calibration,
                evaluation_debt_hours=evaluation_debt_hours, close_hours=CLOSE_HOURS,
                required_hours=required, remaining_hours=remaining,
                deadline_utc=window.deadline_utc.isoformat(), c3_package=is_c3)
