"""Pure G20 paired decisions and finite atomic reservations; no runtime writes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import math
from statistics import median

from .plan import (ADMISSION_CUTOFF_HOURS, AXES, CALIBRATION_HOURS, CAMPAIGN_ID,
                   CONFIRM_SEEDS, CORE_CASES, SCREEN_SEEDS, TRAIN_FINISH_HOURS,
                   WINDOW_HOURS, Case, _object_sha, blocks_for, case_for,
                   valid_sha, validate_case)


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware timestamp is required")
    return value.astimezone(timezone.utc)


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


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
    def train_finish_utc(self):
        return self.t0_utc + timedelta(hours=TRAIN_FINISH_HOURS)

    @property
    def training_deadline_utc(self):
        return self.train_finish_utc

    def remaining_hours(self, now):
        return max(0., (self.deadline_utc - _utc(now)).total_seconds() / 3600.)

    def to_dict(self):
        return dict(campaign_id=CAMPAIGN_ID, t0_utc=self.t0_utc.isoformat(),
                    deadline_utc=self.deadline_utc.isoformat(),
                    admission_cutoff_utc=self.admission_cutoff_utc.isoformat(),
                    train_finish_utc=self.train_finish_utc.isoformat(), close_hours=2.)

    @classmethod
    def from_dict(cls, value):
        result = cls(value["t0_utc"])
        for key, expected in result.to_dict().items():
            if key in value and value[key] != expected:
                raise ValueError(f"Changed G20 shared clock: {key}")
        return result


@dataclass(frozen=True)
class RuntimeObservation:
    server_id: str
    sensor: str
    role: str
    width: int
    depth: tuple[int, ...]
    component: str
    hours: float
    completed: bool
    evaluation_valid: bool
    profile: str | None = None
    peak_training_memory_bytes: int | None = None
    memory_scope: str | None = None
    gpu_uuid: str | None = None


def _p90(values):
    values = sorted(values)
    p = .9 * (len(values) - 1)
    lower, upper = math.floor(p), math.ceil(p)
    return values[lower] + (values[upper] - values[lower]) * (p - lower)


def _timings(case, observations, components):
    result = {component: [] for component in components}
    for row in observations:
        row = RuntimeObservation(**row) if isinstance(row, dict) else row
        if (row.completed is True and row.evaluation_valid is True and
                (row.server_id, row.sensor, row.role, row.width, tuple(row.depth)) ==
                (case.server_id, case.sensor, case.role, case.width, case.depth) and
                row.component in result and _finite(row.hours) and row.hours >= 0):
            result[row.component].append(row.hours)
    return result


def estimate_case_hours(case, observations=()):
    """Never lower the initial reservation; keep unknown postprocessing reserved."""
    values = _timings(case, observations, ("TRAIN", "EVAL", "TRAIN_EVAL"))
    known = [1.25 * _p90(values["TRAIN_EVAL"])] if values["TRAIN_EVAL"] else []
    if values["TRAIN"] and values["EVAL"]:
        known.append(1.25 * (_p90(values["TRAIN"]) + _p90(values["EVAL"])))
    elif values["TRAIN"] or values["EVAL"]:
        # The plan defines only a combined initial reservation. Until both
        # components are measured retain it for the unknown component rather
        # than invent a cheaper evaluation allocation or ignore slow training.
        known.append(case.reservation_hours + 1.25 * sum(_p90(v) for v in values.values() if v))
    return max([case.reservation_hours, *known])


def estimate_calibration_hours(teacher, observations=()):
    values = _timings(teacher, observations, ("CALIBRATION",))["CALIBRATION"]
    return max(CALIBRATION_HOURS[teacher.server_id], 1.25 * _p90(values)) if values else CALIBRATION_HOURS[teacher.server_id]


def valid_reuse(case, receipt):
    return (receipt.get("run_id") == case.run_id and receipt.get("actual_updates") == 50000 and
            all(receipt.get(key) is True for key in ("official_complete", "checksum_verified",
                "reference_equivalent", "source_equivalent", "initialization_equivalent", "evaluation_equivalent")) and
            (case.role != "T" or receipt.get("calibration_validated") is True))


def admission(window, now, block_cases, *, p0_ready, evaluation_debt_hours,
              observations=(), reuse_receipts=None, completed_run_ids=()):
    window = window if isinstance(window, CampaignWindow) else CampaignWindow.from_dict(window)
    now = _utc(now)
    if now < window.t0_utc:
        raise ValueError("Cannot admit before the shared start")
    if not _finite(evaluation_debt_hours) or evaluation_debt_hours < 0:
        raise ValueError("Evaluation debt must be finite and nonnegative")
    cases = tuple(validate_case(case_for(c) if isinstance(c, str) else c) for c in block_cases)
    if not cases or len({c.run_id for c in cases}) != len(cases) or len({(c.server_id, c.block_id) for c in cases}) != 1:
        raise ValueError("Reserve a single nonempty registered atomic block")
    server, block_id = cases[0].server_id, cases[0].block_id
    kw = ({"confirmation_cases": cases} if cases[0].tier == "CONDITIONAL_CONFIRM" else
          {"transfer_cases": cases} if cases[0].tier == "CONDITIONAL_TRANSFER" else {})
    block = next((b for b in blocks_for(server, **kw) if b.block_id == block_id), None)
    if block is None or cases != block.cases:
        raise ValueError("Atomic admission requires the complete canonical block and order")
    receipts = reuse_receipts or {}
    reused = []
    for case in cases:
        if case.run_id in receipts:
            if not valid_reuse(case, receipts[case.run_id]):
                raise ValueError("Existing assets have not been verified equivalent")
            reused.append(case.run_id)
    train_eval = sum(estimate_case_hours(c, observations) for c in cases if c.run_id not in reused)
    calibration = sum(estimate_calibration_hours(c, observations) for c in cases if c.role == "T" and c.run_id not in reused)
    required = train_eval + calibration + evaluation_debt_hours
    dependency_ready = True
    if block_id == "S2_C030_REFERENCE":
        # R3 never consumes time before the priority R2/R4 comparison exists.
        dependency_ready = set(blocks_for("s2")[0].run_ids).issubset(set(completed_run_ids))
    if p0_ready is not True:
        reason = "BLOCKED_INTEGRITY"
    elif not dependency_ready:
        reason = "NOT_ADMITTED_PRIORITY_COMPARISON"
    elif now >= window.deadline_utc:
        reason = "PARTIAL_TIME_LIMIT"
    elif now >= window.admission_cutoff_utc:
        reason = "NOT_ADMITTED_CUTOFF"
    elif now + timedelta(hours=required) > window.train_finish_utc:
        reason = "NOT_ADMITTED_BUDGET"
    else:
        reason = "ADMITTED"
    return dict(campaign_id=CAMPAIGN_ID, server_id=server, block_id=block_id, allowed=reason == "ADMITTED",
                reason=reason, at_utc=now.isoformat(), run_ids=[c.run_id for c in cases],
                reused_run_ids=reused, train_eval_hours=train_eval, calibration_hours=calibration,
                block_hours=train_eval + calibration, evaluation_debt_hours=evaluation_debt_hours,
                required_hours=required, remaining_hours=window.remaining_hours(now),
                train_finish_utc=window.train_finish_utc.isoformat(), deadline_utc=window.deadline_utc.isoformat())


@dataclass(frozen=True)
class RunResult:
    seed: int
    server_id: str
    reference_id: str
    profile: str
    exact50k: dict
    rr_val_selected: dict | None
    p0_passed: bool = False
    paired_identity_verified: bool = False
    official_complete: bool = False
    same_checkpoint_verified: bool = False
    a_on: bool = False
    actual_updates: int = 50000
    n_evaluated: int = 50
    evidence_files: dict = field(default_factory=dict)
    u_init_sha256: str | None = None
    data_view_sha256: str | None = None
    source_sha256: str | None = None
    a_init_sha256: str | None = None
    reference_sha256: str | None = None
    development_target: dict | None = None

    def valid(self):
        return (self.actual_updates == 50000 and self.n_evaluated == 50 and
                all(getattr(self, name) is True for name in ("p0_passed", "paired_identity_verified", "official_complete", "same_checkpoint_verified", "a_on")) and
                _metrics_valid(self.exact50k) and _metrics_valid(self.rr_val_selected) and
                (self.development_target is None or _metrics_valid(self.development_target)) and
                bool(self.evidence_files) and isinstance(self.evidence_files, dict) and
                all(isinstance(path, str) and path and valid_sha(sha) for path, sha in self.evidence_files.items()) and
                all(valid_sha(getattr(self, key)) for key in ("u_init_sha256", "data_view_sha256", "source_sha256", "a_init_sha256", "reference_sha256")))


def _metrics_valid(value):
    return (isinstance(value, dict) and all(_finite(value.get(k)) for k in ("HQNR", "ERGAS", "D_lambda")) and
            0 <= value["HQNR"] <= 1 and value["ERGAS"] > 0 and 0 <= value["D_lambda"] <= 1)


def joint(metrics):
    return bool(_metrics_valid(metrics) and metrics["HQNR"] > .964 and metrics["ERGAS"] < .552)


def joint_flags(metrics):
    return dict(joint=joint(metrics), strong=bool(joint(metrics) and metrics["ERGAS"] < .522))


def _expected_assignments(server, candidate):
    if server not in AXES or candidate not in AXES[server]:
        raise ValueError("Unregistered local candidate")
    return (("R0", "BASE"), ("R1", "BASE")) if server == "s1" else (
        (("R2", "BASE"), (candidate, "BASE")) if server == "s2" else
        (("R0", "BASE"), ("R0", candidate)))


def _validated_pairs(server, candidate, pairs, seeds):
    expected_base, expected_alt = _expected_assignments(server, candidate)
    result = []
    for pair in pairs:
        base = pair["base"] if isinstance(pair["base"], RunResult) else RunResult(**pair["base"])
        alt = pair["candidate"] if isinstance(pair["candidate"], RunResult) else RunResult(**pair["candidate"])
        if (not base.valid() or not alt.valid() or base.seed != pair["seed"] or alt.seed != pair["seed"] or
                base.server_id != server or alt.server_id != server or
                (base.reference_id, base.profile) != expected_base or
                (alt.reference_id, alt.profile) != expected_alt or
                any(getattr(base, k) != getattr(alt, k) for k in ("u_init_sha256", "data_view_sha256", "source_sha256")) or
                (base.reference_id == alt.reference_id and
                 (base.a_init_sha256 != alt.a_init_sha256 or base.reference_sha256 != alt.reference_sha256))):
            raise ValueError("Invalid, incomplete or unmatched local pair")
        result.append((base, alt))
    if sorted(b.seed for b, _ in result) != sorted(seeds):
        raise ValueError("All and only the prespecified paired seeds are required")
    for side in (0, 1):
        if len({pair[side].reference_sha256 for pair in result}) != 1:
            raise ValueError("A multi-seed condition must use one fixed reference bundle")
    return sorted(result, key=lambda pair: pair[0].seed)


def _summary(pairs, selection):
    base = [getattr(b, selection) for b, _ in pairs]
    alt = [getattr(a, selection) for _, a in pairs]
    delta_h = [a["HQNR"] - b["HQNR"] for b, a in zip(base, alt)]
    relative_e = [a["ERGAS"] / b["ERGAS"] - 1 for b, a in zip(base, alt)]
    delta_d = [a["D_lambda"] - b["D_lambda"] for b, a in zip(base, alt)]
    return dict(delta_h=delta_h, relative_e=relative_e, delta_dlambda=delta_d,
                delta_h_median=median(delta_h), relative_e_median=median(relative_e),
                delta_dlambda_median=median(delta_d), candidate_e_median=median(a["ERGAS"] for a in alt),
                joint_seed_count=sum(joint(a) for a in alt),
                candidate_joint_seeds=[a.seed for _, a in pairs if joint(getattr(a, selection))])


def _guards(exact, val):
    return dict(exact_e=exact["relative_e_median"] <= .005 and max(exact["relative_e"]) <= .01,
                exact_dlambda=exact["delta_dlambda_median"] <= .001 and max(exact["delta_dlambda"]) <= .002,
                val_e=val["relative_e_median"] <= .005 and max(val["relative_e"]) <= .01,
                val_h=val["delta_h_median"] >= -.0015)


def paired_screen(server, candidate, pairs):
    _expected_assignments(server, candidate)
    result = dict(campaign_id=CAMPAIGN_ID, server_id=server, candidate=candidate,
                  decision="PAIRED_SCREEN", status="TECHNICAL_INVALID", eligible=False,
                  evidence_files={}, thresholds=dict(delta_h_each=1e-5, delta_h_median=.0015,
                     relative_e_median=.005, relative_e_each=.01, delta_dlambda_median=.001,
                     delta_dlambda_each=.002, validation_delta_h_median=-.0015),
                  test_aware=True, statistical_significance_claim=False)
    try:
        pairs = _validated_pairs(server, candidate, pairs, SCREEN_SEEDS)
    except (KeyError, TypeError, ValueError):
        return result
    exact, val = _summary(pairs, "exact50k"), _summary(pairs, "rr_val_selected")
    criteria = _guards(exact, val)
    criteria["exact_h"] = min(exact["delta_h"]) > 1e-5 and exact["delta_h_median"] >= .0015
    target_joint_seeds = [alt.seed for _, alt in pairs if joint(alt.development_target)]
    status = ("PROMISING_PAIRED" if all(criteria.values()) else "JOINT_SINGLE_RETEST" if
              exact["joint_seed_count"] or val["joint_seed_count"] or target_joint_seeds else
              "HQNR_ONLY_TRADEOFF" if exact["delta_h_median"] > 0 and not all(_guards(exact, val).values()) else "NO_GAIN")
    evidence = {}
    for base, alt in pairs:
        for row in (base, alt):
            for path, sha in row.evidence_files.items():
                if path in evidence and evidence[path] != sha:
                    return result
                evidence[path] = sha
    result.update(status=status, eligible=status in ("PROMISING_PAIRED", "JOINT_SINGLE_RETEST"),
                  exact50k=exact, rr_val_selected=val, criteria=criteria, evidence_files=evidence,
                  development_target_joint_seeds=target_joint_seeds,
                  development_target_use="RETEST_ONLY; never PROMISING/rank/confirmation",
                  pairs=[dict(seed=base.seed, base=asdict(base), candidate=asdict(alt)) for base, alt in pairs])
    return result


def choose_screen(server, candidates, selected_at_utc, previous=None, not_admitted=None):
    if server not in AXES:
        raise ValueError("Unknown server")
    at = _utc(selected_at_utc).isoformat()
    if previous is not None:
        if (previous.get("campaign_id") != CAMPAIGN_ID or previous.get("server_id") != server or
                previous.get("decision") != "SCREEN_SELECTION" or
                previous.get("selected_candidate") not in (*AXES[server], None)):
            raise ValueError("Invalid locked decision")
        return dict(previous)
    validated = []
    seen = set()
    for receipt in candidates:
        candidate = receipt.get("candidate")
        if candidate in seen or candidate not in AXES[server] or receipt.get("server_id") != server:
            raise ValueError("Duplicate or foreign screen candidate")
        seen.add(candidate)
        recomputed = paired_screen(server, candidate, receipt.get("pairs", ()))
        if recomputed != receipt:
            raise ValueError("Screen receipt does not match underlying paired evidence")
        validated.append(receipt)
    not_admitted = not_admitted or {}
    if (any(server != "s2" or candidate != "R3" or not valid_sha(sha)
            for candidate, sha in not_admitted.items()) or seen.intersection(not_admitted) or
            seen.union(not_admitted) != set(AXES[server])):
        raise ValueError("Finish both screen directions or provide the predeclared s2/R3 budget skip")
    eligible = [row for row in validated if row["eligible"]]
    def rank(row):
        return (row["status"] == "PROMISING_PAIRED", row["rr_val_selected"]["joint_seed_count"],
                row["exact50k"]["joint_seed_count"], row["exact50k"]["delta_h_median"],
                -row["exact50k"]["candidate_e_median"])
    # Full tie preserves the predeclared candidate order, not arrival order.
    eligible.sort(key=lambda row: AXES[server].index(row["candidate"]))
    winner = max(eligible, key=rank) if eligible else None
    evidence = {}
    for row in validated:
        for path, sha in row["evidence_files"].items():
            if path in evidence and evidence[path] != sha:
                raise ValueError("Conflicting evidence file hashes")
            evidence[path] = sha
    return dict(campaign_id=CAMPAIGN_ID, server_id=server, decision="SCREEN_SELECTION",
                selected_candidate=winner["candidate"] if winner else None,
                selected_status=winner["status"] if winner else "NO_ELIGIBLE_CANDIDATE",
                selected_at_utc=at, evidence_files=evidence, candidates=validated,
                not_admitted=dict(not_admitted), test_aware=True, locked=True)


def confirm_gain(server, candidate, pairs):
    result = dict(campaign_id=CAMPAIGN_ID, server_id=server, candidate=candidate,
                  decision="FOUR_SEED_CONFIRMATION", status="TECHNICAL_INVALID",
                  reproducible_gain=False, joint_reproduced=False, statistical_significance_claim=False)
    try:
        pairs = _validated_pairs(server, candidate, pairs, (*SCREEN_SEEDS, *CONFIRM_SEEDS))
    except (KeyError, TypeError, ValueError):
        return result
    exact, val = _summary(pairs, "exact50k"), _summary(pairs, "rr_val_selected")
    new = [(b, a) for b, a in pairs if b.seed in CONFIRM_SEEDS]
    new_exact, new_val = _summary(new, "exact50k"), _summary(new, "rr_val_selected")
    criteria = _guards(exact, val)
    criteria.update(three_positive=sum(h > 1e-5 for h in exact["delta_h"]) >= 3,
                    median_h=exact["delta_h_median"] >= .0015,
                    new_positive=any(h > 0 for h in new_exact["delta_h"]))
    gain = all(criteria.values())
    reproduced = new_val["joint_seed_count"] == 2
    result.update(reproducible_gain=gain, joint_reproduced=reproduced, criteria=criteria,
                  exact50k=exact, rr_val_selected=val, new_seed_validation=new_val,
                  status="JOINT_REPRODUCED" if reproduced else "REPRODUCIBLE_GAIN" if gain else "NO_GAIN")
    return result


def lock_transfer(reference_screen, student_screen, selected_at_utc, *, reference_id,
                  reference_sha256, local_controls_verified, pair_identity_verified,
                  evidence_files, previous=None):
    """A locked 2x2 requires an actual promising reference and scalar screen."""
    if previous is not None:
        if previous.get("campaign_id") != CAMPAIGN_ID or previous.get("decision") != "TRANSFER_LOCK":
            raise ValueError("Invalid transfer lock")
        return dict(previous)
    teacher_server, owner = reference_screen.get("server_id"), student_screen.get("server_id")
    if teacher_server not in ("s1", "s2") or owner not in ("s3", "s4", "s5"):
        raise ValueError("Transfer owner must be the original scalar-screen server")
    for screen in (reference_screen, student_screen):
        actual = paired_screen(screen["server_id"], screen.get("candidate"), screen.get("pairs", ()))
        if actual != screen or screen.get("status") != "PROMISING_PAIRED":
            raise ValueError("Both reference and single-scalar screen must be technically valid and promising")
    # Do not silently substitute the baseline reference for the winning one.
    if reference_id != reference_screen["candidate"] or not valid_sha(reference_sha256):
        raise ValueError("Transfer must name the measured promising reference and its SHA")
    if local_controls_verified is not True or pair_identity_verified is not True or not evidence_files:
        raise ValueError("Original-server R0/BASE/XSTAR controls and identities required")
    if not all(valid_sha(value) for value in evidence_files.values()):
        raise ValueError("Transfer evidence hashes required")
    return dict(campaign_id=CAMPAIGN_ID, server_id=owner, decision="TRANSFER_LOCK",
                reference_id=reference_id, reference_sha256=reference_sha256,
                profile=student_screen["candidate"], selected_at_utc=_utc(selected_at_utc).isoformat(),
                reference_promising=True, student_screen_passed=True,
                local_controls_verified=True, pair_identity_verified=True,
                reference_screen_sha256=_object_sha(reference_screen),
                student_screen_sha256=_object_sha(student_screen), evidence_files=dict(evidence_files),
                exploratory=True, test_aware=True)
