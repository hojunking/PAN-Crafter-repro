"""Frozen, finite MIX20H case registry. Importing this module changes no queues.

This is an explicit experiment design, not a seed/profile resolver. Historical
recipe locks, performance results, and old priority lists are deliberately absent.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

CAMPAIGN_ID = "QRC24_MIX20H_20260918_v1"
QUEUE_REVISION = "QRC24_MIX20H_Q1_20260918"
SELECTOR_ID = "HQNR9585_ERGAS2040_v2"
EVAL_MODE = "A_ON"
METHOD = "qrecon_continuous_v1"
VERSION = "v4"
SOURCE_PLAN = "research_log/PAN_AllServers_20H_MixedSeed_HQNR9585_ERGAS_Plan_2026-09-18.md"
STATE_DIR = "work_dir/_qrc24_mix20h"
PLAN_MANIFEST = STATE_DIR + "/plan_manifest.json"
SERVERS = ("s1", "s2", "s3", "s4", "s5")
PROFILES = ("G23", "B20A03")
TEACHER_SHA256 = "16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32"
TAU_R = 0.012463942170143127
Q_REF = 0.3276133416220546
REFERENCE_HOURS = {"s1": 2.45, "s2": 2.60, "s3": 1.50, "s4": 1.55, "s5": 3.40}
BASE_COUNTS = {"s1": 6, "s2": 6, "s3": 10, "s4": 10, "s5": 4}
CLOSE_RESERVE_HOURS = 0.5
PREP_RESERVE_HOURS = 0.5
WINDOW_HOURS = 20.0
GRID_STEPS = tuple(range(1010, 50000, 1010)) + (50000,)

# Each row is one pair: (seed, first profile, second profile, tier).
# Spelled out instead of deriving seeds from server number or reading a lock.
_PAIRS = {
    "s1": ((52001, "G23", "B20A03", "base"),
           (52006, "B20A03", "G23", "base"),
           (52011, "G23", "B20A03", "base"),
           (52016, "B20A03", "G23", "reserve")),
    "s2": ((52002, "B20A03", "G23", "base"),
           (52007, "G23", "B20A03", "base"),
           (52012, "B20A03", "G23", "base"),
           (52017, "G23", "B20A03", "reserve")),
    "s3": ((52003, "G23", "B20A03", "base"),
           (52008, "B20A03", "G23", "base"),
           (52013, "G23", "B20A03", "base"),
           (52018, "B20A03", "G23", "base"),
           (52023, "G23", "B20A03", "base"),
           (52028, "B20A03", "G23", "reserve")),
    "s4": ((52004, "B20A03", "G23", "base"),
           (52009, "G23", "B20A03", "base"),
           (52014, "B20A03", "G23", "base"),
           (52019, "G23", "B20A03", "base"),
           (52024, "B20A03", "G23", "base"),
           (52029, "G23", "B20A03", "reserve")),
    "s5": ((52005, "G23", "B20A03", "base"),
           (52010, "B20A03", "G23", "base"),
           (52015, "G23", "B20A03", "reserve")),
}


def run_name(server, profile, seed):
    return f"PAKD50_QRC24_{server.upper()}_{profile}_W104_D121_WV3_T0_S{seed}_FRESH50_{VERSION}"


@dataclass(frozen=True)
class Case:
    server_id: str
    order: int
    profile: str
    seed: int
    tier: str
    pair_position: int
    version: str = VERSION

    @property
    def run_id(self):
        return run_name(self.server_id, self.profile, self.seed)

    @property
    def pair_id(self):
        return f"M20_{self.server_id.upper()}_S{self.seed}"

    @property
    def pairmate_run_id(self):
        return run_name(self.server_id, "B20A03" if self.profile == "G23" else "G23", self.seed)

    @property
    def beta(self):
        return 0.1 if self.profile == "G23" else 0.2

    def to_dict(self):
        return dict(asdict(self), run_id=self.run_id, original_run_id=self.run_id,
                    pair_id=self.pair_id, pairmate_run_id=self.pairmate_run_id,
                    beta=self.beta, reservation_hours=reservation_hours(self.server_id))


CASES = tuple(Case(srv, 2 * i + pos, profile, seed, tier, pos)
              for srv in SERVERS for i, (seed, first, second, tier) in enumerate(_PAIRS[srv])
              for pos, profile in enumerate((first, second), 1))
_BY_ID = {case.run_id: case for case in CASES}
assert len(CASES) == len(_BY_ID) == 46
assert sum(c.tier == "base" for c in CASES) == 36


def reservation_hours(server):
    """Initial end-to-end estimate: 1.10 R + 0.10 hours, not measured time."""
    return round(1.10 * REFERENCE_HOURS[server] + 0.10, 8)


def cases_for(server):
    if server not in SERVERS:
        raise ValueError(f"unknown MIX20H server: {server!r}")
    return tuple(c for c in CASES if c.server_id == server)


def case_for(run_id):
    try:
        return _BY_ID[run_id]
    except KeyError:
        raise ValueError(f"unregistered MIX20H run: {run_id!r}") from None


def pair_for(run_id):
    case = case_for(run_id)
    return tuple(c for c in cases_for(case.server_id) if c.pair_id == case.pair_id)


def metadata_for(run_id):
    case = case_for(run_id)
    return dict(campaign_id=CAMPAIGN_ID, queue_revision=QUEUE_REVISION,
                **case.to_dict(), selector=SELECTOR_ID, eval_mode=EVAL_MODE,
                method=METHOD, plan_manifest=PLAN_MANIFEST, source_plan=SOURCE_PLAN,
                requires_recipe_lock=False, requires_other_server_results=False)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def registry_manifest(server=None):
    """Static design only; no invented start/deadline/readiness or runtime hashes."""
    cases = CASES if server is None else cases_for(server)
    rows = [metadata_for(c.run_id) for c in cases]
    return dict(campaign_id=CAMPAIGN_ID, queue_revision=QUEUE_REVISION,
                selector=SELECTOR_ID, eval_mode=EVAL_MODE, server_id=server,
                source_plan=SOURCE_PLAN, cases=rows, case_list_sha256=canonical_sha256(rows),
                base_count=sum(c.tier == "base" for c in cases),
                reserve_count=sum(c.tier == "reserve" for c in cases))


def validate_metadata(meta, run_id=None):
    """Reject changed/partial identity; unknown seeds never get a fallback profile."""
    run_id = run_id or meta.get("original_run_id")
    expected = metadata_for(run_id)
    keys = ("campaign_id", "queue_revision", "original_run_id", "server_id", "pair_id",
            "pairmate_run_id", "profile", "seed", "version", "selector", "eval_mode",
            "method", "plan_manifest", "requires_recipe_lock", "requires_other_server_results")
    for key in keys:
        if meta.get(key) != expected[key]:
            raise ValueError(f"MIX20H identity mismatch: {key}: {meta.get(key)!r} != {expected[key]!r}")
    return case_for(run_id)
