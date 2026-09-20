"""Pinned, finite all-GF2 G20 definitions. Importing never admits or starts work."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from qg40.plan import SensorSpec, SplitBinding, sensor_spec as _sensor_spec

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "PANDA_GF2_G20_ALL5_20260920_v1"
REGISTRY_REVISION = "G20_REGISTRY_20260920_v1"
METHOD_REVISION = "QG40_SYNC_FREQ_C4_v1"
SERVERS = ("s1", "s2", "s3", "s4", "s5")
WINDOW_HOURS, ADMISSION_CUTOFF_HOURS, TRAIN_FINISH_HOURS = 20., 16., 18.
CLOSE_HOURS = 2.
GRID_STEPS = tuple(range(1010, 50000, 1010)) + (50000,)
FULLSTATE_STEPS = (10000, 24240, 50000)
SCREEN_SEEDS, CONFIRM_SEEDS = (93001, 93002), (93011, 93012)
SOURCE_PLAN = "research_log/PANDA_GF2_G20_ALL5_FixedArchitecture_ExperimentPlan_2026-09-20.md"
SOURCE_CASES = "research_log/PANDA_GF2_G20_ALL5_Cases_2026-09-20.csv"
SOURCE_SHAS = MappingProxyType({
    SOURCE_PLAN: "9a4048dfb137979c887db2dc0d1dcd88bffdd788106e9acdb26f89f3d86dff8a",
    SOURCE_CASES: "343d6c3ae2ac54611ce7588e7e2dc1e6201dff6386d2e9bc3ac78afac90571b6",
})
SHEET_TABS = MappingProxyType(dict(s1="GF2-s1", s2="GF2-s2", s3="GF2-s3(5090)",
                                   s4="GF2-s4", s5="GF2-s5"))
TEACHER_HOURS = MappingProxyType({server: 1. for server in SERVERS})
STUDENT_HOURS = MappingProxyType({server: 1.5 for server in SERVERS})
CALIBRATION_HOURS = MappingProxyType({server: 1. for server in SERVERS})
SENSORS = MappingProxyType({"GF2": _sensor_spec("GF2")})


def sensor_spec(sensor="GF2"):
    if sensor != "GF2":
        raise ValueError("G20 is GF2-only; QB/WV3 fallback is forbidden")
    return SENSORS[sensor]


def _object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify_sources():
    for path, expected in SOURCE_SHAS.items():
        if _sha(ROOT / path) != expected:
            raise ValueError(f"G20 source changed: {path}; use a new registry revision")
    return dict(SOURCE_SHAS)


@dataclass(frozen=True)
class StudentProfile:
    alpha: float = 1.
    beta: float = .1
    lambda_edge: float = .002
    student_a_peak_lr: float = 3e-6


PROFILES = MappingProxyType({
    "BASE": StudentProfile(), "A1": StudentProfile(student_a_peak_lr=1e-6),
    "A9": StudentProfile(student_a_peak_lr=9e-6), "E1": StudentProfile(lambda_edge=.001),
    "E4": StudentProfile(lambda_edge=.004), "K05": StudentProfile(beta=.05),
    "K20": StudentProfile(beta=.2),
})
STUDENT_PROFILES = tuple(PROFILES)
AXES = MappingProxyType({"s1": ("R1",), "s2": ("R3", "R4"), "s3": ("A1", "A9"),
                         "s4": ("E1", "E4"), "s5": ("K05", "K20")})


@dataclass(frozen=True)
class Reference:
    reference_id: str
    alias: str
    server_id: str
    teacher_seed: int
    lambda_con: float
    teacher_case_id: str | None


REFERENCES = MappingProxyType({
    "R0": Reference("R0", "GF2_TA", "s3", 91001, 1e-4, None),
    "R1": Reference("R1", "GF2_G20_S1_TB_C100", "s1", 91002, 1e-4, "G20-T01"),
    "R2": Reference("R2", "GF2_G20_S2_TB_C100", "s2", 91002, 1e-4, "G20-T02"),
    "R3": Reference("R3", "GF2_G20_S2_TB_C030", "s2", 91002, 3e-5, "G20-T03"),
    "R4": Reference("R4", "GF2_G20_S2_TB_C300", "s2", 91002, 3e-4, "G20-T04"),
})
R0_PINS = MappingProxyType({
    "teacher_alias": "GF2_TA", "teacher_seed": 91001,
    "teacher_sha256": "0fb973376f9f00947d295620d96062b621145607aee311e28c889caefb656dab",
    "calibration_id": "45cc75ebdc29867bd904d69ecefd3f2a3de1a80daf7145e5a6227350754c5da1",
    "tau_R": .005695626139640808, "q_ref": .4086490869522095,
    "q_cache_sha256": "ddbb74a2a97661098a10db7bd9d17f5b7f7c5cb92b65ce39dd35751ed6b6d8ac",
    "method_revision": METHOD_REVISION,
    "source_bundle_sha256": "0cf7bf2af1a81036a115b3b9a181c4cc0d3f292e7a7927383f25c5819dee9d8c",
})
R0_DATA_PINS = MappingProxyType({
    "train": "243a0bc8a4cc0a2740ed24409f9e87afe531d87e9d57524c6bcbec2b5f937621",
    "train_lp": "af43c2aa96c26d92874e61052647e4cf3909e8fcb680a9c584d48cfe20e3fc3b",
    "val": "ae15b19ccc6ece799ebfeb3fb374b1a333238226397441e3d523b0195dff4f43",
    "val_lp": "aad58b4fd6435b695e1d00adc100c5a8d79450ea6e2d69ad33089e2fa7d9feae",
    "rr": "709a9a53b2e0f29d3dcd5c6ca4410c2913b62cc03c104fed6c049911dbe1c8ea",
    "rr_lp": "6535574fe8efb3dbfb72c1938ca2557fb868fd9f32af9598fb7ba367267d9489",
    "fr": "e52d151f262a74f96e59f03124f86a27d80841073ce60376856ae679020d00dc",
    "fr_lp": "f17335ea6c433c39081ad623a22a355a5f8b469e148c77d574e283c2f3165167",
})


@dataclass(frozen=True)
class Case:
    case_id: str
    run_id: str
    server_id: str
    role: str
    reference_id: str
    teacher_seed: int | None
    student_seed: int | None
    profile: str
    tier: str
    status: str
    lambda_con: float | None
    alpha: float | None
    beta: float | None
    lambda_edge: float | None
    student_a_peak_lr: float | None
    block_id: str
    queue_rank: int
    within_block_order: int
    updates: int = 50000
    sensor: str = "GF2"
    resolution_sha256: str | None = None
    reference_sha256: str | None = None

    @property
    def is_resolved(self):
        return self.status == "PLANNED" and self.server_id in SERVERS and self.reference_id in REFERENCES

    @property
    def seed(self):
        return self.teacher_seed if self.role == "T" else self.student_seed

    @property
    def input_layout(self):
        return "P0" if self.role == "T" else "PLH"

    @property
    def width(self):
        return 112 if self.role == "T" else 104

    @property
    def depth(self):
        return (1, 2, 3) if self.role == "T" else (1, 2, 2)

    @property
    def teacher_alias(self):
        return REFERENCES[self.reference_id].alias if self.reference_id in REFERENCES else None

    @property
    def teacher_input_layout(self):
        return "P0"

    @property
    def teacher_run_id(self):
        if self.reference_id not in REFERENCES:
            return None
        ref = REFERENCES[self.reference_id]
        return case_for(ref.teacher_case_id).run_id if ref.teacher_case_id else None

    @property
    def sheet_tab(self):
        return SHEET_TABS.get(self.server_id)

    @property
    def reservation_hours(self):
        return 1. if self.role == "T" else 1.5

    def to_dict(self):
        return dict(asdict(self), seed=self.seed, width=self.width, depth=list(self.depth),
                    input_layout=self.input_layout, teacher_input_layout="P0", teacher_alias=self.teacher_alias,
                    teacher_run_id=self.teacher_run_id, sheet_tab=self.sheet_tab, kind="TRAIN",
                    campaign_id=CAMPAIGN_ID, config_path=f"config/g20/{self.run_id}.yaml",
                    is_resolved=self.is_resolved)


# Whole local controls are included in the initial reference packages. s2 can
# drop R3 only after reserving a complete R2/R4 two-seed comparison.
_CORE_BLOCK_IDS = (
    ("s1", "S1_REFERENCE_PAIR", ("T01", "S01", "S02", "S03", "S04")),
    ("s2", "S2_C100_C300_PAIR", ("T02", "S05", "S10", "T04", "S07", "S08")),
    ("s2", "S2_C030_REFERENCE", ("T03", "S06", "S09")),
    ("s3", "S3_SCREEN_93001", ("S11", "S12", "S13")),
    ("s3", "S3_SCREEN_93002", ("S14", "S15", "S16")),
    ("s4", "S4_SCREEN_93001", ("S17", "S18", "S19")),
    ("s4", "S4_SCREEN_93002", ("S20", "S21", "S22")),
    ("s5", "S5_SCREEN_93001", ("S23", "S24", "S25")),
    ("s5", "S5_SCREEN_93002", ("S26", "S27", "S28")),
)
_BLOCK_BY_CASE = {"G20-" + case_id: (block, rank, order)
                  for rank, (_, block, ids) in enumerate(_CORE_BLOCK_IDS)
                  for order, case_id in enumerate(ids)}


def _read_cases():
    with (ROOT / SOURCE_CASES).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    cases = []
    for row in rows:
        if row["campaign"] != CAMPAIGN_ID or row["updates"] != "50000":
            raise ValueError("Unexpected G20 CSV contract")
        core = row["tier"] == "CORE"
        block, rank, order = _BLOCK_BY_CASE.get(row["case_id"], (
            f"{row['server'].upper()}_CONFIRM" if row["tier"] == "CONDITIONAL_CONFIRM" else "TRANSFER",
            100 if row["tier"] == "CONDITIONAL_CONFIRM" else 200, int(row["case_id"][-2:])))
        numeric = lambda key: float(row[key]) if core and row[key] else None
        case = Case(row["case_id"], row["run_id"], row["server"], row["role"], row["reference"],
                    int(row["teacher_seed"]) if row["teacher_seed"].isdigit() else None,
                    int(row["student_seed"]) if row["student_seed"] else None,
                    row["profile"], row["tier"], row["status"], numeric("lambda_con"),
                    numeric("alpha"), numeric("beta"), numeric("lambda_edge"),
                    numeric("student_a_peak_lr"), block, rank, order)
        if core:
            ref = REFERENCES[case.reference_id]
            if (case.teacher_seed, case.lambda_con) != (ref.teacher_seed, ref.lambda_con):
                raise ValueError("CSV reference recipe mismatch")
            if case.role == "S" and tuple(getattr(case, key) for key in asdict(PROFILES[case.profile])) != tuple(asdict(PROFILES[case.profile]).values()):
                raise ValueError("CSV Student profile mismatch")
            if case.role == "T" and case.profile != {1e-4: "C100", 3e-5: "C030", 3e-4: "C300"}[case.lambda_con]:
                raise ValueError("CSV Teacher profile mismatch")
        elif case.status != "NOT_ADMITTED":
            raise ValueError("Conditional slots must remain NOT_ADMITTED")
        cases.append(case)
    return tuple(cases)


verify_sources()
CASES = _read_cases()
CORE_CASES = tuple(case for case in CASES if case.tier == "CORE")
_BY_ID = MappingProxyType({case.case_id: case for case in CASES})
_BY_RUN = MappingProxyType({case.run_id: case for case in CASES})
if len(CASES) != 56 or len(_BY_ID) != 56 or len(_BY_RUN) != 56 or len(CORE_CASES) != 32:
    raise ValueError("G20 requires exactly 56 slots and 32 core cases")


def case_for(identifier):
    try:
        return _BY_ID[identifier] if identifier in _BY_ID else _BY_RUN[identifier]
    except KeyError as exc:
        raise ValueError(f"Unknown G20 case: {identifier}") from exc


def cases_for(server, include_conditional=False):
    if server not in SERVERS:
        raise ValueError("Unknown G20 server")
    return tuple(case for case in CASES if case.server_id == server and
                 (include_conditional or case.tier == "CORE"))


def teacher_for(reference_id):
    ref = REFERENCES[reference_id]
    return case_for(ref.teacher_case_id) if ref.teacher_case_id else None


def _decision(server, decision):
    if (decision.get("campaign_id") != CAMPAIGN_ID or decision.get("server_id") != server or
            decision.get("decision") != "SCREEN_SELECTION" or
            decision.get("selected_candidate") not in AXES[server] or
            decision.get("selected_status") not in ("PROMISING_PAIRED", "JOINT_SINGLE_RETEST") or
            not decision.get("selected_at_utc") or not decision.get("evidence_files")):
        raise ValueError("A locked, evidenced local screen decision is required")
    return decision["selected_candidate"], _object_sha(decision)


def _resolve(slot, server, reference, profile, decision_sha, reference_shas):
    if profile not in PROFILES or reference not in REFERENCES or server not in SERVERS:
        raise ValueError("Unregistered conditional assignment")
    ref = REFERENCES[reference]
    sha = reference_shas.get(reference)
    if not valid_sha(sha) or not valid_sha(decision_sha):
        raise ValueError("Conditional resolution requires actual reference and decision SHA256")
    role_name = "CONFIRM" if slot.tier == "CONDITIONAL_CONFIRM" else "TRANSFER"
    run = (f"G20_GF2_{role_name}_{server.upper()}_{reference}_TS{ref.teacher_seed}_"
           f"SS{slot.student_seed}_{profile}_PLH_W104_D122_FRESH50_v1")
    return replace(slot, server_id=server, reference_id=reference, teacher_seed=ref.teacher_seed,
                   profile=profile, run_id=run, status="PLANNED", lambda_con=ref.lambda_con,
                   block_id=f"{server.upper()}_{role_name}", resolution_sha256=decision_sha,
                   reference_sha256=sha, **asdict(PROFILES[profile]))


def resolve_confirmation(server, decision, reference_sha256):
    winner, sha = _decision(server, decision)
    result = []
    for slot in cases_for(server, include_conditional=True):
        if slot.tier != "CONDITIONAL_CONFIRM":
            continue
        reference = winner if slot.reference_id == "R_CSTAR" else slot.reference_id
        profile = winner if slot.profile.endswith("_XSTAR") else slot.profile
        result.append(_resolve(slot, server, reference, profile, sha, reference_sha256))
    if len(result) != 4:
        raise ValueError("Confirmation must resolve all four paired slots")
    return tuple(result)


def resolve_transfer(receipt, reference_sha256):
    server, reference, profile = receipt.get("server_id"), receipt.get("reference_id"), receipt.get("profile")
    if (receipt.get("campaign_id") != CAMPAIGN_ID or receipt.get("decision") != "TRANSFER_LOCK" or
            server not in ("s3", "s4", "s5") or profile not in AXES[server] or reference not in ("R1", "R2", "R3", "R4") or
            not receipt.get("selected_at_utc") or not receipt.get("evidence_files") or
            not all(receipt.get(key) is True for key in
                    ("reference_promising", "student_screen_passed", "local_controls_verified", "pair_identity_verified"))):
        raise ValueError("Transfer requires a prepared reference, one winning scalar and original-server controls")
    return tuple(_resolve(slot, server, reference, profile if slot.profile == "XSTAR" else "BASE",
                          _object_sha(receipt), reference_sha256)
                 for slot in CASES if slot.tier == "CONDITIONAL_TRANSFER")


def validate_case(case):
    slot = case_for(case.case_id)
    if not case.is_resolved:
        raise ValueError("Unresolved conditional slot is NOT_ADMITTED, never numerical BASE")
    if slot.tier == "CORE":
        if case != slot:
            raise ValueError("Modified core case")
        return case
    if not valid_sha(case.resolution_sha256) or not valid_sha(case.reference_sha256):
        raise ValueError("Conditional case lacks locked evidence/reference")
    if slot.tier == "CONDITIONAL_CONFIRM":
        allowed_refs = ("R3", "R4") if slot.reference_id == "R_CSTAR" else (slot.reference_id,)
        allowed_profiles = AXES[slot.server_id] if slot.profile.endswith("_XSTAR") else (slot.profile,)
        if case.server_id != slot.server_id or case.reference_id not in allowed_refs or case.profile not in allowed_profiles:
            raise ValueError("Conditional assignment outside the CSV candidate set")
    elif (case.server_id not in ("s3", "s4", "s5") or case.reference_id not in ("R1", "R2", "R3", "R4") or
          case.profile not in (AXES[case.server_id] if slot.profile == "XSTAR" else ("BASE",))):
        raise ValueError("Transfer changed owner/reference or combined Student scalars")
    expected = _resolve(slot, case.server_id, case.reference_id, case.profile,
                        case.resolution_sha256, {case.reference_id: case.reference_sha256})
    if case != expected:
        raise ValueError("Conditional case contains unregistered recipe changes")
    return case


def case_from_config(config):
    """Reconstruct a resolved contract without relying on mutable runtime IDs."""
    try:
        case = Case(**config["g20"]["case_contract"])
    except (KeyError, TypeError) as exc:
        raise ValueError("G20 config needs its complete immutable case_contract") from exc
    return validate_case(case)


@dataclass(frozen=True)
class Block:
    server_id: str
    block_id: str
    queue_rank: int
    run_ids: tuple[str, ...]
    cases: tuple[Case, ...]

    def to_dict(self):
        return dict(server_id=self.server_id, block_id=self.block_id, queue_rank=self.queue_rank,
                    run_ids=list(self.run_ids), case_ids=[case.case_id for case in self.cases])


def blocks_for(server, *, confirmation_cases=(), transfer_cases=(), transfer_first=False):
    if server not in SERVERS:
        raise ValueError("Unknown G20 server")
    result = [Block(server, block, rank, tuple(case_for("G20-" + c).run_id for c in ids),
                    tuple(case_for("G20-" + c) for c in ids))
              for rank, (owner, block, ids) in enumerate(_CORE_BLOCK_IDS) if owner == server]
    extra = ((transfer_cases, "CONDITIONAL_TRANSFER"), (confirmation_cases, "CONDITIONAL_CONFIRM")) if transfer_first else (
        (confirmation_cases, "CONDITIONAL_CONFIRM"), (transfer_cases, "CONDITIONAL_TRANSFER"))
    for cases, tier in extra:
        if not cases:
            continue
        cases = tuple(validate_case(case) for case in cases)
        expected = {case.case_id for case in CASES if case.tier == tier and
                    (tier == "CONDITIONAL_TRANSFER" or case.server_id == server)}
        if (len(cases) != 4 or {case.case_id for case in cases} != expected or
                any(case.server_id != server or case.tier != tier for case in cases) or
                len({case.resolution_sha256 for case in cases}) != 1 or
                len({case.profile for case in cases} - {"BASE"}) > 1 or
                len({case.reference_id for case in cases}) > (1 if tier == "CONDITIONAL_TRANSFER" else 2)):
            raise ValueError("Conditional block must contain the four cases from one locked decision")
        result.append(Block(server, cases[0].block_id, 100 + len(result), tuple(c.run_id for c in cases), cases))
    return tuple(result)


def registry_rows(server=None):
    return [case.to_dict() for case in (CASES if server is None else cases_for(server, True))]


def registry_document():
    return dict(campaign_id=CAMPAIGN_ID, registry_revision=REGISTRY_REVISION,
                method_revision=METHOD_REVISION, source_sha256=dict(SOURCE_SHAS),
                cases=registry_rows(), profiles={key: asdict(value) for key, value in PROFILES.items()},
                references={key: asdict(value) for key, value in REFERENCES.items()},
                r0_expected=dict(R0_PINS), r0_data_snapshot=dict(R0_DATA_PINS),
                window=dict(hours=20, admission_cutoff_hours=16, train_finish_hours=18,
                            close_hours=2, t0_utc=None, deadline_utc=None),
                sensors={"GF2": sensor_spec().to_dict()},
                blocks=[b.to_dict() for server in SERVERS for b in blocks_for(server)])


def registry_sha256():
    return _object_sha(registry_document())


def build_config(case):
    case = validate_case(case_for(case) if isinstance(case, str) else case)
    spec = sensor_spec()
    profile = PROFILES[case.profile] if case.role == "S" else PROFILES["BASE"]
    con = case.lambda_con if case.role == "T" else 0.
    meta = dict(case.to_dict(), case_contract=asdict(case), method_revision=METHOD_REVISION, registry_revision=REGISTRY_REVISION,
                source_plan=SOURCE_PLAN, sensor_spec=spec.to_dict(), sensor_spec_path=None,
                num_bands=4, max_pixel=1023, batch_size=48,
                window_path="work_dir/_g20/campaign_window.json", dataset_manifest=None,
                reference_manifest=None, teacher_checkpoint=None, teacher_sha256=None,
                tau_R=None, q_ref=None, q_cache=None, q_cache_sha256=None,
                teacher_from_scratch=True, backbone_from_scratch=True,
                aligner_init="FRESH_ZERO_LAST_LINEAR" if case.role == "T" else "TEACHER_CLONE",
                init_policy="FH12_named_tensor_fresh9_zero_extra_v1",
                init_generalization="Cplus1_native_channels_zero_extra_LH",
                candidate_grid=list(GRID_STEPS), fullstate_steps=list(FULLSTATE_STEPS),
                teacher_ref_step=50000, no_global_lock=True,
                alpha=profile.alpha, beta=profile.beta, lambda_E=profile.lambda_edge,
                lambda_edge=profile.lambda_edge,
                aligner_lr=1e-5 if case.role == "T" else profile.student_a_peak_lr,
                consistency_weight=con, offset_weight=con, offset_every=2,
                offset_radius=2. if case.role == "T" else 0., view_margin_hr=4,
                corruption_seed=case.seed + 100000,
                calibration=dict(n_patches=3072, seed=1234, probe="AXIS16",
                                 views="FIXED_HV_ROT4", subset_scope="SENSOR_SHARED_TRAIN_IDS"),
                hqnr_threshold=.964, ergas_goal=.552, ergas_strong_goal=.522,
                lr_schedule_id="COSINE_W100_BASE_v1", a_lr_switch_completed_updates=None,
                a_lr_after_multiplier=1.,
                lp=dict(sigma=1.98, kernel_size=41, padding="replicate", decimation="2::4,2::4",
                        generation_dtype="float64", cache_dtype="float32"))
    return dict(seed=case.seed, work_dir=f"work_dir/{case.run_id}", trainer="g20", phase="train",
                num_iter=50000, num_warmup=100, batch_size=48, test_batch_size=1, num_worker=4,
                num_bands=4, max_pixel=1023., mixed_precision="no", learning_rate=1e-4,
                optimizer="AdamW", weight_decay=.01, betas=[.9, .999], eps=1e-8,
                lr_scheduler="cosine", log_iter=100,
                train_feeder_args=dict(dataroot=None, crop=False, hflip=True, vflip=True,
                                       rot=True, ms_size=16, return_meta=True),
                val_feeder_args=dict(dataroot=None), test_reduced_feeder_args=dict(dataroot=None),
                test_full_feeder_args=dict(dataroot=None),
                model_args=dict(hidden_size=case.width, depth=list(case.depth), out_channels=4,
                                attn_locations=[], mode_modulation=False, norm="ln", dropout=0.),
                g20=meta)
