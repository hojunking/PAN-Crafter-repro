"""Immutable QG40 definitions plus the user-selected preparation contract.

Importing defines the finite experiment space; it never starts a clock, binds a
dataset, loads a checkpoint, or authorizes any conditional case.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

CAMPAIGN_ID = "PANDA_QG40_20260920_v1"
REGISTRY_REVISION = "QG40_REGISTRY_20260920_v2_PREPARATION"
METHOD_REVISION = "QG40_SYNC_FREQ_C4_v1"
SERVERS = ("s1", "s2", "s3", "s4", "s5")
STUDENT_PROFILES = ("BASE", "A24R", "B20", "E10")
GRID_STEPS = tuple(range(1010, 50000, 1010)) + (50000,)
FULLSTATE_STEPS = (10000, 24240, 50000)
WINDOW_HOURS, ADMISSION_CUTOFF_HOURS, C3_CUTOFF_HOURS, CLOSE_HOURS = 40., 36., 22., 4.
SOURCE_PLAN = "research_log/0920/PAN_QG40_QB_GF2_40H_ExperimentPlan_2026-09-20.md"
SOURCE_CASES = "research_log/0920/PAN_QG40_Cases_2026-09-20.csv"
SOURCE_BASELINE = "research_log/0920/PAN_QG40_Baseline11_2026-09-20.csv"
SOURCE_PREPARATION = "research_log/0920/PAN_QB_GF2_DistributionAware_Preparation_2026-09-20.md"
SOURCE_SHAS = MappingProxyType({
    SOURCE_PLAN: "a6e3c1994a00ca2e132cb372cfb85242ab2f4c23b866241cb12be5e40d62c697",
    SOURCE_CASES: "baf0e4f5c828ee6ee3932481aac64a7996c97704cf3ed0696ed9494d4b7dc3ac",
    SOURCE_BASELINE: "3d7c20d9efa6c0ee27526ac0b5b0d1a597c5648240d8350b3142bb8c7ff0efbf",
    SOURCE_PREPARATION: "381df3ccb90da4b14f642b701389cbd2f0f3e2e425da54a2d181a5ea277af525",
})
ROOT = Path(__file__).resolve().parents[1]
SHEET_TABS = MappingProxyType({"s1": "QB-s1", "s2": "QB-s2", "s3": "GF2-s3(5090)",
                               "s4": "QB-s4", "s5": "GF2-s5"})
TEACHER_HOURS = MappingProxyType({"s1": 3.5, "s3": 3., "s4": 3.})
CALIBRATION_HOURS = MappingProxyType({"s1": 2.5, "s3": 2.5, "s4": 2.})
STUDENT_HOURS = MappingProxyType({"s1": 2.2, "s2": 2.5, "s3": 1.8, "s4": 2., "s5": 2.5})


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SplitBinding:
    path: str
    sha256: str
    source_identity: str
    lp_path: str | None = None
    lp_sha256: str | None = None

    def validate(self, *, verify_files=True):
        if not self.source_identity or len(self.sha256) != 64:
            raise ValueError("Split binding needs source identity and SHA256")
        if any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("Invalid split SHA256")
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError("Sensor split paths must be explicitly resolved absolute paths")
        if verify_files and (not path.is_file() or _sha(path) != self.sha256):
            raise ValueError(f"Missing or changed sensor split: {path}")
        if (self.lp_path is None) != (self.lp_sha256 is None):
            raise ValueError("LP path and SHA must be bound together")
        if self.lp_path is not None:
            lp = Path(self.lp_path)
            if not lp.is_absolute() or len(self.lp_sha256) != 64:
                raise ValueError("LP binding needs absolute path and SHA256")
            if verify_files and (not lp.is_file() or _sha(lp) != self.lp_sha256):
                raise ValueError("Missing or changed LP cache")


@dataclass(frozen=True)
class SensorSpec:
    sensor: str
    num_bands: int
    max_dn: int
    inverse_scale: float
    mtf_sensor: str
    hqnr_threshold: float
    ergas_goal: float
    ergas_strong_goal: float | None = None
    band_order: tuple[str, ...] | None = None
    splits: tuple[tuple[str, SplitBinding], ...] = ()
    source_provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        expected = {"QB": (2047, .920, 3.570, None), "GF2": (1023, .964, .552, .522)}
        if self.sensor not in expected:
            raise ValueError("QG40 supports QB and GF2 only")
        dn, h, e, strong = expected[self.sensor]
        if (self.num_bands, self.max_dn, self.inverse_scale, self.mtf_sensor,
            self.hqnr_threshold, self.ergas_goal, self.ergas_strong_goal) != (
                4, dn, dn / 2, self.sensor, h, e, strong):
            raise ValueError("Sensor constants do not match the QG40 contract")
        if self.band_order is not None:
            object.__setattr__(self, "band_order", tuple(self.band_order))
            if len(self.band_order) != 4 or len(set(self.band_order)) != 4:
                raise ValueError("Explicit four-band source order required")
        object.__setattr__(self, "splits", tuple(self.splits))
        object.__setattr__(self, "source_provenance", tuple(self.source_provenance))
        if len(dict(self.splits)) != len(self.splits):
            raise ValueError("Duplicate sensor split")

    @property
    def bands(self):
        return self.num_bands

    @property
    def max_pixel(self):
        return self.max_dn

    @property
    def is_bound(self):
        return self.band_order is not None and set(dict(self.splits)) == {"train", "val", "rr", "fr"}

    def split(self, name):
        name = {"valid": "val", "validation": "val"}.get(name, name)
        try:
            return dict(self.splits)[name]
        except KeyError as exc:
            raise ValueError(f"Unbound {self.sensor} split: {name}") from exc

    def bind(self, *, band_order, splits: Mapping, source_provenance: Mapping,
             verify_files=True):
        bound = {}
        for name, value in splits.items():
            name = {"valid": "val", "validation": "val"}.get(name, name)
            if name in bound:
                raise ValueError("Duplicate split alias")
            binding = value if isinstance(value, SplitBinding) else SplitBinding(**value)
            binding.validate(verify_files=verify_files)
            bound[name] = binding
        if set(bound) != {"train", "val", "rr", "fr"}:
            raise ValueError("Bind train, val, rr, and fr explicitly; no path fallback")
        return replace(self, band_order=tuple(band_order), splits=tuple(sorted(bound.items())),
                       source_provenance=tuple(sorted(source_provenance.items())))

    def to_dict(self):
        result = asdict(self)
        result["band_order"] = None if self.band_order is None else list(self.band_order)
        result["splits"] = {name: asdict(value) for name, value in self.splits}
        result["source_provenance"] = dict(self.source_provenance)
        return result

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["splits"] = tuple((name, SplitBinding(**binding)) for name, binding in
                               data.get("splits", {}).items())
        data["source_provenance"] = tuple(data.get("source_provenance", {}).items())
        return cls(**data)


SENSORS = MappingProxyType({
    "QB": SensorSpec("QB", 4, 2047, 1023.5, "QB", .920, 3.570),
    "GF2": SensorSpec("GF2", 4, 1023, 511.5, "GF2", .964, .552, .522),
})


def sensor_spec(sensor):
    try:
        return SENSORS[sensor]
    except KeyError as exc:
        raise ValueError(f"Unsupported QG40 sensor: {sensor}") from exc


@dataclass(frozen=True)
class Case:
    run_id: str
    server_id: str
    sensor: str
    role: str
    reference_id: str
    teacher_seed: int
    student_seed: int | None
    input_layout: str
    width: int
    depth: tuple[int, ...]
    profile: str
    tier: str
    block_id: str
    queue_rank: int
    within_block_order: int
    branch_condition: str
    dependencies: tuple[str, ...]
    teacher_run_id: str
    updates: int
    baseline_id_preserved: bool
    case_id: str
    sheet_tab: str

    @property
    def seed(self):
        return self.teacher_seed if self.role == "T" else self.student_seed

    @property
    def teacher_input_layout(self):
        return "P0"

    @property
    def teacher_alias(self):
        return self.reference_id

    @property
    def reservation_hours(self):
        return (TEACHER_HOURS if self.role == "T" else STUDENT_HOURS)[self.server_id]

    def to_dict(self):
        return dict(asdict(self), depth=list(self.depth), dependencies=list(self.dependencies),
                    seed=self.seed, teacher_input_layout="P0", kind="TRAIN",
                    campaign_id=CAMPAIGN_ID, config_path=f"config/qg40/{self.run_id}.yaml")


def _read_cases(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        if row["campaign"] != CAMPAIGN_ID or row["status"] != "DEFINED_NOT_DEPLOYED":
            raise ValueError("Unexpected campaign source definition")
        result.append(Case(
            run_id=row["run_id"], server_id=row["server"], sensor=row["sensor"],
            role={"Teacher": "T", "Student": "S"}[row["role"]], reference_id=row["reference_id"],
            teacher_seed=int(row["teacher_seed"]), student_seed=int(row["student_seed"]) if row["student_seed"] else None,
            input_layout=row["input"], width=int(row["width"]), depth=tuple(map(int, row["depth"])),
            profile=row["profile"], tier=row["tier"], block_id=row["block"],
            queue_rank=int(row["queue_rank"]), within_block_order=int(row["within_block_order"]),
            branch_condition=row["branch_condition"], dependencies=tuple(json.loads(row["dependencies"])),
            teacher_run_id=row["teacher_run_id"], updates=int(row["updates"]),
            baseline_id_preserved=row["baseline_id_preserved"] == "True", case_id=row["case_id"],
            sheet_tab=row["sheet_tab"]))
    return tuple(result)


def verify_sources():
    for name, expected in SOURCE_SHAS.items():
        if _sha(ROOT / name) != expected:
            raise ValueError(f"QG40 source definition changed; new registry revision required: {name}")
    return dict(SOURCE_SHAS)


verify_sources()
CASES = _read_cases(SOURCE_CASES)
BASELINE_CASES = _read_cases(SOURCE_BASELINE)
_CASES_BY_ID = MappingProxyType({case.run_id: case for case in CASES})
if len(CASES) != 89 or len(_CASES_BY_ID) != 89 or len(BASELINE_CASES) != 11:
    raise ValueError("QG40 requires 89 unique definitions and 11 preserved baseline IDs")
if tuple(case for case in CASES if case.baseline_id_preserved) != BASELINE_CASES:
    raise ValueError("Baseline11 does not match the full registry")


def case_for(run_id):
    try:
        return _CASES_BY_ID[run_id]
    except KeyError as exc:
        raise ValueError(f"Unregistered QG40 run: {run_id}") from exc


def cases_for(server):
    if server not in SERVERS:
        raise ValueError(f"Unknown QG40 server: {server}")
    return tuple(case for case in CASES if case.server_id == server)


def teacher_for(reference_id):
    if reference_id in SERVERS:
        reference_id = {"s1": "QB_TB", "s2": "QB_TA", "s3": "GF2_TA",
                        "s4": "QB_TA", "s5": "GF2_TA"}[reference_id]
    for case in CASES:
        if case.role == "T" and case.reference_id == reference_id:
            return case
    raise ValueError(f"Unknown QG40 reference: {reference_id}")


def active_cases(server, screen="BASE", confirm="BASE", enable_c3=False, include_reserve=False):
    """Resolve definitions only. Evidence and atomic admission are separate gates."""
    cases = cases_for(server)
    if screen not in STUDENT_PROFILES or confirm not in STUDENT_PROFILES:
        raise ValueError("Invalid student branch")
    if confirm != "BASE" and confirm != screen:
        raise ValueError("Confirmation cannot replace the one selected screen profile")
    if server not in ("s2", "s5") and (screen != "BASE" or confirm != "BASE"):
        raise ValueError("Student trials belong only to s2/s5")
    if enable_c3 and server not in ("s1", "s3"):
        raise ValueError("C3 belongs only to s1/s3")
    conditions = {"ALWAYS", f"SCREEN_{server}_EQ_{screen}", f"CONFIRM_{server}_EQ_{confirm}"}
    if enable_c3:
        conditions.add(f"TC3_ENABLE_{server}")
    return tuple(sorted((case for case in cases if case.branch_condition in conditions and
                         (include_reserve or case.tier != "TIME_RESERVE")),
                        key=lambda case: (case.queue_rank, case.within_block_order)))


@dataclass(frozen=True)
class Block:
    server_id: str
    block_id: str
    queue_rank: int
    run_ids: tuple[str, ...]

    @property
    def is_c3(self):
        return "TC3_PACKAGE" in self.block_id

    def to_dict(self):
        return dict(asdict(self), run_ids=list(self.run_ids), is_c3=self.is_c3)


def blocks_for(server, **branches):
    groups = {}
    for case in active_cases(server, **branches):
        groups.setdefault(case.block_id, []).append(case)
    return tuple(Block(server, key, cases[0].queue_rank, tuple(case.run_id for case in cases))
                 for key, cases in groups.items())


def registry_rows(server=None):
    return [case.to_dict() for case in (CASES if server is None else cases_for(server))]


def registry_document():
    actions = [dict(action_id=f"QG40_{case.reference_id}_CAL_v1", kind="CALIBRATION",
                    server_id=case.server_id, reference_id=case.reference_id,
                    teacher_run_id=case.run_id, required_updates=50000,
                    branch_condition=case.branch_condition) for case in CASES if case.role == "T"]
    actions.extend(dict(action_id=f"QG40_{ref}_IMPORT_{server}_v1", kind="IMPORT",
                        server_id=server, reference_id=ref, from_server=owner)
                   for server, ref, owner in (("s2", "QB_TA", "s4"), ("s5", "GF2_TA", "s3")))
    return dict(campaign_id=CAMPAIGN_ID, registry_revision=REGISTRY_REVISION,
                method_revision=METHOD_REVISION, source_sha256=dict(SOURCE_SHAS),
                source_note="Preparation MD selected by user; TA sharing retained; INDEP1 excluded; 40h operating plan retained",
                window=dict(hours=40, admission_cutoff_hours=36, c3_cutoff_hours=22,
                            close_hours=4, t0_utc=None, deadline_utc=None),
                cases=registry_rows(), actions=actions,
                sensors={name: value.to_dict() for name, value in SENSORS.items()})


def registry_sha256():
    return _object_sha(registry_document())


def build_config(case):
    """Unbound recipe; all measured reference values remain null until verified."""
    case = case_for(case) if isinstance(case, str) else case
    if case_for(case.run_id) != case:
        raise ValueError("Modified or unregistered case")
    spec = sensor_spec(case.sensor)
    con = (3e-4 if case.profile == "C3" else 1e-4) if case.role == "T" else 0.
    edge = (.001 if case.profile == "E10" else .002)
    meta = dict(case.to_dict(), method_revision=METHOD_REVISION, registry_revision=REGISTRY_REVISION,
                source_plan=SOURCE_PLAN, sensor_spec=spec.to_dict(), sensor_spec_path=None,
                num_bands=4, max_pixel=spec.max_dn, batch_size=48,
                window_path="work_dir/_qg40/campaign_window.json", dataset_manifest=None,
                reference_manifest=None, teacher_checkpoint=None, teacher_sha256=None,
                tau_R=None, q_ref=None, q_cache=None, q_cache_sha256=None,
                teacher_from_scratch=True, backbone_from_scratch=True,
                aligner_init="FRESH_ZERO_LAST_LINEAR" if case.role == "T" else "TEACHER_CLONE",
                init_policy="FH12_named_tensor_fresh9_zero_extra_v1",
                init_generalization="Cplus1_native_channels_zero_extra_LH",
                candidate_grid=list(GRID_STEPS), fullstate_steps=list(FULLSTATE_STEPS),
                teacher_ref_step=50000, teacher_input_layout="P0", no_global_lock=True,
                alpha=1., beta=.2 if case.profile == "B20" else .1,
                lambda_E=edge, lambda_edge=edge, aligner_lr=1e-5 if case.role == "T" else 3e-6,
                consistency_weight=con, offset_weight=con, offset_every=2,
                offset_radius=2. if case.role == "T" else 0., view_margin_hr=4,
                corruption_seed=case.seed + 100000, calibration=dict(n_patches=3072, seed=1234,
                    probe="AXIS16", views="FIXED_HV_ROT4", subset_scope="SENSOR_SHARED_TRAIN_IDS"),
                hqnr_threshold=spec.hqnr_threshold, ergas_goal=spec.ergas_goal,
                ergas_strong_goal=spec.ergas_strong_goal,
                lr_schedule_id="COSINE_W100_A24R_T24240_DIV3_v1" if case.profile == "A24R" else "COSINE_W100_BASE_v1",
                a_lr_switch_completed_updates=24240 if case.profile == "A24R" else None,
                a_lr_after_multiplier=1 / 3 if case.profile == "A24R" else 1.,
                lp=dict(sigma=1.98, kernel_size=41, padding="replicate", decimation="2::4,2::4",
                        generation_dtype="float64", cache_dtype="float32"))
    return dict(seed=case.seed, work_dir=f"work_dir/{case.run_id}", trainer="qg40", phase="train",
                num_iter=50000, num_warmup=100, batch_size=48, test_batch_size=1, num_worker=4,
                num_bands=4, max_pixel=float(spec.max_dn), mixed_precision="no", learning_rate=1e-4,
                optimizer="AdamW", weight_decay=.01, betas=[.9, .999], eps=1e-8,
                lr_scheduler="cosine", log_iter=100,
                train_feeder_args=dict(dataroot=None, crop=False, hflip=True, vflip=True, rot=True,
                                       ms_size=16, return_meta=True),
                val_feeder_args=dict(dataroot=None), test_reduced_feeder_args=dict(dataroot=None),
                test_full_feeder_args=dict(dataroot=None),
                model_args=dict(hidden_size=case.width, depth=list(case.depth), out_channels=4,
                                attn_locations=[], mode_modulation=False, norm="ln", dropout=0.), qg40=meta)
