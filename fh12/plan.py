"""FH12's explicit, finite experiment registry (importing never starts a clock)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json

CAMPAIGN_ID = "WV3_FH12_TSCRATCH_20260918_v1"
METHOD_REVISION = "FH12_SYNC_FREQ_NATIVE_TEACHER_v1"
REGISTRY_REVISION = "FH12_REGISTRY_20260918_v1"
SOURCE_PLAN = "research_log/PAN_FH12_WV3_LPAN_HPAN_FreshTeacher_12H_2026-09-18.md"
SERVERS = ("s1", "s2", "s3", "s4", "s5")
LAYOUT_CHANNELS = {"P0": 9, "PL": 10, "PH": 10, "PLH": 11}
GRID_STEPS = tuple(range(1010, 50000, 1010)) + (50000,)
WINDOW_HOURS = 12.0
CLOSE_HOURS = .75
SETUP_CALIBRATION_HOURS = 1.25
TEACHER_HOURS = {"s1": 2.6, "s2": 3.0, "s3": 1.7, "s4": 1.7, "s5": 3.2}
STUDENT_HOURS = {"s1": 2.5, "s2": 2.6, "s3": 1.4, "s4": 1.4, "s5": 3.5}
TEACHER_LAYOUT = {"s1": "P0", "s2": "PL", "s3": "PH", "s4": "PLH", "s5": "PLH"}
# order, tier, layout, width, depth. No generated Cartesian product or legacy cases.
_STUDENTS = {
    "s1": ((30,"CORE","P0",104,(1,2,1)), (40,"CORE","PLH",104,(1,2,1))),
    "s2": ((30,"CORE","PL",104,(1,2,1)), (40,"CORE","P0",104,(1,2,1))),
    "s3": ((30,"CORE","PH",104,(1,2,1)), (40,"CORE","P0",104,(1,2,1)),
           (50,"RESERVE_D","PH",104,(1,2,2)), (60,"RESERVE_W","PH",112,(1,2,1))),
    "s4": ((30,"CORE","PLH",104,(1,2,1)), (40,"CORE","P0",104,(1,2,1)),
           (50,"CORE","PL",104,(1,2,1)), (60,"CORE","PH",104,(1,2,1)),
           (70,"RESERVE_D","PLH",104,(1,2,2)), (80,"RESERVE_W","PLH",112,(1,2,1))),
    "s5": ((30,"CORE","PLH",104,(1,2,1)), (40,"RESERVE_D","PLH",104,(1,2,2))),
}


@dataclass(frozen=True)
class Case:
    server_id: str
    order: int
    role: str
    tier: str
    input_layout: str
    width: int
    depth: tuple
    teacher_seed: int
    student_seed: int | None

    @property
    def teacher_input_layout(self):
        return TEACHER_LAYOUT[self.server_id]

    @property
    def seed(self):
        return self.teacher_seed if self.role == "T" else self.student_seed

    @property
    def teacher_run_id(self):
        return (f"FH12_{self.server_id.upper()}_T_{self.teacher_input_layout}"
                f"_W112_D123_WV3_S{self.teacher_seed}_FRESH50_v1")

    @property
    def run_id(self):
        if self.role == "T":
            return self.teacher_run_id
        depth = ''.join(map(str, self.depth))
        return (f"FH12_{self.server_id.upper()}_S_{self.input_layout}_W{self.width}_D{depth}"
                f"_WV3_T{self.teacher_input_layout}_TS{self.teacher_seed}_SS{self.student_seed}_FRESH50_v1")

    @property
    def reservation_hours(self):
        if self.role == "T":
            return TEACHER_HOURS[self.server_id]
        factor = 1.10 if self.tier == "RESERVE_D" else 1.20 if self.tier == "RESERVE_W" else 1.0
        return STUDENT_HOURS[self.server_id] * factor

    def to_dict(self):
        return dict(asdict(self), depth=list(self.depth), run_id=self.run_id,
                    teacher_run_id=self.teacher_run_id, teacher_input_layout=self.teacher_input_layout,
                    seed=self.seed, reservation_hours=self.reservation_hours,
                    config_path=f"config/{self.run_id}.yaml", kind="TRAIN")


CASES = tuple(c for server in SERVERS for c in (
    Case(server, 10, "T", "CORE", TEACHER_LAYOUT[server], 112, (1,2,3),
         71002 if server == "s5" else 71001, None),
    *(Case(server, order, "S", tier, layout, width, depth,
           71002 if server == "s5" else 71001, 72002 if server == "s5" else 72001)
      for order, tier, layout, width, depth in _STUDENTS[server])
))
assert len(CASES) == 21 and sum(c.tier == "CORE" for c in CASES) == 16


def cases_for(server):
    if server not in SERVERS:
        raise ValueError(f"Unknown FH12 server: {server!r}")
    return tuple(c for c in CASES if c.server_id == server)


def teacher_for(server):
    return cases_for(server)[0]


def case_for(run_id):
    return next((c for c in CASES if c.run_id == run_id), None) or _unknown(run_id)


def _unknown(run_id):
    raise ValueError(f"Unregistered FH12 run: {run_id!r}")


def registry_rows(server=None):
    rows = []
    for srv in SERVERS if server is None else (server,):
        teacher = teacher_for(srv)
        rows.append(dict(server_id=srv, order=0, kind="PREFLIGHT", tier="CORE",
                         run_id=f"FH12_{srv.upper()}_PREFLIGHT", depends_on=[]))
        for case in cases_for(srv):
            row = case.to_dict()
            row["depends_on"] = [f"FH12_{srv.upper()}_PREFLIGHT" if case.role == "T"
                                 else f"FH12_{srv.upper()}_CALIBRATION"]
            rows.append(row)
        rows.append(dict(server_id=srv, order=20, kind="CALIBRATION", tier="CORE",
                         run_id=f"FH12_{srv.upper()}_CALIBRATION", teacher_run_id=teacher.run_id,
                         depends_on=[teacher.run_id]))
    return sorted(rows, key=lambda r: (r["server_id"], r["order"]))


def registry_sha256():
    return hashlib.sha256(json.dumps(registry_rows(), sort_keys=True).encode()).hexdigest()


def build_config(case):
    """Portable immutable inputs; data/cache paths are bound by preflight, not copied from T0."""
    case = case_for(case) if isinstance(case, str) else case
    root = "data/PanCollection/WV3/"
    meta = dict(case.to_dict(), campaign_id=CAMPAIGN_ID, method_revision=METHOD_REVISION,
                registry_revision=REGISTRY_REVISION, source_plan=SOURCE_PLAN,
                window_path=f"work_dir/_fh12/{case.server_id}/window.json",
                reference_manifest=(None if case.role == "T" else
                    f"work_dir/_fh12/{case.server_id}/references/{case.teacher_run_id}/reference_manifest.json"),
                teacher_checkpoint=(None if case.role == "T" else
                    f"work_dir/{case.teacher_run_id}/candidates/50000/model.safetensors"),
                teacher_from_scratch=True, candidate_grid=list(GRID_STEPS),
                init_policy="FH12_named_tensor_fresh9_zero_extra_v1", no_global_lock=True,
                alpha=1.0, beta=.1, lambda_E=.002,
                aligner_lr=1e-5 if case.role == "T" else 3e-6,
                offset_weight=1e-4 if case.role == "T" else 0.0,
                offset_every=2, offset_radius=2.0 if case.role == "T" else 0.0,
                view_margin_hr=4, corruption_seed=case.seed + 100000,
                calibration={"n_patches": 3072, "seed": 1234, "probe": "AXIS16"},
                hqnr_threshold=.9585, ergas_goal=2.040, teacher_ref_step=50000)
    return dict(seed=case.seed, work_dir=f"work_dir/{case.run_id}", trainer="fh12",
                phase="train", num_iter=50000, num_warmup=100, batch_size=48,
                test_batch_size=1, num_worker=4, num_bands=8, max_pixel=2047.0, mixed_precision="no",
                learning_rate=1e-4, optimizer="AdamW", weight_decay=.01,
                betas=[.9,.999], eps=1e-8, lr_scheduler="cosine", log_iter=100,
                train_feeder_args=dict(dataroot=root+"train_wv3.h5", crop=False,
                    hflip=True, vflip=True, rot=True, ms_size=16, return_meta=True),
                val_feeder_args=dict(dataroot=root+"valid_wv3.h5"),
                test_reduced_feeder_args=dict(dataroot=root+"reduced_examples_h5/test_wv3_multiExm1.h5"),
                test_full_feeder_args=dict(dataroot=root+"full_examples_mat20/test_wv3_OrigScale_mat20.h5"),
                model_args=dict(hidden_size=case.width, depth=list(case.depth),
                    out_channels=8, attn_locations=[], mode_modulation=False,
                    norm="ln", dropout=0.0), fh12=meta)
