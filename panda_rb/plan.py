"""Read-only B01 protocol. The supplied registry/CSV are the authority."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = Path("research_log/PANDA_REBUTTAL_STAGED_S135_2026-09-28")
CAMPAIGN = "PANDA_REBUTTAL_B01_WV3_S135_20260928_v1"
SERVERS = ("s1", "s3", "s5")
CASES = ("QFULL", "QMEAN", "QSHUF", "QESUR")
MODES = ("A_ON", "A_ZERO_INFERENCE_ONLY")
TEACHER_RUN = "FH12_S1_T_P0_W112_D123_WV3_S71001_FRESH50_v1"
SOURCE_RUN = "FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1"
VAL_GRID = tuple(range(1010, 50000, 1010)) + (50000,)
DIAGNOSTIC_STEPS = (0, 1000, 10000, 25000, 50000)


def campaign_dir(root=ROOT):
    return Path(root) / "work_dir/_panda_rb/20260928/B01"


def binding_path(root=ROOT):
    return campaign_dir(root) / "common/bindings.json"


def weights_dir(root=ROOT):
    return campaign_dir(root) / "common/weights"


def registry(root=ROOT):
    return json.loads((Path(root) / PLAN_DIR / "planning/experiment_registry.json").read_text())


def _csv(name, root):
    with (Path(root) / PLAN_DIR / "planning" / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _server(server):
    if server not in SERVERS:
        raise ValueError("B01 admits only s1/s3/s5; s2/s4 are untouched")


def training_runs(server=None, root=ROOT):
    if server is not None:
        _server(server)
    return [x for x in registry(root)["training_runs"] if server is None or x["server"] == server]


def case_for(run_id, root=ROOT):
    matches = [x for x in training_runs(root=root) if x["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Unregistered B01 run: {run_id}")
    return matches[0]


def run_dir(run_id, root=ROOT):
    row = case_for(run_id, root)
    return campaign_dir(root) / "RB01" / row["server"] / f"R{row['repeat']}" / row["case_id"]


def schedule(server, root=ROOT):
    _server(server)
    rows = [r for r in _csv("server_schedule.csv", root) if r["server"] == server]
    for row in rows:
        for key in ("position", "repeat", "seed"):
            if row[key]:
                row[key] = int(row[key])
    return rows


def shift_grid(root=ROOT):
    return json.loads((Path(root) / PLAN_DIR / "planning/shift_grid_v1.json").read_text())


def shifts(root=ROOT):
    return shift_grid(root)["shifts"]


def protocol_identity(root=ROOT):
    names = ["MASTER_PLAN_KR.md", "B01_RB01_RB02_DETAILED_KR.md"] + [
        "planning/" + n for n in ("experiment_registry.json", "training_runs_24.csv",
                                  "stress_curves_48.csv", "server_schedule.csv", "shift_grid_v1.json")]
    return {n: hashlib.sha256((Path(root) / PLAN_DIR / n).read_bytes()).hexdigest() for n in names}


def validate_plan(root=ROOT):
    reg = registry(root)
    if tuple(reg["servers"]) != SERVERS or reg["method_anchor"]["teacher_run_id"] != TEACHER_RUN:
        raise ValueError("B01 server/F1 anchor changed")
    expected_student = dict(layout='PLH', width=104, depth=[1, 2, 2], norm='ln',
                            attention=False, mode_modulation=False)
    if reg['method_anchor']['student'] != expected_student:
        raise ValueError('B01 Student must remain PLH W104D122 LN without attention/modulation')
    runs, curves = reg["training_runs"], reg["evaluation_tasks"]
    if len(runs) != 24 or len(curves) != 48 or len({r["run_id"] for r in runs}) != 24:
        raise ValueError("B01 requires exactly 24 runs and 48 paired curves")
    def serialized(value):
        return str(value)
    for rows, csv_name, key in ((runs, "training_runs_24.csv", "run_id"),
                               (curves, "stress_curves_48.csv", "task_id")):
        csv_rows = _csv(csv_name, root)
        if len(csv_rows) != len(rows) or {r[key] for r in rows} != {r[key] for r in csv_rows}:
            raise ValueError(f"Registry/CSV membership differs: {csv_name}")
        by_id = {r[key]: r for r in rows}
        for row in csv_rows:
            if any(serialized(by_id[row[key]][k]) != v for k, v in row.items()):
                raise ValueError(f"Registry/CSV field mismatch: {row[key]}")
    expected_seeds = {"s1": [9281101, 9281102], "s3": [9281301, 9281302], "s5": [9281501, 9281502]}
    if reg["independent_student_seeds"] != expected_seeds:
        raise ValueError("Registered independent seed cohort changed")
    for server in SERVERS:
        rows = training_runs(server, root)
        for repeat, seed in enumerate(expected_seeds[server], 1):
            block = [r for r in rows if r["repeat"] == repeat]
            if len(block) != 4 or {r["case_id"] for r in block} != set(CASES):
                raise ValueError("Each paired block requires exactly four cases")
            for row in block:
                if (row["seed"] != seed or row["updates"] != 50000 or row["dataset"] != "WV3"
                        or row["primary_selection"] != "EXACT_50000"
                        or row["secondary_selection"] != "RR_VAL_ERGAS_MIN"):
                    raise ValueError("B01 seed/budget/selector changed")
                tasks = [c for c in curves if c["source_run_id"] == row["run_id"]]
                if (len(tasks) != 2 or {c["mode"] for c in tasks} != set(MODES)
                        or any(c["optimizer_updates"] != 0 or c["curve_points"] != 49 or c["rr_scenes"] != 20
                               or c["source_selection"] != "EXACT_50000" for c in tasks)):
                    raise ValueError("Wrong paired stress curves")
                for task in tasks:
                    if (any(task[k] != row[k] for k in ('server','repeat','seed','case_id','paired_block_id','dataset'))
                            or task['statistical_unit_id'] != row['run_id']
                            or task['mode_is_paired_repeated_measure'] is not True):
                        raise ValueError('Curve repeats are not the same trained Student statistical unit')
        order = schedule(server, root)
        if len(order) != 25 or [x["position"] for x in order] != list(range(1, 26)):
            raise ValueError("Finite local schedule must have 25 ordered actions")
        if order[-1]["action"] != "STOP_FOR_REVIEW":
            raise ValueError("Missing finite STOP_FOR_REVIEW")
        for i, row in enumerate(rows):
            chunk = order[3*i:3*i+3]
            if ([a["action"] for a in chunk] != ["TRAIN_NATIVE_EVAL", "STRESS_CURVE", "STRESS_CURVE"]
                    or [a["mode"] for a in chunk] != ["", *MODES]
                    or any(a["run_id"] != row["run_id"] for a in chunk)):
                raise ValueError("Registry order differs from server schedule")
    grid = shift_grid(root)
    points = grid["shifts"]
    if len(points) != 49 or len({p["id"] for p in points}) != 49 or grid["coordinate_order"] != ["dy", "dx"]:
        raise ValueError("RB02 requires fixed dy/dx 49-point grid")
    expected = [(0., 0.)] + [(r*math.sin(math.radians(a)), r*math.cos(math.radians(a)))
                             for r in (.25, .5, 1., 2., 3., 4.) for a in range(0, 360, 45)]
    if any(not math.isclose(p[k], xy[j], abs_tol=1e-12) for p, xy in zip(points, expected)
           for j, k in enumerate(("dy", "dx"))):
        raise ValueError("Stress grid coordinates differ from preregistration")
    return dict(status="PASS", training_runs=24, stress_curves=48, independent_students_per_case=6,
                servers=list(SERVERS), protocol_identity=protocol_identity(root))


def build_config(run_id, root=ROOT):
    import yaml
    row = case_for(run_id, root)
    cfg = yaml.safe_load((Path(root) / "config" / (SOURCE_RUN + ".yaml")).read_text())
    old = cfg.pop("fh20r1")
    fixed = dict(num_iter=50000, num_warmup=100, batch_size=48, num_bands=8, max_pixel=2047.,
                 mixed_precision='no', learning_rate=1e-4, optimizer='AdamW', weight_decay=.01,
                 betas=[.9,.999], eps=1e-8, lr_scheduler='cosine')
    model = dict(hidden_size=104, depth=[1,2,2], out_channels=8, attn_locations=[],
                 mode_modulation=False, norm='ln', dropout=0.)
    if (any(cfg.get(k) != v for k,v in fixed.items()) or cfg['model_args'] != model
            or old['teacher_alias'] != 'F1' or old['teacher_run_id'] != TEACHER_RUN
            or old['input_layout'] != 'PLH' or old['profile'] != 'BASE'):
        raise ValueError('Original FH20R1 recipe config changed; B01 does not silently inherit a new method')
    cfg.update(seed=row["seed"], trainer="panda_rb", work_dir=str(run_dir(run_id, root)))
    cfg["panda_rb"] = {**copy.deepcopy(row), "campaign_id": CAMPAIGN, "role": "S", "profile": "BASE",
                       "input_layout": "PLH", "teacher_run_id": TEACHER_RUN, "teacher_alias": "F1",
                       "teacher_ref_step": 50000, "aligner_lr": 3e-6, "alpha": 1., "beta": .1,
                       "lambda_E": .002, "view_margin_hr": 4, "init_policy": old["init_policy"],
                       "candidate_grid": list(VAL_GRID), "diagnostic_steps": list(DIAGNOSTIC_STEPS),
                       "bindings_path": str(binding_path(root)), "weights_dir": str(weights_dir(root)),
                       "source_config": "config/" + SOURCE_RUN + ".yaml",
                       "protocol_identity": protocol_identity(root), "source_numeric_method": old["method_revision"]}
    return cfg
