"""Offline donor retention and fixed64 lineage tests; no GPU/network access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from . import reporting as R
from .plan import CAMPAIGN_ID, METHOD_REVISION
from directshift48.test_reporting import config as old_config


def config(server="s1", phase="donor"):
    cfg = old_config(server)
    cfg.update(run_id="D30_" + cfg["run_id"], campaign_id=CAMPAIGN_ID,
               phase=phase, method_revision=METHOD_REVISION)
    if phase == "stage2":
        cfg["student_architecture"] = {"bands": 8 if server == "s1" else 4, "width": 104, "depth": [1, 2, 2]}
    return cfg


def evaluation(cfg=None, step=1000, h=.959, e=2.2):
    cfg = cfg or config()
    ident = {"run_id": cfg["run_id"], "attempt": cfg["attempt"], "seed": cfg["seed"],
        "completed_update": step, "model_state_sha256": R.digest([cfg["run_id"], step]),
        "evaluator_revision": "d" * 64, "inference_policy_digest": "c" * 64}
    rr = {"ERGAS": e, "SCC": .98, "SAM": 2., "PSNR": 40., "SSIM": .98,
          "Q8" if cfg["dataset"] == "WV3" else "Q4": .95, "RMSE": 20., "CC": .99}
    fr = {"HQNR": h, "D_s": .02, "D_lambda": .02, "JQM": .98}
    gate = {"same_state_verified": True, "metrics_complete": True, "metrics_finite": True,
        "full_method_eligible": True, "joint_pass": False, "HQNR": h, "ERGAS": e,
        "scene_completeness": {s: {"expected": 20, "completed": 20,
            "missing": [], "duplicates": [], "unexpected": []} for s in ("rr", "fr")}}
    ev = {"record_type": "EVAL_SUMMARY", "full_method_eligible": True,
        "measurement_domain": "OFFICIAL_FULL20_NATIVE", "metric_function_injection": False,
        "test_aware": True, "eval_protocol_id": "d" * 64, "qualification": gate,
        **{k: deepcopy(ident) for k in ("state_before", "rr_state", "fr_state", "state_after")}}
    for split, metrics in (("rr", rr), ("fr", fr)):
        ev[split] = {"metrics": metrics, "per_scene": [{"scene_id": f"{cfg['dataset']}:{split}:{i}",
            "metrics": deepcopy(metrics)} for i in range(20)]}
    return ev


def persist(root, cfg=None, ev=None, artifact=None):
    cfg = cfg or config(); ev = ev or evaluation(cfg)
    path = Path(root) / "runs" / cfg["run_id"] / "attempt_01"
    path.mkdir(parents=True, exist_ok=True)
    R.atomic_json(path / "config.json", cfg)
    R.atomic_json(path / ("evaluation_%07d.json" % ev["state_before"]["completed_update"]), ev)
    return R.emit_evaluation(root, cfg, ev, artifact,
        timestamp="2026-10-15T05:00:00+00:00")


class ReportingTests(unittest.TestCase):
    def test_saved_below_joint_target_allowed_and_distinguished(self):
        cfg, ev = config(), evaluation()
        ident = ev["state_before"]
        artifact = {"weights_saved": True, "checkpoint_path": "checkpoint.pt",
            "file_sha256": "f" * 64, "model_state_sha256": ident["model_state_sha256"], "identity": ident,
            "selection_reason": "HQNR_GATE_ERGAS_BEST"}
        record = R.make_record(cfg, ev, artifact)
        R.validate_record(record)
        self.assertFalse(record["source"]["extra"]["joint_pass"])
        self.assertTrue(record["source"]["extra"]["weights_saved"])
        self.assertEqual(record["row"][R.INDEX["Checkpoint_SHA"]], "f" * 64)

    def test_mixed_state_and_wrong_saved_state_rejected(self):
        cfg, ev = config(), evaluation()
        ev["fr_state"]["model_state_sha256"] = "a" * 64
        with self.assertRaisesRegex(R.ReportingError, "Mixed"):
            R.make_record(cfg, ev)
        ev = evaluation()
        with self.assertRaises(R.ReportingError):
            R.make_record(cfg, ev, {"weights_saved": True, "file_sha256": "f" * 64,
                "checkpoint_path": "x", "model_state_sha256": "a" * 64, "identity": ev["state_before"]})

    def test_tampered_source_and_unknown_lane_rejected(self):
        record = R.make_record(config(), evaluation())
        record["source"]["metrics"]["HQNR"] = .999
        with self.assertRaises(R.ReportingError):
            R.validate_record(record)
        cfg = config(); cfg["server"] = "s3"
        with self.assertRaises(ValueError):
            R.make_record(cfg, evaluation(cfg))

    def test_idempotent_local_evidence_and_conflict(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, ev = config(), evaluation()
            first = persist(root, cfg, ev)
            R.LocalOutbox(root).enqueue(first)
            self.assertEqual(len((Path(root) / "records.jsonl").read_text().splitlines()), 1)
            with self.assertRaises(R.ReportingError):
                R.emit_evaluation(root, cfg, evaluation(cfg, h=.961),
                    timestamp="2026-10-15T05:00:00+00:00")

    def test_stage2_cost_excludes_teacher_and_freezes_lr(self):
        cfg = config(phase="stage2")
        before = deepcopy(cfg)
        cost = R.cost_config(cfg)
        self.assertEqual(cost["baseline_resolved"]["teacher_architecture"]["width"], 104)
        self.assertEqual(cfg, before)
        row = R.make_record(cfg, evaluation(cfg))["row"]
        self.assertEqual(row[R.INDEX["A_peak_lr"]], 0.)


if __name__ == "__main__":
    unittest.main()
