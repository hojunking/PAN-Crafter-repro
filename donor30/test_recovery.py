"""Recover durable evaluation/checkpoint evidence without re-running training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from . import reporting as R, publication as P
from .recovery import recover_campaign
from .test_reporting import config, evaluation, persist


def prepare(root, saved=False):
    cfg, ev = config(), evaluation()
    run = Path(root) / "runs" / cfg["run_id"] / "attempt_01"
    run.mkdir(parents=True)
    R.atomic_json(run / "config.json", cfg)
    path = run / "evaluation_0001000.json"
    R.atomic_json(path, ev)
    item = None
    if saved:
        model = run / "models/step_0001000.pt"
        model.parent.mkdir(); model.write_bytes(b"LOCAL_VERIFIED_SAVED_STATE")
        item = {"run_id": cfg["run_id"], "attempt": 1, "seed": cfg["seed"],
            "completed_update": 1000, "state_hash": ev["state_before"]["model_state_sha256"],
            "HQNR": ev["fr"]["metrics"]["HQNR"], "ERGAS": ev["rr"]["metrics"]["ERGAS"],
            "SCC": ev["rr"]["metrics"]["SCC"], "checkpoint_verified": True,
            "metrics_complete": True, "metrics_finite": True, "same_state_verified": True,
            "checkpoint_path": str(model), "file_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "server": cfg["server"], "dataset": cfg["dataset"], "phase": cfg["phase"]}
        R.atomic_json(run / "models/selection.json", {"schema": "DONOR30_ATOMIC_SELECTORS_V1",
            "run_id": cfg["run_id"], "threshold": cfg["hqnr_min"], "best_hqnr": item,
            "best_gate_ergas": None, "saved": [item]})
    return cfg, ev, path, item


class RecoveryTests(unittest.TestCase):
    def test_evaluation_before_enqueue_recovered_once(self):
        with tempfile.TemporaryDirectory() as root:
            prepare(root)
            first = recover_campaign(root, "s1")
            self.assertEqual(first["recovered_evaluations"], 1)
            outbox = next((Path(root) / "outbox").glob("D30-*.json"))
            initial = outbox.read_bytes()
            second = recover_campaign(root, "s1")
            self.assertEqual(second["recovered_evaluations"], 0)
            self.assertEqual(outbox.read_bytes(), initial)
            self.assertFalse(P.project_campaign(root, "s1")["conflicts"])

    def test_ledger_before_outbox_preserves_original_timestamp(self):
        with tempfile.TemporaryDirectory() as root:
            original = persist(root)
            path = Path(root) / "outbox" / (original["record_id"] + ".json")
            path.unlink()
            result = recover_campaign(root)
            self.assertEqual(result["recovered_outbox_files"], 1)
            self.assertEqual(json.loads(path.read_text()), original)
            self.assertEqual(len((Path(root) / "records.jsonl").read_text().splitlines()), 1)

    def test_saved_selector_restores_lost_campaign_observation(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, ev, path, item = prepare(root, saved=True)
            result = recover_campaign(root, "s1")
            self.assertEqual(result["observations"][0]["file_sha256"], item["file_sha256"])
            self.assertEqual(result["observations"][0]["evaluation_path"], str(path))
            projected = P.project_campaign(root, "s1")
            self.assertFalse(projected["conflicts"])
            self.assertEqual(projected["rows"][0][P.INDEX["Checkpoint_SHA"]], item["file_sha256"])

    def test_tampered_saved_checkpoint_and_torn_ledger_block(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, item = prepare(root, saved=True)
            Path(item["checkpoint_path"]).write_bytes(b"CORRUPT")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                recover_campaign(root)
        with tempfile.TemporaryDirectory() as root:
            prepare(root)
            (Path(root) / "records.jsonl").write_bytes(b'{"torn":')
            with self.assertRaisesRegex(ValueError, "torn"):
                recover_campaign(root)

    def test_recovery_requires_official_full20_unmixed_production_evidence(self):
        for corruption in ("injection", "masked", "missing_scene", "wrong_mean", "mixed_state"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as root:
                _, ev, path, _ = prepare(root)
                if corruption == "injection":
                    ev["metric_function_injection"] = True
                elif corruption == "masked":
                    ev["measurement_domain"] = "MASKED_FR_DIAGNOSTIC"
                elif corruption == "missing_scene":
                    ev["rr"]["per_scene"].pop()
                elif corruption == "wrong_mean":
                    ev["fr"]["per_scene"][0]["metrics"]["HQNR"] = .1
                else:
                    ev["fr_state"]["model_state_sha256"] = "f" * 64
                R.atomic_json(path, ev)
                with self.assertRaises(ValueError):
                    recover_campaign(root)
                self.assertFalse(list((Path(root) / "outbox").glob("D30-*.json")))


if __name__ == "__main__":
    unittest.main()
