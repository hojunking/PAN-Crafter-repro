"""Small CPU checkpoint tests: real serialization, strict reload, atomic policy."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from . import artifacts as A
from . import plan
from panda_method.calibration import state_dict_sha256


class TinyTrainer:
    def __init__(self):
        self.stage = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.AdamW(self.stage.parameters(), lr=1e-4)
        self.updates, self.poisoned = 0, False

    def step(self):
        self.optimizer.zero_grad()
        self.stage(torch.ones(2, 2)).square().mean().backward()
        self.optimizer.step()
        self.updates += 1

    def state_dict(self):
        if self.poisoned:
            raise RuntimeError("poisoned")
        return deepcopy({"stage": self.stage.state_dict(), "optimizer": self.optimizer.state_dict(),
                         "updates": self.updates})

    def load_state_dict(self, state):
        self.stage.load_state_dict(state["stage"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]


class TinyStream:
    def __init__(self):
        self.rng = np.random.default_rng(21)
        self.cursor = 0

    def state_dict(self):
        return {"rng": deepcopy(self.rng.bit_generator.state), "cursor": self.cursor}


def evaluation(trainer, cfg, h=.960, e=2.3):
    identity = {"run_id": cfg["run_id"], "attempt": cfg["attempt"], "seed": cfg["seed"],
                "completed_update": trainer.updates, "model_state_sha256": state_dict_sha256(trainer.stage)}
    result = {name: deepcopy(identity) for name in ("state_before", "rr_state", "fr_state", "state_after")}
    result["qualification"] = {"same_state_verified": True, "metrics_complete": True, "metrics_finite": True}
    result["rr"] = {"metrics": {"ERGAS": e, "SCC": .98},
                    "per_scene": [{"scene_id": f"RR:{i}", "raw_prediction_sha256": "a" * 64} for i in range(20)]}
    result["fr"] = {"metrics": {"HQNR": h},
                    "per_scene": [{"scene_id": f"FR:{i}", "raw_prediction_sha256": "b" * 64} for i in range(20)]}
    return result


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = plan.resolve_donor_case(plan.default_config())
        self.trainer, self.stream = TinyTrainer(), TinyStream()
        self.trainer.step()
        self.ev = evaluation(self.trainer, self.cfg)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, ev=None, factory=None, verify=None):
        return A.save_candidate(self.trainer, self.stream, ev or self.ev, self.root, self.cfg,
            reload_factory=factory or (lambda cfg: TinyTrainer()), verify_evaluate=verify,
            state_hash_fn=state_dict_sha256)

    def test_real_reload_and_two_atomic_pointers(self):
        result = self.save()
        record, artifact = result["selection_record"], result["artifact"]
        self.assertTrue(artifact["checkpoint_verified"])
        self.assertFalse(artifact["predictions_reverified"])
        self.assertEqual(set(result["reasons"]), {"BEST_HQNR", "HQNR_GATE_ERGAS_BEST"})
        path = Path(record["checkpoint_path"])
        self.assertEqual(path.name, "step_0000001.pt")
        self.assertEqual(A.file_sha256(path), artifact["file_sha256"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertTrue(A._same(payload["trainer"], self.trainer.state_dict()))
        self.assertTrue(A._same(payload["stream"], self.stream.state_dict()))
        self.assertEqual(payload["config"], self.cfg)
        selection = A.read_selection(self.root)
        self.assertEqual(selection["best_hqnr"], selection["best_gate_ergas"])
        self.assertEqual(len(selection["saved"]), 1)
        self.assertTrue(plan._record_valid(record))

    def test_best_hqnr_and_gate_ergas_can_diverge(self):
        first = self.save()
        self.trainer.step()
        second = self.save(evaluation(self.trainer, self.cfg, h=.959, e=2.1))
        self.assertEqual(second["reasons"], ["HQNR_GATE_ERGAS_BEST"])
        selection = A.read_selection(self.root)
        self.assertEqual(selection["best_hqnr"]["completed_update"], 1)
        self.assertEqual(selection["best_gate_ergas"]["completed_update"], 2)
        self.assertEqual(len(selection["saved"]), 2)
        self.assertTrue(Path(first["record"]["checkpoint_path"]).is_file())
        self.trainer.step()
        self.assertIsNone(self.save(evaluation(self.trainer, self.cfg, h=.958, e=2.0)))
        self.assertEqual(len(list((self.root / "models").glob("step_*.pt"))), 2)

    def test_below_gate_hqnr_saved_without_gate_pointer(self):
        result = self.save(evaluation(self.trainer, self.cfg, h=.958))
        self.assertEqual(result["reasons"], ["BEST_HQNR"])
        self.assertIsNone(A.read_selection(self.root)["best_gate_ergas"])
        selected = plan.select_donor([result["record"]], .959)
        self.assertEqual(selected["donor_selection_status"], plan.EXPLORATORY_DONOR)

    def test_mixed_state_or_advanced_live_rejected(self):
        bad = deepcopy(self.ev)
        bad["fr_state"]["model_state_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            self.save(bad)
        self.trainer.step()
        with self.assertRaises(ValueError):
            self.save()
        self.assertFalse((self.root / "models").exists())

    def test_bad_reload_never_updates_selector(self):
        class BadTrainer(TinyTrainer):
            def load_state_dict(self, state):
                super().load_state_dict(state)
                with torch.no_grad():
                    self.stage.weight.add_(1)
        with self.assertRaisesRegex(ValueError, "reload differs"):
            self.save(factory=lambda cfg: BadTrainer())
        self.assertFalse((self.root / "models" / "selection.json").exists())
        self.assertEqual(len(list((self.root / "models").glob("step_*.pt"))), 1)

    def test_live_factory_forbidden(self):
        with self.assertRaisesRegex(ValueError, "independent"):
            self.save(factory=lambda cfg: self.trainer)
        self.assertIsNone(A.read_selection(self.root)["best_hqnr"])

    def test_raw_prediction_reverification_and_failure(self):
        result = self.save(verify=lambda trainer: self.ev)
        self.assertTrue(result["artifact"]["predictions_reverified"])
        self.trainer.step()
        ev = evaluation(self.trainer, self.cfg, h=.970, e=2.0)
        changed = deepcopy(ev)
        changed["fr"]["per_scene"][0]["raw_prediction_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "predictions differ"):
            self.save(ev, verify=lambda trainer: changed)
        self.assertEqual(A.read_selection(self.root)["best_hqnr"]["completed_update"], 1)

    def test_rng_and_stream_not_consumed_by_factory(self):
        before = A.capture_rng()
        stream = self.stream.state_dict()
        def factory(cfg):
            random.random()
            np.random.random()
            torch.rand(10)
            return TinyTrainer()
        self.save(factory=factory)
        self.assertTrue(A._same(before, A.capture_rng()))
        self.assertTrue(A._same(stream, self.stream.state_dict()))

    def test_immutable_path_never_overwritten(self):
        result = self.save()
        path = Path(result["record"]["checkpoint_path"])
        first = path.read_bytes()
        # A new evaluation of the same model can never replace its artifact.
        improved = evaluation(self.trainer, self.cfg, h=.970, e=2.0)
        with self.assertRaisesRegex(ValueError, "different state"):
            self.save(improved)
        self.assertEqual(path.read_bytes(), first)

    def test_resume_not_donor_candidate(self):
        result = A.save_resume(self.trainer, self.stream, self.root, self.cfg,
            state_hash_fn=state_dict_sha256, reason="DEADLINE")
        self.assertTrue(result["resume_only"])
        self.assertFalse(result["donor_eligible"])
        self.assertTrue(Path(result["checkpoint_path"]).is_file())
        self.assertTrue((self.root / "models" / "latest_resume.json").is_file())
        self.assertIsNone(A.read_selection(self.root)["best_hqnr"])
        loaded = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
        self.assertTrue(A._same(loaded["trainer"], self.trainer.state_dict()))


if __name__ == "__main__":
    unittest.main()
