"""Filesystem-backed orchestration tests; no GPU, dataset, network or Docker."""
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from ta2 import controller, training
from ta2.common import atomic_json, load_json


class TinyModel(nn.Module):
    def __init__(self, bands, seed, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.]))

    def initial_manifest(self):
        return dict(initial_weight=0.)


class TinyEngine:
    updates = []

    def __init__(self, model, bundle, cfg, device):
        self.model, self.cfg, self.completed = model, cfg, 0
        self.native_stream_sha256 = "identical-native-stream"

    def state_dict(self):
        return dict(model=self.model.state_dict(), completed=self.completed)

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.completed = state["completed"]

    def update(self):
        self.completed += 1
        with torch.no_grad():
            self.model.weight.add_(1)
        self.updates.append(self.completed)
        return dict(rec=1., epsilon=0., struct=0., completed_step=self.completed)


class TinyLedger:
    """Durable stub confines this test to orchestration (real ledger tested separately)."""
    def __init__(self, path, identity, total):
        self.path, self.total = Path(path), total
        self.path.mkdir(parents=True, exist_ok=True)

    def register(self, report):
        atomic_json(self.path/f"step_{report['completed_step']}.json", report)

    def status(self):
        count = len(list(self.path.glob("step_*.json")))
        return dict(status="HQNR_SELECTION_COMPLETE" if count == self.total else "HQNR_SELECTION_PENDING",
                    completed_step=self.total, candidates_complete=count)


@contextlib.contextmanager
def tiny_runner(evaluate):
    TinyEngine.updates = []
    with contextlib.ExitStack() as stack:
        for name, value in (("TeacherModel", TinyModel), ("UpdateEngine", TinyEngine),
                            ("EvaluationLedger", TinyLedger), ("hqnr_grid", lambda total: tuple(range(1, total+1))),
                            ("evaluate_checkpoint", evaluate), ("disk_guard", lambda directory: None)):
            stack.enter_context(patch.object(training, name, value))
        stack.enter_context(patch("ta2.probes.lightweight_gradient", return_value={"synthetic_test": True}))
        yield


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = dict(seed=101, bands=4, total_updates=3, dataset="QB", run_id="TEST_ONLY")
        self.identity = dict(test_only=True, source="pinned")
        self.bundle = type("Bundle", (), {"datasets": {}})()

    def evaluate(self, model, datasets, path, **kwargs):
        step = kwargs["completed_step"]
        self.assertEqual(model.weight.item(), step)
        return dict(completed_step=step, model_weight=model.weight.item(), context=self.identity)

    def test_evaluation_failure_recovers_without_retraining_completed_steps(self):
        failed = []
        def evaluate(*args, **kwargs):
            if kwargs["completed_step"] == 2 and not failed:
                failed.append(True)
                raise RuntimeError("injected evaluator failure")
            return self.evaluate(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder, tiny_runner(evaluate):
            path = Path(folder)/"run"
            with self.assertRaisesRegex(RuntimeError, "evaluator failure"):
                training.run_training(self.bundle, self.cfg, path, self.identity, "cpu")
            saved = torch.load(path/"latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["state"]["completed"], 2)
            self.assertEqual(TinyEngine.updates, [1, 2])
            result = training.run_training(self.bundle, self.cfg, path, self.identity, "cpu",
                                           finalize_hook=lambda *args: {"status": "COMPLETE"})
            self.assertEqual(TinyEngine.updates, [1, 2, 3])
            self.assertEqual(result["status"], "COMPLETE")

    def test_final_diagnostic_failure_recovers_without_any_optimizer_update(self):
        with tempfile.TemporaryDirectory() as folder, tiny_runner(self.evaluate):
            path = Path(folder)/"run"
            def fail(*args):
                raise OSError("injected diagnostic persistence failure")
            with self.assertRaisesRegex(OSError, "diagnostic"):
                training.run_training(self.bundle, self.cfg, path, self.identity, "cpu", finalize_hook=fail)
            self.assertEqual(TinyEngine.updates, [1, 2, 3])
            result = training.run_training(self.bundle, self.cfg, path, self.identity, "cpu",
                                           finalize_hook=lambda *args: {"status": "COMPLETE"})
            self.assertEqual(TinyEngine.updates, [1, 2, 3])
            self.assertEqual(result["status"], "COMPLETE")

    def test_missing_historical_weights_never_fabricated_from_latest(self):
        with tempfile.TemporaryDirectory() as folder, tiny_runner(self.evaluate):
            path = Path(folder)/"run"
            training.run_training(self.bundle, self.cfg, path, self.identity, "cpu")
            (path/"checkpoints/step_000001.pt").unlink()
            with self.assertRaisesRegex(ValueError, "Lost historical candidate"):
                training.run_training(self.bundle, self.cfg, path, self.identity, "cpu")
            self.assertEqual(TinyEngine.updates, [1, 2, 3])

    def test_changed_identity_blocks_exact_resume(self):
        with tempfile.TemporaryDirectory() as folder, tiny_runner(self.evaluate):
            path = Path(folder)/"run"
            training.run_training(self.bundle, self.cfg, path, self.identity, "cpu")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                training.run_training(self.bundle, self.cfg, path, {"source": "other"}, "cpu")


class ControllerTests(unittest.TestCase):
    def test_no_formal_cpu_run(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            controller.main(["run", "--server", "s1", "--device", "cpu"])

    def test_one_controller_per_exact_local_lane(self):
        with tempfile.TemporaryDirectory() as folder:
            lane = Path(folder)/"s1"
            with controller.lane_lock(lane):
                with self.assertRaisesRegex(RuntimeError, "Another TA2 controller"):
                    with controller.lane_lock(lane):
                        self.fail("second controller acquired the lane")

    def test_warp_smoke_requires_real_parameter_update(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "At least3"):
                controller.smoke(None, "s1", Path(folder), None, {}, "cpu", updates=2)


if __name__ == "__main__":
    unittest.main()
