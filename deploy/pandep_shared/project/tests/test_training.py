import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from pan_shared.checkpoints import CheckpointStore, capture_rng, restore_rng, file_sha
from pan_shared.common import atomic_json
from pan_shared.controller import Controller
from pan_shared.losses import mean_l1
from pan_shared.registry import build_registry
from pan_shared.sampling import BalancedSampler
from pan_shared.train import Trainer, evaluation_guard, events_at, learning_rate


class TinyModel(nn.Module):
    def __init__(self, sensors=("WV3", "GF2", "QB")):
        super().__init__()
        self.sensors = sensors
        self.stems = nn.ModuleDict({s: nn.Conv2d(1, 2, 1) for s in sensors})
        self.trunk = nn.Conv2d(2, 2, 1)
        self.heads = nn.ModuleDict({s: nn.Conv2d(2, 1, 1) for s in sensors})

    def forward(self, sensor, pan, ms, lpan):
        return ms + self.heads[sensor](torch.tanh(self.trunk(self.stems[sensor](pan))))


class TinyDataset:
    def __len__(self):
        return 17

    def fetch(self, sample_ids, rotations=None):
        x = (torch.tensor(sample_ids).float() / 17).reshape(-1, 1, 1, 1).expand(-1, 1, 2, 2)
        r = torch.tensor(rotations or [0] * len(sample_ids)).reshape(-1, 1, 1, 1) / 10
        return dict(pan=x + r, ms=x * 0.7, lpan=x * 0.5, gt=x * 1.2)


def make_trainer(path, **kwargs):
    torch.manual_seed(123)
    model = TinyModel()
    config = dict(mode="SHARED", sensors=list(model.sensors), seed=271001,
                  effective_batch_size=48, microbatch=24, train_fraction=1.0)
    return Trainer(model, {s: TinyDataset() for s in model.sensors}, config,
                   {"run_id": "fixture", "config_sha256": "a" * 64}, path, **kwargs)


def assert_nested(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        test.assertTrue(np.array_equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for k in a:
            assert_nested(test, a[k], b[k])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_nested(test, x, y)
    else:
        test.assertEqual(a, b)


class SamplingTests(unittest.TestCase):
    def make(self, sensors=("WV3", "GF2", "QB")):
        return BalancedSampler(271001, {s: list(range(17)) for s in sensors}, sensors)

    def test_triplets_exactly_balanced(self):
        sampler = self.make()
        for _ in range(20):
            sensors = []
            for _ in range(3):
                batch = sampler.peek(); sensors.append(batch["sensor"]); sampler.commit(batch["token"])
            self.assertEqual(set(sensors), {"WV3", "GF2", "QB"})

    def test_peek_has_no_committed_effect(self):
        sampler = self.make(); original = sampler.state_dict()
        batch = sampler.peek()
        self.assertEqual(batch, sampler.peek())
        self.assertEqual(original, sampler.state_dict())
        self.assertEqual(sum(v["committed_samples"] for v in sampler.exposures().values()), 0)
        with self.assertRaises(ValueError):
            sampler.commit("invalid")

    def test_split_resume_mid_triplet_and_epoch(self):
        sampler = self.make()
        for _ in range(5):
            sampler.commit(sampler.peek()["token"])
        restored = self.make(); restored.load_state_dict(sampler.state_dict())
        for _ in range(20):
            self.assertEqual(sampler.peek(), restored.peek())
            sampler.commit(sampler.peek()["token"]); restored.commit(restored.peek()["token"])
        self.assertEqual(sampler.state_dict(), restored.state_dict())

    def test_single_and_shared_stream_correspond(self):
        shared, single = self.make(), self.make(("WV3",))
        for _ in range(30):
            batch = shared.peek(); shared.commit(batch["token"])
            if batch["sensor"] == "WV3":
                control = single.peek(); single.commit(control["token"])
                for key in ("sample_ids", "rot_ids"):
                    self.assertEqual(batch[key], control[key])

    def test_tail_is_carried_not_discarded(self):
        sampler = self.make(("WV3",)); batch = sampler.peek(); sampler.commit(batch["token"])
        counts = sampler.state_dict()["streams"]["WV3"]["counts"]
        self.assertEqual(sum(counts), 48)
        self.assertTrue(all(c in (2, 3) for c in counts))
        self.assertAlmostEqual(sampler.exposures()["WV3"]["effective_epochs"], 48 / 17)

    def test_subset_identity_rejects_resume(self):
        first = self.make(("WV3",))
        other = BalancedSampler(271001, {"WV3": list(range(8))})
        with self.assertRaisesRegex(ValueError, "BLOCKED_IDENTITY"):
            other.load_state_dict(first.state_dict())


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_loss_band_count_normalization_and_no_clamp(self):
        for bands in (4, 8):
            pred = torch.full((2, bands, 3, 3), 4.0)
            self.assertEqual(mean_l1(pred, torch.zeros_like(pred)).item(), 4.0)

    def test_lr_exact_endpoints_indefinite_floor(self):
        self.assertEqual(learning_rate(100, False), 1e-4)
        self.assertEqual(learning_rate(300, True), 1e-4)
        self.assertAlmostEqual(learning_rate(1, False), 1.99e-6)
        self.assertEqual(learning_rate(50000, False), 1e-5)
        self.assertEqual(learning_rate(150000, True), 1e-5)
        self.assertEqual(learning_rate(3000000, True), 1e-5)

    def test_cadence_full_validation_once_and_shared75k(self):
        self.assertEqual(events_at(50000, True), ["probe", "validation", "exact"])
        self.assertEqual(events_at(75000, True), ["validation", "exact"])
        self.assertEqual(events_at(75000, False), ["validation"])
        self.assertEqual(events_at(2000, True), ["probe"])
        self.assertEqual(events_at(0, True), [])

    def test_exact_resume_parameters_optimizer_rng_next_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            first = make_trainer(Path(directory) / "continuous")
            for _ in range(11):
                first.update()
            reference = copy.deepcopy(first.full_state())
            split = make_trainer(Path(directory) / "split")
            for _ in range(5):
                split.update()
            split.save()
            restored = make_trainer(Path(directory) / "split")
            for _ in range(6):
                restored.update()
            actual = restored.full_state()
            for field in ("model", "optimizer", "sampler", "rng"):
                assert_nested(self, reference[field], actual[field])
            self.assertEqual(first.sampler.peek(), restored.sampler.peek())
            self.assertEqual(first.state["scheduler"], restored.state["scheduler"])

    def test_inactive_parameters_and_moments_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            for _ in range(3):
                trainer.update()
            active = trainer.sampler.peek()["sensor"]
            inactive = {name: parameter for name, parameter in trainer.model.named_parameters()
                        if name.startswith(("stems.", "heads.")) and name.split(".")[1] != active}
            weights = {name: p.clone() for name, p in inactive.items()}
            moments = {name: copy.deepcopy(trainer.optimizer.state[p]) for name, p in inactive.items()}
            trainer.update()
            for name, parameter in inactive.items():
                self.assertIsNone(parameter.grad)
                self.assertTrue(torch.equal(weights[name], parameter))
                assert_nested(self, moments[name], trainer.optimizer.state[parameter])

    def test_nonfinite_post_optimizer_state_never_commits_or_publishes(self):
        for corruption in ("active_parameter", "inactive_parameter", "exp_avg", "exp_avg_sq", "step"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                trainer = make_trainer(directory)
                trainer.save()
                pointer = Path(directory) / "resume/last.json"
                original_pointer = pointer.read_bytes()
                original_sampler = trainer.sampler.state_dict()
                original_parameters = copy.deepcopy(trainer.model.state_dict())
                actual_step = trainer.optimizer.step
                def poisoned_step(*args, **kwargs):
                    result = actual_step(*args, **kwargs)
                    active = next(p for p in trainer.model.parameters() if p.grad is not None)
                    with torch.no_grad():
                        if corruption == "active_parameter":
                            active.reshape(-1)[0] = float("nan")
                        elif corruption == "inactive_parameter":
                            inactive = next(p for p in trainer.model.parameters() if p.grad is None)
                            inactive.reshape(-1)[0] = float("inf")
                        else:
                            trainer.optimizer.state[active][corruption].reshape(-1)[0] = float("inf")
                    return result
                with mock.patch.object(trainer.optimizer, "step", side_effect=poisoned_step):
                    with self.assertRaisesRegex(FloatingPointError, "post-optimizer"):
                        trainer.run_until(1)
                self.assertEqual(trainer.step, 0)
                self.assertTrue(trainer._in_update)
                self.assertEqual(trainer.state["status"], "DIVERGED")
                self.assertEqual(trainer.sampler.state_dict(), original_sampler)
                self.assertEqual(pointer.read_bytes(), original_pointer)
                with self.assertRaisesRegex(RuntimeError, "unsafe fullstate"):
                    trainer.save()
                restored = make_trainer(directory)
                self.assertEqual(restored.step, 0)
                assert_nested(self, restored.model.state_dict(), original_parameters)
                self.assertEqual(restored.sampler.state_dict(), original_sampler)

    def test_evaluation_restores_all_rng_and_mixed_modes(self):
        model = TinyModel(); model.train(); model.trunk.eval()
        before = capture_rng()
        modes = [m.training for m in model.modules()]
        with evaluation_guard(model):
            random.random(); np.random.rand(); torch.rand(2)
            self.assertTrue(all(not m.training for m in model.modules()))
        assert_nested(self, before, capture_rng())
        self.assertEqual(modes, [m.training for m in model.modules()])

    def test_macro_validation_and_earlier_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory); trainer.update()
            result = {"sensors": {s: {"mean_l1": v, "n_samples": n} for s, v, n in
                       (("WV3", .1, 1080), ("GF2", .2, 2201), ("QB", .3, 1905))}}
            trainer._select_best(result)
            self.assertAlmostEqual(trainer.state["best"]["score"], .2)
            trainer.update(); trainer._select_best(result)
            self.assertEqual(trainer.state["best"]["completed_step"], 1)
            self.assertEqual(trainer.state["candidate_count"], 2)

    def test_pause_stores_state_at_optimizer_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            status = trainer.run_until(9, lambda: "PAUSE" if trainer.step == 2 else "RUN")
            self.assertEqual(status, "PAUSED_SAFE")
            restored = make_trainer(directory)
            self.assertEqual(restored.step, 2)
            self.assertEqual(restored.sampler.peek(), trainer.sampler.peek())

    def test_weight_byte_and_state_hash_separate_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory); trainer.update()
            identity = trainer.store.save_weights(trainer.model, trainer.step, trainer.sampler.exposures())
            self.assertEqual(identity["checkpoint_sha256"], file_sha(identity["model_path"]))
            self.assertNotEqual(identity["tensor_state_sha256"], identity["checkpoint_sha256"])
            trainer.store.load_weights(trainer.model, identity)

    def test_corrupt_latest_resume_fallback_and_safe_republication(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory); trainer.update(); trainer.save()
            trainer.update(); trainer.save()
            pointer = json.loads((Path(directory) / "resume/last.json").read_text())
            self.assertEqual(pointer["active"], "B")
            path = Path(directory) / "resume/last_B.pt"
            path.write_bytes(b"interrupted-write")
            good = file_sha(Path(directory) / "resume/last_A.pt")
            restored = make_trainer(directory)
            self.assertEqual(restored.step, 1)
            restored.update()
            with mock.patch("pan_shared.checkpoints._json", side_effect=OSError("power loss before pointer")):
                with self.assertRaises(OSError):
                    restored.save()
            self.assertEqual(good, file_sha(Path(directory) / "resume/last_A.pt"))
            again = make_trainer(directory)
            self.assertEqual(again.step, 1)

    def test_resume_changed_identity_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory); trainer.save()
            store = CheckpointStore(directory, {"run_id": "different"})
            with self.assertRaisesRegex(ValueError, "BLOCKED_IDENTITY"):
                store.load_resume()

    def test_checkpoint_child_symlink_cannot_escape_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "run").mkdir(); (root / "old").mkdir()
            (root / "run/resume").symlink_to(root / "old", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "BLOCKED_PATH"):
                CheckpointStore(root / "run", {"run_id": "fixture"})


class ControllerTests(unittest.TestCase):
    def test_failed_endpoint_evaluation_retries_before_queue_advance(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = Controller(directory, build_registry(), None, None)
            pending = controller.next_task()
            controller.state["current_task"] = pending
            controller.state["progress"][pending["run_id"]] = pending["target_step"]
            self.assertEqual(controller.next_task(), pending)
            controller.state["current_task"] = None
            self.assertNotEqual(controller.next_task()["run_id"], pending["run_id"])

    def test_initial_priority_and_extension_fairness(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = build_registry()
            controller = Controller(directory, registry, None, None)
            first = []
            for _ in range(36):
                task = controller.next_task()
                first.append((controller.runs[task["run_id"]]["case_id"], task["target_step"]))
                controller.state["progress"][task["run_id"]] = task["target_step"]
            self.assertEqual(first[:6], [("C00", 50000), ("S01", 50000), ("S02", 50000),
                                        ("S03", 50000), ("C00", 100000), ("C00", 150000)])
            self.assertEqual(controller.completed_stage(), 1)
            seen = []
            for _ in range(18):
                task = controller.next_task(); seen.append(task["run_id"])
                controller.state["progress"][task["run_id"]] = task["target_step"]
                controller.state["extension_cursor"] = task["extension_cursor_after"]
            self.assertEqual(len(set(seen)), 18)

    def test_operator_stop_not_resurrected(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = build_registry(); root = Path(directory)
            atomic_json(root / "control.json", {"campaign_id": registry["campaign_id"], "action": "STOP"})
            controller = Controller(root, registry, None, None)
            controller.run(until_operator_stop=True)
            self.assertEqual(controller.state["status"], "STOPPED_BY_OPERATOR")
            atomic_json(root / "control.json", {"campaign_id": registry["campaign_id"], "action": "RUN"})
            with self.assertRaisesRegex(RuntimeError, "explicit resume"):
                Controller(root, registry, None, None).run(until_operator_stop=True)

    def test_indefinite_flag_required(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                Controller(directory, build_registry(), None, None).run()


if __name__ == "__main__":
    unittest.main()
