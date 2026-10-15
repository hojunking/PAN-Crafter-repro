"""CPU contract tests; no real campaign data, external donor, or sheet access."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from deploy.pandep_shared.project.src.pan_shared.frontend import fixed_view
from panpv.training import TrainingEngine, fixed_tensor_view, masked_l1_sum


class ToyData:
    def __init__(self, count=5, zero_first=False):
        self.items = []
        for i in range(count):
            gen = torch.Generator().manual_seed(100 + i)
            pan = torch.rand((1, 8, 8), generator=gen) * 2 - 1
            ms = torch.rand((2, 2, 2), generator=gen) * 2 - 1
            mask = torch.ones((1, 8, 8), dtype=torch.bool)
            mask[:, :i + 1, :] = False
            if zero_first and i == 0:
                mask[:] = False
            self.items.append(dict(pan=pan, ms=ms, lpan=ms[:1], gt=pan.repeat(2, 1, 1) * .3,
                                   mask=mask, raw_index=i + 50, crop_origin=[0, 0]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([.17, -.11]).reshape(1, 2, 1, 1))
        self.bias = nn.Parameter(torch.zeros((1, 2, 1, 1)))

    def forward(self, sensor, pan, ms, lpan):
        return F.interpolate(ms, scale_factor=4, mode="bicubic", align_corners=False) + self.scale * pan + self.bias


def engine(path, **kwargs):
    options = dict(method="SC_SCENE", identity={"input_sha": "test_only", "split_sha": "fixed"},
                   micro_batch=12, max_updates=4, block_updates=2, log_interval=2,
                   probe_interval=2, validation_interval=2, diagnostic_steps=(0, 2, 4))
    options.update(kwargs)
    return TrainingEngine(ToyModel(), ToyData(zero_first=True), ToyData(3), path, **options)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_view_matches_existing_numpy_recipe(self):
        array = np.arange(2 * 8 * 8).reshape(2, 8, 8).astype(np.float32)
        for rotation in range(4):
            np.testing.assert_array_equal(fixed_tensor_view(torch.from_numpy(array), rotation).numpy(),
                                          fixed_view(array, rotation))

    def test_unclipped_masked_loss_counts_bands(self):
        pred = torch.tensor([[[[3., 100.]], [[-4., 100.]]]], requires_grad=True)
        target = torch.zeros_like(pred)
        numerator, count = masked_l1_sum(pred, target, torch.tensor([[[[True, False]]]]))
        self.assertEqual(count, 2)
        self.assertEqual(numerator.item(), 7.)
        (numerator / count).backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[[[.5, 0.]], [[-.5, 0.]]]]))

    def test_provisional_diagnostic_dataset_cannot_train(self):
        with tempfile.TemporaryDirectory() as root:
            train = ToyData()
            train.provisional = True
            with self.assertRaisesRegex(ValueError, "BLOCKED_PROVISIONAL_DATA"):
                TrainingEngine(ToyModel(), train, ToyData(), root, method="SC_SCENE", identity={"test": True})

    def test_accumulation_matches_batch48_with_unequal_masks(self):
        with tempfile.TemporaryDirectory() as root:
            small = engine(Path(root) / "small", micro_batch=12)
            full = engine(Path(root) / "full", micro_batch=48)
            small.run_until(1)
            full.run_until(1)
            for name, value in small.model.state_dict().items():
                torch.testing.assert_close(value, full.model.state_dict()[name], rtol=1e-6, atol=1e-8)
            # First-step Adam can hide a wrong denominator when gradient signs
            # happen to agree, so compare actual accumulated gradients too.
            for a, b in zip(small.model.parameters(), full.model.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)
            a = json.loads((small.root / "train_steps.jsonl").read_text())
            b = json.loads((full.root / "train_steps.jsonl").read_text())
            self.assertEqual(a["virtual_indices"], b["virtual_indices"])
            self.assertEqual(a["view_ids"], b["view_ids"])
            self.assertEqual(a["valid_elements"], b["valid_elements"])
            self.assertGreater(a["repeated_within_batch"], 0)
            self.assertNotIn(0, a["virtual_indices"])

    def test_exact_resume_and_trusted_fullstate_gate(self):
        with tempfile.TemporaryDirectory() as root:
            baseline = engine(Path(root) / "baseline")
            baseline.run_until(4)
            interrupted = engine(Path(root) / "interrupted")
            interrupted.run_until(2)
            resumed = engine(Path(root) / "interrupted")
            with self.assertRaisesRegex(ValueError, "explicit resume"):
                resumed.run_until(4)
            with self.assertRaisesRegex(ValueError, "trusted_local"):
                resumed.resume()
            resumed.resume(trusted_local=True)
            resumed.run_until(4)
            for name, value in baseline.model.state_dict().items():
                torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(baseline.sampler.state_dict(), resumed.sampler.state_dict())
            self.assertEqual(baseline.train["samples_seen"], 192)
            self.assertEqual(baseline.train["best_step"], resumed.train["best_step"])
            self.assertEqual(baseline.train["full_validation_steps"], [2, 4])
            self.assertEqual(resumed.train["diagnostics_done"], [0, 2, 4])
            self.assertTrue((resumed.root / "exact_endpoint.json").exists())

    def test_sc_ft_pair_identical_sampling_not_learning_rate(self):
        with tempfile.TemporaryDirectory() as root:
            sc = engine(Path(root) / "sc")
            # Perturb unrelated global RNG; independent sampler is unaffected.
            torch.rand(100)
            np.random.rand(100)
            random.random()
            ft = engine(Path(root) / "ft", method="FT_WV3_SCENE")
            self.assertEqual(sc.sampler.peek(), ft.sampler.peek())
            self.assertEqual(sc.lr, 1e-4)
            self.assertEqual(ft.lr, 1e-5)

    def test_best_tie_prefers_earlier_full_validation_not_step_zero(self):
        with tempfile.TemporaryDirectory() as root:
            run = engine(Path(root) / "tie")
            run.validate = lambda **kwargs: dict(loss_sum=1., valid_elements=10, mean_l1=.1, patches=1)
            run.run_until(4)
            self.assertEqual(run.train["best_step"], 2)
            pointer = json.loads((run.root / "best_val_l1.json").read_text())
            self.assertEqual(pointer["completed_step"], 2)
            exact = json.loads((run.root / "exact_endpoint.json").read_text())
            self.assertEqual(exact["completed_step"], 4)

    def test_pause_only_at_completed_optimizer_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            run = engine(Path(root) / "pause")
            original = run.optimizer.step

            def pause_during_step(*args, **kwargs):
                result = original(*args, **kwargs)
                run.request_stop(15)
                return result

            run.optimizer.step = pause_during_step
            outcome = run.run_until(4)
            self.assertEqual(outcome["step"], 1)
            self.assertEqual(outcome["status"], "PAUSED_SIGNAL")
            state = json.loads((run.root / "latest.json").read_text())
            self.assertEqual(state["completed_step"], 1)
            with self.assertRaisesRegex(RuntimeError, "explicit resume"):
                run.run_until(4)
            resumed = engine(run.root)
            resumed.resume(trusted_local=True)
            self.assertEqual(resumed.train["samples_seen"], 48)

    def test_identity_change_and_corrupt_state_refused(self):
        with tempfile.TemporaryDirectory() as root:
            run = engine(Path(root) / "run")
            run.run_until(1)
            with self.assertRaisesRegex(ValueError, "BLOCKED_IDENTITY"):
                engine(run.root, micro_batch=24)
            pointer = json.loads((run.root / "latest.json").read_text())
            state = run.root / pointer["directory"] / "fullstate.pt"
            with state.open("ab") as stream:
                stream.write(b"bad")
            restored = engine(run.root)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                restored.resume(trusted_local=True)

    def test_stale_preconstructed_engine_cannot_overwrite_another_block(self):
        with tempfile.TemporaryDirectory() as root:
            first = engine(Path(root) / "run")
            stale = engine(Path(root) / "run")
            first.run_until(1)
            previous = (first.root / "latest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "changed externally"):
                stale.run_until(1)
            self.assertEqual(previous, (first.root / "latest.json").read_bytes())

    def test_pause_at_endpoint_does_not_lose_validation_or_exact_selector(self):
        with tempfile.TemporaryDirectory() as root:
            run = engine(Path(root) / "endpoint", max_updates=2)
            original = run.optimizer.step

            def pause_at_two(*args, **kwargs):
                result = original(*args, **kwargs)
                if run.train["step"] == 1:
                    run.request_stop(15)
                return result

            run.optimizer.step = pause_at_two
            paused = run.run_until(2)
            self.assertEqual(paused["step"], 2)
            self.assertEqual(paused["full_validation_steps"], [])
            self.assertFalse((run.root / "exact_endpoint.json").exists())
            resumed = engine(run.root, max_updates=2)
            resumed.resume(trusted_local=True)
            complete = resumed.run_until(2)
            self.assertEqual(complete["status"], "COMPLETE")
            self.assertEqual(complete["full_validation_steps"], [2])
            self.assertEqual(complete["best_step"], 2)
            self.assertTrue((run.root / "exact_endpoint.json").exists())

    def test_audit_cannot_select_best_and_preserves_rng(self):
        with tempfile.TemporaryDirectory() as root:
            events = []

            def diagnostic(model, step, path):
                events.append(step)
                random.random()
                np.random.rand()
                torch.rand(10)
                return dict(HQNR=1. - step * .1, ERGAS=100. - step)

            run = engine(Path(root) / "run", audit_callback=diagnostic)
            run.run_until(4)
            self.assertEqual(events, [0, 2, 4])
            selections = [json.loads(row) for row in (run.root / "selection_history.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in selections], [2, 4])
            self.assertNotIn(0, [row["step"] for row in selections])

    def test_nonfinite_update_does_not_publish_broken_state(self):
        with tempfile.TemporaryDirectory() as root:
            run = engine(Path(root) / "run")
            run.run_until(1)
            previous = (run.root / "latest.json").read_bytes()
            run.train_dataset.items[1]["gt"][0, 0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                run.run_until(2)
            self.assertEqual((run.root / "latest.json").read_bytes(), previous)
            self.assertTrue((run.root / "errors.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
