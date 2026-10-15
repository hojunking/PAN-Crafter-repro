"""CPU-only contracts for the independently trained WV3 replacement donor."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from panpv.common import file_sha
from panpv.pretrain import (RAMDataset, _immutable_json, exclusive, no_test_benchmark,
                           publish_donor, validation_callback)
from deploy.pandep_shared.project.src.pan_shared.frontend import fixed_view, normalize_dn
from deploy.pandep_shared.project.src.pan_shared.train import Trainer, evaluation_guard, learning_rate


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, sensor, pan, ms, lpan):
        assert sensor == "WV3"
        return pan.repeat(1, 8, 1, 1) * self.scale


class ToyData:
    def __init__(self, n=7):
        self.n = n

    def __len__(self):
        return self.n

    def fetch(self, ids, rotations=None):
        pan = torch.stack([torch.full((1, 8, 8), (i + 1) / 10) for i in ids])
        return dict(pan=pan, ms=pan[:, :, ::4, ::4].repeat(1, 8, 1, 1),
                    lpan=pan[:, :, ::4, ::4], gt=pan.repeat(1, 8, 1, 1) * 0.5)


def toy_trainer(root, micro_batch=12):
    return Trainer(ToyModel(), {"WV3": ToyData()},
                   dict(mode="SINGLE", sensors=["WV3"], seed=271001),
                   {"test": True}, root, microbatch=micro_batch,
                   evaluate=validation_callback(ToyData()), benchmark=no_test_benchmark)


class PretrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_micro12_and48_gradients_and_updates_match(self):
        with tempfile.TemporaryDirectory() as name:
            a = toy_trainer(Path(name) / "a", 12)
            b = toy_trainer(Path(name) / "b", 48)
            for _ in range(3):
                x, y = a.update(), b.update()
                self.assertEqual(x["batch"], y["batch"])
                torch.testing.assert_close(a.model.scale.grad, b.model.scale.grad, rtol=2e-6, atol=1e-7)
                torch.testing.assert_close(a.model.scale, b.model.scale, rtol=2e-6, atol=1e-7)

    def test_deployment_schedule_fixed50000(self):
        self.assertAlmostEqual(learning_rate(100, False), 1e-4)
        self.assertAlmostEqual(learning_rate(50000, False), 1e-5)
        self.assertLess(learning_rate(1, False), learning_rate(100, False))

    def test_exact_resume_committed_sample_stream(self):
        with tempfile.TemporaryDirectory() as name:
            a = toy_trainer(Path(name) / "a")
            a.run_until(2)
            pending = a.sampler.peek()
            b = toy_trainer(Path(name) / "a")
            self.assertEqual(b.step, 2)
            self.assertEqual(b.sampler.peek(), pending)
            self.assertEqual(b.state["scheduler"], {"completed_step": 2, "a": 1})
            a.update(); b.update()
            self.assertTrue(torch.equal(a.model.scale, b.model.scale))

    def test_validation_counts_tail_batch_and_is_not_benchmark(self):
        with tempfile.TemporaryDirectory() as name:
            trainer = toy_trainer(name)
            with evaluation_guard(trainer.model):
                result = validation_callback(ToyData(19))(trainer, "validation")
            self.assertTrue(result["complete"])
            self.assertEqual(result["sensors"]["WV3"]["n_samples"], 19)
            self.assertAlmostEqual(result["sensors"]["WV3"]["mean_l1"], 0.3, places=6)
            self.assertIsNone(result["sensors"]["WV3"]["HQNR"])
            self.assertEqual(no_test_benchmark(trainer, {}, "EXACT")["status"], "NOT_EVALUATED_BY_DESIGN")

    def test_early_or_pending_checkpoint_cannot_be_donor(self):
        with tempfile.TemporaryDirectory() as name:
            trainer = toy_trainer(name)
            with self.assertRaisesRegex(ValueError, "EXACT50000"):
                publish_donor(trainer, name, {}, {})
            trainer.state["step"] = 50000
            trainer.state["pending_events"] = ["exact"]
            with self.assertRaisesRegex(ValueError, "EXACT50000"):
                publish_donor(trainer, name, {}, {})

    def test_final_update_stop_does_not_publish_or_auto_chain(self):
        import signal
        from contextlib import ExitStack
        from unittest.mock import Mock, patch
        from panpv.pretrain import run
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            repo = Path(name)
            trainer = Mock(step=50000)
            def finish_with_stop(*args, **kwargs):
                signal.raise_signal(signal.SIGTERM)
                return "BLOCK_COMPLETE"
            trainer.run_until.side_effect = finish_with_stop
            stack.enter_context(patch("panpv.pretrain.prepare_sources", return_value={"dataset_sha256": "a"*64}))
            stack.enter_context(patch("panpv.pretrain.RAMDataset"))
            stack.enter_context(patch("panpv.pretrain.launch_smoke"))
            stack.enter_context(patch("panpv.pretrain.SingleSensorPLHUNet"))
            stack.enter_context(patch("panpv.pretrain.Trainer", return_value=trainer))
            publish = stack.enter_context(patch("panpv.pretrain.publish_donor"))
            result = run(repo / "work_dir" / "donor", repo, device="cpu")
            self.assertEqual(result, {"status": "PAUSED_SAFE", "completed_step": 50000})
            publish.assert_not_called()

    def test_identity_and_lock_fail_closed(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "identity.json"
            _immutable_json(p, {"sha": "a"})
            _immutable_json(p, {"sha": "a"})
            with self.assertRaisesRegex(ValueError, "identity changed"):
                _immutable_json(p, {"sha": "b"})
            with exclusive(name):
                with self.assertRaisesRegex(RuntimeError, "Another donor"):
                    with exclusive(name):
                        pass

    def test_ram_normalizes_once_and_shares_all_augment_views(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, lp = root / "source.h5", root / "lp.h5"
            raw = {"pan": np.arange(2*64*64, dtype=np.float32).reshape(2,1,64,64) % 2048,
                   "ms": np.ones((2,8,16,16), dtype=np.float32)*300,
                   "gt": np.ones((2,8,64,64), dtype=np.float32)*900}
            with h5py.File(source, "w") as handle:
                for key, value in raw.items(): handle[key] = value
            with h5py.File(lp, "w") as handle:
                handle["lpan"] = np.ones((2,1,16,16), dtype=np.float32)*200
            manifest = {"sensors": {"WV3": {"splits": {"train": dict(
                dataroot=str(source), lpan_path=str(lp), sha256=file_sha(source),
                lpan_sha256=file_sha(lp), count=2)}}}}
            data = RAMDataset(manifest, "train")
            batch = data.fetch([1, 0, 1], [3, 1, 3])
            np.testing.assert_array_equal(batch["pan"][0].numpy(),
                normalize_dn(fixed_view(raw["pan"][1], 3), "WV3"))
            self.assertTrue(torch.equal(batch["pan"][0], batch["pan"][2]))
            self.assertFalse(data.arrays["pan"].flags.writeable)
            with self.assertRaises(IndexError): data.fetch([True])
            with self.assertRaises(ValueError): data.fetch([0], [4])
            with h5py.File(source, "r+") as handle: handle["pan"][0,0,0,0] = 1
            # Some filesystems coalesce metadata timestamps for rapid writes.
            # This test checks the metadata-change guard, not a per-batch full
            # H5 checksum: make the simulated source modification observable.
            import os
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, data._stats[0][1] + 1_000_000_000))
            with self.assertRaisesRegex(ValueError, "identity changed"): data.fetch([0])


if __name__ == "__main__":
    unittest.main()
