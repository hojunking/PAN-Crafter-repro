#!/usr/bin/env python3
"""CPU-only FH12 sampler/resume and real training-loop integration tests.

The registered recipe, datasets, backbone and evaluator are patched ONLY inside
these synthetic tests. No production YAML, data, checkpoint or GPU is modified.
"""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import fh12.training as training
from fh12.common import object_sha, sha256
from fh12.plan import build_config, teacher_for, cases_for
from fh12.postrun import select_records


class TinyData(torch.utils.data.Dataset):
    has_gt = True
    def __init__(self):
        self.views = []
    def __len__(self):
        return 8
    def __getitem__(self, index):
        i, rot = index if isinstance(index, tuple) else (index, 0)
        self.views.append((i, rot))
        gt = torch.full((8, 16, 16), .07 * i - .2)
        ms = torch.full((8, 4, 4), .02 * i)
        pan = torch.linspace(-.3, .3, 256).reshape(1, 16, 16) + .01 * rot
        lp = F.avg_pool2d(pan, 4)
        return gt, gt.clone(), ms, lp, pan, torch.tensor([i, rot, 1, 1])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Conv2d(9, 8, 1)
        self.aligner = torch.nn.Linear(9, 2)
        self.dropout = torch.nn.Dropout(.2)
    def predict_delta(self, pan, ms_base):
        return self.aligner(torch.cat((pan, ms_base), 1).mean((2, 3)))
    def forward(self, pan, ms, lpan):
        base = F.interpolate(ms, scale_factor=4, mode="bicubic", align_corners=False)
        delta = self.predict_delta(pan, base)
        x = torch.cat((pan + delta.sum(1)[:, None, None, None] * .01, base), 1)
        y = self.backbone(self.dropout(x)) + base
        return {"y": y, "ms_base": base, "delta": delta}


def fake_build(*args, **kwargs):
    model = TinyModel()
    if kwargs.get("teacher_aligner_state") is not None:
        model.aligner.load_state_dict(kwargs["teacher_aligner_state"])
    return model, {"test_only": True}


def fake_result():
    return dict(fr={"hqnr": .96, "d_lambda": .02, "d_s": .02},
                rr={"scc": .98, "ergas": 2.02}, val_ergas=2.03, seconds=0.,
                shift={"fr_mean": [0., 0.], "fr_max_abs": 0.,
                       "rr_mean": [0., 0.], "rr_max_abs": 0.})


class SyntheticRun:
    def __init__(self, root, case=None):
        self.root = root
        self.case = case or teacher_for("s1")
        self.cfg = build_config(self.case)
        self.cfg.update(batch_size=2, num_worker=0, num_iter=3, num_warmup=0, log_iter=1000)
        self.path = root / "config.yaml"
        self.path.write_text(yaml.safe_dump(self.cfg))
        self.wd = root / self.cfg["work_dir"]
        data = {"splits": {}}
        for split, key in (("train", "train_feeder_args"), ("val", "val_feeder_args"),
                           ("rr", "test_reduced_feeder_args"), ("fr", "test_full_feeder_args")):
            source = root / self.cfg[key]["dataroot"]
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"synthetic identity only")
            data["splits"][split] = {"sha256": sha256(source)}
        dest = root / "work_dir/_fh12" / self.case.server_id / "dataset_manifest.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data))
        self.data = TinyData()

    def run(self, evaluator, resume=False):
        with patch.object(training, "validate_config", return_value=self.case), \
             patch.object(training, "runtime_context", return_value=({}, "2099-01-01T00:00:00+00:00")), \
             patch.object(training, "source_identity", return_value={"test_only": True}), \
             patch.object(training, "build_dataset", side_effect=lambda *a, **kw: self.data), \
             patch.object(training, "build_model", side_effect=fake_build), \
             patch.object(training, "FRMetrics", return_value=object()), \
             patch.object(training, "evaluate_model", side_effect=evaluator), \
             patch.object(training, "GRID_STEPS", (1, 2, 3)):
            return training.train_run(self.path, device="cpu", resume=resume, root=self.root)

    def state(self):
        return torch.load(self.wd / "last/training_state.pt", map_location="cpu", weights_only=False)


class TrainingTests(unittest.TestCase):
    @staticmethod
    def records():
        return [dict(update=step, checkpoint_identity={"update": step, "model_sha256": str(step)},
                     rr=dict(official_complete=True, ergas=e, scc=.98, psnr=37., sam=2.8, q8=.91, ssim=.97),
                     fr=dict(hqnr=h, d_lambda=.02, d_s=.02), val_ergas=v)
                for step, h, e, v in ((1010, .961, 2.06, 2.03), (2020, .959, 2.03, 2.04), (50000, .958, 2.00, 2.02))]

    def test_selection_preserves_distinct_same_checkpoint_outputs(self):
        selected = select_records(self.records(), expected_steps=(1010, 2020, 50000))
        self.assertEqual(selected["raw_max"]["update"], 1010)
        self.assertEqual(selected["target"]["update"], 2020)
        self.assertEqual(selected["exact50k"]["update"], 50000)
        self.assertEqual(selected["rr_val_selected"]["update"], 50000)
        self.assertEqual(selected["e_min_diag"]["update"], 50000)
        self.assertEqual(selected["n_eligible"], 2)

    def test_selection_no_eligible_has_no_fallback_target(self):
        selected = select_records(self.records(), threshold=.999, expected_steps=(1010, 2020, 50000))
        self.assertIsNone(selected["target"])
        self.assertEqual(selected["target_status"], "no_eligible")
        self.assertEqual(selected["raw_max"]["update"], 1010)

    def test_selection_requires_complete_official_unique_grid(self):
        records = self.records()
        for broken in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                select_records(broken, expected_steps=(1010, 2020, 50000))
        records[1]["rr"]["official_complete"] = False
        with self.assertRaises(ValueError):
            select_records(records, expected_steps=(1010, 2020, 50000))

    def test_selection_target_tie_order_scc_psnr_step(self):
        records = self.records()
        for row in records:
            row["rr"]["ergas"] = 2.03
            row["fr"]["hqnr"] = .959
        records[1]["rr"]["scc"] = .99
        self.assertEqual(select_records(records, expected_steps=(1010, 2020, 50000))["target"]["update"], 2020)
        records[2]["rr"]["scc"] = .99
        records[2]["rr"]["psnr"] = 38.
        self.assertEqual(select_records(records, expected_steps=(1010, 2020, 50000))["target"]["update"], 50000)
        records[1]["rr"]["psnr"] = 38.
        self.assertEqual(select_records(records, expected_steps=(1010, 2020, 50000))["target"]["update"], 2020)

    def test_sampler_independent_of_global_rng_and_model_construction(self):
        a = training.BatchStream(19, 3, 72001)
        torch.manual_seed(1); torch.rand(99); random.seed(200)
        b = training.BatchStream(19, 3, 72001)
        self.assertEqual(a.remaining_batches(), b.remaining_batches())
        self.assertEqual(len(set(i for batch in a.remaining_batches() for i, _ in batch)), 18)
        self.assertTrue(all(0 <= rot <= 3 for batch in a.remaining_batches() for _, rot in batch))

    def test_sampler_resume_mid_epoch_and_next_epoch(self):
        a = training.BatchStream(19, 3, 72001)
        a.cursor = 2
        saved = copy.deepcopy(a.state_dict())
        b = training.BatchStream(19, 3, 999)
        b.load_state_dict(saved)
        self.assertEqual(a.remaining_batches(), b.remaining_batches())
        a.new_epoch(); b.new_epoch()
        self.assertEqual(a.remaining_batches(), b.remaining_batches())

    def test_sampler_rejects_corrupt_rotations(self):
        source = training.BatchStream(19, 3, 2)
        for replacement in (torch.zeros(1, dtype=torch.long), torch.full((18,), 7)):
            state = copy.deepcopy(source.state_dict()); state["rotations"] = replacement
            with self.assertRaises(ValueError):
                training.BatchStream(19, 3, 2).load_state_dict(state)

    def test_sampler_rejects_duplicate_and_out_of_range_order(self):
        source = training.BatchStream(19, 3, 2)
        for replacement in (torch.zeros(18, dtype=torch.long), torch.arange(1, 19) + 1):
            state = copy.deepcopy(source.state_dict()); state["order"] = replacement
            with self.assertRaises(ValueError):
                training.BatchStream(19, 3, 2).load_state_dict(state)

    def test_python_numpy_torch_rng_roundtrip(self):
        random.seed(10); np.random.seed(11); torch.manual_seed(12)
        saved = training.rng_state()
        expected = (random.random(), np.random.rand(5), torch.rand(5))
        training.restore_rng(saved)
        actual = (random.random(), np.random.rand(5), torch.rand(5))
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_canonical_config_hash_is_format_independent(self):
        cfg = build_config(teacher_for("s1"))
        self.assertEqual(object_sha(cfg), object_sha(yaml.safe_load(yaml.safe_dump(cfg, sort_keys=True))))

    def test_resume_bundle_publication_is_atomic_across_identity_failure(self):
        with tempfile.TemporaryDirectory(prefix="fh12-atomic-") as temp:
            root=Path(temp)
            training.save_resume_bundle(root,{"update":1},{"update":1})
            before=(root/"last").resolve()
            with patch.object(training,"atomic_json",side_effect=OSError("simulated interrupted identity")):
                with self.assertRaises(OSError):
                    training.save_resume_bundle(root,{"update":2},{"update":2})
            self.assertEqual((root/"last").resolve(),before)
            self.assertEqual(torch.load(root/"last/training_state.pt",weights_only=False)["update"],1)
            training.save_resume_bundle(root,{"update":2},{"update":2})
            self.assertFalse(before.exists())
            identity=json.loads((root/"last/identity.json").read_text())
            self.assertEqual(identity["training_state_sha256"],sha256(root/"last/training_state.pt"))

    def test_interrupted_candidate_write_does_not_publish_partial_candidate(self):
        with tempfile.TemporaryDirectory(prefix="fh12-candidate-") as temp:
            run=SyntheticRun(Path(temp))
            with patch.object(training,"save_file",side_effect=OSError("synthetic partial write")):
                with self.assertRaises(OSError):
                    run.run(lambda *a,**k:fake_result())
            self.assertFalse((run.wd/"candidates/1").exists())
            self.assertEqual(run.state()["update"],1)
            self.assertEqual(run.run(lambda *a,**k:fake_result(),resume=True),0)
            self.assertTrue((run.wd/"candidates/1/identity.json").is_file())

    def test_trainloop_interrupted_evaluation_resumes_same_samples_and_weights(self):
        # Stochastic tiny backbone deliberately tests the promised RNG restore,
        # even though the registered FH12 backbone currently has dropout=0.
        def success(*args, **kwargs):
            torch.rand(3)  # Evaluator DataLoader creation consumes torch RNG.
            return fake_result()
        with tempfile.TemporaryDirectory(prefix="fh12-complete-") as one, \
             tempfile.TemporaryDirectory(prefix="fh12-resume-") as two:
            uninterrupted = SyntheticRun(Path(one))
            self.assertEqual(uninterrupted.run(success), 0)
            expected = uninterrupted.state()
            interrupted = SyntheticRun(Path(two))
            calls = [0]
            def timeout(*args, **kwargs):
                calls[0] += 1
                torch.rand(3)
                if calls[0] == 2:
                    raise TimeoutError("synthetic evaluation deadline")
                return fake_result()
            self.assertEqual(interrupted.run(timeout), 75)
            paused = interrupted.state()
            self.assertEqual(paused["update"], 2)
            status = json.loads((interrupted.wd / "meta/training_status.json").read_text())
            self.assertEqual(status["status"], "PAUSED_DEADLINE")
            self.assertFalse(status["training_complete"])
            self.assertTrue((interrupted.wd / "candidates/2/model.safetensors").exists())
            self.assertEqual(interrupted.run(success, resume=True), 0)
            actual = interrupted.state()
            self.assertEqual(expected["update"], actual["update"])
            self.assertEqual(uninterrupted.data.views, interrupted.data.views)
            for key, value in expected["model_state"].items():
                torch.testing.assert_close(value, actual["model_state"][key], rtol=0, atol=0, msg=key)
            grid = json.loads((interrupted.wd / "official/raw_grid.json").read_text())
            self.assertEqual([r["update"] for r in grid["records"]], [1, 2, 3])
            self.assertTrue(grid["complete"])
            for step in (1, 2, 3):
                identity = json.loads((interrupted.wd / f"candidates/{step}/identity.json").read_text())
                self.assertEqual(identity["model_sha256"], sha256(interrupted.wd / f"candidates/{step}/model.safetensors"))

    def test_resume_rejects_corrupted_full_state_before_loading(self):
        def timeout(*args, **kwargs):
            raise TimeoutError("synthetic pause")
        with tempfile.TemporaryDirectory(prefix="fh12-corrupt-") as temp:
            run = SyntheticRun(Path(temp))
            self.assertEqual(run.run(timeout), 75)
            state_path = run.wd / "last/training_state.pt"
            state = run.state(); state["training_seconds"] += 1000
            torch.save(state, state_path)
            with self.assertRaisesRegex(ValueError, "(hash|SHA|identity|integrity|bytes|checksum)"):
                run.run(lambda *a, **k: fake_result(), resume=True)

    def test_student_bridge_keeps_teacher_frozen_and_records_reference_sha(self):
        # s4 Teacher PLH -> Student P0; no direct Student c is fed to Teacher.
        case = next(c for c in cases_for("s4") if c.role == "S" and c.input_layout == "P0")
        teacher = TinyModel()
        before = {k: v.clone() for k, v in teacher.state_dict().items()}
        reference = {"q_ref": .46875, "tau_R": .125, "teacher_checkpoint_sha256": "a" * 64}
        with tempfile.TemporaryDirectory(prefix="fh12-student-") as temp:
            run = SyntheticRun(Path(temp), case=case)
            teacher_meta = Path(temp) / "work_dir" / case.teacher_run_id / "meta"
            teacher_meta.mkdir(parents=True)
            (teacher_meta / "config.resolved.yaml").write_text(yaml.safe_dump(build_config(teacher_for("s4"))))
            with patch("fh12.calibration.load_reference", return_value=(reference, np.full((8, 4), .46875))), \
                 patch.object(training, "load_checkpoint_model", return_value=(teacher, {"update": 50000})):
                self.assertEqual(run.run(lambda *a, **k: fake_result()), 0)
            for key, expected in before.items():
                torch.testing.assert_close(expected, teacher.state_dict()[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
            self.assertFalse(teacher.training)
            init = json.loads((run.wd / "init_manifest.json").read_text())
            self.assertEqual(init["teacher_ref_sha"], reference["teacher_checkpoint_sha256"])
            self.assertFalse(init["teacher_from_scratch"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
