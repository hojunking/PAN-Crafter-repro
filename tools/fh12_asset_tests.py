#!/usr/bin/env python3
"""Tiny CPU-only FH12 data/calibration tests; never touches real data or GPU."""
import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from fh12.calibration import (CONSTANT_ALIGNER_Q, _teacher_identity, _write_npz_immutable,
                              axis16_q, compute_calibration, load_reference)
from fh12.data import (AUGMENTATION, RECIPE, FH12Dataset, build_dataset, canonical_sha,
                       check_deadline, phase_diagnostics, prepare_data, sha256_file,
                       write_immutable_json)
from tools.repair_lpan import make_lpan

TEST_RELEASE = {"synthetic_test_release": "not-a-production-checkpoint"}


class ConstantTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor([.3, -.2]))

    def predict_delta(self, pan, ms_base):
        return self.offset[None].expand(len(pan), 2)

    def forward(self, pan, ms, lpan):
        return {"y": torch.full((len(pan), 8, *pan.shape[-2:]), .125, device=pan.device)}


class TinyCalibrationData:
    has_gt = True
    def __init__(self):
        self.base_seen, self.aug_seen = [], []

    def __len__(self):
        return 3

    def get_view(self, index, rot=0, augment=False):
        if augment:
            self.aug_seen.append((index, rot))
        return (torch.zeros(8, 16, 16), torch.zeros(8, 16, 16), torch.zeros(8, 4, 4),
                torch.zeros(1, 4, 4), torch.zeros(1, 16, 16), torch.tensor([index, rot, int(augment), int(augment)]))

    def base(self, index):
        self.base_seen.append(index)
        return self.get_view(index)


def fixture_h5(path, n=3, size=32, gt=True):
    rng = np.random.default_rng(334)
    pan = rng.uniform(0, 2047, (n, 1, size, size)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f["pan"] = pan
        f["ms"] = rng.uniform(0, 2047, (n, 8, size // 4, size // 4)).astype(np.float32)
        f["lms"] = np.repeat(np.repeat(f["ms"][:], 4, -1), 4, -2)
        if gt:
            f["gt"] = f["lms"][:]
    return pan


def fixture_teacher(root):
    run = root / "work_dir" / "FH12_TEST_T"
    (run / "meta").mkdir(parents=True)
    candidate = run / "candidates" / "50000"
    candidate.mkdir(parents=True)
    cfg = {"fh12": {"role": "T", "server_id": "s1", "input_layout": "PH"}}
    (run / "meta" / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    save_file({"test": torch.ones(2)}, str(candidate / "model.safetensors"))
    state = {"update": 50000, "config_sha256": canonical_sha(cfg), "model_sha256": sha256_file(candidate / "model.safetensors"),
             "source_identity": TEST_RELEASE}
    torch.save(state, candidate / "training_state.pt")
    (candidate / "identity.json").write_text(json.dumps(dict(state, training_state_sha256=sha256_file(candidate / "training_state.pt"))))
    return run, candidate, cfg, state


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.source_patch = patch("fh12.calibration.source_identity", return_value=TEST_RELEASE)
        self.source_patch.start()
        self.tmp = tempfile.TemporaryDirectory(prefix="fh12-assets-")
        self.root = Path(self.tmp.name)
        self.paths = {}
        for split, count in (("train", 3), ("val", 2), ("rr", 20), ("fr", 20)):
            path = self.root / f"{split}.h5"
            fixture_h5(path, count, gt=split != "fr")
            self.paths[split] = str(path)

    def tearDown(self):
        self.source_patch.stop()
        self.tmp.cleanup()

    def prepare(self):
        return prepare_data(self.root, "s1", data_paths=self.paths)

    def test_recipe_exact_and_original_files_untouched(self):
        before = {k: sha256_file(v) for k, v in self.paths.items()}
        m = self.prepare()
        for split, item in m["splits"].items():
            self.assertEqual(before[split], sha256_file(item["dataroot"]))
            with h5py.File(item["dataroot"]) as source, h5py.File(item["lpan_path"]) as cache:
                expected = make_lpan(source["pan"][:].astype(np.float64)).astype(np.float32)
                np.testing.assert_array_equal(cache["lpan"][:], expected)
            self.assertFalse(Path(item["lpan_path"]).stat().st_mode & 0o222)
        self.assertEqual(m, self.prepare())

    def test_data_count_and_cache_tamper_rejected(self):
        fixture_h5(self.paths["rr"], 19)
        with self.assertRaisesRegex(ValueError, "requires 20"):
            self.prepare()

    def test_cached_source_changed_rejected(self):
        m = self.prepare()
        with h5py.File(self.paths["train"], "r+") as f:
            f["pan"][0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            build_dataset(m, "train")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_no_constructor_global_rng_reset(self):
        m = self.prepare()
        random.seed(9123)
        expected = random.Random(9123).random()
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        build_dataset(m, "train")
        self.assertEqual(expected, random.random())
        torch.testing.assert_close(torch_state, torch.get_rng_state())
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])

    def test_explicit_views_match_fixed_flip_then_rotation(self):
        dataset = build_dataset(self.prepare(), "train")
        base = dataset[1]
        torch.testing.assert_close(base[-1], torch.tensor([1, 0, 0, 0]))
        for rot in range(4):
            row = dataset[(1, rot)]
            for reference, actual in zip(base[:-1], row[:-1]):
                torch.testing.assert_close(torch.rot90(reference.flip((-2, -1)), rot, (-2, -1)), actual)
            torch.testing.assert_close(row[-1], torch.tensor([1, rot, 1, 1]))
        with self.assertRaises(ValueError):
            dataset.get_view(1, rot=1, augment=False)

    def test_fr_tuple_and_normalization(self):
        dataset = build_dataset(self.prepare(), "fr")
        self.assertFalse(dataset.has_gt)
        self.assertEqual(len(dataset[0]), 5)
        with h5py.File(dataset.raw_h5_path) as f:
            expected = torch.from_numpy(f["pan"][0]) * (2 / 2047) - 1
        torch.testing.assert_close(dataset[0][-2], expected)

    def test_implicit_augment_and_missing_explicit_lp_fail(self):
        m = self.prepare()
        with self.assertRaises(ValueError):
            build_dataset(m, "train", augment=True)
        item = m["splits"]["train"]
        with self.assertRaises(FileNotFoundError):
            FH12Dataset(item["dataroot"], self.root / "missing_lp.h5")

    def test_immutable_json_and_expired_deadline(self):
        path = self.root / "asset.json"
        write_immutable_json(path, {"a": 1})
        write_immutable_json(path, {"a": 1})
        with self.assertRaises(ValueError):
            write_immutable_json(path, {"a": 2})
        with self.assertRaises(TimeoutError):
            check_deadline("2000-01-01T00:00:00+00:00")
        with self.assertRaises(ValueError):
            check_deadline("2099-01-01T00:00:00")

    def test_phase_is_measured_not_silently_corrected(self):
        result = phase_diagnostics()
        self.assertEqual(result["phase_id"], RECIPE["phase_id"])
        self.assertAlmostEqual(result["ramp_interior_mean_L_minus_P"], .5, places=5)
        self.assertTrue(np.isfinite(result["impulse_reconstructed_centroid_yx"]).all())

    def test_constant_aligner_axis16_exact_definition(self):
        model = ConstantTeacher()
        q, per_radius, native = axis16_q(model, torch.zeros(2, 1, 16, 16), torch.zeros(2, 8, 4, 4))
        torch.testing.assert_close(q, torch.full((2,), CONSTANT_ALIGNER_Q))
        torch.testing.assert_close(per_radius, torch.tensor([[.125, .25, .5, 1.]]).expand(2, 4))
        torch.testing.assert_close(native, model.offset[None].expand(2, 2))

    def test_tau_final_output_unaugmented_and_q_complete_four_views(self):
        data = TinyCalibrationData()
        model = ConstantTeacher()
        cal, arrays = compute_calibration(model, data, [2, 0], device="cpu", batch_size=2)
        self.assertEqual(cal["tau_R"], .125)
        self.assertEqual(data.base_seen, [2, 0])
        self.assertEqual(set(data.aug_seen), {(i, r) for i in range(3) for r in range(4)})
        self.assertEqual(cal["q_ref"], CONSTANT_ALIGNER_Q)
        self.assertTrue(cal["diagnostics"]["constant_q_warning"])
        self.assertTrue(cal["diagnostics"]["near_constant_aligner_q_warning"])
        self.assertFalse(model.offset.requires_grad)
        np.testing.assert_allclose(cal["q_ref"] / (cal["q_ref"] + arrays["q"]), .5)

    def test_zero_q_ref_is_failure_not_epsilon_repair(self):
        def zeros(model, pan, ms):
            return torch.zeros(len(pan)), torch.zeros(len(pan), 4), torch.zeros(len(pan), 2)
        with patch("fh12.calibration.axis16_q", side_effect=zeros):
            with self.assertRaisesRegex(ValueError, "do not repair zero"):
                compute_calibration(ConstantTeacher(), TinyCalibrationData(), [0, 1], device="cpu")

    def test_immutable_npz_detects_wrong_values(self):
        path = self.root / "q.npz"
        _write_npz_immutable(path, {"q": np.ones((2, 4))})
        _write_npz_immutable(path, {"q": np.ones((2, 4))})
        with self.assertRaises(ValueError):
            _write_npz_immutable(path, {"q": np.zeros((2, 4))})

    def test_exact50k_teacher_identity_and_role(self):
        run, candidate, cfg, state = fixture_teacher(self.root)
        actual_cfg, identity = _teacher_identity(run, self.root, "s1")
        self.assertEqual(actual_cfg, cfg)
        self.assertEqual(identity["teacher_layout"], "PH")
        self.assertEqual(identity["teacher_update"], 50000)
        with self.assertRaises(ValueError):
            _teacher_identity(run, self.root, "s2")
        state["update"] = 49490
        torch.save(state, candidate / "training_state.pt")
        (candidate / "identity.json").write_text(json.dumps(dict(state, training_state_sha256=sha256_file(candidate / "training_state.pt"))))
        with self.assertRaisesRegex(ValueError, "exact update 50000"):
            _teacher_identity(run, self.root, "s1")

    def test_teacher_state_corruption_rejected_before_unpickle(self):
        run, candidate, _, _ = fixture_teacher(self.root)
        (candidate / "training_state.pt").write_bytes(b"invalid pickle and changed checkpoint")
        with patch("fh12.calibration.torch.load") as loader:
            with self.assertRaisesRegex(ValueError, "checksum mismatch before loading"):
                _teacher_identity(run, self.root, "s1")
            loader.assert_not_called()

    def test_teacher_release_drift_rejected(self):
        run, _, _, _ = fixture_teacher(self.root)
        with patch("fh12.calibration.source_identity", return_value={"new_code": True}), \
             patch("fh12.calibration.torch.load") as loader:
            with self.assertRaisesRegex(ValueError, "differs from checkpoint release"):
                _teacher_identity(run, self.root, "s1")
            loader.assert_not_called()

    def test_reference_readback_and_tau_tamper_guard(self):
        data = self.prepare()
        run, _, _, _ = fixture_teacher(self.root)
        _, identity = _teacher_identity(run, self.root, "s1")
        base = self.root / "work_dir" / "_fh12" / "s1"
        cal, arrays = compute_calibration(ConstantTeacher(), TinyCalibrationData(), [0, 2], device="cpu")
        cal.update(identity)
        cal_path = base / "calibration.json"
        write_immutable_json(cal_path, cal)
        q_path = base / "q_cache.npz"
        _write_npz_immutable(q_path, arrays)
        train = data["splits"]["train"]
        manifest = {
            "schema": "FH12_REFERENCE_v1", **identity,
            "tau_R": cal["tau_R"], "q_ref": cal["q_ref"], "q_shape": cal["q_shape"],
            "calibration_indices_sha256": cal["calibration_indices_sha256"],
            "calibration_path": str(cal_path), "calibration_sha256": sha256_file(cal_path),
            "q_cache_path": str(q_path), "q_cache_sha256": sha256_file(q_path),
            "dataset_manifest_path": str(base / "dataset_manifest.json"),
            "dataset_manifest_sha256": sha256_file(base / "dataset_manifest.json"),
            "lpan_manifest_path": str(base / "lpan_manifest.json"),
            "lpan_manifest_sha256": sha256_file(base / "lpan_manifest.json"),
            "augmentation_sha256": canonical_sha(AUGMENTATION), "LP_recipe": RECIPE,
            "train_sha256": train["sha256"], "train_lpan_sha256": train["lpan_sha256"],
            "train_sample_order_sha256": train["sample_order_sha256"],
        }
        ref_path = base / "reference_manifest.json"
        ref_path.write_text(json.dumps(manifest))
        loaded, q = load_reference(ref_path, expected_teacher_run=run.name, data_manifest=data)
        self.assertEqual(loaded, manifest)
        np.testing.assert_array_equal(q, arrays["q"])
        with self.assertRaisesRegex(ValueError, "different Teacher"):
            load_reference(ref_path, expected_teacher_run="old_T0")
        manifest["tau_R"] *= 2
        ref_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "calibrated tau_R"):
            load_reference(ref_path, data_manifest=data)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
