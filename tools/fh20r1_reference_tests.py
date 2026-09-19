#!/usr/bin/env python3
"""CPU-only FH20R1 reference provenance/import tests; sparse temporary H5 fixtures."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from fh12.data import AUGMENTATION, RECIPE, canonical_sha, sha256_file
from fh12.plan import teacher_for, build_config
from fh20r1 import references as R
from fh20r1.historical import (ATOL, RTOL, compare_probe_arrays, historical_snapshot,
                                validate_source_identity)


RELEASE = {"consumer_test_release": "unit-test-only"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    server, run_id = "s1", teacher_for("s1").run_id
    assets = root / "work_dir/_fh12/s1"
    run = root / "work_dir" / run_id
    refdir = assets / "references" / run_id
    candidate = run / "candidates/50000"
    candidate.mkdir(parents=True)
    data = {"recipe": RECIPE, "augmentation": AUGMENTATION, "splits": {}}
    for split, count, side in (("train", 9714, 64), ("val", 1080, 64), ("rr", 20, 256), ("fr", 20, 512)):
        source = root / "data" / f"{split}.h5"
        source.parent.mkdir(parents=True, exist_ok=True)
        # Sparse logical datasets preserve the exact production geometry without
        # allocating gigabytes or using any real training/evaluation data.
        with h5py.File(source, "w") as f:
            for key, channels, size in (("pan", 1, side), ("ms", 8, side//4), ("lms", 8, side)):
                f.create_dataset(key, shape=(count, channels, size, size), dtype="float32")
            if split != "fr":
                f.create_dataset("gt", shape=(count, 8, side, side), dtype="float32")
        source_sha = sha256_file(source)
        lp = assets / "lpan" / f"{split}.h5"
        lp.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(lp, "w") as f:
            f.create_dataset("lpan", shape=(count, 1, side//4, side//4), dtype="float32")
            f.attrs["source_sha256"] = source_sha
            f.attrs["sample_order_sha256"] = "1"*64
            f.attrs["recipe_sha256"] = canonical_sha(RECIPE)
        data["splits"][split] = {"dataroot": str(source), "lpan_path": str(lp), "sha256": source_sha,
                                 "lpan_sha256": sha256_file(lp), "sample_order_sha256": "1"*64, "count": count}
    data_path, lp_path = assets / "dataset_manifest.json", assets / "lpan_manifest.json"
    write_json(data_path, data)
    write_json(lp_path, {"recipe": RECIPE, "splits": data["splits"]})
    cfg = build_config(teacher_for("s1"))
    config_path = run / "meta/config.resolved.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(cfg))
    save_file({"test_tensor": torch.ones(2)}, str(candidate / "model.safetensors"))
    write_json(run / "init_manifest.json", {"policy": "synthetic-unit-fixture"})
    files = {"fh12/model.py": hashlib.sha256(b"old source").hexdigest()}
    source = {"git_release": "a"*40, "files": files, "content_sha256": canonical_sha(files)}
    state = {"update": 50000, "model_sha256": sha256_file(candidate / "model.safetensors"),
             "config_sha256": canonical_sha(cfg), "source_identity": source}
    torch.save(state, candidate / "training_state.pt")
    identity = dict(state, training_state_sha256=sha256_file(candidate / "training_state.pt"))
    write_json(candidate / "identity.json", identity)
    q_path = refdir / "q_cache.npz"
    q_path.parent.mkdir(parents=True)
    indices = np.arange(3072, dtype=np.int64)[::-1].copy()
    q = np.full((9714, 4), .4, dtype=np.float32)
    np.savez(q_path, q=q, calibration_indices=indices)
    origin = {"schema": "FH12_REFERENCE_v1", "teacher_run_id": run_id, "server": server,
              "teacher_layout": "P0", "teacher_update": 50000, "source_identity": source,
              "tau_R": .012, "q_ref": float(np.float32(.4)), "q_shape": [9714, 4],
              "teacher_checkpoint_sha256": state["model_sha256"], "teacher_config_sha256": canonical_sha(cfg),
              "calibration_indices_sha256": canonical_sha(indices.tolist()),
              "LP_recipe": RECIPE, "augmentation": AUGMENTATION,
              "train_sha256": data["splits"]["train"]["sha256"],
              "train_lpan_sha256": data["splits"]["train"]["lpan_sha256"], "train_sample_order_sha256": "1"*64}
    pairs = {"teacher_checkpoint": candidate / "model.safetensors", "teacher_training_state": candidate / "training_state.pt",
             "teacher_config": config_path, "teacher_checkpoint_identity": candidate / "identity.json",
             "q_cache_path": q_path, "dataset_manifest_path": data_path, "lpan_manifest_path": lp_path}
    hashes = {"teacher_checkpoint": "teacher_checkpoint_sha256", "teacher_training_state": "teacher_training_state_sha256",
              "teacher_config": "teacher_config_file_sha256", "teacher_checkpoint_identity": "teacher_checkpoint_identity_sha256",
              "q_cache_path": "q_cache_sha256", "dataset_manifest_path": "dataset_manifest_sha256", "lpan_manifest_path": "lpan_manifest_sha256"}
    for key, path in pairs.items():
        origin[key] = str(path)
        origin[hashes[key]] = sha256_file(path)
    cal_path = refdir / "calibration.json"
    write_json(cal_path, origin)
    origin.update(calibration_path=str(cal_path), calibration_sha256=sha256_file(cal_path))
    write_json(refdir / "reference_manifest.json", origin)
    spec = dict(R.REFERENCE_ASSETS["F1"], expected_sha256=state["model_sha256"])
    return origin, data, spec, refdir / "reference_manifest.json", indices


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fh20r1-reference-tests-")
        self.root = Path(self.temp.name)
        self.origin, self.data, spec, self.origin_path, self.indices = fixture(self.root)
        self.spec_patch = patch.dict(R.REFERENCE_ASSETS, {"F1": spec})
        self.spec_patch.start()
        self.consumer_patch = patch.object(R, "_consumer_identity", return_value=RELEASE)
        self.consumer_patch.start()

    def tearDown(self):
        self.consumer_patch.stop(); self.spec_patch.stop(); self.temp.cleanup()

    def test_readonly_full_origin_validation(self):
        before = {str(p): sha256_file(p) for p in self.root.rglob("*") if p.is_file()}
        cfg, origin, data, q, resolved = R.validate_origin("F1", "s1", self.root)
        self.assertEqual(origin, self.origin)
        self.assertEqual(q.shape, (9714, 4))
        self.assertTrue(Path(resolved["init_manifest"]).is_file())
        after = {str(p): sha256_file(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_full_sha_tail_mismatch_is_not_accepted_as_prefix(self):
        wrong = dict(R.REFERENCE_ASSETS["F1"], expected_sha256=self.origin["teacher_checkpoint_sha256"][:16]+"0"*48)
        with patch.dict(R.REFERENCE_ASSETS, {"F1": wrong}), self.assertRaisesRegex(ValueError, "Cases CSV"):
            R.validate_origin("F1", "s1", self.root)

    def test_wrong_local_server_rejected(self):
        with self.assertRaises(ValueError):
            R.validate_origin("F1", "s2", self.root)

    def test_state_hash_checked_before_unpickle(self):
        Path(self.origin["teacher_training_state"]).write_bytes(b"bad pickle")
        with patch.object(R.torch, "load") as loader:
            with self.assertRaisesRegex(ValueError, "SHA"):
                R.validate_origin("F1", "s1", self.root)
            loader.assert_not_called()

    def test_raw_cache_tamper_is_detected(self):
        Path(self.origin["q_cache_path"]).write_bytes(b"invalid npz")
        with self.assertRaises(ValueError):
            R.validate_origin("F1", "s1", self.root)

    def test_existing_external_symlink_target_path_kept(self):
        with tempfile.TemporaryDirectory(prefix="fh20r1-external-data-") as name:
            p = Path(name) / "data/datasets/wv3/train.h5"
            p.parent.mkdir(parents=True); p.write_bytes(b"external fixture")
            self.assertEqual(R._relocate(str(p), self.root), p)

    def test_new_bridge_does_not_rewrite_old_source_or_scalar(self):
        old_bytes = self.origin_path.read_bytes()
        with patch.object(R, "run_parity", return_value={"status": "PASS", "sample_counts": {"train64": 8, "rr256": 2, "fr512": 2}}):
            path = R.import_reference("s1", "F1", self.root, "cpu")
        bridge = json.loads(path.read_text())
        self.assertEqual(self.origin_path.read_bytes(), old_bytes)
        self.assertEqual(bridge["origin_source_identity"], self.origin["source_identity"])
        self.assertEqual(bridge["consumer_source_identity"], RELEASE)
        self.assertNotEqual(bridge["consumer_source_identity"], bridge["origin_source_identity"])
        self.assertFalse(bridge["exact_resume_authorized"])
        with patch.object(R, "_load_frozen", return_value="frozen teacher"):
            teacher, cfg, loaded, q = R.load_reference("F1", "s1", self.root, "cpu")
            self.assertEqual(teacher, "frozen teacher")
            np.testing.assert_array_equal(R.calibration_indices("F1", "s1", self.root), self.indices)
            self.assertEqual(loaded["tau_R"], self.origin["tau_R"])

    def test_consumer_source_change_blocks_cached_bridge_reuse(self):
        with patch.object(R, "run_parity", return_value={"status": "PASS"}):
            R.import_reference("s1", "F1", self.root, "cpu")
        with patch.object(R, "_consumer_identity", return_value={"different": "code"}), self.assertRaisesRegex(ValueError, "Consumer release"):
            R.load_reference("F1", "s1", self.root, "cpu")

    def test_origin_manifest_edit_blocks_cached_bridge(self):
        with patch.object(R, "run_parity", return_value={"status": "PASS"}):
            R.import_reference("s1", "F1", self.root, "cpu")
        self.origin["tau_R"] = .123
        write_json(self.origin_path, self.origin)
        with self.assertRaises(ValueError):
            R.load_reference("F1", "s1", self.root, "cpu")

    def test_donor_missing_is_structured_block_not_whole_campaign_exception(self):
        state, report = R.load_donor_aligner(self.root)
        self.assertIsNone(state)
        self.assertEqual(report["status"], "BLOCKED_DONOR")
        self.assertFalse(report["donor_available"])
        self.assertEqual(report["fallback"], "F2_PL_vs_PLH")

    def test_donor_extracts_only_A_preserves_nonzero_head(self):
        from pa.aligner import PANGlobalAligner
        run = self.root / "work_dir" / R.DONOR_RUN
        (run / "last").mkdir(parents=True)
        with torch.random.fork_rng(devices=[]):
            a = PANGlobalAligner(8)
        a.fc2.weight.data.fill_(.03)
        a.fc2.bias.data.fill_(.04)
        state = {"aligner."+k: v for k, v in a.state_dict().items()}
        state["backbone.DO_NOT_LOAD"] = torch.tensor([123.])
        container = run / "last/model.safetensors"
        save_file(state, str(container))
        cfg = dict(trainer="po", num_bands=8, po=dict(case="N2_SG", radius_hr=2),
                   train_feeder_args=dict(dataroot=self.data["splits"]["train"]["dataroot"]))
        (run / "meta").mkdir()
        (run / "meta/config.yaml").write_text(yaml.safe_dump(cfg))
        write_json(run / "last_meta.json", {"step": 50000})
        write_json(run / "dataset_hashes.json", {"train_feeder_args": self.data["splits"]["train"]})
        write_json(self.root / "work_dir/_fh12/s2/dataset_manifest.json", self.data)
        with patch.object(R, "DONOR_SHA", sha256_file(container)):
            actual, report = R.load_donor_aligner(self.root)
        self.assertEqual(report["status"], "VERIFIED")
        self.assertFalse(report["U_loaded"])
        self.assertNotIn("backbone.DO_NOT_LOAD", actual)
        torch.testing.assert_close(actual["fc2.weight"], a.fc2.weight)
        torch.testing.assert_close(actual["fc2.bias"], a.fc2.bias)
        self.assertGreater(report["fc2_nonzero_count"], 0)

    def test_n2_calibration_uses_f2_actual_indices_and_fresh_scalars(self):
        from fh20r1.plan import build_config as new_config, case_for
        run = self.root / "work_dir" / R.N2_RUN
        candidate = run / "candidates/50000"
        candidate.mkdir(parents=True)
        cfg = new_config(case_for(R.N2_RUN))
        cfg_path = run / "meta/config.resolved.yaml"
        cfg_path.parent.mkdir(parents=True)
        cfg_path.write_text(yaml.safe_dump(cfg))
        save_file({"test_tensor": torch.ones(2)}, str(candidate / "model.safetensors"))
        (candidate / "training_state.pt").write_bytes(b"mock state, checksum verified before mocked loader")
        identity = dict(update=50000, config_sha256=canonical_sha(cfg), source_identity=RELEASE,
                        model_sha256=sha256_file(candidate / "model.safetensors"),
                        training_state_sha256=sha256_file(candidate / "training_state.pt"))
        write_json(candidate / "identity.json", identity)
        write_json(run / "init_manifest.json", {"synthetic": True})
        f2 = dict(dataset_manifest=self.data, origin_reference=self.origin,
                  resolved_artifacts={key: self.origin[key] for key in ("q_cache_path", "dataset_manifest_path", "lpan_manifest_path")})
        write_json(R.bridge_path("s2", "F2", self.root), f2)
        captured = []
        def compute(model, dataset, indices, **kwargs):
            captured.append(indices.copy())
            cal = dict(tau_R=.05, q_ref=.25, q_shape=[9714, 4], calibration_indices_sha256=canonical_sha(indices.tolist()))
            arrays = dict(q=np.full((9714, 4), .25, dtype=np.float32), calibration_indices=indices.copy())
            return cal, arrays
        with patch.object(R, "load_reference", return_value=(None, None, f2, None)), \
             patch.object(R, "build_dataset", return_value=object()), \
             patch("fh20r1.common.load_checkpoint_model", return_value=(object(), identity)), \
             patch("fh12.calibration.compute_calibration", side_effect=compute):
            path = R.calibrate_n2pl(R.N2_RUN, self.root, "s2", "cpu")
        np.testing.assert_array_equal(captured[0], self.indices)
        bridge = json.loads(path.read_text())
        self.assertEqual(bridge["tau_R"], .05)
        self.assertEqual(bridge["q_ref"], .25)
        self.assertNotEqual(bridge["tau_R"], self.origin["tau_R"])
        self.assertNotEqual(bridge["q_ref"], self.origin["q_ref"])
        with np.load(bridge["resolved_artifacts"]["q_cache_path"]) as q:
            np.testing.assert_array_equal(q["calibration_indices"], self.indices)


class HistoricalTests(unittest.TestCase):
    def test_pinned_source_hash_and_safe_paths(self):
        files = {"fh12/model.py": hashlib.sha256(b"original").hexdigest()}
        source = dict(git_release="a"*40, files=files, content_sha256=canonical_sha(files))
        validate_source_identity(source)
        bad = dict(source, content_sha256="0"*64)
        with self.assertRaises(ValueError): validate_source_identity(bad)
        files = {"../outside.py": "0"*64}
        with self.assertRaises(ValueError):
            validate_source_identity(dict(source, files=files, content_sha256=canonical_sha(files)))

    def test_archive_is_original_commit_bytes_not_working_tree(self):
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as tar:
            member = tarfile.TarInfo("fh12/model.py"); member.size = len(b"original")
            tar.addfile(member, io.BytesIO(b"original"))
        files = {"fh12/model.py": hashlib.sha256(b"original").hexdigest()}
        source = dict(git_release="b"*40, files=files, content_sha256=canonical_sha(files))
        with patch("fh20r1.historical.subprocess.check_output", return_value=payload.getvalue()) as git:
            with historical_snapshot(ROOT, source) as tree:
                self.assertEqual((tree / "fh12/model.py").read_bytes(), b"original")
            self.assertEqual(git.call_args[0][0][3], "b"*40)
        bad = copy.deepcopy(source); bad["files"]["fh12/model.py"] = "0"*64
        bad["content_sha256"] = canonical_sha(bad["files"])
        with patch("fh20r1.historical.subprocess.check_output", return_value=payload.getvalue()), self.assertRaises(ValueError):
            with historical_snapshot(ROOT, bad): pass

    def test_numeric_parity_tolerance_is_fixed_and_nan_rejected(self):
        a = {"train_y": np.zeros((8, 8, 2, 2), dtype=np.float32)}
        b = {"train_y": np.full((8, 8, 2, 2), ATOL/2, dtype=np.float32)}
        compare_probe_arrays(a, b)
        for bad in (np.full_like(a["train_y"], ATOL*2), np.full_like(a["train_y"], np.nan)):
            with self.assertRaises(ValueError): compare_probe_arrays(a, {"train_y": bad})
        self.assertEqual(RTOL, 2e-5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
