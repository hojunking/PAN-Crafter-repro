"""Portable reference integrity tests with synthetic data and tiny CPU probes."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml
from safetensors.torch import save_file

from fh12.data import AUGMENTATION, RECIPE
from qg40.calibration import select_calibration_indices
from qg40.common import atomic_json, camp, object_sha, read_json, sha256
from qg40.model import state_hash
from qg40.plan import build_config, teacher_for
from qg40.references import (FILE_KEYS, export_reference, import_reference,
                             reference_path, validate_reference)
from qg40.training import atomic_torch


class ProbeModel(nn.Module):
    bands = 4
    def __init__(self, gain=1.):
        super().__init__()
        self.register_buffer("gain", torch.tensor(float(gain)))

    def predict_delta(self, pan, base):
        return torch.zeros(len(pan), 2, device=pan.device)

    def forward(self, pan, ms, lp):
        y = F.interpolate(ms, scale_factor=4, mode="bicubic", align_corners=False)
        y = y + self.gain * .25 * pan
        mean = pan.mean((1, 2, 3))
        return dict(y=y, delta=torch.stack((mean, -mean), dim=1))


class ProbeDataset:
    has_gt, base_count = True, 3072

    def __len__(self):
        return self.base_count

    def base(self, index):
        gen = torch.Generator().manual_seed(index + 41)
        ms = torch.randn(4, 2, 2, generator=gen)
        pan = torch.randn(1, 8, 8, generator=gen)
        gt = torch.zeros(4, 8, 8)
        return gt, gt, ms, torch.zeros(1, 2, 2), pan, torch.tensor([index, 0])

    def get_view(self, index, rot, augment):
        row = self.base(index)
        return (*[torch.rot90(x.flip((-2, -1)), rot, (-2, -1)) for x in row[:5]],
                torch.tensor([index, rot, 1, 1]))


class QG40ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.origin, self.consumer = Path(self.temp.name) / "origin", Path(self.temp.name) / "consumer"
        self.case = teacher_for("QB_TA")
        self.release = dict(files={"qg40/model.py": "synthetic"}, content_sha256="synthetic",
                            numeric_method_revision="synthetic", torch=torch.__version__,
                            numpy=np.__version__, scipy="test", skimage="test", cuda=None,
                            cudnn=None, tf32_matmul=False, tf32_cudnn=False)
        self.patch_source = patch("qg40.references.source_identity", return_value=self.release)
        self.patch_source.start(); self.addCleanup(self.patch_source.stop)
        self.patch_model = patch("qg40.references._model", side_effect=lambda *a, **kw: ProbeModel().eval())
        self.patch_model.start(); self.addCleanup(self.patch_model.stop)
        self.patch_data = patch("qg40.data.build_dataset", return_value=ProbeDataset())
        self.patch_data.start(); self.addCleanup(self.patch_data.stop)
        self.data = self.dataset(self.origin, "s4")
        self.dataset(self.consumer, "s2")
        self.path = reference_path("QB_TA", "s4", self.origin)
        folder = self.path.parent
        folder.mkdir(parents=True)
        cfg = build_config(self.case)
        cfg["qg40"]["dataset_manifest"] = str(camp(self.origin, "s4") / "dataset_manifest.json")
        config = folder / "config.yaml"
        config.write_text(yaml.safe_dump(cfg))
        weights = folder / "model.safetensors"
        model_state = ProbeModel().state_dict()
        save_file(model_state, str(weights))
        state_path, identity_path = folder / "training_state.pt", folder / "identity.json"
        identity = dict(update=50000, config_sha256=object_sha(cfg), data_sha256=object_sha(self.data),
                        source_identity=self.release, state_hash=state_hash(model_state),
                        model_sha256=sha256(weights), full_state=True)
        atomic_torch(state_path, dict(identity, full_state=True, model_state=model_state))
        identity["training_state_sha256"] = sha256(state_path)
        atomic_json(identity_path, identity)
        self.indices = select_calibration_indices(3072)
        self.q = np.full((3072, 4), .46875, np.float32)
        q_path, cal_path = folder / "q_cache.npz", folder / "calibration.json"
        with q_path.open("wb") as stream:
            np.savez(stream, q=self.q, calibration_indices=self.indices)
        self.manifest = dict(schema="QG40_REFERENCE_v1", sensor="QB", num_bands=4, max_pixel=2047,
            reference_id="QB_TA", teacher_run_id=self.case.run_id, teacher_layout="P0", teacher_update=50000,
            teacher_checkpoint=str(weights), teacher_checkpoint_sha256=sha256(weights),
            teacher_training_state=str(state_path), teacher_training_state_sha256=sha256(state_path),
            teacher_config=str(config), teacher_config_sha256=object_sha(cfg), teacher_config_file_sha256=sha256(config),
            teacher_checkpoint_identity=str(identity_path), teacher_checkpoint_identity_sha256=sha256(identity_path),
            calibration_path=str(cal_path), q_cache_path=str(q_path), q_cache_sha256=sha256(q_path),
            dataset_manifest_path=str(camp(self.origin, "s4") / "dataset_manifest.json"),
            dataset_manifest_sha256=sha256(camp(self.origin, "s4") / "dataset_manifest.json"),
            source_identity=self.release, tau_R=.2, q_ref=.46875, calibration_n=3072,
            calibration_indices_sha256=object_sha(self.indices.tolist()), q_shape=[3072, 4],
            LP_recipe=RECIPE, augmentation=AUGMENTATION)
        self.manifest['q_cache_parity_path'] = str(folder / 'q_cache_parity.json')
        self.publish()

    def dataset(self, root, server):
        folder = root / "data"
        folder.mkdir(parents=True)
        original, lp = folder / "train.h5", folder / "lp.h5"
        original.write_bytes(b"synthetic immutable raw train bytes")
        lp.write_bytes(b"synthetic immutable native LP bytes")
        splits = {name: dict(dataroot=str(original), lpan_path=str(lp), sha256=sha256(original),
                    lpan_sha256=sha256(lp), lpan_canonical_sha256="canonical-lp", count=count,
                    sample_order_sha256="canonical-source-order", source_identity="original-source-hash",
                    shapes=dict(ms=[count, 4, 16, 16], pan=[count, 1, 64, 64]))
                  for name, count in (("train", 3072), ("val", 32), ("rr", 20), ("fr", 20))}
        data = dict(schema="QG40_DATA_v1", sensor="QB", num_bands=4, max_pixel=2047,
                    band_order=["B", "G", "R", "NIR"], mtf_sensor="QB", recipe=RECIPE,
                    augmentation=AUGMENTATION, source_provenance={"raw_sha256": "source"}, splits=splits)
        atomic_json(camp(root, server) / "dataset_manifest.json", data)
        return data

    def publish(self):
        from qg40.reference_parity import verify_q_cache, identity_for
        from qg40.references import data_signature
        parity = verify_q_cache(ProbeModel(), ProbeDataset(), self.q,
            reference_identity=identity_for(self.manifest), data_identity=object_sha(data_signature(self.data)),
            device='cpu')
        atomic_json(self.manifest['q_cache_parity_path'], parity)
        self.manifest['q_cache_parity_sha256'] = sha256(self.manifest['q_cache_parity_path'])
        cal = {key: self.manifest[key] for key in ("tau_R", "q_ref", "q_shape",
               "calibration_indices_sha256", "teacher_checkpoint_sha256", "source_identity")}
        cal.update(schema="QG40_CALIBRATION_v1", calibration_n=3072, calibration_seed=1234,
                   synthetic_test=False)
        atomic_json(self.manifest["calibration_path"], cal)
        self.manifest["calibration_sha256"] = sha256(self.manifest["calibration_path"])
        atomic_json(self.path, self.manifest)

    def rewrite_cache(self, q, indices):
        with open(self.manifest["q_cache_path"], "wb") as stream:
            np.savez(stream, q=q, calibration_indices=indices)
        self.manifest.update(q_cache_sha256=sha256(self.manifest["q_cache_path"]),
                             q_shape=list(q.shape), calibration_indices_sha256=object_sha(indices.tolist()))
        self.publish()

    def validate(self):
        return validate_reference("QB_TA", "s4", self.origin)

    def archive(self):
        return export_reference("QB_TA", "s4", self.origin, device="cpu")

    def modified_archive(self, change):
        original = self.archive()
        with tarfile.open(original) as tf:
            contents = {item.name: tf.extractfile(item).read() for item in tf.getmembers()}
        change(contents)
        target = Path(self.temp.name) / "modified.tar.gz"
        with tarfile.open(target, "w:gz") as tf:
            for name, raw in contents.items():
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                tf.addfile(member, io.BytesIO(raw))
        return target

    def test_valid_complete_reference_and_portable_cpu_parity_import(self):
        manifest, q, cfg, paths = self.validate()
        self.assertEqual(q.shape, (3072, 4))
        self.assertEqual(manifest["teacher_update"], 50000)
        archive = self.archive()
        self.assertEqual(self.archive(), archive)
        target = import_reference(archive, "QB_TA", "s2", self.consumer, device="cpu")
        bridge = read_json(target)
        self.assertTrue(bridge["complete"])
        self.assertTrue(bridge["parity"]["pass"])
        self.assertEqual(bridge["origin_reference"], manifest)
        self.assertTrue(all(str(self.consumer) in path for path in bridge["resolved_artifacts"].values()))
        restored, cache, _, _ = validate_reference("QB_TA", "s2", self.consumer)
        np.testing.assert_array_equal(cache, q)
        self.assertEqual(restored["teacher_checkpoint_sha256"], manifest["teacher_checkpoint_sha256"])
        self.assertEqual(import_reference(archive, "QB_TA", "s2", self.consumer, device="cpu"), target)

    def test_wrong_teacher_sensor_and_declared_step_rejected(self):
        saved = copy.deepcopy(self.manifest)
        for key, value in (("teacher_run_id", teacher_for("QB_TB").run_id),
                           ("sensor", "GF2"), ("teacher_update", 49999), ("teacher_layout", "PLH")):
            with self.subTest(key=key):
                self.manifest = dict(saved, **{key: value}); self.publish()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_zero_q_truncated_cache_and_permuted_calibration_ids_rejected(self):
        variants = ((np.zeros_like(self.q), self.indices, 0.),
                    (self.q[:-1], self.indices, .46875),
                    (self.q, self.indices[::-1], .46875))
        for q, indices, q_ref in variants:
            with self.subTest(shape=q.shape, q_ref=q_ref):
                self.manifest["q_ref"] = q_ref
                self.rewrite_cache(q, indices)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_corrupt_cache_and_train_source_hash_rejected(self):
        q_path = Path(self.manifest["q_cache_path"])
        original = q_path.read_bytes()
        q_path.write_bytes(original + b"corruption")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.validate()
        q_path.write_bytes(original)
        Path(self.data["splits"]["train"]["dataroot"]).write_bytes(b"changed raw source")
        with self.assertRaisesRegex(ValueError, "training H5/LP changed"):
            self.validate()

    def test_rehashed_incomplete_training_state_rejected(self):
        state_path = Path(self.manifest["teacher_training_state"])
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        for key, value in (("update", 49999), ("full_state", False)):
            with self.subTest(key=key):
                modified = dict(state, **{key: value})
                atomic_torch(state_path, modified)
                self.manifest["teacher_training_state_sha256"] = sha256(state_path)
                identity = read_json(self.manifest["teacher_checkpoint_identity"])
                identity["training_state_sha256"] = sha256(state_path)
                atomic_json(self.manifest["teacher_checkpoint_identity"], identity)
                self.manifest["teacher_checkpoint_identity_sha256"] = sha256(self.manifest["teacher_checkpoint_identity"])
                self.publish()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_transfer_corruption_and_member_traversal_never_publish(self):
        def corrupt(contents):
            name = next(name for name in contents if name.startswith("assets/"))
            contents[name] += b"corrupted in transit"
        for change in (corrupt, lambda contents: contents.update({"../escaped.txt": b"unsafe"}),
                       lambda contents: contents.update({"/absolute.txt": b"unsafe"})):
            archive = self.modified_archive(change)
            with self.assertRaises(ValueError):
                import_reference(archive, "QB_TA", "s2", self.consumer, device="cpu")
            self.assertFalse(reference_path("QB_TA", "s2", self.consumer).exists())
        self.assertFalse((Path(self.temp.name) / "escaped.txt").exists())

    def test_mapping_traversal_rejected_before_model_load_and_no_publication(self):
        def change(contents):
            index = json.loads(contents["index.json"])
            index["mapping"]["teacher_config"] = "../outside.yaml"
            contents["index.json"] = json.dumps(index).encode()
        archive = self.modified_archive(change)
        with patch("qg40.references._model") as model:
            with self.assertRaises(ValueError):
                import_reference(archive, "QB_TA", "s2", self.consumer, device="cpu")
            model.assert_not_called()
        self.assertFalse(reference_path("QB_TA", "s2", self.consumer).exists())

    def test_parity_failure_never_publishes_consumable_bridge(self):
        archive = self.archive()
        with patch("qg40.references._model", return_value=ProbeModel(gain=2.)):
            with self.assertRaisesRegex(ValueError, "parity"):
                import_reference(archive, "QB_TA", "s2", self.consumer, device="cpu")
        self.assertFalse(reference_path("QB_TA", "s2", self.consumer).exists())

    def test_wrong_reference_for_consumer_rejected(self):
        with self.assertRaises(ValueError):
            import_reference(self.archive(), "QB_TB", "s2", self.consumer, device="cpu")

    def test_synthetic_or_rebound_q_parity_receipt_rejected(self):
        path = Path(self.manifest['q_cache_parity_path'])
        receipt = read_json(path)
        for changed in (dict(receipt, synthetic_test=True), dict(receipt, indices=list(reversed(receipt['indices'])))):
            atomic_json(path, changed)
            self.manifest['q_cache_parity_sha256'] = sha256(path)
            atomic_json(self.path, self.manifest)
            with self.assertRaisesRegex(ValueError, 'parity receipt'):
                self.validate()

    def test_load_rechecks_actual_local_online_cache_not_only_stored_receipt(self):
        from qg40.references import load_reference
        with patch('qg40.reference_parity.axis16_q', side_effect=lambda model, pan, ms:
                   (torch.full((len(pan),), .25), None, None)):
            with self.assertRaisesRegex(ValueError, 'q-cache/online'):
                load_reference('QB_TA', 's4', self.origin, device='cpu')

    def test_import_rechecks_actual_q_cache_on_consumer(self):
        archive = self.archive()
        with patch('qg40.reference_parity.axis16_q', side_effect=lambda model, pan, ms:
                   (torch.full((len(pan),), .25), None, None)):
            with self.assertRaisesRegex(ValueError, 'q-cache/online'):
                import_reference(archive, 'QB_TA', 's2', self.consumer, device='cpu')
        self.assertFalse(reference_path('QB_TA', 's2', self.consumer).exists())

    def test_expired_window_prevents_export_or_import_model_work(self):
        archive = self.archive()
        with patch("qg40.references._model") as model:
            with self.assertRaises(TimeoutError):
                export_reference("QB_TA", "s4", self.origin, deadline="2000-01-01T00:00:00Z")
            with self.assertRaises(TimeoutError):
                import_reference(archive, "QB_TA", "s2", self.consumer, deadline="2000-01-01T00:00:00Z")
            model.assert_not_called()
        self.assertFalse(reference_path("QB_TA", "s2", self.consumer).exists())

    def test_import_refuses_filesystem_symlink_without_external_write(self):
        archive = self.archive()
        destination = reference_path("QB_TA", "s2", self.consumer).parent / "imported_bytes"
        destination.mkdir(parents=True)
        external = Path(self.temp.name) / "external"
        external.mkdir()
        (destination / "assets").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            import_reference(archive, "QB_TA", "s2", self.consumer, device="cpu")
        self.assertEqual(list(external.iterdir()), [])
        self.assertFalse(reference_path("QB_TA", "s2", self.consumer).exists())


if __name__ == "__main__":
    unittest.main()
