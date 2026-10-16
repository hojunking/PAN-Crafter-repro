"""Real H5/serialization adapter tests; no production training or cloud writes."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from donor24 import plan
from donor24.evaluation import model_state_hash
from donor30.artifacts import SCHEMA
from donor30.runtime import build_trainer
from panda_method.calibration import state_dict_sha256
from panda_method.edge_data import EdgeDataSources
from ta2.evaluation import file_sha

from panda_stage2_condq.donor import load_donor
from panda_stage2_condq.sources import RegisteredRRTrain, batch_metadata, validate_training_batch


def synthetic_sources(root, count=3):
    """Full native shapes, stored LMS intentionally distinct from bicubic MS."""
    root = Path(root)
    data = root / "train.h5"
    with h5py.File(data, "w") as handle:
        for key, channels, side, value in (("pan", 1, 64, 900), ("ms", 8, 16, 700),
                                          ("lms", 8, 64, 500), ("gt", 8, 64, 800)):
            handle.create_dataset(key, data=np.full((count, channels, side, side), value, np.float32))
    catalog = root / "sources.json"
    catalog.write_text(json.dumps({"schema": "ABLR2_SENSOR_SOURCES_v1", "sensors": {"WV3": {
        "band_order": ["Coastal", "B", "G", "Y", "R", "RE", "NIR1", "NIR2"],
        "splits": {"train": {"path": "train.h5", "sha256": file_sha(data)}}}}}))
    return RegisteredRRTrain.from_catalog(catalog, root=root, sensor="WV3")


class DonorSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.registered = synthetic_sources(cls.root)
        cfg = plan.resolve_case(plan.default_config("s1"), "A0")
        cfg["run_id"] = "SYNTHETIC_ADAPTER_TEST_NOT_AN_EXPERIMENT"
        trainer = build_trainer(cfg, cls.registered.sources)
        trainer.updates = 7
        identity = {"model_state_sha256": model_state_hash(trainer.stage),
                    "completed_update": 7, "run_id": cfg["run_id"], "dataset": "WV3"}
        cls.payload = {"schema": SCHEMA, "config": cfg, "trainer": trainer.state_dict(),
                       "identity": identity, "state_hash": identity["model_state_sha256"]}
        cls.path = cls.root / "donor.pt"
        torch.save(cls.payload, cls.path)
        cls.sha = file_sha(cls.path)
        del trainer

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.old_threads)

    def test_complete_train_only_unaugmented_inventory_and_stored_lms(self):
        data = self.registered
        batches = list(data.calibration_batches(2))
        self.assertEqual([len(b.pan) for b in batches], [2, 1])
        self.assertEqual(tuple(v for b in batches for v in b.lineage.sample_ids), data.sample_ids)
        for batch in batches:
            meta = batch_metadata(batch, data.inventory.manifest(), require_unaugmented=True)
            self.assertTrue(meta["unaugmented"])
            self.assertEqual(meta["native_lms_provenance"], "PROVIDED_STORED_LMS")
            self.assertNotEqual(float(batch.native_lms.mean()), float(batch.ms.mean()))
        self.assertEqual(set(data.sources._files), {("train", False)})

    def test_lineage_float_and_inventory_fail_closed(self):
        batch = self.registered.batch([0])
        binding = self.registered.inventory.manifest()
        for key, value in (("split", "test"), ("resolution", "FR"),
                           ("source_manifest_sha256", "a" * 64),
                           ("sample_ids", ("WV3:rr:0",)),
                           ("sample_ids", ("WV3:train:99999",))):
            with self.assertRaises(ValueError):
                altered = replace(batch, lineage=replace(batch.lineage, **{key: value}))
                validate_training_batch(altered, binding, expected_sample_ids=self.registered.sample_ids)
        with self.assertRaises(ValueError):
            validate_training_batch(replace(batch, pan=batch.pan.double()), binding)
        augmented = self.registered.batch([0], augmentation=(True, False, 0))
        with self.assertRaisesRegex(ValueError, "unaugmented"):
            validate_training_batch(augmented, binding, require_unaugmented=True)

    def test_exact_paired_donor_load_frozen_and_rng_unchanged(self):
        before = torch.random.get_rng_state().clone()
        donor = load_donor(self.path, self.sha, self.registered)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(donor.binding["donor_checkpoint_id"], self.sha)
        self.assertEqual(donor.binding["sample_ids_sha256"], self.registered.inventory.sample_ids_sha256)
        self.assertEqual(donor.binding["aligner_state_sha256"], state_dict_sha256(donor.aligner))
        self.assertEqual(donor.binding["teacher_state_sha256"], state_dict_sha256(donor.teacher))
        self.assertEqual(donor.teacher.spec.width, 112)
        self.assertEqual(donor.teacher.spec.depth, (1, 2, 3))
        donor.assert_frozen()
        with torch.no_grad():
            donor.aligner.fc2.bias.add_(1.)
        with self.assertRaisesRegex(ValueError, "state changed"):
            donor.assert_frozen()

    def test_gt_only_import_never_constructs_teacher(self):
        with patch("panda_stage2_condq.donor.build_reconstructor", side_effect=AssertionError("T constructed")):
            donor = load_donor(self.path, self.sha, self.registered, include_teacher=False)
        self.assertIsNone(donor.teacher)
        self.assertFalse(donor.checkpoint_metadata["teacher_constructed"])
        self.assertEqual(len(donor.binding["teacher_state_sha256"]), 64)

    def test_file_sha_required_and_pair_tampering_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA"):
            load_donor(self.path, "0" * 64, self.registered)
        payload = deepcopy(self.payload)
        payload["trainer"]["stage"]["aligner.fc2.bias"].add_(1.)
        path = self.root / "tampered.pt"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "Actual A/T"):
            load_donor(path, file_sha(path), self.registered)

    def test_old_student_and_wrong_architecture_rejected(self):
        for field, value in (("phase", "stage2"), ("dataset", "GF2")):
            payload = deepcopy(self.payload)
            payload["config"][field] = value
            path = self.root / (field + ".pt")
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_donor(path, file_sha(path), self.registered)
        payload = deepcopy(self.payload)
        payload["config"]["baseline_resolved"]["teacher_architecture"]["width"] = 104
        path = self.root / "wrong_arch.pt"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "W112"):
            load_donor(path, file_sha(path), self.registered)

    def test_registry_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            data = synthetic_sources(temp)
            with data.sources.catalog_path.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "registry"):
                data.verify()


if __name__ == "__main__":
    unittest.main()
