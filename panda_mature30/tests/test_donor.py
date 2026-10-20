"""Same-state import and fail-closed cross-server/source identity."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_warp30.tests.test_stage1 import PINS, evaluation
from panda_mature30.tests.test_stage1 import config
from panda_mature30.stage1 import build_stage1, save_stage1, Stage1Stream, advance_stage1
from panda_mature30.donor import load_donor, _check_source_config, LEGACY_CAMPAIGN
from panda_method.calibration import state_dict_sha256
from donor30.artifacts import _same, capture_rng
from ta2.evaluation import file_sha


class DonorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_native_same_checkpoint_frozen_import_and_no_rng_side_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registered = synthetic_sources(root)
            cfg = config()
            trainer = build_stage1(cfg, registered.sources)
            stream = Stage1Stream(registered.sources, cfg, device="cpu")
            advance_stage1(trainer, stream)
            path = root / "donor.pt"
            saved = save_stage1(path, trainer, stream, evaluation=evaluation(trainer), **PINS)
            before = capture_rng()
            donor = load_donor(path, saved["file_sha256"], registered, server="s3")
            self.assertTrue(_same(before, capture_rng()))
            donor.assert_frozen()
            for role in ("aligner", "teacher"):
                self.assertEqual(state_dict_sha256(getattr(donor, role)),
                                 state_dict_sha256(getattr(trainer.stage, role)))
            self.assertEqual(donor.binding["source_campaign_id"], cfg["campaign_id"])
            with self.assertRaisesRegex(ValueError, "Cross-server"):
                load_donor(path, saved["file_sha256"], registered, server="s1")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                load_donor(path, "0" * 64, registered, server="s3")
            payload = torch.load(path, weights_only=False)
            payload["trainer"]["stage"]["aligner.fc2.bias"] += 1
            corrupt = root / "corrupt.pt"
            torch.save(payload, corrupt)
            with self.assertRaisesRegex(ValueError, "Actual donor tensors"):
                load_donor(corrupt, file_sha(corrupt), registered, server="s3")

    def test_only_exact_local_legacy_global_g8_is_eligible(self):
        cfg = config("s1")
        cfg.update(campaign_id=LEGACY_CAMPAIGN, case="G8_S1_U23")
        _check_source_config(cfg, "s1")
        for key, value in (("upsampler", "bicubic"), ("aligner_outputs", 16),
                           ("case", "B8A_S1_U23"), ("alignment_target", "RGB")):
            bad = deepcopy(cfg)
            bad[key] = value
            with self.assertRaises(ValueError):
                _check_source_config(bad, "s1")
        with self.assertRaises(ValueError):
            _check_source_config(cfg, "s3")


if __name__ == "__main__":
    unittest.main()
