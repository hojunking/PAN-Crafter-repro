"""Synthetic CPU learning/resume tests; never experimental quality evidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from donor30.artifacts import _same, capture_rng
from panda_method.calibration import state_dict_sha256
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_warp30.tests.test_stage1 import PINS, evaluation
from panda_band30.stage1 import build_stage1 as old_build
from panda_band30.plan import resolve_case as old_config
from panda_mature30.plan import resolve_case
from panda_mature30.stage1 import (
    build_stage1, validate_config, save_stage1, load_stage1,
    Stage1Stream, advance_stage1,
)


def config(server="s3"):
    cfg = resolve_case(server, "G8_MATURE_U23")
    cfg["baseline_resolved"].update(effective_batch_RR=1, microbatch_RR=1, accumulation_RR=1)
    return cfg


class Stage1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.registered = synthetic_sources(self.root)

    def test_initial_weights_and_learning_are_exact_old_g8(self):
        cfg = config()
        old = old_config("s1", "G8_S1_U23")
        old["baseline_resolved"].update(effective_batch_RR=1, microbatch_RR=1, accumulation_RR=1)
        a = build_stage1(cfg, self.registered.sources)
        b = old_build(old, self.registered.sources)
        for role in ("aligner", "teacher"):
            self.assertEqual(state_dict_sha256(getattr(a.stage, role)),
                             state_dict_sha256(getattr(b.stage, role)))
        sa = Stage1Stream(self.registered.sources, cfg, device="cpu")
        sb = Stage1Stream(self.registered.sources, old, device="cpu")
        left, right = advance_stage1(a, sa), advance_stage1(b, sb)
        self.assertTrue(_same(left, right))
        for role in ("aligner", "teacher"):
            self.assertEqual(state_dict_sha256(getattr(a.stage, role)),
                             state_dict_sha256(getattr(b.stage, role)))

    def test_complete_resume_keeps_optimizer_stream_rng_and_next_update(self):
        cfg = config()
        a = build_stage1(cfg, self.registered.sources)
        stream = Stage1Stream(self.registered.sources, cfg, device="cpu")
        advance_stage1(a, stream)
        checkpoint = self.root / "stage1.pt"
        saved = save_stage1(checkpoint, a, stream, evaluation=evaluation(a), **PINS)
        expected = advance_stage1(a, stream)
        before = capture_rng()
        b, resumed, payload = load_stage1(checkpoint, saved["file_sha256"], cfg,
            self.registered.sources, restore_global_rng=False, **PINS)
        self.assertTrue(_same(before, capture_rng()))
        actual = advance_stage1(b, resumed)
        self.assertTrue(_same(expected, actual))
        self.assertTrue(_same(a.state_dict(), b.state_dict()))
        self.assertIn("VAL_RRE50", payload["selection_rule"])

    def test_config_rejects_bandwise_bicubic_wrong_campaign(self):
        original = config()
        for key, value in (("aligner_outputs", 16), ("upsampler", "bicubic"),
                           ("server", "s4"), ("campaign_id", "wrong")):
            cfg = deepcopy(original)
            cfg[key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg, self.registered.sources)


if __name__ == "__main__":
    unittest.main()
