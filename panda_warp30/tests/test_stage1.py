"""CPU synthetic checks; metric fixtures are never experimental evidence."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from donor30.artifacts import _same, capture_rng
from donor30.evaluation import model_state_hash
from donor30.masking import masked_mean
from panda_method.calibration import state_dict_sha256
from panda_method.edge_data import augment_edge_batch
from panda_midpoint.resampling import upsample_lrms_after_augmentation
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from pa.warp import warp_pan
from ta2.evaluation import file_sha

from panda_warp30.contract import CAMPAIGN_ID
from panda_warp30.donor import load_donor
from panda_warp30.plan import resolve_case
from panda_warp30.reconstruction import base_from_batch, reconstruct_with_base
from panda_warp30.stage1 import (Stage1Stream, advance_stage1, build_stage1,
                                load_stage1, save_stage1)


PINS = {"code_pin": {"synthetic_unit_test": "a" * 64},
        "evaluator_pin": {"synthetic_not_official_evidence": "b" * 64}}


def config(upsampler="interp23"):
    cfg = resolve_case("s3", "S1_U23" if upsampler == "interp23" else "S1_BIC")
    cfg["run_id"] = "SYNTHETIC_WARP30_NOT_AN_EXPERIMENT_" + upsampler
    cfg["baseline_resolved"].update(effective_batch_RR=1, microbatch_RR=1, accumulation_RR=1)
    return cfg


def evaluation(trainer):
    """Serialization-only fixture, not a computed official metric report."""
    cfg = trainer.config
    identity = {key: cfg[key] for key in ("run_id", "attempt", "seed", "dataset")}
    identity.update(completed_update=trainer.updates, model_state_sha256=model_state_hash(trainer.stage))
    result = {key: deepcopy(identity) for key in ("state_before", "rr_state", "fr_state", "state_after")}
    result.update(qualification={key: True for key in
                  ("same_state_verified", "metrics_complete", "metrics_finite")},
                  full_method_eligible=True, metric_function_injection=False,
                  measurement_domain="OFFICIAL_FULL20_NATIVE",
                  official_reference="original_PAN/native_LMS; FR full512 unmasked; RR all20 dim_cut21")
    for split in ("rr", "fr"):
        result[split] = {"metrics": {"HQNR": .95, "SCC": .98, "ERGAS": 2.1},
                         "per_scene": [{"scene_id": f"WV3:{split}:{i}",
                                        "raw_prediction_sha256": f"{i:064x}"} for i in range(20)]}
    return result


class WarpStage1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registered = synthetic_sources(self.root, count=3)

    def make(self, upsampler="interp23"):
        cfg = config(upsampler)
        return (build_stage1(cfg, self.registered.sources),
                Stage1Stream(self.registered.sources, cfg, device="cpu"), cfg)

    def random_batch(self):
        batch = self.registered.batch([0])
        g = torch.Generator().manual_seed(123)
        return replace(batch, **{key: torch.rand(getattr(batch, key).shape, generator=g) * 1.4 - .7
                                  for key in ("pan", "ms", "native_lms", "gt")})

    def randomize_heads(self, stage):
        # Nonzero outputs are mandatory: a zero residual would hide bypass bugs.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(321)
            stage.teacher.backbone.output[-1].weight.normal_(0., .002)
            stage.teacher.backbone.output[-1].bias.normal_(0., .002)
            stage.aligner.fc2.bias.copy_(torch.tensor([.25, -.5]))

    def test_fresh_arms_have_identical_initial_weights_streams_and_epsilon(self):
        before = capture_rng()
        left, ls, lc = self.make("bicubic")
        right, rs, rc = self.make("interp23")
        self.assertTrue(_same(before, capture_rng()))
        for role in ("aligner", "teacher"):
            self.assertEqual(state_dict_sha256(getattr(left.stage, role)),
                             state_dict_sha256(getattr(right.stage, role)))
        self.assertTrue(_same(ls.state_dict(), rs.state_dict()))
        self.assertEqual(lc["stream_seeds"], rc["stream_seeds"])
        lb, rb = ls.next_update()[0], rs.next_update()[0]
        self.assertEqual(lb.lineage, rb.lineage)
        self.assertTrue(torch.equal(lb.ms, rb.ms))
        lo, ro = left.stage(lb), right.stage(rb)
        self.assertTrue(torch.equal(lo["epsilon_rr"], ro["epsilon_rr"]))
        self.assertTrue(torch.equal(lo["abs_rr"], ro["abs_rr"]))
        self.assertTrue(torch.equal(lo["eps_rr"], ro["eps_rr"]))
        self.assertNotEqual(left.stage.pipeline_manifest(), right.stage.pipeline_manifest())

    def test_nonzero_teacher_uses_explicit_base_and_preserves_aligner_stored_lms(self):
        batch = self.random_batch()
        for upsampler in ("bicubic", "interp23"):
            trainer, _, _ = self.make(upsampler)
            stage = trainer.stage
            self.randomize_heads(stage)
            correction = stage.aligner(batch.pan, batch.native_lms)
            pan = warp_pan(batch.pan, correction)
            base = base_from_batch(batch, upsampler)
            expected = reconstruct_with_base(stage.teacher, pan, base, batch.ms)
            self.assertGreater(float((expected["y"] - base).detach().abs().max()), 1e-5)
            with patch.object(stage.teacher, "forward", side_effect=AssertionError("implicit base used")), \
                 patch.object(stage.aligner, "forward", wraps=stage.aligner.forward) as aligner:
                actual = stage.native_prediction(batch)
            self.assertIs(aligner.call_args.args[0], batch.pan)
            self.assertIs(aligner.call_args.args[1], batch.native_lms)
            for key in ("y", "ms_base", "x_in", "pan_aligned"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            if upsampler == "bicubic":
                old = stage.teacher(pan, batch.ms, batch.native_lms)
                torch.testing.assert_close(actual["y"], old["y"], rtol=0, atol=0)
            else:
                old = stage.teacher(pan, batch.ms, batch.native_lms)
                self.assertGreater(float((actual["y"] - old["y"]).detach().abs().mean()), 1e-3)

    def test_train_eval_parity_with_nonzero_heads_and_unchanged_loss(self):
        batch = self.random_batch()
        for upsampler in ("bicubic", "interp23"):
            trainer, _, _ = self.make(upsampler)
            self.randomize_heads(trainer.stage)
            trainer.stage.train()
            train = trainer.stage.native_prediction(batch)
            trainer.stage.eval()
            with torch.no_grad():
                inference = trainer.stage.native_prediction(batch)
            torch.testing.assert_close(train["y"], inference["y"], rtol=0, atol=0)
            out = trainer.stage(batch)
            expected_rec = masked_mean((out["native"]["y"] - batch.gt).abs(), out["loss_mask"])
            torch.testing.assert_close(out["rec"], expected_rec, rtol=0, atol=0)
            torch.testing.assert_close(out["loss"], out["rec"] + 2 * out["abs_rr"] + .003 * out["eps_rr"],
                                       rtol=0, atol=0)

    def test_interpolation_is_post_augmentation_with_phase_compensation(self):
        batch = self.random_batch()
        for hflip in (False, True):
            for vflip in (False, True):
                for rotation in range(4):
                    augmented = augment_edge_batch(batch, hflip=hflip, vflip=vflip, rot90=rotation)
                    actual = base_from_batch(augmented, "interp23")
                    expected = upsample_lrms_after_augmentation(augmented.ms,
                                                               augmented.lineage.augmentations)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if rotation == 0 and not hflip and not vflip:
                        self.assertGreater(float((actual - F.interpolate(batch.ms, scale_factor=4,
                            mode="bicubic", align_corners=False)).abs().mean()), 1e-3)

    def test_both_arms_exact_next_update_after_complete_resume(self):
        for upsampler in ("bicubic", "interp23"):
            trainer, stream, cfg = self.make(upsampler)
            advance_stage1(trainer, stream)
            before = capture_rng()
            receipt = save_stage1(self.root / (upsampler + ".pt"), trainer, stream, **PINS)
            self.assertTrue(_same(before, capture_rng()))
            expected = advance_stage1(trainer, stream)
            restored, restored_stream, payload = load_stage1(receipt["checkpoint_path"],
                receipt["file_sha256"], cfg, self.registered.sources, **PINS)
            actual = advance_stage1(restored, restored_stream)
            self.assertTrue(_same(actual, expected))
            self.assertTrue(_same(restored.state_dict(), trainer.state_dict()))
            self.assertTrue(_same(restored_stream.state_dict(), stream.state_dict()))
            self.assertTrue(payload["campaign_resume_complete"])
            self.assertEqual(payload["model_state_sha256"], payload["state_hash"])

    def test_same_checkpoint_pair_import_and_effective_base_binding(self):
        for upsampler in ("bicubic", "interp23"):
            trainer, stream, _ = self.make(upsampler)
            advance_stage1(trainer, stream)
            path = self.root / (upsampler + "_candidate.pt")
            receipt = save_stage1(path, trainer, stream, evaluation=evaluation(trainer), **PINS)
            before = capture_rng()
            donor = load_donor(path, receipt["file_sha256"], self.registered)
            self.assertTrue(_same(before, capture_rng()))
            donor.assert_frozen()
            for role in ("aligner", "teacher"):
                self.assertEqual(state_dict_sha256(getattr(donor, role)),
                                 state_dict_sha256(getattr(trainer.stage, role)))
            self.assertEqual(donor.binding["warp30_teacher"],
                             {"upsampler": upsampler, "campaign_id": CAMPAIGN_ID})
            with self.assertRaises(FileExistsError):
                save_stage1(path, trainer, stream, **PINS)

    def test_legacy_opposite_upsampler_wrong_sha_and_source_are_rejected(self):
        trainer, stream, cfg = self.make()
        advance_stage1(trainer, stream)
        path = self.root / "candidate.pt"
        receipt = save_stage1(path, trainer, stream, evaluation=evaluation(trainer), **PINS)
        original = torch.load(path, map_location="cpu", weights_only=False)
        with self.assertRaisesRegex(ValueError, "SHA"):
            load_donor(path, "0" * 64, self.registered)
        with self.assertRaisesRegex(ValueError, "resume mismatch"):
            load_stage1(path, receipt["file_sha256"], config("bicubic"), self.registered.sources, **PINS)
        mutations = [
            lambda p: p.update(schema="DONOR30_CHECKPOINT_V1"),
            lambda p: p["config"].update(upsampler="bicubic"),
            lambda p: p["trainer"]["stage"]["_extra_state"]["sources"].update(rr_manifest_sha256="0" * 64),
            lambda p: p["trainer"]["stage"]["teacher.backbone.output.2.bias"].add_(1.),
            lambda p: p["evaluation"]["fr_state"].update(completed_update=999),
        ]
        for i, mutate in enumerate(mutations):
            altered = deepcopy(original)
            mutate(altered)
            changed = self.root / f"invalid_{i}.pt"
            torch.save(altered, changed)
            with self.assertRaises((ValueError, RuntimeError)):
                load_donor(changed, file_sha(changed), self.registered)

    def test_resume_only_or_incomplete_evaluation_cannot_be_a_donor(self):
        trainer, stream, _ = self.make()
        advance_stage1(trainer, stream)
        path = self.root / "resume_only.pt"
        receipt = save_stage1(path, trainer, stream, **PINS)
        with self.assertRaisesRegex(ValueError, "official"):
            load_donor(path, receipt["file_sha256"], self.registered)
        record = evaluation(trainer)
        record["qualification"]["metrics_complete"] = False
        with self.assertRaises(ValueError):
            save_stage1(self.root / "bad_candidate.pt", trainer, stream, evaluation=record, **PINS)


if __name__ == "__main__":
    unittest.main()
