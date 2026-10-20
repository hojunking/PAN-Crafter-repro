"""Synthetic CPU proof obligations, never official experimental evidence."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from directshift48.method import DirectTrainingSources, ScharrAbsolute
from donor30.artifacts import _same, capture_rng
from donor30.masking import masked_mean
from panda_method.calibration import state_dict_sha256
from panda_method.edge_data import augment_edge_batch
from panda_method.reconstruction import ReconstructionSpec
from panda_midpoint.resampling import upsample_lrms_after_augmentation
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_warp30.plan import resolve_case as old_resolve
from panda_warp30.tests.test_stage1 import PINS, evaluation
from ta2.evaluation import file_sha

from panda_band30.aligner import (BandPANAligner, AllBandScharr, GLOBAL_TARGET, BAND_TARGET,
                                  aligned_pan_views, bandwise_warp, support_for_shifts)
from panda_band30.contract import CAMPAIGN_ID, METHOD_REVISION
from panda_band30.donor import load_donor
from panda_band30.reconstruction import (build_reconstructor, base_from_batch,
                                         reconstruct_with_base)
from panda_band30.stage1 import (BandStage1, BandStage1Trainer, Stage1Stream, advance_stage1,
    build_stage1, save_stage1, load_stage1, sample_band_delta, band_ms_equivariance, AUX_POLICY)


def config(outputs=2, upsampler="interp23"):
    cfg = old_resolve("s3", "S1_U23" if upsampler == "interp23" else "S1_BIC")
    cfg.update(campaign_id=CAMPAIGN_ID, method_revision=METHOD_REVISION, server="s1",
               aligner_outputs=outputs, pan_channels=outputs//2,
               alignment_target=GLOBAL_TARGET if outputs == 2 else BAND_TARGET,
               band_augmentation="NONE" if outputs == 2 else AUX_POLICY,
               run_id=f"SYNTHETIC_BAND30_NOT_AN_EXPERIMENT_{outputs}_{upsampler}")
    cfg["stream_seeds"]["rr_band_augmentation"] = 261019117
    cfg["baseline_resolved"].update(effective_batch_RR=1, microbatch_RR=1, accumulation_RR=1)
    return cfg


class BandStage1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registered = synthetic_sources(self.root, count=3)

    def batch(self, count=1):
        batch = self.registered.batch([i % 3 for i in range(count)])
        g = torch.Generator().manual_seed(1001)
        return replace(batch, **{key: torch.rand(getattr(batch, key).shape, generator=g)*1.4-.7
                                 for key in ("pan", "ms", "native_lms", "gt")})

    @staticmethod
    def nonzero(stage):
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(81)
            stage.aligner.fc2.weight.normal_(0, .01)
            stage.aligner.fc2.bias.normal_(0, .04)
            stage.teacher.backbone.output[-1].weight.normal_(0, .01)
            stage.teacher.backbone.output[-1].bias.normal_(0, .02)

    def make(self, outputs=2, upsampler="interp23"):
        cfg = config(outputs, upsampler)
        return (build_stage1(cfg, self.registered.sources),
                Stage1Stream(self.registered.sources, cfg, device="cpu"), cfg)

    def small_stage(self, outputs=16):
        teacher = build_reconstructor(ReconstructionSpec(8, 8, (1, 2, 3), "P0", "BICUBIC_MS"),
                                      seed=15, pan_channels=outputs//2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            aligner = BandPANAligner(outputs)
        stage = BandStage1(aligner, teacher, upsampler="interp23",
            sources=DirectTrainingSources("WV3", self.registered.sources.manifest_sha("train")),
            epsilon_seed=17, band_augmentation_seed=19, lambda_abs=2., lambda_eps=.003)
        self.nonzero(stage)
        return stage

    def test_global_mean8_and_bandwise_objectives_are_explicit_not_equivalent(self):
        batch = self.batch()
        p = batch.pan
        descriptor = ScharrAbsolute("WV3").descriptor
        expected = (descriptor(p)-descriptor(batch.gt.mean(1, keepdim=True)))[..., 8:-8, 8:-8].abs().mean()
        torch.testing.assert_close(AllBandScharr(2).absolute_loss(p, batch.gt), expected)
        band = AllBandScharr(16).absolute_loss(p.expand(-1, 8, -1, -1), batch.gt)
        target = torch.stack([(descriptor(p)-descriptor(batch.gt[:, i:i+1]))[..., 8:-8, 8:-8].abs().mean()
                              for i in range(8)]).mean()
        torch.testing.assert_close(band, target)
        # Opposite spectral edges cancel before Scharr but not afterwards.
        opposite = p.expand(-1, 8, -1, -1) * p.new_tensor([1, -1]*4)[None, :, None, None]
        zero = torch.zeros_like(p)
        self.assertEqual(float(AllBandScharr(2).absolute_loss(zero, opposite)), 0.)
        self.assertGreater(float(AllBandScharr(16).absolute_loss(zero.expand_as(opposite), opposite)), .01)

    def test_split_stem_equal_views_and_all_eight_nonzero_routes(self):
        spec = ReconstructionSpec(8, 8, (1, 2, 3), "P0", "BICUBIC_MS")
        one = build_reconstructor(spec, seed=3, pan_channels=1)
        eight = build_reconstructor(spec, seed=3, pan_channels=8)
        batch = self.batch()
        base = base_from_batch(batch, "interp23")
        with torch.no_grad():
            one.backbone.output[-1].weight.normal_(0, .01)
            eight.backbone.output[-1].load_state_dict(one.backbone.output[-1].state_dict())
        y1 = reconstruct_with_base(one, batch.pan, base, batch.ms)["y"]
        y8 = reconstruct_with_base(eight, batch.pan.expand(-1, 8, -1, -1), base, batch.ms)["y"]
        self.assertGreater(float((y1-base).abs().max()), 1e-4)
        torch.testing.assert_close(y8, y1, atol=2e-6, rtol=2e-6)
        count = sum(p.numel() for p in eight.parameters()) - sum(p.numel() for p in one.parameters())
        self.assertEqual(count, 7*8*3*3)
        u = torch.zeros(1, 8, 2)
        u[:, 5, 1] = 1
        views = aligned_pan_views(batch.pan, u)
        self.assertTrue(torch.equal(views[:, 0], views[:, 7]))
        self.assertFalse(torch.equal(views[:, 0], views[:, 5]))
        x = torch.cat((views, base), 1).requires_grad_()
        eight.backbone.input(x).square().mean().backward()
        self.assertTrue(bool((x.grad[:, :8].abs().sum((0, 2, 3)) > 0).all()))
        self.assertEqual(eight.spec.input_channels, 16)
        self.assertEqual(sum(p.numel() for p in BandPANAligner(16).parameters()) -
                         sum(p.numel() for p in BandPANAligner(2).parameters()), 462)

    def test_nonzero_heads_loss_aux_only_one_u_and_train_inference_parity(self):
        stage, batch = self.small_stage(), self.batch()
        native = stage.native_prediction(batch)
        self.assertEqual(native["u"].shape, (1, 8, 2))
        with torch.no_grad():
            stage.eval()
            inference = stage.native_prediction(batch)
        torch.testing.assert_close(inference["y"], native["y"], atol=0, rtol=0)
        with patch.object(stage.teacher.backbone, "forward", wraps=stage.teacher.backbone.forward) as unet, \
             patch("panda_band30.stage1.base_from_batch", wraps=base_from_batch) as base:
            out = stage(batch)
        self.assertEqual(unet.call_count, 1)
        self.assertEqual(base.call_count, 1)
        self.assertIs(base.call_args.args[0].ms, batch.ms)
        aux = stage._auxiliary_records[-1]
        self.assertAlmostEqual(float(out["abs_rr"]), .5*(aux["abs_native"]+aux["abs_ms_aux"]), places=6)
        self.assertAlmostEqual(float(out["eps_rr"]), .5*(aux["eps_pan"]+aux["eq_ms"]), places=6)
        torch.testing.assert_close(out["rec"], masked_mean((out["native"]["y"]-batch.gt).abs(), out["loss_mask"]))
        torch.testing.assert_close(out["loss"], out["rec"]+2*out["abs_rr"]+.003*out["eps_rr"])

    def test_ms_equivariance_sign_and_partition_independent_delta(self):
        left, right = torch.Generator().manual_seed(8), torch.Generator().manual_seed(8)
        whole = sample_band_delta(48, left)
        split = torch.cat([sample_band_delta(12, right) for _ in range(4)])
        self.assertTrue(torch.equal(whole, split))
        self.assertLessEqual(float(whole.norm(dim=-1).max()), 2)
        self.assertLessEqual(float((whole[:, :, None]-whole[:, None, :]).norm(dim=-1).max()), 1)
        native = torch.randn(48, 8, 2, requires_grad=True)
        self.assertLess(float(band_ms_equivariance(native.detach()+whole, native, whole)), 1e-7)
        self.assertGreater(float(band_ms_equivariance(native.detach()-whole, native, whole)), .1)
        augmented = (native.detach()+whole+.1).requires_grad_()
        band_ms_equivariance(augmented, native, whole).backward()
        self.assertIsNone(native.grad)
        self.assertIsNotNone(augmented.grad)

    def test_paired_lms_gt_routing_and_support_intersection(self):
        batch = self.batch()
        delta = torch.zeros(1, 8, 2)
        delta[:, 6, 0] = 2
        lms = bandwise_warp(batch.native_lms, delta)
        gt = bandwise_warp(batch.gt, delta)
        torch.testing.assert_close(lms[:, 6, :-2], batch.native_lms[:, 6, 2:], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(gt[:, 6, :-2], batch.gt[:, 6, 2:], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lms[:, 3], batch.native_lms[:, 3], atol=1e-5, rtol=1e-5)
        all_masks = support_for_shifts(64, 64, delta, intersection=False)
        self.assertTrue(torch.equal(support_for_shifts(64, 64, delta), all_masks.all(1, keepdim=True)))

    def test_full48_micro12_gradient_loss_and_pre_step_rollback(self):
        stage = self.small_stage()
        other = deepcopy(stage)
        cfg = dict(gradient_log_interval=100, effective_batch_size=48)
        full = BandStage1Trainer(stage, torch.optim.AdamW(stage.parameters(), lr=1e-4),
                                 microbatch_size=48, **cfg)
        micro = BandStage1Trainer(other, torch.optim.AdamW(other.parameters(), lr=1e-4),
                                  microbatch_size=12, **cfg)
        batch = self.batch(48)
        # Typed slicing preserves sample identity/augmentation metadata.
        pieces = []
        for start in range(0, 48, 12):
            lineage = replace(batch.lineage,
                sample_ids=batch.lineage.sample_ids[start:start+12],
                augmentations=batch.lineage.augmentations[start:start+12])
            pieces.append(replace(batch, lineage=lineage, **{k: getattr(batch, k)[start:start+12]
                               for k in ("pan", "ms", "native_lms", "gt")}))
        a, b = full.step(batch), micro.step_accumulated(pieces)
        for key in ("loss", "rec", "abs_rr", "eps_rr"):
            self.assertAlmostEqual(a[key], b[key], places=6)
        self.assertTrue(torch.equal(stage._band_rng.get_state(), other._band_rng.get_state()))
        self.assertTrue(torch.equal(stage._epsilon_rng.get_state(), other._epsilon_rng.get_state()))
        for p, q in zip(stage.parameters(), other.parameters()):
            torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5)
        before = deepcopy(micro.state_dict())
        valid = other.forward
        calls = [0]
        def fail_second(rr):
            calls[0] += 1
            if calls[0] == 2:
                raise ValueError("synthetic pre-step failure")
            return valid(rr)
        with patch.object(other, "forward", side_effect=fail_second), self.assertRaises(ValueError):
            micro.step_accumulated(pieces)
        self.assertTrue(_same(before, micro.state_dict()))

    def test_fresh_paired_global_upsamplers_and_all16_augmented_bases(self):
        left, _, lc = self.make(2, "interp23")
        right, _, rc = self.make(2, "bicubic")
        for role in ("aligner", "teacher"):
            self.assertEqual(state_dict_sha256(getattr(left.stage, role)),
                             state_dict_sha256(getattr(right.stage, role)))
        batch = self.batch()
        for hf in (False, True):
            for vf in (False, True):
                for rot in range(4):
                    aug = augment_edge_batch(batch, hflip=hf, vflip=vf, rot90=rot)
                    torch.testing.assert_close(base_from_batch(aug, "interp23"),
                        upsample_lrms_after_augmentation(aug.ms, aug.lineage.augmentations), atol=0, rtol=0)

    def test_global_and_band_complete_resume_and_strict_evaluated_donor(self):
        for outputs in (2, 16):
            trainer, stream, cfg = self.make(outputs)
            advance_stage1(trainer, stream)
            path = self.root / f"candidate_{outputs}.pt"
            receipt = save_stage1(path, trainer, stream, evaluation=evaluation(trainer), **PINS)
            expected = advance_stage1(trainer, stream)
            restored, rs, _ = load_stage1(path, receipt["file_sha256"], cfg, self.registered.sources, **PINS)
            actual = advance_stage1(restored, rs)
            self.assertTrue(_same(expected, actual))
            self.assertTrue(_same(trainer.state_dict(), restored.state_dict()))
            before = capture_rng()
            donor = load_donor(path, receipt["file_sha256"], self.registered)
            self.assertTrue(_same(before, capture_rng()))
            donor.assert_frozen()
            self.assertEqual(donor.binding["band30_teacher"]["aligner_outputs"], outputs)
            self.assertEqual(donor.teacher.spec.input_channels, 8+outputs//2)
            with self.assertRaises(ValueError):
                load_donor(path, "0"*64, self.registered)
            payload = torch.load(path, weights_only=False)
            payload["trainer"]["stage"]["_extra_state"]["sources"]["rr_manifest_sha256"] = "0"*64
            changed = self.root / f"changed_{outputs}.pt"
            torch.save(payload, changed)
            with self.assertRaises((ValueError, RuntimeError)):
                load_donor(changed, file_sha(changed), self.registered)

    def test_invalid_predicted_band_support_raises_without_rng_consumption(self):
        stage, batch = self.small_stage(), self.batch()
        with torch.no_grad():
            stage.aligner.fc2.bias[14] = 12
        before = deepcopy(stage.get_extra_state())
        with self.assertRaisesRegex(ValueError, "SUPPORT_VIOLATION"):
            stage(batch)
        self.assertTrue(_same(before, stage.get_extra_state()))

    def test_checkpoint_reads_ignore_atime_only_changes(self):
        trainer, stream, cfg = self.make()
        advance_stage1(trainer, stream)
        path = self.root / "atime_candidate.pt"
        saved = save_stage1(path, trainer, stream, evaluation=evaluation(trainer), **PINS)
        original_stat, counter = Path.stat, [0]

        class ReadAccessStat:
            def __init__(self, stat, tick):
                self.stat, self.tick = stat, tick

            def __getattr__(self, name):
                if name in ("st_atime", "st_atime_ns"):
                    return getattr(self.stat, name) + self.tick
                return getattr(self.stat, name)

            def __eq__(self, other):
                return isinstance(other, ReadAccessStat) and self.stat == other.stat and self.tick == other.tick

        def changing_atime(value, *args, **kwargs):
            stat = original_stat(value, *args, **kwargs)
            if str(value) == str(path):
                counter[0] += 1
                return ReadAccessStat(stat, counter[0])
            return stat

        with patch.object(Path, "stat", changing_atime):
            loaded, _, _ = load_stage1(path, saved["file_sha256"], cfg, self.registered.sources, **PINS)
            donor = load_donor(path, saved["file_sha256"], self.registered)
        self.assertGreater(counter[0], 4)
        self.assertTrue(_same(loaded.state_dict(), trainer.state_dict()))
        donor.assert_frozen()

    def test_rr256_fr512_gt_free_inference_no_auxiliary_rng_or_scale_change(self):
        for outputs in (2, 16):
            stage = self.small_stage(outputs).eval()
            source = self.batch()
            # Constant nonzero head proves inference does not divide/multiply by four.
            with torch.no_grad():
                stage.aligner.fc2.weight.zero_()
                stage.aligner.fc2.bias.fill_(.3)
            for resolution, side in (("RR", 256), ("FR", 512)):
                lineage = replace(source.lineage, split="inference", resolution=resolution,
                                  grid_name="PAN_"+resolution)
                batch = replace(source, pan=torch.rand(1, 1, side, side),
                    ms=torch.rand(1, 8, side//4, side//4),
                    native_lms=torch.rand(1, 8, side, side), gt=None, lineage=lineage)
                before = deepcopy(stage.get_extra_state())
                with torch.no_grad():
                    value = stage.native_prediction(batch)
                self.assertEqual(value["y"].shape, (1, 8, side, side))
                torch.testing.assert_close(value["u"], torch.full_like(value["u"], .3), atol=0, rtol=0)
                self.assertTrue(_same(before, stage.get_extra_state()))
                with self.assertRaises(ValueError):
                    stage(batch)


if __name__ == "__main__":
    unittest.main()
