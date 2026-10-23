"""Meaningful sampling, initialization, gradient, and checkpoint contracts."""
from copy import deepcopy
import unittest

import torch

from panda_band30.reconstruction import (
    BandReconstructionSpec, build_reconstructor as build_original,
    reconstruct_with_base,
)
from panda_next30.reconstruction import (
    SAMPLING_VARIANTS, AverageDownConv, BilinearUpConv, StudentReconstructionSpec,
    build_reconstructor, reconstruction_manifest, assert_sampling_contract,
)


def spec(width=8):
    return BandReconstructionSpec(8, width, (1, 2, 3), "P0", "BICUBIC_MS", 1)


def observations(height=32, width=32):
    generator = torch.Generator().manual_seed(873)
    return (torch.rand(1, 1, height, width, generator=generator),
            torch.rand(1, 8, height, width, generator=generator),
            torch.rand(1, 8, height // 4, width // 4, generator=generator))


def enable_backbone_gradients(model):
    # A fresh model's intentionally zero output head would mask sampler effects.
    with torch.no_grad():
        model.backbone.output[-1].weight.copy_(
            torch.randn(model.backbone.output[-1].weight.shape,
                        generator=torch.Generator().manual_seed(379)) * .01)


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_base_is_bit_exact_and_teacher_cannot_change(self):
        for role in ("S", "T"):
            old = build_original(spec(), seed=261018101, role=role)
            actual = build_reconstructor(spec(), seed=261018101, role=role)
            self.assertEqual(type(actual), type(old))
            self.assertEqual(actual.state_dict().keys(), old.state_dict().keys())
            for name, tensor in old.state_dict().items():
                self.assertTrue(torch.equal(actual.state_dict()[name], tensor), name)
            enable_backbone_gradients(old)
            enable_backbone_gradients(actual)
            for shape in ((32, 32), (32, 48)):
                args = observations(*shape)
                self.assertTrue(torch.equal(reconstruct_with_base(old, *args)["y"],
                                            reconstruct_with_base(actual, *args)["y"]))
        for variant in SAMPLING_VARIANTS[1:]:
            with self.assertRaises(ValueError):
                build_reconstructor(spec(), seed=7, role="T", sampling_variant=variant)

    def test_shared_weights_seed_role_and_rng_are_preserved(self):
        baseline = build_reconstructor(spec(), seed=7)
        state = baseline.state_dict()
        for variant in SAMPLING_VARIANTS[1:]:
            before = torch.random.get_rng_state().clone()
            changed = build_reconstructor(spec(), seed=7, sampling_variant=variant)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            duplicate = build_reconstructor(spec(), seed=7, sampling_variant=variant)
            other = build_reconstructor(spec(), seed=8, sampling_variant=variant)
            for name, tensor in changed.state_dict().items():
                self.assertTrue(torch.equal(tensor, duplicate.state_dict()[name]), name)
                if name in state and not name.startswith(("backbone.up1.", "backbone.up2.")):
                    self.assertTrue(torch.equal(tensor, state[name]), name)
            self.assertFalse(torch.equal(changed.backbone.input.weight, other.backbone.input.weight))
            if variant == "DAVG":
                for name in ("up1", "up2"):
                    self.assertTrue(torch.equal(getattr(changed.backbone, name).conv.weight,
                                                getattr(baseline.backbone, name).conv.weight))

    def test_shapes_gradients_and_actual_parameter_counts(self):
        for variant in SAMPLING_VARIANTS:
            model = build_reconstructor(spec(), seed=5, sampling_variant=variant)
            enable_backbone_gradients(model)
            for height, width in ((32, 32), (32, 48)):
                model.zero_grad(set_to_none=True)
                pan, base, ms = observations(height, width)
                pan.requires_grad_(True)
                result = reconstruct_with_base(model, pan, base, ms)["y"]
                self.assertEqual(result.shape, (1, 8, height, width))
                result.square().mean().backward()
                self.assertTrue(bool(torch.isfinite(pan.grad).all()))
                self.assertGreater(float(pan.grad.abs().sum()), 0.)
                for name in ("down1", "down2", "up2", "up1"):
                    grad = getattr(model.backbone, name).conv.weight.grad
                    self.assertTrue(bool(torch.isfinite(grad).all()), (variant, name))
                    self.assertGreater(float(grad.abs().sum()), 0., (variant, name))
            full = build_reconstructor(spec(112), seed=5, sampling_variant=variant)
            manifest = reconstruction_manifest(full)
            self.assertEqual(assert_sampling_contract(full, variant), manifest)
            expected = 2784328 if variant in ("UBIL", "DAVG_UBIL") else 2658888
            self.assertEqual(manifest["parameter_count"], expected)
            self.assertFalse(manifest["rotation_equivariance_guaranteed"])

    def test_same_architecture_resume_exact_and_cross_architecture_rejected(self):
        models = {variant: build_reconstructor(spec(), seed=5, sampling_variant=variant)
                  for variant in SAMPLING_VARIANTS}
        for variant, model in models.items():
            enable_backbone_gradients(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            args = observations()
            loss = reconstruct_with_base(model, *args)["y"].square().mean()
            loss.backward()
            optimizer.step()
            saved = deepcopy(model.state_dict())
            restored = build_reconstructor(spec(), seed=5, sampling_variant=variant)
            restored.load_state_dict(saved, strict=True)
            self.assertTrue(torch.equal(reconstruct_with_base(model, *args)["y"],
                                        reconstruct_with_base(restored, *args)["y"]))
            for other_variant in SAMPLING_VARIANTS:
                if other_variant != variant:
                    other = build_reconstructor(spec(), seed=5, sampling_variant=other_variant)
                    with self.assertRaises(RuntimeError, msg=(variant, other_variant)):
                        other.load_state_dict(saved, strict=True)
            if variant != "BASE":
                damaged = deepcopy(saved)
                damaged["_sampling30_identity"][0] ^= 1
                with self.assertRaises(RuntimeError):
                    restored.load_state_dict(damaged, strict=False)

    def test_difference_input_preserves_shared_initialization_and_adds_trainable_channel(self):
        from panda_next30.method import student_pan_inputs
        for variant in SAMPLING_VARIANTS:
            with self.subTest(variant=variant):
                baseline=build_reconstructor(spec(),seed=7,sampling_variant=variant)
                diff_spec=StudentReconstructionSpec(8,8,(1,2,3),'P0','BICUBIC_MS',2)
                before=torch.random.get_rng_state().clone()
                diff=build_reconstructor(diff_spec,seed=7,sampling_variant=variant,input_variant='CORRECTED_DIFF')
                self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
                base_weight=baseline.backbone.input.weight; diff_weight=diff.backbone.input.weight
                self.assertTrue(torch.equal(base_weight[:,:1],diff_weight[:,:1]))
                self.assertTrue(torch.equal(base_weight[:,1:],diff_weight[:,2:]))
                self.assertEqual(float(diff_weight[:,1].abs().sum()),0.)
                self.assertEqual(sum(p.numel() for p in diff.parameters())-sum(p.numel() for p in baseline.parameters()),8*9)
                enable_backbone_gradients(baseline); enable_backbone_gradients(diff)
                pan,base,ms=observations(); original=torch.flip(pan,(-1,))
                views=student_pan_inputs(pan,original,'CORRECTED_DIFF')
                expected=reconstruct_with_base(baseline,pan,base,ms)['y']
                actual=reconstruct_with_base(diff,views,base,ms)['y']
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                actual.square().mean().backward()
                self.assertGreater(float(diff.backbone.input.weight.grad[:,1].abs().sum()),0.)
                restored=build_reconstructor(diff_spec,seed=7,sampling_variant=variant,input_variant='CORRECTED_DIFF')
                restored.load_state_dict(diff.state_dict(),strict=True)
                assert_sampling_contract(restored,variant,input_variant='CORRECTED_DIFF')
                with self.assertRaises(ValueError):
                    assert_sampling_contract(restored,variant,input_variant='CORRECTED')
        full=build_reconstructor(StudentReconstructionSpec(8,112,(1,2,3),'P0','BICUBIC_MS',2),
                                 seed=7,input_variant='CORRECTED_DIFF')
        self.assertEqual(sum(p.numel() for p in full.parameters()),2658888+1008)

    def test_manifest_rejects_an_operator_that_does_not_match_the_config(self):
        down = build_reconstructor(spec(), seed=7, sampling_variant="DAVG")
        self.assertIsInstance(down.backbone.down1, AverageDownConv)
        down.backbone.down1.pool.ceil_mode = True
        with self.assertRaises(ValueError):
            reconstruction_manifest(down)
        up = build_reconstructor(spec(), seed=7, sampling_variant="UBIL")
        self.assertIsInstance(up.backbone.up1, BilinearUpConv)
        up.backbone.up1.conv.padding = (0, 0)
        with self.assertRaises(ValueError):
            reconstruction_manifest(up)
        other = build_reconstructor(spec(), seed=7, sampling_variant="DAVG_UBIL")
        with self.assertRaises(ValueError):
            assert_sampling_contract(other, "UBIL")
        other._sampling30_identity[0] ^= 1
        with self.assertRaises(RuntimeError):
            assert_sampling_contract(other, "DAVG_UBIL")


if __name__ == "__main__":
    unittest.main()
