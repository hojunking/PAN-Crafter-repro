"""CPU mathematical and architecture contracts; no dataset or experiment run."""

from dataclasses import replace
import itertools
import math
import unittest

import torch
from torch import nn

from model.pancrafter_paper import ChannelLayerNorm, ResBlock
from pa.aligner import PANGlobalAligner
from pa.offset import sample_offsets
from pa.warp import warp_pan
from panda_method.geometry import (
    SceneAwarePANAligner, ScenePriorRecord, ScenePriorRegistry,
    VerifiedGridTransform, convert_prior_to_grid, transform_prior,
)


def example_record(scene="scene-a", shift=(1., 2.)):
    return ScenePriorRecord(
        source_scene_id=scene, prior_shift_dy_dx=shift, prior_grid_name="native_pan",
        pixel_scale=(1., 1.), origin_phase=(0., 0.), n_windows=20, n_valid=18,
        inlier_fraction=.9, direction_concentration=.95, dispersion=.2, confidence=.8,
        estimator_version="synthetic-test-v1", reference_type="native_lms",
        scene_input_hash="synthetic-only-hash", mapping_rule="synthetic PAN sampling dy/dx",
        accepted=True, acceptance_evidence="synthetic test fixture only; not real-scene approval",
    )


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_init_prior_mixed_scenes_and_no_prior_gradient(self):
        model = SceneAwarePANAligner(4, view_margin=0)
        prior = torch.tensor([[1., 2.], [-3., 4.]], requires_grad=True)
        correction, details = model(torch.randn(2, 1, 16, 20), torch.randn(2, 4, 16, 20),
                                    prior, return_details=True)
        torch.testing.assert_close(correction, prior, rtol=0, atol=0)
        self.assertEqual(tuple(correction.shape), (2, 2))
        for name in ("prior", "direction", "normal"):
            self.assertFalse(details[name].requires_grad)
        correction.square().sum().backward()
        self.assertIsNone(prior.grad)
        self.assertGreater(model.fc2.weight.grad.abs().sum().item(), 0)
        # Zero-initialized head intentionally gates earlier features on first backward.
        self.assertEqual(model.fc1.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.res.out_layers[-1].weight.grad.abs().sum().item(), 0)

    def test_resblock_exact_class_identity_init_and_stems_unchanged(self):
        model = SceneAwarePANAligner(8, view_margin=0)
        legacy = PANGlobalAligner(8)
        self.assertIs(type(model.res), ResBlock)
        self.assertIsInstance(model.res.in_layers[0], ChannelLayerNorm)
        self.assertIsInstance(model.res.out_layers[0], ChannelLayerNorm)
        self.assertIsNone(model.res.mod)
        self.assertIsNone(model.res.se)
        self.assertEqual(model.res.out_layers[2].p, 0)
        self.assertEqual(model.res.in_layers[2].padding, (1, 1))
        self.assertIsNotNone(model.res.in_layers[2].bias)
        x = torch.randn(2, 64, 5, 7)  # Includes negative values: no post-add SiLU.
        torch.testing.assert_close(model.res(x, None), x, rtol=0, atol=0)
        self.assertEqual(sum(isinstance(m, nn.GroupNorm) for m in model.modules()), 4)
        for name in ("pan_stem", "ms_stem", "joint1", "joint2"):
            new, old = getattr(model, name), getattr(legacy, name)
            self.assertEqual(str(new), str(old))
            new.load_state_dict(old.state_dict(), strict=True)
            channels = old[1].in_channels
            image = torch.randn(2, channels, 16, 16)
            torch.testing.assert_close(new(image), old(image), rtol=0, atol=0)
        self.assertEqual(model.fc1.in_features, 66)
        self.assertEqual(model.fc1.out_features, 32)
        self.assertEqual(model.fc2.out_features, 2)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 105522)
        # No implicit migration can load the legacy GN residual/head state.
        with self.assertRaises(RuntimeError):
            model.load_state_dict(legacy.state_dict(), strict=True)

    def test_head_receives_raw_signed_magnitude_not_unit_direction(self):
        model = SceneAwarePANAligner(4, view_margin=0)
        prior = torch.tensor([[1., 0.], [3., 0.], [-3., 0.], [2., -7.]],
                             requires_grad=True)
        pan = torch.randn(1, 1, 16, 16).repeat(4, 1, 1, 1)
        lms = torch.randn(1, 4, 16, 16).repeat(4, 1, 1, 1)
        seen = []
        hook = model.fc1.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        try:
            correction, details = model(pan, lms, prior, return_details=True)
        finally:
            hook.remove()
        self.assertEqual(tuple(seen[0].shape), (4, 66))
        torch.testing.assert_close(seen[0][:, -2:], prior, rtol=0, atol=0)
        torch.testing.assert_close(seen[0][0, :64], seen[0][1, :64], rtol=0, atol=0)
        torch.testing.assert_close(details["direction"][0], details["direction"][1], rtol=0, atol=0)
        self.assertFalse(torch.equal(seen[0][0, -2:], seen[0][1, -2:]))
        # Magnitude conditioning is present even while FC2 intentionally yields
        # zero residual. Zero-initialized alpha/beta need not differ yet.
        torch.testing.assert_close(correction, prior, rtol=0, atol=0)
        torch.testing.assert_close(details["alpha"], torch.zeros(4, 1), rtol=0, atol=0)
        torch.testing.assert_close(details["beta"], torch.zeros(4, 1), rtol=0, atol=0)
        torch.testing.assert_close(details["head_input"], seen[0], rtol=0, atol=0)
        self.assertEqual(tuple(details["fc1_preactivation"].shape), (4, 32))
        self.assertEqual(tuple(details["head_hidden"].shape), (4, 32))

    def test_nonzero_head_can_respond_to_magnitude_without_prior_gradients(self):
        model = SceneAwarePANAligner(4, view_margin=0)
        # A controlled head isolates both signed prior coordinates from image
        # features. This fails for the old direction-only concat despite equal
        # parameter counts and FC tensor shapes.
        with torch.no_grad():
            model.fc1.weight.zero_()
            model.fc1.bias.zero_()
            model.fc1.weight[0, 64] = 1.
            model.fc1.weight[1, 65] = 1.
            model.fc2.weight.zero_()
            model.fc2.weight[0, 0] = 1.
            model.fc2.weight[1, 1] = 1.
        prior = torch.tensor([[1., 0.], [3., 0.], [-3., 0.], [2., -7.]],
                             requires_grad=True)
        pan = torch.randn(1, 1, 16, 16).repeat(4, 1, 1, 1)
        lms = torch.randn(1, 4, 16, 16).repeat(4, 1, 1, 1)
        correction, details = model(pan, lms, prior, return_details=True)
        expected = torch.nn.functional.silu(prior.detach())
        torch.testing.assert_close(details["alpha"], expected[:, :1])
        torch.testing.assert_close(details["beta"], expected[:, 1:])
        self.assertNotEqual(details["alpha"][0].item(), details["alpha"][1].item())
        torch.testing.assert_close(
            correction, prior.detach() + expected[:, :1] * details["direction"]
            + expected[:, 1:] * details["normal"])
        correction.square().sum().backward()
        self.assertIsNone(prior.grad)
        self.assertGreater(model.fc1.weight.grad[:, -2:].abs().sum().item(), 0)
        for name in ("prior", "direction", "normal"):
            self.assertFalse(details[name].requires_grad)

    def test_current_grid_augmented_prior_reaches_head_and_output_identically(self):
        model = SceneAwarePANAligner(4, view_margin=0)
        original = torch.tensor([[4., 12.]], requires_grad=True)
        transform = VerifiedGridTransform(
            "native_pan", "rr_pan", ((.25, 0.), (0., .25)), (.5, -.5),
            "synthetic known differential phase and sampling-grid transform")
        rr_prior = convert_prior_to_grid(original, source_grid="native_pan", transform=transform)
        torch.testing.assert_close(rr_prior, torch.tensor([[1.5, 2.5]]))
        for hflip, vflip, k in itertools.product((False, True), (False, True), range(4)):
            current = transform_prior(rr_prior, hflip=hflip, vflip=vflip, rot90=k)
            with self.subTest(hflip=hflip, vflip=vflip, rot90=k):
                correction, details = model(torch.randn(1, 1, 16, 16),
                                            torch.randn(1, 4, 16, 16), current,
                                            return_details=True)
                torch.testing.assert_close(details["head_input"][:, -2:], current, rtol=0, atol=0)
                torch.testing.assert_close(details["prior"], current, rtol=0, atol=0)
                torch.testing.assert_close(correction, current, rtol=0, atol=0)
                self.assertFalse(details["prior"].requires_grad)

    def test_epsilon_forward_keeps_exact_same_prior_conditioning(self):
        model = SceneAwarePANAligner(4, view_margin=0)
        prior = torch.tensor([[3., -2.], [-1., 4.]])
        epsilon = torch.tensor([[1., .5], [-.5, -1.]])
        pan, lms = torch.randn(2, 1, 16, 16), torch.randn(2, 4, 16, 16)
        _, native = model(pan, lms, prior, return_details=True)
        _, shifted = model(warp_pan(pan, epsilon), lms, prior, return_details=True)
        torch.testing.assert_close(native["head_input"][:, -2:], prior, rtol=0, atol=0)
        torch.testing.assert_close(shifted["head_input"][:, -2:], prior, rtol=0, atol=0)

    def test_conditioning_revision_round_trip_including_parent_and_hash(self):
        from panda_method.calibration import state_dict_sha256

        source = SceneAwarePANAligner(4, view_margin=0)
        target = SceneAwarePANAligner(4, view_margin=0)
        target.load_state_dict(source.state_dict())
        self.assertEqual(state_dict_sha256(source), state_dict_sha256(target))
        marker = source.state_dict()["_conditioning_revision"]
        self.assertEqual(marker.dtype, torch.int64)
        self.assertEqual(marker.shape, torch.Size([]))
        self.assertEqual(marker.item(), 13)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in source.state_dict().values()))
        self.assertEqual(source.conditioning_manifest()["conditioning"],
                         "raw_signed_current_pan_grid_dy_dx")
        self.assertEqual(source.conditioning_manifest()["input_order"][-2:],
                         ["prior_dy", "prior_dx"])
        parent, restored = nn.Sequential(source), nn.Sequential(target)
        restored.load_state_dict(parent.state_dict(), strict=False)
        self.assertEqual(state_dict_sha256(parent), state_dict_sha256(restored))

    def test_legacy_same_shape_checkpoint_rejected_even_nested_non_strict(self):
        for nested, strict in itertools.product((False, True), (False, True)):
            for replacement in (None, torch.tensor(12), torch.tensor(13.),
                                torch.tensor([13]), torch.tensor(True)):
                with self.subTest(nested=nested, strict=strict, replacement=replacement):
                    target_aligner = SceneAwarePANAligner(4, view_margin=0)
                    source_aligner = SceneAwarePANAligner(4, view_margin=0)
                    target = nn.Sequential(target_aligner) if nested else target_aligner
                    source = nn.Sequential(source_aligner) if nested else source_aligner
                    state = source.state_dict()
                    key = ("0." if nested else "") + "_conditioning_revision"
                    if replacement is None:
                        del state[key]  # Exact old v1.2 state keys/shapes.
                    else:
                        state[key] = replacement
                    before = {name: value.clone() for name, value in target.state_dict().items()}
                    with self.assertRaisesRegex(RuntimeError, "conditioning revision"):
                        target.load_state_dict(state, strict=strict)
                    for name, value in target.state_dict().items():
                        torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_large_prior_not_clamped_and_nonfinite_head_fails(self):
        model = SceneAwarePANAligner(4, view_margin=0)
        pan, lms = torch.randn(1, 1, 8, 8), torch.randn(1, 4, 8, 8)
        prior = torch.tensor([[1e20, -2e20]])
        correction, details = model(pan, lms, prior, return_details=True)
        torch.testing.assert_close(correction, prior, rtol=0, atol=0)
        torch.testing.assert_close(details["head_input"][:, -2:], prior, rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(details["fc1_preactivation"]).all()))
        with torch.no_grad():
            model.fc1.weight[0, 64] = 2.
        with self.assertRaisesRegex(ValueError, "FC1 produced nonfinite"):
            model(pan, lms, torch.tensor([[torch.finfo(torch.float32).max, 1.]]))

    def test_output_no_clamp_and_cartesian_round_trip(self):
        model = SceneAwarePANAligner(4, view_margin=0)
        with torch.no_grad():
            model.fc2.bias.copy_(torch.tensor([10., -8.]))
        prior = torch.tensor([[3., 4.], [-2., 1.]])
        correction, details = model(torch.randn(2, 1, 12, 12), torch.randn(2, 4, 12, 12),
                                    prior, return_details=True)
        expected = prior + 10 * details["direction"] - 8 * details["normal"]
        torch.testing.assert_close(correction, expected)
        self.assertGreater(correction.abs().max().item(), 2)
        delta = correction - prior
        torch.testing.assert_close((delta * details["direction"]).sum(-1),
                                   details["alpha"].flatten())
        torch.testing.assert_close((delta * details["normal"]).sum(-1),
                                   details["beta"].flatten())

    def test_later_gradient_routes_reach_shared_resblock(self):
        model = SceneAwarePANAligner(4, view_margin=0)
        with torch.no_grad():
            model.fc2.weight.normal_(std=.01)
        correction = model(torch.randn(2, 1, 16, 16), torch.randn(2, 4, 16, 16),
                           torch.tensor([[1., 2.], [-2., 3.]]))
        correction.square().sum().backward()
        self.assertGreater(model.res.out_layers[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.pan_stem[1].weight.grad.abs().sum().item(), 0)
        # The zero conv still gates its residual branch's first conv initially.
        self.assertEqual(model.res.in_layers[-1].weight.grad.abs().sum().item(), 0)

    def test_lms_is_used_and_view_margin_only_crops_regression(self):
        model = SceneAwarePANAligner(4, view_margin=2)
        seen = []
        hook = model.ms_stem.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        pan, lms = torch.randn(1, 1, 16, 20), torch.randn(1, 4, 16, 20)
        pan_copy, lms_copy = pan.clone(), lms.clone()
        prior = torch.tensor([[1., 2.]])
        model(pan, lms, prior)
        changed = lms.clone()
        changed[..., 4:8, 4:8] += 1
        model(pan, changed, prior)
        hook.remove()
        self.assertEqual(tuple(seen[0].shape), (1, 4, 12, 16))
        self.assertFalse(torch.equal(seen[0], seen[1]))
        torch.testing.assert_close(pan, pan_copy, rtol=0, atol=0)
        torch.testing.assert_close(lms, lms_copy, rtol=0, atol=0)

    def test_invalid_priors_inputs_and_missing_crop_policy(self):
        with self.assertRaises(TypeError):
            SceneAwarePANAligner(4)
        model = SceneAwarePANAligner(4, view_margin=0)
        pan, lms = torch.randn(1, 1, 8, 8), torch.randn(1, 4, 8, 8)
        for prior in (torch.zeros(1, 2), torch.tensor([[float("nan"), 1.]]),
                      torch.tensor([[float("inf"), 1.]]), torch.ones(2),
                      torch.tensor([[torch.finfo(torch.float32).tiny / 2, 0.]])):
            with self.assertRaises(ValueError):
                model(pan, lms, prior)
        with self.assertRaises(ValueError):
            model(pan, lms[:, :3], torch.ones(1, 2))
        with self.assertRaises(ValueError):
            SceneAwarePANAligner(4, view_margin=3)(pan, lms, torch.ones(1, 2))
        with self.assertRaises(ValueError):
            model(pan * float("nan"), lms, torch.ones(1, 2))

    def test_warp_sign_and_all_flip_rot90_impulse_equivariances(self):
        pan = torch.zeros(1, 1, 31, 35, dtype=torch.float64)
        pan[0, 0, 12, 18] = 1.
        prior = torch.tensor([[2., 3.]], dtype=torch.float64)
        corrected = warp_pan(pan, prior)
        peak = torch.nonzero(corrected[0, 0] == corrected.max())[0]
        self.assertEqual(peak.tolist(), [10, 15])  # positive dx moves content left
        for hflip, vflip, k in itertools.product((False, True), (False, True), range(4)):
            with self.subTest(hflip=hflip, vflip=vflip, rot90=k):
                def augment(image):
                    if hflip:
                        image = image.flip(-1)
                    if vflip:
                        image = image.flip(-2)
                    return torch.rot90(image, k, (-2, -1))
                transformed = transform_prior(prior, hflip=hflip, vflip=vflip, rot90=k)
                torch.testing.assert_close(warp_pan(augment(pan), transformed), augment(corrected),
                                           rtol=0, atol=2e-14)
        torch.testing.assert_close(transform_prior(prior, rot90=1),
                                   torch.tensor([[-3., 2.]], dtype=prior.dtype))

    def test_augmentation_prior_detach(self):
        prior = torch.tensor([[2., 3.]], requires_grad=True)
        self.assertFalse(transform_prior(prior, rot90=1).requires_grad)

    def test_registry_mixed_scene_exact_lookup_and_provenance_validation(self):
        registry = ScenePriorRegistry([example_record(), example_record("scene-b", (-1., 3.))])
        torch.testing.assert_close(registry.lookup(["scene-b", "scene-a", "scene-b"],
                                                   grid_name="native_pan"),
                                   torch.tensor([[-1., 3.], [1., 2.], [-1., 3.]]))
        with self.assertRaises(KeyError):
            registry.lookup(["unmapped-patch-index-20"], grid_name="native_pan")
        with self.assertRaises(ValueError):
            registry.lookup(["scene-a"], grid_name="rr_pan")
        with self.assertRaises(ValueError):
            ScenePriorRegistry([example_record(), example_record()])
        for changes in ({"accepted": False}, {"acceptance_evidence": ""},
                        {"scene_input_hash": ""}, {"prior_shift_dy_dx": (0., 0.)},
                        {"prior_shift_dy_dx": (float("nan"), 1.)}, {"n_valid": 0},
                        {"confidence": 2.}, {"pixel_scale": (0., 1.)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(example_record(), **changes)

    def test_prior_and_transform_fingerprints_bind_all_metadata(self):
        record = example_record()
        self.assertEqual(record.fingerprint(), example_record().fingerprint())
        self.assertNotEqual(record.fingerprint(), replace(record, origin_phase=(.5, 0.)).fingerprint())
        self.assertNotEqual(record.fingerprint(), replace(record, prior_shift_dy_dx=(2., 1.)).fingerprint())
        self.assertNotEqual(record.fingerprint(), replace(record, acceptance_evidence="new review").fingerprint())
        other = example_record("scene-b")
        self.assertEqual(ScenePriorRegistry([record, other]).fingerprint(),
                         ScenePriorRegistry([other, record]).fingerprint())
        self.assertNotEqual(ScenePriorRegistry([record]).fingerprint(),
                            ScenePriorRegistry([other]).fingerprint())
        transform = VerifiedGridTransform("a", "b", ((.25, 0.), (0., .25)), (0., 0.),
                                          "synthetic geometry proof")
        self.assertNotEqual(transform.fingerprint(), replace(transform, phase_dy_dx=(.5, 0.)).fingerprint())

    def test_explicit_grid_and_phase_transform(self):
        transform = VerifiedGridTransform(
            source_grid="native_pan", target_grid="rr_pan",
            matrix_dy_dx=((.25, 0.), (0., .25)), phase_dy_dx=(.5, -.5),
            evidence="synthetic known-scale + known differential phase only",
        )
        prior = torch.tensor([[4., 8.]], requires_grad=True)
        result = convert_prior_to_grid(prior, source_grid="native_pan", transform=transform)
        torch.testing.assert_close(result, torch.tensor([[1.5, 1.5]]))
        self.assertFalse(result.requires_grad)
        with self.assertRaises(ValueError):
            convert_prior_to_grid(prior, source_grid="unverified", transform=transform)
        with self.assertRaises(TypeError):
            VerifiedGridTransform(source_grid="a", target_grid="b", evidence="no phase/matrix")
        with self.assertRaises(ValueError):
            replace(transform, evidence="")
        with self.assertRaises(ValueError):
            replace(transform, matrix_dy_dx=((0., 0.), (0., 0.)))
        # A phase conversion producing an undefined zero-direction prior must fail.
        with self.assertRaises(ValueError):
            convert_prior_to_grid(torch.tensor([[-2., 2.]]), source_grid="native_pan",
                                  transform=transform)

    def test_uniform_disk_statistics_at_fixed_method_radius(self):
        # Six theoretical standard errors: this is a sampler smoke tolerance,
        # not a registration/performance acceptance threshold.
        n = 200000
        epsilon = sample_offsets(n, 2., torch.Generator(device="cpu").manual_seed(12345))
        radius = torch.linalg.vector_norm(epsilon.double(), dim=-1)
        self.assertEqual(epsilon.device.type, "cpu")
        self.assertEqual(tuple(epsilon.shape), (n, 2))
        self.assertLessEqual(float(radius.max()), 2. + 2 * torch.finfo(torch.float32).eps)
        axis_mean = epsilon.double().mean(0)
        self.assertTrue(bool((axis_mean.abs() < 6. / math.sqrt(n)).all()))
        self.assertLess(abs(float(radius.mean()) - 4. / 3.), 6 * math.sqrt(2. / (9. * n)))
        # Disk radius density f(sqrt(2))=1/sqrt(2): median SE=1/sqrt(2n).
        self.assertLess(abs(float(radius.median()) - math.sqrt(2.)), 6. / math.sqrt(2. * n))
        # E[x^2]=1, E[x^4]=2 for R=2, so SE(mean(x^2))=1/sqrt(n).
        axis_second_moment = epsilon.double().square().mean(0)
        self.assertTrue(bool(((axis_second_moment - 1.).abs() < 6. / math.sqrt(n)).all()))
        area_quantile = radius.square() / 4.
        for quantile in (.25, .5, .75):
            fraction = float((area_quantile <= quantile).double().mean())
            self.assertLess(abs(fraction - quantile),
                            6 * math.sqrt(quantile * (1 - quantile) / n))

    def test_dedicated_cpu_generator_independence_fresh_draws_and_replay(self):
        generator = torch.Generator(device="cpu").manual_seed(912)
        global_state = torch.random.get_rng_state().clone()
        first = sample_offsets(7, 2., generator)
        torch.testing.assert_close(torch.random.get_rng_state(), global_state, rtol=0, atol=0)
        checkpoint_state = generator.get_state().clone()
        second = sample_offsets(7, 2., generator)
        third = sample_offsets(7, 2., generator)
        self.assertFalse(torch.equal(first, second))
        self.assertFalse(torch.equal(second, third))
        # Data-loader/shuffle/global RNG traffic must not change the epsilon stream.
        torch.rand(501)
        replay = torch.Generator(device="cpu")
        replay.set_state(checkpoint_state)
        torch.testing.assert_close(sample_offsets(7, 2., replay), second, rtol=0, atol=0)
        torch.testing.assert_close(sample_offsets(7, 2., replay), third, rtol=0, atol=0)
        torch.random.set_rng_state(global_state)

    def test_axis16_fixed_no_response_reference(self):
        epsilon = torch.tensor([[radius * dy, radius * dx]
                                for radius in (.25, .5, 1., 2.)
                                for dy, dx in ((1., 0.), (-1., 0.), (0., 1.), (0., -1.))])
        model = SceneAwarePANAligner(4, view_margin=0)
        pan, lms = torch.randn(16, 1, 16, 16), torch.randn(16, 4, 16, 16)
        prior = torch.tensor([[1., 2.]]).expand(16, -1)
        with torch.no_grad():
            native = model(pan, lms, prior)
            shifted = model(warp_pan(pan, epsilon), lms, prior)
        delta = shifted - native
        gain = -(delta * epsilon).sum() / epsilon.square().sum()
        mae = (delta + epsilon).abs().mean()
        epe = torch.linalg.vector_norm(delta + epsilon, dim=-1).mean()
        self.assertEqual(float(gain), 0.)
        self.assertEqual(float(mae), .46875)
        self.assertEqual(float(epe), .9375)
        ideal_delta = -epsilon
        self.assertEqual(float(-(ideal_delta * epsilon).sum() / epsilon.square().sum()), 1.)
        self.assertEqual(float((ideal_delta + epsilon).abs().mean()), 0.)


if __name__ == "__main__":
    unittest.main()
