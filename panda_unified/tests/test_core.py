import unittest
import torch
from torch.nn import functional as F

from panda_unified.core import (
    BilinearUpConv, ReferenceSpatial, adaptive_weight, build_reconstructor,
    erode_scharr_support, inject, masked_mean, reconstruct_with_base,
    reconstruction_manifest, reference_maps, support, warp_pan, warp_support_mask,
)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_near_integer_mask_matches_actual_gather_indices(self):
        shifts = torch.tensor([[-1e-6, -1e-6], [.5, .5], [1-1e-6, 2+1e-6],
                               [-4., 3.], [4.5, -5.25]], dtype=torch.float32)
        actual = warp_support_mask(64, 64, shifts)
        expected = torch.zeros_like(actual)
        for n, (dy, dx) in enumerate(torch.floor(shifts).to(torch.int64)):
            for y in range(64):
                for x in range(64):
                    expected[n, 0, y, x] = all(
                        0 <= y + dy + iy < 64 and 0 <= x + dx + ix < 64
                        for iy in (-3, 4) for ix in (-3, 4))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual[0].sum().item(), 57**2)
        self.assertEqual(actual[1].sum().item(), 57**2)
        # Reproduce the former FP32 floor(x+u) precision disagreement.
        xx = torch.arange(64, dtype=torch.float32) - 1e-6
        wrong = ((xx.floor()-3 >= 0) & (xx.floor()+4 < 64)).sum()
        self.assertEqual(wrong.item(), 56)

    def test_common_domain_is_maximal_full_scharr_stencil(self):
        pan = torch.zeros(2, 1, 64, 64)
        shifts = torch.tensor([[.5, .5], [-1e-6, -1e-6]], requires_grad=True)
        raw = warp_support_mask(64, 64, shifts)
        common = support(pan, shifts)
        self.assertFalse(common.requires_grad)
        self.assertEqual(common.flatten(1).sum(1).tolist(), [55**2, 55**2])
        counts = F.conv2d(raw.float(), torch.ones(1, 1, 3, 3), padding=1)
        self.assertTrue(torch.equal(common, counts == 9))
        self.assertTrue(torch.equal(common, erode_scharr_support(raw)))
        near_integer = torch.tensor([[1-1e-10, -1e-10]], dtype=torch.float64)
        # A FP32 PAN causes the actual warp to round the supplied FP64 shift.
        self.assertTrue(torch.equal(support(pan[:1], near_integer),
                                    support(pan[:1], near_integer.float())))

    def test_boundary_extension_cannot_change_common_scharr_values(self):
        gen = torch.Generator().manual_seed(7)
        pan = torch.rand(2, 1, 32, 32, generator=gen)
        shifts = torch.tensor([[-2.3, .8], [4.2, -5.1]])
        reflected = warp_pan(pan, shifts)
        outside_constant = F.pad(pan, (12, 12, 12, 12), value=17.)
        extended = warp_pan(outside_constant, shifts)[..., 12:-12, 12:-12]
        common = support(pan, shifts)
        edges = ReferenceSpatial('QB', 'RGBN')
        difference = (edges.descriptor(reflected)-edges.descriptor(extended)).abs()
        self.assertEqual(difference[common.expand_as(difference)].max().item(), 0.)
        self.assertGreater(difference[~common.expand_as(difference)].max().item(), 0.)

    def test_warp_remains_differentiable_in_shift(self):
        torch.manual_seed(5)
        image = torch.rand(1, 1, 12, 12, dtype=torch.float64)
        delta = torch.tensor([[.27, -.36]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda u: warp_pan(image, u),
                                                (delta,), atol=2e-5, rtol=1e-4))

    def test_no_roi8_shift_clamp_and_empty_support_rejected(self):
        pan = torch.zeros(1, 1, 64, 64)
        self.assertGreater(support(pan, torch.tensor([[10., -7.]])).sum().item(), 0)
        with self.assertRaisesRegex(ValueError, 'EMPTY_SUPPORT'):
            support(pan, torch.tensor([[100., 0.]]))
        with self.assertRaisesRegex(ValueError, 'clean input crop8'):
            inject(pan, torch.tensor([[10., 0.]]))
        self.assertTrue(torch.isfinite(inject(pan, torch.tensor([[2.9, -.2]]))).all())

    def test_reference_band_projection_is_explicit_for_both_sensors(self):
        for sensor, bands, rgb, nir in [('WV3', 8, (4, 2, 1), 6), ('QB', 4, (2, 1, 0), 3)]:
            image = torch.arange(bands, dtype=torch.float32)[None, :, None, None].expand(2, -1, 8, 8)
            for mode, indices in [('RGB', rgb), ('RGBN', rgb+(nir,))]:
                spatial = ReferenceSpatial(sensor, mode)
                expected = image[:, indices].mean(1, keepdim=True)
                self.assertTrue(torch.equal(spatial.reference(image), expected))
                self.assertTrue(torch.equal(spatial.rgb_luminance(image), expected))
                self.assertEqual(spatial.manifest()['reference_indices'], list(indices))

    def test_spatial_and_reconstruction_share_identical_mask_reduction(self):
        mask = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
        mask[0, :, 1:3, 1:3] = True
        mask[1, :, 1:7, 1:7] = True
        values = torch.zeros(2, 4, 8, 8)
        values[0] = 2; values[1] = 6
        self.assertEqual(masked_mean(values, mask).item(), 4.)
        pan = torch.rand(2, 1, 32, 32)
        gt = torch.rand(2, 4, 32, 32)
        domain = support(pan, torch.tensor([[.4, -.1], [.2, 1.5]]))
        spatial = ReferenceSpatial('QB', 'RGBN')
        error = (spatial.descriptor(pan)-spatial.descriptor(spatial.reference(gt))).abs()
        self.assertEqual(spatial.absolute_loss(pan, gt, mask=domain).item(),
                         masked_mean(error, domain).item())

    def test_adaptive_cues_cover_entire_common_domain_and_are_detached(self):
        pan = torch.zeros(1, 1, 64, 64)
        mask = support(pan, torch.tensor([[.5, .5]]))
        seen = []
        class Stats:
            tau_e = 1.
            def lookup_exact(self, g, q):
                seen.append((g.clone(), q.clone()))
                return torch.ones_like(q)
        e = torch.ones_like(pan, requires_grad=True)
        g, q = torch.full_like(pan, -999.), torch.full_like(pan, -999.)
        g[mask] = 2.; q[mask] = 3.
        weight = adaptive_weight(e, g, q, Stats(), mask, {'alpha': .25, 'gamma': .25, 'beta': 0})
        self.assertFalse(weight.requires_grad)
        self.assertEqual(seen[0][0].numel(), 55**2)
        self.assertTrue((seen[0][0] == 2).all())
        self.assertTrue((seen[0][1] == 3).all())
        self.assertTrue((weight[mask] == 1.25).all())
        self.assertTrue((weight[~mask] == 1).all())
        self.assertEqual(masked_mean(weight * torch.ones_like(pan), mask).item(), 1.25)
        with self.assertRaises(ValueError):
            adaptive_weight(e, g, q, Stats(), mask, {'alpha': .25, 'gamma': .25, 'beta': .1})

    def test_teacher_student_have_same_sampling_and_only_input_channels_differ(self):
        for sensor, width, bands in [('WV3', 112, 8), ('QB', 128, 4)]:
            teacher = build_reconstructor(sensor, width, 17, 'T')
            student = build_reconstructor(sensor, width, 17, 'S')
            for name in ('down1', 'down2', 'up1', 'up2'):
                self.assertIs(type(getattr(teacher.backbone, name)), type(getattr(student.backbone, name)))
            for model in (teacher, student):
                self.assertIs(type(model.backbone.up1), BilinearUpConv)
                self.assertEqual(reconstruction_manifest(model)['base_reference'], 'INTERP23_MS')
            self.assertEqual(teacher.backbone.input.in_channels, bands+1)
            self.assertEqual(student.backbone.input.in_channels, bands+2)
            self.assertEqual((sum(p.numel() for p in student.parameters())-
                              sum(p.numel() for p in teacher.parameters())), width*9)
            self.assertEqual(student.backbone.input.weight[:, 1].count_nonzero().item(), 0)
            ms, base = torch.rand(1, bands, 8, 8), torch.rand(1, bands, 32, 32)
            t = reconstruct_with_base(teacher, torch.rand(1, 1, 32, 32), base, ms)['y']
            s = reconstruct_with_base(student, torch.rand(1, 2, 32, 32), base, ms)['y']
            self.assertEqual(t.shape, base.shape)
            self.assertEqual(s.shape, base.shape)
            (t.mean()+s.mean()).backward()
            self.assertIsNotNone(teacher.backbone.up1.conv.weight.grad)
            self.assertIsNotNone(student.backbone.up1.conv.weight.grad)

    def test_checkpoint_identity_and_no_implicit_bicubic_forward(self):
        model = build_reconstructor('QB', 16, 4, 'T')
        restored = build_reconstructor('QB', 16, 9, 'T')
        restored.load_state_dict(model.state_dict(), strict=True)
        with self.assertRaisesRegex(RuntimeError, 'implicit bicubic'):
            model(None)
        broken = dict(model.state_dict())
        broken['_unified_identity'] = broken['_unified_identity'].clone()
        broken['_unified_identity'][0] ^= 1
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            restored.load_state_dict(broken, strict=True)

    def test_student_parameters_preserve_existing_named_fresh_ubil_initialization(self):
        from panda_tune30.reconstruction import (
            StudentReconstructionSpec as WV3Spec, build_reconstructor as old_wv3,
        )
        from panda_qb_s2.reconstruction import (
            StudentReconstructionSpec as QBSpec, build_reconstructor as old_qb,
        )
        for sensor, bands, width, cls, builder in [('WV3', 8, 112, WV3Spec, old_wv3), ('QB', 4, 128, QBSpec, old_qb)]:
            previous = builder(cls(bands, width, (1, 2, 3), 'P0', 'BICUBIC_MS', 2),
                               seed=77, role='S', sampling_variant='UBIL', input_variant='CORRECTED_DIFF')
            fresh = build_reconstructor(sensor, width, 77, 'S')
            a, b = previous.backbone.state_dict(), fresh.backbone.state_dict()
            self.assertEqual(a.keys(), b.keys())
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), (sensor, name))

    def test_teacher_error_uses_all_bands_while_edges_follow_reference_mode(self):
        gt = torch.zeros(1, 4, 16, 16)
        gt[:, 3, :, 8:] = 1.
        teacher = gt.clone(); teacher[:, 3] += 1
        pan = torch.zeros(1, 1, 16, 16)
        er, gr, qr = reference_maps(ReferenceSpatial('QB', 'RGB'), teacher, gt, pan)
        en, gn, qn = reference_maps(ReferenceSpatial('QB', 'RGBN'), teacher, gt, pan)
        self.assertTrue(torch.equal(er, en))
        self.assertTrue((er == .25).all())
        self.assertEqual(gr.sum().item(), 0)
        self.assertGreater(gn.sum().item(), 0)
        self.assertGreater(qn.sum().item(), qr.sum().item())


if __name__ == '__main__':
    unittest.main()
