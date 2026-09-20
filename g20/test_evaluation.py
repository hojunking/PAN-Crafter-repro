import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from g20.evaluation import (FRMetrics, RR_KEYS, infer, rr_metrics, signed_ds_details,
                            validate_signed_ds)
from g20.plan import sensor_spec
from tools.metrics.q2n import q2n


def record(step, h=.965, e=.54, dl=.01):
    ds = 1 - h / (1 - dl)
    rr = dict.fromkeys(RR_KEYS, .9)
    rr.update(ergas=e, scc=.9, psnr=30, n_scenes=20, crop='20:-21', q_block=32,
              official_complete=True, sensor='GF2', num_bands=4, max_dn=1023)
    scenes = [dict(hqnr=h, d_lambda=dl, d_s=ds) for _ in range(20)]
    fr = dict(hqnr=h, d_lambda=dl, d_s=ds, n_scenes=20, per_scene=scenes,
              reference='native_PAN', support='full512', masking=False,
              aggregation='mean_per_scene_HQNR', hqnr_variant='raw-original',
              sensor='GF2', num_bands=4, max_dn=1023, official_complete=True)
    fr['signed_ds'] = signed_ds_details(np.full((20, 4), .5 + ds), np.full((20, 4), .5), [ds] * 20)
    return dict(update=step, rr=rr, fr=fr, val_ergas=e + .1,
                checkpoint_identity=dict(update=step, model_sha256=f'{step:064x}'))


class EvaluationTests(unittest.TestCase):
    def test_real_q4_and_sensor_range(self):
        rng = np.random.default_rng(9)
        gt = np.broadcast_to(rng.uniform(40, 900, (1, 4, 256, 256)), (20, 4, 256, 256))
        with patch('g20.evaluation.q2n', wraps=q2n) as metric:
            out = rr_metrics(gt + 2, gt, 'GF2')
        self.assertEqual(metric.call_count, 20)
        self.assertEqual(metric.call_args.args[0].shape, (215, 215, 4))
        self.assertNotIn('q8', out)
        self.assertAlmostEqual(out['rmse'], 2)
        self.assertAlmostEqual(out['cc'], 1)
        self.assertAlmostEqual(out['psnr'], 20 * np.log10(1023 / 2))
        with self.assertRaises(ValueError):
            rr_metrics(gt, gt, 'QB')

    def test_signed_ds_exact_20x4_operands_and_reconstruction(self):
        low = np.full((20, 4), .5)
        delta = np.tile([-.2, -.1, .1, .4], (20, 1))
        detail = signed_ds_details(low + delta, low, np.abs(delta).mean(1))
        self.assertEqual(np.asarray(detail['Q_high']).shape, (20, 4))
        np.testing.assert_allclose(detail['delta'], delta)
        self.assertAlmostEqual(detail['signed_mean'], .05)
        self.assertAlmostEqual(detail['abs_mean'], .2)
        self.assertEqual(detail['positive_fraction'], .5)
        self.assertEqual(detail['per_band']['positive_fraction'], [0, 0, 1, 1])
        self.assertTrue(detail['reconstruction_verified'])
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            signed_ds_details(low + delta, low, np.full(20, .7))
        with self.assertRaises(ValueError):
            signed_ds_details(low[:1], low[:1], [.2])

    def test_fr_actual_q_operands_native_pan_not_ncc(self):
        engine = FRMetrics.__new__(FRMetrics)
        engine.spec = replace(sensor_spec(), band_order=('R', 'G', 'B', 'NIR'))
        engine.order, engine.inverse_order = (2, 1, 0, 3), np.array([2, 1, 0, 3])
        engine.wald = object()
        engine.lms = np.broadcast_to(np.arange(4), (20, 512, 512, 4))
        engine.pan = np.broadcast_to(np.arange(512)[None, None, :], (20, 512, 512))
        engine.reference = np.full((20, 4), .5).tolist()
        pred = np.broadcast_to(np.arange(4)[None, :, None, None], (20, 4, 512, 512))
        highs = np.tile([.3, .4, .6, .9], (20, 1))
        highs[10:] = [.4, .45, .55, .7]
        calls = []
        def uqi(band, pan, block):
            self.assertEqual(block, 32)
            np.testing.assert_array_equal(pan, engine.pan[len(calls) // 4])
            value = highs.flat[len(calls)]
            calls.append((band.shape, value))
            return value
        def mtf(a, preset, ratio, wald):
            self.assertEqual(preset, 'gf2')
            self.assertEqual(a[0, 0].tolist(), [2, 1, 0, 3])
            return a
        with patch('tools.metrics.eval_fr._blockproc_uqi', side_effect=uqi), \
             patch('tools.metrics.eval_fr.mtf_filter', side_effect=mtf), \
             patch('g20.evaluation.q2n', side_effect=[(.9, None)] * 10 + [(.7, None)] * 10):
            out = engine(pred)
        self.assertEqual(len(calls), 80)
        np.testing.assert_array_equal(out['signed_ds']['Q_high'], highs)
        np.testing.assert_array_equal(out['signed_ds']['Q_low'], engine.reference)
        self.assertAlmostEqual(out['d_s'], .15)
        self.assertAlmostEqual(out['hqnr'], (.9 * .8 + .7 * .9) / 2)
        self.assertNotAlmostEqual(out['hqnr'], (1 - out['d_s']) * (1 - out['d_lambda']))
        self.assertEqual(out['mtf_gnyq_canonical'], [.3] * 4)
        self.assertFalse(out['masking'])
        self.assertEqual(out['support'], 'full512')
        validate_signed_ds(out)

    def test_signed_ds_tamper_and_raw_hqnr_aggregation_rejected(self):
        original = record(50000)['fr']
        validate_signed_ds(original)
        for mutate in (lambda r: r['signed_ds']['delta'][0].__setitem__(0, 0),
                       lambda r: r['signed_ds'].__setitem__('protocol', 'NCC'),
                       lambda r: r.__setitem__('d_s', .9),
                       lambda r: r['per_scene'][0].__setitem__('hqnr', .1)):
            bad = copy.deepcopy(original)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_signed_ds(bad)

    def test_infer_clips_at_gf2_dn_boundary_and_restores_mode(self):
        class Dataset:
            spec = sensor_spec()
            def __len__(self): return 1
            def __getitem__(self, index):
                return (torch.zeros(4, 8, 8), torch.zeros(4, 2, 2),
                        torch.zeros(1, 2, 2), torch.zeros(1, 8, 8), torch.tensor([0, 0]))
        class Model(torch.nn.Module):
            def forward(self, pan, ms, lp):
                return dict(y=torch.tensor([-2., -1., 1., 2.])[None, :, None, None].expand(1, 4, 8, 8),
                            delta=torch.zeros(1, 2))
        model = Model().train()
        sr, _ = infer(model, Dataset(), 'cpu')
        self.assertEqual(sr[0, :, 0, 0].tolist(), [0., 0., 1023., 1023.])
        self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
