import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from ablr2.plan import sensor_spec
from ablr2 import evaluation as ev


def record(sensor='QB', update=1010):
    spec = sensor_spec(sensor)
    rr_rows = [dict(ergas=1., scc=.8, psnr=35., sam=2., ssim=.9, rmse=3., cc=.95,
                    **{f'q{spec.num_bands}': .85}) for _ in range(20)]
    rr = dict(ev._summary(rr_rows, ev.rr_keys(spec)), sensor=sensor, num_bands=spec.num_bands,
              max_dn=2047, n_scenes=20, crop='20:-21', q_block=32, official_complete=True)
    fr_rows = [dict(hqnr=.98 * .99, d_lambda=.02, d_s=.01) for _ in range(20)]
    fr = dict(ev._summary(fr_rows, ev.FR_KEYS), sensor=sensor, num_bands=spec.num_bands,
              max_dn=2047, reference='native_PAN', support='full512', masking=False,
              aggregation='mean_per_scene_HQNR', hqnr_variant='raw-original', official_complete=True,
              jqm=.8, jqm_status='measured', jqm_variant=ev.JQM_VARIANT,
              signed_ds=ev.signed_ds_details(np.full((20, spec.num_bands), .21), np.full((20, spec.num_bands), .2), [.01] * 20))
    return dict(update=update, checkpoint_identity=dict(update=update, model_sha256='a' * 64),
                rr=rr, fr=fr, val_ergas=1.2, eval_mode='IDENTITY')


class EvaluationTests(unittest.TestCase):
    def test_sensor_band_and_dn_guard(self):
        from types import SimpleNamespace
        self.assertEqual(ev._spec(SimpleNamespace(sensor='GF2',num_bands=4,max_dn=1023)).max_dn,1023)
        for spec in (SimpleNamespace(sensor='GF2', num_bands=4, max_dn=2047),
                     SimpleNamespace(sensor='WV3', num_bands=4, max_dn=2047)):
            with self.assertRaises(ValueError): ev._spec(spec)

    def test_wv3_band_evidence_is_required(self):
        with self.assertRaises(ValueError): ev.canonical_band_indices(sensor_spec('WV3'))
        spec = replace(sensor_spec('WV3'), band_order=('NIR2', 'RE', 'R', 'Y', 'G', 'B', 'Coastal', 'NIR1'))
        self.assertEqual(ev.canonical_band_indices(spec), (6, 5, 4, 3, 2, 1, 7, 0))

    def test_rr_actual_support_and_real_band_dimension(self):
        for sensor, bands in (('WV3', 8), ('QB', 4), ('GF2',4)):
            base = np.arange(256 * 256, dtype=np.float64).reshape(1, 1, 256, 256) / 100 + 1
            gt = np.broadcast_to(base, (20, bands, 256, 256))
            pred = gt + 1
            def q(truth, fused, block, shift):
                self.assertEqual(truth.shape, (215, 215, bands))
                self.assertEqual((block, shift), (32, 32))
                return (.75, None)
            with patch.object(ev, 'q2n', side_effect=q), patch.object(ev, 'scc_dlpan', return_value=.5), \
                    patch.object(ev, 'ssim_skimage', return_value=.6), patch.object(ev, 'sam', return_value=.7):
                result = ev.rr_metrics(pred, gt, sensor)
            self.assertEqual(result[f'q{bands}'], .75)
            self.assertNotIn('q4' if bands == 8 else 'q8', result)
            self.assertAlmostEqual(result['rmse'], 1.)
            self.assertEqual(result['n_scenes'], 20)
            self.assertEqual(result['max_dn'], sensor_spec(sensor).max_dn)

    def test_rr19_and_nan_excluded_border_are_rejected(self):
        a = np.zeros((19, 4, 256, 256), dtype=np.float32)
        with self.assertRaises(ValueError): ev.rr_metrics(a, a, 'QB')
        a = np.zeros((20, 4, 256, 256), dtype=np.float32)
        a[0, 0, 0, 0] = np.nan
        with self.assertRaises(FloatingPointError): ev.rr_metrics(a, a, 'QB')

    def test_signed_ds_reconstruction_and_aggregation(self):
        for sensor in ('QB', 'WV3'):
            item = record(sensor)
            ev.validate_signed_ds(item['fr'])
            item['fr']['per_scene'][0]['hqnr'] = .123
            with self.assertRaises(ValueError): ev.validate_signed_ds(item['fr'])

    def test_fr_keeps_native_full_support_and_canonical_mtf_order(self):
        spec = replace(sensor_spec('WV3'), band_order=('Coastal', 'B', 'G', 'Y', 'R', 'RE', 'NIR1', 'NIR2'))
        engine = ev.FRMetrics.__new__(ev.FRMetrics)
        engine.spec, engine.order, engine.inverse_order = spec, tuple(range(8)), np.arange(8)
        engine.wald, engine.wald_sha256 = object(), None
        engine.lms = np.broadcast_to(np.array(1.), (20, 512, 512, 8))
        engine.pan = np.broadcast_to(np.array(2.), (20, 512, 512))
        engine.reference = [[.2] * 8 for _ in range(20)]
        pred = np.broadcast_to(np.array(3.), (20, 8, 512, 512))
        with patch('tools.metrics.eval_fr.mtf_filter', side_effect=lambda data, sensor, ratio, wald: data), \
                patch('tools.metrics.eval_fr._blockproc_uqi', return_value=.21), patch.object(ev, 'q2n', return_value=(.98, None)):
            result = engine(pred)
        self.assertFalse(result['masking'])
        self.assertEqual(result['support'], 'full512')
        self.assertEqual(result['reference'], 'native_PAN')
        self.assertAlmostEqual(result['hqnr'], .98 * .99)
        self.assertEqual(result['signed_ds']['operand_shape'], [20, 8])


if __name__ == '__main__': unittest.main()
