import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qg40.evaluation import RR_KEYS, FRMetrics, fr_jqm, rr_metrics
from qg40.plan import CASES, GRID_STEPS, sensor_spec
from qg40.postrun import process, select_records
from qg40.common import atomic_json, object_sha, read_json
from tools.metrics.q2n import q2n


def record(step, sensor='QB', h=.921, e=3.5):
    spec = sensor_spec(sensor)
    rr = dict.fromkeys(RR_KEYS, .9)
    rr.update(ergas=e, scc=.9, psnr=30, n_scenes=20, crop='20:-21', q_block=32,
              official_complete=True, sensor=sensor, num_bands=4, max_dn=spec.max_dn)
    fr = dict(hqnr=h, d_lambda=.01, d_s=.02, n_scenes=20, reference='native_PAN', support='full512',
              masking=False, aggregation='mean_per_scene_HQNR', hqnr_variant='raw-original',
              sensor=sensor, num_bands=4, max_dn=spec.max_dn)
    return dict(update=step, rr=rr, fr=fr, val_ergas=e + .1,
                checkpoint_identity=dict(update=step, model_sha256=f'{step:064x}'))


class EvaluationTests(unittest.TestCase):
    def test_real_four_band_q2n(self):
        rng = np.random.default_rng(5)
        a = rng.integers(10, 900, (64, 64, 4)).astype(float)
        self.assertAlmostEqual(q2n(a, a, 32, 32)[0], 1, places=11)
        self.assertLess(q2n(a, np.roll(a, 2, axis=0), 32, 32)[0], .9)

    def test_rr_uses_q4_sensor_range_native_crop_and_all_band_extra(self):
        rng = np.random.default_rng(7)
        truth = rng.uniform(50, 900, (1, 4, 256, 256))
        pred = truth + 2
        # Broadcast avoids duplicating the fixture; evaluation sees the real full20 contract.
        truth = np.broadcast_to(truth, (20, 4, 256, 256))
        pred = np.broadcast_to(pred, truth.shape)
        with patch('qg40.evaluation.q2n', wraps=q2n) as real_q:
            gf = rr_metrics(pred, truth, 'GF2')
        self.assertEqual(real_q.call_count, 20)
        self.assertEqual(real_q.call_args.args[0].shape, (215, 215, 4))
        self.assertNotIn('q8', gf)
        self.assertIn('q4', gf)
        self.assertAlmostEqual(gf['rmse'], 2)
        self.assertAlmostEqual(gf['cc'], 1)
        self.assertAlmostEqual(gf['psnr'], 20 * np.log10(1023 / 2))
        self.assertEqual(gf['max_dn'], 1023)
        corrupted = np.array(pred)
        corrupted[0, 0, 0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            rr_metrics(corrupted, truth, 'GF2')

    def test_strict_sensor_targets_and_grid(self):
        rows = [record(step, h=.920) for step in GRID_STEPS]
        self.assertEqual(select_records(rows, 'QB')['target_status'], 'no_eligible')
        rows[1]['fr']['hqnr'] = .92000001
        rows[1]['rr']['ergas'] = 3.56999
        selected = select_records(rows, 'QB')
        self.assertEqual(selected['target']['update'], 2020)
        self.assertTrue(selected['joint_pass'])
        self.assertEqual(selected['exact50k']['update'], 50000)
        with self.assertRaises(ValueError):
            select_records(rows[:-1], 'QB')
        gf = [record(step, 'GF2', h=.964, e=.521) for step in GRID_STEPS]
        self.assertIsNone(select_records(gf, 'GF2')['target'])
        gf[-1]['fr']['hqnr'] = .96400001
        self.assertTrue(select_records(gf, 'GF2')['strong_joint_pass'])

    def test_jqm_uses_sensor_dn_and_explicit_native_band_order(self):
        spec = replace(sensor_spec('GF2'), band_order=('NIR', 'B', 'G', 'R'))
        sr = np.broadcast_to(np.arange(4)[None, :, None, None], (20, 4, 512, 512))
        ms = np.broadcast_to(np.arange(4)[None, :, None, None], (20, 4, 128, 128))
        pan = np.zeros((20, 1, 512, 512))
        def measure(fused, native_ms, native_pan, sensor, **kwargs):
            self.assertEqual(sensor, 'GF2')
            self.assertEqual(kwargs['R'], 1023)
            self.assertEqual(fused[0, 0].tolist(), [1, 2, 3, 0])
            self.assertEqual(native_ms[0, 0].tolist(), [1, 2, 3, 0])
            self.assertEqual(native_pan.shape, (512, 512))
            return dict(JQM=.9, QLR=.8, QHR=1., w=[.1, .2, .3, .4], w_source='nnls-normalized')
        with patch('tools.metrics.jqm.jqm', side_effect=measure) as jqm:
            result = fr_jqm(sr, ms, pan, spec)
        self.assertEqual(jqm.call_count, 20)
        self.assertEqual(result['per_scene'][0]['w'], [.4, .1, .2, .3])
        self.assertIn('not SIPSA-equivalent', result['variant'])

    def test_selector_ties_and_refuse_q8_or_masked_reference(self):
        rows = [record(step) for step in GRID_STEPS]
        rows[1]['rr']['scc'] = .91
        rows[2]['rr'].update(scc=.91, psnr=31)
        rows[3]['rr'].update(scc=.91, psnr=31)
        self.assertEqual(select_records(rows, 'QB')['target']['update'], 3030)
        self.assertEqual(select_records(rows, 'QB')['raw_max']['update'], 1010)
        for key, value in [('masking', True), ('reference', 'warped_PAN'), ('support', 'crop')]:
            bad = copy.deepcopy(rows)
            bad[0]['fr'][key] = value
            with self.assertRaises(ValueError):
                select_records(bad, 'QB')
        rows[0]['rr']['q8'] = rows[0]['rr'].pop('q4')
        with self.assertRaises(ValueError):
            select_records(rows, 'QB')

    def test_fr_mean_of_scene_hqnr_and_sensor_mtf(self):
        engine = FRMetrics.__new__(FRMetrics)
        engine.spec = replace(sensor_spec('QB'), band_order=('R', 'G', 'B', 'NIR'))
        engine.order, engine.inverse_order = (2, 1, 0, 3), np.array([2, 1, 0, 3])
        engine.wald = object()
        engine.lms = np.broadcast_to(np.arange(4), (20, 512, 512, 4))
        engine.pan = np.zeros((20, 512, 512))
        engine.reference = [[0.] * 4] * 20
        predictions = np.broadcast_to(np.arange(4)[None, :, None, None], (20, 4, 512, 512))
        qvals = iter([(.9, None)] * 10 + [(.7, None)] * 10)
        spatial = iter([.1] * 40 + [.3] * 40)
        def filter_(a, preset, ratio, wald):
            self.assertEqual(preset, 'qb')
            self.assertEqual(a[0, 0].tolist(), [2, 1, 0, 3])
            return a
        with patch('tools.metrics.eval_fr.mtf_filter', side_effect=filter_), patch('qg40.evaluation.q2n', side_effect=lambda *a: next(qvals)), patch('tools.metrics.eval_fr._blockproc_uqi', side_effect=lambda *a: next(spatial)):
            result = engine(predictions)
        self.assertAlmostEqual(result['hqnr'], (.9 * .9 + .7 * .7) / 2)
        self.assertNotAlmostEqual(result['hqnr'], (1-result['d_lambda']) * (1-result['d_s']))
        self.assertFalse(result['masking'])
        self.assertEqual(result['support'], 'full512')

    def test_pending_grid_returns_truthful_status_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = CASES[0].run_id
            wd = root / 'work_dir' / run
            cfg, release = {'qg40': {}}, {'content_sha256': 'release'}
            atomic_json(wd / 'meta/config.resolved.yaml', cfg)  # JSON is valid YAML.
            atomic_json(wd / 'meta/training_status.json', dict(actual_updates=50000, training_complete=True))
            atomic_json(wd / 'official/raw_grid.json', dict(records=[record(1010)], complete=False, source_identity=release))
            with patch('qg40.training.runtime_context', return_value=({}, '2999-01-01T00:00:00+00:00')), patch('qg40.postrun.validate_grid', return_value={}), patch('qg40.postrun.source_identity', return_value=release), patch('qg40.postrun.load_checkpoint_model') as load:
                self.assertEqual(process(run, root=root, device='cpu'), 0)
            status = read_json(wd / 'official/postrun_status.json')
            self.assertFalse(status['official_complete'])
            self.assertEqual(status['n_evaluated'], 1)
            self.assertEqual(status['n_pending'], 49)
            self.assertFalse((wd / 'official/target_selection.json').exists())
            load.assert_not_called()

    def test_pending_grid_repairs_only_missing_saved_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = CASES[0].run_id
            wd = root / 'work_dir' / run
            cfg, release = {'qg40': {}}, {'content_sha256': 'release'}
            atomic_json(wd / 'meta/config.resolved.yaml', cfg)
            atomic_json(wd / 'meta/training_status.json', dict(actual_updates=50000, training_complete=True))
            atomic_json(wd / 'official/raw_grid.json', dict(records=[record(s) for s in GRID_STEPS[:-1]],
                        complete=False, source_identity=release, data_sha256='data'))
            atomic_json(wd / 'official/profile.json', dict(config_sha256=object_sha(cfg), source_identity=release, num_bands=4))
            final = record(50000)
            atomic_json(wd / 'candidates/50000/identity.json', final['checkpoint_identity'])
            (wd / 'candidates/50000/model.safetensors').write_bytes(b'synthetic checkpoint placeholder')
            metrics = {k: v for k, v in final.items() if k not in ('update', 'checkpoint_identity')}
            with patch('qg40.training.runtime_context', return_value=({}, '2999-01-01T00:00:00+00:00')), patch('qg40.postrun.validate_grid', return_value={}), patch('qg40.postrun.source_identity', return_value=release), patch('qg40.postrun.load_checkpoint_model', return_value=(object(), final['checkpoint_identity'])) as load, patch('qg40.data.build_dataset', return_value=object()), patch('qg40.evaluation.FRMetrics', return_value=object()), patch('qg40.evaluation.evaluate_model', return_value=metrics) as evaluate:
                self.assertEqual(process(run, root=root, device='cpu'), 0)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(load.call_args.args[1].name, '50000')
            self.assertTrue(read_json(wd / 'official/postrun_status.json')['official_complete'])
            self.assertTrue(read_json(wd / 'official/raw_grid.json')['complete'])


if __name__ == '__main__':
    unittest.main()
