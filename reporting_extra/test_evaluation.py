"""CPU-only numerical and hostile-artifact tests; no live run or Sheet writes."""
from pathlib import Path
import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from fh12.common import atomic_json, object_sha, read_json, sha256
from fh12.postrun import report_selection, select_records
from reporting_extra import evaluation as ev


def metrics():
    rr = ev._summary([dict(scene_index=i, rmse=2. + i / 20, cc=.98) for i in range(20)], ('rmse', 'cc'))
    rr['crop'] = '20:-21'
    fr = ev._summary([dict(scene_index=i, jqm=.8, qlr=.7, qhr=.9, w=[.125] * 8,
                           w_source='nnls-normalized') for i in range(20)], ('jqm', 'qlr', 'qhr'))
    fr.update(masking=False, w_source='nnls-normalized')
    return dict(rr=rr, fr=fr)


class NumericTests(unittest.TestCase):
    def arrays(self):
        gt = np.broadcast_to(np.arange(8 * 256 * 256, dtype=np.float32).reshape(1, 8, 256, 256) % 1000,
                             (20, 8, 256, 256))
        return gt + np.arange(20, dtype=np.float32)[:, None, None, None], gt

    def test_rr_matches_existing_sheet_formula_and_sample_std(self):
        sr, gt = self.arrays(); result = ev.rr_metrics(sr, gt)
        # Deliberately spell the existing gspread_upload.py formula independently.
        p = sr.astype(np.float64).transpose(0, 2, 3, 1)[:, 20:-21, 20:-21]
        g = gt.astype(np.float64).transpose(0, 2, 3, 1)[:, 20:-21, 20:-21]
        rm = [float(np.sqrt(np.mean((a.ravel() - b.ravel()) ** 2))) for a, b in zip(p, g)]
        cc = [float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) for a, b in zip(p, g)]
        self.assertEqual(result['rmse'], np.mean(rm))
        self.assertEqual(result['cc'], np.mean(cc))
        self.assertEqual(result['standard_deviation']['rmse'], np.std(rm, ddof=1))
        self.assertEqual(result['rmse'], 9.5)  # mean scene RMSE, not pooled dataset RMSE

    def test_rr_crop_excludes_exact_asymmetric_border(self):
        sr, gt = self.arrays(); sr[:] = gt
        sr[:, :, :20] += 100; sr[:, :, -21:] += 100
        sr[:, :, :, :20] += 100; sr[:, :, :, -21:] += 100
        self.assertEqual(ev.rr_metrics(sr, gt)['rmse'], 0.)
        sr[:, :, 20, 20] += 1
        self.assertGreater(ev.rr_metrics(sr, gt)['rmse'], 0.)

    def test_rr_rejects_incomplete_nonfinite_and_constant_scenes(self):
        sr, gt = self.arrays()
        with self.assertRaises(ValueError): ev.rr_metrics(sr[:19], gt[:19])
        sr[0, 0, 0, 0] = np.nan  # even excluded pixels must be finite
        with self.assertRaises(FloatingPointError): ev.rr_metrics(sr, gt)
        a = np.broadcast_to(np.float32(10), (20, 8, 256, 256))
        with self.assertRaises(ValueError): ev.rr_metrics(a, a)

    def test_fr_uses_unshifted_native_reference_and_no_mask(self):
        sr = np.broadcast_to(np.float32(17), (20, 8, 512, 512))
        ms = np.broadcast_to(np.float32(13), (20, 8, 128, 128))
        pan = np.broadcast_to(np.float32(29), (20, 1, 512, 512))
        def fake(fused, native_ms, native_pan, sensor, **kw):
            self.assertEqual(fused.shape, (512, 512, 8)); self.assertTrue(np.all(fused == 17))
            self.assertEqual(native_ms.shape, (128, 128, 8)); self.assertTrue(np.all(native_ms == 13))
            self.assertEqual(native_pan.shape, (512, 512)); self.assertTrue(np.all(native_pan == 29))
            self.assertEqual(sensor, 'WV3')
            self.assertEqual(kw, dict(ratio=4, R=2047., lpf='mtf', window=None, v1=.5))
            return dict(JQM=.8, QLR=.7, QHR=.9, w=[.125] * 8, w_source='nnls-normalized')
        with patch.object(ev, 'jqm', side_effect=fake) as call:
            result = ev.fr_metrics(sr, ms, pan)
        self.assertEqual(call.call_count, 20)
        self.assertFalse(result['masking']); self.assertEqual(result['w_source'], 'nnls-normalized')
        self.assertEqual(result['reference'], 'native_PAN_and_native_LRMS')
        self.assertIn('not SIPSA-equivalent', result['variant'])

    def test_existing_jqm_tiny_numerical_integration(self):
        rng = np.random.default_rng(12)
        ms = rng.uniform(50, 1400, (8, 8, 8)); pan = rng.uniform(50, 1400, (32, 32))
        fused = np.repeat(np.repeat(ms, 4, axis=0), 4, axis=1)
        out = ev.jqm(fused, ms, pan, 'WV3', ratio=4, R=2047., lpf='mtf', window=None, v1=.5)
        self.assertAlmostEqual(out['JQM'], .5 * out['QLR'] + .5 * out['QHR'])
        self.assertEqual(out['w_source'], 'nnls-normalized')
        self.assertAlmostEqual(sum(out['w']), 1.)
        self.assertTrue(0 <= out['JQM'] <= 1)

    def test_precision_restores_global_flags_on_exception(self):
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with self.assertRaises(RuntimeError):
            with ev._precision(dict(tf32_matmul=not before[0], tf32_cudnn=not before[1]), 'cpu'):
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, not before[0])
                raise RuntimeError('synthetic')
        self.assertEqual((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), before)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.run = 'FH12_S1_T_SYNTHETIC'; self.wd = self.root / 'work_dir' / self.run
        files = {key: 'source-sha' for key in ev.INFERENCE_FILES}
        self.source = dict(files=files, content_sha256=object_sha(files), torch=torch.__version__,
                           numpy=np.__version__, scipy=ev.scipy.__version__, skimage=ev.skimage.__version__,
                           tf32_matmul=False, tf32_cudnn=True)
        self.identity_patch = patch.object(ev, 'evaluator_identity', return_value=copy.deepcopy(self.source))
        self.identity_mock = self.identity_patch.start(); self.addCleanup(self.identity_patch.stop)
        self.data = dict(sensor='WV3', max_pixel=2047., splits={})
        for split in ('rr', 'fr'):
            raw = self.root / f'{split}.h5'; lp = self.root / f'{split}_lp.h5'
            raw.write_bytes(b'fake original dataset'); lp.write_bytes(b'fake immutable lp')
            self.data['splits'][split] = dict(count=20, dataroot=str(raw), lpan_path=str(lp),
                                              sha256=sha256(raw), lpan_sha256=sha256(lp))
        atomic_json(self.root / 'work_dir/_fh12/s1/dataset_manifest.json', self.data)
        self.cfg = dict(seed=1, fh12=dict(run_id=self.run, server_id='s1', campaign_id='TEST_CAMPAIGN'))
        (self.wd / 'meta').mkdir(parents=True)
        (self.wd / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(self.cfg))
        atomic_json(self.wd / 'meta/training_status.json', dict(actual_updates=50000, training_complete=True))
        atomic_json(self.wd / 'official/postrun_status.json', dict(official_complete=True))
        records = []
        for i, step in enumerate([1010 * i for i in range(1, 50)] + [50000]):
            folder = self.wd / 'candidates' / str(step); folder.mkdir(parents=True)
            (folder / 'model.safetensors').write_bytes(f'fake model step {step}'.encode())
            ident = dict(update=step, model_sha256=sha256(folder / 'model.safetensors'),
                         config_sha256=object_sha(self.cfg), data_sha256=object_sha(self.data), source_identity=self.source)
            atomic_json(folder / 'identity.json', ident)
            records.append(dict(update=step, checkpoint_identity=ident, val_ergas=2. + i,
                                rr=dict(ergas=3. - i / 1000, scc=.98, psnr=38., sam=2., q8=.9, ssim=.95,
                                        n_scenes=20, crop='20:-21', official_complete=True),
                                fr=dict(hqnr=.96 if i == 0 else .95, d_lambda=.01, d_s=.02,
                                        n_scenes=20, reference='native_PAN', support='full512')))
        self.grid = dict(complete=True, run_id=self.run, campaign_id='TEST_CAMPAIGN',
                         config_sha256=object_sha(self.cfg), data_sha256=object_sha(self.data),
                         source_identity=self.source, records=records)
        self.publish_official()

    def tearDown(self):
        self.tmp.cleanup()

    def publish_official(self):
        atomic_json(self.wd / 'official/raw_grid.json', self.grid)
        selected = select_records(self.grid['records'])
        context = {k: self.grid[k] for k in ('run_id', 'campaign_id', 'config_sha256', 'data_sha256', 'source_identity')}
        for name, (label, key) in ev.SELECTIONS.items():
            extra = dict(target_status=selected['target_status']) if key == 'target' else {}
            atomic_json(self.wd / 'official' / f'{name}.json', report_selection(label, selected[key], context, **extra))

    def compute(self):
        with patch.object(ev, 'build_dataset', return_value='fake'), patch.object(ev, '_evaluate_step', return_value=metrics()) as call:
            out = ev.process(self.run, root=self.root)
        return out, call.call_count

    def test_deduplicates_steps_keeps_original_files_and_validates_cache(self):
        before = {str(p): sha256(p) for p in self.wd.rglob('*') if p.is_file()}
        out, calls = self.compute()
        self.assertEqual(calls, 2); self.assertEqual(out['n_unique_checkpoints'], 2)
        self.assertEqual(out['selections']['raw_max']['step'], 1010)
        self.assertEqual(out['selections']['e_min_diag']['step'], 50000)
        self.assertEqual(before, {p: sha256(p) for p in before})
        self.assertEqual(self.compute()[1], 0)
        self.assertEqual(ev.validate_report(self.run, self.root), out)

    def test_no_eligible_is_empty_and_never_fabricated(self):
        for record in self.grid['records']: record['fr']['hqnr'] = .95
        self.publish_official(); out, _ = self.compute()
        row = out['selections']['target_selection']
        self.assertEqual(row['status'], 'no_eligible'); self.assertIsNone(row['metrics'])
        self.assertIsNone(row['step']); self.assertIsNone(row['checkpoint_sha256'])

    def test_missing_and_mutated_checkpoints_fail_before_inference(self):
        path = self.wd / 'candidates/1010/model.safetensors'
        path.write_bytes(b'wrong model')
        with self.assertRaisesRegex(ValueError, 'Checkpoint'): self.compute()
        path.unlink()
        with self.assertRaises(FileNotFoundError): self.compute()

    def test_missing_data_and_lp_mutation_are_not_fallbacks(self):
        Path(self.data['splits']['rr']['lpan_path']).write_bytes(b'bad lp')
        with self.assertRaisesRegex(ValueError, 'cache changed'): self.compute()
        Path(self.data['splits']['rr']['dataroot']).unlink()
        with self.assertRaises(FileNotFoundError): self.compute()

    def test_selection_tampering_rejected(self):
        path = self.wd / 'official/raw_max.json'; value = read_json(path); value['rr']['ergas'] = 0
        atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, 'selected grid'): self.compute()

    def test_grid_incomplete_and_training_incomplete_are_rejected(self):
        self.grid['complete'] = False; self.publish_official()
        with self.assertRaisesRegex(ValueError, 'grid'): self.compute()
        self.grid['complete'] = True; self.publish_official()
        atomic_json(self.wd / 'meta/training_status.json', dict(actual_updates=49000, training_complete=False))
        with self.assertRaisesRegex(ValueError, 'completed official'): self.compute()

    def test_origin_runtime_or_numerical_code_change_is_rejected(self):
        self.identity_mock.return_value['torch'] = 'different'
        with self.assertRaisesRegex(ValueError, 'runtime'): self.compute()
        self.identity_mock.return_value['torch'] = torch.__version__
        self.identity_mock.return_value['files']['pa/warp.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'implementation changed'): self.compute()

    def test_cache_tamper_is_not_overwritten_or_uploaded(self):
        self.compute(); path = self.wd / 'supplemental_metrics/step_1010.json'
        value = read_json(path); value['rr']['rmse'] = 999; atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, 'checksum'): ev.validate_report(self.run, self.root)
        with self.assertRaisesRegex(ValueError, 'checksum'): self.compute()

    def test_report_tamper_and_source_change_are_hard_failures(self):
        self.compute(); path = self.wd / 'supplemental_metrics/report.json'
        value = read_json(path); value['selections']['raw_max']['step'] = 50000
        atomic_json(path, ev._seal({k: v for k, v in value.items() if k != 'payload_sha256'}))
        with self.assertRaisesRegex(ValueError, 'cache mismatch'): ev.validate_report(self.run, self.root)

    def test_partial_failure_reuses_completed_step_on_retry(self):
        with patch.object(ev, 'build_dataset', return_value='fake'), patch.object(ev, '_evaluate_step', side_effect=[metrics(), RuntimeError('failed')]):
            with self.assertRaises(RuntimeError): ev.process(self.run, root=self.root)
        self.assertTrue((self.wd / 'supplemental_metrics/step_1010.json').is_file())
        self.assertFalse((self.wd / 'supplemental_metrics/report.json').exists())
        self.assertEqual(self.compute()[1], 1)

    def test_new_campaign_uses_same_model_loader_contract(self):
        self.cfg['fh20r1'] = self.cfg.pop('fh12')
        self.cfg['fh20r1']['dataset_manifest'] = 'work_dir/_fh12/s1/dataset_manifest.json'
        (self.wd / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(self.cfg))
        self.grid['config_sha256'] = object_sha(self.cfg)
        for row in self.grid['records']:
            row['checkpoint_identity']['config_sha256'] = object_sha(self.cfg)
            atomic_json(self.wd / 'candidates' / str(row['update']) / 'identity.json', row['checkpoint_identity'])
        self.publish_official()
        self.assertEqual(self.compute()[1], 2)

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError): ev.process('../somewhere', root=self.root)


if __name__ == '__main__':
    unittest.main()
