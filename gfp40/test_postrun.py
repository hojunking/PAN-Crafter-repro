import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from g20.test_evaluation import record
from gfp40.plan import CAMPAIGN_ID, grid_steps, build_config, case_for
from gfp40.postrun import summarize_grid, process
from gfp40.common import atomic_json, object_sha, sha256, read_json
from gfp40.test_assets import TinyPair
from g20.model import state_hash


class PostrunTests(unittest.TestCase):
    def test_endpoint_progress_binds_actual_family_not_default_g025(self):
        from gfp40.postrun import verify_endpoint_progress
        from gfp40.stream import PairedBatchStream
        cfg = build_config('B12')
        cfg['gfp40'].update(dataset_manifest='data.json', reference_manifest='reference.json',
            parent_manifest='parent.json', augmentation_manifest='views.json',
            mixed_calibration_manifest='scales.json', runtime_policy_sha256='a' * 64)
        stream = PairedBatchStream(48, 48, case_for('B12').stream_seed, family='G0125')
        stream.epoch, stream.cursor = 19999, 1
        state = dict(update=20000, sampler=stream.state_dict(),
            exposure_counts=torch.full((48, 4), 5000, dtype=torch.int64),
            gamma_counts={key: torch.tensor([240000, 480000, 240000], dtype=torch.int64)
                          for key in ('drawn', 'effective')},
            optimizer={'param_groups': [dict(name='U', lr=0.), dict(name='A', lr=0.)]},
            scheduler={'_last_lr': [0., 0.]})
        verify_endpoint_progress(cfg, state, {'splits': {'train': {'count': 48}}})
        bad = copy.deepcopy(state)
        bad['gamma_counts']['effective'][0] -= 1
        bad['gamma_counts']['effective'][1] += 1
        with self.assertRaisesRegex(ValueError, 'drawn and effective'):
            verify_endpoint_progress(cfg, bad, {'splits': {'train': {'count': 48}}})
        bad['gamma_counts']['drawn'] = torch.ones(3, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, 'actual consumed'):
            verify_endpoint_progress(cfg, bad, {'splits': {'train': {'count': 48}}})
        state['sampler']['family'] = 'G025'
        with self.assertRaisesRegex(ValueError, 'family identity'):
            verify_endpoint_progress(cfg, state, {'splits': {'train': {'count': 48}}})

    def test_60k_curves_are_not_extra_selection_candidates(self):
        rows = [record(step) for step in grid_steps(60000)]
        selected = summarize_grid(rows, 60000)
        self.assertEqual(selected['n_candidates'], 20)
        self.assertEqual(selected['expected_steps'], list(range(3000, 60001, 3000)))
        self.assertEqual(selected['selections']['EXACT_FINAL']['update'], 60000)
        self.assertTrue(selected['selections']['EXACT_FINAL']['primary'])
        self.assertFalse(selected['selections']['RR_VAL_SELECTED']['primary'])
        for step in (0, 20000, 40000):
            with self.assertRaises(ValueError): summarize_grid(rows + [record(step)], 60000)

    def test_completed_process_preserves_endpoint_and_lineage_cost(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            case = case_for('A01')
            cfg = build_config(case)
            cfg['gfp40'].update(dataset_manifest=str(root / 'data.json'),
                reference_manifest=str(root / 'reference.json'),
                parent_manifest=str(root / 'parent.json'), runtime_policy_sha256='a' * 64)
            data = dict(sensor='GF2', num_bands=4, max_pixel=1023, splits={'train': {'count': 48}})
            release = {'fixture': 'CPU source'}
            atomic_json(root / 'data.json', data)
            atomic_json(root / 'parent.json', dict(parent_model_sha256='e' * 64,
                parent_compute={'training_seconds': 7200.}))
            wd = root / cfg['work_dir']
            config_path = wd / 'meta/config.resolved.yaml'
            atomic_json(config_path, cfg)
            common = dict(config_sha256=object_sha(cfg), data_sha256=object_sha(data),
                          source_identity=release, reference_sha256='f' * 64)
            atomic_json(wd / 'meta/training_start_manifest.json', common)
            atomic_json(wd / 'meta/training_status.json', dict(actual_updates=20000,
                training_complete=True, training_seconds=3600., evaluation_seconds=120., io_seconds=30.))
            model_state = TinyPair().state_dict()
            rows = []
            for step in grid_steps(case.updates):
                row = record(step)
                row['fr'].update(jqm=.81, jqm_variant='test explicit surrogate variant')
                folder = wd / 'candidates' / str(step)
                folder.mkdir(parents=True)
                save_file(model_state, str(folder / 'model.safetensors'))
                identity = dict(update=step, role='S', sensor='GF2', num_bands=4, input_layout='PLH',
                    full_state=step == case.updates, state_hash=state_hash(model_state),
                    model_sha256=sha256(folder / 'model.safetensors'), **common)
                if step == case.updates:
                    full = dict(full_state=True, update=step, precision='fp32', model_state=model_state,
                        model_sha256=identity['model_sha256'], scheduler={'last_epoch': step}, **common)
                    from gfp40.stream import PairedBatchStream
                    stream = PairedBatchStream(48, 48, case.stream_seed)
                    stream.epoch, stream.cursor = 19999, 1
                    full.update(sampler=stream.state_dict(),
                        exposure_counts=torch.full((48, 4), 5000, dtype=torch.int64),
                        gamma_counts={'drawn': torch.tensor([240000, 480000, 240000], dtype=torch.int64),
                                      'effective': torch.tensor([0, 960000, 0], dtype=torch.int64)},
                        optimizer={'param_groups': [dict(name='U', lr=0.), dict(name='A', lr=0.)]})
                    full['scheduler']['_last_lr'] = [0., 0.]
                    torch.save(full, folder / 'training_state.pt')
                    identity['training_state_sha256'] = sha256(folder / 'training_state.pt')
                atomic_json(folder / 'identity.json', identity)
                row['checkpoint_identity'] = identity
                rows.append(row)
            atomic_json(wd / 'official/raw_grid.json', dict(campaign_id=CAMPAIGN_ID,
                run_id=case.run_id, horizon_updates=20000, expected_steps=list(grid_steps(20000)),
                records=rows, **common))
            with patch('gfp40.postrun.source_identity', return_value=release):
                out = process(config_path, root, 'cpu')
            self.assertTrue(out['complete'])
            self.assertEqual(out['local_updates'], 20000)
            self.assertEqual(out['parent_step'], 100000)
            self.assertEqual(out['lifetime_updates'], 120000)
            self.assertEqual(out['training_hours'], 1.)
            self.assertEqual(out['parent_training_hours'], 2.)
            self.assertEqual(out['lineage_training_hours'], 3.)
            self.assertEqual(out['selections']['EXACT_FINAL']['fr']['jqm'], .81)
            self.assertIn('rmse', out['selections']['EXACT_FINAL']['rr'])
            self.assertIn('cc', out['selections']['EXACT_FINAL']['rr'])
            self.assertNotIn('EXACT100K', out['selections'])
            # Re-evaluating saved debt is counted separately, not lost from
            # cost merely because optimization was already completed.
            saved = read_json(wd / 'official/raw_grid.json')
            missing = saved['records'].pop(0)
            atomic_json(wd / 'official/raw_grid.json', saved)
            metrics = {k: missing[k] for k in ('fr', 'rr', 'val_ergas')}
            with patch('gfp40.postrun.source_identity', return_value=release), patch(
                    'g20.data.build_dataset', return_value=None), patch('gfp40.postrun.FRMetrics', return_value=None), patch(
                    'gfp40.postrun.load_candidate', return_value=(TinyPair(), missing['checkpoint_identity'])), patch(
                    'gfp40.postrun.evaluate_model', return_value=metrics):
                paid = process(config_path, root, 'cpu')
            self.assertTrue(paid['complete'])
            self.assertGreater(paid['costs']['postrun_evaluation_seconds'], 0.)
            self.assertGreater(paid['evaluation_hours'], 120. / 3600)
            self.assertEqual(paid['training_hours'], 1.)
            self.assertEqual(paid['completed_at_utc'], out['completed_at_utc'])
            self.assertEqual(paid['official_completed_at_utc'], out['official_completed_at_utc'])
            self.assertEqual(paid['official_evidence_sha256'], out['official_evidence_sha256'])

    def test_native_numerical_functions_are_existing_objects(self):
        import g20.evaluation as old
        import gfp40.evaluation as new
        for key in new.__all__:
            self.assertIs(getattr(old, key), getattr(new, key))

    def test_ft_parent_anchor_is_not_candidate(self):
        rows = [record(s) for s in grid_steps(20000)]
        summary = summarize_grid(rows, 20000)
        self.assertEqual(summary['n_candidates'], 20)
        self.assertEqual(summary['selections']['EXACT_FINAL']['update'], 20000)
        with self.assertRaises(ValueError): summarize_grid([record(0)] + rows, 20000)

    def test_fresh_grid_is_exact_50_including50500_not50000(self):
        rows = [record(s) for s in grid_steps(100000)]
        self.assertIn(50500, grid_steps(100000))
        self.assertNotIn(50000, grid_steps(100000))
        summary = summarize_grid(rows, 100000)
        self.assertEqual(summary['n_candidates'], 50)
        self.assertEqual(summary['selections']['EXACT_FINAL']['update'], 100000)
        wrong = [record(50000) if r['update'] == 50500 else r for r in rows]
        with self.assertRaises(ValueError): summarize_grid(wrong, 100000)

    def test_every_metric_in_selection_comes_from_same_checkpoint(self):
        rows = [record(s, h=.95, e=.55) for s in grid_steps(20000)]
        rows[0] = record(1000, h=.97, e=.60)
        rows[1] = record(2000, h=.93, e=.51)
        rows[2]['val_ergas'] = .01
        out = summarize_grid(rows, 20000)['selections']
        self.assertEqual(out['RAW_AUX']['update'], 1000)
        self.assertEqual(out['RAW_AUX']['rr']['ergas'], .60)
        self.assertEqual(out['RR_VAL_SELECTED']['update'], 3000)
        self.assertEqual(out['RR_VAL_SELECTED']['fr']['hqnr'], .95)
        self.assertEqual(out['EXACT_FINAL']['rr']['ergas'], .55)

    def test_val_tie_earlier_and_raw_aux_hqnr_scc_ergas(self):
        rows = [record(s, h=.96, e=.55) for s in grid_steps(20000)]
        rows[1]['rr']['scc'] += .01
        selections = summarize_grid(rows, 20000)['selections']
        self.assertEqual(selections['RR_VAL_SELECTED']['update'], 1000)
        self.assertEqual(selections['RAW_AUX']['update'], 2000)
        self.assertFalse(selections['RAW_AUX']['primary'])
        self.assertFalse(selections['RR_VAL_SELECTED']['primary'])

    def test_duplicate_incomplete_wrong_sensor_masking_rejected(self):
        rows = [record(s) for s in grid_steps(20000)]
        for invalid in (rows[:-1], rows + [rows[0]], rows[1:] + [record(500)]):
            with self.assertRaises(ValueError): summarize_grid(invalid, 20000)
        for field, value in (('masking', True), ('sensor', 'WV3'), ('n_scenes', 19)):
            bad = copy.deepcopy(rows)
            bad[0]['fr'][field] = value
            with self.assertRaises(ValueError): summarize_grid(bad, 20000)

    def test_signed_ds_operand_corruption_rejected(self):
        rows = [record(s) for s in grid_steps(20000)]
        rows[0]['fr']['signed_ds']['Q_high'][0][0] += .01
        with self.assertRaises(ValueError): summarize_grid(rows, 20000)


if __name__ == '__main__':
    unittest.main()
