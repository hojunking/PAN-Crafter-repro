import copy
import io
from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
import torch
from safetensors.torch import save_file

from g20.test_evaluation import record
from g20.model import state_hash
from fh12.training import BatchStream
from l100.common import atomic_json, object_sha, read_json, sha256
from l100.plan import CAMPAIGN_ID, build_config, case_for, grid_steps
from l100.postrun import (process, process_endpoint, process_partial, report_selection,
                          select_records, selections, validate_grid, verify_fullstate,
                          update_best_metadata)


def fixture(root, case_id='L100I1-S07', *, complete=True):
    case = case_for(case_id)
    cfg = build_config(case)
    cfg['l100'].update(dataset_manifest='data.json', reference_manifest='ref.json')
    data = dict(sensor='GF2', num_bands=4, max_pixel=1023, band_order=['B', 'G', 'R', 'NIR'],
        mtf_sensor='GF2', recipe={'phase_id': 2}, splits={s: dict(count=48 if s == 'train' else 20, sha256='d' * 64,
        lpan_sha256='e' * 64) for s in ('train', 'val', 'rr', 'fr')})
    release = dict(content_sha256='7' * 64, numeric_method_revision='QG40_SYNC_FREQ_C4_v1')
    ref = dict(owner_server=case.server_id, producer_server=case.server_id,
        teacher_seed=case.teacher_seed, teacher_update=case.teacher_updates,
        tau_R=.01, q_ref=.4, teacher_checkpoint_sha256='c' * 64,
        calibration_id='CAL-fixture', q_cache_sha256='b' * 64)
    common = dict(config_sha256=object_sha(cfg), data_sha256=object_sha(data),
        source_identity=release, reference_sha256=object_sha(ref) if case.role == 'S' else None)
    atomic_json(root / 'data.json', data)
    atomic_json(root / 'ref.json', ref)
    wd = root / 'work_dir' / case.run_id
    (wd / 'meta').mkdir(parents=True)
    (wd / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(cfg))
    atomic_json(wd / 'meta/training_start_manifest.json', dict(campaign_id=CAMPAIGN_ID,
        run_id=case.run_id, horizon_updates=case.updates, started_at_utc='2026-09-21T00:00:00+00:00', **common))
    atomic_json(wd / 'meta/training_status.json', dict(actual_updates=case.updates,
        training_complete=True, training_seconds=600, updated_at_utc='2026-09-21T00:20:00+00:00'))
    rows = []
    for step in grid_steps(case.updates):
        row = record(step, h=.966 if step == grid_steps(case.updates)[0] else .965,
                     e=.53 if step == grid_steps(case.updates)[1] else .54)
        row['val_ergas'] = .5 if step == grid_steps(case.updates)[2] else .6
        folder = wd / 'candidates' / str(step)
        folder.mkdir(parents=True)
        weights = folder / 'model.safetensors'
        weights.write_bytes(('synthetic CPU checkpoint ' + str(step)).encode())
        row['checkpoint_identity'] = dict(update=step, model_sha256=sha256(weights),
            role=case.role, sensor='GF2', num_bands=4, input_layout=case.input_layout, **common)
        if step in (50000, case.updates):
            from l100.training import cosine_factor, aligner_factor
            tensors = {'one': torch.tensor([float(step)])}
            save_file(tensors, str(weights))
            identity = row['checkpoint_identity']
            identity.update(full_state=True, state_hash=state_hash(tensors), model_sha256=sha256(weights))
            stream = BatchStream(48, 48, case.seed)
            stream.epoch, stream.cursor = step - 1, 1
            factor = cosine_factor(step, cfg['num_warmup'], case.updates)
            lr = [cfg['learning_rate'] * factor,
                  cfg['l100']['aligner_lr'] * factor * aligner_factor(step, case.profile)]
            state = dict(full_state=True, update=step, model_state=tensors,
                model_sha256=identity['model_sha256'], precision='fp32',
                scheduler={'last_epoch': step, '_last_lr': lr},
                optimizer={'param_groups': [dict(name=name, lr=value) for name, value in zip(('U', 'A'), lr)]},
                sampler=stream.state_dict(), exposure_counts=torch.full((48, 4), step // 4, dtype=torch.int64), **common)
            torch.save(state, folder / 'training_state.pt')
            identity['training_state_sha256'] = sha256(folder / 'training_state.pt')
        atomic_json(folder / 'identity.json', row['checkpoint_identity'])
        if complete or step == case.updates:
            rows.append(row)
    if case.role == 'T':
        ref['teacher_checkpoint_sha256'] = rows[-1]['checkpoint_identity']['model_sha256']
        atomic_json(root / 'ref.json', ref)
    grid = dict(campaign_id=CAMPAIGN_ID, run_id=case.run_id, sensor='GF2', horizon_updates=case.updates,
        expected_steps=list(grid_steps(case.updates)), records=rows, n_evaluated=len(rows),
        complete=complete, **common)
    atomic_json(wd / 'official/raw_grid.json', grid)
    atomic_json(wd / 'official/profile.json', dict(config_sha256=object_sha(cfg), source_identity=release,
        num_bands=4, sensor='GF2', scope='actual C4 fixture', flops_convention='fixture',
        params_m=1., flops_g=2., infer_ms=3.))
    return case, cfg, data, ref, grid


@contextmanager
def runtime(case, grid, ref):
    with ExitStack() as stack:
        stack.enter_context(patch('l100.postrun.apply_runtime_policy'))
        stack.enter_context(patch('l100.upload.apply_runtime_policy'))
        stack.enter_context(patch('l100.training.validate_config', return_value=case))
        stack.enter_context(patch('l100.training.runtime_context', return_value=({'deadline_utc': '2099-01-01T00:00:00+00:00'}, None)))
        stack.enter_context(patch('l100.postrun.source_identity', return_value=grid['source_identity']))
        stack.enter_context(patch('l100.upload.source_identity', return_value=grid['source_identity']))
        stack.enter_context(patch('l100.references.validate_reference', return_value=(ref, None, None, None)))
        yield


class PostrunTests(unittest.TestCase):
    def test_unchanged_numerical_function_objects(self):
        import g20.evaluation as previous
        import l100.evaluation as current
        for name in current.__all__:
            self.assertIs(getattr(current, name), getattr(previous, name))

    def test_100k_grid_midpoint_and_fresh50_are_distinct(self):
        rows = [record(step) for step in grid_steps(100000)]
        selected = select_records(rows, 100000)
        self.assertEqual(len(rows), 50)
        self.assertEqual(selected['exact_final']['update'], 100000)
        self.assertEqual(selected['mid50_of100']['update'], 50000)
        self.assertNotIn(50500, grid_steps(100000))
        for label, fresh in (('EXACT_FINAL', False), ('MID50_OF100', False)):
            doc = report_selection(label, selected['exact_final' if label == 'EXACT_FINAL' else 'mid50_of100'],
                                   {'horizon_updates': 100000})
            self.assertEqual(doc['is_fresh50k'], fresh)
        selected50 = select_records([record(s) for s in grid_steps(50000)], 50000)
        self.assertNotIn('mid50_of100', selected50)
        self.assertTrue(report_selection('EXACT_FINAL', selected50['exact_final'], {'horizon_updates': 50000})['is_fresh50k'])
        with self.assertRaises(ValueError):
            report_selection('MID50_OF100', rows[0], {'horizon_updates': 50000})

    def test_selector_threshold_ties_and_diagnostic_exclusion(self):
        rows = [record(step, h=.964, e=.521) for step in grid_steps(100000)]
        self.assertIsNone(select_records(rows, 100000)['target'])
        rows[1]['fr'] = record(rows[1]['update'], h=.964000001)['fr']
        self.assertEqual(select_records(rows, 100000)['target']['update'], rows[1]['update'])
        for bad in (rows[:-1], rows + [record(1000)], rows[:-1] + [rows[0]],
                    [record(s) for s in grid_steps(50000)]):
            with self.assertRaises(ValueError):
                select_records(bad, 100000)

    def test_all_selectors_keep_correct_primary_and_test_awareness(self):
        selected = select_records([record(s) for s in grid_steps(100000)], 100000)
        for key, label in selections(100000):
            report = report_selection(label, selected[key], {'horizon_updates': 100000})
            self.assertEqual(report['primary'], key in ('exact_final', 'rr_val_selected'))
            self.assertEqual(report['test_aware'], key in ('raw_max', 'target', 'e_min_diag'))

    def test_endpoint_does_not_wait_for_other49_or_mark_official(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root, 'L100I1-T03', complete=False)
            with runtime(case, grid, ref), patch('l100.postrun.evaluate_model') as evaluate:
                status = process_endpoint(case.run_id, device='cpu', root=root)
            self.assertTrue(status['endpoint_complete'])
            self.assertFalse(status['official_complete'])
            self.assertEqual(status['n_evaluated'], 1)
            self.assertEqual(status['n_pending'], 49)
            evaluate.assert_not_called()
            doc = read_json(root / 'work_dir' / case.run_id / 'official/exact_final_endpoint.json')
            self.assertEqual(doc['step'], 100000)
            self.assertFalse(doc['official_complete'])
            self.assertFalse((root / 'work_dir' / case.run_id / 'official/exact_final.json').exists())

    def test_full_processing_uses_registered_horizon(self):
        from l100.upload import row_values
        for case_id, horizon in (('L100I1-S07', 100000), ('L100I1-S01', 50000)):
            with self.subTest(horizon=horizon), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                case, cfg, data, ref, grid = fixture(root, case_id)
                with runtime(case, grid, ref):
                    self.assertEqual(process(case.run_id, 'cpu', root=root), 0)
                    values = row_values(case.run_id, root)
                self.assertEqual(values['EXACT_FINAL step'], horizon)
                self.assertEqual(values['HQNR↑'], values['EXACT_FINAL HQNR(raw)↑'])
                self.assertNotEqual(values['HQNR↑'], values['RAW_MAX HQNR(raw)↑'])
                self.assertFalse(any('Exact50K' in key for key in values))
                self.assertEqual('MID50_OF100 step' in values, horizon == 100000)
                self.assertEqual(values['Date'], '2026-09-21')
                self.assertEqual(values['Train(h)'], 600 / 3600)
                self.assertEqual(values['Wall(h)'], 1200 / 3600)

    def test_missing_endpoint_metrics_evaluates_only_exact_even_with_low_hqnr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root, 'L100I1-T03', complete=False)
            identity = grid['records'][0]['checkpoint_identity']
            metrics = record(case.updates, h=.8, e=2.)
            metrics = {key: metrics[key] for key in ('rr', 'fr', 'val_ergas')}
            grid.update(records=[], n_evaluated=0)
            wd = root / 'work_dir' / case.run_id
            atomic_json(wd / 'official/raw_grid.json', grid)
            output = io.StringIO()
            with runtime(case, grid, ref), patch('g20.data.build_dataset'), \
                    patch('l100.postrun.FRMetrics'), \
                    patch('l100.postrun.load_checkpoint_model', return_value=(object(), identity)) as load, \
                    patch('l100.postrun.evaluate_model', return_value=metrics) as evaluate, \
                    patch('sys.stdout', output):
                status = process_endpoint(case.run_id, 'cpu', root=root)
            self.assertTrue(status['endpoint_complete'])
            self.assertFalse(status['official_complete'])
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(load.call_args.args[1].name, '100000')
            self.assertEqual(read_json(wd / 'official/evaluation_debt.json')['n_pending'], 49)
            self.assertEqual(read_json(wd / 'best_hqnr_meta.json')['hqnr'], .8)
            for token in ('HQNR=', 'SCC=', 'ERGAS='):
                self.assertIn(token, output.getvalue())

    def test_legacy_best_metadata_uses_hqnr_then_scc_then_ergas(self):
        rows = [record(2020, h=.966, e=.6), record(4040, h=.966, e=.5), record(6060, h=.965, e=.1)]
        rows[0]['rr']['scc'] = .91
        rows[1]['rr']['scc'] = .9
        with tempfile.TemporaryDirectory() as tmp:
            update_best_metadata(tmp, dict(records=rows))
            self.assertEqual(read_json(Path(tmp) / 'best_hqnr_meta.json')['step'], 2020)
            rows[0]['rr']['scc'] = .9
            update_best_metadata(tmp, dict(records=rows))
            self.assertEqual(read_json(Path(tmp) / 'best_hqnr_meta.json')['step'], 4040)

    def test_grid_rejects_diagnostics_reference_and_source_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root)
            with runtime(case, grid, ref):
                for mutate in (lambda g: g.__setitem__('horizon_updates', 50000),
                    lambda g: g['records'].append(record(1000)),
                    lambda g: g['records'][0]['checkpoint_identity'].__setitem__('reference_sha256', 'x'),
                    lambda g: g.__setitem__('source_identity', {}),
                    lambda g: g.__setitem__('n_evaluated', 49)):
                    bad = copy.deepcopy(grid)
                    mutate(bad)
                    with self.assertRaises(ValueError):
                        validate_grid(case.run_id, cfg, bad, root)

    def test_pending_grid_and_changed_teacher_owner_never_upload(self):
        from l100.upload import row_values
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root)
            with runtime(case, grid, ref):
                process(case.run_id, 'cpu', root=root)
                ref['owner_server'] = 's3'
                with self.assertRaisesRegex(ValueError, 'local Teacher'):
                    row_values(case.run_id, root)
                ref['owner_server'] = 's4'
                path = root / 'work_dir' / case.run_id / 'official/postrun_status.json'
                status = read_json(path)
                status['official_complete'] = False
                atomic_json(path, status)
                with self.assertRaisesRegex(ValueError, 'all 50'):
                    row_values(case.run_id, root)

    def test_partial_100k_is_not_labeled_a_complete_fresh50(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root, complete=False)
            grid.update(records=[], n_evaluated=0)
            wd = root / 'work_dir' / case.run_id
            atomic_json(wd / 'official/raw_grid.json', grid)
            atomic_json(wd / 'meta/training_status.json', dict(actual_updates=50000, training_complete=False))
            with runtime(case, grid, ref), patch('l100.postrun._repair_saved_grid', side_effect=lambda r,c,g,*a: g):
                status = process_partial(case.run_id, 'cpu', root=root)
            self.assertEqual(status['actual_updates'], 50000)
            self.assertEqual(status['horizon_updates'], 100000)
            self.assertFalse(status['official_complete'])
            self.assertFalse(status['primary_selection_available'])

    def test_midpoint_scheduler_and_fullstate_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root)
            verify_fullstate(case.run_id, cfg, grid, root)
            folder = root / 'work_dir' / case.run_id / 'candidates/50000'
            state = torch.load(folder / 'training_state.pt', weights_only=False)
            original = copy.deepcopy(state)
            for mutate in (lambda s: s['scheduler'].__setitem__('last_epoch', 49999),
                           lambda s: s['scheduler'].__setitem__('_last_lr', [0., 0.]),
                           lambda s: s.__setitem__('reference_sha256', 'wrong'),
                           lambda s: s['exposure_counts'].__setitem__((0, 0), 0)):
                state = copy.deepcopy(original)
                mutate(state)
                torch.save(state, folder / 'training_state.pt')
                identity = read_json(folder / 'identity.json')
                identity['training_state_sha256'] = sha256(folder / 'training_state.pt')
                atomic_json(folder / 'identity.json', identity)
                with self.assertRaises(ValueError):
                    verify_fullstate(case.run_id, cfg, grid, root)

    def test_recovery_date_uses_original_completion_not_recovery_clock(self):
        from l100.upload import row_values
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root)
            path = root / 'work_dir' / case.run_id / 'meta/training_status.json'
            state = read_json(path)
            state.update(recovered_from_exact_final=True, updated_at_utc='2026-09-23T00:00:00+00:00',
                         training_completed_at_utc='2026-09-21T00:20:00+00:00')
            atomic_json(path, state)
            with runtime(case, grid, ref):
                process(case.run_id, 'cpu', root=root)
                values = row_values(case.run_id, root)
                self.assertEqual(values['Date'], '2026-09-21')
                self.assertEqual(values['Wall(h)'], 1200 / 3600)
                state['recovered_from_exact_final'] = False
                atomic_json(path, state)
                values = row_values(case.run_id, root)
                self.assertEqual(values['Date'], '2026-09-21')
                self.assertEqual(values['Wall(h)'], 1200 / 3600)
                state['recovered_from_exact_final'] = True
                state['training_completed_at_utc'] = None
                atomic_json(path, state)
                values = row_values(case.run_id, root)
                self.assertEqual(values['Date'], '2026-09-21')
                self.assertEqual(values['Wall(h)'], '')


if __name__ == '__main__':
    unittest.main()
