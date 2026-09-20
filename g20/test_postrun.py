import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from g20.common import atomic_json, camp, object_sha, read_json, sha256
from g20.model import state_hash
from g20.plan import CAMPAIGN_ID, GRID_STEPS, build_config, case_for
from g20.postrun import (SELECTIONS, process, process_partial, report_selection,
                         screen_result, select_records, validate_record, profile_model)
from g20.test_evaluation import record
from qg40.exposure import exposure_report
from fh12.training import BatchStream


def fixture(root, case_id='G20-S11'):
    case = case_for(case_id) if isinstance(case_id, str) else case_id
    wd = root / 'work_dir' / case.run_id
    cfg = build_config(case)
    cfg['g20']['dataset_manifest'] = 'fixture_data.json'
    cfg['g20']['reference_manifest'] = 'fixture_reference.json'
    data = dict(sensor='GF2', num_bands=4, max_pixel=1023, band_order=['B', 'G', 'R', 'NIR'],
                mtf_sensor='GF2', recipe={'phase_id': 2}, splits={s: dict(
                    count=48 if s == 'train' else 20, sha256='d' * 64, lpan_sha256='e' * 64)
                    for s in ('train', 'val', 'rr', 'fr')})
    ref = dict(tau_R=.01, q_ref=.4, teacher_checkpoint_sha256='c' * 64,
               q_cache_sha256='b' * 64, calibration_id='a' * 64)
    release = dict(content_sha256='7' * 64, numeric_method_revision='QG40_SYNC_FREQ_C4_v1')
    common = dict(config_sha256=object_sha(cfg), data_sha256=object_sha(data),
                  source_identity=release, reference_sha256=object_sha(ref))
    atomic_json(root / 'fixture_data.json', data)
    atomic_json(root / 'fixture_reference.json', ref)
    (wd / 'meta').mkdir(parents=True)
    (wd / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(cfg))
    counts = torch.full((48, 4), 12500, dtype=torch.int64)
    stream = BatchStream(48, 48, case.seed)
    initial_sampler_hash = state_hash({'order': stream.order, 'rotations': stream.rotations})
    stream.epoch, stream.cursor = 49999, 1
    sampler = stream.state_dict()
    exposure = exposure_report(counts, 48, sampler['epoch'], sampler['cursor'])
    atomic_json(wd / 'meta/training_status.json', dict(actual_updates=50000,
        training_complete=True, sample_exposure=exposure, training_seconds=600))
    rows = []
    for step in GRID_STEPS:
        row = record(step, h=.966 if step == 1010 else .965, e=.53 if step == 2020 else .54)
        row['val_ergas'] = .5 if step == 3030 else .6
        folder = wd / 'candidates' / str(step)
        folder.mkdir(parents=True)
        weights = folder / 'model.safetensors'
        weights.write_bytes(('synthetic CPU fixture ' + str(step)).encode())
        identity = dict(update=step, model_sha256=sha256(weights), **common)
        if step == 50000:
            model_state = {'one': torch.tensor([1.])}
            identity['state_hash'] = state_hash(model_state)
            state = dict(full_state=True, update=step, model_state=model_state,
                model_sha256=identity['model_sha256'], exposure_counts=counts, sampler=sampler, **common)
            torch.save(state, folder / 'training_state.pt')
            identity['training_state_sha256'] = sha256(folder / 'training_state.pt')
        atomic_json(folder / 'identity.json', identity)
        row['checkpoint_identity'] = identity
        rows.append(row)
    grid = dict(campaign_id=CAMPAIGN_ID, run_id=case.run_id, records=rows, complete=True, **common)
    atomic_json(wd / 'official/raw_grid.json', grid)
    selected = select_records(rows)
    context = dict(campaign_id=CAMPAIGN_ID, run_id=case.run_id, sensor='GF2',
                   normal_same_step_A_U=True, **common)
    for key, label in SELECTIONS:
        extra = dict(n_evaluated=50)
        if key == 'target':
            extra.update(n_eligible=selected['n_eligible'], target_status=selected['target_status'],
                joint_pass=selected['joint_pass'], strong_joint_pass=selected['strong_joint_pass'],
                threshold_comparison='>', hqnr_threshold=.964, ergas_goal=.552,
                ergas_strong_goal=.522, selector_order=['ergas', '-scc', '-psnr', 'step'])
        atomic_json(wd / 'official' / (('target_selection' if key == 'target' else key) + '.json'),
                    report_selection(label, selected[key], context, **extra))
    atomic_json(wd / 'official/profile.json', dict(config_sha256=object_sha(cfg), source_identity=release,
        num_bands=4, sensor='GF2', scope='actual C4 fixture', flops_convention='fixture',
        params_m=1., flops_g=2., infer_ms=3.))
    atomic_json(wd / 'official/postrun_status.json', dict(official_complete=True, actual_updates=50000,
        config_sha256=object_sha(cfg), source_identity=release))
    atomic_json(wd / 'init_manifest.json', dict(seed=case.student_seed, role='S', bands=4, layout='PLH',
        pretrained_backbone_loads=0, student_aligner_independent_clone=True,
        reference_sha256=object_sha(ref), architecture_sha256='f' * 64, hashes={'U': '1' * 64, 'A': '2' * 64}))
    atomic_json(wd / 'meta/training_start_manifest.json', dict(campaign_id=CAMPAIGN_ID, run_id=case.run_id,
        sampler_hash=initial_sampler_hash, rng_roles=dict(data_order=case.seed + 300000, augmentation=case.seed + 400000,
            corruption=cfg['g20']['corruption_seed'], workers=case.seed + 500000), **common))
    atomic_json(camp(root, case.server_id) / 'preflight.json', dict(complete=True, real_native_forward=True,
        status='LOCAL_READY', sensor='GF2', server_id=case.server_id, source_identity=release,
        dataset_manifest_sha256=object_sha(data)))
    return case, cfg, data, ref, grid


class PostrunTests(unittest.TestCase):
    def test_cost_measures_actual_four_band_frontend_not_historical_c8(self):
        from g20.plan import sensor_spec
        class Dataset:
            spec = sensor_spec()
            def base(self, index):
                return (torch.zeros(4, 8, 8), torch.zeros(4, 8, 8), torch.zeros(4, 2, 2),
                        torch.zeros(1, 2, 2), torch.zeros(1, 8, 8), torch.tensor([0, 0]))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = torch.nn.Parameter(torch.ones(4))
            def forward(self, pan, ms, lp):
                self.assert_channels = ms.shape[1]
                return torch.zeros(1, 4, 8, 8)
        model = Model().train()
        with patch('thop.profile', return_value=(1000, 4)) as profile:
            result = profile_model(model, Dataset(), 'cpu')
        self.assertEqual(profile.call_args.kwargs['inputs'][1].shape[1], 4)
        self.assertEqual(result['num_bands'], 4)
        self.assertEqual(result['sensor'], 'GF2')
        self.assertEqual(result['params_m'], 4 / 1e6)
        self.assertEqual(result['flops_g'], (1000 + (32 * 7 * 64 + 64 + 4 * 64) / 2) / 1e9)
        self.assertTrue(model.training)

    def test_conditional_runtime_id_requires_exact_resolution_receipt(self):
        from g20.plan import resolve_confirmation
        from g20.upload import row_values
        ref = dict(tau_R=.01, q_ref=.4, teacher_checkpoint_sha256='c' * 64,
                   q_cache_sha256='b' * 64, calibration_id='a' * 64)
        decision = dict(campaign_id=CAMPAIGN_ID, server_id='s3', decision='SCREEN_SELECTION',
            selected_candidate='A1', selected_status='PROMISING_PAIRED',
            selected_at_utc='2026-09-20T00:00:00+00:00', evidence_files={'actual.json': '8' * 64})
        resolved = resolve_confirmation('s3', decision, {'R0': object_sha(ref)})[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root, resolved)
            with patch('g20.references.validate_reference', return_value=(ref, None, None, None)):
                with self.assertRaisesRegex(ValueError, 'resolution receipt'):
                    row_values(case.run_id, root)
                receipt = camp(root, 's3') / 'screen_selection.json'
                atomic_json(receipt, decision)
                self.assertEqual(row_values(case.run_id, root)['G20 case id'], case.case_id)
                atomic_json(receipt, dict(decision, selected_candidate='A9'))
                with self.assertRaisesRegex(ValueError, 'resolution receipt'):
                    row_values(case.run_id, root)

    def test_five_fixed_selectors_strict_threshold_and_primary_labels(self):
        rows = [record(step, h=.964, e=.521) for step in GRID_STEPS]
        self.assertIsNone(select_records(rows)['target'])
        rows[1] = record(2020, h=.96400001, e=.521)
        selected = select_records(rows)
        self.assertEqual(selected['target']['update'], 2020)
        self.assertTrue(selected['strong_joint_pass'])
        self.assertEqual(selected['exact50k']['update'], 50000)
        self.assertEqual(selected['rr_val_selected']['update'], 1010)
        with self.assertRaises(ValueError):
            select_records(rows[:-1])
        for key, label in SELECTIONS:
            report = report_selection(label, selected[key], {})
            self.assertEqual(report['primary'], key in ('exact50k', 'rr_val_selected'))
            self.assertEqual(report['test_aware'], key in ('raw_max', 'target', 'e_min_diag'))

    def test_missing_signed_ds_or_changed_native_support_is_not_official(self):
        for mutate in (lambda r: r['fr'].pop('signed_ds'),
                       lambda r: r['rr'].__setitem__('q8', .9),
                       lambda r: r['fr'].__setitem__('masking', True),
                       lambda r: r['checkpoint_identity'].__setitem__('update', 50000)):
            row = record(1010)
            mutate(row)
            with self.assertRaises(ValueError):
                validate_record(row, 'GF2')

    def test_official_row_uses_exact_not_raw_and_screen_has_verified_evidence(self):
        from g20.upload import row_values
        from g20.policy import RunResult
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root)
            with patch('g20.references.validate_reference', return_value=(ref, None, None, None)):
                values = row_values(case.run_id, root)
                self.assertEqual(values['HQNR↑'], values['Exact50K HQNR(raw)↑'])
                self.assertNotEqual(values['HQNR↑'], values['RAW_MAX HQNR(raw)↑'])
                result = screen_result(case.run_id, root)
                self.assertTrue(RunResult(**result).valid())
                self.assertEqual(result['development_target']['ERGAS'], .53)
                self.assertEqual(result['exact50k']['ERGAS'], .54)
                self.assertEqual(result['rr_val_selected']['HQNR'], .965)
                for path, digest in result['evidence_files'].items():
                    self.assertEqual(sha256(path), digest)
                delivery = root / 'work_dir' / case.run_id / 'official/postrun_status.json'
                atomic_json(delivery, dict(read_json(delivery), sheet_uploaded=True, status='UPLOAD_VERIFIED'))
                self.assertEqual(screen_result(case.run_id, root)['evidence_files'], result['evidence_files'])
                p0 = camp(root, case.server_id) / 'preflight.json'
                doc = read_json(p0)
                doc['real_native_forward'] = False
                atomic_json(p0, doc)
                with self.assertRaisesRegex(ValueError, 'P0 readiness'):
                    screen_result(case.run_id, root)

    def test_pending_grid_cannot_fabricate_primary_selection_or_resume_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root)
            wd = root / 'work_dir' / case.run_id
            grid.update(records=[], complete=False)
            atomic_json(wd / 'official/raw_grid.json', grid)
            # No missing candidate is runnable after a simulated closure timeout.
            with patch('g20.training.runtime_context', return_value=({'deadline_utc': '2999-01-01T00:00:00+00:00'}, '2000-01-01')), \
                 patch('g20.postrun.source_identity', return_value=grid['source_identity']), \
                 patch('g20.postrun._repair_saved_grid', return_value=grid), \
                 patch('g20.postrun.load_checkpoint_model') as load:
                self.assertEqual(process(case.run_id, root=root, device='cpu'), 0)
            status = read_json(wd / 'official/postrun_status.json')
            self.assertFalse(status['official_complete'])
            self.assertEqual(status['n_pending'], 50)
            load.assert_not_called()

    def test_saved_candidate_repair_is_evaluation_only_and_uses_closure_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root)
            wd = root / 'work_dir' / case.run_id
            final = grid['records'].pop()
            grid['complete'] = False
            atomic_json(wd / 'official/raw_grid.json', grid)
            metrics = {k: v for k, v in final.items() if k not in ('update', 'checkpoint_identity')}
            with patch('g20.training.runtime_context', return_value=({'deadline_utc': '2999-01-01T00:00:00+00:00'}, '2000-01-01')), \
                 patch('g20.postrun.source_identity', return_value=grid['source_identity']), \
                 patch('g20.data.build_dataset', return_value=object()), \
                 patch('g20.evaluation.FRMetrics', return_value=object()), \
                 patch('g20.evaluation.evaluate_model', return_value=metrics) as evaluate, \
                 patch('g20.postrun.load_checkpoint_model', return_value=(object(), final['checkpoint_identity'])) as load:
                process(case.run_id, root=root, device='cpu')
            self.assertEqual(load.call_count, 1)
            self.assertEqual(load.call_args.args[1].name, '50000')
            self.assertEqual(evaluate.call_args.kwargs['deadline'], '2999-01-01T00:00:00+00:00')
            self.assertTrue(read_json(wd / 'official/postrun_status.json')['official_complete'])

    def test_partial_run_evaluation_has_no_primary_or_official_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, cfg, data, ref, grid = fixture(root)
            wd = root / 'work_dir' / case.run_id
            atomic_json(wd / 'meta/training_status.json', dict(actual_updates=2020, training_complete=False))
            grid.update(records=grid['records'][:2], complete=False)
            atomic_json(wd / 'official/raw_grid.json', grid)
            with patch('g20.training.runtime_context', return_value=({'deadline_utc': '2999-01-01T00:00:00+00:00'}, None)), \
                 patch('g20.postrun.source_identity', return_value=grid['source_identity']), \
                 patch('g20.postrun.load_checkpoint_model') as load:
                status = process_partial(case.run_id, root=root, device='cpu')
            self.assertFalse(status['official_complete'])
            self.assertFalse(status['primary_selection_available'])
            self.assertTrue(status['partial_evaluation_complete'])
            self.assertEqual(status['actual_updates'], 2020)
            self.assertEqual(status['status'], 'PARTIAL_TIME_LIMIT')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
