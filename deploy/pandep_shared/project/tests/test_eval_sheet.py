import ast
import copy
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pan_shared.common import CAMPAIGN, canonical_sha
from pan_shared import evaluate
from pan_shared.sheets import (HEADERS, SHEET_ID, SHEET_TITLE, SheetsUploader, SheetConflict,
                               build_row, cell_value, result_id, inspect_target)
from pan_shared.sheets import inspect_local_writers

ROOT = Path(__file__).resolve().parents[1]


def cell(value):
    kind = 'boolValue' if isinstance(value, bool) else 'numberValue' if type(value) in (int, float) else 'stringValue'
    return {'userEnteredValue': {kind: value}, 'effectiveValue': {kind: value}} if value != '' else {}


def observation():
    return dict(campaign_id=CAMPAIGN, server='s2', case_id='C00', mode='SHARED', sensor='WV3',
                run_id='PDSP_S2_R01_C00_W104_D122_F100_PLH_SE271001_A01', repeat=1, seed=271001, attempt=1,
                width=104, train_fraction=1., train_sensors=['WV3', 'GF2', 'QB'],
                selected_step=150000, completed_step=150000, checkpoint_sha256='a'*64,
                evaluator_sha256='b'*64, selector_scope='EXACT', protocol_id='PANDEP_NATIVE_RR20_FR20_v1',
                recorded_at='2026-10-01T12:00:00+09:00',
                rr=dict(n_scenes=20, ergas=2.1, scc=.98, sam=3., psnr=39., ssim=.97, q8=.91, rmse=20., cc=.98),
                fr=dict(n_scenes=20, hqnr=.95, d_s=.03, d_lambda=.02, jqm=None),
                notes={'source_sha256': 'c'*64, 'train_sensors': ['WV3', 'GF2', 'QB']})


class FakeSheet:
    def __init__(self):
        self.state = {'sheet_id': SHEET_ID, 'title': SHEET_TITLE, 'row_count': 3000,
                      'rows': {1: [cell('KEEP')]+[{}]*63, 5: [cell(v) for v in HEADERS]},
                      'merges': [], 'protected_ranges': []}
        self.state['rows'][5][0]['userEnteredValue'] = {'formulaValue': "=ARRAYFORMULA('_records'!$A$1:$BL$1)"}
        self.writes = []
        self.fail_after_write = False
        self.fail_readback = False

    def inspect(self):
        result = copy.deepcopy(self.state)
        if self.fail_readback and self.writes:
            result['rows'][self.writes[-1]][3] = cell(.01)
        return result

    def extend(self, required_rows):
        self.state['row_count'] = max(required_rows, self.state['row_count'])

    def write_row(self, number, row):
        old = self.state['rows'].get(number, [{}]*64)
        cells = [cell(v) for v in row]; cells[61] = old[61]
        self.state['rows'][number] = cells; self.writes.append(number)
        if self.fail_after_write:
            self.fail_after_write = False
            raise TimeoutError('timeout after committed write')

    def mark_verified(self, number):
        self.state['rows'][number][51] = cell('VERIFIED')


class SheetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.adapter = FakeSheet()
        self.uploader = SheetsUploader(self.adapter, self.temp.name)

    def test_mapping_exact_64_and_blank_unmeasured(self):
        row = build_row(observation())
        self.assertEqual(len(row), 64)
        self.assertEqual(row[25:31], ['WV3', 'Deployment', 'Shared PLH U-Net', CAMPAIGN,
                                             observation()['run_id'], 'C00'])
        self.assertEqual(row[15:25], ['']*10)
        self.assertEqual(row[6], '')
        self.assertEqual(row[41], 'a'*64)
        self.assertEqual(row[53], False)
        self.assertEqual(row[59], canonical_sha([CAMPAIGN, observation()['run_id'], 'a'*64, 'WV3', 'EXACT', 'b'*64]))

    def test_readonly_target_gate_never_mutates(self):
        before = copy.deepcopy(self.adapter.state)
        receipt = inspect_target(self.adapter)
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(receipt['writes'], 0)
        self.assertEqual(before, self.adapter.state)
        self.assertEqual(self.adapter.writes, [])

    def test_local_writer_detection_preserves_own_ancestry_other_tabs(self):
        process = '''1 0 init
10 1 python -m pan_shared.cli run --sheet-id 1198707876
11 10 python -m pan_shared.cli sync-sheet --sheet-id 1198707876
12 1 python research_uploader.py --sheet-id 123456
13 1 bash -lc python -m pan_shared.cli run --sheet-id 1198707876
'''
        self.assertEqual(inspect_local_writers(process, own_pid=11)['status'], 'PASS')
        with self.assertRaisesRegex(SheetConflict, 'PIDs 14'):
            inspect_local_writers(process+'14 1 python deploy_uploader.py --sheet-id 1198707876\n', own_pid=11)

    def test_global_target_writer_lock_crosses_outbox_roots(self):
        import fcntl
        shared_lock = Path(self.temp.name)/'common_target.writer.lock'
        self.adapter.writer_lock_path = shared_lock
        with shared_lock.open('a+') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(SheetConflict, 'another deployment'):
                self.uploader.upload_row(build_row(observation()))
            self.assertEqual(self.adapter.writes, [])
        self.assertEqual(self.uploader.upload_row(build_row(observation()))['status'], 'VERIFIED')

    def test_sensor_and_selector_change_result_identity(self):
        original = observation()
        for key, value in [('sensor', 'GF2'), ('selector_scope', 'BEST_JOINT_VAL_TO_B0003'), ('evaluator_sha256', 'c'*64)]:
            changed = dict(original, **{key: value})
            self.assertNotEqual(result_id(original), result_id(changed))

    def test_upload_retry_duplicate_noop_preserves_header(self):
        header = copy.deepcopy(self.adapter.state['rows'][5])
        o = observation(); self.uploader.enqueue(o)
        receipt = self.uploader.sync(now=0)
        self.assertEqual(receipt[0]['status'], 'VERIFIED')
        self.assertEqual(self.adapter.writes, [6])
        self.assertEqual(self.adapter.state['rows'][5], header)
        self.assertEqual(self.uploader.sync(now=100), [])
        direct = self.uploader.upload_row(build_row(o))
        self.assertTrue(direct['duplicate_noop'])
        self.assertEqual(self.adapter.writes, [6])

    def test_raw_formula_like_strings_and_manual_review_preserved(self):
        o = observation(); o['notes']['text'] = '=1+1'
        row = build_row(o); self.uploader.upload_row(row)
        self.adapter.state['rows'][6][61] = cell('사용자 검토 유지')
        self.uploader.upload_row(row)
        self.assertEqual(cell_value(self.adapter.state['rows'][6][61]), '사용자 검토 유지')
        self.assertNotIn('formulaValue', self.adapter.state['rows'][6][60]['userEnteredValue'])

    def test_unknown_far_right_and_note_only_rows_not_overwritten(self):
        self.adapter.state['rows'][37] = [{}]*63+[cell('manual BL')]
        self.adapter.state['rows'][41] = [{'note': 'reserved'}]+[{}]*63
        before = copy.deepcopy(self.adapter.state['rows'])
        receipt = self.uploader.upload_row(build_row(observation()))
        self.assertEqual(receipt['row_number'], 42)
        for number, row in before.items():
            self.assertEqual(self.adapter.state['rows'][number], row)

    def test_formula_data_region_blocked_even_non_destination(self):
        self.adapter.state['rows'][20] = [{'userEnteredValue': {'formulaValue': '=FILTER(A:A,A:A<>"")'}}]+[{}]*63
        with self.assertRaisesRegex(SheetConflict, 'formula/spill'):
            self.uploader.upload_row(build_row(observation()))
        self.assertFalse(self.adapter.writes)

    def test_effective_only_spill_is_occupied(self):
        self.adapter.state['rows'][20] = [{'effectiveValue': {'stringValue': 'spill'}}]+[{}]*63
        self.assertEqual(self.uploader.upload_row(build_row(observation()))['row_number'], 21)

    def test_bad_header_target_and_duplicate_id_refused(self):
        for key, value in [('sheet_id', 123), ('title', 'WV3-main')]:
            original = self.adapter.state[key]; self.adapter.state[key] = value
            with self.assertRaises(SheetConflict):
                self.uploader.upload_row(build_row(observation()))
            self.adapter.state[key] = original
        self.uploader.upload_row(build_row(observation()))
        self.adapter.state['rows'][7] = copy.deepcopy(self.adapter.state['rows'][6])
        with self.assertRaisesRegex(SheetConflict, 'duplicate'):
            self.uploader.upload_row(build_row(observation()))

    def test_immutable_row_and_outbox_payload_conflict(self):
        o = observation(); self.uploader.enqueue(o)
        changed = copy.deepcopy(o); changed['rr']['ergas'] = 99
        with self.assertRaisesRegex(SheetConflict, 'immutable'):
            self.uploader.enqueue(changed)
        self.uploader.upload_row(build_row(o))
        with self.assertRaisesRegex(SheetConflict, 'immutable'):
            self.uploader.upload_row(build_row(changed))

    def test_timeout_after_write_retries_without_duplicate(self):
        self.adapter.fail_after_write = True
        self.uploader.enqueue(observation())
        self.assertEqual(self.uploader.sync(now=0)[0]['status'], 'RETRY')
        self.assertEqual(self.uploader.sync(now=1), [])
        self.assertEqual(self.uploader.sync(now=10)[0]['status'], 'VERIFIED')
        self.assertEqual(self.adapter.writes, [6])

    def test_journal_precedes_envelope_and_recovers_interrupted_publication(self):
        from pan_shared import sheets
        real_atomic = sheets.atomic_json
        interrupted = []
        def fail_once(path, payload):
            if not interrupted:
                interrupted.append(True)
                raise OSError('interrupted after journal fsync')
            return real_atomic(path, payload)
        with patch.object(sheets, 'atomic_json', side_effect=fail_once):
            with self.assertRaises(OSError):
                self.uploader.enqueue(observation())
        lines = (Path(self.temp.name)/'observations.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(self.uploader.sync(now=0)[0]['status'], 'VERIFIED')
        self.uploader.enqueue(observation())
        self.assertEqual(len((Path(self.temp.name)/'observations.jsonl').read_text().splitlines()), 1)
        self.assertEqual(self.adapter.writes, [6])

    def test_failed_readback_never_marked_verified(self):
        self.adapter.fail_readback = True
        self.uploader.enqueue(observation())
        self.assertEqual(self.uploader.sync(now=0)[0]['status'], 'BLOCKED')
        self.assertEqual(cell_value(self.adapter.state['rows'][6][51]), 'PENDING')

    def test_only_target_rows_extended(self):
        self.adapter.state['rows'][3000] = [cell('old last row')]+[{}]*63
        receipt = self.uploader.upload_row(build_row(observation()))
        self.assertEqual(receipt['row_number'], 3001)
        self.assertEqual(self.adapter.state['row_count'], 3001)

    def test_nan_and_fake_or_partial_observations_refused(self):
        for change in ({'case_id': 'Q00'}, {'seed': 0}, {'server': 's1'}, {'status': 'RUNNING'}):
            with self.assertRaises(ValueError): build_row(dict(observation(), **change))
        o = observation(); o['rr']['ergas'] = float('nan')
        with self.assertRaises(ValueError): build_row(o)
        o = observation(); o['rr']['n_scenes'] = 19
        with self.assertRaises(ValueError): build_row(o)


class DummyModel(torch.nn.Module):
    def forward(self, sensor, pan, ms, lpan):
        # Deliberately advance RNG to prove evaluator restores it.
        random.random(); np.random.rand(); torch.rand(1)
        return torch.zeros(pan.shape[0], 8 if sensor == 'WV3' else 4, *pan.shape[-2:])


class ValidationDataset(torch.utils.data.Dataset):
    augment = False
    def __init__(self, sensor, targets): self.sensor, self.targets = sensor, targets
    def __len__(self): return len(self.targets)
    def __getitem__(self, index):
        bands = 8 if self.sensor == 'WV3' else 4
        return dict(pan=torch.ones(1, 16, 16), ms=torch.ones(bands, 4, 4), lpan=torch.ones(1, 4, 4),
                    gt=torch.full((bands, 16, 16), self.targets[index], dtype=torch.float32))


class EvaluationTests(unittest.TestCase):
    def test_macro_validation_sample_weighting_and_rng_mode_restore(self):
        model = DummyModel().train()
        py, npstate, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        datasets = {'WV3': ValidationDataset('WV3', [.1, .2, .3, .4, .5]),
                    'GF2': ValidationDataset('GF2', [.6, .6]), 'QB': ValidationDataset('QB', [.9])}
        report = evaluate.validation(model, datasets, 'cpu', batch_size=4)
        self.assertAlmostEqual(report['sensors']['WV3']['mean_l1'], .3, places=7)
        self.assertAlmostEqual(report['joint_l1'], .6, places=7)
        self.assertTrue(model.training)
        self.assertEqual(py, random.getstate()); self.assertTrue(np.array_equal(npstate[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(ts, torch.get_rng_state()))

    def test_validation_unclipped_loss(self):
        result = evaluate.validation(DummyModel(), {'WV3': ValidationDataset('WV3', [2.])}, 'cpu')
        self.assertEqual(result['joint_l1'], 2.)

    def test_official_geometry_strict_20(self):
        with self.assertRaisesRegex(ValueError, 'exactly 20'):
            evaluate.rr_metrics(np.zeros((19, 8, 256, 256)), np.zeros((19, 8, 256, 256)), 'WV3')
        with self.assertRaisesRegex(ValueError, 'full20'):
            evaluate.fr_metrics(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), 'GF2')

    def test_rr20_wrapper_crop_and_sensor_metrics(self):
        from types import SimpleNamespace
        source = (ROOT/'vendor_reference'/'metrics'/'ablr2'/'evaluation.py').read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'rr_metrics')
        scope = dict(vars(evaluate), _spec=lambda value: value,
                     rr_keys=lambda spec: ('ergas', 'scc', 'psnr', 'sam', 'ssim', 'rmse', 'cc', 'q'+str(spec.num_bands)))
        scope['q2n'] = lambda *args: (.9, None)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'fixed_ablr2_rr', 'exec'), scope)
        rng = np.random.default_rng(31)
        for sensor, bands, maximum in [('WV3', 8, 2047), ('GF2', 4, 1023), ('QB', 4, 2047)]:
            # Full protocol geometry and 20 per-scene means; Q has independent byte/numerical parity test.
            target = rng.uniform(10, maximum-10, (20, bands, 256, 256)).astype(np.float32)
            pred = target + rng.normal(0, 1, target.shape).astype(np.float32)
            expected = pred[0].transpose(1, 2, 0)[20:-21, 20:-21]
            with patch.object(evaluate, 'q2n', return_value=(.9, None)) as quality:
                report = evaluate.rr_metrics(pred, target, sensor)
            self.assertEqual(report['n_scenes'], 20)
            self.assertEqual(quality.call_count, 20)
            self.assertEqual(quality.call_args[0][0].shape, expected.shape)
            self.assertAlmostEqual(report['q'+str(bands)], .9)
            self.assertEqual(report['crop'], '20:-21')
            self.assertEqual(report['ergas'], np.mean([r['ergas'] for r in report['per_scene']]))
            original = scope['rr_metrics'](pred, target, SimpleNamespace(sensor=sensor, num_bands=bands, max_dn=maximum))
            for key in ('ergas', 'sam', 'scc', 'psnr', 'ssim', 'rmse', 'cc', 'q'+str(bands)):
                self.assertEqual(report[key], original[key], (sensor, key))

    def test_fr20_wrapper_matches_fixed_native_reference(self):
        import contextlib
        from types import SimpleNamespace
        from pan_shared.metrics import eval_fr, interp23, jqm
        from pan_shared.common import file_sha
        source = (ROOT/'vendor_reference'/'metrics'/'ablr2'/'evaluation.py').read_text()
        # Only change import namespaces for the isolated fixture; numerical class body remains fixed.
        source = source.replace('from tools.metrics.eval_fr import load_dlpan, imresize_matlab, _blockproc_uqi',
                                'from pan_shared.metrics.eval_fr import imresize_matlab, _blockproc_uqi')
        source = source.replace('from tools.metrics.eval_fr import mtf_filter, _blockproc_uqi',
                                'from pan_shared.metrics.eval_fr import mtf_filter, _blockproc_uqi')
        nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in ('FRMetrics', 'signed_ds_details')]
        scope = dict(vars(evaluate), _spec=lambda value: value, canonical_band_indices=lambda spec: tuple(range(spec.num_bands)),
                     sha256=file_sha, check_deadline=lambda deadline: None, FR_KEYS=('hqnr', 'd_lambda', 'd_s'))
        import h5py
        scope['h5py'] = h5py
        scope['q2n'] = lambda *args: (.9, None)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'fixed_ablr2_fr', 'exec'), scope)
        rng = np.random.default_rng(214)
        for sensor, bands, maximum in [('WV3', 8, 2047), ('GF2', 4, 1023), ('QB', 4, 2047)]:
            fused = np.broadcast_to(rng.uniform(10, maximum-10, (1, bands, 512, 512)), (20, bands, 512, 512))
            lms = np.broadcast_to(rng.uniform(10, maximum-10, (1, bands, 512, 512)), fused.shape)
            pan = np.broadcast_to(rng.uniform(10, maximum-10, (1, 1, 512, 512)), (20, 1, 512, 512))
            ms = np.broadcast_to(rng.uniform(10, maximum-10, (1, bands, 128, 128)), (20, bands, 128, 128))
            spec = SimpleNamespace(sensor=sensor, num_bands=bands, max_dn=maximum, band_order=list(range(bands)))
            dataset = SimpleNamespace(spec=spec, raw_h5_path='synthetic_fixture_only')
            with patch.object(evaluate, 'q2n', return_value=(.9, None)), \
                 patch.object(jqm, 'jqm', return_value={'JQM': .7}), \
                 patch.object(h5py, 'File', side_effect=lambda *args: contextlib.nullcontext({'lms': lms, 'pan': pan})):
                actual = evaluate.fr_metrics(fused, lms, pan, ms, sensor)
                reference = scope['FRMetrics'](dataset, wald=interp23)(fused)
            for key in ('hqnr', 'd_s', 'd_lambda'):
                self.assertEqual(actual[key], reference[key], (sensor, key))
            self.assertFalse(actual['masking'])
            self.assertEqual(actual['support'], 'full512')
            self.assertEqual(actual['n_scenes'], 20)
            self.assertEqual(actual['signed_ds'], reference['signed_ds']['signed_mean'])

    def test_actual_weight_sha_and_cache_reuse(self):
        import h5py
        from safetensors.torch import save_file
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = torch.nn.Linear(1, 1)
            weight = root/'weights.safetensors'; save_file(model.state_dict(), str(weight))
            raw = root/'raw.h5'
            with h5py.File(raw, 'w') as handle:
                for key in ('gt', 'pan', 'ms', 'lms'): handle[key] = np.ones(1)
            datasets = {'WV3': {'rr': SimpleNamespace(source_path=raw), 'fr': SimpleNamespace(source_path=raw)}}
            metric = dict(n_scenes=20, per_scene=[{'x': 1}]*20)
            with patch.object(evaluate, 'infer', return_value=np.ones(1)) as inference, \
                 patch.object(evaluate, 'rr_metrics', return_value=metric), \
                 patch.object(evaluate, 'fr_metrics', return_value=metric):
                first = evaluate.benchmark_checkpoint(model, weight, datasets, 'cpu', root/'cache')
                second = evaluate.benchmark_checkpoint(model, weight, datasets, 'cpu', root/'cache')
                self.assertEqual(first, second)
                self.assertEqual(inference.call_count, 2)
                with torch.no_grad(): model.weight.add_(1)
                with self.assertRaisesRegex(ValueError, 'evaluated tensors'):
                    evaluate.benchmark_checkpoint(model, weight, datasets, 'cpu', root/'cache')

    def test_export_rejects_unapproved_case_and_missing_provenance(self):
        from pan_shared.export import export_candidate
        with self.assertRaisesRegex(ValueError, 'Only C00'):
            export_candidate(None, {}, {'case_id': 'C11', 'repeat': 1, 'mode': 'SHARED'}, '/tmp', 'EXACT')
        run = dict(case_id='C00', repeat=1, mode='SHARED', sensors=['WV3', 'GF2', 'QB'], width=104, depth=[1,2,2], train_fraction=1.)
        with self.assertRaisesRegex(ValueError, 'Complete source'):
            export_candidate(None, {'completed_step': 150000}, run, '/tmp', 'EXACT')

    def test_candidate_export_complete_whitelist_and_idempotency(self):
        from pan_shared.model import build_model
        from pan_shared.registry import build_registry
        from pan_shared.checkpoints import tensor_state_hash
        from pan_shared.common import file_sha
        from pan_shared.export import export_candidate
        from safetensors.torch import save_file
        run = build_registry()['runs'][0]
        model = build_model(run)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); weight = work/'model.safetensors'
            save_file(model.state_dict(), str(weight))
            identity = dict(completed_step=150000, model_path=str(weight), checkpoint_sha256=file_sha(weight),
                            tensor_state_sha256=tensor_state_hash(model.state_dict()))
            provenance = {k: 'a'*64 for k in ('source_sha256','config_sha256','data_sha256','subset_sha256','protocol_sha256')}
            path = export_candidate(model, identity, run, work, 'EXACT', provenance=provenance)
            self.assertEqual(path, export_candidate(model, identity, run, work, 'EXACT', provenance=provenance))
            manifest = json.loads((path/'release_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'candidate')
            self.assertFalse(manifest['commercial_release_authorized'])
            self.assertFalse(manifest['tile_inference_validated'])
            self.assertEqual(set(manifest['reference_fixtures']), {'WV3','GF2','QB'})
            names = [str(p.relative_to(path)) for p in path.rglob('*')]
            self.assertFalse(any('optimizer' in name or '/metrics/' in name for name in names))
            self.assertEqual(file_sha(path/'model.safetensors'), identity['checkpoint_sha256'])

    def test_primitive_numeric_parity_fixed_reference_functions(self):
        from pan_shared.metrics import eval_rr, extras, q2n, eval_fr, interp23
        bindings = [('tools/metrics/eval_rr.py', eval_rr, ['sam', 'ergas']),
                    ('tools/eval_dlpan.py', extras, ['scc_dlpan', 'psnr_global', 'ssim_skimage']),
                    ('tools/metrics/eval_fr.py', eval_fr, ['imresize_matlab', 'mtf_filter', '_blockproc_uqi'])]
        rng = np.random.default_rng(491)
        for source, current, names in bindings:
            reference = (ROOT/'vendor_reference'/'metrics'/source).read_text()
            scope = dict(vars(current))
            nodes = [n for n in ast.parse(reference).body if isinstance(n, ast.FunctionDef) and n.name in names]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), source, 'exec'), scope)
            for sensor, bands, maximum in [('WV3', 8, 2047), ('GF2', 4, 1023), ('QB', 4, 2047)]:
                target = rng.uniform(10, maximum-10, (64, 64, bands)); pred = target + rng.normal(0, 1, target.shape)
                for name in names:
                    args = (pred, target)
                    if name in ('psnr_global', 'ssim_skimage'): args += (maximum,)
                    elif name == 'imresize_matlab': args = (target[..., 0], .25)
                    elif name == 'mtf_filter': args = (target, sensor.lower(), 4)
                    elif name == '_blockproc_uqi': args = (pred[..., 0], target[..., 0], 32)
                    self.assertTrue(np.array_equal(getattr(current, name)(*args), scope[name](*args)), (sensor, name))
        # Unmodified Q4/Q8 and JQM reference files are byte-identical fixed copies.
        for name in ('q2n', 'jqm'):
            self.assertEqual((ROOT/'src'/'pan_shared'/'metrics'/f'{name}.py').read_bytes(),
                             (ROOT/'vendor_reference'/'metrics'/'tools'/'metrics'/f'{name}.py').read_bytes())
        # Exercise both unchanged hypercomplex sizes and JQM surrogate numerically,
        # independently of the light stand-ins used in full20 wrapper aggregation tests.
        from pan_shared.metrics import jqm
        for sensor, bands, maximum in [('WV3', 8, 2047), ('GF2', 4, 1023), ('QB', 4, 2047)]:
            reference_scopes = {}
            for name in ('q2n', 'jqm'):
                scope = {'__name__': 'isolated_reference_'+name, '__package__': 'pan_shared.metrics'}
                exec(compile((ROOT/'vendor_reference'/'metrics'/'tools'/'metrics'/f'{name}.py').read_text(),
                             'fixed_'+name, 'exec'), scope)
                reference_scopes[name] = scope
            target = rng.uniform(10, maximum-10, (64, 64, bands)); fused = target + rng.normal(0, 2, target.shape)
            self.assertEqual(q2n.q2n(target, fused, 32, 32)[0], reference_scopes['q2n']['q2n'](target, fused, 32, 32)[0])
            ms = rng.uniform(10, maximum-10, (16, 16, bands)); pan = rng.uniform(10, maximum-10, (64, 64))
            actual = jqm.jqm(fused, ms, pan, sensor, R=maximum)
            original = reference_scopes['jqm']['jqm'](fused, ms, pan, sensor, R=maximum)
            for key in ('JQM', 'QLR', 'QHR'):
                self.assertEqual(actual[key], original[key], (sensor, key))
            self.assertTrue(np.array_equal(actual['w'], original['w']))
        reference = (ROOT/'vendor_reference'/'metrics'/'wald_utilities.py').read_text()
        node = next(n for n in ast.parse(reference).body if isinstance(n, ast.FunctionDef) and n.name == 'interp23tap')
        scope = dict(vars(interp23)); exec(compile(ast.Module(body=[node], type_ignores=[]), 'wald', 'exec'), scope)
        fixture = rng.normal(size=(16, 16, 4))
        self.assertTrue(np.array_equal(interp23.interp23tap(fixture, 4), scope['interp23tap'](fixture, 4)))

    def test_evaluator_hash_is_stable_and_protocol_bound(self):
        value = evaluate.evaluator_sha()
        self.assertEqual(len(value), 64)
        self.assertEqual(value, evaluate.evaluator_sha())
        with patch.dict(evaluate.PROTOCOL, {'id': 'different'}):
            self.assertNotEqual(value, evaluate.evaluator_sha())

    def test_import_does_not_pull_legacy_controllers(self):
        import sys
        forbidden = ('ablr2.controller', 'qg40.controller', 'fh12.controller', 'reporting_bridge')
        self.assertFalse(any(name.startswith(forbidden) for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
