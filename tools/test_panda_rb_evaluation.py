"""CPU-only protocol, geometry and hostile-artifact tests; no real inference run."""
from pathlib import Path
import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from fh12.model import FH12Model, sync_frontend
from fh12.common import atomic_json, object_sha, sha256
from panda_rb import evaluation as ev, stress as st
from panda_rb.plan import shifts, shift_grid, VAL_GRID
from panda_rb.reference_probe import probe_indices, _correlation


class FixedA(torch.nn.Module):
    def forward(self, pan, ms):
        return pan.new_tensor([[.75, -1.25]]).expand(len(pan), -1)


class TinyU(torch.nn.Module):
    def forward(self, pan, lp, ms, switches, x_in):
        return (x_in[:, :1] + .2 * x_in[:, 1:2] + .3 * x_in[:, 2:3]).expand(-1, 8, -1, -1)


def selection_manifest(winner=50000, alias=True):
    records = [dict(update=x, val_ergas=1. if x == winner else 2.) for x in VAL_GRID]
    return dict(complete=True, actual_updates=50000, validation_records=records,
                primary=dict(selection_id='EXACT_50000', update=50000, checkpoint_sha256='exact', directory='exact50000'),
                secondary=dict(selection_id='RR_VAL_ERGAS_MIN', update=winner, checkpoint_sha256='exact' if alias else 'val',
                               directory='val_selected', alias_of='EXACT_50000' if alias else None))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(71)

    def tensors(self, size=64):
        return (torch.randn(1, 1, size, size, generator=self.gen),
                torch.randn(1, 8, size // 4, size // 4, generator=self.gen),
                torch.randn(1, 1, size // 4, size // 4, generator=self.gen))

    def test_grid_exactly49_eight_directions_and_six_l2_radii(self):
        grid = shift_grid(); points = st.validate_grid(grid)
        self.assertEqual(len(points), 49)
        self.assertEqual(sum(p['radius_hr'] == 0 for p in points), 1)
        for point in points:
            self.assertAlmostEqual(np.linalg.norm([point['dy'], point['dx']]), point['radius_hr'])
        bad = copy.deepcopy(grid); bad['shifts'][4]['dy'] += .1
        with self.assertRaises(ValueError): st.validate_grid(bad)

    def test_impulse_and_ramp_sign(self):
        result = st.warp_convention_test()
        self.assertTrue(result['passed']); self.assertEqual(result['positive_dx_impulse_peak'], [32, 31])

    def test_zero_reuses_original_native_pan_and_lp_without_filter_or_warp(self):
        pan, ms, lp = self.tensors()
        with patch.object(st, 'regenerate_lp', side_effect=AssertionError('must not filter')), \
             patch.object(st, 'warp_pan', side_effect=AssertionError('must not first-warp')):
            result = st.shifted_inputs(pan, lp, torch.zeros(1, 2))
        self.assertIs(result[0], pan); self.assertIs(result[1], lp)

    def test_nonzero_shifts_pan_once_then_recomputes_lp_from_shifted_pan(self):
        pan, _ms, lp = self.tensors(); eps = torch.tensor([[.25, -.5]])
        original_warp = st.warp_pan
        with patch.object(st, 'warp_pan', wraps=original_warp) as warp, \
             patch.object(st, 'regenerate_lp', wraps=st.regenerate_lp) as low:
            shifted, newlp = st.shifted_inputs(pan, lp, eps)
        self.assertEqual(warp.call_count, 1); self.assertEqual(low.call_count, 1)
        self.assertIs(low.call_args.args[0], shifted)
        self.assertEqual(newlp.shape, lp.shape); self.assertFalse(torch.equal(newlp, lp))
        self.assertTrue(torch.equal(shifted, original_warp(pan, eps)))

    def test_lp_filter_matches_exact_fh12_float64_dn_float32_cache_recipe(self):
        pan, _ms, _lp = self.tensors()
        dn = (pan.numpy().astype(np.float64) + 1) * 1023.5
        expected = torch.from_numpy(st.make_lpan(dn).astype(np.float32)).mul_(2 / 2047).sub_(1)
        self.assertTrue(torch.equal(st.regenerate_lp(pan), expected))

    def test_zero_override_applies_pan_lp_and_signed_hp_not_only_pan(self):
        model = FH12Model(TinyU(), FixedA(), 'PLH'); pan, ms, lp = self.tensors()
        out, estimated, p, l = st.stress_forward(model, pan, ms, lp, torch.zeros(1, 2), 'A_ZERO_INFERENCE_ONLY')
        reference = sync_frontend(pan, ms, lp, torch.zeros(1, 2), 'PLH')
        self.assertTrue(torch.equal(estimated, torch.tensor([[.75, -1.25]])))
        self.assertTrue(torch.equal(out['delta'], torch.zeros(1, 2)))
        for key in ('pan_aligned', 'L', 'H', 'x_in'):
            self.assertTrue(torch.equal(out[key], reference[key]), key)
        normal = model(pan, ms, lp)
        self.assertFalse(torch.equal(out['pan_aligned'], normal['pan_aligned']))
        self.assertFalse(torch.equal(out['L'], normal['L']))

    def test_normal_zero_shift_matches_native_forward_bitwise(self):
        model = FH12Model(TinyU(), FixedA(), 'PLH'); pan, ms, lp = self.tensors()
        out, *_ = st.stress_forward(model, pan, ms, lp, torch.zeros(1, 2), 'A_ON')
        self.assertTrue(torch.equal(out['y'], model(pan, ms, lp)['y']))

    def test_relative_response_is_coordinate_sum_and_sampling_sign(self):
        native = torch.tensor([[.5, -.25]])
        eps = torch.tensor([[1., -2.]])
        self.assertEqual(float(st.relative_response(native - eps, eps, native)), 0)
        self.assertEqual(float(st.relative_response(native, eps, native)), 3)

    def test_coverage_tracks_all_frontend_paths_and_preserves_outliers(self):
        eps = torch.tensor([[0., 0.], [4., 0.], [4., 0.], [4., 0.]])
        applied = torch.tensor([[0., 0.], [-4., 0.], [30., 0.], [300., 0.]])
        original = applied.clone()
        result = st.geometric_coverage(256, 256, eps, applied)
        self.assertEqual(float(result['all_pan_paths'][0]), 1.)
        self.assertEqual(float(result['all_pan_paths'][1]), 1.)
        self.assertLess(float(result['all_pan_paths'][2]), float(result['pan'][2]))
        self.assertEqual(float(result['all_pan_paths'][3]), 0.)
        self.assertTrue(torch.equal(original, applied))

    def test_geometry_rejects_nonfinite_not_silent_clamp(self):
        with self.assertRaises(FloatingPointError):
            st.geometric_coverage(256, 256, torch.zeros(1, 2), torch.tensor([[float('nan'), 0.]]))


class MetricTests(unittest.TestCase):
    def test_stress_metrics_all_use_the_same192_roi(self):
        y, x = np.indices((256, 256)); truth = np.stack([300 + x + y + c * 5 for c in range(8)])
        pred = truth.astype(float) + 2
        baseline = st.stress_metrics(pred, truth)
        pred[:, :32] += 1000; pred[:, -32:] += 1000
        pred[:, :, :32] += 1000; pred[:, :, -32:] += 1000
        self.assertEqual(st.stress_metrics(pred, truth), baseline)
        pred[:, 32, 32] += 100
        self.assertGreater(st.stress_metrics(pred, truth)['ergas'], baseline['ergas'])
        with self.assertRaises(ValueError): st.stress_metrics(pred, truth, margin=20)

    def test_edge_error_is_signed_not_edge_magnitude_matching(self):
        x = np.broadcast_to(np.arange(5), (8, 5, 5))
        self.assertEqual(st.signed_edge_error(x, x), 0)
        self.assertAlmostEqual(st.signed_edge_error(-x, x), .6)

    def test_metric_nan_on_discarded_border_rejected(self):
        a = np.ones((8, 256, 256)); a[0, 0, 0] = np.nan
        with self.assertRaises(FloatingPointError): st.stress_metrics(a, np.ones_like(a))

    def test_raw_contract_requires_all20_chw_float32_dn(self):
        a = np.broadcast_to(np.float32(100), (20, 8, 256, 256))
        ev._raw_checks(a, 'rr')
        for bad in (a[:19], a.astype(np.float64), a + 2047, a - 101):
            with self.assertRaises(ValueError): ev._raw_checks(bad, 'rr')

    def test_fixed_probe_selection_does_not_consume_global_rng(self):
        state = np.random.get_state(); a = probe_indices(1080); b = probe_indices(1080)
        self.assertTrue(np.array_equal(a, b)); self.assertEqual(len(a), 256)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0]); self.assertTrue(np.array_equal(state[1], after[1]))
        self.assertEqual(state[2:], after[2:])
        self.assertEqual(probe_indices(3).tolist(), [0, 1, 2])

    def test_correlations_undefined_are_null_not_fake_zero(self):
        self.assertIsNone(_correlation([1, 1, 1], [1, 2, 3]))
        self.assertEqual(_correlation([1, 2, 3], [3, 2, 1]), -1)

    def test_curve_preserves_all20_failures_and_never_partial_means(self):
        points = shifts(); rows = []
        for shift in points:
            for scene in range(20):
                rows.append(dict(scene_index=scene, shift_id=shift['id'], status='ok',
                    **{k: 1. for k in (*st.STRESS_KEYS, 'relative_response_l1_sum', 'coverage_all_pan_paths')}))
        rows[-1].update(status='nonfinite_prediction_or_correction', ergas=None)
        curve = ev._curve(rows, points)
        self.assertEqual(curve[-1]['n_scenes'], 20); self.assertEqual(curve[-1]['n_failures'], 1)
        self.assertIsNone(curve[-1]['ergas']); self.assertIsNone(curve[-1]['delta_from_zero_ergas'])
        with self.assertRaises(ValueError): ev._curve(rows[:-1], points)


class SelectionAndArtifactTests(unittest.TestCase):
    def test_exact_primary_val_secondary_alias(self):
        a, b = ev.selected_checkpoints(selection_manifest())
        self.assertEqual(a['update'], 50000); self.assertEqual(b['alias_of'], 'EXACT_50000')

    def test_val_selection_earliest_tie_and_no_test_hqnr_use(self):
        manifest = selection_manifest(1010, alias=False)
        manifest['validation_records'][1]['val_ergas'] = 1.
        manifest['validation_records'][1]['hqnr'] = 1.
        self.assertEqual(ev.selected_checkpoints(manifest)[1]['update'], 1010)
        manifest['secondary']['update'] = 2020
        with self.assertRaises(ValueError): ev.selected_checkpoints(manifest)

    def test_incomplete_wrong_grid_nonfinite_and_unaliased_same_sha_rejected(self):
        for edit in (lambda x: x.update(actual_updates=49999),
                     lambda x: x['validation_records'].pop(),
                     lambda x: x['validation_records'][0].update(val_ergas=float('nan')),
                     lambda x: x['secondary'].update(alias_of=None)):
            manifest = selection_manifest(); edit(manifest)
            with self.assertRaises(ValueError): ev.selected_checkpoints(manifest)

    def test_raw_storage_is_immutable_chw_and_checksum_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); p = root / 'testimg0.npy'; array = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
            meta = ev.save_array(p, array)
            self.assertEqual(np.load(p).dtype, np.float32); self.assertTrue(np.array_equal(np.load(p), array))
            ev.save_array(p, array)
            report = ev.seal(dict(complete=True, file_hashes={'testimg0.npy': meta['file_sha256']}))
            self.assertEqual(ev.validate_artifacts(report, root), report)
            with self.assertRaises(ValueError): ev.save_array(p, array + 1)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError): ev.validate_artifacts(report, root)

    def test_forged_payload_and_unsafe_relative_artifact_rejected(self):
        report = ev.seal(dict(complete=True, value=1, file_hashes={}))
        report['value'] = 2
        with self.assertRaises(ValueError): ev.validate_artifacts(report, '/tmp')
        for path in ('../secret', '/secret'):
            report = ev.seal(dict(complete=True, file_hashes={path: 'bad'}))
            with self.assertRaises(ValueError): ev.validate_artifacts(report, '/tmp')


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.run = 'RB01_WV3_S1_R1_SS9281101_QFULL_F50K_v1'; self.wd = self.root / 'new_run'
        self.binding = dict(frozen='F1', version=1)
        self.source = dict(files={}, content_sha256=object_sha({}), tf32_matmul=False, tf32_cudnn=True)
        self.data = dict(splits={s: dict(count=20) for s in ('rr', 'fr')})
        self.cfg = dict(seed=9281101, panda_rb=dict(run_id=self.run, case_id='QFULL'))
        (self.wd / 'meta').mkdir(parents=True)
        (self.wd / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(self.cfg))
        self.manifest = selection_manifest()
        for folder in ('exact50000', 'val_selected'):
            path = self.wd / 'checkpoints' / folder; path.mkdir(parents=True)
            (path / 'model.safetensors').write_bytes(b'same test checkpoint bytes')
            identity = dict(update=50000, model_sha256=sha256(path / 'model.safetensors'),
                config_sha256=object_sha(self.cfg), binding_sha256=object_sha(self.binding), source_identity=self.source)
            atomic_json(path / 'identity.json', identity)
        for key in ('primary', 'secondary'):
            self.manifest[key]['checkpoint_sha256'] = identity['model_sha256']
        atomic_json(self.wd / 'checkpoints/selection_manifest.json', self.manifest)
        patches = [patch.object(ev, 'case_for', return_value=dict(seed=9281101, case_id='QFULL')),
                   patch.object(ev, 'run_dir', return_value=self.wd),
                   patch.object(ev, 'source_identity', return_value=self.source),
                   patch.object(ev, 'evaluator_identity', return_value={'version': 'test'}),
                   patch('panda_rb.binding.load_binding', return_value=(None, None, self.binding, self.data, None))]
        for p in patches: p.start(); self.addCleanup(p.stop)

    def context(self): return ev._context(self.run, self.root / 'binding.json', self.root)

    def edit_identity(self, **changes):
        path = self.wd / 'checkpoints/exact50000/identity.json'
        doc = ev.read_json(path); doc.update(changes); atomic_json(path, doc)

    def test_context_binds_f1_config_release_grid_and_weight_bytes(self):
        ctx = self.context()
        self.assertEqual(ctx['context']['binding_sha256'], object_sha(self.binding))
        self.assertEqual(ctx['selected'][0]['update'], 50000)

    def test_wrong_binding_rejected_before_any_inference(self):
        self.edit_identity(binding_sha256='other Teacher')
        with self.assertRaisesRegex(ValueError, 'F1 binding'): self.context()

    def test_changed_source_config_or_step_rejected(self):
        for change in (dict(source_identity={}), dict(config_sha256='wrong'), dict(update=49999)):
            original = ev.read_json(self.wd / 'checkpoints/exact50000/identity.json')
            self.edit_identity(**change)
            with self.assertRaises(ValueError): self.context()
            atomic_json(self.wd / 'checkpoints/exact50000/identity.json', original)

    def test_changed_weight_bytes_rejected(self):
        (self.wd / 'checkpoints/exact50000/model.safetensors').write_bytes(b'corruption')
        with self.assertRaises(ValueError): self.context()

    def test_selected_checkpoint_cannot_escape_the_new_run(self):
        import shutil
        path = self.wd / 'checkpoints/val_selected'; external = self.root / 'external'
        shutil.move(path, external); path.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'): self.context()

    def test_rr19_cannot_be_used(self):
        self.data['splits']['rr']['count'] = 19
        with self.assertRaisesRegex(ValueError, '20'): self.context()


class NativeIntegrationTests(unittest.TestCase):
    def test_native_same_sha_alias_writes40_ordered_raw_files_and_reuses_verified_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            wd = Path(temp); primary, secondary = ev.selected_checkpoints(selection_manifest())
            ctx = dict(wd=wd, selected=(primary, secondary), data={'splits': {}},
                       context=dict(source_identity=dict(tf32_matmul=False, tf32_cudnn=True)))
            for split in ('rr', 'fr'):
                ctx['data']['splits'][split] = dict(dataroot='native.h5', sha256='native',
                    lpan_path='lp.h5', lpan_sha256='lp', sample_order_sha256='ordered')
            class Dataset:
                arrays = {'ms': None, 'pan': None}
            rr = np.broadcast_to(np.float32(100), (20, 8, 256, 256))
            fr = np.broadcast_to(np.float32(100), (20, 8, 512, 512))
            normal_rr = dict(ergas=2., scc=.9, per_scene=[dict(ergas=2., scc=.9)] * 20)
            normal_fr = dict(hqnr=.95, per_scene=[dict(hqnr=.95)] * 20)
            extra = dict(per_scene=[dict(rmse=1., cc=.9)] * 20)
            with patch.object(ev, '_context', return_value=ctx), \
                 patch.object(ev, 'build_dataset', return_value=Dataset()), \
                 patch.object(ev, '_model', return_value=None), \
                 patch.object(ev, 'infer', side_effect=[(rr, np.zeros((20, 2))), (fr, np.zeros((20, 2)))]) as infer, \
                 patch.object(ev, 'native_gt', return_value=None), \
                 patch.object(ev, 'rr_metrics', return_value=normal_rr), \
                 patch.object(ev, 'rr_extra', return_value=extra), \
                 patch.object(ev, 'fr_extra', return_value=extra), \
                 patch.object(ev, 'FRMetrics', return_value=lambda x: normal_fr):
                report = ev.evaluate_native('test', '/tmp/binding', root=wd, device='cpu')
                self.assertEqual(infer.call_count, 2)
                self.assertEqual(report['n_unique_checkpoints'], 1)
                self.assertEqual(report['selections']['RR_VAL_ERGAS_MIN']['alias_of'], 'EXACT_50000')
                files = list((wd / 'native/Ours_raw').rglob('testimg*.npy'))
                self.assertEqual(len(files), 40)
                for split, expected in (('reduced', (8, 256, 256)), ('full', (8, 512, 512))):
                    for i in range(20):
                        array = np.load(wd / 'native/Ours_raw/WV3' / split / 'pred' / f'testimg{i}.npy')
                        self.assertEqual(array.shape, expected); self.assertEqual(array.dtype, np.float32)
                self.assertEqual(ev.evaluate_native('test', '/tmp/binding', root=wd, device='cpu'), report)
                self.assertEqual(infer.call_count, 2)
                (files[0]).write_bytes(b'corruption')
                with self.assertRaises(ValueError): ev.evaluate_native('test', '/tmp/binding', root=wd, device='cpu')


class StressIntegrationTests(unittest.TestCase):
    def test_full49x20_curve_cache_and_bounded_failure_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            wd = Path(temp); primary, secondary = ev.selected_checkpoints(selection_manifest())
            ctx = dict(wd=wd, selected=(primary, secondary), data={},
                       context=dict(source_identity=dict(tf32_matmul=False, tf32_cudnn=True)))
            atomic_json(wd / 'native/metrics.json', ev.seal(dict(complete=True, context=ctx['context'], file_hashes={})))
            class Dataset:
                def __len__(self): return 20
            class Model:
                def predict_delta(self, pan, base): return torch.zeros(1, 2)
            pan = torch.zeros(1, 1, 256, 256); ms = torch.zeros(1, 8, 64, 64); lp = torch.zeros(1, 1, 64, 64)
            pred = torch.zeros(1, 8, 256, 256)
            result = dict(y=pred, delta=torch.zeros(1, 2))
            fake_metrics = {k: 1. for k in st.STRESS_KEYS}
            with patch.object(ev, '_context', return_value=ctx), \
                 patch.object(ev, 'build_dataset', return_value=Dataset()), \
                 patch.object(ev, '_batch', return_value=(None, ms, lp, pan)), \
                 patch.object(ev, '_model', return_value=Model()), \
                 patch.object(ev, 'stress_forward', return_value=(result, torch.zeros(1, 2), pan, lp)) as forward, \
                 patch.object(ev, 'stress_metrics', return_value=fake_metrics), \
                 patch.object(ev, 'geometric_coverage', return_value={k: torch.tensor([.75]) for k in ('pan', 'lp', 'all_pan_paths')}):
                # Fixed original grid is real; temporary root only owns artifacts.
                with patch.object(ev, 'shift_grid', return_value=shift_grid()):
                    report = ev.evaluate_stress('test', '/tmp/binding', 'A_ZERO_INFERENCE_ONLY', root=wd, device='cpu')
                    self.assertEqual(forward.call_count, 980)
                    self.assertEqual(report['n_observations'], 980); self.assertEqual(report['n_invalid_geometry'], 980)
                    self.assertEqual(report['n_shifts'], 49); self.assertEqual(report['n_scenes'], 20)
                    self.assertEqual(len(list((wd / 'stress/A_ZERO_INFERENCE_ONLY/raw').glob('*.npy'))), 14)
                    self.assertEqual(ev.evaluate_stress('test', '/tmp/binding', 'A_ZERO_INFERENCE_ONLY', root=wd, device='cpu'), report)
                    self.assertEqual(forward.call_count, 980)
                    artifact = next((wd / 'stress/A_ZERO_INFERENCE_ONLY/scenes').glob('*.json'))
                    artifact.write_bytes(b'corruption')
                    with self.assertRaises(ValueError):
                        ev.evaluate_stress('test', '/tmp/binding', 'A_ZERO_INFERENCE_ONLY', root=wd, device='cpu')


class ReferenceProbeTests(unittest.TestCase):
    def test_reference_only_probe_uses_fixed_val_indices_and_no_extra_student_repeats(self):
        from panda_rb import reference_probe as rp
        class Dataset:
            def __len__(self): return 3
            def base(self, i):
                g = torch.Generator().manual_seed(i + 900)
                pan = torch.rand(1, 64, 64, generator=g)
                ms = torch.rand(8, 16, 16, generator=g)
                lp = torch.rand(1, 16, 16, generator=g)
                gt = torch.rand(8, 64, 64, generator=g)
                return gt, gt.clone(), ms, lp, pan, torch.tensor([i, 0, 0, 0])
        model = FH12Model(TinyU(), FixedA(), 'PLH').eval().requires_grad_(False)
        data = dict(splits={'val': {'sha256': 'validation-only'}})
        source = dict(tf32_matmul=False, tf32_cudnn=True)
        with tempfile.TemporaryDirectory() as temp, \
             patch('panda_rb.binding.load_binding', return_value=(model, {}, {}, data, None)), \
             patch.object(rp, 'build_dataset', return_value=Dataset()) as dataset, \
             patch.object(rp, 'shift_grid', return_value=shift_grid()), \
             patch('panda_rb.common.source_identity', return_value=source):
            report = rp.reference_probe('/tmp/binding', root=Path(temp), device='cpu', batch_size=2)
            self.assertEqual(report['n_reference_patches'], 3)
            self.assertEqual(report['n_independent_students'], 0)
            self.assertEqual(report['n_independent_teachers'], 1)
            self.assertFalse(report['identity']['test_inputs_used'])
            self.assertEqual(report['identity']['indices'], [0, 1, 2])
            self.assertEqual(len(report['identity']['shifts']), 16)
            self.assertEqual(dataset.call_args.args[1], 'val')
            self.assertEqual(rp.reference_probe('/tmp/binding', root=Path(temp), device='cpu'), report)


class PreflightStorageTests(unittest.TestCase):
    def test_remaining_budget_rechecked_without_counting_alias_files_twice(self):
        from panda_rb import preflight as pf
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); wd = root / 'one'; wd.mkdir()
            (wd / 'original').write_bytes(b'x' * 40)
            (wd / 'alias').symlink_to(wd / 'original')
            estimate = dict(estimated_per_run_bytes=100, required_bytes=800, runs=8)
            with patch('panda_rb.training.required_storage_bytes', return_value=estimate), \
                 patch.object(pf, 'training_runs', return_value=[{'run_id': 'one'}]), \
                 patch.object(pf, 'run_dir', return_value=wd), \
                 patch.object(pf.shutil, 'disk_usage', return_value=SimpleNamespace(free=10 * 1024**3)):
                result = pf.storage_admission('s1', root)
                self.assertEqual(result['already_retained_bytes'], 40)
                self.assertEqual(result['estimated_remaining_bytes'], 5 * 1024**3 + 60)
            with patch('panda_rb.training.required_storage_bytes', return_value=estimate), \
                 patch.object(pf, 'training_runs', return_value=[]), \
                 patch.object(pf.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
                with self.assertRaises(OSError): pf.storage_admission('s1', root)

    def test_production_preflight_requires_pinned_docker_image(self):
        from panda_rb import preflight as pf
        with patch.object(pf, 'docker_required'), patch('torch.cuda.is_available', return_value=True), \
             patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'image identity'):
                pf.prepare('s1', '/tmp/missing', root=Path('/tmp'), device='cuda')


class ExternalEvaluatorTests(unittest.TestCase):
    def test_external_wald_bytes_are_part_of_portable_evaluator_identity(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'wald.py'; path.write_bytes(b'frozen interp23tap code')
            module = SimpleNamespace(__file__=str(path), interp23tap=lambda a, ratio: np.repeat(np.repeat(a, ratio, 0), ratio, 1))
            with patch('tools.metrics.eval_fr.load_dlpan', return_value=module), \
                 patch.object(ev, '_base_evaluator_identity', return_value={'files': {'metric.py': 'bytes'}}):
                first = ev.evaluator_identity(Path(temp))
                self.assertEqual(first['external_wald_sha256'], sha256(path))
                path.write_bytes(b'different external numerical code')
                second = ev.evaluator_identity(Path(temp))
                self.assertNotEqual(first['content_sha256'], second['content_sha256'])
                module.interp23tap = lambda a, ratio: a
                with self.assertRaises(ValueError): ev.evaluator_identity(Path(temp))

    def test_missing_external_wald_does_not_fall_back_to_a_different_metric(self):
        with patch('tools.metrics.eval_fr.load_dlpan', side_effect=FileNotFoundError('missing Wald mount')):
            with self.assertRaises(FileNotFoundError): ev.evaluator_identity('/tmp')


if __name__ == '__main__':
    unittest.main()
