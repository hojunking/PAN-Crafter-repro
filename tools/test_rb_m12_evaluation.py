"""CPU-only M12 modes/ROI/coverage/selector/evidence regression tests."""
from pathlib import Path
import copy
import csv
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from fh12.model import FH12Model, sync_frontend
from panda_rb_m12 import evaluation as ev, stress as st, reference_probe as rp
from panda_rb_m12.common import atomic_json, object_sha, sha256


def grid():
    points = [dict(id='D000', dy=0., dx=0., radius_hr=0., angle_degrees=None)]
    for radius in (.25, .5, 1., 2., 3., 4.):
        for angle in range(0, 360, 45):
            points.append(dict(id=f'D{len(points):03d}', dy=radius*math.sin(math.radians(angle)),
                               dx=radius*math.cos(math.radians(angle)), radius_hr=radius, angle_degrees=angle))
    return dict(shifts=points)


def selection_manifest(winner=50000, alias=True):
    from panda_rb_m12.plan import VAL_GRID
    return dict(complete=True, actual_updates=50000,
                validation_records=[dict(update=x, val_ergas=1. if x==winner else 2.) for x in VAL_GRID],
                primary=dict(selection_id='EXACT_50000', update=50000, checkpoint_sha256='exact', directory='exact50000'),
                secondary=dict(selection_id='RR_VAL_ERGAS_MIN', update=winner, checkpoint_sha256='exact' if alias else 'val',
                               directory='val_selected', alias_of='EXACT_50000' if alias else None))


class FixedA(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer('offset', torch.tensor([[.75, -1.25]]))
    def forward(self, pan, ms):
        return self.offset.to(pan).expand(len(pan), -1)


class TinyU(torch.nn.Module):
    def forward(self, pan, lp, ms, switches, x_in):
        return (x_in[:, :1]+.2*x_in[:, 1:2]+.3*x_in[:, 2:3]).expand(-1, 8, -1, -1)


class ModeTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(777)
        self.pan = torch.randn(1, 1, 64, 64, generator=g)
        self.ms = torch.randn(1, 8, 16, 16, generator=g)
        self.lp = torch.randn(1, 1, 16, 16, generator=g)
        self.model = FH12Model(TinyU(), FixedA(), 'PLH').eval()

    def test_grid_and_known_sign(self):
        self.assertEqual(len(st.validate_grid(grid())), 49)
        self.assertTrue(st.warp_convention_test()['passed'])
        invalid = grid(); invalid['shifts'][1]['dx'] *= -1
        with self.assertRaises(ValueError): st.validate_grid(invalid)

    def test_d000_all_three_nonzero_modes_equal_native_bitwise_and_reuse_inputs(self):
        native = self.model(self.pan, self.ms, self.lp)
        for mode in ('A_ON', 'A_NATIVE_FIXED', 'KNOWN_SHIFT_INVERSE'):
            out, estimated, p, lp = st.stress_forward(self.model, self.pan, self.ms, self.lp, torch.zeros(1, 2), mode)
            self.assertIs(p, self.pan); self.assertIs(lp, self.lp)
            self.assertTrue(torch.equal(out['y'], native['y']))
            self.assertTrue(torch.equal(out['delta'], native['delta']))

    def test_all_four_modes_use_corresponding_hr_dydx_and_same_sync_frontend(self):
        epsilon = torch.tensor([[1., -2.]])
        c0 = torch.tensor([[.75, -1.25]])
        for mode, expected in (('A_ON', c0), ('A_ZERO_INFERENCE_ONLY', torch.zeros_like(c0)),
                               ('A_NATIVE_FIXED', c0), ('KNOWN_SHIFT_INVERSE', c0-epsilon)):
            out, estimated, p, lp = st.stress_forward(self.model, self.pan, self.ms, self.lp, epsilon, mode)
            self.assertTrue(torch.equal(out['delta'], expected))
            self.assertFalse(torch.equal(p, self.pan)); self.assertFalse(torch.equal(lp, self.lp))
            reference = sync_frontend(p, self.ms, lp, expected, 'PLH')
            for key in ('x_in', 'pan_aligned', 'L', 'H'):
                self.assertTrue(torch.equal(out[key], reference[key]), (mode, key))

    def test_wrong_mode_rejected(self):
        with self.assertRaises(ValueError):
            st.stress_forward(self.model, self.pan, self.ms, self.lp, torch.zeros(1, 2), 'TRAIN_NO_A')

    def test_known_inverse_cancels_added_shift_but_not_native_correction(self):
        c0 = torch.tensor([[.75, -1.25]]); epsilon = torch.tensor([[1., -2.]])
        self.assertEqual(float(st.relative_response(c0-epsilon, epsilon, c0)), 0.)
        self.assertEqual(float(st.relative_response(c0, epsilon, c0)), 3.)
        self.assertFalse(torch.equal(c0-epsilon, -epsilon))

    def test_clone_parity_checks_actual_parameters_buffers_modes_and_outputs(self):
        base = torch.nn.functional.interpolate(self.ms, scale_factor=4, mode='bicubic', align_corners=False)
        clone = copy.deepcopy(self.model)
        self.assertTrue(rp.verify_clone(self.model, clone, self.pan, base)['passed'])
        clone.aligner.offset.add_(.1)
        with self.assertRaises(ValueError): rp.verify_clone(self.model, clone, self.pan, base)
        clone = copy.deepcopy(self.model).train()
        with self.assertRaises(ValueError): rp.verify_clone(self.model, clone, self.pan, base)


class ROITests(unittest.TestCase):
    def arrays(self):
        y, x = np.indices((256, 256))
        truth = np.stack([300+x+y+c*5 for c in range(8)])
        return truth.astype(float)+2, truth

    def test_primary160_aux192_same_fullprediction_prespecified(self):
        pred, truth = self.arrays()
        a, b = st.stress_metrics(pred, truth), st.stress_metrics(pred, truth, 'fixed192')
        pred[:, 35, 35] += 1000
        self.assertEqual(st.stress_metrics(pred, truth), a)
        self.assertGreater(st.stress_metrics(pred, truth, 'fixed192')['ergas'], b['ergas'])
        with self.assertRaises(ValueError): st.stress_metrics(pred, truth, 'adaptive160')

    def test_nan_in_discarded_border_remains_failure(self):
        pred, truth = self.arrays(); pred[:, 0, 0] = np.nan
        with self.assertRaises(FloatingPointError): st.stress_metrics(pred, truth)

    def test_edge_metric_is_signed(self):
        x = np.broadcast_to(np.arange(5), (8, 5, 5))
        self.assertAlmostEqual(st.signed_edge_error(-x, x), .6)

    def test_support_checks_both_warps_filter_lp_upsample_and_preserves_outliers(self):
        eps = torch.tensor([[4., 0.], [4., 0.], [4., 0.]])
        applied = torch.tensor([[-4., 0.], [40., 0.], [300., 0.]])
        original = applied.clone()
        a = st.geometric_coverage(256, 256, eps, applied, margin=48)
        b = st.geometric_coverage(256, 256, eps, applied, margin=32)
        self.assertEqual(float(a['all_pan_paths'][0]), 1.)
        self.assertLess(float(a['all_pan_paths'][1]), 1.)
        self.assertEqual(float(a['all_pan_paths'][2]), 0.)
        self.assertTrue(torch.all(a['all_pan_paths'] >= b['all_pan_paths']))
        self.assertTrue(torch.equal(original, applied))

    def test_crop_manifest20_by11_is_fixed_and_pan_ms_share_coordinates(self):
        m = rp.crop_manifest()
        self.assertEqual(m['scene_indices'], list(range(20)))
        self.assertEqual([r['size'] for r in m['crops']], [64]*5+[128]*5+[256])
        image = torch.arange(256*256).reshape(1, 1, 256, 256)
        for row in m['crops']:
            cropped = rp._crop(image, row)
            self.assertEqual(cropped.shape[-2:], (row['size'], row['size']))
            self.assertEqual(int(cropped[0, 0, 0, 0]), row['top']*256+row['left'])


class AggregateTests(unittest.TestCase):
    def rows(self):
        return [dict(scene_index=i, shift_id=point['id'], status='ok', roi='fixed160',
                     **{k: 1. for k in (*st.STRESS_KEYS, 'relative_response_l1_sum', 'coverage_all_pan_paths')})
                for point in grid()['shifts'] for i in range(20)]

    def test_all49_points_20scenes_and8directions_mandatory(self):
        rows = self.rows(); result = st.summarize_curve(rows, grid())
        self.assertEqual(len(result['points']), 49); self.assertEqual(len(result['radii']), 7)
        self.assertEqual(result['radii'][-1]['n_expected'], 160)
        self.assertEqual(result['radii'][-1]['clean_ergas'], 1.)
        with self.assertRaises(ValueError): st.summarize_curve(rows[:-1], grid())
        rows[-1]['scene_index'] = 18
        with self.assertRaises(ValueError): st.summarize_curve(rows, grid())

    def test_single_support_failure_nulls_cleanradius_but_retains_allraw_means(self):
        rows = self.rows(); rows[-1].update(status='coverage_failure', coverage_all_pan_paths=.5, ergas=21.)
        result = st.summarize_curve(rows, grid())
        last = result['radii'][-1]
        self.assertEqual(last['n_valid'], 159); self.assertEqual(last['n_failures'], 1)
        self.assertIsNone(last['clean_ergas']); self.assertAlmostEqual(last['flagged_all_ergas'], 1.125)
        self.assertEqual(result['points'][-1]['ergas'], 2.)
        self.assertIsNone(result['points'][-1]['clean_ergas'])

    def test_nonfinite_no_available_case_average(self):
        rows = self.rows(); rows[-1].update(status='nonfinite_metric', ergas=None)
        result = st.summarize_curve(rows, grid())
        self.assertIsNone(result['radii'][-1]['flagged_all_ergas'])
        self.assertIsNone(result['points'][-1]['ergas'])

    def test_roi_mix_rejected(self):
        rows = self.rows(); rows[-1]['roi'] = 'fixed192'
        with self.assertRaises(ValueError): st.summarize_curve(rows, grid())


class ArtifactTests(unittest.TestCase):
    def test_selector_exact_and_val_alias(self):
        a, b = ev.selected_checkpoints(selection_manifest())
        self.assertEqual(a['update'], 50000); self.assertEqual(b['alias_of'], 'EXACT_50000')

    def test_val_earliest_tie_ignores_test_hqnr(self):
        m = selection_manifest(1010, False); m['validation_records'][1].update(val_ergas=1., hqnr=1.)
        self.assertEqual(ev.selected_checkpoints(m)[1]['update'], 1010)
        m['secondary']['update'] = 2020
        with self.assertRaises(ValueError): ev.selected_checkpoints(m)

    def test_bad_grid_nan_missing_alias_rejected(self):
        for change in (lambda m: m.update(actual_updates=49999), lambda m: m['validation_records'].pop(),
                       lambda m: m['validation_records'][0].update(val_ergas=float('nan')),
                       lambda m: m['secondary'].update(alias_of=None)):
            m = selection_manifest(); change(m)
            with self.assertRaises(ValueError): ev.selected_checkpoints(m)

    def test_supplemental_scene_index_never_raises_duplicate_dict_keyword(self):
        row = ev._scene_record({'scene_index': 5, 'ergas': 2.}, {'scene_index': 999, 'source_h5_row': -1},
                               'rr', 5, [.25, -.5], 'digest')
        self.assertEqual(row['scene_index'], 5); self.assertEqual(row['source_h5_row'], 5)

    def test_raw_contract_rr20_not19_and_no_rounding(self):
        array = np.broadcast_to(np.float32(100.125), (20, 8, 256, 256))
        ev._raw_checks(array, 'rr')
        for bad in (array[:19], array.astype(np.float64), array+2047):
            with self.assertRaises(ValueError): ev._raw_checks(bad, 'rr')

    def test_immutable_array_sha_and_symlink_escape_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); owned = base/'owned'; owned.mkdir()
            value = np.arange(8, dtype=np.float32)
            metadata = ev.save_array(owned/'raw.npy', value)
            ev.save_array(owned/'raw.npy', value)
            with self.assertRaises(ValueError): ev.save_array(owned/'raw.npy', value+1)
            report = ev.seal(dict(complete=True, file_hashes={'raw.npy': metadata['file_sha256']}))
            self.assertEqual(ev.validate_artifacts(report, owned), report)
            (owned/'escape').symlink_to(base)
            bad = ev.seal(dict(complete=True, file_hashes={'escape/secret': 'wrong'}))
            with self.assertRaises(ValueError): ev.validate_artifacts(bad, owned)

    def test_native_alias_runs_rrfr_once_and_verified_cache_never_reinfer(self):
        with tempfile.TemporaryDirectory() as temp:
            wd = Path(temp); selections = ev.selected_checkpoints(selection_manifest())
            ctx = dict(wd=wd, selected=selections, data={'splits': {}},
                       context={'source_identity': {'tf32_matmul': False, 'tf32_cudnn': True}})
            for s in ('rr', 'fr'):
                ctx['data']['splits'][s] = {k: 'fixed' for k in ('dataroot', 'sha256', 'lpan_path', 'lpan_sha256', 'sample_order_sha256')}
            class Dataset:
                arrays = {'ms': None, 'pan': None}
            tiny = np.zeros((20, 8, 2, 2), dtype=np.float32)
            rr = dict(ergas=2., scc=.9, per_scene=[dict(ergas=2., scc=.9)]*20)
            fr = dict(hqnr=.95, per_scene=[dict(hqnr=.95)]*20)
            extra = dict(per_scene=[dict(scene_index=i+100, rmse=1.) for i in range(20)])
            with patch.object(ev, '_context', return_value=ctx), patch.object(ev, 'build_dataset', return_value=Dataset()), \
                 patch.object(ev, '_model', return_value=None), patch.object(ev, 'infer', return_value=(tiny, np.zeros((20, 2)))) as infer, \
                 patch.object(ev, '_raw_checks'), patch.object(ev, 'native_gt'), patch.object(ev, 'rr_metrics', return_value=rr), \
                 patch.object(ev, 'rr_extra', return_value=extra), patch.object(ev, 'fr_extra', return_value=extra), \
                 patch.object(ev, 'FRMetrics', return_value=lambda pred: fr):
                result = ev.evaluate_native('M12_test', 'binding', root=wd, device='cpu')
                self.assertEqual(infer.call_count, 2); self.assertEqual(result['n_unique_checkpoints'], 1)
                self.assertEqual(result['selections']['RR_VAL_ERGAS_MIN']['alias_of'], 'EXACT_50000')
                self.assertEqual(len(list((wd/'native/Ours_raw').rglob('*.npy'))), 40)
                self.assertEqual(ev.evaluate_native('M12_test', 'binding', root=wd, device='cpu'), result)
                self.assertEqual(infer.call_count, 2)
                (wd/'native/Ours_raw/WV3/reduced/pred/testimg0.npy').write_bytes(b'corrupt')
                with self.assertRaises(ValueError): ev.evaluate_native('M12_test', 'binding', root=wd, device='cpu')


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.wd = self.root/'run'; self.seed = 261001101
        self.run = 'new_M12_run'
        self.cfg = dict(seed=self.seed, panda_rb_m12=dict(run_id=self.run, case_id='QFULL'))
        self.source = dict(files={}, content_sha256=object_sha({}), tf32_matmul=False, tf32_cudnn=True)
        self.binding = dict(common_sha256='common', teacher_checkpoint_sha256='F1', origin_reference={'q_cache_sha256': 'q'},
                            q_ref=.1, tau_R=.01, teacher_config_sha256='config')
        self.data = dict(splits={s: {'count': 20} for s in ('rr', 'fr')})
        self.weights = dict(arrays_sha256={'q': 'q-sha', 'e_bar': 'e-sha'})
        self.initial = dict(hashes={'body': 'U', 'A': 'A', 'full': 'all'})
        self.stream = dict(seed=self.seed, updates=50000, full_stream_sha256='actual')
        (self.wd/'meta').mkdir(parents=True)
        (self.wd/'meta/config.resolved.yaml').write_text(yaml.safe_dump(self.cfg))
        atomic_json(self.wd/'init_manifest.json', self.initial)
        atomic_json(self.wd/'stream_manifest.json', self.stream)
        atomic_json(self.wd/'diagnostics/consumed_weights.json', {'consumed_stream_sha256': 'actual'})
        atomic_json(self.root/'campaign/common/weights'/f'seed_{self.seed}.json', self.weights)
        manifest = selection_manifest()
        manifest.update(initialization_sha256=object_sha(self.initial), consumed_stream_sha256='actual')
        for folder in ('exact50000', 'val_selected'):
            path = self.wd/'checkpoints'/folder; path.mkdir(parents=True)
            (path/'model.safetensors').write_bytes(b'same-checkpoint')
            identity = dict(update=50000, model_sha256=sha256(path/'model.safetensors'),
                            config_sha256=object_sha(self.cfg), source_identity=self.source,
                            binding_sha256=object_sha(self.binding), weights_sha256=object_sha(self.weights))
            atomic_json(path/'identity.json', identity)
        for key in ('primary', 'secondary'): manifest[key]['checkpoint_sha256'] = identity['model_sha256']
        atomic_json(self.wd/'checkpoints/selection_manifest.json', manifest)
        patches = [patch('panda_rb_m12.plan.case_for', return_value=dict(seed=self.seed, case_id='QFULL')),
                   patch('panda_rb_m12.plan.run_dir', return_value=self.wd),
                   patch('panda_rb_m12.common.campaign_dir', return_value=self.root/'campaign'),
                   patch('panda_rb_m12.binding.load_binding', return_value=(None, None, self.binding, self.data, None)),
                   patch.object(ev, 'source_identity', return_value=self.source),
                   patch.object(ev, 'evaluator_identity', return_value={'version': 'test'})]
        for item in patches: item.start(); self.addCleanup(item.stop)

    def context(self): return ev._context(self.run, 'binding', self.root)

    def test_context_binds_realinitial_actualstream_teacher_qe_and_checkpoint(self):
        ctx = self.context()['context']
        self.assertEqual(ctx['initialization_sha256'], object_sha(self.initial))
        self.assertEqual(ctx['consumed_stream_sha256'], 'actual')
        self.assertEqual(ctx['raw_map_identity_sha256'], object_sha(self.weights['arrays_sha256']))
        self.assertEqual(ctx['teacher_sha256'], 'F1')

    def test_different_actual_stream_not_planned_as_actual(self):
        atomic_json(self.wd/'diagnostics/consumed_weights.json', {'consumed_stream_sha256': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'actual committed'): self.context()

    def test_changed_initial_or_weightmap_refused(self):
        atomic_json(self.wd/'init_manifest.json', {'hashes': 'other'})
        with self.assertRaisesRegex(ValueError, 'Initial'): self.context()
        atomic_json(self.wd/'init_manifest.json', self.initial)
        atomic_json(self.root/'campaign/common/weights'/f'seed_{self.seed}.json', {'arrays_sha256': 'changed'})
        with self.assertRaisesRegex(ValueError, 'Raw q/e'): self.context()

    def test_rr19_is_not_an_allowed_native_protocol(self):
        self.data['splits']['rr']['count'] = 19
        with self.assertRaisesRegex(ValueError, '20 RR'): self.context()

    def test_corrupt_weights_or_other_binding_fails_before_inference(self):
        p = self.wd/'checkpoints/exact50000/identity.json'
        identity = ev.read_json(p); identity['binding_sha256'] = 'other'; atomic_json(p, identity)
        with self.assertRaisesRegex(ValueError, 'frozen F1'): self.context()


class IntegrationTests(unittest.TestCase):
    def test_stress_all980_scenes_two_rois_same_forward_and_bounded_failure_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            wd = Path(temp); selected = ev.selected_checkpoints(selection_manifest())
            ctx = dict(wd=wd, selected=selected, data={}, context=dict(case={'case_id': 'QFULL'},
                       source_identity={'tf32_matmul': False, 'tf32_cudnn': True}))
            atomic_json(wd/'native/metrics.json', ev.seal(dict(complete=True, context=ctx['context'], file_hashes={})))
            class Dataset:
                def __len__(self): return 20
            class Model:
                def predict_delta(self, pan, base): return torch.zeros(1, 2)
            pan = torch.zeros(1, 1, 256, 256); ms = torch.zeros(1, 8, 64, 64); lp = torch.zeros(1, 1, 64, 64)
            result = dict(y=torch.zeros(1, 8, 256, 256), delta=torch.zeros(1, 2))
            with patch.object(ev, '_context', return_value=ctx), patch.object(ev, 'build_dataset', return_value=Dataset()), \
                 patch.object(ev, '_batch', return_value=(None, ms, lp, pan)), patch.object(ev, '_model', return_value=Model()), \
                 patch.object(ev, 'stress_forward', return_value=(result, torch.zeros(1, 2), pan, lp)) as forward, \
                 patch.object(ev, 'stress_metrics', return_value={key: 1. for key in st.STRESS_KEYS}) as metrics, \
                 patch.object(ev, 'geometric_coverage', return_value={k: torch.tensor([.75]) for k in ('pan', 'lp', 'all_pan_paths')}), \
                 patch('panda_rb_m12.plan.shift_grid', return_value=grid()):
                report = ev.evaluate_stress('M12_test', 'binding', 'A_ZERO_INFERENCE_ONLY', root=wd, device='cpu')
                self.assertEqual(forward.call_count, 980); self.assertEqual(metrics.call_count, 1960)
                self.assertEqual(report['n_observations'], 980); self.assertEqual(report['n_valid'], 0)
                self.assertEqual(report['n_invalid_geometry'], 980); self.assertEqual(report['status'], 'PROCESSED_WITH_FLAGS')
                self.assertEqual(len(report['curve']), 49); self.assertEqual(len(report['auxiliary_curve']), 49)
                self.assertIsNone(report['radii'][-1]['clean_ergas'])
                self.assertEqual(report['radii'][-1]['flagged_all_ergas'], 1.)
                self.assertEqual(len(list((wd/'stress/A_ZERO_INFERENCE_ONLY/raw').glob('*.npy'))), 14)
                self.assertEqual(ev.evaluate_stress('M12_test', 'binding', 'A_ZERO_INFERENCE_ONLY', root=wd, device='cpu'), report)
                self.assertEqual(forward.call_count, 980)

    def test_response_weak_cancellation_passes_actual_four_clone_parity_and_fixed_allcrops(self):
        class Dataset:
            def __len__(self): return 20
            def base(self, i):
                gt = torch.zeros(8, 256, 256); ms = torch.zeros(8, 64, 64)
                lp = torch.zeros(1, 64, 64); pan = torch.zeros(1, 256, 256)
                return gt, gt, ms, lp, pan, torch.tensor([i, 0, 0, 0])
        teacher = FH12Model(TinyU(), FixedA(), 'PLH').eval().requires_grad_(False)
        rows = [dict(server='s1', case_id='QFULL', repeat=i, seed=261001100+i, run_id=f'new{i}') for i in range(1, 5)]
        data = dict(splits={'rr': dict(count=20, sha256='frozenRR20', sample_order_sha256='order')})
        binding = dict(teacher_checkpoint_sha256='sameF1')
        source = dict(tf32_matmul=False, tf32_cudnn=True)
        with tempfile.TemporaryDirectory() as temp, \
             patch('panda_rb_m12.binding.load_binding', return_value=(teacher, {}, binding, data, None)), \
             patch('panda_rb_m12.plan.training_runs', return_value=rows), patch('panda_rb_m12.plan.shift_grid', return_value=grid()), \
             patch.object(rp, 'source_identity', return_value=source), patch.object(rp, 'build_dataset', return_value=Dataset()), \
             patch.object(rp, 'build_model', side_effect=lambda *args, **kwargs: (copy.deepcopy(teacher), {'hashes': {'A': 'fixed'}})), \
             patch.object(rp, 'array_digest', return_value='deterministic_mock_digest'):
            report = rp.response_probe('binding', 's1', root=Path(temp), device='cpu', recover_old=False)
            self.assertEqual(report['n_independent_students'], 0); self.assertEqual(report['n_independent_teachers'], 1)
            self.assertEqual(report['n_initial_clones'], 4); self.assertEqual(report['n_observations'], 20*49*11)
            self.assertEqual(len(report['initial_clone_parity']), 4*11*3)
            self.assertIn('weak response does not fail', report['outcome'])
            self.assertEqual(rp.response_probe('binding', 's1', root=Path(temp), device='cpu', recover_old=False), report)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
