"""Synthetic CPU fixtures for branch provenance; never campaign observations."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from qg40 import branch_evidence
from qg40.common import atomic_json, object_sha, read_json, sha256
from qg40.diagnostics import diagnostic_indices
from qg40.model import state_hash
from qg40.plan import cases_for, sensor_spec
from qg40.test_evaluation import record


class BranchEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = next(c for c in cases_for('s2') if c.baseline_id_preserved)
        self.run = self.case.run_id
        self.wd = self.root / 'work_dir' / self.run
        self.release = {'files': {'test-fixture': 'not-production'}, 'content_sha256': 'synthetic'}

    def diagnostic_fixture(self, step=24240):
        folder = self.wd / 'restart_fullstates' / str(step)
        folder.mkdir(parents=True, exist_ok=True)
        state = {'synthetic_weight': torch.tensor([float(step)])}
        save_file(state, str(folder / 'model.safetensors'))
        torch.save(dict(full_state=True, update=step, model_state=state), folder / 'training_state.pt')
        identity = dict(update=step, state_hash=state_hash(state), full_state=True,
                        model_sha256=sha256(folder / 'model.safetensors'),
                        training_state_sha256=sha256(folder / 'training_state.pt'),
                        config_sha256='cfg', data_sha256='data', source_identity=self.release,
                        reference_sha256='reference')
        atomic_json(folder / 'identity.json', identity)
        indices = diagnostic_indices(3072)
        arrays_path = self.wd / 'diagnostics' / f'update_{step}.npz'
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(diagnostic_indices=indices, gradient_indices=indices[:24],
                      native_delta=np.zeros((128, 2)), response_gain=np.zeros(128),
                      q=np.full(128, .46875), per_radius_q=np.zeros((128, 4)),
                      e_student=np.ones((128, 64, 64)), e_teacher=np.zeros((128, 64, 64)),
                      band_bias=np.zeros((128, 4)), edge_energy=np.zeros(128),
                      highpass_energy=np.zeros(128), teacher_q=np.full(128, .46875),
                      s=np.full(128, .5), advantage_fraction=np.ones(128))
        np.savez(arrays_path, **arrays)
        rows = [dict(index=int(index), soft_hard=dict(norm_ratio=.1, cosine=1.),
                     edge_hard=dict(norm_ratio=.2, cosine=-.5)) for index in indices[:24]]
        value = dict(schema='QG40_DIAGNOSTICS_v1', status='MEASURED', role='S',
                     synthetic_test=False, update=step, distribution_samples=128, gradient_samples=24,
                     diagnostic_indices_sha256=object_sha(indices.tolist()), gradient_rows=rows,
                     model_state_hash=identity['state_hash'], config_sha256='cfg', data_sha256='data',
                     source_identity=self.release, reference_sha256='reference',
                     arrays_path=str(arrays_path), arrays_sha256=sha256(arrays_path),
                     advantage_fraction=1., soft_hard_gradient_ratio_median=.1,
                     weighted_edge_hard_gradient_ratio=.2, edge_hard_gradient_cosine=-.5,
                     response_gain_median=0.)
        path = self.wd / 'diagnostics' / f'update_{step}.json'
        atomic_json(path, value)
        return path, value, arrays, folder

    def validate(self, step=24240):
        with patch.object(branch_evidence, 'source_identity', return_value=self.release):
            return branch_evidence.diagnostic(self.root, self.run, step)

    def test_diagnostic_rejects_synthetic_counts_and_changed_arrays(self):
        path, value, arrays, folder = self.diagnostic_fixture()
        observed, proof = self.validate()
        self.assertEqual(observed, value)
        self.assertEqual(proof['sha256'], sha256(path))
        for key, bad in [('synthetic_test', True), ('distribution_samples', 127), ('gradient_samples', 23)]:
            changed = dict(value, **{key: bad})
            atomic_json(path, changed)
            with self.assertRaises(ValueError):
                self.validate()
        atomic_json(path, value)
        arrays['native_delta'][0, 0] = 1
        np.savez(value['arrays_path'], **arrays)
        with self.assertRaises(ValueError):
            self.validate()

    def test_diagnostic_rejects_wrong_step_and_corrupt_checkpoint(self):
        path, value, arrays, folder = self.diagnostic_fixture()
        atomic_json(path, dict(value, update=50000))
        with self.assertRaises(ValueError):
            self.validate()
        atomic_json(path, value)
        (folder / 'model.safetensors').write_bytes(b'corrupted synthetic fixture')
        with self.assertRaises(ValueError):
            self.validate()

    def test_diagnostic_rejects_incomplete_arrays_despite_resealed_file(self):
        path, value, arrays, folder = self.diagnostic_fixture()
        arrays['native_delta'] = arrays['native_delta'][:-1]
        np.savez(value['arrays_path'], **arrays)
        atomic_json(path, dict(value, arrays_sha256=sha256(value['arrays_path'])))
        with self.assertRaises(ValueError):
            self.validate()

    def cross_context(self):
        data_path = self.root / 'dataset.json'
        atomic_json(data_path, {})
        cfg = {'qg40': {'dataset_manifest': str(data_path)}}
        grid = {'records': [record(24240), record(50000)]}
        atomic_json(self.wd / 'official/raw_grid.json', grid)
        for step in (24240, 50000):
            path = self.wd / 'diagnostics' / f'delta_{step}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, native_delta=np.full((128, 2), step / 50000))
        return cfg, grid

    def cross_mocks(self, cfg, grid, *, bad_q4=False):
        models = []
        def load(*args, **kwargs):
            model = torch.nn.Module()
            model.aligner = torch.nn.Linear(1, 1, bias=False)
            model.aligner.weight.data.fill_(len(models) + 1)
            models.append(model)
            return model, {'model_sha256': str(len(models))}
        calls = []
        def evaluate(model, *args, **kwargs):
            result = copy.deepcopy(grid['records'][min(len(calls), 1)])
            calls.append(model)
            if len(calls) == 1 and bad_q4:
                result['rr']['q4'] -= .1
            if len(calls) == 3:
                self.assertTrue(torch.equal(models[0].aligner.weight, models[1].aligner.weight))
                result['fr']['hqnr'] += .002
                result['fr']['d_s'] -= .002
            return {k: result[k] for k in ('rr', 'fr')}
        def diagnostic(_root, _run, step):
            return {'arrays_path': str(self.wd / 'diagnostics' / f'delta_{step}.npz')}, {}
        return [patch.object(branch_evidence, '_official', return_value=(cfg, grid, {})),
                patch.object(branch_evidence, 'source_identity', return_value=self.release),
                patch.object(branch_evidence, 'load_checkpoint_model', side_effect=load),
                patch.object(branch_evidence, 'diagnostic', side_effect=diagnostic),
                patch('qg40.data.build_dataset', return_value=object()),
                patch('qg40.evaluation.FRMetrics', return_value=object()),
                patch('qg40.evaluation.evaluate_model', side_effect=evaluate)]

    def test_diagonal_q4_mismatch_prevents_branch_certificate(self):
        from contextlib import ExitStack
        cfg, grid = self.cross_context()
        original = sha256(self.wd / 'official/raw_grid.json')
        with ExitStack() as stack:
            for mock in self.cross_mocks(cfg, grid, bad_q4=True):
                stack.enter_context(mock)
            with self.assertRaises(ValueError):
                branch_evidence.cross_aligner(self.root, self.run, 'cpu', None)
        self.assertFalse((self.wd / 'diagnostics/au_cross_24240_50000.json').exists())
        self.assertEqual(sha256(self.wd / 'official/raw_grid.json'), original)

    def test_valid_cross_is_diagnostic_and_preserves_normal_grid(self):
        from contextlib import ExitStack
        cfg, grid = self.cross_context()
        original = sha256(self.wd / 'official/raw_grid.json')
        with ExitStack() as stack:
            for mock in self.cross_mocks(cfg, grid):
                stack.enter_context(mock)
            result = branch_evidence.cross_aligner(self.root, self.run, 'cpu', None)
        self.assertTrue(result['diagnostic_only'])
        self.assertTrue(result['normal_target_unchanged'])
        self.assertTrue(result['diagonal_reproduced'])
        self.assertTrue(result['native_c_changed'])
        self.assertEqual(sha256(self.wd / 'official/raw_grid.json'), original)


class SensorBoundaryTests(unittest.TestCase):
    def test_inference_clipping_uses_qb_gf2_dn_and_preserves_training_tensor(self):
        from qg40.evaluation import infer
        class Dataset:
            has_gt = True
            def __len__(self):
                return 1
            def __getitem__(self, i):
                x = torch.zeros(4, 4, 4)
                return x, x, torch.zeros(4, 1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 4, 4), torch.tensor([0, 0])
        class Model(torch.nn.Module):
            def forward(self, pan, ms, lp):
                self.output = torch.tensor([-2., -1., 0., 2.]).reshape(1, 4, 1, 1).expand(1, 4, 4, 4)
                return {'y': self.output, 'delta': torch.zeros(1, 2)}
        for sensor, maximum in [('GF2', 1023), ('QB', 2047)]:
            data, model = Dataset(), Model()
            data.spec = sensor_spec(sensor)
            result, _ = infer(model, data, 'cpu')
            np.testing.assert_array_equal(result[0, :, 0, 0], [0., 0., maximum / 2, maximum])
            self.assertTrue(model.training)
            self.assertEqual(model.output[0, 0, 0, 0], -2.)
            self.assertEqual(model.output[0, 3, 0, 0], 2.)

    def test_cost_frontend_is_actual_c4_not_wv3_eight_bands(self):
        from qg40.postrun import profile_model
        class Dataset:
            spec = sensor_spec('GF2')
            def base(self, i):
                x = torch.zeros(4, 256, 256)
                return x, x, torch.zeros(4, 64, 64), torch.zeros(1, 64, 64), torch.zeros(1, 256, 256), torch.tensor([0, 0])
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, pan, ms, lp):
                return {'y': pan.expand(-1, 4, -1, -1) * self.weight, 'delta': torch.zeros(len(pan), 2)}
        with patch('thop.profile', return_value=(0., 0.)):
            measured = profile_model(Model(), Dataset(), 'cpu')
        self.assertEqual(measured['num_bands'], 4)
        self.assertEqual(measured['sensor'], 'GF2')
        self.assertAlmostEqual(measured['frontend_mac_equivalent_g'], (32 * 7 + 5) * 256 ** 2 / 2 / 1e9)
        self.assertIsNone(measured['mem_mb'])

if __name__ == '__main__':
    unittest.main()
