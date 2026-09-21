"""Small CPU regressions for horizon, routing, prefix, clone and exact recovery."""
import copy
from contextlib import ExitStack
import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from fh12.training import BatchStream, atomic_torch
from g20.losses import student_losses as existing_loss
from g20.model import build_model, state_hash
from l100.common import atomic_json, object_sha, sha256
from l100.losses import student_losses, routed_student_backward
from l100.plan import CAMPAIGN_ID, build_config, case_for
from l100.training import (aligner_factor, cosine_factor, make_scheduler,
                            recover_final_status, runtime_context, _clone_check, diagnostic_ids, train_run)


class LocalTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_scheduler_horizon_boundaries_and_midpoint_not_fresh50(self):
        for total in (50000, 100000):
            self.assertEqual(cosine_factor(0, total=total), 0)
            self.assertEqual(cosine_factor(100, total=total), 1)
            self.assertEqual(cosine_factor(total, total=total), 0)
        self.assertGreater(cosine_factor(50000, total=100000), .5)
        self.assertEqual(aligner_factor(0, 'ARW'), .1)
        self.assertEqual(aligner_factor(5000, 'ARW'), .1)
        self.assertAlmostEqual(aligner_factor(7500, 'ARW'), .55)
        self.assertEqual(aligner_factor(10000, 'ARW'), 1)
        with self.assertRaises(ValueError):
            cosine_factor(100001, total=100000)
        with self.assertRaisesRegex(ValueError, 'DEFERRED'):
            build_config(case_for('L100I1-X01'))

    def test_scheduler_state_resume_keeps_identical_next_lrs(self):
        for total in (50000, 100000):
            def create():
                optimizer = torch.optim.AdamW([dict(params=[torch.nn.Parameter(torch.ones(1))], lr=1e-4, name='U'),
                                               dict(params=[torch.nn.Parameter(torch.ones(1))], lr=3e-6, name='A')])
                return optimizer, make_scheduler(optimizer, 'H010', total=total)
            opt, scheduler = create()
            for _ in range(173):
                opt.step(); scheduler.step()
            other, restored = create()
            other.load_state_dict(copy.deepcopy(opt.state_dict()))
            restored.load_state_dict(copy.deepcopy(scheduler.state_dict()))
            for _ in range(13):
                opt.step(); scheduler.step(); other.step(); restored.step()
                self.assertEqual(scheduler.get_last_lr(), restored.get_last_lr())

    def test_initialization_and_data_prefix_do_not_depend_on_horizon_profile(self):
        teacher, _ = build_model('P0', 8, (1, 1, 1), 94001, role='T')
        same, _ = build_model('P0', 8, (1, 1, 1), 94001, role='T')
        self.assertEqual(state_hash(teacher.state_dict()), state_hash(same.state_dict()))
        a, _ = build_model('PLH', 8, (1, 1, 1), 95001, role='S', teacher_aligner_state=teacher.aligner.state_dict())
        b, _ = build_model('PLH', 8, (1, 1, 1), 95001, role='S', teacher_aligner_state=teacher.aligner.state_dict())
        self.assertEqual(state_hash(a.state_dict()), state_hash(b.state_dict()))
        self.assertFalse(set(map(id, a.parameters())) & set(map(id, teacher.parameters())))
        streams = [BatchStream(101, 48, 95001) for _ in range(2)]
        for _ in range(4):
            self.assertEqual(streams[0].remaining_batches(), streams[1].remaining_batches())
            for stream in streams:
                stream.new_epoch()

    def test_base_e1_losses_are_bitwise_existing_operations(self):
        gen = torch.Generator().manual_seed(70)
        y, target, teacher = [torch.randn(2, 4, 8, 8, generator=gen) for _ in range(3)]
        for profile in ('BASE', 'E1'):
            old = existing_loss({'y': y}, {'y': teacher}, target, .02, [.3, .7], profile)
            new = student_losses({'y': y}, {'y': teacher}, target, .02, [.3, .7], profile)
            for key in old:
                self.assertTrue(torch.equal(old[key], new[key]), (profile, key))

    def test_h010_formula_and_both_hard_routes(self):
        y = torch.full((2, 4, 8, 8), .8, requires_grad=True)
        gt, teacher = torch.zeros_like(y), torch.full_like(y, .2, requires_grad=True)
        tau = torch.tensor(.1, requires_grad=True)
        weights = torch.tensor([.2, .7], requires_grad=True)
        result = student_losses({'y': y}, {'y': teacher}, gt, tau, weights, 'H010')
        expected_hard = ((1 + .1 * (.2 / (.2 + .1))) * .8)
        self.assertAlmostEqual(float(result['hard']), expected_hard, places=6)
        self.assertAlmostEqual(float(result['L_A']), expected_hard * .45, places=6)
        result['L_U'].backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(weights.grad)
        self.assertIsNone(tau.grad)
        self.assertIsNotNone(y.grad)

    def test_routed_gradients_match_separate_objectives_not_sum(self):
        class Pair(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(1, 1, bias=False)
                self.aligner = torch.nn.Linear(1, 1, bias=False)
        pair = Pair()
        u, a = pair.backbone.weight, pair.aligner.weight
        losses = dict(L_U=(u + 2 * a).square().sum(), L_A=(3 * u - a).square().sum())
        gu = torch.autograd.grad(losses['L_U'], u, retain_graph=True)[0]
        ga = torch.autograd.grad(losses['L_A'], a, retain_graph=True)[0]
        wrong_a = torch.autograd.grad(losses['L_U'] + losses['L_A'], a, retain_graph=True)[0]
        routed_student_backward(pair, losses)
        self.assertTrue(torch.equal(u.grad, gu))
        self.assertTrue(torch.equal(a.grad, ga))
        self.assertFalse(torch.equal(a.grad, wrong_a))

    def test_clone_p0_forward_and_rng_preservation(self):
        teacher, _ = build_model('P0', 8, (1, 1, 1), 94001, role='T')
        student, _ = build_model('PLH', 8, (1, 1, 1), 95001, role='S', teacher_aligner_state=teacher.aligner.state_dict())
        class Dataset:
            def base(self, index):
                return (torch.zeros(4, 32, 32), torch.zeros(4, 32, 32), torch.ones(4, 8, 8),
                        torch.ones(1, 8, 8), torch.ones(1, 32, 32), torch.tensor([0, 0]))
        rng = torch.get_rng_state().clone()
        self.assertTrue(_clone_check(student, teacher, Dataset(), 'cpu')['passed'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_diagnostic_ids_common_subset_and_rng_isolation(self):
        rng = np.random.get_state()
        first, second = diagnostic_ids(19809), diagnostic_ids(19809)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first['train128'])), 128)
        self.assertEqual(first['gradient24'], first['train128'][:24])
        self.assertEqual(len(first['gradient24']), 24)
        self.assertTrue(np.array_equal(rng[1], np.random.get_state()[1]))
        from qg40.calibration import select_calibration_indices
        expected = sorted(set(first['train128']).intersection(select_calibration_indices(19809).tolist()))
        self.assertEqual(first['train128_calibration_overlap'], expected)


class TrainingPipelineResumeTests(unittest.TestCase):
    """Real C4 model/loss/routed backward/AdamW; only data, authority and eval I/O are synthetic."""

    def test_teacher_and_h010_pause_resume_equal_uninterrupted(self):
        torch.set_num_threads(2)
        for identifier in ('L100I1-T03', 'L100I1-S08'):
            with self.subTest(case=identifier), tempfile.TemporaryDirectory() as directory:
                baseline = self.exercise(Path(directory) / 'baseline', identifier, (4,))
                resumed = self.exercise(Path(directory) / 'resumed', identifier, (2, 4))
                for field in ('model_state', 'optimizer', 'scheduler', 'sampler', 'exposure_counts', 'corruption_rng'):
                    self.assert_tree_equal(baseline[field], resumed[field], field)
                self.assertEqual(baseline['update'], 4)
                self.assertEqual(int(baseline['exposure_counts'].sum()), 8)

    def assert_tree_equal(self, one, two, location):
        if isinstance(one, torch.Tensor):
            self.assertTrue(torch.equal(one, two), location)
        elif isinstance(one, dict):
            self.assertEqual(set(one), set(two), location)
            for key in one:
                self.assert_tree_equal(one[key], two[key], location + '.' + str(key))
        elif isinstance(one, (tuple, list)):
            self.assertEqual(len(one), len(two), location)
            for index, (left, right) in enumerate(zip(one, two)):
                self.assert_tree_equal(left, right, location + '.' + str(index))
        else:
            self.assertEqual(one, two, location)

    def exercise(self, root, identifier, pause_targets):
        root.mkdir(parents=True)
        case = case_for(identifier)
        cfg = build_config(case)
        cfg.update(batch_size=2, num_worker=0)
        cfg['l100'].update(dataset_manifest='data.json', runtime_policy_sha256='1' * 64,
            teacher_checkpoint='teacher.safetensors', teacher_sha256='2' * 64,
            tau_R=.1, q_ref=.2, q_cache='q.npz', q_cache_sha256='3' * 64,
            reference_manifest='reference.json')
        config_path = root / 'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg))
        atomic_json(root / 'data.json', dict(schema='G20_DATA_v1', sensor='GF2', num_bands=4,
            max_pixel=1023, splits={'train': {'count': 3072}}))
        wd = root / cfg['work_dir']

        class TinyDataset:
            def __init__(self, split):
                self.split, self.base_count = split, 3072 if split == 'train' else 1
                self.has_gt, self.augment = True, split == 'train'
                self.spec = SimpleNamespace(inverse_scale=511.5)
            def __len__(self):
                return self.base_count
            def base(self, index):
                generator = torch.Generator().manual_seed(index + 77)
                gt = torch.rand(4, 32, 32, generator=generator) * .2
                ms = torch.nn.functional.avg_pool2d(gt, 4)
                pan = torch.rand(1, 32, 32, generator=generator) * .2
                lp = torch.nn.functional.avg_pool2d(pan, 4)
                return gt, gt.clone(), ms, lp, pan, torch.tensor([index, 0])
            def __getitem__(self, key):
                index, rotation = key if isinstance(key, tuple) else (key, 0)
                values = self.base(index)
                return (*[torch.rot90(x, rotation, (-2, -1)) for x in values[:-1]], torch.tensor([index, rotation]))

        teacher, _ = build_model('P0', 8, (1, 1, 1), case.teacher_seed, role='T', num_bands=4)
        teacher.requires_grad_(False).eval()
        reference = dict(teacher_checkpoint='teacher.safetensors', teacher_checkpoint_sha256='2' * 64,
            tau_R=.1, q_ref=.2, q_cache_path='q.npz', q_cache_sha256='3' * 64,
            teacher_run_id=case.teacher_run_id, teacher_update=case.teacher_updates)
        def tiny_model(layout, width, depth, seed, **kwargs):
            return build_model(layout, 8, (1, 1, 1), seed, **kwargs)
        def fake_evaluation(*args, **kwargs):
            return dict(fr={'hqnr': .9}, rr={'scc': .8, 'ergas': 2.}, val_ergas=2.)
        target = [0]
        def continue_training(_deadline):
            path = wd / 'last/identity.json'
            from l100.common import read
            return read(path).get('update', 0) < target[0]
        end = '2099-01-01T00:00:00+00:00'
        with ExitStack() as stack:
            for name, replacement in (
                ('l100.training.validate_config', lambda _cfg: case),
                ('l100.training.apply_runtime_policy', lambda _root: {'sha256': '1' * 64}),
                ('l100.training.source_identity', lambda _root: {'fixture': 'source'}),
                ('l100.training.runtime_context', lambda *a, **k: ({'deadline_utc': end}, end)),
                ('l100.controller.authorize_train', lambda *a, **k: None),
                ('l100.training.build_dataset', lambda _data, split, **kwargs: TinyDataset(split)),
                ('l100.training.build_model', tiny_model),
                ('l100.training.grid_steps', lambda _total: (1, 2, 3, 4)),
                ('l100.training.fullstate_steps', lambda *_: (0, 2, 4)),
                ('l100.training.diagnostic_steps', lambda *_: (0, 2, 4)),
                ('l100.training.before_deadline', continue_training),
                ('l100.training.FRMetrics', lambda _dataset: object()),
                ('l100.training.evaluate_model', fake_evaluation),
                ('l100.references.load_reference', lambda *a, **k: (teacher, reference, np.full((3072, 4), .2, np.float32))),
            ):
                stack.enter_context(patch(name, replacement))
            for index, target[0] in enumerate(pause_targets):
                self.assertEqual(train_run(config_path, device='cpu', root=root, resume=index > 0), 75)
        state = torch.load(wd / 'last/training_state.pt', map_location='cpu', weights_only=False)
        self.assertEqual(state['update'], 4)
        if case.role == 'T':
            from l100.common import read
            self.assertEqual(read(wd / 'official/raw_grid.json')['records'], [])
            self.assertEqual(len(read(wd / 'official/validation_grid.json')['records']), 4)
        else:
            self.assertTrue((wd / 'best_hqnr_meta.json').is_file())
        return state


class FinalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = case_for('L100I1-T03')
        self.cfg = build_config(self.case)
        self.cfg['l100'].update(dataset_manifest='data.json', runtime_policy_sha256='1' * 64)
        self.wd = self.root / self.cfg['work_dir']
        (self.wd / 'meta').mkdir(parents=True)
        (self.wd / 'meta/config.resolved.yaml').write_text(yaml.safe_dump(self.cfg))
        data = dict(sensor='GF2', num_bands=4, max_pixel=1023, splits={'train': {'count': 48}})
        atomic_json(self.root / 'data.json', data)
        self.context = dict(config_sha256=object_sha(self.cfg), data_sha256=object_sha(data),
                            source_identity={'fixture': 'source'}, reference_sha256=None)
        stream = BatchStream(48, 48, self.case.seed)
        atomic_json(self.wd / 'meta/training_start_manifest.json', dict(campaign_id=CAMPAIGN_ID,
            run_id=self.case.run_id, horizon_updates=self.case.updates, **self.context,
            sampler_hash=state_hash({'order': stream.order, 'rotations': stream.rotations})))
        stream.epoch, stream.cursor = self.case.updates - 1, 1
        self.folder = self.wd / 'candidates' / str(self.case.updates)
        self.folder.mkdir(parents=True)
        tensors = {'weight': torch.ones(1)}
        save_file(tensors, str(self.folder / 'model.safetensors'))
        counts = torch.zeros(48, 4, dtype=torch.int64); counts[0, 0] = self.case.updates * 48
        self.state = dict(full_state=True, update=self.case.updates, precision='fp32',
            scheduler={'last_epoch': self.case.updates}, model_state=tensors,
            model_sha256=sha256(self.folder / 'model.safetensors'), exposure_counts=counts,
            sampler=stream.state_dict(), training_seconds=123., evaluation_seconds=1., io_seconds=2., **self.context)
        self.identity = dict(update=self.case.updates, full_state=True, role='T', sensor='GF2', num_bands=4,
            input_layout='P0', state_hash=state_hash(tensors), model_sha256=self.state['model_sha256'],
            saved_at_utc='2026-09-21T14:00:00+00:00', **self.context)
        self.publish()
        self.policy = patch('l100.training.apply_runtime_policy').start()
        self.addCleanup(patch.stopall)
        patch('l100.training.source_identity', return_value=self.context['source_identity']).start()

    def publish(self):
        atomic_torch(self.folder / 'training_state.pt', self.state)
        self.identity['training_state_sha256'] = sha256(self.folder / 'training_state.pt')
        atomic_json(self.folder / 'identity.json', self.identity)

    def test_exact100k_recovered_without_optimizer_and_original_endtime_preserved(self):
        result = recover_final_status(self.wd, self.cfg, self.root)
        self.assertTrue(result['training_complete'])
        self.assertEqual(result['actual_updates'], 100000)
        self.assertEqual(result['training_completed_at_utc'], '2026-09-21T14:00:00+00:00')
        self.assertEqual(len(result['pending_steps']), 50)
        self.assertFalse(recover_final_status(self.wd, self.cfg, self.root))

    def test_scheduler_or_model_tamper_rejected(self):
        self.state['scheduler']['last_epoch'] = 50000
        self.publish()
        with self.assertRaisesRegex(ValueError, 'scheduler'):
            recover_final_status(self.wd, self.cfg, self.root)

    def test_mid50_cannot_stand_in_for_fresh100_endpoint(self):
        self.identity['update'] = 50000
        self.publish()
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            recover_final_status(self.wd, self.cfg, self.root)


if __name__ == '__main__':
    unittest.main()
