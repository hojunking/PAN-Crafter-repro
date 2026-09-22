"""CPU regressions for fork/route/schedule and exact in-run continuation."""
import copy
import datetime as dt
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fh12.training import BatchStream, restore_rng
from g20.losses import student_losses as old_losses
from gfp40.diagnostics import (border_summary, gradient_probe, low_active_mass,
    parameter_snapshot, parameter_update_norm, preserve_runtime, rng_state)
from gfp40.losses import coefficients, routed_student_backward, student_losses
from gfp40.plan import build_config, case_for, grid_steps
from gfp40.stream import PairedBatchStream, gamma_ids
from gfp40.training import (cosine_factor, initialize_student, make_optimizer,
    make_scheduler, model_hashes, select_validation, val_diverged,
    verify_batch_metadata, verify_paired_initialization, pause_reason, runtime_context)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Conv2d(4, 4, 1)
        self.aligner = torch.nn.Conv2d(1, 4, 1)
    def forward(self, x, pan):
        return {'y': self.backbone(x) + self.aligner(pan)}


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.gt = torch.rand(2, 4, 12, 12)
        self.y = torch.rand_like(self.gt, requires_grad=True)
        self.target = torch.rand_like(self.gt, requires_grad=True)
        self.q = torch.tensor([.3, .9], requires_grad=True)

    def test_base_exact_existing(self):
        new = student_losses({'y': self.y}, {'y': self.target}, self.gt, .04, self.q)
        old = old_losses({'y': self.y}, {'y': self.target}, self.gt, .04, self.q)
        for key in old:
            self.assertTrue(torch.equal(old[key], new[key]), key)

    def test_coefficients(self):
        for profile in ('NATIVE0', 'CTRL', 'MIX'):
            for step in (0, 500, 1000, 60000, 100000):
                self.assertEqual(coefficients(profile, step),
                    dict(alpha=1., beta=.1, lambda_edge=.002, ramp_fraction=1.))
        for invalid in ('H010', 'K1', 'E100', 'B_SELECTED', 'MSTAR'):
            with self.assertRaises(ValueError): coefficients(invalid, 10)

    def test_detached_weights(self):
        loss = student_losses({'y': self.y}, {'y': self.target}, self.gt, .04, self.q, 'MIX', 1000)
        loss['L_U'].backward()
        self.assertIsNone(self.q.grad)
        self.assertIsNone(self.target.grad)
        self.assertIsNotNone(self.y.grad)
        for key in ('difficulty', 'advantage', 'soft_gate', 'q_weights'):
            self.assertFalse(loss[key].requires_grad)

    def test_actual_separate_routing(self):
        model = Tiny()
        for profile in ('NATIVE0', 'CTRL', 'MIX'):
            out = model(self.gt, self.gt[:, :1])
            loss = student_losses(out, {'y': self.target}, self.gt, .04, self.q, profile, 1000)
            u, a = tuple(model.backbone.parameters()), tuple(model.aligner.parameters())
            gu = torch.autograd.grad(loss['L_U'], u, retain_graph=True)
            ga = torch.autograd.grad(loss['L_A'], a, retain_graph=True)
            routed_student_backward(model, loss)
            for p, want in zip(u + a, gu + ga):
                self.assertTrue(torch.equal(p.grad, want))
            actual_a = [p.grad.clone() for p in a]
            if profile == 'NATIVE0': base_a = actual_a
            if profile in ('CTRL', 'MIX'):
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(base_a, actual_a)))

    def test_gradient_and_border_diagnostics(self):
        model = Tiny()
        loss = student_losses(model(self.gt, self.gt[:, :1]), {'y': self.target}, self.gt, .04, self.q, 'MIX', 1000)
        diag = gradient_probe(model, loss)
        self.assertTrue(diag['weighted_terms_exact'])
        self.assertGreater(diag['norms']['hard'], 0)
        self.assertAlmostEqual(diag['norms']['soft'] / diag['norms']['raw_soft'], .1, places=6)
        border = border_summary(loss['e_student'], loss['hard_map'])
        self.assertTrue(border['4']['loss_support_unchanged'])


class StreamTests(unittest.TestCase):
    def test_explicit_existing_rng_mapping(self):
        old, new = BatchStream(131, 12, 96301), PairedBatchStream(131, 12, 96301)
        self.assertTrue(torch.equal(old.order, new.order))
        self.assertTrue(torch.equal(old.rotations, new.rotations))
        self.assertEqual(new.worker_seed, 596301)
        first = new.remaining_batches()[0]
        self.assertEqual([r[2] for r in first], gamma_ids(696301, 0, 12).tolist())
        self.assertEqual(new.gamma_counter, 0)  # prefetch cannot consume draws
        new.advance()
        self.assertEqual(new.gamma_counter, 12)

    def test_prefix_and_exact_resume(self):
        a, b = PairedBatchStream(512, 48, 96301), PairedBatchStream(512, 48, 96301)
        self.assertEqual(a.initial_prefix(), b.initial_prefix())
        for _ in range(3): a.advance()
        b.load_state_dict(a.state_dict())
        self.assertEqual(a.remaining_batches(), b.remaining_batches())
        damaged = a.state_dict(); damaged['gamma_counter'] += 1
        with self.assertRaises(ValueError): b.load_state_dict(damaged)

    def test_gamma_population_rng_isolation(self):
        state = torch.get_rng_state().clone()
        draws = gamma_ids(66501, 0, 100000)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        for i, p in enumerate((.25, .5, .25)):
            self.assertAlmostEqual(float((draws == i).float().mean()), p, delta=.01)
        self.assertTrue(torch.equal(gamma_ids(66501, 310, 100), draws[310:410]))

    def test_actual_native_and_mixed_metadata(self):
        stream = PairedBatchStream(100, 4, 96501)
        row = torch.tensor([[source, view, 1, 1, 1, gamma, uniform] for source, view, gamma, uniform in stream.batch_at(0)])
        self.assertEqual(verify_batch_metadata(stream, row, 'NATIVE0'),
                         [[r[0], r[1], r[3]] for r in stream.batch_at(0)])
        row[:, 4] = row[:, 5]
        verify_batch_metadata(stream, row, 'MIX')
        row[0, 1] = (row[0, 1] + 1) % 4
        with self.assertRaises(ValueError): verify_batch_metadata(stream, row, 'MIX')

    def test_high_family_uses_native_index_zero(self):
        stream = PairedBatchStream(100, 4, 96501, family='HIGH')
        row = torch.tensor([[source, view, 1, 1, 0, gamma, uniform]
                            for source, view, gamma, uniform in stream.batch_at(0)])
        verify_batch_metadata(stream, row, 'CTRL')
        row[:,4] = 1
        with self.assertRaises(ValueError): verify_batch_metadata(stream, row, 'CTRL')

    def test_actual256_consume_and_json_resume(self):
        from gfp40.common import immutable_json, read_json
        stream = PairedBatchStream(1024, 48, 96501)
        expected, consumed = stream.initial_prefix(), []
        for _ in range(6):
            meta = torch.tensor([[s, v, 1, 1, 1, g, u] for s, v, g, u in stream.batch_at(stream.cursor)])
            actual = verify_batch_metadata(stream, meta, 'NATIVE0')
            consumed.extend(actual[:max(0, 256 - len(consumed))]); stream.advance()
        self.assertEqual(consumed, expected['sample_view_uniform'])
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'actual.json'
            immutable_json(file, dict(triples=consumed))
            self.assertEqual(read_json(file)['triples'], expected['sample_view_uniform'])
        resumed = PairedBatchStream(1024, 48, 96501)
        resumed.load_state_dict(stream.state_dict())
        self.assertEqual(resumed.gamma_counter, 288)
        self.assertEqual(resumed.initial_prefix()['sample_view_uniform'], consumed)


class TrainingTests(unittest.TestCase):
    def test_stop_now_control_only(self):
        from gfp40.common import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'control.json'
            with patch('gfp40.training.before_deadline', return_value=True):
                self.assertIsNone(pause_reason(False, 'unused', file))
                atomic_json(file, dict(command='STOP_AFTER_BLOCK'))
                self.assertIsNone(pause_reason(False, 'unused', file))
                atomic_json(file, dict(command='STOP_NOW_SAFE'))
                self.assertEqual(pause_reason(False, 'unused', file), 'PAUSED_CONTROL')
                atomic_json(file, dict(command='CONTINUE'))
                self.assertEqual(pause_reason(True, 'unused', file), 'PAUSED_SIGNAL')
            with patch('gfp40.training.before_deadline', return_value=False):
                self.assertEqual(pause_reason(False, 'unused', file), 'PARTIAL_TIME_LIMIT')

    def test_actual_pair_prefix_json_roundtrip(self):
        from gfp40.common import immutable_json
        prefix = PairedBatchStream(512, 48, 96301).initial_prefix()
        hashes = dict(full='a', U='b', A='c')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'work_dir' / case_for('A01').run_id
            immutable_json(folder / 'init_manifest.json', dict(hashes=hashes))
            immutable_json(folder / 'diagnostics/paired_stream_first256.json', prefix)
            immutable_json(folder / 'diagnostics/paired_stream_first256.json', prefix)
            self.assertEqual(verify_paired_initialization(case_for('A02'), hashes, prefix, tmp), ['A01'])
            with self.assertRaises(ValueError):
                verify_paired_initialization(case_for('A02'), dict(hashes, A='wrong'), prefix, tmp)

    def test_pairing_ignores_family_cdf_not_uniform(self):
        from gfp40.common import immutable_json
        reference = PairedBatchStream(512,48,98201,family='G025').initial_prefix()
        alternative = PairedBatchStream(512,48,98201,family='P025').initial_prefix()
        self.assertNotEqual(reference['draws'], alternative['draws'])
        hashes = dict(full='a',U='b',A='c')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'work_dir'/case_for('B04').run_id
            immutable_json(folder/'init_manifest.json',dict(hashes=hashes))
            immutable_json(folder/'diagnostics/paired_stream_first256.json',reference)
            self.assertEqual(verify_paired_initialization(case_for('B16'),hashes,alternative,tmp),['B04'])
            changed = dict(alternative,sample_view_uniform_sha256='wrong')
            with self.assertRaises(ValueError):
                verify_paired_initialization(case_for('B16'),hashes,changed,tmp)

    def test_fork_preserves_parent_a_not_teacher(self):
        torch.manual_seed(0)
        parent, teacher = Tiny(), Tiny()
        initial = model_hashes(parent)
        with patch('gfp40.training.build_model', side_effect=AssertionError('No fresh construction allowed')):
            student, init = initialize_student(SimpleNamespace(is_ft=True), teacher, parent)
        self.assertIs(student, parent)
        self.assertEqual(model_hashes(student), initial)
        self.assertNotEqual(model_hashes(student)['A'], model_hashes(teacher)['A'])
        self.assertFalse(init['teacher_aligner_recloned'])
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        optimizer = make_optimizer(student, build_config('A01'))
        scheduler = make_scheduler(optimizer)
        self.assertFalse(optimizer.state)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [0., 0.])

    def test_fresh_pair_same_u_and_independent_a(self):
        from g20.model import build_model
        teacher, _ = build_model('P0', 4, (1, 1, 1), 123, role='T')
        case = SimpleNamespace(is_ft=False, model_seed=96611, input_layout='PLH', width=4, depth=(1, 1, 1))
        first, _ = initialize_student(case, teacher)
        second, _ = initialize_student(case, teacher)
        self.assertEqual(model_hashes(first), model_hashes(second))
        self.assertEqual(model_hashes(first)['A'], model_hashes(teacher)['A'])
        self.assertFalse(set(map(id, first.parameters())) & set(map(id, teacher.parameters())))

    def test_scheduler_horizons_and_zero_update_counted(self):
        for total in (20000, 60000, 100000):
            self.assertEqual(cosine_factor(0, total=total), 0.)
            self.assertEqual(cosine_factor(100, total=total), 1.)
            self.assertEqual(cosine_factor(total, total=total), 0.)
        self.assertGreater(cosine_factor(20000, total=60000),0.)
        self.assertEqual(cosine_factor(20000, total=20000),0.)
        model = Tiny(); optimizer = make_optimizer(model, build_config('A01'))
        scheduler = make_scheduler(optimizer)
        before = model_hashes(model)
        for p in model.parameters(): p.grad = torch.ones_like(p)
        optimizer.step(); scheduler.step()
        self.assertEqual(model_hashes(model), before)
        self.assertTrue(all(state['step'].item() == 1 for state in optimizer.state.values()))
        for _ in range(19999): optimizer.step(); scheduler.step()
        self.assertEqual(scheduler.last_epoch, 20000)
        self.assertTrue(all(state['step'].item() == 20000 for state in optimizer.state.values()))
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [0., 0.])

    def test_separate_40h_window_has_37h_optimizer_cutoff(self):
        from gfp40.common import CAMPAIGN_ID, atomic_json
        from gfp40.policy import CampaignWindow
        start=dt.datetime.now(dt.timezone.utc)-dt.timedelta(hours=2)
        end=start+dt.timedelta(hours=40)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'window.json'
            value=CampaignWindow(start,'s3').to_dict()
            atomic_json(path,value)
            cfg={'gfp40':{'window_path':str(path),'server_id':'s3'}}
            _window,opt,final=runtime_context(cfg,tmp,end.isoformat())
            self.assertEqual(dt.datetime.fromisoformat(opt),start+dt.timedelta(hours=37))
            self.assertEqual(final,end.isoformat())
            with self.assertRaises(ValueError): runtime_context(cfg,tmp,(end+dt.timedelta(hours=1)).isoformat())
            cfg['gfp40']['server_id']='s4'
            with self.assertRaises(ValueError): runtime_context(cfg,tmp)
            cfg['gfp40']['server_id']='s3'
            atomic_json(path,dict(value,deadline_utc=(start+dt.timedelta(hours=20)).isoformat()))
            with self.assertRaises(ValueError): runtime_context(cfg,tmp)

    def test_exact_resume_optimizer_sampler_rng(self):
        torch.manual_seed(117); random.seed(71); np.random.seed(3)
        model = Tiny(); cfg = build_config('B02')
        optimizer = make_optimizer(model, cfg); scheduler = make_scheduler(optimizer)
        stream = PairedBatchStream(13, 4, 96401)
        fixed_gt = torch.rand(13, 4, 12, 12)
        target = fixed_gt * .8
        evaluations = []
        def one():
            if stream.cursor == stream.count: stream.new_epoch()
            batch = stream.remaining_batches()[0]
            ids = torch.tensor([r[0] for r in batch])
            gt = fixed_gt[ids]
            # Global RNG is a tested part of resume independently of sampler.
            gt = gt + torch.rand(()) * 1e-4 + random.random() * 1e-4 + np.random.rand() * 1e-4
            optimizer.zero_grad(set_to_none=True)
            loss = student_losses(model(gt, gt[:, :1]), {'y': target[ids]}, gt, .02,
                                  torch.full((4,), .4), 'MIX', stream.completed_updates)
            routed_student_backward(model, loss)
            optimizer.step(); scheduler.step(); stream.advance()
            evaluations.append((stream.completed_updates, model_hashes(model)['full']))
        for _ in range(5): one()
        checkpoint = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), stream=stream.state_dict(), rng=rng_state()))
        for _ in range(7): one()
        expected_hash, expected_eval = model_hashes(model), evaluations[5:]
        expected_opt = copy.deepcopy(optimizer.state_dict())
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); stream.load_state_dict(checkpoint['stream'])
        restore_rng(checkpoint['rng']); evaluations.clear()
        for _ in range(7): one()
        self.assertEqual(model_hashes(model), expected_hash)
        self.assertEqual(evaluations, expected_eval)
        for key, val in expected_opt['state'].items():
            for name, want in val.items(): self.assertTrue(torch.equal(optimizer.state_dict()['state'][key][name], want))

    def test_runtime_guard_and_actual_update_norm(self):
        model = Tiny(); model.train(); model.aligner.eval()
        before = rng_state()
        with preserve_runtime(model):
            model.eval(); torch.rand(3); random.random(); np.random.rand()
        self.assertTrue(model.training)
        self.assertFalse(model.aligner.training)
        self.assertTrue(torch.equal(before['torch'], torch.get_rng_state()))
        previous = parameter_snapshot(model)
        with torch.no_grad(): next(model.aligner.parameters()).add_(.1)
        measured = parameter_update_norm(model, previous)
        self.assertEqual(measured['U'], 0.)
        self.assertGreater(measured['A'], 0.)

    def test_val_selection_and_symmetric_divergence(self):
        records = [dict(update=2000, val_ergas=.55), dict(update=1000, val_ergas=.55)]
        self.assertEqual(select_validation(records)['update'], 1000)
        self.assertTrue(val_diverged(.5, records))
        self.assertFalse(val_diverged(None, records))
        self.assertFalse(val_diverged(.5, records[:1]))
        self.assertEqual(len(grid_steps(20000)), 20)
        self.assertEqual(len(grid_steps(100000)), 50)
        self.assertEqual(len(grid_steps(60000)), 20)
        self.assertNotIn(20000, grid_steps(60000))
        self.assertNotIn(40000, grid_steps(60000))
        self.assertNotIn(0, grid_steps(20000))
        self.assertNotIn(50000, grid_steps(100000))
        low = [dict(local_step=t, gradient={'soft_over_hard': 1e-4}, soft_active_fraction=1e-5)
               for t in (0, 5000, 20000)]
        self.assertEqual(low_active_mass(low)['status'], 'LOW_ACTIVE_MASS')


if __name__ == '__main__':
    unittest.main()
