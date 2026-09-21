import csv
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from ablr2.model import build_model, state_hash
from ablr2.losses import student_losses, teacher_loss, routed_student_backward
from qg40.model import build_model as original_model
from qg40.losses import student_losses as original_losses


def component(key):
    path = Path(__file__).parents[1] / 'research_log/PANDA_ABL_S1WV3_S2QB_AdaptiveRepeat_Bundle_2026-09-21_v2/ABLR2_ComponentCatalog_2026-09-21.csv'
    with path.open(encoding='utf-8-sig') as stream:
        row = next(x for x in csv.DictReader(stream) if x['case_id'] == key)
    for k in ('alpha', 'beta', 'lambda_edge'):
        row[k] = float(row[k])
    for k in ('mask_L', 'mask_H'):
        row[k] = int(row[k])
    for k in ('teacher_predictions_used', 'teacher_q_used', 'teacher_A_clone_used', 'soft_trust', 'soft_advantage'):
        row[k] = row[k] == 'True'
    return row


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def inputs(self, bands):
        return torch.randn(2, 1, 32, 32), torch.randn(2, bands, 8, 8), torch.randn(2, 1, 8, 8)

    def test_teacher_forward_bitwise_legacy_c4_c8(self):
        for bands in (4, 8):
            actual, _ = build_model(bands=bands, seed=781001, role='T', width=8, depth=(1, 1, 1))
            old, _ = original_model('P0', 8, (1, 1, 1), 781001, role='T', bands=bands)
            self.assertEqual(state_hash(actual.state_dict()), state_hash(old.state_dict()))
            inputs = self.inputs(bands)
            self.assertTrue(torch.equal(actual(*inputs)['y'], old(*inputs)['y']))

    def test_full_bitwise_legacy_c4_c8(self):
        for bands in (4, 8):
            teacher, _ = build_model(bands=bands, seed=9, role='T', width=8, depth=(1, 1, 1))
            actual, _ = build_model(bands=bands, seed=19, role='S', component=component('C07'),
                                   teacher_aligner_state=teacher.aligner.state_dict(), width=8, depth=(1, 1, 1))
            old, _ = original_model('PLH', 8, (1, 1, 1), 19, role='S', bands=bands,
                                    teacher_aligner_state=teacher.aligner.state_dict())
            self.assertEqual(state_hash(actual.state_dict()), state_hash(old.state_dict()))
            inputs = self.inputs(bands)
            out, truth = actual(*inputs), torch.randn(2, bands, 32, 32)
            teacher_out = teacher(*inputs)
            self.assertTrue(torch.equal(out['y'], old(*inputs)['y']))
            new_loss = student_losses(out, teacher_out, truth, component('C07'), .17,
                                      torch.tensor([.2,.8]), bands=bands)
            old_loss = original_losses(out, teacher_out, truth, .17, torch.tensor([.2,.8]), bands=bands)
            for key in ('L_U', 'L_A', 'hard_i', 'soft_i', 'edge_i', 'difficulty', 'advantage'):
                self.assertTrue(torch.equal(new_loss[key], old_loss[key]), key)

    def test_student_u_common_all17(self):
        teacher, _ = build_model(bands=4, seed=9, role='T', width=8, depth=(1, 1, 1))
        digests = []
        for i in range(17):
            c = component(f'C{i:02}')
            model, init = build_model(bands=4, seed=19, role='S', component=c,
                teacher_aligner_state=teacher.aligner.state_dict() if c['teacher_A_clone_used'] else None,
                width=8, depth=(1, 1, 1))
            self.assertEqual(model.backbone.input.in_channels, 7)
            self.assertTrue(init['extra_kernel_zero'])
            digests.append(state_hash(model.backbone.state_dict()))
        self.assertEqual(len(set(digests)), 1)

    def test_identity_no_warp_and_mask_in_both_modes(self):
        model, _ = build_model(bands=8, seed=19, role='S', component=component('C00'), width=8, depth=(1,1,1))
        inputs = self.inputs(8)
        with patch('ablr2.model.warp_pan', side_effect=AssertionError('identity must bypass warp')):
            for mode in (True, False):
                model.train(mode)
                out = model(*inputs)
                self.assertTrue(torch.equal(out['pan_aligned'], inputs[0]))
                self.assertEqual(int(torch.count_nonzero(out['x_in'][:,1:3])), 0)
                self.assertEqual(int(torch.count_nonzero(out['delta'])), 0)
        self.assertEqual(list(model.aligner.parameters()), [])

    def test_frozen_a_does_not_step_or_train(self):
        teacher, _ = build_model(bands=4, seed=9, role='T', width=8, depth=(1,1,1))
        c = component('C16')
        student, _ = build_model(bands=4, seed=19, role='S', component=c,
                                teacher_aligner_state=teacher.aligner.state_dict(), width=8, depth=(1,1,1))
        student.train()
        self.assertFalse(student.aligner.training)
        self.assertTrue(all(not p.requires_grad for p in student.aligner.parameters()))
        before = state_hash(student.aligner.state_dict())
        inputs = self.inputs(4)
        loss = student_losses(student(*inputs), teacher(*inputs), torch.randn(2,4,32,32), c,
                              .1, torch.tensor([.4,.6]), bands=4)
        route = routed_student_backward(student, loss)
        opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=.01, weight_decay=.01)
        opt.step()
        self.assertEqual(route['a_parameters'], 0)
        self.assertEqual(state_hash(student.aligner.state_dict()), before)
        self.assertEqual(state_hash(teacher.aligner.state_dict()), before)

    def test_teacher_free_losses_refuse_hidden_teacher_or_calibration(self):
        y = torch.randn(2,4,8,8,requires_grad=True)
        for key in ('C00','C01','C02','C03','C10'):
            c = component(key)
            got = student_losses({'y':y}, None, torch.zeros_like(y), c, bands=4)
            self.assertEqual(float(got['soft']), 0.)
            with self.assertRaises(ValueError):
                student_losses({'y':y}, {'y':y}, torch.zeros_like(y), c, bands=4)
            with self.assertRaises(ValueError):
                student_losses({'y':y}, None, torch.zeros_like(y), c, tau_R=.1, bands=4)

    def test_uniform_c04_and_constant_controls(self):
        y = torch.randn(2,4,8,8,requires_grad=True); target = torch.zeros_like(y)
        teacher = torch.ones_like(y)*10
        uniform = student_losses({'y':y}, {'y':teacher}, target, component('C04'), bands=4)
        self.assertTrue(torch.allclose(uniform['soft'], .1*(y-teacher).abs().mean()))
        mean = student_losses({'y':y}, {'y':teacher}, target, component('C15'), .1, s_bar=.31, bands=4)
        self.assertTrue(torch.equal(mean['q_weights'], torch.full((2,),.31)))
        with self.assertRaises(ValueError):
            student_losses({'y':y}, {'y':teacher}, target, component('C15'), .1, bands=4)

    def test_tzero_skips_only_corruption_stream(self):
        model, _ = build_model(bands=4, seed=9, role='T', width=8, depth=(1,1,1))
        pan, ms, lp = self.inputs(4)
        out = model(pan,ms,lp); gt=torch.zeros_like(out['y'])
        gen = torch.Generator().manual_seed(123)
        before = gen.get_state().clone(); native = torch.get_rng_state().clone()
        loss = teacher_loss(model,out,gt,pan,ms,1,gen,consistency_weight=0.,bands=4)
        self.assertFalse(loss['offset_active'])
        self.assertTrue(torch.equal(before,gen.get_state()))
        teacher_loss(model,out,gt,pan,ms,1,gen,consistency_weight=1e-4,bands=4)
        self.assertFalse(torch.equal(before,gen.get_state()))
        self.assertTrue(torch.equal(native,torch.get_rng_state()))

    def test_gradient_routing_excludes_u_loss_from_a(self):
        model=torch.nn.Module();model.backbone=torch.nn.Linear(1,1,bias=False)
        model.aligner=torch.nn.Linear(1,1,bias=False)
        u=model.backbone.weight;a=model.aligner.weight
        routed_student_backward(model,{'L_U':6*u.sum()+100*a.sum(),'L_A':3*a.sum()+10*u.sum()})
        self.assertEqual(float(u.grad),6.)
        self.assertEqual(float(a.grad),3.)

    def test_teacher_and_gates_are_detached(self):
        y=torch.randn(2,4,8,8,requires_grad=True)
        teacher=torch.randn(2,4,8,8,requires_grad=True)
        q=torch.tensor([.3,.7],requires_grad=True)
        result=student_losses({'y':y},{'y':teacher},torch.zeros_like(y),component('C07'),.1,q,bands=4)
        self.assertFalse(result['difficulty'].requires_grad)
        self.assertFalse(result['advantage'].requires_grad)
        result['L_U'].backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(q.grad)


if __name__ == '__main__':
    unittest.main()
