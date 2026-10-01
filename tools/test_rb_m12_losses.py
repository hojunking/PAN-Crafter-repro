"""Algebra, detach and route isolation for all preregistered M12 cases."""
import unittest
import numpy as np
import torch
from fh12.losses import student_losses as baseline
from panda_rb_m12.losses import CASES, case_weights, student_losses, trust_weighted_mean
from panda_rb.weights import build_weights


class LossTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(107)
        self.y = torch.rand(3,8,12,10,generator=gen).requires_grad_(True)
        self.teacher = torch.rand(3,8,12,10,generator=gen).requires_grad_(True)
        self.gt = torch.rand(3,8,12,10,generator=gen).requires_grad_(True)
        self.w = torch.tensor([.2,.5,.8],requires_grad=True)

    def losses(self, case='QFULL', **kw):
        return student_losses({'y':self.y}, {'y':self.teacher}, self.gt, .1, self.w,
                              case_id=case, **kw)

    def test_qfull_exact_float32_reference(self):
        actual = self.losses()
        expected = baseline({'y':self.y},{'y':self.teacher},self.gt,.1,self.w)
        for key in expected:
            self.assertTrue(torch.equal(actual[key],expected[key]),key)

    def test_routes_are_independent(self):
        base = self.losses(w_align=self.w)
        edge = student_losses({'y':self.y},{'y':self.teacher},self.gt,.1,
                              self.w*.5,self.w,case_id='QEDGE')
        align = self.losses('QALIGN',w_align=self.w*.5)
        self.assertTrue(torch.equal(base['L_A'],edge['L_A']))
        self.assertTrue(torch.equal(base['L_U'],align['L_U']))
        self.assertFalse(torch.equal(base['L_U'],edge['L_U']))
        self.assertFalse(torch.equal(base['L_A'],align['L_A']))

    def test_all_gates_detached_and_live_residuals_exist(self):
        for case in CASES:
            loss = self.losses(case)
            for key in ('difficulty','advantage','applied_advantage','hard_weight','soft_weight','w_edge','w_align'):
                self.assertFalse(loss[key].requires_grad,(case,key))
            grads = torch.autograd.grad(loss['L_U']+loss['L_A'],
                    (self.y,self.teacher,self.gt,self.w),allow_unused=True)
            self.assertTrue(bool((grads[0] != 0).any()))
            self.assertEqual(grads[1:],(None,None,None))

    def test_hspmean_preserves_each_sample_mass(self):
        loss = self.losses('HSPMEAN')
        torch.testing.assert_close(loss['hard_mass_original'],loss['hard_mass_applied'])
        for h in loss['hard_weight']:
            self.assertEqual(float(h.max()-h.min()),0.)

    def test_advmean_preserves_trust_weighted_each_sample_mass(self):
        loss = self.losses('ADVMEAN')
        torch.testing.assert_close(loss['soft_mass_original'],loss['soft_mass_applied'])
        arithmetic = loss['advantage'].flatten(1).mean(1)
        self.assertFalse(torch.equal(arithmetic,loss['applied_advantage'][:,0,0,0]))

    def test_zero_trust_safe_and_no_positive_denominator_clamp(self):
        trust = torch.tensor([[[[0.,0.]]],[[[1e-20,2e-20]]]])
        advantage = torch.tensor([[[[.4,.6]]],[[[.2,.8]]]])
        mean = trust_weighted_mean(trust,advantage)
        self.assertEqual(mean[0].item(),0.)
        self.assertAlmostEqual(mean[1].item(),.6,places=6)
        self.assertTrue(bool(torch.isfinite(mean).all()))

    def test_constant_gate_matches_full_and_zero_advantage_soft_zero(self):
        # et constant -> h constant; es constant -> advantage constant too.
        y, t, gt = torch.ones_like(self.y), torch.ones_like(self.y)*.3, torch.zeros_like(self.y)
        full = student_losses({'y':y},{'y':t},gt,.1,self.w)
        for case in ('HSPMEAN','ADVMEAN'):
            loss = student_losses({'y':y},{'y':t},gt,.1,self.w,case_id=case)
            torch.testing.assert_close(loss['hard'],full['hard'])
            torch.testing.assert_close(loss['soft'],full['soft'])
        zero = student_losses({'y':gt},{'y':t},gt,.1,self.w,case_id='ADVMEAN')
        self.assertEqual(zero['soft'].item(),0.)

    def test_h0_changes_only_hard_and_noadv_keeps_trust_beta(self):
        full,h0,noadv = self.losses(),self.losses('H0'),self.losses('NOADV')
        self.assertTrue(torch.equal(full['soft'],h0['soft']))
        self.assertTrue(torch.equal(full['difficulty'],h0['difficulty']))
        self.assertTrue(torch.equal(full['advantage'],h0['advantage']))
        self.assertTrue(torch.equal(full['hard'],noadv['hard']))
        self.assertTrue(torch.equal(noadv['soft_weight'],.1*full['trust']))
        self.assertTrue(torch.equal(h0['hard_weight'],torch.ones_like(h0['hard_weight'])))

    def test_maps_reuse_frozen_population_and_separate_rng(self):
        q = np.arange(80,dtype=np.float32).reshape(20,4)/100
        e = np.flip(q,axis=0).copy()
        before = torch.get_rng_state().clone()
        arrays, report = build_weights(q,e,.2,261001101)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertAlmostEqual(float(arrays['QMEAN'][0,0]),float(arrays['QFULL'].mean(dtype=np.float64)),places=7)
        for case in CASES:
            edge,align = case_weights(arrays,case)
            self.assertEqual(edge.shape,(20,4)); self.assertEqual(align.shape,(20,4))
        self.assertIs(case_weights(arrays,'QEDGE')[0],arrays['QFULL'])
        self.assertIs(case_weights(arrays,'QEDGE')[1],arrays['QMEAN'])
        self.assertIs(case_weights(arrays,'QALIGN')[0],arrays['QMEAN'])
        for view in range(4):
            self.assertTrue(np.array_equal(np.sort(arrays['QFULL'][:,view]),np.sort(arrays['QESUR'][:,view])))
            for group in range(5):
                ids = np.flatnonzero(arrays['e_stratum'][:,view] == group)
                self.assertTrue(np.array_equal(np.sort(arrays['QFULL'][ids,view]),np.sort(arrays['QSHUF'][ids,view])))
                self.assertFalse(np.any(arrays['shuffle_source'][ids,view] == ids))

    def test_unknown_cases_and_invalid_weights_fail_closed(self):
        with self.assertRaises(ValueError): self.losses('NEW')
        with self.assertRaises(ValueError): self.losses(w_align=torch.zeros(3))


if __name__ == '__main__':
    unittest.main()
