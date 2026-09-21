import copy
from dataclasses import replace
from datetime import timedelta
import unittest

from l100 import plan as p, policy as q


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.window = q.CampaignWindow('2026-09-21T00:00:00+00:00')
        self.block = p.blocks_for('s4')[0]

    def at(self, hours):
        return self.window.t0_utc + timedelta(hours=hours)

    def admit(self, hours=.75, block=None, remaining=None, **kwargs):
        block = block or self.block
        return q.evaluate_admission(self.window, self.at(hours), block,
            q.estimate_block_hours(block) if remaining is None else remaining, **kwargs)

    def test_clock_strict_roundtrip(self):
        self.assertEqual(q.CampaignWindow.from_dict(self.window.to_dict()), self.window)
        for field in ('deadline_utc', 'campaign_id', 'admission_cutoff_utc', 'train_finish_utc'):
            changed = self.window.to_dict(); changed[field] = 'wrong'
            with self.assertRaises(ValueError):
                q.CampaignWindow.from_dict(changed)
        with self.assertRaises(ValueError):
            q.CampaignWindow('2026-09-21T00:00:00')

    def test_exact_initial_whole_core_budget(self):
        for server, expected in dict(s3=17.65, s4=13.05, s5=13.05).items():
            primary = p.SETUP_HOURS + sum(q.estimate_block_hours(b) for b in p.blocks_for(server))
            self.assertAlmostEqual(primary, expected)
        self.assertAlmostEqual(q.estimate_block_hours(p.blocks_for('s3')[0]), 10.8)
        self.assertAlmostEqual(q.estimate_block_hours(self.block), 7.5)

    def test_first_chain_admitted_without_other_servers_performance(self):
        result = self.admit()
        self.assertTrue(result['allowed'])
        self.assertFalse(result['performance_gate'])
        self.assertEqual(result['run_ids'], list(self.block.run_ids))
        self.assertEqual(result['window'], self.window.to_dict())

    def test_local_p0_failure_does_not_change_other_server(self):
        self.assertFalse(self.admit(p0_ready=False)['allowed'])
        self.assertTrue(self.admit(block=p.blocks_for('s5')[0])['allowed'])

    def test_new_cutoff_exact_16_forbidden(self):
        second = p.blocks_for('s4')[1]
        result = self.admit(hours=16, block=second)
        self.assertEqual(result['reason'], 'NOT_ADMITTED_CUTOFF')

    def test_strict_before_18_boundary_not_inclusive(self):
        self.assertEqual(self.admit(hours=10.5)['reason'], 'NOT_ADMITTED_BUDGET')
        self.assertTrue(self.admit(hours=10.499)['allowed'])

    def test_debt_never_disappears_for_fast_teacher_transition(self):
        self.assertTrue(self.admit(hours=9.)['allowed'])
        self.assertFalse(self.admit(hours=9., evaluation_debt_hours=1.6)['allowed'])

    def test_admitted_suffix_continues_after_16_but_before_18(self):
        receipt = self.admit()
        result = self.admit(hours=16.2, remaining=1., admitted_receipt=receipt)
        self.assertTrue(result['allowed'])
        self.assertEqual(result['reason'], 'CONTINUE_ADMITTED_BLOCK')
        self.assertFalse(self.admit(hours=17., remaining=1., admitted_receipt=receipt)['allowed'])

    def test_optimizer_closes_at_18(self):
        receipt = self.admit()
        self.assertEqual(self.admit(hours=18., remaining=0., admitted_receipt=receipt)['reason'], 'OPTIMIZER_CLOSED')

    def test_receipt_cannot_attach_new_cases_or_swap_server_clock(self):
        for key, value in (('run_ids', list(self.block.run_ids)[:-1]), ('server_id', 's5'),
                           ('at_utc', self.at(16).isoformat()), ('registry_sha256', 'a' * 64)):
            receipt = self.admit(); receipt[key] = value
            with self.assertRaises(ValueError):
                self.admit(hours=16.2, remaining=.5, admitted_receipt=receipt)

    def test_partial_block_and_underreservation_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'complete canonical'):
            self.admit(block=replace(self.block, cases=self.block.cases[:1], run_ids=self.block.run_ids[:1]))
        with self.assertRaisesRegex(ValueError, 'cannot omit'):
            self.admit(remaining=1.7)

    def test_invalid_hours_and_prestart(self):
        for value in (True, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.admit(remaining=value)
        with self.assertRaises(ValueError):
            self.admit(hours=-1)

    def test_completed_teacher_still_reserves_unfinished_calibration(self):
        teacher = self.block.cases[0]
        value = q.estimate_block_hours(self.block, completed_run_ids=[teacher.run_id])
        self.assertAlmostEqual(value, 5.8)
        value = q.estimate_block_hours(self.block, completed_run_ids=[teacher.run_id], calibrated_reference_ids=[teacher.reference_id])
        self.assertAlmostEqual(value, 4.8)
        with self.assertRaises(ValueError):
            q.estimate_block_hours(self.block, completed_run_ids=[p.case_for('L100I1-T01').run_id])

    def observation(self, **kw):
        c = self.block.cases[-1]
        row = dict(server_id=c.server_id, sensor='GF2', role=c.role, updates=c.updates,
            width=c.width, depth=list(c.depth), component='TRAIN_EVAL', hours=3., completed=True, evaluation_valid=True)
        row.update(kw)
        return row

    def test_conservative_115_full_subprocess_observations(self):
        case = self.block.cases[-1]
        self.assertAlmostEqual(q.estimate_case_hours(case, [self.observation()]), 3.45)
        self.assertAlmostEqual(q.estimate_case_hours(case, [self.observation(hours=1.)]), 2.4)
        self.assertAlmostEqual(q.estimate_case_hours(case, [self.observation(server_id='s3')]), 2.4)
        self.assertAlmostEqual(q.estimate_case_hours(case, [self.observation(updates=50000)]), 2.4)
        self.assertAlmostEqual(q.estimate_case_hours(case, [self.observation(component='TRAIN')]), 2.4)

    def test_warmup_extrapolation_requires_outside_warmup_and_whole_cost(self):
        case = self.block.cases[-1]
        row = self.observation(component='WARMUP_EXTRAPOLATED', hours=4., completed=False)
        self.assertEqual(q.estimate_case_hours(case, [row]), 2.4)
        row.update(warmup_excluded=True, includes_eval_and_io=True)
        self.assertAlmostEqual(q.estimate_case_hours(case, [row]), 4.6)


class AnalysisTests(unittest.TestCase):
    def reports(self, contrast='S4_H010'):
        result = {}
        for a, b in q.CONTRASTS[contrast]:
            for short, delta in ((a, 0), (b, .002)):
                c = p.case_for('L100I1-' + short)
                result[c.case_id] = dict(case_id=c.case_id, server_id=c.server_id,
                    actual_updates=c.updates, reference_id=c.reference_id, official_complete=True,
                    pair_verified=True, same_checkpoint_verified=True,
                    exact_final=dict(HQNR=.962 + delta, ERGAS=.55, D_lambda=.02),
                    rr_val_selected=dict(HQNR=.962 + delta, ERGAS=.55, D_lambda=.02))
        return result

    def test_h_promising_is_analysis_only(self):
        result = q.classify_pairs('S4_H010', self.reports())
        self.assertIn('H_PROMISING', result['classification'])
        self.assertFalse(result['execution_gate'])
        self.assertFalse(result['automatic_recipe_promotion'])

    def test_missing_pair_is_not_cross_server_substitution(self):
        reports = self.reports(); reports.pop('L100I1-S09')
        self.assertEqual(q.classify_pairs('S4_H010', reports)['classification'], ['PAIR_INCOMPLETE'])
        reports = self.reports(); reports['L100I1-S09']['server_id'] = 's5'
        with self.assertRaises(ValueError):
            q.classify_pairs('S4_H010', reports)

    def test_val_joint_requires_strict_actual_joint_same_selection(self):
        reports = self.reports()
        for case in ('L100I1-S08', 'L100I1-S09'):
            reports[case]['rr_val_selected'].update(HQNR=.9640001, ERGAS=.5519999)
        self.assertIn('JOINT_TWO_SEED', q.classify_pairs('S4_H010', reports)['classification'])
        reports['L100I1-S09']['rr_val_selected']['HQNR'] = .964
        self.assertNotIn('JOINT_TWO_SEED', q.classify_pairs('S4_H010', reports)['classification'])

    def test_exact_gain_cannot_hide_val_ergas_loss(self):
        reports = self.reports()
        reports['L100I1-S09']['rr_val_selected']['ERGAS'] = .56
        self.assertNotIn('H_PROMISING', q.classify_pairs('S4_H010', reports)['classification'])

    def test_reference_length_contrast_preserves_explicit_ids(self):
        result = q.classify_pairs('S3_TEACHER_LENGTH', self.reports('S3_TEACHER_LENGTH'))
        self.assertEqual(result['paired_rows'][0]['baseline'], 'L100I1-S02')
        self.assertEqual(result['paired_rows'][1]['candidate'], 'L100I1-S04')

    def test_rr_gain_val_needs_same_direction_not_repeated_magnitude(self):
        reports = self.reports()
        for identifier in ('L100I1-S08', 'L100I1-S09'):
            reports[identifier]['exact_final'].update(ERGAS=.5445)
            reports[identifier]['rr_val_selected'].update(ERGAS=.54945)
        self.assertIn('RR_GAIN_WITH_H_PRESERVED', q.classify_pairs('S4_H010', reports)['classification'])


if __name__ == '__main__':
    unittest.main()
