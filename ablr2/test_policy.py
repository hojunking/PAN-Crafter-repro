import copy
import unittest

from ablr2.plan import GRAPH, MAIN_CASES, LEGACY_MAIN_CASES
from ablr2.policy import (IMPROVEMENTS, calibrate_thresholds, choose_recipe, choose_recipes,
    choose_relation, classify_relation, flow_dashboard, paired_deltas, recheck_outcome, screen_candidate)


def metric(h=.93, e=3., ev=3., dl=.02):
    return dict(HQNR=h, ERGAS=e, E_val=ev, D_lambda=dl)


def pairs(dh=0., re=0., n=5):
    return [dict(parent=metric(), child=metric(.93+dh, 3*(1+re))) for _ in range(n)]


def thresholds():
    return dict(epsilon_H=1e-5, epsilon_E=1e-4, delta_H=.001, delta_E=.003)


class PolicyTests(unittest.TestCase):
    def test_c17_cannot_change_frozen_legacy_thresholds_or_open_diagnostic_queue(self):
        legacy=[{case:metric(.80+.001*k*i,2+.05*k*i) for i,case in enumerate(LEGACY_MAIN_CASES)} for k in range(5)]
        extended=[dict(panel,C17=metric(.999 if k%2 else .1,100 if k%2 else .01)) for k,panel in enumerate(legacy)]
        self.assertEqual(calibrate_thresholds('GF2',legacy),calibrate_thresholds('GF2',extended))
        self.assertEqual(calibrate_thresholds('GF2',extended)['relation_count'],17)
        reports={'A17_SCRATCH':dict(classification='REVERSAL'),'A17':dict(classification='NEAR_ZERO')}
        self.assertEqual(choose_relation(reports),'A17')
        self.assertIsNone(choose_relation(reports,('A17',)))

    def test_pilot_requires_exact_complete_five_by_seventeen(self):
        panels = [{case: metric() for case in MAIN_CASES} for _ in range(5)]
        actual = calibrate_thresholds('WV3', panels)
        self.assertEqual(actual['epsilon_H'], 1e-5)
        self.assertEqual(actual['epsilon_E'], 1e-4)
        self.assertEqual(actual['delta_H'], .001)
        self.assertEqual(actual['delta_E'], .003)
        self.assertTrue(actual['H_cap_applied'])
        self.assertFalse(actual['statistical_significance_claim'])
        with self.assertRaises(ValueError): calibrate_thresholds('WV3', panels[:4])
        del panels[4]['C16']
        with self.assertRaises(ValueError): calibrate_thresholds('WV3', panels)

    def test_mad_formula_and_caps(self):
        panels = []
        for k in range(5):
            panels.append({case: metric(.80 + .001*k*number, 2+.05*k*number)
                for number, case in enumerate(MAIN_CASES)})
        doc = calibrate_thresholds('QB', panels)
        self.assertEqual(doc['raw_epsilon_H'], .25*sorted(r['MAD_H'] for r in doc['per_relation'])[8])
        self.assertGreaterEqual(doc['epsilon_H'], 1e-5)
        self.assertLessEqual(doc['epsilon_H'], .001)
        self.assertGreaterEqual(doc['epsilon_E'], 1e-4)
        self.assertLessEqual(doc['epsilon_E'], .005)
        self.assertEqual(doc['delta_E'], max(.003, 2*doc['epsilon_E']))

    def test_registered_classifications(self):
        for dh, re, wanted in ((.002, -.01, 'JOINT_GAIN'), (.002, 0, 'H_GAIN_SAFE'),
                (0, -.01, 'RR_GAIN_SAFE'), (-.002, .01, 'REVERSAL'),
                (.002, .01, 'TRADEOFF'), (-.002, -.01, 'TRADEOFF'),
                (0, 0, 'NEAR_ZERO'), (.0002, 0, 'UNSTABLE')):
            self.assertEqual(classify_relation(pairs(dh, re), thresholds())['classification'], wanted)

    def test_reversal_recomputes_relative_ratio(self):
        report = classify_relation(pairs(-.002, .25), thresholds())
        self.assertAlmostEqual(report['paired_deltas'][0]['rE'], .25)
        self.assertAlmostEqual(report['inverse_deltas'][0]['rE'], -.2)
        self.assertEqual(report['classification'], 'REVERSAL')

    def test_support_is_ceiling_eighty_percent(self):
        values = pairs(.002, 0, 6)
        values[0]['child']['HQNR'] = .929
        report = classify_relation(values, thresholds())
        self.assertEqual(report['support_required'], 5)
        self.assertEqual(report['classification'], 'H_GAIN_SAFE')
        values[1]['child']['HQNR'] = .929
        self.assertNotEqual(classify_relation(values, thresholds())['classification'], 'H_GAIN_SAFE')

    def test_individual_h_and_e_guards(self):
        values = pairs(.002, 0)
        values[0]['child']['ERGAS'] = 3.04
        self.assertNotIn(classify_relation(values, thresholds())['classification'], IMPROVEMENTS)
        values = pairs(0, -.01)
        values[0]['child']['HQNR'] = .92
        self.assertNotIn(classify_relation(values, thresholds())['classification'], IMPROVEMENTS)

    def test_invalid_is_not_negative_or_seed_replacement(self):
        self.assertEqual(classify_relation(pairs(n=4), thresholds())['classification'], 'INCOMPLETE')
        values = pairs(); values[0]['child']['ERGAS'] = float('nan')
        self.assertEqual(classify_relation(values, thresholds())['classification'], 'INVALID')
        self.assertIsNone(choose_relation({'E00': dict(classification='INVALID')}))

    def test_checkpoint_sensitivity_preserves_primary(self):
        report = classify_relation(pairs(.002, -.01), thresholds(), pairs(-.002, .01))
        self.assertEqual(report['classification'], 'JOINT_GAIN')
        self.assertEqual(report['flags'], ['CHECKPOINT_SENSITIVE'])
        report = classify_relation(pairs(.002, -.01), thresholds(), pairs())
        self.assertEqual(report['flags'], [])

    def test_priority_once_per_relation_recipe(self):
        reports = {'L01': dict(classification='NEAR_ZERO'), 'L02': dict(classification='UNSTABLE'),
            'D11': dict(classification='REVERSAL'), 'D12': dict(classification='REVERSAL')}
        self.assertEqual(choose_relation(reports), 'D11')
        self.assertEqual(choose_relation(reports, ('D11',)), 'D12')
        self.assertEqual(choose_relation(reports, ('D11', 'D12')), 'L02')
        self.assertIsNone(choose_relation(reports, reports.keys()))
        self.assertEqual(choose_relation(reports, last_measurements={'D11': '2026-09-21', 'D12': '2026-09-20'}), 'D12')

    def test_recipe_bank_priority_is_finite(self):
        self.assertEqual(choose_recipes('D12'), ('R03', 'R01'))
        self.assertEqual(choose_recipes('D11'), ('R04', 'R01'))
        self.assertEqual(choose_recipes('D16'), ('R05', 'R01'))
        self.assertEqual(choose_recipes('D15', ('R01', 'R02', 'R03', 'R04')), ('R05',))
        self.assertEqual(choose_recipes(None, ('R01', 'R02', 'R03', 'R04', 'R05')), ())

    def test_screen_validation_gain_path(self):
        values = [dict(parent=metric(), child=metric(.929, 10., 2.98, .0205)) for _ in range(2)]
        report = screen_candidate('R01', values, thresholds())
        self.assertTrue(report['eligible'])
        self.assertTrue(report['eval_path'])
        self.assertFalse(report['h_path'])
        self.assertTrue(report['test_aware'])
        # RR test ERGAS is deliberately worse: screen uses actual validation E.
        values[1]['child']['E_val'] = 3.001
        self.assertFalse(screen_candidate('R01', values, thresholds())['eligible'])

    def test_screen_h_path_and_guards(self):
        values = [dict(parent=metric(), child=metric(.932, 3., 3.009, .0205)) for _ in range(2)]
        self.assertTrue(screen_candidate('R02', values, thresholds())['h_path'])
        values[0]['child']['E_val'] = 3.04
        self.assertFalse(screen_candidate('R02', values, thresholds())['eligible'])
        values[0]['child']['E_val'] = 3.0; values[0]['child']['D_lambda'] = .023
        self.assertFalse(screen_candidate('R02', values, thresholds())['eligible'])

    def test_screen_two_blocks_required_and_recipe_tiebreak(self):
        self.assertFalse(screen_candidate('R01', [], thresholds())['eligible'])
        a = dict(recipe_id='R01', eligible=True, both_paths=False, median_E_val=2., median_H=.95)
        b = dict(recipe_id='R02', eligible=True, both_paths=True, median_E_val=3., median_H=.94)
        self.assertEqual(choose_recipe('R00', [a, b]), 'R02')
        b['both_paths'] = False
        self.assertEqual(choose_recipe('R00', [a, b]), 'R01')
        a['eligible'] = False; b['eligible'] = False
        self.assertEqual(choose_recipe('R00', [a, b]), 'R00')
        with self.assertRaises(ValueError): choose_recipe('R00', [a, b, a])

    def test_flow_alert_is_not_stop_promotion_or_monotonicity_claim(self):
        reports = {key: dict(classification='NEAR_ZERO') for key in GRAPH}
        reports['E00']['classification'] = 'H_GAIN_SAFE'
        report = flow_dashboard(reports)
        self.assertEqual(report['alert'], 'FLOW_CANDIDATE_DEV')
        self.assertFalse(report['early_stop'])
        self.assertFalse(report['automatic_promotion'])
        self.assertFalse(report['monotonicity_proven'])
        self.assertEqual(report['ladder']['undetected'], 7)
        self.assertEqual(report['full_minus']['undetected'], 9)
        reports['D11']['classification'] = 'REVERSAL'
        self.assertIsNone(flow_dashboard(reports)['alert'])

    def test_recheck_outcomes_do_not_repeat_until_win(self):
        for first, second, wanted in (('REVERSAL', 'REVERSAL', 'REPEATED_NEGATIVE_DEV'),
            ('REVERSAL', 'JOINT_GAIN', 'SEED/REFERENCE_SENSITIVE_DEV'),
            ('NEAR_ZERO', 'NEAR_ZERO', 'SMALL_OR_UNDETECTED_DEV'),
            ('TRADEOFF', 'TRADEOFF', 'PERSISTENT_TRADEOFF_DEV')):
            self.assertEqual(recheck_outcome(dict(classification=first), dict(classification=second)), wanted)


if __name__ == '__main__': unittest.main()
