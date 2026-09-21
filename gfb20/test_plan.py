import copy
import unittest
from gfb20.plan import *


class PlanTests(unittest.TestCase):
    def test_author_registry_and_counts(self):
        self.assertEqual(validate_registry()['cases'],36)
        self.assertEqual([len(cases_for(s)) for s in SERVERS],[16,10,10])
        self.assertEqual(sum(c.role=='T' for c in CASES),0)

    def test_candidate_counts_and_50500_not_diagnostic50000(self):
        self.assertEqual(len(grid_steps(20000)),20)
        self.assertEqual(len(grid_steps(100000)),50)
        self.assertIn(50500,grid_steps(100000))
        self.assertNotIn(50000,grid_steps(100000))
        self.assertIn(50000,diagnostic_steps(100000))
        self.assertNotIn(0,grid_steps(20000))

    def test_case_config_roundtrip_and_guard(self):
        for case in CASES:
            cfg=build_config(case)
            self.assertEqual(validate_config(cfg),case)
            bad=copy.deepcopy(cfg);bad['batch_size']=24
            with self.assertRaises(ValueError):validate_config(bad)
            with self.assertRaises(ValueError):validate_config(cfg,require_bound=True)

    def test_ft_lineage_uses_parent_not_fresh(self):
        self.assertEqual(case_for('A03').lifetime_student_updates,70000)
        self.assertEqual(case_for('A01').lifetime_student_updates,120000)
        self.assertEqual(case_for('A11').lifetime_student_updates,100000)
        self.assertTrue(case_for('A12').is_ft)
        self.assertFalse(case_for('A11').is_ft)

    def test_selected_template_needs_binding_only_allowed_choice(self):
        self.assertEqual(build_config('B08','E100')['gfb20']['profile'],'E100')
        self.assertEqual(case_for('B08').profile,'B_SELECTED')
        with self.assertRaises(ValueError):build_config('B08','H010')
        with self.assertRaises(ValueError):build_config('B01','K1')
        with self.assertRaises(ValueError):cases_for('s1')
        with self.assertRaises(ValueError):cases_for('s2')


if __name__=='__main__':unittest.main()
