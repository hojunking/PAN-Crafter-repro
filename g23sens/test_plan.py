import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g23sens import common as C,plan as P


class PlanTests(unittest.TestCase):
    def test_supplied_bundle_and_exact_generator(self):
        self.assertTrue(P.verify_sources())
        for server in ('s4','s5'):
            for cycle in range(6):
                cases=P.cycle_cases(server,cycle)
                self.assertEqual(len(cases),7);self.assertEqual(cases[0]['case_id'],'BASE')
                for case in cases:
                    P.validate_case(case);self.assertEqual(P.case_for(case['run_id']),case)
                    differences=[k for k,v in case['parameters'].items() if v!=cases[0]['parameters'][k]]
                    self.assertEqual(len(differences),case['case_id']!='BASE')

    def test_config_computational_overrides_and_no_old_budget(self):
        for server in ('s4','s5'):
            for case in P.cycle_cases(server,0):
                cfg=P.build_config(case)
                self.assertEqual(P.validate_config(cfg),case)
                self.assertNotIn('budget',cfg['kdv']);self.assertNotIn('qrc24',cfg['kdv'])
                self.assertNotIn('select_on',cfg);self.assertNotIn('fr_select_indices',cfg)
                self.assertEqual(cfg['kdv']['baseline_run'],case['local_baseline_run_id'])
                self.assertEqual(cfg['kdv']['rec']['tau'],P.TAU0)
                self.assertEqual(cfg['kdv']['rec']['tau']*cfg['kdv']['rec']['tau_scale'],case['resolved_values']['tau_R_used'])
                self.assertEqual(cfg['kdv']['aligner_lr'],case['resolved_values']['A_peak_lr'])
                self.assertEqual(cfg['kdv']['qrecon']['q_ref'],case['resolved_values']['q_ref'])
                self.assertNotIn('q_ref_scale',cfg['kdv']['qrecon'])
                self.assertEqual(cfg['g23sens']['primary_selection'],'EXACT_50000')

    def test_changed_recipe_not_accepted(self):
        cfg=P.build_config(P.make_case('s4',0,'BASE'));cfg['model_args']['depth']=[1,2,2]
        with self.assertRaises(ValueError):P.validate_config(cfg)
        with self.assertRaises(ValueError):P.validate_config(P.build_config(P.make_case('s4',0,'BASE')),require_bound=True)

    def test_unique_attempt_paths_and_server_isolation(self):
        a=P.make_case('s4',0,'BASE');b=P.make_case('s5',0,'BASE')
        self.assertNotEqual(C.run_dir(a),C.run_dir(b))
        self.assertNotEqual(C.run_dir(a),C.run_dir(a,attempt=1))
        for server in ('s1','s2','s3'):
            with self.assertRaises(ValueError):C.camp(C.ROOT,server)
        with self.assertRaises(ValueError):C.run_dir('../foreign')

    def test_no_seed_wrap_and_six_cycle_order(self):
        self.assertEqual([P.seed_for(c) for c in range(3)],[1234,2000000,2000001])
        self.assertEqual(len({tuple(P.order_for('s4',c)) for c in range(6)}),6)
        with self.assertRaises(ValueError):P.seed_for(2**32)
        self.assertEqual(P.next_cursor(4,6),dict(cycle=5,position=0))

    def test_source_has_transitive_metric_dependencies(self):
        src=C.source_identity()
        for name in ('feeders/feeder.py','kdv/qrecon.py','pcrepro/evaluation.py','pcrepro/data.py',
                     'tools/eval_dlpan.py','tools/metrics/q2n.py','gspread/gspread_upload.py'):
            self.assertIn(name,src['files'])
        self.assertEqual(src['content_sha256'],C.object_sha(src['files']))

    def test_immutable_writes_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'identity.json'
            C.immutable_json(path,dict(a=1));C.immutable_json(path,dict(a=1))
            with self.assertRaises(ValueError):C.immutable_json(path,dict(a=2))
            self.assertEqual(C.read(path),dict(a=1))

    def test_real_config_round_trip_preserves_small_numeric_coefficients(self):
        config=P.build_config(P.make_case('s4',0,'BASE'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';C.atomic_json(path,config)
            actual=C.read_config(path)
            self.assertEqual(actual,config);self.assertIsInstance(actual['kdv']['rec']['eps'],float)

    def test_corrupt_full_state_is_distinct_from_changed_source(self):
        from g23sens.controller import resume_available,ResumeUnusable
        cfg=dict(g23sens=dict(source_identity={'test':1},binding_sha256='binding'))
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            self.assertFalse(resume_available(work,cfg))
            identity=dict(config_sha256=C.object_sha(cfg),source_identity={'test':1},
                          bindings_sha256='binding',full_state=True,training_state_sha256='absent')
            C.atomic_json(work/'last/identity.json',identity)
            with self.assertRaises(ResumeUnusable):resume_available(work,cfg)
            identity['source_identity']={'changed':True};C.atomic_json(work/'last/identity.json',identity)
            with self.assertRaises(ValueError):resume_available(work,cfg)


if __name__=='__main__':unittest.main()
