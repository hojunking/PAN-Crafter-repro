import copy
from dataclasses import asdict, replace
import unittest

from l100 import plan as p


class PlanTests(unittest.TestCase):
    def test_md_pinned_and_missing_inputs_disclosed(self):
        self.assertEqual(p.verify_sources(), dict(p.SOURCE_SHAS))
        registry = p.registry_document()
        self.assertEqual(registry['source_status'], 'MD_DERIVED_MISSING_DESIGN_ASSETS')
        self.assertEqual(len(registry['missing_author_design_assets']), 2)
        self.assertEqual(p.registry_sha256(), p._object_sha(registry))

    def test_exact_counts_and_horizons(self):
        self.assertEqual((len(p.CASES), len(p.CORE_CASES), len(p.DEFERRED_CASES)), (22, 18, 4))
        teachers = [c for c in p.CORE_CASES if c.role == 'T']
        students = [c for c in p.CORE_CASES if c.role == 'S']
        self.assertEqual(sorted(c.updates for c in teachers), [50000] + [100000] * 3)
        self.assertEqual(sorted(c.updates for c in students), [50000] * 2 + [100000] * 12)
        self.assertEqual(len({c.run_id for c in p.CASES}), 22)
        self.assertTrue(all(c.run_id.startswith('L100I1_GF2_') and c.run_id.endswith('_v2') for c in p.CASES))

    def test_no_s1_s2_or_cross_server_edges(self):
        for c in p.CASES:
            self.assertEqual(c.server_id, c.teacher_owner)
            self.assertEqual(p.teacher_for(c.reference_id).server_id, c.server_id)
        for server in ('s1', 's2', 's6'):
            with self.assertRaises(ValueError):
                p.cases_for(server)
        self.assertEqual({p.teacher_for(r).seed for r in p.REFERENCES}, {94001, 94002, 94003})

    def test_exact_core_queues(self):
        expected = dict(s3=['T01', 'T02', 'S01', 'S02', 'S03', 'S04', 'S05', 'S06'],
                        s4=['T03', 'S07', 'S08', 'S09', 'S10'], s5=['T04', 'S11', 'S12', 'S13', 'S14'])
        for server, ids in expected.items():
            self.assertEqual([c.case_id for c in p.cases_for(server)], ['L100I1-' + i for i in ids])
            self.assertEqual([c for b in p.blocks_for(server) for c in b.cases], list(p.cases_for(server)))

    def test_atomic_first_chain_and_second_pair(self):
        for server, lengths in dict(s3=(5, 3), s4=(3, 2), s5=(3, 2)).items():
            first, second = p.blocks_for(server)
            self.assertEqual((len(first.cases), len(second.cases)), lengths)
            self.assertEqual(first.cases[0].role, 'T')
            self.assertTrue(all(c.role == 'S' for c in second.cases))
        self.assertEqual(len(p.blocks_for('s3')[0].to_dict()['calibration_ids']), 2)

    def test_deferred_is_registered_never_executable(self):
        self.assertEqual({c.profile for c in p.DEFERRED_CASES}, {'ARW', 'E02'})
        for case in p.DEFERRED_CASES:
            self.assertEqual(case.status, 'DEFERRED_USER_RELEASE')
            with self.assertRaisesRegex(ValueError, 'DEFERRED'):
                p.build_config(case)
            with self.assertRaises(ValueError):
                p.validate_case(replace(case, status='PLANNED', tier='CORE'))
        self.assertEqual(p.PROFILES['ARW'].aligner_schedule, 'ARW')
        self.assertEqual(p.PROFILES['E02'].lambda_edge, .0002)

    def test_exact_candidate_grids(self):
        for horizon in (50000, 100000):
            grid = p.grid_steps(horizon)
            self.assertEqual(len(grid), 50)
            self.assertEqual(tuple(sorted(set(grid))), grid)
            self.assertEqual(grid[-1], horizon)
            self.assertIn(50000, grid)
        self.assertNotIn(50500, p.GRID100)
        self.assertNotIn(1000, p.GRID100)
        self.assertEqual(p.GRID100[24], 50000)
        self.assertEqual(p.GRID100, tuple(50000 if 2 * x == 50500 else 2 * x for x in p.GRID50))
        for invalid in (True, 10000, 50001):
            with self.assertRaises(ValueError):
                p.grid_steps(invalid)

    def test_diagnostics_not_selection_candidates(self):
        self.assertEqual(p.diagnostic_steps('T', 50000), (0, 10000, 25000, 50000))
        self.assertEqual(p.diagnostic_steps('S', 100000), (0, 1000, 5000, 10000, 25000, 50000, 75000, 100000))
        self.assertTrue(set(p.diagnostic_steps('S', 100000)).issubset(p.fullstate_steps('S', 100000)))
        self.assertNotIn(1000, p.grid_steps(100000))

    def test_build_and_case_roundtrip_all_core(self):
        for case in p.CORE_CASES:
            cfg = p.build_config(case)
            self.assertEqual(p.case_from_config(cfg), case)
            self.assertEqual(p.validate_config(cfg), case)
            self.assertEqual(cfg['num_iter'], case.updates)
            self.assertEqual(cfg['l100']['teacher_ref_step'], case.teacher_updates)
            self.assertEqual(cfg['l100']['teacher_owner'], case.server_id)
            self.assertEqual(cfg['model_args']['attn_locations'], [])
            self.assertFalse(cfg['model_args']['mode_modulation'])
            self.assertIsNone(cfg['l100']['tau_R'])
            self.assertIsNone(cfg['l100']['q_cache_sha256'])
            self.assertEqual(cfg['l100']['case_contract'], asdict(case))

    def test_only_one_coefficient_changes(self):
        for base_id, alt_id, changing, old, new in (
            ('S07', 'S08', 'alpha', 1., .1), ('S11', 'S12', 'lambda_edge', .002, .001)):
            base, alt = [p.build_config('L100I1-' + identifier) for identifier in (base_id, alt_id)]
            self.assertEqual(base['seed'], alt['seed'])
            self.assertEqual(base['l100']['teacher_run_id'], alt['l100']['teacher_run_id'])
            for key in ('alpha', 'beta', 'lambda_edge', 'aligner_lr', 'consistency_weight'):
                self.assertEqual((base['l100'][key], alt['l100'][key]),
                    (old, new) if key == changing else (base['l100'][key], base['l100'][key]))

    def test_prefix_seeds_not_dependent_on_profile_horizon_or_run(self):
        for ids in (('T01', 'T02'), ('S01', 'S02', 'S03'), ('S07', 'S08'), ('S11', 'S12')):
            configs = [p.build_config('L100I1-' + identifier) for identifier in ids]
            self.assertEqual(len({c['seed'] for c in configs}), 1)
            self.assertEqual(len({c['l100']['corruption_seed'] for c in configs}), 1)

    def test_strict_recipe_but_runtime_binding_permitted(self):
        cfg = p.build_config('L100I1-S08')
        for key in ('dataset_manifest', 'reference_manifest', 'q_cache'):
            cfg['l100'][key] = '/measured/local/' + key
        cfg['l100']['runtime_policy_sha256'] = 'a' * 64
        cfg['train_feeder_args']['dataroot'] = '/data/train.h5'
        p.validate_config(cfg, require_bound=True)
        for target, field, value in ((cfg, 'num_iter', 50000), (cfg['l100'], 'alpha', 0.),
                                      (cfg['model_args'], 'hidden_size', 96)):
            old = target[field]; target[field] = value
            with self.assertRaises(ValueError):
                p.validate_config(cfg)
            target[field] = old
        cfg['l100']['teacher_owner'] = 's3'
        with self.assertRaises(ValueError):
            p.case_from_config(cfg)

    def test_missing_runtime_pin_rejected_for_training(self):
        with self.assertRaisesRegex(ValueError, 'runtime policy'):
            p.validate_config(p.build_config('L100I1-T01'), require_bound=True)

    def test_profile_registry_and_case_objects_are_immutable(self):
        with self.assertRaises(TypeError):
            p.PROFILES['BASE'] = p.Profile(alpha=0)
        case = p.case_for('L100I1-S07')
        with self.assertRaises(ValueError):
            p.validate_case(replace(case, updates=50000))


if __name__ == '__main__':
    unittest.main()
