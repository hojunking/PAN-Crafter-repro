import copy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ablr2 import plan


def panels(server):
    return [{case.case_id: case for case in plan.cases_for(server)
        if case.sweep_id == f'P{k:02}' and case.role == 'T'} for k in range(1, 6)]


class PlanTests(unittest.TestCase):
    def test_source_and_boot_cardinality(self):
        self.assertEqual(len(plan.verify_sources()), 5)
        self.assertEqual(len(plan.CASES), 190)
        self.assertEqual(len({c.run_id for c in plan.CASES}), 190)
        self.assertEqual(sum(c.role == 'T' for c in plan.CASES), 20)
        self.assertEqual(sum(c.role == 'S' for c in plan.CASES), 170)
        self.assertEqual(len(plan.COMPONENTS), 26)
        self.assertEqual(len(plan.GRAPH), 17)

    def test_every_author_csv_numeric_and_dependency_field_matches_config(self):
        with (plan.ROOT / plan.SOURCE_CASES).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            case = plan.case_for(row['run_id'])
            cfg = plan.build_config(case)
            for key, expected in (('bands', case.num_bands), ('max_dn', case.max_dn),
                ('stored_input_channels', case.stored_input_channels), ('width', case.width),
                ('num_updates', case.updates), ('cosine_total', cfg['num_iter']),
                ('warmup', cfg['num_warmup']), ('batch', cfg['batch_size'])):
                self.assertEqual(int(row[key]), expected, (case.run_id, key))
            self.assertEqual(tuple(map(int,row['depth'].split(','))), case.depth)
            self.assertEqual(float(row['u_peak_lr']), cfg['learning_rate'])
            self.assertEqual(float(row['a_peak_lr']), case.aligner_lr)
            self.assertEqual(row['input_layout'], case.input_layout)
            self.assertEqual(row['requires_teacher'] == 'True', case.requires_teacher)
            self.assertEqual(row['requires_calibration'] == 'True', case.requires_calibration)
            if case.role == 'S':
                for key in ('alpha', 'beta', 'lambda_edge'):
                    self.assertEqual(float(row[key]), getattr(case,key))
            else:
                self.assertEqual(float(row['lambda_con']), case.lambda_con)

    def test_pinned_sources_reject_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError): plan.verify_sources(tmp)

    def test_lanes_are_disjoint_and_protected(self):
        for server in ('s3', 's4', 's5', 's0'):
            with self.assertRaises(ValueError): plan.cases_for(server)
        with self.assertRaises(ValueError): plan.verify_lane('s1', 'QB')
        with self.assertRaises(ValueError): plan.sensor_spec('GF2')

    def test_sensor_specs(self):
        for name, bands, train, val in (('WV3', 8, 9714, 1080), ('QB', 4, 17139, None)):
            spec = plan.sensor_spec(name)
            self.assertEqual(spec.num_bands, bands)
            self.assertEqual(spec.max_dn, 2047)
            self.assertEqual(spec.nominal_train_n, train)
            self.assertEqual(spec.nominal_val_n, val)
            self.assertEqual(plan.sensor_spec(spec.to_dict()), spec)
            changed = spec.to_dict(); changed['max_dn'] = 1023
            with self.assertRaises(ValueError): plan.sensor_spec(changed)

    def test_each_sweep_is_balanced_and_order_is_prespecified(self):
        for server, tbase, sbase in (('s1', 781000, 791000), ('s2', 881000, 891000)):
            for k in range(1, 6):
                cases = [c for c in plan.cases_for(server) if c.sweep_id == f'P{k:02}']
                self.assertEqual(len(cases), 19)
                self.assertEqual([c.case_id for c in cases[:2]], ['TPLUS', 'TZERO'] if k % 2 else ['TZERO', 'TPLUS'])
                self.assertEqual(tuple(c.case_id for c in cases[2:]), plan.student_order(k))
                self.assertEqual({c.seed for c in cases[:2]}, {tbase+k})
                self.assertEqual({c.seed for c in cases[2:]}, {sbase+k})
                self.assertEqual(len({c.paired_init_group for c in cases[2:]}), 1)
                self.assertEqual({c.input_layout for c in cases[:2]}, {'P0'})

    def test_teacher_free_has_zero_hidden_references(self):
        for case in plan.CASES:
            if case.case_id not in ('C00', 'C01', 'C02', 'C10'): continue
            self.assertFalse(case.requires_teacher)
            self.assertFalse(case.requires_calibration)
            self.assertIsNone(case.teacher_seed)
            self.assertIsNone(case.reference_id)
            self.assertIsNone(case.teacher_run_id)

    def test_clone_only_and_uniform_kd_do_not_use_calibration(self):
        for case in plan.CASES:
            if case.case_id not in ('C03', 'C04'): continue
            self.assertTrue(case.requires_teacher)
            self.assertFalse(case.requires_calibration)
            self.assertEqual(case.component['teacher_predictions_used'], case.case_id == 'C04')

    def test_recipe_then_deletion_override(self):
        self.assertEqual(plan.component_config('C07', 'R01')['beta'], .2)
        self.assertEqual(plan.component_config('C13', 'R01')['beta'], 0)
        self.assertEqual(plan.component_config('C12', 'R03')['alpha'], 0)
        self.assertEqual(plan.component_config('C12', 'R03')['soft_trust'], True)
        self.assertEqual(plan.component_config('C14', 'R02')['lambda_edge'], 0)
        self.assertEqual(plan.component_config('C04', 'R01')['soft_mode'], 'UNIFORM')
        self.assertEqual(plan.component_config('C10', 'R04')['teacher'], 'NONE')
        self.assertEqual(plan.component_config('C16', 'R05')['a_peak_lr'], 0)
        self.assertEqual(plan.component_config('C06')['q_edge'], 'CONST_HALF')
        self.assertEqual(plan.component_config('C15')['q_edge'], 'TRAIN_MEAN')

    def test_optional_cases_cannot_be_automatically_built(self):
        for name in ('F01', 'F02') + tuple(f'X{k:02}' for k in range(7)):
            with self.assertRaises(ValueError): plan.component_config(name)

    def test_configs_roundtrip_all_boot_cases(self):
        for case in plan.CASES:
            cfg = plan.build_config(case)
            self.assertEqual(plan.validate_config(cfg), case)
            self.assertEqual(cfg['num_iter'], 50000)
            self.assertEqual(cfg['ablr2']['candidate_grid'], list(plan.GRID_STEPS))
            self.assertEqual(cfg['ablr2']['checkpoint_rule'], 'RR_VALIDATION_ARGMIN_ERGAS_THEN_LOWER_STEP')
            self.assertIn(f'/{case.sensor}/{case.server_id}/runs/', cfg['work_dir'])

    def test_no_horizon_or_metric_rule_override(self):
        for field, value in (('num_iter', 100000), ('batch_size', 32), ('mixed_precision', 'fp16')):
            cfg = plan.build_config(plan.CASES[0]); cfg[field] = value
            with self.assertRaises(ValueError): plan.validate_config(cfg)
        self.assertEqual(len(plan.GRID_STEPS), 50)
        self.assertEqual(plan.GRID_STEPS[-2:], (49490, 50000))
        with self.assertRaises(ValueError): plan.grid_steps(100000)

    def test_data_binding_does_not_enable_hidden_teacher(self):
        c00 = next(c for c in plan.CASES if c.case_id == 'C00')
        cfg = plan.build_config(c00); cfg['ablr2']['dataset_manifest'] = 'verified.json'
        cfg['ablr2']['runtime_policy_sha256'] = 'a'*64
        self.assertEqual(plan.validate_config(cfg, require_bound=True), c00)
        cfg['ablr2']['q_cache'] = 'hidden.npz'
        with self.assertRaises(ValueError): plan.validate_config(cfg)

    def test_boot_contract_is_exact_and_immutable(self):
        with self.assertRaises(ValueError): plan.validate_case(replace(plan.CASES[0], teacher_seed=77))
        with self.assertRaises(ValueError): plan.case_for('C07')

    def test_dynamic_path_tokens_and_revision_cannot_escape(self):
        case = plan.full_wave('s1', 'REFRESH5', 'R00', 'r000', 'W001', 1234, [])[0]
        for field, bad in (('wave', '../outside'), ('run_id', case.run_id.replace('REFRESH5', '../REFRESH5')),
                           ('panel_id', '/tmp/panel'), ('paired_init_group', 'A B'),
                           ('recipe_revision', 'r001'), ('status', 'SUCCESS')):
            with self.assertRaises(ValueError): plan.validate_case(replace(case, **{field: bad}))
        cfg = plan.build_config(case)
        cfg['work_dir'] = 'work_dir/ablr2/QB/s2/runs/' + case.run_id
        with self.assertRaises(ValueError): plan.validate_config(cfg)
        for prefix in ('../', 'elsewhere/', '/tmp/'):
            cfg = plan.build_config(case)
            cfg['work_dir'] = prefix + cfg['work_dir']
            if prefix != '/tmp/':
                with self.assertRaises(ValueError): plan.validate_config(cfg)

    def test_seed_formula_idempotent_and_not_component_dependent(self):
        ledger = []
        seed = plan.derive_seed('WV3', 'REFRESH5', 'r001', 'W01', 'P01', 'S', 'INIT_DATA', 1234, ledger)
        key = 'ABLR2|WV3|REFRESH5|r001|W01|P01|S|INIT_DATA|1234'
        expected = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**31-1)
        self.assertEqual(seed, expected)
        self.assertEqual(plan.derive_seed('WV3', 'REFRESH5', 'r001', 'W01', 'P01', 'S', 'INIT_DATA', 1234, ledger), seed)
        self.assertEqual(len(ledger), 1)
        ledger[0]['seed'] += 1
        with self.assertRaises(ValueError): plan.derive_seed('WV3', 'REFRESH5', 'r001', 'W01', 'P01', 'S', 'INIT_DATA', 1234, ledger)

    def test_seed_collision_has_deterministic_counter(self):
        sample = []
        value = plan.derive_seed('QB', 'REFRESH5', 'r000', 'W1', 'P01', 'S', 'INIT_DATA', 1234, sample)
        collision = [{'key': 'different', 'seed': value}]
        second = plan.derive_seed('QB', 'REFRESH5', 'r000', 'W1', 'P01', 'S', 'INIT_DATA', 1234, collision)
        self.assertNotEqual(value, second)
        self.assertEqual(collision[-1]['collision_counter'], 1)

    def test_refresh_is_whole_wave_before_outcomes(self):
        ledger = []
        wave = plan.full_wave('s1', 'REFRESH5', 'R04', 'r004', 'W001', 1234, ledger)
        self.assertEqual(len(wave), 95)
        self.assertEqual(len(ledger), 10)
        self.assertEqual(len({c.run_id for c in wave}), 95)
        self.assertEqual({c.teacher_kind for c in wave if c.case_id == 'TPLUS'}, {'TC3'})
        self.assertEqual({c.lambda_con for c in wave if c.case_id == 'TPLUS'}, {3e-4})
        self.assertEqual({c.lambda_con for c in wave if c.case_id == 'TZERO'}, {0})
        for c in wave: self.assertEqual(plan.validate_config(plan.build_config(c)), c)
        self.assertEqual(plan.full_wave('s1', 'REFRESH5', 'R04', 'r004', 'W001', 1234, ledger), wave)

    def test_recheck_has_five_pairs_and_anchor_without_new_teachers(self):
        for relation, size in (('L04', 15), ('D11', 10)):
            cases = plan.recheck_cases('s1', relation, 'R00', 'r000', relation+'W01', panels('s1'), 1234, [])
            self.assertEqual(len(cases), size)
            self.assertTrue(all(c.role == 'S' for c in cases))
            for number in range(1, 6):
                block = [c for c in cases if c.sweep_id == f'P{number:02}']
                self.assertIn('C07', [c.case_id for c in block])
                self.assertEqual(len({c.seed for c in block}), 1)
                self.assertEqual({c.teacher_seed for c in block if c.requires_teacher}, {781000+number})

    def test_recheck_pool_cannot_cross_sensor_or_tc3_recipe(self):
        with self.assertRaises(ValueError): plan.recheck_cases('s1', 'L04', 'R00', 'r000', 'W01', panels('s2'), 1, [])
        with self.assertRaises(ValueError): plan.recheck_cases('s1', 'L04', 'R04', 'r004', 'W01', panels('s1'), 1, [])

    def test_screen_fixed_seeds_common_init_and_fresh_tc3(self):
        pool = panels('s2')
        blocks = [dict(teacher_seed=881000+k, student_seed=891000+k, teachers=pool[k-1]) for k in (1, 2)]
        cases = plan.screen_cases('s2', ('R00', 'R01', 'R04'), {'R00': 'r000', 'R01': 'r001', 'R04': 'r004'}, 'F01', blocks, 'D11')
        self.assertEqual(sum(c.role == 'T' for c in cases), 2)
        for c in cases:
            if c.role == 'T': self.assertEqual(c.teacher_kind, 'TC3')
        for sweep in ('P01', 'P02'):
            students = [c for c in cases if c.role == 'S' and c.sweep_id == sweep]
            self.assertEqual(len({c.seed for c in students}), 1)
            self.assertEqual(len({c.paired_init_group for c in students}), 1)
        blocks[0]['student_seed'] += 1
        with self.assertRaises(ValueError): plan.screen_cases('s2', ('R00',), {'R00': 'r000'}, 'F01', blocks)

    def test_equivalent_removed_recipe_changes_are_not_new_observations(self):
        pool = panels('s1')
        blocks = [dict(teacher_seed=781000+k, student_seed=791000+k, teachers=pool[k-1]) for k in (1, 2)]
        jobs = plan.screen_cases('s1', ('R00', 'R01'), {'R00': 'r000', 'R01': 'r001'}, 'F01', blocks, 'D13')
        def signature(case, reference='same_verified_reference'):
            return plan.execution_signature(plan.build_config(case), source_sha256='a'*64,
                data_sha256='b'*64, reference_identity=reference)['sha256']
        c13 = [c for c in jobs if c.case_id == 'C13' and c.sweep_id == 'P01']
        c07 = [c for c in jobs if c.case_id == 'C07' and c.sweep_id == 'P01']
        self.assertEqual(signature(c13[0]), signature(c13[1]))
        self.assertNotEqual(signature(c07[0]), signature(c07[1]))
        self.assertNotEqual(signature(c13[0]), signature(c13[0], 'another_reference'))
        boot = next(c for c in plan.CASES if c.server_id == 's1' and c.sweep_id == 'P01' and c.case_id == 'C13')
        self.assertEqual(signature(boot), signature(c13[0]))

    def test_identity_and_frozen_a_ignore_unused_r05_lr_schedule(self):
        pool = panels('s1')
        blocks = [dict(teacher_seed=781000+k, student_seed=791000+k, teachers=pool[k-1]) for k in (1, 2)]
        jobs = plan.screen_cases('s1', ('R00', 'R05'), {'R00': 'r000', 'R05': 'r005'}, 'F01', blocks, 'D16')
        frozen = [c for c in jobs if c.case_id == 'C16' and c.sweep_id == 'P01']
        signatures = [plan.execution_signature(plan.build_config(c), source_sha256='a'*64,
            data_sha256='b'*64, reference_identity='same')['sha256'] for c in frozen]
        self.assertEqual(signatures[0], signatures[1])

    def test_registry_discloses_no_launch_and_no_fake_targets(self):
        doc = plan.registry_document()
        self.assertEqual(doc['source_status'], 'AUTHOR_BUNDLE_VERIFIED')
        self.assertIsNone(doc['thresholds'])
        self.assertIsNone(doc['max_campaign_cycles'])
        self.assertTrue(doc['require_initial_operator_lease'])
        self.assertFalse(doc['optional_auto_release'])


if __name__ == '__main__': unittest.main()
