import copy
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ablr2.plan import CASES, build_config
from ablr2.analysis import assemble_wave, cumulative_balanced_reports, case_report, write_panel_csv
from ablr2.common import run_dir, atomic_json


class AnalysisTests(unittest.TestCase):
    def panel(self, server='s1'):
        cases = tuple(c for c in CASES if c.server_id == server)
        observations = {}
        for case in cases:
            value = dict(HQNR=.96, ERGAS=1., E_val=1.1, D_lambda=.02, D_s=.02, SAM=2., SCC=.9)
            observations[case.run_id] = dict(run_id=case.run_id, case_id=case.case_id, sensor=case.sensor,
                source_identity={'content_sha256': 'source'}, data_sha256='data', VAL=value, EXACT50K=dict(value),
                init_U_sha256='U' + case.role, init_A_sha256='A', sampler_sha256=str(case.seed),
                rng_roles={'data': case.seed})
        return cases, observations

    def test_complete95_runs_form85_student_panel_not85_independent_repeats(self):
        cases, rows = self.panel()
        report = assemble_wave(cases, rows)
        self.assertEqual(len(report['panelrows']), 85)
        self.assertEqual(report['sweep_count'], 5)
        self.assertEqual(len(report['relations']), 17)
        self.assertTrue(all(r['classification'] == 'NEAR_ZERO' for r in report['relations'].values()))
        self.assertFalse(report['statistical_significance_claim'])

    def test_partial95_is_not_threshold_zero(self):
        cases, rows = self.panel()
        with self.assertRaises(ValueError): assemble_wave(cases[:-1], rows)

    def test_mixed_sensor_thresholds_rejected(self):
        cases, rows = self.panel('s1')
        other_cases, other_rows = self.panel('s2')
        thresholds = assemble_wave(other_cases, other_rows)['thresholds']
        with self.assertRaises(ValueError): assemble_wave(cases, rows, thresholds)

    def test_mixed_source_and_unmatched_stream_rejected(self):
        cases, rows = self.panel()
        rows[cases[-1].run_id]['source_identity'] = {'content_sha256': 'other'}
        with self.assertRaises(ValueError): assemble_wave(cases, rows)

    def test_cumulative_rejects_duplicate_and_targeted_panels(self):
        cases, rows = self.panel()
        report = assemble_wave(cases, rows)
        result = cumulative_balanced_reports([report])
        self.assertEqual(result['sweeps_per_case'], 5)
        self.assertFalse(result['rechecks_included'])
        with self.assertRaises(ValueError): cumulative_balanced_reports([report, report])
        report['phase'] = 'RECHECK5'
        with self.assertRaises(ValueError): cumulative_balanced_reports([report])
        cases, rows = self.panel()
        rows[cases[-1].run_id]['sampler_sha256'] = 'changed'
        with self.assertRaises(ValueError): assemble_wave(cases, rows)

    def test_fit_reuse_preserves_physical_metric_provenance(self):
        physical = next(c for c in CASES if c.case_id == 'C00')
        run = f'ABLR2_{physical.sensor}_{physical.server_id}_{physical.recipe_revision}_FIT_ROUND_F001_{physical.sweep}_{physical.case_id}_SS{physical.seed}_FRESH50'
        logical = replace(physical, phase='FIT_ROUND', wave='F001', run_id=run)
        cfg, original_cfg = build_config(logical), build_config(physical)
        signature = {'sha256': 'same', 'numerical_identity': {'seed': physical.seed}}
        observed = dict(run_id=physical.run_id, source_identity={'content_sha256': 'source'}, data_sha256='data',
                        config_sha256='physical_config', VAL={'HQNR': .961}, independent_observation_run_id=physical.run_id)
        with TemporaryDirectory() as folder:
            receipt = dict(schema='ABLR2_REUSE_v1', logical_case=asdict(logical), physical_run_id=physical.run_id,
                logical_execution_signature=signature, physical_execution_signature=signature,
                source_identity=observed['source_identity'], data_sha256='data', physical_raw_grid_sha256='status')
            atomic_json(run_dir(logical.run_id, folder) / 'meta/reuse.json', receipt)
            with patch('ablr2.analysis.find_config', side_effect=[cfg, original_cfg]), \
                    patch('ablr2.plan.case_for', return_value=physical), \
                    patch('ablr2.common.execution_identity', return_value=signature, create=True), \
                    patch('ablr2.analysis.sha256', return_value='status'), \
                    patch('ablr2.analysis.case_report', return_value=observed):
                actual = case_report(logical, folder)
            self.assertEqual(actual['VAL'], observed['VAL'])
            self.assertEqual(actual['config_sha256'], 'physical_config')
            self.assertEqual(actual['run_id'], logical.run_id)
            self.assertEqual(actual['independent_observation_run_id'], physical.run_id)
            self.assertEqual(actual['incremental_compute_seconds'], 0)
            self.assertFalse(actual['new_independent_observation'])

    def test_reuse_cycle_rejected(self):
        case = CASES[0]
        with self.assertRaises(ValueError): case_report(case, _seen={case.run_id})

    def test_csv_publication_is_atomic_full_precision_and_idempotent(self):
        row = dict(run_id='run', case_id='C00', sensor='WV3', server='s1', phase='BOOT5', wave='BOOT5',
                   sweep='P01', recipe_id='R00', recipe_revision='r000', teacher_seed=None, student_seed=791001,
                   reference_id=None, source_identity={'content_sha256': 'source'}, data_sha256='data',
                   VAL={'HQNR': .961234567891}, EXACT50K={'HQNR': .951234567891})
        report = {'panelrows': [row]}
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'panel.csv'
            with patch('ablr2.analysis.os.link', side_effect=OSError('simulated publication interruption')):
                with self.assertRaises(OSError): write_panel_csv(path, report)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])
            write_panel_csv(path, report)
            original = path.read_bytes()
            self.assertIn(b'0.961234567891', original)
            write_panel_csv(path, report)
            self.assertEqual(path.read_bytes(), original)
            row['VAL']['HQNR'] -= .1
            with self.assertRaises(ValueError): write_panel_csv(path, report)


if __name__ == '__main__': unittest.main()
