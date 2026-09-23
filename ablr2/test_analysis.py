import copy
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ablr2.plan import CASES, build_config
from ablr2.analysis import (assemble_wave, cumulative_balanced_reports, case_report, write_panel_csv,
    analyze_wave,refresh_extended_reports)
from ablr2.common import run_dir, atomic_json,camp,read_json


class AnalysisTests(unittest.TestCase):
    def panel(self, server='s1',extended=False):
        cases = tuple(c for c in CASES if c.server_id == server and (extended or c.case_id!='C17'))
        observations = {}
        for case in cases:
            value = dict(HQNR=.96, ERGAS=1., E_val=1.1, D_lambda=.02, D_s=.02, SAM=2., SCC=.9)
            observations[case.run_id] = dict(run_id=case.run_id, case_id=case.case_id, sensor=case.sensor,
                sweep=case.sweep,
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
        self.assertTrue(report['legacy_core17_complete']);self.assertFalse(report['extended18_complete'])

    def test_extended18_gf2_and_legacy_thresholds_are_separate(self):
        cases,rows=self.panel('s3',extended=True)
        report=assemble_wave(cases,rows)
        legacy_cases,legacy_rows=self.panel('s3')
        legacy=assemble_wave(legacy_cases,legacy_rows)
        self.assertEqual(report['student_runs'],90);self.assertEqual(len(report['relations']),19)
        self.assertTrue(report['extended18_complete']);self.assertTrue(report['legacy_core17_complete'])
        self.assertEqual(report['thresholds'],legacy['thresholds'])
        self.assertTrue(report['relations']['A17_SCRATCH']['diagnostic_only'])
        with self.assertRaises(ValueError):cumulative_balanced_reports([legacy,report])

    def test_a17_repair_anchor_changes_only_a17_not_legacy_threshold_or_c03_table(self):
        cases,rows=self.panel('s3',extended=True)
        original=assemble_wave(cases,rows);anchors={}
        for case in cases:
            if case.case_id=='C03':
                anchor=copy.deepcopy(rows[case.run_id]);anchor['run_id']+='PAIR_REPAIR'
                anchor['VAL']['HQNR']+=.002;anchor['EXACT50K']['HQNR']+=.002
                anchor.update(init_U_sha256='repair_U',sampler_sha256='repair_'+case.sweep,rng_roles={'stream':'repair'})
                anchors[case.sweep]=anchor
            if case.case_id=='C17':rows[case.run_id].update(init_U_sha256='repair_U',sampler_sha256='repair_'+case.sweep,rng_roles={'stream':'repair'})
        repaired=assemble_wave(cases,rows,a17_anchors=anchors)
        self.assertEqual(repaired['thresholds'],original['thresholds'])
        self.assertEqual(repaired['case_statistics']['C03'],original['case_statistics']['C03'])
        self.assertEqual(repaired['relations']['A17']['classification'],'H_GAIN_SAFE')
        self.assertEqual(repaired['relations']['A17_SCRATCH']['reason'],'PAIRING_NOT_VERIFIED')
        self.assertEqual(repaired['student_runs'],90);self.assertFalse(repaired['repair_anchors_counted_as_components'])
        self.assertEqual(len(repaired['a17_anchor_rows']),5)

    def test_source_hash_difference_needs_explicit_verified_bridge(self):
        cases,rows=self.panel('s3',extended=True)
        c17=next(c for c in cases if c.case_id=='C17');row=rows[c17.run_id]
        row['source_identity']={'content_sha256':'historical'}
        row['analysis_source_identity']={'content_sha256':'source'}
        with self.assertRaises(ValueError):assemble_wave(cases,rows)
        row['source_bridge_sha256']='verified-receipt-fixture'
        self.assertTrue(assemble_wave(cases,rows)['extended18_complete'])

    def test_five_completed_teachers_do_not_count_as_c17_coverage(self):
        cases,rows=self.panel('s3',extended=True)
        c17=next(c for c in cases if c.case_id=='C17')
        del rows[c17.run_id]
        with self.assertRaises(ValueError):assemble_wave(cases,rows)

    def test_legacy_report_collision_uses_distinct_verified_extension_directory(self):
        cases,rows=self.panel('s1')
        with TemporaryDirectory() as root:
            original=camp(root,'s1')/'reports/r000/BOOT5/BOOT5/complete_panel_metrics.json'
            atomic_json(original,{'immutable_historical_report':True});before=original.read_bytes()
            with patch('ablr2.analysis.case_report',side_effect=lambda case,root:rows[case.run_id]), \
                    patch('ablr2.analysis.write_panel_csv'),patch('ablr2.plots.render_wave'):
                report=analyze_wave(cases,root=root)
            self.assertEqual(original.read_bytes(),before)
            self.assertIn('/verified_extension/',report['report_path'])
            self.assertTrue(Path(report['report_path']).is_file())

    def test_supplement_waits_for_all_five_c17_then_adds_report_without_state_mutation(self):
        cases,rows=self.panel('s1',extended=True)
        state=dict(server='s1',cycle=7,stages=[dict(stage_id='BOOT5',kind='BOOT5',complete=True,run_ids=[c.run_id for c in cases if c.case_id!='C17'])],
            runs={c.run_id:{'complete':c.case_id!='C17'} for c in cases})
        with TemporaryDirectory() as root:
            folder=camp(root,'s1');atomic_json(folder/'thresholds_v1.json',assemble_wave(cases,rows)['thresholds'])
            path=folder/'reports/r000/BOOT5/BOOT5/extended18/complete_panel_metrics.json'
            report=dict(assemble_wave(cases,rows),report_path=str(path));atomic_json(path,report)
            with patch('ablr2.extension.stage_cases',return_value=cases),patch('ablr2.common.source_identity',return_value={'content_sha256':'source'}), \
                    patch('ablr2.analysis.analyze_wave',return_value=report) as analyze:
                before=copy.deepcopy(state)
                result=refresh_extended_reports(root,'s1',state)
                self.assertEqual(state,before);self.assertEqual(result['BOOT5']['status'],'WAIT_C17_COVERAGE');analyze.assert_not_called()
                for c in cases:
                    if c.case_id=='C17':state['runs'][c.run_id]['complete']=True
                before=copy.deepcopy(state)
                completed=refresh_extended_reports(root,'s1',state)
                self.assertEqual(state,before);self.assertTrue(completed['BOOT5']['extended18_complete']);analyze.assert_called_once()
                refresh_extended_reports(root,'s1',state);analyze.assert_called_once()
                self.assertEqual(read_json(folder/'reports/extended18_status/BOOT5.json')['status'],'COMPLETE')

    def test_supplement_cannot_count_failed_pairing_as_complete_or_change_control(self):
        cases,_=self.panel('s1',extended=True)
        state=dict(server='s1',active_stage='NEXT',stages=[dict(stage_id='BOOT5',kind='BOOT5',complete=True)],
                   runs={c.run_id:{'complete':True} for c in cases})
        with TemporaryDirectory() as root:
            atomic_json(camp(root,'s1')/'thresholds_v1.json',{'sensor':'WV3'})
            with patch('ablr2.extension.stage_cases',return_value=cases),patch('ablr2.common.source_identity',return_value={}), \
                    patch('ablr2.analysis.analyze_wave',side_effect=ValueError('missing matched initial U proof')):
                result=refresh_extended_reports(root,'s1',state)
            self.assertEqual(result['BOOT5']['status'],'PAIRING_NOT_VERIFIED')
            self.assertFalse(result['BOOT5']['extended18_complete']);self.assertEqual(state['active_stage'],'NEXT')

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
