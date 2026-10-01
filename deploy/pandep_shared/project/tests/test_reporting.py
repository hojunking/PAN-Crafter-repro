import copy
import json
import tempfile
import unittest
from pathlib import Path

from pan_shared.common import atomic_json, canonical_sha
from pan_shared.registry import build_registry
from pan_shared.reporting import summarize_stage, write_stage_report
from pan_shared.sheets import result_id


def seal(row):
    row['payload_sha256'] = canonical_sha({k: v for k, v in row.items() if k != 'payload_sha256'})
    return row


def observations(secondary=False):
    rows = []
    for run in build_registry()['runs']:
        for sensor in run['sensors']:
            scope = 'EXACT'
            if secondary:
                scope = ('BEST_JOINT_VAL' if run['mode'] == 'SHARED' else 'BEST_SENSOR_VAL')+f"_TO_B{3 if run['mode']=='SHARED' else 1:04d}"
            endpoint = run['exposure_stage_updates']
            delta = {'C00': 0, 'C11': .01, 'C01': -.01, 'S01': -.02, 'S02': -.02, 'S03': -.02}[run['case_id']]
            row = dict(run, sensor=sensor, train_sensors=run['sensors'],
                completed_step=endpoint, selected_step=endpoint-10000 if secondary else endpoint,
                selector_scope=scope, checkpoint_sha256=canonical_sha([run['run_id'], scope]), evaluator_sha256='a'*64,
                recorded_at='2026-10-01T00:00:00+00:00', protocol_id='PANDEP_NATIVE_RR20_FR20_v1',
                rr={'n_scenes': 20, 'ergas': 2.-delta, 'scc': .98+delta},
                fr={'n_scenes': 20, 'hqnr': .9+delta+run['repeat']*.001},
                notes={'identity': {k: 'a'*64 for k in ('source_sha256','environment_sha256','data_sha256','protocol_sha256')},
                       'selected_through_global_step': endpoint, 'candidate_count': 15,
                       'selected_exposures': {sensor: {'committed_samples': 2400000, 'effective_epochs': 100.}}},
                cost={'wall_hours': 1., 'train_hours': .8})
            rows.append(seal(row))
    return rows


class ReportingTests(unittest.TestCase):
    def test_exact_n_three_sample_sd_and_correct_pairs(self):
        report = summarize_stage(observations(), 1, build_registry())
        base = next(g for g in report['primary'] if g['case_id']=='C00' and g['sensor']=='WV3')
        self.assertEqual(base['metrics']['hqnr']['n'], 3)
        self.assertAlmostEqual(base['metrics']['hqnr']['sample_sd'], .001)
        pairs = {(p['case_id'],p['baseline_case_id'],p['sensor']): p for p in report['primary_paired_differences']}
        self.assertAlmostEqual(pairs['C00','S01','WV3']['metrics']['hqnr']['mean'], .02)
        self.assertAlmostEqual(pairs['C11','C00','GF2']['metrics']['hqnr']['mean'], .01)
        self.assertAlmostEqual(pairs['C01','C00','QB']['metrics']['hqnr']['mean'], -.01)
        self.assertEqual(len(report['run_costs']), 18)

    def test_secondary_not_pooled_with_exact_or_aliases(self):
        original = observations(); rows = original+observations(True)+[copy.deepcopy(original[0])]
        report = summarize_stage(rows, 1, build_registry())
        self.assertTrue(all(g['n_students']==3 for g in report['primary']))
        self.assertTrue(all(g['n_students']==3 for g in report['secondary_validation_selected']))
        self.assertTrue(all(g['metrics']['hqnr']['n']==3 for g in report['secondary_validation_selected']))

    def test_missing_repeat_remains_explicit_and_sd_single_blank(self):
        rows = [r for r in observations() if r['repeat']==1]
        report = summarize_stage(rows, 1, build_registry())
        self.assertTrue(all(g['expected_n']==3 and g['n_students']==1 for g in report['primary']))
        self.assertTrue(all(len(g['missing'])==2 for g in report['primary']))
        self.assertTrue(all(g['metrics']['hqnr']['sample_sd'] is None for g in report['primary']))

    def test_changed_evaluator_is_separate_cohort_not_paired(self):
        rows = observations()
        changed = next(r for r in rows if r['case_id']=='C11' and r['sensor']=='WV3' and r['repeat']==1)
        changed['evaluator_sha256'] = 'b'*64; seal(changed)
        report = summarize_stage(rows, 1, build_registry())
        split = [g for g in report['primary'] if g['case_id']=='C11' and g['sensor']=='WV3']
        self.assertEqual(sorted(g['n_students'] for g in split), [1,2])
        pair = next(p for p in report['primary_paired_differences'] if p['case_id']=='C11' and p['sensor']=='WV3')
        self.assertEqual(pair['n_pairs'], 2)

    def test_auxiliary_75k_is_not_primary(self):
        rows = observations(); half = copy.deepcopy(next(r for r in rows if r['case_id']=='C01'))
        half.update(selected_step=75000, completed_step=75000, checkpoint_sha256='c'*64)
        seal(half)
        report = summarize_stage(rows+[half], 1, build_registry())
        self.assertEqual(sum(a['status']=='MEASURED' for a in report['auxiliary_half75k_full150k']), 1)
        self.assertTrue(all(g['n_students']==3 for g in report['primary']))

    def test_reports_written_idempotent_and_user_text_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for row in observations(): atomic_json(work/'observations'/(result_id(row)+'.json'), row)
            result = write_stage_report(work, 1, build_registry())
            again = write_stage_report(work, 1, build_registry())
            self.assertEqual(result, again)
            markdown = Path(result['markdown'])
            self.assertIn('Secondary', markdown.read_text())
            markdown.write_text(markdown.read_text()+'manual edit\n')
            with self.assertRaisesRegex(ValueError, 'user edits'):
                write_stage_report(work, 1, build_registry())

    def test_unsealed_or_wrong_config_cannot_enter_mean(self):
        rows = observations(); rows[0]['rr']['ergas'] = 99.
        with self.assertRaisesRegex(ValueError, 'seal'):
            summarize_stage(rows, 1, build_registry())
        seal(rows[0]); rows[0]['config_sha256'] = 'c'*64; seal(rows[0])
        with self.assertRaisesRegex(ValueError, 'config_sha256'):
            summarize_stage(rows, 1, build_registry())

    def test_shared_per_sensor_best_step_mixing_refused(self):
        rows = observations(True); rows[0]['checkpoint_sha256'] = 'c'*64; seal(rows[0])
        with self.assertRaisesRegex(ValueError, 'different checkpoint bytes'):
            summarize_stage(rows, 1, build_registry())


if __name__ == '__main__': unittest.main()
