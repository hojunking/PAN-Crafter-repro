from pathlib import Path
import tempfile
import unittest

from panda_rb.common import ROOT, atomic_json
from panda_rb.evaluation import seal
from panda_rb.plan import PLAN_DIR, training_runs, run_dir, shifts
from panda_rb.reporting import summarize, stress_contrasts


class Summaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / PLAN_DIR).parent.mkdir(parents=True)
        (self.root / PLAN_DIR).symlink_to(ROOT / PLAN_DIR, target_is_directory=True)

    def tearDown(self): self.temp.cleanup()

    def put_native(self, case, delta=0, binding='same-F1'):
        selected = dict(checkpoint_sha256='selected-' + str(case['seed']), alias_of=None,
            rr=dict(ergas=2+delta, scc=.9, q8=.91), fr=dict(hqnr=.95),
            supplemental_rr=dict(rmse=20, cc=.99), supplemental_fr=dict(jqm=.94))
        report = seal(dict(complete=True, file_hashes={}, context=dict(case=case,
            binding_common_sha256=binding, data_content_identity={'rr': 'same20'},
            source_identity={'content_sha256': 'same-code', 'torch': 'same-runtime'}),
            selections={'EXACT_50000': selected,
                        'RR_VAL_ERGAS_MIN': dict(selected, alias_of='EXACT_50000')}))
        atomic_json(run_dir(case['run_id'], self.root) / 'native/metrics.json', report)

    def test_alias_not_independent_q8_retained_and_server_paired(self):
        for row in training_runs('s1'):
            self.put_native(row, delta=0 if row['case_id']=='QFULL' else -.2)
        result = summarize(self.root, ['s1'])
        self.assertEqual(result['completed_students'], 8)
        means = [r for r in result['native_summary'] if r['case_id']=='QFULL' and
                 r['selection_id']=='EXACT_50000' and r['server']=='ALL'][0]
        self.assertEqual(means['metrics']['q8']['n'], 2)
        paired = [r for r in result['paired_differences'] if r['case_id']=='QMEAN' and
                  r['selection_id']=='EXACT_50000' and r['server']=='s1'][0]
        self.assertEqual(paired['metrics']['ergas']['n'], 2)
        self.assertEqual(paired['metrics']['ergas']['improved'], 2)
        self.assertEqual([r['seed'] for r in paired['per_seed']], [9281101, 9281102])
        self.assertAlmostEqual(paired['metrics']['ergas']['mean'], -.2)
        self.assertFalse(result['complete'])  # No stress curves in this fixture.

    def test_different_common_teacher_cannot_be_pooled(self):
        rows = training_runs('s1')
        self.put_native(rows[0]); self.put_native(rows[1], binding='wrong-F3')
        with self.assertRaisesRegex(ValueError, 'Cannot pool'): summarize(self.root, ['s1'])

    def curve(self, seed, mode, value):
        return dict(case_id='QFULL', server='s1', repeat=seed-9281100, seed=seed, mode=mode,
            points=[dict(p, ergas=value, psnr=value, sam=value, edge_error_dn=value,
                relative_response_l1_sum=value, coverage_all_pan_paths=1.) for p in shifts()])

    def test_radius_averages_within_student_not_scene_direction_pseudorepeats(self):
        curves = [self.curve(seed, mode, (1 if seed == 9281101 else 3) + (0 if mode == 'A_ON' else 2))
                  for seed in (9281101, 9281102) for mode in ('A_ON', 'A_ZERO_INFERENCE_ONLY')]
        paired, radii = stress_contrasts(curves, ['s1'])
        point = paired[0]['points'][1]['metrics']['ergas']
        self.assertEqual(point['n'], 2)
        self.assertEqual(point['mean'], -2)
        radius = next(r for r in radii if r['server']=='ALL' and r['mode']=='A_ON' and r['radius_hr']==4.)
        self.assertEqual(radius['metrics']['ergas']['n'], 2)
        self.assertEqual(radius['metrics']['ergas']['mean'], 2)
        self.assertAlmostEqual(radius['metrics']['ergas']['sample_std'], 2**.5)

    def test_missing_direction_value_records_failed_student_not_partial_mean(self):
        curves = [self.curve(9281101, 'A_ON', 1), self.curve(9281102, 'A_ON', 3)]
        curves[0]['points'][1]['ergas'] = None
        _, radii = stress_contrasts(curves, ['s1'])
        radius = next(r for r in radii if r['server']=='ALL' and r['mode']=='A_ON' and r['radius_hr']==.25)
        self.assertEqual(radius['metrics']['ergas']['n'], 1)
        self.assertEqual(radius['metrics']['ergas']['failed_students'], 1)
        self.assertEqual(radius['metrics']['ergas']['mean'], 3)


if __name__ == '__main__': unittest.main()
