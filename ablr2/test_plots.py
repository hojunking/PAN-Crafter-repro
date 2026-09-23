import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ablr2.plots import plot_data, render_wave
from ablr2.plan import LEGACY_MAIN_CASES as MAIN_CASES


def complete_report():
    report = dict(schema='ABLR2_BALANCED_PANEL_v1', complete=True, sweep_count=5, student_runs=85,
                  sensor='WV3', source_identity={'content_sha256': 'source'}, data_sha256='data',
                  recipe_id='R00', recipe_revision='r000', phase='BOOT5', wave='BOOT5', panelrows=[])
    for index, case in enumerate(MAIN_CASES):
        for k in range(1, 6):
            report['panelrows'].append(dict(case_id=case, sweep=f'P{k:02}', student_seed=791000+k,
                run_id=f'{case}_P{k:02}', sensor='WV3', source_identity=report['source_identity'], data_sha256='data',
                VAL=dict(HQNR=.95 + .0001 * index + k * .00001, ERGAS=3.0 + .01 * index + k * .001)))
    return report


class PlotTests(unittest.TestCase):
    def test_extended18_has_separate_prior_only_plot_and_all_90_points(self):
        report=complete_report();report.update(schema='ABLR2_BALANCED_PANEL_v2',student_runs=90,extended18_complete=True)
        for k in range(1,6):
            row=copy.deepcopy(next(r for r in report['panelrows'] if r['case_id']=='C03' and r['sweep']==f'P{k:02}'))
            row.update(case_id='C17',run_id=f'C17_P{k:02}')
            report['panelrows'].append(row)
        data=plot_data(report)
        self.assertEqual(len(data['panels']),6)
        self.assertEqual([r['case_id'] for r in data['panels']['alignment_prior_hqnr']['series']],['C02','C17','C03'])
        self.assertEqual(len(report['panelrows']),90)

    def test_fixed_orders_all_five_points_and_negative_effects(self):
        report = complete_report()
        data = plot_data(report)
        self.assertEqual(set(data['panels']), {'ladder_hqnr', 'ladder_ergas', 'removal_hqnr', 'removal_ergas'})
        self.assertEqual([r['case_id'] for r in data['panels']['ladder_hqnr']['series']], list(MAIN_CASES[:8]))
        self.assertEqual([r['case_id'] for r in data['panels']['removal_ergas']['series']], list(MAIN_CASES[8:]))
        self.assertTrue(all(len(row['values']) == 5 for panel in data['panels'].values() for row in panel['series']))
        self.assertTrue(all(v < 0 for v in data['panels']['removal_hqnr']['series'][0]['values']))
        self.assertAlmostEqual(data['panels']['removal_ergas']['series'][0]['mean'], -.01)

    def test_partial_duplicate_mixed_seed_or_nonfinite_plot_refused(self):
        for mutation in ('partial', 'duplicate', 'seed', 'nonfinite'):
            report = complete_report()
            if mutation == 'partial': report['panelrows'].pop()
            if mutation == 'duplicate': report['panelrows'][-1] = report['panelrows'][0]
            if mutation == 'seed': report['panelrows'][0]['student_seed'] = 42
            if mutation == 'nonfinite': report['panelrows'][0]['VAL']['HQNR'] = float('nan')
            with self.assertRaises(ValueError): plot_data(report)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib is not installed')
    def test_agg_render_full_precision_sidecar_and_idempotent_manifest(self):
        report = complete_report()
        with TemporaryDirectory() as folder:
            first = render_wave(report, folder)
            self.assertEqual(len(first['figures']), 4)
            self.assertEqual(first['backend'], 'Agg')
            self.assertTrue(all((Path(folder) / row['file']).stat().st_size > 1000 for row in first['figures'].values()))
            self.assertEqual(render_wave(report, folder), first)
            report['panelrows'][0]['VAL']['HQNR'] -= .1
            with self.assertRaises(ValueError): render_wave(report, folder)


if __name__ == '__main__': unittest.main()
