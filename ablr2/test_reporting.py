"""Presentation tables retain negatives/failures and never merge rechecks."""
import csv
import tempfile
import unittest
from pathlib import Path

from ablr2.common import atomic_json,append_event,camp,read_json
from ablr2.reporting import rebuild


class ReportingTests(unittest.TestCase):
    def test_supplemental_extension_updates_actual_coverage_without_duplicate_legacy_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=camp(root,'s1')
            rows=[dict(run_id=f'{sweep}_{case}',case_id=f'C{case:02}',sweep=f'P{sweep:02}',
                       recipe_revision='r000',VAL={'HQNR':.95},RAW_MAX={'HQNR':.951})
                  for sweep in range(1,6) for case in range(18)]
            old=folder/'legacy95.json';extended=folder/'extended100.json'
            atomic_json(old,dict(coverage='LEGACY_CORE17',panelrows=[r for r in rows if r['case_id']!='C17']))
            atomic_json(extended,dict(coverage='EXTENDED18',panelrows=rows))
            atomic_json(folder/'state.json',dict(stages=[dict(kind='BOOT5',stage_id='BOOT5',
                report_path=str(old),extended_report_path=str(extended),extended18_complete=True)]))
            original=old.read_bytes();state=(folder/'state.json').read_bytes()
            result=rebuild(root,'s1')
            self.assertEqual(result['counts']['complete_panel_metrics'],90)
            self.assertEqual(old.read_bytes(),original)
            self.assertEqual((folder/'state.json').read_bytes(),state)
            with (folder/'reports/complete_panel_metrics.csv').open() as stream:
                actual=list(csv.DictReader(stream))
            self.assertEqual(len({r['run_id'] for r in actual}),90)
            self.assertTrue(all(r['original_panel_report']==str(old) and
                r['source_panel_report']==str(extended) and r['supplemental_extension']=='True' for r in actual))
            with (folder/'reports/component_coverage.csv').open() as stream:
                coverage=list(csv.DictReader(stream))
            self.assertEqual(len(coverage),5)
            self.assertTrue(all(r['extended18_complete']=='True' and r['actual_component_count']=='18'
                                and r['c17_status']=='COMPLETE' for r in coverage))

    def test_separate_tables_preserve_failed_attempt_and_negative_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=camp(root,'s1')
            a=dict(run_id='negative',case_id='C07',recipe_revision='r000',VAL={'HQNR':.8},RAW_MAX={'HQNR':.81})
            b=dict(a,run_id='second',VAL={'HQNR':.9},RAW_MAX={'HQNR':.91})
            full=folder/'full.json';recheck=folder/'recheck.json';verify=folder/'verify.json'
            atomic_json(full,dict(panelrows=[a,b]))
            atomic_json(recheck,dict(outcome='REPEATED_NEGATIVE_DEV',records={'run':a}))
            atomic_json(verify,dict(panelrows=[a]))
            atomic_json(folder/'state.json',dict(stages=[
                dict(kind='BOOT5',stage_id='BOOT5',report_path=str(full)),
                dict(kind='RECHECK5',stage_id='R1',relation_id='E00',report_path=str(recheck)),
                dict(kind='VERIFY5',stage_id='V1',report_path=str(verify),verification_status='VERIFY_NEGATIVE')]))
            append_event(folder/'all_attempts.jsonl','ACTION_ENDED',exit_code=1,reason='DIVERGED')
            result=rebuild(root,'s1')
            self.assertEqual(result['counts'],dict(all_attempts=1,complete_panel_metrics=2,targeted_rechecks=1,
                                                  exploratory_best=1,verification_results=1,component_coverage=2))
            with (folder/'reports/all_attempts.csv').open() as stream:
                self.assertEqual(next(csv.DictReader(stream))['reason'],'DIVERGED')
            with (folder/'reports/verification_results.csv').open() as stream:
                self.assertEqual(next(csv.DictReader(stream))['verification_status'],'VERIFY_NEGATIVE')
            self.assertFalse(camp(root,'s2').exists())


if __name__=='__main__': unittest.main()
