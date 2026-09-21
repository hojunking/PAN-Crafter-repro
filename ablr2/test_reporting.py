"""Presentation tables retain negatives/failures and never merge rechecks."""
import csv
import tempfile
import unittest
from pathlib import Path

from ablr2.common import atomic_json,append_event,camp,read_json
from ablr2.reporting import rebuild


class ReportingTests(unittest.TestCase):
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
                                                  exploratory_best=1,verification_results=1))
            with (folder/'reports/all_attempts.csv').open() as stream:
                self.assertEqual(next(csv.DictReader(stream))['reason'],'DIVERGED')
            with (folder/'reports/verification_results.csv').open() as stream:
                self.assertEqual(next(csv.DictReader(stream))['verification_status'],'VERIFY_NEGATIVE')
            self.assertFalse(camp(root,'s2').exists())


if __name__=='__main__': unittest.main()
