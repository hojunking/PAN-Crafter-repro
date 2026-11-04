import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_wv3_main_rgb import plan, publication as original
from . import publication as reporting


class ReportingTest(unittest.TestCase):
    def evidence(self,h=.95849):
        cfg=plan.resolve('s1','M0_OURS',1)
        e={'step':1000,'rr':{'metrics':{'ERGAS':2.0399}},
           'fr':{'metrics':{'HQNR':h}},'evidence_path':'test.json','evidence_sha256':'e',
           'model_sha256':'m','checkpoint':{'file_sha256':'f'},
           'recorded_at':'2026-11-04T00:00:00+00:00','jqm_variant':'test',
           'inference_parameters':2890794,'elapsed_seconds':10,
           'donor_selection':{'donor_run_id':cfg['donor_run_id']}}
        row,note=original.make_row(cfg,'RUNNING',e,'BEST_HQNR_FALLBACK','REPRESENTATIVE')
        value={'rows':[row],'notes':[note],'proof':{(cfg['run_id'],1000):e},
               'configs':{cfg['run_id']:cfg},'destination':original.destination()}
        return cfg,e,value

    def test_rounding_keeps_raw_selection_and_lr_and_cost(self):
        cfg,e,value=self.evidence()
        fields={'Params(M)':2.890794,'Infer(ms)':9.9872,'Mem(MB)':292.654,
                'FLOPs(G)':143.581327488,'Cost_scope':'test'}
        costs={'measurement':{'sheet_cost':fields,'environment':{},
            'observations':{k:{'sheet_fields':fields} for k in ('rr','fr')}},
            'profile_sha256':'p','checkpoint':{'file_sha256':'f'}}
        times={'Train(h)':1.23456,'Eval(h)':.06543,'Wall(h)':1.34567,'scope':'selected'}
        with patch.object(original,'project',return_value=value),patch.object(reporting.cost,'load',return_value=costs),patch.object(reporting,'timing',return_value=times):
            out=reporting.project(Path('/tmp'),'s1')
        row=out['rows'][0];idx=original.INDEX
        self.assertEqual(row[idx['HQNR↑']],.958)
        self.assertEqual(row[idx['ERGAS↓']],2.040)
        self.assertEqual(row[idx['U_peak_lr']],1e-4)
        self.assertEqual(row[idx['FLOPs(G)']],143.581)
        self.assertEqual(row[idx['Train(h)']],1.235)
        note=json.loads(row[idx['Original description / notes']])
        self.assertFalse(note['publication']['joint_pass'])
        self.assertEqual(note['reporting']['raw_metrics']['HQNR↑'],.95849)
        reporting.verify_owned(row,out,'s1')
        tampered=list(row);tampered[idx['HQNR↑']]=.959
        with self.assertRaisesRegex(ValueError,'Rounded metric'):
            reporting.verify_owned(tampered,out,'s1')

    def test_original_rows_migrate_but_bad_metric_rejected(self):
        cfg,e,value=self.evidence()
        reporting.verify_owned(value['rows'][0],value,'s1')
        bad=copy.deepcopy(value['rows'][0]);bad[original.INDEX['ERGAS↓']]=2.030
        with self.assertRaises(ValueError):reporting.verify_owned(bad,value,'s1')

    def test_cumulative_timing_and_selected_step_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'train.jsonl'
            rows=[{'step':i,'seconds':.25,'at':'2026-11-04T00:00:00+00:00'} for i in range(1,2002)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            records=[{'step':1000,'elapsed_seconds':10,'recorded_at':'2026-11-04T00:10:00+00:00'},
                     {'step':2000,'elapsed_seconds':12,'recorded_at':'2026-11-04T00:20:00+00:00'}]
            one=reporting.timing(d,1000,records);two=reporting.timing(d,2000,records)
            self.assertAlmostEqual(one['Train(h)'],250/3600)
            self.assertAlmostEqual(two['Train(h)'],500/3600)
            self.assertAlmostEqual(two['Eval(h)'],22/3600)
            self.assertAlmostEqual(two['Wall(h)'],1200.25/3600)
            with self.assertRaises(ValueError):reporting.timing(d,2000,records[:1])
            rows[999]['step']=998
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):reporting.timing(d,1000,records)

    def test_format_only_owns_result_cells(self):
        requests=reporting.format_requests([7,24])
        self.assertEqual(requests[0]['repeatCell']['range']['startRowIndex'],6)
        self.assertEqual(requests[1]['repeatCell']['range']['endRowIndex'],24)
        self.assertEqual(requests[0]['repeatCell']['fields'],'userEnteredFormat.numberFormat')
        self.assertEqual(requests[0]['repeatCell']['range']['endColumnIndex'],25)


if __name__=='__main__':unittest.main()
