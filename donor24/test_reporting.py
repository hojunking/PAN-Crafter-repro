from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from donor30.test_reporting import evaluation
from . import plan, reporting as R, publication as P


def config(server='s1'):
    return plan.resolve_case(plan.default_config(server),'A0')


def persist(root,cfg,step=1000,h=.959,e=2.2):
    ev=evaluation(cfg,step,h,e)
    path=Path(root)/'runs'/cfg['run_id']/'attempt_01'
    path.mkdir(parents=True,exist_ok=True)
    R.atomic_json(path/'config.json',cfg)
    R.atomic_json(path/f'evaluation_{step:07d}.json',ev)
    return R.emit_evaluation(root,cfg,ev,timestamp='2026-10-16T00:00:00+00:00')


class ReportingTests(unittest.TestCase):
    def test_retry_after_committed_append_does_not_duplicate(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            c=config();persist(root,c);p=P.project_campaign(root,'s1')
            snap={'rows':[],'notes':[]}
            calls=[]
            def request(method,url,**kwargs):
                calls.append(kwargs['json'])
                for item in kwargs['json']['requests']:
                    for row in item.get('appendCells',{}).get('rows',[]):
                        values=row['values']
                        snap['rows'].append([next(iter(v['userEnteredValue'].values())) for v in values])
                        snap['notes'].append(values[0]['note'])
                raise TimeoutError('server committed, response lost')
            adapter=SimpleNamespace(BASE='https://unused/',snapshot=lambda d:deepcopy(snap),_request=request)
            with patch('sheet_publish_v2.transport._validate_snapshot'):
                with self.assertRaises(TimeoutError):P.reconcile_rows(adapter,p,root,'s1',apply=True)
                result=P.reconcile_rows(adapter,p,root,'s1',apply=True)
            self.assertTrue(result['readback_verified']);self.assertEqual(len(calls),1)
            self.assertEqual(len(snap['rows']),1)

    def test_fallback_to_pass_clears_only_own_obsolete_row(self):
        with tempfile.TemporaryDirectory() as root:
            c=config();persist(root,c);before=P.project_campaign(root,'s1')
            persist(root,c,4000,.961,2.04);after=P.project_campaign(root,'s1')
            foreign=deepcopy(before['rows'][0]);foreign[R.INDEX['Server']]='s3'
            snap={'rows':[foreign,before['rows'][0]],'notes':['foreign',before['notes'][0]]}
            req=P.row_requests(snap,after,'s1')
            self.assertEqual(len(req),2)
            self.assertEqual(req[0]['updateCells']['range']['startRowIndex'],6)
            self.assertEqual(req[0]['updateCells']['rows'][0]['values'][0]['userEnteredValue'],{'stringValue':''})
            self.assertIn('appendCells',req[1])

    def test_s3_wv3_roundtrip_short_run_and_shared_destination(self):
        for server in ('s1','s3'):
            cfg=config(server);r=R.make_record(cfg,evaluation(cfg));R.validate_record(r)
            self.assertEqual(r['row'][R.INDEX['Experiment']],cfg['run_display'])
            self.assertEqual(r['destination']['sheet_title'],'WV3-ablations')
            self.assertEqual(r['row'][R.INDEX['Server']],server)
        c=config('s3');c['dataset']='GF2'
        with self.assertRaises(ValueError):R.make_record(c,evaluation(c))

    def test_display_policy_still_pass_all_or_best_hqnr(self):
        with tempfile.TemporaryDirectory() as root:
            c=config('s3')
            persist(root,c,1000,.9591,2.1);persist(root,c,2000,.961,2.4)
            p=P.project_campaign(root,'s3')
            self.assertFalse(p['conflicts']);self.assertEqual(p['selected'][0]['step'],2000)
            persist(root,c,4000,.960,2.045);persist(root,c,5000,.961,2.04)
            p=P.project_campaign(root,'s3')
            self.assertFalse(p['conflicts']);self.assertEqual([c['step'] for c in p['selected']],[4000,5000])

    def test_missing_cost_and_bad_config_block_publication(self):
        with tempfile.TemporaryDirectory() as root:
            c=config();persist(root,c)
            with patch('experiment_cost.publication.lookup_profile',return_value=None):
                self.assertTrue(P.project(root,'s1')['conflicts'])
                self.assertFalse(P.project(root,'s1',require_cost=False)['conflicts'])
            p=next(Path(root).glob('runs/*/attempt_01/config.json'))
            bad=json.loads(p.read_text());bad['lr_A']=9;R.atomic_json(p,bad)
            self.assertTrue(P.project_campaign(root,'s1')['conflicts'])

    def test_row_scope_preserves_foreign_and_old_campaign(self):
        with tempfile.TemporaryDirectory() as root:
            c=config();persist(root,c)
            p=P.project_campaign(root,'s1')
            own=deepcopy(p['rows'][0]);other=deepcopy(own);old=deepcopy(own)
            other[R.INDEX['Server']]='s3';old[R.INDEX['Campaign_ID']]='OLD'
            own[R.INDEX['HQNR↑']]=.9
            snapshot={'rows':[other,old,own],'notes':['foreign','history',p['notes'][0]]}
            req=P.row_requests(snapshot,p,'s1')
            self.assertEqual(len(req),1)
            self.assertEqual(req[0]['updateCells']['range']['startRowIndex'],7)
            req=P.row_requests({'rows':[other,old],'notes':['foreign','history']},p,'s1')
            self.assertEqual(list(req[0]),['appendCells'])
            # A missing local source must never authorize removal of an own row.
            changed=deepcopy(p);changed['owned_run_keys']=[]
            with self.assertRaisesRegex(ValueError,'absent from local'):
                P.row_requests(snapshot,changed,'s1')

    def test_duplicate_remote_ids_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            c=config();persist(root,c);p=P.project_campaign(root,'s1')
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                P.row_requests({'rows':p['rows']*2,'notes':p['notes']*2},p,'s1')
            ev=next(Path(root).glob('runs/*/attempt_01/evaluation_*.json'))
            d=json.loads(ev.read_text());d['rr']['per_scene'].pop();R.atomic_json(ev,d)
            self.assertTrue(P.project_campaign(root,'s1')['conflicts'])


if __name__=='__main__':unittest.main()
