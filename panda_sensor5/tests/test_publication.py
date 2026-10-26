"""Sensor-specific same-state publication and actual cost graph contracts, offline."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote

from panda_band30.tests.test_reporting import evaluation as original_evaluation
from panda_mature30.tests.test_reporting import reused_receipt
from panda_sensor5 import best_publication as B, cost as C, plan, publication as P, reporting as R
from panda_sensor5.best_publication import transport as T
from panda_sensor5.best_publication.restart import matches_writer
from panda_sensor5.spec import geometry_manifest


def persist(root, case=None, *, server='s1', step=1000, h=None, e=None):
    sensor = R.DATASETS[server]
    case = case or sensor + '_BASE'
    hmin, emax = R.thresholds(server)
    h, e = hmin if h is None else h, emax if e is None else e
    cfg = plan.resolve_case(server, case, 'R1')
    ev = original_evaluation(cfg, step, h, e)
    key = cfg['policy_id'].lower()
    ev['mature30_deployment'] = {'schema':'PANDA_NEXT30_DEPLOYMENT_V1',
        'policy_id':cfg['policy_id'], 'completed_update':step, 'geometry_key':key,
        'geometry':geometry_manifest(key), 'student_sampling':deepcopy(cfg['student_sampling']),
        'student_input':deepcopy(cfg['student_input']), 'student_warp':deepcopy(cfg['student_warp'])}
    for split in ('rr', 'fr'):
        for scene in ev[split]['per_scene']:
            scene['correction_dy_dx'] = [.3,-.4]
            scene['pan_applied_dy_dx'] = [.3,-.4]
    directory = Path(root)/'runs'/cfg['run_id']/'attempt_01'
    directory.mkdir(parents=True, exist_ok=True)
    R.atomic_json(directory/'config.json', cfg)
    R.atomic_json(directory/f'evaluation_{step:07d}.json', ev)
    return R.emit_evaluation(root, cfg, ev, timestamp='2026-10-26T00:00:00+00:00')


def snapshot(value, rows=None, notes=None):
    dest = value['destination']
    return {'metadata': {'spreadsheetId': dest['spreadsheet_id'], 'sheets':[{
        'properties': {'sheetId': dest['sheet_id'], 'title':dest['sheet_title'],
                       'gridProperties':{'columnCount':83, 'rowCount':10000}}}]},
        'header':list(R.HEADERS), 'grid_row_count':10000, 'rows':rows or [],
        'notes':notes or [], 'cells':[]}


class SensorPublicationTests(unittest.TestCase):
    def test_assigned_lanes_and_new_inclusive_targets(self):
        self.assertEqual(R.thresholds('s1'), (.9315,3.5))
        self.assertEqual(R.thresholds('s3'), (.9645,.548))
        for server, sensor, sid in (('s1','QB',261001011),('s3','GF2',261001012)):
            self.assertEqual(R.destination(server)['sheet_title'], sensor+'-ablations')
            self.assertEqual(R.destination(server)['sheet_id'],sid)
            with self.assertRaises(ValueError):
                R.destination(server,'WV3')
            h,e = R.thresholds(server)
            for hh,ee,passed in ((h,e,True),(h-1e-9,e,False),(h,e+1e-9,False)):
                with tempfile.TemporaryDirectory() as root:
                    persist(root,server=server,h=hh,e=ee)
                    view=B.project(root,server)
                    self.assertFalse(view['conflicts'],view['conflicts'])
                    self.assertEqual(len(view['rows']),1)
                    note=json.loads(view['rows'][0][R.INDEX['Original description / notes']])
                    self.assertEqual(note['publication']['publication_joint_pass'],passed)
                    self.assertEqual(note['publication']['hqnr_min'],h)
                    self.assertEqual(note['publication']['ergas_max'],e)
                    B.verify_owned(view['rows'][0],view)

    def test_all_cases_all_passes_and_minimum_ergas_representative(self):
        for server in R.DATASETS:
            h,e=R.thresholds(server)
            with tempfile.TemporaryDirectory() as root:
                runs=[run for group in plan.groups(server) for run in group['runs'] if run[0] not in plan.DONORS]
                self.assertEqual(len(runs),5)
                for case,replica in runs:
                    persist(root,case,server=server,step=1000,h=h+.001,e=e)
                    persist(root,case,server=server,step=2000,h=h,e=e-.001)
                    persist(root,case,server=server,step=3000,h=h-.001,e=e-.01)
                view=B.project(root,server)
                self.assertFalse(view['conflicts'],view['conflicts'])
                self.assertEqual(len(view['rows']),10)
                for row in view['rows']:
                    B.verify_owned(row,view)
                    self.assertEqual(row[R.INDEX['Selected_step']],
                                     2000 if row[R.INDEX['Selector']]==B.SELECTOR else 1000)
                    self.assertEqual(row[R.INDEX['Dataset']],R.DATASETS[server])
                    self.assertIsInstance(row[R.INDEX['Q4/Q8↑']],float)
                    for token in ('6ch','Hard alpha=','gamma=','beta=0','L4 warp'):
                        self.assertIn(token,row[R.INDEX['Approach']])
                    self.assertIn('4-band',row[R.INDEX['Model / inputs']])

    def test_fallback_and_transition_to_qualified_representative(self):
        for server in R.DATASETS:
            h,e=R.thresholds(server)
            with tempfile.TemporaryDirectory() as root:
                persist(root,server=server,step=1000,h=h-.003,e=e-.1)
                persist(root,server=server,step=2000,h=h-.001,e=e+.2)
                persist(root,server=server,step=3000,h=h-.001,e=e+.1)
                before=B.project(root,server)
                self.assertEqual(before['selected'][0]['step'],3000)
                self.assertEqual(before['rows'][0][R.INDEX['Selector']],B.FALLBACK_SELECTOR)
                persist(root,server=server,step=4000,h=h,e=e+.3)
                after=B.project(root,server)
                self.assertEqual(after['selected'][0]['step'],4000)
                self.assertEqual(after['rows'][0][R.INDEX['Result_ID']],before['rows'][0][R.INDEX['Result_ID']])
                self.assertIn('ERGAS 미달',after['rows'][0][R.INDEX['Selection@Step']])
                self.assertEqual(len(B.row_requests(snapshot(after,before['rows'],before['notes']),after,server)),1)

    def test_fresh_donor_selected_rre_receipt_is_counted_as_new_training(self):
        for server in R.DATASETS:
            cfg=plan.resolve_case(server,plan.DONORS[0])
            with tempfile.TemporaryDirectory() as root:
                receipt,ev=reused_receipt(root,server=server,source_config=cfg)
                original=deepcopy((receipt,ev))
                R.emit_evaluation(root,cfg,ev,selector_receipt=receipt)
                view=B.project(root,server)
                self.assertFalse(view['conflicts'],view['conflicts'])
                self.assertEqual((receipt,ev),original)
                row=view['rows'][0]
                self.assertEqual(row[R.INDEX['Selector']],'BEST_VAL_RRE')
                self.assertEqual(row[R.INDEX['Updates']],50000)
                note=json.loads(row[R.INDEX['Original description / notes']])
                self.assertEqual(note['publication']['locally_completed_updates'],50000)
                self.assertFalse(note['imported_source_not_new_training'])

    def test_all_metrics_and_cost_bind_to_representative_step(self):
        with tempfile.TemporaryDirectory() as root:
            h,e=R.thresholds('s1')
            persist(root,step=1000,h=h+.01,e=e+.1)
            record=persist(root,step=2000,h=h,e=e-.1)
            def attach(value, root, **kwargs):
                self.assertEqual(value['selected'][0]['step'],2000)
                value['rows'][0][R.INDEX['Infer(ms)']]=12.3
            with patch.object(C,'attach_cost',side_effect=attach):
                value=B.project(root,'s1')
            self.assertEqual(value['rows'][0][3:15],record['row'][3:15])
            self.assertEqual(value['rows'][0][R.INDEX['Infer(ms)']],12.3)

    def test_correction_reference_state_and_configuration_tamper_rejected(self):
        for server in R.DATASETS:
            cfg=plan.resolve_case(server,R.DATASETS[server]+'_BASE')
            for key,value in (('beta',.05),('alpha',1.),('gamma',2.)):
                bad=deepcopy(cfg);bad['stage2_spec'][key]=value
                with self.assertRaises(ValueError):R._config(bad)
            for key,value in (('hqnr_min',.9585),('ergas_max',2.040)):
                bad=deepcopy(cfg);bad[key]=value
                with self.assertRaises(ValueError):R._config(bad)
            bad=deepcopy(cfg);bad['stage2_spec']['sensor']='WV3'
            with self.assertRaises(ValueError):R._config(bad)
            with tempfile.TemporaryDirectory() as root:
                persist(root,server=server)
                path=next(Path(root).glob('runs/*/attempt_01/evaluation_*.json'))
                ev=json.loads(path.read_text())
                for field,value in (('pan_applied_dy_dx',[0.,0.]),('ms_applied_dy_dx',[1.,0.])):
                    bad=deepcopy(ev);bad['fr']['per_scene'][0][field]=value
                    with self.assertRaises(ValueError):P._verify_corrections(bad,cfg)
                ev['state_after']['model_state_sha256']='b'*64
                R.atomic_json(path,ev)
                self.assertTrue(B.project(root,server)['conflicts'])

    def test_cost_graph_is_four_band_six_input_and_excludes_teacher(self):
        import torch
        from panda_sensor5.cost.profiling import build_deployment, registered_profiles
        previous=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for server in R.DATASETS:
                cfg=plan.resolve_case(server,R.DATASETS[server]+'_BASE')
                graph=build_deployment(cfg,None,1000)
                self.assertEqual(graph.inference.student.spec.bands,4)
                self.assertEqual(graph.inference.student.pan_channels,2)
                count=sum(p.numel() for p in graph.inference.student.parameters())
                self.assertEqual(count,cfg['student_sampling']['student_parameters'])
                self.assertIs(graph.teacher,graph.inference.student)
                profiles=registered_profiles(server)
                self.assertEqual(len(profiles),2)
                self.assertEqual([cfg['phase'] for cfg,_ in profiles],['donor','stage2'])
                sig=C.signature(cfg,1000)
                self.assertEqual(sig['dataset'],R.DATASETS[server])
                self.assertEqual(sig['reconstructor']['bands'],4)
                self.assertEqual(sig['reconstructor_input']['channels'],6)
        finally:torch.set_num_threads(previous)

    def test_host_publication_imports_no_training_or_cloud_dependencies(self):
        code='''
import importlib.abc,sys
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','h5py','google'): raise AssertionError(fullname)
sys.meta_path.insert(0,Reject())
import panda_sensor5.reporting,panda_sensor5.publication,panda_sensor5.best_publication,panda_sensor5.cost
'''
        got=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(got.returncode,0,got.stderr)

    def test_anchored_sheet_append_readback_preserves_foreign_rows(self):
        class Adapter:
            BASE='https://example.invalid/'
            def __init__(self,data):self.data,self.writes=deepcopy(data),[]
            def snapshot(self,destination):return deepcopy(self.data)
            def _request(self,method,url,**kwargs):
                self.writes.append(url)
                title=self.data['metadata']['sheets'][0]['properties']['title']
                if url.endswith(':append'):
                    assert unquote(url).endswith(f"'{title}'!A5:A10000:append")
                    rows=kwargs['json']['values'];start=6+len(self.data['rows'])
                    self.data['rows'].extend(rows);self.data['notes'].extend(['']*len(rows))
                    return {'updates':{'updatedRange':f"'{title}'!A{start}:BL{start+len(rows)-1}",
                        'updatedRows':len(rows),'updatedColumns':64,'updatedCells':64*len(rows)}}
                for request in kwargs['json']['requests']:
                    update=request['updateCells'];index=update['range']['startRowIndex']-5
                    assert update['range']['sheetId']==self.data['metadata']['sheets'][0]['properties']['sheetId']
                    cells=update['rows'][0]['values']
                    self.data['rows'][index]=[next(iter(c.get('userEnteredValue',{}).values()),'') for c in cells]
                    self.data['notes'][index]=cells[0].get('note','')
                return {}
        for server in R.DATASETS:
            with tempfile.TemporaryDirectory() as root:
                persist(root,server=server)
                value=B.project(root,server)
                foreign=['']*64;foreign[0]='previous independent campaign'
                foreign[R.INDEX['Server']]=server
                adapter=Adapter(snapshot(value,[foreign],['preserve this note']))
                state=Path(root)/'transport';state.mkdir()
                self.assertTrue(T.reconcile_rows(adapter,value,state,server,apply=True)['readback_verified'])
                h,e=R.thresholds(server)
                persist(root,server=server,step=2000,h=h+.001,e=e-.01)
                value=B.project(root,server)
                self.assertTrue(T.reconcile_rows(adapter,value,state,server,apply=True)['readback_verified'])
                self.assertEqual(adapter.data['rows'][0],foreign)
                self.assertEqual(adapter.data['notes'][0],'preserve this note')
                before=len(adapter.writes)
                T.reconcile_rows(adapter,value,state,server,apply=True)
                self.assertEqual(len(adapter.writes),before)

    def test_restart_only_matches_new_sensor_publisher(self):
        root=Path('/tmp/sensor5')
        args=['python','-m','panda_sensor5.best_publication','--root',str(root),'--server','s1','--apply','--watch']
        self.assertTrue(matches_writer(args,root,'s1'))
        for module in ('panda_sensor5','panda_tune30.best_publication','panda_hard36.best_publication'):
            changed=list(args);changed[2]=module
            self.assertFalse(matches_writer(changed,root,'s1'))

    def test_profiling_limit_does_not_drop_cases_after_old_thirty_hour_budget(self):
        from panda_sensor5.cost.__main__ import profiling_deadline
        self.assertEqual(profiling_deadline({'started_unix':1.,'elapsed_seconds':200000.},200001.,1200),201201.)
        with self.assertRaises(ValueError):profiling_deadline({'started_unix':1.},2.,-1.)


if __name__=='__main__':unittest.main()
