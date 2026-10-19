"""Offline full-step publication and actual deployment accounting; no cloud writes."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from panda30r.tests.test_reporting import evaluation as old_evaluation
from panda_warp30 import plan, reporting as R, publication as P

ROOT = Path(__file__).resolve().parents[2]


def config(case='U23_PAN'):
    return plan.resolve_case('s3', case)


def evaluation(cfg, step=1000, h=.959, e=2.2):
    value = old_evaluation(cfg, step, h, e)
    value.update(upsampler=cfg['upsampler'], frame=cfg['frame'], shift_scale=cfg['shift_scale'],
                 output_coordinates='MIDPOINT' if cfg['frame']=='mid' else 'MS',
                 inverse_output_warp=cfg['frame'] in ('mid_return','ms_return'), reference_warp=False)
    return value


def persist(root, cfg, step=1000, h=.959, e=2.2):
    ev = evaluation(cfg, step, h, e)
    path = Path(root)/'runs'/cfg['run_id']/'attempt_01'
    path.mkdir(parents=True, exist_ok=True)
    R.atomic_json(path/'config.json', cfg)
    R.atomic_json(path/f'evaluation_{step:07d}.json', ev)
    return R.emit_evaluation(root, cfg, ev, timestamp='2026-10-19T00:00:00+00:00')


class ReportingTests(unittest.TestCase):
    def test_two_phases_all_metrics_and_geometry_are_explicit(self):
        for case in (*plan.DONORS, *plan.CASES):
            cfg = config(case)
            record = R.make_record(cfg, evaluation(cfg))
            R.validate_record(record)
            self.assertTrue(record['record_id'].startswith('PW30-'))
            self.assertEqual(len(record['row']), 64)
            self.assertEqual(record['destination']['sheet_title'], 'WV3-ablations')
            for col in ('RMSE↓','CC↑','JQM↑'):
                self.assertIsInstance(record['row'][R.INDEX[col]], (int,float))
            note = json.loads(record['row'][R.INDEX['Original description / notes']])
            self.assertTrue(note['official_references_unchanged'])
            self.assertEqual(note['geometry'], cfg['geometry'])
            self.assertEqual(note['output_coordinate_system'], 'MIDPOINT' if cfg['frame']=='mid' else 'MS')

    def test_every_verified_step_publishes_even_far_below_quality_targets(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config()
            persist(root,cfg,1000,.91,2.8)
            persist(root,cfg,2000,.962,2.4)
            persist(root,cfg,3000,.959,2.0)
            persist(root,cfg,4000,.962,2.3)
            value = P.project_campaign(root,'s3')
            self.assertFalse(value['conflicts'])
            self.assertEqual([r['step'] for r in value['selected']], [1000,2000,3000,4000])
            self.assertEqual(len((Path(root)/'records.jsonl').read_text().splitlines()),4)
            self.assertEqual(len(list((Path(root)/'outbox').glob('PW30-*.json'))),4)
            notes = [json.loads(r[R.INDEX['Original description / notes']])['publication'] for r in value['rows']]
            self.assertEqual([n['evaluation_step'] for n in notes if n['hqnr_primary_rank_best']], [4000])
            self.assertTrue(all(n['target_thresholds_are_annotations_only'] for n in notes))

    def test_synthetic_partial_mixed_reference_and_policy_tamper_rejected(self):
        for kind in ('partial','synthetic','state','reference','frame','config'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                persist(root,config())
                path = next(Path(root).glob('runs/*/attempt_01/evaluation_*.json'))
                ev = json.loads(path.read_text())
                if kind=='partial': ev['fr']['per_scene'].pop()
                elif kind=='synthetic': ev['metric_function_injection']=True
                elif kind=='state': ev['state_after']['model_state_sha256']='b'*64
                elif kind=='reference': ev['reference_warp']=True
                elif kind=='frame': ev['frame']='mid'
                elif kind=='config':
                    changed=config();changed['upsampler']='bicubic'
                    R.atomic_json(path.parent/'config.json',changed)
                R.atomic_json(path,ev)
                self.assertTrue(P.project_campaign(root,'s3')['conflicts'])

    def test_config_cannot_relabel_legacy_or_geometry(self):
        for key, value in (('campaign_id','old'),('run_id','PMID_U23_old'),('engine_kind','stage2_midpoint'),
                           ('geometry',{}),('frame','mid'),('shift_scale',float('nan')),('server','s1')):
            cfg=config();cfg[key]=value
            with self.assertRaises(ValueError):
                R.make_record(cfg,evaluation(cfg))

    def test_outbox_is_immutable_and_cost_absence_does_not_lose_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=config();record=persist(root,cfg)
            R.LocalOutbox(root).enqueue(record)
            self.assertEqual(len((Path(root)/'records.jsonl').read_text().splitlines()),1)
            bad=deepcopy(record);bad['row'][0]='tamper'
            with self.assertRaises(ValueError): R.LocalOutbox(root).enqueue(bad)
            self.assertTrue(P.project(root,'s3')['conflicts'])
            self.assertFalse(P.project(root,'s3',require_cost=False)['conflicts'])

    def test_all_step_publisher_never_erases_foreign_or_missing_historical_rows(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root,config())
            value=P.project_campaign(root,'s3')
            foreign=['']*64
            foreign[R.INDEX['Campaign_ID']]='FOREIGN_CAMPAIGN'
            snapshot={'rows':[foreign], 'notes':['foreign note']}
            requests=P.row_requests(snapshot,value,'s3')
            self.assertEqual(len(requests),1)
            self.assertIn('appendCells',requests[0])
            own=list(value['rows'][0]);own[R.INDEX['Result_ID']]='VIEW-'+'f'*64
            snapshot={'rows':[foreign,own], 'notes':['foreign note','old official state']}
            with self.assertRaisesRegex(ValueError,'never erase'):
                P.row_requests(snapshot,value,'s3')

    def test_publisher_import_does_not_need_training_or_google_dependencies(self):
        code='''
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','numpy','h5py','google'):
            raise AssertionError('Unexpected dependency: '+fullname)
sys.meta_path.insert(0,Reject())
import panda_warp30.publication, panda_warp30.cost, panda_warp30.transport
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)


class CostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)

    def test_profile_keys_bind_phase_operator_geometry_strength_but_not_seed(self):
        from panda_warp30 import cost
        base=cost.signature(config())
        self.assertEqual(base,cost.signature(plan.resolve_case('s3','U23_PAN','R2')))
        for case in ('S1_U23','S1_BIC','BIC_PAN','U23_MID','U23_MID_RETURN','U23_MS_RETURN','U23_PAN_S050'):
            self.assertNotEqual(base,cost.signature(config(case)))
        for path in ('panda_warp30/method.py','panda_warp30/reconstruction.py','panda_midpoint/resampling.py','align/resample.py'):
            self.assertIn(path,base['source_sha256'])

    def test_profile_graph_executes_real_warps_excludes_training_teacher(self):
        import torch
        from panda_stage2_condq.tests.test_training_evaluation import EvalSources, rr_batch
        from panda_warp30 import cost, method
        for case, calls in (('U23_PAN',1),('U23_MID',2),('U23_MID_RETURN',3),
                            ('U23_MS_RETURN',2),('U23_PAN_S000',0)):
            graph=cost.build_deployment(config(case),EvalSources())
            self.assertFalse(hasattr(graph,'student'))
            self.assertEqual(sum(p.numel() for p in graph.parameters()),
                sum(p.numel() for p in graph.aligner.parameters())+
                sum(p.numel() for p in graph.teacher.parameters()))
            with torch.no_grad():
                graph.aligner.fc2.bias.copy_(torch.tensor([.25,-.5]))
                graph.teacher.backbone.output[-1].bias.fill_(.1)
            with patch.object(method,'warp_pan',wraps=method.warp_pan) as warp:
                output=graph.native_prediction(rr_batch())
            self.assertEqual(warp.call_count,calls,case)
            self.assertEqual(tuple(output['y'].shape),(1,8,64,64))
            self.assertEqual(output['output_coordinates'],'MIDPOINT' if case=='U23_MID' else 'MS')

    def test_donor_profile_uses_new_real_teacher_upsampler(self):
        import torch
        from panda_stage2_condq.tests.test_training_evaluation import EvalSources, rr_batch
        from panda_warp30 import cost
        from panda_warp30.reconstruction import base_from_batch
        for case in ('S1_U23','S1_BIC'):
            graph=cost.build_deployment(config(case),EvalSources())
            with torch.no_grad(): output=graph.native_prediction(rr_batch())
            torch.testing.assert_close(output['ms_base'],base_from_batch(rr_batch(),graph.upsampler))

    def test_returned_output_real_rr256_fr512_inference_has_no_gt_dependency(self):
        from dataclasses import replace
        import torch
        from panda_method.edge_data import EdgeObservationBatch
        from panda_stage2_condq.tests.test_training_evaluation import EvalSources
        from panda_warp30 import cost
        graph=cost.build_deployment(config('U23_MID_RETURN'),EvalSources())
        with torch.no_grad():
            graph.aligner.fc2.bias.copy_(torch.tensor([.25,-.5]))
            graph.teacher.backbone.output[-1].bias.fill_(.1)
        for split,side in (('rr',256),('fr',512)):
            lineage=EvalSources().batch(split,[0]).lineage
            pan=torch.linspace(-.8,.8,side*side).reshape(1,1,side,side)
            ms=torch.zeros(1,8,side//4,side//4)
            lms=torch.ones(1,8,side,side)*.2
            observation=EdgeObservationBatch(pan,ms,lms,lineage,None)
            with torch.no_grad():
                output=graph.native_prediction(observation)
                changed=graph.native_prediction(replace(observation,gt=torch.ones_like(lms)*100))
            self.assertEqual(tuple(output['y'].shape),(1,8,side,side))
            torch.testing.assert_close(output['y'],changed['y'],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
