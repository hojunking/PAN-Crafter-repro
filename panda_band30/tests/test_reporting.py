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
from panda_band30 import plan, reporting as R, publication as P

ROOT = Path(__file__).resolve().parents[2]


def config(case='G8_PAN_U23'):
    return plan.resolve_case('s1', case)


def evaluation(cfg, step=1000, h=.959, e=2.2):
    value = old_evaluation(cfg, step, h, e)
    value.update(upsampler=cfg['upsampler'], frame=cfg['frame'], shift_scale=cfg['shift_scale'],
                 output_coordinates='MIDPOINT' if cfg['frame']=='mid' else 'MS',
                 inverse_output_warp=cfg['frame']=='mid_return', reference_warp=False,
                 aligner_outputs=cfg['aligner_outputs'], pan_channels=cfg['pan_channels'],
                 alignment_target=cfg['alignment_target'])
    for split in ('rr','fr'):
        for row in value[split]['per_scene']:
            row['correction_band_order'] = list(range(8)) if cfg['pan_channels']==8 else None
            for key in ('correction_dy_dx','pan_applied_dy_dx','ms_applied_dy_dx','output_applied_dy_dx'):
                row[key] = [[0.,0.] for _ in range(8)] if cfg['pan_channels']==8 else [0.,0.]
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
        for item in plan.registered_rows():
            cfg = plan.resolve_case('s1', item['case_id'], item['replica'])
            record = R.make_record(cfg, evaluation(cfg))
            R.validate_record(record)
            self.assertTrue(record['record_id'].startswith('PB30-'))
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
            value = P.project_campaign(root,'s1')
            self.assertFalse(value['conflicts'])
            self.assertEqual([r['step'] for r in value['selected']], [1000,2000,3000,4000])
            self.assertEqual(len((Path(root)/'records.jsonl').read_text().splitlines()),4)
            self.assertEqual(len(list((Path(root)/'outbox').glob('PB30-*.json'))),4)
            notes = [json.loads(r[R.INDEX['Original description / notes']])['publication'] for r in value['rows']]
            self.assertEqual([n['evaluation_step'] for n in notes if n['hqnr_primary_rank_best']], [4000])
            self.assertTrue(all(n['target_thresholds_are_annotations_only'] for n in notes))

    def test_all_35_registered_steps_keep_full_precision_local_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=config('B8A_PAN_U23')
            self.assertEqual(len(cfg['evaluation_steps']),35)
            precise=.9412345678912345
            for step in cfg['evaluation_steps']:
                persist(root,cfg,step,precise,2.76543210987654)
            value=P.project_campaign(root,'s1')
            self.assertFalse(value['conflicts'])
            self.assertEqual(len(value['rows']),35)
            record=json.loads(next((Path(root)/'outbox').glob('PB30-*.json')).read_text())
            self.assertEqual(record['source']['metrics']['HQNR'],precise)
            # Existing fixed64 transports full precision; the workbook's
            # display format is separate and must not alter provenance.
            self.assertEqual(record['row'][R.INDEX['HQNR↑']],precise)

    def test_synthetic_partial_mixed_reference_and_policy_tamper_rejected(self):
        for kind in ('partial','synthetic','state','reference','frame','config','shift','collapsed'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                persist(root,config())
                path = next(Path(root).glob('runs/*/attempt_01/evaluation_*.json'))
                ev = json.loads(path.read_text())
                if kind=='partial': ev['fr']['per_scene'].pop()
                elif kind=='synthetic': ev['metric_function_injection']=True
                elif kind=='state': ev['state_after']['model_state_sha256']='b'*64
                elif kind=='reference': ev['reference_warp']=True
                elif kind=='frame': ev['frame']='mid'
                elif kind=='shift': ev['rr']['per_scene'][0]['pan_applied_dy_dx']=[1.,0.]
                elif kind=='collapsed': ev['fr']['per_scene'][0]['correction_dy_dx']=0.
                elif kind=='config':
                    changed=config();changed['upsampler']='bicubic'
                    R.atomic_json(path.parent/'config.json',changed)
                R.atomic_json(path,ev)
                self.assertTrue(P.project_campaign(root,'s1')['conflicts'])

    def test_config_cannot_relabel_legacy_or_geometry(self):
        for key, value in (('campaign_id','old'),('run_id','PMID_U23_old'),('engine_kind','stage2_midpoint'),
                           ('geometry',{}),('frame','mid'),('shift_scale',float('nan')),('server','s3'),
                           ('aligner_outputs',16),('pan_channels',8),('alignment_target','RGB')):
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
            self.assertTrue(P.project(root,'s1')['conflicts'])
            self.assertFalse(P.project(root,'s1',require_cost=False)['conflicts'])

    def test_all_step_publisher_never_erases_foreign_or_missing_historical_rows(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root,config())
            value=P.project_campaign(root,'s1')
            foreign=['']*64
            foreign[R.INDEX['Campaign_ID']]='FOREIGN_CAMPAIGN'
            snapshot={'rows':[foreign], 'notes':['foreign note']}
            requests=P.row_requests(snapshot,value,'s1')
            self.assertEqual(len(requests),1)
            self.assertIn('appendCells',requests[0])
            own=list(value['rows'][0]);own[R.INDEX['Result_ID']]='VIEW-'+'f'*64
            snapshot={'rows':[foreign,own], 'notes':['foreign note','old official state']}
            with self.assertRaisesRegex(ValueError,'never erase'):
                P.row_requests(snapshot,value,'s1')

    def test_publisher_import_does_not_need_training_or_google_dependencies(self):
        code='''
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','numpy','h5py','google'):
            raise AssertionError('Unexpected dependency: '+fullname)
sys.meta_path.insert(0,Reject())
import panda_band30.publication, panda_band30.cost, panda_band30.transport
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
        from panda_band30 import cost
        base=cost.signature(config())
        self.assertEqual(base,cost.signature(plan.resolve_case('s1','G8_PAN_U23','R2')))
        for case in ('G8_S1_U23','G8_S1_BIC','G8_PAN_BIC','G8_MID_U23','G8_MIDRET_U23','B8A_PAN_U23','G8_PAN_U23_S050'):
            self.assertNotEqual(base,cost.signature(config(case)))
        for path in ('panda_band30/method.py','panda_band30/reconstruction.py','panda_midpoint/resampling.py','align/resample.py'):
            self.assertIn(path,base['source_sha256'])

    def test_profile_graph_executes_real_warps_excludes_training_teacher(self):
        import torch
        from panda_stage2_condq.tests.test_training_evaluation import EvalSources, rr_batch
        from panda_band30 import cost, method, aligner
        for case, calls in (('G8_PAN_U23',1),('G8_MID_U23',2),('G8_MIDRET_U23',3),
                            ('G8_PAN_U23_S000',0)):
            graph=cost.build_deployment(config(case),EvalSources())
            self.assertFalse(hasattr(graph,'student'))
            self.assertEqual(sum(p.numel() for p in graph.parameters()),
                sum(p.numel() for p in graph.aligner.parameters())+
                sum(p.numel() for p in graph.teacher.parameters()))
            with torch.no_grad():
                graph.aligner.fc2.bias.copy_(torch.tensor([.25,-.5]))
                graph.teacher.backbone.output[-1].bias.fill_(.1)
            with patch.object(method,'warp_pan',wraps=method.warp_pan) as warp, \
                    patch.object(aligner,'warp_pan',wraps=aligner.warp_pan) as pan_warp:
                output=graph.native_prediction(rr_batch())
            self.assertEqual(warp.call_count+pan_warp.call_count,calls,case)
            self.assertEqual(tuple(output['y'].shape),(1,8,64,64))
            self.assertEqual(output['output_coordinates'],'MIDPOINT' if case=='G8_MID_U23' else 'MS')

    def test_donor_profile_uses_new_real_teacher_upsampler(self):
        import torch
        from panda_stage2_condq.tests.test_training_evaluation import EvalSources, rr_batch
        from panda_band30 import cost
        from panda_band30.reconstruction import base_from_batch
        for case in ('G8_S1_U23','G8_S1_BIC','B8A_S1_U23'):
            graph=cost.build_deployment(config(case),EvalSources())
            with torch.no_grad(): output=graph.native_prediction(rr_batch())
            torch.testing.assert_close(output['ms_base'],base_from_batch(rr_batch(),graph.upsampler))

    def test_bandwise_cost_uses_real_head_and_16_channel_stem(self):
        import torch
        from panda_stage2_condq.tests.test_training_evaluation import EvalSources, rr_batch
        from panda_band30 import cost, aligner
        global_graph=cost.build_deployment(config(),EvalSources())
        band=cost.build_deployment(config('B8A_PAN_U23'),EvalSources())
        self.assertEqual(band.aligner.fc2.out_features,16)
        self.assertEqual(band.teacher.backbone.input.in_channels,16)
        self.assertEqual(sum(p.numel() for p in band.parameters())-
                         sum(p.numel() for p in global_graph.parameters()),7518)
        with torch.no_grad():
            band.aligner.fc2.bias.copy_(torch.arange(16).float()*.02)
        with patch.object(aligner,'warp_pan',wraps=aligner.warp_pan) as warp:
            output=band.native_prediction(rr_batch())
        self.assertEqual(warp.call_count,1)
        self.assertEqual(tuple(warp.call_args.args[0].shape),(8,1,64,64))
        self.assertEqual(tuple(output['u'].shape),(1,8,2))
        self.assertEqual(tuple(output['x_in'].shape),(1,16,64,64))

    def test_returned_output_real_rr256_fr512_inference_has_no_gt_dependency(self):
        from dataclasses import replace
        import torch
        from panda_method.edge_data import EdgeObservationBatch
        from panda_stage2_condq.tests.test_training_evaluation import EvalSources
        from panda_band30 import cost
        graph=cost.build_deployment(config('G8_MIDRET_U23'),EvalSources())
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
