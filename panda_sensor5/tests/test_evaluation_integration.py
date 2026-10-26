"""Execute all40 evaluation and runtime checkpoint orchestration on CPU fixtures."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from donor30.artifacts import _same
from panda_sensor5 import plan
from panda_sensor5.evaluation import official_evaluate
from panda_sensor5.runtime import Campaign
from panda_sensor5.stage1 import build_stage1, Stage1Stream
from panda_stage2_condq.stream import TrainStream, StreamSpec
from panda_sensor5.tests.test_core import sources, fixture, PINS


class EvaluationSources:
    """Small synthetic native grids; official_evaluate permits them only with TEST injection."""
    def __init__(self, actual):self.actual=actual
    def __getattr__(self,name):return getattr(self.actual,name)
    def batch(self,split,rows,*,device='cpu',**kwargs):
        if split not in ('rr','fr'):return self.actual.batch(split,rows,device=device,**kwargs)
        b=self.actual.batch('train',[0],device=device)
        lineage=replace(b.lineage,split='test',resolution=split.upper(),grid_name='PAN_'+split.upper(),
            sample_ids=(f'{self.sensor}:{split}:{rows[0]}',),source_manifest_sha256=self.manifest_sha(split))
        return replace(b,lineage=lineage,gt=None if split=='fr' else b.gt)
    def raw(self,split,rows):
        if split not in ('rr','fr'):return self.actual.raw(split,rows)
        result=self.actual.raw('train',[0])
        if split=='fr':result.pop('gt')
        return result


def metrics():
    def rr(p,a):
        if p.shape!=(4,64,64) or a['gt'].shape!=p.shape:raise AssertionError('Four-band RR prediction required')
        return dict(ergas=1.,scc=.9,sam=2.,psnr=30.,ssim=.9,q4=.9,rmse=5.,cc=.9)
    def fr(p,a):
        if 'gt' in a or p.shape!=(4,64,64):raise AssertionError('FR must be four-band and GT-free')
        return dict(hqnr=.98,d_s=.01,d_lambda=.01,jqm=.9)
    return {'rr':rr,'fr':fr}


class EvaluationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_all40_evaluation_and_runtime_roundtrip_for_both_stages_and_sensors(self):
        for sensor,server in [('QB','s1'),('GF2','s3')]:
            with tempfile.TemporaryDirectory() as tmp:
                registered=sources(tmp,sensor,count=1)
                evaluation_sources=EvaluationSources(registered.sources)
                for phase in ('donor','stage2'):
                    with self.subTest(sensor=sensor,phase=phase):
                        campaign=Campaign.__new__(Campaign)
                        campaign.sources=evaluation_sources;campaign.device='cpu'
                        campaign.code=PINS['code_pin'];campaign.evaluator_pin=PINS['evaluator_pin']
                        campaign.state={'timing':{}}
                        if phase=='donor':
                            cfg=plan.resolve_case(server,plan.DONORS[0])
                            cfg['baseline_resolved'].update(effective_batch_RR=1,microbatch_RR=1,accumulation_RR=1)
                            trainer,stream=campaign.build(cfg)
                        else:
                            trainer,_=fixture(sensor,registered=registered)
                            cfg=deepcopy(trainer.config)
                            stream=TrainStream(registered,StreamSpec(23,34),microbatch_size=1,effective_batch_size=1)
                        campaign.trace(trainer,stream)
                        before=deepcopy(trainer.state_dict() if phase=='donor' else
                            trainer.state_dict(stream_state=stream.state_dict()))
                        result=official_evaluate(trainer,evaluation_sources,metric_functions=metrics(),
                            evaluator_revision='TEST',evaluator_pin=PINS['evaluator_pin'],device='cpu')
                        self.assertEqual(len(result['rr']['per_scene']),20)
                        self.assertEqual(len(result['fr']['per_scene']),20)
                        self.assertIn('Q4',result['rr']['metrics']);self.assertNotIn('Q8',result['rr']['metrics'])
                        self.assertTrue(result['qualification']['same_state_verified'])
                        self.assertTrue(result['qualification']['metrics_complete'])
                        self.assertFalse(result['full_method_eligible'])
                        self.assertFalse(result['qualification']['joint_pass'])
                        after=trainer.state_dict() if phase=='donor' else trainer.state_dict(stream_state=stream.state_dict())
                        self.assertTrue(_same(before,after),'evaluation must preserve trainable/frozen state and Adam')
                        path=Path(tmp)/phase/'models'/'step_0000001.pt'
                        artifact=campaign._save(trainer,stream,path,cfg,evaluation=result)
                        campaign.trace(trainer,stream)
                        trainer,stream=campaign._restore(artifact,trainer,stream,cfg)
                        restored=trainer.state_dict() if phase=='donor' else trainer.state_dict(stream_state=stream.state_dict())
                        self.assertTrue(_same(before,restored),'runtime orchestration must restore exact complete state')

if __name__=='__main__':unittest.main()
