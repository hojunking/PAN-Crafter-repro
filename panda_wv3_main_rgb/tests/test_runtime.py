from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from donor30.artifacts import _same
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_wv3_ablation_rgb import plan as rgb_plan
from panda_wv3_main_rgb import plan,training


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_queue_is_exact_round_major_paired_full_method(self):
        q=plan.queue('s1')
        self.assertEqual((len(q),sum(c['horizon'] for c in q)),(21,696000))
        self.assertEqual(len({c['run_id'] for c in q}),21)
        self.assertEqual(plan.catalog('s1'),(plan.CONFIG/'cases_s1.csv').read_text())
        for repeat in (1,2,3):
            cycle=q[(repeat-1)*7:repeat*7]
            self.assertEqual(cycle[0]['case_id'],'S1_RGB_FULL')
            self.assertEqual({c['repeat'] for c in cycle},{repeat})
            old=rgb_plan.resolve('s3','S1_A5_ALL',repeat)
            for key in ('stream_seeds','lambda_abs','lambda_epsilon','horizon','lr','lr_aligner','warp'):
                self.assertEqual(cycle[0][key],old[key])
            for c in cycle:
                plan.validate(c)
                self.assertNotIn('donor_checkpoint',c)
                self.assertFalse(c['inference_tta']);self.assertFalse(c['rotation_consistency'])
            for c in cycle[1:]:
                self.assertEqual(c['donor_run_id'],cycle[0]['run_id'])
                self.assertEqual(c['stream_seeds'],cycle[1]['stream_seeds'])
                self.assertEqual(c['scale'],1.)
                self.assertGreater(c['alpha'],0);self.assertGreater(c['gamma'],0)
                self.assertEqual(c['beta'],0)
        with self.assertRaises(ValueError):plan.queue('s3')

    def test_failed_donor_blocks_only_its_students(self):
        from panda_wv3_main_rgb import runtime
        state=runtime.initial_state('s1');state['runs'][0]['status']='FAILED'
        for job in state['runs'][1:7]:
            cfg=plan.resolve('s1',job['case'],job['repeat'])
            self.assertTrue(runtime.block_failed_dependency(state,job,cfg))
            self.assertEqual(job['status'],'BLOCKED_DONOR')
        self.assertEqual(runtime.next_job(state)['run_id'],state['runs'][7]['run_id'])
        untouched=runtime.initial_state('s1')
        with self.assertRaisesRegex(ValueError,'completed local'):
            runtime.block_failed_dependency(untouched,untouched['runs'][1],plan.queue('s1')[1])

    def test_rgb_stage1_complete_optimizer_stream_resume(self):
        from panda_wv3_main_rgb.model import Stage1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registered=synthetic_sources(root,count=3)
            cfg=plan.resolve('s1','S1_RGB_FULL',1);cfg.update(batch=1,microbatch=1)
            def build():
                with patch('panda_wv3_main_rgb.plan.validate'):
                    trainer=training.Trainer(cfg,Stage1(cfg,registered.inventory.manifest()))
                return trainer,training.stream_for(cfg,registered,'cpu')
            t,s=build();t.step(s.next_update());training.save(root/'resume.pt',t,s,{'RGB_MAIN':True})
            result=t.step(s.next_update());weights=deepcopy(t.model.state_dict());optimizer=deepcopy(t.optimizer.state_dict())
            t2,s2=build();training.load(root/'resume.pt',t2,s2,{'RGB_MAIN':True})
            self.assertEqual(result,t2.step(s2.next_update()))
            self.assertTrue(_same(weights,t2.model.state_dict()))
            self.assertTrue(_same(optimizer,t2.optimizer.state_dict()))
            self.assertTrue(_same(s.state_dict(),s2.state_dict()))
            with self.assertRaisesRegex(ValueError,'Resume'):
                training.load(root/'resume.pt',t2,s2,{'OLD_MEAN8':True})

    def test_all_evaluated_weights_retained_but_resume_uses_complete_generations(self):
        from panda_wv3_main_rgb import runtime
        from panda_wv3_main_rgb.model import Stage1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registered=synthetic_sources(root,count=3)
            cfg=plan.resolve('s1','S1_RGB_FULL',1);cfg.update(batch=1,microbatch=1)
            with patch('panda_wv3_main_rgb.plan.validate'):
                t=training.Trainer(cfg,Stage1(cfg,registered.inventory.manifest()))
            s=training.stream_for(cfg,registered,'cpu');pins={'RGB_MAIN':True}
            for i in range(1,4):
                t.step(s.next_update())
                runtime.save_resume(root,t,s,pins)
                training.save(root/'models'/f'step_{i:07d}.pt',t,s,pins,immutable=True,evaluated=True)
            self.assertEqual(len(list((root/'resume').glob('*.pt'))),2)
            self.assertEqual(len(list((root/'models').glob('*.pt'))),3)
            evaluated=root/'models'/'step_0000003.pt'
            value=torch.load(evaluated,map_location='cpu',weights_only=False)
            self.assertEqual(value['schema'],'WV3_MAIN_RGB_EVALUATED_MODEL_V1')
            self.assertNotIn('optimizer',value);self.assertNotIn('stream',value)
            self.assertEqual(value['model_sha256'],training.model_hash(t.model))
            self.assertTrue(_same(value['model'],t.model.state_dict()))
            self.assertEqual(runtime.latest_checkpoint(root).parent,root/'resume')
            with self.assertRaisesRegex(ValueError,'Resume'):
                training.load(evaluated,t,s,pins)


if __name__=='__main__':unittest.main()
