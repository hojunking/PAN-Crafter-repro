"""Fresh RGB donor evidence, calibration isolation and full Ours training path."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_stage2_condq.sources import RegisteredRRTrain
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from panda_wv3_ablation.model import adaptive_weight
from panda_wv3_ablation_rgb import selection
from ta2.evaluation import file_sha
from panda_wv3_main_rgb import assets, evaluation, model, plan, training


class RGBDonorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, root):
        root = Path(root)
        registered = synthetic_sources(root, count=3)
        catalog = json.loads(registered.sources.catalog_path.read_text())
        # A separate source identity is sufficient for mocked validation evidence;
        # this test never treats the synthetic three-row data as measured val1080.
        catalog['sensors']['WV3']['splits']['val'] = catalog['sensors']['WV3']['splits']['train'].copy()
        registered.sources.catalog_path.write_text(json.dumps(catalog))
        registered = RegisteredRRTrain.from_catalog(root/'sources.json',root=root,sensor='WV3')
        cfg = plan.resolve('s1','S1_RGB_FULL',1)
        cfg['evaluation_steps'] = [1000,50000]
        cfg['validation_steps'] = [1000,50000]
        cfg['config_sha256'] = plan.digest({k:v for k,v in cfg.items() if k!='config_sha256'})
        student = plan.resolve('s1','M0_OURS',1)
        directory = root/'runs'/cfg['run_id'];(directory/'models').mkdir(parents=True)
        (directory/'config.json').write_text(json.dumps(cfg))
        state = {'campaign_id':cfg['campaign_id'],'server':'s1','runs':[
            {'run_id':cfg['run_id'],'status':'COMPLETED','step':50000}]}
        (root/'campaign_state.json').write_text(json.dumps(state))
        inv = {'schema':'WV3_RGB_ABLATION_VAL_INVENTORY_V1',
            'source_manifest_sha256':registered.sources.manifest_sha('val'),
            'source_h5_sha256':registered.inventory.source_h5_sha256,
            'catalog_sha256':registered.inventory.catalog_sha256,
            'sample_ids':[f'WV3:val:{i}' for i in range(1080)],
            'eligible':[True]*1080,'target_rms':[1.]*1080,'sample_count':1080,
            'eligible_count':1080,'excluded_gt_low_texture_count':0,
            'selector_protocol_sha256':canonical_sha256(selection.PROTOCOL)}
        inv['inventory_sha256'] = canonical_sha256(inv)
        (root/'validation_inventory.json').write_text(json.dumps(inv))
        pins = {'source_release':'a'*64,'evaluator':{'synthetic':True},
            'train_inventory':registered.inventory.manifest(),
            'split_manifests':{'train':registered.sources.manifest_sha('train'),
                              'val':registered.sources.manifest_sha('val')}}
        source = model.Stage1(cfg,registered.inventory.manifest())
        for step, rre, hqnr in ((1000,.2,.90),(50000,.4,.98)):
            with torch.no_grad():source.aligner.fc2.bias.fill_(step/5000000)
            state_hash = training.model_hash(source)
            checkpoint = directory/'models'/f'step_{step:07d}.pt'
            torch.save({'schema':'WV3_MAIN_RGB_COMPLETE_CHECKPOINT_V1','config':cfg,
                'pins':pins,'updates':step,'model':source.state_dict(),'model_sha256':state_hash,
                'optimizer':{},'stream':{},'rng':{}},checkpoint)
            meta = {'path':str(checkpoint),'file_sha256':file_sha(checkpoint),'step':step,
                    'model_sha256':state_hash,'config_sha256':cfg['config_sha256']}
            checkpoint.with_suffix('.json').write_text(json.dumps(meta))
            rows = [{'sample_id':name,'eligible':True,'valid':True,'rre_px':rre,
                     'failure_reason':None} for name in inv['sample_ids']]
            val = dict(selection.aggregate_rows(rows,inv),Val_ERGAS=2.1,
                protocol=selection.PROTOCOL,per_sample=rows)
            (directory/f'validation_{step:07d}.json').write_text(json.dumps(val))
            record = {'schema':'WV3_MAIN_RGB_EVAL_V1','run_id':cfg['run_id'],
                'config_sha256':cfg['config_sha256'],'step':step,'same_state_verified':True,
                'reference_warp':False,'inference_tta':False,'model_sha256':state_hash,
                'eval_protocol_id':plan.digest(pins['evaluator']),'checkpoint':meta,
                'validation':{k:v for k,v in val.items() if k not in ('protocol','per_sample')}}
            for split, keys in [('rr',('CC','ERGAS','PSNR','Q8','RMSE','SAM','SCC','SSIM')),
                                ('fr',('HQNR','D_s','D_lambda','JQM'))]:
                scenes = [{'scene_id':f'WV3:{split}:{i}',
                    'metrics':{key:hqnr if key=='HQNR' else 1. for key in keys},
                    'rre_rgb':{'registration':{'valid':False,'shift':[0.,0.]},'rre_px':None}}
                    for i in range(20)]
                record[split] = {'per_scene':scenes,
                    'metrics':{key:float(np.mean([r['metrics'][key] for r in scenes])) for key in keys}}
            record['test_rre_rgb'] = {s:evaluation.rre_summary(record[s]['per_scene']) for s in ('rr','fr')}
            (directory/f'evaluation_{step:07d}.json').write_text(json.dumps(record))
        return registered,cfg,student,pins,directory

    def test_selection_ignores_test_metrics_and_loads_exact_frozen_rgb_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered,cfg,student,pins,directory = self.fixture(tmp)
            with patch.object(plan,'resolve',return_value=cfg):
                chosen_cfg,receipt = assets._select(student,registered,tmp,pins)
            self.assertEqual(receipt['selected']['step'],1000)  # 50K had much better test HQNR.
            self.assertFalse(receipt['uses_test_for_selection'])
            before = torch.random.get_rng_state().clone()
            donor = assets._load_selected(chosen_cfg,receipt,registered,tmp,pins,device='cpu')
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            self.assertEqual(donor.binding['alignment_target'],'SCHARR_OF_RGB_MEAN_GT')
            self.assertEqual(donor.binding['donor_step'],1000)
            torch.testing.assert_close(donor.aligner.fc2.bias,torch.full((2,),.0002))
            donor.assert_frozen()
            wrong = deepcopy(pins);wrong['source_release']='b'*64
            with self.assertRaisesRegex(ValueError,'same-source'):
                assets._load_selected(chosen_cfg,receipt,registered,tmp,wrong,device='cpu')

    def test_full_stage2_backward_rgb_q_calibration_and_safe_cache_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered,cfg,student,pins,_ = self.fixture(tmp)
            with patch.object(plan,'resolve',return_value=cfg):
                donor,stats = assets.prepare_reference(student,registered,tmp,pins)
            self.assertEqual(stats.binding['alignment_target'],'SCHARR_OF_RGB_MEAN_GT')
            self.assertEqual(stats.metadata['sample_count'],3)
            before = {r:state_dict_sha256(getattr(donor,r)) for r in ('aligner','teacher')}
            net = model.Stage2(student,donor,stats)
            rr = registered.batch([0])
            with patch('panda_wv3_ablation.model.adaptive_weight',wraps=adaptive_weight) as weight:
                out = net(rr)
            q = weight.call_args.args[2].clone()
            out['loss'].backward()
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in net.student.parameters()))
            self.assertTrue(all(p.grad is None for r in ('aligner','teacher') for p in getattr(net,r).parameters()))
            gt = rr.gt.clone();gt[:,[0,3,5,6,7]] += .1
            with patch('panda_wv3_ablation.model.adaptive_weight',wraps=adaptive_weight) as weight:
                net(replace(rr,gt=gt))
            torch.testing.assert_close(q,weight.call_args.args[2],rtol=0,atol=0)
            self.assertEqual(before,{r:state_dict_sha256(getattr(donor,r)) for r in before})
            donor.assert_frozen()
            folder = Path(tmp)/'donors/R1/calibration'
            with patch.object(assets,'frozen_predictions',side_effect=AssertionError('cache recalculated')):
                cached = assets.calibration(donor,registered,folder)
            self.assertEqual(cached.artifact_sha256,stats.artifact_sha256)
            (folder/'full_s100.json').unlink()  # Interrupted after immutable NPZ publication.
            restored = assets.calibration(donor,registered,folder)
            self.assertEqual(restored.artifact_sha256,stats.artifact_sha256)
            other = deepcopy(donor.binding);other['donor_checkpoint_id']='different donor'
            changed = replace(donor,binding=other)
            with self.assertRaisesRegex(ValueError,'another RGB donor'):
                assets.calibration(changed,registered,folder)

    def test_evaluated_snapshot_loads_frozen_donor_without_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered,cfg,student,pins,directory = self.fixture(tmp)
            checkpoint = directory/'models/step_0001000.pt'
            saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
            saved['schema'] = 'WV3_MAIN_RGB_EVALUATED_MODEL_V1'
            for key in ('optimizer','stream','rng'):
                saved.pop(key)
            torch.save(saved,checkpoint)
            meta = json.loads(checkpoint.with_suffix('.json').read_text())
            meta['file_sha256'] = file_sha(checkpoint)
            checkpoint.with_suffix('.json').write_text(json.dumps(meta))
            ep = directory/'evaluation_0001000.json'
            record = json.loads(ep.read_text());record['checkpoint'] = meta
            ep.write_text(json.dumps(record))
            with patch.object(plan,'resolve',return_value=cfg):
                donor_cfg,receipt = assets._select(student,registered,tmp,pins)
            donor = assets._load_selected(donor_cfg,receipt,registered,tmp,pins,device='cpu')
            donor.assert_frozen()
            self.assertEqual(donor.binding['donor_checkpoint_sha256'],meta['file_sha256'])
            self.assertEqual(donor.binding['donor_step'],1000)
            wrong = deepcopy(pins);wrong['source_release'] = 'b'*64
            with self.assertRaisesRegex(ValueError,'same-source'):
                assets._load_selected(donor_cfg,receipt,registered,tmp,wrong,device='cpu')

    def test_incomplete_history_validation_tamper_and_checkpoint_tamper_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered,cfg,student,pins,directory = self.fixture(tmp)
            statepath = Path(tmp)/'campaign_state.json';state=json.loads(statepath.read_text())
            state['runs'][0]['status']='RUNNING';statepath.write_text(json.dumps(state))
            with patch.object(plan,'resolve',return_value=cfg):
                with self.assertRaisesRegex(ValueError,'completed local'):
                    assets._select(student,registered,tmp,pins)
                state['runs'][0]['status']='COMPLETED';statepath.write_text(json.dumps(state))
                p=directory/'validation_0001000.json';original=p.read_text();value=json.loads(original)
                value['RRE50']=.001;p.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'full population'):
                    assets._select(student,registered,tmp,pins)
                p.write_text(original)
                p=directory/'models/step_0001000.pt'
                with p.open('ab') as f:f.write(b'corruption')
                with self.assertRaisesRegex(ValueError,'hash/identity'):
                    assets._select(student,registered,tmp,pins)


if __name__=='__main__':unittest.main()
