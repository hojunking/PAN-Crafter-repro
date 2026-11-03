from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from panda_wv3_ablation import plan as legacy_plan
from panda_wv3_main_rgb import enqueue, plan


class LauncherTests(unittest.TestCase):
    def info(self,campaign=None,running=True,container_id='legacy-id'):
        return {'Id':container_id,'Config':{'Labels':{'panda.campaign':campaign or enqueue.LEGACY_CAMPAIGN,
            'panda.server':'s1','panda.source':'source'}},
            'Mounts':[{'Destination':'/workspace','Source':str(plan.ROOT)}],
            'State':{'Running':running,'ExitCode':0}}

    def fixture(self,root,status='COMPLETED'):
        cfg=legacy_plan.resolve('s1','S2_B3_EQ_ONLY',3)
        directory=root/'runs'/cfg['run_id'];(directory/'models').mkdir(parents=True)
        enqueue.atomic_json(directory/'config.json',cfg)
        target={'run_id':cfg['run_id'],'container_id':'legacy-id','horizon':cfg['horizon'],
                'config_sha256':cfg['config_sha256'],'config_file_sha256':enqueue.sha(directory/'config.json')}
        checkpoint=directory/'models'/f"step_{cfg['horizon']:07d}.pt";checkpoint.write_bytes(b'retained complete checkpoint')
        meta={'path':str(checkpoint),'step':cfg['horizon'],'file_sha256':enqueue.sha(checkpoint),
              'config_sha256':cfg['config_sha256'],'model_sha256':'model'}
        enqueue.atomic_json(checkpoint.with_suffix('.json'),meta)
        record={'schema':'WV3_ABLATION_EVAL_V1','run_id':cfg['run_id'],'config_sha256':cfg['config_sha256'],
                'step':cfg['horizon'],'same_state_verified':True,'model_sha256':'model','checkpoint':meta}
        for split in ('rr','fr'):record[split]={'per_scene':[{'scene_id':f'WV3:{split}:{i}'} for i in range(20)]}
        enqueue.atomic_json(directory/f"evaluation_{cfg['horizon']:07d}.json",record)
        state={'campaign_id':enqueue.LEGACY_CAMPAIGN,'server':'s1','status':'RUNNING','runs':[
            {'case':cfg['case_id'],'repeat':3,'run_id':cfg['run_id'],'status':status,'step':cfg['horizon']}]}
        enqueue.atomic_json(root/'campaign_state.json',state)
        return target,state,checkpoint

    def args(self,**kw):
        value=dict(image='image',dlpan=None,after_current_ablation=True,target_run=None,wait=True)
        value.update(kw);return SimpleNamespace(**value)

    def request(self,target=None):
        return {'schema':enqueue.REQUEST_SCHEMA,'source_sha256':'source','image_id':'image-id',
                'target':target,'dlpan':None,'wait':True}

    def test_completed_means_full_evaluation_and_retained_hash_not_just_final_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target,state,checkpoint=self.fixture(root,'RUNNING')
            with patch.object(enqueue,'legacy_root',return_value=root):
                self.assertFalse(enqueue.target_completed(target))
                state['runs'][0]['status']='COMPLETED';enqueue.atomic_json(root/'campaign_state.json',state)
                self.assertTrue(enqueue.target_completed(target))
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'evidence'):enqueue.target_completed(target)
                state['runs'][0]['status']='FAILED';enqueue.atomic_json(root/'campaign_state.json',state)
                with self.assertRaisesRegex(RuntimeError,'FAILED'):enqueue.target_completed(target)

    def test_explicit_completed_target_does_not_follow_the_new_active_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target,state,_=self.fixture(root)
            state['runs'].append({'case':'NEXT','repeat':3,'run_id':'next','status':'RUNNING','step':100})
            enqueue.atomic_json(root/'campaign_state.json',state)
            with patch.object(enqueue,'legacy_root',return_value=root), \
                 patch.object(enqueue,'inspect_container',return_value=self.info()):
                self.assertEqual(enqueue.capture_target(target['run_id']),target)

    def test_check_only_and_failed_preflight_never_capture_or_stop_training(self):
        argv=['--server','s1','--image','image','--after-current-ablation']
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(enqueue,'handoff_root',return_value=Path(tmp)/'handoff'), \
             patch.object(enqueue,'release_check',return_value='source'), \
             patch.object(enqueue,'inspect_container',return_value=None), \
             patch.object(enqueue,'capture_target') as capture, \
             patch.object(enqueue,'stop_legacy') as stop, \
             patch.object(enqueue,'preflight',return_value='image-id') as preflight:
            enqueue.main(argv+['--check-only'])
            capture.assert_not_called();stop.assert_not_called()
            self.assertFalse((Path(tmp)/'handoff/request.json').exists())
            preflight.side_effect=RuntimeError('preflight failed')
            with self.assertRaisesRegex(RuntimeError,'preflight'):enqueue.main(argv+['--enqueue'])
            capture.assert_not_called();stop.assert_not_called()
            self.assertFalse((Path(tmp)/'handoff/request.json').exists())

    def test_waits_for_target_then_stops_only_legacy_and_starts_main(self):
        events=[];target={'run_id':'original','container_id':'legacy-id'}
        def complete(t):
            events.append('check-target');return events.count('check-target')>1
        def inspect(name):
            if name==enqueue.LEGACY_NAME:return self.info()
            return self.info(campaign=plan.CAMPAIGN_ID,container_id='main-id') if 'start-main' in events else None
        def run(cmd,**kw):
            if cmd[0]=='fake-docker':events.append('start-main')
            return SimpleNamespace(stdout='')
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(enqueue,'handoff_root',return_value=Path(tmp)), \
             patch.object(enqueue,'release_check',return_value='source'), \
             patch.object(enqueue,'target_completed',side_effect=complete), \
             patch.object(enqueue,'inspect_container',side_effect=inspect), \
             patch.object(enqueue,'stop_legacy',side_effect=lambda t:events.append('stop-legacy') or {'ok':True}), \
             patch.object(enqueue,'command',return_value=['fake-docker']), \
             patch.object(enqueue.subprocess,'run',side_effect=run), \
             patch.object(enqueue.time,'sleep'):
            result=enqueue.execute(self.request(target))
        self.assertEqual(result['status'],'STARTED')
        self.assertLess(events.index('stop-legacy'),events.index('start-main'))
        self.assertEqual(events[:2],['check-target','check-target'])

    def test_no_cutover_when_target_container_stops_or_fails(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(enqueue,'handoff_root',return_value=Path(tmp)), \
             patch.object(enqueue,'release_check',return_value='source'), \
             patch.object(enqueue,'target_completed',return_value=False), \
             patch.object(enqueue,'inspect_container',side_effect=[None,self.info(running=False)]), \
             patch.object(enqueue,'stop_legacy') as stop:
            with self.assertRaisesRegex(RuntimeError,'stopped before'):enqueue.execute(self.request({'container_id':'legacy-id'}))
            stop.assert_not_called()

    def test_existing_main_is_idempotent_and_never_stops_legacy(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(enqueue,'handoff_root',return_value=Path(tmp)), \
             patch.object(enqueue,'release_check',return_value='source'), \
             patch.object(enqueue,'inspect_container',return_value=self.info(campaign=plan.CAMPAIGN_ID)), \
             patch.object(enqueue,'stop_legacy') as stop:
            self.assertEqual(enqueue.execute(self.request({'run_id':'original'}))['status'],'RUNNING')
            stop.assert_not_called()

    def test_wrong_container_identity_cannot_be_stopped(self):
        with patch.object(enqueue,'target_completed',return_value=True), \
             patch.object(enqueue,'inspect_container',return_value=self.info(container_id='replaced')), \
             patch.object(enqueue.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'identity changed'):
                enqueue.stop_legacy({'container_id':'legacy-id'})
            run.assert_not_called()
        info=self.info();info['Config']['Labels']['panda.server']='s3'
        with self.assertRaisesRegex(ValueError,'intended s1'):enqueue.verify_container(info,campaign=enqueue.LEGACY_CAMPAIGN)

    def test_registration_keeps_original_target_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(enqueue,'handoff_root',return_value=Path(tmp)), \
             patch.object(enqueue,'capture_target',return_value={'run_id':'original'}) as capture:
            first=enqueue.registered_request(self.args(),'source','image-id')
            self.assertEqual(enqueue.registered_request(self.args(),'source','image-id')['target'],first['target'])
            self.assertEqual(capture.call_count,1)
            with self.assertRaisesRegex(ValueError,'another legacy run'):
                enqueue.registered_request(self.args(target_run='next'),'source','image-id')
            with self.assertRaisesRegex(ValueError,'different source'):
                enqueue.registered_request(self.args(),'new-source','image-id')

    def test_second_supervisor_is_not_spawned_while_original_holds_file_lease(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(enqueue,'handoff_root',return_value=Path(tmp)), \
             patch.object(enqueue,'release_check',return_value='source'), \
             patch.object(enqueue,'inspect_container',return_value=None), \
             patch.object(enqueue,'preflight') as preflight, \
             patch.object(enqueue.subprocess,'Popen') as spawn:
            with (Path(tmp)/'supervisor.lock').open('a') as locked:
                fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
                enqueue.main(['--server','s1','--image','image','--after-current-ablation','--enqueue'])
            preflight.assert_not_called();spawn.assert_not_called()


if __name__=='__main__':unittest.main()
