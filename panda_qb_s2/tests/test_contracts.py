from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_qb_s2 import plan, contract, runner, failures, reporting, enqueue
from panda_qb_s2.best_publication import project
from panda_qb40.tests.test_runner import FakeCampaign, complete
from panda_sensor_refine.orchestration import sha


def campaign(path):
    c=FakeCampaign(path,-100000);c.server='s2';c.config=plan.default_config('s2');return c


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class ContractTests(unittest.TestCase):
    def test_registered_order_and_seed_pairing(self):
        rows=plan.G.PLAN['queues']['s2']
        self.assertEqual(len(rows),12)
        self.assertEqual(len(plan.DONORS),6)
        self.assertEqual(plan.default_config('s2')['after_campaign'],'STOP_NO_WV3_SUCCESSOR')
        self.assertIsNone(plan.BUDGET['total_seconds'])
        for donor,student in zip(rows[::2],rows[1::2]):
            d=plan.resolve_case('s2',donor['case']);s=plan.resolve_case('s2',student['case'])
            self.assertEqual(s['donor_version'],d['case'])
            self.assertEqual(s['seed'],261018101)
            self.assertEqual((d['planned_horizon'],s['planned_horizon']),(50000,30000))
            for c in (d,s):reporting._config(c)
            self.assertIn('S1seed='+str(d['seed']),reporting.approach_description(s))
            self.assertIn('Sseed=261018101',reporting.approach_description(s))

    def test_same_coefficient_different_seed_donor_is_rejected(self):
        cfg=plan.resolve_case('s2','QB6_S2_A200_E030_D101')
        donor=plan.resolve_case('s2','QB6_S1_A200_E030_D101')
        from panda_qb_s2.stage1 import teacher_binding
        t=teacher_binding(donor)
        binding=dict(sensor='QB',source_campaign_id=plan.CAMPAIGN_ID,
                     band30_teacher=t,mature30_teacher=deepcopy(t),donor_run_id=donor['run_id'])
        contract.validate_donor_loss_pair(cfg,binding)
        binding['donor_run_id']=plan.resolve_case('s2','QB6_S1_A200_E030_D301')['run_id']
        with self.assertRaises(ValueError):contract.validate_donor_loss_pair(cfg,binding)

    def test_full_queue_runs_without_old_donors_or_wallclock_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp,patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),patch('panda_sensor_focus.early_stop.run_case',side_effect=complete):
            c=campaign(tmp)
            c.register_external_donor=lambda: self.fail('External donor accessed')
            result=runner.run(c,catalog='fixture')
            self.assertEqual(result['status'],'FINISHED_REGISTERED_QUEUE')
            self.assertEqual([(r['case'],r['replica']) for r in result['runs']],list(plan.G.RUNS['s2']))
            self.assertEqual(c.registered,list(plan.DONORS))

    def test_completed_stage1_is_not_retrained_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp,patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),patch('panda_sensor_focus.early_stop.run_case',side_effect=complete) as execute:
            c=campaign(tmp);cfg=plan.resolve_case('s2',plan.DONORS[0]);complete(c,cfg,None)
            runner.run(c,catalog='fixture')
            self.assertEqual(execute.call_count,11)
            self.assertEqual(c.registered,list(plan.DONORS))

    def test_support_failure_skips_only_its_student_and_keeps_other_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'config/bglobal_rr/qb_original_lms.json'
            target.parent.mkdir(parents=True);target.write_bytes((plan.ROOT/'config/bglobal_rr/qb_original_lms.json').read_bytes())
            folder=root/plan.G.OUTPUT/plan.CAMPAIGN_ID/'s2/QB'
            cfg=plan.resolve_case('s2',plan.DONORS[0]);directory=folder/'runs'/cfg['run_id']/'attempt_01'
            row=dict(run_id=cfg['run_id'],case=cfg['case'],replica='R1',phase='donor',directory=str(directory),
                     status='FAILED',step=12000,poisoned=False,error=failures.SUPPORT_ERROR)
            for name,value in [('config',cfg),('requested_config',cfg),('run_state',row)]:dump(directory/(name+'.json'),value)
            (directory/'train.jsonl').write_text('{"step":12000}\n')
            cp=directory/'models/stop_0012000.pt';cp.parent.mkdir();cp.write_bytes(b'fixture, not a trained checkpoint')
            dump(directory/'models/latest_resume.json',dict(checkpoint_path=str(cp),file_sha256=sha(cp),
                 completed_update=12000,campaign_resume_complete=True,checkpoint_verified=True))
            c=campaign(folder);c.state.update(runs=[deepcopy(row)],started_unix=1000)
            with patch.object(plan,'ROOT',root),patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),patch('panda_sensor_focus.early_stop.run_case',side_effect=complete) as execute:
                result=runner.run(c,catalog='fixture')
                self.assertEqual(execute.call_count,10)
                self.assertEqual(len(result['blocked_runs']),1)
                self.assertEqual(result['status'],failures.STATUS)
                self.assertIn('QB6_S1_A200_E030_D301',c.registered)
                cp.write_bytes(b'changed')
                with self.assertRaises(ValueError):failures.verify(c,row)

    def test_unknown_failure_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp,patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),patch('panda_sensor_focus.early_stop.run_case',side_effect=ValueError('bad source hash')):
            c=campaign(tmp)
            with self.assertRaisesRegex(ValueError,'source hash'):runner.run(c,catalog='fixture')
            self.assertEqual(c.state['status'],'FAILED')

    def test_sheet_target_and_distinct_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=project(tmp,'s2')
            self.assertFalse(value['conflicts'])
            self.assertEqual(value['destination']['sheet_title'],'QB-ablations')
            p=value['publication_thresholds']
            self.assertEqual((p['hqnr_min'],p['ergas_max']),(.9315,3.5))
            self.assertTrue(p['publish_all_joint_passes'])
            self.assertTrue(p['deduplicate_representative_and_pass'])
        self.assertEqual(reporting.RECORD_PREFIX,'PQS2-')

    def test_fresh_container_mounts_and_identity_without_previous_runtime(self):
        image={'Id':'sha256:fixture','Config':{'Env':['PATH=/bin']}}
        with tempfile.TemporaryDirectory() as tmp,patch.object(enqueue,'ROOT',Path(tmp)),patch.object(enqueue,'inspect',return_value=image),patch('edgeres48.docker_mounts.targets',return_value=[Path('/data/qb.h5')]):
            args,identity=enqueue.command('image',name=enqueue.NAME,token='token',source_sha256='x'*64)
            self.assertEqual(sum(m['RW'] for m in identity['mounts']),1)
            self.assertEqual(identity['mounts'][-1]['Destination'],'/workspace/'+plan.G.OUTPUT)
            current={'Image':identity['image'],'Mounts':identity['mounts'],'Config':{
                'Cmd':enqueue.training_args(),'User':identity['user'],'WorkingDir':'/workspace',
                'Env':identity['env'],'Labels':{'panda.campaign':plan.CAMPAIGN_ID,'panda.launch-token':'token','panda.source':'x'*64}}}
            receipt={'identity':identity,'token':'token','source_sha256':'x'*64}
            enqueue.authenticate(current,receipt)
            bad=deepcopy(current);bad['Mounts'][0]['RW']=True
            with self.assertRaises(ValueError):enqueue.authenticate(bad,receipt)
