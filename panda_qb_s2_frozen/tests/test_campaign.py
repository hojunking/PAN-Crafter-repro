from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_qb_s2_frozen import plan,contract,registry as G,runner,reporting,enqueue,donor
from panda_qb_s2_frozen.best_publication import project,verify_owned
from panda_qb_s2_frozen.tests.fixtures import persist
from panda_qb40.tests.test_runner import FakeCampaign,complete
from panda_qb_s2 import plan as original
from panda_qb_s2.stage1 import teacher_binding


class CampaignTests(unittest.TestCase):
    def test_queue_seed_recipe_and_source_protocol(self):
        from panda_qb_s2_frozen.cost.profiling import registered_profiles
        profiles=registered_profiles('s2')
        self.assertEqual(len(profiles),2)
        self.assertEqual({c['student_architecture']['width'] for c,_ in profiles},{112,128})
        self.assertTrue(all(c['phase']=='stage2' for c,_ in profiles))
        self.assertEqual(len(G.RUNS['s2']),22)
        self.assertEqual(len(set(G.RUNS['s2'])),22)
        self.assertEqual(plan.DONORS,())
        self.assertEqual(plan.default_config('s2')['stage1_runs'],0)
        self.assertEqual([r for _,r in G.RUNS['s2']],['R1']*6+['R2']*8+['R3']*8)
        self.assertNotIn(('Q0_BASE','R1'),G.RUNS['s2'])
        self.assertNotIn(('Q6_D2_BASE','R1'),G.RUNS['s2'])
        for c,r in G.RUNS['s2']:
            cfg=plan.resolve_case('s2',c,r);reporting._config(cfg)
            pin=G.donor(cfg['donor_version'])
            self.assertEqual(cfg['seed'],G.SEEDS[r])
            self.assertEqual(cfg['stage2_spec']['beta'],0.)
            self.assertEqual(cfg['shift_scale'],1.)
            self.assertFalse(cfg['stage1_retrained'])
            self.assertEqual(cfg['phase'],'stage2')
            self.assertEqual(pin['selected_step'],48000)
            self.assertIn('/s2/QB/',pin['checkpoint_path'])
            self.assertEqual(cfg['baseline_resolved'],original.baseline('s2'))
            if r=='R1':
                old=original.resolve_case('s2',pin['source_case'].replace('_S1_','_S2_'))
                self.assertEqual(cfg['stream_seeds'],old['stream_seeds'])
                self.assertEqual(cfg['augmentation'],old['augmentation'])
        with self.assertRaises(ValueError):plan.default_config('s3')

    def test_exact_selected_donor_identity_is_required(self):
        cfg=plan.resolve_case('s2','Q3_W128H050');pin=G.donor('D4')
        source=original.resolve_case('s2',pin['source_case']);t=teacher_binding(source)
        binding=dict(sensor='QB',source_campaign_id=source['campaign_id'],
            band30_teacher=t,mature30_teacher=deepcopy(t),donor_run_id=source['run_id'],
            donor_step=48000,donor_checkpoint_sha256=pin['checkpoint_sha256'],
            mature30_selector_sha256=pin['selector_receipt_sha256'])
        contract.validate_donor_loss_pair(cfg,binding)
        for key,bad in [('donor_run_id',source['run_id'].replace('S2','S3')),
                        ('donor_step',50000),('donor_checkpoint_sha256','0'*64),
                        ('mature30_selector_sha256','0'*64)]:
            value=deepcopy(binding);value[key]=bad
            with self.assertRaises(ValueError):contract.validate_donor_loss_pair(cfg,value)
        for path in ('/workspace/work_dir/panda_qb_s2/x/s3/QB/step.pt','/other/step.pt'):
            with self.assertRaises(ValueError):donor.local(path,Path('/repo'))

    def test_missing_sources_never_fall_back_to_training(self):
        with tempfile.TemporaryDirectory() as tmp,patch('panda_qb_s2.stage1.build_stage1') as build:
            with self.assertRaises(FileNotFoundError):donor.verify_sources(tmp)
            build.assert_not_called()

    def test_full_student_queue_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp,patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),patch('panda_sensor_focus.early_stop.run_case',side_effect=complete) as execute:
            camp=FakeCampaign(tmp,-100000);camp.server='s2';camp.config=plan.default_config('s2')
            camp.register_donor=lambda *args:self.fail('Stage1 registered')
            state=runner.run(camp,catalog='fixture')
            self.assertEqual(state['status'],'FINISHED_REGISTERED_QUEUE')
            self.assertEqual(execute.call_count,22)
            self.assertEqual([(r['case'],r['replica']) for r in state['runs']],list(G.RUNS['s2']))
            runner.run(camp,catalog='fixture')
            self.assertEqual(execute.call_count,22)

    def test_unknown_student_error_stops_with_evidence(self):
        with tempfile.TemporaryDirectory() as tmp,patch('panda_sensor_focus.early_stop.finish_stopped',return_value=False),patch('panda_sensor_focus.early_stop.run_case',side_effect=ValueError('source mismatch')):
            camp=FakeCampaign(tmp,-100000);camp.server='s2';camp.config=plan.default_config('s2')
            with self.assertRaisesRegex(ValueError,'source mismatch'):runner.run(camp,catalog='fixture')
            self.assertEqual(camp.state['status'],'FAILED')

    def test_raw_thresholds_rounded_display_and_pass_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            persist(tmp,step=1000,h=.9314999,e=3.4999)
            view=project(tmp,'s2')
            self.assertFalse(view['conflicts'],view['conflicts'])
            self.assertEqual(view['rows'][0][reporting.INDEX['Selector']],'BEST_HQNR_FALLBACK')
            verify_owned(view['rows'][0],view)
            persist(tmp,step=2000,h=.9315001,e=3.5000001)
            persist(tmp,step=3000,h=.94,e=3.49999)
            persist(tmp,step=4000,h=.939,e=3.49)
            view=project(tmp,'s2')
            self.assertFalse(view['conflicts'],view['conflicts'])
            self.assertEqual(len(view['rows']),2)
            self.assertEqual({r[reporting.INDEX['Selected_step']] for r in view['rows']},{3000,4000})
            for row in view['rows']:
                verify_owned(row,view)
                self.assertTrue(row[reporting.INDEX['Run_ID']].startswith('QFS2_'))
                self.assertIn('frozen own s2 Stage1@48K',row[reporting.INDEX['Approach']])
                for v in row[3:15]:
                    if isinstance(v,(int,float)):self.assertEqual(v,round(v,3))

    def test_source_mount_is_read_only_and_output_is_distinct(self):
        image={'Id':'sha256:fixture','Config':{'Env':['PATH=/bin']}}
        with tempfile.TemporaryDirectory() as tmp,patch.object(enqueue,'ROOT',Path(tmp)),patch.object(enqueue,'inspect',return_value=image),patch('edgeres48.docker_mounts.targets',return_value=[]):
            (Path(tmp)/'work_dir/panda_qb_s2').mkdir(parents=True)
            _,identity=enqueue.command('image',name=enqueue.NAME,token='token',source_sha256='a'*64)
            mounts=identity['mounts']
            source=next(m for m in mounts if m['Destination']=='/workspace/work_dir/panda_qb_s2')
            self.assertFalse(source['RW'])
            writable=[m for m in mounts if m['RW']]
            self.assertEqual(len(writable),1)
            self.assertEqual(writable[0]['Destination'],'/workspace/'+G.OUTPUT)
