import copy
from pathlib import Path
import tempfile
import unittest
from pan_shared.registry import build_registry, register, validate_registry
from pan_shared.safety import isolated_paths, admit, SafetyStop, control
from pan_shared.preflight import subset_and_probes

class RegistrySafetyTests(unittest.TestCase):
    def test_18_runs_and_frozen_first_stage(self):
        value=build_registry()
        self.assertEqual(len(value['runs']),18)
        self.assertEqual(len({r['run_id'] for r in value['runs']}),18)
        self.assertEqual([(x['run_id'].split('_')[3],x['target_step']) for x in value['initial_queue'][:6]],
            [('C00',50000),('S01',50000),('S02',50000),('S03',50000),('C00',100000),('C00',150000)])
        self.assertIsNone(value['max_global_steps'])
        self.assertEqual(validate_registry(value),value)

    def test_registration_idempotent_and_no_silent_recipe_change(self):
        with tempfile.TemporaryDirectory() as d:
            register(d,build_registry()); register(d,build_registry())
            with self.assertRaises(ValueError): register(d,build_registry(24))

    def test_half_ids_and_train_probe_within_half(self):
        data={'sensors':{s:{'splits':{'train':{'count':n,'sha256':'a'*64},'val':{'count':200,'sha256':'b'*64}}}
             for s,n in [('WV3',9714),('GF2',19809),('QB',17139)]}}
        result=subset_and_probes(data)
        self.assertEqual([result['sensors'][s]['half_count'] for s in ('WV3','GF2','QB')],[4857,9904,8569])
        self.assertEqual(result,subset_and_probes(data))
        for p in result['sensors'].values():
            self.assertEqual(len(p['train_probe_ids']),128)
            self.assertTrue(set(p['train_probe_ids'])<=set(p['half_ids']))

    def test_paths_reject_symlink_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); old=root/'old'; old.mkdir(); project=root/'new'; project.mkdir()
            (project/'link').symlink_to(old,target_is_directory=True)
            with self.assertRaises(SafetyStop): isolated_paths(project,project/'link'/'run',[old])
            with self.assertRaises(SafetyStop): isolated_paths(old,old/'work',[old])

    def test_busy_gpu_never_admitted(self):
        fixture={'server':'s2','hostname':'fixture','disk':{'free':100*1024**3},
            'gpu':{'returncode':0,'stdout':'GPU-fixture'},
            'gpu_processes':{'returncode':0,'stdout':'GPU-fixture, 987654, existing-training, 9000'}}
        with self.assertRaisesRegex(SafetyStop,'WAIT_RESOURCE'): admit(fixture)
        fixture['gpu_processes']['stdout']=''
        self.assertEqual(admit(fixture)['status'],'RESOURCE_AVAILABLE')
        fixture['server']='s1'
        with self.assertRaisesRegex(SafetyStop,'BLOCKED_SERVER'): admit(fixture)

    def test_controls_persist_and_missing_defaults_paused(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(control(d)['action'],'PAUSE')
            control(d,'STOP'); self.assertEqual(control(d)['action'],'STOP')
