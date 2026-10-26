"""CPU-only durable deadline, paired admission and checkpoint-safe failure tests."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from panda_hard36 import runtime as R


class Clock:
    def __init__(self):
        self.wall, self.monotonic = 200000., 50000.

    def __call__(self):
        return self.wall

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic += seconds


def configuration(server):
    return {'server':server, 'groups':[],
            'budget':{'total_seconds':129600, 'closeout_seconds':1800,
                      'safety_factor':1.08, 'calibration_reserve_seconds':600},
            'time_prior_seconds_per_update':{'s1':.30, 's3':.20}}


def run_config(case='C30', replica='R1', width=112, horizon=30000):
    return {'phase':'stage2', 'case':case, 'replica':replica,
            'run_id':case+'_'+replica, 'checkpoint_selection':{},
            'planned_horizon':horizon, 'inherited_updates':0, 'start_update':0,
            'evaluation_steps':list(range(1000,horizon+1,1000)),
            'student_architecture':{'bands':8, 'width':width, 'depth':[1,2,3]},
            'engine_kind':'stage2_mature', 'upsampler':'interp23',
            'frame':'MS', 'aligner_outputs':2,
            'student_sampling':{'variant':'UBIL'},
            'student_input':{'variant':'CORRECTED_DIFF'},
            'student_warp':{'method':'lanczos4'},
            'stage2_spec':{'seed':1,'total_updates':horizon,'alpha':.25,'gamma':.5}}


class BudgetRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.clock = Clock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(R.plan,'default_config',side_effect=configuration))
        self.stack.enter_context(patch.object(R,'code_pin',return_value={'synthetic':'a'*64}))
        self.stack.enter_context(patch.object(R,'evaluator_identity',return_value={'synthetic':True}))
        self.stack.enter_context(patch.object(R.time,'monotonic',side_effect=lambda:self.clock.monotonic))
        self.stack.enter_context(patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)))
        self.campaign = self.open_campaign()

    def open_campaign(self, server='s1'):
        return R.Campaign(server,output=self.directory.name,device='cpu',clock=self.clock)

    def test_elapsed_preserves_cost_start_restart_and_clock_rollback(self):
        c = self.campaign
        initial, deadline = c.state['started_unix'], c.deadline_unix
        self.clock.advance(2400)
        c.persist()
        resumed = self.open_campaign()
        self.assertEqual(resumed.state['started_unix'],initial)
        self.assertEqual(resumed.deadline_unix,deadline)
        self.assertEqual(resumed.elapsed_seconds(),2400)
        self.clock.wall -= 600
        self.clock.monotonic += 100
        self.assertEqual(resumed.elapsed_seconds(),2500)
        resumed.persist()
        self.assertEqual(self.open_campaign().elapsed_seconds(),2500)

    def test_deadline_reserves_closeout_and_never_extends_total_budget(self):
        c = self.campaign
        self.clock.advance(127799)
        c.check()
        self.clock.advance(1)
        with self.assertRaisesRegex(R.BudgetExpired,'DURABLE_36H'):
            c.check()
        c.closeout = True
        c.check()
        self.clock.advance(1800)
        with self.assertRaisesRegex(R.BudgetExpired,'DURABLE_36H'):
            c.check()

    def test_stop_and_disk_guards_are_independent_of_deadline(self):
        c = self.campaign
        c.stop_requested = True
        with self.assertRaisesRegex(R.BudgetExpired,'STOP_REQUESTED'):
            c.check()
        c.stop_requested = False
        with patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(R.BudgetExpired,'DISK_CLOSEOUT'):
                c.check()

    def test_timing_reuses_graph_across_horizon_seed_lr_and_hard(self):
        cfg = run_config()
        modified = deepcopy(cfg)
        modified.update(planned_horizon=25000, lr_S=8e-5, replica='R2')
        modified['stage2_spec'].update(seed=2,total_updates=25000,alpha=.125,gamma=.25)
        self.assertEqual(R.Campaign.timing_key(cfg),R.Campaign.timing_key(modified))
        self.assertNotEqual(R.Campaign.timing_key(cfg),R.Campaign.timing_key(run_config(width=128)))
        modified['student_sampling']={'variant':'BASE'}
        self.assertNotEqual(R.Campaign.timing_key(cfg),R.Campaign.timing_key(modified))

    def test_server_width_priors_and_cached_calibration_reserve(self):
        c, cfg = self.campaign, run_config()
        with patch.object(c,'calibration_cached',return_value=False):
            self.assertAlmostEqual(c.estimate_seconds(cfg),(30000*.30+600)*1.08)
            self.assertAlmostEqual(c.estimate_seconds(run_config(width=128)),(30000*.30*1.31+600)*1.08)
        with patch.object(c,'calibration_cached',return_value=True):
            self.assertAlmostEqual(c.estimate_seconds(cfg),30000*.30*1.08)
        s3 = self.open_campaign('s3')
        with patch.object(s3,'calibration_cached',return_value=True):
            self.assertAlmostEqual(s3.estimate_seconds(cfg),30000*.20*1.08)

    def test_observed_timing_overrides_prior_and_includes_eval_save(self):
        c, cfg = self.campaign, run_config()
        c.state['timing'][c.timing_key(cfg)] = {'updates_observed':1000,
            'update_p90':.1,'eval_seconds':20.,'save_seconds':3.}
        with patch.object(c,'calibration_cached',return_value=True):
            self.assertAlmostEqual(c.estimate_seconds(cfg),(30000*.1+30*23)*1.08)
        c.state['timing'][c.timing_key(cfg)]['update_p90']=float('nan')
        with patch.object(c,'calibration_cached',return_value=True):
            self.assertAlmostEqual(c.estimate_seconds(cfg),30000*.30*1.08)

    def test_cache_requires_both_receipt_and_matching_file_hash(self):
        c, cfg=self.campaign,run_config()
        cfg['donor_version']='DONOR'
        c.state['donors']['DONOR']={'binding':{}}
        geometry={'scale':1.}
        key=R.plan.canonical_sha256({'geometry':geometry,'binding':{}})[:24]
        path=c.root/'calibration'/('DONOR_'+key+'.npz')
        path.parent.mkdir()
        path.write_bytes(b'synthetic calibration')
        with patch('panda_hard36.calibration.validate_base_binding',return_value={}), \
                patch('panda_hard36.spec.MatureSpec',return_value=SimpleNamespace(geometry_keys=['s100'])), \
                patch('panda_hard36.spec.geometry_manifest',return_value=geometry):
            self.assertFalse(c.calibration_cached(cfg))
            path.with_suffix('.json').write_text(json.dumps({'file_sha256':R.file_sha(path)}))
            self.assertTrue(c.calibration_cached(cfg))
            path.write_bytes(b'changed calibration')
            with self.assertRaisesRegex(ValueError,'SHA changed'):
                c.calibration_cached(cfg)

    def test_pair_is_rejected_before_either_seed_starts(self):
        c = self.campaign
        group={'name':'pair','optional':False,'runs':[('C30','R1'),('C30','R2')]}
        self.clock.advance(127800-19000)
        with patch.object(R.plan,'resolve_case',side_effect=lambda server,case,replica:run_config(case,replica)), \
                patch.object(c,'calibration_cached',return_value=True):
            self.assertFalse(c.admit_group(group))
        receipt=c.state['groups'][-1]
        self.assertEqual(receipt['pending'],['C30_R1','C30_R2'])
        self.assertEqual(receipt['required_seconds'],19440)
        self.assertEqual(receipt['remaining_seconds'],19000)
        self.assertEqual(receipt['reason'],'INSUFFICIENT_36H_BUDGET')
        self.assertEqual(c.state['runs'],[])

    def test_budget_skips_are_terminal_but_not_complete_queue(self):
        c=self.campaign
        c.config['groups']=[{'name':'pair','optional':False,'runs':[('C30','R1'),('C30','R2')]}]
        c.state['donors'][R.plan.DONORS[0]]={'synthetic':True}
        self.clock.advance(127800-100)
        with patch.object(R.plan,'resolve_case',side_effect=lambda server,case,replica:run_config(case,replica)), \
                patch.object(c,'admission'), patch.object(c,'register_donor'), \
                patch.object(c,'migration_skip',return_value=False), \
                patch.object(c,'calibration_cached',return_value=True), \
                patch.object(c,'run_case') as run:
            result=c.run()
        run.assert_not_called()
        self.assertEqual(result['status'],'FINISHED_WITH_BUDGET_SKIPS')

    def test_failed_stop_checkpoint_cannot_claim_safe_stop(self):
        c, cfg=self.campaign,run_config(horizon=1000)
        cfg['evaluation_steps']=[]
        trainer=SimpleNamespace(updates=1,poisoned=False)
        with patch.object(c,'build',return_value=(trainer,object())), \
                patch.object(c,'resume_artifact',return_value=None), \
                patch.object(R,'recover_trace'), patch('panda_hard36.reporting.emit_status'), \
                patch.object(c,'check',side_effect=R.BudgetExpired('DURABLE_36H_DEADLINE')), \
                patch.object(c,'_save',side_effect=OSError('disk write failed')):
            with self.assertRaisesRegex(RuntimeError,'failed to save'):
                c.run_case(cfg)
        row=c.state['runs'][0]
        self.assertEqual(row['status'],'FAILED')
        self.assertIn('disk write failed',row['stop_checkpoint_error'])
        persisted=json.loads((Path(row['directory'])/'run_state.json').read_text())
        self.assertEqual(persisted['status'],'FAILED')


if __name__=='__main__':
    unittest.main()
