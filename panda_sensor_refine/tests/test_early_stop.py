import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from panda_sensor_refine import early_stop as E
from panda_sensor_refine import orchestration as O

class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'runs' / 'test' / 'attempt_01'
        (self.directory / 'models').mkdir(parents=True)
        self.cfg = dict(run_id='test', dataset='QB', phase='stage2', seed=1, attempt=1, planned_horizon=30000)
        self.write_window([.89, .87, .88])

    def write_window(self, values):
        for step, value in zip(E.STEPS, values):
            ident = dict(self.cfg, completed_update=step, model_state_sha256='state'+str(step),
                         evaluator_revision='eval', inference_policy_digest='infer', alignment_protocol_digest='align')
            ev = {k:ident for k in ('state_before','state_after','rr_state','fr_state')}
            ev['mature30_deployment'] = dict(completed_update=step, geometry='unchanged')
            ev.update(full_method_eligible=True, metric_function_injection=False,
                      qualification=dict(same_state_verified=True, metrics_complete=True, metrics_finite=True, HQNR=value,
                          scene_completeness={s:dict(completed=20, expected=20, duplicates=[], missing=[], unexpected=[]) for s in ('rr','fr')}))
            for split in ('rr','fr'):
                ev[split] = dict(metrics={'HQNR':value}, per_scene=[dict(scene_id=f'QB:{split}:{i}', metrics={'HQNR':value}) for i in range(20)])
            model = self.directory/'models'/f'step_{step:07d}.pt'
            model.write_bytes(b'synthetic non-model fixture')
            meta = dict(config=self.cfg, evaluation=ev, artifact=dict(completed_update=step, checkpoint_verified=True,
                        model_state_sha256=ident['model_state_sha256'], file_sha256=E.sha(model)))
            E.atomic_json(self.directory/f'evaluation_{step:07d}.json', ev)
            E.atomic_json(model.with_suffix('.json'), meta)

    def test_strict_window_and_scope(self):
        self.assertTrue(E.decision(self.directory,self.cfg)['stop'])
        for i in range(3):
            values=[.89,.87,.88];values[i]=.9;self.write_window(values)
            self.assertFalse(E.decision(self.directory,self.cfg)['stop'])
        for cfg in (dict(self.cfg,dataset='GF2'),dict(self.cfg,dataset='WV3'),dict(self.cfg,phase='donor')):
            self.assertIsNone(E.decision(self.directory,cfg))

    def test_missing_evaluation_cannot_trigger(self):
        (self.directory/'evaluation_0009000.json').unlink()
        self.assertIsNone(E.decision(self.directory,self.cfg))

    def test_tampering_does_not_trigger_performance_stop(self):
        path=self.directory/'evaluation_0009000.json'; original=E.read(path)
        for key,value in [('metric_function_injection',True),('state_before',{}),('full_method_eligible',False)]:
            ev=copy.deepcopy(original);ev[key]=value;E.atomic_json(path,ev)
            with self.assertRaises(ValueError):E.decision(self.directory,self.cfg)
        E.atomic_json(path,original)
        (self.directory/'models/step_0009000.pt').write_bytes(b'modified')
        with self.assertRaises(ValueError):E.decision(self.directory,self.cfg)

    def test_incomplete_scenes_and_nan_rejected(self):
        for edit in ('scene','nan','protocol'):
            self.write_window([.89,.87,.88])
            path=self.directory/'evaluation_0009000.json';ev=E.read(path)
            if edit=='scene':ev['fr']['per_scene'][0]['scene_id']='QB:fr:1'
            elif edit=='nan':ev['fr']['metrics']['HQNR']=float('nan')
            else:
                for key in ('state_before','state_after','rr_state','fr_state'):ev[key]['inference_policy_digest']='changed'
            path.write_text(json.dumps(ev));mp=self.directory/'models/step_0009000.json';meta=E.read(mp);meta['evaluation']=ev;mp.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):E.decision(self.directory,self.cfg)

    def camp(self,status='STOPPED',step=13549):
        row=dict(run_id='test',directory=str(self.directory),status=status,step=step,poisoned=False)
        E.atomic_json(self.directory/'run_state.json',row)
        model=self.directory/'models'/f'stop_{step:07d}.pt';model.write_bytes(b'complete safe checkpoint fixture')
        E.atomic_json(self.directory/'models/latest_resume.json',dict(checkpoint_path=str(model),completed_update=step,
                        campaign_resume_complete=True,checkpoint_verified=True,file_sha256=E.sha(model)))
        class Fake:
            def persist(self):pass
        camp=Fake();camp.state={'runs':[row]};camp.root=self.directory.parents[2]
        return camp,row

    def safe(self,row,cfg,root=None):
        return O.authenticate_stop(row,cfg,Path(self.temp.name))

    def test_safe_stop_recovery_preserves_actual_step_and_idempotence(self):
        camp,row=self.camp()
        with patch.object(E,'authenticate_stop',side_effect=self.safe),patch.object(E,'emit_status') as emit:
            self.assertTrue(E.finish_stopped(camp,self.cfg))
            self.assertEqual(row['status'],E.STATUS);self.assertEqual(row['step'],13549)
            original=E.read(self.directory/'early_stop.json')
            self.assertTrue(E.finish_stopped(camp,self.cfg))
            self.assertEqual(E.read(self.directory/'early_stop.json'),original)
            self.assertFalse(original['full_horizon_completed']);self.assertEqual(emit.call_count,2)
            (self.directory/'models/latest_resume.json').unlink()
            with self.assertRaises(FileNotFoundError):E.finish_stopped(camp,self.cfg)

    def test_completed_and_failed_runs_are_not_reclassified(self):
        for status in ('COMPLETED','FAILED','RUNNING'):
            camp,row=self.camp(status=status)
            self.assertFalse(E.finish_stopped(camp,self.cfg));self.assertEqual(row['status'],status)

    def test_gate_uses_runtime_safe_stop_and_restores_check(self):
        camp,row=self.camp(status='RUNNING',step=9999)
        class BudgetError(RuntimeError):pass
        original_check=lambda **kwargs:None
        camp.check=original_check
        def run(cfg):
            row['step']=10000
            try:camp.check()
            except BudgetError:
                row['status']='STOPPED'
                E.atomic_json(self.directory/'run_state.json',row)
                p=self.directory/'models/latest_resume.json';a=E.read(p);a['completed_update']=10000;E.atomic_json(p,a)
                raise
        camp.run_case=run
        with patch.object(E,'authenticate_stop',side_effect=self.safe),patch.object(E,'emit_status'):
            E.run_case(camp,self.cfg,BudgetError)
        self.assertIs(camp.check,original_check);self.assertEqual(row['status'],E.STATUS)

    def test_external_signal_never_becomes_futility_completion(self):
        camp,row=self.camp(status='RUNNING',step=10000)
        class BudgetError(RuntimeError):pass
        def check(**kwargs):raise BudgetError('STOP_REQUESTED')
        camp.check=check;camp.run_case=lambda cfg:camp.check()
        with self.assertRaisesRegex(BudgetError,'STOP_REQUESTED'):E.run_case(camp,self.cfg,BudgetError)
        self.assertIs(camp.check,check);self.assertNotIn('early_stopped_runs',camp.state)

    def test_legacy_accounting_keeps_early_stop_separate_from_completed(self):
        camp,row=self.camp();row['status']=E.STATUS
        cfg=dict(self.cfg,loss_pair='pair')
        with patch.object(O,'legacy_configs',return_value=[cfg]),patch.object(O,'legacy_root',return_value=self.directory), \
                patch.object(O,'read',return_value=camp.state),patch.object(E,'verify',return_value={'proof':'stop'}):
            value=O.build_completion(Path(self.temp.name),'s1',['test'])
        self.assertEqual(value['completed'],[]);self.assertEqual(value['early_stopped'],[{'proof':'stop'}])
        self.assertFalse(value['all_original_runs_completed'])

    def test_refine_completion_and_process_exit_allow_wv3_successor(self):
        from panda_sensor_refine import queue,plan,runtime,publication,__main__ as cli
        cfg=dict(self.cfg,case='S',replica='R1')
        state=dict(status='FINISHED_WITH_EXPLICIT_EARLY_STOPS',runs=[dict(run_id='test',status=E.STATUS)],
                   early_stopped_runs={'test':{'proof':'stop'}})
        with patch.object(queue,'read',return_value=state),patch.object(plan,'groups',return_value=[{'runs':[('S','R1')]}]), \
                patch.object(plan,'resolve_case',return_value=cfg),patch.object(E,'verify',return_value={'proof':'stop'}), \
                patch.object(queue,'predecessor_receipt',return_value={'external_donor':{'status':'COMPLETED'}}):
            self.assertTrue(queue.core_complete('s1'))
        with patch('directshift48.__main__.require_idle_gpu'),patch.object(runtime,'Campaign') as campaign:
            campaign.return_value.run.return_value=state;campaign.return_value.root='/tmp/test'
            self.assertEqual(cli.main(['run','--server','s1']),0)
        E.atomic_json(self.directory/'campaign_state.json',state)
        self.assertEqual(publication._terminal_state(self.directory)['status'],state['status'])

if __name__=='__main__':unittest.main()
