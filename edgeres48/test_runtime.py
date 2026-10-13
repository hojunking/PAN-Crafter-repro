"""Controller contracts with mocked trainers/data; no real data or training."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import plan, runtime
from .reporting import atomic_json
from .test_plan import baseline, evaluation, numeric


@contextmanager
def campaign():
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(runtime,'evaluator_identity',return_value={'test':'synthetic'}), \
         patch.object(runtime,'code_pin',return_value={'test':'abc'}):
        c=runtime.Campaign(directory,'s1',baseline(),device='cpu')
        c.state.update(started_unix=time.time(),timing={'update_p90_seconds':.1,
            'rr_fr_eval_seconds':1.,'probe_seconds':2.,'artifact_seconds':3.})
        c.sources=SimpleNamespace()
        (c.root/'method_source').mkdir(exist_ok=True)
        atomic_json(c.root/'method_source/manifest.json',{'synthetic':True})
        yield c


def completed(loss='EF00', *, h=(.959,.959,.959), e=(2.1,2.08,2.06), schedule='H50', et=None):
    c=plan.resolve_case('s1',baseline=baseline(),loss_case=loss,optimizer_case=et,schedule=schedule)
    end=plan.HORIZONS[schedule]
    rows=[]
    for i,(hq,erg,step) in enumerate(zip(h,e,(end-4000,end-2000,end))):
        o=numeric(hq,erg,step,c['run_id'],hq>=.958 and erg<=2.04)
        o.update(plan.target_gap(hq,erg,server='s1'))
        o.update(record_id='EGR2-'+str(i)*64,full_method_eligible=True)
        rows.append(o)
    return {**c,'status':'COMPLETED','completed_update':end,'evaluations':rows}


class ControllerTests(unittest.TestCase):
    def test_probe_spec_uses_pinned_ids_not_runtime_defaults(self):
        cfg=plan.resolve_case('s1',baseline=baseline())
        cfg['baseline_resolved']['probes']['rows']['train']=[7,8]
        self.assertEqual(runtime.probe_spec(cfg)['rows']['train'],[7,8])
        self.assertEqual(runtime.probe_spec(cfg)['response_domains'],['normal'])
        self.assertEqual(runtime.probe_spec(cfg,clean=True)['response_domains'],['normal','clean_support'])

    def test_code_change_rejected(self):
        with campaign() as c,patch.object(runtime,'code_pin',return_value={'test':'changed'}):
            with self.assertRaisesRegex(RuntimeError,'code changed'):c.verify_code()

    def test_admission_registry_and_override_hash_pins_before_data(self):
        with campaign() as c,patch.object(runtime,'EdgeDataSources') as sources:
            c.baseline['source_registry_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'registry differs'):
                c.admission(runtime.ROOT/'ablr2/sensor_sources.json')
            sources.assert_not_called()

    def test_failed_and_disposable_evaluation_never_retains(self):
        for disposable in (False,True):
            with campaign() as c:
                cfg=plan.resolve_case('s1',baseline=baseline())
                trainer=SimpleNamespace(updates=1000,stage=object())
                record=evaluation(h=.97 if disposable else .95,e=2.)
                record['qualification']=plan.qualify_evaluation(record,server='s1',
                    expected_rr_ids=[f'rr{i}' for i in range(20)],expected_fr_ids=[f'fr{i}' for i in range(20)])
                record['elapsed_seconds']=1
                with patch.object(runtime,'official_evaluation',return_value=(record,{})), \
                     patch.object(runtime,'retain_qualified') as retain,patch.object(c,'emit',return_value='record'):
                    out=c.evaluate(trainer,cfg,Mock(),c.root,eligible=not disposable)
                    self.assertFalse(out['joint_pass'])
                    retain.assert_not_called()

    def test_full_qualified_only_retains_and_cost_failure_does_not_discard_gate(self):
        with campaign() as c:
            cfg=plan.resolve_case('s1',baseline=baseline())
            trainer=SimpleNamespace(updates=1000,stage=object())
            record=evaluation();record['elapsed_seconds']=1
            record['qualification']=plan.qualify_evaluation(record,server='s1',
                expected_rr_ids=[f'rr{i}' for i in range(20)],expected_fr_ids=[f'fr{i}' for i in range(20)])
            with patch.object(runtime,'official_evaluation',return_value=(record,{})), \
                 patch.object(runtime,'retain_qualified',return_value={'checkpoint_file_sha256':'a'*64}) as retain, \
                 patch.object(c,'emit',return_value='record') as emit, \
                 patch.object(c,'benchmark',side_effect=RuntimeError('synthetic cost failure')):
                out=c.evaluate(trainer,cfg,Mock(),c.root)
                self.assertTrue(out['joint_pass']);retain.assert_called_once()
                self.assertTrue(any(call.args[1]=='COST_FAILED' for call in emit.call_args_list))

    def test_artifact_failure_is_reported_not_claimed_saved(self):
        with campaign() as c:
            cfg=plan.resolve_case('s1',baseline=baseline());record=evaluation()
            record['qualification']=plan.qualify_evaluation(record,server='s1',
                expected_rr_ids=[f'rr{i}' for i in range(20)],expected_fr_ids=[f'fr{i}' for i in range(20)])
            with patch.object(runtime,'official_evaluation',return_value=(record,{})), \
                 patch.object(runtime,'retain_qualified',side_effect=OSError('synthetic disk full')), \
                 patch.object(c,'emit',return_value='record') as emit:
                with self.assertRaises(OSError):c.evaluate(SimpleNamespace(updates=1000),cfg,Mock(),c.root)
                self.assertTrue(any(call.args[1]=='TARGET_PASS_ARTIFACT_FAILED' for call in emit.call_args_list))

    def test_emit_writes_local_and_preserves_attempt_and_native_metrics(self):
        with campaign() as c:
            cfg=plan.resolve_case('s1',baseline=baseline())
            rid=c.emit(cfg,'OBSERVED',scope='EVAL_SUMMARY',metrics={'HQNR':.95,'ERGAS':2.1,'SCC':.98},
                state_sha='a'*64,extra={'same_state_verified':True})
            record=json.loads((c.outbox.outbox/(rid+'.json')).read_text())
            self.assertEqual(record['source']['fields']['Attempt'],1)
            self.assertEqual(record['source']['metrics']['ERGAS'],2.1)
            self.assertEqual(record['destination']['sheet_title'],'WV3-ablations')
            self.assertTrue((c.root/'records.jsonl').exists())

    def test_raw_eta_includes_all_possible_artifacts_and_probes(self):
        with campaign() as c:
            cfg=plan.resolve_case('s1',baseline=baseline())
            self.assertAlmostEqual(c.estimate(cfg),50000*.1+27*1+27*3+4*2)

    def test_preflight_retains_canonical_runid_and_disposable_flag(self):
        with campaign() as c:
            trainer=SimpleNamespace(updates=300)
            c.latencies.extend([.1]*20)
            seen=[]
            def build(cfg,*args):
                plan.validate_method_config(cfg);seen.append(cfg);return trainer
            def diagnostics(*args,**kwargs):
                atomic_json(c.root/'preflight/diagnostics_0000300.json',{'status':'COMPLETE'})
                return 2
            with patch.object(runtime,'build_trainer',side_effect=build),patch.object(runtime,'RRStream'), \
                 patch.object(c,'trace'),patch.object(c,'evaluate') as evaluate, \
                 patch.object(c,'diagnostics',side_effect=diagnostics),patch.object(c,'benchmark'),patch.object(c,'emit'):
                c.preflight()
            self.assertTrue(seen[0]['disposable_preflight'])
            self.assertFalse(evaluate.call_args.kwargs['eligible'])

    def test_run_failure_record_and_no_implicit_recovery_weights(self):
        with campaign() as c:
            cfg=plan.resolve_case('s1',baseline=baseline())
            fake=SimpleNamespace(updates=0)
            with patch.object(runtime,'build_trainer',return_value=fake),patch.object(runtime,'RRStream'), \
                 patch.object(c,'diagnostics'),patch.object(c,'trace',side_effect=RuntimeError('synthetic update')), \
                 patch.object(c,'emit'),patch.object(runtime,'retain_qualified') as retain:
                row=c.run_case(cfg)
            self.assertEqual(row['status'],'FAILED');retain.assert_not_called()
            self.assertFalse(list(c.root.glob('**/*.pt')))
            self.assertTrue(list(c.root.glob('**/failure.json')))

    def test_failed_run_attempts_never_become_new_seeds(self):
        with campaign() as c:
            cfg=plan.resolve_case('s1',baseline=baseline())
            c.state['runs']=[{**cfg,'status':'FAILED','attempt':1}]
            self.assertEqual(c.next_attempt(cfg),2)
            self.assertTrue(c.completed_recipe(cfg))
            self.assertEqual(cfg['seed'],261013101)

    def test_two_anchor_tradeoff_maximum_and_no_arbitrary_second(self):
        with campaign() as c:
            one=completed('EF01',h=(.960,)*3,e=(2.08,)*3)
            two=completed('EF02',h=(.957,)*3,e=(2.,)*3)
            worse=completed('EF03',h=(.93,)*3,e=(2.5,)*3)
            c.state['runs']=[one,two,worse]
            self.assertEqual({x['loss_case'] for x in c.select_anchors()},{'EF01','EF02'})
            c.state['runs']=[one,worse]
            self.assertEqual(len(c.select_anchors()),1)

    def test_trace_uploads_step10_and_preupdate_gradient_identity(self):
        with campaign() as c:
            cfg=plan.resolve_case('s1',baseline=baseline())
            trainer=SimpleNamespace(updates=9,stage=object())
            def step(_):
                trainer.updates=10
                return {'loss':1.,'term_gradient_norm':{'rec':{'A':.1}},
                        'gradient_norm':{'aligner':.1},'optimizer_update_norm':{'A':.01}}
            trainer.step_accumulated=step
            with patch.object(runtime,'set_learning_rates',return_value={'A':1e-5,'T':1e-4}), \
                 patch.object(runtime,'model_state_hash',return_value='a'*64),patch.object(c,'emit') as emit:
                c.trace(cfg,trainer,Mock(),c.root)
            summary=emit.call_args.kwargs['extra']['numeric_observations']
            self.assertEqual(summary['gradient_parameter_state_update'],9)
            self.assertEqual(summary['attached_model_hash_parameter_state_update'],10)
            self.assertEqual(summary['term_gradient_norm']['rec']['A'],.1)

    def test_closeout_repeat_mean_sd_denominator_and_nonexecuted_inventory(self):
        with campaign() as c:
            r1=completed('EF00',h=(.959,)*3,e=(2.,)*3)
            r2=completed('EF00',h=(.957,)*3,e=(2.1,)*3)
            r2.update(plan.resolve_case('s1',baseline=baseline(),replica='R2'))
            r2.update(status='COMPLETED',completed_update=50000)
            for row in r2['evaluations']:row['run_id']=r2['run_id']
            c.state['runs']=[r1,r2]
            c.state['observations']=r1['evaluations']+r2['evaluations']
            with patch.object(c,'retry_pending_artifacts'):
                result=c.closeout()
            self.assertEqual(result['success_rate']['n_success'],1)
            self.assertEqual(result['success_rate']['n_completed'],2)
            self.assertEqual(result['success_rate']['rate'],.5)
            self.assertAlmostEqual(result['completed_seed_statistics']['HQNR']['mean'],.958)
            self.assertAlmostEqual(result['completed_seed_statistics']['HQNR']['sample_sd'],2**.5*.001)
            self.assertEqual(result['selected_recipe_seeds'][2]['status'],'NOT_STARTED_BUDGET')
            self.assertEqual(len(result['loss_inventory']),20)
            self.assertTrue((c.root/'publisher_close.json').exists())

    def test_large_pareto_closeout_is_split_without_fake_single_state(self):
        with campaign() as c:
            # Artificial 250-point nondominated frontier, not scientific results.
            c.state['observations']=[numeric(.95+i*.00001,2.1+i*.00001,run_id=f'synthetic{i}')
                                     for i in range(250)]
            for i,o in enumerate(c.state['observations']):
                o.update(record_id='EGR2-'+f'{i:064x}',model_state_sha256=f'{i:064x}')
            with patch.object(c,'retry_pending_artifacts'):
                result=c.closeout()
            self.assertEqual(len(result['pareto']),250)
            envelopes=[json.loads(p.read_text()) for p in c.outbox.outbox.glob('*.json')]
            aggregate=[r for r in envelopes if r['source']['extra'].get('aggregate_no_single_model')]
            self.assertGreater(len(aggregate),1)
            self.assertTrue(all(r['identity']['model_state_sha256'] is None for r in aggregate))
            self.assertTrue((c.root/'publisher_close.json').exists())


class DecisionTests(unittest.TestCase):
    def evidence(self,c):
        cfg=plan.resolve_case('s1',baseline=baseline())
        return c.emit(cfg,'DIAGNOSTIC',scope='DIAGNOSTIC_LEAN',state_sha='a'*64,
            extra={'numeric_observations':{'g_edge':.9}})

    def write(self,c,items):
        path=c.root/'decisions.json'
        atomic_json(path,{'schema':'EDGERES48_DECISIONS_V1','decisions':items})
        return path

    def conditional(self,rid):
        return {'decision_id':'ef16-evidence','kind':'CONDITIONAL_EF','loss_case':'EF16',
            'condition':plan.CONDITIONAL_CASES['EF16'],'reason':'synthetic observed conflict',
            'evidence_record_ids':[rid]}

    def test_existing_lane_numeric_evidence_and_immutable_decisions(self):
        with campaign() as c:
            decision=self.conditional(self.evidence(c));self.write(c,[decision])
            self.assertEqual(c.read_decisions(),[decision])
            self.write(c,[{**decision,'reason':'rewritten'}])
            with self.assertRaisesRegex(ValueError,'modified'):c.read_decisions()
            self.write(c,[])
            with self.assertRaisesRegex(ValueError,'append-only'):c.read_decisions()

    def test_unrecorded_evidence_or_status_only_rejected(self):
        with campaign() as c:
            self.write(c,[self.conditional('EGR2-'+'0'*64)])
            with self.assertRaisesRegex(ValueError,'not recorded'):c.read_decisions()
            rid=c.emit(plan.resolve_case('s1',baseline=baseline()),'STARTED')
            self.write(c,[self.conditional(rid)])
            with self.assertRaisesRegex(ValueError,'official metrics'):c.read_decisions()

    def test_conditional_dispatch_uses_full_method_fresh_config(self):
        with campaign() as c:
            decision=self.conditional(self.evidence(c));self.write(c,[decision])
            with patch.object(c,'admit',return_value=True),patch.object(c,'run_case') as run,patch.object(c,'emit'):
                c.dispatch_decisions('CONDITIONAL_EF')
                cfg=run.call_args.args[0]
                self.assertEqual(cfg['loss_case'],'EF16');self.assertTrue(cfg['runnable'])
                self.assertIsNone(cfg['donor_checkpoint'])

    def test_conditional_can_propagate_to_confirmation_or_et(self):
        with campaign() as c:
            decision=self.conditional(self.evidence(c));self.write(c,[decision]);c.read_decisions()
            cfg=plan.resolve_case('s1',baseline=baseline(),loss_case='EF16',replica='R2',
                decision=c.activation_for_case('EF16'))
            self.assertTrue(cfg['runnable']);self.assertEqual(cfg['seed'],261013102)

    def test_horizon_requires_completed_previous_and_three_late_same_run_points(self):
        with campaign() as c:
            run=completed();c.state['runs']=[run]
            d={'loss_case':'EF00','optimizer_case':None,'schedule':'H100',
               'evidence_record_ids':[o['record_id'] for o in run['evaluations']]}
            self.assertTrue(c.horizon_supported(d))
            d['schedule']='H150';self.assertFalse(c.horizon_supported(d))
            d['schedule']='H100';d['evidence_record_ids'].pop();self.assertFalse(c.horizon_supported(d))

    def test_horizon_never_promotes_collapse_or_late_reversal(self):
        with campaign() as c:
            for h,e in [((.959,.957,.955),(2.1,2.08,2.06)),((.959,)*3,(2.1,2.08,2.09))]:
                run=completed(h=h,e=e);c.state['runs']=[run]
                d={'loss_case':'EF00','optimizer_case':None,'schedule':'H100',
                   'evidence_record_ids':[o['record_id'] for o in run['evaluations']]}
                self.assertFalse(c.horizon_supported(d))

    def test_auto_horizon_evidence_and_admission(self):
        with campaign() as c:
            c.state['runs']=[completed()]
            with patch.object(c,'admit',return_value=True),patch.object(c,'run_case') as run,patch.object(c,'emit'):
                c.automatic_horizon()
                cfg=run.call_args.args[0]
                self.assertEqual(cfg['schedule'],'H100');self.assertEqual(cfg['initialization'],'FRESH')
                self.assertTrue(c.state['automatic_decisions'])

    def test_optimizer_decision_cannot_choose_third_unselected_anchor(self):
        with campaign() as c:
            c.state['runs']=[completed('EF00')]
            rid=self.evidence(c)
            d={'decision_id':'bad-anchor','kind':'OPTIMIZER','loss_case':'EF02','optimizer_case':'ET08',
               'reason':'synthetic','evidence_record_ids':[rid]}
            self.write(c,[d])
            with self.assertRaisesRegex(ValueError,'selected Phase-A anchors'):c.dispatch_decisions('OPTIMIZER')


if __name__=='__main__':unittest.main()
