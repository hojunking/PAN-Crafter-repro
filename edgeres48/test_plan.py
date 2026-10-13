"""Synthetic campaign policy tests. No data reads, training or remote writes."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from . import plan

ROOT = Path(__file__).resolve().parents[1]


def baseline(server='s1'):
    return json.loads((ROOT/'config'/'edgeres48'/f'{server}.baseline.json').read_text())


def evaluation(server='s1', h=None, e=None):
    lane = plan.LANES[server]
    ident = {**plan.METHOD_CONTRACT, 'run_id': 'synthetic-only', 'attempt': 1,
             'seed': lane['seeds'][0], 'completed_update': 25000,
             'model_state_sha256': 'a'*64, 'inference_policy_digest': 'native-original-all20',
             'evaluator_revision': 'synthetic-v1', 'edge_protocol_digest': 'fixed-r4-plus5'}
    rr = {'ERGAS': lane['ergas_max'] if e is None else e, 'SCC': .97, 'SAM': 2.,
          'PSNR': 38., 'SSIM': .96, 'RMSE': 12., 'CC': .98,
          'Q8' if server == 's1' else 'Q4': .95}
    fr = {'HQNR': lane['hqnr_min'] if h is None else h, 'D_s': .02,
          'D_lambda': .01, 'JQM': .8}
    return {'full_method_eligible': True, 'server': server, 'dataset': lane['dataset'],
            **ident, **{key: dict(ident) for key in ('state_before','rr_state','fr_state','state_after')},
            'rr': {'metrics': rr, 'per_scene': [{'scene_id': f'rr{i}', 'metrics': dict(rr)} for i in range(20)]},
            'fr': {'metrics': fr, 'per_scene': [{'scene_id': f'fr{i}', 'metrics': dict(fr)} for i in range(20)]}}


def qualify(record, server='s1'):
    return plan.qualify_evaluation(record, server=server,
        expected_rr_ids=[f'rr{i}' for i in range(20)], expected_fr_ids=[f'fr{i}' for i in range(20)])


def numeric(h=.95,e=2.1,step=25000,run_id='unit',passed=False):
    return {'HQNR':h,'ERGAS':e,'completed_update':step,'run_id':run_id,'joint_pass':passed,
            'metrics_complete':True,'metrics_finite':True,'same_state_verified':True}


def history(case='EF01',h=.95,e=2.1):
    return {'server':'s1','dataset':'WV3','schedule':'H50','seed':261013101,
            'replica':'R1','confirmation':False,'run_id':case,'loss_case':case,
            'optimizer_case':None,'completed_update':25000,
            'evaluations':[numeric(h,e,s,case) for s in (1000,2000,10000,20000,24000,25000)]}


class RegistryTests(unittest.TestCase):
    def test_document_hashes_exact(self):
        for rel, expected in [('results_log/'+plan.PLAN_FILE_NAME,plan.PLAN_SHA256),
                              ('research_log/'+plan.METHOD_FILE_NAME,plan.METHOD_SHA256)]:
            self.assertEqual(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),expected)

    def test_lanes_targets_and_seed_triplets(self):
        self.assertEqual([(x['dataset'],x['hqnr_min'],x['ergas_max']) for x in plan.LANES.values()],
                         [('WV3',.958,2.04),('QB',.923,3.5),('GF2',.965,.522)])
        self.assertEqual(plan.LANES['s3']['seeds'],(261013301,261013302,261013303))

    def test_exact_ef_grid_and_conditional_tail(self):
        eps=(.001,.0003,.0001,.003,.001,.001,.001,.0003,.0001,.003,.0003,.0001,.003,.0003,.0001,.003,.00001,.00001,.0001,.001)
        ab=(1,1,1,1,.3,.1,3,.3,.3,.3,.1,.1,.1,3,3,3,.1,1,.03,.03)
        self.assertEqual(list(plan.LOSS_CASES.values()),list(zip(eps,ab)))
        for order in plan.LOSS_ORDER.values():
            self.assertEqual(len(order),16)
            self.assertEqual(set(order),{f'EF{i:02}' for i in range(16)})
        self.assertEqual(plan.LOSS_ORDER['s2'][:6],('EF00','EF02','EF08','EF11','EF01','EF07'))

    def test_optimizer_et08_new_not_old_t08(self):
        self.assertEqual(list(plan.OPTIMIZER_CASES.values()),[
            (.1,1,1),(.3,1,1),(3,1,1),(1,.5,1),(1,1.5,1),(1,1,.1),(1,1,0),(.3,1.5,1),(3,.5,1)])
        self.assertEqual(plan.optimizer_priority('s2','PRIOR_HARMFUL_RESIDUAL_NOT_RECOVERING'),('ET02','ET08'))
        self.assertEqual(plan.optimizer_priority('s1'),('ET01','ET03'))

    def test_cadence_and_no_duplicate_endpoints(self):
        for name,count in (('H50',27),('H100',52),('H150',77)):
            s=plan.evaluation_steps(name)
            self.assertEqual(len(s),count)
            self.assertEqual(s,tuple(sorted(set(s))))
            self.assertEqual(s[-1],plan.HORIZONS[name])
            self.assertIn(25000,s)
            self.assertEqual(s[0],1000)
        for bad in (True,5000,'H25'):
            with self.assertRaises(ValueError): plan.evaluation_steps(bad)

    def test_baselines_and_catalogs_match_executable_policy(self):
        for server in plan.LANES:
            b=baseline(server)
            plan.validate_baseline(b)
            saved=json.loads((ROOT/'config'/'edgeres48'/f'{server}.catalog.json').read_text())
            self.assertEqual(saved,json.loads(json.dumps(plan.catalog(server))))
            self.assertEqual(b['teacher_architecture']['depth'],[1,2,3])
            self.assertEqual(b['teacher_architecture']['width'],112)
            self.assertIn('PROPOSAL',b['baseline_status'])
            self.assertEqual(b['probes']['gradient_rows'][3],list(range(144,192)))

    def test_missing_and_numeric_baseline_fields_rejected(self):
        for k in plan.BASELINE_REQUIRED:
            b=baseline();del b[k]
            with self.subTest(k=k),self.assertRaises(ValueError): plan.validate_baseline(b)
        for k,v in [('lr_A0',0),('lr_U0',float('nan')),('weight_decay0',-1),('warmup_updates0',50000),
                    ('effective_batch_RR',47),('amp',True),('tf32',True),('precision','fp16')]:
            b=baseline();b[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError): plan.validate_baseline(b)

    def test_old_fr_and_scene_prior_baseline_forbidden(self):
        for k,v in [('requires_fr_train',True),('requires_source_scene_map',True),('microbatch_FR',None)]:
            b=baseline();b[k]=v
            with self.assertRaises(ValueError): plan.validate_baseline(b)

    def test_radius_changes_need_preregistered_train_evidence(self):
        b=baseline();b['edge_match']['radius_rr']=b['edge_match']['radius_fr']=6
        with self.assertRaises(ValueError): plan.validate_baseline(b)
        b['edge_radius_train_only_evidence']={'reason':'synthetic decision before official run'}
        plan.validate_baseline(b)
        b['edge_match']['radius_fr']=4
        with self.assertRaises(ValueError): plan.validate_baseline(b)

    def test_full_method_positive_rr_only_and_fresh(self):
        for server in plan.LANES:
            for ef in plan.LOSS_CASES:
                c=plan.resolve_case(server,baseline=baseline(server),loss_case=ef)
                self.assertGreater(c['lambda_abs'],0)
                self.assertGreater(c['lambda_eps_RR'],0)
                self.assertFalse(c['requires_fr_train'])
                self.assertFalse(c['requires_source_scene_map'])
                self.assertNotIn('lambda_eps_FR',c)
                self.assertEqual(c['initialization'],'FRESH')
                self.assertIsNone(c['donor_checkpoint'])

    def test_conditional_cases_never_auto_trigger(self):
        c=plan.resolve_case('s1',baseline=baseline(),loss_case='EF16')
        self.assertEqual(c['status'],'NOT_TRIGGERED');self.assertFalse(c['runnable'])
        decision={'condition':plan.CONDITIONAL_CASES['EF16'],'reason':'recorded evidence',
                  'evidence_record_ids':['synthetic-record']}
        c=plan.resolve_case('s1',baseline=baseline(),loss_case='EF16',decision=decision)
        self.assertTrue(c['runnable'])
        c['conditional_activation']['decision']=None
        with self.assertRaises(ValueError):plan.validate_method_config(c)

    def test_optimizer_multipliers_not_accumulated_and_zero_wd_duplicate(self):
        c=plan.resolve_case('s1',baseline=baseline(),optimizer_case='ET08')
        self.assertEqual(c['lr_A'],baseline()['lr_A0']*3)
        self.assertEqual(c['lr_U'],5e-5)
        b=baseline();b['weight_decay0']=0
        for et in ('ET05','ET06'):
            c=plan.resolve_case('s1',baseline=b,optimizer_case=et)
            self.assertEqual(c['status'],'SKIP_DUPLICATE_CONFIG');self.assertFalse(c['runnable'])

    def test_registered_streams_attempts_replicas_and_long_fresh(self):
        c=plan.resolve_case('s1',baseline=baseline())
        retry=plan.resolve_case('s1',baseline=baseline(),attempt=2)
        self.assertEqual(c['run_id'],retry['run_id'])
        self.assertEqual(c['stream_seeds'],retry['stream_seeds'])
        self.assertEqual(len(set(c['stream_seeds'].values())),6)
        h100=plan.resolve_case('s1',baseline=baseline(),schedule='H100')
        self.assertEqual(c['stream_seeds'],h100['stream_seeds'])
        self.assertIsNone(h100['donor_checkpoint'])
        r2=plan.resolve_case('s1',baseline=baseline(),replica='R2')
        self.assertNotEqual(c['seed'],r2['seed']);self.assertTrue(r2['confirmation'])

    def test_tampered_resolved_recipe_identity_rng_rejected(self):
        for key,val in [('lambda_abs',0),('lambda_eps_FR',.001),('donor_checkpoint','old.pt'),
                        ('edge_protocol_digest','a'),('run_id','EGR2_other'),('attempt',0),
                        ('stream_seeds',{}),('lr_A',1e-2),('epsilon_distribution','SQUARE')]:
            c=plan.resolve_case('s1',baseline=baseline());c[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):plan.validate_method_config(c)

    def test_edge_digest_matches_executable_manifest(self):
        from panda_method.edge_match import EdgeMatcher,EdgeMatchSpec
        b=baseline()
        self.assertEqual(plan.edge_protocol_digest(b),
                         EdgeMatcher(EdgeMatchSpec(**b['edge_match'])).fingerprint())


class GateTests(unittest.TestCase):
    def test_exact_joint_threshold_all_lanes(self):
        for server in plan.LANES:self.assertTrue(qualify(evaluation(server),server)['joint_pass'])

    def test_no_rounding_single_metric_or_nan_pass(self):
        for h,e in [(.958-1e-12,2.04),(.958,2.04+1e-12),(float('nan'),2.),(.99,float('inf'))]:
            self.assertFalse(qualify(evaluation(h=h,e=e))['joint_pass'])

    def test_missing_scene_duplicate_metric_or_nonfinite_blocks(self):
        for change in ('missing','duplicate','metric','nonfinite','mean'):
            r=evaluation()
            if change=='missing':r['rr']['per_scene'].pop()
            if change=='duplicate':r['fr']['per_scene'][-1]['scene_id']='fr0'
            if change=='metric':del r['rr']['per_scene'][0]['metrics']['CC']
            if change=='nonfinite':r['fr']['per_scene'][0]['metrics']['JQM']=float('nan')
            if change=='mean':r['rr']['metrics']['ERGAS']-=.01
            with self.subTest(change=change):self.assertFalse(qualify(r)['joint_pass'])

    def test_every_four_way_identity_field_must_match(self):
        for key in plan.IDENTITY_FIELDS:
            r=evaluation();r['fr_state'][key]='different'
            with self.subTest(key=key):self.assertFalse(qualify(r)['same_state_verified'])
        r=evaluation();r['model_state_sha256']='b'*64
        self.assertIn('OUTER_IDENTITY_MISMATCH',qualify(r)['reasons'])

    def test_full_method_and_lane_guard(self):
        r=evaluation();r['full_method_eligible']=False
        self.assertFalse(qualify(r)['joint_pass'])
        r=evaluation();r['dataset']='GF2'
        self.assertFalse(qualify(r)['joint_pass'])

    def test_hqnr_then_ergas_not_scc_priority(self):
        a=numeric(.96,2.,run_id='a',passed=True);a['SCC']=.9
        b=numeric(.96,2.01,run_id='b',passed=True);b['SCC']=.99
        c=numeric(.99,1.9,run_id='unqualified',passed=False)
        self.assertIs(plan.representative([a,b,c]),a)
        self.assertIs(plan.rr_reserve([a,b,c]),a)
        self.assertIsNone(plan.representative([c]))

    def test_step_then_runid_exact_ties(self):
        a=numeric(.96,2.,step=2000,run_id='z',passed=True)
        b=numeric(.96,2.,step=2000,run_id='a',passed=True)
        c=numeric(.96,2.,step=4000,run_id='0',passed=True)
        self.assertIs(plan.representative([a,b,c]),b)


class PruningAndBudgetTests(unittest.TestCase):
    def prune(self,current,previous):
        return plan.should_prune(current,previous,stagnation_window=3,tolerance=0)

    def test_protected_baseline_confirmations_and_minimum_step(self):
        previous=[history('EF02',.96,2.),history('EF03',.96,2.)]
        for c in [history('EF00'),{**history(),'replica':'R2'},
                  {**history(),'confirmation':True},{**history(),'completed_update':24000}]:
            self.assertFalse(self.prune(c,previous)['prune'])

    def test_two_distinct_recipes_three_common_points_and_stagnation(self):
        c=history();p=[history('EF02',.96,2.),history('EF03',.96,2.)]
        result=self.prune(c,p)
        self.assertTrue(result['prune']);self.assertEqual(result['common_steps'],[20000,24000,25000])
        self.assertFalse(self.prune(c,p[:1])['prune'])
        self.assertFalse(self.prune(c,[p[0],copy.deepcopy(p[0])])['prune'])
        p[1]['evaluations']=p[1]['evaluations'][-2:]
        self.assertFalse(self.prune(c,p)['prune'])

    def test_recent_improvement_protects_even_if_historically_dominated(self):
        c=history();c['evaluations'][-1]['ERGAS']=2.09
        p=[history('EF02',.96,2.),history('EF03',.96,2.)]
        self.assertEqual(self.prune(c,p)['reason'],'TARGET_GAP_IMPROVING')

    def test_cross_seed_cross_horizon_and_invalid_evaluation_not_comparators(self):
        for key,val in [('seed',261013102),('schedule','H100'),('dataset','QB')]:
            p=[history('EF02',.96,2.),history('EF03',.96,2.)];p[1][key]=val
            self.assertFalse(self.prune(history(),p)['prune'])
        p=[history('EF02',.96,2.),history('EF03',.96,2.)]
        p[1]['evaluations'][-1]['same_state_verified']=False
        self.assertFalse(self.prune(history(),p)['prune'])

    def test_pruning_parameters_must_be_explicit_and_valid(self):
        with self.assertRaises(TypeError):plan.should_prune(history(),[])
        for window,tol in [(2,0),(3,-1),(3,float('nan'))]:
            with self.assertRaises(ValueError):plan.should_prune(history(),[],stagnation_window=window,tolerance=tol)

    def test_raw_eta_and_single_safety_multiplier(self):
        eta=plan.estimate_remaining_seconds(remaining_updates=100,update_p90_seconds=2,
            remaining_evaluations=3,rr_fr_eval_seconds=10,artifact_seconds=5,probe_seconds=7)
        self.assertEqual(eta,242)
        a=plan.admit_new_run(elapsed_seconds=0,estimated_to_25k_seconds=120,estimated_full_seconds=eta)
        self.assertEqual(a['estimated_run_seconds'],1.2*eta)
        self.assertEqual(a['confirmation_reserve_seconds'],2*1.2*eta)
        self.assertEqual(a['closing_reserve_seconds'],7200)

    def test_two_repeats_reserved_and_r2_r3_consume_reserve(self):
        kwargs=dict(elapsed_seconds=30*3600,estimated_to_25k_seconds=2*3600,estimated_full_seconds=6*3600)
        self.assertFalse(plan.admit_new_run(**kwargs)['admitted'])
        r2=plan.admit_new_run(**kwargs,remaining_confirmations=1)
        self.assertTrue(r2['admitted'])
        r3=plan.admit_new_run(**kwargs,remaining_confirmations=0)
        self.assertEqual(r3['confirmation_reserve_seconds'],0)

    def test_46h_and_phase_boundaries_only_admit_before_boundaries(self):
        base=dict(estimated_to_25k_seconds=100,estimated_full_seconds=200,remaining_confirmations=0)
        self.assertEqual(plan.admit_new_run(elapsed_seconds=46*3600,**base)['reason'],'NOT_STARTED_NO_NEW_AFTER_46H')
        self.assertEqual(plan.admit_new_run(elapsed_seconds=22*3600,phase_end_hours=22,**base)['reason'],'NOT_STARTED_PHASE_BOUNDARY')
        self.assertTrue(plan.admit_new_run(elapsed_seconds=21.9*3600,phase_end_hours=22,**base)['admitted'])
        with self.assertRaises(ValueError):plan.admit_new_run(elapsed_seconds=0,closing_reserve_seconds=7199,**base)


if __name__=='__main__':unittest.main()
