"""Offline publication semantics, including source integrity and full RR/FR proof."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from .selection import (CAMPAIGN_ID, FLAGS, HEADERS, INDEX, LANES, METHOD, METRICS,
    canonical_json, destination, digest, project, project_campaign, scientific_digest,
    validate_candidate, EvidenceError)
from .source import scan_campaign


def seal(r):
    r['record_id'] = 'EGR2-' + digest(r['identity'])
    source = r['source']; fields = source['fields']
    fields['Result_ID'] = r['record_id']
    fields['Selector'] = r['identity']['selector']
    note = {**METHOD, **source['provenance'], 'extra_parameters': source['extra'],
            'model_state_sha256': r['identity']['model_state_sha256'],
            'eval_protocol_id': r['identity']['eval_protocol']}
    fields['Original description / notes'] = canonical_json(note)
    r['row'] = [fields.get(k) if fields.get(k) is not None else '' for k in HEADERS]
    r['payload_sha256'] = digest(r['row'])
    return r


def record(server='s1', step=1000, h=.95, e=2.1, run='RUN', state=None, selector=None, seed=123):
    dataset = LANES[server][0]
    baseline = {'teacher_architecture': {'width':112,'depth':[1,2,3]}, 'optimizer':'AdamW',
        'optimizer_betas':[.9,.999], 'optimizer_eps':1e-8, 'scheduler_family':'warmup_cosine',
        'scheduler_indexing':'ONE_BASED', 'warmup_updates0':100, 'min_lr':0,
        'effective_batch_RR':48, 'microbatch_RR':12, 'accumulation_RR':4,'precision':'fp32','amp':False,'tf32':False,
        'edge_match': {'radius_rr':4,'radius_fr':4}, 'aligner_architecture':{'name':'ResidualPANAligner'}}
    cfg = {**METHOD,'baseline_resolved':baseline, 'run_id':run,'attempt':1,'seed':seed,'dataset':dataset,'server':server,
           'campaign_id':CAMPAIGN_ID,'case':'EF01','replica':'R1','planned_horizon':50000,'edge_protocol_digest':'b'*64,
           'lr_U':1e-4,'lr_A':1e-5,'lambda_abs':1.,'lambda_epsilon':3e-4,'epsilon_radius_rr':2.,'weight_decay':.01}
    ident = {**METHOD, 'run_id':run,'attempt':1,'seed':seed,'completed_update':step,
             'model_state_sha256':state or digest([run,step]), 'edge_protocol_digest':'b'*64,
             'inference_policy_digest':'c'*64,'evaluator_revision':'d'*64}
    metrics = {'HQNR':h,'ERGAS':e, 'D_s':.02,'D_lambda':.02,'JQM':.98,'SCC':.99,'SAM':2.,'PSNR':40.,
               'SSIM':.98,'Q8' if dataset=='WV3' else 'Q4':.95,'RMSE':20.,'CC':.99}
    extra = {**dict.fromkeys(FLAGS,True),'configuration':cfg,'identity':ident,'HQNR':h,'ERGAS':e,
             'joint_pass':False,'weights_saved':False,'source_evaluation':f'/workspace/lane/runs/{run}/attempt_01/evaluation_{step:07d}.json',
             'scene_completeness':{s:{'expected':20,'completed':20,'missing':[],'duplicates':[],'unexpected':[]} for s in ('rr','fr')}}
    envelope = {'namespace':'PAN_FIXED64_V1:2026-10-13','campaign':CAMPAIGN_ID,'dataset':dataset,'server':server,
                'run':run,'attempt':1,'selected_step':step,'selector':selector or f'OBSERVED_{step}',
                'checkpoint':None,'model_state_sha256':ident['model_state_sha256'],'eval_protocol':'d'*64,'eval_scope':'EVAL_SUMMARY'}
    fields = {'Experiment':run, 'Run_ID':run,'Attempt':1,'Selected_step':step,'Seed':seed,'Repeat':1,'Case_ID':'EF01',
              'Server':server,'Dataset':dataset,'Bucket':'ablations','Campaign_ID':CAMPAIGN_ID,'Checkpoint_SHA':None,
              'Readback':'PENDING','Eval_scope':'EVAL_SUMMARY','Test_aware':True,'U_peak_lr':1e-4,'A_peak_lr':1e-5,
              'Status':'OBSERVED','Date':'2026-10-14T00:00:00+00:00'}
    for k,col in METRICS.items():
        fields[col]=metrics[('Q8' if dataset=='WV3' else 'Q4') if k=='Q' else k]
    return seal({'identity':envelope, 'source':{'fields':fields,'metrics':metrics,'extra':extra,
           'provenance':{'config_sha256':scientific_digest(baseline),'dataset_manifest_sha256':'f'*64,'evaluator_revision':'d'*64}},
           'destination':destination(server),'record_type':'EVAL_SUMMARY','created_at':fields['Date']})


def evidence(root, r):
    cfg = r['source']['extra']['configuration']; ident = r['source']['extra']['identity']
    p=Path(root)/'runs'/cfg['run_id']/'attempt_01'; p.mkdir(parents=True,exist_ok=True)
    (p/'config.json').write_text(canonical_json(cfg))
    rr={k:v for k,v in r['source']['metrics'].items() if k not in ('HQNR','D_s','D_lambda','JQM')}
    fr={k:v for k,v in r['source']['metrics'].items() if k in ('HQNR','D_s','D_lambda','JQM')}
    value={'record_type':'EVAL_SUMMARY','full_method_eligible':True,'measurement_domain':'OFFICIAL_FULL20_NATIVE',
           'metric_function_injection':False,'test_aware':True,'eval_protocol_id':'d'*64,
           **{k:ident for k in ('state_before','rr_state','fr_state','state_after')},
           'qualification':r['source']['extra']}
    for split,metrics in [('rr',rr),('fr',fr)]:
        value[split]={'metrics':metrics, 'per_scene':[{'scene_id':f"{cfg['dataset']}:{split}:{i}",
            'sample_id':f"{cfg['dataset']}:{split}:{i}", 'evaluation_image_id':f"{cfg['dataset']}:{split}:{i}",
            'metrics':metrics} for i in range(20)]}
    path=p/f"evaluation_{ident['completed_update']:07d}.json";path.write_text(canonical_json(value));return path


class SelectionTest(unittest.TestCase):
    def test_inclusive_thresholds_all_sensors(self):
        for server,lane in LANES.items():
            for h,e,expected in [(lane[3],lane[4],'PASS_ALL_STEPS'),(lane[3]-1e-10,lane[4],'BEST_HQNR_FALLBACK'),
                                  (lane[3],lane[4]+1e-10,'BEST_HQNR_FALLBACK')]:
                with self.subTest(server=server,h=h,e=e):
                    out=project([record(server,h=h,e=e)],server)
                    self.assertEqual(out['conflicts'],[])
                    self.assertEqual(out['rows'][0][INDEX['Selector']],expected)

    def test_all_three_passed_steps_are_retained(self):
        out=project([record(step=s,h=.961,e=2.04) for s in (6000,2000,4000)],'s1')
        self.assertEqual([r[INDEX['Selected_step']] for r in out['rows']],[2000,4000,6000])

    def test_fallback_hqnr_not_min_ergas_and_stable_id(self):
        a=record(step=1000,h=.950,e=2.1);b=record(step=2000,h=.951,e=3.)
        x=project([a],'s1');y=project([a,b],'s1')
        self.assertEqual(y['selected'][0]['step'],2000)
        self.assertEqual(x['rows'][0][INDEX['Result_ID']],y['rows'][0][INDEX['Result_ID']])

    def test_first_pass_replaces_fallback_later_fail_does_not_replace_pass(self):
        a=record(step=1000,h=.95);b=record(step=2000,h=.961,e=2.04);c=record(step=3000,h=.98,e=3.)
        out=project([a,b,c],'s1')
        self.assertEqual([c['step'] for c in out['selected']],[2000])
        self.assertNotEqual(project([a],'s1')['rows'][0][59],out['rows'][0][59])

    def test_exact_tie_order(self):
        records=[record(step=4000,h=.952,e=2.09),record(step=2000,h=.951,e=2.0),record(step=1000,h=.952,e=2.1)]
        self.assertEqual(project(records,'s1')['selected'][0]['step'],4000)
        records.append(record(step=3000,h=.952,e=2.09))
        self.assertEqual(project(records,'s1')['selected'][0]['step'],3000)
        x=record(step=3000,h=.952,e=2.09,selector='ALTERNATIVE')
        records.append(x)
        self.assertEqual(project(records,'s1')['selected'][0]['source_result_id'],min(x['record_id'],records[-2]['record_id']))

    def test_independent_runs_seeds(self):
        out=project([record(run='RUN1',seed=1),record(run='RUN2',seed=2)],'s1')
        self.assertEqual(len(out['rows']),2)

    def test_non_summary_excluded(self):
        for scope in ('PREFLIGHT_EVAL','RR_SCENE:a','FR_SCENE:a','TRACE_100_UPDATE_SUMMARY','STATUS','DIAGNOSTIC_LEAN'):
            r=record();r['identity']['eval_scope']=scope;r['record_type']=scope;r['source']['fields']['Eval_scope']=scope;seal(r)
            self.assertEqual(project([r],'s1')['rows'],[])

    def test_partial_flags_counts_and_nonfinite_excluded(self):
        for key in FLAGS:
            r=record();r['source']['extra'][key]=False;seal(r)
            self.assertEqual(project([r],'s1')['rows'],[])
        for key,val in [('completed',19),('duplicates',['x']),('missing',['x']),('unexpected',['x'])]:
            r=record();r['source']['extra']['scene_completeness']['fr'][key]=val;seal(r)
            self.assertEqual(project([r],'s1')['rows'],[])
        r=record();del r['source']['metrics']['SCC'];seal(r)
        self.assertEqual(project([r],'s1')['rows'],[])

    def test_identity_seed_step_and_protocol_rejected(self):
        for key,value in [('seed',999),('completed_update',999),('model_state_sha256','a'*64),('evaluator_revision','b'*64)]:
            r=record();r['source']['extra']['identity'][key]=value;seal(r)
            out=project([r],'s1');self.assertEqual(out['rows'],[]);self.assertTrue(out['conflicts'])

    def test_config_conflict_rejected(self):
        a=record(step=1000);b=record(step=2000)
        b['source']['extra']['configuration']['lambda_abs']=2.;seal(b)
        out=project([a,b],'s1');self.assertEqual(out['rows'],[]);self.assertTrue(out['conflicts'])

    def test_state_and_metric_conflicts(self):
        for b in (record(state='a'*64),record(h=.955,selector='OTHER')):
            out=project([record(),b],'s1');self.assertEqual(out['rows'],[]);self.assertTrue(out['conflicts'])
        a=record(state='a'*64);b=record(step=2000,state='a'*64,h=.955)
        self.assertTrue(project([a,b],'s1')['conflicts'])

    def test_duplicate_delivery_idempotent(self):
        r=record();out=project([r,deepcopy(r)],'s1')
        self.assertEqual(len(out['rows']),1);self.assertEqual(out['conflicts'],[])

    def test_publish_threshold_independent_of_artifact_joint_pass(self):
        r=record(h=.960,e=2.044);r['source']['extra']['joint_pass']=False;seal(r)
        out=project([r],'s1');self.assertEqual(out['rows'][0][42],'PASS_ALL_STEPS')
        self.assertEqual(out['rows'][0][41],'')
        note=json.loads(out['rows'][0][60]);self.assertFalse(note['publication']['artifact']['weights_saved'])

    def test_no_cost_or_old_parameter_fabrication_and_readable_label(self):
        row=project([record()],'s1')['rows'][0]
        self.assertEqual(len(row),64);self.assertEqual(row[15:22],['']*7);self.assertEqual(row[43:48],['']*5)
        self.assertNotIn('OBSERVED',row[2]);self.assertIn('abs=1',row[0]);self.assertIn('50K',row[0])
        self.assertIn('OBSERVED',row[60])

    def test_artifact_requires_actual_same_state_file_digest(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            r=record();evidence(tmp,r)
            path=Path(tmp)/'checkpoint.pt';path.write_bytes(b'not-loaded-by-projector')
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            a=deepcopy(r);a['record_type']='QUAL_ARTIFACT';a['identity']['eval_scope']='QUAL_ARTIFACT'
            a['identity']['checkpoint']=sha;a['source']['fields']['Checkpoint_SHA']=sha
            a['source']['fields']['Eval_scope']='QUAL_ARTIFACT'
            a['source']['extra']['artifact']={'weights_saved':True,'file_sha256':sha,
                'model_state_sha256':r['identity']['model_state_sha256'],
                'identity':r['source']['extra']['identity'],'checkpoint_path':str(path)}
            seal(a)
            out=project([r,a],'s1',evidence_root=tmp)
            self.assertEqual(out['rows'][0][41],sha)
            path.write_bytes(b'changed file')
            out=project([r,a],'s1',evidence_root=tmp)
            self.assertEqual(out['rows'][0][41],'')
            self.assertTrue(json.loads(out['rows'][0][60])['publication']['artifact_warnings'])
            a['identity']['selected_step']=2000;a['source']['fields']['Selected_step']=2000;seal(a)
            self.assertEqual(project([r,a],'s1',evidence_root=tmp)['rows'][0][41],'')

    def test_official_20_plus_20_same_state_file_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=record();path=evidence(tmp,r)
            out=project([r],'s1',evidence_root=tmp);self.assertEqual(len(out['rows']),1)
            e=json.loads(path.read_text());e['fr_state']=dict(e['fr_state'],completed_update=2000);path.write_text(canonical_json(e))
            out=project([r],'s1',evidence_root=tmp);self.assertEqual(out['rows'],[]);self.assertTrue(out['conflicts'])

    def test_duplicate_scene_and_wrong_mean_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=record();path=evidence(tmp,r);e=json.loads(path.read_text())
            e['rr']['per_scene'][1]['scene_id']=e['rr']['per_scene'][0]['scene_id'];path.write_text(canonical_json(e))
            self.assertEqual(project([r],'s1',evidence_root=tmp)['rows'],[])
            path=evidence(tmp,r);e=json.loads(path.read_text());e['fr']['per_scene'][1]['metrics']['HQNR']+=.01
            path.write_text(canonical_json(e));self.assertTrue(project([r],'s1',evidence_root=tmp)['conflicts'])

    def test_official_scene_names_and_optional_lineage_aliases_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            for server in LANES:
                for mutation in ('foreign_unique_scenes','sample_id','evaluation_image_id'):
                    with self.subTest(server=server,mutation=mutation):
                        r=record(server=server);path=evidence(tmp,r);e=json.loads(path.read_text())
                        if mutation=='foreign_unique_scenes':
                            for i,scene in enumerate(e['fr']['per_scene']):
                                scene['scene_id']=scene['sample_id']=scene['evaluation_image_id']=f'unknown:{i}'
                        else:
                            e['fr']['per_scene'][0][mutation]='wrong:fr:0'
                        path.write_text(canonical_json(e));out=project([r],server,evidence_root=tmp)
                        self.assertEqual(out['rows'],[]);self.assertTrue(out['conflicts'])

    def test_missing_or_unreadable_physical_proof_is_publication_conflict(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            r=record();path=evidence(tmp,r)
            path.unlink()
            out=project([r],'s1',evidence_root=tmp)
            self.assertEqual(out['rows'],[]);self.assertTrue(out['conflicts'])
            self.assertIn('PHYSICAL_EVALUATION_PROOF_MISMATCH',out['conflicts'][0]['reason'])
            evidence(tmp,r)
            with patch('sheet_publish_v2.selection.Path.read_bytes',side_effect=PermissionError('permission denied')):
                out=project([r],'s1',evidence_root=tmp)
                self.assertTrue(out['conflicts'])

    def test_scan_keeps_acknowledged_and_ledger_only_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'outbox').mkdir();(root/'receipts').mkdir()
            a=record(step=1000,h=.955);b=record(step=2000,h=.95)
            (root/'records.jsonl').write_text(canonical_json(a)+'\n'+canonical_json(b)+'\n')
            (root/'outbox'/(a['record_id']+'.json')).write_text(canonical_json(a))
            (root/'receipts'/(a['record_id']+'.json')).write_text('{}')
            evidence(tmp,a);evidence(tmp,b)
            out=project_campaign(tmp,'s1');self.assertEqual(out['raw_count'],2);self.assertEqual(out['receipt_file_count'],1)
            self.assertEqual(out['selected'][0]['step'],1000)

    def test_conflicting_ledger_outbox_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'outbox').mkdir();a=record();b=deepcopy(a)
            b['created_at']='2026-10-14T01:00:00+00:00'
            b['source']['fields']['Date']=b['created_at'];seal(b)
            (root/'records.jsonl').write_text(canonical_json(a)+'\n')
            (root/'outbox'/(a['record_id']+'.json')).write_text(canonical_json(b))
            self.assertEqual(scan_campaign(root)['errors'][0]['reason'],'SOURCE_ID_PAYLOAD_CONFLICT')

    def test_partial_ledger_tail_blocks_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'records.jsonl').write_text(canonical_json(record()))
            self.assertEqual(scan_campaign(tmp)['errors'][0]['reason'],'INCOMPLETE_LEDGER_TAIL_RETRY')

    def test_no_frozen_runtime_imports(self):
        import ast
        for name in ('selection.py','source.py','__init__.py'):
            tree=ast.parse((Path(__file__).parent/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    self.assertFalse(any(n.name.startswith(('edgeres48','torch','panda_method')) for n in node.names))
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('edgeres48','torch','panda_method')))


if __name__=='__main__':
    unittest.main()
