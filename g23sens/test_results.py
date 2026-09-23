"""Offline evaluation/selection/report/outbox regressions; no GPU or live Sheet."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from g23sens.common import atomic_json,object_sha,sha256,camp,run_dir
from g23sens.plan import make_case
from g23sens import evaluation as E,postrun as P,reporting as R,upload as U


def summary(code='BASE',server='s4',cycle=0,attempt=0,same=False):
    c=make_case(server,cycle,code);provenance={k:'a'*64 for k in R.PAIR_KEYS}
    provenance.update(case_spec_sha256=c['case_spec_sha256'],runtime_config_sha256='b'*64,
        runtime_commit='abc123',q_weight_sha256='c'*64)
    rr=dict(ergas=2.123456789123,sam=3.,psnr=35.,ssim=.98,scc=.99,q8=.97,rmse=12.,cc=.98,n_scenes=20)
    fr=dict(hqnr=.95851234567,d_lambda=.02,d_s=.021926177887755,jqm=.93,jqm_variant=E.JQM_VARIANT,n_scenes=20)
    records={label:dict(update=50000 if i==0 or same else 1010,checkpoint_sha256=('d' if i==0 or same else 'e')*64,
        evaluation_manifest_sha256=('f' if i==0 or same else '0')*64,rr=deepcopy(rr),fr=deepcopy(fr),
        alias_of='EXACT_50000' if same and i==1 else None) for i,label in enumerate(R.SELECTIONS)}
    return dict(campaign_id=U.CAMPAIGN_ID,case=c,run_id=c['run_id'],attempt=attempt,
        complete=True,status='COMPLETE',actual_updates=50000,training_seconds=7200.1234567,
        paper_identity_status='PAPERSET_IDENTITY_UNVERIFIED',provenance=provenance,selections=records,
        started_at_utc='2026-09-23T01:02:03+00:00',completed_at_utc='2026-09-23T03:05:00+00:00')


class Sheet:
    def __init__(self,title='SENS-G23-WV3-s4'):
        self.title=title;self.col_count=1;self.row_count=1;self.id=7;self.rows=[];self.formats=[];self.writes=0
    def row_values(self,n,**kwargs):return list(self.rows[n-1]) if n<=len(self.rows) else []
    def get_all_values(self,**kwargs):return deepcopy(self.rows)
    def add_cols(self,n):self.col_count+=n
    def add_rows(self,n):self.row_count+=n
    def update(self,range_name,values,value_input_option):
        if value_input_option!='RAW':raise AssertionError('Raw precision required')
        self.writes+=1;m=re.fullmatch(r'([A-Z]+)([0-9]+)',range_name);col=0
        for x in m[1]:col=col*26+ord(x)-64
        row=int(m[2])
        while len(self.rows)<row:self.rows.append([])
        while len(self.rows[row-1])<col-1+len(values[0]):self.rows[row-1].append('')
        self.rows[row-1][col-1:col-1+len(values[0])]=values[0]
    def batch_format(self,formats):self.formats.extend(formats)


class ReportingTests(unittest.TestCase):
    def test_local_paired_values_and_precision(self):
        base=summary();variant=summary('AL05');variant['selections']['EXACT_50000']['rr']['ergas']+=.125
        pair=R.paired_deltas(variant,base,'EXACT_50000')
        self.assertAlmostEqual(pair['values']['ergas'],.125)
        row=U.flatten_summary(variant,base)[0]
        self.assertEqual(row['Train(h)'],variant['training_seconds']/3600)
        self.assertEqual(row['hqnr'],.95851234567)
        self.assertEqual(row['baseline_attempt'],0)
        self.assertNotIn(variant['run_id'],row['Run'])
    def test_foreign_baseline_and_changed_stream_rejected(self):
        base=summary();variant=summary('AL05')
        for key in R.PAIR_KEYS:
            changed=deepcopy(base);changed['provenance'][key]='b'*64
            with self.assertRaises(ValueError):R.paired_deltas(variant,changed,'EXACT_50000')
        with self.assertRaises(ValueError):R.paired_deltas(variant,summary(server='s5'),'EXACT_50000')
        with self.assertRaises(ValueError):R.paired_deltas(variant,summary(cycle=1),'EXACT_50000')
    def test_alias_and_primary_rules(self):
        s=summary(same=True);self.assertEqual(U.flatten_summary(s,s)[1]['selection_alias_of'],'EXACT_50000')
        s['selections']['RR_VAL_ERGAS_MIN']['alias_of']=None
        with self.assertRaises(ValueError):U.validate_summary(s,s['case'])
        s=summary();s['selections']['EXACT_50000']['update']=49490
        with self.assertRaises(ValueError):U.validate_summary(s,s['case'])
    def test_divergence_not_completion_or_extra_seed(self):
        bad=summary('AL05');bad.update(complete=False,status='DIVERGED',actual_updates=1100);bad.pop('selections')
        row=U.flatten_summary(bad)[0]
        self.assertEqual(row['status'],'DIVERGED');self.assertEqual(row['ergas'],'')
        report=R.summarize([summary(),bad],'s4')
        self.assertEqual(len(report['failures']),1);self.assertEqual(len(report['raw']),2)
        bad['status']='COMPLETE'
        with self.assertRaises(ValueError):U.validate_summary(bad,bad['case'])
    def test_seed_statistics_no_cross_server_pool(self):
        rows=[summary(cycle=i) for i in range(3)]
        report=R.summarize(rows,'s4');self.assertEqual(report['aggregate'][0]['independent_student_seeds'],3)
        self.assertFalse(report['duplicate_server_BASE_is_independent_seed'])
        with self.assertRaises(ValueError):R.summarize(rows+[summary(server='s5')],'s4')
        with self.assertRaises(ValueError):R.summarize([summary(),summary(attempt=1)],'s4')


class UploadTests(unittest.TestCase):
    def test_only_dedicated_tabs_and_no_header_migration(self):
        row=U.flatten_summary(summary(),summary())[0]
        for title in ('유의미한결과','PC-Repro-s4','SENS-G23-WV3-s5'):
            sheet=Sheet(title)
            with self.assertRaises(ValueError):U.apply_upsert(sheet,row)
            self.assertEqual(sheet.writes,0)
        sheet=Sheet();sheet.rows=[['old header']]
        with self.assertRaises(ValueError):U.apply_upsert(sheet,row)
        self.assertEqual(sheet.writes,0)
    def test_serialized_row_exact_precision_readback_and_idempotence(self):
        row=U.flatten_summary(summary(),summary())[0];row=json.loads(json.dumps(row,sort_keys=True))
        sheet=Sheet();receipt=U.apply_upsert(sheet,row);again=U.apply_upsert(sheet,row)
        self.assertEqual(receipt['row'],again['row']);self.assertEqual(len(sheet.rows),2)
        self.assertEqual(sheet.rows[1][U.HEADER.index('hqnr')],row['hqnr'])
        self.assertEqual(sheet.rows[1][U.HEADER.index('readback_status')],'READBACK_VERIFIED')
        self.assertTrue(all(f['format']['numberFormat']['pattern'] in ('0.0000','0.0E+00') for f in sheet.formats))
    def test_distinct_selection_and_attempt_keys(self):
        s=summary();rows=U.flatten_summary(s,s);other=summary(attempt=1)
        self.assertEqual(len({r['row_key'] for r in rows+U.flatten_summary(other,other)}),4)
    def test_absent_failure_timestamps_are_blank_not_readback_none(self):
        value=summary('AL05');value.update(complete=False,status='DIVERGED',actual_updates=1010,completed_at_utc=None)
        value.pop('selections');row=U.flatten_summary(value)[0]
        self.assertEqual(row['completed_at_utc'],'')
        self.assertTrue(U.apply_upsert(Sheet(),row)['readback_verified'])
    def test_readback_mismatch_is_not_success(self):
        row=U.flatten_summary(summary(),summary())[0];sheet=Sheet();original=sheet.row_values
        def corrupt(n,**kwargs):
            values=original(n,**kwargs)
            if n==2 and values:values[U.HEADER.index('ergas')]=777
            return values
        sheet.row_values=corrupt
        with self.assertRaises(ValueError):U.apply_upsert(sheet,row)
    def test_inactive_never_reads_credentials(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(U,'_worksheet',side_effect=AssertionError('live access')):
            with self.assertRaises(PermissionError):U.flush_outbox(Path(directory),'s4')
    def test_bounded_retry_does_not_evaluate_or_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sheet=Sheet();envelopes=[]
            for s in (summary(),summary('AL05')):
                for row in U.flatten_summary(s,summary()):
                    e=dict(schema='G23SENS_OUTBOX_v1',server='s4',run_id=s['run_id'],attempt=0,row_key=row['row_key'],
                        payload=row,payload_sha256=object_sha(row),status='PENDING',api_attempts=0)
                    atomic_json(camp(root,'s4')/'outbox'/(e['row_key']+'.json'),e);envelopes.append(e)
            def fresh(root,run_id,attempt=0):return [e for e in envelopes if e['run_id']==run_id]
            with patch.object(U,'queue_run',side_effect=fresh),patch.object(P,'run_postrun',side_effect=AssertionError('no eval')):
                result=U.flush_outbox(root,'s4',writer=sheet,activated=True,limit=1)
            self.assertEqual(len(result),1);self.assertEqual(len(sheet.rows),2)
    def test_pretrainer_failure_ledger_keeps_unknown_updates_and_cost_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);case=make_case('s4',0,'BASE');wd=run_dir(case,root)
            cfg={'g23sens':dict(case=case,attempt=0,source_identity={'content_sha256':'a'*64},binding_sha256='b'*64)}
            atomic_json(wd/'config.json',cfg)
            atomic_json(wd/'meta/attempt_failure.json',dict(kind='TRAIN_TECHNICAL',status='TECHNICAL_FAILED',code=74,
                reason='worker failed before trainer status',attempt=0,run_id=case['run_id'],at_utc='2026-09-23T01:00:00+00:00',final=True,actual_updates=None))
            value=U.status_summary(case['run_id'],root)
            self.assertIsNone(value['actual_updates']);row=U.flatten_summary(value)[0]
            self.assertEqual(row['actual_updates'],'');self.assertEqual(row['Train(h)'],'');self.assertEqual(row['hqnr'],'')
            report=R.build_report(root,'s4')
            self.assertEqual(len(report['failures']),1)
            self.assertEqual(report['aggregate'][0]['technical_failure_attempts'],1)
            self.assertEqual(report['aggregate'][0]['independent_student_seeds'],0)


class FakeDataset:
    sensor='WV3';augment=False;max_pixel=2047.;bands=8
    def __init__(self,split,count=2):
        self.split=self.split_key=split;self.binding={'data':'same locked identity'}
        self.scene_ids=[f'{split}:{i}' for i in range(count)];self.manifest_sha256=object_sha(self.scene_ids)
        self.manifest={'band_order':['coastal','blue','green','yellow','red','rededge','nir1','nir2']}
        self.spec={'identity_validation':{'status':'PAPERSET_IDENTITY_UNVERIFIED'}}
        rng=np.random.default_rng(501)
        self.samples=[dict(pan=rng.uniform(0,2047,(1,64,64)),ms=rng.uniform(0,2047,(8,16,16)),
            lpan=rng.uniform(0,2047,(1,16,16)),lms=rng.uniform(0,2047,(8,64,64)),gt=rng.uniform(0,2047,(8,64,64))) for _ in self.scene_ids]
    def __len__(self):return len(self.samples)
    def raw(self,i):return self.samples[i]
    def __getitem__(self,i):return {k:torch.from_numpy(v.astype(np.float32)).mul(2/2047).sub(1) for k,v in self.samples[i].items()}


class FakeModel(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,pan,ms,lpan):
        self.calls+=1
        return {'y':torch.nn.functional.interpolate(ms,size=pan.shape[-2:],mode='bicubic',align_corners=False),
                'delta':torch.ones((len(pan),2),device=pan.device)*2}


def fr_metric(sr,raw,*args):
    return dict(d_lambda=.1,d_s=.2,hqnr=(1-.1)*(1-.2),jqm=.8)


class EvaluationTests(unittest.TestCase):
    def test_existing_metric_implementations_unchanged(self):
        from pcrepro.evaluation import rr_scene_metrics,fr_scene_metrics
        self.assertIs(E.rr_scene_metrics,rr_scene_metrics);self.assertIs(E.fr_scene_metrics,fr_scene_metrics)
    def test_native_all_scene_evaluation_and_resume(self):
        datasets={k:FakeDataset(k) for k in ('rr','fr')};model=FakeModel();seen=[]
        def metric(sr,raw,*args):seen.append(raw['pan'].copy());return fr_metric(sr,raw,*args)
        with tempfile.TemporaryDirectory() as directory,patch.object(E,'fr_scene_metrics',side_effect=metric):
            folder=Path(directory);first=E.evaluate_checkpoint(model,datasets,'cpu','a'*64,folder,wald=SimpleNamespace())
            calls=model.calls;second=E.evaluate_checkpoint(model,datasets,'cpu','a'*64,folder,wald=SimpleNamespace())
            self.assertEqual(model.calls,calls);self.assertEqual(first,second)
            self.assertEqual(first['rr']['n_scenes'],2);self.assertFalse(first['metadata']['paper_comparable'])
            self.assertFalse(first['fr']['masking']);self.assertFalse(first['fr']['alignment'])
            np.testing.assert_array_equal(seen[0],datasets['fr'].raw(0)['pan'])
            with self.assertRaises(ValueError):E.evaluate_checkpoint(model,datasets,'cpu','b'*64,folder,wald=SimpleNamespace())
    def test_validation_preserves_rng_and_does_not_measure_hqnr(self):
        model=FakeModel();model.train();before=torch.get_rng_state().clone()
        result=E.evaluate_validation(model,FakeDataset('val'),'cpu')
        self.assertIsNone(result['hqnr']);self.assertTrue(model.training)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_validation_exact_legacy_float64_dn_and_batch_reduction(self):
        model=FakeModel();dataset=FakeDataset('val',18);total=0.;count=0
        for start in range(0,len(dataset),16):
            rows=[dataset[i] for i in range(start,min(start+16,len(dataset)))]
            batch={k:torch.stack([r[k] for r in rows]) for k in ('pan','ms','lpan','gt')}
            o=model(batch['pan'],batch['ms'],batch['lpan'])
            pred=(o['y'].clip(-1,1).double()+1)/2*2047;gt=(batch['gt'].double()+1)/2*2047
            mse=((gt-pred)**2).mean((2,3));mu=gt.mean((2,3))
            values=25.*torch.sqrt((mse/mu.clamp_min(1e-12)**2).mean(1));total+=float(values.sum());count+=len(values)
        self.assertEqual(E.evaluate_validation(model,dataset,'cpu')['ergas'],total/count)
    def test_corrupt_scene_aggregate_or_mask_refused(self):
        with patch.object(E,'fr_scene_metrics',side_effect=fr_metric):
            result=E.evaluate_checkpoint(FakeModel(),{k:FakeDataset(k) for k in ('rr','fr')},'cpu','a'*64,wald=SimpleNamespace())
        for change in ('mean','mask','scene'):
            bad=deepcopy(result)
            if change=='mean':bad['rr']['ergas']+=1
            elif change=='mask':bad['fr']['masking']=True
            else:bad['rr']['per_scene'][0]['ergas']+=1
            with self.assertRaises(ValueError):E.validate_metrics(bad)


class SelectionTests(unittest.TestCase):
    def test_completion_uses_supplied_plan_contract_without_paper_identity_inflation(self):
        from g23sens.plan import verify_receipt
        value=summary(same=True);receipt=P.completion_document(value,'a'*64)
        verify_receipt(value['case'],receipt)
        self.assertEqual(receipt['paper_identity_status'],'PAPERSET_IDENTITY_UNVERIFIED')
        self.assertEqual(receipt['secondary']['alias_of'],'EXACT_50000')
        self.assertEqual(receipt['summary_sha256'],object_sha(value))
        bad=deepcopy(receipt);bad['fr_checkpoint_sha256']='b'*64
        with self.assertRaises(ValueError):verify_receipt(value['case'],bad)
    def test_summary_before_receipt_crash_recovers_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            wd=Path(directory);value=summary();atomic_json(wd/'official/summary.json',value)
            cfg={'g23sens':{'attempt':0}};case=value['case']
            with patch.object(P,'_config',return_value=(cfg,case,{},wd)),\
                    patch.object(P,'verify_summary_for_upload',return_value=value) as verify,\
                    patch.object(P,'evaluate_checkpoint',side_effect=AssertionError('no inference on complete summary')):
                self.assertEqual(P.run_postrun('unused',wd,'cpu'),value)
            self.assertTrue((wd/'official/COMPLETE.json').is_file())
            self.assertFalse(verify.call_args_list[0].kwargs['require_completion'])
            self.assertEqual(json.loads((wd/'official/COMPLETE.json').read_text())['summary_file_sha256'],sha256(wd/'official/summary.json'))
    def fixture(self,root):
        c=make_case('s4',0,'BASE');cfg={'g23sens':{'case':c,'source_identity':{},'binding_sha256':'a'*64}}
        wd=Path(root);identities={};records=[]
        for update in P.GRID:
            identity=dict(update=update,config_sha256=object_sha(cfg),source_identity={},bindings_sha256='a'*64,
                case_spec_sha256=c['case_spec_sha256'],teacher_sha256=c['fixed']['teacher_checkpoint_sha256'],
                **{k:'b'*64 for k in ('init_U_sha256','init_A_sha256','stream_sha256','q_raw_sha256','q_weight_sha256')})
            if update==50000:
                folder=wd/'candidates/50000';folder.mkdir(parents=True)
                from g23sens.model import state_hash
                model_state={'x':torch.tensor(1.)};identity['state_hash']=state_hash(model_state)
                state=dict(identity,full_state=True,precision='fp32',model_state=model_state,update=50000,
                    scheduler={'last_epoch':50000},optimizer={'state':{1:{'step':50000}}},rng={'torch':'state'},
                    sampler={'cursor':50000,'completed_updates':50000})
                torch.save(state,folder/'training_state.pt');identity.update(full_state=True,precision='fp32',training_state_sha256=sha256(folder/'training_state.pt'))
            identities[update]=identity;records.append(dict(update=update,val_ergas=1.,checkpoint_identity=identity))
        atomic_json(wd/'meta/training_status.json',dict(status='TRAIN_COMPLETE_EVAL_PENDING',training_complete=True,actual_updates=50000))
        atomic_json(wd/'val_records.json',dict(records=records,selected_step=1010))
        return cfg,wd,identities,records
    def test_tie_chooses_earlier_native_validation_step(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg,wd,ids,_=self.fixture(directory)
            with patch('g23sens.model.load_model',side_effect=lambda cfg,path,device:(None,ids[int(path.name)])):
                chosen=P.selected_evidence(cfg,wd)
            self.assertEqual(chosen['EXACT_50000']['identity']['update'],50000)
            self.assertEqual(chosen['RR_VAL_ERGAS_MIN']['identity']['update'],1010)
    def test_missing_grid_or_wrong_test_selected_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg,wd,ids,records=self.fixture(directory)
            atomic_json(wd/'val_records.json',dict(records=records,selected_step=50000))
            with self.assertRaises(ValueError):P.selected_evidence(cfg,wd)
            atomic_json(wd/'val_records.json',dict(records=records[1:],selected_step=2020))
            with self.assertRaises(ValueError):P.selected_evidence(cfg,wd)


if __name__=='__main__':unittest.main()
