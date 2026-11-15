from copy import deepcopy
import json
import unittest
from panda_input3 import plan, publication as pub
from panda_fr_columns.publication import original_rows, extras
from panda_fr_columns.schema import expand
from . import reporting
from .policy import passes, choose, POLICY, hqnr_passes


def record(cfg,step=1000,**changes):
    t=plan.targets(cfg['sensor'])
    receipt={'schema':'PANDA_INPUT3_FIXED_DONOR_V1','server':cfg['server'],
        'donor_run_id':cfg['donor_run_id'],'selected':{'step':cfg['donor_step']}}
    receipt['receipt_sha256']=plan.digest(receipt)
    r={'step':step,'donor_selection':receipt,
       'rr':{'metrics':{'ERGAS':t['ERGAS_max'],'PSNR':t.get('PSNR_min',38.1)}},
       'fr':{'metrics':{'HQNR':.8,'D_s':.18,'D_lambda':.02,'JQM':.8}},
       'fr_corrected_pan':{'metrics':{'HQNR':t['HQNR_min'],'D_s':.02,'D_lambda':.02,'JQM':.9}},
       'checkpoint':{'file_sha256':'a'*64},'model_sha256':'b'*64,'evidence_path':'fixture.json',
       'evidence_sha256':'c'*64,'recorded_at':'2026-11-15T00:00:00+00:00',
       'elapsed_seconds':1.,'jqm_variant':'fixture'}
    if 'h' in changes:r['fr_corrected_pan']['metrics']['HQNR']=changes['h']
    if 'e' in changes:r['rr']['metrics']['ERGAS']=changes['e']
    if 'p' in changes:r['rr']['metrics']['PSNR']=changes['p']
    return r


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):reporting.install()

    def test_exact_targets_corrected_reference_and_full_precision(self):
        for server in ('s1','s2','s3'):
            cfg=plan.queue(server)[0];t=plan.targets(cfg['sensor'])
            self.assertTrue(passes(record(cfg),cfg))
            self.assertFalse(passes(record(cfg,h=t['HQNR_min']-1e-7),cfg))
            self.assertFalse(passes(record(cfg,e=t['ERGAS_max']+1e-7),cfg))
            self.assertFalse(passes(record(cfg,h=float('nan')),cfg))
            self.assertFalse(passes(record(cfg,e=float('inf')),cfg))
            r=record(cfg,h=t['HQNR_min']-0.1);r['fr']['metrics']['HQNR']=.999
            self.assertFalse(passes(r,cfg))
        gf=plan.queue('s3')[0]
        self.assertFalse(passes(record(gf,p=45.07999999),gf))
        self.assertFalse(passes(record(gf,p=float('nan')),gf))

    def test_never_combine_metrics_from_different_steps_or_use_endpoint_fallback(self):
        for server in ('s1','s2','s3'):
            cfg=plan.queue(server)[0];t=plan.targets(cfg['sensor'])
            rows=[record(cfg,h=t['HQNR_min']-.01),record(cfg,2000,e=t['ERGAS_max']+.1),
                  record(cfg,cfg['horizon'],e=t['ERGAS_max']+.2)]
            selected=choose(rows,'stage2',cfg)
            self.assertEqual(len(selected),1)
            self.assertEqual(selected[0][0]['step'],2000)
            self.assertTrue(hqnr_passes(selected[0][0],cfg))
            self.assertFalse(passes(selected[0][0],cfg))
            self.assertEqual(choose([rows[0]],'stage2',cfg),[])
        cfg=plan.queue('s3')[0]
        self.assertEqual(len(choose([record(cfg,p=45.07)],'stage2',cfg)),1)

    def test_rep_minimum_ergas_and_only_complete_target_steps(self):
        cfg=plan.queue('s3')[0]
        rows=[record(cfg,1000,e=.50,p=45.07),record(cfg,2000,e=.51),
              record(cfg,3000,e=.505),record(cfg,4000,e=.505,p=45.2),
              record(cfg,cfg['horizon'],h=.90,e=.49,p=46)]
        selected=choose(rows,'stage2',cfg)
        self.assertEqual(selected[0][0]['step'],1000)
        self.assertEqual(selected[0][1],'HQNR_QUALIFIED_MIN_ERGAS')
        self.assertEqual([r['step'] for r,_,s in selected if s!='ROW0'],[2000,3000,4000])
        self.assertTrue(all(passes(r,cfg) for r,_,slot in selected if slot!='ROW0'))
        with self.assertRaises(ValueError):choose([rows[0],record(cfg,1000,e=.52)],'stage2',cfg)

    def test_status_has_no_metrics_and_failed_result_cannot_be_built(self):
        cfg=plan.queue('s1')[0]
        row,note=pub.make_row(cfg,'COMPLETED',None,'STATUS','ROW0_ORIGINAL')
        self.assertIn('HQNR 조건 충족 모델 없음',row[pub.INDEX['Selection@Step']])
        self.assertEqual(row[pub.INDEX['Selected_step']],0)
        for metric in pub.METRICS.values():self.assertIn(row[pub.INDEX[metric]],('',None))
        with self.assertRaises(ValueError):pub.make_row(cfg,'COMPLETED',record(cfg,h=.9))
        self.assertFalse(json.loads(note)['publication']['fallback_enabled'])

    def test_dual_fr_preserves_single_checkpoint_and_passing_label(self):
        cfg=plan.queue('s3')[0];r=record(cfg,p=45.08)
        row,note=pub.make_row(cfg,'RUNNING',r,'HQNR_QUALIFIED_MIN_ERGAS','ROW0_ORIGINAL')
        value={'rows':[row],'proof':{(cfg['run_id'],r['step']):r}}
        corr=[c['userEnteredValue']['numberValue'] for c in next(iter(extras(value).values()))]
        physical=expand(row,corr)
        self.assertEqual(physical[3],.8);self.assertEqual(physical[7],.9645)
        self.assertIn('전체 목표 충족',row[pub.INDEX['Selection@Step']])
        self.assertTrue(json.loads(note)['publication']['is_case_representative'])

    def test_representative_exception_labels_each_failed_rr_target(self):
        cfg=plan.queue('s3')[0];r=record(cfg,e=.53,p=45.07)
        row,note=pub.make_row(cfg,'RUNNING',r,'HQNR_QUALIFIED_MIN_ERGAS','ROW0_ORIGINAL')
        label=row[pub.INDEX['Selection@Step']]
        self.assertIn('HQNR 충족',label);self.assertIn('ERGAS 미달',label);self.assertIn('PSNR 미달',label)
        self.assertFalse(json.loads(note)['publication']['all_dataset_targets_pass'])
        with self.assertRaises(ValueError):pub.make_row(cfg,'RUNNING',r,'PASS_ALL_STEPS','STEP_0001000_ORIGINAL')
        r=record(cfg,e=.51,p=45.07)
        row,_=pub.make_row(cfg,'RUNNING',r,'HQNR_QUALIFIED_MIN_ERGAS','ROW0_ORIGINAL')
        self.assertNotIn('ERGAS 미달',row[pub.INDEX['Selection@Step']])
        self.assertIn('PSNR 미달',row[pub.INDEX['Selection@Step']])

    def test_retirement_validates_legacy_and_preserves_foreign_rows(self):
        cfg=plan.queue('s1')[0];r=record(cfg,h=.9)
        with reporting.legacy_view():
            old,note=pub.make_row(cfg,'RUNNING',r,'MIN_ERGAS_NO_GATE','MIN_ERGAS_ORIGINAL')
        wanted,wn=pub.make_row(cfg,'RUNNING',None,'STATUS','ROW0_ORIGINAL')
        value={'rows':[wanted],'notes':[wn],'proof':{(cfg['run_id'],r['step']):r},'configs':{cfg['run_id']:cfg}}
        foreign=['']*64;foreign[pub.INDEX['Campaign_ID']]='other'
        snap={'rows':[foreign,old],'notes':['',note]}
        self.assertEqual(reporting.retired_rows(snap,value,'s1'),[7])
        self.assertEqual(pub.POLICY,POLICY)
        bad=deepcopy(old);bad[pub.INDEX['HQNR↑']]=.999
        with self.assertRaises(ValueError):reporting.retired_rows({'rows':[bad]},value,'s1')
        self.assertEqual(pub.POLICY,POLICY)

    def test_previous_strict_status_can_be_upgraded_to_hqnr_representative(self):
        cfg=plan.queue('s1')[0];r=record(cfg,e=2.1)
        with reporting.legacy_view(reporting.STRICT_V2_POLICY):
            old,note=pub.make_row(cfg,'RUNNING',None,'STATUS','ROW0_ORIGINAL')
        wanted,wn=pub.make_row(cfg,'RUNNING',r,'HQNR_QUALIFIED_MIN_ERGAS','ROW0_ORIGINAL')
        value={'rows':[wanted],'notes':[wn],'proof':{(cfg['run_id'],r['step']):r},'configs':{cfg['run_id']:cfg}}
        self.assertEqual(reporting.retired_rows({'rows':[old]},value,'s1'),[])
        reporting.verify_owned(old,value,'s1')
        reporting.verify_owned(wanted,value,'s1')
        self.assertEqual(pub.POLICY,POLICY)


if __name__=='__main__':unittest.main()
