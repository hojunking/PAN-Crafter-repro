"""C17 append-only migration / actual pairing checks, with no GPU or network."""
from dataclasses import asdict,replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ablr2 import extension as ex
from ablr2 import controller as ctl
from ablr2.common import camp,read_json,atomic_json,run_dir
from ablr2.plan import legacy_cases_for,cases_for,full_wave,case_for,validate_case


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.folder=camp(self.root,'s1')
        self.legacy=legacy_cases_for('s1')
        self.state=dict(campaign_id=self.legacy[0].campaign_id,server='s1',sensor='WV3',cycle=3,
            active_stage='BOOT5',runs={},observations=[dict(hours=9.2)],stages=[dict(stage_id='BOOT5',
            kind='BOOT5',run_ids=[c.run_id for c in self.legacy],complete=False)])
        atomic_json(self.folder/'state.json',self.state)
        atomic_json(self.folder/'cases.json',{'cases':{c.run_id:asdict(c) for c in self.legacy}})

    def test_discovers_all_five_and_preserves_original_state_cases(self):
        before=(self.folder/'state.json').read_bytes()
        original=read_json(self.folder/'cases.json')['cases']
        debts=ex.sync(self.root,'s1',self.state)
        self.assertEqual(len(debts['debts']),5)
        self.assertEqual((self.folder/'state.json').read_bytes(),before)
        after=read_json(self.folder/'cases.json')['cases']
        self.assertEqual({run:after[run] for run in original},original)
        self.assertEqual(len(after),100)
        frozen=(self.folder/'extension/task_ledger.jsonl').read_bytes()
        ex.sync(self.root,'s1',self.state)
        self.assertEqual((self.folder/'extension/task_ledger.jsonl').read_bytes(),frozen)

    def test_registered_refreshes_are_discovered_not_only_boot(self):
        refresh=full_wave('s1','REFRESH5','R04','r004','W3',2026,[])
        registry=read_json(self.folder/'cases.json')
        registry['cases'].update({c.run_id:asdict(c) for c in refresh if c.case_id!='C17'})
        atomic_json(self.folder/'cases.json',registry)
        self.state['stages'].append(dict(stage_id='W3',kind='REFRESH5',
            run_ids=[c.run_id for c in refresh if c.case_id!='C17'],complete=True))
        debts=ex.sync(self.root,'s1',self.state)['debts']
        self.assertEqual(len(debts),10)
        for row in debts.values():
            c=case_for(row['run_id'],self.root)
            self.assertEqual(c.teacher_kind,'TZERO');self.assertEqual(c.alpha,0)
            self.assertFalse(c.requires_calibration)

    def test_fresh_full100_does_not_rewrite_dynamic_c17_rank(self):
        jobs=full_wave('s1','REFRESH5','R00','r000','W1',19,[])
        atomic_json(self.folder/'cases.json',{'cases':{c.run_id:asdict(c) for c in jobs}})
        self.state['stages']=[dict(stage_id='W1',kind='REFRESH5',run_ids=[c.run_id for c in jobs])]
        before=(self.folder/'cases.json').read_bytes()
        ex.sync(self.root,'s1',self.state)
        self.assertEqual((self.folder/'cases.json').read_bytes(),before)

    def test_partial_extension_not_presented_as_balanced18(self):
        ex.sync(self.root,'s1',self.state)
        effective=ex.stage_cases(self.root,'s1',self.state['stages'][0])
        self.assertEqual(len(effective),100)
        for i,c in enumerate(effective):
            if c.case_id=='C17':self.assertEqual(effective[i-1].case_id,'C03')
        one=next(c for c in effective if c.case_id=='C17')
        self.state['runs'][one.run_id]={'complete':True}
        self.assertEqual(len(ex.stage_cases(self.root,'s1',self.state['stages'][0],completed_only=True,state=self.state)),95)
        for c in effective:self.state['runs'][c.run_id]={'complete':True}
        self.assertEqual(len(ex.stage_cases(self.root,'s1',self.state['stages'][0],completed_only=True,state=self.state)),100)

    def test_repair_is_separate_c03_same_seed_not_component18(self):
        anchor=next(c for c in self.legacy if c.case_id=='C03')
        repair=ex.repair_anchor(anchor)
        validate_case(repair)
        self.assertEqual((repair.case_id,repair.seed,repair.teacher_run_id,repair.paired_init_group),
                         ('C03',anchor.seed,anchor.teacher_run_id,anchor.paired_init_group))
        self.assertNotEqual(anchor.run_id,repair.run_id)
        self.assertEqual(repair.phase,'PAIR_REPAIR')

    def test_unavailable_zero_is_visible_and_does_not_substitute_seed(self):
        ex.sync(self.root,'s1',self.state)
        for c in self.legacy:self.state['runs'][c.run_id]={'complete':True}
        with patch.object(ex,'prepare_anchor',side_effect=FileNotFoundError('TZERO missing')):
            self.assertEqual(ex.ready_debts(self.root,'s1',self.state),[])
        debts=read_json(self.folder/'extension/debts.json')['debts']
        self.assertEqual({r['status'] for r in debts.values()},{'REFERENCE_UNAVAILABLE'})

    def test_anchor_unproven_creates_repair_without_rewriting_original(self):
        ex.sync(self.root,'s1',self.state)
        c17=next(c for c in cases_for('s1') if c.case_id=='C17')
        data={'splits':{'train':{'count':9714}}};atomic_json(self.folder/'dataset_manifest.json',data)
        with patch('ablr2.references.validate_teacher_pair',return_value={'passed':True}), \
             patch('ablr2.analysis.case_report',side_effect=ValueError('old source not comparable')), \
             patch.object(ex,'expected_student_identity',return_value={'init_U_sha256':'a'*64}), \
             patch.object(ex,'source_identity',return_value={'source':'new'}):
            repair=ex.prepare_anchor(c17,self.root)
        self.assertEqual(repair.phase,'PAIR_REPAIR')
        receipt=read_json(run_dir(c17.run_id,self.root)/'meta/paired_anchor.json')
        self.assertEqual(receipt['mode'],'PAIR_REPAIR')
        self.assertFalse(receipt['old_observation_rewritten'])
        original=case_for(receipt['original_c03_run_id'],self.root)
        self.assertEqual(original,next(c for c in self.legacy if c.run_id==original.run_id))

    def test_pairing_analysis_requires_receipt_not_same_seed_assumption(self):
        c17=next(c for c in cases_for('s1') if c.case_id=='C17')
        with self.assertRaises(FileNotFoundError):ex.paired_anchor_for(c17,self.root)


if __name__=='__main__':unittest.main()
