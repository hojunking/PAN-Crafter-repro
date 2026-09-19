#!/usr/bin/env python3
import re
import csv
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from fh20r1.plan import *


class PlanTests(unittest.TestCase):
    def test_every_supplied_csv_definition_and_full_reference_sha(self):
        with (ROOT/SOURCE_CASES).open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),145)
        self.assertEqual({r['case_id'] for r in rows},{c.run_id for c in CASES})
        for r in rows:
            c=case_for(r['case_id']); cfg=build_config(c); f=cfg['fh20r1']
            expected=dict(kind='TRAIN',role=c.role,server=c.server_id,tier=c.tier,
                eligible_when=c.eligible_when,layout=c.input_layout,input_channels=str({'P0':9,'PL':10,'PH':10,'PLH':11}[c.input_layout]),
                width=str(c.width),teacher_ref_alias=c.teacher_alias if c.role=='S' else '',
                teacher_seed=str(c.teacher_seed),student_seed=str(c.student_seed or ''),profile=c.profile,
                total_updates='50000',teacher_run_id=c.teacher_run_id if c.role=='S' else '',
                planned_config_path=f'config/{c.run_id}.yaml')
            for key,value in expected.items(): self.assertEqual(r[key],value,(c.run_id,key))
            self.assertEqual(json.loads(r['depth']),list(c.depth))
            self.assertEqual(json.loads(r['block_ids']),[c.block_id])
            for key,value in [('lr_U_peak',cfg['learning_rate']),('lr_A_peak',f['aligner_lr']),
                              ('A_multiplier_after_10000',f['a_lr_after_multiplier']),
                              ('alpha',f['alpha']),('beta',f['beta']),('lambda_E',f['lambda_E'])]:
                if r[key]: self.assertEqual(float(r[key]),value,(c.run_id,key))
                else: self.assertEqual(c.role,'T',(c.run_id,key))
            if c.role=='S' and c.teacher_alias!='N2PL':
                self.assertEqual(r['teacher_checkpoint_sha256'],REFERENCE_ASSETS[c.teacher_alias]['expected_sha256'])
    def test_counts_exclude_mutually_exclusive_alternatives(self):
        self.assertEqual(len(CASES),145)
        self.assertEqual(sum(len(active_cases(s,include_reserve=False)) for s in SERVERS),85)
        self.assertEqual(sum(len(active_cases(s)) for s in SERVERS),125)
        self.assertEqual(len(registry_document()['stages']),11)
        self.assertEqual([len(active_cases(s,include_reserve=False)) for s in SERVERS],[14,11,20,20,20])

    def test_all_block_orders_match_supplied_md_independently(self):
        text=(ROOT/SOURCE_PLAN).read_text()
        sections=re.split(r'^### (FH20R1_S[1-5]_B\d{2}) /[^\n]*\n',text,flags=re.M)
        expected={}
        for i in range(1,len(sections),2):
            code=re.findall(r'```text\n(.*?)```',sections[i+1],flags=re.S)
            expected[sections[i]]=[tuple(x.strip().splitlines()) for x in code[:2]]
        self.assertEqual(len(expected),51)
        for block in BLOCKS:
            self.assertEqual(block.primary_order,expected[block.block_id][0],block.block_id)
            if block.alternative_orders:
                self.assertEqual(list(block.alternative_orders.values()),expected[block.block_id][1:],block.block_id)

    def test_exact_ids_and_alternative_common_baseline_unique(self):
        all_ids={c.run_id for c in CASES}
        mdids=set(re.findall(r'FH20R1_S[1-5]_[ST]_[A-Z0-9_]+_v1',(ROOT/SOURCE_PLAN).read_text()))
        self.assertEqual(all_ids,mdids)
        for s,branch in [('s1',S1_FALLBACK),('s2',S2_FALLBACK)]:
            selected=active_cases(s,branch)
            self.assertEqual(len({c.run_id for c in selected}),len(selected))
            self.assertFalse(any(c.profile=='A10' or c.teacher_alias=='N2PL' for c in selected))

    def test_no_fh12_window_or_source_mutation_in_configs(self):
        for c in CASES:
            cfg=build_config(c); f=cfg['fh20r1']
            self.assertNotIn('fh12',cfg)
            self.assertNotIn('window_path',f)
            self.assertEqual(cfg['num_iter'],50000)
            self.assertEqual(cfg['num_worker'],4)
            self.assertEqual(cfg['batch_size'],48)
            self.assertEqual(f['min_effective_hours'],20)
            self.assertEqual(f['candidate_grid'],list(GRID_STEPS))
            self.assertNotIn(10000,GRID_STEPS)
            self.assertIn(10000,f['fullstate_steps'])
            self.assertEqual(f['init_policy'],'FH12_named_tensor_fresh9_zero_extra_v1')
            self.assertEqual(f['teacher_from_scratch'],False)

    def test_same_seed_pairs_differ_only_declared_policy_or_architecture(self):
        for b in blocks_for('s1'):
            a,z=[build_config(i) for i in b.primary_order]
            self.assertEqual(a['seed'],z['seed'])
            self.assertEqual(a['model_args'],z['model_args'])
            self.assertEqual(a['fh20r1']['corruption_seed'],z['fh20r1']['corruption_seed'])
            self.assertEqual(a['learning_rate'],z['learning_rate'])
        self.assertEqual(build_config(N2_TEACHER_RUN)['seed'],71001)
        self.assertEqual(build_config(N2_TEACHER_RUN)['fh20r1']['aligner_init'],'N2_A_ONLY')

    def test_branch_from_other_server_rejected(self):
        with self.assertRaises(ValueError): active_cases('s3',S1_FALLBACK)


if __name__=='__main__': unittest.main(verbosity=2)
