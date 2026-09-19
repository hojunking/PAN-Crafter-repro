#!/usr/bin/env python3
"""CPU/temp-only FH20R1 deployment and launch-check regressions."""
import csv
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fh20r1 import preflight as P
from fh20r1.common import atomic_json
from fh20r1.plan import SOURCE_CASES,SOURCE_PLAN,cases_for,active_cases,build_config
from tools.gen_fh20r1_configs import generate


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='fh20r1-preflight-')
        self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def definitions(self):
        for name in (SOURCE_CASES,SOURCE_PLAN):
            dest=self.root/name; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(P.ROOT/name,dest)
        generate(self.root)
    def edit_csv(self,edit):
        path=self.root/SOURCE_CASES
        with path.open(encoding='utf-8-sig',newline='') as stream: rows=list(csv.DictReader(stream))
        edit(rows)
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    def test_all_145_supplied_definitions_and_generated_release(self):
        self.definitions(); result=P.audit_definitions(self.root)
        self.assertEqual(result['n_cases'],145);self.assertEqual(result['n_blocks'],51)
    def test_changed_full_teacher_sha_rejected(self):
        self.definitions(); self.edit_csv(lambda rows:rows[0].update(teacher_checkpoint_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'whole SHA'): P.audit_definitions(self.root)
    def test_duplicate_supplied_row_rejected(self):
        self.definitions(); self.edit_csv(lambda rows:rows.append(rows[0]))
        with self.assertRaisesRegex(ValueError,'case set'): P.audit_definitions(self.root)
    def test_changed_generated_config_rejected(self):
        self.definitions(); c=active_cases('s1')[0]
        path=self.root/f'config/{c.run_id}.yaml';path.write_text(path.read_text()+'# deployment drift\n')
        with self.assertRaisesRegex(ValueError,'release mismatch'): P.audit_definitions(self.root)
    def test_check_only_never_activates_or_certifies_numerical_checks(self):
        with patch.object(P,'audit_definitions',return_value={'n_cases':145}), \
             patch('fh20r1.references.validate_origin',return_value=({},dict(teacher_checkpoint_sha256='fixture'),{},{},{})), \
             patch.object(P,'batch_smoke',side_effect=AssertionError('must not smoke')):
            result=P.run_preflight('s1','cpu',self.root,True)
        self.assertFalse(result['preflight_pass']);self.assertFalse(result['runtime_numerics_executed'])
        self.assertFalse((self.root/'work_dir').exists())
    def test_numerical_checks_require_explicit_local_activation(self):
        with patch.object(P,'audit_definitions',return_value={}),patch.object(P,'batch_smoke') as smoke:
            with self.assertRaises(FileNotFoundError): P.run_preflight('s1','cpu',self.root)
            smoke.assert_not_called()
        camp=self.root/'work_dir/_fh20r1/s1'
        atomic_json(camp/'campaign_budget.json',dict(campaign_id=P.CAMPAIGN_ID,actual_start_authorized=False,device='cpu'))
        with patch.object(P,'audit_definitions',return_value={}):
            with self.assertRaisesRegex(ValueError,'explicit'): P.run_preflight('s1','cpu',self.root)
    def test_disk_all_alternatives_reserves_and_only_complete_core_excluded(self):
        fake=types.SimpleNamespace(parameters=lambda:[torch.ones(10)])
        with patch.object(P,'build_model',return_value=(fake,{})), \
             patch.object(P.shutil,'disk_usage',return_value=types.SimpleNamespace(free=10**12)):
            first=P.disk_estimate(self.root,'s1')
            self.assertEqual(set(first['per_run_estimate_bytes']),{c.run_id for c in cases_for('s1')})
            a,b=active_cases('s1',include_reserve=False)[:2]
            atomic_json(self.root/'work_dir'/a.run_id/'official/postrun_status.json',dict(official_complete=True))
            atomic_json(self.root/'work_dir'/b.run_id/'official/postrun_status.json',dict(official_complete=False))
            second=P.disk_estimate(self.root,'s1')
            self.assertEqual(first['core_additional_estimate_bytes']-second['core_additional_estimate_bytes'],first['per_run_estimate_bytes'][a.run_id])
    def test_semantic_reuse_ignores_unstarted_metadata_not_missing_init_crash(self):
        case=active_cases('s1')[0]; cfg=build_config(case)
        path=self.root/'work_dir/unstarted/meta/config.resolved.yaml';path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(cfg))
        atomic_json(self.root/'work_dir/_fh12/s1/dataset_manifest.json',{})
        with patch.object(P,'source_identity',return_value=dict(files={'fh12/model.py':'fixture'})):
            result=P.find_reusable_runs(self.root,'s1',dict(alias='F1',teacher_checkpoint_sha256='fixture'))
        self.assertEqual(result['runs'],{})
    def test_cpu_student_smoke_is_diagnostic_only_and_disposable(self):
        from fh12.model import build_model,state_hash
        teacher,_=build_model('P0',8,[1,1,1],71001,role='T');teacher.eval().requires_grad_(False)
        before=state_hash(teacher.state_dict());bridge=dict(alias='F1',tau_R=.05)
        with patch('fh20r1.references.load_reference',return_value=(teacher,{},bridge,None)):
            result=P.batch_smoke('s1',bridge,'cpu',self.root)
        self.assertEqual(result['actual_batch48_smoke_status'],'NOT_RUN_CPU_DIAGNOSTIC')
        self.assertEqual(result['batch_size'],2);self.assertTrue(result['passed'])
        self.assertEqual(before,state_hash(teacher.state_dict()))
        self.assertFalse(result['parameters_reused_in_actual_training'])


if __name__=='__main__': unittest.main(verbosity=2)
