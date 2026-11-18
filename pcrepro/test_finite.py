"""Bounded execution and measurement checks; no GPU, training run, or network."""
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch.nn import functional as F
from pcrepro import finite
from pcrepro.common import atomic_json,case_camp,run_dir,object_sha,read_json
from pcrepro.plan import Case,FiniteCase,case_for,build_config,validate_config
from pcrepro.finite_cost import prepared
from pcrepro.model import build_model
from tools.additional_models_flops import count_forward,self_check


class FiniteStudyTests(unittest.TestCase):
    def test_two_fresh_50k_cases_and_legacy_routing_are_distinct(self):
        cases=finite.schedule([2025])
        self.assertEqual([(c.dataset,c.seed,c.updates) for c in cases],[('QB',2025,50000),('GF2',2025,50000)])
        for c in cases:
            self.assertEqual(case_for(c.run_id),c)
            self.assertEqual(validate_config(build_config(c)),c)
            self.assertNotEqual(case_camp('/tmp',c),case_camp('/tmp',Case('s5',0,c.dataset)))
        legacy=Case('s5',0,'GF2')
        self.assertEqual(case_for(legacy.run_id),legacy)

    def test_invalid_or_duplicate_scope_is_rejected(self):
        for seeds in ([],[2025,2025],[-1],[True],[2**32]):
            with self.assertRaises(ValueError):finite.schedule(seeds)
        for sensor in ('WV3','WV2'):
            with self.assertRaises(ValueError):FiniteCase(sensor)

    def test_gpu_busy_or_unknown_does_not_stop_processes(self):
        import os
        with mock.patch('pcrepro.handoff.gpu_processes',return_value=[os.getpid()]):finite.require_idle()
        for pids in (None,[os.getpid()+1]):
            with mock.patch('pcrepro.handoff.gpu_processes',return_value=pids):
                with self.assertRaises(RuntimeError):finite.require_idle()

    def test_legacy_upload_and_spool_reject_new_cases(self):
        from pcrepro.upload import upload_run,spool_run
        for fn in (lambda run:upload_run(run,activated=True),spool_run):
            with self.assertRaisesRegex(ValueError,'addtional-models'):fn(FiniteCase('QB').run_id)

    def test_gpu_worker_rejects_s1_before_loading_registration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'gspread').mkdir();(root/'gspread/server.txt').write_text('s1')
            with mock.patch.object(finite,'verify_registration') as registration:
                with self.assertRaisesRegex(ValueError,'s5 routing'):finite.worker(root,'train',FiniteCase('QB').run_id)
                registration.assert_not_called()

    def test_train_requires_matching_full_batch_preflight_and_controller(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=FiniteCase('QB');source={'source':'fixture'}
            cfg=build_config(c,'manifest.json','a'*64,source)
            path=run_dir(c,root)/'meta/config.resolved.yaml';atomic_json(path,cfg)
            registration=dict(cases=[c.to_dict()],source_identity=source)
            atomic_json(finite.folder(root)/'status.json',dict(status='TRAINING',active_run=c.run_id))
            ready=dict(complete=True,source_identity=source,data_sha='a'*64,
                full_batch_smoke=dict(passed=True,pairs=48,mars_batch=96))
            ready_path=finite.folder(root)/'preflight/QB.json';atomic_json(ready_path,ready)
            with mock.patch.object(finite,'verify_registration',return_value=registration):
                self.assertEqual(finite.authorize_train(root,'s5',path),c)
                for field in ('complete','full_batch_smoke','data_sha'):
                    bad=deepcopy(ready);bad.pop(field);atomic_json(ready_path,bad)
                    with self.assertRaises(ValueError):finite.authorize_train(root,'s5',path)
                atomic_json(ready_path,ready)
                atomic_json(finite.folder(root)/'status.json',dict(status='TRAINING',active_run=FiniteCase('GF2').run_id))
                with self.assertRaises(ValueError):finite.authorize_train(root,'s5',path)

    def test_interrupted_prepare_restores_only_missing_configs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bindings=root/'bindings.json'
            atomic_json(bindings,dict(datasets={'QB':'qb.json','GF2':'gf2.json'}))
            registration=dict(cases=[c.to_dict() for c in finite.schedule([2025])],bindings_path=str(bindings),
                data_sha256={'QB':'a'*64,'GF2':'b'*64},source_identity={'fixture':True})
            finite.materialize_configs(registration,root)
            qb=run_dir(FiniteCase('QB'),root)/'meta/config.resolved.yaml'
            qb.unlink();finite.materialize_configs(registration,root);self.assertTrue(qb.is_file())
            atomic_json(qb,{'unrelated':True})
            with self.assertRaises(ValueError):finite.materialize_configs(registration,root)

    def test_runner_finishes_two_cases_and_stops_on_worker_failure(self):
        cases=finite.schedule([2025]);registration=dict(cases=[c.to_dict() for c in cases])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with mock.patch.object(finite,'verify_registration',return_value=registration), \
                    mock.patch.object(finite,'require_s5'),mock.patch.object(finite,'require_idle'), \
                    mock.patch.object(finite,'locked',side_effect=lambda *_:nullcontext()), \
                    mock.patch.object(finite,'export_rows',return_value=dict(complete=True,rows=[{},{}])), \
                    mock.patch.object(finite.subprocess,'call',return_value=0) as child:
                self.assertEqual(finite.run(root),0)
                commands=[call.args[0] for call in child.call_args_list]
                self.assertEqual([c[-1] for c in commands],[cases[0].run_id]*3+[cases[1].run_id]*3)
                self.assertEqual([c[-3] for c in commands],['preflight','train','postrun']*2)
                self.assertEqual(read_json(finite.folder(root)/'status.json')['status'],'COMPLETE')
                child.reset_mock();child.return_value=75
                self.assertEqual(finite.run(root),75);self.assertEqual(child.call_count,1)
                self.assertEqual(read_json(finite.folder(root)/'status.json')['status'],'PAUSED')

    def test_finite_exact_checkpoint_selection_uses_its_own_campaign(self):
        from pcrepro.test_postrun import SelectedEvidenceTests
        from pcrepro.postrun import selected_evidence
        fixture=SelectedEvidenceTests();fixture.setUp()
        try:
            fixture.case=FiniteCase('QB');fixture.wd=run_dir(fixture.case,fixture.root)
            fixture.cfg=build_config(fixture.case,str(fixture.path),object_sha(fixture.manifest),fixture.source)
            atomic_json(fixture.wd/'meta/config.resolved.yaml',fixture.cfg)
            fixture.context.update(campaign_id=fixture.case.to_dict()['campaign_id'],run_id=fixture.case.run_id,
                config_sha=object_sha(fixture.cfg),model_seed=2025,num_bands=4)
            fixture.grid.update(fixture.context);fixture.status.update(fixture.context)
            for identity in fixture.identities.values():identity.update(fixture.context)
            fixture._write()
            result=selected_evidence(fixture.case,fixture.root)
            self.assertEqual(result['EXACT_50000']['identity']['update'],50000)
            self.assertEqual(result['RR_VAL_ERGAS_MIN']['identity']['update'],2000)
        finally:fixture.doCleanups()

    def test_measurement_path_equals_native_nonzero_model(self):
        model=build_model(4,seed=2025,hidden_size=8,depth=(1,1,1),num_heads=2).eval()
        with torch.no_grad():model.output[-1].weight.normal_(0,.01)
        pan=torch.rand(1,1,16,16);ms=torch.rand(1,4,4,4);mode=torch.ones(1,dtype=torch.long)
        with torch.inference_mode():
            native=model(pan,ms);candidate=prepared(model,pan,ms,mode)
            self.assertTrue(torch.equal(native,candidate))
            self.assertFalse(torch.equal(native,F.interpolate(ms,size=(16,16),mode='bicubic',align_corners=False)))
        report=count_forward(lambda:prepared(model,pan,ms,mode))
        self.assertEqual(report['unsupported_operator_calls'],{})
        self.assertGreater(report['components']['softmax_estimate'],0)
        self.assertTrue(any('im2col' in k for k in report['non_arithmetic_calls']))

    def test_arithmetic_counter_transposed_conv_and_unfold(self):
        self_check()
        x=torch.ones(1,2,3,3);layer=torch.nn.ConvTranspose2d(2,3,2,stride=2,bias=True)
        report=count_forward(lambda:layer(x))
        self.assertEqual(report['flops'],2*(1*2*3*3)*3*2*2+1*3*6*6)
        report=count_forward(lambda:F.unfold(x,kernel_size=3,padding=1))
        self.assertEqual(report['flops'],0)
        self.assertEqual(count_forward(lambda:F.gelu(x))['flops'],5*x.numel())
        self.assertEqual(count_forward(lambda:F.gelu(x,approximate='tanh'))['flops'],9*x.numel())


if __name__=='__main__':unittest.main()
