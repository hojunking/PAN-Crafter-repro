import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from donor30.artifacts import _same
from panda_method.calibration import state_dict_sha256
from panda_sensor5.tests.test_core import sources
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.stream import TrainStream, StreamSpec
from panda_qb40 import (plan as old_plan, stage1 as old_stage1, engine as old_engine,
                        spec as old_spec, calibration as old_calibration)
from panda_qb40.tests.test_execution import donor as old_donor, bank, PINS
from panda_qb40.tests.test_runner import FakeCampaign, complete
from panda_qb_s2 import plan, stage1, engine, spec, calibration, runner, reporting


class ScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_six_donors_match_reference_operators_and_requested_loss_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = sources(tmp, 'QB', count=1)
            for case in plan.DONORS:
                with self.subTest(case=case):
                    cfg = plan.resolve_case('s2', case)
                    server = 's1' if cfg['seed'] == 261016101 else 's3'
                    ref = old_stage1.build_stage1(old_plan.resolve_case(server, 'QB5_S1_A400_E030'), reg.sources)
                    ref.stage.lambda_abs = cfg['lambda_abs']
                    ref.stage.lambda_eps = cfg['lambda_epsilon']
                    fresh = stage1.build_stage1(cfg, reg.sources)
                    x, y = ref.stage(reg.batch([0])), fresh.stage(reg.batch([0]))
                    for key in ('loss','rec','abs_rr','eps_rr'):
                        self.assertTrue(torch.equal(x[key],y[key]),key)
                    x['loss'].backward(); y['loss'].backward()
                    for a,b in zip(ref.stage.parameters(),fresh.stage.parameters()):
                        self.assertEqual(a.grad is None,b.grad is None)
                        if a.grad is not None:self.assertTrue(torch.equal(a.grad,b.grad))

    def test_stage1_complete_optimizer_rng_sampler_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg=sources(tmp,'QB',count=12)
            cfg=plan.resolve_case('s2','QB6_S1_A400_E060_D101')
            trainer=stage1.build_stage1(cfg,reg.sources)
            stream=stage1.Stage1Stream(reg.sources,cfg,device='cpu')
            stage1.advance_stage1(trainer,stream)
            saved=stage1.save_stage1(Path(tmp)/'resume.pt',trainer,stream,**PINS)
            expected=stage1.advance_stage1(trainer,stream)
            state=deepcopy(trainer.state_dict())
            fresh,restream,_=stage1.load_stage1(Path(tmp)/'resume.pt',saved['file_sha256'],cfg,reg.sources,**PINS)
            self.assertTrue(_same(expected,stage1.advance_stage1(fresh,restream)))
            self.assertTrue(_same(state,fresh.state_dict()))
            self.assertTrue(_same(stream.state_dict(),restream.state_dict()))

    def test_real_stage1_loss_and_gradients_match_qb40(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = sources(tmp, 'QB', count=1)
            before = old_stage1.build_stage1(old_plan.resolve_case('s1', 'QB5_S1_A400_E030'), reg.sources)
            after = stage1.build_stage1(plan.resolve_case('s2', 'QB6_S1_A400_E030_D101'), reg.sources)
            a, b = before.stage(reg.batch([0])), after.stage(reg.batch([0]))
            for key in ('loss', 'rec', 'abs_rr', 'eps_rr'):
                self.assertTrue(torch.equal(a[key], b[key]), key)
            a['loss'].backward(); b['loss'].backward()
            for (an, ap), (bn, bp) in zip(before.stage.named_parameters(), after.stage.named_parameters()):
                self.assertEqual(an, bn)
                self.assertTrue(torch.equal(ap, bp))
                self.assertEqual(ap.grad is None, bp.grad is None)
                if ap.grad is not None:
                    self.assertTrue(torch.equal(ap.grad, bp.grad), an)

    def test_real_student_parity_and_complete_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = sources(tmp, 'QB', count=12)
            oldcfg = old_plan.resolve_case('s1', 'QB5_A400_E030')
            newcfg = plan.resolve_case('s2', 'QB6_S2_A400_E030_D101')
            original = old_donor(old_plan.resolve_case('s1', 'QB5_S1_A400_E030'), reg)
            binding = deepcopy(original.binding)
            binding['source_campaign_id'] = plan.CAMPAIGN_ID
            for key in ('band30_teacher', 'mature30_teacher'):
                binding[key]['campaign_id'] = plan.CAMPAIGN_ID
            binding['donor_run_id'] = plan.resolve_case('s2', 'QB6_S1_A400_E030_D101')['run_id']
            newdonor = LoadedDonor(original.aligner, original.teacher, binding, {'synthetic_only': True})
            model = spec.MatureSpec(**newcfg['stage2_spec'])
            oldmodel = old_spec.MatureSpec(**oldcfg['stage2_spec'])
            original = old_calibration.bind_donor(original, oldmodel)
            newdonor = calibration.bind_donor(newdonor, model)
            from panda_qb_s2.calibration_core import CalibrationBuilder
            values = torch.linspace(.1, .9, 64*64).reshape(1, 1, 64, 64).repeat(len(reg.sample_ids), 1, 1, 1)
            collector = CalibrationBuilder(list(reg.sample_ids), calibration.binding_for_geometry(newdonor.binding, 's100'),
                k=5, memory_budget_bytes=64*1024**2, include_condq=True, mask_policy=model.mask_policy)
            collector.add(list(reg.sample_ids), values, values.flip(-1), values.flip(-2), sensor='QB',
                          loss_mask=torch.ones_like(values, dtype=torch.bool))
            newstats = {'s100': collector.finalize()}
            oldcfg['donor_binding'] = deepcopy(original.binding)
            newcfg['donor_binding'] = deepcopy(newdonor.binding)
            a = old_engine.build_fresh_stage2(oldcfg, original, bank(original, oldmodel, reg), evaluator_pin=PINS['evaluator_pin'])
            b = engine.build_fresh_stage2(newcfg, newdonor, newstats, evaluator_pin=PINS['evaluator_pin'])
            x, y = a.stage(reg.batch([0])), b.stage(reg.batch([0]))
            self.assertTrue(torch.equal(x['loss'], y['loss']))
            x['loss'].backward(); y['loss'].backward()
            for ap, bp in zip(a.stage.student.parameters(), b.stage.student.parameters()):
                self.assertTrue(torch.equal(ap.grad, bp.grad))
            b.optimizer.zero_grad(set_to_none=True)
            streamspec = StreamSpec(newcfg['stream_seeds']['rr_data'], newcfg['stream_seeds']['rr_augmentation'], **newcfg['augmentation'])
            stream = TrainStream(reg, streamspec, microbatch_size=12, effective_batch_size=48)
            engine.advance(b, stream)
            saved = engine.save_checkpoint(Path(tmp) / 'resume.pt', b, stream, **PINS)
            second = engine.advance(b, stream); expected = deepcopy(b.state_dict())
            restored = engine.build_fresh_stage2(newcfg, newdonor, newstats, evaluator_pin=PINS['evaluator_pin'])
            restream = TrainStream(reg, streamspec, microbatch_size=12, effective_batch_size=48)
            restored, restream = engine.load_checkpoint(Path(tmp) / 'resume.pt', saved['file_sha256'], restored, restream, **PINS)
            self.assertTrue(_same(engine.advance(restored, restream), second))
            self.assertTrue(_same(restored.state_dict(), expected))
            self.assertTrue(_same(restream.state_dict(), stream.state_dict()))
            self.assertEqual(state_dict_sha256(newdonor.aligner), newdonor.binding['aligner_state_sha256'])
