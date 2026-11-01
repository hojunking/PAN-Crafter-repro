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
from panda_qb_rre import plan, stage1, engine, spec, calibration, runner, reporting, finish_current, handoff


class FocusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_queue_and_scientific_recipe_parity(self):
        for server in ('s1', 's3'):
            cfg = plan.default_config(server)
            self.assertEqual((cfg['stage1_runs'], cfg['stage2_runs']), (1, 3))
            self.assertIsNone(cfg['budget']['total_seconds'])
            self.assertEqual(cfg['joint_target'], {'HQNR': .9315, 'ERGAS': 3.5})
            for case, replica in plan.G.RUNS[server]:
                new = plan.resolve_case(server, case, replica)
                old = old_plan.resolve_case(server if replica != 'R3' else 's3', case, replica)
                for key in ('training_spec', 'stage1_loss_parameters', 'geometry', 'student_architecture',
                            'augmentation', 'lr_A', 'lr_U', 'epsilon_radius_rr', 'weight_decay'):
                    self.assertEqual(new[key], old[key], key)
                self.assertEqual(new['seed'], (261016101 if server == 's1' else 261016301)
                                 if new['phase'] == 'donor' else plan.SEEDS[replica])
                self.assertTrue(new['run_id'].startswith('PQRRE_'))
                reporting._config(new)
                self.assertIn('lambda_abs=4', reporting.approach_description(new))
                self.assertIn('lambda_epsilon=0.03', reporting.approach_description(new))

    def test_real_stage1_loss_and_gradients_match_qb40(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = sources(tmp, 'QB', count=1)
            before = old_stage1.build_stage1(old_plan.resolve_case('s1', 'QB5_S1_A400_E030'), reg.sources)
            after = stage1.build_stage1(plan.resolve_case('s1', 'QB5_S1_A400_E030'), reg.sources)
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
            newcfg = plan.resolve_case('s1', 'QB5_A400_E030')
            original = old_donor(old_plan.resolve_case('s1', 'QB5_S1_A400_E030'), reg)
            binding = deepcopy(original.binding)
            binding['source_campaign_id'] = plan.CAMPAIGN_ID
            for key in ('band30_teacher', 'mature30_teacher'):
                binding[key]['campaign_id'] = plan.CAMPAIGN_ID
            binding['donor_run_id'] = plan.resolve_case('s1', 'QB5_S1_A400_E030')['run_id']
            newdonor = LoadedDonor(original.aligner, original.teacher, binding, {'synthetic_only': True})
            model = spec.MatureSpec(**newcfg['stage2_spec'])
            oldmodel = old_spec.MatureSpec(**oldcfg['stage2_spec'])
            original = old_calibration.bind_donor(original, oldmodel)
            newdonor = calibration.bind_donor(newdonor, model)
            from panda_qb_rre.calibration_core import CalibrationBuilder
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

    def test_all_four_jobs_run_even_after_old_40h_window(self):
        with tempfile.TemporaryDirectory() as tmp, patch('panda_sensor_focus.early_stop.finish_stopped', return_value=False), \
                patch('panda_sensor_focus.early_stop.run_case', side_effect=complete):
            camp = FakeCampaign(tmp, -3600); camp.config = plan.default_config('s1')
            result = runner.run(camp, catalog='fixture')
            self.assertEqual(result['status'], 'FINISHED_REGISTERED_QUEUE')
            self.assertEqual([(r['case'], r['replica']) for r in result['runs']], list(plan.G.RUNS['s1']))
            self.assertEqual(camp.registered, ['QB5_S1_A400_E030'])

    def test_finish_current_never_starts_another_old_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = FakeCampaign(tmp, 100); cfg = old_plan.resolve_case('s1', 'QB5_C60LR050')
            camp.state.update(started_unix=123, runs=[dict(run_id=cfg['run_id'], case=cfg['case'],
                replica='R1', status='STOPPED', step=41000)])
            def finish_target(c, config, budget):
                self.assertEqual(config, cfg)
                c.state['runs'][0].update(status='COMPLETED', step=60000)
            with patch('panda_sensor_focus.early_stop.finish_stopped', return_value=False), \
                    patch('panda_sensor_focus.early_stop.run_case', side_effect=finish_target) as execute, \
                    patch('panda_sensor_refine.orchestration.final_evidence', return_value={'step': 60000}):
                result = finish_current.finish(camp, cfg['run_id'], 'fixture')
            self.assertEqual(result['status'], finish_current.PAUSED)
            self.assertEqual(len(result['runs']), 1); execute.assert_called_once()
            self.assertFalse(result['all_registered_completed'])
            self.assertIn(['QB5_C60LR050', 'R2'], result['superseded_unstarted'])

    def test_finish_current_rejects_unknown_and_second_active_run(self):
        cfg = old_plan.resolve_case('s1', 'QB5_C60LR050')
        with self.assertRaises(ValueError):
            finish_current.target_config('s1', cfg['run_id'], {'runs': []})
        with self.assertRaises(ValueError):
            finish_current.target_config('s1', cfg['run_id'], {'runs': [
                {'run_id': cfg['run_id'], 'status': 'STOPPED'}, {'run_id': 'other', 'status': 'RUNNING'}]})

    def test_handoff_never_resumes_wv3_before_verified_focus_closeout(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(handoff, 'control', return_value=Path(tmp)), \
                patch('panda_qb_rre.enqueue.control_dir', return_value=Path(tmp)), \
                patch.object(handoff, 'ensure_process', return_value={'pid': 123}) as launch:
            chain = {'stage': 'RRE_FOCUS'}
            handoff.advance('s1', chain, {})
            self.assertEqual(chain['stage'], 'RRE_FOCUS'); launch.assert_called_once()
            (Path(tmp) / 'status.json').write_text(json.dumps({'stage': 'COMPLETED', 'sheet_verified': False}))
            with self.assertRaises(ValueError):
                handoff.advance('s1', chain, {})


if __name__ == '__main__':
    unittest.main()
