"""Protect registered tuning contrasts and real W128 training/resume on CPU.

The donor and RR image are deterministic synthetic fixtures. Production model
widths, warps, Hard calculation, calibration, optimizer and checkpoint I/O are
executed; only the test batch size is reduced to one.
"""
from collections import Counter
from copy import deepcopy
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from directshift48.method import ScharrAbsolute
from donor30.artifacts import _same
from donor30.masking import masked_mean
from panda_band30.tests.test_method import make_donor
from panda_band30.reconstruction import reconstruct_with_base
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.tests.test_method import batch
from panda30r.tests.test_engine import MemoryStream
from panda_tune30 import plan, calibration as calibration
from panda_tune30.contract import CASES, CAMPAIGN_ID, WARP_MANIFEST
from panda_tune30.engine import (
    ScheduleTrainingSpec, build_fresh_stage2, advance, save_checkpoint,
    load_checkpoint,
)
from panda_tune30.evaluation import configured_pipeline
from panda_tune30.method import frozen_predictions
from panda_tune30.spec import MatureSpec
from panda_tune30.warp import warp_pan
from pa.warp import warp_pan as bicubic_warp_pan


EVALUATOR_PIN = {'test': 'explicit_cpu_synthetic'}
CODE_PIN = {'synthetic_test_pin.py': 'd' * 64}


def training_fixture(case, replica='R1'):
    """Use the real registry/builder with a real, small conditional-q bank."""
    cfg = plan.resolve_case('s3', case, replica)
    cfg['training_spec'].update(microbatch_size=1, effective_batch_size=1)
    spec = MatureSpec(**cfg['stage2_spec'])
    donor = calibration.bind_donor(make_donor(shift=(1., -.5)), spec)
    rr = batch()
    key = spec.geometry_keys[0]
    frozen = frozen_predictions(donor.aligner, donor.teacher, rr, geometry_key=key)
    e, g, q = reference_maps(ScharrAbsolute('WV3'), frozen['teacher_y'], rr.gt,
                             frozen['pan_loss'])
    collector = CalibrationBuilder(
        ['WV3:train:0'], calibration.binding_for_geometry(donor.binding, key),
        k=5, memory_budget_bytes=8 * 1024**2, include_condq=True,
        mask_policy=spec.mask_policy,
    )
    collector.add(['WV3:train:0'], e, g, q, sensor='WV3',
                  loss_mask=frozen['loss_mask'])
    cfg['donor_binding'] = deepcopy(donor.binding)
    return build_fresh_stage2(cfg, donor, {key: collector.finalize()},
                              evaluator_pin=EVALUATOR_PIN, device='cpu')


class RegistryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_both_queues_are_fresh_paired_and_match_csv(self):
        from panda_follow30 import plan as previous
        from panda_tune30.reporting import _config
        for server, count, updates in [('s1', 11, 320000), ('s3', 15, 450000)]:
            donor = plan.resolve_case(server, plan.DONORS[0])
            old_donor = previous.resolve_case(server, previous.DONORS[0])
            _config(donor)
            self.assertEqual(donor['phase'], 'donor')
            self.assertFalse(donor['stage1_retrained'])
            self.assertEqual(donor['baseline_resolved']['teacher_architecture'],
                             old_donor['baseline_resolved']['teacher_architecture'])
            self.assertEqual(donor['baseline_resolved']['teacher_architecture']['width'], 112)
            self.assertFalse(set(donor) & {'student_width', 'student_warp', 'student_input',
                                         'sampling_variant', 'tuning_parameters'})
            groups = plan.groups(server)
            configs = [plan.resolve_case(server, case, replica)
                       for group in groups for case, replica in group['runs']]
            self.assertEqual(len(configs), count)
            self.assertEqual(sum(cfg['continuation_updates'] for cfg in configs), updates)
            self.assertEqual(len({cfg['run_id'] for cfg in configs}), count)
            pairs = Counter((cfg['case'], cfg['replica']) for cfg in configs)
            self.assertTrue(all(n == 1 for n in pairs.values()))
            for group in groups:
                self.assertEqual({replica for _, replica in group['runs']},
                                 {'R3'} if len(group['runs']) == 1 else {'R1', 'R2'})
            for cfg in configs:
                self.assertEqual(cfg['campaign_id'], CAMPAIGN_ID)
                self.assertEqual(cfg['initialization'], 'FRESH')
                self.assertEqual(cfg['inherited_updates'], 0)
                self.assertEqual(cfg['shift_scale'], 1.)
                self.assertEqual(cfg['student_warp'], WARP_MANIFEST)
                self.assertEqual(cfg['input_variant'], 'CORRECTED_DIFF')
                self.assertEqual(cfg['sampling_variant'], 'UBIL')
                self.assertEqual(cfg['student_views'], 1)
                self.assertEqual(cfg['stage2_spec']['beta'], 0.)
                self.assertEqual(cfg['rotation_consistency_weight'], 0.)
                self.assertEqual(cfg['pair_transform'], 'NONE')
                self.assertFalse(cfg['inference_tta'])
                self.assertEqual(cfg['model_variant'], 'RAW')
                self.assertFalse(cfg['stage1_retrained'])
                self.assertFalse(set(cfg) & {'swa', 'ema', 'model_soup', 'averaging'})
                self.assertEqual(set(cfg['student_architecture']),
                                 {'bands', 'width', 'depth', 'input_layout', 'base_reference'})
                control = cfg['comparison_control']
                old = previous.resolve_case(server, control['case'], cfg['replica'])
                self.assertEqual(old['seed'], cfg['seed'])
                self.assertEqual(old['stream_seeds'], cfg['stream_seeds'])
                self.assertNotEqual(old['run_id'], cfg['run_id'])
            with (Path(plan.__file__).parent / f'cases_{server}.csv').open(newline='') as handle:
                self.assertEqual(list(csv.DictReader(handle)),
                                 list(csv.DictReader(io.StringIO(plan.catalog_csv(server)))))

    def test_single_factor_contrasts_and_real_schedule_endpoints(self):
        base = CASES['S100DIFFUBILC30']
        changed = {'C25': {'total_updates'}, 'C35': {'total_updates'},
                   'C30LR080': {'lr_S'}, 'C30LR120': {'lr_S'},
                   'C30G025': {'gamma'}, 'C30A0125': {'alpha'},
                   'C30W128': {'student_width'}}
        for suffix, expected in changed.items():
            case = 'S100DIFFUBIL' + suffix
            self.assertEqual({key for key in base if base[key] != CASES[case][key]}, expected)
            cfg = plan.resolve_case('s3', case, 'R1')
            rate = ScheduleTrainingSpec(**cfg['training_spec'])
            self.assertEqual(rate.rate(100), CASES[case]['lr_S'])
            self.assertEqual(rate.rate(CASES[case]['total_updates']), 0.)
        c25 = ScheduleTrainingSpec(**plan.resolve_case('s3', 'S100DIFFUBILC25')['training_spec'])
        c30 = ScheduleTrainingSpec(**plan.resolve_case('s3', 'S100DIFFUBILC30', 'R3')['training_spec'])
        self.assertLess(c25.rate(20000), c30.rate(20000))

    def test_hard_coefficients_reach_actual_loss_and_student_only_update(self):
        for case, alpha, gamma in [('S100DIFFUBILC30G025', .25, .25),
                                   ('S100DIFFUBILC30A0125', .125, .5)]:
            with self.subTest(case=case):
                trainer = training_fixture(case)
                stage, rr = trainer.stage, batch()
                fixed_a = deepcopy(stage.aligner.state_dict())
                fixed_t = deepcopy(stage.teacher.state_dict())
                initial_s = deepcopy(stage.student.state_dict())
                with patch.object(stage.student.backbone, 'forward',
                                  wraps=stage.student.backbone.forward) as forward:
                    output = stage(rr, return_maps=True)
                self.assertEqual(forward.call_count, 1)
                maps = output['maps']
                error = (output['student']['y'] - rr.gt).abs().mean(1, keepdim=True)
                weight = 1 + alpha * maps['d_e'] + gamma * maps['d_e'] * maps['r_q_bar']
                expected = masked_mean(weight * error, output['loss_mask'])
                torch.testing.assert_close(output['loss'], expected, rtol=0, atol=0)
                first = trainer.step_accumulated([rr])
                self.assertAlmostEqual(first['loss'], float(expected.detach()), places=7)
                self.assertEqual(first['loss'], first['hard_weighted'])
                self.assertEqual(first['soft_weighted'], 0.)
                self.assertEqual(first['consistency_lambda'], 0.)
                self.assertEqual(first['student_views'], 1)
                self.assertFalse(_same(initial_s, stage.student.state_dict()))
                self.assertTrue(_same(fixed_a, stage.aligner.state_dict()))
                self.assertTrue(_same(fixed_t, stage.teacher.state_dict()))
                self.assertTrue(all(p.grad is None for m in (stage.aligner, stage.teacher)
                                    for p in m.parameters()))

    def test_w128_training_uses_w112_teacher_and_distinct_pan_warps(self):
        trainer = training_fixture('S100DIFFUBILC30W128')
        stage, rr = trainer.stage, batch()
        self.assertEqual(stage.student.spec.width, 128)
        self.assertEqual(stage.teacher.spec.width, 112)
        self.assertEqual(stage.student.backbone.input.in_channels, 10)
        self.assertEqual(stage.teacher.backbone.input.in_channels, 9)
        self.assertEqual(sum(p.numel() for p in stage.student.parameters()), 3633800)
        frozen = stage.frozen_predictions(rr)
        expected_pan = warp_pan(rr.pan, frozen['u'])
        torch.testing.assert_close(frozen['pan_input'], expected_pan, rtol=0, atol=0)
        teacher_pan = bicubic_warp_pan(rr.pan, frozen['u'])
        self.assertFalse(torch.equal(expected_pan, teacher_pan))
        expected_teacher = reconstruct_with_base(
            stage.teacher, teacher_pan, frozen['ms_base_frame'], rr.ms)['y']
        torch.testing.assert_close(frozen['teacher_y'], expected_teacher, rtol=0, atol=0)
        captured = []
        hook = stage.student.backbone.input.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        try:
            out = stage.native_prediction(rr)
        finally:
            hook.remove()
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].shape[1], 10)
        torch.testing.assert_close(captured[0][:, :1], expected_pan, rtol=0, atol=0)
        torch.testing.assert_close(captured[0][:, 1:2], expected_pan - rr.pan, rtol=0, atol=0)
        torch.testing.assert_close(captured[0][:, 2:], frozen['ms_base_frame'], rtol=0, atol=0)
        torch.testing.assert_close(out['pan_correction'], frozen['u'], rtol=0, atol=0)
        self.assertEqual(out['y'].shape, rr.gt.shape)

    def test_w128_complete_checkpoint_resumes_same_next_weights_and_adam(self):
        trainer = training_fixture('S100DIFFUBILC30W128')
        stream = MemoryStream(0)
        original_a = deepcopy(trainer.stage.aligner.state_dict())
        original_t = deepcopy(trainer.stage.teacher.state_dict())
        advance(trainer, stream)
        self.assertEqual(trainer.updates, 1)
        pins = {'code_pin': CODE_PIN, 'evaluator_pin': EVALUATOR_PIN}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'w128_step1.pt'
            artifact = save_checkpoint(path, trainer, stream, **pins)
            other = training_fixture('S100DIFFUBILC30W128')
            other_stream = MemoryStream(0)
            load_checkpoint(path, artifact['file_sha256'], other, other_stream, **pins)
            self.assertEqual(other.updates, 1)
            self.assertEqual(other_stream.state_dict(), stream.state_dict())
            advance(trainer, stream)
            advance(other, other_stream)
            self.assertEqual(other.updates, 2)
            self.assertTrue(_same(trainer.stage.student.state_dict(), other.stage.student.state_dict()))
            self.assertTrue(_same(trainer.optimizer.state_dict(), other.optimizer.state_dict()))
            self.assertEqual(stream.state_dict(), other_stream.state_dict())
            self.assertEqual(other.optimizer.param_groups[0]['lr'], other.training.rate(2))
            self.assertTrue(all(int(item['step']) == 2 for item in other.optimizer.state.values()))
            self.assertTrue(_same(original_a, other.stage.aligner.state_dict()))
            self.assertTrue(_same(original_t, other.stage.teacher.state_dict()))
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, trainer, stream, **pins)
            other.config['run_id'] = 'WRONG_RUN'
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                load_checkpoint(path, artifact['file_sha256'], other, other_stream, **pins)

    def test_actual_evaluation_checks_all_five_student_architecture_fields(self):
        trainer = training_fixture('S100DIFFUBILC30W128')
        source = SimpleNamespace(sensor='WV3', manifest_sha=lambda split: 'a' * 64)
        actual = configured_pipeline(trainer, source)
        self.assertEqual(actual['student_warp'], WARP_MANIFEST)
        self.assertEqual(actual['student_input']['channels'], 10)
        self.assertEqual(actual['teacher']['pan_channels'], 1)
        original = deepcopy(trainer.config)
        corrupted = {'bands': 4, 'width': 112, 'depth': [1, 2, 4],
                     'input_layout': 'P1', 'base_reference': 'WRONG_BASE'}
        for key, wrong in corrupted.items():
            with self.subTest(field=key):
                trainer.config = deepcopy(original)
                trainer.config['student_architecture'][key] = wrong
                with self.assertRaises(ValueError):
                    configured_pipeline(trainer, source)
        trainer.config = original
        configured_pipeline(trainer, source)


if __name__ == '__main__':
    unittest.main()
