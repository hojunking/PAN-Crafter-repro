"""Actual W112/D123 midpoint updates and exact complete-state restart on CPU."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import tempfile
import unittest

import torch

from directshift48.method import ScharrAbsolute
from donor30.artifacts import _same
from panda_method.calibration import state_dict_sha256
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.tests.test_method import fixture as original_fixture, batch
from panda_stage2_condq.training import TrainingSpec
from panda30r.tests.test_engine import MemoryStream
from panda_midpoint.calibration import bind_donor
from panda_midpoint.method import frozen_midpoint_predictions
from panda_midpoint import plan
from panda_midpoint.engine import build_fresh_stage2, advance, save_checkpoint, load_checkpoint
from panda_midpoint.runtime import discover_mature_donor, Campaign, improvement
from ta2.evaluation import file_sha


def fixture():
    cfg = plan.resolve_case('s3', 'MID_B005')
    original = original_fixture()
    donor = bind_donor(LoadedDonor(original.aligner, original.teacher,
                                  original.donor_binding, {'test_fixture': True}))
    rr = batch()
    out = frozen_midpoint_predictions(donor.aligner, donor.teacher, rr, sensor='WV3')
    e, g, q = reference_maps(ScharrAbsolute('WV3'), out['teacher_y'],
                             out['gt_midpoint'], out['pan_aligned'])
    builder = CalibrationBuilder(['WV3:train:0'], donor.binding, k=5,
        memory_budget_bytes=4*1024**2, include_condq=True, mask_policy=plan.LOSS_MASK_POLICY)
    builder.add(['WV3:train:0'], e, g, q, sensor='WV3', loss_mask=out['loss_mask'])
    stats = builder.finalize()
    cfg.update(donor_binding=donor.binding, planned_horizon=3, evaluation_steps=[1, 2, 3])
    cfg['training_spec'] = asdict(TrainingSpec(total_updates=3, warmup_updates=0,
        microbatch_size=1, effective_batch_size=1, full_audit_interval=1))
    trainer = build_fresh_stage2(cfg, donor, stats, evaluator_pin={'test': 'explicit'})
    return trainer


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_only_student_updates_and_unchanged_architecture(self):
        trainer = fixture()
        stage = trainer.stage
        before = [state_dict_sha256(m) for m in (stage.aligner, stage.teacher)]
        student = state_dict_sha256(stage.student)
        result = advance(trainer, MemoryStream(0))
        self.assertEqual(result['update'], 1)
        self.assertEqual(before, [state_dict_sha256(m) for m in (stage.aligner, stage.teacher)])
        self.assertNotEqual(student, state_dict_sha256(stage.student))
        self.assertTrue(all(p.grad is None for m in (stage.aligner, stage.teacher) for p in m.parameters()))
        self.assertEqual(stage.student.spec.width, 112)
        self.assertEqual(stage.student.spec.depth, (1, 2, 3))
        trainer.assert_contract()

    def test_complete_checkpoint_identical_next_update_and_legacy_rejected(self):
        trainer, stream = fixture(), MemoryStream(0)
        advance(trainer, stream)
        pins = {'code_pin': {'midpoint.py': 'd'*64}, 'evaluator_pin': {'test': 'explicit'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'resume.pt'
            saved = save_checkpoint(path, trainer, stream, **pins)
            other, other_stream = fixture(), MemoryStream(0)
            load_checkpoint(path, saved['file_sha256'], other, other_stream, **pins)
            advance(trainer, stream)
            advance(other, other_stream)
            self.assertTrue(_same(trainer.state_dict(), other.state_dict()))
            self.assertEqual(stream.state_dict(), other_stream.state_dict())
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, trainer, stream, **pins)
            other.engine_kind = 'stage2_condq'
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                load_checkpoint(path, saved['file_sha256'], other, other_stream, **pins)

    def test_midpoint_binding_and_engine_are_mandatory(self):
        trainer = fixture()
        donor = LoadedDonor(trainer.stage.aligner, trainer.stage.teacher,
                           trainer.stage.donor_binding, {})
        cfg = deepcopy(trainer.config)
        cfg['donor_binding'].pop('midpoint_geometry')
        with self.assertRaisesRegex(ValueError, 'binding'):
            build_fresh_stage2(cfg, donor, trainer.stage.stats)
        cfg = deepcopy(trainer.config)
        cfg['engine_kind'] = 'stage2_condq'
        with self.assertRaisesRegex(ValueError, 'midpoint'):
            build_fresh_stage2(cfg, donor, trainer.stage.stats)

    def test_missing_mature_donor_stops_without_training_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'no automatic training fallback'):
                discover_mature_donor('s3', root=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
        with self.assertRaises(ValueError):
            discover_mature_donor('s4')

    def test_mature_receipt_local_path_remap_and_checksum_guard(self):
        from panda30r import plan as prior
        from panda30r.tests.test_reporting import evaluation
        cfg = prior.resolve_case('s3', 'MATURE', source_update=7000)
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)/'work_dir/panda30r'/prior.CAMPAIGN_ID/'s3/WV3'
            directory = base/'runs'/cfg['run_id']/'attempt_01'
            models = directory/'models'
            models.mkdir(parents=True)
            checkpoint = models/'step_0043000.pt'
            # Discovery audits bytes/receipts; strict tensor import happens in admission.
            checkpoint.write_bytes(b'synthetic discovery-only checkpoint')
            (directory/'run_state.json').write_text(json.dumps({
                'status': 'COMPLETED', 'step': 43000, 'run_id': cfg['run_id']}))
            ev = evaluation(cfg, step=43000)
            for split in ('rr', 'fr'):
                for scene in ev[split]['per_scene']:
                    scene['raw_prediction_sha256'] = 'e'*64
            (directory/'evaluation_0043000.json').write_text(json.dumps(ev))
            (base/'donors').mkdir()
            replay = base/'donors/M_replay_full40.json'
            replay.write_text(json.dumps(ev))
            row = {'source_config': cfg, 'source_server': 's3', 'mature': True,
                'target_campaign_id': prior.CAMPAIGN_ID, 'version': 'M', 'completed_update': 50000,
                'checkpoint_path': '/different/container/path/step_0043000.pt',
                'checkpoint_sha256': file_sha(checkpoint), 'replay_sha256': file_sha(replay)}
            row['binding_sha256'] = plan.canonical_sha256(row)
            receipt = base/'donors/M.json'
            receipt.write_text(json.dumps(row))
            resolved = discover_mature_donor('s3', root=root)
            self.assertEqual(resolved['checkpoint_path'], str(checkpoint.resolve()))
            self.assertEqual(resolved['official_source_state_sha256'], ev['state_before']['model_state_sha256'])
            checkpoint.write_bytes(b'changed bytes')
            with self.assertRaisesRegex(ValueError, 'SHA changed'):
                discover_mature_donor('s3', root=root)

    def test_three_run_default_no_old_budget_and_selector_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Campaign('s3', output=directory, device='cpu')
            self.assertNotIn('budget', campaign.config)
            self.assertEqual(sum(len(g['runs']) for g in campaign.config['groups']), 3)
            campaign.state['started_unix'] -= 10**7
            campaign.check()  # old deadline/elapsed time cannot silently skip this queue
        selection = {'best_selected': {'HQNR': .960, 'ERGAS': 2.1, 'completed_update': 1000},
                     'best_hqnr': {'HQNR': .960, 'ERGAS': 2.1, 'completed_update': 1000}}
        evaluation = {'fr': {'metrics': {'HQNR': .9595}}, 'rr': {'metrics': {'ERGAS': 2.05, 'SCC': .98}},
                      'state_before': {'completed_update': 2000}}
        _, reasons = improvement(evaluation, selection)
        self.assertIn('HQNR_GATE_ERGAS_SELECTOR', reasons)
        self.assertNotIn('BEST_HQNR_BACKUP', reasons)


if __name__ == '__main__':
    unittest.main()
