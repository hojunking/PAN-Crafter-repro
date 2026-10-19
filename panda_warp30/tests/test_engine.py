"""Complete campaign checkpoint and real shuffled RR-H5 stream round trips."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from directshift48.method import ScharrAbsolute
from donor30.artifacts import _same
from panda_stage2_condq.calibration import CalibrationBuilder
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.sources import RegisteredRRTrain
from panda_stage2_condq.stream import TrainStream, StreamSpec
from panda_stage2_condq.training import TrainingSpec
from panda_stage2_condq.tests.test_donor_sources import synthetic_sources
from ta2.evaluation import file_sha

from panda_warp30 import plan
from panda_warp30.calibration import bind_donor
from panda_warp30.engine import build_fresh_stage2, advance, save_checkpoint, load_checkpoint
from panda_warp30.method import frozen_predictions
from panda_warp30.spec import WarpSpec
from panda_warp30.tests.test_method import make_donor


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        synthetic_sources(cls.root, count=5)
        # Distinct samples/nonuniform texture expose broken augmentation replay.
        generator = np.random.default_rng(301019)
        with h5py.File(cls.root/'train.h5', 'r+') as handle:
            for key in ('pan', 'ms', 'lms', 'gt'):
                handle[key][:] = generator.uniform(400, 1500, handle[key].shape).astype(np.float32)
        catalog = json.loads((cls.root/'sources.json').read_text())
        catalog['sensors']['WV3']['splits']['train']['sha256'] = file_sha(cls.root/'train.h5')
        (cls.root/'sources.json').write_text(json.dumps(catalog))
        cls.registered = RegisteredRRTrain.from_catalog(cls.root/'sources.json', root=cls.root, sensor='WV3')
        cls.pins = {'code_pin': {'synthetic_warp30_source.py': 'd'*64},
                    'evaluator_pin': {'protocol': 'SYNTHETIC_OFFLINE_NO_OFFICIAL_METRICS'}}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def fixture(self):
        cfg = plan.resolve_case('s3', 'U23_MID_RETURN')
        cfg.update(planned_horizon=3, evaluation_steps=[1, 2, 3],
                   run_id='PW30_SYNTHETIC_COMPLETE_ENGINE_TEST')
        cfg['training_spec'] = asdict(TrainingSpec(total_updates=3, warmup_updates=0,
            microbatch_size=1, effective_batch_size=2, full_audit_interval=1))
        donor = make_donor()
        donor.binding.update(self.registered.inventory.manifest())
        spec = WarpSpec(**cfg['stage2_spec'])
        donor = bind_donor(donor, spec)
        collector = CalibrationBuilder(self.registered.sample_ids, donor.binding, k=5,
            memory_budget_bytes=8*1024**2, include_condq=True, mask_policy=spec.mask_policy)
        for rr in self.registered.calibration_batches(2):
            f = frozen_predictions(donor.aligner, donor.teacher, rr, spec=spec)
            e, g, q = reference_maps(ScharrAbsolute('WV3'), f['teacher_y'], f['gt_loss'], f['pan_loss'])
            collector.add(rr.lineage.sample_ids, e, g, q, sensor='WV3', loss_mask=f['loss_mask'])
        cfg['donor_binding'] = deepcopy(donor.binding)
        trainer = build_fresh_stage2(cfg, donor, collector.finalize(),
                                    evaluator_pin=self.pins['evaluator_pin'])
        stream = TrainStream(self.registered, StreamSpec(cfg['stream_seeds']['rr_data'],
            cfg['stream_seeds']['rr_augmentation'], **cfg['augmentation']),
            microbatch_size=1, effective_batch_size=2)
        return trainer, stream

    def test_outer_checkpoint_new_engine_and_real_stream_identical_next_update(self):
        trainer, stream = self.fixture()
        advance(trainer, stream)
        self.assertEqual(trainer.updates, 1)
        self.assertEqual(stream.epoch*stream.count+stream.cursor, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'complete.pt'
            saved = save_checkpoint(path, trainer, stream, **self.pins)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(payload['engine_kind'], 'stage2_warp')
            self.assertEqual(payload['pins'], {'code':self.pins['code_pin'],
                                               'evaluator':self.pins['evaluator_pin']})
            self.assertEqual(payload['config'], trainer.config)
            self.assertTrue(payload['resume_only'])
            self.assertEqual(payload['stream'], stream.state_dict())
            other, other_stream = self.fixture()
            returned = load_checkpoint(path, saved['file_sha256'], other, other_stream, **self.pins)
            self.assertEqual(returned, (other, other_stream))
            self.assertTrue(_same(trainer.state_dict(), other.state_dict()))
            self.assertEqual(stream.state_dict(), other_stream.state_dict())
            first = advance(trainer, stream)
            second = advance(other, other_stream)
            self.assertEqual(first['update'], second['update'])
            self.assertTrue(_same(trainer.state_dict(), other.state_dict()))
            self.assertEqual(stream.state_dict(), other_stream.state_dict())
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, trainer, stream, **self.pins)

    def test_wrong_pins_and_engine_rejected_before_stream_or_model_mutation(self):
        trainer, stream = self.fixture()
        advance(trainer, stream)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'complete.pt'
            saved = save_checkpoint(path, trainer, stream, **self.pins)
            other, other_stream = self.fixture()
            before, before_stream = deepcopy(other.state_dict()), deepcopy(other_stream.state_dict())
            for wrong in ({**self.pins,'code_pin':{'synthetic_warp30_source.py':'e'*64}},
                          {**self.pins,'evaluator_pin':{'protocol':'DIFFERENT'}}):
                with self.assertRaisesRegex(ValueError,'contract mismatch'):
                    load_checkpoint(path,saved['file_sha256'],other,other_stream,**wrong)
                self.assertTrue(_same(before,other.state_dict()))
                self.assertEqual(before_stream,other_stream.state_dict())
            other.engine_kind='stage2_midpoint'
            with self.assertRaisesRegex(ValueError,'contract mismatch'):
                load_checkpoint(path,saved['file_sha256'],other,other_stream,**self.pins)
            self.assertTrue(_same(before,other.state_dict()))
            self.assertEqual(before_stream,other_stream.state_dict())
            with self.assertRaisesRegex(ValueError,'evaluator'):
                save_checkpoint(Path(directory)/'wrong_eval.pt',trainer,stream,
                    code_pin=self.pins['code_pin'],evaluator_pin={'protocol':'DIFFERENT'})
            self.assertFalse((Path(directory)/'wrong_eval.pt').exists())


if __name__ == '__main__':
    unittest.main()
