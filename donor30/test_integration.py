"""CPU integration across real builders/streams/checkpoints/calibration/eval.

Tiny U widths and images keep end-to-end tests cheap; a separate test builds
the production W112/d123 -> W104/d122 graphs. Metrics are explicitly synthetic
and must never qualify for production sheet publication.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from directshift48.test_evaluation import SyntheticSources, metrics
from directshift48.test_method import batch
from directshift48.reporting import atomic_json
from edgeres48.stream import RRStream, set_learning_rates
from panda_method.edge_data import augment_edge_batch
from ta2.evaluation import file_sha
from . import artifacts, plan, publication, reporting
from .evaluation import model_state_hash, official_evaluate
from .method import calibrate_training
from .runtime import Campaign, build_trainer, load_local_payload, same_predictions


torch.set_num_threads(1)


class IntegrationSources(SyntheticSources):
    def __init__(self, root, *, sensor="GF2", count=4):
        self.sensor, self.bands, self.peak = sensor, 8 if sensor == "WV3" else 4, 2047. if sensor == "WV3" else 1023.
        self.path, self.count = Path(root) / "synthetic_train.h5", count
        with h5py.File(self.path, "w") as h:
            h.create_dataset("pan", data=np.zeros((count, 1, 32, 32), dtype=np.float32))

    def _verify(self, split):
        if split != "train":
            raise ValueError("Synthetic H5 metadata only exists for train")
        return self.path

    def batch(self, split, rows, *, device="cpu", augmentation=(False, False, 0)):
        if torch.device(device).type != "cpu":
            raise AssertionError("Integration tests may not use GPU")
        resolution = "FR" if split == "fr" else "RR"
        role = {"train": "train", "val": "validation", "rr": "test", "fr": "test"}[split]
        # A registered sample must not change when calibration batch size does.
        # All fixture samples intentionally share pixel values, but have unique IDs.
        one = batch(side=32, sensor=self.sensor, resolution=resolution, split=role, batch_size=1)
        result = replace(one, pan=one.pan.repeat(len(rows), 1, 1, 1),
            ms=one.ms.repeat(len(rows), 1, 1, 1), native_lms=one.native_lms.repeat(len(rows), 1, 1, 1),
            gt=None if one.gt is None else one.gt.repeat(len(rows), 1, 1, 1))
        result = replace(result, lineage=replace(result.lineage,
            sample_ids=tuple(f"{self.sensor}:{split}:{i}" for i in rows),
            augmentations=((False, False, 0),) * len(rows)))
        return augment_edge_batch(result, hflip=augmentation[0], vflip=augmentation[1], rot90=augmentation[2])


def tiny(cfg):
    cfg = deepcopy(cfg)
    b = cfg["baseline_resolved"]
    b["teacher_architecture"].update(width=8, depth=[1, 1, 1])
    b.update(effective_batch_RR=2, microbatch_RR=1, accumulation_RR=2)
    if cfg["phase"] == "stage2":
        cfg["student_architecture"].update(width=8, depth=[1, 1, 1])
    return cfg


def evaluate(trainer, sources):
    trainer.evaluator_pin = {}
    return official_evaluate(trainer, sources, metric_functions=metrics(), evaluator_revision="TEST")


def one_update(trainer, stream):
    rates = set_learning_rates(trainer, trainer.config)
    result = trainer.step_accumulated(stream.next_update())
    assert result["update"] == trainer.updates
    assert trainer.optimizer.param_groups[-1]["lr"] == rates["T"]
    return result


class RuntimeIntegrationTests(unittest.TestCase):
    def test_real_runtime_donor_save_calibrate_all_student_modes_reload_and_reporting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = IntegrationSources(root)
            campaign = plan.default_config("s3")
            cfg = tiny(plan.resolve_donor_case(campaign))
            donor = build_trainer(cfg, sources)
            stream = RRStream(sources, cfg, device="cpu")
            one_update(donor, stream)
            evaluation = evaluate(donor, sources)
            run = root / "runs" / cfg["run_id"] / "attempt_01"
            run.mkdir(parents=True)
            atomic_json(run / "config.json", cfg)
            atomic_json(run / "evaluation_0000001.json", evaluation)
            saved = artifacts.save_candidate(donor, stream, evaluation, run, cfg,
                reload_factory=lambda c: build_trainer(c, sources),
                verify_evaluate=lambda t: evaluate(t, sources))
            self.assertTrue(saved["artifact"]["predictions_reverified"])
            chosen = plan.select_donor([saved["record"]], campaign["donor"]["hqnr_min"])
            self.assertTrue(chosen["donor_hqnr_qualified"])
            payload = load_local_payload(chosen["checkpoint_path"], chosen["file_sha256"])
            restored = build_trainer(payload["config"], sources)
            restored.load_state_dict(payload["trainer"])
            same_predictions(evaluation, evaluate(restored, sources))
            controller = Campaign(campaign, output=root / "controller", device="cpu")
            controller.sources = sources
            controller.state["observations"] = [{**saved["record"],
                "evaluation_path": str(run / "evaluation_0000001.json")}]
            # Only metric computation is injected; selected-donor file loading,
            # re-evaluation matching, full-H5 train inventory and calibration are real.
            with patch.object(controller, "evaluate", side_effect=lambda t: evaluate(t, sources)):
                chosen = controller.prepare_selected_donor()
            self.assertTrue(chosen["reloaded_full40_verified"])
            path = Path(chosen["calibration_path"])
            self.assertEqual(json.loads(path.read_text())["n_samples"], sources.count)
            record = reporting.emit_evaluation(root, cfg, evaluation, saved["artifact"])
            reporting.validate_record(record)
            self.assertEqual(len(record["row"]), 64)
            initial_states = []
            for mode in ("gt", "e", "eq", "uniform"):
                student_cfg = tiny(plan.resolve_stage2_case(campaign, mode, donor=chosen, optional=mode == "uniform"))
                student_cfg.update(donor_checkpoint_sha256=chosen["file_sha256"],
                    calibration_path=str(path), calibration_sha256=file_sha(path))
                trainer = build_trainer(student_cfg, sources)
                initial_states.append(model_state_hash(trainer.stage))
                student_stream = RRStream(sources, student_cfg, device="cpu")
                result = one_update(trainer, student_stream)
                self.assertEqual(result["gradient_norm"]["teacher"], 0.)
                ev = evaluate(trainer, sources)
                directory = root / "runs" / student_cfg["run_id"] / "attempt_01"
                directory.mkdir(parents=True)
                atomic_json(directory / "config.json", student_cfg)
                atomic_json(directory / "evaluation_0000001.json", ev)
                stored = artifacts.save_candidate(trainer, student_stream, ev, directory, student_cfg,
                    reload_factory=lambda c: build_trainer(c, sources),
                    verify_evaluate=lambda t: evaluate(t, sources))
                self.assertTrue(stored["record"]["predictions_reverified"])
                self.assertEqual(stored["record"]["donor_state_hash"], chosen["state_hash"])
                row = reporting.emit_evaluation(root, student_cfg, ev, stored["artifact"])
                reporting.validate_record(row)
                self.assertEqual(row["row"][reporting.INDEX["A_peak_lr"]], 0.)
                self.assertEqual(row["row"][reporting.INDEX["Role"]], "Student+FrozenAligner")
                # Exact optimizer continuation, not just equal inference.
                loaded = load_local_payload(stored["artifact"]["checkpoint_path"], stored["artifact"]["file_sha256"])
                fresh = build_trainer(student_cfg, sources)
                fresh.load_state_dict(loaded["trainer"])
                batches = student_stream.next_update()
                set_learning_rates(trainer, student_cfg)
                set_learning_rates(fresh, student_cfg)
                trainer.step_accumulated(batches)
                fresh.step_accumulated(batches)
                self.assertEqual(model_state_hash(trainer.stage), model_state_hash(fresh.stage))
            self.assertEqual(len(set(initial_states)), 1, "Paired modes must start from identical A/T/S states")
            projected = publication.project_campaign(root, "s3")
            self.assertEqual(projected["rows"], [], "Synthetic injected metrics must not reach production Sheets")
            self.assertEqual(len(projected["conflicts"]), 5)

    def test_production_architecture_build_and_student_training_does_not_hash_frozen_weights_each_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            sources = IntegrationSources(temporary, sensor="WV3")
            campaign = plan.default_config("s1")
            cfg = plan.resolve_donor_case(campaign)
            donor = build_trainer(cfg, sources)
            self.assertEqual((donor.stage.teacher.spec.width, donor.stage.teacher.spec.depth), (112, (1, 2, 3)))
            rr = sources.batch("train", [0])
            result = donor.stage(rr)
            result["loss"].backward()
            donor.optimizer.zero_grad(set_to_none=True)
            scales = calibrate_training(donor.stage, [rr], expected_sample_ids=rr.lineage.sample_ids)
            descriptor = {"run_id": cfg["run_id"], "completed_update": 1, "HQNR": .96, "SCC": .98,
                "ERGAS": 2.3, "checkpoint_path": "synthetic-disposable", "state_hash": model_state_hash(donor.stage),
                "checkpoint_verified": True, "metrics_complete": True, "metrics_finite": True,
                "same_state_verified": True, "server": "s1", "dataset": "WV3", "phase": "donor"}
            chosen = plan.select_donor([descriptor], .959)
            student_cfg = plan.resolve_stage2_case(campaign, "eq", donor=chosen)
            student = build_trainer(student_cfg, sources, donor_stage=donor.stage, calibration=scales)
            self.assertEqual((student.stage.student.spec.width, student.stage.student.spec.depth), (104, (1, 2, 2)))
            with patch("donor30.method.state_dict_sha256", side_effect=AssertionError("Full SHA in hotpath")):
                out = student.stage(rr)
                out["loss"].backward()
            self.assertTrue(all(p.grad is None for p in student.stage.aligner.parameters()))
            self.assertTrue(all(p.grad is None for p in student.stage.teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
