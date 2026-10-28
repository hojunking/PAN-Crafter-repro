"""Executed coefficient, donor/calibration, publication and resume regressions.

Synthetic CPU inputs here are implementation checks, never experiment results.
"""
from copy import deepcopy
import csv
import gc
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from donor30.artifacts import _same
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_sensor5.tests.test_core import sources
from panda_stage2_condq.donor import LoadedDonor

from panda_s1loss9 import plan, reporting
from panda_s1loss9.calibration import bind_donor, binding_for_geometry, validate_stats_bank
from panda_s1loss9.calibration_core import CalibrationBuilder
from panda_s1loss9.contract import LOSS_PAIRS, PAIR_PARAMETERS, validate_loss_pair_config
from panda_s1loss9.engine import build_fresh_stage2
from panda_s1loss9.spec import MatureSpec, geometry_manifest
from panda_s1loss9.stage1 import (build_stage1, teacher_binding, Stage1Stream,
                                 advance_stage1, save_stage1, load_stage1)


PINS = {"code_pin": {"synthetic_cpu_test": "a" * 64},
        "evaluator_pin": {"synthetic_cpu_test": "b" * 64}}


def paired_donor(cfg, registered):
    """Construct real frozen campaign A/T, with explicitly synthetic provenance."""
    trainer = build_stage1(cfg, registered.sources)
    aligner, teacher = trainer.stage.aligner, trainer.stage.teacher
    for model in (aligner, teacher):
        model.requires_grad_(False).eval()
    binding = {
        **registered.inventory.manifest(), "source_split": "RR_train_only",
        "aligner_state_sha256": state_dict_sha256(aligner),
        "teacher_state_sha256": state_dict_sha256(teacher),
        "donor_checkpoint_id": canonical_sha256({"synthetic_case": cfg["case"]}),
        "donor_run_id": cfg["run_id"], "donor_step": 1000,
        "reconstruction_spec": {"bands": 4, "width": 112, "depth": [1, 2, 3],
            "input_layout": "P0", "base_reference": "BICUBIC_MS"},
        "scharr": {"kernel": "absolute_xy_/32", "padding": "reflect1",
            "rgb_indices": [2, 1, 0], "roi_margin": 8},
        "warp": "lanczos4_reflect_FP32_current_pan_px",
        "band30_teacher": teacher_binding(cfg),
        "mature30_teacher": teacher_binding(cfg),
        "band30_reconstruction_spec": teacher.spec.manifest(),
    }
    return LoadedDonor(aligner, teacher, binding, {})


def exact_one_image_bank(donor, spec, registered):
    key = spec.geometry_keys[0]
    ids = list(registered.sample_ids)
    values = torch.linspace(.1, .9, 64 * 64).reshape(1, 1, 64, 64)
    collector = CalibrationBuilder(ids, binding_for_geometry(donor.binding, key),
        k=5, memory_budget_bytes=8 * 1024**2, include_condq=True,
        mask_policy=spec.mask_policy)
    collector.add(ids, values, values.flip(-1), values.flip(-2),
        sensor=spec.sensor, loss_mask=torch.ones_like(values, dtype=torch.bool))
    return {key: collector.finalize()}


class LossContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_full_factorial_fresh_paired_seeds_targets_and_readable_sheet_labels(self):
        self.assertEqual({(a, e) for _, a, e in LOSS_PAIRS},
            {(a, e) for a in (.2, .5, 2.) for e in (.0003, .003, .03)})
        for server, sensor, target in (("s1", "QB", (.9315, 3.5)),
                                        ("s3", "GF2", (.9645, .522))):
            groups = plan.groups(server)
            self.assertEqual(len(groups), 18)
            self.assertIsNone(plan.default_config(server)["budget"]["total_seconds"])
            rows = list(csv.DictReader(io.StringIO(plan.catalog_csv(server))))
            self.assertEqual(len(rows), 18)
            donor_ids, student_ids, receipts, seed_sets = set(), set(), set(), []
            for index, (key, absolute, epsilon) in enumerate(LOSS_PAIRS):
                donor = plan.resolve_case(server, "S1_" + key)
                student = plan.resolve_case(server, sensor + "_" + key)
                self.assertEqual(groups[2 * index]["runs"], [(donor["case"], "R1")])
                self.assertEqual(groups[2 * index + 1]["runs"], [(student["case"], "R1")])
                donor_ids.add(donor["run_id"]); student_ids.add(student["run_id"])
                receipts.add(plan.donor_receipt_path(server, donor["case"]))
                seed_sets.append((donor["stream_seeds"], student["stream_seeds"]))
                self.assertEqual(student["donor_version"], donor["case"])
                self.assertEqual(donor["planned_horizon"], 50000)
                self.assertEqual(student["planned_horizon"], 30000)
                for cfg in (donor, student):
                    validate_loss_pair_config(cfg)
                    reporting._config(cfg)
                    self.assertEqual((cfg["hqnr_min"], cfg["ergas_max"]), target)
                    self.assertEqual(cfg["shift_scale"], 1.)
                    self.assertEqual(cfg["initialization"], "FRESH")
                    self.assertEqual(cfg["inherited_updates"], 0)
                    self.assertEqual(cfg["epsilon_radius_rr"], 3.)
                    description = reporting.approach_description(cfg)
                    self.assertIn(f"lambda_abs={absolute:g}", description)
                    self.assertIn(f"lambda_epsilon={epsilon:g}", description)
                self.assertEqual((student["stage2_spec"]["alpha"],
                    student["stage2_spec"]["beta"], student["stage2_spec"]["gamma"]), (.25, 0., .5))
                self.assertEqual((donor["lr_A"], donor["lr_U"], student["lr_S"]), (1e-5, 1e-4, 1e-4))
                self.assertEqual(donor["baseline_resolved"]["alignment_protocol"]["epsilon"]["radius"], 3.)
                self.assertEqual(rows[2 * index]["donor_case"], donor["case"])
            self.assertEqual((len(donor_ids), len(student_ids), len(receipts)), (9, 9, 9))
            self.assertTrue(all(seeds == seed_sets[0] for seeds in seed_sets))

    def test_all_nine_coefficients_reach_real_aligner_loss_gradients(self):
        # Same train observation, initialization and dedicated epsilon RNG;
        # no loss, warp, model, or epsilon sampler is mocked.
        with tempfile.TemporaryDirectory() as temporary:
            registered = sources(temporary, "QB", count=1)
            rr = registered.batch([0])
            records = {}
            for key, absolute, epsilon in LOSS_PAIRS:
                cfg = plan.resolve_case("s1", "S1_" + key)
                trainer = build_stage1(cfg, registered.sources)
                stage = trainer.stage
                with torch.no_grad():
                    stage.aligner.fc2.bias.copy_(torch.tensor([.375, -.625]))
                calls = []
                hook = stage.teacher.backbone.register_forward_hook(
                    lambda module, args, output: calls.append(1))
                try:
                    out = stage(rr)
                finally:
                    hook.remove()
                self.assertEqual(len(calls), 1)  # No epsilon-shifted recon branch.
                self.assertFalse(stage.pipeline_manifest()["eps"]["shifted_teacher_forward"])
                self.assertEqual(stage.pipeline_manifest()["loss_coefficients"], [1., absolute, epsilon])
                self.assertEqual(stage.teacher_binding(), teacher_binding(cfg))
                parameter = stage.aligner.fc2.bias
                gradients = {name: torch.autograd.grad(out[name], parameter,
                    retain_graph=True)[0].detach().clone()
                    for name in ("loss", "rec", "abs_rr", "eps_rr")}
                # The native prediction is an explicitly detached anchor in L_epsilon.
                anchor = torch.autograd.grad(out["eps_rr"], out["native"]["u"],
                    allow_unused=True, retain_graph=True)[0]
                self.assertIsNone(anchor)
                self.assertGreater(float(gradients["abs_rr"].norm()), 1e-6)
                self.assertGreater(float(gradients["eps_rr"].norm()), .1)
                torch.testing.assert_close(gradients["loss"], gradients["rec"] +
                    absolute * gradients["abs_rr"] + epsilon * gradients["eps_rr"],
                    atol=1e-7, rtol=2e-5)
                records[(absolute, epsilon)] = {"gradients": gradients,
                    "epsilon": out["epsilon_rr"].detach().clone(),
                    "components": torch.tensor([out[k].item() for k in ("rec", "abs_rr", "eps_rr")])}
                del out, stage, trainer, parameter, gradients
                gc.collect()
            baseline = records[(2., .003)]
            for item in records.values():
                torch.testing.assert_close(item["epsilon"], baseline["epsilon"], atol=0, rtol=0)
                torch.testing.assert_close(item["components"], baseline["components"], atol=0, rtol=0)
            for absolute in (.2, .5):
                difference = records[(absolute, .003)]["gradients"]["loss"] - baseline["gradients"]["loss"]
                torch.testing.assert_close(difference, (absolute - 2.) * baseline["gradients"]["abs_rr"],
                    atol=1e-7, rtol=2e-5)
                self.assertGreater(float(difference.norm()), 1e-6)
            for epsilon in (.0003, .03):
                difference = records[(2., epsilon)]["gradients"]["loss"] - baseline["gradients"]["loss"]
                torch.testing.assert_close(difference, (epsilon - .003) * baseline["gradients"]["eps_rr"],
                    atol=1e-7, rtol=2e-5)
                self.assertGreater(float(difference.norm()), 1e-4)

    def test_forged_case_coefficients_and_donor_relabels_rejected(self):
        for server in ("s1", "s3"):
            sensor = plan.dataset_for_server(server)
            for case in ("S1_A020_E0003", sensor + "_A020_E0003"):
                original = plan.resolve_case(server, case)
                changes = ({"donor_version": "S1_A200_E003"},
                    {"stage1_loss_parameters": dict(PAIR_PARAMETERS["A200_E003"])},
                    {"loss_pair": "A200_E003"}, {"lambda_abs": 2.},
                    {"lambda_epsilon": .003}, {"lambda_eps_RR": .03})
                for change in changes:
                    with self.subTest(case=case, change=change):
                        cfg = {**deepcopy(original), **change}
                        with self.assertRaises(ValueError):
                            validate_loss_pair_config(cfg)
                        with self.assertRaises(ValueError):
                            reporting._config(cfg)

    def test_fresh_stage1_publication_cannot_relabel_other_pair_receipt(self):
        from panda_mature30.tests.test_reporting import reused_receipt
        for server in ("s1", "s3"):
            first = plan.resolve_case(server, "S1_A020_E0003")
            second = plan.resolve_case(server, "S1_A200_E003")
            with tempfile.TemporaryDirectory() as temporary:
                receipt, evaluation = reused_receipt(temporary, server=server, source_config=first)
                reporting.emit_evaluation(temporary, first, evaluation, selector_receipt=receipt)
                with self.assertRaisesRegex(ValueError, "reuse or relabel"):
                    reporting.emit_evaluation(temporary, second, evaluation, selector_receipt=receipt)

    def test_publication_joint_pass_and_qualified_or_fallback_representative(self):
        from panda_band30.tests.test_reporting import evaluation as synthetic_evaluation
        from panda_s1loss9 import best_publication

        for server in ("s1", "s3"):
            sensor = plan.dataset_for_server(server)
            cfg = plan.resolve_case(server, sensor + "_A020_E0003")
            h, e = reporting.thresholds(server)
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "runs" / cfg["run_id"] / "attempt_01"
                directory.mkdir(parents=True)
                reporting.atomic_json(directory / "config.json", cfg)
                def emit(step, hqnr, ergas):
                    evaluation = synthetic_evaluation(cfg, step, hqnr, ergas)
                    key = cfg["policy_id"].lower()
                    evaluation["mature30_deployment"] = {
                        "schema": "PANDA_NEXT30_DEPLOYMENT_V1", "policy_id": cfg["policy_id"],
                        "completed_update": step, "geometry_key": key,
                        "geometry": geometry_manifest(key, sensor),
                        "student_sampling": deepcopy(cfg["student_sampling"]),
                        "student_input": deepcopy(cfg["student_input"]),
                        "student_warp": deepcopy(cfg["student_warp"])}
                    reporting.atomic_json(directory / f"evaluation_{step:07d}.json", evaluation)
                    reporting.emit_evaluation(temporary, cfg, evaluation,
                        timestamp="2026-10-26T00:00:00+00:00")
                emit(1000, h - .001, e + .01)
                view = best_publication.project(temporary, server)
                self.assertFalse(view["conflicts"], view["conflicts"])
                self.assertEqual(view["rows"][0][reporting.INDEX["Selector"]], best_publication.FALLBACK_SELECTOR)
                emit(2000, h, e + .005)
                view = best_publication.project(temporary, server)
                self.assertEqual(len(view["rows"]), 1)
                self.assertEqual(view["rows"][0][reporting.INDEX["Selected_step"]], 2000)
                note = json.loads(view["rows"][0][reporting.INDEX["Original description / notes"]])
                self.assertFalse(note["publication"]["publication_joint_pass"])
                emit(3000, h, e)  # Both inclusive boundary comparisons pass.
                emit(4000, h + .001, e - .001)
                view = best_publication.project(temporary, server)
                self.assertFalse(view["conflicts"], view["conflicts"])
                self.assertEqual(len(view["rows"]), 2)
                representative = next(row for row in view["rows"]
                    if row[reporting.INDEX["Selector"]] == best_publication.SELECTOR)
                self.assertEqual(representative[reporting.INDEX["Selected_step"]], 4000)
                for row in view["rows"]:
                    best_publication.verify_owned(row, view)
                    self.assertIn("lambda_abs=0.2", row[reporting.INDEX["Approach"]])
                    self.assertIn("lambda_epsilon=0.0003", row[reporting.INDEX["Approach"]])

    def test_pair_specific_calibration_and_standalone_student_reject_wrong_donor(self):
        with tempfile.TemporaryDirectory() as temporary:
            registered = sources(temporary, "GF2", count=1)
            cfg = plan.resolve_case("s3", "GF2_A020_E0003")
            spec = MatureSpec(**cfg["stage2_spec"])
            first = bind_donor(paired_donor(plan.resolve_case("s3", "S1_A020_E0003"), registered), spec)
            second = bind_donor(paired_donor(plan.resolve_case("s3", "S1_A200_E003"), registered), spec)
            first_bank = exact_one_image_bank(first, spec, registered)
            second_bank = exact_one_image_bank(second, spec, registered)
            validate_stats_bank(first_bank, first.binding, spec, full_hash=True)
            validate_stats_bank(second_bank, second.binding, spec, full_hash=True)
            with self.assertRaises(ValueError):
                validate_stats_bank(first_bank, second.binding, spec)
            geometry = spec.geometry_keys[0]
            hashes = [canonical_sha256({"geometry": geometry_manifest(geometry, spec.sensor),
                                       "binding": d.binding}) for d in (first, second)]
            self.assertNotEqual(*hashes)
            cfg["donor_binding"] = deepcopy(first.binding)
            trainer = build_fresh_stage2(cfg, first, first_bank, evaluator_pin=PINS["evaluator_pin"])
            self.assertEqual(trainer.stage.student.backbone.input.in_channels, 6)
            self.assertEqual(trainer.updates, 0)
            del trainer
            # Both the wrong donor AND its own valid matching CDF are supplied:
            # a builder must still reject disagreement with the registered pair.
            cfg["donor_binding"] = deepcopy(second.binding)
            with self.assertRaises(ValueError):
                build_fresh_stage2(cfg, second, second_bank, evaluator_pin=PINS["evaluator_pin"])

    def test_complete_resume_preserves_low_coefficient_updates_rejects_cross_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            registered = sources(temporary, "QB", count=1)
            cfg = plan.resolve_case("s1", "S1_A020_E0003")
            cfg["baseline_resolved"].update(microbatch_RR=1, effective_batch_RR=1, accumulation_RR=1)
            trainer = build_stage1(cfg, registered.sources)
            stream = Stage1Stream(registered.sources, cfg, device="cpu")
            advance_stage1(trainer, stream)
            path = Path(temporary) / "exact.pt"
            saved = save_stage1(path, trainer, stream, **PINS)
            expected = advance_stage1(trainer, stream)
            restored, resumed_stream, _ = load_stage1(path, saved["file_sha256"], cfg,
                registered.sources, restore_global_rng=False, **PINS)
            actual = advance_stage1(restored, resumed_stream)
            self.assertTrue(_same(expected, actual))
            self.assertTrue(_same(trainer.state_dict(), restored.state_dict()))
            other = plan.resolve_case("s1", "S1_A050_E0003")
            other["baseline_resolved"].update(microbatch_RR=1, effective_batch_RR=1, accumulation_RR=1)
            with self.assertRaises(ValueError):
                load_stage1(path, saved["file_sha256"], other, registered.sources, **PINS)


if __name__ == "__main__":
    unittest.main()
