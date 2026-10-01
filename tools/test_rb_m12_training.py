"""M12 training tests use tiny, disposable CPU models; no experiment data."""
import copy
import inspect
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
import torch

from fh12.model import build_model, state_hash
from fh12.training import BatchStream, rng_state, restore_rng
from fh12.losses import student_losses, routed_student_backward
from panda_rb_m12 import training as tr


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.teacher, _ = build_model("P0", 8, [1, 1, 1], 101, role="T")
        self.teacher.eval().requires_grad_(False)
        self.model, _ = build_model("PLH", 8, [1, 1, 1], 201, role="S",
                                    teacher_aligner_state=self.teacher.aligner.state_dict())
        gen = torch.Generator().manual_seed(44)
        self.batch = (torch.rand(2, 8, 32, 32, generator=gen) * 2 - 1,
                      torch.rand(2, 8, 32, 32, generator=gen),
                      torch.rand(2, 8, 8, 8, generator=gen) * 2 - 1,
                      torch.rand(2, 1, 8, 8, generator=gen) * 2 - 1,
                      torch.rand(2, 1, 32, 32, generator=gen) * 2 - 1,
                      torch.tensor([[1, 0], [3, 2]]))
        self.cfg = dict(learning_rate=1e-4, betas=[.9, .999], eps=1e-8, weight_decay=.01)

    def test_qfull_all_numerics_identical(self):
        q = torch.tensor([.1, .3]); q_ref = .2
        report = tr.one_batch_parity(self.model, self.teacher, self.batch, .1,
                                    q, q_ref, q_ref / (q_ref + q), self.cfg)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(value == 0 for value in report["differences"].values()))

    def test_parity_detects_changed_geometry(self):
        q = torch.tensor([.1, .3])
        with self.assertRaises(AssertionError):
            tr.one_batch_parity(self.model, self.teacher, self.batch, .1,
                                q, .2, torch.ones(2), self.cfg)

    def test_only_geometry_changes(self):
        gt, _, ms, lp, pan, _ = self.batch
        out = self.model(pan, ms, lp)
        tout = self.teacher(pan, ms, lp)
        first = tr.geometry_student_losses(out, tout, gt, .1, torch.tensor([.2, .8]))
        second = tr.geometry_student_losses(out, tout, gt, .1, torch.tensor([.5, .5]))
        for key in ("hard", "soft", "edge", "difficulty", "advantage"):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertFalse(torch.equal(first["L_A"], second["L_A"]))
        self.assertTrue(torch.allclose(first["L_U"], first["hard"] + first["soft"] + first["edge_weighted"]))

    def test_routing_actual_A_is_weighted_hard_only(self):
        # Fresh zero residual heads make dY/dA=0 and would let a broken routing
        # test pass vacuously. Activate this disposable model's output head.
        with torch.no_grad():
            self.model.backbone.output[-1].weight.normal_(0.,.01,generator=torch.Generator().manual_seed(91))
        gt, _, ms, lp, pan, _ = self.batch
        losses = tr.geometry_student_losses(self.model(pan, ms, lp), self.teacher(pan, ms, lp), gt, .1, torch.tensor([.2, .8]))
        expected = torch.autograd.grad(losses["L_A"], self.model.aligner.parameters(), retain_graph=True)
        wrong = torch.autograd.grad(losses['L_U'],self.model.aligner.parameters(),retain_graph=True)
        self.assertGreater(tr._norm(expected),0.)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(expected,wrong)))
        routed_student_backward(self.model, losses)
        for parameter, grad in zip(self.model.aligner.parameters(), expected):
            self.assertTrue(torch.equal(parameter.grad, grad))
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))

    def test_probe_preserves_mixed_child_modes_for_all_new_fitting_cases(self):
        self.model.backbone.eval()
        before = [module.training for module in self.model.modules()]
        for case in ('QEDGE','QALIGN','H0','HSPMEAN','NOADV','ADVMEAN'):
            result = tr.diagnostic_probe(self.model,self.teacher,self.batch,.1,
                      torch.tensor([.2,.8]),align_weights=torch.tensor([.5,.5]),case_id=case)
            self.assertEqual([module.training for module in self.model.modules()],before)
            self.assertEqual(result['case_id'],case)
            self.assertEqual(result['wA_values'],[.5,.5])
            self.assertIn('hard_mass_applied',result)

    def test_actual_sample_view_resume_smoke(self):
        outer = self
        class Dataset:
            def __len__(self): return 12
            def __getitem__(self,index):
                idx,view = index
                values = [field[idx % 2].clone() for field in outer.batch[:-1]]
                values[0] = values[0] + .01*(idx+view)
                return (*values,torch.tensor([idx,view,1,1]))
        cfg = dict(self.cfg,batch_size=2,num_warmup=100,num_iter=50000)
        arrays = {'QFULL':np.full((12,4),.5,np.float32)}
        result = tr.resume_smoke(self.model,self.teacher,Dataset(),.1,arrays,201,cfg,'cpu')
        self.assertEqual(result['status'],'PASS')
        self.assertTrue(result['exact_tensor_equality'])
        self.assertEqual(result['updates'],4)

    def test_diagnostics_preserve_parameters_buffers_rng_and_grads(self):
        self.model.register_buffer("test_buffer", torch.tensor([123.]))
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        state = state_hash(self.model.state_dict())
        saved = rng_state(); gradients = [p.grad for p in self.model.parameters()]
        expected_random = (random.random(), np.random.random(), torch.rand(3))
        restore_rng(saved)
        result = tr.diagnostic_probe(self.model, self.teacher, self.batch, .1,
                                     torch.tensor([.2, .8]), torch.tensor([1., 2.]))
        self.assertEqual(state_hash(self.model.state_dict()), state)
        self.assertTrue(self.model.training)
        self.assertFalse(self.teacher.training)
        self.assertTrue(all(p.grad is grad for p, grad in zip(self.model.parameters(), gradients)))
        actual_random = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected_random[:2], actual_random[:2])
        self.assertTrue(torch.equal(expected_random[2], actual_random[2]))
        self.assertIn("K_beta_included", result)
        self.assertIn("L_A_A", result["gradient_norms"])

    def test_validation_probe_never_invents_train_weight(self):
        result = tr.diagnostic_probe(self.model, self.teacher, self.batch, .1)
        self.assertFalse(result["geometry_applied"])
        self.assertIsNone(result["L_U"])
        self.assertNotIn("gradient_norms", result)

    def test_stream_digest_matches_actual_and_preserves_rng(self):
        before = torch.get_rng_state().clone()
        planned = tr.stream_digest(19, 4, 12, updates=23)
        stream = BatchStream(19, 4, 12); digest = bytes(32).hex()
        for _ in range(23):
            if stream.cursor == stream.count:
                stream.new_epoch()
            meta = torch.cat((torch.tensor(stream.remaining_batches()[0]), torch.ones(4, 2, dtype=torch.int64)), dim=1)
            digest = tr.update_stream_digest(digest, meta); stream.cursor += 1
        self.assertEqual(planned, digest)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertNotEqual(planned, tr.stream_digest(19, 4, 13, 23))

    def test_actual_four_column_metadata_hash_and_augmentation_guard(self):
        pair = torch.tensor([[1, 0], [7, 3]])
        actual = torch.cat((pair, torch.ones(2, 2, dtype=torch.int64)), dim=1)
        import hashlib
        expected = hashlib.sha256(bytes(32) + pair.numpy().astype('<i8').tobytes()).hexdigest()
        self.assertEqual(tr.update_stream_digest(bytes(32).hex(), actual), expected)
        actual[0, 3] = 0
        with self.assertRaises(ValueError):
            tr.update_stream_digest(bytes(32).hex(), actual)
        with self.assertRaises(ValueError):
            tr.update_stream_digest(bytes(32).hex(), pair)

    def test_validation_selection_ties_choose_earlier(self):
        selected = tr.choose_validation([dict(update=3030, val_ergas=2.), dict(update=1010, val_ergas=2.), dict(update=2020, val_ergas=3.)])
        self.assertEqual(selected["update"], 1010)
        with self.assertRaises(ValueError):
            tr.choose_validation([dict(update=1, val_ergas=float("nan"))])

    def test_grid_no_test_sweep_and_exact_primary(self):
        self.assertEqual(len(tr.VAL_STEPS), 50)
        self.assertEqual(tr.VAL_STEPS[-1], 50000)
        source = inspect.getsource(tr.train_run)
        self.assertNotIn("evaluate_model(", source)
        self.assertNotIn("FRMetrics(", source)
        self.assertIn('for split in ("train", "val")', source)
        self.assertEqual(tr.DIAGNOSTIC_STEPS, (0, 1000, 10000, 25000, 50000))

    def test_resume_bundle_byte_and_identity_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = dict(update=3, full_state=True, config_sha256="cfg", token="same")
            tr.save_resume(tmp, snapshot)
            self.assertEqual(tr.read_resume(tmp, {"token": "same"})["update"], 3)
            with self.assertRaises(ValueError):
                tr.read_resume(tmp, {"token": "changed"})
            old = (Path(tmp) / "last").resolve()
            tr.save_resume(tmp, dict(snapshot, update=4))
            self.assertTrue(old.exists())
            (Path(tmp) / "last/training_state.pt").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                tr.read_resume(tmp, {})

    def test_weights_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "exact"
            ident = tr.save_weights(dest, self.model, dict(update=50000, config_sha256="x"))
            self.assertEqual(tr.save_weights(dest, self.model, dict(update=50000, config_sha256="x")), ident)
            with torch.no_grad():
                next(self.model.parameters()).add_(1)
            with self.assertRaises(ValueError):
                tr.save_weights(dest, self.model, dict(update=50000, config_sha256="x"))

    def test_exact_optimizer_scheduler_sampler_resume(self):
        models = [copy.deepcopy(self.model), copy.deepcopy(self.model)]
        opts = [tr.make_optimizer(model, self.cfg) for model in models]
        scheds = [tr.make_scheduler(opt, "BASE", 2, 6) for opt in opts]
        streams = [BatchStream(19, 2, 42), BatchStream(19, 2, 42)]
        gt, _, ms, lp, pan, _ = self.batch
        def step(i):
            opts[i].zero_grad(set_to_none=True)
            losses = student_losses(models[i](pan, ms, lp), self.teacher(pan, ms, lp), gt, .1, torch.tensor([.4, .7]))
            routed_student_backward(models[i], losses); opts[i].step(); scheds[i].step()
            streams[i].cursor += 1
        for _ in range(3):
            step(0); step(1)
        with tempfile.TemporaryDirectory() as tmp:
            snap = dict(update=3, full_state=True, config_sha256="cfg", model=models[1].state_dict(),
                        optimizer=opts[1].state_dict(), scheduler=scheds[1].state_dict(), sampler=streams[1].state_dict(), rng=rng_state())
            tr.save_resume(tmp, snap)
            saved = tr.read_resume(tmp, {"config_sha256": "cfg"})
            models[1] = copy.deepcopy(self.model)
            opts[1] = tr.make_optimizer(models[1], self.cfg)
            scheds[1] = tr.make_scheduler(opts[1], "BASE", 2, 6)
            models[1].load_state_dict(saved["model"]); opts[1].load_state_dict(saved["optimizer"])
            scheds[1].load_state_dict(saved["scheduler"]); streams[1].load_state_dict(saved["sampler"])
            restore_rng(saved["rng"])
            for _ in range(3):
                step(0); step(1)
            self.assertEqual(state_hash(models[0].state_dict()), state_hash(models[1].state_dict()))
            self.assertEqual(scheds[0].state_dict(), scheds[1].state_dict())
            self.assertTrue(torch.equal(streams[0].order, streams[1].order))
            self.assertEqual(streams[0].cursor, streams[1].cursor)

    def test_actual_loop_interrupt_validation_then_exact_resume(self):
        """Exercise the production loop, filesystem commits, selector and recovery."""
        class TinyDataset:
            def __len__(self):
                return 12
            def __getitem__(self, item):
                index, view = item if isinstance(item, tuple) else (item, 0)
                gen = torch.Generator().manual_seed(index * 4 + view + 100)
                return (torch.rand(8, 32, 32, generator=gen) * 2 - 1,
                        torch.rand(8, 32, 32, generator=gen),
                        torch.rand(8, 8, 8, generator=gen) * 2 - 1,
                        torch.rand(1, 8, 8, generator=gen) * 2 - 1,
                        torch.rand(1, 32, 32, generator=gen) * 2 - 1,
                        torch.tensor([index, view, 1, 1]))
        dataset = TinyDataset()
        cfg = dict(self.cfg, seed=201, batch_size=2, num_iter=3, num_warmup=1,
                   num_worker=0, log_iter=100, mixed_precision="no", optimizer="AdamW")
        case = dict(seed=201, case_id="QFULL", server="s1", repeat=1, run_id="tiny")
        q = np.full((12, 4), .2, np.float32)
        binding = dict(tau_R=.1, q_ref=.2)
        arrays = dict(QFULL=np.full((12, 4), .5, np.float32), q=q, e_bar=q,
                      calibration_indices=np.arange(8), e_stratum=np.zeros((12, 4), np.int8))
        actual_builder = build_model
        def tiny_builder(layout, width, depth, seed, **kwargs):
            return actual_builder(layout, 8, [1, 1, 1], seed, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            destination = [Path(tmp) / "clean/QFULL"]
            stack.enter_context(patch("panda_rb_m12.plan.build_config", return_value=cfg))
            stack.enter_context(patch("panda_rb_m12.plan.case_for", return_value=case))
            stack.enter_context(patch("panda_rb_m12.binding.load_binding", side_effect=lambda *_args, **_kwargs: (copy.deepcopy(self.teacher), {}, binding, {"dataset": "tiny"}, q)))
            stack.enter_context(patch("panda_rb.weights.load_weights", return_value=(arrays, {"immutable": "map"})))
            stack.enter_context(patch.object(tr, "require_training_device", return_value=torch.device("cpu")))
            stack.enter_context(patch.object(tr, "_validate_recipe"))
            stack.enter_context(patch.object(tr, "build_model", side_effect=tiny_builder))
            stack.enter_context(patch.object(tr, "build_dataset", return_value=dataset))
            stack.enter_context(patch.object(tr, "run_dir", side_effect=lambda *_args, **_kwargs: destination[0]))
            stack.enter_context(patch.object(tr, "source_identity", return_value={"test": "source"}))
            stack.enter_context(patch.object(tr, "VAL_STEPS", (1, 2, 3)))
            stack.enter_context(patch.object(tr, "DIAGNOSTIC_STEPS", (0, 2, 3)))
            self.assertEqual(tr.train_run("tiny", "binding", "weights", device="cpu", root=tmp), 0)
            clean = tr.read_resume(destination[0] / "checkpoints", {})
            selection = json.loads((destination[0] / "checkpoints/selection_manifest.json").read_text())
            self.assertEqual(selection["actual_updates"], 3)
            self.assertEqual(len(selection["validation_records"]), 3)
            self.assertEqual(clean["consumed_weights"]["count"], 6)
            self.assertEqual(int(clean["consumed_sample_view_counts"].sum()), 6)
            self.assertEqual(len(list((destination[0] / "checkpoints/val_candidates").iterdir())), 3)
            destination[0] = Path(tmp) / "interrupted/QFULL"
            original_validation = tr.validation_ergas
            calls = [0]
            def interrupted_validation(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 2:
                    raise OSError("simulated interrupt before validation commit")
                return original_validation(*args, **kwargs)
            with patch.object(tr, "validation_ergas", side_effect=interrupted_validation):
                with self.assertRaises(OSError):
                    tr.train_run("tiny", "binding", "weights", device="cpu", root=tmp)
            self.assertEqual(tr.train_run("tiny", "binding", "weights", device="cpu", root=tmp, resume=True), 0)
            resumed = tr.read_resume(destination[0] / "checkpoints", {})
            self.assertEqual(state_hash(clean["model_state"]), state_hash(resumed["model_state"]))
            self.assertEqual(clean["validation_records"], resumed["validation_records"])
            self.assertEqual(clean["consumed_stream_sha256"], resumed["consumed_stream_sha256"])
            self.assertTrue(np.array_equal(clean["consumed_sample_view_counts"], resumed["consumed_sample_view_counts"]))
            attempts = [json.loads(p.read_text()) for p in (destination[0] / "meta/attempts").glob("*.json")]
            self.assertEqual({item["status"] for item in attempts}, {"TECHNICAL_FAILURE", "TRAINING_COMPLETE"})
            destination[0] = Path(tmp) / "initial_diag_failure/QFULL"
            with patch.object(tr, "diagnostic_probe", side_effect=OSError("initial diagnostic interrupted")):
                with self.assertRaises(OSError):
                    tr.train_run("tiny", "binding", "weights", device="cpu", root=tmp)
            initial = tr.read_resume(destination[0] / "checkpoints", {})
            self.assertEqual(initial["update"], 0)
            self.assertEqual(tr.train_run("tiny", "binding", "weights", device="cpu", root=tmp, resume=True), 0)
            restarted = tr.read_resume(destination[0] / "checkpoints", {})
            self.assertEqual(state_hash(clean["model_state"]), state_hash(restarted["model_state"]))
            destination[0] = Path(tmp) / 'final_evidence_failure/QFULL'
            original_atomic = tr.atomic_json
            def fail_final(path,value):
                if Path(path).name == 'consumed_weights.json':
                    raise OSError('final evidence interruption')
                return original_atomic(path,value)
            with patch.object(tr,'atomic_json',side_effect=fail_final):
                with self.assertRaises(OSError):
                    tr.train_run('tiny','binding','weights',device='cpu',root=tmp)
            self.assertFalse((destination[0]/'checkpoints/selection_manifest.json').exists())
            self.assertEqual(tr.read_resume(destination[0]/'checkpoints',{})['update'],3)
            self.assertEqual(tr.train_run('tiny','binding','weights',device='cpu',root=tmp,resume=True),0)
            published = json.loads((destination[0]/'checkpoints/selection_manifest.json').read_text())
            self.assertEqual(published['consumed_stream_sha256'],clean['consumed_stream_sha256'])
            initial = json.loads((destination[0]/'init_manifest.json').read_text())
            self.assertEqual(published['initialization_sha256'],tr.object_sha(initial))
            # Production-loop selection of two different maps, not just the
            # isolated loss adapter. The original FULL population remains .5.
            case['case_id'] = 'QEDGE'
            arrays['QMEAN'] = np.full((12,4),.8,np.float32)
            destination[0] = Path(tmp) / 'routed/QEDGE'
            self.assertEqual(tr.train_run('tiny','binding','weights',device='cpu',root=tmp),0)
            routed = tr.read_resume(destination[0] / 'checkpoints',{})
            self.assertEqual(routed['consumed_weights']['count'],6)
            self.assertAlmostEqual(routed['consumed_weights']['sum']/6,.5)
            self.assertAlmostEqual(routed['consumed_aligner_weights']['sum']/6,.8,places=6)
            diagnostic = json.loads((destination[0] / 'diagnostics/step_3.json').read_text())
            self.assertEqual(diagnostic['train']['case_id'],'QEDGE')
            self.assertTrue(diagnostic['geometry_mapping']['aligner_is_population_mean'])


if __name__ == "__main__":
    unittest.main()
