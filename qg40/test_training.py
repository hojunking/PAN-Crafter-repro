"""Small CPU numerical regressions; no dataset, campaign, or GPU activation."""
from __future__ import annotations

import copy
import contextlib
import datetime as dt
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from fh12.model import build_model as old_build_model
from fh12.losses import student_losses as old_student_losses
from fh12.losses import teacher_loss as old_teacher_loss
from qg40.calibration import (CONSTANT_ALIGNER_Q, axis16_q, compute_calibration,
                              select_calibration_indices, calibrate)
from qg40.common import atomic_json, object_sha, sha256
from qg40.losses import routed_student_backward, student_losses, teacher_loss
from qg40.model import build_model, state_hash, sync_frontend
from qg40.plan import CASES, CAMPAIGN_ID, GRID_STEPS, build_config
from qg40.training import (BatchStream, aligner_factor, cosine_factor, make_scheduler,
                           runtime_context, save_resume_bundle, train_run, validate_config, atomic_torch)


class ConstantTeacher(nn.Module):
    bands = 4

    def predict_delta(self, pan, base):
        return torch.zeros(len(pan), 2, device=pan.device)

    def forward(self, pan, ms, lp):
        return {"y": torch.zeros(len(pan), 4, *pan.shape[-2:], device=pan.device)}


class TinyDataset:
    has_gt, base_count = True, 3

    def __len__(self):
        return self.base_count

    def base(self, index):
        gt = torch.full((4, 8, 8), float(index + 1))
        # Unequal pixel errors prove tau uses one pooled full-pixel median.
        gt[:, :7] = float(index) / 10
        return gt, gt, torch.zeros(4, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 8, 8), torch.tensor([index, 0])

    def get_view(self, index, rot, augment):
        return self.base(index)


class QG40TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.gen = torch.Generator().manual_seed(813)

    def batch(self, bands=4):
        return (torch.randn(2, 1, 32, 32, generator=self.gen),
                torch.randn(2, bands, 8, 8, generator=self.gen),
                torch.randn(2, 1, 8, 8, generator=self.gen),
                torch.randn(2, bands, 32, 32, generator=self.gen))

    def activate(self, model):
        with torch.no_grad():
            model.backbone.output[-1].weight.copy_(.03 * torch.randn(
                model.backbone.output[-1].weight.shape, generator=self.gen))
            model.aligner.fc2.weight.copy_(.02 * torch.randn(
                model.aligner.fc2.weight.shape, generator=self.gen))

    def test_c8_exact_initial_state_forward_and_loss(self):
        pan, ms, lp, gt = self.batch(8)
        for layout in ("P0", "PLH"):
            old, _ = old_build_model(layout, 8, [1, 1, 1], 82001)
            new, _ = build_model(layout, 8, [1, 1, 1], 82001, bands=8)
            self.assertEqual(state_hash(old.state_dict()), state_hash(new.state_dict()))
            self.activate(old)
            new.load_state_dict(old.state_dict())
            before, after = old(pan, ms, lp), new(pan, ms, lp)
            for key in before:
                self.assertTrue(torch.equal(before[key], after[key]), (layout, key))
            a = old_student_losses(before, {"y": gt * .5}, gt, .2, [.3, .5])
            b = student_losses(after, {"y": gt * .5}, gt, .2, [.3, .5], bands=8)
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]), key)
            old_con = old_teacher_loss(old, before, gt, pan, ms, 1, torch.Generator().manual_seed(7))
            new_con = teacher_loss(new, after, gt, pan, ms, 1, torch.Generator().manual_seed(7), bands=8)
            for key in ("total", "rec", "off", "epsilon"):
                self.assertTrue(torch.equal(old_con[key], new_con[key]), key)

    def test_c4_stems_output_independent_clone_and_coupling(self):
        teacher, _ = build_model("P0", 8, [1, 1, 1], 81001)
        student, manifest = build_model("PLH", 8, [1, 1, 1], 82001, role="S",
                                        teacher_aligner_state=teacher.aligner.state_dict())
        self.assertEqual(teacher.backbone.input.in_channels, 5)
        self.assertEqual(student.backbone.input.in_channels, 7)
        self.assertEqual(student.aligner.ms_stem[1].in_channels, 4)
        self.assertEqual(student.backbone.output[-1].out_channels, 4)
        self.assertTrue(manifest["extra_kernel_zero"])
        self.assertEqual(state_hash(teacher.aligner.state_dict()), state_hash(student.aligner.state_dict()))
        self.assertFalse(set(map(id, teacher.parameters())) & set(map(id, student.parameters())))
        self.assertTrue(all(p.requires_grad for p in student.aligner.parameters()))
        repeat, _ = build_model("PLH", 8, [1, 1, 1], 82001, role="S",
                                teacher_aligner_state=teacher.aligner.state_dict())
        self.assertEqual(state_hash(student.state_dict()), state_hash(repeat.state_dict()))
        pan, ms, lp, _ = self.batch()
        self.assertEqual(student(pan, ms, lp)["y"].shape, (2, 4, 32, 32))

    def test_frontend_sync_signed_shared_gradient_at_production_sizes(self):
        for size in (64, 256, 512):
            pan = torch.randn(1, 1, size, size, generator=self.gen)
            ms = torch.randn(1, 4, size // 4, size // 4, generator=self.gen)
            lp = torch.randn(1, 1, size // 4, size // 4, generator=self.gen)
            delta = torch.tensor([[1.25, -2.5]], requires_grad=True)
            out = sync_frontend(pan, ms, lp, delta, "PLH")
            self.assertEqual(out["x_in"].shape, (1, 7, size, size))
            self.assertTrue(torch.equal(out["H"], out["pan_aligned"] - out["L"]))
            self.assertTrue(bool((out["H"] < 0).any()))
            grad_l = torch.autograd.grad(out["L"].square().mean(), delta, retain_graph=True)[0]
            grad_h = torch.autograd.grad(out["H"].square().mean(), delta)[0]
            self.assertGreater(float(grad_l.norm()), 0.)
            self.assertGreater(float(grad_h.norm()), 0.)
            self.assertEqual(float(delta[0, 1]), -2.5)

    def test_student_routing_detached_teacher_and_exact_one_axis_profiles(self):
        teacher, _ = build_model("P0", 8, [1, 1, 1], 81001)
        self.activate(teacher)
        teacher.eval().requires_grad_(False)
        student, _ = build_model("PLH", 8, [1, 1, 1], 82001, role="S",
                                 teacher_aligner_state=teacher.aligner.state_dict())
        self.activate(student)
        pan, ms, lp, gt = self.batch()
        out = student(pan, ms, lp)
        with torch.no_grad():
            teacher_out = teacher(pan, ms, lp)
        losses = student_losses(out, teacher_out, gt, .2, [.3, .5])
        params_u, params_a = list(student.backbone.parameters()), list(student.aligner.parameters())
        expected_u = torch.autograd.grad(losses["L_U"], params_u, retain_graph=True)
        expected_a = torch.autograd.grad(losses["L_A"], params_a, retain_graph=True)
        routed_student_backward(student, losses)
        for actual, expected in zip(params_u + params_a, expected_u + expected_a):
            self.assertTrue(torch.equal(actual.grad, expected))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        for profile in ("A24R", "B20", "E10"):
            changed = student_losses(out, teacher_out, gt, .2, [.3, .5], profile)
            self.assertTrue(torch.equal(changed["L_A"], losses["L_A"]))
            self.assertTrue(torch.equal(changed["hard"], losses["hard"]))
            self.assertTrue(torch.equal(changed["soft"], losses["soft"] * (2 if profile == "B20" else 1)))
            self.assertTrue(torch.equal(changed["edge_weighted"], losses["edge_weighted"] * (.5 if profile == "E10" else 1)))

    def test_teacher_odd_only_consistency_rng_c3_and_no_u_synthetic(self):
        model, _ = build_model("P0", 8, [1, 1, 1], 81001)
        self.activate(model)
        pan, ms, lp, gt = self.batch()
        out = model(pan, ms, lp)
        rng = torch.Generator().manual_seed(4)
        initial = rng.get_state().clone()
        even = teacher_loss(model, out, gt, pan, ms, 0, rng)
        self.assertTrue(torch.equal(initial, rng.get_state()))
        self.assertFalse(even["offset_active"])
        odd = teacher_loss(model, out, gt, pan, ms, 1, rng)
        rng.set_state(initial)
        c3 = teacher_loss(model, out, gt, pan, ms, 1, rng, 3e-4)
        self.assertTrue(torch.equal(odd["epsilon"], c3["epsilon"]))
        self.assertTrue(torch.equal(odd["off"], c3["off"]))
        self.assertEqual(c3["offset_weight"], 3e-4)
        only_a = torch.autograd.grad(odd["off"], list(model.backbone.parameters()), allow_unused=True)
        self.assertTrue(all(value is None for value in only_a))
        self.assertTrue(bool((odd["epsilon"].square().sum(1) <= 4).all()))

    def test_calibration_strict_shared_subset_and_constant_q(self):
        indices = select_calibration_indices(3073)
        self.assertEqual(len(np.unique(indices)), 3072)
        np.testing.assert_array_equal(indices, select_calibration_indices(3073))
        with self.assertRaises(ValueError):
            select_calibration_indices(3071)
        dataset = TinyDataset()
        with self.assertRaises(ValueError):
            compute_calibration(ConstantTeacher(), dataset, [0, 1, 2], device="cpu")
        cal, arrays = compute_calibration(ConstantTeacher(), dataset, [0, 1, 2], device="cpu",
                                          batch_size=2, synthetic_test=True)
        errors = np.concatenate([dataset.base(i)[0].abs().mean(0).numpy().reshape(-1) for i in range(3)])
        self.assertEqual(cal["tau_R"], float(np.median(errors)))
        self.assertEqual(cal["q_ref"], CONSTANT_ALIGNER_Q)
        self.assertEqual(arrays["q"].shape, (3, 4))
        np.testing.assert_array_equal(arrays["q"], np.full((3, 4), .46875, np.float32))

    def test_calibration_publishes_only_complete_exact50k_reference_and_reuses(self):
        import yaml
        from safetensors.torch import save_file
        from types import SimpleNamespace
        case = CASES[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = build_config(case)
            cfg["qg40"]["dataset_manifest"] = "data.json"
            run = root / cfg["work_dir"]
            (run / "meta").mkdir(parents=True)
            (run / "meta/config.resolved.yaml").write_text(yaml.safe_dump(cfg))
            start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
            atomic_json(root / cfg["qg40"]["window_path"], dict(campaign_id=CAMPAIGN_ID,
                t0_utc=start.isoformat(), deadline_utc=(start + dt.timedelta(hours=40)).isoformat()))
            original = root / "source.h5"
            original.write_bytes(b"synthetic source identity fixture")
            data = dict(sensor="QB", num_bands=4, max_pixel=2047,
                        splits=dict(train=dict(dataroot=str(original), sha256=sha256(original))))
            atomic_json(root / "data.json", data)
            candidate = run / "candidates/50000"
            candidate.mkdir(parents=True)
            model = nn.Linear(1, 1)
            save_file(model.state_dict(), str(candidate / "model.safetensors"))
            release = {"synthetic_test": True, "content_sha256": "fixture"}
            identity = dict(update=50000, model_sha256=sha256(candidate / "model.safetensors"),
                            config_sha256=object_sha(cfg), data_sha256=object_sha(data),
                            source_identity=release, state_hash=state_hash(model.state_dict()))
            state = dict(identity, full_state=True, model_state=model.state_dict())
            atomic_torch(candidate / "training_state.pt", state)
            identity["training_state_sha256"] = sha256(candidate / "training_state.pt")
            atomic_json(candidate / "identity.json", identity)
            indices = select_calibration_indices(3072)
            values = dict(tau_R=.2, tau_raw_pooled_median=.2, q_ref=.46875, q_shape=[3072, 4],
                          calibration_indices_sha256=object_sha(indices.tolist()))
            arrays = dict(q=np.full((3072, 4), .46875, np.float32), calibration_indices=indices)
            with patch("qg40.common.source_identity", return_value=release), \
                    patch("qg40.common.load_checkpoint_model", return_value=(model, identity)), \
                    patch("qg40.data.build_dataset", return_value=SimpleNamespace(base_count=3072)), \
                    patch("qg40.references.data_signature", return_value={}), \
                    patch("qg40.reference_parity.verify_q_cache", return_value={"synthetic_fixture": True}) as parity, \
                    patch("qg40.error_distribution.write_error_distribution", side_effect=RuntimeError('P1 fixture fault')), \
                    patch("qg40.calibration.compute_calibration", return_value=(values, arrays)) as compute:
                parity.side_effect = ValueError('q-cache/online parity fixture fault')
                with self.assertRaisesRegex(ValueError, 'q-cache/online'):
                    calibrate(case.run_id, root, case.server_id, device='cpu')
                reference_path = root / 'work_dir/_qg40' / case.server_id / 'references' / case.reference_id / 'reference_manifest.json'
                self.assertFalse(reference_path.exists())
                parity.side_effect = None
                parity.reset_mock()
                with contextlib.redirect_stdout(io.StringIO()):
                    destination = calibrate(case.run_id, root, case.server_id, device="cpu")
                self.assertTrue(destination.is_file())
                import json
                manifest = json.loads(destination.read_text())
                report = json.loads((destination.parent / 'reconstruction_distribution.json').read_text())
                self.assertEqual(report['status'], 'DIAGNOSTIC_FAILED')
                self.assertFalse(report['p0_gate'])
                self.assertEqual(manifest["teacher_update"], 50000)
                self.assertEqual(manifest["q_shape"], [3072, 4])
                self.assertEqual(manifest["q_cache_sha256"], sha256(manifest["q_cache_path"]))
                self.assertEqual(manifest["q_cache_parity_sha256"], sha256(manifest["q_cache_parity_path"]))
                parity.assert_called_once()
                with patch("qg40.references.validate_reference") as validate:
                    self.assertEqual(calibrate(case.run_id, root, case.server_id, device="cpu"), destination)
                    validate.assert_called_once()
                self.assertEqual(compute.call_count, 2)

    def test_schedule_boundary_and_grid(self):
        self.assertEqual(len(GRID_STEPS), 50)
        self.assertEqual(tuple(GRID_STEPS), (*range(1010, 50000, 1010), 50000))
        for step in (0, 1, 99, 100, 10000, 24239):
            self.assertEqual(aligner_factor(step, "A24R"), 1.)
        self.assertEqual(aligner_factor(24240, "A24R"), 1 / 3)
        self.assertEqual(aligner_factor(50000, "A24R"), 1 / 3)
        self.assertEqual(cosine_factor(0), 0.)
        self.assertEqual(cosine_factor(100), 1.)
        self.assertEqual(cosine_factor(50000), 0.)

    def test_fixed_diagnostics_routes_gradients_and_restores_model_rng(self):
        from qg40.diagnostics import capture_diagnostics, diagnostic_indices, gradient_statistics
        np.testing.assert_array_equal(diagnostic_indices(3072), diagnostic_indices(3072))
        self.assertEqual(len(diagnostic_indices(3072)), 128)
        stats = gradient_statistics([torch.tensor([2.])], [torch.tensor([-4.])])
        self.assertEqual(stats["norm_ratio"], .5)
        self.assertEqual(stats["cosine"], -1.)
        pan, ms, lp, gt = self.batch()
        class Dataset:
            has_gt, base_count = True, 2
            def __len__(self):
                return 2
            def get_view(self, index, rot, augment):
                return gt[index], gt[index], ms[index], lp[index], pan[index], torch.tensor([index, rot])
        teacher, _ = build_model("P0", 8, [1, 1, 1], 81001)
        self.activate(teacher)
        before = torch.get_rng_state().clone()
        summary, arrays = capture_diagnostics(teacher, None, Dataset(), device="cpu",
            indices=[0, 1], synthetic_test=True, batch_size=2)
        self.assertTrue(teacher.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertIn("weighted_consistency_rec_a_gradient_ratio", summary)
        self.assertIn("e_teacher", arrays)
        teacher.eval().requires_grad_(False)
        student, _ = build_model("PLH", 8, [1, 1, 1], 82001, role="S",
            teacher_aligner_state=teacher.aligner.state_dict())
        self.activate(student)
        summary, arrays = capture_diagnostics(student, teacher, Dataset(), device="cpu",
            tau_R=.2, q_cache=np.full((2, 4), .4), q_ref=.4,
            indices=[0, 1], synthetic_test=True, batch_size=2)
        self.assertTrue(student.training)
        self.assertFalse(teacher.training)
        self.assertIn("soft_hard_gradient_ratio_median", summary)
        self.assertEqual(arrays["e_student"].shape, (2, 32, 32))
        self.assertTrue(all(p.grad is None for p in student.parameters()))

    def test_sampler_optimizer_scheduler_rng_exact_resume(self):
        def setup():
            params = [nn.Parameter(torch.tensor([1.])), nn.Parameter(torch.tensor([2.]))]
            opt = torch.optim.AdamW([dict(params=[params[0]], lr=1e-4, name="U"),
                                     dict(params=[params[1]], lr=3e-6, name="A")])
            return params, opt, make_scheduler(opt, "A24R"), BatchStream(13, 4, 9), torch.Generator().manual_seed(55)
        def advance(parts):
            params, opt, scheduler, stream, rng = parts
            if stream.cursor == stream.count:
                stream.new_epoch()
            batch = stream.remaining_batches()[0]
            opt.zero_grad()
            loss = sum(p.square().sum() for p in params) * torch.rand((), generator=rng)
            loss.backward(); opt.step(); scheduler.step(); stream.cursor += 1
            return batch
        live = setup()
        for _ in range(5):
            advance(live)
        state = copy.deepcopy(dict(params=[p.detach().clone() for p in live[0]], optimizer=live[1].state_dict(),
                                   scheduler=live[2].state_dict(), sampler=live[3].state_dict(), rng=live[4].get_state()))
        resumed = setup()
        with torch.no_grad():
            for param, value in zip(resumed[0], state["params"]):
                param.copy_(value)
        resumed[1].load_state_dict(state["optimizer"]); resumed[2].load_state_dict(state["scheduler"])
        resumed[3].load_state_dict(state["sampler"]); resumed[4].set_state(state["rng"])
        for _ in range(8):
            self.assertEqual(advance(live), advance(resumed))
        for a, b in zip(live[0], resumed[0]):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(live[2].state_dict(), resumed[2].state_dict())

    def test_registered_recipe_rejects_changed_method_and_resume_bundle(self):
        cfg = build_config(CASES[0])
        cfg["qg40"]["dataset_manifest"] = "/bound/data.json"
        self.assertEqual(validate_config(cfg), CASES[0])
        cfg["qg40"]["consistency_weight"] *= 3
        with self.assertRaises(ValueError):
            validate_config(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            save_resume_bundle(tmp, {"full_state": True, "update": 1}, {"update": 1})
            save_resume_bundle(tmp, {"full_state": True, "update": 2}, {"update": 2})
            state = torch.load(Path(tmp) / "last/training_state.pt", weights_only=False)
            self.assertEqual(state["update"], 2)
            self.assertEqual(len(list(Path(tmp).glob(".resume-*"))), 1)

    def test_common_window_exact40h_and_no_fresh_start_after36h(self):
        cfg = build_config(CASES[0])
        now = dt.datetime.now(dt.timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            window = Path(tmp) / cfg["qg40"]["window_path"]
            atomic_json(window, dict(campaign_id=CAMPAIGN_ID, t0_utc=(now - dt.timedelta(hours=37)).isoformat(),
                                     deadline_utc=(now + dt.timedelta(hours=3)).isoformat()))
            with self.assertRaises(TimeoutError):
                runtime_context(cfg, tmp)
            runtime_context(cfg, tmp, resume=True)
            with self.assertRaises(ValueError):
                runtime_context(cfg, tmp, deadline_arg=(now + dt.timedelta(hours=4)).isoformat(), resume=True)

    def test_actual_trainer_deadline_state_and_resume_matches_uninterrupted(self):
        """Only three tiny synthetic updates, inside isolated temporary roots."""
        import yaml
        from torch.utils.data import DataLoader as RealLoader

        class Dataset:
            base_count, has_gt = 48, True
            def __len__(self):
                return 48
            def __getitem__(self, key):
                index, rotation = key
                gen = torch.Generator().manual_seed(index * 4 + rotation)
                pan = torch.randn(1, 16, 16, generator=gen)
                gt = torch.randn(4, 16, 16, generator=gen)
                return (gt, gt, torch.zeros(4, 4, 4), torch.zeros(1, 4, 4), pan,
                        torch.tensor([index, rotation, 1, 1]))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone, self.aligner = nn.Linear(1, 1), nn.Linear(1, 2)
            def predict_delta(self, pan, base):
                return self.aligner(pan.mean((1, 2, 3))[:, None])
            def forward(self, pan, ms, lp):
                delta = self.predict_delta(pan, ms)
                y = self.backbone(pan.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).expand(-1, 4, -1, -1)
                y = y + delta.sum(1)[:, None, None, None]
                return dict(y=y, delta=delta, ms_base=torch.zeros_like(y))

        def initialize(root, case):
            cfg = build_config(case)
            cfg["qg40"]["dataset_manifest"] = "data.json"
            (Path(root) / "config.yaml").write_text(yaml.safe_dump(cfg))
            atomic_json(Path(root) / "data.json", dict(sensor="QB", num_bands=4, max_pixel=2047))
            start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
            atomic_json(Path(root) / cfg["qg40"]["window_path"], dict(campaign_id=CAMPAIGN_ID,
                t0_utc=start.isoformat(), deadline_utc=(start + dt.timedelta(hours=40)).isoformat()))
            return cfg

        def run(root, updates, case, resume=False):
            calls = [0]
            def deadline(_deadline):
                calls[0] += 1
                return calls[0] <= updates * 2
            def loader(*args, **kwargs):
                kwargs["num_workers"] = 0
                return RealLoader(*args, **kwargs)
            def reference(*args, **kwargs):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(55)
                    teacher = Model()
                return teacher, dict(tau_R=.2, q_ref=.4, teacher_run_id=case.teacher_run_id,
                                     sensor=case.sensor), np.full((48, 4), .4, np.float32)
            metric = dict(seconds=0., rr=dict(q4=.8, scc=.9, ergas=3.2),
                          fr=dict(hqnr=.93, d_lambda=.03, d_s=.04), val_ergas=3.1)
            logged = io.StringIO()
            with patch("qg40.data.build_dataset", return_value=Dataset()), \
                    patch("qg40.references.load_reference", side_effect=reference), \
                    patch("qg40.evaluation.FRMetrics", return_value=object()), \
                    patch("qg40.evaluation.evaluate_model", return_value=metric), \
                    patch("qg40.training.GRID_STEPS", (1, 2, 3)), \
                    patch("qg40.training.FULLSTATE_STEPS", (1, 2, 3)), \
                    patch("qg40.diagnostics.capture_diagnostics", side_effect=TimeoutError("P1 budget exhausted")), \
                    patch.dict("sys.modules", {"qg40.alignment_diagnostics": SimpleNamespace(
                        capture_alignment_sizes=lambda *a, **k: {"status": "MEASURED", "synthetic_test": True})}), \
                    patch("qg40.training.build_model", side_effect=lambda *a, **k: (Model(), {})), \
                    patch("qg40.training.source_identity", return_value={"synthetic_test": True}), \
                    patch("qg40.training.before_deadline", side_effect=deadline), \
                    patch("qg40.training.DataLoader", side_effect=loader), \
                    contextlib.redirect_stdout(logged):
                self.assertEqual(train_run(Path(root) / "config.yaml", device="cpu", resume=resume, root=root), 75)
            self.assertIn("rawHQNR=0.93000000 SCC=0.90000000 ERGAS=3.20000000 valERGAS=3.10000000", logged.getvalue())

        def assert_tree_equal(first, second):
            if isinstance(first, torch.Tensor):
                self.assertTrue(torch.equal(first, second))
            elif isinstance(first, np.ndarray):
                np.testing.assert_array_equal(first, second)
            elif isinstance(first, dict):
                self.assertEqual(first.keys(), second.keys())
                for key in first:
                    assert_tree_equal(first[key], second[key])
            elif isinstance(first, (tuple, list)):
                self.assertEqual(len(first), len(second))
                for a, b in zip(first, second):
                    assert_tree_equal(a, b)
            else:
                self.assertEqual(first, second)

        for case in CASES[:2]:
            with self.subTest(role=case.role), tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                cfg = initialize(first, case); initialize(second, case)
                run(first, 3, case)
                run(second, 2, case); run(second, 1, case, resume=True)
                states = [torch.load(Path(root) / cfg["work_dir"] / "last/training_state.pt", weights_only=False)
                          for root in (first, second)]
                self.assertEqual([state["update"] for state in states], [3, 3])
                self.assertEqual(state_hash(states[0]["model_state"]), state_hash(states[1]["model_state"]))
                self.assertTrue(torch.equal(states[0]["corruption_rng"], states[1]["corruption_rng"]))
                self.assertTrue(torch.equal(states[0]['exposure_counts'], states[1]['exposure_counts']))
                self.assertEqual(int(states[1]['exposure_counts'].sum()), 3 * 48)
                self.assertEqual(states[0]["scheduler"], states[1]["scheduler"])
                self.assertTrue(torch.equal(states[0]["sampler"]["data_rng"], states[1]["sampler"]["data_rng"]))
                for key in ('optimizer', 'rng', 'sampler'):
                    assert_tree_equal(states[0][key], states[1][key])
                import json
                status = json.loads((Path(second) / cfg["work_dir"] / "meta/training_status.json").read_text())
                self.assertEqual(status["status"], "INCOMPLETE_AT_DEADLINE")
                self.assertFalse(status["training_complete"])
                self.assertEqual(status["n_evaluated"], 3)
                diagnostic = json.loads((Path(second) / cfg['work_dir'] / 'diagnostics/required_1.json').read_text())
                self.assertEqual(diagnostic['status'], 'INCOMPLETE')
                self.assertFalse(diagnostic['base_training_blocked'])
                grid = json.loads((Path(second) / cfg["work_dir"] / "official/raw_grid.json").read_text())
                self.assertEqual([r["update"] for r in grid["records"]], [1, 2, 3])


def numerical_smoke():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QG40TrainingTests)
    result = unittest.TestResult()
    suite.run(result)
    return dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                failures=[str(test) + "\n" + error for test, error in result.failures + result.errors],
                scope="synthetic CPU numerics; no sensor assets or GPU readiness certification")


if __name__ == "__main__":
    unittest.main()
