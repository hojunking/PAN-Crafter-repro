#!/usr/bin/env python
"""CPU/tempdir tests for MIX20H guards, pair identity and exact deadline resume."""
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from kdv import mix20h_runtime as rt
from kdv.mix20h_plan import CASES, metadata_for
from kdv.resume import EpochState, begin_epoch
from tools.gen_mix20h_configs import build_config


class RandomDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        # Worker augmentation randomness must survive replay, not only shuffle.
        return np.asarray([index / 20., random.random(), np.random.rand()], dtype=np.float32), index


def numpy_collate(batch):
    # Avoid torch tensor FD/socket transfer (sandbox forbids AF_UNIX sockets).
    return np.stack([row[0] for row in batch]), np.asarray([row[1] for row in batch])


class GeneratorState:
    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)

    def state_dict(self):
        return {"rng": self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state["rng"])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mix20h-runtime-")
        self.root = Path(self.tmp.name)
        self.cfg = build_config(CASES[0].run_id)
        self.cfg["kdv"].pop("budget", None)
        self.cfg["work_dir"] = str(self.root / "work_dir" / CASES[0].run_id)
        self.cfg["config"] = str(self.root / "config.yaml")
        Path(self.cfg["config"]).write_text(yaml.safe_dump(self.cfg))
        self.args = SimpleNamespace(**self.cfg)
        now = time.time()
        self.plan = dict(campaign_id=rt.CAMPAIGN, queue_revision=rt.REVISION, server_id="s1", activation_state="ready",
                         config_hashes={CASES[0].run_id: rt.sha256(self.cfg["config"])},
                         campaign_started_at_utc=rt.utc(now - 60), deadline_at_utc=rt.utc(now - 60 + 72000))
        self.plan_path = self.root / "work_dir/_qrc24_mix20h/plan_manifest.json"
        rt.atomic_json(self.plan_path, self.plan)
        self.admission_path = Path(self.cfg["work_dir"]) / "meta/mix20h_admission.json"
        self.admission = dict(campaign_id=rt.CAMPAIGN, queue_revision=rt.REVISION, server_id="s1",
                              run_id=CASES[0].run_id, pair_id=CASES[0].pair_id, status="admitted",
                              deadline_at_utc=self.plan["deadline_at_utc"], config_sha256=rt.sha256(self.cfg["config"]))
        rt.atomic_json(self.admission_path, self.admission)

    def tearDown(self):
        self.tmp.cleanup()

    def runtime(self):
        return rt.Mix20Runtime(self.args, rt.validate_launch(self.args, root=self.root))

    def test_all_registered_definitions_and_old_opt_out(self):
        for case in CASES:
            cfg = build_config(case.run_id)
            cfg["kdv"].pop("budget", None)
            self.assertEqual(rt.validate_definition(cfg)["run_id"], case.run_id)
        self.assertIsNone(rt.validate_launch({"kdv": {"budget": {"required": True}}}))

    def test_admission_missing_changed_seed_and_yaml(self):
        rt.validate_launch(self.args, root=self.root)
        with patch.object(self.args, "seed", 52002), self.assertRaises(ValueError):
            rt.validate_launch(self.args, root=self.root)
        self.admission_path.unlink()
        with self.assertRaises(FileNotFoundError):
            rt.validate_launch(self.args, root=self.root)
        rt.atomic_json(self.admission_path, self.admission)
        with open(self.cfg["config"], "a") as handle:
            handle.write("# config changed after admission\n")
        with self.assertRaises(ValueError):
            rt.validate_launch(self.args, root=self.root)

    def test_reject_profile_definition_and_legacy_budget(self):
        for field, value in (("budget", {"required": True}), ("aligner_schedule", {"freeze_from": 100}),
                             ("phase", {"parent": "earlier"})):
            cfg = copy.deepcopy(self.cfg); cfg["kdv"][field] = value
            with self.assertRaises(ValueError):
                rt.validate_definition(cfg)
        cfg = copy.deepcopy(self.cfg); cfg["kdv"]["rec"]["kd_weight"] = .2
        with self.assertRaises(ValueError):
            rt.validate_definition(cfg)

    def test_persistent_deadline_and_resume_identity(self):
        context = rt.validate_launch(self.args, root=self.root)
        with self.assertRaises(ValueError):
            rt.validate_launch(self.args, root=self.root, now=context["deadline"])
        altered = dict(self.plan, deadline_at_utc=rt.utc(context["deadline"] + 1))
        rt.atomic_json(self.plan_path, altered)
        with self.assertRaises(ValueError):
            rt.validate_launch(self.args, root=self.root)
        rt.atomic_json(self.plan_path, self.plan)
        checkpoint = Path(self.cfg["work_dir"]) / "checkpoint-budget-3"; checkpoint.mkdir()
        for name in rt.RESUME_FILES:
            (checkpoint / name).write_bytes(b"fixture")
        rt.atomic_json(checkpoint / "mix20h_resume.json", dict(context["identity"], step=3, exact_resume=True))
        previous = dict(context["identity"], deadline_at_utc=self.plan["deadline_at_utc"])
        rt.atomic_json(checkpoint.parent / "meta/mix20h_run_manifest.json", previous)
        self.args.resume = str(checkpoint)
        rt.validate_launch(self.args, root=self.root)
        self.args.resume = str(self.root)
        with self.assertRaises(ValueError):
            rt.validate_launch(self.args, root=self.root)

    def test_candidate_resume_requires_post_eval_commit_and_rejects_newer_grid(self):
        context = rt.validate_launch(self.args, root=self.root)
        wd = Path(self.cfg["work_dir"])
        rt.atomic_json(wd / "meta/mix20h_run_manifest.json", dict(context["identity"], deadline_at_utc=self.plan["deadline_at_utc"]))
        candidate = wd / "candidates/step-9090"; candidate.mkdir(parents=True)
        for name in rt.RESUME_FILES:
            (candidate / name).write_bytes(b"fixture")
        binding = dict(run_id=CASES[0].run_id, step=9090, checkpoint_sha256=rt.sha256(candidate / "model.safetensors"),
                       config_sha256=context["config_sha256"], eval_mode="A_ON")
        rt.atomic_json(candidate / "mix20h_identity.json", binding)
        self.args.resume = str(candidate)
        with self.assertRaises(FileNotFoundError):
            rt.validate_launch(self.args, root=self.root)
        rt.atomic_json(candidate / rt.EVALUATION_COMMIT, dict(binding, post_evaluation=True,
                       state_file_sha256={name: rt.sha256(candidate / name) for name in rt.RESUME_FILES}))
        (wd / "checkpoint_metrics.csv").write_text("step\n9090\n")
        rt.validate_launch(self.args, root=self.root)
        (wd / "checkpoint_metrics.csv").write_text("step\n9090\n10100\n")
        with self.assertRaisesRegex(ValueError, "roll back"):
            rt.validate_launch(self.args, root=self.root)
        (wd / "checkpoint_metrics.csv").write_text("step\n9090\n")
        (candidate / "optimizer.bin").write_bytes(b"changed_optimizer_after_commit")
        with self.assertRaises(ValueError):
            rt.validate_launch(self.args, root=self.root)

    def test_50k_partial_grid_never_replays_a_duplicate_eval(self):
        context = rt.validate_launch(self.args, root=self.root)
        wd = Path(self.cfg["work_dir"])
        rt.atomic_json(wd / "meta/mix20h_run_manifest.json", dict(context["identity"], deadline_at_utc=self.plan["deadline_at_utc"]))
        checkpoint = wd / "checkpoint-50000"; checkpoint.mkdir()
        for name in rt.RESUME_FILES:
            (checkpoint / name).write_bytes(b"fixture")
        self.args.resume = str(checkpoint)
        (wd / "checkpoint_metrics.csv").write_text("step\n49490\n")
        rt.validate_launch(self.args, root=self.root)  # Safe: no 50K row yet; eval only.
        (wd / "checkpoint_metrics.csv").write_text("step\n49490\n50000\n")
        with self.assertRaises(ValueError):
            rt.validate_launch(self.args, root=self.root)

    def test_same_epoch_grid_step_without_commit_is_not_resumable(self):
        context = rt.validate_launch(self.args, root=self.root)
        wd = Path(self.cfg["work_dir"])
        rt.atomic_json(wd / "meta/mix20h_run_manifest.json", dict(context["identity"], deadline_at_utc=self.plan["deadline_at_utc"]))
        checkpoint = wd / "epoch-25"; checkpoint.mkdir()
        for name in rt.RESUME_FILES:
            (checkpoint / name).write_bytes(b"fixture")
        self.args.resume = str(checkpoint)
        (wd / "checkpoint_metrics.csv").write_text("step\n4040\n")
        rt.validate_launch(self.args, root=self.root)
        (wd / "checkpoint_metrics.csv").write_text("step\n4040\n5050\n")
        with self.assertRaises(ValueError):
            rt.validate_launch(self.args, root=self.root)

    def test_actual_post_evaluation_commit_restores_rng_and_all_checkpoint_states(self):
        runtime = self.runtime()
        runtime.manifest = dict(runtime.identity, deadline_at_utc=self.plan["deadline_at_utc"], precision="no")
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        first, second, epoch = GeneratorState(1), GeneratorState(2), EpochState()
        for state in (first, second, epoch):
            accelerator.register_for_checkpointing(state)
        epoch.set(torch.get_rng_state(), 8888, 202)  # 9090 is the completed epoch boundary.
        model(torch.ones(2, 2)).square().mean().backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
        candidate = runtime.wd / "candidates/step-9090"
        accelerator.save_state(str(candidate))
        (runtime.meta / "config.yaml").write_bytes(Path(self.cfg["config"]).read_bytes())
        trainer = SimpleNamespace(accelerator=accelerator, _global_step=9090, fr_h5_sha="fixture_FR",
                                  _mix20_evaluator_hash=lambda: "fixture_evaluator")
        runtime.candidate(trainer, candidate, 9090, .959)
        self.assertFalse(runtime.has_committed_evaluation(9090))
        # Simulate RNG consumption after the earlier _select snapshot.
        torch.rand(7); np.random.rand(3); random.random()
        torch.rand(2, generator=first.generator); torch.rand(3, generator=second.generator)
        expected_torch = torch.get_rng_state().clone(); expected_numpy = np.random.get_state(); expected_python = random.getstate()
        expected_first = first.generator.get_state().clone(); expected_second = second.generator.get_state().clone()
        runtime.commit_evaluation(trainer, 9090)
        self.assertTrue(runtime.has_committed_evaluation(9090))
        marker = rt.validate_candidate_commit(candidate, runtime.identity["run_id"], runtime.context["config_sha256"])
        self.assertEqual(set(marker["state_file_sha256"]), set(rt.RESUME_FILES))
        with self.assertRaises(ValueError):
            runtime.commit_evaluation(trainer, 9090)
        torch.rand(20); np.random.rand(20); random.random()
        torch.rand(10, generator=first.generator); torch.rand(10, generator=second.generator)
        accelerator.load_state(str(candidate))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch))
        self.assertTrue(np.array_equal(np.random.get_state()[1], expected_numpy[1]))
        self.assertEqual(np.random.get_state()[2:], expected_numpy[2:])
        self.assertEqual(random.getstate(), expected_python)
        self.assertTrue(torch.equal(first.generator.get_state(), expected_first))
        self.assertTrue(torch.equal(second.generator.get_state(), expected_second))
        accelerator.end_training()

    def test_actual_trainer_at_50k_takes_zero_optimizer_updates(self):
        from train_kdv import KDVTrainer
        from pa.model import PAModel
        trainer = object.__new__(KDVTrainer)
        completed = []
        trainer.mix20h = SimpleNamespace(stop_if_expired=lambda *args: None, training_complete=completed.append)
        trainer.args = SimpleNamespace(num_iter=50000)
        trainer.model = PAModel(torch.nn.Linear(2, 1), None, sampler=False)
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
        trainer.optimizer = SimpleNamespace(step=lambda: self.fail("50K resume must not train again"))
        self.assertEqual(trainer.train(SimpleNamespace(write=lambda _: None), 50000), 50000)
        self.assertEqual(completed, [50000])

    def test_pair_verification_and_mismatch_are_recorded(self):
        runtime = self.runtime()
        runtime.manifest = dict(runtime.identity, pair_recipe_hash="recipe", init_unet_sha256="U",
                                init_aligner_sha256="A", batch_meta_prefix={})
        runtime._write(); runtime._check_pair()
        self.assertEqual(runtime.manifest["pair_status"], "pair_unverified")
        for step in range(rt.META_PREFIX_BATCHES):
            runtime.observe_batch(step, [[step, 0, 1, 1]])
        mate = CASES[1]
        mate_path = Path(self.cfg["work_dir"]).parent / mate.run_id / "meta/mix20h_run_manifest.json"
        other = dict(runtime.manifest, run_id=mate.run_id, profile=mate.profile)
        rt.atomic_json(mate_path, other)
        runtime._check_pair()
        self.assertEqual(runtime.manifest["pair_status"], "pair_verified")
        self.assertEqual(rt._json(mate_path)["pair_status"], "pair_verified")
        other["init_unet_sha256"] = "different"; rt.atomic_json(mate_path, other)
        runtime._check_pair()
        self.assertEqual(runtime.manifest["pair_status"], "pair_unverified")
        self.assertIn("init_unet_sha256_mismatch", runtime.manifest["pair_verification_reasons"])

    def test_export_deadline_does_not_save_older_selected_weights_as_50k(self):
        runtime = self.runtime(); runtime.context["deadline"] = time.time() - 1
        runtime.manifest = dict(runtime.identity, deadline_at_utc=rt.utc(runtime.context["deadline"]))
        last = runtime.wd / "last"; last.mkdir(parents=True)
        (last / "model.safetensors").write_bytes(b"authoritative-exact-50000")
        rt.atomic_json(runtime.wd / "last_meta.json", {"step": 50000})
        accelerator = SimpleNamespace(end_training=lambda: None,
                                      save_state=lambda _: self.fail("export must not serialize selected weights"))
        trainer = SimpleNamespace(accelerator=accelerator, _runs_csv=lambda _: None, _write_cost=lambda _: None)
        with self.assertRaises(SystemExit) as stopped:
            runtime.stop_evaluation_if_expired(trainer, 50000)
        self.assertEqual(stopped.exception.code, 5)
        self.assertEqual(runtime.manifest["status"], "training_complete_eval_pending_budget")
        self.assertTrue(runtime.manifest["training_complete"])
        self.assertEqual((last / "model.safetensors").read_bytes(), b"authoritative-exact-50000")
        self.assertFalse(list(runtime.wd.glob("checkpoint-budget-*")))

    def test_runtime_contract_keeps_effective_defaults(self):
        observed = self.cfg["kdv"]["mix20h"]["expected_runtime"]
        changed = copy.deepcopy(observed); changed["seed"] = 52001
        changed["param_groups"][0]["lr_now"] = 1e-4
        self.assertEqual(rt.runtime_contract(observed), rt.runtime_contract(changed))
        changed["mixed_precision"] = "bf16"
        self.assertNotEqual(rt.runtime_contract(observed), rt.runtime_contract(changed))

    def test_bind_actual_cpu_modules_optimizer_and_runtime_manifest(self):
        from diffusers.optimization import get_scheduler
        from model.pancrafter_paper import PANCrafterPaper
        from pa.aligner import PANGlobalAligner
        from pa.model import PAModel
        from train_kdv import KDVTrainer
        runtime = self.runtime()
        model = PAModel(PANCrafterPaper(**self.args.model_args), PANGlobalAligner(8), aligner_margin=4)
        teacher = copy.deepcopy(model); teacher.eval(); teacher.requires_grad_(False)
        optimizer = torch.optim.AdamW([
            dict(name="backbone", params=list(model.backbone.parameters())),
            dict(name="aligner", params=list(model.aligner.parameters()), lr=3e-6)], lr=1e-4, weight_decay=.01)
        scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=100, num_training_steps=50000)
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        fake = SimpleNamespace(args=self.args, k=self.args.kdv, M=model, teacher=teacher,
                               optimizer=optimizer, lr_scheduler=scheduler, accelerator=accelerator,
                               train_data_loader=list(range(202)), aligner_view_margin=4, aligner_trainable=True,
                               teacher_manifest={"file_sha256": rt.TEACHER_SHA}, donor_manifest={"file_sha256": rt.TEACHER_SHA})
        fake._optimizer_manifest = lambda: KDVTrainer._optimizer_manifest(fake)
        datasets = {key: dict(path="reference", sha256=sha) for key, sha in self.args.kdv["mix20h"]["expected_dataset_hashes"].items()}
        rt.atomic_json(runtime.wd / "dataset_hashes.json", datasets)
        runtime.bind(fake)
        resolved = rt._json(runtime.meta / "resolved_recipe.json")
        self.assertEqual(resolved["runtime"]["mixed_precision"], "no")
        self.assertEqual(resolved["runtime"]["param_groups"][0]["n_elements"], 1903624)
        self.assertEqual(len(runtime.manifest["init_aligner_sha256"]), 64)
        self.assertEqual(runtime.manifest["status"], "training")
        self.assertEqual(runtime.manifest["pair_status"], "pair_unverified")
        with self.assertRaises(ValueError):
            rt.validate_launch(self.args, root=self.root)  # Existing run cannot restart fresh.
        accelerator.end_training()

    def test_deadline_checkpoint_restores_weights_optimizer_scheduler_rng_and_batches(self):
        def build():
            torch.manual_seed(812); np.random.seed(812); random.seed(812)
            accelerator = Accelerator(cpu=True, mixed_precision="no")
            model = torch.nn.Linear(3, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1. - step / 20.)
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
            epoch = EpochState()
            for state in (GeneratorState(1), GeneratorState(2), epoch):
                accelerator.register_for_checkpointing(state)
            loader = DataLoader(RandomDataset(), batch_size=2, shuffle=True, num_workers=2,
                                collate_fn=numpy_collate, timeout=15)
            return accelerator, model, optimizer, scheduler, epoch, loader

        def step(model, optimizer, scheduler, batch):
            values, _ = batch
            loss = model(torch.from_numpy(values)).square().mean()
            loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()

        acc, model, opt, sched, epoch, loader = build()
        iterator, _, _ = begin_epoch(loader, epoch, 0, False)
        reference = []
        for i in range(8):
            batch = next(iterator); reference.append(batch[0].copy()); step(model, opt, sched, batch)
        final_state = copy.deepcopy(acc.unwrap_model(model).state_dict())
        final_opt = copy.deepcopy(opt.state_dict()); final_sched = copy.deepcopy(sched.state_dict())
        acc.end_training()

        acc, model, opt, sched, epoch, loader = build()
        iterator, _, _ = begin_epoch(loader, epoch, 0, False)
        for i in range(3):
            batch = next(iterator); step(model, opt, sched, batch)
        runtime = self.runtime()
        runtime.context["deadline"] = time.time() - 1
        runtime.manifest = dict(runtime.identity, deadline_at_utc=rt.utc(runtime.context["deadline"]))
        trainer = SimpleNamespace(accelerator=acc, _runs_csv=lambda _: None, _write_cost=lambda _: None)
        with self.assertRaises(SystemExit) as stopped:
            runtime.stop_if_expired(trainer, 3)
        self.assertEqual(stopped.exception.code, rt.EXIT_STOPPED_BUDGET)
        checkpoint = Path(runtime.manifest["resume_checkpoint"])
        self.assertTrue((checkpoint / "optimizer.bin").exists())
        self.assertTrue((checkpoint / "scheduler.bin").exists())
        self.assertTrue((checkpoint / "random_states_0.pkl").exists())
        self.assertTrue((checkpoint / "custom_checkpoint_0.pkl").exists())
        self.assertFalse(runtime.manifest["training_complete"])
        self.assertFalse((runtime.wd / "last_meta.json").exists())

        acc, model, opt, sched, epoch, loader = build()
        acc.load_state(str(checkpoint))
        iterator, skip, info = begin_epoch(loader, epoch, 3, True)
        self.assertEqual(skip, 3); self.assertTrue(info["exact"])
        for i in range(3, 8):
            batch = next(iterator)
            self.assertTrue(np.array_equal(batch[0], reference[i]), f"augmented batch {i}")
            step(model, opt, sched, batch)
        for key, value in acc.unwrap_model(model).state_dict().items():
            self.assertTrue(torch.equal(value, final_state[key]), key)
        self.assertEqual(sched.state_dict(), final_sched)
        for param, state in opt.state_dict()["state"].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, final_opt["state"][param][key]))
        acc.end_training()

    def test_same_prestep_split_gradient_beta_changes_only_U(self):
        from train_kdv import KDVTrainer
        def gradients(beta):
            torch.manual_seed(15)
            model = SimpleNamespace(backbone=torch.nn.Linear(2, 1), aligner=torch.nn.Linear(2, 2))
            inputs = torch.tensor([[.2, .4], [.7, .3]])
            output = model.backbone(model.aligner(inputs)); hard = output.square().mean()
            soft = (output - .6).abs().mean(); edge = (output[0] - output[1]).abs().mean()
            trainer = object.__new__(KDVTrainer)
            trainer.accelerator = SimpleNamespace(gradient_accumulation_steps=1)
            KDVTrainer._qrecon_backward(trainer, {"_L_U_t": hard + beta * soft + .002 * edge, "_L_A_t": .5 * hard}, model, True)
            return ([p.grad.clone() for p in model.backbone.parameters()], [p.grad.clone() for p in model.aligner.parameters()])
        u1, a1 = gradients(.1); u2, a2 = gradients(.2)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(u1, u2)))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a1, a2)))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
