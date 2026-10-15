"""Cost integration guards. Never run a GPU profile in these tests."""
from copy import deepcopy
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import torch

from . import cost
from .test_reporting import config


class CostTests(unittest.TestCase):
    def test_cache_hit_requires_no_gpu_or_trainer(self):
        saved = {"status": "verified cache"}
        with patch("experiment_cost.store.find_profile", return_value=saved), \
             patch("donor30.cost._gpu_pids", side_effect=AssertionError("GPU probe forbidden")):
            self.assertIs(cost.ensure_cost(config(), None, retained_trainer=None,
                device="cpu", deadline_unix=0), saved)

    def test_budget_failure_does_not_save_fake_cost(self):
        with patch("experiment_cost.store.find_profile", return_value=None), \
             patch("experiment_cost.store.save_profile") as save:
            with self.assertRaisesRegex(TimeoutError, "120 seconds"):
                cost.ensure_cost(config(), None, retained_trainer=None, device="cpu", deadline_unix=time.time())
            save.assert_not_called()

    def test_cpu_profiling_cannot_publish_production_cost(self):
        with patch("experiment_cost.store.find_profile", return_value=None):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                cost.ensure_cost(config(), None, retained_trainer=None, device="cpu", deadline_unix=time.time()+300)

    def test_retained_model_and_optimizer_cpu_checked(self):
        stage = torch.nn.Linear(2, 2)
        trainer = SimpleNamespace(stage=stage, optimizer=torch.optim.AdamW(stage.parameters()), updates=0)
        cost._assert_retained_cpu(trainer)
        trainer.optimizer.state[stage.weight]["nested"] = {"state": [torch.empty(1, device="meta")]}
        with self.assertRaisesRegex(ValueError, "optimizer"):
            cost._assert_retained_cpu(trainer)
        with self.assertRaisesRegex(ValueError, "Actual retained"):
            cost._assert_retained_cpu(None)

    def test_nvidia_pid_parse_fail_closed(self):
        with patch("subprocess.check_output", return_value="100\n200\n"):
            self.assertEqual(cost._gpu_pids(), {100, 200})
        with patch("subprocess.check_output", return_value="[Not Supported]\n"):
            with self.assertRaises(RuntimeError):
                cost._gpu_pids()

    def test_idle_blas_workspace_released_before_zero_allocation_check(self):
        order = []
        with patch("torch.cuda.synchronize", side_effect=lambda device: order.append("sync")), \
             patch.object(torch._C, "_cuda_clearCublasWorkspaces", create=True,
                          side_effect=lambda: order.append("clear_blas")), \
             patch("torch.cuda.empty_cache", side_effect=lambda: order.append("empty_cache")), \
             patch("torch.cuda.memory_allocated", side_effect=lambda device: order.append("check") or 0):
            cost._require_empty_cuda("cuda")
        self.assertEqual(order, ["sync", "clear_blas", "empty_cache", "check"])

    def test_live_allocation_still_rejected_after_workspace_cleanup(self):
        with patch("torch.cuda.synchronize"), \
             patch.object(torch._C, "_cuda_clearCublasWorkspaces", create=True), \
             patch("torch.cuda.empty_cache"), patch("torch.cuda.memory_allocated", return_value=4096):
            with self.assertRaisesRegex(RuntimeError, "Nonzero CUDA allocation.*4096"):
                cost._require_empty_cuda("cuda")

    def test_fresh_stage2_surrogate_uses_student_only(self):
        cfg = config(phase="stage2")
        cfg["stream_seeds"]["S_init"] = 123
        with patch("directshift48.training.build_direct_stage1", return_value="fresh graph") as build:
            self.assertEqual(cost.build_deployment(cfg, "sources"), "fresh graph")
        kwargs = build.call_args.kwargs
        self.assertEqual(kwargs["teacher_spec"].width, 104)
        self.assertEqual(kwargs["teacher_spec"].depth, (1, 2, 2))
        self.assertEqual(kwargs["teacher_seed"], 123)
        self.assertEqual(cfg["baseline_resolved"]["teacher_architecture"]["width"], 112)

    def test_stage2_labels_explicitly_exclude_frozen_teacher(self):
        fields = {"Cost_scope": "native A+warp+Teacher"}
        value = {"parameter_components": {"aligner": 10, "teacher": 20},
            "memory_definition": "A+T+inputs", "observations": {s: {"sheet_fields": deepcopy(fields)} for s in ("rr", "fr")}}
        cost._stage2_labels(value)
        self.assertEqual(value["parameter_components"], {"aligner": 10, "student": 20})
        self.assertIn("Student", value["observations"]["rr"]["sheet_fields"]["Cost_scope"])


if __name__ == "__main__":
    unittest.main()
