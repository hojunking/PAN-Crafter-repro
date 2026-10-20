"""Controller admission/deadline/amendment tests; no data, GPU or cloud calls."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from directshift48.reporting import atomic_json
from panda_mature30 import plan
from panda_mature30 import runtime as R


class FakeClock:
    def __init__(self):
        self.wall, self.monotonic = 200000., 50000.

    def __call__(self):
        return self.wall

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic += seconds


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = FakeClock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.real_code_pin = R.code_pin
        self.stack.enter_context(patch.object(R, "code_pin", return_value={"synthetic.py": "d"*64}))
        self.stack.enter_context(patch.object(R, "evaluator_identity",
                                             return_value={"protocol": "SYNTHETIC_ONLY"}))
        self.stack.enter_context(patch.object(R.time, "monotonic",
                                             side_effect=lambda: self.clock.monotonic))
        self.stack.enter_context(patch.object(R.shutil, "disk_usage",
                                             return_value=SimpleNamespace(free=10*1024**3)))
        self.stack.enter_context(patch.object(R.torch.cuda, "synchronize",
                                             side_effect=AssertionError("No GPU work")))
        self.stack.enter_context(patch.object(R.torch.cuda, "empty_cache",
                                             side_effect=AssertionError("No GPU work")))
        self.campaign = R.Campaign("s1", output=self.temp.name, device="cpu", clock=self.clock)

    def group(self, name):
        return next(g for g in self.campaign.config["groups"] if g["name"] == name)

    def install_measured_stage2_timing(self):
        cfg = plan.resolve_case("s1", "HALF", "R1")
        self.campaign.state["timing"][self.campaign.timing_key(cfg)] = {
            "updates_observed": 1000, "update_p90": .1, "eval_seconds": 2.,
            "save_seconds": 1.,
        }

    def test_budget_persists_across_restart_and_clock_rollback(self):
        c = self.campaign
        started, deadline = c.state["started_unix"], c.deadline_unix
        self.clock.advance(7200.)
        c.persist()
        restarted = R.Campaign("s1", output=self.temp.name, device="cpu", clock=self.clock)
        self.assertEqual(restarted.state["started_unix"], started)
        self.assertEqual(restarted.deadline_unix, deadline)
        self.assertEqual(restarted.elapsed_seconds(), 7200.)
        self.clock.wall -= 600.
        self.clock.monotonic += 100.
        self.assertEqual(restarted.elapsed_seconds(), 7300.)
        restarted.persist()
        again = R.Campaign("s1", output=self.temp.name, device="cpu", clock=self.clock)
        self.assertGreaterEqual(again.elapsed_seconds(), 7300.)
        self.assertEqual(again.deadline_unix, deadline)

    def test_deadline_reserves_closeout_and_stop_disk_fail_closed(self):
        c = self.campaign
        limit = c.config["budget"]["total_seconds"] - c.config["budget"]["closeout_seconds"]
        self.clock.advance(limit - 1.)
        c.check()
        self.clock.advance(1.)
        with self.assertRaisesRegex(R.BudgetExpired, "DURABLE"):
            c.check()
        c.closeout = True
        c.check()
        self.clock.advance(c.config["budget"]["closeout_seconds"])
        with self.assertRaisesRegex(R.BudgetExpired, "DURABLE"):
            c.check()
        c.stop_requested = True
        with self.assertRaisesRegex(R.BudgetExpired, "STOP_REQUESTED"):
            c.check()
        c.stop_requested = False
        with patch.object(R.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(R.BudgetExpired, "DISK_CLOSEOUT"):
                c.check()

    def test_optional_pair_requires_measured_timing_and_entire_pair_budget(self):
        c, group = self.campaign, self.group("paired_repeat")
        self.assertEqual(len(group["runs"]), 2)
        self.assertFalse(c.admit_group(group))
        receipt = c.state["groups"][-1]
        self.assertFalse(receipt["measured_operator_timing"])
        self.assertEqual(len(receipt["pending"]), 2)
        self.install_measured_stage2_timing()
        self.assertTrue(c.admit_group(group))
        required = c.state["groups"][-1]["required_seconds"]
        one = c.estimate_seconds(plan.resolve_case("s1", *group["runs"][0]))
        self.assertGreater(required, one)
        available = one + (required - one) / 2
        limit = c.config["budget"]["total_seconds"] - c.config["budget"]["closeout_seconds"]
        self.clock.advance(limit - available)
        self.assertFalse(c.admit_group(group))
        receipt = c.state["groups"][-1]
        self.assertGreater(receipt["remaining_seconds"], one)
        self.assertLess(receipt["remaining_seconds"], required)
        self.assertEqual(len(receipt["pending"]), 2)
        self.assertEqual(c.state["runs"], [])  # Admission cannot launch half a pair.

    def test_core_prior_is_allowed_but_labelled_unmeasured(self):
        c = self.campaign
        self.assertTrue(c.admit_group(self.group("full")))
        receipt = c.state["groups"][-1]
        self.assertFalse(receipt["optional"])
        self.assertFalse(receipt["measured_operator_timing"])
        self.assertGreater(receipt["required_seconds"], 0.)

    def test_pending_skip_records_reason_and_refuses_any_started_member(self):
        c, group = self.campaign, self.group("paired_repeat")
        request = {"request_id": "synthetic-revision-1",
                   "reason": "Explicitly drop an unstarted optional repeat",
                   "skip_groups": ["paired_repeat"]}
        atomic_json(c.root/"pending_control.json", request)
        self.assertTrue(c.pending_skip(group))
        receipt = c.state["pending_amendments"][-1]
        self.assertEqual(receipt["reason"], request["reason"])
        self.assertEqual(receipt["request_id"], request["request_id"])
        self.assertFalse(c.pending_skip(self.group("full")))
        cfg = plan.resolve_case("s1", *group["runs"][0])
        for status in ("RUNNING", "STOPPED", "FAILED", "COMPLETED"):
            c.state["runs"] = [{"run_id": cfg["run_id"], "status": status}]
            self.assertTrue(c.group_started(group))
            with self.assertRaisesRegex(ValueError, "Cannot skip started group"):
                c.pending_skip(group)

    def test_pending_control_cannot_change_policies_or_unknown_groups(self):
        good = {"request_id": "id", "reason": "reason", "skip_groups": ["cap_screen"]}
        invalid = [
            {**good, "skip_groups": ["unregistered"]},
            {**good, "skip_groups": "cap_screen"},
            {**good, "policy_id": "LOW"},
            {**good, "reason": ""},
            {**good, "request_id": ""},
        ]
        for value in invalid:
            atomic_json(self.campaign.root/"pending_control.json", value)
            with self.assertRaisesRegex(ValueError, "Only explicit"):
                self.campaign.pending_skip(self.group("cap_screen"))

    def test_code_changes_block_continuation_without_mutating_state(self):
        c = self.campaign
        before = deepcopy(c.state)
        c.verify_code()
        with patch.object(R, "code_pin", return_value={"synthetic.py": "e"*64}):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                c.verify_code()
        self.assertEqual(c.state, before)

    def test_shared_operator_timing_covers_all_registered_stage2_policies(self):
        keys = {self.campaign.timing_key(plan.resolve_case("s1", policy))
                for policy in ("FULL", "HALF", "DOWN", "LOW", "CAP085")}
        self.assertEqual(len(keys), 1)
        stage1_key = self.campaign.timing_key(plan.resolve_case("s1", "G8_MATURE_U23"))
        self.assertNotIn(stage1_key, keys)

    def test_code_pin_includes_new_core_shared_warp_and_registration(self):
        pins = self.real_code_pin()
        for name in ("panda_mature30/spec.py", "panda_mature30/method.py",
                     "panda_mature30/calibration.py", "panda_mature30/engine.py",
                     "panda_band30/reconstruction.py", "pa/warp.py",
                     "panda_midpoint/resampling.py", "ta2/registration.py"):
            self.assertIn(name, pins)
            self.assertRegex(pins[name], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
