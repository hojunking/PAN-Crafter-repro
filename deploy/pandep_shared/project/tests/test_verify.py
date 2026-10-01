import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pan_shared.common import CAMPAIGN, atomic_json, canonical_sha
from pan_shared.safety import SafetyStop
from pan_shared import verify as gate


class VerificationTests(unittest.TestCase):
    def manifest(self):
        return dict(campaign_id=CAMPAIGN, server="s2", hostname="test-s2", microbatch=24,
                    server_identity_file="/server-id", **{key: key + "_identity" for key in gate.IDENTITY_FIELDS})

    def hardware(self, memory="9000"):
        return dict(server="s2", hostname="test-s2", gpu=dict(stdout="GPU-fixture, GPU Model, 24000, " + memory + ", driver-fixture"))

    def run_result(self, *, count=3, success=True, skipped=None, modules=None):
        modules = modules or ["test_model_data", "test_training", "test_registry_safety", "test_eval_sheet", "test_verify"]
        payload = dict(test_count=count, success=success, test_ids=[m + ".Test.test_example" for m in modules],
                       skipped=skipped or [], errors=[], failures=[])
        return SimpleNamespace(returncode=0 if success else 1, stdout="PANDEP_TEST_RESULT=" + json.dumps(payload), stderr="fixture")

    def test_unit_gate_rejects_zero_tests_even_success(self):
        with mock.patch.object(gate.subprocess, "run", return_value=self.run_result(count=0)):
            with self.assertRaisesRegex(SafetyStop, "empty"):
                gate.run_unit_tests()

    def test_unit_gate_rejects_failures_skips_and_missing_categories(self):
        for result in (self.run_result(success=False), self.run_result(skipped=[["test", "missing"]]),
                       self.run_result(modules=["test_training"])):
            with mock.patch.object(gate.subprocess, "run", return_value=result):
                with self.assertRaises(SafetyStop):
                    gate.run_unit_tests()

    def test_unit_gate_validates_fresh_cpu_subprocess(self):
        with mock.patch.object(gate.subprocess, "run", return_value=self.run_result()) as runner:
            receipt = gate.run_unit_tests()
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(runner.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_empty_test_directory_is_not_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SafetyStop):
                gate.tests_manifest(Path(directory))

    def test_hardware_identity_excludes_volatile_free_memory(self):
        self.assertEqual(gate.hardware_identity(self.hardware("9000")), gate.hardware_identity(self.hardware("4000")))
        bad = self.hardware(); bad["gpu"]["stdout"] = "bad"
        with self.assertRaises(SafetyStop):
            gate.hardware_identity(bad)

    def test_cpu_q00_is_not_actual_s2_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SafetyStop, "actual S2 CUDA"):
                gate.q00_smoke(directory, self.manifest(), device="cpu")

    def test_failure_invalidates_previous_pass_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "preflight/gates.json", {"status": "PASS"})
            with mock.patch.object(gate.preflight, "check_identity", side_effect=SafetyStop("identity mismatch")):
                with self.assertRaises(SafetyStop):
                    gate.verify(root, credentials="unused")
            self.assertEqual(json.loads((root / "preflight/gates.json").read_text())["status"], "FAILED")

    def test_verify_only_publishes_pass_after_actual_readonly_sheet_and_q00(self):
        with tempfile.TemporaryDirectory() as directory:
            hardware = gate.hardware_identity(self.hardware())
            with mock.patch.object(gate.preflight, "check_identity", return_value=self.manifest()), \
                 mock.patch.object(gate, "run_unit_tests", return_value={"status": "PASS", "test_count": 42}), \
                 mock.patch.object(gate, "validate_dataset_evidence", return_value={"status": "PASS", "actual_byte_rechecks": 12}), \
                 mock.patch.object(gate, "q00_smoke", return_value={"status": "PASS", "hardware_identity": hardware}) as smoke, \
                 mock.patch("pan_shared.sheets.GoogleSheetAdapter") as adapter, \
                 mock.patch("pan_shared.sheets.inspect_target", return_value={"status": "PASS", "writes": 0}) as inspect:
                receipt = gate.verify(directory, credentials="private-not-read")
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(set(receipt["gates"]), {"G0", "G1", "G2", "G3", "G4", "G5"})
            self.assertFalse(receipt["formal_training_started"])
            self.assertEqual(receipt["sheet_rows_written"], 0)
            smoke.assert_called_once(); inspect.assert_called_once_with(adapter.return_value)
            self.assertFalse(receipt["gates"]["G5"]["actual_sheet_write_test"])

    def test_readonly_sheet_failure_blocks_before_q00(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(gate.preflight, "check_identity", return_value=self.manifest()), \
                 mock.patch.object(gate, "run_unit_tests", return_value={"status": "PASS", "test_count": 42}), \
                 mock.patch.object(gate, "validate_dataset_evidence", return_value={"status": "PASS"}), \
                 mock.patch.object(gate, "q00_smoke") as smoke, \
                 mock.patch("pan_shared.sheets.GoogleSheetAdapter"), \
                 mock.patch("pan_shared.sheets.inspect_target", side_effect=ValueError("schema conflict")):
                with self.assertRaisesRegex(ValueError, "schema conflict"):
                    gate.verify(directory, credentials="unused")
            smoke.assert_not_called()
            self.assertEqual(json.loads((Path(directory) / "preflight/gates.json").read_text())["status"], "FAILED")

    def test_unbound_gates_cannot_admit_formal_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(gate.preflight, "check_identity", return_value=self.manifest()):
                with self.assertRaisesRegex(SafetyStop, "verify --all-gates"):
                    gate.check_gates(directory)

    def test_edited_pass_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            atomic_json(Path(directory) / "preflight/gates.json", {"status": "PASS", "campaign_id": CAMPAIGN})
            with mock.patch.object(gate.preflight, "check_identity", return_value=self.manifest()):
                with self.assertRaisesRegex(SafetyStop, "receipt seal"):
                    gate.check_gates(directory)


if __name__ == "__main__":
    unittest.main()
