"""Controller bridge checks; mock quality/Sheet calls, no cloud or training."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from directshift48.reporting import atomic_json
from panda_mature30.runtime import Campaign
from panda_mature30.plan import resolve_case


class RuntimeSelectionTests(unittest.TestCase):
    def test_s3_registers_rre_donor_and_publishes_same_original_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "runs" / "source" / "attempt_01"
            checkpoint = source / "models" / "step_0050000.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            cfg = json.loads(json.dumps(resolve_case("s3", "G8_MATURE_U23")))
            evaluation = {"recorded_at": "2026-10-20T00:00:00+00:00",
                          "state_before": {"model_state_sha256": "a" * 64}}
            evidence = source / "evaluation_0050000.json"
            atomic_json(evidence, evaluation)
            artifact = {"file_sha256": "b" * 64, "identity": evaluation["state_before"]}
            atomic_json(checkpoint.with_suffix(".json"), {"artifact": artifact})
            selected = {"source_config": deepcopy(cfg), "checkpoint_path": str(checkpoint),
                        "file_sha256": "b" * 64, "source_evaluation_path": str(evidence),
                        "completed_update": 50000}
            receipt = {"selected": selected, "receipt_sha256": "c" * 64}
            campaign = Campaign.__new__(Campaign)
            campaign.server, campaign.device, campaign.root = "s3", "cpu", root
            campaign.state = {"donors": {}}
            campaign.sources, campaign.registered = object(), object()
            campaign.completed = Mock(return_value={"directory": str(source)})
            campaign.check, campaign.persist, campaign.release = Mock(), Mock(), Mock()
            donor = SimpleNamespace(binding={"donor_checkpoint_id": "b" * 64},
                                    assert_frozen=Mock())
            def select(paths, sources, **kwargs):
                self.assertEqual(paths, [checkpoint])
                self.assertIs(sources, campaign.sources)
                atomic_json(kwargs["output"], receipt)
                return deepcopy(receipt)
            with patch("panda_mature30.selection.select_checkpoints", side_effect=select) as selection, \
                 patch("panda_mature30.selection.validate_receipt", side_effect=lambda r: r), \
                 patch("panda_mature30.donor.load_donor", return_value=donor) as load, \
                 patch("panda_mature30.reporting.emit_evaluation") as emit:
                campaign.register_donor("G8_MATURE_U23")
                campaign.register_donor("G8_MATURE_U23")
            self.assertEqual(selection.call_count, 1)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(emit.call_count, 2)
            self.assertEqual(emit.call_args.args[2], evaluation)
            self.assertEqual(emit.call_args.kwargs["artifact"], artifact)
            self.assertEqual(emit.call_args.kwargs["selector_receipt"], receipt)
            state = campaign.state["donors"]["G8_MATURE_U23"]
            self.assertEqual(state["selected_step"], 50000)
            self.assertEqual(state["selector_sha256"], "c" * 64)
            self.assertEqual(state["source_server"], "s3")
            self.assertEqual(load.call_args.kwargs["server"], "s3")

    def test_s3_never_falls_back_without_completed_local_stage1(self):
        campaign = Campaign.__new__(Campaign)
        campaign.server = "s3"
        campaign.check = Mock()
        campaign.completed = Mock(return_value=None)
        with self.assertRaisesRegex(ValueError, "Complete local Stage1"):
            campaign.register_donor("G8_MATURE_U23")


if __name__ == "__main__":
    unittest.main()
