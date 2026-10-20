import json
from pathlib import Path
import tempfile
import unittest

from panda_mature30 import plan
from panda_mature30.transition import completed


class TransitionPlanTests(unittest.TestCase):
    def test_queues_and_full_horizon(self):
        for server in ("s1", "s3"):
            groups = plan.groups(server)
            core = [c for g in groups for c, r in g["runs"] if not g["optional"]]
            self.assertEqual(core, ([] if server == "s1" else ["G8_MATURE_U23"])
                             + ["FULL", "HALF", "DOWN", "LOW"])
            pair = next(g for g in groups if g["name"] == "paired_repeat")
            self.assertEqual(pair["runs"], [("HALF", "R2"), ("DOWN", "R2")])
            for group in groups:
                for case, replica in group["runs"]:
                    cfg = plan.resolve_case(server, case, replica)
                    self.assertEqual(cfg["planned_horizon"], 50000)
                    self.assertEqual(cfg["upsampler"], "interp23")
                    self.assertEqual(cfg["frame"], "pan")
                    self.assertEqual(cfg["aligner_outputs"], 2)
                    self.assertEqual(cfg["stage2_spec"]["beta"], 0.)
                    self.assertEqual(cfg["stage2_spec"]["alpha"], .25)
                    self.assertEqual(cfg["stage2_spec"]["gamma"], .5)
                    self.assertIn("NOSOFT", cfg["run_id"])
                    self.assertIn("HARDONLY", cfg["campaign_id"])

    def test_requires_complete_current_case_and_same_final_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/"models").mkdir()
            state = {"run_id": "current", "step": 49000, "status": "RUNNING"}
            (path/"run_state.json").write_text(json.dumps(state))
            self.assertFalse(completed(path, "current"))
            state.update(step=50000, status="COMPLETED")
            (path/"run_state.json").write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                completed(path, "current")
            (path/"models/step_0050000.pt").write_bytes(b"synthetic retained weights")
            evaluation = {"same_state": True}
            (path/"evaluation_0050000.json").write_text(json.dumps(evaluation))
            sidecar = {"artifact": {"completed_update": 50000},
                       "config": {"run_id": "current"}, "evaluation": evaluation}
            (path/"models/step_0050000.json").write_text(json.dumps(sidecar))
            self.assertTrue(completed(path, "current"))
            with self.assertRaises(ValueError):
                completed(path, "other")


if __name__ == "__main__":
    unittest.main()
