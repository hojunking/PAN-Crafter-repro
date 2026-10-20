"""Cost is deployment evidence, not training-graph timing or copied GPU data."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from panda_mature30 import cost as C, plan, reporting as R, publication as P
from experiment_cost.test_store import measurement


def fixture(cfg, step):
    value = measurement(cfg["server"])
    value["gpu_isolation_assertion"].update(
        mode="STANDALONE_ISOLATED_NO_TRAINER_GPU", training_state_retained_in_memory=False)
    value["deployment_signature_sha256"] = R.digest(C.signature(cfg, step))
    for split in ("rr", "fr"):
        value["observations"][split]["full_pipeline"] = {"samples_ms": [10.] * 10}
    return value


class CostTests(unittest.TestCase):
    def test_signature_covers_stage_server_and_actual_schedule_phase_not_seed(self):
        a = plan.resolve_case("s1", "HALF")
        b = plan.resolve_case("s1", "DOWN")
        self.assertEqual(C.signature(a, 1000), C.signature(b, 10000))
        self.assertNotEqual(C.signature(a, 1000), C.signature(b, 10001))
        self.assertEqual(C.signature(plan.resolve_case("s1", "LOW"), 1), C.signature(b, 30001))
        self.assertEqual(C.signature(a, 1), C.signature(plan.resolve_case("s1", "HALF", "R2"), 50000))
        self.assertNotEqual(C.signature(a, 1), C.signature(plan.resolve_case("s3", "HALF"), 1))
        self.assertNotEqual(C.signature(plan.resolve_case("s1", "G8_MATURE_U23"), 1),
                            C.signature(plan.resolve_case("s1", "FULL"), 1))

    def test_profile_integrity_and_no_cpu_or_concurrent_measurement(self):
        cfg = plan.resolve_case("s1", "FULL")
        with tempfile.TemporaryDirectory() as root:
            saved = C.save_profile(cfg, 1000, root, fixture(cfg, 1000))
            self.assertEqual(saved, C.find_profile(cfg, 4000, root))
            path = Path(saved["path"])
            bad = json.loads(path.read_text())
            bad["measurement"]["observations"]["rr"]["sheet_fields"]["Infer(ms)"] = 999.
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "integrity"):
                C.find_profile(cfg, 1000, root)
        for field, value in (("environment", {"device": "cpu"}),
                             ("optimizer_updates", 1),
                             ("gpu_isolation_assertion", {"other_compute_pids": [123]})):
            bad = fixture(cfg, 1000)
            bad[field] = value
            with self.assertRaises(ValueError):
                C.validate_measurement(bad, C.signature(cfg, 1000))

    def test_timing_attaches_even_without_gpu_profile_and_metrics_unchanged(self):
        cfg = plan.resolve_case("s1", "FULL")
        row = R.serialize_row({"Experiment": "test", "Selected_step": 1000,
                              "HQNR↑": .95, "Result_ID": "test",
                              "Original description / notes": R.canonical_json({
                                  "extra_parameters": {"configuration": cfg}})})
        view = {"rows": [row], "notes": ["{}"], "conflicts": []}
        with tempfile.TemporaryDirectory() as root, patch.object(C, "observed_timing",
                return_value={"sheet_fields": {"Train(h)": 1., "Eval(h)": .1, "Wall(h)": 1.2},
                              "omitted": {}}):
            C.attach_cost(view, root)
            self.assertFalse(view["conflicts"])
            self.assertEqual(view["rows"][0][R.INDEX["Train(h)"]], 1.)
            self.assertEqual(view["rows"][0][R.INDEX["Infer(ms)"]], "")
            C.save_profile(cfg, 1000, root, fixture(cfg, 1000))
            C.attach_cost(view, root, require_cost=True)
            self.assertFalse(view["conflicts"])
            self.assertEqual(view["cost"]["bound_count"], 1)
            self.assertEqual(view["rows"][0][R.INDEX["HQNR↑"]], .95)
            for k in ("Params(M)", "FLOPs(G)", "Infer(ms)", "Mem(MB)", "Train(h)", "Eval(h)", "Wall(h)"):
                self.assertGreater(view["rows"][0][R.INDEX[k]], 0.)

    def test_wall_is_observed_interval_not_train_plus_eval(self):
        cfg = plan.resolve_case("s1", "FULL")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "runs" / cfg["run_id"] / "attempt_01"
            directory.mkdir(parents=True)
            (directory / "run_state.json").write_text(json.dumps({
                "run_id": cfg["run_id"], "attempt": 1, "started_at": "2026-10-20T00:00:00+00:00"}))
            (directory / "evaluation_0001000.json").write_text(json.dumps({
                "state_before": {"run_id": cfg["run_id"], "attempt": 1, "completed_update": 1000},
                "recorded_at": "2026-10-20T02:00:00+00:00"}))
            timing = {"sheet_fields": {"Train(h)": 1., "Eval(h)": .1}, "proof": {}, "omitted": {}}
            with patch("experiment_cost.timing.timing_for_config", return_value=timing):
                got = C.observed_timing(cfg, 1000, root, {"extra_parameters": {}})
            self.assertEqual(got["sheet_fields"]["Wall(h)"], 2.)

    def test_reused_donor_timing_uses_original_run(self):
        from .test_reporting import reused_receipt, config
        with tempfile.TemporaryDirectory() as tmp:
            receipt, _ = reused_receipt(Path(tmp) / "old")
            note = {"extra_parameters": {"selector_receipt": receipt}}
            with patch("experiment_cost.timing.timing_for_config", return_value={
                    "sheet_fields": {"Train(h)": 2.}, "proof": {}, "omitted": {}}) as timing:
                got = C.observed_timing(config(phase="donor"), 50000, tmp, note)
            self.assertEqual(timing.call_args.args[0], receipt["selected"]["source_config"])
            self.assertTrue(got["reused_source_training_not_new_campaign_charge"])

    def test_inference_is_exact_native_student_only(self):
        import torch
        from panda_mature30.cost.profiling import build_deployment
        from panda_mature30.tests.test_schedule_core import batch
        from panda_mature30.method import MatureInference
        for case, step in (("FULL", 1000), ("DOWN", 10001), ("CAP085", 1000)):
            cfg = plan.resolve_case("s1", case)
            graph = build_deployment(cfg, None, step)
            direct = MatureInference(graph.aligner, graph.teacher, graph.inference.spec, completed_update=step)
            with torch.inference_mode():
                a, b = graph.native_prediction(batch()), direct(batch())
            torch.testing.assert_close(a["y"], b["y"], atol=0, rtol=0)
            self.assertEqual(sum(p.numel() for p in graph.parameters()),
                sum(p.numel() for p in graph.aligner.parameters()) +
                sum(p.numel() for p in graph.teacher.parameters()))
            self.assertFalse(hasattr(graph, "optimizer"))

    def test_host_hook_imports_without_torch(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import panda_mature30.cost; assert 'torch' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preparation_counts_against_original_campaign_budget(self):
        from panda_mature30.cost.__main__ import main
        cfg = plan.resolve_case("s1", "FULL")
        fake = {"architecture_key": "a" * 64, "signature": {"geometry_key": "s100"},
                "measurement": {"sheet_cost": {}}}
        real_read = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == plan.ROOT / "gspread/server.txt":
                return "s1"
            return real_read(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as root, \
                patch.object(Path, "read_text", read_text), \
                patch.object(sys, "argv", ["cost", "--server", "s1", "--root", root]), \
                patch("panda_method.edge_data.EdgeDataSources"), \
                patch("panda_stage2_condq.training.configure_precision"), \
                patch("panda_mature30.cost.profiling.registered_profiles", return_value=[(cfg, 1000)]), \
                patch("panda_mature30.cost.profiling.measure", return_value=fake) as measure:
            self.assertEqual(main(), 0)
            state_path = Path(root) / "campaign_state.json"
            original = state_path.read_bytes()
            self.assertEqual(main(), 0)
            self.assertEqual(state_path.read_bytes(), original)
            start = json.loads(original)["started_unix"]
            self.assertLessEqual(measure.call_args.kwargs["deadline_unix"], start + 30 * 3600 - 1800)


if __name__ == "__main__":
    unittest.main()
