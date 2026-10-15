"""Selection, full-population proof, historical ownership and no-write tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import publication as P, reporting as R
from .test_reporting import config, evaluation, persist


class PublicationTests(unittest.TestCase):
    def test_import_has_no_torch_or_google_side_effect(self):
        subprocess.run([sys.executable, "-c", "import sys; import donor30.publication; "
            "assert not any(x in sys.modules for x in ('torch', 'numpy', 'google.auth'))"], check=True)

    def test_donor_gate_does_not_replace_sheet_rule(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config()
            persist(root, cfg, evaluation(cfg, step=1000, h=.9591, e=2.1))
            persist(root, cfg, evaluation(cfg, step=2000, h=.961, e=2.4))
            out = P.project_campaign(root, "s1")
            self.assertEqual(out["conflicts"], [])
            self.assertEqual(out["selected"][0]["step"], 2000)
            self.assertEqual(out["rows"][0][P.INDEX["Selector"]], "BEST_HQNR_FALLBACK")

    def test_all_display_qualified_steps_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config()
            for step, h, e in ((1000, .95, 2.1), (2000, .96, 2.045), (3000, .961, 2.04)):
                persist(root, cfg, evaluation(cfg, step, h, e))
            result = P.project_campaign(root, "s1")
            self.assertEqual(result["conflicts"], [])
            self.assertEqual([c["step"] for c in result["selected"]], [2000, 3000])

    def test_gf2_lane_kept_separate_and_native_threshold(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config("s3")
            persist(root, cfg, evaluation(cfg, h=.965, e=.522))
            result = P.project_campaign(root, "s3")
            self.assertFalse(result["conflicts"])
            self.assertEqual(result["destination"]["sheet_title"], "GF2-ablations")
            self.assertEqual(result["rows"][0][P.INDEX["Selector"]], "PASS_ALL_STEPS")

    def test_mean_tamper_or_incomplete_populations_fail_closed(self):
        for corrupt in ("mean", "missing"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as root:
                persist(root)
                path = next(Path(root).glob("runs/**/evaluation_*.json"))
                ev = json.loads(path.read_text())
                if corrupt == "mean":
                    ev["fr"]["per_scene"][0]["metrics"]["HQNR"] = .1
                else:
                    ev["fr"]["per_scene"].pop()
                R.atomic_json(path, ev)
                result = P.project_campaign(root, "s1")
                self.assertTrue(result["conflicts"])
                self.assertEqual(result["rows"], [])

    def test_saved_file_verified_even_when_not_joint_qualified(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, ev = config(), evaluation()
            path = Path(root) / "runs" / cfg["run_id"] / "attempt_01" / "weights.pt"
            path.parent.mkdir(parents=True); path.write_bytes(b"TEST_CHECKPOINT")
            artifact = {"weights_saved": True, "checkpoint_path": str(path),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_state_sha256": ev["state_before"]["model_state_sha256"], "identity": ev["state_before"]}
            persist(root, cfg, ev, artifact)
            result = P.project_campaign(root, "s1")
            self.assertFalse(result["conflicts"])
            self.assertEqual(result["rows"][0][P.INDEX["Checkpoint_SHA"]], artifact["file_sha256"])
            path.write_bytes(b"TAMPER")
            result = P.project_campaign(root, "s1")
            self.assertTrue(result["conflicts"])
            self.assertEqual(result["rows"], [])

    def test_completed_status_updates_view_without_fake_new_metric(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config(); persist(root, cfg)
            R.emit_status(root, cfg, "COMPLETED", step=50000,
                          timestamp="2026-10-15T06:00:00+00:00")
            result = P.project_campaign(root, "s1")
            self.assertEqual(result["rows"][0][P.INDEX["Status"]], "COMPLETED")
            self.assertEqual(result["rows"][0][P.INDEX["Selected_step"]], 1000)

    def test_incomplete_ledger_and_missing_cost_prevent_write(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            with patch("experiment_cost.publication.lookup_profile", return_value=None):
                result = P.project(root, "s1", legacy_roots=[])
                self.assertTrue(result["conflicts"])
                self.assertEqual(result["cost"]["missing"][0]["reason"], "NO_VERIFIED_ARCHITECTURE_PROFILE")
                result = P.project(root, "s1", legacy_roots=[], require_cost=False)
                self.assertFalse(result["conflicts"])
            with (Path(root) / "records.jsonl").open("ab") as stream:
                stream.write(b'{"unfinished":')
            self.assertTrue(P.project_campaign(root, "s1")["conflicts"])

    def test_historical_direct_and_edgeres_are_preserved(self):
        from directshift48.test_publication import persist as direct_persist, persist_legacy
        from directshift48.test_reporting import record as direct_record
        from sheet_publish_v2.test_selection import record as edge_record
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new, old, edge = root / "new", root / "old", root / "edge"
            persist(new)
            direct_persist(old, direct_record())
            persist_legacy(edge, edge_record())
            with patch("experiment_cost.publication.lookup_profile", return_value=None):
                result = P.project(new, "s1", [edge, old], require_cost=False)
            self.assertFalse(result["conflicts"])
            self.assertEqual(len(result["rows"]), 3)
            self.assertEqual(len(result["owned_run_keys"]), 3)

    def test_competing_writer_is_observed_not_killed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, module in ((100, "donor30"), (200, "experiment_cost"), (300, "donor30")):
                target = root / str(pid); target.mkdir()
                argv = ["python", "-m", module, "publish", "--server", "s1", "--apply"]
                (target / "cmdline").write_bytes(b"\0".join(v.encode() for v in argv))
            found = P.competing_publishers("s1", proc_root=root, own_pid=300)
            self.assertEqual([p["pid"] for p in found], [100, 200])

    def test_cost_attachment_uses_student_and_preserves_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config(phase="stage2"); persist(root, cfg, evaluation(cfg))
            fields = {"Params(M)": 2., "FLOPs(G)": 60., "Infer(ms)": 8.,
                      "Mem(MB)": 250., "Cost_scope": "native RR256 FP32 A+S"}
            profile = {"profile_sha256": "a" * 64, "architecture_key": "b" * 64, "path": "/profiles/local.json",
                "measurement": {"schema_version": "EXPERIMENT_COST_V1", "status": "COMPLETE",
                    "default_sheet_split": "rr", "observations": {s: {"sheet_fields": fields} for s in ("rr", "fr")}}}
            timing = {"sheet_fields": {"Train(h)": .05}, "omitted": {}, "proof": {}}
            with patch("experiment_cost.publication.lookup_profile", return_value=profile) as lookup, \
                 patch("experiment_cost.timing.timing_for_config", return_value=timing):
                result = P.project(root, "s1", legacy_roots=[])
            self.assertFalse(result["conflicts"])
            self.assertEqual(lookup.call_args.args[0]["baseline_resolved"]["teacher_architecture"]["width"], 104)
            row = result["rows"][0]
            self.assertEqual(row[P.INDEX["HQNR↑"]], .959)
            self.assertEqual(row[P.INDEX["Params(M)"]], 2.)
            self.assertEqual(row[P.INDEX["Train(h)"]], .05)

    def test_cached_teacher_graph_relabel_keeps_original_profile_and_numbers(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config(phase="stage2"); persist(root, cfg, evaluation(cfg))
            fields = {"Params(M)": 2.3, "FLOPs(G)": 60.5, "Infer(ms)": 8.1,
                      "Mem(MB)": 250.2, "Cost_scope": "RR256 full native A+warp+Teacher direct-CNN"}
            profile = {"profile_sha256": "a" * 64, "architecture_key": "b" * 64, "path": "/verified/profile.json",
                "measurement": {"schema_version": "EXPERIMENT_COST_V1", "status": "COMPLETE",
                    "default_sheet_split": "rr", "observations": {s: {"sheet_fields": deepcopy(fields)} for s in ("rr", "fr")}}}
            original = deepcopy(profile)
            with patch("experiment_cost.publication.lookup_profile", return_value=profile), \
                 patch("experiment_cost.timing.timing_for_config", return_value={"sheet_fields": {}}):
                result = P.project(root, "s1", legacy_roots=[])
            self.assertFalse(result["conflicts"])
            self.assertEqual(profile, original)
            row = result["rows"][0]
            self.assertIn("A+warp+Student", row[P.INDEX["Cost_scope"]])
            self.assertNotIn("A+warp+Teacher", row[P.INDEX["Cost_scope"]])
            for name in ("Params(M)", "FLOPs(G)", "Infer(ms)", "Mem(MB)"):
                self.assertEqual(row[P.INDEX[name]], fields[name])
            attachment = json.loads(row[P.INDEX["Original description / notes"]])["cost_measurement"]
            self.assertEqual(attachment["profile_sha256"], original["profile_sha256"])
            provenance = attachment["deployment_view_provenance"]
            self.assertTrue(provenance["role_label_adapted"])
            self.assertTrue(provenance["numeric_measurements_unchanged"])
            self.assertEqual(provenance["original_profile_cost_scope"]["rr"], fields["Cost_scope"])

    def test_unknown_remote_row_cannot_be_overwritten(self):
        from sheet_publish_v2.test_transport import Fake
        from sheet_publish_v2.transport import reconcile, TransportError
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            with patch("experiment_cost.publication.lookup_profile", return_value=None):
                result = P.project(root, "s1", legacy_roots=[], require_cost=False)
            adapter = Fake()  # Another historical run absent from our raw evidence.
            with self.assertRaisesRegex(TransportError, "Unknown/foreign"):
                reconcile(adapter, result, Path(root) / "transport", apply=True)
            self.assertEqual(adapter.writes, [])

    def test_local_publish_never_instantiates_google(self):
        with tempfile.TemporaryDirectory() as root, \
             patch("sheet_publish_v2.transport.GoogleSheets", side_effect=AssertionError("network forbidden")), \
             patch("donor30.publication.competing_publishers", return_value=[]), \
             patch("experiment_cost.publication.lookup_profile", return_value=None):
            persist(root)
            self.assertEqual(P.publish(root, "s1", legacy_roots=[], require_cost=False), 0)

    def test_handoff_unknown_writer_never_sends_signal(self):
        from sheet_publish_v2.test_transport import snapshot
        with tempfile.TemporaryDirectory() as root:
            credential = Path(root) / "credential.json"
            credential.write_text("{}")
            projected = P.project_campaign(root, "s1")
            with patch("donor30.publication.project", return_value=projected), \
                 patch("sheet_publish_v2.transport.GoogleSheets") as google, \
                 patch("donor30.publication.competing_publishers", return_value=[{"pid": 123}]), \
                 patch("donor30.publication._writer_identity", side_effect=ValueError("foreign cwd")), \
                 patch("os.kill") as kill, patch("subprocess.Popen") as spawn:
                google.return_value.snapshot.return_value = snapshot(rows=[])
                with self.assertRaisesRegex(ValueError, "foreign cwd"):
                    P.start_publisher(root, "s1", credentials=credential)
                kill.assert_not_called()
                spawn.assert_not_called()

    def test_handoff_launches_only_publisher_after_read_only_preflight(self):
        from sheet_publish_v2.test_transport import snapshot
        with tempfile.TemporaryDirectory() as root:
            credential = Path(root) / "credential.json"; credential.write_text("{}")
            projected = P.project_campaign(root, "s1")
            with patch("donor30.publication.project", return_value=projected), \
                 patch("sheet_publish_v2.transport.GoogleSheets") as google, \
                 patch("donor30.publication.competing_publishers", return_value=[]), \
                 patch("os.kill") as kill, patch("subprocess.Popen") as spawn:
                google.return_value.snapshot.return_value = snapshot(rows=[])
                spawn.return_value.poll.return_value = None
                spawn.return_value.pid = 456
                value = P.start_publisher(root, "s1", credentials=credential)
                self.assertTrue(value["training_untouched"])
                self.assertTrue(value["cloud_readback_pending"])
                self.assertIn("publish", spawn.call_args.args[0])
                self.assertNotIn("run", spawn.call_args.args[0])
                kill.assert_not_called()

    def test_watch_survives_more_than_five_transient_failures(self):
        with tempfile.TemporaryDirectory() as root, \
             patch("donor30.publication.project", side_effect=ValueError("temporary source unavailable")) as project, \
             patch("time.sleep"), patch("sheet_publish_v2.transport.GoogleSheets") as google:
            result = P.publish(root, "s1", google_read_only=True, watch=True,
                               poll_seconds=10, audit_seconds=10, max_polls=6)
            self.assertEqual(result, 1)
            self.assertEqual(project.call_count, 6)
            google.assert_not_called()


if __name__ == "__main__":
    unittest.main()
