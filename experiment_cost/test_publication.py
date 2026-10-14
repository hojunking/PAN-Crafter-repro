"""Offline sidecar-to-view binding, unchanged selection and exclusive writers."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from directshift48.test_publication import persist, persist_legacy
from directshift48.test_reporting import record
from sheet_publish_v2.test_selection import record as legacy_record
from sheet_publish_v2.test_transport import Fake, snapshot
from sheet_publish_v2.transport import reconcile, TransportError
from . import publication as P


def profile(*, split="rr", params=2.764346):
    def fields(ms, flops, memory, hw):
        return {"Params(M)": params, "FLOPs(G)": flops, "Infer(ms)": ms,
                "Mem(MB)": memory,
                "Cost_scope": f"batch1 FP32 PAN{hw} resident whole-pipeline; MAC-as-FLOP; fresh architecture"}
    return {"profile_sha256": "a" * 64, "architecture_key": "b" * 64,
            "path": "/profiles/verified.json", "measurement": {
                "schema_version": "EXPERIMENT_COST_V1", "status": "COMPLETE",
                "default_sheet_split": split, "weight_basis": "FRESH_ARCHITECTURE_NOT_CHECKPOINT",
                "observations": {"rr": {"sheet_fields": fields(12., 30., 200., 256)},
                                 "fr": {"sheet_fields": fields(50., 120., 700., 512)}}}}


class CostPublicationTests(unittest.TestCase):
    def test_import_has_no_torch_numpy_google(self):
        code = "import sys; import experiment_cost.publication; assert not any(k in sys.modules for k in ('torch','numpy','google.auth'))"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_cost_added_without_touching_metric_selection_or_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "lane"
            persist(root, record(step=1000, h=.950, e=2.01))
            persist(root, record(step=2000, h=.951, e=2.2))
            before = P.base.combined_projection(root, "s1", [])
            ledger = (root / "records.jsonl").read_bytes()
            with patch.object(P, "lookup_profile", return_value=profile()):
                after = P.enriched_projection(root, "s1", history_roots=[])
            self.assertFalse(after["conflicts"])
            self.assertEqual(before["selected"], after["selected"])
            self.assertEqual(after["selected"][0]["step"], 2000)
            self.assertEqual(before["owned_run_keys"], after["owned_run_keys"])
            for index, (old, new) in enumerate(zip(before["rows"][0], after["rows"][0])):
                if index not in [P.INDEX[c] for c in (*P.COST_COLUMNS, "Original description / notes")]:
                    self.assertEqual(old, new)
            self.assertEqual(after["rows"][0][P.INDEX["Infer(ms)"]], 12.)
            self.assertEqual(after["rows"][0][P.INDEX["Mem(MB)"]], 200.)
            self.assertEqual((root / "records.jsonl").read_bytes(), ledger)
            note = json.loads(after["rows"][0][P.INDEX["Original description / notes"]])["cost_measurement"]
            self.assertEqual(note["default_sheet_split"], "rr")
            self.assertEqual(note["all_domains"]["fr"]["Infer(ms)"], 50.)
            self.assertTrue(note["same_graph_not_same_weights"])

    def test_all_passed_steps_receive_cost_and_keep_stable_view_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            persist(directory, record(step=1000, h=.96, e=2.04))
            persist(directory, record(step=2000, h=.961, e=2.04))
            before = P.base.combined_projection(directory, "s1", [])
            with patch.object(P, "lookup_profile", return_value=profile()):
                after = P.enriched_projection(directory, "s1", history_roots=[])
            self.assertEqual([r[59] for r in before["rows"]], [r[59] for r in after["rows"]])
            self.assertEqual(after["cost"]["bound_count"], 2)

    def test_history_and_current_are_separately_bound_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            current, history = Path(directory) / "current", Path(directory) / "history"
            persist(current, record()); persist_legacy(history, legacy_record())
            before = P.base.combined_projection(current, "s1", [history])
            seen = []
            def lookup(cfg, **kwargs):
                seen.append((cfg["campaign_id"], kwargs["server"]))
                return profile(params=3. if cfg["campaign_id"] == P.base.CAMPAIGN_ID else 4.)
            with patch.object(P, "lookup_profile", side_effect=lookup):
                after = P.enriched_projection(current, "s1", history_roots=[history])
            self.assertFalse(after["conflicts"])
            self.assertEqual(len(seen), 2)
            self.assertEqual([r[P.INDEX["Params(M)"]] for r in after["rows"]], [4., 3.])
            adapter = Fake(snapshot(before["rows"], before["notes"]))
            result = reconcile(adapter, after, Path(directory) / "state", apply=True)
            self.assertTrue(result["readback_verified"])
            self.assertEqual(adapter.data["rows"], after["rows"])

    def test_missing_cost_does_not_silently_erase_existing_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            persist(directory, record())
            with patch.object(P, "lookup_profile", return_value=profile()):
                complete = P.enriched_projection(directory, "s1", history_roots=[])
            with patch.object(P, "lookup_profile", return_value=None):
                missing = P.enriched_projection(directory, "s1", history_roots=[])
                audit = P.enriched_projection(directory, "s1", history_roots=[], require_cost=False)
            self.assertTrue(missing["conflicts"])
            self.assertFalse(audit["conflicts"])
            self.assertEqual(audit["cost"]["missing"][0]["reason"], "NO_VERIFIED_ARCHITECTURE_PROFILE")
            adapter = Fake(snapshot(complete["rows"], complete["notes"]))
            with self.assertRaises(TransportError):
                reconcile(adapter, missing, Path(directory) / "state", apply=True)
            self.assertFalse(adapter.writes)

    def test_invalid_profile_cannot_be_uploaded(self):
        bad = [profile() for _ in range(5)]
        bad[0]["measurement"]["status"] = "FAILED"
        bad[1]["measurement"]["observations"]["fr"]["sheet_fields"]["Mem(MB)"] = None
        bad[2]["measurement"]["observations"]["rr"]["sheet_fields"]["Infer(ms)"] = float("nan")
        bad[3]["architecture_key"] = "forged"
        bad[4]["measurement"]["observations"]["rr"]["sheet_fields"]["Cost_scope"] = ""
        with tempfile.TemporaryDirectory() as directory:
            persist(directory, record())
            for value in bad:
                with self.subTest(value=value), patch.object(P, "lookup_profile", return_value=value):
                    self.assertTrue(P.enriched_projection(directory, "s1", history_roots=[])["conflicts"])

    def test_loader_rejects_architecture_or_environment_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            persist(directory, record())
            with patch.object(P, "lookup_profile", side_effect=ValueError("ARCHITECTURE_OR_SERVER_MISMATCH")):
                result = P.enriched_projection(directory, "s1", history_roots=[])
            self.assertIn("ARCHITECTURE_OR_SERVER_MISMATCH", str(result["conflicts"]))

    def test_s3_passes_exact_lane_to_profile_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            persist(directory, record(server="s3", h=.97, e=.5))
            with patch.object(P, "lookup_profile", return_value=profile()) as lookup:
                result = P.enriched_projection(directory, "s3", history_roots=[])
            self.assertFalse(result["conflicts"])
            self.assertEqual(lookup.call_args.kwargs["server"], "s3")
            self.assertEqual(lookup.call_args.args[0]["dataset"], "GF2")

    def test_competing_writer_guard_catches_existing_and_new_only_same_lane(self):
        commands = {
            11: ["python", "-m", "directshift48", "publish", "--server", "s1", "--apply"],
            12: ["python", "-m", "experiment_cost", "publish", "--server=s1", "--apply"],
            13: ["python", "-m", "experiment_cost", "publish", "--server", "s3", "--apply"],
            14: ["python", "-m", "experiment_cost", "publish", "--server", "s1"],
            15: ["python", "-m", "directshift48", "run", "--server", "s1"],
            16: ["python", "-m", "experiment_cost", "measure", "--server", "s1"],
        }
        with tempfile.TemporaryDirectory() as directory:
            for pid, argv in commands.items():
                path = Path(directory) / str(pid); path.mkdir()
                (path / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
            self.assertEqual([p["pid"] for p in P.competing_publishers("s1", proc_root=directory)], [11, 12])

    def test_local_projection_has_no_cloud_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "current"; persist(root, record())
            with patch.object(P, "ROOT", Path(directory)), patch.object(P, "publication_revision", return_value={}), \
                    patch.object(P, "lookup_profile", return_value=profile()), \
                    patch("sheet_publish_v2.transport.GoogleSheets", side_effect=AssertionError("No cloud")):
                self.assertEqual(P.publish(root, "s1", history_roots=[]), 0)

    def test_apply_refuses_live_writer_before_cloud_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "current"; persist(root, record())
            with patch.object(P, "ROOT", Path(directory)), patch.object(P, "publication_revision", return_value={}), \
                    patch.object(P, "lookup_profile", return_value=profile()), \
                    patch.object(P, "competing_publishers", return_value=[{"pid": 123}]), \
                    patch("sheet_publish_v2.transport.GoogleSheets", side_effect=AssertionError("No cloud")):
                self.assertEqual(P.publish(root, "s1", apply=True, history_roots=[]), 1)


if __name__ == "__main__":
    unittest.main()
