"""Offline whole-dataset selection, historical coexistence and writer guards."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import publication as P
from .reporting import LocalOutbox, build_record
from .test_reporting import arguments, record
from sheet_publish_v2.test_selection import record as old_record, evidence
from sheet_publish_v2.test_transport import Fake, snapshot
from sheet_publish_v2.transport import reconcile, TransportError


def persist(root, item, *, formal=True):
    if formal:
        evidence(root, item)
    LocalOutbox(root).enqueue(item)


def persist_legacy(root, item):
    root = Path(root)
    (root / "outbox").mkdir(parents=True, exist_ok=True)
    (root / "outbox" / (item["record_id"] + ".json")).write_text(P.canonical_json(item))
    evidence(root, item)


class PublicationTests(unittest.TestCase):
    def test_import_is_torch_numpy_google_free(self):
        script = ("import sys; import directshift48.publication; "
                  "assert not any(x in sys.modules for x in ('torch','numpy','google.auth'))")
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_full_precision_inclusive_thresholds(self):
        for server, lane in P.LANES.items():
            for h, e, expected in ((lane[3], lane[4], "PASS_ALL_STEPS"),
                                    (lane[3] - 1e-12, lane[4], "BEST_HQNR_FALLBACK"),
                                    (lane[3], lane[4] + 1e-12, "BEST_HQNR_FALLBACK")):
                with self.subTest(server=server, h=h, e=e):
                    value = P.project([record(server=server, h=h, e=e)], server)
                    self.assertFalse(value["conflicts"])
                    self.assertEqual(value["rows"][0][P.INDEX["Selector"]], expected)

    def test_fallback_stable_id_best_hqnr_not_lowest_ergas(self):
        low = record(step=1000, h=.950, e=2.01)
        high = record(step=2000, h=.951, e=2.2)
        old, new = P.project([low], "s1"), P.project([low, high], "s1")
        self.assertEqual(new["selected"][0]["step"], 2000)
        self.assertEqual(old["rows"][0][59], new["rows"][0][59])

    def test_pass_all_steps_replace_fallback_and_keep_previous_pass(self):
        values = [record(step=1000), record(step=2000, h=.96, e=2.04),
                  record(step=4000, h=.961, e=2.044), record(step=6000, h=.99, e=3.)]
        projected = P.project(values, "s1")
        self.assertEqual([c["step"] for c in projected["selected"]], [2000, 4000])

    def test_same_step_state_conflict_fails_closed(self):
        first, second = record(), record(selector="SECOND")
        second["identity"]["model_state_sha256"] = "f" * 64
        second["source"]["extra"]["identity"]["model_state_sha256"] = "f" * 64
        args = arguments(selector="SECOND", model_state_sha="f" * 64)
        args["extra"]["identity"]["model_state_sha256"] = "f" * 64
        second = build_record(**args)
        projected = P.project([first, second], "s1")
        self.assertTrue(projected["conflicts"])
        self.assertEqual(projected["rows"], [])

    def test_old_campaign_rejected_by_direct_selector(self):
        projected = P.project([old_record()], "s1")
        self.assertEqual(projected["rows"], [])
        self.assertTrue(projected["conflicts"])

    def test_full_rrfr_evidence_means_and_ids_checked(self):
        with tempfile.TemporaryDirectory() as root:
            item = record()
            persist(root, item)
            self.assertEqual(len(P.project_campaign(root, "s1")["rows"]), 1)
            path = next(Path(root).glob("runs/**/evaluation_*.json"))
            broken = json.loads(path.read_text())
            broken["rr"]["per_scene"][0]["metrics"]["ERGAS"] = 10.
            path.write_text(P.canonical_json(broken))
            value = P.project_campaign(root, "s1")
            self.assertTrue(value["conflicts"])
            self.assertEqual(value["rows"], [])

    def test_all_local_events_including_receipted_read_not_raw_uploaded(self):
        with tempfile.TemporaryDirectory() as root:
            item = record()
            persist(root, item)
            receipts = Path(root) / "receipts"
            (receipts / (item["record_id"] + ".json")).write_text('{"status":"READBACK_VERIFIED"}')
            args = arguments(step=2000, metrics={}, eval_scope="TRACE_100_UPDATE_SUMMARY")
            persist(root, build_record(**args), formal=False)
            value = P.project_campaign(root, "s1")
            self.assertEqual(value["raw_count"], 2)
            self.assertEqual(value["summary_count"], 1)
            self.assertEqual(value["receipt_file_count"], 1)
            self.assertEqual(len(value["rows"]), 1)

    def test_incomplete_local_tail_blocks_publication(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, record())
            with (Path(root) / "records.jsonl").open("ab") as stream:
                stream.write(b'{"interrupted":')
            value = P.project_campaign(root, "s1")
            self.assertTrue(value["conflicts"])
            self.assertIn("INCOMPLETE_LEDGER_TAIL_RETRY", str(value["conflicts"]))

    def test_historical_campaign_rows_preserved_from_authenticated_source(self):
        with tempfile.TemporaryDirectory() as directory:
            direct, old = Path(directory) / "new", Path(directory) / "old"
            persist(direct, record())
            persist_legacy(old, old_record())
            both = P.combined_projection(direct, "s1", [old])
            self.assertEqual(len(both["rows"]), 2)
            self.assertNotEqual(both["rows"][0][P.INDEX["Campaign_ID"]], P.CAMPAIGN_ID)
            self.assertEqual(both["rows"][1][P.INDEX["Campaign_ID"]], P.CAMPAIGN_ID)
            adapter = Fake(snapshot(both["rows"][:1], both["notes"][:1]))
            result = reconcile(adapter, both, Path(directory) / "state", apply=True)
            self.assertTrue(result["readback_verified"])
            self.assertEqual(adapter.data["rows"], both["rows"])

    def test_missing_history_never_erases_remote_old_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            direct, old = Path(directory) / "new", Path(directory) / "old"
            persist(direct, record())
            persist_legacy(old, old_record())
            complete = P.combined_projection(direct, "s1", [old])
            direct_only = P.combined_projection(direct, "s1", [])
            adapter = Fake(snapshot(complete["rows"][:1], complete["notes"][:1]))
            with self.assertRaisesRegex(TransportError, "foreign"):
                reconcile(adapter, direct_only, Path(directory) / "state", apply=True)
            self.assertFalse(adapter.writes)

    def test_duplicate_history_paths_and_foreign_roots_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            direct, old = Path(directory) / "new", Path(directory) / "old"
            persist(direct, record()); persist_legacy(old, old_record())
            with self.assertRaises(P.EvidenceError):
                P.combined_projection(direct, "s1", [old, old])
            with self.assertRaises(P.EvidenceError):
                P.combined_projection(direct, "s1", [direct])

    def test_competing_writer_detection_excludes_train_and_other_lane(self):
        with tempfile.TemporaryDirectory() as directory:
            commands = {11: ["python", "-m", "edgeres48", "publish", "--server", "s1"],
                        12: ["python", "-m", "sheet_publish_v2", "--server=s1", "--apply"],
                        13: ["python", "-m", "edgeres48", "run", "--server", "s1"],
                        14: ["python", "-m", "sheet_publish_v2", "--server", "s3", "--apply"],
                        15: ["python", "-m", "sheet_publish_v2", "--server", "s1", "--google-read-only"],
                        16: ["python", "-m", "directshift48", "publish", "--server", "s1", "--apply"]}
            for pid, argv in commands.items():
                path = Path(directory) / str(pid); path.mkdir()
                (path / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
            self.assertEqual([p["pid"] for p in P.competing_publishers("s1", proc_root=directory, own_pid=16)], [11, 12])

    def test_local_dry_run_no_google_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "current"
            persist(root, record())
            with patch.object(P, "ROOT", Path(directory)), patch.object(P, "publication_revision", return_value={}), \
                    patch("sheet_publish_v2.transport.GoogleSheets", side_effect=AssertionError("No network")):
                self.assertEqual(P.publish(root, "s1", history_roots=[]), 0)

    def test_apply_refuses_old_publisher_before_google_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "current"
            persist(root, record())
            with patch.object(P, "ROOT", Path(directory)), patch.object(P, "publication_revision", return_value={}), \
                    patch.object(P, "competing_publishers", return_value=[{"pid": 111}]), \
                    patch("sheet_publish_v2.transport.GoogleSheets", side_effect=AssertionError("No network")):
                self.assertEqual(P.publish(root, "s1", apply=True, history_roots=[]), 1)


if __name__ == "__main__":
    unittest.main()
