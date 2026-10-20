"""Running HQNR maximum, immutable evidence, and old-row migration."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_mature30 import best_publication as B, publication as P, reporting as R
from panda_mature30.best_publication.restart import matches_writer, stop_writer
from panda_mature30.best_publication.transport import reconcile_rows
from .test_reporting import config, persist, reused_receipt


class BestPublicationTests(unittest.TestCase):
    def test_best_unconditional_and_final_not_added_both_servers(self):
        for server in ("s1", "s3"):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as root:
                cfg = config(server)
                persist(root, cfg, 1000, h=.965, e=2.2)
                persist(root, cfg, 2000, h=.960, e=2.045)
                persist(root, cfg, 50000, h=.961, e=2.04)
                view = B.project(root, server)
                self.assertFalse(view["conflicts"], view["conflicts"])
                self.assertEqual([s["step"] for s in view["selected"]], [1000])
                row = view["rows"][0]
                self.assertEqual(row[R.INDEX["Selector"]], "BEST_HQNR")
                self.assertEqual(row[R.INDEX["HQNR↑"]], .965)
                self.assertEqual(row[R.INDEX["ERGAS↓"]], 2.2)
                self.assertEqual(view["summary_count"], 3)
                B.verify_owned(row, view)

    def test_not_fixed_at_5000_and_row_id_stable_with_all_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=5000, h=.956, e=2.2)
            old = B.project(root, "s1")
            persist(root, step=6000, h=.955, e=2.0)
            self.assertEqual(B.project(root, "s1")["selected"][0]["step"], 5000)
            record = persist(root, step=7000, h=.957, e=2.3)
            new = B.project(root, "s1")
            self.assertEqual(new["selected"][0]["step"], 7000)
            self.assertEqual(old["rows"][0][R.INDEX["Result_ID"]],
                             new["rows"][0][R.INDEX["Result_ID"]])
            self.assertEqual(new["rows"][0][3:15], record["row"][3:15])
            requests = B.row_requests({"rows": old["rows"], "notes": old["notes"]}, new, "s1")
            self.assertEqual(len(requests), 1)
            self.assertIn("updateCells", requests[0])

    def test_ties_ergas_then_earlier_step(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=1000, h=.95, e=2.2)
            persist(root, step=2000, h=.95, e=2.1)
            persist(root, step=5000, h=.95, e=2.1)
            self.assertEqual(B.project(root, "s1")["selected"][0]["step"], 2000)

    def test_best_can_be_final_but_still_one_row(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=1000, h=.95)
            persist(root, step=50000, h=.96)
            view = B.project(root, "s1")
            self.assertEqual(len(view["rows"]), 1)
            self.assertEqual(view["rows"][0][R.INDEX["Selection@Step"]], "HQNR 최고 · 50,000")

    def test_stage1_rre_row_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            receipt, ev = reused_receipt(Path(root) / "old")
            R.emit_evaluation(root, config(phase="donor"), ev, selector_receipt=receipt)
            old, new = P.project(root, "s1"), B.project(root, "s1")
            self.assertEqual(old["rows"], new["rows"])
            B.verify_owned(new["rows"][0], new)

    def test_migrate_legacy_rows_keep_foreign_and_clear_surplus_only(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root, step=1000, h=.965, e=2.2)
            persist(root, step=2000, h=.960, e=2.045)
            persist(root, step=50000, h=.961, e=2.04)
            old, new = P.project(root, "s1"), B.project(root, "s1")
            foreign = [""] * 64
            foreign[R.INDEX["Campaign_ID"]] = R.CAMPAIGN_ID
            foreign[R.INDEX["Server"]] = "s3"
            snap = {"rows": [foreign, *old["rows"]], "notes": ["foreign", *old["notes"]]}
            requests = B.row_requests(snap, new, "s1")
            self.assertEqual(len(requests), 2)
            self.assertEqual([r["updateCells"]["range"]["startRowIndex"] for r in requests], [6, 7])
            self.assertEqual(requests[-1]["updateCells"]["rows"], [{"values": [{}] * 64}])
            bad = deepcopy(snap)
            bad["rows"][1][R.INDEX["HQNR↑"]] = .999
            with self.assertRaisesRegex(ValueError, "metrics"):
                B.row_requests(bad, new, "s1")
            bad = deepcopy(snap)
            note = json.loads(bad["rows"][1][R.INDEX["Original description / notes"]])
            note["publication"]["policy"] = "UNKNOWN"
            bad["rows"][1][R.INDEX["Original description / notes"]] = R.canonical_json(note)
            with self.assertRaisesRegex(ValueError, "Unknown"):
                B.row_requests(bad, new, "s1")

    def test_new_best_rows_reject_tampering(self):
        with tempfile.TemporaryDirectory() as root:
            persist(root)
            view = B.project(root, "s1")
            for column in ("HQNR↑", "Selected_step", "Result_ID", "Seed"):
                bad = deepcopy(view["rows"][0])
                bad[R.INDEX[column]] = "tampered"
                with self.subTest(column=column), self.assertRaises(ValueError):
                    B.verify_owned(bad, view)

    def test_cost_attached_for_selected_step_not_latest_or_final(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config(policy="DOWN")
            persist(root, cfg, 5000, h=.95)
            persist(root, cfg, 20000, h=.97)
            persist(root, cfg, 50000, h=.96)
            def attach(view, root, **kwargs):
                self.assertEqual([s["step"] for s in view["selected"]], [20000])
                self.assertEqual(view["rows"][0][R.INDEX["Selected_step"]], 20000)
                view["rows"][0][R.INDEX["Infer(ms)"]] = 12.3
            with patch("panda_mature30.cost.attach_cost", side_effect=attach) as hooked:
                view = B.project(root, "s1")
            hooked.assert_called_once()
            self.assertEqual(view["rows"][0][R.INDEX["Infer(ms)"]], 12.3)

    def test_restart_match_never_targets_trainer_or_other_server(self):
        root = Path("/tmp/campaign")
        args = ["python", "-m", "panda_mature30.publication", "--root", str(root),
                "--server", "s1", "--apply", "--watch", "--poll-seconds", "60"]
        self.assertTrue(matches_writer(args, root, "s1"))
        self.assertFalse(matches_writer(args, root, "s3"))
        self.assertFalse(matches_writer(args, Path("/tmp/other"), "s1"))
        args[2] = "panda_mature30"
        self.assertFalse(matches_writer(args, root, "s1"))

    def test_python39_restart_fallback_still_verifies_process_identity(self):
        root = Path("/tmp/campaign")
        args = ["python", "-m", "panda_mature30.publication", "--root", str(root),
                "--server", "s1", "--apply", "--watch"]
        stat = "123 (python) " + " ".join(["0"] * 19 + ["42"])
        with patch("pathlib.Path.read_text", return_value=stat), \
             patch("pathlib.Path.read_bytes", return_value=("\0".join(args) + "\0").encode()), \
             patch("panda_mature30.best_publication.restart.hasattr", return_value=False), \
             patch("panda_mature30.best_publication.restart.os.kill") as kill:
            stop_writer(123, args, root, "s1")
            kill.assert_called_once()
            kill.reset_mock()
            with self.assertRaisesRegex(ValueError, "identity"):
                stop_writer(123, args, root, "s3")
            kill.assert_not_called()

    def test_transport_backup_migration_then_idempotent_readback_and_append(self):
        class Adapter:
            BASE = "https://example.invalid/"

            def __init__(self, snapshot):
                self.data = deepcopy(snapshot)
                self.calls = []

            def snapshot(self, destination):
                return deepcopy(self.data)

            def _request(self, method, url, **kwargs):
                self.calls.append(url)
                if url.endswith(":append"):
                    rows = kwargs["json"]["values"]
                    self.data["rows"].extend(rows)
                    self.data["notes"].extend([""] * len(rows))
                    return {"updates": {"updatedRange": "WV3-ablations!A9:BL9"}}
                for req in kwargs["json"]["requests"]:
                    update = req["updateCells"]
                    index = update["range"]["startRowIndex"] - 5
                    values = update["rows"][0]["values"]
                    self.data["rows"][index] = [
                        next(iter(c.get("userEnteredValue", {}).values()), "") for c in values]
                    self.data["notes"][index] = values[0].get("note", "")
                return {}

        with tempfile.TemporaryDirectory() as root:
            persist(root, step=5000, h=.956)
            persist(root, step=50000, h=.95)
            old, new = P.project(root, "s1"), B.project(root, "s1")
            foreign = [""] * 64
            foreign[0], foreign[R.INDEX["Server"]] = "other server", "s3"
            adapter = Adapter({"rows": [foreign, *old["rows"]],
                               "notes": ["foreign", *old["notes"]]})
            state = Path(root) / "transport"
            state.mkdir()
            with patch("panda_mature30.best_publication.transport._validate_snapshot"):
                receipt = reconcile_rows(adapter, new, state, "s1", apply=True)
                self.assertTrue(receipt["readback_verified"])
                self.assertEqual(adapter.data["rows"][0], foreign)
                self.assertEqual(adapter.data["rows"][2], [""] * 64)
                self.assertEqual(len(list(state.glob("cloud_backup_*.json"))), 1)
                writes = len(adapter.calls)
                reconcile_rows(adapter, new, state, "s1", apply=True)
                self.assertEqual(len(adapter.calls), writes)
                persist(root, config(policy="DOWN"), 1000, h=.95)
                newer = B.project(root, "s1")
                receipt = reconcile_rows(adapter, newer, state, "s1", apply=True)
                self.assertEqual(receipt["row_count"], 2)
                self.assertEqual(len(adapter.data["rows"]), 4)
                self.assertEqual(adapter.data["notes"][-1], newer["notes"][0])


if __name__ == "__main__":
    unittest.main()
