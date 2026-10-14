"""Local-only audit tests using authenticated fixed64 fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from directshift48.reporting import LocalOutbox, build_record
from directshift48.test_reporting import arguments, record
from sheet_publish_v2.test_selection import evidence, record as old_record, seal

from .audit import audit_campaign


def fixture(root, summary=None, *, omit_scene=None, summary_present=True, legacy=False, corrupt_scene=False):
    summary = summary or record()
    path = evidence(root, summary)
    evaluation = json.loads(path.read_text())
    records = [summary] if summary_present else []
    for split in ("rr", "fr"):
        for scene in evaluation[split]["per_scene"]:
            if scene["scene_id"] == omit_scene:
                continue
            scope = split.upper() + "_SCENE:" + scene["scene_id"]
            if legacy:
                item = deepcopy(summary)
                item["record_type"] = scope; item["identity"]["eval_scope"] = scope
                item["identity"]["selector"] = scope
                item["source"]["fields"]["Eval_scope"] = scope
                item["source"]["metrics"] = scene["metrics"]
                item["source"]["extra"]["summary_record_id"] = summary["record_id"]
                item = seal(item)
            else:
                args = arguments(server=summary["identity"]["server"], step=summary["identity"]["selected_step"],
                    selector=scope, eval_scope=scope, metrics=scene["metrics"])
                args["extra"]["summary_record_id"] = "wrong" if corrupt_scene else summary["record_id"]
                item = build_record(**args)
            records.append(item)
    root = Path(root); (root / "outbox").mkdir(parents=True, exist_ok=True)
    for item in records:
        (root / "outbox" / (item["record_id"] + ".json")).write_text(json.dumps(item))
    return path


class AuditTests(unittest.TestCase):
    def test_full_evidence_two_servers_read_only(self):
        for server in ("s1", "s3"):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as root:
                fixture(root, record(server=server))
                before = {str(p): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()}
                result = audit_campaign(root, server, history_roots=[])
                self.assertEqual(result["status"], "COMPLETE")
                self.assertEqual(result["counts"]["authenticated_evaluation_count"], 1)
                self.assertEqual(result["selection"]["selected_row_count"], 1)
                self.assertFalse(result["remote_sheet_verified"])
                self.assertEqual(before, {str(p): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()})

    def test_missing_summary_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root, summary_present=False)
            result = audit_campaign(root, "s1", [])
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["counts"]["missing_summaries_count"], 1)
            self.assertEqual(result["counts"]["summary_record_count"], 0)

    def test_missing_scene_and_lineage_reported(self):
        for kwargs in ({"omit_scene": "WV3:rr:3"}, {"corrupt_scene": True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as root:
                fixture(root, **kwargs)
                result = audit_campaign(root, "s1", [])
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertEqual(result["counts"]["incomplete_scene_events_count"], 1)

    def test_orphan_summary_detected(self):
        with tempfile.TemporaryDirectory() as root:
            path = fixture(root); path.unlink()
            result = audit_campaign(root, "s1", [])
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["counts"]["orphan_summaries_count"], 1)

    def test_corrupt_means_cannot_authenticate_summary(self):
        with tempfile.TemporaryDirectory() as root:
            path = fixture(root); value = json.loads(path.read_text())
            value["rr"]["per_scene"][0]["metrics"]["ERGAS"] = 9.
            path.write_text(json.dumps(value))
            result = audit_campaign(root, "s1", [])
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["counts"]["rejected_summaries_count"], 1)

    def test_best_selection_is_intentional_not_missing(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root, record(step=1000, h=.950))
            fixture(root, record(step=2000, h=.955))
            result = audit_campaign(root, "s1", [])
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["selection"]["selected_row_count"], 1)
            self.assertEqual(result["intentional_exclusions"]["valid_steps_not_selected"], 1)

    def test_history_preserved_and_deduplicated_ledger_outbox(self):
        with tempfile.TemporaryDirectory() as directory:
            current, old = Path(directory) / "new", Path(directory) / "old"
            fixture(current)
            fixture(old, old_record(), legacy=True)
            item = record(); LocalOutbox(current).enqueue(item)
            result = audit_campaign(current, "s1", [old])
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["counts"]["summary_record_count"], 2)
            self.assertEqual(result["selection"]["selected_row_count"], 2)

    def test_incomplete_tail_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root)
            (Path(root) / "records.jsonl").write_bytes(b'{"unfinished":')
            result = audit_campaign(root, "s1", [])
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertIn("INCOMPLETE_LEDGER_TAIL_RETRY", str(result))

    def test_preflight_intentionally_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root)
            args = arguments(step=300, eval_scope="PREFLIGHT_EVAL")
            LocalOutbox(root).enqueue(build_record(**args))
            result = audit_campaign(root, "s1", [])
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["intentional_exclusions"]["preflight_evaluations"], 1)


if __name__ == "__main__":
    unittest.main()
