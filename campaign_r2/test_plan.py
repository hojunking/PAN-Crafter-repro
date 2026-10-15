"""Read-only R2 registry and local-state precedence regression tests."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from campaign_r2 import plan
from fh12.common import ROOT, object_sha, sha256
from fh20r1 import plan as base


def branch(server, key="condition"):
    return {key: plan.EXPECTED_BRANCHES[server], "reason": "existing local evidence"}


def state():
    return dict(status="REGISTERED", blocks={}, runs={})


def block(server, number):
    bid = f"FH20R1_{server.upper()}_B{number:02d}"
    return next(b for b in base.blocks_for(server) if b.block_id == bid)


def enter(value, server, number, *, status="RUNNING", run_status="TRAINING"):
    b = block(server, number)
    order = list(b.order_for(plan.EXPECTED_BRANCHES[server]))
    value["blocks"][b.block_id] = dict(status=status, tier=b.tier, run_ids=order)
    if run_status:
        value["runs"][order[0]] = dict(status=run_status, training_started=True)
    return order


class PlanTests(unittest.TestCase):
    def test_exact_definition_and_all_configs(self):
        before = object_sha(base.registry_document())
        result = plan.validate_definition()
        self.assertEqual((result["definitions"], result["core_students"], result["reserve_students"],
                          result["new_training_ids"], result["numeric_configs_checked"]), (145, 84, 40, 0, 145))
        self.assertEqual(result["registry_sha256"], before)
        self.assertEqual(before, object_sha(base.registry_document()))
        self.assertTrue(all(len(row["file_sha256"]) == len(row["config_sha256"]) == 64
                            for row in result["config_hashes"].values()))

    def test_exact_all_server_priorities_and_original_objects(self):
        expected = {"s1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    "s2": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "s3": [1, 2, 6, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14],
                    "s4": [1, 2, 6, 3, 7, 4, 5, 8, 9],
                    "s5": [1, 2, 6, 3, 4, 5, 7, 8]}
        for server, numbers in expected.items():
            with self.subTest(server=server):
                actual = plan.pending_blocks(server, state(), branch(server))
                self.assertEqual([int(b.block_id[-2:]) for b in actual], numbers)
                self.assertTrue(all(any(b is original for original in base.blocks_for(server)) for b in actual))

    def test_overlay_provenance_does_not_invent_missing_companions(self):
        overlay = plan.canonical_overlay()
        self.assertIs(overlay["provenance"]["companion_priority_json_csv_supplied"], False)
        self.assertEqual(overlay["provenance"]["source_plan_sha256"], sha256(ROOT / plan.SOURCE_PLAN))
        self.assertIs(overlay["policy"]["reset_campaign_clock"], False)
        self.assertIs(overlay["policy"]["change_numeric_config"], False)

    def test_historical_branch_key_aliases_all_servers(self):
        for server in base.SERVERS:
            for key in ("branch", "condition"):
                with self.subTest(server=server, key=key):
                    self.assertEqual(plan.require_branch_record(server, branch(server, key)),
                                     plan.EXPECTED_BRANCHES[server])

    def test_missing_conflicting_or_changed_branch_fails(self):
        records = [None, {}, {"reason": "NO_SUPPORT"},
            {"branch": "STANDARD", "condition": "S1_NO_A_SUPPORT_OR_INCONCLUSIVE"},
            {"condition": "S1_A_SUPPORT"}, {"condition": "PRIMARY"},
            {"condition": "S1_NO_A_SUPPORT_OR_INCONCLUSIVE", "server_id": "s2"},
            {"condition": "S1_NO_A_SUPPORT_OR_INCONCLUSIVE", "campaign_id": "new_campaign"}]
        for record in records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                plan.require_branch_record("s1", record)

    def test_s2_donor_newly_available_cannot_upgrade_branch(self):
        with self.assertRaisesRegex(ValueError, "cannot change"):
            plan.require_branch_record("s2", {"condition": base.S2_SUPPORT})

    def test_active_b03_ahead_of_promoted_b06_and_new_blocks(self):
        local = state()
        order = enter(local, "s3", 3)
        before = copy.deepcopy(local)
        queue = plan.planned_queue("s3", local, branch("s3"))
        self.assertEqual(queue[0]["block_id"], "FH20R1_S3_B03")
        self.assertEqual(queue[0]["run_ids"], order)
        self.assertIs(queue[0]["admitted"], True)
        self.assertEqual(queue[0]["run_actions"][order[0]], "resume_or_postrun_existing")
        self.assertEqual(local, before)

    def test_all_admitted_blocks_precede_wholly_new_blocks(self):
        local = state()
        enter(local, "s4", 4, status="INCOMPLETE", run_status="PAUSED")
        enter(local, "s4", 3, status="WAITING_DISK", run_status="PAUSED")
        enter(local, "s4", 5)
        queue = plan.planned_queue("s4", local, branch("s4"))
        self.assertEqual([row["block_id"] for row in queue[:3]],
                         ["FH20R1_S4_B05", "FH20R1_S4_B03", "FH20R1_S4_B04"])
        self.assertTrue(all(row["admitted"] for row in queue[:3]))
        self.assertIs(queue[3]["admitted"], False)

    def test_admitted_reserve_preserved_above_minimum_hours(self):
        local = state()
        local["effective_hours"] = 21.0
        order = enter(local, "s5", 7)
        local["runs"][order[0]]["status"] = "DONE"
        local["runs"][order[1]] = {"status": "TRAINING"}
        queue = plan.planned_queue("s5", local, branch("s5"))
        self.assertEqual(queue[0]["block_id"], "FH20R1_S5_B07")
        self.assertEqual(queue[0]["tier"], "RESERVE")
        self.assertEqual(queue[0]["run_ids"], order)
        self.assertEqual(queue[0]["pending_run_ids"], order[1:])

    def test_completed_blocks_skipped_members_reused_not_fresh(self):
        local = state()
        done = enter(local, "s1", 1, status="DONE", run_status="DONE")
        local["runs"][done[1]] = {"status": "DONE_REUSED"}
        current = enter(local, "s1", 2, run_status="DONE_REUSED")
        queue = plan.planned_queue("s1", local, branch("s1"))
        self.assertEqual(queue[0]["block_id"], "FH20R1_S1_B02")
        self.assertEqual(queue[0]["completed_run_ids"], current[:1])
        self.assertEqual(queue[0]["pending_run_ids"], current[1:])
        self.assertEqual(queue[0]["run_actions"][current[0]], "reuse_verified_complete")
        self.assertEqual(queue[0]["run_actions"][current[1]], "verify_local_artifacts_before_fresh")
        self.assertFalse(any(row["block_id"] == "FH20R1_S1_B01" for row in queue))

    def test_advanced_local_state_does_not_revert_to_snapshot_b02(self):
        local = state()
        for number in (1, 2, 3, 4):
            enter(local, "s5", number, status="DONE", run_status=None)
        enter(local, "s5", 5)
        queue = plan.planned_queue("s5", local, branch("s5"))
        self.assertEqual([row["block_id"] for row in queue[:2]], ["FH20R1_S5_B05", "FH20R1_S5_B06"])

    def test_completed_members_pending_block_finalization_not_fresh(self):
        local = state()
        order = enter(local, "s3", 3, run_status="DONE")
        local["runs"][order[1]] = {"status": "DONE"}
        queue = plan.planned_queue("s3", local, branch("s3"))
        self.assertIs(queue[0]["admitted"], True)
        self.assertEqual(queue[0]["pending_run_ids"], [])
        self.assertEqual(queue[0]["completed_run_ids"], order)

    def test_original_internal_orders_all_servers(self):
        for server in base.SERVERS:
            with self.subTest(server=server):
                rows = plan.planned_queue(server, state(), branch(server))
                expected = {b.block_id: list(b.order_for(plan.EXPECTED_BRANCHES[server]))
                            for b in base.blocks_for(server)}
                self.assertTrue(all(row["run_ids"] == expected[row["block_id"]] for row in rows))

    def test_invalid_runtime_states_fail(self):
        for mutation in ("unknown_block", "unknown_run", "duplicate", "reverse", "wrong_tier",
                         "foreign_server", "foreign_campaign", "missing_admitted_order"):
            with self.subTest(mutation=mutation):
                local = state()
                order = enter(local, "s3", 3)
                row = local["blocks"]["FH20R1_S3_B03"]
                if mutation == "unknown_block":
                    local["blocks"]["FH20R1_S3_B99"] = {}
                elif mutation == "unknown_run":
                    local["runs"]["unregistered_run"] = {"status": "DONE"}
                elif mutation == "duplicate":
                    row["run_ids"] = [order[0], order[0]]
                elif mutation == "reverse":
                    row["run_ids"] = order[::-1]
                elif mutation == "wrong_tier":
                    row["tier"] = "RESERVE"
                elif mutation == "foreign_server":
                    local["server_id"] = "s1"
                elif mutation == "foreign_campaign":
                    local["campaign_id"] = "reset_20h"
                else:
                    row.pop("run_ids")
                with self.assertRaises(ValueError):
                    plan.planned_queue("s3", local, branch("s3"))

    def test_member_activity_without_block_requires_reconciliation(self):
        local = state()
        local["runs"][block("s3", 3).primary_order[0]] = {"status": "TRAINING"}
        with self.assertRaisesRegex(ValueError, "reconcile"):
            plan.planned_queue("s3", local, branch("s3"))

    def test_completed_block_cannot_hide_training_member(self):
        local = state()
        enter(local, "s1", 1, status="DONE", run_status="TRAINING")
        with self.assertRaisesRegex(ValueError, "unfinished"):
            plan.planned_queue("s1", local, branch("s1"))

    def test_unknown_server_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            plan.priority_blocks("s99")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            plan.pending_blocks("s99", state(), {})

    def test_changed_registry_fails(self):
        with patch.object(base, "registry_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "registry definition"):
                plan.validate_definition(require_configs=False)

    def test_source_md_change_requires_review(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / plan.SOURCE_PLAN
            target.parent.mkdir(parents=True)
            target.write_text((ROOT / plan.SOURCE_PLAN).read_text() + "\nChanged\n")
            with self.assertRaisesRegex(ValueError, "source MD"):
                plan.canonical_overlay(directory)

    def test_appendix_membership_checked_independently_of_doc_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / plan.SOURCE_PLAN
            target.parent.mkdir(parents=True)
            start, appendix = (ROOT / plan.SOURCE_PLAN).read_text().split("## 부록 B.", 1)
            original = "FH20R1_S1_S_PLH_W104_D121_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1"
            appendix = appendix.replace(original, original.replace("73101", "99999"), 1)
            target.write_text(start + "## 부록 B." + appendix)
            with patch.object(plan, "EXPECTED_PLAN_SHA256", sha256(target)):
                with self.assertRaisesRegex(ValueError, "Appendix B"):
                    plan.canonical_overlay(directory)

    def test_duplicate_preference_fails(self):
        with patch.dict(plan.CORE_NUMBERS, {"s3": (1, 2, 6, 3, 4, 5, 7, 8, 9, 9)}):
            with self.assertRaisesRegex(ValueError, "membership/duplication"):
                plan.priority_blocks("s3")

    def test_numeric_config_mismatch_fails(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / plan.SOURCE_PLAN
            target.parent.mkdir(parents=True)
            target.write_bytes((ROOT / plan.SOURCE_PLAN).read_bytes())
            configs = Path(directory) / "config"
            configs.mkdir()
            case = base.CASES[0]
            cfg = base.build_config(case)
            cfg["fh20r1"]["beta"] = 0.0
            (configs / f"{case.run_id}.yaml").write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "numeric config changed"):
                plan.validate_definition(directory)


if __name__ == "__main__":
    unittest.main()
