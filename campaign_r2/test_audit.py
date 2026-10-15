"""Synthetic evidence fixtures only: no GPU, datasets, Sheets or live queue."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from campaign_r2 import audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = "FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73102_BASE_FRESH50_v1"
        self.wd = self.root / "work_dir" / self.run

    def write(self, path, doc):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))

    def grid(self, eligible_indices=(2, 3, 6)):
        records = []
        source = {"files": {}, "git_release": "old"}
        for index, step in enumerate(audit.GRID):
            folder = self.wd / "candidates" / str(step)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "model.safetensors").write_bytes(str(step).encode())
            identity = dict(update=step, config_sha256="cfg", data_sha256="data", source_identity=source,
                            model_sha256=audit.sha(folder / "model.safetensors"))
            self.write(folder / "identity.json", identity)
            rr = dict(ergas=3 - index * .01, scc=.9, psnr=40., sam=2., q8=.95, ssim=.97)
            fr = dict(hqnr=.96 if index in eligible_indices else .95, d_lambda=.01, d_s=.02)
            rr.update(official_complete=True, n_scenes=20,
                      per_scene=[dict(ergas=rr["ergas"], band_relative_mse=[.01] * 8) for _ in range(20)])
            fr.update(n_scenes=20, per_scene=[dict(fr) for _ in range(20)])
            records.append(dict(update=step, checkpoint_identity=identity, rr=rr, fr=fr, val_ergas=3 - index * .005))
        grid = dict(campaign_id=audit.CAMPAIGN, run_id=self.run, complete=True,
                    expected_steps=list(audit.GRID), records=records, source_identity=source,
                    config_sha256="cfg", data_sha256="data")
        self.write(self.wd / "official/raw_grid.json", grid)
        self.selections(grid)
        return grid

    def selections(self, grid):
        records = grid["records"]
        for row in records:
            self.write(self.wd / "official" / ("candidate_" + str(row["update"]) + ".json"), row)
        eligible = [r for r in records if r["fr"]["hqnr"] >= .9585]
        chosen = dict(target_selection=min(eligible, key=lambda r: (r["rr"]["ergas"], -r["rr"]["scc"],
                           -r["rr"]["psnr"], r["update"])) if eligible else None,
                      raw_max=max(records, key=lambda r: (r["fr"]["hqnr"], -r["update"])),
                      exact50k=records[-1], rr_val_selected=min(records, key=lambda r: (r["val_ergas"], r["update"])),
                      e_min_diag=min(records, key=lambda r: (r["rr"]["ergas"], r["update"])))
        for name, row in chosen.items():
            doc = {k: grid[k] for k in ("campaign_id", "run_id", "config_sha256", "data_sha256", "source_identity")}
            doc.update(selection_id=audit.SELECTIONS[name], official_complete=True, normal_same_step_A_U=True)
            if row:
                doc.update(step=row["update"], checkpoint_sha256=row["checkpoint_identity"]["model_sha256"],
                           **{key: row[key] for key in ("rr", "fr", "val_ergas", "checkpoint_identity")})
            else:
                doc["selection"] = None
            if name == "target_selection":
                doc.update(n_eligible=len(eligible), n_evaluated=50,
                           target_status="official" if eligible else "no_eligible")
            self.write(self.wd / "official" / (name + ".json"), doc)

    def change(self, relative, change):
        path = self.wd / relative
        doc = audit.read(path)
        change(doc)
        self.write(path, doc)

    def test_fixed50_and_fragmented_eligible_intervals(self):
        self.grid()
        result = audit.selection_audit(self.root, self.run)
        self.assertEqual(result["n_eligible"], 3)
        self.assertEqual([x["length"] for x in result["eligible_intervals"]], [2, 1])
        self.assertEqual(result["longest_eligible_interval"], 2)
        self.assertEqual(result["selections"]["target_selection"]["step"], 7070)
        self.assertEqual(len(result["per_scene_candidates"]), 50)
        self.assertEqual(len(result["per_scene_selection_deltas"]["exact50k_minus_raw_max"]["fr"]), 20)

    def test_no_eligible_is_null_not_zero(self):
        self.grid(())
        result = audit.selection_audit(self.root, self.run)
        self.assertEqual(result["n_eligible"], 0)
        self.assertIsNone(result["selections"]["target_selection"]["rr"])
        self.assertEqual(result["eligible_intervals"], [])

    def test_no_eligible_metric_contamination_rejected(self):
        self.grid(())
        self.change("official/target_selection.json", lambda d: d.update(rr={"ergas": 0}))
        with self.assertRaisesRegex(ValueError, "blank"):
            audit.selection_audit(self.root, self.run)

    def test_eligible_threshold_is_exact_no_tolerance(self):
        grid = self.grid(())
        grid["records"][0]["fr"]["hqnr"] = .9585
        grid["records"][1]["fr"]["hqnr"] = .9585 - 1e-12
        self.write(self.wd / "official/raw_grid.json", grid)
        self.selections(grid)
        self.assertEqual(audit.selection_audit(self.root, self.run)["n_eligible"], 1)

    def test_duplicate_or_missing_grid_rejected(self):
        self.grid()
        self.change("official/raw_grid.json", lambda d: d["records"].pop())
        with self.assertRaisesRegex(ValueError, "fixed 50"):
            audit.selection_audit(self.root, self.run)

    def test_checkpoint_metric_mixing_rejected(self):
        self.grid()
        self.change("official/target_selection.json", lambda d: d["fr"].update(hqnr=.999))
        with self.assertRaisesRegex(ValueError, "mixed"):
            audit.selection_audit(self.root, self.run)

    def test_wrong_selector_step_rejected(self):
        self.grid()
        self.change("official/raw_max.json", lambda d: d.update(step=50000))
        with self.assertRaisesRegex(ValueError, "order"):
            audit.selection_audit(self.root, self.run)

    def test_selected_checkpoint_bytes_rejected(self):
        self.grid()
        (self.wd / "candidates/7070/model.safetensors").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "bytes"):
            audit.selection_audit(self.root, self.run)

    def test_nonselected_identity_mismatch_rejected(self):
        self.grid()
        self.change("candidates/2020/identity.json", lambda d: d.update(data_sha256="other"))
        with self.assertRaisesRegex(ValueError, "identity"):
            audit.selection_audit(self.root, self.run)

    def test_missing_scene_evidence_rejected(self):
        grid = self.grid()
        grid["records"][0]["fr"]["per_scene"].pop()
        self.write(self.wd / "official/raw_grid.json", grid)
        self.selections(grid)
        with self.assertRaisesRegex(ValueError, "per-scene"):
            audit.selection_audit(self.root, self.run)

    def test_nonfinite_metric_rejected(self):
        grid = self.grid()
        grid["records"][0]["rr"].update(ergas=float("nan"))
        self.write(self.wd / "official/raw_grid.json", grid)
        self.selections(grid)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            audit.selection_audit(self.root, self.run)

    def test_scc_psnr_tie_break(self):
        grid = self.grid((2, 3, 6))
        for index in (2, 3, 6):
            grid["records"][index]["rr"].update(ergas=2., scc=.99, psnr=40.)
        grid["records"][3]["rr"]["psnr"] = 41.
        self.write(self.wd / "official/raw_grid.json", grid)
        self.selections(grid)
        self.assertEqual(audit.selection_audit(self.root, self.run)["selections"]["target_selection"]["step"], 4040)

    def test_source_checks_file_bytes_and_git(self):
        path = self.root / "small.py"
        path.write_text("original")
        files = {"small.py": audit.sha(path)}
        source = dict(files=files, content_sha256=audit.object_sha(files), git_release="old")
        with patch("campaign_r2.audit.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "old\n"
            self.assertEqual(audit.verify_source(self.root, source)["status"], "VERIFIED")
            path.write_text("new")
            self.assertEqual(audit.verify_source(self.root, source)["status"], "MISMATCH")
            path.write_text("original")
            run.return_value.stdout = "new\n"
            self.assertEqual(audit.verify_source(self.root, source)["status"], "MISMATCH")

    def test_paths_cannot_escape_repository(self):
        with self.assertRaises(ValueError):
            audit.selection_audit(self.root, "../../outside")

    def ledger(self, rows):
        directory = self.root / "work_dir/_fh20r1/s1"
        self.write(directory / "campaign_budget.json", dict(campaign_id=audit.CAMPAIGN,
                   actual_start_authorized=True, started_at_utc="2026-09-19T00:00:00+00:00"))
        self.write(self.root / "evidence.json", {"original": True})
        base = dict(campaign_id=audit.CAMPAIGN, server_id="s1", evidence="evidence.json")
        (directory / "active_intervals.jsonl").write_text("\n".join(json.dumps(dict(base, **r)) for r in rows))
        return directory

    def test_ledger_union_deduplicates_and_excludes_waiting(self):
        directory = self.ledger([
            dict(id="a", kind="train", start_utc="2026-09-19T00:00:00+00:00", end_utc="2026-09-19T00:00:20+00:00", seconds=20),
            dict(id="b", kind="eval", start_utc="2026-09-19T00:00:10+00:00", end_utc="2026-09-19T00:00:30+00:00", seconds=20),
            dict(id="c", kind="waiting", start_utc="2026-09-19T00:00:30+00:00", end_utc="2026-09-19T01:00:00+00:00", seconds=3570,
                 evidence="resource inventory; excluded from effective time")])
        before = (directory / "active_intervals.jsonl").read_bytes()
        result = audit.ledger_audit(directory, {})
        self.assertEqual(result["effective_seconds"], 30)
        self.assertEqual(result["newly_credited_seconds"], 0)
        self.assertEqual(before, (directory / "active_intervals.jsonl").read_bytes())

    def test_prior_campaign_interval_rejected(self):
        directory = self.ledger([dict(id="a", kind="train", start_utc="2026-09-18T00:00:00+00:00",
                                     end_utc="2026-09-18T00:00:20+00:00", seconds=20)])
        self.assertEqual(audit.ledger_audit(directory, {})["status"], "MISMATCH")

    def test_partial_preflight_is_not_pass(self):
        method, frontend = audit.reuse_preflight(self.root, "s1", {"source": {"status": "VERIFIED"}})
        self.assertEqual(method["status"], "TO_VERIFY")
        self.assertEqual(frontend["status"], "TO_VERIFY")

    def test_missing_branch_evidence_is_inconclusive_not_no_support(self):
        review = audit.branch_review(self.root, "s1")
        self.assertEqual(review["classification"], "INCONCLUSIVE")
        self.assertEqual(review["status"], "TO_VERIFY")

    def reconcile_fixture(self, server, branch, state=None):
        directory = self.root / "work_dir/_fh20r1" / server
        self.write(directory / "status.json", state or {"status": "REGISTERED"})
        self.write(directory / "campaign_budget.json", {"campaign_id": audit.CAMPAIGN})
        self.write(directory / "branch_record.json", branch)
        with patch.object(audit, "verify_source", return_value={"status": "VERIFIED", "errors": []}), \
             patch.object(audit, "ledger_audit", return_value={"status": "VERIFIED_EXISTING_LEDGER", "errors": []}):
            return audit.reconcile(self.root, server)

    def test_s2_s5_original_branches_without_lock_flag_are_valid(self):
        from campaign_r2.plan import EXPECTED_BRANCHES
        for server in ("s2", "s3", "s4", "s5"):
            with self.subTest(server=server):
                result = self.reconcile_fixture(server, {"condition": EXPECTED_BRANCHES[server],
                    "campaign_id": audit.CAMPAIGN, "server_id": server})
                self.assertEqual(result["status"], "VERIFIED_ARTIFACTS", result["errors"])

    def test_wrong_or_conflicting_branch_is_rejected(self):
        for branch in ({"condition": "S2_N2PL_REFERENCE_READY"},
                       {"condition": "S2_DONOR_OR_REFERENCE_UNAVAILABLE", "branch": "different"},
                       {"condition": "S2_DONOR_OR_REFERENCE_UNAVAILABLE", "server_id": "s3"}):
            with self.subTest(branch=branch):
                result = self.reconcile_fixture("s2", branch)
                self.assertEqual(result["status"], "HOLD_NEW_ADMISSION")

    def test_done_reused_without_proof_cannot_skip_training(self):
        result = self.reconcile_fixture("s1", {"branch": "S1_NO_A_SUPPORT_OR_INCONCLUSIVE"},
                    {"runs": {self.run: {"status": "DONE_REUSED"}}})
        self.assertEqual(result["status"], "HOLD_NEW_ADMISSION")
        self.assertEqual(result["completed_official_run_ids"], [])

    def test_reads_actual_applied_priority_receipt(self):
        path = self.root / "work_dir/_fh20r1/s4/priority_r2/applied_readback.json"
        self.write(path, {"priority_revision": audit.REVISION})
        result = self.reconcile_fixture("s4", {"condition": "STANDARD"})
        self.assertEqual(result["current_priority_receipt"]["priority_revision"], audit.REVISION)

    def test_same_seed_pair_not_cross_server_pool(self):
        self.grid()
        report = audit.selection_audit(self.root, self.run)
        rows, groups = audit.block_summaries(self.root, "s1", {self.run: report}, "S1_NO_A_SUPPORT_OR_INCONCLUSIVE")
        block = next(g for g in groups if g["block_id"] == "FH20R1_S1_B02")
        self.assertEqual(block["completed_run_count"], 1)
        self.assertEqual(block["eligible_run_count"], 1)
        self.assertFalse(block["block_complete"])
        self.assertEqual(block["same_teacher_same_seed_differences"], [])
        self.assertEqual(sum(r["completed"] for r in rows), 1)

    def test_failed_completed_asset_blocks_new_admission_not_active_run(self):
        reconciliation = dict(status="VERIFIED_ARTIFACTS", errors=[], source={"status": "VERIFIED"},
             branch={"branch": "S1_NO_A_SUPPORT_OR_INCONCLUSIVE"}, completed_official_run_ids=[self.run],
             upload_pending_run_ids=[], campaign_time={"status": "VERIFIED_EXISTING_LEDGER"}, terminate_active_training=False)
        with patch.object(audit, "reconcile", return_value=reconciliation), \
             patch.object(audit, "reuse_preflight", return_value=({"status": "REUSED_VERIFIED"}, {"status": "PARTIAL_REUSED"})), \
             patch("fh20r1.upload.row_values", side_effect=ValueError("missing selected checkpoint")):
            result = audit.collect(self.root, "s1")
        self.assertFalse(result["admission_integrity_ok"])
        self.assertEqual(result["checks"]["V00"]["status"], "HOLD_NEW_ADMISSION")
        self.assertFalse(result["checks"]["V00"]["terminate_active_training"])
        self.assertFalse((self.root / "work_dir/_fh20r1/s1/priority_r2").exists())


if __name__ == "__main__":
    unittest.main()
