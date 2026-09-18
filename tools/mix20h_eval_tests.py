#!/usr/bin/env python
"""CPU-only M20 evaluation/Sheet contract tests. No dataset, GPU or network access."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
import re
import types
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import mix20h_postrun as post
from tools import qrecon24_select as select
from tools import qrecon24_postrun as oldpost

spec = importlib.util.spec_from_file_location("_test_mix20_upload", ROOT / "gspread/mix20h_upload.py")
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)
RUN = "PAKD50_QRC24_S1_G23_W104_D121_WV3_T0_S52001_FRESH50_v4"


def candidate(step, h=.9586, e=2.039):
    return dict(step=step, hqnr=h, ergas=e, scc=.988, psnr=38., sam=2.7, q8=.93, ssim=.98)


class ContractTests(unittest.TestCase):
    def test_ergas_first_exact_ties(self):
        first, second = candidate(2020), candidate(1010, h=.9602, e=2.070)
        self.assertEqual(select.rank([second, first], select.ORDER_V2)[0]["step"], 2020)
        self.assertEqual(select.rank([candidate(2020), candidate(1010)], select.ORDER_V2)[0]["step"], 1010)
        self.assertFalse(select.rr_pass(candidate(1010, e=2.040))["ergas"])

    def test_completion_v1_incomplete_noeligible(self):
        ident = {"grid": "hash", "selector": post.SELECTOR}
        value = dict(selector=post.SELECTOR, completion_identity=ident, n_candidates=50, target_status="no_eligible", n_eligible=0, target=None)
        self.assertTrue(post.target_complete(value, ident))
        self.assertFalse(post.target_complete(dict(value, selector=select.SELECTOR), ident))
        self.assertFalse(post.target_complete(dict(value, target_status="incomplete_official_rr"), ident))
        value.update(target_status="official", official=True, n_eligible=1, n_official_evaluated=1, n_unevaluated=0, target=candidate(1010))
        self.assertTrue(post.target_complete(value, ident))
        value["target"]["hqnr"] = .958499
        self.assertFalse(post.target_complete(value, ident))

    def test_official_rr_nonfinite_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "selection.json"
            invalid = dict(candidate(1010), scc=float("nan"))
            argv = ["select", RUN, "--official", "--selector", post.SELECTOR, "--out", str(out)]
            with patch.object(select, "ROOT", tmp), patch.object(select, "load_rows", return_value=[candidate(1010)]), \
                    patch.object(select, "evaluator_identity", return_value={}), \
                    patch.object(select, "official_rr", return_value=(invalid, "official")), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(select.main(), 0)
            value = post.read(out)
            self.assertEqual(value["target_status"], "incomplete_official_rr")
            self.assertIsNone(value["target"])
            self.assertIsNone(value["joint_pass"])

    def test_official_rr_exception_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "selection.json"
            argv = ["select", RUN, "--official", "--selector", post.SELECTOR, "--out", str(out)]
            with patch.object(select, "ROOT", tmp), patch.object(select, "load_rows", return_value=[candidate(1010)]), \
                    patch.object(select, "evaluator_identity", return_value={}), \
                    patch.object(select, "official_rr", side_effect=RuntimeError("missing")), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                select.main()
            self.assertEqual(post.read(out)["target_status"], "incomplete_official_rr")

    def test_legacy_backlog_never_selects_m20(self):
        self.assertFalse(oldpost.needs_official(RUN))

    def test_dynamic_headers_no_assumed_block(self):
        headers = ["Run", "NOA HQNR", "Target ERGAS↓", "Date", "Target selector", "Train(h)"]
        mapping = upload.label_map(headers)
        self.assertEqual(mapping["Target selector"], 5)
        self.assertEqual(mapping["Target ERGAS↓"], 3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            upload.label_map(["Run", "Run"])
        self.assertTrue(upload.same_cell("0.958612345678", .958612345678))
        self.assertFalse(upload.same_cell("0.9586", .958612345678))

    def test_identity_change_invalidates_exact_cache(self):
        value = dict(candidate(50000), step=50000, official_complete=True, eval_mode="A_ON", completion_identity={"hash": "old"})
        self.assertTrue(post.exact_complete(value, {"hash": "old"}))
        self.assertFalse(post.exact_complete(value, {"hash": "new"}))
        self.assertFalse(post.exact_complete(dict(value, eval_mode="NOA"), {"hash": "old"}))

    def test_deadline_cli_cannot_extend_or_replace_manifest(self):
        manifest = {"deadline_at_utc": "2026-09-18T20:00:00+00:00"}
        self.assertEqual(post.bound_deadline(manifest), manifest["deadline_at_utc"])
        self.assertEqual(post.bound_deadline(manifest, "2026-09-19T05:00:00+09:00"), manifest["deadline_at_utc"])
        for requested in ("2026-09-18T20:00:01Z", "2026-09-18T19:59:59Z", "2026-09-18T20:00:00"):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                post.bound_deadline(manifest, requested)
        with self.assertRaisesRegex(ValueError, "missing"):
            post.bound_deadline({}, "2026-09-18T20:00:00Z")

    def test_sheet_s3_label_dynamic_columns_readback_and_preservation(self):
        run = RUN.replace("_S1_", "_S3_").replace("S52001", "S52003")
        values = {"M20 server": "s3", "M20 run id": run, "Target checkpoint SHA256": "targetsha",
                  "Exact50K checkpoint SHA256": "exactsha", "Target ERGAS↓": 2.038123456789,
                  "Target HQNR(raw)↑": .958612345678}
        class Worksheet:
            id = upload.GIDS["s3"]
            col_count = 20
            row_count = 10
            cells = {2: ["", "", "NOA", "legacy", "target"],
                     3: ["Run", "Date", "NOA metric", "HQNR↑", "Target ERGAS↓"],
                     4: [run, "keep-date", .8, .95, "old-target"]}
            reads = []

            def row_values(self, row, **kwargs):
                if row == 4:
                    self.reads.append(kwargs)
                return self.cells.get(row, []).copy()

            def get(self, value):
                return [[run]]

            def batch_update(self, edits, **kwargs):
                for edit in edits:
                    match = re.fullmatch(r"([A-Z]+)(\d+)", edit["range"])
                    col = 0
                    for letter in match[1]:
                        col = col * 26 + ord(letter)-64
                    row = self.cells.setdefault(int(match[2]), [])
                    row += [""] * max(0, col-len(row))
                    row[col-1] = edit["values"][0][0]

        ws = Worksheet()
        requested = []
        def worksheet(name):
            requested.append(name)
            return ws
        def column(index):
            result = ""
            while index:
                index, digit = divmod(index-1, 26)
                result = chr(65+digit)+result
            return result
        gu = types.SimpleNamespace(resolve_server=lambda _: "s3(5090)", SHEET="mock", CRED="none",
                                   ORIGIN_ROW=2, sheet_name=lambda ds, srv: f"{ds}-{srv}",
                                   _a1=lambda row, col: f"{column(col)}{row}", _col=column)
        cat = types.SimpleNamespace(canonical_run_key=lambda cell, _: cell)
        api = types.SimpleNamespace(service_account=lambda **_: types.SimpleNamespace(open=lambda _: types.SimpleNamespace(worksheet=worksheet)))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "work_dir").mkdir()
            status_path = Path(tmp) / "work_dir" / run / "results" / post.STATUS_FILE
            post.write(status_path, dict(status="EVAL_COMPLETE_UPLOAD_PENDING", sheet_uploaded=False,
                       training_complete=True, raw_grid_complete=True, official_target_complete=True,
                       exact50k_complete=True, upload_error="prior network error"))
            with patch.object(upload, "ROOT", Path(tmp)), patch.object(upload, "row_values", return_value=values), \
                    patch.object(upload, "local_module", side_effect=lambda name: gu if name == "gspread_upload" else cat), \
                    patch.dict(sys.modules, {"gspread": api}), \
                    patch.object(post, "exact_fr", side_effect=AssertionError("upload must not infer")) as fr_mock, \
                    patch.object(select, "official_rr", side_effect=AssertionError("upload must not infer")) as rr_mock:
                receipt = upload.upload_run(run)
                second_receipt = upload.upload_run(run)
                fr_mock.assert_not_called()
                rr_mock.assert_not_called()
            status = post.read(status_path)
            self.assertTrue(status["sheet_uploaded"])
            self.assertEqual(status["status"], "COMPLETE")
            self.assertTrue(status["official_target_complete"])
            self.assertTrue(status["exact50k_complete"])
            self.assertNotIn("upload_error", status)
            self.assertEqual(receipt["rows"], second_receipt["rows"])
        self.assertTrue(receipt["readback_verified"])
        self.assertEqual(requested, ["WV3-s3(5090)"] * 2)
        self.assertEqual(ws.cells[4][1:4], ["keep-date", .8, .95])
        self.assertEqual(ws.cells[4][4], values["Target ERGAS↓"])
        self.assertEqual(ws.reads, [{"value_render_option": "UNFORMATTED_VALUE"}] * 2)

    def test_grid_full_binding_and_mutation(self):
        import yaml
        from pa.evalviews import PROTOCOL_ID
        with tempfile.TemporaryDirectory() as tmp:
            wd = Path(tmp) / RUN
            (wd / "meta").mkdir(parents=True)
            cfg = dict(num_iter=50000, test_full_feeder_args=dict(dataroot="test_wv3_mat20.h5"), test_reduced_feeder_args=dict(dataroot="test_wv3.h5"))
            (wd / "meta/config.yaml").write_text(yaml.safe_dump(cfg))
            (wd / "meta/finished_at.txt").write_text("done")
            post.write(wd / "last_meta.json", dict(step=50000))
            post.write(wd / "meta/mix20h_run_manifest.json", dict(campaign_id=post.CAMPAIGN, original_run_id=RUN))
            post.write(wd / "selector_state_raw.json", dict(protocol_id=PROTOCOL_ID, fr_h5_sha256="bytes", evaluator_hash="eval", n_scenes=20))
            with open(wd / "checkpoint_metrics.csv", "w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["step", "raw_original.hqnr"])
                writer.writeheader()
                for step in post.GRID:
                    writer.writerow({"step": step, "raw_original.hqnr": .9586})
                    post.write(wd / f"candidates/step-{step}/mix20h_identity.json", dict(run_id=RUN, step=step, checkpoint_sha256="bytes", config_sha256="bytes", fr_h5_sha256="bytes", evaluator_hash="eval", eval_mode="A_ON", raw_hqnr=.9586, precision="no"))
            with patch.object(select, "evaluator_identity", return_value={"version": "mock"}):
                ident, rows, *_ = post.validate_grid(wd, sha=lambda _: "bytes", evaluator_hash="eval")
                self.assertEqual(len(rows), 50)
                self.assertEqual(ident["eval_mode"], "A_ON")
                side = wd / "candidates/step-1010/mix20h_identity.json"
                value = post.read(side)
                post.write(side, dict(value, raw_hqnr=.9600))
                with self.assertRaisesRegex(ValueError, "raw_hqnr"):
                    post.validate_grid(wd, sha=lambda _: "bytes", evaluator_hash="eval")

    def test_upload_failure_does_not_invalidate_completed_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            wd = Path(tmp) / "work_dir" / RUN
            (wd / "meta").mkdir(parents=True)
            (wd / "meta/finished_at.txt").write_text("done")
            post.write(wd / "last_meta.json", dict(step=50000))
            identity = {"id": "test"}
            post.write(wd / "results" / post.TARGET_FILE, dict(selector=post.SELECTOR, completion_identity=identity, n_candidates=50, n_eligible=0, target_status="no_eligible", target=None))
            post.write(wd / "results/exact50k_official_AON.json", dict(candidate(50000), step=50000, eval_mode="A_ON", completion_identity=identity, official_complete=True))
            with patch.object(post, "ROOT", Path(tmp)), patch.object(post, "validate_grid", return_value=(identity, [], {}, {"deadline_at_utc": "2000-01-01T00:00:00+00:00"})), \
                    patch.object(post, "legacy_consistency", return_value={}), contextlib.redirect_stdout(io.StringIO()):
                # The absent upload module in this isolated root emulates an upload failure.
                self.assertEqual(post.process(RUN, "cpu", upload=True), 0)
            value = post.read(wd / "results" / post.STATUS_FILE)
            self.assertEqual(value["status"], "EVAL_COMPLETE_UPLOAD_PENDING")
            self.assertTrue(value["training_complete"])
            self.assertTrue(value["official_target_complete"])
            self.assertTrue(value["exact50k_complete"])
            self.assertFalse(value["sheet_uploaded"])

    def test_completed_postrun_after_deadline_never_reinfers(self):
        with tempfile.TemporaryDirectory() as tmp:
            wd = Path(tmp) / "work_dir" / RUN
            (wd / "meta").mkdir(parents=True)
            (wd / "meta/finished_at.txt").write_text("done")
            post.write(wd / "last_meta.json", dict(step=50000))
            identity = {"id": "test"}
            post.write(wd / "results" / post.TARGET_FILE, dict(selector=post.SELECTOR, completion_identity=identity,
                       n_candidates=50, n_eligible=0, target_status="no_eligible", target=None,
                       h_consistency=dict(abs_diff=0.)))
            post.write(wd / "results/exact50k_official_AON.json", dict(candidate(50000), step=50000, eval_mode="A_ON",
                       completion_identity=identity, official_complete=True))
            with patch.object(post, "ROOT", Path(tmp)), patch.object(post, "validate_grid", return_value=(identity, [], {}, {"deadline_at_utc": "2000-01-01T00:00:00+00:00"})), \
                    patch.object(post, "legacy_consistency", side_effect=AssertionError("must not infer")), \
                    patch.object(post, "exact_fr", side_effect=AssertionError("must not infer")), \
                    patch.object(select, "official_rr", side_effect=AssertionError("must not infer")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(post.process(RUN, "cpu"), 0)


if __name__ == "__main__":
    unittest.main()
