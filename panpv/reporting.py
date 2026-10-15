"""Append-only completed PANPV evaluations; never call a legacy uploader.

Only the observed practical-validation tab is writable. Native row metadata is
inspected before writing; values-only RAW writes retain its formatting and
validation. Column BK (manual Review) is never a write target. Training blocks
remain local; network/credential failures never fail the experiment.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
from pathlib import Path

from .common import atomic_json, canonical_sha, file_sha, load_json

SPREADSHEET_ID = "1_-3KY2DbSk_AOAuExf_ENbe5jAbhFRxzXoyfaDZRoK0"
SHEET_ID = 202610020
SHEET = "practical-validation"
HEADERS = ("Scene_ID|Method|Checkpoint / Selector|FR_HQNR↑|FR_Ds↓|FR_Dlambda↓|FR_JQM↑|"
           "RR_ERGAS↓|RR_SCC↑|RR_SAM_deg↓|RR_PSNR_dB↑|RR_SSIM↑|RR_Q8↑|RR_RMSE↓|RR_CC↑|"
           "Infer_s|Adapt_s|EndToEnd_s|PeakMem_MB|Params_M|Updates|Samples_seen|Effective_epochs|"
           "Fit_scene_ID|Eval_scene_ID|RR_scope|FR_scope|Input_adaptation|Seed|Repeat|Run_ID|"
           "Campaign_ID|Parent_run|Parent_checkpoint_SHA|Checkpoint_SHA|Server|Hardware|Precision|"
           "Train_patches|Val_patches|Test_patches|Train_L1|Val_L1|LR|Product_profile|Bands|"
           "Norm_profile|Degradation_profile|Pair_geometry_ID|Valid_mask_SHA|Status|Readback|"
           "Source_manifest_SHA|Tile_layout_SHA|Evaluator_SHA|Log_URL|Output_URL|Local_artifact_root|"
           "Result_ID|Recorded_at|FR_Reproj_SAM_deg↓|FR_Spatial_corr↑|Review|Notes").split("|")
IDX = {name: i for i, name in enumerate(HEADERS)}
METHODS = {"BROVEY8", "ZS_WV3", "SC_SCENE", "FT_WV3_SCENE"}
RR_KEYS = {"RR_ERGAS↓": "ergas", "RR_SCC↑": "scc", "RR_SAM_deg↓": "sam",
           "RR_PSNR_dB↑": "psnr", "RR_SSIM↑": "ssim", "RR_Q8↑": "q8",
           "RR_RMSE↓": "rmse", "RR_CC↑": "cc"}
FR_KEYS = {"FR_HQNR↑": "hqnr", "FR_Ds↓": "d_s", "FR_Dlambda↓": "d_lambda",
           "FR_Reproj_SAM_deg↓": "reprojection_sam", "FR_Spatial_corr↑": "pan_scc"}
READBACK_TOLERANCE = dict(relative=1e-12, absolute=1e-12,
                          scope="Google Sheets numeric transport only; stored source metrics unchanged")


def _owned(root, path):
    path = Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Reporting artifact escapes its campaign")
    return path


def _number(value):
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Nonfinite/non-numeric metric cannot be published")
    return value


def _sealed_report(report):
    if (report.get("schema") != "PANPV_PRIVATE_TILE_EVALUATION_v1"
            or report.get("status") != "COMPLETE" or report.get("full_scene") is not False
            or report.get("official_benchmark") is not False
            or report.get("payload_sha256") != canonical_sha({k: v for k, v in report.items() if k != "payload_sha256"})):
        raise ValueError("Unsealed, incomplete, or incorrectly scoped evaluation")


def build_row(root, registry, receipt):
    """Construct one immutable observation from a checkpoint-bound report."""
    root = Path(root).resolve()
    if registry.get("execution_server") != "s1" or registry.get("target_sensor") != "WV2":
        raise ValueError("Only the registered s1 WV2 practical campaign is writable")
    if receipt.get("state") != "COMPLETE":
        raise ValueError("Incomplete stage receipt")
    stage = receipt["stage"]
    if stage.get("kind") == "train":
        return None
    if stage.get("kind") not in ("baseline", "final") or stage.get("method") not in METHODS:
        raise ValueError("Unsupported completed evaluation stage")
    method, scene_id = stage["method"], stage["scene_id"]
    entry = next((s for s in registry["scenes"] if s["scene_id"] == scene_id), None)
    if entry is None:
        raise ValueError("Evaluation scene not registered")
    result, report = receipt["result"], receipt["result"]["result"]
    _sealed_report(report)
    identity = report["identity"]
    checkpoint = result.get("checkpoint_sha256")
    selector = result["selector"]
    step = result["completed_step"]
    allowed = {"FIXED"} if stage["kind"] == "baseline" else {"EXACT20K", "BEST_RRVAL_TO_20K"}
    if selector not in allowed or selector != stage.get("selector", "FIXED"):
        raise ValueError("Checkpoint selector/stage mismatch")
    if (identity.get("method") != method or identity.get("scene_id") != scene_id
            or identity.get("source_sha256") != entry["sha256"] or identity.get("final") is not True
            or identity.get("declared_checkpoint_sha256") != checkpoint):
        raise ValueError("Metric report is bound to a different method/scene/checkpoint")
    if method == "BROVEY8":
        if checkpoint is not None or not identity.get("brovey") or step != 0:
            raise ValueError("Brovey cannot claim a learned checkpoint")
    elif not isinstance(checkpoint, str) or len(checkpoint) != 64:
        raise ValueError("Full evaluated checkpoint SHA required")
    if method == "ZS_WV3" and (checkpoint != registry["donor"]["sha256"] or step != 0):
        raise ValueError("ZS must use the registered untouched donor")
    if selector == "EXACT20K" and step != 20000:
        raise ValueError("EXACT20K requires 20000 updates")
    if selector == "BEST_RRVAL_TO_20K" and (step <= 0 or step > 20000 or step % 2000):
        raise ValueError("BEST must be a preregistered validation checkpoint")
    report_path = _owned(root, report["report_path"])
    artifacts = {str(_owned(root, root / a["path"])): a["sha256"] for a in receipt["artifacts"]}
    if (str(report_path) not in artifacts or file_sha(report_path) != artifacts[str(report_path)]
            or load_json(report_path) != report):
        raise ValueError("Persisted evaluation report differs from its stage receipt")
    protocol_binding = registry["protocol"]
    protocol_path = _owned(root, protocol_binding["path"])
    if file_sha(protocol_path) != protocol_binding["sha256"]:
        raise ValueError("Frozen evaluation protocol changed")
    protocol = load_json(protocol_path)["scenes"][scene_id]
    if (identity["protocol_sha256"] != protocol["protocol_sha256"]
            or report["rr"]["protocol"] != protocol["rr_protocol"]
            or report["fr"]["protocol"] != protocol["fr_protocol"]
            or report["rr"]["samples"] != len(protocol["audit"])
            or report["fr"]["tiles"] != len(protocol["audit"])):
        raise ValueError("Evaluation protocol or support counts differ")
    if report["fr"]["metrics"].get("jqm") is not None:
        raise ValueError("Unvalidated practical JQM must remain blank")
    values = {key: "" for key in HEADERS}
    for header, key in RR_KEYS.items():
        values[header] = _number(report["rr"]["metrics"][key])
    for header, key in FR_KEYS.items():
        values[header] = _number(report["fr"]["metrics"][key])
    adaptive = method in ("SC_SCENE", "FT_WV3_SCENE")
    run_id = f"PV_S1_{scene_id}_{method}_R01_SE261002" if adaptive else stage["id"]
    parent_sha = registry["donor"]["sha256"] if method in ("ZS_WV3", "FT_WV3_SCENE") else ""
    key = dict(campaign=registry["campaign_id"], scene=scene_id, method=method, repeat=1,
               checkpoint_sha256=checkpoint, config=identity.get("brovey"), selector=selector,
               source_sha256=entry["sha256"], split_sha256=entry["patch_index_sha256"],
               protocol_sha256=protocol["protocol_sha256"])
    values.update({"Scene_ID": scene_id, "Method": method, "Checkpoint / Selector": selector,
        "Infer_s": _number(report.get("infer_seconds")),
        "Params_M": _number(report.get("params_m")),
        "Hardware": report.get("hardware", ""),
        "Updates": step, "Samples_seen": step * 48 if adaptive else "",
        "Fit_scene_ID": scene_id if adaptive else "", "Eval_scene_ID": scene_id,
        "RR_scope": protocol["rr_protocol"]["scope"] + "; WINDOW_MACRO_NOT_BENCHMARK_RR",
        "FR_scope": protocol["fr_protocol"]["scope"] + "; TILE_MACRO_NOT_FULL_SCENE",
        "Input_adaptation": adaptive, "Seed": registry["seed"] if adaptive else "", "Repeat": 1,
        "Run_ID": run_id, "Campaign_ID": registry["campaign_id"],
        "Parent_checkpoint_SHA": parent_sha, "Checkpoint_SHA": checkpoint or "", "Server": "s1",
        "Precision": "GDAL_FLOAT32" if method == "BROVEY8" else "FP32_AMP_OFF_TF32_OFF",
        "Test_patches": report["rr"]["samples"], "Bands": 8,
        "Norm_profile": "DN_UNSCALED" if method == "BROVEY8" else "2*DN/2047-1_ONCE",
        "Valid_mask_SHA": canonical_sha([[e["rr_mask_sha256"], e["fr_mask_sha256"]] for e in protocol["audit"]]),
        "Status": "EVAL_COMPLETE", "Readback": "",
        "Source_manifest_SHA": canonical_sha(registry["scenes"]),
        "Tile_layout_SHA": protocol["protocol_sha256"],
        "Evaluator_SHA": canonical_sha(protocol["implementation"]),
        "Local_artifact_root": str(report_path.parent), "Result_ID": canonical_sha(key),
        "Recorded_at": receipt["recorded_utc"],
        "Notes": "Private WV2 source-adapted comparison; not official RR20/FR20. "
                 "RR=valid32-window macro; FR=native-tile macro; JQM unvalidated/blank. "
                 + ("Infer_s covers complete inference calls, not metrics/H5 I/O. "
                    if report.get("infer_seconds") is not None else
                    "Infer_s blank: measured eval_seconds includes metric and I/O work. ")
                 + f"Evaluation wall time={report['eval_seconds']} s. "
                 "Log/output URLs blank: artifacts are local, not uploaded. "
                 "Source_manifest_SHA hashes the registered input/split scene list. "
                 + ("BEST search budget=20000 updates; selected checkpoint counts shown. " if selector == "BEST_RRVAL_TO_20K" else "")
                 + ("Selector alias shares outputs; not an independent repetition. " if result.get("reused_stage_id") else "")
                 + ("Brovey configuration SHA=" + canonical_sha(identity["brovey"]) if method == "BROVEY8" else "")})
    if adaptive:
        run_root = _owned(root, root / "runs" / run_id)
        config_path = run_root / "config.resolved.json"
        if config_path.is_file():
            config = load_json(config_path)
            if config.get("method") != method or config.get("seed") != registry["seed"]:
                raise ValueError("Training config identity differs")
            values.update(Train_patches=config["train_length"], Val_patches=config["val_length"],
                          Effective_epochs=step * 48 / len(config["train_positive_support_ids"]),
                          LR=config["learning_rate"])
        cost_path = run_root / "cost_breakdown.json"
        if cost_path.is_file():
            cost = load_json(cost_path)
            if cost.get("step") == 20000 and cost.get("status") == "COMPLETE":
                values["Adapt_s"] = _number(cost["adapt_seconds"])
                values["Notes"] += " Adapt_s includes the full 20K search, including BEST selectors."
    return [values[header] for header in HEADERS]


def _pad(row):
    return list(row) + [""] * (64 - len(row))


class DedicatedSheet:
    def __init__(self, spreadsheet):
        self.book = spreadsheet
        if spreadsheet.id != SPREADSHEET_ID:
            raise ValueError("Wrong spreadsheet; no writes allowed")
        metadata = spreadsheet.fetch_sheet_metadata(params={"fields": "spreadsheetId,sheets.properties"})
        match = [s["properties"] for s in metadata.get("sheets", []) if s["properties"].get("sheetId") == SHEET_ID]
        if len(match) != 1 or match[0].get("title") != SHEET:
            raise ValueError("Dedicated sheet ID/title mismatch; never create/fallback")
        if match[0]["gridProperties"]["columnCount"] < 64:
            raise ValueError("Dedicated header grid too narrow")
        self.last_row = min(1000, match[0]["gridProperties"]["rowCount"])
        if self.values("A5:BL5") != [HEADERS]:
            raise ValueError("Dedicated header changed; no writes allowed")

    def values(self, area):
        return self.book.values_get(f"'{SHEET}'!{area}", params={"valueRenderOption": "UNFORMATTED_VALUE"}).get("values", [])

    def _row_cells(self, row):
        data = self.book.fetch_sheet_metadata(params={"includeGridData": True,
            "ranges": [f"'{SHEET}'!A{row}:BL{row}"],
            "fields": "sheets(properties(sheetId,title),data(startRow,startColumn,rowData(values(userEnteredValue,effectiveValue,dataValidation,chipRuns))))"})
        matched = [s for s in data.get("sheets", []) if s["properties"].get("sheetId") == SHEET_ID]
        if len(matched) != 1 or matched[0]["properties"].get("title") != SHEET:
            raise ValueError("Row metadata targets the wrong sheet")
        rows = matched[0].get("data", [{}])[0].get("rowData", [{}])
        cells = rows[0].get("values", []) if rows else []
        return cells + [{}] * (64 - len(cells))

    def _validate_cells(self, row, values, *, must_be_blank):
        cells = self._row_cells(row)
        for i, cell in enumerate(cells):
            if must_be_blank and (cell.get("userEnteredValue") or cell.get("effectiveValue") or cell.get("chipRuns")):
                raise ValueError("Append destination is not a genuinely blank native row")
            if i == IDX["Review"] or values[i] == "":
                continue
            condition = cell.get("dataValidation", {}).get("condition")
            if condition:
                allowed = [v.get("userEnteredValue") for v in condition.get("values", [])]
                if condition.get("type") != "ONE_OF_LIST" or str(values[i]) not in allowed:
                    raise ValueError("Value violates or cannot verify native validation")

    @staticmethod
    def _matches(actual, expected):
        for i, (a, e) in enumerate(zip(_pad(actual), expected)):
            if i in (IDX["Review"], IDX["Readback"]):
                continue
            if isinstance(e, bool):
                if type(a) is not bool or a != e:
                    return False
            elif isinstance(e, (int, float)):
                if (isinstance(a, bool) or not isinstance(a, (int, float))
                        or not math.isfinite(a) or not math.isfinite(e)
                        or not math.isclose(a, e, rel_tol=READBACK_TOLERANCE["relative"],
                                            abs_tol=READBACK_TOLERANCE["absolute"])):
                    return False
            elif a != e:
                return False
        return True

    def publish(self, row):
        if len(row) != 64 or row[IDX["Status"]] != "EVAL_COMPLETE":
            raise ValueError("Only completed 64-column observations can be published")
        existing, blank = [], None
        for start in range(6, self.last_row + 1, 200):
            stop = min(self.last_row, start + 199)
            rows = self.values(f"A{start}:BL{stop}")
            for offset in range(stop - start + 1):
                value = _pad(rows[offset] if offset < len(rows) else [])
                if value[IDX["Result_ID"]] == row[IDX["Result_ID"]]:
                    existing.append((start + offset, value))
                if blank is None and not any(v != "" for v in value):
                    blank = start + offset
        if len(existing) > 1:
            raise ValueError("Duplicate Result_ID already exists; no mutation")
        if existing:
            target, previous = existing[0]
            if not self._matches(previous, row):
                raise ValueError("Existing Result_ID has different metrics/provenance")
            self._validate_cells(target, row, must_be_blank=False)
        else:
            if blank is None:
                raise ValueError("Dedicated row limit 1000 reached; do not extend table")
            target = blank
            self._validate_cells(target, row, must_be_blank=True)
            # A:BJ is 62 columns; BK is the user's Review and is never written.
            self.book.values_batch_update({"valueInputOption": "RAW", "data": [
                {"range": f"'{SHEET}'!A{target}:BJ{target}", "values": [row[:62]]},
                {"range": f"'{SHEET}'!BL{target}", "values": [[row[63]]]},
            ]})
        live = self.values(f"A{target}:BL{target}")
        if len(live) != 1 or not self._matches(live[0], row):
            raise ValueError("Written values failed exact readback; keep local outbox")
        if _pad(live[0])[IDX["Readback"]] == "VERIFIED":
            return dict(status="SHEET_READBACK_VERIFIED", row=target, result_id=row[IDX["Result_ID"]],
                        spreadsheet_id=SPREADSHEET_ID, sheet_id=SHEET_ID,
                        row_sha256=canonical_sha(row), manual_review_written=False,
                        numeric_readback_tolerance=READBACK_TOLERANCE)
        # VERIFIED becomes true only after the complete metric row matched.
        proof = list(row); proof[IDX["Readback"]] = "VERIFIED"
        self._validate_cells(target, proof, must_be_blank=False)
        self.book.values_batch_update({"valueInputOption": "RAW", "data": [
            {"range": f"'{SHEET}'!AZ{target}", "values": [["VERIFIED"]]}]})
        after = self.values(f"A{target}:BL{target}")
        if len(after) != 1 or not self._matches(after[0], row) or _pad(after[0])[IDX["Readback"]] != "VERIFIED":
            raise ValueError("Verification marker readback failed")
        return dict(status="SHEET_READBACK_VERIFIED", row=target, result_id=row[IDX["Result_ID"]],
                    spreadsheet_id=SPREADSHEET_ID, sheet_id=SHEET_ID,
                    row_sha256=canonical_sha(row), manual_review_written=False,
                    numeric_readback_tolerance=READBACK_TOLERANCE)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def publish_pending(root, registry, *, client=None):
    """Best-effort delivery. Never mutate source receipts, logs, or another tab."""
    root = Path(root).resolve()
    receipts = root / "reporting_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    outcomes, book = [], None
    repo = Path(registry["source_root"]).resolve()
    # The source checkout is mounted read-only in the campaign container.
    # This dedicated tab has one campaign publisher, with only its own root RW.
    with _lock(_owned(root, root / ".reporting.lock")):
        for outbox in sorted((root / "outbox").glob("*.json")):
            record_path = _owned(root, receipts / outbox.name)
            try:
                envelope = load_json(_owned(root, outbox))
                receipt_path = _owned(root, envelope["receipt"])
                if record_path.exists():
                    previous = load_json(record_path)
                    if (previous.get("status") in ("SHEET_READBACK_VERIFIED", "LOCAL_TRAINING_BLOCK_ONLY")
                            and previous.get("receipt_sha256") == file_sha(receipt_path)):
                        continue
                receipt = load_json(receipt_path)
                registry_path = _owned(root, root / "registry.json")
                if load_json(registry_path) != registry or receipt.get("registry_sha256") != file_sha(registry_path):
                    raise ValueError("Stage receipt registry identity mismatch")
                row = build_row(root, registry, receipt)
                if row is None:
                    outcome = dict(status="LOCAL_TRAINING_BLOCK_ONLY", receipt_sha256=file_sha(receipt_path))
                else:
                    if book is None:
                        if client is None:
                            import gspread
                            client = gspread.service_account(filename=str(repo / "gspread/account.json"))
                        book = DedicatedSheet(client.open_by_key(SPREADSHEET_ID))
                    outcome = dict(book.publish(row), receipt_sha256=file_sha(receipt_path))
                atomic_json(record_path, outcome)
            except Exception as error:
                # Error messages from authentication/HTTP may contain credential
                # material; retain the class only, not headers or raw responses.
                outcome = dict(status="LOCAL_OUTBOX_RETAINED", error_type=type(error).__name__)
                atomic_json(record_path, outcome)
                outcomes.append(outcome)
                break  # An outage/schema conflict must not hammer the API.
            outcomes.append(outcome)
    result = dict(status="LOCAL_OUTBOX_RETAINED" if any(o["status"] == "LOCAL_OUTBOX_RETAINED" for o in outcomes)
                  else "PENDING_RESULTS_PROCESSED", outcomes=outcomes, sheet=SHEET,
                  spreadsheet_id=SPREADSHEET_ID, manual_review_written=False)
    atomic_json(root / "reporting_status.json", result)
    return result
