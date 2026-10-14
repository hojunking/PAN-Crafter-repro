"""Fail-closed, fixed64 view replacement; no training/runtime imports.

The Google client is deliberately lazy: importing this module and projecting
local evidence never authenticates or writes to Google. Only ``reconcile`` with
``apply=True`` writes, after native-cell checks and a durable cloud snapshot.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


class TransportError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def code_revision():
    root = Path(__file__).resolve().parent
    files = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.glob("*.py"))}
    return {"files": files, "sha256": digest(files)}


def _cell_value(cell):
    value = cell.get("userEnteredValue", {})
    if "formulaValue" in value:
        raise TransportError("Formula in publication rectangle: refusing overwrite")
    for kind in ("numberValue", "boolValue", "stringValue"):
        if kind in value:
            return value[kind]
    return ""


def _padded(row):
    if len(row) > 64:
        raise TransportError("Fixed64 row exceeds 64 columns")
    return list(row) + [""] * (64 - len(row))


def _typed(value):
    if value is None or value == "":
        # Actual row data must carry the oneof even for blanks. Omitted values
        # can leave stale prior content on some Sheets backends.
        return {"stringValue": ""}
    if type(value) is bool:
        return {"boolValue": value}
    if type(value) in (int, float) and math.isfinite(value):
        return {"numberValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    raise TransportError("Nonfinite or nonprimitive publication value")


def _intersects_data(range_, last_row, sheet_id):
    return (range_.get("sheetId", sheet_id) == sheet_id
            and range_.get("startRowIndex", 0) < last_row
            and range_.get("endRowIndex", last_row) > 5
            and range_.get("startColumnIndex", 0) < 64
            and range_.get("endColumnIndex", 64) > 0)


def _validate_snapshot(snapshot, project):
    from .selection import HEADERS, HEADER_SHA256, SPREADSHEET_ID, LANES
    dest = project["destination"]
    if dest["spreadsheet_id"] != SPREADSHEET_ID:
        raise TransportError("Destination workbook outside allowlist")
    if (dest["sheet_title"], dest["sheet_id"]) not in {(v[1], v[2]) for v in LANES.values()}:
        raise TransportError("Destination tab outside three-server ablations allowlist")
    meta = snapshot["metadata"]
    if meta.get("spreadsheetId") != SPREADSHEET_ID:
        raise TransportError("Google workbook identity mismatch")
    sheets = [s for s in meta.get("sheets", [])
              if s.get("properties", {}).get("sheetId") == dest["sheet_id"]]
    if len(sheets) != 1 or sheets[0]["properties"]["title"] != dest["sheet_title"]:
        raise TransportError("Sheet ID/title mismatch; no automatic destination repair")
    tab = sheets[0]
    grid = tab["properties"]["gridProperties"]
    if grid["columnCount"] != 64:
        raise TransportError("Fixed64 column count changed")
    header = _padded(snapshot["header"])
    if tuple(header) != tuple(HEADERS) or digest(header) != HEADER_SHA256:
        raise TransportError("Fixed64 header digest mismatch")
    last_row = 5 + max(len(snapshot["rows"]), len(project["rows"]))
    for merged in tab.get("merges", []):
        if _intersects_data(merged, last_row, dest["sheet_id"]):
            raise TransportError("Merged cells overlap publication rectangle")
    for table in tab.get("tables", []):
        if _intersects_data(table["range"], last_row, dest["sheet_id"]):
            raise TransportError("Native table overlaps publication rectangle")
    for protection in tab.get("protectedRanges", []):
        if "range" not in protection:
            raise TransportError("Unresolved named protected range; refusing write")
        if _intersects_data(protection["range"], last_row, dest["sheet_id"]):
            raise TransportError("Protected result cells; header protection is preserved")
    # Every occupied cell is checked, not merely the first exemplar. Values-only
    # rewrites must never destroy a formula, chip, validation, or rich text link.
    for row in snapshot.get("cells", []):
        if (not any(_cell_value(cell) != "" for cell in row)
                and any(cell.get("note") for cell in row)):
            raise TransportError("Unowned note-only row in publication rectangle")
        for cell in row:
            _cell_value(cell)
            if any(cell.get(k) for k in ("dataValidation", "chipRuns", "textFormatRuns")):
                raise TransportError("Native cell controls/rich text need manual resolution")
    return tab


def _row_key(row):
    from .selection import INDEX
    return tuple(row[INDEX[key]] for key in
                 ("Campaign_ID", "Dataset", "Server", "Run_ID", "Attempt"))


def validate_ownership(rows, project):
    from .selection import INDEX
    owned = {tuple(key) for key in project.get("owned_run_keys", [])}
    raw_ids = set(project.get("raw_ids", []))
    dest = project["destination"]
    for line, raw in enumerate(rows, start=6):
        row = _padded(raw)
        if not any(value != "" for value in row):
            continue
        rid = row[INDEX["Result_ID"]]
        key = _row_key(row)
        if key not in owned or row[INDEX["Bucket"]] != "ablations":
            raise TransportError("Unknown/foreign row at %s!%d; no cells changed" %
                                 (dest["sheet_title"], line))
        if isinstance(rid, str) and rid.startswith("VIEW-"):
            if row[INDEX["Selector"]] not in ("PASS_ALL_STEPS", "BEST_HQNR_FALLBACK"):
                raise TransportError("Unknown VIEW selector at row %d" % line)
        elif rid not in raw_ids:
            raise TransportError("Raw row absent from immutable local store at row %d" % line)


def _matrix_equal(actual, expected):
    """Full-precision numeric readback tolerates Sheets' JSON float roundtrip only."""
    if len(actual) != len(expected):
        return False
    for arow, erow in zip(actual, expected):
        for a, e in zip(_padded(arow), _padded(erow)):
            if type(e) in (int, float):
                if type(a) not in (int, float) or not math.isclose(a, e, rel_tol=5e-15, abs_tol=0):
                    return False
            elif type(a) is not type(e) or a != e:
                return False
    return True


def _readback_difference(snapshot, rows, notes):
    """Bounded diagnostic evidence, without dumping large BI provenance JSON."""
    from .selection import HEADERS, INDEX
    actual = snapshot["rows"]
    mismatches = []
    for row_number, (got, wanted) in enumerate(zip(actual, rows), start=6):
        for column, (a, e) in enumerate(zip(_padded(got), _padded(wanted))):
            if not _matrix_equal([[a]], [[e]]):
                mismatches.append({"row": row_number, "column": HEADERS[column],
                                   "actual_type": type(a).__name__, "expected_type": type(e).__name__,
                                   "actual_sha256": digest(a), "expected_sha256": digest(e)})
    extra = [{"row": i + 6, "result_id": r[INDEX["Result_ID"]],
              "scope": r[INDEX["Eval_scope"]], "selector": r[INDEX["Selector"]]}
             for i, r in enumerate(actual) if i >= len(rows) and any(v != "" for v in r)]
    return {"actual_extent_rows": len(actual), "expected_rows": len(rows),
            "actual_nonempty_rows": sum(any(v != "" for v in r) for r in actual),
            "cell_mismatch_count": len(mismatches), "cell_mismatches_first10": mismatches[:10],
            "extra_nonempty_count": len(extra), "extra_nonempty_first5": extra[:5],
            "notes_match": snapshot.get("notes", []) == notes}


def reconcile(adapter, project, state_dir, *, apply=False):
    """Rebuild only the one owned server tab, never append raw source events.

    Even an unchanged view is read back before declaring success: a legacy
    publisher or human may have appended since the previous process poll.
    ``adapter.replace`` is a single atomic typed updateCells batch, not a
    clear-followed-by-write operation. An uncertain response is retried by a
    fresh reconcile/read, never by blind append.
    """
    if project.get("conflicts"):
        raise TransportError("Local source conflicts block publication")
    dest = project["destination"]
    rows = [_padded(row) for row in project["rows"]]
    notes = project["notes"]
    if len(notes) != len(rows):
        raise TransportError("A-cell notes count mismatch")
    if len({row[59] for row in rows}) != len(rows):
        raise TransportError("Duplicate display Result_ID")
    snapshot = adapter.snapshot(dest)
    _validate_snapshot(snapshot, project)
    validate_ownership(snapshot["rows"], project)
    source_proof = {key: project.get(key) for key in
                    ("source_digest", "raw_count", "summary_count", "raw_ids",
                     "owned_run_keys", "runs", "selected", "excluded", "source_manifest")}
    report = {"policy": "PAN_PASSED_STEPS_OR_BEST_HQNR_V1",
              "at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "destination": dest, "destination_range": "'%s'!A6:BL%d" %
              (dest["sheet_title"], max(6, 5 + len(rows))),
              "new_row_count": len(rows), "previous_row_count": len(snapshot["rows"]),
              "source": source_proof, "view_hash": project["view_hash"],
              "projector_revision": code_revision(), "applied": False,
              "readback_verified": False}
    state_dir = Path(state_dir)
    if not apply:
        report["action"] = "DRY_RUN"
        atomic_json(state_dir / "last_dry_run.json", report)
        return report
    same = (_matrix_equal(snapshot["rows"], rows) and snapshot.get("notes", []) == notes)
    if same:
        report.update(action="UNCHANGED_VERIFIED", readback_verified=True)
        atomic_json(state_dir / "last_verified.json", report)
        return report
    # Durable complete native-grid snapshot contains the raw tail before any
    # cleanup. Immutable local originals remain the selection source forever.
    backup = {"snapshot": snapshot, "projection": project,
              "projector_revision": report["projector_revision"]}
    backup_sha = digest(backup)
    backup_path = state_dir / "backups" / (backup_sha + ".json")
    if backup_path.exists():
        if digest(json.loads(backup_path.read_text(encoding="utf-8"))) != backup_sha:
            raise TransportError("Existing cloud backup digest mismatch")
    else:
        atomic_json(backup_path, backup)
    # A bounded second read catches live old publishers/human edits after backup.
    immediately_before = adapter.snapshot(dest)
    if digest(immediately_before) != digest(snapshot):
        raise TransportError("Sheet changed during snapshot; retry after competing writer stops")
    report.update(backup_path=str(backup_path), backup_sha256=backup_sha)
    atomic_json(state_dir / "pending_write.json", report)
    occupied_tail_rows = [i for i, row in enumerate(snapshot["rows"])
                          if i >= len(rows) and any(value != "" for value in row)]
    adapter.replace(dest, rows, notes, previous_count=len(snapshot["rows"]),
                    occupied_tail_rows=occupied_tail_rows,
                    grid_row_count=snapshot["grid_row_count"])
    after = adapter.snapshot(dest)
    _validate_snapshot(after, project)
    if not _matrix_equal(after["rows"], rows) or after.get("notes", []) != notes:
        difference = _readback_difference(after, rows, notes)
        atomic_json(state_dir / "readback_mismatch.json", difference)
        raise TransportError("Readback mismatch: actual extent=%d/nonempty=%d, expected=%d, "
                             "cell mismatches=%d, extra nonempty=%d; local intent retained" %
                             (difference["actual_extent_rows"], difference["actual_nonempty_rows"],
                              difference["expected_rows"], difference["cell_mismatch_count"],
                              difference["extra_nonempty_count"]))
    before_other = [s for s in snapshot["metadata"]["sheets"]
                    if s["properties"]["sheetId"] != dest["sheet_id"]]
    after_other = [s for s in after["metadata"]["sheets"]
                   if s["properties"]["sheetId"] != dest["sheet_id"]]
    if before_other != after_other:
        raise TransportError("Other tab metadata changed concurrently; verification withheld")
    report.update(action="REPLACED_VERIFIED", applied=True, readback_verified=True)
    atomic_json(state_dir / "last_verified.json", report)
    return report


class GoogleSheets:
    """Minimal Sheets REST client using the existing server credential file.

    Read retries are bounded; mutation has no automatic retry. No Drive create,
    sharing, append, sorting, deleteDimension, rowCount shrink, or other tabs.
    """
    BASE = "https://sheets.googleapis.com/v4/spreadsheets/"

    def __init__(self, credential_path, *, timeout=45, page_rows=400):
        from google.oauth2 import service_account
        from google.auth.transport.requests import AuthorizedSession
        creds = service_account.Credentials.from_service_account_file(
            str(credential_path), scopes=["https://www.googleapis.com/auth/spreadsheets"])
        self.session = AuthorizedSession(creds)
        self.timeout = timeout
        self.page_rows = min(700, max(1, page_rows))

    def _request(self, method, url, **kwargs):
        attempts = 3 if method == "GET" else 1
        for attempt in range(attempts):
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            if response.status_code < 400:
                return response.json()
            if response.status_code not in (429, 500, 502, 503, 504) or attempt + 1 == attempts:
                raise TransportError("Google HTTP %d: %s" %
                                     (response.status_code, response.text[:600]))
            time.sleep(min(8, 2 ** attempt) + random.random())
        raise TransportError("Unreachable request state")

    def snapshot(self, destination):
        from .selection import SPREADSHEET_ID
        if destination["spreadsheet_id"] != SPREADSHEET_ID:
            raise TransportError("Workbook outside allowlist")
        url = self.BASE + SPREADSHEET_ID
        metadata = self._request("GET", url, params={"fields":
            "spreadsheetId,properties(title),sheets(properties,merges,protectedRanges,basicFilter,tables)"})
        found = [s for s in metadata["sheets"]
                 if s["properties"]["sheetId"] == destination["sheet_id"]]
        if len(found) != 1 or found[0]["properties"]["title"] != destination["sheet_title"]:
            raise TransportError("Live tab does not match pinned ID/title")
        grid_rows = found[0]["properties"]["gridProperties"]["rowCount"]
        title = destination["sheet_title"].replace("'", "''")
        cells, header = [[] for _ in range(max(0, grid_rows - 4))], None
        ranges = [(start, min(grid_rows, start + self.page_rows - 1))
                  for start in range(5, grid_rows + 1, self.page_rows)]
        # Multiple individually bounded rectangles share one metadata request.
        # Empty ranges are omitted by the API, not treated as a reason to stop
        # scanning: an unrelated row after a gap must still block replacement.
        for offset in range(0, len(ranges), 40):
            batch = ranges[offset:offset + 40]
            obj = self._request("GET", url, params={
                "ranges": ["'%s'!A%d:BL%d" % (title, start, end) for start, end in batch],
                "includeGridData": "true", "fields":
                "sheets(data(startRow,startColumn,rowData(values(userEnteredValue,effectiveValue,note,dataValidation,chipRuns,textFormatRuns))))"})
            for sheet in obj.get("sheets", []):
                for block in sheet.get("data", []):
                    row_offset = block.get("startRow", 4) - 4
                    for number, row in enumerate(block.get("rowData", [])):
                        if 0 <= row_offset + number < len(cells):
                            cells[row_offset + number] = row.get("values", [])
        if cells:
            header = [_cell_value(c) for c in cells.pop(0)]
        # Keep controls/notes after the last literal so they cannot be erased by
        # future view growth. Formatting alone is preserved and is not content.
        last = 0
        for number, row in enumerate(cells):
            if any(_cell_value(c) != "" or any(c.get(k) for k in
                       ("note", "dataValidation", "chipRuns", "textFormatRuns")) for c in row):
                last = number + 1
        cells = cells[:last]
        rows = [_padded([_cell_value(c) for c in row]) for row in cells]
        notes = [(row[0].get("note", "") if row else "") for row in cells]
        return {"metadata": metadata, "header": _padded(header or []),
                "grid_row_count": grid_rows, "rows": rows, "notes": notes, "cells": cells}

    def replace(self, destination, rows, notes, *, previous_count, grid_row_count,
                occupied_tail_rows=None):
        from .selection import SPREADSHEET_ID
        if destination["spreadsheet_id"] != SPREADSHEET_ID:
            raise TransportError("Workbook outside allowlist")
        sheet_id = destination["sheet_id"]
        height = max(previous_count, len(rows))
        if not height:
            return
        requests = []
        if len(rows) + 5 > grid_row_count:
            requests.append({"appendDimension": {"sheetId": sheet_id, "dimension": "ROWS",
                                                 "length": len(rows) + 5 - grid_row_count}})
        cell_rows = []
        for index in range(len(rows)):
            values = []
            for column, value in enumerate(rows[index]):
                cell = {"userEnteredValue": _typed(value)}
                # Clear stale row notes as well as stale row values; originals
                # were preserved in the complete cloud snapshot above.
                cell["note"] = notes[index] if index < len(notes) and column == 0 else ""
                values.append(cell)
            cell_rows.append({"values": values})
        if rows:
            requests.append({"updateCells": {"range": {
                "sheetId": sheet_id, "startRowIndex": 5, "endRowIndex": 5 + len(rows),
                "startColumnIndex": 0, "endColumnIndex": 64}, "rows": cell_rows,
                "fields": "userEnteredValue,note"}})
        if occupied_tail_rows is None:
            occupied_tail_rows = list(range(len(rows), previous_count))
        occupied_tail_rows = sorted(set(occupied_tail_rows))
        if any(type(i) is not int or i < len(rows) or i >= previous_count for i in occupied_tail_rows):
            raise TransportError("Tail-clear row outside backed-up observed scope")
        groups = []
        for index in occupied_tail_rows:
            if groups and groups[-1][1] == index:
                groups[-1][1] += 1
            else:
                groups.append([index, index + 1])
        for start, end in groups:
            # The backend ignored omitted-range clears AND repeatCell, including
            # explicit strings. Real rowData updateCells is the proven write
            # operation. Clear only occupied, backed-up owned tail rows; blank
            # gaps need no operation and cannot inflate the payload to megabytes.
            requests.append({"updateCells": {"range": {
                "sheetId": sheet_id, "startRowIndex": 5 + start,
                "endRowIndex": 5 + end, "startColumnIndex": 0, "endColumnIndex": 64},
                "rows": [{"values": [{"userEnteredValue": {"stringValue": ""}, "note": ""}
                                       for _ in range(64)]} for _ in range(end - start)],
                "fields": "userEnteredValue,note"}})
        # Typed stringValue is RAW-equivalent even when the string starts '='.
        self._request("POST", self.BASE + SPREADSHEET_ID + ":batchUpdate",
                      json={"requests": requests})
