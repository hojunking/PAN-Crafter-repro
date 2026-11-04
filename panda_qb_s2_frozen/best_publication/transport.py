"""Passes-and-representative transport with row-scoped append/readback safety."""
from copy import deepcopy
import os
from pathlib import Path
import re
import time
from urllib.parse import quote

from . import INDEX, R, row_requests
from sheet_publish_v2.transport import TransportError, _validate_snapshot as _validate_fixed64

TRANSPORT = "PANDA_S1LOSS9_BEST_HQNR_LOGICAL_TABLE_V2"


def _verify_append_response(response, destination, count, previous_rows):
    """Never silently accept shifted/partial append responses or blind-retry them."""
    title = destination["sheet_title"].replace("'", "''")
    updated = response.get("updates", {})
    match = re.fullmatch(r"(?:'" + re.escape(title) + r"'|"
                         + re.escape(destination["sheet_title"]) + r")!A(\d+):BL(\d+)",
                         updated.get("updatedRange", ""))
    if (not match or int(match[1]) < 6 + previous_rows
            or int(match[2]) - int(match[1]) + 1 != count
            or updated.get("updatedColumns") != 64
            or updated.get("updatedRows") != count
            or updated.get("updatedCells") != 64 * count):
        raise TransportError("Unexpected append bounds; inspect saved intent/response before retry")
    return updated["updatedRange"]


def _validate_snapshot(snapshot, value):
    """Validate the owned A:BL rectangle, not the width of the entire tab.

    The shared legacy validator assumes exclusive ownership of a 64-column
    sheet. This row-scoped publisher owns only A:BL, so additional columns to
    the right are legal. Keep the original snapshot/backup intact and validate
    a bounded metadata view with ALL other legacy safety checks unchanged.
    Never resize the cloud sheet or silently truncate shifted headers/data.
    """
    dest = value["destination"]
    tabs = [s for s in snapshot["metadata"].get("sheets", [])
            if s.get("properties", {}).get("sheetId") == dest["sheet_id"]]
    if len(tabs) != 1:
        raise TransportError("Missing/ambiguous publication tab")
    width = tabs[0]["properties"]["gridProperties"]["columnCount"]
    if type(width) is not int or width < 64:
        raise TransportError("Publication rectangle A:BL is missing or truncated")
    if (len(snapshot["header"]) > 64
            or any(len(r) > 64 for r in snapshot["rows"])
            or any(len(r) > 64 for r in snapshot.get("cells", []))):
        raise TransportError("Snapshot exceeds owned A:BL rectangle")
    metadata = deepcopy(snapshot["metadata"])
    for tab in metadata["sheets"]:
        if tab["properties"]["sheetId"] == dest["sheet_id"]:
            tab["properties"]["gridProperties"]["columnCount"] = 64
    _validate_fixed64(dict(snapshot, metadata=metadata), value)
    return tabs[0]


def prepare(snapshot, value, server):
    if value["conflicts"]:
        raise ValueError("Local evidence/cost conflicts; cloud untouched")
    if len(value["rows"]) != len(value["notes"]) or any(len(r) != 64 for r in value["rows"]):
        raise ValueError("Fixed64 rows/notes mismatch")
    _validate_snapshot(snapshot, value)
    requests = row_requests(snapshot, value, server)
    additions = [r for req in requests for r in req.get("appendCells", {}).get("rows", [])]
    if additions:
        if any(any(v != "" for v in row) and row[0] == "" for row in snapshot["rows"]):
            raise ValueError("Unanchored row: unsafe logical-table append")
        _validate_snapshot(snapshot, dict(
            value, rows=[None] * (len(snapshot["rows"]) + len(additions))))
    return requests, additions


def reconcile_rows(adapter, value, state, server, *, apply=False):
    snapshot = adapter.snapshot(value["destination"])
    requests, additions = prepare(snapshot, value, server)
    if not apply:
        return {"action": "DRY_RUN", "new_rows": len(additions),
                "request_count": len(requests), "readback_verified": False}
    state, stamp = Path(state), str(time.time_ns())
    if requests:
        R.atomic_json(state / ("cloud_backup_" + stamp + ".json"), snapshot)
        R.atomic_json(state / ("append_intent_" + stamp + ".json"),
                      {"transport": TRANSPORT, "destination": value["destination"],
                       "rows": value["rows"], "notes": value["notes"]})
    dest = value["destination"]
    url = adapter.BASE + dest["spreadsheet_id"]
    append_range = None
    if additions:
        rows = [[next(iter(c["userEnteredValue"].values())) for c in r["values"]]
                for r in additions]
        # A wide search can choose an unrelated tail subtable (observed T:V)
        # and shift a complete 64-column record to T:CE. Every occupied row
        # is preflighted to have an A anchor; search ONLY that anchor column.
        # Server-side append still allocates rows atomically across servers.
        table = "'%s'!A5:A%d" % (dest["sheet_title"].replace("'", "''"),
                                   snapshot["grid_row_count"])
        response = adapter._request("POST", url + "/values/" + quote(table, safe="") + ":append",
            params={"valueInputOption": "RAW", "insertDataOption": "OVERWRITE"},
            json={"majorDimension": "ROWS", "values": rows})
        R.atomic_json(state / ("append_response_" + stamp + ".json"), response)
        append_range = _verify_append_response(response, dest, len(rows), len(snapshot["rows"]))
        requests, remaining = prepare(adapter.snapshot(dest), value, server)
        if remaining:
            raise ValueError("Appended IDs missing; inspect intent before retry")
    if any("appendCells" in req for req in requests):
        raise AssertionError("appendCells must never reach Google")
    if requests:
        adapter._request("POST", url + ":batchUpdate", json={"requests": requests})
    fresh = adapter.snapshot(dest)
    pending, _ = prepare(fresh, value, server)
    if pending:
        raise ValueError("Own-row readback differs; retry using a fresh snapshot")
    ids = {r[INDEX["Result_ID"]] for r in value["rows"]}
    return {"action": "OWN_ROWS_VERIFIED", "transport": TRANSPORT,
            "publication_policy": value["policy"], "publisher_pid": os.getpid(),
            "owned_columns": "A:BL", "extra_columns_preserved": True,
            "at": R.utc_now(), "readback_verified": True, "row_count": len(ids),
            "sheet_rows": [i + 6 for i, r in enumerate(fresh["rows"])
                           if r[INDEX["Result_ID"]] in ids],
            "append_range": append_range, "foreign_rows_untouched": True}
