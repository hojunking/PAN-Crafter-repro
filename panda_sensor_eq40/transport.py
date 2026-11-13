"""Own-row updates and atomic sheet-bottom allocation; never guess a table end."""
from copy import deepcopy
import os
from pathlib import Path
import time

from .publication import INDEX, R, row_requests, destination
from sheet_publish_v2.transport import (TransportError, _validate_snapshot as _validate_fixed64,
                                       _matrix_equal)

TRANSPORT = "WV3_HYPER_OWN_ROWS_ATOMIC_APPEND_V2"


def _verify_atomic_append(before, fresh, additions):
    """Locate allocated IDs using readback, without interpreting a logical table."""
    wanted = {}
    for payload in additions:
        row = [next(iter(c['userEnteredValue'].values())) for c in payload['values']]
        wanted[row[INDEX['Result_ID']]] = (row, payload['values'][0]['note'])
    positions = {}
    for i, row in enumerate(fresh['rows']):
        rid = row[INDEX['Result_ID']]
        if rid not in wanted:
            continue
        expected, note = wanted[rid]
        if (rid in positions or i < len(before['rows'])
                or not _matrix_equal([row], [expected]) or fresh['notes'][i] != note):
            raise TransportError('Atomic append readback overlaps data, duplicates IDs, or differs')
        positions[rid] = i + 6
    if set(positions) != set(wanted):
        raise TransportError('Atomic append missing IDs; inspect intent before retry')
    return sorted(positions.values())


def _validate_snapshot(snapshot, value):
    """Validate the owned A:BL rectangle, not the width of the entire tab.

    The shared legacy validator assumes exclusive ownership of a 64-column
    sheet. This row-scoped publisher owns only A:BL, so additional columns to
    the right are legal. Keep the original snapshot/backup intact and validate
    a bounded metadata view with ALL other legacy safety checks unchanged.
    Never resize the cloud sheet or silently truncate shifted headers/data.
    """
    dest = value["destination"]
    if dest not in (destination('s2'),destination('s3')):
        raise TransportError("Destination differs from explicit WV3 hyperparameter tab")
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
    appended_rows = []
    if additions:
        # values.append searches for a *logical table*. Even A-only lookup can
        # select an earlier subtable and OVERWRITE later occupied rows. Instead,
        # appendCells allocates after all sheet data atomically, with the notes.
        # Historical entered empty strings may move the allocation farther down;
        # preserving existing rows takes priority over compacting someone else's
        # data. Never retry a POST blindly; a later poll first reads IDs afresh.
        response = adapter._request("POST", url + ":batchUpdate", json={"requests": [
            {'appendCells': {'sheetId': dest['sheet_id'], 'rows': additions,
                             'fields': 'userEnteredValue,note'}}]})
        R.atomic_json(state / ("append_response_" + stamp + ".json"), response)
        after_append = adapter.snapshot(dest)
        appended_rows = _verify_atomic_append(snapshot, after_append, additions)
        requests, remaining = prepare(after_append, value, server)
        if remaining:
            raise ValueError("Appended IDs missing; inspect intent before retry")
    if any("appendCells" in req for req in requests):
        raise AssertionError("Unexpected second allocation after append readback")
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
            "appended_rows": appended_rows, "foreign_rows_untouched": True}
