"""Best-only policy transport; preserve legacy row-scoped append/readback safety."""
import os
from pathlib import Path
import time
from urllib.parse import quote

from . import INDEX, R, row_requests
from sheet_publish_v2.transport import _validate_snapshot

TRANSPORT = "PANDA_MATURE30_BEST_HQNR_LOGICAL_TABLE_V2"


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
        table = "'%s'!A5:BL" % dest["sheet_title"].replace("'", "''")
        response = adapter._request("POST", url + "/values/" + quote(table, safe="") + ":append",
            params={"valueInputOption": "RAW", "insertDataOption": "OVERWRITE"},
            json={"majorDimension": "ROWS", "values": rows})
        R.atomic_json(state / ("append_response_" + stamp + ".json"), response)
        append_range = response.get("updates", {}).get("updatedRange")
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
            "at": R.utc_now(), "readback_verified": True, "row_count": len(ids),
            "sheet_rows": [i + 6 for i, r in enumerate(fresh["rows"])
                           if r[INDEX["Result_ID"]] in ids],
            "append_range": append_range, "foreign_rows_untouched": True}
