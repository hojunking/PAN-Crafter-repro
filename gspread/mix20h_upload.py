#!/usr/bin/env python
"""Opt-in M20 local-result uploader. No GPU, no old postrun/backlog, no row clear.

All columns are resolved by actual labels. Existing legacy/NOA cells are left
untouched; a missing run row is populated through the existing collector/formatter.
Local flock is shared with tools/_upload.sh and other campaign uploaders.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.mix20h_postrun import read, write, TARGET_FILE, STATUS_FILE, utcnow

SCHEMA = "M20_identity_target_exact50k_v1"
GIDS = dict(s1=994031662, s2=991648123, s3=284220763, s4=2026091404, s5=823586191)


def local_module(name):
    spec = importlib.util.spec_from_file_location("_mix20_" + name, ROOT / "gspread" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row_values(run):
    wd = ROOT / "work_dir" / run
    manifest = read(wd / "meta/mix20h_run_manifest.json")
    status = read(wd / "results" / STATUS_FILE)
    target = read(wd / "results" / TARGET_FILE)
    exact = read(wd / "results/exact50k_official_AON.json")
    from tools.mix20h_postrun import target_complete, exact_complete
    identity = status.get("completion_identity")
    if (not identity or not status.get("training_complete") or not status.get("raw_grid_complete")
            or not target_complete(target, identity) or not exact_complete(exact, identity)):
        raise ValueError("upload requires validated grid and complete official v2/exact50K")
    chosen = target.get("target") or {}
    sha = (chosen.get("rr_identity") or {}).get("checkpoint_sha256", "")
    values = {
        "M20 campaign": manifest.get("campaign_id", ""), "M20 queue revision": manifest.get("queue_revision", ""),
        "M20 server": manifest.get("server_id", manifest.get("server", "")), "M20 profile": manifest.get("profile", ""),
        "M20 seed": manifest.get("seed", ""), "M20 pair id": manifest.get("pair_id", ""), "M20 run id": run,
        "M20 U init SHA256": manifest.get("init_unet_sha256", ""), "M20 A init SHA256": manifest.get("init_aligner_sha256", ""),
        "M20 release": manifest.get("release_sha", ""), "M20 schema": SCHEMA,
        "Target selector": target["selector"], "Target step": chosen.get("step", ""), "Target checkpoint SHA256": sha,
        "Target HQNR(raw)↑": chosen.get("hqnr", ""), "Target ERGAS↓": chosen.get("ergas", ""),
        "Target SCC↑": chosen.get("scc", ""), "Target PSNR↑": chosen.get("psnr", ""),
        "Target SAM↓": chosen.get("sam", ""), "Target Q8↑": chosen.get("q8", ""), "Target SSIM↑": chosen.get("ssim", ""),
        "Target n eligible": target["n_eligible"], "Target status": target["target_status"],
        "Target joint pass": target.get("joint_pass"), "Target official": True,
        "Exact50K step": 50000, "Exact50K checkpoint SHA256": exact["checkpoint_sha256"],
        "Exact50K HQNR(raw)↑": exact["hqnr"], "Exact50K ERGAS↓": exact["ergas"],
        "Exact50K SCC↑": exact["scc"], "Exact50K PSNR↑": exact["psnr"], "Exact50K official": True,
        "M20 actual updates": 50000, "M20 started UTC": manifest.get("started_at_utc", ""),
        "M20 deadline UTC": manifest.get("deadline_at_utc", ""),
        "M20 eval seconds": status.get("evaluation_seconds", ""), "M20 state": "finished",
    }
    if not sha and chosen:
        raise ValueError("target checkpoint SHA missing")
    return {label: "" if value is None else value for label, value in values.items()}


def label_map(headers):
    mapping = {}
    for index, label in enumerate(headers, 1):
        label = str(label).strip()
        if not label:
            continue
        if label in mapping:
            raise ValueError(f"duplicate sheet header: {label}")
        mapping[label] = index
    return mapping


def same_cell(actual, expected):
    if expected == "":
        return actual in ("", None)
    if isinstance(expected, bool):
        return str(actual).strip().lower() == str(expected).lower()
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual)-expected) <= 1e-12 * max(1., abs(expected))
        except (ValueError, TypeError):
            return False
    return str(actual) == str(expected)


def upload_run(run, dry=False):
    values = row_values(run)
    if dry:
        print(json.dumps(values, ensure_ascii=False, indent=2))
        return dict(dry_run=True, schema=SCHEMA)
    gu = local_module("gspread_upload")
    categories = local_module("sheet_categories")
    tab_server = gu.resolve_server(None)
    from tools.gen_pakd50_configs import server_id
    server = server_id(tab_server)
    if server != values["M20 server"] or f"_{server.upper()}_" not in run:
        raise ValueError("local server/run/manifest mismatch; refusing another server's tab")
    import gspread
    lockpath = ROOT / "work_dir/.gspread_write.lock"
    with open(lockpath, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sheet = gspread.service_account(filename=gu.CRED).open(gu.SHEET)
        ws = sheet.worksheet(gu.sheet_name("WV3", tab_server))
        if int(ws.id) != GIDS[server]:
            raise ValueError("worksheet gid mismatch")
        headers = ws.row_values(gu.ORIGIN_ROW + 1)
        labels = label_map(headers)
        if "Run" not in labels:
            raise ValueError("existing Run header missing")
        # Each absent label gets its own new column; old target columns may be noncontiguous.
        missing = [label for label in values if label not in labels]
        first = max(len(headers), len(ws.row_values(gu.ORIGIN_ROW))) + 1
        required = first + len(missing) - 1
        if required > ws.col_count:
            ws.add_cols(required-ws.col_count)
        edits = []
        for col, label in enumerate(missing, first):
            labels[label] = col
            edits += [{"range": gu._a1(gu.ORIGIN_ROW, col), "values": [["M20"]]},
                      {"range": gu._a1(gu.ORIGIN_ROW+1, col), "values": [[label]]}]
        run_col = gu._col(labels["Run"])
        tags = ws.get(f"{run_col}{gu.ORIGIN_ROW+2}:{run_col}")
        rows = [gu.ORIGIN_ROW+2+i for i, row in enumerate(tags)
                if row and categories.canonical_run_key(row[0], server) == run]
        if not rows:
            # Existing uploader's collector and formatter, no --profile or old GPU postrun.
            record = gu.collect(run, False, server)
            if record is None:
                raise ValueError("legacy collector could not load this run")
            rownum = gu.ORIGIN_ROW + 2 + len(tags)
            if rownum > ws.row_count:
                ws.add_rows(rownum-ws.row_count)
            columns = gu.columns_for("WV3")
            formatted = gu.fmt(record, columns)
            for (_group, label, _key, _precision), value in zip(columns, formatted):
                if label not in labels:
                    raise ValueError(f"legacy sheet header missing: {label}")
                edits.append({"range": gu._a1(rownum, labels[label]), "values": [[value]]})
            rows = [rownum]
        for rownum in rows:
            for label, value in values.items():
                edits.append({"range": gu._a1(rownum, labels[label]), "values": [[value]]})
        ws.batch_update(edits, value_input_option="RAW")
        # Confirm all written M20 values, not just a successful HTTP response.
        for rownum in rows:
            actual = ws.row_values(rownum, value_render_option="UNFORMATTED_VALUE")
            for label, expected in values.items():
                col = labels[label]
                got = actual[col-1] if col <= len(actual) else ""
                if not same_cell(got, expected):
                    raise ValueError(f"read-back mismatch at row {rownum}, label {label}")
            run_cell = actual[labels["Run"]-1]
            if categories.canonical_run_key(run_cell, server) != run:
                raise ValueError("read-back canonical run ID mismatch")
        receipt = dict(run_id=run, server=server, gid=ws.id, rows=rows, schema=SCHEMA,
                       target_checkpoint_sha256=values["Target checkpoint SHA256"],
                       exact50k_checkpoint_sha256=values["Exact50K checkpoint SHA256"], readback_verified=True,
                       uploaded_at_utc=utcnow())
        write(ROOT / "work_dir" / run / "results/mix20h_upload_receipt.json", receipt)
        # Direct invocation must have the same completion semantics as postrun --upload.
        # Only advance the local state after all remote values and the canonical ID
        # have been read back successfully; preserve the independent evaluation flags.
        status_path = ROOT / "work_dir" / run / "results" / STATUS_FILE
        status = read(status_path)
        status.update(status="COMPLETE", sheet_uploaded=True, upload_receipt=receipt,
                      updated_at_utc=receipt["uploaded_at_utc"])
        status.pop("upload_error", None)
        write(status_path, status)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--dry-run", action="store_true", help="local JSON only; no network connection")
    args = parser.parse_args()
    print(json.dumps(upload_run(args.run, args.dry_run), ensure_ascii=False))
