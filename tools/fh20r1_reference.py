#!/usr/bin/env python3
"""Explicit FH20R1 local reference import / donor check / N2PL calibration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fh12.common import atomic_json
from fh20r1.references import (N2_RUN, bridge_path, calibrate_n2pl, import_reference, load_donor_aligner, load_reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("import", "donor", "calibrate"))
    parser.add_argument("--server", required=True, choices=("s1", "s2", "s3", "s4", "s5"))
    parser.add_argument("--alias")
    parser.add_argument("--teacher-run", default=N2_RUN)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "donor":
        _, report = load_donor_aligner(ROOT, args.server)
        path = ROOT / "work_dir/_fh20r1" / args.server / "donor_manifest.json"
        atomic_json(path, report)
        print(json.dumps(dict(report, manifest_path=str(path))))
        return 0
    alias = (args.alias or f"F{args.server[1:]}") if args.action == "import" else "N2PL"
    path = bridge_path(args.server, alias, ROOT)
    reused = path.exists()
    if args.action == "import":
        path = import_reference(args.server, alias, ROOT, args.device)
    else:
        path = calibrate_n2pl(args.teacher_run, ROOT, args.server, args.device)
        # Publish required N2PL q/native-c diagnostics from the completed cache.
        # Also runs on cached calibration reuse, so a report-write failure never
        # requires recalibrating or retraining the Teacher.
        from fh20r1.diagnostics import q_report
        _, _, verified, _ = load_reference("N2PL", args.server, ROOT, device="cpu")
        q_report(verified, args.server, ROOT)
    print(json.dumps(dict(status="PASS", complete=True, reused=reused, alias=alias,
                          server=args.server, bridge_manifest=str(path))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
