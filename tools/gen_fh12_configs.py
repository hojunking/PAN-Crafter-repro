#!/usr/bin/env python
"""Generate FH12 portable configs/registry, without touching runtime queues or clocks."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fh12.plan import CASES, SERVERS, REGISTRY_REVISION, build_config, cases_for, registry_rows


def artifacts(server=None):
    selected = CASES if server is None else cases_for(server)
    result = {f"config/{c.run_id}.yaml": "# FH12: generated portable config; no implicit activation.\n" +
              yaml.safe_dump(build_config(c), sort_keys=False) for c in selected}
    for srv in SERVERS if server is None else (server,):
        result[f"config/queues/fh12_{srv}.txt"] = (
            f"# {REGISTRY_REVISION}; use tools/fh12_start.sh, NOT legacy _run_cases.sh\n" +
            '\n'.join(c.run_id for c in cases_for(srv)) + '\n')
    rows = registry_rows()
    fields = sorted(set().union(*(r.keys() for r in rows)))
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    result["config/queues/FH12_case_registry.csv"] = stream.getvalue()
    return result


def generate(root=ROOT, server=None, check=False):
    """Runtime generator: only exact planned config contents may be installed."""
    changed = []
    for name, content in artifacts(server).items():
        path = Path(root)/name
        if path.exists() and path.read_text() == content:
            continue
        changed.append(name)
        if not check:
            if path.exists():
                raise ValueError(f"FH12 config differs from frozen registry; refusing overwrite: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", choices=SERVERS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = generate(server=args.server, check=args.check)
    print(json.dumps({"check":args.check,"missing_or_changed":changed}, indent=2))
    sys.exit(bool(changed) if args.check else 0)
