#!/usr/bin/env python3
"""Generate the finite registry/configs; never activate a runtime campaign."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fh20r1.plan import CASES,SERVERS,blocks_for,build_config,cases_for,registry_document,registry_rows


def artifacts(server=None):
    result={f'config/{c.run_id}.yaml':'# FH20R1 generated definition, not an active queue.\n'+yaml.safe_dump(build_config(c),sort_keys=False)
            for c in (CASES if server is None else cases_for(server))}
    for srv in (SERVERS if server is None else (server,)):
        result[f'config/queues/fh20r1_{srv}.json']=json.dumps(dict(server_id=srv,blocks=[b.to_dict() for b in blocks_for(srv)]),indent=2)+'\n'
    result['config/queues/FH20R1_registry.json']=json.dumps(registry_document(),indent=2)+'\n'
    rows=registry_rows(); fields=list(rows[0]); out=io.StringIO()
    writer=csv.DictWriter(out,fieldnames=fields,lineterminator='\n'); writer.writeheader()
    writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    result['config/queues/FH20R1_case_registry.csv']=out.getvalue()
    return result


def generate(root=ROOT,server=None,check=False):
    changed=[]
    for relative,content in artifacts(server).items():
        path=Path(root)/relative
        if path.is_file() and path.read_text()==content: continue
        changed.append(relative)
        if not check:
            if path.exists(): raise ValueError(f'Refusing to overwrite changed registered config: {path}')
            path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content)
    return changed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--server',choices=SERVERS)
    p.add_argument('--check',action='store_true'); a=p.parse_args()
    changed=generate(server=a.server,check=a.check)
    print(json.dumps(dict(check=a.check,missing_or_changed=changed),indent=2))
    raise SystemExit(int(a.check and bool(changed)))
