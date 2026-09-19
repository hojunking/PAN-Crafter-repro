#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fh20r1.preflight import run_preflight

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--server',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--check-only',action='store_true'); a=p.parse_args()
    r=run_preflight(a.server,a.device,check_only=a.check_only)
    print(json.dumps(dict(status=r['status'],complete=r['complete'],preflight_pass=r['preflight_pass'])))
