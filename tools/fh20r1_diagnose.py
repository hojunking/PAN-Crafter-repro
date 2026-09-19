#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fh20r1.diagnostics import diagnose

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--server',required=True); p.add_argument('--device',default='cuda')
    a=p.parse_args(); result=diagnose(a.server,a.device)
    print(json.dumps(dict(complete=result['complete'],branch=result['branch'],diagnostic_complete=result['diagnostic_complete'])))
