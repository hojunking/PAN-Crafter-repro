#!/usr/bin/env python
"""FH12 local official selections and optional upload; no training side effects."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fh12.postrun import process

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run'); p.add_argument('--device',default='cuda')
    p.add_argument('--deadline-utc'); p.add_argument('--upload',action='store_true')
    p.add_argument('--upload-only',action='store_true')
    a=p.parse_args()
    try:
        rc=process(a.run,a.device,a.deadline_utc,a.upload,a.upload_only)
    except TimeoutError as exc:
        print(str(exc),file=sys.stderr); rc=75
    except Exception as exc:
        print(f'FH12 evaluation failed: {type(exc).__name__}: {exc}',file=sys.stderr); rc=2
    raise SystemExit(rc)
