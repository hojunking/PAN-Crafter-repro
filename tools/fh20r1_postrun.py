#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fh20r1.postrun import process

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--device',default='cuda')
    p.add_argument('--upload',action='store_true'); p.add_argument('--upload-only',action='store_true')
    a=p.parse_args()
    raise SystemExit(process(a.run,device=a.device,upload=a.upload,upload_only=a.upload_only))
