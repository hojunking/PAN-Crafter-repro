#!/usr/bin/env python
"""FH12 standalone trainer. Normally invoked by tools/fh12_start.sh."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fh12.training import train_run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--deadline-utc')
    args=parser.parse_args()
    raise SystemExit(train_run(args.config,args.device,args.resume,args.deadline_utc))
