#!/usr/bin/env python
"""Isolated FH12 calibration process: CUDA context exits before Student admission."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-run',required=True)
    parser.add_argument('--server',required=True,choices=['s1','s2','s3','s4','s5'])
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--deadline-utc',required=True)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    # Intentionally import Torch and allocate GPU models in this child only.
    from fh12.calibration import calibrate
    try:
        result=calibrate(args.teacher_run,args.root,args.server,args.device,args.deadline_utc)
        print(result); return 0
    except TimeoutError as error:
        print(str(error),file=sys.stderr); return 75
    except Exception as error:
        print(f'FH12 calibration failed: {type(error).__name__}: {error}',file=sys.stderr); return 2


if __name__=='__main__': raise SystemExit(main())
