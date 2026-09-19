#!/usr/bin/env python3
"""Explicit FH20R1 trainer; minimum20h belongs to the campaign runner."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fh20r1.training import train_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(train_run(args.config, args.device, args.resume))
