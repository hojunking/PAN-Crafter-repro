#!/usr/bin/env python3
"""Explicit one-run RB01 entry point. Does not activate another campaign."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panda_rb.common import ROOT
from panda_rb.training import train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    return train_run(args.run_id, args.binding, args.weights, args.device, args.resume, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
