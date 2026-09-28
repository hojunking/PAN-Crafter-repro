#!/usr/bin/env python3
"""Explicit local launch only. A git pull never starts an experiment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panda_rb.common import ROOT
from panda_rb.deployment import DEFAULT_IMAGE, start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', required=True, choices=['s1', 's3', 's5'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--binding', type=Path)
    p.add_argument('--image', default=DEFAULT_IMAGE)
    p.add_argument('--gpu', default='0')
    p.add_argument('--asset-root', action='append', default=[])
    p.add_argument('--retry-technical', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    print(json.dumps(start(args.server, args.root, args.binding, args.image, args.gpu,
                           args.asset_root, args.retry_technical, args.dry_run), indent=2))


if __name__ == '__main__': main()
