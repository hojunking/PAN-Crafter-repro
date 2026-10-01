#!/usr/bin/env python3
"""Pull-to-run local M12 entry point; only --activate can launch CUDA work."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panda_rb_m12.common import ROOT
from panda_rb_m12.deployment import start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=['s1', 's3', 's5'], required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--asset-root', type=Path, action='append', default=[])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--retry-technical', action='store_true')
    args = parser.parse_args()
    result = start(args.server, args.root, args.binding, args.gpu, args.asset_root,
                   args.retry_technical, args.dry_run, args.activate)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
