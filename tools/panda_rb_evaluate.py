#!/usr/bin/env python3
"""Native / paired shift / fixed-F1 probe in the new B01 namespace only."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panda_rb.common import ROOT, docker_required
from panda_rb.plan import binding_path
from panda_rb.evaluation import evaluate_native, evaluate_stress
from panda_rb.stress import MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id', nargs='?')
    parser.add_argument('--binding', type=Path, default=binding_path(ROOT))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=('native', 'stress', 'reference-probe', 'all'), default='all')
    parser.add_argument('--mode', choices=MODES)
    args = parser.parse_args()
    docker_required()
    if args.stage != 'reference-probe' and not args.run_id:
        parser.error('run_id is required except for reference-probe')
    if args.stage == 'stress' and args.mode is None:
        parser.error('--mode is required for a scheduled stress stage')
    if args.stage in ('native', 'all'):
        evaluate_native(args.run_id, args.binding, root=ROOT, device=args.device)
    if args.stage in ('stress', 'all'):
        for mode in ((args.mode,) if args.mode else MODES):
            evaluate_stress(args.run_id, args.binding, mode, root=ROOT, device=args.device)
    if args.stage == 'reference-probe':
        from panda_rb.reference_probe import reference_probe
        reference_probe(args.binding, root=ROOT, device=args.device)


if __name__ == '__main__':
    main()
